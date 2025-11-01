use crate::buffer::*;
use crate::instructions::*;
use crate::processor::*;
use crate::rdna4_decoder::*;
use crate::rdna_instructions::*;
use crate::rdna_translator::*;

use std::cell::RefCell;
use std::collections::VecDeque;

static USE_INTERPRETER: bool = false;
static USE_ENTIRE_KERNEL_TRANSLATION: bool = true;

pub trait RegisterFile<T: Copy> {
    fn new(num_elems: usize, count: usize, default: T) -> Self;
    fn get(&self, elem: usize, idx: usize) -> T;

    fn set(&mut self, elem: usize, idx: usize, val: T);

    fn get_vec(&self, idx: usize) -> &[T];

    fn set_vec(&mut self, idx: usize, vals: &[Option<T>]);
}

pub struct RegisterFileImpl<T: Copy> {
    num_elems: usize,
    pub regs: aligned_vec::AVec<T>,
}

impl<T: Copy> RegisterFile<T> for RegisterFileImpl<T> {
    fn new(num_elems: usize, count: usize, default: T) -> Self {
        let mut regs = aligned_vec::AVec::new(32);
        regs.resize(num_elems * count, default);
        RegisterFileImpl {
            num_elems: num_elems,
            regs: regs,
        }
    }

    fn get(&self, elem: usize, idx: usize) -> T {
        if elem >= self.num_elems {
            panic!("Element index out of bounds");
        }
        self.regs[self.num_elems * idx + elem]
    }

    fn set(&mut self, elem: usize, idx: usize, val: T) {
        if elem >= self.num_elems {
            panic!("Element index out of bounds");
        }
        self.regs[self.num_elems * idx + elem] = val
    }

    fn get_vec(&self, idx: usize) -> &[T] {
        let beg = self.num_elems * idx;
        let end = self.num_elems * (idx + 1);
        &self.regs.as_slice()[beg..end]
    }

    fn set_vec(&mut self, idx: usize, vals: &[Option<T>]) {
        for elem in 0..self.num_elems {
            if let Some(val) = vals[elem] {
                self.set(elem, idx, val);
            }
        }
    }
}

#[derive(FromPrimitive)]
#[repr(i32)]
pub enum Signals {
    None = 0,
    EndOfProgram = 1,
    Switch = 2,
    Unknown = 3,
}

pub trait Processor {
    fn step(&mut self) -> Signals;
}

#[derive(Clone, Debug)]
struct Context {
    id: usize,
    pc: u64,
    scc: bool,
    scratch: Rc<RefCell<AVec<u8, ConstAlign<0x1_0000_0000>>>>,
}

struct SIMD32 {
    slots: Vec<Context>,
    ctx: Context,
    insts: Vec<u8>,
    pub sgprs: RegisterFileImpl<u32>,
    pub vgprs: RegisterFileImpl<u32>,
    num_vgprs: usize,
    lds: Rc<RefCell<Vec<u8>>>,
    translator: RDNATranslator,
}

#[inline(always)]
fn u64_from_u32_u32(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | (lo as u64)
}

#[inline(always)]
fn add_u32(a: u32, b: u32, c: u32) -> (u32, bool) {
    let d = (a as u64) + (b as u64) + (c as u64);
    ((d & 0xFFFF_FFFF) as u32, d >= (0x1_0000_0000 as u64))
}

#[inline(always)]
fn mul_u32(a: u32, b: u32) -> u32 {
    let c = (a as u64) * (b as u64);
    (c & 0xFFFFFFFF) as u32
}

fn get_exp_f64(val: f64) -> i16 {
    let bits: u64 = f64::to_bits(val);
    ((bits >> 52) & 0x7ff) as i16
}

fn get_exp_f32(val: f32) -> i16 {
    let bits: u32 = f32::to_bits(val);
    ((bits >> 23) & 0xff) as i16
}

fn div_scale_f32(s0: f32, s1: f32, s2: f32) -> (f32, bool) {
    let mut vcc = false;
    let mut d = s0 * s2 / s1;
    let s1_exp = get_exp_f32(s1);
    let s2_exp = get_exp_f32(s2);
    if s2 == 0.0 || s1 == 0.0 {
        d = f32::NAN;
    } else if s2_exp - s1_exp >= 96 {
        // N/D near MAX_FLOAT
        vcc = true;
        if s0 == s1 {
            // Only scale the denominator
            d = s0 * 64f32.exp2();
        }
    } else if !s1.is_normal() {
        d = s0 * 64f32.exp2();
    } else if (!(1.0 / s1).is_normal()) && (!(s2 / s1).is_normal()) {
        vcc = true;
        if s0 == s1 {
            // Only scale the denominator
            d = s0 * 64f32.exp2();
        }
    } else if !(1.0 / s1).is_normal() {
        d = s0 * (-64f32).exp2();
    } else if !(s2 / s1).is_normal() {
        vcc = true;
        if s0 == s2 {
            // Only scale the numerator
            d = s0 * 64f32.exp2();
        }
    } else if s2_exp <= 23 {
        // Numerator is tiny
        d = s0 * 64f32.exp2();
    }
    (d, vcc)
}

fn div_scale_f64(s0: f64, s1: f64, s2: f64) -> (f64, bool) {
    let mut vcc = false;
    let mut d = s0 * s2 / s1;
    let s1_exp = get_exp_f64(s1);
    let s2_exp = get_exp_f64(s2);
    if s2 == 0.0 || s1 == 0.0 {
        d = f64::NAN;
    } else if s2_exp - s1_exp >= 768 {
        // N/D near MAX_FLOAT
        vcc = true;
        if s0 == s1 {
            // Only scale the denominator
            d = s0 * 128f64.exp2();
        }
    } else if !s1.is_normal() {
        d = s0 * 128f64.exp2();
    } else if (!(1.0 / s1).is_normal()) && (!(s2 / s1).is_normal()) {
        vcc = true;
        if s0 == s1 {
            // Only scale the denominator
            d = s0 * 128f64.exp2();
        }
    } else if !(1.0 / s1).is_normal() {
        d = s0 * (-128f64).exp2();
    } else if !(s2 / s1).is_normal() {
        vcc = true;
        if s0 == s2 {
            // Only scale the numerator
            d = s0 * 128f64.exp2();
        }
    } else if s2_exp <= 53 {
        // Numerator is tiny
        d = s0 * 128f64.exp2();
    }
    (d, vcc)
}

fn div_fixup_f32(s0: f32, s1: f32, s2: f32) -> f32 {
    let sign_out = s1.is_sign_negative() != s2.is_sign_negative();
    if s2.is_nan() {
        s2
    } else if s1.is_nan() {
        s1
    } else if s1 == 0.0 && s2 == 0.0 {
        // 0/0
        u32_to_f32(0xffc00000)
    } else if s1.is_infinite() && s2.is_infinite() {
        // inf/inf
        u32_to_f32(0xffc00000)
    } else if s1 == 0.0 || s2.is_infinite() {
        // x/0, or inf/y
        if sign_out {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    } else if s1.is_infinite() || s2 == 0.0 {
        // x/inf, 0/y
        if sign_out {
            -0.0
        } else {
            0.0
        }
    } else if (get_exp_f32(s2) - get_exp_f32(s1)) < -150 {
        if sign_out {
            -0.0
        } else {
            0.0
        }
    } else if get_exp_f32(s1) == 255 {
        if sign_out {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    } else {
        if sign_out {
            -s0.abs()
        } else {
            s0.abs()
        }
    }
}

fn div_fixup_f64(s0: f64, s1: f64, s2: f64) -> f64 {
    let sign_out = s1.is_sign_negative() != s2.is_sign_negative();
    if s2.is_nan() {
        s2
    } else if s1.is_nan() {
        s1
    } else if s1 == 0.0 && s2 == 0.0 {
        // 0/0
        u64_to_f64(0xfff8000000000000)
    } else if s1.is_infinite() && s2.is_infinite() {
        // inf/inf
        u64_to_f64(0xfff8000000000000)
    } else if s1 == 0.0 || s2.is_infinite() {
        // x/0, or inf/y
        if sign_out {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }
    } else if s1.is_infinite() || s2 == 0.0 {
        // x/inf, 0/y
        if sign_out {
            -0.0
        } else {
            0.0
        }
    } else if (get_exp_f64(s2) - get_exp_f64(s1)) < -1075 {
        if sign_out {
            -0.0
        } else {
            0.0
        }
    } else if get_exp_f64(s1) == 2047 {
        if sign_out {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        }
    } else {
        if sign_out {
            -s0.abs()
        } else {
            s0.abs()
        }
    }
}

fn cmp_class_f32(a: f32, mask: u32) -> bool {
    let mut result = false;
    if (mask & 0x01) != 0 {
        // value is a signaling NaN.
        result |= a.is_nan();
    }
    if (mask & 0x02) != 0 {
        // value is a quiet NaN.
        result |= a.is_nan();
    }
    if (mask & 0x04) != 0 {
        // value is negative infinity.
        result |= a.is_infinite() && a.is_sign_negative();
    }
    if (mask & 0x08) != 0 {
        // value is a negative normal value.
        result |= a.is_normal() && a.is_sign_negative();
    }
    if (mask & 0x10) != 0 {
        // value is a negative denormal value.
        result |= !a.is_normal() && a.is_sign_negative();
    }
    if (mask & 0x20) != 0 {
        // value is negative zero.
        result |= a == 0.0 && a.is_sign_negative();
    }
    if (mask & 0x40) != 0 {
        // value is positive zero.
        result |= a == 0.0 && a.is_sign_positive();
    }
    if (mask & 0x80) != 0 {
        // value is a positive denormal value.
        result |= !a.is_normal() && a.is_sign_positive();
    }
    if (mask & 0x100) != 0 {
        // value is a positive normal value.
        result |= a.is_normal() && a.is_sign_positive();
    }
    if (mask & 0x200) != 0 {
        // value is positive infinity.
        result |= a.is_infinite() && a.is_sign_positive();
    }
    result
}

fn cmp_class_f64(a: f64, mask: u32) -> bool {
    let mut result = false;
    if (mask & 0x01) != 0 {
        // value is a signaling NaN.
        result |= a.is_nan();
    }
    if (mask & 0x02) != 0 {
        // value is a quiet NaN.
        result |= a.is_nan();
    }
    if (mask & 0x04) != 0 {
        // value is negative infinity.
        result |= a.is_infinite() && a.is_sign_negative();
    }
    if (mask & 0x08) != 0 {
        // value is a negative normal value.
        result |= a.is_normal() && a.is_sign_negative();
    }
    if (mask & 0x10) != 0 {
        // value is a negative denormal value.
        result |= !a.is_normal() && a.is_sign_negative();
    }
    if (mask & 0x20) != 0 {
        // value is negative zero.
        result |= a == 0.0 && a.is_sign_negative();
    }
    if (mask & 0x40) != 0 {
        // value is positive zero.
        result |= a == 0.0 && a.is_sign_positive();
    }
    if (mask & 0x80) != 0 {
        // value is a positive denormal value.
        result |= !a.is_normal() && a.is_sign_positive();
    }
    if (mask & 0x100) != 0 {
        // value is a positive normal value.
        result |= a.is_normal() && a.is_sign_positive();
    }
    if (mask & 0x200) != 0 {
        // value is positive infinity.
        result |= a.is_infinite() && a.is_sign_positive();
    }
    result
}

use aligned_vec::AVec;
use aligned_vec::ConstAlign;
use half::f16;
use itertools::Itertools;
use num_traits::ops::mul_add::MulAdd;

#[inline(always)]
fn fma<T: MulAdd<Output = T>>(a: T, b: T, c: T) -> T {
    a.mul_add(b, c)
}

#[inline(always)]
fn u32_to_f32(value: u32) -> f32 {
    f32::from_bits(value)
}

#[inline(always)]
fn u64_to_f64(value: u64) -> f64 {
    f64::from_bits(value)
}

#[inline(always)]
fn f32_to_u32(value: f32) -> u32 {
    f32::to_bits(value)
}

fn abs_neg<T: num::Float>(value: T, abs: u8, neg: u8, idx: usize) -> T {
    let mut result = value;
    if ((abs >> idx) & 1) != 0 {
        result = result.abs();
    }
    if ((neg >> idx) & 1) != 0 {
        result = -result;
    }
    result
}

fn abs_neg_f16(value: f16, abs: u8, neg: u8, idx: usize) -> f16 {
    let mut result = value;
    if ((abs >> idx) & 1) != 0 {
        if result.is_sign_negative() {
            result = -result;
        }
    }
    if ((neg >> idx) & 1) != 0 {
        result = -result;
    }
    result
}

#[inline(always)]
fn u32_to_f32_abs_neg(value: u32, abs: u8, neg: u8, idx: usize) -> f32 {
    let result = f32::from_bits(value);
    abs_neg(result, abs, neg, idx)
}

#[inline(always)]
fn f64_to_u64(value: f64) -> u64 {
    f64::to_bits(value)
}

fn clamp_f32(value: f32, clamp: bool) -> f32 {
    if clamp {
        value.clamp(0.0, 1.0)
    } else {
        value
    }
}

fn clamp_f64(value: f64, clamp: bool) -> f64 {
    if clamp {
        value.clamp(0.0, 1.0)
    } else {
        value
    }
}

fn f32_to_u32_omod_clamp(value: f32, omod: u8, clamp: bool) -> u32 {
    let value = match omod {
        1 => value * 2.0,
        2 => value * 4.0,
        3 => value * 0.5,
        _ => value,
    };
    f32::to_bits(clamp_f32(value, clamp))
}

fn f64_to_u64_omod_clamp(value: f64, omod: u8, clamp: bool) -> u64 {
    let value = match omod {
        1 => value * 2.0,
        2 => value * 4.0,
        3 => value * 0.5,
        _ => value,
    };
    f64::to_bits(clamp_f64(value, clamp))
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Box4Node {
    pub child_index: [u32; 4],
    pub aabb: [Aabb; 4],
    pub parent_addr: u32,
    pub update_counter: u32,
    pub child_count: u32,
}

fn intersect(ray_origin: [f32; 3], inv_direction: [f32; 3], aabb: &Aabb, max_t: f32) -> (f32, f32) {
    let f = [
        (aabb.max[0] - ray_origin[0]) * inv_direction[0],
        (aabb.max[1] - ray_origin[1]) * inv_direction[1],
        (aabb.max[2] - ray_origin[2]) * inv_direction[2],
    ];
    let n = [
        (aabb.min[0] - ray_origin[0]) * inv_direction[0],
        (aabb.min[1] - ray_origin[1]) * inv_direction[1],
        (aabb.min[2] - ray_origin[2]) * inv_direction[2],
    ];
    let tmax = [f[0].max(n[0]), f[1].max(n[1]), f[2].max(n[2])];
    let tmin = [f[0].min(n[0]), f[1].min(n[1]), f[2].min(n[2])];
    let t1 = tmax[0].min(tmax[1].min(tmax[2].min(max_t)));
    let t0 = tmin[0].max(tmin[1].max(tmin[2].max(0.0)));
    (t0, t1)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn intersect_triangle_frac(
    ray_origin: [f32; 3],
    ray_direction: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    flags: u32,
) -> (f32, f32, f32, f32) {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let s1 = cross(ray_direction, e2);
    let denom = dot(s1, e1);
    if denom == 0.0 {
        let result0 = f32::INFINITY;
        let result1 = 0.0;
        let result2 = 0.0;
        let result3 = 0.0;

        return (result0, result1, result2, result3);
    }
    let d = [
        ray_origin[0] - v0[0],
        ray_origin[1] - v0[1],
        ray_origin[2] - v0[2],
    ];
    let b_y = dot(d, s1);
    let s2 = cross(d, e1);
    let b_z = dot(ray_direction, s2);
    let t: f32 = dot(e2, s2);
    let b_x = denom - b_y - b_z;
    let barycentrics = [b_x, b_y, b_z];

    let result0 = if (denom > 0.0)
        && (b_y < 0.0 || b_y > denom || b_z < 0.0 || (b_y + b_z) > denom || (t < 0.0))
    {
        f32::INFINITY
    } else if (denom < 0.0)
        && (b_y > 0.0 || b_y < denom || b_z > 0.0 || (b_y + b_z) < denom || (t > 0.0))
    {
        f32::INFINITY
    } else {
        t
    };

    let result1 = denom;
    let result2 = barycentrics[((flags >> 0) & 3) as usize];
    let result3 = barycentrics[((flags >> 2) & 3) as usize];

    (result0, result1, result2, result3)
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Box8Node {
    data: [u32; 32],
}

impl Box8Node {
    pub fn get_box_node_base(&self) -> u32 {
        self.data[0]
    }

    pub fn get_prim_node_base(&self) -> u32 {
        self.data[1]
    }

    pub fn get_parent_addr(&self) -> u32 {
        self.data[2]
    }

    pub fn get_origin(&self) -> [f32; 3] {
        let x = f32::from_bits(self.data[3]);
        let y = f32::from_bits(self.data[4]);
        let z = f32::from_bits(self.data[5]);
        [x, y, z]
    }

    pub fn get_exponent(&self) -> [u8; 3] {
        let x = (self.data[6] & 0xFF) as u8;
        let y = ((self.data[6] >> 8) & 0xFF) as u8;
        let z = ((self.data[6] >> 16) & 0xFF) as u8;
        [x, y, z]
    }

    pub fn get_child_count(&self) -> u8 {
        ((self.data[6] >> 28) & 0x0F) as u8 + 1
    }

    pub fn get_matrix_id(&self) -> u32 {
        (self.data[7] as u32) & 0x7F
    }

    pub fn get_child_box(&self, index: usize) -> Aabb {
        let exponent = self.get_exponent();
        let origin = self.get_origin();

        let rcp_exponent = [
            f32::from_bits((254 - (exponent[0] as u32) + 12) << 23),
            f32::from_bits((254 - (exponent[1] as u32) + 12) << 23),
            f32::from_bits((254 - (exponent[2] as u32) + 12) << 23),
        ];

        let min_x = origin[0] + (self.data[8 + index * 3] & 0x00000FFF) as f32 / rcp_exponent[0];
        let min_y =
            origin[1] + ((self.data[8 + index * 3] >> 12) & 0x00000FFF) as f32 / rcp_exponent[1];
        let min_z = origin[2] + ((self.data[9 + index * 3]) & 0x00000FFF) as f32 / rcp_exponent[2];
        let max_x = origin[0]
            + if exponent[0] != 0 {
                ((self.data[9 + index * 3] >> 12) & 0x00000FFF) as f32 / rcp_exponent[0]
            } else {
                0.0
            };
        let max_y = origin[1]
            + if exponent[1] != 0 {
                (self.data[10 + index * 3] & 0x00000FFF) as f32 / rcp_exponent[1]
            } else {
                0.0
            };
        let max_z = origin[2]
            + if exponent[2] != 0 {
                ((self.data[10 + index * 3] >> 12) & 0x00000FFF) as f32 / rcp_exponent[2]
            } else {
                0.0
            };

        Aabb {
            min: [min_x, min_y, min_z],
            max: [max_x, max_y, max_z],
        }
    }

    pub fn get_child_type(&self, index: usize) -> u8 {
        ((self.data[10 + index * 3] >> 24) & 0x0F) as u8
    }

    pub fn get_child_addr(&self, index: usize) -> u32 {
        let child_type = self.get_child_type(index);
        let mut child_addr = if child_type == 5 {
            self.data[0] >> 4
        } else {
            self.data[1] >> 4
        };
        for j in 0..index {
            if (self.get_child_type(j) == 5) == (child_type == 5) {
                let node_range = (self.data[10 + j * 3] >> 28) & 0x0F;
                child_addr += node_range;
            }
        }
        child_addr
    }

    pub fn get_child_index(&self, index: usize) -> u32 {
        (self.get_child_addr(index) << 4) | (self.get_child_type(index) as u32)
    }
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
struct TrianglePacketNode {
    data: [u32; 32],
}

impl TrianglePacketNode {
    pub fn read_unaligned_bits(&self, position: u32, length: u32) -> u32 {
        let mut data = 0u64;
        if length != 0 {
            data = self.data[(position / 32) as usize] as u64;
            if (position + length - 1) / 32 != position / 32 {
                data |= (self.data[((position + length - 1) / 32) as usize] as u64) << 32;
            }
            data >>= position % 32;
            data &= (1 << length) - 1;
        }
        data as u32
    }

    pub fn read_vertex(&self, vertex_index: u32) -> [f32; 3] {
        let position = 52 + 96 * vertex_index;

        let x_bits = self.read_unaligned_bits(position + 0 * 32, 32);
        let y_bits = self.read_unaligned_bits(position + 1 * 32, 32);
        let z_bits = self.read_unaligned_bits(position + 2 * 32, 32);

        [
            f32::from_bits(x_bits),
            f32::from_bits(y_bits),
            f32::from_bits(z_bits),
        ]
    }

    pub fn read_descriptor(&self, pair_index: u32, triangle_index: u32) -> [u32; 4] {
        let position = 1024 - (pair_index + 1) * 29;
        let descriptor = self.read_unaligned_bits(position, 29);
        let tri_indices = if triangle_index > 0 {
            descriptor >> 3
        } else {
            descriptor >> 17
        };
        [
            tri_indices & 15,
            (tri_indices >> 4) & 15,
            (tri_indices >> 8) & 15,
            descriptor & 1,
        ]
    }

    pub fn fetch_triangle(&self, pair_index: u32, triangle_index: u32) -> [[f32; 3]; 3] {
        let tri_indices = self.read_descriptor(pair_index, triangle_index);

        let v0 = self.read_vertex(tri_indices[0]);
        let v1 = self.read_vertex(tri_indices[1]);
        let v2 = self.read_vertex(tri_indices[2]);

        [v0, v1, v2]
    }

    pub fn get_triangle_pair_count(&self) -> u32 {
        self.read_unaligned_bits(28, 3) + 1
    }

    pub fn get_index_section_midpoint(&self) -> u32 {
        self.read_unaligned_bits(32 + 10, 10)
    }

    pub fn get_prim_index_anchor_size(&self) -> u32 {
        self.read_unaligned_bits(32 + 0, 5)
    }

    pub fn get_prim_index_payload_size(&self) -> u32 {
        self.read_unaligned_bits(32 + 5, 5)
    }

    pub fn read_prim_index(&self, pair_index: u32, triangle_index: u32) -> u32 {
        let flat_tri_index = 2 * pair_index + triangle_index;

        let prim_index_payload_size = self.get_prim_index_payload_size();
        let prim_index_anchor_size = self.get_prim_index_anchor_size();
        let prim_index_anchor_pos = self.get_index_section_midpoint();

        let prim_index_anchor =
            self.read_unaligned_bits(prim_index_anchor_pos, prim_index_anchor_size);
        if flat_tri_index == 0 {
            return prim_index_anchor;
        }
        let prim_index_payload_pos = prim_index_anchor_pos
            + prim_index_anchor_size
            + (flat_tri_index - 1) * prim_index_payload_size;

        let prim_index = self.read_unaligned_bits(prim_index_payload_pos, prim_index_payload_size);
        let prim_index_mask = (1 << prim_index_payload_size) - 1;

        if prim_index_payload_size >= prim_index_anchor_size {
            prim_index
        } else {
            prim_index | (prim_index_anchor & !prim_index_mask)
        }
    }

    pub fn get_prim_index(&self, pair_index: u32, triangle_index: u32) -> u32 {
        self.read_prim_index(pair_index, triangle_index)
    }

    pub fn is_range_end(&self, pair_index: u32) -> bool {
        let descriptor = self.read_descriptor(pair_index, 0);
        descriptor[3] != 0
    }
}

fn intersect_triangle(
    ray_origin: [f32; 3],
    ray_direction: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> (f32, f32, f32) {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let s1 = cross(ray_direction, e2);
    let denom = dot(s1, e1);
    if denom == 0.0 {
        let result0 = f32::INFINITY;
        let result1 = 0.0;
        let result2 = 0.0;

        return (result0, result1, result2);
    }
    let d = [
        ray_origin[0] - v0[0],
        ray_origin[1] - v0[1],
        ray_origin[2] - v0[2],
    ];
    let inv_denom = 1.0 / denom;
    let b_y = dot(d, s1) * inv_denom;
    let s2 = cross(d, e1);
    let b_z = dot(ray_direction, s2) * inv_denom;
    let t: f32 = dot(e2, s2) * inv_denom;

    let t = if b_y < 0.0 || b_y > 1.0 || b_z < 0.0 || (b_y + b_z) > 1.0 {
        f32::INFINITY
    } else {
        t
    };

    (t, b_y, b_z)
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct TrianglePair {
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    v3: [f32; 3],
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
struct TrianglePairNode {
    pub tri_pair: TrianglePair,
    pub padding: u32,
    pub prim_index: [u32; 2],
    pub flags: u32,
}

impl SIMD32 {
    pub fn dispatch(&mut self, entry_addr: usize, setup_data: Vec<RegisterSetupData>) {
        let num_wavefronts = setup_data.len();
        let num_sgprs = 128;

        for i in 0..num_sgprs {
            for slot in 0..16 {
                self.sgprs.set(0, slot * num_sgprs + i, 0);
            }
        }
        for i in 0..(num_wavefronts * self.num_vgprs) {
            for elem in 0..32 {
                self.vgprs.set(elem, i, 0);
            }
        }
        for wavefront in 0..num_wavefronts {
            let vgpr_offset = wavefront * self.num_vgprs;
            let sgprs = setup_data[wavefront].sgprs;
            let vgprs = setup_data[wavefront].vgprs;
            let user_sgpr_count = setup_data[wavefront].user_sgpr_count;
            for i in 0..user_sgpr_count {
                self.sgprs.set(0, wavefront * num_sgprs + i, sgprs[i]);
            }
            self.sgprs.set(0, wavefront * num_sgprs + 126, 0xFFFFFFFF); // EXEC_LO
            self.sgprs.set(0, wavefront * num_sgprs + 127, 0xFFFFFFFF); // EXEC_HI
            self.sgprs
                .set(0, wavefront * num_sgprs + 117, sgprs[user_sgpr_count]); // TTMP9
            self.sgprs.set(
                0,
                wavefront * num_sgprs + 115,
                (sgprs[user_sgpr_count + 2] << 16) | sgprs[user_sgpr_count + 1],
            ); // TTMP7
            for i in 0..16 {
                for elem in 0..32 {
                    self.vgprs.set(elem, vgpr_offset + i, vgprs[i][elem]);
                }
            }
        }

        let mut slots = Vec::new();
        for (wavefront, data) in setup_data.into_iter().enumerate() {
            slots.push(Context {
                id: wavefront,
                pc: entry_addr as u64,
                scc: false,
                scratch: data.scratch,
            });
        }

        self.slots = slots;
    }

    fn step(&mut self) -> Signals {
        let inst_stream = InstStream {
            insts: &self.insts[self.ctx.pc as usize..],
        };

        if USE_INTERPRETER {
            if let Ok((inst, size)) = decode_rdna4(inst_stream) {
                let result = self.execute_inst(inst);
                self.ctx.pc += size as u64;
                result
            } else {
                let inst = get_u64(&self.insts, self.ctx.pc as usize);
                println!(
                    "Unknown instruction 0x{:08X} at PC: 0x{:08X}",
                    inst & 0xFFFFFFFF,
                    self.ctx.pc
                );
                Signals::Unknown
            }
        } else {
            let pc = self.ctx.pc as u64;
            let block = self.translator.insts_blocks.get_mut(&pc);
            if block.is_some() && self.translator.insts.len() == 0 {
                let block = block.unwrap();

                let sgprs_ptr =
                    self.sgprs.regs.as_mut_ptr().wrapping_add(128 * self.ctx.id) as *mut u32;
                let vgprs_ptr = (self
                    .vgprs
                    .regs
                    .as_mut_ptr()
                    .wrapping_add(self.num_vgprs * self.ctx.id * 32))
                    as *mut u32;
                let scc_ptr = (&mut self.ctx.scc) as *mut bool;
                let lds_ptr = self.lds.borrow_mut().as_mut_ptr();
                let scratch_ptr = self.ctx.scratch.borrow_mut().as_mut_ptr() as u64;

                block.execute(
                    sgprs_ptr,
                    vgprs_ptr,
                    scc_ptr,
                    &mut self.ctx.pc,
                    scratch_ptr,
                    lds_ptr,
                )
            } else if let Ok((inst, size)) = decode_rdna4(inst_stream) {
                self.translator.add_inst(self.ctx.pc as u64, inst.clone());
                let result = if is_terminator(&inst) {
                    if self.translator.insts.len() > 0 {
                        let block = self
                            .translator
                            .get_or_build(self.ctx.scratch.borrow().len() / 32);

                        let sgprs_ptr = self.sgprs.regs.as_mut_ptr().wrapping_add(128 * self.ctx.id)
                            as *mut u32;
                        let vgprs_ptr = (self
                            .vgprs
                            .regs
                            .as_mut_ptr()
                            .wrapping_add(self.num_vgprs * self.ctx.id * 32))
                            as *mut u32;
                        let scc_ptr = (&mut self.ctx.scc) as *mut bool;
                        let lds_ptr = self.lds.borrow_mut().as_mut_ptr();
                        let scratch_ptr = self.ctx.scratch.borrow_mut().as_mut_ptr() as u64;

                        block.execute(
                            sgprs_ptr,
                            vgprs_ptr,
                            scc_ptr,
                            &mut self.ctx.pc,
                            scratch_ptr,
                            lds_ptr,
                        )
                    } else {
                        self.ctx.pc += size as u64;
                        Signals::None
                    }
                } else {
                    self.ctx.pc += size as u64;
                    Signals::None
                };

                result
            } else {
                let inst = get_u64(&self.insts, self.ctx.pc as usize);
                println!(
                    "Unknown instruction 0x{:08X} at PC: 0x{:08X}",
                    inst & 0xFFFFFFFF,
                    self.ctx.pc
                );
                Signals::Unknown
            }
        }
    }

    fn is_execz(&self) -> bool {
        self.get_exec() == 0
    }

    fn is_execnz(&self) -> bool {
        !self.is_execz()
    }

    fn is_vccz(&self) -> bool {
        self.get_vcc() == 0
    }

    fn is_vccnz(&self) -> bool {
        !self.is_vccz()
    }

    fn read_sgpr(&self, idx: usize) -> u32 {
        if idx == 124 {
            0 // NULL
        } else {
            self.sgprs.get(0, self.ctx.id * 128 + idx)
        }
    }

    fn read_sgpr_pair(&self, idx: usize) -> u64 {
        u64_from_u32_u32(self.read_sgpr(idx), self.read_sgpr(idx + 1))
    }

    fn write_sgpr(&mut self, idx: usize, value: u32) {
        self.sgprs.set(0, self.ctx.id * 128 + idx, value);
    }

    fn read_vgpr(&self, elem: usize, idx: usize) -> u32 {
        if idx >= self.num_vgprs {
            panic!();
        }
        self.vgprs.get(elem, self.num_vgprs * self.ctx.id + idx)
    }

    fn read_vgpr_pair(&self, elem: usize, idx: usize) -> u64 {
        u64_from_u32_u32(self.read_vgpr(elem, idx), self.read_vgpr(elem, idx + 1))
    }

    fn write_vgpr(&mut self, elem: usize, idx: usize, value: u32) {
        if idx >= self.num_vgprs {
            panic!();
        }
        self.vgprs
            .set(elem, self.num_vgprs * self.ctx.id + idx, value);
    }

    fn write_vgpr_pair(&mut self, elem: usize, idx: usize, value: u64) {
        self.write_vgpr(elem, idx, (value & 0xFFFFFFFF) as u32);
        self.write_vgpr(elem, idx + 1, ((value >> 32) & 0xFFFFFFFF) as u32);
    }

    fn set_sgpr_bit(&mut self, idx: usize, bit: usize, value: bool) {
        if bit >= 32 {
            let mask = 1 << (bit - 32);
            let old_value = self.read_sgpr(idx + 1);
            self.write_sop_dst(
                idx + 1,
                (old_value & !mask) | ((value as u32) << (bit - 32)),
            );
        } else {
            let mask = 1 << bit;
            let old_value = self.read_sgpr(idx);
            self.write_sop_dst(idx, (old_value & !mask) | ((value as u32) << bit));
        }
    }

    fn get_exec(&self) -> u32 {
        self.read_sgpr(126)
    }

    fn set_exec(&mut self, value: u32) {
        self.write_sgpr(126, value);
    }

    fn get_exec_bit(&self, elem: usize) -> bool {
        if elem >= 32 {
            ((self.read_sgpr(127) >> (elem - 32)) & 1) != 0
        } else {
            ((self.read_sgpr(126) >> elem) & 1) != 0
        }
    }

    fn set_exec_bit(&mut self, elem: usize, value: bool) {
        if elem >= 32 {
            let mask = 1 << (elem - 32);
            let old_value = self.read_sgpr(127);
            self.write_sgpr(127, (old_value & !mask) | ((value as u32) << (elem - 32)));
        } else {
            let mask = 1 << elem;
            let old_value = self.read_sgpr(126);
            self.write_sgpr(126, (old_value & !mask) | ((value as u32) << elem));
        }
    }

    fn get_vcc(&self) -> u32 {
        self.read_sgpr(106)
    }

    fn get_vcc_bit(&self, elem: usize) -> bool {
        if elem >= 32 {
            ((self.read_sgpr(107) >> (elem - 32)) & 1) != 0
        } else {
            ((self.read_sgpr(106) >> elem) & 1) != 0
        }
    }

    fn set_vcc_bit(&mut self, elem: usize, value: bool) {
        if elem >= 32 {
            let mask = 1 << (elem - 32);
            let old_value = self.read_sgpr(107);
            self.write_sgpr(107, (old_value & !mask) | ((value as u32) << (elem - 32)));
        } else {
            let mask = 1 << elem;
            let old_value = self.read_sgpr(106);
            self.write_sgpr(106, (old_value & !mask) | ((value as u32) << elem));
        }
    }

    fn write_sop_dst(&mut self, addr: usize, value: u32) {
        match addr {
            0..=105 => self.write_sgpr(addr, value),
            106 => self.write_sgpr(addr, value),
            107 => self.write_sgpr(addr, value),
            108..=123 => self.write_sgpr(addr, value),
            124 => {}                            // NULL
            126 => self.write_sgpr(addr, value), // EXEC_LO
            127 => self.write_sgpr(addr, value), // EXEC_HI
            _ => panic!(),
        }
    }

    fn write_sop_dst_pair(&mut self, addr: usize, value: u64) {
        self.write_sop_dst(addr, (value & 0xFFFFFFFF) as u32);
        self.write_sop_dst(addr + 1, ((value >> 32) & 0xFFFFFFFF) as u32);
    }

    fn read_scalar_source_operand_u32(&self, addr: SourceOperand) -> u32 {
        match addr {
            SourceOperand::LiteralConstant(value) => value,
            SourceOperand::IntegerConstant(value) => (value & 0xFFFFFFFF) as u32,
            SourceOperand::FloatConstant(value) => f32_to_u32(value as f32),
            SourceOperand::ScalarRegister(value) => self.read_sgpr(value as usize),
            SourceOperand::VectorRegister(_) => panic!(),
            SourceOperand::PrivateBase => panic!(),
        }
    }

    fn read_scalar_source_operand_u64(&self, addr: SourceOperand) -> u64 {
        match addr {
            SourceOperand::LiteralConstant(value) => value as u64,
            SourceOperand::IntegerConstant(value) => value,
            SourceOperand::FloatConstant(value) => f64_to_u64(value),
            SourceOperand::ScalarRegister(value) => self.read_sgpr_pair(value as usize),
            SourceOperand::VectorRegister(_) => panic!(),
            SourceOperand::PrivateBase => self.ctx.scratch.borrow_mut().as_ptr() as u64,
        }
    }

    fn read_vector_source_operand_u32(&self, elem: usize, addr: SourceOperand) -> u32 {
        match addr {
            SourceOperand::LiteralConstant(value) => value,
            SourceOperand::IntegerConstant(value) => (value & 0xFFFFFFFF) as u32,
            SourceOperand::FloatConstant(value) => f32_to_u32(value as f32),
            SourceOperand::ScalarRegister(value) => self.read_sgpr(value as usize),
            SourceOperand::VectorRegister(value) => self.read_vgpr(elem, value as usize),
            SourceOperand::PrivateBase => panic!(),
        }
    }

    fn read_vector_source_operand_f16(&self, elem: usize, addr: SourceOperand) -> f16 {
        match addr {
            SourceOperand::LiteralConstant(value) => f16::from_bits(value as u16 & 0xFFFF),
            SourceOperand::IntegerConstant(value) => f16::from_bits(value as u16 & 0xFFFF),
            SourceOperand::FloatConstant(value) => f16::from_f32(value as f32),
            SourceOperand::ScalarRegister(value) => {
                f16::from_bits((self.read_sgpr(value as usize) & 0xFFFF) as u16)
            }
            SourceOperand::VectorRegister(value) => {
                f16::from_bits((self.read_vgpr(elem, value as usize) & 0xFFFF) as u16)
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    fn read_vector_source_operand_f16_hi(&self, elem: usize, addr: SourceOperand) -> f16 {
        match addr {
            SourceOperand::LiteralConstant(value) => f16::from_bits(value as u16 & 0xFFFF),
            SourceOperand::IntegerConstant(value) => f16::from_bits(value as u16 & 0xFFFF),
            SourceOperand::FloatConstant(value) => f16::from_f32(value as f32),
            SourceOperand::ScalarRegister(value) => {
                f16::from_bits(((self.read_sgpr(value as usize) >> 16) & 0xFFFF) as u16)
            }
            SourceOperand::VectorRegister(value) => {
                f16::from_bits(((self.read_vgpr(elem, value as usize) >> 16) & 0xFFFF) as u16)
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    fn read_vector_source_operand_f16_vec<const N: usize>(
        &self,
        elem: usize,
        addr: SourceOperand,
    ) -> [f16; N] {
        match addr {
            SourceOperand::LiteralConstant(value) => {
                let value = f16::from_bits(value as u16 & 0xFFFF);
                [value; N]
            }
            SourceOperand::IntegerConstant(value) => {
                let value = f16::from_bits(value as u16 & 0xFFFF);
                [value; N]
            }
            SourceOperand::FloatConstant(value) => {
                let value = f16::from_f64(value);
                [value; N]
            }
            SourceOperand::ScalarRegister(value) => {
                let mut result = [f16::ZERO; N];
                for i in (0..N).step_by(2) {
                    assert!(i + 1 < N);
                    let reg_value = self.read_sgpr(value as usize + i / 2);
                    let value_lo = f16::from_bits((reg_value & 0xFFFF) as u16);
                    let value_hi = f16::from_bits((reg_value >> 16) as u16);
                    result[i] = value_lo;
                    result[i + 1] = value_hi;
                }
                return result;
            }
            SourceOperand::VectorRegister(value) => {
                let mut result = [f16::ZERO; N];
                for i in (0..N).step_by(2) {
                    assert!(i + 1 < N);
                    let reg_value = self.read_vgpr(elem, value as usize + i / 2);
                    let value_lo = f16::from_bits((reg_value & 0xFFFF) as u16);
                    let value_hi = f16::from_bits((reg_value >> 16) as u16);
                    result[i] = value_lo;
                    result[i + 1] = value_hi;
                }
                return result;
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    fn read_vector_source_operand_u64(&self, elem: usize, addr: SourceOperand) -> u64 {
        match addr {
            SourceOperand::LiteralConstant(value) => value as u64,
            SourceOperand::IntegerConstant(value) => value,
            SourceOperand::FloatConstant(value) => f64_to_u64(value),
            SourceOperand::ScalarRegister(value) => self.read_sgpr_pair(value as usize),
            SourceOperand::VectorRegister(value) => self.read_vgpr_pair(elem, value as usize),
            SourceOperand::PrivateBase => panic!(),
        }
    }

    fn read_vector_source_operand_f32(&self, elem: usize, addr: SourceOperand) -> f32 {
        match addr {
            SourceOperand::LiteralConstant(value) => u32_to_f32(value),
            SourceOperand::IntegerConstant(value) => u32_to_f32((value & 0xFFFFFFFF) as u32),
            SourceOperand::FloatConstant(value) => value as f32,
            SourceOperand::ScalarRegister(value) => u32_to_f32(self.read_sgpr(value as usize)),
            SourceOperand::VectorRegister(value) => {
                u32_to_f32(self.read_vgpr(elem, value as usize))
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    fn read_vector_source_operand_f32_vec<const N: usize>(
        &self,
        elem: usize,
        addr: SourceOperand,
    ) -> [f32; N] {
        match addr {
            SourceOperand::LiteralConstant(value) => {
                let value = u32_to_f32(value);
                [value; N]
            }
            SourceOperand::IntegerConstant(value) => {
                let value = u32_to_f32((value & 0xFFFFFFFF) as u32);
                [value; N]
            }
            SourceOperand::FloatConstant(value) => {
                let value = value as f32;
                [value; N]
            }
            SourceOperand::ScalarRegister(value) => {
                let mut result = [0.0f32; N];
                for i in 0..N {
                    result[i] = u32_to_f32(self.read_sgpr(value as usize + i));
                }
                return result;
            }
            SourceOperand::VectorRegister(value) => {
                let mut result = [0.0f32; N];
                for i in 0..N {
                    result[i] = u32_to_f32(self.read_vgpr(elem, value as usize + i));
                }
                return result;
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    fn read_vector_source_operand_f64(&self, elem: usize, addr: SourceOperand) -> f64 {
        match addr {
            SourceOperand::LiteralConstant(value) => u64_to_f64((value as u64) << 32),
            SourceOperand::IntegerConstant(value) => u64_to_f64(value),
            SourceOperand::FloatConstant(value) => value,
            SourceOperand::ScalarRegister(value) => u64_to_f64(self.read_sgpr_pair(value as usize)),
            SourceOperand::VectorRegister(value) => {
                u64_to_f64(self.read_vgpr_pair(elem, value as usize))
            }
            SourceOperand::PrivateBase => panic!(),
        }
    }

    fn execute_inst(&mut self, inst: InstFormat) -> Signals {
        match inst {
            InstFormat::SOP1(fields) => self.execute_sop1(fields),
            InstFormat::SOP2(fields) => self.execute_sop2(fields),
            InstFormat::SOPC(fields) => self.execute_sopc(fields),
            InstFormat::SOPK(fields) => self.execute_sopk(fields),
            InstFormat::VOP1(fields) => self.execute_vop1(fields),
            InstFormat::VOP2(fields) => self.execute_vop2(fields),
            InstFormat::VOP3(fields) => self.execute_vop3(fields),
            InstFormat::VOP3SD(fields) => self.execute_vop3sd(fields),
            InstFormat::VOP3P(fields) => self.execute_vop3p(fields),
            InstFormat::VOPC(fields) => self.execute_vopc(fields),
            InstFormat::VOPD(fields) => self.execute_vopd(fields),
            InstFormat::SMEM(fields) => self.execute_smem(fields),
            InstFormat::SOPP(fields) => self.execute_sopp(fields),
            InstFormat::VFLAT(fields) => self.execute_vflat(fields),
            InstFormat::VSCRATCH(fields) => self.execute_vscratch(fields),
            InstFormat::VGLOBAL(fields) => self.execute_vglobal(fields),
            InstFormat::VIMAGE(fields) => self.execute_vimage(fields),
            InstFormat::DS(fields) => self.execute_ds(fields),
        }
    }

    fn execute_sop1(&mut self, inst: SOP1) -> Signals {
        let d = inst.sdst as usize;
        let s0 = inst.ssrc0;

        match inst.op {
            I::S_MOV_B32 => {
                self.s_mov_b32(d, s0);
            }
            I::S_MOV_B64 => {
                self.s_mov_b64(d, s0);
            }
            I::S_CTZ_I32_B32 => {
                self.s_ctz_i32_b32(d, s0);
            }
            I::S_CVT_F32_I32 => {
                self.s_cvt_f32_i32(d, s0);
            }
            I::S_AND_SAVEEXEC_B32 => {
                self.s_and_saveexec_b32(d, s0);
            }
            I::S_OR_SAVEEXEC_B32 => {
                self.s_or_saveexec_b32(d, s0);
            }
            I::S_AND_NOT1_SAVEEXEC_B32 => {
                self.s_and_not1_saveexec_b32(d, s0);
            }
            I::S_SEXT_I32_I16 => {
                self.s_sext_i32_i16(d, s0);
            }
            I::S_BARRIER_SIGNAL => {
                let sig = self.read_scalar_source_operand_u32(s0) as i32;
                assert!(sig == -1);
            }
            I::S_GETPC_B64 => {
                let pc = self.ctx.pc + 4 + self.insts.as_ptr() as u64;
                self.write_sop_dst_pair(d, pc);
            }
            _ => unimplemented!("{:?}", inst.op),
        }

        Signals::None
    }

    fn s_mov_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let d_value = s0_value;
        self.write_sop_dst(d, d_value);
    }

    fn s_mov_b64(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let d_value = s0_value;
        self.write_sop_dst_pair(d, d_value);
    }

    fn s_ctz_i32_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let d_value = match s0_value.trailing_zeros() {
            n if n >= 32 => -1,
            n => n as i32,
        };
        self.write_sop_dst(d, d_value as u32);
    }

    fn s_cvt_f32_i32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let d_value = s0_value as i32 as f32;
        self.write_sop_dst(d, f32_to_u32(d_value));
    }

    fn s_and_saveexec_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let exec_value = self.get_exec();

        self.write_sop_dst(d, exec_value);

        let exec_value = s0_value & exec_value;

        self.set_exec(exec_value);
        self.ctx.scc = exec_value != 0;
    }

    fn s_or_saveexec_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let exec_value = self.get_exec();

        self.write_sop_dst(d, exec_value);

        let exec_value = s0_value | exec_value;

        self.set_exec(exec_value);
        self.ctx.scc = exec_value != 0;
    }

    fn s_and_not1_saveexec_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let exec_value = self.get_exec();

        self.write_sop_dst(d, exec_value);

        let exec_value = s0_value & !exec_value;

        self.set_exec(exec_value);
        self.ctx.scc = exec_value != 0;
    }

    fn s_sext_i32_i16(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as u16;
        let d_value = (s0_value as i16) as i32;

        self.write_sop_dst(d, d_value as u32);
    }

    fn execute_sop2(&mut self, inst: SOP2) -> Signals {
        let d = inst.sdst as usize;
        let s0 = inst.ssrc0;
        let s1 = inst.ssrc1;

        match inst.op {
            I::S_ADD_CO_I32 => {
                self.s_add_co_i32(d, s0, s1);
            }
            I::S_SUB_CO_I32 => {
                self.s_sub_co_i32(d, s0, s1);
            }
            I::S_ADD_NC_U64 => {
                self.s_add_nc_u64(d, s0, s1);
            }
            I::S_AND_B32 => {
                self.s_and_b32(d, s0, s1);
            }
            I::S_OR_B32 => {
                self.s_or_b32(d, s0, s1);
            }
            I::S_XOR_B32 => {
                self.s_xor_b32(d, s0, s1);
            }
            I::S_AND_NOT1_B32 => {
                self.s_and_not1_b32(d, s0, s1);
            }
            I::S_OR_NOT1_B32 => {
                self.s_or_not1_b32(d, s0, s1);
            }
            I::S_CSELECT_B32 => {
                self.s_cselect_b32(d, s0, s1);
            }
            I::S_BFM_B32 => {
                self.s_bfm_b32(d, s0, s1);
            }
            I::S_MUL_U64 => {
                self.s_mul_u64(d, s0, s1);
            }
            I::S_MUL_I32 => {
                self.s_mul_i32(d, s0, s1);
            }
            I::S_MUL_HI_U32 => {
                self.s_mul_hi_u32(d, s0, s1);
            }
            I::S_LSHR_B32 => {
                self.s_lshr_b32(d, s0, s1);
            }
            I::S_LSHL_B32 => {
                self.s_lshl_b32(d, s0, s1);
            }
            I::S_LSHL_B64 => {
                self.s_lshl_b64(d, s0, s1);
            }
            I::S_MAX_U32 => {
                self.s_max_u32(d, s0, s1);
            }
            I::S_ADD_CO_U32 => {
                self.s_add_co_u32(d, s0, s1);
            }
            I::S_ADD_CO_CI_U32 => {
                self.s_add_co_ci_u32(d, s0, s1);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn s_add_co_i32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        let (d_value, scc_value) = s0_value.overflowing_add(s1_value);
        self.write_sop_dst(d, d_value as u32);
        self.ctx.scc = scc_value;
    }

    fn s_sub_co_i32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        let (d_value, scc_value) = s0_value.overflowing_sub(s1_value);
        self.write_sop_dst(d, d_value as u32);
        self.ctx.scc = scc_value;
    }

    fn s_add_nc_u64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        let d_value = s0_value.wrapping_add(s1_value);
        self.write_sop_dst_pair(d, d_value);
    }

    fn s_and_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value & s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_or_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value | s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_xor_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value ^ s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_and_not1_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value & !s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_or_not1_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value | !s1_value;
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_cselect_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = if self.ctx.scc { s0_value } else { s1_value };
        self.write_sop_dst(d, d_value);
    }

    fn s_bfm_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = ((1 << (s0_value & 0x1F)) - 1) << (s1_value & 0x1F);
        self.write_sop_dst(d, d_value);
    }

    fn s_mul_u64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        let d_value = s0_value * s1_value;
        self.write_sop_dst_pair(d, d_value);
    }

    fn s_mul_i32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        let d_value = s0_value.wrapping_mul(s1_value);
        self.write_sop_dst(d, d_value as u32);
    }

    fn s_mul_hi_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = (((s0_value as u64) * (s1_value as u64)) >> 32) as u32;
        self.write_sop_dst(d, d_value);
    }

    fn s_lshr_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value >> (s1_value & 0x1F);
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_lshl_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value << (s1_value & 0x1F);
        self.write_sop_dst(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_lshl_b64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value << (s1_value & 0x3F);
        self.write_sop_dst_pair(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_max_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let scc_value = s0_value >= s1_value;
        let d_value = if scc_value { s0_value } else { s1_value };
        self.write_sop_dst(d, d_value);
        self.ctx.scc = scc_value;
    }

    fn s_add_co_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let (d_value, scc_value) = s0_value.overflowing_add(s1_value);
        self.write_sop_dst(d, d_value);
        self.ctx.scc = scc_value;
    }

    fn s_add_co_ci_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let (d_value, scc_value) = add_u32(s0_value, s1_value, self.ctx.scc as u32);
        self.write_sop_dst(d, d_value as u32);
        self.ctx.scc = scc_value;
    }

    fn execute_sopc(&mut self, inst: SOPC) -> Signals {
        let s0 = inst.ssrc0;
        let s1 = inst.ssrc1;

        match inst.op {
            I::S_CMP_LG_U32 => {
                self.s_cmp_lg_u32(s0, s1);
            }
            I::S_CMP_EQ_U32 => {
                self.s_cmp_eq_u32(s0, s1);
            }
            I::S_CMP_LT_U32 => {
                self.s_cmp_lt_u32(s0, s1);
            }
            I::S_CMP_GE_U32 => {
                self.s_cmp_ge_u32(s0, s1);
            }
            I::S_CMP_LT_I32 => {
                self.s_cmp_lt_i32(s0, s1);
            }
            I::S_CMP_LG_U64 => {
                self.s_cmp_lg_u64(s0, s1);
            }
            I::S_CMP_EQ_U64 => {
                self.s_cmp_eq_u64(s0, s1);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn s_cmp_lg_u32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        self.ctx.scc = s0_value != s1_value;
    }

    fn s_cmp_eq_u32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        self.ctx.scc = s0_value == s1_value;
    }

    fn s_cmp_lt_u32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        self.ctx.scc = s0_value < s1_value;
    }

    fn s_cmp_ge_u32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        self.ctx.scc = s0_value >= s1_value;
    }

    fn s_cmp_lt_i32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        self.ctx.scc = s0_value < s1_value;
    }

    fn s_cmp_lg_u64(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        self.ctx.scc = s0_value != s1_value;
    }

    fn s_cmp_eq_u64(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        self.ctx.scc = s0_value == s1_value;
    }

    fn execute_sopk(&mut self, inst: SOPK) -> Signals {
        let d = inst.sdst as usize;
        let simm16 = inst.simm16 as i16;

        match inst.op {
            I::S_MOVK_I32 => {
                self.write_sop_dst(d, simm16 as i32 as u32);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn execute_vop1(&mut self, inst: VOP1) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0;
        match inst.op {
            I::V_NOP => {}
            I::V_MOV_B32 => {
                self.v_mov_b32_e32(d, s0);
            }
            I::V_READFIRSTLANE_B32 => {
                self.v_readfirstlane_b32_e32(d, s0);
            }
            I::V_CVT_F64_U32 => {
                self.v_cvt_f64_u32_e32(d, s0);
            }
            I::V_RCP_IFLAG_F32 => {
                self.v_rcp_iflag_f32_e32(d, s0);
            }
            I::V_RCP_F32 => {
                self.v_rcp_f32_e32(d, s0);
            }
            I::V_SQRT_F32 => {
                self.v_sqrt_f32_e32(d, s0);
            }
            I::V_RNDNE_F32 => {
                self.v_rndne_f32_e32(d, s0);
            }
            I::V_RCP_F64 => {
                self.v_rcp_f64_e32(d, s0);
            }
            I::V_RSQ_F64 => {
                self.v_rsq_f64_e32(d, s0);
            }
            I::V_FRACT_F64 => {
                self.v_fract_f64_e32(d, s0);
            }
            I::V_CVT_I32_F64 => {
                self.v_cvt_i32_f64_e32(d, s0);
            }
            I::V_CVT_F64_I32 => {
                self.v_cvt_f64_i32_e32(d, s0);
            }
            I::V_CVT_F32_U32 => {
                self.v_cvt_f32_u32_e32(d, s0);
            }
            I::V_CVT_U32_F32 => {
                self.v_cvt_u32_f32_e32(d, s0);
            }
            I::V_CVT_I32_F32 => {
                self.v_cvt_i32_f32_e32(d, s0);
            }
            I::V_RNDNE_F64 => {
                self.v_rndne_f64_e32(d, s0);
            }
            I::V_FREXP_MANT_F32 => {
                self.v_frexp_mant_f32_e32(d, s0);
            }
            I::V_FREXP_EXP_I32_F32 => {
                self.v_frexp_exp_i32_f32_e32(d, s0);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn v_mov_b32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_readfirstlane_b32_e32(&mut self, d: usize, s0: SourceOperand) {
        let exec_value = self.read_sgpr(126);
        let lane = if exec_value == 0 {
            0
        } else {
            exec_value.trailing_zeros() as usize
        };
        let s0_value = self.read_vector_source_operand_u32(lane, s0);
        let d_value = s0_value;
        self.write_sgpr(d, d_value);
    }

    fn v_cvt_f64_u32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value as f64;

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_rcp_iflag_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = 1.0 / s0_value;

            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_rcp_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = 1.0 / s0_value;

            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_sqrt_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = s0_value.sqrt();

            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_rndne_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let mut d_value = (s0_value + 0.5).floor();
            if s0_value.floor() % 2.0 == 0.0 && s0_value.fract() == 0.5 {
                d_value -= 1.0;
            }

            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_rcp_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = 1.0 / s0_value;

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_rsq_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = 1.0 / s0_value.sqrt();

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_fract_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = s0_value.fract();

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_cvt_i32_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = s0_value as i32;

            self.write_vgpr(elem, d, d_value as u32);
        }
    }
    fn v_cvt_f64_i32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let d_value = s0_value as f64;

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_cvt_f32_u32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value as f32;

            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_cvt_u32_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = s0_value as u32;

            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cvt_i32_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = s0_value as i32;

            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_rndne_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let mut d_value = (s0_value + 0.5).floor();
            if s0_value.floor() % 2.0 == 0.0 && s0_value.fract() == 0.5 {
                d_value -= 1.0;
            }

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_frexp_mant_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = if s0_value.is_nan() || s0_value.is_infinite() {
                s0_value
            } else {
                libm::frexpf(s0_value).0
            };

            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_frexp_exp_i32_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = if s0_value.is_nan() || s0_value.is_infinite() {
                0
            } else {
                libm::frexpf(s0_value).1
            };

            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn execute_vop2(&mut self, inst: VOP2) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0;
        let s1 = inst.vsrc1 as usize;
        match inst.op {
            I::V_AND_B32 => {
                self.v_and_b32_e32(d, s0, s1);
            }
            I::V_OR_B32 => {
                self.v_or_b32_e32(d, s0, s1);
            }
            I::V_XOR_B32 => {
                self.v_xor_b32_e32(d, s0, s1);
            }
            I::V_CNDMASK_B32 => {
                self.v_cndmask_b32_e32(d, s0, s1);
            }
            I::V_ADD_NC_U32 => {
                self.v_add_nc_u32_e32(d, s0, s1);
            }
            I::V_SUB_NC_U32 => {
                self.v_sub_nc_u32_e32(d, s0, s1);
            }
            I::V_SUBREV_NC_U32 => {
                self.v_subrev_nc_u32_e32(d, s0, s1);
            }
            I::V_ADD_CO_CI_U32 => {
                self.v_add_co_ci_u32_e32(d, s0, s1);
            }
            I::V_MUL_U32_U24 => {
                self.v_mul_u32_u24_e32(d, s0, s1);
            }
            I::V_ADD_F32 => {
                self.v_add_f32_e32(d, s0, s1);
            }
            I::V_SUB_F32 => {
                self.v_sub_f32_e32(d, s0, s1);
            }
            I::V_MUL_F32 => {
                self.v_mul_f32_e32(d, s0, s1);
            }
            I::V_MUL_F64 => {
                self.v_mul_f64_e32(d, s0, s1);
            }
            I::V_ADD_F64 => {
                self.v_add_f64_e32(d, s0, s1);
            }
            I::V_MAX_NUM_F64 => {
                self.v_max_num_f64_e32(d, s0, s1);
            }
            I::V_LSHLREV_B32 => {
                self.v_lshlrev_b32_e32(d, s0, s1);
            }
            I::V_LSHRREV_B32 => {
                self.v_lshrrev_b32_e32(d, s0, s1);
            }
            I::V_FMAMK_F32 => {
                self.v_fmamk_f32(d, s0, s1, inst.literal_constant.unwrap());
            }
            I::V_FMAAK_F32 => {
                self.v_fmaak_f32(d, s0, s1, inst.literal_constant.unwrap());
            }
            I::V_FMAC_F32 => {
                self.v_fmac_f32_e32(d, s0, s1);
            }
            I::V_LSHLREV_B64 => {
                self.v_lshlrev_b64_e32(d, s0, s1);
            }
            _ => unimplemented!("{:?}", inst),
        }
        Signals::None
    }

    fn v_and_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value & s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_or_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value | s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_xor_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value ^ s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cndmask_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = if self.get_vcc_bit(elem) {
                s1_value
            } else {
                s0_value
            };
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add_nc_u32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value.wrapping_add(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_sub_nc_u32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value.wrapping_sub(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_subrev_nc_u32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s1_value.wrapping_sub(s0_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add_co_ci_u32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let carry = self.get_vcc_bit(elem);
            let (d_value, carry) = add_u32(s0_value, s1_value, carry as u32);
            self.write_vgpr(elem, d, d_value);
            vcc |= (carry as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_mul_u32_u24_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let s0_value = s0_value & 0xFFFFFF;
            let s1_value = s1_value & 0xFFFFFF;
            let d_value = s0_value.wrapping_mul(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add_f32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value + s1_value;
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_sub_f32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value - s1_value;
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_mul_f32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value * s1_value;
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_mul_f64_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value * s1_value;
            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_add_f64_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value + s1_value;
            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_max_num_f64_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value.max(s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_lshlrev_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s1_value << (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshrrev_b32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s1_value >> (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_fmamk_f32(&mut self, d: usize, s0: SourceOperand, s1: usize, literal_constant: u32) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let literal_value = u32_to_f32(literal_constant);
            let d_value = fma(s0_value, literal_value, s1_value);
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_fmaak_f32(&mut self, d: usize, s0: SourceOperand, s1: usize, literal_constant: u32) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let literal_value = u32_to_f32(literal_constant);
            let d_value = fma(s0_value, s1_value, literal_value);
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_fmac_f32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = u32_to_f32(self.read_vgpr(elem, d));
            let d_value = fma(s0_value, s1_value, d_value);
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_lshlrev_b64_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s1_value << (s0_value & 0x3F);
            self.write_vgpr_pair(elem, d, d_value);
        }
    }

    fn execute_vop3(&mut self, inst: VOP3) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0;
        let s1 = inst.src1;
        let s2 = inst.src2;
        let abs = inst.abs;
        let neg = inst.neg;
        let clamp = inst.cm != 0;
        let omod = inst.omod;
        match inst.op {
            I::V_ADD_NC_U16 => {
                self.v_add_nc_u16(d, s0, s1);
            }
            I::V_LSHLREV_B16 => {
                self.v_lshlrev_b16_e64(d, s0, s1);
            }
            I::V_READLANE_B32 => {
                self.v_readlane_b32(d, s0, s1);
            }
            I::V_WRITELANE_B32 => {
                self.v_writelane_b32(d, s0, s1);
            }
            I::V_AND_B32 => {
                self.v_and_b32_e64(d, s0, s1);
            }
            I::V_LSHL_OR_B32 => {
                self.v_lshl_or_b32(d, s0, s1, s2);
            }
            I::V_AND_OR_B32 => {
                self.v_and_or_b32(d, s0, s1, s2);
            }
            I::V_BFE_U32 => {
                self.v_bfe_u32(d, s0, s1, s2);
            }
            I::V_MAX_U32 => {
                self.v_max_u32_e64(d, s0, s1);
            }
            I::V_MIN_U32 => {
                self.v_min_u32_e64(d, s0, s1);
            }
            I::V_ASHRREV_I32 => {
                self.v_ashrrev_i32_e64(d, s0, s1);
            }
            I::V_CMP_EQ_U16 => {
                self.v_cmp_eq_u16_e64(d, s0, s1);
            }
            I::V_CMP_GT_U16 => {
                self.v_cmp_gt_u16_e64(d, s0, s1);
            }
            I::V_CMP_LT_U32 => {
                self.v_cmp_lt_u32_e64(d, s0, s1);
            }
            I::V_MUL_LO_U32 => {
                self.v_mul_lo_u32(d, s0, s1);
            }
            I::V_XOR_B32 => {
                self.v_xor_b32_e64(d, s0, s1);
            }
            I::V_OR3_B32 => {
                self.v_or3_b32(d, s0, s1, s2);
            }
            I::V_XOR3_B32 => {
                self.v_xor3_b32(d, s0, s1, s2);
            }
            I::V_ADD_NC_U32 => {
                self.v_add_nc_u32_e64(d, s0, s1);
            }
            I::V_ADD3_U32 => {
                self.v_add3_u32(d, s0, s1, s2);
            }
            I::V_MAD_U32_U24 => {
                self.v_mad_u32_u24(d, s0, s1, s2);
            }
            I::V_MUL_F32 => {
                self.v_mul_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_FMA_F32 => {
                self.v_fma_f32(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_DIV_FMAS_F32 => {
                self.v_div_fmas_f32(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_DIV_FIXUP_F32 => {
                self.v_div_fixup_f32(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_CMP_GE_F32 => {
                self.v_cmp_ge_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_LT_F32 => {
                self.v_cmp_lt_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_LE_F32 => {
                self.v_cmp_le_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_GT_F32 => {
                self.v_cmp_gt_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_LG_F32 => {
                self.v_cmp_lg_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_CLASS_F32 => {
                self.v_cmp_class_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CVT_F64_U32 => {
                self.v_cvt_f64_u32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_I32_F64 => {
                self.v_cvt_i32_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_ADD_F64 => {
                self.v_add_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_MUL_F64 => {
                self.v_mul_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_RCP_F64 => {
                self.v_rcp_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_RNDNE_F64 => {
                self.v_rndne_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_FMA_F64 => {
                self.v_fma_f64(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_DIV_FMAS_F64 => {
                self.v_div_fmas_f64(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_DIV_FIXUP_F64 => {
                self.v_div_fixup_f64(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_MIN_NUM_F64 => {
                self.v_min_num_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_MAX_NUM_F64 => {
                self.v_max_num_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_LT_F64 => {
                self.v_cmp_lt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NGT_F64 => {
                self.v_cmp_ngt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_GT_F64 => {
                self.v_cmp_gt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NLT_F64 => {
                self.v_cmp_nlt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_LG_F64 => {
                self.v_cmp_lg_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NGE_F64 => {
                self.v_cmp_nge_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_LE_F64 => {
                self.v_cmp_le_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NEQ_F64 => {
                self.v_cmp_neq_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CNDMASK_B32 => {
                self.v_cndmask_b32_e64(d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_LSHLREV_B32 => {
                self.v_lshlrev_b32_e64(d, s0, s1);
            }
            I::V_LSHRREV_B32 => {
                self.v_lshrrev_b32_e64(d, s0, s1);
            }
            I::V_LSHLREV_B64 => {
                self.v_lshlrev_b64_e64(d, s0, s1);
            }
            I::V_LSHRREV_B64 => {
                self.v_lshrrev_b64(d, s0, s1);
            }
            I::V_OR_B32 => {
                self.v_or_b32_e64(d, s0, s1);
            }
            I::V_LDEXP_F64 => {
                self.v_ldexp_f64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_RSQ_F64 => {
                self.v_rsq_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CMP_CLASS_F64 => {
                self.v_cmp_class_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_XAD_U32 => {
                self.v_xad_u32(d, s0, s1, s2);
            }
            I::V_LSHL_ADD_U32 => {
                self.v_lshl_add_u32(d, s0, s1, s2);
            }
            I::V_ADD_LSHL_U32 => {
                self.v_add_lshl_u32(d, s0, s1, s2);
            }
            I::V_CMP_NE_U32 => {
                self.v_cmp_ne_u32_e64(d, s0, s1);
            }
            I::V_CMP_EQ_U32 => {
                self.v_cmp_eq_u32_e64(d, s0, s1);
            }
            I::V_CMP_GT_U32 => {
                self.v_cmp_gt_u32_e64(d, s0, s1);
            }
            I::V_CMPX_NE_U32 => {
                self.v_cmpx_ne_u32_e64(d, s0, s1);
            }
            I::V_CMP_GT_I32 => {
                self.v_cmp_gt_i32_e64(d, s0, s1);
            }
            I::V_CMP_LT_U64 => {
                self.v_cmp_lt_u64_e64(d, s0, s1);
            }
            I::V_CMP_EQ_U64 => {
                self.v_cmp_eq_u64_e64(d, s0, s1);
            }
            I::V_TRIG_PREOP_F64 => {
                self.v_trig_preop_f64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CVT_F32_F16 => {
                self.v_cvt_f32_f16_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_LDEXP_F32 => {
                self.v_ldexp_f32(d, s0, s1, abs, neg, clamp, omod);
            }
            _ => unimplemented!("{:?}", inst),
        }
        Signals::None
    }

    fn v_add_nc_u16(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as u16;
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as u16;
            let d_value = s0_value.wrapping_add(s1_value);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_lshlrev_b16_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as u16;
            let d_value = s1_value << (s0_value & 0xF);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_readlane_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s1_value = (self.read_scalar_source_operand_u32(s1) as usize) & 0x1F;
        let s0_value = self.read_vector_source_operand_u32(s1_value, s0);
        let d_value = s0_value;
        self.write_sgpr(d, d_value);
    }

    fn v_writelane_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s1_value = (self.read_scalar_source_operand_u32(s1) as usize) & 0x1F;
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let d_value = s0_value;
        self.write_vgpr(s1_value, d, d_value);
    }

    fn v_and_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value & s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshl_or_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value << (s1_value & 0x1F)) | s2_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_and_or_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value & s1_value) | s2_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_bfe_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value >> (s1_value & 0x1F)) & ((1 << (s2_value & 0x1F)) - 1);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_max_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value.max(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_min_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value.min(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_ashrrev_i32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as i32;
            let d_value = s1_value >> (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_cmp_eq_u16_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as u16;
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as u16;
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_u16_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as u16;
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as u16;
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_mul_lo_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = mul_u32(s0_value, s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_xor_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value ^ s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_or3_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value | s1_value) | s2_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_xor3_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value ^ s1_value) ^ s2_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add_nc_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value.wrapping_add(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add3_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let (d_value, _) = add_u32(s0_value, s1_value, s2_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_mad_u32_u24(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);

            let s0_value = s0_value & 0xFFFFFF;
            let s1_value = s1_value & 0xFFFFFF;
            let d_value = s0_value.wrapping_mul(s1_value).wrapping_add(s2_value);

            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_mul_f32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let d_value = s0_value * s1_value;
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_fma_f32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f32(elem, s2), abs, neg, 2);
            let d_value = fma(s0_value, s1_value, s2_value);
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_div_fmas_f32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f32(elem, s2), abs, neg, 2);
            let d_value = if self.get_vcc_bit(elem) {
                32f32.exp2() * fma(s0_value, s1_value, s2_value)
            } else {
                fma(s0_value, s1_value, s2_value)
            };
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_div_fixup_f32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f32(elem, s2), abs, neg, 2);
            let d_value = div_fixup_f32(s0_value, s1_value, s2_value);
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cmp_ge_f32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_f32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_le_f32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_f32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lg_f32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_class_f32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let s0_values = (0..32)
            .map(|elem| abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0))
            .collect::<Vec<f32>>();
        let s1_values = (0..32)
            .map(|elem| self.read_vector_source_operand_u32(elem, s1))
            .collect::<Vec<u32>>();

        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = s0_values[elem];
            let s1_value = s1_values[elem];
            let d_value = cmp_class_f32(s0_value, s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cvt_f64_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        _abs: u8,
        _neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value as f64;

            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_i32_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let d_value = s0_value as i32;

            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_add_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value + s1_value;
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_mul_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value * s1_value;
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_rcp_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let d_value = 1.0 / s0_value;
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_rndne_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let mut d_value = (s0_value + 0.5).floor();
            if s0_value.floor() % 2.0 == 0.0 && s0_value.fract() == 0.5 {
                d_value -= 1.0;
            }
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_fma_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f64(elem, s2), abs, neg, 2);
            let d_value = fma(s0_value, s1_value, s2_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_div_fmas_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f64(elem, s2), abs, neg, 2);
            let d_value = if self.get_vcc_bit(elem) {
                64f64.exp2() * fma(s0_value, s1_value, s2_value)
            } else {
                fma(s0_value, s1_value, s2_value)
            };
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_div_fixup_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f64(elem, s2), abs, neg, 2);
            let d_value = div_fixup_f64(s0_value, s1_value, s2_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_min_num_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value.min(s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_max_num_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value.max(s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cmp_lt_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ngt_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = !(s0_value > s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nlt_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = !(s0_value < s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lg_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nge_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = !(s0_value >= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_le_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_neq_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let d_value = !(s0_value == s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cndmask_b32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value =
                u32_to_f32_abs_neg(self.read_vector_source_operand_u32(elem, s0), abs, neg, 0);
            let s1_value =
                u32_to_f32_abs_neg(self.read_vector_source_operand_u32(elem, s1), abs, neg, 1);
            let s2_value = self.read_scalar_source_operand_u32(s2);
            let d_value = if (s2_value >> elem) & 1 != 0 {
                s1_value
            } else {
                s0_value
            };
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_lshlrev_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s1_value << (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshrrev_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s1_value >> (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshlrev_b64_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u64(elem, s1);
            let d_value = s1_value << (s0_value & 0x3F);
            self.write_vgpr_pair(elem, d, d_value);
        }
    }

    fn v_lshrrev_b64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u64(elem, s1);
            let d_value = s1_value >> (s0_value & 0x3F);
            self.write_vgpr_pair(elem, d, d_value);
        }
    }

    fn v_or_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value | s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_ldexp_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as i32;
            let d_value = libm::ldexp(s0_value, s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_rsq_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        _abs: u8,
        _neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = 1.0 / s0_value.sqrt();

            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_cmp_class_f64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        _clamp: bool,
        _omod: u8,
    ) {
        let s0_values = (0..32)
            .map(|elem| abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0))
            .collect::<Vec<f64>>();
        let s1_values = (0..32)
            .map(|elem| self.read_vector_source_operand_u32(elem, s1))
            .collect::<Vec<u32>>();

        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = s0_values[elem];
            let s1_value = s1_values[elem];
            let d_value = cmp_class_f64(s0_value, s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_xad_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, s2: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value ^ s1_value).wrapping_add(s2_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshl_add_u32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = (s0_value << (s1_value & 0x1F)).wrapping_add(s2_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add_lshl_u32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let d_value = s0_value.wrapping_add(s1_value) << (s2_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cmp_ne_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_eq_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ne_u32_e64(&mut self, _d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_i32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as i32;
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_u64_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vector_source_operand_u64(elem, s1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_eq_u64_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vector_source_operand_u64(elem, s1);
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_trig_preop_f64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        const TWO_OVER_PI_FRACTION: [u64; 20] = [
            0xBA10AC06608DF8F6,
            0x25D4D7F6BF623F1A,
            0xE2F67A0E73EF14A5,
            0xD45AEA4F758FD7CB,
            0x136E9E8C7ECD3CBF,
            0xDA3EDA6CFD9E4F96,
            0x301FDE5E2316B414,
            0x50763FF12FFFBC0B,
            0x73E93908BF177BF2,
            0xFC827323AC7306A6,
            0x8909D338E04D68BE,
            0x4E7DD1046BEA5D76,
            0x2439FC3BD6396253,
            0xA5C00C925DD413A3,
            0x8AC36E48DC74849B,
            0x2083FCA2C757BD77,
            0xBB81B6C52B327887,
            0x2A53F84EAFA3EA69,
            0x000145F306DC9C88,
            0x0000000000000000,
        ];

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);

            let mut shift = (s1_value & 0x1F) as i32 * 53;
            if get_exp_f64(s0_value) > 1077 {
                shift += get_exp_f64(s0_value) as i32 - 1077;
            }

            let result = get_bits_u64(&TWO_OVER_PI_FRACTION, (1201 - 53 - shift) as usize, 53);
            let mut scale = -53 - shift;

            if get_exp_f64(s0_value) >= 1968 {
                scale += 128;
            }

            let d_value = libm::ldexp(result as f64, scale);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_f32_f16_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_f16(self.read_vector_source_operand_f16(elem, s0), abs, neg, 0);
            let d_value = s0_value.to_f32();
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_ldexp_f32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as i32;
            let d_value = libm::ldexpf(s0_value, s1_value);
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn execute_vop3sd(&mut self, inst: VOP3SD) -> Signals {
        let d0 = inst.vdst as usize;
        let d1 = inst.sdst as usize;
        let s0 = inst.src0;
        let s1 = inst.src1;
        let s2 = inst.src2;
        let neg = inst.neg;
        let clamp = inst.cm != 0;
        let omod = inst.omod;
        match inst.op {
            I::V_ADD_CO_U32 => {
                self.v_add_co_u32(d0, d1, s0, s1);
            }
            I::V_ADD_CO_CI_U32 => {
                self.v_add_co_ci_u32_e64(d0, d1, s0, s1, s2);
            }
            I::V_MAD_CO_U64_U32 => {
                self.v_mad_co_u64_u32(d0, d1, s0, s1, s2);
            }
            I::V_DIV_SCALE_F32 => {
                self.v_div_scale_f32(d0, d1, s0, s1, s2, 0, neg, clamp, omod);
            }
            I::V_DIV_SCALE_F64 => {
                self.v_div_scale_f64(d0, d1, s0, s1, s2, 0, neg, clamp, omod);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn v_add_co_u32(&mut self, d0: usize, d1: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (d0_value, d1_value) = add_u32(s0_value, s1_value, 0);
            self.write_vgpr(elem, d0, d0_value);
            vcc |= (d1_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d1, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_add_co_ci_u32_e64(
        &mut self,
        d0: usize,
        d1: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_scalar_source_operand_u32(s2);
            let (d0_value, d1_value) = add_u32(s0_value, s1_value, ((s2_value >> elem) & 1) as u32);
            self.write_vgpr(elem, d0, d0_value);
            vcc |= (d1_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d1, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_mad_co_u64_u32(
        &mut self,
        d0: usize,
        d1: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as u64;
            let s1_value = self.read_vector_source_operand_u32(elem, s1) as u64;
            let s2_value = self.read_vector_source_operand_u64(elem, s2);
            let (d0_value, d1_value) = (s0_value * s1_value).overflowing_add(s2_value);
            self.write_vgpr_pair(elem, d0, d0_value);
            vcc |= (d1_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d1, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_div_scale_f32(
        &mut self,
        d: usize,
        sdst: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f32(elem, s2), abs, neg, 2);
            let (d_value, flag) = div_scale_f32(s0_value, s1_value, s2_value);

            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
            vcc |= (flag as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(sdst, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_div_scale_f64(
        &mut self,
        d: usize,
        sdst: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f64(elem, s1), abs, neg, 1);
            let s2_value = abs_neg(self.read_vector_source_operand_f64(elem, s2), abs, neg, 2);
            let (d_value, flag) = div_scale_f64(s0_value, s1_value, s2_value);

            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
            vcc |= (flag as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(sdst, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn execute_vop3p(&mut self, inst: VOP3P) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0;
        let s1 = inst.src1;
        let s2 = inst.src2;
        let opsel = inst.opsel;
        let opsel_hi = inst.opsel_hi | (inst.opsel_hi2 << 2);
        let neg = inst.neg;
        let neg_hi = inst.neg_hi;
        let clamp = inst.cm != 0;
        match inst.op {
            I::V_WMMA_F32_16X16X16_F16 => {
                self.v_wmma_f32_16x16x16_f16(d, s0, s1, s2);
            }
            I::V_FMA_MIXLO_F16 => {
                self.v_fma_mixlo_f16(d, s0, s1, s2, neg_hi, neg, clamp, opsel, opsel_hi);
            }
            _ => unimplemented!("{:?}", inst.op),
        }
        Signals::None
    }

    fn v_wmma_f32_16x16x16_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
    ) {
        let mut matrix_a = [[0f32; 16]; 16];
        let mut matrix_b = [[0f32; 16]; 16];
        let mut matrix_c = [[0f32; 16]; 16];
        let mut matrix_d = [[0f32; 16]; 16];

        for elem in 0..32 {
            let s0_value = self.read_vector_source_operand_f16_vec::<8>(elem, s0);
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        let col = (k + j * 2 + (elem / 16) * 4 + i * 8) as usize;
                        let row = (elem % 16) as usize;
                        matrix_a[row][col] = s0_value[k + j * 2 + i * 4].to_f32();
                    }
                }
            }
            let s1_value = self.read_vector_source_operand_f16_vec::<8>(elem, s1);
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        let row = (k + j * 2 + (elem / 16) * 4 + i * 8) as usize;
                        let col = (elem % 16) as usize;
                        matrix_b[row][col] = s1_value[k + j * 2 + i * 4].to_f32();
                    }
                }
            }
            let s2_value = self.read_vector_source_operand_f32_vec::<8>(elem, s2);
            for i in 0..8 {
                let row = ((elem / 16) * 8 + i) as usize;
                let col = (elem % 16) as usize;
                matrix_c[row][col] = s2_value[i];
            }
        }

        for i in 0..16 {
            for j in 0..16 {
                let mut sum = matrix_c[i][j];
                for k in 0..16 {
                    sum += matrix_a[i][k] * matrix_b[k][j];
                }
                matrix_d[i][j] = sum;
            }
        }

        for elem in 0..32 {
            for i in 0..8 {
                let row = ((elem / 16) * 8 + i) as usize;
                let col = (elem % 16) as usize;
                self.write_vgpr(elem, d + i, f32_to_u32(matrix_d[row][col]));
            }
        }
    }

    fn v_fma_mixlo_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = if opsel_hi & 1 == 0 {
                abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0)
            } else if opsel & 1 == 0 {
                abs_neg_f16(
                    self.read_vector_source_operand_f16_hi(elem, s0),
                    abs,
                    neg,
                    0,
                )
                .to_f32()
            } else {
                abs_neg_f16(self.read_vector_source_operand_f16(elem, s0), abs, neg, 0).to_f32()
            };

            let s1_value = if opsel_hi & 2 == 0 {
                abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1)
            } else if opsel & 2 == 0 {
                abs_neg_f16(
                    self.read_vector_source_operand_f16_hi(elem, s1),
                    abs,
                    neg,
                    1,
                )
                .to_f32()
            } else {
                abs_neg_f16(self.read_vector_source_operand_f16(elem, s1), abs, neg, 1).to_f32()
            };

            let s2_value = if opsel_hi & 4 == 0 {
                abs_neg(self.read_vector_source_operand_f32(elem, s2), abs, neg, 2)
            } else if opsel & 4 == 0 {
                abs_neg_f16(
                    self.read_vector_source_operand_f16_hi(elem, s2),
                    abs,
                    neg,
                    2,
                )
                .to_f32()
            } else {
                abs_neg_f16(self.read_vector_source_operand_f16(elem, s2), abs, neg, 2).to_f32()
            };

            let d_value = fma(s0_value, s1_value, s2_value);

            self.write_vgpr(
                elem,
                d,
                f16::from_f32(clamp_f32(d_value, clamp)).to_bits() as u32,
            );
        }
    }

    fn execute_vopc(&mut self, inst: VOPC) -> Signals {
        let s0 = inst.src0;
        let s1 = inst.vsrc1 as usize;
        match inst.op {
            I::V_CMP_GT_U32 => {
                self.v_cmp_gt_u32_e32(s0, s1);
            }
            I::V_CMP_NE_U32 => {
                self.v_cmp_ne_u32_e32(s0, s1);
            }
            I::V_CMP_EQ_U32 => {
                self.v_cmp_eq_u32_e32(s0, s1);
            }
            I::V_CMP_LT_U32 => {
                self.v_cmp_lt_u32_e32(s0, s1);
            }
            I::V_CMPX_LT_U32 => {
                self.v_cmpx_lt_u32_e32(s0, s1);
            }
            I::V_CMPX_GT_U32 => {
                self.v_cmpx_gt_u32_e32(s0, s1);
            }
            I::V_CMPX_NE_U32 => {
                self.v_cmpx_ne_u32_e32(s0, s1);
            }
            I::V_CMPX_EQ_U32 => {
                self.v_cmpx_eq_u32_e32(s0, s1);
            }
            I::V_CMPX_LT_I32 => {
                self.v_cmpx_lt_i32_e32(s0, s1);
            }
            I::V_CMP_GT_U64 => {
                self.v_cmp_gt_u64_e32(s0, s1);
            }
            I::V_CMP_EQ_U64 => {
                self.v_cmp_eq_u64_e32(s0, s1);
            }
            I::V_CMP_GE_F32 => {
                self.v_cmp_ge_f32_e32(s0, s1);
            }
            I::V_CMP_GT_F32 => {
                self.v_cmp_gt_f32_e32(s0, s1);
            }
            I::V_CMP_LE_F32 => {
                self.v_cmp_le_f32_e32(s0, s1);
            }
            I::V_CMP_LT_F32 => {
                self.v_cmp_lt_f32_e32(s0, s1);
            }
            I::V_CMP_NGT_F32 => {
                self.v_cmp_ngt_f32_e32(s0, s1);
            }
            I::V_CMP_GT_F64 => {
                self.v_cmp_gt_f64_e32(s0, s1);
            }
            I::V_CMP_NLT_F64 => {
                self.v_cmp_nlt_f64_e32(s0, s1);
            }
            I::V_CMP_LT_F64 => {
                self.v_cmp_lt_f64_e32(s0, s1);
            }
            I::V_CMP_LE_F64 => {
                self.v_cmp_le_f64_e32(s0, s1);
            }
            I::V_CMP_NGT_F64 => {
                self.v_cmp_ngt_f64_e32(s0, s1);
            }
            I::V_CMPX_NGT_F64 => {
                self.v_cmpx_ngt_f64_e32(s0, s1);
            }
            I::V_CMPX_NGE_F64 => {
                self.v_cmpx_nge_f64_e32(s0, s1);
            }
            I::V_CMPX_NLT_F64 => {
                self.v_cmpx_nlt_f64_e32(s0, s1);
            }
            _ => {
                unimplemented!()
            }
        }
        Signals::None
    }

    fn v_cmp_gt_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ne_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_eq_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lt_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_gt_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ne_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_eq_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lt_i32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let s1_value = self.read_vgpr(elem, s1) as i32;
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_eq_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ge_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_le_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ngt_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = !(s0_value > s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nlt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value < s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_le_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ngt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value > s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_vcc_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ngt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value > s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nge_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value >= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nlt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value < s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn execute_vopd(&mut self, inst: VOPD) -> Signals {
        let mut dual_result0_u32 = [0u32; 32];
        let mut dual_result1_u32 = [0u32; 32];

        let s0 = inst.src0x;
        let s1 = inst.vsrc1x as usize;
        let d = inst.vdstx as usize;
        match inst.opx {
            I::V_DUAL_CNDMASK_B32 => {
                self.v_dual_cndmask_b32(&mut dual_result0_u32, s0, s1);
            }
            I::V_DUAL_MOV_B32 => {
                self.v_dual_mov_b32(&mut dual_result0_u32, s0);
            }
            I::V_DUAL_FMAC_F32 => {
                self.v_dual_fmac_f32(&mut dual_result0_u32, s0, s1, d);
            }
            I::V_DUAL_MUL_F32 => {
                self.v_dual_mul_f32(&mut dual_result0_u32, s0, s1);
            }
            I::V_DUAL_ADD_F32 => {
                self.v_dual_add_f32(&mut dual_result0_u32, s0, s1);
            }
            I::V_DUAL_SUB_F32 => {
                self.v_dual_sub_f32(&mut dual_result0_u32, s0, s1);
            }
            I::V_DUAL_FMAMK_F32 => {
                self.v_dual_fmamk_f32(
                    &mut dual_result0_u32,
                    s0,
                    s1,
                    inst.literal_constant.unwrap(),
                );
            }
            I::V_DUAL_FMAAK_F32 => {
                self.v_dual_fmaak_f32(
                    &mut dual_result0_u32,
                    s0,
                    s1,
                    inst.literal_constant.unwrap(),
                );
            }
            _ => unimplemented!("{:?}", inst.opx),
        }
        let s0 = inst.src0y;
        let s1 = inst.vsrc1y as usize;
        let d = ((inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)) as usize;
        match inst.opy {
            I::V_DUAL_CNDMASK_B32 => {
                self.v_dual_cndmask_b32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_MOV_B32 => {
                self.v_dual_mov_b32(&mut dual_result1_u32, s0);
            }
            I::V_DUAL_LSHLREV_B32 => {
                self.v_dual_lshlrev_b32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_ADD_NC_U32 => {
                self.v_dual_add_nc_u32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_AND_B32 => {
                self.v_dual_and_b32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_MUL_F32 => {
                self.v_dual_mul_f32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_ADD_F32 => {
                self.v_dual_add_f32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_SUB_F32 => {
                self.v_dual_sub_f32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_FMAC_F32 => {
                self.v_dual_fmac_f32(&mut dual_result1_u32, s0, s1, d);
            }
            I::V_DUAL_FMAAK_F32 => {
                self.v_dual_fmaak_f32(
                    &mut dual_result1_u32,
                    s0,
                    s1,
                    inst.literal_constant.unwrap(),
                );
            }
            _ => unimplemented!("{:?}", inst.opy),
        }
        let d = inst.vdstx as usize;
        match inst.opx {
            I::V_DUAL_CNDMASK_B32
            | I::V_DUAL_MOV_B32
            | I::V_DUAL_FMAC_F32
            | I::V_DUAL_MUL_F32
            | I::V_DUAL_ADD_F32
            | I::V_DUAL_SUB_F32
            | I::V_DUAL_FMAMK_F32
            | I::V_DUAL_FMAAK_F32 => {
                for elem in 0..32 {
                    if !self.get_exec_bit(elem) {
                        continue;
                    }
                    self.write_vgpr(elem, d, dual_result0_u32[elem]);
                }
            }
            _ => unimplemented!(),
        }
        let d = ((inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)) as usize;
        match inst.opy {
            I::V_DUAL_CNDMASK_B32
            | I::V_DUAL_MOV_B32
            | I::V_DUAL_LSHLREV_B32
            | I::V_DUAL_ADD_NC_U32
            | I::V_DUAL_AND_B32
            | I::V_DUAL_MUL_F32
            | I::V_DUAL_ADD_F32
            | I::V_DUAL_SUB_F32
            | I::V_DUAL_FMAC_F32
            | I::V_DUAL_FMAAK_F32 => {
                for elem in 0..32 {
                    if !self.get_exec_bit(elem) {
                        continue;
                    }
                    self.write_vgpr(elem, d, dual_result1_u32[elem]);
                }
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn v_dual_cndmask_b32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = if self.get_vcc_bit(elem) {
                s1_value
            } else {
                s0_value
            };
            d_values[elem] = d_value;
        }
    }

    fn v_dual_mov_b32(&mut self, d_values: &mut [u32], s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value;
            d_values[elem] = d_value;
        }
    }

    fn v_dual_fmac_f32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize, d: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = u32_to_f32(self.read_vgpr(elem, d));
            let d_value = fma(s0_value, s1_value, d_value);
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    fn v_dual_lshlrev_b32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s1_value << (s0_value & 0x1F);
            d_values[elem] = d_value;
        }
    }

    fn v_dual_add_nc_u32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value.wrapping_add(s1_value);
            d_values[elem] = d_value;
        }
    }

    fn v_dual_and_b32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value & s1_value;
            d_values[elem] = d_value;
        }
    }

    fn v_dual_mul_f32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value * s1_value;
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    fn v_dual_add_f32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value + s1_value;
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    fn v_dual_sub_f32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value - s1_value;
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    fn v_dual_fmamk_f32(
        &mut self,
        d_values: &mut [u32],
        s0: SourceOperand,
        s1: usize,
        literal_constant: u32,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let simm32_value = u32_to_f32(literal_constant);
            let d_value = fma(s0_value, simm32_value, s1_value);
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    fn v_dual_fmaak_f32(
        &mut self,
        d_values: &mut [u32],
        s0: SourceOperand,
        s1: usize,
        literal_constant: u32,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let simm32_value = u32_to_f32(literal_constant);
            let d_value = fma(s0_value, s1_value, simm32_value);
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    fn execute_smem(&mut self, inst: SMEM) -> Signals {
        let sdata = inst.sdata as usize;
        let ioffset = inst.ioffset as u64;
        let sbase = (inst.sbase * 2) as usize;
        match inst.op {
            I::S_LOAD_B32 => {
                self.s_load_b32(sdata, sbase, ioffset);
            }
            I::S_LOAD_B64 => {
                self.s_load_b64(sdata, sbase, ioffset);
            }
            I::S_LOAD_B128 => {
                self.s_load_b128(sdata, sbase, ioffset);
            }
            I::S_LOAD_B256 => {
                self.s_load_b256(sdata, sbase, ioffset);
            }
            I::S_LOAD_B512 => {
                self.s_load_b512(sdata, sbase, ioffset);
            }
            I::S_LOAD_B96 => {
                self.s_load_b96(sdata, sbase, ioffset);
            }
            I::S_LOAD_I8 => {
                self.s_load_i8(sdata, sbase, ioffset);
            }
            I::S_LOAD_U8 => {
                self.s_load_u8(sdata, sbase, ioffset);
            }
            I::S_LOAD_I16 => {
                self.s_load_i16(sdata, sbase, ioffset);
            }
            I::S_LOAD_U16 => {
                self.s_load_u16(sdata, sbase, ioffset);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn s_load_b32(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const u32;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, data);
    }

    fn s_load_b64(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..2 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_b128(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..4 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_b256(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..8 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_b512(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..16 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_b96(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        for i in 0..3 {
            let ptr = (sbase_val + ioffset + ((i * 4) as u64)) as *const u32;
            let data = unsafe { *ptr };
            self.write_sgpr(sdata + i, data);
        }
    }

    fn s_load_i8(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const i8;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, (data as i32) as u32);
    }

    fn s_load_u8(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const u8;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, data as u32);
    }

    fn s_load_i16(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const i16;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, (data as i32) as u32);
    }

    fn s_load_u16(&mut self, sdata: usize, sbase: usize, ioffset: u64) {
        let sbase_val = self.read_sgpr_pair(sbase);
        let ptr = (sbase_val + ioffset) as *const u16;
        let data = unsafe { *ptr };
        self.write_sgpr(sdata, data as u32);
    }

    fn execute_vflat(&mut self, inst: VFLAT) -> Signals {
        let vaddr = inst.vaddr as usize;
        let vsrc = inst.vsrc as usize;
        let vdst = inst.vdst as usize;
        let ioffset = inst.ioffset as u32;
        match inst.op {
            I::FLAT_LOAD_B32 => {
                self.flat_load_b32(vaddr, vdst, ioffset);
            }
            I::FLAT_LOAD_B64 => {
                self.flat_load_b64(vaddr, vdst, ioffset);
            }
            I::FLAT_LOAD_B128 => {
                self.flat_load_b128(vaddr, vdst, ioffset);
            }
            I::FLAT_STORE_B32 => {
                self.flat_store_b32(vaddr, vsrc, ioffset);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn flat_load_b32(&mut self, vaddr: usize, vdst: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| self.read_vgpr_pair(elem, vaddr))
            .collect::<Vec<u64>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let ioffset = ((ioffset << 8) as i32) >> 8;

            let scratch_base = self.ctx.scratch.borrow().as_ptr() as u64;
            let scratch_limit = scratch_base + self.ctx.scratch.borrow().len() as u64 / 32;
            let addr = if ((offset[elem] as u64) < scratch_base)
                || ((offset[elem] as u64) >= scratch_limit)
            {
                offset[elem] as i64 + (ioffset as i64)
            } else {
                let lane_addr = offset[elem] as i64 + (ioffset as i64) - scratch_base as i64;
                scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
            };
            let ptr = addr as *mut u32;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, data);
        }
    }

    fn flat_load_b64(&mut self, vaddr: usize, vdst: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| self.read_vgpr_pair(elem, vaddr))
            .collect::<Vec<u64>>();

        for i in 0..2 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let ioffset = ((ioffset << 8) as i32) >> 8;
                let scratch_base = self.ctx.scratch.borrow().as_ptr() as u64;
                let scratch_limit = scratch_base + self.ctx.scratch.borrow().len() as u64 / 32;
                let addr = if ((offset[elem] as u64) < scratch_base)
                    || ((offset[elem] as u64) >= scratch_limit)
                {
                    offset[elem] as i64 + (ioffset as i64) + (i as i64 * 4)
                } else {
                    let lane_addr =
                        offset[elem] as i64 + (ioffset as i64) + i as i64 * 4 - scratch_base as i64;
                    scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
                };
                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn flat_load_b128(&mut self, vaddr: usize, vdst: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| self.read_vgpr_pair(elem, vaddr))
            .collect::<Vec<u64>>();

        for i in 0..4 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let ioffset = ((ioffset << 8) as i32) >> 8;
                let scratch_base = self.ctx.scratch.borrow().as_ptr() as u64;
                let scratch_limit = scratch_base + self.ctx.scratch.borrow().len() as u64 / 32;
                let addr = if ((offset[elem] as u64) < scratch_base)
                    || ((offset[elem] as u64) >= scratch_limit)
                {
                    offset[elem] as i64 + (ioffset as i64) + (i as i64 * 4)
                } else {
                    let lane_addr =
                        offset[elem] as i64 + (ioffset as i64) + i as i64 * 4 - scratch_base as i64;
                    scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
                };
                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn flat_store_b32(&mut self, vaddr: usize, vsrc: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| self.read_vgpr_pair(elem, vaddr))
            .collect::<Vec<u64>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let scratch_base = self.ctx.scratch.borrow_mut().as_ptr() as u64;
            let scratch_limit = scratch_base + self.ctx.scratch.borrow().len() as u64 / 32;
            let addr = if ((offset[elem] as u64) < scratch_base)
                || ((offset[elem] as u64) >= scratch_limit)
            {
                offset[elem] as i64 + (ioffset as i64)
            } else {
                let lane_addr = offset[elem] as i64 + (ioffset as i64) - scratch_base as i64;
                scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
            };
            let data = self.read_vgpr(elem, vsrc);
            let ptr = addr as *mut u32;
            unsafe {
                *ptr = data;
            }
        }
    }

    fn execute_vscratch(&mut self, inst: VSCRATCH) -> Signals {
        let saddr = inst.saddr as usize;
        let vaddr = inst.vaddr as usize;
        let vsrc = inst.vsrc as usize;
        let vdst = inst.vdst as usize;
        let ioffset = inst.ioffset as u32;
        match inst.op {
            I::SCRATCH_STORE_B32 => {
                self.scratch_store_b32(vaddr, vsrc, saddr, ioffset);
            }
            I::SCRATCH_STORE_B64 => {
                self.scratch_store_b64(vaddr, vsrc, saddr, ioffset);
            }
            I::SCRATCH_STORE_B96 => {
                self.scratch_store_b96(vaddr, vsrc, saddr, ioffset);
            }
            I::SCRATCH_STORE_B128 => {
                self.scratch_store_b128(vaddr, vsrc, saddr, ioffset);
            }
            I::SCRATCH_LOAD_B32 => {
                self.scratch_load_b32(vaddr, vdst, saddr, ioffset);
            }
            I::SCRATCH_LOAD_B64 => {
                self.scratch_load_b64(vaddr, vdst, saddr, ioffset);
            }
            I::SCRATCH_LOAD_B96 => {
                self.scratch_load_b96(vaddr, vdst, saddr, ioffset);
            }
            I::SCRATCH_LOAD_B128 => {
                self.scratch_load_b128(vaddr, vdst, saddr, ioffset);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn scratch_store_b32(&mut self, _vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let data = self.read_vgpr(elem, vsrc);
            let vaddr_val = 0u64;
            let saddr_val = self.read_sgpr(saddr) as u64;
            let offset = (vaddr_val + saddr_val + (ioffset as u64)) * 32 + (elem as u64) * 4;
            let addr = self.ctx.scratch.borrow_mut().as_mut_ptr() as u64 + offset;
            assert!(offset < self.ctx.scratch.borrow().len() as u64);

            let ptr = addr as *mut u32;
            unsafe {
                *ptr = data;
            }
        }
    }

    fn scratch_store_b64(&mut self, _vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        for i in 0..2 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let data = self.read_vgpr(elem, vsrc + i);
                let vaddr_val = 0u64;
                let saddr_val = self.read_sgpr(saddr) as u64;
                let offset = (vaddr_val + saddr_val + (ioffset as u64) + (i as u64 * 4)) * 32
                    + (elem as u64) * 4;
                let addr = self.ctx.scratch.borrow_mut().as_mut_ptr() as u64 + offset;
                assert!(offset < self.ctx.scratch.borrow().len() as u64);

                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn scratch_store_b96(&mut self, _vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        for i in 0..3 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let data = self.read_vgpr(elem, vsrc + i);
                let vaddr_val = 0u64;
                let saddr_val = self.read_sgpr(saddr) as u64;
                let offset = (vaddr_val + saddr_val + (ioffset as u64) + (i as u64 * 4)) * 32
                    + (elem as u64) * 4;
                let addr = self.ctx.scratch.borrow_mut().as_mut_ptr() as u64 + offset;
                assert!(offset < self.ctx.scratch.borrow().len() as u64);

                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn scratch_store_b128(&mut self, _vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        for i in 0..4 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let data = self.read_vgpr(elem, vsrc + i);
                let vaddr_val = 0u64;
                let saddr_val = self.read_sgpr(saddr) as u64;
                let offset = (vaddr_val + saddr_val + (ioffset as u64) + (i as u64 * 4)) * 32
                    + (elem as u64) * 4;
                let addr = self.ctx.scratch.borrow_mut().as_mut_ptr() as u64 + offset;
                assert!(offset < self.ctx.scratch.borrow().len() as u64);

                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn scratch_load_b32(&mut self, _vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let vaddr_val = 0u64;
            let saddr_val = self.read_sgpr(saddr) as u64;
            let offset = (vaddr_val + saddr_val + (ioffset as u64)) * 32 + (elem as u64) * 4;
            let addr = self.ctx.scratch.borrow().as_ptr() as u64 + offset;
            assert!(offset < self.ctx.scratch.borrow().len() as u64);

            let ptr = addr as *mut u32;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, data);
        }
    }

    fn scratch_load_b64(&mut self, _vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        for i in 0..2 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let vaddr_val = 0u64;
                let saddr_val = self.read_sgpr(saddr) as u64;
                let offset = (vaddr_val + saddr_val + (ioffset as u64) + (i as u64 * 4)) * 32
                    + (elem as u64) * 4;
                let addr = self.ctx.scratch.borrow().as_ptr() as u64 + offset;
                assert!(offset < self.ctx.scratch.borrow().len() as u64);

                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn scratch_load_b96(&mut self, _vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        for i in 0..3 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let vaddr_val = 0u64;
                let saddr_val = self.read_sgpr(saddr) as u64;
                let offset = (vaddr_val + saddr_val + (ioffset as u64) + (i as u64 * 4)) * 32
                    + (elem as u64) * 4;
                let addr = self.ctx.scratch.borrow().as_ptr() as u64 + offset;
                assert!(offset < self.ctx.scratch.borrow().len() as u64);

                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn scratch_load_b128(&mut self, _vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        for i in 0..4 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let vaddr_val = 0u64;
                let saddr_val = self.read_sgpr(saddr) as u64;
                let offset = (vaddr_val + saddr_val + (ioffset as u64) + (i as u64 * 4)) * 32
                    + (elem as u64) * 4;
                let addr = self.ctx.scratch.borrow().as_ptr() as u64 + offset;
                assert!(offset < self.ctx.scratch.borrow().len() as u64);

                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn execute_vglobal(&mut self, inst: VGLOBAL) -> Signals {
        let saddr = inst.saddr as usize;
        let vaddr = inst.vaddr as usize;
        let vsrc = inst.vsrc as usize;
        let vdst = inst.vdst as usize;
        let ioffset = inst.ioffset as u32;
        match inst.op {
            I::GLOBAL_STORE_B16 => {
                self.global_store_b16(vaddr, vsrc, saddr, ioffset);
            }
            I::GLOBAL_STORE_B32 => {
                self.global_store_b32(vaddr, vsrc, saddr, ioffset);
            }
            I::GLOBAL_STORE_B64 => {
                self.global_store_b64(vaddr, vsrc, saddr, ioffset);
            }
            I::GLOBAL_STORE_B128 => {
                self.global_store_b128(vaddr, vsrc, saddr, ioffset);
            }
            I::GLOBAL_LOAD_U8 => {
                self.global_load_u8(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_LOAD_U16 => {
                self.global_load_u16(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_LOAD_B32 => {
                self.global_load_b32(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_LOAD_B64 => {
                self.global_load_b64(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_LOAD_B128 => {
                self.global_load_b128(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_WB => {}
            I::GLOBAL_INV => {}
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn global_store_b16(&mut self, vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let data = self.read_vgpr(elem, vsrc);
            let addr = offset[elem] + (ioffset as u64);

            let ptr = addr as *mut u16;
            unsafe {
                *ptr = data as u16;
            }
        }
    }

    fn global_store_b32(&mut self, vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let data = self.read_vgpr(elem, vsrc);
            let addr = offset[elem] + (ioffset as u64);

            let ptr = addr as *mut u32;
            unsafe {
                *ptr = data;
            }
        }
    }

    fn global_store_b64(&mut self, vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..2 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let data = self.read_vgpr(elem, vsrc + i);
                let addr = offset[elem] + (ioffset as u64) + (i as u64 * 4);

                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn global_store_b128(&mut self, vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..4 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let data = self.read_vgpr(elem, vsrc + i);
                let addr = offset[elem] + (ioffset as u64) + (i as u64 * 4);

                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn global_load_u8(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let addr = offset[elem] + (ioffset as u64);

            let ptr = addr as *mut u8;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, data as u32);
        }
    }

    fn global_load_u16(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let addr = offset[elem] + (ioffset as u64);

            let ptr = addr as *mut u16;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, data as u32);
        }
    }

    fn global_load_b32(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let addr = offset[elem] + (ioffset as u64);

            let ptr = addr as *mut u32;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, data);
        }
    }

    fn global_load_b64(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..2 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let addr = offset[elem] + (ioffset as u64) + (i as u64 * 4);

                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn global_load_b128(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..4 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let ioffset = ((ioffset << 8) as i32) >> 8;
                let addr = offset[elem] as i64 + (ioffset as i64) + (i as i64 * 4);

                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn execute_vimage(&mut self, inst: VIMAGE) -> Signals {
        let vdata = inst.vdata as usize;
        let vaddr0 = inst.vaddr0 as usize;
        let vaddr1 = inst.vaddr1 as usize;
        let vaddr2 = inst.vaddr2 as usize;
        let vaddr3 = inst.vaddr3 as usize;
        let vaddr4 = inst.vaddr4 as usize;
        let s = inst.rsrc as usize;
        match inst.op {
            I::IMAGE_BVH64_INTERSECT_RAY => {
                self.image_bvh64_intersect_ray(vdata, vaddr0, vaddr1, vaddr2, vaddr3, vaddr4, s);
            }
            I::IMAGE_BVH8_INTERSECT_RAY => {
                self.image_bvh8_intersect_ray(vdata, vaddr0, vaddr1, vaddr2, vaddr3, vaddr4, s);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn image_bvh64_intersect_ray(
        &mut self,
        vdata: usize,
        vaddr0: usize,
        vaddr1: usize,
        vaddr2: usize,
        vaddr3: usize,
        vaddr4: usize,
        s: usize,
    ) {
        let s0_value = self.read_sgpr(s);
        let s1_value = self.read_sgpr(s + 1);
        let _s2_value = self.read_sgpr(s + 2);
        let _s3_value = self.read_sgpr(s + 3);
        let _base_addr = (((s1_value as u64) << 32) | (s0_value as u64)) & ((1u64 << 48) - 1);
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let node_ptr = self.read_vgpr_pair(elem, vaddr0);
            let node_type = (node_ptr & 0x7) as u8;
            match node_type {
                5 => {
                    let node_ptr = (node_ptr & !0x7u64) << 3;
                    let node = unsafe { *(node_ptr as *const Box4Node) };
                    let ray_extent = u32_to_f32(self.read_vgpr(elem, vaddr1));
                    let ray_origin_x = u32_to_f32(self.read_vgpr(elem, vaddr2));
                    let ray_origin_y = u32_to_f32(self.read_vgpr(elem, vaddr2 + 1));
                    let ray_origin_z = u32_to_f32(self.read_vgpr(elem, vaddr2 + 2));
                    let ray_inv_dir_x = u32_to_f32(self.read_vgpr(elem, vaddr4));
                    let ray_inv_dir_y = u32_to_f32(self.read_vgpr(elem, vaddr4 + 1));
                    let ray_inv_dir_z = u32_to_f32(self.read_vgpr(elem, vaddr4 + 2));

                    let mut s0 = intersect(
                        [ray_origin_x, ray_origin_y, ray_origin_z],
                        [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                        &node.aabb[0],
                        ray_extent,
                    );
                    let mut s1 = intersect(
                        [ray_origin_x, ray_origin_y, ray_origin_z],
                        [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                        &node.aabb[1],
                        ray_extent,
                    );
                    let mut s2 = intersect(
                        [ray_origin_x, ray_origin_y, ray_origin_z],
                        [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                        &node.aabb[2],
                        ray_extent,
                    );
                    let mut s3 = intersect(
                        [ray_origin_x, ray_origin_y, ray_origin_z],
                        [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                        &node.aabb[3],
                        ray_extent,
                    );

                    let mut result0 = if s0.0 <= s0.1 {
                        node.child_index[0]
                    } else {
                        0xFFFF_FFFF
                    };
                    let mut result1 = if s1.0 <= s1.1 {
                        node.child_index[1]
                    } else {
                        0xFFFF_FFFF
                    };
                    let mut result2 = if s2.0 <= s2.1 {
                        node.child_index[2]
                    } else {
                        0xFFFF_FFFF
                    };
                    let mut result3 = if s3.0 <= s3.1 {
                        node.child_index[3]
                    } else {
                        0xFFFF_FFFF
                    };

                    let sort = |child_index_a: &mut u32,
                                child_index_b: &mut u32,
                                dist_a: &mut f32,
                                dist_b: &mut f32| {
                        if (*child_index_b != 0xFFFF_FFFF && dist_b < dist_a)
                            || *child_index_a == 0xFFFF_FFFF
                        {
                            let t0 = *dist_a;
                            let t1 = *child_index_a;
                            *child_index_a = *child_index_b;
                            *dist_a = *dist_b;
                            *child_index_b = t1;
                            *dist_b = t0;
                        }
                    };

                    sort(&mut result0, &mut result2, &mut s0.0, &mut s2.0);
                    sort(&mut result1, &mut result3, &mut s1.0, &mut s3.0);
                    sort(&mut result0, &mut result1, &mut s0.0, &mut s1.0);
                    sort(&mut result2, &mut result3, &mut s2.0, &mut s3.0);
                    sort(&mut result1, &mut result2, &mut s1.0, &mut s2.0);

                    self.write_vgpr(elem, vdata, result0);
                    self.write_vgpr(elem, vdata + 1, result1);
                    self.write_vgpr(elem, vdata + 2, result2);
                    self.write_vgpr(elem, vdata + 3, result3);
                }
                0 | 1 => {
                    let node_ptr = (node_ptr & !(0x7u64)) << 3;
                    let node = unsafe { *(node_ptr as *const TrianglePairNode) };
                    let tri = if node_type & 1 == 0 {
                        [node.tri_pair.v0, node.tri_pair.v1, node.tri_pair.v2]
                    } else {
                        [node.tri_pair.v3, node.tri_pair.v2, node.tri_pair.v1]
                    };
                    let ray_origin_x = u32_to_f32(self.read_vgpr(elem, vaddr2));
                    let ray_origin_y = u32_to_f32(self.read_vgpr(elem, vaddr2 + 1));
                    let ray_origin_z = u32_to_f32(self.read_vgpr(elem, vaddr2 + 2));
                    let ray_dir_x = u32_to_f32(self.read_vgpr(elem, vaddr3));
                    let ray_dir_y = u32_to_f32(self.read_vgpr(elem, vaddr3 + 1));
                    let ray_dir_z = u32_to_f32(self.read_vgpr(elem, vaddr3 + 2));
                    let result = intersect_triangle_frac(
                        [ray_origin_x, ray_origin_y, ray_origin_z],
                        [ray_dir_x, ray_dir_y, ray_dir_z],
                        tri[0],
                        tri[1],
                        tri[2],
                        node.flags >> ((node_type & 1) * 8),
                    );
                    self.write_vgpr(elem, vdata, f32_to_u32(result.0));
                    self.write_vgpr(elem, vdata + 1, f32_to_u32(result.1));
                    self.write_vgpr(elem, vdata + 2, f32_to_u32(result.2));
                    self.write_vgpr(elem, vdata + 3, f32_to_u32(result.3));
                }
                _ => {
                    panic!("Unsupported node type: {}", node_type);
                }
            }
        }
    }

    fn image_bvh8_intersect_ray(
        &mut self,
        vdata: usize,
        vaddr0: usize,
        vaddr1: usize,
        vaddr2: usize,
        vaddr3: usize,
        vaddr4: usize,
        s: usize,
    ) {
        let s0_value = self.read_sgpr(s);
        let s1_value = self.read_sgpr(s + 1);
        let _s2_value = self.read_sgpr(s + 2);
        let _s3_value = self.read_sgpr(s + 3);
        let _base_addr = (((s1_value as u64) << 32) | (s0_value as u64)) & ((1u64 << 48) - 1);
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let node_base = self.read_vgpr_pair(elem, vaddr0);
            let node_index = self.read_vgpr(elem, vaddr4);
            let node_ptr = (node_base + (node_index & !0xF) as u64) << 3;
            let node_type = (node_index & 0xF) as u8;
            match node_type {
                0..3 | 8..11 => {
                    let tri_pair_index = (node_type & 3) + ((node_type & 8) >> 1);
                    let node = unsafe { *(node_ptr as *const TrianglePacketNode) };
                    let tri0 = node.fetch_triangle(tri_pair_index as u32, 0);
                    let tri1 = node.fetch_triangle(tri_pair_index as u32, 1);

                    let ray_origin_x = u32_to_f32(self.read_vgpr(elem, vaddr2));
                    let ray_origin_y = u32_to_f32(self.read_vgpr(elem, vaddr2 + 1));
                    let ray_origin_z = u32_to_f32(self.read_vgpr(elem, vaddr2 + 2));
                    let ray_dir_x = u32_to_f32(self.read_vgpr(elem, vaddr3));
                    let ray_dir_y = u32_to_f32(self.read_vgpr(elem, vaddr3 + 1));
                    let ray_dir_z = u32_to_f32(self.read_vgpr(elem, vaddr3 + 2));

                    let result0 = intersect_triangle(
                        [ray_origin_x, ray_origin_y, ray_origin_z],
                        [ray_dir_x, ray_dir_y, ray_dir_z],
                        tri0[0],
                        tri0[1],
                        tri0[2],
                    );
                    let result1 = intersect_triangle(
                        [ray_origin_x, ray_origin_y, ray_origin_z],
                        [ray_dir_x, ray_dir_y, ray_dir_z],
                        tri1[0],
                        tri1[1],
                        tri1[2],
                    );

                    let prim0 = node.get_prim_index(tri_pair_index as u32, 0);
                    let prim1 = node.get_prim_index(tri_pair_index as u32, 1);

                    let node_end = (tri_pair_index as u32 + 1) == node.get_triangle_pair_count();
                    let range_end = node.is_range_end(tri_pair_index as u32);

                    self.write_vgpr(elem, vdata, f32_to_u32(result0.0));
                    self.write_vgpr(elem, vdata + 1, f32_to_u32(result0.1));
                    self.write_vgpr(elem, vdata + 2, f32_to_u32(result0.2));
                    self.write_vgpr(elem, vdata + 3, prim0);
                    self.write_vgpr(elem, vdata + 4, f32_to_u32(result1.0));
                    self.write_vgpr(elem, vdata + 5, f32_to_u32(result1.1));
                    self.write_vgpr(elem, vdata + 6, f32_to_u32(result1.2));
                    self.write_vgpr(elem, vdata + 7, prim1);
                    self.write_vgpr(
                        elem,
                        vdata + 8,
                        ((range_end as u32) << 1) | (node_end as u32),
                    );
                }
                5 => {
                    let node = unsafe { *(node_ptr as *const Box8Node) };
                    let ray_extent = u32_to_f32(self.read_vgpr(elem, vaddr1));
                    let ray_origin_x = u32_to_f32(self.read_vgpr(elem, vaddr2));
                    let ray_origin_y = u32_to_f32(self.read_vgpr(elem, vaddr2 + 1));
                    let ray_origin_z = u32_to_f32(self.read_vgpr(elem, vaddr2 + 2));
                    let ray_dir_x = u32_to_f32(self.read_vgpr(elem, vaddr3));
                    let ray_dir_y = u32_to_f32(self.read_vgpr(elem, vaddr3 + 1));
                    let ray_dir_z = u32_to_f32(self.read_vgpr(elem, vaddr3 + 2));
                    let ray_inv_dir_x = 1.0 / ray_dir_x;
                    let ray_inv_dir_y = 1.0 / ray_dir_y;
                    let ray_inv_dir_z = 1.0 / ray_dir_z;

                    let child_count = node.get_child_count();

                    let boxes = (0..child_count)
                        .map(|i| node.get_child_box(i as usize))
                        .collect::<Vec<Aabb>>();

                    let s = boxes
                        .iter()
                        .map(|aabb| {
                            intersect(
                                [ray_origin_x, ray_origin_y, ray_origin_z],
                                [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                                aabb,
                                ray_extent,
                            )
                        })
                        .collect::<Vec<(f32, f32)>>();

                    let results = (0..8)
                        .map(|i| {
                            if i < 8 && s[i as usize].0 <= s[i as usize].1 {
                                node.get_child_index(i as usize)
                            } else {
                                0xFFFF_FFFF
                            }
                        })
                        .collect::<Vec<u32>>();

                    let results = results
                        .into_iter()
                        .zip(s.into_iter())
                        .sorted_by(|&(_idx_a, (dist_a, _)), &(_idx_b, (dist_b, _))| {
                            if (_idx_b != 0xFFFF_FFFF && dist_b < dist_a) || _idx_a == 0xFFFF_FFFF {
                                std::cmp::Ordering::Greater
                            } else {
                                std::cmp::Ordering::Less
                            }
                        })
                        .map(|(idx, _)| idx)
                        .collect::<Vec<u32>>();

                    for i in 0..8 {
                        self.write_vgpr(elem, vdata + i as usize, results[i as usize]);
                    }
                }
                _ => {
                    panic!("Unsupported node type: {}", node_type);
                }
            }
        }
    }

    fn execute_ds(&mut self, inst: DS) -> Signals {
        let addr = inst.addr as usize;
        let data0 = inst.data0 as usize;
        let vdst = inst.vdst as usize;
        let offset0 = inst.offset0;
        match inst.op {
            I::DS_LOAD_U8 => {
                self.ds_load_u8(addr, vdst, offset0);
            }
            I::DS_STORE_B8 => {
                self.ds_store_b8(addr, data0, offset0);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn ds_load_u8(&mut self, addr: usize, vdst: usize, offset0: u8) {
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let addr = addr[elem] + (offset0 as usize);

            let ptr = lds.wrapping_add(addr);
            let data = unsafe { *ptr };

            self.write_vgpr(elem, vdst, data as u32);
        }
    }

    fn ds_store_b8(&mut self, addr: usize, data0: usize, offset0: u8) {
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow_mut().as_mut_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let data = self.read_vgpr(elem, data0) as u8;
            let addr = addr[elem] + (offset0 as usize);

            let ptr = lds.wrapping_add(addr);
            unsafe {
                *ptr = data;
            }
        }
    }

    fn execute_sopp(&mut self, inst: SOPP) -> Signals {
        let simm16 = inst.simm16 as i16;
        match inst.op {
            I::S_NOP => {}
            I::S_ENDPGM => return Signals::EndOfProgram,
            I::S_WAIT_ALU => {}
            I::S_WAIT_KMCNT => {}
            I::S_WAIT_LOADCNT => {}
            I::S_WAIT_BVHCNT => {}
            I::S_WAIT_SAMPLECNT => {}
            I::S_WAIT_STORECNT => {}
            I::S_WAIT_LOADCNT_DSCNT => {}
            I::S_WAIT_DSCNT => {}
            I::S_CLAUSE => {}
            I::S_DELAY_ALU => {}
            I::S_SENDMSG => {}
            I::S_CBRANCH_EXECZ => {
                if self.is_execz() {
                    self.ctx.pc = ((self.ctx.pc as i64) + ((simm16 as i64) * 4)) as u64;
                }
            }
            I::S_CBRANCH_EXECNZ => {
                if self.is_execnz() {
                    self.ctx.pc = ((self.ctx.pc as i64) + ((simm16 as i64) * 4)) as u64;
                }
            }
            I::S_CBRANCH_VCCZ => {
                if self.is_vccz() {
                    self.ctx.pc = ((self.ctx.pc as i64) + ((simm16 as i64) * 4)) as u64;
                }
            }
            I::S_CBRANCH_VCCNZ => {
                if self.is_vccnz() {
                    self.ctx.pc = ((self.ctx.pc as i64) + ((simm16 as i64) * 4)) as u64;
                }
            }
            I::S_CBRANCH_SCC0 => {
                if !self.ctx.scc {
                    self.ctx.pc = ((self.ctx.pc as i64) + ((simm16 as i64) * 4)) as u64;
                }
            }
            I::S_CBRANCH_SCC1 => {
                if self.ctx.scc {
                    self.ctx.pc = ((self.ctx.pc as i64) + ((simm16 as i64) * 4)) as u64;
                }
            }
            I::S_BRANCH => {
                self.ctx.pc = ((self.ctx.pc as i64) + ((simm16 as i64) * 4)) as u64;
            }
            I::S_BARRIER_WAIT => {
                assert!(simm16 == -1);
                return Signals::Switch;
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
}

#[derive(Debug, Clone)]
struct RegisterSetupData {
    user_sgpr_count: usize,
    sgprs: [u32; 16],
    vgprs: [[u32; 32]; 16],
    scratch: Rc<RefCell<AVec<u8, ConstAlign<0x1_0000_0000>>>>,
}

fn decode_kernel_desc(kd: &[u8]) -> KernelDescriptor {
    KernelDescriptor {
        group_segment_fixed_size: get_u32(kd, 0) as usize,
        private_segment_fixed_size: get_u32(kd, 4) as usize,
        max_flat_workgroup_size: get_u32(kd, 8) as usize,
        is_dynamic_call_stack: get_bit(kd, 12, 0),
        is_xnack_enabled: get_bit(kd, 12, 1),
        kernel_code_entry_byte_offset: get_u64(kd, 16) as usize,
        enable_sgpr_private_segment_buffer: get_bit(kd, 56, 0),
        enable_sgpr_dispatch_ptr: get_bit(kd, 56, 1),
        enable_sgpr_queue_ptr: get_bit(kd, 56, 2),
        enable_sgpr_kernarg_segment_ptr: get_bit(kd, 56, 3),
        enable_sgpr_dispatch_id: get_bit(kd, 56, 4),
        enable_sgpr_flat_scratch_init: get_bit(kd, 56, 5),
        enable_sgpr_private_segment: get_bit(kd, 56, 6),
        enable_sgpr_grid_workgroup_count_x: get_bit(kd, 56, 7),
        enable_sgpr_grid_workgroup_count_y: get_bit(kd, 57, 0),
        enable_sgpr_grid_workgroup_count_z: get_bit(kd, 57, 1),
        granulated_workitem_vgpr_count: (get_bits(kd, 48, 0, 6) as usize + 1) * 8,
        granulated_wavefront_sgpr_count: 0,
        enable_sgpr_private_segment_wave_offset: get_bit(kd, 52, 0),
        user_sgpr_count: get_bits(kd, 52, 1, 5) as usize,
        enable_trap_handler: get_bit(kd, 52, 6),
        enable_sgpr_workgroup_id_x: get_bit(kd, 52, 7),
        enable_sgpr_workgroup_id_y: get_bit(kd, 52, 8),
        enable_sgpr_workgroup_id_z: get_bit(kd, 52, 9),
        enable_sgpr_workgroup_info: get_bit(kd, 52, 10),
        enable_vgpr_workitem_id: get_bits(kd, 52, 11, 2),
    }
}

struct ComputeUnit {
    simds: Vec<Arc<Mutex<SIMD32>>>,
}

use std::collections::HashMap;

impl ComputeUnit {
    pub fn new(pc: usize, insts: Vec<u8>, num_vgprs: usize, lds: Rc<RefCell<Vec<u8>>>) -> Self {
        let mut simds = vec![];
        for _ in 0..2 {
            let num_wave_slot = 16;
            simds.push(Arc::new(Mutex::new(SIMD32 {
                slots: Vec::new(),
                ctx: Context {
                    id: 0,
                    pc: pc as u64,
                    scc: false,
                    scratch: Rc::new(RefCell::new(AVec::new(0x1_0000_0000))),
                },
                insts: insts.clone(),
                sgprs: RegisterFileImpl::new(1, 128 * num_wave_slot, 0),
                vgprs: RegisterFileImpl::new(32, 1536 / 4, 0),
                num_vgprs: num_vgprs,
                lds: lds.clone(),
                translator: RDNATranslator::new(),
            })));
        }

        ComputeUnit { simds: simds }
    }
}

struct WorkgroupProcessor {
    cunits: Vec<ComputeUnit>,
}

use std::rc::Rc;
use std::sync::{Arc, Mutex};
use threadpool::ThreadPool;

pub struct RDNAProcessor<'a> {
    wgps: Vec<WorkgroupProcessor>,
    entry_address: usize,
    kernel_desc: KernelDescriptor,
    aql_packet_address: u64,
    kernel_args_ptr: u64,
    aql: HsaKernelDispatchPacket<'a>,
}

unsafe impl<'a> Send for SIMD32 {}

impl<'a> RDNAProcessor<'a> {
    pub fn new(
        aql: &HsaKernelDispatchPacket<'a>,
        num_cunits: usize,
        wavefront_size: usize,
        mem: &Vec<u8>,
    ) -> Self {
        let insts = aql.kernel_object.object.to_vec();
        let kd = aql.kernel_object.offset;
        let kernel_desc = decode_kernel_desc(&insts[kd..(kd + 64)]);
        let aql_packet_address = (aql as *const HsaKernelDispatchPacket) as u64;
        let num_wgps = num_cunits / 2;

        assert!(num_cunits % 2 == 0, "Number of compute units must be even.");
        assert!(wavefront_size == 32, "Wavefront size must be 32.");

        let mut wgps = vec![];
        for _ in 0..num_wgps {
            let mut cunits_in_wgp = vec![];
            let lds = Rc::new(RefCell::new(vec![0u8; 128 * 1024]));
            for _ in 0..2 {
                let cu = ComputeUnit::new(
                    kd + kernel_desc.kernel_code_entry_byte_offset,
                    mem.clone(),
                    kernel_desc.granulated_workitem_vgpr_count,
                    lds.clone(),
                );
                cunits_in_wgp.push(cu);
            }
            let wgp = WorkgroupProcessor {
                cunits: cunits_in_wgp,
            };

            wgps.push(wgp);
        }

        let kernel_args_ptr = aql.kernarg_address.address();
        let entry_address = kd + kernel_desc.kernel_code_entry_byte_offset;

        // create instance
        RDNAProcessor {
            wgps: wgps,
            kernel_desc: kernel_desc,
            kernel_args_ptr: kernel_args_ptr,
            aql_packet_address: aql_packet_address,
            entry_address: entry_address,
            aql: *aql,
        }
    }

    fn dispatch(
        &self,
        workgroup_id_x: u32,
        workgroup_id_y: u32,
        workgroup_id_z: u32,
        workitem_offset: usize,
    ) -> RegisterSetupData {
        let kernel_args_ptr = self.kernel_args_ptr;
        let aql_packet_address = self.aql_packet_address;
        let kernel_desc = &self.kernel_desc;
        let private_seg_size = self.aql.private_segment_size as u64;

        let mut scratch = AVec::new(0x1_0000_0000);
        scratch.resize(private_seg_size as usize * 32, 0);
        let scratch_base = scratch.as_ptr() as u64;

        // Initialize SGPRS
        let mut sgprs = [0u32; 16];
        let mut sgprs_pos = 0;
        if kernel_desc.enable_sgpr_private_segment_buffer {
            let mut desc_w0 = 0;
            desc_w0 |= scratch_base & ((1 << 48) - 1);
            desc_w0 |= (private_seg_size & ((1 << 14) - 1)) << 48;
            for i in 0..2 {
                sgprs[sgprs_pos + i] = ((desc_w0 >> (i * 32)) & 0xFFFFFFFF) as u32;
            }
            sgprs_pos += 4;
        }
        if kernel_desc.enable_sgpr_dispatch_ptr {
            sgprs[sgprs_pos] = (aql_packet_address & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((aql_packet_address >> 32) & 0xFFFFFFFF) as u32;
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_queue_ptr {
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_kernarg_segment_ptr {
            sgprs[sgprs_pos] = (kernel_args_ptr & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((kernel_args_ptr >> 32) & 0xFFFFFFFF) as u32;
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_dispatch_id {
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_flat_scratch_init {
            sgprs[sgprs_pos] = workitem_offset as u32 * self.aql.private_segment_size;
            sgprs[sgprs_pos + 1] = self.aql.private_segment_size;
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_x && sgprs_pos < 16 {
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_y && sgprs_pos < 16 {
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_z && sgprs_pos < 16 {
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_x {
            sgprs[sgprs_pos] = workgroup_id_x;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_y {
            sgprs[sgprs_pos] = workgroup_id_y;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_z {
            sgprs[sgprs_pos] = workgroup_id_z;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_info {
            sgprs[sgprs_pos] = 0;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_private_segment_wave_offset {
            sgprs[sgprs_pos] = 0;
        }

        // Initialize VGPRS
        let mut vgprs = [[0u32; 32]; 16];
        let vgprs_pos = 0;
        for i in 0..32 {
            let id_x = (i + workitem_offset) % self.aql.workgroup_size_x as usize;
            vgprs[vgprs_pos][i] = id_x as u32;
        }
        if kernel_desc.enable_vgpr_workitem_id > 0 {
            for i in 0..32 {
                let id_y = ((i + workitem_offset) / self.aql.workgroup_size_x as usize)
                    % self.aql.workgroup_size_y as usize;
                vgprs[vgprs_pos][i] |= (id_y as u32) << 10;
            }
        }
        if kernel_desc.enable_vgpr_workitem_id > 1 {
            for i in 0..32 {
                let id_z = ((i + workitem_offset)
                    / (self.aql.workgroup_size_x * self.aql.workgroup_size_y) as usize)
                    % self.aql.workgroup_size_z as usize;
                vgprs[vgprs_pos][i] |= (id_z as u32) << 20;
            }
        }

        RegisterSetupData {
            user_sgpr_count: kernel_desc.user_sgpr_count,
            sgprs: sgprs,
            vgprs: vgprs,
            scratch: Rc::new(RefCell::new(scratch)),
        }
    }

    pub fn execute(&mut self) {
        let workgroup_size_x = self.aql.workgroup_size_x as u32;
        let workgroup_size_y = self.aql.workgroup_size_y as u32;
        let workgroup_size_z = self.aql.workgroup_size_z as u32;

        let workgroup_size = (workgroup_size_x * workgroup_size_y * workgroup_size_z) as usize;

        let num_workgroup_x =
            (self.aql.grid_size_x * workgroup_size_x + workgroup_size_x - 1) / workgroup_size_x;
        let num_workgroup_y =
            (self.aql.grid_size_y * workgroup_size_y + workgroup_size_y - 1) / workgroup_size_y;
        let num_workgroup_z =
            (self.aql.grid_size_z * workgroup_size_z + workgroup_size_z - 1) / workgroup_size_z;

        let num_workgroups = num_workgroup_x * num_workgroup_y * num_workgroup_z;

        use indicatif::{ProgressBar, ProgressStyle};
        let bar = ProgressBar::new(num_workgroups as u64);

        bar.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta_precise}) \n {msg}")
            .progress_chars("#>-"));

        let num_wgps = self.wgps.len();

        let pool = ThreadPool::new(16);

        let insts = self.aql.kernel_object.object.to_vec();
        let entry_address = self.entry_address;

        if USE_ENTIRE_KERNEL_TRANSLATION {
            let mut translator = RDNATranslator::new();

            if translator.insts_blocks.is_empty() {
                let program = RDNAProgram::new(entry_address, &insts);
                translator.build_from_program(&program, self.aql.private_segment_size as usize);
            }

            for wgp in &mut self.wgps {
                for cu in &mut wgp.cunits {
                    for simd in &cu.simds {
                        let mut v = simd.lock().unwrap();
                        v.translator = translator.clone();
                    }
                }
            }
        }

        for workgroup_id_base in (0..num_workgroups).step_by(num_wgps) {
            for wgp_idx in 0..num_wgps {
                let workgroup_id = workgroup_id_base + wgp_idx as u32;
                let workgroup_id_x = workgroup_id % num_workgroup_x;
                let workgroup_id_y = (workgroup_id / num_workgroup_x) % num_workgroup_y;
                let workgroup_id_z =
                    (workgroup_id / (num_workgroup_x * num_workgroup_y)) % num_workgroup_z;

                let mut simds = VecDeque::new();

                for cu_idx in 0..2 {
                    for simd_idx in 0..2 {
                        let mut setup_data = vec![];

                        if cu_idx * 64 + simd_idx * 32 >= workgroup_size {
                            continue;
                        }

                        for workitem_id in (0..workgroup_size).step_by(32 * 2 * 2) {
                            let workitem_offset = workitem_id
                                + cu_idx * 64
                                + simd_idx * 32
                                + wgp_idx * workgroup_size;

                            setup_data.push(self.dispatch(
                                workgroup_id_x,
                                workgroup_id_y,
                                workgroup_id_z,
                                workitem_offset,
                            ));
                        }

                        let simd: Arc<Mutex<SIMD32>> =
                            Arc::clone(&self.wgps[wgp_idx].cunits[cu_idx].simds[simd_idx]);

                        if let Ok(mut v) = simd.lock() {
                            v.dispatch(entry_address, setup_data)
                        }

                        simds.push_back(simd);
                    }
                }

                let bar = bar.clone();
                pool.execute(move || {
                    let is_signal_none = |signal: &Signals| match signal {
                        Signals::None => true,
                        _ => false,
                    };

                    while !simds.is_empty() {
                        if let Some(simd) = simds.pop_front() {
                            if let Ok(mut v) = simd.lock() {
                                let mut switch_ctxs = Vec::new();
                                for ctx in v.slots.clone() {
                                    v.ctx = ctx;
                                    let mut signal = Signals::None;
                                    while is_signal_none(&signal) {
                                        signal = v.step();
                                    }

                                    match signal {
                                        Signals::EndOfProgram => {}
                                        Signals::Switch => switch_ctxs.push(v.ctx.clone()),
                                        _ => panic!(),
                                    }
                                }

                                if switch_ctxs.len() > 0 {
                                    v.slots = switch_ctxs.clone();
                                    simds.push_back(Arc::clone(&simd));
                                }
                            } else {
                                panic!("Failed to lock simd");
                            }
                        } else {
                            panic!("No simd available");
                        }
                    }

                    bar.inc(1);
                });
            }

            pool.join();
        }

        let mut sum_block_call_count = HashMap::new();
        let mut sum_block_elapsed_time = HashMap::new();
        let mut sum_instruction_count = HashMap::new();
        let mut instruction_usage = HashMap::new();
        for wgp in &self.wgps {
            for cu in &wgp.cunits {
                for simd in &cu.simds {
                    let v = simd.lock().unwrap();
                    for (addr, block) in &v.translator.insts_blocks {
                        *sum_block_call_count.entry(*addr).or_insert(0) += block.call_count;
                        *sum_block_elapsed_time.entry(*addr).or_insert(0) += block.elapsed_time;
                        *sum_instruction_count.entry(*addr).or_insert(0) += block.num_instructions;
                        for (inst, count) in block.instruction_usage.clone() {
                            *instruction_usage.entry(inst).or_insert(0) +=
                                count.clone() * block.call_count as u32;
                        }
                    }
                }
            }
        }

        let mut sorted_blocks: Vec<_> = sum_block_elapsed_time.iter().collect();
        sorted_blocks.sort_by(|a, b| b.1.cmp(a.1));
        println!("Block execution summary:");
        for (addr, elapsed_time) in sorted_blocks {
            let call_count = sum_block_call_count.get(addr).unwrap_or(&0);
            println!(
                "Block at 0x{:08X} executed {} times, total elapsed time: {} ms, instruction count: {}",
                addr,
                call_count,
                (*elapsed_time as f64 / 1_000_000.0),
                sum_instruction_count.get(addr).unwrap_or(&0)
            );
        }

        let mut total_elapsed_time = 0;
        for (_, elapsed_time) in &sum_block_elapsed_time {
            total_elapsed_time += elapsed_time;
        }

        println!(
            "\nTotal elapsed time: {} ms",
            (total_elapsed_time as f64 / 1_000_000.0)
        );

        println!("\nInstruction usage summary:");
        let mut sorted_instructions: Vec<_> = instruction_usage.iter().collect();
        sorted_instructions.sort_by(|a, b| b.1.cmp(a.1));
        for (inst, count) in sorted_instructions {
            println!("Instruction {:?} executed {} times", inst, count);
        }

        bar.finish();
    }
}
