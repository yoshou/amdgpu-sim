use crate::decoders::*;
use crate::instructions::*;

pub trait RegisterFile<T: Copy> {
    fn new(num_elems: usize, count: usize, default: T) -> Self;
    fn get(&self, elem: usize, idx: usize) -> T;

    fn set(&mut self, elem: usize, idx: usize, val: T);

    fn get_vec(&self, idx: usize) -> &[T];

    fn set_vec(&mut self, idx: usize, vals: &[Option<T>]);
}

pub struct RegisterFileImpl<T: Copy> {
    num_elems: usize,
    regs: Vec<T>,
}

impl<T: Copy> RegisterFile<T> for RegisterFileImpl<T> {
    fn new(num_elems: usize, count: usize, default: T) -> Self {
        RegisterFileImpl {
            num_elems: num_elems,
            regs: vec![default; num_elems * count],
        }
    }

    fn get(&self, elem: usize, idx: usize) -> T {
        self.regs[self.num_elems * idx + elem]
    }

    fn set(&mut self, elem: usize, idx: usize, val: T) {
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

pub trait Processor {
    fn step(&mut self) -> bool;
}

struct ComputeUnit {
    pc: usize,
    next_pc: usize,
    insts: Vec<u8>,
    pub sgprs: RegisterFileImpl<u32>,
    pub vgprs: RegisterFileImpl<u32>,
    scc: bool,
    exec_lo: u32,
    exec_hi: u32,
    num_sgprs: usize,
    num_vgprs: usize,
}

impl Processor for ComputeUnit {
    fn step(&mut self) -> bool {
        if let Ok((inst, size)) = decode_v3(self.fetch_inst()) {
            self.next_pc = self.get_pc() as usize + size;
            let result = self.execute_inst(inst);
            self.set_pc(self.next_pc as u64);
            result
        } else {
            println!("Unknown instruction.");
            false
        }
    }
}

#[inline(always)]
fn add_i32(a: i32, b: i32) -> (i32, bool) {
    let c = (a as i64) + (b as i64);
    (c as i32, (c > (i32::MAX as i64)) || (c < (i32::MIN as i64)))
}

#[inline(always)]
fn add_u32(a: u32, b: u32, c: u32) -> (u32, bool) {
    let d = (a as u64) + (b as u64) + (c as u64);
    (d as u32, d > (u32::MAX as u64))
}

#[inline(always)]
fn sub_i32(a: i32, b: i32) -> (i32, bool) {
    let c = (a as i64) - (b as i64);
    (c as i32, (c > (i32::MAX as i64)) || (c < (i32::MIN as i64)))
}

#[inline(always)]
fn sub_u32(a: u32, b: u32, c: u32) -> (u32, bool) {
    let d = (a as i64) - (b as i64) - (c as i64);
    (d as u32, (d < 0) || (d > u32::MAX as i64))
}

#[inline(always)]
fn mul_u32(a: u32, b: u32) -> u32 {
    let c = (a as u64) * (b as u64);
    (c & 0xFFFFFFFF) as u32
}

#[inline(always)]
fn mul_i32(a: i32, b: i32) -> i32 {
    let c = (a as i64) * (b as i64);
    c as i32
}

#[inline(always)]
fn min_u32(a: u32, b: u32) -> u32 {
    if a < b {
        a
    } else {
        b
    }
}

#[inline(always)]
fn max_u32(a: u32, b: u32) -> u32 {
    if a >= b {
        a
    } else {
        b
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

fn cmp_f32(a: f32, b: f32, op: OP16) -> bool {
    match op {
        OP16::F => false,
        OP16::LT => a < b,
        OP16::EQ => a == b,
        OP16::LE => a <= b,
        OP16::GT => a > b,
        OP16::LG => a != b,
        OP16::GE => a >= b,
        OP16::TRU => true,
        OP16::NLT => !(a < b),
        OP16::NEQ => !(a == b),
        OP16::NLE => !(a <= b),
        OP16::NGT => !(a > b),
        OP16::NLG => !(a != b),
        OP16::NGE => !(a >= b),
        OP16::O => !a.is_nan() && !b.is_nan(),
        OP16::U => a.is_nan() || b.is_nan(),
        _ => unimplemented!(),
    }
}

fn cmp_i32(a: i32, b: i32, op: OP8) -> bool {
    match op {
        OP8::F => false,
        OP8::LT => a < b,
        OP8::EQ => a == b,
        OP8::LE => a <= b,
        OP8::GT => a > b,
        OP8::LG => a != b,
        OP8::GE => a >= b,
        OP8::TRU => true,
        _ => panic!(),
    }
}

fn cmp_u16(a: u16, b: u16, op: OP8) -> bool {
    match op {
        OP8::F => false,
        OP8::LT => a < b,
        OP8::EQ => a == b,
        OP8::LE => a <= b,
        OP8::GT => a > b,
        OP8::LG => a != b,
        OP8::GE => a >= b,
        OP8::TRU => true,
        _ => panic!(),
    }
}

fn cmp_u32(a: u32, b: u32, op: OP8) -> bool {
    match op {
        OP8::F => false,
        OP8::LT => a < b,
        OP8::EQ => a == b,
        OP8::LE => a <= b,
        OP8::GT => a > b,
        OP8::LG => a != b,
        OP8::GE => a >= b,
        OP8::TRU => true,
        _ => panic!(),
    }
}

use num_traits::ops::mul_add::MulAdd;

#[inline(always)]
fn fma<T: MulAdd<Output = T>>(a: T, b: T, c: T) -> T {
    a.mul_add(b, c)
}

#[derive(Debug, Clone)]
struct KernelDescriptor {
    group_segment_fixed_size: usize,
    private_segment_fixed_size: usize,
    max_flat_workgroup_size: usize,
    is_dynamic_call_stack: bool,
    is_xnack_enabled: bool,
    kernel_code_entry_byte_offset: usize,
    // compute_pgm_rsrc1
    // granulated_workitem_vgpr_count: usize,
    // granulated_wavefront_sgpr_count: usize,
    // priority: u8,
    // float_mode_round_32: u8,
    // float_mode_round_16_64: u8,
    // float_mode_denorm_32: u8,
    // float_mode_denorm_16_64: u8,
    // _priv: bool,
    // enable_dx10_clamp: bool,
    // debug_mode: bool,
    // enable_ieee_mode: bool,
    // bulky: bool,
    // cdbg_user: bool,
    // compute_pgm_rsrc2
    enable_sgpr_private_segment_wave_offset: bool,
    user_sgpr_count: usize,
    enable_trap_handler: bool,
    enable_sgpr_workgroup_id_x: bool,
    enable_sgpr_workgroup_id_y: bool,
    enable_sgpr_workgroup_id_z: bool,
    enable_sgpr_workgroup_info: bool,
    enable_vgpr_workitem_id: u8,
    // enable_exception_address_watch: bool,
    // enable_exception_memory: bool,
    // granulated_lds_size: usize,
    // enable_exception_ieee_754_fp_invalid_operation: bool,
    // enable_exception_fp_denormal_source: bool,
    // enable_exception_ieee_754_fp_division_by_zero: bool,
    // enable_exception_ieee_754_fp_overflow: bool,
    // enable_exception_ieee_754_fp_underflow: bool,
    // enable_exception_ieee_754_fp_inexact: bool,
    // enable_exception_int_divide_by_zero: bool,
    //
    enable_sgpr_private_segment_buffer: bool,
    enable_sgpr_dispatch_ptr: bool,
    enable_sgpr_queue_ptr: bool,
    enable_sgpr_kernarg_segment_ptr: bool,
    enable_sgpr_dispatch_id: bool,
    enable_sgpr_flat_scratch_init: bool,
    enable_sgpr_private_segment: bool,
    enable_sgpr_grid_workgroup_count_x: bool,
    enable_sgpr_grid_workgroup_count_y: bool,
    enable_sgpr_grid_workgroup_count_z: bool,
}

fn get_bit(buffer: &[u8], offset: usize, bit: usize) -> bool {
    ((buffer[offset + (bit >> 3)] >> (bit & 0x7)) & 1) == 1
}

fn get_bits(buffer: &[u8], offset: usize, bit: usize, size: usize) -> u8 {
    (buffer[offset + (bit >> 3)] >> (bit & 0x7)) & ((1 << size) - 1)
}

fn get_u8(buffer: &[u8], offset: usize) -> u8 {
    buffer[offset]
}

fn get_u16(buffer: &[u8], offset: usize) -> u16 {
    let b0 = buffer[offset] as u16;
    let b1 = buffer[offset + 1] as u16;

    b0 | (b1 << 8)
}

fn get_u32(buffer: &[u8], offset: usize) -> u32 {
    let b0 = buffer[offset] as u32;
    let b1 = buffer[offset + 1] as u32;
    let b2 = buffer[offset + 2] as u32;
    let b3 = buffer[offset + 3] as u32;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

fn get_u64(buffer: &[u8], offset: usize) -> u64 {
    let b0 = buffer[offset] as u64;
    let b1 = buffer[offset + 1] as u64;
    let b2 = buffer[offset + 2] as u64;
    let b3 = buffer[offset + 3] as u64;
    let b4 = buffer[offset + 4] as u64;
    let b5 = buffer[offset + 5] as u64;
    let b6 = buffer[offset + 6] as u64;
    let b7 = buffer[offset + 7] as u64;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
}

#[inline(always)]
fn u32_to_f32(value: u32) -> f32 {
    unsafe { std::mem::transmute::<u32, f32>(value) }
}

#[inline(always)]
fn f32_to_u32(value: f32) -> u32 {
    unsafe { std::mem::transmute::<f32, u32>(value) }
}

#[inline(always)]
fn u64_to_f64(value: u64) -> f64 {
    unsafe { std::mem::transmute::<u64, f64>(value) }
}

#[inline(always)]
fn f64_to_u64(value: f64) -> u64 {
    unsafe { std::mem::transmute::<f64, u64>(value) }
}

#[inline(always)]
fn u32_to_f32_abs_neg(value: u32, abs: u8, neg: u8, idx: usize) -> f32 {
    let result = unsafe { std::mem::transmute::<u32, f32>(value) };
    abs_neg(result, abs, neg, idx)
}

#[inline(always)]
fn u64_to_f64_abs_neg(value: u64, abs: u8, neg: u8, idx: usize) -> f64 {
    let result = unsafe { std::mem::transmute::<u64, f64>(value) };
    abs_neg(result, abs, neg, idx)
}

fn clamp_f32(value: f32, min_value: f32, max_value: f32) -> f32 {
    if value < min_value {
        min_value
    } else if value > max_value {
        max_value
    } else {
        value
    }
}

fn f32_to_u32_clamp(value: f32, clamp: bool) -> u32 {
    let value = if clamp {
        clamp_f32(value, 0.0, 1.0)
    } else {
        value
    };
    unsafe { std::mem::transmute::<f32, u32>(value) }
}

#[inline(always)]
fn u64_from_u32_u32(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | (lo as u64)
}

fn get_exp_f64(val: f64) -> i16 {
    let bits: u64 = unsafe { std::mem::transmute(val) };
    ((bits >> 52) & 0x7ff) as i16
}

fn get_exp_f32(val: f32) -> i16 {
    let bits: u32 = unsafe { std::mem::transmute(val) };
    ((bits >> 23) & 0xff) as i16
}

fn get_sign_f32(val: f32) -> bool {
    let bits: u32 = unsafe { std::mem::transmute(val) };
    ((bits >> 31) & 0x1) != 0
}

fn get_sign_f64(val: f64) -> bool {
    let bits: u64 = unsafe { std::mem::transmute(val) };
    ((bits >> 63) & 0x1) != 0
}

fn get_mantissa_f32(val: f32) -> u32 {
    let bits: u32 = unsafe { std::mem::transmute(val) };
    bits & ((1 << 23) - 1)
}

fn get_mantissa_f64(val: f64) -> u64 {
    let bits: u64 = unsafe { std::mem::transmute(val) };
    bits & ((1 << 53) - 1)
}

fn frexp_f64(value: f64) -> (f64, i32) {
    libm::frexp(value)
}

fn frexp_f32(value: f32) -> (f32, i32) {
    libm::frexpf(value)
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
    if s2 == f32::NAN {
        s2
    } else if s1 == f32::NAN {
        s1
    } else if s1 == 0.0 && s2 == 0.0 {
        // 0/0
        u32_to_f32(0xffc00000)
    } else if s1.abs().is_infinite() && s2.abs().is_infinite() {
        // inf/inf
        u32_to_f32(0xffc00000)
    } else if s1 == 0.0 || s2.abs().is_infinite() {
        // x/0, or inf/y
        if sign_out {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        }
    } else if s1.abs().is_infinite() || s2 == 0.0 {
        // x/inf, 0/y
        if sign_out {
            -0.0
        } else {
            0.0
        }
    } else if (get_exp_f32(s2) - get_exp_f32(s1)) < -150 {
        panic!();
    } else if get_exp_f32(s1) == 255 {
        panic!();
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
    if s2 == f64::NAN {
        s2
    } else if s1 == f64::NAN {
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
        panic!();
    } else if get_exp_f64(s1) == 2047 {
        panic!();
    } else {
        if sign_out {
            -s0.abs()
        } else {
            s0.abs()
        }
    }
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

#[derive(Debug, Copy, Clone)]
pub struct Pointer<'a> {
    pub object: &'a [u8],
    pub offset: usize,
}

impl<'a> Pointer<'a> {
    pub fn new(object: &'a [u8], offset: usize) -> Self {
        Pointer {
            object: object,
            offset: offset,
        }
    }

    pub fn address(&self) -> u64 {
        (&self.object[0] as *const u8) as u64
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct hsa_kernel_dispatch_packet_s<'a> {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: Pointer<'a>,
    pub kernarg_address: Pointer<'a>,
    // hsa_signal_t completion_signal;
}

use std::sync::{Arc, Mutex, RwLock};

pub struct SISimulator<'a> {
    cunits: Vec<Arc<Mutex<ComputeUnit>>>,
    entry_address: usize,
    kernel_desc: KernelDescriptor,
    aql_packet_address: u64,
    kernel_args_ptr: u64,
    aql: hsa_kernel_dispatch_packet_s<'a>,
    private_seg_buffer: Vec<u8>,
}

impl<'a> SISimulator<'a> {
    pub fn new(aql: &hsa_kernel_dispatch_packet_s<'a>, num_cunits: usize) -> Self {
        let insts = aql.kernel_object.object.to_vec();
        let kd = aql.kernel_object.offset;
        let kernel_desc = decode_kernel_desc(&insts[kd..(kd + 64)]);
        let aql_packet_address = (aql as *const hsa_kernel_dispatch_packet_s) as u64;

        let mut cunits = vec![];
        for _ in 0..num_cunits {
            let cu = Arc::new(Mutex::new(ComputeUnit::new(
                kd + kernel_desc.kernel_code_entry_byte_offset,
                aql.kernel_object.object.to_vec(),
                100 + 4,
                256,
            )));

            cunits.push(cu);
        }

        let kernel_args_ptr = aql.kernarg_address.address();
        let entry_address = kd + kernel_desc.kernel_code_entry_byte_offset;

        let private_segment_size = aql.private_segment_size as usize;
        let private_seg_buffer: Vec<u8> = vec![0u8; private_segment_size * 256 * num_cunits];

        // create instance
        SISimulator {
            cunits: cunits,
            kernel_desc: kernel_desc,
            kernel_args_ptr: kernel_args_ptr,
            aql_packet_address: aql_packet_address,
            entry_address: entry_address,
            aql: *aql,
            private_seg_buffer: private_seg_buffer,
        }
    }

    fn dispatch(
        &self,
        thread_id: u32,
        workgroup_id_x: u32,
        workgroup_id_y: u32,
        workgroup_id_z: u32,
        workitem_offset: usize,
    ) -> ([u32; 16], [[u32; 64]; 16], usize) {
        let private_seg_ptr = if self.private_seg_buffer.len() > 0 {
            (&self.private_seg_buffer[0] as *const u8) as u64
        } else {
            0
        };

        let kernel_args_ptr = self.kernel_args_ptr;
        let aql_packet_address = self.aql_packet_address;
        let kernel_desc = &self.kernel_desc;
        let private_seg_size = self.aql.private_segment_size as u64;
        // initialize sgprs
        let mut sgprs = [0u32; 16];
        let mut sgprs_pos = 0;
        if kernel_desc.enable_sgpr_private_segment_buffer {
            let mut desc_w0 = 0;
            desc_w0 |=
                (private_seg_ptr + (thread_id as u64) * private_seg_size * 256) & ((1 << 48) - 1);
            desc_w0 |= (private_seg_size & ((1 << 14) - 1)) << 48;
            for i in 0..2 {
                sgprs[sgprs_pos + i] = ((desc_w0 >> (i * 32)) & 0xFFFFFFFF) as u32;
            }
            // println!(
            //     "s[{}..{}]: Private Segment Buffer",
            //     sgprs_pos,
            //     sgprs_pos + 3
            // );
            sgprs_pos += 4;
        }
        if kernel_desc.enable_sgpr_dispatch_ptr {
            sgprs[sgprs_pos] = (aql_packet_address & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((aql_packet_address >> 32) & 0xFFFFFFFF) as u32;
            // println!("s[{}..{}]: Dispatch Ptr", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_queue_ptr {
            // println!("s[{}..{}]: Queue Ptr", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_kernarg_segment_ptr {
            sgprs[sgprs_pos] = (kernel_args_ptr & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((kernel_args_ptr >> 32) & 0xFFFFFFFF) as u32;
            // println!("s[{}..{}]: Kernarg Segment Ptr", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_dispatch_id {
            // println!("s[{}..{}]: Dispatch Id", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_flat_scratch_init {
            sgprs[sgprs_pos] = thread_id * self.aql.private_segment_size;
            sgprs[sgprs_pos + 1] = self.aql.private_segment_size;
            // println!("s[{}..{}]: Flat Scratch Init", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_x && sgprs_pos < 16 {
            // println!("s[{}]: Grid Work-Group Count X", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_y && sgprs_pos < 16 {
            // println!("s[{}]: Grid Work-Group Count Y", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_z && sgprs_pos < 16 {
            // println!("s[{}]: Grid Work-Group Count Z", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_x {
            sgprs[sgprs_pos] = workgroup_id_x;
            // println!("s[{}]: Work-Group Id X", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_y {
            sgprs[sgprs_pos] = workgroup_id_y;
            // println!("s[{}]: Work-Group Id Y", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_z {
            sgprs[sgprs_pos] = workgroup_id_z;
            // println!("s[{}]: Work-Group Id Z", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_info {
            sgprs[sgprs_pos] = 0;
            // println!("s[{}]: Work-Group Info", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_private_segment_wave_offset {
            sgprs[sgprs_pos] = 0;
            // println!("s[{}]: Scratch Wave Offset", sgprs_pos);
            sgprs_pos += 1;
        }

        // initialize vgprs
        let mut vgprs = [[0u32; 64]; 16];
        let mut vgprs_pos = 0;
        for i in 0..64 {
            let id_x = (i + workitem_offset) % self.aql.workgroup_size_x as usize;
            vgprs[vgprs_pos][i] = id_x as u32;
        }
        vgprs_pos += 1;
        if kernel_desc.enable_vgpr_workitem_id > 0 {
            for i in 0..64 {
                let id_y = ((i + workitem_offset) / self.aql.workgroup_size_x as usize)
                    % self.aql.workgroup_size_y as usize;
                vgprs[vgprs_pos][i] = id_y as u32;
            }
            vgprs_pos += 1;
        }
        if kernel_desc.enable_vgpr_workitem_id > 1 {
            for i in 0..64 {
                let id_z = ((i + workitem_offset)
                    / (self.aql.workgroup_size_x * self.aql.workgroup_size_y) as usize)
                    % self.aql.workgroup_size_z as usize;
                vgprs[vgprs_pos][i] = id_z as u32;
            }
            vgprs_pos += 1;
        }

        // initialize pc
        (sgprs, vgprs, self.entry_address)
    }

    pub fn execute(&mut self) {
        let workgroup_size_x = self.aql.workgroup_size_x as u32;
        let workgroup_size_y = self.aql.workgroup_size_y as u32;
        let workgroup_size_z = self.aql.workgroup_size_z as u32;

        let workgroup_size = (workgroup_size_x * workgroup_size_y * workgroup_size_z) as usize;

        let num_workgroup_x = (self.aql.grid_size_x + workgroup_size_x - 1) / workgroup_size_x;
        let num_workgroup_y = (self.aql.grid_size_y + workgroup_size_y - 1) / workgroup_size_y;
        let num_workgroup_z = (self.aql.grid_size_z + workgroup_size_z - 1) / workgroup_size_z;

        let num_workgroups = num_workgroup_x * num_workgroup_y * num_workgroup_z;

        use indicatif::ProgressBar;
        let bar = ProgressBar::new(num_workgroups as u64);

        let num_cunits = self.cunits.len();

        for workgroup_id_base in (0..num_workgroups).step_by(num_cunits) {
            let mut thread_handles = vec![];
            for cu_idx in 0..num_cunits {
                let workgroup_id = workgroup_id_base + cu_idx as u32;
                let workgroup_id_x = workgroup_id % num_workgroup_x;
                let workgroup_id_y = (workgroup_id / num_workgroup_x) % num_workgroup_y;
                let workgroup_id_z =
                    (workgroup_id / (num_workgroup_x * num_workgroup_y)) % num_workgroup_z;

                let mut setup_data = vec![];
                for workitem_id in (0..workgroup_size).step_by(64) {
                    setup_data.push(self.dispatch(
                        cu_idx as u32,
                        workgroup_id_x,
                        workgroup_id_y,
                        workgroup_id_z,
                        workitem_id,
                    ));
                }

                let cu = Arc::clone(&self.cunits[cu_idx]);
                use std::thread;

                let handle = thread::spawn(move || {
                    if let Ok(mut v) = cu.lock() {
                        for s in setup_data {
                            v.dispatch(s);
                        }
                    }
                });
                thread_handles.push(handle);
            }

            for t in thread_handles {
                t.join();
                bar.inc(1);
            }
        }

        bar.finish();
    }
}

fn s_mov_b32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src(s0);
    let d_value = s0_value;
    cu.write_sop_dst(d, d_value);
}

fn s_mov_b64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let d_value = s0_value;
    cu.write_sop_dst_pair(d, d_value);
}

fn s_brev_b32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src(s0);
    let d_value = s0_value.reverse_bits();
    cu.write_sop_dst(d, d_value);
}

fn s_and_saveexec_b64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let exec_value = u64_from_u32_u32(cu.exec_lo, cu.exec_hi);

    cu.write_sop_dst_pair(d, exec_value);

    let exec_value = s0_value & exec_value;

    cu.exec_lo = (exec_value & 0xFFFFFFFF) as u32;
    cu.exec_hi = ((exec_value >> 32) & 0xFFFFFFFF) as u32;
    cu.scc = exec_value != 0;
}
fn s_or_saveexec_b64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let exec_value = u64_from_u32_u32(cu.exec_lo, cu.exec_hi);

    cu.write_sop_dst_pair(d, exec_value);

    let exec_value = s0_value | exec_value;

    cu.exec_lo = (exec_value & 0xFFFFFFFF) as u32;
    cu.exec_hi = ((exec_value >> 32) & 0xFFFFFFFF) as u32;
    cu.scc = exec_value != 0;
}
fn s_getpc_b64(cu: &mut ComputeUnit, d: usize) {
    let d_value = (cu.get_pc() + 4) as u64;
    cu.write_sop_dst_pair(d, d_value);
}
fn s_swappc_b64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let d_value = (cu.get_pc() + 4) as u64;
    cu.write_sop_dst_pair(d, d_value);
    cu.next_pc = s0_value as usize;
}
fn s_setpc_b64(cu: &mut ComputeUnit, s0: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    cu.next_pc = s0_value as usize;
}

fn s_add_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let (d_value, carry) = add_u32(s0_value, s1_value, 0);
    cu.write_sop_dst(d, d_value as u32);
    cu.scc = carry;
}

fn s_sub_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let (d_value, carry) = sub_u32(s0_value, s1_value, 0);
    cu.write_sop_dst(d, d_value as u32);
    cu.scc = carry;
}

fn s_add_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    let (d_value, overflow) = add_i32(s0_value, s1_value);
    cu.write_sop_dst(d, d_value as u32);
    cu.scc = overflow;
}

fn s_addc_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let (d_value, carry) = add_u32(s0_value, s1_value, cu.scc as u32);
    cu.write_sop_dst(d, d_value as u32);
    cu.scc = carry;
}

fn s_sub_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    let (d_value, overflow) = sub_i32(s0_value, s1_value);
    cu.write_sop_dst(d, d_value as u32);
    cu.scc = overflow;
}

fn s_and_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = s0_value & s1_value;
    cu.write_sop_dst(d, d_value);
    cu.scc = d_value != 0;
}

fn s_and_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value & s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.scc = d_value != 0;
}

fn s_andn2_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value & !s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.scc = d_value != 0;
}

fn s_or_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value | s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.scc = d_value != 0;
}

fn s_orn2_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value | !s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.scc = d_value != 0;
}

fn s_xor_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value ^ s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.scc = d_value != 0;
}

fn s_mul_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    let d_value = mul_i32(s0_value, s1_value);
    cu.write_sop_dst(d, d_value as u32);
}

fn s_lshl_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    let d_value = s0_value << ((s1_value) & 0x1F);
    cu.write_sop_dst(d, d_value as u32);
    cu.scc = d_value != 0;
}

fn s_lshr_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = s0_value >> (s1_value & 0x1F);
    cu.write_sop_dst(d, d_value);
    cu.scc = d_value != 0;
}

fn s_bfm_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = ((1 << ((s0_value) & 0x1F)) - 1) << ((s1_value) & 0x1F);
    cu.write_sop_dst(d, d_value);
}

fn s_cmp_eq_u32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    cu.scc = s0_value == s1_value;
}

fn s_cmp_gt_i32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    cu.scc = s0_value > s1_value;
}

fn s_cmp_lt_i32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    cu.scc = s0_value < s1_value;
}

fn s_cmp_lg_u32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    cu.scc = s0_value != s1_value;
}

fn s_load_dword(cu: &mut ComputeUnit, sdata: usize, sbase: usize, soffset: u64) {
    let sbase_val = cu.read_sgpr_pair(sbase);
    let ptr = (sbase_val + soffset) as *const u32;
    let data = unsafe { *ptr };
    cu.write_sgpr(sdata, data);
}

fn s_load_dwordx2(cu: &mut ComputeUnit, sdata: usize, sbase: usize, soffset: u64) {
    let sbase_val = cu.read_sgpr_pair(sbase);
    for i in 0..2 {
        let ptr = (sbase_val + soffset + (i * 4) as u64) as *const u32;
        let data = unsafe { *ptr };
        cu.write_sgpr(sdata + i, data);
    }
}

fn s_load_dwordx4(cu: &mut ComputeUnit, sdata: usize, sbase: usize, soffset: u64) {
    let sbase_val = cu.read_sgpr_pair(sbase);
    for i in 0..4 {
        let ptr = (sbase_val + soffset + (i * 4) as u64) as *const u32;
        let data = unsafe { *ptr };
        cu.write_sgpr(sdata + i, data);
    }
}

fn v_mov_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let d_value = s0_value;
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_bfrev_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let d_value = s0_value.reverse_bits();
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_rsq_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let d_value = 1.0 / s0_value.sqrt();
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_rcp_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let d_value = 1.0 / s0_value;
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_rcp_f64_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let d_value = f64_to_u64(1.0 / s0_value);

        cu.write_vgpr_pair(elem, d, d_value);
    }
}

fn v_sqrt_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let d_value = s0_value.sqrt();
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_ffbh_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let d_value = if s0_value == 0 {
            0xFFFFFFFF
        } else {
            s0_value.leading_zeros()
        };
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_cvt_f32_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let d_value = f32_to_u32(s0_value as f32);

        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_cvt_f64_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0) as i32;
        let d_value = f64_to_u64(s0_value as f64);

        cu.write_vgpr_pair(elem, d, d_value);
    }
}

fn v_cvt_f64_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let d_value = f64_to_u64(s0_value as f64);

        cu.write_vgpr_pair(elem, d, d_value);
    }
}

fn v_cvt_f64_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let d_value = f64_to_u64(s0_value as f64);

        cu.write_vgpr_pair(elem, d, d_value);
    }
}

fn v_cvt_f32_f64_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let d_value = f32_to_u32(s0_value as f32);

        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_cvt_i32_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let d_value = s0_value as i32;

        cu.write_vgpr(elem, d, d_value as u32);
    }
}

fn v_cvt_f32_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0) as i32;
        let d_value = f32_to_u32(s0_value as f32);

        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_frexp_mant_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let (d_value, _) = frexp_f32(s0_value);

        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_frexp_exp_i32_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let (_, d_value) = frexp_f32(s0_value);

        cu.write_vgpr(elem, d, d_value as u32);
    }
}

fn v_rndne_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let mut d_value = (s0_value + 0.5).floor();
        if s0_value.floor() % 2.0 == 0.0 && s0_value.fract() == 0.5 {
            d_value -= 1.0;
        }

        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_fract_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let d_value = s0_value.fract();

        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_rcp_iflag_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let d_value = 1.0 / s0_value;
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_cndmask_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let d_value = if cu.get_vcc(elem) { s1_value } else { s0_value };
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_add_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let (d_value, carry) = add_u32(s0_value, s1_value, 0);
        cu.write_vgpr(elem, d, d_value);
        cu.set_vcc(elem, carry);
    }
}

fn v_addc_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let (d_value, carry) = add_u32(s0_value, s1_value, cu.get_vcc(elem) as u32);
        cu.write_vgpr(elem, d, d_value);
        cu.set_vcc(elem, carry);
    }
}

fn v_and_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let d_value = s0_value & s1_value;
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_or_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let d_value = s0_value | s1_value;
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_xor_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let d_value = s0_value ^ s1_value;
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_lshrrev_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let d_value = s1_value >> (s0_value & 0x1F);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_lshlrev_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let d_value = s1_value << (s0_value & 0x1F);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_min_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let d_value = min_u32(s0_value, s1_value);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_max_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let d_value = max_u32(s0_value, s1_value);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_add_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vgpr(elem, s1));
        let d_value = s0_value + s1_value;
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_sub_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vgpr(elem, s1));
        let d_value = s0_value - s1_value;
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_mul_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vgpr(elem, s1));
        let d_value = s0_value * s1_value;
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_sub_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let (d_value, carry) = sub_u32(s0_value, s1_value, 0);
        cu.write_vgpr(elem, d, d_value);
        cu.set_vcc(elem, carry);
    }
}

fn v_subrev_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let (d_value, carry) = sub_u32(s1_value, s0_value, 0);
        cu.write_vgpr(elem, d, d_value);
        cu.set_vcc(elem, carry);
    }
}

fn v_subbrev_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let (d_value, carry) = sub_u32(s1_value, s0_value, cu.get_vcc(elem) as u32);
        cu.write_vgpr(elem, d, d_value);
        cu.set_vcc(elem, carry);
    }
}

fn v_mac_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vgpr(elem, s1));
        let d_value = u32_to_f32(cu.read_vgpr(elem, d));
        let d_value = s0_value * s1_value + d_value;
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_madak_f32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, k: f32) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vgpr(elem, s1));
        let d_value = fma(s0_value, s1_value, k);
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_min_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vgpr(elem, s1));
        let d_value = if s0_value < s1_value {
            s0_value
        } else {
            s1_value
        };
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_max_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vgpr(elem, s1));
        let d_value = if s0_value >= s1_value {
            s0_value
        } else {
            s1_value
        };
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_cmp_class_f32_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, abs: u8, neg: u8) {
    let s0_values = (0..64)
        .map(|elem| u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0))
        .collect::<Vec<f32>>();
    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s1))
        .collect::<Vec<u32>>();

    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem];
        let s1_value = s1_values[elem];
        let d_value = cmp_class_f32(s0_value, s1_value);
        cu.set_sgpr_bit(d, elem, d_value);
    }
}

fn v_cmp_op_f32_e64(
    cu: &mut ComputeUnit,
    op: OP16,
    d: usize,
    s0: usize,
    s1: usize,
    abs: u8,
    neg: u8,
) {
    let s0_values = (0..64)
        .map(|elem| u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0))
        .collect::<Vec<f32>>();

    let s1_values = (0..64)
        .map(|elem| u32_to_f32_abs_neg(cu.read_vop_src(elem, s1), abs, neg, 1))
        .collect::<Vec<f32>>();

    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem];
        let s1_value = s1_values[elem];
        let d_value = cmp_f32(s0_value, s1_value, op);
        cu.set_sgpr_bit(d, elem, d_value);
    }
}

fn v_cmp_op_u16_e64(cu: &mut ComputeUnit, op: OP8, d: usize, s0: usize, s1: usize) {
    let s0_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s0))
        .collect::<Vec<u32>>();

    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s1))
        .collect::<Vec<u32>>();

    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem] as u16;
        let s1_value = s1_values[elem] as u16;
        let d_value = cmp_u16(s0_value, s1_value, op);
        cu.set_sgpr_bit(d, elem, d_value);
    }
}

fn v_cmp_op_u32_e64(cu: &mut ComputeUnit, op: OP8, d: usize, s0: usize, s1: usize) {
    let s0_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s0))
        .collect::<Vec<u32>>();

    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s1))
        .collect::<Vec<u32>>();

    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem];
        let s1_value = s1_values[elem];
        let d_value = cmp_u32(s0_value, s1_value, op);
        cu.set_sgpr_bit(d, elem, d_value);
    }
}

fn v_cmp_op_i32_e64(cu: &mut ComputeUnit, op: OP8, d: usize, s0: usize, s1: usize) {
    let s0_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s0))
        .collect::<Vec<u32>>();

    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s1))
        .collect::<Vec<u32>>();

    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem] as i32;
        let s1_value = s1_values[elem] as i32;
        let d_value = cmp_i32(s0_value, s1_value, op);
        cu.set_sgpr_bit(d, elem, d_value);
    }
}

fn v_cndmask_b32_e64(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    abs: u8,
    neg: u8,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0);
        let s1_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s1), abs, neg, 1);
        let d_value = if cu.get_sgpr_bit(s2, elem) {
            s1_value
        } else {
            s0_value
        };
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_mac_f32_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, abs: u8, neg: u8) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0);
        let s1_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s1), abs, neg, 1);
        let d_value = u32_to_f32(cu.read_vgpr(elem, d));
        let d_value = s0_value * s1_value + d_value;
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_lshrrev_b32_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let d_value = s1_value >> (s0_value & 0x1F);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_lshlrev_b32_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let d_value = s1_value << (s0_value & 0x1F);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_max_f32_e64(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    abs: u8,
    neg: u8,
    clamp: bool,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0);
        let s1_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s1), abs, neg, 1);
        let d_value = if s0_value >= s1_value {
            s0_value
        } else {
            s1_value
        };
        cu.write_vgpr(elem, d, f32_to_u32_clamp(d_value, clamp));
    }
}

fn v_trunc_f32_e64(cu: &mut ComputeUnit, d: usize, s0: usize, abs: u8, neg: u8, clamp: bool) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0);
        let d_value = (s0_value as i32) as f32;
        cu.write_vgpr(elem, d, f32_to_u32_clamp(d_value, clamp));
    }
}

fn v_mad_f32_e64(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    abs: u8,
    neg: u8,
    clamp: bool,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0);
        let s1_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s1), abs, neg, 1);
        let s2_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s2), abs, neg, 2);
        let d_value = fma(s0_value, s1_value, s2_value);
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_add_f64_e64(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    abs: u8,
    neg: u8,
    clamp: bool,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1);
        let d_value = f64_to_u64(s0_value + s1_value);
        cu.write_vgpr_pair(elem, d, d_value);
    }
}

fn v_mul_f64_e64(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    abs: u8,
    neg: u8,
    clamp: bool,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1);
        let d_value = f64_to_u64(s0_value * s1_value);
        cu.write_vgpr_pair(elem, d, d_value);
    }
}

fn v_mul_lo_u32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let d_value = mul_u32(s0_value, s1_value);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_lshlrev_b64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src_pair(elem, s1);

        let d_value = s1_value << (s0_value & 0x3F);
        cu.write_vgpr_pair(elem, d, d_value);
    }
}

fn v_lshrrev_b64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src_pair(elem, s1);

        let d_value = s1_value >> (s0_value & 0x3F);
        cu.write_vgpr_pair(elem, d, d_value);
    }
}

fn v_bfe_u32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, s2: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let s2_value = cu.read_vop_src(elem, s2);
        let d_value = (s0_value >> (s1_value & 0x1F)) & ((1 << (s2_value & 0x1F)) - 1);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_bfe_i32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, s2: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0) as i32;
        let s1_value = cu.read_vop_src(elem, s1);
        let s2_value = cu.read_vop_src(elem, s2);
        let d_value = (s0_value >> (s1_value & 0x1F)) & ((1 << (s2_value & 0x1F)) - 1);
        cu.write_vgpr(elem, d, d_value as u32);
    }
}

fn v_bfi_b32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, s2: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let s2_value = cu.read_vop_src(elem, s2);
        let d_value = (s0_value & s1_value) | ((!s0_value) & s2_value);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_mul_hi_u32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let d_value = (((s0_value as u64) * (s1_value as u64)) >> 32) as u32;
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_alignbit_b32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, s2: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0) as u64;
        let s1_value = cu.read_vop_src(elem, s1) as u64;
        let s2_value = cu.read_vop_src(elem, s2);
        let d_value = ((((s1_value << 32) | s0_value) >> (s2_value & 0x1F)) & 0xFFFFFFFF) as u32;
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_fma_f64(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1);
        let s2_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s2), abs, neg, 2);
        let d_value = fma(s0_value, s1_value, s2_value);
        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
    }
}

fn v_div_fmas_f64(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1);
        let s2_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s2), abs, neg, 2);
        let d_value = if cu.get_vcc(elem) {
            64f64.exp2() * fma(s0_value, s1_value, s2_value)
        } else {
            fma(s0_value, s1_value, s2_value)
        };
        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
    }
}

fn v_div_fixup_f64(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1);
        let s2_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s2), abs, neg, 2);
        let d_value = div_fixup_f64(s0_value, s1_value, s2_value);
        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
    }
}

fn v_mad_i32_i24(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, s2: usize, clamp: bool) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0) as i64;
        let s1_value = cu.read_vop_src(elem, s1) as i64;
        let s2_value = cu.read_vop_src(elem, s2) as i64;
        let d_value = fma(s0_value, s1_value, s2_value) as i32;
        cu.write_vgpr(elem, d, d_value as u32);
    }
}

fn v_ldexp_f32(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0);
        let s1_value = cu.read_vop_src(elem, s1) as i32;
        let d_value = f32_to_u32(s0_value * (s1_value as f32).exp2());
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_writelane_b32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1) as usize;
    let d_value = s0_value;
    cu.write_vgpr(s1_value, d, d_value);
}

fn v_readlane_b32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s1_value = cu.read_sop_src(s1) as usize;
    let s0_value = cu.read_vop_src(s1_value, s0);
    let d_value = s0_value;
    cu.write_sgpr(d, d_value);
}

fn v_med3_f32(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    abs: u8,
    neg: u8,
    clamp: bool,
    omod: u8,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0);
        let s1_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s1), abs, neg, 1);
        let s2_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s2), abs, neg, 2);
        let d_value = if s0_value.is_nan() || s1_value.is_nan() || s2_value.is_nan() {
            s0_value.min(s1_value).min(s2_value)
        } else if s0_value.max(s1_value).max(s2_value) == s0_value {
            s1_value.max(s2_value)
        } else if s0_value.max(s1_value).max(s2_value) == s1_value {
            s0_value.max(s2_value)
        } else {
            s0_value.max(s1_value)
        };
        cu.write_vgpr(elem, d, f32_to_u32(d_value));
    }
}

fn v_add_u32_e64(cu: &mut ComputeUnit, d: usize, sdst: usize, s0: usize, s1: usize, clamp: bool) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let (d_value, carry) = add_u32(s0_value, s1_value, 0);
        cu.write_vgpr(elem, d, d_value);
        cu.set_sgpr_bit(sdst, elem, carry);
    }
}

fn v_sub_u32_e64(cu: &mut ComputeUnit, d: usize, sdst: usize, s0: usize, s1: usize, clamp: bool) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let (d_value, carry) = sub_u32(s0_value, s1_value, 0);
        cu.write_vgpr(elem, d, d_value);
        cu.set_sgpr_bit(sdst, elem, carry);
    }
}

fn v_addc_u32_e64(
    cu: &mut ComputeUnit,
    d: usize,
    sdst: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    clamp: bool,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let s2_value = cu.read_vop_src(elem, s2);
        let (d_value, carry) = add_u32(s0_value, s1_value, s2_value);
        cu.write_vgpr(elem, d, d_value);
        cu.set_sgpr_bit(sdst, elem, carry);
    }
}

fn v_div_scale_f32(cu: &mut ComputeUnit, d: usize, sdst: usize, s0: usize, s1: usize, s2: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vop_src(elem, s1));
        let s2_value = u32_to_f32(cu.read_vop_src(elem, s2));
        let (d_value, vcc) = div_scale_f32(s0_value, s1_value, s2_value);

        cu.write_vgpr(elem, d, f32_to_u32(d_value));
        cu.set_sgpr_bit(sdst, elem, vcc);
    }
}

fn v_div_scale_f64(cu: &mut ComputeUnit, d: usize, sdst: usize, s0: usize, s1: usize, s2: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let s1_value = u64_to_f64(cu.read_vop_src_pair(elem, s1));
        let s2_value = u64_to_f64(cu.read_vop_src_pair(elem, s2));
        let (d_value, vcc) = div_scale_f64(s0_value, s1_value, s2_value);

        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        cu.set_sgpr_bit(sdst, elem, vcc);
    }
}

fn v_cmp_f32(cu: &mut ComputeUnit, op: OP16, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vgpr(elem, s1));
        let result = cmp_f32(s0_value, s1_value, op);
        cu.set_vcc(elem, result);
    }
}

fn v_cmp_i32(cu: &mut ComputeUnit, op: OP8, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0) as i32;
        let s1_value = cu.read_vgpr(elem, s1) as i32;
        let result = cmp_i32(s0_value, s1_value, op);
        cu.set_vcc(elem, result);
    }
}

fn v_cmp_u32(cu: &mut ComputeUnit, op: OP8, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vgpr(elem, s1);
        let result = cmp_u32(s0_value, s1_value, op);
        cu.set_vcc(elem, result);
    }
}

fn flat_load_dword(cu: &mut ComputeUnit, d: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;
        let data = cu.read_mem_u32(addr_val);
        cu.write_vgpr(elem, d, data);
    }
}

fn flat_load_dwordx2(cu: &mut ComputeUnit, d: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;
        for i in 0..2 {
            let data = cu.read_mem_u32(addr_val + (i * 4) as u64);
            cu.write_vgpr(elem, d + i, data);
        }
    }
}

fn flat_load_dwordx4(cu: &mut ComputeUnit, d: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;
        for i in 0..4 {
            let data = cu.read_mem_u32(addr_val + (i * 4) as u64);
            cu.write_vgpr(elem, d + i, data);
        }
    }
}

fn flat_load_ubyte(cu: &mut ComputeUnit, d: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr);
        let data = cu.read_mem_u8(addr_val);
        cu.write_vgpr(elem, d, data as u32);
    }
}

fn flat_load_ushort(cu: &mut ComputeUnit, d: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;
        let data = cu.read_mem_u16(addr_val);
        cu.write_vgpr(elem, d, data as u32);
    }
}

fn flat_store_byte(cu: &mut ComputeUnit, s: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;
        let data = cu.vgprs.get(elem, s) as u8;
        cu.write_mem_u8(addr_val, data);
    }
}
fn flat_store_short(cu: &mut ComputeUnit, s: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;
        let data = cu.vgprs.get(elem, s) as u16;
        cu.write_mem_u16(addr_val, data);
    }
}
fn flat_store_dword(cu: &mut ComputeUnit, s: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;
        let data = cu.vgprs.get(elem, s);
        cu.write_mem_u32(addr_val, data);
    }
}
fn flat_store_dwordx4(cu: &mut ComputeUnit, s: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;

        for i in 0..4 {
            let data = cu.vgprs.get(elem, s + i);
            cu.write_mem_u32(addr_val + (i * 4) as u64, data);
        }
    }
}
fn buffer_load_dword(
    cu: &mut ComputeUnit,
    d: usize,
    vaddr: usize,
    srsrc: usize,
    soffset: usize,
    offset: u16,
    offen: bool,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let offset = if offen {
            cu.vgprs.get(elem, vaddr) as usize
        } else {
            offset as usize
        };
        let base_addr_val = cu.get_buffer_resource_constant_base(srsrc);
        let stride_val = cu.get_buffer_resource_constant_stride(srsrc);
        let soffset_val = cu.read_sop_src(soffset) as usize;

        let ptr = (base_addr_val + soffset_val + offset + stride_val * elem) as *const u32;

        let data = unsafe { *ptr };
        cu.write_vgpr(elem, d, data);
    }
}
fn buffer_store_dword(
    cu: &mut ComputeUnit,
    s: usize,
    vaddr: usize,
    srsrc: usize,
    soffset: usize,
    offset: u16,
    offen: bool,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let offset = if offen {
            cu.vgprs.get(elem, vaddr) as usize
        } else {
            offset as usize
        };
        let base_addr_val = cu.get_buffer_resource_constant_base(srsrc);
        let stride_val = cu.get_buffer_resource_constant_stride(srsrc);
        let soffset_val = cu.read_sop_src(soffset) as usize;

        let ptr = (base_addr_val + soffset_val + offset + stride_val * elem) as *mut u32;

        let data = cu.vgprs.get(elem, s);
        unsafe {
            *ptr = data;
        }
    }
}
impl ComputeUnit {
    pub fn new(pc: usize, insts: Vec<u8>, num_sgprs: usize, num_vgprs: usize) -> Self {
        // create instance
        ComputeUnit {
            pc: pc,
            next_pc: 0,
            insts: insts,
            sgprs: RegisterFileImpl::new(1, num_sgprs, 0),
            vgprs: RegisterFileImpl::new(64, num_vgprs, 0),
            scc: true,
            exec_lo: 0xFFFFFFFF,
            exec_hi: 0xFFFFFFFF,
            num_sgprs: num_sgprs,
            num_vgprs: num_vgprs,
        }
    }
    pub fn dispatch(&mut self, setup_data: ([u32; 16], [[u32; 64]; 16], usize)) {
        let (sgprs, vgprs, entry_addr) = setup_data;
        for i in 0..self.num_sgprs {
            self.sgprs.set(0, i, 0);
        }
        for i in 0..self.num_vgprs {
            for elem in 0..64 {
                self.vgprs.set(elem, i, 0);
            }
        }
        for i in 0..16 {
            self.sgprs.set(0, i, sgprs[i]);
        }
        for i in 0..16 {
            for elem in 0..64 {
                self.vgprs.set(elem, i, vgprs[i][elem]);
            }
        }

        self.pc = entry_addr;

        while self.step() {}
    }
    fn is_vccz(&self) -> bool {
        (self.get_vcc_hi() == 0) && (self.get_vcc_lo() == 0)
    }
    fn is_execz(&self) -> bool {
        (self.exec_hi == 0) && (self.exec_lo == 0)
    }
    fn is_vccnz(&self) -> bool {
        !self.is_vccz()
    }
    fn set_hw_reg(&mut self, id: usize, offset: usize, size: usize, value: u32) {
        // ignore
    }
    fn read_sgpr(&self, idx: usize) -> u32 {
        self.sgprs.get(0, idx)
    }
    fn read_sgpr_pair(&self, idx: usize) -> u64 {
        u64_from_u32_u32(self.read_sgpr(idx), self.read_sgpr(idx + 1))
    }
    fn write_sgpr(&mut self, idx: usize, value: u32) {
        // if idx == 8 {
        //     println!("Write s[{}] with {:08X} at {:012X}", idx, value, self.pc);
        // }
        self.sgprs.set(0, idx, value);
    }
    fn get_flat_scratch_lo(&self) -> u32 {
        self.read_sgpr(self.num_sgprs - 4)
    }
    fn get_flat_scratch_hi(&self) -> u32 {
        self.read_sgpr(self.num_sgprs - 3)
    }
    fn get_vcc_lo(&self) -> u32 {
        self.read_sgpr(self.num_sgprs - 2)
    }
    fn get_vcc_hi(&self) -> u32 {
        self.read_sgpr(self.num_sgprs - 1)
    }
    fn set_flat_scratch_lo(&mut self, value: u32) {
        self.write_sgpr(self.num_sgprs - 4, value);
    }
    fn set_flat_scratch_hi(&mut self, value: u32) {
        self.write_sgpr(self.num_sgprs - 3, value);
    }
    fn set_vcc_lo(&mut self, value: u32) {
        self.write_sgpr(self.num_sgprs - 2, value);
    }
    fn set_vcc_hi(&mut self, value: u32) {
        self.write_sgpr(self.num_sgprs - 1, value);
    }

    fn read_sop_src(&self, addr: usize) -> u32 {
        match addr {
            0..=101 => self.read_sgpr(addr),
            102 => self.get_flat_scratch_lo(),
            103 => self.get_flat_scratch_hi(),
            106 => self.get_vcc_lo(),
            107 => self.get_vcc_hi(),
            126 => self.exec_lo,
            127 => self.exec_hi,
            128 => 0,
            129..=192 => (addr - 128) as u32,
            193..=208 => (-((addr - 192) as i32)) as u32,
            240 => 0x3f000000, // 0.5
            241 => 0xbf000000, // -0.5
            242 => 0x3f800000, // 1.0
            243 => 0xbf800000, // -1.0
            244 => 0x40000000, // 2.0
            245 => 0xc0000000, // -2.0
            246 => 0x40800000, // 4.0
            247 => 0xc0800000, // -4.0
            248 => 0x3e22f983, // 1/(2*PI)
            255 => self.fetch_literal_constant(),
            _ => panic!(),
        }
    }
    fn read_sop_src_pair(&self, addr: usize) -> u64 {
        match addr {
            0..=101 => u64_from_u32_u32(self.read_sgpr(addr), self.read_sgpr(addr + 1)),
            102 => u64_from_u32_u32(self.get_flat_scratch_lo(), self.get_flat_scratch_hi()),
            106 => u64_from_u32_u32(self.get_vcc_lo(), self.get_vcc_hi()),
            126 => u64_from_u32_u32(self.exec_lo, self.exec_hi),
            128 => 0,
            129..=192 => (addr - 128) as u64,
            193..=208 => (-((addr - 192) as i64)) as u64,
            240 => 0x3fe0000000000000, // 0.5
            241 => 0xbfe0000000000000, // -0.5
            242 => 0x3ff0000000000000, // 1.0
            243 => 0xbff0000000000000, // -1.0
            244 => 0x4000000000000000, // 2.0
            245 => 0xc000000000000000, // -2.0
            246 => 0x4010000000000000, // 4.0
            247 => 0xc010000000000000, // -4.0
            248 => 0x3fc45f306dc8bdc4, // 1/(2*PI)
            255 => self.fetch_literal_constant() as u64,
            _ => panic!(),
        }
    }
    fn write_sop_dst(&mut self, addr: usize, value: u32) {
        match addr {
            0..=101 => self.write_sgpr(addr, value),
            102 => self.set_flat_scratch_lo(value),
            103 => self.set_flat_scratch_hi(value),
            106 => self.set_vcc_lo(value),
            107 => self.set_vcc_hi(value),
            126 => self.exec_lo = value,
            127 => self.exec_hi = value,
            _ => panic!(),
        }
    }
    fn write_sop_dst_pair(&mut self, addr: usize, value: u64) {
        self.write_sop_dst(addr, (value & 0xFFFFFFFF) as u32);
        self.write_sop_dst(addr + 1, ((value >> 32) & 0xFFFFFFFF) as u32);
    }
    fn read_vop_src(&self, elem: usize, addr: usize) -> u32 {
        match addr {
            0..=101 => self.read_sgpr(addr),
            128 => 0,
            129..=192 => (addr - 128) as u32,
            193..=208 => -((addr - 192) as i32) as u32,
            240 => 0x3f000000, // 0.5
            241 => 0xbf000000, // -0.5
            242 => 0x3f800000, // 1.0
            243 => 0xbf800000, // -1.0
            244 => 0x40000000, // 2.0
            245 => 0xc0000000, // -2.0
            246 => 0x40800000, // 4.0
            247 => 0xc0800000, // -4.0
            248 => 0x3e22f983, // 1/(2*PI)
            255 => self.fetch_literal_constant(),
            256..=511 => self.vgprs.get(elem, addr - 256),
            _ => panic!(),
        }
    }
    fn read_vop_src_pair(&self, elem: usize, addr: usize) -> u64 {
        match addr {
            0..=101 => self.read_sgpr_pair(addr),
            128 => 0,
            129..=192 => (addr - 128) as u64,
            193..=208 => (-((addr - 192) as i64)) as u64,
            240 => 0x3fe0000000000000, // 0.5
            241 => 0xbfe0000000000000, // -0.5
            242 => 0x3ff0000000000000, // 1.0
            243 => 0xbff0000000000000, // -1.0
            244 => 0x4000000000000000, // 2.0
            245 => 0xc000000000000000, // -2.0
            246 => 0x4010000000000000, // 4.0
            247 => 0xc010000000000000, // -4.0
            248 => 0x3fc45f306dc8bdc4, // 1/(2*PI)
            255 => self.fetch_literal_constant() as u64,
            256..=511 => self.read_vgpr_pair(elem, addr - 256),
            _ => panic!(),
        }
    }
    fn read_vgpr(&self, elem: usize, idx: usize) -> u32 {
        self.vgprs.get(elem, idx)
    }
    fn read_vgpr_pair(&self, elem: usize, idx: usize) -> u64 {
        u64_from_u32_u32(self.vgprs.get(elem, idx), self.vgprs.get(elem, idx + 1))
    }
    fn write_vgpr(&mut self, elem: usize, idx: usize, value: u32) {
        // if idx == 0 {
        //     println!(
        //         "Write v[{}][{}] with {:08X} at {:012X}",
        //         idx, elem, value, self.pc
        //     );
        // }
        self.vgprs.set(elem, idx, value);
    }
    fn write_vgpr_pair(&mut self, elem: usize, idx: usize, value: u64) {
        self.write_vgpr(elem, idx, (value & 0xFFFFFFFF) as u32);
        self.write_vgpr(elem, idx + 1, ((value >> 32) & 0xFFFFFFFF) as u32);
    }
    fn get_sgpr_bit(&self, idx: usize, bit: usize) -> bool {
        if bit >= 32 {
            let value = self.read_sop_src(idx + 1);
            ((value >> (bit - 32)) & 1) != 0
        } else {
            let value = self.read_sop_src(idx);
            ((value >> bit) & 1) != 0
        }
    }
    fn set_sgpr_bit(&mut self, idx: usize, bit: usize, value: bool) {
        if bit >= 32 {
            let mask = 1 << (bit - 32);
            let old_value = self.read_sop_src(idx + 1);
            self.write_sop_dst(
                idx + 1,
                (old_value & !mask) | ((value as u32) << (bit - 32)),
            );
        } else {
            let mask = 1 << bit;
            let old_value = self.read_sop_src(idx);
            self.write_sop_dst(idx, (old_value & !mask) | ((value as u32) << bit));
        }
    }
    fn set_vcc(&mut self, elem: usize, value: bool) {
        if elem >= 32 {
            let mask: u32 = 1 << (elem - 32);
            self.set_vcc_hi((self.get_vcc_hi() & !mask) | ((value as u32) << (elem - 32)));
        } else {
            let mask: u32 = 1 << elem;
            self.set_vcc_lo((self.get_vcc_lo() & !mask) | ((value as u32) << elem));
        }
    }
    fn get_vcc(&self, elem: usize) -> bool {
        if elem >= 32 {
            ((self.get_vcc_hi() >> (elem - 32)) & 1) != 0
        } else {
            ((self.get_vcc_lo() >> elem) & 1) != 0
        }
    }
    fn get_exec(&self, elem: usize) -> bool {
        if elem >= 32 {
            ((self.exec_hi >> (elem - 32)) & 1) != 0
        } else {
            ((self.exec_lo >> elem) & 1) != 0
        }
    }
    fn read_mem_u8(&mut self, addr: u64) -> u8 {
        let ptr = addr as *mut u8;
        unsafe { *ptr }
    }
    fn write_mem_u8(&mut self, addr: u64, data: u8) {
        let ptr = addr as *mut u8;
        unsafe {
            *ptr = data;
        }
    }
    fn read_mem_u16(&mut self, addr: u64) -> u16 {
        let ptr = addr as *mut u16;
        unsafe { *ptr }
    }
    fn write_mem_u16(&mut self, addr: u64, data: u16) {
        let ptr = addr as *mut u16;
        unsafe {
            *ptr = data;
        }
    }
    fn write_mem_u32(&mut self, addr: u64, data: u32) {
        let ptr = addr as *mut u32;
        unsafe {
            *ptr = data;
        }
    }
    fn read_mem_u32(&mut self, addr: u64) -> u32 {
        let ptr = addr as *mut u32;
        unsafe { *ptr }
    }
    fn get_buffer_resource_constant_base(&self, idx: usize) -> usize {
        let w0 = self.read_sgpr_pair(idx) as usize;
        w0 & ((1 << 48) - 1)
    }
    fn get_buffer_resource_constant_stride(&self, idx: usize) -> usize {
        let w0 = self.read_sgpr_pair(idx) as usize;
        (w0 >> 48) & ((1 << 14) - 1)
    }
    fn execute_sopp(&mut self, inst: SOPP) -> bool {
        let simm16 = inst.SIMM16 as i16;
        match inst.OP {
            I::S_NOP => {}
            I::S_ENDPGM => return false,
            I::S_WAITCNT => {}
            I::S_BARRIER => {}
            I::S_CBRANCH_VCCNZ => {
                if self.is_vccnz() {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_CBRANCH_VCCZ => {
                if self.is_vccz() {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_CBRANCH_EXECZ => {
                if self.is_execz() {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_BRANCH => {
                self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
            }
            I::S_CBRANCH_SCC0 => {
                if !self.scc {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_CBRANCH_SCC1 => {
                if self.scc {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_sopk(&mut self, inst: SOPK) -> bool {
        let simm16 = inst.SIMM16 as i16;
        let d = inst.SDST as usize;
        match inst.OP {
            I::S_MOVK_I32 => {
                self.write_sop_dst(d, (simm16 as i32) as u32);
            }
            I::S_SETREG_IMM32_B32 => {
                let size = (simm16 as u32) & 0x3F;
                let offset = ((simm16 as u32) >> 6) & 0x1F;
                let hw_reg_id = ((simm16 as u32) >> 11) & 0x1F;
                let value = self.fetch_literal_constant();
                self.set_hw_reg(hw_reg_id as usize, offset as usize, size as usize, value);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_sop1(&mut self, inst: SOP1) -> bool {
        let d = inst.SDST as usize;
        let s0 = inst.SSRC0 as usize;

        match inst.OP {
            I::S_MOV_B32 => {
                s_mov_b32(self, d, s0);
            }
            I::S_MOV_B64 => {
                s_mov_b64(self, d, s0);
            }
            I::S_BREV_B32 => {
                s_brev_b32(self, d, s0);
            }
            I::S_AND_SAVEEXEC_B64 => {
                s_and_saveexec_b64(self, d, s0);
            }
            I::S_OR_SAVEEXEC_B64 => {
                s_or_saveexec_b64(self, d, s0);
            }
            I::S_GETPC_B64 => {
                s_getpc_b64(self, d);
            }
            I::S_SWAPPC_B64 => {
                s_swappc_b64(self, d, s0);
            }
            I::S_SETPC_B64 => {
                s_setpc_b64(self, s0);
            }
            _ => unimplemented!(),
        }

        true
    }
    fn execute_sop2(&mut self, inst: SOP2) -> bool {
        let d = inst.SDST as usize;
        let s0 = inst.SSRC0 as usize;
        let s1 = inst.SSRC1 as usize;

        match inst.OP {
            I::S_ADD_U32 => {
                s_add_u32_e32(self, d, s0, s1);
            }
            I::S_SUB_U32 => {
                s_sub_u32_e32(self, d, s0, s1);
            }
            I::S_ADD_I32 => {
                s_add_i32_e32(self, d, s0, s1);
            }
            I::S_ADDC_U32 => {
                s_addc_u32_e32(self, d, s0, s1);
            }
            I::S_SUB_I32 => {
                s_sub_i32_e32(self, d, s0, s1);
            }
            I::S_AND_B32 => {
                s_and_b32_e32(self, d, s0, s1);
            }
            I::S_AND_B64 => {
                s_and_b64_e32(self, d, s0, s1);
            }
            I::S_ANDN2_B64 => {
                s_andn2_b64_e32(self, d, s0, s1);
            }
            I::S_OR_B64 => {
                s_or_b64_e32(self, d, s0, s1);
            }
            I::S_ORN2_B64 => {
                s_orn2_b64_e32(self, d, s0, s1);
            }
            I::S_XOR_B64 => {
                s_xor_b64_e32(self, d, s0, s1);
            }
            I::S_MUL_I32 => {
                s_mul_i32_e32(self, d, s0, s1);
            }
            I::S_LSHL_B32 => {
                s_lshl_i32_e32(self, d, s0, s1);
            }
            I::S_LSHR_B32 => {
                s_lshr_i32_e32(self, d, s0, s1);
            }
            I::S_BFM_B32 => {
                s_bfm_b32_e32(self, d, s0, s1);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_sopc(&mut self, inst: SOPC) -> bool {
        let s0 = inst.SSRC0 as usize;
        let s1 = inst.SSRC1 as usize;

        match inst.OP {
            I::S_CMP_EQ_U32 => {
                s_cmp_eq_u32_e32(self, s0, s1);
            }
            I::S_CMP_GT_I32 => {
                s_cmp_gt_i32_e32(self, s0, s1);
            }
            I::S_CMP_LT_I32 => {
                s_cmp_lt_i32_e32(self, s0, s1);
            }
            I::S_CMP_LG_U32 => {
                s_cmp_lg_u32_e32(self, s0, s1);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_smem(&mut self, inst: SMEM) -> bool {
        let sdata = inst.SDATA as usize;
        let soffset = inst.OFFSET as u64;
        let sbase = (inst.SBASE * 2) as usize;
        match inst.OP {
            I::S_LOAD_DWORD => {
                s_load_dword(self, sdata, sbase, soffset);
            }
            I::S_LOAD_DWORDX2 => {
                s_load_dwordx2(self, sdata, sbase, soffset);
            }
            I::S_LOAD_DWORDX4 => {
                s_load_dwordx4(self, sdata, sbase, soffset);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_vop1(&mut self, inst: VOP1) -> bool {
        let d = inst.VDST as usize;
        let s0 = inst.SRC0 as usize;
        match inst.OP {
            I::V_MOV_B32 => {
                v_mov_b32_e32(self, d, s0);
            }
            I::V_BFREV_B32 => {
                v_bfrev_b32_e32(self, d, s0);
            }
            I::V_RSQ_F32 => {
                v_rsq_f32_e32(self, d, s0);
            }
            I::V_RCP_F32 => {
                v_rcp_f32_e32(self, d, s0);
            }
            I::V_RCP_F64 => {
                v_rcp_f64_e32(self, d, s0);
            }
            I::V_SQRT_F32 => {
                v_sqrt_f32_e32(self, d, s0);
            }
            I::V_FFBH_U32 => {
                v_ffbh_u32_e32(self, d, s0);
            }
            I::V_CVT_F32_U32 => {
                v_cvt_f32_u32_e32(self, d, s0);
            }
            I::V_CVT_F64_I32 => {
                v_cvt_f64_i32_e32(self, d, s0);
            }
            I::V_CVT_F64_U32 => {
                v_cvt_f64_u32_e32(self, d, s0);
            }
            I::V_CVT_F64_F32 => {
                v_cvt_f64_f32_e32(self, d, s0);
            }
            I::V_CVT_F32_F64 => {
                v_cvt_f32_f64_e32(self, d, s0);
            }
            I::V_CVT_I32_F32 => {
                v_cvt_i32_f32_e32(self, d, s0);
            }
            I::V_CVT_F32_I32 => {
                v_cvt_f32_i32_e32(self, d, s0);
            }
            I::V_FREXP_MANT_F32 => {
                v_frexp_mant_f32_e32(self, d, s0);
            }
            I::V_FREXP_EXP_I32_F32 => {
                v_frexp_exp_i32_f32_e32(self, d, s0);
            }
            I::V_RNDNE_F32 => {
                v_rndne_f32_e32(self, d, s0);
            }
            I::V_FRACT_F32 => {
                v_fract_f32_e32(self, d, s0);
            }
            I::V_RCP_IFLAG_F32 => {
                v_rcp_iflag_f32_e32(self, d, s0);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_vop2(&mut self, inst: VOP2) -> bool {
        let d = inst.VDST as usize;
        let s0 = inst.SRC0 as usize;
        let s1 = inst.VSRC1 as usize;

        match inst.OP {
            I::V_CNDMASK_B32 => {
                v_cndmask_b32_e32(self, d, s0, s1);
            }
            I::V_ADD_U32 => {
                v_add_u32_e32(self, d, s0, s1);
            }
            I::V_ADDC_U32 => {
                v_addc_u32_e32(self, d, s0, s1);
            }
            I::V_AND_B32 => {
                v_and_b32_e32(self, d, s0, s1);
            }
            I::V_OR_B32 => {
                v_or_b32_e32(self, d, s0, s1);
            }
            I::V_XOR_B32 => {
                v_xor_b32_e32(self, d, s0, s1);
            }
            I::V_LSHRREV_B32 => {
                v_lshrrev_b32_e32(self, d, s0, s1);
            }
            I::V_LSHLREV_B32 => {
                v_lshlrev_b32_e32(self, d, s0, s1);
            }
            I::V_MIN_U32 => {
                v_min_u32_e32(self, d, s0, s1);
            }
            I::V_MAX_U32 => {
                v_max_u32_e32(self, d, s0, s1);
            }
            I::V_ADD_F32 => {
                v_add_f32_e32(self, d, s0, s1);
            }
            I::V_SUB_F32 => {
                v_sub_f32_e32(self, d, s0, s1);
            }
            I::V_MUL_F32 => {
                v_mul_f32_e32(self, d, s0, s1);
            }
            I::V_SUB_U32 => {
                v_sub_u32_e32(self, d, s0, s1);
            }
            I::V_SUBREV_U32 => {
                v_subrev_u32_e32(self, d, s0, s1);
            }
            I::V_SUBBREV_U32 => {
                v_subbrev_u32_e32(self, d, s0, s1);
            }
            I::V_MAC_F32 => {
                v_mac_f32_e32(self, d, s0, s1);
            }
            I::V_MADAK_F32 => {
                let k = u32_to_f32(self.fetch_literal_constant());
                v_madak_f32(self, d, s0, s1, k);
            }
            I::V_MIN_F32 => {
                v_min_f32_e32(self, d, s0, s1);
            }
            I::V_MAX_F32 => {
                v_max_f32_e32(self, d, s0, s1);
            }
            _ => unimplemented!(),
        }
        true
    }

    fn execute_vop3a(&mut self, inst: VOP3A) -> bool {
        let d = inst.VDST as usize;
        let s0 = inst.SRC0 as usize;
        let s1 = inst.SRC1 as usize;
        let s2 = inst.SRC2 as usize;
        let abs = inst.ABS;
        let neg = inst.NEG;
        let clamp = inst.CLAMP != 0;
        let omod = inst.OMOD;
        match inst.OP {
            I::V_CMP_CLASS_F32 => {
                v_cmp_class_f32_e64(self, d, s0, s1, abs, neg);
            }
            I::V_CMP_F32(op16) => {
                v_cmp_op_f32_e64(self, op16, d, s0, s1, abs, neg);
            }
            I::V_CMP_U16(op8) => {
                v_cmp_op_u16_e64(self, op8, d, s0, s1);
            }
            I::V_CMP_U32(op8) => {
                v_cmp_op_u32_e64(self, op8, d, s0, s1);
            }
            I::V_CMP_I32(op8) => {
                v_cmp_op_i32_e64(self, op8, d, s0, s1);
            }
            I::V_CNDMASK_B32 => {
                v_cndmask_b32_e64(self, d, s0, s1, s2, abs, neg);
            }
            I::V_MAC_F32 => {
                v_mac_f32_e64(self, d, s0, s1, abs, neg);
            }
            I::V_LSHRREV_B32 => {
                v_lshrrev_b32_e64(self, d, s0, s1);
            }
            I::V_LSHLREV_B32 => {
                v_lshlrev_b32_e64(self, d, s0, s1);
            }
            I::V_MAX_F32 => {
                v_max_f32_e64(self, d, s0, s1, abs, neg, clamp);
            }
            I::V_TRUNC_F32 => {
                v_trunc_f32_e64(self, d, s0, abs, neg, clamp);
            }
            I::V_MAD_F32 => {
                v_mad_f32_e64(self, d, s0, s1, s2, abs, neg, clamp);
            }
            I::V_ADD_F64 => {
                v_add_f64_e64(self, d, s0, s1, abs, neg, clamp);
            }
            I::V_MUL_F64 => {
                v_mul_f64_e64(self, d, s0, s1, abs, neg, clamp);
            }
            I::V_MUL_LO_U32 => {
                v_mul_lo_u32(self, d, s0, s1);
            }
            I::V_LSHLREV_B64 => {
                v_lshlrev_b64(self, d, s0, s1);
            }
            I::V_LSHRREV_B64 => {
                v_lshrrev_b64(self, d, s0, s1);
            }
            I::V_BFE_U32 => {
                v_bfe_u32(self, d, s0, s1, s2);
            }
            I::V_BFE_I32 => {
                v_bfe_i32(self, d, s0, s1, s2);
            }
            I::V_BFI_B32 => {
                v_bfi_b32(self, d, s0, s1, s2);
            }
            I::V_MUL_HI_U32 => {
                v_mul_hi_u32(self, d, s0, s1);
            }
            I::V_ALIGNBIT_B32 => {
                v_alignbit_b32(self, d, s0, s1, s2);
            }
            I::V_FMA_F64 => {
                v_fma_f64(self, d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_DIV_FMAS_F64 => {
                v_div_fmas_f64(self, d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_DIV_FIXUP_F64 => {
                v_div_fixup_f64(self, d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_MAD_I32_I24 => {
                v_mad_i32_i24(self, d, s0, s1, s2, clamp);
            }
            I::V_LDEXP_F32 => {
                v_ldexp_f32(self, d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_WRITELANE_B32 => {
                v_writelane_b32(self, d, s0, s1);
            }
            I::V_READLANE_B32 => {
                v_readlane_b32(self, d, s0, s1);
            }
            I::V_MED3_F32 => {
                v_med3_f32(self, d, s0, s1, s2, abs, neg, clamp, omod);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_vop3b(&mut self, inst: VOP3B) -> bool {
        let d = inst.VDST as usize;
        let sd = inst.SDST as usize;
        let s0 = inst.SRC0 as usize;
        let s1 = inst.SRC1 as usize;
        let s2 = inst.SRC2 as usize;
        let clamp = inst.CLAMP != 0;
        let neg = inst.NEG;
        let omod = inst.OMOD;
        match inst.OP {
            I::V_ADD_U32 => {
                v_add_u32_e64(self, d, sd, s0, s1, clamp);
            }
            I::V_SUB_U32 => {
                v_sub_u32_e64(self, d, sd, s0, s1, clamp);
            }
            I::V_ADDC_U32 => {
                v_addc_u32_e64(self, d, sd, s0, s1, s2, clamp);
            }
            I::V_DIV_SCALE_F32 => {
                v_div_scale_f32(self, d, sd, s0, s1, s2);
            }
            I::V_DIV_SCALE_F64 => {
                v_div_scale_f64(self, d, sd, s0, s1, s2);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_vopc(&mut self, inst: VOPC) -> bool {
        let s0 = inst.SRC0 as usize;
        let s1 = inst.VSRC1 as usize;

        match inst.OP {
            I::V_CMP_F32(op16) => {
                v_cmp_f32(self, op16, s0, s1);
            }
            I::V_CMP_I32(op8) => {
                v_cmp_i32(self, op8, s0, s1);
            }
            I::V_CMP_U32(op8) => {
                v_cmp_u32(self, op8, s0, s1);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_flat(&mut self, inst: FLAT) -> bool {
        let s = inst.DATA as usize;
        let d = inst.VDST as usize;
        let addr = inst.ADDR as usize;

        // Flat scratch memory is not supported yet.
        match inst.OP {
            I::FLAT_LOAD_DWORD => {
                flat_load_dword(self, d, addr);
            }
            I::FLAT_LOAD_DWORDX2 => {
                flat_load_dwordx2(self, d, addr);
            }
            I::FLAT_LOAD_DWORDX4 => {
                flat_load_dwordx4(self, d, addr);
            }
            I::FLAT_LOAD_UBYTE => {
                flat_load_ubyte(self, d, addr);
            }
            I::FLAT_LOAD_USHORT => {
                flat_load_ushort(self, d, addr);
            }
            I::FLAT_STORE_BYTE => {
                flat_store_byte(self, s, addr);
            }
            I::FLAT_STORE_SHORT => {
                flat_store_short(self, s, addr);
            }
            I::FLAT_STORE_DWORD => {
                flat_store_dword(self, s, addr);
            }
            I::FLAT_STORE_DWORDX4 => {
                flat_store_dwordx4(self, s, addr);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn execute_mubuf(&mut self, inst: MUBUF) -> bool {
        let d = inst.VDATA as usize;
        let s = inst.VDATA as usize;
        let vaddr = inst.VADDR as usize;
        let srsrc = inst.SRSRC as usize;
        let soffset = inst.SOFFSET as usize;
        let offset = inst.OFFSET;
        let offen = inst.OFFEN != 0;
        match inst.OP {
            I::BUFFER_LOAD_DWORD => {
                buffer_load_dword(self, d, vaddr, srsrc, soffset, offset, offen);
            }
            I::BUFFER_STORE_DWORD => {
                buffer_store_dword(self, s, vaddr, srsrc, soffset, offset, offen);
            }
            _ => unimplemented!(),
        }
        true
    }
    fn get_pc(&self) -> u64 {
        (&self.insts[self.pc] as *const u8) as u64
    }
    fn set_pc(&mut self, value: u64) {
        let base_ptr = (&self.insts[0] as *const u8) as u64;
        self.pc = (value - base_ptr) as usize;
    }
    fn execute_inst(&mut self, inst: InstFormat) -> bool {
        // println!("{:012X}: {:?}", self.pc, inst);
        // if self.pc == 0x000000001B58 {
        //     println!("");
        // }
        match inst {
            InstFormat::SOP1(fields) => self.execute_sop1(fields),
            InstFormat::SOP2(fields) => self.execute_sop2(fields),
            InstFormat::SOPP(fields) => self.execute_sopp(fields),
            InstFormat::SOPK(fields) => self.execute_sopk(fields),
            InstFormat::SOPC(fields) => self.execute_sopc(fields),
            InstFormat::SMEM(fields) => self.execute_smem(fields),
            InstFormat::VOP1(fields) => self.execute_vop1(fields),
            InstFormat::VOP2(fields) => self.execute_vop2(fields),
            InstFormat::VOP3A(fields) => self.execute_vop3a(fields),
            InstFormat::VOP3B(fields) => self.execute_vop3b(fields),
            InstFormat::VOPC(fields) => self.execute_vopc(fields),
            InstFormat::FLAT(fields) => self.execute_flat(fields),
            InstFormat::MUBUF(fields) => self.execute_mubuf(fields),
            _ => unimplemented!(),
        }
    }

    fn fetch_inst(&mut self) -> u64 {
        get_u64(&self.insts, self.pc)
    }
    fn fetch_literal_constant(&self) -> u32 {
        get_u32(&self.insts, self.pc + 4)
    }
}
