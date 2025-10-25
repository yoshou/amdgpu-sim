use crate::buffer::*;
use crate::gcn3_decoder::*;
use crate::gcn_instructions::*;
use crate::instructions::*;
use crate::processor::*;

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

pub enum Signals {
    None,
    EndOfProgram,
    Switch,
    Unknown,
}

pub trait Processor {
    fn step(&mut self) -> Signals;
}

#[derive(Copy, Clone, Debug)]
struct Context {
    id: usize,
    pc: usize,
    scc: bool,
    exec_lo: u32,
    exec_hi: u32,
    m0: u32,
}

struct ComputeUnit {
    ctx: Context,
    next_pc: usize,
    insts: Vec<u8>,
    pub sgprs: RegisterFileImpl<u32>,
    pub vgprs: RegisterFileImpl<u32>,
    num_sgprs: usize,
    num_vgprs: usize,
    pub lds: Vec<u8>,
}

impl Processor for ComputeUnit {
    fn step(&mut self) -> Signals {
        if let Ok((inst, size)) = decode_gcn3(self.fetch_inst()) {
            self.next_pc = self.get_pc() as usize + size;
            let result = self.execute_inst(inst);
            self.set_pc(self.next_pc as u64);
            result
        } else {
            println!("Unknown instruction.");
            Signals::Unknown
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

fn cmp_f64(a: f64, b: f64, op: OP16) -> bool {
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
    }
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
    }
}

fn cmp_u64(a: u64, b: u64, op: OP8) -> bool {
    match op {
        OP8::F => false,
        OP8::LT => a < b,
        OP8::EQ => a == b,
        OP8::LE => a <= b,
        OP8::GT => a > b,
        OP8::LG => a != b,
        OP8::GE => a >= b,
        OP8::TRU => true,
    }
}

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
fn f32_to_u32(value: f32) -> u32 {
    f32::to_bits(value)
}

#[inline(always)]
fn u64_to_f64(value: u64) -> f64 {
    f64::from_bits(value)
}

#[inline(always)]
fn f64_to_u64(value: f64) -> u64 {
    f64::to_bits(value)
}

#[inline(always)]
fn u32_to_f32_abs_neg(value: u32, abs: u8, neg: u8, idx: usize) -> f32 {
    let result = f32::from_bits(value);
    abs_neg(result, abs, neg, idx)
}

#[inline(always)]
fn u64_to_f64_abs_neg(value: u64, abs: u8, neg: u8, idx: usize) -> f64 {
    let result = f64::from_bits(value);
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

fn clamp_f64(value: f64, min_value: f64, max_value: f64) -> f64 {
    if value < min_value {
        min_value
    } else if value > max_value {
        max_value
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
    let value = if clamp {
        clamp_f32(value, 0.0, 1.0)
    } else {
        value
    };
    f32::to_bits(value)
}

fn f64_to_u64_omod_clamp(value: f64, omod: u8, clamp: bool) -> u64 {
    let value = match omod {
        1 => value * 2.0,
        2 => value * 4.0,
        3 => value * 0.5,
        _ => value,
    };
    let value = if clamp {
        clamp_f64(value, 0.0, 1.0)
    } else {
        value
    };
    f64::to_bits(value)
}

#[inline(always)]
fn u64_from_u32_u32(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | (lo as u64)
}

fn get_exp_f64(val: f64) -> i16 {
    let bits: u64 = f64::to_bits(val);
    ((bits >> 52) & 0x7ff) as i16
}

fn get_exp_f32(val: f32) -> i16 {
    let bits: u32 = f32::to_bits(val);
    ((bits >> 23) & 0xff) as i16
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

fn s_movk_i32(cu: &mut ComputeUnit, d: usize, simm16: i16) {
    cu.write_sop_dst(d, (simm16 as i32) as u32);
}

fn s_setreg_imm32_b32(cu: &mut ComputeUnit, simm16: i16, imm32: u32) {
    let size = (simm16 as u32) & 0x3F;
    let offset = ((simm16 as u32) >> 6) & 0x1F;
    let hw_reg_id = ((simm16 as u32) >> 11) & 0x1F;
    cu.set_hw_reg(hw_reg_id as usize, offset as usize, size as usize, imm32);
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

fn s_ff1_i32_b32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src(s0);
    let d_value = s0_value.trailing_zeros();
    let d_value = if s0_value == 0 { -1i32 } else { d_value as i32 };
    cu.write_sop_dst(d, d_value as u32);
}

fn s_and_saveexec_b64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let exec_value = u64_from_u32_u32(cu.ctx.exec_lo, cu.ctx.exec_hi);

    cu.write_sop_dst_pair(d, exec_value);

    let exec_value = s0_value & exec_value;

    cu.ctx.exec_lo = (exec_value & 0xFFFFFFFF) as u32;
    cu.ctx.exec_hi = ((exec_value >> 32) & 0xFFFFFFFF) as u32;
    cu.ctx.scc = exec_value != 0;
}

fn s_or_saveexec_b64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let exec_value = u64_from_u32_u32(cu.ctx.exec_lo, cu.ctx.exec_hi);

    cu.write_sop_dst_pair(d, exec_value);

    let exec_value = s0_value | exec_value;

    cu.ctx.exec_lo = (exec_value & 0xFFFFFFFF) as u32;
    cu.ctx.exec_hi = ((exec_value >> 32) & 0xFFFFFFFF) as u32;
    cu.ctx.scc = exec_value != 0;
}

fn s_andn2_saveexec_b64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let exec_value = u64_from_u32_u32(cu.ctx.exec_lo, cu.ctx.exec_hi);

    cu.write_sop_dst_pair(d, exec_value);

    let exec_value = s0_value & !exec_value;

    cu.ctx.exec_lo = (exec_value & 0xFFFFFFFF) as u32;
    cu.ctx.exec_hi = ((exec_value >> 32) & 0xFFFFFFFF) as u32;
    cu.ctx.scc = exec_value != 0;
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
    cu.ctx.scc = carry;
}

fn s_sub_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let (d_value, carry) = sub_u32(s0_value, s1_value, 0);
    cu.write_sop_dst(d, d_value as u32);
    cu.ctx.scc = carry;
}

fn s_max_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = max_u32(s0_value, s1_value);
    cu.write_sop_dst(d, d_value as u32);
    cu.ctx.scc = d_value == s0_value;
}

fn s_add_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    let (d_value, overflow) = add_i32(s0_value, s1_value);
    cu.write_sop_dst(d, d_value as u32);
    cu.ctx.scc = overflow;
}

fn s_addc_u32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let (d_value, carry) = add_u32(s0_value, s1_value, cu.ctx.scc as u32);
    cu.write_sop_dst(d, d_value as u32);
    cu.ctx.scc = carry;
}

fn s_sub_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    let (d_value, overflow) = sub_i32(s0_value, s1_value);
    cu.write_sop_dst(d, d_value as u32);
    cu.ctx.scc = overflow;
}

fn s_and_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = s0_value & s1_value;
    cu.write_sop_dst(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_and_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value & s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_andn2_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value & !s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_or_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value | s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_orn2_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value | !s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_xor_b64_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = s0_value ^ s1_value;
    cu.write_sop_dst_pair(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_mul_i32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    let d_value = mul_i32(s0_value, s1_value);
    cu.write_sop_dst(d, d_value as u32);
}

fn s_lshl_b32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = s0_value << ((s1_value) & 0x1F);
    cu.write_sop_dst(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_lshr_b32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = s0_value >> (s1_value & 0x1F);
    cu.write_sop_dst(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_bfm_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = ((1 << ((s0_value) & 0x1F)) - 1) << ((s1_value) & 0x1F);
    cu.write_sop_dst(d, d_value);
}

fn s_cselect_b32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = if cu.ctx.scc { s0_value } else { s1_value };
    cu.write_sop_dst(d, d_value);
}

fn s_lshl_b64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = s0_value << ((s1_value) & 0x3F);
    cu.write_sop_dst_pair(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_lshr_b64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src(s1);
    let d_value = s0_value >> ((s1_value) & 0x3F);
    cu.write_sop_dst_pair(d, d_value);
    cu.ctx.scc = d_value != 0;
}

fn s_cselect_b64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    let d_value = if cu.ctx.scc { s0_value } else { s1_value };
    cu.write_sop_dst_pair(d, d_value);
}

fn s_cmp_eq_u32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    cu.ctx.scc = s0_value == s1_value;
}

fn s_cmp_lt_u32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    cu.ctx.scc = s0_value < s1_value;
}

fn s_cmp_ge_u32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    cu.ctx.scc = s0_value >= s1_value;
}

fn s_cmp_gt_i32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    cu.ctx.scc = s0_value > s1_value;
}

fn s_cmp_lt_i32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0) as i32;
    let s1_value = cu.read_sop_src(s1) as i32;
    cu.ctx.scc = s0_value < s1_value;
}

fn s_cmp_lg_u32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    cu.ctx.scc = s0_value != s1_value;
}

fn s_cmp_gt_u32_e32(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src(s0);
    let s1_value = cu.read_sop_src(s1);
    cu.ctx.scc = s0_value > s1_value;
}

fn s_cmp_ne_u64(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    cu.ctx.scc = s0_value != s1_value;
}

fn s_cmp_eq_u64(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    let s0_value = cu.read_sop_src_pair(s0);
    let s1_value = cu.read_sop_src_pair(s1);
    cu.ctx.scc = s0_value == s1_value;
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

fn v_not_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let d_value = !s0_value;
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

fn v_rsq_f64_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let d_value = 1.0 / s0_value.sqrt();

        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
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

fn v_cvt_i32_f64_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let d_value = s0_value as i32;

        cu.write_vgpr(elem, d, d_value as u32);
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

fn v_rndne_f64_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let mut d_value = (s0_value + 0.5).floor();
        if s0_value.floor() % 2.0 == 0.0 && s0_value.fract() == 0.5 {
            d_value -= 1.0;
        }

        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
    }
}

fn v_readfirstlane_b32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    let exec_value = u64_from_u32_u32(cu.ctx.exec_lo, cu.ctx.exec_hi);
    let lane = (exec_value.trailing_zeros() & 0x3F) as usize;
    let s0_value = cu.read_vop_src(lane, s0);
    let d_value = s0_value;
    cu.write_sgpr(d, d_value);
}

fn v_cvt_u32_f32_e32(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let d_value = s0_value as u32;

        cu.write_vgpr(elem, d, d_value);
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

fn v_add_u16_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0) as u16;
        let s1_value = cu.read_vop_src(elem, s1) as u16;
        let d_value = s0_value.saturating_add(s1_value);
        cu.write_vgpr(elem, d, d_value as u32);
    }
}

fn v_cmp_class_f64_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, abs: u8, neg: u8) {
    let s0_values = (0..64)
        .map(|elem| u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0))
        .collect::<Vec<f64>>();
    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s1))
        .collect::<Vec<u32>>();

    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem];
        let s1_value = s1_values[elem];
        let d_value = cmp_class_f64(s0_value, s1_value);
        vcc |= (d_value as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_cmp_class_f32_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, abs: u8, neg: u8) {
    let s0_values = (0..64)
        .map(|elem| u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0))
        .collect::<Vec<f32>>();
    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s1))
        .collect::<Vec<u32>>();

    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem];
        let s1_value = s1_values[elem];
        let d_value = cmp_class_f32(s0_value, s1_value);
        vcc |= (d_value as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_cmp_op_f64_e64(
    cu: &mut ComputeUnit,
    op: OP16,
    d: usize,
    s0: usize,
    s1: usize,
    abs: u8,
    neg: u8,
) {
    let s0_values = (0..64)
        .map(|elem| u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0))
        .collect::<Vec<f64>>();

    let s1_values = (0..64)
        .map(|elem| u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1))
        .collect::<Vec<f64>>();

    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem];
        let s1_value = s1_values[elem];
        let d_value = cmp_f64(s0_value, s1_value, op);
        vcc |= (d_value as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
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

    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem];
        let s1_value = s1_values[elem];
        let d_value = cmp_f32(s0_value, s1_value, op);
        vcc |= (d_value as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_cmp_op_u16_e64(cu: &mut ComputeUnit, op: OP8, d: usize, s0: usize, s1: usize) {
    let s0_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s0))
        .collect::<Vec<u32>>();

    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s1))
        .collect::<Vec<u32>>();

    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem] as u16;
        let s1_value = s1_values[elem] as u16;
        let d_value = cmp_u16(s0_value, s1_value, op);
        vcc |= (d_value as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_cmp_op_u32_e64(cu: &mut ComputeUnit, op: OP8, d: usize, s0: usize, s1: usize) {
    let s0_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s0))
        .collect::<Vec<u32>>();

    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s1))
        .collect::<Vec<u32>>();

    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem];
        let s1_value = s1_values[elem];
        let d_value = cmp_u32(s0_value, s1_value, op);
        vcc |= (d_value as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_cmp_op_u64_e64(cu: &mut ComputeUnit, op: OP8, d: usize, s0: usize, s1: usize) {
    let s0_values = (0..64)
        .map(|elem| cu.read_vop_src_pair(elem, s0))
        .collect::<Vec<u64>>();

    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src_pair(elem, s1))
        .collect::<Vec<u64>>();

    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem];
        let s1_value = s1_values[elem];
        let d_value = cmp_u64(s0_value, s1_value, op);
        vcc |= (d_value as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_cmp_op_i32_e64(cu: &mut ComputeUnit, op: OP8, d: usize, s0: usize, s1: usize) {
    let s0_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s0))
        .collect::<Vec<u32>>();

    let s1_values = (0..64)
        .map(|elem| cu.read_vop_src(elem, s1))
        .collect::<Vec<u32>>();

    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = s0_values[elem] as i32;
        let s1_value = s1_values[elem] as i32;
        let d_value = cmp_i32(s0_value, s1_value, op);
        vcc |= (d_value as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
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

fn v_and_b32_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let d_value = s0_value & s1_value;
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_or_b32_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let d_value = s0_value | s1_value;
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_xor_b32_e64(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let d_value = s1_value ^ s0_value;
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
    omod: u8,
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
        cu.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
    }
}

fn v_trunc_f32_e64(
    cu: &mut ComputeUnit,
    d: usize,
    s0: usize,
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
        let d_value = (s0_value as i32) as f32;
        cu.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
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
    omod: u8,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s0), abs, neg, 0);
        let s1_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s1), abs, neg, 1);
        let s2_value = u32_to_f32_abs_neg(cu.read_vop_src(elem, s2), abs, neg, 2);
        let d_value = fma(s0_value, s1_value, s2_value);
        cu.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
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
    omod: u8,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1);
        let d_value = s0_value + s1_value;
        cu.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
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
    omod: u8,
) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1);
        let d_value = s0_value * s1_value;
        cu.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
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

fn v_ashrrev_i32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1) as i32;
        let d_value = s1_value >> (s0_value & 0x1F);
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

fn v_max_u32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let d_value = s0_value.max(s1_value);
        cu.write_vgpr(elem, d, d_value);
    }
}

fn v_min_u32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let d_value = s0_value.min(s1_value);
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
        cu.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
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
        cu.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
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
        cu.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
    }
}

fn v_mad_i32_i24(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, s2: usize) {
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
        let d_value = s0_value * (s1_value as f32).exp2();
        cu.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
    }
}

fn v_ldexp_f64(
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
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = cu.read_vop_src(elem, s1) as i32;
        let d_value = s0_value * (s1_value as f64).exp2();
        cu.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
    }
}

fn v_max_f64(
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
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1);
        let d_value = s0_value.max(s1_value);
        cu.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
    }
}

fn v_min_f64(
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
        let s0_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s0), abs, neg, 0);
        let s1_value = u64_to_f64_abs_neg(cu.read_vop_src_pair(elem, s1), abs, neg, 1);
        let d_value = s0_value.min(s1_value);
        cu.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
    }
}

fn v_rsq_f64_e64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let d_value = 1.0 / s0_value.sqrt();

        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
    }
}

fn v_rcp_f64_e64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let d_value = 1.0 / s0_value;

        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
    }
}

fn v_cvt_f64_u32_e64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let d_value = f64_to_u64(s0_value as f64);

        cu.write_vgpr_pair(elem, d, d_value);
    }
}

fn v_cvt_i32_f64_e64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let d_value = s0_value as i32;

        cu.write_vgpr(elem, d, d_value as u32);
    }
}

fn v_rndne_f64_e64(cu: &mut ComputeUnit, d: usize, s0: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let mut d_value = (s0_value + 0.5).floor();
        if s0_value.floor() % 2.0 == 0.0 && s0_value.fract() == 0.5 {
            d_value -= 1.0;
        }

        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
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
        cu.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
    }
}

fn v_mad_u64_u32(cu: &mut ComputeUnit, d: usize, s0: usize, s1: usize, s2: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let s2_value = cu.read_vop_src_pair(elem, s2);
        let (d_value, carry) = (s0_value as u64 * s1_value as u64).overflowing_add(s2_value);
        cu.write_vgpr_pair(elem, d, d_value);
        cu.set_vcc(elem, carry);
    }
}

fn v_add_u32_e64(cu: &mut ComputeUnit, d: usize, sdst: usize, s0: usize, s1: usize) {
    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let (d_value, carry) = add_u32(s0_value, s1_value, 0);
        cu.write_vgpr(elem, d, d_value);
        vcc |= (carry as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(sdst, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_sub_u32_e64(cu: &mut ComputeUnit, d: usize, sdst: usize, s0: usize, s1: usize) {
    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let (d_value, carry) = sub_u32(s0_value, s1_value, 0);
        cu.write_vgpr(elem, d, d_value);
        vcc |= (carry as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(sdst, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_addc_u32_e64(cu: &mut ComputeUnit, d: usize, sdst: usize, s0: usize, s1: usize, s2: usize) {
    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = cu.read_vop_src(elem, s0);
        let s1_value = cu.read_vop_src(elem, s1);
        let s2_value = cu.get_sgpr_bit(s2, elem) as u32;
        let (d_value, carry) = add_u32(s0_value, s1_value, s2_value);
        cu.write_vgpr(elem, d, d_value);
        vcc |= (carry as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(sdst, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_div_scale_f32(cu: &mut ComputeUnit, d: usize, sdst: usize, s0: usize, s1: usize, s2: usize) {
    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u32_to_f32(cu.read_vop_src(elem, s0));
        let s1_value = u32_to_f32(cu.read_vop_src(elem, s1));
        let s2_value = u32_to_f32(cu.read_vop_src(elem, s2));
        let (d_value, flag) = div_scale_f32(s0_value, s1_value, s2_value);

        cu.write_vgpr(elem, d, f32_to_u32(d_value));
        vcc |= (flag as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(sdst, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_div_scale_f64(cu: &mut ComputeUnit, d: usize, sdst: usize, s0: usize, s1: usize, s2: usize) {
    let mut vcc = 0u64;
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let s1_value = u64_to_f64(cu.read_vop_src_pair(elem, s1));
        let s2_value = u64_to_f64(cu.read_vop_src_pair(elem, s2));
        let (d_value, flag) = div_scale_f64(s0_value, s1_value, s2_value);

        cu.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        vcc |= (flag as u64) << elem;
    }
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        cu.set_sgpr_bit(sdst, elem, ((vcc >> elem) & 1) != 0);
    }
}

fn v_cmp_f64(cu: &mut ComputeUnit, op: OP16, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let s1_value = u64_to_f64(cu.read_vgpr_pair(elem, s1));
        let result = cmp_f64(s0_value, s1_value, op);
        cu.set_vcc(elem, result);
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

fn v_cmp_class_f64(cu: &mut ComputeUnit, s0: usize, s1: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let s0_value = u64_to_f64(cu.read_vop_src_pair(elem, s0));
        let s1_value = cu.read_vgpr(elem, s1);
        let result = cmp_class_f64(s0_value, s1_value);
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
        let data = cu.read_vgpr(elem, s) as u8;
        cu.write_mem_u8(addr_val, data);
    }
}
fn flat_store_short(cu: &mut ComputeUnit, s: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;
        let data = cu.read_vgpr(elem, s) as u16;
        cu.write_mem_u16(addr_val, data);
    }
}
fn flat_store_dword(cu: &mut ComputeUnit, s: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;
        let data = cu.read_vgpr(elem, s);
        cu.write_mem_u32(addr_val, data);
    }
}
fn flat_store_dwordx2(cu: &mut ComputeUnit, s: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;

        for i in 0..2 {
            let data = cu.read_vgpr(elem, s + i);
            cu.write_mem_u32(addr_val + (i * 4) as u64, data);
        }
    }
}
fn flat_store_dwordx4(cu: &mut ComputeUnit, s: usize, addr: usize) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr_pair(elem, addr) as u64;

        for i in 0..4 {
            let data = cu.read_vgpr(elem, s + i);
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
            cu.read_vgpr(elem, vaddr) as usize
        } else {
            offset as usize
        };
        let base_addr_val = cu.get_buffer_resource_constant_base(srsrc * 4);
        let stride_val = cu.get_buffer_resource_constant_stride(srsrc * 4);
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
            cu.read_vgpr(elem, vaddr) as usize
        } else {
            offset as usize
        };
        let base_addr_val = cu.get_buffer_resource_constant_base(srsrc * 4);
        let stride_val = cu.get_buffer_resource_constant_stride(srsrc * 4);
        let soffset_val = cu.read_sop_src(soffset) as usize;

        let ptr = (base_addr_val + soffset_val + offset + stride_val * elem) as *mut u32;

        let data = cu.read_vgpr(elem, s);
        unsafe {
            *ptr = data;
        }
    }
}
fn ds_write_b8(cu: &mut ComputeUnit, d0: usize, addr: usize, offset: u8) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr(elem, addr) + offset as u32;
        let addr_val = addr_val.min(cu.ctx.m0);
        let data = cu.read_vgpr(elem, d0) as u8;
        cu.write_lds_u8(addr_val, data);
    }
}
fn ds_read_u8(cu: &mut ComputeUnit, r: usize, addr: usize, offset: u8) {
    for elem in 0..64 {
        if !cu.get_exec(elem) {
            continue;
        }
        let addr_val = cu.read_vgpr(elem, addr) + offset as u32;
        let addr_val = addr_val.min(cu.ctx.m0);
        let data = cu.read_lds_u8(addr_val);
        cu.write_vgpr(elem, r, data as u32);
    }
}
impl ComputeUnit {
    pub fn new(pc: usize, insts: Vec<u8>, num_sgprs: usize, num_vgprs: usize) -> Self {
        // create instance
        ComputeUnit {
            ctx: Context {
                id: 0,
                pc: pc,
                scc: false,
                exec_lo: 0xFFFFFFFF,
                exec_hi: 0xFFFFFFFF,
                m0: 0,
            },
            next_pc: 0,
            insts: insts,
            sgprs: RegisterFileImpl::new(1, 512, 0),
            vgprs: RegisterFileImpl::new(64, 256, 0),
            num_sgprs: num_sgprs,
            num_vgprs: num_vgprs,
            lds: vec![0; 32 * 1024],
        }
    }
    pub fn dispatch(
        &mut self,
        entry_addr: usize,
        setup_data: Vec<([u32; 16], [[u32; 64]; 16])>,
        num_wavefronts: usize,
    ) {
        for i in 0..self.num_sgprs {
            self.sgprs.set(0, i, 0);
        }
        for i in 0..self.num_vgprs {
            for elem in 0..64 {
                self.vgprs.set(elem, i, 0);
            }
        }
        for wavefront in 0..num_wavefronts {
            let (sgprs, vgprs) = setup_data[wavefront];
            for i in 0..16 {
                self.sgprs.set(0, wavefront * self.num_sgprs + i, sgprs[i]);
            }
            for i in 0..16 {
                for elem in 0..64 {
                    self.vgprs
                        .set(elem, wavefront * self.num_vgprs + i, vgprs[i][elem]);
                }
            }
        }

        use std::collections::VecDeque;

        let mut ctxs = VecDeque::new();
        for wavefront in 0..num_wavefronts {
            ctxs.push_back(Context {
                id: wavefront,
                pc: entry_addr,
                exec_lo: 0xFFFFFFFF,
                exec_hi: 0xFFFFFFFF,
                scc: false,
                m0: 0,
            })
        }

        let is_signal_none = |signal: &Signals| match signal {
            Signals::None => true,
            _ => false,
        };

        while !ctxs.is_empty() {
            if let Some(ctx) = ctxs.pop_front() {
                self.ctx = ctx;
            }
            let mut signal = self.step();
            while is_signal_none(&signal) {
                signal = self.step();
            }

            match signal {
                Signals::EndOfProgram => {}
                Signals::Switch => ctxs.push_back(self.ctx),
                _ => panic!(),
            }
        }
    }
    fn is_vccz(&self) -> bool {
        (self.get_vcc_hi() == 0) && (self.get_vcc_lo() == 0)
    }
    fn is_execz(&self) -> bool {
        (self.ctx.exec_hi == 0) && (self.ctx.exec_lo == 0)
    }
    fn is_vccnz(&self) -> bool {
        !self.is_vccz()
    }
    fn is_execnz(&self) -> bool {
        !self.is_execz()
    }
    fn set_hw_reg(&mut self, _id: usize, _offset: usize, _size: usize, _value: u32) {
        // ignore
    }
    fn read_sgpr(&self, idx: usize) -> u32 {
        if idx >= self.num_sgprs {
            panic!();
        }
        self.sgprs.get(0, self.num_sgprs * self.ctx.id + idx)
    }
    fn read_sgpr_pair(&self, idx: usize) -> u64 {
        u64_from_u32_u32(self.read_sgpr(idx), self.read_sgpr(idx + 1))
    }
    fn write_sgpr(&mut self, idx: usize, value: u32) {
        if idx >= self.num_sgprs {
            panic!();
        }
        self.sgprs.set(0, self.num_sgprs * self.ctx.id + idx, value);
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
            126 => self.ctx.exec_lo,
            127 => self.ctx.exec_hi,
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
            126 => u64_from_u32_u32(self.ctx.exec_lo, self.ctx.exec_hi),
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
            124 => self.ctx.m0 = value,
            126 => self.ctx.exec_lo = value,
            127 => self.ctx.exec_hi = value,
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
            256..=511 => self.read_vgpr(elem, addr - 256),
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
            ((self.ctx.exec_hi >> (elem - 32)) & 1) != 0
        } else {
            ((self.ctx.exec_lo >> elem) & 1) != 0
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
    fn write_lds_u8(&mut self, addr: u32, data: u8) {
        let ptr = self.lds.as_mut_ptr().wrapping_add(addr as usize);
        unsafe {
            *ptr = data;
        }
    }
    fn read_lds_u8(&mut self, addr: u32) -> u8 {
        let ptr = self.lds.as_ptr().wrapping_add(addr as usize);
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
    fn execute_sopp(&mut self, inst: SOPP) -> Signals {
        let simm16 = inst.simm16 as i16;
        match inst.op {
            I::S_NOP => {}
            I::S_ENDPGM => return Signals::EndOfProgram,
            I::S_WAITCNT => {}
            I::S_BARRIER => return Signals::Switch,
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
            I::S_CBRANCH_EXECNZ => {
                if self.is_execnz() {
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
                if !self.ctx.scc {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            I::S_CBRANCH_SCC1 => {
                if self.ctx.scc {
                    self.next_pc = ((self.get_pc() as i64) + ((simm16 as i64) * 4) + 4) as usize;
                }
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_sopk(&mut self, inst: SOPK) -> Signals {
        let simm16 = inst.simm16 as i16;
        let d = inst.sdst as usize;
        match inst.op {
            I::S_MOVK_I32 => {
                s_movk_i32(self, d, simm16);
            }
            I::S_SETREG_IMM32_B32 => {
                let imm32 = self.fetch_literal_constant();
                s_setreg_imm32_b32(self, simm16, imm32);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_sop1(&mut self, inst: SOP1) -> Signals {
        let d = inst.sdst as usize;
        let s0 = inst.ssrc0 as usize;

        match inst.op {
            I::S_MOV_B32 => {
                s_mov_b32(self, d, s0);
            }
            I::S_MOV_B64 => {
                s_mov_b64(self, d, s0);
            }
            I::S_BREV_B32 => {
                s_brev_b32(self, d, s0);
            }
            I::S_FF1_I32_B32 => {
                s_ff1_i32_b32(self, d, s0);
            }
            I::S_AND_SAVEEXEC_B64 => {
                s_and_saveexec_b64(self, d, s0);
            }
            I::S_OR_SAVEEXEC_B64 => {
                s_or_saveexec_b64(self, d, s0);
            }
            I::S_ANDN2_SAVEEXEC_B64 => {
                s_andn2_saveexec_b64(self, d, s0);
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

        Signals::None
    }
    fn execute_sop2(&mut self, inst: SOP2) -> Signals {
        let d = inst.sdst as usize;
        let s0 = inst.ssrc0 as usize;
        let s1 = inst.ssrc1 as usize;

        match inst.op {
            I::S_ADD_U32 => {
                s_add_u32_e32(self, d, s0, s1);
            }
            I::S_SUB_U32 => {
                s_sub_u32_e32(self, d, s0, s1);
            }
            I::S_MAX_U32 => {
                s_max_u32_e32(self, d, s0, s1);
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
                s_lshl_b32(self, d, s0, s1);
            }
            I::S_LSHR_B32 => {
                s_lshr_b32(self, d, s0, s1);
            }
            I::S_BFM_B32 => {
                s_bfm_b32_e32(self, d, s0, s1);
            }
            I::S_CSELECT_B32 => {
                s_cselect_b32(self, d, s0, s1);
            }
            I::S_LSHL_B64 => {
                s_lshl_b64(self, d, s0, s1);
            }
            I::S_LSHR_B64 => {
                s_lshr_b64(self, d, s0, s1);
            }
            I::S_CSELECT_B64 => {
                s_cselect_b64(self, d, s0, s1);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_sopc(&mut self, inst: SOPC) -> Signals {
        let s0 = inst.ssrc0 as usize;
        let s1 = inst.ssrc1 as usize;

        match inst.op {
            I::S_CMP_EQ_U32 => {
                s_cmp_eq_u32_e32(self, s0, s1);
            }
            I::S_CMP_LT_U32 => {
                s_cmp_lt_u32_e32(self, s0, s1);
            }
            I::S_CMP_GE_U32 => {
                s_cmp_ge_u32_e32(self, s0, s1);
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
            I::S_CMP_GT_U32 => {
                s_cmp_gt_u32_e32(self, s0, s1);
            }
            I::S_CMP_NE_U64 => {
                s_cmp_ne_u64(self, s0, s1);
            }
            I::S_CMP_EQ_U64 => {
                s_cmp_eq_u64(self, s0, s1);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_smem(&mut self, inst: SMEM) -> Signals {
        let sdata = inst.sdata as usize;
        let soffset = inst.offset as u64;
        let sbase = (inst.sbase * 2) as usize;
        match inst.op {
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
        Signals::None
    }
    fn execute_vop1(&mut self, inst: VOP1) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0 as usize;
        match inst.op {
            I::V_MOV_B32 => {
                v_mov_b32_e32(self, d, s0);
            }
            I::V_NOT_B32 => {
                v_not_b32_e32(self, d, s0);
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
            I::V_CVT_I32_F64 => {
                v_cvt_i32_f64_e32(self, d, s0);
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
            I::V_RSQ_F64 => {
                v_rsq_f64_e32(self, d, s0);
            }
            I::V_RNDNE_F64 => {
                v_rndne_f64_e32(self, d, s0);
            }
            I::V_READFIRSTLANE_B32 => {
                v_readfirstlane_b32_e32(self, d, s0);
            }
            I::V_CVT_U32_F32 => {
                v_cvt_u32_f32_e32(self, d, s0);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_vop2(&mut self, inst: VOP2) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0 as usize;
        let s1 = inst.vsrc1 as usize;

        match inst.op {
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
        Signals::None
    }

    fn execute_vop3a(&mut self, inst: VOP3A) -> Signals {
        let d = inst.vdst as usize;
        let s0 = inst.src0 as usize;
        let s1 = inst.src1 as usize;
        let s2 = inst.src2 as usize;
        let abs = inst.abs;
        let neg = inst.neg;
        let clamp = inst.clamp != 0;
        let omod = inst.omod;
        match inst.op {
            I::V_ADD_U16 => {
                v_add_u16_e64(self, d, s0, s1);
            }
            I::V_CMP_CLASS_F64 => {
                v_cmp_class_f64_e64(self, d, s0, s1, abs, neg);
            }
            I::V_CMP_F64(op16) => {
                v_cmp_op_f64_e64(self, op16, d, s0, s1, abs, neg);
            }
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
            I::V_CMP_U64(op8) => {
                v_cmp_op_u64_e64(self, op8, d, s0, s1);
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
            I::V_AND_B32 => {
                v_and_b32_e64(self, d, s0, s1);
            }
            I::V_OR_B32 => {
                v_or_b32_e64(self, d, s0, s1);
            }
            I::V_XOR_B32 => {
                v_xor_b32_e64(self, d, s0, s1);
            }
            I::V_MAX_F32 => {
                v_max_f32_e64(self, d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_TRUNC_F32 => {
                v_trunc_f32_e64(self, d, s0, abs, neg, clamp, omod);
            }
            I::V_MAD_F32 => {
                v_mad_f32_e64(self, d, s0, s1, s2, abs, neg, clamp, omod);
            }
            I::V_ADD_F64 => {
                v_add_f64_e64(self, d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_MUL_F64 => {
                v_mul_f64_e64(self, d, s0, s1, abs, neg, clamp, omod);
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
            I::V_ASHRREV_I32 => {
                v_ashrrev_i32(self, d, s0, s1);
            }
            I::V_BFI_B32 => {
                v_bfi_b32(self, d, s0, s1, s2);
            }
            I::V_MUL_HI_U32 => {
                v_mul_hi_u32(self, d, s0, s1);
            }
            I::V_MAX_U32 => {
                v_max_u32(self, d, s0, s1);
            }
            I::V_MIN_U32 => {
                v_min_u32(self, d, s0, s1);
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
                v_mad_i32_i24(self, d, s0, s1, s2);
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
            I::V_MAD_U64_U32 => {
                v_mad_u64_u32(self, d, s0, s1, s2);
            }
            I::V_LDEXP_F64 => {
                v_ldexp_f64(self, d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_MAX_F64 => {
                v_max_f64(self, d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_MIN_F64 => {
                v_min_f64(self, d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_RSQ_F64 => {
                v_rsq_f64_e64(self, d, s0);
            }
            I::V_RCP_F64 => {
                v_rcp_f64_e64(self, d, s0);
            }
            I::V_CVT_F64_U32 => {
                v_cvt_f64_u32_e64(self, d, s0);
            }
            I::V_RNDNE_F64 => {
                v_rndne_f64_e64(self, d, s0);
            }
            I::V_CVT_I32_F64 => {
                v_cvt_i32_f64_e64(self, d, s0);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_vop3b(&mut self, inst: VOP3B) -> Signals {
        let d = inst.vdst as usize;
        let sd = inst.sdst as usize;
        let s0 = inst.src0 as usize;
        let s1 = inst.src1 as usize;
        let s2 = inst.src2 as usize;
        match inst.op {
            I::V_ADD_U32 => {
                v_add_u32_e64(self, d, sd, s0, s1);
            }
            I::V_SUB_U32 => {
                v_sub_u32_e64(self, d, sd, s0, s1);
            }
            I::V_ADDC_U32 => {
                v_addc_u32_e64(self, d, sd, s0, s1, s2);
            }
            I::V_DIV_SCALE_F32 => {
                v_div_scale_f32(self, d, sd, s0, s1, s2);
            }
            I::V_DIV_SCALE_F64 => {
                v_div_scale_f64(self, d, sd, s0, s1, s2);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_vopc(&mut self, inst: VOPC) -> Signals {
        let s0 = inst.src0 as usize;
        let s1 = inst.vsrc1 as usize;

        match inst.op {
            I::V_CMP_F64(op16) => {
                v_cmp_f64(self, op16, s0, s1);
            }
            I::V_CMP_F32(op16) => {
                v_cmp_f32(self, op16, s0, s1);
            }
            I::V_CMP_I32(op8) => {
                v_cmp_i32(self, op8, s0, s1);
            }
            I::V_CMP_U32(op8) => {
                v_cmp_u32(self, op8, s0, s1);
            }
            I::V_CMP_CLASS_F64 => {
                v_cmp_class_f64(self, s0, s1);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_flat(&mut self, inst: FLAT) -> Signals {
        let s = inst.data as usize;
        let d = inst.vdst as usize;
        let addr = inst.addr as usize;

        // Flat scratch memory is not supported yet.
        match inst.op {
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
            I::FLAT_STORE_DWORDX2 => {
                flat_store_dwordx2(self, s, addr);
            }
            I::FLAT_STORE_DWORDX4 => {
                flat_store_dwordx4(self, s, addr);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_mubuf(&mut self, inst: MUBUF) -> Signals {
        let d = inst.vdata as usize;
        let s = inst.vdata as usize;
        let vaddr = inst.vaddr as usize;
        let srsrc = inst.srsrc as usize;
        let soffset = inst.soffset as usize;
        let offset = inst.offset;
        let offen = inst.offen != 0;
        match inst.op {
            I::BUFFER_LOAD_DWORD => {
                buffer_load_dword(self, d, vaddr, srsrc, soffset, offset, offen);
            }
            I::BUFFER_STORE_DWORD => {
                buffer_store_dword(self, s, vaddr, srsrc, soffset, offset, offen);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn execute_ds(&mut self, inst: DS) -> Signals {
        let addr = inst.addr as usize;
        let d0 = inst.data0 as usize;
        let r = inst.vdst as usize;
        let offset = inst.offset0 as u8;
        match inst.op {
            I::DS_WRITE_B8 => {
                ds_write_b8(self, d0, addr, offset);
            }
            I::DS_READ_U8 => {
                ds_read_u8(self, r, addr, offset);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }
    fn get_pc(&self) -> u64 {
        (&self.insts[self.ctx.pc] as *const u8) as u64
    }
    fn set_pc(&mut self, value: u64) {
        let base_ptr = (&self.insts[0] as *const u8) as u64;
        self.ctx.pc = (value - base_ptr) as usize;
    }
    fn execute_inst(&mut self, inst: InstFormat) -> Signals {
        // println!("{:012X}: {:?}", self.ctx.pc, inst);
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
            InstFormat::DS(fields) => self.execute_ds(fields),
            _ => unimplemented!(),
        }
    }

    fn fetch_inst(&mut self) -> u64 {
        get_u64(&self.insts, self.ctx.pc)
    }
    fn fetch_literal_constant(&self) -> u32 {
        get_u32(&self.insts, self.ctx.pc + 4)
    }
}

use std::sync::{Arc, Mutex};

pub struct GCNProcessor<'a> {
    cunits: Vec<Arc<Mutex<ComputeUnit>>>,
    entry_address: usize,
    kernel_desc: KernelDescriptor,
    aql_packet_address: u64,
    kernel_args_ptr: u64,
    aql: HsaKernelDispatchPacket<'a>,
    private_seg_buffer: Vec<u8>,
}

impl<'a> GCNProcessor<'a> {
    pub fn new(aql: &HsaKernelDispatchPacket<'a>, num_cunits: usize, mem: &Vec<u8>) -> Self {
        let workgroup_size_x = aql.workgroup_size_x as u32;
        let workgroup_size_y = aql.workgroup_size_y as u32;
        let workgroup_size_z = aql.workgroup_size_z as u32;

        let workgroup_size = (workgroup_size_x * workgroup_size_y * workgroup_size_z) as usize;

        let insts = aql.kernel_object.object.to_vec();
        let kd = aql.kernel_object.offset;
        let kernel_desc = decode_kernel_desc(&insts[kd..(kd + 64)]);
        let aql_packet_address = (aql as *const HsaKernelDispatchPacket) as u64;

        let mut cunits = vec![];
        for _ in 0..num_cunits {
            let cu = Arc::new(Mutex::new(ComputeUnit::new(
                kd + kernel_desc.kernel_code_entry_byte_offset,
                mem.clone(),
                kernel_desc.granulated_wavefront_sgpr_count,
                kernel_desc.granulated_workitem_vgpr_count,
            )));

            cunits.push(cu);
        }

        let kernel_args_ptr = aql.kernarg_address.address();
        let entry_address = kd + kernel_desc.kernel_code_entry_byte_offset;

        let private_segment_size = aql.private_segment_size as usize;
        let private_seg_buffer: Vec<u8> =
            vec![0u8; private_segment_size * num_cunits * workgroup_size];

        // create instance
        GCNProcessor {
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
    ) -> ([u32; 16], [[u32; 64]; 16]) {
        let private_seg_ptr = if self.private_seg_buffer.len() > 0 {
            (&self.private_seg_buffer[0] as *const u8) as u64
        } else {
            0
        };

        let kernel_args_ptr = self.kernel_args_ptr;
        let aql_packet_address = self.aql_packet_address;
        let kernel_desc = &self.kernel_desc;
        let private_seg_size = self.aql.private_segment_size as u64;

        // Initialize SGPRS
        let mut sgprs = [0u32; 16];
        let mut sgprs_pos = 0;
        if kernel_desc.enable_sgpr_private_segment_buffer {
            let mut desc_w0 = 0;
            desc_w0 |= (private_seg_ptr + (thread_id as u64) * private_seg_size) & ((1 << 48) - 1);
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
            sgprs[sgprs_pos] = thread_id * self.aql.private_segment_size;
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

        // Initialize VGPRs
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
        }

        (sgprs, vgprs)
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
                        (workitem_id + workgroup_size * cu_idx) as u32,
                        workgroup_id_x,
                        workgroup_id_y,
                        workgroup_id_z,
                        workitem_id,
                    ));
                }

                let entry_address = self.entry_address;

                let cu = Arc::clone(&self.cunits[cu_idx]);
                use std::thread;

                let handle = thread::spawn(move || {
                    if let Ok(mut v) = cu.lock() {
                        v.dispatch(entry_address, setup_data, workgroup_size / 64);
                    }
                });
                thread_handles.push(handle);
            }

            for t in thread_handles {
                t.join().unwrap();
                bar.inc(1);
            }
        }

        bar.finish();
    }
}
