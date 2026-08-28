use crate::buffer::*;
use crate::instructions::*;
use crate::processor::*;
use crate::rdna4_decoder::*;
use crate::rdna_instructions::*;
use crate::rdna_translator::*;

use std::cell::RefCell;
use std::collections::VecDeque;

static USE_ENTIRE_KERNEL_TRANSLATION: bool = true;

/// Which execution engine a processor runs its waves on. Selected per instance
/// so that one process can exercise both, which the conformance tests rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Engine {
    Interpreter,
    LlvmJit,
}

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
    engine: Engine,
}

#[inline(always)]
fn u64_from_u32_u32(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | (lo as u64)
}

#[inline(always)]
fn u32_from_u16_u16(lo: u16, hi: u16) -> u32 {
    ((hi as u32) << 16) | (lo as u32)
}

#[inline(always)]
fn add_u32(a: u32, b: u32, c: u32) -> (u32, bool) {
    let d = (a as u64) + (b as u64) + (c as u64);
    ((d & 0xFFFF_FFFF) as u32, d >= (0x1_0000_0000 as u64))
}

#[inline(always)]
fn sub_u32(a: u32, b: u32, c: u32) -> (u32, bool) {
    let d = (a as u64).wrapping_sub(b as u64).wrapping_sub(c as u64);
    (
        (d & 0xFFFF_FFFF) as u32,
        (b as u64) + (c as u64) > (a as u64),
    )
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
    // ISA §V_DIV_SCALE_F32, with the three places the part departs from the
    // pseudo code: the quotient-underflow case triggers on the exponent
    // difference, symmetric with the overflow one; the reciprocal test is a
    // test on the denominator's own exponent; and the NaN a zero operand
    // returns is the negative quiet one, with VCC set.
    let denominator_exponent = get_exp_f32(s1);
    let numerator_exponent = get_exp_f32(s2);
    let delta = numerator_exponent - denominator_exponent;

    let returns_nan = s1 == 0.0 || s2 == 0.0;
    let overflows = delta >= 96;
    let underflows = delta <= -96;
    let reciprocal_is_subnormal = denominator_exponent >= 253;
    let operands_are_tiny = denominator_exponent == 0 || numerator_exponent <= 23;

    let scaled_up = libm::ldexpf(s0, 64);
    let scaled_down = libm::ldexpf(s0, -64);
    // Only the operand this call was handed is scaled; the other one passes
    // through unchanged.
    let denominator_up = if s0 == s1 { scaled_up } else { s0 };
    let denominator_down = if s0 == s1 { scaled_down } else { s0 };
    let numerator_up = if s0 == s2 { scaled_up } else { s0 };

    // The cases are applied in reverse, so that an earlier one wins.
    let mut d = s0;
    if operands_are_tiny {
        d = scaled_up;
    }
    if reciprocal_is_subnormal {
        d = scaled_down;
    }
    if underflows {
        // Scale the numerator up, unless the reciprocal is already subnormal,
        // in which case the denominator is scaled down.
        d = if reciprocal_is_subnormal {
            denominator_down
        } else {
            numerator_up
        };
    }
    if overflows {
        d = denominator_up;
    }
    if returns_nan {
        d = u32_to_f32(0xFFC0_0000);
    }

    (d, overflows || underflows)
}

fn div_scale_f64(s0: f64, s1: f64, s2: f64) -> (f64, bool) {
    // ISA §V_DIV_SCALE_F64, with the three places the part departs from the
    // pseudo code: the quotient-underflow case triggers on the exponent
    // difference, symmetric with the overflow one; the reciprocal test is a
    // test on the denominator's own exponent; and the NaN a zero operand
    // returns is the negative quiet one, with VCC set.
    let denominator_exponent = get_exp_f64(s1);
    let numerator_exponent = get_exp_f64(s2);
    let delta = numerator_exponent - denominator_exponent;

    let returns_nan = s1 == 0.0 || s2 == 0.0;
    let overflows = delta >= 768;
    let underflows = delta <= -768;
    let reciprocal_is_subnormal = denominator_exponent >= 2045;
    let operands_are_tiny = denominator_exponent == 0 || numerator_exponent <= 53;

    let scaled_up = libm::ldexp(s0, 128);
    let scaled_down = libm::ldexp(s0, -128);
    // Only the operand this call was handed is scaled; the other one passes
    // through unchanged.
    let denominator_up = if s0 == s1 { scaled_up } else { s0 };
    let denominator_down = if s0 == s1 { scaled_down } else { s0 };
    let numerator_up = if s0 == s2 { scaled_up } else { s0 };

    // The cases are applied in reverse, so that an earlier one wins.
    let mut d = s0;
    if operands_are_tiny {
        d = scaled_up;
    }
    if reciprocal_is_subnormal {
        d = scaled_down;
    }
    if underflows {
        // Scale the numerator up, unless the reciprocal is already subnormal,
        // in which case the denominator is scaled down.
        d = if reciprocal_is_subnormal {
            denominator_down
        } else {
            numerator_up
        };
    }
    if overflows {
        d = denominator_up;
    }
    if returns_nan {
        d = u64_to_f64(0xFFF8_0000_0000_0000);
    }

    (d, overflows || underflows)
}

fn div_fixup_f32(s0: f32, s1: f32, s2: f32) -> f32 {
    let sign_out = s1.is_sign_negative() != s2.is_sign_negative();
    if s2.is_nan() {
        quiet_nan_f32(s2)
    } else if s1.is_nan() {
        quiet_nan_f32(s1)
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
    } else if s0.is_nan() {
        // A NaN quotient means the division overflowed, which the fixup turns
        // into an infinity of the quotient's sign rather than passing it on.
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
        quiet_nan_f64(s2)
    } else if s1.is_nan() {
        quiet_nan_f64(s1)
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
    } else if s0.is_nan() {
        // A NaN quotient means the division overflowed, which the fixup turns
        // into an infinity of the quotient's sign rather than passing it on.
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

/// ABS and NEG act on the sign bit of the operand, whatever the compare reads it
/// as, so an integer compare sees them too.
fn abs_neg_bits(value: u64, abs: u8, neg: u8, idx: usize, width: u32) -> u64 {
    let sign = 1u64 << (width - 1);
    let mut result = value;
    if ((abs >> idx) & 1) != 0 {
        result &= !sign;
    }
    if ((neg >> idx) & 1) != 0 {
        result ^= sign;
    }
    result
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

/// A signalling NaN is quieted on the way out, keeping its payload and sign.
/// The hardware subtracts by negating and adding: a NaN operand comes through
/// quieted, and the negated one comes through with its sign flipped.
fn sub_f32(a: f32, b: f32) -> f32 {
    let negated = -b;
    if a.is_nan() {
        quiet_nan_f32(a)
    } else if negated.is_nan() {
        quiet_nan_f32(negated)
    } else {
        a + negated
    }
}

fn quiet_nan_f32(value: f32) -> f32 {
    if value.is_nan() {
        f32::from_bits(value.to_bits() | 0x0040_0000)
    } else {
        value
    }
}

fn quiet_nan_f64(value: f64) -> f64 {
    if value.is_nan() {
        f64::from_bits(value.to_bits() | 0x0008_0000_0000_0000)
    } else {
        value
    }
}

/// The transcendental unit flushes denormals on input and on output, whatever
/// the mode register says (ISA §V_RCP_F32, §V_SQRT_F32, §V_RSQ_F32).
fn ftz_f32(value: f32) -> f32 {
    if value != 0.0 && value.abs() < f32::MIN_POSITIVE {
        f32::copysign(0.0, value)
    } else {
        value
    }
}

/// The half of a packed source that a selector names: OPSEL names the one that
/// feeds the low result and OPSEL_HI the one that feeds the high result.
fn packed_half(raw: u32, select: u8, idx: usize) -> u16 {
    if (select >> idx) & 1 != 0 {
        (raw >> 16) as u16
    } else {
        raw as u16
    }
}

/// The two halves of a packed source as the instruction sees them, each with
/// NEG or NEG_HI applied to its sign bit.
fn packed_halves(raw: u32, opsel: u8, opsel_hi: u8, neg: u8, neg_hi: u8, idx: usize) -> (u16, u16) {
    let mut low = packed_half(raw, opsel, idx);
    let mut high = packed_half(raw, opsel_hi, idx);
    if (neg >> idx) & 1 != 0 {
        low ^= 0x8000;
    }
    if (neg_hi >> idx) & 1 != 0 {
        high ^= 0x8000;
    }
    (low, high)
}

/// The dword a dot product reads its terms out of, assembled from the halves
/// OPSEL names. NEG has no say here: it tells a dot product whether a source is
/// signed rather than negating it.
fn packed_dword(raw: u32, opsel: u8, opsel_hi: u8, idx: usize) -> u32 {
    let low = packed_half(raw, opsel, idx);
    let high = packed_half(raw, opsel_hi, idx);
    (low as u32) | ((high as u32) << 16)
}

/// CLAMP saturates a packed unsigned result rather than letting it wrap, so the
/// operation is done wide enough to see that.
fn clamp_u16(value: i64, clamp: bool) -> u16 {
    if clamp {
        value.clamp(0, u16::MAX as i64) as u16
    } else {
        value as u16
    }
}

/// CLAMP saturates a packed signed result the same way.
fn clamp_i16(value: i64, clamp: bool) -> u16 {
    if clamp {
        value.clamp(i16::MIN as i64, i16::MAX as i64) as i16 as u16
    } else {
        value as i16 as u16
    }
}

fn clamp_f16(value: f16, clamp: bool) -> f16 {
    if clamp {
        // CLAMP turns a NaN into zero rather than passing it through.
        if value.is_nan() {
            f16::from_f32(0.0)
        } else {
            f16::from_f32(value.to_f32().clamp(0.0, 1.0))
        }
    } else {
        value
    }
}

/// IEEE minNum and maxNum: a NaN gives way to the other operand.
fn min_num_f16(a: f16, b: f16) -> f16 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a.to_f32() < b.to_f32() {
        a
    } else {
        b
    }
}

fn max_num_f16(a: f16, b: f16) -> f16 {
    if a.is_nan() {
        b
    } else if b.is_nan() {
        a
    } else if a.to_f32() > b.to_f32() {
        a
    } else {
        b
    }
}

/// IEEE 754-2019 minimum and maximum: a NaN propagates, and the sign of a zero
/// decides between the two zeroes.
fn minimum_f16(a: f16, b: f16) -> f16 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if a.to_f32() == 0.0 && b.to_f32() == 0.0 {
        if a.to_bits() & 0x8000 != 0 {
            a
        } else {
            b
        }
    } else if a.to_f32() < b.to_f32() {
        a
    } else {
        b
    }
}

fn maximum_f16(a: f16, b: f16) -> f16 {
    if a.is_nan() {
        a
    } else if b.is_nan() {
        b
    } else if a.to_f32() == 0.0 && b.to_f32() == 0.0 {
        if a.to_bits() & 0x8000 == 0 {
            a
        } else {
            b
        }
    } else if a.to_f32() > b.to_f32() {
        a
    } else {
        b
    }
}

/// A bfloat16 is the top half of an f32.
fn bf16_to_f32(value: u16) -> f32 {
    f32::from_bits((value as u32) << 16)
}

fn clamp_f32(value: f32, clamp: bool) -> f32 {
    if clamp {
        // CLAMP turns a NaN into zero rather than passing it through.
        if value.is_nan() {
            0.0
        } else {
            value.clamp(0.0, 1.0)
        }
    } else {
        value
    }
}

fn clamp_f64(value: f64, clamp: bool) -> f64 {
    if clamp {
        // CLAMP turns a NaN into zero rather than passing it through.
        if value.is_nan() {
            0.0
        } else {
            value.clamp(0.0, 1.0)
        }
    } else {
        value
    }
}

/// OMOD and CLAMP act on the 16-bit result, which is the width the instruction
/// wrote.
fn f16_to_u32_omod_clamp(value: f16, omod: u8, clamp: bool) -> u32 {
    let scaled = match omod {
        1 => value.to_f32() * 2.0,
        2 => value.to_f32() * 4.0,
        3 => value.to_f32() * 0.5,
        _ => value.to_f32(),
    };
    let clamped = if clamp {
        // CLAMP turns a NaN into zero rather than passing it through.
        if scaled.is_nan() {
            0.0
        } else {
            scaled.clamp(0.0, 1.0)
        }
    } else {
        scaled
    };
    f16::from_f32(clamped).to_bits() as u32
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

        if self.engine == Engine::Interpreter {
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

    /// A 32-bit literal reaching a signed 64-bit operand is sign-extended, where
    /// an unsigned or bitwise one is zero-extended.
    fn read_scalar_source_operand_i64(&self, addr: SourceOperand) -> i64 {
        match addr {
            SourceOperand::LiteralConstant(value) => value as i32 as i64,
            _ => self.read_scalar_source_operand_u64(addr) as i64,
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

    /// The source a mixed-precision instruction reads: {OPSEL_HI[i], OPSEL[i]}
    /// selects it, OPSEL_HI=0 meaning a whole f32 and OPSEL otherwise picking
    /// which half of the dword to widen.
    fn read_vector_source_operand_mix_f32(
        &self,
        elem: usize,
        addr: SourceOperand,
        idx: usize,
        opsel: u8,
        opsel_hi: u8,
    ) -> f32 {
        if (opsel_hi >> idx) & 1 == 0 {
            self.read_vector_source_operand_f32(elem, addr)
        } else if (opsel >> idx) & 1 != 0 {
            self.read_vector_source_operand_f16_hi(elem, addr).to_f32()
        } else {
            self.read_vector_source_operand_f16(elem, addr).to_f32()
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
            InstFormat::VSAMPLE(fields) => self.execute_vsample(fields),
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
            I::S_CVT_F32_U32 => {
                self.s_cvt_f32_u32(d, s0);
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
            I::S_CVT_I32_F32 => {
                self.s_cvt_i32_f32(d, s0);
            }
            I::S_CVT_U32_F32 => {
                self.s_cvt_u32_f32(d, s0);
            }
            I::S_XOR_SAVEEXEC_B32 => {
                self.s_xor_saveexec_b32(d, s0);
            }
            op => unimplemented!("{:?}", op),
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

    fn s_cvt_f32_u32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let d_value = s0_value as f32;
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
            I::S_BFE_U32 => {
                self.s_bfe_u32(d, s0, s1);
            }
            I::S_AND_B64 => {
                self.s_and_b64(d, s0, s1);
            }
            I::S_OR_B64 => {
                self.s_or_b64(d, s0, s1);
            }
            I::S_XOR_B64 => {
                self.s_xor_b64(d, s0, s1);
            }
            I::S_ASHR_I64 => {
                self.s_ashr_i64(d, s0, s1);
            }
            I::S_LSHR_B64 => {
                self.s_lshr_b64(d, s0, s1);
            }
            I::S_CSELECT_B64 => {
                self.s_cselect_b64(d, s0, s1);
            }
            op => unimplemented!("{:?}", op),
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

    fn s_cvt_i32_f32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = u32_to_f32(self.read_scalar_source_operand_u32(s0));
        self.write_sop_dst(d, s0_value as i32 as u32);
    }

    fn s_cvt_u32_f32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = u32_to_f32(self.read_scalar_source_operand_u32(s0));
        self.write_sop_dst(d, s0_value as u32);
    }

    fn s_xor_saveexec_b32(&mut self, d: usize, s0: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let exec_value = self.get_exec();

        self.write_sop_dst(d, exec_value);

        let exec_value = s0_value ^ exec_value;

        self.set_exec(exec_value);
        self.ctx.scc = exec_value != 0;
    }

    fn s_and_b64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        let d_value = s0_value & s1_value;
        self.write_sop_dst_pair(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_or_b64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        let d_value = s0_value | s1_value;
        self.write_sop_dst_pair(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_xor_b64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        let d_value = s0_value ^ s1_value;
        self.write_sop_dst_pair(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_ashr_i64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_i64(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = (s0_value >> (s1_value & 0x3F)) as u64;
        self.write_sop_dst_pair(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_lshr_b64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let d_value = s0_value >> (s1_value & 0x3F);
        self.write_sop_dst_pair(d, d_value);
        self.ctx.scc = d_value != 0;
    }

    fn s_cselect_b64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u64(s0);
        let s1_value = self.read_scalar_source_operand_u64(s1);
        let d_value = if self.ctx.scc { s0_value } else { s1_value };
        self.write_sop_dst_pair(d, d_value);
    }

    fn s_cmp_eq_i32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        self.ctx.scc = s0_value == s1_value;
    }

    fn s_cmp_ge_i32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        self.ctx.scc = s0_value >= s1_value;
    }

    fn s_cmp_gt_i32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        self.ctx.scc = s0_value > s1_value;
    }

    fn s_cmp_le_i32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        self.ctx.scc = s0_value <= s1_value;
    }

    fn s_cmp_lg_i32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0) as i32;
        let s1_value = self.read_scalar_source_operand_u32(s1) as i32;
        self.ctx.scc = s0_value != s1_value;
    }

    fn s_cmp_le_u32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        self.ctx.scc = s0_value <= s1_value;
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
        let d_value = s0_value.wrapping_mul(s1_value);
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

    fn s_bfe_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        let offset = s1_value & 0x1F;
        let width = ((s1_value >> 16) & 0x7F).min(32);
        let mask = ((1u64 << width) - 1) as u32;
        let d_value = (s0_value >> offset) & mask;
        self.write_sop_dst(d, d_value as u32);
        self.ctx.scc = d_value != 0;
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
            I::S_CMP_GT_U32 => {
                self.s_cmp_gt_u32(s0, s1);
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
            I::S_CMP_EQ_I32 => {
                self.s_cmp_eq_i32(s0, s1);
            }
            I::S_CMP_GE_I32 => {
                self.s_cmp_ge_i32(s0, s1);
            }
            I::S_CMP_GT_I32 => {
                self.s_cmp_gt_i32(s0, s1);
            }
            I::S_CMP_LE_I32 => {
                self.s_cmp_le_i32(s0, s1);
            }
            I::S_CMP_LG_I32 => {
                self.s_cmp_lg_i32(s0, s1);
            }
            I::S_CMP_LE_U32 => {
                self.s_cmp_le_u32(s0, s1);
            }
            op => unimplemented!("{:?}", op),
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

    fn s_cmp_gt_u32(&mut self, s0: SourceOperand, s1: SourceOperand) {
        let s0_value = self.read_scalar_source_operand_u32(s0);
        let s1_value = self.read_scalar_source_operand_u32(s1);
        self.ctx.scc = s0_value > s1_value;
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
            I::S_CMOVK_I32 => {
                if self.ctx.scc {
                    self.write_sop_dst(d, simm16 as i32 as u32);
                }
            }
            I::S_MULK_I32 => {
                let d_value = (self.read_sgpr(d) as i32).wrapping_mul(simm16 as i32);
                self.write_sop_dst(d, d_value as u32);
            }
            op => unimplemented!("{:?}", op),
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
            I::V_CVT_F32_F16 => {
                self.v_cvt_f32_f16_e32(d, s0);
            }
            I::V_FLOOR_F32 => {
                self.v_floor_f32_e32(d, s0);
            }
            I::V_BFREV_B32 => {
                self.v_bfrev_b32_e32(d, s0);
            }
            I::V_CEIL_F32 => {
                self.v_ceil_f32_e32(d, s0);
            }
            I::V_CLZ_I32_U32 => {
                self.v_clz_i32_u32_e32(d, s0);
            }
            I::V_COS_F32 => {
                self.v_cos_f32_e32(d, s0);
            }
            I::V_SIN_F32 => {
                self.v_sin_f32_e32(d, s0);
            }
            I::V_EXP_F32 => {
                self.v_exp_f32_e32(d, s0);
            }
            I::V_LOG_F32 => {
                self.v_log_f32_e32(d, s0);
            }
            I::V_CVT_F16_F32 => {
                self.v_cvt_f16_f32_e32(d, s0);
            }
            I::V_CVT_F32_F64 => {
                self.v_cvt_f32_f64_e32(d, s0);
            }
            I::V_CVT_F32_I32 => {
                self.v_cvt_f32_i32_e32(d, s0);
            }
            I::V_CVT_F64_F32 => {
                self.v_cvt_f64_f32_e32(d, s0);
            }
            I::V_CVT_U32_F64 => {
                self.v_cvt_u32_f64_e32(d, s0);
            }
            I::V_FLOOR_F64 => {
                self.v_floor_f64_e32(d, s0);
            }
            I::V_TRUNC_F32 => {
                self.v_trunc_f32_e32(d, s0);
            }
            I::V_TRUNC_F64 => {
                self.v_trunc_f64_e32(d, s0);
            }
            I::V_FREXP_EXP_I32_F64 => {
                self.v_frexp_exp_i32_f64_e32(d, s0);
            }
            I::V_FREXP_MANT_F64 => {
                self.v_frexp_mant_f64_e32(d, s0);
            }
            I::V_NOT_B32 => {
                self.v_not_b32_e32(d, s0);
            }
            I::V_RSQ_F32 => {
                self.v_rsq_f32_e32(d, s0);
            }
            I::V_SQRT_F64 => {
                self.v_sqrt_f64_e32(d, s0);
            }
            op => unimplemented!("{:?}", op),
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
            let d_value = ftz_f32(1.0 / ftz_f32(s0_value));

            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_rcp_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = ftz_f32(1.0 / ftz_f32(s0_value));

            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_sqrt_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = ftz_f32(ftz_f32(s0_value).sqrt());

            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_rndne_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            // roundToIntegralTiesToEven. Rounding by hand as floor(x + 0.5)
            // loses the sign of a zero result, which the hardware keeps.
            let d_value = quiet_nan_f32(s0_value.round_ties_even());

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
            // FLOOR-based fractional part (ISA §V_FRACT): fract(-1.2) = 0.8.
            // Rust f64::fract() truncates, giving the wrong sign for negatives.
            // The result is in [0,1), so clamp: the subtraction rounds up to
            // exactly 1.0 for a tiny negative input.
            let frac = s0_value - s0_value.floor();
            let d_value = if frac >= 1.0 {
                f64::from_bits(0x3FEF_FFFF_FFFF_FFFF)
            } else {
                frac
            };

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
            // roundToIntegralTiesToEven. Rounding by hand as floor(x + 0.5)
            // loses the sign of a zero result, which the hardware keeps.
            let d_value = quiet_nan_f64(s0_value.round_ties_even());

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

            self.write_vgpr(elem, d, f32_to_u32(quiet_nan_f32(d_value)));
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
            I::V_ASHRREV_I32 => {
                self.v_ashrrev_i32_e32(d, s0, s1);
            }
            I::V_MAX_U32 => {
                self.v_max_u32_e32(d, s0, s1);
            }
            I::V_MIN_NUM_F64 => {
                self.v_min_num_f64_e32(d, s0, s1);
            }
            I::V_MIN_U32 => {
                self.v_min_u32_e32(d, s0, s1);
            }
            I::V_MAX_I32 => {
                self.v_max_i32_e32(d, s0, s1);
            }
            I::V_MIN_I32 => {
                self.v_min_i32_e32(d, s0, s1);
            }
            I::V_MAX_NUM_F32 => {
                self.v_max_num_f32_e32(d, s0, s1);
            }
            I::V_MIN_NUM_F32 => {
                self.v_min_num_f32_e32(d, s0, s1);
            }
            I::V_MUL_I32_I24 => {
                self.v_mul_i32_i24_e32(d, s0, s1);
            }
            I::V_SUBREV_F32 => {
                self.v_subrev_f32_e32(d, s0, s1);
            }
            I::V_SUB_CO_CI_U32 => {
                self.v_sub_co_ci_u32_e32(d, s0, s1);
            }
            I::V_SUBREV_CO_CI_U32 => {
                self.v_subrev_co_ci_u32_e32(d, s0, s1);
            }
            op => unimplemented!("{:?}", op),
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

    fn v_subrev_nc_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            // CLAMP saturates the unsigned result instead of letting it wrap.
            let d_value = if clamp {
                s1_value.saturating_sub(s0_value)
            } else {
                s1_value.wrapping_sub(s0_value)
            };
            self.write_vgpr(elem, d, d_value as u32);
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

    fn v_sub_co_ci_u32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let carry = self.get_vcc_bit(elem);
            let (d_value, carry) = sub_u32(s0_value, s1_value, carry as u32);
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

    fn v_subrev_co_ci_u32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let carry = self.get_vcc_bit(elem);
            let (d_value, carry) = sub_u32(s1_value, s0_value, carry as u32);
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

    fn v_mul_u32_u24_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = (s0_value & 0x00FF_FFFF).wrapping_mul(s1_value & 0x00FF_FFFF);
            self.write_vgpr(elem, d, d_value as u32);
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
            // The hardware subtracts by negating the operand and adding, so a NaN
            // operand reaches the result with its sign flipped.
            let d_value = sub_f32(s0_value, s1_value);
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_sub_f32_e64(
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
            // The hardware subtracts by negating the operand and adding, so a NaN
            // operand reaches the result with its sign flipped.
            let d_value = sub_f32(s0_value, s1_value);
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
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
                self.v_add_nc_u16(d, s0, s1, abs, neg, clamp);
            }
            I::V_LSHLREV_B16 => {
                self.v_lshlrev_b16_e64(d, s0, s1, abs, neg);
            }
            I::V_READLANE_B32 => {
                self.v_readlane_b32(d, s0, s1);
            }
            I::V_WRITELANE_B32 => {
                self.v_writelane_b32(d, s0, s1);
            }
            I::V_AND_B32 => {
                self.v_and_b32_e64(d, s0, s1, abs, neg);
            }
            I::V_LSHL_OR_B32 => {
                self.v_lshl_or_b32(d, s0, s1, s2, abs, neg);
            }
            I::V_AND_OR_B32 => {
                self.v_and_or_b32(d, s0, s1, s2, abs, neg);
            }
            I::V_BFE_U32 => {
                self.v_bfe_u32(d, s0, s1, s2, abs, neg);
            }
            I::V_MAX_U32 => {
                self.v_max_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_MIN_U32 => {
                self.v_min_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_ASHRREV_I32 => {
                self.v_ashrrev_i32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_EQ_U16 => {
                self.v_cmp_eq_u16_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_GT_U16 => {
                self.v_cmp_gt_u16_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_LT_U32 => {
                self.v_cmp_lt_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_MUL_LO_U32 => {
                self.v_mul_lo_u32(d, s0, s1, abs, neg);
            }
            I::V_MUL_HI_U32 => {
                self.v_mul_hi_u32(d, s0, s1, abs, neg);
            }
            I::V_XOR_B32 => {
                self.v_xor_b32_e64(d, s0, s1, abs, neg);
            }
            I::V_OR3_B32 => {
                self.v_or3_b32(d, s0, s1, s2, abs, neg);
            }
            I::V_XOR3_B32 => {
                self.v_xor3_b32(d, s0, s1, s2, abs, neg);
            }
            I::V_ADD_NC_U32 => {
                self.v_add_nc_u32_e64(d, s0, s1, abs, neg, clamp);
            }
            I::V_SUB_NC_U32 => {
                self.v_sub_nc_u32_e64(d, s0, s1, abs, neg, clamp);
            }
            I::V_ADD3_U32 => {
                self.v_add3_u32(d, s0, s1, s2, abs, neg);
            }
            I::V_ALIGNBIT_B32 => {
                self.v_alignbit_b32(d, s0, s1, s2, abs, neg);
            }
            I::V_BFI_B32 => {
                self.v_bfi_b32(d, s0, s1, s2, abs, neg);
            }
            I::V_BCNT_U32_B32 => {
                self.v_bcnt_u32_b32(d, s0, s1, abs, neg);
            }
            I::V_MAD_U32_U24 => {
                self.v_mad_u32_u24(d, s0, s1, s2, abs, neg);
            }
            I::V_ADD_F32 => {
                self.v_add_f32_e64(d, s0, s1, abs, neg, clamp, omod);
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
            I::V_CVT_F32_U32 => {
                self.v_cvt_f32_u32_e64(d, s0, abs, neg, clamp, omod);
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
            I::V_RCP_F32 => {
                self.v_rcp_f32_e64(d, s0, abs, neg, clamp, omod);
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
                self.v_lshlrev_b32_e64(d, s0, s1, abs, neg);
            }
            I::V_LSHRREV_B32 => {
                self.v_lshrrev_b32_e64(d, s0, s1, abs, neg);
            }
            I::V_LSHLREV_B64 => {
                self.v_lshlrev_b64_e64(d, s0, s1, abs, neg);
            }
            I::V_LSHRREV_B64 => {
                self.v_lshrrev_b64(d, s0, s1, abs, neg);
            }
            I::V_OR_B32 => {
                self.v_or_b32_e64(d, s0, s1, abs, neg);
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
                self.v_xad_u32(d, s0, s1, s2, abs, neg);
            }
            I::V_LSHL_ADD_U32 => {
                self.v_lshl_add_u32(d, s0, s1, s2, abs, neg);
            }
            I::V_ADD_LSHL_U32 => {
                self.v_add_lshl_u32(d, s0, s1, s2, abs, neg);
            }
            I::V_CMP_NE_U32 => {
                self.v_cmp_ne_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_EQ_U32 => {
                self.v_cmp_eq_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_GT_U32 => {
                self.v_cmp_gt_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_GE_U32 => {
                self.v_cmp_ge_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_NE_U32 => {
                self.v_cmpx_ne_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_GT_I32 => {
                self.v_cmp_gt_i32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_LT_U64 => {
                self.v_cmp_lt_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_EQ_U64 => {
                self.v_cmp_eq_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_TRIG_PREOP_F64 => {
                self.v_trig_preop_f64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CVT_F32_F16 => {
                self.v_cvt_f32_f16_e64(d, s0, abs, neg, clamp, omod, inst.opsel);
            }
            I::V_LDEXP_F32 => {
                self.v_ldexp_f32(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_FMAC_F32 => {
                self.v_fmac_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_FLOOR_F32 => {
                self.v_floor_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_S_RCP_F32 => {
                self.v_s_rcp_f32(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_F64_I32 => {
                self.v_cvt_f64_i32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_I32_F32 => {
                self.v_cvt_i32_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_U32_F32 => {
                self.v_cvt_u32_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_FRACT_F64 => {
                self.v_fract_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_FREXP_EXP_I32_F32 => {
                self.v_frexp_exp_i32_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_FREXP_MANT_F32 => {
                self.v_frexp_mant_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_MOV_B32 => {
                self.v_mov_b32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_RCP_IFLAG_F32 => {
                self.v_rcp_iflag_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_RNDNE_F32 => {
                self.v_rndne_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_SQRT_F32 => {
                self.v_sqrt_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CMPX_EQ_U32 => {
                self.v_cmpx_eq_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_GT_U32 => {
                self.v_cmpx_gt_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_LT_I32 => {
                self.v_cmpx_lt_i32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_LT_U32 => {
                self.v_cmpx_lt_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_NGE_F64 => {
                self.v_cmpx_nge_f64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_NGT_F64 => {
                self.v_cmpx_ngt_f64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_NLT_F64 => {
                self.v_cmpx_nlt_f64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_GT_U64 => {
                self.v_cmp_gt_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_NGT_F32 => {
                self.v_cmp_ngt_f32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_EQ_F32 => {
                self.v_cmpx_eq_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_EQ_F64 => {
                self.v_cmpx_eq_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_EQ_I64 => {
                self.v_cmpx_eq_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_EQ_U64 => {
                self.v_cmpx_eq_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_GE_F32 => {
                self.v_cmpx_ge_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_GE_F64 => {
                self.v_cmpx_ge_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_GE_I64 => {
                self.v_cmpx_ge_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_GE_U32 => {
                self.v_cmpx_ge_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_GE_U64 => {
                self.v_cmpx_ge_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_GT_F32 => {
                self.v_cmpx_gt_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_GT_F64 => {
                self.v_cmpx_gt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_GT_I32 => {
                self.v_cmpx_gt_i32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_GT_I64 => {
                self.v_cmpx_gt_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_GT_U64 => {
                self.v_cmpx_gt_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_LE_F32 => {
                self.v_cmpx_le_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_LE_F64 => {
                self.v_cmpx_le_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_LE_I64 => {
                self.v_cmpx_le_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_LE_U32 => {
                self.v_cmpx_le_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_LE_U64 => {
                self.v_cmpx_le_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_LG_F32 => {
                self.v_cmpx_lg_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_LG_F64 => {
                self.v_cmpx_lg_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_LT_F32 => {
                self.v_cmpx_lt_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_LT_F64 => {
                self.v_cmpx_lt_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_LT_I64 => {
                self.v_cmpx_lt_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_LT_U64 => {
                self.v_cmpx_lt_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_NEQ_F32 => {
                self.v_cmpx_neq_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_NEQ_F64 => {
                self.v_cmpx_neq_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_NE_I64 => {
                self.v_cmpx_ne_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_NE_U64 => {
                self.v_cmpx_ne_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMPX_NGE_F32 => {
                self.v_cmpx_nge_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_NGT_F32 => {
                self.v_cmpx_ngt_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_NLE_F32 => {
                self.v_cmpx_nle_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_NLE_F64 => {
                self.v_cmpx_nle_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMPX_NLT_F32 => {
                self.v_cmpx_nlt_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_EQ_F32 => {
                self.v_cmp_eq_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_EQ_F64 => {
                self.v_cmp_eq_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_EQ_I64 => {
                self.v_cmp_eq_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_GE_F64 => {
                self.v_cmp_ge_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_GE_I64 => {
                self.v_cmp_ge_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_GE_U64 => {
                self.v_cmp_ge_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_GT_I64 => {
                self.v_cmp_gt_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_LE_I64 => {
                self.v_cmp_le_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_LE_U32 => {
                self.v_cmp_le_u32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_LE_U64 => {
                self.v_cmp_le_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_LT_I32 => {
                self.v_cmp_lt_i32_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_LT_I64 => {
                self.v_cmp_lt_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_NEQ_F32 => {
                self.v_cmp_neq_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NE_I64 => {
                self.v_cmp_ne_i64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_NE_U64 => {
                self.v_cmp_ne_u64_e64(d, s0, s1, abs, neg);
            }
            I::V_CMP_NGE_F32 => {
                self.v_cmp_nge_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NLE_F32 => {
                self.v_cmp_nle_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NLE_F64 => {
                self.v_cmp_nle_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_NLT_F32 => {
                self.v_cmp_nlt_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_O_F32 => {
                self.v_cmp_o_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_CMP_O_F64 => {
                self.v_cmp_o_f64_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_BFREV_B32 => {
                self.v_bfrev_b32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CEIL_F32 => {
                self.v_ceil_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CLZ_I32_U32 => {
                self.v_clz_i32_u32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_COS_F32 => {
                self.v_cos_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_SIN_F32 => {
                self.v_sin_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_EXP_F32 => {
                self.v_exp_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_LOG_F32 => {
                self.v_log_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_F16_F32 => {
                self.v_cvt_f16_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_F32_F64 => {
                self.v_cvt_f32_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_F32_I32 => {
                self.v_cvt_f32_i32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_F64_F32 => {
                self.v_cvt_f64_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_CVT_U32_F64 => {
                self.v_cvt_u32_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_FLOOR_F64 => {
                self.v_floor_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_TRUNC_F32 => {
                self.v_trunc_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_TRUNC_F64 => {
                self.v_trunc_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_FREXP_EXP_I32_F64 => {
                self.v_frexp_exp_i32_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_FREXP_MANT_F64 => {
                self.v_frexp_mant_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_NOT_B32 => {
                self.v_not_b32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_RSQ_F32 => {
                self.v_rsq_f32_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_SQRT_F64 => {
                self.v_sqrt_f64_e64(d, s0, abs, neg, clamp, omod);
            }
            I::V_MAX_I32 => {
                self.v_max_i32_e64(d, s0, s1, abs, neg);
            }
            I::V_MIN_I32 => {
                self.v_min_i32_e64(d, s0, s1, abs, neg);
            }
            I::V_MUL_I32_I24 => {
                self.v_mul_i32_i24_e64(d, s0, s1, abs, neg);
            }
            I::V_MUL_U32_U24 => {
                self.v_mul_u32_u24_e64(d, s0, s1, abs, neg);
            }
            I::V_SUBREV_NC_U32 => {
                self.v_subrev_nc_u32_e64(d, s0, s1, abs, neg, clamp);
            }
            I::V_MAX_NUM_F32 => {
                self.v_max_num_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_MIN_NUM_F32 => {
                self.v_min_num_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_SUB_F32 => {
                self.v_sub_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            I::V_SUBREV_F32 => {
                self.v_subrev_f32_e64(d, s0, s1, abs, neg, clamp, omod);
            }
            op => unimplemented!("{:?}", op),
        }
        Signals::None
    }

    fn v_add_nc_u16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u16 as u64,
                abs,
                neg,
                0,
                16,
            ) as u16;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u16 as u64,
                abs,
                neg,
                1,
                16,
            ) as u16;
            // CLAMP saturates the unsigned result instead of letting it wrap.
            let d_value = if clamp {
                s0_value.saturating_add(s1_value)
            } else {
                s0_value.wrapping_add(s1_value)
            };
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_lshlrev_b16_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as u16;
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

    fn v_and_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = s0_value & s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshl_or_b32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
            let d_value = (s0_value << (s1_value & 0x1F)) | s2_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_and_or_b32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
            let d_value = (s0_value & s1_value) | s2_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_bfe_u32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
            let d_value = (s0_value >> (s1_value & 0x1F)) & ((1 << (s2_value & 0x1F)) - 1);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_max_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = s0_value.max(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_min_u32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = s0_value.min(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_ashrrev_i32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
            let d_value = s1_value >> (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_cmp_eq_u16_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u16 as u64,
                abs,
                neg,
                0,
                16,
            ) as u16;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u16 as u64,
                abs,
                neg,
                1,
                16,
            ) as u16;
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

    fn v_cmp_gt_u16_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u16 as u64,
                abs,
                neg,
                0,
                16,
            ) as u16;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u16 as u64,
                abs,
                neg,
                1,
                16,
            ) as u16;
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

    fn v_cmp_lt_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_mul_lo_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = mul_u32(s0_value, s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_mul_hi_u32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = ((s0_value as u64 * s1_value as u64) >> 32) as u32;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_xor_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = s0_value ^ s1_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_or3_b32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
            let d_value = (s0_value | s1_value) | s2_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_xor3_b32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
            let d_value = (s0_value ^ s1_value) ^ s2_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add_nc_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            // CLAMP saturates the unsigned result instead of letting it wrap.
            let d_value = if clamp {
                s0_value.saturating_add(s1_value)
            } else {
                s0_value.wrapping_add(s1_value)
            };
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_sub_nc_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            // CLAMP saturates the unsigned result instead of letting it wrap.
            let d_value = if clamp {
                s0_value.saturating_sub(s1_value)
            } else {
                s0_value.wrapping_sub(s1_value)
            };
            self.write_vgpr(elem, d, d_value);
        }
    }

    /// The bit rotate of the 64-bit pair {S0, S1} by S2, keeping the low dword.
    fn v_alignbit_b32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
            let pair = u64_from_u32_u32(s1_value, s0_value);
            let d_value = (pair >> (s2_value & 0x1F)) as u32;
            self.write_vgpr(elem, d, d_value);
        }
    }

    /// The bitfield insert: S0 is the mask, and the result takes those bits
    /// from S1 and the rest from S2.
    fn v_bfi_b32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
            let d_value = (s0_value & s1_value) | (!s0_value & s2_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    /// The population count of S0, added to S1.
    fn v_bcnt_u32_b32(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = s0_value.count_ones().wrapping_add(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add3_u32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
            let (d_value, _) = add_u32(s0_value, s1_value, s2_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_mad_u32_u24(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;

            let s0_value = s0_value & 0xFFFFFF;
            let s1_value = s1_value & 0xFFFFFF;
            let d_value = s0_value.wrapping_mul(s1_value).wrapping_add(s2_value);

            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_add_f32_e64(
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
            let d_value = s0_value + s1_value;
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
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
            // LG = ORDERED not-equal (false if either is NaN); ISA §V_CMP_LG.
            let d_value = s0_value < s1_value || s0_value > s1_value;
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

    fn v_cvt_f32_u32_e64(
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
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let d_value = s0_value as f32;

            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_f64_u32_e64(
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
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
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

    fn v_cvt_f64_i32_e64(
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
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32 as i32;
            let d_value = s0_value as f64;

            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_i32_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = s0_value as i32;

            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_cvt_u32_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = s0_value as u32;

            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_fract_f64_e64(
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
            // FLOOR-based fractional part (ISA §V_FRACT): fract(-1.2) = 0.8.
            // Rust f64::fract() truncates, giving the wrong sign for negatives.
            // The result is in [0,1), so clamp: the subtraction rounds up to
            // exactly 1.0 for a tiny negative input.
            let frac = s0_value - s0_value.floor();
            let d_value = if frac >= 1.0 {
                f64::from_bits(0x3FEF_FFFF_FFFF_FFFF)
            } else {
                frac
            };

            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_frexp_exp_i32_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = if s0_value.is_nan() || s0_value.is_infinite() {
                0
            } else {
                libm::frexpf(s0_value).1
            };

            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_frexp_mant_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = if s0_value.is_nan() || s0_value.is_infinite() {
                s0_value
            } else {
                libm::frexpf(s0_value).0
            };

            self.write_vgpr(
                elem,
                d,
                f32_to_u32_omod_clamp(quiet_nan_f32(d_value), omod, clamp),
            );
        }
    }

    fn v_mov_b32_e64(
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
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let d_value = s0_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_rcp_iflag_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = ftz_f32(1.0 / ftz_f32(s0_value));

            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_rndne_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            // roundToIntegralTiesToEven. Rounding by hand as floor(x + 0.5)
            // loses the sign of a zero result, which the hardware keeps.
            let d_value = quiet_nan_f32(s0_value.round_ties_even());

            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_sqrt_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = ftz_f32(ftz_f32(s0_value).sqrt());

            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_f32_f16_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            // This encoding has no OPSEL, so the source is always the low half.
            let s0_value = self.read_vector_source_operand_f16(elem, s0);
            let d_value = s0_value.to_f32();
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_floor_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let mut d_value = s0_value.trunc();
            if (s0_value < 0.0) && (s0_value != d_value) {
                d_value += -1.0;
            }
            self.write_vgpr(elem, d, f32_to_u32(quiet_nan_f32(d_value)));
        }
    }

    fn v_ashrrev_i32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1) as i32;
            let d_value = s1_value >> (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_max_u32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value.max(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_min_num_f64_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value.min(s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_min_u32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value.min(s1_value);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cmp_class_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let s0_values = (0..32)
            .map(|elem| self.read_vector_source_operand_f32(elem, s0))
            .collect::<Vec<f32>>();
        let s1_values = (0..32)
            .map(|elem| self.read_vgpr(elem, s1))
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
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_class_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let s0_values = (0..32)
            .map(|elem| self.read_vector_source_operand_f64(elem, s0))
            .collect::<Vec<f64>>();
        let s1_values = (0..32)
            .map(|elem| self.read_vgpr(elem, s1))
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
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_eq_u16_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as u16;
            let s1_value = self.read_vgpr(elem, s1) as u16;
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ge_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_i32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let s1_value = self.read_vgpr(elem, s1) as i32;
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_u16_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as u16;
            let s1_value = self.read_vgpr(elem, s1) as u16;
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lg_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            // LG = ORDERED not-equal (false if either is NaN); ISA §V_CMP_LG.
            let d_value = s0_value < s1_value || s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lg_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            // LG = ORDERED not-equal (false if either is NaN); ISA §V_CMP_LG.
            let d_value = s0_value < s1_value || s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_neq_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value == s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nge_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_eq_u32_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_cmpx_gt_u32_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_cmpx_lt_i32_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32 as i32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
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

    fn v_cmpx_lt_u32_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_cmpx_nge_f64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ngt_f64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nlt_f64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_u64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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

    fn v_cmp_ngt_f32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let s1_value = abs_neg(self.read_vector_source_operand_f32(elem, s1), abs, neg, 1);
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

    fn v_cmpx_eq_f32_e64(
        &mut self,
        _d: usize,
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

    fn v_cmpx_eq_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
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

    fn v_cmpx_eq_f64_e64(
        &mut self,
        _d: usize,
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

    fn v_cmpx_eq_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
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

    fn v_cmpx_eq_i64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmpx_eq_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
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

    fn v_cmpx_eq_u64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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

    fn v_cmpx_eq_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_f32_e64(
        &mut self,
        _d: usize,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_f64_e64(
        &mut self,
        _d: usize,
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
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_i64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_u32_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_u64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ge_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_gt_f32_e64(
        &mut self,
        _d: usize,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_gt_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_gt_f64_e64(
        &mut self,
        _d: usize,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_gt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_gt_i32_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32 as i32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
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

    fn v_cmpx_gt_i32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let s1_value = self.read_vgpr(elem, s1) as i32;
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

    fn v_cmpx_gt_i64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmpx_gt_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
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

    fn v_cmpx_gt_u64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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

    fn v_cmpx_gt_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_f32_e64(
        &mut self,
        _d: usize,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_f64_e64(
        &mut self,
        _d: usize,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_i64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_u32_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_u64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_le_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lg_f32_e64(
        &mut self,
        _d: usize,
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
            // LG = ORDERED not-equal (false if either is NaN); ISA §V_CMP_LG.
            let d_value = s0_value < s1_value || s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lg_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            // LG = ORDERED not-equal (false if either is NaN); ISA §V_CMP_LG.
            let d_value = s0_value < s1_value || s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lg_f64_e64(
        &mut self,
        _d: usize,
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
            // LG = ORDERED not-equal (false if either is NaN); ISA §V_CMP_LG.
            let d_value = s0_value < s1_value || s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lg_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            // LG = ORDERED not-equal (false if either is NaN); ISA §V_CMP_LG.
            let d_value = s0_value < s1_value || s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lt_f32_e64(
        &mut self,
        _d: usize,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lt_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lt_f64_e64(
        &mut self,
        _d: usize,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lt_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_lt_i64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmpx_lt_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
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

    fn v_cmpx_lt_u64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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

    fn v_cmpx_lt_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
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

    fn v_cmpx_neq_f32_e64(
        &mut self,
        _d: usize,
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
            let d_value = !(s0_value == s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_neq_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = !(s0_value == s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_neq_f64_e64(
        &mut self,
        _d: usize,
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_neq_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value == s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_ne_i64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmpx_ne_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
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

    fn v_cmpx_ne_u64_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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

    fn v_cmpx_ne_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
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

    fn v_cmpx_nge_f32_e64(
        &mut self,
        _d: usize,
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

    fn v_cmpx_nge_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
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

    fn v_cmpx_ngt_f32_e64(
        &mut self,
        _d: usize,
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

    fn v_cmpx_ngt_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nle_f32_e64(
        &mut self,
        _d: usize,
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
            let d_value = !(s0_value <= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nle_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = !(s0_value <= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nle_f64_e64(
        &mut self,
        _d: usize,
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
            let d_value = !(s0_value <= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nle_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value <= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_exec_bit(elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmpx_nlt_f32_e64(
        &mut self,
        _d: usize,
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

    fn v_cmpx_nlt_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
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

    fn v_cmp_eq_f32_e64(
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

    fn v_cmp_eq_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_eq_f64_e64(
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

    fn v_cmp_eq_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_eq_i64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmp_eq_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
            let d_value = s0_value == s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ge_f64_e64(
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

    fn v_cmp_ge_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ge_i64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmp_ge_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ge_u64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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

    fn v_cmp_ge_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s0_value >= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_gt_i64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmp_gt_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
            let d_value = s0_value > s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_le_i64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmp_le_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_le_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_cmp_le_u32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_le_u64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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

    fn v_cmp_le_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s0_value <= s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_i32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32 as i32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
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

    fn v_cmp_lt_i32_e32(&mut self, s0: SourceOperand, s1: usize) {
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
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_lt_i64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmp_lt_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
            let d_value = s0_value < s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_neq_f32_e64(
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

    fn v_cmp_neq_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = !(s0_value == s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ne_i64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64 as i64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64 as i64;
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

    fn v_cmp_ne_i64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0) as i64;
            let s1_value = self.read_vgpr_pair(elem, s1) as i64;
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_ne_u64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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

    fn v_cmp_ne_u64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u64(elem, s0);
            let s1_value = self.read_vgpr_pair(elem, s1);
            let d_value = s0_value != s1_value;
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nge_f32_e64(
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

    fn v_cmp_nge_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = !(s0_value >= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nle_f32_e64(
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
            let d_value = !(s0_value <= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nle_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = !(s0_value <= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nle_f64_e64(
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
            let d_value = !(s0_value <= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nle_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !(s0_value <= s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_nlt_f32_e64(
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

    fn v_cmp_nlt_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = !(s0_value < s1_value);
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_o_f32_e64(
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
            let d_value = !s0_value.is_nan() && !s1_value.is_nan();
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_o_f32_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = !s0_value.is_nan() && !s1_value.is_nan();
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_o_f64_e64(
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
            let d_value = !s0_value.is_nan() && !s1_value.is_nan();
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(d, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_cmp_o_f64_e32(&mut self, s0: SourceOperand, s1: usize) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let s1_value = u64_to_f64(self.read_vgpr_pair(elem, s1));
            let d_value = !s0_value.is_nan() && !s1_value.is_nan();
            vcc |= (d_value as u32) << elem;
        }
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            self.set_sgpr_bit(106, elem, ((vcc >> elem) & 1) != 0);
        }
    }

    fn v_bfrev_b32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = s0_value.reverse_bits();
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_bfrev_b32_e64(
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
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let d_value = s0_value.reverse_bits();
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_ceil_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = quiet_nan_f32(s0_value.ceil());
            self.write_vgpr(elem, d, f32_to_u32(quiet_nan_f32(d_value)));
        }
    }

    fn v_ceil_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = quiet_nan_f32(s0_value.ceil());
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_clz_i32_u32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = if s0_value == 0 {
                -1i32 as u32
            } else {
                s0_value.leading_zeros()
            };
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_clz_i32_u32_e64(
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
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let d_value = if s0_value == 0 {
                -1i32 as u32
            } else {
                s0_value.leading_zeros()
            };
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cos_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            // The input is in revolutions; see V_SIN_F32 for the reduction.
            // Reduce to the nearest turn; see V_SIN_F32.
            let turns = s0_value - s0_value.round_ties_even();
            let d_value = if turns == 0.0 {
                1.0
            } else if turns.abs() == 0.25 {
                0.0
            } else if turns.abs() == 0.5 {
                -1.0
            } else {
                (turns * std::f32::consts::TAU).cos()
            };
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_cos_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            // The input is in revolutions; see V_SIN_F32 for the reduction.
            // Reduce to the nearest turn; see V_SIN_F32.
            let turns = s0_value - s0_value.round_ties_even();
            let d_value = if turns == 0.0 {
                1.0
            } else if turns.abs() == 0.25 {
                0.0
            } else if turns.abs() == 0.5 {
                -1.0
            } else {
                (turns * std::f32::consts::TAU).cos()
            };
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_sin_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            // The input is in revolutions. The hardware reduces it to one turn before
            // scaling, so a huge argument still lands on an exact value, and the
            // quarter turns are exact rather than the 1e-16 a scaled sine gives
            // (ISA §V_SIN_F32 functional examples).
            let d_value = if s0_value == 0.0 {
                s0_value
            } else {
                // Reduce to the nearest turn, which keeps a tiny argument exact
                // rather than folding it against 1.0.
                let turns = s0_value - s0_value.round_ties_even();
                if turns == 0.0 {
                    0.0
                } else if turns == 0.25 {
                    1.0
                } else if turns == -0.25 {
                    -1.0
                } else if turns.abs() == 0.5 {
                    0.0
                } else {
                    (turns * std::f32::consts::TAU).sin()
                }
            };
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_sin_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            // The input is in revolutions. The hardware reduces it to one turn before
            // scaling, so a huge argument still lands on an exact value, and the
            // quarter turns are exact rather than the 1e-16 a scaled sine gives
            // (ISA §V_SIN_F32 functional examples).
            let d_value = if s0_value == 0.0 {
                s0_value
            } else {
                // Reduce to the nearest turn, which keeps a tiny argument exact
                // rather than folding it against 1.0.
                let turns = s0_value - s0_value.round_ties_even();
                if turns == 0.0 {
                    0.0
                } else if turns == 0.25 {
                    1.0
                } else if turns == -0.25 {
                    -1.0
                } else if turns.abs() == 0.5 {
                    0.0
                } else {
                    (turns * std::f32::consts::TAU).sin()
                }
            };
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_exp_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = ftz_f32(ftz_f32(s0_value).exp2());
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_exp_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = ftz_f32(ftz_f32(s0_value).exp2());
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_log_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = ftz_f32(ftz_f32(s0_value).log2());
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_log_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = ftz_f32(ftz_f32(s0_value).log2());
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_f16_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = f16::from_f32(s0_value);
            self.write_vgpr(elem, d, d_value.to_bits() as u32);
        }
    }

    fn v_cvt_f16_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = f16::from_f32(s0_value);
            self.write_vgpr(elem, d, f16_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_f32_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = s0_value as f32;
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_cvt_f32_f64_e64(
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
            let d_value = s0_value as f32;
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_f32_i32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let d_value = s0_value as f32;
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_cvt_f32_i32_e64(
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
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32 as i32;
            let d_value = s0_value as f32;
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_f64_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = s0_value as f64;
            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_cvt_f64_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = s0_value as f64;
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_cvt_u32_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = s0_value as u32;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cvt_u32_f64_e64(
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
            let d_value = s0_value as u32;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_floor_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = quiet_nan_f64(s0_value.floor());
            self.write_vgpr_pair(elem, d, f64_to_u64(quiet_nan_f64(d_value)));
        }
    }

    fn v_floor_f64_e64(
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
            let d_value = quiet_nan_f64(s0_value.floor());
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_trunc_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = quiet_nan_f32(s0_value.trunc());
            self.write_vgpr(elem, d, f32_to_u32(quiet_nan_f32(d_value)));
        }
    }

    fn v_trunc_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = quiet_nan_f32(s0_value.trunc());
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_trunc_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = quiet_nan_f64(s0_value.trunc());
            self.write_vgpr_pair(elem, d, f64_to_u64(quiet_nan_f64(d_value)));
        }
    }

    fn v_trunc_f64_e64(
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
            let d_value = quiet_nan_f64(s0_value.trunc());
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_frexp_exp_i32_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = if s0_value.is_nan() || s0_value.is_infinite() {
                0
            } else {
                libm::frexp(s0_value).1 as u32
            };
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_frexp_exp_i32_f64_e64(
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
            let d_value = if s0_value.is_nan() || s0_value.is_infinite() {
                0
            } else {
                libm::frexp(s0_value).1 as u32
            };
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_frexp_mant_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = if s0_value.is_nan() || s0_value.is_infinite() {
                s0_value
            } else {
                libm::frexp(s0_value).0
            };
            self.write_vgpr_pair(elem, d, f64_to_u64(quiet_nan_f64(d_value)));
        }
    }

    fn v_frexp_mant_f64_e64(
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
            let d_value = if s0_value.is_nan() || s0_value.is_infinite() {
                s0_value
            } else {
                libm::frexp(s0_value).0
            };
            self.write_vgpr_pair(
                elem,
                d,
                f64_to_u64_omod_clamp(quiet_nan_f64(d_value), omod, clamp),
            );
        }
    }

    fn v_not_b32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let d_value = !s0_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_not_b32_e64(
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
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let d_value = !s0_value;
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_rsq_f32_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let d_value = ftz_f32(1.0 / ftz_f32(s0_value).sqrt());
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_rsq_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = ftz_f32(1.0 / ftz_f32(s0_value).sqrt());
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_sqrt_f64_e32(&mut self, d: usize, s0: SourceOperand) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f64(elem, s0);
            let d_value = s0_value.sqrt();
            self.write_vgpr_pair(elem, d, f64_to_u64(d_value));
        }
    }

    fn v_sqrt_f64_e64(
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
            let d_value = s0_value.sqrt();
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_max_i32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let s1_value = self.read_vgpr(elem, s1) as i32;
            let d_value = s0_value.max(s1_value);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_max_i32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32 as i32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
            let d_value = s0_value.max(s1_value);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_min_i32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let s1_value = self.read_vgpr(elem, s1) as i32;
            let d_value = s0_value.min(s1_value);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_min_i32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32 as i32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
            let d_value = s0_value.min(s1_value);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_max_num_f32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value.max(s1_value);
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_max_num_f32_e64(
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
            let d_value = s0_value.max(s1_value);
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_min_num_f32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value.min(s1_value);
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_min_num_f32_e64(
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
            let d_value = s0_value.min(s1_value);
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_mul_i32_i24_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0) as i32;
            let s1_value = self.read_vgpr(elem, s1) as i32;
            let d_value = (s0_value << 8 >> 8).wrapping_mul(s1_value << 8 >> 8);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_mul_i32_i24_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32 as i32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
            let d_value = (s0_value << 8 >> 8).wrapping_mul(s1_value << 8 >> 8);
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    fn v_subrev_f32_e32(&mut self, d: usize, s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            // The hardware subtracts by negating the operand and adding, so a NaN
            // operand reaches the result with its sign flipped.
            let d_value = sub_f32(s1_value, s0_value);
            self.write_vgpr(elem, d, f32_to_u32(d_value));
        }
    }

    fn v_subrev_f32_e64(
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
            // The hardware subtracts by negating the operand and adding, so a NaN
            // operand reaches the result with its sign flipped.
            let d_value = sub_f32(s1_value, s0_value);
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_rcp_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let d_value = ftz_f32(1.0 / ftz_f32(s0_value));
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
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
            let d_value = quiet_nan_f64(s0_value.round_ties_even());
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
            // LG = ORDERED not-equal (false if either is NaN); ISA §V_CMP_LG.
            let d_value = s0_value < s1_value || s0_value > s1_value;
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

    fn v_lshlrev_b32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = s1_value << (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshrrev_b32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let d_value = s1_value >> (s0_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_lshlrev_b64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1),
                abs,
                neg,
                1,
                64,
            ) as u64;
            let d_value = s1_value << (s0_value & 0x3F);
            self.write_vgpr_pair(elem, d, d_value);
        }
    }

    fn v_lshrrev_b64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1),
                abs,
                neg,
                1,
                64,
            );
            let d_value = s1_value >> (s0_value & 0x3F);
            self.write_vgpr_pair(elem, d, d_value);
        }
    }

    fn v_or_b32_e64(&mut self, d: usize, s0: SourceOperand, s1: SourceOperand, abs: u8, neg: u8) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
            let d_value = libm::ldexp(s0_value, s1_value);
            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_rsq_f64_e64(
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
            // Apply VOP3 input abs/neg and output clamp/omod modifiers, like the
            // sibling v_rcp_f64_e64 (they were previously ignored).
            let s0_value = abs_neg(self.read_vector_source_operand_f64(elem, s0), abs, neg, 0);
            let d_value = 1.0 / s0_value.sqrt();

            self.write_vgpr_pair(elem, d, f64_to_u64_omod_clamp(d_value, omod, clamp));
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

    fn v_xad_u32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
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
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
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
        abs: u8,
        neg: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s2) as u64,
                abs,
                neg,
                2,
                32,
            ) as u32;
            let d_value = s0_value.wrapping_add(s1_value) << (s2_value & 0x1F);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_cmp_ne_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_cmp_eq_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_cmp_gt_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_cmp_ge_u32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_cmpx_ne_u32_e64(
        &mut self,
        _d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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

    fn v_cmp_gt_i32_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32 as i32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
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

    fn v_cmp_lt_u64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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

    fn v_cmp_eq_u64_e64(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s0) as u64,
                abs,
                neg,
                0,
                64,
            ) as u64;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u64(elem, s1) as u64,
                abs,
                neg,
                1,
                64,
            ) as u64;
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
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;

            let mut shift = (s1_value & 0x1F) as i32 * 53;
            if get_exp_f64(s0_value) > 1077 {
                shift += get_exp_f64(s0_value) as i32 - 1077;
            }

            // A shift past the end of the table asks for bits the fraction no
            // longer has, which are zero.
            let offset = 1201 - 53 - shift;
            let result = if offset < 0 {
                0
            } else {
                get_bits_u64(&TWO_OVER_PI_FRACTION, offset as usize, 53)
            };
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
        opsel: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            // RDNA4 ISA: OPSEL[0] selects src0's f16 half (1=high, 0=low).
            let raw = if opsel & 1 != 0 {
                self.read_vector_source_operand_f16_hi(elem, s0)
            } else {
                self.read_vector_source_operand_f16(elem, s0)
            };
            let s0_value = abs_neg_f16(raw, abs, neg, 0);
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
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32 as i32;
            let d_value = libm::ldexpf(s0_value, s1_value);
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_fmac_f32_e64(
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
            let d_value = u32_to_f32(self.read_vgpr(elem, d));
            let d_value = fma(s0_value, s1_value, d_value);
            self.write_vgpr(elem, d, f32_to_u32_omod_clamp(d_value, omod, clamp));
        }
    }

    fn v_floor_f32_e64(
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
            let s0_value = abs_neg(self.read_vector_source_operand_f32(elem, s0), abs, neg, 0);
            let mut d_value = s0_value.trunc();
            if (s0_value < 0.0) && (s0_value != d_value) {
                d_value += -1.0;
            }
            self.write_vgpr(
                elem,
                d,
                f32_to_u32_omod_clamp(quiet_nan_f32(d_value), omod, clamp),
            );
        }
    }

    fn v_s_rcp_f32(
        &mut self,
        d: usize,
        s0: SourceOperand,
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    ) {
        let s0_value = abs_neg(
            u32_to_f32(self.read_scalar_source_operand_u32(s0)),
            abs,
            neg,
            0,
        );
        let d_value = 1.0 / s0_value;
        self.write_sgpr(d, f32_to_u32_omod_clamp(d_value, omod, clamp));
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
                // VOP3SD spends the ABS field on SDST, so only NEG reaches the sources.
                self.v_add_co_ci_u32_e64(d0, d1, s0, s1, s2, 0, neg);
            }
            I::V_SUB_CO_U32 => {
                self.v_sub_co_u32(d0, d1, s0, s1);
            }
            I::V_SUBREV_CO_U32 => {
                self.v_subrev_co_u32(d0, d1, s0, s1);
            }
            I::V_SUB_CO_CI_U32 => {
                // VOP3SD spends the ABS field on SDST, so only NEG reaches the sources.
                self.v_sub_co_ci_u32_e64(d0, d1, s0, s1, s2, 0, neg);
            }
            I::V_SUBREV_CO_CI_U32 => {
                self.v_subrev_co_ci_u32_e64(d0, d1, s0, s1, s2, 0, neg);
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
            op => unimplemented!("{:?}", op),
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

    /// ISA §V_SUB_CO_U32: the borrow out lands in the scalar destination.
    fn v_sub_co_u32(&mut self, d0: usize, d1: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (d0_value, d1_value) = sub_u32(s0_value, s1_value, 0);
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

    /// ISA §V_SUBREV_CO_U32: the same, with the sources the other way round.
    fn v_subrev_co_u32(&mut self, d0: usize, d1: usize, s0: SourceOperand, s1: SourceOperand) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (d0_value, d1_value) = sub_u32(s1_value, s0_value, 0);
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

    /// ISA §V_SUB_CO_CI_U32: the third source is the borrow in, one bit per
    /// lane, and the borrow out goes to the scalar destination.
    fn v_sub_co_ci_u32_e64(
        &mut self,
        d0: usize,
        d1: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = self.read_scalar_source_operand_u32(s2);
            let (d0_value, d1_value) = sub_u32(s0_value, s1_value, (s2_value >> elem) & 1);
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

    /// ISA §V_SUBREV_CO_CI_U32: the same, with the sources the other way round.
    fn v_subrev_co_ci_u32_e64(
        &mut self,
        d0: usize,
        d1: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
            let s2_value = self.read_scalar_source_operand_u32(s2);
            let (d0_value, d1_value) = sub_u32(s1_value, s0_value, (s2_value >> elem) & 1);
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
        abs: u8,
        neg: u8,
    ) {
        let mut vcc = 0u32;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s0) as u64,
                abs,
                neg,
                0,
                32,
            ) as u32;
            let s1_value = abs_neg_bits(
                self.read_vector_source_operand_u32(elem, s1) as u64,
                abs,
                neg,
                1,
                32,
            ) as u32;
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
            I::V_PK_ADD_F16 => {
                self.v_pk_add_f16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MUL_F16 => {
                self.v_pk_mul_f16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_FMA_F16 => {
                self.v_pk_fma_f16(d, s0, s1, s2, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MIN_NUM_F16 => {
                self.v_pk_min_num_f16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MAX_NUM_F16 => {
                self.v_pk_max_num_f16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MINIMUM_F16 => {
                self.v_pk_minimum_f16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MAXIMUM_F16 => {
                self.v_pk_maximum_f16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_ADD_U16 => {
                self.v_pk_add_u16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_SUB_U16 => {
                self.v_pk_sub_u16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MAD_U16 => {
                self.v_pk_mad_u16(d, s0, s1, s2, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MAX_U16 => {
                self.v_pk_max_u16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MIN_U16 => {
                self.v_pk_min_u16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_ADD_I16 => {
                self.v_pk_add_i16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_SUB_I16 => {
                self.v_pk_sub_i16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MAD_I16 => {
                self.v_pk_mad_i16(d, s0, s1, s2, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MAX_I16 => {
                self.v_pk_max_i16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MIN_I16 => {
                self.v_pk_min_i16(d, s0, s1, neg, neg_hi, clamp, opsel, opsel_hi);
            }
            I::V_PK_MUL_LO_U16 => {
                self.v_pk_mul_lo_u16(d, s0, s1, neg, neg_hi, opsel, opsel_hi);
            }
            I::V_PK_LSHLREV_B16 => {
                self.v_pk_lshlrev_b16(d, s0, s1, neg, neg_hi, opsel, opsel_hi);
            }
            I::V_PK_LSHRREV_B16 => {
                self.v_pk_lshrrev_b16(d, s0, s1, neg, neg_hi, opsel, opsel_hi);
            }
            I::V_PK_ASHRREV_I16 => {
                self.v_pk_ashrrev_i16(d, s0, s1, neg, neg_hi, opsel, opsel_hi);
            }
            I::V_DOT2_F32_F16 => {
                self.v_dot2_f32_f16(d, s0, s1, s2, neg, neg_hi, opsel, opsel_hi);
            }
            I::V_DOT2_F32_BF16 => {
                self.v_dot2_f32_bf16(d, s0, s1, s2, neg, neg_hi, opsel, opsel_hi);
            }
            I::V_DOT4_U32_U8 => {
                self.v_dot4_u32_u8(d, s0, s1, s2, opsel, opsel_hi);
            }
            I::V_DOT4_I32_IU8 => {
                self.v_dot4_i32_iu8(d, s0, s1, s2, neg, opsel, opsel_hi);
            }
            I::V_DOT8_U32_U4 => {
                self.v_dot8_u32_u4(d, s0, s1, s2, opsel, opsel_hi);
            }
            I::V_DOT8_I32_IU4 => {
                self.v_dot8_i32_iu4(d, s0, s1, s2, neg, opsel, opsel_hi);
            }
            I::V_FMA_MIX_F32 => {
                self.v_fma_mix_f32(d, s0, s1, s2, neg_hi, neg, clamp, opsel, opsel_hi);
            }
            I::V_FMA_MIXLO_F16 => {
                self.v_fma_mixlo_f16(d, s0, s1, s2, neg_hi, neg, clamp, opsel, opsel_hi);
            }
            I::V_FMA_MIXHI_F16 => {
                self.v_fma_mixhi_f16(d, s0, s1, s2, neg_hi, neg, clamp, opsel, opsel_hi);
            }
            I::V_WMMA_F32_16X16X16_F16 => {
                self.v_wmma_f32_16x16x16_f16(d, s0, s1, s2);
            }
            op => unimplemented!("{:?}", op),
        }
        Signals::None
    }

    /// The sum is exact in double precision, so the half sees one rounding.
    fn v_pk_add_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low =
                f16::from_f64(f16::from_bits(s0_low).to_f64() + f16::from_bits(s1_low).to_f64());
            let low = clamp_f16(low, clamp).to_bits();
            let high =
                f16::from_f64(f16::from_bits(s0_high).to_f64() + f16::from_bits(s1_high).to_f64());
            let high = clamp_f16(high, clamp).to_bits();
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_mul_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low =
                f16::from_f64(f16::from_bits(s0_low).to_f64() * f16::from_bits(s1_low).to_f64());
            let low = clamp_f16(low, clamp).to_bits();
            let high =
                f16::from_f64(f16::from_bits(s0_high).to_f64() * f16::from_bits(s1_high).to_f64());
            let high = clamp_f16(high, clamp).to_bits();
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_fma_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let (s2_low, s2_high) = packed_halves(s2_value, opsel, opsel_hi, neg, neg_hi, 2);
            let low = f16::from_f64(
                f16::from_bits(s0_low).to_f64() * f16::from_bits(s1_low).to_f64()
                    + f16::from_bits(s2_low).to_f64(),
            );
            let low = clamp_f16(low, clamp).to_bits();
            let high = f16::from_f64(
                f16::from_bits(s0_high).to_f64() * f16::from_bits(s1_high).to_f64()
                    + f16::from_bits(s2_high).to_f64(),
            );
            let high = clamp_f16(high, clamp).to_bits();
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_min_num_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = min_num_f16(f16::from_bits(s0_low), f16::from_bits(s1_low));
            let low = clamp_f16(low, clamp).to_bits();
            let high = min_num_f16(f16::from_bits(s0_high), f16::from_bits(s1_high));
            let high = clamp_f16(high, clamp).to_bits();
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_max_num_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = max_num_f16(f16::from_bits(s0_low), f16::from_bits(s1_low));
            let low = clamp_f16(low, clamp).to_bits();
            let high = max_num_f16(f16::from_bits(s0_high), f16::from_bits(s1_high));
            let high = clamp_f16(high, clamp).to_bits();
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_minimum_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = minimum_f16(f16::from_bits(s0_low), f16::from_bits(s1_low));
            let low = clamp_f16(low, clamp).to_bits();
            let high = minimum_f16(f16::from_bits(s0_high), f16::from_bits(s1_high));
            let high = clamp_f16(high, clamp).to_bits();
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_maximum_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = maximum_f16(f16::from_bits(s0_low), f16::from_bits(s1_low));
            let low = clamp_f16(low, clamp).to_bits();
            let high = maximum_f16(f16::from_bits(s0_high), f16::from_bits(s1_high));
            let high = clamp_f16(high, clamp).to_bits();
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_add_u16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = clamp_u16(s0_low as i64 + s1_low as i64, clamp);
            let high = clamp_u16(s0_high as i64 + s1_high as i64, clamp);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_sub_u16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = clamp_u16(s0_low as i64 - s1_low as i64, clamp);
            let high = clamp_u16(s0_high as i64 - s1_high as i64, clamp);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_mad_u16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let (s2_low, s2_high) = packed_halves(s2_value, opsel, opsel_hi, neg, neg_hi, 2);
            let low = clamp_u16(s0_low as i64 * s1_low as i64 + s2_low as i64, clamp);
            let high = clamp_u16(s0_high as i64 * s1_high as i64 + s2_high as i64, clamp);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_max_u16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = clamp_u16((s0_low as i64).max(s1_low as i64), clamp);
            let high = clamp_u16((s0_high as i64).max(s1_high as i64), clamp);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_min_u16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = clamp_u16((s0_low as i64).min(s1_low as i64), clamp);
            let high = clamp_u16((s0_high as i64).min(s1_high as i64), clamp);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_add_i16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = clamp_i16(s0_low as i16 as i64 + s1_low as i16 as i64, clamp);
            let high = clamp_i16(s0_high as i16 as i64 + s1_high as i16 as i64, clamp);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_sub_i16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = clamp_i16(s0_low as i16 as i64 - s1_low as i16 as i64, clamp);
            let high = clamp_i16(s0_high as i16 as i64 - s1_high as i16 as i64, clamp);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_mad_i16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let s2_value = self.read_vector_source_operand_u32(elem, s2);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let (s2_low, s2_high) = packed_halves(s2_value, opsel, opsel_hi, neg, neg_hi, 2);
            let low = clamp_i16(
                s0_low as i16 as i64 * s1_low as i16 as i64 + s2_low as i16 as i64,
                clamp,
            );
            let high = clamp_i16(
                s0_high as i16 as i64 * s1_high as i16 as i64 + s2_high as i16 as i64,
                clamp,
            );
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_max_i16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = clamp_i16((s0_low as i16 as i64).max(s1_low as i16 as i64), clamp);
            let high = clamp_i16((s0_high as i16 as i64).max(s1_high as i16 as i64), clamp);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_min_i16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        clamp: bool,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = clamp_i16((s0_low as i16 as i64).min(s1_low as i16 as i64), clamp);
            let high = clamp_i16((s0_high as i16 as i64).min(s1_high as i16 as i64), clamp);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    /// The low half is what this one keeps, saturation or not.
    fn v_pk_mul_lo_u16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = s0_low.wrapping_mul(s1_low);
            let high = s0_high.wrapping_mul(s1_high);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    /// The shift amount is the first source; the value is the second.
    fn v_pk_lshlrev_b16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = s1_low << (s0_low & 15);
            let high = s1_high << (s0_high & 15);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_lshrrev_b16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = s1_low >> (s0_low & 15);
            let high = s1_high >> (s0_high & 15);
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    fn v_pk_ashrrev_i16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        neg: u8,
        neg_hi: u8,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let low = ((s1_low as i16) >> (s0_low & 15)) as u16;
            let high = ((s1_high as i16) >> (s0_high & 15)) as u16;
            let d_value = u32_from_u16_u16(low, high);
            self.write_vgpr(elem, d, d_value);
        }
    }

    /// The addend is a whole dword rather than a pair of halves, and either of
    /// its sign bits negates it. CLAMP has no say over a dot product's result.
    fn v_dot2_f32_f16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        neg: u8,
        neg_hi: u8,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let s2_value = self.read_vector_source_operand_f32(elem, s2);
            let s2_value = if ((neg | neg_hi) >> 2) & 1 != 0 {
                -s2_value
            } else {
                s2_value
            };
            // The products and the addend are exact in double precision, so
            // this is the sum before the accumulator rounds it.
            let sum = s2_value as f64
                + f16::from_bits(s0_low).to_f64() * f16::from_bits(s1_low).to_f64()
                + f16::from_bits(s0_high).to_f64() * f16::from_bits(s1_high).to_f64();
            self.write_vgpr(elem, d, f32_to_u32(sum as f32));
        }
    }

    fn v_dot2_f32_bf16(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        neg: u8,
        neg_hi: u8,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vector_source_operand_u32(elem, s1);
            let (s0_low, s0_high) = packed_halves(s0_value, opsel, opsel_hi, neg, neg_hi, 0);
            let (s1_low, s1_high) = packed_halves(s1_value, opsel, opsel_hi, neg, neg_hi, 1);
            let s2_value = self.read_vector_source_operand_f32(elem, s2);
            let s2_value = if ((neg | neg_hi) >> 2) & 1 != 0 {
                -s2_value
            } else {
                s2_value
            };
            let sum = s2_value as f64
                + bf16_to_f32(s0_low) as f64 * bf16_to_f32(s1_low) as f64
                + bf16_to_f32(s0_high) as f64 * bf16_to_f32(s1_high) as f64;
            self.write_vgpr(elem, d, f32_to_u32(sum as f32));
        }
    }

    /// Four terms of eight bits each, added to the 32-bit third source.
    fn v_dot4_u32_u8(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = packed_dword(
                self.read_vector_source_operand_u32(elem, s0),
                opsel,
                opsel_hi,
                0,
            );
            let s1_value = packed_dword(
                self.read_vector_source_operand_u32(elem, s1),
                opsel,
                opsel_hi,
                1,
            );
            let mut d_value = self.read_vector_source_operand_u32(elem, s2) as i32 as i64;
            for term in 0..4 {
                let shift = term * 8;
                d_value += ((s0_value >> shift) as u8 as i64) * ((s1_value >> shift) as u8 as i64);
            }
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    /// The same four terms, except that NEG says whether each multiplied source
    /// is read as signed rather than negating it.
    fn v_dot4_i32_iu8(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        neg: u8,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = packed_dword(
                self.read_vector_source_operand_u32(elem, s0),
                opsel,
                opsel_hi,
                0,
            );
            let s1_value = packed_dword(
                self.read_vector_source_operand_u32(elem, s1),
                opsel,
                opsel_hi,
                1,
            );
            let s0_signed = (neg & 1) != 0;
            let s1_signed = (neg & 2) != 0;
            let mut d_value = self.read_vector_source_operand_u32(elem, s2) as i32 as i64;
            for term in 0..4 {
                let shift = term * 8;
                let a = (s0_value >> shift) as u8;
                let b = (s1_value >> shift) as u8;
                let a = if s0_signed { a as i8 as i64 } else { a as i64 };
                let b = if s1_signed { b as i8 as i64 } else { b as i64 };
                d_value += a * b;
            }
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    /// Eight terms of four bits each, added to the 32-bit third source.
    fn v_dot8_u32_u4(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = packed_dword(
                self.read_vector_source_operand_u32(elem, s0),
                opsel,
                opsel_hi,
                0,
            );
            let s1_value = packed_dword(
                self.read_vector_source_operand_u32(elem, s1),
                opsel,
                opsel_hi,
                1,
            );
            let mut d_value = self.read_vector_source_operand_u32(elem, s2) as i32 as i64;
            for term in 0..8 {
                let shift = term * 4;
                d_value +=
                    (((s0_value >> shift) & 0xF) as i64) * (((s1_value >> shift) & 0xF) as i64);
            }
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    /// The same eight terms, with NEG picking a signed reading of each source.
    fn v_dot8_i32_iu4(
        &mut self,
        d: usize,
        s0: SourceOperand,
        s1: SourceOperand,
        s2: SourceOperand,
        neg: u8,
        opsel: u8,
        opsel_hi: u8,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = packed_dword(
                self.read_vector_source_operand_u32(elem, s0),
                opsel,
                opsel_hi,
                0,
            );
            let s1_value = packed_dword(
                self.read_vector_source_operand_u32(elem, s1),
                opsel,
                opsel_hi,
                1,
            );
            let s0_signed = (neg & 1) != 0;
            let s1_signed = (neg & 2) != 0;
            let mut d_value = self.read_vector_source_operand_u32(elem, s2) as i32 as i64;
            for term in 0..8 {
                let shift = term * 4;
                let a = (s0_value >> shift) & 0xF;
                let b = (s1_value >> shift) & 0xF;
                let a = if s0_signed && a & 0x8 != 0 {
                    a as i64 - 16
                } else {
                    a as i64
                };
                let b = if s1_signed && b & 0x8 != 0 {
                    b as i64 - 16
                } else {
                    b as i64
                };
                d_value += a * b;
            }
            self.write_vgpr(elem, d, d_value as u32);
        }
    }

    /// The mixed-precision fused multiply-add, whole-width result. Each source
    /// is an f32 or one half of a dword: {OPSEL_HI[i], OPSEL[i]} selects it,
    /// OPSEL_HI=0 meaning an f32 and OPSEL otherwise picking which half of the
    /// dword to widen. NEG_HI is the abs modifier here (passed as `abs`), and
    /// NEG the sign one.
    fn v_fma_mix_f32(
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
            let s0_value = self.read_vector_source_operand_mix_f32(elem, s0, 0, opsel, opsel_hi);
            let s0_value = abs_neg(s0_value, abs, neg, 0);
            let s1_value = self.read_vector_source_operand_mix_f32(elem, s1, 1, opsel, opsel_hi);
            let s1_value = abs_neg(s1_value, abs, neg, 1);
            let s2_value = self.read_vector_source_operand_mix_f32(elem, s2, 2, opsel, opsel_hi);
            let s2_value = abs_neg(s2_value, abs, neg, 2);

            let d_value = fma(s0_value, s1_value, s2_value);

            self.write_vgpr(elem, d, f32_to_u32(clamp_f32(d_value, clamp)));
        }
    }

    /// `D0[15:0].f16 = f32_to_f16(fma)`: the low half is written and the high
    /// half of the destination is preserved.
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
            let s0_value = self.read_vector_source_operand_mix_f32(elem, s0, 0, opsel, opsel_hi);
            let s0_value = abs_neg(s0_value, abs, neg, 0);
            let s1_value = self.read_vector_source_operand_mix_f32(elem, s1, 1, opsel, opsel_hi);
            let s1_value = abs_neg(s1_value, abs, neg, 1);
            let s2_value = self.read_vector_source_operand_mix_f32(elem, s2, 2, opsel, opsel_hi);
            let s2_value = abs_neg(s2_value, abs, neg, 2);

            let d_value = f16::from_f32(clamp_f32(fma(s0_value, s1_value, s2_value), clamp));

            let kept = self.read_vgpr(elem, d) & 0xFFFF_0000;
            self.write_vgpr(elem, d, kept | d_value.to_bits() as u32);
        }
    }

    /// `D0[31:16].f16 = f32_to_f16(fma)`: the high half is written and the low
    /// half of the destination is preserved.
    fn v_fma_mixhi_f16(
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
            let s0_value = self.read_vector_source_operand_mix_f32(elem, s0, 0, opsel, opsel_hi);
            let s0_value = abs_neg(s0_value, abs, neg, 0);
            let s1_value = self.read_vector_source_operand_mix_f32(elem, s1, 1, opsel, opsel_hi);
            let s1_value = abs_neg(s1_value, abs, neg, 1);
            let s2_value = self.read_vector_source_operand_mix_f32(elem, s2, 2, opsel, opsel_hi);
            let s2_value = abs_neg(s2_value, abs, neg, 2);

            let d_value = f16::from_f32(clamp_f32(fma(s0_value, s1_value, s2_value), clamp));

            let kept = self.read_vgpr(elem, d) & 0x0000_FFFF;
            self.write_vgpr(elem, d, kept | ((d_value.to_bits() as u32) << 16));
        }
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
            I::V_CMP_CLASS_F32 => {
                self.v_cmp_class_f32_e32(s0, s1);
            }
            I::V_CMP_CLASS_F64 => {
                self.v_cmp_class_f64_e32(s0, s1);
            }
            I::V_CMP_EQ_U16 => {
                self.v_cmp_eq_u16_e32(s0, s1);
            }
            I::V_CMP_GE_U32 => {
                self.v_cmp_ge_u32_e32(s0, s1);
            }
            I::V_CMP_GT_I32 => {
                self.v_cmp_gt_i32_e32(s0, s1);
            }
            I::V_CMP_GT_U16 => {
                self.v_cmp_gt_u16_e32(s0, s1);
            }
            I::V_CMP_LG_F32 => {
                self.v_cmp_lg_f32_e32(s0, s1);
            }
            I::V_CMP_LG_F64 => {
                self.v_cmp_lg_f64_e32(s0, s1);
            }
            I::V_CMP_LT_U64 => {
                self.v_cmp_lt_u64_e32(s0, s1);
            }
            I::V_CMP_NEQ_F64 => {
                self.v_cmp_neq_f64_e32(s0, s1);
            }
            I::V_CMP_NGE_F64 => {
                self.v_cmp_nge_f64_e32(s0, s1);
            }
            I::V_CMPX_EQ_F32 => {
                self.v_cmpx_eq_f32_e32(s0, s1);
            }
            I::V_CMPX_EQ_F64 => {
                self.v_cmpx_eq_f64_e32(s0, s1);
            }
            I::V_CMPX_EQ_I64 => {
                self.v_cmpx_eq_i64_e32(s0, s1);
            }
            I::V_CMPX_EQ_U64 => {
                self.v_cmpx_eq_u64_e32(s0, s1);
            }
            I::V_CMPX_GE_F32 => {
                self.v_cmpx_ge_f32_e32(s0, s1);
            }
            I::V_CMPX_GE_F64 => {
                self.v_cmpx_ge_f64_e32(s0, s1);
            }
            I::V_CMPX_GE_I64 => {
                self.v_cmpx_ge_i64_e32(s0, s1);
            }
            I::V_CMPX_GE_U32 => {
                self.v_cmpx_ge_u32_e32(s0, s1);
            }
            I::V_CMPX_GE_U64 => {
                self.v_cmpx_ge_u64_e32(s0, s1);
            }
            I::V_CMPX_GT_F32 => {
                self.v_cmpx_gt_f32_e32(s0, s1);
            }
            I::V_CMPX_GT_F64 => {
                self.v_cmpx_gt_f64_e32(s0, s1);
            }
            I::V_CMPX_GT_I32 => {
                self.v_cmpx_gt_i32_e32(s0, s1);
            }
            I::V_CMPX_GT_I64 => {
                self.v_cmpx_gt_i64_e32(s0, s1);
            }
            I::V_CMPX_GT_U64 => {
                self.v_cmpx_gt_u64_e32(s0, s1);
            }
            I::V_CMPX_LE_F32 => {
                self.v_cmpx_le_f32_e32(s0, s1);
            }
            I::V_CMPX_LE_F64 => {
                self.v_cmpx_le_f64_e32(s0, s1);
            }
            I::V_CMPX_LE_I64 => {
                self.v_cmpx_le_i64_e32(s0, s1);
            }
            I::V_CMPX_LE_U32 => {
                self.v_cmpx_le_u32_e32(s0, s1);
            }
            I::V_CMPX_LE_U64 => {
                self.v_cmpx_le_u64_e32(s0, s1);
            }
            I::V_CMPX_LG_F32 => {
                self.v_cmpx_lg_f32_e32(s0, s1);
            }
            I::V_CMPX_LG_F64 => {
                self.v_cmpx_lg_f64_e32(s0, s1);
            }
            I::V_CMPX_LT_F32 => {
                self.v_cmpx_lt_f32_e32(s0, s1);
            }
            I::V_CMPX_LT_F64 => {
                self.v_cmpx_lt_f64_e32(s0, s1);
            }
            I::V_CMPX_LT_I64 => {
                self.v_cmpx_lt_i64_e32(s0, s1);
            }
            I::V_CMPX_LT_U64 => {
                self.v_cmpx_lt_u64_e32(s0, s1);
            }
            I::V_CMPX_NEQ_F32 => {
                self.v_cmpx_neq_f32_e32(s0, s1);
            }
            I::V_CMPX_NEQ_F64 => {
                self.v_cmpx_neq_f64_e32(s0, s1);
            }
            I::V_CMPX_NE_I64 => {
                self.v_cmpx_ne_i64_e32(s0, s1);
            }
            I::V_CMPX_NE_U64 => {
                self.v_cmpx_ne_u64_e32(s0, s1);
            }
            I::V_CMPX_NGE_F32 => {
                self.v_cmpx_nge_f32_e32(s0, s1);
            }
            I::V_CMPX_NGT_F32 => {
                self.v_cmpx_ngt_f32_e32(s0, s1);
            }
            I::V_CMPX_NLE_F32 => {
                self.v_cmpx_nle_f32_e32(s0, s1);
            }
            I::V_CMPX_NLE_F64 => {
                self.v_cmpx_nle_f64_e32(s0, s1);
            }
            I::V_CMPX_NLT_F32 => {
                self.v_cmpx_nlt_f32_e32(s0, s1);
            }
            I::V_CMP_EQ_F32 => {
                self.v_cmp_eq_f32_e32(s0, s1);
            }
            I::V_CMP_EQ_F64 => {
                self.v_cmp_eq_f64_e32(s0, s1);
            }
            I::V_CMP_EQ_I64 => {
                self.v_cmp_eq_i64_e32(s0, s1);
            }
            I::V_CMP_GE_F64 => {
                self.v_cmp_ge_f64_e32(s0, s1);
            }
            I::V_CMP_GE_I64 => {
                self.v_cmp_ge_i64_e32(s0, s1);
            }
            I::V_CMP_GE_U64 => {
                self.v_cmp_ge_u64_e32(s0, s1);
            }
            I::V_CMP_GT_I64 => {
                self.v_cmp_gt_i64_e32(s0, s1);
            }
            I::V_CMP_LE_I64 => {
                self.v_cmp_le_i64_e32(s0, s1);
            }
            I::V_CMP_LE_U32 => {
                self.v_cmp_le_u32_e32(s0, s1);
            }
            I::V_CMP_LE_U64 => {
                self.v_cmp_le_u64_e32(s0, s1);
            }
            I::V_CMP_LT_I32 => {
                self.v_cmp_lt_i32_e32(s0, s1);
            }
            I::V_CMP_LT_I64 => {
                self.v_cmp_lt_i64_e32(s0, s1);
            }
            I::V_CMP_NEQ_F32 => {
                self.v_cmp_neq_f32_e32(s0, s1);
            }
            I::V_CMP_NE_I64 => {
                self.v_cmp_ne_i64_e32(s0, s1);
            }
            I::V_CMP_NE_U64 => {
                self.v_cmp_ne_u64_e32(s0, s1);
            }
            I::V_CMP_NGE_F32 => {
                self.v_cmp_nge_f32_e32(s0, s1);
            }
            I::V_CMP_NLE_F32 => {
                self.v_cmp_nle_f32_e32(s0, s1);
            }
            I::V_CMP_NLE_F64 => {
                self.v_cmp_nle_f64_e32(s0, s1);
            }
            I::V_CMP_NLT_F32 => {
                self.v_cmp_nlt_f32_e32(s0, s1);
            }
            I::V_CMP_O_F32 => {
                self.v_cmp_o_f32_e32(s0, s1);
            }
            I::V_CMP_O_F64 => {
                self.v_cmp_o_f64_e32(s0, s1);
            }
            op => unimplemented!("{:?}", op),
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
            I::V_DUAL_SUBREV_F32 => {
                self.v_dual_subrev_f32(&mut dual_result0_u32, s0, s1);
            }
            I::V_DUAL_MUL_DX9_ZERO_F32 => {
                self.v_dual_mul_dx9_zero_f32(&mut dual_result0_u32, s0, s1);
            }
            I::V_DUAL_MAX_NUM_F32 => {
                self.v_dual_max_num_f32(&mut dual_result0_u32, s0, s1);
            }
            I::V_DUAL_MIN_NUM_F32 => {
                self.v_dual_min_num_f32(&mut dual_result0_u32, s0, s1);
            }
            I::V_DUAL_DOT2ACC_F32_F16 => {
                self.v_dual_dot2acc_f32_f16(&mut dual_result0_u32, s0, s1, d);
            }
            I::V_DUAL_DOT2ACC_F32_BF16 => {
                self.v_dual_dot2acc_f32_bf16(&mut dual_result0_u32, s0, s1, d);
            }
            op => unimplemented!("{:?}", op),
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
            I::V_DUAL_FMAMK_F32 => {
                self.v_dual_fmamk_f32(
                    &mut dual_result1_u32,
                    s0,
                    s1,
                    inst.literal_constant.unwrap(),
                );
            }
            I::V_DUAL_SUBREV_F32 => {
                self.v_dual_subrev_f32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_MUL_DX9_ZERO_F32 => {
                self.v_dual_mul_dx9_zero_f32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_MAX_NUM_F32 => {
                self.v_dual_max_num_f32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_MIN_NUM_F32 => {
                self.v_dual_min_num_f32(&mut dual_result1_u32, s0, s1);
            }
            I::V_DUAL_DOT2ACC_F32_F16 => {
                self.v_dual_dot2acc_f32_f16(&mut dual_result1_u32, s0, s1, d);
            }
            I::V_DUAL_DOT2ACC_F32_BF16 => {
                self.v_dual_dot2acc_f32_bf16(&mut dual_result1_u32, s0, s1, d);
            }
            op => unimplemented!("{:?}", op),
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
            | I::V_DUAL_FMAAK_F32
            | I::V_DUAL_SUBREV_F32
            | I::V_DUAL_MUL_DX9_ZERO_F32
            | I::V_DUAL_MAX_NUM_F32
            | I::V_DUAL_MIN_NUM_F32
            | I::V_DUAL_DOT2ACC_F32_F16
            | I::V_DUAL_DOT2ACC_F32_BF16 => {
                for elem in 0..32 {
                    if !self.get_exec_bit(elem) {
                        continue;
                    }
                    self.write_vgpr(elem, d, dual_result0_u32[elem]);
                }
            }
            op => unimplemented!("{:?}", op),
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
            | I::V_DUAL_FMAAK_F32
            | I::V_DUAL_FMAMK_F32
            | I::V_DUAL_SUBREV_F32
            | I::V_DUAL_MUL_DX9_ZERO_F32
            | I::V_DUAL_MAX_NUM_F32
            | I::V_DUAL_MIN_NUM_F32
            | I::V_DUAL_DOT2ACC_F32_F16
            | I::V_DUAL_DOT2ACC_F32_BF16 => {
                for elem in 0..32 {
                    if !self.get_exec_bit(elem) {
                        continue;
                    }
                    self.write_vgpr(elem, d, dual_result1_u32[elem]);
                }
            }
            op => unimplemented!("{:?}", op),
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
            let d_value = sub_f32(s0_value, s1_value);
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    /// ISA §V_DUAL_SUBREV_F32: the subtract with its sources the other way
    /// round.
    fn v_dual_subrev_f32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = sub_f32(s1_value, s0_value);
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    /// ISA §V_MUL_DX9_ZERO_F32: DX9 rules, where a zero operand gives zero
    /// whatever the other one is.
    fn v_dual_mul_dx9_zero_f32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = if s0_value == 0.0 || s1_value == 0.0 {
                0.0
            } else {
                s0_value * s1_value
            };
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    fn v_dual_max_num_f32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value.max(s1_value);
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    fn v_dual_min_num_f32(&mut self, d_values: &mut [u32], s0: SourceOperand, s1: usize) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_f32(elem, s0);
            let s1_value = u32_to_f32(self.read_vgpr(elem, s1));
            let d_value = s0_value.min(s1_value);
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    /// ISA §V_DUAL_DOT2ACC_F32_F16: the two halves of each source are widened,
    /// multiplied and added to what the destination already holds.
    fn v_dual_dot2acc_f32_f16(
        &mut self,
        d_values: &mut [u32],
        s0: SourceOperand,
        s1: usize,
        d: usize,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s1_value = self.read_vgpr(elem, s1);
            let mut d_value = u32_to_f32(self.read_vgpr(elem, d));
            d_value += self.read_vector_source_operand_f16(elem, s0).to_f32()
                * f16::from_bits(s1_value as u16).to_f32();
            d_value += self.read_vector_source_operand_f16_hi(elem, s0).to_f32()
                * f16::from_bits((s1_value >> 16) as u16).to_f32();
            d_values[elem] = f32_to_u32(d_value);
        }
    }

    /// ISA §V_DUAL_DOT2ACC_F32_BF16: the same over brain floats.
    fn v_dual_dot2acc_f32_bf16(
        &mut self,
        d_values: &mut [u32],
        s0: SourceOperand,
        s1: usize,
        d: usize,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let s0_value = self.read_vector_source_operand_u32(elem, s0);
            let s1_value = self.read_vgpr(elem, s1);
            let mut d_value = u32_to_f32(self.read_vgpr(elem, d));
            d_value += bf16_to_f32(s0_value as u16) * bf16_to_f32(s1_value as u16);
            d_value += bf16_to_f32((s0_value >> 16) as u16) * bf16_to_f32((s1_value >> 16) as u16);
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
            op => unimplemented!("{:?}", op),
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

    fn flat_load_u8(&mut self, vaddr: usize, vdst: usize, ioffset: u32) {
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
            let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                offset[elem] as i64 + (ioffset as i64)
            } else {
                let lane_addr = offset[elem] as i64 + (ioffset as i64) - scratch_base as i64;
                scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
            };
            let ptr = addr as *mut u8;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, data as u32);
        }
    }

    fn flat_load_i8(&mut self, vaddr: usize, vdst: usize, ioffset: u32) {
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
            let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                offset[elem] as i64 + (ioffset as i64)
            } else {
                let lane_addr = offset[elem] as i64 + (ioffset as i64) - scratch_base as i64;
                scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
            };
            let ptr = addr as *mut i8;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, (data as i32) as u32);
        }
    }

    fn flat_load_u16(&mut self, vaddr: usize, vdst: usize, ioffset: u32) {
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
            let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                offset[elem] as i64 + (ioffset as i64)
            } else {
                let lane_addr = offset[elem] as i64 + (ioffset as i64) - scratch_base as i64;
                scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
            };
            let ptr = addr as *mut u16;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, data as u32);
        }
    }

    fn flat_load_i16(&mut self, vaddr: usize, vdst: usize, ioffset: u32) {
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
            let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                offset[elem] as i64 + (ioffset as i64)
            } else {
                let lane_addr = offset[elem] as i64 + (ioffset as i64) - scratch_base as i64;
                scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
            };
            let ptr = addr as *mut i16;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, (data as i32) as u32);
        }
    }

    fn flat_load_b96(&mut self, vaddr: usize, vdst: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| self.read_vgpr_pair(elem, vaddr))
            .collect::<Vec<u64>>();

        for i in 0..3 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let ioffset = ((ioffset << 8) as i32) >> 8;
                let scratch_base = self.ctx.scratch.borrow().as_ptr() as u64;
                let scratch_limit = scratch_base + self.ctx.scratch.borrow().len() as u64 / 32;
                let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                    offset[elem] as i64 + (ioffset as i64) + i as i64 * 4
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

    fn flat_store_b8(&mut self, vaddr: usize, vsrc: usize, ioffset: u32) {
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
            let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                offset[elem] as i64 + (ioffset as i64)
            } else {
                let lane_addr = offset[elem] as i64 + (ioffset as i64) - scratch_base as i64;
                scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
            };
            let data = self.read_vgpr(elem, vsrc);
            let ptr = addr as *mut u8;
            unsafe {
                *ptr = data as u8;
            }
        }
    }

    fn flat_store_b16(&mut self, vaddr: usize, vsrc: usize, ioffset: u32) {
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
            let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                offset[elem] as i64 + (ioffset as i64)
            } else {
                let lane_addr = offset[elem] as i64 + (ioffset as i64) - scratch_base as i64;
                scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
            };
            let data = self.read_vgpr(elem, vsrc);
            let ptr = addr as *mut u16;
            unsafe {
                *ptr = data as u16;
            }
        }
    }

    fn flat_store_b64(&mut self, vaddr: usize, vsrc: usize, ioffset: u32) {
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
                let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                    offset[elem] as i64 + (ioffset as i64) + i as i64 * 4
                } else {
                    let lane_addr =
                        offset[elem] as i64 + (ioffset as i64) + i as i64 * 4 - scratch_base as i64;
                    scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
                };
                let data = self.read_vgpr(elem, vsrc + i);
                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn flat_store_b96(&mut self, vaddr: usize, vsrc: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| self.read_vgpr_pair(elem, vaddr))
            .collect::<Vec<u64>>();

        for i in 0..3 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let ioffset = ((ioffset << 8) as i32) >> 8;
                let scratch_base = self.ctx.scratch.borrow().as_ptr() as u64;
                let scratch_limit = scratch_base + self.ctx.scratch.borrow().len() as u64 / 32;
                let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                    offset[elem] as i64 + (ioffset as i64) + i as i64 * 4
                } else {
                    let lane_addr =
                        offset[elem] as i64 + (ioffset as i64) + i as i64 * 4 - scratch_base as i64;
                    scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
                };
                let data = self.read_vgpr(elem, vsrc + i);
                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
    }

    fn flat_store_b128(&mut self, vaddr: usize, vsrc: usize, ioffset: u32) {
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
                let addr = if (offset[elem] < scratch_base) || (offset[elem] >= scratch_limit) {
                    offset[elem] as i64 + (ioffset as i64) + i as i64 * 4
                } else {
                    let lane_addr =
                        offset[elem] as i64 + (ioffset as i64) + i as i64 * 4 - scratch_base as i64;
                    scratch_base as i64 + lane_addr * 32 + elem as i64 * 4
                };
                let data = self.read_vgpr(elem, vsrc + i);
                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
            }
        }
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
            I::FLAT_LOAD_U8 => {
                self.flat_load_u8(vaddr, vdst, ioffset);
            }
            I::FLAT_LOAD_I8 => {
                self.flat_load_i8(vaddr, vdst, ioffset);
            }
            I::FLAT_LOAD_U16 => {
                self.flat_load_u16(vaddr, vdst, ioffset);
            }
            I::FLAT_LOAD_I16 => {
                self.flat_load_i16(vaddr, vdst, ioffset);
            }
            I::FLAT_LOAD_B96 => {
                self.flat_load_b96(vaddr, vdst, ioffset);
            }
            I::FLAT_STORE_B8 => {
                self.flat_store_b8(vaddr, vsrc, ioffset);
            }
            I::FLAT_STORE_B16 => {
                self.flat_store_b16(vaddr, vsrc, ioffset);
            }
            I::FLAT_STORE_B64 => {
                self.flat_store_b64(vaddr, vsrc, ioffset);
            }
            I::FLAT_STORE_B96 => {
                self.flat_store_b96(vaddr, vsrc, ioffset);
            }
            I::FLAT_STORE_B128 => {
                self.flat_store_b128(vaddr, vsrc, ioffset);
            }
            op => unimplemented!("{:?}", op),
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
            let ioffset = ((ioffset << 8) as i32) >> 8;
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
        // SVE says whether the VGPR takes part in the address; SADDR being NULL
        // says the same of the SGPR.
        let use_vaddr = inst.sve != 0;
        let use_saddr = inst.saddr != 0x7C;
        let ioffset = ((inst.ioffset << 8) as i32 >> 8) as i64;
        match inst.op {
            I::SCRATCH_LOAD_U8 => {
                self.scratch_load_u8(vaddr, vdst, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_LOAD_I8 => {
                self.scratch_load_i8(vaddr, vdst, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_LOAD_U16 => {
                self.scratch_load_u16(vaddr, vdst, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_LOAD_I16 => {
                self.scratch_load_i16(vaddr, vdst, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_LOAD_B32 => {
                self.scratch_load_b32(vaddr, vdst, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_LOAD_B64 => {
                self.scratch_load_b64(vaddr, vdst, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_LOAD_B96 => {
                self.scratch_load_b96(vaddr, vdst, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_LOAD_B128 => {
                self.scratch_load_b128(vaddr, vdst, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_STORE_B8 => {
                self.scratch_store_b8(vaddr, vsrc, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_STORE_B16 => {
                self.scratch_store_b16(vaddr, vsrc, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_STORE_B32 => {
                self.scratch_store_b32(vaddr, vsrc, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_STORE_B64 => {
                self.scratch_store_b64(vaddr, vsrc, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_STORE_B96 => {
                self.scratch_store_b96(vaddr, vsrc, saddr, ioffset, use_vaddr, use_saddr);
            }
            I::SCRATCH_STORE_B128 => {
                self.scratch_store_b128(vaddr, vsrc, saddr, ioffset, use_vaddr, use_saddr);
            }
            op => unimplemented!("{:?}", op),
        }
        Signals::None
    }

    /// Where byte `offset` of a lane's own view of the private segment lives.
    /// The segment is swizzled across the lanes a dword at a time, so the dword
    /// at `offset` sits a lane-stride apart from its neighbours and an
    /// unaligned access runs on into the next dword's slot.
    fn scratch_address(&self, elem: usize, offset: i64) -> u64 {
        let base = self.ctx.scratch.borrow().as_ptr() as u64;
        let dword = offset >> 2;
        let byte = (offset & 3) as u64;
        (base as i64 + dword * 128) as u64 + (elem as u64) * 4 + byte
    }

    /// The address the lane's access starts at. VADDR and SADDR each take part
    /// only when the instruction says so.
    fn scratch_offset(
        &self,
        elem: usize,
        vaddr: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) -> i64 {
        let vaddr_value = if use_vaddr {
            self.read_vgpr(elem, vaddr) as i64
        } else {
            0
        };
        // The scratch SGPR offset is a signed 32-bit byte offset.
        let saddr_value = if use_saddr {
            self.read_sgpr(saddr) as i32 as i64
        } else {
            0
        };
        vaddr_value + saddr_value + ioffset
    }

    fn scratch_load_u8(
        &mut self,
        vaddr: usize,
        vdst: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            let ptr = self.scratch_address(elem, start) as *const u8;
            let data = unsafe { *ptr } as u32;
            self.write_vgpr(elem, vdst, data);
        }
    }

    fn scratch_load_i8(
        &mut self,
        vaddr: usize,
        vdst: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            let ptr = self.scratch_address(elem, start) as *const u8;
            let data = unsafe { *ptr } as i8 as i32 as u32;
            self.write_vgpr(elem, vdst, data);
        }
    }

    /// A sub-dword access can straddle two swizzled dwords, so it is put
    /// together a byte at a time.
    fn scratch_load_u16(
        &mut self,
        vaddr: usize,
        vdst: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            let mut data = 0u32;
            for byte in 0..2 {
                let ptr = self.scratch_address(elem, start + byte) as *const u8;
                data |= (unsafe { *ptr } as u32) << (byte * 8);
            }
            let data = data;
            self.write_vgpr(elem, vdst, data);
        }
    }

    fn scratch_load_i16(
        &mut self,
        vaddr: usize,
        vdst: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            let mut data = 0u32;
            for byte in 0..2 {
                let ptr = self.scratch_address(elem, start + byte) as *const u8;
                data |= (unsafe { *ptr } as u32) << (byte * 8);
            }
            let data = data as u16 as i16 as i32 as u32;
            self.write_vgpr(elem, vdst, data);
        }
    }

    /// Each dword is assembled a byte at a time for the same reason.
    fn scratch_load_b32(
        &mut self,
        vaddr: usize,
        vdst: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for word in 0..1 {
                let mut data = 0u32;
                for byte in 0..4 {
                    let ptr =
                        self.scratch_address(elem, start + (word * 4) as i64 + byte) as *const u8;
                    data |= (unsafe { *ptr } as u32) << (byte * 8);
                }
                self.write_vgpr(elem, vdst + word, data);
            }
        }
    }

    fn scratch_load_b64(
        &mut self,
        vaddr: usize,
        vdst: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for word in 0..2 {
                let mut data = 0u32;
                for byte in 0..4 {
                    let ptr =
                        self.scratch_address(elem, start + (word * 4) as i64 + byte) as *const u8;
                    data |= (unsafe { *ptr } as u32) << (byte * 8);
                }
                self.write_vgpr(elem, vdst + word, data);
            }
        }
    }

    fn scratch_load_b96(
        &mut self,
        vaddr: usize,
        vdst: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for word in 0..3 {
                let mut data = 0u32;
                for byte in 0..4 {
                    let ptr =
                        self.scratch_address(elem, start + (word * 4) as i64 + byte) as *const u8;
                    data |= (unsafe { *ptr } as u32) << (byte * 8);
                }
                self.write_vgpr(elem, vdst + word, data);
            }
        }
    }

    fn scratch_load_b128(
        &mut self,
        vaddr: usize,
        vdst: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for word in 0..4 {
                let mut data = 0u32;
                for byte in 0..4 {
                    let ptr =
                        self.scratch_address(elem, start + (word * 4) as i64 + byte) as *const u8;
                    data |= (unsafe { *ptr } as u32) << (byte * 8);
                }
                self.write_vgpr(elem, vdst + word, data);
            }
        }
    }

    /// The bytes go out one at a time, since the swizzle can put them in two
    /// different dwords.
    fn scratch_store_b8(
        &mut self,
        vaddr: usize,
        vsrc: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for byte in 0..1 {
                let word = byte / 4;
                let data = self.read_vgpr(elem, vsrc + word);
                let ptr = self.scratch_address(elem, start + byte as i64) as *mut u8;
                unsafe {
                    *ptr = (data >> ((byte % 4) * 8)) as u8;
                }
            }
        }
    }

    fn scratch_store_b16(
        &mut self,
        vaddr: usize,
        vsrc: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for byte in 0..2 {
                let word = byte / 4;
                let data = self.read_vgpr(elem, vsrc + word);
                let ptr = self.scratch_address(elem, start + byte as i64) as *mut u8;
                unsafe {
                    *ptr = (data >> ((byte % 4) * 8)) as u8;
                }
            }
        }
    }

    fn scratch_store_b32(
        &mut self,
        vaddr: usize,
        vsrc: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for byte in 0..4 {
                let word = byte / 4;
                let data = self.read_vgpr(elem, vsrc + word);
                let ptr = self.scratch_address(elem, start + byte as i64) as *mut u8;
                unsafe {
                    *ptr = (data >> ((byte % 4) * 8)) as u8;
                }
            }
        }
    }

    fn scratch_store_b64(
        &mut self,
        vaddr: usize,
        vsrc: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for byte in 0..8 {
                let word = byte / 4;
                let data = self.read_vgpr(elem, vsrc + word);
                let ptr = self.scratch_address(elem, start + byte as i64) as *mut u8;
                unsafe {
                    *ptr = (data >> ((byte % 4) * 8)) as u8;
                }
            }
        }
    }

    fn scratch_store_b96(
        &mut self,
        vaddr: usize,
        vsrc: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for byte in 0..12 {
                let word = byte / 4;
                let data = self.read_vgpr(elem, vsrc + word);
                let ptr = self.scratch_address(elem, start + byte as i64) as *mut u8;
                unsafe {
                    *ptr = (data >> ((byte % 4) * 8)) as u8;
                }
            }
        }
    }

    fn scratch_store_b128(
        &mut self,
        vaddr: usize,
        vsrc: usize,
        saddr: usize,
        ioffset: i64,
        use_vaddr: bool,
        use_saddr: bool,
    ) {
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let start = self.scratch_offset(elem, vaddr, saddr, ioffset, use_vaddr, use_saddr);
            for byte in 0..16 {
                let word = byte / 4;
                let data = self.read_vgpr(elem, vsrc + word);
                let ptr = self.scratch_address(elem, start + byte as i64) as *mut u8;
                unsafe {
                    *ptr = (data >> ((byte % 4) * 8)) as u8;
                }
            }
        }
    }

    fn global_load_i8(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
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
            let addr = offset[elem].wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64);

            let ptr = addr as *mut i8;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, (data as i32) as u32);
        }
    }

    fn global_load_i16(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
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
            let addr = offset[elem].wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64);

            let ptr = addr as *mut i16;
            let data = unsafe { *ptr };
            self.write_vgpr(elem, vdst, (data as i32) as u32);
        }
    }

    fn global_load_b96(&mut self, vaddr: usize, vdst: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..3 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let addr = offset[elem]
                    .wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64)
                    .wrapping_add(i as u64 * 4);

                let ptr = addr as *mut u32;
                let data = unsafe { *ptr };
                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn global_store_b8(&mut self, vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
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
            let addr = offset[elem].wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64);

            let ptr = addr as *mut u8;
            unsafe {
                *ptr = data as u8;
            }
        }
    }

    fn global_store_b96(&mut self, vaddr: usize, vsrc: usize, saddr: usize, ioffset: u32) {
        let offset = (0..32)
            .map(|elem| {
                if saddr != 124 {
                    self.read_sgpr_pair(saddr) + self.read_vgpr(elem, vaddr) as u64
                } else {
                    self.read_vgpr_pair(elem, vaddr)
                }
            })
            .collect::<Vec<u64>>();

        for i in 0..3 {
            for elem in 0..32 {
                if !self.get_exec_bit(elem) {
                    continue;
                }
                let data = self.read_vgpr(elem, vsrc + i);
                let addr = offset[elem]
                    .wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64)
                    .wrapping_add(i as u64 * 4);

                let ptr = addr as *mut u32;
                unsafe {
                    *ptr = data;
                }
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
            I::GLOBAL_ATOMIC_ADD_U32 => {
                self.global_atomic_add_u32(vaddr, vdst, vsrc, saddr, ioffset);
            }
            I::GLOBAL_LOAD_I8 => {
                self.global_load_i8(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_LOAD_I16 => {
                self.global_load_i16(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_LOAD_B96 => {
                self.global_load_b96(vaddr, vdst, saddr, ioffset);
            }
            I::GLOBAL_STORE_B8 => {
                self.global_store_b8(vaddr, vsrc, saddr, ioffset);
            }
            I::GLOBAL_STORE_B96 => {
                self.global_store_b96(vaddr, vsrc, saddr, ioffset);
            }
            I::GLOBAL_WB => {}
            I::GLOBAL_INV => {}
            op => unimplemented!("{:?}", op),
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
            let addr = offset[elem].wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64);

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
            let addr = offset[elem].wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64);

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
                let addr = offset[elem]
                    .wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64)
                    .wrapping_add(i as u64 * 4);

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
                let addr = offset[elem]
                    .wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64)
                    .wrapping_add(i as u64 * 4);

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
            let addr = offset[elem].wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64);

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
            let addr = offset[elem].wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64);

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
            let addr = offset[elem].wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64);

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
                let addr = offset[elem]
                    .wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64)
                    .wrapping_add(i as u64 * 4);

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

    fn global_atomic_add_u32(
        &mut self,
        vaddr: usize,
        vdst: usize,
        vsrc: usize,
        saddr: usize,
        ioffset: u32,
    ) {
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
            let addr = offset[elem].wrapping_add(((ioffset << 8) as i32 >> 8) as i64 as u64);
            let data = self.read_vgpr(elem, vsrc);

            let ptr = addr as *mut u32;
            let data = unsafe {
                use std::sync::atomic::{AtomicU32, Ordering};
                AtomicU32::from_ptr(ptr).fetch_add(data, Ordering::SeqCst)
            };
            self.write_vgpr(elem, vdst, data);
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
            op => unimplemented!("{:?}", op),
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
        // The resource holds the base of the BVH in 256-byte units, and the
        // node pointer counts from there in eight-byte ones.
        let base_addr = ((((s1_value as u64) & 0xFF) << 32) | (s0_value as u64)) << 8;
        let sort_triangles_first = (s1_value >> 20) & 1 != 0;
        let box_sort = (s1_value >> 31) & 1 != 0;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let node_ptr = self.read_vgpr_pair(elem, vaddr0);
            let node_type = (node_ptr & 0x7) as u8;
            match node_type {
                5 => {
                    let node_ptr = base_addr + ((node_ptr & !0x7u64) << 3);
                    let node = unsafe { *(node_ptr as *const Box4Node) };
                    let ray_extent = u32_to_f32(self.read_vgpr(elem, vaddr1));
                    let ray_origin_x = u32_to_f32(self.read_vgpr(elem, vaddr2));
                    let ray_origin_y = u32_to_f32(self.read_vgpr(elem, vaddr2 + 1));
                    let ray_origin_z = u32_to_f32(self.read_vgpr(elem, vaddr2 + 2));
                    let ray_inv_dir_x = u32_to_f32(self.read_vgpr(elem, vaddr4));
                    let ray_inv_dir_y = u32_to_f32(self.read_vgpr(elem, vaddr4 + 1));
                    let ray_inv_dir_z = u32_to_f32(self.read_vgpr(elem, vaddr4 + 2));

                    let mut children = [(0u32, 0u32, 0.0f32); 4];
                    for (i, child) in children.iter_mut().enumerate() {
                        let (t0, t1) = intersect(
                            [ray_origin_x, ray_origin_y, ray_origin_z],
                            [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                            &node.aabb[i],
                            ray_extent,
                        );
                        let index = node.child_index[i];
                        let rank = if sort_triangles_first {
                            box4_child_rank(index & 7)
                        } else {
                            0
                        };
                        *child = (if t0 <= t1 { index } else { 0xFFFF_FFFF }, rank, t0);
                    }

                    // The children are sorted only if the resource asks for
                    // it; otherwise they come back in the order the node holds
                    // them. Ranking them is work the common case does not
                    // need, so the two orders have a pass each.
                    if box_sort && sort_triangles_first {
                        for (a, b) in BOX4_NETWORK {
                            if sorts_before(children[b], children[a]) {
                                children.swap(a, b);
                            }
                        }
                    } else if box_sort {
                        for (a, b) in BOX4_NETWORK {
                            if closer_than(children[b], children[a]) {
                                children.swap(a, b);
                            }
                        }
                    }

                    for (i, child) in children.iter().enumerate() {
                        self.write_vgpr(elem, vdata + i, child.0);
                    }
                }
                0 | 1 => {
                    let node_ptr = base_addr + ((node_ptr & !(0x7u64)) << 3);
                    let node = unsafe { *(node_ptr as *const TrianglePairNode) };
                    let tri = if node_type & 1 == 0 {
                        [node.tri_pair.v0, node.tri_pair.v1, node.tri_pair.v2]
                    } else {
                        // The pair's second triangle, wound the way the part
                        // takes it: the same three vertices as (v3, v2, v1),
                        // rotated so that the barycentrics come back in the
                        // order the flags name.
                        [node.tri_pair.v1, node.tri_pair.v3, node.tri_pair.v2]
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
        // The resource holds the base of the BVH in 256-byte units, and the
        // node base and offset count from there in eight-byte ones.
        let base_addr = ((((s1_value as u64) & 0xFF) << 32) | (s0_value as u64)) << 8;
        let sort_triangles_first = (s1_value >> 20) & 1 != 0;
        let box_sort = (s1_value >> 31) & 1 != 0;
        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let node_base = self.read_vgpr_pair(elem, vaddr0);
            let node_index = self.read_vgpr(elem, vaddr4);
            let node_ptr = base_addr + ((node_base + (node_index & !0xF) as u64) << 3);
            let node_type = (node_index & 0xF) as u8;
            match node_type {
                0..=3 | 8..=11 => {
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

                    // A triangle is named by its primitive index doubled, with
                    // the low bit saying which way round the ray met it.
                    let prim0 = (node.get_prim_index(tri_pair_index as u32, 0) << 1)
                        | (result0.3 < 0.0) as u32;
                    let prim1 = (node.get_prim_index(tri_pair_index as u32, 1) << 1)
                        | (result1.3 < 0.0) as u32;

                    let node_end = (tri_pair_index as u32 + 1) == node.get_triangle_pair_count();
                    let range_end = node.is_range_end(tri_pair_index as u32);
                    let ends = ((range_end as u32) << 1) | (node_end as u32);

                    self.write_vgpr(elem, vdata, f32_to_u32(result0.0));
                    self.write_vgpr(elem, vdata + 1, f32_to_u32(result0.1));
                    self.write_vgpr(elem, vdata + 2, f32_to_u32(result0.2));
                    self.write_vgpr(elem, vdata + 3, prim0);
                    self.write_vgpr(elem, vdata + 4, f32_to_u32(result1.0));
                    self.write_vgpr(elem, vdata + 5, f32_to_u32(result1.1));
                    self.write_vgpr(elem, vdata + 6, f32_to_u32(result1.2));
                    self.write_vgpr(elem, vdata + 7, prim1);
                    // The last two dwords stand for the pair's two triangles,
                    // and both say where the pair ends.
                    self.write_vgpr(elem, vdata + 8, ends);
                    self.write_vgpr(elem, vdata + 9, ends);
                }
                5 => {
                    let node = unsafe { *(node_ptr as *const Box8Node) };
                    let ray_extent = u32_to_f32(self.read_vgpr(elem, vaddr1));
                    let instance_mask = self.read_vgpr(elem, vaddr1 + 1);
                    let ray_origin_x = u32_to_f32(self.read_vgpr(elem, vaddr2));
                    let ray_origin_y = u32_to_f32(self.read_vgpr(elem, vaddr2 + 1));
                    let ray_origin_z = u32_to_f32(self.read_vgpr(elem, vaddr2 + 2));
                    let ray_dir_x = u32_to_f32(self.read_vgpr(elem, vaddr3));
                    let ray_dir_y = u32_to_f32(self.read_vgpr(elem, vaddr3 + 1));
                    let ray_dir_z = u32_to_f32(self.read_vgpr(elem, vaddr3 + 2));
                    let ray_inv_dir_x = 1.0 / ray_dir_x;
                    let ray_inv_dir_y = 1.0 / ray_dir_y;
                    let ray_inv_dir_z = 1.0 / ray_dir_z;

                    let child_count = node.get_child_count() as usize;

                    // A child the node does not have is a miss, and so is one
                    // whose instance mask has nothing in common with the ray's.
                    let results = (0..8)
                        .map(|i| {
                            if i >= child_count {
                                return (0xFFFF_FFFF, 0, f32::INFINITY);
                            }
                            let (t0, t1) = intersect(
                                [ray_origin_x, ray_origin_y, ray_origin_z],
                                [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                                &node.get_child_box(i),
                                ray_extent,
                            );
                            let hit = t0 <= t1 && (instance_mask & node.get_child_mask(i)) != 0;
                            let index = if hit {
                                node.get_child_index(i)
                            } else {
                                0xFFFF_FFFF
                            };
                            let rank = if sort_triangles_first {
                                box8_child_rank(node.get_child_type(i) as u32)
                            } else {
                                0
                            };
                            (index, rank, t0)
                        })
                        .collect::<Vec<(u32, u32, f32)>>();

                    // The children come back sorted only if the resource asks for
                    // it; otherwise in the order the node holds them.
                    let results = if box_sort {
                        results
                            .into_iter()
                            .sorted_by(|&a, &b| {
                                if sorts_before(b, a) {
                                    std::cmp::Ordering::Greater
                                } else {
                                    std::cmp::Ordering::Less
                                }
                            })
                            .map(|(index, _, _)| index)
                            .collect::<Vec<u32>>()
                    } else {
                        results
                            .into_iter()
                            .map(|(index, _, _)| index)
                            .collect::<Vec<u32>>()
                    };

                    for i in 0..8 {
                        self.write_vgpr(elem, vdata + i as usize, results[i as usize]);
                    }
                    // A box node has nothing to say in the two dwords a
                    // triangle node names its triangles with.
                    self.write_vgpr(elem, vdata + 8, 0xFFFF_FFFF);
                    self.write_vgpr(elem, vdata + 9, 0xFFFF_FFFF);
                }
                _ => {
                    panic!("Unsupported node type: {}", node_type);
                }
            }
        }
    }

    fn execute_vsample(&mut self, inst: VSAMPLE) -> Signals {
        let vdata = inst.vdata as usize;
        let vaddr0 = inst.vaddr0 as usize;
        let vaddr1 = inst.vaddr1 as usize;
        let rsrc = inst.rsrc as usize;
        let samp = inst.samp as usize;
        match inst.op {
            I::IMAGE_SAMPLE_LZ => {
                self.image_sample_lz(
                    vdata,
                    vaddr0,
                    vaddr1,
                    rsrc,
                    samp,
                    inst.dmask as u32,
                    inst.unrm as u32,
                );
            }
            op => unimplemented!("{:?}", op),
        }
        Signals::None
    }

    fn image_sample_lz(
        &mut self,
        vdata: usize,
        vaddr0: usize,
        vaddr1: usize,
        rsrc: usize,
        samp: usize,
        dmask: u32,
        unrm: u32,
    ) {
        let samp_value = (0..4)
            .map(|i| self.read_sgpr(samp + i))
            .collect::<Vec<u32>>();
        let rsrc_value = (0..8)
            .map(|i| self.read_sgpr(rsrc + i))
            .collect::<Vec<u32>>();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }

            let u = u32_to_f32(self.read_vgpr(elem, vaddr0));
            let v = u32_to_f32(self.read_vgpr(elem, vaddr1));

            // The components the DMASK asks for go to consecutive registers.
            // The fetch itself is the one the JIT calls, so the two engines
            // read an image the same way by construction.
            let mut vgpr = vdata;
            for component in 0..4 {
                if dmask & (1 << component) == 0 {
                    continue;
                }
                let data = image_sample_lz(
                    rsrc_value[0],
                    rsrc_value[1],
                    rsrc_value[2],
                    rsrc_value[3],
                    rsrc_value[4],
                    rsrc_value[5],
                    rsrc_value[6],
                    rsrc_value[7],
                    samp_value[0],
                    samp_value[1],
                    samp_value[2],
                    samp_value[3],
                    component,
                    unrm,
                    u,
                    v,
                );
                self.write_vgpr(elem, vgpr, data);
                vgpr += 1;
            }
        }
    }

    fn ds_load_u8_16(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let ptr = lds.wrapping_add(addr[elem] + offset) as *const u8;
            let data = unsafe { ptr.read_unaligned() };

            self.write_vgpr(elem, vdst, data as u32);
        }
    }

    fn ds_load_i8(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let ptr = lds.wrapping_add(addr[elem] + offset) as *const i8;
            let data = unsafe { ptr.read_unaligned() };

            self.write_vgpr(elem, vdst, (data as i32) as u32);
        }
    }

    fn ds_load_u16(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let ptr = lds.wrapping_add(addr[elem] + offset) as *const u16;
            let data = unsafe { ptr.read_unaligned() };

            self.write_vgpr(elem, vdst, data as u32);
        }
    }

    fn ds_load_i16(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let ptr = lds.wrapping_add(addr[elem] + offset) as *const i16;
            let data = unsafe { ptr.read_unaligned() };

            self.write_vgpr(elem, vdst, (data as i32) as u32);
        }
    }

    fn ds_load_b32(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for i in 0..1 {
                let ptr = lds.wrapping_add(addr[elem] + offset + i * 4) as *const u32;
                let data = unsafe { ptr.read_unaligned() };

                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn ds_load_b64(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for i in 0..2 {
                let ptr = lds.wrapping_add(addr[elem] + offset + i * 4) as *const u32;
                let data = unsafe { ptr.read_unaligned() };

                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn ds_load_b96(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for i in 0..3 {
                let ptr = lds.wrapping_add(addr[elem] + offset + i * 4) as *const u32;
                let data = unsafe { ptr.read_unaligned() };

                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn ds_load_b128(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for i in 0..4 {
                let ptr = lds.wrapping_add(addr[elem] + offset + i * 4) as *const u32;
                let data = unsafe { ptr.read_unaligned() };

                self.write_vgpr(elem, vdst + i, data);
            }
        }
    }

    fn ds_store_b8_16(&mut self, addr: usize, data0: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow_mut().as_mut_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let data = self.read_vgpr(elem, data0) as u8;
            let ptr = lds.wrapping_add(addr[elem] + offset) as *mut u8;
            unsafe {
                ptr.write_unaligned(data);
            }
        }
    }

    fn ds_store_b16(&mut self, addr: usize, data0: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow_mut().as_mut_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            let data = self.read_vgpr(elem, data0) as u16;
            let ptr = lds.wrapping_add(addr[elem] + offset) as *mut u16;
            unsafe {
                ptr.write_unaligned(data);
            }
        }
    }

    fn ds_store_b32(&mut self, addr: usize, data0: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow_mut().as_mut_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for i in 0..1 {
                let data = self.read_vgpr(elem, data0 + i);
                let ptr = lds.wrapping_add(addr[elem] + offset + i * 4) as *mut u32;
                unsafe {
                    ptr.write_unaligned(data);
                }
            }
        }
    }

    fn ds_store_b64(&mut self, addr: usize, data0: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow_mut().as_mut_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for i in 0..2 {
                let data = self.read_vgpr(elem, data0 + i);
                let ptr = lds.wrapping_add(addr[elem] + offset + i * 4) as *mut u32;
                unsafe {
                    ptr.write_unaligned(data);
                }
            }
        }
    }

    fn ds_store_b96(&mut self, addr: usize, data0: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow_mut().as_mut_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for i in 0..3 {
                let data = self.read_vgpr(elem, data0 + i);
                let ptr = lds.wrapping_add(addr[elem] + offset + i * 4) as *mut u32;
                unsafe {
                    ptr.write_unaligned(data);
                }
            }
        }
    }

    fn ds_store_b128(&mut self, addr: usize, data0: usize, offset0: u8, offset1: u8) {
        // A single-address DS instruction takes the whole 16-bit offset.
        let offset = ((offset1 as usize) << 8) | (offset0 as usize);
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow_mut().as_mut_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for i in 0..4 {
                let data = self.read_vgpr(elem, data0 + i);
                let ptr = lds.wrapping_add(addr[elem] + offset + i * 4) as *mut u32;
                unsafe {
                    ptr.write_unaligned(data);
                }
            }
        }
    }

    fn ds_load_2addr_b32(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A two-address DS instruction indexes the two offsets by the size it
        // moves rather than adding them as bytes.
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for (slot, offset) in [offset0, offset1].iter().enumerate() {
                for i in 0..1 {
                    let byte = addr[elem] + (*offset as usize) * 4 + i * 4;
                    let ptr = lds.wrapping_add(byte) as *const u32;
                    let data = unsafe { ptr.read_unaligned() };

                    self.write_vgpr(elem, vdst + slot * 1 + i, data);
                }
            }
        }
    }

    fn ds_load_2addr_b64(&mut self, addr: usize, vdst: usize, offset0: u8, offset1: u8) {
        // A two-address DS instruction indexes the two offsets by the size it
        // moves rather than adding them as bytes.
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow().as_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for (slot, offset) in [offset0, offset1].iter().enumerate() {
                for i in 0..2 {
                    let byte = addr[elem] + (*offset as usize) * 8 + i * 4;
                    let ptr = lds.wrapping_add(byte) as *const u32;
                    let data = unsafe { ptr.read_unaligned() };

                    self.write_vgpr(elem, vdst + slot * 2 + i, data);
                }
            }
        }
    }

    fn ds_store_2addr_b32(
        &mut self,
        addr: usize,
        data0: usize,
        data1: usize,
        offset0: u8,
        offset1: u8,
    ) {
        // A two-address DS instruction indexes the two offsets by the size it
        // moves rather than adding them as bytes.
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow_mut().as_mut_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for (data, offset) in [(data0, offset0), (data1, offset1)] {
                for i in 0..1 {
                    let value = self.read_vgpr(elem, data + i);
                    let byte = addr[elem] + (offset as usize) * 4 + i * 4;
                    let ptr = lds.wrapping_add(byte) as *mut u32;
                    unsafe {
                        ptr.write_unaligned(value);
                    }
                }
            }
        }
    }

    fn ds_store_2addr_b64(
        &mut self,
        addr: usize,
        data0: usize,
        data1: usize,
        offset0: u8,
        offset1: u8,
    ) {
        // A two-address DS instruction indexes the two offsets by the size it
        // moves rather than adding them as bytes.
        let addr = (0..32)
            .map(|elem| self.read_vgpr(elem, addr) as usize)
            .collect::<Vec<usize>>();

        let lds = self.lds.borrow_mut().as_mut_ptr();

        for elem in 0..32 {
            if !self.get_exec_bit(elem) {
                continue;
            }
            for (data, offset) in [(data0, offset0), (data1, offset1)] {
                for i in 0..2 {
                    let value = self.read_vgpr(elem, data + i);
                    let byte = addr[elem] + (offset as usize) * 8 + i * 4;
                    let ptr = lds.wrapping_add(byte) as *mut u32;
                    unsafe {
                        ptr.write_unaligned(value);
                    }
                }
            }
        }
    }

    fn execute_ds(&mut self, inst: DS) -> Signals {
        let addr = inst.addr as usize;
        let data0 = inst.data0 as usize;
        let data1 = inst.data1 as usize;
        let vdst = inst.vdst as usize;
        let offset0 = inst.offset0;
        let offset1 = inst.offset1;
        match inst.op {
            I::DS_LOAD_U8 => {
                self.ds_load_u8_16(addr, vdst, offset0, offset1);
            }
            I::DS_LOAD_I8 => {
                self.ds_load_i8(addr, vdst, offset0, offset1);
            }
            I::DS_LOAD_U16 => {
                self.ds_load_u16(addr, vdst, offset0, offset1);
            }
            I::DS_LOAD_I16 => {
                self.ds_load_i16(addr, vdst, offset0, offset1);
            }
            I::DS_LOAD_B32 => {
                self.ds_load_b32(addr, vdst, offset0, offset1);
            }
            I::DS_LOAD_B64 => {
                self.ds_load_b64(addr, vdst, offset0, offset1);
            }
            I::DS_LOAD_B96 => {
                self.ds_load_b96(addr, vdst, offset0, offset1);
            }
            I::DS_LOAD_B128 => {
                self.ds_load_b128(addr, vdst, offset0, offset1);
            }
            I::DS_STORE_B8 => {
                self.ds_store_b8_16(addr, data0, offset0, offset1);
            }
            I::DS_STORE_B16 => {
                self.ds_store_b16(addr, data0, offset0, offset1);
            }
            I::DS_STORE_B32 => {
                self.ds_store_b32(addr, data0, offset0, offset1);
            }
            I::DS_STORE_B64 => {
                self.ds_store_b64(addr, data0, offset0, offset1);
            }
            I::DS_STORE_B96 => {
                self.ds_store_b96(addr, data0, offset0, offset1);
            }
            I::DS_STORE_B128 => {
                self.ds_store_b128(addr, data0, offset0, offset1);
            }
            I::DS_LOAD_2ADDR_B32 => {
                self.ds_load_2addr_b32(addr, vdst, offset0, offset1);
            }
            I::DS_LOAD_2ADDR_B64 => {
                self.ds_load_2addr_b64(addr, vdst, offset0, offset1);
            }
            I::DS_STORE_2ADDR_B32 => {
                self.ds_store_2addr_b32(addr, data0, data1, offset0, offset1);
            }
            I::DS_STORE_2ADDR_B64 => {
                self.ds_store_2addr_b64(addr, data0, data1, offset0, offset1);
            }
            I::DS_BPERMUTE_B32 => {
                self.ds_bpermute_b32(addr, data0, vdst, offset0);
            }
            op => unimplemented!("{:?}", op),
        }
        Signals::None
    }

    fn ds_bpermute_b32(&mut self, addr: usize, data0: usize, vdst: usize, offset0: u8) {
        let values = (0..32)
            .map(|elem| self.read_vgpr(elem, data0))
            .collect::<Vec<u32>>();
        let active = (0..32)
            .map(|elem| self.get_exec_bit(elem))
            .collect::<Vec<bool>>();

        for elem in 0..32 {
            if !active[elem] {
                continue;
            }

            let lane =
                ((self.read_vgpr(elem, addr).wrapping_add(offset0 as u32) >> 2) & 31) as usize;
            let value = if active[lane] { values[lane] } else { 0 };
            self.write_vgpr(elem, vdst, value);
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
            op => unimplemented!("{:?}", op),
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
    pub fn new(
        pc: usize,
        insts: Vec<u8>,
        num_vgprs: usize,
        lds: Rc<RefCell<Vec<u8>>>,
        engine: Engine,
    ) -> Self {
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
                engine: engine,
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
    engine: Engine,
}

unsafe impl<'a> Send for SIMD32 {}

impl<'a> RDNAProcessor<'a> {
    pub fn new(
        aql: &HsaKernelDispatchPacket<'a>,
        num_cunits: usize,
        wavefront_size: usize,
        mem: &Vec<u8>,
    ) -> Self {
        Self::with_engine(aql, num_cunits, wavefront_size, mem, Engine::LlvmJit)
    }

    pub fn with_engine(
        aql: &HsaKernelDispatchPacket<'a>,
        num_cunits: usize,
        wavefront_size: usize,
        mem: &Vec<u8>,
        engine: Engine,
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
                    engine,
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
            engine: engine,
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

        if USE_ENTIRE_KERNEL_TRANSLATION && self.engine == Engine::LlvmJit {
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
