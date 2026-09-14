//! ISA §16.12 division macro primitives. FIXUP consumes the supplied
//! approximation; recognizing a complete division macro is a separate pass.
//! FMAS is fused, supports input denormals and retains output denormals.
use super::*;
#[cfg(test)]
mod tests;

// Hardware captures in tests/isa/vop3/scalar_dst.rs take precedence over
// §16.12's pseudocode (user decision, 2026-09-06). Zero operands produce
// negative qNaN and set the flag. The quotient thresholds are symmetric
// exponent differences; a very large denominator scales down.
fn scale_operand(e: &Emitter, ty: Ty, a: &[Value]) -> Vec<Value> {
    let ir = e.ir;
    let (word, fraction, mask, threshold, reciprocal, tiny, amount, nan) = if ty == Ty::F32 {
        (Ty::I32, 23, 0xff, 96, 253, 23, 64i32, 0xffc0_0000)
    } else {
        (
            Ty::I64,
            52,
            0x7ff,
            768,
            2045,
            53,
            128,
            0xfff8_0000_0000_0000,
        )
    };
    let k = |x: i32| e.constant(Ty::I32, x as u32 as u64);
    let exponent = |value| {
        let bits = ir.bitcast(value, e.ty(word));
        let bits = ir.lshr(bits, e.constant(word, fraction));
        let bits = ir.and(bits, e.constant(word, mask));
        if word == Ty::I64 {
            ir.trunc(bits, e.ty(Ty::I32))
        } else {
            bits
        }
    };
    let den = exponent(a[1]);
    let num = exponent(a[2]);
    let delta = ir.sub(num, den);
    let over = ir.icmp(IntPred::Sge, delta, k(threshold));
    let under = ir.icmp(IntPred::Sle, delta, k(-threshold));
    let reciprocal_tiny = ir.icmp(IntPred::Sge, den, k(reciprocal));
    let operands_tiny = ir.or(
        ir.icmp(IntPred::Eq, den, k(0)),
        ir.icmp(IntPred::Sle, num, k(tiny)),
    );
    let zero = e.constant(ty, 0);
    let invalid = ir.or(
        ir.fcmp(FloatPred::Oeq, a[1], zero),
        ir.fcmp(FloatPred::Oeq, a[2], zero),
    );
    let is_den = ir.fcmp(FloatPred::Oeq, a[0], a[1]);
    let is_num = ir.fcmp(FloatPred::Oeq, a[0], a[2]);
    let den_up = ir.select(is_den, k(amount), k(0));
    let den_down = ir.select(is_den, k(-amount), k(0));
    let num_up = ir.select(is_num, k(amount), k(0));
    let mut shift = ir.select(operands_tiny, k(amount), k(0));
    shift = ir.select(reciprocal_tiny, k(-amount), shift);
    let under_shift = ir.select(reciprocal_tiny, den_down, num_up);
    shift = ir.select(under, under_shift, shift);
    shift = ir.select(over, den_up, shift);
    let scaled = if ty == Ty::F32 {
        scale::f32(e, &[a[0], shift])
    } else {
        scale::f64(e, &[a[0], shift])
    };
    let unchanged = ir.icmp(IntPred::Eq, shift, k(0));
    let result = ir.select(unchanged, a[0], scaled);
    let result = ir.select(invalid, e.constant(ty, nan), result);
    let flag = ir.or(invalid, ir.or(over, under));
    vec![result, flag]
}
pub(super) fn scale_f32(e: &Emitter, a: &[Value]) -> Vec<Value> {
    scale_operand(e, Ty::F32, a)
}
pub(super) fn scale_f64(e: &Emitter, a: &[Value]) -> Vec<Value> {
    scale_operand(e, Ty::F64, a)
}

#[cfg(test)]
pub(in crate::rdna_spmd) fn reference_scale(ty: Ty, operands: [u64; 3]) -> (u64, bool) {
    // §16.12 scaling rules with the captured exponent thresholds and the
    // authorized zero-input rule. libm supplies independently rounded scaling.
    let (a, den, num, fraction, exponent_mask, limit, reciprocal, tiny, amount, nan) =
        if ty == Ty::F32 {
            (
                f32::from_bits(operands[0] as u32) as f64,
                f32::from_bits(operands[1] as u32) as f64,
                f32::from_bits(operands[2] as u32) as f64,
                23,
                255,
                96,
                253,
                23,
                64,
                0xffc0_0000,
            )
        } else {
            (
                f64::from_bits(operands[0]),
                f64::from_bits(operands[1]),
                f64::from_bits(operands[2]),
                52,
                2047,
                768,
                2045,
                53,
                128,
                0xfff8_0000_0000_0000,
            )
        };
    if den == 0. || num == 0. {
        return (nan, true);
    }
    let den_exp = ((operands[1] >> fraction) & exponent_mask) as i32;
    let num_exp = ((operands[2] >> fraction) & exponent_mask) as i32;
    let delta = num_exp - den_exp;
    let shift = match delta {
        d if d >= limit => {
            if a == den {
                amount
            } else {
                0
            }
        }
        d if d <= -limit && den_exp >= reciprocal => {
            if a == den {
                -amount
            } else {
                0
            }
        }
        d if d <= -limit => {
            if a == num {
                amount
            } else {
                0
            }
        }
        _ if den_exp >= reciprocal => -amount,
        _ if den_exp == 0 || num_exp <= tiny => amount,
        _ => 0,
    };
    let result = if shift == 0 {
        operands[0]
    } else if ty == Ty::F32 {
        libm::scalbnf(f32::from_bits(operands[0] as u32), shift).to_bits() as u64
    } else {
        libm::scalbn(a, shift).to_bits()
    };
    (result, delta.abs() >= limit)
}

fn fmas(e: &Emitter, ty: Ty, a: &[Value]) -> Value {
    let fused = e.call(&format!("llvm.fma.{}", e.suffix(ty)), ty, &a[..3]);
    let factor = if ty == Ty::F32 {
        (2f32.powi(32)).to_bits() as u64
    } else {
        (2f64.powi(64)).to_bits()
    };
    let scaled = e.ir.fmul(fused, e.constant(ty, factor));
    e.ir.select(a[3], scaled, fused)
}
pub(super) fn fmas_f32(e: &Emitter, a: &[Value]) -> Value {
    fmas(e, Ty::F32, a)
}
pub(super) fn fmas_f64(e: &Emitter, a: &[Value]) -> Value {
    fmas(e, Ty::F64, a)
}

pub(super) fn fixup_f32(e: &Emitter, a: &[Value]) -> Value {
    fixup(e, Ty::F32, a)
}
fn fixup(e: &Emitter, ty: Ty, a: &[Value]) -> Value {
    let ir = e.ir;
    let (word, fraction, exponent_mask, sign, infinity, underflow) = if ty == Ty::F32 {
        (Ty::I32, 23, 0xff, 0x8000_0000u64, 0x7f80_0000u64, -150i32)
    } else {
        (
            Ty::I64,
            52,
            0x7ff,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            -1075,
        )
    };
    let k = |bits| e.constant(word, bits);
    let bits = a[..3]
        .iter()
        .map(|&a| ir.bitcast(a, e.ty(word)))
        .collect::<Vec<_>>();
    let or = |x, y| ir.or(x, y);
    let and = |x, y| ir.and(x, y);
    let magnitude = |index: usize| and(bits[index], k(sign - 1));
    let is = |predicate, index: usize, value| ir.icmp(predicate, magnitude(index), k(value));
    let zero = |index| is(IntPred::Eq, index, 0);
    let infinite = |index| is(IntPred::Eq, index, infinity);
    let nan = |index| is(IntPred::Ugt, index, infinity);
    let quiet = |index: usize| or(bits[index], k(1 << (fraction - 1)));
    let sign_out = and(ir.xor(bits[1], bits[2]), k(sign));
    let signed_inf = or(sign_out, k(infinity));
    let exponent = |index: usize| {
        let bits = ir.lshr(bits[index], k(fraction));
        let bits = and(bits, k(exponent_mask));
        if word == Ty::I64 {
            ir.trunc(bits, e.ty(Ty::I32))
        } else {
            bits
        }
    };
    let delta = ir.sub(exponent(2), exponent(1));
    let tiny = ir.icmp(
        IntPred::Slt,
        delta,
        e.constant(Ty::I32, underflow as u32 as u64),
    );
    let full_exp = ir.icmp(IntPred::Eq, exponent(1), e.constant(Ty::I32, exponent_mask));
    let select = |c, t, f| ir.select(c, t, f);
    let invalid = or(and(zero(1), zero(2)), and(infinite(1), infinite(2)));
    let over = or(zero(1), infinite(2));
    let under = or(infinite(1), zero(2));
    let mut special = signed_inf;
    special = select(tiny, sign_out, special);
    special = select(under, sign_out, special);
    special = select(over, signed_inf, special);
    special = select(invalid, k(sign | infinity | (1 << (fraction - 1))), special);
    special = select(nan(1), quiet(1), special);
    special = select(nan(2), quiet(2), special);
    let fixed = or(
        or(full_exp, tiny),
        or(or(under, over), or(invalid, or(nan(1), nan(2)))),
    );
    let capped = select(nan(0), k(infinity), magnitude(0));
    let result = select(fixed, special, or(capped, sign_out));
    ir.bitcast(result, e.ty(ty))
}

pub(super) fn fixup_f64(e: &Emitter, a: &[Value]) -> Value {
    fixup(e, Ty::F64, a)
}
