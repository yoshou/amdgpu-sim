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
unsafe fn scale_operand(e: &Emitter, ty: Ty, a: &[LLVMValueRef]) -> Vec<LLVMValueRef> {
    use llvm_sys::{LLVMIntPredicate::*, LLVMRealPredicate::*};
    let n = b"\0".as_ptr().cast();
    let (word, fraction, mask, threshold, reciprocal, tiny, amount, nan) = if ty == Ty::F32 {
        (Ty::I32, 23, 0xff, 96, 253, 23, 64i32, 0xffc0_0000)
    } else { (Ty::I64, 52, 0x7ff, 768, 2045, 53, 128, 0xfff8_0000_0000_0000) };
    let k = |x: i32| e.constant(Ty::I32, x as u32 as u64);
    let exponent = |value| {
        let bits = LLVMBuildBitCast(e.b, value, e.ty(word), n);
        let bits = LLVMBuildLShr(e.b, bits, e.constant(word, fraction), n);
        let bits = LLVMBuildAnd(e.b, bits, e.constant(word, mask), n);
        if word == Ty::I64 { LLVMBuildTrunc(e.b, bits, e.ty(Ty::I32), n) } else { bits }
    };
    let den = exponent(a[1]); let num = exponent(a[2]);
    let delta = LLVMBuildSub(e.b, num, den, n);
    let over = LLVMBuildICmp(e.b, LLVMIntSGE, delta, k(threshold), n);
    let under = LLVMBuildICmp(e.b, LLVMIntSLE, delta, k(-threshold), n);
    let reciprocal_tiny = LLVMBuildICmp(e.b, LLVMIntSGE, den, k(reciprocal), n);
    let operands_tiny = LLVMBuildOr(e.b, LLVMBuildICmp(e.b, LLVMIntEQ, den, k(0), n),
        LLVMBuildICmp(e.b, LLVMIntSLE, num, k(tiny), n), n);
    let zero = e.constant(ty, 0);
    let invalid = LLVMBuildOr(e.b, LLVMBuildFCmp(e.b, LLVMRealOEQ, a[1], zero, n),
        LLVMBuildFCmp(e.b, LLVMRealOEQ, a[2], zero, n), n);
    let is_den = LLVMBuildFCmp(e.b, LLVMRealOEQ, a[0], a[1], n);
    let is_num = LLVMBuildFCmp(e.b, LLVMRealOEQ, a[0], a[2], n);
    let den_up = LLVMBuildSelect(e.b, is_den, k(amount), k(0), n);
    let den_down = LLVMBuildSelect(e.b, is_den, k(-amount), k(0), n);
    let num_up = LLVMBuildSelect(e.b, is_num, k(amount), k(0), n);
    let mut shift = LLVMBuildSelect(e.b, operands_tiny, k(amount), k(0), n);
    shift = LLVMBuildSelect(e.b, reciprocal_tiny, k(-amount), shift, n);
    let under_shift = LLVMBuildSelect(e.b, reciprocal_tiny, den_down, num_up, n);
    shift = LLVMBuildSelect(e.b, under, under_shift, shift, n);
    shift = LLVMBuildSelect(e.b, over, den_up, shift, n);
    let scaled = if ty == Ty::F32 { scale::f32(e, &[a[0], shift]) } else { scale::f64(e, &[a[0], shift]) };
    let unchanged = LLVMBuildICmp(e.b, LLVMIntEQ, shift, k(0), n);
    let result = LLVMBuildSelect(e.b, unchanged, a[0], scaled, n);
    let result = LLVMBuildSelect(e.b, invalid, e.constant(ty, nan), result, n);
    let flag = LLVMBuildOr(e.b, invalid, LLVMBuildOr(e.b, over, under, n), n);
    vec![result, flag]
}
pub(super) unsafe fn scale_f32(e: &Emitter, a: &[LLVMValueRef]) -> Vec<LLVMValueRef> { scale_operand(e, Ty::F32, a) }
pub(super) unsafe fn scale_f64(e: &Emitter, a: &[LLVMValueRef]) -> Vec<LLVMValueRef> { scale_operand(e, Ty::F64, a) }

#[cfg(test)]
pub(in crate::rdna_spmd) fn reference_scale(ty: Ty, operands: [u64; 3]) -> (u64, bool) {
    // §16.12 scaling rules with the captured exponent thresholds and the
    // authorized zero-input rule. libm supplies independently rounded scaling.
    let (a, den, num, fraction, exponent_mask, limit, reciprocal, tiny, amount, nan) = if ty == Ty::F32 {
        (f32::from_bits(operands[0] as u32) as f64, f32::from_bits(operands[1] as u32) as f64,
         f32::from_bits(operands[2] as u32) as f64, 23, 255, 96, 253, 23, 64, 0xffc0_0000)
    } else { (f64::from_bits(operands[0]), f64::from_bits(operands[1]), f64::from_bits(operands[2]),
        52, 2047, 768, 2045, 53, 128, 0xfff8_0000_0000_0000) };
    if den == 0. || num == 0. { return (nan, true); }
    let den_exp = ((operands[1] >> fraction) & exponent_mask) as i32;
    let num_exp = ((operands[2] >> fraction) & exponent_mask) as i32;
    let delta = num_exp - den_exp;
    let shift = match delta {
        d if d >= limit => if a == den { amount } else { 0 },
        d if d <= -limit && den_exp >= reciprocal => if a == den { -amount } else { 0 },
        d if d <= -limit => if a == num { amount } else { 0 },
        _ if den_exp >= reciprocal => -amount,
        _ if den_exp == 0 || num_exp <= tiny => amount,
        _ => 0,
    };
    let result = if shift == 0 { operands[0] }
        else if ty == Ty::F32 { libm::scalbnf(f32::from_bits(operands[0] as u32), shift).to_bits() as u64 }
        else { libm::scalbn(a, shift).to_bits() };
    (result, delta.abs() >= limit)
}

unsafe fn fmas(e: &Emitter, ty: Ty, a: &[LLVMValueRef]) -> LLVMValueRef {
    let fused = e.call(&format!("llvm.fma.{}", e.suffix(ty)), ty, &a[..3]);
    let factor = if ty == Ty::F32 { (2f32.powi(32)).to_bits() as u64 } else { (2f64.powi(64)).to_bits() };
    let scaled = LLVMBuildFMul(e.b, fused, e.constant(ty, factor), b"\0".as_ptr().cast());
    LLVMBuildSelect(e.b, a[3], scaled, fused, b"\0".as_ptr().cast())
}
pub(super) unsafe fn fmas_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { fmas(e, Ty::F32, a) }
pub(super) unsafe fn fmas_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { fmas(e, Ty::F64, a) }

pub(super) unsafe fn fixup_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { fixup(e, Ty::F32, a) }
unsafe fn fixup(e: &Emitter, ty: Ty, a: &[LLVMValueRef]) -> LLVMValueRef {
    use llvm_sys::LLVMIntPredicate::*;
    let n = b"\0".as_ptr().cast();
    let (word, fraction, exponent_mask, sign, infinity, underflow) = if ty == Ty::F32 {
        (Ty::I32, 23, 0xff, 0x8000_0000u64, 0x7f80_0000u64, -150i32)
    } else { (Ty::I64, 52, 0x7ff, 0x8000_0000_0000_0000, 0x7ff0_0000_0000_0000, -1075) };
    let k = |bits| e.constant(word, bits);
    let class = |value, mask| e.call(&format!("llvm.is.fpclass.{}", e.suffix(ty)), Ty::I1,
        &[value, LLVMConstInt(LLVMInt32TypeInContext(e.ctx), mask, 0)]);
    let bits = a[..3].iter().map(|&a| LLVMBuildBitCast(e.b, a, e.ty(word), n)).collect::<Vec<_>>();
    let quiet_bit = k(1 << (fraction - 1));
    let quiet = |index| LLVMBuildSelect(e.b, class(a[index], 3), LLVMBuildOr(e.b, bits[index], quiet_bit, n), bits[index], n);
    let sign_out = LLVMBuildAnd(e.b, LLVMBuildXor(e.b, bits[1], bits[2], n), k(sign), n);
    let signed_inf = LLVMBuildOr(e.b, sign_out, k(infinity), n);
    let den_zero = class(a[1], 0x60); let num_zero = class(a[2], 0x60);
    let den_inf = class(a[1], 0x204); let num_inf = class(a[2], 0x204);
    let exponent = |index| {
        let bits = LLVMBuildLShr(e.b, bits[index], k(fraction), n);
        let bits = LLVMBuildAnd(e.b, bits, k(exponent_mask), n);
        if word == Ty::I64 { LLVMBuildTrunc(e.b, bits, e.ty(Ty::I32), n) } else { bits }
    };
    let den_exp = exponent(1); let num_exp = exponent(2);
    let delta = LLVMBuildSub(e.b, num_exp, den_exp, n);
    let tiny = LLVMBuildICmp(e.b, LLVMIntSLT, delta, e.constant(Ty::I32, underflow as u32 as u64), n);
    let full_exp = LLVMBuildICmp(e.b, LLVMIntEQ, den_exp, e.constant(Ty::I32, exponent_mask), n);
    let magnitude = LLVMBuildAnd(e.b, quiet(0), k(sign - 1), n);
    let mut result = LLVMBuildOr(e.b, magnitude, sign_out, n);
    result = LLVMBuildSelect(e.b, full_exp, signed_inf, result, n);
    result = LLVMBuildSelect(e.b, tiny, sign_out, result, n);
    result = LLVMBuildSelect(e.b, LLVMBuildOr(e.b, den_inf, num_zero, n), sign_out, result, n);
    result = LLVMBuildSelect(e.b, LLVMBuildOr(e.b, den_zero, num_inf, n), signed_inf, result, n);
    let invalid = LLVMBuildOr(e.b, LLVMBuildAnd(e.b, den_zero, num_zero, n), LLVMBuildAnd(e.b, den_inf, num_inf, n), n);
    result = LLVMBuildSelect(e.b, invalid, k(sign | infinity | (1 << (fraction - 1))), result, n);
    result = LLVMBuildSelect(e.b, class(a[1], 3), quiet(1), result, n);
    result = LLVMBuildSelect(e.b, class(a[2], 3), quiet(2), result, n);
    LLVMBuildBitCast(e.b, result, e.ty(ty), n)
}

/// Preserve the existing SPMD division expansion; its quotient is formed from
/// the original denominator and numerator before applying the fixup.
pub(super) unsafe fn fixup_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef {
    let [quotient,denominator,numerator]=[a[0],a[1],a[2]];
    use llvm_sys::LLVMIntPredicate::*;
    const INFINITY: u64 = 0x7FF0_0000_0000_0000;
    let n = b"\0".as_ptr().cast();
    let k = |v: u64| e.constant(Ty::I64,v);
    let bits = |v: LLVMValueRef| LLVMBuildBitCast(e.b, v, e.ty(Ty::I64), n);
    let icmp = |p, x, y| LLVMBuildICmp(e.b, p, x, y, n);
    let or = |x, y| LLVMBuildOr(e.b, x, y, n);
    let and = |x, y| LLVMBuildAnd(e.b, x, y, n);
    let select = |c, t, f| LLVMBuildSelect(e.b, c, t, f, n);

    let b = bits(denominator);
    let c = bits(numerator);
    let abs_b = LLVMBuildAnd(e.b,b, k(0x7FFF_FFFF_FFFF_FFFF), n);
    let abs_c = LLVMBuildAnd(e.b,c, k(0x7FFF_FFFF_FFFF_FFFF), n);
    let b_nan = icmp(LLVMIntUGT, abs_b, k(INFINITY));
    let c_nan = icmp(LLVMIntUGT, abs_c, k(INFINITY));
    let both_zero = and(icmp(LLVMIntEQ, abs_b, k(0)), icmp(LLVMIntEQ, abs_c, k(0)));
    let both_infinite = and(
        icmp(LLVMIntEQ, abs_b, k(INFINITY)),
        icmp(LLVMIntEQ, abs_c, k(INFINITY)),
    );
    let exponent =
        |v: LLVMValueRef| LLVMBuildAnd(e.b,LLVMBuildLShr(e.b, v, k(52), n), k(0x7FF), n);
    let underflow = icmp(
        LLVMIntSLT,
        LLVMBuildSub(e.b,exponent(c), exponent(b), n),
        k((-1075i64) as u64),
    );

    // The answer for those cases, chosen in reverse so that the earlier
    // ones of the ISA's order win. It is made of the operands alone, so
    // only the last select sits on the quotient's dependency chain.
    let quiet = |v: LLVMValueRef| LLVMBuildOr(e.b,v, k(0x0008_0000_0000_0000), n);
    let signed_zero = LLVMBuildAnd(e.b,LLVMBuildXor(e.b,b, c, n), k(0x8000_0000_0000_0000), n);
    let mut fixed = signed_zero;
    fixed = select(or(both_zero, both_infinite), k(0xFFF8_0000_0000_0000), fixed);
    fixed = select(b_nan, quiet(b), fixed);
    fixed = select(c_nan, quiet(c), fixed);
    let fix = or(or(underflow, or(both_zero, both_infinite)), or(b_nan, c_nan));
    LLVMBuildBitCast(e.b, select(fix, fixed, bits(quotient)), e.ty(Ty::F64), n)
}
