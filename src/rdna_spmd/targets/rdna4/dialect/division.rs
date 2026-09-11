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
    let bits = a[..3].iter().map(|&a| LLVMBuildBitCast(e.b, a, e.ty(word), n)).collect::<Vec<_>>();
    let or = |x, y| LLVMBuildOr(e.b, x, y, n);
    let and = |x, y| LLVMBuildAnd(e.b, x, y, n);
    let magnitude = |index: usize| and(bits[index], k(sign - 1));
    let is = |predicate, index: usize, value| LLVMBuildICmp(e.b, predicate, magnitude(index), k(value), n);
    let zero = |index| is(LLVMIntEQ, index, 0);
    let infinite = |index| is(LLVMIntEQ, index, infinity);
    let nan = |index| is(LLVMIntUGT, index, infinity);
    let quiet = |index: usize| or(bits[index], k(1 << (fraction - 1)));
    let sign_out = and(LLVMBuildXor(e.b, bits[1], bits[2], n), k(sign));
    let signed_inf = or(sign_out, k(infinity));
    let exponent = |index: usize| {
        let bits = LLVMBuildLShr(e.b, bits[index], k(fraction), n);
        let bits = and(bits, k(exponent_mask));
        if word == Ty::I64 { LLVMBuildTrunc(e.b, bits, e.ty(Ty::I32), n) } else { bits }
    };
    let delta = LLVMBuildSub(e.b, exponent(2), exponent(1), n);
    let tiny = LLVMBuildICmp(e.b, LLVMIntSLT, delta, e.constant(Ty::I32, underflow as u32 as u64), n);
    let full_exp = LLVMBuildICmp(e.b, LLVMIntEQ, exponent(1), e.constant(Ty::I32, exponent_mask), n);
    let select = |c, t, f| LLVMBuildSelect(e.b, c, t, f, n);
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
    let fixed = or(or(full_exp, tiny), or(or(under, over), or(invalid, or(nan(1), nan(2)))));
    let capped = select(nan(0), k(infinity), magnitude(0));
    let result = select(fixed, special, or(capped, sign_out));
    LLVMBuildBitCast(e.b, result, e.ty(ty), n)

}

pub(super) unsafe fn fixup_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { fixup(e, Ty::F64, a) }
