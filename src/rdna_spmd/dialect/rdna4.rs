//! RDNA4 ISA §16.8 and §16.12 reciprocal, reciprocal square root and square
//! root. The design selects exact host division/sqrt within the ISA error
//! allowance. F32 flushes input/output subnormals to signed zero; F64 preserves
//! them. Modifiers are applied by lift before/after the target operation.
use super::{DialectRegistry, Effect, Implementation, Operation, TargetOp};
use crate::instructions::I;
use crate::rdna_spmd::{ir::Ty, codegen::ops::Emitter};
use llvm_sys::{core::*, prelude::*};
mod scale;
mod reduction;
mod division;
mod image;
pub(in crate::rdna_spmd) mod bvh;
#[cfg(test)]
pub(in crate::rdna_spmd) use reduction::reference as reference_reduction;

pub(in crate::rdna_spmd) const ID: u32 = 0x52444e34;

pub(super) fn register(registry: &mut DialectRegistry) -> Result<(), &'static str> {
    registry.register(ID, 36, Operation { name: "image_sample_lz", effect: Effect::ReadGlobal { every_lane: true },
        immediates: &[(12, 3)], inputs: &[Ty::I32, Ty::I32, Ty::I32, Ty::I32, Ty::I32, Ty::I32, Ty::I32, Ty::I32,
            Ty::I32, Ty::I32, Ty::I32, Ty::I32, Ty::I32, Ty::I1, Ty::F32, Ty::F32],
        outputs: vec![Ty::I32], lower: Implementation::Single(image::sample) })?;
    for (id, name, ty, lower) in [
        (1, "rcp.f32", Ty::F32, rcp_f32 as unsafe fn(&Emitter, &[LLVMValueRef]) -> LLVMValueRef),
        (2, "rcp.f64", Ty::F64, rcp_f64),
        (3, "rsq.f32", Ty::F32, rsq_f32),
        (4, "rsq.f64", Ty::F64, rsq_f64),
        (5, "sqrt.f32", Ty::F32, sqrt_f32),
        (6, "sqrt.f64", Ty::F64, sqrt_f64),
        (20, "floor.f32", Ty::F32, floor_f32),
        (21, "ceil.f32", Ty::F32, ceil_f32),
        (22, "trunc.f32", Ty::F32, trunc_f32),
        (23, "rndne.f32", Ty::F32, round_f32),
        (24, "rndne.f64", Ty::F64, round_f64),
        (25, "fract.f64", Ty::F64, fract_f64),
        (26, "floor.f64", Ty::F64, floor_f64),
        (27, "trunc.f64", Ty::F64, trunc_f64),
        (28, "exp2.f32", Ty::F32, exp2_f32),
        (29, "log2.f32", Ty::F32, log2_f32),
        (30, "sin.f32", Ty::F32, sin_f32),
        (31, "cos.f32", Ty::F32, cos_f32),
    ] {
        registry.register(ID, id, Operation { effect: Effect::Pure, immediates: &[], name, inputs: if ty == Ty::F32 { &[Ty::F32] } else { &[Ty::F64] }, outputs: vec![ty], lower: Implementation::Single(lower) })?;
    }
    for (id, name, input, output, lower) in [
        (9, "frexp_mant.f32", Ty::F32, Ty::F32, mant_f32 as unsafe fn(&Emitter, &[LLVMValueRef]) -> LLVMValueRef),
        (10, "frexp_mant.f64", Ty::F64, Ty::F64, mant_f64),
        (11, "frexp_exp.f32", Ty::F32, Ty::I32, exp_f32),
        (12, "frexp_exp.f64", Ty::F64, Ty::I32, exp_f64),
    ] {
        registry.register(ID, id, Operation { effect: Effect::Pure, immediates: &[], name, inputs: if input == Ty::F32 { &[Ty::F32] } else { &[Ty::F64] }, outputs: vec![output], lower: Implementation::Single(lower) })?;
    }
    registry.register(ID, 32, Operation { effect: Effect::Pure, immediates: &[], name: "cmp_class.f32", inputs: &[Ty::F32, Ty::I32], outputs: vec![Ty::I1], lower: Implementation::Single(class_f32) })?;
    registry.register(ID, 33, Operation { effect: Effect::Pure, immediates: &[], name: "cmp_class.f64", inputs: &[Ty::F64, Ty::I32], outputs: vec![Ty::I1], lower: Implementation::Single(class_f64) })?;
    registry.register(ID, 34, Operation { effect: Effect::Pure, immediates: &[], name: "cvt.f32.f16", inputs: &[Ty::I32], outputs: vec![Ty::F32], lower: Implementation::Single(from_half) })?;
    registry.register(ID, 35, Operation { effect: Effect::Pure, immediates: &[], name: "cvt.f16.f32", inputs: &[Ty::F32], outputs: vec![Ty::I32], lower: Implementation::Single(to_half) })?;
    registry.register(ID, 7, Operation { effect: Effect::Pure, immediates: &[], name: "ldexp.f32", inputs: &[Ty::F32, Ty::I32], outputs: vec![Ty::F32], lower: Implementation::Single(scale::f32) })?;
    registry.register(ID, 8, Operation { effect: Effect::Pure, immediates: &[], name: "ldexp.f64", inputs: &[Ty::F64, Ty::I32], outputs: vec![Ty::F64], lower: Implementation::Single(scale::f64) })?;
    registry.register(ID, 13, Operation { effect: Effect::Pure, immediates: &[], name: "trig_preop.f64", inputs: &[Ty::F64, Ty::I32], outputs: vec![Ty::F64], lower: Implementation::Single(reduction::lower) })?;
    registry.register(ID, 17, Operation { effect: Effect::Pure, immediates: &[], name: "div_fixup.f64", inputs: &[Ty::F64, Ty::F64, Ty::F64], outputs: vec![Ty::F64], lower: Implementation::Single(division::fixup_f64) })?;
    registry.register(ID, 16, Operation { effect: Effect::Pure, immediates: &[], name: "div_fixup.f32", inputs: &[Ty::F32, Ty::F32, Ty::F32], outputs: vec![Ty::F32], lower: Implementation::Single(division::fixup_f32) })?;
    registry.register(ID, 18, Operation { effect: Effect::Pure, immediates: &[], name: "div_fmas.f32", inputs: &[Ty::F32, Ty::F32, Ty::F32, Ty::I1], outputs: vec![Ty::F32], lower: Implementation::Single(division::fmas_f32) })?;
    registry.register(ID, 19, Operation { effect: Effect::Pure, immediates: &[], name: "div_fmas.f64", inputs: &[Ty::F64, Ty::F64, Ty::F64, Ty::I1], outputs: vec![Ty::F64], lower: Implementation::Single(division::fmas_f64) })?;
    registry.register(ID, 14, Operation { effect: Effect::Pure, immediates: &[], name: "div_scale.f32", inputs: &[Ty::F32, Ty::F32, Ty::F32], outputs: vec![Ty::F32, Ty::I1], lower: Implementation::Multiple(division::scale_f32) })?;
    registry.register(ID, 15, Operation { effect: Effect::Pure, immediates: &[], name: "div_scale.f64", inputs: &[Ty::F64, Ty::F64, Ty::F64], outputs: vec![Ty::F64, Ty::I1], lower: Implementation::Multiple(division::scale_f64) })?;
    registry.register(ID, 37, Operation { effect: Effect::ReadGlobal { every_lane: false }, immediates: &[], name: "image_bvh64_intersect_ray", inputs: &[Ty::I32,Ty::I32,Ty::I64,Ty::I32,Ty::I32,Ty::I32,Ty::I32,Ty::I32,Ty::I32,Ty::I32,Ty::I32,Ty::I32,Ty::I32,Ty::I1,Ty::I32], outputs: vec![Ty::I32;4], lower: Implementation::Multiple(bvh::lower) })?;
    Ok(())
}

pub(in crate::rdna_spmd) fn image_sample(registry: &DialectRegistry) -> TargetOp {
    registry.lookup(ID, "image_sample_lz").expect("missing RDNA4 image provider")
}

pub(in crate::rdna_spmd) fn comparison(registry: &DialectRegistry, ty: Ty) -> TargetOp {
    registry.lookup(ID, match ty { Ty::F32 => "cmp_class.f32", Ty::F64 => "cmp_class.f64", _ => unreachable!() })
        .expect("missing RDNA4 comparison provider")
}

// §16.9/§16.12 CMP_CLASS: the ten ISA class bits are the IEEE categories in
// the same order as LLVM's class immediate. The selector itself is a runtime
// lane value, so it must never occupy that intrinsic's immediate argument.
unsafe fn classify(e: &Emitter, ty: Ty, value: LLVMValueRef, selector: LLVMValueRef) -> LLVMValueRef {
    let n = b"\0".as_ptr().cast();
    let mut result = e.constant(Ty::I1, 0);
    for index in 0..10 {
        let class = e.call(&format!("llvm.is.fpclass.{}", e.suffix(ty)), Ty::I1,
            &[value, LLVMConstInt(LLVMInt32TypeInContext(e.ctx), 1 << index, 0)]);
        let requested = LLVMBuildAnd(e.b, selector, e.constant(Ty::I32, 1 << index), n);
        let requested = LLVMBuildICmp(e.b, llvm_sys::LLVMIntPredicate::LLVMIntNE, requested, e.constant(Ty::I32, 0), n);
        result = LLVMBuildOr(e.b, result, LLVMBuildAnd(e.b, class, requested, n), n);
    }
    result
}
unsafe fn class_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { classify(e, Ty::F32, a[0], a[1]) }
unsafe fn class_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { classify(e, Ty::F64, a[0], a[1]) }

#[cfg(test)]
pub(in crate::rdna_spmd) fn reference_class(ty: Ty, bits: u64, selector: u32) -> bool {
    let (fraction_bits, exponent_bits, sign_bit) = match ty { Ty::F32 => (23, 8, 31), Ty::F64 => (52, 11, 63), _ => unreachable!() };
    let negative = bits >> sign_bit != 0;
    let fraction = bits & ((1 << fraction_bits) - 1);
    let exponent = bits >> fraction_bits & ((1 << exponent_bits) - 1);
    let index = if exponent == (1 << exponent_bits) - 1 {
        if fraction == 0 { if negative { 2 } else { 9 } }
        else if fraction >> (fraction_bits - 1) == 0 { 0 } else { 1 }
    } else if exponent == 0 {
        if fraction == 0 { if negative { 5 } else { 6 } }
        else if negative { 4 } else { 7 }
    } else if negative { 3 } else { 8 };
    selector >> index & 1 != 0
}

pub(in crate::rdna_spmd) fn unary(registry: &DialectRegistry, op: I) -> Option<TargetOp> {
    let name = match op {
        I::V_RCP_F32 | I::V_RCP_IFLAG_F32 | I::V_S_RCP_F32 => "rcp.f32",
        I::V_RCP_F64 => "rcp.f64",
        I::V_RSQ_F32 => "rsq.f32", I::V_RSQ_F64 => "rsq.f64",
        I::V_SQRT_F32 => "sqrt.f32", I::V_SQRT_F64 => "sqrt.f64",
        I::V_FLOOR_F32 => "floor.f32", I::V_FLOOR_F64 => "floor.f64",
        I::V_CEIL_F32 => "ceil.f32",
        I::V_TRUNC_F32 => "trunc.f32", I::V_TRUNC_F64 => "trunc.f64",
        I::V_RNDNE_F32 => "rndne.f32", I::V_RNDNE_F64 => "rndne.f64",
        I::V_FRACT_F64 => "fract.f64",
        I::V_FREXP_MANT_F32 => "frexp_mant.f32", I::V_FREXP_MANT_F64 => "frexp_mant.f64",
        I::V_FREXP_EXP_I32_F32 => "frexp_exp.f32", I::V_FREXP_EXP_I32_F64 => "frexp_exp.f64",
        I::V_CVT_F32_F16 => "cvt.f32.f16", I::V_CVT_F16_F32 => "cvt.f16.f32",
        I::V_EXP_F32 => "exp2.f32", I::V_LOG_F32 => "log2.f32",
        I::V_SIN_F32 => "sin.f32", I::V_COS_F32 => "cos.f32",
        _ => return None,
    };
    Some(registry.lookup(ID, name).expect("missing RDNA4 provider"))
}

pub(in crate::rdna_spmd) fn binary(registry: &DialectRegistry, op: I) -> Option<TargetOp> {
    let name = match op { I::V_LDEXP_F32 => "ldexp.f32", I::V_LDEXP_F64 => "ldexp.f64",
        I::V_TRIG_PREOP_F64 => "trig_preop.f64", _ => return None };
    Some(registry.lookup(ID, name).expect("missing RDNA4 provider"))
}

pub(in crate::rdna_spmd) fn division(registry: &DialectRegistry, op: I) -> Option<TargetOp> {
    let name = match op { I::V_DIV_FIXUP_F32 => "div_fixup.f32", I::V_DIV_FIXUP_F64 => "div_fixup.f64",
        I::V_DIV_SCALE_F32 => "div_scale.f32", I::V_DIV_SCALE_F64 => "div_scale.f64",
        I::V_DIV_FMAS_F32 => "div_fmas.f32", I::V_DIV_FMAS_F64 => "div_fmas.f64", _ => return None };
    Some(registry.lookup(ID, name).expect("missing RDNA4 provider"))
}

unsafe fn flush(e: &Emitter, value: LLVMValueRef) -> LLVMValueRef {
    let n = b"\0".as_ptr().cast();
    // is.fpclass has a scalar immediate even for vector operands. Selecting
    // the sign bit preserves negative zero and all NaN/Inf payloads.
    let tiny = e.call(&format!("llvm.is.fpclass.{}", e.suffix(Ty::F32)), Ty::I1,
        &[value, LLVMConstInt(LLVMInt32TypeInContext(e.ctx), 0x90, 0)]);
    let bits = LLVMBuildBitCast(e.b, value, e.ty(Ty::I32), n);
    let sign = LLVMBuildAnd(e.b, bits, e.constant(Ty::I32, 0x8000_0000), n);
    LLVMBuildBitCast(e.b, LLVMBuildSelect(e.b, tiny, sign, bits, n), e.ty(Ty::F32), n)
}

unsafe fn math(e: &Emitter, ty: Ty, mut value: LLVMValueRef, sqrt: bool, reciprocal: bool) -> LLVMValueRef {
    if ty == Ty::F32 { value = flush(e, value); }
    if sqrt { value = e.call(&format!("llvm.sqrt.{}", e.suffix(ty)), ty, &[value]); }
    if reciprocal {
        let one = e.constant(ty, if ty == Ty::F32 { 1f32.to_bits() as u64 } else { 1f64.to_bits() });
        value = LLVMBuildFDiv(e.b, one, value, b"\0".as_ptr().cast());
    }
    if ty == Ty::F32 { value = flush(e, value); }
    value
}
unsafe fn rcp_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { math(e, Ty::F32, a[0], false, true) }
unsafe fn rcp_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { math(e, Ty::F64, a[0], false, true) }
unsafe fn rsq_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { math(e, Ty::F32, a[0], true, true) }
unsafe fn rsq_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { math(e, Ty::F64, a[0], true, true) }
unsafe fn sqrt_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { math(e, Ty::F32, a[0], true, false) }
unsafe fn sqrt_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { math(e, Ty::F64, a[0], true, false) }

// §16.8/§16.12 EXP/LOG allow 1 ULP and flush input/output denormals.
// Use host exp2/log2 with explicit ISA flushing, without approximate fast-math.
unsafe fn exp2_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef {
    flush(e, e.call(&format!("llvm.exp2.{}", e.suffix(Ty::F32)), Ty::F32, &[flush(e, a[0])]))
}
unsafe fn log2_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef {
    let input = flush(e, a[0]);
    let result = flush(e, e.call(&format!("llvm.log2.{}", e.suffix(Ty::F32)), Ty::F32, &[input]));
    // ISA special-value table: negative nonzero inputs produce -qNaN.
    // LLVM constant folding and the host library choose different NaN signs.
    let negative = LLVMBuildFCmp(e.b, llvm_sys::LLVMRealPredicate::LLVMRealOLT,
        input, e.constant(Ty::F32, 0), b"\0".as_ptr().cast());
    LLVMBuildSelect(e.b, negative, e.constant(Ty::F32, 0xffc0_0000), result, b"\0".as_ptr().cast())
}

// §16.8/§16.12 SIN/COS consume revolutions and support the full finite
// range. Removing the nearest integer is exact for binary f32 inputs: small
// inputs subtract zero, intermediate inputs satisfy Sterbenz, large inputs
// are already integers. Evaluate the reduced angle in f64 to avoid a rounded
// f32 multiple of PI. Quarter/half turns have exact results. The named
// hardware fixtures additionally pin signed zero and NaN payload behavior.
unsafe fn trig(e: &Emitter, value: LLVMValueRef, cosine: bool) -> LLVMValueRef {
    use llvm_sys::LLVMRealPredicate::*;
    let n = b"\0".as_ptr().cast();
    let rounded = e.call(&format!("llvm.roundeven.{}", e.suffix(Ty::F32)), Ty::F32, &[value]);
    let reduced = LLVMBuildFSub(e.b, value, rounded, n);
    let wide = LLVMBuildFPExt(e.b, reduced, e.ty(Ty::F64), n);
    let angle = LLVMBuildFMul(e.b, wide, e.constant(Ty::F64, std::f64::consts::TAU.to_bits()), n);
    let op = if cosine { "cos" } else { "sin" };
    let result = e.call(&format!("llvm.{op}.{}", e.suffix(Ty::F64)), Ty::F64, &[angle]);
    let mut result = LLVMBuildFPTrunc(e.b, result, e.ty(Ty::F32), n);
    let abs = e.call(&format!("llvm.fabs.{}", e.suffix(Ty::F32)), Ty::F32, &[reduced]);
    let zero = e.constant(Ty::F32, 0);
    let exact_zero = LLVMBuildFCmp(e.b, LLVMRealOEQ, abs,
        e.constant(Ty::F32, if cosine { 0.25f32 } else { 0.5f32 }.to_bits() as u64), n);
    result = LLVMBuildSelect(e.b, exact_zero, zero, result, n);
    if !cosine {
        let integral = LLVMBuildFCmp(e.b, LLVMRealOEQ, abs, zero, n);
        result = LLVMBuildSelect(e.b, integral, zero, result, n);
        let input_zero = LLVMBuildFCmp(e.b, LLVMRealOEQ, value, zero, n);
        result = LLVMBuildSelect(e.b, input_zero, value, result, n);
    }
    result
}
unsafe fn sin_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { trig(e, a[0], false) }
unsafe fn cos_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { trig(e, a[0], true) }

// §16.8/§16.12 fixed rounding operations preserve subnormals and signed zero.
unsafe fn rounding(e: &Emitter, ty: Ty, op: &str, a: LLVMValueRef) -> LLVMValueRef {
    e.call(&format!("llvm.{op}.{}", e.suffix(ty)), ty, &[a])
}
unsafe fn floor_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { rounding(e, Ty::F32, "floor", a[0]) }
unsafe fn floor_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { rounding(e, Ty::F64, "floor", a[0]) }
unsafe fn ceil_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { rounding(e, Ty::F32, "ceil", a[0]) }
unsafe fn trunc_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { rounding(e, Ty::F32, "trunc", a[0]) }
unsafe fn trunc_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { rounding(e, Ty::F64, "trunc", a[0]) }
unsafe fn round_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { rounding(e, Ty::F32, "roundeven", a[0]) }
unsafe fn round_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { rounding(e, Ty::F64, "roundeven", a[0]) }
unsafe fn fract_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef {
    let n = b"\0".as_ptr().cast();
    let floor = floor_f64(e, a);
    let value = LLVMBuildFSub(e.b, a[0], floor, n);
    // §16.8 V_FRACT_F64 clamps to the largest representable number below 1.
    // Ordered comparison retains NaNs (minnum would replace them by the cap).
    let cap = e.constant(Ty::F64, 0x3fef_ffff_ffff_ffff);
    let over = LLVMBuildFCmp(e.b, llvm_sys::LLVMRealPredicate::LLVMRealOGT, value, cap, n);
    LLVMBuildSelect(e.b, over, cap, value, n)
}

// §16.8 frexp: normalize the significand using integer shifts, so a host's
// floating-point denormal mode cannot erase the smallest input values.
unsafe fn frexp(e: &Emitter, ty: Ty, input: LLVMValueRef) -> (LLVMValueRef, LLVMValueRef) {
    use llvm_sys::LLVMIntPredicate::*;
    let n = b"\0".as_ptr().cast();
    let (word, fraction_bits, exponent_bits, bias) = if ty == Ty::F32 { (Ty::I32, 23, 8, 127) } else { (Ty::I64, 52, 11, 1023) };
    let k = |x| e.constant(word, x);
    let bits = LLVMBuildBitCast(e.b, input, e.ty(word), n);
    let sign = LLVMBuildAnd(e.b, bits, k(1u64 << (word.bits() - 1)), n);
    let fraction = LLVMBuildAnd(e.b, bits, k((1u64 << fraction_bits) - 1), n);
    let exponent = LLVMBuildAnd(e.b, LLVMBuildLShr(e.b, bits, k(fraction_bits), n), k((1 << exponent_bits) - 1), n);
    let exp_zero = LLVMBuildICmp(e.b, LLVMIntEQ, exponent, k(0), n);
    let special = LLVMBuildICmp(e.b, LLVMIntEQ, exponent, k((1 << exponent_bits) - 1), n);
    let fraction_zero = LLVMBuildICmp(e.b, LLVMIntEQ, fraction, k(0), n);
    let zero = LLVMBuildAnd(e.b, exp_zero, fraction_zero, n);
    let lz = e.call(&format!("llvm.ctlz.{}", e.suffix(word)), word,
        &[fraction, LLVMConstInt(LLVMInt1TypeInContext(e.ctx), 0, 0)]);
    let shift = LLVMBuildSub(e.b, lz, k(exponent_bits), n);
    // The non-selected normal-input path also remains free of poison shifts.
    let safe_shift = LLVMBuildAnd(e.b, shift, k(word.bits() as u64 - 1), n);
    let normalized = LLVMBuildAnd(e.b, LLVMBuildShl(e.b, fraction, safe_shift, n), k((1u64 << fraction_bits) - 1), n);
    let mantissa = LLVMBuildSelect(e.b, exp_zero, normalized, fraction, n);
    let mantissa = LLVMBuildOr(e.b, LLVMBuildOr(e.b, sign, mantissa, n), k((bias - 1) << fraction_bits), n);
    let nan = LLVMBuildAnd(e.b, special, LLVMBuildNot(e.b, fraction_zero, n), n);
    let quiet = LLVMBuildOr(e.b, bits, k(1 << (fraction_bits - 1)), n);
    let special_bits = LLVMBuildSelect(e.b, nan, quiet, bits, n);
    let mantissa = LLVMBuildSelect(e.b, LLVMBuildOr(e.b, zero, special, n), special_bits, mantissa, n);
    let normal_exp = LLVMBuildSub(e.b, exponent, k(bias - 1), n);
    let sub_exp = LLVMBuildSub(e.b, LLVMBuildSub(e.b, k(2), k(bias), n), shift, n);
    let exp = LLVMBuildSelect(e.b, exp_zero, sub_exp, normal_exp, n);
    let exp = LLVMBuildSelect(e.b, LLVMBuildOr(e.b, zero, special, n), k(0), exp, n);
    let exp = if word == Ty::I64 { LLVMBuildTrunc(e.b, exp, e.ty(Ty::I32), n) } else { exp };
    (LLVMBuildBitCast(e.b, mantissa, e.ty(ty), n), exp)
}
unsafe fn mant_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { frexp(e, Ty::F32, a[0]).0 }
unsafe fn mant_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { frexp(e, Ty::F64, a[0]).0 }
unsafe fn exp_f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { frexp(e, Ty::F32, a[0]).1 }
unsafe fn exp_f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { frexp(e, Ty::F64, a[0]).1 }

// §16.8/§16.12 CVT_F32_F16 and CVT_F16_F32, default RNE rounding with
// subnormal support. Half types remain local to the provider, never core IR.
unsafe fn from_half(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef {
    let n = b"\0".as_ptr().cast();
    let i16 = e.shaped(LLVMInt16TypeInContext(e.ctx));
    let f16 = e.shaped(LLVMHalfTypeInContext(e.ctx));
    let bits = LLVMBuildTrunc(e.b, a[0], i16, n);
    let value = LLVMBuildBitCast(e.b, bits, f16, n);
    LLVMBuildFPExt(e.b, value, e.ty(Ty::F32), n)
}
unsafe fn to_half(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef {
    let n = b"\0".as_ptr().cast();
    let i16 = e.shaped(LLVMInt16TypeInContext(e.ctx));
    let f16 = e.shaped(LLVMHalfTypeInContext(e.ctx));
    let value = LLVMBuildFPTrunc(e.b, a[0], f16, n);
    let bits = LLVMBuildBitCast(e.b, value, i16, n);
    LLVMBuildZExt(e.b, bits, e.ty(Ty::I32), n)
}

#[cfg(test)]
pub(in crate::rdna_spmd) fn reference(op: I, bits: u64) -> u64 {
    fn flush(v: f32) -> f32 { if v.is_subnormal() { 0f32.copysign(v) } else { v } }
    match op {
        I::V_RCP_F32 | I::V_RCP_IFLAG_F32 | I::V_S_RCP_F32 => flush(1.0 / flush(f32::from_bits(bits as u32))).to_bits() as u64,
        I::V_RSQ_F32 => flush(1.0 / flush(f32::from_bits(bits as u32)).sqrt()).to_bits() as u64,
        I::V_SQRT_F32 => flush(flush(f32::from_bits(bits as u32)).sqrt()).to_bits() as u64,
        I::V_RCP_F64 => (1.0 / f64::from_bits(bits)).to_bits(),
        I::V_RSQ_F64 => (1.0 / f64::from_bits(bits).sqrt()).to_bits(),
        I::V_SQRT_F64 => f64::from_bits(bits).sqrt().to_bits(),
        I::V_EXP_F32 => flush(flush(f32::from_bits(bits as u32)).exp2()).to_bits() as u64,
        I::V_LOG_F32 => {
            let x = flush(f32::from_bits(bits as u32));
            if x < 0.0 { 0xffc0_0000 } else { flush(x.log2()).to_bits() as u64 }
        }
        I::V_SIN_F32 | I::V_COS_F32 => {
            let x = f32::from_bits(bits as u32);
            if x.is_nan() { return bits | 0x0040_0000; }
            if x.is_infinite() { return 0xffc0_0000; }
            let r = x as f64 - (x as f64).round_ties_even();
            let result = if matches!(op, I::V_SIN_F32) {
                if x == 0.0 { x } else if r == 0.0 || r.abs() == 0.5 { 0.0 }
                else { (r * std::f64::consts::TAU).sin() as f32 }
            } else if r.abs() == 0.25 { 0.0 }
            else { (r * std::f64::consts::TAU).cos() as f32 };
            result.to_bits() as u64
        }
        I::V_CVT_F32_F16 => half::f16::from_bits(bits as u16).to_f32().to_bits() as u64,
        I::V_CVT_F16_F32 => half::f16::from_f32(f32::from_bits(bits as u32)).to_bits() as u64,
        I::V_FLOOR_F32 => f32::from_bits(bits as u32).floor().to_bits() as u64,
        I::V_FLOOR_F64 => f64::from_bits(bits).floor().to_bits(),
        I::V_CEIL_F32 => f32::from_bits(bits as u32).ceil().to_bits() as u64,
        I::V_TRUNC_F32 => f32::from_bits(bits as u32).trunc().to_bits() as u64,
        I::V_TRUNC_F64 => f64::from_bits(bits).trunc().to_bits(),
        I::V_RNDNE_F32 => f32::from_bits(bits as u32).round_ties_even().to_bits() as u64,
        I::V_RNDNE_F64 => f64::from_bits(bits).round_ties_even().to_bits(),
        I::V_FRACT_F64 => {
            let v = f64::from_bits(bits); let v = v - v.floor();
            let cap = f64::from_bits(0x3fef_ffff_ffff_ffff);
            (if v > cap { cap } else { v }).to_bits()
        }
        I::V_FREXP_MANT_F32 | I::V_FREXP_MANT_F64 | I::V_FREXP_EXP_I32_F32 | I::V_FREXP_EXP_I32_F64 => {
            let wide = matches!(op, I::V_FREXP_MANT_F64 | I::V_FREXP_EXP_I32_F64);
            let mant = matches!(op, I::V_FREXP_MANT_F32 | I::V_FREXP_MANT_F64);
            let (width, frac, bias) = if wide { (64, 52, 1023) } else { (32, 23, 127) };
            let sign = bits & (1 << (width - 1)); let fraction = bits & ((1 << frac) - 1);
            let exp = ((bits & !sign) >> frac) as i32;
            if exp == bias * 2 + 1 {
                if mant { bits | if fraction != 0 { 1 << (frac - 1) } else { 0 } } else { 0 }
            } else if exp == 0 && fraction == 0 { if mant { bits } else { 0 } }
            else if exp == 0 {
                let shift = frac + fraction.leading_zeros() - 63;
                if mant { sign | (((bias - 1) as u64) << frac) | ((fraction << shift) & ((1 << frac) - 1)) }
                else { (2 - bias - shift as i32) as u32 as u64 }
            } else if mant { sign | (((bias - 1) as u64) << frac) | fraction }
            else { (exp - bias + 1) as u32 as u64 }
        }
        _ => unreachable!(),
    }
}

pub(in crate::rdna_spmd) fn bvh(registry:&DialectRegistry)->TargetOp {registry.lookup(ID,"image_bvh64_intersect_ray").expect("missing BVH provider")}
