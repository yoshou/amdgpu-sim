//! ISA §16.12 LDEXP: x * 2^n with one final rounding. AVX-512 SCALEF
//! retains the existing native vector implementation. Portable prescaling
//! prevents premature underflow: negative chunks are applied only below Emin
//! and retain a full significand's headroom before the final multiplication.
use super::*;
use llvm_sys::{LLVMIntPredicate::*, LLVMTypeKind};

/// Apply the existing normal-power proof before native code generation.
/// The replacement is ordinary typed SSA; no emitter mode flag is needed.
pub(in crate::rdna_spmd) fn fold_normal(expr: &crate::rdna_spmd::ir::typed::VerifiedExpr, registry: &DialectRegistry)
    -> crate::rdna_spmd::ir::typed::VerifiedExpr {
    use crate::rdna_spmd::ir::typed::*;
    use crate::rdna_spmd::dialect::Arguments;
    let old = expr.expr();
    let target = registry.lookup(ID, "ldexp.f64").unwrap();
    let mut new = Expr { params: old.params.clone(), insts: Vec::new(), results: Vec::new() };
    let mut values = (0..old.params.len()).map(ValueId).collect::<Vec<_>>();
    let mut next = old.params.len();
    for inst in &old.insts {
        let mut push = |ty, op| { let id = ValueId(next); next += 1; new.insts.push(ExprInst::Core(ty, op)); id };
        match inst {
            ExprInst::Core(ty, op) => { let value = push(*ty, op.map(|id| values[id.0])); values.push(value); }
            ExprInst::Target { op, args: Arguments::Binary([x, exp]), .. } if *op == target => {
                let x = values[x.0]; let exp = values[exp.0];
                let exp = push(Ty::I64, Op::Convert(Cvt::SExt, Ty::I64, exp));
                let bias = push(Ty::I64, Op::Const(Ty::I64, 1023));
                let exp = push(Ty::I64, Op::Int(IntOp::Add, exp, bias));
                let shift = push(Ty::I64, Op::Const(Ty::I64, 52));
                let bits = push(Ty::I64, Op::Int(IntOp::Shl, exp, shift));
                let power = push(Ty::F64, Op::Convert(Cvt::Bitcast, Ty::F64, bits));
                let result = push(Ty::F64, Op::Float(FloatOp::Mul, x, power));
                values.push(result);
            }
            ExprInst::Target { op, args, outputs } => {
                let args = args.map(|id| values[id.0]);
                values.extend((next..next + outputs.len()).map(ValueId)); next += outputs.len();
                new.insts.push(ExprInst::Target { op: *op, args, outputs: outputs.clone() });
            }
        }
    }
    new.results = old.results.iter().map(|id| values[id.0]).collect();
    new.verify_with(registry).expect("invalid normal ldexp rewrite")
}

pub(super) unsafe fn f32(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { scale(e, Ty::F32, a[0], a[1]) }
pub(super) unsafe fn f64(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef { scale(e, Ty::F64, a[0], a[1]) }

unsafe fn scale(e: &Emitter, ty: Ty, value: LLVMValueRef, exponent: LLVMValueRef) -> LLVMValueRef {
    #[cfg(target_arch = "x86_64")]
    if LLVMGetTypeKind(LLVMTypeOf(value)) == LLVMTypeKind::LLVMVectorTypeKind
        && std::arch::is_x86_feature_detected!("avx512f") {
        let w = LLVMGetVectorSize(LLVMTypeOf(value));
        let native = if ty == Ty::F32 { 16 } else { 8 };
        if w >= native || (w == native / 2 && std::arch::is_x86_feature_detected!("avx512vl")) {
            return native_scale(e, ty, value, exponent, w, native);
        }
    }
    portable(e, ty, value, exponent)
}

unsafe fn power(e: &Emitter, ty: Ty, exponent: LLVMValueRef) -> LLVMValueRef {
    let n = b"\0".as_ptr().cast();
    let (it, bias, fraction) = if ty == Ty::F32 { (Ty::I32, 127, 23) } else { (Ty::I64, 1023, 52) };
    let exponent = if it == Ty::I64 { LLVMBuildSExt(e.b, exponent, e.ty(it), n) } else { exponent };
    let biased = LLVMBuildAdd(e.b, exponent, e.constant(it, bias), n);
    let bits = LLVMBuildShl(e.b, biased, e.constant(it, fraction), n);
    LLVMBuildBitCast(e.b, bits, e.ty(ty), n)
}

unsafe fn portable(e: &Emitter, ty: Ty, mut value: LLVMValueRef, mut exp: LLVMValueRef) -> LLVMValueRef {
    let n = b"\0".as_ptr().cast();
    let (max, min, precision) = if ty == Ty::F32 { (127i32, -126i32, 24) } else { (1023, -1022, 53) };
    let k = |v: i32| e.constant(Ty::I32, v as u32 as u64);
    for _ in 0..2 {
        let high = LLVMBuildICmp(e.b, LLVMIntSGT, exp, k(max), n);
        let low = LLVMBuildICmp(e.b, LLVMIntSLT, exp, k(min), n);
        let step = LLVMBuildSelect(e.b, low, k(min + precision), k(0), n);
        let step = LLVMBuildSelect(e.b, high, k(max), step, n);
        exp = LLVMBuildSub(e.b, exp, step, n);
        value = LLVMBuildFMul(e.b, value, power(e, ty, step), n);
    }
    let high = LLVMBuildICmp(e.b, LLVMIntSGT, exp, k(max), n);
    exp = LLVMBuildSelect(e.b, high, k(max), exp, n);
    let low = LLVMBuildICmp(e.b, LLVMIntSLT, exp, k(min), n);
    exp = LLVMBuildSelect(e.b, low, k(min), exp, n);
    LLVMBuildFMul(e.b, value, power(e, ty, exp), n)
}

unsafe fn native_scale(e: &Emitter, ty: Ty, value: LLVMValueRef, exponent: LLVMValueRef, w: u32, native: u32) -> LLVMValueRef {
    let n = b"\0".as_ptr().cast();
    let lanes = w.min(native);
    let scalar = if ty == Ty::F32 { LLVMFloatTypeInContext(e.ctx) } else { LLVMDoubleTypeInContext(e.ctx) };
    let chunk_ty = LLVMVectorType(scalar, lanes);
    let ci = |v| LLVMConstInt(LLVMInt32TypeInContext(e.ctx), v, 0);
    let module = LLVMGetGlobalParent(LLVMGetBasicBlockParent(LLVMGetInsertBlock(e.b)));
    let name = std::ffi::CString::new(format!("llvm.x86.avx512.mask.scalef.{}.{}",
        if ty == Ty::F32 { "ps" } else { "pd" }, if lanes == native { 512 } else { 256 })).unwrap();
    let mut parts = Vec::new();
    for index in 0..w / lanes {
        let mut a = value; let mut b = exponent;
        if w != lanes {
            let mask = LLVMConstVector((0..lanes).map(|lane| ci((index * lanes + lane) as u64)).collect::<Vec<_>>().as_mut_ptr(), lanes);
            a = LLVMBuildShuffleVector(e.b, a, LLVMGetPoison(LLVMTypeOf(a)), mask, n);
            b = LLVMBuildShuffleVector(e.b, b, LLVMGetPoison(LLVMTypeOf(b)), mask, n);
        }
        let b = LLVMBuildSIToFP(e.b, b, chunk_ty, n);
        let mask = LLVMConstInt(LLVMIntTypeInContext(e.ctx, if lanes == 16 { 16 } else { 8 }), (1 << lanes) - 1, 0);
        let mut args = vec![a, b, a, mask];
        if lanes == native { args.push(ci(4)); }
        let mut types = args.iter().map(|&a| LLVMTypeOf(a)).collect::<Vec<_>>();
        let ft = LLVMFunctionType(chunk_ty, types.as_mut_ptr(), types.len() as u32, 0);
        let mut f = LLVMGetNamedFunction(module, name.as_ptr());
        if f.is_null() { f = LLVMAddFunction(module, name.as_ptr(), ft); }
        parts.push(LLVMBuildCall2(e.b, ft, f, args.as_mut_ptr(), args.len() as u32, n));
    }
    if parts.len() == 1 { parts[0] } else {
        assert_eq!(parts.len(), 2);
        let mask = LLVMConstVector((0..w).map(|lane| ci(lane as u64)).collect::<Vec<_>>().as_mut_ptr(), w);
        LLVMBuildShuffleVector(e.b, parts[0], parts[1], mask, n)
    }
}
