//! Width-independent LLVM emission for verified core SSA expressions.
//! No RDNA opcodes, register accesses, instruction recognition or lane loops.

use llvm_sys as llvm;
use llvm::prelude::{LLVMBuilderRef, LLVMValueRef};
use super::ir::typed::{IntOp, IntPred, Op, Ty, VerifiedExpr};

/// Parameters already have the scalar or packet LLVM types chosen by the
/// existing emitter. LLVM arithmetic uses the same rules for both shapes.
pub(super) unsafe fn emit(
    b: LLVMBuilderRef, expr: &VerifiedExpr, params: &[LLVMValueRef],
) -> LLVMValueRef {
    let expr = expr.expr();
    assert_eq!(params.len(), expr.params.len());
    let mut shape = None;
    for (&value, ty) in params.iter().zip(&expr.params) {
        let llvm_ty = llvm::core::LLVMTypeOf(value);
        let (element, lanes) = if llvm::core::LLVMGetTypeKind(llvm_ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
            (llvm::core::LLVMGetElementType(llvm_ty), Some(llvm::core::LLVMGetVectorSize(llvm_ty)))
        } else { (llvm_ty, None) };
        assert_eq!(llvm::core::LLVMGetTypeKind(element), llvm::LLVMTypeKind::LLVMIntegerTypeKind);
        assert_eq!(llvm::core::LLVMGetIntTypeWidth(element), match ty { Ty::I1 => 1, Ty::I32 => 32 });
        if let Some(previous) = shape { assert_eq!(previous, lanes, "mixed LLVM packet shapes"); }
        shape = Some(lanes);
    }
    let mut values = Vec::with_capacity(params.len() + expr.insts.len());
    values.extend_from_slice(params);
    let n = b"\0".as_ptr().cast();
    for &(_, op) in &expr.insts {
        let v = match op {
            Op::Int(op, a, c) => {
                let (a, c) = (values[a.0], values[c.0]);
                match op {
                    IntOp::Add => llvm::core::LLVMBuildAdd(b, a, c, n),
                    IntOp::Sub => llvm::core::LLVMBuildSub(b, a, c, n),
                    IntOp::And => llvm::core::LLVMBuildAnd(b, a, c, n),
                    IntOp::Or => llvm::core::LLVMBuildOr(b, a, c, n),
                    IntOp::Xor => llvm::core::LLVMBuildXor(b, a, c, n),
                    IntOp::Shl | IntOp::LShr => {
                        // Core i32 shifts reduce the amount modulo 32 before
                        // LLVM emission (LLVM's oversized shifts are poison).
                        let ty = llvm::core::LLVMTypeOf(c);
                        let mask = if llvm::core::LLVMGetTypeKind(ty) == llvm::LLVMTypeKind::LLVMVectorTypeKind {
                            let k = llvm::core::LLVMConstInt(llvm::core::LLVMGetElementType(ty), 31, 0);
                            let mut lanes = vec![k; llvm::core::LLVMGetVectorSize(ty) as usize];
                            llvm::core::LLVMConstVector(lanes.as_mut_ptr(), lanes.len() as u32)
                        } else {
                            llvm::core::LLVMConstInt(ty, 31, 0)
                        };
                        let c = llvm::core::LLVMBuildAnd(b, c, mask, n);
                        if op == IntOp::Shl { llvm::core::LLVMBuildShl(b, a, c, n) }
                        else { llvm::core::LLVMBuildLShr(b, a, c, n) }
                    }
                }
            }
            Op::Cmp(pred, a, c) => llvm::core::LLVMBuildICmp(b, match pred {
                IntPred::Ult => llvm::LLVMIntPredicate::LLVMIntULT,
                IntPred::Ugt => llvm::LLVMIntPredicate::LLVMIntUGT,
            }, values[a.0], values[c.0], n),
            Op::Select(c, a, d) => llvm::core::LLVMBuildSelect(b, values[c.0], values[a.0], values[d.0], n),
        };
        values.push(v);
    }
    values[expr.result.0]
}
