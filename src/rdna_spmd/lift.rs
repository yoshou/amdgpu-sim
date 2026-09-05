//! Incremental lift into typed SSA. Legacy instructions remain explicit.
//!
//! Initially only the common VOP2 i32 ALU subset is lifted. Register reads,
//! EXEC predication, writes and f64 shadow invalidation remain in the adapter;
//! in particular this does not claim to fix the legacy EXEC/VCC word semantics.
//! The adapter uses existing LLVM values/allocas, not a new runtime register
//! buffer or per-lane helper call at each instruction.

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, VOP2};
use super::ir::typed::{Expr, IntOp, IntPred, Op, Ty, ValueId, VerifiedExpr};

pub(super) enum Lowering<'a> {
    TypedAlu { source: &'a VOP2, expr: VerifiedExpr },
    Legacy(&'a InstFormat),
}

pub(super) fn instruction(inst: &InstFormat) -> Lowering<'_> {
    let source = match inst {
        InstFormat::VOP2(i) => i,
        _ => return Lowering::Legacy(inst),
    };
    let a = ValueId(0);
    let b = ValueId(1);
    let op = match source.op {
        I::V_ADD_NC_U32 => Op::Int(IntOp::Add, a, b),
        I::V_SUB_NC_U32 => Op::Int(IntOp::Sub, a, b),
        I::V_SUBREV_NC_U32 => Op::Int(IntOp::Sub, b, a),
        I::V_AND_B32 => Op::Int(IntOp::And, a, b),
        I::V_OR_B32 => Op::Int(IntOp::Or, a, b),
        I::V_XOR_B32 => Op::Int(IntOp::Xor, a, b),
        I::V_LSHLREV_B32 => Op::Int(IntOp::Shl, b, a),
        I::V_LSHRREV_B32 => Op::Int(IntOp::LShr, b, a),
        I::V_MIN_U32 => Op::Cmp(IntPred::Ult, a, b),
        I::V_MAX_U32 => Op::Cmp(IntPred::Ugt, a, b),
        _ => return Lowering::Legacy(inst),
    };
    let insts = if matches!(op, Op::Cmp(..)) {
        vec![(Ty::I1, op), (Ty::I32, Op::Select(ValueId(2), a, b))]
    } else {
        vec![(Ty::I32, op)]
    };
    let result = ValueId(1 + insts.len());
    let expr = Expr { params: vec![Ty::I32, Ty::I32], insts, result }
        .verify().expect("invalid VOP2 lift");
    Lowering::TypedAlu { source, expr }
}

#[cfg(test)]
mod tests;
