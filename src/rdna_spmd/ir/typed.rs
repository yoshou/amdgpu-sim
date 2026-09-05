//! The first, pure-expression subset of the typed SSA IR.
//!
//! Values have no ISA register identity or packet width. Parameters bridge to
//! the existing register IR during migration; CFG, memory effects and wave
//! operations still belong to that IR. Expressions are emitted in place, never
//! scheduled across legacy instructions. This is not yet a whole-kernel Func.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ty { I1, I32 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ValueId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntOp { Add, Sub, And, Or, Xor, Shl, LShr }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntPred { Ult, Ugt }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Int(IntOp, ValueId, ValueId),
    Cmp(IntPred, ValueId, ValueId),
    Select(ValueId, ValueId, ValueId),
}

impl Op {
    fn result_type(self, types: &[Ty]) -> Result<Ty, &'static str> {
        let ty = |v: ValueId| types.get(v.0).copied().ok_or("undefined or non-dominating value");
        match self {
            Self::Int(op, a, b) => {
                let a = ty(a)?;
                if a != ty(b)? { return Err("integer operand type mismatch"); }
                if a == Ty::I1 && !matches!(op, IntOp::And | IntOp::Or | IntOp::Xor) {
                    return Err("arithmetic requires i32");
                }
                Ok(a)
            }
            Self::Cmp(_, a, b) => {
                if ty(a)? != Ty::I32 || ty(b)? != Ty::I32 { return Err("comparison requires i32"); }
                Ok(Ty::I1)
            }
            Self::Select(c, a, b) => {
                if ty(c)? != Ty::I1 { return Err("select condition requires i1"); }
                let a = ty(a)?;
                if a != ty(b)? { return Err("select operand type mismatch"); }
                Ok(a)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Expr {
    pub params: Vec<Ty>,
    pub insts: Vec<(Ty, Op)>,
    pub result: ValueId,
}

/// Only verified, immutable SSA can reach code generation.
pub(crate) struct VerifiedExpr(Expr);

impl Expr {
    pub fn verify(self) -> Result<VerifiedExpr, &'static str> {
        let mut types = self.params.clone();
        for &(declared, op) in &self.insts {
            if op.result_type(&types)? != declared { return Err("result type mismatch"); }
            types.push(declared);
        }
        if self.result.0 >= types.len() { return Err("undefined result"); }
        Ok(VerifiedExpr(self))
    }
}

impl VerifiedExpr {
    pub fn expr(&self) -> &Expr { &self.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verifies_multistep_ssa_with_boolean_result_and_select() {
        Expr { params: vec![Ty::I32, Ty::I32], insts: vec![
            (Ty::I1, Op::Cmp(IntPred::Ult, ValueId(0), ValueId(1))),
            (Ty::I32, Op::Select(ValueId(2), ValueId(0), ValueId(1))),
        ], result: ValueId(3) }.verify().unwrap();
    }

    #[test]
    fn rejects_forward_self_and_missing_references() {
        for value in [1, 2, 99] {
            assert!(Expr { params: vec![Ty::I32], insts: vec![
                (Ty::I32, Op::Int(IntOp::Add, ValueId(0), ValueId(value))),
            ], result: ValueId(1) }.verify().is_err());
        }
        assert!(Expr { params: vec![], insts: vec![], result: ValueId(0) }.verify().is_err());
    }

    #[test]
    fn rejects_invalid_operand_and_result_types() {
        for (ty, op) in [
            (Ty::I32, Op::Int(IntOp::Add, ValueId(0), ValueId(1))),
            (Ty::I1, Op::Int(IntOp::Add, ValueId(1), ValueId(1))),
            (Ty::I32, Op::Cmp(IntPred::Ult, ValueId(0), ValueId(0))),
            (Ty::I32, Op::Select(ValueId(0), ValueId(0), ValueId(0))),
            (Ty::I32, Op::Select(ValueId(1), ValueId(0), ValueId(1))),
        ] {
            assert!(Expr { params: vec![Ty::I32, Ty::I1], insts: vec![(ty, op)], result: ValueId(2) }.verify().is_err());
        }
    }
}
