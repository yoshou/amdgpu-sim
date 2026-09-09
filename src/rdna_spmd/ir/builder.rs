use super::{Op, Ty, ValueId};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Expr {
    pub params: Vec<Ty>,
    pub insts: Vec<ExprInst>,
    pub results: Vec<ValueId>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ExprInst {
    Core(Ty, Op),
    Target { op: crate::rdna_spmd::dialect::TargetOp,
        args: crate::rdna_spmd::dialect::Arguments, outputs: Vec<Ty> },
}
impl From<(Ty, Op)> for ExprInst { fn from((ty, op): (Ty, Op)) -> Self { Self::Core(ty, op) } }
impl ExprInst {
    pub fn result_types(&self) -> &[Ty] {
        match self { Self::Core(ty, _) => std::slice::from_ref(ty), Self::Target { outputs, .. } => outputs }
    }
}
