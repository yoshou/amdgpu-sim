use super::dialect::DialectRegistry;
use super::ir::{Func, Ty};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParameterSource {
    Vgpr(u32),
    Sgpr(u32),
    MaskBit(u32),
    Scc,
}

#[derive(Clone, Debug)]
pub(crate) struct Parameter {
    pub source: ParameterSource,
    pub ty: Ty,
}

#[derive(Clone)]
pub struct Program {
    pub(crate) registry: Arc<DialectRegistry>,
    pub(crate) ir: Func,
    pub(crate) parameter_inputs: Vec<Parameter>,
    pub(crate) revision: u64,
}

pub trait CompilationInput {
    fn to_ssa(&self) -> Program;
}
impl<T: CompilationInput + ?Sized> CompilationInput for &T {
    fn to_ssa(&self) -> Program {
        (**self).to_ssa()
    }
}
impl CompilationInput for Program {
    fn to_ssa(&self) -> Program {
        self.clone()
    }
}
