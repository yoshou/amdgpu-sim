use super::ir::DialectRegistry;
use super::ir::{Func, Parameter};
use std::sync::Arc;

#[derive(Clone)]
pub struct Program {
    pub(crate) registry: Arc<DialectRegistry>,
    pub(crate) ir: Func,
    pub(crate) parameter_inputs: Vec<Parameter>,
}
