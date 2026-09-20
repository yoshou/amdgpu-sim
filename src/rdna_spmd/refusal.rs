//! Why a program cannot be read as a lane program or lowered into packets: it
//! then runs as a wave program.

use crate::rdna_spmd::ir::BlockId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Refusal {
    pub block: BlockId,
    pub index: Option<usize>,
    pub reason: &'static str,
}

impl Refusal {
    pub(crate) fn at(block: BlockId, index: Option<usize>, reason: &'static str) -> Self {
        Self {
            block,
            index,
            reason,
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, out: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.index {
            Some(index) => write!(out, "b{}:{}: {}", self.block.0, index, self.reason),
            None => write!(out, "b{}: {}", self.block.0, self.reason),
        }
    }
}
