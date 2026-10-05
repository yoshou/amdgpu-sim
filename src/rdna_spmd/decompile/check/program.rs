use super::super::hazard::Hazards;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeMap;

pub(super) struct Program<'a> {
    pub(super) f: &'a Func,
    pub(super) facts: &'a Facts,
    pub(super) inputs: &'a [Parameter],
    pub(super) exec_index: Option<usize>,
    pub(super) loops: &'a Loops,
    pub(super) hazards: &'a Hazards,
    pub(super) meetings: BTreeMap<(BlockId, usize), usize>,
    pub(super) rank: BTreeMap<BlockId, usize>,
}

impl<'a> Program<'a> {
    pub(super) fn new(
        f: &'a Func,
        facts: &'a Facts,
        inputs: &'a [Parameter],
        exec_index: Option<usize>,
        loops: &'a Loops,
        hazards: &'a Hazards,
    ) -> Self {
        Self {
            f,
            facts,
            inputs,
            exec_index,
            loops,
            hazards,
            meetings: hazards
                .meetings
                .iter()
                .enumerate()
                .map(|(i, &position)| (position, i))
                .collect(),
            rank: facts
                .order
                .iter()
                .enumerate()
                .map(|(r, &b)| (b, r))
                .collect(),
        }
    }

    pub(super) fn partnered(&self, at: (BlockId, usize)) -> bool {
        let hazards = self.hazards;
        hazards
            .accesses
            .iter()
            .position(|a| (a.block, a.index) == at)
            .is_some_and(|i| hazards.together.iter().chain(&hazards.apart).any(|&(p, q)| p == i || q == i))
    }
}
