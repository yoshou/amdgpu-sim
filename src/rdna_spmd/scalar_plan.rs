//! Analysis input for scalar whole-kernel, writeback and cooperative emission.
//! Borrows the source IR so its instruction-indexed facts cannot outlive edits.

use std::collections::BTreeMap;

use super::ir::ScalarProgram;
use super::regtype::RegSet;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ScalarMode {
    Whole,
    Writeback,
    Cooperative,
}

pub(super) struct ScalarBlockPlan {
    pub active: Vec<bool>,
    pub f64_fresh: RegSet,
    pub sgpr_fresh: u128,
}

pub(super) struct ScalarPlan<'a> {
    pub program: &'a ScalarProgram,
    pub mode: ScalarMode,
    pub blocks: BTreeMap<usize, ScalarBlockPlan>,
}

impl<'a> ScalarPlan<'a> {
    pub fn new(program: &'a ScalarProgram, mode: ScalarMode) -> Self {
        let active = super::active::analyze_states(program);
        let f64_fresh = super::freshness::analyze(program);
        let sgpr_fresh = super::freshness::analyze_sgpr(program);
        let blocks = program.blocks.iter().map(|(&pc, block)| {
            (pc, ScalarBlockPlan {
                active: super::active::body_active_states(block, active[&pc]),
                f64_fresh: f64_fresh[&pc],
                sgpr_fresh: sgpr_fresh[&pc],
            })
        }).collect();
        Self { program, mode, blocks }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{InstFormat, SourceOperand, SOP1, VOP1};
    use super::super::ir::{ScalarBlock, Terminator};

    #[test]
    fn instruction_activity_is_sampled_before_exec_writes() {
        let program = ScalarProgram {
            entry_pc: 1,
            blocks: BTreeMap::from([(1, ScalarBlock {
                pc: 1,
                body: vec![
                    InstFormat::SOP1(SOP1 {
                        ssrc0: SourceOperand::IntegerConstant(0), op: I::S_MOV_B32, sdst: 126,
                    }),
                    InstFormat::VOP1(VOP1 {
                        src0: SourceOperand::IntegerConstant(1), op: I::V_MOV_B32, vdst: 1,
                    }),
                ],
                term: Terminator::Return,
            })]),
        };
        for mode in [ScalarMode::Whole, ScalarMode::Writeback, ScalarMode::Cooperative] {
            let plan = ScalarPlan::new(&program, mode);
            assert_eq!(plan.blocks[&1].active, [true, false]);
        }
    }
}
