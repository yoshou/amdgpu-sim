//! Order the existing IR preparation, optimizations, analyses and emitters.
//!
//! The public free functions remain compatibility entrypoints. Constructed or
//! boundary-split ScalarPrograms go straight to analysis when compiled; passes
//! run during preparation only, retaining the existing invocation count/order.
//! This still uses the current register IR and execution engines.

use std::collections::BTreeMap;

use crate::rdna_instructions::InstFormat;
use crate::rdna_translator::RDNAProgram;

use super::boundary::BoundaryIo;
use super::emit::{CoopKernel, ScalarKernel};
use super::emit_vec::{CoopVecKernel, VecKernel};
use super::ir::{self, ScalarBlock, ScalarProgram};
use super::packet_plan::{cooperative_vgpr_count, PacketPlan};
use super::scalar_plan::{ScalarMode, ScalarPlan};

/// Coordinates the existing SPMD compilation stages.
pub struct Compiler { registry: std::sync::Arc<super::dialect::DialectRegistry> }
impl Default for Compiler {
    fn default() -> Self { Self { registry: std::sync::Arc::new(super::dialect::DialectRegistry::rdna4()) } }
}

impl Compiler {
    /// Prepare the current register IR from the decoded CFG.
    pub fn build_scalar_program(&self, program: &RDNAProgram) -> ScalarProgram {
        let mut blocks = BTreeMap::new();
        for (&pc, block) in program.blocks() {
            blocks.insert(pc, self.prepare_block(pc, block.insts().to_vec(), block.next_pcs()));
        }
        let mut program = ScalarProgram { entry_pc: program.entry_pc(), blocks };
        // Cross-block sqrt recognition requires the complete normalized CFG.
        super::mathcombine::fold_sqrt(&mut program);
        program
    }

    fn prepare_block(&self, pc: usize, mut insts: Vec<InstFormat>, next_pcs: &[usize]) -> ScalarBlock {
        // The existing local sqrt/DCE pass treats the last instruction as live.
        // It must run before terminator extraction and scheduling-no-op removal.
        super::combine::combine_block(&mut insts);
        let mut block = ir::lower_block(pc, &insts, next_pcs);
        // The current emitter computes DIV_FIXUP's quotient from its original
        // operands. Preserve this backend-specific optimization at this stage.
        super::combine::collapse_div_expansions(&mut block.body);
        block
    }

    /// Retains the existing de-SIMT register/mask adapter. Local lane-spill and
    /// readfirstlane lowering keep that adapter's contract during migration;
    /// general 32-lane effects must be split with `split_at_xlane` and executed
    /// by a wave/cooperative dispatcher.
    pub fn compile_program(&self, program: &ScalarProgram, num_vgprs: usize) -> ScalarKernel {
        let plan = ScalarPlan::with_registry(self.registry.clone(), program, ScalarMode::Whole);
        super::emit::compile_program(&plan,num_vgprs)
    }

    pub fn compile_program_vec(&self, program: &ScalarProgram, num_vgprs: usize, width: u32) -> VecKernel {
        let plan = PacketPlan::with_return_state(self.registry.clone(), program, width, None,false);
        super::emit_vec::compile_program(&plan,num_vgprs.max(256))
    }

    pub fn compile_cooperative_vec(&self, program: &ScalarProgram, num_vgprs: usize, width: u32) -> CoopVecKernel {
        let boundary = program.blocks.values().filter_map(|b| match &b.term {
            super::ir::Terminator::Yield {resume,action} => Some((*resume,action.io())), _=>None,
        }).collect();
        self.compile_packet_cooperative(program,num_vgprs,width,&boundary)
    }

    /// Compile a program already split at barriers for the existing scheduler.
    pub fn compile_cooperative(&self, program: &ScalarProgram, num_vgprs: usize) -> CoopKernel {
        let plan = ScalarPlan::with_registry(self.registry.clone(), program, ScalarMode::Cooperative);
        super::emit::compile_cooperative(&plan, num_vgprs)
    }

    pub(super) fn compile_writeback(&self, program: &ScalarProgram, num_vgprs: usize) -> ScalarKernel {
        let plan = ScalarPlan::with_registry(self.registry.clone(), program, ScalarMode::Writeback);
        super::emit::compile_program(&plan, num_vgprs)
    }

    pub(super) fn compile_packet_cooperative(
        &self,
        program: &ScalarProgram,
        num_vgprs: usize,
        width: u32,
        boundary: &BTreeMap<usize, BoundaryIo>,
    ) -> CoopVecKernel {
        assert!(matches!(width, 1 | 2 | 4 | 8 | 16));
        let num_vgprs = cooperative_vgpr_count(program, num_vgprs, boundary);
        let plan = PacketPlan::with_registry(self.registry.clone(), program, width, Some(boundary));
        super::emit_vec::compile_cooperative(&plan, num_vgprs)
    }
}

/// Prepare the current optimized Scalar IR; preserves the existing API.
pub fn build_scalar_program(program: &RDNAProgram) -> ScalarProgram {
    Compiler::default().build_scalar_program(program)
}

pub fn compile_program(program: &ScalarProgram, num_vgprs: usize) -> ScalarKernel {
    Compiler::default().compile_program(program, num_vgprs)
}

pub fn compile_program_vec(program: &ScalarProgram, num_vgprs: usize, width: u32) -> VecKernel {
    Compiler::default().compile_program_vec(program, num_vgprs, width)
}

pub fn compile_cooperative(program: &ScalarProgram, num_vgprs: usize) -> CoopKernel {
    Compiler::default().compile_cooperative(program, num_vgprs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{SourceOperand, SOPP, VOP1};
    use super::super::ir::{Cond, Terminator};

    fn mov(value: u32) -> InstFormat {
        InstFormat::VOP1(VOP1 {
            src0: SourceOperand::LiteralConstant(value), op: I::V_MOV_B32, vdst: 1,
        })
    }

    fn control(op: I) -> InstFormat {
        InstFormat::SOPP(SOPP { simm16: 0, op })
    }

    #[test]
    fn normalization_preserves_writes_and_compiler_applies_dce() {
        let insts = vec![mov(1), mov(2), control(I::S_NOP), control(I::S_ENDPGM)];
        let raw = ir::lower_block(4, &insts, &[]);
        assert_eq!(raw.body.len(), 2);
        assert!(matches!(raw.term, Terminator::Return));
        let prepared = Compiler::default().prepare_block(4, insts, &[]);
        assert_eq!(prepared.body.len(), 1);
        assert!(matches!(&prepared.body[0], InstFormat::VOP1(i)
            if matches!(i.src0, SourceOperand::LiteralConstant(2))));
        assert!(matches!(prepared.term, Terminator::Return));
    }

    #[test]
    fn preparation_preserves_fallthrough_instruction_and_branch_successors() {
        let fallthrough = Compiler::default().prepare_block(4, vec![mov(1), mov(2)], &[8]);
        assert_eq!(fallthrough.body.len(), 1);
        assert!(matches!(fallthrough.term, Terminator::Jump(8)));
        let branch = Compiler::default().prepare_block(4, vec![mov(2), control(I::S_CBRANCH_EXECZ)], &[8, 12]);
        assert_eq!(branch.body.len(), 1);
        assert!(matches!(branch.term, Terminator::Branch {
            cond: Cond::ExecZ, taken: 12, fallthrough: 8,
        }));
    }
}
