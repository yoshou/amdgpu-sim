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

pub(super) struct ScalarBlockPlan<'a> {
    pub instructions: Vec<super::lift::Lowering<'a>>,
    pub active: Vec<bool>,
    pub f64_fresh: RegSet,
    pub sgpr_fresh: u128,
}

pub(super) struct ScalarPlan<'a> {
    pub program: &'a ScalarProgram,
    pub mode: ScalarMode,
    pub function: super::lift::function::Function,
    pub blocks: BTreeMap<usize, ScalarBlockPlan<'a>>,
}

impl<'a> ScalarPlan<'a> {
    #[cfg(test)]
    pub fn new(program: &'a ScalarProgram, mode: ScalarMode) -> Self { Self::with_registry(std::sync::Arc::new(super::dialect::DialectRegistry::rdna4()), program, mode) }
    pub fn with_registry(registry: std::sync::Arc<super::dialect::DialectRegistry>, program: &'a ScalarProgram, mode: ScalarMode) -> Self {
        let f64_fresh = super::freshness::analyze(program);
        let sgpr_fresh = super::freshness::analyze_sgpr(program);
        let mut blocks: BTreeMap<_, _> = program.blocks.iter().map(|(&pc, block)| {
            (pc, ScalarBlockPlan {
                instructions: block.body.iter().map(|inst| super::lift::instruction_with_registry(inst, &registry)).collect(),
                active: vec![false;block.body.len()],
                f64_fresh: f64_fresh[&pc],
                sgpr_fresh: sgpr_fresh[&pc],
            })
        }).collect();
        {
            let active=super::active::analyze_states(program);
            for (&pc,block) in &mut blocks {
                block.active=super::active::body_active_states(&program.blocks[&pc],active[&pc]);
            }
        }
        let lowerings = blocks.iter().map(|(&pc, block)| (pc, block.instructions.iter().collect())).collect();
        let preparation=super::lift::function::Preparation::Scalar {
            active:blocks.iter().flat_map(|(&pc,b)|b.active.iter().enumerate()
                .filter_map(move |(index,&active)|active.then_some((pc,index)))).collect(),
            dispatch:mode==ScalarMode::Whole,
        };
        let function=super::lift::function::Function::new(registry,program,&lowerings,None,preparation);
        Self { program, mode, blocks, function }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{InstFormat, SourceOperand, SOP1, VOP1};
    use super::super::ir::{ScalarBlock, Terminator};

    #[test]
    fn scalar_mask_blend_preserves_existing_whole_value_selection() {
        use crate::rdna_instructions::SOP2;
        use super::super::Compiler;
        let reg=SourceOperand::ScalarRegister;
        for change_mask in [false,true] {
            let mut body=vec![InstFormat::SOP2(SOP2 {op:I::S_AND_NOT1_B32,ssrc0:reg(2),ssrc1:reg(106),sdst:10})];
            if change_mask {body.push(InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,ssrc0:SourceOperand::IntegerConstant(0),sdst:106}));}
            body.extend([
                InstFormat::SOP2(SOP2 {op:I::S_AND_B32,ssrc0:reg(3),ssrc1:reg(106),sdst:11}),
                InstFormat::SOP2(SOP2 {op:I::S_OR_B32,ssrc0:reg(10),ssrc1:reg(11),sdst:12}),
            ]);
            let program=ScalarProgram {entry_pc:0,blocks:BTreeMap::from([(0,ScalarBlock {pc:0,body,term:Terminator::Return})])};
            let kernel=Compiler::default().compile_writeback(&program,256);
            for bit in [0,1] {
                let mut s=[0u32;128];let mut v=[0u32;256];
                s[2]=0x12345678;s[3]=0x87654321;s[106]=bit;s[126]=1;
                unsafe {kernel.run(s.as_mut_ptr(),v.as_mut_ptr(),0);}
                assert_eq!(s[10],0x12345678 & !bit);
                assert_eq!(s[11],if change_mask {0} else {0x87654321 & bit});
                assert_eq!(s[12],if change_mask||bit==0 {0x12345678} else {0x87654321});
            }
        }
    }

    #[test]
    fn existing_instruction_activity_is_applied_to_ssa_in_all_modes() {
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
