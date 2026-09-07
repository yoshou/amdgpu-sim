//! Decode/lift once, prepare typed SSA, then select native representation and
//! emit LLVM. Prepared programs and CFG fragments retain their SSA graph.

use std::collections::BTreeMap;

#[cfg(test)]
use crate::rdna_instructions::InstFormat;
use crate::rdna_translator::RDNAProgram;

use super::boundary::BoundaryIo;
use super::emit::{CoopKernel, ScalarKernel};
use super::emit_vec::{CoopVecKernel, VecKernel};
#[cfg(test)]
use super::ir::{self, ScalarBlock, ScalarProgram};
use super::program::{Program, CompilationInput};
use super::packet_plan::PacketPlan;
use super::scalar_plan::{ScalarMode, ScalarPlan};

use super::lift::function::{Function, LiftedFunction, Preparation};
use super::lift::optimize::Position;
use super::passes::Driver;
use super::ir::typed::cfg::BlockId;

pub(super) fn input_passes(f: &mut LiftedFunction, mut positions: BTreeMap<usize, Vec<Position>>) {
    let driver = Driver::new();
    let pcs: Vec<usize> = f.blocks.keys().copied().collect();
    driver.run(f, "local_square_roots", |f| for &pc in &pcs { f.local_square_roots(pc, positions.get_mut(&pc).unwrap()); }).unwrap();
    let limit = 1 + f.state.sites.values().map(|s| s.len()).sum::<usize>();
    driver.fixpoint(f, "local_dead", limit, |f| for &pc in &pcs { f.local_dead(pc, positions.get_mut(&pc).unwrap()); }).unwrap();
    driver.run(f, "local_divisions", |f| for &pc in &pcs { f.local_divisions(pc); }).unwrap();
    driver.run(f, "cross_block_square_roots", |f| f.cross_block_square_roots()).unwrap();
    driver.run(f, "compact", |f| f.compact()).unwrap();
}

pub(super) fn query_passes(f: &mut LiftedFunction) {
    use super::lift::InputSource;
    use crate::rdna_instructions::SourceOperand;
    let driver = Driver::new();
    driver.run(f, "local_write_lane", |f| {
        for (&pc, block) in &mut f.blocks {
            if let Some(plan) = &mut block.yield_values {
                if let Some(end) = super::passes::local_write_lane(&mut f.ir, BlockId(pc), plan.core.end) {
                    plan.core.end = end;
                    plan.local = true;
                }
            }
        }
    }).unwrap();
    let entry = &f.ir.blocks[&f.ir.entry];
    let uniform_entry: Vec<_> = f.parameter_inputs.iter().zip(&entry.params).filter_map(|(input, parameter)|
        matches!(input.source, InputSource::Operand(SourceOperand::ScalarRegister(_)) | InputSource::Scc)
            .then_some(parameter.0)).collect();
    driver.run(f, "constant_queries", |f| super::passes::constant_queries(&mut f.ir, &[])).unwrap();
    driver.run(f, "uniform_queries", |f| super::passes::uniform_queries(&mut f.ir, &uniform_entry)).unwrap();
    driver.run(f, "constant_yield_arguments", |f| {
        use super::ir::typed::effect::{EffectOp, WaveOp};
        let constants = super::analysis::constants(&f.ir);
        for block in f.blocks.values_mut() {
            if let Some(plan) = &mut block.yield_values {
                for (index, id) in plan.arguments.iter().enumerate() {
                    if plan.layout.op == EffectOp::Wave(WaveOp::Wmma)
                        || plan.layout.op == EffectOp::Wave(WaveOp::WriteLane) && index == 2 { continue; }
                    if let Some(bits) = constants[id.0] {
                        plan.layout.arguments[index] = super::yield_values::Argument::Constant(bits as u32);
                    }
                }
            }
        }
    }).unwrap();
}

pub(super) fn packet_passes(f: &mut LiftedFunction) {
    let driver = Driver::new();
    let normal: BTreeMap<usize, Vec<bool>> = f.state.sites.iter().map(|(&pc, sites)| (pc, super::sqrt_idiom::analyze(sites).0)).collect();
    driver.run(f, "fold_normal_scales", |f| for (pc, normal) in &normal { f.fold_normal_scales(*pc, normal); }).unwrap();
}

pub(super) fn preparation_passes(f: &mut LiftedFunction, preparation: Preparation) -> (bool, Option<Vec<bool>>) {
    let driver = Driver::new();
    let mut scalar_live = None;
    let observable_return = match preparation {
        Preparation::Packet { inactive, observe_return } => {
            driver.run(f, "packet_state", |f| Function::packet_state(&mut f.ir)).unwrap();
            if !observe_return {
                driver.run(f, "elide_inactive_updates", |f| Function::elide_inactive_updates(&f.blocks, &mut f.ir, inactive)).unwrap();
                driver.run(f, "assume_dispatch_exec", |f| Function::assume_dispatch_exec(&f.parameter_inputs, &mut f.ir)).unwrap();
            }
            observe_return
        }
        Preparation::Scalar { active, dispatch } => {
            driver.run(f, "packet_state", |f| Function::packet_state(&mut f.ir)).unwrap();
            driver.run(f, "scalar_mask_selects", |f| Function::scalar_mask_selects(&f.blocks, &mut f.ir)).unwrap();
            if !dispatch {
                scalar_live = Some(super::analysis::live_values(&f.ir, f.blocks.values().flat_map(|b| &b.outgoing).copied()));
            }
            driver.run(f, "elide_active_updates", |f| Function::elide_active_updates(&f.blocks, &mut f.ir, active)).unwrap();
            if dispatch { driver.run(f, "assume_dispatch_exec", |f| Function::assume_dispatch_exec(&f.parameter_inputs, &mut f.ir)).unwrap(); }
            !dispatch
        }
        #[cfg(test)]
        Preparation::Inspect => true,
    };
    (observable_return, scalar_live)
}

/// Coordinates the existing SPMD compilation stages.
pub struct Compiler { registry: std::sync::Arc<super::dialect::DialectRegistry> }
impl Default for Compiler {
    fn default() -> Self { Self { registry: std::sync::Arc::new(super::dialect::DialectRegistry::rdna4()) } }
}

impl Compiler {
    /// Lift decoded input and run preparation passes entirely in typed SSA.
    pub fn build_scalar_program(&self, program: &RDNAProgram) -> Program {
        Program::decoded(program,self.registry.clone())
    }

    /// Compile lane-local execution. General 32-lane effects are split with
    /// `split_at_xlane` and executed by a wave/cooperative dispatcher.
    pub fn compile_program(&self, program: &impl CompilationInput, num_vgprs: usize) -> ScalarKernel {
        let plan = ScalarPlan::from_ssa(program.to_ssa().function, ScalarMode::Whole);
        super::emit::compile_program(&plan,num_vgprs)
    }

    pub fn compile_program_vec(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32) -> VecKernel {
        let plan = PacketPlan::from_ssa(program.to_ssa().function, width, None,false);
        super::emit_vec::compile_program(&plan,num_vgprs.max(256))
    }

    pub fn compile_cooperative_vec(&self, program: &impl CompilationInput, num_vgprs: usize, width: u32) -> CoopVecKernel {
        let program=program.to_ssa();
        let boundary=program.boundary_io();
        self.compile_packet_cooperative(&program,num_vgprs,width,&boundary)
    }

    /// Compile a program already split at barriers for the existing scheduler.
    pub fn compile_cooperative(&self, program: &impl CompilationInput, num_vgprs: usize) -> CoopKernel {
        let plan = ScalarPlan::from_ssa(program.to_ssa().function, ScalarMode::Cooperative);
        super::emit::compile_cooperative(&plan, num_vgprs)
    }

    pub(super) fn compile_writeback(&self, program: &impl CompilationInput, num_vgprs: usize) -> ScalarKernel {
        let plan = ScalarPlan::from_ssa(program.to_ssa().function, ScalarMode::Writeback);
        super::emit::compile_program(&plan, num_vgprs)
    }

    pub(super) fn compile_packet_cooperative(
        &self,
        program: &impl CompilationInput,
        num_vgprs: usize,
        width: u32,
        boundary: &BTreeMap<usize, BoundaryIo>,
    ) -> CoopVecKernel {
        assert!(matches!(width, 1 | 2 | 4 | 8 | 16));
        let program=program.to_ssa();
        let num_vgprs = program.vgpr_count(num_vgprs,boundary);
        let plan = PacketPlan::from_ssa(program.function, width, Some(boundary), true);
        super::emit_vec::compile_cooperative(&plan, num_vgprs)
    }
}

/// Prepare an owned typed SSA program from the decoded CFG.
pub fn build_scalar_program(program: &RDNAProgram) -> Program {
    Compiler::default().build_scalar_program(program)
}

pub fn compile_program(program: &impl CompilationInput, num_vgprs: usize) -> ScalarKernel {
    Compiler::default().compile_program(program, num_vgprs)
}

pub fn compile_program_vec(program: &impl CompilationInput, num_vgprs: usize, width: u32) -> VecKernel {
    Compiler::default().compile_program_vec(program, num_vgprs, width)
}

pub fn compile_cooperative(program: &impl CompilationInput, num_vgprs: usize) -> CoopKernel {
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

    fn prepared(insts:Vec<InstFormat>,next:&[usize])->Program {
        use super::super::lift::optimize::Position;
        let source=ir::lower_block(4,&insts,next);
        let mut blocks=BTreeMap::from([(4,source)]);
        for &pc in next {blocks.entry(pc).or_insert(ScalarBlock {pc,body:vec![],term:Terminator::Return});}
        let source=ScalarProgram {entry_pc:4,blocks};
        let mut program=source.to_ssa();
        let mut index=0;let mut positions=Vec::new();
        for inst in &insts {
            if matches!(inst,InstFormat::SOPP(_)) {
                let words=program.function.words_before(4,index);
                positions.push(Position::Control(super::super::lift::rewrite::observation(inst,&words,&words)));
            }else{positions.push(Position::Instruction(index));index+=1;}
        }
        let mut positions=BTreeMap::from([(4,positions)]);
        for &pc in next {positions.entry(pc).or_default();}
        input_passes(&mut program.function,positions);
        program.function.ir.clone().verify_with(&program.function.registry).unwrap();
        assert_eq!(source.blocks[&4].body.len(),insts.iter().filter(|i|!matches!(i,InstFormat::SOPP(_))).count());
        program
    }
    #[test]
    fn normalization_preserves_writes_and_compiler_applies_dce() {
        let insts=vec![mov(1),mov(2),control(I::S_NOP),control(I::S_ENDPGM)];
        assert_eq!(ir::lower_block(4,&insts,&[]).body.len(),2);
        let p=prepared(insts,&[]);
        assert_eq!(p.function.blocks[&4].instructions.len(),1);
        assert!(matches!(p.function.ir.blocks[&super::super::ir::typed::cfg::BlockId(4)].term,super::super::ir::typed::cfg::Term::Ret));
    }
    #[test]
    fn preparation_preserves_fallthrough_instruction_and_branch_successors() {
        use super::super::ir::typed::cfg::{BlockId,Term};
        let p=prepared(vec![mov(1),mov(2)],&[8]);
        assert_eq!(p.function.blocks[&4].instructions.len(),1);
        assert!(matches!(&p.function.ir.blocks[&BlockId(4)].term,Term::Br(e) if e.dst==BlockId(8)));
        let p=prepared(vec![mov(2),control(I::S_CBRANCH_EXECZ)],&[8,12]);
        assert_eq!(p.function.blocks[&4].instructions.len(),1);
        assert_eq!(p.function.state.conditions[&4],Cond::ExecZ);
        assert!(matches!(&p.function.ir.blocks[&BlockId(4)].term,Term::CondBr {yes,no,..} if yes.dst==BlockId(12)&&no.dst==BlockId(8)));
    }
}
