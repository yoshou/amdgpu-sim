//! Owned typed SSA between decode/lift, preparation and codegen.
use std::collections::BTreeMap;
use std::sync::Arc;
use super::lift::function::LiftedFunction;
use super::decode::ScalarProgram;
use super::dialect::DialectRegistry;

/// Typed SSA prepared for reuse across execution widths and schedules.
/// It owns no decoded instruction stream.
#[derive(Clone)]
pub struct Program { pub(super) function: LiftedFunction }

/// Inputs accepted by the compiler. Decoded fixtures are lifted at this
/// boundary; an already prepared program retains its SSA graph.
pub trait CompilationInput {
    fn to_ssa(&self) -> Program;
}
impl<T: CompilationInput + ?Sized> CompilationInput for &T {
    fn to_ssa(&self) -> Program { (**self).to_ssa() }
}
impl CompilationInput for Program {
    fn to_ssa(&self) -> Program { self.clone() }
}
impl CompilationInput for ScalarProgram {
    fn to_ssa(&self) -> Program { Program::lift(self, Arc::new(DialectRegistry::rdna4())) }
}
impl Program {
    pub(super) fn lift(source: &ScalarProgram, registry: Arc<DialectRegistry>) -> Self {
        use super::lift::{Lowering,wave::YieldAction};
        use super::ir::EffectOp;
        use crate::{instructions::I,rdna_instructions::{InstFormat,SourceOperand}};
        // Resolve the combined barrier while decoding, before SSA exists. Its
        // signal/wait both retain the original source proof barrier and ID.
        let mut normalized=source.clone();
        let lowered:BTreeMap<_,Vec<_>>=normalized.blocks.iter_mut().map(|(&pc,b)| {
            let mut body=Vec::new();let mut lowerings=Vec::new();
            for inst in &b.body {
                if matches!(inst,InstFormat::SOPP(i) if matches!(i.op,I::S_BARRIER)) {
                    for op in [EffectOp::BarrierSignal {is_first:false},EffectOp::BarrierWait] {
                        body.push(inst.clone());
                        lowerings.push(Lowering::Wave(YieldAction::new(op,
                            vec![super::lift::wave::Operand::Source(SourceOperand::LiteralConstant(u32::MAX))],vec![])));
                    }
                }else{body.push(inst.clone());lowerings.push(super::lift::instruction_with_registry(inst,&registry));}
            }
            b.body=body;(pc,lowerings)
        }).collect();
        let refs=lowered.iter().map(|(&pc,b)|(pc,b.iter().collect())).collect();
        Self {function:super::lift::function::lift(registry,&normalized,&refs)}
    }

    pub(super) fn decoded(source: &super::decode::Decoded, registry: Arc<DialectRegistry>) -> Self {
        let normalized = ScalarProgram { entry_pc:source.entry_pc,blocks:source.blocks.iter().map(|(&pc,b)|
            (pc,super::decode::lower_block(pc,&b.insts,&b.next_pcs))).collect() };
        let mut program = Self::lift(&normalized,registry);
        super::compiler::input_passes_ir(&mut program.function);
        program
    }
}

impl Program {
    pub(super) fn vgpr_count(&self, declared: usize) -> usize {
        use crate::rdna_instructions::SourceOperand;
        let needed=self.function.parameter_inputs.iter().filter_map(|i|match i.source {
            super::lift::InputSource::Operand(SourceOperand::VectorRegister(r))=>Some(r as usize+1),_=>None,
        }).max().unwrap_or(1);
        declared.max(needed)
    }
    pub(super) fn schedule(&self, accept:impl Fn(&super::ir::EffectOp)->bool) -> (Self,BTreeMap<usize,super::ir::EffectOp>) {
        use super::ir::{Inst, EffectOp};
        let mut out=self.clone();
        let mut yields=BTreeMap::new();
        for block in out.function.ir.blocks.values_mut() {
            for inst in &mut block.insts {
                let Inst::Effect {provenance,op,..}=inst else {continue};
                if *provenance&(1<<63)!=0||!matches!(op,EffectOp::Wave(_)|EffectOp::BarrierSignal {..}|EffectOp::BarrierWait) {continue;}
                if accept(op) {
                    *provenance|=super::lift::wave::SCHEDULED;
                    yields.insert(super::codegen::resume_key(*provenance),*op);
                }
            }
        }
        out.function.revision+=1;
        (out,yields)
    }
}

pub(crate) fn split_at_barriers(program: &impl CompilationInput) -> Program {
    program.to_ssa().schedule(|op| !matches!(op, super::ir::EffectOp::Wave(_))).0
}
