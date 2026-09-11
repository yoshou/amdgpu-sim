//! Owned typed SSA between decode/lift, preparation and codegen.
use std::collections::BTreeMap;
use std::sync::Arc;
use super::dialect::DialectRegistry;
use super::ir::{Func, Ty};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ParameterSource { Vgpr(u32), Sgpr(u32), MaskBit(u32), Scc }

#[derive(Clone, Debug)]
pub(crate) struct Parameter { pub source: ParameterSource, pub ty: Ty }

#[derive(Clone)]
pub(crate) struct LiftedFunction {
    pub registry: Arc<DialectRegistry>,
    pub ir: Func,
    pub parameter_inputs: Vec<Parameter>,
    pub revision: u64,
}

/// Typed SSA prepared for reuse across execution widths and schedules.
/// It owns no decoded instruction stream.
#[derive(Clone)]
pub struct Program { pub(crate) function: LiftedFunction }

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
impl Program {
    pub(super) fn vgpr_count(&self, declared: usize) -> usize {
        let needed=self.function.parameter_inputs.iter().filter_map(|i|match i.source {
            ParameterSource::Vgpr(r)=>Some(r as usize+1),_=>None,
        }).max().unwrap_or(1);
        declared.max(needed)
    }
    pub(super) fn schedule(self, accept:impl Fn(&super::ir::EffectOp,&[super::ir::ValueId])->bool) -> (Self,BTreeMap<usize,super::ir::EffectOp>) {
        use super::ir::{Inst, EffectOp};
        let mut out=self;
        let mut yields=BTreeMap::new();
        for block in out.function.ir.blocks.values_mut() {
            for inst in &mut block.insts {
                let Inst::Effect {provenance,op,inputs,..}=inst else {continue};
                if !matches!(op,EffectOp::Wave(_)|EffectOp::BarrierSignal {..}|EffectOp::BarrierWait) {continue;}
                if accept(op,inputs) {
                    *provenance|=crate::rdna_spmd::ir::SCHEDULED;
                    yields.insert(super::codegen::resume_key(*provenance),*op);
                }
            }
        }
        out.function.revision+=1;
        (out,yields)
    }
}

pub(crate) fn exchange(op: &super::ir::EffectOp, inputs: &[super::ir::ValueId], constants: &[Option<u64>]) -> bool {
    use super::ir::{EffectOp, WaveOp};
    let known = |v: super::ir::ValueId| constants[v.0].is_some();
    match op {
        EffectOp::Wave(WaveOp::ReadLane) => !(known(inputs[1]) && known(inputs[2])),
        EffectOp::Wave(WaveOp::WriteLane) => !(known(inputs[1]) && known(inputs[3])),
        EffectOp::Wave(_) => true,
        _ => false,
    }
}

pub(crate) fn split_at_effects(program: Program, whole_wave: bool) -> Program {
    use super::ir::{EffectOp, WaveOp};
    program.schedule(|op, _| match op {
        EffectOp::Wave(WaveOp::WriteLane) => false,
        EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane) => !whole_wave,
        EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => true,
        _ => false,
    }).0
}
