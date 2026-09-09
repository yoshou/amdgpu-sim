use super::super::ir::*;
use super::super::program::{Parameter, ParameterSource};
use super::super::dialect::DialectRegistry;
use super::{Analyses, Pass};

pub(crate) fn assume_dispatch_exec(parameter_inputs: &[Parameter], registry: &DialectRegistry, f: &mut Func) -> usize {
    let index = super::super::compiler::exec_index(parameter_inputs, registry);
    let exec = f.blocks[&f.entry].params[index].0;
    super::constant_queries(f, &[exec])
}

pub(crate) fn packet_state(f: &mut Func) -> usize {
    let mut count = 0;
    for block in f.blocks.values_mut() {
        for inst in &mut block.insts {
            if let Inst::Effect { provenance, op, inputs, outputs } = inst {
                if *provenance & (1u64 << 63) != 0 {
                    let op = match *op {
                        EffectOp::Wave(WaveOp::Any) => PacketOp::Any,
                        EffectOp::Wave(WaveOp::Ballot) => PacketOp::Ballot,
                        _ => continue,
                    };
                    *inst = Inst::Packet { op, input: inputs[0], output: outputs[0].0 };
                    count += 1;
                }
            }
        }
    }
    count
}

pub(crate) fn packet_entry(f: &Func, inputs: &[Parameter], aligned: bool) -> super::super::analysis::uniformity::Entry {
    let entry = &f.blocks[&f.entry];
    let mut uniform = Vec::new();
    let mut affine = Vec::new();
    let mut varying = Vec::new();
    for (input, &(id, _)) in inputs.iter().zip(&entry.params) {
        match input.source {
            ParameterSource::Vgpr(0) => if aligned { affine.push((id, 1, Some((0, 10)))) } else { varying.push(id) },
            ParameterSource::Vgpr(_) | ParameterSource::Sgpr(_) | ParameterSource::Scc => uniform.push(id),
            _ => {}
        }
    }
    super::super::analysis::uniformity::Entry { uniform, affine, varying }
}

pub(crate) struct PacketState;
impl Pass for PacketState {
    fn name(&self) -> &str { "packet_state" }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool { packet_state(f) > 0 }
}

pub(crate) struct AssumeDispatchExec<'a> { pub inputs: &'a [Parameter] }
impl Pass for AssumeDispatchExec<'_> {
    fn name(&self) -> &str { "assume_dispatch_exec" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        assume_dispatch_exec(self.inputs, analyses.context().registry, f) > 0
    }
}

pub(crate) struct DiscardReturn;
impl Pass for DiscardReturn {
    fn name(&self) -> &str { "discard_return" }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool {
        let mut count = 0;
        for block in f.blocks.values_mut() {
            if let Term::Ret(args) = &mut block.term {
                if !args.is_empty() { args.clear(); count += 1; }
            }
        }
        count > 0
    }
}
