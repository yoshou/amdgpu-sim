use super::super::ir::*;
use super::super::program::{Parameter, ParameterSource};
use super::super::dialect::DialectRegistry;
use super::{Analyses, Pass};

pub(crate) fn assume_dispatch_exec(parameter_inputs: &[Parameter], registry: &DialectRegistry, f: &mut Func) -> usize {
    let index = super::super::compiler::exec_index(parameter_inputs, registry);
    let exec = f.blocks[&f.entry].params[index].0;
    super::constant_queries(f, &[exec])
}

fn branch_local_queries(f: &Func) -> std::collections::BTreeSet<ValueId> {
    use std::collections::BTreeSet;
    let mut critical = vec![false; f.types.len()];
    let seed = |critical: &mut Vec<bool>, v: ValueId| critical[v.0] = true;
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Effect { inputs, .. } => for &v in inputs { seed(&mut critical, v) },
                Inst::Target { .. } => super::dce::operands(inst, |v| critical[v.0] = true),
                _ => {}
            }
        }
        match &block.term {
            Term::CondBr { cond, .. } => seed(&mut critical, *cond),
            Term::Ret(args) => for &v in args { seed(&mut critical, v) },
            Term::Br(_) => {}
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for block in f.blocks.values() {
            for inst in &block.insts {
                let live = match inst {
                    Inst::Core { value, .. } | Inst::Packet { output: value, .. } => critical[value.0],
                    Inst::Effect { outputs, .. } | Inst::Target { outputs, .. } => outputs.iter().any(|o| critical[o.0.0]),
                };
                if live {
                    super::dce::operands(inst, |v| if !critical[v.0] { critical[v.0] = true; changed = true });
                }
            }
            let edges: Vec<&Edge> = match &block.term {
                Term::Br(e) => vec![e],
                Term::CondBr { yes, no, .. } => vec![yes, no],
                Term::Ret(_) => Vec::new(),
            };
            for edge in edges {
                for (index, &arg) in edge.args.iter().enumerate() {
                    if critical[f.blocks[&edge.dst].params[index].0.0] && !critical[arg.0] {
                        critical[arg.0] = true;
                        changed = true;
                    }
                }
            }
        }
    }
    let mut local = BTreeSet::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            if let Inst::Effect { provenance, op: EffectOp::Wave(WaveOp::Any), outputs, .. } = inst {
                if *provenance & (1u64 << 63) != 0 && !critical[outputs[0].0.0] { local.insert(outputs[0].0); }
            }
        }
    }
    local
}

pub(crate) fn packet_state(f: &mut Func, whole_wave: bool) -> usize {
    let scheduled = f.blocks.values().flat_map(|b| &b.insts)
        .any(|inst| matches!(inst, Inst::Effect { provenance, .. } if *provenance & crate::rdna_spmd::ir::SCHEDULED != 0));
    let all = !scheduled || whole_wave;
    let mut count = 0;
    loop {
        let local = if all { Default::default() } else { branch_local_queries(f) };
        let mut round = 0;
        for block in f.blocks.values_mut() {
            for inst in &mut block.insts {
                if let Inst::Effect { provenance, op, inputs, outputs } = inst {
                    if *provenance & (1u64 << 63) != 0 {
                        let op = match *op {
                            EffectOp::Wave(WaveOp::Any) if all || local.contains(&outputs[0].0) => PacketOp::Any,
                            EffectOp::Wave(WaveOp::Ballot) if all => PacketOp::Ballot,
                            _ => continue,
                        };
                        *inst = Inst::Packet { op, input: inputs[0], output: outputs[0].0 };
                        round += 1;
                    }
                }
            }
        }
        count += round;
        if round == 0 || all { break; }
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
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        packet_state(f, analyses.context().lanes >= 32) > 0
    }
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
