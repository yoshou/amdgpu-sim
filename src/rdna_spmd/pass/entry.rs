use super::super::analysis::{Constants, Preserved};
use super::super::ir::*;
use super::{Analyses, Pass};

fn branch_local_queries(f: &Func) -> std::collections::BTreeSet<ValueId> {
    use std::collections::BTreeSet;
    let mut critical = vec![false; f.types.len()];
    let seed = |critical: &mut Vec<bool>, v: ValueId| critical[v.0] = true;
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Effect { inputs, .. } => {
                    for &v in inputs {
                        seed(&mut critical, v)
                    }
                }
                Inst::Target { .. } => super::dce::operands(inst, |v| critical[v.0] = true),
                _ => {}
            }
        }
        match &block.term {
            Term::CondBr { cond, .. } => seed(&mut critical, *cond),
            Term::Ret(args) => {
                for &v in args {
                    seed(&mut critical, v)
                }
            }
            Term::Br(_) => {}
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for block in f.blocks.values() {
            for inst in &block.insts {
                let live = match inst {
                    Inst::Core { value, .. } | Inst::Packet { output: value, .. } => {
                        critical[value.0]
                    }
                    Inst::Effect { outputs, .. } | Inst::Target { outputs, .. } => {
                        outputs.iter().any(|o| critical[o.0 .0])
                    }
                };
                if live {
                    super::dce::operands(inst, |v| {
                        if !critical[v.0] {
                            critical[v.0] = true;
                            changed = true
                        }
                    });
                }
            }
            let edges: Vec<&Edge> = match &block.term {
                Term::Br(e) => vec![e],
                Term::CondBr { yes, no, .. } => vec![yes, no],
                Term::Ret(_) => Vec::new(),
            };
            for edge in edges {
                for (index, &arg) in edge.args.iter().enumerate() {
                    if critical[f.blocks[&edge.dst].params[index].0 .0] && !critical[arg.0] {
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
            if let Inst::Effect {
                provenance,
                op: EffectOp::Wave(WaveOp::Any),
                outputs,
                ..
            } = inst
            {
                if *provenance & (1u64 << 63) != 0 && !critical[outputs[0].0 .0] {
                    local.insert(outputs[0].0);
                }
            }
        }
    }
    local
}

pub(crate) fn packet_state(f: &mut Func, whole_wave: bool) -> usize {
    let all = whole_wave;
    let mut count = 0;
    loop {
        let local = if all {
            Default::default()
        } else {
            branch_local_queries(f)
        };
        let mut round = 0;
        for block in f.blocks.values_mut() {
            for inst in &mut block.insts {
                if let Inst::Effect {
                    provenance,
                    op,
                    inputs,
                    outputs,
                } = inst
                {
                    if *provenance & (1u64 << 63) != 0 {
                        let op = match *op {
                            EffectOp::Wave(WaveOp::Any) if all || local.contains(&outputs[0].0) => {
                                PacketOp::Any
                            }
                            EffectOp::Wave(WaveOp::Ballot) if all => PacketOp::Ballot,
                            _ => continue,
                        };
                        *inst = Inst::Packet {
                            op,
                            input: inputs[0],
                            output: outputs[0].0,
                        };
                        round += 1;
                    }
                }
            }
        }
        count += round;
        if round == 0 || all {
            break;
        }
    }
    count
}

pub(crate) struct PacketState;
impl Pass for PacketState {
    fn name(&self) -> &str {
        "packet_state"
    }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        packet_state(f, analyses.context().lanes >= 32) > 0
    }
    fn preserves(&self) -> Preserved {
        Preserved::of::<Constants>()
    }
}

