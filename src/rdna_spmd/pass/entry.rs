use super::super::analysis::{Constants, DispatchConstants, Preserved};
use super::super::ir::{*, Env, IntOp, IntPred, Op, Ty};
use super::{Analyses, Pass};

pub(crate) fn constant_queries(f: &mut Func, facts: &[Option<u64>]) -> usize {
    let mut count = 0;
    for block in f.blocks.values_mut() {
        for inst in &mut block.insts {
            let query = match inst {
                Inst::Packet { op: PacketOp::Any, input, output } => Some((*input, *output)),
                Inst::Effect { op: EffectOp::Wave(WaveOp::Any), inputs, outputs, .. } => Some((inputs[0], outputs[0].0)),
                _ => None,
            };
            if let Some((input, output)) = query {
                if let Some(bits) = facts[input.0] {
                    *inst = Inst::Core { value: output, ty: Ty::I1, op: Op::Const(Ty::I1, bits) };
                    count += 1;
                }
            }
        }
    }
    count
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
        .any(|inst| matches!(inst, Inst::Effect { provenance, .. } if *provenance & SCHEDULED != 0));
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

/// WriteLane requires wave-uniform value and selector. Lane i's result is
/// exactly `i == (selector & 31) ? value : old[i]`, so no value crosses a
/// packet boundary and the rendezvous can be removed.
pub(crate) fn local_write_lanes(f: &mut Func) -> usize {
    let scheduled_reads = f.blocks.values().flat_map(|b| &b.insts).any(|inst| matches!(inst,
        Inst::Effect { op: EffectOp::Wave(WaveOp::ReadLane), provenance, .. } if provenance & SCHEDULED != 0));
    if !scheduled_reads { return 0; }
    let mut count = 0;
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for id in ids {
        let old = std::mem::take(&mut f.blocks.get_mut(&id).unwrap().insts);
        let mut insts = Vec::with_capacity(old.len());
        for inst in old {
            match inst {
                Inst::Effect { op: EffectOp::Wave(WaveOp::WriteLane), inputs, outputs, provenance } if provenance & (SCHEDULED | (1 << 63)) == 0 => {
                    let (result, ty) = outputs[0];
                    assert_eq!(ty, Ty::I32);
                    let lane = f.value(Ty::I32); let mask = f.value(Ty::I32);
                    let selector = f.value(Ty::I32); let selected = f.value(Ty::I1);
                    insts.push(Inst::Core { value: lane, ty: Ty::I32, op: Op::Env(Env::LaneId) });
                    insts.push(Inst::Core { value: mask, ty: Ty::I32, op: Op::Const(Ty::I32, 31) });
                    insts.push(Inst::Core { value: selector, ty: Ty::I32, op: Op::Int(IntOp::And, inputs[1], mask) });
                    insts.push(Inst::Core { value: selected, ty: Ty::I1, op: Op::Cmp(IntPred::Eq, lane, selector) });
                    insts.push(Inst::Core { value: result, ty: Ty::I32, op: Op::Select(selected, inputs[0], inputs[2]) });
                    count += 1;
                }
                other => insts.push(other),
            }
        }
        f.blocks.get_mut(&id).unwrap().insts = insts;
    }
    count
}

pub(crate) struct PacketState;
impl Pass for PacketState {
    fn name(&self) -> &str { "packet_state" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        packet_state(f, analyses.context().lanes >= 32) > 0
    }
    fn preserves(&self) -> Preserved { Preserved::of::<Constants>().and::<DispatchConstants>() }
}

pub(crate) struct AssumeDispatchExec;
impl Pass for AssumeDispatchExec {
    fn name(&self) -> &str { "assume_dispatch_exec" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let facts = analyses.get::<DispatchConstants>(f);
        constant_queries(f, &facts) > 0
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

pub(crate) struct LocalWriteLanes;
impl Pass for LocalWriteLanes {
    fn name(&self) -> &str { "local_write_lanes" }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool { local_write_lanes(f) > 0 }
}
