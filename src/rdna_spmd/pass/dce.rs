use super::super::analysis::{Analyses, Masking};
use super::super::ir::{ValueId, *};
use std::collections::BTreeMap;
use std::marker::PhantomData;

fn count_uses(f: &Func, uses: &mut [usize]) {
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Core { op, .. } => {
                    op.map(|v| {
                        uses[v.0] += 1;
                        v
                    });
                }
                Inst::Packet { input, .. } => uses[input.0] += 1,
                Inst::Target { args, .. } => {
                    for v in args.values() {
                        uses[v.0] += 1;
                    }
                }
                Inst::Effect { inputs, .. } => {
                    for v in inputs {
                        uses[v.0] += 1;
                    }
                }
            }
        }
        match &block.term {
            Term::Br(e) => {
                for v in &e.args {
                    uses[v.0] += 1;
                }
            }
            Term::CondBr { cond, yes, no } => {
                uses[cond.0] += 1;
                for v in yes.args.iter().chain(&no.args) {
                    uses[v.0] += 1;
                }
            }
            Term::Ret(args) => {
                for v in args {
                    uses[v.0] += 1;
                }
            }
        }
    }
}

pub(crate) struct Dce;
impl super::Pass for Dce {
    fn name(&self) -> &str {
        "dce"
    }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool {
        run(f) > 0
    }
}

pub(crate) struct DeadParams<M>(pub(crate) PhantomData<fn() -> M>);
impl<M: Masking> super::Pass for DeadParams<M> {
    fn name(&self) -> &str {
        "dead_params"
    }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool {
        dead_params::<M>(f) > 0
    }
}

pub(crate) fn run(f: &mut Func) -> usize {
    let mut removed = 0;
    loop {
        let mut uses = vec![0usize; f.types.len()];
        count_uses(f, &mut uses);
        let mut round = 0;
        for block in f.blocks.values_mut() {
            let before = block.insts.len();
            block.insts.retain(|inst| match inst {
                Inst::Core { value, .. } | Inst::Packet { output: value, .. } => uses[value.0] != 0,
                Inst::Target {
                    provenance: None,
                    outputs,
                    ..
                } => outputs.iter().any(|(v, _)| uses[v.0] != 0),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any | WaveOp::Ballot),
                    outputs,
                    ..
                } => outputs.iter().any(|(v, _)| uses[v.0] != 0),
                _ => true,
            });
            round += before - block.insts.len();
        }
        removed += round;
        if round == 0 {
            break;
        }
    }
    if removed != 0 {
        f.compact();
    }
    let _ = ValueId(0);
    removed
}

fn live_values<M: Masking>(f: &Func) -> Vec<bool> {
    let mut producer: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut parameter: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut incoming: BTreeMap<BlockId, Vec<&Edge>> = BTreeMap::new();
    let mut pending: Vec<ValueId> = Vec::new();
    for (&id, block) in &f.blocks {
        for (index, &(p, ty)) in block.params.iter().enumerate() {
            parameter[p.0] = Some((id, index));
            if M::positional(ty) || id == f.entry {
                pending.push(p);
            }
        }
        for (index, inst) in block.insts.iter().enumerate() {
            let observed = match inst {
                Inst::Effect { op, .. } => {
                    !matches!(op, EffectOp::Wave(WaveOp::Any | WaveOp::Ballot))
                }
                Inst::Target { provenance, .. } => provenance.is_some(),
                _ => false,
            };
            match inst {
                Inst::Core { value, .. } | Inst::Packet { output: value, .. } => {
                    producer[value.0] = Some((id, index))
                }
                Inst::Effect { outputs, .. } | Inst::Target { outputs, .. } => {
                    for &(v, _) in outputs {
                        producer[v.0] = Some((id, index));
                    }
                }
            }
            if observed {
                operands(inst, |v| pending.push(v));
            }
        }
        match &block.term {
            Term::CondBr { cond, .. } => pending.push(*cond),
            Term::Ret(args) => pending.extend(args.iter().copied()),
            Term::Br(_) => {}
        }
        for edge in block.term.edges() {
            incoming.entry(edge.dst).or_default().push(edge);
        }
    }
    let mut live = vec![false; f.types.len()];
    while let Some(value) = pending.pop() {
        if live[value.0] {
            continue;
        }
        live[value.0] = true;
        if let Some((block, index)) = producer[value.0] {
            operands(&f.blocks[&block].insts[index], |v| pending.push(v));
        }
        if let Some((block, index)) = parameter[value.0] {
            for edge in incoming.get(&block).into_iter().flatten() {
                pending.push(edge.args[index]);
            }
        }
    }
    live
}

pub(crate) fn operands(inst: &Inst, mut f: impl FnMut(ValueId)) {
    match inst {
        Inst::Core { op, .. } => {
            op.map(|v| {
                f(v);
                v
            });
        }
        Inst::Packet { input, .. } => f(*input),
        Inst::Effect { inputs, .. } => {
            for &v in inputs {
                f(v)
            }
        }
        Inst::Target { args, .. } => {
            for &v in args.values() {
                f(v)
            }
        }
    }
}

pub(crate) fn dead_params<M: Masking>(f: &mut Func) -> usize {
    let mut removed = 0;
    loop {
        let live = live_values::<M>(f);
        for block in f.blocks.values_mut() {
            let before = block.insts.len();
            block.insts.retain(|inst| match inst {
                Inst::Core { value, .. } | Inst::Packet { output: value, .. } => live[value.0],
                Inst::Effect { op, outputs, .. } => {
                    !matches!(op, EffectOp::Wave(WaveOp::Any | WaveOp::Ballot))
                        || outputs.iter().any(|(v, _)| live[v.0])
                }
                Inst::Target {
                    provenance,
                    outputs,
                    ..
                } => provenance.is_some() || outputs.iter().any(|(v, _)| live[v.0]),
            });
            removed += before - block.insts.len();
        }
        let mut dead: BTreeMap<BlockId, Vec<usize>> = BTreeMap::new();
        for (&id, block) in &f.blocks {
            if id == f.entry {
                continue;
            }
            let indices: Vec<usize> = block
                .params
                .iter()
                .enumerate()
                .filter(|(_, &(v, _))| !live[v.0])
                .map(|(i, _)| i)
                .collect();
            if !indices.is_empty() {
                dead.insert(id, indices);
            }
        }
        if dead.is_empty() {
            break;
        }
        for (id, indices) in &dead {
            let block = f.blocks.get_mut(id).unwrap();
            for &index in indices.iter().rev() {
                block.params.remove(index);
            }
            removed += indices.len();
        }
        for block in f.blocks.values_mut() {
            let edit = |edge: &mut Edge| {
                if let Some(indices) = dead.get(&edge.dst) {
                    for &index in indices.iter().rev() {
                        edge.args.remove(index);
                    }
                }
            };
            match &mut block.term {
                Term::Br(e) => edit(e),
                Term::CondBr { yes, no, .. } => {
                    edit(yes);
                    edit(no);
                }
                Term::Ret(_) => {}
            }
        }
    }
    if removed != 0 {
        f.compact();
    }
    removed
}
