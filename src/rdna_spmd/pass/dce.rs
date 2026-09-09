use super::super::ir::{*, Cvt, Op, Ty, ValueId};
use super::super::analysis::masks::Masks;
use std::collections::{BTreeMap, BTreeSet};

fn count_uses(f: &Func, uses: &mut [usize]) {
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Core { op, .. } => { op.map(|v| { uses[v.0] += 1; v }); }
                Inst::Packet { input, .. } => uses[input.0] += 1,
                Inst::Target { args, .. } => for v in args.values() { uses[v.0] += 1; },
                Inst::Effect { inputs, .. } => for v in inputs { uses[v.0] += 1; },
            }
        }
        match &block.term {
            Term::Br(e) => for v in &e.args { uses[v.0] += 1; },
            Term::CondBr { cond, yes, no } => { uses[cond.0] += 1; for v in yes.args.iter().chain(&no.args) { uses[v.0] += 1; } }
            Term::Ret(args) => for v in args { uses[v.0] += 1; },
        }
    }
}

enum Demand { Active, Overwritten(ValueId) }

fn needed(f: &Func, masks: &Masks, exec_index: usize) -> Vec<bool> {
    let entry = &f.blocks[&f.entry];
    let k = entry.params[..exec_index].iter().filter(|p| p.1 == Ty::I1).count();
    let exec_of = |block: &Block| block.params.iter().filter(|p| p.1 == Ty::I1).nth(k).map(|p| p.0);
    let mut params: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut producers: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut incoming: BTreeMap<BlockId, Vec<&Edge>> = BTreeMap::new();
    let mut pending: Vec<(ValueId, Demand)> = Vec::new();
    for (&id, block) in &f.blocks {
        for (index, &(p, _)) in block.params.iter().enumerate() { params[p.0] = Some((id, index)); }
        for (index, inst) in block.insts.iter().enumerate() {
            match inst {
                Inst::Core { value, .. } | Inst::Packet { output: value, .. } => producers[value.0] = Some((id, index)),
                Inst::Effect { inputs, outputs, .. } => { pending.extend(inputs.iter().map(|&v| (v, Demand::Active))); for &(v, _) in outputs { producers[v.0] = Some((id, index)); } }
                Inst::Target { provenance, args, outputs, .. } => {
                    if provenance.is_some() { pending.extend(args.values().iter().map(|&v| (v, Demand::Active))); }
                    for &(v, _) in outputs { producers[v.0] = Some((id, index)); }
                }
            }
        }
        match &block.term {
            Term::CondBr { cond, .. } => pending.push((*cond, Demand::Active)),
            Term::Ret(args) => pending.extend(args.iter().map(|&v| (v, Demand::Active))),
            Term::Br(_) => {}
        }
        for edge in block.term.edges() { incoming.entry(edge.dst).or_default().push(edge); }
    }
    let mut needed = vec![false; f.types.len()];
    let mut overwritten: BTreeSet<(ValueId, ValueId)> = BTreeSet::new();
    while let Some((v, demand)) = pending.pop() {
        if needed[v.0] { continue; }
        if let Demand::Overwritten(exec) = demand {
            if !overwritten.insert((v, exec)) { continue; }
            match producers[v.0].map(|(block, index)| &f.blocks[&block].insts[index]) {
                Some(Inst::Core { op: Op::Select(_, _, old), .. }) if masks.predicated[v.0].is_some_and(|(_, own)| own == exec) => pending.push((*old, Demand::Overwritten(exec))),
                Some(Inst::Core { op: Op::UnpackLo(x) | Op::UnpackHi(x) | Op::Convert(Cvt::Bitcast, _, x), .. }) => pending.push((*x, Demand::Overwritten(exec))),
                Some(Inst::Core { op: Op::Pack64(lo, hi), .. }) => { pending.push((*lo, Demand::Overwritten(exec))); pending.push((*hi, Demand::Overwritten(exec))); }
                Some(_) => pending.push((v, Demand::Active)),
                None => match params[v.0] {
                    Some((block, index)) if exec_of(&f.blocks[&block]) == Some(exec) => {
                        let position = f.blocks[&block].params.iter().position(|p| p.0 == exec).unwrap();
                        for edge in incoming.get(&block).into_iter().flatten() { pending.push((edge.args[index], Demand::Overwritten(edge.args[position]))); }
                    }
                    _ => pending.push((v, Demand::Active)),
                },
            }
            continue;
        }
        needed[v.0] = true;
        if let Some((block, index)) = producers[v.0] {
            match &f.blocks[&block].insts[index] {
                Inst::Core { op: Op::Select(c, new, old), .. } if masks.predicated[v.0].is_some() => {
                    pending.push((*c, Demand::Active)); pending.push((*new, Demand::Active));
                    if masks.exposed[v.0] != 0 { pending.push((*old, Demand::Overwritten(masks.predicated[v.0].unwrap().1))); }
                }
                Inst::Core { op, .. } => { op.map(|a| { pending.push((a, Demand::Active)); a }); }
                Inst::Packet { input, .. } => pending.push((*input, Demand::Active)),
                Inst::Target { args, .. } => pending.extend(args.values().iter().map(|&a| (a, Demand::Active))),
                Inst::Effect { .. } => {}
            }
        } else if let Some((block, index)) = params[v.0] {
            for edge in incoming.get(&block).into_iter().flatten() { pending.push((edge.args[index], Demand::Active)); }
        }
    }
    needed
}

pub(crate) fn dead_writes(f: &mut Func, masks: &Masks, exec_index: usize) -> usize {
    let needed = needed(f, masks, exec_index);
    let mut renames: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            if let Inst::Core { value, op: Op::Select(_, _, old), .. } = inst {
                if masks.predicated[value.0].is_some() && !needed[value.0] { renames.insert(*value, *old); }
            }
        }
    }
    if renames.is_empty() { return 0; }
    super::simplify::rename(f, &renames);
    for block in f.blocks.values_mut() {
        block.insts.retain(|inst| !matches!(inst, Inst::Core { value, .. } if renames.contains_key(value)));
    }
    let count = renames.len();
    run(f);
    super::compact(f);
    count
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
                Inst::Target { provenance: None, outputs, .. } => outputs.iter().any(|(v, _)| uses[v.0] != 0),
                _ => true,
            });
            round += before - block.insts.len();
        }
        removed += round;
        if round == 0 { break; }
    }
    if removed != 0 { super::compact(f); }
    let _ = ValueId(0);
    removed
}

pub(crate) fn dead_params(f: &mut Func) -> usize {
    let mut removed = 0;
    loop {
        let mut uses = vec![0usize; f.types.len()];
        count_uses(f, &mut uses);
        let mut dead: BTreeMap<BlockId, Vec<usize>> = BTreeMap::new();
        for (&id, block) in &f.blocks {
            if id == f.entry { continue; }
            let indices: Vec<usize> = block.params.iter().enumerate().filter(|(_, &(v, ty))| ty != Ty::I1 && uses[v.0] == 0).map(|(i, _)| i).collect();
            if !indices.is_empty() { dead.insert(id, indices); }
        }
        if dead.is_empty() { break; }
        for (id, indices) in &dead {
            let block = f.blocks.get_mut(id).unwrap();
            for &index in indices.iter().rev() { block.params.remove(index); }
            removed += indices.len();
        }
        for block in f.blocks.values_mut() {
            let edit = |edge: &mut Edge| if let Some(indices) = dead.get(&edge.dst) { for &index in indices.iter().rev() { edge.args.remove(index); } };
            match &mut block.term {
                Term::Br(e) => edit(e),
                Term::CondBr { yes, no, .. } => { edit(yes); edit(no); }
                Term::Ret(_) => {}
            }
        }
    }
    if removed != 0 { super::compact(f); }
    removed
}
