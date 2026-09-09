use super::super::ir::{*, Op, Ty, ValueId};
use std::collections::{BTreeMap, BTreeSet};

fn definitions(f: &Func) -> Vec<Option<Op>> {
    let mut out = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts { if let Inst::Core { value, op, .. } = inst { out[value.0] = Some(*op); } }
    }
    out
}

enum Source { Packed(ValueId), Carried(BlockId, usize), Fresh }

fn source(a: ValueId, b: ValueId, defs: &[Option<Op>], params: &BTreeMap<ValueId, (BlockId, usize)>, f: &Func) -> Source {
    if let (Some(Op::UnpackLo(x)), Some(Op::UnpackHi(y))) = (defs[a.0], defs[b.0]) {
        if x == y && f.types[x.0] == Ty::I64 { return Source::Packed(x); }
    }
    if let (Some(&(block, index)), Some(&(other, next))) = (params.get(&a), params.get(&b)) {
        if block == other && next == index + 1 { return Source::Carried(block, index); }
    }
    Source::Fresh
}

fn observed(a: ValueId, b: ValueId, defs: &[Option<Op>], out: &mut BTreeSet<(ValueId, ValueId)>) {
    match (defs[a.0], defs[b.0]) {
        (Some(Op::Select(c, x, y)), Some(Op::Select(d, z, w))) if c == d => { observed(x, z, defs, out); observed(y, w, defs, out); }
        _ => { out.insert((a, b)); }
    }
}

pub(crate) fn run(f: &mut Func, uniform: &[bool]) -> usize {
    let defs = definitions(f);
    let mut params: BTreeMap<ValueId, (BlockId, usize)> = BTreeMap::new();
    let mut incoming: BTreeMap<BlockId, Vec<(BlockId, usize)>> = BTreeMap::new();
    let mut packed: BTreeSet<(ValueId, ValueId)> = BTreeSet::new();
    let mut float: BTreeSet<ValueId> = BTreeSet::new();
    for block in f.blocks.values() {
        for inst in &block.insts { if let Inst::Core { op: Op::Convert(super::super::ir::Cvt::Bitcast, Ty::F64, a), .. } = inst { float.insert(*a); } }
    }
    for (&id, block) in &f.blocks {
        for (index, &(p, _)) in block.params.iter().enumerate() { params.insert(p, (id, index)); }
        for (edge_index, edge) in block.term.edges().iter().enumerate() { incoming.entry(edge.dst).or_default().push((id, edge_index)); }
        for inst in &block.insts { if let Inst::Core { op: Op::Pack64(a, b), value, .. } = inst { if float.contains(value) { observed(*a, *b, &defs, &mut packed); } } }
    }
    let edge_args = |f: &Func, from: BlockId, edge_index: usize| -> Vec<ValueId> { f.blocks[&from].term.edges()[edge_index].args.clone() };
    let eligible = |block: &Block, index: usize| {
        index + 1 < block.params.len() && block.params[index].1 == Ty::I32 && block.params[index + 1].1 == Ty::I32
            && uniform[block.params[index].0.0] == uniform[block.params[index + 1].0.0]
    };
    let mut candidates: BTreeSet<(BlockId, usize)> = BTreeSet::new();
    for (&id, block) in &f.blocks {
        if id == f.entry { continue; }
        for index in 0..block.params.len() {
            if eligible(block, index) && packed.contains(&(block.params[index].0, block.params[index + 1].0)) { candidates.insert((id, index)); }
        }
    }
    loop {
        let mut added = Vec::new();
        for &(id, index) in &candidates {
            for &(from, edge_index) in incoming.get(&id).into_iter().flatten() {
                let args = edge_args(f, from, edge_index);
                if let Source::Carried(block, pi) = source(args[index], args[index + 1], &defs, &params, f) {
                    if block != f.entry && eligible(&f.blocks[&block], pi) && !candidates.contains(&(block, pi)) { added.push((block, pi)); }
                }
            }
        }
        if added.is_empty() { break; }
        candidates.extend(added);
    }
    let mut kept: BTreeSet<(BlockId, usize)> = BTreeSet::new();
    let mut last: Option<(BlockId, usize)> = None;
    for &(id, index) in &candidates {
        if !incoming.contains_key(&id) { continue; }
        if last.is_some_and(|(previous, at)| previous == id && index == at + 1) { continue; }
        kept.insert((id, index));
        last = Some((id, index));
    }
    let candidates = kept;
    if candidates.is_empty() { return 0; }
    let mut merged: BTreeMap<(BlockId, usize), ValueId> = BTreeMap::new();
    for &(id, index) in &candidates { merged.insert((id, index), f.value(Ty::I64)); }
    let mut retained: BTreeMap<BlockId, Vec<(ValueId, usize)>> = BTreeMap::new();
    for &(id, index) in &candidates {
        let p = merged[&(id, index)];
        let block = f.blocks.get_mut(&id).unwrap();
        let (lo, _) = block.params[index];
        let (hi, _) = block.params[index + 1];
        let mixed = uniform[lo.0] != uniform[hi.0];
        if mixed && uniform[lo.0] { retained.entry(id).or_default().push((lo, index)); }
        else { block.insts.insert(0, Inst::Core { value: lo, ty: Ty::I32, op: Op::UnpackLo(p) }); }
        if mixed && uniform[hi.0] { retained.entry(id).or_default().push((hi, index + 1)); }
        else { block.insts.insert(0, Inst::Core { value: hi, ty: Ty::I32, op: Op::UnpackHi(p) }); }
    }
    let mut removals: BTreeMap<BlockId, Vec<usize>> = BTreeMap::new();
    for &(id, index) in &candidates { removals.entry(id).or_default().push(index); }
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for from in ids {
        let mut term = f.blocks[&from].term.clone();
        let mut fresh: Vec<Inst> = Vec::new();
        let edges: Vec<&mut Edge> = match &mut term { Term::Br(e) => vec![e], Term::CondBr { yes, no, .. } => vec![yes, no], Term::Ret(_) => vec![] };
        for edge in edges {
            let Some(indices) = removals.get(&edge.dst) else { continue; };
            let mut args = edge.args.clone();
            let extra: Vec<ValueId> = retained.get(&edge.dst).map(|kept| kept.iter().map(|&(_, at)| args[at]).collect()).unwrap_or_default();
            for &index in indices.iter().rev() {
                let replacement = match source(args[index], args[index + 1], &defs, &params, f) {
                    Source::Packed(x) => x,
                    Source::Carried(block, pi) if merged.contains_key(&(block, pi)) => merged[&(block, pi)],
                    _ => {
                        let value = f.value(Ty::I64);
                        fresh.push(Inst::Core { value, ty: Ty::I64, op: Op::Pack64(args[index], args[index + 1]) });
                        value
                    }
                };
                args[index] = replacement;
                args.remove(index + 1);
            }
            args.extend(extra);
            edge.args = args;
        }
        let block = f.blocks.get_mut(&from).unwrap();
        block.insts.extend(fresh);
        block.term = term;
    }
    for (id, indices) in &removals {
        let block = f.blocks.get_mut(id).unwrap();
        let mut sorted = indices.clone();
        sorted.sort_unstable();
        for &index in sorted.iter().rev() {
            block.params[index] = (merged[&(*id, index)], Ty::I64);
            block.params.remove(index + 1);
        }
        for &(kept, _) in retained.get(id).into_iter().flatten() { block.params.push((kept, Ty::I32)); }
    }
    candidates.len()
}
