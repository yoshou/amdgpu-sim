use super::super::analysis::{Analyses, Uniformity};
use super::super::ir::{*, Op, Ty, ValueId};
use std::collections::{BTreeMap, BTreeSet};

enum Source { Packed(ValueId), Carried(BlockId, usize), Fresh }

fn source(a: ValueId, b: ValueId, defs: &[Option<Op>], params: &[Option<(BlockId, usize)>], f: &Func) -> Source {
    if let (Some(Op::UnpackLo(x)), Some(Op::UnpackHi(y))) = (defs[a.0], defs[b.0]) {
        if x == y && f.types[x.0] == Ty::I64 { return Source::Packed(x); }
    }
    if let (Some((block, index)), Some((other, next))) = (params[a.0], params[b.0]) {
        if block == other && next == index + 1 { return Source::Carried(block, index); }
    }
    Source::Fresh
}

fn observed(a: ValueId, b: ValueId, defs: &[Option<Op>], out: &mut [Vec<ValueId>]) {
    match (defs[a.0], defs[b.0]) {
        (Some(Op::Select(c, x, y)), Some(Op::Select(d, z, w))) if c == d => { observed(x, z, defs, out); observed(y, w, defs, out); }
        _ => { if !out[a.0].contains(&b) { out[a.0].push(b); } }
    }
}

pub(crate) struct Pairs;
impl super::Pass for Pairs {
    fn name(&self) -> &str { "pairs" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let uniform = analyses.get::<Uniformity>(f).uniform();
        run(f, &uniform) > 0
    }
}

pub(crate) fn run(f: &mut Func, uniform: &[bool]) -> usize {
    let defs = f.definitions();
    let mut params: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut position = vec![usize::MAX; f.blocks.keys().map(|b| b.0 + 1).max().unwrap_or(0)];
    for (at, id) in f.blocks.keys().enumerate() { position[id.0] = at; }
    let mut incoming: Vec<Vec<&Edge>> = (0..f.blocks.len()).map(|_| Vec::new()).collect();
    let mut packed: Vec<Vec<ValueId>> = vec![Vec::new(); f.types.len()];
    let mut float = vec![false; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts { if let Inst::Core { op: Op::Convert(super::super::ir::Cvt::Bitcast, Ty::F64, a), .. } = inst { float[a.0] = true; } }
    }
    for (&id, block) in &f.blocks {
        for (index, &(p, _)) in block.params.iter().enumerate() { params[p.0] = Some((id, index)); }
        let _ = id;
        for edge in block.term.edges() { incoming[position[edge.dst.0]].push(edge); }
        for inst in &block.insts { if let Inst::Core { op: Op::Pack64(a, b), value, .. } = inst { if float[value.0] { observed(*a, *b, &defs, &mut packed); } } }
    }
    let eligible = |block: &Block, index: usize| {
        index + 1 < block.params.len() && block.params[index].1 == Ty::I32 && block.params[index + 1].1 == Ty::I32
            && uniform[block.params[index].0.0] == uniform[block.params[index + 1].0.0]
    };
    let mut candidates: BTreeSet<(BlockId, usize)> = BTreeSet::new();
    for (&id, block) in &f.blocks {
        if id == f.entry { continue; }
        for index in 0..block.params.len() {
            if eligible(block, index) && packed[block.params[index].0 .0].contains(&block.params[index + 1].0) { candidates.insert((id, index)); }
        }
    }
    let mut pending: Vec<(BlockId, usize)> = candidates.iter().copied().collect();
    while let Some((id, index)) = pending.pop() {
        for edge in &incoming[position[id.0]] {
            let args = &edge.args;
            if let Source::Carried(block, pi) = source(args[index], args[index + 1], &defs, &params, f) {
                if block != f.entry && eligible(&f.blocks[&block], pi) && candidates.insert((block, pi)) { pending.push((block, pi)); }
            }
        }
    }
    let mut kept: BTreeSet<(BlockId, usize)> = BTreeSet::new();
    let mut last: Option<(BlockId, usize)> = None;
    for &(id, index) in &candidates {
        if incoming[position[id.0]].is_empty() { continue; }
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
