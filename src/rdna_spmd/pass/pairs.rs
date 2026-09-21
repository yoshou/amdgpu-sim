use super::super::analysis::{Analyses, Constants, Uniformity};
use super::super::ir::{Op, Ty, ValueId, *};
use std::collections::{BTreeMap, BTreeSet};

enum Source {
    Packed(ValueId),
    Carried(BlockId, usize),
    Fresh,
}

fn source(
    a: ValueId,
    b: ValueId,
    defs: &[Option<Op>],
    params: &[Option<(BlockId, usize)>],
    f: &Func,
) -> Source {
    if let (Some(Op::UnpackLo(x)), Some(Op::UnpackHi(y))) = (defs[a.0], defs[b.0]) {
        if x == y && f.types[x.0] == Ty::I64 {
            return Source::Packed(x);
        }
    }
    if let (Some((block, index)), Some((other, next))) = (params[a.0], params[b.0]) {
        if block == other && next == index + 1 {
            return Source::Carried(block, index);
        }
    }
    Source::Fresh
}

fn observed(a: ValueId, b: ValueId, defs: &[Option<Op>], out: &mut [Vec<ValueId>]) {
    match (defs[a.0], defs[b.0]) {
        (Some(Op::Select(c, x, y)), Some(Op::Select(d, z, w))) if c == d => {
            observed(x, z, defs, out);
            observed(y, w, defs, out);
        }
        _ => {
            if !out[a.0].contains(&b) {
                out[a.0].push(b);
            }
        }
    }
}

pub struct Pairs;
impl super::Pass for Pairs {
    fn name(&self) -> &str {
        "pairs"
    }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let uniform = analyses.get::<Uniformity>(f).uniform();
        run(f, &uniform) > 0
    }
}

fn run(f: &mut Func, uniform: &[bool]) -> usize {
    let defs = f.definitions();
    let mut params: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut position = vec![usize::MAX; f.blocks.keys().map(|b| b.0 + 1).max().unwrap_or(0)];
    for (at, id) in f.blocks.keys().enumerate() {
        position[id.0] = at;
    }
    let mut incoming: Vec<Vec<&Edge>> = (0..f.blocks.len()).map(|_| Vec::new()).collect();
    let mut packed: Vec<Vec<ValueId>> = vec![Vec::new(); f.types.len()];
    let mut wide = vec![false; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Core {
                    op: Op::UnpackLo(_) | Op::UnpackHi(_),
                    ..
                } => {}
                _ => inst.for_each_operand(|v| wide[v.0] = true),
            }
        }
        for edge in block.term.edges() {
            for &v in &edge.args {
                wide[v.0] = true;
            }
        }
        match &block.term {
            Term::CondBr { cond, .. } => wide[cond.0] = true,
            Term::Ret(args) => {
                for &v in args {
                    wide[v.0] = true;
                }
            }
            Term::Br(_) => {}
        }
    }
    for (&id, block) in &f.blocks {
        for (index, &(p, _)) in block.params.iter().enumerate() {
            params[p.0] = Some((id, index));
        }
        for edge in block.term.edges() {
            incoming[position[edge.dst.0]].push(edge);
        }
        for inst in &block.insts {
            if let Inst::Core {
                op: Op::Pack64(a, b),
                value,
                ..
            } = inst
            {
                if wide[value.0] {
                    observed(*a, *b, &defs, &mut packed);
                }
            }
        }
    }
    let eligible = |block: &Block, index: usize| {
        index + 1 < block.params.len()
            && block.params[index].1 == Ty::I32
            && block.params[index + 1].1 == Ty::I32
            && uniform[block.params[index].0 .0] == uniform[block.params[index + 1].0 .0]
    };
    let mut slots: BTreeMap<(BlockId, usize), usize> = BTreeMap::new();
    for (&id, block) in &f.blocks {
        if id == f.entry {
            continue;
        }
        for index in 0..block.params.len() {
            if eligible(block, index) {
                let next = slots.len();
                slots.insert((id, index), next);
            }
        }
    }
    let mut class: Vec<usize> = (0..slots.len()).collect();
    fn find(class: &mut Vec<usize>, mut i: usize) -> usize {
        while class[i] != i {
            class[i] = class[class[i]];
            i = class[i];
        }
        i
    }
    for (&(id, index), &slot) in &slots {
        for edge in &incoming[position[id.0]] {
            if let Source::Carried(block, pi) =
                source(edge.args[index], edge.args[index + 1], &defs, &params, f)
            {
                if let Some(&other) = slots.get(&(block, pi)) {
                    let (a, b) = (find(&mut class, slot), find(&mut class, other));
                    class[a] = b;
                }
            }
        }
    }
    let mut observed_class = vec![false; slots.len()];
    for (&(id, index), &slot) in &slots {
        let block = &f.blocks[&id];
        if packed[block.params[index].0 .0].contains(&block.params[index + 1].0) {
            let root = find(&mut class, slot);
            observed_class[root] = true;
        }
    }
    let candidates: BTreeSet<(BlockId, usize)> = slots
        .iter()
        .filter(|(_, &slot)| observed_class[find(&mut class, slot)])
        .map(|(&pair, _)| pair)
        .collect();
    let mut kept: BTreeSet<(BlockId, usize)> = BTreeSet::new();
    let mut last: Option<(BlockId, usize)> = None;
    for &(id, index) in &candidates {
        if incoming[position[id.0]].is_empty() {
            continue;
        }
        if last.is_some_and(|(previous, at)| previous == id && index == at + 1) {
            continue;
        }
        kept.insert((id, index));
        last = Some((id, index));
    }
    let candidates = kept;
    if candidates.is_empty() {
        return 0;
    }
    let mut merged: BTreeMap<(BlockId, usize), ValueId> = BTreeMap::new();
    for &(id, index) in &candidates {
        merged.insert((id, index), f.value(Ty::I64));
    }
    let mut retained: BTreeMap<BlockId, Vec<(ValueId, usize)>> = BTreeMap::new();
    for &(id, index) in &candidates {
        let p = merged[&(id, index)];
        let block = f.blocks.get_mut(&id).unwrap();
        let (lo, _) = block.params[index];
        let (hi, _) = block.params[index + 1];
        let mixed = uniform[lo.0] != uniform[hi.0];
        if mixed && uniform[lo.0] {
            retained.entry(id).or_default().push((lo, index));
        } else {
            block.insts.insert(
                0,
                Inst::Core {
                    value: lo,
                    ty: Ty::I32,
                    op: Op::UnpackLo(p),
                },
            );
        }
        if mixed && uniform[hi.0] {
            retained.entry(id).or_default().push((hi, index + 1));
        } else {
            block.insts.insert(
                0,
                Inst::Core {
                    value: hi,
                    ty: Ty::I32,
                    op: Op::UnpackHi(p),
                },
            );
        }
    }
    let mut removals: BTreeMap<BlockId, Vec<usize>> = BTreeMap::new();
    for &(id, index) in &candidates {
        removals.entry(id).or_default().push(index);
    }
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for from in ids {
        let mut term = f.blocks[&from].term.clone();
        let mut fresh: Vec<Inst> = Vec::new();
        let edges: Vec<&mut Edge> = match &mut term {
            Term::Br(e) => vec![e],
            Term::CondBr { yes, no, .. } => vec![yes, no],
            Term::Ret(_) => vec![],
        };
        for edge in edges {
            let Some(indices) = removals.get(&edge.dst) else {
                continue;
            };
            let mut args = edge.args.clone();
            let extra: Vec<ValueId> = retained
                .get(&edge.dst)
                .map(|kept| kept.iter().map(|&(_, at)| args[at]).collect())
                .unwrap_or_default();
            for &index in indices.iter().rev() {
                let replacement = match source(args[index], args[index + 1], &defs, &params, f) {
                    Source::Packed(x) => x,
                    Source::Carried(block, pi) if merged.contains_key(&(block, pi)) => {
                        merged[&(block, pi)]
                    }
                    _ => {
                        let value = f.value(Ty::I64);
                        fresh.push(Inst::Core {
                            value,
                            ty: Ty::I64,
                            op: Op::Pack64(args[index], args[index + 1]),
                        });
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
        for &(kept, _) in retained.get(id).into_iter().flatten() {
            block.params.push((kept, Ty::I32));
        }
    }
    candidates.len()
}

pub struct WideMemory;
impl super::Pass for WideMemory {
    fn name(&self) -> &str {
        "wide_memory"
    }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let constants = analyses.get::<Constants>(f);
        wide_loads(f, &constants) + wide_stores(f, &constants) > 0
    }
}

fn located(defs: &[Option<Op>], constants: &[Option<u64>], mut v: ValueId) -> (ValueId, i64) {
    let mut offset = 0i64;
    while let Some(Op::Int(IntOp::Add, x, k)) = defs[v.0] {
        let Some(k) = constants[k.0] else { break };
        offset += k as i64;
        v = x;
    }
    (v, offset)
}

struct WordAccess {
    index: usize,
    provenance: u64,
    semantics: MemorySemantics,
    root: ValueId,
    offset: i64,
    exec: ValueId,
    value: ValueId,
}

impl WordAccess {
    fn precedes(&self, other: &WordAccess) -> bool {
        self.provenance >> 8 == other.provenance >> 8
            && self.semantics == other.semantics
            && self.root == other.root
            && self.offset + 4 == other.offset
            && self.exec == other.exec
    }
}

fn word_effect(
    inst: &Inst,
    expected: MemoryOp,
) -> Option<(u64, MemorySemantics, &[ValueId], &[(ValueId, Ty)])> {
    match inst {
        Inst::Effect {
            provenance,
            op:
                EffectOp::Memory {
                    space: Space::Global,
                    op,
                    semantics,
                },
            inputs,
            outputs,
        } if *op == expected && !semantics.volatile => {
            Some((*provenance, *semantics, inputs, outputs))
        }
        _ => None,
    }
}

fn wide_stores(f: &mut Func, constants: &[Option<u64>]) -> usize {
    let defs = f.definitions();
    let half = |v: ValueId| match defs[v.0] {
        Some(Op::UnpackLo(x)) => Some((x, false)),
        Some(Op::UnpackHi(x)) => Some((x, true)),
        _ => None,
    };
    let mut count = 0;
    for block in f.blocks.values_mut() {
        let mut pending: Vec<WordAccess> = Vec::new();
        let mut removed: Vec<usize> = Vec::new();
        for index in 0..block.insts.len() {
            let Some((provenance, semantics, inputs, _)) =
                word_effect(&block.insts[index], MemoryOp::Store(MemSize::B32))
            else {
                if matches!(block.insts[index], Inst::Effect { .. }) {
                    pending.clear();
                }
                continue;
            };
            let (root, offset) = located(&defs, constants, inputs[0]);
            let store = WordAccess {
                index,
                provenance,
                semantics,
                root,
                offset,
                exec: inputs[2],
                value: inputs[1],
            };
            let Some((x, true)) = half(store.value) else {
                pending.push(store);
                continue;
            };
            let partner = pending
                .iter()
                .position(|first| first.precedes(&store) && half(first.value) == Some((x, false)));
            match partner {
                Some(k) => {
                    let first = pending.remove(k).index;
                    if let Inst::Effect {
                        op: EffectOp::Memory { op, .. },
                        inputs,
                        ..
                    } = &mut block.insts[first]
                    {
                        *op = MemoryOp::Store(MemSize::B64);
                        inputs[1] = x;
                    }
                    removed.push(index);
                    count += 1;
                }
                None => pending.push(store),
            }
        }
        for &index in removed.iter().rev() {
            block.insts.remove(index);
        }
    }
    count
}

fn wide_loads(f: &mut Func, constants: &[Option<u64>]) -> usize {
    let defs = f.definitions();
    let mut uses: Vec<Vec<ValueId>> = vec![Vec::new(); f.types.len()];
    let mut packs: BTreeMap<ValueId, (ValueId, ValueId)> = BTreeMap::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Core {
                    value,
                    op: Op::Pack64(a, b),
                    ..
                } => {
                    packs.insert(*value, (*a, *b));
                    uses[a.0].push(*value);
                    uses[b.0].push(*value);
                }
                _ => inst.for_each_operand(|v| uses[v.0].push(ValueId(usize::MAX))),
            }
        }
        for edge in block.term.edges() {
            for &v in &edge.args {
                uses[v.0].push(ValueId(usize::MAX));
            }
        }
        if let Term::CondBr { cond, .. } = &block.term {
            uses[cond.0].push(ValueId(usize::MAX));
        }
    }
    let only_packed_together = |lo: ValueId, hi: ValueId| {
        !uses[lo.0].is_empty()
            && uses[lo.0]
                .iter()
                .chain(&uses[hi.0])
                .all(|u| packs.get(u).is_some_and(|&(a, b)| a == lo && b == hi))
    };
    let mut renames: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut count = 0;
    let mut next = f.types.len();
    let mut fresh: Vec<Ty> = Vec::new();
    for block in f.blocks.values_mut() {
        let mut pending: Vec<WordAccess> = Vec::new();
        let mut removed: Vec<usize> = Vec::new();
        for index in 0..block.insts.len() {
            let Some((provenance, semantics, inputs, outputs)) =
                word_effect(&block.insts[index], MemoryOp::Load(MemSize::B32))
            else {
                if matches!(block.insts[index], Inst::Effect { .. }) {
                    pending.clear();
                }
                continue;
            };
            let (root, offset) = located(&defs, constants, inputs[0]);
            let load = WordAccess {
                index,
                provenance,
                semantics,
                root,
                offset,
                exec: inputs[1],
                value: outputs[0].0,
            };
            let partner = pending
                .iter()
                .position(|first| first.precedes(&load))
                .filter(|&k| only_packed_together(pending[k].value, load.value));
            match partner {
                Some(k) => {
                    let first = pending.remove(k);
                    let wide = ValueId(next);
                    next += 1;
                    fresh.push(Ty::I64);
                    if let Inst::Effect {
                        op: EffectOp::Memory { op, .. },
                        outputs,
                        ..
                    } = &mut block.insts[first.index]
                    {
                        *op = MemoryOp::Load(MemSize::B64);
                        *outputs = vec![(wide, Ty::I64)];
                    }
                    removed.push(index);
                    for &u in &uses[first.value.0] {
                        renames.insert(u, wide);
                    }
                    count += 1;
                }
                None => pending.push(load),
            }
        }
        for &index in removed.iter().rev() {
            block.insts.remove(index);
        }
    }
    if count == 0 {
        return 0;
    }
    f.types.extend(fresh);
    for block in f.blocks.values_mut() {
        block.insts.retain(
            |inst| !matches!(inst, Inst::Core { value, .. } if renames.contains_key(value)),
        );
    }
    f.rename(&renames);
    f.compact();
    count
}
