use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct Store {
    pub(super) index: usize,
    pub(super) address: ValueId,
    pub(super) data: Option<ValueId>,
    pub(super) predicate: ValueId,
    pub(super) bytes: u32,
}

pub(super) fn private_stores(f: &Func, facts: &Facts) -> HashMap<BlockId, Vec<Store>> {
    let mut out: HashMap<BlockId, Vec<Store>> = HashMap::default();
    for &b in &facts.order {
        for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
            if let Inst::Effect {
                op:
                    EffectOp::Memory {
                        space: Space::Scratch,
                        op,
                        ..
                    },
                inputs,
                ..
            } = inst
            {
                let (bytes, data) = match *op {
                    MemoryOp::Store(size) => (size.bytes(), Some(inputs[1])),
                    MemoryOp::Load(_) | MemoryOp::Fence => continue,
                    _ => (4, None),
                };
                out.entry(b).or_default().push(Store {
                    index,
                    address: inputs[0],
                    data,
                    predicate: inputs[op.mask_input()],
                    bytes,
                });
            }
        }
    }
    out
}

pub(super) fn dominators(f: &Func, facts: &Facts) -> Vec<usize> {
    let n = facts.order.len();
    let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
    let mut idom: Vec<Option<usize>> = vec![None; n];
    if n == 0 {
        return Vec::new();
    }
    idom[0] = Some(0);
    let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| {
        while a != b {
            while a > b {
                a = idom[a].unwrap();
            }
            while b > a {
                b = idom[b].unwrap();
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for r in 1..n {
            let mut new: Option<usize> = None;
            for &(pred, _) in &facts.incoming[&facts.order[r]] {
                let p = rank[&pred];
                if idom[p].is_none() {
                    continue;
                }
                new = Some(match new {
                    None => p,
                    Some(q) => intersect(&idom, p, q),
                });
            }
            if new.is_some() && new != idom[r] {
                idom[r] = new;
                changed = true;
            }
        }
    }
    let _ = f;
    idom.into_iter().map(|d| d.unwrap_or(0)).collect()
}

pub const PRIVATE_MEMORY: ValueId = ValueId(usize::MAX);

pub(super) fn users(f: &Func, facts: &Facts) -> HashMap<ValueId, Vec<ValueId>> {
    let mut users: HashMap<ValueId, Vec<ValueId>> = HashMap::default();
    for &b in &facts.order {
        let block = &f.blocks[&b];
        for inst in &block.insts {
            let mut outputs: Vec<ValueId> = Vec::new();
            inst.for_each_output(|o| outputs.push(o));
            let private = matches!(
                inst,
                Inst::Effect {
                    op: EffectOp::Memory {
                        space: Space::Scratch,
                        ..
                    },
                    ..
                }
            );
            let writes = private
                && matches!(inst, Inst::Effect { op: EffectOp::Memory { op, .. }, .. } if !matches!(op, MemoryOp::Load(_) | MemoryOp::Fence));
            for o in inst.operands() {
                let entry = users.entry(o).or_default();
                entry.extend(outputs.iter().copied());
                if writes {
                    entry.push(PRIVATE_MEMORY);
                }
            }
            if private {
                users.entry(PRIVATE_MEMORY).or_default().extend(outputs.iter().copied());
            }
        }
        for edge in block.term.edges() {
            let params = &f.blocks[&edge.dst].params;
            for (arg, (param, _)) in edge.args.iter().zip(params) {
                users.entry(*arg).or_default().push(*param);
            }
        }
    }
    users
}

pub(super) fn loops(facts: &Facts, idom: &[usize], reaches: &[Vec<bool>]) -> HashMap<BlockId, Vec<BlockId>> {
    let n = facts.order.len();
    let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
    let headers: Vec<usize> = (0..n)
        .filter(|&r| facts.incoming[&facts.order[r]].iter().any(|&(p, _)| rank[&p] >= r))
        .collect();
    let dominates = |h: usize, mut b: usize| loop {
        if b == h {
            return true;
        }
        if b == 0 {
            return false;
        }
        b = idom[b];
    };
    (0..n)
        .map(|r| {
            let around = headers
                .iter()
                .filter(|&&h| dominates(h, r) && (h == r || reaches[r][h]))
                .map(|&h| facts.order[h])
                .collect();
            (facts.order[r], around)
        })
        .collect()
}

pub(super) fn reaches(f: &Func, facts: &Facts) -> Vec<Vec<bool>> {
    let n = facts.order.len();
    let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
    let successors: Vec<Vec<usize>> = facts
        .order
        .iter()
        .map(|b| f.blocks[b].term.edges().map(|e| rank[&e.dst]).collect())
        .collect();
    (0..n)
        .map(|start| {
            let mut seen = vec![false; n];
            let mut stack: Vec<usize> = successors[start].clone();
            while let Some(x) = stack.pop() {
                if seen[x] {
                    continue;
                }
                seen[x] = true;
                stack.extend(successors[x].iter().copied());
            }
            seen
        })
        .collect()
}

pub struct Copies(Vec<ValueId>);

impl Copies {
    pub fn get(&self, v: &ValueId) -> Option<&ValueId> {
        let root = &self.0[v.0];
        (root != v).then_some(root)
    }

    pub fn contains_key(&self, v: &ValueId) -> bool {
        self.0[v.0] != *v
    }
}

pub(super) fn copies(f: &Func, facts: &Facts) -> Copies {
    const NONE: u32 = u32::MAX;
    let n = f.types.len();
    let edges: Vec<Vec<&[ValueId]>> = facts
        .order
        .iter()
        .map(|b| {
            facts.incoming[b]
                .iter()
                .map(|&(pred, slot)| &f.blocks[&pred].term.edges().nth(slot).unwrap().args[..])
                .collect()
        })
        .collect();
    let mut place = vec![(NONE, 0u32); n];
    let mut params: Vec<ValueId> = Vec::new();
    for (r, &b) in facts.order.iter().enumerate() {
        if b == f.entry {
            continue;
        }
        for (index, &(p, _)) in f.blocks[&b].params.iter().enumerate() {
            params.push(p);
            place[p.0] = (r as u32, index as u32);
        }
    }
    let args = |p: ValueId| {
        let (r, index) = place[p.0];
        edges[r as usize].iter().map(move |e| e[index as usize])
    };
    let mut order = vec![NONE; n];
    let mut low = vec![0u32; n];
    let mut on = vec![false; n];
    let mut component = vec![NONE; n];
    let mut root: Vec<ValueId> = (0..n).map(ValueId).collect();
    let mut stack: Vec<ValueId> = Vec::new();
    let mut members: Vec<ValueId> = Vec::new();
    let (mut next, mut count) = (0u32, 0u32);
    for &start in &params {
        if order[start.0] != NONE {
            continue;
        }
        let mut frames: Vec<(ValueId, usize)> = vec![(start, 0)];
        order[start.0] = next;
        low[start.0] = next;
        next += 1;
        stack.push(start);
        on[start.0] = true;
        while let Some(&mut (v, ref mut at)) = frames.last_mut() {
            let (r, index) = place[v.0];
            let list = &edges[r as usize];
            if *at < list.len() {
                let a = list[*at][index as usize];
                *at += 1;
                if place[a.0].0 == NONE {
                    continue;
                }
                if order[a.0] == NONE {
                    order[a.0] = next;
                    low[a.0] = next;
                    next += 1;
                    stack.push(a);
                    on[a.0] = true;
                    frames.push((a, 0));
                } else if on[a.0] {
                    low[v.0] = low[v.0].min(order[a.0]);
                }
                continue;
            }
            frames.pop();
            if let Some(&(parent, _)) = frames.last() {
                low[parent.0] = low[parent.0].min(low[v.0]);
            }
            if low[v.0] != order[v.0] {
                continue;
            }
            members.clear();
            loop {
                let x = stack.pop().unwrap();
                on[x.0] = false;
                component[x.0] = count;
                members.push(x);
                if x == v {
                    break;
                }
            }
            let mut outside: Option<ValueId> = None;
            let mut single = true;
            'scan: for &p in &members {
                for a in args(p) {
                    let a = root[a.0];
                    if component[a.0] == count {
                        continue;
                    }
                    match outside {
                        None => outside = Some(a),
                        Some(o) if o == a => {}
                        Some(_) => {
                            single = false;
                            break 'scan;
                        }
                    }
                }
            }
            if let (true, Some(value)) = (single, outside) {
                for &p in &members {
                    root[p.0] = value;
                }
            }
            count += 1;
        }
    }
    Copies(root)
}
