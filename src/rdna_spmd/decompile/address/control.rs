use super::form::*;
use super::limits::*;
use super::queries::*;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) type Edges = Vec<(BlockId, usize)>;

pub(super) fn decide<'a, Q: Queries<'a>>(q: &mut Q, cond: ValueId) -> Option<bool> {
    let lanes: Vec<usize> = (0..q.program().lanes()).filter(|&l| q.symbols().valid(l)).collect();
    if q.program().facts.uniform[cond.0] {
        return lanes.first().and_then(|&l| q.bit(cond, l, None).0);
    }
    let mut known: Option<bool> = None;
    for l in lanes {
        match (q.bit(cond, l, None).0, known) {
            (Some(bit), None) => known = Some(bit),
            (Some(bit), Some(old)) if bit != old => return None,
            _ => {}
        }
    }
    known
}

pub(super) fn takes<'a, Q: Queries<'a>>(q: &mut Q, pred: BlockId, slot: usize) -> bool {
    q.decision(pred).is_none_or(|yes| (slot == 0) == yes)
}

pub(super) fn can_take<'a, Q: Queries<'a>>(q: &mut Q, pred: BlockId, slot: usize, block: BlockId) -> bool {
    if q.program().rank[&pred] >= q.program().rank[&block] {
        return true;
    }
    q.reached(pred) && takes(q, pred, slot)
}

pub(super) fn edges_into<'a, Q: Queries<'a>>(q: &mut Q, block: BlockId) -> (Edges, Edges) {
    let own = q.program().rank[&block];
    let facts = q.program().facts;
    let taken: Edges = facts.incoming[&block]
        .iter()
        .copied()
        .filter(|&(pred, slot)| can_take(q, pred, slot, block))
        .collect();
    taken.into_iter().partition(|&(pred, _)| q.program().rank[&pred] < own)
}

pub(super) fn incoming<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, block: BlockId, index: usize) -> Option<Vec<ValueId>> {
    Some(incoming_edges(q, v, block, index)?.into_iter().map(|(_, a)| a).collect())
}

pub(super) fn incoming_edges<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, block: BlockId, index: usize) -> Option<Vec<((BlockId, usize), ValueId)>> {
    let header = q.program().headers.contains(&block);
    let own = q.program().rank[&block];
    let mut out = Vec::new();
    let facts = q.program().facts;
    for &(pred, slot) in &facts.incoming[&block] {
        if !can_take(q, pred, slot, block) {
            continue;
        }
        let arg = q.program().f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
        if header && q.program().rank[&pred] >= own {
            if arg != v {
                return None;
            }
            continue;
        }
        out.push(((pred, slot), arg));
    }
    Some(out)
}

pub(super) fn entering_value<'a, Q: Queries<'a>>(q: &mut Q, entering: &[(BlockId, usize)], header: BlockId, index: usize, lane: usize) -> Option<Value> {
    let mut first: Option<Value> = None;
    for &e in entering {
        let value = q.operand(q.program().edge_arg(e, index), header, lane, None).0;
        match &first {
            None => first = Some(value),
            Some(old) if *old == value => {}
            Some(_) => return None,
        }
    }
    first
}

pub(super) fn compute_limits<'a, Q: Queries<'a>>(q: &mut Q, b: BlockId) -> Limits {
    if b == q.program().f.entry {
        return Limits::default();
    }
    if q.program().headers.contains(&b) {
        let d = q.program().facts.order[q.program().idom[q.program().rank[&b]]];
        return (*q.limits(d)).clone();
    }
    let mut out: Option<Limits> = None;
    for (pred, slot) in q.program().facts.incoming[&b].clone() {
        let mut here = (*q.limits(pred)).clone();
        if let Some((cond, taken)) = q.program().edge_condition(pred, slot) {
            let limits = condition_limits(q, cond, taken, None);
            here = both_limits(here, limits);
        }
        let joined = match out {
            None => here,
            Some(old) => either_limits(old, here),
        };
        if joined.is_empty() {
            return joined;
        }
        out = Some(joined);
    }
    out.unwrap_or_default()
}

pub(super) fn condition_limits<'a, Q: Queries<'a>>(q: &mut Q, cond: ValueId, taken: bool, lane: Option<(BlockId, usize)>) -> Limits {
    let (tree, shared) = q.reason(|conditions, program| conditions.condition(program, cond, taken));
    let mut memo = shared.then(HashMap::default);
    limits_of(q, &tree, lane, &mut memo)
}

fn limits_of<'a, Q: Queries<'a>>(q: &mut Q, cond: &Cond, lane: Option<(BlockId, usize)>, memo: &mut Option<HashMap<(ValueId, bool), Limits>>) -> Limits {
    let (key, parts, all) = match cond {
        Cond::Leaf(c, holds) => return comparison_limits(q, *c, *holds, lane),
        Cond::All(v, holds, parts) => ((*v, *holds), parts, true),
        Cond::Any(v, holds, parts) => ((*v, *holds), parts, false),
    };
    if let Some(found) = memo.as_ref().and_then(|m| m.get(&key)) {
        return found.clone();
    }
    let mut joined: Option<Limits> = None;
    for part in parts {
        let own = limits_of(q, part, lane, memo);
        joined = Some(match joined {
            None => own,
            Some(old) if all => both_limits(old, own),
            Some(old) => either_limits(old, own),
        });
    }
    let limits = joined.unwrap_or_default();
    if let Some(memo) = memo {
        memo.insert(key, limits.clone());
    }
    limits
}

fn comparison_limits<'a, Q: Queries<'a>>(q: &mut Q, cond: ValueId, taken: bool, lane: Option<(BlockId, usize)>) -> Limits {
    match q.program().facts.op(q.program().f, cond) {
        Some(Op::Cmp(p, x, y)) if q.program().f.types[x.0] == Ty::I32 && (lane.is_some() || q.program().facts.uniform[x.0] && q.program().facts.uniform[y.0]) => {
            let pred = if taken { p } else { negated(p) };
            let (fx, fy) = match lane {
                Some((at, l)) => (q.operand(x, at, l, None).0.form, q.operand(y, at, l, None).0.form),
                None => (q.value(x, 0, None).0.form, q.value(y, 0, None).0.form),
            };
            match (fx.as_constant(), fy.as_constant()) {
                (None, Some(k)) => Limits {
                    classes: vec![class_limit(&fx, &satisfying(pred, k))],
                    orders: Vec::new(),
                },
                (Some(k), None) => Limits {
                    classes: vec![class_limit(&fy, &satisfying(swapped(pred), k))],
                    orders: Vec::new(),
                },
                (None, None) if fx.sub(&fy).as_constant().is_none() => Limits {
                    classes: Vec::new(),
                    orders: order_limit(&fx, &fy, pred),
                },
                _ => Limits::default(),
            }
        }
        _ => Limits::default(),
    }
}

pub(super) fn bounds_at<'a, Q: Queries<'a>>(q: &mut Q, form: &Form, at: BlockId) -> Option<(u64, u64)> {
    let plain = q.symbols().bounds(form);
    if form.as_constant().is_some() {
        return plain;
    }
    let limits = q.limits(at);
    let class = Form {
        constant: 0,
        terms: form.terms.clone(),
    };
    if !limits.classes.iter().any(|(c, _)| *c == class) {
        return plain;
    }
    let pieces = q.symbols().pieces(form, &limits);
    match (pieces.first(), pieces.last()) {
        (Some(&(low, _)), Some(&(_, high))) => Some((low, high)),
        _ => plain,
    }
}

pub(super) fn last_trip<'a, Q: Queries<'a>>(q: &mut Q, header: BlockId, trips: Unknown) -> Option<u32> {
    let own = q.program().rank[&header];
    let mut guards: Vec<(ValueId, bool)> = Vec::new();
    for &(pred, slot) in &q.program().facts.incoming[&header].clone() {
        if q.program().rank[&pred] < own {
            continue;
        }
        let edge = q.program().guard(pred, slot)?;
        if !guards.contains(&edge) {
            guards.push(edge);
        }
    }
    let linear = |f: &Form| match f.terms.as_slice() {
        [] => Some((f.constant, 0)),
        [(u, k)] if *u == trips => Some((f.constant, *k)),
        _ => None,
    };
    let mut shape: Option<(IntPred, bool, (u32, u32), (u32, u32))> = None;
    for (cond, taken) in guards {
        let cond = q.program().copies.get(&cond).copied().unwrap_or(cond);
        let Some(Op::Cmp(pred, a, b)) = q.program().facts.op(q.program().f, cond) else {
            return None;
        };
        if q.program().f.types[a.0] != Ty::I32 {
            return None;
        }
        let mut lanes: Vec<usize> = (0..q.program().lanes()).filter(|&l| q.symbols().valid(l)).collect();
        if q.program().facts.uniform[a.0] && q.program().facts.uniform[b.0] {
            lanes.truncate(1);
        }
        let mut operands: Option<(Form, Form)> = None;
        for l in lanes {
            let x = q.value(a, l, None).0.form;
            let y = q.value(b, l, None).0.form;
            match &operands {
                None => operands = Some((x, y)),
                Some((ox, oy)) if *ox == x && *oy == y => {}
                Some(_) => return None,
            }
        }
        let (x, y) = operands?;
        let guard = (pred, taken, linear(&x)?, linear(&y)?);
        match shape {
            None => shape = Some(guard),
            Some(old) if old == guard => {}
            Some(_) => return None,
        }
    }
    let (pred, taken, x, y) = shape?;
    first_failure(pred, taken, x, y)
}
