use super::super::kernel::{Atom, Queries};
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) fn float_within<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, v: ValueId, p: FloatPred, a: ValueId, b: ValueId) -> Bdd {
    let atom = q.atom(Atom::Bit(v));
    let (Some((x, target)), Site::Inst { block, index }) = (float_test(f, facts, p, a, b), facts.site[v.0]) else {
        return atom;
    };
    if target.is_empty() {
        return Bdd::FALSE;
    }
    if target == [(0, FLOAT_TOP), (FLOAT_NAN, FLOAT_NAN)] {
        return Bdd::TRUE;
    }
    let mut sets = Vec::new();
    for inst in &f.blocks[&block].insts[..index] {
        let Inst::Core { value, op: Op::FCmp(r, c, d), .. } = inst else {
            continue;
        };
        let Some((y, set)) = float_test(f, facts, *r, *c, *d) else {
            continue;
        };
        if y != x {
            continue;
        }
        let holds = q.bit(f, facts, *value);
        if set == target {
            return holds;
        }
        sets.push((holds, set));
    }
    if sets.is_empty() {
        return atom;
    }
    regions(q.m(), &sets, &target, atom, FLOAT_NAN)
}

fn float_edges<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, block: BlockId, index: usize) -> Vec<(ValueId, ValueId, bool, Bdd)> {
    let mut edges = Vec::new();
    for inst in &f.blocks[&block].insts[..index] {
        let Inst::Core { value, op: Op::FCmp(r, c, d), .. } = inst else {
            continue;
        };
        let (c, d) = (*c, *d);
        if c == d || facts.constant(f, c).is_some() || facts.constant(f, d).is_some() {
            continue;
        }
        let bit = q.bit(f, facts, *value);
        let not = q.m().not(bit);
        use FloatPred::*;
        let (guard, from, to, strict, both) = match r {
            Olt => (bit, c, d, true, false),
            Ole => (bit, c, d, false, false),
            Ogt => (bit, d, c, true, false),
            Oge => (bit, d, c, false, false),
            Oeq => (bit, c, d, false, true),
            Uge => (not, c, d, true, false),
            Ugt => (not, c, d, false, false),
            Ule => (not, d, c, true, false),
            Ult => (not, d, c, false, false),
            Une => (not, c, d, false, true),
            _ => continue,
        };
        edges.push((from, to, strict, guard));
        if both {
            edges.push((to, from, strict, guard));
        }
    }
    edges
}

pub(super) fn float_relation<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, v: ValueId, p: FloatPred, a: ValueId, b: ValueId) -> Bdd {
    let atom = q.atom(Atom::Bit(v));
    let Site::Inst { block, index } = facts.site[v.0] else {
        return atom;
    };
    if a == b || facts.constant(f, a).is_some() || facts.constant(f, b).is_some() {
        return atom;
    }
    use FloatPred::*;
    let (x, y, strict) = match p {
        Olt | Ult => (a, b, true),
        Ogt | Ugt => (b, a, true),
        Ole | Ule => (a, b, false),
        Oge | Uge => (b, a, false),
        Oeq | Une => (a, b, false),
        _ => return atom,
    };
    let edges = float_edges(q, f, facts, block, index);
    if edges.is_empty() {
        return atom;
    }
    let m = q.m();
    if matches!(p, Oeq | Une) {
        let (up, above) = paths(m, &edges, a, b);
        let (down, below) = paths(m, &edges, b, a);
        let must = m.and(up, down);
        let apart = m.or(above, below);
        let holds = if p == Une { m.not(atom) } else { atom };
        let open = m.not(apart);
        let allowed = m.and(holds, open);
        let equal = m.or(allowed, must);
        return if p == Une { m.not(equal) } else { equal };
    }
    let (reach, sharp) = paths(m, &edges, x, y);
    let (back_reach, back_sharp) = paths(m, &edges, y, x);
    let (forward, back) = if strict { (sharp, back_reach) } else { (reach, back_sharp) };
    let open = m.not(back);
    let allowed = m.and(atom, open);
    m.or(allowed, forward)
}

fn regions(m: &mut Manager, sets: &[(Bdd, Vec<(u64, u64)>)], target: &[(u64, u64)], free: Bdd, top: u64) -> Bdd {
    let mut points: Vec<u64> = vec![0, top + 1];
    for (_, set) in sets {
        for &(low, high) in set {
            points.push(low);
            points.push(high + 1);
        }
    }
    points.sort_unstable();
    points.dedup();
    let member = |set: &[(u64, u64)], x: u64| set.iter().any(|&(low, high)| low <= x && x <= high);
    let mut regions: Vec<(Vec<bool>, bool, bool)> = Vec::new();
    for w in points.windows(2) {
        let (low, high) = (w[0], w[1] - 1);
        let covered: u64 = target
            .iter()
            .map(|&(a, b)| {
                let (a, b) = (a.max(low), b.min(high));
                if a <= b {
                    b - a + 1
                } else {
                    0
                }
            })
            .sum();
        let (some, all) = (covered > 0, covered == high - low + 1);
        let key: Vec<bool> = sets.iter().map(|(_, set)| member(set, low)).collect();
        match regions.iter_mut().find(|(k, _, _)| *k == key) {
            Some(region) => {
                region.1 |= some;
                region.2 &= all;
            }
            None => regions.push((key, some, all)),
        }
    }
    let mut result = Bdd::FALSE;
    for (key, some, all) in regions {
        if !some {
            continue;
        }
        let mut cell = if all { Bdd::TRUE } else { free };
        for ((holds, _), inside) in sets.iter().zip(key) {
            let literal = if inside { *holds } else { m.not(*holds) };
            cell = m.and(cell, literal);
            if cell == Bdd::FALSE {
                break;
            }
        }
        result = m.or(result, cell);
    }
    result
}

fn paths(m: &mut Manager, edges: &[(ValueId, ValueId, bool, Bdd)], from: ValueId, to: ValueId) -> (Bdd, Bdd) {
    let mut any: HashMap<ValueId, Bdd> = HashMap::default();
    let mut strict: HashMap<ValueId, Bdd> = HashMap::default();
    any.insert(from, Bdd::TRUE);
    loop {
        let mut changed = false;
        for &(a, b, is_strict, guard) in edges {
            let reach = any.get(&a).copied().unwrap_or(Bdd::FALSE);
            let sharp = strict.get(&a).copied().unwrap_or(Bdd::FALSE);
            if reach == Bdd::FALSE && sharp == Bdd::FALSE {
                continue;
            }
            let step = m.and(reach, guard);
            let old = any.get(&b).copied().unwrap_or(Bdd::FALSE);
            let new = m.or(old, step);
            if new != old {
                any.insert(b, new);
                changed = true;
            }
            let through = if is_strict { step } else { m.and(sharp, guard) };
            let old = strict.get(&b).copied().unwrap_or(Bdd::FALSE);
            let new = m.or(old, through);
            if new != old {
                strict.insert(b, new);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let reach = if from == to { Bdd::TRUE } else { any.get(&to).copied().unwrap_or(Bdd::FALSE) };
    (reach, strict.get(&to).copied().unwrap_or(Bdd::FALSE))
}

pub(super) fn float_order<Q: Queries>(
    q: &mut Q,
    f: &Func,
    facts: &Facts,
    pred: FloatPred,
    a: ValueId,
    b: ValueId,
    leaf: Bdd,
    memo: &mut HashMap<(ValueId, ValueId), Bdd>,
) -> Bdd {
    if let Some(&g) = memo.get(&(a, b)) {
        return g;
    }
    let float = |v: ValueId| -> Option<f64> {
        let k = facts.constant(f, v)?;
        match f.types[v.0] {
            Ty::F32 => Some(f32::from_bits(k as u32) as f64),
            Ty::F64 => Some(f64::from_bits(k)),
            _ => None,
        }
    };
    let g = if let (Some(x), Some(y)) = (float(a), float(b)) {
        Manager::constant(float_compare(pred, x, y))
    } else if let Some(Op::Select(c, yes, no)) = facts.op(f, a) {
        let c = q.bit(f, facts, c);
        let yes = float_order(q, f, facts, pred, yes, b, leaf, memo);
        let no = float_order(q, f, facts, pred, no, b, leaf, memo);
        q.m().ite(c, yes, no)
    } else if let Some(Op::Select(c, yes, no)) = facts.op(f, b) {
        let c = q.bit(f, facts, c);
        let yes = float_order(q, f, facts, pred, a, yes, leaf, memo);
        let no = float_order(q, f, facts, pred, a, no, leaf, memo);
        q.m().ite(c, yes, no)
    } else {
        leaf
    };
    memo.insert((a, b), g);
    g
}

pub(in super::super::super) fn float_compare(pred: FloatPred, x: f64, y: f64) -> bool {
    let unordered = x.is_nan() || y.is_nan();
    match pred {
        FloatPred::Oeq => !unordered && x == y,
        FloatPred::Ogt => !unordered && x > y,
        FloatPred::Oge => !unordered && x >= y,
        FloatPred::Olt => !unordered && x < y,
        FloatPred::Ole => !unordered && x <= y,
        FloatPred::One => !unordered && x != y,
        FloatPred::Ord => !unordered,
        FloatPred::Uno => unordered,
        FloatPred::Ueq => unordered || x == y,
        FloatPred::Ugt => unordered || x > y,
        FloatPred::Uge => unordered || x >= y,
        FloatPred::Ult => unordered || x < y,
        FloatPred::Ule => unordered || x <= y,
        FloatPred::Une => unordered || x != y,
    }
}

const FLOAT_OFFSET: u64 = 0x7f80_0000;
const FLOAT_TOP: u64 = 2 * FLOAT_OFFSET;
const FLOAT_NAN: u64 = FLOAT_TOP + 1;

fn float_key(bits: u32) -> Option<u64> {
    if f32::from_bits(bits).is_nan() {
        return None;
    }
    let magnitude = (bits & 0x7fff_ffff) as u64;
    Some(if bits >> 31 == 1 { FLOAT_OFFSET - magnitude } else { FLOAT_OFFSET + magnitude })
}

fn float_test(f: &Func, facts: &Facts, p: FloatPred, a: ValueId, b: ValueId) -> Option<(ValueId, Vec<(u64, u64)>)> {
    if f.types[a.0] != Ty::F32 {
        return None;
    }
    let (x, k, p) = match (facts.constant(f, a), facts.constant(f, b)) {
        (None, Some(k)) => (a, k, p),
        (Some(k), None) => {
            let swapped = match p {
                FloatPred::Olt => FloatPred::Ogt,
                FloatPred::Ogt => FloatPred::Olt,
                FloatPred::Ole => FloatPred::Oge,
                FloatPred::Oge => FloatPred::Ole,
                FloatPred::Ult => FloatPred::Ugt,
                FloatPred::Ugt => FloatPred::Ult,
                FloatPred::Ule => FloatPred::Uge,
                FloatPred::Uge => FloatPred::Ule,
                other => other,
            };
            (b, k, swapped)
        }
        _ => return None,
    };
    let unordered = matches!(p, FloatPred::Uno | FloatPred::Ueq | FloatPred::Ugt | FloatPred::Uge | FloatPred::Ult | FloatPred::Ule | FloatPred::Une);
    let mut set = match float_key(k as u32) {
        None if unordered => vec![(0, FLOAT_TOP)],
        None => Vec::new(),
        Some(t) => {
            let below = if t == 0 { Vec::new() } else { vec![(0, t - 1)] };
            let above = if t == FLOAT_TOP { Vec::new() } else { vec![(t + 1, FLOAT_TOP)] };
            match p {
                FloatPred::Olt | FloatPred::Ult => below,
                FloatPred::Ole | FloatPred::Ule => vec![(0, t)],
                FloatPred::Ogt | FloatPred::Ugt => above,
                FloatPred::Oge | FloatPred::Uge => vec![(t, FLOAT_TOP)],
                FloatPred::Oeq | FloatPred::Ueq => vec![(t, t)],
                FloatPred::One | FloatPred::Une => below.into_iter().chain(above).collect(),
                FloatPred::Ord => vec![(0, FLOAT_TOP)],
                FloatPred::Uno => Vec::new(),
            }
        }
    };
    if unordered {
        set.push((FLOAT_NAN, FLOAT_NAN));
    }
    Some((x, set))
}
