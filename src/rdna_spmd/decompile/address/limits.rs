use super::super::encoding::{mirrored, outcomes};
use super::form::*;
use super::graph::Copies;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) fn first_failure(pred: IntPred, taken: bool, x: (u32, u32), y: (u32, u32)) -> Option<u32> {
    if matches!(pred, IntPred::Eq | IntPred::Ne) {
        let (c, k) = (x.0.wrapping_sub(y.0), x.1.wrapping_sub(y.1));
        if (pred == IntPred::Ne) == taken {
            if k == 0 {
                return (c == 0).then_some(0);
            }
            let shift = k.trailing_zeros();
            if c & ((1u64 << shift) - 1) as u32 != 0 {
                return None;
            }
            let odd = k >> shift;
            let inverse = (0..5).fold(odd, |inv, _| inv.wrapping_mul(2u32.wrapping_sub(odd.wrapping_mul(inv))));
            let low_mask = if shift == 0 { u32::MAX } else { (1u32 << (32 - shift)) - 1 };
            return Some((c.wrapping_neg() >> shift).wrapping_mul(inverse) & low_mask);
        }
        return if c != 0 {
            Some(0)
        } else {
            (k != 0).then_some(1)
        };
    }
    let (moving, fixed, pred) = match (x.1, y.1) {
        (_, 0) => (x, y.0, pred),
        (0, _) => (y, x.0, swapped(pred)),
        _ => return None,
    };
    let signed = matches!(pred, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
    let wide = |v: u32| if signed { v as i32 as i64 } else { v as i64 };
    let (c, k, bound) = (wide(moving.0), moving.1 as i32 as i64, wide(fixed));
    let stays = |t: i64| {
        let m = c + k * t;
        let holds = match pred {
            IntPred::Ult | IntPred::Slt => m < bound,
            IntPred::Ule | IntPred::Sle => m <= bound,
            IntPred::Ugt | IntPred::Sgt => m > bound,
            _ => m >= bound,
        };
        holds == taken
    };
    if !stays(0) {
        return Some(0);
    }
    let first = match k.signum() {
        0 => return None,
        _ => {
            let edge = (bound - c) / k;
            (edge.max(0)..=edge.max(0) + 2).find(|&t| !stays(t))?
        }
    };
    let (low, high) = if signed { (i32::MIN as i64, i32::MAX as i64) } else { (0, u32::MAX as i64) };
    let last = c + k * first;
    (low..=high).contains(&last).then_some(first as u32)
}

pub(super) fn negated(pred: IntPred) -> IntPred {
    match pred {
        IntPred::Eq => IntPred::Ne,
        IntPred::Ne => IntPred::Eq,
        IntPred::Ult => IntPred::Uge,
        IntPred::Uge => IntPred::Ult,
        IntPred::Ule => IntPred::Ugt,
        IntPred::Ugt => IntPred::Ule,
        IntPred::Slt => IntPred::Sge,
        IntPred::Sge => IntPred::Slt,
        IntPred::Sle => IntPred::Sgt,
        IntPred::Sgt => IntPred::Sle,
    }
}

pub enum Cond {
    Leaf(ValueId, bool),
    All(ValueId, bool, Vec<std::rc::Rc<Cond>>),
    Any(ValueId, bool, Vec<std::rc::Rc<Cond>>),
}

pub fn condition(f: &Func, facts: &Facts, copies: &Copies, cond: ValueId, taken: bool) -> std::rc::Rc<Cond> {
    fn build(f: &Func, facts: &Facts, copies: &Copies, cond: ValueId, taken: bool, memo: &mut HashMap<(ValueId, bool), std::rc::Rc<Cond>>) -> std::rc::Rc<Cond> {
        let cond = copies.get(&cond).copied().unwrap_or(cond);
        if let Some(found) = memo.get(&(cond, taken)) {
            return found.clone();
        }
        let node = match facts.op(f, cond) {
            Some(Op::Int(k @ (IntOp::And | IntOp::Or), a, b)) if f.types[cond.0] == Ty::I1 => {
                let parts = vec![build(f, facts, copies, a, taken, memo), build(f, facts, copies, b, taken, memo)];
                if (k == IntOp::And) == taken {
                    Cond::All(cond, taken, parts)
                } else {
                    Cond::Any(cond, taken, parts)
                }
            }
            Some(Op::Int(IntOp::Xor, a, one)) if f.types[a.0] == Ty::I1 && facts.constant(f, one) == Some(1) => {
                Cond::All(cond, taken, vec![build(f, facts, copies, a, !taken, memo)])
            }
            _ => Cond::Leaf(cond, taken),
        };
        let node = std::rc::Rc::new(node);
        memo.insert((cond, taken), node.clone());
        node
    }
    build(f, facts, copies, cond, taken, &mut HashMap::default())
}

pub fn literals(cond: &Cond, out: &mut Vec<(ValueId, bool)>) {
    let (Cond::Leaf(v, holds) | Cond::All(v, holds, _) | Cond::Any(v, holds, _)) = cond;
    if out.contains(&(*v, *holds)) {
        return;
    }
    out.push((*v, *holds));
    if let Cond::All(_, _, parts) = cond {
        for part in parts {
            literals(part, out);
        }
    }
}

pub(super) fn shares(cond: &Cond) -> bool {
    match cond {
        Cond::Leaf(..) => false,
        Cond::All(_, _, parts) | Cond::Any(_, _, parts) => parts.iter().any(|p| std::rc::Rc::strong_count(p) > 1 || shares(p)),
    }
}

pub(super) fn satisfying(pred: IntPred, k: u32) -> Vec<(u64, u64)> {
    let max = u32::MAX as i64;
    let unsigned = |low: i64, high: i64| if low > high { Vec::new() } else { vec![(low as u64, high as u64)] };
    let signed = |low: i64, high: i64| {
        if low > high {
            Vec::new()
        } else if low >= 0 || high < 0 {
            vec![(low as i32 as u32 as u64, high as i32 as u32 as u64)]
        } else {
            vec![(0, high as u64), (low as i32 as u32 as u64, u32::MAX as u64)]
        }
    };
    let (u, s) = (k as i64, k as i32 as i64);
    let (least, most) = (i32::MIN as i64, i32::MAX as i64);
    match pred {
        IntPred::Eq => vec![(k as u64, k as u64)],
        IntPred::Ne => [unsigned(0, u - 1), unsigned(u + 1, max)].concat(),
        IntPred::Ult => unsigned(0, u - 1),
        IntPred::Ule => unsigned(0, u),
        IntPred::Ugt => unsigned(u + 1, max),
        IntPred::Uge => unsigned(u, max),
        IntPred::Slt => signed(least, s - 1),
        IntPred::Sle => signed(least, s),
        IntPred::Sgt => signed(s + 1, most),
        IntPred::Sge => signed(s, most),
    }
}

pub(super) fn shifted_pieces(set: &[(u64, u64)], k: u32) -> Vec<(u64, u64)> {
    let whole = 1u64 << 32;
    let mut out = Vec::new();
    for &(low, high) in set {
        let (low, high) = (low + k as u64, high + k as u64);
        if high < whole {
            out.push((low, high));
        } else if low >= whole {
            out.push((low - whole, high - whole));
        } else {
            out.push((low, whole - 1));
            out.push((0, high - whole));
        }
    }
    normalized(out)
}

pub(super) fn intersected(a: &[(u64, u64)], b: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    for &(la, ha) in a {
        for &(lb, hb) in b {
            let (low, high) = (la.max(lb), ha.min(hb));
            if low <= high {
                out.push((low, high));
            }
        }
    }
    normalized(out)
}

pub(super) fn normalized(mut set: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    set.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (low, high) in set {
        match out.last_mut() {
            Some(last) if low <= last.1 + 1 => last.1 = last.1.max(high),
            _ => out.push((low, high)),
        }
    }
    out
}

pub type Classes = Vec<(Form, Vec<(u64, u64)>)>;

#[derive(Clone, Debug, Default)]
pub(super) struct Limits {
    pub(super) classes: Classes,
    pub(super) orders: Vec<(Form, Form, u8)>,
}

impl Limits {
    pub(super) fn is_empty(&self) -> bool {
        self.classes.is_empty() && self.orders.is_empty()
    }
}

pub(super) fn both_limits(a: Limits, b: Limits) -> Limits {
    Limits {
        classes: conjoined(a.classes, b.classes),
        orders: orders_conjoined(a.orders, b.orders),
    }
}

pub(super) fn either_limits(a: Limits, b: Limits) -> Limits {
    Limits {
        classes: disjoined(a.classes, b.classes),
        orders: orders_disjoined(a.orders, b.orders),
    }
}

pub(super) fn order_limit(x: &Form, y: &Form, pred: IntPred) -> Vec<(Form, Form, u8)> {
    if x == y {
        return Vec::new();
    }
    let mask = outcomes(pred);
    if (x.terms.as_slice(), x.constant) <= (y.terms.as_slice(), y.constant) {
        vec![(x.clone(), y.clone(), mask)]
    } else {
        vec![(y.clone(), x.clone(), mirrored(mask))]
    }
}

pub(super) fn orders_conjoined(mut a: Vec<(Form, Form, u8)>, b: Vec<(Form, Form, u8)>) -> Vec<(Form, Form, u8)> {
    for (x, y, mask) in b {
        match a.iter_mut().find(|(p, q, _)| *p == x && *q == y) {
            Some((_, _, old)) => *old &= mask,
            None => a.push((x, y, mask)),
        }
    }
    a
}

pub(super) fn orders_disjoined(a: Vec<(Form, Form, u8)>, b: Vec<(Form, Form, u8)>) -> Vec<(Form, Form, u8)> {
    a.into_iter()
        .filter_map(|(x, y, mask)| {
            let (_, _, other) = b.iter().find(|(p, q, _)| *p == x && *q == y)?;
            Some((x, y, mask | other))
        })
        .collect()
}

pub(super) fn class_limit(form: &Form, values: &[(u64, u64)]) -> (Form, Vec<(u64, u64)>) {
    let class = Form {
        constant: 0,
        terms: form.terms.clone(),
    };
    (class, normalized(shifted_pieces(values, form.constant.wrapping_neg())))
}

pub(super) fn conjoined(mut a: Classes, b: Classes) -> Classes {
    for (class, set) in b {
        match a.iter_mut().find(|(c, _)| *c == class) {
            Some((_, old)) => *old = intersected(old, &set),
            None => a.push((class, set)),
        }
    }
    a
}

pub(super) fn disjoined(a: Classes, b: Classes) -> Classes {
    a.into_iter()
        .filter_map(|(class, set)| {
            let (_, other) = b.iter().find(|(c, _)| *c == class)?;
            Some((class, normalized([set, other.clone()].concat())))
        })
        .collect()
}

pub(super) fn swapped(pred: IntPred) -> IntPred {
    match pred {
        IntPred::Ult => IntPred::Ugt,
        IntPred::Ule => IntPred::Uge,
        IntPred::Ugt => IntPred::Ult,
        IntPred::Uge => IntPred::Ule,
        IntPred::Slt => IntPred::Sgt,
        IntPred::Sle => IntPred::Sge,
        IntPred::Sgt => IntPred::Slt,
        IntPred::Sge => IntPred::Sle,
        other => other,
    }
}

pub(super) fn offset(pred: IntPred, d: u32, bounds: (u64, u64)) -> Option<bool> {
    let signed = d as i32 as i64;
    let (low, high) = (bounds.0 as i64, bounds.1 as i64);
    let greater = if signed > 0 && high + signed < 1 << 32 {
        true
    } else if signed < 0 && low + signed >= 0 {
        false
    } else {
        return None;
    };
    let signed_ok = high + signed.max(0) < 1 << 31;
    match pred {
        IntPred::Ugt | IntPred::Uge => Some(greater),
        IntPred::Ult | IntPred::Ule => Some(!greater),
        IntPred::Sgt | IntPred::Sge if signed_ok => Some(greater),
        IntPred::Slt | IntPred::Sle if signed_ok => Some(!greater),
        _ => None,
    }
}

pub(super) fn decide(pred: IntPred, x: (u64, u64), y: (u64, u64)) -> Option<bool> {
    if x.0 == x.1 && y.0 == y.1 {
        return Some(compare(pred, x.0 as u32, y.0 as u32));
    }
    let signed = matches!(pred, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
    if signed && (x.1 >= 1 << 31 || y.1 >= 1 << 31) {
        return None;
    }
    let below = x.1 < y.0;
    let above = x.0 > y.1;
    let (le, ge) = (x.1 <= y.0, x.0 >= y.1);
    match pred {
        IntPred::Eq => (below || above).then_some(false),
        IntPred::Ne => (below || above).then_some(true),
        IntPred::Ult | IntPred::Slt => {
            if below {
                Some(true)
            } else if ge {
                Some(false)
            } else {
                None
            }
        }
        IntPred::Ule | IntPred::Sle => {
            if le {
                Some(true)
            } else if above {
                Some(false)
            } else {
                None
            }
        }
        IntPred::Ugt | IntPred::Sgt => {
            if above {
                Some(true)
            } else if le {
                Some(false)
            } else {
                None
            }
        }
        IntPred::Uge | IntPred::Sge => {
            if ge {
                Some(true)
            } else if below {
                Some(false)
            } else {
                None
            }
        }
    }
}

pub fn compare(pred: IntPred, x: u32, y: u32) -> bool {
    let (sx, sy) = (x as i32, y as i32);
    match pred {
        IntPred::Eq => x == y,
        IntPred::Ne => x != y,
        IntPred::Ult => x < y,
        IntPred::Ule => x <= y,
        IntPred::Ugt => x > y,
        IntPred::Uge => x >= y,
        IntPred::Slt => sx < sy,
        IntPred::Sle => sx <= sy,
        IntPred::Sgt => sx > sy,
        IntPred::Sge => sx >= sy,
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::testing::*;
    use super::*;

    const PREDICATES: [IntPred; 10] = [
        IntPred::Eq,
        IntPred::Ne,
        IntPred::Ult,
        IntPred::Ugt,
        IntPred::Ule,
        IntPred::Uge,
        IntPred::Slt,
        IntPred::Sgt,
        IntPred::Sle,
        IntPred::Sge,
    ];

    fn at(x: (u32, u32), t: u32) -> u32 {
        x.0.wrapping_add(x.1.wrapping_mul(t))
    }

    #[test]
    fn first_failure_names_an_iteration_that_leaves_the_loop() {
        let mut r = Random::new(5);
        for _ in 0..200000 {
            let pred = PREDICATES[r.below(10) as usize];
            let taken = r.below(2) == 0;
            let x = (interesting(&mut r), if r.below(2) == 0 { 0 } else { interesting(&mut r) });
            let y = (interesting(&mut r), if r.below(3) == 0 { interesting(&mut r) } else { 0 });
            if let Some(last) = first_failure(pred, taken, x, y) {
                assert_ne!(
                    compare(pred, at(x, last), at(y, last)),
                    taken,
                    "{:?} taken={} {:?} {:?}: the loop still runs after iteration {}",
                    pred, taken, x, y, last
                );
            }
        }
    }

    #[test]
    fn first_failure_names_the_first_iteration_that_leaves_the_loop() {
        let mut r = Random::new(9);
        for _ in 0..200000 {
            let pred = PREDICATES[r.below(10) as usize];
            let taken = r.below(2) == 0;
            let x = (interesting(&mut r), if r.below(2) == 0 { 0 } else { interesting(&mut r) });
            let y = (interesting(&mut r), if r.below(3) == 0 { interesting(&mut r) } else { 0 });
            if let Some(last) = first_failure(pred, taken, x, y) {
                if let Some(t) = (0..last.min(1 << 12)).find(|&t| compare(pred, at(x, t), at(y, t)) != taken) {
                    panic!(
                        "{:?} taken={} {:?} {:?}: the loop leaves after iteration {}, before {}",
                        pred, taken, x, y, t, last
                    );
                }
            }
        }
    }

    fn range(r: &mut Random) -> (u64, u64) {
        let low = interesting(r) as u64;
        let width = r.below(4);
        (low, (low + width).min(u32::MAX as u64))
    }

    #[test]
    fn decide_answers_only_what_holds_across_both_ranges() {
        let mut r = Random::new(13);
        for _ in 0..100000 {
            let pred = PREDICATES[r.below(10) as usize];
            let (x, y) = (range(&mut r), range(&mut r));
            if let Some(answer) = decide(pred, x, y) {
                for a in x.0..=x.1 {
                    for b in y.0..=y.1 {
                        assert_eq!(
                            compare(pred, a as u32, b as u32),
                            answer,
                            "{:?} over {:?} and {:?} at {} and {}",
                            pred, x, y, a, b
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn offset_answers_only_what_holds_for_every_value_in_range() {
        let mut r = Random::new(17);
        for _ in 0..100000 {
            let pred = PREDICATES[r.below(10) as usize];
            let y = range(&mut r);
            let d = interesting(&mut r);
            if let Some(answer) = offset(pred, d, y) {
                for b in y.0..=y.1 {
                    let b = b as u32;
                    assert_eq!(
                        compare(pred, b.wrapping_add(d), b),
                        answer,
                        "{:?}: y in {:?}, x = y + {:#x}, at y = {}",
                        pred, y, d, b
                    );
                }
            }
        }
    }
}
