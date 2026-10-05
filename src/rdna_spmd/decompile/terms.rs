use super::encoding::{negated, outcomes, relation_options, WORD};
use super::linear::{Linear, Problem, Var};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

#[derive(Clone)]
pub(super) struct Terms<'a> {
    f: &'a Func,
    facts: &'a Facts,
    problem: Problem,
    values: HashMap<ValueId, Var>,
    words: HashMap<ValueId, Var>,
    lane: Option<Var>,
    substitution: HashMap<ValueId, ValueId>,
}

impl<'a> Terms<'a> {
    pub(super) fn new(f: &'a Func, facts: &'a Facts) -> Self {
        Terms {
            f,
            facts,
            problem: Problem::default(),
            values: HashMap::default(),
            words: HashMap::default(),
            lane: None,
            substitution: HashMap::default(),
        }
    }

    pub(super) fn substituting(f: &'a Func, facts: &'a Facts, substitution: HashMap<ValueId, ValueId>) -> Self {
        let mut terms = Self::new(f, facts);
        terms.substitution = substitution;
        terms
    }

    fn opaque(&mut self, v: ValueId) -> Linear {
        let x = match self.values.get(&v) {
            Some(&x) => x,
            None => {
                let (low, high) = interval(self.f, self.facts, v, 0).unwrap_or((0, u32::MAX));
                let x = self.problem.between(low as i128, high as i128);
                self.values.insert(v, x);
                x
            }
        };
        Linear::constant(0).term(x, 1)
    }

    fn lane_id(&mut self) -> Linear {
        let lane = match self.lane {
            Some(lane) => lane,
            None => {
                let lane = self.problem.between(0, 31);
                self.lane = Some(lane);
                lane
            }
        };
        Linear::constant(0).term(lane, 1)
    }

    fn linear(&mut self, v: ValueId, depth: usize) -> Linear {
        let v = self.substitution.get(&v).copied().unwrap_or(v);
        if self.f.types[v.0] != Ty::I32 || depth > 32 {
            return self.opaque(v);
        }
        match self.facts.op(self.f, v) {
            Some(Op::Const(_, k)) => Linear::constant(k as u32 as i128),
            Some(Op::Env(Env::LaneId)) => self.lane_id(),
            Some(Op::Int(IntOp::Add, a, b)) => self.linear(a, depth + 1).plus(&self.linear(b, depth + 1), 1),
            Some(Op::Int(IntOp::Sub, a, b)) => self.linear(a, depth + 1).plus(&self.linear(b, depth + 1), -1),
            Some(Op::Int(IntOp::Mul, a, b)) => match (self.facts.constant(self.f, a), self.facts.constant(self.f, b)) {
                (_, Some(k)) => Linear::constant(0).plus(&self.linear(a, depth + 1), k as u32 as i128),
                (Some(k), _) => Linear::constant(0).plus(&self.linear(b, depth + 1), k as u32 as i128),
                _ => self.opaque(v),
            },
            Some(Op::Int(IntOp::Shl, a, s)) => match self.facts.constant(self.f, s) {
                Some(k) if k < 32 => Linear::constant(0).plus(&self.linear(a, depth + 1), 1 << k),
                _ => self.opaque(v),
            },
            _ => self.opaque(v),
        }
    }

    fn word(&mut self, v: ValueId) -> Var {
        if let Some(&w) = self.words.get(&v) {
            return w;
        }
        let e = self.linear(v, 0);
        let w = match self.range(&e) {
            Some((lo, hi)) if lo >= 0 && hi < WORD => match e.terms.as_slice() {
                [(x, 1)] if e.constant == 0 => *x,
                _ => {
                    let w = self.problem.between(lo, hi);
                    self.problem.equal(e.term(w, -1));
                    w
                }
            },
            range => {
                let w = self.problem.between(0, WORD - 1);
                let m = match range {
                    Some((lo, hi)) => self.problem.between(lo.div_euclid(WORD), hi.div_euclid(WORD)),
                    None => self.problem.free(),
                };
                self.problem.equal(e.term(m, -WORD).term(w, -1));
                w
            }
        };
        self.words.insert(v, w);
        w
    }

    fn range(&self, e: &Linear) -> Option<(i128, i128)> {
        let (mut lo, mut hi) = (e.constant, e.constant);
        for &(x, c) in &e.terms {
            let (Some(l), Some(h)) = self.problem.domains[x].hull()? else {
                return None;
            };
            let (a, b) = (c * l, c * h);
            lo += a.min(b);
            hi += a.max(b);
        }
        Some((lo, hi))
    }

    pub(super) fn leaves(&self, v: ValueId, depth: usize, out: &mut BTreeSet<ValueId>, lane: &mut bool) {
        let v = self.substitution.get(&v).copied().unwrap_or(v);
        if self.f.types[v.0] != Ty::I32 || depth > 32 {
            out.insert(v);
            return;
        }
        match self.facts.op(self.f, v) {
            Some(Op::Const(..)) => {}
            Some(Op::Env(Env::LaneId)) => *lane = true,
            Some(Op::Int(IntOp::Add | IntOp::Sub, a, b)) => {
                self.leaves(a, depth + 1, out, lane);
                self.leaves(b, depth + 1, out, lane);
            }
            Some(Op::Int(IntOp::Mul, a, b)) => match (self.facts.constant(self.f, a), self.facts.constant(self.f, b)) {
                (_, Some(_)) => self.leaves(a, depth + 1, out, lane),
                (Some(_), _) => self.leaves(b, depth + 1, out, lane),
                _ => {
                    out.insert(v);
                }
            },
            Some(Op::Int(IntOp::Shl, a, s)) => match self.facts.constant(self.f, s) {
                Some(k) if k < 32 => self.leaves(a, depth + 1, out, lane),
                _ => {
                    out.insert(v);
                }
            },
            _ => {
                out.insert(v);
            }
        }
    }

    pub(super) fn worlds(&mut self, preds: &[(IntPred, ValueId, ValueId)], limit: usize) -> Option<Vec<Vec<bool>>> {
        for &(_, x, y) in preds {
            self.word(x);
            self.word(y);
        }
        let mut out = BTreeSet::new();
        let mut chosen = Vec::with_capacity(preds.len());
        self.enumerate(preds, &mut chosen, &mut out, limit)?;
        Some(out.into_iter().collect())
    }

    fn enumerate(&mut self, preds: &[(IntPred, ValueId, ValueId)], chosen: &mut Vec<bool>, out: &mut BTreeSet<Vec<bool>>, limit: usize) -> Option<()> {
        if chosen.len() == preds.len() {
            if out.insert(chosen.clone()) && out.len() > limit {
                return None;
            }
            return Some(());
        }
        let (pred, x, y) = preds[chosen.len()];
        let (wx, wy) = (self.word(x), self.word(y));
        for value in [true, false] {
            let p = if value { pred } else { negated(pred) };
            for clause in relation_options(outcomes(p), (wx, wy)) {
                let mark = (self.problem.equal.len(), self.problem.at_least.len());
                self.problem.equal.extend(clause.equal);
                self.problem.at_least.extend(clause.at_least);
                let open = self.problem.feasible() != Some(false);
                let mut r = Some(());
                if open {
                    chosen.push(value);
                    r = self.enumerate(preds, chosen, out, limit);
                    chosen.pop();
                }
                self.problem.equal.truncate(mark.0);
                self.problem.at_least.truncate(mark.1);
                r?;
            }
        }
        Some(())
    }

    pub(super) fn uniform(&mut self, (pred, x, y): (IntPred, ValueId, ValueId)) -> bool {
        let (wx, wy) = (self.word(x), self.word(y));
        let mut other = self.clone();
        other.words.clear();
        other.lane = None;
        other.values.retain(|v, _| self.facts.uniform[v.0]);
        let (ox, oy) = (other.word(x), other.word(y));
        let mut problem = other.problem;
        problem.either.push(relation_options(outcomes(pred), (wx, wy)));
        problem.either.push(relation_options(outcomes(negated(pred)), (ox, oy)));
        problem.feasible() == Some(false)
    }

    pub(super) fn decided(&mut self, pred: IntPred, x: ValueId, y: ValueId) -> Option<bool> {
        let (wx, wy) = (self.word(x), self.word(y));
        let mut answers = [None; 2];
        for (k, p) in [pred, negated(pred)].iter().copied().enumerate() {
            let mut problem = self.problem.clone();
            problem.either.push(relation_options(outcomes(p), (wx, wy)));
            answers[k] = problem.feasible();
        }
        match answers {
            [Some(false), Some(true)] => Some(false),
            [Some(true), Some(false)] => Some(true),
            _ => None,
        }
    }
}

pub(super) fn interval(f: &Func, facts: &Facts, v: ValueId, depth: usize) -> Option<(u32, u32)> {
    if depth > 16 {
        return None;
    }
    if let Some(k) = facts.constant(f, v) {
        return Some((k as u32, k as u32));
    }
    let full = (0, u32::MAX);
    let range = |x: ValueId| interval(f, facts, x, depth + 1);
    match facts.inst(f, v)? {
        Inst::Core { op, .. } => match *op {
            Op::Env(Env::LaneId) => Some((0, 31)),
            Op::Int(IntOp::And, a, b) => {
                let (x, y) = (range(a).unwrap_or(full), range(b).unwrap_or(full));
                Some((0, x.1.min(y.1)))
            }
            Op::Int(IntOp::Or, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                let top = x.1.max(y.1);
                let ones = if top == 0 { 0 } else { u32::MAX >> top.leading_zeros() };
                Some((x.0.max(y.0), ones))
            }
            Op::Int(IntOp::Add, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                let high = x.1 as u64 + y.1 as u64;
                (high <= u32::MAX as u64).then(|| (x.0 + y.0, high as u32))
            }
            Op::Int(IntOp::Sub, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                (x.0 >= y.1).then(|| (x.0 - y.1, x.1 - y.0))
            }
            Op::Int(IntOp::Mul, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                let high = x.1 as u64 * y.1 as u64;
                (high <= u32::MAX as u64).then(|| (x.0 * y.0, high as u32))
            }
            Op::Int(IntOp::Shl, a, s) => {
                let k = facts.constant(f, s)? as u32 & 31;
                let x = range(a)?;
                let high = (x.1 as u64) << k;
                (high <= u32::MAX as u64).then(|| (x.0 << k, high as u32))
            }
            Op::Int(IntOp::Xor, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                let top = x.1.max(y.1);
                Some((0, if top == 0 { 0 } else { u32::MAX >> top.leading_zeros() }))
            }
            Op::Int(IntOp::LShr, a, s) => {
                let k = facts.constant(f, s)? as u32 & 31;
                let x = range(a).unwrap_or(full);
                Some((x.0 >> k, x.1 >> k))
            }
            Op::Int(IntOp::AShr, a, s) => {
                let k = facts.constant(f, s)? as u32 & 31;
                let x = range(a)?;
                (x.1 < 1 << 31).then(|| (x.0 >> k, x.1 >> k))
            }
            Op::Select(_, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                Some((x.0.min(y.0), x.1.max(y.1)))
            }
            Op::Convert(Cvt::ZExt, Ty::I32, a) if f.types[a.0] == Ty::I1 => Some((0, 1)),
            Op::PopulationCount(_) | Op::LeadingZeros(_) | Op::TrailingZeros(_) => Some((0, 32)),
            _ => None,
        },
        Inst::Effect {
            op: EffectOp::Memory { op: MemoryOp::Load(size), .. },
            ..
        } => match size {
            MemSize::U8 => Some((0, 0xff)),
            MemSize::U16 => Some((0, 0xffff)),
            _ => None,
        },
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    #[test]
    fn interval_holds_every_value_the_operations_give() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let byte = b.load(e, Space::Global, MemSize::U8, table, yes);
        let half = b.load(e, Space::Global, MemSize::U16, table, yes);
        let word = b.load(e, Space::Global, MemSize::B32, table, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let c = |b: &mut Build, k: u64| b.constant(e, Ty::I32, k);
        let (three, five, forty) = (c(&mut b, 3), c(&mut b, 5), c(&mut b, 40));
        let mut cases: Vec<(&str, ValueId, Box<dyn Fn(u32, u32, u32, u32) -> u32>)> = Vec::new();
        let mut optional: Vec<(&str, ValueId, Box<dyn Fn(u32, u32, u32, u32) -> u32>)> = Vec::new();
        let v = b.int(e, IntOp::And, word, forty);
        cases.push(("w & 40", v, Box::new(|_, _, w, _| w & 40)));
        let v = b.int(e, IntOp::Or, byte, three);
        cases.push(("b | 3", v, Box::new(|x, _, _, _| x | 3)));
        let v = b.int(e, IntOp::Add, half, lane);
        cases.push(("h + lane", v, Box::new(|_, h, _, l| h + l)));
        let v = b.int(e, IntOp::LShr, half, five);
        cases.push(("h >> 5", v, Box::new(|_, h, _, _| h >> 5)));
        let v = b.int(e, IntOp::LShr, word, forty);
        cases.push(("w >> 40", v, Box::new(|_, _, w, _| w >> (40 & 31))));
        let small = b.cmp(e, IntPred::Ult, word, forty);
        let v = b.core(e, Ty::I32, Op::Select(small, byte, five));
        cases.push(("select(w < 40, b, 5)", v, Box::new(|x, _, w, _| if w < 40 { x } else { 5 })));
        let v = b.core(e, Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, small));
        cases.push(("zext(w < 40)", v, Box::new(|_, _, w, _| (w < 40) as u32)));
        let v = b.core(e, Ty::I32, Op::PopulationCount(word));
        cases.push(("popcount(w)", v, Box::new(|_, _, w, _| w.count_ones())));
        let v = b.core(e, Ty::I32, Op::TrailingZeros(word));
        cases.push(("trailing zeros of w", v, Box::new(|_, _, w, _| w.trailing_zeros())));
        let bit = c(&mut b, 0x100);
        let top = b.int(e, IntOp::Or, half, bit);
        let v = b.int(e, IntOp::Sub, top, byte);
        cases.push(("(h | 256) - b", v, Box::new(|x, h, _, _| (h | 256).wrapping_sub(x))));
        let v = b.int(e, IntOp::Mul, half, forty);
        cases.push(("h * 40", v, Box::new(|_, h, _, _| h.wrapping_mul(40))));
        let v = b.int(e, IntOp::Mul, byte, lane);
        cases.push(("b * lane", v, Box::new(|x, _, _, l| x.wrapping_mul(l))));
        let v = b.int(e, IntOp::Shl, byte, five);
        cases.push(("b << 5", v, Box::new(|x, _, _, _| x << 5)));
        let v = b.int(e, IntOp::Shl, half, forty);
        cases.push(("h << 40", v, Box::new(|_, h, _, _| h << (40 & 31))));
        let v = b.int(e, IntOp::Xor, byte, lane);
        cases.push(("b ^ lane", v, Box::new(|x, _, _, l| x ^ l)));
        let v = b.int(e, IntOp::AShr, half, three);
        cases.push(("h >>> 3", v, Box::new(|_, h, _, _| ((h as i32) >> 3) as u32)));
        let v = b.int(e, IntOp::Sub, byte, half);
        optional.push(("b - h", v, Box::new(|x, h, _, _| x.wrapping_sub(h))));
        let big = c(&mut b, 0x2_0000);
        let v = b.int(e, IntOp::Mul, half, big);
        optional.push(("h * 2^17", v, Box::new(|_, h, _, _| h.wrapping_mul(0x2_0000))));
        let seventeen = c(&mut b, 17);
        let v = b.int(e, IntOp::Shl, half, seventeen);
        optional.push(("h << 17", v, Box::new(|_, h, _, _| h << 17)));
        let v = b.int(e, IntOp::AShr, word, three);
        optional.push(("w >>> 3", v, Box::new(|_, _, w, _| ((w as i32) >> 3) as u32)));
        let one = c(&mut b, 1);
        let bit = b.int(e, IntOp::And, byte, one);
        let small = b.int(e, IntOp::Add, bit, one);
        let odd = c(&mut b, 0x8000_0001);
        let v = b.int(e, IntOp::Mul, small, odd);
        optional.push(("((b & 1) + 1) * 0x80000001", v, Box::new(|x, _, _, _| ((x & 1) + 1).wrapping_mul(0x8000_0001))));
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let mut r = Random::new(37);
        let mut wrong = Vec::new();
        let required = cases.len();
        cases.extend(optional);
        for (i, (name, v, truth)) in cases.iter().enumerate() {
            let Some((low, high)) = interval(&b.f, &facts, *v, 0) else {
                if i < required {
                    wrong.push(format!("{} has no interval", name));
                }
                continue;
            };
            for _ in 0..2000 {
                let w = match r.below(3) {
                    0 => r.below(64) as u32,
                    1 => u32::MAX - r.below(64) as u32,
                    _ => r.next() as u32,
                };
                let t = truth((w & 0xff) as u32, w & 0xffff, w, r.below(32) as u32);
                if t < low || t > high {
                    wrong.push(format!("{} is {} outside [{}, {}]", name, t, low, high));
                    break;
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", wrong);
    }
}
