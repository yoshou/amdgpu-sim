use super::check::interval;
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
