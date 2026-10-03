use super::check::interval;
use super::encoding::{negated, outcomes, relation_options, WORD};
use super::linear::{Linear, Problem, Var};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) struct Terms<'a> {
    f: &'a Func,
    facts: &'a Facts,
    problem: Problem,
    values: HashMap<ValueId, Var>,
    words: HashMap<ValueId, Var>,
    lane: Option<[Var; 5]>,
    signs: HashMap<Var, Var>,
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
            signs: HashMap::default(),
        }
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
        let bits = match self.lane {
            Some(bits) => bits,
            None => {
                let bits = [0; 5].map(|_| self.problem.between(0, 1));
                self.lane = Some(bits);
                bits
            }
        };
        let mut e = Linear::constant(0);
        for (i, &b) in bits.iter().enumerate() {
            e.add(b, 1 << i);
        }
        e
    }

    fn linear(&mut self, v: ValueId, depth: usize) -> Linear {
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
        let w = self.problem.between(0, WORD - 1);
        let m = self.problem.free();
        self.problem.equal(e.term(m, -WORD).term(w, -1));
        self.words.insert(v, w);
        w
    }

    pub(super) fn decided(&mut self, pred: IntPred, x: ValueId, y: ValueId) -> Option<bool> {
        let (wx, wy) = (self.word(x), self.word(y));
        let mut answers = [None; 2];
        for (k, p) in [pred, negated(pred)].iter().copied().enumerate() {
            let options = relation_options(&mut self.problem, &mut self.signs, outcomes(p), (wx, wy));
            let mut problem = self.problem.clone();
            problem.either.push(options);
            answers[k] = problem.feasible();
        }
        match answers {
            [Some(false), Some(true)] => Some(false),
            [Some(true), Some(false)] => Some(true),
            _ => None,
        }
    }
}
