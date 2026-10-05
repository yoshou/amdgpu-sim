use super::super::logic::{Atom, Logic};
use super::program::Program;
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) struct Masks {
    masked: Vec<bool>,
    faithful: Vec<bool>,
}

impl Masks {
    pub(super) fn unsolved(n: usize) -> Self {
        Self {
            masked: vec![false; n],
            faithful: vec![false; n],
        }
    }

    pub(super) fn solve(program: &Program, logic: &mut Logic) -> Self {
        let mut this = Self::unsolved(program.f.types.len());
        this.settle(program, logic);
        this
    }

    #[inline]
    pub(super) fn masked(&self, v: ValueId) -> bool {
        self.masked[v.0]
    }

    #[inline]
    pub(super) fn faithful(&self, v: ValueId) -> bool {
        self.faithful[v.0]
    }

    fn settle(&mut self, program: &Program, logic: &mut Logic) {
        let (f, facts) = (program.f, program.facts);
        let n = f.types.len();
        let word = |v: ValueId| f.types[v.0] == Ty::I32 && facts.lane_word[v.0];
        let Some(ei) = program.exec_index else {
            self.masked = vec![true; n];
            self.faithful = (0..n).map(|v| word(ValueId(v))).collect();
            return;
        };
        for &b in &facts.order {
            let block = &f.blocks[&b];
            let exec = block.params[ei].0;
            for (index, &(param, ty)) in block.params.iter().enumerate() {
                let cleared = matches!(
                    program.inputs.get(index).map(|p| p.source),
                    Some(ParameterSource::MaskBit(_))
                );
                let assumed = b != f.entry || param == exec || cleared;
                self.masked[param.0] = assumed && (ty == Ty::I1 || word(param));
                self.faithful[param.0] = assumed && word(param);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &facts.order {
                let block = &f.blocks[&b];
                let exec = block.params[ei].0;
                let mut active = logic.atom(Atom::Bit(exec));
                for &(param, ty) in &block.params {
                    if param != exec && self.masked[param.0] {
                        let atom = if ty == Ty::I1 {
                            Atom::Bit(param)
                        } else {
                            Atom::View(param)
                        };
                        let known = logic.atom(atom);
                        active = logic.m.or(active, known);
                    }
                }
                let mut lockstep = HashMap::default();
                for inst in &block.insts {
                    for v in inst.outputs() {
                        let ty = f.types[v.0];
                        if ty != Ty::I1 && ty != Ty::I32 {
                            continue;
                        }
                        let formula = if ty == Ty::I1 {
                            logic.bit(f, facts, v)
                        } else {
                            logic.view(f, facts, v)
                        };
                        let masked = logic.m.implies(formula, active) || (ty == Ty::I1 && self.tests_a_masked_word(program, v));
                        if self.masked[v.0] != masked {
                            self.masked[v.0] = masked;
                            changed = true;
                        }
                        if word(v) {
                            let faithful =
                                self.lockstep_view(program, logic, v, active, &mut lockstep) == Some(formula);
                            if self.faithful[v.0] != faithful {
                                self.faithful[v.0] = faithful;
                                changed = true;
                            }
                        }
                    }
                }
            }
            for &b in &facts.order {
                if b == f.entry {
                    continue;
                }
                let block = &f.blocks[&b];
                let exec = block.params[ei].0;
                for (index, &(param, _)) in block.params.iter().enumerate() {
                    if param == exec {
                        continue;
                    }
                    if self.masked[param.0]
                        && !facts.arguments(f, b, index).all(|a| self.masked[a.0])
                    {
                        self.masked[param.0] = false;
                        changed = true;
                    }
                    if self.faithful[param.0]
                        && !facts
                            .arguments(f, b, index)
                            .all(|a| !word(a) || self.faithful[a.0])
                    {
                        self.faithful[param.0] = false;
                        changed = true;
                    }
                }
            }
        }
    }

    fn tests_a_masked_word(&self, program: &Program, v: ValueId) -> bool {
        let (f, facts) = (program.f, program.facts);
        let zero = |x: ValueId| facts.constant(f, x) == Some(0);
        match facts.op(f, v) {
            Some(Op::Cmp(IntPred::Ne, a, b)) if zero(b) => self.zero_off(program, a, 0),
            Some(Op::Cmp(IntPred::Ne, a, b)) if zero(a) => self.zero_off(program, b, 0),
            Some(Op::Cmp(IntPred::Ugt, a, b)) if zero(b) => self.zero_off(program, a, 0),
            Some(Op::Cmp(IntPred::Ult, a, b)) if zero(a) => self.zero_off(program, b, 0),
            _ => false,
        }
    }

    fn zero_off(&self, program: &Program, x: ValueId, depth: usize) -> bool {
        let (f, facts) = (program.f, program.facts);
        if facts.constant(f, x) == Some(0) {
            return true;
        }
        if depth > 16 {
            return false;
        }
        let next = |y: ValueId| self.zero_off(program, y, depth + 1);
        match facts.op(f, x) {
            Some(Op::Convert(Cvt::ZExt | Cvt::SExt, _, b)) if f.types[b.0] == Ty::I1 => self.masked[b.0],
            Some(Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, _, b)) => next(b),
            Some(Op::Int(IntOp::Mul | IntOp::And, a, b)) => next(a) || next(b),
            Some(Op::Int(IntOp::Shl | IntOp::LShr | IntOp::AShr, a, _)) => next(a),
            Some(Op::Int(IntOp::Add | IntOp::Sub | IntOp::Or | IntOp::Xor, a, b)) => next(a) && next(b),
            Some(Op::Select(k, a, b)) => (self.masked[k.0] && next(b)) || (next(a) && next(b)),
            Some(_) => false,
            None => match facts.site[x.0] {
                Site::Param { block, index } if block != f.entry && Some(index) != program.exec_index => {
                    facts.arguments(f, block, index).all(next)
                }
                _ => false,
            },
        }
    }

    fn lockstep_view(
        &self,
        program: &Program,
        logic: &mut Logic,
        w: ValueId,
        active: Bdd,
        memo: &mut HashMap<ValueId, Option<Bdd>>,
    ) -> Option<Bdd> {
        if let Some(&l) = memo.get(&w) {
            return l;
        }
        let (f, facts) = (program.f, program.facts);
        let l = if !facts.lane_word[w.0] {
            Some(logic.view(f, facts, w))
        } else {
            match facts.inst(f, w) {
                None => self.faithful[w.0].then(|| logic.view(f, facts, w)),
                Some(Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Ballot),
                    inputs,
                    ..
                }) => {
                    let x = logic.bit(f, facts, inputs[0]);
                    Some(logic.m.and(x, active))
                }
                Some(Inst::Core { op, .. }) => match *op {
                    Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                        match (
                            self.lockstep_view(program, logic, a, active, memo),
                            self.lockstep_view(program, logic, b, active, memo),
                        ) {
                            (Some(a), Some(b)) => Some(match k {
                                IntOp::And => logic.m.and(a, b),
                                IntOp::Or => logic.m.or(a, b),
                                _ => logic.m.xor(a, b),
                            }),
                            _ => None,
                        }
                    }
                    Op::Select(c, a, b) => {
                        match (
                            self.lockstep_view(program, logic, a, active, memo),
                            self.lockstep_view(program, logic, b, active, memo),
                        ) {
                            (Some(a), Some(b)) => {
                                let c = logic.bit(f, facts, c);
                                Some(logic.m.ite(c, a, b))
                            }
                            _ => None,
                        }
                    }
                    Op::Convert(Cvt::Bitcast, Ty::I32, a) => self.lockstep_view(program, logic, a, active, memo),
                    _ => Some(logic.view(f, facts, w)),
                },
                Some(_) => None,
            }
        };
        memo.insert(w, l);
        l
    }
}
