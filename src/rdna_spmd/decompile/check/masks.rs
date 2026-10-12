use super::super::logic::{Atom, Logic};
use super::program::Program;
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

type Reads = Vec<(ValueId, bool, bool)>;

#[derive(Clone)]
pub(super) struct Masks {
    masked: Vec<bool>,
    faithful: Vec<bool>,
}

fn any_of(logic: &mut Logic, atoms: impl Iterator<Item = Atom>) -> Bdd {
    let mut literals: Vec<(u32, Bdd)> = atoms
        .map(|atom| {
            let literal = logic.atom(atom);
            (logic.m.decompose(literal).unwrap().0, literal)
        })
        .collect();
    literals.sort_unstable_by_key(|&(var, _)| std::cmp::Reverse(var));
    literals.into_iter().fold(Bdd::FALSE, |acc, (_, literal)| logic.m.or(literal, acc))
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

    fn read(&self, v: ValueId, reads: &mut Reads) -> (bool, bool) {
        let flags = (self.masked[v.0], self.faithful[v.0]);
        reads.push((v, flags.0, flags.1));
        flags
    }

    fn unchanged(&self, reads: &Reads) -> bool {
        reads.iter().all(|&(v, masked, faithful)| self.masked[v.0] == masked && self.faithful[v.0] == faithful)
    }

    fn settle(&mut self, program: &Program, logic: &mut Logic) {
        let (f, facts) = (program.f, program.facts);
        let n = f.types.len();
        let word = |v: ValueId| crate::rdna_spmd::analysis::facts::is_word(f.types[v.0]) && facts.lane_word[v.0];
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
        let mut argument = vec![false; n];
        for block in f.blocks.values() {
            for edge in block.term.edges() {
                for &a in &edge.args {
                    argument[a.0] = true;
                }
            }
        }
        let mut actives: Vec<Option<(Vec<ValueId>, Bdd)>> = vec![None; facts.order.len()];
        let mut seen: Vec<Option<Reads>> = vec![None; facts.order.len()];
        let mut changed = true;
        while changed {
            changed = false;
            for (rank, &b) in facts.order.iter().enumerate() {
                if seen[rank].as_ref().is_some_and(|reads| self.unchanged(reads)) {
                    continue;
                }
                let mut reads = Reads::new();
                let block = &f.blocks[&b];
                let exec = block.params[ei].0;
                let known: Vec<ValueId> = block
                    .params
                    .iter()
                    .map(|&(param, _)| param)
                    .filter(|&param| param == exec || self.read(param, &mut reads).0)
                    .collect();
                let active = match &actives[rank] {
                    Some((same, active)) if *same == known => *active,
                    _ => {
                        let atoms = known.iter().map(|&param| {
                            if param == exec || f.types[param.0] == Ty::I1 {
                                Atom::Bit(param)
                            } else {
                                Atom::View(param)
                            }
                        });
                        let active = any_of(logic, atoms);
                        actives[rank] = Some((known, active));
                        active
                    }
                };
                let mut lockstep = HashMap::default();
                for inst in &block.insts {
                    for v in inst.outputs() {
                        let ty = f.types[v.0];
                        if ty != Ty::I1 && !crate::rdna_spmd::analysis::facts::is_word(ty) {
                            continue;
                        }
                        if ty != Ty::I1 && !word(v) && !argument[v.0] {
                            continue;
                        }
                        let formula = if ty == Ty::I1 {
                            logic.bit(f, facts, v)
                        } else {
                            logic.view(f, facts, v)
                        };
                        let masked = logic.m.implies(formula, active) || (ty == Ty::I1 && self.tests_a_masked_word(program, v, &mut reads));
                        if self.masked[v.0] != masked {
                            self.masked[v.0] = masked;
                            changed = true;
                        }
                        if word(v) {
                            let faithful =
                                self.lockstep_view(program, logic, v, active, &mut lockstep, &mut reads) == Some(formula);
                            if self.faithful[v.0] != faithful {
                                self.faithful[v.0] = faithful;
                                changed = true;
                            }
                        }
                    }
                }
                reads.sort_unstable();
                reads.dedup();
                seen[rank] = Some(reads);
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

    fn tests_a_masked_word(&self, program: &Program, v: ValueId, reads: &mut Reads) -> bool {
        let (f, facts) = (program.f, program.facts);
        let zero = |x: ValueId| facts.constant(f, x) == Some(0);
        match facts.op(f, v) {
            Some(Op::Cmp(IntPred::Ne, a, b)) if zero(b) => self.zero_off(program, a, 0, reads),
            Some(Op::Cmp(IntPred::Ne, a, b)) if zero(a) => self.zero_off(program, b, 0, reads),
            Some(Op::Cmp(IntPred::Ugt, a, b)) if zero(b) => self.zero_off(program, a, 0, reads),
            Some(Op::Cmp(IntPred::Ult, a, b)) if zero(a) => self.zero_off(program, b, 0, reads),
            _ => false,
        }
    }

    fn zero_off(&self, program: &Program, x: ValueId, depth: usize, reads: &mut Reads) -> bool {
        let (f, facts) = (program.f, program.facts);
        if facts.constant(f, x) == Some(0) {
            return true;
        }
        if depth > 16 {
            return false;
        }
        let next = |y: ValueId, reads: &mut Reads| self.zero_off(program, y, depth + 1, reads);
        match facts.op(f, x) {
            Some(Op::Convert(Cvt::ZExt | Cvt::SExt, _, b)) if f.types[b.0] == Ty::I1 => self.read(b, reads).0,
            Some(Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, _, b)) => next(b, reads),
            Some(Op::Int(IntOp::Mul | IntOp::And, a, b)) => next(a, reads) || next(b, reads),
            Some(Op::Int(IntOp::Shl | IntOp::LShr | IntOp::AShr, a, _)) => next(a, reads),
            Some(Op::Int(IntOp::Add | IntOp::Sub | IntOp::Or | IntOp::Xor, a, b)) => next(a, reads) && next(b, reads),
            Some(Op::Select(k, a, b)) => (self.read(k, reads).0 && next(b, reads)) || (next(a, reads) && next(b, reads)),
            Some(_) => false,
            None => match facts.site[x.0] {
                Site::Param { block, index } if block != f.entry && Some(index) != program.exec_index => {
                    facts.arguments(f, block, index).all(|a| next(a, reads))
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
        reads: &mut Reads,
    ) -> Option<Bdd> {
        if let Some(&l) = memo.get(&w) {
            return l;
        }
        let (f, facts) = (program.f, program.facts);
        let l = if !facts.lane_word[w.0] {
            Some(logic.view(f, facts, w))
        } else {
            match facts.inst(f, w) {
                None => self.read(w, reads).1.then(|| logic.view(f, facts, w)),
                Some(Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Ballot { high }),
                    inputs,
                    ..
                }) => {
                    let x = logic.bit(f, facts, inputs[0]);
                    let own = logic.m.and(x, active);
                    let other = logic.atom(Atom::View(w));
                    Some(logic.half(*high, own, other))
                }
                Some(Inst::Core { op, .. }) => match *op {
                    Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                        match (
                            self.lockstep_view(program, logic, a, active, memo, reads),
                            self.lockstep_view(program, logic, b, active, memo, reads),
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
                            self.lockstep_view(program, logic, a, active, memo, reads),
                            self.lockstep_view(program, logic, b, active, memo, reads),
                        ) {
                            (Some(a), Some(b)) => {
                                let c = logic.bit(f, facts, c);
                                Some(logic.m.ite(c, a, b))
                            }
                            _ => None,
                        }
                    }
                    Op::Convert(Cvt::Bitcast, Ty::I32 | Ty::I64, a) => self.lockstep_view(program, logic, a, active, memo, reads),
                    Op::Pack64(a, b) => {
                        let low = self.lockstep_view(program, logic, a, active, memo, reads);
                        if f.lanes == 32 {
                            low
                        } else {
                            match (low, self.lockstep_view(program, logic, b, active, memo, reads)) {
                                (Some(low), Some(high)) => {
                                    let upper = logic.atom(Atom::Lane(5));
                                    Some(logic.m.ite(upper, high, low))
                                }
                                _ => None,
                            }
                        }
                    }
                    Op::UnpackLo(a) => self.lockstep_view(program, logic, a, active, memo, reads).map(|own| {
                        let other = logic.atom(Atom::View(w));
                        logic.half(false, own, other)
                    }),
                    Op::UnpackHi(a) if f.lanes == 64 => self.lockstep_view(program, logic, a, active, memo, reads).map(|own| {
                        let other = logic.atom(Atom::View(w));
                        logic.half(true, own, other)
                    }),
                    _ => Some(logic.view(f, facts, w)),
                },
                Some(_) => None,
            }
        };
        memo.insert(w, l);
        l
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::testing::*;
    use super::*;
    use crate::rdna_spmd::analysis::facts::Facts;
    use std::collections::BTreeSet;

    #[test]
    fn any_of_is_the_disjunction_of_its_atoms_in_any_order() {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
        let types = [Ty::I1, Ty::I32, Ty::I1, Ty::I1, Ty::I32, Ty::I1, Ty::I32, Ty::I1];
        let (next, params) = b.block(&types);
        let zero = b.constant(BlockId(0), Ty::I32, 0);
        let args = types.iter().map(|&ty| if ty == Ty::I1 { p[0] } else { zero }).collect();
        b.br(BlockId(0), next, args);
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let pool: Vec<Atom> = params
            .iter()
            .zip(&types)
            .map(|(&v, &ty)| if ty == Ty::I1 { Atom::Bit(v) } else { Atom::View(v) })
            .chain([Atom::Lane(0), Atom::Lane(5)])
            .collect();
        let mut r = Random::new(5);
        for _ in 0..300 {
            let mut atoms: Vec<Atom> = pool.iter().copied().filter(|_| r.below(2) == 0).collect();
            for i in (1..atoms.len()).rev() {
                atoms.swap(i, r.below(i as u64 + 1) as usize);
            }
            let mut want = Bdd::FALSE;
            for &a in &atoms {
                let x = logic.atom(a);
                want = logic.m.or(want, x);
            }
            assert_eq!(any_of(&mut logic, atoms.iter().copied()), want, "{:?}", atoms);
        }
    }
}
