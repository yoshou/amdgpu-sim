use super::super::super::logic::{Atom, PATH};
use super::super::queries::Queries;
use super::joint::WAVE;
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) struct Decisions {
    assume: Bdd,
    unreliable: HashMap<u32, bool>,
    decided: HashMap<(Bdd, Bdd, usize), Option<bool>>,
}

impl Decisions {
    pub(super) fn new(assume: Bdd) -> Self {
        Self {
            assume,
            unreliable: HashMap::default(),
            decided: HashMap::default(),
        }
    }

    #[inline]
    pub(super) fn assume(&self) -> Bdd {
        self.assume
    }

    pub(super) fn reliable<'a, Q: Queries<'a>>(&self, q: &mut Q, leaves: Bdd) -> bool {
        if leaves == Bdd::FALSE {
            return true;
        }
        let assume = q.and(self.assume, q.safe());
        q.and(assume, leaves) == Bdd::FALSE
    }

    fn is_unreliable<'a, Q: Queries<'a>>(&mut self, q: &mut Q, var: u32) -> bool {
        if let Some(&u) = self.unreliable.get(&var) {
            return u;
        }
        let assume = self.assume;
        let differs = |q: &mut Q, v: ValueId| {
            let h = q.h(v);
            h != Bdd::FALSE && q.and(assume, h) != Bdd::FALSE
        };
        let u = match q.logic().atom_of(var) {
            Atom::Bit(v) | Atom::View(v) | Atom::WordBit(v, _) => differs(q, v),
            Atom::Cell(block, group, _) => {
                let leaves: Vec<ValueId> = q.logic().cell_leaves(block, group).to_vec();
                leaves.iter().any(|&v| differs(q, v))
            }
            Atom::Some(block, _) => {
                let sources = q.logic().answer_sources(block);
                sources.iter().any(|&v| differs(q, v))
            }
            _ => false,
        };
        self.unreliable.insert(var, u);
        u
    }

    fn unknowns<'a, Q: Queries<'a>>(&mut self, q: &mut Q, g: Bdd, side: usize, forms: bool) -> Vec<u32> {
        let support = q.logic().support(g);
        support
            .iter()
            .copied()
            .filter(|&v| match q.logic().atom_of(v) {
                Atom::Fresh(PATH, ..) => false,
                Atom::Fresh(..) => true,
                Atom::Term(..) => forms,
                _ => side == WAVE && self.is_unreliable(q, v),
            })
            .collect()
    }

    pub(super) fn decide<'a, Q: Queries<'a>>(&mut self, q: &mut Q, g: Bdd, cond: Bdd, side: usize) -> Option<bool> {
        if let Some(k) = g.constant() {
            return Some(k);
        }
        let cond = q.and(cond, q.safe());
        if let Some(&d) = self.decided.get(&(g, cond, side)) {
            return d;
        }
        let unknown = self.unknowns(q, g, side, true);
        let holds = q.logic().forall(&unknown, g);
        let d = if q.logic().m.implies(cond, holds) {
            Some(true)
        } else {
            let ng = q.logic().m.not(g);
            let fails = q.logic().forall(&unknown, ng);
            if q.logic().m.implies(cond, fails) {
                Some(false)
            } else {
                None
            }
        };
        self.decided.insert((g, cond, side), d);
        d
    }

    pub(super) fn weaken<'a, Q: Queries<'a>>(&mut self, q: &mut Q, g: Bdd, side: usize) -> Bdd {
        let unknown = self.unknowns(q, g, side, false);
        q.logic().exists(&unknown, g)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::super::hazard::Hazards;
    use super::super::super::super::logic::Logic;
    use super::super::super::super::testing::*;
    use super::super::super::differences::Differences;
    use super::super::super::program::Program;
    use super::super::super::Mode;
    use super::super::joint::{JOINT, LANE, PATHS};
    use super::*;
    use crate::rdna_spmd::analysis::facts::Facts;
    use crate::rdna_spmd::analysis::loops::Loops;
    use std::collections::BTreeSet;

    fn with_differences(test: impl FnOnce(&mut Differences)) {
        let (b, _) = Build::kernel();
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let loops = Loops::new(f, &facts).unwrap();
        let hazards = Hazards {
            accesses: Vec::new(),
            together: BTreeSet::new(),
            apart: BTreeSet::new(),
            idle: BTreeSet::new(),
            meetings: Vec::new(),
        };
        let logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let program = Program::new(f, &facts, &b.inputs, Some(0), &loops, &hazards);
        let mut differences = Differences::new(program, logic, Mode::Search);
        test(&mut differences);
    }

    #[test]
    fn decide_and_weaken_answer_over_the_paths_the_condition_allows() {
        with_differences(|q| {
            let mut decisions = Decisions::new(Bdd::TRUE);
            let mut r = Random::new(53);
            let context = [Atom::Lane(0), Atom::Lane(1)];
            let unknowns = [Atom::Fresh(LANE, ValueId(1001), 0), Atom::Fresh(JOINT, ValueId(0), 1)];
            let paths = [Atom::Fresh(PATH, ValueId(0), PATHS - 1), Atom::Fresh(PATH, ValueId(0), PATHS)];
            let tied: Vec<Atom> = context.iter().chain(&paths).copied().collect();
            let all: Vec<Atom> = tied.iter().chain(&unknowns).copied().collect();
            let tied_vars: Vec<u32> = tied.iter().map(|&a| variable(q.logic(), a)).collect();
            let unknown_vars: Vec<u32> = unknowns.iter().map(|&a| variable(q.logic(), a)).collect();
            let mut wrong = Vec::new();
            for trial in 0..400 {
                let cond = random_function(q.logic(), &mut r, &tied, 3);
                if cond == Bdd::FALSE {
                    continue;
                }
                let g = random_function(q.logic(), &mut r, &all, 4);
                let (mut every, mut none) = (true, true);
                let mut expected_weak = Vec::new();
                for row in 0..1u32 << tied_vars.len() {
                    let mut some = false;
                    for hidden in 0..1u32 << unknown_vars.len() {
                        let value = |var: u32| match tied_vars.iter().position(|&v| v == var) {
                            Some(i) => row >> i & 1 == 1,
                            None => hidden >> unknown_vars.iter().position(|&v| v == var).unwrap() & 1 == 1,
                        };
                        let holds = evaluate(&q.logic.m, g, &value);
                        some |= holds;
                        if evaluate(&q.logic.m, cond, &value) {
                            every &= holds;
                            none &= !holds;
                        }
                    }
                    expected_weak.push(some);
                }
                let expected = if every { Some(true) } else if none { Some(false) } else { None };
                if decisions.decide(q, g, cond, LANE) != expected {
                    wrong.push(format!("trial {} decides {:?}, expected {:?}", trial, decisions.decide(q, g, cond, LANE), expected));
                }
                let weak = decisions.weaken(q, g, LANE);
                for (row, &some) in expected_weak.iter().enumerate() {
                    let value = |var: u32| row >> tied_vars.iter().position(|&v| v == var).unwrap() & 1 == 1;
                    if evaluate(&q.logic.m, weak, &value) != some {
                        wrong.push(format!("trial {} weakens to {} at {}, expected {}", trial, !some, row, some));
                    }
                }
            }
            assert!(wrong.is_empty(), "decide and weaken must quantify the unknowns and keep the paths: {:?}", &wrong[..wrong.len().min(8)]);
        });
    }
}
