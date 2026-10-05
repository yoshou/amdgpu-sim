use super::answers::Answers;
use super::atoms::{Atom, Atoms, Choice};
use super::bits::{compute_bit, compute_view};
use super::cells::Cells;
use super::lanes::LaneValues;
use super::policy::Policy;
use super::queries::Queries;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::rc::Rc;

pub(super) struct Values {
    pub(super) atoms: Atoms,
    pub(super) policy: Policy,
    pub(super) lanes: LaneValues,
    pub(super) cells: Cells,
    pub(super) answers: Answers,
    bits: HashMap<ValueId, Bdd>,
    views: HashMap<ValueId, Bdd>,
}

impl Values {
    pub(super) fn new(atoms: Atoms, policy: Policy) -> Self {
        Self {
            atoms,
            policy,
            lanes: LaneValues::default(),
            cells: Cells::default(),
            answers: Answers::default(),
            bits: HashMap::default(),
            views: HashMap::default(),
        }
    }
}

pub(super) struct Eval<'x> {
    pub(super) m: &'x mut Manager,
    pub(super) values: &'x mut Values,
}

impl Queries for Eval<'_> {
    #[inline]
    fn m(&mut self) -> &mut Manager {
        self.m
    }

    #[inline]
    fn atoms(&self) -> &Atoms {
        &self.values.atoms
    }

    #[inline]
    fn atom(&mut self, atom: Atom) -> Bdd {
        self.values.atoms.atom(self.m, atom)
    }

    #[inline]
    fn support(&mut self, f: Bdd) -> Rc<Vec<u32>> {
        self.values.atoms.support(self.m, f)
    }

    #[inline]
    fn local(&mut self, c: Choice) -> Bdd {
        self.values.policy.local(&mut self.values.atoms, self.m, c)
    }

    #[inline]
    fn materialized(&self, facts: &Facts, v: ValueId) -> Bdd {
        self.values.policy.materialized(facts, v)
    }

    #[inline]
    fn carried(&self, param: ValueId) -> bool {
        self.values.policy.carried(param)
    }

    #[inline]
    fn lane_values(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Option<[u32; 32]> {
        self.values.lanes.of(f, facts, v, 0)
    }

    #[inline]
    fn cells(&self) -> &Cells {
        &self.values.cells
    }

    #[inline]
    fn cells_mut(&mut self) -> &mut Cells {
        &mut self.values.cells
    }

    #[inline]
    fn answers(&self) -> &Answers {
        &self.values.answers
    }

    #[inline]
    fn answers_mut(&mut self) -> &mut Answers {
        &mut self.values.answers
    }

    fn bit(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Bdd {
        if let Some(&b) = self.values.bits.get(&v) {
            return b;
        }
        let b = compute_bit(self, f, facts, v);
        self.values.bits.insert(v, b);
        b
    }

    fn view(&mut self, f: &Func, facts: &Facts, w: ValueId) -> Bdd {
        if let Some(&b) = self.values.views.get(&w) {
            return b;
        }
        let b = compute_view(self, f, facts, w);
        self.values.views.insert(w, b);
        b
    }
}
