use super::atoms::{Atom, Atoms, Choice};
use super::policy::Policy;
use super::queries::{Queries, Rules};
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::rc::Rc;

pub(in super::super) struct Values<S> {
    pub(in super::super) atoms: Atoms,
    pub(in super::super) policy: Policy,
    pub(in super::super) state: S,
    bits: HashMap<ValueId, Bdd>,
    views: HashMap<ValueId, Bdd>,
}

impl<S> Values<S> {
    pub(in super::super) fn new(atoms: Atoms, policy: Policy, state: S) -> Self {
        Self {
            atoms,
            policy,
            state,
            bits: HashMap::default(),
            views: HashMap::default(),
        }
    }
}

pub(in super::super) struct Eval<'x, S> {
    pub(in super::super) m: &'x mut Manager,
    pub(in super::super) values: &'x mut Values<S>,
}

impl<S: Rules> Queries for Eval<'_, S> {
    type State = S;

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
    fn state(&self) -> &S {
        &self.values.state
    }

    #[inline]
    fn state_mut(&mut self) -> &mut S {
        &mut self.values.state
    }

    fn bit(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Bdd {
        if let Some(&b) = self.values.bits.get(&v) {
            return b;
        }
        let b = S::bit(self, f, facts, v);
        self.values.bits.insert(v, b);
        b
    }

    fn view(&mut self, f: &Func, facts: &Facts, w: ValueId) -> Bdd {
        if let Some(&b) = self.values.views.get(&w) {
            return b;
        }
        let b = S::view(self, f, facts, w);
        self.values.views.insert(w, b);
        b
    }
}
