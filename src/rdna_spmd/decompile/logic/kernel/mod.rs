mod atoms;
mod edges;
mod patterns;
mod policy;
mod queries;
mod values;

pub use atoms::{Atom, Choice, PATH};
pub use patterns::{constant_choices, lane_test, projected_word};
pub use policy::{choices, Kept};

pub(super) use atoms::{exists, forall, Atoms};
pub(super) use edges::Binding;
pub(super) use queries::{Queries, Rules};
pub(super) use values::Eval;

use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::ir::*;
use edges::Edges;
use policy::Policy;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use values::Values;

pub struct Kernel<S> {
    values: Values<S>,
    edges: Edges,
}

impl<S: Rules> Kernel<S> {
    pub fn fixed(f: &Func, facts: &Facts, kept: &BTreeSet<Choice>, tags: &[Choice], state: S) -> Self {
        Self {
            values: Values::new(Atoms::new(f, facts, false, tags), Policy::fixed(kept, tags), state),
            edges: Edges::default(),
        }
    }

    pub fn open(m: &mut Manager, f: &Func, facts: &Facts, listed: &[Choice], state: S) -> Self {
        let mut atoms = Atoms::new(f, facts, true, listed);
        let policy = Policy::open(f, facts, listed, &mut atoms, m);
        Self {
            values: Values::new(atoms, policy, state),
            edges: Edges::default(),
        }
    }

    #[inline]
    pub fn eval<'x>(&'x mut self, m: &'x mut Manager) -> Eval<'x, S> {
        Eval {
            m,
            values: &mut self.values,
        }
    }

    pub fn state(&self) -> &S {
        &self.values.state
    }

    pub fn state_mut(&mut self) -> &mut S {
        &mut self.values.state
    }

    pub fn keep(&mut self, kept: &BTreeSet<Choice>) {
        self.values.policy.keep(kept);
    }

    pub fn is_open(&self) -> bool {
        self.values.policy.is_open()
    }

    pub fn local(&mut self, m: &mut Manager, c: Choice) -> Bdd {
        self.values.policy.local(&mut self.values.atoms, m, c)
    }

    pub fn materialized(&self, facts: &Facts, v: ValueId) -> Bdd {
        self.values.policy.materialized(facts, v)
    }

    pub fn tag(&mut self, m: &mut Manager, c: Choice) -> Bdd {
        self.values.policy.tag(&mut self.values.atoms, m, c)
    }

    pub fn carried(&self, param: ValueId) -> bool {
        self.values.policy.carried(param)
    }

    pub fn all_local(&self) -> Bdd {
        self.values.policy.all_local()
    }

    pub fn possible_policies(&mut self, m: &mut Manager, condition: Bdd) -> Bdd {
        policy::possible_policies(&mut self.values.atoms, m, condition)
    }

    pub fn choose(&self, m: &mut Manager, safe: Bdd) -> Kept {
        self.values.policy.choose(&self.values.atoms, m, safe)
    }

    pub fn settled(&mut self, m: &mut Manager, f: Bdd, kept: &BTreeSet<Choice>) -> Bdd {
        policy::settled(&mut self.values.atoms, m, f, kept)
    }

    pub fn atom(&mut self, m: &mut Manager, atom: Atom) -> Bdd {
        self.values.atoms.atom(m, atom)
    }

    pub fn atom_of(&self, var: u32) -> Atom {
        self.values.atoms.of(var)
    }

    pub fn support(&mut self, m: &mut Manager, f: Bdd) -> Rc<Vec<u32>> {
        self.values.atoms.support(m, f)
    }

    pub fn uniform_atom(&self, facts: &Facts, var: u32) -> bool {
        self.values.state.uniform(&self.values.atoms, facts, var)
    }

    pub fn scope(&self, facts: &Facts, var: u32) -> Option<BlockId> {
        atoms::scope(self.values.atoms.of(var), facts)
    }

    pub fn image(
        &mut self,
        m: &mut Manager,
        f: &Func,
        facts: &Facts,
        src: BlockId,
        slot: usize,
        formula: Bdd,
    ) -> Bdd {
        let mut q = Eval {
            m,
            values: &mut self.values,
        };
        self.edges.image(&mut q, f, facts, src, slot, formula)
    }

    #[cfg(test)]
    pub fn post(&mut self, m: &mut Manager, f: &Func, facts: &Facts, src: BlockId, slot: usize, formula: Bdd) -> Bdd {
        let mut q = Eval {
            m,
            values: &mut self.values,
        };
        self.edges.post(&mut q, f, facts, src, slot, formula)
    }

    pub fn reach(
        &mut self,
        m: &mut Manager,
        f: &Func,
        facts: &Facts,
        start: BlockId,
        formula: Bdd,
    ) -> BTreeMap<BlockId, Bdd> {
        let mut q = Eval {
            m,
            values: &mut self.values,
        };
        self.edges.reach(&mut q, f, facts, start, formula)
    }
}
