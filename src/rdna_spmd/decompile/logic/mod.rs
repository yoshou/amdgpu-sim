mod answers;
mod atoms;
mod bits;
mod cells;
mod edges;
mod floats;
mod lanes;
mod patterns;
mod policy;
mod queries;
mod values;

#[cfg(test)]
mod tests;

pub use atoms::{Atom, Choice, PATH};
#[cfg(test)]
pub(super) use floats::float_compare;
pub use patterns::{constant_choices, lane_test, projected_word};
pub use policy::{choices, Kept};

use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::ir::*;
use atoms::Atoms;
use edges::Edges;
use policy::Policy;
use queries::Queries;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;
use values::{Eval, Values};

type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::rdna_spmd::hash::Mix>>;

pub struct Logic {
    pub m: Manager,
    values: Values,
    edges: Edges,
}

impl Logic {
    #[inline]
    fn eval(&mut self) -> Eval<'_> {
        Eval {
            m: &mut self.m,
            values: &mut self.values,
        }
    }

    pub fn fixed(f: &Func, facts: &Facts, kept: &BTreeSet<Choice>, tags: &[Choice]) -> Self {
        Self {
            m: Manager::new(),
            values: Values::new(Atoms::new(f, facts, false, tags), Policy::fixed(kept, tags)),
            edges: Edges::default(),
        }
    }

    pub fn open(f: &Func, facts: &Facts, listed: &[Choice]) -> Self {
        let mut m = Manager::new();
        let mut atoms = Atoms::new(f, facts, true, listed);
        let policy = Policy::open(f, facts, listed, &mut atoms, &mut m);
        Self {
            m,
            values: Values::new(atoms, policy),
            edges: Edges::default(),
        }
    }

    pub fn keep(&mut self, kept: &BTreeSet<Choice>) {
        self.values.policy.keep(kept);
    }

    pub fn is_open(&self) -> bool {
        self.values.policy.is_open()
    }

    pub fn local(&mut self, c: Choice) -> Bdd {
        self.values.policy.local(&mut self.values.atoms, &mut self.m, c)
    }

    pub fn materialized(&self, facts: &Facts, v: ValueId) -> Bdd {
        self.values.policy.materialized(facts, v)
    }

    pub fn tag(&mut self, c: Choice) -> Bdd {
        self.values.policy.tag(&mut self.values.atoms, &mut self.m, c)
    }

    pub fn carried(&self, param: ValueId) -> bool {
        self.values.policy.carried(param)
    }

    pub fn all_local(&self) -> Bdd {
        self.values.policy.all_local()
    }

    pub fn possible_policies(&mut self, condition: Bdd) -> Bdd {
        policy::possible_policies(&mut self.values.atoms, &mut self.m, condition)
    }

    pub fn choose(&mut self, safe: Bdd) -> Kept {
        self.values.policy.choose(&self.values.atoms, &mut self.m, safe)
    }

    pub fn settled(&mut self, f: Bdd, kept: &BTreeSet<Choice>) -> Bdd {
        policy::settled(&mut self.values.atoms, &mut self.m, f, kept)
    }

    pub fn exists(&mut self, vars: &[u32], f: Bdd) -> Bdd {
        atoms::exists(&mut self.m, vars, f)
    }

    pub fn forall(&mut self, vars: &[u32], f: Bdd) -> Bdd {
        atoms::forall(&mut self.m, vars, f)
    }

    pub fn atom(&mut self, atom: Atom) -> Bdd {
        self.values.atoms.atom(&mut self.m, atom)
    }

    pub fn atom_of(&self, var: u32) -> Atom {
        self.values.atoms.of(var)
    }

    pub fn support(&mut self, f: Bdd) -> Rc<Vec<u32>> {
        self.values.atoms.support(&mut self.m, f)
    }

    pub fn uniform_atom(&self, facts: &Facts, var: u32) -> bool {
        cells::uniform_atom(&self.values.atoms, &self.values.cells, facts, var)
    }

    pub fn scope(&self, facts: &Facts, var: u32) -> Option<BlockId> {
        atoms::scope(self.values.atoms.of(var), facts)
    }

    pub fn cell_leaves(&self, block: BlockId, group: u16) -> &[ValueId] {
        self.values.cells.leaves(block, group)
    }

    pub fn answer_sources(&self, block: BlockId) -> Vec<ValueId> {
        self.values.answers.sources(block)
    }

    pub fn consistent(&mut self, x: Bdd) -> Bdd {
        answers::consistent(&mut self.eval(), x)
    }

    pub fn bit(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Bdd {
        self.eval().bit(f, facts, v)
    }

    pub fn view(&mut self, f: &Func, facts: &Facts, w: ValueId) -> Bdd {
        self.eval().view(f, facts, w)
    }

    pub fn word(&mut self, k: u32) -> Bdd {
        self.eval().word(k)
    }

    pub fn lane_bits(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Option<[(u32, u32); 32]> {
        if f.types[v.0] != Ty::I32 {
            return None;
        }
        Some(self.values.lanes.known_bits(f, facts, v, 0))
    }

    pub fn lane_function(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Option<[u32; 32]> {
        self.values.lanes.of(f, facts, v, 0)
    }

    pub fn lane_dependent(&mut self, f: Bdd) -> bool {
        self.eval().lane_dependent(f)
    }

    pub fn at_lane(&mut self, f: Bdd, lane: u32) -> Bdd {
        self.eval().at_lane(f, lane)
    }

    #[cfg(test)]
    pub fn wave_answer(&mut self, f: &Func, facts: &Facts, out: ValueId) -> Bdd {
        answers::wave_answer(&mut self.eval(), f, facts, out)
    }

    pub fn answer(&mut self, f: &Func, facts: &Facts, out: ValueId, g: Bdd) -> Bdd {
        answers::answer(&mut self.eval(), f, facts, out, g)
    }

    pub fn lane_is(&mut self, f: &Func, facts: &Facts, block: BlockId, target: ValueId) -> Option<Bdd> {
        bits::lane_is(&mut self.eval(), f, facts, block, target)
    }

    pub fn lanes(&mut self, holds: impl Fn(u32) -> bool) -> Bdd {
        self.eval().lanes(holds)
    }

    pub fn image(
        &mut self,
        f: &Func,
        facts: &Facts,
        src: BlockId,
        slot: usize,
        formula: Bdd,
    ) -> Bdd {
        let mut q = Eval {
            m: &mut self.m,
            values: &mut self.values,
        };
        self.edges.image(&mut q, f, facts, src, slot, formula)
    }

    #[cfg(test)]
    fn post(&mut self, f: &Func, facts: &Facts, src: BlockId, slot: usize, formula: Bdd) -> Bdd {
        let mut q = Eval {
            m: &mut self.m,
            values: &mut self.values,
        };
        self.edges.post(&mut q, f, facts, src, slot, formula)
    }

    pub fn reach(
        &mut self,
        f: &Func,
        facts: &Facts,
        start: BlockId,
        formula: Bdd,
    ) -> BTreeMap<BlockId, Bdd> {
        let mut q = Eval {
            m: &mut self.m,
            values: &mut self.values,
        };
        self.edges.reach(&mut q, f, facts, start, formula)
    }
}
