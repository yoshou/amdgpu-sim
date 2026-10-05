mod kernel;
mod rules;

#[cfg(test)]
mod tests;

pub use kernel::{choices, constant_choices, lane_test, projected_word, Atom, Choice, Kept, PATH};
#[cfg(test)]
pub(super) use rules::float_compare;

use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::ir::*;
use kernel::{Eval, Kernel, Queries};
use rules::State;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

pub struct Logic {
    pub m: Manager,
    kernel: Kernel<State>,
}

impl Logic {
    #[inline]
    fn eval(&mut self) -> Eval<'_, State> {
        self.kernel.eval(&mut self.m)
    }

    pub fn fixed(f: &Func, facts: &Facts, kept: &BTreeSet<Choice>, tags: &[Choice]) -> Self {
        Self {
            m: Manager::new(),
            kernel: Kernel::fixed(f, facts, kept, tags, State::default()),
        }
    }

    pub fn open(f: &Func, facts: &Facts, listed: &[Choice]) -> Self {
        let mut m = Manager::new();
        let kernel = Kernel::open(&mut m, f, facts, listed, State::default());
        Self { m, kernel }
    }

    pub fn keep(&mut self, kept: &BTreeSet<Choice>) {
        self.kernel.keep(kept);
    }

    pub fn is_open(&self) -> bool {
        self.kernel.is_open()
    }

    pub fn local(&mut self, c: Choice) -> Bdd {
        self.kernel.local(&mut self.m, c)
    }

    pub fn materialized(&self, facts: &Facts, v: ValueId) -> Bdd {
        self.kernel.materialized(facts, v)
    }

    pub fn tag(&mut self, c: Choice) -> Bdd {
        self.kernel.tag(&mut self.m, c)
    }

    pub fn carried(&self, param: ValueId) -> bool {
        self.kernel.carried(param)
    }

    pub fn all_local(&self) -> Bdd {
        self.kernel.all_local()
    }

    pub fn possible_policies(&mut self, condition: Bdd) -> Bdd {
        self.kernel.possible_policies(&mut self.m, condition)
    }

    pub fn choose(&mut self, safe: Bdd) -> Kept {
        self.kernel.choose(&mut self.m, safe)
    }

    pub fn settled(&mut self, f: Bdd, kept: &BTreeSet<Choice>) -> Bdd {
        self.kernel.settled(&mut self.m, f, kept)
    }

    pub fn exists(&mut self, vars: &[u32], f: Bdd) -> Bdd {
        kernel::exists(&mut self.m, vars, f)
    }

    pub fn forall(&mut self, vars: &[u32], f: Bdd) -> Bdd {
        kernel::forall(&mut self.m, vars, f)
    }

    pub fn atom(&mut self, atom: Atom) -> Bdd {
        self.kernel.atom(&mut self.m, atom)
    }

    pub fn atom_of(&self, var: u32) -> Atom {
        self.kernel.atom_of(var)
    }

    pub fn support(&mut self, f: Bdd) -> Rc<Vec<u32>> {
        self.kernel.support(&mut self.m, f)
    }

    pub fn uniform_atom(&self, facts: &Facts, var: u32) -> bool {
        self.kernel.uniform_atom(facts, var)
    }

    pub fn scope(&self, facts: &Facts, var: u32) -> Option<BlockId> {
        self.kernel.scope(facts, var)
    }

    pub fn cell_leaves(&self, block: BlockId, group: u16) -> &[ValueId] {
        self.kernel.state().cell_leaves(block, group)
    }

    pub fn answer_sources(&self, block: BlockId) -> Vec<ValueId> {
        self.kernel.state().answer_sources(block)
    }

    pub fn consistent(&mut self, x: Bdd) -> Bdd {
        rules::consistent(&mut self.eval(), x)
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
        self.kernel.state_mut().lane_bits(f, facts, v)
    }

    pub fn lane_function(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Option<[u32; 32]> {
        self.kernel.state_mut().lane_function(f, facts, v)
    }

    pub fn lane_dependent(&mut self, f: Bdd) -> bool {
        self.eval().lane_dependent(f)
    }

    pub fn at_lane(&mut self, f: Bdd, lane: u32) -> Bdd {
        self.eval().at_lane(f, lane)
    }

    #[cfg(test)]
    pub fn wave_answer(&mut self, f: &Func, facts: &Facts, out: ValueId) -> Bdd {
        rules::wave_answer(&mut self.eval(), f, facts, out)
    }

    pub fn answer(&mut self, f: &Func, facts: &Facts, out: ValueId, g: Bdd) -> Bdd {
        rules::answer(&mut self.eval(), f, facts, out, g)
    }

    pub fn lane_is(&mut self, f: &Func, facts: &Facts, block: BlockId, target: ValueId) -> Option<Bdd> {
        rules::lane_is(&mut self.eval(), f, facts, block, target)
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
        self.kernel.image(&mut self.m, f, facts, src, slot, formula)
    }

    #[cfg(test)]
    fn post(&mut self, f: &Func, facts: &Facts, src: BlockId, slot: usize, formula: Bdd) -> Bdd {
        self.kernel.post(&mut self.m, f, facts, src, slot, formula)
    }

    pub fn reach(
        &mut self,
        f: &Func,
        facts: &Facts,
        start: BlockId,
        formula: Bdd,
    ) -> BTreeMap<BlockId, Bdd> {
        self.kernel.reach(&mut self.m, f, facts, start, formula)
    }
}
