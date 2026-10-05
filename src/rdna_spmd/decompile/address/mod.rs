mod form;
mod graph;
mod trail;

mod limits;
mod program;

mod symbols;

mod queries;

mod bits;
mod control;
mod fields;
mod memory;
mod wide;
mod word;

mod values;

mod access;
mod origins;

#[cfg(test)]
mod tests;

pub use form::{aligns, Form, Region, Regions, Unknown, UnknownInfo, Value, Wide, LANES};
pub use graph::{Copies, PRIVATE_MEMORY};
pub use limits::{compare, condition, literals, Classes};

use super::hazard::Reach;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::engine::EntryLayout;
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::ir::*;
use form::Assumed;
use origins::Origins;
use program::{Conditions, Program};
use queries::Queries;
use std::collections::BTreeSet;
use values::Values;

type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::rdna_spmd::hash::Mix>>;

pub struct Addresses<'a> {
    values: Values<'a>,
    origins: Origins,
}

impl<'a> Addresses<'a> {
    pub fn new(
        f: &'a Func,
        facts: &'a Facts,
        inputs: &'a [Parameter],
        exec: u32,
        entry: EntryLayout,
        env: &'a Environment,
        headers: BTreeSet<BlockId>,
        registry: &DialectRegistry,
    ) -> Self {
        let mut conditions = Conditions::new(f);
        let program = Program::new(f, facts, inputs, exec, entry, env, headers, registry, &mut conditions);
        Self {
            values: Values::new(program, conditions),
            origins: Origins::default(),
        }
    }

    pub fn enter(&mut self, wave: usize) {
        if self.values.enter(wave) {
            self.origins.enter();
        }
    }

    pub fn unknowns(&self) -> &[UnknownInfo] {
        &self.values.symbols().unknowns
    }

    pub fn trip(&self, header: BlockId) -> Option<Unknown> {
        self.values.symbols().trip(header)
    }

    pub fn waves(&self) -> usize {
        self.values.symbols().waves()
    }

    pub fn valid(&self, lane: usize) -> bool {
        self.values.symbols().valid(lane)
    }

    pub fn bounds(&self, form: &Form) -> Option<(u64, u64)> {
        self.values.symbols().bounds(form)
    }

    pub fn wide(&self, v: ValueId) -> bool {
        self.values.program().f.types[v.0] == Ty::I64
    }

    pub fn reaches_block(&mut self, b: BlockId) -> bool {
        self.values.reached(b)
    }

    pub fn operand(&mut self, x: ValueId, at: BlockId, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
        self.values.operand(x, at, lane, assume)
    }

    pub fn bit(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
        self.values.bit(v, lane, assume)
    }

    pub fn high(&mut self, v: ValueId, lane: usize) -> Form {
        self.values.high(v, lane)
    }

    pub fn wide_value(&mut self, v: ValueId, at: BlockId, lane: usize) -> Option<Wide> {
        access::wide_at(&mut self.values, v, at, lane, 0)
    }

    pub fn resource_span(&mut self, at: (BlockId, usize), reach: Reach, lane: usize) -> Option<(Value, u32, Option<Form>)> {
        access::resource_span(&mut self.values, at, reach, lane)
    }

    pub fn access_classes(&mut self, block: BlockId, predicate: Option<ValueId>, lane: usize) -> Classes {
        access::access_classes(&mut self.values, block, predicate, lane)
    }

    pub fn regions(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>, refine: bool) -> Regions {
        self.origins.regions(&mut self.values, v, lane, assume, refine)
    }

    pub fn read_regions(&mut self, at: (BlockId, usize), lane: usize, exec: Option<ValueId>, refine: bool) -> Regions {
        self.origins.read_regions(&mut self.values, at, lane, exec, refine)
    }

    pub fn settle_loops(&mut self) -> bool {
        self.origins.settle_loops(&mut self.values)
    }

    pub fn pending(&self) -> BTreeSet<(ValueId, u8, bool)> {
        self.origins.pending()
    }

    pub fn forget(&mut self, kept: &BTreeSet<(ValueId, u8, bool)>) {
        self.origins.forget(kept)
    }

    pub fn exposable(&self) -> BTreeSet<u64> {
        self.values.program().exposable()
    }

    pub fn expose(&mut self, open: &[u64], stake: &[u64]) -> Vec<u64> {
        self.origins.expose(&mut self.values, open, stake)
    }
}
