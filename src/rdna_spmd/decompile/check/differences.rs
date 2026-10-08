use super::super::logic::{Atom, Choice, Logic};
use super::lattice::{Lattice, Sent};
use super::masks::Masks;
use super::orderings::Orderings;
use super::program::Program;
use super::queries::Queries;
use super::rules::settled;
use super::verdict::Verdict;
use super::{Mode, Violation};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Differences<'a> {
    program: Program<'a>,
    pub(super) logic: Logic,
    orderings: Orderings,
    masks: Masks,
    lattice: Lattice,
    verdict: Verdict,
    settles: HashMap<ValueId, bool>,
    basis: Option<(BTreeMap<BlockId, Bdd>, Masks)>,
}

impl<'a> Differences<'a> {
    pub(super) fn new(program: Program<'a>, logic: Logic, mode: Mode) -> Self {
        let n = program.f.types.len();
        Self {
            program,
            logic,
            orderings: Orderings::default(),
            masks: Masks::unsolved(n),
            lattice: Lattice::new(n),
            verdict: Verdict::new(mode),
            settles: HashMap::default(),
            basis: None,
        }
    }

    pub(super) fn based(mut self, reach: BTreeMap<BlockId, Bdd>, masks: Masks) -> Self {
        self.basis = Some((reach, masks));
        self
    }

    pub(super) fn basis(self) -> (Logic, BTreeMap<BlockId, Bdd>, Masks) {
        let (reach, masks) = self.basis.expect("a check hands on what it started from");
        (self.logic, reach, masks)
    }

    #[inline]
    pub(super) fn mode(&self) -> Mode {
        self.verdict.mode()
    }

    pub(super) fn violations(&self) -> &[Violation] {
        self.verdict.violations()
    }

    pub(super) fn exhausted(&self) -> Option<(BlockId, usize, &'static str)> {
        self.verdict.exhausted()
    }

    pub(super) fn everyone(&mut self, kept: &BTreeSet<Choice>) -> BTreeSet<u64> {
        self.verdict.everyone(&mut self.logic, kept)
    }

    pub(super) fn detouring(&mut self, on: bool) {
        self.verdict.detouring(on);
    }

    pub(super) fn eager(&mut self) {
        self.verdict.eager();
    }

    #[cfg(test)]
    pub(super) fn masks(&self) -> &Masks {
        &self.masks
    }

    pub(super) fn start(&mut self) {
        let f = self.program.f;
        let start = match self.program.exec_index {
            Some(index) => self
                .logic
                .atom(Atom::Bit(f.blocks[&f.entry].params[index].0)),
            None => Bdd::TRUE,
        };
        let (reach, masks) = match self.basis.take() {
            Some(basis) => basis,
            None => {
                let reach = self.logic.reach(f, self.program.facts, f.entry, start);
                let masks = Masks::solve(&self.program, &mut self.logic);
                (reach, masks)
            }
        };
        self.lattice.start(reach.clone());
        self.orderings = Orderings::new(&self.program, &mut self.logic);
        self.masks = masks.clone();
        self.basis = Some((reach, masks));
    }

    pub(super) fn reorder(&mut self) -> Orderings {
        self.orderings = Orderings::new(&self.program, &mut self.logic);
        self.orderings.clone()
    }

    #[inline]
    pub(super) fn raise_h(&mut self, v: ValueId, x: Bdd) -> bool {
        self.lattice.raise_h(&mut self.logic.m, v, x)
    }

    #[inline]
    pub(super) fn raise_word(&mut self, v: ValueId, x: Bdd) -> bool {
        self.lattice.raise_word(&mut self.logic.m, v, x)
    }

    #[inline]
    pub(super) fn arrive(&mut self, b: BlockId, k: usize, width: usize, c: Bdd) -> bool {
        self.lattice.arrive(&mut self.logic.m, b, k, width, c)
    }

    #[inline]
    pub(super) fn unsent(&mut self, key: Sent, x: Bdd) -> Bdd {
        self.lattice.unsent(&mut self.logic.m, key, x)
    }

    #[inline]
    pub(super) fn arrivals(&self, b: BlockId) -> Option<&Vec<Bdd>> {
        self.lattice.arrivals(b)
    }

    pub(super) fn edge_difference(&mut self, pred: BlockId, slot: usize, difference: Bdd) -> Bdd {
        let guard = match self.lattice.guard(pred, slot) {
            Some(g) => g,
            None => {
                let mut guard = self.lattice.reachable(pred);
                if let Term::CondBr { cond, .. } = self.program.f.blocks[&pred].term {
                    let bit = self.bit(cond);
                    let taken = if slot == 0 { bit } else { self.not(bit) };
                    guard = self.and(guard, taken);
                }
                self.lattice.keep_guard(pred, slot, guard);
                guard
            }
        };
        let difference = self.and(difference, guard);
        self.logic.image(self.program.f, self.program.facts, pred, slot, difference)
    }
}

impl<'a> Queries<'a> for Differences<'a> {
    #[inline]
    fn program(&self) -> &Program<'a> {
        &self.program
    }

    #[inline]
    fn logic(&mut self) -> &mut Logic {
        &mut self.logic
    }

    #[inline]
    fn safe(&self) -> Bdd {
        self.lattice.safe()
    }

    #[inline]
    fn h(&self, v: ValueId) -> Bdd {
        self.lattice.h(v)
    }

    #[inline]
    fn word(&self, v: ValueId) -> Bdd {
        self.lattice.word(v)
    }

    #[inline]
    fn reachable(&self, b: BlockId) -> Bdd {
        self.lattice.reachable(b)
    }

    #[inline]
    fn loaded(&self, at: (BlockId, usize)) -> Option<Bdd> {
        self.orderings.loaded(at)
    }

    #[inline]
    fn reordered(&self, at: (BlockId, usize)) -> Option<Bdd> {
        self.orderings.reordered(at)
    }

    #[inline]
    fn masked(&self, v: ValueId) -> bool {
        self.masks.masked(v)
    }

    #[inline]
    fn faithful(&self, v: ValueId) -> bool {
        self.masks.faithful(v)
    }

    fn settles(&mut self, v: ValueId, op: Op) -> bool {
        if let Some(&known) = self.settles.get(&v) {
            return known;
        }
        let known = settled(self.program.f, self.program.facts, op);
        self.settles.insert(v, known);
        known
    }

    fn require(&mut self, block: BlockId, index: usize, reason: &'static str, difference: Bdd) {
        let difference = self.logic.m.and(difference, self.lattice.safe());
        let difference = self.logic.consistent(difference);
        if difference == Bdd::FALSE {
            return;
        }
        self.verdict.violated(Violation {
            block,
            index,
            reason,
            condition: difference,
        });
        if self.verdict.mode() == Mode::Direct {
            let bad = self.logic.possible_policies(difference);
            let good = self.logic.m.not(bad);
            if !self.lattice.narrow(&mut self.logic.m, good) {
                self.verdict.exhaust(block, index, reason);
            }
        }
    }

    #[inline]
    fn stopped(&self) -> bool {
        self.verdict.stopped()
    }

    #[inline]
    fn demand(&mut self, provenance: u64, condition: Bdd) {
        self.verdict.demand(&mut self.logic.m, provenance, condition);
    }
}
