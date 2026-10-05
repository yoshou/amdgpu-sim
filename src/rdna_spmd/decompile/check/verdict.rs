use super::super::logic::{Choice, Logic};
use super::{Mode, Violation};
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct Verdict {
    mode: Mode,
    demands: BTreeMap<u64, Bdd>,
    violations: Vec<Violation>,
    exhausted: Option<(BlockId, usize, &'static str)>,
    stopped: bool,
    detouring: bool,
}

impl Verdict {
    pub(super) fn new(mode: Mode) -> Self {
        Self {
            mode,
            demands: BTreeMap::new(),
            violations: Vec::new(),
            exhausted: None,
            stopped: false,
            detouring: false,
        }
    }

    #[inline]
    pub(super) fn mode(&self) -> Mode {
        self.mode
    }

    #[inline]
    pub(super) fn stopped(&self) -> bool {
        self.stopped
    }

    pub(super) fn violations(&self) -> &[Violation] {
        &self.violations
    }

    pub(super) fn exhausted(&self) -> Option<(BlockId, usize, &'static str)> {
        self.exhausted
    }

    pub(super) fn detouring(&mut self, on: bool) {
        self.detouring = on;
    }

    pub(super) fn violated(&mut self, violation: Violation) {
        self.violations.push(violation);
        if self.mode == Mode::Search {
            self.stopped |= !self.detouring;
        }
    }

    pub(super) fn exhaust(&mut self, block: BlockId, index: usize, reason: &'static str) {
        self.exhausted = Some((block, index, reason));
        self.stopped = true;
    }

    pub(super) fn demand(&mut self, m: &mut Manager, provenance: u64, condition: Bdd) {
        if condition == Bdd::FALSE {
            return;
        }
        let old = self.demands.get(&provenance).copied().unwrap_or(Bdd::FALSE);
        let joined = m.or(old, condition);
        self.demands.insert(provenance, joined);
    }

    pub(super) fn everyone(&mut self, logic: &mut Logic, kept: &BTreeSet<Choice>) -> BTreeSet<u64> {
        let demands = std::mem::take(&mut self.demands);
        demands
            .into_iter()
            .filter_map(|(p, condition)| {
                (logic.settled(condition, kept) == Bdd::TRUE).then_some(p)
            })
            .collect()
    }
}
