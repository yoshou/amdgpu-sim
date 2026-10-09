mod differences;
mod explore;
mod lanes;
mod lattice;
mod masks;
mod orderings;
mod program;
mod queries;
mod rules;
mod schedule;
mod verdict;

#[cfg(test)]
mod tests;

use super::hazard::Hazards;
use super::logic::{Choice, Logic};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::ir::*;
use differences::Differences;
use explore::{Explore, Memo};
use orderings::Orderings;
use program::Program;
use queries::Queries;
use schedule::Schedule;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Search,

    Direct,
}

pub struct Basis {
    logic: Logic,
    reach: BTreeMap<BlockId, Bdd>,
    masks: masks::Masks,
}

pub struct Violation {
    pub block: BlockId,
    pub index: usize,
    pub reason: &'static str,
    pub condition: Bdd,
}

pub struct Check<'a> {
    differences: Differences<'a>,
    schedule: Schedule,
    version: BTreeMap<BlockId, usize>,
    detoured: BTreeMap<BlockId, usize>,
    memo: Memo,
}

impl<'a> Check<'a> {
    pub fn new(
        f: &'a Func,
        facts: &'a Facts,
        inputs: &'a [Parameter],
        exec_index: Option<usize>,
        loops: &'a Loops,
        hazards: &'a Hazards,
        logic: Logic,
    ) -> Self {
        let mode = if logic.is_open() { Mode::Direct } else { Mode::Search };
        let program = Program::new(f, facts, inputs, exec_index, loops, hazards);
        Self {
            schedule: Schedule::new(mode, &program),
            differences: Differences::new(program, logic, mode),
            version: BTreeMap::new(),
            detoured: BTreeMap::new(),
            memo: Memo::default(),
        }
    }

    pub fn resume(
        f: &'a Func,
        facts: &'a Facts,
        inputs: &'a [Parameter],
        exec_index: Option<usize>,
        loops: &'a Loops,
        hazards: &'a Hazards,
        basis: Basis,
        kept: &BTreeSet<Choice>,
    ) -> Self {
        let Basis { mut logic, reach, masks } = basis;
        logic.keep(kept);
        let program = Program::new(f, facts, inputs, exec_index, loops, hazards);
        Self {
            schedule: Schedule::new(Mode::Search, &program),
            differences: Differences::new(program, logic, Mode::Search).based(reach, masks),
            version: BTreeMap::new(),
            detoured: BTreeMap::new(),
            memo: Memo::default(),
        }
    }

    pub fn retire(self) -> Basis {
        let (logic, reach, masks) = self.differences.basis();
        Basis { logic, reach, masks }
    }

    pub fn logic(&mut self) -> &mut Logic {
        &mut self.differences.logic
    }

    pub fn eager(&mut self) {
        self.differences.eager();
    }

    pub fn safe(&self) -> Bdd {
        self.differences.safe()
    }

    pub fn violations(&self) -> &[Violation] {
        self.differences.violations()
    }

    pub fn exhausted(&self) -> Option<(BlockId, usize, &'static str)> {
        self.differences.exhausted()
    }

    pub fn orderings(&mut self) -> Orderings {
        self.differences.reorder()
    }

    pub fn everyone(&mut self, kept: &BTreeSet<Choice>) -> BTreeSet<u64> {
        self.differences.everyone(kept)
    }

    pub fn run(&mut self) -> bool {
        self.differences.start();
        loop {
            self.settle();
            if self.differences.stopped() {
                break;
            }
            self.differences.detouring(true);
            let arrivals = self.detours();
            self.differences.detouring(false);
            if self.differences.stopped()
                || (self.differences.mode() == Mode::Search && !self.differences.violations().is_empty())
            {
                break;
            }
            let mut changed = false;
            for (b, contributions) in arrivals {
                let width = contributions.len();
                for (k, c) in contributions.into_iter().enumerate() {
                    if self.differences.arrive(b, k, width, c) {
                        changed = true;
                        self.schedule.arrived(self.differences.program(), b, k);
                    }
                }
            }
            if !changed {
                break;
            }
        }
        match self.differences.mode() {
            Mode::Search => self.differences.violations().is_empty(),
            Mode::Direct => self.differences.exhausted().is_none(),
        }
    }

    fn settle(&mut self) {
        while let Some(r) = self.schedule.next() {
            let b = self.differences.program().facts.order[r];
            let safe = self.differences.safe();
            self.transfer(b);
            if self.differences.stopped() {
                return;
            }
            let narrowed = self.differences.safe() != safe;
            self.schedule.transferred(self.differences.program(), b, narrowed);
        }
    }

    fn raise_h(&mut self, v: ValueId, x: Bdd) {
        if self.differences.raise_h(v, x) {
            self.touch(v);
        }
    }

    fn raise_word(&mut self, v: ValueId, x: Bdd) {
        if self.differences.raise_word(v, x) {
            self.touch(v);
        }
    }

    fn touch(&mut self, v: ValueId) {
        let program = self.differences.program();
        let (Site::Param { block, .. } | Site::Inst { block, .. }) = program.facts.site[v.0] else {
            return;
        };
        *self.version.entry(block).or_default() += 1;
        self.schedule.rose(program, v);
    }

    fn transfer(&mut self, b: BlockId) {
        let (f, facts) = (self.differences.program().f, self.differences.program().facts);
        let block = &f.blocks[&b];
        let params = self.schedule.params(b, block.params.len());
        if !params.is_empty() && b != f.entry {
            let reachable = self.differences.reachable(b);
            let arrivals = self.differences.arrivals(b).cloned();
            for k in params {
                let d = &mut self.differences;
                let param = block.params[k].0;
                if !d.logic.carried(param) {
                    continue;
                }
                let mut h = arrivals.as_ref().map_or(Bdd::FALSE, |a| a[k]);
                let mut word = h;
                let is_word = facts.lane_word[param.0];
                let mode = if is_word {
                    d.logic.materialized(facts, param)
                } else {
                    Bdd::FALSE
                };
                for &(pred, slot) in &facts.incoming[&b] {
                    let arg = f.blocks[&pred].term.edges().nth(slot).unwrap().args[k];
                    let x = d.h(arg);
                    let x = d.unsent((b, k, pred, slot, false), x);
                    if x != Bdd::FALSE {
                        let image = d.edge_difference(pred, slot, x);
                        h = d.or(h, image);
                    }
                    if mode != Bdd::FALSE {
                        let x = d.whole(arg);
                        let x = d.unsent((b, k, pred, slot, true), x);
                        if x != Bdd::FALSE {
                            let image = d.edge_difference(pred, slot, x);
                            word = d.or(word, image);
                        }
                    }
                }
                if h != Bdd::FALSE {
                    h = d.and(h, reachable);
                    self.raise_h(param, h);
                }
                if is_word {
                    let d = &mut self.differences;
                    let word = d.and(word, reachable);
                    let x = d.logic.m.ite(mode, word, h);
                    self.raise_word(param, x);
                }
            }
        }
        if !self.schedule.pending(b) {
            return;
        }
        for (index, inst) in block.insts.iter().enumerate() {
            if !self.schedule.take(b, index) {
                continue;
            }
            let x = rules::inst(&mut self.differences, b, index, inst);
            if self.differences.stopped() {
                return;
            }
            for (m, v) in inst.outputs().into_iter().enumerate() {
                let x = match inst {
                    Inst::Effect {
                        op: EffectOp::Wave(WaveOp::Wmma),
                        inputs,
                        ..
                    } => rules::wmma_output(&mut self.differences, inputs, m),
                    _ => x,
                };
                self.raise_h(v, x);
                if facts.lane_word[v.0] {
                    let mode = self.differences.logic.materialized(facts, v);
                    if mode == Bdd::FALSE {
                        self.raise_word(v, x);
                    } else {
                        let word = rules::word_difference(&mut self.differences, inst, v);
                        let y = self.differences.logic.m.ite(mode, word, x);
                        self.raise_word(v, y);
                    }
                }
            }
        }
    }

    fn detours(&mut self) -> BTreeMap<BlockId, Vec<Bdd>> {
        let (f, facts) = (self.differences.program().f, self.differences.program().facts);
        let mut arrivals = BTreeMap::new();
        for &b in &facts.order {
            let Term::CondBr { cond, yes, no } = &f.blocks[&b].term else {
                continue;
            };
            let d = &mut self.differences;
            let hc = d.h(*cond);
            if hc == Bdd::FALSE || (yes.dst == no.dst && yes.args == no.args) {
                continue;
            }
            let version = self.version.get(&b).copied().unwrap_or(0);
            if self.detoured.insert(b, version) == Some(version) {
                continue;
            }
            let fc = d.bit(*cond);
            let nfc = d.not(fc);
            let reachable = d.reachable(b);
            for (wave, lane, taken) in [(0, 1, nfc), (1, 0, fc)] {
                let d = &mut self.differences;
                let assume = d.and(hc, taken);
                let assume = d.and(assume, reachable);
                let assume = d.and(assume, d.safe());
                if assume == Bdd::FALSE {
                    continue;
                }
                let program = d.program();
                let rank = |x: BlockId| program.rank[&x];
                let (stays, leaves) = ([yes, no][lane].dst, [yes, no][wave].dst);
                let spins = (0..program.loops.count()).any(|l| {
                    program.loops.contains(l, rank(b)) && program.loops.contains(l, rank(stays)) && !program.loops.contains(l, rank(leaves))
                });
                if spins {
                    let at = f.blocks[&b].insts.len();
                    d.require(b, at, "a converted lane may stay in a loop the wave leaves", assume);
                    if d.stopped() {
                        return arrivals;
                    }
                    continue;
                }
                let parts = match d.mode() {
                    Mode::Search => vec![Bdd::TRUE],
                    Mode::Direct => {
                        let all_local = d.logic.all_local();
                        vec![all_local, d.not(all_local)]
                    }
                };
                for part in parts {
                    let sliced = self.differences.and(assume, part);
                    if sliced == Bdd::FALSE {
                        continue;
                    }
                    Explore::new(&mut self.differences, &mut self.memo, b, sliced).run(wave, lane, &mut arrivals);
                    if self.differences.stopped() {
                        return arrivals;
                    }
                }
            }
        }
        arrivals
    }
}
