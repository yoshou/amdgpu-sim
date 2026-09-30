use super::hazard::{Hazards, Kind};
use super::address::compare;
use super::logic::{constant_choices, lane_test, projected_word, Atom, Choice, Logic, PATH};
#[cfg(test)]
use super::logic::float_compare;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site, Use};
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::rc::Rc;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Search,

    Direct,
}

pub struct Violation {
    pub block: BlockId,
    pub index: usize,
    pub reason: &'static str,
    pub condition: Bdd,
}

pub struct Check<'a> {
    f: &'a Func,
    facts: &'a Facts,
    inputs: &'a [Parameter],
    exec_index: Option<usize>,
    loops: &'a Loops,
    hazards: &'a Hazards,
    meetings: BTreeMap<(BlockId, usize), usize>,
    loaded: BTreeMap<(BlockId, usize), Bdd>,
    reordered: BTreeMap<(BlockId, usize), Bdd>,
    rank: BTreeMap<BlockId, usize>,
    pub logic: Logic,
    mode: Mode,
    pub safe: Bdd,
    h: Vec<Bdd>,

    words: Vec<Bdd>,
    arrivals: BTreeMap<BlockId, Vec<Bdd>>,
    reach: BTreeMap<BlockId, Bdd>,

    masked: Vec<bool>,
    faithful: Vec<bool>,
    pub demands: BTreeMap<u64, Bdd>,
    pub violations: Vec<Violation>,
    pub exhausted: Option<(BlockId, usize, &'static str)>,
    stopped: bool,
    detouring: bool,

    version: BTreeMap<BlockId, usize>,
    detoured: BTreeMap<BlockId, usize>,
    guards: HashMap<(usize, usize), Bdd>,
    positions: BTreeMap<BlockId, Rc<HashMap<ValueId, usize>>>,
    fresh_in: HashMap<Bdd, bool>,
    settles: HashMap<ValueId, bool>,

    dirty_params: BTreeMap<BlockId, BTreeSet<usize>>,
    dirty_insts: BTreeMap<BlockId, BTreeSet<usize>>,
    queue: BinaryHeap<Reverse<usize>>,
    queued: Vec<bool>,

    stale: Vec<bool>,
    rose: bool,
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
        let n = f.types.len();
        let blocks = facts.order.len();
        let search = mode == Mode::Search;
        let dirty_insts = match mode {
            Mode::Search => facts
                .order
                .iter()
                .map(|&b| (b, (0..f.blocks[&b].insts.len()).collect()))
                .collect(),
            Mode::Direct => BTreeMap::new(),
        };
        let queue = match mode {
            Mode::Search => (0..blocks).map(Reverse).collect(),
            Mode::Direct => BinaryHeap::new(),
        };
        Self {
            f,
            facts,
            inputs,
            exec_index,
            loops,
            hazards,
            meetings: hazards
                .meetings
                .iter()
                .enumerate()
                .map(|(i, &position)| (position, i))
                .collect(),
            loaded: BTreeMap::new(),
            reordered: BTreeMap::new(),
            rank: facts
                .order
                .iter()
                .enumerate()
                .map(|(r, &b)| (b, r))
                .collect(),
            logic,
            mode,
            safe: Bdd::TRUE,
            h: vec![Bdd::FALSE; n],
            words: vec![Bdd::FALSE; n],
            arrivals: BTreeMap::new(),
            reach: BTreeMap::new(),
            masked: vec![false; n],
            faithful: vec![false; n],
            demands: BTreeMap::new(),
            violations: Vec::new(),
            exhausted: None,
            stopped: false,
            detouring: false,
            version: BTreeMap::new(),
            detoured: BTreeMap::new(),
            guards: HashMap::default(),
            positions: BTreeMap::new(),
            fresh_in: HashMap::default(),
            settles: HashMap::default(),
            dirty_params: BTreeMap::new(),
            dirty_insts,
            queue,
            queued: vec![search; blocks],
            stale: vec![!search; blocks],
            rose: false,
        }
    }

    pub fn orderings(&mut self) -> (BTreeMap<(BlockId, usize), Bdd>, BTreeMap<(BlockId, usize), Bdd>) {
        self.loaded.clear();
        self.reordered.clear();
        self.order_accesses();
        (self.loaded.clone(), self.reordered.clone())
    }

    pub fn run(&mut self) -> bool {
        let f = self.f;
        let start = match self.exec_index {
            Some(index) => self
                .logic
                .atom(Atom::Bit(f.blocks[&f.entry].params[index].0)),
            None => Bdd::TRUE,
        };
        self.reach = self.logic.reach(f, self.facts, f.entry, start);
        self.order_accesses();
        self.solve_masked();
        loop {
            self.settle();
            if self.stopped {
                break;
            }
            self.detouring = true;
            let arrivals = self.detours();
            self.detouring = false;
            if self.stopped || (self.mode == Mode::Search && !self.violations.is_empty()) {
                break;
            }
            let mut changed = false;
            for (b, contributions) in arrivals {
                let width = contributions.len();
                for (k, c) in contributions.into_iter().enumerate() {
                    let c = self.and(c, self.safe);
                    if c == Bdd::FALSE {
                        continue;
                    }
                    let old = self
                        .arrivals
                        .entry(b)
                        .or_insert_with(|| vec![Bdd::FALSE; width])[k];
                    let joined = self.or(old, c);
                    if joined != old {
                        self.arrivals.get_mut(&b).unwrap()[k] = joined;
                        changed = true;
                        match self.mode {
                            Mode::Search => self.dirty(b, Some(k), None),
                            Mode::Direct => self.stale[self.rank[&b]] = true,
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        match self.mode {
            Mode::Search => self.violations.is_empty(),
            Mode::Direct => self.exhausted.is_none(),
        }
    }

    pub fn everyone(&mut self, kept: &BTreeSet<Choice>) -> BTreeSet<u64> {
        let demands = std::mem::take(&mut self.demands);
        demands
            .into_iter()
            .filter_map(|(p, condition)| {
                (self.logic.settled(condition, kept) == Bdd::TRUE).then_some(p)
            })
            .collect()
    }

    fn collective(&mut self, inst: &Inst) -> Option<Bdd> {
        let Inst::Effect {
            op: EffectOp::Wave(op),
            outputs,
            ..
        } = inst
        else {
            return None;
        };
        let out = outputs[0].0;
        match op {
            WaveOp::Any => {
                let local = self.logic.local(Choice::Query(out));
                Some(self.not(local))
            }
            WaveOp::Ballot => Some(self.logic.materialized(self.facts, out)),
            _ => None,
        }
    }

    fn reads_a_varying_first_lane(&self, inst: &Inst) -> bool {
        match inst {
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::ReadFirstLane),
                inputs,
                ..
            } => !self.facts.uniform[inputs[0].0],
            _ => false,
        }
    }

    fn reads_back_its_own_constant(&self, s: usize, t: usize) -> bool {
        let accesses = &self.hazards.accesses;
        let (w, r) = (&accesses[s], &accesses[t]);
        let same = w.block == r.block
            && w.index < r.index
            && w.kind == Kind::Write
            && r.kind == Kind::Read
            && w.space == r.space
            && w.address.is_some()
            && w.address == r.address
            && w.predicate == r.predicate
            && w.bytes >= r.bytes;
        if !same || !w.address.is_some_and(|a| self.facts.uniform[a.0]) {
            return false;
        }
        let Inst::Effect {
            op: EffectOp::Memory {
                op: MemoryOp::Store(_),
                ..
            },
            inputs,
            ..
        } = &self.f.blocks[&w.block].insts[w.index]
        else {
            return false;
        };
        self.facts.constant(self.f, inputs[1]).is_some()
            && !accesses
                .iter()
                .any(|a| a.block == w.block && a.index > w.index && a.index < r.index && a.kind != Kind::Read)
    }

    fn order_accesses(&mut self) {
        let hazards = self.hazards;
        let conflicts = hazards.conflicts();
        if conflicts.is_empty() {
            return;
        }
        let (f, facts) = (self.f, self.facts);
        let mut partners: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for &(p, q) in &conflicts {
            partners.entry(p).or_default().insert(q);
            partners.entry(q).or_default().insert(p);
        }
        let pair = |p: usize, q: usize| (p.min(q), p.max(q));
        let at: BTreeMap<(BlockId, usize), usize> = hazards
            .accesses
            .iter()
            .enumerate()
            .map(|(i, a)| ((a.block, a.index), i))
            .collect();
        let n = facts.order.len();
        for (&s, targets) in &partners {
            let source = &hazards.accesses[s];
            let source_rank = self.rank[&source.block];
            let mut entry = vec![[Bdd::FALSE; 2]; n];
            let mut reaches: BTreeMap<usize, [Bdd; 2]> = BTreeMap::new();
            let mut work: BTreeSet<usize> = BTreeSet::from([source_rank]);
            while let Some(r) = work.pop_first() {
                let b = facts.order[r];
                let [mut within, mut around] = entry[r];
                let mut fresh = Bdd::FALSE;
                for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
                    if let Some(&m) = self.meetings.get(&(b, index)) {
                        let local = self.logic.local(Choice::Meet(m));
                        let tag = self.logic.tag(Choice::Meet(m));
                        let left = self.and(local, tag);
                        within = self.and(within, left);
                        around = self.and(around, left);
                        fresh = self.and(fresh, left);
                    }
                    if meets_every_lane(inst) || self.reads_a_varying_first_lane(inst) {
                        within = Bdd::FALSE;
                        around = Bdd::FALSE;
                        fresh = Bdd::FALSE;
                    }
                    if let Some(kept) = self.collective(inst) {
                        let converted = self.not(kept);
                        within = self.and(within, converted);
                        around = self.and(around, converted);
                        fresh = self.and(fresh, converted);
                    }
                    if let Some(&t) = at.get(&(b, index)) {
                        if targets.contains(&t) {
                            let key = pair(s, t);
                            let again = hazards.accesses[t].instruction == source.instruction;
                            let mut pending = [Bdd::FALSE; 2];
                            if hazards.together.contains(&key) && !again {
                                pending[0] = self.or(within, fresh);
                            }
                            if hazards.apart.contains(&key) {
                                pending[1] = around;
                            }
                            if pending != [Bdd::FALSE; 2] {
                                let old = reaches.get(&t).copied().unwrap_or([Bdd::FALSE; 2]);
                                let joined = [self.or(old[0], pending[0]), self.or(old[1], pending[1])];
                                reaches.insert(t, joined);
                            }
                        }
                    }
                    if (b, index) == (source.block, source.index) {
                        fresh = Bdd::TRUE;
                    }
                }
                let within = self.or(within, fresh);
                if within == Bdd::FALSE && around == Bdd::FALSE {
                    continue;
                }
                for e in f.blocks[&b].term.edges() {
                    let d = self.rank[&e.dst];
                    let back = d <= r
                        && (0..self.loops.count()).any(|l| {
                            self.loops.header(l) == d && self.loops.contains(l, source_rank)
                        });
                    let (w, a) = if back {
                        (Bdd::FALSE, self.or(within, around))
                    } else {
                        (within, around)
                    };
                    let [ow, oa] = entry[d];
                    let (nw, na) = (self.or(ow, w), self.or(oa, a));
                    if (nw, na) != (ow, oa) {
                        entry[d] = [nw, na];
                        work.insert(d);
                    }
                }
            }
            for (t, parts) in reaches {
                let target = &hazards.accesses[t];
                let key = pair(s, t);
                let reads = |this: &mut Self, reader: usize| -> Bdd {
                    let side = if reader == key.0 { 0 } else { 1 };
                    let mut joined = Bdd::FALSE;
                    for (k, apart) in [(0, false), (1, true)] {
                        let mut part = parts[k];
                        if part == Bdd::FALSE {
                            continue;
                        }
                        if hazards.idle.contains(&(key.0, key.1, apart, side)) {
                            if let Some(e) = hazards.accesses[reader].exec {
                                let active = this.bit(e);
                                let idle = this.not(active);
                                part = this.and(part, idle);
                            }
                        }
                        joined = this.or(joined, part);
                    }
                    joined
                };
                let add = |this: &mut Self, loaded: bool, at: (BlockId, usize), extra: Bdd| {
                    let map = if loaded { &mut this.loaded } else { &mut this.reordered };
                    let old = map.get(&at).copied().unwrap_or(Bdd::FALSE);
                    let joined = this.logic.m.or(old, extra);
                    let map = if loaded { &mut this.loaded } else { &mut this.reordered };
                    map.insert(at, joined);
                };
                if source.kind.writes() && target.kind.reads() && !self.reads_back_its_own_constant(s, t) {
                    let extra = reads(self, t);
                    add(self, true, (target.block, target.index), extra);
                }
                if source.kind.reads() && target.kind.writes() {
                    let extra = reads(self, s);
                    add(self, true, (source.block, source.index), extra);
                }
                if source.kind.writes() && target.kind.writes() {
                    let extra = self.or(parts[0], parts[1]);
                    add(self, false, (target.block, target.index), extra);
                }
            }
        }
    }

    fn bit(&mut self, v: ValueId) -> Bdd {
        self.logic.bit(self.f, self.facts, v)
    }

    fn view(&mut self, v: ValueId) -> Bdd {
        self.logic.view(self.f, self.facts, v)
    }

    fn and(&mut self, a: Bdd, b: Bdd) -> Bdd {
        self.logic.m.and(a, b)
    }

    fn or(&mut self, a: Bdd, b: Bdd) -> Bdd {
        self.logic.m.or(a, b)
    }

    fn not(&mut self, a: Bdd) -> Bdd {
        self.logic.m.not(a)
    }

    fn partnered(&self, at: (BlockId, usize)) -> bool {
        let hazards = self.hazards;
        hazards
            .accesses
            .iter()
            .position(|a| (a.block, a.index) == at)
            .is_some_and(|i| hazards.together.iter().chain(&hazards.apart).any(|&(p, q)| p == i || q == i))
    }

    fn reachable(&self, b: BlockId) -> Bdd {
        self.reach.get(&b).copied().unwrap_or(Bdd::FALSE)
    }

    fn whole(&self, v: ValueId) -> Bdd {
        if self.facts.lane_word[v.0] {
            self.words[v.0]
        } else {
            self.h[v.0]
        }
    }

    fn wmma_output(&mut self, inputs: &[ValueId], m: usize) -> Bdd {
        let fragments = self.any_of(&inputs[..8]);
        let fragments = self.some_lane(fragments);
        let accumulator = self.whole(inputs[8 + m]);
        self.or(accumulator, fragments)
    }

    fn any_of(&mut self, values: &[ValueId]) -> Bdd {
        let mut r = Bdd::FALSE;
        for &v in values {
            let x = self.whole(v);
            r = self.or(r, x);
        }
        r
    }

    fn require(&mut self, block: BlockId, index: usize, reason: &'static str, difference: Bdd) {
        let difference = self.and(difference, self.safe);
        if difference == Bdd::FALSE {
            return;
        }
        self.violations.push(Violation {
            block,
            index,
            reason,
            condition: difference,
        });
        match self.mode {
            Mode::Search => self.stopped |= !self.detouring,
            Mode::Direct => {
                let bad = self.logic.possible_policies(difference);
                let good = self.not(bad);
                self.safe = self.and(self.safe, good);
                if self.safe == Bdd::FALSE {
                    self.exhausted = Some((block, index, reason));
                    self.stopped = true;
                    return;
                }
                self.restrict();
            }
        }
    }

    fn restrict(&mut self) {
        let safe = self.safe;
        let m = &mut self.logic.m;
        for x in self.h.iter_mut().chain(self.words.iter_mut()) {
            if *x != Bdd::FALSE {
                *x = m.and(*x, safe);
            }
        }
        for r in self.reach.values_mut() {
            *r = m.and(*r, safe);
        }
        for contributions in self.arrivals.values_mut() {
            for x in contributions {
                *x = m.and(*x, safe);
            }
        }
        self.guards.clear();
    }

    fn demand(&mut self, provenance: u64, condition: Bdd) {
        if condition == Bdd::FALSE {
            return;
        }
        let old = self.demands.get(&provenance).copied().unwrap_or(Bdd::FALSE);
        let joined = self.or(old, condition);
        self.demands.insert(provenance, joined);
    }

    fn tests_a_masked_word(&self, v: ValueId) -> bool {
        let (f, facts) = (self.f, self.facts);
        let zero = |x: ValueId| facts.constant(f, x) == Some(0);
        match facts.op(f, v) {
            Some(Op::Cmp(IntPred::Ne, a, b)) if zero(b) => self.zero_off(a, 0),
            Some(Op::Cmp(IntPred::Ne, a, b)) if zero(a) => self.zero_off(b, 0),
            Some(Op::Cmp(IntPred::Ugt, a, b)) if zero(b) => self.zero_off(a, 0),
            Some(Op::Cmp(IntPred::Ult, a, b)) if zero(a) => self.zero_off(b, 0),
            _ => false,
        }
    }

    fn zero_off(&self, x: ValueId, depth: usize) -> bool {
        let (f, facts) = (self.f, self.facts);
        if facts.constant(f, x) == Some(0) {
            return true;
        }
        if depth > 16 {
            return false;
        }
        let next = |y: ValueId| self.zero_off(y, depth + 1);
        match facts.op(f, x) {
            Some(Op::Convert(Cvt::ZExt | Cvt::SExt, _, b)) if f.types[b.0] == Ty::I1 => self.masked[b.0],
            Some(Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, _, b)) => next(b),
            Some(Op::Int(IntOp::Mul | IntOp::And, a, b)) => next(a) || next(b),
            Some(Op::Int(IntOp::Shl | IntOp::LShr | IntOp::AShr, a, _)) => next(a),
            Some(Op::Int(IntOp::Add | IntOp::Sub | IntOp::Or | IntOp::Xor, a, b)) => next(a) && next(b),
            Some(Op::Select(k, a, b)) => (self.masked[k.0] && next(b)) || (next(a) && next(b)),
            Some(_) => false,
            None => match facts.site[x.0] {
                Site::Param { block, index } if block != f.entry && Some(index) != self.exec_index => {
                    facts.arguments(f, block, index).all(next)
                }
                _ => false,
            },
        }
    }

    fn solve_masked(&mut self) {
        let (f, facts) = (self.f, self.facts);
        let n = f.types.len();
        let word = |v: ValueId| f.types[v.0] == Ty::I32 && facts.lane_word[v.0];
        let Some(ei) = self.exec_index else {
            self.masked = vec![true; n];
            self.faithful = (0..n).map(|v| word(ValueId(v))).collect();
            return;
        };
        for &b in &facts.order {
            let block = &f.blocks[&b];
            let exec = block.params[ei].0;
            for (index, &(param, ty)) in block.params.iter().enumerate() {
                let cleared = matches!(
                    self.inputs.get(index).map(|p| p.source),
                    Some(ParameterSource::MaskBit(_))
                );
                let assumed = b != f.entry || param == exec || cleared;
                self.masked[param.0] = assumed && (ty == Ty::I1 || word(param));
                self.faithful[param.0] = assumed && word(param);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &b in &facts.order {
                let block = &f.blocks[&b];
                let exec = block.params[ei].0;
                let mut active = self.logic.atom(Atom::Bit(exec));
                for &(param, ty) in &block.params {
                    if param != exec && self.masked[param.0] {
                        let atom = if ty == Ty::I1 {
                            Atom::Bit(param)
                        } else {
                            Atom::View(param)
                        };
                        let known = self.logic.atom(atom);
                        active = self.or(active, known);
                    }
                }
                let mut lockstep = HashMap::default();
                for inst in &block.insts {
                    for v in inst.outputs() {
                        let ty = f.types[v.0];
                        if ty != Ty::I1 && ty != Ty::I32 {
                            continue;
                        }
                        let formula = if ty == Ty::I1 {
                            self.bit(v)
                        } else {
                            self.view(v)
                        };
                        let masked = self.logic.m.implies(formula, active) || (ty == Ty::I1 && self.tests_a_masked_word(v));
                        if self.masked[v.0] != masked {
                            self.masked[v.0] = masked;
                            changed = true;
                        }
                        if word(v) {
                            let faithful =
                                self.lockstep_view(v, active, &mut lockstep) == Some(formula);
                            if self.faithful[v.0] != faithful {
                                self.faithful[v.0] = faithful;
                                changed = true;
                            }
                        }
                    }
                }
            }
            for &b in &facts.order {
                if b == f.entry {
                    continue;
                }
                let block = &f.blocks[&b];
                let exec = block.params[ei].0;
                for (index, &(param, _)) in block.params.iter().enumerate() {
                    if param == exec {
                        continue;
                    }
                    if self.masked[param.0]
                        && !facts.arguments(f, b, index).all(|a| self.masked[a.0])
                    {
                        self.masked[param.0] = false;
                        changed = true;
                    }
                    if self.faithful[param.0]
                        && !facts
                            .arguments(f, b, index)
                            .all(|a| !word(a) || self.faithful[a.0])
                    {
                        self.faithful[param.0] = false;
                        changed = true;
                    }
                }
            }
        }
    }

    fn lockstep_view(
        &mut self,
        w: ValueId,
        active: Bdd,
        memo: &mut HashMap<ValueId, Option<Bdd>>,
    ) -> Option<Bdd> {
        if let Some(&l) = memo.get(&w) {
            return l;
        }
        let (f, facts) = (self.f, self.facts);
        let l = if !facts.lane_word[w.0] {
            Some(self.view(w))
        } else {
            match facts.inst(f, w) {
                None => self.faithful[w.0].then(|| self.view(w)),
                Some(Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Ballot),
                    inputs,
                    ..
                }) => {
                    let x = self.bit(inputs[0]);
                    Some(self.and(x, active))
                }
                Some(Inst::Core { op, .. }) => match *op {
                    Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                        match (
                            self.lockstep_view(a, active, memo),
                            self.lockstep_view(b, active, memo),
                        ) {
                            (Some(a), Some(b)) => Some(match k {
                                IntOp::And => self.and(a, b),
                                IntOp::Or => self.or(a, b),
                                _ => self.logic.m.xor(a, b),
                            }),
                            _ => None,
                        }
                    }
                    Op::Select(c, a, b) => {
                        match (
                            self.lockstep_view(a, active, memo),
                            self.lockstep_view(b, active, memo),
                        ) {
                            (Some(a), Some(b)) => {
                                let c = self.bit(c);
                                Some(self.logic.m.ite(c, a, b))
                            }
                            _ => None,
                        }
                    }
                    Op::Convert(Cvt::Bitcast, Ty::I32, a) => self.lockstep_view(a, active, memo),
                    _ => Some(self.view(w)),
                },
                Some(_) => None,
            }
        };
        memo.insert(w, l);
        l
    }

    fn raise_h(&mut self, v: ValueId, x: Bdd) {
        if x == Bdd::FALSE {
            return;
        }
        let x = self.and(x, self.safe);
        let joined = self.or(self.h[v.0], x);
        if joined != self.h[v.0] {
            self.h[v.0] = joined;
            self.touch(v);
        }
    }

    fn raise_word(&mut self, v: ValueId, x: Bdd) {
        if x == Bdd::FALSE {
            return;
        }
        let x = self.and(x, self.safe);
        let joined = self.or(self.words[v.0], x);
        if joined != self.words[v.0] {
            self.words[v.0] = joined;
            self.touch(v);
        }
    }

    fn touch(&mut self, v: ValueId) {
        let (f, facts) = (self.f, self.facts);
        let (Site::Param { block, .. } | Site::Inst { block, .. }) = facts.site[v.0] else {
            return;
        };
        *self.version.entry(block).or_default() += 1;
        self.rose = true;
        if self.mode != Mode::Search {
            return;
        }
        for &u in &facts.uses[v.0] {
            match u {
                Use::Inst { block, index } => self.dirty(block, None, Some(index)),
                Use::Arg { block, edge, index } => {
                    let dst = f.blocks[&block].term.edges().nth(edge).unwrap().dst;
                    self.dirty(dst, Some(index), None);
                }
                Use::Cond(_) | Use::Ret(_) => {}
            }
        }
    }

    fn dirty(&mut self, b: BlockId, param: Option<usize>, inst: Option<usize>) {
        if let Some(k) = param {
            self.dirty_params.entry(b).or_default().insert(k);
        }
        if let Some(i) = inst {
            self.dirty_insts.entry(b).or_default().insert(i);
        }
        let r = self.rank[&b];
        if !self.queued[r] {
            self.queued[r] = true;
            self.queue.push(Reverse(r));
        }
    }

    fn edge_difference(&mut self, pred: BlockId, slot: usize, difference: Bdd) -> Bdd {
        let guard = match self.guards.get(&(pred.0, slot)) {
            Some(&g) => g,
            None => {
                let mut guard = self.reachable(pred);
                if let Term::CondBr { cond, .. } = self.f.blocks[&pred].term {
                    let bit = self.bit(cond);
                    let taken = if slot == 0 { bit } else { self.not(bit) };
                    guard = self.and(guard, taken);
                }
                self.guards.insert((pred.0, slot), guard);
                guard
            }
        };
        let difference = self.and(difference, guard);
        self.logic.image(self.f, self.facts, pred, slot, difference)
    }

    fn settle(&mut self) {
        match self.mode {
            Mode::Search => {
                while let Some(Reverse(r)) = self.queue.pop() {
                    self.queued[r] = false;
                    self.transfer(self.facts.order[r]);
                    if self.stopped {
                        return;
                    }
                }
            }
            Mode::Direct => loop {
                let mut changed = false;
                for i in 0..self.facts.order.len() {
                    if !std::mem::take(&mut self.stale[i]) {
                        continue;
                    }
                    let b = self.facts.order[i];
                    let safe = self.safe;
                    self.rose = false;
                    self.transfer(b);
                    if self.stopped {
                        return;
                    }
                    if self.safe != safe {
                        self.stale.fill(true);
                        changed = true;
                    }
                    if self.rose {
                        changed = true;
                        for e in self.f.blocks[&b].term.edges() {
                            self.stale[self.rank[&e.dst]] = true;
                        }
                    }
                }
                if !changed {
                    return;
                }
            },
        }
    }

    fn transfer(&mut self, b: BlockId) {
        let (f, facts) = (self.f, self.facts);
        let block = &f.blocks[&b];
        let params: Vec<usize> = match self.mode {
            Mode::Search => self
                .dirty_params
                .remove(&b)
                .map_or_else(Vec::new, |s| s.into_iter().collect()),
            Mode::Direct => (0..block.params.len()).collect(),
        };
        if !params.is_empty() && b != f.entry {
            let reachable = self.reachable(b);
            let arrivals = self.arrivals.get(&b).cloned();
            for k in params {
                let param = block.params[k].0;
                if !self.logic.carried(param) {
                    continue;
                }
                let mut d = arrivals.as_ref().map_or(Bdd::FALSE, |a| a[k]);
                let mut word = d;
                let is_word = facts.lane_word[param.0];
                let mode = if is_word {
                    self.logic.materialized(facts, param)
                } else {
                    Bdd::FALSE
                };
                for &(pred, slot) in &facts.incoming[&b] {
                    let arg = f.blocks[&pred].term.edges().nth(slot).unwrap().args[k];
                    let x = self.h[arg.0];
                    if x != Bdd::FALSE {
                        let image = self.edge_difference(pred, slot, x);
                        d = self.or(d, image);
                    }
                    if mode != Bdd::FALSE {
                        let x = self.whole(arg);
                        if x != Bdd::FALSE {
                            let image = self.edge_difference(pred, slot, x);
                            word = self.or(word, image);
                        }
                    }
                }
                if d != Bdd::FALSE {
                    d = self.and(d, reachable);
                    self.raise_h(param, d);
                }
                if is_word {
                    let word = self.and(word, reachable);
                    let x = self.logic.m.ite(mode, word, d);
                    self.raise_word(param, x);
                }
            }
        }
        if self.mode == Mode::Search && self.dirty_insts.get(&b).is_none_or(|s| s.is_empty()) {
            return;
        }
        for (index, inst) in block.insts.iter().enumerate() {
            if self.mode == Mode::Search
                && !self
                    .dirty_insts
                    .get_mut(&b)
                    .is_some_and(|s| s.remove(&index))
            {
                continue;
            }
            let x = self.inst(b, index, inst);
            if self.stopped {
                return;
            }
            for (m, v) in inst.outputs().into_iter().enumerate() {
                let x = match inst {
                    Inst::Effect {
                        op: EffectOp::Wave(WaveOp::Wmma),
                        inputs,
                        ..
                    } => self.wmma_output(inputs, m),
                    _ => x,
                };
                self.raise_h(v, x);
                if facts.lane_word[v.0] {
                    let mode = self.logic.materialized(facts, v);
                    if mode == Bdd::FALSE {
                        self.raise_word(v, x);
                    } else {
                        let word = self.word_difference(inst, v);
                        let y = self.logic.m.ite(mode, word, x);
                        self.raise_word(v, y);
                    }
                }
            }
        }
    }

    fn some_lane(&mut self, x: Bdd) -> Bdd {
        let varying: Vec<u32> = self
            .logic
            .support(x)
            .iter()
            .copied()
            .filter(|&n| !self.logic.uniform_atom(self.facts, n))
            .collect();
        self.logic.exists(&varying, x)
    }

    fn read_from(&mut self, h: Bdd, source: impl Fn(usize) -> u32) -> Bdd {
        let mut seen: HashMap<u32, Bdd> = HashMap::default();
        let mut any = Bdd::FALSE;
        for l in 0..32 {
            let s = source(l);
            let there = match seen.get(&s) {
                Some(&there) => there,
                None => {
                    let at = self.logic.at_lane(h, s);
                    let there = self.some_lane(at);
                    seen.insert(s, there);
                    there
                }
            };
            let here = self.logic.lanes(|x| x == l as u32);
            let reads = self.and(here, there);
            any = self.or(any, reads);
        }
        any
    }

    fn read_from_any(&mut self, h: Bdd, sources: impl Fn(usize) -> u32) -> Bdd {
        let mut seen: HashMap<u32, Bdd> = HashMap::default();
        let mut any = Bdd::FALSE;
        for l in 0..32 {
            let set = sources(l);
            let there = match seen.get(&set) {
                Some(&there) => there,
                None => {
                    let mut there = Bdd::FALSE;
                    for s in 0..32u32 {
                        if set >> s & 1 == 1 {
                            let at = self.logic.at_lane(h, s);
                            let read = self.some_lane(at);
                            there = self.or(there, read);
                        }
                    }
                    seen.insert(set, there);
                    there
                }
            };
            let here = self.logic.lanes(|x| x == l as u32);
            let reads = self.and(here, there);
            any = self.or(any, reads);
        }
        any
    }

    fn masked_read(&mut self, mask: ValueId, x: ValueId) -> Bdd {
        let hm = self.h[mask.0];
        let hx = self.whole(x);
        if hm == Bdd::FALSE && hx == Bdd::FALSE {
            return Bdd::FALSE;
        }
        let fm = self.bit(mask);
        let set = self.and(fm, hx);
        let read = self.or(hm, set);
        self.some_lane(read)
    }

    fn other_lanes(&mut self, x: Bdd) -> Bdd {
        if !self.logic.lane_dependent(x) {
            return self.some_lane(x);
        }
        let mut any = Bdd::FALSE;
        for l in 0..32u32 {
            let at = self.logic.at_lane(x, l);
            let there = self.some_lane(at);
            if there == Bdd::FALSE {
                continue;
            }
            let elsewhere = self.logic.lanes(|y| y != l);
            let reads = self.and(elsewhere, there);
            any = self.or(any, reads);
        }
        any
    }

    fn query(&mut self, x: Bdd, hx: Bdd, tag: Bdd) -> Bdd {
        let whole = self.or(x, hx);
        let others = self.other_lanes(whole);
        let absent = self.not(x);
        let differs = self.and(absent, others);
        let differs = self.and(differs, tag);
        if differs == Bdd::FALSE {
            return hx;
        }
        self.or(hx, differs)
    }

    fn word_difference(&mut self, inst: &Inst, value: ValueId) -> Bdd {
        match inst {
            Inst::Effect {
                provenance,
                op: EffectOp::Wave(WaveOp::Ballot),
                inputs,
                ..
            } => {
                if !self.faithful[value.0] {
                    let whole = self.logic.materialized(self.facts, value);
                    self.demand(*provenance, whole);
                }
                let x = self.h[inputs[0].0];
                self.some_lane(x)
            }
            Inst::Core {
                op: Op::Select(c, a, b),
                ..
            } => {
                let (wa, wb) = (self.whole(*a), self.whole(*b));
                let arms = if wa == Bdd::FALSE && wb == Bdd::FALSE {
                    Bdd::FALSE
                } else {
                    let fc = self.bit(*c);
                    self.logic.m.ite(fc, wa, wb)
                };
                self.or(self.h[c.0], arms)
            }
            _ => self.any_of(&inst.operands()),
        }
    }

    fn absorbs(&mut self, fx: Bdd, hx: Bdd, conjunction: bool) -> Bdd {
        let value = if conjunction { self.not(fx) } else { fx };
        let settled = self.not(hx);
        self.and(value, settled)
    }

    fn settles(&mut self, v: ValueId, op: Op) -> bool {
        if let Some(&known) = self.settles.get(&v) {
            return known;
        }
        let known = settled(self.f, self.facts, op);
        self.settles.insert(v, known);
        known
    }

    fn inst(&mut self, b: BlockId, index: usize, inst: &Inst) -> Bdd {
        let (f, facts) = (self.f, self.facts);
        match inst {
            Inst::Core { value, ty, op } => match *op {
                Op::Const(..) | Op::Env(_) => Bdd::FALSE,
                Op::Select(c, x, y) => {
                    let (hx, hy) = if facts.lane_word[value.0] {
                        (self.h[x.0], self.h[y.0])
                    } else {
                        (self.whole(x), self.whole(y))
                    };
                    let same = x == y || facts.constant(f, x).is_some_and(|k| facts.constant(f, y) == Some(k));
                    if same {
                        return self.or(hx, hy);
                    }
                    let hc = self.h[c.0];
                    let arms = if hx == Bdd::FALSE && hy == Bdd::FALSE {
                        Bdd::FALSE
                    } else {
                        let fc = self.bit(c);
                        let taken = self.and(fc, hx);
                        let nfc = self.not(fc);
                        let other = self.and(nfc, hy);
                        self.or(taken, other)
                    };
                    self.or(hc, arms)
                }
                Op::Int(k @ (IntOp::And | IntOp::Or), x, y)
                    if *ty == Ty::I1 || facts.lane_word[value.0] =>
                {
                    let (hx, hy) = (self.h[x.0], self.h[y.0]);
                    if hx == Bdd::FALSE && hy == Bdd::FALSE {
                        return Bdd::FALSE;
                    }
                    let (fx, fy) = if *ty == Ty::I1 {
                        (self.bit(x), self.bit(y))
                    } else {
                        (self.view(x), self.view(y))
                    };
                    let conjunction = k == IntOp::And;
                    let zx = self.absorbs(fx, hx, conjunction);
                    let zy = self.absorbs(fy, hy, conjunction);
                    let either = self.or(hx, hy);
                    let open_x = self.not(zx);
                    let open_y = self.not(zy);
                    let open = self.and(open_x, open_y);
                    self.and(either, open)
                }
                Op::Int(IntOp::Xor, x, y) if facts.lane_word[value.0] => {
                    self.or(self.h[x.0], self.h[y.0])
                }
                Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Mul), x, y)
                    if [x, y].iter().any(|&v| {
                        let ones = if *ty == Ty::I64 { u64::MAX } else { u32::MAX as u64 };
                        facts.constant(f, v) == Some(if k == IntOp::Or { ones } else { 0 })
                    }) =>
                {
                    Bdd::FALSE
                }
                Op::Convert(Cvt::Bitcast, Ty::I32, x) if facts.lane_word[value.0] => self.h[x.0],
                Op::Convert(Cvt::Trunc, Ty::I1, s) => match projected_word(f, facts, s) {
                    Some(w) if facts.lane_word[w.0] => self.h[w.0],
                    _ => self.h[s.0],
                },
                Op::Int(IntOp::LShr, w, lane)
                    if facts.lane_word[w.0] && facts.is_lane_id(f, lane) =>
                {
                    self.h[w.0]
                }
                Op::Cmp(IntPred::Eq | IntPred::Ne, x, y) if lane_test(f, facts, x, y).is_some() => {
                    let w = lane_test(f, facts, x, y).unwrap();
                    let mode = self.logic.materialized(facts, w);
                    if mode == Bdd::TRUE {
                        return self.whole(w);
                    }
                    let fw = self.view(w);
                    let hw = self.h[w.0];
                    let tag = self.logic.tag(Choice::Word(w));
                    let local = self.query(fw, hw, tag);
                    let whole = self.whole(w);
                    self.logic.m.ite(mode, whole, local)
                }
                other if self.settles(*value, other) => Bdd::FALSE,
                _ => self.any_of(&inst.operands()),
            },
            Inst::Target { args, .. } => {
                let operands = self.any_of(args.values());
                let loaded = self.loaded.get(&(b, index)).copied().unwrap_or(Bdd::FALSE);
                self.or(operands, loaded)
            }
            Inst::Packet { .. } => unreachable!("a packet query in a wave program"),
            Inst::Effect {
                provenance,
                op,
                inputs,
                outputs,
            } => match op {
                EffectOp::Wave(WaveOp::Any) => {
                    let (out, x) = (outputs[0].0, inputs[0]);
                    let local = self.logic.local(Choice::Query(out));
                    if !self.masked[x.0] {
                        let kept = self.not(local);
                        self.demand(*provenance, kept);
                    }
                    let hx = self.h[x.0];
                    if local == Bdd::FALSE {
                        return self.some_lane(hx);
                    }
                    let fx = self.bit(x);
                    let tag = self.logic.tag(Choice::Query(out));
                    let wave = self.logic.wave_answer(f, facts, out);
                    let tag = self.and(tag, wave);
                    let answered = self.query(fx, hx, tag);
                    if local == Bdd::TRUE {
                        return answered;
                    }
                    let gathered = self.some_lane(hx);
                    self.logic.m.ite(local, answered, gathered)
                }
                EffectOp::Wave(WaveOp::Ballot) => self.h[inputs[0].0],
                EffectOp::Wave(WaveOp::ReadFirstLane) => {
                    let (x, mask) = (inputs[0], inputs[1]);
                    if facts.uniform[x.0] {
                        return self.whole(x);
                    }
                    if !self.masked[mask.0] {
                        self.demand(*provenance, Bdd::TRUE);
                    }
                    let first = self.masked_read(mask, x);
                    let hx = self.whole(x);
                    if hx == Bdd::FALSE {
                        return first;
                    }
                    let fm = self.bit(mask);
                    let clear = self.not(fm);
                    let unset = self.and(clear, hx);
                    let zero = self.some_lane(unset);
                    let zero = self.and(clear, zero);
                    self.or(first, zero)
                }
                EffectOp::Wave(WaveOp::ReadLane) => {
                    let (x, selector) = (inputs[0], inputs[1]);
                    let own = self.whole(selector);
                    let hx = self.whole(x);
                    let read = match (self.logic.lane_function(f, facts, selector), constant_choices(f, facts, selector)) {
                        (Some(sources), _) => self.read_from(hx, |l| sources[l] & 31),
                        (None, Some(lanes)) => {
                            let mut any = Bdd::FALSE;
                            for lane in lanes {
                                let there = self.logic.at_lane(hx, lane as u32 & 31);
                                let there = self.some_lane(there);
                                any = self.or(any, there);
                            }
                            any
                        }
                        (None, None) => self.some_lane(hx),
                    };
                    self.or(own, read)
                }
                EffectOp::Wave(WaveOp::WriteLane) => {
                    let written = self.any_of(&inputs[..2]);
                    let written = self.logic.at_lane(written, 0);
                    let written = self.some_lane(written);
                    let old = self.any_of(&inputs[2..]);
                    match facts.constant(f, inputs[1]) {
                        Some(k) => {
                            let target = self.logic.lanes(|l| l as u64 == k & 31);
                            self.logic.m.ite(target, written, old)
                        }
                        None => match self.logic.lane_is(f, facts, b, inputs[1]) {
                            Some(target) => self.logic.m.ite(target, written, old),
                            None => self.or(old, written),
                        },
                    }
                }
                EffectOp::Wave(op @ (WaveOp::Bpermute | WaveOp::BpermuteFi)) => {
                    let (index, x, mask) = (inputs[0], inputs[1], inputs[2]);
                    let read = match self.logic.lane_function(f, facts, index) {
                        Some(indices) => {
                            let hx = self.whole(x);
                            let h = if *op == WaveOp::Bpermute {
                                let hm = self.h[mask.0];
                                let fm = self.bit(mask);
                                let set = self.and(fm, hx);
                                self.or(hm, set)
                            } else {
                                hx
                            };
                            self.read_from(h, |l| (indices[l] >> 2) & 31)
                        }
                        None => match self.logic.lane_bits(f, facts, index) {
                            Some(bits) if bits.iter().any(|&(m, _)| (m >> 2) & 31 != 0) => {
                                let hx = self.whole(x);
                                let h = if *op == WaveOp::Bpermute {
                                    let hm = self.h[mask.0];
                                    let fm = self.bit(mask);
                                    let set = self.and(fm, hx);
                                    self.or(hm, set)
                                } else {
                                    hx
                                };
                                self.read_from_any(h, |l| {
                                    let (m, v) = bits[l];
                                    let (m, v) = ((m >> 2) & 31, (v >> 2) & 31);
                                    (0..32u32).filter(|s| s & m == v & m).fold(0u32, |set, s| set | 1 << s)
                                })
                            }
                            _ if *op == WaveOp::Bpermute => self.masked_read(mask, x),
                            _ => {
                                let hx = self.whole(x);
                                self.some_lane(hx)
                            }
                        },
                    };
                    let own = self.whole(index);
                    self.or(own, read)
                }
                EffectOp::Wave(WaveOp::Wmma) => {
                    let mut any = Bdd::FALSE;
                    for m in 0..8 {
                        let output = self.wmma_output(inputs, m);
                        any = self.or(any, output);
                    }
                    any
                }
                EffectOp::Memory {
                    op: MemoryOp::Load(_),
                    ..
                } => {
                    let fp = self.bit(inputs[1]);
                    let absent = self.not(fp);
                    let operands = self.any_of(&inputs[..2]);
                    let differs = self.or(operands, absent);
                    let loaded = self.loaded.get(&(b, index)).copied().unwrap_or(Bdd::FALSE);
                    self.or(differs, loaded)
                }
                EffectOp::Memory {
                    op: MemoryOp::Fence,
                    ..
                } => Bdd::FALSE,
                EffectOp::Memory { op: memory, .. } => {
                    let m = memory.mask_input();
                    let pred = inputs[m];
                    let reachable = self.reachable(b);
                    let hp = self.h[pred.0];
                    if hp != Bdd::FALSE {
                        let happens = self.and(hp, reachable);
                        self.require(
                            b,
                            index,
                            "whether a store happens depends on the other lanes",
                            happens,
                        );
                        if self.stopped {
                            return Bdd::FALSE;
                        }
                    }
                    let fp = self.bit(pred);
                    let operands = self.any_of(&inputs[..m]);
                    if operands != Bdd::FALSE {
                        let performed = self.and(fp, reachable);
                        let writes = self.and(performed, operands);
                        self.require(
                            b,
                            index,
                            "what a store writes depends on the other lanes",
                            writes,
                        );
                        if self.stopped {
                            return Bdd::FALSE;
                        }
                    }
                    if let Some(&order) = self.reordered.get(&(b, index)) {
                        let performed = self.and(fp, reachable);
                        let reordered = self.and(performed, order);
                        self.require(
                            b,
                            index,
                            "another lane may write these bytes on the other side of the store",
                            reordered,
                        );
                        if self.stopped {
                            return Bdd::FALSE;
                        }
                    }
                    let absent = self.not(fp);
                    let differs = self.or(operands, absent);
                    let loaded = self.loaded.get(&(b, index)).copied().unwrap_or(Bdd::FALSE);
                    self.or(differs, loaded)
                }
                EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => {
                    self.any_of(inputs)
                }
            },
        }
    }

    fn detours(&mut self) -> BTreeMap<BlockId, Vec<Bdd>> {
        let (f, facts) = (self.f, self.facts);
        let mut arrivals = BTreeMap::new();
        for &b in &facts.order {
            let Term::CondBr { cond, yes, no } = &f.blocks[&b].term else {
                continue;
            };
            let hc = self.h[cond.0];
            if hc == Bdd::FALSE || yes.dst == no.dst {
                continue;
            }
            let version = self.version.get(&b).copied().unwrap_or(0);
            if self.detoured.insert(b, version) == Some(version) {
                continue;
            }
            let fc = self.bit(*cond);
            let nfc = self.not(fc);
            let reachable = self.reachable(b);
            for (wave, lane, taken) in [(0, 1, nfc), (1, 0, fc)] {
                let assume = self.and(hc, taken);
                let assume = self.and(assume, reachable);
                let assume = self.and(assume, self.safe);
                if assume == Bdd::FALSE {
                    continue;
                }
                let parts = match self.mode {
                    Mode::Search => vec![Bdd::TRUE],
                    Mode::Direct => {
                        let all_local = self.logic.all_local();
                        vec![all_local, self.not(all_local)]
                    }
                };
                for part in parts {
                    let sliced = self.and(assume, part);
                    if sliced == Bdd::FALSE {
                        continue;
                    }
                    Explore::new(self, b, sliced).run(wave, lane, &mut arrivals);
                    if self.stopped {
                        return arrivals;
                    }
                }
            }
        }
        arrivals
    }
}

fn meets_every_lane(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Effect {
            op: EffectOp::BarrierSignal { .. }
                | EffectOp::BarrierWait
                | EffectOp::Wave(
                    WaveOp::ReadLane
                        | WaveOp::WriteLane
                        | WaveOp::Bpermute
                        | WaveOp::BpermuteFi
                        | WaveOp::Wmma
                ),
            ..
        }
    )
}

const WAVE: usize = 0;
const LANE: usize = 1;
const JOINT: usize = 2;
const PATHS: u32 = 1 << 15;

#[derive(Clone, PartialEq, Eq, Hash)]
enum Form {
    Value(ValueId),
    Core(Ty, Op),
    Target(TargetOp, Vec<usize>, usize),
    Load(Space, MemSize, usize),
    Hazard(BlockId, usize, usize),
    Opaque(usize, ValueId),
    Linear(Ty, Vec<(usize, u64)>, u64),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Desc {
    same: Option<usize>,
    bits: Option<Bdd>,
}

struct Pair {
    cond: Bdd,
    sides: [Vec<Desc>; 2],
    writes: [Vec<Write>; 2],
    entries: BTreeMap<(Key, usize), (Bdd, [Vec<Desc>; 2])>,
}

#[derive(Clone, Copy, PartialEq)]
struct Write {
    at: (BlockId, usize),
    space: Space,
    op: MemoryOp,
    address: Option<usize>,
    data: Option<usize>,
    mask: Bdd,
    partnered: bool,
}

type Key = [Option<BlockId>; 2];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Leaf {
    Term(usize),
    Value(ValueId),
}

struct Evaluation {
    side: usize,
    block: BlockId,
    cond: Bdd,
    descs: HashMap<ValueId, Desc>,
    writes: Vec<Write>,
    stored: bool,
}

struct Explore<'c, 'a> {
    check: &'c mut Check<'a>,
    branch: BlockId,
    assume: Bdd,
    unreliable: HashMap<u32, bool>,
    decided: HashMap<(Bdd, Bdd, usize), Option<bool>>,
    terms: Vec<Form>,
    types: Vec<Ty>,
    leaves: Vec<Bdd>,
    index: HashMap<Form, usize>,
    visited: BTreeSet<BlockId>,
    headers: BTreeSet<BlockId>,
    paths: u32,
}

impl<'c, 'a> Explore<'c, 'a> {
    fn new(check: &'c mut Check<'a>, branch: BlockId, assume: Bdd) -> Self {
        let headers = (0..check.loops.count()).map(|l| check.facts.order[check.loops.header(l)]).collect();
        Self {
            check,
            branch,
            assume,
            unreliable: HashMap::default(),
            decided: HashMap::default(),
            terms: Vec::new(),
            types: Vec::new(),
            leaves: Vec::new(),
            index: HashMap::default(),
            visited: BTreeSet::new(),
            headers,
            paths: PATHS,
        }
    }

    fn logic(&mut self) -> &mut Logic {
        &mut self.check.logic
    }

    fn fresh(&mut self, side: usize, v: ValueId, position: u32) -> Bdd {
        self.logic().atom(Atom::Fresh(side, v, position))
    }

    fn linear_parts(&self, t: usize) -> (Vec<(usize, u64)>, u64) {
        match &self.terms[t] {
            Form::Linear(_, terms, constant) => (terms.clone(), *constant),
            Form::Core(_, Op::Const(_, k)) => (Vec::new(), *k),
            _ => (vec![(t, 1)], 0),
        }
    }

    fn combine(&self, ty: Ty, k: IntOp, x: usize, y: usize) -> Option<(Vec<(usize, u64)>, u64)> {
        let (mut xs, xk) = self.linear_parts(x);
        let (ys, yk) = self.linear_parts(y);
        let scale = |terms: &[(usize, u64)], constant: u64, by: u64| -> (Vec<(usize, u64)>, u64) {
            (terms.iter().map(|&(t, c)| (t, c.wrapping_mul(by))).collect(), constant.wrapping_mul(by))
        };
        Some(match k {
            IntOp::Add => {
                xs.extend(ys);
                (xs, xk.wrapping_add(yk))
            }
            IntOp::Sub => {
                xs.extend(ys.iter().map(|&(t, c)| (t, c.wrapping_neg())));
                (xs, xk.wrapping_sub(yk))
            }
            IntOp::Mul if ys.is_empty() => scale(&xs, xk, yk),
            IntOp::Mul if xs.is_empty() => scale(&ys, yk, xk),
            IntOp::Shl if ys.is_empty() && yk < 32 => scale(&xs, xk, 1u64 << yk),
            _ => return None,
        })
        .filter(|_| matches!(ty, Ty::I32 | Ty::I64))
    }

    fn intern_linear(&mut self, ty: Ty, terms: Vec<(usize, u64)>, constant: u64) -> usize {
        let mask = if ty == Ty::I64 { u64::MAX } else { (1u64 << ty.bits()) - 1 };
        let mut merged: BTreeMap<usize, u64> = BTreeMap::new();
        for (t, c) in terms {
            let e = merged.entry(t).or_insert(0);
            *e = e.wrapping_add(c) & mask;
        }
        let terms: Vec<(usize, u64)> = merged.into_iter().filter(|&(_, c)| c != 0).collect();
        let constant = constant & mask;
        match terms.as_slice() {
            [] => self.intern(ty, Form::Core(ty, Op::Const(ty, constant))),
            [(t, 1)] if constant == 0 => *t,
            _ => self.intern(ty, Form::Linear(ty, terms, constant)),
        }
    }

    fn constant_of(&self, t: usize) -> Option<u64> {
        match self.terms[t] {
            Form::Core(_, Op::Const(_, k)) => Some(k),
            _ => None,
        }
    }

    fn intern(&mut self, ty: Ty, form: Form) -> usize {
        let form = match form {
            Form::Core(t, Op::Int(k @ (IntOp::Add | IntOp::Mul | IntOp::And | IntOp::Or | IntOp::Xor), x, y)) if x.0 > y.0 => {
                Form::Core(t, Op::Int(k, y, x))
            }
            Form::Core(t, Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), x, y)) if x.0 > y.0 => Form::Core(t, Op::Cmp(p, y, x)),
            other => other,
        };
        match &form {
            Form::Core(_, Op::Convert(Cvt::Bitcast, _, a)) => {
                let a = a.0;
                if self.types[a] == ty {
                    return a;
                }
                if let Form::Core(_, Op::Convert(Cvt::Bitcast, _, b)) = self.terms[a] {
                    if self.types[b.0] == ty {
                        return b.0;
                    }
                }
            }
            Form::Core(_, Op::UnpackLo(p) | Op::UnpackHi(p)) => {
                let low = matches!(form, Form::Core(_, Op::UnpackLo(_)));
                let half = |e: &mut Self, x: ValueId| {
                    let op = if low {
                        Op::UnpackLo(x)
                    } else {
                        Op::UnpackHi(x)
                    };
                    e.intern(Ty::I32, Form::Core(Ty::I32, op))
                };
                match self.terms[p.0].clone() {
                    Form::Core(_, Op::Pack64(lo, hi)) => return if low { lo.0 } else { hi.0 },
                    Form::Core(_, Op::Const(_, k)) => {
                        let word = if low { k & 0xffff_ffff } else { k >> 32 };
                        return self.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, word)));
                    }
                    Form::Core(_, Op::Convert(Cvt::ZExt, _, a)) if self.types[a.0] == Ty::I32 => {
                        return if low {
                            a.0
                        } else {
                            self.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, 0)))
                        };
                    }
                    Form::Core(_, Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), x, y)) => {
                        let (x, y) = (half(self, x), half(self, y));
                        return self.intern(
                            Ty::I32,
                            Form::Core(Ty::I32, Op::Int(k, ValueId(x), ValueId(y))),
                        );
                    }
                    Form::Core(_, Op::Int(k @ (IntOp::LShr | IntOp::Shl), x, s))
                        if self.constant_of(s.0) == Some(32) =>
                    {
                        let other = match (k, low) {
                            (IntOp::LShr, true) => Some(Op::UnpackHi(x)),
                            (IntOp::Shl, false) => Some(Op::UnpackLo(x)),
                            _ => None,
                        };
                        return match other {
                            Some(op) => self.intern(Ty::I32, Form::Core(Ty::I32, op)),
                            None => {
                                self.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, 0)))
                            }
                        };
                    }
                    _ => {}
                }
            }
            Form::Core(_, Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), x, y)) => {
                return self.gathered(ty, *k, x.0, y.0);
            }
            Form::Core(_, Op::Pack64(lo, hi)) => {
                if let (Form::Core(_, Op::UnpackLo(a)), Form::Core(_, Op::UnpackHi(b))) =
                    (&self.terms[lo.0], &self.terms[hi.0])
                {
                    if a == b {
                        return a.0;
                    }
                }
            }
            Form::Core(_, Op::Int(k @ (IntOp::Add | IntOp::Sub | IntOp::Mul | IntOp::Shl), x, y))
                if matches!(ty, Ty::I32 | Ty::I64) =>
            {
                if let Some((terms, constant)) = self.combine(ty, *k, x.0, y.0) {
                    return self.intern_linear(ty, terms, constant);
                }
                if *k == IntOp::Mul {
                    if let Some(t) = self.multiplied(ty, x.0, y.0) {
                        return t;
                    }
                }
            }
            _ => {}
        }
        self.insert(ty, form)
    }

    fn multiplied(&mut self, ty: Ty, x: usize, y: usize) -> Option<usize> {
        let (xs, xk) = self.linear_parts(x);
        let (ys, yk) = self.linear_parts(y);
        if xs.len() * ys.len() > 16 {
            return None;
        }
        let mut terms: Vec<(usize, u64)> = Vec::new();
        terms.extend(xs.iter().map(|&(t, c)| (t, c.wrapping_mul(yk))));
        terms.extend(ys.iter().map(|&(t, c)| (t, c.wrapping_mul(xk))));
        for &(a, c) in &xs {
            for &(b, d) in &ys {
                let mut factors = self.operands_of(IntOp::Mul, a);
                factors.extend(self.operands_of(IntOp::Mul, b));
                if factors.len() > 8 {
                    return None;
                }
                factors.sort_unstable();
                let m = self.chained(ty, IntOp::Mul, &factors);
                terms.push((m, c.wrapping_mul(d)));
            }
        }
        Some(self.intern_linear(ty, terms, xk.wrapping_mul(yk)))
    }

    fn operands_of(&self, k: IntOp, t: usize) -> Vec<usize> {
        match self.terms[t] {
            Form::Core(_, Op::Int(op, a, b)) if op == k => {
                let mut out = self.operands_of(k, a.0);
                out.extend(self.operands_of(k, b.0));
                out
            }
            _ => vec![t],
        }
    }

    fn chained(&mut self, ty: Ty, k: IntOp, parts: &[usize]) -> usize {
        let mut t = parts[0];
        for &p in &parts[1..] {
            t = self.insert(ty, Form::Core(ty, Op::Int(k, ValueId(t), ValueId(p))));
        }
        t
    }

    fn gathered(&mut self, ty: Ty, k: IntOp, x: usize, y: usize) -> usize {
        let ones = if ty == Ty::I64 { u64::MAX } else { (1u64 << ty.bits()) - 1 };
        let mut parts = self.operands_of(k, x);
        parts.extend(self.operands_of(k, y));
        let mut constant: Option<u64> = None;
        let mut rest: Vec<usize> = Vec::new();
        for p in parts {
            match self.constant_of(p) {
                Some(c) => {
                    constant = Some(match (k, constant) {
                        (_, None) => c & ones,
                        (IntOp::And, Some(a)) => a & c,
                        (IntOp::Or, Some(a)) => a | c,
                        (_, Some(a)) => (a ^ c) & ones,
                    })
                }
                None => rest.push(p),
            }
        }
        rest.sort_unstable();
        if k == IntOp::Xor {
            let mut kept: Vec<usize> = Vec::new();
            for p in rest {
                if kept.last() == Some(&p) {
                    kept.pop();
                } else {
                    kept.push(p);
                }
            }
            rest = kept;
        } else {
            rest.dedup();
        }
        let identity = if k == IntOp::And { ones } else { 0 };
        let absorbing = match k {
            IntOp::And => Some(0),
            IntOp::Or => Some(ones),
            _ => None,
        };
        if let Some(c) = constant.filter(|&c| Some(c) == absorbing) {
            return self.intern(ty, Form::Core(ty, Op::Const(ty, c)));
        }
        if let Some(c) = constant.filter(|&c| c != identity) {
            let t = self.intern(ty, Form::Core(ty, Op::Const(ty, c)));
            rest.push(t);
        }
        match rest.as_slice() {
            [] => self.intern(ty, Form::Core(ty, Op::Const(ty, identity))),
            _ => self.chained(ty, k, &rest),
        }
    }

    fn insert(&mut self, ty: Ty, form: Form) -> usize {
        if let Some(&t) = self.index.get(&form) {
            return t;
        }
        let leaves = match &form {
            Form::Value(v) => self.check.whole(*v),
            Form::Core(_, op) => {
                let mut children = Vec::new();
                op.map(|c| {
                    children.push(c.0);
                    c
                });
                let mut h = Bdd::FALSE;
                for c in children {
                    h = self.check.or(h, self.leaves[c]);
                }
                h
            }
            Form::Target(_, args, _) => {
                let mut h = Bdd::FALSE;
                for &c in args {
                    h = self.check.or(h, self.leaves[c]);
                }
                h
            }
            Form::Load(_, _, a) => self.leaves[*a],
            Form::Hazard(block, index, t) => {
                let loaded = self.check.loaded[&(*block, *index)];
                self.check.or(self.leaves[*t], loaded)
            }
            Form::Opaque(..) => Bdd::TRUE,
            Form::Linear(_, terms, _) => {
                let mut h = Bdd::FALSE;
                for &(c, _) in terms {
                    h = self.check.or(h, self.leaves[c]);
                }
                h
            }
        };
        let t = self.terms.len();
        self.terms.push(form.clone());
        self.types.push(ty);
        self.leaves.push(leaves);
        self.index.insert(form, t);
        t
    }

    fn reliable(&mut self, t: usize) -> bool {
        let leaves = self.leaves[t];
        if leaves == Bdd::FALSE {
            return true;
        }
        let assume = self.check.and(self.assume, self.check.safe);
        self.check.and(assume, leaves) == Bdd::FALSE
    }

    fn loaded(&mut self, ty: Ty, v: ValueId, t: usize) -> usize {
        match self.check.facts.site[v.0] {
            Site::Inst { block, index } if self.check.loaded.contains_key(&(block, index)) => {
                self.intern(ty, Form::Hazard(block, index, t))
            }
            _ => t,
        }
    }

    fn form_bits(&mut self, side: usize, v: ValueId, t: usize, view: bool) -> Bdd {
        if self.reliable(t) {
            self.logic().atom(Atom::Term(t, view))
        } else {
            self.fresh(side, v, 0)
        }
    }

    fn is_unreliable(&mut self, var: u32) -> bool {
        if let Some(&u) = self.unreliable.get(&var) {
            return u;
        }
        let u = match self.check.logic.atom_of(var) {
            Atom::Bit(v) | Atom::View(v) => {
                let h = self.check.h[v.0];
                h != Bdd::FALSE && self.check.and(self.assume, h) != Bdd::FALSE
            }
            _ => false,
        };
        self.unreliable.insert(var, u);
        u
    }

    fn unknowns(&mut self, g: Bdd, side: usize, forms: bool) -> Vec<u32> {
        let support = self.logic().support(g);
        support
            .iter()
            .copied()
            .filter(|&v| match self.check.logic.atom_of(v) {
                Atom::Fresh(PATH, ..) => false,
                Atom::Fresh(..) => true,
                Atom::Term(..) => forms,
                _ => side == WAVE && self.is_unreliable(v),
            })
            .collect()
    }

    fn decide(&mut self, g: Bdd, cond: Bdd, side: usize) -> Option<bool> {
        if let Some(k) = g.constant() {
            return Some(k);
        }
        let cond = self.check.and(cond, self.check.safe);
        if let Some(&d) = self.decided.get(&(g, cond, side)) {
            return d;
        }
        let unknown = self.unknowns(g, side, true);
        let holds = self.logic().forall(&unknown, g);
        let d = if self.logic().m.implies(cond, holds) {
            Some(true)
        } else {
            let ng = self.logic().m.not(g);
            let fails = self.logic().forall(&unknown, ng);
            if self.logic().m.implies(cond, fails) {
                Some(false)
            } else {
                None
            }
        };
        self.decided.insert((g, cond, side), d);
        d
    }

    fn weaken(&mut self, g: Bdd, side: usize) -> Bdd {
        let unknown = self.unknowns(g, side, false);
        self.logic().exists(&unknown, g)
    }

    fn start(&mut self, v: ValueId) -> Desc {
        let (f, facts) = (self.check.f, self.check.facts);
        let ty = f.types[v.0];
        let bits = match ty {
            Ty::I1 => Some(self.check.bit(v)),
            Ty::I32 if facts.viewed[v.0] => Some(self.check.view(v)),
            _ => None,
        };
        let same = Some(self.intern(ty, Form::Value(v)));
        Desc { same, bits }
    }

    fn fresh_support(&mut self, g: Bdd) -> Vec<u32> {
        let support = self.logic().support(g);
        support
            .iter()
            .copied()
            .filter(|&v| matches!(self.check.logic.atom_of(v), Atom::Fresh(..)))
            .collect()
    }

    fn canonical(&mut self, cond: Bdd, sides: [Vec<Desc>; 2]) -> (Bdd, [Vec<Desc>; 2]) {
        let mut order: Vec<u32> = Vec::new();
        for g in sides.iter().flatten().filter_map(|d| d.bits) {
            if self.has_fresh(g) {
                for v in self.fresh_support(g) {
                    if !order.contains(&v) {
                        order.push(v);
                    }
                }
            }
        }
        let absent: Vec<u32> = self
            .fresh_support(cond)
            .into_iter()
            .filter(|v| !order.contains(v))
            .collect();
        let cond = self.logic().exists(&absent, cond);
        let (paths, values): (Vec<u32>, Vec<u32>) = order
            .iter()
            .partition(|&&v| matches!(self.check.logic.atom_of(v), Atom::Fresh(PATH, ..)));
        assert!(paths.len() < PATHS as usize, "too many paths");
        let mut map: HashMap<u32, Bdd> = HashMap::default();
        let targets = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, JOINT, i as u32 + 1))
            .chain(paths.iter().enumerate().map(|(i, &v)| (v, PATH, PATHS + i as u32)));
        for (v, kind, position) in targets.collect::<Vec<_>>() {
            let atom = Atom::Fresh(kind, ValueId(0), position);
            if self.check.logic.atom_of(v) != atom {
                let target = self.logic().atom(atom);
                map.insert(v, target);
            }
        }
        let Some(&last) = map.keys().max() else {
            return (cond, sides);
        };
        let mut roots: Vec<Bdd> = sides.iter().flatten().filter_map(|d| d.bits).collect();
        roots.push(cond);
        let renamed = self.logic().m.compose_many(&roots, &|v| map.get(&v).copied(), last);
        let mut renamed = renamed.into_iter();
        let mut out = sides;
        for d in out.iter_mut().flatten() {
            if d.bits.is_some() {
                d.bits = renamed.next();
            }
        }
        (renamed.next().unwrap(), out)
    }

    fn join(&mut self, (oc, old): (Bdd, &[Vec<Desc>; 2]), (nc, new): (Bdd, &[Vec<Desc>; 2])) -> (Bdd, [Vec<Desc>; 2]) {
        let mut out = old.clone();
        let differ = old.iter().flatten().zip(new.iter().flatten()).any(|(o, n)| o.bits != n.bits);
        if !differ {
            for (d, n) in out.iter_mut().flatten().zip(new.iter().flatten()) {
                if d.same != n.same {
                    d.same = None;
                }
            }
            return (self.check.or(oc, nc), out);
        }
        assert!(self.paths > 0, "too many joins");
        self.paths -= 1;
        let path = self.fresh(PATH, ValueId(0), self.paths);
        for (d, n) in out.iter_mut().flatten().zip(new.iter().flatten()) {
            let same = if d.same == n.same { d.same } else { None };
            let bits = match (d.bits, n.bits) {
                (Some(a), Some(b)) if a == b => Some(a),
                (Some(a), Some(b)) => Some(self.logic().m.ite(path, a, b)),
                _ => None,
            };
            *d = Desc { same, bits };
        }
        (self.logic().m.ite(path, oc, nc), out)
    }

    fn joined(&mut self, entries: &BTreeMap<(Key, usize), (Bdd, [Vec<Desc>; 2])>) -> (Bdd, [Vec<Desc>; 2]) {
        let mut values = entries.values();
        let (c, s) = values.next().unwrap();
        let mut joined = (*c, s.clone());
        for (c, s) in values {
            joined = self.join((joined.0, &joined.1), (*c, s));
        }
        joined
    }

    fn has_fresh(&mut self, g: Bdd) -> bool {
        if let Some(&f) = self.check.fresh_in.get(&g) {
            return f;
        }
        let support = self.logic().support(g);
        let f = support
            .iter()
            .any(|&v| matches!(self.check.logic.atom_of(v), Atom::Fresh(..)));
        self.check.fresh_in.insert(g, f);
        f
    }

    fn merge(&mut self, side: usize, param: ValueId, cond: Bdd, old: Desc, new: Desc) -> Desc {
        let same = if old.same == new.same { old.same } else { None };
        let bits = match (old.bits, new.bits) {
            (Some(a), Some(b)) if a == b => Some(a),
            (Some(a), Some(b)) => match (self.decide(a, cond, side), self.decide(b, cond, side)) {
                (Some(x), Some(y)) if x == y => Some(Manager::constant(x)),
                _ => Some(self.between(side, param, a, b)),
            },
            _ => None,
        };
        Desc { same, bits }
    }

    fn between(&mut self, side: usize, param: ValueId, a: Bdd, b: Bdd) -> Bdd {
        if self.has_fresh(a) || self.has_fresh(b) {
            return self.fresh(side, param, 0);
        }
        let low = self.check.and(a, b);
        let high = self.check.or(a, b);
        if low == Bdd::FALSE && high == Bdd::TRUE {
            return self.fresh(side, param, 0);
        }
        let choice = self.fresh(side, param, u32::MAX);
        let open = self.check.and(choice, high);
        self.check.or(low, open)
    }

    fn positions(&mut self, x: BlockId) -> Rc<HashMap<ValueId, usize>> {
        if let Some(p) = self.check.positions.get(&x) {
            return p.clone();
        }
        let p: Rc<HashMap<ValueId, usize>> = Rc::new(
            self.check.f.blocks[&x]
                .params
                .iter()
                .enumerate()
                .map(|(j, &(v, _))| (v, j))
                .collect(),
        );
        self.check.positions.insert(x, p.clone());
        p
    }

    fn run(&mut self, wave: usize, lane: usize, arrivals: &mut BTreeMap<BlockId, Vec<Bdd>>) {
        let f = self.check.f;
        let edges: Vec<&Edge> = f.blocks[&self.branch].term.edges().collect();
        let (e0, e1) = (edges[wave], edges[lane]);
        let sides = [
            e0.args.iter().map(|&a| self.start(a)).collect(),
            e1.args.iter().map(|&a| self.start(a)).collect(),
        ];
        let first: Key = [Some(e0.dst), Some(e1.dst)];
        let exit = f.blocks.len();
        let priority = |check: &Check, key: &Key| -> usize {
            key.iter().map(|b| b.map_or(exit, |b| check.rank[&b])).sum()
        };
        let mut pairs: BTreeMap<Key, Pair> = BTreeMap::from([(
            first,
            Pair {
                cond: self.assume,
                sides: sides.clone(),
                writes: [Vec::new(), Vec::new()],
                entries: BTreeMap::from([(([None, None], 0), (self.assume, sides))]),
            },
        )]);
        let mut worklist = BTreeSet::from([(priority(self.check, &first), first)]);
        while let Some((_, key)) = worklist.pop_first() {
            let pair = &pairs[&key];
            let (cond, sides, writes) = (pair.cond, pair.sides.clone(), pair.writes.clone());
            let cond = self.check.and(cond, self.check.safe);
            if cond == Bdd::FALSE {
                continue;
            }
            let side = match key {
                [None, None] => {
                    self.match_writes(cond, &writes);
                    if self.check.stopped {
                        return;
                    }
                    continue;
                }
                [Some(a), Some(b)] if a == b => {
                    self.match_writes(cond, &writes);
                    if self.check.stopped {
                        return;
                    }
                    self.meet(a, cond, &sides, arrivals);
                    continue;
                }
                [Some(a), Some(b)] => {
                    if self
                        .check
                        .loops
                        .before(self.check.rank[&a], self.check.rank[&b])
                    {
                        WAVE
                    } else {
                        LANE
                    }
                }
                [Some(_), None] => WAVE,
                [None, Some(_)] => LANE,
            };
            let block = key[side].unwrap();
            let steps = self.step(side, block, &sides[side], cond, !writes[side].is_empty());
            if self.check.stopped {
                return;
            }
            for (slot, (dst, descs, constraint, done)) in steps.into_iter().enumerate() {
                let c = self.check.and(cond, constraint);
                let c = self.check.and(c, self.check.safe);
                if c == Bdd::FALSE {
                    continue;
                }
                let mut next = key;
                next[side] = dst;
                let incoming = if side == WAVE {
                    [descs, sides[LANE].clone()]
                } else {
                    [sides[WAVE].clone(), descs]
                };
                let looped = next.iter().flatten().any(|b| self.headers.contains(b));
                let (c, incoming) = if looped { self.canonical(c, incoming) } else { (c, incoming) };
                let mut traces = writes.clone();
                traces[side].extend(done);
                let merged = match pairs.get(&next) {
                    None => Pair {
                        cond: c,
                        sides: incoming.clone(),
                        writes: traces,
                        entries: BTreeMap::from([((key, slot), (c, incoming))]),
                    },
                    Some(old) => {
                        let (ocond, osides) = (old.cond, old.sides.clone());
                        if old.writes != traces {
                            for s in [WAVE, LANE] {
                                self.unmatched(c, &traces[s][..], s);
                                self.unmatched(ocond, &old.writes[s][..], s);
                            }
                            if self.check.stopped {
                                return;
                            }
                        }
                        let kept = old.writes.clone();
                        if !looped {
                            let mut entries = old.entries.clone();
                            let arrival = (c, incoming);
                            if entries.get(&(key, slot)) == Some(&arrival) {
                                continue;
                            }
                            let (mcond, msides) = match entries.insert((key, slot), arrival.clone()) {
                                None => self.join((ocond, &osides), (arrival.0, &arrival.1)),
                                Some(_) => self.joined(&entries),
                            };
                            let changed = mcond != ocond || msides != osides;
                            pairs.insert(
                                next,
                                Pair {
                                    cond: mcond,
                                    sides: msides,
                                    writes: kept,
                                    entries,
                                },
                            );
                            if changed {
                                worklist.insert((priority(self.check, &next), next));
                            }
                            continue;
                        }
                        let mcond = self.check.or(ocond, c);
                        let mut grew = mcond != ocond;
                        let mut merged = osides.clone();
                        for s in [WAVE, LANE] {
                            let Some(blk) = next[s] else { continue };
                            if osides[s] == incoming[s] {
                                continue;
                            }
                            let mut classes: Vec<((Bdd, Bdd), Bdd)> = Vec::new();
                            for (k, &(param, _)) in f.blocks[&blk].params.iter().enumerate() {
                                let (o, n) = (osides[s][k], incoming[s][k]);
                                if o == n {
                                    continue;
                                }
                                let shared = match (o.bits, n.bits) {
                                    (Some(ob), Some(nb)) => {
                                        let (no, nn) = (self.logic().m.not(ob), self.logic().m.not(nb));
                                        classes.iter().find_map(|&(pair, bits)| {
                                            if pair == (ob, nb) {
                                                Some(bits)
                                            } else if pair == (no, nn) {
                                                Some(self.check.logic.m.not(bits))
                                            } else {
                                                None
                                            }
                                        })
                                    }
                                    _ => None,
                                };
                                let m = match shared {
                                    Some(bits) => Desc {
                                        same: if o.same == n.same { o.same } else { None },
                                        bits: Some(bits),
                                    },
                                    None => {
                                        let m = self.merge(s, param, mcond, o, n);
                                        if let (Some(ob), Some(nb), Some(mb)) = (o.bits, n.bits, m.bits) {
                                            classes.push(((ob, nb), mb));
                                        }
                                        m
                                    }
                                };
                                if m != o {
                                    merged[s][k] = m;
                                    grew = true;
                                }
                            }
                        }
                        if !grew {
                            continue;
                        }
                        let (mcond, merged) = self.canonical(mcond, merged);
                        if merged == osides && mcond == ocond {
                            continue;
                        }
                        Pair {
                            cond: mcond,
                            sides: merged,
                            writes: kept,
                            entries: BTreeMap::new(),
                        }
                    }
                };
                pairs.insert(next, merged);
                worklist.insert((priority(self.check, &next), next));
            }
        }
    }

    fn step(
        &mut self,
        side: usize,
        x: BlockId,
        params: &[Desc],
        cond: Bdd,
        stored: bool,
    ) -> Vec<(Option<BlockId>, Vec<Desc>, Bdd, Vec<Write>)> {
        let f = self.check.f;
        let block = &f.blocks[&x];
        let mut ev = Evaluation {
            side,
            block: x,
            cond,
            descs: block
                .params
                .iter()
                .map(|&(v, _)| v)
                .zip(params.iter().copied())
                .collect(),
            writes: Vec::new(),
            stored,
        };
        self.visited.insert(ev.block);
        self.check_effects(&mut ev);
        if self.check.stopped {
            return Vec::new();
        }
        let followed: Vec<(usize, Bdd)> = match &block.term {
            Term::Ret(_) => return vec![(None, Vec::new(), Bdd::TRUE, ev.writes)],
            Term::Br(_) => vec![(0, Bdd::TRUE)],
            Term::CondBr { cond: c, .. } => {
                let g = match self.get(&mut ev, *c).bits {
                    Some(g) => g,
                    None => self.fresh(side, *c, 0),
                };
                match self.decide(g, cond, side) {
                    Some(true) => vec![(0, Bdd::TRUE)],
                    Some(false) => vec![(1, Bdd::TRUE)],
                    None => {
                        let ng = self.logic().m.not(g);
                        vec![(0, self.weaken(g, side)), (1, self.weaken(ng, side))]
                    }
                }
            }
        };
        let position = self.positions(x);
        let mut out = Vec::new();
        for (slot, constraint) in followed {
            let edge = block.term.edges().nth(slot).unwrap();
            let mut args = Vec::with_capacity(edge.args.len());
            for &arg in &edge.args {
                let d = match position.get(&arg) {
                    Some(&j) => params[j],
                    None => self.get(&mut ev, arg),
                };
                args.push(d);
            }
            out.push((Some(edge.dst), args, constraint, ev.writes.clone()));
        }
        out
    }

    fn get(&mut self, ev: &mut Evaluation, v: ValueId) -> Desc {
        if let Some(&d) = ev.descs.get(&v) {
            return d;
        }
        let inst = self
            .check
            .facts
            .inst(self.check.f, v)
            .expect("a detour reads a value its block does not define");
        self.compute(ev, inst);
        ev.descs[&v]
    }

    fn bits_of(&mut self, ev: &mut Evaluation, v: ValueId) -> Bdd {
        match self.get(ev, v).bits {
            Some(g) => g,
            None => self.fresh(ev.side, v, 0),
        }
    }

    fn formed(&mut self, side: usize, v: ValueId, t: Option<usize>) -> Desc {
        let (f, facts) = (self.check.f, self.check.facts);
        let ty = f.types[v.0];
        if ty == Ty::I1 || (ty == Ty::I32 && facts.viewed[v.0]) {
            let bits = match t {
                Some(t) => self.form_bits(side, v, t, ty == Ty::I32),
                None => self.fresh(side, v, 0),
            };
            Desc {
                same: t,
                bits: Some(bits),
            }
        } else {
            Desc {
                same: t,
                bits: None,
            }
        }
    }

    fn operand(&mut self, ev: &mut Evaluation, v: ValueId) -> usize {
        match self.get(ev, v).same {
            Some(t) => t,
            None => {
                let ty = self.check.f.types[v.0];
                self.intern(ty, Form::Opaque(ev.side, v))
            }
        }
    }

    fn form(&mut self, ev: &mut Evaluation, ty: Ty, op: Op) -> usize {
        let mut children = Vec::new();
        op.map(|c| {
            children.push(c);
            c
        });
        let terms: Vec<usize> = children.into_iter().map(|c| self.operand(ev, c)).collect();
        let mut next = terms.into_iter();
        let mapped = op.map(|_| ValueId(next.next().unwrap()));
        self.intern(ty, Form::Core(ty, mapped))
    }

    fn compute(&mut self, ev: &mut Evaluation, inst: &Inst) {
        let (f, facts) = (self.check.f, self.check.facts);
        let (side, cond) = (ev.side, ev.cond);
        match inst {
            Inst::Core { value, ty, op } => {
                let (value, ty, op) = (*value, *ty, *op);
                let lane_only = match ty {
                    Ty::I1 => true,
                    Ty::I32 => facts.viewed[value.0],
                    _ => false,
                };
                let lanes = if lane_only { self.logic().lane_function(f, facts, value) } else { None };
                let desc = match op {
                    _ if lanes.is_some() => {
                        let values = lanes.unwrap();
                        let bits = if ty == Ty::I1 {
                            self.logic().lanes(|l| values[l as usize] & 1 == 1)
                        } else {
                            self.logic().lanes(|l| values[l as usize] >> l & 1 == 1)
                        };
                        Desc {
                            same: Some(self.form(ev, ty, op)),
                            bits: Some(bits),
                        }
                    }
                    Op::Const(_, k) => {
                        let bits = match ty {
                            Ty::I1 => Some(Manager::constant(k != 0)),
                            Ty::I32 if facts.viewed[value.0] => Some(self.logic().word(k as u32)),
                            _ => None,
                        };
                        Desc {
                            same: Some(self.form(ev, ty, op)),
                            bits,
                        }
                    }
                    Op::Env(Env::ValidLane) => Desc {
                        same: Some(self.form(ev, ty, op)),
                        bits: Some(Bdd::TRUE),
                    },
                    Op::Select(c, a, b) => {
                        let gc = self.bits_of(ev, c);
                        match self.decide(gc, cond, side) {
                            Some(true) => self.get(ev, a),
                            Some(false) => self.get(ev, b),
                            None => {
                                let (da, db) = (self.get(ev, a), self.get(ev, b));
                                if da == db {
                                    da
                                } else {
                                    let same = self.form(ev, ty, op);
                                    let bits = match (da.bits, db.bits) {
                                        (Some(ga), Some(gb)) => {
                                            Some(self.logic().m.ite(gc, ga, gb))
                                        }
                                        _ => self.formed(side, value, Some(same)).bits,
                                    };
                                    Desc {
                                        same: Some(same),
                                        bits,
                                    }
                                }
                            }
                        }
                    }
                    Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b)
                        if matches!(ty, Ty::I1 | Ty::I32) =>
                    {
                        if ty == Ty::I32 && !facts.viewed[value.0] {
                            Desc {
                                same: Some(self.form(ev, ty, op)),
                                bits: None,
                            }
                        } else {
                            let ga = self.bits_of(ev, a);
                            let gb = self.bits_of(ev, b);
                            let m = &mut self.check.logic.m;
                            let g = match k {
                                IntOp::And => m.and(ga, gb),
                                IntOp::Or => m.or(ga, gb),
                                _ => m.xor(ga, gb),
                            };
                            Desc {
                                same: Some(self.form(ev, ty, op)),
                                bits: Some(g),
                            }
                        }
                    }
                    Op::Convert(Cvt::Bitcast, to, a) if f.types[a.0] == to => self.get(ev, a),
                    Op::Convert(Cvt::Trunc, Ty::I1, s) if projected_word(f, facts, s).is_some() => {
                        let w = projected_word(f, facts, s).unwrap();
                        let same = self.form(ev, ty, op);
                        Desc {
                            same: Some(same),
                            bits: Some(self.bits_of(ev, w)),
                        }
                    }
                    Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b) if f.types[a.0] == Ty::I1 => {
                        let ga = self.bits_of(ev, a);
                        let gb = self.bits_of(ev, b);
                        let m = &mut self.check.logic.m;
                        let g = if p == IntPred::Ne {
                            m.xor(ga, gb)
                        } else {
                            m.iff(ga, gb)
                        };
                        Desc {
                            same: Some(self.form(ev, ty, op)),
                            bits: Some(g),
                        }
                    }
                    Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b)
                        if lane_test(f, facts, a, b).is_some_and(|w| !facts.materialized[w.0]) =>
                    {
                        let w = lane_test(f, facts, a, b).unwrap();
                        let bit = self.bits_of(ev, w);
                        let answer = self.answer(side, value, bit, cond);
                        let g = if p == IntPred::Ne {
                            answer
                        } else {
                            self.logic().m.not(answer)
                        };
                        let mode = self.check.logic.materialized(facts, w);
                        let g = if mode == Bdd::FALSE {
                            g
                        } else {
                            let t = self.form(ev, ty, op);
                            let full = self.formed(side, value, Some(t)).bits.unwrap();
                            self.logic().m.ite(mode, full, g)
                        };
                        Desc {
                            same: None,
                            bits: Some(g),
                        }
                    }
                    _ => {
                        let t = self.form(ev, ty, op);
                        self.formed(side, value, Some(t))
                    }
                };
                ev.descs.insert(value, desc);
            }
            Inst::Effect {
                op,
                inputs,
                outputs,
                ..
            } => match op {
                EffectOp::Wave(WaveOp::Any) => {
                    let out = outputs[0].0;
                    let local = self.check.logic.local(Choice::Query(out));
                    let answer = if local == Bdd::FALSE {
                        Bdd::FALSE
                    } else {
                        let bit = self.bits_of(ev, inputs[0]);
                        self.answer(side, out, bit, cond)
                    };
                    let g = if local == Bdd::TRUE {
                        answer
                    } else {
                        let wave = self.fresh(side, out, 0);
                        self.logic().m.ite(local, answer, wave)
                    };
                    ev.descs.insert(
                        out,
                        Desc {
                            same: None,
                            bits: Some(g),
                        },
                    );
                }
                EffectOp::Wave(WaveOp::Ballot) => {
                    let g = self.bits_of(ev, inputs[0]);
                    ev.descs.insert(
                        outputs[0].0,
                        Desc {
                            same: None,
                            bits: Some(g),
                        },
                    );
                }
                EffectOp::Memory {
                    space,
                    op: MemoryOp::Load(size),
                    ..
                } => {
                    let (out, ty) = outputs[0];
                    let pred = self.bits_of(ev, inputs[1]);
                    let same = if !ev.stored && self.decide(pred, cond, side) == Some(true) {
                        let a = self.operand(ev, inputs[0]);
                        let t = self.intern(ty, Form::Load(*space, *size, a));
                        Some(self.loaded(ty, out, t))
                    } else {
                        None
                    };
                    let d = self.formed(side, out, same);
                    ev.descs.insert(out, d);
                }
                _ => {
                    for &(v, _) in outputs {
                        let d = self.formed(side, v, None);
                        ev.descs.insert(v, d);
                    }
                }
            },
            Inst::Target {
                op, args, outputs, ..
            } => {
                let terms: Vec<usize> =
                    args.values().iter().map(|&a| self.operand(ev, a)).collect();
                for (i, &(v, ty)) in outputs.iter().enumerate() {
                    let t = self.intern(ty, Form::Target(*op, terms.clone(), i));
                    let t = self.loaded(ty, v, t);
                    let d = self.formed(side, v, Some(t));
                    ev.descs.insert(v, d);
                }
            }
            Inst::Packet { .. } => unreachable!("a packet query in a wave program"),
        }
    }

    fn check_effects(&mut self, ev: &mut Evaluation) {
        let (f, facts) = (self.check.f, self.check.facts);
        for (index, inst) in f.blocks[&ev.block].insts.iter().enumerate() {
            if let Some(&m) = self.check.meetings.get(&(ev.block, index)) {
                let local = self.check.logic.local(Choice::Meet(m));
                let kept = self.check.not(local);
                let differs = self.check.and(ev.cond, kept);
                self.check.require(
                    ev.block,
                    index,
                    "a retained meeting may have different participating lanes",
                    differs,
                );
                if self.check.stopped {
                    return;
                }
            }
            let Inst::Effect {
                op,
                inputs,
                outputs,
                ..
            } = inst
            else {
                continue;
            };
            let collective = match op {
                EffectOp::Wave(WaveOp::Any) => {
                    let local = self.check.logic.local(Choice::Query(outputs[0].0));
                    self.check.not(local)
                }
                EffectOp::Wave(WaveOp::Ballot) => {
                    self.check.logic.materialized(facts, outputs[0].0)
                }
                EffectOp::Wave(WaveOp::ReadFirstLane) => {
                    Manager::constant(!facts.uniform[inputs[0].0])
                }
                EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => {
                    Bdd::TRUE
                }
                EffectOp::Memory { .. } => Bdd::FALSE,
            };
            if collective != Bdd::FALSE {
                let differs = self.check.and(ev.cond, collective);
                self.check.require(
                    ev.block,
                    index,
                    "a retained collective may have different participating lanes",
                    differs,
                );
                if self.check.stopped {
                    return;
                }
            }
            if let EffectOp::Memory {
                op:
                    memory @ (MemoryOp::Store(_)
                    | MemoryOp::AtomicAdd(_)
                    | MemoryOp::AtomicRmw(_)
                    | MemoryOp::AtomicCmpSwap),
                ..
            } = op
            {
                let pred = self.bits_of(ev, inputs[memory.mask_input()]);
                if self.decide(pred, ev.cond, ev.side) != Some(false) {
                    let at = (ev.block, index);
                    let space = match op {
                        EffectOp::Memory { space, .. } => *space,
                        _ => unreachable!(),
                    };
                    let address = self.get(ev, inputs[0]).same;
                    let data = match memory {
                        MemoryOp::AtomicCmpSwap => match (self.get(ev, inputs[1]).same, self.get(ev, inputs[2]).same) {
                            (Some(d), Some(c)) => Some(self.intern(Ty::I64, Form::Core(Ty::I64, Op::Pack64(ValueId(d), ValueId(c))))),
                            _ => None,
                        },
                        _ => self.get(ev, inputs[1]).same,
                    };
                    ev.writes.push(Write {
                        at,
                        space,
                        op: *memory,
                        address,
                        data,
                        mask: pred,
                        partnered: self.check.partnered(at),
                    });
                    ev.stored = true;
                }
            }
        }
    }

    fn answer(&mut self, side: usize, v: ValueId, bit: Bdd, cond: Bdd) -> Bdd {
        if side == LANE {
            return bit;
        }
        if self.decide(bit, cond, side) == Some(true) {
            return Bdd::TRUE;
        }
        let others = self.fresh(side, v, 0);
        self.logic().m.or(bit, others)
    }

    fn unmatched(&mut self, cond: Bdd, writes: &[Write], side: usize) {
        let reason = if side == WAVE {
            "the wave program may store while the programs are apart"
        } else {
            "the lane program may store while the programs are apart"
        };
        for w in writes {
            self.check.require(w.at.0, w.at.1, reason, cond);
            if self.check.stopped {
                return;
            }
        }
    }

    fn match_writes(&mut self, cond: Bdd, writes: &[Vec<Write>; 2]) {
        let (wave, lane) = (&writes[WAVE], &writes[LANE]);
        let same = |w: &Write, l: &Write| {
            w.space == l.space
                && w.op == l.op
                && w.address.is_some()
                && w.address == l.address
                && w.data.is_some()
                && w.data == l.data
                && w.mask == l.mask
        };
        let prefix = wave.iter().zip(lane).take_while(|(w, l)| same(w, l)).count();
        let mut pairs: Vec<(usize, usize)> = (0..prefix).map(|i| (i, i)).collect();
        let (rest_wave, rest_lane) = (&wave[prefix..], &lane[prefix..]);
        if rest_wave.len() == rest_lane.len() && self.pairwise_apart(rest_wave) && self.pairwise_apart(rest_lane) {
            let mut taken = vec![false; rest_lane.len()];
            for (i, w) in rest_wave.iter().enumerate() {
                if let Some(j) = (0..rest_lane.len()).find(|&j| !taken[j] && same(w, &rest_lane[j])) {
                    taken[j] = true;
                    pairs.push((prefix + i, prefix + j));
                }
            }
        }
        pairs.retain(|&(i, j)| {
            let (w, l) = (&wave[i], &lane[j]);
            (!w.partnered || self.partners_outside(w.at)) && (!l.partnered || self.partners_outside(l.at))
        });
        for &(i, _) in &pairs {
            let w = wave[i];
            let mut differs = self.check.or(self.leaves[w.address.unwrap()], self.leaves[w.data.unwrap()]);
            let atoms: Vec<u32> = self.logic().support(w.mask).iter().copied().collect();
            for var in atoms {
                if let Atom::Bit(v) | Atom::View(v) = self.check.logic.atom_of(var) {
                    differs = self.check.or(differs, self.check.h[v.0]);
                } else if !matches!(self.check.logic.atom_of(var), Atom::Lane(_) | Atom::Marker(_)) {
                    differs = Bdd::TRUE;
                }
            }
            let differs = self.check.and(cond, differs);
            self.check.require(w.at.0, w.at.1, "the programs may store different words while apart", differs);
            if self.check.stopped {
                return;
            }
        }
        let left_wave: Vec<Write> = (0..wave.len()).filter(|&i| !pairs.iter().any(|&(x, _)| x == i)).map(|i| wave[i]).collect();
        let left_lane: Vec<Write> = (0..lane.len()).filter(|&j| !pairs.iter().any(|&(_, y)| y == j)).map(|j| lane[j]).collect();
        self.unmatched(cond, &left_wave, WAVE);
        if self.check.stopped {
            return;
        }
        self.unmatched(cond, &left_lane, LANE);
    }

    fn pairwise_apart(&self, writes: &[Write]) -> bool {
        writes.iter().enumerate().all(|(i, a)| writes[i + 1..].iter().all(|b| self.apart(a, b) || self.alike(a, b)))
    }

    fn alike(&self, a: &Write, b: &Write) -> bool {
        let MemoryOp::Store(size) = a.op else {
            return false;
        };
        if a.op != b.op || a.space != b.space || a.data.is_none() || a.data != b.data {
            return false;
        }
        let (Some(x), Some(y)) = (a.address, b.address) else {
            return false;
        };
        let bytes = size.bytes() as u64;
        let (terms, constant) = self.difference(x, y);
        let modulus: u128 = 1 << self.types[x].bits();
        let aligned = |c: u64| (c as u128 % modulus) % bytes as u128 == 0;
        modulus % bytes as u128 == 0 && aligned(constant) && terms.values().all(|&c| aligned(c))
    }

    fn apart(&self, a: &Write, b: &Write) -> bool {
        if a.space != b.space {
            return true;
        }
        let (Some(x), Some(y)) = (a.address, b.address) else {
            return false;
        };
        let (terms, constant) = self.difference(x, y);
        if !terms.is_empty() {
            return false;
        }
        let bits = self.types[x].bits();
        let modulus: u128 = 1 << bits;
        let d = constant as u128 % modulus;
        let bytes = |w: &Write| match w.op {
            MemoryOp::Store(size) => size.bytes() as u128,
            _ => 4,
        };
        d >= bytes(b) && modulus - d >= bytes(a)
    }

    fn difference(&self, x: usize, y: usize) -> (BTreeMap<Leaf, u64>, u64) {
        let mask = if self.types[x].bits() >= 64 { u64::MAX } else { (1u64 << self.types[x].bits()) - 1 };
        let (mut terms, mut constant) = (BTreeMap::new(), 0u64);
        self.expand(x, 1, &mut terms, &mut constant, 0);
        self.expand(y, u64::MAX, &mut terms, &mut constant, 0);
        terms.retain(|_, c| *c & mask != 0);
        (terms.into_iter().map(|(l, c)| (l, c & mask)).collect(), constant & mask)
    }

    fn expand(&self, t: usize, scale: u64, terms: &mut BTreeMap<Leaf, u64>, constant: &mut u64, depth: usize) {
        match &self.terms[t] {
            Form::Linear(_, parts, k) => {
                *constant = constant.wrapping_add(k.wrapping_mul(scale));
                for &(u, c) in parts {
                    self.expand(u, scale.wrapping_mul(c), terms, constant, depth);
                }
            }
            Form::Core(_, Op::Const(_, k)) => *constant = constant.wrapping_add(k.wrapping_mul(scale)),
            Form::Value(v) => self.expand_value(*v, scale, terms, constant, depth),
            _ => {
                let c = terms.entry(Leaf::Term(t)).or_insert(0);
                *c = c.wrapping_add(scale);
            }
        }
    }

    fn expand_value(&self, v: ValueId, scale: u64, terms: &mut BTreeMap<Leaf, u64>, constant: &mut u64, depth: usize) {
        let (f, facts) = (self.check.f, self.check.facts);
        if let Some(k) = facts.constant(f, v) {
            *constant = constant.wrapping_add(k.wrapping_mul(scale));
            return;
        }
        let wide = |x: ValueId| f.types[x.0] == f.types[v.0];
        let op = if depth > 16 || !matches!(f.types[v.0], Ty::I32 | Ty::I64) { None } else { facts.op(f, v) };
        match op {
            Some(Op::Int(IntOp::Add, a, b)) if wide(a) && wide(b) => {
                self.expand_value(a, scale, terms, constant, depth + 1);
                self.expand_value(b, scale, terms, constant, depth + 1);
            }
            Some(Op::Int(IntOp::Sub, a, b)) if wide(a) && wide(b) => {
                self.expand_value(a, scale, terms, constant, depth + 1);
                self.expand_value(b, scale.wrapping_neg(), terms, constant, depth + 1);
            }
            Some(Op::Int(IntOp::Mul, a, k)) if wide(a) && facts.constant(f, k).is_some() => {
                self.expand_value(a, scale.wrapping_mul(facts.constant(f, k).unwrap()), terms, constant, depth + 1);
            }
            Some(Op::Int(IntOp::Mul, k, a)) if wide(a) && facts.constant(f, k).is_some() => {
                self.expand_value(a, scale.wrapping_mul(facts.constant(f, k).unwrap()), terms, constant, depth + 1);
            }
            Some(Op::Int(IntOp::Shl, a, k)) if wide(a) && facts.constant(f, k).is_some_and(|k| k < f.types[v.0].bits() as u64) => {
                self.expand_value(a, scale.wrapping_mul(1u64 << facts.constant(f, k).unwrap()), terms, constant, depth + 1);
            }
            _ => {
                let c = terms.entry(Leaf::Value(v)).or_insert(0);
                *c = c.wrapping_add(scale);
            }
        }
    }

    fn partners_outside(&self, at: (BlockId, usize)) -> bool {
        let hazards = self.check.hazards;
        let Some(i) = hazards.accesses.iter().position(|a| (a.block, a.index) == at) else {
            return true;
        };
        hazards
            .together
            .iter()
            .chain(&hazards.apart)
            .filter_map(|&(p, q)| if p == i { Some(q) } else if q == i { Some(p) } else { None })
            .all(|j| !self.visited.contains(&hazards.accesses[j].block))
    }

    fn meet(
        &mut self,
        block: BlockId,
        cond: Bdd,
        sides: &[Vec<Desc>; 2],
        arrivals: &mut BTreeMap<BlockId, Vec<Bdd>>,
    ) {
        let (f, facts) = (self.check.f, self.check.facts);
        let dst = &f.blocks[&block];
        let mut relation = Bdd::TRUE;
        for (k, &(param, ty)) in dst.params.iter().enumerate() {
            if !self.check.logic.carried(param) {
                continue;
            }
            let atom = match ty {
                Ty::I1 => Atom::Bit(param),
                Ty::I32 if facts.viewed[param.0] => Atom::View(param),
                _ => continue,
            };
            if let Some(g) = sides[LANE][k].bits {
                let a = self.logic().atom(atom);
                let link = self.logic().m.iff(a, g);
                relation = self.check.and(relation, link);
            }
        }
        for (k, &(param, ty)) in dst.params.iter().enumerate() {
            if !self.check.logic.carried(param) {
                continue;
            }
            let (w, l) = (sides[WAVE][k], sides[LANE][k]);
            let boolean =
                ty == Ty::I1 || (facts.lane_word[param.0] && !facts.materialized[param.0]);
            let same = w.same.is_some() && w.same == l.same;
            let mut differs = if same {
                let leaves = self.leaves[w.same.unwrap()];
                self.check.and(cond, leaves)
            } else if let (true, Some(gw), Some(gl)) = (boolean, w.bits, l.bits) {
                let equal = self.logic().m.iff(gw, gl);
                if self.decide(equal, cond, WAVE) == Some(true) {
                    let mut atoms: BTreeSet<u32> =
                        self.logic().support(gw).iter().copied().collect();
                    atoms.extend(self.logic().support(gl).iter().copied());
                    let mut hs = Bdd::FALSE;
                    for var in atoms {
                        if let Atom::Bit(v) | Atom::View(v) = self.check.logic.atom_of(var) {
                            hs = self.check.or(hs, self.check.h[v.0]);
                        }
                    }
                    self.check.and(cond, hs)
                } else {
                    cond
                }
            } else {
                cond
            };
            if ty == Ty::I32 && facts.lane_word[param.0] && !same {
                let mode = self.check.logic.materialized(facts, param);
                let full = self.check.and(cond, mode);
                differs = self.check.or(differs, full);
            }
            if differs == Bdd::FALSE {
                continue;
            }
            let joint = self.check.and(differs, relation);
            let logic = &mut self.check.logic;
            let foreign: Vec<u32> = logic
                .support(joint)
                .iter()
                .copied()
                .filter(|&v| {
                    !matches!(logic.atom_of(v), Atom::Marker(_))
                        && logic.scope(facts, v) != Some(block)
                })
                .collect();
            let restated = logic.exists(&foreign, joint);
            if restated == Bdd::FALSE {
                continue;
            }
            let entry = arrivals
                .entry(block)
                .or_insert_with(|| vec![Bdd::FALSE; dst.params.len()]);
            entry[k] = self.check.logic.m.or(entry[k], restated);
        }
    }
}

fn settled(f: &Func, facts: &Facts, op: Op) -> bool {
    let (Op::Cmp(_, x, y) | Op::Int(_, x, y)) = op else {
        return false;
    };
    if x == y && matches!(op, Op::Int(IntOp::Sub | IntOp::Xor, ..) | Op::Cmp(..)) {
        return true;
    }
    if f.types[x.0] != Ty::I32 {
        return false;
    }
    if let Op::Cmp(p, ..) = op {
        if let (Some(a), Some(b)) = (interval(f, facts, x, 0), interval(f, facts, y, 0)) {
            if decided(p, a, b).is_some() {
                return true;
            }
        }
    }
    let (Some(xs), Some(ys)) = (constant_choices(f, facts, x), constant_choices(f, facts, y)) else {
        return false;
    };
    let mut answers = Vec::new();
    for &a in &xs {
        for &b in &ys {
            let (a, b) = (a as u32, b as u32);
            answers.push(match op {
                Op::Cmp(p, ..) => compare(p, a, b) as u32,
                Op::Int(k, ..) => match k {
                    IntOp::Add => a.wrapping_add(b),
                    IntOp::Sub => a.wrapping_sub(b),
                    IntOp::Mul => a.wrapping_mul(b),
                    IntOp::And => a & b,
                    IntOp::Or => a | b,
                    IntOp::Xor => a ^ b,
                    IntOp::Shl => a << (b & 31),
                    IntOp::LShr => a >> (b & 31),
                    IntOp::AShr => ((a as i32) >> (b & 31)) as u32,
                },
                _ => return false,
            });
        }
    }
    answers.iter().all(|&t| t == answers[0])
}

pub(super) fn interval(f: &Func, facts: &Facts, v: ValueId, depth: usize) -> Option<(u32, u32)> {
    if depth > 16 {
        return None;
    }
    if let Some(k) = facts.constant(f, v) {
        return Some((k as u32, k as u32));
    }
    let full = (0, u32::MAX);
    let range = |x: ValueId| interval(f, facts, x, depth + 1);
    match facts.inst(f, v)? {
        Inst::Core { op, .. } => match *op {
            Op::Env(Env::LaneId) => Some((0, 31)),
            Op::Int(IntOp::And, a, b) => {
                let (x, y) = (range(a).unwrap_or(full), range(b).unwrap_or(full));
                Some((0, x.1.min(y.1)))
            }
            Op::Int(IntOp::Or, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                let top = x.1.max(y.1);
                let ones = if top == 0 { 0 } else { u32::MAX >> top.leading_zeros() };
                Some((x.0.max(y.0), ones))
            }
            Op::Int(IntOp::Add, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                let high = x.1 as u64 + y.1 as u64;
                (high <= u32::MAX as u64).then(|| (x.0 + y.0, high as u32))
            }
            Op::Int(IntOp::Sub, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                (x.0 >= y.1).then(|| (x.0 - y.1, x.1 - y.0))
            }
            Op::Int(IntOp::Mul, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                let high = x.1 as u64 * y.1 as u64;
                (high <= u32::MAX as u64).then(|| (x.0 * y.0, high as u32))
            }
            Op::Int(IntOp::Shl, a, s) => {
                let k = facts.constant(f, s)? as u32 & 31;
                let x = range(a)?;
                let high = (x.1 as u64) << k;
                (high <= u32::MAX as u64).then(|| (x.0 << k, high as u32))
            }
            Op::Int(IntOp::Xor, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                let top = x.1.max(y.1);
                Some((0, if top == 0 { 0 } else { u32::MAX >> top.leading_zeros() }))
            }
            Op::Int(IntOp::LShr, a, s) => {
                let k = facts.constant(f, s)? as u32 & 31;
                let x = range(a).unwrap_or(full);
                Some((x.0 >> k, x.1 >> k))
            }
            Op::Int(IntOp::AShr, a, s) => {
                let k = facts.constant(f, s)? as u32 & 31;
                let x = range(a)?;
                (x.1 < 1 << 31).then(|| (x.0 >> k, x.1 >> k))
            }
            Op::Select(_, a, b) => {
                let (x, y) = (range(a)?, range(b)?);
                Some((x.0.min(y.0), x.1.max(y.1)))
            }
            Op::Convert(Cvt::ZExt, Ty::I32, a) if f.types[a.0] == Ty::I1 => Some((0, 1)),
            Op::PopulationCount(_) | Op::LeadingZeros(_) | Op::TrailingZeros(_) => Some((0, 32)),
            _ => None,
        },
        Inst::Effect {
            op: EffectOp::Memory { op: MemoryOp::Load(size), .. },
            ..
        } => match size {
            MemSize::U8 => Some((0, 0xff)),
            MemSize::U16 => Some((0, 0xffff)),
            _ => None,
        },
        _ => None,
    }
}

fn decided(p: IntPred, x: (u32, u32), y: (u32, u32)) -> Option<bool> {
    let signed = matches!(p, IntPred::Slt | IntPred::Sgt | IntPred::Sle | IntPred::Sge);
    if signed && (x.1 >= 1 << 31 || y.1 >= 1 << 31) {
        return None;
    }
    let below = |a: (u32, u32), b: (u32, u32)| {
        if a.1 < b.0 {
            Some(true)
        } else if a.0 >= b.1 {
            Some(false)
        } else {
            None
        }
    };
    match p {
        IntPred::Eq | IntPred::Ne => {
            let equal = if x.1 < y.0 || y.1 < x.0 {
                Some(false)
            } else if x.0 == x.1 && y.0 == y.1 && x.0 == y.0 {
                Some(true)
            } else {
                None
            };
            equal.map(|e| e == (p == IntPred::Eq))
        }
        IntPred::Ult | IntPred::Slt => below(x, y),
        IntPred::Ugt | IntPred::Sgt => below(y, x),
        IntPred::Ule | IntPred::Sle => below(y, x).map(|b| !b),
        IntPred::Uge | IntPred::Sge => below(x, y).map(|b| !b),
    }
}

#[cfg(test)]
mod tests {
    use super::super::logic::Kept;
    use super::super::testing::*;
    use super::super::{direct, search};
    use super::*;

    fn no_hazards() -> Hazards {
        Hazards {
            accesses: Vec::new(),
            together: BTreeSet::new(),
            apart: BTreeSet::new(),
            idle: BTreeSet::new(),
            meetings: Vec::new(),
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Reader {
        ReadLane,
        ReadFirstLane,
        WriteLane,
        Bpermute,
        BpermuteFi,
        Wmma,
    }

    fn exchange(b: &mut Build, e: BlockId, reader: Reader, x: ValueId, exec: ValueId) -> ValueId {
        let zero = b.constant(e, Ty::I32, 0);
        match reader {
            Reader::ReadLane => b.wave(e, WaveOp::ReadLane, vec![x, zero, zero]),
            Reader::ReadFirstLane => b.wave(e, WaveOp::ReadFirstLane, vec![x, exec]),
            Reader::WriteLane => {
                let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
                let one = b.constant(e, Ty::I32, 1);
                b.wave(e, WaveOp::WriteLane, vec![x, one, lane, zero])
            }
            Reader::Bpermute => b.wave(e, WaveOp::Bpermute, vec![zero, x, exec]),
            Reader::BpermuteFi => b.wave(e, WaveOp::BpermuteFi, vec![zero, x, exec]),
            Reader::Wmma => {
                let fzero = b.constant(e, Ty::F32, 0);
                let mut inputs = vec![x; 8];
                inputs.extend([fzero; 8]);
                let outputs = b.effect(e, EffectOp::Wave(WaveOp::Wmma), inputs);
                b.core(e, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, outputs[0]))
            }
        }
    }

    fn reads_another_lane(reader: Reader) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let flags = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
        let other = match reader {
            Reader::WriteLane => zero,
            _ => lane,
        };
        let x = b.core(e, Ty::I32, Op::Select(q, ones, other));
        let y = exchange(&mut b, e, reader, x, k.exec);
        let address = byte_offset(&mut b, e, buf, lane, 4);
        b.store(e, Space::Global, MemSize::B32, address, y, c);
        (b, q)
    }

    fn reads_outside_exec(reader: Reader) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
        let idle = b.core(e, Ty::I32, Op::Select(q, ones, lane));
        let x = b.core(e, Ty::I32, Op::Select(k.exec, lane, idle));
        let y = exchange(&mut b, e, reader, x, k.exec);
        store_own(&mut b, &k, e, y, k.exec);
        (b, q)
    }

    struct Flagged {
        b: Build,
        k: Kernel,
        buf: ValueId,
        lane: ValueId,
        c: ValueId,
    }

    fn flagged() -> Flagged {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let flags = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        Flagged {
            b,
            k,
            buf,
            lane,
            c,
        }
    }

    fn both(b: &Build) -> [Kept; 2] {
        [
            search::prove(&b.f, &b.inputs, Some(0), &no_hazards()).0,
            direct::prove(&b.f, &b.inputs, Some(0), &no_hazards()).0,
        ]
    }

    #[test]
    fn prove_keeps_a_query_that_decides_whether_lanes_store() {
        let Flagged { mut b, k, buf, lane, c, .. } = flagged();
        let e = BlockId(0);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (join, _) = b.block(&[Ty::I1]);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        b.cond_br(e, q, (then, vec![k.exec, own]), (join, vec![k.exec]));
        let one = b.constant(then, Ty::I32, 1);
        b.store(then, Space::Global, MemSize::B32, t[1], one, t[0]);
        b.br(then, join, vec![t[0]]);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag skips the store the wave makes", wrong);
    }

    #[test]
    fn prove_keeps_a_query_that_moves_the_address() {
        let Flagged { mut b, k, buf, lane, c, .. } = flagged();
        let e = BlockId(0);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let near = b.constant(e, Ty::I64, 0);
        let far = b.constant(e, Ty::I64, 4);
        let shift = b.core(e, Ty::I64, Op::Select(q, near, far));
        let own = byte_offset(&mut b, e, buf, lane, 8);
        let address = b.int(e, IntOp::Add, own, shift);
        let one = b.constant(e, Ty::I32, 1);
        b.store(e, Space::Global, MemSize::B32, address, one, k.exec);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag stores four bytes further", wrong);
    }

    #[test]
    fn prove_keeps_a_query_that_picks_the_word_a_lane_loads() {
        let Flagged { mut b, k, buf, lane, c, .. } = flagged();
        let e = BlockId(0);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let table = k.buffer(&mut b, e, 16);
        let four = b.constant(e, Ty::I64, 4);
        let second = b.int(e, IntOp::Add, table, four);
        let from = b.core(e, Ty::I64, Op::Select(q, table, second));
        let v = b.load(e, Space::Global, MemSize::B32, from, k.exec);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, v, k.exec);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag loads the second word of the table", wrong);
    }

    #[test]
    fn prove_keeps_a_query_that_masks_a_store() {
        let Flagged { mut b, k, buf, lane, c, .. } = flagged();
        let e = BlockId(0);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let mask = b.int(e, IntOp::And, q, k.exec);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let one = b.constant(e, Ty::I32, 1);
        b.store(e, Space::Global, MemSize::B32, own, one, mask);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag skips its store", wrong);
    }

    #[test]
    fn prove_keeps_a_ballot_whose_lane_test_masks_a_store() {
        let Flagged { mut b, k, buf, lane, c, .. } = flagged();
        let e = BlockId(0);
        let w = b.wave(e, WaveOp::Ballot, vec![c]);
        let zero = b.constant(e, Ty::I32, 0);
        let any = b.cmp(e, IntPred::Ne, w, zero);
        let mask = b.int(e, IntOp::And, any, k.exec);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let one = b.constant(e, Ty::I32, 1);
        b.store(e, Space::Global, MemSize::B32, own, one, mask);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.words.contains(&w))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag reads its own bit of the ballot as zero", wrong);
    }

    #[test]
    fn prove_keeps_a_query_that_ends_a_loop() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let flags = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, zero, buf, flags]);
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        let wave = b.constant(body, Ty::I32, 32);
        let row = b.int(body, IntOp::Mul, p[1], wave);
        let item = b.int(body, IntOp::Add, row, lane);
        let own = byte_offset(&mut b, body, p[3], item, 4);
        let flag = b.load(body, Space::Global, MemSize::B32, own, p[0]);
        let z = b.constant(body, Ty::I32, 0);
        let set = b.cmp(body, IntPred::Ne, flag, z);
        let c = b.int(body, IntOp::And, set, p[0]);
        let out = byte_offset(&mut b, body, p[2], item, 4);
        let one = b.constant(body, Ty::I32, 1);
        b.store(body, Space::Global, MemSize::B32, out, one, p[0]);
        let next = b.int(body, IntOp::Add, p[1], one);
        let q = b.wave(body, WaveOp::Any, vec![c]);
        b.cond_br(body, q, (body, vec![p[0], next, p[2], p[3]]), (exit, vec![p[0]]));
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag leaves the loop while the wave stores another row", wrong);
    }

    fn carried_query(latch: bool) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let flags = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let yes = b.constant(e, Ty::I1, 1);
        let shape = [Ty::I1, Ty::I1, Ty::I32, Ty::I64, Ty::I64];
        let (body, p) = b.block(&shape);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, yes, zero, buf, flags]);
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        let wave = b.constant(body, Ty::I32, 32);
        let row = b.int(body, IntOp::Mul, p[2], wave);
        let item = b.int(body, IntOp::Add, row, lane);
        let own = byte_offset(&mut b, body, p[4], item, 4);
        let always = b.constant(body, Ty::I1, 1);
        let flag = b.load(body, Space::Global, MemSize::B32, own, always);
        let z = b.constant(body, Ty::I32, 0);
        let c = b.cmp(body, IntPred::Ne, flag, z);
        let q = b.wave(body, WaveOp::Any, vec![c]);
        let mask = b.int(body, IntOp::And, p[1], p[0]);
        let out = byte_offset(&mut b, body, p[3], item, 4);
        let one = b.constant(body, Ty::I32, 1);
        b.store(body, Space::Global, MemSize::B32, out, one, mask);
        let next = b.int(body, IntOp::Add, p[2], one);
        let four = b.constant(body, Ty::I32, 4);
        let below = b.cmp(body, IntPred::Ult, next, four);
        let again = b.int(body, IntOp::And, p[1], below);
        let back = vec![p[0], q, next, p[3], p[4]];
        if latch {
            let (latch, l) = b.block(&shape);
            b.cond_br(body, again, (latch, back), (exit, vec![p[0]]));
            b.br(latch, body, l);
        } else {
            b.cond_br(body, again, (body, back), (exit, vec![p[0]]));
        }
        (b, q)
    }

    #[test]
    fn prove_keeps_a_query_a_self_loop_carries_into_its_own_condition() {
        let (b, q) = carried_query(false);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag in row 0 leaves the loop and skips row 1, which the wave stores", wrong);
    }

    #[test]
    fn prove_keeps_a_query_a_loop_with_a_latch_carries_into_its_own_condition() {
        let (b, q) = carried_query(true);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag in row 0 leaves the loop and skips row 1, which the wave stores", wrong);
    }

    fn detour_home(join: bool) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let flags = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let yes = b.constant(e, Ty::I1, 1);
        let head_shape = [Ty::I1, Ty::I1, Ty::I32, Ty::I64, Ty::I64];
        let arm_shape = [Ty::I1, Ty::I32, Ty::I64, Ty::I64];
        let (head, h) = b.block(&head_shape);
        let (taken, t) = b.block(&arm_shape);
        let (other, o) = b.block(&arm_shape);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, head, vec![k.exec, yes, zero, buf, flags]);
        let lane = b.core(head, Ty::I32, Op::Env(Env::LaneId));
        let wave = b.constant(head, Ty::I32, 32);
        let row = b.int(head, IntOp::Mul, h[2], wave);
        let item = b.int(head, IntOp::Add, row, lane);
        let out = byte_offset(&mut b, head, h[3], item, 4);
        let mask = b.int(head, IntOp::And, h[1], h[0]);
        let one = b.constant(head, Ty::I32, 1);
        b.store(head, Space::Global, MemSize::B32, out, one, mask);
        let own = byte_offset(&mut b, head, h[4], item, 4);
        let always = b.constant(head, Ty::I1, 1);
        let flag = b.load(head, Space::Global, MemSize::B32, own, always);
        let z = b.constant(head, Ty::I32, 0);
        let c = b.cmp(head, IntPred::Ne, flag, z);
        let q = b.wave(head, WaveOp::Any, vec![c]);
        let go = b.int(head, IntOp::And, q, h[1]);
        let next = b.int(head, IntOp::Add, h[2], one);
        b.cond_br(head, go, (taken, vec![h[0], next, h[3], h[4]]), (other, vec![h[0], next, h[3], h[4]]));
        let on = b.constant(taken, Ty::I1, 1);
        let off = b.constant(other, Ty::I1, 0);
        if join {
            let (meet, m) = b.block(&head_shape);
            b.br(taken, meet, vec![t[0], on, t[1], t[2], t[3]]);
            let four = b.constant(other, Ty::I32, 4);
            let below = b.cmp(other, IntPred::Ult, o[1], four);
            b.cond_br(other, below, (meet, vec![o[0], off, o[1], o[2], o[3]]), (exit, vec![o[0]]));
            b.br(meet, head, m);
        } else {
            b.br(taken, head, vec![t[0], on, t[1], t[2], t[3]]);
            let four = b.constant(other, Ty::I32, 4);
            let below = b.cmp(other, IntPred::Ult, o[1], four);
            b.cond_br(other, below, (head, vec![o[0], off, o[1], o[2], o[3]]), (exit, vec![o[0]]));
        }
        (b, q)
    }

    #[test]
    fn prove_keeps_a_query_whose_detour_ends_where_it_began() {
        let (b, q) = detour_home(false);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag takes the other arm, clears its bit and skips row 1, which the wave stores", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_detour_ends_at_a_join_before_the_header() {
        let (b, q) = detour_home(true);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag takes the other arm, clears its bit and skips row 1, which the wave stores", wrong);
    }

    #[test]
    fn prove_demands_every_lane_for_a_kept_query_over_unmasked_bits() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let flags = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, flags, lane, 4);
        let yes = b.constant(e, Ty::I1, 1);
        let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let at = b.here(e);
        let q = b.wave(e, WaveOp::Any, vec![set]);
        let one = b.constant(e, Ty::I32, 1);
        let data = b.core(e, Ty::I32, Op::Select(q, one, zero));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, data, k.exec);
        let Inst::Effect { provenance, .. } = b.f.blocks[&e].insts[at.1] else { unreachable!() };
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
            assert!(kept.queries.contains(&q), "{}: a lane without the flag stores zero", name);
            assert!(everyone.contains(&provenance), "{}: the query reads the flags of lanes whose exec is clear", name);
        }
    }

    type Position = (BlockId, usize);

    fn meetings(program: &crate::rdna_spmd::program::Program, hazards: &Hazards) -> Vec<(&'static str, BTreeSet<Position>)> {
        let (f, inputs) = (&program.ir, &program.parameter_inputs);
        let search = search::prove(f, inputs, Some(0), hazards).0;
        let direct = direct::prove(f, inputs, Some(0), hazards).0;
        vec![("search", search), ("direct", direct)]
            .into_iter()
            .map(|(name, kept)| (name, kept.meets.iter().map(|&m| hazards.meetings[m]).collect()))
            .collect()
    }

    fn keeps_exactly(program: &crate::rdna_spmd::program::Program, hazards: &Hazards, expected: &[Position], why: &str) {
        let expected: BTreeSet<Position> = expected.iter().copied().collect();
        let wrong: Vec<(&str, BTreeSet<Position>)> =
            meetings(program, hazards).into_iter().filter(|(_, kept)| *kept != expected).collect();
        assert!(wrong.is_empty(), "{}: expected meetings before {:?}, kept {:?}", why, expected, wrong);
    }

    fn store(b: &mut Build, block: BlockId, address: ValueId, data: u64, mask: ValueId) -> Position {
        let value = b.constant(block, Ty::I32, data);
        let at = b.here(block);
        b.store(block, Space::Global, MemSize::B32, address, value, mask);
        at
    }

    #[test]
    fn prove_orders_two_stores_to_one_word_with_one_meeting() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let s1 = store(&mut b, e, buf, 1, k.exec);
        let s2 = store(&mut b, e, buf, 2, k.exec);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        keeps_exactly(&program, &hazards, &[s2], "every lane's second store must follow every lane's first");
    }

    #[test]
    fn prove_orders_three_stores_to_one_word_with_the_two_meetings_between_them() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let s1 = store(&mut b, e, buf, 1, k.exec);
        let s2 = store(&mut b, e, buf, 2, k.exec);
        let s3 = store(&mut b, e, buf, 3, k.exec);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(s1, s2), (s2, s3), (s1, s3)], &[], &[]);
        keeps_exactly(&program, &hazards, &[s2, s3], "each store to the word must follow the one before it");
    }

    #[test]
    fn prove_orders_stores_of_disjoint_lane_halves() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let sixteen = b.constant(e, Ty::I32, 16);
        let low = b.cmp(e, IntPred::Ult, lane, sixteen);
        let high = b.cmp(e, IntPred::Uge, lane, sixteen);
        let low = b.int(e, IntOp::And, low, k.exec);
        let high = b.int(e, IntOp::And, high, k.exec);
        let s1 = store(&mut b, e, buf, 1, low);
        let s2 = store(&mut b, e, buf, 2, high);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        keeps_exactly(&program, &hazards, &[s2], "the upper half's store must follow the lower half's");
    }

    #[test]
    fn prove_orders_a_store_in_a_later_block() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let s1 = store(&mut b, e, buf, 1, k.exec);
        let (next, n) = b.block(&[Ty::I1, Ty::I64]);
        b.br(e, next, vec![k.exec, buf]);
        let s2 = store(&mut b, next, n[1], 2, n[0]);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        keeps_exactly(&program, &hazards, &[s2], "the store in the next block must follow the first");
    }

    fn read_after(write: impl Fn(&mut Build, BlockId, ValueId) -> ValueId) -> (crate::rdna_spmd::program::Program, Position, Position) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let out = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let data = write(&mut b, e, lane);
        let s = b.here(e);
        b.store(e, Space::Global, MemSize::B32, buf, data, k.exec);
        let l = b.here(e);
        let v = b.load(e, Space::Global, MemSize::B32, buf, k.exec);
        let own = byte_offset(&mut b, e, out, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, v, k.exec);
        (b.program(), s, l)
    }

    #[test]
    fn prove_orders_a_load_after_the_stores_whose_word_it_reads() {
        let (program, s, l) = read_after(|_, _, lane| lane);
        let hazards = Hazards::given(&program, &[(s, l)], &[], &[]);
        keeps_exactly(&program, &hazards, &[l], "every lane must read the one word the wave's store leaves, not its own");
    }

    #[test]
    fn prove_orders_a_load_after_a_constant_store_only_some_lanes_make() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let out = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let sixteen = b.constant(e, Ty::I32, 16);
        let low = b.cmp(e, IntPred::Ult, lane, sixteen);
        let some = b.int(e, IntOp::And, low, k.exec);
        let seven = b.constant(e, Ty::I32, 7);
        let s = b.here(e);
        b.store(e, Space::Global, MemSize::B32, buf, seven, some);
        let l = b.here(e);
        let v = b.load(e, Space::Global, MemSize::B32, buf, k.exec);
        let own = byte_offset(&mut b, e, out, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, v, k.exec);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(s, l)], &[], &[]);
        keeps_exactly(&program, &hazards, &[l], "lanes 16 to 31 read the word without storing it, so they must wait for lanes 0 to 15");
    }

    #[test]
    fn prove_needs_no_meeting_when_every_lane_stores_the_word_it_reads_back() {
        let (program, s, l) = read_after(|b, e, _| b.constant(e, Ty::I32, 7));
        let hazards = Hazards::given(&program, &[(s, l)], &[], &[]);
        keeps_exactly(&program, &hazards, &[], "every lane stores 7 before it reads, and no lane stores anything else there");
    }

    #[test]
    fn prove_orders_a_store_after_the_loads_that_read_the_old_word() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let out = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let l = b.here(e);
        let v = b.load(e, Space::Global, MemSize::B32, buf, k.exec);
        let s = b.here(e);
        b.store(e, Space::Global, MemSize::B32, buf, lane, k.exec);
        let own = byte_offset(&mut b, e, out, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, v, k.exec);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(l, s)], &[], &[]);
        keeps_exactly(&program, &hazards, &[s], "no lane may overwrite the word before every lane has read it");
    }

    #[test]
    fn prove_needs_no_meeting_for_a_load_whose_word_nothing_uses() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let s = b.here(e);
        b.store(e, Space::Global, MemSize::B32, buf, lane, k.exec);
        let l = b.here(e);
        b.load(e, Space::Global, MemSize::B32, buf, k.exec);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(s, l)], &[], &[]);
        keeps_exactly(&program, &hazards, &[], "the loaded word reaches no store");
    }

    fn with_between(between: impl Fn(&mut Build, BlockId, &Kernel)) -> (crate::rdna_spmd::program::Program, Position, Position) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let s1 = store(&mut b, e, buf, 1, k.exec);
        between(&mut b, e, &k);
        let s2 = store(&mut b, e, buf, 2, k.exec);
        (b.program(), s1, s2)
    }

    #[test]
    fn prove_needs_no_meeting_across_a_barrier() {
        let (program, s1, s2) = with_between(|b, e, _| {
            let id = b.constant(e, Ty::I32, 0);
            b.effect(e, EffectOp::BarrierSignal { is_first: false }, vec![id]);
            b.effect(e, EffectOp::BarrierWait, vec![id]);
        });
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        keeps_exactly(&program, &hazards, &[], "the barrier already aligns every lane between the stores");
    }

    #[test]
    fn prove_needs_no_meeting_across_a_lane_read() {
        let (program, s1, s2) = with_between(|b, e, k| {
            let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
            let zero = b.constant(e, Ty::I32, 0);
            let y = b.wave(e, WaveOp::ReadLane, vec![lane, zero, zero]);
            let out = k.buffer(b, e, 8);
            let own = byte_offset(b, e, out, lane, 4);
            b.store(e, Space::Global, MemSize::B32, own, y, k.exec);
        });
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        keeps_exactly(&program, &hazards, &[], "the lane read already aligns every lane between the stores");
    }

    #[test]
    fn search_needs_no_meeting_a_query_kept_later_already_orders() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let s1 = store(&mut b, e, buf, 1, k.exec);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let flags = k.buffer(&mut b, e, 8);
        let own = byte_offset(&mut b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let s2 = store(&mut b, e, buf, 2, k.exec);
        let one = b.constant(e, Ty::I32, 1);
        let data = b.core(e, Ty::I32, Op::Select(q, one, zero));
        let out = k.buffer(&mut b, e, 16);
        let slot = byte_offset(&mut b, e, out, lane, 4);
        b.store(e, Space::Global, MemSize::B32, slot, data, k.exec);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards);
            assert_eq!(kept.queries.len(), 1, "{}: a lane without the flag stores 0 unless the query stays", name);
        }
        keeps_exactly(&program, &hazards, &[], "the kept query between the stores already aligns every lane");
    }

    fn either_answer() -> (Build, ValueId, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flag = per_lane(&mut b, &k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let first = b.wave(e, WaveOp::Any, vec![c]);
        let second = b.wave(e, WaveOp::Any, vec![c]);
        let either = b.int(e, IntOp::Or, first, second);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let data = b.core(e, Ty::I32, Op::Select(either, one, two));
        store_own(&mut b, &k, e, data, k.exec);
        (b, first, second)
    }

    #[test]
    fn prove_keeps_one_of_two_queries_either_of_which_answers_a_store() {
        let (b, first, second) = either_answer();
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&first) && !kept.queries.contains(&second))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag stores 2 unless one of the queries stays", wrong);
    }

    #[test]
    fn prove_keeps_both_of_two_queries_over_different_words_either_of_which_answers_a_store() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let zero = b.constant(e, Ty::I32, 0);
        let flags: Vec<ValueId> = [8, 12]
            .iter()
            .map(|&at| {
                let flag = per_lane(&mut b, &k, e, at);
                let set = b.cmp(e, IntPred::Ne, flag, zero);
                let c = b.int(e, IntOp::And, set, k.exec);
                b.wave(e, WaveOp::Any, vec![c])
            })
            .collect();
        let either = b.int(e, IntOp::Or, flags[0], flags[1]);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let data = b.core(e, Ty::I32, Op::Select(either, one, two));
        store_own(&mut b, &k, e, data, k.exec);
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&flags[0]) || !kept.queries.contains(&flags[1]))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: any(c) | any(d) is 1 in a lane with neither bit only when some other lane has one", wrong);
    }

    #[test]
    fn prove_keeps_only_one_of_two_queries_either_of_which_answers_a_store() {
        let (b, first, second) = either_answer();
        let wrong: Vec<&str> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| kept.queries.contains(&first) && kept.queries.contains(&second))
            .map(|(name, _)| *name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a kept query answers any(c), which already holds wherever the other query's own bit c does", wrong);
    }

    #[test]
    fn prove_needs_no_meeting_across_a_kept_query() {
        let (program, s1, s2) = with_between(|b, e, k| {
            let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
            let flags = k.buffer(b, e, 8);
            let own = byte_offset(b, e, flags, lane, 4);
            let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
            let zero = b.constant(e, Ty::I32, 0);
            let set = b.cmp(e, IntPred::Ne, flag, zero);
            let c = b.int(e, IntOp::And, set, k.exec);
            let q = b.wave(e, WaveOp::Any, vec![c]);
            let one = b.constant(e, Ty::I32, 1);
            let data = b.core(e, Ty::I32, Op::Select(q, one, zero));
            let out = k.buffer(b, e, 16);
            let slot = byte_offset(b, e, out, lane, 4);
            b.store(e, Space::Global, MemSize::B32, slot, data, k.exec);
        });
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards);
            assert_eq!(kept.queries.len(), 1, "{}: a lane without the flag stores 0 unless the query stays", name);
        }
        keeps_exactly(&program, &hazards, &[], "the kept query already aligns every lane between the stores");
    }

    #[test]
    fn prove_orders_two_stores_across_a_converted_query() {
        let (program, s1, s2) = with_between(|b, e, k| {
            let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
            let flags = k.buffer(b, e, 8);
            let own = byte_offset(b, e, flags, lane, 4);
            let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
            let zero = b.constant(e, Ty::I32, 0);
            let set = b.cmp(e, IntPred::Ne, flag, zero);
            let c = b.int(e, IntOp::And, set, k.exec);
            let q = b.wave(e, WaveOp::Any, vec![c]);
            let one = b.constant(e, Ty::I32, 1);
            let data = b.core(e, Ty::I32, Op::Select(q, one, zero));
            let out = k.buffer(b, e, 16);
            let slot = byte_offset(b, e, out, lane, 4);
            b.store(e, Space::Global, MemSize::B32, slot, data, c);
        });
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards);
            assert!(kept.queries.is_empty(), "{}: only lanes with the flag store the answer, and they see true either way", name);
        }
        keeps_exactly(&program, &hazards, &[s2], "the converted query aligns no lanes, so the second store needs its meeting");
    }

    fn flag_between(b: &mut Build, e: BlockId, k: &Kernel) -> (ValueId, ValueId, ValueId) {
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let flags = k.buffer(b, e, 8);
        let own = byte_offset(b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let out = k.buffer(b, e, 16);
        let slot = byte_offset(b, e, out, lane, 4);
        (flag, c, slot)
    }

    #[test]
    fn prove_needs_no_meeting_across_a_ballot_whose_whole_word_is_stored() {
        let (program, s1, s2) = with_between(|b, e, k| {
            let (_, c, slot) = flag_between(b, e, k);
            let w = b.wave(e, WaveOp::Ballot, vec![c]);
            b.store(e, Space::Global, MemSize::B32, slot, w, k.exec);
        });
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        keeps_exactly(&program, &hazards, &[], "the stored ballot word is computed by the whole wave, which aligns every lane between the stores");
    }

    #[test]
    fn prove_orders_two_stores_across_a_ballot_whose_own_bit_alone_is_used() {
        let (program, s1, s2) = with_between(|b, e, k| {
            let (_, c, slot) = flag_between(b, e, k);
            let w = b.wave(e, WaveOp::Ballot, vec![c]);
            let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
            let shifted = b.int(e, IntOp::LShr, w, lane);
            let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
            let one = b.constant(e, Ty::I32, 1);
            b.store(e, Space::Global, MemSize::B32, slot, one, bit);
        });
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        keeps_exactly(&program, &hazards, &[s2], "each lane computes its own bit of the ballot, which aligns no lanes");
    }

    #[test]
    fn prove_needs_no_meeting_across_a_first_lane_read_of_a_varying_word() {
        let (program, s1, s2) = with_between(|b, e, k| {
            let (flag, _, slot) = flag_between(b, e, k);
            let x = b.wave(e, WaveOp::ReadFirstLane, vec![flag, k.exec]);
            b.store(e, Space::Global, MemSize::B32, slot, x, k.exec);
        });
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        keeps_exactly(&program, &hazards, &[], "reading the first lane's flag stays a wave operation, which aligns every lane between the stores");
    }

    #[test]
    fn prove_orders_two_stores_across_a_first_lane_read_of_a_uniform_word() {
        let (program, s1, s2) = with_between(|b, e, k| {
            let (_, _, slot) = flag_between(b, e, k);
            let seven = b.constant(e, Ty::I32, 7);
            let x = b.wave(e, WaveOp::ReadFirstLane, vec![seven, k.exec]);
            b.store(e, Space::Global, MemSize::B32, slot, x, k.exec);
        });
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        keeps_exactly(&program, &hazards, &[s2], "every lane holds 7, so the read becomes the lane's own 7 and aligns no lanes");
    }

    #[test]
    fn prove_needs_no_meeting_when_only_inactive_lanes_read_the_word() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let out = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let zero = b.constant(e, Ty::I32, 0);
        let first = b.cmp(e, IntPred::Eq, lane, zero);
        let exec = b.int(e, IntOp::And, first, k.exec);
        let own = byte_offset(&mut b, e, out, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        b.br(e, then, vec![exec, buf, own]);
        let s = store(&mut b, then, t[1], 7, t[0]);
        let yes = b.constant(then, Ty::I1, 1);
        let l = b.here(then);
        let v = b.load(then, Space::Global, MemSize::B32, t[1], yes);
        b.store(then, Space::Global, MemSize::B32, t[2], v, t[0]);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(s, l)], &[], &[(l, s, false)]);
        keeps_exactly(&program, &hazards, &[], "only lane 0 keeps what it loads, and only lanes with exec clear read lane 0's word");
    }

    fn sliding_loop(lane_step: bool) -> (crate::rdna_spmd::program::Program, Position) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, zero, buf]);
        let index = if lane_step {
            let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
            b.int(body, IntOp::Add, p[1], lane)
        } else {
            p[1]
        };
        let address = byte_offset(&mut b, body, p[2], index, 4);
        let s = b.here(body);
        b.store(body, Space::Global, MemSize::B32, address, p[1], p[0]);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[1], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        b.cond_br(body, again, (body, vec![p[0], next, p[2]]), (exit, vec![p[0]]));
        (b.program(), s)
    }

    #[test]
    fn prove_orders_iterations_that_store_to_each_other_words() {
        let (program, s) = sliding_loop(true);
        let hazards = Hazards::given(&program, &[], &[(s, s)], &[]);
        keeps_exactly(&program, &hazards, &[s], "lane a's store in iteration i + 1 must follow lane a + 1's in iteration i");
    }

    #[test]
    fn prove_needs_no_meeting_for_one_instruction_within_one_iteration() {
        let (program, s) = sliding_loop(false);
        let hazards = Hazards::given(&program, &[(s, s)], &[], &[]);
        keeps_exactly(&program, &hazards, &[], "lanes of one instruction have no order in the wave either");
    }

    fn converted(b: &Build) -> Vec<&'static str> {
        ["search", "direct"]
            .iter()
            .zip(both(b))
            .filter(|(_, kept)| !kept.queries.is_empty() || !kept.words.is_empty())
            .map(|(name, _)| *name)
            .collect()
    }

    fn uniform_load(b: &mut Build, k: &Kernel, e: BlockId, offset: u64) -> ValueId {
        let table = k.buffer(b, e, offset);
        let yes = b.constant(e, Ty::I1, 1);
        b.load(e, Space::Global, MemSize::B32, table, yes)
    }

    fn per_lane(b: &mut Build, k: &Kernel, e: BlockId, offset: u64) -> ValueId {
        let table = k.buffer(b, e, offset);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(b, e, table, lane, 4);
        b.load(e, Space::Global, MemSize::B32, own, k.exec)
    }

    fn query_data(b: &mut Build, k: &Kernel, e: BlockId) -> ValueId {
        let flag = per_lane(b, k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        b.core(e, Ty::I32, Op::Select(q, one, two))
    }

    fn store_own(b: &mut Build, k: &Kernel, e: BlockId, data: ValueId, mask: ValueId) {
        let buf = k.buffer(b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(b, e, buf, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, data, mask);
    }

    fn two_orders(first: (IntPred, bool), second: (IntPred, bool)) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let v = per_lane(&mut b, &k, e, 16);
        let w = per_lane(&mut b, &k, e, 24);
        let order = |b: &mut Build, (p, swap): (IntPred, bool)| if swap { b.cmp(e, p, w, v) } else { b.cmp(e, p, v, w) };
        let one = order(&mut b, first);
        let other = order(&mut b, second);
        let both = b.int(e, IntOp::And, one, other);
        let mask = b.int(e, IntOp::And, both, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_two_orders_that_may_both_hold_mask() {
        let b = two_orders((IntPred::Ult, false), (IntPred::Ule, false));
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: v < w and v <= w both hold when v < w, so lanes store the answer", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_store_an_order_and_its_swap_mask() {
        let b = two_orders((IntPred::Slt, false), (IntPred::Sgt, true));
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: w s> v is v s< w, so lanes with v s< w store the answer", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_store_two_opposite_orders_that_hold_at_equality_mask() {
        let b = two_orders((IntPred::Ule, false), (IntPred::Ule, true));
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: v <= w and w <= v both hold when v == w", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_store_opposite_orders_of_different_signedness_mask() {
        let b = two_orders((IntPred::Slt, false), (IntPred::Ult, true));
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: v = -1 and w = 0 give v s< w and w u< v", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_store_an_equality_and_an_order_that_hold_together_mask() {
        let b = two_orders((IntPred::Eq, false), (IntPred::Ule, false));
        assert!(keeps(&b).is_empty(), "{:?}: v == w gives v <= w", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_an_equality_and_a_strict_order_mask() {
        let b = two_orders((IntPred::Eq, false), (IntPred::Ult, false));
        assert!(converted(&b).is_empty(), "{:?}: v == w and v < w never hold together", converted(&b));
    }

    fn orders_in_two_blocks(opposite: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let v = per_lane(&mut b, &k, e, 16);
        let w = per_lane(&mut b, &k, e, 24);
        let first = b.cmp(e, IntPred::Ult, v, w);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32, Ty::I1]);
        b.br(e, next, vec![k.exec, own, data, v, w, first]);
        let second = b.cmp(next, if opposite { IntPred::Uge } else { IntPred::Ule }, p[3], p[4]);
        let both = b.int(next, IntOp::And, p[5], second);
        let mask = b.int(next, IntOp::And, both, p[0]);
        b.store(next, Space::Global, MemSize::B32, p[1], p[2], mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_orders_in_two_blocks_that_may_both_hold_mask() {
        let b = orders_in_two_blocks(false);
        assert!(keeps(&b).is_empty(), "{:?}: v < w gives v <= w in the next block", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_opposite_orders_in_two_blocks_mask() {
        let b = orders_in_two_blocks(true);
        assert!(converted(&b).is_empty(), "{:?}: v < w in the first block rules out v >= w in the next", converted(&b));
    }

    fn float_bounds(above: u64) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let word = per_lane(&mut b, &k, e, 16);
        let x = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, word));
        let five = b.constant(e, Ty::F32, 0x40a0_0000);
        let bound = b.constant(e, Ty::F32, above);
        let small = b.core(e, Ty::I1, Op::FCmp(FloatPred::Olt, x, five));
        let large = b.core(e, Ty::I1, Op::FCmp(FloatPred::Ogt, x, bound));
        let both = b.int(e, IntOp::And, small, large);
        let mask = b.int(e, IntOp::And, both, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_two_float_bounds_that_may_both_hold_mask() {
        let b = float_bounds(0x3f80_0000);
        assert!(keeps(&b).is_empty(), "{:?}: x = 2.0 is below 5.0 and above 1.0", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_disjoint_float_bounds_mask() {
        let b = float_bounds(0x4120_0000);
        assert!(converted(&b).is_empty(), "{:?}: no float is below 5.0 and above 10.0", converted(&b));
    }

    fn two_bounds(below: u64, above: u64) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let v = per_lane(&mut b, &k, e, 16);
        let below = b.constant(e, Ty::I32, below);
        let above = b.constant(e, Ty::I32, above);
        let small = b.cmp(e, IntPred::Ult, v, below);
        let large = b.cmp(e, IntPred::Ugt, v, above);
        let both = b.int(e, IntOp::And, small, large);
        let mask = b.int(e, IntOp::And, both, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    fn bounded(first: (IntPred, u64), second: (IntPred, u64)) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let v = per_lane(&mut b, &k, e, 16);
        let mut bits = Vec::new();
        for (p, bound) in [first, second] {
            let bound = b.constant(e, Ty::I32, bound);
            bits.push(b.cmp(e, p, v, bound));
        }
        let both = b.int(e, IntOp::And, bits[0], bits[1]);
        let mask = b.int(e, IntOp::And, both, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_follows_pairs_of_bounds_on_one_word() {
        let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
        let bounds = [0u32, 1, 4, 5, 6, 10, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
        let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
        let signed = |p: IntPred| matches!(p, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
        let equality = |p: IntPred| matches!(p, IntPred::Eq | IntPred::Ne);
        let mut r = Random::new(29);
        let mut wrong = Vec::new();
        for _ in 0..400 {
            let (p1, k1) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
            let (p2, k2) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
            let mut candidates = vec![0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
            for k in [k1, k2] {
                candidates.extend([k.wrapping_sub(1), k, k.wrapping_add(1)]);
            }
            let possible = candidates.iter().any(|&v| holds(p1, v, k1) && holds(p2, v, k2));
            let b = bounded((p1, k1 as u64), (p2, k2 as u64));
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                let converted = kept.queries.is_empty();
                if possible && converted {
                    wrong.push(format!("{}: v {:?} {} and v {:?} {} can both hold, yet the query was converted", name, p1, k1, p2, k2));
                }
                if !possible && !converted {
                    wrong.push(format!("{}: v {:?} {} and v {:?} {} never both hold, yet the query stays", name, p1, k1, p2, k2));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn prove_follows_a_bound_into_a_block_and_a_bound_inside_it() {
        let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
        let bounds = [0u32, 1, 4, 5, 6, 10, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
        let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
        let mut r = Random::new(41);
        let mut wrong = Vec::new();
        for _ in 0..200 {
            let (p1, k1) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
            let (p2, k2) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
            let mut candidates = vec![0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
            for k in [k1, k2] {
                candidates.extend([k.wrapping_sub(1), k, k.wrapping_add(1)]);
            }
            let possible = candidates.iter().any(|&v| holds(p1, v, k1) && holds(p2, v, k2));
            let b = branch_then_store((p1, k1 as u64), (p2, k2 as u64));
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                let converted = kept.queries.is_empty();
                if possible && converted {
                    wrong.push(format!("{}: x {:?} {} into the block and x {:?} {} inside it can both hold, yet the query was converted", name, p1, k1, p2, k2));
                }
                if !possible && !converted {
                    wrong.push(format!("{}: x {:?} {} into the block and x {:?} {} inside it never both hold, yet the query stays", name, p1, k1, p2, k2));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    fn ordered_words(tests: &[(IntPred, usize, usize)]) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let words: Vec<ValueId> = [16, 24, 32].iter().map(|&o| per_lane(&mut b, &k, e, o)).collect();
        let mut mask = k.exec;
        for &(p, i, j) in tests {
            let c = b.cmp(e, p, words[i], words[j]);
            mask = b.int(e, IntOp::And, mask, c);
        }
        store_own(&mut b, &k, e, data, mask);
        b
    }

    fn ordered_and_bounded_words(tests: &[(IntPred, usize, Option<usize>, u32)]) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let words: Vec<ValueId> = [16, 24].iter().map(|&o| per_lane(&mut b, &k, e, o)).collect();
        let mut mask = k.exec;
        for &(p, i, j, bound) in tests {
            let other = match j {
                Some(j) => words[j],
                None => b.constant(e, Ty::I32, bound as u64),
            };
            let c = b.cmp(e, p, words[i], other);
            mask = b.int(e, IntOp::And, mask, c);
        }
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_follows_orders_of_both_signs_among_words_bounded_by_constants() {
        let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
        let bounds = [0u32, 1, 5, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_ffff];
        let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
        let points = [0u32, 1, 2, 4, 5, 6, 0x7fff_fffe, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0x8000_0002, 0xffff_fffe, 0xffff_ffff];
        let mut r = Random::new(71);
        let mut wrong = Vec::new();
        for _ in 0..300 {
            let count = 2 + r.below(3) as usize;
            let tests: Vec<(IntPred, usize, Option<usize>, u32)> = (0..count)
                .map(|_| {
                    let i = r.below(2) as usize;
                    let p = preds[r.below(10) as usize];
                    if r.below(2) == 0 {
                        (p, i, Some(1 - i), 0)
                    } else {
                        (p, i, None, bounds[r.below(bounds.len() as u64) as usize])
                    }
                })
                .collect();
            let possible = points.iter().any(|&x| {
                points.iter().any(|&y| {
                    tests.iter().all(|&(p, i, j, bound)| {
                        let w = [x, y];
                        holds(p, w[i], j.map_or(bound, |j| w[j]))
                    })
                })
            });
            let b = ordered_and_bounded_words(&tests);
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                if possible && kept.queries.is_empty() {
                    wrong.push(format!("{}: {:?} can all hold, yet the query was converted", name, tests));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    fn branch_then_store_offset(entry: (IntPred, u32), offset: u32, store: (IntPred, u32)) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let table = k.buffer(&mut b, e, 16);
        let yes = b.constant(e, Ty::I1, 1);
        let x = b.load(e, Space::Global, MemSize::B32, table, yes);
        let bound = b.constant(e, Ty::I32, entry.1 as u64);
        let enters = b.cmp(e, entry.0, x, bound);
        let shift = b.constant(e, Ty::I32, offset as u64);
        let moved = b.int(e, IntOp::Add, x, shift);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.cond_br(e, enters, (then, vec![k.exec, moved, data, own]), (exit, vec![k.exec]));
        let limit = b.constant(then, Ty::I32, store.1 as u64);
        let test = b.cmp(then, store.0, t[1], limit);
        let mask = b.int(then, IntOp::And, test, t[0]);
        b.store(then, Space::Global, MemSize::B32, t[3], t[2], mask);
        b
    }

    #[test]
    fn prove_follows_a_bound_carried_through_an_offset_into_the_next_block() {
        let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
        let bounds = [0u32, 1, 4, 5, 6, 10, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
        let offsets = [0u32, 1, 5, 0x8000_0000, 0xffff_ffff];
        let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
        let mut r = Random::new(73);
        let mut wrong = Vec::new();
        for _ in 0..200 {
            let (p1, k1) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
            let (p2, k2) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
            let c = offsets[r.below(offsets.len() as u64) as usize];
            let mut candidates = vec![0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
            for k in [k1, k2.wrapping_sub(c)] {
                candidates.extend([k.wrapping_sub(1), k, k.wrapping_add(1)]);
            }
            for x in candidates.clone() {
                candidates.push(x.wrapping_sub(c));
            }
            let possible = candidates.iter().any(|&x| holds(p1, x, k1) && holds(p2, x.wrapping_add(c), k2));
            let b = branch_then_store_offset((p1, k1), c, (p2, k2));
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                let converted = kept.queries.is_empty();
                if possible && converted {
                    wrong.push(format!("{}: x {:?} {} into the block and x + {} {:?} {} inside it can both hold, yet the query was converted", name, p1, k1, c, p2, k2));
                }
                if !possible && !converted {
                    wrong.push(format!("{}: x {:?} {} into the block and x + {} {:?} {} inside it never both hold, yet the query stays", name, p1, k1, c, p2, k2));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    fn orders_then_compare(first: &[(IntPred, bool)], second: (IntPred, bool)) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let v = per_lane(&mut b, &k, e, 16);
        let w = per_lane(&mut b, &k, e, 24);
        let mut known = b.constant(e, Ty::I1, 1);
        for &(p, swap) in first {
            let c = if swap { b.cmp(e, p, w, v) } else { b.cmp(e, p, v, w) };
            known = b.int(e, IntOp::And, known, c);
        }
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32, Ty::I1]);
        b.br(e, next, vec![k.exec, own, data, v, w, known]);
        let later = if second.1 { b.cmp(next, second.0, p[4], p[3]) } else { b.cmp(next, second.0, p[3], p[4]) };
        let both_hold = b.int(next, IntOp::And, p[5], later);
        let mask = b.int(next, IntOp::And, both_hold, p[0]);
        b.store(next, Space::Global, MemSize::B32, p[1], p[2], mask);
        b
    }

    #[test]
    fn prove_follows_orders_of_two_words_into_the_next_block() {
        let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
        let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
        let points = [0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_fffe, 0xffff_ffff];
        let family = |p: IntPred| match p {
            IntPred::Eq | IntPred::Ne => 0,
            IntPred::Ult | IntPred::Ule | IntPred::Ugt | IntPred::Uge => 1,
            _ => 2,
        };
        let mut r = Random::new(79);
        let mut wrong = Vec::new();
        for _ in 0..300 {
            let first: Vec<(IntPred, bool)> = (0..1 + r.below(2)).map(|_| (preds[r.below(10) as usize], r.below(2) == 0)).collect();
            let second = (preds[r.below(10) as usize], r.below(2) == 0);
            let test = |(p, swap): (IntPred, bool), v: u32, w: u32| if swap { holds(p, w, v) } else { holds(p, v, w) };
            let possible = points.iter().any(|&v| points.iter().any(|&w| first.iter().all(|&t| test(t, v, w)) && test(second, v, w)));
            let families: BTreeSet<i32> = first.iter().chain([&second]).map(|&(p, _)| family(p)).filter(|&f| f != 0).collect();
            let b = orders_then_compare(&first, second);
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                let converted = kept.queries.is_empty();
                if possible && converted {
                    wrong.push(format!("{}: {:?} then {:?} can both hold, yet the query was converted", name, first, second));
                }
                if !possible && !converted && families.len() <= 1 {
                    wrong.push(format!("{}: {:?} then {:?} never both hold, yet the query stays", name, first, second));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn prove_follows_orders_among_three_words() {
        let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
        let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
        let points = [0u32, 1, 2, 0x8000_0000, 0x8000_0001, 0x8000_0002, 0xffff_fffe, 0xffff_ffff];
        let family = |p: IntPred| match p {
            IntPred::Eq | IntPred::Ne => 0,
            IntPred::Ult | IntPred::Ule | IntPred::Ugt | IntPred::Uge => 1,
            _ => 2,
        };
        let mut r = Random::new(43);
        let mut wrong = Vec::new();
        for _ in 0..300 {
            let count = 2 + r.below(2) as usize;
            let tests: Vec<(IntPred, usize, usize)> = (0..count)
                .map(|_| {
                    let i = r.below(3) as usize;
                    let j = (i + 1 + r.below(2) as usize) % 3;
                    (preds[r.below(10) as usize], i, j)
                })
                .collect();
            let possible = points.iter().any(|&x| {
                points.iter().any(|&y| points.iter().any(|&z| tests.iter().all(|&(p, i, j)| holds(p, [x, y, z][i], [x, y, z][j]))))
            });
            let families: BTreeSet<i32> = tests.iter().map(|&(p, _, _)| family(p)).filter(|&f| f != 0).collect();
            let b = ordered_words(&tests);
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                let converted = kept.queries.is_empty();
                if possible && converted {
                    wrong.push(format!("{}: {:?} can all hold, yet the query was converted", name, tests));
                }
                if !possible && !converted && families.len() <= 1 {
                    wrong.push(format!("{}: {:?} never all hold, yet the query stays", name, tests));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    fn float_pair(first: (FloatPred, f32), second: (FloatPred, f32)) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let word = per_lane(&mut b, &k, e, 16);
        let x = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, word));
        let mut mask = k.exec;
        for (p, c) in [first, second] {
            let bound = b.constant(e, Ty::F32, c.to_bits() as u64);
            let t = b.core(e, Ty::I1, Op::FCmp(p, x, bound));
            mask = b.int(e, IntOp::And, mask, t);
        }
        store_own(&mut b, &k, e, data, mask);
        b
    }

    fn float_orders_of_two_words(tests: &[(FloatPred, bool)]) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let v = per_lane(&mut b, &k, e, 16);
        let w = per_lane(&mut b, &k, e, 24);
        let x = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, v));
        let y = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, w));
        let mut mask = k.exec;
        for &(p, swap) in tests {
            let t = if swap { b.core(e, Ty::I1, Op::FCmp(p, y, x)) } else { b.core(e, Ty::I1, Op::FCmp(p, x, y)) };
            mask = b.int(e, IntOp::And, mask, t);
        }
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_follows_orders_of_two_float_words() {
        use FloatPred::*;
        let preds = [Oeq, Ogt, Oge, Olt, Ole, One, Ord, Uno, Ueq, Ugt, Uge, Ult, Ule, Une];
        let points = [0.0f32, -0.0, 1.0, -1.0, 2.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let mut r = Random::new(83);
        let mut wrong = Vec::new();
        for _ in 0..300 {
            let tests: Vec<(FloatPred, bool)> = (0..2 + r.below(2)).map(|_| (preds[r.below(14) as usize], r.below(2) == 0)).collect();
            let holds = |(p, swap): (FloatPred, bool), x: f32, y: f32| {
                let (a, b) = if swap { (y, x) } else { (x, y) };
                super::super::logic::float_compare(p, a as f64, b as f64)
            };
            let possible = points.iter().any(|&x| points.iter().any(|&y| tests.iter().all(|&t| holds(t, x, y))));
            let ordered = tests.iter().all(|&(p, _)| matches!(p, Oeq | Ogt | Oge | Olt | Ole));
            let b = float_orders_of_two_words(&tests);
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                let converted = kept.queries.is_empty();
                if possible && converted {
                    wrong.push(format!("{}: {:?} can all hold, yet the query was converted", name, tests));
                }
                if !possible && !converted && ordered {
                    wrong.push(format!("{}: {:?} never all hold, yet the query stays", name, tests));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn prove_follows_pairs_of_float_bounds_on_one_word() {
        use FloatPred::*;
        let preds = [Oeq, Ogt, Oge, Olt, Ole, One, Ord, Uno, Ueq, Ugt, Uge, Ult, Ule, Une];
        let bounds = [0.0f32, -0.0, 1.0, -1.0, 5.0, 10.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let mut r = Random::new(47);
        let mut wrong = Vec::new();
        for _ in 0..300 {
            let (p1, k1) = (preds[r.below(14) as usize], bounds[r.below(9) as usize]);
            let (p2, k2) = (preds[r.below(14) as usize], bounds[r.below(9) as usize]);
            let mut candidates = vec![0.0f32, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN];
            for k in [k1, k2] {
                if !k.is_nan() {
                    candidates.extend([f32::from_bits(k.to_bits().wrapping_sub(1)), k, f32::from_bits(k.to_bits().wrapping_add(1))]);
                    candidates.extend([k - 0.5, k + 0.5]);
                }
            }
            let possible = candidates.iter().any(|&v| float_compare(p1, v as f64, k1 as f64) && float_compare(p2, v as f64, k2 as f64));
            let b = float_pair((p1, k1), (p2, k2));
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                let converted = kept.queries.is_empty();
                if possible && converted {
                    wrong.push(format!("{}: x {:?} {} and x {:?} {} can both hold, yet the query was converted", name, p1, k1, p2, k2));
                }
                if !possible && !converted {
                    wrong.push(format!("{}: x {:?} {} and x {:?} {} never both hold, yet the query stays", name, p1, k1, p2, k2));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn prove_converts_a_query_whose_store_two_different_equalities_mask() {
        let b = bounded((IntPred::Eq, 5), (IntPred::Eq, 7));
        assert!(converted(&b).is_empty(), "{:?}: v == 5 and v == 7 never hold together", converted(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_an_equality_and_its_negation_mask() {
        let b = bounded((IntPred::Eq, 5), (IntPred::Ne, 5));
        assert!(converted(&b).is_empty(), "{:?}: v == 5 and v != 5 never hold together", converted(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_an_equality_outside_a_bound_masks() {
        let b = bounded((IntPred::Eq, 5), (IntPred::Ult, 3));
        assert!(converted(&b).is_empty(), "{:?}: v == 5 and v < 3 never hold together", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_store_an_equality_inside_a_bound_masks() {
        for second in [(IntPred::Ult, 6), (IntPred::Eq, 5), (IntPred::Ne, 7)] {
            let b = bounded((IntPred::Eq, 5), second);
            let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
            assert!(wrong.is_empty(), "{:?}: v == 5 and v {:?} {} both hold for v = 5", wrong, second.0, second.1);
        }
    }

    #[test]
    fn prove_converts_a_query_whose_store_disjoint_ranges_mask() {
        let b = two_bounds(5, 10);
        assert!(converted(&b).is_empty(), "{:?}: v < 5 and v > 10 never hold together, so the store never runs", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_store_overlapping_ranges_mask() {
        let b = two_bounds(10, 5);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: v < 10 and v > 5 both hold for v = 7", wrong);
    }

    #[test]
    fn prove_converts_a_query_whose_store_opposite_orders_mask() {
        let b = two_orders((IntPred::Ult, false), (IntPred::Ult, true));
        assert!(converted(&b).is_empty(), "{:?}: v < w and w < v never hold together, so the store never runs", converted(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_a_contradiction_masks() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let v = per_lane(&mut b, &k, e, 16);
        let w = per_lane(&mut b, &k, e, 24);
        let below = b.cmp(e, IntPred::Ult, v, w);
        let above = b.cmp(e, IntPred::Uge, v, w);
        let never = b.int(e, IntOp::And, below, above);
        let mask = b.int(e, IntOp::And, never, k.exec);
        store_own(&mut b, &k, e, data, mask);
        assert!(converted(&b).is_empty(), "{:?}: v < w and v >= w never hold together, so the store never runs", converted(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_disjoint_constants_mask() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let none = b.int(e, IntOp::And, one, two);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let shifted = b.int(e, IntOp::LShr, none, lane);
        let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let mask = b.int(e, IntOp::And, bit, k.exec);
        store_own(&mut b, &k, e, data, mask);
        assert!(converted(&b).is_empty(), "{:?}: 1 & 2 is 0, so the store never runs", converted(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_the_branch_into_its_block_rules_out() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let table = k.buffer(&mut b, e, 16);
        let yes = b.constant(e, Ty::I1, 1);
        let x = b.load(e, Space::Global, MemSize::B32, table, yes);
        let five = b.constant(e, Ty::I32, 5);
        let is_five = b.cmp(e, IntPred::Eq, x, five);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.cond_br(e, is_five, (then, vec![k.exec, x, data, own]), (exit, vec![k.exec]));
        let five = b.constant(then, Ty::I32, 5);
        let still = b.cmp(then, IntPred::Ne, t[1], five);
        let mask = b.int(then, IntOp::And, still, t[0]);
        b.store(then, Space::Global, MemSize::B32, t[3], t[2], mask);
        assert!(converted(&b).is_empty(), "{:?}: the block runs only when x is 5, where the store's mask x != 5 is false", converted(&b));
    }

    fn branch_then_store(entry: (IntPred, u64), store: (IntPred, u64)) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let table = k.buffer(&mut b, e, 16);
        let yes = b.constant(e, Ty::I1, 1);
        let x = b.load(e, Space::Global, MemSize::B32, table, yes);
        let bound = b.constant(e, Ty::I32, entry.1);
        let enters = b.cmp(e, entry.0, x, bound);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.cond_br(e, enters, (then, vec![k.exec, x, data, own]), (exit, vec![k.exec]));
        let limit = b.constant(then, Ty::I32, store.1);
        let test = b.cmp(then, store.0, t[1], limit);
        let mask = b.int(then, IntOp::And, test, t[0]);
        b.store(then, Space::Global, MemSize::B32, t[3], t[2], mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_the_branch_into_its_block_lets_through() {
        let b = branch_then_store((IntPred::Eq, 5), (IntPred::Eq, 5));
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: the block runs only when x is 5, where the store's mask x == 5 holds", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_store_a_branch_on_another_constant_lets_through() {
        let b = branch_then_store((IntPred::Eq, 6), (IntPred::Ne, 5));
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: x is 6 in the block, where x != 5 holds", wrong);
    }

    #[test]
    fn prove_converts_a_query_whose_store_the_bound_into_its_block_rules_out() {
        let b = branch_then_store((IntPred::Ult, 10), (IntPred::Uge, 10));
        assert!(converted(&b).is_empty(), "{:?}: x < 10 in the block, where x >= 10 is false", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_store_the_bound_into_its_block_lets_through() {
        let b = branch_then_store((IntPred::Ult, 10), (IntPred::Ult, 10));
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: x < 10 in the block, where the store's mask x < 10 holds", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_store_a_bound_of_the_other_signedness_lets_through() {
        let b = branch_then_store((IntPred::Slt, 10), (IntPred::Uge, 10));
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: x = -1 is below 10 signed and at least 10 unsigned", wrong);
    }

    fn chosen_constants(low: u64) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let flag = per_lane(&mut b, &k, e, 16);
        let zero = b.constant(e, Ty::I32, 0);
        let pick = b.cmp(e, IntPred::Ne, flag, zero);
        let (two, four) = (b.constant(e, Ty::I32, 2), b.constant(e, Ty::I32, 4));
        let chosen = b.core(e, Ty::I32, Op::Select(pick, two, four));
        let low = b.constant(e, Ty::I32, low);
        let both = b.int(e, IntOp::And, low, chosen);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let shifted = b.int(e, IntOp::LShr, both, lane);
        let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let mask = b.int(e, IntOp::And, bit, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    fn constants_over_a_ballot(low: u64) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let flag = per_lane(&mut b, &k, e, 16);
        let zero = b.constant(e, Ty::I32, 0);
        let pick = b.cmp(e, IntPred::Ne, flag, zero);
        let pick = b.int(e, IntOp::And, pick, k.exec);
        let word = b.wave(e, WaveOp::Ballot, vec![pick]);
        let low = b.constant(e, Ty::I32, low);
        let two = b.constant(e, Ty::I32, 2);
        let inner = b.int(e, IntOp::And, word, low);
        let both = b.int(e, IntOp::And, inner, two);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let shifted = b.int(e, IntOp::LShr, both, lane);
        let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let mask = b.int(e, IntOp::And, bit, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_store_two_constants_over_a_ballot_mask_off() {
        let b = constants_over_a_ballot(1);
        assert!(converted(&b).is_empty(), "{:?}: ballot & 1 & 2 is 0, so the store never runs", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_store_two_constants_over_a_ballot_may_mask_on() {
        let b = constants_over_a_ballot(3);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: ballot & 3 & 2 keeps bit 1, so lane 1 stores whenever its flag is set", wrong);
    }

    #[test]
    fn prove_converts_a_query_whose_store_a_constant_and_a_chosen_constant_mask_off() {
        let b = chosen_constants(1);
        assert!(converted(&b).is_empty(), "{:?}: 1 & 2 and 1 & 4 are 0, so the store never runs", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_store_a_constant_and_a_chosen_constant_may_mask_on() {
        let b = chosen_constants(3);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: 3 & 2 sets bit 1, so lane 1 stores whenever it picks 2", wrong);
    }

    #[test]
    fn prove_converts_a_query_whose_value_an_and_with_zero_drops() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let zero = b.constant(e, Ty::I32, 0);
        let nothing = b.int(e, IntOp::And, data, zero);
        store_own(&mut b, &k, e, nothing, k.exec);
        assert!(converted(&b).is_empty(), "{:?}: every lane stores 0 whatever the query answers", converted(&b));
    }

    fn written_first_lane(only: u64) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flag = per_lane(&mut b, &k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let target = b.constant(e, Ty::I32, only);
        let mine = b.cmp(e, IntPred::Eq, lane, target);
        let both = b.int(e, IntOp::And, set, mine);
        let c = b.int(e, IntOp::And, both, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let x = b.core(e, Ty::I32, Op::Select(q, one, two));
        let five = b.constant(e, Ty::I32, 5);
        let y = b.wave(e, WaveOp::WriteLane, vec![x, five, lane]);
        store_own(&mut b, &k, e, y, k.exec);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_word_a_lane_write_takes_from_the_one_lane_that_answers() {
        let b = written_first_lane(0);
        assert!(converted(&b).is_empty(), "{:?}: the write takes lane 0's word, and lane 0's own answer is the wave's", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_word_a_lane_write_takes_from_a_lane_that_cannot_answer() {
        let b = written_first_lane(1);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: lane 0 alone answers false while the wave answers lane 1's flag", wrong);
    }

    fn written_elsewhere(skip: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let x = b.core(e, Ty::I32, Op::Select(q, one, two));
        let five = b.constant(e, Ty::I32, 5);
        let y = b.wave(e, WaveOp::WriteLane, vec![x, five, lane]);
        let mask = if skip {
            let other = b.cmp(e, IntPred::Ne, lane, five);
            b.int(e, IntOp::And, other, k.exec)
        } else {
            k.exec
        };
        store_own(&mut b, &k, e, y, mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_word_a_lane_write_puts_into_a_lane_that_stores() {
        let b = written_elsewhere(false);
        assert!(keeps(&b).is_empty(), "{:?}: lane 5 stores lane 0's word, whose answer is lane 0's own flag", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_word_a_lane_write_puts_only_into_a_lane_that_never_stores() {
        let b = written_elsewhere(true);
        assert!(converted(&b).is_empty(), "{:?}: every storing lane keeps its own lane id", converted(&b));
    }

    fn offset_lanes(uniform: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let u = uniform_load(&mut b, &k, e, 16);
        let shifted = b.int(e, IntOp::Add, lane, u);
        let five = b.constant(e, Ty::I32, 5);
        let other = if uniform { b.int(e, IntOp::Add, lane, five) } else { five };
        let t = b.cmp(e, IntPred::Eq, shifted, other);
        let c = b.int(e, IntOp::And, t, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let data = b.core(e, Ty::I32, Op::Select(q, one, two));
        store_own(&mut b, &k, e, data, k.exec);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_bit_one_lane_offset_by_a_word_sets() {
        let b = offset_lanes(false);
        assert!(keeps(&b).is_empty(), "{:?}: lane + u == 5 holds in lane 5 - u alone", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_bit_every_lane_offset_by_a_word_shares() {
        let b = offset_lanes(true);
        assert!(converted(&b).is_empty(), "{:?}: lane + u == lane + 5 is u == 5 in every lane", converted(&b));
    }

    fn permuted(op: WaveOp, index: impl Fn(&mut Build, BlockId, ValueId) -> ValueId) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let zero = b.constant(e, Ty::I32, 0);
        let fifteen = b.constant(e, Ty::I32, 15);
        let low = b.int(e, IntOp::And, lane, fifteen);
        let first = b.cmp(e, IntPred::Eq, low, zero);
        let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
        let picked = b.core(e, Ty::I32, Op::Select(q, ones, lane));
        let x = b.core(e, Ty::I32, Op::Select(first, lane, picked));
        let index = index(&mut b, e, lane);
        let y = match op {
            WaveOp::ReadLane => b.wave(e, WaveOp::ReadLane, vec![x, index, zero]),
            _ => b.wave(e, op, vec![index, x, k.exec]),
        };
        store_own(&mut b, &k, e, y, k.exec);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_word_only_lanes_a_permute_skips_hold() {
        let mut wrong = Vec::new();
        for op in [WaveOp::Bpermute, WaveOp::BpermuteFi] {
            let b = permuted(op, |b, e, _| b.constant(e, Ty::I32, 0));
            if !converted(&b).is_empty() {
                wrong.push(format!("{:?} of lane 0: {:?}", op, converted(&b)));
            }
            let b = permuted(op, |b, e, lane| {
                let sixteen = b.constant(e, Ty::I32, 16);
                let pick = b.int(e, IntOp::And, lane, sixteen);
                let two = b.constant(e, Ty::I32, 2);
                b.int(e, IntOp::Shl, pick, two)
            });
            if !converted(&b).is_empty() {
                wrong.push(format!("{:?} of lane 0 or 16: {:?}", op, converted(&b)));
            }
        }
        let b = permuted(WaveOp::ReadLane, |b, e, lane| {
            let sixteen = b.constant(e, Ty::I32, 16);
            b.int(e, IntOp::And, lane, sixteen)
        });
        if !converted(&b).is_empty() {
            wrong.push(format!("lane read of lane 0 or 16: {:?}", converted(&b)));
        }
        assert!(wrong.is_empty(), "lanes 0 and 16 hold their lane id whatever the query answers: {:?}", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_word_lane_zero_alone_reads_from_lane_one() {
        let mut wrong = Vec::new();
        for op in [WaveOp::Bpermute, WaveOp::BpermuteFi, WaveOp::ReadLane] {
            let (mut b, k) = Build::kernel();
            let e = BlockId(0);
            let q = flag_query(&mut b, &k, e);
            let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
            let (zero, one) = (b.constant(e, Ty::I32, 0), b.constant(e, Ty::I32, 1));
            let second = b.cmp(e, IntPred::Eq, lane, one);
            let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
            let picked = b.core(e, Ty::I32, Op::Select(q, ones, zero));
            let x = b.core(e, Ty::I32, Op::Select(second, picked, lane));
            let next = b.int(e, IntOp::Xor, lane, one);
            let y = match op {
                WaveOp::ReadLane => b.wave(e, WaveOp::ReadLane, vec![x, next, zero]),
                _ => {
                    let two = b.constant(e, Ty::I32, 2);
                    let byte = b.int(e, IntOp::Shl, next, two);
                    b.wave(e, op, vec![byte, x, k.exec])
                }
            };
            let first = b.cmp(e, IntPred::Eq, lane, zero);
            let mask = b.int(e, IntOp::And, first, k.exec);
            store_own(&mut b, &k, e, y, mask);
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                if kept.queries.is_empty() {
                    wrong.push(format!("{:?} {}", op, name));
                }
            }
        }
        assert!(wrong.is_empty(), "lane 0 stores lane 1's word, which depends on the query: {:?}", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_word_the_lane_a_permute_takes_holds() {
        let mut wrong = Vec::new();
        for op in [WaveOp::Bpermute, WaveOp::BpermuteFi, WaveOp::ReadLane] {
            let b = permuted(op, |b, e, lane| {
                let one = b.constant(e, Ty::I32, 1);
                let next = b.int(e, IntOp::Xor, lane, one);
                if op == WaveOp::ReadLane {
                    next
                } else {
                    let two = b.constant(e, Ty::I32, 2);
                    b.int(e, IntOp::Shl, next, two)
                }
            });
            for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
                if kept.queries.is_empty() {
                    wrong.push(format!("{:?} {}", op, name));
                }
            }
        }
        assert!(wrong.is_empty(), "lane 0 reads lane 1, whose word depends on the query: {:?}", wrong);
    }

    fn apart_merge(reach_store: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let u = uniform_load(&mut b, &k, e, 24);
        let five = b.constant(e, Ty::I32, 5);
        let uniform = b.cmp(e, IntPred::Eq, u, five);
        let flag = per_lane(&mut b, &k, e, 28);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, flag, zero);
        let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (first, x) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
        let (second, y) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
        let (merged, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (arm, vec![k.exec, own, c, uniform]), (join, vec![k.exec]));
        b.cond_br(arm, a[3], (first, vec![a[0], a[1], a[2]]), (second, vec![a[0], a[1], a[2]]));
        b.br(first, merged, vec![x[0], x[1], x[2], x[2]]);
        let never = b.constant(second, Ty::I1, 0);
        b.br(second, merged, vec![y[0], y[1], y[2], never]);
        let mask = if reach_store {
            m[3]
        } else {
            let yes = b.constant(merged, Ty::I1, 1);
            let not_c = b.int(merged, IntOp::Xor, m[2], yes);
            let only = b.int(merged, IntOp::And, m[3], not_c);
            b.int(merged, IntOp::And, only, m[0])
        };
        let one = b.constant(merged, Ty::I32, 1);
        b.store(merged, Space::Global, MemSize::B32, m[1], one, mask);
        b.br(merged, join, vec![m[0]]);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_arm_merges_two_paths_that_never_store() {
        let b = apart_merge(false);
        assert!(converted(&b).is_empty(), "{:?}: z is c or false, so z & !c never holds and the arm never stores", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arm_merges_two_paths_one_of_which_stores() {
        let b = apart_merge(true);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: z is c on the first path, so lanes with c store when the wave takes the arm", wrong);
    }

    fn apart_merge_answer(reach_store: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let u = uniform_load(&mut b, &k, e, 24);
        let five = b.constant(e, Ty::I32, 5);
        let uniform = b.cmp(e, IntPred::Eq, u, five);
        let w = uniform_load(&mut b, &k, e, 32);
        let seven = b.constant(e, Ty::I32, 7);
        let other = b.cmp(e, IntPred::Eq, w, seven);
        let flag = per_lane(&mut b, &k, e, 28);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, flag, zero);
        let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I1]);
        let (first, x) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (second, y) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
        let (merged, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (arm, vec![k.exec, own, c, uniform, other]), (join, vec![k.exec]));
        b.cond_br(arm, a[3], (first, vec![a[0], a[1], a[2], a[4]]), (second, vec![a[0], a[1], a[2]]));
        let r = b.wave(first, WaveOp::Any, vec![x[3]]);
        let z = b.int(first, IntOp::And, x[2], r);
        b.br(first, merged, vec![x[0], x[1], x[2], z]);
        let never = b.constant(second, Ty::I1, 0);
        b.br(second, merged, vec![y[0], y[1], y[2], never]);
        let mask = if reach_store {
            b.int(merged, IntOp::And, m[3], m[0])
        } else {
            let yes = b.constant(merged, Ty::I1, 1);
            let not_c = b.int(merged, IntOp::Xor, m[2], yes);
            let only = b.int(merged, IntOp::And, m[3], not_c);
            b.int(merged, IntOp::And, only, m[0])
        };
        let one = b.constant(merged, Ty::I32, 1);
        b.store(merged, Space::Global, MemSize::B32, m[1], one, mask);
        b.br(merged, join, vec![m[0]]);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_arm_merges_an_answer_and_a_path_that_never_store() {
        let b = apart_merge_answer(false);
        assert!(converted(&b).is_empty(), "{:?}: z is c & any(v) or false, so z & !c never holds and the arm never stores", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arm_merges_an_answer_that_stores_and_another_path() {
        let b = apart_merge_answer(true);
        assert!(keeps(&b).is_empty(), "{:?}: z is c & any(v) on the first path, so lanes with c store when the wave takes the arm and v holds", keeps(&b));
    }

    fn with_explore(test: impl FnOnce(&mut Explore)) {
        let (b, _) = Build::kernel();
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let loops = Loops::new(f, &facts).unwrap();
        let hazards = no_hazards();
        let logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let mut check = Check::new(f, &facts, &b.inputs, Some(0), &loops, &hazards, logic);
        let mut explore = Explore::new(&mut check, f.entry, Bdd::TRUE);
        test(&mut explore);
    }

    fn evaluate(m: &Manager, mut g: Bdd, value: &dyn Fn(u32) -> bool) -> bool {
        while let Some((var, low, high)) = m.decompose(g) {
            g = if value(var) { high } else { low };
        }
        g == Bdd::TRUE
    }

    fn variable(logic: &mut Logic, atom: Atom) -> u32 {
        let g = logic.atom(atom);
        logic.m.decompose(g).unwrap().0
    }

    fn random_function(logic: &mut Logic, r: &mut Random, pool: &[Atom], most: usize) -> Bdd {
        let mut atoms: Vec<Atom> = Vec::new();
        for _ in 0..1 + r.below(most as u64) {
            let a = pool[r.below(pool.len() as u64) as usize];
            if !atoms.contains(&a) {
                atoms.push(a);
            }
        }
        let vars: Vec<Bdd> = atoms.iter().map(|&a| logic.atom(a)).collect();
        let mut g = Bdd::FALSE;
        for row in 0..1u32 << vars.len() {
            if r.below(2) == 0 {
                continue;
            }
            let mut minterm = Bdd::TRUE;
            for (i, &v) in vars.iter().enumerate() {
                let literal = if row >> i & 1 == 1 { v } else { logic.m.not(v) };
                minterm = logic.m.and(minterm, literal);
            }
            g = logic.m.or(g, minterm);
        }
        g
    }

    fn random_state(logic: &mut Logic, r: &mut Random, context: &[Atom], unknowns: &[Atom], paths: &[Atom]) -> (Bdd, [Vec<Desc>; 2]) {
        let tied: Vec<Atom> = context.iter().chain(paths).copied().collect();
        let all: Vec<Atom> = tied.iter().chain(unknowns).copied().collect();
        let cond = random_function(logic, r, &tied, 3);
        let desc = |logic: &mut Logic, r: &mut Random| Desc {
            same: None,
            bits: Some(random_function(logic, r, &all, 4)),
        };
        let wave = vec![desc(logic, r), desc(logic, r)];
        let lane = vec![desc(logic, r)];
        (cond, [wave, lane])
    }

    fn held(logic: &mut Logic, (cond, sides): (Bdd, &[Vec<Desc>; 2]), context: &[(u32, bool)]) -> BTreeSet<Vec<bool>> {
        let roots: Vec<Bdd> = sides.iter().flatten().filter_map(|d| d.bits).collect();
        let mut hidden: Vec<u32> = Vec::new();
        for &g in roots.iter().chain([cond].iter()) {
            for &v in logic.support(g).iter() {
                if matches!(logic.atom_of(v), Atom::Fresh(..)) && !hidden.contains(&v) {
                    hidden.push(v);
                }
            }
        }
        let mut out = BTreeSet::new();
        for row in 0..1u64 << hidden.len() {
            let value = |var: u32| match hidden.iter().position(|&h| h == var) {
                Some(i) => row >> i & 1 == 1,
                None => context.iter().find(|&&(v, _)| v == var).unwrap().1,
            };
            if evaluate(&logic.m, cond, &value) {
                out.insert(roots.iter().map(|&g| evaluate(&logic.m, g, &value)).collect());
            }
        }
        out
    }

    fn contexts(logic: &mut Logic, context: &[Atom]) -> Vec<Vec<(u32, bool)>> {
        let vars: Vec<u32> = context.iter().map(|&a| variable(logic, a)).collect();
        (0..1u32 << vars.len())
            .map(|row| vars.iter().enumerate().map(|(i, &v)| (v, row >> i & 1 == 1)).collect())
            .collect()
    }

    fn computed(e: &Explore, t: usize, leaves: &[u64]) -> u64 {
        let mask = u32::MAX as u64;
        match &e.terms[t] {
            Form::Opaque(i, _) => leaves[*i - 1000],
            Form::Core(_, Op::Const(_, k)) => *k & mask,
            Form::Core(_, Op::Int(op, a, b)) => {
                let (x, y) = (computed(e, a.0, leaves), computed(e, b.0, leaves));
                (match op {
                    IntOp::Add => x.wrapping_add(y),
                    IntOp::Sub => x.wrapping_sub(y),
                    IntOp::Mul => x.wrapping_mul(y),
                    IntOp::And => x & y,
                    IntOp::Or => x | y,
                    IntOp::Xor => x ^ y,
                    _ => panic!("no {:?} in these terms", op),
                }) & mask
            }
            Form::Linear(_, parts, k) => parts.iter().fold(*k, |acc, &(p, c)| acc.wrapping_add(c.wrapping_mul(computed(e, p, leaves)))) & mask,
            other => panic!("no {:?} in these terms", std::mem::discriminant(other)),
        }
    }

    fn random_term(e: &mut Explore, r: &mut Random, depth: usize, leaves: &[usize]) -> (usize, Box<dyn Fn(&[u64]) -> u64>) {
        if depth == 0 || r.below(4) == 0 {
            if r.below(3) == 0 {
                let k = [0u64, 1, 2, 3, 5, 0xff, 0xffff_ffff, 0x8000_0000][r.below(8) as usize];
                return (e.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, k))), Box::new(move |_| k));
            }
            let i = r.below(leaves.len() as u64) as usize;
            return (leaves[i], Box::new(move |v| v[i]));
        }
        let ops = [IntOp::Add, IntOp::Sub, IntOp::Mul, IntOp::And, IntOp::Or, IntOp::Xor];
        let op = ops[r.below(ops.len() as u64) as usize];
        let (a, fa) = random_term(e, r, depth - 1, leaves);
        let (b, fb) = random_term(e, r, depth - 1, leaves);
        let t = e.intern(Ty::I32, Form::Core(Ty::I32, Op::Int(op, ValueId(a), ValueId(b))));
        let mask = u32::MAX as u64;
        (
            t,
            Box::new(move |v| {
                let (x, y) = (fa(v), fb(v));
                (match op {
                    IntOp::Add => x.wrapping_add(y),
                    IntOp::Sub => x.wrapping_sub(y),
                    IntOp::Mul => x.wrapping_mul(y),
                    IntOp::And => x & y,
                    IntOp::Or => x | y,
                    _ => x ^ y,
                }) & mask
            }),
        )
    }

    #[test]
    fn interned_words_compute_what_their_operations_compute() {
        with_explore(|e| {
            let leaves: Vec<usize> = (0..3).map(|i| e.intern(Ty::I32, Form::Opaque(1000 + i, ValueId(0)))).collect();
            let mut r = Random::new(113);
            let mut wrong = Vec::new();
            for trial in 0..400 {
                let (t, truth) = random_term(e, &mut r, 4, &leaves);
                for _ in 0..8 {
                    let values: Vec<u64> = (0..3)
                        .map(|_| match r.below(3) {
                            0 => r.below(8),
                            1 => (r.next() as u32) as u64,
                            _ => [0xffff_ffffu64, 0x8000_0000, 0x7fff_ffff][r.below(3) as usize],
                        })
                        .collect();
                    if computed(e, t, &values) != truth(&values) {
                        wrong.push(format!("trial {} at {:?}: {:#x} not {:#x}", trial, values, computed(e, t, &values), truth(&values)));
                    }
                }
            }
            assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(8)]);
        });
    }

    #[test]
    fn interned_words_are_one_term_for_regrouped_distributed_and_reordered_operations() {
        with_explore(|e| {
            let [x, y, z]: [usize; 3] = std::array::from_fn(|i| e.intern(Ty::I32, Form::Opaque(1000 + i, ValueId(0))));
            let k = |e: &mut Explore, c: u64| e.intern(Ty::I32, Form::Core(Ty::I32, Op::Const(Ty::I32, c)));
            let op = |e: &mut Explore, o: IntOp, a: usize, b: usize| e.intern(Ty::I32, Form::Core(Ty::I32, Op::Int(o, ValueId(a), ValueId(b))));
            use IntOp::*;
            let mut loose = Vec::new();
            let mut same = |name: &str, a: usize, b: usize| {
                if a != b {
                    loose.push(name.to_string());
                }
            };
            let (xy, yz, zx) = (op(e, Mul, x, y), op(e, Mul, y, z), op(e, Mul, z, x));
            let (a, b, c) = (op(e, Mul, xy, z), op(e, Mul, x, yz), op(e, Mul, zx, y));
            same("(x y) z and x (y z)", a, b);
            same("(x y) z and (z x) y", a, c);
            let sum = op(e, Add, x, y);
            let (xz, yz2) = (op(e, Mul, x, z), op(e, Mul, y, z));
            let (a, b) = (op(e, Mul, sum, z), op(e, Add, xz, yz2));
            same("(x + y) z and x z + y z", a, b);
            let one = k(e, 1);
            let (up, down) = (op(e, Add, x, one), op(e, Sub, x, one));
            let xx = op(e, Mul, x, x);
            let (a, b) = (op(e, Mul, up, down), op(e, Sub, xx, one));
            same("(x + 1)(x - 1) and x x - 1", a, b);
            let (xy, yz) = (op(e, And, x, y), op(e, And, y, z));
            let (a, b) = (op(e, And, xy, z), op(e, And, x, yz));
            same("(x & y) & z and x & (y & z)", a, b);
            let zx = op(e, And, z, x);
            let c = op(e, And, zx, y);
            same("(x & y) & z and (z & x) & y", a, c);
            let (xo, xa) = (op(e, Or, x, x), op(e, And, x, x));
            same("x | x and x", xo, x);
            same("x & x and x", xa, x);
            let zero = k(e, 0);
            let xx = op(e, Xor, x, x);
            same("x ^ x and 0", xx, zero);
            let xy = op(e, Xor, x, y);
            let xyx = op(e, Xor, xy, x);
            same("(x ^ y) ^ x and y", xyx, y);
            let ones = k(e, 0xffff_ffff);
            let a = op(e, And, x, ones);
            same("x & ~0 and x", a, x);
            let (three, five) = (k(e, 3), k(e, 5));
            let x3 = op(e, And, x, three);
            let a = op(e, And, x3, five);
            let b = op(e, And, x, one);
            same("(x & 3) & 5 and x & 1", a, b);
            let x3 = op(e, Or, x, three);
            let a = op(e, Or, x3, five);
            let seven = k(e, 7);
            let b = op(e, Or, seven, x);
            same("(x | 3) | 5 and 7 | x", a, b);
            let a = op(e, Or, x, ones);
            same("x | ~0 and ~0", a, ones);
            assert!(loose.is_empty(), "each pair computes the same word: {:?}", loose);
        });
    }

    #[test]
    fn join_holds_exactly_the_states_of_every_arrival() {
        with_explore(|e| {
            let mut r = Random::new(43);
            let context = [Atom::Lane(0), Atom::Lane(1)];
            let unknowns = [Atom::Fresh(WAVE, ValueId(1000), 0), Atom::Fresh(LANE, ValueId(1001), 0), Atom::Fresh(JOINT, ValueId(0), 1)];
            e.paths = PATHS - 8;
            let paths = [Atom::Fresh(PATH, ValueId(0), PATHS - 1), Atom::Fresh(PATH, ValueId(0), PATHS - 2), Atom::Fresh(PATH, ValueId(0), PATHS)];
            let contexts = contexts(e.logic(), &context);
            let mut wrong = Vec::new();
            for trial in 0..300 {
                let mut arrivals = vec![random_state(e.logic(), &mut r, &context, &unknowns, &paths)];
                for _ in 0..2 {
                    let mut next = random_state(e.logic(), &mut r, &context, &unknowns, &paths);
                    let first = arrivals[0].1.clone();
                    match r.below(4) {
                        0 => next.1 = first,
                        1 => next.1[WAVE][0] = first[WAVE][0],
                        _ => {}
                    }
                    arrivals.push(next);
                }
                let two = e.join((arrivals[0].0, &arrivals[0].1), (arrivals[1].0, &arrivals[1].1));
                let entries: BTreeMap<(Key, usize), (Bdd, [Vec<Desc>; 2])> =
                    arrivals.iter().enumerate().map(|(i, a)| (([None, None], i), a.clone())).collect();
                let three = e.joined(&entries);
                for context in &contexts {
                    let each: Vec<BTreeSet<Vec<bool>>> = arrivals.iter().map(|a| held(e.logic(), (a.0, &a.1), context)).collect();
                    let expected_two: BTreeSet<Vec<bool>> = each[0].union(&each[1]).cloned().collect();
                    let expected_three: BTreeSet<Vec<bool>> = expected_two.union(&each[2]).cloned().collect();
                    if held(e.logic(), (two.0, &two.1), context) != expected_two {
                        wrong.push(format!("trial {} two arrivals under {:?}", trial, context));
                    }
                    if held(e.logic(), (three.0, &three.1), context) != expected_three {
                        wrong.push(format!("trial {} three arrivals under {:?}", trial, context));
                    }
                }
            }
            assert!(wrong.is_empty(), "the join must hold the states of the arrivals and nothing else: {:?}", &wrong[..wrong.len().min(8)]);
        });
    }

    #[test]
    fn canonical_names_hold_the_same_states_in_normal_form() {
        with_explore(|e| {
            let mut r = Random::new(47);
            let context = [Atom::Lane(0), Atom::Lane(1)];
            let unknowns = [
                Atom::Fresh(WAVE, ValueId(1000), 0),
                Atom::Fresh(LANE, ValueId(1001), 0),
                Atom::Fresh(WAVE, ValueId(1002), u32::MAX),
                Atom::Fresh(JOINT, ValueId(0), 3),
                Atom::Fresh(JOINT, ValueId(0), 1),
            ];
            let paths = [Atom::Fresh(PATH, ValueId(0), PATHS - 1), Atom::Fresh(PATH, ValueId(0), PATHS + 2), Atom::Fresh(PATH, ValueId(0), 5)];
            for position in 1..=unknowns.len() as u32 {
                e.fresh(JOINT, ValueId(0), position);
            }
            let contexts = contexts(e.logic(), &context);
            let mut wrong = Vec::new();
            for trial in 0..300 {
                let (cond, sides) = random_state(e.logic(), &mut r, &context, &unknowns, &paths);
                let (cc, cs) = e.canonical(cond, sides.clone());
                for context in &contexts {
                    if held(e.logic(), (cc, &cs), context) != held(e.logic(), (cond, &sides), context) {
                        wrong.push(format!("trial {} changes the states under {:?}", trial, context));
                    }
                }
                let mut names: Vec<Atom> = Vec::new();
                for g in cs.iter().flatten().filter_map(|d| d.bits).chain([cc]) {
                    for v in e.fresh_support(g) {
                        let a = e.check.logic.atom_of(v);
                        if !names.contains(&a) {
                            names.push(a);
                        }
                    }
                }
                let values = names.iter().filter(|a| matches!(a, Atom::Fresh(JOINT, ..))).count() as u32;
                let tied = names.iter().filter(|a| matches!(a, Atom::Fresh(PATH, ..))).count() as u32;
                let normal = names.iter().all(|a| match *a {
                    Atom::Fresh(JOINT, ValueId(0), p) => (1..=values).contains(&p),
                    Atom::Fresh(PATH, ValueId(0), p) => (PATHS..PATHS + tied).contains(&p),
                    _ => false,
                });
                if !normal {
                    wrong.push(format!("trial {} leaves names {:?}", trial, names));
                }
                if e.canonical(cc, cs.clone()) != (cc, cs) {
                    wrong.push(format!("trial {} is not settled by one pass", trial));
                }
            }
            assert!(wrong.is_empty(), "renaming must keep the states and settle the names: {:?}", &wrong[..wrong.len().min(8)]);
        });
    }

    #[test]
    fn decide_and_weaken_answer_over_the_paths_the_condition_allows() {
        with_explore(|e| {
            let mut r = Random::new(53);
            let context = [Atom::Lane(0), Atom::Lane(1)];
            let unknowns = [Atom::Fresh(LANE, ValueId(1001), 0), Atom::Fresh(JOINT, ValueId(0), 1)];
            let paths = [Atom::Fresh(PATH, ValueId(0), PATHS - 1), Atom::Fresh(PATH, ValueId(0), PATHS)];
            let tied: Vec<Atom> = context.iter().chain(&paths).copied().collect();
            let all: Vec<Atom> = tied.iter().chain(&unknowns).copied().collect();
            let tied_vars: Vec<u32> = tied.iter().map(|&a| variable(e.logic(), a)).collect();
            let unknown_vars: Vec<u32> = unknowns.iter().map(|&a| variable(e.logic(), a)).collect();
            let mut wrong = Vec::new();
            for trial in 0..400 {
                let cond = random_function(e.logic(), &mut r, &tied, 3);
                if cond == Bdd::FALSE {
                    continue;
                }
                let g = random_function(e.logic(), &mut r, &all, 4);
                let (mut every, mut none) = (true, true);
                let mut expected_weak = Vec::new();
                for row in 0..1u32 << tied_vars.len() {
                    let mut some = false;
                    for hidden in 0..1u32 << unknown_vars.len() {
                        let value = |var: u32| match tied_vars.iter().position(|&v| v == var) {
                            Some(i) => row >> i & 1 == 1,
                            None => hidden >> unknown_vars.iter().position(|&v| v == var).unwrap() & 1 == 1,
                        };
                        let holds = evaluate(&e.check.logic.m, g, &value);
                        some |= holds;
                        if evaluate(&e.check.logic.m, cond, &value) {
                            every &= holds;
                            none &= !holds;
                        }
                    }
                    expected_weak.push(some);
                }
                let expected = if every { Some(true) } else if none { Some(false) } else { None };
                if e.decide(g, cond, LANE) != expected {
                    wrong.push(format!("trial {} decides {:?}, expected {:?}", trial, e.decide(g, cond, LANE), expected));
                }
                let weak = e.weaken(g, LANE);
                for (row, &some) in expected_weak.iter().enumerate() {
                    let value = |var: u32| row >> tied_vars.iter().position(|&v| v == var).unwrap() & 1 == 1;
                    if evaluate(&e.check.logic.m, weak, &value) != some {
                        wrong.push(format!("trial {} weakens to {} at {}, expected {}", trial, !some, row, some));
                    }
                }
            }
            assert!(wrong.is_empty(), "decide and weaken must quantify the unknowns and keep the paths: {:?}", &wrong[..wrong.len().min(8)]);
        });
    }

    fn apart_merge_pair(differ: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let u = uniform_load(&mut b, &k, e, 24);
        let five = b.constant(e, Ty::I32, 5);
        let uniform = b.cmp(e, IntPred::Eq, u, five);
        let flag = per_lane(&mut b, &k, e, 28);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, flag, zero);
        let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (first, x) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
        let (second, y) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
        let (merged, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (arm, vec![k.exec, own, c, uniform]), (join, vec![k.exec]));
        b.cond_br(arm, a[3], (first, vec![a[0], a[1], a[2]]), (second, vec![a[0], a[1], a[2]]));
        b.br(first, merged, vec![x[0], x[1], x[2], x[2]]);
        let yes = b.constant(second, Ty::I1, 1);
        let not_c = b.int(second, IntOp::Xor, y[2], yes);
        let other = if differ { y[2] } else { not_c };
        b.br(second, merged, vec![y[0], y[1], not_c, other]);
        let yes = b.constant(merged, Ty::I1, 1);
        let not_other = b.int(merged, IntOp::Xor, m[3], yes);
        let only = b.int(merged, IntOp::And, m[2], not_other);
        let mask = b.int(merged, IntOp::And, only, m[0]);
        let one = b.constant(merged, Ty::I32, 1);
        b.store(merged, Space::Global, MemSize::B32, m[1], one, mask);
        b.br(merged, join, vec![m[0]]);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_arm_merges_two_bits_that_every_path_keeps_equal() {
        let b = apart_merge_pair(false);
        assert!(converted(&b).is_empty(), "{:?}: the two bits are (c, c) on one path and (!c, !c) on the other, so s & !t never holds and the arm never stores", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arm_merges_two_bits_that_one_path_sets_apart() {
        let b = apart_merge_pair(true);
        assert!(keeps(&b).is_empty(), "{:?}: the two bits are (!c, c) on the second path, so lanes without c store when the wave takes it", keeps(&b));
    }

    fn apart_merge_branch(reach_store: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let u = uniform_load(&mut b, &k, e, 24);
        let five = b.constant(e, Ty::I32, 5);
        let uniform = b.cmp(e, IntPred::Eq, u, five);
        let flag = per_lane(&mut b, &k, e, 28);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, flag, zero);
        let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (first, x) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (second, y) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (merged, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (arm, vec![k.exec, own, c, uniform]), (join, vec![k.exec]));
        b.cond_br(arm, a[3], (first, vec![a[0], a[1], a[2], a[3]]), (second, vec![a[0], a[1], a[2], a[3]]));
        b.br(first, merged, vec![x[0], x[1], x[2], x[3]]);
        let never = b.constant(second, Ty::I1, 0);
        b.br(second, merged, vec![y[0], y[1], never, y[3]]);
        let branch = if reach_store {
            m[3]
        } else {
            let yes = b.constant(merged, Ty::I1, 1);
            b.int(merged, IntOp::Xor, m[3], yes)
        };
        let only = b.int(merged, IntOp::And, m[2], branch);
        let mask = b.int(merged, IntOp::And, only, m[0]);
        let one = b.constant(merged, Ty::I32, 1);
        b.store(merged, Space::Global, MemSize::B32, m[1], one, mask);
        b.br(merged, join, vec![m[0]]);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_arm_merges_a_bit_only_the_branch_to_it_sets() {
        let b = apart_merge_branch(false);
        assert!(converted(&b).is_empty(), "{:?}: z is c on the path taken when u holds and false on the other, so z & !u never holds and the arm never stores", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arm_merges_a_bit_that_stores_where_the_branch_to_it_holds() {
        let b = apart_merge_branch(true);
        assert!(keeps(&b).is_empty(), "{:?}: z is c on the path taken when u holds, so lanes with c store when the wave takes the arm and u holds", keeps(&b));
    }

    fn apart_lane_mask(contradiction: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (then, vec![k.exec, own]), (join, vec![k.exec]));
        let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
        let sixteen = b.constant(then, Ty::I32, 16);
        let low = b.cmp(then, IntPred::Ult, lane, sixteen);
        let mask = if contradiction {
            let high = b.cmp(then, IntPred::Uge, lane, sixteen);
            b.int(then, IntOp::And, low, high)
        } else {
            low
        };
        let mask = b.int(then, IntOp::And, mask, t[0]);
        let one = b.constant(then, Ty::I32, 1);
        b.store(then, Space::Global, MemSize::B32, t[1], one, mask);
        b.br(then, join, vec![t[0]]);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_arm_stores_under_a_lane_contradiction() {
        let b = apart_lane_mask(true);
        assert!(converted(&b).is_empty(), "{:?}: lane < 16 and lane >= 16 never hold together, so the arm never stores", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arm_stores_in_the_low_lanes() {
        let b = apart_lane_mask(false);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: lanes 0 to 15 store when the wave takes the arm", wrong);
    }

    fn lane_query(target: u64) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let target = b.constant(e, Ty::I32, target);
        let only = b.cmp(e, IntPred::Eq, lane, target);
        let c = b.int(e, IntOp::And, only, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let data = b.core(e, Ty::I32, Op::Select(q, one, two));
        store_own(&mut b, &k, e, data, k.exec);
        b
    }

    fn lane_bits_query(target: u64) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let one = b.constant(e, Ty::I32, 1);
        let odd = b.int(e, IntOp::And, lane, one);
        let target = b.constant(e, Ty::I32, target);
        let hit = b.cmp(e, IntPred::Eq, odd, target);
        let c = b.int(e, IntOp::And, hit, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let two = b.constant(e, Ty::I32, 2);
        let data = b.core(e, Ty::I32, Op::Select(q, one, two));
        store_own(&mut b, &k, e, data, k.exec);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_lane_bit_never_matches() {
        let b = lane_bits_query(2);
        assert!(converted(&b).is_empty(), "{:?}: lane & 1 is 0 or 1, never 2", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_lane_bit_matches_in_odd_lanes() {
        let b = lane_bits_query(1);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: odd lanes answer true, even lanes alone answer false", wrong);
    }

    #[test]
    fn prove_converts_a_query_no_lane_can_answer_true() {
        let b = lane_query(99);
        assert!(converted(&b).is_empty(), "{:?}: no lane is lane 99, so the query is false in the wave and in every lane", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_one_lane_can_answer_true() {
        let b = lane_query(5);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: when lane 5 is active the wave answers true, but every other lane alone answers false", wrong);
    }

    #[test]
    fn prove_converts_a_query_whose_arms_store_the_same_word() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flag = per_lane(&mut b, &k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (then, vec![k.exec, own]), (other, vec![k.exec, own]));
        for (block, p) in [(then, t), (other, o)] {
            let one = b.constant(block, Ty::I32, 1);
            b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
            b.br(block, join, vec![p[0]]);
        }
        assert!(converted(&b).is_empty(), "{:?}: both arms store 1 to the lane's own word", converted(&b));
    }

    fn arms_store(arm: impl Fn(&mut Build, BlockId, usize, &[ValueId]) -> Option<ValueId>) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flag = per_lane(&mut b, &k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        let (join, j) = b.block(&[Ty::I1, Ty::I32]);
        b.cond_br(e, q, (then, vec![k.exec, own, lane]), (other, vec![k.exec, own, lane]));
        for (side, (block, p)) in [(then, t), (other, o)].iter().enumerate() {
            let carried = arm(&mut b, *block, side, p).unwrap_or(p[2]);
            b.br(*block, join, vec![p[0], carried]);
        }
        let out = k.buffer(&mut b, join, 16);
        let target = byte_offset(&mut b, join, out, j[1], 4);
        let one = b.constant(join, Ty::I32, 1);
        b.store(join, Space::Global, MemSize::B32, target, one, j[0]);
        b
    }

    fn keeps(b: &Build) -> Vec<&'static str> {
        ["search", "direct"].iter().zip(both(b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect()
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_store_different_values() {
        let b = arms_store(|b, block, side, p| {
            let value = b.constant(block, Ty::I32, 1 + side as u64);
            b.store(block, Space::Global, MemSize::B32, p[1], value, p[0]);
            None
        });
        assert!(keeps(&b).is_empty(), "{:?}: one arm stores 1 and the other 2", keeps(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_store_to_different_words() {
        let b = arms_store(|b, block, side, p| {
            let at = b.constant(block, Ty::I64, 4 * side as u64);
            let address = b.int(block, IntOp::Add, p[1], at);
            let one = b.constant(block, Ty::I32, 1);
            b.store(block, Space::Global, MemSize::B32, address, one, p[0]);
            None
        });
        assert!(keeps(&b).is_empty(), "{:?}: one arm stores to the lane's word and the other to the next", keeps(&b));
    }

    #[test]
    fn prove_keeps_a_query_only_one_of_whose_arms_stores() {
        let b = arms_store(|b, block, side, p| {
            if side == 0 {
                let one = b.constant(block, Ty::I32, 1);
                b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
            }
            None
        });
        assert!(keeps(&b).is_empty(), "{:?}: only one arm stores", keeps(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_store_under_different_masks() {
        let b = arms_store(|b, block, side, p| {
            let mask = if side == 0 {
                p[0]
            } else {
                let sixteen = b.constant(block, Ty::I32, 16);
                let low = b.cmp(block, IntPred::Ult, p[2], sixteen);
                b.int(block, IntOp::And, low, p[0])
            };
            let one = b.constant(block, Ty::I32, 1);
            b.store(block, Space::Global, MemSize::B32, p[1], one, mask);
            None
        });
        assert!(keeps(&b).is_empty(), "{:?}: lanes 16 to 31 store in one arm only", keeps(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_load_the_stored_word_at_different_times() {
        let b = arms_store(|b, block, side, p| {
            let one = b.constant(block, Ty::I32, 1);
            if side == 0 {
                b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
                Some(b.load(block, Space::Global, MemSize::B32, p[1], p[0]))
            } else {
                let old = b.load(block, Space::Global, MemSize::B32, p[1], p[0]);
                b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
                Some(old)
            }
        });
        assert!(keeps(&b).is_empty(), "{:?}: one arm reads back 1 and the other the word before the store", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_arms_store_one_word_computed_in_each_arm() {
        let b = arms_store(|b, block, _, p| {
            let zero = b.constant(block, Ty::I64, 0);
            let address = b.int(block, IntOp::Add, p[1], zero);
            let one = b.constant(block, Ty::I32, 1);
            b.store(block, Space::Global, MemSize::B32, address, one, p[0]);
            None
        });
        assert!(converted(&b).is_empty(), "{:?}: both arms store 1 to the lane's own word", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_add_different_amounts_atomically() {
        let b = arms_store(|b, block, side, p| {
            let amount = b.constant(block, Ty::I32, 1 + side as u64);
            b.effect(block, memory(Space::Global, MemoryOp::AtomicAdd(Numeric::Unsigned)), vec![p[1], amount, p[0]]);
            None
        });
        assert!(keeps(&b).is_empty(), "{:?}: one arm adds 1 and the other 2", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_arms_add_the_same_amount_atomically() {
        let b = arms_store(|b, block, _, p| {
            let one = b.constant(block, Ty::I32, 1);
            b.effect(block, memory(Space::Global, MemoryOp::AtomicAdd(Numeric::Unsigned)), vec![p[1], one, p[0]]);
            None
        });
        assert!(converted(&b).is_empty(), "{:?}: both arms add 1 to the lane's own word", converted(&b));
    }

    fn arms_store_for_neighbours(second: u64) -> Vec<&'static str> {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flag = per_lane(&mut b, &k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let one = b.constant(e, Ty::I32, 1);
        let next = b.int(e, IntOp::Xor, lane, one);
        let neighbour = byte_offset(&mut b, e, buf, next, 4);
        let out = k.buffer(&mut b, e, 16);
        let slot = byte_offset(&mut b, e, out, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I64, Ty::I64]);
        let (join, j) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let args = vec![k.exec, own, neighbour, slot];
        b.cond_br(e, q, (then, args.clone()), (other, args));
        let mut stores = Vec::new();
        for (side, (block, p)) in vec![(then, t), (other, o)].into_iter().enumerate() {
            let value = b.constant(block, Ty::I32, if side == 0 { 1 } else { second });
            stores.push(b.here(block));
            b.store(block, Space::Global, MemSize::B32, p[1], value, p[0]);
            b.br(block, join, vec![p[0], p[2], p[3]]);
        }
        let l = b.here(join);
        let read = b.load(join, Space::Global, MemSize::B32, j[1], j[0]);
        b.store(join, Space::Global, MemSize::B32, j[2], read, j[0]);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(stores[0], l), (stores[1], l)], &[], &[]);
        let (f, inputs) = (&program.ir, &program.parameter_inputs);
        vec![("search", search::prove(f, inputs, Some(0), &hazards).0), ("direct", direct::prove(f, inputs, Some(0), &hazards).0)]
            .into_iter()
            .filter(|(_, kept)| kept.queries.contains(&q) == (second == 1))
            .map(|(name, _)| name)
            .collect()
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_store_the_same_words_a_neighbour_overwrites() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flag = per_lane(&mut b, &k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let one = b.constant(e, Ty::I32, 1);
        let next = b.int(e, IntOp::Xor, lane, one);
        let neighbour = byte_offset(&mut b, e, buf, next, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (join, _) = b.block(&[Ty::I1]);
        let args = vec![k.exec, own, neighbour];
        b.cond_br(e, q, (then, args.clone()), (other, args));
        let mut stores = Vec::new();
        for (block, p) in vec![(then, t), (other, o)] {
            let first = b.constant(block, Ty::I32, 1);
            let second = b.constant(block, Ty::I32, 2);
            let s1 = b.here(block);
            b.store(block, Space::Global, MemSize::B32, p[1], first, p[0]);
            let s2 = b.here(block);
            b.store(block, Space::Global, MemSize::B32, p[2], second, p[0]);
            stores.push((s1, s2));
            b.br(block, join, vec![p[0]]);
        }
        let program = b.program();
        let pairs = [(stores[0].0, stores[0].1), (stores[1].0, stores[1].1), (stores[0].0, stores[1].1), (stores[1].0, stores[0].1)];
        let hazards = Hazards::given(&program, &pairs, &[], &[]);
        let (f, inputs) = (&program.ir, &program.parameter_inputs);
        let wrong: Vec<&str> = vec![("search", search::prove(f, inputs, Some(0), &hazards).0), ("direct", direct::prove(f, inputs, Some(0), &hazards).0)]
            .into_iter()
            .filter(|(_, kept)| !kept.queries.contains(&q))
            .map(|(name, _)| name)
            .collect();
        assert!(wrong.is_empty(), "{:?}: a lane in one arm may write 2 into its neighbour's word before the neighbour, in the other arm, writes 1 into it", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_store_different_values_a_neighbour_reads() {
        let wrong = arms_store_for_neighbours(2);
        assert!(wrong.is_empty(), "{:?}: one arm stores 1 and the other 2, which the neighbour reads back", wrong);
    }

    #[test]
    fn prove_converts_a_query_whose_arms_store_the_same_word_a_neighbour_reads() {
        let wrong = arms_store_for_neighbours(1);
        assert!(wrong.is_empty(), "{:?}: both arms store 1 to the lane's own word, and a meeting before the neighbour's read orders either", wrong);
    }

    fn bound_into_a_difference(inside: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let x = per_lane(&mut b, &k, e, 16);
        let five = b.constant(e, Ty::I32, 5);
        let small = b.cmp(e, IntPred::Ult, x, five);
        let three = b.constant(e, Ty::I32, 3);
        let v = b.core(e, Ty::I32, Op::Select(small, data, three));
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
        b.br(e, next, vec![k.exec, own, x, v]);
        let five = b.constant(next, Ty::I32, 5);
        let test = b.cmp(next, if inside { IntPred::Ult } else { IntPred::Uge }, p[2], five);
        let mask = b.int(next, IntOp::And, test, p[0]);
        b.store(next, Space::Global, MemSize::B32, p[1], p[3], mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_word_a_bound_carried_into_the_next_block_lets_through() {
        let b = bound_into_a_difference(true);
        assert!(keeps(&b).is_empty(), "{:?}: lanes with x < 5 store the answer", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_word_a_bound_carried_into_the_next_block_rules_out() {
        let b = bound_into_a_difference(false);
        assert!(converted(&b).is_empty(), "{:?}: lanes with x >= 5 store 3, which the query never reaches", converted(&b));
    }

    fn arms_store_two(order: bool, second: u64) -> Build {
        arms_store(move |b, block, side, p| {
            let past = b.constant(block, Ty::I64, 128);
            let next = b.int(block, IntOp::Add, p[1], past);
            let one = b.constant(block, Ty::I32, 1);
            let other = b.constant(block, Ty::I32, if side == 0 { 2 } else { second });
            if side == 0 || !order {
                b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
                b.store(block, Space::Global, MemSize::B32, next, other, p[0]);
            } else {
                b.store(block, Space::Global, MemSize::B32, next, other, p[0]);
                b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
            }
            None
        })
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_store_a_second_word_differently() {
        let b = arms_store_two(true, 3);
        assert!(keeps(&b).is_empty(), "{:?}: one arm stores 2 into the word 32 further and the other 3", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_arms_store_two_words_in_either_order() {
        let b = arms_store_two(true, 2);
        assert!(converted(&b).is_empty(), "{:?}: both arms store 1 into the lane's word and 2 into the word 32 further, which no other lane of the wave touches", converted(&b));
    }

    fn ordered_three(closed: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let x = per_lane(&mut b, &k, e, 16);
        let y = per_lane(&mut b, &k, e, 24);
        let z = per_lane(&mut b, &k, e, 32);
        let xy = b.cmp(e, IntPred::Ult, x, y);
        let yz = b.cmp(e, IntPred::Ult, y, z);
        let third = if closed { b.cmp(e, IntPred::Ult, z, x) } else { b.cmp(e, IntPred::Ult, x, z) };
        let both = b.int(e, IntOp::And, xy, yz);
        let all = b.int(e, IntOp::And, both, third);
        let mask = b.int(e, IntOp::And, all, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_three_ordered_words_mask() {
        let b = ordered_three(false);
        assert!(keeps(&b).is_empty(), "{:?}: x < y < z holds for x = 0, y = 1, z = 2", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_a_cycle_of_orders_masks() {
        let b = ordered_three(true);
        assert!(converted(&b).is_empty(), "{:?}: x < y < z < x never holds", converted(&b));
    }

    fn signed_and_unsigned(below: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let x = per_lane(&mut b, &k, e, 16);
        let zero = b.constant(e, Ty::I32, 0);
        let ten = b.constant(e, Ty::I32, 10);
        let negative = b.cmp(e, IntPred::Slt, x, zero);
        let small = b.cmp(e, if below { IntPred::Ult } else { IntPred::Uge }, x, ten);
        let both = b.int(e, IntOp::And, negative, small);
        let mask = b.int(e, IntOp::And, both, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_a_negative_word_at_least_ten_unsigned_masks() {
        let b = signed_and_unsigned(false);
        assert!(keeps(&b).is_empty(), "{:?}: -1 is below 0 signed and at least 10 unsigned", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_a_negative_word_below_ten_unsigned_masks() {
        let b = signed_and_unsigned(true);
        assert!(converted(&b).is_empty(), "{:?}: a word below 0 signed is at least 2^31 unsigned", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_store_a_bound_that_admits_the_constant_lets_through() {
        let b = branch_then_store((IntPred::Ult, 5), (IntPred::Eq, 3));
        assert!(keeps(&b).is_empty(), "{:?}: x = 3 is below 5", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_a_bound_into_its_block_rules_out_by_equality() {
        let b = branch_then_store((IntPred::Ult, 5), (IntPred::Eq, 7));
        assert!(converted(&b).is_empty(), "{:?}: x < 5 in the block, where x == 7 is false", converted(&b));
    }

    fn permuted_from(odd: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let zero = b.constant(e, Ty::I32, 0);
        let first = b.cmp(e, IntPred::Eq, lane, zero);
        let seven = b.constant(e, Ty::I32, 7);
        let x = b.core(e, Ty::I32, Op::Select(first, data, seven));
        let u = uniform_load(&mut b, &k, e, 16);
        let two = b.constant(e, Ty::I32, 2);
        let bit = b.int(e, IntOp::And, u, two);
        let one = b.constant(e, Ty::I32, 1);
        let source = if odd {
            let raised = b.int(e, IntOp::Or, lane, one);
            b.int(e, IntOp::Or, raised, bit)
        } else {
            b.int(e, IntOp::And, u, one)
        };
        let four = b.constant(e, Ty::I32, 4);
        let index = b.int(e, IntOp::Mul, source, four);
        let yes = b.constant(e, Ty::I1, 1);
        let read = b.wave(e, WaveOp::Bpermute, vec![index, x, yes]);
        store_own(&mut b, &k, e, read, k.exec);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_word_a_permute_may_fetch_from_lane_zero() {
        let b = permuted_from(false);
        assert!(keeps(&b).is_empty(), "{:?}: when u is even every lane fetches lane 0's word, which the query picks", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_word_a_permute_never_fetches() {
        let b = permuted_from(true);
        assert!(converted(&b).is_empty(), "{:?}: every lane fetches from an odd lane, whose word is 7", converted(&b));
    }

    fn over_active_bits(outside: u64, every: bool) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let sixteen = b.constant(e, Ty::I32, 16);
        let low = b.cmp(e, IntPred::Ult, lane, sixteen);
        let exec = b.int(e, IntOp::And, low, k.exec);
        let flags = k.buffer(&mut b, e, 8);
        let buf = k.buffer(&mut b, e, 0);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        b.br(e, then, vec![exec, flags, buf]);
        let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, then, t[1], lane, 4);
        let yes = b.constant(then, Ty::I1, 1);
        let flag = b.load(then, Space::Global, MemSize::B32, own, yes);
        let other = b.constant(then, Ty::I32, outside);
        let v = b.core(then, Ty::I32, Op::Select(t[0], flag, other));
        let five = b.constant(then, Ty::I32, 5);
        let small = b.cmp(then, IntPred::Ult, v, five);
        let at = b.here(then);
        let q = b.wave(then, WaveOp::Any, vec![small]);
        let one = b.constant(then, Ty::I32, 1);
        let two = b.constant(then, Ty::I32, 2);
        let data = b.core(then, Ty::I32, Op::Select(q, one, two));
        let out = byte_offset(&mut b, then, t[2], lane, 4);
        b.store(then, Space::Global, MemSize::B32, out, data, t[0]);
        let Inst::Effect { provenance, .. } = b.f.blocks[&then].insts[at.1] else { unreachable!() };
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
            assert!(kept.queries.contains(&q), "{}: a lane whose flag is small stores 1, the others 2", name);
            assert_eq!(everyone.contains(&provenance), every, "{}: a lane with exec clear sees {}", name, outside);
        }
    }

    fn over_active_floats(outside: u64, every: bool) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let sixteen = b.constant(e, Ty::I32, 16);
        let low = b.cmp(e, IntPred::Ult, lane, sixteen);
        let exec = b.int(e, IntOp::And, low, k.exec);
        let flags = k.buffer(&mut b, e, 8);
        let buf = k.buffer(&mut b, e, 0);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        b.br(e, then, vec![exec, flags, buf]);
        let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, then, t[1], lane, 4);
        let yes = b.constant(then, Ty::I1, 1);
        let word = b.load(then, Space::Global, MemSize::B32, own, yes);
        let flag = b.core(then, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, word));
        let other = b.constant(then, Ty::F32, outside);
        let v = b.core(then, Ty::F32, Op::Select(t[0], flag, other));
        let five = b.constant(then, Ty::F32, 0x40a0_0000);
        let small = b.core(then, Ty::I1, Op::FCmp(FloatPred::Olt, v, five));
        let at = b.here(then);
        let q = b.wave(then, WaveOp::Any, vec![small]);
        let one = b.constant(then, Ty::I32, 1);
        let two = b.constant(then, Ty::I32, 2);
        let data = b.core(then, Ty::I32, Op::Select(q, one, two));
        let out = byte_offset(&mut b, then, t[2], lane, 4);
        b.store(then, Space::Global, MemSize::B32, out, data, t[0]);
        let Inst::Effect { provenance, .. } = b.f.blocks[&then].insts[at.1] else { unreachable!() };
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
            assert!(kept.queries.contains(&q), "{}: a lane whose flag is small stores 1, the others 2", name);
            assert_eq!(everyone.contains(&provenance), every, "{}: a lane with exec clear sees {:#x}", name, outside);
        }
    }

    #[test]
    fn prove_demands_no_lane_for_a_kept_query_over_float_bits_only_active_lanes_set() {
        over_active_floats(0x42c8_0000, false);
    }

    #[test]
    fn prove_demands_every_lane_for_a_kept_query_over_float_bits_inactive_lanes_set() {
        over_active_floats(0x4040_0000, true);
    }

    fn over_active_products(masked: bool, every: bool) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let sixteen = b.constant(e, Ty::I32, 16);
        let low = b.cmp(e, IntPred::Ult, lane, sixteen);
        let exec = b.int(e, IntOp::And, low, k.exec);
        let flags = k.buffer(&mut b, e, 8);
        let buf = k.buffer(&mut b, e, 0);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        b.br(e, then, vec![exec, flags, buf]);
        let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, then, t[1], lane, 4);
        let yes = b.constant(then, Ty::I1, 1);
        let flag = b.load(then, Space::Global, MemSize::B32, own, yes);
        let factor = b.core(then, Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, if masked { t[0] } else { yes }));
        let v = b.int(then, IntOp::Mul, flag, factor);
        let zero = b.constant(then, Ty::I32, 0);
        let set = b.cmp(then, IntPred::Ne, v, zero);
        let at = b.here(then);
        let q = b.wave(then, WaveOp::Any, vec![set]);
        let one = b.constant(then, Ty::I32, 1);
        let two = b.constant(then, Ty::I32, 2);
        let data = b.core(then, Ty::I32, Op::Select(q, one, two));
        let out = byte_offset(&mut b, then, t[2], lane, 4);
        b.store(then, Space::Global, MemSize::B32, out, data, t[0]);
        let Inst::Effect { provenance, .. } = b.f.blocks[&then].insts[at.1] else { unreachable!() };
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
            assert!(kept.queries.contains(&q), "{}: a lane whose flag is set stores 1, the others 2", name);
            assert_eq!(everyone.contains(&provenance), every, "{}: a lane with exec clear multiplies its flag by {}", name, !masked as u32);
        }
    }

    #[test]
    fn prove_demands_no_lane_for_a_kept_query_over_products_with_the_exec_bit() {
        over_active_products(true, false);
    }

    #[test]
    fn prove_demands_every_lane_for_a_kept_query_over_products_with_one() {
        over_active_products(false, true);
    }

    #[test]
    fn prove_demands_no_lane_for_a_kept_query_over_bits_only_active_lanes_set() {
        over_active_bits(100, false);
    }

    #[test]
    fn prove_demands_every_lane_for_a_kept_query_over_bits_inactive_lanes_set() {
        over_active_bits(3, true);
    }

    fn flag_query(b: &mut Build, k: &Kernel, e: BlockId) -> ValueId {
        let flag = per_lane(b, k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        b.wave(e, WaveOp::Any, vec![c])
    }

    #[test]
    fn prove_keeps_a_query_whose_arm_holds_a_meeting() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let buf = k.buffer(&mut b, e, 16);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (then, vec![k.exec, buf]), (join, vec![k.exec]));
        let s1 = store(&mut b, then, t[1], 1, t[0]);
        let s2 = store(&mut b, then, t[1], 2, t[0]);
        b.br(then, join, vec![t[0]]);
        let program = b.program();
        let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards);
            assert!(kept.queries.contains(&q), "{}: a lane without the flag skips the arm whose meeting every lane must reach", name);
            let meets: BTreeSet<Position> = kept.meets.iter().map(|&m| hazards.meetings[m]).collect();
            assert_eq!(meets, BTreeSet::from([s2]), "{}: the second store must follow the first", name);
        }
    }

    fn arms_carry(first: u64, second: u64) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64]);
        let (join, j) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        b.cond_br(e, q, (then, vec![k.exec, own]), (other, vec![k.exec, own]));
        let x = b.constant(then, Ty::I32, first);
        b.br(then, join, vec![t[0], t[1], x]);
        let y = b.constant(other, Ty::I32, second);
        b.br(other, join, vec![o[0], o[1], y]);
        b.store(join, Space::Global, MemSize::B32, j[1], j[2], j[0]);
        (b, q)
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_carry_different_words_to_a_store() {
        let (b, q) = arms_carry(1, 2);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| !kept.queries.contains(&q)).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag stores 2 where the wave stores 1", wrong);
    }

    #[test]
    fn prove_converts_a_query_whose_arms_carry_the_same_word() {
        let (b, _) = arms_carry(1, 1);
        assert!(converted(&b).is_empty(), "{:?}: both arms hand the store 1", converted(&b));
    }

    fn ballot_count(masked: bool) -> (Build, u64) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let five = b.constant(e, Ty::I32, 5);
        let low = b.cmp(e, IntPred::Ult, lane, five);
        let x = if masked { b.int(e, IntOp::And, low, k.exec) } else { low };
        let at = b.here(e);
        let w = b.wave(e, WaveOp::Ballot, vec![x]);
        let count = b.core(e, Ty::I32, Op::PopulationCount(w));
        store_own(&mut b, &k, e, count, k.exec);
        let Inst::Effect { provenance, .. } = b.f.blocks[&e].insts[at.1] else { unreachable!() };
        (b, provenance)
    }

    #[test]
    fn prove_demands_every_lane_for_a_ballot_of_unmasked_bits() {
        let (b, provenance) = ballot_count(false);
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (_, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
            assert!(everyone.contains(&provenance), "{}: the ballot counts the bits of lanes whose exec is clear", name);
        }
    }

    #[test]
    fn prove_demands_no_lane_for_a_ballot_of_masked_bits() {
        let (b, provenance) = ballot_count(true);
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (_, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
            assert!(!everyone.contains(&provenance), "{}: a lane with exec clear adds no bit", name);
        }
    }

    #[test]
    fn prove_keeps_a_query_that_picks_which_whole_word_a_count_reads() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let four = b.constant(e, Ty::I32, 4);
        let low = b.cmp(e, IntPred::Ult, lane, four);
        let low = b.int(e, IntOp::And, low, k.exec);
        let w1 = b.wave(e, WaveOp::Ballot, vec![low]);
        let w2 = b.wave(e, WaveOp::Ballot, vec![k.exec]);
        let w = b.core(e, Ty::I32, Op::Select(q, w1, w2));
        let count = b.core(e, Ty::I32, Op::PopulationCount(w));
        store_own(&mut b, &k, e, count, k.exec);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| !kept.queries.contains(&q)).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: a lane without the flag counts 32 bits where the wave counts 4", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_answer_a_kept_query_gathers_from_other_lanes() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = flag_query(&mut b, &k, e);
        let flag = per_lane(&mut b, &k, e, 16);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let x = b.int(e, IntOp::And, first, c);
        let second = b.wave(e, WaveOp::Any, vec![x]);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let data = b.core(e, Ty::I32, Op::Select(second, one, two));
        store_own(&mut b, &k, e, data, k.exec);
        let wrong: Vec<(&str, usize)> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&first) || !kept.queries.contains(&second))
            .map(|(n, kept)| (*n, kept.queries.len()))
            .collect();
        assert!(
            wrong.is_empty(),
            "{:?}: with the first query converted, the second gathers d_i & c_i instead of any(d) & c_i, which differ when d and c are set in different lanes",
            wrong
        );
    }

    #[test]
    fn prove_keeps_a_query_that_only_other_lanes_feed_into_a_kept_query() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = flag_query(&mut b, &k, e);
        let flag = per_lane(&mut b, &k, e, 16);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let x = b.int(e, IntOp::And, first, c);
        let second = b.wave(e, WaveOp::Any, vec![x]);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let data = b.core(e, Ty::I32, Op::Select(second, one, two));
        let yes = b.constant(e, Ty::I1, 1);
        let clear = b.int(e, IntOp::Xor, set, yes);
        let mask = b.int(e, IntOp::And, clear, k.exec);
        store_own(&mut b, &k, e, data, mask);
        let wrong: Vec<(&str, Vec<bool>)> = ["search", "direct"]
            .iter()
            .zip(both(&b))
            .filter(|(_, kept)| !kept.queries.contains(&first) || !kept.queries.contains(&second))
            .map(|(n, kept)| (*n, vec![kept.queries.contains(&first), kept.queries.contains(&second)]))
            .collect();
        assert!(
            wrong.is_empty(),
            "{:?} (first kept, second kept): the lanes that store have c clear, yet the second query gathers d_i & c_i from the others instead of any(d) & c_i",
            wrong
        );
    }

    #[test]
    fn prove_converts_a_query_whose_arm_stores_only_for_lanes_that_hold_the_bit() {
        let Flagged { mut b, k, buf, lane, c } = flagged();
        let e = BlockId(0);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
        let (join, _) = b.block(&[Ty::I1]);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        b.cond_br(e, q, (then, vec![k.exec, own, c]), (join, vec![k.exec]));
        let one = b.constant(then, Ty::I32, 1);
        b.store(then, Space::Global, MemSize::B32, t[1], one, t[2]);
        b.br(then, join, vec![t[0]]);
        assert!(converted(&b).is_empty(), "{:?}: a lane that stores holds the bit, so it takes the arm in both programs", converted(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_moved_address_only_lanes_that_hold_the_bit_use() {
        let Flagged { mut b, buf, lane, c, .. } = flagged();
        let e = BlockId(0);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let near = b.constant(e, Ty::I64, 0);
        let far = b.constant(e, Ty::I64, 4);
        let shift = b.core(e, Ty::I64, Op::Select(q, near, far));
        let own = byte_offset(&mut b, e, buf, lane, 8);
        let address = b.int(e, IntOp::Add, own, shift);
        let one = b.constant(e, Ty::I32, 1);
        b.store(e, Space::Global, MemSize::B32, address, one, c);
        assert!(converted(&b).is_empty(), "{:?}: a lane that stores holds the bit and sees the query true in both programs", converted(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_picked_word_only_lanes_that_hold_the_bit_load() {
        let Flagged { mut b, k, buf, lane, c } = flagged();
        let e = BlockId(0);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let table = k.buffer(&mut b, e, 16);
        let four = b.constant(e, Ty::I64, 4);
        let second = b.int(e, IntOp::Add, table, four);
        let from = b.core(e, Ty::I64, Op::Select(q, table, second));
        let v = b.load(e, Space::Global, MemSize::B32, from, c);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, v, c);
        assert!(converted(&b).is_empty(), "{:?}: a lane that loads and stores holds the bit and picks the first word in both programs", converted(&b));
    }

    #[test]
    fn prove_converts_a_query_that_masks_only_lanes_that_hold_the_bit() {
        let Flagged { mut b, buf, lane, c, .. } = flagged();
        let e = BlockId(0);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let mask = b.int(e, IntOp::And, q, c);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let one = b.constant(e, Ty::I32, 1);
        b.store(e, Space::Global, MemSize::B32, own, one, mask);
        assert!(converted(&b).is_empty(), "{:?}: q & c is c in both programs", converted(&b));
    }

    #[test]
    fn prove_converts_a_ballot_whose_lane_test_only_lanes_that_hold_the_bit_use() {
        let Flagged { mut b, buf, lane, c, .. } = flagged();
        let e = BlockId(0);
        let w = b.wave(e, WaveOp::Ballot, vec![c]);
        let zero = b.constant(e, Ty::I32, 0);
        let any = b.cmp(e, IntPred::Ne, w, zero);
        let mask = b.int(e, IntOp::And, any, c);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let one = b.constant(e, Ty::I32, 1);
        b.store(e, Space::Global, MemSize::B32, own, one, mask);
        assert!(converted(&b).is_empty(), "{:?}: (ballot(c) != 0) & c is c in both programs", converted(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_whole_word_only_lanes_that_hold_the_bit_count() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flag = per_lane(&mut b, &k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let four = b.constant(e, Ty::I32, 4);
        let low = b.cmp(e, IntPred::Ult, lane, four);
        let low = b.int(e, IntOp::And, low, k.exec);
        let w1 = b.wave(e, WaveOp::Ballot, vec![low]);
        let w2 = b.wave(e, WaveOp::Ballot, vec![k.exec]);
        let w = b.core(e, Ty::I32, Op::Select(q, w1, w2));
        let count = b.core(e, Ty::I32, Op::PopulationCount(w));
        store_own(&mut b, &k, e, count, c);
        assert!(converted(&b).is_empty(), "{:?}: a lane that stores holds the bit and counts the first word in both programs", converted(&b));
    }

    fn arms_compute(then_arm: usize, other_arm: usize) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let one = b.constant(e, Ty::I32, 1);
        let next = b.int(e, IntOp::Add, lane, one);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
        let (join, j) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        b.cond_br(e, q, (then, vec![k.exec, own, lane, next]), (other, vec![k.exec, own, lane, next]));
        for (block, p, arm) in [(then, t, then_arm), (other, o, other_arm)] {
            let (x, y) = (p[2], p[3]);
            let v = match arm {
                0 => x,
                1 => {
                    let pair = b.core(block, Ty::I64, Op::Pack64(x, y));
                    b.core(block, Ty::I32, Op::UnpackLo(pair))
                }
                2 => {
                    let pair = b.core(block, Ty::I64, Op::Pack64(x, y));
                    b.core(block, Ty::I32, Op::UnpackHi(pair))
                }
                3 => {
                    let ones = b.constant(block, Ty::I32, 0xffff_ffff);
                    b.int(block, IntOp::And, x, ones)
                }
                4 => {
                    let zero = b.constant(block, Ty::I32, 0);
                    b.int(block, IntOp::Xor, x, zero)
                }
                5 => {
                    let float = b.core(block, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, x));
                    b.core(block, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, float))
                }
                6 => {
                    let pair = b.core(block, Ty::I64, Op::Pack64(x, y));
                    let thirty_two = b.constant(block, Ty::I64, 32);
                    let down = b.int(block, IntOp::LShr, pair, thirty_two);
                    b.core(block, Ty::I32, Op::UnpackLo(down))
                }
                8 => b.int(block, IntOp::Add, x, x),
                9 => {
                    let two = b.constant(block, Ty::I32, 2);
                    b.int(block, IntOp::Mul, x, two)
                }
                10 => {
                    let one = b.constant(block, Ty::I32, 1);
                    b.int(block, IntOp::Shl, x, one)
                }
                11 => {
                    let one = b.constant(block, Ty::I32, 1);
                    let up = b.int(block, IntOp::Add, x, one);
                    b.int(block, IntOp::Sub, up, one)
                }
                12 => {
                    let three = b.constant(block, Ty::I32, 3);
                    b.int(block, IntOp::Mul, x, three)
                }
                13 => {
                    let two = b.constant(block, Ty::I32, 2);
                    b.int(block, IntOp::Shl, x, two)
                }
                14 => b.int(block, IntOp::Mul, x, y),
                15 => b.int(block, IntOp::Mul, y, x),
                16 => b.int(block, IntOp::Mul, x, x),
                _ => y,
            };
            b.br(block, join, vec![p[0], p[1], v]);
        }
        b.store(join, Space::Global, MemSize::B32, j[1], j[2], j[0]);
        (b, q)
    }

    #[test]
    fn prove_converts_a_query_whose_arms_compute_one_word_in_different_ways() {
        let cases = [
            ("lo(pack(x, y)) and x", 1, 0),
            ("x & ~0 and x", 3, 0),
            ("x ^ 0 and x", 4, 0),
            ("bitcast round trip and x", 5, 0),
            ("hi(pack(x, y)) and y", 2, 7),
            ("lo(pack(x, y) >> 32) and y", 6, 7),
            ("x + x and x * 2", 8, 9),
            ("x << 1 and x * 2", 10, 9),
            ("(x + 1) - 1 and x", 11, 0),
        ];
        let kept: Vec<(&str, Vec<&str>)> = cases
            .iter()
            .map(|&(name, a, b)| (name, converted(&arms_compute(a, b).0)))
            .filter(|(_, names)| !names.is_empty())
            .collect();
        assert!(kept.is_empty(), "{:?}: both arms hand the store the same word", kept);
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_multiply_different_words() {
        let (b, _) = arms_compute(14, 16);
        assert!(keeps(&b).is_empty(), "{:?}: one arm stores x * y and the other x * x", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_arms_multiply_two_words_in_either_order() {
        let (b, _) = arms_compute(14, 15);
        assert!(converted(&b).is_empty(), "{:?}: x * y and y * x are one word", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_compute_different_words() {
        let cases = [
            ("hi(pack(x, y)) and x", 2, 0),
            ("lo(pack(x, y)) and y", 1, 7),
            ("lo(pack(x, y) >> 32) and x", 6, 0),
            ("x and y", 0, 7),
            ("x + x and x * 3", 8, 12),
            ("x << 1 and x << 2", 10, 13),
            ("(x + 1) - 1 and y", 11, 7),
        ];
        let wrong: Vec<(&str, Vec<&str>)> = cases
            .iter()
            .map(|&(name, a, b)| {
                let (program, q) = arms_compute(a, b);
                let names: Vec<&str> = ["search", "direct"]
                    .iter()
                    .zip(both(&program))
                    .filter(|(_, kept)| !kept.queries.contains(&q))
                    .map(|(n, _)| *n)
                    .collect();
                (name, names)
            })
            .filter(|(_, names)| !names.is_empty())
            .collect();
        assert!(wrong.is_empty(), "{:?}: the arms hand the store the lane id and the lane id + 1", wrong);
    }

    #[test]
    fn search_keeps_a_query_that_another_lane_reads_through_a_lane_exchange() {
        let converted: Vec<Reader> = [Reader::ReadLane, Reader::ReadFirstLane, Reader::WriteLane, Reader::Bpermute, Reader::BpermuteFi, Reader::Wmma]
            .iter()
            .copied()
            .filter(|&reader| {
                let (b, q) = reads_another_lane(reader);
                let (kept, _) = search::prove(&b.f, &b.inputs, Some(0), &no_hazards());
                !kept.queries.contains(&q)
            })
            .collect();
        assert!(
            converted.is_empty(),
            "{:?}: when lane 0 alone has a zero flag, lane 0 answers the query false and hands the storing lanes 0 instead of the pair of halves 1.0",
            converted
        );
    }

    #[test]
    fn direct_keeps_a_query_that_another_lane_reads_through_a_lane_exchange() {
        let converted: Vec<Reader> = [Reader::ReadLane, Reader::ReadFirstLane, Reader::WriteLane, Reader::Bpermute, Reader::BpermuteFi, Reader::Wmma]
            .iter()
            .copied()
            .filter(|&reader| {
                let (b, q) = reads_another_lane(reader);
                let (kept, _) = direct::prove(&b.f, &b.inputs, Some(0), &no_hazards());
                !kept.queries.contains(&q)
            })
            .collect();
        assert!(
            converted.is_empty(),
            "{:?}: when lane 0 alone has a zero flag, lane 0 answers the query false and hands the storing lanes 0 instead of the pair of halves 1.0",
            converted
        );
    }

    #[test]
    fn prove_keeps_a_query_whose_word_lanes_outside_exec_hand_to_a_read_that_ignores_exec() {
        let converted: Vec<(Reader, &str)> = [Reader::ReadLane, Reader::BpermuteFi, Reader::Wmma]
            .iter()
            .flat_map(|&reader| {
                let (b, q) = reads_outside_exec(reader);
                ["search", "direct"]
                    .iter()
                    .zip(both(&b))
                    .filter(|(_, kept)| !kept.queries.contains(&q))
                    .map(|(name, _)| (reader, *name))
                    .collect::<Vec<_>>()
            })
            .collect();
        assert!(
            converted.is_empty(),
            "{:?}: when lane 0 has exec clear and lane 1 the flag, lane 0 answers the query false and hands the storing lanes 0 instead of the pair of halves 1.0",
            converted
        );
    }

    #[test]
    fn prove_converts_a_query_whose_word_only_lanes_a_masked_read_skips_hold() {
        let kept: Vec<(Reader, Vec<&str>)> = [Reader::ReadFirstLane, Reader::Bpermute]
            .iter()
            .map(|&reader| (reader, converted(&reads_outside_exec(reader).0)))
            .filter(|(_, names)| !names.is_empty())
            .collect();
        assert!(
            kept.is_empty(),
            "{:?}: only lanes with exec clear hold a word the query picks, and these reads take no such lane's word while a lane stores",
            kept
        );
    }

    fn lane_read(selector: impl Fn(&mut Build, BlockId, &Kernel) -> ValueId) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let zero = b.constant(e, Ty::I32, 0);
        let first = b.cmp(e, IntPred::Eq, lane, zero);
        let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
        let picked = b.core(e, Ty::I32, Op::Select(q, ones, lane));
        let x = b.core(e, Ty::I32, Op::Select(first, lane, picked));
        let selector = selector(&mut b, e, &k);
        let y = b.wave(e, WaveOp::ReadLane, vec![x, selector, zero]);
        store_own(&mut b, &k, e, y, k.exec);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_word_only_lanes_a_lane_read_skips_hold() {
        let b = lane_read(|b, e, _| b.constant(e, Ty::I32, 32));
        assert!(converted(&b).is_empty(), "{:?}: the read takes lane 32 & 31 = 0, whose word is its lane id 0 whatever the query answers", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_word_the_lane_a_lane_read_takes_holds() {
        let b = lane_read(|b, e, _| b.constant(e, Ty::I32, 1));
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: lane 1's word is 0x3c003c00 or 1 by the query", wrong);
    }

    #[test]
    fn prove_keeps_a_query_whose_word_one_of_the_lanes_a_lane_read_may_take_holds() {
        let b = lane_read(|b, e, k| {
            let flag = per_lane(b, k, e, 24);
            let zero = b.constant(e, Ty::I32, 0);
            let set = b.cmp(e, IntPred::Ne, flag, zero);
            let one = b.constant(e, Ty::I32, 1);
            b.core(e, Ty::I32, Op::Select(set, zero, one))
        });
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: lanes without the flag read lane 1, whose word depends on the query", wrong);
    }

    fn accumulator_query(stored: usize) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let fzero = b.constant(e, Ty::F32, 0);
        let fone = b.constant(e, Ty::F32, 0x3f80_0000);
        let second = b.core(e, Ty::F32, Op::Select(q, fone, fzero));
        let mut inputs = vec![lane; 8];
        inputs.extend([fzero, second]);
        inputs.extend([fzero; 6]);
        let outputs = b.effect(e, EffectOp::Wave(WaveOp::Wmma), inputs);
        let word = b.core(e, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, outputs[stored]));
        store_own(&mut b, &k, e, word, k.exec);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_word_only_an_accumulator_the_stored_output_skips_holds() {
        let b = accumulator_query(0);
        assert!(converted(&b).is_empty(), "{:?}: output 0 adds the products to accumulator 0 alone, and only accumulator 1 depends on the query", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_word_the_accumulator_of_the_stored_output_holds() {
        let b = accumulator_query(1);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: output 1 starts from accumulator 1, which is 1 or 0 by the query", wrong);
    }

    #[test]
    fn prove_keeps_a_query_that_lane_zero_hands_to_a_first_lane_read_over_an_empty_mask() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let flag = per_lane(&mut b, &k, e, 16);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let mask = b.int(e, IntOp::And, set, k.exec);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let first = b.cmp(e, IntPred::Eq, lane, zero);
        let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
        let picked = b.core(e, Ty::I32, Op::Select(q, ones, lane));
        let idle = b.core(e, Ty::I32, Op::Select(first, picked, lane));
        let x = b.core(e, Ty::I32, Op::Select(mask, lane, idle));
        let y = b.wave(e, WaveOp::ReadFirstLane, vec![x, mask]);
        let yes = b.constant(e, Ty::I1, 1);
        let clear = b.int(e, IntOp::Xor, mask, yes);
        let others = b.int(e, IntOp::Xor, first, yes);
        let stores = b.int(e, IntOp::And, clear, others);
        let stores = b.int(e, IntOp::And, stores, k.exec);
        store_own(&mut b, &k, e, y, stores);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| !kept.queries.contains(&q)).map(|(n, _)| *n).collect();
        assert!(
            wrong.is_empty(),
            "{:?}: when no lane sets the mask, the read takes lane 0's word, and with the flag clear in lane 0 and set in lane 1, lane 0 answers the query false and hands lane 1 its lane id 0 instead of the pair of halves 1.0",
            wrong
        );
    }

    fn orders_in_two_blocks_of_kinds(second: IntPred) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let v = per_lane(&mut b, &k, e, 16);
        let w = per_lane(&mut b, &k, e, 24);
        let first = b.cmp(e, IntPred::Ult, v, w);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32, Ty::I1]);
        b.br(e, next, vec![k.exec, own, data, v, w, first]);
        let later = b.cmp(next, second, p[4], p[3]);
        let both = b.int(next, IntOp::And, p[5], later);
        let mask = b.int(next, IntOp::And, both, p[0]);
        b.store(next, Space::Global, MemSize::B32, p[1], p[2], mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_an_order_and_its_swap_in_the_next_block_mask() {
        let b = orders_in_two_blocks_of_kinds(IntPred::Ugt);
        assert!(keeps(&b).is_empty(), "{:?}: v < w is w > v, so lanes with v < w store", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_an_order_and_an_equality_in_the_next_block_mask() {
        let b = orders_in_two_blocks_of_kinds(IntPred::Eq);
        assert!(converted(&b).is_empty(), "{:?}: v < w in the first block rules out w == v in the next", converted(&b));
    }

    fn orders_of_both_signs(bounded: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let x = per_lane(&mut b, &k, e, 16);
        let y = per_lane(&mut b, &k, e, 24);
        let half = b.constant(e, Ty::I32, 1 << 31);
        let signed = b.cmp(e, IntPred::Slt, x, y);
        let unsigned = b.cmp(e, IntPred::Ult, y, x);
        let y_small = b.cmp(e, IntPred::Ult, y, half);
        let orders = b.int(e, IntOp::And, signed, unsigned);
        let mut all = b.int(e, IntOp::And, orders, y_small);
        if bounded {
            let x_small = b.cmp(e, IntPred::Ult, x, half);
            all = b.int(e, IntOp::And, all, x_small);
        }
        let mask = b.int(e, IntOp::And, all, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_opposite_orders_of_both_signs_mask_for_a_negative_word() {
        let b = orders_of_both_signs(false);
        assert!(keeps(&b).is_empty(), "{:?}: x = -1 and y = 0 give x s< y and y u< x", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_opposite_orders_of_both_signs_mask_for_two_small_words() {
        let b = orders_of_both_signs(true);
        assert!(converted(&b).is_empty(), "{:?}: below 2^31 the signed and unsigned orders agree, so x s< y and y u< x never hold together", converted(&b));
    }

    fn float_orders(opposite: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let v = per_lane(&mut b, &k, e, 16);
        let w = per_lane(&mut b, &k, e, 24);
        let x = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, v));
        let y = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, w));
        let below = b.core(e, Ty::I1, Op::FCmp(FloatPred::Olt, x, y));
        let other = if opposite { b.core(e, Ty::I1, Op::FCmp(FloatPred::Olt, y, x)) } else { b.core(e, Ty::I1, Op::FCmp(FloatPred::Ogt, y, x)) };
        let both = b.int(e, IntOp::And, below, other);
        let mask = b.int(e, IntOp::And, both, k.exec);
        store_own(&mut b, &k, e, data, mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_a_float_order_and_its_swap_mask() {
        let b = float_orders(false);
        assert!(keeps(&b).is_empty(), "{:?}: x < y is y > x, so lanes with x < y store", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_opposite_float_orders_mask() {
        let b = float_orders(true);
        assert!(converted(&b).is_empty(), "{:?}: x < y and y < x never hold together", converted(&b));
    }

    fn bound_carried_through_a_sum(limit: u64) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let table = k.buffer(&mut b, e, 16);
        let yes = b.constant(e, Ty::I1, 1);
        let x = b.load(e, Space::Global, MemSize::B32, table, yes);
        let five = b.constant(e, Ty::I32, 5);
        let enters = b.cmp(e, IntPred::Ult, x, five);
        let one = b.constant(e, Ty::I32, 1);
        let next = b.int(e, IntOp::Add, x, one);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.cond_br(e, enters, (then, vec![k.exec, next, data, own]), (exit, vec![k.exec]));
        let limit = b.constant(then, Ty::I32, limit);
        let test = b.cmp(then, IntPred::Eq, t[1], limit);
        let mask = b.int(then, IntOp::And, test, t[0]);
        b.store(then, Space::Global, MemSize::B32, t[3], t[2], mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_store_a_bound_carried_through_a_sum_lets_through() {
        let b = bound_carried_through_a_sum(3);
        assert!(keeps(&b).is_empty(), "{:?}: x = 2 is below 5 and x + 1 is 3", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_store_a_bound_carried_through_a_sum_rules_out() {
        let b = bound_carried_through_a_sum(7);
        assert!(converted(&b).is_empty(), "{:?}: x < 5 in the first block, so the x + 1 the block receives is never 7", converted(&b));
    }

    fn over_active_products_carried(masked: bool, every: bool) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let sixteen = b.constant(e, Ty::I32, 16);
        let low = b.cmp(e, IntPred::Ult, lane, sixteen);
        let exec = b.int(e, IntOp::And, low, k.exec);
        let flags = k.buffer(&mut b, e, 8);
        let buf = k.buffer(&mut b, e, 0);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (next, n) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        b.br(e, then, vec![exec, flags, buf]);
        let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, then, t[1], lane, 4);
        let yes = b.constant(then, Ty::I1, 1);
        let flag = b.load(then, Space::Global, MemSize::B32, own, yes);
        let factor = b.core(then, Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, if masked { t[0] } else { yes }));
        let v = b.int(then, IntOp::Mul, flag, factor);
        b.br(then, next, vec![t[0], t[2], v]);
        let zero = b.constant(next, Ty::I32, 0);
        let set = b.cmp(next, IntPred::Ne, n[2], zero);
        let at = b.here(next);
        let q = b.wave(next, WaveOp::Any, vec![set]);
        let one = b.constant(next, Ty::I32, 1);
        let two = b.constant(next, Ty::I32, 2);
        let data = b.core(next, Ty::I32, Op::Select(q, one, two));
        let lane = b.core(next, Ty::I32, Op::Env(Env::LaneId));
        let out = byte_offset(&mut b, next, n[1], lane, 4);
        b.store(next, Space::Global, MemSize::B32, out, data, n[0]);
        let Inst::Effect { provenance, .. } = b.f.blocks[&next].insts[at.1] else { unreachable!() };
        for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
            let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
            assert!(kept.queries.contains(&q), "{}: a lane whose flag is set stores 1, the others 2", name);
            assert_eq!(everyone.contains(&provenance), every, "{}: a lane with exec clear carries its flag times {}", name, !masked as u32);
        }
    }

    #[test]
    fn prove_demands_no_lane_for_a_kept_query_over_products_with_the_exec_bit_carried_into_the_next_block() {
        over_active_products_carried(true, false);
    }

    #[test]
    fn prove_demands_every_lane_for_a_kept_query_over_products_with_one_carried_into_the_next_block() {
        over_active_products_carried(false, true);
    }

    fn permuted_from_a_sum(odd: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let data = query_data(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let zero = b.constant(e, Ty::I32, 0);
        let first = b.cmp(e, IntPred::Eq, lane, zero);
        let seven = b.constant(e, Ty::I32, 7);
        let x = b.core(e, Ty::I32, Op::Select(first, data, seven));
        let one = b.constant(e, Ty::I32, 1);
        let doubled = b.int(e, IntOp::Shl, lane, one);
        let u = uniform_load(&mut b, &k, e, 16);
        let bit = b.int(e, IntOp::And, u, one);
        let two = b.constant(e, Ty::I32, 2);
        let far = b.int(e, IntOp::Shl, bit, two);
        let base = if odd { b.int(e, IntOp::Add, doubled, one) } else { doubled };
        let source = b.int(e, IntOp::Add, base, far);
        let four = b.constant(e, Ty::I32, 4);
        let index = b.int(e, IntOp::Mul, source, four);
        let yes = b.constant(e, Ty::I1, 1);
        let read = b.wave(e, WaveOp::Bpermute, vec![index, x, yes]);
        store_own(&mut b, &k, e, read, k.exec);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_word_a_permute_of_doubled_lanes_fetches_from_lane_zero() {
        let b = permuted_from_a_sum(false);
        assert!(keeps(&b).is_empty(), "{:?}: when u is even lanes 0 and 16 fetch lane 0's word, which the query picks", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_word_a_permute_of_doubled_lanes_plus_one_never_fetches() {
        let b = permuted_from_a_sum(true);
        assert!(converted(&b).is_empty(), "{:?}: 2 lane + 1 + 4 (u & 1) is odd, so every lane fetches from an odd lane, whose word is 7", converted(&b));
    }

    fn written_to_a_loaded_lane(skip: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let x = b.core(e, Ty::I32, Op::Select(q, one, two));
        let u = uniform_load(&mut b, &k, e, 16);
        let thirty_one = b.constant(e, Ty::I32, 31);
        let target = b.int(e, IntOp::And, u, thirty_one);
        let y = b.wave(e, WaveOp::WriteLane, vec![x, target, lane]);
        let mask = if skip {
            let other = b.cmp(e, IntPred::Ne, lane, target);
            b.int(e, IntOp::And, other, k.exec)
        } else {
            k.exec
        };
        store_own(&mut b, &k, e, y, mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_word_a_lane_write_puts_into_a_loaded_lane_that_stores() {
        let b = written_to_a_loaded_lane(false);
        assert!(keeps(&b).is_empty(), "{:?}: lane u & 31 stores lane 0's word, whose answer is lane 0's own flag", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_word_a_lane_write_puts_only_into_a_loaded_lane_that_never_stores() {
        let b = written_to_a_loaded_lane(true);
        assert!(converted(&b).is_empty(), "{:?}: every storing lane keeps its own lane id", converted(&b));
    }

    enum Written {
        OnlyTarget,
        SkipNext,
        SkipWide,
    }

    fn lane_written_under(shape: Written) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let x = b.core(e, Ty::I32, Op::Select(q, one, two));
        let u = uniform_load(&mut b, &k, e, 16);
        let thirty_one = b.constant(e, Ty::I32, 31);
        let target = match shape {
            Written::SkipWide => u,
            _ => b.int(e, IntOp::And, u, thirty_one),
        };
        let y = b.wave(e, WaveOp::WriteLane, vec![x, target, lane]);
        let test = match shape {
            Written::OnlyTarget => b.cmp(e, IntPred::Eq, lane, target),
            Written::SkipNext => {
                let next = b.int(e, IntOp::Add, target, one);
                b.cmp(e, IntPred::Ne, lane, next)
            }
            Written::SkipWide => b.cmp(e, IntPred::Ne, target, lane),
        };
        let mask = b.int(e, IntOp::And, test, k.exec);
        store_own(&mut b, &k, e, y, mask);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_word_a_lane_write_puts_into_the_only_lane_that_stores() {
        let b = lane_written_under(Written::OnlyTarget);
        assert!(keeps(&b).is_empty(), "{:?}: only lane u & 31 stores, and it stores the written word", keeps(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_lane_write_target_differs_from_the_lane_the_store_skips() {
        let b = lane_written_under(Written::SkipNext);
        assert!(keeps(&b).is_empty(), "{:?}: the store skips lane u & 31 + 1, so lane u & 31 stores the written word", keeps(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_lane_write_target_may_pass_the_last_lane() {
        let b = lane_written_under(Written::SkipWide);
        assert!(keeps(&b).is_empty(), "{:?}: u = 33 writes lane 1, while the store skips no lane", keeps(&b));
    }

    fn offset_lanes_below(shared: bool) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let table = k.buffer(&mut b, e, 16);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U8, table, yes);
        let shifted = b.int(e, IntOp::Add, lane, u);
        let five = b.constant(e, Ty::I32, 5);
        let other = if shared { b.int(e, IntOp::Add, lane, five) } else { five };
        let t = b.cmp(e, IntPred::Ult, shifted, other);
        let c = b.int(e, IntOp::And, t, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let data = b.core(e, Ty::I32, Op::Select(q, one, two));
        store_own(&mut b, &k, e, data, k.exec);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_bit_lanes_offset_by_a_byte_order_differently() {
        let b = offset_lanes_below(false);
        assert!(keeps(&b).is_empty(), "{:?}: lane + u < 5 holds in the lanes below 5 - u alone", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_bit_every_lane_offset_by_a_byte_orders_alike() {
        let b = offset_lanes_below(true);
        assert!(converted(&b).is_empty(), "{:?}: lane + u < lane + 5 is u < 5 in every lane, as neither side wraps", converted(&b));
    }

    fn apart_loop_pair(opposite: bool) -> Build {
        looped_pair(opposite, Flips::Both)
    }

    enum Flips {
        Both,
        OnlyFirst,
        SecondOnALoadedBit,
    }

    fn looped_pair(opposite: bool, flips: Flips) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let table = k.buffer(&mut b, e, 16);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        let everywhere = b.constant(e, Ty::I1, 1);
        let word = b.load(e, Space::Global, MemSize::B32, own, everywhere);
        let nothing = b.constant(e, Ty::I32, 0);
        let d = b.cmp(e, IntPred::Ne, word, nothing);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let flag = per_lane(&mut b, &k, e, 28);
        let zero = b.constant(e, Ty::I32, 0);
        let c = b.cmp(e, IntPred::Ne, flag, zero);
        let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I32, Ty::I1]);
        let (after, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (arm, vec![k.exec, own, c, d]), (join, vec![k.exec]));
        let yes = b.constant(arm, Ty::I1, 1);
        let not_c = b.int(arm, IntOp::Xor, a[2], yes);
        let zero = b.constant(arm, Ty::I32, 0);
        let second = if opposite { not_c } else { a[2] };
        b.br(arm, body, vec![a[0], a[1], a[2], second, zero, a[3]]);
        let yes = b.constant(body, Ty::I1, 1);
        let s = b.int(body, IntOp::Xor, p[2], yes);
        let t = match flips {
            Flips::Both => b.int(body, IntOp::Xor, p[3], yes),
            Flips::OnlyFirst => p[3],
            Flips::SecondOnALoadedBit => b.int(body, IntOp::Xor, p[3], p[5]),
        };
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[4], one);
        let three = b.constant(body, Ty::I32, 3);
        let again = b.cmp(body, IntPred::Ult, next, three);
        b.cond_br(body, again, (body, vec![p[0], p[1], s, t, next, p[5]]), (after, vec![p[0], p[1], s, t]));
        let yes = b.constant(after, Ty::I1, 1);
        let not_t = b.int(after, IntOp::Xor, m[3], yes);
        let only = b.int(after, IntOp::And, m[2], not_t);
        let mask = b.int(after, IntOp::And, only, m[0]);
        let one = b.constant(after, Ty::I32, 1);
        b.store(after, Space::Global, MemSize::B32, m[1], one, mask);
        b.br(after, join, vec![m[0]]);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_arm_loops_two_bits_that_stay_opposite() {
        let b = apart_loop_pair(true);
        assert!(keeps(&b).is_empty(), "{:?}: the loop carries (c, !c) flipped three times, so s & !t is !c and lanes without c store", keeps(&b));
    }

    fn while_pair(flips: Flips) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let table = k.buffer(&mut b, e, 16);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let slot = byte_offset(&mut b, e, table, lane, 4);
        let everywhere = b.constant(e, Ty::I1, 1);
        let word = b.load(e, Space::Global, MemSize::B32, slot, everywhere);
        let nothing = b.constant(e, Ty::I32, 0);
        let d = b.cmp(e, IntPred::Ne, word, nothing);
        let buf = k.buffer(&mut b, e, 0);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let flag = per_lane(&mut b, &k, e, 28);
        let c = b.cmp(e, IntPred::Ne, flag, nothing);
        let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (header, h) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I32, Ty::I1]);
        let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I32, Ty::I1]);
        let (after, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
        let (join, _) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (arm, vec![k.exec, own, c, d]), (join, vec![k.exec]));
        let zero = b.constant(arm, Ty::I32, 0);
        b.br(arm, header, vec![a[0], a[1], a[2], a[2], zero, a[3]]);
        let three = b.constant(header, Ty::I32, 3);
        let again = b.cmp(header, IntPred::Ult, h[4], three);
        b.cond_br(header, again, (body, h.clone()), (after, vec![h[0], h[1], h[2], h[3]]));
        let yes = b.constant(body, Ty::I1, 1);
        let s = b.int(body, IntOp::Xor, p[2], yes);
        let t = match flips {
            Flips::Both => b.int(body, IntOp::Xor, p[3], yes),
            Flips::OnlyFirst => p[3],
            Flips::SecondOnALoadedBit => b.int(body, IntOp::Xor, p[3], p[5]),
        };
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[4], one);
        b.br(body, header, vec![p[0], p[1], s, t, next, p[5]]);
        let yes = b.constant(after, Ty::I1, 1);
        let not_t = b.int(after, IntOp::Xor, m[3], yes);
        let only = b.int(after, IntOp::And, m[2], not_t);
        let mask = b.int(after, IntOp::And, only, m[0]);
        let one = b.constant(after, Ty::I32, 1);
        b.store(after, Space::Global, MemSize::B32, m[1], one, mask);
        b.br(after, join, vec![m[0]]);
        b
    }

    #[test]
    fn prove_converts_a_query_whose_while_loop_flips_two_bits_that_stay_equal() {
        let b = while_pair(Flips::Both);
        assert!(converted(&b).is_empty(), "{:?}: the header always holds s = t, so s & !t never holds", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_while_loop_flips_one_of_two_equal_bits_on_a_loaded_bit() {
        let b = while_pair(Flips::SecondOnALoadedBit);
        assert!(keeps(&b).is_empty(), "{:?}: after one pass s is !c and t is c ^ d, so s & !t is !c & !d", keeps(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arm_loops_two_equal_bits_and_flips_only_one() {
        let b = looped_pair(false, Flips::OnlyFirst);
        assert!(keeps(&b).is_empty(), "{:?}: after three flips s is !c while t stays c, so lanes without c store", keeps(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arm_loops_two_equal_bits_and_flips_one_on_a_loaded_bit() {
        let b = looped_pair(false, Flips::SecondOnALoadedBit);
        assert!(keeps(&b).is_empty(), "{:?}: t flips only where the loaded bit d is set, so after three flips s & !t is !c & !d", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_arm_loops_two_bits_that_stay_equal() {
        let b = apart_loop_pair(false);
        assert!(converted(&b).is_empty(), "{:?}: the loop flips both bits together, so s & !t never holds and the arm never stores", converted(&b));
    }

    fn arms_regroup(cases: &[(usize, usize)]) -> Vec<Build> {
        cases
            .iter()
            .map(|&(then_arm, other_arm)| {
                let (mut b, k) = Build::kernel();
                let e = BlockId(0);
                let q = flag_query(&mut b, &k, e);
                let buf = k.buffer(&mut b, e, 0);
                let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
                let one = b.constant(e, Ty::I32, 1);
                let two = b.constant(e, Ty::I32, 2);
                let y = b.int(e, IntOp::Add, lane, one);
                let z = b.int(e, IntOp::Add, lane, two);
                let own = byte_offset(&mut b, e, buf, lane, 4);
                let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32]);
                let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32]);
                let (join, j) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
                b.cond_br(e, q, (then, vec![k.exec, own, lane, y, z]), (other, vec![k.exec, own, lane, y, z]));
                for (block, p, arm) in [(then, t, then_arm), (other, o, other_arm)] {
                    let (x, y, z) = (p[2], p[3], p[4]);
                    let v = match arm {
                        0 => {
                            let xy = b.int(block, IntOp::Mul, x, y);
                            b.int(block, IntOp::Mul, xy, z)
                        }
                        1 => {
                            let yz = b.int(block, IntOp::Mul, y, z);
                            b.int(block, IntOp::Mul, x, yz)
                        }
                        2 => {
                            let sum = b.int(block, IntOp::Add, x, y);
                            b.int(block, IntOp::Mul, sum, z)
                        }
                        3 => {
                            let xz = b.int(block, IntOp::Mul, x, z);
                            let yz = b.int(block, IntOp::Mul, y, z);
                            b.int(block, IntOp::Add, xz, yz)
                        }
                        4 => {
                            let xy = b.int(block, IntOp::And, x, y);
                            b.int(block, IntOp::And, xy, z)
                        }
                        5 => {
                            let yz = b.int(block, IntOp::And, y, z);
                            b.int(block, IntOp::And, x, yz)
                        }
                        _ => {
                            let xz = b.int(block, IntOp::Mul, x, z);
                            b.int(block, IntOp::Add, xz, y)
                        }
                    };
                    b.br(block, join, vec![p[0], p[1], v]);
                }
                b.store(join, Space::Global, MemSize::B32, j[1], j[2], j[0]);
                b
            })
            .collect()
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_compute_different_products_of_sums() {
        let wrong: Vec<Vec<&str>> = arms_regroup(&[(2, 6)]).iter().map(keeps).filter(|w| !w.is_empty()).collect();
        assert!(wrong.is_empty(), "{:?}: (x + y) z and x z + y differ", wrong);
    }

    #[test]
    fn prove_converts_a_query_whose_arms_regroup_or_distribute_one_word() {
        let names = ["(x y) z and x (y z)", "(x + y) z and x z + y z", "(x & y) & z and x & (y & z)"];
        let kept: Vec<(&str, Vec<&str>)> = names
            .iter()
            .zip(arms_regroup(&[(0, 1), (2, 3), (4, 5)]))
            .map(|(&name, b)| (name, converted(&b)))
            .filter(|(_, names)| !names.is_empty())
            .collect();
        assert!(kept.is_empty(), "{:?}: both arms hand the store the same word", kept);
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_swap_different_words_atomically() {
        let b = arms_store(|b, block, side, p| {
            let expected = b.constant(block, Ty::I32, 0);
            let desired = b.constant(block, Ty::I32, 1 + side as u64);
            b.effect(block, memory(Space::Global, MemoryOp::AtomicCmpSwap), vec![p[1], desired, expected, p[0]]);
            None
        });
        assert!(keeps(&b).is_empty(), "{:?}: one arm swaps in 1 and the other 2", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_arms_swap_the_same_word_atomically() {
        let b = arms_store(|b, block, _, p| {
            let expected = b.constant(block, Ty::I32, 0);
            let desired = b.constant(block, Ty::I32, 1);
            b.effect(block, memory(Space::Global, MemoryOp::AtomicCmpSwap), vec![p[1], desired, expected, p[0]]);
            None
        });
        assert!(converted(&b).is_empty(), "{:?}: both arms swap 1 into the lane's own word when it holds 0", converted(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_swap_the_same_word_against_different_expected_words() {
        let b = arms_store(|b, block, side, p| {
            let expected = b.constant(block, Ty::I32, 2 * side as u64);
            let desired = b.constant(block, Ty::I32, 1);
            b.effect(block, memory(Space::Global, MemoryOp::AtomicCmpSwap), vec![p[1], desired, expected, p[0]]);
            None
        });
        assert!(keeps(&b).is_empty(), "{:?}: one arm swaps 1 in over 0 and the other over 2", keeps(&b));
    }

    fn arms_store_two_that_may_meet(second: u64) -> Build {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flag = per_lane(&mut b, &k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let u = uniform_load(&mut b, &k, e, 16);
        let one = b.constant(e, Ty::I32, 1);
        let bit = b.int(e, IntOp::And, u, one);
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, bit));
        let row = b.constant(e, Ty::I64, 128);
        let step = b.int(e, IntOp::Mul, wide, row);
        let maybe = b.int(e, IntOp::Add, own, step);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (join, j) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (then, vec![k.exec, own, maybe]), (other, vec![k.exec, own, maybe]));
        for (side, (block, p)) in [(then, t), (other, o)].iter().enumerate() {
            let five = b.constant(*block, Ty::I32, 5);
            let later = b.constant(*block, Ty::I32, if side == 0 { 5 } else { second });
            if side == 0 {
                b.store(*block, Space::Global, MemSize::B32, p[1], five, p[0]);
                b.store(*block, Space::Global, MemSize::B32, p[2], later, p[0]);
            } else {
                b.store(*block, Space::Global, MemSize::B32, p[2], later, p[0]);
                b.store(*block, Space::Global, MemSize::B32, p[1], five, p[0]);
            }
            b.br(*block, join, vec![p[0]]);
        }
        let out = k.buffer(&mut b, join, 24);
        let lane = b.core(join, Ty::I32, Op::Env(Env::LaneId));
        let target = byte_offset(&mut b, join, out, lane, 4);
        let one = b.constant(join, Ty::I32, 1);
        b.store(join, Space::Global, MemSize::B32, target, one, j[0]);
        b
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_store_two_values_into_two_words_that_may_meet_in_either_order() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flag = per_lane(&mut b, &k, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let u = uniform_load(&mut b, &k, e, 16);
        let one = b.constant(e, Ty::I32, 1);
        let bit = b.int(e, IntOp::And, u, one);
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, bit));
        let row = b.constant(e, Ty::I64, 128);
        let step = b.int(e, IntOp::Mul, wide, row);
        let maybe = b.int(e, IntOp::Add, own, step);
        let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
        let (join, j) = b.block(&[Ty::I1]);
        b.cond_br(e, q, (then, vec![k.exec, own, maybe]), (other, vec![k.exec, own, maybe]));
        for (side, (block, p)) in [(then, t), (other, o)].iter().enumerate() {
            let five = b.constant(*block, Ty::I32, 5);
            let six = b.constant(*block, Ty::I32, 6);
            if side == 0 {
                b.store(*block, Space::Global, MemSize::B32, p[1], five, p[0]);
                b.store(*block, Space::Global, MemSize::B32, p[2], six, p[0]);
            } else {
                b.store(*block, Space::Global, MemSize::B32, p[2], six, p[0]);
                b.store(*block, Space::Global, MemSize::B32, p[1], five, p[0]);
            }
            b.br(*block, join, vec![p[0]]);
        }
        let out = k.buffer(&mut b, join, 24);
        let lane = b.core(join, Ty::I32, Op::Env(Env::LaneId));
        let target = byte_offset(&mut b, join, out, lane, 4);
        let one = b.constant(join, Ty::I32, 1);
        b.store(join, Space::Global, MemSize::B32, target, one, j[0]);
        assert!(keeps(&b).is_empty(), "{:?}: when u is even the two words are one, which the first arm leaves 6 and the second 5", keeps(&b));
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_store_a_word_that_may_meet_another_differently() {
        let b = arms_store_two_that_may_meet(6);
        assert!(keeps(&b).is_empty(), "{:?}: one arm stores 5 and the other 6 into the second word", keeps(&b));
    }

    #[test]
    fn prove_converts_a_query_whose_arms_store_the_same_value_into_two_words_that_may_meet_in_either_order() {
        let b = arms_store_two_that_may_meet(5);
        assert!(converted(&b).is_empty(), "{:?}: both arms leave 5 in the lane's word and in the word u & 1 rows further, whether or not they are one word", converted(&b));
    }
}

#[cfg(test)]
mod difference_tests {
    use super::super::testing::*;
    use super::*;

    const PREDICATES: [IntPred; 10] = [
        IntPred::Eq,
        IntPred::Ne,
        IntPred::Ult,
        IntPred::Ugt,
        IntPred::Ule,
        IntPred::Uge,
        IntPred::Slt,
        IntPred::Sgt,
        IntPred::Sle,
        IntPred::Sge,
    ];

    #[test]
    fn masked_tests_of_words_hold_only_where_inactive_lanes_see_zero() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let sixteen = b.constant(e, Ty::I32, 16);
        let low = b.cmp(e, IntPred::Ult, lane, sixteen);
        let exec = b.int(e, IntOp::And, low, k.exec);
        let flags = k.buffer(&mut b, e, 8);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        b.br(e, then, vec![exec, flags]);
        let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, then, t[1], lane, 4);
        let yes = b.constant(then, Ty::I1, 1);
        let flag = b.load(then, Space::Global, MemSize::B32, own, yes);
        let on = b.core(then, Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, t[0]));
        let all = b.core(then, Ty::I32, Op::Convert(Cvt::SExt, Ty::I32, t[0]));
        let zero = b.constant(then, Ty::I32, 0);
        let three = b.constant(then, Ty::I32, 3);
        let product = b.int(then, IntOp::Mul, flag, on);
        let masked = b.int(then, IntOp::And, all, flag);
        let shifted = b.int(then, IntOp::Shl, product, three);
        let sum = b.int(then, IntOp::Add, product, masked);
        let plus = b.int(then, IntOp::Add, flag, on);
        let chosen = b.core(then, Ty::I32, Op::Select(t[0], flag, zero));
        let other = b.core(then, Ty::I32, Op::Select(t[0], zero, flag));
        let words = [("flag * zext(exec)", product, true), ("sext(exec) & flag", masked, true), ("(flag * zext(exec)) << 3", shifted, true), ("sum of two masked words", sum, true), ("flag + zext(exec)", plus, false), ("select(exec, flag, 0)", chosen, true), ("select(exec, 0, flag)", other, false), ("flag", flag, false)];
        let mut tests = Vec::new();
        for &(name, w, expected) in &words {
            tests.push((name, b.cmp(then, IntPred::Ne, w, zero), expected));
            tests.push((name, b.cmp(then, IntPred::Ugt, w, zero), expected));
            tests.push((name, b.cmp(then, IntPred::Eq, w, zero), false));
        }
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
        let mut check = Check::new(f, &facts, &b.inputs, Some(0), &loops, &hazards, logic);
        assert!(check.run());
        let wrong: Vec<String> = tests
            .iter()
            .filter(|&&(_, v, expected)| check.masked[v.0] != expected)
            .map(|&(name, v, expected)| format!("{:?} over {}: masked {}, expected {}", f.types[v.0], name, check.masked[v.0], expected))
            .collect();
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn decided_answers_only_what_every_pair_of_values_gives() {
        let mut r = Random::new(29);
        let mut wrong = Vec::new();
        let bases = [0u32, 5, 30, 0x7fff_fff0, 0x8000_0000, 0xffff_fff0];
        for _ in 0..3000 {
            let mut pick = |r: &mut Random| {
                let low = bases[r.below(bases.len() as u64) as usize].wrapping_add(r.below(12) as u32);
                (low, low.saturating_add(r.below(12) as u32))
            };
            let (x, y) = (pick(&mut r), pick(&mut r));
            for p in PREDICATES {
                if let Some(answer) = decided(p, x, y) {
                    let found = (x.0..=x.1).any(|a| (y.0..=y.1).any(|b| compare(p, a, b) != answer));
                    if found {
                        wrong.push(format!("{:?} {:?} {:?} decided {}", p, x, y, answer));
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{} wrong, first {:?}", wrong.len(), &wrong[..wrong.len().min(5)]);
    }

    #[test]
    fn decided_answers_every_pair_of_ranges_one_answer_covers() {
        let mut r = Random::new(31);
        let mut missed = Vec::new();
        for _ in 0..3000 {
            let mut pick = |r: &mut Random| {
                let low = r.below(40) as u32;
                (low, low + r.below(6) as u32)
            };
            let (x, y) = (pick(&mut r), pick(&mut r));
            for p in PREDICATES {
                let all = |answer: bool| (x.0..=x.1).all(|a| (y.0..=y.1).all(|b| compare(p, a, b) == answer));
                let truth = if all(true) { Some(true) } else if all(false) { Some(false) } else { None };
                if truth.is_some() && decided(p, x, y) != truth {
                    missed.push(format!("{:?} {:?} {:?} is always {:?}", p, x, y, truth));
                }
            }
        }
        assert!(missed.is_empty(), "{} missed, first {:?}", missed.len(), &missed[..missed.len().min(5)]);
    }

    #[test]
    fn interval_holds_every_value_the_operations_give() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let byte = b.load(e, Space::Global, MemSize::U8, table, yes);
        let half = b.load(e, Space::Global, MemSize::U16, table, yes);
        let word = b.load(e, Space::Global, MemSize::B32, table, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let c = |b: &mut Build, k: u64| b.constant(e, Ty::I32, k);
        let (three, five, forty) = (c(&mut b, 3), c(&mut b, 5), c(&mut b, 40));
        let mut cases: Vec<(&str, ValueId, Box<dyn Fn(u32, u32, u32, u32) -> u32>)> = Vec::new();
        let mut optional: Vec<(&str, ValueId, Box<dyn Fn(u32, u32, u32, u32) -> u32>)> = Vec::new();
        let v = b.int(e, IntOp::And, word, forty);
        cases.push(("w & 40", v, Box::new(|_, _, w, _| w & 40)));
        let v = b.int(e, IntOp::Or, byte, three);
        cases.push(("b | 3", v, Box::new(|x, _, _, _| x | 3)));
        let v = b.int(e, IntOp::Add, half, lane);
        cases.push(("h + lane", v, Box::new(|_, h, _, l| h + l)));
        let v = b.int(e, IntOp::LShr, half, five);
        cases.push(("h >> 5", v, Box::new(|_, h, _, _| h >> 5)));
        let v = b.int(e, IntOp::LShr, word, forty);
        cases.push(("w >> 40", v, Box::new(|_, _, w, _| w >> (40 & 31))));
        let small = b.cmp(e, IntPred::Ult, word, forty);
        let v = b.core(e, Ty::I32, Op::Select(small, byte, five));
        cases.push(("select(w < 40, b, 5)", v, Box::new(|x, _, w, _| if w < 40 { x } else { 5 })));
        let v = b.core(e, Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, small));
        cases.push(("zext(w < 40)", v, Box::new(|_, _, w, _| (w < 40) as u32)));
        let v = b.core(e, Ty::I32, Op::PopulationCount(word));
        cases.push(("popcount(w)", v, Box::new(|_, _, w, _| w.count_ones())));
        let v = b.core(e, Ty::I32, Op::TrailingZeros(word));
        cases.push(("trailing zeros of w", v, Box::new(|_, _, w, _| w.trailing_zeros())));
        let bit = c(&mut b, 0x100);
        let top = b.int(e, IntOp::Or, half, bit);
        let v = b.int(e, IntOp::Sub, top, byte);
        cases.push(("(h | 256) - b", v, Box::new(|x, h, _, _| (h | 256).wrapping_sub(x))));
        let v = b.int(e, IntOp::Mul, half, forty);
        cases.push(("h * 40", v, Box::new(|_, h, _, _| h.wrapping_mul(40))));
        let v = b.int(e, IntOp::Mul, byte, lane);
        cases.push(("b * lane", v, Box::new(|x, _, _, l| x.wrapping_mul(l))));
        let v = b.int(e, IntOp::Shl, byte, five);
        cases.push(("b << 5", v, Box::new(|x, _, _, _| x << 5)));
        let v = b.int(e, IntOp::Shl, half, forty);
        cases.push(("h << 40", v, Box::new(|_, h, _, _| h << (40 & 31))));
        let v = b.int(e, IntOp::Xor, byte, lane);
        cases.push(("b ^ lane", v, Box::new(|x, _, _, l| x ^ l)));
        let v = b.int(e, IntOp::AShr, half, three);
        cases.push(("h >>> 3", v, Box::new(|_, h, _, _| ((h as i32) >> 3) as u32)));
        let v = b.int(e, IntOp::Sub, byte, half);
        optional.push(("b - h", v, Box::new(|x, h, _, _| x.wrapping_sub(h))));
        let big = c(&mut b, 0x2_0000);
        let v = b.int(e, IntOp::Mul, half, big);
        optional.push(("h * 2^17", v, Box::new(|_, h, _, _| h.wrapping_mul(0x2_0000))));
        let seventeen = c(&mut b, 17);
        let v = b.int(e, IntOp::Shl, half, seventeen);
        optional.push(("h << 17", v, Box::new(|_, h, _, _| h << 17)));
        let v = b.int(e, IntOp::AShr, word, three);
        optional.push(("w >>> 3", v, Box::new(|_, _, w, _| ((w as i32) >> 3) as u32)));
        let one = c(&mut b, 1);
        let bit = b.int(e, IntOp::And, byte, one);
        let small = b.int(e, IntOp::Add, bit, one);
        let odd = c(&mut b, 0x8000_0001);
        let v = b.int(e, IntOp::Mul, small, odd);
        optional.push(("((b & 1) + 1) * 0x80000001", v, Box::new(|x, _, _, _| ((x & 1) + 1).wrapping_mul(0x8000_0001))));
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let mut r = Random::new(37);
        let mut wrong = Vec::new();
        let required = cases.len();
        cases.extend(optional);
        for (i, (name, v, truth)) in cases.iter().enumerate() {
            let Some((low, high)) = interval(&b.f, &facts, *v, 0) else {
                if i < required {
                    wrong.push(format!("{} has no interval", name));
                }
                continue;
            };
            for _ in 0..2000 {
                let w = match r.below(3) {
                    0 => r.below(64) as u32,
                    1 => u32::MAX - r.below(64) as u32,
                    _ => r.next() as u32,
                };
                let t = truth((w & 0xff) as u32, w & 0xffff, w, r.below(32) as u32);
                if t < low || t > high {
                    wrong.push(format!("{} is {} outside [{}, {}]", name, t, low, high));
                    break;
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    type Formula = Box<dyn Fn(&mut Logic, &dyn Fn(ValueId) -> Bdd) -> Bdd>;

    struct Case {
        name: &'static str,
        value: ValueId,
        truth: Formula,
    }

    fn check_differences(b: &Build, kept: &[Choice], cases: &[Case], exact: bool) -> Vec<String> {
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
        let kept: BTreeSet<Choice> = kept.iter().copied().collect();
        let logic = Logic::fixed(f, &facts, &kept, &[]);
        let mut check = Check::new(f, &facts, &b.inputs, Some(0), &loops, &hazards, logic);
        assert!(check.run(), "the program has no store, so nothing can be violated");
        let mut wrong = Vec::new();
        for case in cases {
            let h = if facts.lane_word[case.value.0] && facts.materialized[case.value.0] {
                check.words[case.value.0]
            } else {
                check.h[case.value.0]
            };
            let logic = &mut check.logic;
            let truth = {
                let mut atoms: HashMap<ValueId, Bdd> = HashMap::default();
                for v in 0..f.types.len() {
                    if f.types[v] == Ty::I1 {
                        atoms.insert(ValueId(v), logic.atom(Atom::Bit(ValueId(v))));
                    }
                }
                (case.truth)(logic, &|v| atoms[&v])
            };
            let ok = if exact { logic.m.implies(h, truth) } else { logic.m.implies(truth, h) };
            if !ok {
                wrong.push(case.name.to_string());
            }
        }
        wrong
    }

    struct Program {
        b: Build,
        q: ValueId,
        cases: Vec<Case>,
    }

    fn program() -> Program {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        let yes = b.constant(e, Ty::I1, 1);
        let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let other = b.load(e, Space::Global, MemSize::B32, table, yes);
        let d = b.cmp(e, IntPred::Ult, other, flag);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let five = b.constant(e, Ty::I32, 5);
        let mut cases = Vec::new();
        let exec = k.exec;
        let differs = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
            let c = l.m.and(bit(set), bit(exec));
            let absent = l.m.not(c);
            l.m.and(absent, bit(q))
        };
        cases.push(Case { name: "any(c)", value: q, truth: Box::new(differs) });
        let v = b.core(e, Ty::I32, Op::Select(q, one, two));
        cases.push(Case { name: "select(q, 1, 2)", value: v, truth: Box::new(differs) });
        let v = b.core(e, Ty::I32, Op::Select(q, one, one));
        cases.push(Case { name: "select(q, 1, 1)", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let v = b.int(e, IntOp::And, q, d);
        cases.push(Case {
            name: "q & d",
            value: v,
            truth: Box::new(move |l, bit| {
                let x = differs(l, bit);
                l.m.and(x, bit(d))
            }),
        });
        let v = b.int(e, IntOp::Or, q, d);
        cases.push(Case {
            name: "q | d",
            value: v,
            truth: Box::new(move |l, bit| {
                let x = differs(l, bit);
                let nd = l.m.not(bit(d));
                l.m.and(x, nd)
            }),
        });
        let v = b.int(e, IntOp::Xor, q, d);
        cases.push(Case { name: "q ^ d", value: v, truth: Box::new(differs) });
        let s = b.core(e, Ty::I32, Op::Select(q, one, two));
        let v = b.cmp(e, IntPred::Ult, s, two);
        cases.push(Case { name: "select(q, 1, 2) < 2", value: v, truth: Box::new(differs) });
        let v = b.cmp(e, IntPred::Ult, s, five);
        cases.push(Case { name: "select(q, 1, 2) < 5", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let table2 = k.buffer(&mut b, e, 16);
        let address = b.core(e, Ty::I64, Op::Select(q, table, table2));
        let v = b.load(e, Space::Global, MemSize::B32, address, yes);
        cases.push(Case { name: "load from select(q, a, b)", value: v, truth: Box::new(differs) });
        let v = b.load(e, Space::Global, MemSize::B32, table, q);
        let unperformed = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
            let c = l.m.and(bit(set), bit(exec));
            l.m.not(c)
        };
        cases.push(Case { name: "load masked by q", value: v, truth: Box::new(unperformed) });
        let v = b.load(e, Space::Global, MemSize::B32, table, yes);
        cases.push(Case { name: "unmasked load of a fixed word", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let masked = b.int(e, IntOp::And, q, k.exec);
        let w = b.wave(e, WaveOp::Ballot, vec![masked]);
        let v = b.core(e, Ty::I32, Op::PopulationCount(w));
        cases.push(Case { name: "popcount(ballot(q & exec))", value: v, truth: Box::new(move |_, bit| bit(q)) });
        let kept = b.wave(e, WaveOp::Any, vec![masked]);
        cases.push(Case {
            name: "any(q & exec)",
            value: kept,
            truth: Box::new(move |l, bit| {
                let c = l.m.and(bit(set), bit(exec));
                let absent = l.m.not(c);
                let active = l.m.and(bit(exec), bit(q));
                let answered = l.m.or(bit(kept), active);
                l.m.and(absent, answered)
            }),
        });
        let picked = b.core(e, Ty::I32, Op::Select(q, one, lane));
        let v = b.wave(e, WaveOp::ReadLane, vec![picked, zero, zero]);
        cases.push(Case { name: "readlane(select(q, 1, lane), 0)", value: v, truth: Box::new(move |_, bit| bit(q)) });
        let v = b.int(e, IntOp::Add, flag, one);
        cases.push(Case { name: "flag + 1", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let v = b.core(e, Ty::I32, Op::Select(d, s, s));
        cases.push(Case { name: "select(d, s, s) for s = select(q, 1, 2)", value: v, truth: Box::new(differs) });
        let v = b.int(e, IntOp::And, s, one);
        cases.push(Case { name: "select(q, 1, 2) & 1", value: v, truth: Box::new(differs) });
        let v = b.int(e, IntOp::And, s, zero);
        cases.push(Case { name: "select(q, 1, 2) & 0", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let all = b.constant(e, Ty::I32, 0xffff_ffff);
        let v = b.int(e, IntOp::Or, s, all);
        cases.push(Case { name: "select(q, 1, 2) | ~0", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let v = b.int(e, IntOp::Or, s, two);
        cases.push(Case { name: "select(q, 1, 2) | 2", value: v, truth: Box::new(differs) });
        let v = b.int(e, IntOp::Mul, s, zero);
        cases.push(Case { name: "select(q, 1, 2) * 0", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let v = b.int(e, IntOp::Mul, s, one);
        cases.push(Case { name: "select(q, 1, 2) * 1", value: v, truth: Box::new(differs) });
        Program { b, q, cases }
    }

    #[test]
    fn differences_hold_every_state_where_the_programs_can_disagree() {
        let Program { b, cases, .. } = program();
        let wrong = check_differences(&b, &[], &cases, false);
        assert!(wrong.is_empty(), "differences that miss a disagreement: {:?}", wrong);
    }

    fn choices_program() -> (Build, Vec<Case>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        let yes = b.constant(e, Ty::I1, 1);
        let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let exec = k.exec;
        let differs = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
            let c = l.m.and(bit(set), bit(exec));
            let absent = l.m.not(c);
            l.m.and(absent, bit(q))
        };
        let k = |b: &mut Build, x: u64| b.constant(e, Ty::I32, x);
        let mut cases = Vec::new();
        let (one, two, three, four, five) = (k(&mut b, 1), k(&mut b, 2), k(&mut b, 3), k(&mut b, 4), k(&mut b, 5));
        let s = b.core(e, Ty::I32, Op::Select(q, one, three));
        let v = b.int(e, IntOp::And, s, one);
        cases.push(Case { name: "select(q, 1, 3) & 1", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let s = b.core(e, Ty::I32, Op::Select(q, four, five));
        let v = b.int(e, IntOp::LShr, s, one);
        cases.push(Case { name: "select(q, 4, 5) >> 1", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let s = b.core(e, Ty::I32, Op::Select(q, one, two));
        let v = b.int(e, IntOp::And, s, one);
        cases.push(Case { name: "select(q, 1, 2) & 1", value: v, truth: Box::new(differs) });
        let s = b.core(e, Ty::I32, Op::Select(q, two, four));
        let v = b.int(e, IntOp::LShr, s, one);
        cases.push(Case { name: "select(q, 2, 4) >> 1", value: v, truth: Box::new(differs) });
        (b, cases)
    }

    fn ranges_program() -> (Build, Vec<Case>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        let yes = b.constant(e, Ty::I1, 1);
        let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let exec = k.exec;
        let differs = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
            let c = l.m.and(bit(set), bit(exec));
            let absent = l.m.not(c);
            l.m.and(absent, bit(q))
        };
        let k = |b: &mut Build, x: u64| b.constant(e, Ty::I32, x);
        let mut cases = Vec::new();
        let (one, three, four, five) = (k(&mut b, 1), k(&mut b, 3), k(&mut b, 4), k(&mut b, 5));
        let next = b.int(e, IntOp::Add, flag, one);
        let x = b.core(e, Ty::I32, Op::Select(q, flag, next));
        let v = b.int(e, IntOp::Sub, x, x);
        cases.push(Case { name: "x - x for x = select(q, flag, flag + 1)", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let v = b.int(e, IntOp::Sub, x, flag);
        cases.push(Case { name: "x - flag for x = select(q, flag, flag + 1)", value: v, truth: Box::new(differs) });
        let low = b.int(e, IntOp::And, flag, three);
        let s = b.core(e, Ty::I32, Op::Select(q, low, four));
        let v = b.cmp(e, IntPred::Ult, s, five);
        cases.push(Case { name: "select(q, flag & 3, 4) < 5", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let v = b.cmp(e, IntPred::Ult, s, four);
        cases.push(Case { name: "select(q, flag & 3, 4) < 4", value: v, truth: Box::new(differs) });
        (b, cases)
    }

    fn more_ranges_program() -> (Build, Vec<Case>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        let yes = b.constant(e, Ty::I1, 1);
        let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let exec = k.exec;
        let differs = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
            let c = l.m.and(bit(set), bit(exec));
            let absent = l.m.not(c);
            l.m.and(absent, bit(q))
        };
        let k = |b: &mut Build, x: u64| b.constant(e, Ty::I32, x);
        let (one, two, three, four, eight, nine) = (k(&mut b, 1), k(&mut b, 2), k(&mut b, 3), k(&mut b, 4), k(&mut b, 8), k(&mut b, 9));
        let low = b.int(e, IntOp::And, flag, three);
        let s = b.core(e, Ty::I32, Op::Select(q, low, four));
        let mut cases = Vec::new();
        let doubled = b.int(e, IntOp::Mul, s, two);
        let v = b.cmp(e, IntPred::Ult, doubled, nine);
        cases.push(Case { name: "select(q, flag & 3, 4) * 2 < 9", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let v = b.cmp(e, IntPred::Ult, doubled, eight);
        cases.push(Case { name: "select(q, flag & 3, 4) * 2 < 8", value: v, truth: Box::new(differs) });
        let shifted = b.int(e, IntOp::Shl, s, one);
        let v = b.cmp(e, IntPred::Ult, shifted, nine);
        cases.push(Case { name: "select(q, flag & 3, 4) << 1 < 9", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let flipped = b.int(e, IntOp::Xor, s, one);
        let v = b.cmp(e, IntPred::Ult, flipped, eight);
        cases.push(Case { name: "select(q, flag & 3, 4) ^ 1 < 8", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let left = b.int(e, IntOp::Sub, eight, s);
        let v = b.cmp(e, IntPred::Uge, left, four);
        cases.push(Case { name: "8 - select(q, flag & 3, 4) >= 4", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let v = b.cmp(e, IntPred::Ugt, left, four);
        cases.push(Case { name: "8 - select(q, flag & 3, 4) > 4", value: v, truth: Box::new(differs) });
        (b, cases)
    }

    #[test]
    fn differences_vanish_for_more_operations_the_ranges_of_their_operands_decide() {
        let (b, cases) = more_ranges_program();
        let loose = check_differences(&b, &[], &cases, true);
        assert!(loose.is_empty(), "differences that claim a disagreement that cannot happen: {:?}", loose);
    }

    #[test]
    fn differences_hold_for_more_operations_the_ranges_of_their_operands_leave_open() {
        let (b, cases) = more_ranges_program();
        let missed = check_differences(&b, &[], &cases, false);
        assert!(missed.is_empty(), "differences that miss a possible disagreement: {:?}", missed);
    }

    #[test]
    fn differences_vanish_for_operations_the_ranges_of_their_operands_decide() {
        let (b, cases) = ranges_program();
        let loose = check_differences(&b, &[], &cases, true);
        assert!(loose.is_empty(), "differences that claim a disagreement that cannot happen: {:?}", loose);
    }

    #[test]
    fn differences_hold_for_operations_the_ranges_of_their_operands_leave_open() {
        let (b, cases) = ranges_program();
        let missed = check_differences(&b, &[], &cases, false);
        assert!(missed.is_empty(), "differences that miss a possible disagreement: {:?}", missed);
    }

    #[test]
    fn differences_vanish_for_operations_every_constant_choice_agrees_on() {
        let (b, cases) = choices_program();
        let loose = check_differences(&b, &[], &cases, true);
        assert!(loose.is_empty(), "differences that claim a disagreement that cannot happen: {:?}", loose);
    }

    #[test]
    fn differences_hold_for_operations_the_constant_choices_split() {
        let (b, cases) = choices_program();
        let missed = check_differences(&b, &[], &cases, false);
        assert!(missed.is_empty(), "differences that miss a possible disagreement: {:?}", missed);
    }

    #[test]
    fn differences_hold_only_states_where_the_programs_can_disagree() {
        let Program { b, cases, .. } = program();
        let loose = check_differences(&b, &[], &cases, true);
        assert!(loose.is_empty(), "differences that claim a disagreement that cannot happen: {:?}", loose);
    }

    #[test]
    fn differences_vanish_when_the_query_is_kept() {
        let Program { b, q, cases } = program();
        let (sound, exact): (Vec<String>, Vec<String>) = {
            let only: Vec<Case> = cases
                .into_iter()
                .filter(|c| matches!(c.name, "any(c)" | "select(q, 1, 2)" | "q & d" | "load masked by q"))
                .map(|c| Case {
                    name: c.name,
                    value: c.value,
                    truth: if c.name == "load masked by q" {
                        Box::new(move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| l.m.not(bit(q)))
                    } else {
                        Box::new(|_: &mut Logic, _: &dyn Fn(ValueId) -> Bdd| Bdd::FALSE)
                    },
                })
                .collect();
            (
                check_differences(&b, &[Choice::Query(q)], &only, false),
                check_differences(&b, &[Choice::Query(q)], &only, true),
            )
        };
        assert!(
            sound.is_empty() && exact.is_empty(),
            "with the query kept only an unperformed load differs: {:?} {:?}",
            sound,
            exact
        );
    }
}
