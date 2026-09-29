use super::hazard::Hazards;
use super::address::compare;
use super::logic::{constant_choices, lane_test, projected_word, Atom, Choice, Logic};
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
            dirty_params: BTreeMap::new(),
            dirty_insts,
            queue,
            queued: vec![search; blocks],
            stale: vec![!search; blocks],
            rose: false,
        }
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
                if source.kind.writes() && target.kind.reads() {
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
                        let masked = self.logic.m.implies(formula, active);
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

    fn query(&mut self, x: Bdd, hx: Bdd, tag: Bdd) -> Bdd {
        let whole = self.or(x, hx);
        let others = self.some_lane(whole);
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
                Op::Cmp(pred, x, y) if f.types[x.0] == Ty::I32 && settled(f, facts, pred, x, y) => Bdd::FALSE,
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
                    let read = match constant_choices(f, facts, selector) {
                        Some(lanes) => {
                            let mut any = Bdd::FALSE;
                            for lane in lanes {
                                let there = self.logic.at_lane(hx, lane as u32 & 31);
                                let there = self.some_lane(there);
                                any = self.or(any, there);
                            }
                            any
                        }
                        None => self.some_lane(hx),
                    };
                    self.or(own, read)
                }
                EffectOp::Wave(WaveOp::WriteLane) => {
                    let written = self.any_of(&inputs[..2]);
                    let written = self.some_lane(written);
                    let old = self.any_of(&inputs[2..]);
                    self.or(old, written)
                }
                EffectOp::Wave(op @ (WaveOp::Bpermute | WaveOp::BpermuteFi)) => {
                    let (index, x, mask) = (inputs[0], inputs[1], inputs[2]);
                    let read = if *op == WaveOp::Bpermute {
                        self.masked_read(mask, x)
                    } else {
                        let hx = self.whole(x);
                        self.some_lane(hx)
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

#[derive(Clone, PartialEq, Eq, Hash)]
enum Form {
    Value(ValueId),
    Core(Ty, Op),
    Target(TargetOp, Vec<usize>, usize),
    Load(Space, MemSize, usize),
    Hazard(BlockId, usize, usize),
    Opaque(usize, ValueId),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Desc {
    same: Option<usize>,
    bits: Option<Bdd>,
}

struct Pair {
    cond: Bdd,
    sides: [Vec<Desc>; 2],
}

type Key = [Option<BlockId>; 2];

struct Evaluation {
    side: usize,
    block: BlockId,
    cond: Bdd,
    descs: HashMap<ValueId, Desc>,
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
}

impl<'c, 'a> Explore<'c, 'a> {
    fn new(check: &'c mut Check<'a>, branch: BlockId, assume: Bdd) -> Self {
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
        }
    }

    fn logic(&mut self) -> &mut Logic {
        &mut self.check.logic
    }

    fn fresh(&mut self, side: usize, v: ValueId, position: u32) -> Bdd {
        self.logic().atom(Atom::Fresh(side, v, position))
    }

    fn constant_of(&self, t: usize) -> Option<u64> {
        match self.terms[t] {
            Form::Core(_, Op::Const(_, k)) => Some(k),
            _ => None,
        }
    }

    fn intern(&mut self, ty: Ty, form: Form) -> usize {
        let ones = |ty: Ty| {
            if ty == Ty::I64 {
                u64::MAX
            } else {
                (1u64 << ty.bits()) - 1
            }
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
                for (a, b) in [(x.0, y.0), (y.0, x.0)] {
                    match (k, self.constant_of(a)) {
                        (IntOp::And, Some(0)) => return a,
                        (IntOp::And, Some(c)) if c == ones(ty) => return b,
                        (IntOp::Or | IntOp::Xor, Some(0)) => return b,
                        _ => {}
                    }
                }
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
            _ => {}
        }
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

    fn canon(&mut self, side: usize, param: ValueId, g: Bdd) -> Bdd {
        let support = self.logic().support(g);
        let fresh: Vec<u32> = support
            .iter()
            .copied()
            .filter(|&v| matches!(self.check.logic.atom_of(v), Atom::Fresh(..)))
            .collect();
        if fresh.is_empty() {
            return g;
        }
        let renamed: Vec<Bdd> = (0..fresh.len())
            .map(|i| self.fresh(side, param, i as u32 + 1))
            .collect();
        self.logic().m.compose(g, &|v| {
            fresh.iter().position(|&x| x == v).map(|i| renamed[i])
        })
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
                _ => Some(self.fresh(side, param, 0)),
            },
            _ => None,
        };
        Desc { same, bits }
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
        let mut pairs: BTreeMap<Key, Pair> = BTreeMap::from([(
            first,
            Pair {
                cond: self.assume,
                sides,
            },
        )]);
        let mut worklist = vec![first];
        while let Some(key) = worklist.pop() {
            let pair = &pairs[&key];
            let (cond, sides) = (pair.cond, pair.sides.clone());
            let cond = self.check.and(cond, self.check.safe);
            if cond == Bdd::FALSE {
                continue;
            }
            let side = match key {
                [None, None] => continue,
                [Some(a), Some(b)] if a == b => {
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
            let steps = self.step(side, block, &sides[side], cond);
            if self.check.stopped {
                return;
            }
            for (dst, descs, constraint) in steps {
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
                let merged = match pairs.get(&next) {
                    None => Pair {
                        cond: c,
                        sides: incoming,
                    },
                    Some(old) => {
                        let (ocond, osides) = (old.cond, old.sides.clone());
                        let mcond = self.check.or(ocond, c);
                        let mut grew = mcond != ocond;
                        let mut merged = osides.clone();
                        for s in [WAVE, LANE] {
                            let Some(blk) = next[s] else { continue };
                            if osides[s] == incoming[s] {
                                continue;
                            }
                            for (k, &(param, _)) in f.blocks[&blk].params.iter().enumerate() {
                                let (o, n) = (osides[s][k], incoming[s][k]);
                                if o == n {
                                    continue;
                                }
                                let m = self.merge(s, param, mcond, o, n);
                                if m != o {
                                    merged[s][k] = m;
                                    grew = true;
                                }
                            }
                        }
                        if !grew {
                            continue;
                        }
                        Pair {
                            cond: mcond,
                            sides: merged,
                        }
                    }
                };
                pairs.insert(next, merged);
                if !worklist.contains(&next) {
                    worklist.push(next);
                }
            }
        }
    }

    fn step(
        &mut self,
        side: usize,
        x: BlockId,
        params: &[Desc],
        cond: Bdd,
    ) -> Vec<(Option<BlockId>, Vec<Desc>, Bdd)> {
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
        };
        self.check_effects(&mut ev);
        if self.check.stopped {
            return Vec::new();
        }
        let followed: Vec<(usize, Bdd)> = match &block.term {
            Term::Ret(_) => return vec![(None, Vec::new(), Bdd::TRUE)],
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
            let dst = &f.blocks[&edge.dst];
            let mut args = Vec::with_capacity(edge.args.len());
            for (&arg, &(param, _)) in edge.args.iter().zip(&dst.params) {
                let mut d = match position.get(&arg) {
                    Some(&j) => params[j],
                    None => self.get(&mut ev, arg),
                };
                if let Some(g) = d.bits {
                    if self.has_fresh(g) {
                        d.bits = Some(self.canon(side, param, g));
                    }
                }
                args.push(d);
            }
            out.push((Some(edge.dst), args, constraint));
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
                let desc = match op {
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
                    let same = if self.decide(pred, cond, side) == Some(true) {
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
                    let reason = if ev.side == WAVE {
                        "the wave program may store while the programs are apart"
                    } else {
                        "the lane program may store while the programs are apart"
                    };
                    self.check.require(ev.block, index, reason, ev.cond);
                    if self.check.stopped {
                        return;
                    }
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

fn settled(f: &Func, facts: &Facts, pred: IntPred, x: ValueId, y: ValueId) -> bool {
    let (Some(xs), Some(ys)) = (constant_choices(f, facts, x), constant_choices(f, facts, y)) else {
        return false;
    };
    let answers: Vec<bool> = xs
        .iter()
        .flat_map(|&a| ys.iter().map(move |&b| compare(pred, a as u32, b as u32)))
        .collect();
    answers.iter().all(|&t| t == answers[0])
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
        let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge];
        let bounds = [0u32, 1, 4, 5, 6, 10, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
        let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
        let signed = |p: IntPred| matches!(p, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
        let mut r = Random::new(29);
        let mut wrong = Vec::new();
        for _ in 0..300 {
            let (p1, k1) = (preds[r.below(8) as usize], bounds[r.below(9) as usize]);
            let (p2, k2) = (preds[r.below(8) as usize], bounds[r.below(9) as usize]);
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
                if !possible && !converted && signed(p1) == signed(p2) {
                    wrong.push(format!("{}: v {:?} {} and v {:?} {} never both hold, yet the query stays", name, p1, k1, p2, k2));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
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
        ];
        let kept: Vec<(&str, Vec<&str>)> = cases
            .iter()
            .map(|&(name, a, b)| (name, converted(&arms_compute(a, b).0)))
            .filter(|(_, names)| !names.is_empty())
            .collect();
        assert!(kept.is_empty(), "{:?}: both arms hand the store the same word", kept);
    }

    #[test]
    fn prove_keeps_a_query_whose_arms_compute_different_words() {
        let cases = [("hi(pack(x, y)) and x", 2, 0), ("lo(pack(x, y)) and y", 1, 7), ("lo(pack(x, y) >> 32) and x", 6, 0), ("x and y", 0, 7)];
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
}

#[cfg(test)]
mod difference_tests {
    use super::super::testing::*;
    use super::*;

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
            l.m.not(c)
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
        cases.push(Case { name: "load masked by q", value: v, truth: Box::new(differs) });
        let v = b.load(e, Space::Global, MemSize::B32, table, yes);
        cases.push(Case { name: "unmasked load of a fixed word", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
        let masked = b.int(e, IntOp::And, q, k.exec);
        let w = b.wave(e, WaveOp::Ballot, vec![masked]);
        let v = b.core(e, Ty::I32, Op::PopulationCount(w));
        cases.push(Case { name: "popcount(ballot(q & exec))", value: v, truth: Box::new(|_, _| Bdd::TRUE) });
        let kept = b.wave(e, WaveOp::Any, vec![masked]);
        cases.push(Case { name: "any(q & exec)", value: kept, truth: Box::new(differs) });
        let picked = b.core(e, Ty::I32, Op::Select(q, one, lane));
        let v = b.wave(e, WaveOp::ReadLane, vec![picked, zero, zero]);
        cases.push(Case { name: "readlane(select(q, 1, lane), 0)", value: v, truth: Box::new(|_, _| Bdd::TRUE) });
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
