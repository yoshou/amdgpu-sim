use super::logic::{lane_test, projected_word, Atom, Logic};
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
        logic: Logic,
        mode: Mode,
    ) -> Self {
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

    pub fn everyone(&mut self, kept: &BTreeSet<ValueId>) -> BTreeSet<u64> {
        let demands = std::mem::take(&mut self.demands);
        demands
            .into_iter()
            .filter_map(|(p, condition)| {
                (self.logic.settled(condition, kept) == Bdd::TRUE).then_some(p)
            })
            .collect()
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
            for v in inst.outputs() {
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

    fn query(&mut self, x: Bdd, hx: Bdd, tag: Bdd) -> Bdd {
        let whole = self.or(x, hx);
        let varying: Vec<u32> = self
            .logic
            .support(whole)
            .iter()
            .copied()
            .filter(|&n| !self.logic.uniform_atom(self.facts, n))
            .collect();
        let others = self.logic.exists(&varying, whole);
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
                if x == Bdd::FALSE {
                    return x;
                }
                let varying: Vec<u32> = self
                    .logic
                    .support(x)
                    .iter()
                    .copied()
                    .filter(|&n| !self.logic.uniform_atom(self.facts, n))
                    .collect();
                self.logic.exists(&varying, x)
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
                    let tag = self.logic.tag(w);
                    let local = self.query(fw, hw, tag);
                    let whole = self.whole(w);
                    self.logic.m.ite(mode, whole, local)
                }
                _ => self.any_of(&inst.operands()),
            },
            Inst::Target { args, .. } => self.any_of(args.values()),
            Inst::Packet { .. } => unreachable!("a packet query in a wave program"),
            Inst::Effect {
                provenance,
                op,
                inputs,
                outputs,
            } => match op {
                EffectOp::Wave(WaveOp::Any) => {
                    let (out, x) = (outputs[0].0, inputs[0]);
                    let local = self.logic.local(out);
                    if !self.masked[x.0] {
                        let kept = self.not(local);
                        self.demand(*provenance, kept);
                    }
                    let hx = self.h[x.0];
                    if local == Bdd::FALSE {
                        return hx;
                    }
                    let fx = self.bit(x);
                    let tag = self.logic.tag(out);
                    let answered = self.query(fx, hx, tag);
                    self.logic.m.ite(local, answered, hx)
                }
                EffectOp::Wave(WaveOp::Ballot) => self.h[inputs[0].0],
                EffectOp::Wave(WaveOp::ReadFirstLane) => {
                    if facts.uniform[inputs[0].0] {
                        return self.whole(inputs[0]);
                    }
                    if !self.masked[inputs[1].0] {
                        self.demand(*provenance, Bdd::TRUE);
                    }
                    self.h[inputs[0].0]
                }
                EffectOp::Memory {
                    op: MemoryOp::Load(_),
                    ..
                } => {
                    let fp = self.bit(inputs[1]);
                    let absent = self.not(fp);
                    let operands = self.any_of(&inputs[..2]);
                    self.or(operands, absent)
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
                    let absent = self.not(fp);
                    self.or(operands, absent)
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

const WAVE: usize = 0;
const LANE: usize = 1;

#[derive(Clone, PartialEq, Eq, Hash)]
enum Form {
    Value(ValueId),
    Core(Ty, Op),
    Target(TargetOp, Vec<usize>, usize),
    Load(Space, MemSize, usize),
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
                            Ty::I32 if facts.viewed[value.0] => Some(match k as u32 {
                                0 => Bdd::FALSE,
                                u32::MAX => Bdd::TRUE,
                                k => self.logic().atom(Atom::Constant(k)),
                            }),
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
                    let local = self.check.logic.local(out);
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
                        Some(self.intern(ty, Form::Load(*space, *size, a)))
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
                    let local = self.check.logic.local(outputs[0].0);
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
