use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::engine::{EntryLayout, WORKGROUP_ID_X, WORKGROUP_ID_YZ};
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::hash::HashMap;
type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::rdna_spmd::hash::Mix>>;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

pub const LANES: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Region {
    Allocation(u64),
    Exposed,
    Kernarg,
    Dispatch,
    Lds,
    Private,
}

pub type Unknown = u32;

#[derive(Clone, Debug)]
pub struct UnknownInfo {
    pub shared: bool,
    pub block: BlockId,
    pub rank: usize,
    pub range: Option<(u32, u32)>,
    pub through: Vec<BlockId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Form {
    pub constant: u32,
    pub terms: Vec<(Unknown, u32)>,
}

impl Form {
    pub fn constant(k: u32) -> Self {
        Self {
            constant: k,
            terms: Vec::new(),
        }
    }

    fn unknown(u: Unknown) -> Self {
        Self {
            constant: 0,
            terms: vec![(u, 1)],
        }
    }

    pub fn as_constant(&self) -> Option<u32> {
        self.terms.is_empty().then_some(self.constant)
    }

    fn combine(&self, other: &Form, sign: u32) -> Form {
        let mut terms = Vec::with_capacity(self.terms.len() + other.terms.len());
        let (mut i, mut j) = (0, 0);
        while i < self.terms.len() || j < other.terms.len() {
            let take_self = j == other.terms.len()
                || (i < self.terms.len() && self.terms[i].0 < other.terms[j].0);
            let take_other = i == self.terms.len()
                || (j < other.terms.len() && other.terms[j].0 < self.terms[i].0);
            if take_self {
                terms.push(self.terms[i]);
                i += 1;
            } else if take_other {
                let (u, c) = other.terms[j];
                terms.push((u, c.wrapping_mul(sign)));
                j += 1;
            } else {
                let (u, c) = self.terms[i];
                let sum = c.wrapping_add(other.terms[j].1.wrapping_mul(sign));
                if sum != 0 {
                    terms.push((u, sum));
                }
                i += 1;
                j += 1;
            }
        }
        Form {
            constant: self.constant.wrapping_add(other.constant.wrapping_mul(sign)),
            terms,
        }
    }

    pub fn add(&self, other: &Form) -> Form {
        self.combine(other, 1)
    }

    pub fn sub(&self, other: &Form) -> Form {
        self.combine(other, u32::MAX)
    }

    fn scale(&self, k: u32) -> Form {
        if k == 0 {
            return Form::constant(0);
        }
        Form {
            constant: self.constant.wrapping_mul(k),
            terms: self
                .terms
                .iter()
                .filter_map(|&(u, c)| {
                    let c = c.wrapping_mul(k);
                    (c != 0).then_some((u, c))
                })
                .collect(),
        }
    }

    fn alignment(&self) -> u32 {
        self.terms
            .iter()
            .map(|&(_, c)| c.trailing_zeros())
            .min()
            .unwrap_or(32)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Value {
    pub form: Form,
    pub region: Option<Region>,
}

impl Value {
    fn of(form: Form) -> Self {
        Self { form, region: None }
    }

    fn constant(k: u32) -> Self {
        Self::of(Form::constant(k))
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Value(ValueId, Option<u8>),
    Workgroup(usize),
    Trip(BlockId),
    Derived(ValueId, Vec<(Unknown, u32)>, u32),
    Left(Unknown, u8),
    Cycle(ValueId, u8),
    Guess(ValueId, u8),
    GuessSlot(Slot),
    Held(Slot, usize),
}

enum Entry {
    Value((ValueId, u8, Option<ValueId>)),
    Bit((ValueId, u8, Option<ValueId>)),
    Key(Key),
    Slot(Slot),
    LoopBits((BlockId, u8)),
    Step(StepKey),
    KeptBits(KeptKey),
}

#[derive(Default)]
struct Fixed {
    values: Cached<ValueKey, Assumed<Value>>,
    bits: Cached<ValueKey, Assumed<Option<bool>>>,
    slots: Cached<Slot, Option<Value>>,
    keys: Cached<Key, Unknown>,
    loop_bits: Cached<(BlockId, u8), HashMap<usize, bool>>,
    steps: Cached<StepKey, (Option<u32>, bool)>,
    kept_bits: Cached<KeptKey, Vec<(usize, bool)>>,
}

#[derive(Clone, Debug, Default)]
struct Guesses {
    depth: Depth,
    params: Vec<ValueId>,
    slots: Vec<(u32, u32)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Target {
    Param(usize),
    Slot(u32, u32),
}

const NESTED: usize = 3;

type Depth = usize;
const FREE: Depth = usize::MAX;

type Cached<K, V> = HashMap<K, (V, Depth)>;
type ValueKey = (ValueId, u8, Option<ValueId>);
type StepKey = (BlockId, u8, Target, Option<Region>);
type KeptKey = (BlockId, u8, Vec<(usize, bool)>);
type Edges = Vec<(BlockId, usize)>;

type Slot = (BlockId, u32, u32, u8);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reliance {
    used: bool,
    open: bool,
}

impl std::ops::BitOrAssign for Reliance {
    fn bitor_assign(&mut self, other: Self) {
        self.used |= other.used;
        self.open |= other.open;
    }
}

type Assumed<T> = (T, Reliance);

fn unassumed<T>(x: T) -> Assumed<T> {
    (x, Reliance::default())
}

pub struct Addresses<'a> {
    f: &'a Func,
    facts: &'a Facts,
    inputs: &'a [Parameter],
    exec: u32,
    exec_index: Option<usize>,
    entry: EntryLayout,
    env: &'a Environment,
    headers: BTreeSet<BlockId>,
    rank: HashMap<BlockId, usize>,
    copies: HashMap<ValueId, ValueId>,
    stores: HashMap<BlockId, Vec<Store>>,
    pub unknowns: Vec<UnknownInfo>,
    keys: Cached<Key, Unknown>,
    wave: usize,
    entered: Option<usize>,
    taken: HashSet<(BlockId, usize)>,
    reached: HashSet<BlockId>,
    decided: usize,
    values: Cached<ValueKey, Assumed<Value>>,
    bits: Cached<ValueKey, Assumed<Option<bool>>>,
    fixed: Option<(ValueId, u32)>,
    fixed_values: Cached<ValueKey, Assumed<Value>>,
    fixed_bits: Cached<ValueKey, Assumed<Option<bool>>>,
    fixed_keys: Cached<Key, Unknown>,
    fixed_loop_bits: Cached<(BlockId, u8), HashMap<usize, bool>>,
    fixed_steps: Cached<StepKey, (Option<u32>, bool)>,
    fixed_kept_bits: Cached<KeptKey, Vec<(usize, bool)>>,
    steps: Cached<StepKey, (Option<u32>, bool)>,
    kept_bits: Cached<KeptKey, Vec<(usize, bool)>>,
    loop_bits: Cached<(BlockId, u8), HashMap<usize, bool>>,
    guessing_about: HashMap<(BlockId, u8), Guesses>,
    fixed_store: HashMap<(ValueId, u32), Fixed>,
    users: HashMap<ValueId, Vec<ValueId>>,
    affected: HashMap<ValueId, std::rc::Rc<HashSet<ValueId>>>,
    affected_now: Option<std::rc::Rc<HashSet<ValueId>>>,
    implied: HashMap<ValueId, Vec<ValueId>>,
    assumable: HashSet<ValueId>,
    loops: HashMap<BlockId, Vec<BlockId>>,
    active: HashMap<(ValueId, u8, Option<ValueId>, bool), usize>,
    guessed: HashMap<(ValueId, u8), (Value, Depth)>,
    guessed_bits: HashMap<(ValueId, u8), (bool, Depth)>,
    pending: HashMap<Unknown, Depth>,
    deps: std::cell::RefCell<Vec<Depth>>,
    journal: Vec<(Entry, Depth)>,
    checking: usize,
    guessing: usize,
    slots: Cached<Slot, Option<Value>>,
    fixed_slots: Cached<Slot, Option<Value>>,
    guessed_slots: Cached<Slot, Option<Value>>,
    active_slots: HashMap<Slot, usize>,
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
    ) -> Self {
        let idom = dominators(f, facts);
        let reaches = reaches(f, facts);
        let loops = loops(facts, &idom, &reaches);
        let mut this = Self {
            f,
            facts,
            inputs,
            exec,
            exec_index: inputs.iter().position(
                |p| matches!(p.source, ParameterSource::MaskBit(r) if r == exec),
            ),
            entry,
            env,
            headers,
            rank: facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect(),
            copies: copies(f, facts),
            stores: private_stores(f, facts),
            unknowns: Vec::new(),
            keys: HashMap::default(),
            wave: 0,
            entered: None,
            taken: HashSet::default(),
            reached: HashSet::default(),
            decided: 0,
            values: HashMap::default(),
            bits: HashMap::default(),
            fixed: None,
            fixed_values: HashMap::default(),
            fixed_bits: HashMap::default(),
            fixed_keys: HashMap::default(),
            fixed_loop_bits: HashMap::default(),
            fixed_steps: HashMap::default(),
            fixed_kept_bits: HashMap::default(),
            steps: HashMap::default(),
            kept_bits: HashMap::default(),
            loop_bits: HashMap::default(),
            guessing_about: HashMap::default(),
            fixed_store: HashMap::default(),
            users: users(f, facts),
            affected: HashMap::default(),
            affected_now: None,
            implied: HashMap::default(),
            assumable: HashSet::default(),
            loops,
            active: HashMap::default(),
            guessed: HashMap::default(),
            guessed_bits: HashMap::default(),
            pending: HashMap::default(),
            deps: std::cell::RefCell::new(Vec::new()),
            journal: Vec::new(),
            checking: 0,
            guessing: 0,
            slots: HashMap::default(),
            fixed_slots: HashMap::default(),
            guessed_slots: HashMap::default(),
            active_slots: HashMap::default(),
        };
        for &b in &facts.order {
            for inst in &f.blocks[&b].insts {
                if let Inst::Effect {
                    op: EffectOp::Memory { op, .. },
                    inputs,
                    ..
                } = inst
                {
                    if let Some(&p) = inputs.get(op.mask_input()) {
                        let implied = this.assumptions(p);
                        this.assumable.extend(implied);
                    }
                }
            }
        }
        this
    }

    pub fn waves(&self) -> usize {
        (self.env.workgroup_size() as usize).div_ceil(LANES)
    }

    pub fn enter(&mut self, wave: usize) {
        if self.entered == Some(wave) {
            return;
        }
        self.entered = Some(wave);
        self.wave = wave;
        self.unknowns.clear();
        self.keys.clear();
        self.values.clear();
        self.bits.clear();
        self.fixed_values.clear();
        self.fixed_bits.clear();
        self.fixed_keys.clear();
        self.fixed_loop_bits.clear();
        self.loop_bits.clear();
        self.fixed_steps.clear();
        self.steps.clear();
        self.fixed_kept_bits.clear();
        self.kept_bits.clear();
        self.fixed_store.clear();
        self.slots.clear();
        self.fixed_slots.clear();
        self.taken.clear();
        self.reached.clear();
        self.decided = 0;
        let mut headers: Vec<BlockId> = self.headers.iter().copied().collect();
        headers.sort_by_key(|h| self.rank[h]);
        let trips: Vec<Unknown> = headers.iter().map(|&h| self.trips(h)).collect();
        self.pending = trips.iter().map(|&u| (u, 1)).collect();
        self.sandbox(|this| this.decide_branches());
        for (&header, &u) in headers.iter().zip(&trips) {
            let (last, _) = self.sandbox(|this| this.last_trip(header, u));
            if let Some(last) = last {
                self.unknowns[u as usize].range = Some((0, last));
            }
            self.pending.remove(&u);
        }
    }

    fn decide_branches(&mut self) {
        self.reached.insert(self.f.entry);
        let lanes: Vec<usize> = (0..LANES).filter(|&l| self.valid(l)).collect();
        for (rank, b) in self.facts.order.clone().into_iter().enumerate() {
            self.decided = rank;
            if !self.reached.contains(&b) {
                continue;
            }
            let decided = match self.f.blocks[&b].term {
                Term::CondBr { cond, .. } if self.facts.uniform[cond.0] => {
                    lanes.first().and_then(|&l| self.bit(cond, l, None).0)
                }
                Term::CondBr { cond, .. } => {
                    let mut agreed: Option<Option<bool>> = None;
                    for &l in &lanes {
                        let bit = self.bit(cond, l, None).0;
                        match agreed {
                            None => agreed = Some(bit),
                            Some(old) if old == bit => {}
                            Some(_) => agreed = Some(None),
                        }
                    }
                    agreed.flatten()
                }
                _ => None,
            };
            let edges: Vec<BlockId> = self.f.blocks[&b].term.edges().map(|e| e.dst).collect();
            for (slot, dst) in edges.into_iter().enumerate() {
                if decided.is_none_or(|yes| (slot == 0) == yes) {
                    self.taken.insert((b, slot));
                    self.reached.insert(dst);
                }
            }
        }
        self.decided = self.facts.order.len();
    }

    pub fn reaches_block(&self, b: BlockId) -> bool {
        self.reached.contains(&b)
    }

    fn can_take(&self, pred: BlockId, slot: usize, block: BlockId) -> bool {
        let from = self.rank[&pred];
        if from >= self.rank[&block] {
            return true;
        }
        if from >= self.decided {
            self.depend(self.checking);
            return true;
        }
        self.taken.contains(&(pred, slot))
    }

    pub fn valid(&self, lane: usize) -> bool {
        ((self.wave * LANES + lane) as u32) < self.env.workgroup_size()
    }

    fn ids(&self, lane: usize) -> (u32, u32, u32) {
        let flat = (self.wave * LANES + lane) as u32;
        let [bx, by, _] = self.env.block;
        (flat % bx, (flat / bx) % by, flat / (bx * by))
    }

    fn intern(&mut self, key: Key, mut info: UnknownInfo) -> Unknown {
        let found = match self.fixed {
            Some(_) => self.fixed_keys.get(&key).or_else(|| self.keys.get(&key)),
            None => self.keys.get(&key),
        };
        if let Some(&(u, depth)) = found {
            self.depend(depth);
            return u;
        }
        let u = self.unknowns.len() as Unknown;
        info.rank = self.rank.get(&info.block).copied().unwrap_or(0);
        self.unknowns.push(info);
        let depth = self.current();
        if depth != FREE {
            self.journal.push((Entry::Key(key.clone()), depth));
        }
        if self.fixed.is_some() {
            self.fixed_keys.insert(key, (u, depth));
        } else {
            self.keys.insert(key, (u, depth));
        }
        u
    }

    fn begin(&mut self, key: (ValueId, u8, Option<ValueId>, bool)) -> Option<Option<usize>> {
        match self.active.insert(key, self.guessing) {
            Some(level) if level == self.guessing => None,
            outside => Some(outside),
        }
    }

    fn end(&mut self, key: (ValueId, u8, Option<ValueId>, bool), outside: Option<usize>) {
        match outside {
            Some(level) => self.active.insert(key, level),
            None => self.active.remove(&key),
        };
    }

    fn begin_slot(&mut self, key: Slot) -> Option<Option<usize>> {
        match self.active_slots.insert(key, self.guessing) {
            Some(level) if level == self.guessing => None,
            outside => Some(outside),
        }
    }

    fn depend(&self, depth: Depth) {
        if depth == FREE {
            return;
        }
        if let Some(top) = self.deps.borrow_mut().last_mut() {
            *top = (*top).min(depth);
        }
    }

    fn current(&self) -> Depth {
        self.deps.borrow().last().copied().unwrap_or(FREE)
    }

    fn sandbox<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> (T, Depth) {
        let mark = self.journal.len();
        self.checking += 1;
        let depth = self.checking;
        self.deps.borrow_mut().push(FREE);
        let result = f(self);
        let rests = self.deps.borrow_mut().pop().unwrap_or(FREE);
        self.checking -= 1;
        let entries: Vec<(Entry, Depth)> = self.journal.drain(mark..).collect();
        for (entry, at) in entries {
            if at < depth {
                self.journal.push((entry, at));
                continue;
            }
            match entry {
                Entry::Value(key) => {
                    if self.values.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.values.remove(&key);
                    }
                    if self.fixed_values.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.fixed_values.remove(&key);
                    }
                }
                Entry::Bit(key) => {
                    if self.bits.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.bits.remove(&key);
                    }
                    if self.fixed_bits.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.fixed_bits.remove(&key);
                    }
                }
                Entry::Key(key) => {
                    if self.keys.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.keys.remove(&key);
                    }
                    if self.fixed_keys.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.fixed_keys.remove(&key);
                    }
                }
                Entry::Slot(key) => {
                    if self.slots.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.slots.remove(&key);
                    }
                    if self.fixed_slots.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.fixed_slots.remove(&key);
                    }
                }
                Entry::LoopBits(key) => {
                    if self.loop_bits.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.loop_bits.remove(&key);
                    }
                    if self.fixed_loop_bits.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.fixed_loop_bits.remove(&key);
                    }
                }
                Entry::Step(key) => {
                    if self.steps.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.steps.remove(&key);
                    }
                    if self.fixed_steps.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.fixed_steps.remove(&key);
                    }
                }
                Entry::KeptBits(key) => {
                    if self.kept_bits.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.kept_bits.remove(&key);
                    }
                    if self.fixed_kept_bits.get(&key).is_some_and(|e| e.1 >= depth) {
                        self.fixed_kept_bits.remove(&key);
                    }
                }
            }
        }
        let outer = if rests < depth { rests } else { FREE };
        self.depend(outer);
        (result, outer)
    }

    fn frame<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> (T, Depth) {
        self.deps.borrow_mut().push(FREE);
        let result = f(self);
        let depth = self.deps.borrow_mut().pop().unwrap_or(FREE);
        self.depend(depth);
        (result, depth)
    }

    fn cache_value(&mut self, key: (ValueId, u8, Option<ValueId>), r: Assumed<Value>, depth: Depth) {
        if depth != FREE {
            self.journal.push((Entry::Value(key), depth));
        }
        if self.fixed.is_some() {
            self.fixed_values.insert(key, (r, depth));
        } else {
            self.values.insert(key, (r, depth));
        }
    }

    fn cache_bit(&mut self, key: (ValueId, u8, Option<ValueId>), r: Assumed<Option<bool>>, depth: Depth) {
        if depth != FREE {
            self.journal.push((Entry::Bit(key), depth));
        }
        if self.fixed.is_some() {
            self.fixed_bits.insert(key, (r, depth));
        } else {
            self.bits.insert(key, (r, depth));
        }
    }

    fn block_of(&self, v: ValueId) -> BlockId {
        match self.facts.site[v.0] {
            Site::Param { block, .. } | Site::Inst { block, .. } => block,
            Site::Unreached => self.f.entry,
        }
    }

    fn opaque(&mut self, v: ValueId, lane: usize, range: Option<(u32, u32)>) -> Value {
        let shared = self.facts.uniform[v.0];
        let key = Key::Value(v, (!shared).then_some(lane as u8));
        let block = self.block_of(v);
        let u = self.intern(
            key,
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range,
                through: Vec::new(),
            },
        );
        Value::of(Form::unknown(u))
    }

    pub fn operand(&mut self, x: ValueId, at: BlockId, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
        let (value, reliance) = self.value(x, lane, assume);
        (self.leave(value, at, lane), reliance)
    }

    fn leave(&mut self, value: Value, at: BlockId, lane: usize) -> Value {
        let leaving: Vec<bool> = {
            let here = self.loops.get(&at).map(|l| l.as_slice()).unwrap_or(&[]);
            let leaves = |u: Unknown| self.unknowns[u as usize].through.iter().any(|h| !here.contains(h));
            if !value.form.terms.iter().any(|&(u, _)| leaves(u)) {
                return value;
            }
            value.form.terms.iter().map(|&(u, _)| leaves(u)).collect()
        };
        let mut form = Form::constant(value.form.constant);
        for (&(u, c), leaves) in value.form.terms.iter().zip(leaving) {
            let u = if leaves {
                let info = self.unknowns[u as usize].clone();
                self.intern(
                    Key::Left(u, lane as u8),
                    UnknownInfo {
                        rank: 0,
                        shared: false,
                        block: info.block,
                        range: info.range,
                        through: Vec::new(),
                    },
                )
            } else {
                u
            };
            form = form.add(&Form::unknown(u).scale(c));
        }
        Value {
            form,
            region: value.region,
        }
    }

    fn range(&self, u: Unknown) -> Option<(u32, u32)> {
        if let Some(&depth) = self.pending.get(&u) {
            self.depend(depth);
        }
        self.unknowns[u as usize].range
    }

    fn slot_before(&mut self, at: (BlockId, usize), address: u32, bytes: u32, lane: usize) -> Option<Value> {
        let (block, index) = at;
        let stores: Vec<Store> = self
            .stores
            .get(&block)
            .map(|list| list.iter().filter(|w| w.index < index).rev().copied().collect())
            .unwrap_or_default();
        for w in stores {
            let ran = self.bit(w.predicate, lane, None).0;
            if ran == Some(false) {
                continue;
            }
            let target = self.value(w.address, lane, Some(w.predicate)).0.form.as_constant()?;
            let (t, a) = (target as u64, address as u64);
            if t + w.bytes as u64 <= a || a + bytes as u64 <= t {
                continue;
            }
            if target != address || w.bytes != bytes {
                return None;
            }
            let data = self.value(w.data?, lane, Some(w.predicate)).0;
            if ran == Some(true) {
                return Some(data);
            }
            let before = self.slot_before((block, w.index), address, bytes, lane)?;
            return self.merge_held(vec![before, data], (block, address, bytes, lane as u8), w.index + 1);
        }
        self.slot_entry(block, address, bytes, lane)
    }

    fn slot_entry(&mut self, block: BlockId, address: u32, bytes: u32, lane: usize) -> Option<Value> {
        if block == self.f.entry {
            return None;
        }
        let key: Slot = (block, address, bytes, lane as u8);
        if let Some((guess, depth)) = self.guessed_slots.get(&key).cloned() {
            self.depend(depth);
            return guess;
        }
        if let Some(depth) = self.guessing_about.get(&(block, lane as u8)).map(|g| g.depth) {
            self.depend(depth);
            let symbol = self.symbol(Key::GuessSlot(key), block);
            let guess = Value::of(Form::unknown(symbol));
            self.guessed_slots.insert(key, (Some(guess.clone()), depth));
            if let Some(guesses) = self.guessing_about.get_mut(&(block, lane as u8)) {
                guesses.slots.push((address, bytes));
            }
            return Some(guess);
        }
        let cached = if self.fixed.is_some() {
            self.fixed_slots.get(&key).cloned()
        } else {
            self.slots.get(&key).cloned()
        };
        if let Some((r, depth)) = cached {
            self.depend(depth);
            return r;
        }
        let outside = self.begin_slot(key)?;
        let (result, depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(this.checking);
            }
            this.join_slot(block, address, bytes, lane)
        });
        match outside {
            Some(level) => self.active_slots.insert(key, level),
            None => self.active_slots.remove(&key),
        };
        if depth != FREE {
            self.journal.push((Entry::Slot(key), depth));
        }
        if self.fixed.is_some() {
            self.fixed_slots.insert(key, (result.clone(), depth));
        } else {
            self.slots.insert(key, (result.clone(), depth));
        }
        result
    }

    fn join_slot(&mut self, block: BlockId, address: u32, bytes: u32, lane: usize) -> Option<Value> {
        let (_, back) = self.edges_into(block);
        if back.is_empty() {
            return self.entering_slot(block, address, bytes, lane);
        }
        self.recur(block, lane, Target::Slot(address, bytes))
    }

    fn entering_slot(&mut self, block: BlockId, address: u32, bytes: u32, lane: usize) -> Option<Value> {
        let (entering, _) = self.edges_into(block);
        let mut values: Vec<Value> = Vec::new();
        for (pred, _) in entering {
            let end = self.f.blocks[&pred].insts.len();
            values.push(self.slot_before((pred, end), address, bytes, lane)?);
        }
        self.merge_held(values, (block, address, bytes, lane as u8), 0)
    }

    fn merge_held(&mut self, values: Vec<Value>, slot: Slot, index: usize) -> Option<Value> {
        let first = values.first()?.clone();
        if values.iter().all(|v| *v == first) {
            return Some(first);
        }
        let region = first.region.filter(|r| values.iter().all(|v| v.region == Some(*r)))?;
        let u = self.intern(
            Key::Held(slot, index),
            UnknownInfo {
                rank: 0,
                shared: false,
                block: slot.0,
                range: None,
                through: Vec::new(),
            },
        );
        Some(Value {
            form: Form::unknown(u),
            region: Some(region),
        })
    }

    fn may_guess(&mut self, header: BlockId, lane: usize) -> bool {
        if self.guessing_about.contains_key(&(header, lane as u8)) || self.guessing >= NESTED {
            self.depend(self.checking);
            return false;
        }
        true
    }

    fn guess_about<T>(&mut self, header: BlockId, lane: usize, bits: &HashMap<usize, bool>, f: impl FnOnce(&mut Self) -> T) -> T {
        let l = lane as u8;
        self.sandbox(|this| {
            let depth = this.checking;
            this.guessing += 1;
            let params: Vec<ValueId> = this.f.blocks[&header].params.iter().map(|p| p.0).collect();
            for (&index, &b) in bits {
                this.guessed_bits.insert((params[index], l), (b, depth));
            }
            this.guessing_about.insert(
                (header, l),
                Guesses {
                    depth,
                    ..Guesses::default()
                },
            );
            let result = f(this);
            for &index in bits.keys() {
                this.guessed_bits.remove(&(params[index], l));
            }
            if let Some(guesses) = this.guessing_about.remove(&(header, l)) {
                for v in guesses.params {
                    this.guessed.remove(&(v, l));
                }
                for (address, bytes) in guesses.slots {
                    this.guessed_slots.remove(&(header, address, bytes, l));
                }
            }
            this.guessing -= 1;
            result
        })
        .0
    }

    fn loop_bits(&mut self, header: BlockId, lane: usize) -> HashMap<usize, bool> {
        let key = (header, lane as u8);
        let cached = if self.fixed.is_some() {
            self.fixed_loop_bits.get(&key).cloned()
        } else {
            self.loop_bits.get(&key).cloned()
        };
        if let Some((bits, depth)) = cached {
            self.depend(depth);
            return bits;
        }
        if !self.may_guess(header, lane) {
            return HashMap::default();
        }
        let (bits, depth) = self.frame(|this| {
            let params: Vec<(ValueId, Ty)> = this.f.blocks[&header].params.clone();
            let (entering, back) = this.edges_into(header);
            let mut bits: HashMap<usize, bool> = HashMap::default();
            for (index, &(_, ty)) in params.iter().enumerate() {
                if ty != Ty::I1 {
                    continue;
                }
                let mut first: Option<Option<bool>> = None;
                for &e in &entering {
                    let bit = this.bit(this.edge_arg(e, index), lane, None).0;
                    match first {
                        None => first = Some(bit),
                        Some(old) if old == bit => {}
                        Some(_) => first = Some(None),
                    }
                }
                if let Some(Some(b)) = first {
                    bits.insert(index, b);
                }
            }
            let mut entered: Vec<(usize, bool)> = bits.iter().map(|(&i, &b)| (i, b)).collect();
            entered.sort_unstable();
            let fixpoint_key = (header, lane as u8, entered);
            let cached = if this.fixed.is_some() {
                this.fixed_kept_bits.get(&fixpoint_key).cloned()
            } else {
                this.kept_bits.get(&fixpoint_key).cloned()
            };
            if let Some((kept, depth)) = cached {
                this.depend(depth);
                return kept.into_iter().collect();
            }
            let (bits, depth) = this.frame(|this| this.keep_bits(header, lane, bits, &back));
            let mut kept: Vec<(usize, bool)> = bits.iter().map(|(&i, &b)| (i, b)).collect();
            kept.sort_unstable();
            if depth != FREE {
                this.journal.push((Entry::KeptBits(fixpoint_key.clone()), depth));
            }
            if this.fixed.is_some() {
                this.fixed_kept_bits.insert(fixpoint_key, (kept, depth));
            } else {
                this.kept_bits.insert(fixpoint_key, (kept, depth));
            }
            bits
        });
        if depth != FREE {
            self.journal.push((Entry::LoopBits(key), depth));
        }
        if self.fixed.is_some() {
            self.fixed_loop_bits.insert(key, (bits.clone(), depth));
        } else {
            self.loop_bits.insert(key, (bits.clone(), depth));
        }
        bits
    }

    fn keep_bits(
        &mut self,
        header: BlockId,
        lane: usize,
        mut bits: HashMap<usize, bool>,
        back: &[(BlockId, usize)],
    ) -> HashMap<usize, bool> {
        let this = self;
        {
            loop {
                let guessed = bits.clone();
                let failed: Vec<usize> = this.guess_about(header, lane, &guessed, |this| {
                    guessed
                        .iter()
                        .filter(|&(&index, &b)| {
                            back.iter().any(|&e| {
                                let a = this.edge_arg(e, index);
                                this.bit(a, lane, None).0 != Some(b)
                            })
                        })
                        .map(|(&index, _)| index)
                        .collect()
                });
                if failed.is_empty() {
                    return bits;
                }
                for index in failed {
                    bits.remove(&index);
                }
            }
        }
    }

    fn recur(&mut self, header: BlockId, lane: usize, target: Target) -> Option<Value> {
        if !self.may_guess(header, lane) {
            return None;
        }
        let bits = self.loop_bits(header, lane);
        let (entering, back) = self.edges_into(header);
        let first = match target {
            Target::Param(index) => {
                let mut first: Option<Value> = None;
                for &e in &entering {
                    let value = self.operand(self.edge_arg(e, index), header, lane, None).0;
                    match &first {
                        None => first = Some(value),
                        Some(old) if *old == value => {}
                        Some(_) => return None,
                    }
                }
                first?
            }
            Target::Slot(address, bytes) => self.entering_slot(header, address, bytes, lane)?,
        };
        let l = lane as u8;
        let region = first.region;
        let step_key = (header, l, target, region);
        let cached = if self.fixed.is_some() {
            self.fixed_steps.get(&step_key).cloned()
        } else {
            self.steps.get(&step_key).cloned()
        };
        let (step, keeps) = match cached {
            Some((step, depth)) => {
                self.depend(depth);
                step
            }
            None => {
                let (step, depth) = self.frame(|this| this.find_step(header, lane, target, region, &bits, &back));
                if depth != FREE {
                    self.journal.push((Entry::Step(step_key), depth));
                }
                if self.fixed.is_some() {
                    self.fixed_steps.insert(step_key, (step, depth));
                } else {
                    self.steps.insert(step_key, (step, depth));
                }
                step
            }
        };
        if let Some(step) = step {
            return Some(self.advance(first, step, header));
        }
        let symbol = match target {
            Target::Param(index) => {
                let v = self.f.blocks[&header].params[index].0;
                self.symbol(Key::Guess(v, l), header)
            }
            Target::Slot(address, bytes) => self.symbol(Key::GuessSlot((header, address, bytes, l)), header),
        };
        (keeps && region.is_some()).then(|| Value {
            form: Form::unknown(symbol),
            region,
        })
    }

    fn find_step(
        &mut self,
        header: BlockId,
        lane: usize,
        target: Target,
        region: Option<Region>,
        bits: &HashMap<usize, bool>,
        back: &[(BlockId, usize)],
    ) -> (Option<u32>, bool) {
        let l = lane as u8;
        self.guess_about(header, lane, bits, |this| {
            let depth = this.checking;
            let symbol = match target {
                Target::Param(index) => {
                    let v = this.f.blocks[&header].params[index].0;
                    let symbol = this.symbol(Key::Guess(v, l), header);
                    let guess = Value {
                        form: Form::unknown(symbol),
                        region,
                    };
                    this.guessed.insert((v, l), (guess, depth));
                    if let Some(guesses) = this.guessing_about.get_mut(&(header, l)) {
                        guesses.params.push(v);
                    }
                    symbol
                }
                Target::Slot(address, bytes) => {
                    let key: Slot = (header, address, bytes, l);
                    let symbol = this.symbol(Key::GuessSlot(key), header);
                    let guess = Value {
                        form: Form::unknown(symbol),
                        region,
                    };
                    this.guessed_slots.insert(key, (Some(guess), depth));
                    if let Some(guesses) = this.guessing_about.get_mut(&(header, l)) {
                        guesses.slots.push((address, bytes));
                    }
                    symbol
                }
            };
            let mut step = None;
            let mut stepped = true;
            let mut keeps = true;
            for &e in back {
                let value = match target {
                    Target::Param(index) => {
                        let a = this.edge_arg(e, index);
                        Some(this.operand(a, header, lane, None).0)
                    }
                    Target::Slot(address, bytes) => {
                        let end = this.f.blocks[&e.0].insts.len();
                        this.slot_before((e.0, end), address, bytes, lane)
                    }
                };
                keeps &= value.as_ref().is_some_and(|v| v.region == region);
                stepped = stepped && agree(&mut step, value.as_ref().and_then(|v| added(v, symbol, region)));
                if !stepped && !keeps {
                    break;
                }
            }
            (if stepped { step } else { None }, keeps)
        })
    }

    fn bounds(&self, form: &Form) -> Option<(u64, u64)> {
        let (mut low, mut high) = (form.constant as u64, form.constant as u64);
        for &(u, c) in &form.terms {
            let (l, h) = self.range(u)?;
            low += c as u64 * l as u64;
            high += c as u64 * h as u64;
        }
        (high < 1 << 32).then_some((low, high))
    }

    fn assumes(&mut self, a: ValueId, v: ValueId) -> bool {
        if !self.implied.contains_key(&a) {
            self.assumptions(a);
        }
        self.implied[&a].contains(&v)
    }

    fn assumptions(&mut self, predicate: ValueId) -> Vec<ValueId> {
        if let Some(list) = self.implied.get(&predicate) {
            return list.clone();
        }
        let mut list = Vec::new();
        let mut pending = vec![predicate];
        while let Some(p) = pending.pop() {
            let p = self.copies.get(&p).copied().unwrap_or(p);
            if list.contains(&p) {
                continue;
            }
            list.push(p);
            if let Some(Op::Int(IntOp::And, a, b)) = self.facts.op(self.f, p) {
                if self.f.types[p.0] == Ty::I1 {
                    pending.push(a);
                    pending.push(b);
                }
            }
        }
        self.implied.insert(predicate, list.clone());
        list
    }

    pub fn with_value<T>(&mut self, v: ValueId, k: u32, f: impl FnOnce(&mut Self) -> T) -> T {
        let affected = self.affected_by(v);
        let stored = self.fixed_store.remove(&(v, k)).unwrap_or_default();
        self.fixed_values = stored.values;
        self.fixed_bits = stored.bits;
        self.fixed_slots = stored.slots;
        self.fixed_keys = stored.keys;
        self.fixed_loop_bits = stored.loop_bits;
        self.fixed_steps = stored.steps;
        self.fixed_kept_bits = stored.kept_bits;
        self.fixed = Some((v, k));
        self.affected_now = Some(affected);
        let r = f(self);
        self.fixed = None;
        self.affected_now = None;
        let stored = Fixed {
            values: std::mem::take(&mut self.fixed_values),
            bits: std::mem::take(&mut self.fixed_bits),
            slots: std::mem::take(&mut self.fixed_slots),
            keys: std::mem::take(&mut self.fixed_keys),
            loop_bits: std::mem::take(&mut self.fixed_loop_bits),
            steps: std::mem::take(&mut self.fixed_steps),
            kept_bits: std::mem::take(&mut self.fixed_kept_bits),
        };
        self.fixed_store.insert((v, k), stored);
        r
    }

    fn affected_by(&mut self, v: ValueId) -> std::rc::Rc<HashSet<ValueId>> {
        if let Some(set) = self.affected.get(&v) {
            return set.clone();
        }
        let mut set: HashSet<ValueId> = HashSet::default();
        let mut pending = vec![v];
        let mut memory = false;
        while let Some(x) = pending.pop() {
            if !set.insert(x) {
                continue;
            }
            for &y in self.users.get(&x).map(|u| u.as_slice()).unwrap_or(&[]) {
                if y == PRIVATE_MEMORY {
                    if !memory {
                        memory = true;
                        pending.extend(self.users.get(&PRIVATE_MEMORY).into_iter().flatten().copied());
                    }
                } else {
                    pending.push(y);
                }
            }
        }
        let set = std::rc::Rc::new(set);
        self.affected.insert(v, set.clone());
        set
    }

    fn unaffected(&self, v: ValueId) -> bool {
        self.affected_now.as_ref().is_some_and(|set| !set.contains(&v))
    }

    pub fn program_value(&mut self, u: Unknown) -> Option<ValueId> {
        let v = self.keys.iter().find_map(|(key, &(x, _))| match key {
            Key::Value(v, None) if x == u => Some(*v),
            Key::Derived(v, ..) if x == u => Some(*v),
            _ => None,
        })?;
        let unknown = Form::unknown(u);
        let lanes: Vec<usize> = (0..LANES).filter(|&l| self.valid(l)).collect();
        let everywhere = lanes.into_iter().all(|l| self.value(v, l, None).0.form == unknown);
        everywhere.then_some(v)
    }

    pub fn value(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
        let l = lane as u8;
        if let Some((guess, depth)) = self.guessed.get(&(v, l)).cloned() {
            self.depend(depth);
            return unassumed(guess);
        }
        if let Site::Param { block, index } = self.facts.site[v.0] {
            let carried = |this: &Self| !this.copies.contains_key(&v) && this.incoming(v, block, index).is_none();
            if let Some(depth) = self.guessing_about.get(&(block, l)).map(|g| g.depth).filter(|_| carried(self)) {
                self.depend(depth);
                let symbol = self.symbol(Key::Guess(v, l), block);
                let guess = Value::of(Form::unknown(symbol));
                self.guessed.insert((v, l), (guess.clone(), depth));
                if let Some(guesses) = self.guessing_about.get_mut(&(block, l)) {
                    guesses.params.push(v);
                }
                return unassumed(guess);
            }
        }
        if self.unaffected(v) {
            let (fixed, affected) = (self.fixed.take(), self.affected_now.take());
            let r = self.value(v, lane, assume);
            (self.fixed, self.affected_now) = (fixed, affected);
            return r;
        }
        let hit = if let Some((fixed, k)) = self.fixed {
            if fixed == v {
                return unassumed(Value::constant(k));
            }
            self.fixed_values.get(&(v, l, assume)).cloned()
        } else {
            match self.values.get(&(v, l, None)) {
                Some(e) if assume.is_none() || !e.0 .1.open => Some(e.clone()),
                _ => assume.and_then(|_| self.values.get(&(v, l, assume)).cloned()),
            }
        };
        if let Some((r, depth)) = hit {
            self.depend(depth);
            return r;
        }
        let key = (v, l, assume, false);
        let Some(outside) = self.begin(key) else {
            let shared = self.facts.uniform[v.0];
            let block = self.block_of(v);
            let u = self.intern(
                Key::Cycle(v, if shared { 0 } else { l }),
                UnknownInfo {
                    rank: 0,
                    shared,
                    block,
                    range: None,
                    through: Vec::new(),
                },
            );
            return unassumed(Value::of(Form::unknown(u)));
        };
        let (r, depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(this.checking);
            }
            this.compute(v, lane, assume)
        });
        self.end(key, outside);
        if self.fixed.is_some() {
            self.cache_value((v, l, assume), r.clone(), depth);
            return r;
        }
        if assume.is_some() {
            self.cache_value((v, l, assume), r.clone(), depth);
        }
        if !r.1.used {
            self.cache_value((v, l, None), r.clone(), depth);
        }
        r
    }

    pub fn bit(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
        if let Some(a) = assume {
            let root = self.copies.get(&v).copied().unwrap_or(v);
            if self.assumes(a, root) {
                return (
                    Some(true),
                    Reliance {
                        used: true,
                        open: false,
                    },
                );
            }
        }
        let r = self.bit_unless_assumed(v, lane, assume);
        let root = self.copies.get(&v).copied().unwrap_or(v);
        if r.0.is_none() && self.assumable.contains(&root) {
            return (r.0, Reliance { open: true, ..r.1 });
        }
        r
    }

    fn bit_unless_assumed(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
        let l = lane as u8;
        if let Some(&(guess, depth)) = self.guessed_bits.get(&(v, l)) {
            self.depend(depth);
            return unassumed(Some(guess));
        }
        if self.unaffected(v) {
            let (fixed, affected) = (self.fixed.take(), self.affected_now.take());
            let r = self.bit_unless_assumed(v, lane, assume);
            (self.fixed, self.affected_now) = (fixed, affected);
            return r;
        }
        let hit = if self.fixed.is_some() {
            self.fixed_bits.get(&(v, l, assume)).copied()
        } else {
            match self.bits.get(&(v, l, None)) {
                Some(&e) if assume.is_none() || !e.0 .1.open => Some(e),
                _ => assume.and_then(|_| self.bits.get(&(v, l, assume)).copied()),
            }
        };
        if let Some((r, depth)) = hit {
            self.depend(depth);
            return r;
        }
        let key = (v, l, assume, true);
        let Some(outside) = self.begin(key) else {
            return unassumed(None);
        };
        let (r, depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(this.checking);
            }
            this.compute_bit(v, lane, assume)
        });
        self.end(key, outside);
        if self.fixed.is_some() {
            self.cache_bit((v, l, assume), r, depth);
            return r;
        }
        if assume.is_some() {
            self.cache_bit((v, l, assume), r, depth);
        }
        if !r.1.used {
            self.cache_bit((v, l, None), r, depth);
        }
        r
    }

    fn compute(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
        let (f, facts) = (self.f, self.facts);
        if let Some(&root) = self.copies.get(&v) {
            return self.value(root, lane, assume);
        }
        match facts.site[v.0] {
            Site::Param { block, index } if block == f.entry => unassumed(self.input(v, index, lane)),
            Site::Param { block, index } => self.join(v, block, index, lane),
            Site::Inst { block, index } => match &f.blocks[&block].insts[index] {
                Inst::Core { op, .. } => self.core(v, *op, lane, assume),
                Inst::Effect {
                    op, inputs, outputs, ..
                } => self.effect(v, *op, inputs, outputs, lane, assume),
                _ => unassumed(self.opaque(v, lane, None)),
            },
            Site::Unreached => unassumed(self.opaque(v, lane, None)),
        }
    }

    fn input(&mut self, v: ValueId, index: usize, lane: usize) -> Value {
        match self.inputs[index].source {
            ParameterSource::Vgpr(0) => {
                let (x, y, z) = self.ids(lane);
                Value::constant(x | y << 10 | z << 20)
            }
            ParameterSource::Sgpr(n) if Some(n) == self.entry.kernarg_ptr => Value {
                form: Form::constant(0),
                region: Some(Region::Kernarg),
            },
            ParameterSource::Sgpr(n) if Some(n) == self.entry.dispatch_ptr => Value {
                form: Form::constant(0),
                region: Some(Region::Dispatch),
            },
            ParameterSource::Sgpr(WORKGROUP_ID_X) if self.entry.workgroup_id_x => {
                Value::of(Form::unknown(self.workgroup(0)))
            }
            ParameterSource::Sgpr(WORKGROUP_ID_YZ) if self.entry.workgroup_id_yz => {
                let (y, z) = (self.workgroup(1), self.workgroup(2));
                Value::of(Form::unknown(y).add(&Form::unknown(z).scale(1 << 16)))
            }
            _ => self.opaque(v, lane, None),
        }
    }

    fn workgroup(&mut self, axis: usize) -> Unknown {
        let count = self.env.grid[axis].max(1);
        let entry = self.f.entry;
        self.intern(
            Key::Workgroup(axis),
            UnknownInfo {
                rank: 0,
                shared: true,
                block: entry,
                range: Some((0, count - 1)),
                through: Vec::new(),
            },
        )
    }

    fn incoming(&self, v: ValueId, block: BlockId, index: usize) -> Option<Vec<ValueId>> {
        let header = self.headers.contains(&block);
        let own = self.rank[&block];
        let mut out = Vec::new();
        for &(pred, slot) in &self.facts.incoming[&block] {
            if !self.can_take(pred, slot, block) {
                continue;
            }
            let arg = self.f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
            if header && self.rank[&pred] >= own {
                if arg != v {
                    return None;
                }
                continue;
            }
            out.push(arg);
        }
        Some(out)
    }

    fn join(&mut self, v: ValueId, block: BlockId, index: usize, lane: usize) -> Assumed<Value> {
        let Some(arguments) = self.incoming(v, block, index) else {
            if let Some(value) = self.stepped(v, block, index, lane) {
                return unassumed(value);
            }
            if let Some(value) = self.recur(block, lane, Target::Param(index)) {
                return unassumed(value);
            }
            let range = self.induction(v, block, index, lane);
            return unassumed(self.opaque(v, lane, range));
        };
        let mut joined: Option<Value> = None;
        let mut region: Option<Option<Region>> = None;
        let mut agreed = true;
        for a in arguments {
            let (value, _) = self.operand(a, block, lane, None);
            region = Some(match region {
                None => value.region,
                Some(r) if r == value.region => r,
                Some(_) => None,
            });
            match &joined {
                None => joined = Some(value),
                Some(old) if *old == value => {}
                Some(_) => agreed = false,
            }
        }
        if !agreed {
            return unassumed(Value {
                region: region.flatten(),
                ..self.opaque(v, lane, None)
            });
        }
        unassumed(joined.unwrap_or_else(|| self.opaque(v, lane, None)))
    }

    fn core(&mut self, v: ValueId, op: Op, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
        let ty = self.f.types[v.0];
        let wide = ty == Ty::I64;
        let at = self.block_of(v);
        let mut used = Reliance::default();
        macro_rules! get {
            ($this:expr, $x:expr) => {{
                let (value, u) = $this.operand($x, at, lane, assume);
                used |= u;
                value
            }};
        }
        let value = match op {
            Op::Const(_, k) => Value::constant(k as u32),
            Op::Env(Env::LaneId) => Value::constant(lane as u32),
            Op::Env(Env::ScratchBase) => Value {
                form: Form::constant(0),
                region: Some(Region::Private),
            },
            Op::Int(IntOp::Add, a, b) => {
                let (a, b) = (get!(self, a), get!(self, b));
                let region = match (a.region, b.region) {
                    (Some(r), None) | (None, Some(r)) => Some(r),
                    _ => None,
                };
                Value {
                    form: a.form.add(&b.form),
                    region,
                }
            }
            Op::Int(IntOp::Sub, a, b) => {
                let (a, b) = (get!(self, a), get!(self, b));
                let region = match (a.region, b.region) {
                    (Some(r), None) => Some(r),
                    _ => None,
                };
                Value {
                    form: a.form.sub(&b.form),
                    region,
                }
            }
            Op::Int(IntOp::Mul, a, b) => {
                let (a, b) = (get!(self, a), get!(self, b));
                match (a.form.as_constant(), b.form.as_constant()) {
                    (Some(k), _) => Value::of(b.form.scale(k)),
                    (_, Some(k)) => Value::of(a.form.scale(k)),
                    _ => self.opaque(v, lane, None),
                }
            }
            Op::Int(IntOp::Shl, a, s) => {
                let (a, s) = (get!(self, a), get!(self, s));
                match s.form.as_constant() {
                    Some(k) if k < 32 => Value::of(a.form.scale(1 << k)),
                    Some(k) if wide && k < 64 => Value::constant(0),
                    _ => self.opaque(v, lane, None),
                }
            }
            Op::Int(IntOp::LShr, a, s) if !wide => {
                let (a, s) = (get!(self, a), get!(self, s));
                match s.form.as_constant() {
                    Some(0) => a,
                    Some(k) if k < 32 => self.shift(v, &a.form, k, lane),
                    _ => self.opaque(v, lane, None),
                }
            }
            Op::Int(IntOp::AShr, a, s) if !wide => {
                let (a, s) = (get!(self, a), get!(self, s));
                match (a.form.as_constant(), s.form.as_constant()) {
                    (Some(x), Some(k)) if k < 32 => Value::constant(((x as i32) >> k) as u32),
                    _ => self.opaque(v, lane, None),
                }
            }
            Op::Int(IntOp::And, a, b) => {
                let (a, b) = (get!(self, a), get!(self, b));
                match (a.form.as_constant(), b.form.as_constant()) {
                    (Some(x), Some(y)) => Value::constant(x & y),
                    (Some(m), None) | (None, Some(m)) => {
                        let form = if a.form.as_constant().is_some() { &b.form } else { &a.form };
                        match low_bits(form, m, IntOp::And) {
                            Some(result) => Value::of(result),
                            None => self.mask(v, form, m, lane),
                        }
                    }
                    _ => self.opaque(v, lane, None),
                }
            }
            Op::Int(k @ (IntOp::Or | IntOp::Xor), a, b) => {
                let (a, b) = (get!(self, a), get!(self, b));
                match (a.form.as_constant(), b.form.as_constant()) {
                    (Some(x), Some(y)) => Value::constant(if k == IntOp::Or { x | y } else { x ^ y }),
                    (_, Some(u32::MAX)) if k == IntOp::Xor && !wide => {
                        Value::of(Form::constant(u32::MAX).sub(&a.form))
                    }
                    (Some(u32::MAX), _) if k == IntOp::Xor && !wide => {
                        Value::of(Form::constant(u32::MAX).sub(&b.form))
                    }
                    _ if self.disjoint_bits(&a.form, &b.form) => Value {
                        form: a.form.add(&b.form),
                        region: a.region.or(b.region),
                    },
                    (None, Some(c)) if low_bits(&a.form, c, k).is_some() => Value {
                        form: low_bits(&a.form, c, k).unwrap(),
                        region: a.region,
                    },
                    (Some(c), None) if low_bits(&b.form, c, k).is_some() => Value {
                        form: low_bits(&b.form, c, k).unwrap(),
                        region: b.region,
                    },
                    _ => self.opaque(v, lane, None),
                }
            }
            Op::Select(c, a, b) => {
                let (bit, u) = self.bit(c, lane, assume);
                used |= u;
                match bit {
                    Some(true) => get!(self, a),
                    Some(false) => get!(self, b),
                    None => {
                        let (x, y) = (get!(self, a), get!(self, b));
                        if x == y {
                            x
                        } else {
                            let region = if x.region == y.region { x.region } else { None };
                            Value {
                                region,
                                ..self.opaque(v, lane, None)
                            }
                        }
                    }
                }
            }
            Op::Convert(Cvt::ZExt | Cvt::SExt, _, a) if self.f.types[a.0] == Ty::I1 => {
                let (bit, u) = self.bit(a, lane, assume);
                used |= u;
                match bit {
                    Some(b) => Value::constant(b as u32),
                    None => self.opaque(v, lane, Some((0, 1))),
                }
            }
            Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, to, a)
                if to.bits() >= 32 && self.f.types[a.0].bits() >= 32 =>
            {
                get!(self, a)
            }
            Op::Pack64(lo, _) | Op::UnpackLo(lo) => get!(self, lo),
            Op::UnpackHi(x) => match self.facts.op(self.f, x) {
                Some(Op::Pack64(_, hi)) => get!(self, hi),
                Some(Op::Convert(Cvt::ZExt, _, a)) if self.f.types[a.0] == Ty::I32 => {
                    Value::constant(0)
                }
                _ => self.opaque(v, lane, None),
            },
            Op::TrailingZeros(a) | Op::LeadingZeros(a) | Op::PopulationCount(a) | Op::ReverseBits(a)
                if !wide =>
            {
                match get!(self, a).form.as_constant() {
                    Some(x) => Value::constant(match op {
                        Op::TrailingZeros(_) => x.trailing_zeros(),
                        Op::LeadingZeros(_) => x.leading_zeros(),
                        Op::PopulationCount(_) => x.count_ones(),
                        _ => x.reverse_bits(),
                    }),
                    None => self.opaque(v, lane, None),
                }
            }
            _ => self.opaque(v, lane, None),
        };
        (value, used)
    }

    fn through(&self, form: &Form) -> Vec<BlockId> {
        let mut out: Vec<BlockId> = Vec::new();
        for &(u, _) in &form.terms {
            for &h in &self.unknowns[u as usize].through {
                if !out.contains(&h) {
                    out.push(h);
                }
            }
        }
        out
    }

    fn shift(&mut self, v: ValueId, form: &Form, k: u32, lane: usize) -> Value {
        if let Some(x) = form.as_constant() {
            return Value::constant(x >> k);
        }
        if form.terms.iter().any(|&(u, _)| !self.unknowns[u as usize].shared) {
            return self.opaque(v, lane, None);
        }
        let Some((_, high)) = self.bounds(form) else {
            return self.opaque(v, lane, None);
        };
        let j = form.alignment();
        if k <= j {
            return Value::of(Form {
                constant: form.constant >> k,
                terms: form.terms.iter().map(|&(u, c)| (u, c >> k)).collect(),
            });
        }
        let above = form.constant >> j;
        let block = self.block_of(v);
        let through = self.through(form);
        let u = self.intern(
            Key::Derived(v, form.terms.clone(), above),
            UnknownInfo {
                rank: 0,
                shared: true,
                block,
                range: Some((0, (high >> k) as u32)),
                through,
            },
        );
        Value::of(Form::unknown(u))
    }

    fn mask(&mut self, v: ValueId, form: &Form, m: u32, lane: usize) -> Value {
        if m == u32::MAX {
            return Value::of(form.clone());
        }
        if let Some((_, high)) = self.bounds(form) {
            let j = form.alignment().min(32);
            let low = if j >= 32 { form.constant } else { form.constant & ((1u32 << j) - 1) };
            let top = 64 - high.leading_zeros();
            let upto = if top >= 32 { u32::MAX } else { (1u32 << top) - 1 };
            let above = if j >= 32 { 0 } else { upto & !((1u32 << j) - 1) };
            if above & !m == 0 {
                return Value::of(form.sub(&Form::constant(low & !m)));
            }
        }
        let low = m.wrapping_add(1).is_power_of_two() || m == u32::MAX;
        if low {
            let k = m.count_ones();
            let kept: Vec<(Unknown, u32)> = form
                .terms
                .iter()
                .copied()
                .filter(|&(_, c)| c.trailing_zeros() < k)
                .collect();
            let rest = Form {
                constant: form.constant,
                terms: kept,
            };
            if let Some(x) = rest.as_constant() {
                return Value::constant(x & m);
            }
            if let Some((_, high)) = self.bounds(&rest) {
                if high <= m as u64 {
                    return Value::of(rest);
                }
            }
            let j = rest.alignment();
            if rest.terms.iter().all(|&(u, _)| self.unknowns[u as usize].shared) {
                let above = form.constant >> j;
                let block = self.block_of(v);
                let through = self.through(&rest);
                let u = self.intern(
                    Key::Derived(v, rest.terms.clone(), above),
                    UnknownInfo {
                        rank: 0,
                        shared: true,
                        block,
                        range: Some((0, m >> j)),
                        through,
                    },
                );
                let below = form.constant & ((1u32 << j) - 1);
                return Value::of(Form::unknown(u).scale(1 << j).add(&Form::constant(below)));
            }
            return self.opaque(v, lane, Some((0, m)));
        }
        let high = !m;
        if high.wrapping_add(1).is_power_of_two() {
            let k = high.count_ones();
            if form.alignment() >= k {
                return Value::of(form.sub(&Form::constant(form.constant & high)));
            }
        }
        self.opaque(v, lane, Some((0, m)))
    }

    fn disjoint_bits(&self, a: &Form, b: &Form) -> bool {
        let fits = |this: &Self, low: &Form, high: &Form| {
            let zeros = high.alignment().min(high.constant.trailing_zeros());
            match this.bounds(low) {
                Some((_, top)) => zeros >= 32 || top < 1u64 << zeros,
                None => false,
            }
        };
        fits(self, a, b) || fits(self, b, a)
    }

    fn effect(
        &mut self,
        v: ValueId,
        op: EffectOp,
        inputs: &[ValueId],
        outputs: &[(ValueId, Ty)],
        lane: usize,
        assume: Option<ValueId>,
    ) -> Assumed<Value> {
        let _ = outputs;
        match op {
            EffectOp::Memory {
                op: MemoryOp::Load(size),
                ..
            } => {
                let at = self.block_of(v);
                let (address, used) = self.operand(inputs[0], at, lane, assume);
                (self.load(v, &address, size, lane), used)
            }
            EffectOp::Wave(WaveOp::ReadFirstLane) => unassumed(self.uniform_read(v, inputs[0], None)),
            EffectOp::Wave(WaveOp::ReadLane) => {
                let (selector, _) = self.value(inputs[1], lane, None);
                let chosen = selector.form.as_constant().map(|k| (k & 31) as usize);
                unassumed(self.uniform_read(v, inputs[0], chosen))
            }
            EffectOp::Wave(WaveOp::WriteLane) => {
                let (selector, _) = self.value(inputs[1], lane, None);
                match selector.form.as_constant() {
                    Some(k) if (k & 31) as usize == lane => {
                        let at = self.block_of(v);
                        unassumed(self.operand(inputs[0], at, lane, None).0)
                    }
                    Some(_) => {
                        let at = self.block_of(v);
                        unassumed(self.operand(inputs[2], at, lane, None).0)
                    }
                    None => unassumed(self.opaque(v, lane, None)),
                }
            }
            EffectOp::Wave(WaveOp::Ballot) => {
                let mut mask = 0u32;
                for l in 0..LANES {
                    if !self.valid(l) {
                        continue;
                    }
                    match self.bit(inputs[0], l, None).0 {
                        Some(true) => mask |= 1 << l,
                        Some(false) => {}
                        None => return unassumed(self.opaque(v, lane, None)),
                    }
                }
                unassumed(Value::constant(mask))
            }
            _ => unassumed(self.opaque(v, lane, None)),
        }
    }

    fn uniform_read(&mut self, v: ValueId, x: ValueId, from: Option<usize>) -> Value {
        let lanes: Vec<usize> = match from {
            Some(l) => vec![l],
            None => (0..LANES).filter(|&l| self.valid(l)).collect(),
        };
        let mut agreed: Option<Value> = None;
        let at = self.block_of(v);
        for l in lanes {
            let (value, _) = self.operand(x, at, l, None);
            if agreed.as_ref().is_some_and(|a| *a != value) {
                return self.uniform(v);
            }
            agreed = Some(value);
        }
        agreed.unwrap_or_else(|| self.uniform(v))
    }

    fn uniform(&mut self, v: ValueId) -> Value {
        let block = self.block_of(v);
        let u = self.intern(
            Key::Value(v, None),
            UnknownInfo {
                rank: 0,
                shared: true,
                block,
                range: None,
                through: Vec::new(),
            },
        );
        Value::of(Form::unknown(u))
    }

    fn load(&mut self, v: ValueId, address: &Value, size: MemSize, lane: usize) -> Value {
        let offset = address.form.as_constant();
        let bytes = size.bytes().min(4);
        if let Site::Inst { block, index } = self.facts.site[v.0] {
            if let Inst::Effect {
                op:
                    EffectOp::Memory {
                        space: Space::Scratch,
                        ..
                    },
                ..
            } = &self.f.blocks[&block].insts[index]
            {
                if let (MemSize::B32, Some(a)) = (size, address.form.as_constant()) {
                    if let Some(value) = self.slot_before((block, index), a, bytes, lane) {
                        return self.leave(value, block, lane);
                    }
                }
                return self.opaque(v, lane, None);
            }
        }
        match (address.region, offset) {
            (Some(Region::Kernarg), Some(at)) => {
                if bytes == 4 {
                    if let Some(binding) = self.env.binding(at) {
                        return Value {
                            form: Form::constant(binding.pointer as u32),
                            region: Some(Region::Allocation(binding.allocation)),
                        };
                    }
                    if let Some(binding) = at.checked_sub(4).and_then(|o| self.env.binding(o)) {
                        return Value::constant((binding.pointer >> 32) as u32);
                    }
                }
                let word = self.env.kernarg_word(at, bytes);
                Value::constant(extend(word, size))
            }
            (Some(Region::Dispatch), Some(at)) => match self.dispatch_word(at, bytes) {
                Some(word) => Value::constant(extend(word, size)),
                None => self.opaque(v, lane, None),
            },
            _ => {
                let range = match size {
                    MemSize::U8 => Some((0, 0xff)),
                    MemSize::U16 => Some((0, 0xffff)),
                    _ => None,
                };
                self.opaque(v, lane, range)
            }
        }
    }

    fn dispatch_word(&self, at: u32, bytes: u32) -> Option<u32> {
        let [bx, by, bz] = self.env.block;
        let [gx, gy, gz] = self.env.grid;
        let mut packet = [0u8; 24];
        for (k, n) in [bx, by, bz].iter().enumerate() {
            packet[4 + 2 * k..6 + 2 * k].copy_from_slice(&(*n as u16).to_le_bytes());
        }
        for (k, n) in [gx * bx, gy * by, gz * bz].iter().enumerate() {
            packet[12 + 4 * k..16 + 4 * k].copy_from_slice(&n.to_le_bytes());
        }
        let at = at as usize;
        let end = at + bytes as usize;
        (end <= packet.len() && at >= 4).then(|| {
            let mut word = [0u8; 4];
            word[..bytes as usize].copy_from_slice(&packet[at..end]);
            u32::from_le_bytes(word)
        })
    }

    fn compute_bit(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
        let (f, facts) = (self.f, self.facts);
        if let Some(&root) = self.copies.get(&v) {
            return self.bit(root, lane, assume);
        }
        match facts.site[v.0] {
            Site::Param { block, index } if block == f.entry => {
                let valid = self.valid(lane);
                match self.inputs[index].source {
                    ParameterSource::MaskBit(r) if r == self.exec => unassumed(Some(valid)),
                    _ => unassumed(None),
                }
            }
            Site::Param { block, index } => {
                let Some(arguments) = self.incoming(v, block, index) else {
                    if let Some(&bit) = self.loop_bits(block, lane).get(&index) {
                        return unassumed(Some(bit));
                    }
                    return unassumed(self.narrowed(v, block, index, lane));
                };
                let mut joined: Option<Option<bool>> = None;
                for a in arguments {
                    let (bit, _) = self.bit(a, lane, None);
                    match joined {
                        None => joined = Some(bit),
                        Some(old) if old == bit => {}
                        Some(_) => return unassumed(None),
                    }
                }
                unassumed(joined.flatten())
            }
            Site::Inst { block, index } => match f.blocks[&block].insts[index] {
                Inst::Core { op, .. } => self.core_bit(op, lane, assume),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any),
                    ref inputs,
                    ..
                } => {
                    let input = inputs[0];
                    let mut unknown = false;
                    for l in 0..LANES {
                        if !self.valid(l) {
                            continue;
                        }
                        match self.bit(input, l, None).0 {
                            Some(true) => return unassumed(Some(true)),
                            Some(false) => {}
                            None => unknown = true,
                        }
                    }
                    unassumed((!unknown).then_some(false))
                }
                _ => unassumed(None),
            },
            Site::Unreached => unassumed(None),
        }
    }

    fn core_bit(&mut self, op: Op, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
        let mut used = Reliance::default();
        macro_rules! bit {
            ($this:expr, $x:expr) => {{
                let (b, u) = $this.bit($x, lane, assume);
                used |= u;
                b
            }};
        }
        let result = match op {
            Op::Const(_, k) => Some(k & 1 != 0),
            Op::Env(Env::ValidLane) => Some(self.valid(lane)),
            Op::Int(IntOp::And, a, b) if self.aperture(a, b, lane).is_some() => {
                self.aperture(a, b, lane)
            }
            Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                let x = bit!(self, a);
                if k == IntOp::And && x == Some(false) {
                    Some(false)
                } else if k == IntOp::Or && x == Some(true) {
                    Some(true)
                } else {
                    let y = bit!(self, b);
                    match (k, x, y) {
                        (IntOp::And, _, Some(false)) => Some(false),
                        (IntOp::Or, _, Some(true)) => Some(true),
                        (_, Some(x), Some(y)) => Some(match k {
                            IntOp::And => x && y,
                            IntOp::Or => x || y,
                            _ => x != y,
                        }),
                        _ => None,
                    }
                }
            }
            Op::Select(c, a, b) => match bit!(self, c) {
                Some(true) => bit!(self, a),
                Some(false) => bit!(self, b),
                None => {
                    let (x, y) = (bit!(self, a), bit!(self, b));
                    if x == y {
                        x
                    } else {
                        None
                    }
                }
            },
            Op::Convert(Cvt::Trunc, Ty::I1, x) | Op::Convert(Cvt::Bitcast, Ty::I1, x)
                if self.f.types[x.0] != Ty::I1 =>
            {
                let (value, u) = self.value(x, lane, assume);
                used |= u;
                match value.form.as_constant() {
                    Some(k) => Some(k & 1 != 0),
                    None => match self.facts.op(self.f, x) {
                        Some(Op::Int(IntOp::LShr, w, s)) => {
                            match self.value(s, lane, None).0.form.as_constant() {
                                Some(k) if k < 32 => self.word_bit(w, k as usize, 12),
                                _ => None,
                            }
                        }
                        _ => None,
                    },
                }
            }
            Op::Convert(_, Ty::I1, x) => bit!(self, x),
            Op::Cmp(pred, a, b) if self.f.types[a.0] == Ty::I1 => {
                match (bit!(self, a), bit!(self, b)) {
                    (Some(x), Some(y)) => match pred {
                        IntPred::Eq => Some(x == y),
                        IntPred::Ne => Some(x != y),
                        _ => None,
                    },
                    _ => None,
                }
            }
            Op::Cmp(pred, a, b) => {
                let (x, u) = self.value(a, lane, assume);
                used |= u;
                let (y, u) = self.value(b, lane, assume);
                used |= u;
                let wide = self.f.types[a.0] == Ty::I64;
                let difference = x.form.sub(&y.form);
                match (pred, difference.as_constant()) {
                    (IntPred::Ne, Some(d)) if d != 0 => Some(true),
                    (IntPred::Eq, Some(d)) if d != 0 => Some(false),
                    (IntPred::Eq, Some(0)) => Some(true),
                    (IntPred::Ne, Some(0)) => Some(false),
                    _ if wide => None,
                    (_, Some(d)) => match (self.bounds(&x.form), self.bounds(&y.form)) {
                        (Some(bx), Some(by)) => decide(pred, bx, by).or_else(|| offset(pred, d, by)),
                        (_, Some(by)) => offset(pred, d, by),
                        _ => None,
                    },
                    _ => match (self.bounds(&x.form), self.bounds(&y.form)) {
                        (Some(bx), Some(by)) => decide(pred, bx, by),
                        _ => None,
                    },
                }
            }
            _ => None,
        };
        (result, used)
    }
}

#[derive(Clone, Copy, Debug)]
struct Store {
    index: usize,
    address: ValueId,
    data: Option<ValueId>,
    predicate: ValueId,
    bytes: u32,
}

fn private_stores(f: &Func, facts: &Facts) -> HashMap<BlockId, Vec<Store>> {
    let mut out: HashMap<BlockId, Vec<Store>> = HashMap::default();
    for &b in &facts.order {
        for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
            if let Inst::Effect {
                op:
                    EffectOp::Memory {
                        space: Space::Scratch,
                        op,
                        ..
                    },
                inputs,
                ..
            } = inst
            {
                let (bytes, data) = match *op {
                    MemoryOp::Store(size) => (size.bytes(), Some(inputs[1])),
                    MemoryOp::Load(_) | MemoryOp::Fence => continue,
                    _ => (4, None),
                };
                out.entry(b).or_default().push(Store {
                    index,
                    address: inputs[0],
                    data,
                    predicate: inputs[op.mask_input()],
                    bytes,
                });
            }
        }
    }
    out
}

fn dominators(f: &Func, facts: &Facts) -> Vec<usize> {
    let n = facts.order.len();
    let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
    let mut idom: Vec<Option<usize>> = vec![None; n];
    if n == 0 {
        return Vec::new();
    }
    idom[0] = Some(0);
    let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| {
        while a != b {
            while a > b {
                a = idom[a].unwrap();
            }
            while b > a {
                b = idom[b].unwrap();
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for r in 1..n {
            let mut new: Option<usize> = None;
            for &(pred, _) in &facts.incoming[&facts.order[r]] {
                let p = rank[&pred];
                if idom[p].is_none() {
                    continue;
                }
                new = Some(match new {
                    None => p,
                    Some(q) => intersect(&idom, p, q),
                });
            }
            if new.is_some() && new != idom[r] {
                idom[r] = new;
                changed = true;
            }
        }
    }
    let _ = f;
    idom.into_iter().map(|d| d.unwrap_or(0)).collect()
}

const PRIVATE_MEMORY: ValueId = ValueId(usize::MAX);

fn users(f: &Func, facts: &Facts) -> HashMap<ValueId, Vec<ValueId>> {
    let mut users: HashMap<ValueId, Vec<ValueId>> = HashMap::default();
    for &b in &facts.order {
        let block = &f.blocks[&b];
        for inst in &block.insts {
            let mut outputs: Vec<ValueId> = Vec::new();
            inst.for_each_output(|o| outputs.push(o));
            let private = matches!(
                inst,
                Inst::Effect {
                    op: EffectOp::Memory {
                        space: Space::Scratch,
                        ..
                    },
                    ..
                }
            );
            let writes = private
                && matches!(inst, Inst::Effect { op: EffectOp::Memory { op, .. }, .. } if !matches!(op, MemoryOp::Load(_) | MemoryOp::Fence));
            for o in inst.operands() {
                let entry = users.entry(o).or_default();
                entry.extend(outputs.iter().copied());
                if writes {
                    entry.push(PRIVATE_MEMORY);
                }
            }
            if private {
                users.entry(PRIVATE_MEMORY).or_default().extend(outputs.iter().copied());
            }
        }
        for edge in block.term.edges() {
            let params = &f.blocks[&edge.dst].params;
            for (arg, (param, _)) in edge.args.iter().zip(params) {
                users.entry(*arg).or_default().push(*param);
            }
        }
    }
    users
}

fn loops(facts: &Facts, idom: &[usize], reaches: &[Vec<bool>]) -> HashMap<BlockId, Vec<BlockId>> {
    let n = facts.order.len();
    let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
    let headers: Vec<usize> = (0..n)
        .filter(|&r| facts.incoming[&facts.order[r]].iter().any(|&(p, _)| rank[&p] >= r))
        .collect();
    let dominates = |h: usize, mut b: usize| loop {
        if b == h {
            return true;
        }
        if b == 0 {
            return false;
        }
        b = idom[b];
    };
    (0..n)
        .map(|r| {
            let around = headers
                .iter()
                .filter(|&&h| dominates(h, r) && (h == r || reaches[r][h]))
                .map(|&h| facts.order[h])
                .collect();
            (facts.order[r], around)
        })
        .collect()
}

fn reaches(f: &Func, facts: &Facts) -> Vec<Vec<bool>> {
    let n = facts.order.len();
    let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
    let successors: Vec<Vec<usize>> = facts
        .order
        .iter()
        .map(|b| f.blocks[b].term.edges().map(|e| rank[&e.dst]).collect())
        .collect();
    (0..n)
        .map(|start| {
            let mut seen = vec![false; n];
            let mut stack: Vec<usize> = successors[start].clone();
            while let Some(x) = stack.pop() {
                if seen[x] {
                    continue;
                }
                seen[x] = true;
                stack.extend(successors[x].iter().copied());
            }
            seen
        })
        .collect()
}

fn copies(f: &Func, facts: &Facts) -> HashMap<ValueId, ValueId> {
    let mut params: Vec<ValueId> = Vec::new();
    let mut incoming: HashMap<ValueId, Vec<ValueId>> = HashMap::default();
    for &b in &facts.order {
        if b == f.entry {
            continue;
        }
        for (index, &(p, _)) in f.blocks[&b].params.iter().enumerate() {
            params.push(p);
            incoming.insert(p, facts.arguments(f, b, index).collect());
        }
    }
    struct Walk {
        index: HashMap<ValueId, usize>,
        low: HashMap<ValueId, usize>,
        stack: Vec<ValueId>,
        on: HashSet<ValueId>,
        next: usize,
        components: Vec<Vec<ValueId>>,
    }
    let mut w = Walk {
        index: HashMap::default(),
        low: HashMap::default(),
        stack: Vec::new(),
        on: HashSet::default(),
        next: 0,
        components: Vec::new(),
    };
    for &root in &params {
        if w.index.contains_key(&root) {
            continue;
        }
        let mut frames: Vec<(ValueId, usize)> = vec![(root, 0)];
        w.index.insert(root, w.next);
        w.low.insert(root, w.next);
        w.next += 1;
        w.stack.push(root);
        w.on.insert(root);
        while let Some(&mut (v, ref mut at)) = frames.last_mut() {
            let args = &incoming[&v];
            if *at < args.len() {
                let a = args[*at];
                *at += 1;
                if !incoming.contains_key(&a) {
                    continue;
                }
                if !w.index.contains_key(&a) {
                    w.index.insert(a, w.next);
                    w.low.insert(a, w.next);
                    w.next += 1;
                    w.stack.push(a);
                    w.on.insert(a);
                    frames.push((a, 0));
                } else if w.on.contains(&a) {
                    let low = w.low[&v].min(w.index[&a]);
                    w.low.insert(v, low);
                }
                continue;
            }
            frames.pop();
            if let Some(&(parent, _)) = frames.last() {
                let low = w.low[&parent].min(w.low[&v]);
                w.low.insert(parent, low);
            }
            if w.low[&v] == w.index[&v] {
                let mut component = Vec::new();
                loop {
                    let x = w.stack.pop().unwrap();
                    w.on.remove(&x);
                    component.push(x);
                    if x == v {
                        break;
                    }
                }
                w.components.push(component);
            }
        }
    }
    let mut copies: HashMap<ValueId, ValueId> = HashMap::default();
    for component in w.components {
        let inside: HashSet<ValueId> = component.iter().copied().collect();
        let mut outside: Option<ValueId> = None;
        let mut single = true;
        'scan: for p in &component {
            for &a in &incoming[p] {
                let a = copies.get(&a).copied().unwrap_or(a);
                if inside.contains(&a) {
                    continue;
                }
                match outside {
                    None => outside = Some(a),
                    Some(o) if o == a => {}
                    Some(_) => {
                        single = false;
                        break 'scan;
                    }
                }
            }
        }
        if let (true, Some(value)) = (single, outside) {
            for p in component {
                copies.insert(p, value);
            }
        }
    }
    copies
}

impl Addresses<'_> {
    fn aperture(&mut self, a: ValueId, b: ValueId, lane: usize) -> Option<bool> {
        let (Some(Op::Cmp(IntPred::Uge, base, low)), Some(Op::Cmp(IntPred::Ult, again, _))) =
            (self.facts.op(self.f, a), self.facts.op(self.f, b))
        else {
            return None;
        };
        if base != again || self.facts.op(self.f, low) != Some(Op::Env(Env::ScratchBase)) {
            return None;
        }
        let (pointer, _) = self.value(base, lane, None);
        pointer.region.map(|r| r == Region::Private)
    }
}

impl Addresses<'_> {
    fn stepped(&mut self, v: ValueId, block: BlockId, index: usize, lane: usize) -> Option<Value> {
        let exec_index = self.exec_index?;
        let own = self.rank[&block];
        let resolve = |this: &Self, x: ValueId| this.copies.get(&x).copied().unwrap_or(x);
        let mut step: Option<u32> = None;
        let mut entering: Option<Value> = None;
        for &(pred, slot) in &self.facts.incoming[&block].clone() {
            if !self.can_take(pred, slot, block) {
                continue;
            }
            let edge = self.f.blocks[&pred].term.edges().nth(slot).unwrap();
            let (arg, mask) = (edge.args[index], edge.args[exec_index]);
            if self.rank[&pred] < own {
                let (value, _) = self.operand(arg, block, lane, None);
                match &entering {
                    None => entering = Some(value),
                    Some(old) if *old == value => {}
                    Some(_) => return None,
                }
                continue;
            }
            let arg = resolve(self, arg);
            let edge = self.edge_condition(pred, slot);
            let added = match self.facts.op(self.f, arg) {
                Some(Op::Select(c, x, y)) if self.source(y, lane) == (v, lane) && self.implies(mask, c, edge, 8) => x,
                _ => arg,
            };
            let k = match self.source(added, lane) {
                (x, l) if x == v && l == lane => 0,
                (x, l) if l == lane => {
                    let other = self.increment(x, v, lane)?;
                    self.value(other, lane, None).0.form.as_constant()?
                }
                _ => return None,
            };
            match step {
                None => step = Some(k),
                Some(old) if old == k => {}
                Some(_) => return None,
            }
        }
        let (step, entering) = (step?, entering?);
        if step == 0 {
            return Some(entering);
        }
        let trips = self.trips(block);
        Some(Value {
            form: entering.form.add(&Form::unknown(trips).scale(step)),
            region: entering.region,
        })
    }

    fn source(&self, x: ValueId, lane: usize) -> (ValueId, usize) {
        let (mut x, mut lane) = (x, lane);
        for _ in 0..64 {
            x = self.copies.get(&x).copied().unwrap_or(x);
            match self.facts.inst(self.f, x) {
                Some(Inst::Effect {
                    op: EffectOp::Wave(WaveOp::WriteLane),
                    inputs,
                    ..
                }) => match self.facts.constant(self.f, inputs[1]) {
                    Some(k) if (k & 31) as usize == lane => x = inputs[0],
                    Some(_) => x = inputs[2],
                    None => break,
                },
                Some(Inst::Effect {
                    op: EffectOp::Wave(WaveOp::ReadLane),
                    inputs,
                    ..
                }) => match self.facts.constant(self.f, inputs[1]) {
                    Some(k) => {
                        lane = (k & 31) as usize;
                        x = inputs[0];
                    }
                    None => break,
                },
                _ => break,
            }
        }
        (x, lane)
    }

    fn increment(&self, x: ValueId, v: ValueId, lane: usize) -> Option<ValueId> {
        let resolve = |x: ValueId| self.copies.get(&x).copied().unwrap_or(x);
        let low_word_is_v = |a: ValueId| {
            self.source(a, lane) == (v, lane)
                || matches!(
                    self.facts.op(self.f, resolve(a)),
                    Some(Op::Convert(Cvt::ZExt | Cvt::SExt, Ty::I64, w) | Op::Pack64(w, _))
                        if self.source(w, lane) == (v, lane)
                )
        };
        match self.facts.op(self.f, resolve(x))? {
            Op::Int(IntOp::Add, a, b) if low_word_is_v(a) => Some(b),
            Op::Int(IntOp::Add, a, b) if low_word_is_v(b) => Some(a),
            Op::Convert(Cvt::Trunc, Ty::I32, w) | Op::UnpackLo(w) if self.f.types[w.0] == Ty::I64 => {
                self.increment(w, v, lane)
            }
            _ => None,
        }
    }

    fn edge_arg(&self, (pred, slot): (BlockId, usize), index: usize) -> ValueId {
        self.f.blocks[&pred].term.edges().nth(slot).unwrap().args[index]
    }

    fn edges_into(&self, block: BlockId) -> (Edges, Edges) {
        let own = self.rank[&block];
        self.facts.incoming[&block]
            .iter()
            .copied()
            .filter(|&(pred, slot)| self.can_take(pred, slot, block))
            .partition(|&(pred, _)| self.rank[&pred] < own)
    }

    fn symbol(&mut self, key: Key, header: BlockId) -> Unknown {
        self.intern(
            key,
            UnknownInfo {
                rank: 0,
                shared: false,
                block: header,
                range: None,
                through: vec![header],
            },
        )
    }

    fn advance(&mut self, first: Value, step: u32, header: BlockId) -> Value {
        if step == 0 {
            return first;
        }
        let trips = self.trips(header);
        Value {
            form: first.form.add(&Form::unknown(trips).scale(step)),
            region: first.region,
        }
    }

    fn trips(&mut self, header: BlockId) -> Unknown {
        self.intern(
            Key::Trip(header),
            UnknownInfo {
                rank: 0,
                shared: true,
                block: header,
                range: None,
                through: vec![header],
            },
        )
    }

    fn last_trip(&mut self, header: BlockId, trips: Unknown) -> Option<u32> {
        let own = self.rank[&header];
        let mut condition: Option<(ValueId, bool)> = None;
        for &(pred, slot) in &self.facts.incoming[&header].clone() {
            if self.rank[&pred] < own {
                continue;
            }
            let edge = self.guard(pred, slot)?;
            match condition {
                None => condition = Some(edge),
                Some(old) if old == edge => {}
                Some(_) => return None,
            }
        }
        let (cond, taken) = condition?;
        let cond = self.copies.get(&cond).copied().unwrap_or(cond);
        let Some(Op::Cmp(pred, a, b)) = self.facts.op(self.f, cond) else {
            return None;
        };
        if self.f.types[a.0] != Ty::I32 {
            return None;
        }
        let mut lanes: Vec<usize> = (0..LANES).filter(|&l| self.valid(l)).collect();
        if self.facts.uniform[a.0] && self.facts.uniform[b.0] {
            lanes.truncate(1);
        }
        let mut operands: Option<(Form, Form)> = None;
        for l in lanes {
            let x = self.value(a, l, None).0.form;
            let y = self.value(b, l, None).0.form;
            match &operands {
                None => operands = Some((x, y)),
                Some((ox, oy)) if *ox == x && *oy == y => {}
                Some(_) => return None,
            }
        }
        let (x, y) = operands?;
        let linear = |f: &Form| match f.terms.as_slice() {
            [] => Some((f.constant, 0)),
            [(u, k)] if *u == trips => Some((f.constant, *k)),
            _ => None,
        };
        first_failure(pred, taken, linear(&x)?, linear(&y)?)
    }

    fn induction(&mut self, v: ValueId, block: BlockId, index: usize, lane: usize) -> Option<(u32, u32)> {
        let own = self.rank[&block];
        let mut high = 0u64;
        for &(pred, slot) in &self.facts.incoming[&block].clone() {
            if !self.can_take(pred, slot, block) {
                continue;
            }
            let arg = self.f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
            if self.rank[&pred] >= own {
                let arg = self.copies.get(&arg).copied().unwrap_or(arg);
                let Some(Op::Int(IntOp::LShr, x, k)) = self.facts.op(self.f, arg) else {
                    return None;
                };
                let x = self.copies.get(&x).copied().unwrap_or(x);
                if x != v || !matches!(self.facts.constant(self.f, k), Some(k) if k >= 1) {
                    return None;
                }
            } else {
                let (value, _) = self.value(arg, lane, None);
                high = high.max(self.bounds(&value.form)?.1);
            }
        }
        Some((0, high as u32))
    }

    fn word_bit(&mut self, w: ValueId, bit: usize, depth: usize) -> Option<bool> {
        let w = self.copies.get(&w).copied().unwrap_or(w);
        if let Some(k) = self.value(w, bit, None).0.form.as_constant() {
            return Some(k >> bit & 1 != 0);
        }
        if depth == 0 {
            return None;
        }
        if let Site::Param { block, index } = self.facts.site[w.0] {
            if block == self.f.entry {
                return None;
            }
            let own = self.rank[&block];
            let mut entering: Option<Option<bool>> = None;
            for &(pred, slot) in &self.facts.incoming[&block].clone() {
                if !self.can_take(pred, slot, block) {
                    continue;
                }
                let arg = self.f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
                if self.headers.contains(&block) && self.rank[&pred] >= own {
                    let edge = self.edge_condition(pred, slot);
                    if !self.word_implies(arg, w, edge, 8) {
                        return None;
                    }
                    continue;
                }
                let b = self.word_bit(arg, bit, depth - 1);
                match entering {
                    None => entering = Some(b),
                    Some(old) if old == b => {}
                    Some(_) => return None,
                }
            }
            let entering = entering.flatten();
            return if self.headers.contains(&block) {
                (entering == Some(false)).then_some(false)
            } else {
                entering
            };
        }
        match self.facts.inst(self.f, w) {
            Some(Inst::Core { op, .. }) => match *op {
                Op::Const(_, k) => Some(k >> bit & 1 != 0),
                Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                    let x = self.word_bit(a, bit, depth - 1);
                    match (k, x) {
                        (IntOp::And, Some(false)) => return Some(false),
                        (IntOp::Or, Some(true)) => return Some(true),
                        _ => {}
                    }
                    let y = self.word_bit(b, bit, depth - 1);
                    match (k, x, y) {
                        (IntOp::And, _, Some(false)) => Some(false),
                        (IntOp::Or, _, Some(true)) => Some(true),
                        (_, Some(x), Some(y)) => Some(match k {
                            IntOp::And => x && y,
                            IntOp::Or => x || y,
                            _ => x != y,
                        }),
                        _ => None,
                    }
                }
                Op::Select(c, a, b) => match self.bit(c, bit, None).0 {
                    Some(true) => self.word_bit(a, bit, depth - 1),
                    Some(false) => self.word_bit(b, bit, depth - 1),
                    None => {
                        let (x, y) = (self.word_bit(a, bit, depth - 1), self.word_bit(b, bit, depth - 1));
                        if x == y {
                            x
                        } else {
                            None
                        }
                    }
                },
                Op::Convert(Cvt::Bitcast, _, a) => self.word_bit(a, bit, depth - 1),
                _ => None,
            },
            Some(Inst::Effect {
                op: EffectOp::Wave(WaveOp::Ballot),
                inputs,
                ..
            }) => {
                if !self.valid(bit) {
                    return Some(false);
                }
                self.bit(inputs[0], bit, None).0
            }
            _ => None,
        }
    }

    fn narrowed(&mut self, v: ValueId, block: BlockId, index: usize, lane: usize) -> Option<bool> {
        let own = self.rank[&block];
        let mut entering = Vec::new();
        for &(pred, slot) in &self.facts.incoming[&block].clone() {
            if !self.can_take(pred, slot, block) {
                continue;
            }
            let arg = self.f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
            if self.rank[&pred] >= own {
                let edge = self.edge_condition(pred, slot);
                if !self.implies(arg, v, edge, 8) {
                    return None;
                }
            } else {
                entering.push(arg);
            }
        }
        for a in entering {
            if self.bit(a, lane, None).0 != Some(false) {
                return None;
            }
        }
        Some(false)
    }

    fn decided(&self, c: ValueId, edge: Option<(ValueId, bool)>) -> Option<bool> {
        let (e, taken) = edge?;
        let c = self.copies.get(&c).copied().unwrap_or(c);
        if c == e {
            return Some(taken);
        }
        match self.facts.inst(self.f, c) {
            Some(Inst::Effect {
                op: EffectOp::Wave(WaveOp::Any),
                inputs,
                ..
            }) if inputs[0] == e => {
                if taken {
                    Some(true)
                } else {
                    self.facts.uniform[e.0].then_some(false)
                }
            }
            Some(Inst::Core {
                op: Op::Int(IntOp::Xor, x, one),
                ..
            }) if *x == e && self.facts.constant(self.f, *one) == Some(1) => Some(!taken),
            _ => None,
        }
    }

    fn guard(&self, pred: BlockId, slot: usize) -> Option<(ValueId, bool)> {
        let (mut block, mut slot) = (pred, slot);
        for _ in 0..16 {
            if let Some(condition) = self.edge_condition(block, slot) {
                return Some(condition);
            }
            match self.facts.incoming[&block].as_slice() {
                [(p, s)] => (block, slot) = (*p, *s),
                _ => return None,
            }
        }
        None
    }

    fn edge_condition(&self, pred: BlockId, slot: usize) -> Option<(ValueId, bool)> {
        match self.f.blocks[&pred].term {
            Term::CondBr { cond, .. } => Some((cond, slot == 0)),
            _ => None,
        }
    }

    fn implies(&self, a: ValueId, v: ValueId, edge: Option<(ValueId, bool)>, depth: usize) -> bool {
        let a = self.copies.get(&a).copied().unwrap_or(a);
        let v = self.copies.get(&v).copied().unwrap_or(v);
        if a == v {
            return true;
        }
        if depth == 0 {
            return false;
        }
        match self.facts.op(self.f, a) {
            Some(Op::Int(IntOp::And, x, y)) => {
                self.implies(x, v, edge, depth - 1) || self.implies(y, v, edge, depth - 1)
            }
            Some(Op::Select(c, x, y)) => match self.decided(c, edge) {
                Some(taken) => self.implies(if taken { x } else { y }, v, edge, depth - 1),
                None => self.implies(x, v, edge, depth - 1) && self.implies(y, v, edge, depth - 1),
            },
            Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => match self.facts.op(self.f, shifted) {
                Some(Op::Int(IntOp::LShr, w, s)) if self.facts.op(self.f, s) == Some(Op::Env(Env::LaneId)) => {
                    self.word_holds(w, v, edge, depth - 1)
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn word_holds(&self, w: ValueId, v: ValueId, edge: Option<(ValueId, bool)>, depth: usize) -> bool {
        let w = self.copies.get(&w).copied().unwrap_or(w);
        if depth == 0 {
            return false;
        }
        match self.facts.inst(self.f, w) {
            Some(Inst::Effect {
                op: EffectOp::Wave(WaveOp::Ballot),
                inputs,
                ..
            }) => self.implies(inputs[0], v, edge, depth - 1),
            Some(Inst::Core { op, .. }) => match *op {
                Op::Int(IntOp::And, x, y) => {
                    self.word_holds(x, v, edge, depth - 1) || self.word_holds(y, v, edge, depth - 1)
                }
                Op::Int(IntOp::Or, x, y) => {
                    self.word_holds(x, v, edge, depth - 1) && self.word_holds(y, v, edge, depth - 1)
                }
                Op::Select(c, x, y) => match self.decided(c, edge) {
                    Some(taken) => self.word_holds(if taken { x } else { y }, v, edge, depth - 1),
                    None => self.word_holds(x, v, edge, depth - 1) && self.word_holds(y, v, edge, depth - 1),
                },
                Op::Convert(Cvt::Bitcast, _, x) => self.word_holds(x, v, edge, depth - 1),
                Op::Const(_, 0) => true,
                _ => false,
            },
            _ => false,
        }
    }

    fn word_implies(&self, a: ValueId, w: ValueId, edge: Option<(ValueId, bool)>, depth: usize) -> bool {
        let a = self.copies.get(&a).copied().unwrap_or(a);
        let w = self.copies.get(&w).copied().unwrap_or(w);
        if a == w {
            return true;
        }
        if depth == 0 {
            return false;
        }
        match self.facts.op(self.f, a) {
            Some(Op::Int(IntOp::And, x, y)) => {
                self.word_implies(x, w, edge, depth - 1) || self.word_implies(y, w, edge, depth - 1)
            }
            Some(Op::Select(c, x, y)) => match self.decided(c, edge) {
                Some(taken) => self.word_implies(if taken { x } else { y }, w, edge, depth - 1),
                None => self.word_implies(x, w, edge, depth - 1) && self.word_implies(y, w, edge, depth - 1),
            },
            Some(Op::Convert(Cvt::Bitcast, _, x)) => self.word_implies(x, w, edge, depth - 1),
            _ => false,
        }
    }
}

fn added(value: &Value, symbol: Unknown, region: Option<Region>) -> Option<u32> {
    (value.region == region && value.form.terms == [(symbol, 1)]).then_some(value.form.constant)
}

fn agree(step: &mut Option<u32>, found: Option<u32>) -> bool {
    match (found, *step) {
        (Some(k), None) => {
            *step = Some(k);
            true
        }
        (Some(k), Some(old)) => k == old,
        (None, _) => false,
    }
}

fn low_bits(form: &Form, c: u32, op: IntOp) -> Option<Form> {
    let j = form.alignment();
    if form.terms.is_empty() || j >= 32 || c >= 1 << j {
        return None;
    }
    let low = form.constant & ((1 << j) - 1);
    let result = match op {
        IntOp::And => return Some(Form::constant(low & c)),
        IntOp::Or => low | c,
        IntOp::Xor => low ^ c,
        _ => return None,
    };
    Some(form.sub(&Form::constant(low)).add(&Form::constant(result)))
}

fn first_failure(pred: IntPred, taken: bool, x: (u32, u32), y: (u32, u32)) -> Option<u32> {
    if matches!(pred, IntPred::Eq | IntPred::Ne) {
        let (c, k) = (x.0.wrapping_sub(y.0), x.1.wrapping_sub(y.1));
        if (pred == IntPred::Ne) == taken {
            if k == 0 {
                return (c == 0).then_some(0);
            }
            let shift = k.trailing_zeros();
            if c & ((1u64 << shift) - 1) as u32 != 0 {
                return None;
            }
            let odd = k >> shift;
            let inverse = (0..5).fold(odd, |inv, _| inv.wrapping_mul(2u32.wrapping_sub(odd.wrapping_mul(inv))));
            let low_mask = if shift == 0 { u32::MAX } else { (1u32 << (32 - shift)) - 1 };
            return Some((c.wrapping_neg() >> shift).wrapping_mul(inverse) & low_mask);
        }
        return if c != 0 {
            Some(0)
        } else {
            (k != 0).then_some(1)
        };
    }
    let (moving, fixed, pred) = match (x.1, y.1) {
        (_, 0) => (x, y.0, pred),
        (0, _) => (y, x.0, swapped(pred)),
        _ => return None,
    };
    let signed = matches!(pred, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
    let wide = |v: u32| if signed { v as i32 as i64 } else { v as i64 };
    let (c, k, bound) = (wide(moving.0), moving.1 as i32 as i64, wide(fixed));
    let stays = |t: i64| {
        let m = c + k * t;
        let holds = match pred {
            IntPred::Ult | IntPred::Slt => m < bound,
            IntPred::Ule | IntPred::Sle => m <= bound,
            IntPred::Ugt | IntPred::Sgt => m > bound,
            _ => m >= bound,
        };
        holds == taken
    };
    if !stays(0) {
        return Some(0);
    }
    let first = match k.signum() {
        0 => return None,
        _ => {
            let edge = (bound - c) / k;
            (edge.max(0)..=edge.max(0) + 2).find(|&t| !stays(t))?
        }
    };
    let (low, high) = if signed { (i32::MIN as i64, i32::MAX as i64) } else { (0, u32::MAX as i64) };
    let last = c + k * first;
    (low..=high).contains(&last).then_some(first as u32)
}

fn swapped(pred: IntPred) -> IntPred {
    match pred {
        IntPred::Ult => IntPred::Ugt,
        IntPred::Ule => IntPred::Uge,
        IntPred::Ugt => IntPred::Ult,
        IntPred::Uge => IntPred::Ule,
        IntPred::Slt => IntPred::Sgt,
        IntPred::Sle => IntPred::Sge,
        IntPred::Sgt => IntPred::Slt,
        IntPred::Sge => IntPred::Sle,
        other => other,
    }
}

fn offset(pred: IntPred, d: u32, bounds: (u64, u64)) -> Option<bool> {
    let signed = d as i32 as i64;
    let (low, high) = (bounds.0 as i64, bounds.1 as i64);
    let greater = if signed > 0 && high + signed < 1 << 32 {
        true
    } else if signed < 0 && low + signed >= 0 {
        false
    } else {
        return None;
    };
    let signed_ok = high + signed.max(0) < 1 << 31;
    match pred {
        IntPred::Ugt | IntPred::Uge => Some(greater),
        IntPred::Ult | IntPred::Ule => Some(!greater),
        IntPred::Sgt | IntPred::Sge if signed_ok => Some(greater),
        IntPred::Slt | IntPred::Sle if signed_ok => Some(!greater),
        _ => None,
    }
}

fn extend(word: u32, size: MemSize) -> u32 {
    match size {
        MemSize::U8 => word & 0xff,
        MemSize::U16 => word & 0xffff,
        MemSize::I8 => word as u8 as i8 as i32 as u32,
        MemSize::I16 => word as u16 as i16 as i32 as u32,
        MemSize::B32 | MemSize::B64 => word,
    }
}

fn decide(pred: IntPred, x: (u64, u64), y: (u64, u64)) -> Option<bool> {
    if x.0 == x.1 && y.0 == y.1 {
        return Some(compare(pred, x.0 as u32, y.0 as u32));
    }
    let signed = matches!(pred, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
    if signed && (x.1 >= 1 << 31 || y.1 >= 1 << 31) {
        return None;
    }
    let below = x.1 < y.0;
    let above = x.0 > y.1;
    let (le, ge) = (x.1 <= y.0, x.0 >= y.1);
    match pred {
        IntPred::Eq => (below || above).then_some(false),
        IntPred::Ne => (below || above).then_some(true),
        IntPred::Ult | IntPred::Slt => {
            if below {
                Some(true)
            } else if ge {
                Some(false)
            } else {
                None
            }
        }
        IntPred::Ule | IntPred::Sle => {
            if le {
                Some(true)
            } else if above {
                Some(false)
            } else {
                None
            }
        }
        IntPred::Ugt | IntPred::Sgt => {
            if above {
                Some(true)
            } else if le {
                Some(false)
            } else {
                None
            }
        }
        IntPred::Uge | IntPred::Sge => {
            if ge {
                Some(true)
            } else if below {
                Some(false)
            } else {
                None
            }
        }
    }
}

fn compare(pred: IntPred, x: u32, y: u32) -> bool {
    let (sx, sy) = (x as i32, y as i32);
    match pred {
        IntPred::Eq => x == y,
        IntPred::Ne => x != y,
        IntPred::Ult => x < y,
        IntPred::Ule => x <= y,
        IntPred::Ugt => x > y,
        IntPred::Uge => x >= y,
        IntPred::Slt => sx < sy,
        IntPred::Sle => sx <= sy,
        IntPred::Sgt => sx > sy,
        IntPred::Sge => sx >= sy,
    }
}
