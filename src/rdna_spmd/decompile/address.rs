use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::engine::{EntryLayout, WORKGROUP_ID_X, WORKGROUP_ID_YZ};
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::hash::HashMap;
type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::rdna_spmd::hash::Mix>>;
use super::encoding::{mirrored, outcomes, Encoding};
use super::hazard::Reach;
use super::provenance::{bounds, Exposure, Spill};
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};

pub const LANES: usize = 32;
const SEQUENCE: usize = 1 << 16;

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
    pub values: Option<std::rc::Rc<[u32]>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wide {
    pub constant: i128,
    pub terms: Vec<(Unknown, i128)>,
    pub words: Vec<(Form, i128)>,
}

impl Wide {
    fn plus(&self, other: &Wide, sign: i128) -> Wide {
        let mut terms = self.terms.clone();
        for &(u, c) in &other.terms {
            match terms.iter_mut().find(|(v, _)| *v == u) {
                Some((_, old)) => *old += sign * c,
                None => terms.push((u, sign * c)),
            }
        }
        terms.retain(|&(_, c)| c != 0);
        terms.sort_unstable();
        let mut words = self.words.clone();
        for (f, c) in &other.words {
            match words.iter_mut().find(|(g, _)| g == f) {
                Some((_, old)) => *old += sign * c,
                None => words.push((f.clone(), sign * c)),
            }
        }
        words.retain(|(_, c)| *c != 0);
        Wide {
            constant: self.constant + sign * other.constant,
            terms,
            words,
        }
    }

    fn times(&self, k: i128) -> Wide {
        Wide {
            constant: self.constant * k,
            terms: self.terms.iter().filter(|_| k != 0).map(|&(u, c)| (u, c * k)).collect(),
            words: self.words.iter().filter(|_| k != 0).map(|(f, c)| (f.clone(), c * k)).collect(),
        }
    }
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
    Shifted(Option<BlockId>, Vec<(Unknown, u32)>, u32),
    ShiftCarry(Option<BlockId>, Vec<(Unknown, u32)>, u32, u32),
    ShiftWrap(Option<BlockId>, Vec<(Unknown, u32)>, u32, u32),
    Masked(Option<BlockId>, Vec<(Unknown, u32)>, u32),
    MaskCarry(Option<BlockId>, Vec<(Unknown, u32)>, u32, u32),
    Product(Option<BlockId>, u8, Box<(Form, Form)>),
    Monomial(Option<BlockId>, Vec<Unknown>),
    Both(Option<BlockId>, u8, Box<(Form, Form)>),
    Power(Option<BlockId>, u8, Box<Form>),
    Shift(Option<BlockId>, u8, IntOp, Box<(Form, Form)>),
    Selector(BlockId, u64),
    Pattern(BlockId, u64, Vec<u32>),
    Chosen(ValueId, u8),
    Low(ValueId, Form, u32),
    Spread(ValueId, u8, Vec<Form>),
    HeldSpread(Slot, usize, Vec<Form>),
    Left(Unknown, u8),
    Base(Region),
    Sequence(ValueId, u8),
    High(ValueId, u8),
    Carry(ValueId, u8),
    Cycle(ValueId, u8),
    Guess(ValueId, u8),
    GuessSlot(Slot),
    Held(Slot, usize),
}

#[derive(Clone, PartialEq)]
enum Reached {
    Value(Value),
    Same,
}

enum Entry {
    Value((ValueId, u8, Option<ValueId>)),
    Bit((ValueId, u8, Option<ValueId>)),
    Key(Key),
    Slot(Slot),
    LoopBits(BlockId),
    Step(StepKey),
    Summarized(BlockId),
    Decision(BlockId),
    Limits(BlockId),
    Reach(BlockId),
    Region(ValueKey, bool),
    WordBit((ValueId, u8)),
    High((ValueId, u8)),
}

#[derive(Clone, Debug, Default)]
struct Guesses {
    depth: Depth,
    params: Vec<(ValueId, u8)>,
    slots: Vec<(u32, u32, u8)>,
    summary: bool,
    found: Vec<(Target, u8, Option<Region>)>,
}

#[derive(Clone)]
struct Goal {
    target: Target,
    lane: u8,
    region: Option<Region>,
    assumed: Option<Region>,
}

enum Outcome {
    Bits(Vec<(usize, u8)>, Vec<Goal>),
    Values(Vec<Step>, Vec<usize>, Vec<Goal>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Target {
    Param(usize),
    Slot(u32, u32),
}

type Depth = (usize, usize);
const FREE: Depth = (usize::MAX, 0);

fn level(d: usize) -> Depth {
    (d, d)
}

type Cached<K, V> = HashMap<K, (V, Depth)>;
type ValueKey = (ValueId, u8, Option<ValueId>);
type StepKey = (BlockId, u8, Target, Option<Region>);
type Step = (Option<u32>, bool, Option<(u32, u32)>);
type ImplicationKey = (Implication, ValueId, ValueId, Option<(ValueId, bool)>);
type Bits = std::rc::Rc<Vec<((usize, u8), bool)>>;
type Edges = Vec<(BlockId, usize)>;

type Slot = (BlockId, u32, u32, u8);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Reliance {
    used: bool,
    open: bool,
    lane: bool,
}

impl std::ops::BitOrAssign for Reliance {
    fn bitor_assign(&mut self, other: Self) {
        self.used |= other.used;
        self.open |= other.open;
        self.lane |= other.lane;
    }
}

const ALL: u8 = u8::MAX;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Regions {
    any: bool,
    list: Vec<Option<Region>>,
}

impl Regions {
    #[inline]
    pub fn one(r: Option<Region>) -> Self {
        Self {
            any: false,
            list: vec![r],
        }
    }

    #[inline]
    fn any() -> Self {
        Self {
            any: true,
            list: Vec::new(),
        }
    }

    #[inline]
    pub(super) fn add(&mut self, r: Option<Region>) {
        if let Err(i) = self.list.binary_search(&r) {
            self.list.insert(i, r);
        }
    }

    #[inline]
    pub fn union(&mut self, other: &Regions) {
        self.any |= other.any;
        for &r in &other.list {
            self.add(r);
        }
    }

    #[inline]
    fn joined(mut self, other: &Regions) -> Self {
        self.union(other);
        self
    }

    fn combine(&self, other: &Regions, f: impl Fn(Option<Region>, Option<Region>) -> Option<Region>) -> Self {
        if self.any || other.any {
            return Self::any();
        }
        let mut out = Self::default();
        for &a in &self.list {
            for &b in &other.list {
                out.add(f(a, b));
            }
        }
        out
    }

    fn single(&self) -> Option<Option<Region>> {
        match self.list.as_slice() {
            [r] if !self.any => Some(*r),
            _ => None,
        }
    }

    pub fn single_region(&self) -> bool {
        self.single().is_some()
    }

    #[inline]
    fn within(&self, other: &Regions) -> bool {
        other.any || (!self.any && self.list.iter().all(|r| other.list.contains(r)))
    }

    pub fn overlaps(&self, other: &Regions, f: impl Fn(Option<Region>, Option<Region>) -> bool) -> bool {
        self.any || other.any || self.list.iter().any(|&a| other.list.iter().any(|&b| f(a, b)))
    }

    #[inline]
    pub fn reaches(&self, r: Option<Region>, f: impl Fn(Option<Region>, Option<Region>) -> bool) -> bool {
        self.any || self.list.iter().any(|&a| f(a, r))
    }

    #[inline]
    pub fn lost(&self) -> bool {
        self.any || self.list.contains(&None)
    }

}

fn in_lane<T>(x: T) -> Assumed<T> {
    (
        x,
        Reliance {
            lane: true,
            ..Reliance::default()
        },
    )
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
    copies: Copies,
    narrowing_edges: HashSet<(BlockId, usize)>,
    narrowable: Vec<bool>,
    conditions: std::cell::RefCell<Vec<Option<(std::rc::Rc<Cond>, bool)>>>,
    stores: HashMap<BlockId, Vec<Store>>,
    pub unknowns: Vec<UnknownInfo>,
    keys: Cached<Key, Unknown>,
    wave: usize,
    entered: Option<usize>,
    decisions: Cached<BlockId, Option<bool>>,
    limited: Cached<BlockId, std::rc::Rc<Limits>>,
    limiting: HashMap<BlockId, usize>,
    reach: Cached<BlockId, bool>,
    deciding: HashMap<BlockId, usize>,
    reaching: HashMap<BlockId, usize>,
    values: Cached<ValueKey, Assumed<Value>>,
    bits: Cached<ValueKey, Assumed<Option<bool>>>,
    steps: Cached<StepKey, Step>,
    summarized: Cached<BlockId, ()>,
    summarizing: HashMap<BlockId, usize>,
    loop_bits: Cached<BlockId, Bits>,
    guessing_about: HashMap<BlockId, Guesses>,
    users: HashMap<ValueId, Vec<ValueId>>,
    implied: HashMap<ValueId, Vec<ValueId>>,
    implications: std::cell::RefCell<HashMap<ImplicationKey, bool>>,
    assumable: HashSet<ValueId>,
    equated: HashSet<ValueId>,
    derived: HashMap<Unknown, Vec<ValueId>>,
    monomials: HashMap<Unknown, Vec<Unknown>>,
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
    guessed_slots: Cached<Slot, Option<Value>>,
    active_slots: HashMap<Slot, usize>,
    regions: Cached<ValueKey, Assumed<Regions>>,
    refined: Cached<ValueKey, Assumed<Regions>>,
    active_regions: HashSet<(ValueId, u8, Option<ValueId>, bool)>,
    word_bits: Cached<(ValueId, u8), Option<bool>>,
    active_words: HashMap<(ValueId, u8), usize>,
    known: Vec<Option<Option<Region>>>,
    plain: Vec<bool>,
    bounds: HashMap<ValueId, Vec<Regions>>,
    looped: BTreeMap<(ValueId, u8, bool), usize>,
    checked: BTreeSet<(ValueId, u8, bool)>,
    grown: HashMap<(ValueId, u8, bool), Regions>,
    split: HashSet<(ValueId, bool)>,
    loaded: HashMap<ValueId, Regions>,
    spills: Vec<Spill>,
    sources: HashMap<ValueId, Vec<usize>>,
    exposing: Vec<Exposure>,
    highs: Cached<(ValueId, u8), Form>,
    active_highs: HashSet<(ValueId, u8)>,
    opaque_highs: HashSet<Unknown>,
    readbacks: HashMap<(BlockId, usize), Option<(ValueId, Option<ValueId>, Option<u64>)>>,
    idom: Vec<usize>,
    written: HashMap<(BlockId, usize), Regions>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Implication {
    Bit,
    Holds,
    Word,
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
            narrowing_edges: HashSet::default(),
            narrowable: Vec::new(),
            conditions: std::cell::RefCell::new(vec![None; 2 * f.types.len()]),
            stores: private_stores(f, facts),
            unknowns: Vec::new(),
            keys: HashMap::default(),
            wave: 0,
            entered: None,
            decisions: HashMap::default(),
            limited: HashMap::default(),
            limiting: HashMap::default(),
            reach: HashMap::default(),
            deciding: HashMap::default(),
            reaching: HashMap::default(),
            values: HashMap::default(),
            bits: HashMap::default(),
            steps: HashMap::default(),
            summarized: HashMap::default(),
            summarizing: HashMap::default(),
            loop_bits: HashMap::default(),
            guessing_about: HashMap::default(),
            users: users(f, facts),
            implied: HashMap::default(),
            implications: std::cell::RefCell::new(HashMap::default()),
            assumable: HashSet::default(),
            equated: HashSet::default(),
            derived: HashMap::default(),
            monomials: HashMap::default(),
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
            guessed_slots: HashMap::default(),
            active_slots: HashMap::default(),
            regions: HashMap::default(),
            refined: HashMap::default(),
            active_regions: HashSet::default(),
            word_bits: HashMap::default(),
            active_words: HashMap::default(),
            known: Vec::new(),
            plain: Vec::new(),
            bounds: HashMap::default(),
            looped: BTreeMap::new(),
            checked: BTreeSet::new(),
            grown: HashMap::default(),
            split: HashSet::default(),
            loaded: HashMap::default(),
            spills: Vec::new(),
            sources: HashMap::default(),
            exposing: Vec::new(),
            highs: HashMap::default(),
            active_highs: HashSet::default(),
            opaque_highs: HashSet::default(),
            readbacks: HashMap::default(),
            idom: idom.clone(),
            written: HashMap::default(),
        };
        let found = bounds(f, facts, &this.copies, &this.users, inputs, &this.entry, env, &this.headers, registry);
        let mut narrowing = HashSet::default();
        for &b in &facts.order {
            if let Term::CondBr { cond, .. } = f.blocks[&b].term {
                for slot in 0..2 {
                    if !this.fixed_words(cond, slot == 0).is_empty() {
                        narrowing.insert((b, slot));
                    }
                }
            }
        }
        let mut narrowable = vec![false; f.types.len()];
        for &b in &facts.order {
            if b != f.entry && facts.incoming[&b].iter().any(|e| narrowing.contains(e)) {
                for &(p, ty) in &f.blocks[&b].params {
                    narrowable[p.0] = matches!(ty, Ty::I32 | Ty::I64);
                }
            }
        }
        this.narrowing_edges = narrowing;
        this.narrowable = narrowable;
        (this.known, this.plain, this.bounds, this.loaded) = (found.known, found.plain, found.carried, found.loaded);
        (this.spills, this.exposing, this.written) = (found.spills, found.exposing, found.written);
        for &v in this.bounds.keys() {
            let Site::Inst { block, index } = facts.site[v.0] else {
                continue;
            };
            let at = this.rank[&block];
            let from: Vec<usize> = (0..this.spills.len())
                .filter(|&i| {
                    let (b, k) = this.spills[i].at;
                    (b == block && k < index) || reaches[this.rank[&b]][at]
                })
                .collect();
            this.sources.insert(v, from);
        }
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
                        for (x, _) in this.fixed_words(p, true) {
                            if f.types[x.0] == Ty::I32 {
                                this.equated.insert(x);
                            }
                        }
                        this.assumable.extend(implied);
                    }
                }
            }
        }
        this
    }

    pub fn trip(&self, header: BlockId) -> Option<Unknown> {
        self.keys.get(&Key::Trip(header)).map(|&(u, _)| u)
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
        self.derived.clear();
        self.monomials.clear();
        self.values.clear();
        self.bits.clear();
        self.loop_bits.clear();
        self.steps.clear();
        self.summarized.clear();
        self.slots.clear();
        self.regions.clear();
        self.refined.clear();
        self.word_bits.clear();
        self.highs.clear();
        self.opaque_highs.clear();
        self.decisions.clear();
        self.limited.clear();
        self.reach.clear();
        self.looped.clear();
        self.checked.clear();
        let mut headers: Vec<BlockId> = self.headers.iter().copied().collect();
        headers.sort_by_key(|h| self.rank[h]);
        let trips: Vec<Unknown> = headers.iter().map(|&h| self.trips(h)).collect();
        self.pending = trips.iter().map(|&u| (u, level(1))).collect();
        for (&header, &u) in headers.iter().zip(&trips) {
            let (last, _) = self.sandbox(|this| this.last_trip(header, u));
            if let Some(last) = last {
                self.unknowns[u as usize].range = Some((0, last));
            }
            self.pending.remove(&u);
        }
    }

    fn decision(&mut self, b: BlockId) -> Option<bool> {
        if let Some(&(decided, depth)) = self.decisions.get(&b) {
            self.depend(depth);
            return decided;
        }
        let Term::CondBr { cond, .. } = self.f.blocks[&b].term else {
            return None;
        };
        let outside = match self.deciding.insert(b, self.guessing) {
            Some(level) if level == self.guessing => return None,
            outside => outside,
        };
        let (decided, depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(level(this.checking));
            }
            this.decide(cond)
        });
        match outside {
            Some(level) => self.deciding.insert(b, level),
            None => self.deciding.remove(&b),
        };
        if depth != FREE {
            self.journal.push((Entry::Decision(b), depth));
        }
        self.decisions.insert(b, (decided, depth));
        decided
    }

    fn decide(&mut self, cond: ValueId) -> Option<bool> {
        let lanes: Vec<usize> = (0..LANES).filter(|&l| self.valid(l)).collect();
        if self.facts.uniform[cond.0] {
            return lanes.first().and_then(|&l| self.bit(cond, l, None).0);
        }
        let mut known: Option<bool> = None;
        for l in lanes {
            match (self.bit(cond, l, None).0, known) {
                (Some(bit), None) => known = Some(bit),
                (Some(bit), Some(old)) if bit != old => return None,
                _ => {}
            }
        }
        known
    }

    fn reached(&mut self, b: BlockId) -> bool {
        if b == self.f.entry {
            return true;
        }
        if let Some(&(reached, depth)) = self.reach.get(&b) {
            self.depend(depth);
            return reached;
        }
        let outside = match self.reaching.insert(b, self.guessing) {
            Some(level) if level == self.guessing => return true,
            outside => outside,
        };
        let (reached, depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(level(this.checking));
            }
            let facts = this.facts;
            let own = this.rank[&b];
            facts.incoming[&b]
                .iter()
                .any(|&(pred, slot)| this.rank[&pred] < own && this.reached(pred) && this.takes(pred, slot))
        });
        match outside {
            Some(level) => self.reaching.insert(b, level),
            None => self.reaching.remove(&b),
        };
        if depth != FREE {
            self.journal.push((Entry::Reach(b), depth));
        }
        self.reach.insert(b, (reached, depth));
        reached
    }

    fn takes(&mut self, pred: BlockId, slot: usize) -> bool {
        self.decision(pred).is_none_or(|yes| (slot == 0) == yes)
    }

    pub fn reaches_block(&mut self, b: BlockId) -> bool {
        self.reached(b)
    }

    fn can_take(&mut self, pred: BlockId, slot: usize, block: BlockId) -> bool {
        if self.rank[&pred] >= self.rank[&block] {
            return true;
        }
        self.reached(pred) && self.takes(pred, slot)
    }

    pub fn valid(&self, lane: usize) -> bool {
        ((self.wave * LANES + lane) as u32) < self.env.workgroup_size()
    }

    fn ids(&self, lane: usize) -> (u32, u32, u32) {
        let flat = (self.wave * LANES + lane) as u32;
        let [bx, by, _] = self.env.block;
        (flat % bx, (flat / bx) % by, flat / (bx * by))
    }

    fn known_key(&self, key: &Key) -> Option<Unknown> {
        let found = self.keys.get(key);
        found.map(|&(u, depth)| {
            self.depend(depth);
            u
        })
    }

    fn intern(&mut self, key: Key, mut info: UnknownInfo) -> Unknown {
        let found = self.keys.get(&key);
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
        self.keys.insert(key, (u, depth));
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
            *top = (top.0.min(depth.0), top.1.max(depth.1));
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
            if at.1 < depth {
                self.journal.push((entry, at));
                continue;
            }
            match entry {
                Entry::Value(key) => {
                    if self.values.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.values.remove(&key);
                    }
                }
                Entry::Bit(key) => {
                    if self.bits.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.bits.remove(&key);
                    }
                }
                Entry::Key(key) => {
                    if self.keys.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.keys.remove(&key);
                    }
                }
                Entry::Slot(key) => {
                    if self.slots.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.slots.remove(&key);
                    }
                }
                Entry::LoopBits(key) => {
                    if self.loop_bits.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.loop_bits.remove(&key);
                    }
                }
                Entry::Step(key) => {
                    if self.steps.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.steps.remove(&key);
                    }
                }
                Entry::Region(key, refine) => {
                    let cache = if refine { &mut self.refined } else { &mut self.regions };
                    if cache.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        cache.remove(&key);
                    }
                }
                Entry::WordBit(key) => {
                    if self.word_bits.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.word_bits.remove(&key);
                    }
                }
                Entry::High(key) => {
                    if self.highs.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.highs.remove(&key);
                    }
                }
                Entry::Decision(key) => {
                    if self.decisions.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.decisions.remove(&key);
                    }
                }
                Entry::Limits(key) => {
                    if self.limited.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.limited.remove(&key);
                    }
                }
                Entry::Reach(key) => {
                    if self.reach.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.reach.remove(&key);
                    }
                }
                Entry::Summarized(key) => {
                    if self.summarized.get(&key).is_some_and(|e| e.1 .1 >= depth) {
                        self.summarized.remove(&key);
                    }
                }
            }
        }
        let outer = if rests.1 < depth {
            rests
        } else if rests.0 >= depth {
            FREE
        } else {
            (rests.0, depth - 1)
        };
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
        self.values.insert(key, (r, depth));
    }

    fn cache_bit(&mut self, key: (ValueId, u8, Option<ValueId>), r: Assumed<Option<bool>>, depth: Depth) {
        if depth != FREE {
            self.journal.push((Entry::Bit(key), depth));
        }
        self.bits.insert(key, (r, depth));
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
                values: None,
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
                    Key::Left(u, if info.shared { ALL } else { lane as u8 }),
                    UnknownInfo {
                        rank: 0,
                        shared: info.shared,
                        block: info.block,
                        range: info.range,
                        through: Vec::new(),
                        values: info.values.clone(),
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

    fn slot_in_block(&mut self, at: (BlockId, usize), address: &Form, bytes: u32, lane: usize) -> Option<Value> {
        let mut memo = HashMap::default();
        match self.symbolic_slot(at, address, bytes, lane, None, &mut memo)? {
            Reached::Value(value) => Some(value),
            Reached::Same => None,
        }
    }

    fn symbolic_slot(
        &mut self,
        at: (BlockId, usize),
        address: &Form,
        bytes: u32,
        lane: usize,
        boundary: Option<BlockId>,
        memo: &mut HashMap<(BlockId, Option<BlockId>), Option<Reached>>,
    ) -> Option<Reached> {
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
            let target = self.value(w.address, lane, Some(w.predicate)).0.form;
            if target.terms != address.terms {
                return None;
            }
            let d = target.constant.wrapping_sub(address.constant);
            if d != 0 {
                if d >= bytes && d.wrapping_neg() >= w.bytes {
                    continue;
                }
                return None;
            }
            if w.bytes != bytes || ran != Some(true) {
                return None;
            }
            return Some(Reached::Value(self.value(w.data?, lane, Some(w.predicate)).0));
        }
        if Some(block) == boundary {
            return Some(Reached::Same);
        }
        if block == self.f.entry {
            return None;
        }
        if let Some(known) = memo.get(&(block, boundary)) {
            return known.clone();
        }
        memo.insert((block, boundary), None);
        let found = self.reaching_block_start(block, address, bytes, lane, boundary, memo);
        memo.insert((block, boundary), found.clone());
        found
    }

    fn reaching_block_start(
        &mut self,
        block: BlockId,
        address: &Form,
        bytes: u32,
        lane: usize,
        boundary: Option<BlockId>,
        memo: &mut HashMap<(BlockId, Option<BlockId>), Option<Reached>>,
    ) -> Option<Reached> {
        let (entering, back) = self.edges_into(block);
        if entering.is_empty() {
            return None;
        }
        let mut found: Option<Reached> = None;
        for (pred, _) in entering {
            let end = self.f.blocks[&pred].insts.len();
            let reached = self.symbolic_slot((pred, end), address, bytes, lane, boundary, memo)?;
            match &found {
                Some(old) if *old != reached => return None,
                _ => found = Some(reached),
            }
        }
        for (pred, _) in back {
            let end = self.f.blocks[&pred].insts.len();
            match self.symbolic_slot((pred, end), address, bytes, lane, Some(block), memo)? {
                Reached::Same => {}
                Reached::Value(v) if found == Some(Reached::Value(v.clone())) => {}
                _ => return None,
            }
        }
        found
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
        if let Some(depth) = self.guessing_about.get(&block).map(|g| g.depth) {
            self.depend(depth);
            let region = self.assumed_entry(block, Target::Slot(address, bytes), lane);
            let symbol = self.symbol(Key::GuessSlot(key), block);
            let guess = Value {
                form: Form::unknown(symbol),
                region,
            };
            self.guessed_slots.insert(key, (Some(guess.clone()), depth));
            if let Some(guesses) = self.guessing_about.get_mut(&block) {
                guesses.slots.push((address, bytes, lane as u8));
            }
            return Some(guess);
        }
        let cached = self.slots.get(&key).cloned();
        if let Some((r, depth)) = cached {
            self.depend(depth);
            return r;
        }
        let outside = self.begin_slot(key)?;
        let (result, depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(level(this.checking));
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
        self.slots.insert(key, (result.clone(), depth));
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

    fn spread(&mut self, key: Key, shared: bool, block: BlockId, forms: &[Form]) -> Option<Form> {
        let first = forms.first()?;
        if forms.iter().any(|f| f.terms != first.terms) {
            return None;
        }
        let low = forms.iter().map(|f| f.constant).min()?;
        let high = forms.iter().map(|f| f.constant).max()?;
        let step = forms.iter().fold(0u32, |g, f| {
            let (mut a, mut b) = (g, f.constant - low);
            while b != 0 {
                (a, b) = (b, a % b);
            }
            a
        });
        if step == 0 {
            return Some(first.clone());
        }
        let u = self.intern(
            key,
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range: Some((0, (high - low) / step)),
                through: Vec::new(),
                values: None,
            },
        );
        let base = Form {
            constant: low,
            terms: first.terms.clone(),
        };
        Some(base.add(&Form::unknown(u).scale(step)))
    }

    fn merge_held(&mut self, values: Vec<Value>, slot: Slot, index: usize) -> Option<Value> {
        let first = values.first()?.clone();
        if values.iter().all(|v| *v == first) {
            return Some(first);
        }
        if values.iter().all(|v| v.region == first.region) {
            let forms: Vec<Form> = values.iter().map(|v| v.form.clone()).collect();
            if let Some(form) = self.spread(Key::HeldSpread(slot, index, forms.clone()), false, slot.0, &forms) {
                return Some(Value {
                    form,
                    region: first.region,
                });
            }
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
                values: None,
            },
        );
        Some(Value {
            form: Form::unknown(u),
            region: Some(region),
        })
    }

    fn may_guess(&mut self, header: BlockId) -> bool {
        if self.guessing_about.contains_key(&header) {
            self.depend(level(self.checking));
            return false;
        }
        true
    }

    fn guess_about<T>(
        &mut self,
        header: BlockId,
        bits: &[((usize, u8), bool)],
        installs: &[(Target, u8, Option<Region>)],
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        self.sandbox(|this| {
            let depth = level(this.checking);
            this.guessing += 1;
            let params: Vec<ValueId> = this.f.blocks[&header].params.iter().map(|p| p.0).collect();
            for &((index, l), b) in bits {
                this.guessed_bits.insert((params[index], l), (b, depth));
            }
            this.guessing_about.insert(
                header,
                Guesses {
                    depth,
                    ..Guesses::default()
                },
            );
            for &(target, l, region) in installs {
                let key = match target {
                    Target::Param(index) => Key::Guess(params[index], l),
                    Target::Slot(address, bytes) => Key::GuessSlot((header, address, bytes, l)),
                };
                let guess = Value {
                    form: Form::unknown(this.symbol(key, header)),
                    region,
                };
                let guesses = this.guessing_about.get_mut(&header).unwrap();
                match target {
                    Target::Param(index) => {
                        guesses.params.push((params[index], l));
                        this.guessed.insert((params[index], l), (guess, depth));
                    }
                    Target::Slot(address, bytes) => {
                        guesses.slots.push((address, bytes, l));
                        this.guessed_slots.insert((header, address, bytes, l), (Some(guess), depth));
                    }
                }
            }
            let result = f(this);
            for &((index, l), _) in bits {
                this.guessed_bits.remove(&(params[index], l));
            }
            if let Some(guesses) = this.guessing_about.remove(&header) {
                for key in guesses.params {
                    this.guessed.remove(&key);
                }
                for (address, bytes, l) in guesses.slots {
                    this.guessed_slots.remove(&(header, address, bytes, l));
                }
            }
            this.guessing -= 1;
            result
        })
        .0
    }

    fn loop_bits(&mut self, header: BlockId) -> Bits {
        let cached = self.loop_bits.get(&header).cloned();
        if let Some((bits, depth)) = cached {
            self.depend(depth);
            return bits;
        }
        if !self.may_guess(header) {
            return Bits::default();
        }
        self.summarize(header, None);
        let cached = self.loop_bits.get(&header).cloned();
        match cached {
            Some((bits, depth)) => {
                self.depend(depth);
                bits
            }
            None => Bits::default(),
        }
    }

    fn loop_bit(&mut self, header: BlockId, index: usize, lane: usize) -> Option<bool> {
        let bits = self.loop_bits(header);
        let key = (index, lane as u8);
        bits.binary_search_by(|&(k, _)| k.cmp(&key)).ok().map(|i| bits[i].1)
    }

    fn assumed_entry(&mut self, header: BlockId, target: Target, lane: usize) -> Option<Region> {
        if !self.guessing_about.get(&header).is_some_and(|g| g.summary) {
            return None;
        }
        let first = match target {
            Target::Param(index) => {
                let (entering, _) = self.edges_into(header);
                self.entering_value(&entering, header, index, lane)
            }
            Target::Slot(address, bytes) => self.entering_slot(header, address, bytes, lane),
        }?;
        if let Some(guesses) = self.guessing_about.get_mut(&header) {
            guesses.found.push((target, lane as u8, first.region));
        }
        first.region
    }

    fn summarize(&mut self, header: BlockId, seed: Option<Target>) {
        let done = self.summarized.get(&header).copied();
        if seed.is_none() {
            if let Some((_, depth)) = done {
                self.depend(depth);
                return;
            }
        }
        if !self.may_guess(header) {
            return;
        }
        let outside = match self.summarizing.insert(header, self.guessing) {
            Some(level) if level == self.guessing => return,
            outside => outside,
        };
        let ((bits, steps), depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(level(this.checking));
            }
            this.summary(header, seed)
        });
        match outside {
            Some(level) => self.summarizing.insert(header, level),
            None => self.summarizing.remove(&header),
        };
        if depth != FREE {
            self.journal.push((Entry::LoopBits(header), depth));
            self.journal.push((Entry::Summarized(header), depth));
        }
        self.loop_bits.insert(header, (bits, depth));
        self.summarized.insert(header, ((), depth));
        for (key, step) in steps {
            if depth != FREE {
                self.journal.push((Entry::Step(key), depth));
            }
            self.steps.insert(key, (step, depth));
        }
    }

    fn summary(&mut self, header: BlockId, seed: Option<Target>) -> (Bits, Vec<(StepKey, Step)>) {
        let params: Vec<(ValueId, Ty)> = self.f.blocks[&header].params.clone();
        let (entering, back) = self.edges_into(header);
        let lanes: Vec<usize> = (0..LANES).filter(|&l| self.valid(l)).collect();
        let known = self.loop_bits.get(&header).cloned();
        let mut bits: Vec<((usize, u8), bool)> = match known {
            Some((bits, depth)) => {
                self.depend(depth);
                (*bits).clone()
            }
            None => {
                let mut bits = Vec::new();
                for (index, &(_, ty)) in params.iter().enumerate() {
                    if ty != Ty::I1 {
                        continue;
                    }
                    for &lane in &lanes {
                        let mut first: Option<Option<bool>> = None;
                        for &e in &entering {
                            let bit = self.bit(self.edge_arg(e, index), lane, None).0;
                            match first {
                                None => first = Some(bit),
                                Some(old) if old == bit => {}
                                Some(_) => first = Some(None),
                            }
                        }
                        if let Some(Some(b)) = first {
                            bits.push(((index, lane as u8), b));
                        }
                    }
                }
                bits
            }
        };
        let mut goals: Vec<Goal> = Vec::new();
        if let Some(target) = seed {
            for &lane in &lanes {
                let first = match target {
                    Target::Param(index) => {
                        if self.canonical(params[index].0, lane) != lane {
                            continue;
                        }
                        self.entering_value(&entering, header, index, lane)
                    }
                    Target::Slot(address, bytes) => self.entering_slot(header, address, bytes, lane),
                };
                if let Some(first) = first {
                    goals.push(Goal {
                        target,
                        lane: lane as u8,
                        region: first.region,
                        assumed: first.region,
                    });
                }
            }
        }
        bits.sort_unstable();
        loop {
            let guessed = bits.clone();
            let installs: Vec<(Target, u8, Option<Region>)> =
                goals.iter().map(|g| (g.target, g.lane, g.assumed)).collect();
            let outcome = self.guess_about(header, &guessed, &installs, |this| {
                if let Some(g) = this.guessing_about.get_mut(&header) {
                    g.summary = true;
                }
                let failed: Vec<(usize, u8)> = guessed
                    .iter()
                    .filter(|&&((index, l), b)| {
                        back.iter().any(|&e| {
                            let a = this.edge_arg(e, index);
                            this.bit(a, l as usize, None).0 != Some(b)
                        })
                    })
                    .map(|&(key, _)| key)
                    .collect();
                let mut all: Vec<Goal> = goals.clone();
                let take = |this: &mut Self, all: &mut Vec<Goal>| {
                    let found = std::mem::take(&mut this.guessing_about.get_mut(&header).unwrap().found);
                    for (target, lane, region) in found {
                        if !all.iter().any(|g| g.target == target && g.lane == lane) {
                            all.push(Goal {
                                target,
                                lane,
                                region,
                                assumed: region,
                            });
                        }
                    }
                };
                take(this, &mut all);
                if !failed.is_empty() {
                    return Outcome::Bits(failed, all);
                }
                let mut results = Vec::with_capacity(all.len());
                let mut downgraded = Vec::new();
                let mut i = 0;
                while i < all.len() {
                    let g = all[i].clone();
                    let lane = g.lane as usize;
                    let symbol = match g.target {
                        Target::Param(index) => this.symbol(Key::Guess(params[index].0, g.lane), header),
                        Target::Slot(address, bytes) => {
                            this.symbol(Key::GuessSlot((header, address, bytes, g.lane)), header)
                        }
                    };
                    let mut step = None;
                    let mut stepped = true;
                    let mut keeps = true;
                    let mut held = true;
                    let mut affine = Some(None);
                    for &e in &back {
                        let value = match g.target {
                            Target::Param(index) => {
                                let a = this.edge_arg(e, index);
                                Some(this.operand(a, header, lane, None).0)
                            }
                            Target::Slot(address, bytes) => {
                                let end = this.f.blocks[&e.0].insts.len();
                                this.slot_before((e.0, end), address, bytes, lane)
                            }
                        };
                        keeps &= value.as_ref().is_some_and(|v| v.region == g.region);
                        held &= value.as_ref().is_some_and(|v| v.region == g.assumed);
                        stepped = stepped && agree(&mut step, value.as_ref().and_then(|v| added(v, symbol, g.region)));
                        affine = recurrence(affine, value.as_ref(), symbol);
                    }
                    if g.assumed.is_some() && !held {
                        downgraded.push(i);
                    }
                    results.push((if stepped { step } else { None }, keeps, affine.flatten()));
                    take(this, &mut all);
                    i += 1;
                }
                Outcome::Values(results, downgraded, all)
            });
            match outcome {
                Outcome::Bits(failed, all) => {
                    bits.retain(|(key, _)| !failed.contains(key));
                    goals = all;
                }
                Outcome::Values(results, downgraded, all) => {
                    goals = all;
                    if downgraded.is_empty() {
                        let steps = goals
                            .iter()
                            .zip(results)
                            .map(|(g, r)| ((header, g.lane, g.target, g.region), r))
                            .collect();
                        return (std::rc::Rc::new(bits), steps);
                    }
                    for i in downgraded {
                        goals[i].assumed = None;
                    }
                }
            }
        }
    }

    fn entering_value(&mut self, entering: &[(BlockId, usize)], header: BlockId, index: usize, lane: usize) -> Option<Value> {
        let mut first: Option<Value> = None;
        for &e in entering {
            let value = self.operand(self.edge_arg(e, index), header, lane, None).0;
            match &first {
                None => first = Some(value),
                Some(old) if *old == value => {}
                Some(_) => return None,
            }
        }
        first
    }

    fn recur(&mut self, header: BlockId, lane: usize, target: Target) -> Option<Value> {
        if !self.may_guess(header) {
            return None;
        }
        let (entering, back) = self.edges_into(header);
        let first = match target {
            Target::Param(index) => self.entering_value(&entering, header, index, lane)?,
            Target::Slot(address, bytes) => self.entering_slot(header, address, bytes, lane)?,
        };
        let l = lane as u8;
        let region = first.region;
        let step_key = (header, l, target, region);
        let lookup = |this: &mut Self| {
            let cached = this.steps.get(&step_key).cloned();
            cached.map(|(step, depth)| {
                this.depend(depth);
                step
            })
        };
        let found = match lookup(self) {
            Some(step) => Some(step),
            None => {
                self.summarize(header, Some(target));
                lookup(self)
            }
        };
        let (step, keeps, _) = match found {
            Some(step) => step,
            None => {
                let bits = self.loop_bits(header);
                let (step, depth) = self.frame(|this| this.find_step(header, lane, target, region, &bits, &back));
                if depth != FREE {
                    self.journal.push((Entry::Step(step_key), depth));
                }
                self.steps.insert(step_key, (step, depth));
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

    fn sequence(&mut self, v: ValueId, header: BlockId, index: usize, lane: usize) -> Option<Value> {
        let trips = self.trips(header);
        let last = self.unknowns[trips as usize].range?.1;
        if last as usize >= SEQUENCE {
            return None;
        }
        let l = lane as u8;
        let key = (header, l, Target::Param(index), None);
        let cached = self.steps.get(&key).cloned();
        let ((_, _, affine), depth) = cached?;
        self.depend(depth);
        let (entering, back) = self.edges_into(header);
        let first = self.entering_value(&entering, header, index, lane)?;
        if first.region.is_some() {
            return None;
        }
        let starts = self.starts(&first.form, SEQUENCE / (last as usize + 1))?;
        let mut values = Vec::with_capacity(starts.len() * (last as usize + 1));
        match affine {
            Some((scale, step)) => {
                for &start in &starts {
                    let mut x = start;
                    for _ in 0..=last {
                        values.push(x);
                        x = x.wrapping_mul(scale).wrapping_add(step);
                    }
                }
            }
            None => {
                let mut args: Vec<ValueId> = back.iter().map(|&edge| self.edge_arg(edge, index)).collect();
                args.sort_unstable();
                args.dedup();
                let mut seen: BTreeSet<u32> = starts.iter().copied().collect();
                let mut frontier = starts.clone();
                for _ in 0..last {
                    let mut next = Vec::new();
                    for &x in &frontier {
                        for &arg in &args {
                            let y = self.concrete(arg, v, x, header, lane, 0)? as u32;
                            if seen.insert(y) {
                                next.push(y);
                            }
                        }
                    }
                    if seen.len() > SEQUENCE {
                        return None;
                    }
                    frontier = next;
                }
                values.extend(seen);
            }
        }
        values.sort_unstable();
        values.dedup();
        let shared = self.facts.uniform[v.0];
        let u = self.intern(
            Key::Sequence(v, if shared { ALL } else { l }),
            UnknownInfo {
                rank: 0,
                shared,
                block: header,
                range: Some((values[0], values[values.len() - 1])),
                through: vec![header],
                values: Some(values.into()),
            },
        );
        Some(Value::of(Form::unknown(u)))
    }

    fn starts(&mut self, form: &Form, most: usize) -> Option<Vec<u32>> {
        if let Some(k) = form.as_constant() {
            return Some(vec![k]);
        }
        let &[(u, c)] = form.terms.as_slice() else {
            return None;
        };
        let range = self.range(u);
        let choices: Vec<u32> = match (&self.unknowns[u as usize].values, range) {
            (Some(set), _) => set.to_vec(),
            (None, Some((low, high))) if ((high - low) as usize) < most => (low..=high).collect(),
            _ => return None,
        };
        Some(choices.into_iter().map(|x| form.constant.wrapping_add(c.wrapping_mul(x))).collect())
    }

    fn concrete(&mut self, x: ValueId, param: ValueId, value: u32, header: BlockId, lane: usize, depth: usize) -> Option<u64> {
        if depth > 64 {
            return None;
        }
        let x = self.copies.get(&x).copied().unwrap_or(x);
        if x == param {
            return Some(value as u64);
        }
        let ty = self.f.types[x.0];
        let bits = ty.bits() as u64;
        let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
        let inside = self.loops.get(&self.block_of(x)).is_some_and(|l| l.contains(&header));
        if !inside {
            return match ty {
                Ty::I1 => self.bit(x, lane, None).0.map(u64::from),
                Ty::I32 => self.value(x, lane, None).0.form.as_constant().map(u64::from),
                _ => self.facts.constant(self.f, x),
            };
        }
        let signed = |a: u64, bits: u64| ((a << (64 - bits)) as i64) >> (64 - bits);
        let get = |this: &mut Self, a: ValueId| this.concrete(a, param, value, header, lane, depth + 1);
        let result = match self.facts.op(self.f, x)? {
            Op::Const(_, k) => k,
            Op::Env(Env::LaneId) => lane as u64,
            Op::Int(k, a, b) => {
                let (a, b) = (get(self, a)?, get(self, b)?);
                let amount = b & (bits - 1);
                match k {
                    IntOp::Add => a.wrapping_add(b),
                    IntOp::Sub => a.wrapping_sub(b),
                    IntOp::Mul => a.wrapping_mul(b),
                    IntOp::And => a & b,
                    IntOp::Or => a | b,
                    IntOp::Xor => a ^ b,
                    IntOp::Shl => a << amount,
                    IntOp::LShr => a >> amount,
                    IntOp::AShr => (signed(a, bits) >> amount) as u64,
                }
            }
            Op::Cmp(p, a, b) => {
                let width = self.f.types[a.0].bits() as u64;
                let (a, b) = (get(self, a)?, get(self, b)?);
                let (sa, sb) = (signed(a, width), signed(b, width));
                (match p {
                    IntPred::Eq => a == b,
                    IntPred::Ne => a != b,
                    IntPred::Ult => a < b,
                    IntPred::Ugt => a > b,
                    IntPred::Ule => a <= b,
                    IntPred::Uge => a >= b,
                    IntPred::Slt => sa < sb,
                    IntPred::Sgt => sa > sb,
                    IntPred::Sle => sa <= sb,
                    IntPred::Sge => sa >= sb,
                }) as u64
            }
            Op::Select(c, a, b) => {
                if get(self, c)? != 0 {
                    get(self, a)?
                } else {
                    get(self, b)?
                }
            }
            Op::Convert(Cvt::ZExt | Cvt::Trunc, _, a) => get(self, a)?,
            Op::Convert(Cvt::SExt, _, a) => {
                let from = self.f.types[a.0].bits() as u64;
                signed(get(self, a)?, from) as u64
            }
            _ => return None,
        };
        Some(result & mask)
    }

    fn find_step(
        &mut self,
        header: BlockId,
        lane: usize,
        target: Target,
        region: Option<Region>,
        bits: &[((usize, u8), bool)],
        back: &[(BlockId, usize)],
    ) -> Step {
        let l = lane as u8;
        self.guess_about(header, bits, &[(target, l, region)], |this| {
            let symbol = match target {
                Target::Param(index) => {
                    let v = this.f.blocks[&header].params[index].0;
                    this.symbol(Key::Guess(v, l), header)
                }
                Target::Slot(address, bytes) => this.symbol(Key::GuessSlot((header, address, bytes, l)), header),
            };
            let mut step = None;
            let mut stepped = true;
            let mut keeps = true;
            let mut affine = Some(None);
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
                affine = recurrence(affine, value.as_ref(), symbol);
                if !stepped && !keeps && affine.is_none() {
                    break;
                }
            }
            (if stepped { step } else { None }, keeps, affine.flatten())
        })
    }

    pub(super) fn bounds(&self, form: &Form) -> Option<(u64, u64)> {
        let (mut low, mut high) = (form.constant as u64, form.constant as u64);
        for &(u, c) in &form.terms {
            let (l, h) = self.range(u)?;
            low += c as u64 * l as u64;
            high += c as u64 * h as u64;
        }
        (high < 1 << 32).then_some((low, high))
    }

    fn condition(&self, cond: ValueId, taken: bool) -> (std::rc::Rc<Cond>, bool) {
        let slot = 2 * cond.0 + taken as usize;
        if let Some(found) = &self.conditions.borrow()[slot] {
            return found.clone();
        }
        let tree = condition(self.f, self.facts, &self.copies, cond, taken);
        let found = (tree.clone(), shares(&tree));
        self.conditions.borrow_mut()[slot] = Some(found.clone());
        found
    }

    fn literals(&self, cond: ValueId, taken: bool) -> Vec<(ValueId, bool)> {
        let mut out = Vec::new();
        literals(&self.condition(cond, taken).0, &mut out);
        out
    }

    fn fixed_by(&self, (c, holds): (ValueId, bool)) -> Option<(ValueId, u32)> {
        let Some(Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), x, y)) = self.facts.op(self.f, c) else {
            return None;
        };
        if (p == IntPred::Eq) != holds || !matches!(self.f.types[x.0], Ty::I32 | Ty::I64) {
            return None;
        }
        let root = |v: ValueId| self.copies.get(&v).copied().unwrap_or(v);
        match (self.facts.constant(self.f, x), self.facts.constant(self.f, y)) {
            (_, Some(k)) => Some((root(x), k as u32)),
            (Some(k), None) => Some((root(y), k as u32)),
            _ => None,
        }
    }

    fn fixed_words(&self, cond: ValueId, taken: bool) -> Vec<(ValueId, u32)> {
        self.literals(cond, taken).into_iter().filter_map(|l| self.fixed_by(l)).collect()
    }

    fn assumed_constant(&mut self, a: ValueId, v: ValueId) -> Option<u32> {
        let root = self.copies.get(&v).copied().unwrap_or(v);
        if !self.equated.contains(&root) {
            return None;
        }
        self.fixed_words(a, true).into_iter().find(|&(x, _)| x == root).map(|(_, k)| k)
    }

    fn assumes(&mut self, a: ValueId, v: ValueId) -> bool {
        let known = match self.implied.get(&a) {
            Some(list) => list.contains(&v),
            None => self.assumptions(a).contains(&v),
        };
        known || self.implies(a, v, None)
    }

    fn assumptions(&mut self, predicate: ValueId) -> Vec<ValueId> {
        if let Some(list) = self.implied.get(&predicate) {
            return list.clone();
        }
        let list: Vec<ValueId> = self.literals(predicate, true).into_iter().filter(|l| l.1).map(|l| l.0).collect();
        self.implied.insert(predicate, list.clone());
        list
    }

    fn canonical(&self, v: ValueId, lane: usize) -> usize {
        if self.facts.uniform[v.0] && self.valid(lane) {
            0
        } else {
            lane
        }
    }

    pub fn value(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
        if let Some(k) = assume.and_then(|a| self.assumed_constant(a, v)) {
            return (
                Value::constant(k),
                Reliance {
                    used: true,
                    ..Reliance::default()
                },
            );
        }
        let lane = self.canonical(v, lane);
        let l = lane as u8;
        if let Some((guess, depth)) = self.guessed.get(&(v, l)).cloned() {
            self.depend(depth);
            return unassumed(guess);
        }
        if let Site::Param { block, index } = self.facts.site[v.0] {
            if let Some(depth) = self
                .guessing_about
                .get(&block)
                .map(|g| g.depth)
                .filter(|_| self.carried(v, block, index))
            {
                self.depend(depth);
                let region = self.assumed_entry(block, Target::Param(index), lane);
                let symbol = self.symbol(Key::Guess(v, l), block);
                let guess = Value {
                    form: Form::unknown(symbol),
                    region,
                };
                self.guessed.insert((v, l), (guess.clone(), depth));
                if let Some(guesses) = self.guessing_about.get_mut(&block) {
                    guesses.params.push((v, l));
                }
                return unassumed(guess);
            }
        }
        let hit = match self.values.get(&(v, l, None)) {
            Some(e) if assume.is_none() || !e.0 .1.open => Some(e.clone()),
            _ => assume.and_then(|_| self.values.get(&(v, l, assume)).cloned()),
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
                    values: None,
                },
            );
            return unassumed(Value::of(Form::unknown(u)));
        };
        let (r, depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(level(this.checking));
            }
            this.compute(v, lane, assume)
        });
        self.end(key, outside);
        if assume.is_some() {
            self.cache_value((v, l, assume), r.clone(), depth);
        }
        let root = self.copies.get(&v).copied().unwrap_or(v);
        let r = if assume.is_none() && self.equated.contains(&root) {
            (r.0, Reliance { open: true, ..r.1 })
        } else {
            r
        };
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
                        ..Reliance::default()
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
        let lane = self.canonical(v, lane);
        let l = lane as u8;
        if let Some(&(guess, depth)) = self.guessed_bits.get(&(v, l)) {
            self.depend(depth);
            return unassumed(Some(guess));
        }
        let hit = match self.bits.get(&(v, l, None)) {
            Some(&e) if assume.is_none() || !e.0 .1.open => Some(e),
            _ => assume.and_then(|_| self.bits.get(&(v, l, assume)).copied()),
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
                this.depend(level(this.checking));
            }
            this.compute_bit(v, lane, assume)
        });
        self.end(key, outside);
        if assume.is_some() {
            self.cache_bit((v, l, assume), r, depth);
        }
        if !r.1.used {
            self.cache_bit((v, l, None), r, depth);
        }
        r
    }

    pub fn regions(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>, refine: bool) -> Regions {
        self.assumed_regions(v, lane, assume, refine).0
    }

    pub fn read_regions(&mut self, at: (BlockId, usize), lane: usize, exec: Option<ValueId>, refine: bool) -> Regions {
        let mut set = Regions::one(None);
        if let Inst::Target { args, .. } = &self.f.blocks[&at.0].insts[at.1] {
            for &x in args.values() {
                match self.known[x.0] {
                    Some(r) => set.add(r),
                    None => set.union(&self.regions(x, lane, exec, refine)),
                }
            }
        }
        set
    }

    fn assumed_regions(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>, refine: bool) -> Assumed<Regions> {
        if let Some(r) = self.known[v.0] {
            return unassumed(Regions::one(r));
        }
        let lane = self.canonical(v, lane);
        let l = lane as u8;
        let cache = if refine { &self.refined } else { &self.regions };
        let open = |e: &(Assumed<Regions>, Depth)| assume.is_some() && e.0 .1.open;
        let hit = match (cache.get(&(v, ALL, None)), cache.get(&(v, l, None))) {
            (Some(e), _) if !open(e) => Some(e.clone()),
            (_, Some(e)) if !open(e) => Some(e.clone()),
            _ => assume.and_then(|_| cache.get(&(v, ALL, assume)).or_else(|| cache.get(&(v, l, assume))).cloned()),
        };
        if let Some((r, depth)) = hit {
            self.depend(depth);
            return r;
        }
        let key = (v, l, assume, refine);
        if !self.active_regions.insert(key) {
            return in_lane(Regions::any());
        }
        let mut reliance = Reliance::default();
        let (set, depth) = self.frame(|this| this.compute_regions(v, lane, assume, refine, &mut reliance));
        self.active_regions.remove(&key);
        let r = (set, reliance);
        let cache = |this: &mut Self, key: ValueKey| {
            if depth != FREE {
                this.journal.push((Entry::Region(key, refine), depth));
            }
            let cache = if refine { &mut this.refined } else { &mut this.regions };
            cache.insert(key, (r.clone(), depth));
        };
        let lanes = if reliance.lane { l } else { ALL };
        if assume.is_some() {
            cache(self, (v, lanes, assume));
        }
        if !reliance.used {
            cache(self, (v, lanes, None));
        }
        r
    }

    fn compute_regions(
        &mut self,
        v: ValueId,
        lane: usize,
        assume: Option<ValueId>,
        refine: bool,
        reliance: &mut Reliance,
    ) -> Regions {
        macro_rules! regions {
            ($x:expr) => {{
                let (r, u) = self.assumed_regions($x, lane, assume, refine);
                *reliance |= u;
                r
            }};
        }
        let none = || Regions::one(None);
        if let Some(&root) = self.copies.get(&v) {
            return regions!(root);
        }
        let (f, facts) = (self.f, self.facts);
        match facts.site[v.0] {
            Site::Param { block, index } if block == f.entry => match self.inputs[index].source {
                ParameterSource::Vgpr(n) if n != 0 => Regions::default(),
                ParameterSource::Sgpr(n) if Some(n) == self.entry.kernarg_ptr => Regions::one(Some(Region::Kernarg)),
                ParameterSource::Sgpr(n) if Some(n) == self.entry.dispatch_ptr => Regions::one(Some(Region::Dispatch)),
                _ => none(),
            },
            Site::Param { block, index } => {
                let header = self.headers.contains(&block);
                let own = self.rank[&block];
                let mut set = Regions::default();
                for &e in &facts.incoming[&block] {
                    if header && self.rank[&e.0] >= own {
                        continue;
                    }
                    let (r, u) = self.assumed_regions(self.edge_arg(e, index), lane, None, refine);
                    reliance.lane |= u.lane;
                    set.union(&r);
                }
                if self.plain[v.0] {
                    set.add(None);
                }
                if self.bounds.contains_key(&v) {
                    self.carry(v, lane, refine, &mut set, reliance);
                }
                set
            }
            Site::Inst { block, index } => match &f.blocks[&block].insts[index] {
                Inst::Core { op, .. } => match *op {
                    Op::Int(IntOp::Add, a, b) => regions!(a).combine(&regions!(b), |x, y| match (x, y) {
                        (Some(r), None) | (None, Some(r)) => Some(r),
                        _ => None,
                    }),
                    Op::Int(IntOp::Sub, a, b) => regions!(a).combine(&regions!(b), |x, y| match (x, y) {
                        (Some(r), None) => Some(r),
                        _ => None,
                    }),
                    Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, to, a)
                        if to.bits() >= 32 && f.types[a.0].bits() >= 32 =>
                    {
                        regions!(a)
                    }
                    Op::Pack64(lo, _) | Op::UnpackLo(lo) => regions!(lo),
                    Op::UnpackHi(x) => match facts.op(f, x) {
                        Some(Op::Pack64(_, hi)) => regions!(hi),
                        _ => none(),
                    },
                    Op::Select(c, a, b) => {
                        let root = self.copies.get(&c).copied().unwrap_or(c);
                        if assume.is_some_and(|p| self.assumes(p, root)) {
                            reliance.used = true;
                            return regions!(a);
                        }
                        let (x, y) = (regions!(a), regions!(b));
                        if x.single().is_some() && x == y {
                            return x;
                        }
                        if refine {
                            let (bit, u) = self.bit(c, lane, assume);
                            *reliance |= u;
                            reliance.lane |= !facts.uniform[c.0];
                            match bit {
                                Some(true) => return x,
                                Some(false) => return y,
                                None => {}
                            }
                        }
                        reliance.open = true;
                        x.joined(&y)
                    }
                    Op::Int(IntOp::And, a, b) => {
                        let mask = |x: ValueId| match facts.op(f, x) {
                            Some(Op::Const(_, k)) => Some(k as u32),
                            _ => None,
                        };
                        match (mask(a), mask(b)) {
                            (None, Some(m)) if aligns(m) => regions!(a),
                            (Some(m), None) if aligns(m) => regions!(b),
                            (None, None) => {
                                let mut set = regions!(a).combine(&regions!(b), |x, y| match (x, y) {
                                    (Some(r), None) | (None, Some(r)) => Some(r),
                                    _ => None,
                                });
                                set.add(None);
                                set
                            }
                            _ => none(),
                        }
                    }
                    Op::Int(IntOp::Or | IntOp::Xor, a, b) => {
                        let (x, y) = (regions!(a), regions!(b));
                        if x.any || y.any {
                            return Regions::any();
                        }
                        let mut set = Regions::default();
                        for &r in x.list.iter().chain(&y.list) {
                            if r.is_some() {
                                set.add(r);
                            }
                        }
                        if x.list.contains(&None) && y.list.contains(&None) {
                            set.add(None);
                        }
                        set
                    }
                    Op::Int(IntOp::LShr, a, _) if f.types[v.0] != Ty::I64 => {
                        let mut set = regions!(a);
                        set.add(None);
                        set
                    }
                    Op::Env(Env::ScratchBase) => Regions::one(Some(Region::Private)),
                    _ => none(),
                },
                Inst::Effect {
                    op: EffectOp::Memory {
                        op: MemoryOp::Load(_),
                        space,
                        ..
                    },
                    inputs,
                    ..
                } => {
                    let address = regions!(inputs[0]);
                    let kernarg = address.any || address.list.contains(&Some(Region::Kernarg));
                    if *space == Space::Scratch || kernarg {
                        let (value, u) = self.value(v, lane, assume);
                        *reliance |= u;
                        reliance.lane |= !facts.uniform[v.0];
                        let mut set = Regions::one(value.region);
                        if (self.bounds.contains_key(&v) || self.loaded.contains_key(&v)) && self.unresolved(v, lane, &value) {
                            match self.loaded.get(&v) {
                                Some(loaded) => set.union(loaded),
                                None => self.carry(v, lane, refine, &mut set, reliance),
                            }
                        }
                        set
                    } else {
                        none()
                    }
                }
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::ReadFirstLane),
                    inputs,
                    ..
                } => self.lanes_regions(inputs[0], false),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::ReadLane),
                    inputs,
                    ..
                } => {
                    reliance.lane = true;
                    match self.value(inputs[1], lane, None).0.form.as_constant() {
                        Some(k) if !self.valid((k & 31) as usize) => none(),
                        Some(k) => self.assumed_regions(inputs[0], (k & 31) as usize, None, refine).0,
                        None => {
                            let partial = !(0..LANES).all(|l| self.valid(l));
                            self.lanes_regions(inputs[0], partial)
                        }
                    }
                }
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::WriteLane),
                    inputs,
                    ..
                } => {
                    reliance.lane = true;
                    match self.value(inputs[1], lane, None).0.form.as_constant() {
                        Some(k) if (k & 31) as usize == lane => self.assumed_regions(inputs[0], lane, None, refine).0,
                        Some(_) => self.assumed_regions(inputs[2], lane, None, refine).0,
                        None => {
                            let mut set = self.assumed_regions(inputs[0], lane, None, refine).0;
                            set.union(&self.assumed_regions(inputs[2], lane, None, refine).0);
                            set
                        }
                    }
                }
                Inst::Effect {
                    op: EffectOp::Wave(op @ (WaveOp::Bpermute | WaveOp::BpermuteFi)),
                    inputs,
                    ..
                } => {
                    reliance.lane = true;
                    let silent = *op == WaveOp::Bpermute || !(0..LANES).all(|l| self.valid(l));
                    self.lanes_regions(inputs[1], silent)
                }
                _ => none(),
            },
            Site::Unreached => none(),
        }
    }

    fn carry(&mut self, v: ValueId, lane: usize, refine: bool, set: &mut Regions, reliance: &mut Reliance) {
        for tier in [false, true] {
            if tier && !refine {
                continue;
            }
            if let Some(grown) = self.grown.get(&(v, ALL, tier)) {
                set.union(grown);
            }
            if self.split.contains(&(v, tier)) {
                reliance.lane = true;
                if let Some(grown) = self.grown.get(&(v, lane as u8, tier)) {
                    set.union(grown);
                }
            }
        }
        let bound = &self.bounds[&v];
        if !bound[if bound.len() == 1 { 0 } else { lane }].within(set) {
            let key = (v, if reliance.lane { lane as u8 } else { ALL }, refine);
            self.looped.entry(key).or_insert(lane);
        }
    }

    fn stored_regions(&mut self, v: ValueId, lane: usize, refine: bool, held: &Regions) -> Assumed<Regions> {
        let mut set = Regions::default();
        let Site::Inst { block, index } = self.facts.site[v.0] else {
            return (set, Reliance::default());
        };
        let Inst::Effect {
            op: EffectOp::Memory {
                op: MemoryOp::Load(size), ..
            },
            inputs,
            ..
        } = &self.f.blocks[&block].insts[index]
        else {
            return (set, Reliance::default());
        };
        let (address, mut reliance) = self.value(inputs[0], lane, Some(inputs[1]));
        reliance.lane |= !self.facts.uniform[inputs[0].0];
        let words = self
            .bounds(&address.form)
            .map(|(low, high)| ((low / 4) as u32, (high + size.bytes() as u64).div_ceil(4) as u32));
        let sources = self.sources.get(&v).cloned().unwrap_or_default();
        for i in sources {
            if let (Some((a, b)), Some((c, d))) = (words, self.spills[i].words) {
                if b <= c || d <= a {
                    continue;
                }
            }
            if self.spills[i].part(lane).within(held) {
                continue;
            }
            let mask = self.spills[i].mask;
            for k in 0..self.spills[i].data.len() {
                let (r, u) = self.assumed_regions(self.spills[i].data[k], lane, Some(mask), refine);
                reliance |= u;
                set.union(&r);
            }
        }
        (set, reliance)
    }

    fn feeding(&mut self, v: ValueId, lane: usize, refine: bool, held: &Regions) -> Assumed<Regions> {
        match self.facts.site[v.0] {
            Site::Param { .. } => self.back_regions(v, lane, refine),
            _ => self.stored_regions(v, lane, refine, held),
        }
    }

    fn back_regions(&mut self, v: ValueId, lane: usize, refine: bool) -> Assumed<Regions> {
        let mut set = Regions::default();
        let mut reliance = Reliance::default();
        let Site::Param { block, index } = self.facts.site[v.0] else {
            return (set, reliance);
        };
        let own = self.rank[&block];
        let back: Vec<(BlockId, usize)> = self.facts.incoming[&block]
            .iter()
            .copied()
            .filter(|&(pred, _)| self.rank[&pred] >= own)
            .collect();
        for e in back {
            let (r, u) = self.assumed_regions(self.edge_arg(e, index), lane, None, refine);
            reliance |= u;
            set.union(&r);
        }
        (set, reliance)
    }

    pub fn settle_loops(&mut self) -> bool {
        let mut grew = false;
        loop {
            let pending: Vec<((ValueId, u8, bool), usize)> = self
                .looped
                .iter()
                .filter(|(key, _)| !self.checked.contains(key))
                .map(|(&key, &lane)| (key, lane))
                .collect();
            if pending.is_empty() {
                break;
            }
            for ((v, l, refine), first) in pending {
                self.checked.insert((v, l, refine));
                let held = self.regions(v, first, None, refine);
                let bound = self.bounds[&v].clone();
                let all: Vec<usize> = (0..LANES).filter(|&k| self.valid(k)).collect();
                let (lanes, mut known) = match (l == ALL, bound.len() > 1) {
                    (true, true) => (all, None),
                    (true, false) => {
                        let (found, u) = self.feeding(v, first, false, &held);
                        (if u.lane { all } else { vec![first] }, Some((found, u)))
                    }
                    _ => (vec![first], None),
                };
                for lane in lanes {
                    let own = &bound[if bound.len() == 1 { 0 } else { lane }];
                    if own.within(&held) {
                        continue;
                    }
                    let (found, u) = if !own.any && own.list.iter().all(Option::is_none) {
                        (own.clone(), in_lane(()).1)
                    } else {
                        let cached = if lane == first { known.take() } else { None };
                        let coarse = match cached {
                            Some(coarse) => coarse,
                            None => self.feeding(v, lane, false, &held),
                        };
                        if coarse.0.within(&held) {
                            continue;
                        }
                        if refine {
                            self.feeding(v, lane, true, &held)
                        } else {
                            coarse
                        }
                    };
                    if found.within(&held) {
                        continue;
                    }
                    let key = if l == ALL && !u.lane {
                        (v, ALL, refine)
                    } else {
                        self.split.insert((v, refine));
                        (v, lane as u8, refine)
                    };
                    let grown = self.grown.entry(key).or_default();
                    if !found.within(grown) {
                        grown.union(&found);
                        grew = true;
                    }
                }
            }
        }
        if grew {
            self.regions.clear();
            self.refined.clear();
            self.looped.clear();
            self.checked.clear();
        }
        grew
    }

    fn unresolved(&self, v: ValueId, lane: usize, value: &Value) -> bool {
        let key = Key::Value(v, (!self.facts.uniform[v.0]).then_some(lane as u8));
        let found = self.keys.get(&key);
        value.region.is_none() && value.form.constant == 0 && found.is_some_and(|&(u, _)| value.form.terms == [(u, 1)])
    }

    pub fn pending(&self) -> BTreeSet<(ValueId, u8, bool)> {
        self.looped.keys().copied().collect()
    }

    pub fn forget(&mut self, kept: &BTreeSet<(ValueId, u8, bool)>) {
        self.looped.retain(|key, _| kept.contains(key));
        self.regions.clear();
        self.refined.clear();
    }

    pub fn exposable(&self) -> BTreeSet<u64> {
        self.exposing.iter().flat_map(|e| e.candidates.iter().copied()).collect()
    }

    pub fn expose(&mut self, open: &[u64], stake: &[u64]) -> Vec<u64> {
        let mut found: Vec<u64> = Vec::new();
        let lanes: Vec<usize> = (0..LANES).filter(|&l| self.valid(l)).collect();
        for e in self.exposing.clone() {
            let mut fresh: Vec<u64> = e
                .candidates
                .iter()
                .copied()
                .filter(|id| stake.contains(id) && !open.contains(id) && !found.contains(id))
                .collect();
            for &lane in &lanes {
                for &x in &e.operands {
                    if fresh.is_empty() {
                        break;
                    }
                    let meets = |set: &Regions, id: u64| set.any || set.list.contains(&Some(Region::Allocation(id)));
                    let coarse = self.regions(x, lane, e.assume, false);
                    if !fresh.iter().any(|&id| meets(&coarse, id)) {
                        continue;
                    }
                    let refined = self.regions(x, lane, e.assume, true);
                    fresh.retain(|&id| {
                        let hit = meets(&refined, id);
                        if hit {
                            found.push(id);
                        }
                        !hit
                    });
                }
            }
        }
        found
    }

    fn lanes_regions(&mut self, x: ValueId, silent: bool) -> Regions {
        let mut set = Regions::default();
        if silent {
            set.add(None);
        }
        for l in 0..LANES {
            if self.valid(l) {
                set.union(&self.regions(x, l, None, false));
            }
        }
        set
    }

    fn compute(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
        let (f, facts) = (self.f, self.facts);
        if let Some(&root) = self.copies.get(&v) {
            if self.narrowable[v.0] {
                if let Some(form) = self.narrowed_everywhere(v, lane) {
                    return unassumed(Value::of(form));
                }
            }
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
            ParameterSource::Sgpr(n) if Some(n) == self.entry.kernarg_ptr => self.base(Region::Kernarg),
            ParameterSource::Sgpr(n) if Some(n) == self.entry.dispatch_ptr => self.base(Region::Dispatch),
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

    fn base(&mut self, region: Region) -> Value {
        let entry = self.f.entry;
        let u = self.intern(
            Key::Base(region),
            UnknownInfo {
                rank: 0,
                shared: true,
                block: entry,
                range: None,
                through: Vec::new(),
                values: None,
            },
        );
        Value {
            form: Form::unknown(u),
            region: Some(region),
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
                values: None,
            },
        )
    }

    fn carried(&self, v: ValueId, block: BlockId, index: usize) -> bool {
        let own = self.rank[&block];
        !self.copies.contains_key(&v)
            && self.headers.contains(&block)
            && self.facts.incoming[&block]
                .iter()
                .any(|&(pred, slot)| self.rank[&pred] >= own && self.edge_arg((pred, slot), index) != v)
    }

    fn incoming(&mut self, v: ValueId, block: BlockId, index: usize) -> Option<Vec<ValueId>> {
        Some(self.incoming_edges(v, block, index)?.into_iter().map(|(_, a)| a).collect())
    }

    fn incoming_edges(&mut self, v: ValueId, block: BlockId, index: usize) -> Option<Vec<((BlockId, usize), ValueId)>> {
        let header = self.headers.contains(&block);
        let own = self.rank[&block];
        let mut out = Vec::new();
        let facts = self.facts;
        for &(pred, slot) in &facts.incoming[&block] {
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
            out.push(((pred, slot), arg));
        }
        Some(out)
    }

    fn equal_on_edge(&self, cond: ValueId, taken: bool, arg: ValueId) -> Option<u32> {
        let arg = self.copies.get(&arg).copied().unwrap_or(arg);
        self.fixed_words(cond, taken).into_iter().find(|&(x, _)| x == arg).map(|(_, k)| k)
    }

    fn limits(&mut self, b: BlockId) -> std::rc::Rc<Limits> {
        if let Some((found, depth)) = self.limited.get(&b) {
            let found = found.clone();
            self.depend(*depth);
            return found;
        }
        let outside = match self.limiting.insert(b, self.guessing) {
            Some(level) if level == self.guessing => return std::rc::Rc::default(),
            outside => outside,
        };
        let (limits, depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(level(this.checking));
            }
            this.compute_limits(b)
        });
        match outside {
            Some(level) => self.limiting.insert(b, level),
            None => self.limiting.remove(&b),
        };
        let limits = std::rc::Rc::new(limits);
        if depth != FREE {
            self.journal.push((Entry::Limits(b), depth));
        }
        self.limited.insert(b, (limits.clone(), depth));
        limits
    }

    fn compute_limits(&mut self, b: BlockId) -> Limits {
        if b == self.f.entry {
            return Limits::default();
        }
        if self.headers.contains(&b) {
            let d = self.facts.order[self.idom[self.rank[&b]]];
            return (*self.limits(d)).clone();
        }
        let mut out: Option<Limits> = None;
        for (pred, slot) in self.facts.incoming[&b].clone() {
            let mut here = (*self.limits(pred)).clone();
            if let Some((cond, taken)) = self.edge_condition(pred, slot) {
                let limits = self.condition_limits(cond, taken, None);
                here = both_limits(here, limits);
            }
            let joined = match out {
                None => here,
                Some(old) => either_limits(old, here),
            };
            if joined.is_empty() {
                return joined;
            }
            out = Some(joined);
        }
        out.unwrap_or_default()
    }

    fn condition_limits(&mut self, cond: ValueId, taken: bool, lane: Option<(BlockId, usize)>) -> Limits {
        let (tree, shared) = self.condition(cond, taken);
        let mut memo = shared.then(HashMap::default);
        self.limits_of(&tree, lane, &mut memo)
    }

    fn limits_of(&mut self, cond: &Cond, lane: Option<(BlockId, usize)>, memo: &mut Option<HashMap<(ValueId, bool), Limits>>) -> Limits {
        let (key, parts, all) = match cond {
            Cond::Leaf(c, holds) => return self.comparison_limits(*c, *holds, lane),
            Cond::All(v, holds, parts) => ((*v, *holds), parts, true),
            Cond::Any(v, holds, parts) => ((*v, *holds), parts, false),
        };
        if let Some(found) = memo.as_ref().and_then(|m| m.get(&key)) {
            return found.clone();
        }
        let mut joined: Option<Limits> = None;
        for part in parts {
            let own = self.limits_of(part, lane, memo);
            joined = Some(match joined {
                None => own,
                Some(old) if all => both_limits(old, own),
                Some(old) => either_limits(old, own),
            });
        }
        let limits = joined.unwrap_or_default();
        if let Some(memo) = memo {
            memo.insert(key, limits.clone());
        }
        limits
    }

    fn comparison_limits(&mut self, cond: ValueId, taken: bool, lane: Option<(BlockId, usize)>) -> Limits {
        match self.facts.op(self.f, cond) {
            Some(Op::Cmp(p, x, y)) if self.f.types[x.0] == Ty::I32 && (lane.is_some() || self.facts.uniform[x.0] && self.facts.uniform[y.0]) => {
                let pred = if taken { p } else { negated(p) };
                let (fx, fy) = match lane {
                    Some((at, l)) => (self.operand(x, at, l, None).0.form, self.operand(y, at, l, None).0.form),
                    None => (self.value(x, 0, None).0.form, self.value(y, 0, None).0.form),
                };
                match (fx.as_constant(), fy.as_constant()) {
                    (None, Some(k)) => Limits {
                        classes: vec![class_limit(&fx, &satisfying(pred, k))],
                        orders: Vec::new(),
                    },
                    (Some(k), None) => Limits {
                        classes: vec![class_limit(&fy, &satisfying(swapped(pred), k))],
                        orders: Vec::new(),
                    },
                    (None, None) if fx.sub(&fy).as_constant().is_none() => Limits {
                        classes: Vec::new(),
                        orders: order_limit(&fx, &fy, pred),
                    },
                    _ => Limits::default(),
                }
            }
            _ => Limits::default(),
        }
    }

    fn bounds_at(&mut self, form: &Form, at: BlockId) -> Option<(u64, u64)> {
        let plain = self.bounds(form);
        if form.as_constant().is_some() {
            return plain;
        }
        let limits = self.limits(at);
        let class = Form {
            constant: 0,
            terms: form.terms.clone(),
        };
        if !limits.classes.iter().any(|(c, _)| *c == class) {
            return plain;
        }
        let pieces = self.pieces(form, &limits);
        match (pieces.first(), pieces.last()) {
            (Some(&(low, _)), Some(&(_, high))) => Some((low, high)),
            _ => plain,
        }
    }

    pub fn access_classes(&mut self, block: BlockId, predicate: Option<ValueId>, lane: usize) -> Classes {
        let mut limits = (*self.limits(block)).clone();
        if let Some(p) = predicate {
            let own = self.condition_limits(p, true, Some((block, lane)));
            limits = both_limits(limits, own);
        }
        limits.classes
    }

    pub fn wide_value(&mut self, v: ValueId, at: BlockId, lane: usize) -> Option<Wide> {
        self.wide_at(v, at, lane, 0)
    }

    fn wide_at(&mut self, v: ValueId, at: BlockId, lane: usize, depth: usize) -> Option<Wide> {
        if depth > 12 {
            return None;
        }
        let v = self.copies.get(&v).copied().unwrap_or(v);
        if self.f.types[v.0] != Ty::I64 {
            return None;
        }
        if let Some(low) = self.operand(v, at, lane, None).0.form.as_constant() {
            if let Some(high) = self.high(v, lane).as_constant() {
                return Some(Wide {
                    constant: (high as i128) << 32 | low as i128,
                    terms: Vec::new(),
                    words: Vec::new(),
                });
            }
        }
        let wide = match self.facts.op(self.f, v)? {
            Op::Convert(Cvt::ZExt, Ty::I64, x) if self.f.types[x.0] == Ty::I32 => {
                let form = self.operand(x, at, lane, None).0.form;
                if form.as_constant().is_none() && !(form.constant == 0 && matches!(form.terms.as_slice(), [(_, 1)])) && self.bounds(&form).is_none() {
                    Wide {
                        constant: 0,
                        terms: Vec::new(),
                        words: vec![(form, 1)],
                    }
                } else {
                    Wide {
                        constant: form.constant as i128,
                        terms: form.terms.iter().map(|&(u, c)| (u, c as i128)).collect(),
                        words: Vec::new(),
                    }
                }
            }
            Op::Int(k @ (IntOp::Add | IntOp::Sub), a, b) => {
                let x = self.wide_at(a, at, lane, depth + 1)?;
                let y = self.wide_at(b, at, lane, depth + 1)?;
                x.plus(&y, if k == IntOp::Add { 1 } else { -1 })
            }
            Op::Int(IntOp::Mul, a, b) => match (self.facts.constant(self.f, a), self.facts.constant(self.f, b)) {
                (_, Some(k)) => self.wide_at(a, at, lane, depth + 1)?.times(k as i128),
                (Some(k), _) => self.wide_at(b, at, lane, depth + 1)?.times(k as i128),
                _ => return None,
            },
            Op::Int(IntOp::Shl, a, s) => {
                let k = self.facts.constant(self.f, s)?;
                if k >= 64 {
                    return None;
                }
                self.wide_at(a, at, lane, depth + 1)?.times(1i128 << k)
            }
            _ => return None,
        };
        let (mut low, mut high) = (wide.constant, wide.constant);
        let ranges = wide
            .terms
            .iter()
            .map(|&(u, c)| (c, self.range(u).unwrap_or((0, u32::MAX))))
            .chain(wide.words.iter().map(|&(_, c)| (c, (0, u32::MAX))))
            .collect::<Vec<_>>();
        for (c, (l, h)) in ranges {
            let (a, b) = (c * l as i128, c * h as i128);
            low += a.min(b);
            high += a.max(b);
        }
        (low >= 0 && high < 1 << 64).then_some(wide)
    }

    fn pieces(&self, x: &Form, limits: &Limits) -> Vec<(u64, u64)> {
        let whole = vec![self.bounds(x).unwrap_or((0, u32::MAX as u64))];
        let class = Form {
            constant: 0,
            terms: x.terms.clone(),
        };
        match limits.classes.iter().find(|(c, _)| *c == class) {
            Some((_, set)) => intersected(&whole, &shifted_pieces(set, x.constant)),
            None => whole,
        }
    }

    fn narrowing(&self, (pred, slot): (BlockId, usize), arg: ValueId) -> Option<u32> {
        if !self.narrowing_edges.contains(&(pred, slot)) {
            return None;
        }
        let (cond, taken) = self.edge_condition(pred, slot)?;
        self.equal_on_edge(cond, taken, arg)
    }

    fn narrowed_form(&mut self, (pred, slot): (BlockId, usize), arg: ValueId, at: BlockId, lane: usize) -> Option<Form> {
        if !self.narrowing_edges.contains(&(pred, slot)) {
            return None;
        }
        if let Some(k) = self.narrowing((pred, slot), arg) {
            return Some(Form::constant(k));
        }
        let (cond, taken) = self.edge_condition(pred, slot)?;
        let fixed = self.fixed_words(cond, taken);
        let mut form = self.operand(arg, at, lane, None).0.form;
        let mut changed = false;
        for (x, k) in fixed {
            if self.f.types[x.0] != Ty::I32 {
                continue;
            }
            let fx = self.operand(x, at, lane, None).0.form;
            let &[(t, 1)] = fx.terms.as_slice() else {
                continue;
            };
            let Some(&(_, c)) = form.terms.iter().find(|&&(u, _)| u == t) else {
                continue;
            };
            let value = k.wrapping_sub(fx.constant);
            form = form.sub(&Form::unknown(t).scale(c)).add(&Form::constant(value.wrapping_mul(c)));
            changed = true;
        }
        changed.then_some(form)
    }

    fn narrowed_everywhere(&mut self, v: ValueId, lane: usize) -> Option<Form> {
        let Site::Param { block, index } = self.facts.site[v.0] else {
            return None;
        };
        if block == self.f.entry || !matches!(self.f.types[v.0], Ty::I32 | Ty::I64) {
            return None;
        }
        let facts = self.facts;
        let own = self.rank[&block];
        let root = |this: &Self, x: ValueId| this.copies.get(&x).copied().unwrap_or(x);
        let mut narrowed: Vec<(BlockId, usize, Option<Form>)> = Vec::new();
        for &(pred, slot) in &facts.incoming[&block] {
            let arg = self.edge_arg((pred, slot), index);
            if self.rank[&pred] >= own {
                if arg == v || root(self, arg) == root(self, v) {
                    continue;
                }
                return None;
            }
            let form = if self.f.types[v.0] == Ty::I32 {
                self.narrowed_form((pred, slot), arg, block, lane)
            } else {
                self.narrowing((pred, slot), arg).map(Form::constant)
            };
            narrowed.push((pred, slot, form));
        }
        if narrowed.iter().all(|(_, _, k)| k.is_none()) {
            return None;
        }
        let mut found: Option<Form> = None;
        for (pred, slot, k) in narrowed {
            if k.is_some() && k == found {
                continue;
            }
            if !self.can_take(pred, slot, block) {
                continue;
            }
            let k = k?;
            if found.is_some() {
                return None;
            }
            found = Some(k);
        }
        found
    }

    fn join(&mut self, v: ValueId, block: BlockId, index: usize, lane: usize) -> Assumed<Value> {
        let Some(arguments) = self.incoming_edges(v, block, index) else {
            if let Some(value) = self.stepped(v, block, index, lane) {
                return unassumed(value);
            }
            if let Some(value) = self.recur(block, lane, Target::Param(index)) {
                return unassumed(value);
            }
            if let Some(value) = self.sequence(v, block, index, lane) {
                return unassumed(value);
            }
            let range = self.induction(v, block, index, lane);
            return unassumed(self.opaque(v, lane, range));
        };
        let mut joined: Option<Value> = None;
        let mut region: Option<Option<Region>> = None;
        let mut agreed = true;
        let mut forms: Vec<Form> = Vec::new();
        let narrowable = self.narrowable[v.0] && !self.headers.contains(&block);
        for &(edge, a) in &arguments {
            let narrowed = if !narrowable {
                None
            } else if self.f.types[v.0] == Ty::I32 {
                self.narrowed_form(edge, a, block, lane)
            } else {
                self.narrowing(edge, a).map(Form::constant)
            };
            let value = match narrowed {
                Some(form) => Value::of(form),
                None => self.operand(a, block, lane, None).0,
            };
            region = Some(match region {
                None => value.region,
                Some(r) if r == value.region => r,
                Some(_) => None,
            });
            forms.push(value.form.clone());
            match &joined {
                None => joined = Some(value),
                Some(old) if *old == value => {}
                Some(_) => agreed = false,
            }
        }
        if !agreed {
            let edges: Vec<(BlockId, usize)> = arguments.iter().map(|&(e, _)| e).collect();
            if let Some(form) = self.selected(v, block, &edges, &forms, lane) {
                return unassumed(Value {
                    form,
                    region: region.flatten(),
                });
            }
            let shared = self.facts.uniform[v.0];
            let key = Key::Spread(v, if shared { 0 } else { lane as u8 }, forms.clone());
            let spread = self.spread(key, shared, block, &forms);
            return unassumed(Value {
                region: region.flatten(),
                ..spread.map(Value::of).unwrap_or_else(|| self.opaque(v, lane, None))
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
            Op::Env(Env::ScratchBase) => self.base(Region::Private),
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
                    _ => self.product(v, &a.form, &b.form, lane),
                }
            }
            Op::Int(IntOp::Shl, a, s) => {
                let (a, s) = (get!(self, a), get!(self, s));
                let s = Value::of(match s.form.as_constant() {
                    Some(k) => Form::constant(k & (ty.bits() - 1)),
                    None => s.form,
                });
                match s.form.as_constant() {
                    Some(k) if k < 32 => Value::of(a.form.scale(1 << k)),
                    Some(k) if wide && k < 64 => Value::constant(0),
                    None if !wide || self.bounds(&s.form).is_some_and(|(_, high)| high < 32) => {
                        let power = self.power(v, &s.form, lane);
                        match a.form.as_constant() {
                            Some(k) => Value::of(power.scale(k)),
                            None => self.product(v, &a.form, &power, lane),
                        }
                    }
                    None => Value::of(self.shifted_by(v, IntOp::Shl, &a.form, &s.form, lane)),
                    _ => self.opaque(v, lane, None),
                }
            }
            Op::Int(IntOp::LShr, a, s) if !wide => {
                let (a, s) = (get!(self, a), get!(self, s));
                match s.form.as_constant().map(|k| k & 31) {
                    Some(0) => a,
                    Some(k) => self.shift(v, &a.form, k, lane),
                    None => Value::of(self.shifted_by(v, IntOp::LShr, &a.form, &s.form, lane)),
                }
            }
            Op::Int(IntOp::AShr, a, s) if !wide => {
                let (a, s) = (get!(self, a), get!(self, s));
                match (a.form.as_constant(), s.form.as_constant().map(|k| k & 31)) {
                    (Some(x), Some(k)) => Value::constant(((x as i32) >> k) as u32),
                    (None, Some(k)) if self.bounds(&a.form).is_some_and(|(_, high)| high < 1 << 31) => {
                        self.shift(v, &a.form, k, lane)
                    }
                    (_, None) => Value::of(self.shifted_by(v, IntOp::AShr, &a.form, &s.form, lane)),
                    _ => self.opaque(v, lane, None),
                }
            }
            Op::Int(kind @ (IntOp::LShr | IntOp::AShr), x, s) => {
                let (a, s) = (get!(self, x), get!(self, s));
                let Some(high) = self.known_high(x, at, lane) else {
                    return (self.opaque(v, lane, None), used);
                };
                let signed = kind == IntOp::AShr;
                let non_negative = self.bounds(&high).is_some_and(|(_, top)| top < 1 << 31);
                match s.form.as_constant().map(|k| k & 63) {
                    Some(0) => a,
                    Some(k) if k >= 32 && (!signed || non_negative) => match self.shifted_part(v, &high, k - 32, lane) {
                        Some(form) => Value::of(form),
                        None => self.opaque(v, lane, None),
                    },
                    Some(k) if k < 32 => match self.shifted_part(v, &a.form, k, lane) {
                        Some(form) => Value::of(form.add(&high.scale(1u32 << (32 - k)))),
                        None => self.opaque(v, lane, None),
                    },
                    _ => self.opaque(v, lane, None),
                }
            }
            Op::Int(IntOp::And, a, b) => {
                let (a, b) = (get!(self, a), get!(self, b));
                let region = match (a.form.as_constant(), b.form.as_constant()) {
                    (_, Some(m)) if b.region.is_none() && aligns(m) => a.region,
                    (Some(m), _) if a.region.is_none() && aligns(m) => b.region,
                    _ => None,
                };
                let masked = match (a.form.as_constant(), b.form.as_constant()) {
                    (Some(x), Some(y)) => Value::constant(x & y),
                    (Some(m), None) | (None, Some(m)) => {
                        let form = if a.form.as_constant().is_some() { &b.form } else { &a.form };
                        match low_bits(form, m, IntOp::And) {
                            Some(result) => Value::of(result),
                            None => self.mask(v, form, m, lane),
                        }
                    }
                    _ => Value::of(self.both(v, &a.form, &b.form, lane)),
                };
                Value { region, ..masked }
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
                    (None, Some(c)) if c != 0 => match self.split_low_bits(v, &a.form, c, k, lane) {
                        Some(x) => Value { region: a.region, ..x },
                        None => self.opaque(v, lane, None),
                    },
                    (Some(c), None) if c != 0 => match self.split_low_bits(v, &b.form, c, k, lane) {
                        Some(x) => Value { region: b.region, ..x },
                        None => self.opaque(v, lane, None),
                    },
                    (None, None) => {
                        let both = self.both(v, &a.form, &b.form, lane);
                        let common = if k == IntOp::Or { both } else { both.scale(2) };
                        Value::of(a.form.add(&b.form).sub(&common))
                    }
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
                        } else if let Some(form) = self.chosen(v, c, &x.form, &y.form, lane) {
                            let region = if x.region == y.region { x.region } else { None };
                            Value { form, region }
                        } else {
                            let region = if x.region == y.region { x.region } else { None };
                            let shared = self.facts.uniform[v.0];
                            let forms = vec![x.form.clone(), y.form.clone()];
                            let key = Key::Spread(v, if shared { 0 } else { lane as u8 }, forms.clone());
                            let block = self.block_of(v);
                            match self.spread(key, shared, block, &forms) {
                                Some(form) => Value { form, region },
                                None => Value {
                                    region,
                                    ..self.opaque(v, lane, None)
                                },
                            }
                        }
                    }
                }
            }
            Op::Convert(k @ (Cvt::ZExt | Cvt::SExt), _, a) if self.f.types[a.0] == Ty::I1 => {
                let (bit, u) = self.bit(a, lane, assume);
                used |= u;
                let ones = if k == Cvt::SExt { u32::MAX } else { 1 };
                match bit {
                    Some(b) => Value::constant(if b { ones } else { 0 }),
                    None => Value::of(self.opaque(v, lane, Some((0, 1))).form.scale(ones)),
                }
            }
            Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, to, a)
                if to.bits() >= 32 && self.f.types[a.0].bits() >= 32 =>
            {
                get!(self, a)
            }
            Op::Convert(k @ (Cvt::FloatToSignedSatRtz | Cvt::FloatToUnsignedSatRtz), to, a) => {
                match self.converted(k, to, a) {
                    Some(bits) => Value::constant(bits as u32),
                    None => self.opaque(v, lane, None),
                }
            }
            Op::Pack64(lo, _) | Op::UnpackLo(lo) => get!(self, lo),
            Op::UnpackHi(x) => match self.facts.op(self.f, x) {
                Some(Op::Pack64(_, hi)) => get!(self, hi),
                _ => match self.known_high(x, at, lane) {
                    Some(high) => Value::of(high),
                    None => self.opaque(v, lane, None),
                },
            },
            Op::TrailingZeros(a) | Op::LeadingZeros(a) | Op::PopulationCount(a) | Op::ReverseBits(a)
                if !wide =>
            {
                let x = get!(self, a).form;
                match x.as_constant() {
                    Some(x) => Value::constant(match op {
                        Op::TrailingZeros(_) => x.trailing_zeros(),
                        Op::LeadingZeros(_) => x.leading_zeros(),
                        Op::PopulationCount(_) => x.count_ones(),
                        _ => x.reverse_bits(),
                    }),
                    None => {
                        let (low, high) = self.bounds(&x).map_or((0, u32::MAX), |(l, h)| (l as u32, h as u32));
                        let range = match op {
                            Op::PopulationCount(_) => Some(((low != 0) as u32, 32 - high.leading_zeros())),
                            Op::LeadingZeros(_) => Some((high.leading_zeros(), low.leading_zeros())),
                            Op::TrailingZeros(_) if low != 0 => Some((0, 31 - high.leading_zeros())),
                            Op::TrailingZeros(_) => Some((0, 32)),
                            _ => None,
                        };
                        self.opaque(v, lane, range)
                    }
                }
            }
            _ => self.opaque(v, lane, None),
        };
        (value, used)
    }

    fn converted(&self, k: Cvt, to: Ty, a: ValueId) -> Option<u64> {
        let bits = self.facts.constant(self.f, a)?;
        let x = match self.f.types[a.0] {
            Ty::F32 => f32::from_bits(bits as u32) as f64,
            Ty::F64 => f64::from_bits(bits),
            _ => return None,
        };
        let signed = k == Cvt::FloatToSignedSatRtz;
        Some(match (to, signed) {
            (Ty::I1, true) => (x <= -1.0) as u64,
            (Ty::I1, false) => (x >= 1.0) as u64,
            (Ty::I32, true) => x as i32 as u32 as u64,
            (Ty::I32, false) => x as u32 as u64,
            (Ty::I64, true) => x as i64 as u64,
            (Ty::I64, false) => x as u64,
            _ => return None,
        })
    }

    fn shifted_part(&mut self, v: ValueId, form: &Form, k: u32, lane: usize) -> Option<Form> {
        if let Some(x) = form.as_constant() {
            return Some(Form::constant(x >> k));
        }
        if k == 0 {
            return Some(form.clone());
        }
        form.terms
            .iter()
            .all(|&(u, _)| self.unknowns[u as usize].shared)
            .then(|| self.shift(v, form, k, lane).form)
    }

    fn high_operand(&mut self, x: ValueId, at: BlockId, lane: usize) -> Form {
        let high = self.high(x, lane);
        self.leave(Value::of(high), at, lane).form
    }

    pub fn resource_span(&mut self, at: (BlockId, usize), reach: Reach, lane: usize) -> Option<(Value, u32, Option<Form>)> {
        let Inst::Target { args, outputs, .. } = &self.f.blocks[&at.0].insts[at.1] else {
            return None;
        };
        let (args, v) = (args.values().to_vec(), outputs.first()?.0);
        let word = |this: &mut Self, index: usize| -> Option<Form> {
            let x = *args.get(index)?;
            Some(this.operand(x, at.0, lane, None).0.form)
        };
        let base = word(self, 0)?.scale(256);
        match reach {
            Reach::Anywhere => None,
            Reach::Node { offset, shift, bytes, kinds } => {
                let mut sum = Form::constant(0);
                let mut exact: Option<u64> = Some(0);
                for &(index, mask) in offset {
                    let arg = *args.get(index)?;
                    let x = word(self, index)?;
                    sum = sum.add(&self.masked_part(v, &x, mask as u32, lane)?);
                    exact = match (exact, x.as_constant()) {
                        (Some(acc), Some(low)) => {
                            let high = if self.f.types[arg.0] == Ty::I64 { self.high(arg, lane).as_constant() } else { Some(0) };
                            high.map(|high| acc.wrapping_add((((high as u64) << 32) | low as u64) & mask))
                        }
                        _ => None,
                    };
                }
                let bytes = match kinds {
                    Some((index, mask, table)) => {
                        let x = word(self, index)?;
                        let known = x.terms.iter().all(|&(_, c)| (c as u64) & mask == 0);
                        match known {
                            true => table.iter().find(|&&(kind, _)| kind == (x.constant as u64) & mask).map_or(bytes, |&(_, b)| b),
                            false => bytes,
                        }
                    }
                    None => bytes,
                };
                let narrow = offset.iter().all(|&(index, mask)| {
                    let arg = args[index];
                    self.f.types[arg.0] != Ty::I64 || mask >> 32 == 0 || self.high(arg, lane).as_constant().is_some_and(|h| (h as u64) & (mask >> 32) == 0)
                });
                let high = match (word(self, 0)?.as_constant(), word(self, 1)?.as_constant(), exact) {
                    (Some(r0), Some(r1), Some(offset)) => {
                        let start = ((((r1 & 0xff) as u64) << 40) | ((r0 as u64) << 8)).wrapping_add(offset << shift);
                        Some(Form::constant((start >> 32) as u32))
                    }
                    (Some(r0), Some(r1), None) if narrow => self.bounds(&sum).and_then(|(low, high)| {
                        let origin = (((r1 & 0xff) as u64) << 40) | ((r0 as u64) << 8);
                        let first = origin.wrapping_add(low << shift) >> 32;
                        let last = origin.wrapping_add(high << shift) >> 32;
                        (high < 1 << 32 && high << shift >> shift == high && first == last).then(|| Form::constant(first as u32))
                    }),
                    _ => None,
                };
                Some((Value::of(base.add(&sum.scale(1u32 << shift))), bytes, high))
            }
            Reach::Image => {
                let (w1, w2, w4) = (word(self, 1)?.as_constant()?, word(self, 2)?.as_constant()?, word(self, 4)?.as_constant()?);
                let width = ((w1 >> 30) | (w2 & 0x3fff) << 2) as u64 + 1;
                let height = ((w2 >> 14) & 0xffff) as u64 + 1;
                let pitch = (w4 & 0xffff) as u64;
                let row = if pitch != 0 { pitch + 1 } else { width }.div_ceil(128) * 128;
                let constant = |this: &Self, index: usize| args.get(index).and_then(|&x| this.facts.constant(this.f, x));
                let sampler: Option<Vec<u32>> = (8..12).map(|i| constant(self, i).map(|k| k as u32)).collect();
                let point = sampler.as_ref().is_some_and(|s| crate::buffer::get_bits_u32(s, 84, 2) == 0);
                let coordinates = [self.coordinate(args.get(14).copied(), at.0, lane), self.coordinate(args.get(15).copied(), at.0, lane)];
                let axis = |coord: usize, size: u64, index: usize| -> Option<Option<(u64, u64)>> {
                    let full = Some(Some((0, size - 1)));
                    let (Some(sampler), Some(unrm), Some((low, high))) = (sampler.as_ref(), constant(self, 13), coordinates[coord - 14]) else {
                        return full;
                    };
                    if !point {
                        return full;
                    }
                    let unnormalized = unrm != 0 || crate::buffer::get_bits_u32(sampler, 15, 1) != 0;
                    let texel = |c: f32| if unnormalized { c } else { c * size as f32 }.floor();
                    let (first, last) = (texel(low), texel(high));
                    if !(first >= i32::MIN as f32 && last <= i32::MAX as f32 && last - first <= 4.0 * size as f32) {
                        return full;
                    }
                    let mode = match crate::buffer::get_bits_u32(sampler, index * 3, 3) {
                        0 if unnormalized => 2,
                        1 if unnormalized => 3,
                        mode => mode,
                    };
                    let mut span: Option<(u64, u64)> = None;
                    for t in first as i32..=last as i32 {
                        if let Some(x) = crate::rdna_translator::bvh::clamp_texel(t, size as i32, mode) {
                            let x = x as u64;
                            span = Some(span.map_or((x, x), |(a, b)| (a.min(x), b.max(x))));
                        }
                    }
                    Some(span)
                };
                let (Some(x), Some(y)) = (axis(14, width, 0)?, axis(15, height, 1)?) else {
                    return Some((Value::of(base), 0, None));
                };
                let (start, end) = (y.0 * row + x.0, y.1 * row + x.1 + 1);
                (end < 1 << 31).then(|| (Value::of(base.add(&Form::constant(start as u32))), (end - start) as u32, None))
            }
        }
    }

    fn coordinate(&mut self, x: Option<ValueId>, at: BlockId, lane: usize) -> Option<(f32, f32)> {
        let x = x?;
        if let Some(k) = self.facts.constant(self.f, x) {
            let c = f32::from_bits(k as u32);
            return (!c.is_nan()).then_some((c, c));
        }
        let Some(Op::Convert(cvt @ (Cvt::UnsignedToFloatRte | Cvt::SignedToFloatRte), Ty::F32, i)) = self.facts.op(self.f, x) else {
            return None;
        };
        if self.f.types[i.0] != Ty::I32 {
            return None;
        }
        let form = self.operand(i, at, lane, None).0.form;
        let (low, high) = match form.as_constant() {
            Some(k) => (k as u64, k as u64),
            None => self.bounds(&form)?,
        };
        if high >= 1 << 32 || (cvt == Cvt::SignedToFloatRte && high >= 1 << 31) {
            return None;
        }
        Some((low as f32, high as f32))
    }

    fn masked_part(&mut self, v: ValueId, form: &Form, m: u32, lane: usize) -> Option<Form> {
        if m == u32::MAX {
            return Some(form.clone());
        }
        if let Some(x) = form.as_constant() {
            return Some(Form::constant(x & m));
        }
        if let Some(low) = low_bits(form, m, IntOp::And) {
            return Some(low);
        }
        let aligning = (!m).wrapping_add(1).is_power_of_two();
        let shared = form.terms.iter().all(|&(u, _)| self.unknowns[u as usize].shared);
        (aligning && shared).then(|| self.mask(v, form, m, lane).form)
    }

    pub fn wide(&self, v: ValueId) -> bool {
        self.f.types[v.0] == Ty::I64
    }

    pub fn high(&mut self, v: ValueId, lane: usize) -> Form {
        let v = self.copies.get(&v).copied().unwrap_or(v);
        let lane = self.canonical(v, lane);
        let key = (v, lane as u8);
        let cached = self.highs.get(&key).cloned();
        if let Some((form, depth)) = cached {
            self.depend(depth);
            return form;
        }
        if !self.active_highs.insert(key) {
            return self.opaque_high(v, lane);
        }
        let (found, depth) = self.frame(|this| {
            let found = this.compute_high(v, lane);
            found.unwrap_or_else(|| this.opaque_high(v, lane))
        });
        self.active_highs.remove(&key);
        if depth != FREE {
            self.journal.push((Entry::High(key), depth));
        }
        self.highs.insert(key, (found.clone(), depth));
        found
    }

    fn opaque_high(&mut self, v: ValueId, lane: usize) -> Form {
        let shared = self.facts.uniform[v.0];
        let block = self.block_of(v);
        let u = self.intern(
            Key::High(v, if shared { ALL } else { lane as u8 }),
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range: None,
                through: Vec::new(),
                values: None,
            },
        );
        self.opaque_highs.insert(u);
        Form::unknown(u)
    }

    fn plain_high(&self, x: ValueId) -> bool {
        let x = self.copies.get(&x).copied().unwrap_or(x);
        match self.facts.inst(self.f, x) {
            Some(Inst::Core { op, .. }) => matches!(op, Op::Const(..) | Op::Pack64(..) | Op::Convert(Cvt::ZExt | Cvt::SExt, ..)),
            Some(Inst::Effect {
                op: EffectOp::Memory {
                    op: MemoryOp::Load(MemSize::B64),
                    ..
                },
                ..
            }) => true,
            _ => false,
        }
    }

    fn summed_high(&self, x: ValueId, seen: &mut HashMap<ValueId, bool>) -> bool {
        let x = self.copies.get(&x).copied().unwrap_or(x);
        if let Some(&known) = seen.get(&x) {
            return known;
        }
        seen.insert(x, false);
        let known = self.plain_high(x)
            || matches!(self.facts.op(self.f, x), Some(Op::Int(IntOp::Add | IntOp::Sub, a, b)) if self.summed_high(a, seen) && self.summed_high(b, seen));
        seen.insert(x, known);
        known
    }

    fn known_high(&mut self, x: ValueId, at: BlockId, lane: usize) -> Option<Form> {
        let x = self.copies.get(&x).copied().unwrap_or(x);
        if !self.summed_high(x, &mut HashMap::default()) {
            return None;
        }
        let high = self.high_operand(x, at, lane);
        (!high.terms.iter().any(|(u, _)| self.opaque_highs.contains(u))).then_some(high)
    }

    fn compute_high(&mut self, v: ValueId, lane: usize) -> Option<Form> {
        let f = self.f;
        if f.types[v.0] != Ty::I64 {
            return None;
        }
        let low = |this: &mut Self, x: ValueId, at: BlockId| this.operand(x, at, lane, None).0.form;
        match self.facts.site[v.0] {
            Site::Param { block, .. } if block == f.entry || self.headers.contains(&block) => None,
            Site::Param { block, index } => {
                let mut joined: Option<Form> = None;
                for a in self.incoming(v, block, index)? {
                    let high = self.high_operand(a, block, lane);
                    match &joined {
                        None => joined = Some(high),
                        Some(old) if *old == high => {}
                        Some(_) => return None,
                    }
                }
                joined
            }
            Site::Inst { block, index } => match &f.blocks[&block].insts[index] {
                Inst::Core { op, .. } => match *op {
                    Op::Const(_, k) => Some(Form::constant((k >> 32) as u32)),
                    Op::Pack64(_, hi) => Some(low(self, hi, block)),
                    Op::Convert(Cvt::ZExt, _, _) => Some(Form::constant(0)),
                    Op::Convert(Cvt::SExt, _, a) => {
                        let negative = if f.types[a.0] == Ty::I1 {
                            self.bit(a, lane, None).0?
                        } else {
                            let word = low(self, a, block);
                            let (lowest, highest) = self.bounds(&word)?;
                            if highest < 1 << 31 {
                                false
                            } else if lowest >= 1 << 31 {
                                true
                            } else {
                                return None;
                            }
                        };
                        Some(Form::constant(if negative { u32::MAX } else { 0 }))
                    }
                    Op::Int(k @ (IntOp::Add | IntOp::Sub), a, b) => {
                        let (la, lb) = (low(self, a, block), low(self, b, block));
                        let (ha, hb) = (self.high_operand(a, block, lane), self.high_operand(b, block, lane));
                        let carry = self.word_carry(v, lane, &la, &lb, k == IntOp::Sub);
                        Some(if k == IntOp::Add {
                            ha.add(&hb).add(&carry)
                        } else {
                            ha.sub(&hb).sub(&carry)
                        })
                    }
                    Op::Int(k @ (IntOp::Shl | IntOp::LShr | IntOp::AShr), a, s) => {
                        let amount = low(self, s, block).as_constant()? & 63;
                        let (la, ha) = (low(self, a, block), self.high_operand(a, block, lane));
                        match (k, amount) {
                            (_, 0) => Some(ha),
                            (IntOp::Shl, k) if k >= 32 => Some(la.scale(1u32 << (k - 32))),
                            (IntOp::Shl, k) => {
                                let spilled = self.shifted_part(v, &la, 32 - k, lane)?;
                                Some(ha.scale(1u32 << k).add(&spilled))
                            }
                            (IntOp::LShr, k) if k >= 32 => Some(Form::constant(0)),
                            (_, k) if self.bounds(&ha).is_some_and(|(_, top)| top < 1 << 31) => {
                                if k >= 32 {
                                    Some(Form::constant(0))
                                } else {
                                    self.shifted_part(v, &ha, k, lane)
                                }
                            }
                            (IntOp::LShr, k) => self.shifted_part(v, &ha, k, lane),
                            _ => None,
                        }
                    }
                    Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                        let (m, other) = match (self.facts.constant(f, a), self.facts.constant(f, b)) {
                            (_, Some(m)) => (m, a),
                            (Some(m), _) => (m, b),
                            _ => return None,
                        };
                        let m = (m >> 32) as u32;
                        let high = self.high_operand(other, block, lane);
                        match (k, m, high.as_constant()) {
                            (_, _, Some(h)) => Some(Form::constant(match k {
                                IntOp::And => h & m,
                                IntOp::Or => h | m,
                                _ => h ^ m,
                            })),
                            (IntOp::And, 0, _) => Some(Form::constant(0)),
                            (IntOp::And, u32::MAX, _) | (IntOp::Or | IntOp::Xor, 0, _) => Some(high),
                            _ => None,
                        }
                    }
                    Op::Select(c, a, b) => match self.bit(c, lane, None).0 {
                        Some(true) => Some(self.high_operand(a, block, lane)),
                        Some(false) => Some(self.high_operand(b, block, lane)),
                        None => {
                            let (x, y) = (self.high_operand(a, block, lane), self.high_operand(b, block, lane));
                            (x == y).then_some(x)
                        }
                    },
                    _ => None,
                },
                Inst::Effect {
                    op:
                        EffectOp::Memory {
                            op: MemoryOp::Load(MemSize::B64),
                            ..
                        },
                    inputs,
                    ..
                } => {
                    let address = self.operand(inputs[0], block, lane, None).0;
                    if address.region != Some(Region::Kernarg) {
                        return None;
                    }
                    let at = address.form.sub(&self.base(Region::Kernarg).form).as_constant()?;
                    Some(Form::constant(match self.env.binding(at) {
                        Some(binding) => (binding.pointer >> 32) as u32,
                        None => self.env.kernarg_word(at + 4, 4),
                    }))
                }
                _ => None,
            },
            Site::Unreached => None,
        }
    }

    fn word_carry(&mut self, v: ValueId, lane: usize, la: &Form, lb: &Form, borrow: bool) -> Form {
        let at = self.block_of(v);
        let (bounds_a, bounds_b) = (self.bounds_at(la, at), self.bounds_at(lb, at));
        let decided = if borrow {
            if lb.as_constant() == Some(0) || la == lb {
                Some(false)
            } else {
                match (bounds_a, bounds_b) {
                    (Some((low_a, high_a)), Some((low_b, high_b))) if low_a >= high_b || high_a < low_b => Some(high_a < low_b),
                    _ => None,
                }
            }
        } else if la.as_constant() == Some(0) || lb.as_constant() == Some(0) {
            Some(false)
        } else {
            match (bounds_a, bounds_b) {
                (Some((low_a, high_a)), Some((low_b, high_b))) if high_a + high_b < 1 << 32 || low_a + low_b >= 1 << 32 => {
                    Some(low_a + low_b >= 1 << 32)
                }
                _ => None,
            }
        };
        if let Some(c) = decided {
            return Form::constant(c as u32);
        }
        let shared = self.facts.uniform[v.0];
        let block = self.block_of(v);
        let u = self.intern(
            Key::Carry(v, if shared { ALL } else { lane as u8 }),
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range: Some((0, 1)),
                through: Vec::new(),
                values: None,
            },
        );
        Form::unknown(u)
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

    fn variance(&self, forms: &[&Form], at: BlockId) -> (BlockId, Option<BlockId>) {
        let none: &[BlockId] = &[];
        let mut best: Option<(BlockId, &[BlockId])> = None;
        for form in forms {
            for &(u, _) in &form.terms {
                let b = self.unknowns[u as usize].block;
                let around = self.loops.get(&b).map_or(none, |l| l.as_slice());
                best = match best {
                    None => Some((b, around)),
                    Some((_, current)) if current.iter().all(|h| around.contains(h)) => Some((b, around)),
                    Some(kept) if around.iter().all(|h| kept.1.contains(h)) => Some(kept),
                    Some(_) => {
                        let varying = self.loops.get(&at).map_or(none, |l| l.as_slice()).iter().copied().filter(|h| {
                            forms.iter().any(|f| {
                                f.terms.iter().any(|&(u, _)| self.loops.get(&self.unknowns[u as usize].block).is_some_and(|l| l.contains(h)))
                            })
                        });
                        let inner = varying.max_by_key(|h| self.rank[h]).unwrap_or(self.f.entry);
                        return (inner, Some(inner));
                    }
                };
            }
        }
        match best {
            Some((b, _)) => (b, None),
            None => (at, Some(at)),
        }
    }

    fn derivation(&self, forms: &[&Form], lane: usize) -> (bool, u8, Vec<BlockId>) {
        let shared = forms.iter().all(|f| f.terms.iter().all(|&(u, _)| self.unknowns[u as usize].shared));
        let mut through: Vec<BlockId> = Vec::new();
        for form in forms {
            for h in self.through(form) {
                if !through.contains(&h) {
                    through.push(h);
                }
            }
        }
        (shared, if shared { ALL } else { lane as u8 }, through)
    }

    fn selected(&mut self, v: ValueId, block: BlockId, edges: &[(BlockId, usize)], forms: &[Form], lane: usize) -> Option<Form> {
        let n = forms.len();
        if n < 2 || n != edges.len() {
            return None;
        }
        let first = &forms[0];
        let differences: Vec<Form> = forms.iter().map(|f| f.sub(first)).collect();
        let step = differences[1].as_constant().filter(|&d| {
            differences
                .iter()
                .enumerate()
                .all(|(e, x)| x.as_constant() == Some(d.wrapping_mul(e as u32)))
        });
        let pattern: Option<Vec<u32>> = differences.iter().map(|d| d.as_constant()).collect();
        if step.is_none() && pattern.is_none() && (n != 2 || self.bounds(&differences[1]).or_else(|| self.bounds(&differences[1].scale(u32::MAX))).is_none()) {
            return None;
        }
        let incoming = &self.facts.incoming[&block];
        if incoming.len() > 64 {
            return None;
        }
        let reached = incoming
            .iter()
            .enumerate()
            .filter(|(_, e)| edges.contains(e))
            .fold(0u64, |m, (i, _)| m | 1 << i);
        let through = self.loops.get(&block).cloned().unwrap_or_default();
        if let (None, Some(pattern)) = (step, pattern) {
            let mut values = pattern.clone();
            values.sort_unstable();
            values.dedup();
            let d = self.intern(
                Key::Pattern(block, reached, pattern),
                UnknownInfo {
                    rank: 0,
                    shared: true,
                    block,
                    range: Some((values[0], values[values.len() - 1])),
                    through,
                    values: Some(values.into()),
                },
            );
            return Some(first.add(&Form::unknown(d)));
        }
        let s = self.intern(
            Key::Selector(block, reached),
            UnknownInfo {
                rank: 0,
                shared: true,
                block,
                range: Some((0, n as u32 - 1)),
                through,
                values: None,
            },
        );
        let selector = Form::unknown(s);
        Some(match step {
            Some(d) => first.add(&selector.scale(d)),
            None => first.add(&self.times_selector(v, &differences[1], &selector, lane)),
        })
    }

    fn times_selector(&mut self, v: ValueId, d: &Form, selector: &Form, lane: usize) -> Form {
        if let Some(k) = d.as_constant() {
            return selector.scale(k);
        }
        if self.bounds(d).is_none() {
            let negated = d.scale(u32::MAX);
            if self.bounds(&negated).is_some() {
                return Form::constant(0).sub(&self.product(v, &negated, selector, lane).form);
            }
        }
        self.product(v, d, selector, lane).form
    }

    fn chosen(&mut self, v: ValueId, c: ValueId, x: &Form, y: &Form, lane: usize) -> Option<Form> {
        let difference = x.sub(y);
        if difference.as_constant().is_none() && self.bounds(&difference).or_else(|| self.bounds(&difference.scale(u32::MAX))).is_none() {
            return None;
        }
        let c = self.copies.get(&c).copied().unwrap_or(c);
        let shared = self.facts.uniform[c.0];
        let block = self.block_of(c);
        let through = self.loops.get(&block).cloned().unwrap_or_default();
        let bit = self.intern(
            Key::Chosen(c, if shared { ALL } else { lane as u8 }),
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range: Some((0, 1)),
                through,
                values: None,
            },
        );
        Some(y.add(&self.times_selector(v, &difference, &Form::unknown(bit), lane)))
    }

    fn product(&mut self, v: ValueId, a: &Form, b: &Form, lane: usize) -> Value {
        let degree = |this: &Self, f: &Form| f.terms.iter().map(|&(u, _)| this.factors(u).len()).max().unwrap_or(0);
        if a.terms.len() * b.terms.len() <= 16 && degree(self, a) + degree(self, b) <= 8 {
            let mut out = Form::constant(a.constant.wrapping_mul(b.constant));
            for (x, k) in [(a, b.constant), (b, a.constant)] {
                for &(u, c) in &x.terms {
                    out = out.add(&Form::unknown(u).scale(c.wrapping_mul(k)));
                }
            }
            for &(x, cx) in &a.terms {
                for &(y, cy) in &b.terms {
                    let m = self.monomial(v, x, y, lane);
                    out = out.add(&Form::unknown(m).scale(cx.wrapping_mul(cy)));
                }
            }
            return Value::of(out);
        }
        let (a, b) = if (a.terms.as_slice(), a.constant) <= (b.terms.as_slice(), b.constant) { (a, b) } else { (b, a) };
        let shared = [a, b].iter().all(|f| f.terms.iter().all(|&(u, _)| self.unknowns[u as usize].shared));
        let (block, key) = self.variance(&[a, b], self.block_of(v));
        let product = Key::Product(key, if shared { ALL } else { lane as u8 }, Box::new((a.clone(), b.clone())));
        if let Some(u) = self.known_key(&product) {
            let list = self.derived.entry(u).or_default();
            if !list.contains(&v) {
                list.push(v);
            }
            return Value::of(Form::unknown(u));
        }
        let (shared, l, through) = self.derivation(&[a, b], lane);
        let range = match (self.bounds(a), self.bounds(b)) {
            (Some((la, ha)), Some((lb, hb))) if ha * hb < 1 << 32 => Some(((la * lb) as u32, (ha * hb) as u32)),
            _ => None,
        };
        let u = self.intern(
            Key::Product(key, l, Box::new((a.clone(), b.clone()))),
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range,
                through,
                values: None,
            },
        );
        self.derived.entry(u).or_default().push(v);
        Value::of(Form::unknown(u))
    }

    fn factors(&self, u: Unknown) -> Vec<Unknown> {
        self.monomials.get(&u).cloned().unwrap_or_else(|| vec![u])
    }

    fn monomial(&mut self, v: ValueId, x: Unknown, y: Unknown, lane: usize) -> Unknown {
        let mut factors = self.factors(x);
        factors.extend(self.factors(y));
        factors.sort_unstable();
        let forms: Vec<Form> = factors.iter().map(|&u| Form::unknown(u)).collect();
        let refs: Vec<&Form> = forms.iter().collect();
        let (block, key) = self.variance(&refs, self.block_of(v));
        let monomial = Key::Monomial(key, factors.clone());
        if let Some(u) = self.known_key(&monomial) {
            let list = self.derived.entry(u).or_default();
            if !list.contains(&v) {
                list.push(v);
            }
            return u;
        }
        let (shared, _, through) = self.derivation(&refs, lane);
        let mut range = Some((1u64, 1u64));
        for &u in &factors {
            range = match (range, self.range(u)) {
                (Some((low, high)), Some((l, h))) if high * h as u64 >> 32 == 0 => Some((low * l as u64, high * h as u64)),
                _ => None,
            };
        }
        let u = self.intern(
            monomial,
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range: range.map(|(low, high)| (low as u32, high as u32)),
                through,
                values: None,
            },
        );
        self.monomials.insert(u, factors);
        self.derived.entry(u).or_default().push(v);
        u
    }

    fn both(&mut self, v: ValueId, a: &Form, b: &Form, lane: usize) -> Form {
        let (a, b) = if (a.terms.as_slice(), a.constant) <= (b.terms.as_slice(), b.constant) { (a, b) } else { (b, a) };
        let (shared, l, through) = self.derivation(&[a, b], lane);
        let (block, key) = self.variance(&[a, b], self.block_of(v));
        let top = |bounds: Option<(u64, u64)>| bounds.map_or(u32::MAX, |(_, high)| high as u32);
        let range = Some((0, top(self.bounds(a)).min(top(self.bounds(b)))));
        let u = self.intern(
            Key::Both(key, l, Box::new((a.clone(), b.clone()))),
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range,
                through,
                values: None,
            },
        );
        Form::unknown(u)
    }

    fn shifted_by(&mut self, v: ValueId, kind: IntOp, a: &Form, s: &Form, lane: usize) -> Form {
        let (shared, l, through) = self.derivation(&[a, s], lane);
        let (block, key) = self.variance(&[a, s], self.block_of(v));
        let range = match (kind, self.bounds(a)) {
            (IntOp::LShr, Some((_, high))) if high < 1 << 32 => Some((0, high as u32)),
            (IntOp::AShr, Some((_, high))) if high < 1 << 31 => Some((0, high as u32)),
            _ => None,
        };
        let u = self.intern(
            Key::Shift(key, l, kind, Box::new((a.clone(), s.clone()))),
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range,
                through,
                values: None,
            },
        );
        Form::unknown(u)
    }

    fn power(&mut self, v: ValueId, s: &Form, lane: usize) -> Form {
        let (shared, l, through) = self.derivation(&[s], lane);
        let (block, key) = self.variance(&[s], self.block_of(v));
        let range = match self.bounds(s) {
            Some((low, high)) if high < 32 => Some((1u32 << low, 1u32 << high)),
            _ => Some((1, 1 << 31)),
        };
        let u = self.intern(
            Key::Power(key, l, Box::new(s.clone())),
            UnknownInfo {
                rank: 0,
                shared,
                block,
                range,
                through,
                values: None,
            },
        );
        Form::unknown(u)
    }

    fn shift(&mut self, v: ValueId, form: &Form, k: u32, lane: usize) -> Value {
        if let Some(x) = form.as_constant() {
            return Value::constant(x >> k);
        }
        if k == 0 {
            return Value::of(form.clone());
        }
        let at = self.block_of(v);
        if let Some((low, high)) = self.bounds_at(form, at) {
            if low >> k == high >> k {
                return Value::constant((low >> k) as u32);
            }
        }
        if form.terms.iter().any(|&(u, _)| !self.unknowns[u as usize].shared) {
            return self.opaque(v, lane, None);
        }
        let terms = Form {
            constant: 0,
            terms: form.terms.clone(),
        };
        let bounds = self.bounds(&terms);
        let c = form.constant;
        let (block, key) = self.variance(&[&terms], self.block_of(v));
        let mut wrap = 0u32;
        let mut result = if k <= terms.alignment() {
            if bounds.is_none() {
                wrap = (1u32 << k) - 1;
            }
            Form {
                constant: 0,
                terms: form.terms.iter().map(|&(u, c)| (u, c >> k)).collect(),
            }
        } else {
            let (low, high) = bounds.unwrap_or((0, u32::MAX as u64));
            let through = self.through(&terms);
            let u = self.intern(
                Key::Shifted(key, form.terms.clone(), k),
                UnknownInfo {
                    rank: 0,
                    shared: true,
                    block,
                    range: Some(((low >> k) as u32, (high >> k) as u32)),
                    through,
                    values: None,
                },
            );
            self.derived.entry(u).or_default().push(v);
            let mut high_part = Form::unknown(u);
            let below = c & ((1u32 << k) - 1);
            if below != 0 {
                let carry = self.intern(
                    Key::ShiftCarry(key, form.terms.clone(), k, below),
                    UnknownInfo {
                        rank: 0,
                        shared: true,
                        block,
                        range: Some((0, 1)),
                        through: Vec::new(),
                        values: None,
                    },
                );
                high_part = high_part.add(&Form::unknown(carry));
            }
            high_part
        };
        result = result.add(&Form::constant(c >> k));
        let wraps = c != 0 && bounds.is_none_or(|(_, high)| high + c as u64 >= 1 << 32);
        if wraps {
            wrap += 1;
        }
        if wrap > 0 {
            let w = self.intern(
                Key::ShiftWrap(key, form.terms.clone(), k, c),
                UnknownInfo {
                    rank: 0,
                    shared: true,
                    block,
                    range: Some((0, wrap)),
                    through: Vec::new(),
                    values: None,
                },
            );
            result = result.sub(&Form::unknown(w).scale(1u32 << (32 - k)));
        }
        Value::of(result)
    }

    fn mask(&mut self, v: ValueId, form: &Form, m: u32, lane: usize) -> Value {
        if m == u32::MAX {
            return Value::of(form.clone());
        }
        let at = self.block_of(v);
        if let Some((_, high)) = self.bounds_at(form, at) {
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
            if let Some((low, high)) = self.bounds(&rest) {
                if high <= u32::MAX as u64 && low & !(m as u64) == high & !(m as u64) {
                    return Value::of(rest.sub(&Form::constant((low & !(m as u64)) as u32)));
                }
            }
            let j = rest.alignment();
            if rest.terms.iter().all(|&(u, _)| self.unknowns[u as usize].shared) {
                let (block, key) = self.variance(&[&rest], self.block_of(v));
                let through = self.through(&rest);
                let terms = Form {
                    constant: 0,
                    terms: rest.terms.clone(),
                };
                let whole = match self.bounds(&terms) {
                    Some((low, high)) if low & !(m as u64) == high & !(m as u64) => {
                        Some((terms.sub(&Form::constant((low & !(m as u64)) as u32)), high - (low & !(m as u64))))
                    }
                    _ => None,
                };
                let (masked, top) = match whole {
                    Some(exact) => exact,
                    None => {
                        let u = self.intern(
                            Key::Masked(key, rest.terms.clone(), m),
                            UnknownInfo {
                                rank: 0,
                                shared: true,
                                block,
                                range: Some((0, m >> j)),
                                through,
                                values: None,
                            },
                        );
                        self.derived.entry(u).or_default().push(v);
                        (Form::unknown(u).scale(1 << j), m as u64)
                    }
                };
                let below = form.constant & ((1u32 << j) - 1);
                let carried = (form.constant - below) & m;
                let mut result = masked.add(&Form::constant(below));
                if carried != 0 && top + carried as u64 + below as u64 > m as u64 {
                    let carry = self.intern(
                        Key::MaskCarry(key, rest.terms.clone(), m, carried + below),
                        UnknownInfo {
                            rank: 0,
                            shared: true,
                            block,
                            range: Some((0, 1)),
                            through: Vec::new(),
                            values: None,
                        },
                    );
                    result = result.add(&Form::constant(carried)).sub(&Form::unknown(carry).scale(m + 1));
                } else {
                    result = result.add(&Form::constant(carried));
                }
                return Value::of(result);
            }
            return self.opaque(v, lane, Some((0, m)));
        }
        let high = !m;
        if high.wrapping_add(1).is_power_of_two() {
            let k = high.count_ones();
            if form.alignment() >= k {
                return Value::of(form.sub(&Form::constant(form.constant & high)));
            }
            if form.terms.iter().all(|&(u, _)| self.unknowns[u as usize].shared) {
                let above = self.shift(v, form, k, lane);
                return Value::of(above.form.scale(1 << k));
            }
        }
        let s = m.trailing_zeros();
        let field = m >> s;
        if s > 0 && field.wrapping_add(1).is_power_of_two() && form.terms.iter().all(|&(u, _)| self.unknowns[u as usize].shared) {
            let above = self.shift(v, form, s, lane);
            let inner = self.mask(v, &above.form, field, lane);
            return Value::of(inner.form.scale(1 << s));
        }
        self.opaque(v, lane, Some((0, m)))
    }

    fn split_low_bits(&mut self, v: ValueId, form: &Form, c: u32, op: IntOp, lane: usize) -> Option<Value> {
        if form.terms.iter().any(|&(u, _)| !self.unknowns[u as usize].shared) {
            return None;
        }
        let j = 32 - c.leading_zeros();
        let top = if j == 32 { u32::MAX } else { (1u32 << j) - 1 };
        let above = if j == 32 { Form::constant(0) } else { self.shift(v, form, j, lane).form.scale(1 << j) };
        if op == IntOp::Or && c == top {
            return Some(Value::of(above.add(&Form::constant(c))));
        }
        let range = if op == IntOp::Or { (c, top) } else { (0, top) };
        let block = self.block_of(v);
        let through = self.through(form);
        let low = self.intern(
            Key::Low(v, form.clone(), c),
            UnknownInfo {
                rank: 0,
                shared: true,
                block,
                range: Some(range),
                through,
                values: None,
            },
        );
        Some(Value::of(above.add(&Form::unknown(low))))
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
                if let Some(value) = self.read_back(v, lane) {
                    return (value, used);
                }
                (self.load(v, &address, size, lane), used)
            }
            EffectOp::Wave(WaveOp::ReadFirstLane) => {
                let mut first = Some(0);
                let lanes: Vec<usize> = (0..LANES).filter(|&l| self.valid(l)).collect();
                for l in lanes {
                    match self.bit(inputs[1], l, None).0 {
                        Some(true) => {
                            first = Some(l);
                            break;
                        }
                        Some(false) => {}
                        None => {
                            first = None;
                            break;
                        }
                    }
                }
                unassumed(self.uniform_read(v, inputs[0], first))
            }
            EffectOp::Wave(WaveOp::ReadLane) => {
                let (selector, _) = self.value(inputs[1], lane, None);
                match selector.form.as_constant().map(|k| (k & 31) as usize) {
                    Some(chosen) if !self.valid(chosen) => unassumed(Value::constant(0)),
                    Some(chosen) => unassumed(self.uniform_read(v, inputs[0], Some(chosen))),
                    None => unassumed(self.any_lane_read(v, inputs[0], inputs[1], lane)),
                }
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
            EffectOp::Wave(op @ (WaveOp::Bpermute | WaveOp::BpermuteFi)) => {
                let (index, _) = self.value(inputs[0], lane, None);
                let Some(byte) = index.form.as_constant() else {
                    return unassumed(self.opaque(v, lane, None));
                };
                let source = ((byte >> 2) & 31) as usize;
                if !self.valid(source) {
                    return unassumed(Value::constant(0));
                }
                let taken = match op {
                    WaveOp::Bpermute => self.bit(inputs[2], source, None).0,
                    _ => Some(true),
                };
                let at = self.block_of(v);
                match taken {
                    Some(true) => unassumed(self.operand(inputs[1], at, source, None).0),
                    Some(false) => unassumed(Value::constant(0)),
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

    fn any_lane_read(&mut self, v: ValueId, x: ValueId, selector: ValueId, lane: usize) -> Value {
        let at = self.block_of(v);
        let mut agreed: Option<Value> = None;
        let mut differ = false;
        let lanes: Vec<usize> = (0..LANES).filter(|&l| self.valid(l)).collect();
        for l in lanes {
            let (value, _) = self.operand(x, at, l, None);
            differ |= agreed.as_ref().is_some_and(|a| *a != value);
            agreed = Some(value);
        }
        if !(0..LANES).all(|l| self.valid(l)) && agreed.as_ref().is_some_and(|a| a.form.as_constant() != Some(0)) {
            differ = true;
        }
        match agreed {
            Some(value) if !differ => value,
            _ if self.facts.uniform[selector.0] => self.uniform(v),
            _ => self.opaque(v, lane, None),
        }
    }

    fn read_back(&mut self, v: ValueId, lane: usize) -> Option<Value> {
        let Site::Inst { block, index } = self.facts.site[v.0] else {
            return None;
        };
        let found = match self.readbacks.get(&(block, index)) {
            Some(&found) => found,
            None => {
                let found = self.written_back(block, index);
                self.readbacks.insert((block, index), found);
                found
            }
        };
        let (data, base, target) = found?;
        if let Some(base) = base {
            let start = self.operand(base, block, lane, None).0;
            let wide = self.f.types[base.0] == Ty::I64;
            if start.form.as_constant().is_none() || (wide && self.high(base, lane).as_constant().is_none()) {
                return None;
            }
        }
        if let Some(target) = target {
            let Inst::Effect { inputs, .. } = &self.f.blocks[&block].insts[index] else {
                return None;
            };
            if self.operand(inputs[0], block, lane, None).0.region != Some(Region::Allocation(target)) {
                return None;
            }
        }
        Some(self.operand(data, block, lane, None).0)
    }

    fn written_back(&self, block: BlockId, index: usize) -> Option<(ValueId, Option<ValueId>, Option<u64>)> {
        let (f, facts) = (self.f, self.facts);
        let root = |x: ValueId| self.copies.get(&x).copied().unwrap_or(x);
        let Inst::Effect {
            op:
                EffectOp::Memory {
                    space,
                    op: MemoryOp::Load(MemSize::B32),
                    ..
                },
            inputs,
            ..
        } = &f.blocks[&block].insts[index]
        else {
            return None;
        };
        let space = *space;
        if !matches!(space, Space::Global | Space::Lds) {
            return None;
        }
        let (address, predicate) = (root(inputs[0]), root(inputs[1]));
        let store_in = |b: BlockId, before: usize| {
            f.blocks[&b].insts[..before].iter().enumerate().rev().find_map(|(i, inst)| match inst {
                Inst::Effect {
                    op:
                        EffectOp::Memory {
                            space: s,
                            op: MemoryOp::Store(MemSize::B32),
                            ..
                        },
                    inputs,
                    ..
                } if *s == space && root(inputs[0]) == address => Some((i, root(inputs[1]), root(inputs[2]))),
                _ => None,
            })
        };
        let (mut at, mut before) = (block, index);
        let (store_block, (store, data, mask)) = loop {
            if let Some(found) = store_in(at, before) {
                break (at, found);
            }
            let r = self.rank[&at];
            if r == 0 {
                return None;
            }
            at = facts.order[self.idom[r]];
            before = f.blocks[&at].insts.len();
        };
        if mask != predicate && facts.constant(f, mask) != Some(1) {
            return None;
        }
        let base = self.determined(address, data, space == Space::Global)?;
        if space == Space::Lds {
            let alone = facts.order.iter().all(|&b| {
                f.blocks[&b].insts.iter().enumerate().all(|(i, inst)| match inst {
                    Inst::Effect {
                        op: EffectOp::Memory { space: Space::Lds, op, .. },
                        ..
                    } => matches!(op, MemoryOp::Load(_) | MemoryOp::Fence) || (b, i) == (store_block, store),
                    _ => true,
                })
            });
            return alone.then_some((data, base, None));
        }
        let set = self.written.get(&(store_block, store))?;
        let [Some(Region::Allocation(target))] = set.list.iter().filter(|r| r.is_some()).copied().collect::<Vec<_>>()[..] else {
            return None;
        };
        if set.any || self.env.exposed.contains(&target) || self.exposable().contains(&target) {
            return None;
        }
        let region = Some(Region::Allocation(target));
        let alone = self.written.iter().all(|(&at, set)| at == (store_block, store) || (!set.any && !set.list.contains(&region)));
        alone.then_some((data, base, Some(target)))
    }

    fn determined(&self, address: ValueId, data: ValueId, wide: bool) -> Option<Option<ValueId>> {
        let (f, facts) = (self.f, self.facts);
        let root = |x: ValueId| self.copies.get(&x).copied().unwrap_or(x);
        let top = |x: ValueId| -> Option<u64> {
            match (facts.op(f, x), facts.site[x.0]) {
                (Some(Op::Env(Env::LaneId)), _) => Some(LANES as u64 - 1),
                (_, Site::Param { block, index }) if block == f.entry && matches!(self.inputs[index].source, ParameterSource::Vgpr(0)) => {
                    let [bx, by, bz] = self.env.block.map(|n| n.max(1) as u64 - 1);
                    Some(bx | by << 10 | bz << 20)
                }
                (Some(Op::Int(IntOp::And, a, b)), _) => facts.constant(f, a).or(facts.constant(f, b)).map(|k| k & 0xffff_ffff),
                _ => match facts.inst(f, x) {
                    Some(Inst::Effect {
                        op: EffectOp::Memory { op: MemoryOp::Load(MemSize::U8), .. },
                        ..
                    }) => Some(0xff),
                    Some(Inst::Effect {
                        op: EffectOp::Memory { op: MemoryOp::Load(MemSize::U16), .. },
                        ..
                    }) => Some(0xffff),
                    _ => None,
                },
            }
        };
        let scaled = |offset: ValueId| -> Option<(ValueId, u64)> {
            let (inner, scale) = match facts.op(f, root(offset))? {
                Op::Int(IntOp::Mul, a, b) => match (facts.constant(f, a), facts.constant(f, b)) {
                    (_, Some(k)) => (a, k),
                    (Some(k), _) => (b, k),
                    _ => return None,
                },
                Op::Int(IntOp::Shl, a, s) => (a, 1u64 << (facts.constant(f, s)? & if wide { 63 } else { 31 })),
                _ => return None,
            };
            if wide {
                let Some(Op::Convert(Cvt::ZExt, Ty::I64, d)) = facts.op(f, root(inner)) else {
                    return None;
                };
                (scale <= 1 << 32).then_some((root(d), scale))
            } else {
                Some((root(inner), scale))
            }
        };
        let mut rest = root(address);
        let mut indices: Vec<(ValueId, u64)> = Vec::new();
        let base = loop {
            if indices.len() > 4 {
                return None;
            }
            match facts.op(f, rest) {
                Some(Op::Int(IntOp::Add, a, b)) => match (scaled(b), scaled(a)) {
                    (Some(index), _) => {
                        indices.push(index);
                        rest = root(a);
                    }
                    (None, Some(index)) => {
                        indices.push(index);
                        rest = root(b);
                    }
                    (None, None) if indices.is_empty() => return None,
                    (None, None) => break Some(rest),
                },
                _ => match scaled(rest) {
                    Some(index) if !wide => {
                        indices.push(index);
                        break None;
                    }
                    _ if indices.is_empty() => return None,
                    _ => break Some(rest),
                },
            }
        };
        indices.sort_by_key(|&(_, scale)| scale);
        let mut reach: u128 = 0;
        let last = indices.len() - 1;
        for (i, &(x, scale)) in indices.iter().enumerate() {
            if (scale as u128) < reach + 4 {
                return None;
            }
            match top(x) {
                Some(t) => reach += scale as u128 * t as u128,
                None if wide && i == last => {}
                None => return None,
            }
        }
        if !wide && reach >= 1 << 32 {
            return None;
        }
        let allowed: Vec<ValueId> = indices.iter().map(|&(x, _)| x).collect();
        self.only_of(data, &allowed, 0).then_some(base)
    }

    fn only_of(&self, v: ValueId, allowed: &[ValueId], depth: usize) -> bool {
        let v = self.copies.get(&v).copied().unwrap_or(v);
        if allowed.contains(&v) {
            return true;
        }
        if depth > 16 {
            return false;
        }
        match self.facts.op(self.f, v) {
            Some(Op::Const(..)) => true,
            Some(op @ (Op::Int(..) | Op::Cmp(..) | Op::Select(..) | Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc, _, _))) => {
                let mut only = true;
                op.map(|a| {
                    only &= self.only_of(a, allowed, depth + 1);
                    a
                });
                only
            }
            _ => false,
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
                values: None,
            },
        );
        Value::of(Form::unknown(u))
    }

    fn load(&mut self, v: ValueId, address: &Value, size: MemSize, lane: usize) -> Value {
        let offset = match address.region {
            Some(r @ (Region::Kernarg | Region::Dispatch)) => address.form.sub(&self.base(r).form).as_constant(),
            _ => None,
        };
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
                if let (MemSize::B32 | MemSize::B64, Some(a)) = (size, address.form.as_constant()) {
                    if let Some(value) = self.slot_before((block, index), a, size.bytes(), lane) {
                        return self.leave(value, block, lane);
                    }
                } else if matches!(size, MemSize::B32 | MemSize::B64) {
                    if let Some(value) = self.slot_in_block((block, index), &address.form, size.bytes(), lane) {
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
                    if let Some(bit) = self.loop_bit(block, index, lane) {
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
                Inst::Core { op, .. } => self.core_bit(op, block, lane, assume),
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

    fn core_bit(&mut self, op: Op, block: BlockId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
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
                                Some(k) if k < 32 && (self.facts.uniform[w.0] || k as usize == lane) => {
                                    self.word_bit(w, k as usize)
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    },
                }
            }
            Op::Convert(k @ (Cvt::FloatToSignedSatRtz | Cvt::FloatToUnsignedSatRtz), Ty::I1, x) => {
                self.converted(k, Ty::I1, x).map(|b| b != 0)
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
                let plain = match (pred, difference.as_constant()) {
                    (IntPred::Ne, Some(d)) if d != 0 => Some(true),
                    (IntPred::Eq, Some(d)) if d != 0 => Some(false),
                    _ if wide => None,
                    (IntPred::Eq | IntPred::Ule | IntPred::Uge | IntPred::Sle | IntPred::Sge, Some(0)) => Some(true),
                    (IntPred::Ne | IntPred::Ult | IntPred::Ugt | IntPred::Slt | IntPred::Sgt, Some(0)) => Some(false),
                    (_, Some(d)) => match (self.bounds(&x.form), self.bounds(&y.form)) {
                        (Some(bx), Some(by)) => decide(pred, bx, by).or_else(|| offset(pred, d, by)),
                        (_, Some(by)) => offset(pred, d, by),
                        _ => None,
                    },
                    _ => match (self.bounds(&x.form), self.bounds(&y.form)) {
                        (Some(bx), Some(by)) => decide(pred, bx, by),
                        _ => None,
                    },
                };
                if plain.is_some() || wide {
                    plain
                } else {
                    let limits = self.limits(block);
                    let mut encoding = Encoding::new(&self.unknowns, &|_: &UnknownInfo| false);
                    encoding.decide(pred, &x.form, &y.form, &limits.classes, &limits.orders)
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

pub(super) const PRIVATE_MEMORY: ValueId = ValueId(usize::MAX);

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

pub(super) struct Copies(Vec<ValueId>);

impl Copies {
    pub(super) fn get(&self, v: &ValueId) -> Option<&ValueId> {
        let root = &self.0[v.0];
        (root != v).then_some(root)
    }

    pub(super) fn contains_key(&self, v: &ValueId) -> bool {
        self.0[v.0] != *v
    }
}

fn copies(f: &Func, facts: &Facts) -> Copies {
    const NONE: u32 = u32::MAX;
    let n = f.types.len();
    let edges: Vec<Vec<&[ValueId]>> = facts
        .order
        .iter()
        .map(|b| {
            facts.incoming[b]
                .iter()
                .map(|&(pred, slot)| &f.blocks[&pred].term.edges().nth(slot).unwrap().args[..])
                .collect()
        })
        .collect();
    let mut place = vec![(NONE, 0u32); n];
    let mut params: Vec<ValueId> = Vec::new();
    for (r, &b) in facts.order.iter().enumerate() {
        if b == f.entry {
            continue;
        }
        for (index, &(p, _)) in f.blocks[&b].params.iter().enumerate() {
            params.push(p);
            place[p.0] = (r as u32, index as u32);
        }
    }
    let args = |p: ValueId| {
        let (r, index) = place[p.0];
        edges[r as usize].iter().map(move |e| e[index as usize])
    };
    let mut order = vec![NONE; n];
    let mut low = vec![0u32; n];
    let mut on = vec![false; n];
    let mut component = vec![NONE; n];
    let mut root: Vec<ValueId> = (0..n).map(ValueId).collect();
    let mut stack: Vec<ValueId> = Vec::new();
    let mut members: Vec<ValueId> = Vec::new();
    let (mut next, mut count) = (0u32, 0u32);
    for &start in &params {
        if order[start.0] != NONE {
            continue;
        }
        let mut frames: Vec<(ValueId, usize)> = vec![(start, 0)];
        order[start.0] = next;
        low[start.0] = next;
        next += 1;
        stack.push(start);
        on[start.0] = true;
        while let Some(&mut (v, ref mut at)) = frames.last_mut() {
            let (r, index) = place[v.0];
            let list = &edges[r as usize];
            if *at < list.len() {
                let a = list[*at][index as usize];
                *at += 1;
                if place[a.0].0 == NONE {
                    continue;
                }
                if order[a.0] == NONE {
                    order[a.0] = next;
                    low[a.0] = next;
                    next += 1;
                    stack.push(a);
                    on[a.0] = true;
                    frames.push((a, 0));
                } else if on[a.0] {
                    low[v.0] = low[v.0].min(order[a.0]);
                }
                continue;
            }
            frames.pop();
            if let Some(&(parent, _)) = frames.last() {
                low[parent.0] = low[parent.0].min(low[v.0]);
            }
            if low[v.0] != order[v.0] {
                continue;
            }
            members.clear();
            loop {
                let x = stack.pop().unwrap();
                on[x.0] = false;
                component[x.0] = count;
                members.push(x);
                if x == v {
                    break;
                }
            }
            let mut outside: Option<ValueId> = None;
            let mut single = true;
            'scan: for &p in &members {
                for a in args(p) {
                    let a = root[a.0];
                    if component[a.0] == count {
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
                for &p in &members {
                    root[p.0] = value;
                }
            }
            count += 1;
        }
    }
    Copies(root)
}

impl Addresses<'_> {
    fn aperture(&mut self, a: ValueId, b: ValueId, lane: usize) -> Option<bool> {
        let (f, facts) = (self.f, self.facts);
        let (Some(Op::Cmp(IntPred::Uge, base, low)), Some(Op::Cmp(IntPred::Ult, again, high))) = (facts.op(f, a), facts.op(f, b))
        else {
            return None;
        };
        let env = |x: ValueId| match facts.op(f, x) {
            Some(Op::Env(e)) => Some(e),
            _ => None,
        };
        let bounded = match facts.op(f, high) {
            Some(Op::Int(IntOp::Add, x, y)) => matches!(
                (env(x), env(y)),
                (Some(Env::ScratchBase), Some(Env::ScratchSize)) | (Some(Env::ScratchSize), Some(Env::ScratchBase))
            ),
            _ => false,
        };
        if base != again || env(low) != Some(Env::ScratchBase) || !bounded {
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
                Some(Op::Select(c, x, y)) if self.source(y, lane) == (v, lane) && self.implies(mask, c, edge) => x,
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
        loop {
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

    fn edges_into(&mut self, block: BlockId) -> (Edges, Edges) {
        let own = self.rank[&block];
        let facts = self.facts;
        let taken: Edges = facts.incoming[&block]
            .iter()
            .copied()
            .filter(|&(pred, slot)| self.can_take(pred, slot, block))
            .collect();
        taken.into_iter().partition(|&(pred, _)| self.rank[&pred] < own)
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
                values: None,
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
                values: None,
            },
        )
    }

    fn last_trip(&mut self, header: BlockId, trips: Unknown) -> Option<u32> {
        let own = self.rank[&header];
        let mut guards: Vec<(ValueId, bool)> = Vec::new();
        for &(pred, slot) in &self.facts.incoming[&header].clone() {
            if self.rank[&pred] < own {
                continue;
            }
            let edge = self.guard(pred, slot)?;
            if !guards.contains(&edge) {
                guards.push(edge);
            }
        }
        let linear = |f: &Form| match f.terms.as_slice() {
            [] => Some((f.constant, 0)),
            [(u, k)] if *u == trips => Some((f.constant, *k)),
            _ => None,
        };
        let mut shape: Option<(IntPred, bool, (u32, u32), (u32, u32))> = None;
        for (cond, taken) in guards {
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
            let guard = (pred, taken, linear(&x)?, linear(&y)?);
            match shape {
                None => shape = Some(guard),
                Some(old) if old == guard => {}
                Some(_) => return None,
            }
        }
        let (pred, taken, x, y) = shape?;
        first_failure(pred, taken, x, y)
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

    fn word_bit(&mut self, w: ValueId, bit: usize) -> Option<bool> {
        let w = self.copies.get(&w).copied().unwrap_or(w);
        let key = (w, bit as u8);
        let cached = self.word_bits.get(&key).copied();
        if let Some((r, depth)) = cached {
            self.depend(depth);
            return r;
        }
        let active = (w, bit as u8);
        let outside = match self.active_words.insert(active, self.guessing) {
            Some(level) if level == self.guessing => return None,
            outside => outside,
        };
        let (r, depth) = self.frame(|this| {
            if outside.is_some() {
                this.depend(level(this.checking));
            }
            this.compute_word_bit(w, bit)
        });
        match outside {
            Some(level) => self.active_words.insert(active, level),
            None => self.active_words.remove(&active),
        };
        if depth != FREE {
            self.journal.push((Entry::WordBit(key), depth));
        }
        self.word_bits.insert(key, (r, depth));
        r
    }

    fn compute_word_bit(&mut self, w: ValueId, bit: usize) -> Option<bool> {
        if let Some(k) = self.value(w, bit, None).0.form.as_constant() {
            return Some(k >> bit & 1 != 0);
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
                    if !self.word_implies(arg, w, edge) {
                        return None;
                    }
                    continue;
                }
                let b = self.word_bit(arg, bit);
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
                    let x = self.word_bit(a, bit);
                    match (k, x) {
                        (IntOp::And, Some(false)) => return Some(false),
                        (IntOp::Or, Some(true)) => return Some(true),
                        _ => {}
                    }
                    let y = self.word_bit(b, bit);
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
                    Some(true) => self.word_bit(a, bit),
                    Some(false) => self.word_bit(b, bit),
                    None => {
                        let (x, y) = (self.word_bit(a, bit), self.word_bit(b, bit));
                        if x == y {
                            x
                        } else {
                            None
                        }
                    }
                },
                Op::Convert(Cvt::Bitcast, _, a) => self.word_bit(a, bit),
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
                if !self.implies(arg, v, edge) {
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
        loop {
            if let Some(condition) = self.edge_condition(block, slot) {
                return Some(condition);
            }
            match self.facts.incoming[&block].as_slice() {
                [(p, s)] => (block, slot) = (*p, *s),
                _ => return None,
            }
        }
    }

    fn edge_condition(&self, pred: BlockId, slot: usize) -> Option<(ValueId, bool)> {
        match self.f.blocks[&pred].term {
            Term::CondBr { cond, .. } => Some((cond, slot == 0)),
            _ => None,
        }
    }

    fn remembered(
        &self,
        kind: Implication,
        a: ValueId,
        v: ValueId,
        edge: Option<(ValueId, bool)>,
        f: impl FnOnce(&Self) -> bool,
    ) -> bool {
        let key = (kind, a, v, edge);
        if let Some(&holds) = self.implications.borrow().get(&key) {
            return holds;
        }
        let holds = f(self);
        self.implications.borrow_mut().insert(key, holds);
        holds
    }

    fn implies(&self, a: ValueId, v: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        let a = self.copies.get(&a).copied().unwrap_or(a);
        let v = self.copies.get(&v).copied().unwrap_or(v);
        if a == v {
            return true;
        }
        self.remembered(Implication::Bit, a, v, edge, |this| this.derives(a, v, edge))
    }

    fn derives(&self, a: ValueId, v: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        match self.facts.op(self.f, a) {
            Some(Op::Int(IntOp::And, x, y)) => {
                self.implies(x, v, edge) || self.implies(y, v, edge)
            }
            Some(Op::Select(c, x, y)) => match self.decided(c, edge) {
                Some(taken) => self.implies(if taken { x } else { y }, v, edge),
                None => self.implies(x, v, edge) && self.implies(y, v, edge),
            },
            Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => match self.facts.op(self.f, shifted) {
                Some(Op::Int(IntOp::LShr, w, s)) if self.facts.op(self.f, s) == Some(Op::Env(Env::LaneId)) => {
                    self.word_holds(w, v, edge)
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn word_holds(&self, w: ValueId, v: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        let w = self.copies.get(&w).copied().unwrap_or(w);
        let v = self.copies.get(&v).copied().unwrap_or(v);
        self.remembered(Implication::Holds, w, v, edge, |this| this.holds(w, v, edge))
    }

    fn holds(&self, w: ValueId, v: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        match self.facts.inst(self.f, w) {
            Some(Inst::Effect {
                op: EffectOp::Wave(WaveOp::Ballot),
                inputs,
                ..
            }) => self.implies(inputs[0], v, edge),
            Some(Inst::Core { op, .. }) => match *op {
                Op::Int(IntOp::And, x, y) => {
                    self.word_holds(x, v, edge) || self.word_holds(y, v, edge)
                }
                Op::Int(IntOp::Or, x, y) => {
                    self.word_holds(x, v, edge) && self.word_holds(y, v, edge)
                }
                Op::Select(c, x, y) => match self.decided(c, edge) {
                    Some(taken) => self.word_holds(if taken { x } else { y }, v, edge),
                    None => self.word_holds(x, v, edge) && self.word_holds(y, v, edge),
                },
                Op::Convert(Cvt::Bitcast, _, x) => self.word_holds(x, v, edge),
                Op::Const(_, 0) => true,
                _ => false,
            },
            _ => false,
        }
    }

    fn word_implies(&self, a: ValueId, w: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        let a = self.copies.get(&a).copied().unwrap_or(a);
        let w = self.copies.get(&w).copied().unwrap_or(w);
        if a == w {
            return true;
        }
        self.remembered(Implication::Word, a, w, edge, |this| this.narrows(a, w, edge))
    }

    fn narrows(&self, a: ValueId, w: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        match self.facts.op(self.f, a) {
            Some(Op::Int(IntOp::And, x, y)) => {
                self.word_implies(x, w, edge) || self.word_implies(y, w, edge)
            }
            Some(Op::Select(c, x, y)) => match self.decided(c, edge) {
                Some(taken) => self.word_implies(if taken { x } else { y }, w, edge),
                None => self.word_implies(x, w, edge) && self.word_implies(y, w, edge),
            },
            Some(Op::Convert(Cvt::Bitcast, _, x)) => self.word_implies(x, w, edge),
            _ => false,
        }
    }
}

fn added(value: &Value, symbol: Unknown, region: Option<Region>) -> Option<u32> {
    (value.region == region && value.form.terms == [(symbol, 1)]).then_some(value.form.constant)
}

fn recurrence(sofar: Option<Option<(u32, u32)>>, value: Option<&Value>, symbol: Unknown) -> Option<Option<(u32, u32)>> {
    let value = value.filter(|v| v.region.is_none())?;
    let affine = match value.form.terms.as_slice() {
        [] => (0, value.form.constant),
        [(u, c)] if *u == symbol => (*c, value.form.constant),
        _ => return None,
    };
    match sofar? {
        None => Some(Some(affine)),
        Some(old) if old == affine => Some(Some(affine)),
        Some(_) => None,
    }
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

pub(super) fn aligns(m: u32) -> bool {
    (!m).wrapping_add(1).is_power_of_two()
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

fn negated(pred: IntPred) -> IntPred {
    match pred {
        IntPred::Eq => IntPred::Ne,
        IntPred::Ne => IntPred::Eq,
        IntPred::Ult => IntPred::Uge,
        IntPred::Uge => IntPred::Ult,
        IntPred::Ule => IntPred::Ugt,
        IntPred::Ugt => IntPred::Ule,
        IntPred::Slt => IntPred::Sge,
        IntPred::Sge => IntPred::Slt,
        IntPred::Sle => IntPred::Sgt,
        IntPred::Sgt => IntPred::Sle,
    }
}

pub(super) enum Cond {
    Leaf(ValueId, bool),
    All(ValueId, bool, Vec<std::rc::Rc<Cond>>),
    Any(ValueId, bool, Vec<std::rc::Rc<Cond>>),
}

pub(super) fn condition(f: &Func, facts: &Facts, copies: &Copies, cond: ValueId, taken: bool) -> std::rc::Rc<Cond> {
    fn build(f: &Func, facts: &Facts, copies: &Copies, cond: ValueId, taken: bool, memo: &mut HashMap<(ValueId, bool), std::rc::Rc<Cond>>) -> std::rc::Rc<Cond> {
        let cond = copies.get(&cond).copied().unwrap_or(cond);
        if let Some(found) = memo.get(&(cond, taken)) {
            return found.clone();
        }
        let node = match facts.op(f, cond) {
            Some(Op::Int(k @ (IntOp::And | IntOp::Or), a, b)) if f.types[cond.0] == Ty::I1 => {
                let parts = vec![build(f, facts, copies, a, taken, memo), build(f, facts, copies, b, taken, memo)];
                if (k == IntOp::And) == taken {
                    Cond::All(cond, taken, parts)
                } else {
                    Cond::Any(cond, taken, parts)
                }
            }
            Some(Op::Int(IntOp::Xor, a, one)) if f.types[a.0] == Ty::I1 && facts.constant(f, one) == Some(1) => {
                Cond::All(cond, taken, vec![build(f, facts, copies, a, !taken, memo)])
            }
            _ => Cond::Leaf(cond, taken),
        };
        let node = std::rc::Rc::new(node);
        memo.insert((cond, taken), node.clone());
        node
    }
    build(f, facts, copies, cond, taken, &mut HashMap::default())
}

pub(super) fn literals(cond: &Cond, out: &mut Vec<(ValueId, bool)>) {
    let (Cond::Leaf(v, holds) | Cond::All(v, holds, _) | Cond::Any(v, holds, _)) = cond;
    if out.contains(&(*v, *holds)) {
        return;
    }
    out.push((*v, *holds));
    if let Cond::All(_, _, parts) = cond {
        for part in parts {
            literals(part, out);
        }
    }
}

fn shares(cond: &Cond) -> bool {
    match cond {
        Cond::Leaf(..) => false,
        Cond::All(_, _, parts) | Cond::Any(_, _, parts) => parts.iter().any(|p| std::rc::Rc::strong_count(p) > 1 || shares(p)),
    }
}

fn satisfying(pred: IntPred, k: u32) -> Vec<(u64, u64)> {
    let max = u32::MAX as i64;
    let unsigned = |low: i64, high: i64| if low > high { Vec::new() } else { vec![(low as u64, high as u64)] };
    let signed = |low: i64, high: i64| {
        if low > high {
            Vec::new()
        } else if low >= 0 || high < 0 {
            vec![(low as i32 as u32 as u64, high as i32 as u32 as u64)]
        } else {
            vec![(0, high as u64), (low as i32 as u32 as u64, u32::MAX as u64)]
        }
    };
    let (u, s) = (k as i64, k as i32 as i64);
    let (least, most) = (i32::MIN as i64, i32::MAX as i64);
    match pred {
        IntPred::Eq => vec![(k as u64, k as u64)],
        IntPred::Ne => [unsigned(0, u - 1), unsigned(u + 1, max)].concat(),
        IntPred::Ult => unsigned(0, u - 1),
        IntPred::Ule => unsigned(0, u),
        IntPred::Ugt => unsigned(u + 1, max),
        IntPred::Uge => unsigned(u, max),
        IntPred::Slt => signed(least, s - 1),
        IntPred::Sle => signed(least, s),
        IntPred::Sgt => signed(s + 1, most),
        IntPred::Sge => signed(s, most),
    }
}

fn shifted_pieces(set: &[(u64, u64)], k: u32) -> Vec<(u64, u64)> {
    let whole = 1u64 << 32;
    let mut out = Vec::new();
    for &(low, high) in set {
        let (low, high) = (low + k as u64, high + k as u64);
        if high < whole {
            out.push((low, high));
        } else if low >= whole {
            out.push((low - whole, high - whole));
        } else {
            out.push((low, whole - 1));
            out.push((0, high - whole));
        }
    }
    normalized(out)
}

fn intersected(a: &[(u64, u64)], b: &[(u64, u64)]) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    for &(la, ha) in a {
        for &(lb, hb) in b {
            let (low, high) = (la.max(lb), ha.min(hb));
            if low <= high {
                out.push((low, high));
            }
        }
    }
    normalized(out)
}

fn normalized(mut set: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    set.sort_unstable();
    let mut out: Vec<(u64, u64)> = Vec::new();
    for (low, high) in set {
        match out.last_mut() {
            Some(last) if low <= last.1 + 1 => last.1 = last.1.max(high),
            _ => out.push((low, high)),
        }
    }
    out
}

pub(super) type Classes = Vec<(Form, Vec<(u64, u64)>)>;

#[derive(Clone, Debug, Default)]
struct Limits {
    classes: Classes,
    orders: Vec<(Form, Form, u8)>,
}

impl Limits {
    fn is_empty(&self) -> bool {
        self.classes.is_empty() && self.orders.is_empty()
    }
}

fn both_limits(a: Limits, b: Limits) -> Limits {
    Limits {
        classes: conjoined(a.classes, b.classes),
        orders: orders_conjoined(a.orders, b.orders),
    }
}

fn either_limits(a: Limits, b: Limits) -> Limits {
    Limits {
        classes: disjoined(a.classes, b.classes),
        orders: orders_disjoined(a.orders, b.orders),
    }
}

fn order_limit(x: &Form, y: &Form, pred: IntPred) -> Vec<(Form, Form, u8)> {
    if x == y {
        return Vec::new();
    }
    let mask = outcomes(pred);
    if (x.terms.as_slice(), x.constant) <= (y.terms.as_slice(), y.constant) {
        vec![(x.clone(), y.clone(), mask)]
    } else {
        vec![(y.clone(), x.clone(), mirrored(mask))]
    }
}

fn orders_conjoined(mut a: Vec<(Form, Form, u8)>, b: Vec<(Form, Form, u8)>) -> Vec<(Form, Form, u8)> {
    for (x, y, mask) in b {
        match a.iter_mut().find(|(p, q, _)| *p == x && *q == y) {
            Some((_, _, old)) => *old &= mask,
            None => a.push((x, y, mask)),
        }
    }
    a
}

fn orders_disjoined(a: Vec<(Form, Form, u8)>, b: Vec<(Form, Form, u8)>) -> Vec<(Form, Form, u8)> {
    a.into_iter()
        .filter_map(|(x, y, mask)| {
            let (_, _, other) = b.iter().find(|(p, q, _)| *p == x && *q == y)?;
            Some((x, y, mask | other))
        })
        .collect()
}

fn class_limit(form: &Form, values: &[(u64, u64)]) -> (Form, Vec<(u64, u64)>) {
    let class = Form {
        constant: 0,
        terms: form.terms.clone(),
    };
    (class, normalized(shifted_pieces(values, form.constant.wrapping_neg())))
}

fn conjoined(mut a: Classes, b: Classes) -> Classes {
    for (class, set) in b {
        match a.iter_mut().find(|(c, _)| *c == class) {
            Some((_, old)) => *old = intersected(old, &set),
            None => a.push((class, set)),
        }
    }
    a
}

fn disjoined(a: Classes, b: Classes) -> Classes {
    a.into_iter()
        .filter_map(|(class, set)| {
            let (_, other) = b.iter().find(|(c, _)| *c == class)?;
            Some((class, normalized([set, other.clone()].concat())))
        })
        .collect()
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

pub(super) fn compare(pred: IntPred, x: u32, y: u32) -> bool {
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

#[cfg(test)]
mod tests {
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

    fn at(x: (u32, u32), t: u32) -> u32 {
        x.0.wrapping_add(x.1.wrapping_mul(t))
    }

    fn interesting(r: &mut Random) -> u32 {
        match r.below(5) {
            0 => r.below(16) as u32,
            1 => (r.below(16) as u32).wrapping_neg(),
            2 => 0x8000_0000u32.wrapping_add(r.below(16) as u32).wrapping_sub(8),
            3 => [1, 2, 3, 4, 0xffff_ffff, 0xffff_fffe, 0x8000_0000, 0x7fff_ffff][r.below(8) as usize],
            _ => r.next() as u32,
        }
    }

    #[test]
    fn first_failure_names_an_iteration_that_leaves_the_loop() {
        let mut r = Random::new(5);
        for _ in 0..200000 {
            let pred = PREDICATES[r.below(10) as usize];
            let taken = r.below(2) == 0;
            let x = (interesting(&mut r), if r.below(2) == 0 { 0 } else { interesting(&mut r) });
            let y = (interesting(&mut r), if r.below(3) == 0 { interesting(&mut r) } else { 0 });
            if let Some(last) = first_failure(pred, taken, x, y) {
                assert_ne!(
                    compare(pred, at(x, last), at(y, last)),
                    taken,
                    "{:?} taken={} {:?} {:?}: the loop still runs after iteration {}",
                    pred, taken, x, y, last
                );
            }
        }
    }

    #[test]
    fn first_failure_names_the_first_iteration_that_leaves_the_loop() {
        let mut r = Random::new(9);
        for _ in 0..200000 {
            let pred = PREDICATES[r.below(10) as usize];
            let taken = r.below(2) == 0;
            let x = (interesting(&mut r), if r.below(2) == 0 { 0 } else { interesting(&mut r) });
            let y = (interesting(&mut r), if r.below(3) == 0 { interesting(&mut r) } else { 0 });
            if let Some(last) = first_failure(pred, taken, x, y) {
                if let Some(t) = (0..last.min(1 << 12)).find(|&t| compare(pred, at(x, t), at(y, t)) != taken) {
                    panic!(
                        "{:?} taken={} {:?} {:?}: the loop leaves after iteration {}, before {}",
                        pred, taken, x, y, t, last
                    );
                }
            }
        }
    }

    fn range(r: &mut Random) -> (u64, u64) {
        let low = interesting(r) as u64;
        let width = r.below(4);
        (low, (low + width).min(u32::MAX as u64))
    }

    #[test]
    fn decide_answers_only_what_holds_across_both_ranges() {
        let mut r = Random::new(13);
        for _ in 0..100000 {
            let pred = PREDICATES[r.below(10) as usize];
            let (x, y) = (range(&mut r), range(&mut r));
            if let Some(answer) = decide(pred, x, y) {
                for a in x.0..=x.1 {
                    for b in y.0..=y.1 {
                        assert_eq!(
                            compare(pred, a as u32, b as u32),
                            answer,
                            "{:?} over {:?} and {:?} at {} and {}",
                            pred, x, y, a, b
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn offset_answers_only_what_holds_for_every_value_in_range() {
        let mut r = Random::new(17);
        for _ in 0..100000 {
            let pred = PREDICATES[r.below(10) as usize];
            let y = range(&mut r);
            let d = interesting(&mut r);
            if let Some(answer) = offset(pred, d, y) {
                for b in y.0..=y.1 {
                    let b = b as u32;
                    assert_eq!(
                        compare(pred, b.wrapping_add(d), b),
                        answer,
                        "{:?}: y in {:?}, x = y + {:#x}, at y = {}",
                        pred, y, d, b
                    );
                }
            }
        }
    }

    #[test]
    fn low_bits_matches_the_operation_for_every_value_of_the_unknowns() {
        let mut r = Random::new(19);
        for _ in 0..20000 {
            let form = Form {
                constant: interesting(&mut r),
                terms: vec![(0, (interesting(&mut r) | 1) << r.below(8)), (1, (interesting(&mut r) | 1) << r.below(8))],
            };
            let c = r.below(512) as u32;
            for op in [IntOp::And, IntOp::Or, IntOp::Xor] {
                let Some(result) = low_bits(&form, c, op) else { continue };
                for _ in 0..16 {
                    let values = [r.next() as u32, r.next() as u32];
                    let value = |f: &Form| {
                        f.terms
                            .iter()
                            .fold(f.constant, |a, &(u, k)| a.wrapping_add(k.wrapping_mul(values[u as usize])))
                    };
                    let v = value(&form);
                    let expected = match op {
                        IntOp::And => v & c,
                        IntOp::Or => v | c,
                        _ => v ^ c,
                    };
                    assert_eq!(value(&result), expected, "{:?} {:?} {:#x} at {:?}", form, op, c, values);
                }
            }
        }
    }

    #[test]
    fn forms_add_subtract_and_scale_as_words() {
        let mut r = Random::new(23);
        for _ in 0..20000 {
            let form = |r: &mut Random| {
                let mut terms: Vec<(Unknown, u32)> = Vec::new();
                for u in 0..4 {
                    let c = interesting(r);
                    if r.below(2) == 0 && c != 0 {
                        terms.push((u, c));
                    }
                }
                terms.sort();
                Form {
                    constant: interesting(r),
                    terms,
                }
            };
            let (a, b) = (form(&mut r), form(&mut r));
            let k = interesting(&mut r);
            let values: Vec<u32> = (0..4).map(|_| r.next() as u32).collect();
            let value = |f: &Form| {
                f.terms
                    .iter()
                    .fold(f.constant, |acc, &(u, c)| acc.wrapping_add(c.wrapping_mul(values[u as usize])))
            };
            let canonical = |f: &Form| {
                f.terms.windows(2).all(|w| w[0].0 < w[1].0) && f.terms.iter().all(|&(_, c)| c != 0)
            };
            for (result, expected) in [
                (a.add(&b), value(&a).wrapping_add(value(&b))),
                (a.sub(&b), value(&a).wrapping_sub(value(&b))),
                (a.scale(k), value(&a).wrapping_mul(k)),
            ] {
                assert_eq!(value(&result), expected);
                assert!(canonical(&result), "{:?}", result);
            }
            assert_eq!(a.sub(&a).as_constant(), Some(0));
        }
    }

    #[test]
    fn extend_widens_each_size_as_the_load_does() {
        for word in [0u32, 0x7f, 0x80, 0xff, 0x7fff, 0x8000, 0xffff, 0x1234_5678, 0xffff_ffff] {
            assert_eq!(extend(word, MemSize::U8), word & 0xff);
            assert_eq!(extend(word, MemSize::I8), word as u8 as i8 as i32 as u32);
            assert_eq!(extend(word, MemSize::U16), word & 0xffff);
            assert_eq!(extend(word, MemSize::I16), word as u16 as i16 as i32 as u32);
            assert_eq!(extend(word, MemSize::B32), word);
        }
    }
}

#[cfg(test)]
mod facts_tests {
    use super::super::testing::*;
    use super::*;
    use crate::rdna_spmd::analysis::loops::Loops;

    pub(super) fn addresses<T>(b: &Build, env: &Environment, f: impl FnOnce(&mut Addresses) -> T) -> T {
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let loops = Loops::new(&b.f, &facts).expect("a reducible test program");
        let headers: BTreeSet<BlockId> = (0..loops.count()).map(|l| facts.order[loops.header(l)]).collect();
        let mut a = Addresses::new(&b.f, &facts, &b.inputs, EXEC, b.entry, env, headers, &b.registry);
        a.enter(0);
        f(&mut a)
    }

    #[derive(Clone, Copy, Debug)]
    enum Case {
        Int(IntOp, bool),
        Cmp(IntPred),
        Select,
        Count(u8),
        Extend(Cvt, Ty),
        Wide(IntOp),
        Pack,
        Bits(IntOp),
        Truncate,
    }

    fn leaf(b: &mut Build, e: BlockId, lane: ValueId, m: u32, c: u32) -> ValueId {
        let km = b.constant(e, Ty::I32, m as u64);
        let kc = b.constant(e, Ty::I32, c as u64);
        let scaled = b.int(e, IntOp::Mul, lane, km);
        b.int(e, IntOp::Add, scaled, kc)
    }

    fn compare(pred: IntPred, x: u32, y: u32) -> bool {
        super::compare(pred, x, y)
    }

    enum Truth {
        Word(Vec<u32>),
        Bit(Vec<bool>),
    }

    fn cases(seed: u64, count: usize) -> Vec<(Build, ValueId, Truth, String)> {
        use IntOp::*;
        let mut r = Random::new(seed);
        let ops = [Add, Sub, Mul, And, Or, Xor, Shl, LShr, AShr];
        let preds = [
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
        let pick = |r: &mut Random| -> u32 {
            match r.below(4) {
                0 => r.below(8) as u32,
                1 => (r.below(8) as u32).wrapping_neg(),
                2 => 0x8000_0000u32.wrapping_add(r.below(4) as u32),
                _ => r.next() as u32,
            }
        };
        let mut out = Vec::new();
        for _ in 0..count {
            let (mut b, _) = Build::kernel();
            let e = BlockId(0);
            let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
            let (m1, c1, m2, c2) = (pick(&mut r), pick(&mut r), pick(&mut r), pick(&mut r));
            let x = leaf(&mut b, e, lane, m1, c1);
            let y = leaf(&mut b, e, lane, m2, c2);
            let tx: Vec<u32> = (0..32u32).map(|l| l.wrapping_mul(m1).wrapping_add(c1)).collect();
            let ty: Vec<u32> = (0..32u32).map(|l| l.wrapping_mul(m2).wrapping_add(c2)).collect();
            let case = match r.below(9) {
                0 => Case::Int(ops[r.below(9) as usize], false),
                1 => Case::Int(ops[r.below(9) as usize], true),
                2 => Case::Cmp(preds[r.below(10) as usize]),
                3 => Case::Select,
                4 => Case::Count(r.below(4) as u8),
                5 => Case::Extend(if r.below(2) == 0 { Cvt::ZExt } else { Cvt::SExt }, if r.below(2) == 0 { Ty::I32 } else { Ty::I64 }),
                6 => Case::Wide([Add, Sub, Mul, And, Or, Xor, Shl][r.below(7) as usize]),
                7 => Case::Pack,
                _ => {
                    if r.below(2) == 0 {
                        Case::Bits([And, Or, Xor][r.below(3) as usize])
                    } else {
                        Case::Truncate
                    }
                }
            };
            let (v, truth) = match case {
                Case::Int(op, constant_shift) => {
                    let amount = r.below(32) as u32;
                    let (y, ty) = if matches!(op, Shl | LShr | AShr) || constant_shift {
                        if matches!(op, Shl | LShr | AShr) {
                            (b.constant(e, Ty::I32, amount as u64), vec![amount; 32])
                        } else {
                            let k = pick(&mut r);
                            (b.constant(e, Ty::I32, k as u64), vec![k; 32])
                        }
                    } else {
                        (y, ty.clone())
                    };
                    let v = b.int(e, op, x, y);
                    let t = (0..32)
                        .map(|l| {
                            let (a, s) = (tx[l], ty[l]);
                            match op {
                                Add => a.wrapping_add(s),
                                Sub => a.wrapping_sub(s),
                                Mul => a.wrapping_mul(s),
                                And => a & s,
                                Or => a | s,
                                Xor => a ^ s,
                                Shl => a << s,
                                LShr => a >> s,
                                _ => ((a as i32) >> s) as u32,
                            }
                        })
                        .collect();
                    (v, Truth::Word(t))
                }
                Case::Cmp(pred) => {
                    let v = b.cmp(e, pred, x, y);
                    (v, Truth::Bit((0..32).map(|l| compare(pred, tx[l], ty[l])).collect()))
                }
                Case::Select => {
                    let pred = preds[r.below(10) as usize];
                    let c = b.cmp(e, pred, x, y);
                    let v = b.core(e, Ty::I32, Op::Select(c, x, y));
                    (v, Truth::Word((0..32).map(|l| if compare(pred, tx[l], ty[l]) { tx[l] } else { ty[l] }).collect()))
                }
                Case::Count(k) => {
                    let op = [Op::PopulationCount(x), Op::TrailingZeros(x), Op::LeadingZeros(x), Op::ReverseBits(x)][k as usize];
                    let v = b.core(e, Ty::I32, op);
                    let t = (0..32)
                        .map(|l| {
                            let a = tx[l];
                            match k {
                                0 => a.count_ones(),
                                1 => a.trailing_zeros(),
                                2 => a.leading_zeros(),
                                _ => a.reverse_bits(),
                            }
                        })
                        .collect();
                    (v, Truth::Word(t))
                }
                Case::Extend(cvt, to) => {
                    let pred = preds[r.below(10) as usize];
                    let c = b.cmp(e, pred, x, y);
                    let v = b.core(e, to, Op::Convert(cvt, to, c));
                    let t = (0..32)
                        .map(|l| match (compare(pred, tx[l], ty[l]), cvt) {
                            (false, _) => 0,
                            (true, Cvt::ZExt) => 1,
                            (true, _) => u32::MAX,
                        })
                        .collect();
                    (v, Truth::Word(t))
                }
                Case::Wide(op) => {
                    let wx = b.core(e, Ty::I64, Op::Convert(Cvt::SExt, Ty::I64, x));
                    let wy = b.core(e, Ty::I64, Op::Pack64(y, x));
                    let amount = r.below(64);
                    let wy = if op == Shl { b.constant(e, Ty::I64, amount) } else { wy };
                    let v = b.int(e, op, wx, wy);
                    let t = (0..32)
                        .map(|l| {
                            let a = tx[l] as i32 as i64 as u64;
                            let s = if op == Shl { amount } else { ty[l] as u64 | (tx[l] as u64) << 32 };
                            let r = match op {
                                Add => a.wrapping_add(s),
                                Sub => a.wrapping_sub(s),
                                Mul => a.wrapping_mul(s),
                                And => a & s,
                                Or => a | s,
                                Xor => a ^ s,
                                _ => a << s,
                            };
                            r as u32
                        })
                        .collect();
                    (v, Truth::Word(t))
                }
                Case::Pack => {
                    let p = b.core(e, Ty::I64, Op::Pack64(x, y));
                    let high = r.below(2) == 0;
                    let v = b.core(e, Ty::I32, if high { Op::UnpackHi(p) } else { Op::UnpackLo(p) });
                    (v, Truth::Word(if high { ty.clone() } else { tx.clone() }))
                }
                Case::Bits(op) => {
                    let p1 = preds[r.below(10) as usize];
                    let p2 = preds[r.below(10) as usize];
                    let c1 = b.cmp(e, p1, x, y);
                    let c2 = b.cmp(e, p2, y, x);
                    let v = b.int(e, op, c1, c2);
                    let t = (0..32)
                        .map(|l| {
                            let (a, s) = (compare(p1, tx[l], ty[l]), compare(p2, ty[l], tx[l]));
                            match op {
                                And => a && s,
                                Or => a || s,
                                _ => a != s,
                            }
                        })
                        .collect();
                    (v, Truth::Bit(t))
                }
                Case::Truncate => {
                    let k = r.below(32);
                    let amount = b.constant(e, Ty::I32, k);
                    let shifted = b.int(e, LShr, x, amount);
                    let v = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
                    (v, Truth::Bit((0..32).map(|l| tx[l] >> k & 1 != 0).collect()))
                }
            };
            out.push((b, v, truth, format!("{:?}", case)));
        }
        out
    }

    fn decided(seed: u64, count: usize) -> (Vec<String>, Vec<String>) {
        let env = environment(32, &[(0, 1, 0x1000)]);
        let mut wrong = Vec::new();
        let mut undecided = Vec::new();
        for (b, v, truth, name) in cases(seed, count) {
            addresses(&b, &env, |a| {
                for lane in 0..32 {
                    match &truth {
                        Truth::Word(t) => match a.value(v, lane, None).0.form.as_constant() {
                            Some(k) if k != t[lane] => wrong.push(format!("{} lane {}: {:#x}, not {:#x}", name, lane, k, t[lane])),
                            Some(_) => {}
                            None => undecided.push(format!("{} lane {}", name, lane)),
                        },
                        Truth::Bit(t) => match a.bit(v, lane, None).0 {
                            Some(k) if k != t[lane] => wrong.push(format!("{} lane {}: {}, not {}", name, lane, k, t[lane])),
                            Some(_) => {}
                            None => undecided.push(format!("{} lane {}", name, lane)),
                        },
                    }
                }
            });
        }
        (wrong, undecided)
    }

    struct Unknowns {
        b: Build,
        words: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> u32>)>,
        bits: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> bool>)>,
    }

    fn unknowns() -> Unknowns {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U16, table, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, table, lane, 4);
        let w = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let c = |b: &mut Build, k: u64| b.constant(e, Ty::I32, k);
        let mut words: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> u32>)> = Vec::new();
        let mut bits: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> bool>)> = Vec::new();
        let four = c(&mut b, 4);
        let lane4 = b.int(e, IntOp::Mul, lane, four);
        let v = b.int(e, IntOp::Add, u, lane4);
        words.push(("u + 4 lane", v, Box::new(|u, _, l| u.wrapping_add(4 * l))));
        let eight = c(&mut b, 8);
        let u4 = b.int(e, IntOp::Mul, u, four);
        let v = b.int(e, IntOp::Add, u4, eight);
        words.push(("4u + 8", v, Box::new(|u, _, _| u.wrapping_mul(4).wrapping_add(8))));
        let one = c(&mut b, 1);
        let v = b.int(e, IntOp::LShr, u, one);
        words.push(("u >> 1", v, Box::new(|u, _, _| u >> 1)));
        let two = c(&mut b, 2);
        let u2 = b.int(e, IntOp::Add, u, two);
        let v = b.int(e, IntOp::LShr, u2, one);
        words.push(("(u + 2) >> 1", v, Box::new(|u, _, _| (u + 2) >> 1)));
        let ff = c(&mut b, 0xff);
        let v = b.int(e, IntOp::And, u, ff);
        words.push(("u & 0xff", v, Box::new(|u, _, _| u & 0xff)));
        let fffc = c(&mut b, 0xfffc);
        let v = b.int(e, IntOp::And, u, fffc);
        words.push(("u & 0xfffc", v, Box::new(|u, _, _| u & 0xfffc)));
        let fourteen = c(&mut b, 14);
        let quarter = b.int(e, IntOp::LShr, u, fourteen);
        let seven = c(&mut b, 7);
        for base in [8u32, 6] {
            let k = c(&mut b, base as u64);
            let shifted = b.int(e, IntOp::Add, quarter, k);
            let v = b.int(e, IntOp::And, shifted, seven);
            let name: &'static str = if base == 8 { "((u >> 14) + 8) & 7" } else { "((u >> 14) + 6) & 7" };
            words.push((name, v, Box::new(move |u, _, _| ((u >> 14) + base) & 7)));
        }
        let high = c(&mut b, 0x1_0000);
        let v = b.int(e, IntOp::Or, u, high);
        words.push(("u | 0x10000", v, Box::new(|u, _, _| u | 0x1_0000)));
        let ones = c(&mut b, 0xffff_ffff);
        let v = b.int(e, IntOp::Xor, u, ones);
        words.push(("u ^ ~0", v, Box::new(|u, _, _| !u)));
        let u_shl = b.int(e, IntOp::Shl, u, two);
        let v = b.int(e, IntOp::Add, u_shl, lane);
        words.push(("(u << 2) + lane", v, Box::new(|u, _, l| (u << 2).wrapping_add(l))));
        let seventeen = c(&mut b, 17);
        let v = b.int(e, IntOp::LShr, u, seventeen);
        words.push(("u >> 17", v, Box::new(|u, _, _| u >> 17)));
        let v = b.int(e, IntOp::Add, w, lane);
        words.push(("w + lane", v, Box::new(|_, w, l| w.wrapping_add(l))));
        let three = c(&mut b, 3);
        let w3 = b.int(e, IntOp::And, w, three);
        let v = b.int(e, IntOp::Mul, w3, four);
        words.push(("(w & 3) * 4", v, Box::new(|_, w, _| (w & 3) * 4)));
        let hundred = c(&mut b, 100);
        let small = b.cmp(e, IntPred::Ult, u, hundred);
        let v = b.core(e, Ty::I32, Op::Select(small, u, hundred));
        words.push(("min(u, 100)", v, Box::new(|u, _, _| u.min(100))));
        let next = b.int(e, IntOp::Add, u, one);
        let past = c(&mut b, 101);
        let v = b.core(e, Ty::I32, Op::Select(small, next, past));
        words.push(("select(u < 100, u + 1, 101)", v, Box::new(|u, _, _| if u < 100 { u + 1 } else { 101 })));
        let wide_small = b.cmp(e, IntPred::Ult, w, hundred);
        let v = b.core(e, Ty::I32, Op::Select(wide_small, u, w));
        words.push(("select(w < 100, u, w)", v, Box::new(|u, w, _| if w < 100 { u } else { w })));
        let v = b.core(e, Ty::I32, Op::Select(wide_small, next, w));
        words.push(("select(w < 100, u + 1, w)", v, Box::new(|u, w, _| if w < 100 { u + 1 } else { w })));
        let v = b.core(e, Ty::I32, Op::Select(wide_small, u, hundred));
        words.push(("select(w < 100, u, 100)", v, Box::new(|u, w, _| if w < 100 { u } else { 100 })));
        let thousand = c(&mut b, 1000);
        let below = b.cmp(e, IntPred::Ult, u, thousand);
        let v = b.core(e, Ty::I32, Op::Select(below, u, hundred));
        words.push(("select(u < 1000, u, 100)", v, Box::new(|u, _, _| if u < 1000 { u } else { 100 })));
        let low_byte = b.int(e, IntOp::And, w, ff);
        let more = b.int(e, IntOp::Add, u, low_byte);
        let v = b.core(e, Ty::I32, Op::Select(small, more, u));
        words.push(("select(u < 100, u + (w & 0xff), u)", v, Box::new(|u, w, _| if u < 100 { u + (w & 0xff) } else { u })));
        let v = b.core(e, Ty::I32, Op::Select(small, u, more));
        words.push(("select(u < 100, u, u + (w & 0xff))", v, Box::new(|u, w, _| if u < 100 { u } else { u + (w & 0xff) })));
        let twice = b.int(e, IntOp::Add, u, u);
        let v = b.core(e, Ty::I32, Op::Select(small, u, twice));
        words.push(("select(u < 100, u, 2u)", v, Box::new(|u, _, _| if u < 100 { u } else { 2 * u })));
        let v = b.int(e, IntOp::Mul, u, u);
        words.push(("u * u", v, Box::new(|u, _, _| u.wrapping_mul(u))));
        let sixteen = c(&mut b, 16);
        let up = b.int(e, IntOp::Shl, u, sixteen);
        let v = b.int(e, IntOp::LShr, up, sixteen);
        words.push(("(u << 16) >> 16", v, Box::new(|u, _, _| (u << 16) >> 16)));
        let v = b.int(e, IntOp::Sub, u, one);
        words.push(("u - 1", v, Box::new(|u, _, _| u.wrapping_sub(1))));
        let v = b.int(e, IntOp::Sub, u2, u);
        words.push(("(u + 2) - u", v, Box::new(|_, _, _| 2)));
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, u));
        let hi = b.core(e, Ty::I32, Op::UnpackHi(wide));
        words.push(("hi(zext u)", hi, Box::new(|_, _, _| 0)));
        let both = b.int(e, IntOp::And, u, w);
        words.push(("u & w", both, Box::new(|u, w, _| u & w)));
        let either = b.int(e, IntOp::Or, w, u);
        words.push(("w | u", either, Box::new(|u, w, _| w | u)));
        let v = b.int(e, IntOp::Xor, u, w);
        words.push(("u ^ w", v, Box::new(|u, w, _| u ^ w)));
        let v = b.int(e, IntOp::Add, both, either);
        words.push(("(u & w) + (w | u)", v, Box::new(|u, w, _| (u & w).wrapping_add(w | u))));
        let v = b.int(e, IntOp::Add, u, w);
        words.push(("u + w", v, Box::new(|u, w, _| u.wrapping_add(w))));
        let v = b.int(e, IntOp::Mul, u, w);
        words.push(("u * w", v, Box::new(|u, w, _| u.wrapping_mul(w))));
        let v = b.int(e, IntOp::Mul, w, u);
        words.push(("w * u", v, Box::new(|u, w, _| w.wrapping_mul(u))));
        let v = b.int(e, IntOp::Mul, u2, w);
        words.push(("(u + 2) * w", v, Box::new(|u, w, _| (u + 2).wrapping_mul(w))));
        let thirty_one = c(&mut b, 31);
        let s = b.int(e, IntOp::And, w, thirty_one);
        let v = b.int(e, IntOp::Shl, u, s);
        words.push(("u << (w & 31)", v, Box::new(|u, w, _| u << (w & 31))));
        let power = b.int(e, IntOp::Shl, one, s);
        let v = b.int(e, IntOp::Mul, power, u);
        words.push(("(1 << (w & 31)) * u", v, Box::new(|u, w, _| (1u32 << (w & 31)).wrapping_mul(u))));
        let v = b.int(e, IntOp::Shl, w, u);
        words.push(("w << u", v, Box::new(|u, w, _| w << (u & 31))));
        let v = b.int(e, IntOp::LShr, u, s);
        words.push(("u >> (w & 31)", v, Box::new(|u, w, _| u >> (w & 31))));
        let v = b.int(e, IntOp::AShr, u, s);
        words.push(("u >>> (w & 31)", v, Box::new(|u, w, _| ((u as i32) >> (w & 31)) as u32)));
        let low_half = c(&mut b, 0xffff);
        let half = b.int(e, IntOp::And, u, low_half);
        let v = b.int(e, IntOp::LShr, half, s);
        words.push(("(u & 0xffff) >> (w & 31)", v, Box::new(|u, w, _| (u & 0xffff) >> (w & 31))));
        let v = b.int(e, IntOp::AShr, half, s);
        words.push(("(u & 0xffff) >>> (w & 31)", v, Box::new(|u, w, _| (u & 0xffff) >> (w & 31))));
        let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, u));
        let wide_s = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, s));
        let shifted = b.int(e, IntOp::Shl, wide, wide_s);
        let v = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, shifted));
        words.push(("low(zext(u) << (w & 31))", v, Box::new(|u, w, _| ((u as u64) << (w & 31)) as u32)));
        let sixty_three = c(&mut b, 63);
        let far = b.int(e, IntOp::And, w, sixty_three);
        let wide_far = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, far));
        let shifted = b.int(e, IntOp::Shl, wide, wide_far);
        let v = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, shifted));
        words.push(("low(zext(u) << (w & 63))", v, Box::new(|u, w, _| ((u as u64) << (w & 63)) as u32)));
        let wide_w = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, w));
        let sum = b.int(e, IntOp::Add, wide, wide_w);
        let again = b.int(e, IntOp::Add, sum, wide);
        let v = b.core(e, Ty::I32, Op::UnpackHi(again));
        words.push(("hi((zext(u) + zext(w)) + zext(u))", v, Box::new(|u, w, _| ((u as u64 + w as u64 + u as u64) >> 32) as u32)));
        let back = b.int(e, IntOp::Sub, again, wide_w);
        let v = b.core(e, Ty::I32, Op::UnpackHi(back));
        words.push(("hi((zext(u) + zext(w)) + zext(u) - zext(w))", v, Box::new(|u, _, _| ((2 * u as u64) >> 32) as u32)));
        for (name, op) in [("popcount(u)", 0u8), ("leading zeros of u", 1), ("trailing zeros of u", 2), ("popcount(w)", 3), ("trailing zeros of w", 4)] {
            let v = match op {
                0 => b.core(e, Ty::I32, Op::PopulationCount(u)),
                1 => b.core(e, Ty::I32, Op::LeadingZeros(u)),
                2 => b.core(e, Ty::I32, Op::TrailingZeros(u)),
                3 => b.core(e, Ty::I32, Op::PopulationCount(w)),
                _ => b.core(e, Ty::I32, Op::TrailingZeros(w)),
            };
            words.push((
                name,
                v,
                Box::new(move |u, w, _| match op {
                    0 => u.count_ones(),
                    1 => u.leading_zeros(),
                    2 => u.trailing_zeros(),
                    3 => w.count_ones(),
                    _ => w.trailing_zeros(),
                }),
            ));
        }
        let big = c(&mut b, 70000);
        let v = b.cmp(e, IntPred::Ult, u, big);
        bits.push(("u < 70000", v, Box::new(|u, _, _| u < 70000)));
        let v = b.cmp(e, IntPred::Ult, u, hundred);
        bits.push(("u < 100", v, Box::new(|u, _, _| u < 100)));
        let top = b.int(e, IntOp::LShr, u, sixteen);
        let zero = c(&mut b, 0);
        let v = b.cmp(e, IntPred::Eq, top, zero);
        bits.push(("u >> 16 == 0", v, Box::new(|u, _, _| u >> 16 == 0)));
        let v = b.cmp(e, IntPred::Ne, u2, u);
        bits.push(("u + 2 != u", v, Box::new(|_, _, _| true)));
        let v = b.cmp(e, IntPred::Ult, w, w);
        bits.push(("w < w", v, Box::new(|_, _, _| false)));
        let v = b.cmp(e, IntPred::Slt, u, zero);
        bits.push(("u < 0 signed", v, Box::new(|u, _, _| (u as i32) < 0)));
        let v = b.cmp(e, IntPred::Ugt, u2, u);
        bits.push(("u + 2 > u", v, Box::new(|u, _, _| u.wrapping_add(2) > u)));
        Unknowns { b, words, bits }
    }

    fn samples() -> Vec<(u32, Vec<u32>)> {
        let mut r = Random::new(43);
        let mut values = vec![0u32, 1, 2, 3, 99, 100, 101, 255, 256, 32767, 32768, 65534, 65535];
        for _ in 0..8 {
            values.push(r.below(65536) as u32);
        }
        values.into_iter().map(|u| (u, (0..32).map(|_| r.next() as u32).collect())).collect()
    }

    fn representable(unknowns: &[UnknownInfo], form: &Form, truth: u32) -> bool {
        super::super::hazard::may_overlap_for_tests(unknowns, form, &Form::constant(truth))
    }

    fn exactly_representable(unknowns: &[UnknownInfo], form: &Form, truth: u32) -> Option<bool> {
        let choices: Vec<Vec<u32>> = form
            .terms
            .iter()
            .map(|&(u, _)| match (&unknowns[u as usize].values, unknowns[u as usize].range) {
                (Some(set), _) => Some(set.to_vec()),
                (None, Some((lo, hi))) if hi - lo < 4096 => Some((lo..=hi).collect()),
                _ => None,
            })
            .collect::<Option<_>>()?;
        let mut index: Vec<usize> = vec![0; choices.len()];
        loop {
            let value = form
                .terms
                .iter()
                .zip(&index)
                .zip(&choices)
                .fold(form.constant, |acc, ((&(_, c), &i), set)| acc.wrapping_add(c.wrapping_mul(set[i])));
            if value == truth {
                return Some(true);
            }
            let mut k = 0;
            loop {
                if k == index.len() {
                    return Some(false);
                }
                if index[k] + 1 < choices[k].len() {
                    index[k] += 1;
                    break;
                }
                index[k] = 0;
                k += 1;
            }
        }
    }

    #[test]
    fn shifts_of_wrapping_words_decide_only_what_holds() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let z = b.load(e, Space::Global, MemSize::B32, table, yes);
        let two = b.constant(e, Ty::I32, 2);
        let scaled = b.int(e, IntOp::Shl, z, two);
        let back = b.int(e, IntOp::LShr, scaled, two);
        let same = b.cmp(e, IntPred::Eq, back, z);
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let decided = addresses(&b, &env, |a| a.bit(same, 0, None).0);
        assert_eq!(decided, None, "(z << 2) >> 2 == z fails for z >= 2^30, so it cannot be decided true");
    }

    #[test]
    fn arithmetic_shifts_and_field_masks_keep_what_the_operations_determine() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U16, table, yes);
        let one = b.constant(e, Ty::I32, 1);
        let halved = b.int(e, IntOp::AShr, u, one);
        let field = b.constant(e, Ty::I32, 0x0ff0);
        let masked = b.int(e, IntOp::And, u, field);
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut loose = Vec::new();
        addresses(&b, &env, |a| {
            let f = a.value(halved, 0, None).0.form;
            if a.bounds(&f) != Some((0, 32767)) {
                loose.push(format!("u ashr 1: {:?} over {:?}, not within [0, 32767]", f, a.bounds(&f)));
            }
            let f = a.value(masked, 0, None).0.form;
            for (value, holds) in [(0u32, true), (0x10, true), (0x0ff0, true), (0x0ff1, false), (0x18, false)] {
                if exactly_representable(&a.unknowns, &f, value) != Some(holds) {
                    loose.push(format!("u & 0xff0: {:?} at {:#x}: not {}", f, value, holds));
                }
            }
        });
        assert!(loose.is_empty(), "{:?}", loose);
    }

    fn joined_values() -> (Build, Vec<(ValueId, Vec<Box<dyn Fn(u32) -> u32>>)>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let x = b.load(e, Space::Global, MemSize::B32, table, yes);
        let zero = b.constant(e, Ty::I32, 0);
        let first = b.cmp(e, IntPred::Ne, x, zero);
        let one = b.constant(e, Ty::I32, 1);
        let second = b.cmp(e, IntPred::Ugt, x, one);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let thirty_one = b.constant(e, Ty::I32, 31);
        let mirrored = b.int(e, IntOp::Sub, thirty_one, lane);
        let eight = b.constant(e, Ty::I32, 8);
        let past = b.int(e, IntOp::Add, lane, eight);
        let (five, nine, thirteen) = (b.constant(e, Ty::I32, 5), b.constant(e, Ty::I32, 9), b.constant(e, Ty::I32, 13));
        let (below, hundred) = (b.constant(e, Ty::I32, 0xffff_fff0), b.constant(e, Ty::I32, 100));
        let (middle, _) = b.block(&[Ty::I1]);
        let (join, j) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32, Ty::I32]);
        b.cond_br(e, first, (middle, vec![k.exec]), (join, vec![k.exec, lane, five, lane, five]));
        let exec = b.f.blocks[&middle].params[0].0;
        b.cond_br(
            middle,
            second,
            (join, vec![exec, mirrored, nine, past, below]),
            (join, vec![exec, mirrored, thirteen, lane, hundred]),
        );
        let values: Vec<(ValueId, Vec<Box<dyn Fn(u32) -> u32>>)> = vec![
            (j[1], vec![Box::new(|l| l), Box::new(|l| 31 - l)]),
            (j[2], vec![Box::new(|_| 5), Box::new(|_| 9), Box::new(|_| 13)]),
            (j[3], vec![Box::new(|l| l), Box::new(|l| l + 8)]),
            (j[4], vec![Box::new(|_| 5), Box::new(|_| 0xffff_fff0), Box::new(|_| 100)]),
        ];
        (b, values)
    }

    #[test]
    fn joins_hold_every_value_an_incoming_edge_brings() {
        let (b, values) = joined_values();
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut wrong = Vec::new();
        addresses(&b, &env, |a| {
            for (v, truths) in &values {
                for l in 0..32 {
                    let f = a.value(*v, l, None).0.form;
                    for truth in truths {
                        if !representable(&a.unknowns, &f, truth(l as u32)) {
                            wrong.push(format!("{:?} lane {}: {:?} cannot be {}", v, l, f, truth(l as u32)));
                        }
                    }
                }
            }
        });
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn joins_hold_only_the_values_on_the_steps_between_them() {
        let (b, values) = joined_values();
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut loose = Vec::new();
        addresses(&b, &env, |a| {
            let f = a.value(values[1].0, 3, None).0.form;
            for (value, holds) in [(5u32, true), (7, false), (9, true), (11, false), (13, true), (17, false)] {
                if exactly_representable(&a.unknowns, &f, value) != Some(holds) {
                    loose.push(format!("{:?} at {}: not {}", f, value, holds));
                }
            }
            let f = a.value(values[3].0, 3, None).0.form;
            for (value, holds) in [(5u32, true), (0xffff_fff0, true), (100, true), (0, false), (52, false), (0xffff_fff1, false), (6, false)] {
                if exactly_representable(&a.unknowns, &f, value) != Some(holds) {
                    loose.push(format!("{:?} at {:#x}: not {}", f, value, holds));
                }
            }
        });
        assert!(loose.is_empty(), "{:?}", loose);
    }

    fn low_bit_masks() -> (Build, Vec<(&'static str, ValueId, Box<dyn Fn(u32) -> u32>)>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let z = b.load(e, Space::Global, MemSize::B32, table, yes);
        let c = |b: &mut Build, k: u64| b.constant(e, Ty::I32, k);
        let mut words: Vec<(&'static str, ValueId, Box<dyn Fn(u32) -> u32>)> = Vec::new();
        let one = c(&mut b, 1);
        let v = b.int(e, IntOp::Or, z, one);
        words.push(("z | 1", v, Box::new(|z| z | 1)));
        let seven = c(&mut b, 7);
        let v = b.int(e, IntOp::Or, seven, z);
        words.push(("7 | z", v, Box::new(|z| z | 7)));
        let clear = c(&mut b, !7u32 as u64);
        let v = b.int(e, IntOp::And, z, clear);
        words.push(("z & ~7", v, Box::new(|z| z & !7)));
        let two = c(&mut b, 2);
        let v = b.int(e, IntOp::Or, z, two);
        words.push(("z | 2", v, Box::new(|z| z | 2)));
        let three = c(&mut b, 3);
        let bumped = b.int(e, IntOp::Add, z, three);
        let v = b.int(e, IntOp::Or, bumped, one);
        words.push(("(z + 3) | 1", v, Box::new(|z| z.wrapping_add(3) | 1)));
        let five = c(&mut b, 5);
        let v = b.int(e, IntOp::Or, z, five);
        words.push(("z | 5", v, Box::new(|z| z | 5)));
        let v = b.int(e, IntOp::Xor, z, three);
        words.push(("z ^ 3", v, Box::new(|z| z ^ 3)));
        let top = c(&mut b, 0x8000_0001);
        let v = b.int(e, IntOp::Xor, z, top);
        words.push(("z ^ 0x80000001", v, Box::new(|z| z ^ 0x8000_0001)));
        (b, words)
    }

    #[test]
    fn low_bit_masks_hold_every_value_of_a_uniform_word() {
        let (b, words) = low_bit_masks();
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut wrong = Vec::new();
        addresses(&b, &env, |a| {
            let forms: Vec<Form> = words.iter().map(|&(_, v, _)| a.value(v, 0, None).0.form).collect();
            let mut r = Random::new(71);
            let mut values = vec![0u32, 1, 2, 6, 7, 8, 0x7fff_ffff, 0x8000_0000, u32::MAX - 2, u32::MAX];
            values.extend((0..40).map(|_| r.next() as u32));
            for z in values {
                for (i, (name, _, truth)) in words.iter().enumerate() {
                    if !representable(&a.unknowns, &forms[i], truth(z)) {
                        wrong.push(format!("{} at z = {:#x}: {:?} cannot be {:#x}", name, z, forms[i], truth(z)));
                    }
                }
            }
        });
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn low_bit_masks_keep_the_high_part_of_a_uniform_word() {
        let (b, words) = low_bit_masks();
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut loose = Vec::new();
        addresses(&b, &env, |a| {
            let &(_, odd, _) = words.iter().find(|w| w.0 == "(z + 3) | 1").unwrap();
            let f = a.value(odd, 0, None).0.form;
            if f.terms.is_empty() || f.terms.iter().any(|&(_, c)| c % 2 != 0) || f.constant % 2 != 1 {
                loose.push(format!("(z + 3) | 1: {:?} can be even", f));
            }
            for (name, scale, constant) in [("z | 1", 2u32, 1u32), ("7 | z", 8, 7), ("z & ~7", 8, 0)] {
                let &(_, v, _) = words.iter().find(|w| w.0 == name).unwrap();
                let f = a.value(v, 0, None).0.form;
                let shaped = f.constant == constant && f.terms.len() == 1 && f.terms[0].1 == scale;
                if !shaped {
                    loose.push(format!("{}: {:?}, not {} times one unknown plus {}", name, f, scale, constant));
                }
            }
            for (name, scale, low) in [("z | 2", 4u32, (2u32, 3u32)), ("z | 5", 8, (5, 7)), ("z ^ 3", 4, (0, 3))] {
                let &(_, v, _) = words.iter().find(|w| w.0 == name).unwrap();
                let f = a.value(v, 0, None).0.form;
                let ranges: Vec<Option<(u32, u32)>> = f.terms.iter().map(|&(u, _)| a.unknowns[u as usize].range).collect();
                let shaped = f.constant == 0
                    && f.terms.len() == 2
                    && f.terms.iter().any(|&(_, c)| c == scale)
                    && f.terms.iter().zip(&ranges).any(|(&(_, c), &r)| c == 1 && r == Some(low));
                if !shaped {
                    loose.push(format!("{}: {:?} over {:?}, not {} times the high part plus a low part in {:?}", name, f, ranges, scale, low));
                }
            }
        });
        assert!(loose.is_empty(), "{:?}", loose);
    }

    #[test]
    fn value_forms_hold_every_value_the_loaded_words_can_give() {
        let Unknowns { b, words, bits } = unknowns();
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut wrong = Vec::new();
        addresses(&b, &env, |a| {
            let forms: Vec<Vec<Form>> = words.iter().map(|&(_, v, _)| (0..32).map(|l| a.value(v, l, None).0.form).collect()).collect();
            let decided: Vec<Vec<Option<bool>>> = bits.iter().map(|&(_, v, _)| (0..32).map(|l| a.bit(v, l, None).0).collect()).collect();
            for (u, w) in samples() {
                for (i, (name, _, truth)) in words.iter().enumerate() {
                    for l in 0..32 {
                        let t = truth(u, w[l], l as u32);
                        if !representable(&a.unknowns, &forms[i][l], t) {
                            wrong.push(format!("{} lane {} at u = {}: {:?} cannot be {:#x}", name, l, u, forms[i][l], t));
                        }
                    }
                }
                for (i, (name1, _, t1)) in words.iter().enumerate() {
                    for (j, (name2, _, t2)) in words.iter().enumerate() {
                        for (l1, l2) in [(0usize, 0usize), (0, 5), (3, 17)] {
                            let (f1, f2) = (&forms[i][l1], &forms[j][l2]);
                            if f1.terms.is_empty() || f1.terms != f2.terms {
                                continue;
                            }
                            let d = f1.constant.wrapping_sub(f2.constant);
                            let t = t1(u, w[l1], l1 as u32).wrapping_sub(t2(u, w[l2], l2 as u32));
                            if d != t {
                                wrong.push(format!("{} lane {} minus {} lane {} at u = {}: {} not {}", name1, l1, name2, l2, u, d as i32, t as i32));
                            }
                        }
                    }
                }
                for (i, (name, _, truth)) in bits.iter().enumerate() {
                    for l in 0..32 {
                        if let Some(k) = decided[i][l] {
                            if k != truth(u, w[l], l as u32) {
                                wrong.push(format!("{} lane {} at u = {}: decided {}", name, l, u, k));
                            }
                        }
                    }
                }
            }
        });
        wrong.sort();
        wrong.dedup();
        assert!(wrong.is_empty(), "{} wrong, first {:?}", wrong.len(), &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn value_forms_keep_what_the_operations_determine() {
        let Unknowns { b, words, bits } = unknowns();
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut loose = Vec::new();
        addresses(&b, &env, |a| {
            let form = |a: &mut Addresses, name: &str| {
                let &(_, v, _) = words.iter().find(|w| w.0 == name).unwrap();
                a.value(v, 3, None).0.form
            };
            let base = form(a, "u + 4 lane");
            let bounds = |a: &Addresses, f: &Form| a.bounds(f);
            for (name, expected) in [
                ("4u + 8", Some((4u32, 8u32))),
                ("u | 0x10000", Some((1, 0x1_0000))),
                ("u ^ ~0", Some((u32::MAX, u32::MAX))),
                ("(u << 16) >> 16", Some((1, 0))),
                ("u - 1", Some((1, u32::MAX))),
            ] {
                let f = form(a, name);
                let (scale, constant) = expected.unwrap();
                let want = Form {
                    constant,
                    terms: base.terms.iter().map(|&(t, c)| (t, c.wrapping_mul(scale))).collect(),
                };
                if f != want {
                    loose.push(format!("{}: {:?}, not {:?}", name, f, want));
                }
            }
            for (name, range) in [("u >> 1", (0u64, 32767u64)), ("(u + 2) >> 1", (1, 32768)), ("u & 0xff", (0, 255)), ("((u >> 14) + 8) & 7", (0, 3)), ("u >> 17", (0, 0)), ("(u + 2) - u", (2, 2)), ("hi(zext u)", (0, 0))] {
                let f = form(a, name);
                match bounds(a, &f) {
                    Some(found) if found == range => {}
                    other => loose.push(format!("{}: bounds {:?}, not {:?}", name, other, range)),
                }
            }
            for (name, want) in [("u < 70000", true), ("u >> 16 == 0", true), ("u + 2 != u", true), ("w < w", false), ("u < 0 signed", false)] {
                let &(_, v, _) = bits.iter().find(|x| x.0 == name).unwrap();
                if a.bit(v, 3, None).0 != Some(want) {
                    loose.push(format!("{}: undecided", name));
                }
            }
        });
        assert!(loose.is_empty(), "{:?}", loose);
    }

    struct Placed {
        b: Build,
        cases: Vec<(&'static str, ValueId, Box<dyn Fn(usize) -> Option<Region>>)>,
    }

    fn placed() -> Placed {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let yes = b.constant(e, Ty::I1, 1);
        let v = b.load(e, Space::Global, MemSize::B32, second, yes);
        let zero = b.constant(e, Ty::I32, 0);
        let unknown = b.cmp(e, IntPred::Ne, v, zero);
        let five = b.constant(e, Ty::I32, 5);
        let low = b.cmp(e, IntPred::Ult, lane, five);
        let alloc = |id| Some(Region::Allocation(id));
        let mut cases: Vec<(&'static str, ValueId, Box<dyn Fn(usize) -> Option<Region>>)> = Vec::new();
        cases.push(("buf", first, Box::new(move |_| alloc(1))));
        let own = byte_offset(&mut b, e, first, lane, 4);
        cases.push(("buf + 4 lane", own, Box::new(move |_| alloc(1))));
        let lanes = b.core(e, Ty::I64, Op::Select(low, first, second));
        cases.push(("select(lane < 5, first, second)", lanes, Box::new(move |l| alloc(if l < 5 { 1 } else { 2 }))));
        let lo = b.core(e, Ty::I32, Op::UnpackLo(first));
        let hi = b.core(e, Ty::I32, Op::UnpackHi(first));
        let eight = b.constant(e, Ty::I32, 8);
        let lo8 = b.int(e, IntOp::Add, lo, eight);
        let rebuilt = b.core(e, Ty::I64, Op::Pack64(lo8, hi));
        cases.push(("pack(lo + 8, hi)", rebuilt, Box::new(move |_| alloc(1))));
        let base = b.core(e, Ty::I64, Op::Env(Env::ScratchBase));
        let sixteen = b.constant(e, Ty::I64, 16);
        let private = b.int(e, IntOp::Add, base, sixteen);
        cases.push(("scratch base + 16", private, Box::new(|_| Some(Region::Private))));
        let kernarg = b.core(e, Ty::I64, Op::Pack64(k.kernarg.0, k.kernarg.1));
        cases.push(("kernarg pointer", kernarg, Box::new(|_| Some(Region::Kernarg))));
        let zero64 = b.constant(e, Ty::I64, 0);
        let or = b.int(e, IntOp::Or, first, zero64);
        cases.push(("buf | 0", or, Box::new(move |_| alloc(1))));
        let three = b.constant(e, Ty::I32, 3);
        let read = b.wave(e, WaveOp::ReadLane, vec![lo, three, zero]);
        let back = b.core(e, Ty::I64, Op::Pack64(read, hi));
        cases.push(("pack(readlane(lo, 3), hi)", back, Box::new(move |_| alloc(1))));
        let slot = b.constant(e, Ty::I32, 16);
        b.store(e, Space::Scratch, MemSize::B64, slot, first, k.exec);
        let spilled = b.load(e, Space::Scratch, MemSize::B64, slot, k.exec);
        cases.push(("reloaded spill of buf", spilled, Box::new(move |_| alloc(1))));
        let either = b.core(e, Ty::I64, Op::Select(unknown, first, second));
        cases.push(("select(loaded bit, first, second)", either, Box::new(move |_| None)));
        let integer = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
        cases.push(("zext lane", integer, Box::new(|_| None)));
        Placed { b, cases }
    }

    fn settle(a: &mut Addresses, cases: &[(&'static str, ValueId, Box<dyn Fn(usize) -> Option<Region>>)]) {
        loop {
            for (_, v, _) in cases {
                for l in 0..32 {
                    for refine in [false, true] {
                        a.regions(*v, l, None, refine);
                    }
                }
            }
            if !a.settle_loops() {
                return;
            }
        }
    }

    #[test]
    fn regions_hold_the_region_every_address_points_into() {
        let Placed { b, cases } = placed();
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut wrong = Vec::new();
        addresses(&b, &env, |a| {
            settle(a, &cases);
            for (name, v, truth) in &cases {
                for l in [0usize, 3, 7, 31] {
                    let truths: Vec<Option<Region>> = match truth(l) {
                        None if *name == "select(loaded bit, first, second)" => vec![Some(Region::Allocation(1)), Some(Region::Allocation(2))],
                        t => vec![t],
                    };
                    for refine in [false, true] {
                        let set = a.regions(*v, l, None, refine);
                        for t in &truths {
                            let meets = |x: Option<Region>, y: Option<Region>| x == y;
                            if !set.reaches(*t, meets) && !(t.is_none() && set.lost()) {
                                wrong.push(format!("{} lane {} refine {}: {:?} misses {:?}", name, l, refine, set, t));
                            }
                        }
                    }
                    let value = a.value(*v, l, None).0;
                    if let Some(r) = value.region {
                        if !truths.contains(&Some(r)) {
                            wrong.push(format!("{} lane {}: value says {:?}, truth {:?}", name, l, r, truths));
                        }
                    }
                }
            }
        });
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn regions_name_only_the_region_an_address_points_into_when_it_is_known() {
        let Placed { b, cases } = placed();
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let mut loose = Vec::new();
        addresses(&b, &env, |a| {
            settle(a, &cases);
            for (name, v, truth) in &cases {
                if matches!(*name, "select(loaded bit, first, second)" | "zext lane") {
                    continue;
                }
                for l in [0usize, 7] {
                    let set = a.regions(*v, l, None, true);
                    if set != Regions::one(truth(l)) {
                        loose.push(format!("{} lane {}: {:?}", name, l, set));
                    }
                }
            }
        });
        assert!(loose.is_empty(), "{:?}", loose);
    }

    struct Branches {
        b: Build,
        blocks: Vec<(&'static str, BlockId, bool)>,
    }

    fn branches(lanes: u32) -> Branches {
        let (mut b, k, extra) = Build::kernel_with(&[(ParameterSource::Sgpr(crate::rdna_spmd::engine::WORKGROUP_ID_X), Ty::I32)]);
        b.entry.workgroup_id_x = true;
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U16, table, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let mut blocks = Vec::new();
        let mut at = e;
        let mut exec = k.exec;
        let mut conditions: Vec<(&'static str, &'static str, Box<dyn Fn(&mut Build, BlockId) -> ValueId>, bool, bool)> = Vec::new();
        conditions.push((
            "u < 70000 taken",
            "u < 70000 not taken",
            Box::new(move |b, x| {
                let k = b.constant(x, Ty::I32, 70000);
                b.cmp(x, IntPred::Ult, u, k)
            }),
            true,
            false,
        ));
        conditions.push((
            "u < 100 taken",
            "u < 100 not taken",
            Box::new(move |b, x| {
                let k = b.constant(x, Ty::I32, 100);
                b.cmp(x, IntPred::Ult, u, k)
            }),
            true,
            true,
        ));
        conditions.push((
            "any(lane == 3) taken",
            "any(lane == 3) not taken",
            Box::new(move |b, x| {
                let three = b.constant(x, Ty::I32, 3);
                let is = b.cmp(x, IntPred::Eq, lane, three);
                b.wave(x, WaveOp::Any, vec![is])
            }),
            lanes > 3,
            lanes <= 3,
        ));
        let wgid = extra[0];
        conditions.push((
            "workgroup < 4 taken",
            "workgroup < 4 not taken",
            Box::new(move |b, x| {
                let four = b.constant(x, Ty::I32, 4);
                b.cmp(x, IntPred::Ult, wgid, four)
            }),
            true,
            false,
        ));
        let mut reached = true;
        for (yes_name, no_name, condition, taken, other) in conditions {
            let c = condition(&mut b, at);
            let (then, t) = b.block(&[Ty::I1]);
            let (skip, _) = b.block(&[Ty::I1]);
            b.cond_br(at, c, (then, vec![exec]), (skip, vec![exec]));
            blocks.push((yes_name, then, reached && taken));
            blocks.push((no_name, skip, reached && other));
            reached = reached && taken;
            at = then;
            exec = t[0];
        }
        Branches { b, blocks }
    }

    fn reaching(lanes: u32) -> (Vec<String>, Vec<String>) {
        let Branches { b, blocks } = branches(lanes);
        let mut env = environment(lanes, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        env.grid = [4, 1, 1];
        let (mut wrong, mut loose) = (Vec::new(), Vec::new());
        addresses(&b, &env, |a| {
            for (name, block, truth) in &blocks {
                match (a.reaches_block(*block), *truth) {
                    (false, true) => wrong.push(format!("{}: said unreachable", name)),
                    (true, false) => loose.push(format!("{}: said reachable", name)),
                    _ => {}
                }
            }
        });
        (wrong, loose)
    }

    #[test]
    fn reaches_block_drops_only_blocks_no_execution_reaches() {
        let mut wrong = reaching(32).0;
        wrong.extend(reaching(2).0);
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn reaches_block_drops_every_block_whose_branch_is_decided_against_it() {
        let mut loose = reaching(32).1;
        loose.extend(reaching(2).1);
        assert!(loose.is_empty(), "{:?}", loose);
    }

    struct Counted {
        b: Build,
        index: ValueId,
        truth: Vec<u32>,
    }

    fn counted(start: u32, step: u32, pred: IntPred, limit: u32) -> Counted {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let s = b.constant(e, Ty::I32, start as u64);
        let (body, p) = b.block(&[Ty::I1, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, s]);
        let d = b.constant(body, Ty::I32, step as u64);
        let next = b.int(body, IntOp::Add, p[1], d);
        let l = b.constant(body, Ty::I32, limit as u64);
        let again = b.cmp(body, pred, next, l);
        b.cond_br(body, again, (body, vec![p[0], next]), (exit, vec![p[0]]));
        let mut truth = vec![start];
        let mut i = start;
        loop {
            let next = i.wrapping_add(step);
            if !super::compare(pred, next, limit) || truth.len() > 1000 {
                break;
            }
            truth.push(next);
            i = next;
        }
        Counted { b, index: p[1], truth }
    }

    fn loops() -> Vec<(&'static str, Counted)> {
        vec![
            ("0, 1, ... while < 4", counted(0, 1, IntPred::Ult, 4)),
            ("0, 2, ... while != 8", counted(0, 2, IntPred::Ne, 8)),
            ("10, 7, ... while > 0 signed", counted(10, (-3i32) as u32, IntPred::Sgt, 0)),
            ("0, 5, ... while <= 20", counted(0, 5, IntPred::Ule, 20)),
            ("3, 4, ... while < 3", counted(3, 1, IntPred::Ult, 3)),
        ]
    }

    #[test]
    fn loop_values_hold_every_value_an_iteration_gives() {
        let env = environment(32, &[(0, 1, 0x1000)]);
        let mut wrong = Vec::new();
        for (name, c) in loops() {
            addresses(&c.b, &env, |a| {
                let form = a.value(c.index, 0, None).0.form;
                for &t in &c.truth {
                    if !representable(&a.unknowns, &form, t) {
                        wrong.push(format!("{}: {:?} cannot be {}", name, form, t));
                    }
                }
            });
        }
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn loop_values_hold_only_the_values_the_iterations_give() {
        let env = environment(32, &[(0, 1, 0x1000)]);
        let mut loose = Vec::new();
        for (name, c) in loops() {
            addresses(&c.b, &env, |a| {
                let form = a.value(c.index, 0, None).0.form;
                let extra: Vec<u32> = vec![c.truth[0].wrapping_sub(1), c.truth.last().unwrap().wrapping_add(1), 0x7fff_ffff]
                    .into_iter()
                    .filter(|x| !c.truth.contains(x))
                    .filter(|&x| exactly_representable(&a.unknowns, &form, x) != Some(false))
                    .collect();
                if !extra.is_empty() {
                    loose.push(format!("{}: {:?} also holds {:?}", name, form, extra));
                }
            });
        }
        assert!(loose.is_empty(), "{:?}", loose);
    }

    fn carried_param(alternate: bool) -> (Build, ValueId, [u32; 2]) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I64, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, first, second, zero]);
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[3], one);
        let four = b.constant(body, Ty::I32, 4);
        let again = b.cmp(body, IntPred::Ult, next, four);
        let (carried, spare) = if alternate { (p[2], p[1]) } else { (p[1], p[2]) };
        b.cond_br(body, again, (body, vec![p[0], carried, spare, next]), (exit, vec![p[0]]));
        (b, p[1], [0x1000, 0x2000])
    }

    #[test]
    fn a_parameter_the_loop_passes_back_unchanged_keeps_its_entering_value() {
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let (b, p, _) = carried_param(false);
        let form = addresses(&b, &env, |a| a.value(p, 0, None).0.form);
        assert_eq!(form, Form::constant(0x1000));
    }

    #[test]
    fn a_parameter_the_loop_swaps_holds_both_of_its_values() {
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)]);
        let (b, p, truth) = carried_param(true);
        let missed: Vec<u32> = addresses(&b, &env, |a| {
            let form = a.value(p, 0, None).0.form;
            truth.iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "the parameter is the first pointer in even iterations and the second in odd ones: {:?}", missed);
    }

    #[test]
    fn value_and_bit_are_right_whenever_they_decide_an_operation_of_lane_known_words() {
        let (wrong, _) = decided(41, 3000);
        assert!(wrong.is_empty(), "{} wrong, first {:?}", wrong.len(), &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn value_and_bit_decide_every_operation_of_lane_known_words() {
        let (_, undecided) = decided(41, 3000);
        let mut kinds: Vec<String> = undecided.iter().map(|s| s.split(" lane").next().unwrap().to_string()).collect();
        kinds.sort();
        kinds.dedup();
        assert!(undecided.is_empty(), "{} undecided, kinds {:?}", undecided.len(), kinds);
    }

    fn scratch(b: &mut Build, e: BlockId, offset: u64) -> (ValueId, ValueId) {
        let base = b.core(e, Ty::I64, Op::Env(Env::ScratchBase));
        let k = b.constant(e, Ty::I64, offset);
        (base, b.int(e, IntOp::Add, base, k))
    }

    fn aperture(bound: impl Fn(&mut Build, BlockId, ValueId) -> ValueId) -> Option<bool> {
        let (mut b, _) = Build::kernel();
        let e = BlockId(0);
        let (base, p) = scratch(&mut b, e, 8);
        let end = bound(&mut b, e, base);
        let above = b.cmp(e, IntPred::Uge, p, base);
        let below = b.cmp(e, IntPred::Ult, p, end);
        let inside = b.int(e, IntOp::And, above, below);
        addresses(&b, &environment(32, &[]), |a| a.bit(inside, 0, None).0)
    }

    #[test]
    fn aperture_tests_find_a_scratch_pointer_below_the_scratch_size() {
        let found = aperture(|b, e, base| {
            let size = b.core(e, Ty::I64, Op::Env(Env::ScratchSize));
            b.int(e, IntOp::Add, base, size)
        });
        assert_eq!(found, Some(true));
    }

    #[test]
    fn aperture_tests_decide_only_what_their_bound_gives() {
        let found = aperture(|b, e, base| {
            let four = b.constant(e, Ty::I64, 4);
            b.int(e, IntOp::Add, base, four)
        });
        assert_ne!(found, Some(true), "base + 8 is not below base + 4");
    }

    #[test]
    fn comparisons_of_offsets_into_one_region_decide_what_holds_for_every_base() {
        let (mut b, _) = Build::kernel();
        let e = BlockId(0);
        let (_, far) = scratch(&mut b, e, 16);
        let (_, near) = scratch(&mut b, e, 8);
        let far_low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, far));
        let near_low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, near));
        let cases = [
            ("base + 16 != base + 8", b.cmp(e, IntPred::Ne, far, near)),
            ("low(base + 16) != low(base + 8)", b.cmp(e, IntPred::Ne, far_low, near_low)),
            ("low(base + 16) == low(base + 16)", b.cmp(e, IntPred::Eq, far_low, far_low)),
        ];
        let undecided: Vec<&str> = addresses(&b, &environment(32, &[]), |a| {
            cases.iter().filter(|c| a.bit(c.1, 0, None).0 != Some(true)).map(|c| c.0).collect()
        });
        assert!(undecided.is_empty(), "{:?}", undecided);
    }

    #[test]
    fn comparisons_of_offsets_into_a_region_decide_nothing_that_depends_on_its_base() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let (_, p) = scratch(&mut b, e, 16);
        let low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, p));
        let sixteen = b.constant(e, Ty::I32, 16);
        let hundred = b.constant(e, Ty::I32, 100);
        let zero = b.constant(e, Ty::I32, 0);
        let address = b.constant(e, Ty::I64, 0x1010);
        let cases = [
            ("low(base + 16) == 16", b.cmp(e, IntPred::Eq, low, sixteen)),
            ("low(base + 16) < 100", b.cmp(e, IntPred::Ult, low, hundred)),
            ("low(kernarg) == 0", b.cmp(e, IntPred::Eq, k.kernarg.0, zero)),
            ("base + 16 == 0x1010", b.cmp(e, IntPred::Eq, p, address)),
        ];
        let decided: Vec<&str> = addresses(&b, &environment(32, &[]), |a| {
            cases.iter().filter(|c| a.bit(c.1, 0, None).0.is_some()).map(|c| c.0).collect()
        });
        assert!(decided.is_empty(), "each holds for some bases and not for others: {:?}", decided);
    }

    #[test]
    fn values_computed_from_offsets_into_a_region_fix_nothing_that_depends_on_its_base() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let (_, p) = scratch(&mut b, e, 16);
        let low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, p));
        let four = b.constant(e, Ty::I32, 4);
        let two = b.constant(e, Ty::I32, 2);
        let three = b.constant(e, Ty::I64, 3);
        let kernarg = b.core(e, Ty::I64, Op::Pack64(k.kernarg.0, k.kernarg.1));
        let cases = [
            ("low(base + 16) >> 4", b.int(e, IntOp::LShr, low, four)),
            ("low(base + 16) * 2", b.int(e, IntOp::Mul, low, two)),
            ("(base + 16) | 3", b.int(e, IntOp::Or, p, three)),
            ("(base + 16) - kernarg", b.int(e, IntOp::Sub, p, kernarg)),
            ("low(kernarg) >> 4", b.int(e, IntOp::LShr, k.kernarg.0, four)),
        ];
        let fixed: Vec<String> = addresses(&b, &environment(32, &[]), |a| {
            cases
                .iter()
                .filter_map(|c| {
                    let value = a.value(c.1, 0, None).0;
                    (value.region.is_none() && value.form.as_constant().is_some())
                        .then(|| format!("{} = {:#x}", c.0, value.form.constant))
                })
                .collect()
        });
        assert!(fixed.is_empty(), "each depends on where the regions start: {:?}", fixed);
    }

    #[test]
    fn values_computed_from_offsets_into_one_region_keep_what_holds_for_every_base() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let (base, far) = scratch(&mut b, e, 16);
        let (_, near) = scratch(&mut b, e, 8);
        let far_low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, far));
        let near_low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, near));
        let high = b.constant(e, Ty::I64, 0xffff_ffff_0000_0000);
        let aperture = b.int(e, IntOp::And, base, high);
        let kernarg = b.core(e, Ty::I64, Op::Pack64(k.kernarg.0, k.kernarg.1));
        let twelve = b.constant(e, Ty::I64, 12);
        let past = b.int(e, IntOp::Add, kernarg, twelve);
        let cases = [
            ("(base + 16) - base", b.int(e, IntOp::Sub, far, base), 16),
            ("low(base + 16) - low(base + 8)", b.int(e, IntOp::Sub, far_low, near_low), 8),
            ("base & 0xffffffff00000000", aperture, 0),
            ("(kernarg + 12) - kernarg", b.int(e, IntOp::Sub, past, kernarg), 12),
        ];
        let loose: Vec<String> = addresses(&b, &environment(32, &[]), |a| {
            cases
                .iter()
                .filter_map(|c| {
                    let value = a.value(c.1, 0, None).0;
                    (value.form.as_constant() != Some(c.2)).then(|| format!("{}: {:?}", c.0, value.form))
                })
                .collect()
        });
        assert!(loose.is_empty(), "{:?}", loose);
    }

    fn lane_read(selector: impl Fn(&mut Build, &Kernel, BlockId, ValueId) -> ValueId) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let four = b.constant(e, Ty::I32, 4);
        let eight = b.constant(e, Ty::I32, 8);
        let scaled = b.int(e, IntOp::Mul, lane, four);
        let x = b.int(e, IntOp::Add, scaled, eight);
        let s = selector(&mut b, &k, e, lane);
        let register = b.constant(e, Ty::I32, 0);
        let read = b.wave(e, WaveOp::ReadLane, vec![x, s, register]);
        (b, read)
    }

    #[test]
    fn lane_reads_of_lanes_the_wave_lacks_fix_no_value_but_zero() {
        let (b, read) = lane_read(|b, _, e, _| b.constant(e, Ty::I32, 20));
        let form = addresses(&b, &environment(16, &[]), |a| a.value(read, 3, None).0.form);
        assert!(form.as_constant().is_none_or(|k| k == 0), "lane 20 is not in a wave of 16 lanes, so the read gives 0, not {:?}", form);
    }

    #[test]
    fn lane_reads_of_lanes_the_wave_lacks_give_zero() {
        let (b, read) = lane_read(|b, _, e, _| b.constant(e, Ty::I32, 20));
        let form = addresses(&b, &environment(16, &[]), |a| a.value(read, 3, None).0.form);
        assert_eq!(form, Form::constant(0));
    }

    #[test]
    fn lane_reads_of_one_word_in_a_partial_wave_fix_no_value_but_zero() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let seven = b.constant(e, Ty::I32, 7);
        let table = k.buffer(&mut b, e, 0);
        let yes = b.constant(e, Ty::I1, 1);
        let selector = b.load(e, Space::Global, MemSize::B32, table, yes);
        let register = b.constant(e, Ty::I32, 0);
        let read = b.wave(e, WaveOp::ReadLane, vec![seven, selector, register]);
        let form = addresses(&b, &environment(16, &[(0, 1, 0x1000)]), |a| a.value(read, 3, None).0.form);
        assert!(form.as_constant().is_none_or(|k| k == 0), "the selector may name a lane the wave of 16 lacks: {:?}", form);
    }

    #[test]
    fn lane_reads_with_a_uniform_selector_give_every_lane_one_value() {
        let (b, read) = lane_read(|b, k, e, _| {
            let table = k.buffer(b, e, 0);
            let yes = b.constant(e, Ty::I1, 1);
            b.load(e, Space::Global, MemSize::B32, table, yes)
        });
        let env = environment(32, &[(0, 1, 0x1000)]);
        let (third, fifth) = addresses(&b, &env, |a| (a.value(read, 3, None).0.form, a.value(read, 5, None).0.form));
        assert_eq!(third, fifth, "every lane reads the lane the one selector names");
    }

    #[test]
    fn lane_reads_with_a_varying_selector_read_each_lane_own_source() {
        let (b, read) = lane_read(|_, _, _, lane| lane);
        let form = addresses(&b, &environment(32, &[]), |a| a.value(read, 3, None).0.form);
        assert!(form.as_constant().is_none_or(|k| k == 20), "lane 3 reads its own 3 * 4 + 8, not {:?}", form);
    }

    #[test]
    fn lane_reads_with_a_selector_each_lane_loads_leave_the_lanes_apart() {
        let (b, read) = lane_read(|b, k, e, lane| {
            let buf = k.buffer(b, e, 0);
            let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
            let address = b.int(e, IntOp::Add, buf, wide);
            let yes = b.constant(e, Ty::I1, 1);
            b.load(e, Space::Global, MemSize::U8, address, yes)
        });
        let env = environment(32, &[(0, 1, 0x1000)]);
        let (third, fifth) = addresses(&b, &env, |a| (a.value(read, 3, None).0.form, a.value(read, 5, None).0.form));
        assert!(third != fifth || third.as_constant().is_some(), "lanes 3 and 5 may read different lanes: {:?}", third);
    }

    #[test]
    fn scratch_pointers_read_from_another_lane_equal_the_lane_own() {
        let (mut b, _) = Build::kernel();
        let e = BlockId(0);
        let (_, p) = scratch(&mut b, e, 16);
        let low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, p));
        let zero = b.constant(e, Ty::I32, 0);
        let register = b.constant(e, Ty::I32, 0);
        let read = b.wave(e, WaveOp::ReadLane, vec![low, zero, register]);
        let same = b.cmp(e, IntPred::Eq, read, low);
        let found = addresses(&b, &environment(32, &[]), |a| a.bit(same, 3, None).0);
        assert_eq!(found, Some(true), "every lane has the same scratch base");
    }

    #[test]
    fn float_conversions_to_a_bit_decide_only_what_the_saturating_conversion_gives() {
        let (mut b, _) = Build::kernel();
        let e = BlockId(0);
        let cases: Vec<(&str, Cvt, f32, bool)> = vec![
            ("signed -1.0", Cvt::FloatToSignedSatRtz, -1.0, true),
            ("signed 1.0000001", Cvt::FloatToSignedSatRtz, f32::from_bits(0x3f80_0001), false),
            ("signed -2.5", Cvt::FloatToSignedSatRtz, -2.5, true),
            ("unsigned 1.0", Cvt::FloatToUnsignedSatRtz, 1.0, true),
            ("unsigned 3.0", Cvt::FloatToUnsignedSatRtz, 3.0, true),
            ("unsigned 0.5", Cvt::FloatToUnsignedSatRtz, 0.5, false),
        ];
        let bits: Vec<(&str, ValueId, bool)> = cases
            .iter()
            .map(|&(name, cvt, x, truth)| {
                let k = b.constant(e, Ty::F32, x.to_bits() as u64);
                (name, b.core(e, Ty::I1, Op::Convert(cvt, Ty::I1, k)), truth)
            })
            .collect();
        let wrong: Vec<String> = addresses(&b, &environment(32, &[]), |a| {
            bits.iter()
                .filter_map(|&(name, v, truth)| {
                    let found = a.bit(v, 0, None).0;
                    found.is_some_and(|x| x != truth).then(|| format!("{}: {:?}", name, found))
                })
                .collect()
        });
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    struct Wide {
        b: Build,
        cases: Vec<(&'static str, ValueId, Vec<u64>)>,
    }

    fn wide_values() -> Wide {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let c64 = |b: &mut Build, k: u64| b.constant(e, Ty::I64, k);
        let c32 = |b: &mut Build, k: u64| b.constant(e, Ty::I32, k);
        let table = k.buffer(&mut b, e, 8);
        let yes = b.constant(e, Ty::I1, 1);
        let u = b.load(e, Space::Global, MemSize::U8, table, yes);
        let wide_u = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, u));
        let mut cases = Vec::new();
        let lo = c32(&mut b, 0x89ab_cdef);
        let hi = c32(&mut b, 0x0123_4567);
        let packed = b.core(e, Ty::I64, Op::Pack64(lo, hi));
        cases.push(("pack", packed, vec![0x0123_4567_89ab_cdef]));
        let near = c64(&mut b, 0xffff_fff0);
        let twenty = c64(&mut b, 0x20);
        let carried = b.int(e, IntOp::Add, near, twenty);
        cases.push(("0xfffffff0 + 0x20", carried, vec![0x1_0000_0010]));
        let sixteen = c64(&mut b, 0x10);
        let small = b.int(e, IntOp::Sub, sixteen, twenty);
        cases.push(("0x10 - 0x20", small, vec![0xffff_ffff_ffff_fff0]));
        let edge = c64(&mut b, 0xffff_ff80);
        let maybe = b.int(e, IntOp::Add, edge, wide_u);
        cases.push(("0xffffff80 + u", maybe, (0..256u64).map(|x| 0xffff_ff80 + x).collect()));
        let four = c64(&mut b, 36);
        let lifted = b.int(e, IntOp::Shl, wide_u, four);
        cases.push(("u << 36", lifted, (0..256u64).map(|x| x << 36).collect()));
        let eight = c64(&mut b, 8);
        let dropped = b.int(e, IntOp::LShr, packed, eight);
        cases.push(("pack >> 8", dropped, vec![0x0123_4567_89ab_cdef >> 8]));
        let far = c64(&mut b, 40);
        let top = b.int(e, IntOp::LShr, packed, far);
        cases.push(("pack >> 40", top, vec![0x0123_4567_89ab_cdef >> 40]));
        let minus = c32(&mut b, 0xffff_fff0);
        let extended = b.core(e, Ty::I64, Op::Convert(Cvt::SExt, Ty::I64, minus));
        cases.push(("sext -16", extended, vec![0xffff_ffff_ffff_fff0]));
        let masked_bits = c64(&mut b, 0xff00_0000_ffff_0000);
        let masked = b.int(e, IntOp::And, packed, masked_bits);
        cases.push(("pack & mask", masked, vec![0x0123_4567_89ab_cdef & 0xff00_0000_ffff_0000]));
        let shifted_up = b.int(e, IntOp::Shl, packed, eight);
        cases.push(("pack << 8", shifted_up, vec![0x0123_4567_89ab_cdef << 8]));
        Wide { b, cases }
    }

    #[test]
    fn high_words_hold_the_high_half_of_every_wide_value() {
        let Wide { b, cases } = wide_values();
        let missed: Vec<String> = addresses(&b, &environment(32, &[(8, 1, 0x1000)]), |a| {
            let mut missed = Vec::new();
            for (name, v, truths) in &cases {
                let (low, high) = (a.value(*v, 0, None).0.form, a.high(*v, 0));
                for &t in truths {
                    if !representable(&a.unknowns, &low, t as u32) || !representable(&a.unknowns, &high, (t >> 32) as u32) {
                        missed.push(format!("{}: {:#x} with {:?} and {:?}", name, t, low, high));
                        break;
                    }
                }
            }
            missed
        });
        assert!(missed.is_empty(), "{:?}", missed);
    }

    #[test]
    fn high_words_are_exact_where_the_operations_determine_them() {
        let Wide { b, cases } = wide_values();
        let loose: Vec<String> = addresses(&b, &environment(32, &[(8, 1, 0x1000)]), |a| {
            cases
                .iter()
                .filter(|(_, _, truths)| truths.len() == 1)
                .filter_map(|(name, v, truths)| {
                    let (low, high) = (a.value(*v, 0, None).0.form, a.high(*v, 0));
                    let t = truths[0];
                    (low.as_constant() != Some(t as u32) || high.as_constant() != Some((t >> 32) as u32))
                        .then(|| format!("{}: {:?} and {:?}", name, low, high))
                })
                .collect()
        });
        assert!(loose.is_empty(), "{:?}", loose);
    }

    fn affine_loop(trips: u64) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let one = b.constant(e, Ty::I32, 1);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, one, zero]);
        let three = b.constant(body, Ty::I32, 3);
        let tripled = b.int(body, IntOp::Mul, p[1], three);
        let one = b.constant(body, Ty::I32, 1);
        let next_value = b.int(body, IntOp::Add, tripled, one);
        let next = b.int(body, IntOp::Add, p[2], one);
        let limit = b.constant(body, Ty::I32, trips);
        let again = b.cmp(body, IntPred::Ult, next, limit);
        b.cond_br(body, again, (body, vec![p[0], next_value, next]), (exit, vec![p[0]]));
        (b, p[1])
    }

    fn carried_values(b: &Build, v: ValueId) -> Option<Vec<u32>> {
        addresses(b, &environment(32, &[]), |a| {
            let form = a.value(v, 0, None).0.form;
            match form.terms.as_slice() {
                [(u, 1)] if form.constant == 0 => a.unknowns[*u as usize].values.as_ref().map(|s| s.to_vec()),
                _ => None,
            }
        })
    }

    #[test]
    fn loop_values_hold_every_value_an_affine_step_carries() {
        let (b, v) = affine_loop(5);
        let truth = [1u32, 4, 13, 40, 121];
        let missed: Vec<u32> = addresses(&b, &environment(32, &[]), |a| {
            let form = a.value(v, 0, None).0.form;
            truth.iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "the parameter runs 1, 4, 13, 40, 121: {:?}", missed);
    }

    #[test]
    fn loop_values_are_exactly_those_an_affine_step_carries() {
        let (b, v) = affine_loop(5);
        assert_eq!(carried_values(&b, v), Some(vec![1, 4, 13, 40, 121]));
    }

    fn affine_loop_from_either() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let is_zero = b.cmp(e, IntPred::Eq, u, zero);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let start = b.core(e, Ty::I32, Op::Select(is_zero, one, two));
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, start, zero]);
        let three = b.constant(body, Ty::I32, 3);
        let tripled = b.int(body, IntOp::Mul, p[1], three);
        let one = b.constant(body, Ty::I32, 1);
        let next_value = b.int(body, IntOp::Add, tripled, one);
        let next = b.int(body, IntOp::Add, p[2], one);
        let limit = b.constant(body, Ty::I32, 5);
        let again = b.cmp(body, IntPred::Ult, next, limit);
        b.cond_br(body, again, (body, vec![p[0], next_value, next]), (exit, vec![p[0]]));
        (b, p[1])
    }

    #[test]
    fn loop_values_hold_every_value_an_affine_step_carries_from_either_start() {
        let (b, v) = affine_loop_from_either();
        let truth = [1u32, 2, 4, 7, 13, 22, 40, 67, 121, 202];
        let missed: Vec<u32> = addresses(&b, &environment(32, &[]), |a| {
            let form = a.value(v, 0, None).0.form;
            truth.iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "the parameter runs 1, 4, 13, 40, 121 or 2, 7, 22, 67, 202: {:?}", missed);
    }

    #[test]
    fn loop_values_are_exactly_those_an_affine_step_carries_from_either_start() {
        let (b, v) = affine_loop_from_either();
        assert_eq!(carried_values(&b, v), Some(vec![1, 2, 4, 7, 13, 22, 40, 67, 121, 202]));
    }

    fn hundred_affine_values() -> Vec<u32> {
        let mut values: Vec<u32> = std::iter::successors(Some(1u32), |x| Some(x.wrapping_mul(3).wrapping_add(1))).take(100).collect();
        values.sort_unstable();
        values
    }

    #[test]
    fn loop_values_hold_every_value_an_affine_step_carries_over_a_hundred_iterations() {
        let (b, v) = affine_loop(100);
        let truth = hundred_affine_values();
        let missed: Vec<u32> = addresses(&b, &environment(32, &[]), |a| {
            let form = a.value(v, 0, None).0.form;
            truth.iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "the parameter runs 1, 4, 13 and on for 100 iterations: {:?}", missed);
    }

    #[test]
    fn loop_values_are_exactly_those_an_affine_step_carries_over_a_hundred_iterations() {
        let (b, v) = affine_loop(100);
        let carried = carried_values(&b, v).map(|mut s| {
            s.sort_unstable();
            s
        });
        assert_eq!(carried, Some(hundred_affine_values()));
    }

    #[test]
    fn float_conversions_to_integers_give_what_the_saturating_conversion_gives() {
        let (mut b, _) = Build::kernel();
        let e = BlockId(0);
        let cases: Vec<(&str, Cvt, Ty, f32, u32)> = vec![
            ("signed bit of -1.0", Cvt::FloatToSignedSatRtz, Ty::I1, -1.0, 1),
            ("signed bit of -0.5", Cvt::FloatToSignedSatRtz, Ty::I1, -0.5, 0),
            ("signed bit of 1.0", Cvt::FloatToSignedSatRtz, Ty::I1, 1.0, 0),
            ("unsigned bit of 1.0", Cvt::FloatToUnsignedSatRtz, Ty::I1, 1.0, 1),
            ("unsigned bit of 0.99", Cvt::FloatToUnsignedSatRtz, Ty::I1, 0.99, 0),
            ("unsigned bit of NaN", Cvt::FloatToUnsignedSatRtz, Ty::I1, f32::NAN, 0),
            ("signed word of -2.5", Cvt::FloatToSignedSatRtz, Ty::I32, -2.5, (-2i32) as u32),
            ("signed word of 3e9", Cvt::FloatToSignedSatRtz, Ty::I32, 3e9, i32::MAX as u32),
            ("unsigned word of -7.0", Cvt::FloatToUnsignedSatRtz, Ty::I32, -7.0, 0),
            ("unsigned word of 5e9", Cvt::FloatToUnsignedSatRtz, Ty::I32, 5e9, u32::MAX),
            ("signed low word of -1.0", Cvt::FloatToSignedSatRtz, Ty::I64, -1.0, u32::MAX),
            ("unsigned low word of 2^33 + 2^10", Cvt::FloatToUnsignedSatRtz, Ty::I64, 8589935616.0, 1024),
        ];
        let converted: Vec<(&str, ValueId, Ty, u32)> = cases
            .iter()
            .map(|&(name, cvt, to, x, truth)| {
                let k = b.constant(e, Ty::F32, x.to_bits() as u64);
                (name, b.core(e, to, Op::Convert(cvt, to, k)), to, truth)
            })
            .collect();
        let loose: Vec<String> = addresses(&b, &environment(32, &[]), |a| {
            converted
                .iter()
                .filter_map(|&(name, v, to, truth)| {
                    let found = if to == Ty::I1 {
                        a.bit(v, 0, None).0.map(|x| x as u32)
                    } else {
                        a.value(v, 0, None).0.form.as_constant()
                    };
                    (found != Some(truth)).then(|| format!("{}: {:?}", name, found))
                })
                .collect()
        });
        assert!(loose.is_empty(), "{:?}", loose);
    }

    fn loaded_word(b: &mut Build, k: &Kernel, e: BlockId, offset: u64, size: MemSize) -> ValueId {
        let table = k.buffer(b, e, 8);
        let at = b.constant(e, Ty::I64, offset);
        let address = b.int(e, IntOp::Add, table, at);
        let yes = b.constant(e, Ty::I1, 1);
        b.load(e, Space::Global, size, address, yes)
    }

    fn two_words() -> Environment {
        environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)])
    }

    fn identities() -> (Build, Vec<(&'static str, ValueId)>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
        let byte = loaded_word(&mut b, &k, e, 8, MemSize::U8);
        let thirty_one = b.constant(e, Ty::I32, 31);
        let s = b.int(e, IntOp::And, byte, thirty_one);
        let uv = b.int(e, IntOp::Mul, u, v);
        let vu = b.int(e, IntOp::Mul, v, u);
        let products = b.int(e, IntOp::Sub, uv, vu);
        let both = b.int(e, IntOp::And, u, v);
        let either = b.int(e, IntOp::Or, u, v);
        let sum = b.int(e, IntOp::Add, u, v);
        let parts = b.int(e, IntOp::Add, both, either);
        let bits = b.int(e, IntOp::Sub, parts, sum);
        let shifted = b.int(e, IntOp::Shl, u, s);
        let one = b.constant(e, Ty::I32, 1);
        let power = b.int(e, IntOp::Shl, one, s);
        let scaled = b.int(e, IntOp::Mul, u, power);
        let shifts = b.int(e, IntOp::Sub, shifted, scaled);
        (b, vec![("u * v - v * u", products), ("(u & v) + (u | v) - (u + v)", bits), ("(u << s) - u * (1 << s)", shifts)])
    }

    #[test]
    fn identities_of_nonlinear_operations_hold_zero() {
        let (b, cases) = identities();
        let missed: Vec<&str> = addresses(&b, &two_words(), |a| {
            cases
                .iter()
                .filter(|c| {
                    let form = a.value(c.1, 3, None).0.form;
                    !representable(&a.unknowns, &form, 0)
                })
                .map(|c| c.0)
                .collect()
        });
        assert!(missed.is_empty(), "{:?}", missed);
    }

    #[test]
    fn identities_of_nonlinear_operations_give_zero() {
        let (b, cases) = identities();
        let loose: Vec<&str> = addresses(&b, &two_words(), |a| {
            cases.iter().filter(|c| a.value(c.1, 3, None).0.form.as_constant() != Some(0)).map(|c| c.0).collect()
        });
        assert!(loose.is_empty(), "each is 0 for every u, v and s: {:?}", loose);
    }

    fn counted_bits() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let count = b.core(e, Ty::I32, Op::PopulationCount(u));
        let limit = b.constant(e, Ty::I32, 32);
        let within = b.cmp(e, IntPred::Ule, count, limit);
        (b, within)
    }

    #[test]
    fn bit_counts_decide_nothing_false() {
        let (b, within) = counted_bits();
        let found = addresses(&b, &two_words(), |a| a.bit(within, 0, None).0);
        assert_ne!(found, Some(false), "a word has at most 32 bits set");
    }

    #[test]
    fn bit_counts_stay_within_the_width() {
        let (b, within) = counted_bits();
        let found = addresses(&b, &two_words(), |a| a.bit(within, 0, None).0);
        assert_eq!(found, Some(true), "a word has at most 32 bits set");
    }

    fn wide_sums() -> (Build, Vec<(&'static str, ValueId, u32)>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let step = b.constant(e, Ty::I64, 256);
        let past = b.int(e, IntOp::Add, buf, step);
        let eight = b.constant(e, Ty::I64, 8);
        let units = b.int(e, IntOp::LShr, past, eight);
        let low = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, units));
        let far = b.constant(e, Ty::I64, 0x1_0000_0000);
        let above = b.int(e, IntOp::Add, buf, far);
        let high = b.core(e, Ty::I32, Op::UnpackHi(above));
        (b, vec![("low((buf + 256) >> 8)", low, 0x11), ("high(buf + 2^32)", high, 1)])
    }

    #[test]
    fn wide_shifts_and_high_halves_of_sums_hold_their_values() {
        let (b, cases) = wide_sums();
        let missed: Vec<&str> = addresses(&b, &two_words(), |a| {
            cases
                .iter()
                .filter(|c| {
                    let form = a.value(c.1, 0, None).0.form;
                    !representable(&a.unknowns, &form, c.2)
                })
                .map(|c| c.0)
                .collect()
        });
        assert!(missed.is_empty(), "{:?}", missed);
    }

    #[test]
    fn wide_shifts_and_high_halves_of_sums_give_their_values() {
        let (b, cases) = wide_sums();
        let loose: Vec<String> = addresses(&b, &two_words(), |a| {
            cases
                .iter()
                .filter_map(|c| {
                    let form = a.value(c.1, 0, None).0.form;
                    (form.as_constant() != Some(c.2)).then(|| format!("{}: {:?}", c.0, form))
                })
                .collect()
        });
        assert!(loose.is_empty(), "buf is 0x1000: {:?}", loose);
    }

    fn shared_word() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let four = b.constant(e, Ty::I32, 4);
        let own = b.int(e, IntOp::Mul, lane, four);
        b.store(e, Space::Lds, MemSize::B32, own, lane, k.exec);
        let back = b.load(e, Space::Lds, MemSize::B32, own, k.exec);
        (b, back)
    }

    #[test]
    fn lds_words_read_back_hold_the_word_the_lane_stored() {
        let (b, back) = shared_word();
        let held = addresses(&b, &two_words(), |a| {
            let form = a.value(back, 3, None).0.form;
            representable(&a.unknowns, &form, 3)
        });
        assert!(held, "lane 3 reads back the 3 it stored");
    }

    #[test]
    fn lds_words_read_back_give_the_word_the_lane_stored() {
        let (b, back) = shared_word();
        let form = addresses(&b, &two_words(), |a| a.value(back, 3, None).0.form);
        assert_eq!(form, Form::constant(3), "word 3 of the LDS only ever holds 3");
    }

    fn squared_loop() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let one = b.constant(e, Ty::I32, 1);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, one, zero]);
        let squared = b.int(body, IntOp::Mul, p[1], p[1]);
        let one = b.constant(body, Ty::I32, 1);
        let next_value = b.int(body, IntOp::Add, squared, one);
        let next = b.int(body, IntOp::Add, p[2], one);
        let three = b.constant(body, Ty::I32, 3);
        let again = b.cmp(body, IntPred::Ult, next, three);
        b.cond_br(body, again, (body, vec![p[0], next_value, next]), (exit, vec![p[0]]));
        (b, p[1])
    }

    #[test]
    fn loop_values_hold_every_value_a_square_step_carries() {
        let (b, v) = squared_loop();
        let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
            let form = a.value(v, 0, None).0.form;
            [1u32, 2, 5].iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "the parameter runs 1, 2, 5: {:?}", missed);
    }

    #[test]
    fn loop_values_are_exactly_those_a_square_step_carries() {
        let (b, v) = squared_loop();
        assert_eq!(carried_values(&b, v), Some(vec![1, 2, 5]));
    }

    fn halves_in_two_blocks() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let one = b.constant(e, Ty::I32, 1);
        let x = b.int(e, IntOp::LShr, u, one);
        let (next, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        b.br(e, next, vec![k.exec, u, x]);
        let one = b.constant(next, Ty::I32, 1);
        let y = b.int(next, IntOp::LShr, p[1], one);
        let difference = b.int(next, IntOp::Sub, p[2], y);
        (b, difference)
    }

    #[test]
    fn halves_of_one_word_in_two_blocks_hold_their_difference() {
        let (b, difference) = halves_in_two_blocks();
        let held = addresses(&b, &two_words(), |a| {
            let form = a.value(difference, 0, None).0.form;
            representable(&a.unknowns, &form, 0)
        });
        assert!(held);
    }

    #[test]
    fn halves_of_one_word_in_two_blocks_are_equal() {
        let (b, difference) = halves_in_two_blocks();
        let form = addresses(&b, &two_words(), |a| a.value(difference, 0, None).0.form);
        assert_eq!(form, Form::constant(0), "both are u >> 1 of the same u");
    }

    fn chosen_after_branch(entered: bool) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let is_zero = b.cmp(e, IntPred::Eq, u, zero);
        let (then, t) = b.block(&[Ty::I1, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32]);
        b.cond_br(e, is_zero, (then, vec![k.exec, u]), (other, vec![k.exec, u]));
        let (block, p) = if entered { (then, t) } else { (other, o) };
        let zero = b.constant(block, Ty::I32, 0);
        let test = b.cmp(block, IntPred::Eq, p[1], zero);
        let seven = b.constant(block, Ty::I32, 7);
        let nine = b.constant(block, Ty::I32, 9);
        let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
        (b, chosen)
    }

    #[test]
    fn selects_after_a_branch_hold_the_arm_the_branch_leaves() {
        let missed: Vec<bool> = [true, false]
            .iter()
            .copied()
            .filter(|&entered| {
                let (b, chosen) = chosen_after_branch(entered);
                let truth = if entered { 7 } else { 9 };
                !addresses(&b, &two_words(), |a| {
                    let form = a.value(chosen, 0, None).0.form;
                    representable(&a.unknowns, &form, truth)
                })
            })
            .collect();
        assert!(missed.is_empty(), "{:?}", missed);
    }

    #[test]
    fn selects_after_a_branch_take_the_arm_the_branch_decides() {
        let (b, chosen) = chosen_after_branch(true);
        let form = addresses(&b, &two_words(), |a| a.value(chosen, 0, None).0.form);
        assert_eq!(form, Form::constant(7), "the block runs only when u is 0");
    }

    fn zero_cases(b: &Build, cases: &[(&'static str, ValueId)], exact: bool) -> Vec<String> {
        addresses(b, &two_words(), |a| {
            cases
                .iter()
                .filter_map(|&(name, v)| {
                    let form = a.value(v, 3, None).0.form;
                    let wrong = if exact { form.as_constant() != Some(0) } else { !representable(&a.unknowns, &form, 0) };
                    wrong.then(|| format!("{}: {:?}", name, form))
                })
                .collect()
        })
    }

    fn distributed_products() -> (Build, Vec<(&'static str, ValueId)>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
        let w = loaded_word(&mut b, &k, e, 8, MemSize::B32);
        let one = b.constant(e, Ty::I32, 1);
        let next = b.int(e, IntOp::Add, v, one);
        let product = b.int(e, IntOp::Mul, u, next);
        let uv = b.int(e, IntOp::Mul, u, v);
        let sum = b.int(e, IntOp::Add, uv, u);
        let distributed = b.int(e, IntOp::Sub, product, sum);
        let vw = b.int(e, IntOp::Mul, v, w);
        let left = b.int(e, IntOp::Mul, u, vw);
        let uv = b.int(e, IntOp::Mul, u, v);
        let right = b.int(e, IntOp::Mul, uv, w);
        let associated = b.int(e, IntOp::Sub, left, right);
        (b, vec![("u * (v + 1) - (u * v + u)", distributed), ("u * (v * w) - (u * v) * w", associated)])
    }

    #[test]
    fn distributed_and_regrouped_products_hold_zero() {
        let (b, cases) = distributed_products();
        let missed = zero_cases(&b, &cases, false);
        assert!(missed.is_empty(), "{:?}", missed);
    }

    fn polynomials() -> (Build, [ValueId; 3], Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> u32>)>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let v = loaded_word(&mut b, &k, e, 4, MemSize::U16);
        let w = loaded_word(&mut b, &k, e, 8, MemSize::U8);
        let c = |b: &mut Build, k: u32| b.constant(e, Ty::I32, k as u64);
        let mut out: Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32, u32) -> u32>)> = Vec::new();
        let (three, five, seven) = (c(&mut b, 3), c(&mut b, 5), c(&mut b, 7));
        let u3 = b.int(e, IntOp::Add, u, three);
        let v5 = b.int(e, IntOp::Add, v, five);
        let x = b.int(e, IntOp::Mul, u3, v5);
        out.push(("(u + 3)(v + 5)", x, Box::new(|u, v, _| u.wrapping_add(3).wrapping_mul(v + 5))));
        let difference = b.int(e, IntOp::Sub, u, v);
        let sum = b.int(e, IntOp::Add, u, v);
        let x = b.int(e, IntOp::Mul, difference, sum);
        out.push(("(u - v)(u + v)", x, Box::new(|u, v, _| u.wrapping_sub(v).wrapping_mul(u.wrapping_add(v)))));
        let vw = b.int(e, IntOp::Mul, v, w);
        let x = b.int(e, IntOp::Mul, u, vw);
        out.push(("u (v w)", x, Box::new(|u, v, w| u.wrapping_mul(v * w))));
        let uv = b.int(e, IntOp::Mul, u, v);
        let uv7 = b.int(e, IntOp::Add, uv, seven);
        let two = c(&mut b, 2);
        let w2 = b.int(e, IntOp::Sub, w, two);
        let x = b.int(e, IntOp::Mul, uv7, w2);
        out.push(("(u v + 7)(w - 2)", x, Box::new(|u, v, w| u.wrapping_mul(v).wrapping_add(7).wrapping_mul(w.wrapping_sub(2)))));
        let (four, one) = (c(&mut b, 4), c(&mut b, 1));
        let u2 = b.int(e, IntOp::Mul, u, two);
        let v3 = b.int(e, IntOp::Mul, v, three);
        let left = b.int(e, IntOp::Add, u2, v3);
        let w4 = b.int(e, IntOp::Mul, w, four);
        let right = b.int(e, IntOp::Add, w4, one);
        let lr = b.int(e, IntOp::Mul, left, right);
        let x = b.int(e, IntOp::Mul, lr, u);
        out.push(("(2u + 3v)(4w + 1) u", x, Box::new(|u, v, w| u.wrapping_mul(2).wrapping_add(3 * v).wrapping_mul(4 * w + 1).wrapping_mul(u))));
        let u1 = b.int(e, IntOp::Add, u, one);
        let square = b.int(e, IntOp::Mul, u1, u1);
        let x = b.int(e, IntOp::Mul, square, u1);
        out.push(("(u + 1)^3", x, Box::new(|u, _, _| u.wrapping_add(1).wrapping_mul(u.wrapping_add(1)).wrapping_mul(u.wrapping_add(1)))));
        let vv = b.int(e, IntOp::Mul, v, v);
        let x = b.int(e, IntOp::Mul, vv, vv);
        out.push(("v^4", x, Box::new(|_, v, _| v.wrapping_mul(v).wrapping_mul(v).wrapping_mul(v))));
        let sixteen = c(&mut b, 16);
        let shifted = b.int(e, IntOp::Shl, v, sixteen);
        let x = b.int(e, IntOp::Mul, shifted, w);
        out.push(("(v << 16) w", x, Box::new(|_, v, w| (v << 16).wrapping_mul(w))));
        (b, [u, v, w], out)
    }

    #[test]
    fn products_expand_into_monomials_whose_factors_multiply_to_the_true_value() {
        let (b, words, cases) = polynomials();
        let mut wrong = Vec::new();
        addresses(&b, &two_words(), |a| {
            let bases: Vec<Form> = words.iter().map(|&x| a.value(x, 0, None).0.form).collect();
            let forms: Vec<Form> = cases.iter().map(|&(_, x, _)| a.value(x, 0, None).0.form).collect();
            let mut r = Random::new(109);
            let mut samples: Vec<[u32; 3]> = vec![[0, 0, 0], [1, 1, 1], [u32::MAX, 65535, 255], [0x8000_0000, 0x8000, 0x80], [12345, 678, 9]];
            samples.extend((0..40).map(|_| [r.next() as u32, r.below(65536) as u32, r.below(256) as u32]));
            for sample in samples {
                let value = |u: Unknown| -> Option<u32> {
                    let base = |u: Unknown| bases.iter().position(|f| *f == Form::unknown(u)).map(|i| sample[i]);
                    match a.monomials.get(&u) {
                        Some(factors) => factors.iter().try_fold(1u32, |p, &f| Some(p.wrapping_mul(base(f)?))),
                        None => base(u),
                    }
                };
                for (i, (name, _, truth)) in cases.iter().enumerate() {
                    let evaluated = forms[i].terms.iter().try_fold(forms[i].constant, |acc, &(u, c)| Some(acc.wrapping_add(c.wrapping_mul(value(u)?))));
                    let t = truth(sample[0], sample[1], sample[2]);
                    if evaluated != Some(t) {
                        wrong.push(format!("{} at {:?}: {:?} gives {:?}, not {:#x}", name, sample, forms[i], evaluated, t));
                    }
                }
                for (u, factors) in &a.monomials {
                    let Some(p) = value(*u) else {
                        continue;
                    };
                    if let Some((low, high)) = a.unknowns[*u as usize].range {
                        if p < low || p > high {
                            wrong.push(format!("monomial {:?} at {:?} is {:#x}, outside {:?}", factors, sample, p, (low, high)));
                        }
                    }
                }
            }
        });
        wrong.sort();
        wrong.dedup();
        assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn wide_values_are_the_addresses_as_integers_whenever_they_are_given() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let s = loaded_word(&mut b, &k, e, 4, MemSize::U8);
        let t = loaded_word(&mut b, &k, e, 8, MemSize::U16);
        let zext = |b: &mut Build, x: ValueId| b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, x));
        let wide = |b: &mut Build, k: u64| b.constant(e, Ty::I64, k);
        let (zu, zs, zt) = (zext(&mut b, u), zext(&mut b, s), zext(&mut b, t));
        let mut cases: Vec<(&'static str, ValueId, bool, Box<dyn Fn(u64, u64, u64) -> u64>)> = Vec::new();
        let four = wide(&mut b, 4);
        let scaled = b.int(e, IntOp::Mul, zu, four);
        let x = b.int(e, IntOp::Add, buf, scaled);
        cases.push(("buf + 4u", x, true, Box::new(|u, _, _| 0x1000 + 4 * u)));
        let eight = wide(&mut b, 8);
        let y = b.int(e, IntOp::Sub, x, eight);
        cases.push(("buf + 4u - 8", y, true, Box::new(|u, _, _| 0x1000 + 4 * u - 8)));
        let three = wide(&mut b, 3);
        let shifted = b.int(e, IntOp::Shl, zu, three);
        let two = wide(&mut b, 2);
        let doubled = b.int(e, IntOp::Mul, two, zs);
        let x = b.int(e, IntOp::Add, shifted, doubled);
        cases.push(("(u << 3) + 2s", x, true, Box::new(|u, s, _| (u << 3) + 2 * s)));
        let row = wide(&mut b, 256);
        let rows = b.int(e, IntOp::Mul, zt, row);
        let x = b.int(e, IntOp::Add, zs, rows);
        cases.push(("s + 256t", x, true, Box::new(|_, s, t| s + 256 * t)));
        let big = wide(&mut b, 1 << 33);
        let far = b.int(e, IntOp::Mul, zu, big);
        let x = b.int(e, IntOp::Add, buf, far);
        cases.push(("buf + 2^33 u", x, false, Box::new(|u, _, _| 0x1000u64.wrapping_add(u.wrapping_mul(1 << 33)))));
        let five = b.constant(e, Ty::I32, 5);
        let bumped = b.int(e, IntOp::Add, u, five);
        let x = zext(&mut b, bumped);
        cases.push(("zext(u + 5)", x, true, Box::new(|u, _, _| (u as u32).wrapping_add(5) as u64)));
        let bumped = b.int(e, IntOp::Add, t, five);
        let x = zext(&mut b, bumped);
        cases.push(("zext(t + 5)", x, true, Box::new(|_, _, t| t + 5)));
        let x = b.int(e, IntOp::Sub, zs, zt);
        cases.push(("s - t", x, false, Box::new(|_, s, t| s.wrapping_sub(t))));
        let mut wrong = Vec::new();
        addresses(&b, &two_words(), |a| {
            let bases: Vec<Form> = [u, s, t].iter().map(|&x| a.value(x, 0, None).0.form).collect();
            let forms: Vec<Option<super::Wide>> = cases.iter().map(|&(_, x, _, _)| a.wide_value(x, e, 0)).collect();
            let mut r = Random::new(127);
            let mut samples: Vec<[u64; 3]> = vec![[0, 0, 0], [u32::MAX as u64, 255, 65535], [0x8000_0000, 128, 32768], [1, 1, 1]];
            samples.extend((0..40).map(|_| [r.next() as u32 as u64, r.below(256), r.below(65536)]));
            for (i, (name, _, given, truth)) in cases.iter().enumerate() {
                match &forms[i] {
                    None if *given => wrong.push(format!("{}: no wide form", name)),
                    None => {}
                    Some(w) => {
                        for sample in &samples {
                            let at = |x: Unknown| bases.iter().position(|f| *f == Form::unknown(x)).map(|i| sample[i]);
                            let terms = w.terms.iter().try_fold(w.constant, |acc, &(x, c)| Some(acc + c * at(x)? as i128));
                            let value = terms.and_then(|terms| w.words.iter().try_fold(terms, |acc, (f, c)| {
                                let word = f.terms.iter().try_fold(f.constant, |acc, &(x, k)| Some(acc.wrapping_add(k.wrapping_mul(at(x)? as u32))))?;
                                Some(acc + c * word as i128)
                            }));
                            let t = truth(sample[0], sample[1], sample[2]) as i128;
                            if value != Some(t) {
                                wrong.push(format!("{} at {:?}: {:?} gives {:?}, not {:#x}", name, sample, w, value, t));
                            }
                        }
                    }
                }
            }
        });
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn distributed_and_regrouped_products_give_zero() {
        let (b, cases) = distributed_products();
        let loose = zero_cases(&b, &cases, true);
        assert!(loose.is_empty(), "each is 0 for every u, v and w: {:?}", loose);
    }

    fn repeated_shifts() -> (Build, Vec<(&'static str, ValueId)>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
        let byte = loaded_word(&mut b, &k, e, 8, MemSize::U8);
        let thirty_one = b.constant(e, Ty::I32, 31);
        let s = b.int(e, IntOp::And, byte, thirty_one);
        let mut cases = Vec::new();
        for (name, op) in [("(u >> s) - (u >> s)", IntOp::LShr), ("(u >>> s) - (u >>> s)", IntOp::AShr)] {
            let first = b.int(e, op, u, s);
            let second = b.int(e, op, u, s);
            cases.push((name, b.int(e, IntOp::Sub, first, second)));
        }
        let wide_u = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, u));
        let wide_v = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, v));
        let wide_s = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, s));
        let uv = b.int(e, IntOp::Mul, wide_u, wide_v);
        let vu = b.int(e, IntOp::Mul, wide_v, wide_u);
        let products = b.int(e, IntOp::Sub, uv, vu);
        cases.push(("low(zext(u) * zext(v) - zext(v) * zext(u))", b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, products))));
        let first = b.int(e, IntOp::Shl, wide_u, wide_s);
        let second = b.int(e, IntOp::Shl, wide_u, wide_s);
        let shifts = b.int(e, IntOp::Sub, first, second);
        cases.push(("low((zext(u) << s) - (zext(u) << s))", b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, shifts))));
        (b, cases)
    }

    #[test]
    fn repeated_variable_and_wide_shifts_and_products_hold_zero_differences() {
        let (b, cases) = repeated_shifts();
        let missed = zero_cases(&b, &cases, false);
        assert!(missed.is_empty(), "{:?}", missed);
    }

    #[test]
    fn repeated_variable_and_wide_shifts_and_products_give_zero_differences() {
        let (b, cases) = repeated_shifts();
        let loose = zero_cases(&b, &cases, true);
        assert!(loose.is_empty(), "each is the same value twice: {:?}", loose);
    }

    fn products_after_two_loops() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let table = k.buffer(&mut b, e, 8);
        let zero = b.constant(e, Ty::I32, 0);
        let (first, f) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        let (middle, m) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
        let (second, g) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
        let (after, a) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
        let (last, l) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32]);
        b.br(e, first, vec![k.exec, table, zero]);
        let yes = b.constant(first, Ty::I1, 1);
        let x = b.load(first, Space::Global, MemSize::B32, f[1], yes);
        let one = b.constant(first, Ty::I32, 1);
        let next = b.int(first, IntOp::Add, f[2], one);
        let two = b.constant(first, Ty::I32, 2);
        let again = b.cmp(first, IntPred::Ult, next, two);
        b.cond_br(first, again, (first, vec![f[0], f[1], next]), (middle, vec![f[0], f[1], x]));
        let zero = b.constant(middle, Ty::I32, 0);
        b.br(middle, second, vec![m[0], m[1], m[2], zero]);
        let four = b.constant(second, Ty::I64, 4);
        let at = b.int(second, IntOp::Add, g[1], four);
        let yes = b.constant(second, Ty::I1, 1);
        let y = b.load(second, Space::Global, MemSize::B32, at, yes);
        let one = b.constant(second, Ty::I32, 1);
        let next = b.int(second, IntOp::Add, g[3], one);
        let two = b.constant(second, Ty::I32, 2);
        let again = b.cmp(second, IntPred::Ult, next, two);
        b.cond_br(second, again, (second, vec![g[0], g[1], g[2], next]), (after, vec![g[0], g[1], g[2], y]));
        let product = b.int(after, IntOp::Mul, a[2], a[3]);
        b.br(after, last, vec![a[0], a[2], a[3], product]);
        let again = b.int(last, IntOp::Mul, l[1], l[2]);
        let difference = b.int(last, IntOp::Sub, l[3], again);
        (b, difference)
    }

    #[test]
    fn products_of_words_from_two_loops_in_two_blocks_hold_their_difference() {
        let (b, difference) = products_after_two_loops();
        let held = addresses(&b, &two_words(), |a| {
            let form = a.value(difference, 0, None).0.form;
            representable(&a.unknowns, &form, 0)
        });
        assert!(held);
    }

    #[test]
    fn products_of_words_from_two_loops_in_two_blocks_are_equal() {
        let (b, difference) = products_after_two_loops();
        let form = addresses(&b, &two_words(), |a| a.value(difference, 0, None).0.form);
        assert_eq!(form, Form::constant(0), "both are x * y of the same x from the first loop and y from the second");
    }

    fn chosen_after_bound(entered: bool) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let four = b.constant(e, Ty::I32, 4);
        let below = b.cmp(e, IntPred::Ult, u, four);
        let (then, t) = b.block(&[Ty::I1, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32]);
        b.cond_br(e, below, (then, vec![k.exec, u]), (other, vec![k.exec, u]));
        let (block, p) = if entered { (then, t) } else { (other, o) };
        let eight = b.constant(block, Ty::I32, 8);
        let test = b.cmp(block, IntPred::Ult, p[1], eight);
        let seven = b.constant(block, Ty::I32, 7);
        let nine = b.constant(block, Ty::I32, 9);
        let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
        (b, chosen)
    }

    #[test]
    fn selects_after_a_bound_hold_the_arms_the_bound_leaves() {
        let missed: Vec<(bool, u32)> = [(true, 7u32), (false, 7), (false, 9)]
            .iter()
            .copied()
            .filter(|&(entered, truth)| {
                let (b, chosen) = chosen_after_bound(entered);
                !addresses(&b, &two_words(), |a| {
                    let form = a.value(chosen, 0, None).0.form;
                    representable(&a.unknowns, &form, truth)
                })
            })
            .collect();
        assert!(missed.is_empty(), "u below 4 gives 7; u of 4 or more gives 7 below 8 and 9 above: {:?}", missed);
    }

    #[test]
    fn selects_after_a_bound_take_the_arm_the_bound_decides() {
        let (b, chosen) = chosen_after_bound(true);
        let form = addresses(&b, &two_words(), |a| a.value(chosen, 0, None).0.form);
        assert_eq!(form, Form::constant(7), "the block runs only when u is below 4, so u is below 8");
    }

    type WordCases = Vec<(&'static str, ValueId, Box<dyn Fn(u32, u32) -> u32>)>;

    fn operations_after_a_bound(entered: bool) -> (Build, WordCases, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let sixteen = b.constant(e, Ty::I32, 16);
        let small = b.cmp(e, IntPred::Ult, u, sixteen);
        let (then, t) = b.block(&[Ty::I1, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32]);
        b.cond_br(e, small, (then, vec![k.exec, u]), (other, vec![k.exec, u]));
        let (block, p) = if entered { (then, t) } else { (other, o) };
        let mut cases: WordCases = Vec::new();
        let ff = b.constant(block, Ty::I32, 0xff);
        let x = b.int(block, IntOp::And, p[1], ff);
        cases.push(("u & 0xff", x, Box::new(|u, _| u & 0xff)));
        let eight = b.constant(block, Ty::I32, 8);
        let x = b.int(block, IntOp::LShr, p[1], eight);
        cases.push(("u >> 8", x, Box::new(|u, _| u >> 8)));
        let wide = b.core(block, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, p[1]));
        let near = b.constant(block, Ty::I64, 0xffff_fff0);
        let sum = b.int(block, IntOp::Add, wide, near);
        let x = b.core(block, Ty::I32, Op::UnpackHi(sum));
        cases.push(("high(u + 0xfffffff0)", x, Box::new(|u, _| ((u as u64 + 0xffff_fff0) >> 32) as u32)));
        (b, cases, p[1])
    }

    #[test]
    fn operations_after_a_bound_hold_every_value_the_bound_leaves() {
        let mut wrong = Vec::new();
        for entered in [true, false] {
            let (b, cases, _) = operations_after_a_bound(entered);
            addresses(&b, &two_words(), |a| {
                let forms: Vec<Form> = cases.iter().map(|&(_, x, _)| a.value(x, 0, None).0.form).collect();
                for u in [0u32, 1, 15, 16, 17, 255, 256, 0x1234, u32::MAX - 16, u32::MAX] {
                    if (u < 16) != entered {
                        continue;
                    }
                    for (i, (name, _, truth)) in cases.iter().enumerate() {
                        if !representable(&a.unknowns, &forms[i], truth(u, 0)) {
                            wrong.push(format!("{} in {} at {}: {:?}", name, entered, u, forms[i]));
                        }
                    }
                }
            });
        }
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn operations_after_a_bound_settle_what_the_bound_decides() {
        let (b, cases, u) = operations_after_a_bound(true);
        let loose = addresses(&b, &two_words(), |a| {
            let word = a.value(u, 0, None).0.form;
            let expected = [word, Form::constant(0), Form::constant(0)];
            cases
                .iter()
                .zip(expected)
                .filter_map(|(&(name, x, _), want)| {
                    let form = a.value(x, 0, None).0.form;
                    (form != want).then(|| format!("{}: {:?}, not {:?}", name, form, want))
                })
                .collect::<Vec<_>>()
        });
        assert!(loose.is_empty(), "the block runs only when u is below 16: {:?}", loose);
    }

    fn selects_after_an_order(entered: bool) -> (Build, WordCases) {
        use IntPred::*;
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
        let below = b.cmp(e, Ult, u, v);
        let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        b.cond_br(e, below, (then, vec![k.exec, u, v]), (other, vec![k.exec, u, v]));
        let (block, p) = if entered { (then, t) } else { (other, o) };
        let (x, y) = (p[1], p[2]);
        let seven = b.constant(block, Ty::I32, 7);
        let nine = b.constant(block, Ty::I32, 9);
        let mut cases: WordCases = Vec::new();
        for (name, pred, flipped) in [
            ("v > u", Ugt, true),
            ("u >= v", Uge, false),
            ("u != v", Ne, false),
            ("v <= u", Ule, true),
            ("u <= v", Ule, false),
            ("u == v", Eq, false),
            ("u < v signed", Slt, false),
        ] {
            let (first, second) = if flipped { (y, x) } else { (x, y) };
            let test = b.cmp(block, pred, first, second);
            let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
            let holds = move |u: u32, v: u32| {
                let (a, c) = if flipped { (v, u) } else { (u, v) };
                if compare(pred, a, c) { 7 } else { 9 }
            };
            cases.push((name, chosen, Box::new(holds)));
        }
        (b, cases)
    }

    #[test]
    fn selects_after_an_order_hold_the_arms_the_order_leaves() {
        let mut wrong = Vec::new();
        for entered in [true, false] {
            let (b, cases) = selects_after_an_order(entered);
            addresses(&b, &two_words(), |a| {
                let forms: Vec<Form> = cases.iter().map(|&(_, x, _)| a.value(x, 0, None).0.form).collect();
                let samples = [(0u32, 1u32), (1, 0), (5, 5), (0, 0x8000_0000), (0x8000_0000, 0), (u32::MAX, 3), (3, u32::MAX), (7, 9)];
                for (u, v) in samples {
                    if (u < v) != entered {
                        continue;
                    }
                    for (i, (name, _, truth)) in cases.iter().enumerate() {
                        if !representable(&a.unknowns, &forms[i], truth(u, v)) {
                            wrong.push(format!("{} in {} at ({}, {}): {:?}", name, entered, u, v, forms[i]));
                        }
                    }
                }
            });
        }
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    fn outcome(x: u32, y: u32) -> u8 {
        if x == y {
            return 1;
        }
        match (x < y, (x as i32) < (y as i32)) {
            (true, true) => 2,
            (true, false) => 4,
            (false, true) => 8,
            (false, false) => 16,
        }
    }

    const OUTCOMES: [(u32, u32); 5] = [(5, 5), (1, 2), (1, 0x8000_0000), (0x8000_0000, 1), (2, 1)];

    #[test]
    fn order_outcomes_hold_exactly_the_pairs_each_predicate_accepts() {
        let mut r = Random::new(137);
        let mut samples: Vec<(u32, u32)> = OUTCOMES.to_vec();
        let points = [0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, u32::MAX - 1, u32::MAX];
        for &x in &points {
            for &y in &points {
                samples.push((x, y));
            }
        }
        samples.extend((0..200).map(|_| (r.next() as u32, r.next() as u32)));
        let mut wrong = Vec::new();
        for pred in PREDICATES {
            for &(x, y) in &samples {
                if (outcomes(pred) & outcome(x, y) != 0) != compare(pred, x, y) {
                    wrong.push(format!("{:?} at ({:#x}, {:#x})", pred, x, y));
                }
                if (mirrored(outcomes(pred)) & outcome(y, x) != 0) != compare(pred, x, y) {
                    wrong.push(format!("{:?} swapped at ({:#x}, {:#x})", pred, x, y));
                }
            }
        }
        let seen: u8 = samples.iter().fold(0, |m, &(x, y)| m | outcome(x, y));
        assert_eq!(seen, 31, "every outcome occurs");
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn orders_conjoin_exactly_and_disjoin_exactly_on_one_pair() {
        let words = [Form::unknown(0), Form::unknown(1), Form::unknown(2)];
        let holds = |orders: &[(Form, Form, u8)], values: [u32; 3]| {
            orders.iter().all(|(x, y, mask)| {
                let at = |f: &Form| values[words.iter().position(|w| w == f).unwrap()];
                mask & outcome(at(x), at(y)) != 0
            })
        };
        let mut r = Random::new(139);
        let mut wrong = Vec::new();
        for _ in 0..400 {
            let mut build = |r: &mut Random| {
                let mut orders = Vec::new();
                for _ in 0..r.below(3) {
                    let (i, j) = (r.below(3) as usize, r.below(3) as usize);
                    let pred = PREDICATES[r.below(10) as usize];
                    orders = orders_conjoined(orders, order_limit(&words[i], &words[j], pred));
                }
                orders
            };
            let (a, b) = (build(&mut r), build(&mut r));
            let both = orders_conjoined(a.clone(), b.clone());
            let either = orders_disjoined(a.clone(), b.clone());
            let one_pair = a.len() == 1 && b.len() == 1 && (&a[0].0, &a[0].1) == (&b[0].0, &b[0].1);
            for x in OUTCOMES.iter().flat_map(|&(p, q)| [p, q]) {
                for y in OUTCOMES.iter().flat_map(|&(p, q)| [p, q]) {
                    for z in [0u32, 1, 2, 5, 0x8000_0000] {
                        let values = [x, y, z];
                        let (ha, hb) = (holds(&a, values), holds(&b, values));
                        if holds(&both, values) != (ha && hb) {
                            wrong.push(format!("{:?} and {:?} at {:?}", a, b, values));
                        }
                        if (ha || hb) && !holds(&either, values) {
                            wrong.push(format!("{:?} or {:?} misses {:?}", a, b, values));
                        }
                        if one_pair && holds(&either, values) != (ha || hb) {
                            wrong.push(format!("{:?} or {:?} loose at {:?}", a, b, values));
                        }
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(6)]);
    }

    fn branches_on_two_words() -> (Build, Vec<(&'static str, Box<dyn Fn(u32, u32) -> bool>, Vec<(IntPred, bool, ValueId)>)>) {
        use IntPred::*;
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
        let fork = |b: &mut Build, at: BlockId, cond: ValueId, exec: ValueId| {
            let (yes, y) = b.block(&[Ty::I1]);
            let (no, n) = b.block(&[Ty::I1]);
            b.cond_br(at, cond, (yes, vec![exec]), (no, vec![exec]));
            ((yes, y[0]), (no, n[0]))
        };
        let c1 = b.cmp(e, Ult, u, v);
        let ((b1, e1), (x1, f1)) = fork(&mut b, e, c1, k.exec);
        let c2 = b.cmp(b1, Sgt, v, u);
        let ((b2, e2), (x2, f2)) = fork(&mut b, b1, c2, e1);
        let equal = b.cmp(x1, Eq, u, v);
        let above = b.cmp(x1, Sgt, u, v);
        let c3 = b.int(x1, IntOp::Or, equal, above);
        let ((b3, _), (x3, _)) = fork(&mut b, x1, c3, f1);
        let (j, _) = b.block(&[Ty::I1]);
        let (z, _) = b.block(&[Ty::I1]);
        let c4 = b.cmp(b2, Ne, u, v);
        b.cond_br(b2, c4, (j, vec![e2]), (z, vec![e2]));
        let c5 = b.cmp(x2, Uge, v, u);
        b.cond_br(x2, c5, (j, vec![f2]), (z, vec![f2]));
        let c1 = |u: u32, v: u32| u < v;
        let c2 = |u: u32, v: u32| (v as i32) > (u as i32);
        let c3 = |u: u32, v: u32| u == v || (u as i32) > (v as i32);
        let c4 = |u: u32, v: u32| u != v;
        let c5 = |u: u32, v: u32| v >= u;
        let reach: Vec<(&'static str, BlockId, Box<dyn Fn(u32, u32) -> bool>)> = vec![
            ("b1", b1, Box::new(move |u, v| c1(u, v))),
            ("x1", x1, Box::new(move |u, v| !c1(u, v))),
            ("b2", b2, Box::new(move |u, v| c1(u, v) && c2(u, v))),
            ("x2", x2, Box::new(move |u, v| c1(u, v) && !c2(u, v))),
            ("b3", b3, Box::new(move |u, v| !c1(u, v) && c3(u, v))),
            ("x3", x3, Box::new(move |u, v| !c1(u, v) && !c3(u, v))),
            ("j", j, Box::new(move |u, v| c1(u, v) && if c2(u, v) { c4(u, v) } else { c5(u, v) })),
            ("z", z, Box::new(move |u, v| c1(u, v) && if c2(u, v) { !c4(u, v) } else { !c5(u, v) })),
        ];
        let mut blocks = Vec::new();
        for (name, block, reaches) in reach {
            let mut queries = Vec::new();
            for pred in PREDICATES {
                queries.push((pred, false, b.cmp(block, pred, u, v)));
                queries.push((pred, true, b.cmp(block, pred, v, u)));
            }
            blocks.push((name, reaches, queries));
        }
        (b, blocks)
    }

    #[test]
    fn comparisons_under_order_branches_decide_exactly_what_the_orders_settle() {
        let (b, blocks) = branches_on_two_words();
        let mut wrong = Vec::new();
        addresses(&b, &two_words(), |a| {
            for (name, reaches, queries) in &blocks {
                for &(pred, flipped, q) in queries {
                    let mut seen = [false; 2];
                    for &(u, v) in &OUTCOMES {
                        if reaches(u, v) {
                            let holds = if flipped { compare(pred, v, u) } else { compare(pred, u, v) };
                            seen[holds as usize] = true;
                        }
                    }
                    let truth = match seen {
                        [false, false] => continue,
                        [true, false] => Some(false),
                        [false, true] => Some(true),
                        _ => None,
                    };
                    let got = a.bit(q, 0, None).0;
                    if got != truth {
                        wrong.push(format!("{}: {:?} flipped {}: {:?}, truth {:?}", name, pred, flipped, got, truth));
                    }
                }
            }
        });
        assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
    }

    #[test]
    fn selects_after_an_order_take_the_arm_the_order_decides() {
        let (b, cases) = selects_after_an_order(true);
        let loose = addresses(&b, &two_words(), |a| {
            cases
                .iter()
                .filter(|&&(name, _, _)| name != "u < v signed")
                .filter_map(|&(name, x, ref truth)| {
                    let form = a.value(x, 0, None).0.form;
                    let want = Form::constant(truth(0, 1));
                    (form != want).then(|| format!("{}: {:?}, not {:?}", name, form, want))
                })
                .collect::<Vec<_>>()
        });
        assert!(loose.is_empty(), "the block runs only when u < v: {:?}", loose);
    }

    const PREDICATES: [IntPred; 10] = [
        IntPred::Eq,
        IntPred::Ne,
        IntPred::Ult,
        IntPred::Ule,
        IntPred::Ugt,
        IntPred::Uge,
        IntPred::Slt,
        IntPred::Sle,
        IntPred::Sgt,
        IntPred::Sge,
    ];

    fn within(set: &[(u64, u64)], x: u32) -> bool {
        set.iter().any(|&(low, high)| low <= x as u64 && x as u64 <= high)
    }

    fn ends_of(set: &[(u64, u64)]) -> Vec<u32> {
        set.iter()
            .flat_map(|&(low, high)| [low as u32, (low as u32).wrapping_sub(1), high as u32, (high as u32).wrapping_add(1)])
            .collect()
    }

    fn well_formed(set: &[(u64, u64)]) -> bool {
        set.iter().all(|&(low, high)| low <= high && high <= u32::MAX as u64) && set.windows(2).all(|w| w[0].1 + 1 < w[1].0)
    }

    fn random_pieces(r: &mut Random) -> Vec<(u64, u64)> {
        let points = [0u32, 1, 5, 100, 0x7fff_fff0, 0x7fff_ffff, 0x8000_0000, 0x8000_0010, u32::MAX - 3, u32::MAX];
        let mut set = Vec::new();
        for _ in 0..r.below(3) {
            let mut pick = || if r.below(2) == 0 { points[r.below(points.len() as u64) as usize] } else { r.next() as u32 };
            let (x, y) = (pick(), pick());
            set.push((x.min(y) as u64, x.max(y) as u64));
        }
        set
    }

    #[test]
    fn satisfying_holds_exactly_the_words_each_predicate_accepts() {
        let mut r = Random::new(97);
        let mut constants = vec![0u32, 1, 2, 7, 0x7fff_fffe, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, u32::MAX - 1, u32::MAX];
        constants.extend((0..20).map(|_| r.next() as u32));
        let mut wrong = Vec::new();
        for pred in PREDICATES {
            for &k in &constants {
                let set = satisfying(pred, k);
                if !set.iter().all(|&(low, high)| low <= high && high <= u32::MAX as u64) || !set.windows(2).all(|w| w[0].1 < w[1].0) {
                    wrong.push(format!("{:?} {:#x}: {:?}", pred, k, set));
                }
                let mut words = constants.clone();
                words.extend([k.wrapping_sub(1), k, k.wrapping_add(1)]);
                words.extend(ends_of(&set));
                for x in words {
                    if within(&set, x) != compare(pred, x, k) {
                        wrong.push(format!("{:?} {:#x} at {:#x}: {:?}", pred, k, x, set));
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn piece_operations_hold_exactly_the_words_their_pieces_hold() {
        let mut r = Random::new(101);
        let mut wrong = Vec::new();
        for _ in 0..600 {
            let (a, b) = (random_pieces(&mut r), random_pieces(&mut r));
            let k = match r.below(3) {
                0 => r.below(8) as u32,
                1 => (r.below(8) as u32).wrapping_neg(),
                _ => r.next() as u32,
            };
            let shifted = shifted_pieces(&a, k);
            let common = intersected(&a, &b);
            let merged = normalized([a.clone(), b.clone()].concat());
            for set in [&shifted, &common, &merged] {
                if !well_formed(set) {
                    wrong.push(format!("{:?} and {:?}: not normal {:?}", a, b, set));
                }
            }
            let mut words: Vec<u32> = [ends_of(&a), ends_of(&b), ends_of(&common), ends_of(&merged)].concat();
            words.extend(ends_of(&shifted).iter().map(|y| y.wrapping_sub(k)));
            words.extend([0, 0x8000_0000, u32::MAX]);
            for x in words {
                if within(&shifted, x.wrapping_add(k)) != within(&a, x) {
                    wrong.push(format!("{:?} + {:#x} at {:#x}: {:?}", a, k, x, shifted));
                }
                if within(&common, x) != (within(&a, x) && within(&b, x)) {
                    wrong.push(format!("{:?} and {:?} at {:#x}: {:?}", a, b, x, common));
                }
                if within(&merged, x) != (within(&a, x) || within(&b, x)) {
                    wrong.push(format!("{:?} or {:?} at {:#x}: {:?}", a, b, x, merged));
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn limits_of_word_classes_conjoin_exactly_and_disjoin_exactly_within_one_class() {
        let (u, v) = (Form::unknown(0), Form::unknown(1));
        let holds = |limits: &[(Form, Vec<(u64, u64)>)], x: u32, y: u32| {
            limits.iter().all(|(c, set)| within(set, if *c == u { x } else { y }))
        };
        let mut r = Random::new(103);
        let mut wrong = Vec::new();
        for _ in 0..400 {
            let mut build = |r: &mut Random| {
                let mut limits = Vec::new();
                for class in [&u, &v] {
                    if r.below(3) != 0 {
                        let c = if r.below(2) == 0 { r.below(8) as u32 } else { r.next() as u32 };
                        let values = random_pieces(r);
                        let limit = class_limit(&class.add(&Form::constant(c)), &values);
                        if limit.0 != *class || !well_formed(&limit.1) {
                            wrong.push(format!("{:?} + {:#x} in {:?}: {:?}", class, c, values, limit));
                        }
                        for x in [ends_of(&values).iter().map(|y| y.wrapping_sub(c)).collect(), ends_of(&limit.1)].concat() {
                            if within(&limit.1, x) != within(&values, x.wrapping_add(c)) {
                                wrong.push(format!("{:?} + {:#x} in {:?} at {:#x}: {:?}", class, c, values, x, limit));
                            }
                        }
                        limits.push(limit);
                    }
                }
                limits
            };
            let (a, b) = (build(&mut r), build(&mut r));
            let both = conjoined(a.clone(), b.clone());
            let either = disjoined(a.clone(), b.clone());
            let one_class = a.len() == 1 && b.len() == 1 && a[0].0 == b[0].0;
            let words = |class: &Form| -> Vec<u32> {
                let mut out = vec![0, 0x8000_0000, u32::MAX];
                for limits in [&a, &b, &both, &either] {
                    for (c, set) in limits.iter() {
                        if c == class {
                            out.extend(ends_of(set));
                        }
                    }
                }
                out
            };
            let (xs, ys) = (words(&u), words(&v));
            for &x in &xs {
                for &y in &ys {
                    let (ha, hb) = (holds(&a, x, y), holds(&b, x, y));
                    if holds(&both, x, y) != (ha && hb) {
                        wrong.push(format!("{:?} and {:?} at ({:#x}, {:#x}): {:?}", a, b, x, y, both));
                    }
                    if (ha || hb) && !holds(&either, x, y) {
                        wrong.push(format!("{:?} or {:?} at ({:#x}, {:#x}) misses: {:?}", a, b, x, y, either));
                    }
                    if one_class && holds(&either, x, y) != (ha || hb) {
                        wrong.push(format!("{:?} or {:?} at ({:#x}, {:#x}) loose: {:?}", a, b, x, y, either));
                    }
                }
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(10)]);
    }

    struct BranchesOnOneWord {
        b: Build,
        conditions: Vec<(IntPred, u32, u32)>,
        blocks: Vec<(&'static str, Box<dyn Fn(u32) -> bool>, Vec<(IntPred, bool, u32, u32, ValueId)>)>,
    }

    fn branches_on_one_word() -> BranchesOnOneWord {
        use IntPred::*;
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let against = |b: &mut Build, at: BlockId, pred: IntPred, x: ValueId, c: u32| {
            let c = b.constant(at, Ty::I32, c as u64);
            b.cmp(at, pred, x, c)
        };
        let fork = |b: &mut Build, at: BlockId, cond: ValueId, exec: ValueId| {
            let (yes, y) = b.block(&[Ty::I1]);
            let (no, n) = b.block(&[Ty::I1]);
            b.cond_br(at, cond, (yes, vec![exec]), (no, vec![exec]));
            ((yes, y[0]), (no, n[0]))
        };
        let c1 = against(&mut b, e, Ult, u, 1000);
        let ((b1, e1), (x1, f1)) = fork(&mut b, e, c1, k.exec);
        let five = b.constant(b1, Ty::I32, 5);
        let u5 = b.int(b1, IntOp::Add, u, five);
        let above = against(&mut b, b1, Ugt, u5, 20);
        let big = against(&mut b, b1, Uge, u, 500);
        let one = b.constant(b1, Ty::I1, 1);
        let small = b.int(b1, IntOp::Xor, big, one);
        let c2 = b.int(b1, IntOp::And, above, small);
        let ((b2, e2), (x2, _)) = fork(&mut b, b1, c2, e1);
        let three_hundred = b.constant(b2, Ty::I32, 300);
        let m = b.int(b2, IntOp::Sub, u, three_hundred);
        let c3 = against(&mut b, b2, Sge, m, 0);
        let ((b3, e3), (x3, _)) = fork(&mut b, b2, c3, e2);
        let c4 = against(&mut b, b3, Ne, u, 400);
        let ((b4, e4), (x4, _)) = fork(&mut b, b3, c4, e3);
        let low = against(&mut b, b4, Ult, u, 320);
        let high = against(&mut b, b4, Ugt, u, 450);
        let c5 = b.int(b4, IntOp::Or, low, high);
        let ((b5, e5), (x5, f5)) = fork(&mut b, b4, c5, e4);
        let (j5, _) = b.block(&[Ty::I1]);
        let (z5, _) = b.block(&[Ty::I1]);
        let (y5, _) = b.block(&[Ty::I1]);
        let c8 = against(&mut b, b5, Ult, u, 310);
        b.cond_br(b5, c8, (j5, vec![e5]), (z5, vec![e5]));
        let c7 = against(&mut b, x5, Uge, u, 440);
        b.cond_br(x5, c7, (j5, vec![f5]), (y5, vec![f5]));
        let bound = b.constant(x1, Ty::I32, 0x9000_0000);
        let c6 = b.cmp(x1, Ugt, bound, u);
        let ((b6, _), (x6, _)) = fork(&mut b, x1, c6, f1);
        let conditions = vec![
            (Ult, 0, 1000),
            (Ugt, 5, 20),
            (Uge, 0, 500),
            (Sge, 300u32.wrapping_neg(), 0),
            (Ne, 0, 400),
            (Ult, 0, 320),
            (Ugt, 0, 450),
            (Ult, 0, 0x9000_0000),
            (Ult, 0, 310),
            (Uge, 0, 440),
        ];
        let c1 = |u: u32| u < 1000;
        let c2 = |u: u32| u.wrapping_add(5) > 20 && u < 500;
        let c3 = |u: u32| u.wrapping_sub(300) as i32 >= 0;
        let c4 = |u: u32| u != 400;
        let c5 = |u: u32| !(320..=450).contains(&u);
        let c6 = |u: u32| u < 0x9000_0000;
        let c7 = |u: u32| u >= 440;
        let c8 = |u: u32| u < 310;
        let reach: Vec<(&'static str, BlockId, Box<dyn Fn(u32) -> bool>)> = vec![
            ("b1", b1, Box::new(move |u| c1(u))),
            ("x1", x1, Box::new(move |u| !c1(u))),
            ("b2", b2, Box::new(move |u| c1(u) && c2(u))),
            ("x2", x2, Box::new(move |u| c1(u) && !c2(u))),
            ("b3", b3, Box::new(move |u| c1(u) && c2(u) && c3(u))),
            ("x3", x3, Box::new(move |u| c1(u) && c2(u) && !c3(u))),
            ("b4", b4, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u))),
            ("x4", x4, Box::new(move |u| c1(u) && c2(u) && c3(u) && !c4(u))),
            ("b5", b5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && c5(u))),
            ("x5", x5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && !c5(u))),
            ("j5", j5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && if c5(u) { c8(u) } else { c7(u) })),
            ("z5", z5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && c5(u) && !c8(u))),
            ("y5", y5, Box::new(move |u| c1(u) && c2(u) && c3(u) && c4(u) && !c5(u) && !c7(u))),
            ("b6", b6, Box::new(move |u| !c1(u) && c6(u))),
            ("x6", x6, Box::new(move |u| !c1(u) && !c6(u))),
        ];
        let offsets = [0u32, 1, u32::MAX, 0x8000_0000, 700];
        let constants = [0u32, 15, 16, 300, 319, 400, 401, 451, 499, 500, 999, 1000, 0x7fff_ffff, 0x8000_0000, 0x9000_0000, u32::MAX];
        let mut blocks = Vec::new();
        for (name, block, reaches) in reach {
            let mut queries = Vec::new();
            for &offset in &offsets {
                let x = if offset == 0 {
                    u
                } else {
                    let c = b.constant(block, Ty::I32, offset as u64);
                    b.int(block, IntOp::Add, u, c)
                };
                for &c in &constants {
                    let kc = b.constant(block, Ty::I32, c as u64);
                    for pred in PREDICATES {
                        queries.push((pred, false, offset, c, b.cmp(block, pred, x, kc)));
                        queries.push((pred, true, offset, c, b.cmp(block, pred, kc, x)));
                    }
                }
            }
            blocks.push((name, reaches, queries));
        }
        BranchesOnOneWord { b, conditions, blocks }
    }

    fn settled_by_branches(branches: &BranchesOnOneWord, pred: IntPred, flipped: bool, offset: u32, c: u32, reaches: &dyn Fn(u32) -> bool) -> Option<Option<bool>> {
        let holds = |u: u32| {
            let x = u.wrapping_add(offset);
            if flipped {
                compare(pred, c, x)
            } else {
                compare(pred, x, c)
            }
        };
        let mut starts = vec![0u32];
        for &(_, shift, k) in branches.conditions.iter().chain([(pred, offset, c)].iter()) {
            starts.extend([k.wrapping_sub(shift), k.wrapping_sub(shift).wrapping_add(1), shift.wrapping_neg(), 0x8000_0000u32.wrapping_sub(shift)]);
        }
        let mut seen = [false; 2];
        for u in starts {
            if reaches(u) {
                seen[holds(u) as usize] = true;
            }
        }
        match seen {
            [false, false] => None,
            [true, false] => Some(Some(false)),
            [false, true] => Some(Some(true)),
            _ => Some(None),
        }
    }

    #[test]
    fn comparisons_under_branches_decide_only_what_every_reaching_word_gives() {
        let branches = branches_on_one_word();
        let mut wrong = Vec::new();
        addresses(&branches.b, &two_words(), |a| {
            for (name, reaches, queries) in &branches.blocks {
                for &(pred, flipped, offset, c, q) in queries {
                    let Some(truth) = settled_by_branches(&branches, pred, flipped, offset, c, reaches.as_ref()) else {
                        continue;
                    };
                    let got = a.bit(q, 0, None).0;
                    if got.is_some() && got != truth {
                        wrong.push(format!("{}: {:?} flipped {} (u + {:#x}) against {:#x}: {:?}, truth {:?}", name, pred, flipped, offset, c, got, truth));
                    }
                }
            }
        });
        assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
    }

    #[test]
    fn comparisons_under_branches_decide_what_the_conditions_on_one_word_settle() {
        let branches = branches_on_one_word();
        let mut loose = Vec::new();
        addresses(&branches.b, &two_words(), |a| {
            for (name, reaches, queries) in &branches.blocks {
                for &(pred, flipped, offset, c, q) in queries {
                    let Some(truth) = settled_by_branches(&branches, pred, flipped, offset, c, reaches.as_ref()) else {
                        continue;
                    };
                    let got = a.bit(q, 0, None).0;
                    if truth.is_some() && got != truth {
                        loose.push(format!("{}: {:?} flipped {} (u + {:#x}) against {:#x}: {:?}, truth {:?}", name, pred, flipped, offset, c, got, truth));
                    }
                }
            }
        });
        assert!(loose.is_empty(), "{} loose: {:?}", loose.len(), &loose[..loose.len().min(10)]);
    }

    fn chosen_after_branch_on_a_source(entered: bool) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let is_zero = b.cmp(e, IntPred::Eq, u, zero);
        let one = b.constant(e, Ty::I32, 1);
        let w = b.int(e, IntOp::Add, u, one);
        let (then, t) = b.block(&[Ty::I1, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32]);
        b.cond_br(e, is_zero, (then, vec![k.exec, w]), (other, vec![k.exec, w]));
        let (block, p) = if entered { (then, t) } else { (other, o) };
        let one = b.constant(block, Ty::I32, 1);
        let test = b.cmp(block, IntPred::Eq, p[1], one);
        let seven = b.constant(block, Ty::I32, 7);
        let nine = b.constant(block, Ty::I32, 9);
        let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
        (b, chosen)
    }

    #[test]
    fn selects_on_a_word_derived_from_the_branch_word_hold_the_arm_the_branch_leaves() {
        let missed: Vec<bool> = [true, false]
            .iter()
            .copied()
            .filter(|&entered| {
                let (b, chosen) = chosen_after_branch_on_a_source(entered);
                let truth = if entered { 7 } else { 9 };
                !addresses(&b, &two_words(), |a| {
                    let form = a.value(chosen, 0, None).0.form;
                    representable(&a.unknowns, &form, truth)
                })
            })
            .collect();
        assert!(missed.is_empty(), "{:?}", missed);
    }

    #[test]
    fn selects_on_a_word_derived_from_the_branch_word_take_the_arm_the_branch_decides() {
        let (b, chosen) = chosen_after_branch_on_a_source(true);
        let form = addresses(&b, &two_words(), |a| a.value(chosen, 0, None).0.form);
        assert_eq!(form, Form::constant(7), "the block runs only when u is 0, so u + 1 is 1");
    }

    fn chosen_in_a_loop_entered_on_zero() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let is_zero = b.cmp(e, IntPred::Eq, u, zero);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.cond_br(e, is_zero, (body, vec![k.exec, u, zero]), (exit, vec![k.exec]));
        let zero = b.constant(body, Ty::I32, 0);
        let test = b.cmp(body, IntPred::Eq, p[1], zero);
        let seven = b.constant(body, Ty::I32, 7);
        let nine = b.constant(body, Ty::I32, 9);
        let chosen = b.core(body, Ty::I32, Op::Select(test, seven, nine));
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[2], one);
        let three = b.constant(body, Ty::I32, 3);
        let again = b.cmp(body, IntPred::Ult, next, three);
        b.cond_br(body, again, (body, vec![p[0], p[1], next]), (exit, vec![p[0]]));
        (b, chosen)
    }

    #[test]
    fn selects_in_a_loop_entered_on_zero_hold_the_arm_zero_takes() {
        let (b, chosen) = chosen_in_a_loop_entered_on_zero();
        let held = addresses(&b, &two_words(), |a| {
            let form = a.value(chosen, 0, None).0.form;
            representable(&a.unknowns, &form, 7)
        });
        assert!(held);
    }

    #[test]
    fn selects_in_a_loop_entered_on_zero_take_the_arm_zero_takes() {
        let (b, chosen) = chosen_in_a_loop_entered_on_zero();
        let form = addresses(&b, &two_words(), |a| a.value(chosen, 0, None).0.form);
        assert_eq!(form, Form::constant(7), "the loop runs only when u is 0 and carries u unchanged");
    }

    fn affine_values(start: u32, trips: usize) -> Vec<u32> {
        std::iter::successors(Some(start), |x| Some(x.wrapping_mul(3).wrapping_add(1))).take(trips).collect()
    }

    #[test]
    fn loop_values_hold_every_value_an_affine_step_carries_over_five_thousand_iterations() {
        let (b, v) = affine_loop(5000);
        let truth = affine_values(1, 5000);
        let missed: Vec<u32> = addresses(&b, &environment(32, &[]), |a| {
            let form = a.value(v, 0, None).0.form;
            truth.iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "{} values missed", missed.len());
    }

    #[test]
    fn loop_values_are_exactly_those_an_affine_step_carries_over_five_thousand_iterations() {
        let (b, v) = affine_loop(5000);
        let mut truth = affine_values(1, 5000);
        truth.sort_unstable();
        truth.dedup();
        let carried = carried_values(&b, v).map(|mut s| {
            s.sort_unstable();
            s
        });
        assert!(carried.as_ref() == Some(&truth), "the parameter runs 1, 4, 13 and on for 5000 iterations; found {:?} values", carried.map(|s| s.len()));
    }

    fn affine_loop_from_a_byte() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let mask = b.constant(e, Ty::I32, 127);
        let start = b.int(e, IntOp::And, u, mask);
        let zero = b.constant(e, Ty::I32, 0);
        let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, body, vec![k.exec, start, zero]);
        let three = b.constant(body, Ty::I32, 3);
        let tripled = b.int(body, IntOp::Mul, p[1], three);
        let one = b.constant(body, Ty::I32, 1);
        let next_value = b.int(body, IntOp::Add, tripled, one);
        let next = b.int(body, IntOp::Add, p[2], one);
        let limit = b.constant(body, Ty::I32, 3);
        let again = b.cmp(body, IntPred::Ult, next, limit);
        b.cond_br(body, again, (body, vec![p[0], next_value, next]), (exit, vec![p[0]]));
        (b, p[1])
    }

    fn affine_values_from_a_byte() -> Vec<u32> {
        let mut values: Vec<u32> = (0..128).flat_map(|s| affine_values(s, 3)).collect();
        values.sort_unstable();
        values.dedup();
        values
    }

    #[test]
    fn loop_values_hold_every_value_an_affine_step_carries_from_a_hundred_and_twenty_eight_starts() {
        let (b, v) = affine_loop_from_a_byte();
        let truth = affine_values_from_a_byte();
        let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
            let form = a.value(v, 0, None).0.form;
            truth.iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "{:?}", missed);
    }

    #[test]
    fn loop_values_are_exactly_those_an_affine_step_carries_from_a_hundred_and_twenty_eight_starts() {
        let (b, v) = affine_loop_from_a_byte();
        let carried = addresses(&b, &two_words(), |a| {
            let form = a.value(v, 0, None).0.form;
            match form.terms.as_slice() {
                [(u, 1)] if form.constant == 0 => a.unknowns[*u as usize].values.as_ref().map(|s| {
                    let mut s = s.to_vec();
                    s.sort_unstable();
                    s
                }),
                _ => None,
            }
        });
        assert!(carried == Some(affine_values_from_a_byte()), "the start is below 128 and the loop runs three times; found {:?} values", carried.map(|s| s.len()));
    }

    fn squared_loop_with_two_back_edges() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let one = b.constant(e, Ty::I32, 1);
        let zero = b.constant(e, Ty::I32, 0);
        let (head, h) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (left, l) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (right, r) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, head, vec![k.exec, one, zero]);
        let squared = b.int(head, IntOp::Mul, h[1], h[1]);
        let one = b.constant(head, Ty::I32, 1);
        let next_value = b.int(head, IntOp::Add, squared, one);
        let next = b.int(head, IntOp::Add, h[2], one);
        let low = b.int(head, IntOp::And, next, one);
        let odd = b.cmp(head, IntPred::Eq, low, one);
        b.cond_br(head, odd, (left, vec![h[0], next_value, next]), (right, vec![h[0], next_value, next]));
        for (block, p) in [(left, &l), (right, &r)] {
            let three = b.constant(block, Ty::I32, 3);
            let again = b.cmp(block, IntPred::Ult, p[2], three);
            b.cond_br(block, again, (head, vec![p[0], p[1], p[2]]), (exit, vec![p[0]]));
        }
        (b, h[1])
    }

    #[test]
    fn loop_values_hold_every_value_a_square_step_carries_along_two_back_edges() {
        let (b, v) = squared_loop_with_two_back_edges();
        let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
            let form = a.value(v, 0, None).0.form;
            [1u32, 2, 5].iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "the parameter runs 1, 2, 5: {:?}", missed);
    }

    #[test]
    fn loop_values_are_exactly_those_a_square_step_carries_along_two_back_edges() {
        let (b, v) = squared_loop_with_two_back_edges();
        assert_eq!(carried_values(&b, v), Some(vec![1, 2, 5]));
    }

    fn sums_of_sums() -> (Build, Vec<(&'static str, ValueId, u32)>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let far = b.constant(e, Ty::I64, 0x1_0000_0000);
        let above = b.int(e, IntOp::Add, buf, far);
        let twice = b.int(e, IntOp::Add, above, far);
        let high = b.core(e, Ty::I32, Op::UnpackHi(twice));
        let back = b.int(e, IntOp::Sub, twice, far);
        let back_high = b.core(e, Ty::I32, Op::UnpackHi(back));
        (b, vec![("high((buf + 2^32) + 2^32)", high, 2), ("high(((buf + 2^32) + 2^32) - 2^32)", back_high, 1)])
    }

    #[test]
    fn high_halves_of_sums_of_sums_hold_their_values() {
        let (b, cases) = sums_of_sums();
        let missed: Vec<&str> = addresses(&b, &two_words(), |a| {
            cases
                .iter()
                .filter(|c| {
                    let form = a.value(c.1, 0, None).0.form;
                    !representable(&a.unknowns, &form, c.2)
                })
                .map(|c| c.0)
                .collect()
        });
        assert!(missed.is_empty(), "{:?}", missed);
    }

    #[test]
    fn high_halves_of_sums_of_sums_give_their_values() {
        let (b, cases) = sums_of_sums();
        let loose: Vec<String> = addresses(&b, &two_words(), |a| {
            cases
                .iter()
                .filter_map(|c| {
                    let form = a.value(c.1, 0, None).0.form;
                    (form.as_constant() != Some(c.2)).then(|| format!("{}: {:?}", c.0, form))
                })
                .collect()
        });
        assert!(loose.is_empty(), "buf is 0x1000: {:?}", loose);
    }

    fn shared_word_by_item() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let four = b.constant(e, Ty::I32, 4);
        let own = b.int(e, IntOp::Mul, k.item, four);
        b.store(e, Space::Lds, MemSize::B32, own, k.item, k.exec);
        let back = b.load(e, Space::Lds, MemSize::B32, own, k.exec);
        (b, back)
    }

    #[test]
    fn lds_words_read_back_by_work_item_hold_the_word_the_item_stored() {
        let (b, back) = shared_word_by_item();
        let held = addresses(&b, &two_words(), |a| {
            let form = a.value(back, 3, None).0.form;
            representable(&a.unknowns, &form, 3)
        });
        assert!(held, "item 3 reads back the 3 it stored");
    }

    #[test]
    fn lds_words_read_back_by_work_item_give_the_word_the_item_stored() {
        let (b, back) = shared_word_by_item();
        let form = addresses(&b, &two_words(), |a| a.value(back, 3, None).0.form);
        assert_eq!(form, Form::constant(3), "word 3 of the LDS only ever holds 3, which only item 3 writes");
    }

    fn squared_loop_with_two_bounds() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let one = b.constant(e, Ty::I32, 1);
        let zero = b.constant(e, Ty::I32, 0);
        let (head, h) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (even, l) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (odd, r) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, head, vec![k.exec, one, zero]);
        let squared = b.int(head, IntOp::Mul, h[1], h[1]);
        let one = b.constant(head, Ty::I32, 1);
        let next_value = b.int(head, IntOp::Add, squared, one);
        let next = b.int(head, IntOp::Add, h[2], one);
        let low = b.int(head, IntOp::And, next, one);
        let is_odd = b.cmp(head, IntPred::Eq, low, one);
        b.cond_br(head, is_odd, (odd, vec![h[0], next_value, next]), (even, vec![h[0], next_value, next]));
        for (block, p, bound) in [(even, &l, 3u64), (odd, &r, 5)] {
            let limit = b.constant(block, Ty::I32, bound);
            let again = b.cmp(block, IntPred::Ult, p[2], limit);
            b.cond_br(block, again, (head, vec![p[0], p[1], p[2]]), (exit, vec![p[0]]));
        }
        (b, h[1])
    }

    #[test]
    fn loop_values_hold_every_value_a_square_step_carries_along_two_back_edges_with_different_bounds() {
        let (b, v) = squared_loop_with_two_bounds();
        let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
            let form = a.value(v, 0, None).0.form;
            [1u32, 2, 5, 26].iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "the odd back edge goes on below 5 and the even one below 3, so the loop runs four times and carries 1, 2, 5, 26: {:?}", missed);
    }

    fn chosen_after_offset_branch(entered: bool) -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let three = b.constant(e, Ty::I32, 3);
        let x = b.int(e, IntOp::Add, u, three);
        let five = b.constant(e, Ty::I32, 5);
        let hit = b.cmp(e, IntPred::Eq, x, five);
        let two = b.constant(e, Ty::I32, 2);
        let doubled = b.int(e, IntOp::Mul, u, two);
        let one = b.constant(e, Ty::I32, 1);
        let w = b.int(e, IntOp::Add, doubled, one);
        let (then, t) = b.block(&[Ty::I1, Ty::I32]);
        let (other, o) = b.block(&[Ty::I1, Ty::I32]);
        b.cond_br(e, hit, (then, vec![k.exec, w]), (other, vec![k.exec, w]));
        let (block, p) = if entered { (then, t) } else { (other, o) };
        let five = b.constant(block, Ty::I32, 5);
        let test = b.cmp(block, IntPred::Eq, p[1], five);
        let seven = b.constant(block, Ty::I32, 7);
        let nine = b.constant(block, Ty::I32, 9);
        let chosen = b.core(block, Ty::I32, Op::Select(test, seven, nine));
        (b, chosen)
    }

    #[test]
    fn selects_on_a_word_derived_from_an_offset_branch_word_hold_the_arms_the_branch_leaves() {
        let missed: Vec<(bool, u32)> = [(true, 7u32), (false, 7), (false, 9)]
            .iter()
            .copied()
            .filter(|&(entered, truth)| {
                let (b, chosen) = chosen_after_offset_branch(entered);
                !addresses(&b, &two_words(), |a| {
                    let form = a.value(chosen, 0, None).0.form;
                    representable(&a.unknowns, &form, truth)
                })
            })
            .collect();
        assert!(missed.is_empty(), "u + 3 = 5 gives u = 2 and 2u + 1 = 5; otherwise u = 2^31 + 2 still gives 5: {:?}", missed);
    }

    #[test]
    fn selects_on_a_word_derived_from_an_offset_branch_word_take_the_arm_the_branch_decides() {
        let (b, chosen) = chosen_after_offset_branch(true);
        let form = addresses(&b, &two_words(), |a| a.value(chosen, 0, None).0.form);
        assert_eq!(form, Form::constant(7), "the block runs only when u + 3 is 5, so 2u + 1 is 5");
    }

    fn chosen_in_a_loop_entered_on_zero_that_counts_up() -> (Build, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let zero = b.constant(e, Ty::I32, 0);
        let is_zero = b.cmp(e, IntPred::Eq, u, zero);
        let (body, p) = b.block(&[Ty::I1, Ty::I32]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.cond_br(e, is_zero, (body, vec![k.exec, u]), (exit, vec![k.exec]));
        let zero = b.constant(body, Ty::I32, 0);
        let test = b.cmp(body, IntPred::Eq, p[1], zero);
        let seven = b.constant(body, Ty::I32, 7);
        let nine = b.constant(body, Ty::I32, 9);
        let chosen = b.core(body, Ty::I32, Op::Select(test, seven, nine));
        let one = b.constant(body, Ty::I32, 1);
        let next = b.int(body, IntOp::Add, p[1], one);
        let three = b.constant(body, Ty::I32, 3);
        let again = b.cmp(body, IntPred::Ult, next, three);
        b.cond_br(body, again, (body, vec![p[0], next]), (exit, vec![p[0]]));
        (b, chosen)
    }

    #[test]
    fn selects_in_a_loop_entered_on_zero_that_counts_up_hold_both_arms() {
        let (b, chosen) = chosen_in_a_loop_entered_on_zero_that_counts_up();
        let missed: Vec<u32> = addresses(&b, &two_words(), |a| {
            let form = a.value(chosen, 0, None).0.form;
            [7u32, 9].iter().copied().filter(|&t| !representable(&a.unknowns, &form, t)).collect()
        });
        assert!(missed.is_empty(), "the first iteration sees 0 and gives 7, the next ones 1 and 2 and give 9: {:?}", missed);
    }

    fn branches_on_chained_orders() -> (Build, Vec<(&'static str, Box<dyn Fn(u32, u32, u32) -> bool>, Vec<(IntPred, ValueId)>)>) {
        use IntPred::*;
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::B32);
        let v = loaded_word(&mut b, &k, e, 4, MemSize::B32);
        let w = loaded_word(&mut b, &k, e, 8, MemSize::B32);
        let fork = |b: &mut Build, at: BlockId, cond: ValueId, exec: ValueId| {
            let (yes, y) = b.block(&[Ty::I1]);
            let (no, n) = b.block(&[Ty::I1]);
            b.cond_br(at, cond, (yes, vec![exec]), (no, vec![exec]));
            ((yes, y[0]), (no, n[0]))
        };
        let c1 = b.cmp(e, Ult, u, v);
        let ((b1, e1), (x1, _)) = fork(&mut b, e, c1, k.exec);
        let c2 = b.cmp(b1, Ult, v, w);
        let ((b2, _), (x2, _)) = fork(&mut b, b1, c2, e1);
        let c1 = |u: u32, v: u32, _: u32| u < v;
        let c2 = |_: u32, v: u32, w: u32| v < w;
        let reach: Vec<(&'static str, BlockId, Box<dyn Fn(u32, u32, u32) -> bool>)> = vec![
            ("b1", b1, Box::new(move |u, v, w| c1(u, v, w))),
            ("x1", x1, Box::new(move |u, v, w| !c1(u, v, w))),
            ("b2", b2, Box::new(move |u, v, w| c1(u, v, w) && c2(u, v, w))),
            ("x2", x2, Box::new(move |u, v, w| c1(u, v, w) && !c2(u, v, w))),
        ];
        let mut blocks = Vec::new();
        for (name, block, reaches) in reach {
            let queries = PREDICATES.iter().map(|&pred| (pred, b.cmp(block, pred, u, w))).collect();
            blocks.push((name, reaches, queries));
        }
        (b, blocks)
    }

    fn chained_order_answers(exact: bool) -> Vec<String> {
        let (b, blocks) = branches_on_chained_orders();
        let points = [0u32, 1, 2, 3, 0x7fff_fffe, 0x7fff_ffff, 0x8000_0000, u32::MAX - 1, u32::MAX];
        let mut wrong = Vec::new();
        addresses(&b, &two_words(), |a| {
            for (name, reaches, queries) in &blocks {
                for &(pred, q) in queries {
                    let mut seen = [false; 2];
                    for &u in &points {
                        for &v in &points {
                            for &w in &points {
                                if reaches(u, v, w) {
                                    seen[compare(pred, u, w) as usize] = true;
                                }
                            }
                        }
                    }
                    let truth = match seen {
                        [false, false] => continue,
                        [true, false] => Some(false),
                        [false, true] => Some(true),
                        _ => None,
                    };
                    let got = a.bit(q, 0, None).0;
                    let ok = if exact { got == truth } else { got.is_none() || got == truth };
                    if !ok {
                        wrong.push(format!("{}: u {:?} w: {:?}, truth {:?}", name, pred, got, truth));
                    }
                }
            }
        });
        wrong
    }

    #[test]
    fn comparisons_under_chained_orders_decide_exactly_what_the_orders_settle_together() {
        let wrong = chained_order_answers(true);
        assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
    }

    #[test]
    fn comparisons_under_chained_orders_hold_every_value_the_orders_leave() {
        let wrong = chained_order_answers(false);
        assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
    }

    fn branches_on_an_order_over_bytes() -> (Build, BlockId, Vec<(IntPred, u32, ValueId)>) {
        use IntPred::*;
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let u = loaded_word(&mut b, &k, e, 0, MemSize::U8);
        let v = loaded_word(&mut b, &k, e, 4, MemSize::U8);
        let ten = b.constant(e, Ty::I32, 10);
        let shifted = b.int(e, IntOp::Add, u, ten);
        let c = b.cmp(e, Ugt, v, shifted);
        let (yes, y) = b.block(&[Ty::I1]);
        let (no, n) = b.block(&[Ty::I1]);
        b.cond_br(e, c, (yes, vec![k.exec]), (no, vec![k.exec]));
        let _ = (y, n);
        let mut queries = Vec::new();
        for &bound in &[9u32, 10, 11, 12, 254, 255] {
            let k = b.constant(yes, Ty::I32, bound as u64);
            for pred in PREDICATES {
                queries.push((pred, bound, b.cmp(yes, pred, v, k)));
            }
        }
        (b, yes, queries)
    }

    fn byte_order_answers(exact: bool) -> Vec<String> {
        let (b, _, queries) = branches_on_an_order_over_bytes();
        let mut wrong = Vec::new();
        addresses(&b, &two_words(), |a| {
            for &(pred, bound, q) in &queries {
                let mut seen = [false; 2];
                for u in 0u32..256 {
                    for v in 0u32..256 {
                        if v > u + 10 {
                            seen[compare(pred, v, bound) as usize] = true;
                        }
                    }
                }
                let truth = match seen {
                    [false, false] => continue,
                    [true, false] => Some(false),
                    [false, true] => Some(true),
                    _ => None,
                };
                let got = a.bit(q, 0, None).0;
                let ok = if exact { got == truth } else { got.is_none() || got == truth };
                if !ok {
                    wrong.push(format!("v {:?} {}: {:?}, truth {:?}", pred, bound, got, truth));
                }
            }
        });
        wrong
    }

    #[test]
    fn comparisons_with_constants_under_an_order_over_bounded_words_decide_exactly_what_the_order_settles() {
        let wrong = byte_order_answers(true);
        assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
    }

    #[test]
    fn comparisons_with_constants_under_an_order_over_bounded_words_hold_every_value_the_order_leaves() {
        let wrong = byte_order_answers(false);
        assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(10)]);
    }
}
