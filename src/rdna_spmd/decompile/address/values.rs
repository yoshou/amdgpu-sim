use super::bits::{compute_bit, compute_word_bit};
use super::control::*;
use super::form::*;
use super::limits::Limits;
use super::memory::*;
use super::program::{Conditions, Program};
use super::queries::*;
use super::symbols::{Key, Slot, Symbols};
use super::trail::*;
use super::wide::compute_high;
use super::word::{compute, concrete};
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

const SEQUENCE: usize = 1 << 16;

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

type StepKey = (BlockId, u8, Target, Option<Region>);

type Step = (Option<u32>, bool, Option<(u32, u32)>);

type Bits = std::rc::Rc<Vec<((usize, u8), bool)>>;

pub(super) struct Values<'a> {
    symbols: Symbols<'a>,
    conditions: Conditions,
    entered: Option<usize>,
    memo: Memo,
    journal: Journal<Entry>,
    guessing: Guessing,
}

struct Table<K, V> {
    done: Cached<K, V>,
    active: Guard<K>,
}

trait Query {
    type Key: Copy + std::hash::Hash + Eq;
    type Answer: Clone;
    fn table(memo: &mut Memo) -> &mut Table<Self::Key, Self::Answer>;
    fn entry(key: Self::Key) -> Entry;
}

macro_rules! tables {
    ($($table:ident: $query:ident($key:ty) => $answer:ty,)*) => {
        #[derive(Default)]
        struct Memo {
            $($table: Table<$key, $answer>,)*
        }

        enum Entry {
            $($query($key),)*
        }

        impl Memo {
            fn clear(&mut self) {
                $(self.$table.done.clear();)*
            }

            #[inline]
            fn evict(&mut self, entry: Entry, depth: usize) {
                match entry {
                    $(Entry::$query(key) => evict(&mut self.$table.done, &key, depth),)*
                }
            }
        }

        $(
            struct $query;

            impl Query for $query {
                type Key = $key;
                type Answer = $answer;

                #[inline]
                fn table(memo: &mut Memo) -> &mut Table<$key, $answer> {
                    &mut memo.$table
                }

                #[inline]
                fn entry(key: $key) -> Entry {
                    Entry::$query(key)
                }
            }
        )*
    };
}

tables! {
    values: ValueOf(ValueKey) => Assumed<Value>,
    bits: BitOf(ValueKey) => Assumed<Option<bool>>,
    loop_bits: LoopBits(BlockId) => Bits,
    steps: StepOf(StepKey) => Step,
    summarized: Summarized(BlockId) => (),
    slots: SlotAt(Slot) => Option<Value>,
    word_bits: WordBit((ValueId, u8)) => Option<bool>,
    highs: HighOf((ValueId, u8)) => Form,
    decisions: Decision(BlockId) => Option<bool>,
    limited: LimitsAt(BlockId) => std::rc::Rc<Limits>,
    reach: Reaching(BlockId) => bool,
}

struct Guard<K>(HashMap<K, usize>);

#[derive(Default)]
struct Guessing {
    level: usize,
    about: HashMap<BlockId, Guesses>,
    values: HashMap<(ValueId, u8), (Value, Depth)>,
    bits: HashMap<(ValueId, u8), (bool, Depth)>,
    slots: Cached<Slot, Option<Value>>,
}

impl<K, V> Default for Table<K, V> {
    fn default() -> Self {
        Self {
            done: Cached::default(),
            active: Guard::default(),
        }
    }
}

impl<K> Default for Guard<K> {
    fn default() -> Self {
        Self(HashMap::default())
    }
}

impl<K: std::hash::Hash + Eq> Guard<K> {
    #[inline]
    fn enter(&mut self, key: K, level: usize) -> Option<Option<usize>> {
        match self.0.insert(key, level) {
            Some(outer) if outer == level => None,
            outside => Some(outside),
        }
    }

    #[inline]
    fn leave(&mut self, key: K, outside: Option<usize>) {
        match outside {
            Some(level) => self.0.insert(key, level),
            None => self.0.remove(&key),
        };
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

impl<'a> Values<'a> {
    pub(super) fn new(program: Program<'a>, conditions: Conditions) -> Self {
        Self {
            symbols: Symbols::new(program),
            conditions,
            entered: None,
            memo: Memo::default(),
            journal: Journal::default(),
            guessing: Guessing::default(),
        }
    }

    pub(super) fn unreached(&self, b: BlockId) -> bool {
        self.memo.reach.done.get(&b).is_some_and(|&(reached, _)| !reached)
    }

    pub(super) fn enter(&mut self, wave: usize) -> bool {
        if self.entered == Some(wave) {
            return false;
        }
        self.entered = Some(wave);
        self.symbols.enter(wave);
        self.memo.clear();
        let mut headers: Vec<BlockId> = self.symbols.program.headers.iter().copied().collect();
        headers.sort_by_key(|h| self.symbols.program.rank[h]);
        let trips: Vec<Unknown> = headers.iter().map(|&h| self.symbols.trips(h)).collect();
        self.symbols.pending = trips.iter().map(|&u| (u, level(1))).collect();
        for (&header, &u) in headers.iter().zip(&trips) {
            let (last, _) = self.sandbox(|this| last_trip(this, header, u));
            if let Some(last) = last {
                self.symbols.unknowns[u as usize].range = Some((0, last));
            }
            self.symbols.pending.remove(&u);
        }
        true
    }

    fn sandbox<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> (T, Depth) {
        let opened = self.symbols.open();
        let mark = self.journal.mark();
        let result = f(self);
        let memo = &mut self.memo;
        self.journal.settle(mark, opened.1, |entry| memo.evict(entry, opened.1));
        let outer = self.symbols.close(opened);
        (result, outer)
    }

    #[inline]
    fn frame<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> (T, Depth) {
        self.symbols.trail.push();
        let result = f(self);
        (result, self.symbols.trail.pop())
    }

    #[inline]
    fn known<Q: Query>(&mut self, key: &Q::Key) -> Option<Q::Answer> {
        let (answer, depth) = Q::table(&mut self.memo).done.get(key)?.clone();
        self.symbols.trail.depend(depth);
        Some(answer)
    }

    #[inline]
    fn guarded<Q: Query, T>(&mut self, key: Q::Key, f: impl FnOnce(&mut Self) -> T) -> Option<(T, Depth)> {
        let outside = Q::table(&mut self.memo).active.enter(key, self.guessing.level)?;
        let found = self.frame(|this| {
            if outside.is_some() {
                this.symbols.trail.depend(level(this.symbols.trail.checking));
            }
            f(this)
        });
        Q::table(&mut self.memo).active.leave(key, outside);
        Some(found)
    }

    #[inline]
    fn store<Q: Query>(&mut self, key: Q::Key, answer: Q::Answer, depth: Depth) {
        self.journal.note(Q::entry(key), depth);
        Q::table(&mut self.memo).done.insert(key, (answer, depth));
    }

    #[inline]
    fn known_assuming<Q: Query<Key = ValueKey, Answer = Assumed<T>>, T: Clone>(&mut self, (v, l, assume): ValueKey) -> Option<Assumed<T>> {
        let done = &Q::table(&mut self.memo).done;
        let hit = match done.get(&(v, l, None)) {
            Some(e) if assume.is_none() || !e.0 .1.open => Some(e.clone()),
            _ => assume.and_then(|_| done.get(&(v, l, assume)).cloned()),
        };
        let (r, depth) = hit?;
        self.symbols.trail.depend(depth);
        Some(r)
    }

    #[inline]
    fn store_assuming<Q: Query<Key = ValueKey, Answer = Assumed<T>>, T: Clone>(&mut self, (v, l, assume): ValueKey, r: &Assumed<T>, depth: Depth) {
        if assume.is_some() {
            self.store::<Q>((v, l, assume), r.clone(), depth);
        }
        if !r.1.used {
            self.store::<Q>((v, l, None), r.clone(), depth);
        }
    }

    fn may_guess(&mut self, header: BlockId) -> bool {
        if self.guessing.about.contains_key(&header) {
            self.symbols.trail.depend(level(self.symbols.trail.checking));
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
            let depth = level(this.symbols.trail.checking);
            this.guessing.level += 1;
            let params: Vec<ValueId> = this.symbols.program.f.blocks[&header].params.iter().map(|p| p.0).collect();
            for &((index, l), b) in bits {
                this.guessing.bits.insert((params[index], l), (b, depth));
            }
            this.guessing.about.insert(
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
                    form: Form::unknown(this.symbols.symbol(key, header)),
                    region,
                };
                let guesses = this.guessing.about.get_mut(&header).unwrap();
                match target {
                    Target::Param(index) => {
                        guesses.params.push((params[index], l));
                        this.guessing.values.insert((params[index], l), (guess, depth));
                    }
                    Target::Slot(address, bytes) => {
                        guesses.slots.push((address, bytes, l));
                        this.guessing.slots.insert((header, address, bytes, l), (Some(guess), depth));
                    }
                }
            }
            let result = f(this);
            for &((index, l), _) in bits {
                this.guessing.bits.remove(&(params[index], l));
            }
            if let Some(guesses) = this.guessing.about.remove(&header) {
                for key in guesses.params {
                    this.guessing.values.remove(&key);
                }
                for (address, bytes, l) in guesses.slots {
                    this.guessing.slots.remove(&(header, address, bytes, l));
                }
            }
            this.guessing.level -= 1;
            result
        })
        .0
    }

    fn loop_bits(&mut self, header: BlockId) -> Bits {
        if let Some(bits) = self.known::<LoopBits>(&header) {
            return bits;
        }
        if !self.may_guess(header) {
            return Bits::default();
        }
        self.summarize(header, None);
        self.known::<LoopBits>(&header).unwrap_or_default()
    }

    fn assumed_entry(&mut self, header: BlockId, target: Target, lane: usize) -> Option<Region> {
        if !self.guessing.about.get(&header).is_some_and(|g| g.summary) {
            return None;
        }
        let first = match target {
            Target::Param(index) => {
                let (entering, _) = edges_into(self, header);
                entering_value(self, &entering, header, index, lane)
            }
            Target::Slot(address, bytes) => entering_slot(self, header, address, bytes, lane),
        }?;
        if let Some(guesses) = self.guessing.about.get_mut(&header) {
            guesses.found.push((target, lane as u8, first.region));
        }
        first.region
    }

    fn summarize(&mut self, header: BlockId, seed: Option<Target>) {
        if seed.is_none() && self.known::<Summarized>(&header).is_some() {
            return;
        }
        if !self.may_guess(header) {
            return;
        }
        let Some(((bits, steps), depth)) = self.guarded::<Summarized, _>(header, |this| this.summary(header, seed)) else {
            return;
        };
        self.store::<LoopBits>(header, bits, depth);
        self.store::<Summarized>(header, (), depth);
        for (key, step) in steps {
            self.store::<StepOf>(key, step, depth);
        }
    }

    fn summary(&mut self, header: BlockId, seed: Option<Target>) -> (Bits, Vec<(StepKey, Step)>) {
        let params: Vec<(ValueId, Ty)> = self.symbols.program.f.blocks[&header].params.clone();
        let (entering, back) = edges_into(self, header);
        let lanes: Vec<usize> = (0..self.symbols.program.lanes()).filter(|&l| self.symbols.valid(l)).collect();
        let mut bits: Vec<((usize, u8), bool)> = match self.known::<LoopBits>(&header) {
            Some(bits) => (*bits).clone(),
            None => {
                let mut bits = Vec::new();
                for (index, &(_, ty)) in params.iter().enumerate() {
                    if ty != Ty::I1 {
                        continue;
                    }
                    for &lane in &lanes {
                        let mut first: Option<Option<bool>> = None;
                        for &e in &entering {
                            let bit = self.bit(self.symbols.program.edge_arg(e, index), lane, None).0;
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
                        if self.symbols.canonical(params[index].0, lane) != lane {
                            continue;
                        }
                        entering_value(self, &entering, header, index, lane)
                    }
                    Target::Slot(address, bytes) => entering_slot(self, header, address, bytes, lane),
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
                if let Some(g) = this.guessing.about.get_mut(&header) {
                    g.summary = true;
                }
                let failed: Vec<(usize, u8)> = guessed
                    .iter()
                    .filter(|&&((index, l), b)| {
                        back.iter().any(|&e| {
                            let a = this.symbols.program.edge_arg(e, index);
                            this.bit(a, l as usize, None).0 != Some(b)
                        })
                    })
                    .map(|&(key, _)| key)
                    .collect();
                let mut all: Vec<Goal> = goals.clone();
                let take = |this: &mut Self, all: &mut Vec<Goal>| {
                    let found = std::mem::take(&mut this.guessing.about.get_mut(&header).unwrap().found);
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
                        Target::Param(index) => this.symbols.symbol(Key::Guess(params[index].0, g.lane), header),
                        Target::Slot(address, bytes) => {
                            this.symbols.symbol(Key::GuessSlot((header, address, bytes, g.lane)), header)
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
                                let a = this.symbols.program.edge_arg(e, index);
                                Some(this.operand(a, header, lane, None).0)
                            }
                            Target::Slot(address, bytes) => {
                                let end = this.symbols.program.f.blocks[&e.0].insts.len();
                                slot_before(this, (e.0, end), address, bytes, lane)
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
                    let v = this.symbols.program.f.blocks[&header].params[index].0;
                    this.symbols.symbol(Key::Guess(v, l), header)
                }
                Target::Slot(address, bytes) => this.symbols.symbol(Key::GuessSlot((header, address, bytes, l)), header),
            };
            let mut step = None;
            let mut stepped = true;
            let mut keeps = true;
            let mut affine = Some(None);
            for &e in back {
                let value = match target {
                    Target::Param(index) => {
                        let a = this.symbols.program.edge_arg(e, index);
                        Some(this.operand(a, header, lane, None).0)
                    }
                    Target::Slot(address, bytes) => {
                        let end = this.symbols.program.f.blocks[&e.0].insts.len();
                        slot_before(this, (e.0, end), address, bytes, lane)
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

    fn bit_unless_assumed(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
        let lane = self.symbols.canonical(v, lane);
        let l = lane as u8;
        if let Some(&(guess, depth)) = self.guessing.bits.get(&(v, l)) {
            self.symbols.trail.depend(depth);
            return unassumed(Some(guess));
        }
        let key = (v, l, assume);
        if let Some(r) = self.known_assuming::<BitOf, _>(key) {
            return r;
        }
        let Some((r, depth)) = self.guarded::<BitOf, _>(key, |this| compute_bit(this, v, lane, assume)) else {
            return unassumed(None);
        };
        self.store_assuming::<BitOf, _>(key, &r, depth);
        r
    }
}

impl<'a> Queries<'a> for Values<'a> {
    #[inline]
    fn program(&self) -> &Program<'a> {
        &self.symbols.program
    }

    #[inline]
    fn symbols(&self) -> &Symbols<'a> {
        &self.symbols
    }

    #[inline]
    fn symbols_mut(&mut self) -> &mut Symbols<'a> {
        &mut self.symbols
    }

    #[inline]
    fn reason<T>(&mut self, f: impl FnOnce(&mut Conditions, &Program<'a>) -> T) -> T {
        f(&mut self.conditions, &self.symbols.program)
    }

    fn value(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Value> {
        if let Some(k) = assume.and_then(|a| self.conditions.assumed_constant(&self.symbols.program, a, v)) {
            return (
                Value::constant(k),
                Reliance {
                    used: true,
                    ..Reliance::default()
                },
            );
        }
        let lane = self.symbols.canonical(v, lane);
        let l = lane as u8;
        if let Some((guess, depth)) = self.guessing.values.get(&(v, l)).cloned() {
            self.symbols.trail.depend(depth);
            return unassumed(guess);
        }
        if let Site::Param { block, index } = self.symbols.program.facts.site[v.0] {
            if let Some(depth) = self
                .guessing.about
                .get(&block)
                .map(|g| g.depth)
                .filter(|_| self.symbols.program.carried(v, block, index))
            {
                self.symbols.trail.depend(depth);
                let region = self.assumed_entry(block, Target::Param(index), lane);
                let symbol = self.symbols.symbol(Key::Guess(v, l), block);
                let guess = Value {
                    form: Form::unknown(symbol),
                    region,
                };
                self.guessing.values.insert((v, l), (guess.clone(), depth));
                if let Some(guesses) = self.guessing.about.get_mut(&block) {
                    guesses.params.push((v, l));
                }
                return unassumed(guess);
            }
        }
        let key = (v, l, assume);
        if let Some(r) = self.known_assuming::<ValueOf, _>(key) {
            return r;
        }
        let Some((r, depth)) = self.guarded::<ValueOf, _>(key, |this| compute(this, v, lane, assume)) else {
            let shared = self.symbols.program.facts.uniform[v.0];
            let block = self.symbols.program.block_of(v);
            let u = self.symbols.intern(
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
        let root = self.symbols.program.copies.get(&v).copied().unwrap_or(v);
        let r = if assume.is_none() && self.symbols.program.equated.contains(&root) {
            (r.0, Reliance { open: true, ..r.1 })
        } else {
            r
        };
        self.store_assuming::<ValueOf, _>(key, &r, depth);
        r
    }

    fn bit(&mut self, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
        if let Some(a) = assume {
            let root = self.symbols.program.copies.get(&v).copied().unwrap_or(v);
            if self.conditions.assumes(&self.symbols.program, a, root) {
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
        let root = self.symbols.program.copies.get(&v).copied().unwrap_or(v);
        if r.0.is_none() && self.symbols.program.assumable.contains(&root) {
            return (r.0, Reliance { open: true, ..r.1 });
        }
        r
    }

    fn limits(&mut self, b: BlockId) -> std::rc::Rc<Limits> {
        if let Some(found) = self.known::<LimitsAt>(&b) {
            return found;
        }
        let Some((limits, depth)) = self.guarded::<LimitsAt, _>(b, |this| compute_limits(this, b)) else {
            return std::rc::Rc::default();
        };
        let limits = std::rc::Rc::new(limits);
        self.store::<LimitsAt>(b, limits.clone(), depth);
        limits
    }

    fn high(&mut self, v: ValueId, lane: usize) -> Form {
        let v = self.symbols.program.copies.get(&v).copied().unwrap_or(v);
        let lane = self.symbols.canonical(v, lane);
        let key = (v, lane as u8);
        if let Some(form) = self.known::<HighOf>(&key) {
            return form;
        }
        let found = self.guarded::<HighOf, _>(key, |this| {
            let found = compute_high(this, v, lane);
            found.unwrap_or_else(|| this.symbols.opaque_high(v, lane))
        });
        let Some((found, depth)) = found else {
            return self.symbols.opaque_high(v, lane);
        };
        self.store::<HighOf>(key, found.clone(), depth);
        found
    }

    fn word_bit(&mut self, w: ValueId, bit: usize) -> Option<bool> {
        let w = self.symbols.program.copies.get(&w).copied().unwrap_or(w);
        let key = (w, bit as u8);
        if let Some(r) = self.known::<WordBit>(&key) {
            return r;
        }
        let (r, depth) = self.guarded::<WordBit, _>(key, |this| compute_word_bit(this, w, bit))?;
        self.store::<WordBit>(key, r, depth);
        r
    }

    fn slot_entry(&mut self, block: BlockId, address: u32, bytes: u32, lane: usize) -> Option<Value> {
        if block == self.symbols.program.f.entry {
            return None;
        }
        let key: Slot = (block, address, bytes, lane as u8);
        if let Some((guess, depth)) = self.guessing.slots.get(&key).cloned() {
            self.symbols.trail.depend(depth);
            return guess;
        }
        if let Some(depth) = self.guessing.about.get(&block).map(|g| g.depth) {
            self.symbols.trail.depend(depth);
            let region = self.assumed_entry(block, Target::Slot(address, bytes), lane);
            let symbol = self.symbols.symbol(Key::GuessSlot(key), block);
            let guess = Value {
                form: Form::unknown(symbol),
                region,
            };
            self.guessing.slots.insert(key, (Some(guess.clone()), depth));
            if let Some(guesses) = self.guessing.about.get_mut(&block) {
                guesses.slots.push((address, bytes, lane as u8));
            }
            return Some(guess);
        }
        if let Some(r) = self.known::<SlotAt>(&key) {
            return r;
        }
        let (result, depth) = self.guarded::<SlotAt, _>(key, |this| join_slot(this, block, address, bytes, lane))?;
        self.store::<SlotAt>(key, result.clone(), depth);
        result
    }

    fn decision(&mut self, b: BlockId) -> Option<bool> {
        if let Some(decided) = self.known::<Decision>(&b) {
            return decided;
        }
        let Term::CondBr { cond, .. } = self.symbols.program.f.blocks[&b].term else {
            return None;
        };
        let (decided, depth) = self.guarded::<Decision, _>(b, |this| decide(this, cond))?;
        self.store::<Decision>(b, decided, depth);
        decided
    }

    fn reached(&mut self, b: BlockId) -> bool {
        if b == self.symbols.program.f.entry {
            return true;
        }
        if let Some(reached) = self.known::<Reaching>(&b) {
            return reached;
        }
        let found = self.guarded::<Reaching, _>(b, |this| {
            let facts = this.symbols.program.facts;
            let own = this.symbols.program.rank[&b];
            facts.incoming[&b]
                .iter()
                .any(|&(pred, slot)| this.symbols.program.rank[&pred] < own && this.reached(pred) && takes(this, pred, slot))
        });
        let Some((reached, depth)) = found else {
            return true;
        };
        self.store::<Reaching>(b, reached, depth);
        reached
    }

    fn loop_bit(&mut self, header: BlockId, index: usize, lane: usize) -> Option<bool> {
        let bits = self.loop_bits(header);
        let key = (index, lane as u8);
        bits.binary_search_by(|&(k, _)| k.cmp(&key)).ok().map(|i| bits[i].1)
    }

    fn recur(&mut self, header: BlockId, lane: usize, target: Target) -> Option<Value> {
        if !self.may_guess(header) {
            return None;
        }
        let (entering, back) = edges_into(self, header);
        let first = match target {
            Target::Param(index) => entering_value(self, &entering, header, index, lane)?,
            Target::Slot(address, bytes) => entering_slot(self, header, address, bytes, lane)?,
        };
        let l = lane as u8;
        let region = first.region;
        let step_key = (header, l, target, region);
        let found = match self.known::<StepOf>(&step_key) {
            Some(step) => Some(step),
            None => {
                self.summarize(header, Some(target));
                self.known::<StepOf>(&step_key)
            }
        };
        let (step, keeps, _) = match found {
            Some(step) => step,
            None => {
                let bits = self.loop_bits(header);
                let (step, depth) = self.frame(|this| this.find_step(header, lane, target, region, &bits, &back));
                self.store::<StepOf>(step_key, step, depth);
                step
            }
        };
        if let Some(step) = step {
            return Some(self.symbols.advance(first, step, header));
        }
        let symbol = match target {
            Target::Param(index) => {
                let v = self.symbols.program.f.blocks[&header].params[index].0;
                self.symbols.symbol(Key::Guess(v, l), header)
            }
            Target::Slot(address, bytes) => self.symbols.symbol(Key::GuessSlot((header, address, bytes, l)), header),
        };
        (keeps && region.is_some()).then(|| Value {
            form: Form::unknown(symbol),
            region,
        })
    }

    fn sequence(&mut self, v: ValueId, header: BlockId, index: usize, lane: usize) -> Option<Value> {
        let trips = self.symbols.trips(header);
        let last = self.symbols.unknowns[trips as usize].range?.1;
        if last as usize >= SEQUENCE {
            return None;
        }
        let l = lane as u8;
        let key = (header, l, Target::Param(index), None);
        let (_, _, affine) = self.known::<StepOf>(&key)?;
        let (entering, back) = edges_into(self, header);
        let first = entering_value(self, &entering, header, index, lane)?;
        if first.region.is_some() {
            return None;
        }
        let starts = self.symbols.starts(&first.form, SEQUENCE / (last as usize + 1))?;
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
                let mut args: Vec<ValueId> = back.iter().map(|&edge| self.symbols.program.edge_arg(edge, index)).collect();
                args.sort_unstable();
                args.dedup();
                let mut seen: BTreeSet<u32> = starts.iter().copied().collect();
                let mut frontier = starts.clone();
                for _ in 0..last {
                    let mut next = Vec::new();
                    for &x in &frontier {
                        for &arg in &args {
                            let y = concrete(self, arg, v, x, header, lane, 0)? as u32;
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
        let shared = self.symbols.program.facts.uniform[v.0];
        let u = self.symbols.intern(
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
}
