use super::form::*;
use super::limits::{intersected, shifted_pieces, Limits};
use super::program::Program;
use super::trail::*;
use super::HashSet;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum Key {
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

pub(super) type Slot = (BlockId, u32, u32, u8);

pub(super) struct Symbols<'a> {
    pub(super) program: Program<'a>,
    pub(super) trail: Trail,
    pub(super) journal: Journal<Key>,
    pub(super) wave: usize,
    pub(super) unknowns: Vec<UnknownInfo>,
    keys: Cached<Key, Unknown>,
    pub(super) derived: HashMap<Unknown, Vec<ValueId>>,
    pub(super) monomials: HashMap<Unknown, Vec<Unknown>>,
    pub(super) opaque_highs: HashSet<Unknown>,
    pub(super) pending: HashMap<Unknown, Depth>,
}

impl<'a> Symbols<'a> {
    pub(super) fn enter(&mut self, wave: usize) {
        self.wave = wave;
        self.unknowns.clear();
        self.keys.clear();
        self.derived.clear();
        self.monomials.clear();
        self.opaque_highs.clear();
    }

    #[inline]
    pub(super) fn open(&mut self) -> (usize, usize) {
        (self.journal.mark(), self.trail.open())
    }

    #[inline]
    pub(super) fn close(&mut self, (mark, depth): (usize, usize)) -> Depth {
        let keys = &mut self.keys;
        self.journal.settle(mark, depth, |key| evict(keys, key, depth));
        self.trail.close(depth)
    }

    pub(super) fn valid(&self, lane: usize) -> bool {
        let lanes = self.program.lanes();
        lane < lanes && ((self.wave * lanes + lane) as u32) < self.program.env.workgroup_size()
    }

    fn ids(&self, lane: usize) -> (u32, u32, u32) {
        let flat = (self.wave * self.program.lanes() + lane) as u32;
        let [bx, by, _] = self.program.env.block;
        (flat % bx, (flat / bx) % by, flat / (bx * by))
    }

    pub(super) fn canonical(&self, v: ValueId, lane: usize) -> usize {
        if self.program.facts.uniform[v.0] && self.valid(lane) {
            0
        } else {
            lane
        }
    }

    fn known_key(&self, key: &Key) -> Option<Unknown> {
        let found = self.keys.get(key);
        found.map(|&(u, depth)| {
            self.trail.depend(depth);
            u
        })
    }

    pub(super) fn intern(&mut self, key: Key, mut info: UnknownInfo) -> Unknown {
        let found = self.keys.get(&key);
        if let Some(&(u, depth)) = found {
            self.trail.depend(depth);
            return u;
        }
        let u = self.unknowns.len() as Unknown;
        info.rank = self.program.rank.get(&info.block).copied().unwrap_or(0);
        self.unknowns.push(info);
        let depth = self.trail.current();
        if depth != FREE {
            self.journal.note(key.clone(), depth);
        }
        self.keys.insert(key, (u, depth));
        u
    }

    pub(super) fn opaque(&mut self, v: ValueId, lane: usize, range: Option<(u32, u32)>) -> Value {
        let shared = self.program.facts.uniform[v.0];
        let key = Key::Value(v, (!shared).then_some(lane as u8));
        let block = self.program.block_of(v);
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

    pub(super) fn leave(&mut self, value: Value, at: BlockId, lane: usize) -> Value {
        let leaving: Vec<bool> = {
            let here = self.program.loops.get(&at).map(|l| l.as_slice()).unwrap_or(&[]);
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

    pub(super) fn range(&self, u: Unknown) -> Option<(u32, u32)> {
        if let Some(&depth) = self.pending.get(&u) {
            self.trail.depend(depth);
        }
        self.unknowns[u as usize].range
    }

    pub(super) fn spread(&mut self, key: Key, shared: bool, block: BlockId, forms: &[Form]) -> Option<Form> {
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

    pub(super) fn merge_held(&mut self, values: Vec<Value>, slot: Slot, index: usize) -> Option<Value> {
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

    pub(super) fn starts(&mut self, form: &Form, most: usize) -> Option<Vec<u32>> {
        if let Some(k) = form.as_constant() {
            return Some(vec![k]);
        }
        let &[(u, c)] = form.terms.as_slice() else {
            return None;
        };
        let range = self.range(u);
        let choices: Vec<u32> = match (&self.unknowns[u as usize].values, range) {
            (Some(set), _) if set.len() <= most => set.to_vec(),
            (_, Some((low, high))) if ((high - low) as usize) < most => (low..=high).collect(),
            _ => return None,
        };
        Some(choices.into_iter().map(|x| form.constant.wrapping_add(c.wrapping_mul(x))).collect())
    }

    pub(super) fn bounds(&self, form: &Form) -> Option<(u64, u64)> {
        let (mut low, mut high) = (form.constant as u64, form.constant as u64);
        for &(u, c) in &form.terms {
            let (l, h) = self.range(u)?;
            low = low.checked_add(c as u64 * l as u64)?;
            high = high.checked_add(c as u64 * h as u64)?;
        }
        (high < 1 << 32).then_some((low, high))
    }

    pub(super) fn input(&mut self, v: ValueId, index: usize, lane: usize) -> Value {
        let entry = self.program.entry;
        match self.program.inputs[index].source {
            ParameterSource::Vgpr(r) if entry.workitem_register(r) => {
                let (x, y, z) = self.ids(lane);
                let ids = [x, y, z];
                Value::constant(entry.workitem_fields(r).fold(0, |sum, (axis, shift)| sum | ids[axis] << shift))
            }
            ParameterSource::Sgpr(n) if Some(n) == entry.kernarg_ptr => self.base(Region::Kernarg),
            ParameterSource::Sgpr(n) if Some(n) == entry.dispatch_ptr => self.base(Region::Dispatch),
            ParameterSource::Sgpr(n) if entry.workgroup_fields(n).next().is_some() => {
                let fields: Vec<_> = entry.workgroup_fields(n).collect();
                let mut form = Form::constant(0);
                for (axis, shift) in fields {
                    form = form.add(&Form::unknown(self.workgroup(axis)).scale(1 << shift));
                }
                Value::of(form)
            }
            _ => self.opaque(v, lane, None),
        }
    }

    pub(super) fn base(&mut self, region: Region) -> Value {
        let entry = self.program.f.entry;
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
        let count = self.program.env.grid[axis].max(1);
        let entry = self.program.f.entry;
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

    pub(super) fn pieces(&self, x: &Form, limits: &Limits) -> Vec<(u64, u64)> {
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

    pub(super) fn opaque_high(&mut self, v: ValueId, lane: usize) -> Form {
        let shared = self.program.facts.uniform[v.0];
        let block = self.program.block_of(v);
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

    pub(super) fn through(&self, form: &Form) -> Vec<BlockId> {
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

    pub(super) fn variance(&self, forms: &[&Form], at: BlockId) -> (BlockId, Option<BlockId>) {
        let none: &[BlockId] = &[];
        let mut best: Option<(BlockId, &[BlockId])> = None;
        for form in forms {
            for &(u, _) in &form.terms {
                let b = self.unknowns[u as usize].block;
                let around = self.program.loops.get(&b).map_or(none, |l| l.as_slice());
                best = match best {
                    None => Some((b, around)),
                    Some((_, current)) if current.iter().all(|h| around.contains(h)) => Some((b, around)),
                    Some(kept) if around.iter().all(|h| kept.1.contains(h)) => Some(kept),
                    Some(_) => {
                        let varying = self.program.loops.get(&at).map_or(none, |l| l.as_slice()).iter().copied().filter(|h| {
                            forms.iter().any(|f| {
                                f.terms.iter().any(|&(u, _)| self.program.loops.get(&self.unknowns[u as usize].block).is_some_and(|l| l.contains(h)))
                            })
                        });
                        let inner = varying.max_by_key(|h| self.program.rank[h]).unwrap_or(self.program.f.entry);
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

    pub(super) fn selected(&mut self, v: ValueId, block: BlockId, edges: &[(BlockId, usize)], forms: &[Form], lane: usize) -> Option<Form> {
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
        let incoming = &self.program.facts.incoming[&block];
        if incoming.len() > 64 {
            return None;
        }
        let reached = incoming
            .iter()
            .enumerate()
            .filter(|(_, e)| edges.contains(e))
            .fold(0u64, |m, (i, _)| m | 1 << i);
        let through = self.program.loops.get(&block).cloned().unwrap_or_default();
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

    pub(super) fn chosen(&mut self, v: ValueId, c: ValueId, x: &Form, y: &Form, lane: usize) -> Option<Form> {
        let difference = x.sub(y);
        if difference.as_constant().is_none() && self.bounds(&difference).or_else(|| self.bounds(&difference.scale(u32::MAX))).is_none() {
            return None;
        }
        let c = self.program.copies.get(&c).copied().unwrap_or(c);
        let shared = self.program.facts.uniform[c.0];
        let block = self.program.block_of(c);
        let through = self.program.loops.get(&block).cloned().unwrap_or_default();
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

    pub(super) fn product(&mut self, v: ValueId, a: &Form, b: &Form, lane: usize) -> Value {
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
        let (block, key) = self.variance(&[a, b], self.program.block_of(v));
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
        let (block, key) = self.variance(&refs, self.program.block_of(v));
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

    pub(super) fn both(&mut self, v: ValueId, a: &Form, b: &Form, lane: usize) -> Form {
        let (a, b) = if (a.terms.as_slice(), a.constant) <= (b.terms.as_slice(), b.constant) { (a, b) } else { (b, a) };
        let (shared, l, through) = self.derivation(&[a, b], lane);
        let (block, key) = self.variance(&[a, b], self.program.block_of(v));
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

    pub(super) fn shifted_by(&mut self, v: ValueId, kind: IntOp, a: &Form, s: &Form, lane: usize) -> Form {
        let (shared, l, through) = self.derivation(&[a, s], lane);
        let (block, key) = self.variance(&[a, s], self.program.block_of(v));
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

    pub(super) fn power(&mut self, v: ValueId, s: &Form, lane: usize) -> Form {
        let (shared, l, through) = self.derivation(&[s], lane);
        let (block, key) = self.variance(&[s], self.program.block_of(v));
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

    pub(super) fn disjoint_bits(&self, a: &Form, b: &Form) -> bool {
        let fits = |this: &Self, low: &Form, high: &Form| {
            let zeros = high.alignment().min(high.constant.trailing_zeros());
            match this.bounds(low) {
                Some((_, top)) => zeros >= 32 || top < 1u64 << zeros,
                None => false,
            }
        };
        fits(self, a, b) || fits(self, b, a)
    }

    pub(super) fn uniform(&mut self, v: ValueId) -> Value {
        let block = self.program.block_of(v);
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

    pub(super) fn symbol(&mut self, key: Key, header: BlockId) -> Unknown {
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

    pub(super) fn advance(&mut self, first: Value, step: u32, header: BlockId) -> Value {
        if step == 0 {
            return first;
        }
        let trips = self.trips(header);
        Value {
            form: first.form.add(&Form::unknown(trips).scale(step)),
            region: first.region,
        }
    }

    pub(super) fn trips(&mut self, header: BlockId) -> Unknown {
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

    pub(super) fn unresolved(&self, v: ValueId, lane: usize, value: &Value) -> bool {
        let key = Key::Value(v, (!self.program.facts.uniform[v.0]).then_some(lane as u8));
        let found = self.keys.get(&key);
        value.region.is_none() && value.form.constant == 0 && found.is_some_and(|&(u, _)| value.form.terms == [(u, 1)])
    }

    pub(super) fn trip(&self, header: BlockId) -> Option<Unknown> {
        self.keys.get(&Key::Trip(header)).map(|&(u, _)| u)
    }

    pub(super) fn waves(&self) -> usize {
        (self.program.env.workgroup_size() as usize).div_ceil(self.program.lanes())
    }
}

impl<'a> Symbols<'a> {
    pub(super) fn new(program: Program<'a>) -> Self {
        Self {
            program,
            trail: Trail::default(),
            journal: Journal::default(),
            wave: 0,
            unknowns: Vec::new(),
            keys: Cached::default(),
            derived: HashMap::default(),
            monomials: HashMap::default(),
            opaque_highs: HashSet::default(),
            pending: HashMap::default(),
        }
    }
}
