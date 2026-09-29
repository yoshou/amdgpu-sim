use super::address::{aligns, Copies, Region, Regions, LANES, PRIVATE_MEMORY};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::engine::EntryLayout;
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

const FIXED: [Option<Region>; 4] = [None, Some(Region::Kernarg), Some(Region::Dispatch), Some(Region::Private)];

pub(super) struct Bounds {
    pub(super) known: Vec<Option<Option<Region>>>,
    pub(super) plain: Vec<bool>,
    pub(super) carried: HashMap<ValueId, Vec<Regions>>,
    pub(super) loaded: HashMap<ValueId, Regions>,
    pub(super) spills: Vec<Spill>,
    pub(super) exposing: Vec<Exposure>,
    pub(super) written: HashMap<(BlockId, usize), Regions>,
}

pub(super) struct Spill {
    pub(super) at: (BlockId, usize),
    pub(super) words: Option<(u32, u32)>,
    pub(super) data: Vec<ValueId>,
    pub(super) mask: ValueId,
    pub(super) parts: Vec<Regions>,
}

impl Spill {
    pub(super) fn part(&self, lane: usize) -> &Regions {
        &self.parts[if self.parts.len() == 1 { 0 } else { lane }]
    }
}

#[derive(Clone)]
pub(super) struct Exposure {
    pub(super) operands: Vec<ValueId>,
    pub(super) assume: Option<ValueId>,
    pub(super) candidates: Vec<u64>,
}

#[derive(Clone, Copy, PartialEq)]
enum Demand {
    Dead,
    Under(ValueId),
    All,
}

impl Demand {
    fn meet(self, other: Demand) -> Demand {
        match (self, other) {
            (Demand::Dead, x) | (x, Demand::Dead) => x,
            (Demand::Under(x), Demand::Under(y)) if x == y => self,
            _ => Demand::All,
        }
    }
}

struct Sets<'a> {
    f: &'a Func,
    facts: &'a Facts,
    copies: &'a Copies,
    inputs: &'a [Parameter],
    entry: &'a EntryLayout,
    allocations: Vec<u64>,
    bindings: HashMap<u32, u64>,
    words: usize,
    folded: Vec<Option<u32>>,
    rank: HashMap<BlockId, usize>,
    incoming: Vec<Vec<(&'a [ValueId], bool)>>,
    bits: Vec<u64>,
    slots: HashMap<u32, Vec<u64>>,
    anywhere: Vec<u64>,
    partial: bool,
    parts: HashMap<ValueId, Vec<u64>>,
}

impl<'a> Sets<'a> {
    fn new(
        f: &'a Func,
        facts: &'a Facts,
        copies: &'a Copies,
        inputs: &'a [Parameter],
        entry: &'a EntryLayout,
        env: &Environment,
        headers: &BTreeSet<BlockId>,
    ) -> Self {
        let mut allocations: Vec<u64> = env.bindings.iter().map(|b| b.allocation).collect();
        allocations.sort_unstable();
        allocations.dedup();
        let words = (FIXED.len() + allocations.len()).div_ceil(64);
        let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
        let incoming = facts
            .order
            .iter()
            .map(|b| {
                facts.incoming[b]
                    .iter()
                    .map(|&(pred, slot)| {
                        let edge = f.blocks[&pred].term.edges().nth(slot).unwrap();
                        (&edge.args[..], headers.contains(b) && rank[&pred] >= rank[b])
                    })
                    .collect()
            })
            .collect();
        let mut folded = vec![None; f.types.len()];
        for b in &facts.order {
            for inst in &f.blocks[b].insts {
                if let Inst::Core { value, op, .. } = inst {
                    folded[value.0] = match *op {
                        Op::Const(_, k) => Some(k as u32),
                        Op::Int(IntOp::Add, x, y) => folded[x.0].zip(folded[y.0]).map(|(x, y): (u32, u32)| x.wrapping_add(y)),
                        Op::Int(IntOp::Sub, x, y) => folded[x.0].zip(folded[y.0]).map(|(x, y): (u32, u32)| x.wrapping_sub(y)),
                        _ => None,
                    };
                }
            }
        }
        Self {
            f,
            facts,
            copies,
            inputs,
            entry,
            allocations,
            bindings: env.bindings.iter().map(|b| (b.offset, b.allocation)).collect(),
            words,
            folded,
            rank,
            incoming,
            bits: vec![0; f.types.len() * words],
            slots: HashMap::default(),
            anywhere: vec![0; words],
            partial: env.workgroup_size() as usize % LANES != 0,
            parts: HashMap::default(),
        }
    }

    fn index(&self, r: Option<Region>) -> usize {
        match r {
            Some(Region::Allocation(id)) => FIXED.len() + self.allocations.binary_search(&id).unwrap(),
            r => FIXED.iter().position(|&x| x == r).unwrap(),
        }
    }

    fn mark(&self, set: &mut [u64], r: Option<Region>) {
        let i = self.index(r);
        set[i / 64] |= 1 << (i % 64);
    }

    fn has(&self, set: &[u64], r: Option<Region>) -> bool {
        let i = self.index(r);
        set[i / 64] >> (i % 64) & 1 != 0
    }

    fn of(&self, x: ValueId) -> &[u64] {
        &self.bits[x.0 * self.words..(x.0 + 1) * self.words]
    }

    fn part(&self, x: ValueId, lane: usize) -> &[u64] {
        match self.parts.get(&x) {
            Some(parts) => &parts[lane * self.words..(lane + 1) * self.words],
            None => self.of(x),
        }
    }

    fn split(&self, v: ValueId) -> Option<Vec<u64>> {
        let w = self.words;
        match self.facts.site[v.0] {
            Site::Param { .. } if self.parts.is_empty() => None,
            Site::Param { block, index } if block != self.f.entry => {
                if let Some(root) = self.copies.get(&v) {
                    return self.parts.get(root).cloned();
                }
                let incoming = &self.incoming[self.rank[&block]];
                if !incoming.iter().any(|&(args, _)| self.parts.contains_key(&args[index])) {
                    return None;
                }
                let mut out = vec![0; LANES * w];
                for &(args, _) in incoming {
                    for lane in 0..LANES {
                        merge(&mut out[lane * w..(lane + 1) * w], self.part(args[index], lane));
                    }
                }
                Some(out)
            }
            Site::Inst { block, index } => match &self.f.blocks[&block].insts[index] {
                Inst::Core {
                    op: Op::Select(_, a, b),
                    ..
                } if self.parts.contains_key(a) || self.parts.contains_key(b) => {
                    let mut out = vec![0; LANES * w];
                    for lane in 0..LANES {
                        merge(&mut out[lane * w..(lane + 1) * w], self.part(*a, lane));
                        merge(&mut out[lane * w..(lane + 1) * w], self.part(*b, lane));
                    }
                    Some(out)
                }
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::WriteLane),
                    inputs,
                    ..
                } => {
                    let mut out = vec![0; LANES * w];
                    for lane in 0..LANES {
                        out[lane * w..(lane + 1) * w].copy_from_slice(self.part(inputs[2], lane));
                    }
                    match self.constant(inputs[1]) {
                        Some(k) => {
                            let k = (k & 31) as usize;
                            out[k * w..(k + 1) * w].copy_from_slice(self.of(inputs[0]));
                        }
                        None => {
                            for lane in 0..LANES {
                                merge(&mut out[lane * w..(lane + 1) * w], self.of(inputs[0]));
                            }
                        }
                    }
                    Some(out)
                }
                Inst::Effect {
                    op: EffectOp::Memory {
                        op: MemoryOp::Load(size),
                        space: Space::Scratch,
                        ..
                    },
                    inputs,
                    ..
                } if !self.slots.is_empty() => {
                    let mut out = vec![0; LANES * w];
                    for lane in 0..LANES {
                        let part = &mut out[lane * w..(lane + 1) * w];
                        self.slot(inputs[0], size.bytes(), lane, part);
                        part[0] |= 1;
                    }
                    Some(out)
                }
                _ => None,
            },
            Site::Unreached => None,
            Site::Param { .. } => None,
        }
    }

    fn constant(&self, x: ValueId) -> Option<u32> {
        match self.facts.op(self.f, x) {
            Some(Op::Const(_, k)) => Some(k as u32),
            _ => None,
        }
    }

    fn offset(&self, mut x: ValueId) -> Option<u32> {
        let mut at = 0u32;
        loop {
            x = self.copies.get(&x).copied().unwrap_or(x);
            match self.facts.site[x.0] {
                Site::Param { block, index } if block == self.f.entry => {
                    return match self.inputs[index].source {
                        ParameterSource::Sgpr(n) if Some(n) == self.entry.kernarg_ptr => Some(at),
                        _ => None,
                    };
                }
                _ => match self.facts.op(self.f, x)? {
                    Op::Pack64(lo, _) => x = lo,
                    Op::Int(IntOp::Add, a, b) => match (self.folded[a.0], self.folded[b.0]) {
                        (None, Some(k)) => (x, at) = (a, at.wrapping_add(k)),
                        (Some(k), None) => (x, at) = (b, at.wrapping_add(k)),
                        _ => return None,
                    },
                    _ => return None,
                },
            }
        }
    }

    fn eval(&self, v: ValueId, acc: &mut [u64]) {
        match self.facts.site[v.0] {
            Site::Param { block, index } => self.param(self.rank[&block], block, index, v, acc),
            Site::Inst { block, index } => self.inst(&self.f.blocks[&block].insts[index], v, acc),
            Site::Unreached => {
                acc.fill(0);
                acc[0] = 1;
            }
        }
    }

    fn param(&self, rank: usize, block: BlockId, index: usize, v: ValueId, acc: &mut [u64]) {
        acc.fill(0);
        if let Some(&root) = self.copies.get(&v) {
            merge(acc, self.of(root));
            return;
        }
        if block == self.f.entry {
            let r = match self.inputs[index].source {
                ParameterSource::Vgpr(n) if n != 0 => return,
                ParameterSource::Sgpr(n) if Some(n) == self.entry.kernarg_ptr => Some(Region::Kernarg),
                ParameterSource::Sgpr(n) if Some(n) == self.entry.dispatch_ptr => Some(Region::Dispatch),
                _ => None,
            };
            self.mark(acc, r);
            return;
        }
        for &(args, _) in &self.incoming[rank] {
            merge(acc, self.of(args[index]));
        }
    }

    fn inst(&self, inst: &Inst, v: ValueId, acc: &mut [u64]) {
        acc.fill(0);
        let (f, facts) = (self.f, self.facts);
        match inst {
            Inst::Core { op, .. } => match *op {
                Op::Int(IntOp::Or | IntOp::Xor, a, b) => {
                    let (x, y) = (self.of(a), self.of(b));
                    pointers(acc, x);
                    pointers(acc, y);
                    if x[0] & y[0] & 1 != 0 {
                        acc[0] |= 1;
                    }
                }
                Op::Int(IntOp::Add, a, b) => {
                    let (x, y) = (self.of(a), self.of(b));
                    if y[0] & 1 != 0 {
                        pointers(acc, x);
                    }
                    if x[0] & 1 != 0 {
                        pointers(acc, y);
                    }
                    if x[0] & y[0] & 1 != 0 || (points(x) && points(y)) {
                        acc[0] |= 1;
                    }
                }
                Op::Int(IntOp::Sub, a, b) => {
                    let (x, y) = (self.of(a), self.of(b));
                    if y[0] & 1 != 0 {
                        pointers(acc, x);
                    }
                    if x[0] & 1 != 0 || (points(x) && points(y)) {
                        acc[0] |= 1;
                    }
                }
                Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, to, a)
                    if to.bits() >= 32 && f.types[a.0].bits() >= 32 =>
                {
                    merge(acc, self.of(a));
                }
                Op::Pack64(lo, _) | Op::UnpackLo(lo) => {
                    merge(acc, self.of(lo));
                }
                Op::UnpackHi(x) => match facts.op(f, x) {
                    Some(Op::Pack64(_, hi)) => {
                        merge(acc, self.of(hi));
                    }
                    _ => acc[0] |= 1,
                },
                Op::Select(_, a, b) => {
                    merge(acc, self.of(a));
                    merge(acc, self.of(b));
                }
                Op::Int(IntOp::And, a, b) => match (self.constant(a), self.constant(b)) {
                    (None, Some(m)) if aligns(m) => {
                        merge(acc, self.of(a));
                    }
                    (Some(m), None) if aligns(m) => {
                        merge(acc, self.of(b));
                    }
                    (None, None) => {
                        let (x, y) = (self.of(a), self.of(b));
                        if y[0] & 1 != 0 {
                            pointers(acc, x);
                        }
                        if x[0] & 1 != 0 {
                            pointers(acc, y);
                        }
                        acc[0] |= 1;
                    }
                    _ => acc[0] |= 1,
                },
                Op::Int(IntOp::LShr, a, _) if f.types[v.0] != Ty::I64 => {
                    merge(acc, self.of(a));
                    acc[0] |= 1;
                }
                Op::Env(Env::ScratchBase) => self.mark(acc, Some(Region::Private)),
                _ => acc[0] |= 1,
            },
            Inst::Effect {
                op: EffectOp::Memory {
                    op: MemoryOp::Load(size),
                    space,
                    ..
                },
                inputs,
                ..
            } => {
                acc[0] |= 1;
                if *space == Space::Scratch {
                    for lane in 0..LANES {
                        self.slot(inputs[0], size.bytes(), lane, acc);
                    }
                } else if self.has(self.of(inputs[0]), Some(Region::Kernarg)) {
                    match self.offset(inputs[0]) {
                        Some(at) if size.bytes() >= 4 => {
                            if let Some(&id) = self.bindings.get(&at) {
                                self.mark(acc, Some(Region::Allocation(id)));
                            }
                        }
                        Some(_) => {}
                        None => {
                            for &id in &self.allocations {
                                self.mark(acc, Some(Region::Allocation(id)));
                            }
                        }
                    }
                }
            }
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::ReadFirstLane),
                inputs,
                ..
            } => {
                merge(acc, self.of(inputs[0]));
            }
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::ReadLane),
                inputs,
                ..
            } => match self.constant(inputs[1]) {
                Some(k) => {
                    merge(acc, self.part(inputs[0], (k & 31) as usize));
                }
                None => {
                    merge(acc, self.of(inputs[0]));
                    if self.partial {
                        acc[0] |= 1;
                    }
                }
            },
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::WriteLane),
                ..
            } => {
                if let Some(parts) = self.split(v) {
                    for lane in 0..LANES {
                        merge(acc, &parts[lane * self.words..(lane + 1) * self.words]);
                    }
                }
            }
            Inst::Effect {
                op: EffectOp::Wave(op @ (WaveOp::Bpermute | WaveOp::BpermuteFi)),
                inputs,
                ..
            } => {
                merge(acc, self.of(inputs[1]));
                if *op == WaveOp::Bpermute || self.partial {
                    acc[0] |= 1;
                }
            }
            _ => acc[0] |= 1,
        }
    }

    fn store(&mut self, inst: &Inst) -> bool {
        let Inst::Effect {
            op: EffectOp::Memory {
                op,
                space: Space::Scratch,
                ..
            },
            inputs,
            ..
        } = inst
        else {
            return false;
        };
        if matches!(op, MemoryOp::Load(_) | MemoryOp::Fence) {
            return false;
        }
        let mut data = vec![0; self.words];
        for &d in &inputs[1..op.mask_input()] {
            merge(&mut data, self.of(d));
        }
        data[0] &= !1;
        if !points(&data) {
            return false;
        }
        let bytes = match op {
            MemoryOp::Store(size) => size.bytes(),
            _ => 4,
        };
        let w = self.words;
        match self.folded[inputs[0].0] {
            Some(t) => {
                let mut lanes = vec![0; LANES * w];
                for lane in 0..LANES {
                    for &d in &inputs[1..op.mask_input()] {
                        merge(&mut lanes[lane * w..(lane + 1) * w], self.part(d, lane));
                    }
                    lanes[lane * w] &= !1;
                }
                let mut changed = false;
                for word in t / 4..(t + bytes).div_ceil(4) {
                    changed |= merge(self.slots.entry(word).or_insert_with(|| vec![0; LANES * w]), &lanes);
                }
                changed
            }
            None => merge(&mut self.anywhere, &data),
        }
    }

    fn slot(&self, address: ValueId, bytes: u32, lane: usize, acc: &mut [u64]) {
        let w = self.words;
        merge(acc, &self.anywhere);
        match self.folded[address.0] {
            Some(t) => {
                for word in t / 4..(t + bytes).div_ceil(4) {
                    if let Some(slot) = self.slots.get(&word) {
                        merge(acc, &slot[lane * w..(lane + 1) * w]);
                    }
                }
            }
            None => {
                for slot in self.slots.values() {
                    merge(acc, &slot[lane * w..(lane + 1) * w]);
                }
            }
        }
    }

    fn exposed(&self, v: ValueId, op: Op, out: &mut [ValueId; 3]) -> usize {
        let wide = self.f.types[v.0] == Ty::I64;
        let high = |s: ValueId| wide && self.constant(s).is_some_and(|k| k >= 32);
        let list: &[ValueId] = match op {
            Op::Int(IntOp::And, a, b) => match (self.constant(a), self.constant(b)) {
                (None, Some(m)) | (Some(m), None) if aligns(m) => &[],
                _ => &[a, b],
            },
            Op::Int(IntOp::Mul | IntOp::Shl, a, b) | Op::Float(_, a, b) => &[a, b],
            Op::Int(IntOp::AShr, a, b) if !high(b) => &[a, b],
            Op::Int(IntOp::LShr, a, b) if wide && !high(b) => &[a, b],
            Op::Pack64(_, a) | Op::ReverseBits(a) | Op::Unary(_, a) => &[a],
            Op::Fma(a, b, c) | Op::MulAdd(a, b, c) => &[a, b, c],
            Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, ..) => &[],
            Op::Convert(_, _, a) => &[a],
            _ => &[],
        };
        out[..list.len()].copy_from_slice(list);
        list.len()
    }

    fn conjuncts(&self, c: ValueId) -> Vec<ValueId> {
        let mut list = Vec::new();
        let mut pending = vec![c];
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
        list
    }

    fn under(&self, x: ValueId, held: &[ValueId], memo: &mut HashMap<ValueId, Vec<u64>>) -> Vec<u64> {
        let x = self.copies.get(&x).copied().unwrap_or(x);
        if !points(self.of(x)) {
            return self.of(x).to_vec();
        }
        if let Some(found) = memo.get(&x) {
            return found.clone();
        }
        let (f, facts) = (self.f, self.facts);
        let mut acc = vec![0; self.words];
        match facts.op(f, x) {
            Some(Op::Select(c, a, b)) => {
                let c = self.copies.get(&c).copied().unwrap_or(c);
                if held.contains(&c) {
                    acc = self.under(a, held, memo);
                } else {
                    merge(&mut acc, &self.under(a, held, memo));
                    merge(&mut acc, &self.under(b, held, memo));
                }
            }
            Some(Op::Int(IntOp::Or | IntOp::Xor, a, b)) => {
                let (y, z) = (self.under(a, held, memo), self.under(b, held, memo));
                pointers(&mut acc, &y);
                pointers(&mut acc, &z);
                if y[0] & z[0] & 1 != 0 {
                    acc[0] |= 1;
                }
            }
            Some(Op::Int(IntOp::Add, a, b)) => {
                let (y, z) = (self.under(a, held, memo), self.under(b, held, memo));
                if z[0] & 1 != 0 {
                    pointers(&mut acc, &y);
                }
                if y[0] & 1 != 0 {
                    pointers(&mut acc, &z);
                }
                if y[0] & z[0] & 1 != 0 || (points(&y) && points(&z)) {
                    acc[0] |= 1;
                }
            }
            Some(Op::Int(IntOp::Sub, a, b)) => {
                let (y, z) = (self.under(a, held, memo), self.under(b, held, memo));
                if z[0] & 1 != 0 {
                    pointers(&mut acc, &y);
                }
                if y[0] & 1 != 0 || (points(&y) && points(&z)) {
                    acc[0] |= 1;
                }
            }
            Some(Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, to, a))
                if to.bits() >= 32 && f.types[a.0].bits() >= 32 =>
            {
                acc = self.under(a, held, memo);
            }
            Some(Op::Pack64(lo, _) | Op::UnpackLo(lo)) => acc = self.under(lo, held, memo),
            Some(Op::UnpackHi(y)) => match facts.op(f, y) {
                Some(Op::Pack64(_, hi)) => acc = self.under(hi, held, memo),
                _ => acc[0] |= 1,
            },
            Some(Op::Int(IntOp::And, a, b)) => match (self.constant(a), self.constant(b)) {
                (None, Some(m)) if aligns(m) => acc = self.under(a, held, memo),
                (Some(m), None) if aligns(m) => acc = self.under(b, held, memo),
                (None, None) => {
                    let (y, z) = (self.under(a, held, memo), self.under(b, held, memo));
                    if z[0] & 1 != 0 {
                        pointers(&mut acc, &y);
                    }
                    if y[0] & 1 != 0 {
                        pointers(&mut acc, &z);
                    }
                    acc[0] |= 1;
                }
                _ => acc[0] |= 1,
            },
            Some(Op::Int(IntOp::LShr, a, _)) if f.types[x.0] != Ty::I64 => {
                merge(&mut acc, &self.under(a, held, memo));
                acc[0] |= 1;
            }
            _ => acc.copy_from_slice(self.of(x)),
        }
        memo.insert(x, acc.clone());
        acc
    }

    fn update(&mut self, v: ValueId, acc: &[u64]) -> bool {
        let mut changed = merge(&mut self.bits[v.0 * self.words..(v.0 + 1) * self.words], acc);
        if let Some(parts) = self.split(v) {
            match self.parts.get_mut(&v) {
                Some(old) => changed |= merge(old, &parts),
                None => {
                    self.parts.insert(v, parts);
                    changed = true;
                }
            }
        }
        changed
    }

    fn region(&self, i: usize) -> Option<Region> {
        match FIXED.get(i) {
            Some(&r) => r,
            None => Some(Region::Allocation(self.allocations[i - FIXED.len()])),
        }
    }

    fn regions(&self, set: &[u64]) -> Regions {
        let mut out = Regions::default();
        for i in 0..FIXED.len() + self.allocations.len() {
            if set[i / 64] >> (i % 64) & 1 != 0 {
                out.add(self.region(i));
            }
        }
        out
    }
}

pub(super) fn bounds(
    f: &Func,
    facts: &Facts,
    copies: &Copies,
    users: &HashMap<ValueId, Vec<ValueId>>,
    inputs: &[Parameter],
    entry: &EntryLayout,
    env: &Environment,
    headers: &BTreeSet<BlockId>,
    registry: &DialectRegistry,
) -> Bounds {
    let mut sets = Sets::new(f, facts, copies, inputs, entry, env, headers);
    let mut acc = vec![0; sets.words];
    let mut loads: Vec<ValueId> = Vec::new();
    let mut writes: Vec<(BlockId, usize)> = Vec::new();
    let mut carried: Vec<(usize, usize, ValueId)> = Vec::new();
    let mut pending: Vec<ValueId> = Vec::new();
    for (r, &block) in facts.order.iter().enumerate() {
        for (index, &(v, _)) in f.blocks[&block].params.iter().enumerate() {
            if sets.incoming[r].iter().any(|&(args, back)| back && args[index] != v) {
                pending.push(v);
                if !copies.contains_key(&v) {
                    carried.push((r, index, v));
                }
            }
            sets.param(r, block, index, v, &mut acc);
            sets.update(v, &acc);
        }
        for (index, inst) in f.blocks[&block].insts.iter().enumerate() {
            if let Inst::Effect {
                op: EffectOp::Memory {
                    op,
                    space: Space::Scratch,
                    ..
                },
                outputs,
                ..
            } = inst
            {
                match op {
                    MemoryOp::Load(_) => loads.extend(outputs.iter().map(|o| o.0)),
                    MemoryOp::Fence => {}
                    _ => writes.push((block, index)),
                }
            }
            sets.store(inst);
            inst.for_each_output(|o| {
                sets.inst(inst, o, &mut acc);
                sets.update(o, &acc);
            });
        }
    }
    if !sets.slots.is_empty() || points(&sets.anywhere) {
        pending.extend(&loads);
    }
    while let Some(v) = pending.pop() {
        sets.eval(v, &mut acc);
        if !sets.update(v, &acc) {
            continue;
        }
        let mut memory = false;
        for &u in users.get(&v).into_iter().flatten() {
            if u == PRIVATE_MEMORY {
                memory = true;
                continue;
            }
            pending.push(u);
            if matches!(facts.op(f, u), Some(Op::Pack64(..))) {
                pending.extend(users.get(&u).into_iter().flatten().filter(|&&w| w != PRIVATE_MEMORY));
            }
        }
        if memory {
            let mut changed = false;
            for &(block, index) in &writes {
                changed |= sets.store(&f.blocks[&block].insts[index]);
            }
            if changed {
                pending.extend(&loads);
            }
        }
    }
    let mut bounds = HashMap::default();
    let mut plain = vec![false; f.types.len()];
    for (r, index, v) in carried {
        let back: Vec<ValueId> = sets.incoming[r]
            .iter()
            .filter(|&&(args, back)| back && args[index] != v)
            .map(|&(args, _)| args[index])
            .collect();
        acc.fill(0);
        for &a in &back {
            merge(&mut acc, sets.of(a));
        }
        if !points(&acc) {
            plain[v.0] = acc[0] & 1 != 0;
            continue;
        }
        let lanes = if back.iter().any(|a| sets.parts.contains_key(a)) { LANES } else { 1 };
        let mut parts = Vec::with_capacity(lanes);
        for lane in 0..lanes {
            acc.fill(0);
            for &a in &back {
                merge(&mut acc, sets.part(a, lane));
            }
            parts.push(sets.regions(&acc));
        }
        bounds.insert(v, parts);
    }
    let known = (0..f.types.len())
        .map(|x| {
            let set = sets.of(ValueId(x));
            (set.iter().map(|w| w.count_ones()).sum::<u32>() == 1).then(|| {
                let (word, bits) = set.iter().enumerate().find(|&(_, &w)| w != 0).unwrap();
                sets.region(word * 64 + bits.trailing_zeros() as usize)
            })
        })
        .collect();
    let used = uses(f, facts);
    let mut demand: Option<Vec<Demand>> = None;
    let mut wanted = vec![0; sets.words];
    for &id in &sets.allocations {
        if !env.exposed.contains(&id) {
            sets.mark(&mut wanted, Some(Region::Allocation(id)));
        }
    }
    let mut memos: HashMap<ValueId, (Vec<ValueId>, HashMap<ValueId, Vec<u64>>)> = HashMap::default();
    let mut exposing = Vec::new();
    let mut spills: Vec<Spill> = Vec::new();
    let mut buffer = [ValueId(0); 3];
    for &b in &facts.order {
        for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
            let (operands, lanewise, assume, scratch): (&[ValueId], _, _, _) = match inst {
                Inst::Core { value, op, .. } => {
                    let n = sets.exposed(*value, *op, &mut buffer);
                    (&buffer[..n], true, None, None)
                }
                Inst::Target { op, args, .. } if pure(registry, *op) => (args.values(), true, None, None),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Wmma),
                    inputs,
                    ..
                } => (&inputs[..], false, None, None),
                Inst::Effect {
                    op: EffectOp::Memory { op, space, .. },
                    inputs,
                    ..
                } if !matches!(op, MemoryOp::Load(_) | MemoryOp::Fence) => {
                    let m = op.mask_input();
                    let bytes = match op {
                        MemoryOp::Store(size) => size.bytes(),
                        _ => 4,
                    };
                    let scratch = (*space == Space::Scratch).then_some((inputs[0], bytes));
                    (&inputs[1..m], false, Some(inputs[m]), scratch)
                }
                _ => continue,
            };
            let reaching = |set: &[u64]| set.iter().zip(&wanted).any(|(x, y)| x & y != 0);
            let relevant = match scratch {
                Some(_) => operands.iter().any(|&x| points(sets.of(x))),
                None => operands.iter().any(|&x| reaching(sets.of(x))),
            };
            if !relevant {
                continue;
            }
            let mut assume = assume;
            if !matches!(inst, Inst::Effect { op: EffectOp::Memory { .. }, .. }) {
                let mut data = false;
                inst.for_each_output(|o| data |= used[o.0]);
                if !data {
                    continue;
                }
                let demand = demand.get_or_insert_with(|| demands(&sets, registry));
                let mut out = Demand::Dead;
                inst.for_each_output(|o| out = out.meet(demand[o.0]));
                match (out, lanewise) {
                    (Demand::Dead, _) => continue,
                    (Demand::Under(c), true) => assume = Some(c),
                    _ => {}
                }
            }
            acc.fill(0);
            for &x in operands {
                match assume {
                    Some(c) => {
                        let (held, memo) = memos.entry(c).or_insert_with(|| (sets.conjuncts(c), HashMap::default()));
                        merge(&mut acc, &sets.under(x, held, memo));
                    }
                    None => {
                        merge(&mut acc, sets.of(x));
                    }
                }
            }
            if let Some((address, bytes)) = scratch {
                acc[0] &= !1;
                if !points(&acc) {
                    continue;
                }
                let words = sets.folded[address.0].map(|t| (t / 4, (t + bytes).div_ceil(4)));
                let parts = if operands.iter().any(|x| sets.parts.contains_key(x)) {
                    (0..LANES)
                        .map(|lane| {
                            let mut part = vec![0; sets.words];
                            for &x in operands {
                                merge(&mut part, sets.part(x, lane));
                            }
                            for (p, a) in part.iter_mut().zip(&acc) {
                                *p &= a;
                            }
                            sets.regions(&part)
                        })
                        .collect()
                } else {
                    vec![sets.regions(&acc)]
                };
                spills.push(Spill {
                    at: (b, index),
                    words,
                    data: operands.to_vec(),
                    mask: assume.unwrap(),
                    parts,
                });
                continue;
            }
            let candidates: Vec<u64> = sets
                .allocations
                .iter()
                .copied()
                .filter(|&id| sets.has(&acc, Some(Region::Allocation(id))) && sets.has(&wanted, Some(Region::Allocation(id))))
                .collect();
            if !candidates.is_empty() {
                exposing.push(Exposure {
                    operands: operands.to_vec(),
                    assume,
                    candidates,
                });
            }
        }
    }
    let mut loaded = HashMap::default();
    for b in &facts.order {
        for inst in &f.blocks[b].insts {
            let Inst::Effect {
                op: EffectOp::Memory {
                    op: MemoryOp::Load(size),
                    space,
                    ..
                },
                inputs,
                outputs,
                ..
            } = inst
            else {
                continue;
            };
            let v = outputs[0].0;
            if *space != Space::Scratch {
                if points(sets.of(v)) {
                    let mut set = sets.regions(sets.of(v));
                    set.add(None);
                    loaded.insert(v, set);
                }
                continue;
            }
            let words = sets.folded[inputs[0].0].map(|t| (t / 4, (t + size.bytes()).div_ceil(4)));
            let from: Vec<&Spill> = spills
                .iter()
                .filter(|sp| match (words, sp.words) {
                    (Some((a, b)), Some((c, d))) => a < d && c < b,
                    _ => true,
                })
                .collect();
            let lanes = if from.iter().any(|sp| sp.parts.len() > 1) { LANES } else { 1 };
            let mut parts = Vec::with_capacity(lanes);
            let mut pointing = false;
            for lane in 0..lanes {
                let mut set = Regions::one(None);
                for sp in &from {
                    set.union(sp.part(lane));
                }
                pointing |= set != Regions::one(None);
                parts.push(set);
            }
            if pointing {
                bounds.insert(v, parts);
            }
        }
    }
    let mut written = HashMap::default();
    for &b in &facts.order {
        for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
            if let Inst::Effect {
                op:
                    EffectOp::Memory {
                        op,
                        space: Space::Global,
                        ..
                    },
                inputs,
                ..
            } = inst
            {
                if !matches!(op, MemoryOp::Load(_) | MemoryOp::Fence) {
                    written.insert((b, index), sets.regions(sets.of(inputs[0])));
                }
            }
        }
    }
    Bounds {
        known,
        plain,
        carried: bounds,
        loaded,
        spills,
        exposing,
        written,
    }
}

fn pure(registry: &DialectRegistry, op: TargetOp) -> bool {
    registry.operation(op).map_or(true, |spec| spec.effect == Effect::Pure)
}

fn demands(sets: &Sets, registry: &DialectRegistry) -> Vec<Demand> {
    let (f, facts) = (sets.f, sets.facts);
    let mut demand = vec![Demand::Dead; f.types.len()];
    let mut held: HashMap<ValueId, Vec<ValueId>> = HashMap::default();
    loop {
        let mut changed = false;
        for b in facts.order.iter().rev() {
            let block = &f.blocks[b];
            for edge in block.term.edges() {
                for (&x, &(p, _)) in edge.args.iter().zip(&f.blocks[&edge.dst].params) {
                    let d = demand[p.0];
                    changed |= use_as(&mut demand, x, d);
                }
            }
            for inst in block.insts.iter().rev() {
                let mut out = Demand::Dead;
                inst.for_each_output(|o| out = out.meet(demand[o.0]));
                match inst {
                    Inst::Core { ty: Ty::I1, .. }
                    | Inst::Core {
                        op: Op::TrailingZeros(_) | Op::LeadingZeros(_) | Op::PopulationCount(_),
                        ..
                    } => {}
                    Inst::Core {
                        op: Op::Select(c, a, b),
                        ..
                    } => {
                        let arm = if out == Demand::Dead { out } else { Demand::Under(*c) };
                        let other = match out {
                            Demand::Under(k) => {
                                let root = sets.copies.get(c).copied().unwrap_or(*c);
                                let conjuncts = held.entry(k).or_insert_with(|| sets.conjuncts(k));
                                if conjuncts.contains(&root) {
                                    Demand::Dead
                                } else {
                                    out
                                }
                            }
                            _ => out,
                        };
                        changed |= use_as(&mut demand, *a, arm);
                        changed |= use_as(&mut demand, *b, other);
                    }
                    Inst::Core { .. } => inst.for_each_operand(|x| changed |= use_as(&mut demand, x, out)),
                    Inst::Target { op, .. } if pure(registry, *op) => {
                        inst.for_each_operand(|x| changed |= use_as(&mut demand, x, out))
                    }
                    Inst::Effect {
                        op: EffectOp::Memory { op, .. },
                        inputs,
                        ..
                    } if inputs.len() > op.mask_input() => {
                        let m = op.mask_input();
                        for &x in &inputs[..m] {
                            changed |= use_as(&mut demand, x, Demand::Under(inputs[m]));
                        }
                    }
                    _ => inst.for_each_operand(|x| changed |= use_as(&mut demand, x, Demand::All)),
                }
            }
        }
        if !changed {
            return demand;
        }
    }
}

fn uses(f: &Func, facts: &Facts) -> Vec<bool> {
    let mut used = vec![false; f.types.len()];
    for b in &facts.order {
        let block = &f.blocks[b];
        for edge in block.term.edges() {
            for &x in &edge.args {
                used[x.0] = true;
            }
        }
        for inst in &block.insts {
            match inst {
                Inst::Core { ty: Ty::I1, .. }
                | Inst::Core {
                    op: Op::TrailingZeros(_) | Op::LeadingZeros(_) | Op::PopulationCount(_),
                    ..
                } => {}
                Inst::Core {
                    op: Op::Select(_, a, b),
                    ..
                } => {
                    used[a.0] = true;
                    used[b.0] = true;
                }
                Inst::Effect {
                    op: EffectOp::Memory { op, .. },
                    inputs,
                    ..
                } if inputs.len() > op.mask_input() => {
                    for &x in &inputs[..op.mask_input()] {
                        used[x.0] = true;
                    }
                }
                _ => inst.for_each_operand(|x| used[x.0] = true),
            }
        }
    }
    used
}

fn use_as(demand: &mut [Demand], x: ValueId, d: Demand) -> bool {
    let met = demand[x.0].meet(d);
    let changed = met != demand[x.0];
    demand[x.0] = met;
    changed
}

fn merge(into: &mut [u64], from: &[u64]) -> bool {
    let mut changed = false;
    for (x, &y) in into.iter_mut().zip(from) {
        changed |= *x | y != *x;
        *x |= y;
    }
    changed
}

fn pointers(into: &mut [u64], from: &[u64]) {
    into[0] |= from[0] & !1;
    merge(&mut into[1..], &from[1..]);
}

fn points(set: &[u64]) -> bool {
    set[0] & !1 != 0 || set[1..].iter().any(|&w| w != 0)
}
