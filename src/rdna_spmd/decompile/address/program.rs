use super::super::provenance::{bounds, Bounds};
use super::form::*;
use super::graph::*;
use super::limits::*;
use super::HashSet;
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::engine::EntryLayout;
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

pub(super) type ImplicationKey = (Implication, ValueId, ValueId, Option<(ValueId, bool)>);

pub(super) struct Program<'a> {
    pub(super) f: &'a Func,
    pub(super) facts: &'a Facts,
    pub(super) inputs: &'a [Parameter],
    pub(super) exec: u32,
    pub(super) exec_index: Option<usize>,
    pub(super) entry: EntryLayout,
    pub(super) env: &'a Environment,
    pub(super) headers: BTreeSet<BlockId>,
    pub(super) rank: HashMap<BlockId, usize>,
    pub(super) idom: Vec<usize>,
    pub(super) loops: HashMap<BlockId, Vec<BlockId>>,
    pub(super) copies: Copies,
    pub(super) stores: HashMap<BlockId, Vec<Store>>,
    pub(super) narrowing_edges: HashSet<(BlockId, usize)>,
    pub(super) narrowable: Vec<bool>,
    pub(super) guards: Vec<((BlockId, usize), Option<ValueId>)>,
    pub(super) assumable: HashSet<ValueId>,
    pub(super) equated: HashSet<ValueId>,
    pub(super) provenance: Bounds,
    pub(super) sources: HashMap<ValueId, Vec<usize>>,
}

pub(super) struct Conditions {
    trees: std::cell::RefCell<Vec<Option<(std::rc::Rc<Cond>, bool)>>>,
    implications: std::cell::RefCell<HashMap<ImplicationKey, bool>>,
    implied: HashMap<ValueId, Vec<ValueId>>,
    safe_steps: HashMap<ValueId, bool>,
    readbacks: HashMap<(BlockId, usize), Option<WrittenBack>>,
}

pub(super) type WrittenBack = (ValueId, Option<ValueId>, Option<u64>);

impl<'a> Program<'a> {
    pub(super) fn new(
        f: &'a Func,
        facts: &'a Facts,
        inputs: &'a [Parameter],
        exec: u32,
        entry: EntryLayout,
        env: &'a Environment,
        headers: BTreeSet<BlockId>,
        registry: &DialectRegistry,
        conditions: &mut Conditions,
    ) -> Self {
        let idom = dominators(f, facts);
        let reaches = reaches(f, facts);
        let loops = loops(facts, &idom, &reaches);
        let copies = copies(f, facts);
        let provenance = bounds(f, facts, &copies, &users(f, facts), inputs, &entry, env, &headers, registry);
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
            idom,
            loops,
            copies,
            stores: private_stores(f, facts),
            narrowing_edges: HashSet::default(),
            narrowable: Vec::new(),
            guards: Vec::new(),
            assumable: HashSet::default(),
            equated: HashSet::default(),
            provenance,
            sources: HashMap::default(),
        };
        let mut narrowing = HashSet::default();
        for &b in &facts.order {
            if let Term::CondBr { cond, .. } = f.blocks[&b].term {
                for slot in 0..2 {
                    if !conditions.fixed_words(&this, cond, slot == 0).is_empty() {
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
        for &b in &facts.order {
            for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
                let guard = match inst {
                    Inst::Effect { op: EffectOp::Memory { op: MemoryOp::Fence, .. }, .. } => continue,
                    Inst::Effect { op: EffectOp::Memory { op, .. }, inputs, .. } => inputs.get(op.mask_input()).copied(),
                    Inst::Effect { op: EffectOp::Wave(WaveOp::Any | WaveOp::Ballot { .. }), inputs, .. } => Some(inputs[0]),
                    Inst::Effect { .. } => None,
                    Inst::Target { op, args, .. } => match registry.operation(*op).map(|spec| spec.effect) {
                        Ok(Effect::Pure) | Err(_) => continue,
                        Ok(Effect::ReadGlobal { every_lane: false }) => args.values().iter().copied().find(|a| f.types[a.0] == Ty::I1),
                        Ok(_) => None,
                    },
                    _ => continue,
                };
                this.guards.push(((b, index), guard));
            }
        }
        this.narrowable = narrowable;
        for &v in this.provenance.carried.keys() {
            let Site::Inst { block, index } = facts.site[v.0] else {
                continue;
            };
            let at = this.rank[&block];
            let from: Vec<usize> = (0..this.provenance.spills.len())
                .filter(|&i| {
                    let (b, k) = this.provenance.spills[i].at;
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
                        let implied = conditions.assumptions(&this, p);
                        for (x, _) in conditions.fixed_words(&this, p, true) {
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

    pub(super) fn lanes(&self) -> usize {
        self.f.lanes as usize
    }

    pub(super) fn block_of(&self, v: ValueId) -> BlockId {
        match self.facts.site[v.0] {
            Site::Param { block, .. } | Site::Inst { block, .. } => block,
            Site::Unreached => self.f.entry,
        }
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

    pub(super) fn exposable(&self) -> BTreeSet<u64> {
        self.provenance.exposing.iter().flat_map(|e| e.candidates.iter().copied()).collect()
    }

    pub(super) fn carried(&self, v: ValueId, block: BlockId, index: usize) -> bool {
        let own = self.rank[&block];
        !self.copies.contains_key(&v)
            && self.headers.contains(&block)
            && self.facts.incoming[&block]
                .iter()
                .any(|&(pred, slot)| self.rank[&pred] >= own && self.edge_arg((pred, slot), index) != v)
    }

    pub(super) fn converted(&self, k: Cvt, to: Ty, a: ValueId) -> Option<u64> {
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

    pub(super) fn summed_high(&self, x: ValueId, seen: &mut HashMap<ValueId, bool>) -> bool {
        let x = self.copies.get(&x).copied().unwrap_or(x);
        if let Some(&known) = seen.get(&x) {
            return known;
        }
        seen.insert(x, false);
        let known = self.plain_high(x)
            || matches!(self.facts.op(self.f, x), Some(Op::Int(IntOp::Add | IntOp::Sub | IntOp::And | IntOp::Or | IntOp::Xor, a, b) | Op::Select(_, a, b)) if self.summed_high(a, seen) && self.summed_high(b, seen));
        seen.insert(x, known);
        known
    }

    pub(super) fn written_back(&self, block: BlockId, index: usize) -> Option<(ValueId, Option<ValueId>, Option<u64>)> {
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
        let set = self.provenance.written.get(&(store_block, store))?;
        let [Some(Region::Allocation(target))] = set.list.iter().filter(|r| r.is_some()).copied().collect::<Vec<_>>()[..] else {
            return None;
        };
        if set.any || self.env.exposed.contains(&target) || self.exposable().contains(&target) {
            return None;
        }
        let region = Some(Region::Allocation(target));
        let alone = self.provenance.written.iter().all(|(&at, set)| at == (store_block, store) || (!set.any && !set.list.contains(&region)));
        alone.then_some((data, base, Some(target)))
    }

    fn determined(&self, address: ValueId, data: ValueId, wide: bool) -> Option<Option<ValueId>> {
        let (f, facts) = (self.f, self.facts);
        let root = |x: ValueId| self.copies.get(&x).copied().unwrap_or(x);
        let top = |x: ValueId| -> Option<u64> {
            match (facts.op(f, x), facts.site[x.0]) {
                (Some(Op::Env(Env::LaneId)), _) => Some(f.lanes as u64 - 1),
                (_, Site::Param { block, index }) if block == f.entry && matches!(self.inputs[index].source, ParameterSource::Vgpr(r) if self.entry.workitem_register(r)) => {
                    let ParameterSource::Vgpr(r) = self.inputs[index].source else { unreachable!() };
                    let tops = self.env.block.map(|n| n.max(1) as u64 - 1);
                    Some(self.entry.workitem_fields(r).fold(0, |sum, (axis, shift)| sum | tops[axis] << shift))
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

    pub(super) fn dispatch_word(&self, at: u32, bytes: u32) -> Option<u32> {
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

    pub(super) fn source(&self, x: ValueId, lane: usize) -> (ValueId, usize) {
        let (mut x, mut lane) = (x, lane);
        loop {
            x = self.copies.get(&x).copied().unwrap_or(x);
            match self.facts.inst(self.f, x) {
                Some(Inst::Effect {
                    op: EffectOp::Wave(WaveOp::WriteLane),
                    inputs,
                    ..
                }) => match self.facts.constant(self.f, inputs[1]) {
                    Some(k) if (k as usize) & (self.lanes() - 1) == lane => x = inputs[0],
                    Some(_) => x = inputs[2],
                    None => break,
                },
                Some(Inst::Effect {
                    op: EffectOp::Wave(WaveOp::ReadLane),
                    inputs,
                    ..
                }) => match self.facts.constant(self.f, inputs[1]) {
                    Some(k) => {
                        lane = (k as usize) & (self.lanes() - 1);
                        x = inputs[0];
                    }
                    None => break,
                },
                _ => break,
            }
        }
        (x, lane)
    }

    pub(super) fn increment(&self, x: ValueId, v: ValueId, lane: usize) -> Option<ValueId> {
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

    pub(super) fn edge_arg(&self, (pred, slot): (BlockId, usize), index: usize) -> ValueId {
        self.f.blocks[&pred].term.edges().nth(slot).unwrap().args[index]
    }

    pub(super) fn decided(&self, c: ValueId, edge: Option<(ValueId, bool)>) -> Option<bool> {
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

    pub(super) fn guard(&self, pred: BlockId, slot: usize) -> Option<(ValueId, bool)> {
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

    pub(super) fn edge_condition(&self, pred: BlockId, slot: usize) -> Option<(ValueId, bool)> {
        match self.f.blocks[&pred].term {
            Term::CondBr { cond, .. } => Some((cond, slot == 0)),
            _ => None,
        }
    }
}

impl Conditions {
    pub(super) fn new(f: &Func) -> Self {
        Self {
            trees: std::cell::RefCell::new(vec![None; 2 * f.types.len()]),
            implications: std::cell::RefCell::new(HashMap::default()),
            implied: HashMap::default(),
            safe_steps: HashMap::default(),
            readbacks: HashMap::default(),
        }
    }

    pub(super) fn written_back(&mut self, program: &Program, block: BlockId, index: usize) -> Option<WrittenBack> {
        match self.readbacks.get(&(block, index)) {
            Some(&found) => found,
            None => {
                let found = program.written_back(block, index);
                self.readbacks.insert((block, index), found);
                found
            }
        }
    }

    pub(super) fn condition(&self, program: &Program, cond: ValueId, taken: bool) -> (std::rc::Rc<Cond>, bool) {
        let slot = 2 * cond.0 + taken as usize;
        if let Some(found) = &self.trees.borrow()[slot] {
            return found.clone();
        }
        let tree = condition(program.f, program.facts, &program.copies, cond, taken);
        let found = (tree.clone(), shares(&tree));
        self.trees.borrow_mut()[slot] = Some(found.clone());
        found
    }

    pub(super) fn literals(&self, program: &Program, cond: ValueId, taken: bool) -> Vec<(ValueId, bool)> {
        let mut out = Vec::new();
        literals(&self.condition(program, cond, taken).0, &mut out);
        out
    }

    pub(super) fn fixed_words(&self, program: &Program, cond: ValueId, taken: bool) -> Vec<(ValueId, u32)> {
        self.literals(program, cond, taken).into_iter().filter_map(|l| program.fixed_by(l)).collect()
    }

    pub(super) fn assumed_constant(&mut self, program: &Program, a: ValueId, v: ValueId) -> Option<u32> {
        let root = program.copies.get(&v).copied().unwrap_or(v);
        if !program.equated.contains(&root) {
            return None;
        }
        self.fixed_words(program, a, true).into_iter().find(|&(x, _)| x == root).map(|(_, k)| k)
    }

    pub(super) fn assumes(&mut self, program: &Program, a: ValueId, v: ValueId) -> bool {
        let known = match self.implied.get(&a) {
            Some(list) => list.contains(&v),
            None => self.assumptions(program, a).contains(&v),
        };
        known || self.implies(program, a, v, None)
    }

    fn assumptions(&mut self, program: &Program, predicate: ValueId) -> Vec<ValueId> {
        if let Some(list) = self.implied.get(&predicate) {
            return list.clone();
        }
        let list: Vec<ValueId> = self.literals(program, predicate, true).into_iter().filter(|l| l.1).map(|l| l.0).collect();
        self.implied.insert(predicate, list.clone());
        list
    }

    fn equal_on_edge(&self, program: &Program, cond: ValueId, taken: bool, arg: ValueId) -> Option<u32> {
        let arg = program.copies.get(&arg).copied().unwrap_or(arg);
        self.fixed_words(program, cond, taken).into_iter().find(|&(x, _)| x == arg).map(|(_, k)| k)
    }

    pub(super) fn narrowing(&self, program: &Program, (pred, slot): (BlockId, usize), arg: ValueId) -> Option<u32> {
        if !program.narrowing_edges.contains(&(pred, slot)) {
            return None;
        }
        let (cond, taken) = program.edge_condition(pred, slot)?;
        self.equal_on_edge(program, cond, taken, arg)
    }

    pub(super) fn step_safe(&mut self, program: &Program, v: ValueId, header: BlockId) -> bool {
        if let Some(&safe) = self.safe_steps.get(&v) {
            return safe;
        }
        let safe = self.compute_step_safe(program, v, header);
        self.safe_steps.insert(v, safe);
        safe
    }

    fn compute_step_safe(&mut self, program: &Program, v: ValueId, header: BlockId) -> bool {
        let Some(i) = program.exec_index else {
            return false;
        };
        let (f, facts) = (program.f, program.facts);
        let exec = f.blocks[&header].params[i].0;
        let own = program.rank[&header];
        let masks: Vec<ValueId> = facts.incoming[&header]
            .iter()
            .filter(|&&(pred, _)| program.rank[&pred] >= own)
            .map(|&e| program.edge_arg(e, i))
            .collect();
        if !masks.iter().all(|&m| self.within(program, m, exec, &mut Vec::new())) {
            return false;
        }
        let mut depends = vec![false; f.types.len()];
        depends[v.0] = true;
        loop {
            let mut grew = false;
            for &b in &facts.order {
                if b != f.entry {
                    for (index, &(p, _)) in f.blocks[&b].params.iter().enumerate() {
                        if !depends[p.0] && facts.incoming[&b].iter().any(|&e| depends[program.edge_arg(e, index).0]) {
                            depends[p.0] = true;
                            grew = true;
                        }
                    }
                }
                for inst in &f.blocks[&b].insts {
                    let mut any = false;
                    inst.for_each_operand(|x| any |= depends[x.0]);
                    if any {
                        for out in inst.outputs() {
                            if !depends[out.0] {
                                depends[out.0] = true;
                                grew = true;
                            }
                        }
                    }
                }
            }
            if !grew {
                break;
            }
        }
        let guards = program.guards.clone();
        guards.iter().all(|&((b, index), guard)| {
            let mut any = false;
            f.blocks[&b].insts[index].for_each_operand(|x| any |= depends[x.0]);
            !any || guard.is_some_and(|g| self.within(program, g, exec, &mut Vec::new()))
        })
    }

    pub(super) fn within(&mut self, program: &Program, a: ValueId, mask: ValueId, visiting: &mut Vec<ValueId>) -> bool {
        let mask = program.copies.get(&mask).copied().unwrap_or(mask);
        for c in self.assumptions(program, a) {
            if c == mask {
                return true;
            }
            let Site::Param { block, index } = program.facts.site[c.0] else {
                continue;
            };
            if block == program.f.entry {
                continue;
            }
            if visiting.contains(&c) {
                return true;
            }
            visiting.push(c);
            let edges = program.facts.incoming[&block].clone();
            let carried = edges.iter().all(|&e| {
                let arg = program.edge_arg(e, index);
                self.within(program, arg, mask, visiting)
            });
            visiting.pop();
            if carried {
                return true;
            }
        }
        false
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

    pub(super) fn implies(&self, program: &Program, a: ValueId, v: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        let a = program.copies.get(&a).copied().unwrap_or(a);
        let v = program.copies.get(&v).copied().unwrap_or(v);
        if a == v {
            return true;
        }
        self.remembered(Implication::Bit, a, v, edge, |this| this.derives(program, a, v, edge))
    }

    fn derives(&self, program: &Program, a: ValueId, v: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        match program.facts.op(program.f, a) {
            Some(Op::Int(IntOp::And, x, y)) => {
                self.implies(program, x, v, edge) || self.implies(program, y, v, edge)
            }
            Some(Op::Select(c, x, y)) => match program.decided(c, edge) {
                Some(taken) => self.implies(program, if taken { x } else { y }, v, edge),
                None => self.implies(program, x, v, edge) && self.implies(program, y, v, edge),
            },
            Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => match program.facts.op(program.f, shifted) {
                Some(Op::Int(IntOp::LShr, w, s)) if program.facts.is_lane_shift(program.f, w, s) => {
                    self.word_holds(program, w, v, edge)
                }
                _ => false,
            },
            _ => false,
        }
    }

    fn word_holds(&self, program: &Program, w: ValueId, v: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        let w = program.copies.get(&w).copied().unwrap_or(w);
        let v = program.copies.get(&v).copied().unwrap_or(v);
        self.remembered(Implication::Holds, w, v, edge, |this| this.holds(program, w, v, edge))
    }

    pub(super) fn holds(&self, program: &Program, w: ValueId, v: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        match program.facts.inst(program.f, w) {
            Some(Inst::Effect {
                op: EffectOp::Wave(WaveOp::Ballot { high: false }),
                inputs,
                ..
            }) if program.f.lanes == 32 => self.implies(program, inputs[0], v, edge),
            Some(Inst::Core { op, .. }) => match *op {
                Op::Int(IntOp::And, x, y) => {
                    self.word_holds(program, x, v, edge) || self.word_holds(program, y, v, edge)
                }
                Op::Int(IntOp::Or, x, y) => {
                    self.word_holds(program, x, v, edge) && self.word_holds(program, y, v, edge)
                }
                Op::Select(c, x, y) => match program.decided(c, edge) {
                    Some(taken) => self.word_holds(program, if taken { x } else { y }, v, edge),
                    None => self.word_holds(program, x, v, edge) && self.word_holds(program, y, v, edge),
                },
                Op::Convert(Cvt::Bitcast, _, x) => self.word_holds(program, x, v, edge),
                Op::Pack64(low, high) if program.f.lanes == 64 => {
                    self.half_holds(program, low, false, v, edge) && self.half_holds(program, high, true, v, edge)
                }
                Op::Const(_, 0) => true,
                _ => false,
            },
            _ => false,
        }
    }

    fn half_holds(&self, program: &Program, w: ValueId, high: bool, v: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        let w = program.copies.get(&w).copied().unwrap_or(w);
        let v = program.copies.get(&v).copied().unwrap_or(v);
        self.remembered(Implication::Half(high), w, v, edge, |this| match program.facts.inst(program.f, w) {
            Some(Inst::Effect {
                op: EffectOp::Wave(WaveOp::Ballot { high: half }),
                inputs,
                ..
            }) if *half == high => this.implies(program, inputs[0], v, edge),
            Some(Inst::Core { op, .. }) => match *op {
                Op::Int(IntOp::And, x, y) => {
                    this.half_holds(program, x, high, v, edge) || this.half_holds(program, y, high, v, edge)
                }
                Op::Int(IntOp::Or, x, y) => {
                    this.half_holds(program, x, high, v, edge) && this.half_holds(program, y, high, v, edge)
                }
                Op::Select(c, x, y) => match program.decided(c, edge) {
                    Some(taken) => this.half_holds(program, if taken { x } else { y }, high, v, edge),
                    None => this.half_holds(program, x, high, v, edge) && this.half_holds(program, y, high, v, edge),
                },
                Op::Convert(Cvt::Bitcast, _, x) => this.half_holds(program, x, high, v, edge),
                Op::Const(_, 0) => true,
                _ => false,
            },
            _ => false,
        })
    }

    pub(super) fn word_implies(&self, program: &Program, a: ValueId, w: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        let a = program.copies.get(&a).copied().unwrap_or(a);
        let w = program.copies.get(&w).copied().unwrap_or(w);
        if a == w {
            return true;
        }
        self.remembered(Implication::Word, a, w, edge, |this| this.narrows(program, a, w, edge))
    }

    fn narrows(&self, program: &Program, a: ValueId, w: ValueId, edge: Option<(ValueId, bool)>) -> bool {
        match program.facts.op(program.f, a) {
            Some(Op::Int(IntOp::And, x, y)) => {
                self.word_implies(program, x, w, edge) || self.word_implies(program, y, w, edge)
            }
            Some(Op::Select(c, x, y)) => match program.decided(c, edge) {
                Some(taken) => self.word_implies(program, if taken { x } else { y }, w, edge),
                None => self.word_implies(program, x, w, edge) && self.word_implies(program, y, w, edge),
            },
            Some(Op::Convert(Cvt::Bitcast, _, x)) => self.word_implies(program, x, w, edge),
            _ => false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Implication {
    Bit,
    Holds,
    Half(bool),
    Word,
}
