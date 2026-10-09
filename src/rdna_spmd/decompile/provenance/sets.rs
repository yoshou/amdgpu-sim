use super::super::address::{aligns, Region, PRIVATE_MEMORY};
use super::layout::{merge, pointers, points, Layout};
use super::program::Program;
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) struct Sets<'p, 'a> {
    program: &'p Program<'a>,
    layout: &'p Layout,
    words: usize,
    bits: Vec<u64>,
    slots: HashMap<u32, Vec<u64>>,
    anywhere: Vec<u64>,
    parts: HashMap<ValueId, Vec<u64>>,
}

impl<'p, 'a> Sets<'p, 'a> {
    pub(super) fn solve(
        program: &'p Program<'a>,
        layout: &'p Layout,
        users: &HashMap<ValueId, Vec<ValueId>>,
    ) -> Self {
        let (f, facts) = (program.f, program.facts);
        let words = layout.words();
        let mut sets = Self {
            program,
            layout,
            words,
            bits: vec![0; f.types.len() * words],
            slots: HashMap::default(),
            anywhere: vec![0; words],
            parts: HashMap::default(),
        };
        let mut acc = vec![0; words];
        let mut loads: Vec<ValueId> = Vec::new();
        let mut writes: Vec<(BlockId, usize)> = Vec::new();
        let mut pending: Vec<ValueId> = program.carried.iter().map(|&(_, _, v)| v).collect();
        for (r, &block) in facts.order.iter().enumerate() {
            for (index, &(v, _)) in f.blocks[&block].params.iter().enumerate() {
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
        sets
    }

    #[inline]
    pub(super) fn program(&self) -> &'p Program<'a> {
        self.program
    }

    #[inline]
    pub(super) fn layout(&self) -> &'p Layout {
        self.layout
    }

    #[inline]
    pub(super) fn words(&self) -> usize {
        self.words
    }

    pub(super) fn of(&self, x: ValueId) -> &[u64] {
        &self.bits[x.0 * self.words..(x.0 + 1) * self.words]
    }

    pub(super) fn part(&self, x: ValueId, lane: usize) -> &[u64] {
        match self.parts.get(&x) {
            Some(parts) => &parts[lane * self.words..(lane + 1) * self.words],
            None => self.of(x),
        }
    }

    #[inline]
    pub(super) fn has_parts(&self, x: ValueId) -> bool {
        self.parts.contains_key(&x)
    }

    fn split(&self, v: ValueId) -> Option<Vec<u64>> {
        let w = self.words;
        match self.program.facts.site[v.0] {
            Site::Param { .. } if self.parts.is_empty() => None,
            Site::Param { block, index } if block != self.program.f.entry => {
                if let Some(root) = self.program.copies.get(&v) {
                    return self.parts.get(root).cloned();
                }
                let incoming = &self.program.incoming[self.program.rank[&block]];
                if !incoming.iter().any(|&(args, _)| self.parts.contains_key(&args[index])) {
                    return None;
                }
                let mut out = vec![0; self.program.lanes() * w];
                for &(args, _) in incoming {
                    for lane in 0..self.program.lanes() {
                        merge(&mut out[lane * w..(lane + 1) * w], self.part(args[index], lane));
                    }
                }
                Some(out)
            }
            Site::Inst { block, index } => match &self.program.f.blocks[&block].insts[index] {
                Inst::Core {
                    op: Op::Select(_, a, b),
                    ..
                } if self.parts.contains_key(a) || self.parts.contains_key(b) => {
                    let mut out = vec![0; self.program.lanes() * w];
                    for lane in 0..self.program.lanes() {
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
                    let mut out = vec![0; self.program.lanes() * w];
                    for lane in 0..self.program.lanes() {
                        out[lane * w..(lane + 1) * w].copy_from_slice(self.part(inputs[2], lane));
                    }
                    match self.program.constant(inputs[1]) {
                        Some(k) => {
                            let k = (k & (self.program.lanes() as u32 - 1)) as usize;
                            out[k * w..(k + 1) * w].copy_from_slice(self.of(inputs[0]));
                        }
                        None => {
                            for lane in 0..self.program.lanes() {
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
                    let mut out = vec![0; self.program.lanes() * w];
                    for lane in 0..self.program.lanes() {
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

    fn eval(&self, v: ValueId, acc: &mut [u64]) {
        match self.program.facts.site[v.0] {
            Site::Param { block, index } => self.param(self.program.rank[&block], block, index, v, acc),
            Site::Inst { block, index } => self.inst(&self.program.f.blocks[&block].insts[index], v, acc),
            Site::Unreached => {
                acc.fill(0);
                acc[0] = 1;
            }
        }
    }

    fn param(&self, rank: usize, block: BlockId, index: usize, v: ValueId, acc: &mut [u64]) {
        acc.fill(0);
        if let Some(&root) = self.program.copies.get(&v) {
            merge(acc, self.of(root));
            return;
        }
        if block == self.program.f.entry {
            let r = match self.program.inputs[index].source {
                ParameterSource::Sgpr(n) if Some(n) == self.program.entry.kernarg_ptr => Some(Region::Kernarg),
                ParameterSource::Sgpr(n) if Some(n) == self.program.entry.dispatch_ptr => Some(Region::Dispatch),
                _ => None,
            };
            self.layout.mark(acc, r);
            return;
        }
        for &(args, _) in &self.program.incoming[rank] {
            merge(acc, self.of(args[index]));
        }
    }

    fn inst(&self, inst: &Inst, v: ValueId, acc: &mut [u64]) {
        acc.fill(0);
        let (f, facts) = (self.program.f, self.program.facts);
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
                    if y[0] & 1 != 0 || (points(x) && points(y)) {
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
                Op::Int(IntOp::And, a, b) => match (self.program.constant(a), self.program.constant(b)) {
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
                Op::Env(Env::ScratchBase) => self.layout.mark(acc, Some(Region::Private)),
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
                    for lane in 0..self.program.lanes() {
                        self.slot(inputs[0], size.bytes(), lane, acc);
                    }
                } else if self.layout.has(self.of(inputs[0]), Some(Region::Kernarg)) {
                    match self.program.offset(inputs[0]) {
                        Some(at) if size.bytes() >= 4 => {
                            if let Some(&id) = self.program.bindings.get(&at) {
                                self.layout.mark(acc, Some(Region::Allocation(id)));
                            }
                        }
                        None if size.bytes() >= 4 => {
                            for &id in self.layout.allocations() {
                                self.layout.mark(acc, Some(Region::Allocation(id)));
                            }
                        }
                        _ => {}
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
            } => match self.program.constant(inputs[1]) {
                Some(k) => {
                    let k = (k & (self.program.lanes() as u32 - 1)) as usize;
                    merge(acc, self.part(inputs[0], k));
                    if self.program.lacks(k) {
                        acc[0] |= 1;
                    }
                }
                None => {
                    merge(acc, self.of(inputs[0]));
                    if self.program.partial() {
                        acc[0] |= 1;
                    }
                }
            },
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::WriteLane),
                ..
            } => {
                if let Some(parts) = self.split(v) {
                    for lane in 0..self.program.lanes() {
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
                if *op == WaveOp::Bpermute || self.program.partial() {
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
        match self.program.words(inputs[0], bytes) {
            Some((first, end)) => {
                let count = self.program.lanes();
                let mut lanes = vec![0; count * w];
                for lane in 0..count {
                    for &d in &inputs[1..op.mask_input()] {
                        merge(&mut lanes[lane * w..(lane + 1) * w], self.part(d, lane));
                    }
                    lanes[lane * w] &= !1;
                }
                let mut changed = false;
                for word in first..end {
                    changed |= merge(self.slots.entry(word).or_insert_with(|| vec![0; count * w]), &lanes);
                }
                changed
            }
            None => merge(&mut self.anywhere, &data),
        }
    }

    fn slot(&self, address: ValueId, bytes: u32, lane: usize, acc: &mut [u64]) {
        let w = self.words;
        merge(acc, &self.anywhere);
        match self.program.words(address, bytes) {
            Some((first, end)) => {
                for word in first..end {
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
}
