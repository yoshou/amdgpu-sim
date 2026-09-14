use super::*;
use super::super::analysis::memory::Form;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::rdna_spmd) enum Lanes {
    Gather,
    Broadcast,
    Lds,
    Affine { allocated: bool },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::rdna_spmd) enum StoreShape {
    Narrow,
    Lds,
    Affine,
    Tile,
    Scatter,
}

pub(super) fn transpose_tile(width: u32) -> u32 {
    width.min(super::super::host::Vectors::detect().lanes(64))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::rdna_spmd) enum GlobalLoad { Gather, Broadcast, Frame { stride_words: u32, offset_words: u32 } }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::rdna_spmd) enum Shape {
    Fence,
    ScalarWords,
    AtomicAdd { grouped: bool },
    Store(StoreShape),
    NarrowLoad,
    PrivateTile { tile: u32 },
    Frame { stride_words: u32, offset_words: u32, group: u32 },
    Words { lanes: Lanes, pairs: bool },
    ScalarLoad { pairs: bool },
    ScalarStore,
    ScalarAtomic,
}

pub(in crate::rdna_spmd) fn global_load(access: &Access, uniform: &[bool], affine: &BTreeMap<ValueId, u32>) -> GlobalLoad {
    if std::env::var("AMDGPU_SIM_DEBUG_SHAPE").map_or(false, |v| v.contains("gather")) { return GlobalLoad::Gather; }
    if !matches!(access.form, Form::Global { .. }) { return GlobalLoad::Gather; }
    if !matches!(access.op, MemoryOp::Load(MemSize::B32 | MemSize::B64)) || !(1..=4).contains(&access.words) { return GlobalLoad::Gather; }
    if uniform[access.base.0] { return GlobalLoad::Broadcast; }
    if access.form == (Form::Global { scalar_base: false }) {
        if let Some(&stride) = affine.get(&access.base) {
            let words = stride / 4;
            let offset_word = access.offset / 4;
            let span = access.words * (access.size().bytes() / 4);
            if words >= 1 && access.offset % 4 == 0 && offset_word >= 0 && offset_word as u32 + span <= words {
                return GlobalLoad::Frame { stride_words: words, offset_words: offset_word as u32 };
            }
        }
    }
    GlobalLoad::Gather
}

pub(in crate::rdna_spmd) fn shape(access: &Access, width: Option<u32>, load: GlobalLoad, constants: &[Option<u64>]) -> Shape {
    if access.op == MemoryOp::Fence { return Shape::Fence; }
    let Some(width) = width else {
        if access.op == MemoryOp::AtomicAdd { return Shape::ScalarAtomic; }
        if access.stores() { return Shape::ScalarStore; }
        return Shape::ScalarLoad { pairs: matches!(access.form, Form::Global { .. }) && access.size() == MemSize::B32 && !access.semantics.volatile };
    };
    if access.scalar() { return Shape::ScalarWords; }
    if access.op == MemoryOp::AtomicAdd {
        let grouped = width >= 4 && access.space == Space::Global && !access.returns() && !access.semantics.volatile;
        return Shape::AtomicAdd { grouped };
    }
    let affine = access.form == (Form::Scratch { uniform: true });
    if access.stores() {
        return Shape::Store(if access.size().bytes() < 4 { StoreShape::Narrow }
            else if access.space == Space::Lds { StoreShape::Lds }
            else if affine && access.words >= 2 && access.static_scratch_end(constants).is_some() { StoreShape::Tile }
            else if affine { StoreShape::Affine }
            else { StoreShape::Scatter });
    }
    if access.size() == MemSize::B64 {
        return match load {
            GlobalLoad::Frame { stride_words, offset_words } => Shape::Frame { stride_words, offset_words, group: width.min(8) },
            GlobalLoad::Broadcast => Shape::Words { lanes: Lanes::Broadcast, pairs: false },
            GlobalLoad::Gather => Shape::Words { lanes: Lanes::Gather, pairs: false },
        };
    }
    if access.size() != MemSize::B32 { return Shape::NarrowLoad; }
    let allocated = access.static_scratch_end(constants).is_some();
    if allocated && access.words >= 2 && !std::env::var("AMDGPU_SIM_DEBUG_SHAPE").map_or(false, |v| v.contains("notile")) {
        return Shape::PrivateTile { tile: if width % 4 == 0 { 4 } else if width % 2 == 0 { 2 } else { 1 } };
    }
    let global = matches!(access.form, Form::Global { .. });
    if let (true, GlobalLoad::Frame { stride_words, offset_words }) = (global, load) {
        return Shape::Frame { stride_words, offset_words, group: width.min(8) };
    }
    let lanes = if global && load == GlobalLoad::Broadcast { Lanes::Broadcast }
        else if access.space == Space::Lds { Lanes::Lds }
        else if affine { Lanes::Affine { allocated } }
        else { Lanes::Gather };
    Shape::Words { lanes, pairs: global }
}

pub(in crate::rdna_spmd) fn clusters(f: &Func, accesses: &[Access], width: Option<u32>, uniform: &[bool], affine: &BTreeMap<ValueId, u32>) -> BTreeMap<usize, Cluster> {
    let mut out = BTreeMap::new();
    let Some(width) = width else { return out; };
    if !width.is_power_of_two() || std::env::var("AMDGPU_SIM_DEBUG_SHAPE").map_or(false, |v| v.contains("nocluster")) { return out; }
    let mut start = 0;
    while start < accesses.len() {
        let first = &accesses[start];
        let eligible = |a: &Access| a.form == (Form::Global { scalar_base: false }) && a.base == first.base
            && (a.op == MemoryOp::Load(MemSize::B32) && matches!(a.words, 2 | 4) || a.op == MemoryOp::Load(MemSize::B64) && matches!(a.words, 1 | 2));
        if !eligible(first) || uniform[first.base.0] || affine.contains_key(&first.base) { start += 1; continue; }
        let mut ranges = vec![(first.offset, first.offset + (first.size().bytes() * first.words) as i64)];
        let mut len = 1;
        while start + len < accesses.len() {
            let prev = &accesses[start + len - 1];
            let next = &accesses[start + len];
            if next.block != prev.block || !eligible(next) { break; }
            let between = &f.blocks[&prev.block].insts[prev.end..next.start];
            if between.iter().any(|inst| !matches!(inst, Inst::Core { .. })) { break; }
            ranges.push((next.offset, next.offset + (next.size().bytes() * next.words) as i64));
            len += 1;
        }
        if len >= 3 {
            let mut sorted = ranges.clone();
            sorted.sort();
            let lo = sorted[0].0;
            let mut hi = sorted[0].1;
            let mut contiguous = true;
            for &(a, b) in &sorted[1..] { if a > hi { contiguous = false; break; } hi = hi.max(b); }
            if contiguous && sorted.iter().all(|&(a, _)| (a - lo) % 8 == 0) && (hi - lo) % 8 == 0 {
                let span = ((hi - lo) / 8) as u32;
                if span >= 4 {
                    out.insert(start, Cluster { members: len, lo, span, tile: transpose_tile(width) });
                    start += len;
                    continue;
                }
            }
        }
        start += 1;
    }
    out
}

impl<'a> Cg<'a> {
    fn masked_call(&self, prefix: &str, overloads: &[Type], args: &[Value], ptr_pos: u32, align: u64) -> Value {
        let call = self.ir.call_intrinsic(prefix, overloads, args);
        self.ir.set_call_align(call, ptr_pos + 1, align);
        call
    }
    pub(super) fn vptr(&self) -> Type { self.ir.ptr().vector(self.width()) }
    pub(super) fn vi32(&self) -> Type { self.ir.i32().vector(self.width()) }
    pub(super) fn vi64(&self) -> Type { self.ir.i64().vector(self.width()) }
    pub(super) fn vi1(&self) -> Type { self.ir.i1().vector(self.width()) }
    fn ptr_at_vec(&self, addr: Value, off: u64) -> Value {
        let a = self.ir.add(addr, self.splat(self.ci64(off)));
        self.ir.inttoptr(a, self.vptr())
    }
    fn masked_gather_ty(&self, ptrs: Value, mask: Value, elem: Type, align: u64) -> Value {
        let velem = elem.vector(self.width());
        let passthru = velem.null();
        self.masked_call("llvm.masked.gather.", &[velem, self.vptr()], &[ptrs, mask, passthru], 0, align)
    }
    fn masked_scatter_ty(&self, val: Value, ptrs: Value, mask: Value, elem: Type, align: u64) {
        let velem = elem.vector(self.width());
        self.masked_call("llvm.masked.scatter.", &[velem, self.vptr()], &[val, ptrs, mask], 1, align);
    }
    fn reduce(&self, operation: &str, ret: Type, operand: Value) -> Value {
        self.ir.call_named(&format!("llvm.vector.reduce.{operation}"), ret, &[operand.ty()], &[operand])
    }
    fn any_active(&self, exec: Value, nonempty: bool) -> Value {
        if nonempty { self.ir.ci1(true) } else {
            self.reduce(&format!("or.v{}i1", self.width()), self.ir.i1(), exec)
        }
    }
    fn bcast_load(&self, ptrs: Value, exec: Value, nonempty: bool, elem: Type) -> Value {
        let p0 = self.ir.extract_at(ptrs, 0);
        let any = self.any_active(exec, nonempty);
        let p0 = self.ir.select(any, p0, self.sink);
        let v = self.ir.load(elem, p0);
        self.splat(v)
    }
    fn affine_load(&self, ptrs: Value, exec: Value, allocated: bool) -> Value {
        let ir = self.ir;
        let mut v = self.vi32().poison();
        for l in 0..self.width() {
            let p = ir.extract_at(ptrs, l);
            let active = ir.extract_at(exec, l);
            let p = if allocated { p } else { ir.select(active, p, self.sink) };
            let ld = ir.load(ir.i32(), p).set_alignment(4);
            v = ir.insert_at(v, ld, l);
        }
        v
    }
    fn affine_store(&self, val: Value, ptrs: Value, exec: Value) {
        let ir = self.ir;
        let sink = ir.ptrtoint(self.store_sink, ir.i64());
        let addr_i = ir.ptrtoint(ptrs, self.vi64());
        let safe = ir.select(exec, addr_i, self.splat(sink));
        for l in 0..self.width() {
            let a = ir.extract_at(safe, l);
            let p = ir.inttoptr(a, ir.ptr());
            let d = ir.extract_at(val, l);
            ir.store(d, p).set_alignment(4);
        }
    }
    fn split_pair(&self, d: Value) -> (Value, Value) {
        let ir = self.ir;
        let bits = ir.bitcast(d, self.vec_ty(ir.i64()));
        let lo = ir.trunc(bits, self.vec_ty(ir.i32()));
        let shifted = ir.lshr(bits, self.splat(self.ci64(32)));
        let hi = ir.trunc(shifted, self.vec_ty(ir.i32()));
        (lo, hi)
    }
    fn vwiden_i32(&self, value: Value, have: u32, want: u32) -> Value {
        if have == want { return value; }
        let idx: Vec<u32> = (0..want).map(|k| k.min(have - 1)).collect();
        self.ir.shuffle_by(value, value.ty().poison(), &idx)
    }
    fn vconcat_i32(&self, parts: &[Value]) -> Value {
        let mut cur = parts.to_vec();
        while cur.len() > 1 {
            let mut next = Vec::with_capacity((cur.len() + 1) / 2);
            let mut i = 0;
            while i + 1 < cur.len() {
                let (a, b) = (cur[i], cur[i + 1]);
                let na = a.ty().vector_size();
                let nb = b.ty().vector_size();
                let wide = na.max(nb);
                let (a, b) = (self.vwiden_i32(a, na, wide), self.vwiden_i32(b, nb, wide));
                let idx: Vec<u32> = (0..na).chain(wide..wide + nb).collect();
                next.push(self.ir.shuffle_by(a, b, &idx));
                i += 2;
            }
            if i < cur.len() { next.push(cur[i]); }
            cur = next;
        }
        cur[0]
    }
    fn transpose_rows(&self, rows: &[Value], span: u32, tile: u32) -> Vec<Value> {
        let rowty = rows[0].ty();
        let shuf = |x: Value, y: Value, m: &[u32]| -> Value { self.ir.shuffle_by(x, y, m) };
        let nblk = (self.width() / tile) as usize;
        let mut cols: Vec<Vec<Value>> = vec![Vec::with_capacity(nblk); span as usize];
        for blk in 0..nblk {
            let begin = blk * tile as usize;
            let r = &rows[begin..begin + tile as usize];
            let mut base = 0u32;
            while base + tile <= span {
                let idx: Vec<u32> = (base..base + tile).collect();
                let poison = rowty.poison();
                let mut cur: Vec<Value> = r.iter().map(|&row| shuf(row, poison, &idx)).collect();
                let mut step = 1u32;
                while step < tile {
                    let (lo_mask, hi_mask) = transpose_pair_masks(tile, step);
                    let mut next = vec![UNDEFINED; tile as usize];
                    for i in 0..tile {
                        if i & step != 0 { continue; }
                        let j = i | step;
                        next[i as usize] = shuf(cur[i as usize], cur[j as usize], &lo_mask);
                        next[j as usize] = shuf(cur[i as usize], cur[j as usize], &hi_mask);
                    }
                    cur = next;
                    step <<= 1;
                }
                for cix in 0..tile { cols[(base + cix) as usize].push(cur[cix as usize]); }
                base += tile;
            }
            for f in base..span {
                let mut parts: Vec<Value> = if tile == 1 {
                    vec![shuf(r[0], rowty.poison(), &[f])]
                } else {
                    r.chunks(2).map(|pair| shuf(pair[0], pair[1], &[f, span + f])).collect()
                };
                let mut n = 2u32;
                while parts.len() > 1 {
                    let cat: Vec<u32> = (0..2 * n).collect();
                    parts = parts.chunks(2).map(|pair| shuf(pair[0], pair[1], &cat)).collect();
                    n *= 2;
                }
                cols[f as usize].push(parts[0]);
            }
        }
        cols.into_iter().map(|mut parts| {
            let mut n = tile;
            while parts.len() > 1 {
                let cat: Vec<u32> = (0..2 * n).collect();
                parts = parts.chunks(2).map(|p| shuf(p[0], p[1], &cat)).collect();
                n *= 2;
            }
            parts[0]
        }).collect()
    }

    pub(super) fn emit_cluster(&mut self, members: &[usize], cluster: &Cluster) {
        let ir = self.ir;
        let first = &self.p.accesses[members[0]];
        let exec = self.vector(first.mask);
        let exec = self.em.to_bool(exec);
        let addr = self.vector(first.base);
        let zero = self.vi64().null();
        let masked = ir.select(exec, addr, zero);
        let p_any = self.reduce(&format!("umax.v{}i64", self.width()), ir.i64(), masked);
        let safe = ir.select(exec, addr, self.splat(p_any));
        let rowty = ir.f64().vector(cluster.span);
        let any = self.reduce(&format!("or.v{}i1", self.width()), ir.i1(), exec);
        let rowmask = {
            let poison = ir.i1().vector(cluster.span).poison();
            let ins = ir.insert_at(poison, any, 0);
            ir.shuffle(ins, poison, ir.i32().vector(cluster.span).null())
        };
        let rows: Vec<Value> = (0..self.width()).map(|l| {
            let a = ir.extract_at(safe, l);
            let a = ir.add(a, self.ci64(cluster.lo as u64));
            let p = ir.inttoptr(a, ir.ptr());
            self.masked_call("llvm.masked.load.", &[rowty, ir.ptr()], &[p, rowmask, rowty.poison()], 0, 4)
        }).collect();
        let cols = self.transpose_rows(&rows, cluster.span, cluster.tile);
        for &m in members {
            let access = &self.p.accesses[m];
            let f0 = ((access.offset - cluster.lo) / 8) as usize;
            let results = access.results.clone();
            if access.size() == MemSize::B64 {
                for j in 0..access.words as usize {
                    let wide = ir.bitcast(cols[f0 + j], self.vi64());
                    self.define(results[j], wide);
                }
                continue;
            }
            for j in 0..(access.words / 2) as usize {
                let (lo, hi) = self.split_pair(cols[f0 + j]);
                self.define(results[2 * j], lo);
                self.define(results[2 * j + 1], hi);
            }
        }
    }

    pub(super) fn emit_memory(&mut self, index: usize, block: BlockId, at: usize) {
        let ir = self.ir;
        let access = &self.p.accesses[index];
        let shape = self.p.shapes[index];
        if shape == Shape::Fence {
            let order = match access.semantics.ordering {
                Ordering::Acquire => Atomic::Acquire,
                Ordering::Release => Atomic::Release,
                _ => Atomic::SequentiallyConsistent,
            };
            ir.fence(order);
            return;
        }
        let elem = match access.size().bytes() { 1 => ir.i8(), 2 => ir.i16(), 8 => ir.i64(), _ => ir.i32() };
        let size = access.size();
        let words = access.words;
        let offsets = access.offsets.clone();
        let results = access.results.clone();
        let data = access.data.clone();
        let (space, volatile) = (access.space, access.semantics.volatile);
        let flat_private: Vec<ValueId> = access.private.clone();
        let inside = access.inside;
        let address = access.address;
        let mask = access.mask;
        let base = access.base;
        let nonempty = self.p.exec.nonempty_at(block, at);
        match shape {
            Shape::ScalarWords | Shape::ScalarLoad { .. } | Shape::ScalarStore | Shape::ScalarAtomic => {
                let mut addr = self.scalar(address);
                if self.p.width.is_none() {
                    if let Some(inside) = inside {
                        let offset = ir.sub(self.scratch_vec, self.scratch_base_scalar);
                        let physical = ir.add(addr, offset);
                        let inside = self.scalar(inside);
                        addr = ir.select(inside, physical, addr);
                    }
                    if space != Space::Global {
                        let extended = if space == Space::Scratch { ir.sext(addr, ir.i64()) } else { ir.zext(addr, ir.i64()) };
                        addr = ir.add(if space == Space::Scratch { self.scratch_vec } else { self.lds_base }, extended);
                    }
                }
                let predicated = shape != Shape::ScalarWords && !access.scalar();
                let active = if predicated { Some(self.scalar(mask)) } else { None };
                let guard = |cg: &Self, a: Value| -> Value {
                    match active { Some(active) => { let dummy = ir.ptrtoint(cg.sink, ir.i64()); ir.select(active, a, dummy) } None => a }
                };
                if shape == Shape::ScalarAtomic {
                    let p = ir.inttoptr(guard(self, addr), ir.ptr());
                    let d = self.scalar(data[0]);
                    let old = ir.atomic_add(p, d, Atomic::SequentiallyConsistent);
                    if let Some(&r) = results.first() { self.define(r, old); }
                    return;
                }
                if shape == Shape::ScalarStore {
                    for k in 0..words as usize {
                        let value = self.scalar(data[k]);
                        let value = if size.bytes() >= 4 { value } else { ir.trunc(value, elem) };
                        let a = guard(self, ir.add(addr, self.ci64(offsets[k] as u64)));
                        let p = ir.inttoptr(a, ir.ptr());
                        ir.store(value, p).set_alignment(if space == Space::Lds { 1 } else { size.bytes() }).set_volatile(volatile);
                    }
                    return;
                }
                let load_addr = guard(self, addr);
                let pairs = matches!(shape, Shape::ScalarLoad { pairs: true });
                let mut k = 0usize;
                while k < words as usize {
                    if pairs && k + 1 < words as usize {
                        let p = ir.inttoptr(ir.add(load_addr, self.ci64(offsets[k] as u64)), ir.ptr());
                        let value = ir.load(ir.f64(), p).set_alignment(4);
                        let (lo, hi) = self.split_pair(value);
                        self.define(results[k], lo);
                        self.define(results[k + 1], hi);
                        self.loaded_pairs.insert((results[k], results[k + 1]), value);
                        k += 2;
                    } else {
                        let p = ir.inttoptr(ir.add(load_addr, self.ci64(offsets[k] as u64)), ir.ptr());
                        let load = ir.load(elem, p);
                        load.set_alignment(if shape == Shape::ScalarWords { if size.bytes() >= 4 { 4 } else { 1 } } else if space == Space::Lds || size.bytes() < 4 { 1 } else { 4 });
                        if shape != Shape::ScalarWords { load.set_volatile(volatile); }
                        let value = if size.bytes() >= 4 { load } else if size.signed() { ir.sext(load, ir.i32()) } else { ir.zext(load, ir.i32()) };
                        self.define(results[k], value);
                        k += 1;
                    }
                }
                for (&r, &private) in results.iter().zip(&flat_private) { let v = self.values[r.0]; self.define(private, v); }
                return;
            }
            _ => {}
        }
        let mut addr = self.vector(address);
        let exec = self.vector(mask);
        let exec = self.em.to_bool(exec);
        if space == Space::Scratch {
            addr = ir.add(self.scratch_vec, ir.sext(addr, self.vi64()));
        }
        if space == Space::Lds {
            addr = ir.add(self.splat(self.lds_base), ir.zext(addr, self.vi64()));
        }
        if let Some(inside) = inside {
            let lane_offset = ir.sub(self.scratch_vec, self.splat(self.scratch_base_scalar));
            let physical = ir.add(addr, lane_offset);
            let inside = self.vector(inside);
            let inside = self.em.to_bool(inside);
            addr = ir.select(inside, physical, addr);
        }
        match shape {
            Shape::AtomicAdd { grouped } => {
                let d = self.vector(data[0]);
                if grouped { self.emit_grouped_atomic_add(addr, d, exec); return; }
                let packed_exec = self.vec_to_mask(exec);
                let ptrs = self.ptr_at_vec(addr, 0);
                let mut result = self.vi32().poison();
                for k in 0..self.width() {
                    let bit = ir.and(ir.lshr(packed_exec, self.ci32(k)), self.ci32(1));
                    let active = ir.icmp(IntPred::Ne, bit, self.ci32(0));
                    let ptr = ir.extract_at(ptrs, k);
                    let ptr = ir.select(active, ptr, self.sink);
                    let value = ir.extract_at(d, k);
                    let value = ir.select(active, value, self.ci32(0));
                    let old = ir.atomic_add(ptr, value, Atomic::SequentiallyConsistent);
                    result = ir.insert_at(result, old, k);
                }
                if let Some(&r) = results.first() { self.define(r, result); }
            }
            Shape::Store(StoreShape::Tile) => {
                let cols: Vec<Value> = (0..words as usize).map(|k| self.vector(data[k])).collect();
                let base = self.ptr_at_vec(addr, offsets[0] as u64);
                let sink = ir.ptrtoint(self.tile_sink, ir.i64());
                let addr_i = ir.ptrtoint(base, self.vi64());
                let safe = ir.select(exec, addr_i, self.splat(sink));
                let w = self.width();
                for l in 0..w {
                    let mut parts = Vec::new();
                    let mut k = 0usize;
                    while k < words as usize {
                        if k + 1 < words as usize {
                            parts.push(ir.shuffle_by(cols[k], cols[k + 1], &[l, w + l]));
                            k += 2;
                        } else {
                            parts.push(ir.shuffle_by(cols[k], cols[k].ty().poison(), &[l]));
                            k += 1;
                        }
                    }
                    let row = self.vconcat_i32(&parts);
                    let a = ir.extract_at(safe, l);
                    let p = ir.inttoptr(a, ir.ptr());
                    ir.store(row, p).set_alignment(4);
                }
            }
            Shape::Store(kind) => {
                for k in 0..words as usize {
                    let ptrs = self.ptr_at_vec(addr, offsets[k] as u64);
                    let value = self.vector(data[k]);
                    match kind {
                        StoreShape::Narrow => {
                            let value = ir.trunc(value, elem.vector(self.width()));
                            self.masked_scatter_ty(value, ptrs, exec, elem, 1);
                        }
                        StoreShape::Lds => self.masked_scatter_ty(value, ptrs, exec, ir.i32(), 1),
                        StoreShape::Affine => self.affine_store(value, ptrs, exec),
                        StoreShape::Tile => unreachable!("tile stores are emitted as rows"),
                        StoreShape::Scatter => self.masked_scatter_ty(value, ptrs, exec, elem, 4),
                    }
                }
            }
            Shape::NarrowLoad => {
                let value = self.masked_gather_ty(self.ptr_at_vec(addr, 0), exec, elem, 1);
                let value = if size.signed() { ir.sext(value, self.vi32()) } else { ir.zext(value, self.vi32()) };
                self.define(results[0], value);
            }
            Shape::PrivateTile { tile } => {
                let rowty = ir.i32().vector(words);
                let rows: Vec<_> = (0..self.width()).map(|l| {
                    let a = ir.extract_at(addr, l);
                    let p = ir.inttoptr(a, ir.ptr());
                    ir.load(rowty, p).set_alignment(4)
                }).collect();
                let cols = self.transpose_rows(&rows, words, tile);
                for k in 0..words as usize { self.define(results[k], cols[k]); }
            }
            Shape::Frame { stride_words: sp4, offset_words: ioff_w, group: grp } => {
                let base_v = self.vector(base);
                let nblk = self.width() / grp;
                let blkty = ir.i32().vector(grp * sp4);
                let poison_blk = blkty.poison();
                let blocks: Vec<Value> = (0..nblk).map(|g| {
                    let a = ir.extract_at(base_v, g * grp);
                    let p = ir.inttoptr(a, ir.ptr());
                    ir.load(blkty, p).set_alignment(4)
                }).collect();
                let per = size.bytes() / 4;
                let extract = |cg: &Self, fw: u32| -> Value {
                    let parts: Vec<Value> = blocks.iter().map(|&blk| {
                        let idx: Vec<u32> = (0..grp).flat_map(|lane| (0..per).map(move |w| lane * sp4 + fw + w)).collect();
                        ir.shuffle_by(blk, poison_blk, &idx)
                    }).collect();
                    let joined = cg.vconcat_i32(&parts);
                    if per == 1 { joined } else { ir.bitcast(joined, cg.vi64()) }
                };
                for k in 0..words as usize { let v = extract(self, ioff_w + k as u32 * per); self.define(results[k], v); }
            }
            Shape::Words { lanes, pairs } => {
                let mut k = 0usize;
                while k < words as usize {
                    if pairs && k + 1 < words as usize {
                        let ptrs = self.ptr_at_vec(addr, offsets[k] as u64);
                        let d = match lanes {
                            Lanes::Broadcast => self.bcast_load(ptrs, exec, nonempty, ir.f64()),
                            _ => self.masked_gather_ty(ptrs, exec, ir.f64(), 4),
                        };
                        let (lo, hi) = self.split_pair(d);
                        self.define(results[k], lo);
                        self.define(results[k + 1], hi);
                        self.loaded_pairs.insert((results[k], results[k + 1]), d);
                        k += 2;
                    } else {
                        let ptrs = self.ptr_at_vec(addr, offsets[k] as u64);
                        let d = match lanes {
                            Lanes::Broadcast => self.bcast_load(ptrs, exec, nonempty, elem),
                            Lanes::Lds => self.masked_gather_ty(ptrs, exec, ir.i32(), 1),
                            Lanes::Affine { allocated } => self.affine_load(ptrs, exec, allocated),
                            Lanes::Gather => self.masked_gather_ty(ptrs, exec, elem, 4),
                        };
                        self.define(results[k], d);
                        k += 1;
                    }
                }
            }
            _ => unreachable!(),
        }
        for (&r, &private) in results.iter().zip(&flat_private) { let v = self.values[r.0]; self.define(private, v); }
    }

    fn emit_grouped_atomic_add(&self, addresses: Value, values: Value, exec: Value) {
        let ir = self.ir;
        let addresses = ir.freeze(addresses);
        let entry = ir.insert_block();
        let function = entry.function();
        let header = ir.append_block(function, "atomic.groups");
        let body = ir.append_block(function, "atomic.group");
        let done = ir.append_block(function, "atomic.done");
        let initial = self.vec_to_mask(exec);
        ir.br(header);
        ir.position_at_end(header);
        let pending = ir.phi(ir.i32());
        pending.add_incoming(&[(initial, entry)]);
        let nonempty = ir.icmp(IntPred::Ne, pending, self.ci32(0));
        ir.cond_br(nonempty, body, done);
        ir.position_at_end(body);
        let lane = ir.call_named("llvm.cttz.i32", ir.i32(), &[ir.i32(), ir.i1()], &[pending, ir.ci1(true)]);
        let address = ir.extract(addresses, lane);
        let equal = ir.icmp(IntPred::Eq, addresses, self.splat(address));
        let members = ir.and(self.vec_to_mask(equal), pending);
        let mask = self.mask_to_vec(members);
        let addends = ir.select(mask, values, self.vi32().null());
        let sum = self.reduce(&format!("add.v{}i32", self.width()), ir.i32(), addends);
        let pointer = ir.inttoptr(address, ir.ptr());
        ir.atomic_add(pointer, sum, Atomic::SequentiallyConsistent);
        let remaining = ir.and(pending, ir.not(members));
        let backedge = ir.insert_block();
        ir.br(header);
        pending.add_incoming(&[(remaining, backedge)]);
        ir.position_at_end(done);
    }
}

fn transpose_pair_masks(tile: u32, step: u32) -> (Vec<u32>, Vec<u32>) {
    let mask = |high: bool| {
        (0..tile).map(|position| {
            let from_second = position & step != 0;
            let element = (position & !step) | if high { step } else { 0 };
            element + if from_second { tile } else { 0 }
        }).collect()
    };
    (mask(false), mask(true))
}
