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
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let native = if std::arch::is_x86_feature_detected!("avx512f") {
        8
    } else if std::arch::is_x86_feature_detected!("avx2") {
        4
    } else {
        2
    };
    #[cfg(target_arch = "aarch64")]
    let native = 2;
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
    let native = 1;
    width.min(native)
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
    if access.op != MemoryOp::Load(MemSize::B32) || !(1..=4).contains(&access.words) { return GlobalLoad::Gather; }
    if uniform[access.base.0] { return GlobalLoad::Broadcast; }
    if access.form == (Form::Global { scalar_base: false }) {
        if let Some(&stride) = affine.get(&access.base) {
            let words = stride / 4;
            let offset_word = access.offset / 4;
            if words >= 1 && access.offset % 4 == 0 && offset_word >= 0 && offset_word as u32 + access.words <= words {
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
        return Shape::Store(if access.size() != MemSize::B32 { StoreShape::Narrow }
            else if access.space == Space::Lds { StoreShape::Lds }
            else if affine && access.words >= 2 && access.static_scratch_end(constants).is_some() { StoreShape::Tile }
            else if affine { StoreShape::Affine }
            else { StoreShape::Scatter });
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
        let eligible = |a: &Access| a.form == (Form::Global { scalar_base: false }) && a.op == MemoryOp::Load(MemSize::B32) && matches!(a.words, 2 | 4) && a.base == first.base;
        if !eligible(first) || uniform[first.base.0] || affine.contains_key(&first.base) { start += 1; continue; }
        let mut ranges = vec![(first.offset, first.offset + 4 * first.words as i64)];
        let mut len = 1;
        while start + len < accesses.len() {
            let prev = &accesses[start + len - 1];
            let next = &accesses[start + len];
            if next.block != prev.block || !eligible(next) { break; }
            let between = &f.blocks[&prev.block].insts[prev.end..next.start];
            if between.iter().any(|inst| !matches!(inst, Inst::Core { .. })) { break; }
            ranges.push((next.offset, next.offset + 4 * next.words as i64));
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
    unsafe fn call(&self, name: &str, ret: LLVMTypeRef, params: &[LLVMTypeRef], args: &[LLVMValueRef]) -> LLVMValueRef {
        let cname = cstr(name);
        let mut f = LLVMGetNamedFunction(self.module, cname.as_ptr());
        let fty = LLVMFunctionType(ret, params.as_ptr() as *mut _, params.len() as u32, 0);
        if f.is_null() { f = LLVMAddFunction(self.module, cname.as_ptr(), fty); }
        LLVMBuildCall2(self.b, fty, f, args.as_ptr() as *mut _, args.len() as u32, self.n())
    }
    unsafe fn masked_call(&self, prefix: &str, overloads: &[LLVMTypeRef], args: &[LLVMValueRef], ptr_pos: u32, align: u64) -> LLVMValueRef {
        let id = LLVMLookupIntrinsicID(prefix.as_ptr() as *const _, prefix.len());
        let mut overloads = overloads.to_vec();
        let f = LLVMGetIntrinsicDeclaration(self.module, id, overloads.as_mut_ptr(), overloads.len());
        let fty = LLVMGlobalGetValueType(f);
        let mut args = args.to_vec();
        let call = LLVMBuildCall2(self.b, fty, f, args.as_mut_ptr(), args.len() as u32, self.n());
        let name = b"align";
        let kind = LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
        let attr = LLVMCreateEnumAttribute(self.ctx, kind, align);
        LLVMAddCallSiteAttribute(call, ptr_pos + 1, attr);
        call
    }
    pub(super) unsafe fn vptr(&self) -> LLVMTypeRef { LLVMVectorType(self.ptr, self.width()) }
    pub(super) unsafe fn vi32(&self) -> LLVMTypeRef { LLVMVectorType(self.i32t, self.width()) }
    pub(super) unsafe fn vi64(&self) -> LLVMTypeRef { LLVMVectorType(self.i64t, self.width()) }
    pub(super) unsafe fn vi1(&self) -> LLVMTypeRef { LLVMVectorType(self.i1, self.width()) }
    unsafe fn ptr_at_vec(&self, addr: LLVMValueRef, off: u64) -> LLVMValueRef {
        let a = LLVMBuildAdd(self.b, addr, self.splat(self.ci64(off)), self.n());
        LLVMBuildIntToPtr(self.b, a, self.vptr(), self.n())
    }
    unsafe fn masked_gather_ty(&self, ptrs: LLVMValueRef, mask: LLVMValueRef, elem: LLVMTypeRef, align: u64) -> LLVMValueRef {
        let velem = LLVMVectorType(elem, self.width());
        let passthru = LLVMConstNull(velem);
        self.masked_call("llvm.masked.gather.", &[velem, self.vptr()], &[ptrs, mask, passthru], 0, align)
    }
    unsafe fn masked_scatter_ty(&self, val: LLVMValueRef, ptrs: LLVMValueRef, mask: LLVMValueRef, elem: LLVMTypeRef, align: u64) {
        let velem = LLVMVectorType(elem, self.width());
        self.masked_call("llvm.masked.scatter.", &[velem, self.vptr()], &[val, ptrs, mask], 1, align);
    }
    unsafe fn any_active(&self, exec: LLVMValueRef, nonempty: bool) -> LLVMValueRef {
        if nonempty { LLVMConstInt(self.i1, 1, 0) } else {
            self.call(&format!("llvm.vector.reduce.or.v{}i1", self.width()), self.i1, &[self.vi1()], &[exec])
        }
    }
    unsafe fn bcast_load(&self, ptrs: LLVMValueRef, exec: LLVMValueRef, nonempty: bool, elem: LLVMTypeRef) -> LLVMValueRef {
        let p0 = LLVMBuildExtractElement(self.b, ptrs, self.ci32(0), self.n());
        let any = self.any_active(exec, nonempty);
        let p0 = LLVMBuildSelect(self.b, any, p0, self.bvh_scratch, self.n());
        let v = LLVMBuildLoad2(self.b, elem, p0, self.n());
        self.splat(v)
    }
    unsafe fn affine_load(&self, ptrs: LLVMValueRef, exec: LLVMValueRef, allocated: bool) -> LLVMValueRef {
        let n = self.n();
        let mut v = LLVMGetPoison(self.vi32());
        for l in 0..self.width() {
            let p = LLVMBuildExtractElement(self.b, ptrs, self.ci32(l), n);
            let active = LLVMBuildExtractElement(self.b, exec, self.ci32(l), n);
            let p = if allocated { p } else { LLVMBuildSelect(self.b, active, p, self.bvh_scratch, n) };
            let ld = LLVMBuildLoad2(self.b, self.i32t, p, n);
            LLVMSetAlignment(ld, 4);
            v = LLVMBuildInsertElement(self.b, v, ld, self.ci32(l), n);
        }
        v
    }
    unsafe fn affine_store(&self, val: LLVMValueRef, ptrs: LLVMValueRef, exec: LLVMValueRef) {
        let n = self.n();
        let sink = LLVMBuildPtrToInt(self.b, self.store_sink, self.i64t, n);
        let addr_i = LLVMBuildPtrToInt(self.b, ptrs, self.vi64(), n);
        let safe = LLVMBuildSelect(self.b, exec, addr_i, self.splat(sink), n);
        for l in 0..self.width() {
            let a = LLVMBuildExtractElement(self.b, safe, self.ci32(l), n);
            let p = LLVMBuildIntToPtr(self.b, a, self.ptr, n);
            let d = LLVMBuildExtractElement(self.b, val, self.ci32(l), n);
            let st = LLVMBuildStore(self.b, d, p);
            LLVMSetAlignment(st, 4);
        }
    }
    unsafe fn split_pair(&self, d: LLVMValueRef) -> (LLVMValueRef, LLVMValueRef) {
        let n = self.n();
        let bits = LLVMBuildBitCast(self.b, d, self.vec_ty(self.i64t), n);
        let lo = LLVMBuildTrunc(self.b, bits, self.vec_ty(self.i32t), n);
        let shifted = LLVMBuildLShr(self.b, bits, self.splat(self.ci64(32)), n);
        let hi = LLVMBuildTrunc(self.b, shifted, self.vec_ty(self.i32t), n);
        (lo, hi)
    }
    unsafe fn vconcat_i32(&self, parts: &[LLVMValueRef]) -> LLVMValueRef {
        let mut cur = parts.to_vec();
        while cur.len() > 1 {
            let mut next = Vec::with_capacity((cur.len() + 1) / 2);
            let mut i = 0;
            while i + 1 < cur.len() {
                let (a, b) = (cur[i], cur[i + 1]);
                let na = LLVMGetVectorSize(LLVMTypeOf(a));
                let nb = LLVMGetVectorSize(LLVMTypeOf(b));
                let mut idx: Vec<LLVMValueRef> = (0..na + nb).map(|k| self.ci32(k)).collect();
                let mask = LLVMConstVector(idx.as_mut_ptr(), idx.len() as u32);
                next.push(LLVMBuildShuffleVector(self.b, a, b, mask, self.n()));
                i += 2;
            }
            if i < cur.len() { next.push(cur[i]); }
            cur = next;
        }
        cur[0]
    }
    unsafe fn transpose_rows(&self, rows: &[LLVMValueRef], span: u32, tile: u32) -> Vec<LLVMValueRef> {
        let rowty = LLVMTypeOf(rows[0]);
        let shuf = |x: LLVMValueRef, y: LLVMValueRef, m: &[u32]| -> LLVMValueRef {
            let mut mv: Vec<LLVMValueRef> = m.iter().map(|&i| self.ci32(i)).collect();
            let mask = LLVMConstVector(mv.as_mut_ptr(), mv.len() as u32);
            LLVMBuildShuffleVector(self.b, x, y, mask, self.n())
        };
        let nblk = (self.width() / tile) as usize;
        let mut cols: Vec<Vec<LLVMValueRef>> = vec![Vec::with_capacity(nblk); span as usize];
        for blk in 0..nblk {
            let begin = blk * tile as usize;
            let r = &rows[begin..begin + tile as usize];
            let mut base = 0u32;
            while base + tile <= span {
                let idx: Vec<u32> = (base..base + tile).collect();
                let poison = LLVMGetPoison(rowty);
                let mut cur: Vec<LLVMValueRef> = r.iter().map(|&row| shuf(row, poison, &idx)).collect();
                let mut step = 1u32;
                while step < tile {
                    let (lo_mask, hi_mask) = transpose_pair_masks(tile, step);
                    let mut next = vec![std::ptr::null_mut(); tile as usize];
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
                let mut parts: Vec<LLVMValueRef> = if tile == 1 {
                    vec![shuf(r[0], LLVMGetPoison(rowty), &[f])]
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

    pub(super) unsafe fn emit_cluster(&mut self, members: &[usize], cluster: &Cluster) {
        let n = self.n();
        let first = &self.p.accesses[members[0]];
        let exec = self.vector(first.mask);
        let addr = self.vector(first.base);
        let zero = LLVMConstNull(self.vi64());
        let masked = LLVMBuildSelect(self.b, exec, addr, zero, n);
        let p_any = self.call(&format!("llvm.vector.reduce.umax.v{}i64", self.width()), self.i64t, &[self.vi64()], &[masked]);
        let safe = LLVMBuildSelect(self.b, exec, addr, self.splat(p_any), n);
        let rowty = LLVMVectorType(self.f64t, cluster.span);
        let any = self.call(&format!("llvm.vector.reduce.or.v{}i1", self.width()), self.i1, &[self.vi1()], &[exec]);
        let rowmask = { let poison = LLVMGetPoison(LLVMVectorType(self.i1, cluster.span)); let ins = LLVMBuildInsertElement(self.b, poison, any, self.ci32(0), n); LLVMBuildShuffleVector(self.b, ins, poison, LLVMConstNull(LLVMVectorType(self.i32t, cluster.span)), n) };
        let rows: Vec<LLVMValueRef> = (0..self.width()).map(|l| {
            let a = LLVMBuildExtractElement(self.b, safe, self.ci32(l), n);
            let a = LLVMBuildAdd(self.b, a, self.ci64(cluster.lo as u64), n);
            let p = LLVMBuildIntToPtr(self.b, a, self.ptr, n);
            self.masked_call("llvm.masked.load.", &[rowty, self.ptr], &[p, rowmask, LLVMGetPoison(rowty)], 0, 4)
        }).collect();
        let cols = self.transpose_rows(&rows, cluster.span, cluster.tile);
        for &m in members {
            let access = &self.p.accesses[m];
            let f0 = ((access.offset - cluster.lo) / 8) as usize;
            let results = access.results.clone();
            for j in 0..(access.words / 2) as usize {
                let (lo, hi) = self.split_pair(cols[f0 + j]);
                self.define(results[2 * j], lo);
                self.define(results[2 * j + 1], hi);
            }
        }
    }

    pub(super) unsafe fn emit_memory(&mut self, index: usize, block: BlockId, at: usize) {
        let access = &self.p.accesses[index];
        let shape = self.p.shapes[index];
        let n = self.n();
        if shape == Shape::Fence {
            let order = match access.semantics.ordering {
                Ordering::Acquire => llvm::LLVMAtomicOrdering::LLVMAtomicOrderingAcquire,
                Ordering::Release => llvm::LLVMAtomicOrdering::LLVMAtomicOrderingRelease,
                _ => llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
            };
            LLVMBuildFence(self.b, order, 0, n);
            return;
        }
        let elem = match access.size().bytes() { 1 => LLVMInt8TypeInContext(self.ctx), 2 => LLVMInt16TypeInContext(self.ctx), _ => self.i32t };
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
                        let offset = LLVMBuildSub(self.b, self.scratch_vec, self.scratch_base_scalar, n);
                        let physical = LLVMBuildAdd(self.b, addr, offset, n);
                        let inside = self.scalar(inside);
                        addr = LLVMBuildSelect(self.b, inside, physical, addr, n);
                    }
                    if space != Space::Global {
                        let extended = if space == Space::Scratch { LLVMBuildSExt(self.b, addr, self.i64t, n) } else { LLVMBuildZExt(self.b, addr, self.i64t, n) };
                        addr = LLVMBuildAdd(self.b, if space == Space::Scratch { self.scratch_vec } else { self.lds_base }, extended, n);
                    }
                }
                let predicated = shape != Shape::ScalarWords && !access.scalar();
                let active = if predicated { Some(self.scalar(mask)) } else { None };
                let guard = |cg: &Self, a: LLVMValueRef| -> LLVMValueRef {
                    match active { Some(active) => { let dummy = LLVMBuildPtrToInt(cg.b, cg.bvh_scratch, cg.i64t, n); LLVMBuildSelect(cg.b, active, a, dummy, n) } None => a }
                };
                if shape == Shape::ScalarAtomic {
                    let p = LLVMBuildIntToPtr(self.b, guard(self, addr), self.ptr, n);
                    let d = self.scalar(data[0]);
                    let old = LLVMBuildAtomicRMW(self.b, llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd, p, d, llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent, 0);
                    if let Some(&r) = results.first() { self.define(r, old); }
                    return;
                }
                if shape == Shape::ScalarStore {
                    for k in 0..words as usize {
                        let value = self.scalar(data[k]);
                        let value = if size == MemSize::B32 { value } else { LLVMBuildTrunc(self.b, value, elem, n) };
                        let a = guard(self, LLVMBuildAdd(self.b, addr, self.ci64(offsets[k] as u64), n));
                        let p = LLVMBuildIntToPtr(self.b, a, self.ptr, n);
                        let store = LLVMBuildStore(self.b, value, p);
                        LLVMSetAlignment(store, if space == Space::Lds { 1 } else { size.bytes() });
                        LLVMSetVolatile(store, volatile as i32);
                    }
                    return;
                }
                let load_addr = guard(self, addr);
                let pairs = matches!(shape, Shape::ScalarLoad { pairs: true });
                let mut k = 0usize;
                while k < words as usize {
                    if pairs && k + 1 < words as usize {
                        let p = LLVMBuildIntToPtr(self.b, LLVMBuildAdd(self.b, load_addr, self.ci64(offsets[k] as u64), n), self.ptr, n);
                        let value = LLVMBuildLoad2(self.b, self.f64t, p, n);
                        LLVMSetAlignment(value, 4);
                        let (lo, hi) = self.split_pair(value);
                        self.define(results[k], lo);
                        self.define(results[k + 1], hi);
                        self.loaded_pairs.insert((results[k], results[k + 1]), value);
                        k += 2;
                    } else {
                        let p = LLVMBuildIntToPtr(self.b, LLVMBuildAdd(self.b, load_addr, self.ci64(offsets[k] as u64), n), self.ptr, n);
                        let load = LLVMBuildLoad2(self.b, elem, p, n);
                        LLVMSetAlignment(load, if shape == Shape::ScalarWords { if size == MemSize::B32 { 4 } else { 1 } } else if space == Space::Lds || size != MemSize::B32 { 1 } else { 4 });
                        if shape != Shape::ScalarWords { LLVMSetVolatile(load, volatile as i32); }
                        let value = if size == MemSize::B32 { load } else if size.signed() { LLVMBuildSExt(self.b, load, self.i32t, n) } else { LLVMBuildZExt(self.b, load, self.i32t, n) };
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
        if space == Space::Scratch {
            addr = LLVMBuildAdd(self.b, self.scratch_vec, LLVMBuildSExt(self.b, addr, self.vi64(), n), n);
        }
        if space == Space::Lds {
            addr = LLVMBuildAdd(self.b, self.splat(self.lds_base), LLVMBuildZExt(self.b, addr, self.vi64(), n), n);
        }
        if let Some(inside) = inside {
            let lane_offset = LLVMBuildSub(self.b, self.scratch_vec, self.splat(self.scratch_base_scalar), n);
            let physical = LLVMBuildAdd(self.b, addr, lane_offset, n);
            let inside = self.vector(inside);
            addr = LLVMBuildSelect(self.b, inside, physical, addr, n);
        }
        match shape {
            Shape::AtomicAdd { grouped } => {
                let d = self.vector(data[0]);
                if grouped { self.emit_grouped_atomic_add(addr, d, exec); return; }
                let packed_exec = self.vec_to_mask(exec);
                let ptrs = self.ptr_at_vec(addr, 0);
                let mut result = LLVMGetPoison(self.vi32());
                for k in 0..self.width() {
                    let bit = LLVMBuildAnd(self.b, LLVMBuildLShr(self.b, packed_exec, self.ci32(k), n), self.ci32(1), n);
                    let active = LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, bit, self.ci32(0), n);
                    let ptr = LLVMBuildExtractElement(self.b, ptrs, self.ci32(k), n);
                    let ptr = LLVMBuildSelect(self.b, active, ptr, self.bvh_scratch, n);
                    let value = LLVMBuildExtractElement(self.b, d, self.ci32(k), n);
                    let value = LLVMBuildSelect(self.b, active, value, self.ci32(0), n);
                    let old = LLVMBuildAtomicRMW(self.b, llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd, ptr, value, llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent, 0);
                    result = LLVMBuildInsertElement(self.b, result, old, self.ci32(k), n);
                }
                if let Some(&r) = results.first() { self.define(r, result); }
            }
            Shape::Store(StoreShape::Tile) => {
                let cols: Vec<LLVMValueRef> = (0..words as usize).map(|k| self.vector(data[k])).collect();
                let base = self.ptr_at_vec(addr, offsets[0] as u64);
                let sink = LLVMBuildPtrToInt(self.b, self.tile_sink, self.i64t, n);
                let addr_i = LLVMBuildPtrToInt(self.b, base, self.vi64(), n);
                let safe = LLVMBuildSelect(self.b, exec, addr_i, self.splat(sink), n);
                let w = self.width();
                for l in 0..w {
                    let mut parts = Vec::new();
                    let mut k = 0usize;
                    while k < words as usize {
                        if k + 1 < words as usize {
                            let mut mask = [self.ci32(l), self.ci32(w + l)];
                            parts.push(LLVMBuildShuffleVector(self.b, cols[k], cols[k + 1], LLVMConstVector(mask.as_mut_ptr(), 2), n));
                            k += 2;
                        } else {
                            let mut mask = [self.ci32(l)];
                            parts.push(LLVMBuildShuffleVector(self.b, cols[k], LLVMGetPoison(LLVMTypeOf(cols[k])), LLVMConstVector(mask.as_mut_ptr(), 1), n));
                            k += 1;
                        }
                    }
                    let row = self.vconcat_i32(&parts);
                    let a = LLVMBuildExtractElement(self.b, safe, self.ci32(l), n);
                    let p = LLVMBuildIntToPtr(self.b, a, self.ptr, n);
                    let st = LLVMBuildStore(self.b, row, p);
                    LLVMSetAlignment(st, 4);
                }
            }
            Shape::Store(kind) => {
                for k in 0..words as usize {
                    let ptrs = self.ptr_at_vec(addr, offsets[k] as u64);
                    let value = self.vector(data[k]);
                    match kind {
                        StoreShape::Narrow => {
                            let value = LLVMBuildTrunc(self.b, value, LLVMVectorType(elem, self.width()), n);
                            self.masked_scatter_ty(value, ptrs, exec, elem, 1);
                        }
                        StoreShape::Lds => self.masked_scatter_ty(value, ptrs, exec, self.i32t, 1),
                        StoreShape::Affine => self.affine_store(value, ptrs, exec),
                        StoreShape::Tile => unreachable!("tile stores are emitted as rows"),
                        StoreShape::Scatter => self.masked_scatter_ty(value, ptrs, exec, self.i32t, 4),
                    }
                }
            }
            Shape::NarrowLoad => {
                let value = self.masked_gather_ty(self.ptr_at_vec(addr, 0), exec, elem, 1);
                let value = if size.signed() { LLVMBuildSExt(self.b, value, self.vi32(), n) } else { LLVMBuildZExt(self.b, value, self.vi32(), n) };
                self.define(results[0], value);
            }
            Shape::PrivateTile { tile } => {
                let rowty = LLVMVectorType(self.i32t, words);
                let rows: Vec<_> = (0..self.width()).map(|l| {
                    let a = LLVMBuildExtractElement(self.b, addr, self.ci32(l), n);
                    let p = LLVMBuildIntToPtr(self.b, a, self.ptr, n);
                    let load = LLVMBuildLoad2(self.b, rowty, p, n);
                    LLVMSetAlignment(load, 4);
                    load
                }).collect();
                let cols = self.transpose_rows(&rows, words, tile);
                for k in 0..words as usize { self.define(results[k], cols[k]); }
            }
            Shape::Frame { stride_words: sp4, offset_words: ioff_w, group: grp } => {
                let base_v = self.vector(base);
                let nblk = self.width() / grp;
                let blkty = LLVMVectorType(self.i32t, grp * sp4);
                let poison_blk = LLVMGetPoison(blkty);
                let blocks: Vec<LLVMValueRef> = (0..nblk).map(|g| {
                    let a = LLVMBuildExtractElement(self.b, base_v, self.ci32(g * grp), n);
                    let p = LLVMBuildIntToPtr(self.b, a, self.ptr, n);
                    let ld = LLVMBuildLoad2(self.b, blkty, p, n);
                    LLVMSetAlignment(ld, 4);
                    ld
                }).collect();
                let extract = |cg: &Self, fw: u32| -> LLVMValueRef {
                    let parts: Vec<LLVMValueRef> = blocks.iter().map(|&blk| {
                        let mut idx: Vec<LLVMValueRef> = (0..grp).map(|lane| cg.ci32(lane * sp4 + fw)).collect();
                        let mask = LLVMConstVector(idx.as_mut_ptr(), grp);
                        LLVMBuildShuffleVector(cg.b, blk, poison_blk, mask, n)
                    }).collect();
                    cg.vconcat_i32(&parts)
                };
                for k in 0..words as usize { let v = extract(self, ioff_w + k as u32); self.define(results[k], v); }
            }
            Shape::Words { lanes, pairs } => {
                let mut k = 0usize;
                while k < words as usize {
                    if pairs && k + 1 < words as usize {
                        let ptrs = self.ptr_at_vec(addr, offsets[k] as u64);
                        let d = match lanes {
                            Lanes::Broadcast => self.bcast_load(ptrs, exec, nonempty, self.f64t),
                            _ => self.masked_gather_ty(ptrs, exec, self.f64t, 4),
                        };
                        let (lo, hi) = self.split_pair(d);
                        self.define(results[k], lo);
                        self.define(results[k + 1], hi);
                        self.loaded_pairs.insert((results[k], results[k + 1]), d);
                        k += 2;
                    } else {
                        let ptrs = self.ptr_at_vec(addr, offsets[k] as u64);
                        let d = match lanes {
                            Lanes::Broadcast => self.bcast_load(ptrs, exec, nonempty, self.i32t),
                            Lanes::Lds => self.masked_gather_ty(ptrs, exec, self.i32t, 1),
                            Lanes::Affine { allocated } => self.affine_load(ptrs, exec, allocated),
                            Lanes::Gather => self.masked_gather_ty(ptrs, exec, self.i32t, 4),
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

    unsafe fn emit_grouped_atomic_add(&self, addresses: LLVMValueRef, values: LLVMValueRef, exec: LLVMValueRef) {
        let b = self.b; let n = self.n();
        let addresses = LLVMBuildFreeze(b, addresses, n);
        let entry = LLVMGetInsertBlock(b); let function = LLVMGetBasicBlockParent(entry);
        let header = LLVMAppendBasicBlockInContext(self.ctx, function, cstr("atomic.groups").as_ptr());
        let body = LLVMAppendBasicBlockInContext(self.ctx, function, cstr("atomic.group").as_ptr());
        let done = LLVMAppendBasicBlockInContext(self.ctx, function, cstr("atomic.done").as_ptr());
        let initial = self.vec_to_mask(exec);
        LLVMBuildBr(b, header); LLVMPositionBuilderAtEnd(b, header);
        let pending = LLVMBuildPhi(b, self.i32t, n);
        LLVMAddIncoming(pending, [initial].as_mut_ptr(), [entry].as_mut_ptr(), 1);
        let nonempty = LLVMBuildICmp(b, llvm::LLVMIntPredicate::LLVMIntNE, pending, self.ci32(0), n);
        LLVMBuildCondBr(b, nonempty, body, done); LLVMPositionBuilderAtEnd(b, body);
        let lane = self.call("llvm.cttz.i32", self.i32t, &[self.i32t, self.i1], &[pending, LLVMConstInt(self.i1, 1, 0)]);
        let address = LLVMBuildExtractElement(b, addresses, lane, n);
        let equal = LLVMBuildICmp(b, llvm::LLVMIntPredicate::LLVMIntEQ, addresses, self.splat(address), n);
        let members = LLVMBuildAnd(b, self.vec_to_mask(equal), pending, n);
        let mask = self.mask_to_vec(members);
        let addends = LLVMBuildSelect(b, mask, values, LLVMConstNull(self.vi32()), n);
        let sum = self.call(&format!("llvm.vector.reduce.add.v{}i32", self.width()), self.i32t, &[self.vi32()], &[addends]);
        let pointer = LLVMBuildIntToPtr(b, address, self.ptr, n);
        LLVMBuildAtomicRMW(b, llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd, pointer, sum, llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent, 0);
        let remaining = LLVMBuildAnd(b, pending, LLVMBuildNot(b, members, n), n);
        let backedge = LLVMGetInsertBlock(b);
        LLVMBuildBr(b, header); LLVMAddIncoming(pending, [remaining].as_mut_ptr(), [backedge].as_mut_ptr(), 1);
        LLVMPositionBuilderAtEnd(b, done);
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
