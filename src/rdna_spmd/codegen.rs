pub(super) mod memory;
pub(super) mod ops;
pub(super) mod wave;

use std::collections::BTreeMap;
use std::ffi::CString;

use llvm_sys as llvm;
use llvm::core::*;
use llvm::prelude::*;

use super::analysis::masks::Exec;
use super::analysis::memory::Access;
use super::ir::{*, Cvt, Env, IntOp, Op, Ty, ValueId};
use super::lift::{Input, InputSource};
use ops::Emitter;
use crate::rdna_instructions::SourceOperand;


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Abi { Whole, Cooperative }

pub(super) struct Cluster {
    pub members: usize,
    pub lo: i64,
    pub span: u32,
    pub tile: u32,
}

pub(super) struct Prepared {
    pub registry: std::sync::Arc<super::dialect::DialectRegistry>,
    pub ir: VerifiedFunc,
    pub inputs: Vec<Input>,
    pub width: Option<u32>,
    pub abi: Abi,
    pub observable_return: bool,
    pub uniform: Vec<bool>,
    pub exec: Exec,
    pub constants: Vec<Option<u64>>,
    pub accesses: Vec<Access>,
    pub shapes: Vec<memory::Shape>,
    pub clusters: BTreeMap<usize, Cluster>,
    pub yields: BTreeMap<u64, super::engine::yields::YieldValues>,
    pub min_private_bytes: usize,
    pub num_vgprs: usize,
}

pub(super) fn resume_key(provenance: u64) -> usize { (provenance & !super::lift::wave::SCHEDULED) as usize }

impl Prepared {
    pub fn resume_layouts(&self) -> Vec<super::engine::yields::YieldValues> {
        self.yields.values().cloned().collect()
    }
    pub fn resume_index(&self, provenance: u64) -> usize {
        self.yields.keys().position(|&key| key == provenance).expect("scheduled effect lacks a yield layout")
    }
}

fn cstr(s: &str) -> CString { CString::new(s).unwrap() }

pub(super) fn reverse_postorder(f: &Func) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut visited = std::collections::BTreeSet::new();
    let mut stack: Vec<(BlockId, usize)> = vec![(f.entry, 0)];
    visited.insert(f.entry);
    while let Some((id, next)) = stack.last_mut() {
        let edges = f.blocks[id].term.edges();
        if *next < edges.len() {
            let dst = edges[*next].dst;
            *next += 1;
            if visited.insert(dst) { stack.push((dst, 0)); }
        } else {
            order.push(*id);
            stack.pop();
        }
    }
    order.reverse();
    for &id in f.blocks.keys() { if visited.insert(id) { order.push(id); } }
    order
}

pub(super) struct Cg<'a> {
    p: &'a Prepared,
    ctx: LLVMContextRef,
    module: LLVMModuleRef,
    func: LLVMValueRef,
    b: LLVMBuilderRef,
    em: Emitter,
    sem: Emitter,
    values: Vec<LLVMValueRef>,
    vectors: Vec<LLVMValueRef>,
    scalars: Vec<LLVMValueRef>,
    bbs: BTreeMap<BlockId, LLVMBasicBlockRef>,
    phis: BTreeMap<BlockId, Vec<LLVMValueRef>>,
    incoming: BTreeMap<BlockId, Vec<(LLVMBasicBlockRef, Vec<LLVMValueRef>)>>,
    param_scalar: Vec<bool>,
    types: Vec<Ty>,
    definitions: Vec<Option<Op>>,
    access_at: BTreeMap<(BlockId, usize), usize>,
    skip: std::collections::BTreeSet<(BlockId, usize)>,
    current: BlockId,
    sgprs_p: LLVMValueRef,
    vgprs_p: LLVMValueRef,
    scratch_base_scalar: LLVMValueRef,
    scratch_vec: LLVMValueRef,
    lds_base: LLVMValueRef,
    spill_base: LLVMValueRef,
    spill: std::cell::RefCell<BTreeMap<(u32, u32), usize>>,
    loaded_pairs: BTreeMap<(ValueId, ValueId), LLVMValueRef>,
    valid_mask: LLVMValueRef,
    yield_frame: LLVMValueRef,
    sink: LLVMValueRef,
    store_sink: LLVMValueRef,
    tile_sink: LLVMValueRef,
    i1: LLVMTypeRef, i32t: LLVMTypeRef, i64t: LLVMTypeRef, f32t: LLVMTypeRef, f64t: LLVMTypeRef, ptr: LLVMTypeRef,
}

pub(super) unsafe fn compile(p: &Prepared, name: &str, mode: super::jit::Mode) -> super::jit::NativeCode {
    let native = super::jit::Module::new(name);
    let ctx = native.ctx;
    let module = native.module;
    let b = native.builder;
    let i1 = LLVMInt1TypeInContext(ctx);
    let i32t = LLVMInt32TypeInContext(ctx);
    let i64t = LLVMInt64TypeInContext(ctx);
    let f32t = LLVMFloatTypeInContext(ctx);
    let f64t = LLVMDoubleTypeInContext(ctx);
    let ptr = LLVMPointerTypeInContext(ctx, 0);
    let void = LLVMVoidTypeInContext(ctx);
    let coop = p.abi == Abi::Cooperative;
    let func = if coop {
        let mut params = [ptr, ptr, i64t, i64t, ptr, i64t, i64t, ptr, i32t];
        LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, LLVMFunctionType(i64t, params.as_mut_ptr(), 9, 0))
    } else if p.width.is_some() {
        let mut params = [ptr, ptr, i64t, i64t];
        LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, LLVMFunctionType(void, params.as_mut_ptr(), 4, 0))
    } else {
        let mut params = [ptr, ptr, i64t];
        LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, LLVMFunctionType(void, params.as_mut_ptr(), 3, 0))
    };
    let entry = LLVMAppendBasicBlockInContext(ctx, func, b"entry\0".as_ptr() as *const _);
    LLVMPositionBuilderAtEnd(b, entry);
    let n = b"\0".as_ptr().cast();
    let sgprs_p = LLVMGetParam(func, 0);
    let vgprs_p = LLVMGetParam(func, 1);
    let scratch_base = LLVMGetParam(func, 2);
    let scratch_stride = if coop || p.width.is_some() { LLVMGetParam(func, 3) } else { LLVMConstInt(i64t, 0, 0) };
    let lane_base = if coop { LLVMGetParam(func, 6) } else { LLVMConstInt(i64t, 0, 0) };
    let lds_base = if coop { LLVMGetParam(func, 5) } else if p.width.is_some() { LLVMConstInt(i64t, 0, 0) } else { LLVMGetUndef(i64t) };
    let width_lanes = p.width.unwrap_or(1);
    let valid_mask = if coop { LLVMGetParam(func, 8) } else { LLVMConstInt(i32t, if p.width.is_some() { (1u64 << width_lanes) - 1 } else { u32::MAX as u64 }, 0) };
    let spill_base = if coop { LLVMGetParam(func, 4) } else {
        LLVMBuildArrayAlloca(b, i32t, LLVMConstInt(i32t, super::engine::kernel::COOP_SPILL_SLOTS as u64, 0), n)
    };
    let scratch_base_scalar = if coop || p.width.is_none() {
        let aperture = LLVMBuildAnd(b, scratch_base, LLVMConstInt(i64t, 0xffff_ffff_0000_0000, 0), n);
        let sized = LLVMBuildICmp(b, llvm::LLVMIntPredicate::LLVMIntNE, scratch_stride, LLVMConstInt(i64t, 0, 0), n);
        LLVMBuildSelect(b, sized, aperture, scratch_base, n)
    } else { scratch_base };
    let sink = LLVMBuildArrayAlloca(b, i32t, LLVMConstInt(i32t, 10, 0), n);
    let mut em = Emitter::new(b, p.width, p.registry.clone());
    em.state = p.registry.lowering_state(&em, sink);
    let mut sem = Emitter::new(b, None, p.registry.clone());
    sem.state = p.registry.lowering_state(&sem, sink);
    let scratch_env = if p.width.is_some() { (scratch_base_scalar, scratch_stride) } else {
        let base = if coop { LLVMBuildAdd(b, scratch_base, LLVMBuildMul(b, scratch_stride, lane_base, n), n) } else { scratch_base };
        (scratch_base_scalar, if coop { scratch_stride } else { LLVMConstInt(i64t, 0, 0) }).0;
        (base, scratch_stride)
    };
    em.scratch = Some((scratch_base_scalar, if coop || p.width.is_some() { scratch_stride } else { LLVMConstInt(i64t, 0, 0) }));
    sem.scratch = em.scratch;
    let cells = p.yields.values().map(|l| l.cells()).max().unwrap_or(0);
    let yield_frame = if cells == 0 { LLVMConstNull(ptr) } else {
        let frame = LLVMBuildAlloca(b, LLVMArrayType2(i32t, cells as u64 * width_lanes as u64), cstr("yield.values").as_ptr());
        LLVMSetAlignment(frame, 64);
        frame
    };
    let f = p.ir.func();
    let mut definitions = vec![None; f.types.len()];
    for block in f.blocks.values() { for inst in &block.insts { if let Inst::Core { value, op, .. } = inst { definitions[value.0] = Some(*op); } } }
    let mut access_at = BTreeMap::new();
    let mut skip = std::collections::BTreeSet::new();
    for (index, access) in p.accesses.iter().enumerate() {
        access_at.insert((access.block, access.effects[0]), index);
        for &e in &access.effects[1..] { skip.insert((access.block, e)); }
    }
    for (&start, cluster) in &p.clusters {
        for member in start + 1..start + cluster.members { for &e in &p.accesses[member].effects { skip.insert((p.accesses[member].block, e)); } }
    }
    let mut cg = Cg {
        p, ctx, module, func, b, em, sem,
        values: vec![std::ptr::null_mut(); f.types.len()],
        vectors: vec![std::ptr::null_mut(); f.types.len()],
        scalars: vec![std::ptr::null_mut(); f.types.len()],
        bbs: BTreeMap::new(), phis: BTreeMap::new(), incoming: BTreeMap::new(),
        param_scalar: (0..f.types.len()).map(|v| p.width.is_none() || (p.uniform[v] && std::env::var("AMDGPU_SIM_NOSCALAR").map_or(true, |x| x != "1"))).collect(),
        types: f.types.clone(), definitions, access_at, skip,
        current: f.entry,
        sgprs_p, vgprs_p, scratch_base_scalar, scratch_vec: scratch_base, lds_base, spill_base,
        spill: std::cell::RefCell::new(BTreeMap::new()),
        loaded_pairs: BTreeMap::new(),
        valid_mask, yield_frame, sink,
        store_sink: LLVMBuildAlloca(b, i64t, cstr("store_sink").as_ptr()),
        tile_sink: LLVMBuildArrayAlloca(b, i32t, LLVMConstInt(i32t, 64, 0), cstr("tile_sink").as_ptr()),
        i1, i32t, i64t, f32t, f64t, ptr,
    };
    let _ = scratch_env;
    if let Some(w) = p.width {
        let mut lanes: Vec<LLVMValueRef> = (0..w).map(|k| LLVMConstInt(i64t, k as u64, 0)).collect();
        let lane_idx = LLVMConstVector(lanes.as_mut_ptr(), w);
        let base_v = cg.splat64(scratch_base);
        let stride_v = cg.splat64(scratch_stride);
        let lane_base_v = cg.splat64(lane_base);
        let scratch_lane = LLVMBuildAdd(b, lane_base_v, lane_idx, n);
        let off = LLVMBuildMul(b, scratch_lane, stride_v, n);
        cg.scratch_vec = LLVMBuildAdd(b, base_v, off, n);
    } else {
        cg.scratch_vec = if coop { LLVMBuildAdd(b, scratch_base, LLVMBuildMul(b, scratch_stride, lane_base, n), n) } else { scratch_base };
    }
    let valid_vec = cg.mask_to_vec(valid_mask);
    cg.em.valid_lane = Some(valid_vec);
    cg.em.set_lane_id(lane_base);
    cg.sem.set_lane_id(lane_base);
    cg.sem.valid_lane = Some(LLVMBuildICmp(b, llvm::LLVMIntPredicate::LLVMIntNE, LLVMBuildAnd(b, valid_mask, LLVMConstInt(i32t, 1, 0), n), LLVMConstInt(i32t, 0, 0), n));
    for &id in f.blocks.keys() {
        let name = cstr(&format!("b{:x}", id.0));
        cg.bbs.insert(id, LLVMAppendBasicBlockInContext(ctx, func, name.as_ptr()));
    }
    cg.load_entry();
    LLVMBuildBr(b, cg.bbs[&f.entry]);
    let order: Vec<BlockId> = if std::env::var("AMDGPU_SIM_RPO").map_or(true, |v| v != "0") { reverse_postorder(f) } else { f.blocks.keys().copied().collect() };
    let counts = std::env::var("AMDGPU_SIM_BLOCK_COUNTS").ok().map(|path| {
        let ty = LLVMArrayType2(i64t, f.blocks.len() as u64);
        let global = LLVMAddGlobal(module, ty, b"block_counts\0".as_ptr().cast());
        LLVMSetInitializer(global, LLVMConstNull(ty));
        let index: BTreeMap<BlockId, usize> = f.blocks.keys().enumerate().map(|(i, &id)| (id, i)).collect();
        let text: String = f.blocks.keys().map(|id| format!("{} b{:x}\n", index[id], id.0)).collect();
        std::fs::write(format!("{path}.blocks"), text).unwrap();
        (path, global, index)
    });
    for id in order {
        let block = &f.blocks[&id];
        LLVMPositionBuilderAtEnd(b, cg.bbs[&id]);
        cg.current = id;
        cg.begin_block(id, block);
        if let Some((_, global, index)) = &counts {
            let slot = LLVMBuildGEP2(b, i64t, *global, [LLVMConstInt(i64t, index[&id] as u64, 0)].as_mut_ptr(), 1, n);
            LLVMBuildAtomicRMW(b, llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd, slot, LLVMConstInt(i64t, 1, 0), llvm::LLVMAtomicOrdering::LLVMAtomicOrderingMonotonic, 0);
        }
        for (index, inst) in block.insts.iter().enumerate() {
            if cg.skip.contains(&(id, index)) { continue; }
            cg.emit_inst(id, index, inst);
        }
        cg.emit_term(id, block);
    }
    cg.finish_phis();
    let mut code = native.finish(mode);
    if let Some((path, _, index)) = counts { code.block_counts = Some((path, index.len())); }
    code
}

impl<'a> Cg<'a> {
    fn regs(&self) -> super::dialect::Registers { self.p.registry.registers() }

    fn n(&self) -> *const std::ffi::c_char { b"\0".as_ptr() as *const _ }
    fn mask_words(&self) -> bool { self.p.width.is_some() && std::env::var("AMDGPU_SIM_MASK_WORDS").map_or(false, |v| v == "1") }
    fn width(&self) -> u32 { self.p.width.unwrap_or(1) }
    unsafe fn ci32(&self, v: u32) -> LLVMValueRef { LLVMConstInt(self.i32t, v as u64, 0) }
    unsafe fn ci64(&self, v: u64) -> LLVMValueRef { LLVMConstInt(self.i64t, v, 0) }
    unsafe fn vec_ty(&self, scalar: LLVMTypeRef) -> LLVMTypeRef { self.p.width.map_or(scalar, |w| LLVMVectorType(scalar, w)) }
    unsafe fn splat(&self, v: LLVMValueRef) -> LLVMValueRef {
        let Some(w) = self.p.width else { return v; };
        let vty = LLVMVectorType(LLVMTypeOf(v), w);
        let poison = LLVMGetPoison(vty);
        let ins = LLVMBuildInsertElement(self.b, poison, v, self.ci32(0), self.n());
        let mask = LLVMConstNull(LLVMVectorType(self.i32t, w));
        LLVMBuildShuffleVector(self.b, ins, poison, mask, self.n())
    }
    unsafe fn splat64(&self, v: LLVMValueRef) -> LLVMValueRef { self.splat(v) }
    unsafe fn is_vector(&self, v: LLVMValueRef) -> bool { LLVMGetTypeKind(LLVMTypeOf(v)) == llvm::LLVMTypeKind::LLVMVectorTypeKind }
    unsafe fn mask_to_vec(&self, word: LLVMValueRef) -> LLVMValueRef {
        match self.p.width {
            Some(w) => {
                let iw = LLVMIntTypeInContext(self.ctx, w);
                let bits = LLVMBuildTrunc(self.b, word, iw, self.n());
                LLVMBuildBitCast(self.b, bits, LLVMVectorType(self.i1, w), self.n())
            }
            None => LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, LLVMBuildAnd(self.b, word, self.ci32(1), self.n()), self.ci32(0), self.n()),
        }
    }
    unsafe fn vec_to_mask(&self, v: LLVMValueRef) -> LLVMValueRef {
        match self.p.width {
            Some(w) => {
                let iw = LLVMIntTypeInContext(self.ctx, w);
                let bits = LLVMBuildBitCast(self.b, v, iw, self.n());
                LLVMBuildZExt(self.b, bits, self.i32t, self.n())
            }
            None => LLVMBuildZExt(self.b, v, self.i32t, self.n()),
        }
    }
    fn describe(&self, v: ValueId) -> String {
        let f = self.p.ir.func();
        for (&id, block) in &f.blocks {
            if let Some(index) = block.params.iter().position(|p| p.0 == v) { return format!("parameter {index} of b{:x}", id.0); }
            for (index, inst) in block.insts.iter().enumerate() {
                let defined = match inst {
                    Inst::Core { value, .. } | Inst::Packet { output: value, .. } => *value == v,
                    Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => outputs.iter().any(|o| o.0 == v),
                };
                if defined {
                    let accesses: Vec<String> = self.p.accesses.iter().enumerate().filter(|(_, a)| a.block == id && a.effects.contains(&index) || a.block == id && (a.start..a.end).contains(&index))
                        .map(|(k, a)| format!("access {k} effects {:?} shape {:?} cluster {:?} results {:?}", a.effects, self.p.shapes[k], self.p.clusters.get(&k).map(|c| c.members), a.results)).collect();
                    return format!("b{:x}:{index} {} (current b{:x}, skipped {}) {:?}", id.0, super::ir::print::inst(&self.p.registry, &f.types, inst), self.current.0, self.skip.contains(&(id, index)), accesses);
                }
            }
        }
        "undefined".into()
    }
    unsafe fn vector(&mut self, v: ValueId) -> LLVMValueRef {
        let value = self.values[v.0];
        assert!(!value.is_null(), "undefined SSA value v{}: {}", v.0, self.describe(v));
        if self.p.width.is_none() || self.is_vector(value) { return value; }
        if !self.vectors[v.0].is_null() { return self.vectors[v.0]; }
        let out = self.splat(value);
        self.vectors[v.0] = out;
        out
    }
    unsafe fn scalar(&mut self, v: ValueId) -> LLVMValueRef {
        let value = self.values[v.0];
        assert!(!value.is_null(), "undefined SSA value v{}: {}", v.0, self.describe(v));
        if !self.is_vector(value) { return value; }
        if !self.scalars[v.0].is_null() { return self.scalars[v.0]; }
        let out = LLVMBuildExtractElement(self.b, value, self.ci32(0), self.n());
        self.scalars[v.0] = out;
        out
    }
    unsafe fn shaped(&mut self, v: ValueId, scalar: bool) -> LLVMValueRef { if scalar { self.scalar(v) } else { self.vector(v) } }
    unsafe fn define(&mut self, v: ValueId, value: LLVMValueRef) {
        self.values[v.0] = value;
        self.vectors[v.0] = std::ptr::null_mut();
        self.scalars[v.0] = std::ptr::null_mut();
    }

    unsafe fn load_entry(&mut self) {
        let f = self.p.ir.func();
        let entry = &f.blocks[&f.entry];
        let coop = self.p.abi == Abi::Cooperative;
        let n = self.n();
        for (index, &(id, ty)) in entry.params.iter().enumerate() {
            let input = &self.p.inputs[index];
            let value = match input.source {
                InputSource::Operand(SourceOperand::VectorRegister(r)) => {
                    let r = r as u32;
                    if let Some(w) = self.p.width {
                        let gep = LLVMBuildGEP2(self.b, self.i32t, self.vgprs_p, [self.ci32(r * w)].as_mut_ptr(), 1, n);
                        let load = LLVMBuildLoad2(self.b, LLVMVectorType(self.i32t, w), gep, n);
                        LLVMSetAlignment(load, 4);
                        load
                    } else {
                        let gep = LLVMBuildGEP2(self.b, self.i32t, self.vgprs_p, [self.ci32(r)].as_mut_ptr(), 1, n);
                        LLVMBuildLoad2(self.b, self.i32t, gep, n)
                    }
                }
                InputSource::Operand(SourceOperand::ScalarRegister(r)) => {
                    let r = r as u32;
                    if r == self.regs().null { self.ci32(0) } else {
                        let gep = LLVMBuildGEP2(self.b, self.i32t, self.sgprs_p, [self.ci32(r)].as_mut_ptr(), 1, n);
                        LLVMBuildLoad2(self.b, self.i32t, gep, n)
                    }
                }
                InputSource::MaskBit(r) => {
                    let word = if r == self.regs().exec && self.p.abi == Abi::Whole {
                        if self.p.width.is_some() { self.ci32(((1u64 << self.width()) - 1) as u32) } else { self.ci32(1) }
                    } else {
                        let gep = LLVMBuildGEP2(self.b, self.i32t, self.sgprs_p, [self.ci32(r)].as_mut_ptr(), 1, n);
                        let word = LLVMBuildLoad2(self.b, self.i32t, gep, n);
                        if r == self.regs().exec && coop { LLVMBuildAnd(self.b, word, self.valid_mask, n) } else { word }
                    };
                    self.mask_to_vec(word)
                }
                InputSource::Scc => {
                    if coop {
                        let gep = LLVMBuildGEP2(self.b, self.i32t, self.sgprs_p, [self.ci32(self.regs().scc_slot)].as_mut_ptr(), 1, n);
                        let word = LLVMBuildLoad2(self.b, self.i32t, gep, n);
                        LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, word, self.ci32(0), n)
                    } else { LLVMConstInt(self.i1, 0, 0) }
                }
                _ => panic!("unsupported entry parameter binding {:?}", input),
            };
            assert_eq!(ty, input.ty);
            self.define(id, value);
        }
    }

    unsafe fn begin_block(&mut self, id: BlockId, block: &Block) {
        let f = self.p.ir.func();
        if id == f.entry { return; }
        let mut phis = Vec::new();
        let words = self.mask_words();
        for &(v, ty) in &block.params {
            let scalar = self.param_scalar[v.0];
            let t = if scalar { self.sem.ty(ty) } else if ty == Ty::I1 && words { LLVMIntTypeInContext(self.ctx, self.width()) } else { self.em.ty(ty) };
            let phi = LLVMBuildPhi(self.b, t, self.n());
            phis.push(phi);
            self.define(v, phi);
        }
        for (&(v, ty), &phi) in block.params.iter().zip(&phis) {
            if !self.param_scalar[v.0] && ty == Ty::I1 && words {
                let vec = LLVMBuildBitCast(self.b, phi, self.em.ty(Ty::I1), self.n());
                self.define(v, vec);
            }
        }
        self.phis.insert(id, phis);
    }

    unsafe fn finish_phis(&mut self) {
        let f = self.p.ir.func();
        for (&id, phis) in &self.phis {
            let incoming = self.incoming.get(&id).cloned().unwrap_or_default();
            for (index, &phi) in phis.iter().enumerate() {
                let mut blocks: Vec<LLVMBasicBlockRef> = incoming.iter().map(|(bb, _)| *bb).collect();
                let mut values: Vec<LLVMValueRef> = incoming.iter().map(|(_, args)| args[index]).collect();
                if values.is_empty() {
                    let ty = LLVMTypeOf(phi);
                    LLVMReplaceAllUsesWith(phi, LLVMGetUndef(ty));
                    LLVMInstructionEraseFromParent(phi);
                    continue;
                }
                LLVMAddIncoming(phi, values.as_mut_ptr(), blocks.as_mut_ptr(), values.len() as u32);
            }
        }
        let _ = f;
    }

    unsafe fn edge_args(&mut self, edge: &Edge) -> Vec<LLVMValueRef> {
        let f = self.p.ir.func();
        let params: Vec<_> = f.blocks[&edge.dst].params.iter().map(|p| p.0).collect();
        let words = self.mask_words();
        edge.args.iter().zip(params).map(|(&arg, param)| {
            let scalar = self.param_scalar[param.0];
            let value = self.shaped(arg, scalar);
            if !scalar && words && self.types[param.0] == Ty::I1 {
                LLVMBuildBitCast(self.b, value, LLVMIntTypeInContext(self.ctx, self.width()), self.n())
            } else { value }
        }).collect()
    }

    unsafe fn branch_to(&mut self, edge: &Edge) -> LLVMBasicBlockRef {
        let args = self.edge_args(edge);
        let from = LLVMGetInsertBlock(self.b);
        self.incoming.entry(edge.dst).or_default().push((from, args));
        self.bbs[&edge.dst]
    }

    unsafe fn emit_term(&mut self, id: BlockId, block: &Block) {
        let n = self.n();
        match &block.term {
            Term::Br(edge) => { let bb = self.branch_to(edge); LLVMBuildBr(self.b, bb); }
            Term::CondBr { cond, yes, no } => {
                let c = self.scalar(*cond);
                let yes_bb = self.branch_to(yes);
                let no_bb = self.branch_to(no);
                LLVMBuildCondBr(self.b, c, yes_bb, no_bb);
            }
            Term::Ret(args) => {
                let coop = self.p.abi == Abi::Cooperative;
                if coop && self.p.observable_return { self.store_return(args); }
                if coop { LLVMBuildRet(self.b, self.ci64(super::engine::kernel::COOP_DONE)); } else { LLVMBuildRetVoid(self.b); }
            }
        }
        let _ = (id, n);
    }

    unsafe fn any_of_word(&mut self, input: ValueId) -> Option<LLVMValueRef> {
        let Some(w) = self.p.width else { return None; };
        let (bit, valid) = match self.definitions[input.0] {
            Some(Op::Int(IntOp::And, a, b)) if matches!(self.definitions[b.0], Some(Op::Env(Env::ValidLane))) => (a, true),
            Some(Op::Int(IntOp::And, a, b)) if matches!(self.definitions[a.0], Some(Op::Env(Env::ValidLane))) => (b, true),
            _ => (input, false),
        };
        let Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) = self.definitions[bit.0] else { return None; };
        let Some(Op::Int(IntOp::LShr, word, lane)) = self.definitions[shifted.0] else { return None; };
        if !matches!(self.definitions[lane.0], Some(Op::Env(Env::PacketLaneId))) || !self.p.uniform[word.0] { return None; }
        let n = self.n();
        let word = self.scalar(word);
        let mut bits = LLVMBuildAnd(self.b, word, self.ci32(((1u64 << w) - 1) as u32), n);
        if valid { bits = LLVMBuildAnd(self.b, bits, self.valid_mask, n); }
        Some(LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, bits, self.ci32(0), n))
    }

    unsafe fn store_return(&mut self, args: &[ValueId]) {
        let n = self.n();
        let entry = &self.p.ir.func().blocks[&self.p.ir.func().entry];
        for (index, &arg) in args.iter().enumerate() {
            let input = &self.p.inputs[index];
            if arg == entry.params[index].0 { continue; }
            match input.source {
                InputSource::Operand(SourceOperand::VectorRegister(r)) => {
                    let r = r as u32;
                    if r as usize >= self.p.num_vgprs { continue; }
                    let value = self.vector(arg);
                    let gep = LLVMBuildGEP2(self.b, self.i32t, self.vgprs_p, [self.ci32(r * self.width())].as_mut_ptr(), 1, n);
                    let store = LLVMBuildStore(self.b, value, gep);
                    LLVMSetAlignment(store, 4);
                }
                InputSource::Operand(SourceOperand::ScalarRegister(r)) => {
                    let r = r as u32;
                    if r == self.regs().null { continue; }
                    let value = self.scalar(arg);
                    let gep = LLVMBuildGEP2(self.b, self.i32t, self.sgprs_p, [self.ci32(r)].as_mut_ptr(), 1, n);
                    LLVMBuildStore(self.b, value, gep);
                }
                InputSource::MaskBit(r) => {
                    let value = self.vector(arg);
                    let word = self.vec_to_mask(value);
                    let gep = LLVMBuildGEP2(self.b, self.i32t, self.sgprs_p, [self.ci32(r)].as_mut_ptr(), 1, n);
                    LLVMBuildStore(self.b, word, gep);
                }
                InputSource::Scc => {
                    let value = self.scalar(arg);
                    let word = LLVMBuildZExt(self.b, value, self.i32t, n);
                    let gep = LLVMBuildGEP2(self.b, self.i32t, self.sgprs_p, [self.ci32(self.regs().scc_slot)].as_mut_ptr(), 1, n);
                    LLVMBuildStore(self.b, word, gep);
                }
                _ => {}
            }
        }
    }

    unsafe fn emit_inst(&mut self, id: BlockId, index: usize, inst: &Inst) {
        match inst {
            Inst::Core { value, ty, op } => self.emit_core(*value, *ty, *op),
            Inst::Target { op, args, outputs, .. } => {
                let values: Vec<ValueId> = args.values().to_vec();
                let scalar = self.p.width.is_none();
                let mut table = self.values.clone();
                for a in &values { table[a.0] = self.shaped(*a, scalar); }
                let emitter = if scalar { &self.sem } else { &self.em };
                let results = emitter.target(*op, *args, &table);
                for (&(out, _), result) in outputs.iter().zip(results) { self.define(out, result); }
            }
            Inst::Packet { op, input, output } => {
                let n = self.n();
                if *op == PacketOp::Any {
                    if let Some(result) = self.any_of_word(*input) { self.define(*output, result); return; }
                }
                let bits = match self.p.width {
                    Some(w) => {
                        let v = self.vector(*input);
                        let iw = LLVMIntTypeInContext(self.ctx, w);
                        LLVMBuildZExt(self.b, LLVMBuildBitCast(self.b, v, iw, n), self.i32t, n)
                    }
                    None => { let v = self.scalar(*input); LLVMBuildZExt(self.b, v, self.i32t, n) }
                };
                let result = if *op == PacketOp::Any { LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, bits, self.ci32(0), n) } else { bits };
                self.define(*output, result);
            }
            Inst::Effect { provenance, op, inputs, outputs } => match op {
                EffectOp::Memory { .. } => {
                    let access = self.access_at[&(id, index)];
                    if let Some(cluster) = self.p.clusters.get(&access) {
                        let members: Vec<usize> = (access..access + cluster.members).collect();
                        self.emit_cluster(&members, cluster);
                    } else {
                        self.emit_memory(access, id, index);
                    }
                }
                EffectOp::Wave(_) | EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => {
                    if *provenance & super::lift::wave::SCHEDULED != 0 || !matches!(op, EffectOp::Wave(_)) {
                        self.emit_yield(*provenance, inputs, outputs);
                    } else {
                        self.emit_local_wave(*op, inputs, outputs);
                    }
                }
            },
        }
    }

    unsafe fn emit_core(&mut self, value: ValueId, ty: Ty, op: Op) {
        if let Op::Pack64(a, b) = op {
            if let Some(&loaded) = self.loaded_pairs.get(&(a, b)) {
                let n = self.n();
                let result = LLVMBuildBitCast(self.b, loaded, self.vec_ty(self.i64t), n);
                self.define(value, result);
                return;
            }
        }
        let n = self.n();
        if let Op::Convert(Cvt::Trunc, Ty::I1, shift) = op {
            if let Some(Op::Int(IntOp::LShr, word, lane)) = self.definitions[shift.0] {
                if matches!(self.definitions[lane.0], Some(Op::Env(Env::PacketLaneId))) && self.p.uniform[word.0] {
                    let w = self.scalar(word);
                    let out = self.mask_to_vec(w);
                    self.define(value, out);
                    return;
                }
            }
        }
        if let Op::Select(c, a, b) = op {
            if a == b { let v = self.values[a.0]; self.define(value, v); return; }
            let _ = c;
        }
        let mut args = vec![];
        op.map(|id| { if !args.contains(&id) { args.push(id); } id });
        let scalar = self.p.width.is_none() || (!matches!(op, Op::Env(Env::PacketLaneId | Env::LaneId | Env::ValidLane))
            && std::env::var("AMDGPU_SIM_NOSCALAR").map_or(true, |x| x != "1")
            && args.iter().all(|a| !self.is_vector(self.values[a.0])));
        let mut table = self.values.clone();
        for a in &args { table[a.0] = self.shaped(*a, scalar); }
        let result = if scalar { self.sem.op(ty, op, &table) } else { self.em.op(ty, op, &table) };
        let _ = n;
        self.define(value, result);
    }
}
