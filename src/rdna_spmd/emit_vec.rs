//! Width-W SPMD codegen: lower a typed SSA function to a native function that
//! processes **W work-items at once** (one per SIMD lane), and JIT it with ORC.
//!
//! It uses the same scalar control-flow graph as the single-work-item emitter,
//! but represents each VGPR as a host SIMD vector containing W work-items.
//! Per-lane writes normally preserve inactive lanes according to EXEC. A
//! liveness analysis may omit that preservation for transient values that are
//! not observed after control-flow reconvergence.
//!
//! Key model (general, not kernel-specific):
//!   * **SGPR / SALU / SOPC / SMEM / SCC stay scalar** — they are wavefront-
//!     uniform by the RDNA ISA. Lane masks (EXEC/VCC/saved masks) live in the
//!     low W bits of their (scalar i32) register, exactly as 32-bit wave masks,
//!     so all the mask bit-arithmetic (saveexec / s_and / s_or / reconverge) is
//!     unchanged 32-bit integer code.
//!   * **VGPR / VALU / VMEM / compares widen** to `<W×i32>` / `<W×f64>`.
//!     Conversions happen only at the EXEC-mask <-> vector boundary
//!     (`mask_to_vec` for predication, `vec_to_mask` for compares).
//!   * Control flow follows the kernel's own EXEC machinery: `s_cbranch_execz`
//!     tests "no lane active" (low W bits all zero) = the masked-SIMT model.
//!   * Vector writes are predicated per-lane on EXEC, except when
//!     [`vec_live`](super::vec_live) determines that the destination is not
//!     observed after reconvergence.
//!
//! The scalar [`super::emit`] path is separate. Callers select this vector path
//! by compiling with a width `W > 0`.

mod memory;

use std::collections::BTreeMap;
use std::ffi::CString;

use llvm_sys as llvm;
use llvm::prelude::{LLVMBasicBlockRef, LLVMBuilderRef, LLVMTypeRef, LLVMValueRef};

use crate::rdna_instructions::SourceOperand;

use super::boundary::RegSet;
use super::packet_plan::{PacketPlan, InstructionAction};
use super::memory_shape::{Lanes, PacketShape, StoreShape};
use super::sqrt_idiom::SqrtCollapse;

const EXEC: u32 = 126;

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// A JIT-compiled width-W kernel. Processes W work-items per `run` call.
pub struct VecKernel {
    code: super::jit::NativeCode,
    pub num_vgprs: usize,
    pub width: u32,
    /// Per-lane allocation required by statically addressed private loads.
    pub min_private_bytes: usize,
}
impl VecKernel {
    /// Run W work-items. `sgprs` -> 128 u32 (shared/uniform); `vgprs` ->
    /// `num_vgprs * W` u32 in SoA layout (register r, lanes 0..W at `r*W`);
    /// `scratch_base` = base of W contiguous per-lane private segments of
    /// `scratch_stride` bytes each, at least `min_private_bytes`. All W segments
    /// must be allocated even if EXEC disables a lane.
    pub unsafe fn run(&self, sgprs: *mut u32, vgprs: *mut u32, scratch_base: u64, scratch_stride: u64) {
        let f = std::mem::transmute::<u64, extern "C" fn(*mut u32, *mut u32, u64, u64)>(self.code.address());
        f(sgprs, vgprs, scratch_base, scratch_stride);
    }
}

/// A resumable width-W packet used by the wave-owned cross-lane scheduler.
/// One invocation carries W lanes through SSA values, suspending at wave
/// boundaries to exchange typed operands and results with the scheduler.
pub struct CoopVecKernel {
    pub(crate) yields: BTreeMap<usize, super::yield_values::YieldValues>,
    pub(super) code: super::jit::NativeCode,
    pub num_vgprs: usize,
    pub width: u32,
    /// Per-lane allocation required by statically addressed private loads.
    pub min_private_bytes: usize,
}

impl CoopVecKernel {
    /// Entry address of the compiled packet kernel. It is not callable
    /// directly: the kernel yields by switching stacks, so it has to be
    /// started on a fiber ([`super::fiber::FiberCtx`] documents its
    /// arguments).
    pub fn addr(&self) -> u64 {
        self.code.address()
    }
}

use super::native_state::{CellId, State};

struct Cg {
    state: std::cell::RefCell<State>,
    ctx: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    func: LLVMValueRef,
    b: LLVMBuilderRef,
    yield_frame: LLVMValueRef,
    w: u32,
    coop: bool,
    num_vgprs: usize,
    // per-lane scratch base vector <W×i64> (broadcast(base) + lane*stride)
    scratch_vec: LLVMValueRef,
    store_sink: std::cell::Cell<LLVMValueRef>,
    // uniform scratch base (i64 scalar) — lane 0's segment. Used for scalar
    // SRC_PRIVATE_BASE reads (`s_mov_b64 s[..], src_private_base`): the aperture
    // high word is uniform across lanes, and the kernel adds the per-lane low
    // offset from VGPRs.
    scratch_base_scalar: LLVMValueRef,
    lds_base: LLVMValueRef,
    // per-lane scratch segment stride in bytes (i64 scalar). The private aperture
    // for a lane is [scratch_base, scratch_base+stride).
    scratch_stride: LLVMValueRef,
    sgpr: Vec<CellId>, // 128 scalar i32 representations (incl. EXEC/VCC mask regs)
    vgpr: Vec<CellId>, // num_vgprs <W×i32> representations
    // Parallel <W×f64> storage for each VGPR pair whose low register is r. This
    // lets an f64 value remain one vector across instructions instead of being
    // repeatedly rebuilt from two <W×i32> halves. `f64_fresh` bit r means that
    // this f64 storage contains the current value of r:r+1. Predicated writes
    // update it with a per-lane select, so it remains valid for inactive lanes.
    vgpr_f64: Vec<CellId>,
    // 256-bit RegSet, not u128: VGPRs number up to 256 and `& 127` indexing
    // aliases pair p with p+128 (a v153:v154 write would falsely mark v25:v26).
    f64_fresh: std::cell::Cell<super::regtype::RegSet>,
    // Lazy i32-half sync for shadow (non-canonical) pairs: bit p set = pair p's
    // f64 cell is fresh AND its two i32 slots have NOT been synced since the
    // last f64 write. While a pair is fresh, i32 reads extract from the cell,
    // so the slots only need materializing when (a) an i32 write lands on one
    // half (the *other* half's slot is synced first), or (b) control flow
    // leaves for a block where the pair is not must-fresh (synced in
    // emit_term). This removes the per-f64-write trunc/lshr/store×2
    // write-through that showed as vpmovqd/vpsrlq/vinserti in the profile.
    stale: std::cell::Cell<super::regtype::RegSet>,
    // Native SSA pair availability at block entry, used by
    // emit_term to decide which stale pairs must be synced on an out-edge.
    fresh_in: std::collections::BTreeMap<usize, super::regtype::RegSet>,
    // VGPR pairs used consistently as f64 values are stored only in their
    // <W×f64> cell; a 32-bit access extracts or replaces the requested half.
    // `f64c` contains the low registers of those pairs.
    f64c: super::regtype::RegSet,
    nonempty_exec: std::cell::Cell<bool>,
    valid_mask: LLVMValueRef, // immutable allocated lanes, independent of EXEC
    scc: CellId,       // scalar i1
    // scalar types
    i1: LLVMTypeRef,
    i32t: LLVMTypeRef,
    i64t: LLVMTypeRef,
    f32t: LLVMTypeRef,
    f64t: LLVMTypeRef,
    iw: LLVMTypeRef,  // integer of W bits (mask packing)
    ptr: LLVMTypeRef,
    // vector types
    vi1: LLVMTypeRef,   // <W×i1>
    vi32: LLVMTypeRef,  // <W×i32>
    vi64: LLVMTypeRef,  // <W×i64>
    vf32: LLVMTypeRef,  // <W×f32>
    vf64: LLVMTypeRef,  // <W×f64>
    // Reusable scratch for redirecting inactive atomic operations.
    bvh_scratch: LLVMValueRef,
    // Packet passed to the ray-trace helper.  Keeping all W lanes in one call
    // avoids spilling and restoring the whole JIT register state once per lane.
    bvh_packet: LLVMValueRef,
    bvh_packet_ty: LLVMTypeRef,
    // Dedicated lane-spill buffer (`[COOP_SPILL_SLOTS x i32]` alloca) plus a slot
    // index per (spill VGPR, constant lane). Models the uniform writelane/readlane
    // spill idiom: because the spilled value is wavefront-uniform (a scalar SGPR),
    // "lane K of vD" is the same across the W packed lanes, so a single scalar slot
    // keyed by (vD, K) suffices — no cross-lane vector traffic. Mirrors the scalar
    // [`super::emit`] path's `spill_base`/`spill`.
    spill_base: LLVMValueRef,
    spill: std::cell::RefCell<BTreeMap<(u32, u32), usize>>,
    // Preserve inactive lanes on vector writes unless liveness proves that the
    // destination is not observed after reconvergence.
    predicate: std::cell::Cell<bool>,
    structured_loop_masks: Option<StructuredLoopMasks>,
    current_pc: std::cell::Cell<usize>,
}

/// Lane-mask storage used inside one conservatively selected leaf loop. EXEC,
/// VCC, and saved masks stay as `<W x i1>` values within the loop and are
/// converted back to their packed SGPR representation on every loop exit.
///
/// The leaf-loop restriction is deliberate. A parent-loop experiment required
/// scalar and vector mask copies plus a runtime tag and was slower. Applying the
/// same scheme to EXEC was incorrect when a saved mask remained live across the
/// boundary. Any extension must prove mask ownership at every exit and account
/// for representation-conversion overhead.
struct StructuredLoopMasks {
    header: usize,
    body: std::collections::BTreeSet<usize>,
    masks: BTreeMap<u32, CellId>,
    init_bb: LLVMBasicBlockRef,
    exit_bbs: BTreeMap<(usize, usize), LLVMBasicBlockRef>,
    active: std::cell::Cell<bool>,
}

impl Cg {
    unsafe fn n(&self) -> *const std::ffi::c_char {
        b"\0".as_ptr() as *const std::ffi::c_char
    }

    // ---- intrinsics / externals -----------------------------------------
    unsafe fn get_func(&self, name: &str, ret: LLVMTypeRef, params: &[LLVMTypeRef]) -> (LLVMValueRef, LLVMTypeRef) {
        let cname = cstr(name);
        let mut f = llvm::core::LLVMGetNamedFunction(self.module, cname.as_ptr());
        let fty = llvm::core::LLVMFunctionType(ret, params.as_ptr() as *mut _, params.len() as u32, 0);
        if f.is_null() {
            f = llvm::core::LLVMAddFunction(self.module, cname.as_ptr(), fty);
        }
        (f, fty)
    }
    unsafe fn call(&self, name: &str, ret: LLVMTypeRef, params: &[LLVMTypeRef], args: &[LLVMValueRef]) -> LLVMValueRef {
        let (f, fty) = self.get_func(name, ret, params);
        llvm::core::LLVMBuildCall2(self.b, fty, f, args.as_ptr() as *mut _, args.len() as u32, self.n())
    }

    /// Emit a call to a masked gather/scatter intrinsic. Their alignment is an
    /// `align` attribute on the pointer argument rather than an operand, so
    /// `args` holds only the value operands and `ptr_pos` says which is the
    /// pointer. `overloads` selects the intrinsic's overloaded types.
    unsafe fn masked_call(
        &self,
        prefix: &str,
        overloads: &[LLVMTypeRef],
        args: &[LLVMValueRef],
        ptr_pos: u32,
        align: u64,
    ) -> LLVMValueRef {
        let id = llvm::core::LLVMLookupIntrinsicID(prefix.as_ptr() as *const _, prefix.len());
        let mut overloads = overloads.to_vec();
        let f = llvm::core::LLVMGetIntrinsicDeclaration(
            self.module,
            id,
            overloads.as_mut_ptr(),
            overloads.len(),
        );
        let fty = llvm::core::LLVMGlobalGetValueType(f);
        let mut args = args.to_vec();
        let call = llvm::core::LLVMBuildCall2(
            self.b,
            fty,
            f,
            args.as_mut_ptr(),
            args.len() as u32,
            self.n(),
        );
        let name = b"align";
        let kind =
            llvm::core::LLVMGetEnumAttributeKindForName(name.as_ptr() as *const _, name.len());
        let attr = llvm::core::LLVMCreateEnumAttribute(self.ctx, kind, align);
        llvm::core::LLVMAddCallSiteAttribute(call, ptr_pos + 1, attr);
        call
    }

    // ---- constants -------------------------------------------------------
    unsafe fn ci32(&self, v: u32) -> LLVMValueRef { llvm::core::LLVMConstInt(self.i32t, v as u64, 0) }
    unsafe fn ci64(&self, v: u64) -> LLVMValueRef { llvm::core::LLVMConstInt(self.i64t, v, 0) }
    unsafe fn cf64(&self, v: f64) -> LLVMValueRef { llvm::core::LLVMConstReal(self.f64t, v) }

    /// Broadcast a scalar to <W×ty>.
    unsafe fn splat(&self, v: LLVMValueRef, vty: LLVMTypeRef) -> LLVMValueRef {
        let poison = llvm::core::LLVMGetPoison(vty);
        let ins = llvm::core::LLVMBuildInsertElement(self.b, poison, v, self.ci32(0), self.n());
        let mask = llvm::core::LLVMConstNull(llvm::core::LLVMVectorType(self.i32t, llvm::core::LLVMGetVectorSize(vty)));
        llvm::core::LLVMBuildShuffleVector(self.b, ins, poison, mask, self.n())
    }
    unsafe fn vci32(&self, v: u32) -> LLVMValueRef { self.splat(self.ci32(v), self.vi32) }
    unsafe fn vcf64(&self, v: f64) -> LLVMValueRef { self.splat(self.cf64(v), self.vf64) }

    // ---- mask <-> vector boundary ---------------------------------------
    /// Low W bits of a scalar mask register -> <W×i1> (per-lane active).
    unsafe fn mask_to_vec(&self, m32: LLVMValueRef) -> LLVMValueRef {
        let mw = llvm::core::LLVMBuildTrunc(self.b, m32, self.iw, self.n());
        llvm::core::LLVMBuildBitCast(self.b, mw, self.vi1, self.n())
    }
    /// <W×i1> -> scalar i32 with bit i = lane i (high bits 0).
    unsafe fn vec_to_mask(&self, v: LLVMValueRef) -> LLVMValueRef {
        let mw = llvm::core::LLVMBuildBitCast(self.b, v, self.iw, self.n());
        llvm::core::LLVMBuildZExt(self.b, mw, self.i32t, self.n())
    }
    unsafe fn structured_mask(&self, reg: u32) -> Option<LLVMValueRef> {
        let loop_masks = self.structured_loop_masks.as_ref()?;
        if !loop_masks.active.get() { return None; }
        let cell = *loop_masks.masks.get(&reg)?;
        Some(self.state.borrow_mut().read(self.b, cell))
    }
    unsafe fn store_structured_mask(&self, reg: u32, value: LLVMValueRef) -> bool {
        let Some(loop_masks) = self.structured_loop_masks.as_ref() else { return false; };
        if !loop_masks.active.get() { return false; }
        let Some(&cell) = loop_masks.masks.get(&reg) else { return false; };
        self.state.borrow_mut().write(self.b, cell, value);
        true
    }
    fn has_structured_mask(&self, reg: u32) -> bool {
        self.structured_loop_masks.as_ref().map_or(false, |loop_masks| {
            loop_masks.active.get() && loop_masks.masks.contains_key(&reg)
        })
    }


    unsafe fn sync_structured_masks_to_sgpr(&self) {
        let Some(loop_masks) = self.structured_loop_masks.as_ref() else { return; };
        for (&reg, &cell) in &loop_masks.masks {
            let value = self.state.borrow_mut().read(self.b, cell);
            self.st_sgpr32_raw(reg, self.vec_to_mask(value));
        }
    }
    unsafe fn structured_mask_target(
        &self,
        from: usize,
        to: usize,
        bbs: &BTreeMap<usize, LLVMBasicBlockRef>,
    ) -> LLVMBasicBlockRef {
        if let Some(loop_masks) = self.structured_loop_masks.as_ref() {
            if to == loop_masks.header && !loop_masks.body.contains(&from) {
                return loop_masks.init_bb;
            }
            if loop_masks.body.contains(&from) && !loop_masks.body.contains(&to) {
                return loop_masks.exit_bbs[&(from, to)];
            }
        }
        bbs[&to]
    }
    unsafe fn exec_vec(&self) -> LLVMValueRef {
        self.structured_mask(EXEC).unwrap_or_else(||self.mask_to_vec(self.ld_sgpr32_raw(EXEC)))
    }

    // ---- scalar register access (SGPR/SCC) -------------------------------
    unsafe fn ld_sgpr32(&self, i: u32) -> LLVMValueRef {
        self.ld_sgpr32_raw(i)
    }
    unsafe fn st_sgpr32(&self, i: u32, v: LLVMValueRef) {
        assert!(!matches!(i,126|106),"mask state requires a typed I1 definition");
        self.st_sgpr32_raw(i, v);
    }
    unsafe fn ld_sgpr32_raw(&self, i: u32) -> LLVMValueRef {
        if i == 124 { return self.ci32(0); }
        self.state.borrow_mut().read(self.b,self.sgpr[i as usize])
    }
    unsafe fn st_sgpr32_raw(&self, i: u32, v: LLVMValueRef) {
        if i == 124 { return; }
        self.state.borrow_mut().write(self.b, self.sgpr[i as usize], v);
    }
    unsafe fn ld_sgpr64(&self, i: u32) -> LLVMValueRef {
        let lo = self.zext64s(self.ld_sgpr32(i));
        let hi = self.zext64s(self.ld_sgpr32(i + 1));
        let hi = llvm::core::LLVMBuildShl(self.b, hi, self.ci64(32), self.n());
        llvm::core::LLVMBuildOr(self.b, hi, lo, self.n())
    }
    unsafe fn st_sgpr64(&self, i: u32, v: LLVMValueRef) {
        let lo = llvm::core::LLVMBuildTrunc(self.b, v, self.i32t, self.n());
        let hi = llvm::core::LLVMBuildLShr(self.b, v, self.ci64(32), self.n());
        let hi = llvm::core::LLVMBuildTrunc(self.b, hi, self.i32t, self.n());
        self.st_sgpr32(i, lo);
        self.st_sgpr32(i + 1, hi);
    }
    unsafe fn zext64s(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildZExt(self.b, v, self.i64t, self.n())
    }
    unsafe fn ld_scc(&self) -> LLVMValueRef { self.state.borrow_mut().read(self.b, self.scc) }
    unsafe fn st_scc(&self, v: LLVMValueRef) { self.state.borrow_mut().write(self.b, self.scc, v); }
    unsafe fn st_scc_nz(&self, v32: LLVMValueRef) {
        let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, v32, self.ci32(0), self.n());
        self.st_scc(c);
    }

    // ---- vector register access (VGPR) -----------------------------------
    // The <W×i32> slot is always the authoritative backing; the <W×f64> cell is a
    // synced shadow for f64 readers (avoids i32→f64 reconstruction). f64 writes
    // update both; `f64_fresh` bit (pair-low) = cell currently holds the pair.
    fn fresh(&self, p: u32) -> bool { super::regtype::bget(&self.f64_fresh.get(), p) }
    fn set_fresh(&self, p: u32, on: bool) {
        let mut s = self.f64_fresh.get();
        let r = (p & 255) as usize;
        if on { s[r / 128] |= 1u128 << (r % 128); } else { s[r / 128] &= !(1u128 << (r % 128)); }
        self.f64_fresh.set(s);
    }
    /// If reg `i` is a half of a *fresh* shadow pair, that pair's low reg.
    fn fresh_low(&self, i: u32) -> Option<u32> {
        if self.fresh(i) { Some(i) }
        else if i > 0 && self.fresh(i - 1) { Some(i - 1) }
        else { None }
    }
    fn is_stale(&self, p: u32) -> bool { super::regtype::bget(&self.stale.get(), p) }
    fn set_stale(&self, p: u32, on: bool) {
        let mut s = self.stale.get();
        let r = (p & 255) as usize;
        if on { s[r / 128] |= 1u128 << (r % 128); } else { s[r / 128] &= !(1u128 << (r % 128)); }
        self.stale.set(s);
    }
    /// Extract half `i` of pair `p` from the f64 cell as <W×i32>.
    unsafe fn cell_half(&self, p: u32, i: u32) -> LLVMValueRef {
        let bits = self.cell_bits(p);
        let half = if i == p { bits } else {
            llvm::core::LLVMBuildLShr(self.b, bits, self.splat(self.ci64(32), self.vi64), self.n())
        };
        llvm::core::LLVMBuildTrunc(self.b, half, self.vi32, self.n())
    }
    /// Sync half `i` of fresh pair `p` from the cell into its i32 slot.
    unsafe fn sync_half(&self, p: u32, i: u32) {
        if i as usize >= self.vgpr.len() {
            return;
        }
        let v = self.cell_half(p, i);
        self.state.borrow_mut().write(self.b, self.vgpr[i as usize], v);
    }
    /// At a block's out-edges: materialize the i32 slots of every lazily-stale
    /// pair that is not must-fresh in ALL successors (their emitters would read
    /// the slots directly).
    unsafe fn sync_stale_for(&self, succs: &[usize]) {
        let st = self.stale.get();
        if st == [0; 2] { return; }
        let mut keep = [!0u128; 2];
        for &s in succs {
            let f = self.fresh_in.get(&s).copied().unwrap_or([0; 2]);
            keep[0] &= f[0];
            keep[1] &= f[1];
        }
        let mut left = st;
        for w in 0..2 {
            let mut bits = st[w] & !keep[w];
            left[w] &= !bits;
            while bits != 0 {
                let r = w as u32 * 128 + bits.trailing_zeros();
                bits &= bits - 1;
                self.sync_half(r, r);
                self.sync_half(r, r + 1);
            }
        }
        self.stale.set(left);
    }
    /// If reg `i` belongs to an f64-canonical pair, its low reg (cell = sole
    /// storage). The pairs are disjoint so at most one holds.
    fn f64c_low(&self, i: u32) -> Option<u32> {
        if super::regtype::bget(&self.f64c, i) { Some(i) }
        else if i > 0 && super::regtype::bget(&self.f64c, i - 1) { Some(i - 1) }
        else { None }
    }
    unsafe fn ld_pair_f64(&self, p: u32) -> LLVMValueRef {
        self.state.borrow_mut().read(self.b, self.vgpr_f64[p as usize])
    }
    // i64-bits view of a canonical pair's cell (raw 64-bit shifts — NOT v_lshr,
    // which masks the count to 31).
    unsafe fn cell_bits(&self, p: u32) -> LLVMValueRef {
        llvm::core::LLVMBuildBitCast(self.b, self.ld_pair_f64(p), self.vi64, self.n())
    }
    unsafe fn ld_vgpr32(&self, i: u32) -> LLVMValueRef {
        if let Some(p) = self.f64c_low(i) {
            return self.cell_half(p, i);
        }
        // Fresh shadow pair: the cell is current (and the slot may be lazily
        // stale) — extract the half from the cell.
        if let Some(p) = self.fresh_low(i) {
            return self.cell_half(p, i);
        }
        self.state.borrow_mut().read(self.b, self.vgpr[i as usize])
    }
    /// Per-lane predicate: inactive lanes keep their old value (read from the
    /// canonical cell when applicable, else the i32 slot).
    unsafe fn pred_vgpr32(&self, i: u32, v: LLVMValueRef) -> LLVMValueRef {
        if self.predicate.get() {
            let old = self.ld_vgpr32(i);
            llvm::core::LLVMBuildSelect(self.b, self.exec_vec(), v, old, self.n())
        } else {
            v
        }
    }
    unsafe fn st_vgpr32(&self, i: u32, v: LLVMValueRef) {
        let v = self.pred_vgpr32(i, v);
        // f64-canonical: insert the (predicated) half into the cell (sole storage).
        if let Some(p) = self.f64c_low(i) {
            let old = self.cell_bits(p);
            let z = llvm::core::LLVMBuildZExt(self.b, v, self.vi64, self.n());
            let nb = if i == p {
                self.v_or(self.v_and(old, self.splat(self.ci64(0xFFFF_FFFF_0000_0000), self.vi64)), z)
            } else {
                let zhi = llvm::core::LLVMBuildShl(self.b, z, self.splat(self.ci64(32), self.vi64), self.n());
                self.v_or(self.v_and(old, self.splat(self.ci64(0x0000_0000_FFFF_FFFF), self.vi64)), zhi)
            };
            let d = llvm::core::LLVMBuildBitCast(self.b, nb, self.vf64, self.n());
            self.state.borrow_mut().write(self.b, self.vgpr_f64[p as usize], d);
            return;
        }
        // Lazily-synced fresh pairs containing reg i lose freshness below, so
        // their *other* half's slot (still governed by the cell) must be
        // materialized first. (The predicated old value of reg i itself was
        // already read from the cell by pred_vgpr32 above.)
        for p in [i.wrapping_sub(1), i] {
            if p != u32::MAX && self.fresh(p) && self.is_stale(p) {
                let other = if p == i { i + 1 } else { p };
                self.sync_half(p, other);
                self.set_stale(p, false);
            }
        }
        // A 32-bit write invalidates the (shadow) f64 cell of pairs i and i-1.
        self.set_fresh(i, false);
        if i > 0 { self.set_fresh(i - 1, false); }
        self.state.borrow_mut().write(self.b, self.vgpr[i as usize], v);
    }
    unsafe fn set_f64_fresh(&self, fresh: super::regtype::RegSet) { self.f64_fresh.set(fresh); }
    unsafe fn ld_vgpr64(&self, i: u32) -> LLVMValueRef {
        let lo = self.zext64v(self.ld_vgpr32(i));
        let hi = self.zext64v(self.ld_vgpr32(i + 1));
        let hi = llvm::core::LLVMBuildShl(self.b, hi, self.splat(self.ci64(32), self.vi64), self.n());
        llvm::core::LLVMBuildOr(self.b, hi, lo, self.n())
    }
    unsafe fn st_vgpr64(&self, i: u32, v: LLVMValueRef) {
        let lo = llvm::core::LLVMBuildTrunc(self.b, v, self.vi32, self.n());
        let hi = llvm::core::LLVMBuildLShr(self.b, v, self.splat(self.ci64(32), self.vi64), self.n());
        let hi = llvm::core::LLVMBuildTrunc(self.b, hi, self.vi32, self.n());
        self.st_vgpr32(i, lo);
        self.st_vgpr32(i + 1, hi);
    }
    unsafe fn ld_vgpr_f64(&self, i: u32) -> LLVMValueRef {
        // f64-canonical pair: the cell is the sole storage (one <W×f64> phi).
        if self.f64c_low(i) == Some(i) { return self.ld_pair_f64(i); }
        // Else the freshness-shadow path: cell if fresh, else reconstruct + memo.
        if self.fresh(i) { return self.ld_pair_f64(i); }
        let u = self.ld_vgpr64(i);
        let d = llvm::core::LLVMBuildBitCast(self.b, u, self.vf64, self.n());
        self.state.borrow_mut().write(self.b, self.vgpr_f64[i as usize], d);
        self.set_fresh(i, true);
        self.set_stale(i, false); // memoized from the slots — they are current
        d
    }
    unsafe fn st_vgpr_f64(&self, i: u32, v: LLVMValueRef) {
        // f64-canonical: store the (predicated) double directly to its sole cell —
        // no i32 backing, no shadow bookkeeping.
        if self.f64c_low(i) == Some(i) {
            let vp = if self.predicate.get() {
                llvm::core::LLVMBuildSelect(self.b, self.exec_vec(), v, self.ld_pair_f64(i), self.n())
            } else { v };
            self.state.borrow_mut().write(self.b, self.vgpr_f64[i as usize], vp);
            return;
        }
        // Non-canonical: predicated value written to both the shadow cell and the
        // i32 backing halves (kept in sync) so integer readers see it.
        let vp = if self.predicate.get() {
            let old = self.ld_vgpr_f64(i);
            llvm::core::LLVMBuildSelect(self.b, self.exec_vec(), v, old, self.n())
        } else {
            v
        };
        // Overlapping fresh pairs (i-1,i) / (i+1,i+2) lose freshness below; if
        // lazily stale, their non-overwritten half's slot is synced first.
        // (Their overwritten half's slot is governed by this pair's new cell.)
        if i > 0 && self.fresh(i - 1) && self.is_stale(i - 1) {
            self.sync_half(i - 1, i - 1);
        }
        if self.fresh(i + 1) && self.is_stale(i + 1) {
            self.sync_half(i + 1, i + 2);
        }
        // The i32 slots are synced lazily (see `stale`): store only the cell.
        self.state.borrow_mut().write(self.b, self.vgpr_f64[i as usize], vp);
        if i > 0 { self.set_fresh(i - 1, false); }
        self.set_fresh(i + 1, false);
        self.set_fresh(i, true);
        if i > 0 { self.set_stale(i - 1, false); }
        self.set_stale(i + 1, false);
        self.set_stale(i, true);
    }
    unsafe fn zext64v(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildZExt(self.b, v, self.vi64, self.n())
    }

    // ---- vector source operands (VALU): SGPR/const broadcast, VGPR per-lane
    unsafe fn vsrc_u32(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.vci32(*v),
            SourceOperand::IntegerConstant(v) => self.vci32(*v as u32),
            SourceOperand::FloatConstant(v) => self.vci32((*v as f32).to_bits()),
            SourceOperand::ScalarRegister(r) => self.splat(self.ld_sgpr32(*r as u32), self.vi32),
            SourceOperand::VectorRegister(r) => self.ld_vgpr32(*r as u32),
            SourceOperand::PrivateBase => self.splat(llvm::core::LLVMBuildTrunc(self.b, self.scratch_base_scalar, self.i32t, self.n()),self.vi32),
        }
    }
    unsafe fn vsrc_u64(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.splat(self.ci64(*v as u64), self.vi64),
            SourceOperand::IntegerConstant(v) => self.splat(self.ci64(*v), self.vi64),
            SourceOperand::FloatConstant(v) => self.splat(self.ci64(v.to_bits()), self.vi64),
            SourceOperand::ScalarRegister(r) => self.splat(self.ld_sgpr64(*r as u32), self.vi64),
            SourceOperand::VectorRegister(r) => self.ld_vgpr64(*r as u32),
            SourceOperand::PrivateBase => self.splat(self.scratch_base_scalar,self.vi64),
        }
    }
    unsafe fn vsrc_f64(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.vcf64(f64::from_bits((*v as u64) << 32)),
            SourceOperand::IntegerConstant(v) => self.vcf64(f64::from_bits((*v as u64) << 32)),
            SourceOperand::FloatConstant(v) => self.vcf64(*v),
            SourceOperand::ScalarRegister(r) => {
                let u = self.splat(self.ld_sgpr64(*r as u32), self.vi64);
                llvm::core::LLVMBuildBitCast(self.b, u, self.vf64, self.n())
            }
            SourceOperand::VectorRegister(r) => self.ld_vgpr_f64(*r as u32),
            SourceOperand::PrivateBase => panic!("f64 from private base"),
        }
    }

    // ---- scalar source operands (SALU) -----------------------------------
    unsafe fn ssrc_u32(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.ci32(*v),
            SourceOperand::IntegerConstant(v) => self.ci32(*v as u32),
            SourceOperand::FloatConstant(v) => self.ci32((*v as f32).to_bits()),
            SourceOperand::ScalarRegister(r) => self.ld_sgpr32(*r as u32),
            SourceOperand::VectorRegister(r) => panic!("scalar op reads VGPR {}", r),
            SourceOperand::PrivateBase => llvm::core::LLVMBuildTrunc(self.b, self.scratch_base_scalar, self.i32t, self.n()),
        }
    }
    unsafe fn ssrc_u64(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.ci64(*v as u64),
            SourceOperand::IntegerConstant(v) => self.ci64(*v),
            SourceOperand::FloatConstant(v) => self.ci64(v.to_bits()),
            SourceOperand::ScalarRegister(r) => self.ld_sgpr64(*r as u32),
            SourceOperand::PrivateBase => self.scratch_base_scalar,
            SourceOperand::VectorRegister(r) => panic!("scalar op reads VGPR {}", r),
        }
    }

    // ---- vector helpers --------------------------------------------------
    unsafe fn v_and(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildAnd(self.b, a, b, self.n()) }
    unsafe fn v_or(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildOr(self.b, a, b, self.n()) }

    unsafe fn v_add(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildAdd(self.b, a, b, self.n()) }


    /// Mangled vector-f64 intrinsic name, e.g. `llvm.sqrt.v8f64`.
    fn vfn(&self, name: &str) -> String { format!("llvm.{}.v{}f64", name, self.w) }
    unsafe fn vsqrt(&self, a: LLVMValueRef) -> LLVMValueRef {
        self.call(&self.vfn("sqrt"), self.vf64, &[self.vf64], &[a])
    }






    // ---- f32 vector helpers ----------------------------------------------
    unsafe fn vcf32(&self, v: f32) -> LLVMValueRef {
        self.splat(llvm::core::LLVMConstReal(self.f32t, v as f64), self.vf32)
    }
    unsafe fn vf32_bits(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildBitCast(self.b, v, self.vi32, self.n())
    }









    unsafe fn vsrc_f32(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::FloatConstant(v) => self.vcf32(*v as f32),
            _ => llvm::core::LLVMBuildBitCast(self.b, self.vsrc_u32(op), self.vf32, self.n()),
        }
    }
    // Per-lane mask bit (i1 vector) of a lane-mask register's low W bits.

    // ---- compares -> lane mask ------------------------------------------
    /// Write a <W×i1> compare into a lane-mask register's low W bits. Inactive
    /// lanes contribute 0 (AND with EXEC), matching the masked backend; for
    /// V_CMPX (dest == EXEC) this yields EXEC = cmp & old_EXEC.
    unsafe fn st_cmp(&self, dest: u32, cmp_vec: LLVMValueRef) {
        if self.structured_loop_masks.as_ref().map_or(false, |loop_masks| loop_masks.active.get() && loop_masks.masks.contains_key(&dest)) {
            let masked = if dest == EXEC || self.predicate.get() {
                llvm::core::LLVMBuildAnd(self.b, cmp_vec, self.exec_vec(), self.n())
            } else {
                cmp_vec
            };
            self.store_structured_mask(dest, masked);
            return;
        }
        let z = self.vec_to_mask(cmp_vec);
        // A V_CMPX narrows EXEC (EXEC = cmp & old_EXEC): this is intrinsic to the
        // instruction and must hold even when VGPR-write predication is elided —
        // otherwise a lane a prior branch masked off gets reactivated, and its
        // stale (garbage) address feeds the next predicated memory op. So mask
        // unconditionally when writing EXEC; VCC follows the elision predicate.
        let masked = if dest == EXEC || self.predicate.get() {
            self.v_and(z, self.ld_sgpr32(EXEC))
        } else {
            z
        };
        self.st_sgpr32(dest, masked);
    }
    /// Store an i1-vector mask into a generic SGPR (e.g. a VOP3SD carry-out).
    /// Inactive lanes write 0 (`& EXEC`), matching the GPU's per-lane mask
    /// semantics — REQUIRED for soundness once VGPR writes are elided, since an
    /// elided (garbage) source would otherwise set a stale mask bit for an
    /// inactive lane that a later reconverged lane reads.
    unsafe fn st_mask(&self, reg: u32, v: LLVMValueRef) {
        if self.has_structured_mask(reg) {
            let masked = llvm::core::LLVMBuildAnd(self.b, v, self.exec_vec(), self.n());
            self.store_structured_mask(reg, masked);
            return;
        }
        let z = self.vec_to_mask(v);
        let masked = self.v_and(z, self.ld_sgpr32(EXEC));
        self.st_sgpr32(reg, masked);
    }

    unsafe fn ptr_at_vec(&self, addr: LLVMValueRef, off: u64) -> LLVMValueRef {
        // addr: <W×i64>; returns <W×ptr> (inttoptr).
        let a = self.v_add(addr, self.splat(self.ci64(off), self.vi64));
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        llvm::core::LLVMBuildIntToPtr(self.b, a, vptr, self.n())
    }
}

// =====================================================================
//  Public entry
// =====================================================================

pub(super) fn compile_program(plan: &PacketPlan<'_>, num_vgprs: usize) -> VecKernel {
    let code = unsafe { compile_inner(plan, num_vgprs) };
    VecKernel { code, num_vgprs, min_private_bytes: plan.function.min_private_bytes(), width: plan.width }
}

pub(super) fn compile_cooperative(plan: &PacketPlan<'_>, num_vgprs: usize) -> CoopVecKernel {
    let code = unsafe { compile_inner(plan, num_vgprs) };
    let yields = plan.function.value_yields();
    if yields.values().any(|value| value.op == super::ir::typed::effect::EffectOp::Wave(super::ir::typed::effect::WaveOp::Wmma)) {
        super::wmma::warm(plan.width as usize);
    }
    CoopVecKernel { code, num_vgprs, min_private_bytes: plan.function.min_private_bytes(), width: plan.width, yields }
}

/// `boundary` selects the ABI: `Some` compiles a resumable cooperative packet
/// (fiber), `None` a whole-program kernel.
unsafe fn compile_inner(
    plan: &PacketPlan<'_>,
    num_vgprs: usize,
) -> super::jit::NativeCode {
    let boundary = plan.boundary;
    let w = plan.width;
    let coop = boundary.is_some();

    let native = super::jit::Module::new("vec_kernel");
    let ctx = native.ctx;
    let module = native.module;
    let b = native.builder;

    let i1 = llvm::core::LLVMInt1TypeInContext(ctx);
    let i32t = llvm::core::LLVMInt32TypeInContext(ctx);
    let i64t = llvm::core::LLVMInt64TypeInContext(ctx);
    let f32t = llvm::core::LLVMFloatTypeInContext(ctx);
    let f64t = llvm::core::LLVMDoubleTypeInContext(ctx);
    let iw = llvm::core::LLVMIntTypeInContext(ctx, w);
    let ptr = llvm::core::LLVMPointerTypeInContext(ctx, 0);
    let void = llvm::core::LLVMVoidTypeInContext(ctx);
    let vi1 = llvm::core::LLVMVectorType(i1, w);
    let vi32 = llvm::core::LLVMVectorType(i32t, w);
    let vi64 = llvm::core::LLVMVectorType(i64t, w);
    let vf32 = llvm::core::LLVMVectorType(f32t, w);
    let vf64 = llvm::core::LLVMVectorType(f64t, w);

    // Normal: `void kernel(sgprs, vgprs, scratch_base, scratch_stride)`.
    // Cooperative packet:
    // `i64 kernel(sgprs[129], vgprs, wave_scratch_base, scratch_stride,
    //             spill, resume_pc, packet_lane_base)`.
    let func = if coop {
        // Fiber ABI: the cooperative arguments plus the FiberCtx (see
        // `super::fiber::FiberCtx`), shared LDS, and immutable lane validity.
        let mut params = [ptr, ptr, i64t, i64t, ptr, i64t, i64t, ptr, i32t];
        let fty = llvm::core::LLVMFunctionType(i64t, params.as_mut_ptr(), 9, 0);
        llvm::core::LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, fty)
    } else {
        let mut params = [ptr, ptr, i64t, i64t];
        let fty = llvm::core::LLVMFunctionType(void, params.as_mut_ptr(), 4, 0);
        llvm::core::LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, fty)
    };

    let sgprs_p = llvm::core::LLVMGetParam(func, 0);
    let scratch_base = llvm::core::LLVMGetParam(func, 2);
    let scratch_stride = llvm::core::LLVMGetParam(func, 3);
    let packet_lane_base = if coop {
        llvm::core::LLVMGetParam(func, 6)
    } else {
        llvm::core::LLVMConstInt(i64t, 0, 0)
    };

    let entry = llvm::core::LLVMAppendBasicBlockInContext(ctx, func, b"entry\0".as_ptr() as *const _);
    llvm::core::LLVMPositionBuilderAtEnd(b, entry);

    // lane index const <0,1,..,W-1> : <W×i64>
    let mut lane_consts: Vec<LLVMValueRef> = (0..w).map(|k| llvm::core::LLVMConstInt(i64t, k as u64, 0)).collect();
    let lane_idx_i64 = llvm::core::LLVMConstVector(lane_consts.as_mut_ptr(), w);

    // per-lane scratch base = broadcast(base) + lane*stride
    let mut state = State::default();
    let mut sgpr = Vec::with_capacity(128);
    for _ in 0..128 { sgpr.push(state.add(i32t)); }
    let mut vgpr = Vec::with_capacity(num_vgprs);
    for _ in 0..num_vgprs { vgpr.push(state.add(vi32)); }
    let mut vgpr_f64 = Vec::with_capacity(num_vgprs + 1);
    for _ in 0..num_vgprs + 1 { vgpr_f64.push(state.add(vf64)); }
    let scc = state.add(i1);
    let bvh_scratch = llvm::core::LLVMBuildArrayAlloca(b, i32t, llvm::core::LLVMConstInt(i32t, 10, 0), b"\0".as_ptr() as *const _);
    let packet_i64 = llvm::core::LLVMArrayType2(i64t, 16);
    let packet_f32 = llvm::core::LLVMArrayType2(f32t, 16);
    let packet_i32 = llvm::core::LLVMArrayType2(i32t, 16);
    let mut packet_fields = [packet_i64; 15];
    packet_fields[1..11].fill(packet_f32);
    packet_fields[11..15].fill(packet_i32);
    let bvh_packet_ty = llvm::core::LLVMStructTypeInContext(
        ctx,
        packet_fields.as_mut_ptr(),
        packet_fields.len() as u32,
        0,
    );
    let bvh_packet = llvm::core::LLVMBuildAlloca(b, bvh_packet_ty, b"\0".as_ptr() as *const _);
    llvm::core::LLVMSetAlignment(bvh_packet, 64);
    let spill_base = if coop {
        llvm::core::LLVMGetParam(func, 4)
    } else {
        llvm::core::LLVMBuildArrayAlloca(
            b,
            i32t,
            llvm::core::LLVMConstInt(i32t, super::emit::COOP_SPILL_SLOTS as u64, 0),
            b"\0".as_ptr() as *const _,
        )
    };

    let yield_cells = plan.function.value_yields().values().map(|p| p.cells()).max().unwrap_or(0);
    let yield_frame = if yield_cells == 0 { llvm::core::LLVMConstNull(ptr) } else {
        let frame = llvm::core::LLVMBuildAlloca(b, llvm::core::LLVMArrayType2(i32t, yield_cells as u64 * w as u64), cstr("yield.values").as_ptr());
        llvm::core::LLVMSetAlignment(frame, 64);
        frame
    };
    let mut cg = Cg {
        yield_frame,
        state: std::cell::RefCell::new(state),
        ctx, module, func, b, w, coop, num_vgprs,
        scratch_vec: scratch_base, // placeholder, set below
        store_sink: std::cell::Cell::new(std::ptr::null_mut()),
        scratch_base_scalar: if coop {
            let n=b"\0".as_ptr().cast();
            let aperture=llvm::core::LLVMBuildAnd(b,scratch_base,llvm::core::LLVMConstInt(i64t,0xffff_ffff_0000_0000,0),n);
            let sized=llvm::core::LLVMBuildICmp(b,llvm::LLVMIntPredicate::LLVMIntNE,scratch_stride,llvm::core::LLVMConstInt(i64t,0,0),n);
            llvm::core::LLVMBuildSelect(b,sized,aperture,scratch_base,n)
        } else {scratch_base},
        lds_base: if coop {llvm::core::LLVMGetParam(func,5)}else{llvm::core::LLVMConstInt(i64t,0,0)},
        scratch_stride,
        valid_mask: if coop {llvm::core::LLVMGetParam(func,8)}else{llvm::core::LLVMConstInt(i32t,(1u64<<w)-1,0)},
        sgpr, vgpr, vgpr_f64, scc,
        f64_fresh: std::cell::Cell::new([0; 2]),
        stale: std::cell::Cell::new([0; 2]),
        fresh_in: plan.blocks.iter().map(|(&pc, block)| (pc, block.fresh)).collect(),
        f64c: plan.f64_pairs,
        nonempty_exec: std::cell::Cell::new(false),
        i1, i32t, i64t, f32t, f64t, iw, ptr,
        vi1, vi32, vi64, vf32, vf64,
        bvh_scratch, bvh_packet, bvh_packet_ty,
        spill_base,
        spill: std::cell::RefCell::new(BTreeMap::new()),
        predicate: std::cell::Cell::new(false),
        structured_loop_masks: None,
        current_pc: std::cell::Cell::new(usize::MAX),
    };

    if let Some(region) = plan.mask_region.as_ref() {
        let mask_regs = &region.registers;
        let masks = mask_regs.iter().copied().map(|reg| {
            (reg, cg.state.borrow_mut().add(vi1))
        }).collect();
        let init_bb = llvm::core::LLVMAppendBasicBlockInContext(ctx, func, cstr("structured_mask_init").as_ptr());
        cg.structured_loop_masks = Some(StructuredLoopMasks {
            header: region.header,
            body: region.body.clone(),
            masks,
            init_bb,
            exit_bbs: BTreeMap::new(),
            active: std::cell::Cell::new(false),
        });
    }

    // scratch_vec = splat(base) + (packet_lane_base + lane_idx) * stride.
    // The normal whole-kernel vector path uses packet_lane_base=0.
    let base_v = cg.splat(scratch_base, vi64);
    let stride_v = cg.splat(scratch_stride, vi64);
    let lane_base_v = cg.splat(packet_lane_base, vi64);
    let scratch_lane = cg.v_add(lane_base_v, lane_idx_i64);
    let off = llvm::core::LLVMBuildMul(b, scratch_lane, stride_v, cg.n());
    cg.scratch_vec = cg.v_add(base_v, off);
    cg.store_sink.set(llvm::core::LLVMBuildAlloca(b, i64t, cstr("store_sink").as_ptr()));

    // Load the packet's incoming register state into SSA definitions. Values
    // live across a fiber call are preserved by the native calling convention.
    cg.emit_load(None);
    if coop {
        // SCC is persisted in the packet-local extension slot sgprs[128].
        let gep = llvm::core::LLVMBuildGEP2(
            b,
            i32t,
            sgprs_p,
            [cg.ci32(128)].as_mut_ptr(),
            1,
            cg.n(),
        );
        let persisted_scc = llvm::core::LLVMBuildLoad2(b, i32t, gep, cg.n());
        cg.st_scc_nz(persisted_scc);
    } else {
        // Whole-program entry starts with all packed lanes active.
        let init_exec = if w >= 32 { 0xFFFF_FFFFu32 } else { (1u32 << w) - 1 };
        cg.st_sgpr32_raw(EXEC, cg.ci32(init_exec));
        cg.st_scc(llvm::core::LLVMConstInt(i1, 0, 0));
    }

    cg.predicate.set(true);

    let mut bbs: BTreeMap<usize, LLVMBasicBlockRef> = BTreeMap::new();
    for &pc in plan.function.blocks.keys() {
        let name = cstr(&format!("b{:x}", pc));
        bbs.insert(pc, llvm::core::LLVMAppendBasicBlockInContext(ctx, func, name.as_ptr()));
    }
    if let Some(loop_masks) = cg.structured_loop_masks.as_mut() {
        for &(from, to) in &plan.mask_region.as_ref().unwrap().exits {
            let name = cstr(&format!("structured_mask_exit_{from:x}_{to:x}"));
            loop_masks.exit_bbs.insert((from, to), llvm::core::LLVMAppendBasicBlockInContext(ctx, func, name.as_ptr()));
        }
    }
    // A fiber enters the kernel once and suspends in place at a boundary, so
    // there is no resume dispatch here.
    let mut ssa = super::typed_codegen::Values::new(&plan.function, b, Some(plan.width));
    ssa.set_scratch_environment(cg.scratch_base_scalar, cg.scratch_stride);
    ssa.set_bvh_storage(super::dialect::rdna4::bvh::Storage {scratch: cg.bvh_scratch, packet: cg.bvh_packet, packet_ty: cg.bvh_packet_ty});
    llvm::core::LLVMBuildBr(b, bbs[&plan.function.ir.func().entry.0]);
    for (&pc, block_plan) in &plan.blocks {
        llvm::core::LLVMPositionBuilderAtEnd(b, bbs[&pc]);
        cg.current_pc.set(pc);
        if let Some(loop_masks) = cg.structured_loop_masks.as_ref() {
            loop_masks.active.set(loop_masks.body.contains(&pc));
        }
        // Seed f64-cell freshness from the cross-block analysis. Sound because the
        // cell is now populated on *every* f64 def the analysis counts — f64-op
        // producers (predicated `st_vgpr_f64`) and global f64 loads (gathered as
        // <W×f64>) — so an analysis-fresh pair's cell holds the correct per-lane
        // value on entry. Removes the cross-block i32→f64 reconstruction.
        cg.set_f64_fresh(block_plan.fresh);
        // Conservatively assume every fresh-on-entry shadow pair's i32 slots are
        // stale (a predecessor may have skipped the sync); canonical pairs have
        // no live slots and are excluded so edge syncs don't write dead stores.
        cg.stale.set(block_plan.stale);
        let specialize = block_plan.specialize;
        // Both variants consume the same precomputed instruction choices.
        let variants: Vec<(bool, LLVMBasicBlockRef)> = if specialize {
            let fast = llvm::core::LLVMAppendBasicBlockInContext(ctx, func, cstr(&format!("b{:x}.allactive", pc)).as_ptr());
            let slow = llvm::core::LLVMAppendBasicBlockInContext(ctx, func, cstr(&format!("b{:x}.masked", pc)).as_ptr());
            let mask = llvm::core::LLVMBuildBitCast(b, cg.exec_vec(), cg.iw, cg.n());
            let all = llvm::core::LLVMConstInt(cg.iw, u64::MAX, 0);
            let full = llvm::core::LLVMBuildICmp(b, llvm::LLVMIntPredicate::LLVMIntEQ, mask, all, cg.n());
            llvm::core::LLVMBuildCondBr(b, full, fast, slow);
            vec![(false,fast),(true,slow)]
        } else {
            vec![(true,bbs[&pc])]
        };
        // Both clones start from the block's entry facts.
        let entry_facts = (
            cg.f64_fresh.get(),
            cg.stale.get(),
            cg.spill.borrow().clone(),
        );

        for (variant_pred, variant_bb) in variants {
            if specialize {
                // Rewind to the block's entry facts: the second clone emits the
                // same instructions from the same state, only unpredicated.
                llvm::core::LLVMPositionBuilderAtEnd(b, variant_bb);
                cg.f64_fresh.set(entry_facts.0);
                cg.stale.set(entry_facts.1);
                *cg.spill.borrow_mut() = entry_facts.2.clone();
            }
            ssa.begin_block(&plan.function, pc);
            let mut sqrt_inputs: std::collections::HashMap<usize, LLVMValueRef> =
                std::collections::HashMap::new();
            for (idx, instruction) in block_plan.instructions.iter().enumerate() {
                cg.nonempty_exec.set(instruction.nonempty_exec || (!variant_pred && instruction.entry_exec_unchanged));
                if let Some(SqrtCollapse::Capture { site, src }) = &instruction.sqrt {
                    sqrt_inputs.insert(*site, cg.typed_input(src, false));
                }
                if matches!(instruction.action, InstructionAction::ClusterMember) {
                    continue;
                }
                cg.predicate.set((variant_pred || !instruction.entry_exec_unchanged) && !instruction.elide_predicate);
                if let Some(SqrtCollapse::Rescale { site, vdst }) = &instruction.sqrt {
                    let root = cg.vsqrt(sqrt_inputs[site]);
                    cg.st_vgpr_f64(*vdst as u32, root);
                } else if let InstructionAction::Cluster(c) = &instruction.action {
                    let members: Vec<_> = (idx..idx+c.len)
                        .map(|index| &plan.function.blocks[&pc].memory[&index]).collect();
                    ssa.prepare_memory(&plan.function,pc,idx,|p,scalar|cg.memory_parameter(p,scalar));
                    let preds: Vec<bool> = block_plan.instructions[idx..idx + c.len]
                        .iter()
                        .map(|member| (variant_pred || !member.entry_exec_unchanged) && !member.elide_predicate)
                        .collect();
                    cg.emit_memory_cluster(&members, ssa.value(members[0].base), c.lo, c.span, c.tile, &preds);
                } else {
                    match &plan.function.blocks[&pc].instructions[idx] {
                        Some(_) => {
                            ssa.emit(&plan.function, pc, idx, !cg.predicate.get(), |reg|!cg.has_structured_mask(reg),
                                |reg|cg.ld_sgpr32_raw(reg),
                                |input, scalar| cg.typed_input(input, scalar), |output, value, word| {
                                // Typed results already include their EXEC select.
                                let predicate=cg.predicate.replace(false);
                                if let (super::lift::Output::MaskBit(reg),Some(word))=(output,word) {
                                    if !cg.store_structured_mask(reg,value) {cg.st_sgpr32_raw(reg,word);}
                                } else {cg.typed_output(output, value);}
                                cg.predicate.set(predicate);
                            });
                        }
                        None if plan.function.blocks[&pc].wave.contains_key(&idx) => {
                    let (action,wave)=&plan.function.blocks[&pc].wave[&idx];
                    cg.emit_local_wave(action,|reg,raw| {
                        let (ty,value)=ssa.effect_result(wave.results[0].0,wave.definitions[0].1,raw);
                        let output=if ty==super::ir::typed::Ty::I1 {super::lift::Output::MaskBit(reg)}
                            else {super::lift::Output::Scalar(reg,ty)};
                        cg.typed_output(output,value);
                    });
                },
                None => {
                    ssa.prepare_memory(&plan.function,pc,idx,|p,scalar|cg.memory_parameter(p,scalar));
                            cg.emit_memory(&plan.function.blocks[&pc].memory[&idx],instruction.memory.unwrap(),&ssa,|k|ssa.memory_data(&plan.function,pc,idx,k));
                        }
                    }
                }
            }
            ssa.condition(&plan.function, pc, |reg|cg.structured_mask(reg).map_or_else(||cg.ld_sgpr32_raw(reg),|mask|cg.vec_to_mask(mask)), |input| cg.typed_input(input, true));
            cg.emit_term(&plan.function.terminator(pc), &bbs, &mut ssa, &plan.function);
        }
    }

    if let Some(loop_masks) = cg.structured_loop_masks.as_ref() {
        loop_masks.active.set(false);
        llvm::core::LLVMPositionBuilderAtEnd(b, loop_masks.init_bb);
        for (&reg, &cell) in &loop_masks.masks {
            let scalar = cg.ld_sgpr32(reg);
            cg.state.borrow_mut().write(b, cell, cg.mask_to_vec(scalar));
        }
        llvm::core::LLVMBuildBr(b, bbs[&loop_masks.header]);
        for (&(_, to), &exit_bb) in &loop_masks.exit_bbs {
            llvm::core::LLVMPositionBuilderAtEnd(b, exit_bb);
            cg.sync_structured_masks_to_sgpr();
            llvm::core::LLVMBuildBr(b, bbs[&to]);
        }
    }

    cg.state.borrow_mut().finish(func);
    native.finish(super::jit::Mode::Packet)
}

impl Cg {
    /// Load registers from the caller's packet buffers into packed SSA
    /// definitions. `set` selects which (`None` = the whole register file, which
    /// is what kernel entry needs).
    unsafe fn emit_load(&self, set: Option<&RegSet>) {
        let sgprs_p = llvm::core::LLVMGetParam(self.func, 0);
        let vgprs_p = llvm::core::LLVMGetParam(self.func, 1);
        let num_vgprs = self.num_vgprs as u32;
        let want_sgpr = |reg: u32| set.map_or(true, |s| s.has_sgpr(reg));
        // An f64-canonical pair's cell is seeded from both halves, so selecting
        // either half pulls in its partner.
        let want_vgpr = |reg: u32| {
            let selected = |r: u32| r < num_vgprs && set.map_or(true, |s| s.has_vgpr(r));
            selected(reg)
                || (super::regtype::bget(&self.f64c, reg) && selected(reg + 1))
                || (reg > 0 && super::regtype::bget(&self.f64c, reg - 1) && selected(reg - 1))
        };
        for reg in 0..128u32 {
            if !want_sgpr(reg) {
                continue;
            }
            let gep = llvm::core::LLVMBuildGEP2(self.b, self.i32t, sgprs_p, [self.ci32(reg)].as_mut_ptr(), 1, self.n());
            let value = llvm::core::LLVMBuildLoad2(self.b, self.i32t, gep, self.n());
            let value=if reg==EXEC&&self.coop {self.v_and(value,self.valid_mask)} else {value};
            self.st_sgpr32_raw(reg, value);
        }
        for reg in 0..num_vgprs {
            if !want_vgpr(reg) {
                continue;
            }
            // lanes of register `reg` are contiguous at vgprs_p[reg*W ..][..W]
            let gep = llvm::core::LLVMBuildGEP2(self.b, self.i32t, vgprs_p, [self.ci32(reg * self.w)].as_mut_ptr(), 1, self.n());
            let load = llvm::core::LLVMBuildLoad2(self.b, self.vi32, gep, self.n());
            llvm::core::LLVMSetAlignment(load, 4);
            self.state.borrow_mut().write(self.b, self.vgpr[reg as usize], load);
        }
        if set.map_or(true, |s| s.scc) {
            let p=llvm::core::LLVMBuildGEP2(self.b,self.i32t,sgprs_p,[self.ci32(128)].as_mut_ptr(),1,self.n());
            let value=llvm::core::LLVMBuildLoad2(self.b,self.i32t,p,self.n());
            let bit=llvm::core::LLVMBuildICmp(self.b,llvm::LLVMIntPredicate::LLVMIntNE,value,self.ci32(0),self.n());
            self.state.borrow_mut().write(self.b, self.scc, bit);
        }
        // Seed each selected f64-canonical pair's cell from its two i32 halves
        // (the cell is its sole storage; the i32 slots stay unused).
        for pair in 0..num_vgprs {
            if super::regtype::bget(&self.f64c, pair) && (pair + 1) < num_vgprs && want_vgpr(pair) {
                let lo = self.zext64v(self.state.borrow_mut().read(self.b, self.vgpr[pair as usize]));
                let hi = self.zext64v(self.state.borrow_mut().read(self.b, self.vgpr[pair as usize + 1]));
                let hi = llvm::core::LLVMBuildShl(self.b, hi, self.splat(self.ci64(32), self.vi64), self.n());
                let value = llvm::core::LLVMBuildBitCast(self.b, self.v_or(hi, lo), self.vf64, self.n());
                self.state.borrow_mut().write(self.b, self.vgpr_f64[pair as usize], value);
            }
        }
    }

    /// Store registers back to the caller's packet buffers. `set` selects which
    /// (`None` = the whole register file plus SCC). Values are read through the
    /// canonical accessors, so lazily materialized f64 cells are stored
    /// correctly without first forcing their shadow slots.
    unsafe fn emit_store(&self, set: Option<&RegSet>) {
        let sgprs_p = llvm::core::LLVMGetParam(self.func, 0);
        let vgprs_p = llvm::core::LLVMGetParam(self.func, 1);
        for reg in 0..128u32 {
            // NULL reads are constant and writes are discarded; its buffer
            // slot is not part of the state transferred to the scheduler.
            if reg == 124 { continue; }
            if set.map_or(false, |s| !s.has_sgpr(reg)) {
                continue;
            }
            let gep = llvm::core::LLVMBuildGEP2(self.b, self.i32t, sgprs_p, [self.ci32(reg)].as_mut_ptr(), 1, self.n());
            llvm::core::LLVMBuildStore(self.b, self.ld_sgpr32(reg), gep);
        }
        if set.map_or(true, |s| s.scc) {
            // SCC lives in the packet-local extension slot sgprs[128].
            let gep = llvm::core::LLVMBuildGEP2(self.b, self.i32t, sgprs_p, [self.ci32(128)].as_mut_ptr(), 1, self.n());
            let scc = llvm::core::LLVMBuildZExt(self.b, self.ld_scc(), self.i32t, self.n());
            llvm::core::LLVMBuildStore(self.b, scc, gep);
        }
        for reg in 0..self.num_vgprs as u32 {
            if set.map_or(false, |s| !s.has_vgpr(reg)) {
                continue;
            }
            let gep = llvm::core::LLVMBuildGEP2(self.b, self.i32t, vgprs_p, [self.ci32(reg * self.w)].as_mut_ptr(), 1, self.n());
            let store = llvm::core::LLVMBuildStore(self.b, self.ld_vgpr32(reg), gep);
            llvm::core::LLVMSetAlignment(store, 4);
        }
    }

    unsafe fn emit_value_yield(&self, resume: usize, plan: &super::lift::wave::Plan, values: &mut super::typed_codegen::Values) {
        use llvm::core::*;
        use super::ir::typed::{Ty, effect::{EffectOp, WaveOp}};
        use super::lift::{Output, wave::Destination};
        let predicated = matches!(plan.layout.op, EffectOp::Wave(WaveOp::Bpermute | WaveOp::BpermuteFi));
        let old_predicate = self.predicate.replace(predicated);
        values.yield_values(plan, self.yield_frame, LLVMGetParam(self.func, 7), resume, |destination, ty, result, bits| {
            match destination {
                Destination::Vgpr(reg) => self.typed_output(Output::Vgpr(reg, ty), result),
                Destination::Sgpr(reg) => {
                    if ty==Ty::I1 {self.typed_output(Output::MaskBit(reg),result);} else {
                        let result = LLVMBuildExtractElement(self.b, bits, self.ci32(0), self.n());
                        self.typed_output(Output::Scalar(reg, Ty::I32), result);
                    }
                }
                Destination::Scc => {
                    let result = LLVMBuildExtractElement(self.b, result, self.ci32(0), self.n());
                    self.typed_output(Output::Scc, result);
                }
            }
        });
        self.predicate.set(old_predicate);
    }

    unsafe fn emit_term(&self, term: &super::lift::function::Control, bbs: &BTreeMap<usize, LLVMBasicBlockRef>, values: &mut super::typed_codegen::Values, function: &super::lift::function::Function) {
        match term {
            super::lift::function::Control::Return => {
                if self.coop {
                    // The direct cooperative API observes returned registers;
                    // normal grid dispatch observes only kernel memory effects.
                    if function.observable_return { self.emit_store(None); }
                    llvm::core::LLVMBuildRet(self.b, self.ci64(super::emit::COOP_DONE));
                } else {
                    llvm::core::LLVMBuildRetVoid(self.b);
                }
            }
            super::lift::function::Control::Jump(t) => {
                self.sync_stale_for(&[*t]);
                llvm::core::LLVMBuildBr(self.b, self.structured_mask_target(self.current_pc.get(), *t, bbs));
            }
            super::lift::function::Control::Branch { cond, taken, fallthrough } => {
                let c = values.value(*cond);
                self.sync_stale_for(&[*taken, *fallthrough]);
                llvm::core::LLVMBuildCondBr(
                    self.b,
                    c,
                    self.structured_mask_target(self.current_pc.get(), *taken, bbs),
                    self.structured_mask_target(self.current_pc.get(), *fallthrough, bbs),
                );
            }
            super::lift::function::Control::Yield { resume } => {
                let pc = self.current_pc.get();
                let plan = function.blocks[&pc].yield_values.as_ref().expect("yield lacks SSA value plan");
                values.prepare_yield(function, pc, llvm::core::LLVMGetParam(self.func,6), |input| self.typed_input(input, false));
                self.sync_stale_for(&[*resume]);
                self.emit_value_yield(*resume, plan, values);
                llvm::core::LLVMBuildBr(self.b, bbs[resume]);
            }
        }
    }


}

// =====================================================================
//  Instruction emission
// =====================================================================

impl Cg {







    // ---- lane-local spill (uniform writelane/readlane idiom) -------------
    // See the `spill_base`/`spill` field docs. Keyed by (spill VGPR, constant
    // lane); returns a GEP into the per-invocation scalar spill buffer.
    unsafe fn spill_slot_ptr(&self, vgpr: u32, lane: u32) -> LLVMValueRef {
        let idx = {
            let mut m = self.spill.borrow_mut();
            let next = m.len();
            *m.entry((vgpr, lane)).or_insert(next)
        };
        assert!(
            idx < super::emit::COOP_SPILL_SLOTS,
            "too many writelane/readlane spill slots ({} >= {})",
            idx, super::emit::COOP_SPILL_SLOTS
        );
        llvm::core::LLVMBuildGEP2(self.b, self.i32t, self.spill_base, [self.ci32(idx as u32)].as_mut_ptr(), 1, self.n())
    }

    /// Extract <W×i1> from a source operand that names a lane-mask SGPR (low W bits).
    unsafe fn src_mask_vec(&self, op: &SourceOperand) -> LLVMValueRef {
        let m = match op {
            SourceOperand::ScalarRegister(r) => self.ld_sgpr32(*r as u32),
            SourceOperand::LiteralConstant(v) => self.ci32(*v),
            SourceOperand::IntegerConstant(v) => self.ci32(*v as u32),
            _ => panic!("mask vec from {:?}", op),
        };
        self.mask_to_vec(m)
    }



    // ---- SALU (scalar, uniform) -----------------------------------------






    // ---- VGLOBAL (per-lane gather/scatter) ------------------------------

    /// Concatenate equal-length <n×i32> vectors into one <sum×i32> via a balanced
    /// shuffle tree (parts.len() is a power of two on the affine-frame path, so
    /// pairs always match in length).
    unsafe fn vconcat_i32(&self, parts: &[LLVMValueRef]) -> LLVMValueRef {
        let mut cur = parts.to_vec();
        while cur.len() > 1 {
            let mut next = Vec::with_capacity((cur.len() + 1) / 2);
            let mut i = 0;
            while i + 1 < cur.len() {
                let a = cur[i];
                let b = cur[i + 1];
                let na = llvm::core::LLVMGetVectorSize(llvm::core::LLVMTypeOf(a));
                let nb = llvm::core::LLVMGetVectorSize(llvm::core::LLVMTypeOf(b));
                let mut idx: Vec<LLVMValueRef> = (0..na + nb).map(|k| self.ci32(k)).collect();
                let mask = llvm::core::LLVMConstVector(idx.as_mut_ptr(), idx.len() as u32);
                next.push(llvm::core::LLVMBuildShuffleVector(self.b, a, b, mask, self.n()));
                i += 2;
            }
            if i < cur.len() { next.push(cur[i]); }
            cur = next;
        }
        cur[0]
    }
    /// Emit a divergent-pointer load cluster (see `load_cluster::analyze`) as W per-lane
    /// contiguous <span×f64> loads + a shuffle transpose, replacing the member
    /// masked gathers. Inactive lanes may hold garbage pointers: they are
    /// substituted with an active lane's pointer (umax over active lanes) —
    /// sound because the block only runs with EXEC≠0 (the same invariant the
    /// uniform-address broadcast path relies on), any active lane's record is
    /// dereferenceable over the whole span (contiguity checked in detection),
    /// and inactive lanes' loaded values are merged away by the predicated
    /// stores (or dead, when the elide analysis dropped the predicate).
    unsafe fn emit_memory_cluster(&self, members: &[&super::lift::memory::Plan], addr: LLVMValueRef, lo: i64, span: u32, tile: u32, preds: &[bool]) {
        let exec = self.exec_vec();
        let zero = llvm::core::LLVMConstNull(self.vi64);
        let masked = llvm::core::LLVMBuildSelect(self.b, exec, addr, zero, self.n());
        let p_any = self.call(
            &format!("llvm.vector.reduce.umax.v{}i64", self.w),
            self.i64t, &[self.vi64], &[masked],
        );
        let safe = llvm::core::LLVMBuildSelect(self.b, exec, addr, self.splat(p_any, self.vi64), self.n());
        let rowty = llvm::core::LLVMVectorType(self.f64t, span);
        let any = self.call(&format!("llvm.vector.reduce.or.v{}i1",self.w),self.i1,&[self.vi1],&[exec]);
        let rowmask = self.splat(any,llvm::core::LLVMVectorType(self.i1,span));
        let rows: Vec<LLVMValueRef> = (0..self.w)
            .map(|l| {
                let a = llvm::core::LLVMBuildExtractElement(self.b, safe, self.ci32(l), self.n());
                let a = llvm::core::LLVMBuildAdd(self.b, a, self.ci64(lo as u64), self.n());
                let p = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, self.n());
                let ld = self.masked_call("llvm.masked.load.", &[rowty,self.ptr], &[p,rowmask,llvm::core::LLVMGetPoison(rowty)],0,4);
                ld
            })
            .collect();
        let cols = self.transpose_rows(&rows, span, tile);
        for (m, g) in members.iter().enumerate() {
            self.predicate.set(preds[m]);
            let pairs = g.memory.words / 2;
            let offset = match g.memory.address { super::lift::memory::Address::Global {offset,..} => offset,_=>unreachable!() };
            let f0 = ((offset - lo) / 8) as u32;
            for j in 0..pairs {
                self.st_vgpr_f64(g.memory.dest + 2 * j, cols[(f0 + j) as usize]);
            }
        }
    }
    /// Transpose W lane-major rows (<span×f64> each) into span column vectors
    /// (<W×f64> each). Each power-of-two tile uses log2(tile) butterfly stages;
    /// wider packets concatenate independently transposed tiles.
    unsafe fn transpose_rows(
        &self,
        rows: &[LLVMValueRef],
        span: u32,
        tile: u32,
    ) -> Vec<LLVMValueRef> {
        debug_assert!(tile.is_power_of_two() && tile <= 8 && self.w % tile == 0);
        let rowty = llvm::core::LLVMTypeOf(rows[0]);
        let shuf = |x: LLVMValueRef, y: LLVMValueRef, m: &[u32]| -> LLVMValueRef {
            let mut mv: Vec<LLVMValueRef> = m.iter().map(|&i| self.ci32(i)).collect();
            let mask = llvm::core::LLVMConstVector(mv.as_mut_ptr(), mv.len() as u32);
            llvm::core::LLVMBuildShuffleVector(self.b, x, y, mask, self.n())
        };
        let nblk = (self.w / tile) as usize;
        let mut cols: Vec<Vec<LLVMValueRef>> = vec![Vec::with_capacity(nblk); span as usize];
        for blk in 0..nblk {
            let begin = blk * tile as usize;
            let r = &rows[begin..begin + tile as usize];
            let mut base = 0u32;
            while base + tile <= span {
                let idx: Vec<u32> = (base..base + tile).collect();
                let poison = llvm::core::LLVMGetPoison(rowty);
                let mut cur: Vec<LLVMValueRef> =
                    r.iter().map(|&row| shuf(row, poison, &idx)).collect();
                let mut step = 1u32;
                while step < tile {
                    let (lo_mask, hi_mask) = transpose_pair_masks(tile, step);
                    let mut next = vec![std::ptr::null_mut(); tile as usize];
                    for i in 0..tile {
                        if i & step != 0 {
                            continue;
                        }
                        let j = i | step;
                        next[i as usize] = shuf(
                            cur[i as usize],
                            cur[j as usize],
                            &lo_mask,
                        );
                        next[j as usize] = shuf(
                            cur[i as usize],
                            cur[j as usize],
                            &hi_mask,
                        );
                    }
                    cur = next;
                    step <<= 1;
                }
                for cix in 0..tile {
                    cols[(base + cix) as usize].push(cur[cix as usize]);
                }
                base += tile;
            }
            for f in base..span {
                let mut parts: Vec<LLVMValueRef> = if tile == 1 {
                    vec![shuf(r[0], llvm::core::LLVMGetPoison(rowty), &[f])]
                } else {
                    r.chunks(2)
                        .map(|pair| shuf(pair[0], pair[1], &[f, span + f]))
                        .collect()
                };
                let mut n = 2u32;
                while parts.len() > 1 {
                    let cat: Vec<u32> = (0..2 * n).collect();
                    parts = parts
                        .chunks(2)
                        .map(|pair| shuf(pair[0], pair[1], &cat))
                        .collect();
                    n *= 2;
                }
                cols[f as usize].push(parts[0]);
            }
        }
        cols.into_iter()
            .map(|mut parts| {
                let mut n = tile;
                while parts.len() > 1 {
                    let cat: Vec<u32> = (0..2 * n).collect();
                    parts = parts.chunks(2).map(|p| shuf(p[0], p[1], &cat)).collect();
                    n *= 2;
                }
                parts[0]
            })
            .collect()
    }
    unsafe fn masked_gather_f64(&self, ptrs: LLVMValueRef, mask: LLVMValueRef) -> LLVMValueRef {
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        let passthru = llvm::core::LLVMConstNull(self.vf64);
        self.masked_call("llvm.masked.gather.", &[self.vf64, vptr], &[ptrs, mask, passthru], 0, 4)
    }
    // Uniform-address load: lane-0's (shared) address loaded scalar + broadcast.
    // Used only where the address VGPR is proven uniform across the packed lanes
    // (so all lanes' address is identical), replacing an expensive gather.
    unsafe fn bcast_load_i32(&self, ptrs: LLVMValueRef) -> LLVMValueRef {
        let p0 = llvm::core::LLVMBuildExtractElement(self.b, ptrs, self.ci32(0), self.n());
        let any = if self.nonempty_exec.get() {llvm::core::LLVMConstInt(self.i1,1,0)}else{self.call(&format!("llvm.vector.reduce.or.v{}i1", self.w), self.i1, &[self.vi1], &[self.exec_vec()])};
        let p0 = llvm::core::LLVMBuildSelect(self.b,any,p0,self.bvh_scratch,self.n());
        let v = llvm::core::LLVMBuildLoad2(self.b, self.i32t, p0, self.n());
        self.splat(v, self.vi32)
    }
    unsafe fn bcast_load_f64(&self, ptrs: LLVMValueRef) -> LLVMValueRef {
        let p0 = llvm::core::LLVMBuildExtractElement(self.b, ptrs, self.ci32(0), self.n());
        let any = if self.nonempty_exec.get() {llvm::core::LLVMConstInt(self.i1,1,0)}else{self.call(&format!("llvm.vector.reduce.or.v{}i1", self.w), self.i1, &[self.vi1], &[self.exec_vec()])};
        let p0 = llvm::core::LLVMBuildSelect(self.b,any,p0,self.bvh_scratch,self.n());
        let v = llvm::core::LLVMBuildLoad2(self.b, self.f64t, p0, self.n());
        self.splat(v, self.vf64)
    }
    /// Load `<W x i32>` through lane-affine pointers (see `emit_vscratch`).
    unsafe fn affine_load(&self, ptrs: LLVMValueRef, allocated: bool) -> LLVMValueRef {
        let n = self.n();
        let exec=self.exec_vec();
        let mut v = llvm::core::LLVMGetPoison(self.vi32);
        for l in 0..self.w {
            let p = llvm::core::LLVMBuildExtractElement(self.b, ptrs, self.ci32(l), n);
            let active=llvm::core::LLVMBuildExtractElement(self.b,exec,self.ci32(l),n);
            let p=if allocated {p}else{llvm::core::LLVMBuildSelect(self.b,active,p,self.bvh_scratch,n)};
            let ld = llvm::core::LLVMBuildLoad2(self.b, self.i32t, p, n);
            llvm::core::LLVMSetAlignment(ld, 4);
            v = llvm::core::LLVMBuildInsertElement(self.b, v, ld, self.ci32(l), n);
        }
        v
    }

    /// Store `<W x i32>` through lane-affine pointers. A store must not be
    /// observable from an inactive lane, so instead of predicating each one the
    /// inactive lanes are redirected to a sink whose value is never read.
    unsafe fn affine_store(&self, val: LLVMValueRef, ptrs: LLVMValueRef, exec: LLVMValueRef) {
        let n = self.n();
        let sink = llvm::core::LLVMBuildPtrToInt(self.b, self.store_sink.get(), self.i64t, n);
        let addr_i = llvm::core::LLVMBuildPtrToInt(self.b, ptrs, self.vi64, n);
        let safe = llvm::core::LLVMBuildSelect(self.b, exec, addr_i, self.splat(sink, self.vi64), n);
        for l in 0..self.w {
            let a = llvm::core::LLVMBuildExtractElement(self.b, safe, self.ci32(l), n);
            let p = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, n);
            let d = llvm::core::LLVMBuildExtractElement(self.b, val, self.ci32(l), n);
            let st = llvm::core::LLVMBuildStore(self.b, d, p);
            llvm::core::LLVMSetAlignment(st, 4);
        }
    }

    unsafe fn masked_gather(&self, ptrs: LLVMValueRef, mask: LLVMValueRef) -> LLVMValueRef {
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        let passthru = llvm::core::LLVMConstNull(self.vi32);
        self.masked_call("llvm.masked.gather.", &[self.vi32, vptr], &[ptrs, mask, passthru], 0, 4)
    }
    unsafe fn masked_scatter(&self, val: LLVMValueRef, ptrs: LLVMValueRef, mask: LLVMValueRef) {
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        self.masked_call("llvm.masked.scatter.", &[self.vi32, vptr], &[val, ptrs, mask], 1, 4);
    }
    /// Typed masked gather: `<W×elem>` load through per-lane pointers, inactive
    /// lanes read 0. Used for VFLAT sub-word loads.
    unsafe fn masked_gather_ty(&self, ptrs: LLVMValueRef, mask: LLVMValueRef, elem: LLVMTypeRef) -> LLVMValueRef {
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        let velem = llvm::core::LLVMVectorType(elem, self.w);
        let passthru = llvm::core::LLVMConstNull(velem);
        self.masked_call("llvm.masked.gather.", &[velem, vptr], &[ptrs, mask, passthru], 0, 1)
    }
    unsafe fn masked_scatter_ty(&self, val: LLVMValueRef, ptrs: LLVMValueRef, mask: LLVMValueRef, elem: LLVMTypeRef) {
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        let velem = llvm::core::LLVMVectorType(elem, self.w);
        self.masked_call("llvm.masked.scatter.", &[velem, vptr], &[val, ptrs, mask], 1, 1);
    }

    // ---- VFLAT (per-lane flat gather/scatter) — flat addressing matches the
    // global path; each packed lane holds its own byte address.

}

/// Shuffle masks for one butterfly stage of a square power-of-two transpose.
/// `step` selects the row/column index bit exchanged at this stage.
fn transpose_pair_masks(tile: u32, step: u32) -> (Vec<u32>, Vec<u32>) {
    debug_assert!(tile.is_power_of_two());
    debug_assert!(step.is_power_of_two() && step < tile);
    let mask = |high: bool| {
        (0..tile)
            .map(|position| {
                let from_second = position & step != 0;
                let element = (position & !step) | if high { step } else { 0 };
                element + if from_second { tile } else { 0 }
            })
            .collect()
    };
    (mask(false), mask(true))
}

#[cfg(test)]
mod transpose_mask_tests {
    use super::transpose_pair_masks;

    #[test]
    fn power_of_two_butterfly_masks_transpose_square_tiles() {
        for tile in [1usize, 2, 4, 8, 16] {
            let mut rows: Vec<Vec<usize>> = (0..tile)
                .map(|row| (0..tile).map(|col| row * tile + col).collect())
                .collect();
            let mut step = 1usize;
            while step < tile {
                let (lo, hi) = transpose_pair_masks(tile as u32, step as u32);
                let mut next = vec![vec![]; tile];
                for row in 0..tile {
                    if row & step != 0 {
                        continue;
                    }
                    let other = row | step;
                    let joined: Vec<usize> = rows[row]
                        .iter()
                        .chain(&rows[other])
                        .copied()
                        .collect();
                    next[row] = lo.iter().map(|&index| joined[index as usize]).collect();
                    next[other] = hi.iter().map(|&index| joined[index as usize]).collect();
                }
                rows = next;
                step <<= 1;
            }
            for col in 0..tile {
                for lane in 0..tile {
                    assert_eq!(rows[col][lane], lane * tile + col, "tile={tile}");
                }
            }
        }
    }
}

// f64 compare opcode -> (predicate, invert result)


/// Constant lane index of a writelane/readlane operand, if it is a constant.
fn lane_const(op: &SourceOperand) -> Option<u32> {
    match op {
        SourceOperand::IntegerConstant(v) => Some(*v as u32),
        SourceOperand::LiteralConstant(v) => Some(*v),
        _ => None,
    }
}

/// The VGPR number of a vector-register operand.
fn vreg_of(op: &SourceOperand) -> Option<u32> {
    match op {
        SourceOperand::VectorRegister(r) => Some(*r as u32),
        _ => None,
    }
}

impl Cg {
    unsafe fn typed_input(&self, input: &super::lift::Input, scalar: bool) -> LLVMValueRef {
        use super::ir::typed::Ty;
        if matches!(input.source,super::lift::InputSource::ExecPredicate) {return self.exec_vec();}
        if let super::lift::InputSource::MaskBit(reg) = input.source {
            return self.structured_mask(reg).unwrap_or_else(||self.mask_to_vec(self.ld_sgpr32_raw(reg)));
        }
        if matches!(input.source, super::lift::InputSource::Scc) {
            let flag = self.ld_scc();
            return if scalar { flag } else { self.splat(flag, self.vi1) };
        }
        if scalar {
            return match input.ty {
                Ty::I32 => self.ssrc_u32(input.source.operand()),
                Ty::I64 => self.ssrc_u64(input.source.operand()),
                Ty::F32 => llvm::core::LLVMBuildBitCast(self.b, self.ssrc_u32(input.source.operand()), self.f32t, self.n()),
                Ty::F64 => llvm::core::LLVMBuildBitCast(self.b, self.ssrc_u64(input.source.operand()), self.f64t, self.n()),
                Ty::I1 => unreachable!("scalar boolean input requires SCC binding"),
            };
        }
        match input.ty {
            Ty::I1 => self.src_mask_vec(input.source.operand()),
            Ty::I32 => self.vsrc_u32(input.source.operand()),
            Ty::I64 => self.vsrc_u64(input.source.operand()),
            Ty::F32 => self.vsrc_f32(input.source.operand()),
            Ty::F64 => self.vsrc_f64(input.source.operand()),
        }
    }
    unsafe fn typed_output(&self, output: super::lift::Output, result: LLVMValueRef) {
        use super::ir::typed::Ty;
        use super::lift::Output;
        match output {
            Output::Vgpr(reg, Ty::I32) => self.st_vgpr32(reg, result),
            Output::Vgpr(reg, Ty::I64) => self.st_vgpr64(reg, result),
            Output::Vgpr(reg, Ty::F32) => self.st_vgpr32(reg, self.vf32_bits(result)),
            Output::Vgpr(reg, Ty::F64) => self.st_vgpr_f64(reg, result),
            Output::Compare(reg) => self.st_cmp(reg, result),
            Output::Mask(reg) => self.st_mask(reg, result),
            Output::MaskBit(reg) => {
                if !self.store_structured_mask(reg,result) {self.st_sgpr32_raw(reg,self.vec_to_mask(result));}
            },
            Output::Scalar(reg, Ty::I32) => self.st_sgpr32(reg, result),
            Output::Scalar(reg, Ty::I64) => self.st_sgpr64(reg, result),
            Output::Scalar(reg, Ty::F32) => self.st_sgpr32(reg, llvm::core::LLVMBuildBitCast(self.b, result, self.i32t, self.n())),
            Output::Scalar(reg, Ty::F64) => self.st_sgpr64(reg, llvm::core::LLVMBuildBitCast(self.b, result, self.i64t, self.n())),
            Output::Scc => self.st_scc(result),
            Output::Scalar(_, Ty::I1) => unreachable!("boolean SGPR output"),
            Output::Vgpr(_, Ty::I1) => unreachable!("boolean VGPR output"),
        }
    }
}

impl Cg {
    unsafe fn emit_local_wave(&self, action:&super::lift::wave::YieldAction,mut write:impl FnMut(u32,LLVMValueRef)) {
        use super::lift::wave::{Operand,Destination};
        use super::ir::typed::effect::{EffectOp,WaveOp};
        let source=|index|match &action.inputs[index]{Operand::Source(s)=>s,_=>panic!("local wave operand requires register binding")};
        let dst=match action.outputs[0]{Destination::Sgpr(r)|Destination::Vgpr(r)=>r,_=>panic!("barrier requires cooperative dispatch")};
        let op=match action.op{EffectOp::Wave(op)=>op,_=>panic!("barrier requires cooperative dispatch")};
        match op {
            WaveOp::ReadFirstLane => {
                // Value from the lowest active lane, broadcast to an SGPR (uniform).
                // cttz with is_zero_undef=false yields W for EXEC==0 (a block may
                // run predicated with EXEC==0); clamp the index to a valid lane so
                // the uniform destination never receives poison.
                let src = self.vsrc_u32(source(0));
                let exec = self.ld_sgpr32(EXEC);
                let tz = self.call("llvm.cttz.i32", self.i32t, &[self.i32t, self.i1], &[exec, llvm::core::LLVMConstInt(self.i1, 0, 0)]);
                let over = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntUGE, tz, self.ci32(self.w), self.n());
                let idx = llvm::core::LLVMBuildSelect(self.b, over, self.ci32(0), tz, self.n());
                let v = llvm::core::LLVMBuildExtractElement(self.b, src, idx, self.n());
                write(dst,v);
                        }
            WaveOp::WriteLane => {
                let lane = lane_const(source(1))
                    .expect("vec: v_writelane_b32 needs a constant lane (non-uniform cross-lane unsupported)");
                let val = self.ssrc_u32(source(0));
                let slot = self.spill_slot_ptr(dst, lane);
                llvm::core::LLVMBuildStore(self.b, val, slot);
                        }
            WaveOp::ReadLane => {
                let lane = lane_const(source(1))
                    .expect("vec: v_readlane_b32 needs a constant lane");
                let src = vreg_of(source(0))
                    .expect("vec: v_readlane_b32 source must be a VGPR");
                let slot = self.spill_slot_ptr(src, lane);
                let v = llvm::core::LLVMBuildLoad2(self.b, self.i32t, slot, self.n());
                write(dst,v);
                        }
            _=>panic!("wave operation requires cooperative dispatch"),
        }
    }
}
