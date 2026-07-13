//! Width-W SPMD codegen: lower a [`ScalarProgram`] to a native function that
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

use std::collections::BTreeMap;
use std::ffi::CString;

use llvm_sys as llvm;
use llvm::prelude::{LLVMBasicBlockRef, LLVMBuilderRef, LLVMTypeRef, LLVMValueRef};

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand, SMEM, SOP1, SOP2, SOPK, VFLAT, VGLOBAL, VIMAGE, VOP1, VOP2, VOP3, VOP3P, VOP3SD, VOPC, VOPD, VSAMPLE, VSCRATCH};

use super::ir::{Cond, ScalarProgram, Terminator};

const EXEC: u32 = 126;
const VCC: u32 = 106;

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

fn succs_for_emit(block: &super::ir::ScalarBlock) -> Vec<usize> {
    match block.term {
        Terminator::Return => vec![],
        Terminator::Jump(target) => vec![target],
        Terminator::Branch { taken, fallthrough, .. } => vec![taken, fallthrough],
        Terminator::Barrier { resume } => vec![resume],
    }
}

fn normal_f64_pow2_exponent(op: &SourceOperand) -> bool {
    let value = match op {
        SourceOperand::IntegerConstant(value) => *value as u32 as i32,
        SourceOperand::LiteralConstant(value) => *value as i32,
        SourceOperand::FloatConstant(value) => (*value as f32).to_bits() as i32,
        _ => return false,
    };
    (-1022..=1023).contains(&value)
}

fn normal_pow2_cndmask_def(inst: &InstFormat) -> Option<u32> {
    let InstFormat::VOP3(i) = inst else { return None };
    (matches!(i.op, I::V_CNDMASK_B32)
        && i.abs == 0
        && i.neg == 0
        && normal_f64_pow2_exponent(&i.src0)
        && normal_f64_pow2_exponent(&i.src1))
        .then_some(i.vdst as u32)
}

/// Recognize the object-level normalization idiom
/// `scale -> sqrt -> refine -> classify -> rescale`.  Returning instruction
/// indices keeps the profitability decision local to the complete idiom rather
/// than expanding every individually-safe LDEXP in the object.
fn normal_sqrt_ldexp_indices(body: &[InstFormat]) -> Vec<bool> {
    let mut fast = vec![false; body.len()];
    for start in 0..body.len().saturating_sub(5) {
        let Some(first_exp) = normal_pow2_cndmask_def(&body[start]) else { continue };
        let InstFormat::VOP3(first_scale) = &body[start + 1] else { continue };
        if !matches!(first_scale.op, I::V_LDEXP_F64)
            || !matches!(first_scale.src1, SourceOperand::VectorRegister(r) if r as u32 == first_exp)
        {
            continue;
        }
        let InstFormat::VOP1(sqrt) = &body[start + 2] else { continue };
        if !matches!(sqrt.op, I::V_SQRT_F64)
            || !matches!(sqrt.src0, SourceOperand::VectorRegister(r) if r == first_scale.vdst)
        {
            continue;
        }
        let Some(second_exp) = normal_pow2_cndmask_def(&body[start + 3]) else { continue };
        let InstFormat::VOP3(class) = &body[start + 4] else { continue };
        if !matches!(class.op, I::V_CMP_CLASS_F64)
            || !matches!(class.src0, SourceOperand::VectorRegister(r) if r == first_scale.vdst)
        {
            continue;
        }
        let InstFormat::VOP3(second_scale) = &body[start + 5] else { continue };
        if !matches!(second_scale.op, I::V_LDEXP_F64)
            || !matches!(second_scale.src0, SourceOperand::VectorRegister(r) if r == sqrt.vdst)
            || !matches!(second_scale.src1, SourceOperand::VectorRegister(r) if r as u32 == second_exp)
        {
            continue;
        }
        fast[start + 1] = true;
        fast[start + 5] = true;
    }
    fast
}

#[cfg(test)]
pub(super) fn normal_sqrt_ldexp_sites(program: &ScalarProgram) -> Vec<(usize, usize)> {
    program
        .blocks
        .iter()
        .flat_map(|(&pc, block)| {
            normal_sqrt_ldexp_indices(&block.body)
                .into_iter()
                .enumerate()
                .filter_map(move |(index, fast)| fast.then_some((pc, index)))
        })
        .collect()
}

#[cfg(test)]
mod normal_sqrt_tests {
    use super::*;

    fn vop3(
        op: I,
        vdst: u8,
        src0: SourceOperand,
        src1: SourceOperand,
        src2: SourceOperand,
    ) -> InstFormat {
        InstFormat::VOP3(VOP3 {
            vdst,
            abs: 0,
            opsel: 0,
            cm: 0,
            op,
            src0,
            src1,
            src2,
            omod: 0,
            neg: 0,
        })
    }

    fn normalization_body(first_exp: u32) -> Vec<InstFormat> {
        vec![
            vop3(
                I::V_CNDMASK_B32,
                10,
                SourceOperand::IntegerConstant(0),
                SourceOperand::LiteralConstant(first_exp),
                SourceOperand::ScalarRegister(VCC as u8),
            ),
            vop3(
                I::V_LDEXP_F64,
                8,
                SourceOperand::VectorRegister(8),
                SourceOperand::VectorRegister(10),
                SourceOperand::ScalarRegister(0),
            ),
            InstFormat::VOP1(VOP1 {
                src0: SourceOperand::VectorRegister(8),
                op: I::V_SQRT_F64,
                vdst: 10,
            }),
            vop3(
                I::V_CNDMASK_B32,
                12,
                SourceOperand::IntegerConstant(0),
                SourceOperand::LiteralConstant((-128i32) as u32),
                SourceOperand::ScalarRegister(VCC as u8),
            ),
            vop3(
                I::V_CMP_CLASS_F64,
                VCC as u8,
                SourceOperand::VectorRegister(8),
                SourceOperand::LiteralConstant(0x260),
                SourceOperand::ScalarRegister(0),
            ),
            vop3(
                I::V_LDEXP_F64,
                10,
                SourceOperand::VectorRegister(10),
                SourceOperand::VectorRegister(12),
                SourceOperand::ScalarRegister(0),
            ),
        ]
    }

    #[test]
    fn recognizes_normal_power_of_two_sqrt_idiom() {
        assert_eq!(
            normal_sqrt_ldexp_indices(&normalization_body(256)),
            [false, true, false, false, false, true]
        );
    }

    #[test]
    fn rejects_non_normal_power_of_two_exponent() {
        assert!(normal_sqrt_ldexp_indices(&normalization_body(1024))
            .iter()
            .all(|fast| !fast));
    }

    #[test]
    fn rejects_mismatched_exponent_def_use() {
        let mut body = normalization_body(256);
        let InstFormat::VOP3(scale) = &mut body[1] else { unreachable!() };
        scale.src1 = SourceOperand::VectorRegister(11);
        assert!(normal_sqrt_ldexp_indices(&body).iter().all(|fast| !fast));
    }

    #[test]
    fn rejects_non_constant_exponent_choice() {
        let mut body = normalization_body(256);
        let InstFormat::VOP3(select) = &mut body[0] else { unreachable!() };
        select.src0 = SourceOperand::ScalarRegister(4);
        assert!(normal_sqrt_ldexp_indices(&body).iter().all(|fast| !fast));
    }

    #[test]
    fn rejects_modified_or_incomplete_idiom() {
        let mut modified = normalization_body(256);
        let InstFormat::VOP3(select) = &mut modified[0] else { unreachable!() };
        select.abs = 1;
        assert!(normal_sqrt_ldexp_indices(&modified).iter().all(|fast| !fast));

        let mut incomplete = normalization_body(256);
        incomplete.remove(4);
        assert!(normal_sqrt_ldexp_indices(&incomplete).iter().all(|fast| !fast));
    }
}

/// A JIT-compiled width-W kernel. Processes W work-items per `run` call.
pub struct VecKernel {
    addr: u64,
    pub num_vgprs: usize,
    pub width: u32,
}
unsafe impl Send for VecKernel {}
unsafe impl Sync for VecKernel {}

impl VecKernel {
    /// Run W work-items. `sgprs` -> 128 u32 (shared/uniform); `vgprs` ->
    /// `num_vgprs * W` u32 in SoA layout (register r, lanes 0..W at `r*W`);
    /// `scratch_base` = base of W contiguous per-lane private segments of
    /// `scratch_stride` bytes each.
    pub unsafe fn run(&self, sgprs: *mut u32, vgprs: *mut u32, scratch_base: u64, scratch_stride: u64) {
        let f = std::mem::transmute::<u64, extern "C" fn(*mut u32, *mut u32, u64, u64)>(self.addr);
        f(sgprs, vgprs, scratch_base, scratch_stride);
    }
}

/// A resumable width-W packet used by the wave-owned cross-lane scheduler.
/// One call advances W adjacent lanes of one GPU wave to the next lifted
/// cross-lane boundary (or `s_endpgm`) and writes the packet state back.
pub struct CoopVecKernel {
    addr: u64,
    pub num_vgprs: usize,
    pub width: u32,
    pub entry_pc: usize,
}

unsafe impl Send for CoopVecKernel {}
unsafe impl Sync for CoopVecKernel {}

impl CoopVecKernel {
    /// `sgprs` points to 129 packet-local scalar slots (SCC at 128), `vgprs`
    /// uses register-major SoA layout (`num_vgprs * W`), and `spill` persists
    /// the uniform readlane/writelane spill idiom across yields. `lane_base` is
    /// the packet's first lane within its owning 32-lane wave and is used only
    /// to select the corresponding private scratch segments.
    pub unsafe fn run(
        &self,
        sgprs: *mut u32,
        vgprs: *mut u32,
        scratch_base: u64,
        scratch_stride: u64,
        spill: *mut u32,
        resume_pc: u64,
        lane_base: u64,
    ) -> u64 {
        let f = std::mem::transmute::<
            u64,
            extern "C" fn(*mut u32, *mut u32, u64, u64, *mut u32, u64, u64) -> u64,
        >(self.addr);
        f(
            sgprs,
            vgprs,
            scratch_base,
            scratch_stride,
            spill,
            resume_pc,
            lane_base,
        )
    }
}

struct Cg {
    ctx: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    func: LLVMValueRef,
    b: LLVMBuilderRef,
    w: u32,
    coop: bool,
    writeback_vgprs: usize,
    // per-lane scratch base vector <W×i64> (broadcast(base) + lane*stride)
    scratch_vec: LLVMValueRef,
    // uniform scratch base (i64 scalar) — lane 0's segment. Used for scalar
    // SRC_PRIVATE_BASE reads (`s_mov_b64 s[..], src_private_base`): the aperture
    // high word is uniform across lanes, and the kernel adds the per-lane low
    // offset from VGPRs.
    scratch_base_scalar: LLVMValueRef,
    // per-lane scratch segment stride in bytes (i64 scalar). The private aperture
    // for a lane is [scratch_base, scratch_base+stride).
    scratch_stride: LLVMValueRef,
    sgpr: Vec<LLVMValueRef>, // 128 scalar i32 allocas (incl. EXEC/VCC mask regs)
    vgpr: Vec<LLVMValueRef>, // num_vgprs <W×i32> allocas
    // Parallel <W×f64> storage for each VGPR pair whose low register is r. This
    // lets an f64 value remain one vector across instructions instead of being
    // repeatedly rebuilt from two <W×i32> halves. `f64_fresh` bit r means that
    // this f64 storage contains the current value of r:r+1. Predicated writes
    // update it with a per-lane select, so it remains valid for inactive lanes.
    vgpr_f64: Vec<LLVMValueRef>,
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
    // Cross-block must-freshness in-sets (freshness::analyze), used by
    // emit_term to decide which stale pairs must be synced on an out-edge.
    fresh_in: std::collections::BTreeMap<usize, super::regtype::RegSet>,
    // VGPR pairs used consistently as f64 values are stored only in their
    // <W×f64> cell; a 32-bit access extracts or replaces the requested half.
    // `f64c` contains the low registers of those pairs.
    f64c: super::regtype::RegSet,
    // Flow-sensitive divergent-VGPR set at the current program point (seeded per
    // block, transferred per instruction). A global load whose address VGPR is
    // *uniform* here loads one location for all lanes ⇒ scalar broadcast instead
    // of a gather (the IPC-crushing op). Flow-sensitive so a reg reused as a
    // uniform address after being divergent f64 data is seen as uniform.
    div_cur: std::cell::Cell<[u128; 2]>,
    // Affine frame pointers at the current point: VGPR pair → per-work-item stride
    // (bytes). A load with such an address is a contiguous vector load + transpose
    // instead of a gather (see vec_live::frame_transfer).
    frame_cur: std::cell::RefCell<std::collections::HashMap<u32, u32>>,
    scc: LLVMValueRef,       // scalar i1 alloca
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
    // Reusable [10 x i32] entry scratch for per-lane ray-trace helper results.
    bvh_scratch: LLVMValueRef,
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
    // The current V_LDEXP exponent is object-proven to produce a normal f64
    // power of two, so exact multiplication can replace the generic lowering.
    ldexp_normal_pow2: std::cell::Cell<bool>,
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
    masks: BTreeMap<u32, LLVMValueRef>,
    init_bb: LLVMBasicBlockRef,
    exit_bbs: BTreeMap<(usize, usize), LLVMBasicBlockRef>,
    active: std::cell::Cell<bool>,
}

impl Cg {
    unsafe fn n(&self) -> *const i8 {
        b"\0".as_ptr() as *const i8
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

    // ---- constants -------------------------------------------------------
    unsafe fn ci32(&self, v: u32) -> LLVMValueRef { llvm::core::LLVMConstInt(self.i32t, v as u64, 0) }
    unsafe fn ci64(&self, v: u64) -> LLVMValueRef { llvm::core::LLVMConstInt(self.i64t, v, 0) }
    unsafe fn cf64(&self, v: f64) -> LLVMValueRef { llvm::core::LLVMConstReal(self.f64t, v) }

    /// Broadcast a scalar to <W×ty>.
    unsafe fn splat(&self, v: LLVMValueRef, vty: LLVMTypeRef) -> LLVMValueRef {
        let poison = llvm::core::LLVMGetPoison(vty);
        let ins = llvm::core::LLVMBuildInsertElement(self.b, poison, v, self.ci32(0), self.n());
        let mask = llvm::core::LLVMConstNull(llvm::core::LLVMVectorType(self.i32t, self.w));
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
        Some(llvm::core::LLVMBuildLoad2(self.b, self.vi1, cell, self.n()))
    }
    unsafe fn store_structured_mask(&self, reg: u32, value: LLVMValueRef) -> bool {
        let Some(loop_masks) = self.structured_loop_masks.as_ref() else { return false; };
        if !loop_masks.active.get() { return false; }
        let Some(&cell) = loop_masks.masks.get(&reg) else { return false; };
        llvm::core::LLVMBuildStore(self.b, value, cell);
        true
    }
    fn has_structured_mask(&self, reg: u32) -> bool {
        self.structured_loop_masks.as_ref().map_or(false, |loop_masks| {
            loop_masks.active.get() && loop_masks.masks.contains_key(&reg)
        })
    }
    unsafe fn mask_src(&self, op: &SourceOperand) -> LLVMValueRef {
        if let SourceOperand::ScalarRegister(reg) = op {
            if let Some(value) = self.structured_mask(*reg as u32) { return value; }
        }
        self.mask_to_vec(self.ssrc_u32(op))
    }
    unsafe fn mask_any(&self, mask: LLVMValueRef) -> LLVMValueRef {
        let packed = self.vec_to_mask(mask);
        llvm::core::LLVMBuildICmp(
            self.b,
            llvm::LLVMIntPredicate::LLVMIntNE,
            packed,
            self.ci32(0),
            self.n(),
        )
    }
    unsafe fn sync_structured_masks_to_sgpr(&self) {
        let Some(loop_masks) = self.structured_loop_masks.as_ref() else { return; };
        for (&reg, &cell) in &loop_masks.masks {
            let value = llvm::core::LLVMBuildLoad2(self.b, self.vi1, cell, self.n());
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
        self.structured_mask(EXEC).unwrap_or_else(|| self.mask_to_vec(self.ld_sgpr32(EXEC)))
    }

    // ---- scalar register access (SGPR/SCC) -------------------------------
    unsafe fn ld_sgpr32(&self, i: u32) -> LLVMValueRef {
        self.ld_sgpr32_raw(i)
    }
    unsafe fn st_sgpr32(&self, i: u32, v: LLVMValueRef) {
        self.st_sgpr32_raw(i, v);
    }
    unsafe fn ld_sgpr32_raw(&self, i: u32) -> LLVMValueRef {
        llvm::core::LLVMBuildLoad2(self.b, self.i32t, self.sgpr[i as usize], self.n())
    }
    unsafe fn st_sgpr32_raw(&self, i: u32, v: LLVMValueRef) {
        llvm::core::LLVMBuildStore(self.b, v, self.sgpr[i as usize]);
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
    unsafe fn ld_scc(&self) -> LLVMValueRef { llvm::core::LLVMBuildLoad2(self.b, self.i1, self.scc, self.n()) }
    unsafe fn st_scc(&self, v: LLVMValueRef) { llvm::core::LLVMBuildStore(self.b, v, self.scc); }
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
        llvm::core::LLVMBuildStore(self.b, v, self.vgpr[i as usize]);
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
        llvm::core::LLVMBuildLoad2(self.b, self.vf64, self.vgpr_f64[p as usize], self.n())
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
        llvm::core::LLVMBuildLoad2(self.b, self.vi32, self.vgpr[i as usize], self.n())
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
            llvm::core::LLVMBuildStore(self.b, d, self.vgpr_f64[p as usize]);
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
        llvm::core::LLVMBuildStore(self.b, v, self.vgpr[i as usize]);
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
        llvm::core::LLVMBuildStore(self.b, d, self.vgpr_f64[i as usize]);
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
            llvm::core::LLVMBuildStore(self.b, vp, self.vgpr_f64[i as usize]);
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
        llvm::core::LLVMBuildStore(self.b, vp, self.vgpr_f64[i as usize]);
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
            SourceOperand::PrivateBase => llvm::core::LLVMBuildTrunc(self.b, self.scratch_vec, self.vi32, self.n()),
        }
    }
    unsafe fn vsrc_u64(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.splat(self.ci64(*v as u64), self.vi64),
            SourceOperand::IntegerConstant(v) => self.splat(self.ci64(*v), self.vi64),
            SourceOperand::FloatConstant(v) => self.splat(self.ci64(v.to_bits()), self.vi64),
            SourceOperand::ScalarRegister(r) => self.splat(self.ld_sgpr64(*r as u32), self.vi64),
            SourceOperand::VectorRegister(r) => self.ld_vgpr64(*r as u32),
            SourceOperand::PrivateBase => self.scratch_vec,
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
    unsafe fn v_xor(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildXor(self.b, a, b, self.n()) }
    unsafe fn v_add(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildAdd(self.b, a, b, self.n()) }
    unsafe fn v_shl(&self, a: LLVMValueRef, amt: LLVMValueRef) -> LLVMValueRef {
        let amt = self.v_and(amt, self.vci32(31));
        llvm::core::LLVMBuildShl(self.b, a, amt, self.n())
    }
    unsafe fn v_lshr(&self, a: LLVMValueRef, amt: LLVMValueRef) -> LLVMValueRef {
        let amt = self.v_and(amt, self.vci32(31));
        llvm::core::LLVMBuildLShr(self.b, a, amt, self.n())
    }
    /// Mangled vector-f64 intrinsic name, e.g. `llvm.sqrt.v8f64`.
    fn vfn(&self, name: &str) -> String { format!("llvm.{}.v{}f64", name, self.w) }
    unsafe fn vfmuladd(&self, a: LLVMValueRef, b: LLVMValueRef, c: LLVMValueRef) -> LLVMValueRef {
        let vfty = self.vf64;
        self.call(&self.vfn("fmuladd"), vfty, &[vfty, vfty, vfty], &[a, b, c])
    }
    unsafe fn vsqrt(&self, a: LLVMValueRef) -> LLVMValueRef {
        self.call(&self.vfn("sqrt"), self.vf64, &[self.vf64], &[a])
    }
    unsafe fn vfloor(&self, a: LLVMValueRef) -> LLVMValueRef {
        self.call(&self.vfn("floor"), self.vf64, &[self.vf64], &[a])
    }
    unsafe fn vfdiv(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildFDiv(self.b, a, b, self.n()) }
    unsafe fn vfmul(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildFMul(self.b, a, b, self.n()) }
    unsafe fn vfadd(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef { llvm::core::LLVMBuildFAdd(self.b, a, b, self.n()) }

    unsafe fn vabsneg_f64(&self, v: LLVMValueRef, abs: u8, neg: u8, idx: u32) -> LLVMValueRef {
        let mut v = v;
        if (abs >> idx) & 1 != 0 {
            v = self.call(&self.vfn("fabs"), self.vf64, &[self.vf64], &[v]);
        }
        if (neg >> idx) & 1 != 0 {
            v = llvm::core::LLVMBuildFNeg(self.b, v, self.n());
        }
        v
    }
    unsafe fn vclamp_f64(&self, mut v: LLVMValueRef, clamp: u8) -> LLVMValueRef {
        if clamp & 1 != 0 {
            v = self.call(&self.vfn("minnum"), self.vf64, &[self.vf64, self.vf64], &[v, self.vcf64(1.0)]);
            v = self.call(&self.vfn("maxnum"), self.vf64, &[self.vf64, self.vf64], &[v, self.vcf64(0.0)]);
        }
        v
    }

    // ---- f32 vector helpers ----------------------------------------------
    fn vfn32(&self, name: &str) -> String { format!("llvm.{}.v{}f32", name, self.w) }
    unsafe fn vcf32(&self, v: f32) -> LLVMValueRef {
        self.splat(llvm::core::LLVMConstReal(self.f32t, v as f64), self.vf32)
    }
    unsafe fn vf32_bits(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildBitCast(self.b, v, self.vi32, self.n())
    }
    unsafe fn vf32_of(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildBitCast(self.b, v, self.vf32, self.n())
    }
    unsafe fn vfsub(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildFSub(self.b, a, b, self.n())
    }
    unsafe fn vcf32_bits(&self, bits: u32) -> LLVMValueRef {
        self.splat(llvm::core::LLVMConstBitCast(self.ci32(bits), self.f32t), self.vf32)
    }
    /// Lane-mask register presented as <W×i32> 0/1 per lane.
    unsafe fn mask_to_u32(&self, reg: u32) -> LLVMValueRef {
        llvm::core::LLVMBuildZExt(self.b, self.mask_to_vec(self.ld_sgpr32(reg)), self.vi32, self.n())
    }
    unsafe fn vsrc_f32(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::FloatConstant(v) => self.vcf32(*v as f32),
            _ => llvm::core::LLVMBuildBitCast(self.b, self.vsrc_u32(op), self.vf32, self.n()),
        }
    }
    unsafe fn vsrc_f16_f32(&self, op: &SourceOperand, high: bool) -> LLVMValueRef {
        let mut bits = self.vsrc_u32(op);
        if high {
            bits = llvm::core::LLVMBuildLShr(self.b, bits, self.vci32(16), self.n());
        }
        let i16t = llvm::core::LLVMInt16TypeInContext(self.ctx);
        let vi16 = llvm::core::LLVMVectorType(i16t, self.w);
        let vf16 = llvm::core::LLVMVectorType(llvm::core::LLVMHalfTypeInContext(self.ctx), self.w);
        let bits = llvm::core::LLVMBuildTrunc(self.b, bits, vi16, self.n());
        let half = llvm::core::LLVMBuildBitCast(self.b, bits, vf16, self.n());
        llvm::core::LLVMBuildFPExt(self.b, half, self.vf32, self.n())
    }
    unsafe fn vf32_to_f16_bits(&self, value: LLVMValueRef) -> LLVMValueRef {
        let i16t = llvm::core::LLVMInt16TypeInContext(self.ctx);
        let vi16 = llvm::core::LLVMVectorType(i16t, self.w);
        let vf16 = llvm::core::LLVMVectorType(llvm::core::LLVMHalfTypeInContext(self.ctx), self.w);
        let half = llvm::core::LLVMBuildFPTrunc(self.b, value, vf16, self.n());
        let bits = llvm::core::LLVMBuildBitCast(self.b, half, vi16, self.n());
        llvm::core::LLVMBuildZExt(self.b, bits, self.vi32, self.n())
    }
    unsafe fn vabsneg_f32(&self, v: LLVMValueRef, abs: u8, neg: u8, idx: u32) -> LLVMValueRef {
        let mut v = v;
        if (abs >> idx) & 1 != 0 {
            v = self.call(&self.vfn32("fabs"), self.vf32, &[self.vf32], &[v]);
        }
        if (neg >> idx) & 1 != 0 {
            v = llvm::core::LLVMBuildFNeg(self.b, v, self.n());
        }
        v
    }
    unsafe fn vfma32(&self, a: LLVMValueRef, b: LLVMValueRef, c: LLVMValueRef) -> LLVMValueRef {
        self.call(&self.vfn32("fma"), self.vf32, &[self.vf32, self.vf32, self.vf32], &[a, b, c])
    }
    /// Per-lane runtime call for f32 ops with no vector intrinsic (bit-exact
    /// with the scalar path's helper), gathering results into a <W×?> vector.
    unsafe fn per_lane(&self, name: &str, rty: LLVMTypeRef, in_vecs: &[LLVMValueRef], in_scalar_tys: &[LLVMTypeRef], out_vty: LLVMTypeRef) -> LLVMValueRef {
        let mut res = llvm::core::LLVMGetPoison(out_vty);
        for k in 0..self.w {
            let args: Vec<LLVMValueRef> = in_vecs.iter().map(|v| llvm::core::LLVMBuildExtractElement(self.b, *v, self.ci32(k), self.n())).collect();
            let rk = self.call(name, rty, in_scalar_tys, &args);
            res = llvm::core::LLVMBuildInsertElement(self.b, res, rk, self.ci32(k), self.n());
        }
        res
    }

    /// `x * 2^exp` per lane. On AVX-512 hosts this is `vscalefpd` (hardware
    /// scalbn: x·2^⌊y⌋, single IEEE rounding, denormal/inf-correct — the exact
    /// semantics of the clamped-multiply chain below), 5 ops per 16 lanes
    /// instead of ~45; otherwise the clamped-multiply chain.
    unsafe fn vldexp(&self, value: LLVMValueRef, exp: LLVMValueRef) -> LLVMValueRef {
        if self.w == 4
            && std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512vl")
        {
            return self.vldexp_scalef_256(value, exp);
        }
        if self.w % 8 == 0 && std::arch::is_x86_feature_detected!("avx512f") {
            return self.vldexp_scalef(value, exp);
        }
        self.vldexp_chain(value, exp)
    }

    /// Exact `x * 2^exp` lowering when object analysis proves that the
    /// constructed power of two is a normal f64 value.
    unsafe fn vldexp_normal_pow2(&self, value: LLVMValueRef, exp: LLVMValueRef) -> LLVMValueRef {
        let exp64 = llvm::core::LLVMBuildSExt(self.b, exp, self.vi64, self.n());
        let biased = self.v_add(exp64, self.splat(self.ci64(1023), self.vi64));
        let bits = llvm::core::LLVMBuildShl(
            self.b,
            biased,
            self.splat(self.ci64(52), self.vi64),
            self.n(),
        );
        let scale = llvm::core::LLVMBuildBitCast(self.b, bits, self.vf64, self.n());
        self.vfmul(value, scale)
    }

    /// ldexp via `vscalefpd` for a 4-lane YMM vector (AVX-512VL form).
    unsafe fn vldexp_scalef_256(&self, value: LLVMValueRef, exp: LLVMValueRef) -> LLVMValueRef {
        let i8t = llvm::core::LLVMInt8TypeInContext(self.ctx);
        let ef = llvm::core::LLVMBuildSIToFP(self.b, exp, self.vf64, self.n());
        self.call(
            "llvm.x86.avx512.mask.scalef.pd.256",
            self.vf64,
            &[self.vf64, self.vf64, self.vf64, i8t],
            &[
                value,
                ef,
                value,
                llvm::core::LLVMConstInt(i8t, 0x0F, 0),
            ],
        )
    }

    /// ldexp via `vscalefpd` in 8-lane chunks (the 512-bit intrinsic width).
    unsafe fn vldexp_scalef(&self, value: LLVMValueRef, exp: LLVMValueRef) -> LLVMValueRef {
        let n = self.n();
        let i8t = llvm::core::LLVMInt8TypeInContext(self.ctx);
        let v8f64 = llvm::core::LLVMVectorType(self.f64t, 8);
        let v8i32 = llvm::core::LLVMVectorType(self.i32t, 8);
        let chunks = (self.w / 8) as usize;
        let mut parts: Vec<LLVMValueRef> = Vec::with_capacity(chunks);
        for c in 0..chunks {
            let (xc, ec) = if chunks == 1 {
                (value, exp)
            } else {
                let mut mask: Vec<LLVMValueRef> =
                    (0..8).map(|k| self.ci32((c * 8 + k) as u32)).collect();
                let m = llvm::core::LLVMConstVector(mask.as_mut_ptr(), 8);
                let poison_f = llvm::core::LLVMGetPoison(self.vf64);
                let poison_i = llvm::core::LLVMGetPoison(self.vi32);
                (
                    llvm::core::LLVMBuildShuffleVector(self.b, value, poison_f, m, n),
                    llvm::core::LLVMBuildShuffleVector(self.b, exp, poison_i, m, n),
                )
            };
            let ef = llvm::core::LLVMBuildSIToFP(self.b, ec, v8f64, n);
            let _ = v8i32;
            let r = self.call(
                "llvm.x86.avx512.mask.scalef.pd.512",
                v8f64,
                &[v8f64, v8f64, v8f64, i8t, self.i32t],
                &[xc, ef, xc, llvm::core::LLVMConstInt(i8t, 0xFF, 0), self.ci32(4)],
            );
            parts.push(r);
        }
        // Concatenate the 8-lane parts back to <W×f64> (pairwise shuffle tree).
        let mut width = 8u32;
        let mut vals = parts;
        while vals.len() > 1 {
            let mut next = Vec::with_capacity(vals.len() / 2);
            for p in vals.chunks(2) {
                let mut mask: Vec<LLVMValueRef> =
                    (0..width * 2).map(|k| self.ci32(k)).collect();
                let m = llvm::core::LLVMConstVector(mask.as_mut_ptr(), width * 2);
                next.push(llvm::core::LLVMBuildShuffleVector(self.b, p[0], p[1], m, n));
            }
            width *= 2;
            vals = next;
        }
        vals[0]
    }

    /// Portable fallback: 3 clamped power-of-two multiplies (avoids scalbn).
    unsafe fn vldexp_chain(&self, value: LLVMValueRef, exp: LLVMValueRef) -> LLVMValueRef {
        use llvm::LLVMIntPredicate::*;
        let mut result = value;
        let mut remaining = exp;
        for _ in 0..3 {
            let hi = self.vci32(1023);
            let lo = self.splat(llvm::core::LLVMConstInt(self.i32t, (-1022i32) as u64, 1), self.vi32);
            let c1 = llvm::core::LLVMBuildICmp(self.b, LLVMIntSLT, remaining, hi, self.n());
            let step = llvm::core::LLVMBuildSelect(self.b, c1, remaining, hi, self.n());
            let c2 = llvm::core::LLVMBuildICmp(self.b, LLVMIntSGT, step, lo, self.n());
            let step = llvm::core::LLVMBuildSelect(self.b, c2, step, lo, self.n());
            remaining = llvm::core::LLVMBuildSub(self.b, remaining, step, self.n());
            let step64 = llvm::core::LLVMBuildSExt(self.b, step, self.vi64, self.n());
            let biased = self.v_add(step64, self.splat(self.ci64(1023), self.vi64));
            let bits = llvm::core::LLVMBuildShl(self.b, biased, self.splat(self.ci64(52), self.vi64), self.n());
            let scale = llvm::core::LLVMBuildBitCast(self.b, bits, self.vf64, self.n());
            result = self.vfmul(result, scale);
        }
        result
    }

    /// The 1280-bit 2/π fraction table as a private module-level constant
    /// (same words as [super::runtime::TWO_OVER_PI_FRACTION]).
    unsafe fn trig_table(&self) -> LLVMValueRef {
        let name = b"two_over_pi_fraction\0";
        let g = llvm::core::LLVMGetNamedGlobal(self.module, name.as_ptr() as *const _);
        if !g.is_null() { return g; }
        let words = super::runtime::TWO_OVER_PI_FRACTION;
        let aty = llvm::core::LLVMArrayType2(self.i64t, words.len() as u64);
        let g = llvm::core::LLVMAddGlobal(self.module, aty, name.as_ptr() as *const _);
        let vals: Vec<LLVMValueRef> = words.iter().map(|&v| self.ci64(v)).collect();
        let init = llvm::core::LLVMConstArray2(self.i64t, vals.as_ptr() as *mut _, vals.len() as u64);
        llvm::core::LLVMSetInitializer(g, init);
        llvm::core::LLVMSetGlobalConstant(g, 1);
        llvm::core::LLVMSetLinkage(g, llvm::LLVMLinkage::LLVMPrivateLinkage);
        g
    }

    /// Vectorized `V_TRIG_PREOP_F64` (Payne–Hanek segment extraction), bit-exact
    /// with the scalar runtime helper: the 53-bit table extraction is integer-only
    /// (gather + funnel shift), and [Self::vldexp] rounds once at the final
    /// multiply exactly like `libm::ldexp`.
    unsafe fn v_trig_preop(&self, a: LLVMValueRef, s: LLVMValueRef) -> LLVMValueRef {
        use llvm::LLVMIntPredicate::*;
        let n = self.n();
        // exp = biased exponent of |a|; shift = (s & 31)*53 + max(exp - 1077, 0)
        let abits = llvm::core::LLVMBuildBitCast(self.b, a, self.vi64, n);
        let e64 = llvm::core::LLVMBuildLShr(self.b, abits, self.splat(self.ci64(52), self.vi64), n);
        let e64 = llvm::core::LLVMBuildAnd(self.b, e64, self.splat(self.ci64(0x7ff), self.vi64), n);
        let exp = llvm::core::LLVMBuildTrunc(self.b, e64, self.vi32, n);
        let seg = llvm::core::LLVMBuildAnd(self.b, s, self.vci32(31), n);
        let seg = llvm::core::LLVMBuildMul(self.b, seg, self.vci32(53), n);
        let over = llvm::core::LLVMBuildICmp(self.b, LLVMIntSGT, exp, self.vci32(1077), n);
        let extra = llvm::core::LLVMBuildSub(self.b, exp, self.vci32(1077), n);
        let extra = llvm::core::LLVMBuildSelect(self.b, over, extra, self.vci32(0), n);
        let shift = llvm::core::LLVMBuildAdd(self.b, seg, extra, n);
        // bit_offset = 1201 - 53 - shift; out of table (< 0) ⇒ result 0.
        let bit_offset = llvm::core::LLVMBuildSub(self.b, self.vci32(1201 - 53), shift, n);
        let valid = llvm::core::LLVMBuildICmp(self.b, LLVMIntSGE, bit_offset, self.vci32(0), n);
        // scale = -53 - shift (+128 if exp >= 1968)
        let hiexp = llvm::core::LLVMBuildICmp(self.b, LLVMIntSGE, exp, self.vci32(1968), n);
        let base = llvm::core::LLVMBuildSelect(
            self.b, hiexp, self.vci32((-53i32 + 128) as u32), self.vci32((-53i32) as u32), n);
        let scale = llvm::core::LLVMBuildSub(self.b, base, shift, n);
        // 64-bit window [table[word+1] : table[word]] >> bit, masked to 53 bits.
        // Invalid lanes are masked out of the gathers (their word index is wild).
        let word = llvm::core::LLVMBuildAShr(self.b, bit_offset, self.vci32(6), n);
        let bit = llvm::core::LLVMBuildAnd(self.b, bit_offset, self.vci32(63), n);
        let tbl = self.trig_table();
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        let gather = format!("llvm.masked.gather.v{}i64.v{}p0", self.w, self.w);
        let zero64 = llvm::core::LLVMConstNull(self.vi64);
        let mut idx = [word];
        let ptrs_lo = llvm::core::LLVMBuildGEP2(self.b, self.i64t, tbl, idx.as_mut_ptr(), 1, n);
        let lo = self.call(&gather, self.vi64, &[vptr, self.i32t, self.vi1, self.vi64],
                           &[ptrs_lo, self.ci32(8), valid, zero64]);
        let word1 = llvm::core::LLVMBuildAdd(self.b, word, self.vci32(1), n);
        let mut idx = [word1];
        let ptrs_hi = llvm::core::LLVMBuildGEP2(self.b, self.i64t, tbl, idx.as_mut_ptr(), 1, n);
        let hi = self.call(&gather, self.vi64, &[vptr, self.i32t, self.vi1, self.vi64],
                           &[ptrs_hi, self.ci32(8), valid, zero64]);
        let bit64 = llvm::core::LLVMBuildZExt(self.b, bit, self.vi64, n);
        let ext = self.call(&format!("llvm.fshr.v{}i64", self.w), self.vi64,
                            &[self.vi64, self.vi64, self.vi64], &[hi, lo, bit64]);
        let frac = llvm::core::LLVMBuildAnd(self.b, ext, self.splat(self.ci64((1u64 << 53) - 1), self.vi64), n);
        let rf = llvm::core::LLVMBuildUIToFP(self.b, frac, self.vf64, n);
        let scaled = self.vldexp(rf, scale);
        llvm::core::LLVMBuildSelect(self.b, valid, scaled, llvm::core::LLVMConstNull(self.vf64), n)
    }

    // Per-lane mask bit (i1 vector) of a lane-mask register's low W bits.
    unsafe fn vcc_vec(&self) -> LLVMValueRef {
        self.structured_mask(VCC).unwrap_or_else(|| self.mask_to_vec(self.ld_sgpr32(VCC)))
    }

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
    unsafe fn st_sdst_mask(&self, sdst: u8, v: LLVMValueRef) {
        if sdst == 124 || sdst == 125 { return; }
        self.st_mask(sdst as u32, v);
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

pub fn compile_program(program: &ScalarProgram, num_vgprs: usize, width: u32) -> VecKernel {
    let num_vgprs = num_vgprs.max(256);
    let addr = unsafe { compile_inner(program, num_vgprs, width, false, &BTreeMap::new()) };
    VecKernel { addr, num_vgprs, width }
}

/// Compile a resumable width-W packet whose host-side barriers do not mutate
/// VGPRs. Cross-lane programs must use `coop_xlane::compile_xlane_vec` so the
/// divergence/frame analyses receive writelane and WMMA boundary effects.
pub fn compile_cooperative(
    program: &ScalarProgram,
    num_vgprs: usize,
    width: u32,
) -> CoopVecKernel {
    compile_cooperative_with_boundary_writes(program, num_vgprs, width, &BTreeMap::new())
}

pub(super) fn compile_cooperative_with_boundary_writes(
    program: &ScalarProgram,
    num_vgprs: usize,
    width: u32,
    boundary_writes: &BTreeMap<usize, Vec<u32>>,
) -> CoopVecKernel {
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16));
    let num_vgprs = num_vgprs.max(256);
    let addr = unsafe { compile_inner(program, num_vgprs, width, true, boundary_writes) };
    CoopVecKernel {
        addr,
        num_vgprs,
        width,
        entry_pc: program.entry_pc,
    }
}

unsafe fn compile_inner(
    program: &ScalarProgram,
    num_vgprs: usize,
    width: u32,
    coop: bool,
    boundary_writes: &BTreeMap<usize, Vec<u32>>,
) -> u64 {
    let w = width;

    llvm::target::LLVM_InitializeNativeTarget();
    llvm::target::LLVM_InitializeNativeAsmParser();
    llvm::target::LLVM_InitializeNativeAsmPrinter();

    let ctx = llvm::core::LLVMContextCreate();
    let module = llvm::core::LLVMModuleCreateWithNameInContext(b"vec_kernel\0".as_ptr() as *const _, ctx);
    let b = llvm::core::LLVMCreateBuilderInContext(ctx);

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
        let mut params = [ptr, ptr, i64t, i64t, ptr, i64t, i64t];
        let fty = llvm::core::LLVMFunctionType(i64t, params.as_mut_ptr(), 7, 0);
        llvm::core::LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, fty)
    } else {
        let mut params = [ptr, ptr, i64t, i64t];
        let fty = llvm::core::LLVMFunctionType(void, params.as_mut_ptr(), 4, 0);
        llvm::core::LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, fty)
    };

    let sgprs_p = llvm::core::LLVMGetParam(func, 0);
    let vgprs_p = llvm::core::LLVMGetParam(func, 1);
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
    let mut sgpr = Vec::with_capacity(128);
    for _ in 0..128 { sgpr.push(llvm::core::LLVMBuildAlloca(b, i32t, b"\0".as_ptr() as *const _)); }
    let mut vgpr = Vec::with_capacity(num_vgprs);
    for _ in 0..num_vgprs { vgpr.push(llvm::core::LLVMBuildAlloca(b, vi32, b"\0".as_ptr() as *const _)); }
    let mut vgpr_f64 = Vec::with_capacity(num_vgprs + 1);
    for _ in 0..num_vgprs + 1 { vgpr_f64.push(llvm::core::LLVMBuildAlloca(b, vf64, b"\0".as_ptr() as *const _)); }
    let scc = llvm::core::LLVMBuildAlloca(b, i1, b"\0".as_ptr() as *const _);
    let bvh_scratch = llvm::core::LLVMBuildArrayAlloca(b, i32t, llvm::core::LLVMConstInt(i32t, 10, 0), b"\0".as_ptr() as *const _);
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

    let mut cg = Cg {
        ctx, module, func, b, w, coop, writeback_vgprs: num_vgprs,
        scratch_vec: scratch_base, // placeholder, set below
        scratch_base_scalar: scratch_base,
        scratch_stride,
        sgpr, vgpr, vgpr_f64, scc,
        f64_fresh: std::cell::Cell::new([0; 2]),
        stale: std::cell::Cell::new([0; 2]),
        fresh_in: std::collections::BTreeMap::new(),
        f64c: super::regtype::f64_read_pairs(program),
        div_cur: std::cell::Cell::new({ let mut s = [0u128; 2]; s[0] |= 1; s }),
        frame_cur: std::cell::RefCell::new(std::collections::HashMap::new()),
        i1, i32t, i64t, f32t, f64t, iw, ptr,
        vi1, vi32, vi64, vf32, vf64,
        bvh_scratch,
        spill_base,
        spill: std::cell::RefCell::new(BTreeMap::new()),
        predicate: std::cell::Cell::new(false),
        ldexp_normal_pow2: std::cell::Cell::new(false),
        structured_loop_masks: None,
        current_pc: std::cell::Cell::new(usize::MAX),
    };

    // Select at most one leaf loop whose EXEC save/restore pairs are entirely
    // local to that loop. All other blocks keep lane masks packed in SGPRs.
    let structured_mask_region = (!coop).then(|| super::analyze_structured(program))
        .and_then(|plan| plan.loops.into_iter().find(|region| {
            region.children.is_empty()
                && !region.mask_stack.local_scopes.is_empty()
                && region.mask_stack.boundary_live_saved.is_empty()
                && region.mask_stack.unrestored_saved.is_empty()
                && region.control.branch_conditions.iter().all(|cond| matches!(cond, Cond::ExecZ | Cond::ExecNz | Cond::Scc0 | Cond::Scc1))
        }));
    if let Some(region) = structured_mask_region.as_ref() {
        let mask_regs = region.mask_stack.mask_sgprs.clone();
        let masks = mask_regs.iter().copied().map(|reg| {
            (reg, llvm::core::LLVMBuildAlloca(b, vi1, cg.n()))
        }).collect();
        let init_bb = llvm::core::LLVMAppendBasicBlockInContext(ctx, func, cstr("structured_mask_init").as_ptr());
        cg.structured_loop_masks = Some(StructuredLoopMasks {
            header: region.header,
            body: region.body.iter().copied().collect(),
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

    // Initialize SGPRs (scalar, from uniform sgprs_p) and VGPRs (<W×i32> SoA).
    for i in 0..128u32 {
        let gep = llvm::core::LLVMBuildGEP2(b, i32t, sgprs_p, [cg.ci32(i)].as_mut_ptr(), 1, cg.n());
        let v = llvm::core::LLVMBuildLoad2(b, i32t, gep, cg.n());
        cg.st_sgpr32(i, v);
    }
    for i in 0..num_vgprs as u32 {
        // lanes of register i are contiguous at vgprs_p[i*W .. i*W+W]
        let gep = llvm::core::LLVMBuildGEP2(b, i32t, vgprs_p, [cg.ci32(i * w)].as_mut_ptr(), 1, cg.n());
        let ld = llvm::core::LLVMBuildLoad2(b, vi32, gep, cg.n());
        llvm::core::LLVMSetAlignment(ld, 4);
        llvm::core::LLVMBuildStore(b, ld, cg.vgpr[i as usize]);
    }
    // Seed each f64-canonical pair's cell directly from its two i32 arg halves
    // (the cell is its sole storage; the i32 slots are unused for canonical pairs).
    for p in 0..num_vgprs as u32 {
        if super::regtype::bget(&cg.f64c, p) && (p + 1) < num_vgprs as u32 {
            let lo = cg.zext64v(llvm::core::LLVMBuildLoad2(b, vi32, cg.vgpr[p as usize], cg.n()));
            let hi = cg.zext64v(llvm::core::LLVMBuildLoad2(b, vi32, cg.vgpr[p as usize + 1], cg.n()));
            let hi = llvm::core::LLVMBuildShl(b, hi, cg.splat(cg.ci64(32), vi64), cg.n());
            let u = cg.v_or(hi, lo);
            let d = llvm::core::LLVMBuildBitCast(b, u, vf64, cg.n());
            llvm::core::LLVMBuildStore(b, d, cg.vgpr_f64[p as usize]);
        }
    }
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
        cg.st_sgpr32(EXEC, cg.ci32(init_exec));
        llvm::core::LLVMBuildStore(b, llvm::core::LLVMConstInt(i1, 0, 0), scc);
    }

    cg.predicate.set(true);

    let mut bbs: BTreeMap<usize, LLVMBasicBlockRef> = BTreeMap::new();
    for (&pc, _) in &program.blocks {
        let name = cstr(&format!("b{:x}", pc));
        bbs.insert(pc, llvm::core::LLVMAppendBasicBlockInContext(ctx, func, name.as_ptr()));
    }
    if let Some(loop_masks) = cg.structured_loop_masks.as_mut() {
        let exits: Vec<(usize, usize)> = loop_masks.body.iter().copied().flat_map(|from| {
            succs_for_emit(&program.blocks[&from]).into_iter()
                .filter(|to| !loop_masks.body.contains(to))
                .map(move |to| (from, to))
        }).collect();
        for (from, to) in exits {
            let name = cstr(&format!("structured_mask_exit_{from:x}_{to:x}"));
            loop_masks.exit_bbs.insert((from, to), llvm::core::LLVMAppendBasicBlockInContext(ctx, func, name.as_ptr()));
        }
    }
    if coop {
        let resume = llvm::core::LLVMGetParam(func, 5);
        let mut targets: Vec<usize> = program
            .blocks
            .values()
            .filter_map(|block| match block.term {
                Terminator::Barrier { resume } => Some(resume),
                _ => None,
            })
            .collect();
        targets.sort_unstable();
        targets.dedup();
        let switch = llvm::core::LLVMBuildSwitch(
            b,
            resume,
            bbs[&program.entry_pc],
            targets.len() as u32,
        );
        for pc in targets {
            llvm::core::LLVMAddCase(switch, cg.ci64(pc as u64), bbs[&pc]);
        }
    } else {
        llvm::core::LLVMBuildBr(b, bbs[&program.entry_pc]);
    }

    let base_pred = true;
    // Mask elision: a VGPR write skips EXEC predication when none of its
    // destinations is live at a CFG reconvergence point or caller-observed
    // fragment exit. Reconvergence-visible state stays predicated, while
    // transient arithmetic may run mask-free; memory and compare writes remain
    // EXEC-masked. See [super::vec_live].
    let elide = super::vec_live::analyze_with_exit_live(program, &[]);
    let f64_fresh_in = super::freshness::analyze(program);
    cg.fresh_in = f64_fresh_in.clone();
    let div_in = if coop {
        let mut seed = [0u128; 2];
        seed[0] |= 1;
        super::vec_live::divergent_entry_with_seed_and_boundary_writes(
            program,
            seed,
            boundary_writes,
        )
    } else {
        super::vec_live::divergent_entry(program)
    };
    let frame_in = if coop {
        super::vec_live::frame_entry_with_boundary_writes(program, boundary_writes)
    } else {
        super::vec_live::frame_entry(program)
    };
    // Enable the reconvergence-liveness rule computed above (see vec_live):
    // arithmetic temps unmasked, reconvergence-visible state updates masked.
    // The cooperative ABI materializes the register file at every coroutine
    // boundary. Keep all VALU writes EXEC-predicated there; the whole-program
    // path can still use the reconvergence-liveness elision.
    let do_elide = !coop;
    // Divergent-pointer load clustering (record load + transpose): needs whole
    // 8-lane transpose blocks.
    let do_cluster = w % 8 == 0;

    for (&pc, block) in &program.blocks {
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
        cg.set_f64_fresh(f64_fresh_in[&pc]);
        // Conservatively assume every fresh-on-entry shadow pair's i32 slots are
        // stale (a predecessor may have skipped the sync); canonical pairs have
        // no live slots and are excluded so edge syncs don't write dead stores.
        cg.stale.set({
            let mut st = f64_fresh_in[&pc];
            for r in 0..256u32 {
                if super::regtype::bget(&st, r) && super::regtype::bget(&cg.f64c, r) {
                    st[(r / 128) as usize] &= !(1u128 << (r % 128));
                }
            }
            st
        });
        cg.div_cur.set(div_in[&pc]);
        *cg.frame_cur.borrow_mut() = frame_in[&pc].clone();
        let flags = &elide[&pc];
        let normal_sqrt_ldexp = normal_sqrt_ldexp_indices(&block.body);
        // Instructions already emitted as part of a divergent-pointer load
        // cluster (record load + transpose); bookkeeping below still runs.
        let mut cluster_rest = 0usize;
        for (idx, inst) in block.body.iter().enumerate() {
            cg.ldexp_normal_pow2.set(normal_sqrt_ldexp[idx]);
            if cluster_rest > 0 {
                cluster_rest -= 1;
            } else {
            cg.predicate.set(base_pred && !(do_elide && flags[idx]));
            let cluster = if do_cluster { vg_cluster(&block.body[idx..]) } else { None };
            if let Some(c) = cluster.filter(|c| cg.cluster_applicable(c)) {
                let members: Vec<&VGLOBAL> = block.body[idx..idx + c.len]
                    .iter()
                    .map(|i| match i { InstFormat::VGLOBAL(g) => g, _ => unreachable!() })
                    .collect();
                let preds: Vec<bool> = (0..c.len).map(|m| base_pred && !(do_elide && flags[idx + m])).collect();
                cg.emit_vglobal_cluster(&members, c.lo, c.span, &preds);
                cluster_rest = c.len - 1;
            } else {
                cg.emit_inst(inst);
            }
            }
            // advance flow-sensitive divergence + frame-pointer tracking
            let mut d = cg.div_cur.get();
            super::vec_live::div_transfer(inst, &mut d);
            cg.div_cur.set(d);
            super::vec_live::frame_transfer(inst, &mut cg.frame_cur.borrow_mut());
        }
        cg.emit_term(&block.term, &bbs);
    }

    if let Some(loop_masks) = cg.structured_loop_masks.as_ref() {
        loop_masks.active.set(false);
        llvm::core::LLVMPositionBuilderAtEnd(b, loop_masks.init_bb);
        for (&reg, &cell) in &loop_masks.masks {
            let scalar = cg.ld_sgpr32(reg);
            llvm::core::LLVMBuildStore(b, cg.mask_to_vec(scalar), cell);
        }
        llvm::core::LLVMBuildBr(b, bbs[&loop_masks.header]);
        for (&(_, to), &exit_bb) in &loop_masks.exit_bbs {
            llvm::core::LLVMPositionBuilderAtEnd(b, exit_bb);
            cg.sync_structured_masks_to_sgpr();
            llvm::core::LLVMBuildBr(b, bbs[&to]);
        }
    }

    finalize(ctx, module, func)
}

impl Cg {
    /// Persist packet state at a cross-lane yield or final return. Values are
    /// loaded through the canonical accessors so lazily materialized f64 cells
    /// are written back correctly without first forcing their shadow slots.
    unsafe fn emit_coop_writeback(&self) {
        let sgprs_p = llvm::core::LLVMGetParam(self.func, 0);
        let vgprs_p = llvm::core::LLVMGetParam(self.func, 1);
        for i in 0..128u32 {
            let gep = llvm::core::LLVMBuildGEP2(
                self.b,
                self.i32t,
                sgprs_p,
                [self.ci32(i)].as_mut_ptr(),
                1,
                self.n(),
            );
            llvm::core::LLVMBuildStore(self.b, self.ld_sgpr32(i), gep);
        }
        let scc_gep = llvm::core::LLVMBuildGEP2(
            self.b,
            self.i32t,
            sgprs_p,
            [self.ci32(128)].as_mut_ptr(),
            1,
            self.n(),
        );
        let scc = llvm::core::LLVMBuildZExt(self.b, self.ld_scc(), self.i32t, self.n());
        llvm::core::LLVMBuildStore(self.b, scc, scc_gep);

        for i in 0..self.writeback_vgprs as u32 {
            let gep = llvm::core::LLVMBuildGEP2(
                self.b,
                self.i32t,
                vgprs_p,
                [self.ci32(i * self.w)].as_mut_ptr(),
                1,
                self.n(),
            );
            let store = llvm::core::LLVMBuildStore(self.b, self.ld_vgpr32(i), gep);
            llvm::core::LLVMSetAlignment(store, 4);
        }
    }

    unsafe fn emit_term(&self, term: &Terminator, bbs: &BTreeMap<usize, LLVMBasicBlockRef>) {
        match term {
            Terminator::Return => {
                if self.coop {
                    self.emit_coop_writeback();
                    llvm::core::LLVMBuildRet(self.b, self.ci64(super::emit::COOP_DONE));
                } else {
                    llvm::core::LLVMBuildRetVoid(self.b);
                }
            }
            Terminator::Jump(t) => {
                self.sync_stale_for(&[*t]);
                llvm::core::LLVMBuildBr(self.b, self.structured_mask_target(self.current_pc.get(), *t, bbs));
            }
            Terminator::Branch { cond, taken, fallthrough } => {
                let c = self.taken_cond(*cond);
                self.sync_stale_for(&[*taken, *fallthrough]);
                llvm::core::LLVMBuildCondBr(
                    self.b,
                    c,
                    self.structured_mask_target(self.current_pc.get(), *taken, bbs),
                    self.structured_mask_target(self.current_pc.get(), *fallthrough, bbs),
                );
            }
            Terminator::Barrier { resume } => {
                if !self.coop {
                    panic!("the packed-work-item backend does not support workgroup barriers");
                }
                self.emit_coop_writeback();
                llvm::core::LLVMBuildRet(self.b, self.ci64(*resume as u64));
            }
        }
    }

    /// Branch conditions test the **packed lanes** (low W bits): an `execz` arm
    /// is taken iff no packed work-item is active.
    unsafe fn taken_cond(&self, cond: Cond) -> LLVMValueRef {
        use llvm::LLVMIntPredicate::*;
        let lanes = if self.w >= 32 { 0xFFFF_FFFFu32 } else { (1u32 << self.w) - 1 };
        match cond {
            Cond::ExecZ => {
                if self.has_structured_mask(EXEC) {
                    return llvm::core::LLVMBuildICmp(
                        self.b,
                        LLVMIntEQ,
                        self.vec_to_mask(self.exec_vec()),
                        self.ci32(0),
                        self.n(),
                    );
                }
                let e = self.v_and(self.ld_sgpr32(EXEC), self.ci32(lanes));
                llvm::core::LLVMBuildICmp(self.b, LLVMIntEQ, e, self.ci32(0), self.n())
            }
            Cond::ExecNz => {
                if self.has_structured_mask(EXEC) {
                    return llvm::core::LLVMBuildICmp(
                        self.b,
                        LLVMIntNE,
                        self.vec_to_mask(self.exec_vec()),
                        self.ci32(0),
                        self.n(),
                    );
                }
                let e = self.v_and(self.ld_sgpr32(EXEC), self.ci32(lanes));
                llvm::core::LLVMBuildICmp(self.b, LLVMIntNE, e, self.ci32(0), self.n())
            }
            Cond::VccZ => {
                let e = self.v_and(self.ld_sgpr32(VCC), self.ci32(lanes));
                llvm::core::LLVMBuildICmp(self.b, LLVMIntEQ, e, self.ci32(0), self.n())
            }
            Cond::VccNz => {
                let e = self.v_and(self.ld_sgpr32(VCC), self.ci32(lanes));
                llvm::core::LLVMBuildICmp(self.b, LLVMIntNE, e, self.ci32(0), self.n())
            }
            Cond::Scc1 => self.ld_scc(),
            Cond::Scc0 => {
                let s = self.ld_scc();
                llvm::core::LLVMBuildICmp(self.b, LLVMIntEQ, s, llvm::core::LLVMConstInt(self.i1, 0, 0), self.n())
            }
        }
    }
}

fn finalize(
    ctx: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    func: LLVMValueRef,
) -> u64 {
    unsafe {
        let mut err = std::ptr::null_mut();
        if llvm::analysis::LLVMVerifyModule(
            module,
            llvm::analysis::LLVMVerifierFailureAction::LLVMPrintMessageAction,
            &mut err,
        ) != 0 {
            if !err.is_null() {
                let s = std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned();
                panic!("vec module failed verification:\n{}", s);
            }
        }
        let triple = llvm::target_machine::LLVMGetDefaultTargetTriple();
        let mut target = std::ptr::null_mut();
        let mut terr = std::ptr::null_mut();
        llvm::target_machine::LLVMGetTargetFromTriple(triple, &mut target, &mut terr);
        let cpu = llvm::target_machine::LLVMGetHostCPUName();
        let feat = llvm::target_machine::LLVMGetHostCPUFeatures();
        let tm = llvm::target_machine::LLVMCreateTargetMachine(
            target, triple, cpu, feat,
            llvm::target_machine::LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            llvm::target_machine::LLVMRelocMode::LLVMRelocDefault,
            llvm::target_machine::LLVMCodeModel::LLVMCodeModelJITDefault,
        );
        let opts = llvm::transforms::pass_builder::LLVMCreatePassBuilderOptions();
        let passes_str: std::ffi::CString = std::ffi::CString::new("default<O3>").unwrap();
        let perr = llvm::transforms::pass_builder::LLVMRunPasses(module, passes_str.as_ptr(), tm, opts);
        if !perr.is_null() {
            let msg = llvm::error::LLVMGetErrorMessage(perr);
            let s = std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned();
            panic!("vec passes failed: {}", s);
        }
        let _ = func;
        let jit_builder = llvm::orc2::lljit::LLVMOrcCreateLLJITBuilder();
        let jtmb = llvm::orc2::LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm);
        llvm::orc2::lljit::LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(jit_builder, jtmb);
        let mut jit = std::ptr::null_mut();
        let e = llvm::orc2::lljit::LLVMOrcCreateLLJIT(&mut jit, jit_builder);
        if !e.is_null() { panic!("create LLJIT failed"); }
        let dylib = llvm::orc2::lljit::LLVMOrcLLJITGetMainJITDylib(jit);
        let gp = llvm::orc2::lljit::LLVMOrcLLJITGetGlobalPrefix(jit);
        let mut dg = std::ptr::null_mut();
        llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(&mut dg, gp, None, std::ptr::null_mut());
        llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, dg);
        let lib_path: &[u8] = if cfg!(debug_assertions) {
            b"target/debug/libamdgpu_sim.so\0"
        } else {
            b"target/release/libamdgpu_sim.so\0"
        };
        let mut dg2 = std::ptr::null_mut();
        llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(&mut dg2, lib_path.as_ptr() as *const _, gp, None, std::ptr::null_mut());
        llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, dg2);
        let tsctx = llvm::orc2::LLVMOrcCreateNewThreadSafeContext();
        let tsm = llvm::orc2::LLVMOrcCreateNewThreadSafeModule(module, tsctx);
        let e = llvm::orc2::lljit::LLVMOrcLLJITAddLLVMIRModule(jit, dylib, tsm);
        if !e.is_null() { panic!("add module failed"); }
        let mut addr = 0u64;
        let e = llvm::orc2::lljit::LLVMOrcLLJITLookup(jit, &mut addr, b"kernel\0".as_ptr() as *const _);
        if !e.is_null() { panic!("lookup kernel failed"); }
        std::mem::forget(Box::new(jit));
        let _ = ctx;
        addr
    }
}

// =====================================================================
//  Instruction emission
// =====================================================================

impl Cg {
    unsafe fn emit_inst(&self, inst: &InstFormat) {
        match inst {
            InstFormat::VOP1(i) => self.emit_vop1(i),
            InstFormat::VOP2(i) => self.emit_vop2(i),
            InstFormat::VOP3(i) => self.emit_vop3(i),
            InstFormat::VOP3P(i) => self.emit_vop3p(i),
            InstFormat::VOP3SD(i) => self.emit_vop3sd(i),
            InstFormat::VOPC(i) => self.emit_vopc(i),
            InstFormat::VOPD(i) => self.emit_vopd(i),
            InstFormat::SOP1(i) => self.emit_sop1(i),
            InstFormat::SOP2(i) => self.emit_sop2(i),
            InstFormat::SOPC(i) => self.emit_sopc(i),
            InstFormat::SMEM(i) => self.emit_smem(i),
            InstFormat::VGLOBAL(i) => self.emit_vglobal(i),
            InstFormat::SOPK(i) => self.emit_sopk(i),
            InstFormat::VFLAT(i) => self.emit_vflat(i),
            InstFormat::VSCRATCH(i) => self.emit_vscratch(i),
            InstFormat::VIMAGE(i) => self.emit_vimage(i),
            InstFormat::VSAMPLE(i) => self.emit_vsample(i),
            other => panic!("vec: unsupported instruction {:?}", other),
        }
    }

    unsafe fn emit_vop1(&self, i: &VOP1) {
        match i.op {
            I::V_MOV_B32 => { let v = self.vsrc_u32(&i.src0); self.st_vgpr32(i.vdst as u32, v); }
            I::V_CVT_F64_I32 => { let s = self.vsrc_u32(&i.src0); let v = llvm::core::LLVMBuildSIToFP(self.b, s, self.vf64, self.n()); self.st_vgpr_f64(i.vdst as u32, v); }
            I::V_CVT_F64_U32 => { let s = self.vsrc_u32(&i.src0); let v = llvm::core::LLVMBuildUIToFP(self.b, s, self.vf64, self.n()); self.st_vgpr_f64(i.vdst as u32, v); }
            I::V_CVT_I32_F64 => { let s = self.vsrc_f64(&i.src0); let v = self.call(&format!("llvm.fptosi.sat.v{}i32.v{}f64", self.w, self.w), self.vi32, &[self.vf64], &[s]); self.st_vgpr32(i.vdst as u32, v); }
            I::V_RCP_F64 => { let s = self.vsrc_f64(&i.src0); let v = self.vfdiv(self.vcf64(1.0), s); self.st_vgpr_f64(i.vdst as u32, v); }
            I::V_RSQ_F64 => { let s = self.vsrc_f64(&i.src0); let q = self.vsqrt(s); let v = self.vfdiv(self.vcf64(1.0), q); self.st_vgpr_f64(i.vdst as u32, v); }
            I::V_SQRT_F64 => { let s = self.vsrc_f64(&i.src0); let v = self.vsqrt(s); self.st_vgpr_f64(i.vdst as u32, v); }
            I::V_FRACT_F64 => { let s = self.vsrc_f64(&i.src0); let f = self.vfloor(s); let v = self.vfadd(s, llvm::core::LLVMBuildFNeg(self.b, f, self.n())); self.st_vgpr_f64(i.vdst as u32, v); }
            I::V_RNDNE_F64 => { let s = self.vsrc_f64(&i.src0); let v = self.call(&self.vfn("roundeven"), self.vf64, &[self.vf64], &[s]); self.st_vgpr_f64(i.vdst as u32, v); }
            I::V_RCP_F32 | I::V_RCP_IFLAG_F32 => { let s = self.vsrc_f32(&i.src0); let v = self.vfdiv(self.vcf32(1.0), s); self.st_vgpr32(i.vdst as u32, self.vf32_bits(v)); }
            I::V_SQRT_F32 => { let v = self.call(&self.vfn32("sqrt"), self.vf32, &[self.vf32], &[self.vsrc_f32(&i.src0)]); self.st_vgpr32(i.vdst as u32, self.vf32_bits(v)); }
            I::V_RSQ_F32 => { let q = self.call(&self.vfn32("sqrt"), self.vf32, &[self.vf32], &[self.vsrc_f32(&i.src0)]); self.st_vgpr32(i.vdst as u32, self.vf32_bits(self.vfdiv(self.vcf32(1.0), q))); }
            I::V_RNDNE_F32 | I::V_FLOOR_F32 | I::V_CEIL_F32 | I::V_TRUNC_F32 => {
                let name = match i.op { I::V_RNDNE_F32 => "roundeven", I::V_FLOOR_F32 => "floor", I::V_CEIL_F32 => "ceil", _ => "trunc" };
                let v = self.call(&self.vfn32(name), self.vf32, &[self.vf32], &[self.vsrc_f32(&i.src0)]);
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(v));
            }
            I::V_CLZ_I32_U32 => {
                let x = self.vsrc_u32(&i.src0);
                let lz = self.call(&format!("llvm.ctlz.v{}i32", self.w), self.vi32, &[self.vi32, self.i1], &[x, llvm::core::LLVMConstInt(self.i1, 0, 0)]);
                let is0 = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntEQ, x, self.vci32(0), self.n());
                self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildSelect(self.b, is0, self.vci32(0xFFFF_FFFF), lz, self.n()));
            }
            I::V_FREXP_MANT_F32 => { let v = self.per_lane("scalar_frexp_mant_f32", self.f32t, &[self.vsrc_f32(&i.src0)], &[self.f32t], self.vf32); self.st_vgpr32(i.vdst as u32, self.vf32_bits(v)); }
            I::V_FREXP_EXP_I32_F32 => { let v = self.per_lane("scalar_frexp_exp_f32", self.i32t, &[self.vsrc_f32(&i.src0)], &[self.f32t], self.vi32); self.st_vgpr32(i.vdst as u32, v); }
            I::V_CVT_U32_F32 => { let v = self.call(&format!("llvm.fptoui.sat.v{}i32.v{}f32", self.w, self.w), self.vi32, &[self.vf32], &[self.vsrc_f32(&i.src0)]); self.st_vgpr32(i.vdst as u32, v); }
            I::V_CVT_I32_F32 => { let v = self.call(&format!("llvm.fptosi.sat.v{}i32.v{}f32", self.w, self.w), self.vi32, &[self.vf32], &[self.vsrc_f32(&i.src0)]); self.st_vgpr32(i.vdst as u32, v); }
            I::V_CVT_F32_I32 => { let f = llvm::core::LLVMBuildSIToFP(self.b, self.vsrc_u32(&i.src0), self.vf32, self.n()); self.st_vgpr32(i.vdst as u32, self.vf32_bits(f)); }
            I::V_CVT_F32_U32 => { let f = llvm::core::LLVMBuildUIToFP(self.b, self.vsrc_u32(&i.src0), self.vf32, self.n()); self.st_vgpr32(i.vdst as u32, self.vf32_bits(f)); }
            I::V_READFIRSTLANE_B32 => {
                // Value from the lowest active lane, broadcast to an SGPR (uniform).
                // cttz with is_zero_undef=false yields W for EXEC==0 (a block may
                // run predicated with EXEC==0); clamp the index to a valid lane so
                // the uniform destination never receives poison.
                let src = self.vsrc_u32(&i.src0);
                let exec = self.ld_sgpr32(EXEC);
                let tz = self.call("llvm.cttz.i32", self.i32t, &[self.i32t, self.i1], &[exec, llvm::core::LLVMConstInt(self.i1, 0, 0)]);
                let over = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntUGE, tz, self.ci32(self.w), self.n());
                let idx = llvm::core::LLVMBuildSelect(self.b, over, self.ci32(0), tz, self.n());
                let v = llvm::core::LLVMBuildExtractElement(self.b, src, idx, self.n());
                self.st_sgpr32(i.vdst as u32, v);
            }
            _ => panic!("vec: unsupported VOP1 {:?}", i.op),
        }
    }

    unsafe fn emit_vop2(&self, i: &VOP2) {
        match i.op {
            I::V_ADD_F64 => { let a = self.vsrc_f64(&i.src0); let b = self.ld_vgpr_f64(i.vsrc1 as u32); let r = self.vfadd(a, b); self.st_vgpr_f64(i.vdst as u32, r); return; }
            I::V_MUL_F64 => { let a = self.vsrc_f64(&i.src0); let b = self.ld_vgpr_f64(i.vsrc1 as u32); let r = self.vfmul(a, b); self.st_vgpr_f64(i.vdst as u32, r); return; }
            I::V_MAX_NUM_F64 | I::V_MIN_NUM_F64 => {
                let a = self.vsrc_f64(&i.src0); let b = self.ld_vgpr_f64(i.vsrc1 as u32);
                let name = if matches!(i.op, I::V_MAX_NUM_F64) { self.vfn("maxnum") } else { self.vfn("minnum") };
                let r = self.call(&name, self.vf64, &[self.vf64, self.vf64], &[a, b]); self.st_vgpr_f64(i.vdst as u32, r); return;
            }
            I::V_MUL_F32 => { let a = self.vsrc_f32(&i.src0); let b = self.vf32_of(self.ld_vgpr32(i.vsrc1 as u32)); self.st_vgpr32(i.vdst as u32, self.vf32_bits(self.vfmul(a, b))); return; }
            I::V_ADD_F32 | I::V_SUB_F32 | I::V_SUBREV_F32 => {
                let a = self.vsrc_f32(&i.src0); let b = self.vf32_of(self.ld_vgpr32(i.vsrc1 as u32));
                let r = match i.op { I::V_ADD_F32 => self.vfadd(a, b), I::V_SUB_F32 => self.vfsub(a, b), _ => self.vfsub(b, a) };
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(r)); return;
            }
            I::V_FMAC_F32 => {
                let d = self.vf32_of(self.ld_vgpr32(i.vdst as u32));
                let r = self.vfma32(self.vsrc_f32(&i.src0), self.vf32_of(self.ld_vgpr32(i.vsrc1 as u32)), d);
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(r)); return;
            }
            I::V_FMAMK_F32 => {
                let k = self.vcf32_bits(i.literal_constant.unwrap());
                let r = self.vfma32(self.vsrc_f32(&i.src0), k, self.vf32_of(self.ld_vgpr32(i.vsrc1 as u32)));
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(r)); return;
            }
            I::V_FMAAK_F32 => {
                let k = self.vcf32_bits(i.literal_constant.unwrap());
                let r = self.vfma32(self.vsrc_f32(&i.src0), self.vf32_of(self.ld_vgpr32(i.vsrc1 as u32)), k);
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(r)); return;
            }
            I::V_ADD_CO_CI_U32 => {
                let s0 = self.zext64v(self.vsrc_u32(&i.src0));
                let s1 = self.zext64v(self.ld_vgpr32(i.vsrc1 as u32));
                let cin = self.zext64v(self.v_and(self.mask_to_u32(VCC), self.vci32(1)));
                let sum = self.v_add(self.v_add(s0, s1), cin);
                self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildTrunc(self.b, sum, self.vi32, self.n()));
                let cout = llvm::core::LLVMBuildTrunc(self.b, llvm::core::LLVMBuildLShr(self.b, sum, self.splat(self.ci64(32), self.vi64), self.n()), self.vi32, self.n());
                let cout = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, cout, self.vci32(0), self.n());
                self.st_mask(VCC, cout); return;
            }
            I::V_LSHLREV_B64 => {
                let amt = self.v_and(self.vsrc_u32(&i.src0), self.vci32(63));
                let amt = llvm::core::LLVMBuildZExt(self.b, amt, self.vi64, self.n());
                let v = self.ld_vgpr64(i.vsrc1 as u32);
                let r = llvm::core::LLVMBuildShl(self.b, v, amt, self.n());
                self.st_vgpr64(i.vdst as u32, r); return;
            }
            I::V_SUB_CO_CI_U32 | I::V_SUBREV_CO_CI_U32 => {
                let s0 = self.zext64v(self.vsrc_u32(&i.src0));
                let s1 = self.zext64v(self.ld_vgpr32(i.vsrc1 as u32));
                let (a, b) = if matches!(i.op, I::V_SUBREV_CO_CI_U32) { (s1, s0) } else { (s0, s1) };
                let cin = self.zext64v(self.v_and(self.mask_to_u32(VCC), self.vci32(1)));
                let diff = llvm::core::LLVMBuildSub(self.b, llvm::core::LLVMBuildSub(self.b, a, b, self.n()), cin, self.n());
                self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildTrunc(self.b, diff, self.vi32, self.n()));
                let bo = llvm::core::LLVMBuildTrunc(self.b, llvm::core::LLVMBuildLShr(self.b, diff, self.splat(self.ci64(32), self.vi64), self.n()), self.vi32, self.n());
                let bo = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, bo, self.vci32(0), self.n());
                self.st_mask(VCC, bo); return;
            }
            _ => {}
        }
        let s0 = self.vsrc_u32(&i.src0);
        let s1 = self.ld_vgpr32(i.vsrc1 as u32);
        let r = match i.op {
            I::V_ADD_NC_U32 => self.v_add(s0, s1),
            I::V_SUB_NC_U32 => llvm::core::LLVMBuildSub(self.b, s0, s1, self.n()),
            I::V_SUBREV_NC_U32 => llvm::core::LLVMBuildSub(self.b, s1, s0, self.n()),
            I::V_AND_B32 => self.v_and(s0, s1),
            I::V_XOR_B32 => self.v_xor(s0, s1),
            I::V_OR_B32 => self.v_or(s0, s1),
            I::V_LSHLREV_B32 => self.v_shl(s1, s0),
            I::V_LSHRREV_B32 => self.v_lshr(s1, s0),
            I::V_MUL_LO_U32 => llvm::core::LLVMBuildMul(self.b, s0, s1, self.n()),
            I::V_MAX_U32 => { let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntUGT, s0, s1, self.n()); llvm::core::LLVMBuildSelect(self.b, c, s0, s1, self.n()) }
            I::V_MIN_U32 => { let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntULT, s0, s1, self.n()); llvm::core::LLVMBuildSelect(self.b, c, s0, s1, self.n()) }
            I::V_MAX_I32 => { let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntSGT, s0, s1, self.n()); llvm::core::LLVMBuildSelect(self.b, c, s0, s1, self.n()) }
            I::V_MIN_I32 => { let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntSLT, s0, s1, self.n()); llvm::core::LLVMBuildSelect(self.b, c, s0, s1, self.n()) }
            I::V_CNDMASK_B32 => { let c = self.vcc_vec(); llvm::core::LLVMBuildSelect(self.b, c, s1, s0, self.n()) }
            _ => panic!("vec: unsupported VOP2 {:?}", i.op),
        };
        self.st_vgpr32(i.vdst as u32, r);
    }

    unsafe fn emit_vop3(&self, i: &VOP3) {
        match i.op {
            I::V_ADD_NC_U32 => self.vop3_int(i, |c, a, b, _| c.v_add(a, b)),
            I::V_AND_B32 => self.vop3_int(i, |c, a, b, _| c.v_and(a, b)),
            I::V_XOR_B32 => self.vop3_int(i, |c, a, b, _| c.v_xor(a, b)),
            I::V_OR_B32 => self.vop3_int(i, |c, a, b, _| c.v_or(a, b)),
            I::V_LSHLREV_B32 => self.vop3_int(i, |c, a, b, _| c.v_shl(b, a)),
            I::V_LSHRREV_B32 => self.vop3_int(i, |c, a, b, _| c.v_lshr(b, a)),
            I::V_MUL_LO_U32 => self.vop3_int(i, |c, a, b, _| llvm::core::LLVMBuildMul(c.b, a, b, c.n())),
            I::V_ADD3_U32 => { let a = self.vsrc_u32(&i.src0); let b = self.vsrc_u32(&i.src1); let cc = self.vsrc_u32(&i.src2); let r = self.v_add(self.v_add(a, b), cc); self.st_vgpr32(i.vdst as u32, r); }
            I::V_XOR3_B32 => { let a = self.vsrc_u32(&i.src0); let b = self.vsrc_u32(&i.src1); let cc = self.vsrc_u32(&i.src2); let r = self.v_xor(self.v_xor(a, b), cc); self.st_vgpr32(i.vdst as u32, r); }
            I::V_XAD_U32 => { let a = self.vsrc_u32(&i.src0); let b = self.vsrc_u32(&i.src1); let cc = self.vsrc_u32(&i.src2); let r = self.v_add(self.v_xor(a, b), cc); self.st_vgpr32(i.vdst as u32, r); }
            I::V_BFE_U32 => {
                let data = self.vsrc_u32(&i.src0); let off = self.vsrc_u32(&i.src1); let wid = self.v_and(self.vsrc_u32(&i.src2), self.vci32(31));
                let one = self.vci32(1);
                let mask = llvm::core::LLVMBuildSub(self.b, llvm::core::LLVMBuildShl(self.b, one, wid, self.n()), self.vci32(1), self.n());
                let r = self.v_and(self.v_lshr(data, off), mask); self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_CNDMASK_B32 => {
                // e64: abs/neg apply to s0/s1 as floats (no-op when 0); explicit
                // condition operand (src2). Matches v_cndmask_b32_e64.
                let s0 = self.vf32_bits(self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0));
                let s1 = self.vf32_bits(self.vabsneg_f32(self.vsrc_f32(&i.src1), i.abs, i.neg, 1));
                let c = self.src_mask_vec(&i.src2);
                let r = llvm::core::LLVMBuildSelect(self.b, c, s1, s0, self.n()); self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_LSHL_OR_B32 | I::V_LSHL_ADD_U32 => {
                let s0 = self.vsrc_u32(&i.src0); let s1 = self.v_and(self.vsrc_u32(&i.src1), self.vci32(31)); let s2 = self.vsrc_u32(&i.src2);
                let sh = self.v_shl(s0, s1);
                let r = if matches!(i.op, I::V_LSHL_OR_B32) { self.v_or(sh, s2) } else { self.v_add(sh, s2) };
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_ADD_LSHL_U32 => {
                let s2 = self.v_and(self.vsrc_u32(&i.src2), self.vci32(31));
                let r = self.v_shl(self.v_add(self.vsrc_u32(&i.src0), self.vsrc_u32(&i.src1)), s2);
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_AND_OR_B32 => { let r = self.v_or(self.v_and(self.vsrc_u32(&i.src0), self.vsrc_u32(&i.src1)), self.vsrc_u32(&i.src2)); self.st_vgpr32(i.vdst as u32, r); }
            I::V_OR3_B32 => { let r = self.v_or(self.v_or(self.vsrc_u32(&i.src0), self.vsrc_u32(&i.src1)), self.vsrc_u32(&i.src2)); self.st_vgpr32(i.vdst as u32, r); }
            I::V_BFI_B32 => {
                let a = self.vsrc_u32(&i.src0);
                let not_a = self.v_xor(a, self.vci32(0xFFFF_FFFF));
                let r = self.v_or(self.v_and(a, self.vsrc_u32(&i.src1)), self.v_and(not_a, self.vsrc_u32(&i.src2)));
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_ALIGNBIT_B32 => {
                let s0 = self.zext64v(self.vsrc_u32(&i.src0));
                let s1 = self.zext64v(self.vsrc_u32(&i.src1));
                let amt = self.zext64v(self.v_and(self.vsrc_u32(&i.src2), self.vci32(0x1F)));
                // {S0,S1}: S0 is the MSBs, S1 the LSBs (ISA §V_ALIGNBIT_B32).
                let concat = self.v_or(llvm::core::LLVMBuildShl(self.b, s0, self.splat(self.ci64(32), self.vi64), self.n()), s1);
                let r = llvm::core::LLVMBuildTrunc(self.b, llvm::core::LLVMBuildLShr(self.b, concat, amt, self.n()), self.vi32, self.n());
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_LSHLREV_B16 => { let amt = self.v_and(self.vsrc_u32(&i.src0), self.vci32(15)); let v = self.v_and(self.vsrc_u32(&i.src1), self.vci32(0xffff)); self.st_vgpr32(i.vdst as u32, self.v_and(self.v_shl(v, amt), self.vci32(0xffff))); }
            I::V_LSHRREV_B16 => { let amt = self.v_and(self.vsrc_u32(&i.src0), self.vci32(15)); let v = self.v_and(self.vsrc_u32(&i.src1), self.vci32(0xffff)); self.st_vgpr32(i.vdst as u32, self.v_lshr(v, amt)); }
            I::V_LSHLREV_B64 => { let amt = self.zext64v(self.v_and(self.vsrc_u32(&i.src0), self.vci32(63))); let v = self.vsrc_u64(&i.src1); self.st_vgpr64(i.vdst as u32, llvm::core::LLVMBuildShl(self.b, v, amt, self.n())); }
            I::V_LSHRREV_B64 => { let amt = self.zext64v(self.v_and(self.vsrc_u32(&i.src0), self.vci32(63))); let v = self.vsrc_u64(&i.src1); self.st_vgpr64(i.vdst as u32, llvm::core::LLVMBuildLShr(self.b, v, amt, self.n())); }
            I::V_ASHRREV_I32 => { let amt = self.v_and(self.vsrc_u32(&i.src0), self.vci32(31)); let v = self.vsrc_u32(&i.src1); self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildAShr(self.b, v, amt, self.n())); }
            I::V_MUL_F32 => { let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0); let b = self.vabsneg_f32(self.vsrc_f32(&i.src1), i.abs, i.neg, 1); self.st_vgpr32(i.vdst as u32, self.vf32_bits(self.vfmul(a, b))); }
            I::V_ADD_F32 | I::V_SUB_F32 => {
                let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0); let b = self.vabsneg_f32(self.vsrc_f32(&i.src1), i.abs, i.neg, 1);
                let r = if matches!(i.op, I::V_ADD_F32) { self.vfadd(a, b) } else { self.vfsub(a, b) };
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(r));
            }
            I::V_MAX_F32 | I::V_MIN_F32 | I::V_MAX_NUM_F32 | I::V_MIN_NUM_F32 => {
                let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0); let b = self.vabsneg_f32(self.vsrc_f32(&i.src1), i.abs, i.neg, 1);
                let name = if matches!(i.op, I::V_MAX_F32 | I::V_MAX_NUM_F32) { self.vfn32("maxnum") } else { self.vfn32("minnum") };
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(self.call(&name, self.vf32, &[self.vf32, self.vf32], &[a, b])));
            }
            I::V_FMA_F32 => {
                let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.vabsneg_f32(self.vsrc_f32(&i.src1), i.abs, i.neg, 1);
                let c = self.vabsneg_f32(self.vsrc_f32(&i.src2), i.abs, i.neg, 2);
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(self.vfma32(a, b, c)));
            }
            I::V_FMAC_F32 => {
                let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.vabsneg_f32(self.vsrc_f32(&i.src1), i.abs, i.neg, 1);
                let c = self.vf32_of(self.ld_vgpr32(i.vdst as u32));
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(self.vfma32(a, b, c)));
            }
            I::V_SUB_NC_U32 => self.vop3_int(i, |c, a, b, _| llvm::core::LLVMBuildSub(c.b, a, b, c.n())),
            I::V_MUL_HI_U32 => {
                let a = self.zext64v(self.vsrc_u32(&i.src0));
                let b = self.zext64v(self.vsrc_u32(&i.src1));
                let prod = llvm::core::LLVMBuildMul(self.b, a, b, self.n());
                let hi = llvm::core::LLVMBuildLShr(self.b, prod, self.splat(self.ci64(32), self.vi64), self.n());
                self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildTrunc(self.b, hi, self.vi32, self.n()));
            }
            I::V_CVT_F32_U32 => {
                let f = llvm::core::LLVMBuildUIToFP(self.b, self.vsrc_u32(&i.src0), self.vf32, self.n());
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(f));
            }
            I::V_RCP_F32 | I::V_RCP_IFLAG_F32 => {
                let s = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0);
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(self.vfdiv(self.vcf32(1.0), s)));
            }
            I::V_RNDNE_F32 | I::V_FLOOR_F32 | I::V_CEIL_F32 | I::V_TRUNC_F32 => {
                let name = match i.op { I::V_RNDNE_F32 => "roundeven", I::V_FLOOR_F32 => "floor", I::V_CEIL_F32 => "ceil", _ => "trunc" };
                let s = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0);
                let v = self.call(&self.vfn32(name), self.vf32, &[self.vf32], &[s]);
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(v));
            }
            I::V_S_RCP_F32 => {
                // Scalar reciprocal: uniform SGPR source and SGPR destination.
                let s = llvm::core::LLVMBuildBitCast(self.b, self.ssrc_u32(&i.src0), self.f32t, self.n());
                let one = llvm::core::LLVMConstReal(self.f32t, 1.0);
                let v = llvm::core::LLVMBuildFDiv(self.b, one, s, self.n());
                self.st_sgpr32(i.vdst as u32, llvm::core::LLVMBuildBitCast(self.b, v, self.i32t, self.n()));
            }
            I::V_DIV_FIXUP_F32 => {
                let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.vabsneg_f32(self.vsrc_f32(&i.src1), i.abs, i.neg, 1);
                let c = self.vabsneg_f32(self.vsrc_f32(&i.src2), i.abs, i.neg, 2);
                let r = self.per_lane("scalar_div_fixup_f32", self.f32t, &[a, b, c], &[self.f32t, self.f32t, self.f32t], self.vf32);
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(r));
            }
            I::V_DIV_FMAS_F32 => {
                let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.vabsneg_f32(self.vsrc_f32(&i.src1), i.abs, i.neg, 1);
                let c = self.vabsneg_f32(self.vsrc_f32(&i.src2), i.abs, i.neg, 2);
                let fma = self.vfma32(a, b, c);
                let scaled = self.vfmul(self.vcf32_bits(0x4F80_0000), fma);
                let cond = self.vcc_vec();
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(llvm::core::LLVMBuildSelect(self.b, cond, scaled, fma, self.n())));
            }
            I::V_LDEXP_F32 => {
                let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0);
                let e = self.vsrc_u32(&i.src1);
                let r = self.call(&format!("llvm.ldexp.v{}f32.v{}i32", self.w, self.w), self.vf32, &[self.vf32, self.vi32], &[a, e]);
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(r));
            }
            op if f32_pred(op).is_some() => {
                let (pred, invert) = f32_pred(op).unwrap();
                let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.vabsneg_f32(self.vsrc_f32(&i.src1), i.abs, i.neg, 1);
                let mut c = llvm::core::LLVMBuildFCmp(self.b, pred, a, b, self.n());
                if invert { c = llvm::core::LLVMBuildNot(self.b, c, self.n()); }
                self.st_cmp(i.vdst as u32, c);
            }
            I::V_CMP_CLASS_F32 => {
                let a = self.vsrc_f32(&i.src0);
                let s = match &i.src1 {
                    SourceOperand::LiteralConstant(v) => self.ci32(*v),
                    SourceOperand::IntegerConstant(v) => self.ci32(*v as u32),
                    SourceOperand::ScalarRegister(r) => self.ld_sgpr32(*r as u32),
                    _ => llvm::core::LLVMBuildExtractElement(self.b, self.vsrc_u32(&i.src1), self.ci32(0), self.n()),
                };
                let c = self.call(&format!("llvm.is.fpclass.v{}f32", self.w), self.vi1, &[self.vf32, self.i32t], &[a, s]);
                self.st_cmp(i.vdst as u32, c);
            }
            op if int64_pred(op).is_some() => {
                let pred = int64_pred(op).unwrap();
                let a = self.vsrc_u64(&i.src0); let b = self.vsrc_u64(&i.src1);
                let c = llvm::core::LLVMBuildICmp(self.b, pred, a, b, self.n());
                self.st_cmp(i.vdst as u32, c);
            }
            I::V_ADD_F64 => { let a = self.vabsneg_f64(self.vsrc_f64(&i.src0), i.abs, i.neg, 0); let b = self.vabsneg_f64(self.vsrc_f64(&i.src1), i.abs, i.neg, 1); let r = self.vfadd(a, b); self.st_vgpr_f64(i.vdst as u32, r); }
            I::V_MUL_F64 => { let a = self.vabsneg_f64(self.vsrc_f64(&i.src0), i.abs, i.neg, 0); let b = self.vabsneg_f64(self.vsrc_f64(&i.src1), i.abs, i.neg, 1); let r = self.vfmul(a, b); self.st_vgpr_f64(i.vdst as u32, r); }
            I::V_FMA_F64 => { let a = self.vabsneg_f64(self.vsrc_f64(&i.src0), i.abs, i.neg, 0); let b = self.vabsneg_f64(self.vsrc_f64(&i.src1), i.abs, i.neg, 1); let c = self.vabsneg_f64(self.vsrc_f64(&i.src2), i.abs, i.neg, 2); let r = self.vfmuladd(a, b, c); self.st_vgpr_f64(i.vdst as u32, r); }
            I::V_CVT_F32_F16 => {
                // OPSEL[0] chooses the high/low half of each packed source lane.
                let mut bits = self.vsrc_u32(&i.src0);
                if i.opsel & 1 != 0 {
                    bits = llvm::core::LLVMBuildLShr(self.b, bits, self.vci32(16), self.n());
                }
                let i16t = llvm::core::LLVMInt16TypeInContext(self.ctx);
                let vi16 = llvm::core::LLVMVectorType(i16t, self.w);
                let vf16 = llvm::core::LLVMVectorType(llvm::core::LLVMHalfTypeInContext(self.ctx), self.w);
                let bits = llvm::core::LLVMBuildTrunc(self.b, bits, vi16, self.n());
                let half = llvm::core::LLVMBuildBitCast(self.b, bits, vf16, self.n());
                let wide = llvm::core::LLVMBuildFPExt(self.b, half, self.vf32, self.n());
                let r = self.vabsneg_f32(wide, i.abs, i.neg, 0);
                self.st_vgpr32(i.vdst as u32, self.vf32_bits(r));
            }
            I::V_MAX_NUM_F64 => { let a = self.vabsneg_f64(self.vsrc_f64(&i.src0), i.abs, i.neg, 0); let b = self.vabsneg_f64(self.vsrc_f64(&i.src1), i.abs, i.neg, 1); let r = self.call(&self.vfn("maxnum"), self.vf64, &[self.vf64, self.vf64], &[a, b]); let r = self.vclamp_f64(r, i.cm); self.st_vgpr_f64(i.vdst as u32, r); }
            I::V_LDEXP_F64 => {
                let a = self.vsrc_f64(&i.src0);
                let e = self.vsrc_u32(&i.src1);
                let r = if self.ldexp_normal_pow2.get() {
                    self.vldexp_normal_pow2(a, e)
                } else {
                    self.vldexp(a, e)
                };
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            I::V_DIV_SCALE_F64 => { let a = self.vabsneg_f64(self.vsrc_f64(&i.src0), i.abs, i.neg, 0); let b = self.vabsneg_f64(self.vsrc_f64(&i.src1), i.abs, i.neg, 1); let c = self.vabsneg_f64(self.vsrc_f64(&i.src2), i.abs, i.neg, 2); let r = self.vfdiv(self.vfmul(a, c), b); self.st_vgpr_f64(i.vdst as u32, r); }
            I::V_DIV_FIXUP_F64 => { let b = self.vabsneg_f64(self.vsrc_f64(&i.src1), i.abs, i.neg, 1); let c = self.vabsneg_f64(self.vsrc_f64(&i.src2), i.abs, i.neg, 2); let r = self.vfdiv(c, b); self.st_vgpr_f64(i.vdst as u32, r); }
            I::V_DIV_FMAS_F64 => {
                let a = self.vsrc_f64(&i.src0); let b = self.vsrc_f64(&i.src1); let c = self.vsrc_f64(&i.src2);
                let fma = self.vfmuladd(a, b, c);
                let scaled = self.vfmul(fma, self.vcf64(f64::from_bits(0x43F0000000000000)));
                let cond = self.vcc_vec();
                let r = llvm::core::LLVMBuildSelect(self.b, cond, scaled, fma, self.n()); self.st_vgpr_f64(i.vdst as u32, r);
            }
            I::V_TRIG_PREOP_F64 => {
                // Hot in trig-heavy kernels (smallpt: ~15% of W=16 samples as the
                // per-lane runtime call). Default: fully vectorized in IR — the
                // Payne–Hanek segment extraction is pure integer bit manipulation
                // plus a final scalbn, so lane results stay bit-exact with the
                // scalar helper.
                let a = self.vsrc_f64(&i.src0); let s = self.vsrc_u32(&i.src1);
                let r = self.v_trig_preop(a, s);
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            op if f64_pred(op).is_some() => {
                let (pred, invert) = f64_pred(op).unwrap();
                let a = self.vabsneg_f64(self.vsrc_f64(&i.src0), i.abs, i.neg, 0);
                let b = self.vabsneg_f64(self.vsrc_f64(&i.src1), i.abs, i.neg, 1);
                let mut c = llvm::core::LLVMBuildFCmp(self.b, pred, a, b, self.n());
                if invert { c = llvm::core::LLVMBuildNot(self.b, c, self.n()); }
                self.st_cmp(i.vdst as u32, c);
            }
            I::V_CMP_CLASS_F64 => {
                // llvm.is.fpclass's test mask is a scalar i32 immarg (uniform).
                let a = self.vsrc_f64(&i.src0);
                let s = match &i.src1 {
                    SourceOperand::LiteralConstant(v) => self.ci32(*v),
                    SourceOperand::IntegerConstant(v) => self.ci32(*v as u32),
                    SourceOperand::ScalarRegister(r) => self.ld_sgpr32(*r as u32),
                    _ => llvm::core::LLVMBuildExtractElement(self.b, self.vsrc_u32(&i.src1), self.ci32(0), self.n()),
                };
                let name = format!("llvm.is.fpclass.v{}f64", self.w);
                let c = self.call(&name, self.vi1, &[self.vf64, self.i32t], &[a, s]);
                self.st_cmp(i.vdst as u32, c);
            }
            op if int_pred(op).is_some() => {
                let pred = int_pred(op).unwrap();
                let a = self.vsrc_u32(&i.src0); let b = self.vsrc_u32(&i.src1);
                let c = llvm::core::LLVMBuildICmp(self.b, pred, a, b, self.n());
                self.st_cmp(i.vdst as u32, c);
            }
            // ----- uniform cross-lane spill idiom (constant lane) -----
            // The value is a uniform SGPR (scalar), so "lane K of vD" is identical
            // across the W packed lanes; model it as a scalar slot keyed by (vD, K).
            // Mirrors the scalar path; a non-constant lane or VGPR source would be
            // genuine cross-lane and is rejected here.
            I::V_WRITELANE_B32 => {
                let lane = lane_const(&i.src1)
                    .expect("vec: v_writelane_b32 needs a constant lane (non-uniform cross-lane unsupported)");
                let val = self.ssrc_u32(&i.src0);
                let slot = self.spill_slot_ptr(i.vdst as u32, lane);
                llvm::core::LLVMBuildStore(self.b, val, slot);
            }
            I::V_READLANE_B32 => {
                let lane = lane_const(&i.src1)
                    .expect("vec: v_readlane_b32 needs a constant lane");
                let src = vreg_of(&i.src0)
                    .expect("vec: v_readlane_b32 source must be a VGPR");
                let slot = self.spill_slot_ptr(src, lane);
                let v = llvm::core::LLVMBuildLoad2(self.b, self.i32t, slot, self.n());
                self.st_sgpr32(i.vdst as u32, v);
            }
            _ => panic!("vec: unsupported VOP3 {:?}", i.op),
        }
    }

    unsafe fn emit_vop3p(&self, i: &VOP3P) {
        match i.op {
            I::V_FMA_MIXLO_F16 => {
                let opsel_hi = i.opsel_hi | (i.opsel_hi2 << 2);
                let src = |op: &SourceOperand, idx: u32| -> LLVMValueRef {
                    let value = if (opsel_hi >> idx) & 1 == 0 {
                        self.vsrc_f32(op)
                    } else {
                        self.vsrc_f16_f32(op, (i.opsel >> idx) & 1 != 0)
                    };
                    self.vabsneg_f32(value, i.neg_hi, i.neg, idx)
                };
                let value = self.vfma32(src(&i.src0, 0), src(&i.src1, 1), src(&i.src2, 2));
                let lo = self.vf32_to_f16_bits(value);
                let hi = self.v_and(self.ld_vgpr32(i.vdst as u32), self.vci32(0xffff_0000));
                self.st_vgpr32(i.vdst as u32, self.v_or(hi, lo));
            }
            _ => panic!("vec: unsupported VOP3P {:?}", i.op),
        }
    }

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

    unsafe fn emit_vop3sd(&self, i: &VOP3SD) {
        match i.op {
            I::V_ADD_CO_U32 => {
                let s0 = self.zext64v(self.vsrc_u32(&i.src0));
                let s1 = self.zext64v(self.vsrc_u32(&i.src1));
                let sum = self.v_add(s0, s1);
                let lo = llvm::core::LLVMBuildTrunc(self.b, sum, self.vi32, self.n());
                self.st_vgpr32(i.vdst as u32, lo);
                let cout = llvm::core::LLVMBuildLShr(self.b, sum, self.splat(self.ci64(32), self.vi64), self.n());
                let cout = llvm::core::LLVMBuildTrunc(self.b, cout, self.vi32, self.n());
                let cout = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, cout, self.vci32(0), self.n());
                self.st_sdst_mask(i.sdst, cout);
            }
            I::V_ADD_CO_CI_U32 => {
                let s0 = self.zext64v(self.vsrc_u32(&i.src0));
                let s1 = self.zext64v(self.vsrc_u32(&i.src1));
                let cin = self.zext64v(self.v_and(self.src_mask_as_u32(&i.src2), self.vci32(1)));
                let sum = self.v_add(self.v_add(s0, s1), cin);
                let lo = llvm::core::LLVMBuildTrunc(self.b, sum, self.vi32, self.n());
                self.st_vgpr32(i.vdst as u32, lo);
                let cout = llvm::core::LLVMBuildLShr(self.b, sum, self.splat(self.ci64(32), self.vi64), self.n());
                let cout = llvm::core::LLVMBuildTrunc(self.b, cout, self.vi32, self.n());
                let cout = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, cout, self.vci32(0), self.n());
                self.st_sdst_mask(i.sdst, cout);
            }
            I::V_MAD_CO_U64_U32 => {
                let s0 = self.zext64v(self.vsrc_u32(&i.src0));
                let s1 = self.zext64v(self.vsrc_u32(&i.src1));
                let s2 = self.vsrc_u64(&i.src2);
                let prod = llvm::core::LLVMBuildMul(self.b, s0, s1, self.n());
                let d = self.v_add(prod, s2);
                // carry = d < prod (unsigned wrap)
                let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntULT, d, prod, self.n());
                self.st_vgpr64(i.vdst as u32, d);
                self.st_sdst_mask(i.sdst, c);
            }
            I::V_DIV_SCALE_F64 => {
                let a = self.vabsneg_f64(self.vsrc_f64(&i.src0), 0, i.neg, 0);
                let b = self.vabsneg_f64(self.vsrc_f64(&i.src1), 0, i.neg, 1);
                let c = self.vabsneg_f64(self.vsrc_f64(&i.src2), 0, i.neg, 2);
                let r = self.vfdiv(self.vfmul(a, c), b);
                self.st_vgpr_f64(i.vdst as u32, r);
                let zero = llvm::core::LLVMConstNull(self.vi1);
                self.st_sdst_mask(i.sdst, zero);
            }
            I::V_DIV_SCALE_F32 => {
                // Per-lane scalar_div_scale_f32 returns u64: value bits | flag<<32.
                let a = self.vabsneg_f32(self.vsrc_f32(&i.src0), 0, i.neg, 0);
                let b = self.vabsneg_f32(self.vsrc_f32(&i.src1), 0, i.neg, 1);
                let c = self.vabsneg_f32(self.vsrc_f32(&i.src2), 0, i.neg, 2);
                let packed = self.per_lane("scalar_div_scale_f32", self.i64t, &[a, b, c], &[self.f32t, self.f32t, self.f32t], self.vi64);
                let val = llvm::core::LLVMBuildTrunc(self.b, packed, self.vi32, self.n());
                self.st_vgpr32(i.vdst as u32, val);
                let flag = llvm::core::LLVMBuildTrunc(self.b, llvm::core::LLVMBuildLShr(self.b, packed, self.splat(self.ci64(32), self.vi64), self.n()), self.vi32, self.n());
                let flag = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, flag, self.vci32(0), self.n());
                self.st_sdst_mask(i.sdst, flag);
            }
            _ => panic!("vec: unsupported VOP3SD {:?}", i.op),
        }
    }
    unsafe fn src_mask_as_u32(&self, op: &SourceOperand) -> LLVMValueRef {
        // a lane-mask SGPR source presented as <W×i32> (0/1 per lane from its bits)
        let v = self.src_mask_vec(op);
        llvm::core::LLVMBuildZExt(self.b, v, self.vi32, self.n())
    }

    unsafe fn vop3_int<F: Fn(&Cg, LLVMValueRef, LLVMValueRef, LLVMValueRef) -> LLVMValueRef>(&self, i: &VOP3, f: F) {
        let a = self.vsrc_u32(&i.src0);
        let b = self.vsrc_u32(&i.src1);
        let c = self.vsrc_u32(&i.src2);
        let r = f(self, a, b, c);
        self.st_vgpr32(i.vdst as u32, r);
    }

    unsafe fn emit_vopc(&self, i: &VOPC) {
        let is_cmpx = format!("{:?}", i.op).starts_with("V_CMPX");
        let dest = if is_cmpx { EXEC } else { VCC };
        if let Some((pred, invert)) = f32_pred(i.op) {
            let a = self.vsrc_f32(&i.src0); let b = self.vf32_of(self.ld_vgpr32(i.vsrc1 as u32));
            let mut c = llvm::core::LLVMBuildFCmp(self.b, pred, a, b, self.n());
            if invert { c = llvm::core::LLVMBuildNot(self.b, c, self.n()); }
            self.st_cmp(dest, c);
        } else if let Some((pred, invert)) = f64_pred(i.op) {
            let a = self.vsrc_f64(&i.src0); let b = self.ld_vgpr_f64(i.vsrc1 as u32);
            let mut c = llvm::core::LLVMBuildFCmp(self.b, pred, a, b, self.n());
            if invert { c = llvm::core::LLVMBuildNot(self.b, c, self.n()); }
            self.st_cmp(dest, c);
        } else if let Some(pred) = int64_pred(i.op) {
            let a = self.vsrc_u64(&i.src0); let b = self.ld_vgpr64(i.vsrc1 as u32);
            let c = llvm::core::LLVMBuildICmp(self.b, pred, a, b, self.n());
            self.st_cmp(dest, c);
        } else if let Some(pred) = int_pred(i.op) {
            let a = self.vsrc_u32(&i.src0); let b = self.ld_vgpr32(i.vsrc1 as u32);
            let c = llvm::core::LLVMBuildICmp(self.b, pred, a, b, self.n());
            self.st_cmp(dest, c);
        } else {
            panic!("vec: unsupported VOPC {:?}", i.op);
        }
    }

    unsafe fn emit_vopd(&self, i: &VOPD) {
        let dx = i.vdstx as u32;
        let dy = ((i.vdsty as u32) << 1) | ((dx & 1) ^ 1);
        let rx = self.eval_vopd_half(i.opx, &i.src0x, i.vsrc1x, dx, i.literal_constant);
        let ry = self.eval_vopd_half(i.opy, &i.src0y, i.vsrc1y, dy, i.literal_constant);
        self.st_vgpr32(dx, rx);
        self.st_vgpr32(dy, ry);
    }
    unsafe fn eval_vopd_half(&self, op: I, src0: &SourceOperand, vsrc1: u8, dst: u32, lit: Option<u32>) -> LLVMValueRef {
        let s0 = self.vsrc_u32(src0);
        let f = |cg: &Cg, r: u32| unsafe { cg.vf32_of(cg.ld_vgpr32(r)) };
        match op {
            I::V_DUAL_MOV_B32 => s0,
            I::V_DUAL_AND_B32 => self.v_and(s0, self.ld_vgpr32(vsrc1 as u32)),
            I::V_DUAL_ADD_NC_U32 => self.v_add(s0, self.ld_vgpr32(vsrc1 as u32)),
            I::V_DUAL_LSHLREV_B32 => self.v_shl(self.ld_vgpr32(vsrc1 as u32), s0),
            I::V_DUAL_CNDMASK_B32 => { let s1 = self.ld_vgpr32(vsrc1 as u32); let c = self.vcc_vec(); llvm::core::LLVMBuildSelect(self.b, c, s1, s0, self.n()) }
            I::V_DUAL_MUL_F32 => self.vf32_bits(self.vfmul(self.vsrc_f32(src0), f(self, vsrc1 as u32))),
            I::V_DUAL_ADD_F32 => self.vf32_bits(self.vfadd(self.vsrc_f32(src0), f(self, vsrc1 as u32))),
            I::V_DUAL_SUB_F32 => self.vf32_bits(self.vfsub(self.vsrc_f32(src0), f(self, vsrc1 as u32))),
            I::V_DUAL_SUBREV_F32 => self.vf32_bits(self.vfsub(f(self, vsrc1 as u32), self.vsrc_f32(src0))),
            I::V_DUAL_FMAC_F32 => self.vf32_bits(self.vfma32(self.vsrc_f32(src0), f(self, vsrc1 as u32), f(self, dst))),
            I::V_DUAL_FMAMK_F32 => { let k = self.vcf32_bits(lit.unwrap()); self.vf32_bits(self.vfma32(self.vsrc_f32(src0), k, f(self, vsrc1 as u32))) }
            I::V_DUAL_FMAAK_F32 => { let k = self.vcf32_bits(lit.unwrap()); self.vf32_bits(self.vfma32(self.vsrc_f32(src0), f(self, vsrc1 as u32), k)) }
            _ => panic!("vec: unsupported VOPD half {:?}", op),
        }
    }

    // ---- SALU (scalar, uniform) -----------------------------------------
    unsafe fn emit_sop1(&self, i: &SOP1) {
        if self.has_structured_mask(i.sdst as u32) {
            match i.op {
                I::S_MOV_B32 => {
                    if self.store_structured_mask(i.sdst as u32, self.mask_src(&i.ssrc0)) { return; }
                }
                I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 | I::S_OR_SAVEEXEC_B32 | I::S_XOR_SAVEEXEC_B32 => {
                    let old = self.exec_vec();
                    self.store_structured_mask(i.sdst as u32, old);
                    let rhs = self.mask_src(&i.ssrc0);
                    let next = match i.op {
                        I::S_AND_SAVEEXEC_B32 => llvm::core::LLVMBuildAnd(self.b, rhs, old, self.n()),
                        I::S_AND_NOT1_SAVEEXEC_B32 => llvm::core::LLVMBuildAnd(self.b, rhs, llvm::core::LLVMBuildNot(self.b, old, self.n()), self.n()),
                        I::S_OR_SAVEEXEC_B32 => llvm::core::LLVMBuildOr(self.b, rhs, old, self.n()),
                        I::S_XOR_SAVEEXEC_B32 => llvm::core::LLVMBuildXor(self.b, rhs, old, self.n()),
                        _ => unreachable!(),
                    };
                    self.store_structured_mask(EXEC, next);
                    self.st_scc(self.mask_any(next));
                    return;
                }
                _ => {}
            }
        }
        match i.op {
            I::S_MOV_B32 => {
                let v = self.ssrc_u32(&i.ssrc0);
                self.st_sgpr32(i.sdst as u32, v);
            }
            I::S_MOV_B64 => { let v = self.ssrc_u64(&i.ssrc0); self.st_sgpr64(i.sdst as u32, v); }
            I::S_AND_SAVEEXEC_B32 => {
                let s0 = self.ssrc_u32(&i.ssrc0); let old = self.ld_sgpr32(EXEC);
                self.st_sgpr32(i.sdst as u32, old);
                let ne = llvm::core::LLVMBuildAnd(self.b, s0, old, self.n());
                self.st_sgpr32(EXEC, ne); self.st_scc_nz(ne);
            }
            I::S_AND_NOT1_SAVEEXEC_B32 => {
                let s0 = self.ssrc_u32(&i.ssrc0); let old = self.ld_sgpr32(EXEC);
                self.st_sgpr32(i.sdst as u32, old);
                let ne = llvm::core::LLVMBuildAnd(self.b, s0, llvm::core::LLVMBuildNot(self.b, old, self.n()), self.n());
                self.st_sgpr32(EXEC, ne); self.st_scc_nz(ne);
            }
            I::S_OR_SAVEEXEC_B32 => {
                let s0 = self.ssrc_u32(&i.ssrc0); let old = self.ld_sgpr32(EXEC);
                self.st_sgpr32(i.sdst as u32, old);
                let ne = llvm::core::LLVMBuildOr(self.b, s0, old, self.n());
                self.st_sgpr32(EXEC, ne); self.st_scc_nz(ne);
            }
            I::S_CTZ_I32_B32 => {
                let x = self.ssrc_u32(&i.ssrc0);
                let tz = self.call("llvm.cttz.i32", self.i32t, &[self.i32t, self.i1], &[x, llvm::core::LLVMConstInt(self.i1, 0, 0)]);
                let is0 = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntEQ, x, self.ci32(0), self.n());
                let r = llvm::core::LLVMBuildSelect(self.b, is0, self.ci32(0xFFFF_FFFF), tz, self.n());
                self.st_sgpr32(i.sdst as u32, r);
            }
            I::S_CVT_F32_I32 => { let f = llvm::core::LLVMBuildSIToFP(self.b, self.ssrc_u32(&i.ssrc0), self.f32t, self.n()); self.st_sgpr32(i.sdst as u32, llvm::core::LLVMBuildBitCast(self.b, f, self.i32t, self.n())); }
            I::S_CVT_F32_U32 => { let f = llvm::core::LLVMBuildUIToFP(self.b, self.ssrc_u32(&i.ssrc0), self.f32t, self.n()); self.st_sgpr32(i.sdst as u32, llvm::core::LLVMBuildBitCast(self.b, f, self.i32t, self.n())); }
            I::S_CVT_I32_F32 => { let f = llvm::core::LLVMBuildBitCast(self.b, self.ssrc_u32(&i.ssrc0), self.f32t, self.n()); self.st_sgpr32(i.sdst as u32, self.call("llvm.fptosi.sat.i32.f32", self.i32t, &[self.f32t], &[f])); }
            I::S_CVT_U32_F32 => { let f = llvm::core::LLVMBuildBitCast(self.b, self.ssrc_u32(&i.ssrc0), self.f32t, self.n()); self.st_sgpr32(i.sdst as u32, self.call("llvm.fptoui.sat.i32.f32", self.i32t, &[self.f32t], &[f])); }
            _ => panic!("vec: unsupported SOP1 {:?}", i.op),
        }
    }
    unsafe fn emit_sop2(&self, i: &SOP2) {
        use llvm::core::*;
        if self.has_structured_mask(i.sdst as u32)
            && matches!(i.op, I::S_AND_B32 | I::S_OR_B32 | I::S_XOR_B32 | I::S_AND_NOT1_B32 | I::S_OR_NOT1_B32)
        {
            let a = self.mask_src(&i.ssrc0);
            let b = self.mask_src(&i.ssrc1);
            let value = match i.op {
                I::S_AND_B32 => LLVMBuildAnd(self.b, a, b, self.n()),
                I::S_OR_B32 => LLVMBuildOr(self.b, a, b, self.n()),
                I::S_XOR_B32 => LLVMBuildXor(self.b, a, b, self.n()),
                I::S_AND_NOT1_B32 => LLVMBuildAnd(self.b, a, LLVMBuildNot(self.b, b, self.n()), self.n()),
                I::S_OR_NOT1_B32 => LLVMBuildOr(self.b, a, LLVMBuildNot(self.b, b, self.n()), self.n()),
                _ => unreachable!(),
            };
            self.store_structured_mask(i.sdst as u32, value);
            self.st_scc(self.mask_any(value));
            return;
        }
        match i.op {
            I::S_ADD_U32 => {
                // Unsigned add: SCC = carry-out (ISA §S_ADD_U32).
                let a = self.zext64s(self.ssrc_u32(&i.ssrc0)); let b = self.zext64s(self.ssrc_u32(&i.ssrc1));
                let sum = LLVMBuildAdd(self.b, a, b, self.n());
                let lo = LLVMBuildTrunc(self.b, sum, self.i32t, self.n());
                self.st_sgpr32(i.sdst as u32, lo);
                let cout = LLVMBuildTrunc(self.b, LLVMBuildLShr(self.b, sum, self.ci64(32), self.n()), self.i32t, self.n());
                self.st_scc_nz(cout);
            }
            I::S_ADD_CO_I32 | I::S_ADD_I32 => {
                // Signed add: SCC = signed OVERFLOW (ISA §S_ADD_CO_I32), not carry.
                let a = self.ssrc_u32(&i.ssrc0); let b = self.ssrc_u32(&i.ssrc1);
                let ov = self.call("llvm.sadd.with.overflow.i32",
                    LLVMStructTypeInContext(self.ctx, [self.i32t, self.i1].as_ptr() as *mut _, 2, 0),
                    &[self.i32t, self.i32t], &[a, b]);
                let r = LLVMBuildExtractValue(self.b, ov, 0, self.n());
                let o = LLVMBuildExtractValue(self.b, ov, 1, self.n());
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc(o);
            }
            I::S_ADD_NC_U64 => { let a = self.ssrc_u64(&i.ssrc0); let b = self.ssrc_u64(&i.ssrc1); let r = LLVMBuildAdd(self.b, a, b, self.n()); self.st_sgpr64(i.sdst as u32, r); }
            I::S_AND_B32 => self.sop2_logic(i, |c, a, b| LLVMBuildAnd(c.b, a, b, c.n())),
            I::S_OR_B32 => self.sop2_logic(i, |c, a, b| LLVMBuildOr(c.b, a, b, c.n())),
            I::S_XOR_B32 => self.sop2_logic(i, |c, a, b| LLVMBuildXor(c.b, a, b, c.n())),
            I::S_AND_NOT1_B32 => self.sop2_logic(i, |c, a, b| LLVMBuildAnd(c.b, a, LLVMBuildNot(c.b, b, c.n()), c.n())),
            I::S_OR_NOT1_B32 => self.sop2_logic(i, |c, a, b| LLVMBuildOr(c.b, a, LLVMBuildNot(c.b, b, c.n()), c.n())),
            I::S_LSHR_B32 => self.sop2_logic(i, |c, a, b| { let amt = LLVMBuildAnd(c.b, b, c.ci32(31), c.n()); LLVMBuildLShr(c.b, a, amt, c.n()) }),
            I::S_LSHL_B32 => self.sop2_logic(i, |c, a, b| { let amt = LLVMBuildAnd(c.b, b, c.ci32(31), c.n()); LLVMBuildShl(c.b, a, amt, c.n()) }),
            I::S_CSELECT_B32 => { let a = self.ssrc_u32(&i.ssrc0); let b = self.ssrc_u32(&i.ssrc1); let c = self.ld_scc(); let r = LLVMBuildSelect(self.b, c, a, b, self.n()); self.st_sgpr32(i.sdst as u32, r); }
            I::S_SUB_CO_I32 => {
                // Signed sub: SCC = signed overflow (ISA §S_SUB_CO_I32).
                let a = self.ssrc_u32(&i.ssrc0); let b = self.ssrc_u32(&i.ssrc1);
                let ov = self.call("llvm.ssub.with.overflow.i32",
                    LLVMStructTypeInContext(self.ctx, [self.i32t, self.i1].as_ptr() as *mut _, 2, 0),
                    &[self.i32t, self.i32t], &[a, b]);
                let r = LLVMBuildExtractValue(self.b, ov, 0, self.n());
                let o = LLVMBuildExtractValue(self.b, ov, 1, self.n());
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc(o);
            }
            I::S_MUL_I32 => {
                let r = LLVMBuildMul(self.b, self.ssrc_u32(&i.ssrc0), self.ssrc_u32(&i.ssrc1), self.n());
                self.st_sgpr32(i.sdst as u32, r);
            }
            I::S_MUL_HI_U32 => {
                let a = self.zext64s(self.ssrc_u32(&i.ssrc0)); let b = self.zext64s(self.ssrc_u32(&i.ssrc1));
                let hi = LLVMBuildLShr(self.b, LLVMBuildMul(self.b, a, b, self.n()), self.ci64(32), self.n());
                self.st_sgpr32(i.sdst as u32, LLVMBuildTrunc(self.b, hi, self.i32t, self.n()));
            }
            I::S_LSHL_B64 => {
                let a = self.ssrc_u64(&i.ssrc0);
                let amt = LLVMBuildAnd(self.b, self.ssrc_u64(&i.ssrc1), self.ci64(63), self.n());
                let r = LLVMBuildShl(self.b, a, amt, self.n());
                self.st_sgpr64(i.sdst as u32, r);
                self.st_scc(LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, r, self.ci64(0), self.n()));
            }
            I::S_BFE_U32 => {
                let data = self.ssrc_u32(&i.ssrc0); let control = self.ssrc_u32(&i.ssrc1);
                let offset = LLVMBuildAnd(self.b, control, self.ci32(0x1f), self.n());
                let width = LLVMBuildAnd(self.b, LLVMBuildLShr(self.b, control, self.ci32(16), self.n()), self.ci32(0x7f), self.n());
                let shifted = LLVMBuildLShr(self.b, data, offset, self.n());
                let mask = LLVMBuildSub(self.b, LLVMBuildShl(self.b, self.ci32(1), width, self.n()), self.ci32(1), self.n());
                let r = LLVMBuildAnd(self.b, shifted, mask, self.n());
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc_nz(r);
            }
            _ => panic!("vec: unsupported SOP2 {:?}", i.op),
        }
    }
    unsafe fn emit_sopk(&self, i: &SOPK) {
        use llvm::LLVMIntPredicate::*;
        let imm = self.ci32(i.simm16 as i16 as i32 as u32);
        match i.op {
            I::S_MOVK_I32 => self.st_sgpr32(i.sdst as u32, imm),
            I::S_CMOVK_I32 => {
                let scc = self.ld_scc();
                let old = self.ld_sgpr32(i.sdst as u32);
                let nv = llvm::core::LLVMBuildSelect(self.b, scc, imm, old, self.n());
                self.st_sgpr32(i.sdst as u32, nv);
            }
            I::S_ADDK_I32 => {
                let a = self.ld_sgpr32(i.sdst as u32);
                let sum = self.call(
                    "llvm.sadd.with.overflow.i32",
                    llvm::core::LLVMStructTypeInContext(self.ctx, [self.i32t, self.i1].as_ptr() as *mut _, 2, 0),
                    &[self.i32t, self.i32t], &[a, imm],
                );
                let r = llvm::core::LLVMBuildExtractValue(self.b, sum, 0, self.n());
                let ov = llvm::core::LLVMBuildExtractValue(self.b, sum, 1, self.n());
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc(ov);
            }
            I::S_MULK_I32 => {
                let a = self.ld_sgpr32(i.sdst as u32);
                self.st_sgpr32(i.sdst as u32, llvm::core::LLVMBuildMul(self.b, a, imm, self.n()));
            }
            I::S_CMPK_EQ_I32 | I::S_CMPK_LG_I32 | I::S_CMPK_GT_I32 | I::S_CMPK_GE_I32
            | I::S_CMPK_LT_I32 | I::S_CMPK_LE_I32 | I::S_CMPK_EQ_U32 | I::S_CMPK_LG_U32
            | I::S_CMPK_GT_U32 | I::S_CMPK_GE_U32 | I::S_CMPK_LT_U32 | I::S_CMPK_LE_U32 => {
                let a = self.ld_sgpr32(i.sdst as u32);
                let p = match i.op {
                    I::S_CMPK_EQ_I32 => LLVMIntEQ, I::S_CMPK_LG_I32 => LLVMIntNE,
                    I::S_CMPK_GT_I32 => LLVMIntSGT, I::S_CMPK_GE_I32 => LLVMIntSGE,
                    I::S_CMPK_LT_I32 => LLVMIntSLT, I::S_CMPK_LE_I32 => LLVMIntSLE,
                    I::S_CMPK_EQ_U32 => LLVMIntEQ, I::S_CMPK_LG_U32 => LLVMIntNE,
                    I::S_CMPK_GT_U32 => LLVMIntUGT, I::S_CMPK_GE_U32 => LLVMIntUGE,
                    I::S_CMPK_LT_U32 => LLVMIntULT, _ => LLVMIntULE,
                };
                let c = llvm::core::LLVMBuildICmp(self.b, p, a, imm, self.n());
                self.st_scc(c);
            }
            _ => panic!("vec: unsupported SOPK {:?}", i.op),
        }
    }
    unsafe fn sop2_logic<F: Fn(&Cg, LLVMValueRef, LLVMValueRef) -> LLVMValueRef>(&self, i: &SOP2, f: F) {
        let a = self.ssrc_u32(&i.ssrc0); let b = self.ssrc_u32(&i.ssrc1);
        let r = f(self, a, b);
        self.st_sgpr32(i.sdst as u32, r); self.st_scc_nz(r);
    }
    unsafe fn emit_sopc(&self, i: &crate::rdna_instructions::SOPC) {
        use llvm::LLVMIntPredicate::*;
        match i.op {
            I::S_CMP_EQ_U32 | I::S_CMP_LG_U32 | I::S_CMP_GT_U32 | I::S_CMP_LT_U32 | I::S_CMP_GE_U32 | I::S_CMP_LE_U32 => {
                let a = self.ssrc_u32(&i.ssrc0); let b = self.ssrc_u32(&i.ssrc1);
                let p = match i.op { I::S_CMP_EQ_U32 => LLVMIntEQ, I::S_CMP_LG_U32 => LLVMIntNE, I::S_CMP_GT_U32 => LLVMIntUGT, I::S_CMP_LT_U32 => LLVMIntULT, I::S_CMP_GE_U32 => LLVMIntUGE, I::S_CMP_LE_U32 => LLVMIntULE, _ => unreachable!() };
                let c = llvm::core::LLVMBuildICmp(self.b, p, a, b, self.n()); self.st_scc(c);
            }
            I::S_CMP_EQ_U64 | I::S_CMP_LG_U64 => {
                let a = self.ssrc_u64(&i.ssrc0); let b = self.ssrc_u64(&i.ssrc1);
                let p = if matches!(i.op, I::S_CMP_EQ_U64) { LLVMIntEQ } else { LLVMIntNE };
                let c = llvm::core::LLVMBuildICmp(self.b, p, a, b, self.n()); self.st_scc(c);
            }
            _ => panic!("vec: unsupported SOPC {:?}", i.op),
        }
    }
    unsafe fn emit_smem(&self, i: &SMEM) {
        let words = match i.op { I::S_LOAD_B32 => 1, I::S_LOAD_B64 => 2, I::S_LOAD_B96 => 3, I::S_LOAD_B128 => 4, I::S_LOAD_B256 => 8, I::S_LOAD_U16 => 1, _ => panic!("vec: unsupported SMEM {:?}", i.op) };
        let base = self.ld_sgpr64(i.sbase as u32 * 2);
        for k in 0..words {
            let a = llvm::core::LLVMBuildAdd(self.b, base, self.ci64((i.ioffset as u64) + (k as u64) * 4), self.n());
            let ptr = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, self.n());
            let d = if matches!(i.op, I::S_LOAD_U16) {
                let i16t = llvm::core::LLVMInt16TypeInContext(self.ctx);
                let v = llvm::core::LLVMBuildLoad2(self.b, i16t, ptr, self.n());
                llvm::core::LLVMBuildZExt(self.b, v, self.i32t, self.n())
            } else {
                llvm::core::LLVMBuildLoad2(self.b, self.i32t, ptr, self.n())
            };
            self.st_sgpr32(i.sdata as u32 + k, d);
        }
    }

    // ---- VGLOBAL (per-lane gather/scatter) ------------------------------
    unsafe fn emit_vglobal(&self, i: &VGLOBAL) {
        if matches!(i.op, I::GLOBAL_WB | I::GLOBAL_INV) {
            return;
        }
        let sext_ioffset = (((i.ioffset << 8) as i32) >> 8) as i64 as u64;
        // base address per lane (<W×i64>)
        let base = if i.saddr != 124 {
            let s = self.splat(self.ld_sgpr64(i.saddr as u32), self.vi64);
            let v = self.zext64v(self.ld_vgpr32(i.vaddr as u32));
            self.v_add(s, v)
        } else {
            self.ld_vgpr64(i.vaddr as u32)
        };
        let addr = self.v_add(base, self.splat(self.ci64(sext_ioffset), self.vi64));

        // Sub-dword global memory uses the same typed masked gather/scatter
        // intrinsics as FLAT memory, then extends/truncates at the VGPR boundary.
        if matches!(
            i.op,
            I::GLOBAL_LOAD_U8 | I::GLOBAL_LOAD_I8 | I::GLOBAL_LOAD_U16 | I::GLOBAL_LOAD_I16
        ) {
            let (elem, mnem, signed) = match i.op {
                I::GLOBAL_LOAD_U8 => (llvm::core::LLVMInt8TypeInContext(self.ctx), "i8", false),
                I::GLOBAL_LOAD_I8 => (llvm::core::LLVMInt8TypeInContext(self.ctx), "i8", true),
                I::GLOBAL_LOAD_U16 => (llvm::core::LLVMInt16TypeInContext(self.ctx), "i16", false),
                _ => (llvm::core::LLVMInt16TypeInContext(self.ctx), "i16", true),
            };
            let ptrs = self.ptr_at_vec(addr, 0);
            let value = self.masked_gather_ty(ptrs, self.exec_vec(), elem, mnem);
            let value = if signed {
                llvm::core::LLVMBuildSExt(self.b, value, self.vi32, self.n())
            } else {
                llvm::core::LLVMBuildZExt(self.b, value, self.vi32, self.n())
            };
            self.st_vgpr32(i.vdst as u32, value);
            return;
        }

        if matches!(i.op, I::GLOBAL_STORE_B8 | I::GLOBAL_STORE_B16) {
            let (elem, mnem) = if matches!(i.op, I::GLOBAL_STORE_B8) {
                (llvm::core::LLVMInt8TypeInContext(self.ctx), "i8")
            } else {
                (llvm::core::LLVMInt16TypeInContext(self.ctx), "i16")
            };
            let velem = llvm::core::LLVMVectorType(elem, self.w);
            let value = llvm::core::LLVMBuildTrunc(
                self.b,
                self.ld_vgpr32(i.vsrc as u32),
                velem,
                self.n(),
            );
            let ptrs = self.ptr_at_vec(addr, 0);
            self.masked_scatter_ty(value, ptrs, self.exec_vec(), elem, mnem);
            return;
        }

        if matches!(i.op, I::GLOBAL_ATOMIC_ADD_U32) {
            // Per-lane atomicrmw: each active lane adds its data to its own global
            // address (lanes may collide — e.g. histogram bins — so the atomics
            // serialize and accumulate correctly). Inactive lanes redirect to a
            // throwaway thread-local scratch slot and add 0, so neither a wild
            // pointer is dereferenced nor real memory perturbed.
            let exec = self.ld_sgpr32(EXEC);
            let ptrs = self.ptr_at_vec(addr, 0);
            let data = self.ld_vgpr32(i.vsrc as u32);
            let dummy = self.bvh_scratch;
            let mut result = llvm::core::LLVMGetPoison(self.vi32);
            for k in 0..self.w {
                let bit = llvm::core::LLVMBuildAnd(self.b, llvm::core::LLVMBuildLShr(self.b, exec, self.ci32(k), self.n()), self.ci32(1), self.n());
                let active = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, bit, self.ci32(0), self.n());
                let ptr_k = llvm::core::LLVMBuildExtractElement(self.b, ptrs, self.ci32(k), self.n());
                let ptr_use = llvm::core::LLVMBuildSelect(self.b, active, ptr_k, dummy, self.n());
                let data_k = llvm::core::LLVMBuildExtractElement(self.b, data, self.ci32(k), self.n());
                let data_use = llvm::core::LLVMBuildSelect(self.b, active, data_k, self.ci32(0), self.n());
                let old = llvm::core::LLVMBuildAtomicRMW(
                    self.b,
                    llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
                    ptr_use,
                    data_use,
                    llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                    0,
                );
                result = llvm::core::LLVMBuildInsertElement(self.b, result, old, self.ci32(k), self.n());
            }
            self.st_vgpr32(i.vdst as u32, result);
            return;
        }

        let (is_store, words) = match i.op {
            I::GLOBAL_LOAD_B32 => (false, 1), I::GLOBAL_LOAD_B64 => (false, 2), I::GLOBAL_LOAD_B96 => (false, 3), I::GLOBAL_LOAD_B128 => (false, 4),
            I::GLOBAL_STORE_B32 => (true, 1), I::GLOBAL_STORE_B64 => (true, 2), I::GLOBAL_STORE_B96 => (true, 3), I::GLOBAL_STORE_B128 => (true, 4),
            _ => panic!("vec: unsupported VGLOBAL {:?}", i.op),
        };
        let exec = self.exec_vec();

        if is_store {
            for k in 0..words {
                let ptrs = self.ptr_at_vec(addr, (k as u64) * 4);
                let d = self.ld_vgpr32(i.vsrc as u32 + k);
                self.masked_scatter(d, ptrs, exec);
            }
            return;
        }
        // Uniform-address load (address VGPR identical in all packed lanes) → one
        // scalar load + broadcast instead of a gather. Sound: the block only runs
        // when EXEC≠0 (some lane active), so the shared address is dereferenceable,
        // and all lanes get the same value. `noelide`-style: must check the actual
        // address-input regs are uniform.
        let uniform_addr = if i.saddr != 124 {
            self.vgpr_uniform(i.vaddr as u32) // base = uniform SGPR + uniform vaddr
        } else {
            self.vgpr_uniform(i.vaddr as u32) && self.vgpr_uniform(i.vaddr as u32 + 1)
        };
        // Coalesced affine-frame load: address = vgpr0*stride + uniform (a
        // per-work-item record). The packed lanes' records are contiguous in
        // *vgpr0* order, so a contiguous vector load + transpose (shufflevector)
        // extracts each field as <W×T> — replacing `words` gathers with a few
        // loads + shuffles. vgpr0-derived ⇒ valid (can't fault on inactive lanes).
        //
        // Contiguity subtlety: the records are contiguous only where consecutive
        // lanes have consecutive vgpr0 (the packed work-item id). vgpr0 is laid
        // out `lx | ly<<10 | lz<<20`, so it is consecutive only *within* a
        // workgroup row (wg_x lanes). Loading one W*stride block from lane 0
        // therefore reads past the row when W spans a row boundary (e.g. W=32 on
        // a 16-wide workgroup: lanes 16..31 live on the next row, 1024*stride
        // away — the old single-block load read uninitialized memory for them,
        // giving nondeterministic output). Instead load one block PER 8-lane
        // group from *that group's own* base address: contiguity is then only
        // assumed within 8 lanes, which holds whenever wg_x is a multiple of 8.
        let fstride = if i.saddr == 124 && !uniform_addr {
            self.frame_cur.borrow().get(&(i.vaddr as u32)).copied()
        } else { None };
        if let Some(stride_bytes) = fstride {
            let sp4 = stride_bytes / 4;
            let ioff_w = (sext_ioffset as i64) / 4;
            if sp4 >= 1 && sext_ioffset % 4 == 0 && ioff_w >= 0 && (ioff_w as u32 + words) <= sp4 {
                let grp = self.w.min(8); // lanes per contiguous group (W-aligned, ≤8)
                let nblk = self.w / grp;
                let blkty = llvm::core::LLVMVectorType(self.i32t, grp * sp4);
                let poison_blk = llvm::core::LLVMGetPoison(blkty);
                // Per-group contiguous load from the group's first lane's address.
                let blocks: Vec<LLVMValueRef> = (0..nblk)
                    .map(|g| {
                        let a = llvm::core::LLVMBuildExtractElement(self.b, base, self.ci32(g * grp), self.n());
                        let p = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, self.n());
                        let ld = llvm::core::LLVMBuildLoad2(self.b, blkty, p, self.n());
                        llvm::core::LLVMSetAlignment(ld, 4);
                        ld
                    })
                    .collect();
                let extract = |fw: u32| -> LLVMValueRef {
                    // Transpose field `fw` out of each group, then concat groups.
                    let parts: Vec<LLVMValueRef> = blocks
                        .iter()
                        .map(|&blk| {
                            let mut idx: Vec<LLVMValueRef> = (0..grp).map(|lane| self.ci32(lane * sp4 + fw)).collect();
                            let mask = llvm::core::LLVMConstVector(idx.as_mut_ptr(), grp);
                            llvm::core::LLVMBuildShuffleVector(self.b, blk, poison_blk, mask, self.n())
                        })
                        .collect();
                    self.vconcat_i32(&parts)
                };
                let mut k = 0u32;
                while k < words {
                    if k + 1 < words {
                        let lo = extract(ioff_w as u32 + k);
                        let hi = extract(ioff_w as u32 + k + 1);
                        let lo64 = self.zext64v(lo);
                        let hi64 = llvm::core::LLVMBuildShl(self.b, self.zext64v(hi), self.splat(self.ci64(32), self.vi64), self.n());
                        let u = self.v_or(hi64, lo64);
                        let d = llvm::core::LLVMBuildBitCast(self.b, u, self.vf64, self.n());
                        self.st_vgpr_f64(i.vdst as u32 + k, d);
                        k += 2;
                    } else {
                        self.st_vgpr32(i.vdst as u32 + k, extract(ioff_w as u32 + k));
                        k += 1;
                    }
                }
                return;
            }
        }
        let mut k = 0u32;
        while k < words {
            if k + 1 < words {
                let ptrs = self.ptr_at_vec(addr, (k as u64) * 4);
                let d = if uniform_addr { self.bcast_load_f64(ptrs) } else { self.masked_gather_f64(ptrs, exec) };
                self.st_vgpr_f64(i.vdst as u32 + k, d);
                k += 2;
            } else {
                let ptrs = self.ptr_at_vec(addr, (k as u64) * 4);
                let d = if uniform_addr { self.bcast_load_i32(ptrs) } else { self.masked_gather(ptrs, exec) };
                self.st_vgpr32(i.vdst as u32 + k, d);
                k += 1;
            }
        }
    }
    fn vgpr_uniform(&self, r: u32) -> bool {
        (self.div_cur.get()[(r >> 7) as usize] >> (r & 127)) & 1 == 0
    }
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
    /// Whether a detected load cluster should really take the transpose path:
    /// a uniform address is served better by scalar-load + broadcast, and an
    /// affine-frame address by the contiguous coalesced/register-resident
    /// paths — both handled per-member by `emit_vglobal`.
    fn cluster_applicable(&self, c: &VgCluster) -> bool {
        !(self.vgpr_uniform(c.vaddr) && self.vgpr_uniform(c.vaddr + 1))
            && !self.frame_cur.borrow().contains_key(&c.vaddr)
    }
    /// Emit a divergent-pointer load cluster (see `vg_cluster`) as W per-lane
    /// contiguous <span×f64> loads + a shuffle transpose, replacing span×(W/8)
    /// masked gathers. Inactive lanes may hold garbage pointers: they are
    /// substituted with an active lane's pointer (umax over active lanes) —
    /// sound because the block only runs with EXEC≠0 (the same invariant the
    /// uniform-address broadcast path relies on), any active lane's record is
    /// dereferenceable over the whole span (contiguity checked in detection),
    /// and inactive lanes' loaded values are merged away by the predicated
    /// stores (or dead, when the elide analysis dropped the predicate).
    unsafe fn emit_vglobal_cluster(&self, members: &[&VGLOBAL], lo: i64, span: u32, preds: &[bool]) {
        let addr = self.ld_vgpr64(members[0].vaddr as u32);
        let exec = self.exec_vec();
        let zero = llvm::core::LLVMConstNull(self.vi64);
        let masked = llvm::core::LLVMBuildSelect(self.b, exec, addr, zero, self.n());
        let p_any = self.call(
            &format!("llvm.vector.reduce.umax.v{}i64", self.w),
            self.i64t, &[self.vi64], &[masked],
        );
        let safe = llvm::core::LLVMBuildSelect(self.b, exec, addr, self.splat(p_any, self.vi64), self.n());
        let rowty = llvm::core::LLVMVectorType(self.f64t, span);
        let rows: Vec<LLVMValueRef> = (0..self.w)
            .map(|l| {
                let a = llvm::core::LLVMBuildExtractElement(self.b, safe, self.ci32(l), self.n());
                let a = llvm::core::LLVMBuildAdd(self.b, a, self.ci64(lo as u64), self.n());
                let p = llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, self.n());
                let ld = llvm::core::LLVMBuildLoad2(self.b, rowty, p, self.n());
                llvm::core::LLVMSetAlignment(ld, 4);
                ld
            })
            .collect();
        let cols = self.transpose_rows(&rows, span);
        for (m, g) in members.iter().enumerate() {
            self.predicate.set(preds[m]);
            let pairs = if matches!(g.op, I::GLOBAL_LOAD_B64) { 1u32 } else { 2 };
            let f0 = ((vg_sext_ioff(g.ioffset) - lo) / 8) as u32;
            for j in 0..pairs {
                self.st_vgpr_f64(g.vdst as u32 + 2 * j, cols[(f0 + j) as usize]);
            }
        }
    }
    /// Transpose W lane-major rows (<span×f64> each) into span column vectors
    /// (<W×f64> each): 3-stage 8×8 butterfly per 8-lane block (24 two-input
    /// v8f64 shuffles, each a single vpermt2pd/vunpck/vshuff64x2), a 7-shuffle
    /// pairwise tree per tail field, then a free per-block concat.
    unsafe fn transpose_rows(&self, rows: &[LLVMValueRef], span: u32) -> Vec<LLVMValueRef> {
        let rowty = llvm::core::LLVMTypeOf(rows[0]);
        let shuf = |x: LLVMValueRef, y: LLVMValueRef, m: &[u32]| -> LLVMValueRef {
            let mut mv: Vec<LLVMValueRef> = m.iter().map(|&i| self.ci32(i)).collect();
            let mask = llvm::core::LLVMConstVector(mv.as_mut_ptr(), mv.len() as u32);
            llvm::core::LLVMBuildShuffleVector(self.b, x, y, mask, self.n())
        };
        let nblk = (self.w / 8) as usize;
        let mut cols: Vec<Vec<LLVMValueRef>> = vec![Vec::with_capacity(nblk); span as usize];
        for blk in 0..nblk {
            let r = &rows[blk * 8..blk * 8 + 8];
            let mut base = 0u32;
            while base + 8 <= span {
                let idx: Vec<u32> = (base..base + 8).collect();
                let sub: Vec<LLVMValueRef> =
                    (0..8).map(|l| shuf(r[l], llvm::core::LLVMGetPoison(rowty), &idx)).collect();
                // stage k swaps bit k of (row index, element index); after all
                // three, c[a][e] = sub[e][a] — verified algebraically.
                let mut a = [std::ptr::null_mut(); 8];
                for i in 0..4 {
                    a[2 * i] = shuf(sub[2 * i], sub[2 * i + 1], &[0, 8, 2, 10, 4, 12, 6, 14]);
                    a[2 * i + 1] = shuf(sub[2 * i], sub[2 * i + 1], &[1, 9, 3, 11, 5, 13, 7, 15]);
                }
                let mut c2 = [std::ptr::null_mut(); 8];
                for &(i, j) in &[(0usize, 2usize), (1, 3), (4, 6), (5, 7)] {
                    c2[i] = shuf(a[i], a[j], &[0, 1, 8, 9, 4, 5, 12, 13]);
                    c2[j] = shuf(a[i], a[j], &[2, 3, 10, 11, 6, 7, 14, 15]);
                }
                let mut c3 = [std::ptr::null_mut(); 8];
                for i in 0..4 {
                    c3[i] = shuf(c2[i], c2[i + 4], &[0, 1, 2, 3, 8, 9, 10, 11]);
                    c3[i + 4] = shuf(c2[i], c2[i + 4], &[4, 5, 6, 7, 12, 13, 14, 15]);
                }
                for cix in 0..8u32 {
                    cols[(base + cix) as usize].push(c3[cix as usize]);
                }
                base += 8;
            }
            for f in base..span {
                let l1: Vec<LLVMValueRef> =
                    (0..4).map(|i| shuf(r[2 * i], r[2 * i + 1], &[f, span + f])).collect();
                let l2: Vec<LLVMValueRef> =
                    (0..2).map(|i| shuf(l1[2 * i], l1[2 * i + 1], &[0, 1, 2, 3])).collect();
                cols[f as usize].push(shuf(l2[0], l2[1], &[0, 1, 2, 3, 4, 5, 6, 7]));
            }
        }
        cols.into_iter()
            .map(|mut parts| {
                let mut n = 8u32;
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
        let align = self.ci32(4);
        let name = format!("llvm.masked.gather.v{}f64.v{}p0", self.w, self.w);
        self.call(&name, self.vf64, &[vptr, self.i32t, self.vi1, self.vf64], &[ptrs, align, mask, passthru])
    }
    // Uniform-address load: lane-0's (shared) address loaded scalar + broadcast.
    // Used only where the address VGPR is proven uniform across the packed lanes
    // (so all lanes' address is identical), replacing an expensive gather.
    unsafe fn bcast_load_i32(&self, ptrs: LLVMValueRef) -> LLVMValueRef {
        let p0 = llvm::core::LLVMBuildExtractElement(self.b, ptrs, self.ci32(0), self.n());
        let v = llvm::core::LLVMBuildLoad2(self.b, self.i32t, p0, self.n());
        self.splat(v, self.vi32)
    }
    unsafe fn bcast_load_f64(&self, ptrs: LLVMValueRef) -> LLVMValueRef {
        let p0 = llvm::core::LLVMBuildExtractElement(self.b, ptrs, self.ci32(0), self.n());
        let v = llvm::core::LLVMBuildLoad2(self.b, self.f64t, p0, self.n());
        self.splat(v, self.vf64)
    }
    unsafe fn masked_gather(&self, ptrs: LLVMValueRef, mask: LLVMValueRef) -> LLVMValueRef {
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        let passthru = llvm::core::LLVMConstNull(self.vi32);
        let align = self.ci32(4);
        let name = format!("llvm.masked.gather.v{}i32.v{}p0", self.w, self.w);
        self.call(&name, self.vi32, &[vptr, self.i32t, self.vi1, self.vi32], &[ptrs, align, mask, passthru])
    }
    unsafe fn masked_scatter(&self, val: LLVMValueRef, ptrs: LLVMValueRef, mask: LLVMValueRef) {
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        let align = self.ci32(4);
        let name = format!("llvm.masked.scatter.v{}i32.v{}p0", self.w, self.w);
        self.call(&name, llvm::core::LLVMVoidTypeInContext(self.ctx), &[self.vi32, vptr, self.i32t, self.vi1], &[val, ptrs, align, mask]);
    }
    /// Typed masked gather: `<W×elem>` load through per-lane pointers, inactive
    /// lanes read 0. Used for VFLAT sub-word loads.
    unsafe fn masked_gather_ty(&self, ptrs: LLVMValueRef, mask: LLVMValueRef, elem: LLVMTypeRef, mnem: &str) -> LLVMValueRef {
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        let velem = llvm::core::LLVMVectorType(elem, self.w);
        let passthru = llvm::core::LLVMConstNull(velem);
        let align = self.ci32(1);
        let name = format!("llvm.masked.gather.v{}{}.v{}p0", self.w, mnem, self.w);
        self.call(&name, velem, &[vptr, self.i32t, self.vi1, velem], &[ptrs, align, mask, passthru])
    }
    unsafe fn masked_scatter_ty(&self, val: LLVMValueRef, ptrs: LLVMValueRef, mask: LLVMValueRef, elem: LLVMTypeRef, mnem: &str) {
        let vptr = llvm::core::LLVMVectorType(self.ptr, self.w);
        let velem = llvm::core::LLVMVectorType(elem, self.w);
        let align = self.ci32(1);
        let name = format!("llvm.masked.scatter.v{}{}.v{}p0", self.w, mnem, self.w);
        self.call(&name, llvm::core::LLVMVoidTypeInContext(self.ctx), &[velem, vptr, self.i32t, self.vi1], &[val, ptrs, align, mask]);
    }

    // ---- VFLAT (per-lane flat gather/scatter) — flat addressing matches the
    // global path; each packed lane holds its own byte address.
    /// Flat-scratch aperture redirect: a per-lane flat address that falls inside
    /// the private aperture [scratch_base, scratch_base+stride) is a private
    /// (scratch) access. The kernel forms a *uniform* logical private pointer and
    /// relies on the hardware per-lane swizzle; the masked interpreter emulates
    /// this. This backend stores private data per-lane in block layout
    /// (`scratch_vec` = base + lane*stride), so map the uniform logical offset to
    /// the lane's own segment: physical = flat_addr + lane*stride. This keeps
    /// VFLAT-private and VSCRATCH consistent (both address base+lane*stride+off).
    ///
    /// Per ISA §11.2/11.3 the aperture test uses ONLY the base address (the VGPR
    /// pair), before IOFFSET is added; the offset is added afterward. So `base` is
    /// the pre-IOFFSET address for the test and `addr` = base + ioffset.
    unsafe fn flat_redirect(&self, base: LLVMValueRef, addr: LLVMValueRef) -> LLVMValueRef {
        use llvm::LLVMIntPredicate::*;
        let sb = self.splat(self.scratch_base_scalar, self.vi64);
        let ap_hi = self.v_add(sb, self.splat(self.scratch_stride, self.vi64));
        let ge = llvm::core::LLVMBuildICmp(self.b, LLVMIntUGE, base, sb, self.n());
        let lt = llvm::core::LLVMBuildICmp(self.b, LLVMIntULT, base, ap_hi, self.n());
        let in_ap = self.v_and(ge, lt);
        // physical = addr + lane*stride, where lane*stride = scratch_vec - base.
        let lane_off = llvm::core::LLVMBuildSub(self.b, self.scratch_vec, sb, self.n());
        let priv_addr = self.v_add(addr, lane_off);
        llvm::core::LLVMBuildSelect(self.b, in_ap, priv_addr, addr, self.n())
    }
    unsafe fn emit_vflat(&self, i: &VFLAT) {
        let sext_ioffset = (((i.ioffset << 8) as i32) >> 8) as i64 as u64;
        let base = if i.saddr != 124 {
            let s = self.splat(self.ld_sgpr64(i.saddr as u32), self.vi64);
            let v = self.zext64v(self.ld_vgpr32(i.vaddr as u32));
            self.v_add(s, v)
        } else {
            self.ld_vgpr64(i.vaddr as u32)
        };
        let addr = self.v_add(base, self.splat(self.ci64(sext_ioffset), self.vi64));
        let addr = self.flat_redirect(base, addr);
        let exec = self.exec_vec();
        let i16t = llvm::core::LLVMInt16TypeInContext(self.ctx);
        let i8t = llvm::core::LLVMInt8TypeInContext(self.ctx);
        match i.op {
            I::FLAT_LOAD_U8 | I::FLAT_LOAD_I8 | I::FLAT_LOAD_U16 | I::FLAT_LOAD_I16 => {
                let (elem, mnem, signed) = match i.op {
                    I::FLAT_LOAD_U8 => (i8t, "i8", false),
                    I::FLAT_LOAD_I8 => (i8t, "i8", true),
                    I::FLAT_LOAD_U16 => (i16t, "i16", false),
                    _ => (i16t, "i16", true),
                };
                let ptrs = self.ptr_at_vec(addr, 0);
                let v = self.masked_gather_ty(ptrs, exec, elem, mnem);
                let z = if signed { llvm::core::LLVMBuildSExt(self.b, v, self.vi32, self.n()) }
                        else { llvm::core::LLVMBuildZExt(self.b, v, self.vi32, self.n()) };
                self.st_vgpr32(i.vdst as u32, z);
                return;
            }
            I::FLAT_STORE_B8 | I::FLAT_STORE_B16 => {
                let (elem, mnem) = if matches!(i.op, I::FLAT_STORE_B8) { (i8t, "i8") } else { (i16t, "i16") };
                let t = llvm::core::LLVMBuildTrunc(self.b, self.ld_vgpr32(i.vsrc as u32), llvm::core::LLVMVectorType(elem, self.w), self.n());
                let ptrs = self.ptr_at_vec(addr, 0);
                self.masked_scatter_ty(t, ptrs, exec, elem, mnem);
                return;
            }
            _ => {}
        }
        let (is_store, words) = match i.op {
            I::FLAT_LOAD_B32 => (false, 1), I::FLAT_LOAD_B64 => (false, 2),
            I::FLAT_LOAD_B96 => (false, 3), I::FLAT_LOAD_B128 => (false, 4),
            I::FLAT_STORE_B32 => (true, 1), I::FLAT_STORE_B64 => (true, 2),
            I::FLAT_STORE_B96 => (true, 3), I::FLAT_STORE_B128 => (true, 4),
            _ => panic!("vec: unsupported VFLAT {:?}", i.op),
        };
        if is_store {
            for k in 0..words {
                let ptrs = self.ptr_at_vec(addr, (k as u64) * 4);
                let d = self.ld_vgpr32(i.vsrc as u32 + k);
                self.masked_scatter(d, ptrs, exec);
            }
            return;
        }
        for k in 0..words {
            let ptrs = self.ptr_at_vec(addr, (k as u64) * 4);
            let d = self.masked_gather(ptrs, exec);
            self.st_vgpr32(i.vdst as u32 + k, d);
        }
    }

    // ---- VSCRATCH (per-lane private scratch) — each lane addresses its own
    // scratch segment (`scratch_vec` = base + lane*stride).
    unsafe fn emit_vscratch(&self, i: &VSCRATCH) {
        let sext_ioffset = (((i.ioffset << 8) as i32) >> 8) as i64 as u64;
        let mut addr = self.v_add(self.scratch_vec, self.splat(self.ci64(sext_ioffset), self.vi64));
        // Scratch SGPR/VGPR offsets are SIGNED 32-bit byte offsets (ISA §11.2).
        if i.saddr != 124 && i.saddr != 127 {
            let s = llvm::core::LLVMBuildSExt(self.b, self.ld_sgpr32(i.saddr as u32), self.i64t, self.n());
            addr = self.v_add(addr, self.splat(s, self.vi64));
        }
        if i.sve != 0 {
            let v = llvm::core::LLVMBuildSExt(self.b, self.ld_vgpr32(i.vaddr as u32), self.vi64, self.n());
            addr = self.v_add(addr, v);
        }
        let (is_store, words) = match i.op {
            I::SCRATCH_LOAD_B32 => (false, 1), I::SCRATCH_LOAD_B64 => (false, 2),
            I::SCRATCH_LOAD_B96 => (false, 3), I::SCRATCH_LOAD_B128 => (false, 4),
            I::SCRATCH_STORE_B32 => (true, 1), I::SCRATCH_STORE_B64 => (true, 2),
            I::SCRATCH_STORE_B96 => (true, 3), I::SCRATCH_STORE_B128 => (true, 4),
            _ => panic!("vec: unsupported VSCRATCH {:?}", i.op),
        };
        let exec = self.exec_vec();
        for k in 0..words {
            let ptrs = self.ptr_at_vec(addr, (k as u64) * 4);
            if is_store {
                let d = self.ld_vgpr32(i.vsrc as u32 + k);
                self.masked_scatter(d, ptrs, exec);
            } else {
                let d = self.masked_gather(ptrs, exec);
                self.st_vgpr32(i.vdst as u32 + k, d);
            }
        }
    }

    // ---- VIMAGE (hardware ray-tracing BVH intersect) — per-lane call to the
    // native `image_bvh64_intersect_ray` helper. Inactive lanes may hold a
    // garbage node address, which the traversal would dereference, so their
    // address is replaced by an active lane's (umax over EXEC-masked addresses,
    // the same substitution the divergent-pointer load cluster uses); their
    // results are discarded by the EXEC-predicated destination writes.
    unsafe fn emit_vimage(&self, i: &VIMAGE) {
        match i.op {
            I::IMAGE_BVH64_INTERSECT_RAY => {
                let exec = self.exec_vec();
                let node = self.ld_vgpr64(i.vaddr0 as u32);
                let zero = llvm::core::LLVMConstNull(self.vi64);
                let masked = llvm::core::LLVMBuildSelect(self.b, exec, node, zero, self.n());
                let any = self.call(
                    &format!("llvm.vector.reduce.umax.v{}i64", self.w),
                    self.i64t, &[self.vi64], &[masked],
                );
                let safe = llvm::core::LLVMBuildSelect(self.b, exec, node, self.splat(any, self.vi64), self.n());

                let bits_to_f32_lane = |r: u32, k: u32| -> LLVMValueRef {
                    let v = self.ld_vgpr32(r);
                    let e = llvm::core::LLVMBuildExtractElement(self.b, v, self.ci32(k), self.n());
                    llvm::core::LLVMBuildBitCast(self.b, e, self.f32t, self.n())
                };
                let scratch_ptr = |k: u32| -> LLVMValueRef {
                    llvm::core::LLVMBuildGEP2(self.b, self.i32t, self.bvh_scratch, [self.ci32(k)].as_mut_ptr(), 1, self.n())
                };
                let params = [
                    self.ptr, self.ptr, self.ptr, self.ptr, self.i64t,
                    self.f32t, self.f32t, self.f32t, self.f32t,
                    self.f32t, self.f32t, self.f32t, self.f32t, self.f32t, self.f32t,
                ];
                let mut outs: [LLVMValueRef; 4] = [llvm::core::LLVMGetPoison(self.vi32); 4];
                for lane in 0..self.w {
                    let node_l = llvm::core::LLVMBuildExtractElement(self.b, safe, self.ci32(lane), self.n());
                    let args = [
                        scratch_ptr(0), scratch_ptr(1), scratch_ptr(2), scratch_ptr(3),
                        node_l,
                        bits_to_f32_lane(i.vaddr1 as u32, lane),
                        bits_to_f32_lane(i.vaddr2 as u32, lane), bits_to_f32_lane(i.vaddr2 as u32 + 1, lane), bits_to_f32_lane(i.vaddr2 as u32 + 2, lane),
                        bits_to_f32_lane(i.vaddr3 as u32, lane), bits_to_f32_lane(i.vaddr3 as u32 + 1, lane), bits_to_f32_lane(i.vaddr3 as u32 + 2, lane),
                        bits_to_f32_lane(i.vaddr4 as u32, lane), bits_to_f32_lane(i.vaddr4 as u32 + 1, lane), bits_to_f32_lane(i.vaddr4 as u32 + 2, lane),
                    ];
                    self.call("image_bvh64_intersect_ray", llvm::core::LLVMVoidTypeInContext(self.ctx), &params, &args);
                    for k in 0..4u32 {
                        let v = llvm::core::LLVMBuildLoad2(self.b, self.i32t, scratch_ptr(k), self.n());
                        outs[k as usize] = llvm::core::LLVMBuildInsertElement(self.b, outs[k as usize], v, self.ci32(lane), self.n());
                    }
                }
                for k in 0..4u32 {
                    self.st_vgpr32(i.vdata as u32 + k, outs[k as usize]);
                }
            }
            _ => panic!("vec: unsupported VIMAGE {:?}", i.op),
        }
    }

    // ---- VSAMPLE (texture sample) ---------------------------------------
    // The runtime sampler `image_sample_lz` is scalar (bit-exact with the scalar
    // path's helper), so call it once per lane: the descriptor (rsrc) is uniform
    // and broadcast, the coordinates are per-lane f32. It bounds-checks the
    // texel index, so inactive lanes (undef coords) are harmless and their
    // result is dropped by the predicated destination write.
    unsafe fn emit_vsample(&self, i: &VSAMPLE) {
        match i.op {
            I::IMAGE_SAMPLE_LZ => {
                let mut in_vecs: Vec<LLVMValueRef> =
                    (0..8).map(|k| self.splat(self.ld_sgpr32(i.rsrc as u32 + k), self.vi32)).collect();
                in_vecs.push(self.vf32_of(self.ld_vgpr32(i.vaddr0 as u32)));
                in_vecs.push(self.vf32_of(self.ld_vgpr32(i.vaddr1 as u32)));
                let in_tys = [
                    self.i32t, self.i32t, self.i32t, self.i32t,
                    self.i32t, self.i32t, self.i32t, self.i32t,
                    self.f32t, self.f32t,
                ];
                let data = self.per_lane("image_sample_lz", self.i32t, &in_vecs, &in_tys, self.vi32);
                self.st_vgpr32(i.vdata as u32, data);
            }
            _ => panic!("vec: unsupported VSAMPLE {:?}", i.op),
        }
    }
}

// ---- divergent-pointer load clustering ------------------------------------
// A run of *consecutive* VGLOBAL f64-shaped loads (B64/B128) off the same
// divergent per-lane pointer (saddr=124), together covering a contiguous,
// pairwise 8-aligned byte span, is one per-lane record read (smallpt: the
// 9-f64 sphere record at 0x8..0x50). `vg_cluster` recognizes the run so it can
// be emitted as W contiguous per-lane vector loads + a shuffle transpose
// instead of span × W/8 masked gathers — the gathers and their per-gather
// mask refills (kmovq) were ~8% of W=16 kernel cycles. Consecutive-only keeps
// this trivially sound: no instruction intervenes, so EXEC, memory and the
// address VGPRs cannot change inside the run (scheduling no-ops are already
// filtered out of `body` by ir::is_noop).
struct VgCluster {
    len: usize, // number of member instructions (≥ 3)
    vaddr: u32, // shared per-lane pointer pair
    lo: i64,    // lowest sign-extended ioffset (span start, bytes)
    span: u32,  // span length in f64 fields
}

fn vg_sext_ioff(io: u32) -> i64 {
    (((io << 8) as i32) >> 8) as i64
}

fn vg_cluster(body: &[InstFormat]) -> Option<VgCluster> {
    fn f64_words(g: &VGLOBAL) -> Option<u32> {
        match g.op {
            I::GLOBAL_LOAD_B64 => Some(2),
            I::GLOBAL_LOAD_B128 => Some(4),
            _ => None,
        }
    }
    let first = match body.first()? {
        InstFormat::VGLOBAL(g) if g.saddr == 124 && f64_words(g).is_some() => g,
        _ => return None,
    };
    let vaddr = first.vaddr;
    let mut members: Vec<&VGLOBAL> = Vec::new();
    for inst in body {
        let g = match inst {
            InstFormat::VGLOBAL(g) => g,
            _ => break,
        };
        let Some(w) = f64_words(g) else { break };
        if g.saddr != 124 || g.vaddr != vaddr {
            break;
        }
        members.push(g);
        // This load overwrites the pointer pair: later loads would read the
        // NEW address — stop extending (this member itself is still fine).
        let dst = g.vdst as u32..g.vdst as u32 + w;
        if dst.contains(&(vaddr as u32)) || dst.contains(&(vaddr as u32 + 1)) {
            break;
        }
    }
    if members.len() < 3 {
        return None;
    }
    // Contiguous coverage: every byte in [lo, hi) is read by some member, so
    // for an ACTIVE lane the whole span is guest-dereferenced (fault-safe).
    let mut ranges: Vec<(i64, i64)> = members
        .iter()
        .map(|g| {
            let s = vg_sext_ioff(g.ioffset);
            (s, s + 4 * f64_words(g).unwrap() as i64)
        })
        .collect();
    ranges.sort();
    let lo = ranges[0].0;
    let mut hi = ranges[0].1;
    for &(a, b) in &ranges[1..] {
        if a > hi {
            return None;
        }
        hi = hi.max(b);
    }
    // Members must decompose into f64 columns of the span.
    if members.iter().any(|g| (vg_sext_ioff(g.ioffset) - lo) % 8 != 0) || (hi - lo) % 8 != 0 {
        return None;
    }
    let span = ((hi - lo) / 8) as u32;
    if span < 4 {
        return None;
    }
    Some(VgCluster { len: members.len(), vaddr: vaddr as u32, lo, span })
}

// f64 compare opcode -> (predicate, invert result)
fn f32_pred(op: I) -> Option<(llvm::LLVMRealPredicate, bool)> {
    use llvm::LLVMRealPredicate::*;
    Some(match op {
        I::V_CMP_GT_F32 | I::V_CMPX_GT_F32 => (LLVMRealOGT, false),
        I::V_CMP_LT_F32 | I::V_CMPX_LT_F32 => (LLVMRealOLT, false),
        I::V_CMP_LE_F32 | I::V_CMPX_LE_F32 => (LLVMRealOLE, false),
        I::V_CMP_GE_F32 | I::V_CMPX_GE_F32 => (LLVMRealOGE, false),
        I::V_CMP_EQ_F32 | I::V_CMPX_EQ_F32 => (LLVMRealOEQ, false),
        I::V_CMP_LG_F32 | I::V_CMPX_LG_F32 => (LLVMRealONE, false),
        I::V_CMP_NLT_F32 | I::V_CMPX_NLT_F32 => (LLVMRealOLT, true),
        I::V_CMP_NGT_F32 | I::V_CMPX_NGT_F32 => (LLVMRealOGT, true),
        I::V_CMP_NGE_F32 | I::V_CMPX_NGE_F32 => (LLVMRealOGE, true),
        I::V_CMP_NLE_F32 | I::V_CMPX_NLE_F32 => (LLVMRealOLE, true),
        I::V_CMP_NEQ_F32 | I::V_CMPX_NEQ_F32 => (LLVMRealOEQ, true),
        _ => return None,
    })
}

fn int64_pred(op: I) -> Option<llvm::LLVMIntPredicate> {
    use llvm::LLVMIntPredicate::*;
    Some(match op {
        I::V_CMP_EQ_U64 | I::V_CMPX_EQ_U64 | I::V_CMP_EQ_I64 | I::V_CMPX_EQ_I64 => LLVMIntEQ,
        I::V_CMP_NE_U64 | I::V_CMPX_NE_U64 | I::V_CMP_NE_I64 | I::V_CMPX_NE_I64 => LLVMIntNE,
        I::V_CMP_GT_U64 | I::V_CMPX_GT_U64 => LLVMIntUGT,
        I::V_CMP_LT_U64 | I::V_CMPX_LT_U64 => LLVMIntULT,
        I::V_CMP_GE_U64 | I::V_CMPX_GE_U64 => LLVMIntUGE,
        I::V_CMP_LE_U64 | I::V_CMPX_LE_U64 => LLVMIntULE,
        I::V_CMP_GT_I64 | I::V_CMPX_GT_I64 => LLVMIntSGT,
        I::V_CMP_LT_I64 | I::V_CMPX_LT_I64 => LLVMIntSLT,
        I::V_CMP_GE_I64 | I::V_CMPX_GE_I64 => LLVMIntSGE,
        I::V_CMP_LE_I64 | I::V_CMPX_LE_I64 => LLVMIntSLE,
        _ => return None,
    })
}

fn f64_pred(op: I) -> Option<(llvm::LLVMRealPredicate, bool)> {
    use llvm::LLVMRealPredicate::*;
    Some(match op {
        I::V_CMP_GT_F64 | I::V_CMPX_GT_F64 => (LLVMRealOGT, false),
        I::V_CMP_LT_F64 | I::V_CMPX_LT_F64 => (LLVMRealOLT, false),
        I::V_CMP_LE_F64 | I::V_CMPX_LE_F64 => (LLVMRealOLE, false),
        I::V_CMP_GE_F64 | I::V_CMPX_GE_F64 => (LLVMRealOGE, false),
        I::V_CMP_EQ_F64 | I::V_CMPX_EQ_F64 => (LLVMRealOEQ, false),
        I::V_CMP_LG_F64 | I::V_CMPX_LG_F64 => (LLVMRealONE, false),
        I::V_CMP_NLT_F64 | I::V_CMPX_NLT_F64 => (LLVMRealOLT, true),
        I::V_CMP_NGT_F64 | I::V_CMPX_NGT_F64 => (LLVMRealOGT, true),
        I::V_CMP_NGE_F64 | I::V_CMPX_NGE_F64 => (LLVMRealOGE, true),
        I::V_CMP_NLE_F64 | I::V_CMPX_NLE_F64 => (LLVMRealOLE, true),
        I::V_CMP_NEQ_F64 | I::V_CMPX_NEQ_F64 => (LLVMRealOEQ, true),
        _ => return None,
    })
}
fn int_pred(op: I) -> Option<llvm::LLVMIntPredicate> {
    use llvm::LLVMIntPredicate::*;
    Some(match op {
        I::V_CMP_EQ_U32 | I::V_CMPX_EQ_U32 => LLVMIntEQ,
        I::V_CMP_NE_U32 | I::V_CMPX_NE_U32 => LLVMIntNE,
        I::V_CMP_GT_U32 | I::V_CMPX_GT_U32 => LLVMIntUGT,
        I::V_CMP_LT_U32 | I::V_CMPX_LT_U32 => LLVMIntULT,
        I::V_CMP_GE_U32 | I::V_CMPX_GE_U32 => LLVMIntUGE,
        I::V_CMP_LE_U32 | I::V_CMPX_LE_U32 => LLVMIntULE,
        I::V_CMP_LT_I32 | I::V_CMPX_LT_I32 => LLVMIntSLT,
        I::V_CMP_GT_I32 | I::V_CMPX_GT_I32 => LLVMIntSGT,
        _ => return None,
    })
}

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
