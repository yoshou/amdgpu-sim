//! LLVM codegen: lower a [`ScalarProgram`] to a single-work-item native
//! function and JIT it with ORC.
//!
//! Register model: SGPR/VGPR/SCC are `alloca` slots initialized once from the
//! incoming pointers at entry. Because their addresses never escape, `mem2reg`
//! promotes them to SSA values, which is what lets LLVM optimize the body to
//! native quality. Output leaves the kernel through `global_store` to absolute
//! host addresses (loaded out of the kernarg buffer), so no register write-back
//! is needed. There are no barriers in the target kernel, so the function runs
//! to completion in one call.

use std::collections::BTreeMap;
use std::ffi::CString;

use llvm_sys as llvm;
use llvm::prelude::{LLVMBasicBlockRef, LLVMBuilderRef, LLVMTypeRef, LLVMValueRef};

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand, DS, SMEM, SOP1, SOP2, SOPK, VFLAT, VGLOBAL, VIMAGE, VOP1, VOP2, VOP3, VOP3P, VOP3SD, VOPC, VOPD, VSAMPLE, VSCRATCH};

use super::ir::{Cond, ScalarProgram, Terminator};

/// The SGPR number if `o` is a scalar register operand.
fn sreg(o: &SourceOperand) -> Option<u32> {
    match o {
        SourceOperand::ScalarRegister(r) => Some(*r as u32),
        _ => None,
    }
}

/// One side of a recognized `(A & !mask) | (B & mask)` expression. It is
/// recorded when an SGPR is defined by an EXEC/VCC-masked `and` and consumed at
/// the matching `s_or` to emit a `select` (→ cmov) instead of the 32-bit
/// `(B&M)|(A&~M)` blend.
#[derive(Clone, Copy)]
enum MaskDef {
    /// `dst = A & ~M` (S_AND_NOT1): `a` is A's value, `m` the mask SGPR.
    AndNot1 { a: LLVMValueRef, m: u32 },
    /// `dst = B & M` (S_AND): `b` is B's value, `cond` = M's bit0 as i1.
    And { b: LLVMValueRef, cond: LLVMValueRef, m: u32 },
}

const EXEC: u32 = 126;
const VCC: u32 = 106;

/// A JIT-compiled single-work-item kernel. The machine code lives for the
/// process lifetime (the owning LLJIT is intentionally leaked), so the function
/// pointer is safe to call from many threads concurrently with disjoint data.
pub struct ScalarKernel {
    addr: u64,
    pub num_vgprs: usize,
}

unsafe impl Send for ScalarKernel {}
unsafe impl Sync for ScalarKernel {}

impl ScalarKernel {
    /// Run one work-item. `sgprs` points to 128 u32 slots, `vgprs` to
    /// `num_vgprs` u32 slots (both set up by the dispatcher).
    pub unsafe fn run(&self, sgprs: *mut u32, vgprs: *mut u32, scratch_base: u64) {
        let f = std::mem::transmute::<
            u64,
            extern "C" fn(*mut u32, *mut u32, u64),
        >(self.addr);
        f(sgprs, vgprs, scratch_base);
    }
}

/// A JIT-compiled cooperative work-item kernel: like [`ScalarKernel`] but the
/// function yields at workgroup barriers. One call runs the work-item from
/// `resume_pc` to the next barrier (or to `s_endpgm`), persisting registers/SCC
/// into the caller's buffers so the next call resumes correctly.
pub struct CoopKernel {
    addr: u64,
    pub num_vgprs: usize,
    /// pc passed as `resume_pc` on the first call (the program entry block).
    pub entry_pc: usize,
}

unsafe impl Send for CoopKernel {}
unsafe impl Sync for CoopKernel {}

/// Return sentinel meaning the work-item reached `s_endpgm`.
pub const COOP_DONE: u64 = u64::MAX;

/// Size (in u32 slots) of the cooperative per-work-item SGPR buffer: the 128
/// architectural SGPRs plus SCC persisted at index 128 (RDNA4 has no SGPR there;
/// it is a private convention for carrying the condition code across a barrier).
pub const COOP_SGPR_BUF: usize = 129;
/// Size (in u32 slots) of the dedicated per-work-item lane-spill buffer. This is
/// NOT architectural register state — it backs the uniform writelane/readlane
/// idiom (values the compiler stashes in fixed VGPR lanes) so those slots survive
/// barrier yields. Kept separate from the SGPR/VGPR files to avoid pretending
/// RDNA4 has registers it does not.
pub const COOP_SPILL_SLOTS: usize = 256;

impl CoopKernel {
    /// Run one work-item from `resume_pc`. `sgprs` points to [`COOP_SGPR_BUF`]
    /// u32 slots (128 SGPRs + SCC at index 128), `vgprs` to `num_vgprs` slots,
    /// `lds` to the workgroup's shared LDS, and `spill` to this work-item's
    /// [`COOP_SPILL_SLOTS`]-slot persistent lane-spill buffer. Returns the next
    /// resume pc, or [`COOP_DONE`].
    pub unsafe fn run(
        &self,
        sgprs: *mut u32,
        vgprs: *mut u32,
        scratch_base: u64,
        lds_base: u64,
        spill: *mut u32,
        resume_pc: u64,
    ) -> u64 {
        let f = std::mem::transmute::<
            u64,
            extern "C" fn(*mut u32, *mut u32, u64, u64, *mut u32, u64) -> u64,
        >(self.addr);
        f(sgprs, vgprs, scratch_base, lds_base, spill, resume_pc)
    }
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

struct Cg {
    ctx: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    b: LLVMBuilderRef,
    func: LLVMValueRef,
    scratch_base: LLVMValueRef,
    sgpr: Vec<LLVMValueRef>, // 128 i32 allocas
    vgpr: Vec<LLVMValueRef>, // num_vgprs i32 allocas
    scc: LLVMValueRef,       // i1 alloca
    // cached types
    i1: LLVMTypeRef,
    i8: LLVMTypeRef,
    i32t: LLVMTypeRef,
    i64t: LLVMTypeRef,
    f32t: LLVMTypeRef,
    f64t: LLVMTypeRef,
    ptr: LLVMTypeRef,
    // When true, vector (VGPR/VMEM) writes are predicated on EXEC bit 0: an
    // inactive lane preserves the old value, matching the masked backend's
    // per-lane semantics. Disabled during the entry register init.
    predicate: std::cell::Cell<bool>,
    // f64 register typing: a parallel `double` alloca per VGPR pair (low reg).
    // f64 ops read/write these directly so a double crosses blocks as one f64
    // phi (mem2reg) instead of two i32 phis + reconstruction. `f64_fresh` is the
    // running bitmask (bit r = shadow[r] holds the current value of pair r:r+1),
    // seeded from the freshness analysis at each block entry and updated as
    // instructions emit.
    vgpr_f64: Vec<LLVMValueRef>,
    f64_fresh: std::cell::Cell<super::regtype::RegSet>,
    // EXEC(126)/VCC(106) are architecturally lane masks, never data. For a single
    // lane they carry one meaningful bit, so we keep them as i1 allocas and
    // convert at the i32 boundary (zext on read, trunc on write). This lets LLVM
    // fold the wavefront mask arithmetic (s_and/s_or/saveexec) down to i1 logic
    // instead of emitting 32-bit `andn/and/or` + AVX-512 `kmovd` per iteration.
    exec_i1: LLVMValueRef,
    vcc_i1: LLVMValueRef,
    // i64 shadow per SGPR pair (low reg) — a loop-carried 64-bit base pointer
    // flows as one i64 phi instead of two i32 phis + per-iteration reconstruct.
    sgpr_i64: Vec<LLVMValueRef>,
    sgpr_fresh: std::cell::Cell<u128>,
    // De-SIMT mask-select fusion: per-block record of SGPRs defined by an
    // EXEC/VCC-masked `and`/`and_not1`. At the matching `s_or` the pair
    // `(B&M)|(A&~M)` is emitted as `select(M, B, A)` (→ cmov, as native), and the
    // dead `and/andn` DCE away. Cleared at block start; invalidated per write
    // (all entries when EXEC/VCC change).
    mask_def: std::cell::RefCell<BTreeMap<u32, MaskDef>>,
    writeback: bool,
    writeback_vgprs: usize,
    // Cooperative (workgroup-barrier) mode: the function has signature
    // `(sgprs, vgprs, scratch, lds, spill, resume_pc:i64) -> i64` and yields at
    // barriers.
    coop: bool,
    // LDS base pointer (function param) in cooperative mode; undef otherwise.
    lds_base: LLVMValueRef,
    // Dedicated lane-spill buffer pointer (function param) in cooperative mode.
    spill_base: LLVMValueRef,
    // Slot index per (spill VGPR, constant lane) into `spill_base` for the uniform
    // writelane/readlane idiom. See `spill_slot_ptr`.
    spill: std::cell::RefCell<BTreeMap<(u32, u32), usize>>,
    // Reusable [10 x i32] entry-block scratch for ray-trace helper results and
    // the predicated-store dummy (allocated once, never grows the stack).
    bvh_scratch: LLVMValueRef,
}

impl Cg {
    unsafe fn n(&self) -> *const i8 {
        b"\0".as_ptr() as *const i8
    }

    // ---- intrinsic / external function declaration -----------------------
    unsafe fn get_func(&self, name: &str, ret: LLVMTypeRef, params: &[LLVMTypeRef]) -> (LLVMValueRef, LLVMTypeRef) {
        let cname = cstr(name);
        let mut f = llvm::core::LLVMGetNamedFunction(self.module, cname.as_ptr());
        let fty = llvm::core::LLVMFunctionType(
            ret,
            params.as_ptr() as *mut _,
            params.len() as u32,
            0,
        );
        if f.is_null() {
            f = llvm::core::LLVMAddFunction(self.module, cname.as_ptr(), fty);
        }
        (f, fty)
    }

    unsafe fn call(&self, name: &str, ret: LLVMTypeRef, params: &[LLVMTypeRef], args: &[LLVMValueRef]) -> LLVMValueRef {
        let (f, fty) = self.get_func(name, ret, params);
        llvm::core::LLVMBuildCall2(
            self.b,
            fty,
            f,
            args.as_ptr() as *mut _,
            args.len() as u32,
            self.n(),
        )
    }

    // ---- register access -------------------------------------------------
    unsafe fn ld_sgpr32(&self, i: u32) -> LLVMValueRef {
        // EXEC/VCC live as i1; present them to integer consumers as 0/1.
        if i == EXEC {
            let b = llvm::core::LLVMBuildLoad2(self.b, self.i1, self.exec_i1, self.n());
            return llvm::core::LLVMBuildZExt(self.b, b, self.i32t, self.n());
        }
        if i == VCC {
            let b = llvm::core::LLVMBuildLoad2(self.b, self.i1, self.vcc_i1, self.n());
            return llvm::core::LLVMBuildZExt(self.b, b, self.i32t, self.n());
        }
        llvm::core::LLVMBuildLoad2(self.b, self.i32t, self.sgpr[i as usize], self.n())
    }
    unsafe fn st_sgpr32(&self, i: u32, v: LLVMValueRef) {
        // Invalidate pending mask-select records: changing EXEC/VCC changes the
        // mask value, so drop all; any other write drops that register's record.
        {
            let mut md = self.mask_def.borrow_mut();
            if i == EXEC || i == VCC { md.clear(); } else { md.remove(&i); }
        }
        // EXEC/VCC: keep only the single meaningful lane bit (bit 0) as i1.
        if i == EXEC || i == VCC {
            let bit = llvm::core::LLVMBuildTrunc(self.b, v, self.i1, self.n());
            let slot = if i == EXEC { self.exec_i1 } else { self.vcc_i1 };
            llvm::core::LLVMBuildStore(self.b, bit, slot);
            return;
        }
        // A 32-bit write clobbers the i64 shadow of pairs i (i:i+1) and i-1.
        let mut fr = self.sgpr_fresh.get();
        fr &= !(1u128 << (i & 127));
        if i > 0 { fr &= !(1u128 << ((i - 1) & 127)); }
        self.sgpr_fresh.set(fr);
        llvm::core::LLVMBuildStore(self.b, v, self.sgpr[i as usize]);
    }
    unsafe fn pred_vgpr32(&self, i: u32, v: LLVMValueRef) -> LLVMValueRef {
        if self.predicate.get() {
            let old = llvm::core::LLVMBuildLoad2(self.b, self.i32t, self.vgpr[i as usize], self.n());
            let active = llvm::core::LLVMBuildICmp(
                self.b,
                llvm::LLVMIntPredicate::LLVMIntNE,
                self.b_and(self.ld_sgpr32(EXEC), self.ci32(1)),
                self.ci32(0),
                self.n(),
            );
            llvm::core::LLVMBuildSelect(self.b, active, v, old, self.n())
        } else {
            v
        }
    }
    unsafe fn ld_vgpr32(&self, i: u32) -> LLVMValueRef {
        llvm::core::LLVMBuildLoad2(self.b, self.i32t, self.vgpr[i as usize], self.n())
    }
    unsafe fn st_vgpr32(&self, i: u32, v: LLVMValueRef) {
        self.f64_fresh_clr(i);
        if i > 0 { self.f64_fresh_clr(i - 1); }
        let v = self.pred_vgpr32(i, v);
        llvm::core::LLVMBuildStore(self.b, v, self.vgpr[i as usize]);
    }
    /// Seed the running f64-fresh bitmask at a block boundary from the
    /// cross-block freshness analysis.
    unsafe fn set_f64_fresh(&self, fresh: super::regtype::RegSet) {
        self.f64_fresh.set(fresh);
    }
    // Pair-freshness bit ops over the 256-bit RegSet (VGPRs number up to 256;
    // `& 127` u128 indexing would alias pair p with p+128).
    fn f64_fresh_get(&self, p: u32) -> bool {
        super::regtype::bget(&self.f64_fresh.get(), p)
    }
    fn f64_fresh_setbit(&self, p: u32) {
        let mut s = self.f64_fresh.get();
        let r = (p & 255) as usize;
        s[r / 128] |= 1u128 << (r % 128);
        self.f64_fresh.set(s);
    }
    fn f64_fresh_clr(&self, p: u32) {
        let mut s = self.f64_fresh.get();
        let r = (p & 255) as usize;
        s[r / 128] &= !(1u128 << (r % 128));
        self.f64_fresh.set(s);
    }
    unsafe fn ld_scc(&self) -> LLVMValueRef {
        llvm::core::LLVMBuildLoad2(self.b, self.i1, self.scc, self.n())
    }
    unsafe fn st_scc(&self, v: LLVMValueRef) {
        // v is i1
        llvm::core::LLVMBuildStore(self.b, v, self.scc);
    }
    /// SCC = (value != 0)
    unsafe fn st_scc_nz(&self, v32: LLVMValueRef) {
        let z = self.ci32(0);
        let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, v32, z, self.n());
        self.st_scc(c);
    }

    unsafe fn zext64(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildZExt(self.b, v, self.i64t, self.n())
    }

    /// `x * 2^exp` computed inline as three clamped power-of-two multiplies,
    /// matching the masked backend. `llvm.ldexp.f64.i32` lowers to a `scalbn`
    /// libcall on x86 (measured at ~20% of total runtime), so we avoid it. The
    /// 3 steps × [-1022,1023] cover the full i32 exponent range with correct
    /// overflow/underflow/denormal rounding.
    unsafe fn ldexp_inline(&self, value: LLVMValueRef, exp: LLVMValueRef) -> LLVMValueRef {
        use llvm::LLVMIntPredicate::*;
        let mut result = value;
        let mut remaining = exp;
        for _ in 0..3 {
            // step = clamp(remaining, -1022, 1023)
            let hi = self.ci32(1023);
            let lo = llvm::core::LLVMConstInt(self.i32t, (-1022i32) as u64, 1);
            let c1 = llvm::core::LLVMBuildICmp(self.b, LLVMIntSLT, remaining, hi, self.n());
            let step = llvm::core::LLVMBuildSelect(self.b, c1, remaining, hi, self.n());
            let c2 = llvm::core::LLVMBuildICmp(self.b, LLVMIntSGT, step, lo, self.n());
            let step = llvm::core::LLVMBuildSelect(self.b, c2, step, lo, self.n());
            remaining = llvm::core::LLVMBuildSub(self.b, remaining, step, self.n());
            // scale = bitcast((sext(step) + 1023) << 52)
            let step64 = llvm::core::LLVMBuildSExt(self.b, step, self.i64t, self.n());
            let biased = self.b_add(step64, self.ci64(1023));
            let bits = llvm::core::LLVMBuildShl(self.b, biased, self.ci64(52), self.n());
            let scale = llvm::core::LLVMBuildBitCast(self.b, bits, self.f64t, self.n());
            result = self.fmf(llvm::core::LLVMBuildFMul(self.b, result, scale, self.n()));
        }
        result
    }

    unsafe fn ld_sgpr64(&self, i: u32) -> LLVMValueRef {
        if self.sgpr_fresh.get() & (1u128 << (i & 127)) != 0 {
            return llvm::core::LLVMBuildLoad2(self.b, self.i64t, self.sgpr_i64[i as usize], self.n());
        }
        let lo = self.zext64(self.ld_sgpr32(i));
        let hi = self.zext64(self.ld_sgpr32(i + 1));
        let hi = llvm::core::LLVMBuildShl(self.b, hi, self.ci64(32), self.n());
        let v = llvm::core::LLVMBuildOr(self.b, hi, lo, self.n());
        llvm::core::LLVMBuildStore(self.b, v, self.sgpr_i64[i as usize]);
        self.sgpr_fresh.set(self.sgpr_fresh.get() | (1u128 << (i & 127)));
        v
    }
    unsafe fn st_sgpr64(&self, i: u32, v: LLVMValueRef) {
        let lo = llvm::core::LLVMBuildTrunc(self.b, v, self.i32t, self.n());
        let hi = llvm::core::LLVMBuildLShr(self.b, v, self.ci64(32), self.n());
        let hi = llvm::core::LLVMBuildTrunc(self.b, hi, self.i32t, self.n());
        self.st_sgpr32(i, lo); // clears sgpr_fresh for i-1/i
        self.st_sgpr32(i + 1, hi); // clears for i/i+1
        llvm::core::LLVMBuildStore(self.b, v, self.sgpr_i64[i as usize]);
        self.sgpr_fresh.set(self.sgpr_fresh.get() | (1u128 << (i & 127)));
    }
    unsafe fn ld_vgpr64(&self, i: u32) -> LLVMValueRef {
        let lo = self.zext64(self.ld_vgpr32(i));
        let hi = self.zext64(self.ld_vgpr32(i + 1));
        let hi = llvm::core::LLVMBuildShl(self.b, hi, self.ci64(32), self.n());
        llvm::core::LLVMBuildOr(self.b, hi, lo, self.n())
    }
    unsafe fn ld_vgpr_f64(&self, i: u32) -> LLVMValueRef {
        // Freshness path: read the shadow if fresh, else reconstruct + memoize.
        if self.f64_fresh_get(i) {
            return llvm::core::LLVMBuildLoad2(self.b, self.f64t, self.vgpr_f64[i as usize], self.n());
        }
        let u = self.ld_vgpr64(i);
        let d = llvm::core::LLVMBuildBitCast(self.b, u, self.f64t, self.n());
        llvm::core::LLVMBuildStore(self.b, d, self.vgpr_f64[i as usize]);
        self.f64_fresh_setbit(i);
        d
    }
    unsafe fn st_vgpr64(&self, i: u32, v: LLVMValueRef) {
        let lo = llvm::core::LLVMBuildTrunc(self.b, v, self.i32t, self.n());
        let hi = llvm::core::LLVMBuildLShr(self.b, v, self.ci64(32), self.n());
        let hi = llvm::core::LLVMBuildTrunc(self.b, hi, self.i32t, self.n());
        self.st_vgpr32(i, lo);
        self.st_vgpr32(i + 1, hi);
    }
    unsafe fn st_vgpr_f64(&self, i: u32, v: LLVMValueRef) {
        // Freshness path: store i32 halves + refresh the double shadow.
        let u = llvm::core::LLVMBuildBitCast(self.b, v, self.i64t, self.n());
        self.st_vgpr64(i, u);
        if !self.predicate.get() {
            llvm::core::LLVMBuildStore(self.b, v, self.vgpr_f64[i as usize]);
            self.f64_fresh_setbit(i);
        }
    }

    // ---- constants -------------------------------------------------------
    unsafe fn ci32(&self, v: u32) -> LLVMValueRef {
        llvm::core::LLVMConstInt(self.i32t, v as u64, 0)
    }
    unsafe fn ci64(&self, v: u64) -> LLVMValueRef {
        llvm::core::LLVMConstInt(self.i64t, v, 0)
    }
    unsafe fn cf64(&self, v: f64) -> LLVMValueRef {
        llvm::core::LLVMConstReal(self.f64t, v)
    }
    unsafe fn cf32(&self, v: f32) -> LLVMValueRef {
        llvm::core::LLVMConstReal(self.f32t, v as f64)
    }
    // f32 lives in a VGPR as raw i32 bits; convert at the arithmetic boundary.
    unsafe fn f32_bits(&self, v: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildBitCast(self.b, v, self.i32t, self.n())
    }
    unsafe fn src_f32(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::FloatConstant(v) => self.cf32(*v as f32),
            _ => llvm::core::LLVMBuildBitCast(self.b, self.src_u32(op), self.f32t, self.n()),
        }
    }
    unsafe fn f16ty(&self) -> LLVMTypeRef { llvm::core::LLVMHalfTypeInContext(self.ctx) }
    unsafe fn i16ty(&self) -> LLVMTypeRef { llvm::core::LLVMInt16TypeInContext(self.ctx) }
    /// Low (bit 0) f16 of `op`, widened to f32.
    unsafe fn src_f16lo_f32(&self, op: &SourceOperand) -> LLVMValueRef {
        let b16 = llvm::core::LLVMBuildTrunc(self.b, self.src_u32(op), self.i16ty(), self.n());
        let h = llvm::core::LLVMBuildBitCast(self.b, b16, self.f16ty(), self.n());
        llvm::core::LLVMBuildFPExt(self.b, h, self.f32t, self.n())
    }
    /// High (bit 16) f16 of `op`, widened to f32.
    unsafe fn src_f16hi_f32(&self, op: &SourceOperand) -> LLVMValueRef {
        let hi = llvm::core::LLVMBuildLShr(self.b, self.src_u32(op), self.ci32(16), self.n());
        let b16 = llvm::core::LLVMBuildTrunc(self.b, hi, self.i16ty(), self.n());
        let h = llvm::core::LLVMBuildBitCast(self.b, b16, self.f16ty(), self.n());
        llvm::core::LLVMBuildFPExt(self.b, h, self.f32t, self.n())
    }
    /// Round f32 `v` to f16, returned as i32 (f16 bits in low 16, high 0).
    unsafe fn f32_to_f16_bits(&self, v: LLVMValueRef) -> LLVMValueRef {
        let h = llvm::core::LLVMBuildFPTrunc(self.b, v, self.f16ty(), self.n());
        let b16 = llvm::core::LLVMBuildBitCast(self.b, h, self.i16ty(), self.n());
        llvm::core::LLVMBuildZExt(self.b, b16, self.i32t, self.n())
    }
    unsafe fn fmul_f32(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildFMul(self.b, a, b, self.n())
    }
    unsafe fn fdiv_f32(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildFDiv(self.b, a, b, self.n())
    }

    // ---- source operands -------------------------------------------------
    unsafe fn src_u32(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.ci32(*v),
            SourceOperand::IntegerConstant(v) => self.ci32(*v as u32),
            SourceOperand::ScalarRegister(r) => self.ld_sgpr32(*r as u32),
            SourceOperand::VectorRegister(r) => self.ld_vgpr32(*r as u32),
            SourceOperand::FloatConstant(v) => self.ci32((*v as f32).to_bits()),
            SourceOperand::PrivateBase => llvm::core::LLVMBuildTrunc(self.b, self.scratch_base, self.i32t, self.n()),
        }
    }
    unsafe fn src_u64(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.ci64(*v as u64),
            SourceOperand::IntegerConstant(v) => self.ci64(*v),
            SourceOperand::ScalarRegister(r) => self.ld_sgpr64(*r as u32),
            SourceOperand::VectorRegister(r) => self.ld_vgpr64(*r as u32),
            SourceOperand::PrivateBase => self.scratch_base,
            SourceOperand::FloatConstant(v) => self.ci64(v.to_bits()),
        }
    }
    unsafe fn src_f64(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.cf64(f64::from_bits((*v as u64) << 32)),
            SourceOperand::IntegerConstant(v) => self.cf64(f64::from_bits((*v as u64) << 32)),
            SourceOperand::FloatConstant(v) => self.cf64(*v),
            SourceOperand::ScalarRegister(r) => {
                let u = self.ld_sgpr64(*r as u32);
                llvm::core::LLVMBuildBitCast(self.b, u, self.f64t, self.n())
            }
            SourceOperand::VectorRegister(r) => self.ld_vgpr_f64(*r as u32),
            SourceOperand::PrivateBase => panic!("f64 from private base"),
        }
    }

    unsafe fn absneg_f64(&self, v: LLVMValueRef, abs: u8, neg: u8, idx: u32) -> LLVMValueRef {
        let mut v = v;
        if (abs >> idx) & 1 != 0 {
            // fabs via llvm.fabs.f64
            v = self.call("llvm.fabs.f64", self.f64t, &[self.f64t], &[v]);
        }
        if (neg >> idx) & 1 != 0 {
            v = llvm::core::LLVMBuildFNeg(self.b, v, self.n());
        }
        v
    }
    unsafe fn absneg_f32(&self, v: LLVMValueRef, abs: u8, neg: u8, idx: u32) -> LLVMValueRef {
        let mut v = v;
        if (abs >> idx) & 1 != 0 {
            v = self.call("llvm.fabs.f32", self.f32t, &[self.f32t], &[v]);
        }
        if (neg >> idx) & 1 != 0 {
            v = llvm::core::LLVMBuildFNeg(self.b, v, self.n());
        }
        v
    }

    // VCC bit 0 (single lane) as i1: (vcc & 1) != 0
    unsafe fn vcc_bit(&self) -> LLVMValueRef {
        let vcc = self.ld_sgpr32(VCC);
        let m = llvm::core::LLVMBuildAnd(self.b, vcc, self.ci32(1), self.n());
        llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, m, self.ci32(0), self.n())
    }
    // Bit 0 of any lane-mask register (EXEC/VCC live as i1; others as i32) as i1.
    unsafe fn mask_bit(&self, reg: u32) -> LLVMValueRef {
        let m = llvm::core::LLVMBuildAnd(self.b, self.ld_sgpr32(reg), self.ci32(1), self.n());
        llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, m, self.ci32(0), self.n())
    }
    // Store an i1 into a lane-mask register (VCC/EXEC/SGPR). The kernel's code
    // was compiled for a 32-lane wavefront and manipulates 32-bit masks
    // (e.g. `~EXEC & m`). To make that logic correct for our single active lane
    // (lane 0), we *broadcast* lane 0's bit across all 32 bits: a mask is always
    // 0x00000000 or 0xFFFFFFFF, exactly as if all 32 lanes agreed. (sext, not
    // zext.)
    unsafe fn st_mask(&self, reg: u32, bit_i1: LLVMValueRef) {
        let z = llvm::core::LLVMBuildZExt(self.b, bit_i1, self.i32t, self.n());
        self.st_sgpr32(reg, z);
    }
}

// =====================================================================
//  Public entry
// =====================================================================

pub fn compile_program(program: &ScalarProgram, num_vgprs: usize) -> ScalarKernel {
    unsafe { compile_inner(program, num_vgprs, false, true, false) }
}

pub fn compile_program_writeback(program: &ScalarProgram, num_vgprs: usize) -> ScalarKernel {
    unsafe { compile_inner(program, num_vgprs, true, false, false) }
}

/// Compile a barrier-split program for the cooperative workgroup scheduler. The
/// input must already have gone through [`super::ir::split_at_barriers`]. The
/// resulting function has signature
/// `extern "C" fn(*mut u32 /*sgprs[129]*/, *mut u32 /*vgprs*/, u64 /*scratch*/,
///                u64 /*lds*/, *mut u32 /*spill*/, u64 /*resume_pc*/) -> u64`
/// (see [`CoopKernel`]).
pub fn compile_cooperative(program: &ScalarProgram, num_vgprs: usize) -> CoopKernel {
    let sk = unsafe { compile_inner(program, num_vgprs, true, false, true) };
    CoopKernel { addr: sk.addr, num_vgprs: sk.num_vgprs, entry_pc: program.entry_pc }
}

unsafe fn compile_inner(
    program: &ScalarProgram,
    num_vgprs: usize,
    writeback: bool,
    force_exec: bool,
    coop: bool,
) -> ScalarKernel {
    // VGPR slots: allocate a safe upper bound (RDNA max is 256) since the
    // granulated descriptor count can underestimate the actual max index.
    let num_vgprs = num_vgprs.max(256);

    llvm::target::LLVM_InitializeNativeTarget();
    llvm::target::LLVM_InitializeNativeAsmParser();
    llvm::target::LLVM_InitializeNativeAsmPrinter();

    let ctx = llvm::core::LLVMContextCreate();
    let module = llvm::core::LLVMModuleCreateWithNameInContext(b"scalar_kernel\0".as_ptr() as *const _, ctx);
    let b = llvm::core::LLVMCreateBuilderInContext(ctx);

    let i1 = llvm::core::LLVMInt1TypeInContext(ctx);
    let i8 = llvm::core::LLVMInt8TypeInContext(ctx);
    let i32t = llvm::core::LLVMInt32TypeInContext(ctx);
    let i64t = llvm::core::LLVMInt64TypeInContext(ctx);
    let f32t = llvm::core::LLVMFloatTypeInContext(ctx);
    let f64t = llvm::core::LLVMDoubleTypeInContext(ctx);
    let ptr = llvm::core::LLVMPointerTypeInContext(ctx, 0);
    let void = llvm::core::LLVMVoidTypeInContext(ctx);

    // Non-coop: `void kernel(u32* sgprs, u32* vgprs, u64 scratch)`.
    // Coop:     `i64  kernel(u32* sgprs, u32* vgprs, u64 scratch, u64 lds,
    //                        u32* spill, i64 resume)`.
    let func = if coop {
        let mut params = [ptr, ptr, i64t, i64t, ptr, i64t];
        let fty = llvm::core::LLVMFunctionType(i64t, params.as_mut_ptr(), 6, 0);
        llvm::core::LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, fty)
    } else {
        let mut params = [ptr, ptr, i64t];
        let fty = llvm::core::LLVMFunctionType(void, params.as_mut_ptr(), 3, 0);
        llvm::core::LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, fty)
    };

    let sgprs_p = llvm::core::LLVMGetParam(func, 0);
    let vgprs_p = llvm::core::LLVMGetParam(func, 1);
    let scratch_base = llvm::core::LLVMGetParam(func, 2);
    let lds_base = if coop {
        llvm::core::LLVMGetParam(func, 3)
    } else {
        llvm::core::LLVMGetUndef(i64t)
    };
    let spill_base = if coop {
        llvm::core::LLVMGetParam(func, 4)
    } else {
        llvm::core::LLVMGetUndef(ptr)
    };

    let entry = llvm::core::LLVMAppendBasicBlockInContext(ctx, func, b"entry\0".as_ptr() as *const _);
    llvm::core::LLVMPositionBuilderAtEnd(b, entry);

    // Allocate register slots and the SCC flag.
    let mut sgpr = Vec::with_capacity(128);
    for _ in 0..128 {
        sgpr.push(llvm::core::LLVMBuildAlloca(b, i32t, b"\0".as_ptr() as *const _));
    }
    let mut vgpr = Vec::with_capacity(num_vgprs);
    for _ in 0..num_vgprs {
        vgpr.push(llvm::core::LLVMBuildAlloca(b, i32t, b"\0".as_ptr() as *const _));
    }
    // Parallel `double` shadow per VGPR pair (low reg). +1 so the high half of
    // the last pair has a slot.
    let mut vgpr_f64 = Vec::with_capacity(num_vgprs + 1);
    for _ in 0..num_vgprs + 1 {
        vgpr_f64.push(llvm::core::LLVMBuildAlloca(b, f64t, b"\0".as_ptr() as *const _));
    }
    let scc = llvm::core::LLVMBuildAlloca(b, i1, b"\0".as_ptr() as *const _);
    let exec_i1 = llvm::core::LLVMBuildAlloca(b, i1, b"\0".as_ptr() as *const _);
    let vcc_i1 = llvm::core::LLVMBuildAlloca(b, i1, b"\0".as_ptr() as *const _);
    let mut sgpr_i64 = Vec::with_capacity(129);
    for _ in 0..129 {
        sgpr_i64.push(llvm::core::LLVMBuildAlloca(b, i64t, b"\0".as_ptr() as *const _));
    }
    let bvh_scratch = llvm::core::LLVMBuildArrayAlloca(b, i32t, llvm::core::LLVMConstInt(i32t, 10, 0), b"\0".as_ptr() as *const _);
    let spill_base = if coop {
        spill_base
    } else {
        llvm::core::LLVMBuildArrayAlloca(
            b,
            i32t,
            llvm::core::LLVMConstInt(i32t, COOP_SPILL_SLOTS as u64, 0),
            b"\0".as_ptr() as *const _,
        )
    };

    let cg = Cg {
        ctx, module, b, func, scratch_base,
        sgpr, vgpr, scc, i1, i8, i32t, i64t, f32t, f64t, ptr,
        predicate: std::cell::Cell::new(false),
        mask_def: std::cell::RefCell::new(BTreeMap::new()),
        vgpr_f64,
        f64_fresh: std::cell::Cell::new([0; 2]),
        exec_i1,
        vcc_i1,
        sgpr_i64,
        sgpr_fresh: std::cell::Cell::new(0),
        writeback,
        writeback_vgprs: num_vgprs,
        coop,
        lds_base,
        spill_base,
        spill: std::cell::RefCell::new(BTreeMap::new()),
        bvh_scratch,
    };

    // Initialize register slots from the incoming pointers (unpredicated).
    for i in 0..128u32 {
        let gep = llvm::core::LLVMBuildGEP2(b, i32t, sgprs_p, [cg.ci32(i)].as_mut_ptr(), 1, cg.n());
        let v = llvm::core::LLVMBuildLoad2(b, i32t, gep, cg.n());
        cg.st_sgpr32(i, v);
    }
    for i in 0..num_vgprs as u32 {
        let gep = llvm::core::LLVMBuildGEP2(b, i32t, vgprs_p, [cg.ci32(i)].as_mut_ptr(), 1, cg.n());
        let v = llvm::core::LLVMBuildLoad2(b, i32t, gep, cg.n());
        cg.st_vgpr32(i, v);
    }
    if force_exec {
        cg.st_sgpr32(EXEC, cg.ci32(1));
    }
    if coop {
        // Resume-capable: reload SCC from the persisted slot (sgprs[128]); the
        // scheduler seeds it to 0 before the first call. EXEC/VCC ride the sgprs
        // buffer (indices 126/106) and are reloaded by the init loop above.
        let gep = llvm::core::LLVMBuildGEP2(b, i32t, sgprs_p, [cg.ci32(128)].as_mut_ptr(), 1, cg.n());
        let s = llvm::core::LLVMBuildLoad2(b, i32t, gep, cg.n());
        cg.st_scc_nz(s);
    } else {
        llvm::core::LLVMBuildStore(b, llvm::core::LLVMConstInt(i1, 0, 0), scc);
    }

    // From here on, predicate vector writes on EXEC bit 0.
    cg.predicate.set(true);

    // Create a basic block per scalar block.
    let mut bbs: BTreeMap<usize, LLVMBasicBlockRef> = BTreeMap::new();
    for (&pc, _) in &program.blocks {
        let name = cstr(&format!("b{:x}", pc));
        bbs.insert(pc, llvm::core::LLVMAppendBasicBlockInContext(ctx, func, name.as_ptr()));
    }

    if coop {
        // Resume dispatch: jump to the block named by `resume_pc`. The first call
        // passes `entry_pc` (→ default); every barrier's post-block pc is a case.
        let resume = llvm::core::LLVMGetParam(func, 5);
        let mut targets: Vec<usize> = Vec::new();
        for block in program.blocks.values() {
            if let Terminator::Barrier { resume } = block.term {
                targets.push(resume);
            }
        }
        let sw = llvm::core::LLVMBuildSwitch(b, resume, bbs[&program.entry_pc], targets.len() as u32);
        for pc in targets {
            llvm::core::LLVMAddCase(sw, cg.ci64(pc as u64), bbs[&pc]);
        }
    } else {
        llvm::core::LLVMBuildBr(b, bbs[&program.entry_pc]);
    }

    // De-SIMT lane-active analysis: where EXEC[0] is provably 1, the SIMT mask is
    // inert, so vector writes/compares need no predication.
    let active = super::active::analyze_states(program);
    let f64_fresh_in = super::freshness::analyze(program);
    let sgpr_fresh_in = super::freshness::analyze_sgpr(program);

    for (&pc, block) in &program.blocks {
        llvm::core::LLVMPositionBuilderAtEnd(b, bbs[&pc]);
        cg.set_f64_fresh(f64_fresh_in[&pc]);
        cg.sgpr_fresh.set(sgpr_fresh_in[&pc]);
        let states = super::active::body_active_states(block, active[&pc]);
        cg.mask_def.borrow_mut().clear();
        for (idx, inst) in block.body.iter().enumerate() {
            // Predicate this instruction's vector writes/compares unless the lane
            // is provably active here (then the mask is a no-op and we drop it).
            cg.predicate.set(!states[idx]);
            cg.emit_inst(inst);
        }
        cg.emit_term(&block.term, &bbs);
    }

    finalize(ctx, module, func, num_vgprs)
}

// =====================================================================
//  Terminators
// =====================================================================

impl Cg {
    unsafe fn emit_writeback(&self, num_vgprs: usize) {
        for i in 0..128u32 {
            let gep = llvm::core::LLVMBuildGEP2(
                self.b,
                self.i32t,
                llvm::core::LLVMGetParam(self.func, 0),
                [self.ci32(i)].as_mut_ptr(),
                1,
                self.n(),
            );
            let v = self.ld_sgpr32(i);
            llvm::core::LLVMBuildStore(self.b, v, gep);
        }
        for i in 0..num_vgprs as u32 {
            let gep = llvm::core::LLVMBuildGEP2(
                self.b,
                self.i32t,
                llvm::core::LLVMGetParam(self.func, 1),
                [self.ci32(i)].as_mut_ptr(),
                1,
                self.n(),
            );
            let v = self.ld_vgpr32(i);
            llvm::core::LLVMBuildStore(self.b, v, gep);
        }
    }

    /// Persist SCC into the reserved sgprs[128] slot (cooperative resume state).
    unsafe fn emit_scc_writeback(&self) {
        let z = llvm::core::LLVMBuildZExt(self.b, self.ld_scc(), self.i32t, self.n());
        let gep = llvm::core::LLVMBuildGEP2(
            self.b, self.i32t, llvm::core::LLVMGetParam(self.func, 0),
            [self.ci32(128)].as_mut_ptr(), 1, self.n(),
        );
        llvm::core::LLVMBuildStore(self.b, z, gep);
    }

    unsafe fn emit_term(&self, term: &Terminator, bbs: &BTreeMap<usize, LLVMBasicBlockRef>) {
        match term {
            Terminator::Return => {
                if self.coop {
                    // End of work-item: persist state, return the DONE sentinel.
                    self.emit_writeback(self.writeback_vgprs);
                    self.emit_scc_writeback();
                    llvm::core::LLVMBuildRet(self.b, self.ci64(u64::MAX));
                } else {
                    if self.writeback {
                        self.emit_writeback(self.writeback_vgprs);
                    }
                    llvm::core::LLVMBuildRetVoid(self.b);
                }
            }
            Terminator::Barrier { resume } => {
                // Yield: persist full register/SCC state and return the resume pc
                // so the scheduler re-enters here after every work-item syncs.
                self.emit_writeback(self.writeback_vgprs);
                self.emit_scc_writeback();
                llvm::core::LLVMBuildRet(self.b, self.ci64(*resume as u64));
            }
            Terminator::Jump(t) => {
                llvm::core::LLVMBuildBr(self.b, bbs[t]);
            }
            Terminator::Branch { cond, taken, fallthrough } => {
                let taken_cond = self.taken_cond(*cond);
                llvm::core::LLVMBuildCondBr(self.b, taken_cond, bbs[taken], bbs[fallthrough]);
            }
        }
    }

    unsafe fn taken_cond(&self, cond: Cond) -> LLVMValueRef {
        use llvm::LLVMIntPredicate::*;
        // The kernel was compiled for a 32-lane wavefront, and its EXEC/VCC
        // branch conditions test the whole 32-bit lane mask (including
        // hardcoded `-1` = "all lanes"). This function executes only lane 0, so
        // a branch must depend only on lane 0's bit. Testing `(mask & 1)` recovers
        // scalar control flow: e.g. a wavefront reconvergence loop on
        // `~EXEC & 0xFFFFFFFF` correctly terminates once lane 0 is active
        // (`~EXEC & 1 == 0`), instead of spinning on the 31 nonexistent lanes.
        match cond {
            Cond::ExecZ => {
                let e = self.b_and(self.ld_sgpr32(EXEC), self.ci32(1));
                llvm::core::LLVMBuildICmp(self.b, LLVMIntEQ, e, self.ci32(0), self.n())
            }
            Cond::ExecNz => {
                let e = self.b_and(self.ld_sgpr32(EXEC), self.ci32(1));
                llvm::core::LLVMBuildICmp(self.b, LLVMIntNE, e, self.ci32(0), self.n())
            }
            Cond::VccZ => {
                let e = self.b_and(self.ld_sgpr32(VCC), self.ci32(1));
                llvm::core::LLVMBuildICmp(self.b, LLVMIntEQ, e, self.ci32(0), self.n())
            }
            Cond::VccNz => {
                let e = self.b_and(self.ld_sgpr32(VCC), self.ci32(1));
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
    num_vgprs: usize,
) -> ScalarKernel {
    unsafe {
        // Verify.
        let mut err = std::ptr::null_mut();
        if llvm::analysis::LLVMVerifyModule(
            module,
            llvm::analysis::LLVMVerifierFailureAction::LLVMPrintMessageAction,
            &mut err,
        ) != 0
        {
            if !err.is_null() {
                let s = std::ffi::CStr::from_ptr(err).to_string_lossy().into_owned();
                panic!("scalar module failed verification:\n{}", s);
            }
        }

        // Target machine.
        let triple = llvm::target_machine::LLVMGetDefaultTargetTriple();
        let mut target = std::ptr::null_mut();
        let mut terr = std::ptr::null_mut();
        llvm::target_machine::LLVMGetTargetFromTriple(triple, &mut target, &mut terr);
        let cpu = llvm::target_machine::LLVMGetHostCPUName();
        let feat_host = llvm::target_machine::LLVMGetHostCPUFeatures();
        // The kernel is scalar f64, so AVX-512 brings no vector benefit but makes
        // LLVM lower every f64 `select` (V_CNDMASK) to a `kmovd` + masked
        // `vmovsd` (native smallpt uses cheaper `vblendvpd`/cmov). Measured ~10%
        // faster with AVX-512 disabled, so it is always disabled here.
        let host = std::ffi::CStr::from_ptr(feat_host).to_string_lossy();
        let s = format!("{},-avx512f,-avx512vl,-avx512dq,-avx512bw,-avx512cd", host);
        let feat_cstr = std::ffi::CString::new(s).unwrap();
        let feat = feat_cstr.as_ptr();
        // NB: Small code model + PIC makes f64 constants load RIP-relative
        // (eliminates the `movabs $abs; vmovsd (%reg)` form, ~3% of cycles in
        // `movabs`), but it also defeats LLVM's LICM hoist of the loop-invariant
        // reciprocal in the sphere loop (vdivsd cycle-share 0.4% → 9.1%), a net
        // wash-to-regression (cycles 1330.6B → 1336.9B). So keep JITDefault.
        let tm = llvm::target_machine::LLVMCreateTargetMachine(
            target, triple, cpu, feat,
            llvm::target_machine::LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            llvm::target_machine::LLVMRelocMode::LLVMRelocDefault,
            llvm::target_machine::LLVMCodeModel::LLVMCodeModelJITDefault,
        );

        // Optimize: promote register slots to SSA (mem2reg) then O3 cleanup.
        // The full module-level `default<O3>` lets LLVM apply its most aggressive
        // cross-block reasoning (e.g. coalescing the paired i32 register phis +
        // f64 reconstruction back toward a single value).
        let opts = llvm::transforms::pass_builder::LLVMCreatePassBuilderOptions();
        let passes_str: std::ffi::CString = std::ffi::CString::new("default<O3>").unwrap();
        let perr = llvm::transforms::pass_builder::LLVMRunPasses(
            module, passes_str.as_ptr(), tm, opts,
        );
        let _ = func;
        if !perr.is_null() {
            let msg = llvm::error::LLVMGetErrorMessage(perr);
            let s = std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned();
            panic!("scalar passes failed: {}", s);
        }
        // ORC LLJIT.
        let jit_builder = llvm::orc2::lljit::LLVMOrcCreateLLJITBuilder();
        let jtmb = llvm::orc2::LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm);
        llvm::orc2::lljit::LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(jit_builder, jtmb);
        let mut jit = std::ptr::null_mut();
        let e = llvm::orc2::lljit::LLVMOrcCreateLLJIT(&mut jit, jit_builder);
        if !e.is_null() {
            panic!("create LLJIT failed");
        }
        let dylib = llvm::orc2::lljit::LLVMOrcLLJITGetMainJITDylib(jit);
        let gp = llvm::orc2::lljit::LLVMOrcLLJITGetGlobalPrefix(jit);

        // Resolve process symbols (for runtime helpers like v_trig_preop_f64).
        let mut dg = std::ptr::null_mut();
        llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(&mut dg, gp, None, std::ptr::null_mut());
        llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, dg);

        let lib_path: &[u8] = if cfg!(debug_assertions) {
            b"target/debug/libamdgpu_sim.so\0"
        } else {
            b"target/release/libamdgpu_sim.so\0"
        };
        let mut dg2 = std::ptr::null_mut();
        llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(
            &mut dg2, lib_path.as_ptr() as *const _, gp, None, std::ptr::null_mut());
        llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, dg2);

        let tsctx = llvm::orc2::LLVMOrcCreateNewThreadSafeContext();
        let tsm = llvm::orc2::LLVMOrcCreateNewThreadSafeModule(module, tsctx);
        let e = llvm::orc2::lljit::LLVMOrcLLJITAddLLVMIRModule(jit, dylib, tsm);
        if !e.is_null() {
            panic!("add module failed");
        }

        let mut addr = 0u64;
        let e = llvm::orc2::lljit::LLVMOrcLLJITLookup(jit, &mut addr, b"kernel\0".as_ptr() as *const _);
        if !e.is_null() {
            panic!("lookup kernel failed");
        }

        // Intentionally leak `jit` so the compiled code stays mapped.
        std::mem::forget(Box::new(jit));
        let _ = ctx;

        ScalarKernel { addr, num_vgprs }
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
            InstFormat::SOPK(i) => self.emit_sopk(i),
            InstFormat::SOPC(i) => self.emit_sopc(i),
            InstFormat::SMEM(i) => self.emit_smem(i),
            InstFormat::VGLOBAL(i) => self.emit_vglobal(i),
            InstFormat::VFLAT(i) => self.emit_vflat(i),
            InstFormat::VIMAGE(i) => self.emit_vimage(i),
            InstFormat::VSAMPLE(i) => self.emit_vsample(i),
            InstFormat::VSCRATCH(i) => self.emit_vscratch(i),
            InstFormat::DS(i) => self.emit_ds(i),
            other => panic!("scalar: unsupported instruction {:?}", other),
        }
    }

    // ---- helpers ---------------------------------------------------------
    unsafe fn b_and(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildAnd(self.b, a, b, self.n())
    }
    unsafe fn b_or(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildOr(self.b, a, b, self.n())
    }
    unsafe fn b_xor(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildXor(self.b, a, b, self.n())
    }
    unsafe fn b_add(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildAdd(self.b, a, b, self.n())
    }
    unsafe fn b_sub(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildSub(self.b, a, b, self.n())
    }
    unsafe fn b_not(&self, a: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildNot(self.b, a, self.n())
    }
    unsafe fn shl(&self, a: LLVMValueRef, amt: LLVMValueRef) -> LLVMValueRef {
        let amt = self.b_and(amt, self.ci32(31));
        llvm::core::LLVMBuildShl(self.b, a, amt, self.n())
    }
    unsafe fn lshr(&self, a: LLVMValueRef, amt: LLVMValueRef) -> LLVMValueRef {
        let amt = self.b_and(amt, self.ci32(31));
        llvm::core::LLVMBuildLShr(self.b, a, amt, self.n())
    }
    /// FP instructions carry no fast-math flags (bit-exact with the masked
    /// backend); pass-through kept so call sites read uniformly.
    unsafe fn fmf(&self, v: LLVMValueRef) -> LLVMValueRef {
        v
    }
    unsafe fn fmuladd(&self, a: LLVMValueRef, b: LLVMValueRef, c: LLVMValueRef) -> LLVMValueRef {
        let v = self.call("llvm.fmuladd.f64", self.f64t, &[self.f64t, self.f64t, self.f64t], &[a, b, c]);
        self.fmf(v)
    }
    unsafe fn sqrt(&self, a: LLVMValueRef) -> LLVMValueRef {
        let v = self.call("llvm.sqrt.f64", self.f64t, &[self.f64t], &[a]);
        self.fmf(v)
    }
    unsafe fn floor(&self, a: LLVMValueRef) -> LLVMValueRef {
        self.call("llvm.floor.f64", self.f64t, &[self.f64t], &[a])
    }
    unsafe fn fdiv(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        self.fmf(llvm::core::LLVMBuildFDiv(self.b, a, b, self.n()))
    }
    unsafe fn fmul(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        self.fmf(llvm::core::LLVMBuildFMul(self.b, a, b, self.n()))
    }
    unsafe fn ptr_at(&self, addr: LLVMValueRef, off: u64) -> LLVMValueRef {
        let a = self.b_add(addr, self.ci64(off));
        llvm::core::LLVMBuildIntToPtr(self.b, a, self.ptr, self.n())
    }
    /// Effective store address honoring EXEC predication: when predicating and
    /// the single lane is inactive (EXEC bit 0 == 0), a store must be a no-op, so
    /// redirect it to the always-mapped `bvh_scratch` throwaway. Vector stores in
    /// EXEC-masked regions (a `v_cmpx` guard with no branch) otherwise write to
    /// stale, predicated-away register values.
    unsafe fn store_addr(&self, addr: LLVMValueRef) -> LLVMValueRef {
        if self.predicate.get() {
            let active = llvm::core::LLVMBuildICmp(
                self.b,
                llvm::LLVMIntPredicate::LLVMIntNE,
                self.b_and(self.ld_sgpr32(EXEC), self.ci32(1)),
                self.ci32(0),
                self.n(),
            );
            let dummy = llvm::core::LLVMBuildPtrToInt(self.b, self.bvh_scratch, self.i64t, self.n());
            llvm::core::LLVMBuildSelect(self.b, active, addr, dummy, self.n())
        } else {
            addr
        }
    }

    // ---- VOP1 ------------------------------------------------------------
    unsafe fn emit_vop1(&self, i: &VOP1) {
        match i.op {
            I::V_MOV_B32 => {
                let v = self.src_u32(&i.src0);
                self.st_vgpr32(i.vdst as u32, v);
            }
            // Broadcast lane 0 to an SGPR. In the per-work-item model each lane's
            // value stands in for "the first lane"; correct when the source is
            // uniform (the compiler's use here) — otherwise it would be a genuine
            // cross-lane read.
            I::V_READFIRSTLANE_B32 => {
                let v = self.src_u32(&i.src0);
                self.st_sgpr32(i.vdst as u32, v);
            }
            I::V_RCP_IFLAG_F32 | I::V_RCP_F32 => {
                let s = self.src_f32(&i.src0);
                let v = self.fdiv_f32(self.cf32(1.0), s);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(v));
            }
            I::V_SQRT_F32 => {
                let v = self.call("llvm.sqrt.f32", self.f32t, &[self.f32t], &[self.src_f32(&i.src0)]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(v));
            }
            I::V_RSQ_F32 => {
                let q = self.call("llvm.sqrt.f32", self.f32t, &[self.f32t], &[self.src_f32(&i.src0)]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(self.fdiv_f32(self.cf32(1.0), q)));
            }
            I::V_RNDNE_F32 | I::V_FLOOR_F32 | I::V_CEIL_F32 | I::V_TRUNC_F32 => {
                let name = match i.op {
                    I::V_RNDNE_F32 => "llvm.roundeven.f32", I::V_FLOOR_F32 => "llvm.floor.f32",
                    I::V_CEIL_F32 => "llvm.ceil.f32", _ => "llvm.trunc.f32",
                };
                let v = self.call(name, self.f32t, &[self.f32t], &[self.src_f32(&i.src0)]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(v));
            }
            I::V_CLZ_I32_U32 => {
                let x = self.src_u32(&i.src0);
                let lz = self.call("llvm.ctlz.i32", self.i32t, &[self.i32t, self.i1], &[x, llvm::core::LLVMConstInt(self.i1, 0, 0)]);
                let is0 = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntEQ, x, self.ci32(0), self.n());
                self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildSelect(self.b, is0, self.ci32(0xFFFF_FFFF), lz, self.n()));
            }
            I::V_FREXP_MANT_F32 => {
                let v = self.call("scalar_frexp_mant_f32", self.f32t, &[self.f32t], &[self.src_f32(&i.src0)]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(v));
            }
            I::V_FREXP_EXP_I32_F32 => {
                let v = self.call("scalar_frexp_exp_f32", self.i32t, &[self.f32t], &[self.src_f32(&i.src0)]);
                self.st_vgpr32(i.vdst as u32, v);
            }
            I::V_CVT_U32_F32 => {
                // ISA: out-of-range saturates, NaN→0 — llvm.fptoui.sat does exactly this.
                let s = self.src_f32(&i.src0);
                let v = self.call("llvm.fptoui.sat.i32.f32", self.i32t, &[self.f32t], &[s]);
                self.st_vgpr32(i.vdst as u32, v);
            }
            I::V_CVT_I32_F32 => {
                let v = self.call("llvm.fptosi.sat.i32.f32", self.i32t, &[self.f32t], &[self.src_f32(&i.src0)]);
                self.st_vgpr32(i.vdst as u32, v);
            }
            I::V_CVT_F32_I32 => {
                let f = llvm::core::LLVMBuildSIToFP(self.b, self.src_u32(&i.src0), self.f32t, self.n());
                self.st_vgpr32(i.vdst as u32, self.f32_bits(f));
            }
            I::V_CVT_F32_U32 => {
                let s = self.src_u32(&i.src0);
                let f = llvm::core::LLVMBuildUIToFP(self.b, s, self.f32t, self.n());
                self.st_vgpr32(i.vdst as u32, self.f32_bits(f));
            }
            I::V_CVT_F64_I32 => {
                let s = self.src_u32(&i.src0);
                let v = llvm::core::LLVMBuildSIToFP(self.b, s, self.f64t, self.n());
                self.st_vgpr_f64(i.vdst as u32, v);
            }
            I::V_CVT_F64_U32 => {
                let s = self.src_u32(&i.src0);
                let v = llvm::core::LLVMBuildUIToFP(self.b, s, self.f64t, self.n());
                self.st_vgpr_f64(i.vdst as u32, v);
            }
            I::V_CVT_I32_F64 => {
                let s = self.src_f64(&i.src0);
                let v = self.call("llvm.fptosi.sat.i32.f64", self.i32t, &[self.f64t], &[s]);
                self.st_vgpr32(i.vdst as u32, v);
            }
            I::V_RCP_F64 => {
                let s = self.src_f64(&i.src0);
                let v = self.fdiv(self.cf64(1.0), s);
                self.st_vgpr_f64(i.vdst as u32, v);
            }
            I::V_RSQ_F64 => {
                let s = self.src_f64(&i.src0);
                let q = self.sqrt(s);
                let v = self.fdiv(self.cf64(1.0), q);
                self.st_vgpr_f64(i.vdst as u32, v);
            }
            I::V_SQRT_F64 => {
                let s = self.src_f64(&i.src0);
                let v = self.sqrt(s);
                self.st_vgpr_f64(i.vdst as u32, v);
            }
            I::V_FRACT_F64 => {
                let s = self.src_f64(&i.src0);
                let f = self.floor(s);
                let v = self.fmf(llvm::core::LLVMBuildFSub(self.b, s, f, self.n()));
                self.st_vgpr_f64(i.vdst as u32, v);
            }
            I::V_RNDNE_F64 => {
                let s = self.src_f64(&i.src0);
                let v = self.call("llvm.roundeven.f64", self.f64t, &[self.f64t], &[s]);
                self.st_vgpr_f64(i.vdst as u32, v);
            }
            _ => panic!("scalar: unsupported VOP1 {:?}", i.op),
        }
    }

    // ---- VOP2 ------------------------------------------------------------
    unsafe fn emit_vop2(&self, i: &VOP2) {
        // f64 forms (no source modifiers in VOP2).
        match i.op {
            I::V_ADD_F64 => {
                let a = self.src_f64(&i.src0);
                let b = self.ld_vgpr_f64(i.vsrc1 as u32);
                let r = self.fmf(llvm::core::LLVMBuildFAdd(self.b, a, b, self.n()));
                self.st_vgpr_f64(i.vdst as u32, r);
                return;
            }
            I::V_MUL_F64 => {
                let a = self.src_f64(&i.src0);
                let b = self.ld_vgpr_f64(i.vsrc1 as u32);
                let r = self.fmul(a, b);
                self.st_vgpr_f64(i.vdst as u32, r);
                return;
            }
            I::V_MAX_NUM_F64 | I::V_MIN_NUM_F64 => {
                let a = self.src_f64(&i.src0);
                let b = self.ld_vgpr_f64(i.vsrc1 as u32);
                let name = if matches!(i.op, I::V_MAX_NUM_F64) { "llvm.maxnum.f64" } else { "llvm.minnum.f64" };
                let r = self.call(name, self.f64t, &[self.f64t, self.f64t], &[a, b]);
                self.st_vgpr_f64(i.vdst as u32, r);
                return;
            }
            I::V_LSHLREV_B64 => {
                let amt = self.b_and(self.src_u32(&i.src0), self.ci32(63));
                let amt = llvm::core::LLVMBuildZExt(self.b, amt, self.i64t, self.n());
                let v = self.ld_vgpr64(i.vsrc1 as u32);
                let r = llvm::core::LLVMBuildShl(self.b, v, amt, self.n());
                self.st_vgpr64(i.vdst as u32, r);
                return;
            }
            I::V_MUL_F32 => {
                let a = self.src_f32(&i.src0);
                let b = llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(i.vsrc1 as u32), self.f32t, self.n());
                let r = self.fmul_f32(a, b);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
                return;
            }
            I::V_ADD_F32 | I::V_SUB_F32 | I::V_SUBREV_F32 => {
                let a = self.src_f32(&i.src0);
                let b = llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(i.vsrc1 as u32), self.f32t, self.n());
                let r = match i.op {
                    I::V_ADD_F32 => llvm::core::LLVMBuildFAdd(self.b, a, b, self.n()),
                    I::V_SUB_F32 => llvm::core::LLVMBuildFSub(self.b, a, b, self.n()),
                    _ => llvm::core::LLVMBuildFSub(self.b, b, a, self.n()),
                };
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
                return;
            }
            I::V_FMAC_F32 => {
                let s0 = self.src_f32(&i.src0);
                let s1 = llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(i.vsrc1 as u32), self.f32t, self.n());
                let d = llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(i.vdst as u32), self.f32t, self.n());
                let r = self.call("llvm.fma.f32", self.f32t, &[self.f32t, self.f32t, self.f32t], &[s0, s1, d]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
                return;
            }
            I::V_FMAMK_F32 => {
                let s0 = self.src_f32(&i.src0);
                let s1 = llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(i.vsrc1 as u32), self.f32t, self.n());
                let k = llvm::core::LLVMConstBitCast(self.ci32(i.literal_constant.unwrap()), self.f32t);
                let r = self.call("llvm.fma.f32", self.f32t, &[self.f32t, self.f32t, self.f32t], &[s0, k, s1]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
                return;
            }
            I::V_FMAAK_F32 => {
                let s0 = self.src_f32(&i.src0);
                let s1 = llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(i.vsrc1 as u32), self.f32t, self.n());
                let k = llvm::core::LLVMConstBitCast(self.ci32(i.literal_constant.unwrap()), self.f32t);
                let r = self.call("llvm.fma.f32", self.f32t, &[self.f32t, self.f32t, self.f32t], &[s0, s1, k]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
                return;
            }
            I::V_ADD_CO_CI_U32 => {
                let s0 = self.zext64(self.src_u32(&i.src0));
                let s1 = self.zext64(self.ld_vgpr32(i.vsrc1 as u32));
                let cin = llvm::core::LLVMBuildZExt(self.b, self.vcc_bit(), self.i64t, self.n());
                let sum = self.b_add(self.b_add(s0, s1), cin);
                self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildTrunc(self.b, sum, self.i32t, self.n()));
                let cout = llvm::core::LLVMBuildLShr(self.b, sum, self.ci64(32), self.n());
                let cout = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, cout, self.ci64(0), self.n());
                self.st_mask(VCC, cout);
                return;
            }
            I::V_SUB_CO_CI_U32 | I::V_SUBREV_CO_CI_U32 => {
                let s0 = self.zext64(self.src_u32(&i.src0));
                let s1 = self.zext64(self.ld_vgpr32(i.vsrc1 as u32));
                let (a, b) = if matches!(i.op, I::V_SUBREV_CO_CI_U32) { (s1, s0) } else { (s0, s1) };
                let cin = llvm::core::LLVMBuildZExt(self.b, self.vcc_bit(), self.i64t, self.n());
                let diff = self.b_sub(self.b_sub(a, b), cin);
                self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildTrunc(self.b, diff, self.i32t, self.n()));
                let bo = llvm::core::LLVMBuildLShr(self.b, diff, self.ci64(32), self.n());
                let bo = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, bo, self.ci64(0), self.n());
                self.st_mask(VCC, bo);
                return;
            }
            _ => {}
        }
        let s0 = self.src_u32(&i.src0);
        let s1 = self.ld_vgpr32(i.vsrc1 as u32);
        let r = match i.op {
            I::V_ADD_NC_U32 => self.b_add(s0, s1),
            I::V_SUB_NC_U32 => self.b_sub(s0, s1),
            I::V_SUBREV_NC_U32 => self.b_sub(s1, s0),
            I::V_ADD_NC_U16 => self.b_and(self.b_add(s0, s1), self.ci32(0xffff)),
            I::V_AND_B32 => self.b_and(s0, s1),
            I::V_XOR_B32 => self.b_xor(s0, s1),
            I::V_OR_B32 => self.b_or(s0, s1),
            I::V_LSHLREV_B32 => self.shl(s1, s0),
            I::V_LSHRREV_B32 => self.lshr(s1, s0),
            I::V_MAX_U32 => {
                let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntUGT, s0, s1, self.n());
                llvm::core::LLVMBuildSelect(self.b, c, s0, s1, self.n())
            }
            I::V_MIN_U32 => {
                let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntULT, s0, s1, self.n());
                llvm::core::LLVMBuildSelect(self.b, c, s0, s1, self.n())
            }
            I::V_CNDMASK_B32 => {
                // implicit VCC condition
                let c = self.vcc_bit();
                llvm::core::LLVMBuildSelect(self.b, c, s1, s0, self.n())
            }
            _ => panic!("scalar: unsupported VOP2 {:?}", i.op),
        };
        self.st_vgpr32(i.vdst as u32, r);
    }

    // ---- VOP3 ------------------------------------------------------------
    unsafe fn emit_vop3(&self, i: &VOP3) {
        match i.op {
            // ----- integer -----
            I::V_ADD_NC_U32 => self.vop3_int(i, |c, a, b, _| c.b_add(a, b)),
            I::V_SUB_NC_U32 => self.vop3_int(i, |c, a, b, _| c.b_sub(a, b)),
            I::V_SUBREV_NC_U32 => self.vop3_int(i, |c, a, b, _| c.b_sub(b, a)),
            I::V_AND_B32 => self.vop3_int(i, |c, a, b, _| c.b_and(a, b)),
            I::V_XOR_B32 => self.vop3_int(i, |c, a, b, _| c.b_xor(a, b)),
            I::V_OR_B32 => self.vop3_int(i, |c, a, b, _| c.b_or(a, b)),
            I::V_LSHLREV_B32 => self.vop3_int(i, |c, a, b, _| c.shl(b, a)),
            I::V_LSHRREV_B32 => self.vop3_int(i, |c, a, b, _| c.lshr(b, a)),
            I::V_MUL_LO_U32 => self.vop3_int(i, |c, a, b, _| llvm::core::LLVMBuildMul(c.b, a, b, c.n())),
            I::V_MUL_HI_U32 => {
                let a = self.zext64(self.src_u32(&i.src0));
                let b = self.zext64(self.src_u32(&i.src1));
                let prod = llvm::core::LLVMBuildMul(self.b, a, b, self.n());
                let hi = llvm::core::LLVMBuildLShr(self.b, prod, self.ci64(32), self.n());
                self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildTrunc(self.b, hi, self.i32t, self.n()));
            }
            I::V_MAX_U32 => {
                let a = self.src_u32(&i.src0);
                let b = self.src_u32(&i.src1);
                let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntUGT, a, b, self.n());
                let r = llvm::core::LLVMBuildSelect(self.b, c, a, b, self.n());
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_MIN_U32 => {
                let a = self.src_u32(&i.src0);
                let b = self.src_u32(&i.src1);
                let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntULT, a, b, self.n());
                let r = llvm::core::LLVMBuildSelect(self.b, c, a, b, self.n());
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_LSHLREV_B64 => {
                let amt = self.b_and(self.src_u32(&i.src0), self.ci32(63));
                let amt = llvm::core::LLVMBuildZExt(self.b, amt, self.i64t, self.n());
                let v = self.src_u64(&i.src1);
                let r = llvm::core::LLVMBuildShl(self.b, v, amt, self.n());
                self.st_vgpr64(i.vdst as u32, r);
            }
            I::V_LSHRREV_B64 => {
                let amt = self.b_and(self.src_u32(&i.src0), self.ci32(63));
                let amt = llvm::core::LLVMBuildZExt(self.b, amt, self.i64t, self.n());
                let v = self.src_u64(&i.src1);
                self.st_vgpr64(i.vdst as u32, llvm::core::LLVMBuildLShr(self.b, v, amt, self.n()));
            }
            I::V_ADD3_U32 => {
                let a = self.src_u32(&i.src0);
                let b = self.src_u32(&i.src1);
                let cc = self.src_u32(&i.src2);
                let r = self.b_add(self.b_add(a, b), cc);
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_XOR3_B32 => {
                let a = self.src_u32(&i.src0);
                let b = self.src_u32(&i.src1);
                let cc = self.src_u32(&i.src2);
                let r = self.b_xor(self.b_xor(a, b), cc);
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_XAD_U32 => {
                let a = self.src_u32(&i.src0);
                let b = self.src_u32(&i.src1);
                let cc = self.src_u32(&i.src2);
                let r = self.b_add(self.b_xor(a, b), cc);
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_BFE_U32 => {
                let data = self.src_u32(&i.src0);
                let off = self.src_u32(&i.src1);
                let wid = self.src_u32(&i.src2);
                let wid = self.b_and(wid, self.ci32(31));
                let one = self.ci32(1);
                let mask = llvm::core::LLVMBuildSub(self.b, llvm::core::LLVMBuildShl(self.b, one, wid, self.n()), self.ci32(1), self.n());
                let r = self.b_and(self.lshr(data, off), mask);
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_CNDMASK_B32 => {
                // e64 form applies abs/neg (as floats) to s0/s1 and uses an
                // explicit condition operand (src2), matching the interpreter's
                // v_cndmask_b32_e64. abs=neg=0 is a bit-preserving no-op.
                let s0 = self.f32_bits(self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0));
                let s1 = self.f32_bits(self.absneg_f32(self.src_f32(&i.src1), i.abs, i.neg, 1));
                let s2 = self.src_u32(&i.src2);
                let m = self.b_and(s2, self.ci32(1));
                let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, m, self.ci32(0), self.n());
                let r = llvm::core::LLVMBuildSelect(self.b, c, s1, s0, self.n());
                self.st_vgpr32(i.vdst as u32, r);
            }
            // ----- f64 -----
            I::V_ADD_F64 => {
                let a = self.absneg_f64(self.src_f64(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f64(self.src_f64(&i.src1), i.abs, i.neg, 1);
                let r = self.fmf(llvm::core::LLVMBuildFAdd(self.b, a, b, self.n()));
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            I::V_MUL_F64 => {
                let a = self.absneg_f64(self.src_f64(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f64(self.src_f64(&i.src1), i.abs, i.neg, 1);
                let r = self.fmul(a, b);
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            I::V_FMA_F64 => {
                let a = self.absneg_f64(self.src_f64(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f64(self.src_f64(&i.src1), i.abs, i.neg, 1);
                let c = self.absneg_f64(self.src_f64(&i.src2), i.abs, i.neg, 2);
                let r = self.fmuladd(a, b, c);
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            I::V_MAX_NUM_F64 => {
                let a = self.absneg_f64(self.src_f64(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f64(self.src_f64(&i.src1), i.abs, i.neg, 1);
                let r = self.call("llvm.maxnum.f64", self.f64t, &[self.f64t, self.f64t], &[a, b]);
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            I::V_LDEXP_F64 => {
                let a = self.src_f64(&i.src0);
                let e = self.src_u32(&i.src1);
                let r = self.ldexp_inline(a, e);
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            I::V_DIV_SCALE_F64 => {
                let a = self.absneg_f64(self.src_f64(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f64(self.src_f64(&i.src1), i.abs, i.neg, 1);
                let c = self.absneg_f64(self.src_f64(&i.src2), i.abs, i.neg, 2);
                let r = self.fdiv(self.fmul(a, c), b);
                self.st_vgpr_f64(i.vdst as u32, r);
                // sdst (vcc) := 0 -- VOP3SD field; left unset in this prototype.
            }
            I::V_DIV_FIXUP_F64 => {
                let b = self.absneg_f64(self.src_f64(&i.src1), i.abs, i.neg, 1);
                let c = self.absneg_f64(self.src_f64(&i.src2), i.abs, i.neg, 2);
                let r = self.fdiv(c, b);
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            I::V_DIV_FMAS_F64 => {
                let a = self.src_f64(&i.src0);
                let b = self.src_f64(&i.src1);
                let c = self.src_f64(&i.src2);
                let fma = self.fmuladd(a, b, c);
                let scaled = self.fmul(fma, self.cf64(f64::from_bits(0x43F0000000000000)));
                let cond = self.vcc_bit();
                let r = llvm::core::LLVMBuildSelect(self.b, cond, scaled, fma, self.n());
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            I::V_DIV_FIXUP_F32 => {
                let a = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f32(self.src_f32(&i.src1), i.abs, i.neg, 1);
                let c = self.absneg_f32(self.src_f32(&i.src2), i.abs, i.neg, 2);
                let r = self.call("scalar_div_fixup_f32", self.f32t, &[self.f32t, self.f32t, self.f32t], &[a, b, c]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
            }
            I::V_DIV_FMAS_F32 => {
                let a = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f32(self.src_f32(&i.src1), i.abs, i.neg, 1);
                let c = self.absneg_f32(self.src_f32(&i.src2), i.abs, i.neg, 2);
                let fma = self.call("llvm.fma.f32", self.f32t, &[self.f32t, self.f32t, self.f32t], &[a, b, c]);
                // VCC: result *= 2^32. Interpreter computes 2^32 * fma; f32 mul is
                // commutative so the rounding is identical.
                let scaled = self.fmul_f32(llvm::core::LLVMConstBitCast(self.ci32(0x4F80_0000), self.f32t), fma);
                let cond = self.vcc_bit();
                let r = llvm::core::LLVMBuildSelect(self.b, cond, scaled, fma, self.n());
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
            }
            I::V_LDEXP_F32 => {
                let a = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let e = self.src_u32(&i.src1);
                let r = self.call("llvm.ldexp.f32.i32", self.f32t, &[self.f32t, self.i32t], &[a, e]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
            }
            I::V_TRIG_PREOP_F64 => {
                let a = self.src_f64(&i.src0);
                let s = self.src_u32(&i.src1);
                let r = self.call("v_trig_preop_f64", self.f64t, &[self.f64t, self.i32t], &[a, s]);
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            // ----- f32 compares encoded as VOP3 (dest = vdst sgpr mask) -----
            op if f32_pred(op).is_some() => {
                let (pred, invert) = f32_pred(op).unwrap();
                let a = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f32(self.src_f32(&i.src1), i.abs, i.neg, 1);
                let mut c = llvm::core::LLVMBuildFCmp(self.b, pred, a, b, self.n());
                if invert { c = self.b_not(c); }
                self.st_cmp(i.vdst as u32, c);
            }
            // ----- f64 compares encoded as VOP3 (dest = vdst sgpr mask) -----
            op if f64_pred(op).is_some() => {
                let (pred, invert) = f64_pred(op).unwrap();
                let a = self.absneg_f64(self.src_f64(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f64(self.src_f64(&i.src1), i.abs, i.neg, 1);
                let mut c = llvm::core::LLVMBuildFCmp(self.b, pred, a, b, self.n());
                if invert {
                    c = self.b_not(c);
                }
                self.st_cmp(i.vdst as u32, c);
            }
            I::V_CMP_CLASS_F64 => {
                let a = self.src_f64(&i.src0);
                let s = self.src_u32(&i.src1);
                let c = self.call("llvm.is.fpclass.f64", self.i1, &[self.f64t, self.i32t], &[a, s]);
                self.st_cmp(i.vdst as u32, c);
            }
            I::V_CMP_CLASS_F32 => {
                let a = self.src_f32(&i.src0);
                let s = self.src_u32(&i.src1);
                let c = self.call("llvm.is.fpclass.f32", self.i1, &[self.f32t, self.i32t], &[a, s]);
                self.st_cmp(i.vdst as u32, c);
            }
            op if int64_pred(op).is_some() => {
                let pred = int64_pred(op).unwrap();
                let a = self.src_u64(&i.src0);
                let b = self.src_u64(&i.src1);
                let c = llvm::core::LLVMBuildICmp(self.b, pred, a, b, self.n());
                self.st_cmp(i.vdst as u32, c);
            }
            op if int_pred(op).is_some() => {
                let pred = int_pred(op).unwrap();
                let a = self.src_u32(&i.src0);
                let b = self.src_u32(&i.src1);
                let c = llvm::core::LLVMBuildICmp(self.b, pred, a, b, self.n());
                self.st_cmp(i.vdst as u32, c);
            }
            // ----- more integer -----
            I::V_ASHRREV_I32 => self.vop3_int(i, |c, a, b, _| {
                let amt = c.b_and(a, c.ci32(31));
                llvm::core::LLVMBuildAShr(c.b, b, amt, c.n())
            }),
            I::V_LSHL_OR_B32 => {
                let s0 = self.src_u32(&i.src0);
                let s1 = self.b_and(self.src_u32(&i.src1), self.ci32(31));
                let s2 = self.src_u32(&i.src2);
                let r = self.b_or(llvm::core::LLVMBuildShl(self.b, s0, s1, self.n()), s2);
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_LSHL_ADD_U32 => {
                let s0 = self.src_u32(&i.src0);
                let s1 = self.b_and(self.src_u32(&i.src1), self.ci32(31));
                let s2 = self.src_u32(&i.src2);
                let r = self.b_add(llvm::core::LLVMBuildShl(self.b, s0, s1, self.n()), s2);
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_ADD_LSHL_U32 => {
                let s0 = self.src_u32(&i.src0);
                let s1 = self.src_u32(&i.src1);
                let s2 = self.b_and(self.src_u32(&i.src2), self.ci32(31));
                let r = llvm::core::LLVMBuildShl(self.b, self.b_add(s0, s1), s2, self.n());
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_AND_OR_B32 => {
                let r = self.b_or(self.b_and(self.src_u32(&i.src0), self.src_u32(&i.src1)), self.src_u32(&i.src2));
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_OR3_B32 => {
                let r = self.b_or(self.b_or(self.src_u32(&i.src0), self.src_u32(&i.src1)), self.src_u32(&i.src2));
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_BFI_B32 => {
                let a = self.src_u32(&i.src0);
                let r = self.b_or(self.b_and(a, self.src_u32(&i.src1)), self.b_and(self.b_not(a), self.src_u32(&i.src2)));
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_ALIGNBIT_B32 => {
                let s0 = self.zext64(self.src_u32(&i.src0));
                let s1 = self.zext64(self.src_u32(&i.src1));
                let amt = self.zext64(self.b_and(self.src_u32(&i.src2), self.ci32(0x1F)));
                // {S0,S1}: S0 is the MSBs, S1 the LSBs (ISA §V_ALIGNBIT_B32).
                let concat = self.b_or(llvm::core::LLVMBuildShl(self.b, s0, self.ci64(32), self.n()), s1);
                let r = llvm::core::LLVMBuildTrunc(self.b, llvm::core::LLVMBuildLShr(self.b, concat, amt, self.n()), self.i32t, self.n());
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_LSHLREV_B16 => {
                let amt = self.b_and(self.src_u32(&i.src0), self.ci32(15));
                let v = self.b_and(self.src_u32(&i.src1), self.ci32(0xffff));
                let r = self.b_and(llvm::core::LLVMBuildShl(self.b, v, amt, self.n()), self.ci32(0xffff));
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_LSHRREV_B16 => {
                let amt = self.b_and(self.src_u32(&i.src0), self.ci32(15));
                let v = self.b_and(self.src_u32(&i.src1), self.ci32(0xffff));
                self.st_vgpr32(i.vdst as u32, llvm::core::LLVMBuildLShr(self.b, v, amt, self.n()));
            }
            I::V_ADD_NC_U16 => {
                let a = self.src_u32(&i.src0);
                let b = self.src_u32(&i.src1);
                let r = self.b_and(self.b_add(a, b), self.ci32(0xffff));
                self.st_vgpr32(i.vdst as u32, r);
            }
            I::V_MUL_F32 => {
                let a = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f32(self.src_f32(&i.src1), i.abs, i.neg, 1);
                let r = self.fmul_f32(a, b);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
            }
            I::V_ADD_F32 | I::V_SUB_F32 => {
                let a = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f32(self.src_f32(&i.src1), i.abs, i.neg, 1);
                let r = if matches!(i.op, I::V_ADD_F32) { llvm::core::LLVMBuildFAdd(self.b, a, b, self.n()) }
                        else { llvm::core::LLVMBuildFSub(self.b, a, b, self.n()) };
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
            }
            I::V_MAX_F32 | I::V_MIN_F32 | I::V_MAX_NUM_F32 | I::V_MIN_NUM_F32 => {
                let a = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f32(self.src_f32(&i.src1), i.abs, i.neg, 1);
                let name = if matches!(i.op, I::V_MAX_F32 | I::V_MAX_NUM_F32) { "llvm.maxnum.f32" } else { "llvm.minnum.f32" };
                let r = self.call(name, self.f32t, &[self.f32t, self.f32t], &[a, b]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
            }
            I::V_CVT_U32_F32 => {
                let s = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let v = self.call("llvm.fptoui.sat.i32.f32", self.i32t, &[self.f32t], &[s]);
                self.st_vgpr32(i.vdst as u32, v);
            }
            I::V_CVT_I32_F32 => {
                let s = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let v = self.call("llvm.fptosi.sat.i32.f32", self.i32t, &[self.f32t], &[s]);
                self.st_vgpr32(i.vdst as u32, v);
            }
            I::V_CVT_F32_I32 => {
                let s = self.src_u32(&i.src0);
                let f = llvm::core::LLVMBuildSIToFP(self.b, s, self.f32t, self.n());
                let f = self.absneg_f32(f, i.abs, i.neg, 0);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(f));
            }
            I::V_CVT_F32_U32 => {
                let s = self.src_u32(&i.src0);
                let f = llvm::core::LLVMBuildUIToFP(self.b, s, self.f32t, self.n());
                let f = self.absneg_f32(f, i.abs, i.neg, 0);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(f));
            }
            I::V_RCP_IFLAG_F32 | I::V_RCP_F32 => {
                let s = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let v = self.fdiv_f32(self.cf32(1.0), s);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(v));
            }
            I::V_S_RCP_F32 => {
                let s = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let v = self.fdiv_f32(self.cf32(1.0), s);
                self.st_sgpr32(i.vdst as u32, self.f32_bits(v));
            }
            I::V_RNDNE_F32 | I::V_FLOOR_F32 | I::V_CEIL_F32 | I::V_TRUNC_F32 => {
                let name = match i.op {
                    I::V_RNDNE_F32 => "llvm.roundeven.f32",
                    I::V_FLOOR_F32 => "llvm.floor.f32",
                    I::V_CEIL_F32 => "llvm.ceil.f32",
                    _ => "llvm.trunc.f32",
                };
                let s = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let v = self.call(name, self.f32t, &[self.f32t], &[s]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(v));
            }
            I::V_FMA_F32 => {
                let a = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f32(self.src_f32(&i.src1), i.abs, i.neg, 1);
                let c = self.absneg_f32(self.src_f32(&i.src2), i.abs, i.neg, 2);
                let r = self.call("llvm.fma.f32", self.f32t, &[self.f32t, self.f32t, self.f32t], &[a, b, c]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
            }
            I::V_FMAC_F32 => {
                let a = self.absneg_f32(self.src_f32(&i.src0), i.abs, i.neg, 0);
                let b = self.absneg_f32(self.src_f32(&i.src1), i.abs, i.neg, 1);
                let c = llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(i.vdst as u32), self.f32t, self.n());
                let r = self.call("llvm.fma.f32", self.f32t, &[self.f32t, self.f32t, self.f32t], &[a, b, c]);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
            }
            I::V_CVT_F32_F16 => {
                // RDNA4 ISA: `D0.f32 = f16_to_f32(S0.f16)`. OPSEL[0] selects src0's
                // f16 half (1=high, 0=low). abs/neg (f16) commute with the widening.
                let f = if i.opsel & 1 != 0 { self.src_f16hi_f32(&i.src0) } else { self.src_f16lo_f32(&i.src0) };
                let r = self.absneg_f32(f, i.abs, i.neg, 0);
                self.st_vgpr32(i.vdst as u32, self.f32_bits(r));
            }
            I::V_CMP_EQ_U16 | I::V_CMP_GT_U16 => {
                let a = self.b_and(self.src_u32(&i.src0), self.ci32(0xffff));
                let b = self.b_and(self.src_u32(&i.src1), self.ci32(0xffff));
                let pred = if matches!(i.op, I::V_CMP_EQ_U16) {
                    llvm::LLVMIntPredicate::LLVMIntEQ
                } else {
                    llvm::LLVMIntPredicate::LLVMIntUGT
                };
                let c = llvm::core::LLVMBuildICmp(self.b, pred, a, b, self.n());
                self.st_cmp(i.vdst as u32, c);
            }
            // ----- uniform cross-lane spill idiom (constant lane) -----
            I::V_WRITELANE_B32 => {
                let lane = lane_const(&i.src1)
                    .expect("scalar: v_writelane_b32 needs a constant lane (non-uniform cross-lane unsupported)");
                let val = self.src_u32(&i.src0);
                let slot = self.spill_slot_ptr(i.vdst as u32, lane);
                llvm::core::LLVMBuildStore(self.b, val, slot);
            }
            I::V_READLANE_B32 => {
                let lane = lane_const(&i.src1)
                    .expect("scalar: v_readlane_b32 needs a constant lane");
                let src = vreg_of(&i.src0)
                    .expect("scalar: v_readlane_b32 source must be a VGPR");
                let slot = self.spill_slot_ptr(src, lane);
                let v = llvm::core::LLVMBuildLoad2(self.b, self.i32t, slot, self.n());
                self.st_sgpr32(i.vdst as u32, v);
            }
            _ => panic!("scalar: unsupported VOP3 {:?}", i.op),
        }
    }

    // ---- VOP3P (packed / mixed precision) — lane-local per-work-item ----
    // (V_WMMA_F32_16X16X16_F16 is a cross-lane op; it is split out to a wave-level
    // boundary before compilation, so it never reaches here.)
    unsafe fn emit_vop3p(&self, i: &VOP3P) {
        match i.op {
            I::V_FMA_MIXLO_F16 => {
                // RDNA4 ISA §V_FMA_MIXLO_F16: each source `i` is selected by
                // {OPSEL_HI[i], OPSEL[i]} — OPSEL_HI=0 → f32; OPSEL_HI=1 & OPSEL=1 →
                // hi f16; OPSEL_HI=1 & OPSEL=0 → lo f16. NEG_HI is an abs modifier.
                let opsel_hi = i.opsel_hi | (i.opsel_hi2 << 2);
                let src = |op: &SourceOperand, idx: u32| -> LLVMValueRef {
                    let f = if (opsel_hi >> idx) & 1 == 0 {
                        self.src_f32(op)
                    } else if (i.opsel >> idx) & 1 != 0 {
                        self.src_f16hi_f32(op)
                    } else {
                        self.src_f16lo_f32(op)
                    };
                    self.absneg_f32(f, i.neg_hi, i.neg, idx)
                };
                let a = src(&i.src0, 0);
                let b = src(&i.src1, 1);
                let c = src(&i.src2, 2);
                // `D0[15:0].f16 = f32_to_f16(fma(...))`: FMA in f32, round to f16,
                // write the low 16 bits and preserve vdst's high 16 (no clamp).
                let r = self.call("llvm.fma.f32", self.f32t, &[self.f32t, self.f32t, self.f32t], &[a, b, c]);
                let lo = self.f32_to_f16_bits(r); // high 16 already 0
                let hi = self.b_and(self.ld_vgpr32(i.vdst as u32), self.ci32(0xffff_0000));
                self.st_vgpr32(i.vdst as u32, self.b_or(hi, lo));
            }
            _ => panic!("scalar: unsupported VOP3P {:?}", i.op),
        }
    }

    /// Write a carry/mask bit to `sdst`, unless it is the null register.
    unsafe fn st_sdst_mask(&self, sdst: u8, bit_i1: LLVMValueRef) {
        if sdst == 124 || sdst == 125 {
            return;
        }
        self.st_mask(sdst as u32, bit_i1);
    }

    // ---- VOP3SD (vector op with scalar carry dest) ----------------------
    unsafe fn emit_vop3sd(&self, i: &VOP3SD) {
        match i.op {
            I::V_ADD_CO_U32 => {
                let s0 = self.zext64(self.src_u32(&i.src0));
                let s1 = self.zext64(self.src_u32(&i.src1));
                let sum = self.b_add(s0, s1);
                let lo = llvm::core::LLVMBuildTrunc(self.b, sum, self.i32t, self.n());
                self.st_vgpr32(i.vdst as u32, lo);
                let cout = llvm::core::LLVMBuildLShr(self.b, sum, self.ci64(32), self.n());
                let cout = llvm::core::LLVMBuildTrunc(self.b, cout, self.i32t, self.n());
                let cout = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, cout, self.ci32(0), self.n());
                self.st_sdst_mask(i.sdst, cout);
            }
            I::V_ADD_CO_CI_U32 => {
                let s0 = self.zext64(self.src_u32(&i.src0));
                let s1 = self.zext64(self.src_u32(&i.src1));
                let cin = self.zext64(self.b_and(self.src_u32(&i.src2), self.ci32(1)));
                let sum = self.b_add(self.b_add(s0, s1), cin);
                let lo = llvm::core::LLVMBuildTrunc(self.b, sum, self.i32t, self.n());
                self.st_vgpr32(i.vdst as u32, lo);
                let cout = llvm::core::LLVMBuildLShr(self.b, sum, self.ci64(32), self.n());
                let cout = llvm::core::LLVMBuildTrunc(self.b, cout, self.i32t, self.n());
                let cout = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, cout, self.ci32(0), self.n());
                self.st_sdst_mask(i.sdst, cout);
            }
            I::V_MAD_CO_U64_U32 => {
                let s0 = self.zext64(self.src_u32(&i.src0));
                let s1 = self.zext64(self.src_u32(&i.src1));
                let s2 = self.src_u64(&i.src2);
                let prod = llvm::core::LLVMBuildMul(self.b, s0, s1, self.n());
                // 128-bit add to detect carry: use uadd.with.overflow on i64.
                let ov = self.call(
                    "llvm.uadd.with.overflow.i64",
                    llvm::core::LLVMStructTypeInContext(self.ctx, [self.i64t, self.i1].as_mut_ptr(), 2, 0),
                    &[self.i64t, self.i64t],
                    &[prod, s2],
                );
                let d = llvm::core::LLVMBuildExtractValue(self.b, ov, 0, self.n());
                let c = llvm::core::LLVMBuildExtractValue(self.b, ov, 1, self.n());
                self.st_vgpr64(i.vdst as u32, d);
                self.st_sdst_mask(i.sdst, c);
            }
            I::V_DIV_SCALE_F64 => {
                let a = self.absneg_f64(self.src_f64(&i.src0), 0, i.neg, 0);
                let b = self.absneg_f64(self.src_f64(&i.src1), 0, i.neg, 1);
                let c = self.absneg_f64(self.src_f64(&i.src2), 0, i.neg, 2);
                let r = self.fdiv(self.fmul(a, c), b);
                self.st_vgpr_f64(i.vdst as u32, r);
                let zero = llvm::core::LLVMConstInt(self.i1, 0, 0);
                self.st_sdst_mask(i.sdst, zero);
            }
            I::V_DIV_SCALE_F32 => {
                let a = self.absneg_f32(self.src_f32(&i.src0), 0, i.neg, 0);
                let b = self.absneg_f32(self.src_f32(&i.src1), 0, i.neg, 1);
                let c = self.absneg_f32(self.src_f32(&i.src2), 0, i.neg, 2);
                let packed = self.call("scalar_div_scale_f32", self.i64t, &[self.f32t, self.f32t, self.f32t], &[a, b, c]);
                let val = llvm::core::LLVMBuildTrunc(self.b, packed, self.i32t, self.n());
                self.st_vgpr32(i.vdst as u32, val);
                let flag = llvm::core::LLVMBuildTrunc(self.b, llvm::core::LLVMBuildLShr(self.b, packed, self.ci64(32), self.n()), self.i1, self.n());
                self.st_sdst_mask(i.sdst, flag);
            }
            _ => panic!("scalar: unsupported VOP3SD {:?}", i.op),
        }
    }

    unsafe fn vop3_int<F: Fn(&Cg, LLVMValueRef, LLVMValueRef, LLVMValueRef) -> LLVMValueRef>(&self, i: &VOP3, f: F) {
        let a = self.src_u32(&i.src0);
        let b = self.src_u32(&i.src1);
        let c = self.src_u32(&i.src2);
        let r = f(self, a, b, c);
        self.st_vgpr32(i.vdst as u32, r);
    }

    // ---- VOPC ------------------------------------------------------------
    unsafe fn emit_vopc(&self, i: &VOPC) {
        let is_cmpx = format!("{:?}", i.op).starts_with("V_CMPX");
        let dest = if is_cmpx { EXEC } else { VCC };
        if let Some((pred, invert)) = f32_pred(i.op) {
            let a = self.src_f32(&i.src0);
            let b = llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(i.vsrc1 as u32), self.f32t, self.n());
            let mut c = llvm::core::LLVMBuildFCmp(self.b, pred, a, b, self.n());
            if invert { c = self.b_not(c); }
            self.st_cmp(dest, c);
        } else if let Some((pred, invert)) = f64_pred(i.op) {
            let a = self.src_f64(&i.src0);
            let b = self.ld_vgpr_f64(i.vsrc1 as u32);
            let mut c = llvm::core::LLVMBuildFCmp(self.b, pred, a, b, self.n());
            if invert { c = self.b_not(c); }
            self.st_cmp(dest, c);
        } else if let Some(pred) = int64_pred(i.op) {
            let a = self.src_u64(&i.src0);
            let b = self.ld_vgpr64(i.vsrc1 as u32);
            let c = llvm::core::LLVMBuildICmp(self.b, pred, a, b, self.n());
            self.st_cmp(dest, c);
        } else if let Some(pred) = int_pred(i.op) {
            let a = self.src_u32(&i.src0);
            let b = self.ld_vgpr32(i.vsrc1 as u32);
            let c = llvm::core::LLVMBuildICmp(self.b, pred, a, b, self.n());
            self.st_cmp(dest, c);
        } else {
            panic!("scalar: unsupported VOPC {:?}", i.op);
        }
    }

    // Store a compare result into a lane mask. Inactive lanes read 0, so the
    // bit is ANDed with EXEC; for V_CMPX (dest == EXEC) this yields the correct
    // EXEC = cmp & old_EXEC semantics.
    unsafe fn st_cmp(&self, dest: u32, cmp_i1: LLVMValueRef) {
        let z = llvm::core::LLVMBuildZExt(self.b, cmp_i1, self.i32t, self.n());
        // When the lane is provably active (predication off), EXEC[0]==1 so the
        // mask is identity: VCC = cmp, and V_CMPX's EXEC = cmp & 1 = cmp. Drop it.
        let masked = if self.predicate.get() {
            let exec0 = self.b_and(self.ld_sgpr32(EXEC), self.ci32(1));
            self.b_and(z, exec0)
        } else {
            z
        };
        self.st_sgpr32(dest, masked);
    }

    // ---- VOPD (dual issue) ----------------------------------------------
    // VOPD executes both ops in parallel: *both* halves read their inputs
    // before *either* writes its destination. Compute both results first, then
    // store, so a dependency where opY reads opX's dst sees the old value.
    unsafe fn emit_vopd(&self, i: &VOPD) {
        // VOPD destination encoding: the X op uses vdstx directly; the Y op's
        // real VGPR is (vdsty << 1) | ((vdstx & 1) ^ 1) (opposite parity of X).
        let dx = i.vdstx as u32;
        let dy = ((i.vdsty as u32) << 1) | ((dx & 1) ^ 1);
        // Both halves read their inputs (incl. old dst for FMAC) before either writes.
        let rx = self.eval_vopd_half(i.opx, &i.src0x, i.vsrc1x, dx, i.literal_constant);
        let ry = self.eval_vopd_half(i.opy, &i.src0y, i.vsrc1y, dy, i.literal_constant);
        self.st_vgpr32(dx, rx);
        self.st_vgpr32(dy, ry);
    }
    unsafe fn eval_vopd_half(&self, op: I, src0: &SourceOperand, vsrc1: u8, dst: u32, lit: Option<u32>) -> LLVMValueRef {
        let s0 = self.src_u32(src0);
        let f32b = |cg: &Cg, r: u32| unsafe { llvm::core::LLVMBuildBitCast(cg.b, cg.ld_vgpr32(r), cg.f32t, cg.n()) };
        let fma3 = |cg: &Cg, a, b, c| unsafe { cg.call("llvm.fma.f32", cg.f32t, &[cg.f32t, cg.f32t, cg.f32t], &[a, b, c]) };
        match op {
            I::V_DUAL_MOV_B32 => s0,
            I::V_DUAL_AND_B32 => self.b_and(s0, self.ld_vgpr32(vsrc1 as u32)),
            I::V_DUAL_ADD_NC_U32 => self.b_add(s0, self.ld_vgpr32(vsrc1 as u32)),
            I::V_DUAL_LSHLREV_B32 => self.shl(self.ld_vgpr32(vsrc1 as u32), s0),
            I::V_DUAL_CNDMASK_B32 => {
                let s1 = self.ld_vgpr32(vsrc1 as u32);
                let c = self.vcc_bit();
                llvm::core::LLVMBuildSelect(self.b, c, s1, s0, self.n())
            }
            I::V_DUAL_MUL_F32 => self.f32_bits(self.fmul_f32(self.src_f32(src0), f32b(self, vsrc1 as u32))),
            I::V_DUAL_ADD_F32 => self.f32_bits(llvm::core::LLVMBuildFAdd(self.b, self.src_f32(src0), f32b(self, vsrc1 as u32), self.n())),
            I::V_DUAL_SUB_F32 => self.f32_bits(llvm::core::LLVMBuildFSub(self.b, self.src_f32(src0), f32b(self, vsrc1 as u32), self.n())),
            I::V_DUAL_SUBREV_F32 => self.f32_bits(llvm::core::LLVMBuildFSub(self.b, f32b(self, vsrc1 as u32), self.src_f32(src0), self.n())),
            I::V_DUAL_FMAC_F32 => self.f32_bits(fma3(self, self.src_f32(src0), f32b(self, vsrc1 as u32), f32b(self, dst))),
            I::V_DUAL_FMAMK_F32 => {
                let k = llvm::core::LLVMConstBitCast(self.ci32(lit.unwrap()), self.f32t);
                self.f32_bits(fma3(self, self.src_f32(src0), k, f32b(self, vsrc1 as u32)))
            }
            I::V_DUAL_FMAAK_F32 => {
                let k = llvm::core::LLVMConstBitCast(self.ci32(lit.unwrap()), self.f32t);
                self.f32_bits(fma3(self, self.src_f32(src0), f32b(self, vsrc1 as u32), k))
            }
            _ => panic!("scalar: unsupported VOPD half {:?}", op),
        }
    }

    // ---- SOP1 ------------------------------------------------------------
    unsafe fn emit_sop1(&self, i: &SOP1) {
        match i.op {
            I::S_MOV_B32 => {
                let v = self.src_u32(&i.ssrc0);
                self.st_sgpr32(i.sdst as u32, v);
            }
            I::S_MOV_B64 => {
                let v = self.src_u64(&i.ssrc0);
                self.st_sgpr64(i.sdst as u32, v);
            }
            I::S_AND_SAVEEXEC_B32 => {
                let s0 = self.src_u32(&i.ssrc0);
                let old = self.ld_sgpr32(EXEC);
                self.st_sgpr32(i.sdst as u32, old);
                let ne = self.b_and(s0, old);
                self.st_sgpr32(EXEC, ne);
                self.st_scc_nz(ne);
            }
            I::S_AND_NOT1_SAVEEXEC_B32 => {
                let s0 = self.src_u32(&i.ssrc0);
                let old = self.ld_sgpr32(EXEC);
                self.st_sgpr32(i.sdst as u32, old);
                let ne = self.b_and(s0, self.b_not(old));
                self.st_sgpr32(EXEC, ne);
                self.st_scc_nz(ne);
            }
            I::S_OR_SAVEEXEC_B32 => {
                let s0 = self.src_u32(&i.ssrc0);
                let old = self.ld_sgpr32(EXEC);
                self.st_sgpr32(i.sdst as u32, old);
                let ne = self.b_or(s0, old);
                self.st_sgpr32(EXEC, ne);
                self.st_scc_nz(ne);
            }
            I::S_CTZ_I32_B32 => {
                let x = self.src_u32(&i.ssrc0);
                // cttz(x, is_zero_undef=false) yields 32 for x==0; s_ctz wants -1.
                let tz = self.call("llvm.cttz.i32", self.i32t, &[self.i32t, self.i1], &[x, llvm::core::LLVMConstInt(self.i1, 0, 0)]);
                let is0 = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntEQ, x, self.ci32(0), self.n());
                let r = llvm::core::LLVMBuildSelect(self.b, is0, self.ci32(0xFFFF_FFFF), tz, self.n());
                self.st_sgpr32(i.sdst as u32, r);
            }
            I::S_CVT_F32_I32 => {
                let f = llvm::core::LLVMBuildSIToFP(self.b, self.src_u32(&i.ssrc0), self.f32t, self.n());
                self.st_sgpr32(i.sdst as u32, self.f32_bits(f));
            }
            I::S_CVT_F32_U32 => {
                let f = llvm::core::LLVMBuildUIToFP(self.b, self.src_u32(&i.ssrc0), self.f32t, self.n());
                self.st_sgpr32(i.sdst as u32, self.f32_bits(f));
            }
            I::S_CVT_I32_F32 => {
                let v = self.call("llvm.fptosi.sat.i32.f32", self.i32t, &[self.f32t], &[self.src_f32(&i.ssrc0)]);
                self.st_sgpr32(i.sdst as u32, v);
            }
            I::S_CVT_U32_F32 => {
                let v = self.call("llvm.fptoui.sat.i32.f32", self.i32t, &[self.f32t], &[self.src_f32(&i.ssrc0)]);
                self.st_sgpr32(i.sdst as u32, v);
            }
            _ => panic!("scalar: unsupported SOP1 {:?}", i.op),
        }
    }

    // ---- SOP2 ------------------------------------------------------------
    unsafe fn emit_sop2(&self, i: &SOP2) {
        match i.op {
            I::S_ADD_U32 => {
                // Unsigned add: SCC = carry-out (ISA §S_ADD_U32).
                let a = self.zext64(self.src_u32(&i.ssrc0));
                let b = self.zext64(self.src_u32(&i.ssrc1));
                let sum = self.b_add(a, b);
                let lo = llvm::core::LLVMBuildTrunc(self.b, sum, self.i32t, self.n());
                self.st_sgpr32(i.sdst as u32, lo);
                let cout = llvm::core::LLVMBuildLShr(self.b, sum, self.ci64(32), self.n());
                let cout = llvm::core::LLVMBuildTrunc(self.b, cout, self.i32t, self.n());
                self.st_scc_nz(cout);
            }
            I::S_ADD_CO_I32 | I::S_ADD_I32 => {
                // Signed add: SCC = signed OVERFLOW (ISA §S_ADD_CO_I32), not carry.
                let a = self.src_u32(&i.ssrc0);
                let b = self.src_u32(&i.ssrc1);
                let ov = self.call(
                    "llvm.sadd.with.overflow.i32",
                    llvm::core::LLVMStructTypeInContext(self.ctx, [self.i32t, self.i1].as_mut_ptr(), 2, 0),
                    &[self.i32t, self.i32t],
                    &[a, b],
                );
                let r = llvm::core::LLVMBuildExtractValue(self.b, ov, 0, self.n());
                let c = llvm::core::LLVMBuildExtractValue(self.b, ov, 1, self.n());
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc(c);
            }
            I::S_SUB_CO_I32 => {
                let a = self.src_u32(&i.ssrc0);
                let b = self.src_u32(&i.ssrc1);
                let ov = self.call(
                    "llvm.ssub.with.overflow.i32",
                    llvm::core::LLVMStructTypeInContext(self.ctx, [self.i32t, self.i1].as_mut_ptr(), 2, 0),
                    &[self.i32t, self.i32t],
                    &[a, b],
                );
                let r = llvm::core::LLVMBuildExtractValue(self.b, ov, 0, self.n());
                let c = llvm::core::LLVMBuildExtractValue(self.b, ov, 1, self.n());
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc(c);
            }
            I::S_ADD_NC_U64 => {
                let a = self.src_u64(&i.ssrc0);
                let b = self.src_u64(&i.ssrc1);
                let r = self.b_add(a, b);
                self.st_sgpr64(i.sdst as u32, r);
            }
            I::S_AND_B32 => {
                let a = self.src_u32(&i.ssrc0);
                let b = self.src_u32(&i.ssrc1);
                let r = self.b_and(a, b);
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc_nz(r);
                // Record `dst = B & M` for the mask-select idiom (M = EXEC/VCC).
                if i.sdst as u32 != EXEC && i.sdst as u32 != VCC {
                    let rec = if sreg(&i.ssrc1).map_or(false, |m| m == EXEC || m == VCC) {
                        Some((sreg(&i.ssrc1).unwrap(), a)) // M=ssrc1, B=ssrc0(=a)
                    } else if sreg(&i.ssrc0).map_or(false, |m| m == EXEC || m == VCC) {
                        Some((sreg(&i.ssrc0).unwrap(), b)) // M=ssrc0, B=ssrc1(=b)
                    } else {
                        None
                    };
                    if let Some((m, bval)) = rec {
                        let cond = self.mask_bit(m);
                        self.mask_def
                            .borrow_mut()
                            .insert(i.sdst as u32, MaskDef::And { b: bval, cond, m });
                    }
                }
            }
            I::S_OR_B32 => {
                // Consume the mask-select idiom: `Dr = (A&~M) | (B&M)` → select.
                if let (Some(d0), Some(d1)) = (sreg(&i.ssrc0), sreg(&i.ssrc1)) {
                    let pick = {
                        let md = self.mask_def.borrow();
                        match (md.get(&d0).copied(), md.get(&d1).copied()) {
                            (Some(MaskDef::AndNot1 { a, m: ma }), Some(MaskDef::And { b, cond, m: mb }))
                            | (Some(MaskDef::And { b, cond, m: mb }), Some(MaskDef::AndNot1 { a, m: ma }))
                                if ma == mb =>
                            {
                                Some((a, b, cond))
                            }
                            _ => None,
                        }
                    };
                    if let Some((a, b, cond)) = pick {
                        let r = llvm::core::LLVMBuildSelect(self.b, cond, b, a, self.n());
                        self.st_sgpr32(i.sdst as u32, r);
                        self.st_scc_nz(r);
                        return;
                    }
                }
                self.sop2_logic(i, |c, a, b| c.b_or(a, b));
            }
            I::S_XOR_B32 => self.sop2_logic(i, |c, a, b| c.b_xor(a, b)),
            I::S_AND_NOT1_B32 => {
                let a = self.src_u32(&i.ssrc0);
                let b = self.src_u32(&i.ssrc1);
                let r = self.b_and(a, self.b_not(b));
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc_nz(r);
                // Record `dst = A & ~M` for the mask-select idiom (M = ssrc1).
                if i.sdst as u32 != EXEC && i.sdst as u32 != VCC {
                    if let Some(m) = sreg(&i.ssrc1) {
                        if m == EXEC || m == VCC {
                            self.mask_def
                                .borrow_mut()
                                .insert(i.sdst as u32, MaskDef::AndNot1 { a, m });
                        }
                    }
                }
            }
            I::S_OR_NOT1_B32 => self.sop2_logic(i, |c, a, b| c.b_or(a, c.b_not(b))),
            I::S_LSHR_B32 => self.sop2_logic(i, |c, a, b| c.lshr(a, b)),
            I::S_LSHL_B32 => self.sop2_logic(i, |c, a, b| c.shl(a, b)),
            I::S_BFM_B32 => {
                let width = self.b_and(self.src_u32(&i.ssrc0), self.ci32(31));
                let offset = self.b_and(self.src_u32(&i.ssrc1), self.ci32(31));
                let ones = llvm::core::LLVMBuildSub(
                    self.b,
                    llvm::core::LLVMBuildShl(self.b, self.ci32(1), width, self.n()),
                    self.ci32(1),
                    self.n(),
                );
                let r = llvm::core::LLVMBuildShl(self.b, ones, offset, self.n());
                self.st_sgpr32(i.sdst as u32, r);
            }
            I::S_BFE_U32 => {
                let data = self.src_u32(&i.ssrc0);
                let control = self.src_u32(&i.ssrc1);
                let offset = self.b_and(control, self.ci32(0x1f));
                let width = self.b_and(
                    llvm::core::LLVMBuildLShr(self.b, control, self.ci32(16), self.n()),
                    self.ci32(0x7f),
                );
                let shifted = llvm::core::LLVMBuildLShr(self.b, data, offset, self.n());
                let mask = llvm::core::LLVMBuildSub(
                    self.b,
                    llvm::core::LLVMBuildShl(self.b, self.ci32(1), width, self.n()),
                    self.ci32(1),
                    self.n(),
                );
                let r = self.b_and(shifted, mask);
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc_nz(r);
            }
            I::S_CSELECT_B32 => {
                let a = self.src_u32(&i.ssrc0);
                let b = self.src_u32(&i.ssrc1);
                let c = self.ld_scc();
                let r = llvm::core::LLVMBuildSelect(self.b, c, a, b, self.n());
                self.st_sgpr32(i.sdst as u32, r);
            }
            I::S_MUL_I32 => {
                let a = self.src_u32(&i.ssrc0);
                let b = self.src_u32(&i.ssrc1);
                let r = llvm::core::LLVMBuildMul(self.b, a, b, self.n());
                self.st_sgpr32(i.sdst as u32, r);
            }
            I::S_MUL_HI_U32 => {
                let a = self.zext64(self.src_u32(&i.ssrc0));
                let b = self.zext64(self.src_u32(&i.ssrc1));
                let prod = llvm::core::LLVMBuildMul(self.b, a, b, self.n());
                let hi = llvm::core::LLVMBuildLShr(self.b, prod, self.ci64(32), self.n());
                self.st_sgpr32(i.sdst as u32, llvm::core::LLVMBuildTrunc(self.b, hi, self.i32t, self.n()));
            }
            I::S_LSHL_B64 => {
                let a = self.src_u64(&i.ssrc0);
                let amt = llvm::core::LLVMBuildAnd(self.b, self.src_u64(&i.ssrc1), self.ci64(63), self.n());
                let r = llvm::core::LLVMBuildShl(self.b, a, amt, self.n());
                self.st_sgpr64(i.sdst as u32, r);
                let nz = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, r, self.ci64(0), self.n());
                self.st_scc(nz);
            }
            I::S_MAX_U32 => {
                let a = self.src_u32(&i.ssrc0);
                let b = self.src_u32(&i.ssrc1);
                let c = llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntUGT, a, b, self.n());
                let r = llvm::core::LLVMBuildSelect(self.b, c, a, b, self.n());
                self.st_sgpr32(i.sdst as u32, r);
                self.st_scc(c);
            }
            _ => panic!("scalar: unsupported SOP2 {:?}", i.op),
        }
    }
    unsafe fn sop2_logic<F: Fn(&Cg, LLVMValueRef, LLVMValueRef) -> LLVMValueRef>(&self, i: &SOP2, f: F) {
        let a = self.src_u32(&i.ssrc0);
        let b = self.src_u32(&i.ssrc1);
        let r = f(self, a, b);
        self.st_sgpr32(i.sdst as u32, r);
        self.st_scc_nz(r);
    }

    // ---- SOPK ------------------------------------------------------------
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
            _ => panic!("scalar: unsupported SOPK {:?}", i.op),
        }
    }

    // ---- VFLAT (flat load/store): flat addressing matches the global path.
    unsafe fn emit_vflat(&self, i: &VFLAT) {
        let sext_ioffset = (((i.ioffset << 8) as i32) >> 8) as i64 as u64;
        let base = if i.saddr != 124 {
            self.b_add(self.ld_sgpr64(i.saddr as u32), self.zext64(self.ld_vgpr32(i.vaddr as u32)))
        } else {
            self.ld_vgpr64(i.vaddr as u32)
        };
        let addr = self.b_add(base, self.ci64(sext_ioffset));
        match i.op {
            I::FLAT_LOAD_U8 | I::FLAT_LOAD_I8 | I::FLAT_LOAD_U16 | I::FLAT_LOAD_I16 => {
                let (ty, signed) = match i.op {
                    I::FLAT_LOAD_U8 => (self.i8, false),
                    I::FLAT_LOAD_I8 => (self.i8, true),
                    I::FLAT_LOAD_U16 => (llvm::core::LLVMInt16TypeInContext(self.ctx), false),
                    _ => (llvm::core::LLVMInt16TypeInContext(self.ctx), true),
                };
                let v = llvm::core::LLVMBuildLoad2(self.b, ty, self.ptr_at(addr, 0), self.n());
                let z = if signed { llvm::core::LLVMBuildSExt(self.b, v, self.i32t, self.n()) }
                        else { llvm::core::LLVMBuildZExt(self.b, v, self.i32t, self.n()) };
                self.st_vgpr32(i.vdst as u32, z);
                return;
            }
            I::FLAT_STORE_B8 | I::FLAT_STORE_B16 => {
                let ty = if matches!(i.op, I::FLAT_STORE_B8) { self.i8 } else { llvm::core::LLVMInt16TypeInContext(self.ctx) };
                let t = llvm::core::LLVMBuildTrunc(self.b, self.ld_vgpr32(i.vsrc as u32), ty, self.n());
                llvm::core::LLVMBuildStore(self.b, t, self.ptr_at(self.store_addr(addr), 0));
                return;
            }
            _ => {}
        }
        let (is_store, words) = match i.op {
            I::FLAT_LOAD_B32 => (false, 1), I::FLAT_LOAD_B64 => (false, 2),
            I::FLAT_LOAD_B96 => (false, 3), I::FLAT_LOAD_B128 => (false, 4),
            I::FLAT_STORE_B32 => (true, 1), I::FLAT_STORE_B64 => (true, 2),
            I::FLAT_STORE_B96 => (true, 3), I::FLAT_STORE_B128 => (true, 4),
            _ => panic!("scalar: unsupported VFLAT {:?}", i.op),
        };
        if is_store {
            for k in 0..words {
                let d = self.ld_vgpr32(i.vsrc as u32 + k);
                let a = self.store_addr(self.b_add(addr, self.ci64((k as u64) * 4)));
                llvm::core::LLVMBuildStore(self.b, d, self.ptr_at(a, 0));
            }
            return;
        }
        for k in 0..words {
            let d = llvm::core::LLVMBuildLoad2(self.b, self.i32t, self.ptr_at(addr, (k as u64) * 4), self.n());
            self.st_vgpr32(i.vdst as u32 + k, d);
        }
    }

    // ---- VIMAGE (hardware ray-tracing BVH intersect) — call the native
    // `image_bvh64_intersect_ray` helper; results land in bvh_scratch.
    unsafe fn emit_vimage(&self, i: &VIMAGE) {
        let bits_to_f32 = |r: u32| -> LLVMValueRef {
            llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(r), self.f32t, self.n())
        };
        let scratch_ptr = |k: u32| -> LLVMValueRef {
            llvm::core::LLVMBuildGEP2(self.b, self.i32t, self.bvh_scratch, [self.ci32(k)].as_mut_ptr(), 1, self.n())
        };
        match i.op {
            I::IMAGE_BVH64_INTERSECT_RAY => {
                let node_addr = self.ld_vgpr64(i.vaddr0 as u32);
                let params = [
                    self.ptr, self.ptr, self.ptr, self.ptr, self.i64t,
                    self.f32t, self.f32t, self.f32t, self.f32t,
                    self.f32t, self.f32t, self.f32t, self.f32t, self.f32t, self.f32t,
                ];
                let args = [
                    scratch_ptr(0), scratch_ptr(1), scratch_ptr(2), scratch_ptr(3),
                    node_addr,
                    bits_to_f32(i.vaddr1 as u32),
                    bits_to_f32(i.vaddr2 as u32), bits_to_f32(i.vaddr2 as u32 + 1), bits_to_f32(i.vaddr2 as u32 + 2),
                    bits_to_f32(i.vaddr3 as u32), bits_to_f32(i.vaddr3 as u32 + 1), bits_to_f32(i.vaddr3 as u32 + 2),
                    bits_to_f32(i.vaddr4 as u32), bits_to_f32(i.vaddr4 as u32 + 1), bits_to_f32(i.vaddr4 as u32 + 2),
                ];
                self.call("image_bvh64_intersect_ray", llvm::core::LLVMVoidTypeInContext(self.ctx), &params, &args);
                for k in 0..4u32 {
                    let v = llvm::core::LLVMBuildLoad2(self.b, self.i32t, scratch_ptr(k), self.n());
                    self.st_vgpr32(i.vdata as u32 + k, v);
                }
            }
            _ => panic!("scalar: unsupported VIMAGE {:?}", i.op),
        }
    }

    // ---- SOPC (scalar compare -> SCC) -----------------------------------
    unsafe fn emit_sopc(&self, i: &crate::rdna_instructions::SOPC) {
        use llvm::LLVMIntPredicate::*;
        match i.op {
            I::S_CMP_EQ_U32 | I::S_CMP_LG_U32 | I::S_CMP_GT_U32 | I::S_CMP_LT_U32
            | I::S_CMP_GE_U32 | I::S_CMP_LE_U32 => {
                let a = self.src_u32(&i.ssrc0);
                let b = self.src_u32(&i.ssrc1);
                let p = match i.op {
                    I::S_CMP_EQ_U32 => LLVMIntEQ,
                    I::S_CMP_LG_U32 => LLVMIntNE,
                    I::S_CMP_GT_U32 => LLVMIntUGT,
                    I::S_CMP_LT_U32 => LLVMIntULT,
                    I::S_CMP_GE_U32 => LLVMIntUGE,
                    I::S_CMP_LE_U32 => LLVMIntULE,
                    _ => unreachable!(),
                };
                let c = llvm::core::LLVMBuildICmp(self.b, p, a, b, self.n());
                self.st_scc(c);
            }
            I::S_CMP_LT_I32 | I::S_CMP_GT_I32 | I::S_CMP_GE_I32 | I::S_CMP_LE_I32
            | I::S_CMP_EQ_I32 | I::S_CMP_LG_I32 => {
                let a = self.src_u32(&i.ssrc0);
                let b = self.src_u32(&i.ssrc1);
                let p = match i.op {
                    I::S_CMP_LT_I32 => LLVMIntSLT,
                    I::S_CMP_GT_I32 => LLVMIntSGT,
                    I::S_CMP_GE_I32 => LLVMIntSGE,
                    I::S_CMP_LE_I32 => LLVMIntSLE,
                    I::S_CMP_EQ_I32 => LLVMIntEQ,
                    I::S_CMP_LG_I32 => LLVMIntNE,
                    _ => unreachable!(),
                };
                let c = llvm::core::LLVMBuildICmp(self.b, p, a, b, self.n());
                self.st_scc(c);
            }
            I::S_CMP_EQ_U64 | I::S_CMP_LG_U64 => {
                let a = self.src_u64(&i.ssrc0);
                let b = self.src_u64(&i.ssrc1);
                let p = if matches!(i.op, I::S_CMP_EQ_U64) { LLVMIntEQ } else { LLVMIntNE };
                let c = llvm::core::LLVMBuildICmp(self.b, p, a, b, self.n());
                self.st_scc(c);
            }
            _ => panic!("scalar: unsupported SOPC {:?}", i.op),
        }
    }

    // ---- SMEM (scalar load) ---------------------------------------------
    unsafe fn emit_smem(&self, i: &SMEM) {
        let words = match i.op {
            I::S_LOAD_B32 => 1,
            I::S_LOAD_B64 => 2,
            I::S_LOAD_B96 => 3,
            I::S_LOAD_B128 => 4,
            I::S_LOAD_B256 => 8,
            I::S_LOAD_U16 => 1,
            _ => panic!("scalar: unsupported SMEM {:?}", i.op),
        };
        let base = self.ld_sgpr64(i.sbase as u32 * 2);
        for k in 0..words {
            let ptr = self.ptr_at(base, (i.ioffset as u64) + (k as u64) * 4);
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

    // ---- VGLOBAL (global load/store) ------------------------------------
    unsafe fn emit_vglobal(&self, i: &VGLOBAL) {
        // Cache maintenance: coherent flat memory here, so writeback/invalidate
        // are no-ops.
        if matches!(i.op, I::GLOBAL_WB | I::GLOBAL_INV) {
            return;
        }
        let sext_ioffset = (((i.ioffset << 8) as i32) >> 8) as i64 as u64;
        let base = if i.saddr != 124 {
            let s = self.ld_sgpr64(i.saddr as u32);
            let v = self.zext64(self.ld_vgpr32(i.vaddr as u32));
            self.b_add(s, v)
        } else {
            self.ld_vgpr64(i.vaddr as u32)
        };
        let addr = self.b_add(base, self.ci64(sext_ioffset));

        // Sub-dword loads (zero/sign-extended into a 32-bit VGPR).
        match i.op {
            I::GLOBAL_LOAD_U8 | I::GLOBAL_LOAD_I8 | I::GLOBAL_LOAD_U16 | I::GLOBAL_LOAD_I16 => {
                let (ty, signed, _bytes) = match i.op {
                    I::GLOBAL_LOAD_U8 => (self.i8, false, 1u64),
                    I::GLOBAL_LOAD_I8 => (self.i8, true, 1),
                    I::GLOBAL_LOAD_U16 => (llvm::core::LLVMInt16TypeInContext(self.ctx), false, 2),
                    _ => (llvm::core::LLVMInt16TypeInContext(self.ctx), true, 2),
                };
                let ptr = self.ptr_at(addr, 0);
                let v = llvm::core::LLVMBuildLoad2(self.b, ty, ptr, self.n());
                let z = if signed {
                    llvm::core::LLVMBuildSExt(self.b, v, self.i32t, self.n())
                } else {
                    llvm::core::LLVMBuildZExt(self.b, v, self.i32t, self.n())
                };
                self.st_vgpr32(i.vdst as u32, z);
                return;
            }
            I::GLOBAL_STORE_B16 => {
                let a = self.store_addr(addr);
                let v = llvm::core::LLVMBuildTrunc(self.b, self.ld_vgpr32(i.vsrc as u32), self.i16ty(), self.n());
                llvm::core::LLVMBuildStore(self.b, v, self.ptr_at(a, 0));
                return;
            }
            I::GLOBAL_ATOMIC_ADD_U32 => {
                let data = self.ld_vgpr32(i.vsrc as u32);
                let a = self.store_addr(addr);
                let ptr = self.ptr_at(a, 0);
                let old = llvm::core::LLVMBuildAtomicRMW(
                    self.b,
                    llvm::LLVMAtomicRMWBinOp::LLVMAtomicRMWBinOpAdd,
                    ptr,
                    data,
                    llvm::LLVMAtomicOrdering::LLVMAtomicOrderingSequentiallyConsistent,
                    0,
                );
                self.st_vgpr32(i.vdst as u32, old);
                return;
            }
            _ => {}
        }

        let (is_store, words) = match i.op {
            I::GLOBAL_LOAD_B32 => (false, 1),
            I::GLOBAL_LOAD_B64 => (false, 2),
            I::GLOBAL_LOAD_B96 => (false, 3),
            I::GLOBAL_LOAD_B128 => (false, 4),
            I::GLOBAL_STORE_B32 => (true, 1),
            I::GLOBAL_STORE_B64 => (true, 2),
            I::GLOBAL_STORE_B96 => (true, 3),
            I::GLOBAL_STORE_B128 => (true, 4),
            _ => panic!("scalar: unsupported VGLOBAL {:?}", i.op),
        };
        if is_store {
            for k in 0..words {
                let d = self.ld_vgpr32(i.vsrc as u32 + k);
                let a = self.store_addr(self.b_add(addr, self.ci64((k as u64) * 4)));
                llvm::core::LLVMBuildStore(self.b, d, self.ptr_at(a, 0));
            }
            return;
        }
        // Load: pull aligned pairs directly as `double` into the f64 shadow (one
        // native movsd, no i32→f64 reconstruction) and store the i32 halves for
        // integer readers (DCE drops them when the data is only read as f64).
        let mut k = 0u32;
        while k < words {
            if k + 1 < words {
                let ptr = self.ptr_at(addr, (k as u64) * 4);
                let d = llvm::core::LLVMBuildLoad2(self.b, self.f64t, ptr, self.n());
                self.st_vgpr_f64(i.vdst as u32 + k, d);
                k += 2;
            } else {
                let ptr = self.ptr_at(addr, (k as u64) * 4);
                let d = llvm::core::LLVMBuildLoad2(self.b, self.i32t, ptr, self.n());
                self.st_vgpr32(i.vdst as u32 + k, d);
                k += 1;
            }
        }
    }

    // ---- VSCRATCH (per-work-item private memory) -------------------------
    // The scalar path gives each work-item its own scratch buffer at
    // `scratch_base`, so — unlike the 32-lane-interleaved vector layout — the
    // address is simply `scratch_base + saddr + ioffset` (bytes).
    unsafe fn emit_vscratch(&self, i: &VSCRATCH) {
        let sext_ioffset = (((i.ioffset << 8) as i32) >> 8) as i64 as u64;
        let mut addr = self.b_add(self.scratch_base, self.ci64(sext_ioffset));
        // saddr NULL is encoded as 124 (SGPR_NULL) / 127; otherwise a byte offset.
        // Scratch SGPR/VGPR offsets are SIGNED 32-bit byte offsets (ISA §11.2).
        if i.saddr != 124 && i.saddr != 127 {
            addr = self.b_add(addr, llvm::core::LLVMBuildSExt(self.b, self.ld_sgpr32(i.saddr as u32), self.i64t, self.n()));
        }
        if i.sve != 0 {
            addr = self.b_add(addr, llvm::core::LLVMBuildSExt(self.b, self.ld_vgpr32(i.vaddr as u32), self.i64t, self.n()));
        }
        let (is_store, words) = match i.op {
            I::SCRATCH_LOAD_B32 => (false, 1),
            I::SCRATCH_LOAD_B64 => (false, 2),
            I::SCRATCH_LOAD_B96 => (false, 3),
            I::SCRATCH_LOAD_B128 => (false, 4),
            I::SCRATCH_STORE_B32 => (true, 1),
            I::SCRATCH_STORE_B64 => (true, 2),
            I::SCRATCH_STORE_B96 => (true, 3),
            I::SCRATCH_STORE_B128 => (true, 4),
            _ => panic!("scalar: unsupported VSCRATCH {:?}", i.op),
        };
        for k in 0..words {
            if is_store {
                let d = self.ld_vgpr32(i.vsrc as u32 + k);
                let a = self.store_addr(self.b_add(addr, self.ci64((k as u64) * 4)));
                llvm::core::LLVMBuildStore(self.b, d, self.ptr_at(a, 0));
            } else {
                let ptr = self.ptr_at(addr, (k as u64) * 4);
                let d = llvm::core::LLVMBuildLoad2(self.b, self.i32t, ptr, self.n());
                self.st_vgpr32(i.vdst as u32 + k, d);
            }
        }
    }

    // ---- VSAMPLE (texture sample) ---------------------------------------
    unsafe fn emit_vsample(&self, i: &VSAMPLE) {
        match i.op {
            I::IMAGE_SAMPLE_LZ => {
                let bits_to_f32 = |r: u32| -> LLVMValueRef {
                    llvm::core::LLVMBuildBitCast(self.b, self.ld_vgpr32(r), self.f32t, self.n())
                };
                let args = [
                    self.ld_sgpr32(i.rsrc as u32),
                    self.ld_sgpr32(i.rsrc as u32 + 1),
                    self.ld_sgpr32(i.rsrc as u32 + 2),
                    self.ld_sgpr32(i.rsrc as u32 + 3),
                    self.ld_sgpr32(i.rsrc as u32 + 4),
                    self.ld_sgpr32(i.rsrc as u32 + 5),
                    self.ld_sgpr32(i.rsrc as u32 + 6),
                    self.ld_sgpr32(i.rsrc as u32 + 7),
                    bits_to_f32(i.vaddr0 as u32),
                    bits_to_f32(i.vaddr1 as u32),
                ];
                let params = [
                    self.i32t, self.i32t, self.i32t, self.i32t,
                    self.i32t, self.i32t, self.i32t, self.i32t,
                    self.f32t, self.f32t,
                ];
                let data = self.call("image_sample_lz", self.i32t, &params, &args);
                self.st_vgpr32(i.vdata as u32, data);
            }
            _ => panic!("scalar: unsupported VSAMPLE {:?}", i.op),
        }
    }

    // ---- DS (workgroup shared LDS, cooperative path only) ----------------
    unsafe fn emit_ds(&self, i: &DS) {
        // Byte address into shared LDS: vgpr[addr] + offset0, from LDS base 0.
        let off = self.b_add(self.zext64(self.ld_vgpr32(i.addr as u32)), self.ci64(i.offset0 as u64));
        let raw = self.b_add(self.lds_base, off);
        let ptr = llvm::core::LLVMBuildIntToPtr(self.b, raw, self.ptr, self.n());
        match i.op {
            I::DS_STORE_B8 => {
                let d = self.ld_vgpr32(i.data0 as u32);
                let byte = llvm::core::LLVMBuildTrunc(self.b, d, self.i8, self.n());
                llvm::core::LLVMBuildStore(self.b, byte, ptr);
            }
            I::DS_LOAD_U8 => {
                let byte = llvm::core::LLVMBuildLoad2(self.b, self.i8, ptr, self.n());
                let z = llvm::core::LLVMBuildZExt(self.b, byte, self.i32t, self.n());
                self.st_vgpr32(i.vdst as u32, z);
            }
            _ => panic!("scalar: unsupported DS {:?}", i.op),
        }
    }

    // ---- lane-local spill (uniform writelane/readlane idiom) -------------
    // The compiler spills *uniform* SGPRs into fixed lanes of a scratch VGPR via
    // `v_writelane`/`v_readlane`. Because the value is uniform, "lane K of vD" is
    // the same for every work-item, so we model it per-work-item as a private
    // slot keyed by (vD, K) — no cross-lane traffic. A non-constant lane or a
    // non-SGPR source would be genuine cross-lane and is rejected upstream.
    //
    // The slot lives in the dedicated *lane-spill* buffer (`spill_base[i]`), which
    // the scheduler persists across barrier yields: the compiler frequently
    // writelanes a uniform value in one barrier generation and readlanes it in a
    // later one. It is deliberately separate from the SGPR/VGPR files — RDNA4 has
    // no registers there — so it never masquerades as architectural state.
    unsafe fn spill_slot_ptr(&self, vgpr: u32, lane: u32) -> LLVMValueRef {
        let idx = {
            let mut m = self.spill.borrow_mut();
            let next = m.len();
            *m.entry((vgpr, lane)).or_insert(next)
        };
        assert!(
            idx < COOP_SPILL_SLOTS,
            "too many writelane/readlane spill slots ({} >= {})", idx, COOP_SPILL_SLOTS
        );
        llvm::core::LLVMBuildGEP2(
            self.b,
            self.i32t,
            self.spill_base,
            [self.ci32(idx as u32)].as_mut_ptr(),
            1,
            self.n(),
        )
    }
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

// integer compare opcode -> predicate
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
