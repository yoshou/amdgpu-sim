//! LLVM codegen: lower a typed SSA function to a single-work-item native
//! function and JIT it with ORC.
//!
//! Register model: SGPR/VGPR/SCC are SSA definitions initialized from incoming
//! pointers at entry, with explicit phis at native control-flow joins.
//! Output leaves the kernel through `global_store` to absolute
//! host addresses (loaded out of the kernarg buffer), so no register write-back
//! is needed. There are no barriers in the target kernel, so the function runs
//! to completion in one call.

mod memory;

use std::collections::BTreeMap;
use std::ffi::CString;

use llvm_sys as llvm;
use llvm::prelude::{LLVMBasicBlockRef, LLVMBuilderRef, LLVMTypeRef, LLVMValueRef};

use crate::rdna_instructions::SourceOperand;

use super::scalar_plan::{ScalarMode, ScalarPlan};

const EXEC: u32 = 126;
const VCC: u32 = 106;

/// A JIT-compiled single-work-item kernel owning its executable memory.
/// Concurrent calls borrow the kernel and use disjoint dispatch state.
pub struct ScalarKernel {
    code: super::jit::NativeCode,
    pub num_vgprs: usize,
}

impl ScalarKernel {
    /// Run one work-item. `sgprs` points to 128 u32 slots, `vgprs` to
    /// `num_vgprs` u32 slots (both set up by the dispatcher).
    pub unsafe fn run(&self, sgprs: *mut u32, vgprs: *mut u32, scratch_base: u64) {
        let f = std::mem::transmute::<
            u64,
            extern "C" fn(*mut u32, *mut u32, u64),
        >(self.code.address());
        f(sgprs, vgprs, scratch_base);
    }
}

/// Scalar-shaped native code uses the same resumable kernel and fiber ABI as
/// packet code. Width one describes its scheduler layout, not its LLVM shape.
pub type CoopKernel = super::emit_vec::CoopVecKernel;

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

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

use super::native_state::{CellId, State};

struct Cg {
    state: std::cell::RefCell<State>,
    ctx: llvm::prelude::LLVMContextRef,
    b: LLVMBuilderRef,
    func: LLVMValueRef,
    scratch_base: LLVMValueRef,
    private_base: LLVMValueRef,
    private_size: LLVMValueRef,
    yield_frame: LLVMValueRef,
    sgpr: Vec<CellId>, // 128 i32 representations
    vgpr: Vec<CellId>, // num_vgprs i32 representations
    scc: CellId,       // i1
    // cached types
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
    // f64 register typing: a parallel `double` representation per VGPR pair (low reg).
    // f64 ops read/write these directly so a double crosses blocks as one f64
    // phi instead of two i32 phis + reconstruction. `f64_fresh` is the
    // running bitmask (bit r = shadow[r] holds the current value of pair r:r+1),
    // seeded from the freshness analysis at each block entry and updated as
    // instructions emit.
    vgpr_f64: Vec<CellId>,
    f64_fresh: std::cell::Cell<super::regtype::RegSet>,
    // EXEC(126)/VCC(106) are architecturally lane masks, never data. For a single
    // lane they carry one meaningful bit, so we keep them as i1 values and
    // convert at the i32 boundary (zext on read, trunc on write). This lets LLVM
    // fold the wavefront mask arithmetic (s_and/s_or/saveexec) down to i1 logic
    // instead of emitting 32-bit `andn/and/or` + AVX-512 `kmovd` per iteration.
    exec_i1: CellId,
    vcc_i1: CellId,
    // i64 shadow per SGPR pair (low reg) — a loop-carried 64-bit base pointer
    // flows as one i64 phi instead of two i32 phis + per-iteration reconstruct.
    sgpr_i64: Vec<CellId>,
    sgpr_fresh: std::cell::Cell<u128>,
    // De-SIMT mask-select fusion: per-block record of SGPRs defined by an
    // EXEC/VCC-masked `and`/`and_not1`. At the matching `s_or` the pair
    // `(B&M)|(A&~M)` is emitted as `select(M, B, A)` (→ cmov, as native), and the
    // dead `and/andn` DCE away. Cleared at block start; invalidated per write
    // (all entries when EXEC/VCC change).
    writeback: bool,
    writeback_vgprs: usize,
    writeback_words: super::boundary::RegSet,
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
    unsafe fn n(&self) -> *const std::ffi::c_char {
        b"\0".as_ptr() as *const std::ffi::c_char
    }

    // ---- intrinsic / external function declaration -----------------------




    // ---- register access -------------------------------------------------
    unsafe fn ld_sgpr32(&self, i: u32) -> LLVMValueRef {
        if i == 124 { return self.ci32(0); }
        // EXEC/VCC live as i1; present them to integer consumers as 0/1.
        if i == EXEC {
            let b = self.state.borrow_mut().read(self.b, self.exec_i1);
            return llvm::core::LLVMBuildZExt(self.b, b, self.i32t, self.n());
        }
        if i == VCC {
            let b = self.state.borrow_mut().read(self.b, self.vcc_i1);
            return llvm::core::LLVMBuildZExt(self.b, b, self.i32t, self.n());
        }
        self.state.borrow_mut().read(self.b, self.sgpr[i as usize])
    }
    unsafe fn st_sgpr32(&self, i: u32, v: LLVMValueRef) {
        if i == 124 { return; }
        assert!(!matches!(i,EXEC|VCC),"mask state requires a typed I1 definition");
        // A 32-bit write clobbers the i64 shadow of pairs i (i:i+1) and i-1.
        let mut fr = self.sgpr_fresh.get();
        fr &= !(1u128 << (i & 127));
        if i > 0 { fr &= !(1u128 << ((i - 1) & 127)); }
        self.sgpr_fresh.set(fr);
        self.state.borrow_mut().write(self.b, self.sgpr[i as usize], v);
    }
    unsafe fn pred_vgpr32(&self, i: u32, v: LLVMValueRef) -> LLVMValueRef {
        if self.predicate.get() {
            let old = self.state.borrow_mut().read(self.b, self.vgpr[i as usize]);
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
        self.state.borrow_mut().read(self.b, self.vgpr[i as usize])
    }
    unsafe fn st_vgpr32(&self, i: u32, v: LLVMValueRef) {
        self.f64_fresh_clr(i);
        if i > 0 { self.f64_fresh_clr(i - 1); }
        let v = self.pred_vgpr32(i, v);
        self.state.borrow_mut().write(self.b, self.vgpr[i as usize], v);
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
        self.state.borrow_mut().read(self.b, self.scc)
    }
    unsafe fn st_scc(&self, v: LLVMValueRef) {
        // v is i1
        self.state.borrow_mut().write(self.b, self.scc, v);
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

    unsafe fn ld_sgpr64(&self, i: u32) -> LLVMValueRef {
        let ordinary = !matches!(i, 105 | 106 | 123 | 124 | 125 | 126 | 127);
        if ordinary && self.sgpr_fresh.get() & (1u128 << (i & 127)) != 0 {
            return self.state.borrow_mut().read(self.b, self.sgpr_i64[i as usize]);
        }
        let lo = self.zext64(self.ld_sgpr32(i));
        let hi = self.zext64(self.ld_sgpr32(i + 1));
        let hi = llvm::core::LLVMBuildShl(self.b, hi, self.ci64(32), self.n());
        let v = llvm::core::LLVMBuildOr(self.b, hi, lo, self.n());
        if ordinary {
            self.state.borrow_mut().write(self.b, self.sgpr_i64[i as usize], v);
            self.sgpr_fresh.set(self.sgpr_fresh.get() | (1u128 << (i & 127)));
        }
        v
    }
    unsafe fn st_sgpr64(&self, i: u32, v: LLVMValueRef) {
        let lo = llvm::core::LLVMBuildTrunc(self.b, v, self.i32t, self.n());
        let hi = llvm::core::LLVMBuildLShr(self.b, v, self.ci64(32), self.n());
        let hi = llvm::core::LLVMBuildTrunc(self.b, hi, self.i32t, self.n());
        self.st_sgpr32(i, lo); // clears sgpr_fresh for i-1/i
        self.st_sgpr32(i + 1, hi); // clears for i/i+1
        if !matches!(i, 105 | 106 | 123 | 124 | 125 | 126 | 127) {
            self.state.borrow_mut().write(self.b, self.sgpr_i64[i as usize], v);
            self.sgpr_fresh.set(self.sgpr_fresh.get() | (1u128 << (i & 127)));
        }
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
            return self.state.borrow_mut().read(self.b, self.vgpr_f64[i as usize]);
        }
        let u = self.ld_vgpr64(i);
        let d = llvm::core::LLVMBuildBitCast(self.b, u, self.f64t, self.n());
        self.state.borrow_mut().write(self.b, self.vgpr_f64[i as usize], d);
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
            self.state.borrow_mut().write(self.b, self.vgpr_f64[i as usize], v);
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
    unsafe fn i16ty(&self) -> LLVMTypeRef { llvm::core::LLVMInt16TypeInContext(self.ctx) }
    /// Low (bit 0) f16 of `op`, widened to f32.
    /// High (bit 16) f16 of `op`, widened to f32.
    /// Round f32 `v` to f16, returned as i32 (f16 bits in low 16, high 0).

    // ---- source operands -------------------------------------------------
    unsafe fn src_u32(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.ci32(*v),
            SourceOperand::IntegerConstant(v) => self.ci32(*v as u32),
            SourceOperand::ScalarRegister(r) => self.ld_sgpr32(*r as u32),
            SourceOperand::VectorRegister(r) => self.ld_vgpr32(*r as u32),
            SourceOperand::FloatConstant(v) => self.ci32((*v as f32).to_bits()),
            SourceOperand::PrivateBase => llvm::core::LLVMBuildTrunc(self.b, self.private_base, self.i32t, self.n()),
        }
    }
    unsafe fn src_u64(&self, op: &SourceOperand) -> LLVMValueRef {
        match op {
            SourceOperand::LiteralConstant(v) => self.ci64(*v as u64),
            SourceOperand::IntegerConstant(v) => self.ci64(*v),
            SourceOperand::ScalarRegister(r) => self.ld_sgpr64(*r as u32),
            SourceOperand::VectorRegister(r) => self.ld_vgpr64(*r as u32),
            SourceOperand::PrivateBase => self.private_base,
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

pub(super) fn compile_program(plan: &ScalarPlan, num_vgprs: usize) -> ScalarKernel {
    unsafe { compile_inner(plan, num_vgprs) }
}

pub(super) fn compile_cooperative(plan: &ScalarPlan, num_vgprs: usize) -> CoopKernel {
    let sk = unsafe { compile_inner(plan, num_vgprs) };
    let yields = plan.function.value_yields();
    if yields.values().any(|p| p.op == super::ir::typed::effect::EffectOp::Wave(super::ir::typed::effect::WaveOp::Wmma)) {
        super::wmma::warm(1);
    }
    CoopKernel { code: sk.code, num_vgprs: sk.num_vgprs, width: 1,
        min_private_bytes: plan.function.min_private_bytes(), yields }
}

unsafe fn compile_inner(
    plan: &ScalarPlan,
    num_vgprs: usize,
) -> ScalarKernel {
    let writeback = plan.mode != ScalarMode::Whole && plan.function.observable_return;
    let force_exec = plan.mode == ScalarMode::Whole;
    let coop = plan.mode == ScalarMode::Cooperative;
    // VGPR slots: allocate a safe upper bound (RDNA max is 256) since the
    // granulated descriptor count can underestimate the actual max index.
    let num_vgprs = num_vgprs.max(256);

    let native = super::jit::Module::new("scalar_kernel");
    let ctx = native.ctx;
    let module = native.module;
    let b = native.builder;

    let i1 = llvm::core::LLVMInt1TypeInContext(ctx);
    let i8 = llvm::core::LLVMInt8TypeInContext(ctx);
    let i32t = llvm::core::LLVMInt32TypeInContext(ctx);
    let i64t = llvm::core::LLVMInt64TypeInContext(ctx);
    let f32t = llvm::core::LLVMFloatTypeInContext(ctx);
    let f64t = llvm::core::LLVMDoubleTypeInContext(ctx);
    let ptr = llvm::core::LLVMPointerTypeInContext(ctx, 0);
    let void = llvm::core::LLVMVoidTypeInContext(ctx);

    // Non-coop: `void kernel(u32* sgprs, u32* vgprs, u64 scratch)`.
    // Coop uses the common nine-argument fiber ABI (see fiber::KernelArgs).
    let func = if coop {
        let mut params = [ptr, ptr, i64t, i64t, ptr, i64t, i64t, ptr, i32t];
        let fty = llvm::core::LLVMFunctionType(i64t, params.as_mut_ptr(), 9, 0);
        llvm::core::LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, fty)
    } else {
        let mut params = [ptr, ptr, i64t];
        let fty = llvm::core::LLVMFunctionType(void, params.as_mut_ptr(), 3, 0);
        llvm::core::LLVMAddFunction(module, b"kernel\0".as_ptr() as *const _, fty)
    };

    let sgprs_p = llvm::core::LLVMGetParam(func, 0);
    let vgprs_p = llvm::core::LLVMGetParam(func, 1);
    let scratch_base = llvm::core::LLVMGetParam(func, 2);
    let private_size=if coop {llvm::core::LLVMGetParam(func,3)} else {llvm::core::LLVMConstInt(i64t,0,0)};
    let lds_base = if coop {
        llvm::core::LLVMGetParam(func, 5)
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

    // SH_MEM_BASES exposes the aperture's high word. Its virtual base must
    // therefore stay fixed when the scheduler advances to another wave.
    let aperture=llvm::core::LLVMBuildAnd(b,scratch_base,llvm::core::LLVMConstInt(i64t,0xffff_ffff_0000_0000,0),b"\0".as_ptr().cast());
    let sized=llvm::core::LLVMBuildICmp(b,llvm::LLVMIntPredicate::LLVMIntNE,private_size,llvm::core::LLVMConstInt(i64t,0,0),b"\0".as_ptr().cast());
    let private_base=llvm::core::LLVMBuildSelect(b,sized,aperture,scratch_base,b"\0".as_ptr().cast());

    let scratch_base = if coop {
        let offset = llvm::core::LLVMBuildMul(b, llvm::core::LLVMGetParam(func, 3),
            llvm::core::LLVMGetParam(func, 6), b"\0".as_ptr().cast());
        llvm::core::LLVMBuildAdd(b, scratch_base, offset, b"\0".as_ptr().cast())
    } else { scratch_base };
    let cells = plan.function.value_yields().values().map(|p| p.cells()).max().unwrap_or(0);
    let yield_frame = if cells == 0 { llvm::core::LLVMConstNull(ptr) } else {
        let frame = llvm::core::LLVMBuildAlloca(b, llvm::core::LLVMArrayType2(i32t, cells as u64), cstr("yield.values").as_ptr());
        llvm::core::LLVMSetAlignment(frame, 64);
        frame
    };

    // Declare typed register representations and the SCC flag.
    let mut state = State::default();
    let mut sgpr = Vec::with_capacity(128);
    for _ in 0..128 {
        sgpr.push(state.add(i32t));
    }
    let mut vgpr = Vec::with_capacity(num_vgprs);
    for _ in 0..num_vgprs {
        vgpr.push(state.add(i32t));
    }
    // Parallel `double` shadow per VGPR pair (low reg). +1 so the high half of
    // the last pair has a slot.
    let mut vgpr_f64 = Vec::with_capacity(num_vgprs + 1);
    for _ in 0..num_vgprs + 1 {
        vgpr_f64.push(state.add(f64t));
    }
    let scc = state.add(i1);
    let exec_i1 = state.add(i1);
    let vcc_i1 = state.add(i1);
    let mut sgpr_i64 = Vec::with_capacity(129);
    for _ in 0..129 {
        sgpr_i64.push(state.add(i64t));
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
        state: std::cell::RefCell::new(state),
        writeback_words: plan.function.written,
        ctx, b, func, scratch_base, private_base, private_size, yield_frame,
        sgpr, vgpr, scc, i8, i32t, i64t, f32t, f64t, ptr,
        predicate: std::cell::Cell::new(false),
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
        if matches!(i,106|126) {
            let bit=llvm::core::LLVMBuildTrunc(b,v,i1,cg.n());
            cg.typed_output(super::lift::Output::MaskBit(i),bit);
        } else {cg.st_sgpr32(i,v);}
    }
    for i in 0..num_vgprs as u32 {
        let gep = llvm::core::LLVMBuildGEP2(b, i32t, vgprs_p, [cg.ci32(i)].as_mut_ptr(), 1, cg.n());
        let v = llvm::core::LLVMBuildLoad2(b, i32t, gep, cg.n());
        cg.st_vgpr32(i, v);
    }
    if force_exec {
        cg.typed_output(super::lift::Output::MaskBit(EXEC),llvm::core::LLVMConstInt(i1,1,0));
    }
    if coop {
        // Initial SCC arrives in the private entry slot. It subsequently
        // survives every yield as a native SSA value.
        let gep = llvm::core::LLVMBuildGEP2(b, i32t, sgprs_p, [cg.ci32(128)].as_mut_ptr(), 1, cg.n());
        let s = llvm::core::LLVMBuildLoad2(b, i32t, gep, cg.n());
        cg.st_scc_nz(s);
    } else {
        cg.st_scc(llvm::core::LLVMConstInt(i1, 0, 0));
    }

    // From here on, predicate vector writes on EXEC bit 0.
    cg.predicate.set(true);

    // Create a basic block per scalar block.
    let mut bbs: BTreeMap<usize, LLVMBasicBlockRef> = BTreeMap::new();
    for &pc in plan.function.blocks.keys() {
        let name = cstr(&format!("b{:x}", pc));
        bbs.insert(pc, llvm::core::LLVMAppendBasicBlockInContext(ctx, func, name.as_ptr()));
    }

    let mut ssa = super::typed_codegen::Values::new(&plan.function, b, None);
    ssa.set_scratch_environment(cg.private_base, cg.private_size);
    ssa.set_bvh_storage(super::dialect::rdna4::bvh::Storage {scratch: cg.bvh_scratch, packet: std::ptr::null_mut(), packet_ty: std::ptr::null_mut()});
    llvm::core::LLVMBuildBr(b, bbs[&plan.function.ir.func().entry.0]);
    for &pc in plan.function.blocks.keys() {
        llvm::core::LLVMPositionBuilderAtEnd(b, bbs[&pc]);
        ssa.begin_block(&plan.function, pc);
        let facts = &plan.blocks[&pc];
        cg.set_f64_fresh(facts.f64_fresh);
        cg.sgpr_fresh.set(facts.sgpr_fresh);
        let states = &facts.active;
        for (idx, instruction) in plan.function.blocks[&pc].instructions.iter().enumerate() {
            // Predicate this instruction's vector writes/compares unless the lane
            // is provably active here (then the mask is a no-op and we drop it).
            cg.predicate.set(!states[idx]);
            match instruction {
                Some(_) => {
                    ssa.emit(&plan.function, pc, idx, false, |_|true,
                        |reg|if matches!(reg,106|126) {cg.typed_input(&super::lift::Input {source:super::lift::InputSource::MaskBit(reg),ty:super::ir::typed::Ty::I1})} else {cg.ld_sgpr32(reg)},
                        |input, _| cg.typed_input(input), |output, value, _| {
                        let predicate=cg.predicate.replace(false);
                        cg.typed_output(output, value);
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
                    ssa.prepare_memory(&plan.function, pc, idx, |p, scalar| cg.memory_parameter(p, scalar));
                    cg.emit_memory(&plan.function.blocks[&pc].memory[&idx], facts.memory[&idx], &ssa, |k| ssa.memory_data(&plan.function, pc, idx, k));
                }
            }
        }
        ssa.condition(&plan.function, pc, |reg|cg.ld_sgpr32(reg), |input| cg.typed_input(input));
        cg.emit_term(pc, &plan.function, &bbs, &mut ssa);
    }

    cg.state.borrow_mut().finish(func);
    ScalarKernel { code: native.finish(super::jit::Mode::Scalar), num_vgprs }
}

// =====================================================================
//  Terminators
// =====================================================================

impl Cg {
    unsafe fn emit_writeback(&self, num_vgprs: usize) {
        for i in 0..128u32 {
            // NULL has no architectural storage to persist across a yield.
            if !self.writeback_words.has_sgpr(i) { continue; }
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
            if !self.writeback_words.has_vgpr(i) { continue; }
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

    unsafe fn emit_term(&self, pc: usize, function: &super::lift::function::Function, bbs: &BTreeMap<usize, LLVMBasicBlockRef>, values: &mut super::typed_codegen::Values) {
        match &function.terminator(pc) {
            super::lift::function::Control::Return => {
                if self.coop {
                    // End of work-item: persist state, return the DONE sentinel.
                    if self.writeback {
                        self.emit_writeback(self.writeback_vgprs);
                        self.emit_scc_writeback();
                    }
                    llvm::core::LLVMBuildRet(self.b, self.ci64(u64::MAX));
                } else {
                    if self.writeback {
                        self.emit_writeback(self.writeback_vgprs);
                    }
                    llvm::core::LLVMBuildRetVoid(self.b);
                }
            }
            super::lift::function::Control::Yield { resume } => {
                use super::ir::typed::effect::{EffectOp, WaveOp};
                use super::lift::{Output, wave::Destination};
                let plan = function.blocks[&pc].yield_values.as_ref().expect("yield lacks SSA value plan");
                values.prepare_yield(function, pc, llvm::core::LLVMGetParam(self.func,6), |input| self.typed_input(input));
                let predicated = matches!(plan.layout.op, EffectOp::Wave(WaveOp::Bpermute | WaveOp::BpermuteFi));
                let previous = self.predicate.replace(predicated);
                values.yield_values(plan, self.yield_frame, llvm::core::LLVMGetParam(self.func, 7), *resume,
                    |destination, ty, result, bits| match destination {
                        Destination::Vgpr(reg) => self.typed_output(Output::Vgpr(reg, ty), result),
                        Destination::Sgpr(reg) => if ty==super::ir::typed::Ty::I1 {self.typed_output(Output::MaskBit(reg),result)} else {self.typed_output(Output::Scalar(reg, super::ir::typed::Ty::I32), bits)},
                        Destination::Scc => self.typed_output(Output::Scc, result),
                    });
                self.predicate.set(previous);
                llvm::core::LLVMBuildBr(self.b, bbs[resume]);
            }
            super::lift::function::Control::Jump(t) => {
                llvm::core::LLVMBuildBr(self.b, bbs[t]);
            }
            super::lift::function::Control::Branch { cond, taken, fallthrough } => {
                let taken_cond = values.value(*cond);
                llvm::core::LLVMBuildCondBr(self.b, taken_cond, bbs[taken], bbs[fallthrough]);
            }
        }
    }


}

// =====================================================================
//  Instruction emission
// =====================================================================

impl Cg {


    // ---- helpers ---------------------------------------------------------
    unsafe fn b_and(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildAnd(self.b, a, b, self.n())
    }


    unsafe fn b_add(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        llvm::core::LLVMBuildAdd(self.b, a, b, self.n())
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



    // ---- VOP3 ------------------------------------------------------------




    // ---- VOPC ------------------------------------------------------------

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

    // ---- SOP1 ------------------------------------------------------------


    // ---- SOP2 ------------------------------------------------------------




    // ---- VFLAT (flat load/store): flat addressing matches the global path.


    // ---- VIMAGE (hardware ray-tracing BVH intersect) — call the native
    // `image_bvh64_intersect_ray` helper; results land in bvh_scratch.



    // ---- SMEM (scalar load) ---------------------------------------------


    // ---- VGLOBAL (global load/store) ------------------------------------


    // ---- VSCRATCH (per-work-item private memory) -------------------------
    // The scalar path gives each work-item its own scratch buffer at
    // `scratch_base`, so — unlike the 32-lane-interleaved vector layout — the
    // address is simply `scratch_base + saddr + ioffset` (bytes).




    // ---- DS (workgroup shared Lcooperative path only) ----------------


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


// integer compare opcode -> predicate


impl Cg {
    unsafe fn typed_input(&self, input: &super::lift::Input) -> LLVMValueRef {
        use super::ir::typed::Ty;
        if matches!(input.source,super::lift::InputSource::ExecPredicate) {
            return self.typed_input(&super::lift::Input {source:super::lift::InputSource::Operand(SourceOperand::ScalarRegister(126)),ty:Ty::I1});
        }
        if let super::lift::InputSource::MaskBit(reg) = input.source {
            let cell=if reg==EXEC {self.exec_i1} else {assert_eq!(reg,VCC);self.vcc_i1};
            return self.state.borrow_mut().read(self.b,cell);
        }
        if matches!(input.source, super::lift::InputSource::Scc) { return self.ld_scc(); }
        match input.ty {
            Ty::I1 => llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, self.b_and(self.src_u32(input.source.operand()), self.ci32(1)), self.ci32(0), self.n()),
            Ty::I32 => self.src_u32(input.source.operand()),
            Ty::I64 => self.src_u64(input.source.operand()),
            Ty::F32 => self.src_f32(input.source.operand()),
            Ty::F64 => self.src_f64(input.source.operand()),
        }
    }
    unsafe fn typed_output(&self, output: super::lift::Output, result: LLVMValueRef) {
        use super::ir::typed::Ty;
        use super::lift::Output;
        match output {
            Output::Vgpr(reg, Ty::I32) => self.st_vgpr32(reg, result),
            Output::Vgpr(reg, Ty::I64) => self.st_vgpr64(reg, result),
            Output::Vgpr(reg, Ty::F32) => self.st_vgpr32(reg, self.f32_bits(result)),
            Output::Vgpr(reg, Ty::F64) => self.st_vgpr_f64(reg, result),
            Output::Compare(reg) => self.st_cmp(reg, result),
            Output::Mask(reg) => self.st_mask(reg, result),
            Output::MaskBit(reg) => {
                let cell = if reg == EXEC { self.exec_i1 } else { assert_eq!(reg,VCC);self.vcc_i1 };
                self.state.borrow_mut().write(self.b,cell,result);
            },
            Output::Scalar(reg, Ty::I32) => self.st_sgpr32(reg, result),
            Output::Scalar(reg, Ty::I64) => self.st_sgpr64(reg, result),
            Output::Scalar(reg, Ty::F32) => self.st_sgpr32(reg, self.f32_bits(result)),
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
                let v = self.src_u32(source(0));
                write(dst,v);
                        }
            WaveOp::WriteLane => {
                let lane = lane_const(source(1))
                    .expect("scalar: v_writelane_b32 needs a constant lane (non-uniform cross-lane unsupported)");
                let val = self.src_u32(source(0));
                let slot = self.spill_slot_ptr(dst, lane);
                llvm::core::LLVMBuildStore(self.b, val, slot);
                        }
            WaveOp::ReadLane => {
                let lane = lane_const(source(1))
                    .expect("scalar: v_readlane_b32 needs a constant lane");
                let src = vreg_of(source(0))
                    .expect("scalar: v_readlane_b32 source must be a VGPR");
                let slot = self.spill_slot_ptr(src, lane);
                let v = llvm::core::LLVMBuildLoad2(self.b, self.i32t, slot, self.n());
                write(dst,v);
                        }
            _=>panic!("wave operation requires cooperative dispatch"),
        }
    }
}
