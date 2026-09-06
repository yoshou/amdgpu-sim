//! LLVM codegen: lower a [`ScalarProgram`](super::ir::ScalarProgram) to a single-work-item native
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

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand, SOP1, SOP2, VIMAGE, VOP3};

use super::scalar_plan::{ScalarMode, ScalarPlan};

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
    module: llvm::prelude::LLVMModuleRef,
    b: LLVMBuilderRef,
    func: LLVMValueRef,
    scratch_base: LLVMValueRef,
    yield_frame: LLVMValueRef,
    sgpr: Vec<CellId>, // 128 i32 representations
    vgpr: Vec<CellId>, // num_vgprs i32 representations
    scc: CellId,       // i1
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
    mask_def: std::cell::RefCell<BTreeMap<u32, MaskDef>>,
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
            self.state.borrow_mut().write(self.b, slot, bit);
            return;
        }
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
    /// The fixup the ISA applies to a division's quotient, as the interpreter's
    /// `div_fixup_f64` and the JIT's `emit_div_fixup_f64` do. This backend has
    /// no quotient of its own to fix up -- the expansion feeding S0 is
    /// collapsed away -- so it divides the original operands and answers for
    /// the cases a division does not answer the ISA's way: a NaN operand, 0/0
    /// and inf/inf, and a quotient too small to reach the smallest subnormal.
    unsafe fn div_fixup_f64(
        &self,
        quotient: LLVMValueRef,
        denominator: LLVMValueRef,
        numerator: LLVMValueRef,
    ) -> LLVMValueRef {
        use llvm::LLVMIntPredicate::*;
        const INFINITY: u64 = 0x7FF0_0000_0000_0000;
        let n = self.n();
        let k = |v: u64| self.ci64(v);
        let bits = |v: LLVMValueRef| llvm::core::LLVMBuildBitCast(self.b, v, self.i64t, n);
        let icmp = |p, x, y| llvm::core::LLVMBuildICmp(self.b, p, x, y, n);
        let or = |x, y| llvm::core::LLVMBuildOr(self.b, x, y, n);
        let and = |x, y| llvm::core::LLVMBuildAnd(self.b, x, y, n);
        let select = |c, t, f| llvm::core::LLVMBuildSelect(self.b, c, t, f, n);

        let b = bits(denominator);
        let c = bits(numerator);
        let abs_b = self.b_and(b, k(0x7FFF_FFFF_FFFF_FFFF));
        let abs_c = self.b_and(c, k(0x7FFF_FFFF_FFFF_FFFF));
        let b_nan = icmp(LLVMIntUGT, abs_b, k(INFINITY));
        let c_nan = icmp(LLVMIntUGT, abs_c, k(INFINITY));
        let both_zero = and(icmp(LLVMIntEQ, abs_b, k(0)), icmp(LLVMIntEQ, abs_c, k(0)));
        let both_infinite = and(
            icmp(LLVMIntEQ, abs_b, k(INFINITY)),
            icmp(LLVMIntEQ, abs_c, k(INFINITY)),
        );
        let exponent =
            |v: LLVMValueRef| self.b_and(llvm::core::LLVMBuildLShr(self.b, v, k(52), n), k(0x7FF));
        let underflow = icmp(
            LLVMIntSLT,
            self.b_sub(exponent(c), exponent(b)),
            k((-1075i64) as u64),
        );

        // The answer for those cases, chosen in reverse so that the earlier
        // ones of the ISA's order win. It is made of the operands alone, so
        // only the last select sits on the quotient's dependency chain.
        let quiet = |v: LLVMValueRef| self.b_or(v, k(0x0008_0000_0000_0000));
        let signed_zero = self.b_and(self.b_xor(b, c), k(0x8000_0000_0000_0000));
        let mut fixed = signed_zero;
        fixed = select(or(both_zero, both_infinite), k(0xFFF8_0000_0000_0000), fixed);
        fixed = select(b_nan, quiet(b), fixed);
        fixed = select(c_nan, quiet(c), fixed);
        let fix = or(or(underflow, or(both_zero, both_infinite)), or(b_nan, c_nan));
        llvm::core::LLVMBuildBitCast(self.b, select(fix, fixed, bits(quotient)), self.f64t, n)
    }


    // VCC bit 0 (single lane) as i1: (vcc & 1) != 0
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

pub(super) fn compile_program(plan: &ScalarPlan<'_>, num_vgprs: usize) -> ScalarKernel {
    unsafe { compile_inner(plan, num_vgprs) }
}

pub(super) fn compile_cooperative(plan: &ScalarPlan<'_>, num_vgprs: usize) -> CoopKernel {
    let sk = unsafe { compile_inner(plan, num_vgprs) };
    let yields = plan.function.value_yields();
    if yields.values().any(|p| p.op == super::ir::typed::effect::EffectOp::Wave(super::ir::typed::effect::WaveOp::Wmma)) {
        super::wmma::warm(1);
    }
    CoopKernel { code: sk.code, num_vgprs: sk.num_vgprs, width: 1,
        min_private_bytes: plan.function.min_private_bytes(), yields }
}

unsafe fn compile_inner(
    plan: &ScalarPlan<'_>,
    num_vgprs: usize,
) -> ScalarKernel {
    let program = plan.program;
    let writeback = plan.mode != ScalarMode::Whole;
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

    let scratch_base = if coop {
        let offset = llvm::core::LLVMBuildMul(b, llvm::core::LLVMGetParam(func, 3),
            llvm::core::LLVMGetParam(func, 6), b"\0".as_ptr().cast());
        llvm::core::LLVMBuildAdd(b, scratch_base, offset, b"\0".as_ptr().cast())
    } else { scratch_base };
    let cells = plan.function.blocks.values().filter_map(|p| p.yield_values.as_ref())
        .map(|p| p.layout.cells()).max().unwrap_or(0);
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
        ctx, module, b, func, scratch_base, yield_frame,
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
    for (&pc, _) in &program.blocks {
        let name = cstr(&format!("b{:x}", pc));
        bbs.insert(pc, llvm::core::LLVMAppendBasicBlockInContext(ctx, func, name.as_ptr()));
    }

    llvm::core::LLVMBuildBr(b, bbs[&program.entry_pc]);

    let mut ssa = super::typed_codegen::Values::new(&plan.function, b, None);
    for &pc in program.blocks.keys() {
        llvm::core::LLVMPositionBuilderAtEnd(b, bbs[&pc]);
        ssa.begin_block(&plan.function, pc);
        let facts = &plan.blocks[&pc];
        cg.set_f64_fresh(facts.f64_fresh);
        cg.sgpr_fresh.set(facts.sgpr_fresh);
        let states = &facts.active;
        cg.mask_def.borrow_mut().clear();
        for (idx, instruction) in facts.instructions.iter().enumerate() {
            // Predicate this instruction's vector writes/compares unless the lane
            // is provably active here (then the mask is a no-op and we drop it).
            cg.predicate.set(!states[idx]);
            match instruction {
                super::lift::Lowering::TypedAlu { .. } => {
                    ssa.emit(&plan.function, pc, idx, |input, _| cg.typed_input(input), |output, value| {
                        cg.typed_output(output, value);
                    });
                }
                super::lift::Lowering::Wave(_) => cg.emit_local_wave(&plan.function.blocks[&pc].wave[&idx]),
                super::lift::Lowering::Memory(_) => {
                    ssa.prepare_memory(&plan.function, pc, idx, |p, scalar| cg.memory_parameter(p, scalar));
                    cg.emit_memory(&plan.function.blocks[&pc].memory[&idx], &ssa, |k| ssa.memory_data(&plan.function, pc, idx, k));
                }
                super::lift::Lowering::Legacy(inst) => cg.emit_inst(inst),
            }
        }
        ssa.condition(&plan.function, pc, |input| cg.typed_input(input));
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
                        Destination::Sgpr(reg) => self.typed_output(Output::Scalar(reg, super::ir::typed::Ty::I32), bits),
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
    unsafe fn emit_inst(&self, inst: &InstFormat) {
        match inst {
            InstFormat::VOP3(i) => self.emit_vop3(i),
            InstFormat::SOP1(i) => self.emit_sop1(i),
            InstFormat::SOP2(i) => self.emit_sop2(i),
            InstFormat::VIMAGE(i) => self.emit_vimage(i),
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


    /// FP instructions carry no fast-math flags (bit-exact with the masked
    /// backend); pass-through kept so call sites read uniformly.
    unsafe fn fmf(&self, v: LLVMValueRef) -> LLVMValueRef {
        v
    }
    unsafe fn fdiv(&self, a: LLVMValueRef, b: LLVMValueRef) -> LLVMValueRef {
        self.fmf(llvm::core::LLVMBuildFDiv(self.b, a, b, self.n()))
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
    unsafe fn emit_vop3(&self, i: &VOP3) {
        match i.op {
            // ----- integer -----
            I::V_DIV_FIXUP_F64 => {
                let b = self.absneg_f64(self.src_f64(&i.src1), i.abs, i.neg, 1);
                let c = self.absneg_f64(self.src_f64(&i.src2), i.abs, i.neg, 2);
                let r = self.div_fixup_f64(self.fdiv(c, b), b, c);
                self.st_vgpr_f64(i.vdst as u32, r);
            }
            _ => panic!("scalar: unsupported VOP3 {:?}", i.op),
        }
    }



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
    unsafe fn emit_sop1(&self, i: &SOP1) {
        match i.op {
            I::S_MOV_B32 => {
                let v = self.src_u32(&i.ssrc0);
                self.st_sgpr32(i.sdst as u32, v);
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
            _ => panic!("scalar: unsupported SOP1 {:?}", i.op),
        }
    }

    // ---- SOP2 ------------------------------------------------------------
    unsafe fn emit_sop2(&self, i: &SOP2) {
        match i.op {
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


    // ---- VFLAT (flat load/store): flat addressing matches the global path.


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
                    self.ptr, self.ptr, self.ptr, self.ptr, self.i32t, self.i32t, self.i64t,
                    self.f32t, self.f32t, self.f32t, self.f32t,
                    self.f32t, self.f32t, self.f32t, self.f32t, self.f32t, self.f32t,
                ];
                let args = [
                    scratch_ptr(0), scratch_ptr(1), scratch_ptr(2), scratch_ptr(3),
                    // The resource, which names where the BVH is and how its
                    // children are sorted.
                    self.ld_sgpr32(i.rsrc as u32), self.ld_sgpr32(i.rsrc as u32 + 1),
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
        if let super::lift::InputSource::PacketMaskAny(reg) = input.source {
            let word = self.b_and(self.ld_sgpr32(reg), self.ci32(1));
            return llvm::core::LLVMBuildICmp(self.b, llvm::LLVMIntPredicate::LLVMIntNE, word, self.ci32(0), self.n());
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
    unsafe fn emit_local_wave(&self, action:&super::lift::wave::YieldAction) {
        use super::lift::wave::{Operand,Destination};
        use super::ir::typed::effect::{EffectOp,WaveOp};
        let source=|index|match &action.inputs[index]{Operand::Source(s)=>s,_=>panic!("local wave operand requires register binding")};
        let dst=match action.outputs[0]{Destination::Sgpr(r)|Destination::Vgpr(r)=>r,_=>panic!("barrier requires cooperative dispatch")};
        let op=match action.op{EffectOp::Wave(op)=>op,_=>panic!("barrier requires cooperative dispatch")};
        match op {
            WaveOp::ReadFirstLane => {
                let v = self.src_u32(source(0));
                self.st_sgpr32(dst, v);
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
                self.st_sgpr32(dst, v);
                        }
            _=>panic!("wave operation requires cooperative dispatch"),
        }
    }
}
