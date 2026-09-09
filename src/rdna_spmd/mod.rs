//! de-SIMT SPMD backend.
//!
//! This is a self-contained alternative to the masked-SIMD `rdna_translator`.
//! Instead of vectorizing a 32-lane wavefront into masked `<16 x iN>`
//! straight-line code, it recovers a *scalar* control-flow graph for a single
//! work-item (de-SIMT) and runs the work-items as independent SPMD program
//! instances across CPU threads for throughput.
//!
//! Pipeline:
//! ```text
//! Compiler: decoded CFG -> lift -> typed SSA passes and analyses -> LLVM IR -> JIT
//! ```
//!
//! It reuses the existing
//! [`RDNAProgram`](crate::rdna_translator::RDNAProgram) CFG builder, recovers a
//! scalar IR ([`ir`]), lifts it to typed SSA and JITs either a single-work-item
//! body or a width-W SPMD body that packs W work-items per SIMD vector
//! ([`codegen`]); cross-lane and barrier kernels yield to the cooperative
//! schedulers.
//!
//! # Performance notes
//!
//! Retained optimizations include contiguous record loads with an in-register
//! transpose, vector `V_TRIG_PREOP_F64`, a guarded scale/sqrt/rescale lowering,
//! collapsing that whole idiom to one correctly-rounded `sqrt`, all-lanes-active
//! block specialization, reconvergence-aware removal of inactive-lane selects,
//! vector lane masks in eligible leaf loops, and scalar mask-expression folding.
//!
//! The packed kernel is bound by dependency-chain latency, not by instruction
//! throughput. On smallpt at W=16 the hardware retires ~1.6 instructions per
//! cycle while ~35% of all cycles retire nothing waiting for an operation to
//! complete (of which ~21 points are waiting on a load); the bounce loop carries
//! ~77 zmm-equivalents of live state against 32 architectural vector registers,
//! and its spill reloads hit L1 (0.06% L1D miss rate) — so the cost is exposed
//! latency, not cache misses. Consequently an optimization only pays if it
//! shortens the chain: measure it in *cycles*, not wall time (the test machine's
//! clock sags ~10% over a benchmark run, which is larger than most effects).
//!
//! Experiments removed after measurement:
//!
//! - Fusing paired 32-bit moves/selects of an f64 pair into one `<W×double>`
//!   operation removed 4.8% of instructions and cost 4.4% more cycles: it merges
//!   two independent 32-bit chains into one, and the removed repacking had been
//!   executing in the shadow of the chain for free.
//! - W=32 packing (+22% instructions, +17% cycles) and W=8 (-13% instructions,
//!   +15% cycles) both lose: at W=16 each f64 operation is two independent
//!   512-bit operations, which is where the packed path's instruction-level
//!   parallelism comes from.
//! - Leaving loop-invariant VGPRs in memory to relieve register pressure — 33 of
//!   the bounce loop's 81 live registers are never written in it — was neutral at
//!   best: pinning them needs volatile accesses, which also blocks reuse, and one
//!   reload per use costs more than the freed register saves.
//! - A region-precise inactive-lane select elision (EXEC region identity per
//!   program point, so a write is predicated only when a lane that was inactive
//!   can observe it) is sound but finds no additional *static* site on this
//!   kernel: the predicated values really are read after reconvergence. Dropping
//!   predication anyway shrinks the function 45% and halves loop pressure, which
//!   is what motivated recovering the same win dynamically — see the
//!   all-lanes-active specialization in [`emit_vec`].
//! - Lowering a constant-exponent `V_LDEXP_F64` to a constant multiply instead of
//!   `vscalefpd` cost 2% cycles (the constant needs a pool load).
//! - Dropping the redundant divisions the division idiom computes ahead of
//!   `V_DIV_FIXUP_F64`, forcing 64-byte spill-slot alignment, oversubscribing
//!   threads past the SMT count, and retuning the record-load transpose tile were
//!   all within noise.
//! - Atomic work queues did not offset their synchronization overhead.
//! - Hot/cold splitting and explicit hot traces did not reduce live state inside
//!   the hot region; transferring state at the new boundaries was costly.
//! - Running part of a W=16 kernel at W=4 or twice at W=8 added conversion and
//!   repeated-execution overhead without enough register-pressure relief.
//! - Explicit continuation frames duplicated state movement already optimized
//!   by LLVM.
//! - Lane compaction moved control state but did not eliminate any packets.
//! - Extending vector mask storage into parent loops required scalar/vector
//!   duplication and tags; the EXEC variant was also incorrect for masks live
//!   across the boundary.
//! - Input prepacking, alternative register allocation, narrower liveness,
//!   smaller mask writeback, reduced optimization levels, and packet reordering
//!   were neutral or worse.
//! - JIT caching, dispatch specialization, and prefetching had too little
//!   profile contribution for the tested long-running workload.


mod analysis;
mod engine;
mod pass;
mod ir;
mod compiler;
mod codegen;
mod decode;
mod program;
pub use program::{Program, CompilationInput};
mod lift;
mod jit;
mod dialect;


pub use engine::cooperative::{dispatch_cooperative, dispatch_cooperative_vec};
pub use engine::xlane::{
    compile_xlane_vec, compile_xlane_vec_layout, dispatch_xlane, dispatch_xlane_vec, split_at_xlane, XlaneOp,
};
pub use engine::dispatch::{dispatch_parallel, dispatch_parallel_vec, GridDims};
pub use compiler::{Compiler, decode_program, compile_cooperative, compile_program, compile_program_vec, compile_program_vec_layout};
pub use engine::kernel::{CoopKernel, CoopVecKernel, ScalarKernel, VecKernel};
pub use decode::{Cond, ScalarBlock, ScalarProgram, Terminator};
pub use program::split_at_barriers;

/// Recommended default width-W work-item packing (W in {1,2,4,8,16}); 0 = off
/// (the single-lane scalar path). See [`codegen`] for the packed register
/// representation.
///
/// Returns **W=16 on AVX-512 hosts**, else 0 (the scalar path). Narrow vector
/// widths were slower than the scalar path in the tested workloads, so this
/// heuristic does not select them. Callers may choose W explicitly; this is not
/// a performance guarantee for every workload or host.
pub fn default_width() -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            return 16;
        }
    }
    0
}
