//! JIT-compiled kernels and the fiber ABI constants shared by the dispatchers.

/// A JIT-compiled single-work-item kernel owning its executable memory.
/// Concurrent calls borrow the kernel and use disjoint dispatch state.
pub struct ScalarKernel {
    code: super::super::jit::NativeCode,
    pub num_vgprs: usize,
}

impl ScalarKernel {
    pub(in crate::rdna_spmd) fn from_code(code: super::super::jit::NativeCode, num_vgprs: usize) -> Self { Self { code, num_vgprs } }
    /// Run one work-item. `sgprs` points to 128 u32 slots, `vgprs` to
    /// `num_vgprs` u32 slots (both set up by the dispatcher).
    pub unsafe fn run(&self, sgprs: *mut u32, vgprs: *mut u32, scratch_base: u64) {
        let f = std::mem::transmute::<u64, extern "C" fn(*mut u32, *mut u32, u64)>(self.code.address());
        f(sgprs, vgprs, scratch_base);
    }
}

/// Scalar-shaped native code uses the same resumable kernel and fiber ABI as
/// packet code. Width one describes its scheduler layout, not its LLVM shape.
pub type CoopKernel = CoopVecKernel;

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

/// A JIT-compiled width-W kernel. Processes W work-items per `run` call.
pub struct VecKernel {
    code: super::super::jit::NativeCode,
    pub num_vgprs: usize,
    pub width: u32,
    /// Per-lane allocation required by statically addressed private loads.
    pub min_private_bytes: usize,
    pub workgroup_x: Option<u32>,
}
impl VecKernel {
    pub(in crate::rdna_spmd) fn from_code(code: super::super::jit::NativeCode, num_vgprs: usize, width: u32, min_private_bytes: usize, workgroup_x: Option<u32>) -> Self {
        Self { code, num_vgprs, width, min_private_bytes, workgroup_x }
    }
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
    pub(crate) yields: Vec<super::yields::YieldValues>,
    pub(in crate::rdna_spmd) code: super::super::jit::NativeCode,
    pub num_vgprs: usize,
    pub width: u32,
    /// Per-lane allocation required by statically addressed private loads.
    pub min_private_bytes: usize,
    pub workgroup_x: Option<u32>,
}

impl CoopVecKernel {
    pub(in crate::rdna_spmd) fn from_code(code: super::super::jit::NativeCode, yields: Vec<super::yields::YieldValues>, num_vgprs: usize, width: u32, min_private_bytes: usize, workgroup_x: Option<u32>) -> Self {
        Self { yields, code, num_vgprs, width, min_private_bytes, workgroup_x }
    }
    /// Entry address of the compiled packet kernel. It is not callable
    /// directly: the kernel yields by switching stacks, so it has to be
    /// started on a fiber ([`super::fiber::FiberCtx`] documents its
    /// arguments).
    pub fn addr(&self) -> u64 {
        self.code.address()
    }
}
