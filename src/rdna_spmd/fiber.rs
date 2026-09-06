//! Stackful fibers for the cross-lane packet dispatch.
//!
//! A packet kernel has to stop at every wave-level boundary and continue after
//! the host applied it. Stopping by *returning* means the whole packet register
//! file must be in memory at the boundary (to survive the return) and be loaded
//! again on re-entry, once per boundary. A fiber keeps the invocation alive on
//! its own stack instead: a yield is a callee-saved context switch, so values
//! the compiler is holding in registers — a K-loop's accumulators, say — stay
//! there across the boundary, and only the operands of the host op itself move
//! through memory.
//!
//! The JIT binds its yield entrypoints to the driver's crate instance. The
//! explicit context carries all state; no thread-local scheduler is required.
//!
//! Implemented for x86-64 SysV and AArch64 AAPCS64. The switch saves that
//! ABI's callee-saved registers and the stack pointer; every other register is
//! caller-saved, so the compiler already preserves across the yield call
//! whatever it still needs.

/// Arguments of a compiled packet kernel (see `emit_vec::compile_cooperative`).
pub struct KernelArgs {
    pub entry: u64,
    pub sgprs: *mut u32,
    pub vgprs: *mut u32,
    pub spill: *mut u32,
    pub scratch_base: u64,
    pub scratch_stride: u64,
    pub lane_base: u64,
    pub lds_base: u64,
    pub valid_mask: u32, // allocated work items, independent of mutable EXEC
}

/// Driver/kernel shared state. The first two fields have fixed offsets for
/// the assembly context switch.
#[repr(C)]
pub struct FiberCtx {
    driver_rsp: usize,
    fiber_rsp: usize,
    args: KernelArgs,
    values: *mut u32,
}

/// Deepest bytes of a stack, which no kernel should ever reach. They are
/// stamped with [`STACK_POISON`] and checked whenever the fiber is re-armed,
/// so an overflow surfaces as a panic instead of silent corruption of whatever
/// the allocator placed below. Only the guard is stamped: the rest of the
/// stack stays untouched, so it costs nothing until a kernel uses it.
const STACK_GUARD_BYTES: usize = 256;
const STACK_POISON: u8 = 0xA5;

/// Reported by [`Fiber::resume`] when the kernel ran to completion.
pub const FIBER_DONE: u64 = super::emit::COOP_DONE;

type KernelFn = unsafe extern "C" fn(
    *mut u32,      // sgprs
    *mut u32,      // vgprs
    u64,           // scratch_base
    u64,           // scratch_stride
    *mut u32,      // spill
    u64,           // workgroup LDS base
    u64,           // packet lane base
    *mut FiberCtx, // this context
    u32,          // valid packet lanes
) -> u64;

/// A packet kernel invocation with its own stack. [`Fiber::start`] arms it for
/// a run and [`Fiber::resume`] advances it to the next boundary, so one fiber
/// serves every wave a worker owns.
pub struct Fiber {
    // Boxed: the trampoline keeps this address on the fiber stack, so the
    // context must not move once armed.
    ctx: Box<FiberCtx>,
    stack: std::rc::Rc<std::cell::UnsafeCell<Vec<std::mem::MaybeUninit<u8>>>>,
    stack_offset: usize,
    stack_bytes: usize,
}

impl Fiber {
    /// Allocate a fiber with a `stack_bytes` stack. A packet kernel does not
    /// recurse, so its native frame is bounded;
    /// the guard region catches a kernel whose frame exceeds what the caller
    /// allowed for.
    pub fn new(stack_bytes: usize) -> Self {
        Self::batch(1, stack_bytes).pop().unwrap()
    }

    /// Allocate a worker's stacks together. Each fiber owns a disjoint range;
    /// the shared allocation stays alive until the last fiber is dropped.
    /// Keeping this ownership local also avoids an allocator call per lane.
    pub(crate) fn batch(count: usize, stack_bytes: usize) -> Vec<Self> {
        assert!(stack_bytes > STACK_GUARD_BYTES, "fiber stack too small");
        let stack_bytes = stack_bytes.checked_add(15).unwrap() & !15;
        let total = count.checked_mul(stack_bytes).expect("fiber stack allocation overflow");
        let mut storage = Vec::<std::mem::MaybeUninit<u8>>::with_capacity(total);
        // Native code initializes saved slots before reading them. Only the
        // overflow guards need initialization by the driver.
        unsafe { storage.set_len(total); }
        for index in 0..count {
            storage[index * stack_bytes..index * stack_bytes + STACK_GUARD_BYTES]
                .fill(std::mem::MaybeUninit::new(STACK_POISON));
        }
        let stack = std::rc::Rc::new(std::cell::UnsafeCell::new(storage));
        (0..count).map(|index| Self {
            ctx: Box::new(FiberCtx {
                driver_rsp: 0,
                fiber_rsp: 0,
                values: std::ptr::null_mut(),
                args: KernelArgs {
                    entry: 0,
                    sgprs: std::ptr::null_mut(),
                    vgprs: std::ptr::null_mut(),
                    spill: std::ptr::null_mut(),
                    lds_base: 0,
                    valid_mask: u32::MAX,
                    scratch_base: 0,
                    scratch_stride: 0,
                    lane_base: 0,
                },
            }),
            stack: stack.clone(), stack_offset: index * stack_bytes, stack_bytes,
        }).collect()
    }

    /// Arm the fiber to run `args` from the top of its stack, discarding any
    /// state of a previous run. The frame it writes is exactly what [`switch`]
    /// restores, so the first resume lands in [`trampoline`] with the context
    /// pointer in the register that shim expects.
    pub fn start(&mut self, args: KernelArgs) {
        // No Rust reference to a stack range is kept across native execution.
        // The allocation never moves or resizes, and ranges never overlap.
        let base = unsafe { (*self.stack.get()).as_mut_ptr().add(self.stack_offset) };
        assert!(
            (0..STACK_GUARD_BYTES).all(|i| unsafe { (*base.add(i)).assume_init() == STACK_POISON }),
            "fiber stack overflowed: the kernel reached the deepest bytes"
        );
        self.ctx.args = args;
        self.ctx.driver_rsp = 0;
        self.ctx.values = std::ptr::null_mut();
        let top = (base as usize + self.stack_bytes) & !0xF;
        let ctx = &mut *self.ctx as *mut FiberCtx;
        self.ctx.fiber_rsp = unsafe { initial_frame(top, ctx) };
    }

    /// Run until the kernel's next yield and return that boundary's resume pc,
    /// or [`FIBER_DONE`] if the kernel finished.
    pub fn resume(&mut self) -> u64 {
        let ctx = &mut *self.ctx;
        ctx.values = std::ptr::null_mut();
        unsafe { switch(&mut ctx.driver_rsp, ctx.fiber_rsp, 0) }
    }

    pub(crate) fn yield_values(&self) -> *mut u32 {
        assert!(!self.ctx.values.is_null(), "fiber has no pending SSA values");
        self.ctx.values
    }
}

/// Write the frame [`switch`] pops and return the stack pointer to park.
///
/// x86-64: six callee-saved slots — rbx carrying the context pointer — below
/// the trampoline's return address at a 16-byte-aligned top.
#[cfg(target_arch = "x86_64")]
unsafe fn initial_frame(top: usize, ctx: *mut FiberCtx) -> usize {
    let slot = |i: usize| (top - 8 * i) as *mut usize;
    unsafe {
        *slot(1) = trampoline as *const () as usize; // return address
        *slot(2) = 0; // rbp
        *slot(3) = ctx as usize; // rbx: the trampoline's argument
        for i in 4..=7 {
            *slot(i) = 0; // r12-r15
        }
    }
    top - 8 * 7
}

/// AArch64: the callee-saved frame `switch` restores, in its own order —
/// x19-x28, then x29/x30, then d8-d15. The trampoline reads the context from
/// x19 and returns through x30.
#[cfg(target_arch = "aarch64")]
unsafe fn initial_frame(top: usize, ctx: *mut FiberCtx) -> usize {
    const FRAME_BYTES: usize = 160;
    let base = top - FRAME_BYTES;
    unsafe {
        std::ptr::write_bytes(base as *mut u8, 0, FRAME_BYTES);
        *(base as *mut usize) = ctx as usize; // x19
        *((base + 88) as *mut usize) = trampoline as *const () as usize; // x30
    }
    base
}

/// Body of a fiber: run the kernel, then hand [`FIBER_DONE`] to the driver.
extern "C" fn main(ctx: *mut FiberCtx) -> ! {
    unsafe {
        let args = &(*ctx).args;
        let kernel: KernelFn = std::mem::transmute::<u64, KernelFn>(args.entry);
        let done = kernel(
            args.sgprs,
            args.vgprs,
            args.scratch_base,
            args.scratch_stride,
            args.spill,
            args.lds_base,
            args.lane_base,
            ctx,
            args.valid_mask,
        );
        debug_assert_eq!(done, FIBER_DONE, "packet kernel returned without finishing");
        switch(&mut (*ctx).fiber_rsp, (*ctx).driver_rsp, FIBER_DONE);
        unreachable!("resumed a finished fiber");
    }
}

/// Suspend with a dense, typed effect frame owned by the JIT invocation.
/// The scheduler may access the frame only until it resumes this fiber.
#[unsafe(no_mangle)]
pub extern "C" fn amdgpu_sim_fiber_yield_values(ctx: *mut FiberCtx, id: u64, values: *mut u32) {
    unsafe {
        (*ctx).values = values;
        switch(&mut (*ctx).fiber_rsp, (*ctx).driver_rsp, id);
    }
}

#[cfg(target_arch = "x86_64")]
std::arch::global_asm!(
    // u64 switch(usize* save_rsp /*rdi*/, usize to_rsp /*rsi*/, u64 value /*rdx*/)
    // Parks the callee-saved registers and rsp in *save_rsp, adopts to_rsp,
    // and returns `value` on the adopted context.
    ".globl amdgpu_sim_fiber_switch",
    ".hidden amdgpu_sim_fiber_switch",
    "amdgpu_sim_fiber_switch:",
    "push rbp",
    "push rbx",
    "push r12",
    "push r13",
    "push r14",
    "push r15",
    "mov [rdi], rsp",
    "mov rsp, rsi",
    "mov rax, rdx",
    "pop r15",
    "pop r14",
    "pop r13",
    "pop r12",
    "pop rbx",
    "pop rbp",
    "ret",
    // Entry shim of a fresh fiber: `Fiber::start` leaves the context pointer in
    // the rbx slot and this address in the return slot, so the first switch
    // lands here with rbx = ctx and a 16-aligned rsp (the call then gives
    // `main` a standard frame).
    ".globl amdgpu_sim_fiber_trampoline",
    ".hidden amdgpu_sim_fiber_trampoline",
    "amdgpu_sim_fiber_trampoline:",
    "mov rdi, rbx",
    "call {main}",
    "ud2",
    main = sym main,
);

#[cfg(target_arch = "aarch64")]
std::arch::global_asm!(
    // u64 switch(usize* save_sp /*x0*/, usize to_sp /*x1*/, u64 value /*x2*/)
    // AAPCS64 callee-saved state is x19-x28, x29/x30 and the low halves of
    // v8-v15; 160 bytes hold it and keep the 16-byte stack alignment.
    ".globl amdgpu_sim_fiber_switch",
    ".hidden amdgpu_sim_fiber_switch",
    "amdgpu_sim_fiber_switch:",
    "sub sp, sp, #160",
    "stp x19, x20, [sp, #0]",
    "stp x21, x22, [sp, #16]",
    "stp x23, x24, [sp, #32]",
    "stp x25, x26, [sp, #48]",
    "stp x27, x28, [sp, #64]",
    "stp x29, x30, [sp, #80]",
    "stp d8,  d9,  [sp, #96]",
    "stp d10, d11, [sp, #112]",
    "stp d12, d13, [sp, #128]",
    "stp d14, d15, [sp, #144]",
    "mov x3, sp",
    "str x3, [x0]",
    "mov sp, x1",
    "ldp x19, x20, [sp, #0]",
    "ldp x21, x22, [sp, #16]",
    "ldp x23, x24, [sp, #32]",
    "ldp x25, x26, [sp, #48]",
    "ldp x27, x28, [sp, #64]",
    "ldp x29, x30, [sp, #80]",
    "ldp d8,  d9,  [sp, #96]",
    "ldp d10, d11, [sp, #112]",
    "ldp d12, d13, [sp, #128]",
    "ldp d14, d15, [sp, #144]",
    "add sp, sp, #160",
    "mov x0, x2",
    "ret",
    // Entry shim of a fresh fiber: `initial_frame` leaves the context pointer
    // in the x19 slot and this address in the x30 slot.
    ".globl amdgpu_sim_fiber_trampoline",
    ".hidden amdgpu_sim_fiber_trampoline",
    "amdgpu_sim_fiber_trampoline:",
    "mov x0, x19",
    "bl {main}",
    "brk #1",
    main = sym main,
);

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
unsafe extern "C" {
    #[link_name = "amdgpu_sim_fiber_switch"]
    fn switch(save_sp: *mut usize, to_sp: usize, value: u64) -> u64;
    #[link_name = "amdgpu_sim_fiber_trampoline"]
    fn trampoline();
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn switch(_save_sp: *mut usize, _to_sp: usize, _value: u64) -> u64 {
    unimplemented!("stackful fibers are implemented for x86-64 and aarch64")
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn trampoline() {
    unimplemented!("stackful fibers are implemented for x86-64 and aarch64")
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
unsafe fn initial_frame(_top: usize, _ctx: *mut FiberCtx) -> usize {
    unimplemented!("stackful fibers are implemented for x86-64 and aarch64")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Yields twice with distinct pcs, carrying a local across both switches —
    /// the property a stackful fiber provides and a return-based one does not.
    unsafe extern "C" fn counting_kernel(
        sgprs: *mut u32,
        _vgprs: *mut u32,
        _scratch_base: u64,
        _scratch_stride: u64,
        _spill: *mut u32,
        _unused: u64,
        _lane_base: u64,
        ctx: *mut FiberCtx,
        _valid_mask: u32,
    ) -> u64 {
        let mut acc = unsafe { *sgprs } as u64;
        amdgpu_sim_fiber_yield_values(ctx, 100 + acc, sgprs);
        acc += 1;
        amdgpu_sim_fiber_yield_values(ctx, 100 + acc, sgprs);
        unsafe { *sgprs = (acc + 1) as u32 };
        FIBER_DONE
    }

    fn args_for(sgprs: &mut [u32; 1]) -> KernelArgs {
        KernelArgs {
            lds_base: 0,
            valid_mask: u32::MAX,
            entry: counting_kernel as *const () as u64,
            sgprs: sgprs.as_mut_ptr(),
            vgprs: std::ptr::null_mut(),
            spill: std::ptr::null_mut(),
            scratch_base: 0,
            scratch_stride: 0,
            lane_base: 0,
        }
    }

    #[test]
    fn resumes_to_each_yield_then_finishes() {
        let mut sgprs = [7u32];
        let mut fiber = Fiber::new(64 * 1024);
        fiber.start(args_for(&mut sgprs));
        assert_eq!(fiber.resume(), 107);
        assert_eq!(fiber.resume(), 108);
        assert_eq!(fiber.resume(), FIBER_DONE);
        assert_eq!(sgprs[0], 9);
    }

    #[test]
    fn restarts_on_the_same_stack() {
        let mut fiber = Fiber::new(64 * 1024);
        for start in [3u32, 20, 100] {
            let mut sgprs = [start];
            fiber.start(args_for(&mut sgprs));
            assert_eq!(fiber.resume(), 100 + start as u64);
            while fiber.resume() != FIBER_DONE {}
            assert_eq!(sgprs[0], start + 2);
        }
    }

    #[test]
    fn batch_stacks_survive_interleaving_and_sibling_destruction() {
        let mut fibers = Fiber::batch(32, 32 * 1024);
        let mut values: Vec<_> = (0..32).map(|i| [i]).collect();
        for (fiber, value) in fibers.iter_mut().zip(&mut values) {
            fiber.start(args_for(value));
        }
        for (i, fiber) in fibers.iter_mut().enumerate() {
            assert_eq!(fiber.resume(), 100 + i as u64);
        }
        // The remaining frames retain the allocation after siblings drop.
        fibers.truncate(17);
        for (i, fiber) in fibers.iter_mut().enumerate().rev() {
            assert_eq!(fiber.resume(), 101 + i as u64);
            assert_eq!(fiber.resume(), FIBER_DONE);
            assert_eq!(values[i][0], i as u32 + 2);
        }
    }
}
