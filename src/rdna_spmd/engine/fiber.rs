#[derive(Clone, Copy)]
pub struct KernelArgs {
    pub entry: u64,
    pub sgprs: *mut u32,
    pub vgprs: *mut u32,
    pub scratch_base: u64,
    pub scratch_stride: u64,
    pub lane_base: u64,
    pub lds_base: u64,
    pub valid_mask: u32,
    pub frame: *mut u32,
}

#[repr(C)]
pub struct FiberCtx {
    driver_rsp: usize,
    fiber_rsp: usize,
    args: KernelArgs,
    values: *mut u32,
}

const STACK_GUARD_BYTES: usize = 64;
const STACK_POISON: u8 = 0xA5;

pub(crate) type KernelFn = unsafe extern "C" fn(
    *mut u32,
    *mut u32,
    u64,
    u64,
    u64,
    u64,
    *mut FiberCtx,
    u32,
    *mut u32,
) -> u64;

pub struct Fiber {

    ctx: Box<FiberCtx>,
    stack: std::rc::Rc<std::cell::UnsafeCell<Storage>>,
    stack_offset: usize,
    stack_bytes: usize,
}

struct Storage {
    bytes: Vec<std::mem::MaybeUninit<u8>>,
    count: usize,
    stack_bytes: usize,
}

static POOL: std::sync::Mutex<Vec<Storage>> = std::sync::Mutex::new(Vec::new());
const POOL_LIMIT: usize = 64;

impl Storage {
    fn acquire(count: usize, stack_bytes: usize) -> Self {
        let total = count
            .checked_mul(stack_bytes)
            .expect("fiber stack allocation overflow");
        let mut pool = POOL.lock().unwrap();
        let bytes = match pool
            .iter()
            .position(|s| s.count == count && s.stack_bytes == stack_bytes)
        {
            Some(index) => std::mem::take(&mut pool.swap_remove(index).bytes),
            None => {
                let mut storage = Vec::<std::mem::MaybeUninit<u8>>::with_capacity(total);
                unsafe {
                    storage.set_len(total);
                }
                storage
            }
        };
        let mut storage = Storage {
            bytes,
            count,
            stack_bytes,
        };
        for index in 0..count {
            storage.bytes[index * stack_bytes..index * stack_bytes + STACK_GUARD_BYTES]
                .fill(std::mem::MaybeUninit::new(STACK_POISON));
        }
        storage
    }
    fn guard_intact(&self, index: usize) -> bool {
        self.bytes[index * self.stack_bytes..index * self.stack_bytes + STACK_GUARD_BYTES]
            .iter()
            .all(|b| unsafe { b.assume_init() } == STACK_POISON)
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        if self.bytes.is_empty() {
            return;
        }
        for index in 0..self.count {
            assert!(
                self.guard_intact(index),
                "fiber stack overflowed: the kernel reached the deepest bytes"
            );
        }
        let mut pool = POOL.lock().unwrap();
        if pool.len() < POOL_LIMIT {
            pool.push(Storage {
                bytes: std::mem::take(&mut self.bytes),
                count: self.count,
                stack_bytes: self.stack_bytes,
            });
        }
    }
}

impl Fiber {

    pub(crate) fn batch(count: usize, stack_bytes: usize) -> Vec<Self> {
        assert!(stack_bytes > STACK_GUARD_BYTES, "fiber stack too small");
        let stack_bytes = stack_bytes.checked_add(15).unwrap() & !15;

        let stack = std::rc::Rc::new(std::cell::UnsafeCell::new(Storage::acquire(
            count,
            stack_bytes,
        )));
        (0..count)
            .map(|index| Self {
                ctx: Box::new(FiberCtx {
                    driver_rsp: 0,
                    fiber_rsp: 0,
                    values: std::ptr::null_mut(),
                    args: KernelArgs {
                        entry: 0,
                        sgprs: std::ptr::null_mut(),
                        vgprs: std::ptr::null_mut(),
                        lds_base: 0,
                        valid_mask: u32::MAX,
                        scratch_base: 0,
                        scratch_stride: 0,
                        lane_base: 0,
                        frame: std::ptr::null_mut(),
                    },
                }),
                stack: stack.clone(),
                stack_offset: index * stack_bytes,
                stack_bytes,
            })
            .collect()
    }

    pub fn start(&mut self, args: KernelArgs) {

        let base = unsafe {
            (*self.stack.get())
                .bytes
                .as_mut_ptr()
                .add(self.stack_offset)
        };
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

    pub fn resume(&mut self) -> u64 {
        let ctx = &mut *self.ctx;
        ctx.values = std::ptr::null_mut();
        unsafe { switch(&mut ctx.driver_rsp, ctx.fiber_rsp, 0) }
    }

    pub(crate) fn yield_values(&self) -> *mut u32 {
        assert!(
            !self.ctx.values.is_null(),
            "fiber has no pending SSA values"
        );
        self.ctx.values
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn initial_frame(top: usize, ctx: *mut FiberCtx) -> usize {
    let slot = |i: usize| (top - 8 * i) as *mut usize;
    unsafe {
        *slot(1) = trampoline as *const () as usize;
        *slot(2) = 0;
        *slot(3) = ctx as usize;
        for i in 4..=7 {
            *slot(i) = 0;
        }
    }
    top - 8 * 7
}

#[cfg(target_arch = "aarch64")]
unsafe fn initial_frame(top: usize, ctx: *mut FiberCtx) -> usize {
    const FRAME_BYTES: usize = 160;
    let base = top - FRAME_BYTES;
    unsafe {
        std::ptr::write_bytes(base as *mut u8, 0, FRAME_BYTES);
        *(base as *mut usize) = ctx as usize;
        *((base + 88) as *mut usize) = trampoline as *const () as usize;
    }
    base
}

extern "C" fn main(ctx: *mut FiberCtx) -> ! {
    unsafe {
        let args = &(*ctx).args;
        let kernel: KernelFn = std::mem::transmute::<u64, KernelFn>(args.entry);
        let left = kernel(
            args.sgprs,
            args.vgprs,
            args.scratch_base,
            args.scratch_stride,
            args.lds_base,
            args.lane_base,
            ctx,
            args.valid_mask,
            args.frame,
        );
        switch(&mut (*ctx).fiber_rsp, (*ctx).driver_rsp, left);
        unreachable!("resumed a finished fiber");
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn amdgpu_sim_fiber_yield_values(ctx: *mut FiberCtx, id: u64, values: *mut u32) {
    unsafe {
        (*ctx).values = values;
        switch(&mut (*ctx).fiber_rsp, (*ctx).driver_rsp, id);
    }
}

#[cfg(target_arch = "x86_64")]
std::arch::global_asm!(

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
