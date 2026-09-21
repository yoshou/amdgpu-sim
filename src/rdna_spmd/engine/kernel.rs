pub struct ScalarKernel {
    code: super::super::native::jit::NativeCode,
    pub num_vgprs: usize,
    pub(crate) group: bool,
}

impl ScalarKernel {
    pub(in crate::rdna_spmd) fn from_code(
        code: super::super::native::jit::NativeCode,
        num_vgprs: usize,
        group: bool,
    ) -> Self {
        Self {
            code,
            num_vgprs,
            group,
        }
    }

    pub unsafe fn run(&self, sgprs: *mut u32, vgprs: *mut u32, scratch_base: u64, lds_base: u64) {
        if self.group {
            let f = std::mem::transmute::<u64, extern "C" fn(*mut u32, *mut u32, u64, u64)>(
                self.code.address(),
            );
            f(sgprs, vgprs, scratch_base, lds_base);
        } else {
            let f = std::mem::transmute::<u64, extern "C" fn(*mut u32, *mut u32, u64)>(
                self.code.address(),
            );
            f(sgprs, vgprs, scratch_base);
        }
    }
}

pub const COOP_DONE: u64 = u64::MAX;

pub const COOP_SGPR_BUF: usize = 129;

pub const COOP_SPILL_SLOTS: usize = 256;

pub struct VecKernel {
    code: super::super::native::jit::NativeCode,
    pub num_vgprs: usize,
    pub width: u32,

    pub min_private_bytes: usize,
    pub workgroup_x: Option<u32>,
    pub(crate) group: bool,
}
impl VecKernel {
    pub(in crate::rdna_spmd) fn from_code(
        code: super::super::native::jit::NativeCode,
        num_vgprs: usize,
        width: u32,
        min_private_bytes: usize,
        workgroup_x: Option<u32>,
        group: bool,
    ) -> Self {
        Self {
            code,
            num_vgprs,
            width,
            min_private_bytes,
            workgroup_x,
            group,
        }
    }

    pub unsafe fn run(
        &self,
        sgprs: *mut u32,
        vgprs: *mut u32,
        scratch_base: u64,
        scratch_stride: u64,
        valid_mask: u32,
        lds_base: u64,
    ) {
        if self.group {
            let f = std::mem::transmute::<u64, extern "C" fn(*mut u32, *mut u32, u64, u64, u32, u64)>(
                self.code.address(),
            );
            f(
                sgprs,
                vgprs,
                scratch_base,
                scratch_stride,
                valid_mask,
                lds_base,
            );
        } else {
            let f = std::mem::transmute::<u64, extern "C" fn(*mut u32, *mut u32, u64, u64)>(
                self.code.address(),
            );
            f(sgprs, vgprs, scratch_base, scratch_stride);
        }
    }
}

pub struct CoopVecKernel {
    pub(crate) yields: Vec<Vec<super::yields::YieldValues>>,
    pub(crate) registers: super::super::dialect::Registers,
    pub(in crate::rdna_spmd) code: super::super::native::jit::NativeCode,
    pub num_vgprs: usize,
    pub width: u32,

    pub min_private_bytes: usize,
    pub workgroup_x: Option<u32>,
}

impl CoopVecKernel {
    pub(in crate::rdna_spmd) fn from_code(
        code: super::super::native::jit::NativeCode,
        yields: Vec<Vec<super::yields::YieldValues>>,
        num_vgprs: usize,
        width: u32,
        min_private_bytes: usize,
        workgroup_x: Option<u32>,
        registers: super::super::dialect::Registers,
    ) -> Self {
        assert_eq!(registers.scc_slot as usize + 1, COOP_SGPR_BUF);
        Self {
            yields,
            registers,
            code,
            num_vgprs,
            width,
            min_private_bytes,
            workgroup_x,
        }
    }

    pub fn addr(&self) -> u64 {
        self.code.address()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduler {
    Independent,
    Wave,
    Workgroup,
}

pub(crate) enum Code {
    Scalar(ScalarKernel),
    Packet(VecKernel),
    Cooperative(CoopVecKernel),
}

pub struct Kernel {
    pub(crate) code: Code,
    scheduler: Scheduler,
    width: u32,
}

impl Kernel {
    pub(crate) fn new(code: Code, scheduler: Scheduler, width: u32) -> Self {
        Self {
            code,
            scheduler,
            width,
        }
    }
    pub fn width(&self) -> u32 {
        self.width
    }
    pub fn scheduler(&self) -> Scheduler {
        self.scheduler
    }
}
