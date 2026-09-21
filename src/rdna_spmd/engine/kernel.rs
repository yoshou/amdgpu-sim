pub const SGPR_BUF: usize = 129;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduler {
    Independent,
    Wave,
    Workgroup,
}

pub(crate) struct Region {
    pub address: u64,
    pub scheduler: Scheduler,
    pub children: Vec<usize>,
}

pub struct Kernel {
    _code: super::super::native::jit::NativeCode,
    pub(crate) regions: Vec<Region>,
    pub(crate) yields: Vec<Vec<super::super::codegen::yields::YieldValues>>,
    pub(crate) registers: super::super::ir::Registers,
    pub(crate) frame_words: usize,
    pub(crate) num_vgprs: usize,
    pub(crate) min_private_bytes: usize,
    pub(crate) workgroup_x: Option<u32>,
    scheduler: Scheduler,
    width: u32,
}

impl Kernel {
    pub(in crate::rdna_spmd) fn new(
        code: super::super::native::jit::NativeCode,
        regions: Vec<Region>,
        yields: Vec<Vec<super::super::codegen::yields::YieldValues>>,
        registers: super::super::ir::Registers,
        frame_words: usize,
        num_vgprs: usize,
        min_private_bytes: usize,
        workgroup_x: Option<u32>,
        scheduler: Scheduler,
        width: u32,
    ) -> Self {
        assert!(!regions.is_empty(), "a kernel without its outermost region");
        assert_eq!(registers.scc_slot as usize + 1, SGPR_BUF);
        Self {
            _code: code,
            regions,
            yields,
            registers,
            frame_words,
            num_vgprs,
            min_private_bytes,
            workgroup_x,
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
