pub const SGPR_BUF: usize = 129;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheduler {
    Independent,
    Wave,
    Workgroup,
}

pub struct Region {
    pub address: u64,
    pub scheduler: Scheduler,
    pub children: Vec<usize>,
}

pub struct Kernel {
    _code: super::super::native::jit::NativeCode,
    pub(super) regions: Vec<Region>,
    pub(super) yields: Vec<Vec<super::super::codegen::YieldValues>>,
    pub(super) registers: super::super::ir::Registers,
    pub(super) frame_words: usize,
    pub(super) num_vgprs: usize,
    pub(super) min_private_bytes: usize,
    pub(super) workgroup_x: Option<u32>,
    pub(super) scheduler: Scheduler,
    pub(super) width: u32,
}

impl Kernel {
    pub(crate) fn new(
        code: super::super::native::jit::NativeCode,
        regions: Vec<Region>,
        yields: Vec<Vec<super::super::codegen::YieldValues>>,
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
}
