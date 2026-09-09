pub mod fiber;
pub(super) mod barrier;
pub(super) mod dispatch;
pub(super) mod kernel;
pub(super) mod scheduler;
pub(super) mod wmma;
pub(super) mod xlane;
pub(super) mod yields;

use crate::processor::KernelDescriptor;
use kernel::Kernel;

pub fn dispatch(
    kernel: &Kernel,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: dispatch::GridDims,
    private_segment_size: u32,
    group_segment_size: usize,
    num_threads: usize,
) {
    scheduler::run(scheduler::View::of(kernel), kd, kernarg_ptr, aql_packet_addr, dims, private_segment_size, group_segment_size, num_threads)
}
