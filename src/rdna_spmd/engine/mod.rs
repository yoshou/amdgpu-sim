pub mod fiber;
pub(super) mod barrier;
pub(super) mod cooperative;
pub(super) mod dispatch;
pub(super) mod kernel;
pub(super) mod wmma;
pub(super) mod xlane;
pub(super) mod yields;

use crate::processor::KernelDescriptor;
use kernel::{Code, Kernel, Scheduler};

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
    match (&kernel.code, kernel.scheduler()) {
        (Code::Scalar(k), _) => dispatch::dispatch_parallel(k, kd, kernarg_ptr, aql_packet_addr, dims, private_segment_size, num_threads),
        (Code::Packet(k), _) => dispatch::dispatch_parallel_vec(k, kd, kernarg_ptr, aql_packet_addr, dims, private_segment_size, num_threads),
        (Code::Cooperative(k), Scheduler::Wave) => xlane::dispatch_xlane_vec(k, kd, kernarg_ptr, aql_packet_addr, dims, private_segment_size, num_threads),
        (Code::Cooperative(k), Scheduler::Workgroup) => cooperative::dispatch_cooperative_vec(k, kd, kernarg_ptr, aql_packet_addr, dims, private_segment_size, group_segment_size, num_threads),
        (Code::Cooperative(_), Scheduler::Independent) => unreachable!("cooperative code needs a wave or workgroup scheduler"),
    }
}
