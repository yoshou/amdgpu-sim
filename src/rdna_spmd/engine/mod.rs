mod barrier;
mod dispatch;
mod fiber;
mod kernel;
mod scheduler;
mod wmma;
mod yields;

pub use dispatch::{EntryLayout, GridDims};
#[cfg(test)]
pub use dispatch::{Field, WORKGROUP_ID_X};
pub use fiber::yield_address;
pub use kernel::{Kernel, Region, Scheduler};
pub use wmma::warm_wmma;

pub fn dispatch(
    kernel: &Kernel,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    group_segment_size: usize,
    num_threads: usize,
) {
    scheduler::run(
        scheduler::View::of(kernel),
        kernarg_ptr,
        aql_packet_addr,
        dims,
        private_segment_size,
        group_segment_size,
        num_threads,
    )
}
