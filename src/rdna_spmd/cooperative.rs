//! Workgroup-cooperative dispatch for kernels with shared LDS + barriers.
//!
//! Unlike [`dispatch_parallel`](super::dispatch::dispatch_parallel), which runs
//! every work-item independently to completion, a cooperative kernel
//! synchronizes its work-items at `s_barrier` and communicates through shared
//! LDS. This scheduler mirrors the masked-vector reference
//! ([`RDNAProcessor`](crate::rdna_processor)): each work-item is a resumable
//! coroutine ([`CoopKernel`] yields at each barrier, returning its resume pc),
//! and one host thread drives a whole workgroup round-robin, advancing every
//! work-item one *barrier generation* per pass over shared, zeroed LDS.
//!
//! For kernels with uniform barrier participation, no work-item leaves
//! generation K until every live work-item has been run to that generation's
//! barrier in the current pass. The scheduler assumes that work-items do not
//! terminate early or reach barriers non-uniformly; it does not validate those
//! conditions.

use std::thread;

use crate::processor::KernelDescriptor;

use super::dispatch::{setup_sgprs, GridDims};
use super::emit::{CoopKernel, COOP_DONE, COOP_SGPR_BUF, COOP_SPILL_SLOTS};

const EXEC: usize = 126;
const SCC: usize = 128; // reserved sgprs slot for persisted SCC (see emit.rs)

/// Run a cooperative kernel over the whole grid across `num_threads` CPU threads.
/// Each workgroup runs entirely on one thread with its own zeroed LDS buffer of
/// `group_segment_size` bytes.
pub fn dispatch_cooperative(
    kernel: &CoopKernel,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    group_segment_size: usize,
    num_threads: usize,
) {
    let num_threads = num_threads.max(1);

    let wg_size = dims.workgroup_size() as usize;
    let num_wg = (dims.num_wg_x * dims.num_wg_y * dims.num_wg_z) as u64;
    let num_vgprs = kernel.num_vgprs.max(1);
    let scratch_u64 = (private_segment_size as usize / 8) + 2;
    // The kernel descriptor often reports 0 here (LDS rounded/allocated
    // dynamically), so — like the vector RDNAProcessor, which allocates a fixed
    // 128 KiB LDS regardless — fall back to that size.
    let lds_bytes = group_segment_size.max(128 * 1024);
    let entry_pc = kernel.entry_pc as u64;

    thread::scope(|scope| {
        for tid in 0..num_threads {
            let kernel = &kernel;
            let kd = &kd;
            let dims = dims;
            scope.spawn(move || {
                // Per-work-item register/scratch state, reused across the
                // workgroups this thread owns.
                let mut sgprs: Vec<[u32; COOP_SGPR_BUF]> = vec![[0u32; COOP_SGPR_BUF]; wg_size];
                // Dedicated per-work-item lane-spill buffer (NOT architectural
                // registers) for the uniform writelane/readlane idiom; must
                // persist across barrier yields.
                let mut spill: Vec<[u32; COOP_SPILL_SLOTS]> = vec![[0u32; COOP_SPILL_SLOTS]; wg_size];
                let mut vgprs: Vec<Vec<u32>> = vec![vec![0u32; num_vgprs]; wg_size];
                let scratch: Vec<Vec<u64>> = vec![vec![0u64; scratch_u64]; wg_size];
                let mut resume: Vec<u64> = vec![0; wg_size];
                let mut done: Vec<bool> = vec![false; wg_size];
                let mut lds: Vec<u8> = vec![0u8; lds_bytes];

                let mut wg = tid as u64;
                while wg < num_wg {
                    let wg_id = (
                        (wg % dims.num_wg_x as u64) as u32,
                        ((wg / dims.num_wg_x as u64) % dims.num_wg_y as u64) as u32,
                        ((wg / (dims.num_wg_x as u64 * dims.num_wg_y as u64)) % dims.num_wg_z as u64) as u32,
                    );

                    // Zero shared LDS for this workgroup.
                    for b in lds.iter_mut() {
                        *b = 0;
                    }
                    let lds_base = lds.as_mut_ptr() as u64;

                    // Initialize every work-item's state.
                    for wi in 0..wg_size {
                        let scratch_base = scratch[wi].as_ptr() as u64;
                        let s = setup_sgprs(
                            kd,
                            kernarg_ptr,
                            aql_packet_addr,
                            scratch_base,
                            private_segment_size,
                            wg_id,
                        );
                        sgprs[wi][..128].copy_from_slice(&s);
                        sgprs[wi][EXEC] = 1; // single active lane
                        sgprs[wi][SCC] = 0;
                        // Fresh lane-spill buffer per workgroup so a reused
                        // thread's prior workgroup does not leak spilled values.
                        spill[wi] = [0u32; COOP_SPILL_SLOTS];

                        // Local work-item id (x,y,z) packed into VGPR0.
                        let lx = (wi as u32) % dims.wg_x;
                        let ly = ((wi as u32) / dims.wg_x) % dims.wg_y;
                        let lz = (wi as u32) / (dims.wg_x * dims.wg_y);
                        for v in vgprs[wi].iter_mut() {
                            *v = 0;
                        }
                        vgprs[wi][0] = lx | (ly << 10) | (lz << 20);

                        resume[wi] = entry_pc;
                        done[wi] = false;
                    }

                    // Cooperative round-robin: each pass advances every live
                    // work-item to its next barrier (or to s_endpgm).
                    let mut pass = 0u64;
                    loop {
                        let mut any = false;
                        for wi in 0..wg_size {
                            if done[wi] {
                                continue;
                            }
                            any = true;
                            let scratch_base = scratch[wi].as_ptr() as u64;
                            let r = unsafe {
                                kernel.run(
                                    sgprs[wi].as_mut_ptr(),
                                    vgprs[wi].as_mut_ptr(),
                                    scratch_base,
                                    lds_base,
                                    spill[wi].as_mut_ptr(),
                                    resume[wi],
                                )
                            };
                            if r == COOP_DONE {
                                done[wi] = true;
                            } else {
                                resume[wi] = r;
                            }
                        }
                        if !any {
                            break;
                        }
                        pass += 1;
                        if pass > 100_000 {
                            panic!("dispatch_cooperative: barrier round-robin did not converge \
                                    (wg={}, {} passes) — non-uniform barrier or bad resume", wg, pass);
                        }
                    }

                    wg += num_threads as u64;
                }
            });
        }
    });
}
