//! Workgroup-cooperative dispatch for kernels with shared LDS + barriers.
//!
//! Scalar and packet native code share a fiber ABI and the same scheduler.
//! Each worker owns complete workgroups, including zeroed LDS and barrier
//! rounds. Native SSA values survive suspension; only effect operands/results
//! are exchanged with the scheduler.
//!
//! Signal and wait are separate typed effects. Barrier rounds count waves;
//! signal-is-first reports the first wave in that round. The driver validates
//! uniform yield PCs within each wave and diagnoses barrier deadlocks.

use std::thread;

use crate::processor::KernelDescriptor;

use super::dispatch::{setup_sgprs, GridDims};
use super::kernel::{COOP_SGPR_BUF, COOP_SPILL_SLOTS};


/// Run a cooperative kernel over the whole grid across `num_threads` CPU threads.
/// Each workgroup runs entirely on one thread with its own zeroed LDS buffer of
/// `group_segment_size` bytes.
/// The existing packet fiber ABI extended with workgroup LDS and typed barrier
/// rounds. Every worker owns complete workgroups; only yield operands/results
/// cross the packet/host boundary.
pub(crate) fn dispatch_cooperative_vec(
    kernel: &super::kernel::CoopVecKernel,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    group_segment_size: usize,
    num_threads: usize,
) {
    use super::fiber::{Fiber, KernelArgs, FIBER_DONE};
    use super::super::ir::EffectOp;
    let width = kernel.width as usize;
    let exec = kernel.registers.exec as usize;
    let ppw = 32 / width;
    let wg_size = dims.workgroup_size() as usize;
    let waves = wg_size.div_ceil(32);
    let packets = waves * ppw;
    let num_wg = dims.num_wg_x as u64 * dims.num_wg_y as u64 * dims.num_wg_z as u64;
    if num_wg == 0 { return; }
    // Each worker owns whole workgroups. Do not allocate LDS and suspended
    // stacks for workers to which no workgroup can be assigned.
    let threads = num_threads.max(1).min(num_wg as usize);
    let stride = (private_segment_size as usize).max(kernel.min_private_bytes).div_ceil(16) * 16;
    thread::scope(|scope| {
        for tid in 0..threads {
            scope.spawn(move || {
                let mut sgprs = vec![[0u32; COOP_SGPR_BUF]; packets];
                let mut vgprs = vec![vec![0u32; kernel.num_vgprs * width]; packets];
                let mut spill = vec![vec![0u32; COOP_SPILL_SLOTS]; packets];
                let mut fibers = Fiber::batch(packets, 32 << 10);
                let mut scratch =
                    aligned_vec::AVec::<u8, aligned_vec::ConstAlign<0x1_0000_0000>>::new(
                        0x1_0000_0000,
                    );
                scratch.resize(waves * 32 * stride, 0);
                // Existing code objects can declare zero fixed LDS while using
                // the processor's dynamic 128 KiB aperture.
                let mut lds = vec![0u8; group_segment_size.max(128 * 1024)];
                let mut wg = tid as u64;
                while wg < num_wg {
                    let wg_id = (
                        (wg % dims.num_wg_x as u64) as u32,
                        ((wg / dims.num_wg_x as u64) % dims.num_wg_y as u64) as u32,
                        (wg / (dims.num_wg_x as u64 * dims.num_wg_y as u64)) as u32,
                    );
                    scratch.fill(0);
                    lds.fill(0);
                    let mut done = vec![false; packets];
                    let mut resume = vec![0u64; packets];
                    let mut waiting = vec![None; waves];
                    let mut barriers = super::barrier::Barriers::new(waves);
                    for p in 0..packets {
                        let local = p * width;
                        let valid = wg_size.saturating_sub(local).min(width);
                        sgprs[p] = [0; COOP_SGPR_BUF];
                        vgprs[p].fill(0);
                        spill[p].fill(0);
                        let wave = p / ppw;
                        let sb = if stride == 0 { 0 } else { (unsafe { scratch.as_ptr().add(wave * 32 * stride) }) as u64 };
                        sgprs[p][..128].copy_from_slice(&setup_sgprs(
                            kd,
                            kernarg_ptr,
                            aql_packet_addr,
                            sb,
                            private_segment_size,
                            wg_id,
                        ));
                        sgprs[p][exec] = ((1u64 << valid) - 1) as u32;
                        done[p] = valid == 0;
                        for lane in 0..valid {
                            let wi = (local + lane) as u32;
                            let x = wi % dims.wg_x;
                            let y = (wi / dims.wg_x) % dims.wg_y;
                            let z = wi / (dims.wg_x * dims.wg_y);
                            vgprs[p][lane] = x | (y << 10) | (z << 20);
                        }
                        if !done[p] {
                            fibers[p].start(KernelArgs {
                                entry: kernel.addr(),
                                sgprs: sgprs[p].as_mut_ptr(),
                                vgprs: vgprs[p].as_mut_ptr(),
                                spill: spill[p].as_mut_ptr(),
                                scratch_base: sb,
                                scratch_stride: stride as u64,
                                lane_base: ((p % ppw) * width) as u64,
                                valid_mask: ((1u64 << valid) - 1) as u32,
                                lds_base: lds.as_mut_ptr() as u64,
                            });
                        }
                    }
                    let mut passes = 0;
                    loop {
                        let mut progress = false;
                        let mut live = false;
                        for wave in 0..waves {
                            let range = wave * ppw..(wave + 1) * ppw;
                            if range.clone().all(|p| done[p]) {
                                continue;
                            }
                            live = true;
                            if let Some(id) = waiting[wave] {
                                if !barriers.wait(wave, id) {
                                    continue;
                                }
                                waiting[wave] = None;
                            }
                            progress = true;
                            let mut pc = None;
                            for p in range.clone() {
                                if !done[p] {
                                    let r = fibers[p].resume();
                                    if let Some(other) = pc {
                                        assert_eq!(other, r, "nonuniform packet yield");
                                    } else {
                                        pc = Some(r);
                                    }
                                    resume[p] = r;
                                    done[p] = r == FIBER_DONE;
                                }
                            }
                            let pc = pc.unwrap();
                            if pc == FIBER_DONE {
                                continue;
                            }
                            let action = kernel
                                .yields
                                .get(pc as usize)
                                .expect("packet yield lacks typed effect");
                            if action.is_wave() {
                                let count = wg_size.saturating_sub(wave * 32).min(32);
                                let valid = if count == 32 {
                                    u32::MAX
                                } else {
                                    (1u32 << count) - 1
                                };
                                action.apply_wave(width, valid, &fibers[range]);
                            } else {
                                let count = wg_size.saturating_sub(wave * 32).min(32);
                                let valid = if count == 32 { u32::MAX } else { (1u32 << count) - 1 };
                                let id = action.uniform_id(width, valid, &fibers[range.clone()]) & 31;
                                match action.op {
                                    EffectOp::BarrierSignal { is_first } => {
                                        let first = barriers.signal(wave, id);
                                        if is_first { action.broadcast_result(width, valid, &fibers[range], first as u32); }
                                    }
                                    EffectOp::BarrierWait => waiting[wave] = Some(id),
                                    _ => unreachable!(),
                                }
                            }
                        }
                        if !live {
                            break;
                        }
                        assert!(progress, "workgroup barrier deadlock");
                        passes += 1;
                        assert!(passes < 100_000, "packet dispatch did not converge");
                    }
                    wg += threads as u64;
                }
            });
        }
    });
}
