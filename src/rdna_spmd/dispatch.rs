//! OpenMP-style parallel dispatch of independent scalar work-items.
//!
//! The target kernel has no cross-lane ops and no barriers (see the M1 scan),
//! so every work-item is independent: we map the grid onto all CPU cores and
//! run one [`ScalarKernel`] invocation per work-item. Per-work-item register
//! files and scratch are thread-local; output is written by the kernel itself
//! through global stores into the (disjoint, per-pixel) result buffer.

use std::thread;

use crate::processor::KernelDescriptor;

use super::emit::ScalarKernel;
use super::emit_vec::VecKernel;

/// Grid geometry (workgroup counts and per-workgroup sizes).
#[derive(Clone, Copy)]
pub struct GridDims {
    pub num_wg_x: u32,
    pub num_wg_y: u32,
    pub num_wg_z: u32,
    pub wg_x: u32,
    pub wg_y: u32,
    pub wg_z: u32,
}

impl GridDims {
    pub fn workgroup_size(&self) -> u32 {
        self.wg_x * self.wg_y * self.wg_z
    }
    pub fn total_workitems(&self) -> u64 {
        (self.num_wg_x * self.num_wg_y * self.num_wg_z) as u64 * self.workgroup_size() as u64
    }
}

/// Build the 128-entry SGPR file for one work-item, mirroring the masked
/// backend's `dispatch()` system-SGPR layout.
pub(super) fn setup_sgprs(
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    scratch_base: u64,
    private_segment_size: u32,
    wg_id: (u32, u32, u32),
) -> [u32; 128] {
    let mut s = [0u32; 128];
    let mut p = 0usize;

    if kd.enable_sgpr_private_segment_buffer {
        let mut w0: u64 = 0;
        w0 |= scratch_base & ((1 << 48) - 1);
        w0 |= ((private_segment_size as u64) & ((1 << 14) - 1)) << 48;
        s[0] = w0 as u32;
        s[1] = (w0 >> 32) as u32;
        p += 4;
    }
    if kd.enable_sgpr_dispatch_ptr {
        s[p] = aql_packet_addr as u32;
        s[p + 1] = (aql_packet_addr >> 32) as u32;
        p += 2;
    }
    if kd.enable_sgpr_queue_ptr {
        p += 2;
    }
    if kd.enable_sgpr_kernarg_segment_ptr {
        s[p] = kernarg_ptr as u32;
        s[p + 1] = (kernarg_ptr >> 32) as u32;
        p += 2;
    }
    if kd.enable_sgpr_dispatch_id {
        p += 2;
    }
    if kd.enable_sgpr_flat_scratch_init {
        // Each work-item owns its scratch buffer, so the offset is 0.
        s[p] = 0;
        s[p + 1] = private_segment_size;
        p += 2;
    }
    if kd.enable_sgpr_grid_workgroup_count_x && p < 16 {
        p += 1;
    }
    if kd.enable_sgpr_grid_workgroup_count_y && p < 16 {
        p += 1;
    }
    if kd.enable_sgpr_grid_workgroup_count_z && p < 16 {
        p += 1;
    }
    // Workgroup IDs are delivered in architected high SGPRs (TTMP), matching the
    // masked dispatch: sgpr117 = wgid_x, sgpr115 = (wgid_z << 16) | wgid_y.
    if kd.enable_sgpr_workgroup_id_x {
        s[117] = wg_id.0;
    }
    if kd.enable_sgpr_workgroup_id_y || kd.enable_sgpr_workgroup_id_z {
        s[115] = (wg_id.2 << 16) | wg_id.1;
    }
    if kd.enable_sgpr_workgroup_info {
        s[p] = 0;
        p += 1;
    }
    if kd.enable_sgpr_private_segment_wave_offset {
        s[p] = 0;
    }
    s
}

/// Run the whole grid in parallel across `num_threads` CPU threads.
pub fn dispatch_parallel(
    kernel: &ScalarKernel,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    num_threads: usize,
) {
    let total = dims.total_workitems();
    let wg_size = dims.workgroup_size();
    let num_vgprs = kernel.num_vgprs.max(1);
    // Scratch in u64 units for 8-byte alignment; round up to cover the segment.
    let scratch_u64 = (private_segment_size as usize / 8) + 2;

    thread::scope(|scope| {
        for tid in 0..num_threads {
            let kernel = &kernel;
            let kd = &kd;
            let dims = dims;
            scope.spawn(move || {
                let mut sgprs;
                let mut vgprs = vec![0u32; num_vgprs];
                // Scratch must be 4 GiB-aligned so its low 32 address bits are 0:
                // kernels using flat-scratch addressing read SRC_PRIVATE_BASE and
                // force the low word to 0 when forming flat pointers into private
                // memory (matches the interpreter's `AVec<u8, ConstAlign<4GiB>>`).
                let mut scratch: aligned_vec::AVec<u8, aligned_vec::ConstAlign<0x1_0000_0000>> =
                    aligned_vec::AVec::new(0x1_0000_0000);
                scratch.resize(scratch_u64 * 8, 0u8);

                let mut t = tid as u64;
                while t < total {
                    let wg = (t / wg_size as u64) as u32;
                    let local = (t % wg_size as u64) as u32;

                    let wg_id = (
                        wg % dims.num_wg_x,
                        (wg / dims.num_wg_x) % dims.num_wg_y,
                        (wg / (dims.num_wg_x * dims.num_wg_y)) % dims.num_wg_z,
                    );
                    let lx = local % dims.wg_x;
                    let ly = (local / dims.wg_x) % dims.wg_y;
                    let lz = (local / (dims.wg_x * dims.wg_y)) % dims.wg_z;

                    let scratch_base = scratch.as_ptr() as u64;
                    sgprs = setup_sgprs(
                        kd,
                        kernarg_ptr,
                        aql_packet_addr,
                        scratch_base,
                        private_segment_size,
                        wg_id,
                    );

                    // VGPR0: packed work-item id (x | y<<10 | z<<20).
                    for v in vgprs.iter_mut() {
                        *v = 0;
                    }
                    vgprs[0] = lx | (ly << 10) | (lz << 20);

                    unsafe {
                        kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), scratch_base);
                    }

                    t += num_threads as u64;
                }
            });
        }
    });
}

/// Width-W work-item packing (SPMD-on-SIMD): each [`VecKernel`] call runs `W`
/// work-items of one workgroup at once (one per SIMD lane). VGPRs are laid out
/// SoA — register `r`'s W lanes are contiguous at `r*W` — and each lane gets its
/// own private scratch segment.
pub fn dispatch_parallel_vec(
    kernel: &VecKernel,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    num_threads: usize,
) {
    dispatch_parallel_vec_impl(
        kernel,
        kd,
        kernarg_ptr,
        aql_packet_addr,
        dims,
        private_segment_size,
        num_threads,
    );
}

fn linear_local_id(dims: GridDims, width: u32, packet: u32, lane: u32) -> (u32, u32, u32) {
    let linear = packet * width + lane;
    (
        linear % dims.wg_x,
        (linear / dims.wg_x) % dims.wg_y,
        linear / (dims.wg_x * dims.wg_y),
    )
}

fn dispatch_parallel_vec_impl(
    kernel: &VecKernel,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    num_threads: usize,
) {
    let w = kernel.width as u64;
    let wg_size = dims.workgroup_size() as u64;
    assert!(wg_size % w == 0, "workgroup size {} not divisible by W={}", wg_size, w);
    let num_wg = (dims.num_wg_x * dims.num_wg_y * dims.num_wg_z) as u64;
    let groups_per_wg = wg_size / w;
    let total_groups = num_wg * groups_per_wg;
    let num_vgprs = kernel.num_vgprs.max(1);
    // Per-lane padded scratch segment (u64 units); stride in bytes spaces the W
    // per-lane segments so each work-item owns disjoint private memory.
    let scratch_u64 = (private_segment_size as usize / 8) + 2;
    let stride_bytes = (scratch_u64 * 8) as u64;

    thread::scope(|scope| {
        for tid in 0..num_threads {
            let kernel = &kernel;
            let kd = &kd;
            let dims = dims;
            scope.spawn(move || {
                let mut sgprs;
                let mut vgprs = vec![0u32; num_vgprs * w as usize];
                // 4 GiB-aligned so the base's low 32 bits are 0: kernels using
                // flat-scratch addressing read SRC_PRIVATE_BASE for the flat
                // pointer's high word and add the per-lane low offset, so a
                // nonzero base low word would corrupt every private pointer
                // (matches the scalar path and the interpreter's aligned AVec).
                let mut scratch: aligned_vec::AVec<u8, aligned_vec::ConstAlign<0x1_0000_0000>> =
                    aligned_vec::AVec::new(0x1_0000_0000);
                scratch.resize(scratch_u64 * w as usize * 8, 0u8);

                let mut g = tid as u64;
                while g < total_groups {
                    let wg = g / groups_per_wg;
                    let packet = (g % groups_per_wg) as u32;

                    let wg_id = (
                        (wg % dims.num_wg_x as u64) as u32,
                        ((wg / dims.num_wg_x as u64) % dims.num_wg_y as u64) as u32,
                        ((wg / (dims.num_wg_x as u64 * dims.num_wg_y as u64)) % dims.num_wg_z as u64) as u32,
                    );

                    let scratch_base = scratch.as_ptr() as u64;
                    sgprs = setup_sgprs(kd, kernarg_ptr, aql_packet_addr, scratch_base, private_segment_size, wg_id);

                    // VGPR0 lane k = packed linear local work-item id.
                    for v in vgprs.iter_mut() { *v = 0; }
                    for k in 0..w {
                        let (lx, ly, lz) = linear_local_id(dims, kernel.width, packet, k as u32);
                        vgprs[k as usize] = lx | (ly << 10) | (lz << 20); // register 0, lane k
                    }

                    unsafe {
                        kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), scratch_base, stride_bytes);
                    }
                    g += num_threads as u64;
                }
            });
        }
    });
}
