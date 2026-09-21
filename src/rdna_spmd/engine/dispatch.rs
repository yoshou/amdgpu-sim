use crate::processor::KernelDescriptor;

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

#[inline]
pub fn setup_sgprs(
    s: &mut [u32],
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    scratch_base: u64,
    private_segment_size: u32,
    wg_id: (u32, u32, u32),
) {
    s.fill(0);
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
}
