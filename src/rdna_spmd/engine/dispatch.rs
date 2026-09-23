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
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EntryLayout {
    pub private_segment_buffer: Option<u32>,
    pub dispatch_ptr: Option<u32>,
    pub kernarg_ptr: Option<u32>,
    pub flat_scratch_init: Option<u32>,
    pub workgroup_info: Option<u32>,
    pub private_segment_wave_offset: Option<u32>,
    pub workgroup_id_x: bool,
    pub workgroup_id_yz: bool,
}

pub const WORKGROUP_ID_X: u32 = 117;
pub const WORKGROUP_ID_YZ: u32 = 115;

impl EntryLayout {
    pub fn of(kd: &KernelDescriptor) -> Self {
        let mut layout = Self::default();
        let mut p = 0u32;
        let mut take = |enabled: bool, count: u32| {
            let at = enabled.then_some(p);
            if enabled {
                p += count;
            }
            at
        };
        layout.private_segment_buffer = take(kd.enable_sgpr_private_segment_buffer, 4);
        layout.dispatch_ptr = take(kd.enable_sgpr_dispatch_ptr, 2);
        take(kd.enable_sgpr_queue_ptr, 2);
        layout.kernarg_ptr = take(kd.enable_sgpr_kernarg_segment_ptr, 2);
        take(kd.enable_sgpr_dispatch_id, 2);
        layout.flat_scratch_init = take(kd.enable_sgpr_flat_scratch_init, 2);
        for enabled in [
            kd.enable_sgpr_grid_workgroup_count_x,
            kd.enable_sgpr_grid_workgroup_count_y,
            kd.enable_sgpr_grid_workgroup_count_z,
        ] {
            if enabled && p < 16 {
                p += 1;
            }
        }
        layout.workgroup_id_x = kd.enable_sgpr_workgroup_id_x;
        layout.workgroup_id_yz = kd.enable_sgpr_workgroup_id_y || kd.enable_sgpr_workgroup_id_z;
        layout.workgroup_info = kd.enable_sgpr_workgroup_info.then_some(p);
        if kd.enable_sgpr_workgroup_info {
            p += 1;
        }
        layout.private_segment_wave_offset = kd.enable_sgpr_private_segment_wave_offset.then_some(p);
        layout
    }
}

#[inline]
pub fn setup_sgprs(
    s: &mut [u32],
    layout: &EntryLayout,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    scratch_base: u64,
    private_segment_size: u32,
    wg_id: (u32, u32, u32),
) {
    s.fill(0);
    if let Some(p) = layout.private_segment_buffer {
        let mut w0: u64 = 0;
        w0 |= scratch_base & ((1 << 48) - 1);
        w0 |= ((private_segment_size as u64) & ((1 << 14) - 1)) << 48;
        s[p as usize] = w0 as u32;
        s[p as usize + 1] = (w0 >> 32) as u32;
    }
    if let Some(p) = layout.dispatch_ptr {
        s[p as usize] = aql_packet_addr as u32;
        s[p as usize + 1] = (aql_packet_addr >> 32) as u32;
    }
    if let Some(p) = layout.kernarg_ptr {
        s[p as usize] = kernarg_ptr as u32;
        s[p as usize + 1] = (kernarg_ptr >> 32) as u32;
    }
    if let Some(p) = layout.flat_scratch_init {
        s[p as usize] = 0;
        s[p as usize + 1] = private_segment_size;
    }
    if layout.workgroup_id_x {
        s[WORKGROUP_ID_X as usize] = wg_id.0;
    }
    if layout.workgroup_id_yz {
        s[WORKGROUP_ID_YZ as usize] = (wg_id.2 << 16) | wg_id.1;
    }
    if let Some(p) = layout.workgroup_info {
        s[p as usize] = 0;
    }
    if let Some(p) = layout.private_segment_wave_offset {
        s[p as usize] = 0;
    }
}
