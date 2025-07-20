#[derive(Debug, Clone)]
pub struct KernelDescriptor {
    pub group_segment_fixed_size: usize,
    pub private_segment_fixed_size: usize,
    pub max_flat_workgroup_size: usize,
    pub is_dynamic_call_stack: bool,
    pub is_xnack_enabled: bool,
    pub kernel_code_entry_byte_offset: usize,
    // compute_pgm_rsrc1
    pub granulated_workitem_vgpr_count: usize,
    pub granulated_wavefront_sgpr_count: usize,
    // priority: u8,
    // float_mode_round_32: u8,
    // float_mode_round_16_64: u8,
    // float_mode_denorm_32: u8,
    // float_mode_denorm_16_64: u8,
    // _priv: bool,
    // enable_dx10_clamp: bool,
    // debug_mode: bool,
    // enable_ieee_mode: bool,
    // bulky: bool,
    // cdbg_user: bool,
    // compute_pgm_rsrc2
    pub enable_sgpr_private_segment_wave_offset: bool,
    pub user_sgpr_count: usize,
    pub enable_trap_handler: bool,
    pub enable_sgpr_workgroup_id_x: bool,
    pub enable_sgpr_workgroup_id_y: bool,
    pub enable_sgpr_workgroup_id_z: bool,
    pub enable_sgpr_workgroup_info: bool,
    pub enable_vgpr_workitem_id: u8,
    // enable_exception_address_watch: bool,
    // enable_exception_memory: bool,
    // granulated_lds_size: usize,
    // enable_exception_ieee_754_fp_invalid_operation: bool,
    // enable_exception_fp_denormal_source: bool,
    // enable_exception_ieee_754_fp_division_by_zero: bool,
    // enable_exception_ieee_754_fp_overflow: bool,
    // enable_exception_ieee_754_fp_underflow: bool,
    // enable_exception_ieee_754_fp_inexact: bool,
    // enable_exception_int_divide_by_zero: bool,
    //
    pub enable_sgpr_private_segment_buffer: bool,
    pub enable_sgpr_dispatch_ptr: bool,
    pub enable_sgpr_queue_ptr: bool,
    pub enable_sgpr_kernarg_segment_ptr: bool,
    pub enable_sgpr_dispatch_id: bool,
    pub enable_sgpr_flat_scratch_init: bool,
    pub enable_sgpr_private_segment: bool,
    pub enable_sgpr_grid_workgroup_count_x: bool,
    pub enable_sgpr_grid_workgroup_count_y: bool,
    pub enable_sgpr_grid_workgroup_count_z: bool,
}

fn get_bit(buffer: &[u8], offset: usize, bit: usize) -> bool {
    ((buffer[offset + (bit >> 3)] >> (bit & 0x7)) & 1) == 1
}

fn get_bits(buffer: &[u8], offset: usize, bit: usize, size: usize) -> u8 {
    ((get_u32(buffer, offset + (bit >> 3)) >> (bit & 0x7)) & ((1 << size) - 1)) as u8
}

fn get_u8(buffer: &[u8], offset: usize) -> u8 {
    buffer[offset]
}

fn get_u16(buffer: &[u8], offset: usize) -> u16 {
    let b0 = buffer[offset] as u16;
    let b1 = buffer[offset + 1] as u16;

    b0 | (b1 << 8)
}

fn get_u32(buffer: &[u8], offset: usize) -> u32 {
    let b0 = buffer[offset] as u32;
    let b1 = buffer[offset + 1] as u32;
    let b2 = buffer[offset + 2] as u32;
    let b3 = buffer[offset + 3] as u32;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

fn get_u64(buffer: &[u8], offset: usize) -> u64 {
    let b0 = buffer[offset] as u64;
    let b1 = buffer[offset + 1] as u64;
    let b2 = buffer[offset + 2] as u64;
    let b3 = buffer[offset + 3] as u64;
    let b4 = buffer[offset + 4] as u64;
    let b5 = buffer[offset + 5] as u64;
    let b6 = buffer[offset + 6] as u64;
    let b7 = buffer[offset + 7] as u64;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
}

pub fn decode_kernel_desc(kd: &[u8]) -> KernelDescriptor {
    KernelDescriptor {
        group_segment_fixed_size: get_u32(kd, 0) as usize,
        private_segment_fixed_size: get_u32(kd, 4) as usize,
        max_flat_workgroup_size: get_u32(kd, 8) as usize,
        is_dynamic_call_stack: get_bit(kd, 12, 0),
        is_xnack_enabled: get_bit(kd, 12, 1),
        kernel_code_entry_byte_offset: get_u64(kd, 16) as usize,
        enable_sgpr_private_segment_buffer: get_bit(kd, 56, 0),
        enable_sgpr_dispatch_ptr: get_bit(kd, 56, 1),
        enable_sgpr_queue_ptr: get_bit(kd, 56, 2),
        enable_sgpr_kernarg_segment_ptr: get_bit(kd, 56, 3),
        enable_sgpr_dispatch_id: get_bit(kd, 56, 4),
        enable_sgpr_flat_scratch_init: get_bit(kd, 56, 5),
        enable_sgpr_private_segment: get_bit(kd, 56, 6),
        enable_sgpr_grid_workgroup_count_x: get_bit(kd, 56, 7),
        enable_sgpr_grid_workgroup_count_y: get_bit(kd, 57, 0),
        enable_sgpr_grid_workgroup_count_z: get_bit(kd, 57, 1),
        granulated_workitem_vgpr_count: (get_bits(kd, 48, 0, 6) as usize + 1) * 4,
        granulated_wavefront_sgpr_count: (get_bits(kd, 48, 6, 4) as usize + 1) * 8,
        enable_sgpr_private_segment_wave_offset: get_bit(kd, 52, 0),
        user_sgpr_count: get_bits(kd, 52, 1, 5) as usize,
        enable_trap_handler: get_bit(kd, 52, 6),
        enable_sgpr_workgroup_id_x: get_bit(kd, 52, 7),
        enable_sgpr_workgroup_id_y: get_bit(kd, 52, 8),
        enable_sgpr_workgroup_id_z: get_bit(kd, 52, 9),
        enable_sgpr_workgroup_info: get_bit(kd, 52, 10),
        enable_vgpr_workitem_id: get_bits(kd, 52, 11, 2),
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Pointer<'a> {
    pub object: &'a [u8],
    pub offset: usize,
}

impl<'a> Pointer<'a> {
    pub fn new(object: &'a [u8], offset: usize) -> Self {
        Pointer {
            object: object,
            offset: offset,
        }
    }

    pub fn address(&self) -> u64 {
        (&self.object[0] as *const u8) as u64
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct HsaKernelDispatchPacket<'a> {
    pub header: u16,
    pub setup: u16,
    pub workgroup_size_x: u16,
    pub workgroup_size_y: u16,
    pub workgroup_size_z: u16,
    pub grid_size_x: u32,
    pub grid_size_y: u32,
    pub grid_size_z: u32,
    pub private_segment_size: u32,
    pub group_segment_size: u32,
    pub kernel_object: Pointer<'a>,
    pub kernarg_address: Pointer<'a>,
    // hsa_signal_t completion_signal;
}
