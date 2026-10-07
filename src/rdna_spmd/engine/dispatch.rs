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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field {
    pub register: u32,
    pub shift: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EntryLayout {
    pub private_segment_buffer: Option<u32>,
    pub dispatch_ptr: Option<u32>,
    pub kernarg_ptr: Option<u32>,
    pub flat_scratch_init: Option<u32>,
    pub workgroup_info: Option<u32>,
    pub private_segment_wave_offset: Option<u32>,
    pub workgroup_ids: [Option<Field>; 3],
    pub workitem_ids: [Option<Field>; 3],
}

pub const WORKGROUP_ID_X: u32 = 117;
pub const WORKGROUP_ID_YZ: u32 = 115;

const fn field(register: u32, shift: u32) -> Option<Field> {
    Some(Field { register, shift })
}

impl EntryLayout {
    pub const PACKED: [Option<Field>; 3] = [field(0, 0), field(0, 10), field(0, 20)];

    fn user(kd: &KernelDescriptor) -> (Self, u32) {
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
        (layout, p)
    }

    pub fn of(kd: &KernelDescriptor) -> Self {
        let (mut layout, mut p) = Self::user(kd);
        if kd.enable_sgpr_workgroup_id_x {
            layout.workgroup_ids[0] = field(WORKGROUP_ID_X, 0);
        }
        if kd.enable_sgpr_workgroup_id_y || kd.enable_sgpr_workgroup_id_z {
            layout.workgroup_ids[1] = field(WORKGROUP_ID_YZ, 0);
            layout.workgroup_ids[2] = field(WORKGROUP_ID_YZ, 16);
        }
        layout.workitem_ids = Self::PACKED;
        layout.workgroup_info = kd.enable_sgpr_workgroup_info.then_some(p);
        if kd.enable_sgpr_workgroup_info {
            p += 1;
        }
        layout.private_segment_wave_offset = kd.enable_sgpr_private_segment_wave_offset.then_some(p);
        layout
    }

    pub fn separate(kd: &KernelDescriptor) -> Result<Self, String> {
        let (mut layout, mut p) = Self::user(kd);
        if p as usize != kd.user_sgpr_count {
            return Err(format!(
                "the descriptor counts {} user SGPRs but enables {}",
                kd.user_sgpr_count, p
            ));
        }
        let enabled = [
            kd.enable_sgpr_workgroup_id_x,
            kd.enable_sgpr_workgroup_id_y,
            kd.enable_sgpr_workgroup_id_z,
        ];
        for (axis, &on) in enabled.iter().enumerate() {
            if on {
                layout.workgroup_ids[axis] = field(p, 0);
                p += 1;
            }
        }
        layout.workgroup_info = kd.enable_sgpr_workgroup_info.then_some(p);
        if kd.enable_sgpr_workgroup_info {
            p += 1;
        }
        layout.private_segment_wave_offset = kd.enable_sgpr_private_segment_wave_offset.then_some(p);
        for axis in 0..=kd.enable_vgpr_workitem_id.min(2) as usize {
            layout.workitem_ids[axis] = field(axis as u32, 0);
        }
        Ok(layout)
    }

    pub fn workgroup_fields(&self, sgpr: u32) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.workgroup_ids
            .iter()
            .enumerate()
            .filter_map(move |(axis, f)| f.filter(|f| f.register == sgpr).map(|f| (axis, f.shift)))
    }

    pub fn workitem_fields(&self, vgpr: u32) -> impl Iterator<Item = (usize, u32)> + '_ {
        self.workitem_ids
            .iter()
            .enumerate()
            .filter_map(move |(axis, f)| f.filter(|f| f.register == vgpr).map(|f| (axis, f.shift)))
    }

    pub fn workitem_register(&self, vgpr: u32) -> bool {
        self.workitem_fields(vgpr).next().is_some()
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
    let ids = [wg_id.0, wg_id.1, wg_id.2];
    for (axis, f) in layout.workgroup_ids.iter().enumerate() {
        if let Some(f) = f {
            s[f.register as usize] |= ids[axis] << f.shift;
        }
    }
    if let Some(p) = layout.workgroup_info {
        s[p as usize] = 0;
    }
    if let Some(p) = layout.private_segment_wave_offset {
        s[p as usize] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor() -> KernelDescriptor {
        crate::processor::decode_kernel_desc(&[0u8; 64])
    }

    #[test]
    fn separate_layouts_place_ids_after_the_user_sgprs() {
        let mut kd = descriptor();
        kd.enable_sgpr_private_segment_buffer = true;
        kd.enable_sgpr_kernarg_segment_ptr = true;
        kd.enable_sgpr_workgroup_id_x = true;
        kd.enable_sgpr_workgroup_id_z = true;
        kd.enable_sgpr_workgroup_info = true;
        kd.enable_sgpr_private_segment_wave_offset = true;
        kd.enable_vgpr_workitem_id = 2;
        kd.user_sgpr_count = 6;
        let layout = EntryLayout::separate(&kd).unwrap();
        assert_eq!(layout.private_segment_buffer, Some(0));
        assert_eq!(layout.kernarg_ptr, Some(4));
        assert_eq!(layout.workgroup_ids, [field(6, 0), None, field(7, 0)]);
        assert_eq!(layout.workgroup_info, Some(8));
        assert_eq!(layout.private_segment_wave_offset, Some(9));
        assert_eq!(layout.workitem_ids, [field(0, 0), field(1, 0), field(2, 0)]);
        assert_eq!(layout.workgroup_fields(7).collect::<Vec<_>>(), vec![(2, 0)]);
        assert_eq!(layout.workitem_fields(1).collect::<Vec<_>>(), vec![(1, 0)]);
        assert!(layout.workitem_register(2) && !layout.workitem_register(3));
        kd.enable_vgpr_workitem_id = 0;
        let narrow = EntryLayout::separate(&kd).unwrap();
        assert_eq!(narrow.workitem_ids, [field(0, 0), None, None]);
        kd.user_sgpr_count = 8;
        assert!(EntryLayout::separate(&kd).is_err(), "a descriptor whose user SGPR count disagrees with its enables");
    }

    #[test]
    fn architected_layouts_pack_the_ids() {
        let mut kd = descriptor();
        kd.enable_sgpr_kernarg_segment_ptr = true;
        kd.enable_sgpr_workgroup_id_x = true;
        kd.enable_sgpr_workgroup_id_y = true;
        kd.enable_sgpr_private_segment_wave_offset = true;
        let layout = EntryLayout::of(&kd);
        assert_eq!(layout.kernarg_ptr, Some(0));
        assert_eq!(layout.workgroup_ids, [field(WORKGROUP_ID_X, 0), field(WORKGROUP_ID_YZ, 0), field(WORKGROUP_ID_YZ, 16)]);
        assert_eq!(layout.private_segment_wave_offset, Some(2));
        assert_eq!(layout.workitem_ids, EntryLayout::PACKED);
        assert_eq!(layout.workitem_fields(0).collect::<Vec<_>>(), vec![(0, 0), (1, 10), (2, 20)]);
        assert_eq!(layout.workgroup_fields(WORKGROUP_ID_YZ).collect::<Vec<_>>(), vec![(1, 0), (2, 16)]);
    }

    #[test]
    fn setup_writes_every_workgroup_id_where_the_layout_puts_it() {
        let mut kd = descriptor();
        kd.enable_sgpr_kernarg_segment_ptr = true;
        kd.enable_sgpr_workgroup_id_x = true;
        kd.enable_sgpr_workgroup_id_y = true;
        kd.enable_sgpr_workgroup_id_z = true;
        kd.user_sgpr_count = 2;
        let mut s = [7u32; 128];
        setup_sgprs(&mut s, &EntryLayout::separate(&kd).unwrap(), 0x1234_5678_9abc, 0, 0, 0, (5, 6, 7));
        assert_eq!(&s[..6], &[0x5678_9abc, 0x1234, 5, 6, 7, 0]);
        let mut s = [7u32; 128];
        setup_sgprs(&mut s, &EntryLayout::of(&kd), 0, 0, 0, 0, (5, 6, 7));
        assert_eq!((s[WORKGROUP_ID_X as usize], s[WORKGROUP_ID_YZ as usize]), (5, 7 << 16 | 6));
    }
}
