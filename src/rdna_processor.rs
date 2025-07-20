use crate::instructions::*;
use crate::processor::*;
use crate::rdna4_decoder::*;

pub trait RegisterFile<T: Copy> {
    fn new(num_elems: usize, count: usize, default: T) -> Self;
    fn get(&self, elem: usize, idx: usize) -> T;

    fn set(&mut self, elem: usize, idx: usize, val: T);

    fn get_vec(&self, idx: usize) -> &[T];

    fn set_vec(&mut self, idx: usize, vals: &[Option<T>]);
}

pub struct RegisterFileImpl<T: Copy> {
    num_elems: usize,
    regs: Vec<T>,
}

impl<T: Copy> RegisterFile<T> for RegisterFileImpl<T> {
    fn new(num_elems: usize, count: usize, default: T) -> Self {
        RegisterFileImpl {
            num_elems: num_elems,
            regs: vec![default; num_elems * count],
        }
    }

    fn get(&self, elem: usize, idx: usize) -> T {
        self.regs[self.num_elems * idx + elem]
    }

    fn set(&mut self, elem: usize, idx: usize, val: T) {
        self.regs[self.num_elems * idx + elem] = val
    }

    fn get_vec(&self, idx: usize) -> &[T] {
        let beg = self.num_elems * idx;
        let end = self.num_elems * (idx + 1);
        &self.regs.as_slice()[beg..end]
    }

    fn set_vec(&mut self, idx: usize, vals: &[Option<T>]) {
        for elem in 0..self.num_elems {
            if let Some(val) = vals[elem] {
                self.set(elem, idx, val);
            }
        }
    }
}

pub enum Signals {
    None,
    EndOfProgram,
    Switch,
    Unknown,
}

pub trait Processor {
    fn step(&mut self) -> Signals;
}

#[derive(Copy, Clone, Debug)]
struct Context {
    id: usize,
    pc: usize,
}

struct SIMD32 {
    ctx: Context,
    next_pc: usize,
    insts: Vec<u8>,
    pub sgprs: RegisterFileImpl<u32>,
    pub vgprs: RegisterFileImpl<u32>,
    num_vgprs: usize,
}

struct ComputeUnit {
    simds: Vec<SIMD32>,
}

impl ComputeUnit {
    pub fn new(pc: usize, insts: Vec<u8>, num_vgprs: usize) -> Self {
        let mut simds = vec![];
        for _ in 0..2 {
            let num_wave_slot = 16;
            let mut sgprs = RegisterFileImpl::new(num_wave_slot, 128, 0);
            for i in 0..num_wave_slot {
                sgprs.set(i, 126, 0xFFFFFFFF); // EXEC_LO
                sgprs.set(i, 127, 0xFFFFFFFF); // EXEC_HI
            }
            simds.push(SIMD32 {
                ctx: Context { id: 0, pc: pc },
                next_pc: 0,
                insts: insts.clone(),
                sgprs: sgprs,
                vgprs: RegisterFileImpl::new(32, 1536 / 4, 0),
                num_vgprs: num_vgprs,
            });
        }

        ComputeUnit { simds: simds }
    }

    pub fn dispatch(&mut self, entry_addr: usize, setup_data: Vec<RegisterSetupData>) {
        for simd_idx in 0..self.simds.len() {
            let setup_data = setup_data
                .iter()
                .skip(simd_idx)
                .step_by(self.simds.len())
                .cloned()
                .collect();
            self.simds[simd_idx].dispatch(entry_addr, setup_data);
        }
    }
}

struct WorkgroupProcessor {
    cunits: Vec<Arc<Mutex<ComputeUnit>>>,
}

use std::sync::{Arc, Mutex};

pub struct RDNAProcessor<'a> {
    wgps: Vec<Arc<Mutex<WorkgroupProcessor>>>,
    entry_address: usize,
    kernel_desc: KernelDescriptor,
    aql_packet_address: u64,
    kernel_args_ptr: u64,
    aql: HsaKernelDispatchPacket<'a>,
    private_seg_buffer: Vec<u8>,
}

impl Processor for ComputeUnit {
    fn step(&mut self) -> Signals {
        for simd in &mut self.simds {
            simd.step();
        }
        Signals::None
    }
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

#[inline(always)]
fn u64_from_u32_u32(lo: u32, hi: u32) -> u64 {
    ((hi as u64) << 32) | (lo as u64)
}

impl SIMD32 {
    pub fn dispatch(&mut self, entry_addr: usize, setup_data: Vec<RegisterSetupData>) {
        let num_wavefronts = setup_data.len();

        for i in 0..128 {
            for slot in 0..16 {
                self.sgprs.set(slot, i, 0);
            }
        }
        for i in 0..self.num_vgprs {
            for elem in 0..32 {
                self.vgprs.set(elem, i, 0);
            }
        }
        for wavefront in 0..num_wavefronts {
            let vgpr_offset = wavefront * self.num_vgprs;
            let sgprs = setup_data[wavefront].sgprs;
            let vgprs = setup_data[wavefront].vgprs;
            let user_sgpr_count = setup_data[wavefront].user_sgpr_count;
            for i in 0..user_sgpr_count {
                self.sgprs.set(wavefront, i, sgprs[i]);
            }
            self.sgprs.set(wavefront, 117, sgprs[user_sgpr_count]); // TTMP9
            self.sgprs.set(
                wavefront,
                115,
                sgprs[user_sgpr_count + 1] << 16 | sgprs[user_sgpr_count + 2],
            ); // TTMP7
            for i in 0..16 {
                for elem in 0..32 {
                    self.vgprs.set(elem, vgpr_offset + i, vgprs[i][elem]);
                }
            }
        }

        use std::collections::VecDeque;

        let mut ctxs = VecDeque::new();
        for wavefront in 0..num_wavefronts {
            ctxs.push_back(Context {
                id: wavefront,
                pc: entry_addr,
            })
        }

        let is_signal_none = |signal: &Signals| match signal {
            Signals::None => true,
            _ => false,
        };

        while !ctxs.is_empty() {
            if let Some(ctx) = ctxs.pop_front() {
                self.ctx = ctx;
            }
            let mut signal = self.step();
            while is_signal_none(&signal) {
                signal = self.step();
            }

            match signal {
                Signals::EndOfProgram => {}
                Signals::Switch => ctxs.push_back(self.ctx),
                _ => panic!(),
            }
        }
    }

    fn step(&mut self) -> Signals {
        let inst = self.fetch_inst();
        println!("Fetched instruction: 0x{:08X}", inst & 0xFFFFFFFF);
        if let Ok((inst, size)) = decode_rdna4(inst) {
            println!("Executing instruction: {:?}", inst);
            self.next_pc = self.get_pc() as usize + size;
            let result = self.execute_inst(inst);
            self.set_pc(self.next_pc as u64);
            result
        } else {
            println!(
                "Unknown instruction 0x{:08X} at PC: 0x{:08X}",
                inst & 0xFFFFFFFF,
                self.ctx.pc
            );
            Signals::Unknown
        }
    }

    fn get_pc(&self) -> u64 {
        (&self.insts[self.ctx.pc] as *const u8) as u64
    }

    fn set_pc(&mut self, value: u64) {
        let base_ptr = (&self.insts[0] as *const u8) as u64;
        self.ctx.pc = (value - base_ptr) as usize;
    }

    fn read_sgpr(&self, idx: usize) -> u32 {
        self.sgprs.get(self.ctx.id, idx)
    }

    fn write_sgpr(&mut self, idx: usize, value: u32) {
        self.sgprs.set(self.ctx.id, idx, value);
    }

    fn read_vgpr(&self, elem: usize, idx: usize) -> u32 {
        if idx >= self.num_vgprs {
            panic!();
        }
        self.vgprs.get(elem, self.num_vgprs * self.ctx.id + idx)
    }

    fn write_vgpr(&mut self, elem: usize, idx: usize, value: u32) {
        if idx >= self.num_vgprs {
            panic!();
        }
        self.vgprs
            .set(elem, self.num_vgprs * self.ctx.id + idx, value);
    }

    fn execute_inst(&mut self, inst: InstFormat) -> Signals {
        // println!("{:012X}: {:?}", self.ctx.pc, inst);
        match inst {
            InstFormat::SOP1(fields) => self.execute_sop1(fields),
            InstFormat::VOP3(fields) => self.execute_vop3(fields),
            _ => unimplemented!(),
        }
    }

    fn fetch_inst(&mut self) -> u64 {
        get_u64(&self.insts, self.ctx.pc)
    }

    fn execute_sop1(&mut self, inst: SOP1) -> Signals {
        let d = inst.SDST as usize;
        let s0 = inst.SSRC0 as usize;

        match inst.OP {
            I::S_MOV_B32 => {
                self.s_mov_b32(d, s0);
            }
            I::S_MOV_B64 => {
                self.s_mov_b64(d, s0);
            }
            _ => unimplemented!(),
        }

        Signals::None
    }

    fn execute_vop3(&mut self, inst: VOP3) -> Signals {
        let d = inst.VDST as usize;
        let s0 = inst.SRC0 as usize;
        let s1 = inst.SRC1 as usize;
        match inst.OP {
            I::V_READLANE_B32 => {
                self.v_readlane_b32(d, s0, s1);
            }
            I::V_WRITELANE_B32 => {
                self.v_writelane_b32(d, s0, s1);
            }
            _ => unimplemented!(),
        }
        Signals::None
    }

    fn read_sop_src(&self, addr: usize) -> u32 {
        match addr {
            0..=105 => self.read_sgpr(addr),
            106 => self.read_sgpr(addr),
            107 => self.read_sgpr(addr),
            108..=123 => self.read_sgpr(addr),
            126 => self.read_sgpr(addr), // EXEC_LO
            127 => self.read_sgpr(addr), // EXEC_HI
            128 => 0,
            129..=192 => (addr - 128) as u32,
            193..=208 => (-((addr - 192) as i32)) as u32,
            240 => 0x3f000000, // 0.5
            241 => 0xbf000000, // -0.5
            242 => 0x3f800000, // 1.0
            243 => 0xbf800000, // -1.0
            244 => 0x40000000, // 2.0
            245 => 0xc0000000, // -2.0
            246 => 0x40800000, // 4.0
            247 => 0xc0800000, // -4.0
            248 => 0x3e22f983, // 1/(2*PI)
            // 255 => self.fetch_literal_constant(),
            _ => panic!(),
        }
    }

    fn read_sop_src_pair(&self, addr: usize) -> u64 {
        match addr {
            0..=105 => u64_from_u32_u32(self.read_sgpr(addr), self.read_sgpr(addr + 1)),
            // 102 => u64_from_u32_u32(self.get_flat_scratch_lo(), self.get_flat_scratch_hi()),
            // 106 => u64_from_u32_u32(self.get_vcc_lo(), self.get_vcc_hi()),
            // 126 => u64_from_u32_u32(self.ctx.exec_lo, self.ctx.exec_hi),
            128 => 0,
            129..=192 => (addr - 128) as u64,
            193..=208 => (-((addr - 192) as i64)) as u64,
            240 => 0x3fe0000000000000, // 0.5
            241 => 0xbfe0000000000000, // -0.5
            242 => 0x3ff0000000000000, // 1.0
            243 => 0xbff0000000000000, // -1.0
            244 => 0x4000000000000000, // 2.0
            245 => 0xc000000000000000, // -2.0
            246 => 0x4010000000000000, // 4.0
            247 => 0xc010000000000000, // -4.0
            248 => 0x3fc45f306dc8bdc4, // 1/(2*PI)
            // 255 => self.fetch_literal_constant() as u64,
            _ => panic!(),
        }
    }

    fn write_sop_dst(&mut self, addr: usize, value: u32) {
        match addr {
            0..=105 => self.write_sgpr(addr, value),
            106 => self.write_sgpr(addr, value),
            107 => self.write_sgpr(addr, value),
            108..=123 => self.write_sgpr(addr, value),
            126 => self.write_sgpr(addr, value), // EXEC_LO
            127 => self.write_sgpr(addr, value), // EXEC_HI
            _ => panic!(),
        }
    }

    fn write_sop_dst_pair(&mut self, addr: usize, value: u64) {
        self.write_sop_dst(addr, (value & 0xFFFFFFFF) as u32);
        self.write_sop_dst(addr + 1, ((value >> 32) & 0xFFFFFFFF) as u32);
    }

    fn read_vop_src(&self, elem: usize, addr: usize) -> u32 {
        match addr {
            0..=101 => self.read_sgpr(addr),
            128 => 0,
            129..=192 => (addr - 128) as u32,
            193..=208 => -((addr - 192) as i32) as u32,
            240 => 0x3f000000, // 0.5
            241 => 0xbf000000, // -0.5
            242 => 0x3f800000, // 1.0
            243 => 0xbf800000, // -1.0
            244 => 0x40000000, // 2.0
            245 => 0xc0000000, // -2.0
            246 => 0x40800000, // 4.0
            247 => 0xc0800000, // -4.0
            248 => 0x3e22f983, // 1/(2*PI)
            // 255 => self.fetch_literal_constant(),
            256..=511 => self.read_vgpr(elem, addr - 256),
            _ => panic!(),
        }
    }

    fn s_mov_b32(&mut self, d: usize, s0: usize) {
        let s0_value = self.read_sop_src(s0);
        let d_value = s0_value;
        self.write_sop_dst(d, d_value);
    }

    fn s_mov_b64(&mut self, d: usize, s0: usize) {
        let s0_value = self.read_sop_src_pair(s0);
        let d_value = s0_value;
        self.write_sop_dst_pair(d, d_value);
    }

    fn v_readlane_b32(&mut self, d: usize, s0: usize, s1: usize) {
        let s1_value = self.read_sop_src(s1) as usize;
        let s0_value = self.read_vop_src(s1_value, s0);
        let d_value = s0_value;
        self.write_sgpr(d, d_value);
    }

    fn v_writelane_b32(&mut self, d: usize, s0: usize, s1: usize) {
        let s0_value = self.read_sop_src(s0);
        let s1_value = self.read_sop_src(s1) as usize;
        let d_value = s0_value;
        self.write_vgpr(s1_value, d, d_value);
    }
}

#[derive(Debug, Clone)]
struct RegisterSetupData {
    user_sgpr_count: usize,
    sgprs: [u32; 16],
    vgprs: [[u32; 32]; 16],
}

fn get_bit(buffer: &[u8], offset: usize, bit: usize) -> bool {
    ((buffer[offset + (bit >> 3)] >> (bit & 0x7)) & 1) == 1
}

fn get_bits(buffer: &[u8], offset: usize, bit: usize, size: usize) -> u8 {
    ((get_u32(buffer, offset + (bit >> 3)) >> (bit & 0x7)) & ((1 << size) - 1)) as u8
}

fn get_u32(buffer: &[u8], offset: usize) -> u32 {
    let b0 = buffer[offset] as u32;
    let b1 = buffer[offset + 1] as u32;
    let b2 = buffer[offset + 2] as u32;
    let b3 = buffer[offset + 3] as u32;

    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
}

fn decode_kernel_desc(kd: &[u8]) -> KernelDescriptor {
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
        granulated_workitem_vgpr_count: (get_bits(kd, 48, 0, 6) as usize + 1) * 8,
        granulated_wavefront_sgpr_count: 0,
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

impl<'a> RDNAProcessor<'a> {
    pub fn new(
        aql: &HsaKernelDispatchPacket<'a>,
        num_cunits: usize,
        wavefront_size: usize,
        mem: &Vec<u8>,
    ) -> Self {
        let insts = aql.kernel_object.object.to_vec();
        let kd = aql.kernel_object.offset;
        let kernel_desc = decode_kernel_desc(&insts[kd..(kd + 64)]);
        let aql_packet_address = (aql as *const HsaKernelDispatchPacket) as u64;
        let num_wgps = num_cunits / 2;

        assert!(num_cunits % 2 == 0, "Number of compute units must be even.");
        assert!(wavefront_size == 32, "Wavefront size must be 32.");

        let mut wgps = vec![];
        for _ in 0..num_wgps {
            let mut cunits_in_wgp = vec![];
            for _ in 0..2 {
                let cu = Arc::new(Mutex::new(ComputeUnit::new(
                    kd + kernel_desc.kernel_code_entry_byte_offset,
                    mem.clone(),
                    kernel_desc.granulated_workitem_vgpr_count,
                )));
                cunits_in_wgp.push(cu);
            }
            let wgp = Arc::new(Mutex::new(WorkgroupProcessor {
                cunits: cunits_in_wgp,
            }));

            wgps.push(wgp);
        }

        let kernel_args_ptr = aql.kernarg_address.address();
        let entry_address = kd + kernel_desc.kernel_code_entry_byte_offset;

        let private_segment_size = aql.private_segment_size as usize;
        let private_seg_buffer: Vec<u8> = vec![0u8; private_segment_size * 256 * num_cunits];

        // create instance
        RDNAProcessor {
            wgps: wgps,
            kernel_desc: kernel_desc,
            kernel_args_ptr: kernel_args_ptr,
            aql_packet_address: aql_packet_address,
            entry_address: entry_address,
            aql: *aql,
            private_seg_buffer: private_seg_buffer,
        }
    }

    fn dispatch(
        &self,
        thread_id: u32,
        workgroup_id_x: u32,
        workgroup_id_y: u32,
        workgroup_id_z: u32,
        workitem_offset: usize,
    ) -> RegisterSetupData {
        let private_seg_ptr = if self.private_seg_buffer.len() > 0 {
            (&self.private_seg_buffer[0] as *const u8) as u64
        } else {
            0
        };

        let kernel_args_ptr = self.kernel_args_ptr;
        let aql_packet_address = self.aql_packet_address;
        let kernel_desc = &self.kernel_desc;
        let private_seg_size = self.aql.private_segment_size as u64;

        // Initialize SGPRS
        let mut sgprs = [0u32; 16];
        let mut sgprs_pos = 0;
        if kernel_desc.enable_sgpr_private_segment_buffer {
            let mut desc_w0 = 0;
            desc_w0 |=
                (private_seg_ptr + (thread_id as u64) * private_seg_size * 256) & ((1 << 48) - 1);
            desc_w0 |= (private_seg_size & ((1 << 14) - 1)) << 48;
            for i in 0..2 {
                sgprs[sgprs_pos + i] = ((desc_w0 >> (i * 32)) & 0xFFFFFFFF) as u32;
            }
            sgprs_pos += 4;
        }
        if kernel_desc.enable_sgpr_dispatch_ptr {
            sgprs[sgprs_pos] = (aql_packet_address & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((aql_packet_address >> 32) & 0xFFFFFFFF) as u32;
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_queue_ptr {
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_kernarg_segment_ptr {
            sgprs[sgprs_pos] = (kernel_args_ptr & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((kernel_args_ptr >> 32) & 0xFFFFFFFF) as u32;
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_dispatch_id {
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_flat_scratch_init {
            sgprs[sgprs_pos] = thread_id * self.aql.private_segment_size;
            sgprs[sgprs_pos + 1] = self.aql.private_segment_size;
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_x && sgprs_pos < 16 {
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_y && sgprs_pos < 16 {
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_z && sgprs_pos < 16 {
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_x {
            sgprs[sgprs_pos] = workgroup_id_x;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_y {
            sgprs[sgprs_pos] = workgroup_id_y;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_z {
            sgprs[sgprs_pos] = workgroup_id_z;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_info {
            sgprs[sgprs_pos] = 0;
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_private_segment_wave_offset {
            sgprs[sgprs_pos] = 0;
        }

        // Initialize VGPRS
        let mut vgprs = [[0u32; 32]; 16];
        let mut vgprs_pos = 0;
        for i in 0..32 {
            let id_x = (i + workitem_offset) % self.aql.workgroup_size_x as usize;
            vgprs[vgprs_pos][i] = id_x as u32;
        }
        vgprs_pos += 1;
        if kernel_desc.enable_vgpr_workitem_id > 0 {
            for i in 0..32 {
                let id_y = ((i + workitem_offset) / self.aql.workgroup_size_x as usize)
                    % self.aql.workgroup_size_y as usize;
                vgprs[vgprs_pos][i] = id_y as u32;
            }
            vgprs_pos += 1;
        }
        if kernel_desc.enable_vgpr_workitem_id > 1 {
            for i in 0..32 {
                let id_z = ((i + workitem_offset)
                    / (self.aql.workgroup_size_x * self.aql.workgroup_size_y) as usize)
                    % self.aql.workgroup_size_z as usize;
                vgprs[vgprs_pos][i] = id_z as u32;
            }
        }

        RegisterSetupData {
            user_sgpr_count: kernel_desc.user_sgpr_count,
            sgprs: sgprs,
            vgprs: vgprs,
        }
    }

    pub fn execute(&mut self) {
        let workgroup_size_x = self.aql.workgroup_size_x as u32;
        let workgroup_size_y = self.aql.workgroup_size_y as u32;
        let workgroup_size_z = self.aql.workgroup_size_z as u32;

        let workgroup_size = (workgroup_size_x * workgroup_size_y * workgroup_size_z) as usize;

        let num_workgroup_x = (self.aql.grid_size_x + workgroup_size_x - 1) / workgroup_size_x;
        let num_workgroup_y = (self.aql.grid_size_y + workgroup_size_y - 1) / workgroup_size_y;
        let num_workgroup_z = (self.aql.grid_size_z + workgroup_size_z - 1) / workgroup_size_z;

        let num_workgroups = num_workgroup_x * num_workgroup_y * num_workgroup_z;

        use indicatif::ProgressBar;
        let bar = ProgressBar::new(num_workgroups as u64);

        let num_wgps = self.wgps.len();

        for workgroup_id_base in (0..num_workgroups).step_by(num_wgps) {
            for wgp_idx in 0..num_wgps {
                let workgroup_id = workgroup_id_base + wgp_idx as u32;
                let workgroup_id_x = workgroup_id % num_workgroup_x;
                let workgroup_id_y = (workgroup_id / num_workgroup_x) % num_workgroup_y;
                let workgroup_id_z =
                    (workgroup_id / (num_workgroup_x * num_workgroup_y)) % num_workgroup_z;

                let mut setup_data = vec![];
                for workitem_id in (0..workgroup_size).step_by(32) {
                    setup_data.push(self.dispatch(
                        wgp_idx as u32,
                        workgroup_id_x,
                        workgroup_id_y,
                        workgroup_id_z,
                        workitem_id,
                    ));
                }

                let entry_address = self.entry_address;

                for cu_idx in 0..2 {
                    let cu = Arc::clone(&self.wgps[wgp_idx].lock().unwrap().cunits[cu_idx]);
                    let setup_data = setup_data.iter().skip(cu_idx).step_by(2).cloned().collect();

                    use std::thread;

                    let handle = thread::spawn(move || {
                        if let Ok(mut v) = cu.lock() {
                            v.dispatch(entry_address, setup_data);
                        }
                    });
                    handle.join().unwrap();
                }
            }
        }

        bar.finish();
    }
}
