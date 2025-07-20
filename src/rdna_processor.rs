use crate::processor::*;
use crate::rdna4_decoder::*;
use crate::instructions::*;

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
    scc: bool,
}

struct SIMD32 {
    ctx: Context,
    next_pc: usize,
    insts: Vec<u8>,
    pub sgprs: RegisterFileImpl<u32>,
    pub vgprs: RegisterFileImpl<u32>,
    num_sgprs: usize,
    num_vgprs: usize,
}

struct ComputeUnit {
    simds: Vec<SIMD32>,
}

impl ComputeUnit {
    pub fn new(pc: usize, insts: Vec<u8>, num_sgprs: usize, num_vgprs: usize) -> Self {
        let mut simds = vec![];
        for _ in 0..2 {
            let num_wave_slot = 16;
            let mut sgprs = RegisterFileImpl::new(num_wave_slot, 128, 0);
            for i in 0..num_wave_slot {
                sgprs.set(i, 126, 0xFFFFFFFF); // EXEC_LO
                sgprs.set(i, 127, 0xFFFFFFFF); // EXEC_HI
            }
            simds.push(
                SIMD32 {
                ctx: Context {
                    id: 0,
                    pc: pc,
                    scc: true,
                },
                next_pc: 0,
                insts: insts.clone(),
                sgprs: sgprs,
                vgprs: RegisterFileImpl::new(32, 1536 / 4, 0),
                num_sgprs: num_sgprs,
                num_vgprs: num_vgprs,
            });
        }

        ComputeUnit {
            simds: simds,
        }
    }
    
    pub fn dispatch(
        &mut self,
        entry_addr: usize,
        setup_data: Vec<([u32; 16], [[u32; 32]; 16])>,
        num_wavefronts: usize,
    ) {
        for simd_idx in 0..self.simds.len() {
            let setup_data = setup_data.iter().skip(simd_idx).step_by(2).cloned().collect();
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
    aql: hsa_kernel_dispatch_packet_s<'a>,
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

impl SIMD32 {
    pub fn dispatch(
        &mut self,
        entry_addr: usize,
        setup_data: Vec<([u32; 16], [[u32; 32]; 16])>
    ) {
        let num_wavefronts = setup_data.len();
        
        for i in 0..self.num_sgprs {
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
            let (sgprs, vgprs) = setup_data[wavefront];
            for i in 0..16 {
                self.sgprs.set(wavefront, i, sgprs[i]);
            }
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
                scc: false,
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
            println!("Unknown instruction 0x{:08X} at PC: 0x{:08X}", inst & 0xFFFFFFFF, self.ctx.pc);
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

    fn execute_inst(&mut self, inst: InstFormat) -> Signals {
        // println!("{:012X}: {:?}", self.ctx.pc, inst);
        match inst {
            InstFormat::SOP1(fields) => self.execute_sop1(fields),
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
    
    fn s_mov_b32(&mut self, d: usize, s0: usize) {
        let s0_value = self.read_sop_src(s0);
        let d_value = s0_value;
        self.write_sop_dst(d, d_value);
    }
}

impl<'a> RDNAProcessor<'a> {
    pub fn new(aql: &hsa_kernel_dispatch_packet_s<'a>, num_cunits: usize, wavefront_size: usize, mem: &Vec<u8>) -> Self {
        let insts = aql.kernel_object.object.to_vec();
        let kd = aql.kernel_object.offset;
        let kernel_desc = decode_kernel_desc(&insts[kd..(kd + 64)]);
        let aql_packet_address = (aql as *const hsa_kernel_dispatch_packet_s) as u64;
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
                    kernel_desc.granulated_wavefront_sgpr_count,
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
    ) -> ([u32; 16], [[u32; 32]; 16]) {
        let private_seg_ptr = if self.private_seg_buffer.len() > 0 {
            (&self.private_seg_buffer[0] as *const u8) as u64
        } else {
            0
        };

        let kernel_args_ptr = self.kernel_args_ptr;
        let aql_packet_address = self.aql_packet_address;
        let kernel_desc = &self.kernel_desc;
        let private_seg_size = self.aql.private_segment_size as u64;
        // initialize sgprs
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
            // println!(
            //     "s[{}..{}]: Private Segment Buffer",
            //     sgprs_pos,
            //     sgprs_pos + 3
            // );
            sgprs_pos += 4;
        }
        if kernel_desc.enable_sgpr_dispatch_ptr {
            sgprs[sgprs_pos] = (aql_packet_address & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((aql_packet_address >> 32) & 0xFFFFFFFF) as u32;
            // println!("s[{}..{}]: Dispatch Ptr", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_queue_ptr {
            // println!("s[{}..{}]: Queue Ptr", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_kernarg_segment_ptr {
            sgprs[sgprs_pos] = (kernel_args_ptr & 0xFFFFFFFF) as u32;
            sgprs[sgprs_pos + 1] = ((kernel_args_ptr >> 32) & 0xFFFFFFFF) as u32;
            // println!("s[{}..{}]: Kernarg Segment Ptr", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_dispatch_id {
            // println!("s[{}..{}]: Dispatch Id", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_flat_scratch_init {
            sgprs[sgprs_pos] = thread_id * self.aql.private_segment_size;
            sgprs[sgprs_pos + 1] = self.aql.private_segment_size;
            // println!("s[{}..{}]: Flat Scratch Init", sgprs_pos, sgprs_pos + 1);
            sgprs_pos += 2;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_x && sgprs_pos < 16 {
            // println!("s[{}]: Grid Work-Group Count X", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_y && sgprs_pos < 16 {
            // println!("s[{}]: Grid Work-Group Count Y", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_grid_workgroup_count_z && sgprs_pos < 16 {
            // println!("s[{}]: Grid Work-Group Count Z", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_x {
            sgprs[sgprs_pos] = workgroup_id_x;
            // println!("s[{}]: Work-Group Id X", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_y {
            sgprs[sgprs_pos] = workgroup_id_y;
            // println!("s[{}]: Work-Group Id Y", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_id_z {
            sgprs[sgprs_pos] = workgroup_id_z;
            // println!("s[{}]: Work-Group Id Z", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_workgroup_info {
            sgprs[sgprs_pos] = 0;
            // println!("s[{}]: Work-Group Info", sgprs_pos);
            sgprs_pos += 1;
        }
        if kernel_desc.enable_sgpr_private_segment_wave_offset {
            sgprs[sgprs_pos] = 0;
            // println!("s[{}]: Scratch Wave Offset", sgprs_pos);
            sgprs_pos += 1;
        }

        // initialize vgprs
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
            vgprs_pos += 1;
        }

        // initialize pc
        (sgprs, vgprs)
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
            // let mut thread_handles = vec![];
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
                            v.dispatch(entry_address, setup_data, workgroup_size / 64);
                        }
                    });
                    handle.join();
                    // thread_handles.push(handle);
                }
            }

            // for t in thread_handles {
            //     t.join();
            //     bar.inc(1);
            // }
        }

        bar.finish();
    }
}
