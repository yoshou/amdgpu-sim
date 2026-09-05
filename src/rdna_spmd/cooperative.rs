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
//! Signal and wait are separate typed effects. Barrier rounds count waves;
//! signal-is-first reports the first wave in that round. The driver validates
//! uniform yield PCs within each wave and diagnoses barrier deadlocks.

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
    let storage_size=wg_size.div_ceil(32)*32;
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
                let mut sgprs: Vec<[u32; COOP_SGPR_BUF]> = vec![[0u32; COOP_SGPR_BUF]; storage_size];
                // Dedicated per-work-item lane-spill buffer (NOT architectural
                // registers) for the uniform writelane/readlane idiom; must
                // persist across barrier yields.
                let mut spill: Vec<[u32; COOP_SPILL_SLOTS]> = vec![[0u32; COOP_SPILL_SLOTS]; storage_size];
                let mut vgprs: Vec<Vec<u32>> = vec![vec![0u32; num_vgprs]; storage_size];
                let scratch: Vec<Vec<u64>> = vec![vec![0u64; scratch_u64]; storage_size];
                let mut resume: Vec<u64> = vec![0; storage_size];
                let mut done: Vec<bool> = vec![false; storage_size];
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

                    let waves=wg_size.div_ceil(32);
                    let mut barriers=super::barrier::Barriers::new(waves);
                    let mut waiting=vec![None;waves];
                    let mut passes=0;
                    loop {
                        let mut progress=false;
                        let mut live=false;
                        for wave in 0..waves {
                            let range=wave*32..((wave+1)*32).min(wg_size);
                            if range.clone().all(|wi|done[wi]){continue;}
                            live=true;
                            if let Some(id)=waiting[wave]{
                                if !barriers.wait(wave,id){continue;}
                                waiting[wave]=None;
                            }
                            progress=true;
                            let mut boundary=None;
                            for wi in range.clone(){
                                assert!(!done[wi],"nonuniform wave termination at barrier");
                                let r=unsafe{kernel.run(sgprs[wi].as_mut_ptr(),vgprs[wi].as_mut_ptr(),scratch[wi].as_ptr()as u64,lds_base,spill[wi].as_mut_ptr(),resume[wi])};
                                if let Some(pc)=boundary{assert_eq!(pc,r,"nonuniform wave yield");}else{boundary=Some(r);}
                                done[wi]=r==COOP_DONE;resume[wi]=r;
                            }
                            let pc=boundary.unwrap();if pc==COOP_DONE{continue;}
                            let action=kernel.yields.get(&(pc as usize)).expect("yield lacks typed synchronization effect");
                            if action.is_wave(){
                                let count=range.len();let valid=if count==32{u32::MAX}else{(1u32<<count)-1};
                                super::coop_xlane::apply_xlane(action,valid,&mut sgprs[wave*32..(wave+1)*32],&mut vgprs[wave*32..(wave+1)*32]);
                                continue;
                            }
                            let read=|wi:usize,s:&crate::rdna_instructions::SourceOperand|match s{
                                crate::rdna_instructions::SourceOperand::ScalarRegister(r)=>sgprs[wi][*r as usize],
                                crate::rdna_instructions::SourceOperand::LiteralConstant(v)=>*v,
                                crate::rdna_instructions::SourceOperand::IntegerConstant(v)=>*v as u32,
                                _=>panic!("barrier ID must be uniform scalar"),
                            };
                            let raw_id=action.inputs[0].eval(range.start,&read,&|wi|sgprs[wi][EXEC]&1!=0);
                            let id=raw_id & 31;
                            for wi in range.clone(){assert_eq!(raw_id,action.inputs[0].eval(wi,&read,&|wi|sgprs[wi][EXEC]&1!=0),"nonuniform barrier ID");}
                            use super::ir::typed::effect::EffectOp;
                            match action.op{
                                EffectOp::BarrierSignal{is_first}=>{
                                    let first=barriers.signal(wave,id);
                                    if is_first{for wi in range{sgprs[wi][SCC]=first as u32;}}
                                },
                                EffectOp::BarrierWait=>{waiting[wave]=Some(id);},
                                _=>panic!("wave operation must be scheduled by the wave driver"),
                            }
                        }
                        if !live{break;}
                        assert!(progress,"workgroup barrier deadlock");
                        passes+=1;assert!(passes<100_000,"cooperative dispatch did not converge");
                    }

                    wg += num_threads as u64;
                }
            });
        }
    });
}

/// The existing packet fiber ABI extended with workgroup LDS and typed barrier
/// rounds. Every worker owns complete workgroups; only yield operands/results
/// cross the packet/host boundary.
pub fn dispatch_cooperative_vec(
    kernel: &super::emit_vec::CoopVecKernel,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    group_segment_size: usize,
    num_threads: usize,
) {
    use super::fiber::{Fiber, KernelArgs, FIBER_DONE};
    use super::ir::typed::effect::EffectOp;
    use crate::rdna_instructions::SourceOperand;
    let width = kernel.width as usize;
    let ppw = 32 / width;
    let wg_size = dims.workgroup_size() as usize;
    let waves = wg_size.div_ceil(32);
    let packets = waves * ppw;
    let num_wg = dims.num_wg_x as u64 * dims.num_wg_y as u64 * dims.num_wg_z as u64;
    let threads = num_threads.max(1);
    let stride = ((private_segment_size as usize + 15) & !15)
        .max(16)
        .max(kernel.min_private_bytes.div_ceil(16) * 16);
    thread::scope(|scope| {
        for tid in 0..threads {
            scope.spawn(move || {
                let mut sgprs = vec![[0u32; COOP_SGPR_BUF]; packets];
                let mut vgprs = vec![vec![0u32; kernel.num_vgprs * width]; packets];
                let mut spill = vec![vec![0u32; COOP_SPILL_SLOTS]; packets];
                let mut fibers: Vec<_> = (0..packets).map(|_| Fiber::new(32 << 10)).collect();
                let mut scratch =
                    aligned_vec::AVec::<u8, aligned_vec::ConstAlign<0x1_0000_0000>>::new(
                        0x1_0000_0000,
                    );
                scratch.resize(waves * 32 * stride, 0);
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
                        let sb = unsafe { scratch.as_ptr().add(wave * 32 * stride) } as u64;
                        sgprs[p][..128].copy_from_slice(&setup_sgprs(
                            kd,
                            kernarg_ptr,
                            aql_packet_addr,
                            sb,
                            private_segment_size,
                            wg_id,
                        ));
                        sgprs[p][EXEC] = ((1u64 << valid) - 1) as u32;
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
                                .get(&(pc as usize))
                                .expect("packet yield lacks typed effect");
                            if action.is_wave() {
                                let count = wg_size.saturating_sub(wave * 32).min(32);
                                let valid = if count == 32 {
                                    u32::MAX
                                } else {
                                    (1u32 << count) - 1
                                };
                                super::coop_xlane::apply_xlane_packets(
                                    action,
                                    width,
                                    valid,
                                    &mut sgprs[range.clone()],
                                    &mut vgprs[range],
                                );
                            } else {
                                let read = |lane: usize, s: &SourceOperand| match s {
                                    SourceOperand::ScalarRegister(r) => {
                                        sgprs[range.start + lane / width][*r as usize]
                                    }
                                    SourceOperand::IntegerConstant(v) => *v as u32,
                                    SourceOperand::LiteralConstant(v) => *v,
                                    _ => panic!("barrier ID must be scalar"),
                                };
                                let raw_id = action.inputs[0].eval(0, &read, &|_| true);
                                let id = raw_id & 31;
                                for p in range.clone() {
                                    if p * width < wg_size {
                                        assert_eq!(
                                            raw_id,
                                            action.inputs[0].eval(
                                                (p - range.start) * width,
                                                &read,
                                                &|_| true
                                            ),
                                            "nonuniform barrier ID"
                                        );
                                    }
                                }
                                match action.op {
                                    EffectOp::BarrierSignal { is_first } => {
                                        let first = barriers.signal(wave, id);
                                        if is_first {
                                            for p in range {
                                                sgprs[p][SCC] = first as u32;
                                            }
                                        }
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
