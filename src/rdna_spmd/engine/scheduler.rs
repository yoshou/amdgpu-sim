use std::thread;

use crate::processor::KernelDescriptor;

use super::dispatch::{setup_sgprs, GridDims};
use super::fiber::{Fiber, KernelArgs, FIBER_DONE};
use super::kernel::{Code, CoopVecKernel, Kernel, ScalarKernel, Scheduler, VecKernel, COOP_SGPR_BUF, COOP_SPILL_SLOTS};
use super::yields::YieldValues;
use super::super::ir::EffectOp;

const WAVE: usize = 32;
const FIBER_STACK_BYTES: usize = 32 << 10;
const LDS_MIN_BYTES: usize = 128 * 1024;
const POOL_LIMIT: usize = 64;

#[derive(Clone, Copy)]
enum Invoke<'a> {
    Scalar(&'a ScalarKernel),
    Packet(&'a VecKernel),
    Fiber(&'a CoopVecKernel),
}

#[derive(Clone, Copy)]
pub(crate) struct View<'a> {
    invoke: Invoke<'a>,
    scheduler: Scheduler,
    width: usize,
    num_vgprs: usize,
    min_private_bytes: usize,
    workgroup_x: Option<u32>,
    yields: &'a [YieldValues],
    exec: usize,
}

impl<'a> View<'a> {
    pub(crate) fn of(kernel: &'a Kernel) -> Self {
        match &kernel.code {
            Code::Scalar(k) => Self { invoke: Invoke::Scalar(k), scheduler: kernel.scheduler(), width: 1, num_vgprs: k.num_vgprs, min_private_bytes: 0, workgroup_x: None, yields: &[], exec: 0 },
            Code::Packet(k) => Self { invoke: Invoke::Packet(k), scheduler: kernel.scheduler(), width: k.width as usize, num_vgprs: k.num_vgprs, min_private_bytes: k.min_private_bytes, workgroup_x: k.workgroup_x, yields: &[], exec: 0 },
            Code::Cooperative(k) => Self::cooperative(k, kernel.scheduler()),
        }
    }
    pub(crate) fn cooperative(k: &'a CoopVecKernel, scheduler: Scheduler) -> Self {
        Self { invoke: Invoke::Fiber(k), scheduler, width: k.width as usize, num_vgprs: k.num_vgprs, min_private_bytes: k.min_private_bytes, workgroup_x: k.workgroup_x, yields: &k.yields, exec: k.registers.exec as usize }
    }
    fn fibers(&self) -> bool { matches!(self.invoke, Invoke::Fiber(_)) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit { Packet, Wave, Workgroup }

struct State {
    sgprs: Vec<[u32; COOP_SGPR_BUF]>,
    vgprs: Vec<Vec<u32>>,
    spill: Vec<Vec<u32>>,
    fibers: Vec<Fiber>,
    resume: Vec<u64>,
    done: Vec<bool>,
    waiting: Vec<Option<u32>>,
    valid: Vec<u32>,
    scratch: aligned_vec::AVec<u8, aligned_vec::ConstAlign<0x1_0000_0000>>,
    lds: Vec<u8>,
}

unsafe impl Send for State {}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape { packets: usize, waves: usize, words: usize, fibers: bool, scratch: usize, lds: usize }

static POOL: std::sync::Mutex<Vec<(Shape, State)>> = std::sync::Mutex::new(Vec::new());

fn acquire(shape: Shape) -> State {
    let mut pool = POOL.lock().unwrap();
    if let Some(index) = pool.iter().position(|(s, _)| *s == shape) {
        return pool.swap_remove(index).1;
    }
    drop(pool);
    let mut scratch = aligned_vec::AVec::new(0x1_0000_0000);
    scratch.resize(shape.scratch, 0u8);
    State {
        sgprs: vec![[0u32; COOP_SGPR_BUF]; shape.packets],
        vgprs: (0..shape.packets).map(|_| vec![0u32; shape.words]).collect(),
        spill: if shape.fibers { (0..shape.packets).map(|_| vec![0u32; COOP_SPILL_SLOTS]).collect() } else { Vec::new() },
        fibers: if shape.fibers { Fiber::batch(shape.packets, FIBER_STACK_BYTES) } else { Vec::new() },
        resume: vec![0; shape.packets],
        done: vec![true; shape.packets],
        waiting: vec![None; shape.waves],
        valid: vec![0; shape.waves],
        scratch,
        lds: vec![0u8; shape.lds],
    }
}

fn release(shape: Shape, state: State) {
    let mut pool = POOL.lock().unwrap();
    if pool.len() < POOL_LIMIT { pool.push((shape, state)); }
}

struct Engine<'a> {
    view: View<'a>,
    kd: &'a KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    unit: Unit,
    wg_size: usize,
    packets_per_wave: usize,
    waves_per_wg: usize,
    packets_per_wg: usize,
    units: u64,
    stride: usize,
    shape: Shape,
}

pub(crate) fn run(
    view: View,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    group_segment_size: usize,
    num_threads: usize,
) {
    let width = view.width;
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16));
    let wg_size = dims.workgroup_size() as usize;
    let unit = match view.scheduler {
        Scheduler::Independent => Unit::Packet,
        Scheduler::Wave => Unit::Wave,
        Scheduler::Workgroup => Unit::Workgroup,
    };
    match view.workgroup_x {
        Some(x) => assert_eq!(x, dims.wg_x, "kernel compiled for another workgroup width"),
        None if unit != Unit::Workgroup => assert!(dims.wg_x as usize % width == 0, "workgroup width {} not divisible by W={}; compile with the workgroup layout", dims.wg_x, width),
        None => {}
    }
    if unit == Unit::Packet { assert!(wg_size % width == 0, "workgroup size {} not divisible by W={}", wg_size, width); }
    let num_wg = dims.num_wg_x as u64 * dims.num_wg_y as u64 * dims.num_wg_z as u64;
    let packets_per_wave = WAVE / width;
    let waves_per_wg = wg_size.div_ceil(WAVE);
    let packets_per_wg = match unit { Unit::Packet => wg_size / width, _ => waves_per_wg * packets_per_wave };
    let (unit_packets, unit_waves) = match unit {
        Unit::Packet => (1, 1),
        Unit::Wave => (packets_per_wave, 1),
        Unit::Workgroup => (packets_per_wg, waves_per_wg),
    };
    let units = match unit {
        Unit::Packet => num_wg * packets_per_wg as u64,
        Unit::Wave => num_wg * waves_per_wg as u64,
        Unit::Workgroup => num_wg,
    };
    if units == 0 { return; }
    let stride = match unit {
        Unit::Workgroup => (private_segment_size as usize).max(view.min_private_bytes).div_ceil(16) * 16,
        Unit::Wave => (private_segment_size as usize).max(view.min_private_bytes).div_ceil(8) * 8,
        Unit::Packet => ((private_segment_size as usize / 8 + 2).max(view.min_private_bytes.div_ceil(8))) * 8,
    };
    let scratch = match unit { Unit::Packet => width * stride, _ => unit_waves * WAVE * stride };
    let lds = if unit == Unit::Workgroup { group_segment_size.max(LDS_MIN_BYTES) } else { 0 };
    let shape = Shape { packets: unit_packets, waves: unit_waves, words: view.num_vgprs.max(1) * width, fibers: view.fibers(), scratch, lds };
    let engine = Engine { view, kd, kernarg_ptr, aql_packet_addr, dims, private_segment_size, unit, wg_size, packets_per_wave, waves_per_wg, packets_per_wg, units, stride, shape };
    let threads = num_threads.max(1).min(units as usize);
    thread::scope(|scope| {
        for tid in 0..threads {
            let engine = &engine;
            scope.spawn(move || {
                let mut state = acquire(engine.shape);
                let mut index = tid as u64;
                while index < engine.units {
                    engine.run_unit(index, &mut state);
                    index += threads as u64;
                }
                release(engine.shape, state);
            });
        }
    });
}

impl Engine<'_> {
    fn locate(&self, index: u64) -> (u64, usize) {
        match self.unit {
            Unit::Packet => (index / self.packets_per_wg as u64, (index % self.packets_per_wg as u64) as usize * self.view.width),
            Unit::Wave => (index / self.waves_per_wg as u64, (index % self.waves_per_wg as u64) as usize * WAVE),
            Unit::Workgroup => (index, 0),
        }
    }

    #[inline(never)]
    fn start_unit(&self, index: u64, state: &mut State) {
        let (wg, local_base) = self.locate(index);
        let wg_id = (
            (wg % self.dims.num_wg_x as u64) as u32,
            ((wg / self.dims.num_wg_x as u64) % self.dims.num_wg_y as u64) as u32,
            ((wg / (self.dims.num_wg_x as u64 * self.dims.num_wg_y as u64)) % self.dims.num_wg_z as u64) as u32,
        );
        let fibers = self.view.fibers();
        if fibers {
            state.scratch.fill(0);
            state.lds.fill(0);
        }
        let width = self.view.width;
        for packet in 0..self.shape.packets {
            let wave = packet / self.packets_per_wave;
            let wave_base = if self.unit == Unit::Packet { 0 } else { wave * WAVE * self.stride };
            let scratch_base = if state.scratch.is_empty() { 0 } else { (unsafe { state.scratch.as_ptr().add(wave_base) }) as u64 };
            let local = local_base + packet * width;
            let valid_lanes = self.wg_size.saturating_sub(local).min(width);
            let sgprs = &mut state.sgprs[packet];
            setup_sgprs(&mut sgprs[..], self.kd, self.kernarg_ptr, self.aql_packet_addr, scratch_base, self.private_segment_size, wg_id);
            let valid_mask = if valid_lanes == 32 { u32::MAX } else { ((1u64 << valid_lanes) - 1) as u32 };
            if fibers { sgprs[self.view.exec] = valid_mask; }
            let vgprs = &mut state.vgprs[packet];
            vgprs.fill(0);
            for lane in 0..valid_lanes {
                let item = (local + lane) as u32;
                let x = item % self.dims.wg_x;
                let y = (item / self.dims.wg_x) % self.dims.wg_y;
                let z = item / (self.dims.wg_x * self.dims.wg_y);
                vgprs[lane] = x | (y << 10) | (z << 20);
            }
            state.done[packet] = valid_lanes == 0;
            if fibers {
                state.spill[packet].fill(0);
                if !state.done[packet] {
                    let Invoke::Fiber(kernel) = self.view.invoke else { unreachable!() };
                    state.fibers[packet].start(KernelArgs {
                        entry: kernel.addr(),
                        sgprs: sgprs.as_mut_ptr(),
                        vgprs: vgprs.as_mut_ptr(),
                        spill: state.spill[packet].as_mut_ptr(),
                        scratch_base,
                        scratch_stride: self.stride as u64,
                        lane_base: ((packet % self.packets_per_wave) * width) as u64,
                        valid_mask,
                        lds_base: if state.lds.is_empty() { 0 } else { state.lds.as_mut_ptr() as u64 },
                    });
                }
            }
        }
        if fibers {
            for wave in 0..self.shape.waves {
                state.waiting[wave] = None;
                let count = self.wg_size.saturating_sub(local_base + wave * WAVE).min(WAVE);
                state.valid[wave] = if count == WAVE { u32::MAX } else { (1u32 << count) - 1 };
            }
        }
    }

    #[inline(never)]
    fn run_unit(&self, index: u64, state: &mut State) {
        self.start_unit(index, state);
        match self.view.invoke {
            Invoke::Scalar(kernel) => {
                let scratch_base = if state.scratch.is_empty() { 0 } else { state.scratch.as_ptr() as u64 };
                unsafe { kernel.run(state.sgprs[0].as_mut_ptr(), state.vgprs[0].as_mut_ptr(), scratch_base); }
            }
            Invoke::Packet(kernel) => {
                let scratch_base = if state.scratch.is_empty() { 0 } else { state.scratch.as_ptr() as u64 };
                unsafe { kernel.run(state.sgprs[0].as_mut_ptr(), state.vgprs[0].as_mut_ptr(), scratch_base, self.stride as u64); }
            }
            Invoke::Fiber(_) => self.run_fibers(state),
        }
    }

    fn run_wave(&self, state: &mut State) {
        let width = self.view.width;
        let valid = state.valid[0];
        let mut live: u32 = 0;
        for packet in 0..self.packets_per_wave { if !state.done[packet] { live |= 1 << packet; } }
        while live != 0 {
            let mut boundary = FIBER_DONE;
            let mut remaining = live;
            while remaining != 0 {
                let packet = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let pc = state.fibers[packet].resume();
                if pc == FIBER_DONE {
                    live &= !(1 << packet);
                } else if boundary == FIBER_DONE {
                    boundary = pc;
                } else if boundary != pc {
                    panic!("wave reached a non-uniform boundary (packet {} at {}, others at {})", packet, pc, boundary);
                }
            }
            if boundary == FIBER_DONE { break; }
            let action = self.view.yields.get(boundary as usize).expect("packet yield lacks a typed effect");
            action.apply_wave(width, valid, &state.fibers);
        }
        for packet in 0..self.packets_per_wave { state.done[packet] = true; }
    }

    fn run_fibers(&self, state: &mut State) {
        if self.unit == Unit::Wave { return self.run_wave(state); }
        let ppw = self.packets_per_wave;
        let width = self.view.width;
        let mut barriers = super::barrier::Barriers::new(self.shape.waves);
        let mut passes = 0usize;
        loop {
            let mut progress = false;
            let mut live = false;
            for wave in 0..self.shape.waves {
                let range = wave * ppw..(wave + 1) * ppw;
                if range.clone().all(|p| state.done[p]) { continue; }
                live = true;
                if let Some(id) = state.waiting[wave] {
                    if !barriers.wait(wave, id) { continue; }
                    state.waiting[wave] = None;
                }
                progress = true;
                let mut boundary = None;
                for p in range.clone() {
                    if state.done[p] { continue; }
                    let r = state.fibers[p].resume();
                    match boundary {
                        Some(other) => assert_eq!(other, r, "wave reached a non-uniform boundary"),
                        None => boundary = Some(r),
                    }
                    state.resume[p] = r;
                    state.done[p] = r == FIBER_DONE;
                }
                let pc = boundary.unwrap();
                if pc == FIBER_DONE { continue; }
                let action = self.view.yields.get(pc as usize).expect("packet yield lacks a typed effect");
                let valid = state.valid[wave];
                if action.is_wave() {
                    action.apply_wave(width, valid, &state.fibers[range]);
                } else {
                    assert_eq!(self.unit, Unit::Workgroup, "barrier outside a workgroup scheduler");
                    let id = action.uniform_id(width, valid, &state.fibers[range.clone()]) & 31;
                    match action.op {
                        EffectOp::BarrierSignal { is_first } => {
                            let first = barriers.signal(wave, id);
                            if is_first { action.broadcast_result(width, valid, &state.fibers[range], first as u32); }
                        }
                        EffectOp::BarrierWait => state.waiting[wave] = Some(id),
                        _ => unreachable!(),
                    }
                }
            }
            if !live { break; }
            assert!(progress, "workgroup barrier deadlock");
            passes += 1;
            assert!(passes < 100_000, "packet dispatch did not converge");
        }
    }
}

#[cfg(test)]
pub(crate) fn dispatch_cooperative_vec(
    kernel: &CoopVecKernel,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    group_segment_size: usize,
    num_threads: usize,
) {
    run(View::cooperative(kernel, Scheduler::Workgroup), kd, kernarg_ptr, aql_packet_addr, dims, private_segment_size, group_segment_size, num_threads)
}
