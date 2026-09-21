use std::thread;

use crate::processor::KernelDescriptor;

use super::super::ir::EffectOp;
use super::dispatch::{setup_sgprs, GridDims};
use super::fiber::{Fiber, KernelArgs, FIBER_DONE};
use super::kernel::{
    Kernel, Region, Scheduler, COOP_ENTER, COOP_LEAVE, COOP_SGPR_BUF,
    COOP_SPILL_SLOTS,
};
use super::yields::YieldValues;

const WAVE: usize = 32;

fn lanes_mask(width: usize) -> u32 {
    if width >= 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    }
}
fn fiber_stack_bytes(width: usize) -> usize {
    (8 << 10) * width.max(4)
}
const LDS_MIN_BYTES: usize = 128 * 1024;
const POOL_LIMIT: usize = 64;

#[derive(Clone)]
pub(crate) struct View<'a> {
    regions: &'a [Region],
    scheduler: Scheduler,
    width: usize,
    num_vgprs: usize,
    min_private_bytes: usize,
    workgroup_x: Option<u32>,
    yields: &'a [Vec<YieldValues>],
    exec: usize,
    frame: usize,
    depth: Vec<usize>,
}

impl<'a> View<'a> {
    pub(crate) fn of(kernel: &'a Kernel) -> Self {
        let regions = &kernel.regions;
        let mut depth = vec![0; regions.len()];
        let mut pending = vec![0];
        while let Some(r) = pending.pop() {
            for &child in &regions[r].children {
                depth[child] = depth[r] + 1;
                pending.push(child);
            }
        }
        Self {
            regions,
            scheduler: kernel.scheduler(),
            width: kernel.width() as usize,
            num_vgprs: kernel.num_vgprs,
            min_private_bytes: kernel.min_private_bytes,
            workgroup_x: kernel.workgroup_x,
            yields: &kernel.yields,
            exec: kernel.registers.exec as usize,
            frame: kernel.frame_words.max(1),
            depth,
        }
    }
    fn fibers(&self) -> usize {
        self.depth.iter().max().map_or(1, |deepest| deepest + 1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    Packet,
    Wave,
    Workgroup,
}

struct State {
    sgprs: Vec<[u32; COOP_SGPR_BUF]>,
    vgprs: Vec<Vec<u32>>,
    spill: Vec<Vec<u32>>,
    fibers: Vec<Vec<Fiber>>,
    frames: Vec<Vec<u32>>,
    args: Vec<Option<KernelArgs>>,
    active: Vec<Vec<usize>>,
    yielded: Vec<usize>,
    done: Vec<bool>,
    waiting: Vec<Option<u32>>,
    valid: Vec<u32>,
    scratch: aligned_vec::AVec<u8, aligned_vec::ConstAlign<0x1_0000_0000>>,
    lds: Vec<u8>,
}

unsafe impl Send for State {}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Shape {
    packets: usize,
    waves: usize,
    words: usize,
    fibers: usize,
    frame: usize,
    scratch: usize,
    lds: usize,
    stack: usize,
}

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
        vgprs: (0..shape.packets)
            .map(|_| vec![0u32; shape.words])
            .collect(),
        spill: (0..shape.packets)
            .map(|_| vec![0u32; COOP_SPILL_SLOTS])
            .collect(),
        fibers: (0..shape.fibers)
            .map(|_| Fiber::batch(shape.packets, shape.stack))
            .collect(),
        frames: (0..shape.packets)
            .map(|_| vec![0u32; shape.frame])
            .collect(),
        args: vec![None; shape.packets],
        active: vec![Vec::new(); shape.packets],
        yielded: vec![0; shape.packets],
        done: vec![true; shape.packets],
        waiting: vec![None; shape.waves],
        valid: vec![0; shape.waves],
        scratch,
        lds: vec![0u8; shape.lds],
    }
}

fn release(shape: Shape, state: State) {
    let mut pool = POOL.lock().unwrap();
    if pool.len() < POOL_LIMIT {
        pool.push((shape, state));
    }
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
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16 | 32));
    let wg_size = dims.workgroup_size() as usize;
    let unit = match view.scheduler {
        Scheduler::Independent => Unit::Packet,
        Scheduler::Wave => Unit::Wave,
        Scheduler::Workgroup => Unit::Workgroup,
    };
    match view.workgroup_x {
        Some(x) => assert_eq!(x, dims.wg_x, "kernel compiled for another workgroup width"),
        None if unit != Unit::Workgroup => assert!(
            dims.wg_x as usize % width == 0,
            "workgroup width {} not divisible by W={}; compile with the workgroup layout",
            dims.wg_x,
            width
        ),
        None => {}
    }
    if unit == Unit::Packet {
        assert!(
            wg_size % width == 0,
            "workgroup size {} not divisible by W={}",
            wg_size,
            width
        );
    }
    let num_wg = dims.num_wg_x as u64 * dims.num_wg_y as u64 * dims.num_wg_z as u64;
    let packets_per_wave = WAVE / width;
    let waves_per_wg = wg_size.div_ceil(WAVE);
    let packets_per_wg = match unit {
        Unit::Packet => wg_size / width,
        _ => waves_per_wg * packets_per_wave,
    };
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
    if units == 0 {
        return;
    }
    let stride = match unit {
        Unit::Workgroup => {
            (private_segment_size as usize)
                .max(view.min_private_bytes)
                .div_ceil(16)
                * 16
        }
        Unit::Wave => {
            (private_segment_size as usize)
                .max(view.min_private_bytes)
                .div_ceil(8)
                * 8
        }
        Unit::Packet => {
            ((private_segment_size as usize / 8 + 2).max(view.min_private_bytes.div_ceil(8))) * 8
        }
    };
    let scratch = match unit {
        Unit::Packet => width * stride,
        _ => unit_waves * WAVE * stride,
    };
    let lds = if unit == Unit::Workgroup {
        group_segment_size.max(LDS_MIN_BYTES)
    } else {
        0
    };
    let shape = Shape {
        packets: unit_packets,
        waves: unit_waves,
        words: view.num_vgprs.max(1) * width,
        fibers: view.fibers(),
        frame: view.frame,
        scratch,
        lds,
        stack: fiber_stack_bytes(width),
    };
    let engine = Engine {
        view,
        kd,
        kernarg_ptr,
        aql_packet_addr,
        dims,
        private_segment_size,
        unit,
        wg_size,
        packets_per_wave,
        waves_per_wg,
        packets_per_wg,
        units,
        stride,
        shape,
    };
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
            Unit::Packet => (
                index / self.packets_per_wg as u64,
                (index % self.packets_per_wg as u64) as usize * self.view.width,
            ),
            Unit::Wave => (
                index / self.waves_per_wg as u64,
                (index % self.waves_per_wg as u64) as usize * WAVE,
            ),
            Unit::Workgroup => (index, 0),
        }
    }

    #[inline(never)]
    fn start_unit(&self, index: u64, state: &mut State) {
        let (wg, local_base) = self.locate(index);
        let wg_id = (
            (wg % self.dims.num_wg_x as u64) as u32,
            ((wg / self.dims.num_wg_x as u64) % self.dims.num_wg_y as u64) as u32,
            ((wg / (self.dims.num_wg_x as u64 * self.dims.num_wg_y as u64))
                % self.dims.num_wg_z as u64) as u32,
        );
        if self.unit != Unit::Packet {
            state.lds.fill(0);
        }
        let width = self.view.width;
        for packet in 0..self.shape.packets {
            let wave = packet / self.packets_per_wave;
            let wave_base = if self.unit == Unit::Packet {
                0
            } else {
                wave * WAVE * self.stride
            };
            let scratch_base = if state.scratch.is_empty() {
                0
            } else {
                (unsafe { state.scratch.as_ptr().add(wave_base) }) as u64
            };
            let local = local_base + packet * width;
            let valid_lanes = self.wg_size.saturating_sub(local).min(width);
            let sgprs = &mut state.sgprs[packet];
            setup_sgprs(
                &mut sgprs[..],
                self.kd,
                self.kernarg_ptr,
                self.aql_packet_addr,
                scratch_base,
                self.private_segment_size,
                wg_id,
            );
            let valid_mask = if valid_lanes == 32 {
                u32::MAX
            } else {
                ((1u64 << valid_lanes) - 1) as u32
            };
            let wave_valid = {
                let count = self
                    .wg_size
                    .saturating_sub(local_base + (packet / self.packets_per_wave) * WAVE)
                    .min(WAVE);
                if count >= WAVE {
                    u32::MAX
                } else {
                    ((1u64 << count) - 1) as u32
                }
            };
            sgprs[self.view.exec] = valid_mask;
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
            {
                state.spill[packet].fill(0);
                let args = KernelArgs {
                    entry: self.view.regions[0].address,
                    sgprs: sgprs.as_mut_ptr(),
                    vgprs: vgprs.as_mut_ptr(),
                    spill: state.spill[packet].as_mut_ptr(),
                    scratch_base,
                    scratch_stride: self.stride as u64,
                    lane_base: ((packet % self.packets_per_wave) * width) as u64,
                    valid_mask: wave_valid,
                    lds_base: if state.lds.is_empty() {
                        0
                    } else {
                        state.lds.as_mut_ptr() as u64
                    },
                    frame: state.frames[packet].as_mut_ptr(),
                };
                state.args[packet] = Some(args);
                state.active[packet].clear();
                state.active[packet].push(0);
                if !state.done[packet] {
                    state.fibers[0][packet].start(args);
                }
            }
        }
        {
            for wave in 0..self.shape.waves {
                state.waiting[wave] = None;
                let count = self
                    .wg_size
                    .saturating_sub(local_base + wave * WAVE)
                    .min(WAVE);
                state.valid[wave] = if count == WAVE {
                    u32::MAX
                } else {
                    (1u32 << count) - 1
                };
            }
        }
    }

    #[inline(never)]
    fn run_unit(&self, index: u64, state: &mut State) {
        self.start_unit(index, state);
        self.run_regions(self.view.regions, state);
    }

    fn advance(&self, regions: &[Region], packet: usize, state: &mut State) -> u64 {
        loop {
            let at = *state.active[packet]
                .last()
                .expect("a packet left its outermost region");
            let level = self.view.depth[at];
            let left = state.fibers[level][packet].resume();
            if left == FIBER_DONE {
                assert_eq!(at, 0, "a nested region finished the kernel");
                return left;
            }
            if left & COOP_ENTER != 0 {
                let child = regions[at].children[(left & 0xffff_ffff) as usize];
                state.active[packet].push(child);
                let args = KernelArgs {
                    entry: regions[child].address,
                    ..state.args[packet].unwrap()
                };
                state.fibers[self.view.depth[child]][packet].start(args);
            } else if left & COOP_LEAVE != 0 {
                state.active[packet].pop();
                assert!(
                    !state.active[packet].is_empty(),
                    "the outermost region was left, not finished"
                );
                state.frames[packet][0] = (left & 0xffff_ffff) as u32;
            } else {
                assert_ne!(
                    regions[at].scheduler,
                    Scheduler::Independent,
                    "an independent region reached a meeting point"
                );
                state.yielded[packet] = level;
                return left;
            }
        }
    }

    fn run_wave(&self, regions: &[Region], state: &mut State) {
        let width = self.view.width;
        let valid = state.valid[0];
        let mut live: u32 = 0;
        for packet in 0..self.packets_per_wave {
            if !state.done[packet] {
                live |= 1 << packet;
            }
        }
        while live != 0 {
            let mut boundary = FIBER_DONE;
            let mut remaining = live;
            while remaining != 0 {
                let packet = remaining.trailing_zeros() as usize;
                remaining &= remaining - 1;
                let pc = self.advance(regions, packet, state);
                if pc == FIBER_DONE {
                    live &= !(1 << packet);
                } else if boundary == FIBER_DONE {
                    boundary = pc;
                } else if boundary != pc {
                    panic!(
                        "wave reached a non-uniform boundary (packet {} at {}, others at {})",
                        packet, pc, boundary
                    );
                }
            }
            if boundary == FIBER_DONE {
                break;
            }
            let group = self
                .view
                .yields
                .get(boundary as usize)
                .expect("packet yield lacks a typed effect");
            let holders: u32 = (0..self.packets_per_wave as u32)
                .filter(|p| valid >> (p * width as u32) & lanes_mask(width) != 0)
                .map(|p| 1 << p)
                .sum();
            assert_eq!(
                live & holders,
                holders,
                "wave met without all of its packets"
            );
            let level = state.yielded[live.trailing_zeros() as usize];
            for action in group {
                action.apply_wave(width, valid, &state.fibers[level]);
            }
        }
        for packet in 0..self.packets_per_wave {
            state.done[packet] = true;
        }
    }

    fn run_regions(&self, regions: &[Region], state: &mut State) {
        if self.unit == Unit::Packet {
            for packet in 0..self.shape.packets {
                if !state.done[packet] {
                    let left = self.advance(regions, packet, state);
                    assert_eq!(left, FIBER_DONE, "an independent packet reached a meeting point");
                    state.done[packet] = true;
                }
            }
            return;
        }
        if self.unit == Unit::Wave {
            return self.run_wave(regions, state);
        }
        let ppw = self.packets_per_wave;
        let width = self.view.width;
        let mut barriers = super::barrier::Barriers::new(self.shape.waves);
        let mut passes = 0usize;
        loop {
            let mut progress = false;
            let mut live = false;
            for wave in 0..self.shape.waves {
                let range = wave * ppw..(wave + 1) * ppw;
                if range.clone().all(|p| state.done[p]) {
                    continue;
                }
                live = true;
                if let Some(id) = state.waiting[wave] {
                    if !barriers.wait(wave, id) {
                        continue;
                    }
                    state.waiting[wave] = None;
                }
                progress = true;
                let mut boundary = None;
                let mut level = 0;
                for p in range.clone() {
                    if state.done[p] {
                        continue;
                    }
                    let r = self.advance(regions, p, state);
                    match boundary {
                        Some(other) => assert_eq!(other, r, "wave reached a non-uniform boundary"),
                        None => boundary = Some(r),
                    }
                    level = state.yielded[p];
                    state.done[p] = r == FIBER_DONE;
                }
                let pc = boundary.unwrap();
                if pc == FIBER_DONE {
                    continue;
                }
                let group = self
                    .view
                    .yields
                    .get(pc as usize)
                    .expect("packet yield lacks a typed effect");
                let action = &group[0];
                let valid = state.valid[wave];
                if action.is_wave() {
                    assert!(
                        range
                            .clone()
                            .enumerate()
                            .all(|(p, packet)| state.done[packet]
                                == (valid >> (p * width) & lanes_mask(width) == 0)),
                        "wave met without all of its packets"
                    );
                    for action in group {
                        action.apply_wave(width, valid, &state.fibers[level][range.clone()]);
                    }
                } else {
                    assert_eq!(
                        self.unit,
                        Unit::Workgroup,
                        "barrier outside a workgroup scheduler"
                    );
                    let id = action.uniform_id(width, valid, &state.fibers[level][range.clone()]) & 31;
                    match action.op {
                        EffectOp::BarrierSignal { is_first } => {
                            let first = barriers.signal(wave, id);
                            if is_first {
                                action.broadcast_result(
                                    width,
                                    valid,
                                    &state.fibers[level][range],
                                    first as u32,
                                );
                            }
                        }
                        EffectOp::BarrierWait => state.waiting[wave] = Some(id),
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
    }
}
