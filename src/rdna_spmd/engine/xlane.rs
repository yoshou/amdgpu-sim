//! Cooperative *wavefront* dispatch for kernels whose only cross-lane traffic is
//! a wave-wide op (`v_wmma_*`, `v_readlane`/`v_writelane`) — including one that
//! sits **inside a loop**.
//!
//! This reuses the same coroutine machinery as the workgroup-barrier scheduler
//! ([`super::cooperative`]): each lane is a resumable, single-lane (W=0) scalar
//! coroutine ([`CoopKernel`]) that yields at a boundary and returns its resume
//! pc. Where [`super::super::segmented`] handles a cross-lane op *outside* loops by
//! splitting the kernel into `[pre, boundary, post]` fragments, that model cannot
//! express a boundary reached many times by a back-edge. Modelling the boundary
//! as a coroutine *yield* removes that restriction: the loop's back-edge simply
//! flows back to the post-yield block, so the same boundary yields once per
//! iteration and the host driver runs the wave-level op each time.
//!
//! Only the cross-lane ops are lifted; every other instruction runs per-lane on
//! the scalar backend. For a *uniform* boundary (the same op reached the same
//! number of times by all lanes — true of `rocwmma`'s K-loop) all 32 lanes yield
//! at the same pc in lockstep, so the driver applies one wave-level op per pass.

use std::collections::BTreeMap;
use std::thread;

#[cfg(test)]
use half::f16;

use crate::processor::KernelDescriptor;
#[cfg(test)]
use crate::rdna_instructions::SourceOperand;

use super::dispatch::{setup_sgprs, GridDims};
use super::kernel::{COOP_SGPR_BUF, COOP_SPILL_SLOTS};
#[cfg(test)]
use super::super::lift::regs::RegSet;
use super::kernel::CoopVecKernel;
use super::fiber::{Fiber, KernelArgs, FIBER_DONE};

const WAVE: usize = 32;

/// The wave-level effect applied at a yield, keyed by the resume value the
/// fiber reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct XlaneOp(pub(in crate::rdna_spmd) super::super::ir::EffectOp);
#[cfg(test)]
use super::super::lift::wave::{Destination, YieldAction};
#[cfg(test)]
use super::super::lift::wave::Operand;
#[cfg(test)]
use super::super::ir::{EffectOp, WaveOp};

pub(crate) fn split_at_xlane(program: &impl super::super::CompilationInput) -> (super::super::Program, BTreeMap<usize, XlaneOp>) {
    let (program, ops) = program.to_ssa().schedule(|op| matches!(op, super::super::ir::EffectOp::Wave(w) if *w != super::super::ir::WaveOp::WriteLane));
    (program, ops.into_iter().map(|(key, op)| (key, XlaneOp(op))).collect())
}

/// Packed counterpart of [`dispatch_xlane`]. Each CPU worker owns complete
/// 32-lane waves; it advances all `32 / W` packets to a lifted cross-lane
/// boundary, applies the operation once to their typed argument/result frames,
/// and resumes the packets. No packet of a wave is scheduled on another thread.
pub(crate) fn dispatch_xlane_vec(
    kernel: &CoopVecKernel,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    num_threads: usize,
) {
    let width = kernel.width as usize;
    match kernel.workgroup_x {
        Some(x) => assert_eq!(x, dims.wg_x, "kernel compiled for another workgroup width"),
        None => assert!(dims.wg_x as usize % width == 0, "workgroup width {} not divisible by W={}; compile with the workgroup layout", dims.wg_x, width),
    }
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16));
    let wg_size = dims.workgroup_size() as usize;
    let scratch_u64 = (private_segment_size as usize).max(kernel.min_private_bytes).div_ceil(8);
    let dispatch = VecDispatch {
        kernel,
        xlane: &kernel.yields,
        kd,
        kernarg_ptr,
        aql_packet_addr,
        dims,
        private_segment_size,
        width,
        packets_per_wave: WAVE / width,
        wg_size,
        waves_per_wg: (wg_size + WAVE - 1) / WAVE,
        scratch_bytes: scratch_u64 * WAVE * 8,
        scratch_stride: (scratch_u64 * 8) as u64,
        exec: kernel.registers.exec as usize,
    };
    let num_wg = (dims.num_wg_x * dims.num_wg_y * dims.num_wg_z) as u64;
    let total_waves = num_wg * dispatch.waves_per_wg as u64;
    let num_threads = num_threads.max(1);

    thread::scope(|scope| {
        for tid in 0..num_threads {
            let dispatch = &dispatch;
            scope.spawn(move || {
                let mut bufs = dispatch.acquire_bufs();
                let mut wave = tid as u64;
                while wave < total_waves {
                    dispatch.run_wave(wave, &mut bufs);
                    wave += num_threads as u64;
                }
                release_bufs(bufs);
            });
        }
    });
}

/// The wave-invariant half of [`dispatch_xlane_vec`].
struct VecDispatch<'a> {
    kernel: &'a CoopVecKernel,
    xlane: &'a [super::yields::YieldValues],
    kd: &'a KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    width: usize,
    packets_per_wave: usize,
    wg_size: usize,
    waves_per_wg: usize,
    scratch_bytes: usize,
    scratch_stride: u64,
    exec: usize,
}

/// Per-worker state, reused for every wave the worker runs.
struct WaveBufs {
    sgprs: Vec<[u32; COOP_SGPR_BUF]>,
    vgprs: Vec<Vec<u32>>,
    spill: Vec<Vec<u32>>,
    fibers: Vec<Fiber>,
    resume: Vec<u64>,
    done: Vec<bool>,
    /// 4 GiB-aligned so its low 32 bits are zero: kernels using flat-scratch
    /// addressing take the pointer's high word from SRC_PRIVATE_BASE and add a
    /// per-lane low offset, so a nonzero low word would corrupt every private
    /// pointer.
    scratch: aligned_vec::AVec<u8, aligned_vec::ConstAlign<0x1_0000_0000>>,
}

/// Stack per packet fiber. A wave allocates `32 / W` of them, so the size is
/// kept modest; [`Fiber`] guards the deepest bytes, so a kernel whose frame
/// does not fit fails loudly rather than silently.
const FIBER_STACK_BYTES: usize = 32 << 10;

unsafe impl Send for WaveBufs {}

static BUF_POOL: std::sync::Mutex<Vec<WaveBufs>> = std::sync::Mutex::new(Vec::new());
const BUF_POOL_LIMIT: usize = 64;

fn release_bufs(bufs: WaveBufs) {
    let mut pool = BUF_POOL.lock().unwrap();
    if pool.len() < BUF_POOL_LIMIT { pool.push(bufs); }
}

impl VecDispatch<'_> {
    fn acquire_bufs(&self) -> WaveBufs {
        let packets = self.packets_per_wave;
        let words = self.kernel.num_vgprs * self.width;
        let mut pool = BUF_POOL.lock().unwrap();
        let found = pool.iter().position(|b| b.fibers.len() == packets && b.vgprs[0].len() == words && b.scratch.len() == self.scratch_bytes);
        match found {
            Some(index) => pool.swap_remove(index),
            None => { drop(pool); self.new_bufs() }
        }
    }
    fn new_bufs(&self) -> WaveBufs {
        let packets = self.packets_per_wave;
        let mut scratch = aligned_vec::AVec::new(0x1_0000_0000);
        scratch.resize(self.scratch_bytes, 0u8);
        WaveBufs {
            sgprs: vec![[0u32; COOP_SGPR_BUF]; packets],
            vgprs: (0..packets).map(|_| vec![0u32; self.kernel.num_vgprs * self.width]).collect(),
            spill: (0..packets).map(|_| vec![0u32; COOP_SPILL_SLOTS]).collect(),
            fibers: Fiber::batch(packets, FIBER_STACK_BYTES),
            resume: vec![0; packets],
            done: vec![true; packets],
            scratch,
        }
    }

    /// Reset every packet of `wave` to its entry state and arm its fiber.
    #[inline(never)]
    fn start_wave(&self, wave: u64, bufs: &mut WaveBufs) {
        let wg = wave / self.waves_per_wg as u64;
        let local_base = (wave % self.waves_per_wg as u64) as usize * WAVE;
        let wg_id = (
            (wg % self.dims.num_wg_x as u64) as u32,
            ((wg / self.dims.num_wg_x as u64) % self.dims.num_wg_y as u64) as u32,
            ((wg / (self.dims.num_wg_x as u64 * self.dims.num_wg_y as u64))
                % self.dims.num_wg_z as u64) as u32,
        );

        bufs.scratch.fill(0);
        let scratch_base = if bufs.scratch.is_empty() { 0 } else { bufs.scratch.as_ptr() as u64 };
        let initial_sgprs = setup_sgprs(
            self.kd,
            self.kernarg_ptr,
            self.aql_packet_addr,
            scratch_base,
            self.private_segment_size,
            wg_id,
        );

        for packet in 0..self.packets_per_wave {
            bufs.sgprs[packet] = [0u32; COOP_SGPR_BUF];
            bufs.sgprs[packet][..128].copy_from_slice(&initial_sgprs);
            bufs.vgprs[packet].fill(0);
            bufs.spill[packet].fill(0);

            let packet_base = local_base + packet * self.width;
            let valid_lanes = self.wg_size.saturating_sub(packet_base).min(self.width);
            bufs.done[packet] = valid_lanes == 0; // tail packet of a partial wave
            if bufs.done[packet] {
                continue;
            }
            let exec = self.exec;
            bufs.sgprs[packet][exec] = if valid_lanes == 32 {
                u32::MAX
            } else {
                ((1u64 << valid_lanes) - 1) as u32
            };
            for lane in 0..valid_lanes {
                let local = (packet_base + lane) as u32;
                let x = local % self.dims.wg_x;
                let y = (local / self.dims.wg_x) % self.dims.wg_y;
                let z = local / (self.dims.wg_x * self.dims.wg_y);
                bufs.vgprs[packet][lane] = x | (y << 10) | (z << 20);
            }
            bufs.fibers[packet].start(KernelArgs {
                lds_base: 0,
                valid_mask: bufs.sgprs[packet][exec],
                entry: self.kernel.addr(),
                sgprs: bufs.sgprs[packet].as_mut_ptr(),
                vgprs: bufs.vgprs[packet].as_mut_ptr(),
                spill: bufs.spill[packet].as_mut_ptr(),
                scratch_base,
                scratch_stride: self.scratch_stride,
                lane_base: (packet * self.width) as u64,
            });
        }
    }

    /// Run one 32-lane wave to completion: advance every live packet to its
    /// next boundary, apply the wave-level op once there, repeat.
    fn run_wave(&self, wave: u64, bufs: &mut WaveBufs) {
        self.start_wave(wave, bufs);
        let count = self.wg_size.saturating_sub((wave % self.waves_per_wg as u64) as usize * 32).min(32);
        let valid = if count == 32 { u32::MAX } else { (1u32 << count) - 1 };
        loop {
            let mut live = false;
            for packet in 0..self.packets_per_wave {
                if bufs.done[packet] {
                    continue;
                }
                live = true;
                let pc = bufs.fibers[packet].resume();
                bufs.done[packet] = pc == FIBER_DONE;
                bufs.resume[packet] = pc;
            }
            if !live {
                return;
            }
            if let Some(pc) = self.boundary_pc(wave, bufs) {
                let op = self.xlane.get(pc as usize).unwrap_or_else(|| {
                    panic!("dispatch_xlane_vec: yield {} has no cross-lane op", pc)
                });
                op.apply_wave(self.width, valid, &bufs.fibers);
            }
        }
    }

    /// The boundary every live packet stopped at, or `None` once they all
    /// finished. A wave-level op is one operation over all 32 lanes, so the
    /// live packets have to agree on which boundary they reached.
    fn boundary_pc(&self, wave: u64, bufs: &WaveBufs) -> Option<u64> {
        let mut boundary = None;
        for packet in 0..self.packets_per_wave {
            if bufs.done[packet] {
                continue;
            }
            match boundary {
                None => boundary = Some(bufs.resume[packet]),
                Some(pc) if pc != bufs.resume[packet] => panic!(
                    "dispatch_xlane_vec: wave {} reached a non-uniform boundary \
                     (packet {} at {:#x}, others at {:#x})",
                    wave, packet, bufs.resume[packet], pc
                ),
                _ => {}
            }
        }
        boundary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wmma_apply(vdst: usize, a: usize, b: usize, c: usize, vgprs: &mut [Vec<u32>]) {
        let lanes = vgprs.len().min(WAVE);
        let frag_f16 = |lane: usize, base: usize, m: usize| -> f32 {
            let word = vgprs[lane][base + m / 2];
            let bits = if m % 2 == 0 { (word & 0xffff) as u16 } else { (word >> 16) as u16 };
            f16::from_bits(bits).to_f32()
        };

        let mut mat_a = [0f32; 256];
        let mut mat_b = [0f32; 256];
        let mut mat_c = [0f32; 256];
        for e in 0..lanes {
            for i in 0..2 {
                for j in 0..2 {
                    for k in 0..2 {
                        let elem = k + j * 2 + i * 4;
                        let col = (k + j * 2 + i * 8) + (e / 16) * 4;
                        let row = e % 16;
                        mat_a[row * 16 + col] = frag_f16(e, a, elem);
                        let row_b = (k + j * 2 + i * 8) + (e / 16) * 4;
                        let col_b = e % 16;
                        mat_b[row_b * 16 + col_b] = frag_f16(e, b, elem);
                    }
                }
            }
            for m in 0..8 {
                let row = m + (e / 16) * 8;
                let col = e % 16;
                mat_c[row * 16 + col] = f32::from_bits(vgprs[e][c + m]);
            }
        }

        let mut mat_d = [0f32; 256];
        for i in 0..16 {
            for j in 0..16 {
                let mut acc = mat_c[i * 16 + j];
                for k in 0..16 {
                    acc += mat_a[i * 16 + k] * mat_b[k * 16 + j];
                }
                mat_d[i * 16 + j] = acc;
            }
        }

        for e in 0..lanes {
            for m in 0..8 {
                let row = m + (e / 16) * 8;
                let col = e % 16;
                vgprs[e][vdst + m] = mat_d[row * 16 + col].to_bits();
            }
        }
    }



    fn packet_vgpr(vgprs: &[Vec<u32>], width: usize, lane: usize, reg: usize) -> u32 {
        let packet = lane / width;
        let packet_lane = lane % width;
        vgprs[packet][reg * width + packet_lane]
    }

    fn set_packet_vgpr(
        vgprs: &mut [Vec<u32>],
        width: usize,
        lane: usize,
        reg: usize,
        value: u32,
    ) {
        let packet = lane / width;
        let packet_lane = lane % width;
        vgprs[packet][reg * width + packet_lane] = value;
    }


    #[test]
    fn packet_wmma_matches_scalar_lane_layout_at_every_supported_width() {
        const REGS: usize = 64;
        const VDST: usize = 32;
        const A: usize = 0;
        const B: usize = 4;
        const C: usize = 16;

        // Two input sets: arbitrary bit patterns — whose f16 fragments are
        // mostly Inf/NaN/subnormal — and the same patterns forced finite by
        // clearing one exponent bit of every f16 half and f32 accumulator.
        for finite in [false, true] {
            let mut state = vec![vec![0u32; REGS]; WAVE];
            let mut seed = 0x9e37_79b9u32;
            for lane in &mut state {
                for (reg, value) in lane.iter_mut().enumerate() {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    *value = seed;
                    if finite {
                        if (A..A + 4).contains(&reg) || (B..B + 4).contains(&reg) {
                            *value &= !0x4000_4000; // f16 halves
                        } else if (C..C + 8).contains(&reg) {
                            *value &= !0x4000_0000; // f32 accumulator
                        }
                    }
                }
            }

            let mut scalar = state.clone();
            wmma_apply(VDST, A, B, C, &mut scalar);
            for width in [1usize, 2, 4, 8, 16] {
                let mut packets = vec![vec![0u32; REGS * width]; WAVE / width];
                for lane in 0..WAVE {
                    for reg in 0..REGS {
                        set_packet_vgpr(&mut packets, width, lane, reg, state[lane][reg]);
                    }
                }
                crate::rdna_spmd::engine::wmma::apply(VDST as u32, A as u32, B as u32, C as u32, width, &mut packets);
                for lane in 0..WAVE {
                    for reg in VDST..VDST + 8 {
                        let actual = packet_vgpr(&packets, width, lane, reg);
                        let expected = scalar[lane][reg];
                        if actual == expected {
                            continue;
                        }
                        // Which NaN payload survives `acc + a * b` depends on
                        // the operand order the target picks for a commutative
                        // add, so a produced NaN is only required to *be* a
                        // NaN. Every other result must match bit for bit.
                        assert!(
                            !finite
                                && f32::from_bits(actual).is_nan()
                                && f32::from_bits(expected).is_nan(),
                            "width={}, lane={}, reg={}: \
                             {:#010x} != {:#010x} (finite inputs: {})",
                            width, lane, reg, actual, expected, finite
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn boundary_io_describes_each_lifted_op() {
        let ops = BTreeMap::from([
            (
                10,
                YieldAction::new(EffectOp::Wave(WaveOp::WriteLane),vec![Operand::Source(SourceOperand::ScalarRegister(5)),Operand::Source(SourceOperand::IntegerConstant(3)),Operand::Source(SourceOperand::VectorRegister(7)),Operand::Source(SourceOperand::IntegerConstant(7))],vec![Destination::Vgpr(7)]),
            ),
            (20, YieldAction::new(EffectOp::Wave(WaveOp::Wmma),(0..16).map(|r|Operand::Source(SourceOperand::VectorRegister(r))).collect(),(32..40).map(Destination::Vgpr).collect())),
            (
                30,
                YieldAction::new(EffectOp::Wave(WaveOp::ReadLane),vec![Operand::Source(SourceOperand::VectorRegister(1)),Operand::Source(SourceOperand::IntegerConstant(0)),Operand::Source(SourceOperand::IntegerConstant(1))],vec![Destination::Sgpr(2)]),
            ),
        ]);
        let io: BTreeMap<_,_> = ops.iter().map(|(&pc,action)|(pc,action.io())).collect();
        let vgprs = |set: &RegSet| set.vgprs().collect::<Vec<_>>();

        // writelane: a scalar value in, one lane of vdst out — and vdst is
        // read back too, so the lanes it does not touch survive the yield.
        assert_eq!(vgprs(&io[&10].reads), vec![7]);
        assert_eq!(vgprs(&io[&10].writes), vec![7]);
        assert!(io[&10].reads.has_sgpr(5));

        // WMMA: 4 + 4 f16 fragment registers and the 8-register accumulator
        // in, the 8-register result out.
        assert_eq!(vgprs(&io[&20].reads), (0..16).collect::<Vec<_>>());
        assert_eq!(vgprs(&io[&20].writes), (32..40).collect::<Vec<_>>());

        // readlane: one VGPR in, a uniform SGPR out.
        assert_eq!(vgprs(&io[&30].reads), vec![1]);
        assert!(vgprs(&io[&30].writes).is_empty());
        assert!(io[&30].writes.has_sgpr(2));
    }
}
