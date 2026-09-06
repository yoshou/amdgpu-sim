//! Cooperative *wavefront* dispatch for kernels whose only cross-lane traffic is
//! a wave-wide op (`v_wmma_*`, `v_readlane`/`v_writelane`) — including one that
//! sits **inside a loop**.
//!
//! This reuses the same coroutine machinery as the workgroup-barrier scheduler
//! ([`super::cooperative`]): each lane is a resumable, single-lane (W=0) scalar
//! coroutine ([`CoopKernel`]) that yields at a boundary and returns its resume
//! pc. Where [`super::segmented`] handles a cross-lane op *outside* loops by
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
use crate::rdna_instructions::SourceOperand;

use super::dispatch::{setup_sgprs, GridDims};
use super::emit::{CoopKernel, COOP_SGPR_BUF, COOP_SPILL_SLOTS};
#[cfg(test)]
use super::boundary::RegSet;
use super::emit_vec::CoopVecKernel;
use super::fiber::{Fiber, KernelArgs, FIBER_DONE};
use super::ir::ScalarProgram;

const WAVE: usize = 32;
const EXEC: usize = 126;

pub use super::lift::wave::YieldAction as XlaneOp;
#[cfg(test)]
use super::lift::wave::Destination;
#[cfg(test)]
use super::lift::wave::Operand;
#[cfg(test)]
use super::ir::typed::effect::{EffectOp, WaveOp};

pub(super) fn eval_uniform<const N: usize>(sgprs: &[[u32; N]], source: &SourceOperand) -> u32 {
    match source {
        SourceOperand::LiteralConstant(v) => *v,
        SourceOperand::IntegerConstant(v) => *v as u32,
        SourceOperand::FloatConstant(v) => (*v as f32).to_bits(),
        SourceOperand::ScalarRegister(r) => sgprs[0][*r as usize],
        _ => unreachable!("verified uniform wave operand"),
    }
}

pub fn split_at_xlane(program: &ScalarProgram) -> (ScalarProgram, BTreeMap<usize, XlaneOp>) {
    super::lift::wave::split(program, |action| action.is_wave())
}

/// Compile a split cross-lane program into a width-W packet kernel, passing
/// the boundary IO derived from the program's typed yields.
pub fn compile_xlane_vec(
    program: &ScalarProgram,
    _xlane: &BTreeMap<usize, XlaneOp>,
    num_vgprs: usize,
    width: u32,
) -> CoopVecKernel {
    super::compiler::Compiler::default().compile_cooperative_vec(program, num_vgprs, width)
}

/// Run a cross-lane cooperative kernel over the whole grid, one 32-lane wavefront
/// at a time. `xlane` maps each yield's resume pc to the wave-level op to apply
/// there (built by [`split_at_xlane`]).
pub fn dispatch_xlane(
    kernel: &CoopKernel,
    xlane: &BTreeMap<usize, XlaneOp>,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    num_threads: usize,
) {
    dispatch_xlane_vec(kernel, xlane, kd, kernarg_ptr, aql_packet_addr, dims,
        private_segment_size, num_threads);
}

/// Packed counterpart of [`dispatch_xlane`]. Each CPU worker owns complete
/// 32-lane waves; it advances all `32 / W` packets to a lifted cross-lane
/// boundary, applies the operation once to their typed argument/result frames,
/// and resumes the packets. No packet of a wave is scheduled on another thread.
pub fn dispatch_xlane_vec(
    kernel: &CoopVecKernel,
    _xlane: &BTreeMap<usize, XlaneOp>,
    kd: &KernelDescriptor,
    kernarg_ptr: u64,
    aql_packet_addr: u64,
    dims: GridDims,
    private_segment_size: u32,
    num_threads: usize,
) {
    let width = kernel.width as usize;
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
    };
    let num_wg = (dims.num_wg_x * dims.num_wg_y * dims.num_wg_z) as u64;
    let total_waves = num_wg * dispatch.waves_per_wg as u64;
    let num_threads = num_threads.max(1);

    thread::scope(|scope| {
        for tid in 0..num_threads {
            let dispatch = &dispatch;
            scope.spawn(move || {
                let mut bufs = dispatch.new_bufs();
                let mut wave = tid as u64;
                while wave < total_waves {
                    dispatch.run_wave(wave, &mut bufs);
                    wave += num_threads as u64;
                }
            });
        }
    });
}

/// The wave-invariant half of [`dispatch_xlane_vec`].
struct VecDispatch<'a> {
    kernel: &'a CoopVecKernel,
    xlane: &'a BTreeMap<usize, super::yield_values::YieldValues>,
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

impl VecDispatch<'_> {
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
            bufs.sgprs[packet][EXEC] = if valid_lanes == 32 {
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
                valid_mask: bufs.sgprs[packet][EXEC],
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
                let op = self.xlane.get(&(pc as usize)).unwrap_or_else(|| {
                    panic!("dispatch_xlane_vec: yield at {:#x} has no cross-lane op", pc)
                });
                let count=self.wg_size.saturating_sub((wave % self.waves_per_wg as u64) as usize*32).min(32);
                let valid=if count==32{u32::MAX}else{(1u32<<count)-1};
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
                crate::rdna_spmd::wmma::apply(VDST as u32, A as u32, B as u32, C as u32, width, &mut packets);
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
                XlaneOp::new(EffectOp::Wave(WaveOp::WriteLane),vec![Operand::Source(SourceOperand::ScalarRegister(5)),Operand::Source(SourceOperand::IntegerConstant(3)),Operand::Source(SourceOperand::VectorRegister(7))],vec![Destination::Vgpr(7)]),
            ),
            (20, XlaneOp::new(EffectOp::Wave(WaveOp::Wmma),(0..16).map(|r|Operand::Source(SourceOperand::VectorRegister(r))).collect(),(32..40).map(Destination::Vgpr).collect())),
            (
                30,
                XlaneOp::new(EffectOp::Wave(WaveOp::ReadLane),vec![Operand::Source(SourceOperand::VectorRegister(1)),Operand::Source(SourceOperand::IntegerConstant(0))],vec![Destination::Sgpr(2)]),
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
