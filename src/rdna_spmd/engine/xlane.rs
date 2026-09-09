//! Wave scheduler: cooperative dispatch for kernels whose cross-lane traffic is
//! a wave-wide op (`v_wmma_*`, `ds_bpermute`, non-constant `v_readlane`) that
//! may sit inside a loop.
//!
//! It shares the fiber machinery with the workgroup scheduler
//! ([`super::cooperative`]): each packet is a resumable coroutine that yields at
//! a scheduled effect and reports its resume index. Modelling the boundary as a
//! yield lets a loop's back-edge flow back to the post-yield block, so the same
//! boundary yields once per iteration and the driver applies the wave-level op
//! each time. A worker owns whole 32-lane waves; for a uniform boundary all
//! packets of a wave stop at the same index and the op is applied once per pass.
//!
//! [`crate::rdna_spmd::compile`] selects this scheduler from the program;
//! kernels without exchange ops run on the independent dispatcher and kernels
//! with workgroup barriers on the workgroup scheduler.

use std::collections::BTreeMap;

#[cfg(test)]
use half::f16;

#[cfg(test)]
use crate::rdna_instructions::SourceOperand;

#[cfg(test)]
use crate::rdna_spmd::targets::rdna4::lift::regs::RegSet;

#[cfg(test)]
const WAVE: usize = 32;

/// The wave-level effect applied at a yield, keyed by the resume value the
/// fiber reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct XlaneOp(pub(in crate::rdna_spmd) super::super::ir::EffectOp);
#[cfg(test)]
use crate::rdna_spmd::targets::rdna4::lift::wave::{Destination, YieldAction};
#[cfg(test)]
use crate::rdna_spmd::targets::rdna4::lift::wave::Operand;
#[cfg(test)]
use super::super::ir::{EffectOp, WaveOp};

pub(crate) fn split_at_xlane(program: &impl super::super::CompilationInput) -> (super::super::Program, BTreeMap<usize, XlaneOp>) {
    let (program, ops) = program.to_ssa().schedule(|op| matches!(op, super::super::ir::EffectOp::Wave(w) if *w != super::super::ir::WaveOp::WriteLane));
    (program, ops.into_iter().map(|(key, op)| (key, XlaneOp(op))).collect())
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
