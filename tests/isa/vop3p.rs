//! VOP3P: the packed and mixed-precision instructions.
//!
//! Every operand and modifier the format has is a field of the case: the three
//! sources, the half each of them supplies (OPSEL and OPSEL_HI), the sign
//! applied to each half (NEG and NEG_HI) and CLAMP. Results are compared
//! bit-exactly -- these are packed integer and half-precision results, and the
//! manual grants no error on any of them.
//!
//! The WMMA and SWMMAC opcodes share this encoding but read whole matrices
//! spread across the wave, so they need a harness of their own and are not
//! covered here.

use crate::compare::*;
use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// One VOP3P case.
pub(crate) struct Vop3pCase {
    pub(crate) src0: Src,
    pub(crate) src1: Src,
    pub(crate) src2: Src,
    /// One bit per source: which half feeds the low result.
    pub(crate) opsel: u32,
    /// One bit per source: which half feeds the high result.
    pub(crate) opsel_hi: u32,
    /// One bit per source: negate the half that feeds the low result.
    pub(crate) neg: u32,
    /// One bit per source: negate the half that feeds the high result.
    pub(crate) neg_hi: u32,
    pub(crate) clamp: bool,
    /// v6 after the instruction.
    pub(crate) expected: u32,
}

/// Bit-exact comparison against captured hardware.
pub(crate) fn check_vop3p(op: u32, cases: &[Vop3pCase]) {
    check_vop3p_ulp(op, 0, cases);
}

/// As above, with a tolerance on finite non-zero results. The dot products need
/// one: their accumulator does not round the way a sequence of f32 additions
/// would, and no ordering of exact arithmetic reproduces it. The threshold is
/// the largest difference measured against the hardware, not a guess.
pub(crate) fn check_vop3p_ulp(op: u32, ulp: i64, cases: &[Vop3pCase]) {
    let harness = Harness::vop3();

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        // One case per run: the SGPR sources are wave-uniform, so cases that
        // name an SGPR cannot share a wave with cases that use a different
        // value there.
        let mut src = vec![0u32; LANES * harness.src_stride];
        let mut uni = vec![0u32; 8];
        let mut literal = Vec::new();
        let mut field = [0u32; 3];
        for (position, s) in [case.src0, case.src1, case.src2].iter().enumerate() {
            field[position] = match s {
                Src::Vgpr(value) => {
                    for lane in 0..LANES {
                        src[lane * harness.src_stride + position * 2] = *value as u32;
                        src[lane * harness.src_stride + position * 2 + 1] = (*value >> 32) as u32;
                    }
                    vgpr(position as u32 * 2)
                }
                Src::Sgpr(value) => {
                    uni[position * 2] = *value as u32;
                    uni[position * 2 + 1] = (*value >> 32) as u32;
                    10 + position as u32 * 2
                }
                Src::Inline(encoding) => *encoding,
                Src::Literal(value) => {
                    literal.push(*value as u32);
                    255
                }
            };
        }
        let mut words = vop3p(
            op,
            6,
            field[0],
            field[1],
            field[2],
            case.opsel,
            case.opsel_hi,
            case.neg,
            case.neg_hi,
            case.clamp,
        )
        .to_vec();
        words.extend(literal);

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = harness.run(engine, &words, &src, &uni)[0];
            if got == case.expected {
                continue;
            }
            // Special values are pinned in every case, so the tolerance never
            // applies to them.
            let special = is_nan_f32(case.expected)
                || is_nan_f32(got)
                || is_zero_f32(case.expected)
                || is_zero_f32(got)
                || is_inf_f32(case.expected)
                || is_inf_f32(got);
            if !special && ulp_f32(got, case.expected) <= ulp {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} (opsel={:#05b} opsel_hi={:#05b} neg={:#05b} neg_hi={:#05b} clamp={}) hardware=0x{:08X} simulator=0x{:08X}",
                engine_name(engine),
                i,
                case.opsel,
                case.opsel_hi,
                case.neg,
                case.neg_hi,
                case.clamp,
                case.expected,
                got,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} case-results differ from hardware:\n{}",
        failures.len(),
        cases.len() * 2,
        failures.join("\n"),
    );
}

#[test]
fn v_pk_mad_i16_vop3p() {
    check_vop3p(
        0,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_B800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_B800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xB800_B800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x3800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7FFF_8000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
        ],
    );
}

#[test]
fn v_pk_mul_lo_u16_vop3p() {
    check_vop3p(
        1,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_pk_add_i16_vop3p() {
    check_vop3p(
        2,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_7E00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_8200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x8200_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x8200_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x0200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x0200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7FFF_FC00 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
        ],
    );
}

#[test]
fn v_pk_sub_i16_vop3p() {
    check_vop3p(
        3,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_8000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_FA00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_FE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFA00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x7E00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x7E00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBE00_4000 },
        ],
    );
}

#[test]
fn v_pk_lshlrev_b16_vop3p() {
    check_vop3p(
        4,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
        ],
    );
}

#[test]
fn v_pk_lshrrev_b16_vop3p() {
    check_vop3p(
        5,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
        ],
    );
}

#[test]
fn v_pk_ashrrev_i16_vop3p() {
    check_vop3p(
        6,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
        ],
    );
}

#[test]
fn v_pk_max_i16_vop3p() {
    check_vop3p(
        7,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_0000 },
        ],
    );
}

#[test]
fn v_pk_min_i16_vop3p() {
    check_vop3p(
        8,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4000_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_C000 },
        ],
    );
}

#[test]
fn v_pk_mad_u16_vop3p() {
    check_vop3p(
        9,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_B800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_B800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xB800_B800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x3800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_FFFF },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
        ],
    );
}

#[test]
fn v_pk_add_u16_vop3p() {
    check_vop3p(
        10,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_7E00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_8200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x8200_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x8200_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x0200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x0200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8200_FC00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
        ],
    );
}

#[test]
fn v_pk_sub_u16_vop3p() {
    check_vop3p(
        11,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_8000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_FA00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_FE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFA00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_FC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x7E00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x7E00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFE00_7C00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBE00_4000 },
        ],
    );
}

#[test]
fn v_pk_max_u16_vop3p() {
    check_vop3p(
        12,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
        ],
    );
}

#[test]
fn v_pk_min_u16_vop3p() {
    check_vop3p(
        13,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4000_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_pk_fma_f16_vop3p() {
    check_vop3p(
        14,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4580_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4580_C300 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4580_4300 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4580_C100 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4580_4580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBE00_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC300_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4300_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC100_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4580_4100 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4580_4100 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4580_C100 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xC680_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC680_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4680_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4580_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4580_BE00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
        ],
    );
}

#[test]
fn v_pk_add_f16_vop3p() {
    check_vop3p(
        15,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4500_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4500_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4500_4400 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4500_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4500_4500 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBC00_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4400_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBC00_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4500_C200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4500_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4500_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x3C00_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xBC00_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4500_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4500_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4500_BC00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
        ],
    );
}

#[test]
fn v_pk_mul_f16_vop3p() {
    check_vop3p(
        16,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_C400 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_4600 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC400_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4600_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4600_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4600_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xC600_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC600_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4600_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_C000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_8000 },
        ],
    );
}

#[test]
fn v_dot2_f32_f16_vop3p() {
    // The accumulator is 2 ULP from exact arithmetic at worst here.
    check_vop3p_ulp(
        19,
        2,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x407F_FF7F },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3FFF_FEFE },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x410F_FFE0 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x407F_FF7F },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x413F_FFE0 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC080_0040 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC0C0_0040 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3F7F_FDFE },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC080_0040 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x40FF_FFC0 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x40FF_FFC0 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4080_0040 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xC100_0020 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC100_0020 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4080_0040 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x407F_FF7F },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x407F_FF7F },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x407F_FF7F },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
        ],
    );
}

#[test]
fn v_dot4_i32_iu8_vop3p() {
    check_vop3p(
        22,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7880 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_57F8 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_5900 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_9200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_9500 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7478 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_9200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3980 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
        ],
    );
}

#[test]
fn v_dot4_u32_u8_vop3p() {
    check_vop3p(
        23,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7880 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_57F8 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_5900 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_9200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_9500 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7478 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_9200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_7580 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
        ],
    );
}

#[test]
fn v_dot8_i32_iu4_vop3p() {
    check_vop3p(
        24,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3840 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3820 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3848 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3854 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3848 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3848 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3804 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
        ],
    );
}

#[test]
fn v_dot8_u32_u4_vop3p() {
    check_vop3p(
        25,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3840 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3820 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3848 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3854 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3848 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3848 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3834 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
        ],
    );
}

#[test]
fn v_dot2_f32_bf16_vop3p() {
    // The accumulator is 1 ULP from exact arithmetic at worst here.
    check_vop3p_ulp(
        26,
        1,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x427F_EFF7 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x426F_FFF7 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4280_7FFC },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x427F_EFF7 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x42FF_FFFC },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBD00_200E },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC080_8040 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3E6F_F7FC },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBD00_200E },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4280_07FC },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4280_07FC },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x427F_F008 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xC280_0804 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC280_0804 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x427F_F008 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x427F_EFF7 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x427F_EFF7 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x427F_EFF7 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xB800_3800 },
        ],
    );
}

#[test]
fn v_pk_min_num_f16_vop3p() {
    check_vop3p(
        27,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_C000 },
        ],
    );
}

#[test]
fn v_pk_max_num_f16_vop3p() {
    check_vop3p(
        28,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4200_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_0000 },
        ],
    );
}

#[test]
fn v_pk_minimum_f16_vop3p() {
    check_vop3p(
        29,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xC000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0xC200_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_C000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_C000 },
        ],
    );
}

#[test]
fn v_pk_maximum_f16_vop3p() {
    check_vop3p(
        30,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4200 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4200_BC00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4200_4000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x4000_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_3C00 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_3C00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_0000 },
        ],
    );
}

#[test]
fn v_fma_mix_f32_vop3p() {
    check_vop3p(
        32,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBFC0_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC060_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4060_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC020_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x40B0_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4280_FC56 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_BFF8 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC080_3C40 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4281_FC5A },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4020_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4020_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xC020_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xBFC0_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x4020_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0xBFC0_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBFC0_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBFC0_0000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3F00_0000 },
        ],
    );
}

#[test]
fn v_fma_mixlo_f16_vop3p() {
    check_vop3p(
        33,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_C300 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_4300 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_C100 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_4580 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_5408 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_5006 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_C402 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_5410 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x0000_4100 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x0000_4100 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0x0000_C100 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0x0000_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x0000_4100 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0x0000_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_BE00 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_BE00 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_3800 },
        ],
    );
}

#[test]
fn v_fma_mixhi_f16_vop3p() {
    check_vop3p(
        34,
        &[
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBE00_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b001, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC300_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b010, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4300_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b100, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC100_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b111, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4580_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b000, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x5408_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b001, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x5006_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b010, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC402_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b100, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x5410_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b001, neg_hi: 0b000, clamp: false,
                expected: 0x4100_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b010, neg_hi: 0b000, clamp: false,
                expected: 0x4100_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b100, neg_hi: 0b000, clamp: false,
                expected: 0xC100_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b001, clamp: false,
                expected: 0xBE00_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b010, clamp: false,
                expected: 0x4100_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b100, clamp: false,
                expected: 0xBE00_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 },
            Vop3pCase { src0: Src::Sgpr(0x40003C00), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBE00_0000 },
            Vop3pCase { src0: Src::Vgpr(0x40003C00), src1: Src::Sgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBE00_0000 },
            Vop3pCase { src0: Src::Inline(128), src1: Src::Vgpr(0x4200C000), src2: Src::Vgpr(0xB8003800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3800_0000 },
        ],
    );
}

#[test]
fn v_pk_mad_i16_special_values_vop3p() {
    check_vop3p(
        0,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0002 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_0000 }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0004 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0006_0000 }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0030_0022 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0033_0020 }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0002 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_0000 }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0004 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0006_0000 }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0030_0022 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0033_0020 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8008_8004 }, // INT16_MAX, INT16_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8009_8003 }, // INT16_MIN, INT16_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0008_0005 }, // -1, 1 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000A_0003 }, // 1, -1 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0009_0004 }, // 0, 0 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0008_0003 }, // -1, -1 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x000F_0010),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0018_0014 }, // 15, 16 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0010_000F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0019_0013 }, // 16, 15 in src2
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7FFF_7FFF }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7FFF_7FFF }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0002 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0002_0000 }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_mul_lo_u16_special_values_vop3p() {
    check_vop3p(
        1,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFD_0000 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_FFFE }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFD_0002 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_FFFE }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFD_FFFE }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x002D_0020 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0030_001E }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFD_0000 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_FFFE }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFD_0002 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_FFFE }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFD_FFFE }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x002D_0020 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0030_001E }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0001_0000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0001 }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0001_0001 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0001_0001 }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_add_i16_special_values_vop3p() {
    check_vop3p(
        2,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8002_8002 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_8001 }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0003 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0004_0001 }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0001 }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0012_0012 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0013_0011 }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8002_8002 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_8001 }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0003 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0004_0001 }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0001 }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0012_0012 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0013_0011 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7FFF_8000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_7FFF }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFE_0002 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0002_FFFE }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_sub_i16_special_values_vop3p() {
    check_vop3p(
        3,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFC_7FFE }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFD_7FFD }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFC_FFFF }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFE_FFFD }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFD_FFFE }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFC_FFFD }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000C_000E }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000D_000D }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8004_8002 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_8003 }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0004_0001 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0003 }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0004_0003 }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFF4_FFF2 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFF3_FFF3 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_lshlrev_b16_special_values_vop3p() {
    check_vop3p(
        4,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0002 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0000 }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0004 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0006_0000 }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0000 }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0002 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0000 }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFF8_0000 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_FFFC }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFF8_0004 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0008_FFFC }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFF8_FFFC }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0078_0040 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0080_003C }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_8000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_8000 }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_0002 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0002_8000 }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_lshrrev_b16_special_values_vop3p() {
    check_vop3p(
        5,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0002 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0000 }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0001 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_0000 }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0002 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0000 }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0FFF_2000 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x1000_1FFF }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x1FFF_0000 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_3FFF }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x1FFF_3FFF }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_0004 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0003 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_8000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_0000 }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0001_0000 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0001 }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_ashrrev_i16_special_values_vop3p() {
    check_vop3p(
        6,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0002 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0000 }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0001 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_0000 }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0002 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0000 }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0FFF_E000 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xF000_1FFF }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_0000 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_FFFF }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_FFFF }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_0004 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0003 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_8000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_0000 }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_0000 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_FFFF }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_max_i16_special_values_vop3p() {
    check_vop3p(
        7,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFF_0002 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_7FFF }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000F_0010 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0010_000F }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFF_0002 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_7FFF }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000F_0010 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0010_000F }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7FFF_8000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_7FFF }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_0001 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0001_FFFF }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_min_i16_special_values_vop3p() {
    check_vop3p(
        8,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_8000 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0002 }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_0001 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_FFFF }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_FFFF }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_8000 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0002 }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_0001 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_FFFF }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_FFFF }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7FFF_8000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_7FFF }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_0001 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0001_FFFF }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_mad_u16_special_values_vop3p() {
    check_vop3p(
        9,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0002 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_0000 }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0004 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0006_0000 }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0030_0022 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0033_0020 }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_0002 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_0000 }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0004 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0006_0000 }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0030_0022 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0033_0020 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8008_8004 }, // INT16_MAX, INT16_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8009_8003 }, // INT16_MIN, INT16_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0008_0005 }, // -1, 1 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000A_0003 }, // 1, -1 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0009_0004 }, // 0, 0 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0008_0003 }, // -1, -1 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x000F_0010),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0018_0014 }, // 15, 16 in src2
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0010_000F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0019_0013 }, // 16, 15 in src2
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_FFFF }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_FFFF }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_0002 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0002_FFFF }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_add_u16_special_values_vop3p() {
    check_vop3p(
        10,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8002_8002 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_8001 }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0003 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0004_0001 }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0001 }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0012_0012 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0013_0011 }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8002_8002 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_8001 }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0003 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0004_0001 }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0001 }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0012_0012 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0013_0011 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFE_FFFF }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_FFFE }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_0002 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0002_FFFF }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_sub_u16_special_values_vop3p() {
    check_vop3p(
        11,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFC_7FFE }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFD_7FFD }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFC_FFFF }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFE_FFFD }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFD_FFFE }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFC_FFFD }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000C_000E }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000D_000D }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8004_8002 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8003_8003 }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0004_0001 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_0003 }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0004_0003 }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFF4_FFF2 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFF3_FFF3 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_max_u16_special_values_vop3p() {
    check_vop3p(
        12,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFF_8000 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_7FFF }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_0002 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_FFFF }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_FFFF }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000F_0010 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0010_000F }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7FFF_8000 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8000_7FFF }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_0002 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_FFFF }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_FFFF }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x000F_0010 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0010_000F }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7FFF_8000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_7FFF }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_0001 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0001_FFFF }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_min_u16_special_values_vop3p() {
    check_vop3p(
        13,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // INT16_MAX, INT16_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // INT16_MIN, INT16_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0001 }, // -1, 1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_0002 }, // 1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0, 0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // -1, -1 in src0
            Vop3pCase { src0: Src::Vgpr(0x000F_0010), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 15, 16 in src0
            Vop3pCase { src0: Src::Vgpr(0x0010_000F), src1: Src::Vgpr(0x0003_0002), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 16, 15 in src0
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // INT16_MAX, INT16_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // INT16_MIN, INT16_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0001 }, // -1, 1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_0002 }, // 1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0, 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // -1, -1 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x000F_0010), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 15, 16 in src1
            Vop3pCase { src0: Src::Vgpr(0x0003_0002), src1: Src::Vgpr(0x0010_000F), src2: Src::Vgpr(0x0003_0002),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0003_0002 }, // 16, 15 in src1
            Vop3pCase { src0: Src::Vgpr(0x7FFF_8000), src1: Src::Vgpr(0x7FFF_8000), src2: Src::Vgpr(0x7FFF_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7FFF_8000 }, // INT16_MAX, INT16_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8000_7FFF), src1: Src::Vgpr(0x8000_7FFF), src2: Src::Vgpr(0x8000_7FFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8000_7FFF }, // INT16_MIN, INT16_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_0001), src1: Src::Vgpr(0xFFFF_0001), src2: Src::Vgpr(0xFFFF_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_0001 }, // -1, 1 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0001_FFFF), src1: Src::Vgpr(0x0001_FFFF), src2: Src::Vgpr(0x0001_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0001_FFFF }, // 1, -1 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_fma_f16_special_values_vop3p() {
    check_vop3p(
        14,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // +inf, +0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_3C00 }, // -inf, -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_4000 }, // qNaN, 1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_0000 }, // sNaN, -1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_03FF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7BFF_0400), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // max normal, min normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3800), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_3E00 }, // 2.0, 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0xBC00_7BFF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_7BFF }, // -1.0, max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // +inf, +0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_3C00 }, // -inf, -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_4000 }, // qNaN, 1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_0000 }, // sNaN, -1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x0001_03FF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7BFF_0400), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // max normal, min normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3800), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_3E00 }, // 2.0, 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xBC00_7BFF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_7BFF }, // -1.0, max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // +inf, +0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_3C00 }, // -inf, -0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_4000 }, // qNaN, 1.0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_0000 }, // sNaN, -1.0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x0001_03FF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4400_3C00 }, // min denorm, max denorm in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7BFF_0400),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7BFF_3C00 }, // max normal, min normal in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4600_3E00 }, // 2.0, 0.5 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0xBC00_7BFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4200_7BFF }, // -1.0, max normal in src2
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 }, // +inf, +0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -inf, -0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_3C00 }, // qNaN, 1.0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // sNaN, -1.0 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_add_f16_special_values_vop3p() {
    check_vop3p(
        15,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // +inf, +0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_3C00 }, // -inf, -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_4000 }, // qNaN, 1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_0000 }, // sNaN, -1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_03FF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7BFF_0400), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7BFF_3C00 }, // max normal, min normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3800), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4400_3E00 }, // 2.0, 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0xBC00_7BFF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_7BFF }, // -1.0, max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // +inf, +0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_3C00 }, // -inf, -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_4000 }, // qNaN, 1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_0000 }, // sNaN, -1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x0001_03FF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7BFF_0400), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7BFF_3C00 }, // max normal, min normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3800), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4400_3E00 }, // 2.0, 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xBC00_7BFF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3C00_7BFF }, // -1.0, max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 }, // +inf, +0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -inf, -0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_3C00 }, // qNaN, 1.0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // sNaN, -1.0 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_mul_f16_special_values_vop3p() {
    check_vop3p(
        16,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_0000 }, // +inf, +0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_8000 }, // -inf, -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_3C00 }, // qNaN, 1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_BC00 }, // sNaN, -1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_03FF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_03FF }, // min denorm, max denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7BFF_0400), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_0400 }, // max normal, min normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3800), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4400_3800 }, // 2.0, 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0xBC00_7BFF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_7BFF }, // -1.0, max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_0000 }, // +inf, +0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_8000 }, // -inf, -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_3C00 }, // qNaN, 1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_BC00 }, // sNaN, -1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x0001_03FF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0002_03FF }, // min denorm, max denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7BFF_0400), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_0400 }, // max normal, min normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3800), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4400_3800 }, // 2.0, 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xBC00_7BFF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xC000_7BFF }, // -1.0, max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 }, // +inf, +0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 }, // -inf, -0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_3C00 }, // qNaN, 1.0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_3C00 }, // sNaN, -1.0 everywhere, clamped
        ],
    );
}

#[test]
fn v_dot2_f32_f16_special_values_vop3p() {
    // A NaN reaching a multiplied source is left out: the part
    // answers those with a payload of its own that the manual does
    // not describe. A NaN in the accumulator, which it does, is here.
    check_vop3p_ulp(
        19,
        2,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F80_0000 }, // +inf, +0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFF80_0000 }, // -inf, -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_03FF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3D00 }, // min denorm, max denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7BFF_0400), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x47FF_E100 }, // max normal, min normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3800), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x40D0_1E00 }, // 2.0, 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0xBC00_7BFF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x477F_E000 }, // -1.0, max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F80_0000 }, // +inf, +0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFF80_0000 }, // -inf, -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x0001_03FF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3D00 }, // min denorm, max denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7BFF_0400), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x47FF_E100 }, // max normal, min normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3800), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x40D0_1E00 }, // 2.0, 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xBC00_7BFF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x477F_E000 }, // -1.0, max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_0000 }, // +inf, +0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_8000 }, // -inf, -0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_3C00 }, // qNaN, 1.0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7D00_BC00 }, // sNaN, -1.0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x0001_03FF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x40A0_0000 }, // min denorm, max denorm in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7BFF_0400),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7BFF_0400 }, // max normal, min normal in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x40E0_1C00 }, // 2.0, 0.5 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0xBC00_7BFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x409F_BFC2 }, // -1.0, max normal in src2
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7F80_0000 }, // +inf, +0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7F80_0000 }, // -inf, -0 everywhere, clamped
        ],
    );
}

#[test]
fn v_dot4_i32_iu8_special_values_vop3p() {
    check_vop3p(
        22,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7F7F_7F7F), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_07FA }, // every byte INT8_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0x8080_8080), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0804 }, // every byte INT8_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0CFA }, // every byte -1 / UINT8_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte 0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src0
            Vop3pCase { src0: Src::Vgpr(0x807F_7F80), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_07FF }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x7F7F_7F7F), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_07FA }, // every byte INT8_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x8080_8080), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0804 }, // every byte INT8_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0CFA }, // every byte -1 / UINT8_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x807F_7F80), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_07FF }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x7F7F_7F7F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F7F_7F9D }, // every byte INT8_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x8080_8080),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8080_809E }, // every byte INT8_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_001D }, // every byte -1 / UINT8_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_001E }, // every byte 0 in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x807F_7F80),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x807F_7F9E }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x7F7F_7F7F), src1: Src::Vgpr(0x7F7F_7F7F), src2: Src::Vgpr(0x7F7F_7F7F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7F80_7B83 }, // every byte INT8_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8080_8080), src1: Src::Vgpr(0x8080_8080), src2: Src::Vgpr(0x8080_8080),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8081_8080 }, // every byte INT8_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0003_F803 }, // every byte -1 / UINT8_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // every byte 0 everywhere, clamped
        ],
    );
}

#[test]
fn v_dot4_u32_u8_special_values_vop3p() {
    check_vop3p(
        23,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7F7F_7F7F), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_07FA }, // every byte INT8_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0x8080_8080), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0804 }, // every byte INT8_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0CFA }, // every byte -1 / UINT8_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte 0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src0
            Vop3pCase { src0: Src::Vgpr(0x807F_7F80), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_07FF }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x7F7F_7F7F), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_07FA }, // every byte INT8_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x8080_8080), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0804 }, // every byte INT8_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0CFA }, // every byte -1 / UINT8_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x807F_7F80), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_07FF }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x7F7F_7F7F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F7F_7F9D }, // every byte INT8_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x8080_8080),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8080_809E }, // every byte INT8_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_001D }, // every byte -1 / UINT8_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_001E }, // every byte 0 in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x807F_7F80),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x807F_7F9E }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x7F7F_7F7F), src1: Src::Vgpr(0x7F7F_7F7F), src2: Src::Vgpr(0x7F7F_7F7F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7F80_7B83 }, // every byte INT8_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8080_8080), src1: Src::Vgpr(0x8080_8080), src2: Src::Vgpr(0x8080_8080),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8081_8080 }, // every byte INT8_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_FFFF }, // every byte -1 / UINT8_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // every byte 0 everywhere, clamped
        ],
    );
}

#[test]
fn v_dot8_i32_iu4_special_values_vop3p() {
    check_vop3p(
        24,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7F7F_7F7F), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_039A }, // every byte INT8_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0x8080_8080), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte INT8_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_039A }, // every byte -1 / UINT8_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte 0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src0
            Vop3pCase { src0: Src::Vgpr(0x807F_7F80), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_034F }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x7F7F_7F7F), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_039A }, // every byte INT8_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x8080_8080), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte INT8_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_039A }, // every byte -1 / UINT8_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x807F_7F80), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_034F }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x7F7F_7F7F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F7F_7F9D }, // every byte INT8_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x8080_8080),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8080_809E }, // every byte INT8_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_001D }, // every byte -1 / UINT8_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_001E }, // every byte 0 in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x807F_7F80),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x807F_7F9E }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x7F7F_7F7F), src1: Src::Vgpr(0x7F7F_7F7F), src2: Src::Vgpr(0x7F7F_7F7F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7F7F_83C7 }, // every byte INT8_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8080_8080), src1: Src::Vgpr(0x8080_8080), src2: Src::Vgpr(0x8080_8080),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8080_8180 }, // every byte INT8_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0707 }, // every byte -1 / UINT8_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // every byte 0 everywhere, clamped
        ],
    );
}

#[test]
fn v_dot8_u32_u4_special_values_vop3p() {
    check_vop3p(
        25,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7F7F_7F7F), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_039A }, // every byte INT8_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0x8080_8080), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte INT8_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_039A }, // every byte -1 / UINT8_MAX in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte 0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src0
            Vop3pCase { src0: Src::Vgpr(0x807F_7F80), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_034F }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src0
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x7F7F_7F7F), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_039A }, // every byte INT8_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x8080_8080), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte INT8_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_039A }, // every byte -1 / UINT8_MAX in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0304 }, // every byte 0 in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x807F_7F80), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_034F }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src1
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x7F7F_7F7F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F7F_7F9D }, // every byte INT8_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x8080_8080),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x8080_809E }, // every byte INT8_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_001D }, // every byte -1 / UINT8_MAX in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_001E }, // every byte 0 in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x0102_0304),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0102_0322 }, // 1, 2, 3, 4 in src2
            Vop3pCase { src0: Src::Vgpr(0x0102_0304), src1: Src::Vgpr(0x0102_0304), src2: Src::Vgpr(0x807F_7F80),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x807F_7F9E }, // INT8_MIN, INT8_MAX, INT8_MAX, INT8_MIN in src2
            Vop3pCase { src0: Src::Vgpr(0x7F7F_7F7F), src1: Src::Vgpr(0x7F7F_7F7F), src2: Src::Vgpr(0x7F7F_7F7F),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7F7F_83C7 }, // every byte INT8_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x8080_8080), src1: Src::Vgpr(0x8080_8080), src2: Src::Vgpr(0x8080_8080),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x8080_8180 }, // every byte INT8_MIN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0xFFFF_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0xFFFF_FFFF }, // every byte -1 / UINT8_MAX everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // every byte 0 everywhere, clamped
        ],
    );
}

#[test]
fn v_dot2_f32_bf16_special_values_vop3p() {
    // A NaN reaching a multiplied source is left out: the part
    // answers those with a payload of its own that the manual does
    // not describe. A NaN in the accumulator, which it does, is here.
    check_vop3p_ulp(
        26,
        1,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C80_0000 }, // +inf, +0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC80_0000 }, // -inf, -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_03FF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7BFF_0400), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C7F_0000 }, // max normal, min normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3800), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x40C0_1E00 }, // 2.0, 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0xBC00_7BFF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x787F_0000 }, // -1.0, max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C80_0000 }, // +inf, +0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC80_0000 }, // -inf, -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x0001_03FF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7BFF_0400), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C7F_0000 }, // max normal, min normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3800), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x40C0_1E00 }, // 2.0, 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xBC00_7BFF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x787F_0000 }, // -1.0, max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_0000 }, // +inf, +0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_8000 }, // -inf, -0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_3C00 }, // qNaN, 1.0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7D00_BC00 }, // sNaN, -1.0 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x0001_03FF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4080_0080 }, // min denorm, max denorm in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x7BFF_0400),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7BFF_0400 }, // max normal, min normal in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3800),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x40C0_1C80 }, // 2.0, 0.5 in src2
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0xBC00_7BFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x407F_8084 }, // -1.0, max normal in src2
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7F80_0000 }, // +inf, +0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x7F80_0000 }, // -inf, -0 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_min_num_f16_special_values_vop3p() {
    check_vop3p(
        27,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_0000 }, // +inf, +0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_8000 }, // -inf, -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // qNaN, 1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_BC00 }, // sNaN, -1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_03FF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_03FF }, // min denorm, max denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7BFF_0400), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_0400 }, // max normal, min normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3800), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3800 }, // 2.0, 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0xBC00_7BFF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBC00_3C00 }, // -1.0, max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_0000 }, // +inf, +0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_8000 }, // -inf, -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // qNaN, 1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_BC00 }, // sNaN, -1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x0001_03FF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_03FF }, // min denorm, max denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7BFF_0400), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_0400 }, // max normal, min normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3800), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3800 }, // 2.0, 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xBC00_7BFF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBC00_3C00 }, // -1.0, max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 }, // +inf, +0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -inf, -0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_3C00 }, // qNaN, 1.0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // sNaN, -1.0 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_max_num_f16_special_values_vop3p() {
    check_vop3p(
        28,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // +inf, +0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // -inf, -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // qNaN, 1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // sNaN, -1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_03FF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7BFF_0400), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7BFF_3C00 }, // max normal, min normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3800), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // 2.0, 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0xBC00_7BFF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_7BFF }, // -1.0, max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // +inf, +0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // -inf, -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // qNaN, 1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // sNaN, -1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x0001_03FF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7BFF_0400), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7BFF_3C00 }, // max normal, min normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3800), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // 2.0, 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xBC00_7BFF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_7BFF }, // -1.0, max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 }, // +inf, +0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -inf, -0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_3C00 }, // qNaN, 1.0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // sNaN, -1.0 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_minimum_f16_special_values_vop3p() {
    check_vop3p(
        29,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_0000 }, // +inf, +0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_8000 }, // -inf, -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_3C00 }, // qNaN, 1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_BC00 }, // sNaN, -1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_03FF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_03FF }, // min denorm, max denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7BFF_0400), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_0400 }, // max normal, min normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3800), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3800 }, // 2.0, 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0xBC00_7BFF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBC00_3C00 }, // -1.0, max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_0000 }, // +inf, +0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFC00_8000 }, // -inf, -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_3C00 }, // qNaN, 1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_BC00 }, // sNaN, -1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x0001_03FF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_03FF }, // min denorm, max denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7BFF_0400), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_0400 }, // max normal, min normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3800), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3800 }, // 2.0, 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xBC00_7BFF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xBC00_3C00 }, // -1.0, max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 }, // +inf, +0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -inf, -0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_3C00 }, // qNaN, 1.0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // sNaN, -1.0 everywhere, clamped
        ],
    );
}

#[test]
fn v_pk_maximum_f16_special_values_vop3p() {
    check_vop3p(
        30,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // +inf, +0 in src0
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // -inf, -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_3C00 }, // qNaN, 1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_3C00 }, // sNaN, -1.0 in src0
            Vop3pCase { src0: Src::Vgpr(0x0001_03FF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7BFF_0400), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7BFF_3C00 }, // max normal, min normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3800), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // 2.0, 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0xBC00_7BFF), src1: Src::Vgpr(0x4000_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_7BFF }, // -1.0, max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7C00_3C00 }, // +inf, +0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // -inf, -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7E00_3C00 }, // qNaN, 1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7F00_3C00 }, // sNaN, -1.0 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x0001_03FF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // min denorm, max denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x7BFF_0400), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x7BFF_3C00 }, // max normal, min normal in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0x4000_3800), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_3C00 }, // 2.0, 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x4000_3C00), src1: Src::Vgpr(0xBC00_7BFF), src2: Src::Vgpr(0x4000_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x4000_7BFF }, // -1.0, max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x7C00_0000), src1: Src::Vgpr(0x7C00_0000), src2: Src::Vgpr(0x7C00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x3C00_0000 }, // +inf, +0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFC00_8000), src1: Src::Vgpr(0xFC00_8000), src2: Src::Vgpr(0xFC00_8000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -inf, -0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7E00_3C00), src1: Src::Vgpr(0x7E00_3C00), src2: Src::Vgpr(0x7E00_3C00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_3C00 }, // qNaN, 1.0 everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7D00_BC00), src1: Src::Vgpr(0x7D00_BC00), src2: Src::Vgpr(0x7D00_BC00),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // sNaN, -1.0 everywhere, clamped
        ],
    );
}

#[test]
fn v_fma_mix_f32_special_values_vop3p() {
    check_vop3p(
        32,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // +inf in src0
            Vop3pCase { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -inf in src0
            Vop3pCase { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // qNaN in src0
            Vop3pCase { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // sNaN in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // min denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_E000 }, // max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // +inf in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -inf in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // qNaN in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // sNaN in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // min denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_E000 }, // max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // +inf in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0xFF80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -inf in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7FC0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // qNaN in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7FA0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // sNaN in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x0000_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x3380_0000 }, // min denorm in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7F7F_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_E000 }, // max normal in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x8000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -0 in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0.5 in src2
            Vop3pCase { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x7F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // +inf everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0xFF80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -inf everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x7FC0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // qNaN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0x7FA0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // sNaN everywhere, clamped
        ],
    );
}

#[test]
fn v_fma_mixlo_f16_special_values_vop3p() {
    check_vop3p(
        33,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // +inf in src0
            Vop3pCase { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -inf in src0
            Vop3pCase { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // qNaN in src0
            Vop3pCase { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // sNaN in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // min denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_FFFF }, // max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // +inf in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -inf in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // qNaN in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // sNaN in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // min denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_FFFF }, // max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // +inf in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0xFF80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -inf in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7FC0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // qNaN in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7FA0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // sNaN in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x0000_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0001 }, // min denorm in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7F7F_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_FFFF }, // max normal in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x8000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -0 in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0.5 in src2
            Vop3pCase { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x7F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // +inf everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0xFF80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -inf everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x7FC0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // qNaN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0x7FA0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // sNaN everywhere, clamped
        ],
    );
}

#[test]
fn v_fma_mixhi_f16_special_values_vop3p() {
    check_vop3p(
        34,
        &[
            Vop3pCase { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // +inf in src0
            Vop3pCase { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -inf in src0
            Vop3pCase { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // qNaN in src0
            Vop3pCase { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // sNaN in src0
            Vop3pCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // min denorm in src0
            Vop3pCase { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_0000 }, // max normal in src0
            Vop3pCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -0 in src0
            Vop3pCase { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0.5 in src0
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // +inf in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -inf in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // qNaN in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // sNaN in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // min denorm in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_0000 }, // max normal in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -0 in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0x3F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0.5 in src1
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // +inf in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0xFF80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -inf in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7FC0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // qNaN in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7FA0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // sNaN in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x0000_0001),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0001_0000 }, // min denorm in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x7F7F_FFFF),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0xFFFF_0000 }, // max normal in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x8000_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // -0 in src2
            Vop3pCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F00_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: false,
                expected: 0x0000_0000 }, // 0.5 in src2
            Vop3pCase { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x7F80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // +inf everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0xFF80_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // -inf everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x7FC0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // qNaN everywhere, clamped
            Vop3pCase { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0x7FA0_0000),
                opsel: 0b000, opsel_hi: 0b111, neg: 0b000, neg_hi: 0b000, clamp: true,
                expected: 0x0000_0000 }, // sNaN everywhere, clamped
        ],
    );
}
