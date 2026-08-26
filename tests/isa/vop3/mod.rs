//! VOP3: three sources, each taking the full operand field, with abs and neg
//! per source and clamp and omod on the result.
//!
//! The case shapes and the comparison live here; the tests themselves are split
//! by how many sources the instruction reads, since that decides what a sweep
//! means.

use crate::compare::*;
use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// One VOP3 case. Every operand and modifier the format has is a field, so a
/// test cannot leave one unstated.
pub(crate) struct Vop3F32 {
    src0: Src,
    src1: Src,
    src2: Src,
    /// One bit per source position.
    abs: u32,
    /// One bit per source position.
    neg: u32,
    clamp: bool,
    /// 0 = x1, 1 = x2, 2 = x4, 3 = /2.
    omod: u32,
    expected: u32,
}

/// Bit-exact comparison of a VOP3 f32 instruction against captured hardware.
pub(crate) fn check_vop3_f32(op: u32, cases: &[Vop3F32]) {
    check_vop3_f32_ulp(op, 0, cases);
}

/// As above, with a tolerance on finite non-zero results.
pub(crate) fn check_vop3_f32_ulp(op: u32, ulp: i64, cases: &[Vop3F32]) {
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
        let mut words = vop3(
            op, 6, field[0], field[1], field[2], case.abs, case.neg, case.clamp, case.omod,
        )
        .to_vec();
        words.extend(literal);

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = harness.run(engine, &words, &src, &uni)[0];
            if got == case.expected {
                continue;
            }
            // Special values are pinned by the manual in every case, so the
            // tolerance never applies to them. A denormal result is not one of
            // them: a flush to zero still fails here, since the flushed side is
            // a zero, while a denormal an instruction's granted error apart
            // from the hardware's is within what the manual allows.
            let special = is_nan_f32(case.expected)
                || is_nan_f32(got)
                || is_zero_f32(case.expected)
                || is_zero_f32(got)
                || is_inf_f32(case.expected)
                || is_inf_f32(got);
            let distance = ulp_f32(got, case.expected);
            if !special && distance <= ulp {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} (abs={:#03b} neg={:#03b} clamp={} omod={}) hardware={} simulator={}{}",
                engine_name(engine),
                i,
                case.abs,
                case.neg,
                case.clamp,
                case.omod,
                show_f32(case.expected),
                show_f32(got),
                if special {
                    String::new()
                } else {
                    format!(" ({} ULP, allowed {})", distance, ulp)
                },
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

/// One VOP3 case with 64-bit operands.
pub(crate) struct Vop3F64 {
    src0: Src,
    src1: Src,
    src2: Src,
    abs: u32,
    neg: u32,
    clamp: bool,
    omod: u32,
    expected: u64,
}

/// Bit-exact comparison of a VOP3 f64 instruction against captured hardware.
pub(crate) fn check_vop3_f64(op: u32, cases: &[Vop3F64]) {
    check_vop3_f64_ulp(op, 0, cases);
}

/// As above, with a tolerance on finite non-zero results.
pub(crate) fn check_vop3_f64_ulp(op: u32, ulp: i128, cases: &[Vop3F64]) {
    let harness = Harness::vop3();

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
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
        let mut words = vop3(
            op, 6, field[0], field[1], field[2], case.abs, case.neg, case.clamp, case.omod,
        )
        .to_vec();
        words.extend(literal);

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = harness.run(engine, &words, &src, &uni);
            let got = out[0] as u64 | ((out[1] as u64) << 32);
            if got == case.expected {
                continue;
            }
            // Special values are pinned by the manual in every case, so the
            // tolerance never applies to them. A denormal result is not one of
            // them: a flush to zero still fails here, since the flushed side is
            // a zero, while a denormal an instruction's granted error apart
            // from the hardware's is within what the manual allows.
            let special = is_nan_f64(case.expected)
                || is_nan_f64(got)
                || is_zero_f64(case.expected)
                || is_zero_f64(got)
                || is_inf_f64(case.expected)
                || is_inf_f64(got);
            let distance = ulp_f64(got, case.expected);
            if !special && distance <= ulp {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} (abs={:#03b} neg={:#03b} clamp={} omod={}) hardware={} simulator={}{}",
                engine_name(engine), i, case.abs, case.neg, case.clamp, case.omod,
                show_f64(case.expected), show_f64(got),
                if special { String::new() } else { format!(" ({} ULP, allowed {})", distance, ulp) },
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

/// Bit-exact comparison of a VOP3 instruction with an integer or packed-half
/// result, against captured hardware.
pub(crate) fn check_vop3_u32(op: u32, cases: &[Vop3F32]) {
    check_vop3_f32_ulp(op, 0, cases);
}

/// As above, for the VOP3 encoding.
pub(crate) fn check_vop3_u32_ulp(op: u32, tolerance: i64, cases: &[Vop3F32]) {
    let harness = Harness::vop3();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
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
        let mut words = vop3(
            op, 6, field[0], field[1], field[2], case.abs, case.neg, case.clamp, case.omod,
        )
        .to_vec();
        words.extend(literal);

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = harness.run(engine, &words, &src, &uni)[0];
            let distance = (got as i64 - case.expected as i64).abs();
            if distance <= tolerance {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} (abs={:#03b} neg={:#03b} clamp={} omod={}) hardware=0x{:08X} simulator=0x{:08X} (differ by {}, allowed {})",
                engine_name(engine), i, case.abs, case.neg, case.clamp, case.omod,
                case.expected, got, distance, tolerance,
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

/// The SGPR a VOP3SD case names as its scalar destination.
const SDST: u32 = 16;

/// One VOP3SD case: the operands, and both destinations the format writes.
pub(crate) struct Vop3sdCase {
    src0: Src,
    src1: Src,
    src2: Src,
    /// One bit per source position.
    neg: u32,
    /// The destination register pair of the lane the case is read from.
    expected: u64,
    /// The lane mask the instruction wrote to its scalar destination.
    expected_sdst: u32,
}

/// Bit-exact comparison of a VOP3SD instruction against captured hardware. The
/// sources are wave-uniform, so every lane computes the same thing and the
/// scalar destination is the same mask whichever lane is read.
pub(crate) fn check_vop3sd(op: u32, cases: &[Vop3sdCase]) {
    let harness = Harness::vop3();

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
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
        let mut words = vop3sd(op, 6, SDST, field[0], field[1], field[2], case.neg).to_vec();
        words.extend(literal);

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = harness.run(engine, &words, &src, &uni);
            let got = out[0] as u64 | ((out[1] as u64) << 32);
            let got_sdst = out[2];
            if got == case.expected && got_sdst == case.expected_sdst {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} (neg={:#03b}) hardware=({}, sdst {:#010X}) simulator=({}, sdst {:#010X})",
                engine_name(engine),
                i,
                case.neg,
                show_f64(case.expected),
                case.expected_sdst,
                show_f64(got),
                got_sdst,
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

mod binary;
mod scalar_dst;
mod ternary;
mod unary;
