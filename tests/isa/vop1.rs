//! VOP1: one source, one vector destination.

use crate::compare::*;
use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// Bit-exact comparison of a VOP1 f32 instruction against captured hardware.
pub(crate) fn check_vop1_f32(op: u32, cases: &[(u64, u32)]) {
    check_vop1_f32_ulp(op, 0, cases);
}

/// As above, but finite non-zero results may differ from hardware by up to
/// `ulp`. NaN, +-0, +-inf and denormal results are still compared bit-exactly.
pub(crate) fn check_vop1_f32_ulp(op: u32, ulp: i64, cases: &[(u64, u32)]) {
    assert!(cases.len() <= LANES, "at most {} cases per call", LANES);
    let harness = Harness::vop1();

    let mut src = vec![0u32; LANES * harness.src_stride];
    for (i, (input, _)) in cases.iter().enumerate() {
        src[i * harness.src_stride] = *input as u32;
        src[i * harness.src_stride + 1] = (*input >> 32) as u32;
    }
    let uni = vec![0u32; 8];
    let words = [vop1(op, 6, vgpr(0))];

    let mut failures = Vec::new();
    for engine in [Engine::Interpreter, Engine::LlvmJit] {
        let out = harness.run(engine, &words, &src, &uni);
        for (i, (input, hw)) in cases.iter().enumerate() {
            let got = out[i * harness.out_stride];
            if got == *hw {
                continue;
            }
            // Special values are pinned by the manual in every case, so the
            // tolerance never applies to them. A denormal result is not one of
            // them: a flush to zero still fails here, since the flushed side is
            // a zero, while a denormal an instruction's granted error apart
            // from the hardware's is within what the manual allows.
            let special = is_nan_f32(*hw)
                || is_nan_f32(got)
                || is_zero_f32(*hw)
                || is_zero_f32(got)
                || is_inf_f32(*hw)
                || is_inf_f32(got);
            let distance = ulp_f32(got, *hw);
            if !special && distance <= ulp {
                continue;
            }
            failures.push(format!(
                "  {:<11} src0={} hardware={} simulator={}{}",
                engine_name(engine),
                show_f32(*input as u32),
                show_f32(*hw),
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

/// Bit-exact comparison of a VOP1 f64 instruction against captured hardware.
pub(crate) fn check_vop1_f64(op: u32, cases: &[(u64, u64)]) {
    check_vop1_f64_ulp(op, 0, cases);
}

/// As above, but finite non-zero results may differ from hardware by up to
/// `ulp`. NaN, +-0, +-inf and denormal results are still compared bit-exactly.
pub(crate) fn check_vop1_f64_ulp(op: u32, ulp: i128, cases: &[(u64, u64)]) {
    assert!(cases.len() <= LANES, "at most {} cases per call", LANES);
    let harness = Harness::vop1();

    let mut src = vec![0u32; LANES * harness.src_stride];
    for (i, (input, _)) in cases.iter().enumerate() {
        src[i * harness.src_stride] = *input as u32;
        src[i * harness.src_stride + 1] = (*input >> 32) as u32;
    }
    let uni = vec![0u32; 8];
    let words = [vop1(op, 6, vgpr(0))];

    let mut failures = Vec::new();
    for engine in [Engine::Interpreter, Engine::LlvmJit] {
        let out = harness.run(engine, &words, &src, &uni);
        for (i, (input, hw)) in cases.iter().enumerate() {
            let lo = out[i * harness.out_stride] as u64;
            let got = lo | ((out[i * harness.out_stride + 1] as u64) << 32);
            if got == *hw {
                continue;
            }
            let special = is_nan_f64(*hw)
                || is_nan_f64(got)
                || is_zero_f64(*hw)
                || is_zero_f64(got)
                || is_inf_f64(*hw)
                || is_inf_f64(got);
            let distance = ulp_f64(got, *hw);
            if !special && distance <= ulp {
                continue;
            }
            failures.push(format!(
                "  {:<11} src0={} hardware={} simulator={}{}",
                engine_name(engine),
                show_f64(*input),
                show_f64(*hw),
                show_f64(got),
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

/// Bit-exact comparison of a VOP1 instruction whose result is an integer or a
/// packed half, against captured hardware.
pub(crate) fn check_vop1_u32(op: u32, cases: &[(u64, u32)]) {
    check_vop1_f32_ulp(op, 0, cases);
}

/// Integer results with a tolerance: the manual grants "1ULP accuracy" to the
/// float-to-integer conversions, which means the integer may be off by one --
/// a plain difference, not a float ULP distance.
pub(crate) fn check_vop1_u32_ulp(op: u32, tolerance: i64, cases: &[(u64, u32)]) {
    let harness = Harness::vop1();
    let mut src = vec![0u32; LANES * harness.src_stride];
    for (i, (input, _)) in cases.iter().enumerate() {
        src[i * harness.src_stride] = *input as u32;
        src[i * harness.src_stride + 1] = (*input >> 32) as u32;
    }
    let uni = vec![0u32; 8];
    let words = [vop1(op, 6, vgpr(0))];

    let mut failures = Vec::new();
    for engine in [Engine::Interpreter, Engine::LlvmJit] {
        let out = harness.run(engine, &words, &src, &uni);
        for (i, (input, hw)) in cases.iter().enumerate() {
            let got = out[i * harness.out_stride];
            let distance = (got as i64 - *hw as i64).abs();
            if distance <= tolerance {
                continue;
            }
            failures.push(format!(
                "  {:<11} src0=0x{:016X} hardware=0x{:08X} simulator=0x{:08X} (differ by {}, allowed {})",
                engine_name(engine), input, hw, got, distance, tolerance,
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

// -------------------------------------------------------------------- tests
#[test]
pub(crate) fn v_bfrev_b32_vop1() {
    // V_BFREV_B32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_u32(
        56,
        &[
            (0x0000_0000, 0x0000_0000), // 0
            (0x0000_0001, 0x8000_0000), // 1
            (0xFFFF_FFFF, 0xFFFF_FFFF), // -1 / UINT_MAX
            (0x8000_0000, 0x0000_0001), // INT_MIN
            (0x7FFF_FFFF, 0xFFFF_FFFE), // INT_MAX
            (0x0000_0002, 0x4000_0000), // 2
            (0x0000_FFFF, 0xFFFF_0000), // 0xFFFF
            (0xDEAD_BEEF, 0xF77D_B57B), // 0xDEADBEEF
            (0x0000_0010, 0x0800_0000), // 16
            (0x0000_00FF, 0xFF00_0000), // 255
        ],
    );
}

#[test]
pub(crate) fn v_ceil_f32_vop1() {
    // V_CEIL_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        34,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x8000_0000), // -0
            (0x3F80_0000, 0x3F80_0000), // 1.0
            (0xBF80_0000, 0xBF80_0000), // -1.0
            (0x7F80_0000, 0x7F80_0000), // +inf
            (0xFF80_0000, 0xFF80_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x3F80_0000), // min denorm
            (0x807F_FFFF, 0x8000_0000), // max -denorm
            (0x0080_0000, 0x3F80_0000), // min normal
            (0x7F7F_FFFF, 0x7F7F_FFFF), // max normal
            (0x3F00_0000, 0x3F80_0000), // 0.5
            (0x3FC0_0000, 0x4000_0000), // 1.5
            (0x4000_0000, 0x4000_0000), // 2.0
            (0xC020_0000, 0xC000_0000), // -2.5
            (0x4049_0FDB, 0x4080_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_clz_i32_u32_vop1() {
    // V_CLZ_I32_U32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_u32(
        57,
        &[
            (0x0000_0000, 0xFFFF_FFFF), // 0
            (0x0000_0001, 0x0000_001F), // 1
            (0xFFFF_FFFF, 0x0000_0000), // -1 / UINT_MAX
            (0x8000_0000, 0x0000_0000), // INT_MIN
            (0x7FFF_FFFF, 0x0000_0001), // INT_MAX
            (0x0000_0002, 0x0000_001E), // 2
            (0x0000_FFFF, 0x0000_0010), // 0xFFFF
            (0xDEAD_BEEF, 0x0000_0000), // 0xDEADBEEF
            (0x0000_0010, 0x0000_001B), // 16
            (0x0000_00FF, 0x0000_0018), // 255
        ],
    );
}

#[test]
pub(crate) fn v_cos_f32_vop1() {
    // V_COS_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        54,
        &[
            (0x0000_0000, 0x3F80_0000), // +0
            (0x8000_0000, 0x3F80_0000), // -0
            (0x3F80_0000, 0x3F80_0000), // 1.0
            (0xBF80_0000, 0x3F80_0000), // -1.0
            (0x7F80_0000, 0xFFC0_0000), // +inf
            (0xFF80_0000, 0xFFC0_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x3F80_0000), // min denorm
            (0x807F_FFFF, 0x3F80_0000), // max -denorm
            (0x0080_0000, 0x3F80_0000), // min normal
            (0x7F7F_FFFF, 0x3F80_0000), // max normal
            (0x3F00_0000, 0xBF80_0000), // 0.5
            (0x3FC0_0000, 0xBF80_0000), // 1.5
            (0x4000_0000, 0x3F80_0000), // 2.0
            (0xC020_0000, 0xBF80_0000), // -2.5
            (0x4049_0FDB, 0x3F21_32CB), // pi
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f16_f32_vop1() {
    // V_CVT_F16_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_u32(
        10,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x0000_8000), // -0
            (0x3F80_0000, 0x0000_3C00), // 1.0
            (0xBF80_0000, 0x0000_BC00), // -1.0
            (0x7F80_0000, 0x0000_7C00), // +inf
            (0xFF80_0000, 0x0000_FC00), // -inf
            (0x7FC0_0000, 0x0000_7E00), // qNaN
            (0x7FA0_0000, 0x0000_7F00), // sNaN
            (0x0000_0001, 0x0000_0000), // min denorm
            (0x807F_FFFF, 0x0000_8000), // max -denorm
            (0x0080_0000, 0x0000_0000), // min normal
            (0x7F7F_FFFF, 0x0000_7C00), // max normal
            (0x3F00_0000, 0x0000_3800), // 0.5
            (0x3FC0_0000, 0x0000_3E00), // 1.5
            (0x4000_0000, 0x0000_4000), // 2.0
            (0xC020_0000, 0x0000_C100), // -2.5
            (0x4049_0FDB, 0x0000_4248), // pi
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f32_f16_vop1() {
    // V_CVT_F32_F16 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        11,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x0000_8000, 0x8000_0000), // -0
            (0x0000_3C00, 0x3F80_0000), // 1.0
            (0x0000_BC00, 0xBF80_0000), // -1.0
            (0x0000_7C00, 0x7F80_0000), // +inf
            (0x0000_FC00, 0xFF80_0000), // -inf
            (0x0000_7E00, 0x7FC0_0000), // qNaN
            (0x0000_7D00, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x3380_0000), // min denorm
            (0x0000_7BFF, 0x477F_E000), // max normal
            (0x0000_4000, 0x4000_0000), // 2.0
            (0x0000_3800, 0x3F00_0000), // 0.5
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f32_f64_vop1() {
    // V_CVT_F32_F64 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        15,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000), // +0
            (0x8000_0000_0000_0000, 0x8000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x3F80_0000), // 1.0
            (0xBFF0_0000_0000_0000, 0xBF80_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0x7F80_0000), // +inf
            (0xFFF0_0000_0000_0000, 0xFF80_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x7FC0_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x7FE0_0000), // sNaN
            (0x0000_0000_0000_0001, 0x0000_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0x8000_0000), // max -denorm
            (0x0010_0000_0000_0000, 0x0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x7F80_0000), // max normal
            (0x3FE0_0000_0000_0000, 0x3F00_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x3FC0_0000), // 1.5
            (0x4000_0000_0000_0000, 0x4000_0000), // 2.0
            (0xC004_0000_0000_0000, 0xC020_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x4049_0FDB), // pi
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f32_i32_vop1() {
    // V_CVT_F32_I32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        5,
        &[
            (0x0000_0000, 0x0000_0000), // 0
            (0x0000_0001, 0x3F80_0000), // 1
            (0xFFFF_FFFF, 0xBF80_0000), // -1 / UINT_MAX
            (0x8000_0000, 0xCF00_0000), // INT_MIN
            (0x7FFF_FFFF, 0x4F00_0000), // INT_MAX
            (0x0000_0002, 0x4000_0000), // 2
            (0x0000_FFFF, 0x477F_FF00), // 0xFFFF
            (0xDEAD_BEEF, 0xCE05_4904), // 0xDEADBEEF
            (0x0000_0010, 0x4180_0000), // 16
            (0x0000_00FF, 0x437F_0000), // 255
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f32_u32_vop1() {
    // V_CVT_F32_U32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        6,
        &[
            (0x0000_0000, 0x0000_0000), // 0
            (0x0000_0001, 0x3F80_0000), // 1
            (0xFFFF_FFFF, 0x4F80_0000), // -1 / UINT_MAX
            (0x8000_0000, 0x4F00_0000), // INT_MIN
            (0x7FFF_FFFF, 0x4F00_0000), // INT_MAX
            (0x0000_0002, 0x4000_0000), // 2
            (0x0000_FFFF, 0x477F_FF00), // 0xFFFF
            (0xDEAD_BEEF, 0x4F5E_ADBF), // 0xDEADBEEF
            (0x0000_0010, 0x4180_0000), // 16
            (0x0000_00FF, 0x437F_0000), // 255
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f64_f32_vop1() {
    // V_CVT_F64_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f64(
        16,
        &[
            (0x0000_0000, 0x0000_0000_0000_0000), // +0
            (0x8000_0000, 0x8000_0000_0000_0000), // -0
            (0x3F80_0000, 0x3FF0_0000_0000_0000), // 1.0
            (0xBF80_0000, 0xBFF0_0000_0000_0000), // -1.0
            (0x7F80_0000, 0x7FF0_0000_0000_0000), // +inf
            (0xFF80_0000, 0xFFF0_0000_0000_0000), // -inf
            (0x7FC0_0000, 0x7FF8_0000_0000_0000), // qNaN
            (0x7FA0_0000, 0x7FFC_0000_0000_0000), // sNaN
            (0x0000_0001, 0x36A0_0000_0000_0000), // min denorm
            (0x807F_FFFF, 0xB80F_FFFF_C000_0000), // max -denorm
            (0x0080_0000, 0x3810_0000_0000_0000), // min normal
            (0x7F7F_FFFF, 0x47EF_FFFF_E000_0000), // max normal
            (0x3F00_0000, 0x3FE0_0000_0000_0000), // 0.5
            (0x3FC0_0000, 0x3FF8_0000_0000_0000), // 1.5
            (0x4000_0000, 0x4000_0000_0000_0000), // 2.0
            (0xC020_0000, 0xC004_0000_0000_0000), // -2.5
            (0x4049_0FDB, 0x4009_21FB_6000_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f64_i32_vop1() {
    // V_CVT_F64_I32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f64(
        4,
        &[
            (0x0000_0000, 0x0000_0000_0000_0000), // 0
            (0x0000_0001, 0x3FF0_0000_0000_0000), // 1
            (0xFFFF_FFFF, 0xBFF0_0000_0000_0000), // -1 / UINT_MAX
            (0x8000_0000, 0xC1E0_0000_0000_0000), // INT_MIN
            (0x7FFF_FFFF, 0x41DF_FFFF_FFC0_0000), // INT_MAX
            (0x0000_0002, 0x4000_0000_0000_0000), // 2
            (0x0000_FFFF, 0x40EF_FFE0_0000_0000), // 0xFFFF
            (0xDEAD_BEEF, 0xC1C0_A920_8880_0000), // 0xDEADBEEF
            (0x0000_0010, 0x4030_0000_0000_0000), // 16
            (0x0000_00FF, 0x406F_E000_0000_0000), // 255
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f64_u32_vop1() {
    // V_CVT_F64_U32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f64(
        22,
        &[
            (0x0000_0000, 0x0000_0000_0000_0000), // 0
            (0x0000_0001, 0x3FF0_0000_0000_0000), // 1
            (0xFFFF_FFFF, 0x41EF_FFFF_FFE0_0000), // -1 / UINT_MAX
            (0x8000_0000, 0x41E0_0000_0000_0000), // INT_MIN
            (0x7FFF_FFFF, 0x41DF_FFFF_FFC0_0000), // INT_MAX
            (0x0000_0002, 0x4000_0000_0000_0000), // 2
            (0x0000_FFFF, 0x40EF_FFE0_0000_0000), // 0xFFFF
            (0xDEAD_BEEF, 0x41EB_D5B7_DDE0_0000), // 0xDEADBEEF
            (0x0000_0010, 0x4030_0000_0000_0000), // 16
            (0x0000_00FF, 0x406F_E000_0000_0000), // 255
        ],
    );
}

#[test]
pub(crate) fn v_cvt_i32_f32_vop1() {
    // V_CVT_I32_F32 in the VOP1 encoding. ISA: "1ULP accuracy".
    check_vop1_u32_ulp(
        8,
        1,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x0000_0000), // -0
            (0x3F80_0000, 0x0000_0001), // 1.0
            (0xBF80_0000, 0xFFFF_FFFF), // -1.0
            (0x7F80_0000, 0x7FFF_FFFF), // +inf
            (0xFF80_0000, 0x8000_0000), // -inf
            (0x7FC0_0000, 0x0000_0000), // qNaN
            (0x7FA0_0000, 0x0000_0000), // sNaN
            (0x0000_0001, 0x0000_0000), // min denorm
            (0x807F_FFFF, 0x0000_0000), // max -denorm
            (0x0080_0000, 0x0000_0000), // min normal
            (0x7F7F_FFFF, 0x7FFF_FFFF), // max normal
            (0x3F00_0000, 0x0000_0000), // 0.5
            (0x3FC0_0000, 0x0000_0001), // 1.5
            (0x4000_0000, 0x0000_0002), // 2.0
            (0xC020_0000, 0xFFFF_FFFE), // -2.5
            (0x4049_0FDB, 0x0000_0003), // pi
        ],
    );
}

#[test]
pub(crate) fn v_cvt_i32_f64_vop1() {
    // V_CVT_I32_F64 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_u32(
        3,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000), // +0
            (0x8000_0000_0000_0000, 0x0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x0000_0001), // 1.0
            (0xBFF0_0000_0000_0000, 0xFFFF_FFFF), // -1.0
            (0x7FF0_0000_0000_0000, 0x7FFF_FFFF), // +inf
            (0xFFF0_0000_0000_0000, 0x8000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x0000_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0x0000_0000), // max -denorm
            (0x0010_0000_0000_0000, 0x0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x7FFF_FFFF), // max normal
            (0x3FE0_0000_0000_0000, 0x0000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x0000_0001), // 1.5
            (0x4000_0000_0000_0000, 0x0000_0002), // 2.0
            (0xC004_0000_0000_0000, 0xFFFF_FFFE), // -2.5
            (0x4009_21FB_5444_2D18, 0x0000_0003), // pi
        ],
    );
}

#[test]
pub(crate) fn v_cvt_u32_f32_vop1() {
    // V_CVT_U32_F32 in the VOP1 encoding. ISA: "1ULP accuracy".
    check_vop1_u32_ulp(
        7,
        1,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x0000_0000), // -0
            (0x3F80_0000, 0x0000_0001), // 1.0
            (0xBF80_0000, 0x0000_0000), // -1.0
            (0x7F80_0000, 0xFFFF_FFFF), // +inf
            (0xFF80_0000, 0x0000_0000), // -inf
            (0x7FC0_0000, 0x0000_0000), // qNaN
            (0x7FA0_0000, 0x0000_0000), // sNaN
            (0x0000_0001, 0x0000_0000), // min denorm
            (0x807F_FFFF, 0x0000_0000), // max -denorm
            (0x0080_0000, 0x0000_0000), // min normal
            (0x7F7F_FFFF, 0xFFFF_FFFF), // max normal
            (0x3F00_0000, 0x0000_0000), // 0.5
            (0x3FC0_0000, 0x0000_0001), // 1.5
            (0x4000_0000, 0x0000_0002), // 2.0
            (0xC020_0000, 0x0000_0000), // -2.5
            (0x4049_0FDB, 0x0000_0003), // pi
        ],
    );
}

#[test]
pub(crate) fn v_cvt_u32_f64_vop1() {
    // V_CVT_U32_F64 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_u32(
        21,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000), // +0
            (0x8000_0000_0000_0000, 0x0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x0000_0001), // 1.0
            (0xBFF0_0000_0000_0000, 0x0000_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0xFFFF_FFFF), // +inf
            (0xFFF0_0000_0000_0000, 0x0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x0000_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0x0000_0000), // max -denorm
            (0x0010_0000_0000_0000, 0x0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0xFFFF_FFFF), // max normal
            (0x3FE0_0000_0000_0000, 0x0000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x0000_0001), // 1.5
            (0x4000_0000_0000_0000, 0x0000_0002), // 2.0
            (0xC004_0000_0000_0000, 0x0000_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x0000_0003), // pi
        ],
    );
}

#[test]
pub(crate) fn v_exp_f32_vop1() {
    // V_EXP_F32 in the VOP1 encoding. ISA: "1ULP accuracy, denormals are flushed".
    check_vop1_f32_ulp(
        37,
        1,
        &[
            (0x0000_0000, 0x3F80_0000), // +0
            (0x8000_0000, 0x3F80_0000), // -0
            (0x3F80_0000, 0x4000_0000), // 1.0
            (0xBF80_0000, 0x3F00_0000), // -1.0
            (0x7F80_0000, 0x7F80_0000), // +inf
            (0xFF80_0000, 0x0000_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x3F80_0000), // min denorm
            (0x807F_FFFF, 0x3F80_0000), // max -denorm
            (0x0080_0000, 0x3F80_0000), // min normal
            (0x7F7F_FFFF, 0x7F80_0000), // max normal
            (0x3F00_0000, 0x3FB5_04F3), // 0.5
            (0x3FC0_0000, 0x4035_04F3), // 1.5
            (0x4000_0000, 0x4080_0000), // 2.0
            (0xC020_0000, 0x3E35_04F3), // -2.5
            (0x4049_0FDB, 0x410D_331C), // pi
        ],
    );
}

#[test]
pub(crate) fn v_floor_f32_vop1() {
    // V_FLOOR_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        36,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x8000_0000), // -0
            (0x3F80_0000, 0x3F80_0000), // 1.0
            (0xBF80_0000, 0xBF80_0000), // -1.0
            (0x7F80_0000, 0x7F80_0000), // +inf
            (0xFF80_0000, 0xFF80_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x0000_0000), // min denorm
            (0x807F_FFFF, 0xBF80_0000), // max -denorm
            (0x0080_0000, 0x0000_0000), // min normal
            (0x7F7F_FFFF, 0x7F7F_FFFF), // max normal
            (0x3F00_0000, 0x0000_0000), // 0.5
            (0x3FC0_0000, 0x3F80_0000), // 1.5
            (0x4000_0000, 0x4000_0000), // 2.0
            (0xC020_0000, 0xC040_0000), // -2.5
            (0x4049_0FDB, 0x4040_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_floor_f64_vop1() {
    // V_FLOOR_F64 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f64(
        26,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x3FF0_0000_0000_0000), // 1.0
            (0xBFF0_0000_0000_0000, 0xBFF0_0000_0000_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0x7FF0_0000_0000_0000), // +inf
            (0xFFF0_0000_0000_0000, 0xFFF0_0000_0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x7FFC_0000_0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x0000_0000_0000_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0xBFF0_0000_0000_0000), // max -denorm
            (0x0010_0000_0000_0000, 0x0000_0000_0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x7FEF_FFFF_FFFF_FFFF), // max normal
            (0x3FE0_0000_0000_0000, 0x0000_0000_0000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x3FF0_0000_0000_0000), // 1.5
            (0x4000_0000_0000_0000, 0x4000_0000_0000_0000), // 2.0
            (0xC004_0000_0000_0000, 0xC008_0000_0000_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x4008_0000_0000_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_fract_f64_vop1() {
    // V_FRACT_F64 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f64(
        62,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
            (0x8000_0000_0000_0000, 0x0000_0000_0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x0000_0000_0000_0000), // 1.0
            (0xBFF0_0000_0000_0000, 0x0000_0000_0000_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0xFFF8_0000_0000_0000), // +inf
            (0xFFF0_0000_0000_0000, 0xFFF8_0000_0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x7FFC_0000_0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x0000_0000_0000_0001), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0x3FEF_FFFF_FFFF_FFFF), // max -denorm
            (0x0010_0000_0000_0000, 0x0010_0000_0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x0000_0000_0000_0000), // max normal
            (0x3FE0_0000_0000_0000, 0x3FE0_0000_0000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x3FE0_0000_0000_0000), // 1.5
            (0x4000_0000_0000_0000, 0x0000_0000_0000_0000), // 2.0
            (0xC004_0000_0000_0000, 0x3FE0_0000_0000_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x3FC2_1FB5_4442_D180), // pi
        ],
    );
}

#[test]
pub(crate) fn v_frexp_exp_i32_f32_vop1() {
    // V_FREXP_EXP_I32_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_u32(
        63,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x0000_0000), // -0
            (0x3F80_0000, 0x0000_0001), // 1.0
            (0xBF80_0000, 0x0000_0001), // -1.0
            (0x7F80_0000, 0x0000_0000), // +inf
            (0xFF80_0000, 0x0000_0000), // -inf
            (0x7FC0_0000, 0x0000_0000), // qNaN
            (0x7FA0_0000, 0x0000_0000), // sNaN
            (0x0000_0001, 0xFFFF_FF6C), // min denorm
            (0x807F_FFFF, 0xFFFF_FF82), // max -denorm
            (0x0080_0000, 0xFFFF_FF83), // min normal
            (0x7F7F_FFFF, 0x0000_0080), // max normal
            (0x3F00_0000, 0x0000_0000), // 0.5
            (0x3FC0_0000, 0x0000_0001), // 1.5
            (0x4000_0000, 0x0000_0002), // 2.0
            (0xC020_0000, 0x0000_0002), // -2.5
            (0x4049_0FDB, 0x0000_0002), // pi
        ],
    );
}

#[test]
pub(crate) fn v_frexp_exp_i32_f64_vop1() {
    // V_FREXP_EXP_I32_F64 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_u32(
        60,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000), // +0
            (0x8000_0000_0000_0000, 0x0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x0000_0001), // 1.0
            (0xBFF0_0000_0000_0000, 0x0000_0001), // -1.0
            (0x7FF0_0000_0000_0000, 0x0000_0000), // +inf
            (0xFFF0_0000_0000_0000, 0x0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0xFFFF_FBCF), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0xFFFF_FC02), // max -denorm
            (0x0010_0000_0000_0000, 0xFFFF_FC03), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x0000_0400), // max normal
            (0x3FE0_0000_0000_0000, 0x0000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x0000_0001), // 1.5
            (0x4000_0000_0000_0000, 0x0000_0002), // 2.0
            (0xC004_0000_0000_0000, 0x0000_0002), // -2.5
            (0x4009_21FB_5444_2D18, 0x0000_0002), // pi
        ],
    );
}

#[test]
pub(crate) fn v_frexp_mant_f32_vop1() {
    // V_FREXP_MANT_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        64,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x8000_0000), // -0
            (0x3F80_0000, 0x3F00_0000), // 1.0
            (0xBF80_0000, 0xBF00_0000), // -1.0
            (0x7F80_0000, 0x7F80_0000), // +inf
            (0xFF80_0000, 0xFF80_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x3F00_0000), // min denorm
            (0x807F_FFFF, 0xBF7F_FFFE), // max -denorm
            (0x0080_0000, 0x3F00_0000), // min normal
            (0x7F7F_FFFF, 0x3F7F_FFFF), // max normal
            (0x3F00_0000, 0x3F00_0000), // 0.5
            (0x3FC0_0000, 0x3F40_0000), // 1.5
            (0x4000_0000, 0x3F00_0000), // 2.0
            (0xC020_0000, 0xBF20_0000), // -2.5
            (0x4049_0FDB, 0x3F49_0FDB), // pi
        ],
    );
}

#[test]
pub(crate) fn v_frexp_mant_f64_vop1() {
    // V_FREXP_MANT_F64 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f64(
        61,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x3FE0_0000_0000_0000), // 1.0
            (0xBFF0_0000_0000_0000, 0xBFE0_0000_0000_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0x7FF0_0000_0000_0000), // +inf
            (0xFFF0_0000_0000_0000, 0xFFF0_0000_0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x7FFC_0000_0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x3FE0_0000_0000_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0xBFEF_FFFF_FFFF_FFFE), // max -denorm
            (0x0010_0000_0000_0000, 0x3FE0_0000_0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x3FEF_FFFF_FFFF_FFFF), // max normal
            (0x3FE0_0000_0000_0000, 0x3FE0_0000_0000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x3FE8_0000_0000_0000), // 1.5
            (0x4000_0000_0000_0000, 0x3FE0_0000_0000_0000), // 2.0
            (0xC004_0000_0000_0000, 0xBFE4_0000_0000_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x3FE9_21FB_5444_2D18), // pi
        ],
    );
}

#[test]
pub(crate) fn v_log_f32_vop1() {
    // V_LOG_F32 in the VOP1 encoding. ISA: "1ULP accuracy, denormals are flushed".
    check_vop1_f32_ulp(
        39,
        1,
        &[
            (0x0000_0000, 0xFF80_0000), // +0
            (0x8000_0000, 0xFF80_0000), // -0
            (0x3F80_0000, 0x0000_0000), // 1.0
            (0xBF80_0000, 0xFFC0_0000), // -1.0
            (0x7F80_0000, 0x7F80_0000), // +inf
            (0xFF80_0000, 0xFFC0_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0xFF80_0000), // min denorm
            (0x807F_FFFF, 0xFF80_0000), // max -denorm
            (0x0080_0000, 0xC2FC_0000), // min normal
            (0x7F7F_FFFF, 0x42FF_FFFF), // max normal
            (0x3F00_0000, 0xBF80_0000), // 0.5
            (0x3FC0_0000, 0x3F15_C01A), // 1.5
            (0x4000_0000, 0x3F80_0000), // 2.0
            (0xC020_0000, 0xFFC0_0000), // -2.5
            (0x4049_0FDB, 0x3FD3_643A), // pi
        ],
    );
}

#[test]
pub(crate) fn v_mov_b32_vop1() {
    // V_MOV_B32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_u32(
        1,
        &[
            (0x0000_0000, 0x0000_0000), // 0
            (0x0000_0001, 0x0000_0001), // 1
            (0xFFFF_FFFF, 0xFFFF_FFFF), // -1 / UINT_MAX
            (0x8000_0000, 0x8000_0000), // INT_MIN
            (0x7FFF_FFFF, 0x7FFF_FFFF), // INT_MAX
            (0x0000_0002, 0x0000_0002), // 2
            (0x0000_FFFF, 0x0000_FFFF), // 0xFFFF
            (0xDEAD_BEEF, 0xDEAD_BEEF), // 0xDEADBEEF
            (0x0000_0010, 0x0000_0010), // 16
            (0x0000_00FF, 0x0000_00FF), // 255
        ],
    );
}

#[test]
pub(crate) fn v_not_b32_vop1() {
    // V_NOT_B32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_u32(
        55,
        &[
            (0x0000_0000, 0xFFFF_FFFF), // 0
            (0x0000_0001, 0xFFFF_FFFE), // 1
            (0xFFFF_FFFF, 0x0000_0000), // -1 / UINT_MAX
            (0x8000_0000, 0x7FFF_FFFF), // INT_MIN
            (0x7FFF_FFFF, 0x8000_0000), // INT_MAX
            (0x0000_0002, 0xFFFF_FFFD), // 2
            (0x0000_FFFF, 0xFFFF_0000), // 0xFFFF
            (0xDEAD_BEEF, 0x2152_4110), // 0xDEADBEEF
            (0x0000_0010, 0xFFFF_FFEF), // 16
            (0x0000_00FF, 0xFFFF_FF00), // 255
        ],
    );
}

#[test]
pub(crate) fn v_rcp_f32_vop1() {
    // V_RCP_F32 in the VOP1 encoding. ISA: "1ULP accuracy ... Denormals are flushed".
    check_vop1_f32_ulp(
        42,
        1,
        &[
            (0x0000_0000, 0x7F80_0000), // +0
            (0x8000_0000, 0xFF80_0000), // -0
            (0x3F80_0000, 0x3F80_0000), // 1.0
            (0xBF80_0000, 0xBF80_0000), // -1.0
            (0x7F80_0000, 0x0000_0000), // +inf
            (0xFF80_0000, 0x8000_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x7F80_0000), // min denorm
            (0x807F_FFFF, 0xFF80_0000), // max -denorm
            (0x0080_0000, 0x7E80_0000), // min normal
            (0x7F7F_FFFF, 0x0000_0000), // max normal
            (0x3F00_0000, 0x4000_0000), // 0.5
            (0x3FC0_0000, 0x3F2A_AAAA), // 1.5
            (0x4000_0000, 0x3F00_0000), // 2.0
            (0xC020_0000, 0xBECC_CCCD), // -2.5
            (0x4049_0FDB, 0x3EA2_F983), // pi
        ],
    );
}

#[test]
pub(crate) fn v_rcp_f64_vop1() {
    // V_RCP_F64 in the VOP1 encoding. ISA: "(2**29)ULP accuracy".
    check_vop1_f64_ulp(
        47,
        1 << 29,
        &[
            (0x0000_0000_0000_0000, 0x7FF0_0000_0000_0000), // +0
            (0x8000_0000_0000_0000, 0xFFF0_0000_0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x3FF0_0000_0000_0000), // 1.0
            (0xBFF0_0000_0000_0000, 0xBFF0_0000_0000_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0x0000_0000_0000_0000), // +inf
            (0xFFF0_0000_0000_0000, 0x8000_0000_0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x7FFC_0000_0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x7FF0_0000_0000_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0xFFD0_0000_0AF8_0000), // max -denorm
            (0x0010_0000_0000_0000, 0x7FD0_0000_0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x0004_0000_0000_0001), // max normal
            (0x3FE0_0000_0000_0000, 0x4000_0000_0000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x3FE5_5555_5400_0000), // 1.5
            (0x4000_0000_0000_0000, 0x3FE0_0000_0000_0000), // 2.0
            (0xC004_0000_0000_0000, 0xBFD9_9999_9C00_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x3FD4_5F30_6C40_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_rcp_iflag_f32_vop1() {
    // V_RCP_IFLAG_F32 in the VOP1 encoding. measured: 1 ULP, as for V_RCP_F32.
    check_vop1_f32_ulp(
        43,
        1,
        &[
            (0x0000_0000, 0x7F80_0000), // +0
            (0x8000_0000, 0xFF80_0000), // -0
            (0x3F80_0000, 0x3F80_0000), // 1.0
            (0xBF80_0000, 0xBF80_0000), // -1.0
            (0x7F80_0000, 0x0000_0000), // +inf
            (0xFF80_0000, 0x8000_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x7F80_0000), // min denorm
            (0x807F_FFFF, 0xFF80_0000), // max -denorm
            (0x0080_0000, 0x7E80_0000), // min normal
            (0x7F7F_FFFF, 0x0000_0000), // max normal
            (0x3F00_0000, 0x4000_0000), // 0.5
            (0x3FC0_0000, 0x3F2A_AAAA), // 1.5
            (0x4000_0000, 0x3F00_0000), // 2.0
            (0xC020_0000, 0xBECC_CCCD), // -2.5
            (0x4049_0FDB, 0x3EA2_F983), // pi
        ],
    );
}

#[test]
pub(crate) fn v_rndne_f32_vop1() {
    // V_RNDNE_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        35,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x8000_0000), // -0
            (0x3F80_0000, 0x3F80_0000), // 1.0
            (0xBF80_0000, 0xBF80_0000), // -1.0
            (0x7F80_0000, 0x7F80_0000), // +inf
            (0xFF80_0000, 0xFF80_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x0000_0000), // min denorm
            (0x807F_FFFF, 0x8000_0000), // max -denorm
            (0x0080_0000, 0x0000_0000), // min normal
            (0x7F7F_FFFF, 0x7F7F_FFFF), // max normal
            (0x3F00_0000, 0x0000_0000), // 0.5
            (0x3FC0_0000, 0x4000_0000), // 1.5
            (0x4000_0000, 0x4000_0000), // 2.0
            (0xC020_0000, 0xC000_0000), // -2.5
            (0x4049_0FDB, 0x4040_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_rndne_f64_vop1() {
    // V_RNDNE_F64 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f64(
        25,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x3FF0_0000_0000_0000), // 1.0
            (0xBFF0_0000_0000_0000, 0xBFF0_0000_0000_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0x7FF0_0000_0000_0000), // +inf
            (0xFFF0_0000_0000_0000, 0xFFF0_0000_0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x7FFC_0000_0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x0000_0000_0000_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000), // max -denorm
            (0x0010_0000_0000_0000, 0x0000_0000_0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x7FEF_FFFF_FFFF_FFFF), // max normal
            (0x3FE0_0000_0000_0000, 0x0000_0000_0000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x4000_0000_0000_0000), // 1.5
            (0x4000_0000_0000_0000, 0x4000_0000_0000_0000), // 2.0
            (0xC004_0000_0000_0000, 0xC000_0000_0000_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x4008_0000_0000_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_rsq_f32_vop1() {
    // V_RSQ_F32 in the VOP1 encoding. ISA: "1ULP accuracy, denormals are flushed".
    check_vop1_f32_ulp(
        46,
        1,
        &[
            (0x0000_0000, 0x7F80_0000), // +0
            (0x8000_0000, 0xFF80_0000), // -0
            (0x3F80_0000, 0x3F80_0000), // 1.0
            (0xBF80_0000, 0xFFC0_0000), // -1.0
            (0x7F80_0000, 0x0000_0000), // +inf
            (0xFF80_0000, 0xFFC0_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x7F80_0000), // min denorm
            (0x807F_FFFF, 0xFF80_0000), // max -denorm
            (0x0080_0000, 0x5F00_0000), // min normal
            (0x7F7F_FFFF, 0x1F80_0000), // max normal
            (0x3F00_0000, 0x3FB5_04F3), // 0.5
            (0x3FC0_0000, 0x3F51_05EC), // 1.5
            (0x4000_0000, 0x3F35_04F3), // 2.0
            (0xC020_0000, 0xFFC0_0000), // -2.5
            (0x4049_0FDB, 0x3F10_6EBA), // pi
        ],
    );
}

#[test]
pub(crate) fn v_rsq_f64_vop1() {
    // V_RSQ_F64 in the VOP1 encoding. ISA: "(2**29)ULP accuracy".
    check_vop1_f64_ulp(
        49,
        1 << 29,
        &[
            (0x0000_0000_0000_0000, 0x7FF0_0000_0000_0000), // +0
            (0x8000_0000_0000_0000, 0xFFF0_0000_0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x3FF0_0000_0000_0000), // 1.0
            (0xBFF0_0000_0000_0000, 0xFFF8_0000_0000_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0x0000_0000_0000_0000), // +inf
            (0xFFF0_0000_0000_0000, 0xFFF8_0000_0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x7FFC_0000_0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x6180_0000_0000_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0xFFF8_0000_0000_0000), // max -denorm
            (0x0010_0000_0000_0000, 0x5FE0_0000_0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x1FF0_0000_019E_0000), // max normal
            (0x3FE0_0000_0000_0000, 0x3FF6_A09E_6000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x3FEA_20BD_7400_0000), // 1.5
            (0x4000_0000_0000_0000, 0x3FE6_A09E_6000_0000), // 2.0
            (0xC004_0000_0000_0000, 0xFFF8_0000_0000_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x3FE2_0DD7_53BE_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_sin_f32_vop1() {
    // V_SIN_F32. The manual states no accuracy, and the hardware's sine is not
    // the one libm computes: the arguments that land in the denormal range
    // measure 6 ULP apart, which is the threshold used here.
    check_vop1_f32_ulp(
        53,
        6,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x8000_0000), // -0
            (0x3F80_0000, 0x0000_0000), // 1.0
            (0xBF80_0000, 0x0000_0000), // -1.0
            (0x7F80_0000, 0xFFC0_0000), // +inf
            (0xFF80_0000, 0xFFC0_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x0000_0006), // min denorm
            (0x807F_FFFF, 0x81C9_0FD3), // max -denorm
            (0x0080_0000, 0x01C9_0FD5), // min normal
            (0x7F7F_FFFF, 0x0000_0000), // max normal
            (0x3F00_0000, 0x0000_0000), // 0.5
            (0x3FC0_0000, 0x0000_0000), // 1.5
            (0x4000_0000, 0x0000_0000), // 2.0
            (0xC020_0000, 0x0000_0000), // -2.5
            (0x4049_0FDB, 0x3F46_DFE0), // pi
        ],
    );
}

#[test]
pub(crate) fn v_sqrt_f32_vop1() {
    // V_SQRT_F32 in the VOP1 encoding. ISA: "1ULP accuracy, denormals are flushed".
    check_vop1_f32_ulp(
        51,
        1,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x8000_0000), // -0
            (0x3F80_0000, 0x3F80_0000), // 1.0
            (0xBF80_0000, 0xFFC0_0000), // -1.0
            (0x7F80_0000, 0x7F80_0000), // +inf
            (0xFF80_0000, 0xFFC0_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x0000_0000), // min denorm
            (0x807F_FFFF, 0x8000_0000), // max -denorm
            (0x0080_0000, 0x2000_0000), // min normal
            (0x7F7F_FFFF, 0x5F7F_FFFF), // max normal
            (0x3F00_0000, 0x3F35_04F3), // 0.5
            (0x3FC0_0000, 0x3F9C_C470), // 1.5
            (0x4000_0000, 0x3FB5_04F3), // 2.0
            (0xC020_0000, 0xFFC0_0000), // -2.5
            (0x4049_0FDB, 0x3FE2_DFC5), // pi
        ],
    );
}

#[test]
pub(crate) fn v_sqrt_f64_vop1() {
    // V_SQRT_F64 in the VOP1 encoding. ISA: "(2**29)ULP accuracy".
    check_vop1_f64_ulp(
        52,
        1 << 29,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x3FF0_0000_0400_0000), // 1.0
            (0xBFF0_0000_0000_0000, 0xFFF8_0000_0000_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0x7FF0_0000_0000_0000), // +inf
            (0xFFF0_0000_0000_0000, 0xFFF8_0000_0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x7FFC_0000_0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x1E60_0000_0400_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0xFFF8_0000_0000_0000), // max -denorm
            (0x0010_0000_0000_0000, 0x2000_0000_0400_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x5FEF_FFFF_FC08_0000), // max normal
            (0x3FE0_0000_0000_0000, 0x3FE6_A09E_6400_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x3FF3_988E_1400_0000), // 1.5
            (0x4000_0000_0000_0000, 0x3FF6_A09E_6400_0000), // 2.0
            (0xC004_0000_0000_0000, 0xFFF8_0000_0000_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x3FFC_5BF8_9518_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_trunc_f32_vop1() {
    // V_TRUNC_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f32(
        33,
        &[
            (0x0000_0000, 0x0000_0000), // +0
            (0x8000_0000, 0x8000_0000), // -0
            (0x3F80_0000, 0x3F80_0000), // 1.0
            (0xBF80_0000, 0xBF80_0000), // -1.0
            (0x7F80_0000, 0x7F80_0000), // +inf
            (0xFF80_0000, 0xFF80_0000), // -inf
            (0x7FC0_0000, 0x7FC0_0000), // qNaN
            (0x7FA0_0000, 0x7FE0_0000), // sNaN
            (0x0000_0001, 0x0000_0000), // min denorm
            (0x807F_FFFF, 0x8000_0000), // max -denorm
            (0x0080_0000, 0x0000_0000), // min normal
            (0x7F7F_FFFF, 0x7F7F_FFFF), // max normal
            (0x3F00_0000, 0x0000_0000), // 0.5
            (0x3FC0_0000, 0x3F80_0000), // 1.5
            (0x4000_0000, 0x4000_0000), // 2.0
            (0xC020_0000, 0xC000_0000), // -2.5
            (0x4049_0FDB, 0x4040_0000), // pi
        ],
    );
}

#[test]
pub(crate) fn v_trunc_f64_vop1() {
    // V_TRUNC_F64 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
    // code determines the result exactly.
    check_vop1_f64(
        23,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000), // -0
            (0x3FF0_0000_0000_0000, 0x3FF0_0000_0000_0000), // 1.0
            (0xBFF0_0000_0000_0000, 0xBFF0_0000_0000_0000), // -1.0
            (0x7FF0_0000_0000_0000, 0x7FF0_0000_0000_0000), // +inf
            (0xFFF0_0000_0000_0000, 0xFFF0_0000_0000_0000), // -inf
            (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0000), // qNaN
            (0x7FF4_0000_0000_0000, 0x7FFC_0000_0000_0000), // sNaN
            (0x0000_0000_0000_0001, 0x0000_0000_0000_0000), // min denorm
            (0x800F_FFFF_FFFF_FFFF, 0x8000_0000_0000_0000), // max -denorm
            (0x0010_0000_0000_0000, 0x0000_0000_0000_0000), // min normal
            (0x7FEF_FFFF_FFFF_FFFF, 0x7FEF_FFFF_FFFF_FFFF), // max normal
            (0x3FE0_0000_0000_0000, 0x0000_0000_0000_0000), // 0.5
            (0x3FF8_0000_0000_0000, 0x3FF0_0000_0000_0000), // 1.5
            (0x4000_0000_0000_0000, 0x4000_0000_0000_0000), // 2.0
            (0xC004_0000_0000_0000, 0xC000_0000_0000_0000), // -2.5
            (0x4009_21FB_5444_2D18, 0x4008_0000_0000_0000), // pi
        ],
    );
}
