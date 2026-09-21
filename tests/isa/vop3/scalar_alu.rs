//! The VOP3 instructions that read their source and write their
//! result as scalars.
//!
//! The wave computes one value, not one per lane, so a case states
//! the single word the instruction left in its scalar destination.

use super::*;
use crate::encoding::Src;

#[test]
fn v_s_log_f32_vop3() {
    // V_S_LOG_F32. ISA: "1ULP accuracy, denormals are flushed".
    check_vop3_scalar_f32_ulp(
        642,
        1,
        &[
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // +0
            Vop3ScalarF32 { src0: Src::Sgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -0
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xBF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3ScalarF32 { src0: Src::Sgpr(0xFF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -inf
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FA0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // min denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x807F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // max -denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x0080_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC2FC_0000 }, // min normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F7F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x42FF_FFFF }, // max normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // 0.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x3FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F15_C01A }, // 1.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 2.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xC020_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -2.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4049_0FDB), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FD3_643A }, // pi
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // neg src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // abs src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F80_0000 }, // clamp
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4000_0000 }, // omod 1
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4080_0000 }, // omod 2
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3F00_0000 }, // omod 3
            Vop3ScalarF32 { src0: Src::Inline(240), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
fn v_s_exp_f32_vop3() {
    // V_S_EXP_F32. ISA: "1ULP accuracy, denormals are flushed".
    check_vop3_scalar_f32_ulp(
        640,
        1,
        &[
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // +0
            Vop3ScalarF32 { src0: Src::Sgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // -0
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xBF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // -1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3ScalarF32 { src0: Src::Sgpr(0xFF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -inf
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FA0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // min denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x807F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // max -denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x0080_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // min normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F7F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FB5_04F3 }, // 0.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x3FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4035_04F3 }, // 1.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // 2.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xC020_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3E35_04F3 }, // -2.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4049_0FDB), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x410D_331C }, // pi
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3E80_0000 }, // neg src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // abs src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F80_0000 }, // clamp
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4100_0000 }, // omod 1
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4180_0000 }, // omod 2
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x4000_0000 }, // omod 3
            Vop3ScalarF32 { src0: Src::Inline(240), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FB5_04F3 }, // src0 an inline constant
        ],
    );
}

#[test]
fn v_s_rcp_f32_vop3() {
    // V_S_RCP_F32. ISA: "1ULP accuracy ... Denormals are flushed".
    check_vop3_scalar_f32_ulp(
        644,
        1,
        &[
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +0
            Vop3ScalarF32 { src0: Src::Sgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -0
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xBF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +inf
            Vop3ScalarF32 { src0: Src::Sgpr(0xFF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -inf
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FA0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // min denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x807F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // max -denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x0080_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7E80_0000 }, // min normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F7F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // max normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 0.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x3FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F2A_AAAA }, // 1.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 2.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xC020_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBECC_CCCD }, // -2.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4049_0FDB), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3EA2_F983 }, // pi
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xBF00_0000 }, // neg src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // abs src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F00_0000 }, // clamp
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x3F80_0000 }, // omod 1
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4000_0000 }, // omod 2
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3E80_0000 }, // omod 3
            Vop3ScalarF32 { src0: Src::Inline(240), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
fn v_s_sqrt_f32_vop3() {
    // V_S_SQRT_F32. ISA: "1ULP accuracy, denormals are flushed".
    check_vop3_scalar_f32_ulp(
        648,
        1,
        &[
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3ScalarF32 { src0: Src::Sgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xBF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3ScalarF32 { src0: Src::Sgpr(0xFF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -inf
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FA0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x807F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // max -denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x0080_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2000_0000 }, // min normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F7F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5F7F_FFFF }, // max normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F35_04F3 }, // 0.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x3FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F9C_C470 }, // 1.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FB5_04F3 }, // 2.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xC020_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -2.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4049_0FDB), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE2_DFC5 }, // pi
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // neg src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FB5_04F3 }, // abs src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F80_0000 }, // clamp
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4035_04F3 }, // omod 1
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x40B5_04F3 }, // omod 2
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3F35_04F3 }, // omod 3
            Vop3ScalarF32 { src0: Src::Inline(240), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F35_04F3 }, // src0 an inline constant
        ],
    );
}

#[test]
fn v_s_rsq_f32_vop3() {
    // V_S_RSQ_F32. ISA: "1ULP accuracy, denormals are flushed".
    check_vop3_scalar_f32_ulp(
        646,
        1,
        &[
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +0
            Vop3ScalarF32 { src0: Src::Sgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -0
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xBF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -1.0
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +inf
            Vop3ScalarF32 { src0: Src::Sgpr(0xFF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -inf
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x7FA0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3ScalarF32 { src0: Src::Sgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // min denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x807F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // max -denorm
            Vop3ScalarF32 { src0: Src::Sgpr(0x0080_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5F00_0000 }, // min normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x7F7F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1F80_0000 }, // max normal
            Vop3ScalarF32 { src0: Src::Sgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FB5_04F3 }, // 0.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x3FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F51_05EC }, // 1.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F35_04F3 }, // 2.0
            Vop3ScalarF32 { src0: Src::Sgpr(0xC020_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -2.5
            Vop3ScalarF32 { src0: Src::Sgpr(0x4049_0FDB), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F10_6EBA }, // pi
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // neg src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3F35_04F3 }, // abs src0
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F35_04F3 }, // clamp
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x3FB5_04F3 }, // omod 1
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4035_04F3 }, // omod 2
            Vop3ScalarF32 { src0: Src::Sgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3EB5_04F3 }, // omod 3
            Vop3ScalarF32 { src0: Src::Inline(240), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FB5_04F3 }, // src0 an inline constant
        ],
    );
}

