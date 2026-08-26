//! The VOP3 encoding of the instructions that have no other one.
//!
//! Three-source arithmetic, the bitfield operations and the
//! double-precision helpers: none of these has a VOP1, VOP2 or VOPC
//! form, so this is the only encoding they are tested in.

use super::*;
use crate::encoding::Src;

#[test]
fn v_fma_f32_vop3() {
    // V_FMA_F32.
    check_vop3_f32(
        531,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4020_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4060_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4090_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC090_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40D9_0FDB }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FA0_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4030_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4060_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC050_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40A6_CBE4 }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // +0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // -0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // 1.0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0xBF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // -1.0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x7F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0xFF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x7FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x7FA0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // min denorm in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x807F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // max -denorm in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0080_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // min normal in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x7F7F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4060_0000 }, // 0.5 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4090_0000 }, // 1.5 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40A0_0000 }, // 2.0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0xC020_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // -2.5 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x4049_0FDB), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40C4_87EE }, // pi in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xC020_0000 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4060_0000 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xC020_0000 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x4060_0000 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x4020_0000 }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x4060_0000 }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F80_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x40E0_0000 }, // omod 1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4160_0000 }, // omod 2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3FE0_0000 }, // omod 3
        ],
    );
}

#[test]
fn v_div_fixup_f32_vop3() {
    // V_DIV_FIXUP_F32.
    // The quotient's special cases: src0 is the quotient, src1 the numerator
    // and src2 the denominator.
    check_vop3_f32(
        551,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x007F_FFFF }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0080_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4020_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4049_0FDB }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0xBF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -1.0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x7F80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0xFF80_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x7FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x7FA0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min denorm in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x807F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // max -denorm in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0080_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min normal in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x7F7F_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max normal in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 0.5 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3FC0_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.5 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 2.0 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0xC020_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -2.5 in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x4049_0FDB), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // pi in src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F80_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4040_0000 }, // omod 1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x40C0_0000 }, // omod 2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F00_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3F40_0000 }, // omod 3
        ],
    );
}

#[test]
fn v_ldexp_f32_vop3() {
    // V_LDEXP_F32.
    // src1 is a signed integer exponent, not a float.
    check_vop3_f32(
        796,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4100_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC100_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0008 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x81FF_FFFE }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0200_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4140_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4180_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC1A0_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x41C9_0FDB }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xC140_0000 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4140_0000 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_0000 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x4140_0000 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F80_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x41C0_0000 }, // omod 1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4240_0000 }, // omod 2
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x40C0_0000 }, // omod 3
        ],
    );
}

#[test]
fn v_fma_f64_vop3() {
    // V_FMA_F64.
    check_vop3_f64(
        532,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // +0 in src0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // -0 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4004_0000_0000_0000 }, // 1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // -1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src0
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // min denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // max denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // min normal in src0
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // max normal in src0
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 0.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x400C_0000_0000_0000 }, // 1.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4012_0000_0000_0000 }, // 2.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC012_0000_0000_0000 }, // -2.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x401B_21FB_5444_2D18 }, // pi in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // +0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // -0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xBFF0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF4_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // min denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // max denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // min normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // max normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FE0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF4_0000_0000_0000 }, // 0.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4006_0000_0000_0000 }, // 1.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x400C_0000_0000_0000 }, // 2.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xC004_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC00A_0000_0000_0000 }, // -2.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4009_21FB_5444_2D18), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4014_D97C_7F33_21D2 }, // pi in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // +0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x8000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // -0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4010_0000_0000_0000 }, // 1.0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0xBFF0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // -1.0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x7FF0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0xFFF0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x7FF8_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x7FF4_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // min denorm in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // max denorm in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x0010_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // min normal in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x400C_0000_0000_0000 }, // 0.5 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FF8_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4012_0000_0000_0000 }, // 1.5 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x4000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4014_0000_0000_0000 }, // 2.0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0xC004_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // -2.5 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x4009_21FB_5444_2D18), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4018_90FD_AA22_168C }, // pi in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // neg src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x400C_0000_0000_0000 }, // abs src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // neg src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x400C_0000_0000_0000 }, // abs src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x4004_0000_0000_0000 }, // neg src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x400C_0000_0000_0000 }, // abs src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x401C_0000_0000_0000 }, // omod 1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x402C_0000_0000_0000 }, // omod 2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3FFC_0000_0000_0000 }, // omod 3
        ],
    );
}

#[test]
fn v_div_fixup_f64_vop3() {
    // V_DIV_FIXUP_F64.
    check_vop3_f64(
        552,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0 in src0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // -0 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // -1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src0
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // -inf in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // qNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // sNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0001 }, // min denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x000F_FFFF_FFFF_FFFF }, // max denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0010_0000_0000_0000 }, // min normal in src0
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src0
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 0.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 2.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4004_0000_0000_0000 }, // -2.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4009_21FB_5444_2D18 }, // pi in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xBFF0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // -1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF4_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FE0_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 0.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 2.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xC004_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // -2.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4009_21FB_5444_2D18), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // pi in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x8000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0xBFF0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // -1.0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x7FF0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0xFFF0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x7FF8_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x7FF4_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max denorm in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x0010_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min normal in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max normal in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 0.5 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FF8_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x4000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 2.0 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0xC004_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // -2.5 in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x4009_21FB_5444_2D18), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // pi in src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // neg src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // abs src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // neg src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // abs src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // neg src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // abs src2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4008_0000_0000_0000 }, // omod 1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4018_0000_0000_0000 }, // omod 2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FE0_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3FE8_0000_0000_0000 }, // omod 3
        ],
    );
}

#[test]
fn v_ldexp_f64_vop3() {
    // V_LDEXP_F64.
    // src1 is a signed integer exponent, not a float.
    check_vop3_f64(
        811,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0 in src0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4020_0000_0000_0000 }, // 1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000_0000_0000 }, // -1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src0
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0008 }, // min denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x003F_FFFF_FFFF_FFFE }, // max denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0040_0000_0000_0000 }, // min normal in src0
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // max normal in src0
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4010_0000_0000_0000 }, // 0.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4028_0000_0000_0000 }, // 1.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4030_0000_0000_0000 }, // 2.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC034_0000_0000_0000 }, // -2.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4039_21FB_5444_2D18 }, // pi in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // +0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xBFF0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // +inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // qNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF4_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // sNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // min denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE8_0000_0000_0000 }, // max denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE8_0000_0000_0000 }, // max normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FE0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 0.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 2.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xC004_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -2.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4009_21FB_5444_2D18), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // pi in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xC028_0000_0000_0000 }, // neg src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4028_0000_0000_0000 }, // abs src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // neg src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x4028_0000_0000_0000 }, // abs src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4038_0000_0000_0000 }, // omod 1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4048_0000_0000_0000 }, // omod 2
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0003), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x4018_0000_0000_0000 }, // omod 3
        ],
    );
}

#[test]
fn v_trig_preop_f64_vop3() {
    // V_TRIG_PREOP_F64.
    // src1 selects which 53-bit window of 2/pi is returned.
    check_vop3_f64(
        815,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // +0 in src0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // -0 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // 1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // -1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0B43_DD63_F5F2_F8BD }, // +inf in src0
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0B43_DD63_F5F2_F8BD }, // -inf in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0B43_DD63_F5F2_F8BD }, // qNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0B43_DD63_F5F2_F8BD }, // sNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // min denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // max denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // min normal in src0
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0B43_DD63_F5F2_F8BC }, // max normal in src0
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // 0.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // 1.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // 2.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // -2.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // pi in src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // +0 in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // -0 in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // 1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0xBFF0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // -1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // +inf in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // -inf in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // qNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x7FF4_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // sNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3C94_A7F0_9D5F_47D4 }, // min denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x000F_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // max denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // min normal in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // max normal in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x3FE0_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // 0.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // 1.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // 2.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0xC004_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // -2.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x4009_21FB_5444_2D18), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // pi in src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // neg src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // abs src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // neg src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // abs src1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3FE4_5F30_6DC9_C882 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x3FF4_5F30_6DC9_C882 }, // omod 1
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4004_5F30_6DC9_C882 }, // omod 2
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3FD4_5F30_6DC9_C882 }, // omod 3
        ],
    );
}

#[test]
fn v_add3_u32_vop3() {
    // V_ADD3_U32.
    check_vop3_u32(
        597,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0015 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0016 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0014 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0015 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0014 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0017 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_0014 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BF04 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0025 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0114 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0016 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0015 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001A }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0014 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0012 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0013 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0012 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0015 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_0012 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BF02 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0023 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0112 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0016 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0011 }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0014 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0013 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0008 }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0009 }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0007 }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0008 }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0007 }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000A }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_0007 }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEF7 }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0107 }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000B }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0006 }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0009 }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0008 }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000D }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x8000_0018 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x8000_0018 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x8000_0018 }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0018 }, // clamp
        ],
    );
}

#[test]
fn v_and_or_b32_vop3() {
    // V_AND_OR_B32.
    check_vop3_u32(
        599,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0001 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFFF }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0000 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFFF }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0002 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFFF }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_BEEF }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0010 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_00FF }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0003 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFFE }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0001 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0000 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0005 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0001 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEFF_BEEF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x80FF_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5EFF_BEEF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0002 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_BEEF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEFF_BEEF }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0000 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_00EF }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0003 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEFF_BEEE }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x80FF_0001 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40FF_0000 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0005 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_BEEF }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEFF }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEFF }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_BEEF }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_BEEF }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x00FF_BEEF }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x00FF_BEEF }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x80FF_BEEF }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x00FF_BEEF }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x80FF_BEEF }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x00FF_BEEF }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x00FF_BEEF }, // clamp
        ],
    );
}

#[test]
fn v_or3_b32_vop3() {
    // V_OR3_B32.
    check_vop3_u32(
        600,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF0 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF1 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0FF0 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF2 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BFFF }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF0 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FFF }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF3 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0FF1 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0FF0 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF5 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F0F }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F0F }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0F0F }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F0F }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BFEF }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F1F }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FFF }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F0F }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0F0F }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0F0F }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F0F }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_00FF }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEFF }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_00FF }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_00FF }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x8000_0FFF }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FFF }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x8000_0FFF }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FFF }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x8000_0FFF }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FFF }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00F0), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0FFF }, // clamp
        ],
    );
}

#[test]
fn v_xor3_b32_vop3() {
    // V_XOR3_B32.
    check_vop3_u32(
        576,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFFF }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFFE }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF00_0000 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x80FF_FFFF }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F00_0000 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFFD }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0000 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDE52_4110 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFEF }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FF00 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFFC }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF00_0001 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x80FF_FFFE }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40FF_FFFF }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_FFFA }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDE52_BEEF }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDE52_BEEE }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x21AD_4110 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5E52_BEEF }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xA1AD_4110 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDE52_BEED }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDE52_4110 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_0000 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDE52_BEFF }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDE52_BE10 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDE52_BEEC }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x21AD_4111 }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5E52_BEEE }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x9E52_BEEF }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDE52_BEEA }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4110 }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4111 }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_BEEF }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_4110 }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xA152_BEEF }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4112 }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4100 }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_41EF }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4113 }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_BEEE }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_4111 }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x9EAD_4110 }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4115 }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x5E52_4110 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x5E52_4110 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x5E52_4110 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xDE52_4110 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x5E52_4110 }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0xDE52_4110 }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x00FF_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xDE52_4110 }, // clamp
        ],
    );
}

#[test]
fn v_bfi_b32_vop3() {
    // V_BFI_B32.
    // src0 is the mask: the result takes those bits from src1 and the rest
    // from src2.
    check_vop3_u32(
        530,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_5678 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_5679 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x9234_5678 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEEF }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_567A }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_BEEF }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEBD_FEFF }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_5668 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_56EF }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_567B }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEE }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x9234_5679 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5234_5678 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_567D }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_0001 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_0002 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_FFFF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_BEEF }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_0010 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_00FF }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_0003 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_FFFE }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_0001 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_0000 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_0005 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_BEEF }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_BEEF }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_BEEF }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_BEEF }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_BEEF }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_BEEF }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x9234_BEEF }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x1234_BEEF }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x1234_BEEF }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x1234_BEEF }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x9234_BEEF }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x1234_BEEF }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x1234_5678), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x1234_BEEF }, // clamp
        ],
    );
}

#[test]
fn v_alignbit_b32_vop3() {
    // V_ALIGNBIT_B32.
    // src2 is a bit rotate amount; only its low five bits count.
    check_vop3_u32(
        534,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0012_3456 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0112_3456 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF12_3456 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0012_3456 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF12_3456 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0212_3456 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF12_3456 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF12_3456 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1012_3456 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF12_3456 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0312_3456 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFE12_3456 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0112_3456 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0012_3456 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0512_3456 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF00_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF00_0000 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF80_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF7F_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF00_0000 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF00_00FF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEFDE_ADBE }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF00_0000 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF00_0000 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF00_0000 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEFFF_FFFF }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF80_0000 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF40_0000 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEF00_0000 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_5678 }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x891A_2B3C }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBD5B_7DDE }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_5678 }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBD5B_7DDE }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC48D_159E }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBD5B_7DDE }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7DDE_2468 }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBEEF_1234 }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBD5B_7DDE }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xE246_8ACF }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7AB6_FBBC }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x891A_2B3C }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1234_5678 }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7891_A2B3 }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xEF12_3456 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0xEF12_3456 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xEF92_3456 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xEF12_3456 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0xEF12_3456 }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0xEF12_3456 }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xEF12_3456 }, // clamp
        ],
    );
}

#[test]
fn v_mad_u32_u24_vop3() {
    // V_MAD_U32_U24.
    // The multiplied sources are 24 bits wide; the high byte is ignored.
    check_vop3_u32(
        523,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0015 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0500_000B }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0500_000B }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001A }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0005_000B }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0364_BABB }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0060 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_050B }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0500_0006 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0015 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0029 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0300_000D }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0300_000D }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0016 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0003_000D }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0209_3CDD }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_030D }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0019 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0300_000A }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000F }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000E }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_000F }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_000E }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0011 }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_000E }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEFE }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_010E }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0012 }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000D }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0010 }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_000F }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0014 }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_001F }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_001F }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x8000_001F }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_001F }, // clamp
        ],
    );
}

#[test]
fn v_lshl_add_u32_vop3() {
    // V_LSHL_ADD_U32.
    check_vop3_u32(
        582,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0020 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0010_0000 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEADB_EF00 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0110 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_1000 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFF0 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0020 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0060 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0016 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0010 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0010 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001C }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0010 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_8010 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0003_0010 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0010 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0028 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0010 }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0016 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0070 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0031 }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_002F }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0030 }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_002F }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0032 }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_002F }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BF1F }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_012F }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0033 }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_002E }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0031 }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0030 }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0035 }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0040 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_0040 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x8000_0040 }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0040 }, // clamp
        ],
    );
}

#[test]
fn v_add_lshl_u32_vop3() {
    // V_ADD_LSHL_U32.
    check_vop3_u32(
        583,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0050 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0060 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0050 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0070 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0010_0040 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEADB_EF40 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0150 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_1040 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0080 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0060 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0050 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00A0 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0020 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0020 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0050 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0010_0020 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEADB_EF20 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0130 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_1020 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0060 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0080 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0008 }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0008 }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0020 }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0004_0000 }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0008_0000 }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0040 }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0008 }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0100 }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0080 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0080 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_0080 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0080 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x0000_0080 }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x0000_0080 }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0004), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0080 }, // clamp
        ],
    );
}

#[test]
fn v_lshl_or_b32_vop3() {
    // V_LSHL_OR_B32.
    check_vop3_u32(
        598,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F00 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F10 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFF0 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F00 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFF0 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F20 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x000F_FFF0 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEADB_EFF0 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F00 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF0 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F30 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFE0 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F10 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F00 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F50 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F0F }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F1E }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0F00 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F0F }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0F00 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F3C }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0F00 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0007_8F00 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x000F_0F00 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0F00 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F78 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0F00 }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F1E }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0F0F }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FE0 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00F0 }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00F1 }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_00F0 }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00F2 }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEFF }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00F0 }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00F3 }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_00F1 }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_00F0 }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00F5 }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0FF0 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF0 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_0FF0 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF0 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x8000_0FF0 }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF0 }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0x0000_000F), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0F00), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0FF0 }, // clamp
        ],
    );
}

#[test]
fn v_xad_u32_vop3() {
    // V_XAD_U32.
    check_vop3_u32(
        581,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_000F }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_000E }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_0010 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8001_000F }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_0010 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_000D }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4120 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FF10 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_000C }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_0011 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8001_000E }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4001_000F }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_000A }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEFF }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEFE }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4120 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEFF }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xA152_4120 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEFD }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4120 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BF0F }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BE20 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEFC }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4121 }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEFE }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x9EAD_BEFF }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEFA }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4110 }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4111 }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_410F }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_4110 }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_410F }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4112 }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAE_410F }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBD5A_FFFF }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4120 }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_420F }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4113 }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_410E }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_4111 }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1EAD_4110 }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4115 }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x5EAD_4120 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_4120 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x5EAD_4120 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4120 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x5EAD_4120 }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_4120 }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xDEAD_4120 }, // clamp
        ],
    );
}

#[test]
fn v_bfe_u32_vop3() {
    // V_BFE_U32.
    // src1 is the first bit and src2 the width; both count only their low five
    // bits.
    check_vop3_u32(
        528,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00EE }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000F }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00EF }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0077 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00EF }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00BB }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_005B }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00AD }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00DD }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0077 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00EF }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00F7 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0xFFFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0DEA_DBEE }, // -1 / UINT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x8000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x7FFF_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0DEA_DBEE }, // INT_MAX in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0002), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_FFFF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0DEA_DBEE }, // 0xFFFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0xDEAD_BEEF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_5BEE }, // 0xDEADBEEF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0010), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_DBEE }, // 16 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_00FF), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0DEA_DBEE }, // 0xFF in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0003), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0006 }, // 3 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0xFFFF_FFFE), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0DEA_DBEE }, // -2 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x8000_0001), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN + 1 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x4000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0x40000000 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0005), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000E }, // 5 in src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_00EE }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_00EE }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_00EE }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_00EE }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 4, clamp: false, omod: 0, expected: 0x0000_00EE }, // neg src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 4, neg: 0, clamp: false, omod: 0, expected: 0x0000_00EE }, // abs src2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0004), src2: Src::Vgpr(0x0000_0008), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_00EE }, // clamp
        ],
    );
}

#[test]
fn v_mul_lo_u32_vop3() {
    // V_MUL_LO_U32.
    check_vop3_u32(
        812,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFD }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFD }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0006 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0002_FFFD }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x9C09_3CCD }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_02FD }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0009 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFA }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0003 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000F }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_0001 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFE_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFE_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0002_0002 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x9D9C_BEEF }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0010_0010 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00FF_00FF }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0003_0003 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFD_FFFE }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8001_0001 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0005_0005 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x8003_0003 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0003_0003 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x8003_0003 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0003_0003 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0001_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0003_0003 }, // clamp
        ],
    );
}

#[test]
fn v_mul_hi_u32_vop3() {
    // V_MUL_HI_U32.
    check_vop3_u32(
        813,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FFF_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_7FFF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x6F56_DF77 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0008 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_007F }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2000_0000 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0000 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x4000_0001 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0001 }, // clamp
        ],
    );
}

#[test]
fn v_lshlrev_b16_vop3() {
    // V_LSHLREV_B16.
    // src0 is the shift amount; the high half of the destination is preserved.
    check_vop3_u32(
        824,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_7DDE }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_8000 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_8000 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FBBC }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_8000 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_8000 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_8000 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_F778 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_C000 }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_7DDE }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_DDE0 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFF0 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFF0 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0020 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFF0 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_EEF0 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0100 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0FF0 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFE0 }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0050 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_EEF0 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_EEF0 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_EEF0 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_EEF0 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0004), src1: Src::Vgpr(0x0000_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_EEF0 }, // clamp
        ],
    );
}

#[test]
fn v_add_nc_u16_vop3() {
    // V_ADD_NC_U16.
    // A 16-bit add; the high half of the destination is preserved.
    check_vop3_u32(
        771,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4321 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4322 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4320 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4321 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4320 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4323 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4320 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0210 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4331 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4420 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4324 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_431F }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4322 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4321 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4326 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEF0 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEE }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEE }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEF1 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEE }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_7DDE }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEFF }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BFEE }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEF2 }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEED }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEF0 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEEF }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BEF4 }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_8210 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_8210 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_8210 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0210 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0x0000_BEEF), src1: Src::Vgpr(0x0000_4321), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_FFFF }, // clamp
        ],
    );
}

#[test]
fn v_bcnt_u32_b32_vop3() {
    // V_BCNT_U32_B32.
    // The population count of src0, added to src1.
    check_vop3_u32(
        798,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0020 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0008 }, // 0xFF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 3 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFE), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // -2 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0001), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // INT_MIN + 1 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 0x40000000 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0005), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 5 in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0019 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0017 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0018 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0017 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001A }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_0017 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BF07 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0028 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0117 }, // 0xFF in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001B }, // 3 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0xFFFF_FFFE), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0016 }, // -2 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x8000_0001), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0019 }, // INT_MIN + 1 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0018 }, // 0x40000000 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0005), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001D }, // 5 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0017 }, // neg src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0017 }, // abs src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x8000_0018 }, // neg src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // abs src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x0000_0000), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0018 }, // clamp
        ],
    );
}
