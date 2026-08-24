//! The VOP3 encoding of the instructions that read two sources.

use super::*;
use crate::encoding::Src;

#[test]
pub(crate) fn v_add_f32_vop3() {
    // V_ADD_F32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f32(
        259,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4020_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4060_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4094_87EE }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4020_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4060_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4094_87EE }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4080_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xC080_0000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xC080_0000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC000_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC080_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBF00_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Sgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC090_0000 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_add_f64_vop3() {
    // V_ADD_F64 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f64(
        258,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // +0 in src0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -0 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4004_0000_0000_0000 }, // 1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // -1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src0
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max -denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min normal in src0
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src0
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 0.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // 1.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x400C_0000_0000_0000 }, // 2.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -2.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4012_90FD_AA22_168C }, // pi in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // +0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4004_0000_0000_0000 }, // 1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xBFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // -1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF4_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max -denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FE0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 0.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // 1.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x400C_0000_0000_0000 }, // 2.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xC004_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -2.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4009_21FB_5444_2D18), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4012_90FD_AA22_168C }, // pi in src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4010_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // abs on src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4010_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xC010_0000_0000_0000 }, // neg on src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xC010_0000_0000_0000 }, // abs then neg on both
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC000_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC010_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFE0_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Sgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // src1 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFE0_0000_0000_0000 }, // src0 an inline constant
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC012_0000_0000_0000 }, // src1 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_add_nc_u32_vop3() {
    // V_ADD_NC_U32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        293,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0004 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0003 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0002 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0005 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_0002 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEF2 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0102 }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0004 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0003 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0002 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0005 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_0002 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEF2 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0102 }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEF2 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEF2 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x5EAD_BEF2 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x5EAD_BEF2 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x5EAD_BEF2 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xDEAD_BEF2 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xDEAD_BEF2 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xDEAD_BEF2 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xDEAD_BEF2 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEF2 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEF2 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEE }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEF2 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_and_b32_vop3() {
    // V_AND_B32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        283,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0003 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x8000_0003 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x8000_0003 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0003 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0003 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0003 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0003 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_ashrrev_i32_vop3() {
    // V_ASHRREV_I32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        282,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xF000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0FFF_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_1FFF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFBD5_B7DD }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xFFFF_0000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xFFFF_0000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_fmac_f32_vop3() {
    // V_FMAC_F32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f32(
        299,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x80BF_FFFE }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00C0_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F40_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4010_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4096_CBE4 }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x80BF_FFFE }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00C0_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F40_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4010_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4096_CBE4 }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4070_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4070_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x4070_0000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x4070_0000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC0F0_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC170_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFF0_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Sgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC040_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40A0_0000 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_lshlrev_b32_vop3() {
    // V_LSHLREV_B32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        280,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0006 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000C }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_8000 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0003_0000 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0008 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFF8 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFF8 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0007_FFF8 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xF56D_F778 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0080 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_07F8 }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0001_8000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0001_8000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0001_8000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0001_8000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x0001_8000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0001_8000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0001_8000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0001_8000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0001_8000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_8000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_8000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_8000 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_8000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_lshlrev_b64_vop3() {
    // V_LSHLREV_B64 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f64(
        287,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 0 in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // 1 in src0
            Vop3F64 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // -1 / UINT_MAX in src0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // INT_MIN in src0
            Vop3F64 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // INT_MAX in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFE0_0000_0000_0000 }, // 2 in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 0xFFFF in src0
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 0xDEADBEEF in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 16 in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 255 in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0 in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // -0 in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000_0000_0000 }, // 1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xBFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000_0000_0000 }, // -1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000_0000_0000 }, // +inf in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000_0000_0000 }, // -inf in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000_0000_0000 }, // qNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FF4_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFA0_0000_0000_0000 }, // sNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0008 }, // min denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x007F_FFFF_FFFF_FFF8 }, // max -denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0080_0000_0000_0000 }, // min normal in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF7F_FFFF_FFFF_FFF8 }, // max normal in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x3FE0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF00_0000_0000_0000 }, // 0.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000_0000_0000 }, // 1.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 2.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xC004_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0020_0000_0000_0000 }, // -2.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x4009_21FB_5444_2D18), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0049_0FDA_A221_68C0 }, // pi in src1
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // abs on src1
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // neg on src1
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // abs then neg on both
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // src1 from an SGPR
            Vop3F64 { src0: Src::Inline(193), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // src0 an inline constant
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // src1 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_lshrrev_b32_vop3() {
    // V_LSHRREV_B32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        281,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1FFF_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0FFF_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_1FFF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1BD5_B7DD }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0001_0000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x0001_0000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0001_FFFF }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_max_i32_vop3() {
    // V_MAX_I32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        274,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEEF }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x5EAD_BEEF }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0003 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0003 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0003 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0003 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_max_num_f32_vop3() {
    // V_MAX_NUM_F32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f32(
        278,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4049_0FDB }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4049_0FDB }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4020_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4020_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F80_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4040_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x40C0_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3F40_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Sgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_max_num_f64_vop3() {
    // V_MAX_NUM_F64 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f64(
        270,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // +0 in src0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -0 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src0
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -inf in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // qNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // sNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max -denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min normal in src0
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src0
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 0.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 2.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -2.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4009_21FB_5444_2D18 }, // pi in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // +0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xBFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // qNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF4_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // sNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max -denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // min normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FE0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 0.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 2.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xC004_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // -2.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4009_21FB_5444_2D18), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4009_21FB_5444_2D18 }, // pi in src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4004_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // abs on src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4004_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // neg on src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // abs then neg on both
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4008_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4018_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3FE8_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Sgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // src1 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // src0 an inline constant
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // src1 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_max_u32_vop3() {
    // V_MAX_U32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        276,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEEF }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x5EAD_BEEF }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xDEAD_BEEF }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xDEAD_BEEF }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xDEAD_BEEF }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xDEAD_BEEF }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_min_i32_vop3() {
    // V_MIN_I32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        273,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0003 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x8000_0003 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x8000_0003 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xDEAD_BEEF }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xDEAD_BEEF }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xDEAD_BEEF }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xDEAD_BEEF }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_min_num_f32_vop3() {
    // V_MIN_NUM_F32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f32(
        277,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x807F_FFFF }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0080_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x807F_FFFF }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0080_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xC020_0000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xC020_0000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC0A0_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC120_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFA0_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Sgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_min_num_f64_vop3() {
    // V_MIN_NUM_F64 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f64(
        269,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0 in src0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // +inf in src0
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // qNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // sNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0001 }, // min denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x800F_FFFF_FFFF_FFFF }, // max -denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0010_0000_0000_0000 }, // min normal in src0
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max normal in src0
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 0.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 2.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // -2.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // pi in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xBFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // +inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // qNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF4_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // sNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0001 }, // min denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x800F_FFFF_FFFF_FFFF }, // max -denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0010_0000_0000_0000 }, // min normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // max normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FE0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 0.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 2.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xC004_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // -2.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4009_21FB_5444_2D18), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // pi in src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // abs on src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // neg on src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // abs then neg on both
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC014_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC024_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFF4_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Sgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // src1 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // src0 an inline constant
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // src1 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_min_u32_vop3() {
    // V_MIN_U32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        275,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0003 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x8000_0003 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x8000_0003 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0003 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0003 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0003 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0003 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_mul_f32_vop3() {
    // V_MUL_F32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f32(
        264,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x80BF_FFFE }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00C0_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F40_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4010_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4096_CBE4 }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x80BF_FFFE }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x00C0_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F40_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4010_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4096_CBE4 }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4070_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4070_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x4070_0000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x4070_0000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC0F0_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC170_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFF0_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Sgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC040_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40A0_0000 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC070_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_mul_f64_vop3() {
    // V_MUL_F64 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f64(
        262,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0 in src0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // -1.0 in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src0
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src0
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0002 }, // min denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8017_FFFF_FFFF_FFFE }, // max -denorm in src0
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0018_0000_0000_0000 }, // min normal in src0
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // max normal in src0
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE8_0000_0000_0000 }, // 0.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4002_0000_0000_0000 }, // 1.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // 2.0 in src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC00E_0000_0000_0000 }, // -2.5 in src0
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4012_D97C_7F33_21D2 }, // pi in src0
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xBFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF8_0000_0000_0000 }, // -1.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF4_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0002 }, // min denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8017_FFFF_FFFF_FFFE }, // max -denorm in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0018_0000_0000_0000 }, // min normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // max normal in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FE0_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE8_0000_0000_0000 }, // 0.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4002_0000_0000_0000 }, // 1.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // 2.0 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0xC004_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC00E_0000_0000_0000 }, // -2.5 in src1
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0x4009_21FB_5444_2D18), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4012_D97C_7F33_21D2 }, // pi in src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x400E_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xC00E_0000_0000_0000 }, // abs on src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x400E_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x400E_0000_0000_0000 }, // neg on src1
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x400E_0000_0000_0000 }, // abs then neg on both
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC01E_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC02E_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFFE_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC00E_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Sgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC00E_0000_0000_0000 }, // src1 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FF8_0000_0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC008_0000_0000_0000 }, // src0 an inline constant
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4014_0000_0000_0000 }, // src1 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_mul_i32_i24_vop3() {
    // V_MUL_I32_I24 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        265,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFD }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFD }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0006 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0002_FFFD }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_02FD }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFD }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFD }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0006 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0002_FFFD }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_02FD }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xFF09_3CCD }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xFF09_3CCD }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xFF09_3CCD }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xFF09_3CCD }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFD }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0052_4111 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF09_3CCD }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_mul_u32_u24_vop3() {
    // V_MUL_U32_U24 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        267,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x02FF_FFFD }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x02FF_FFFD }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0006 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0002_FFFD }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0209_3CCD }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_02FD }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x02FF_FFFD }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x02FF_FFFD }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0006 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0002_FFFD }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0209_3CCD }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0030 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_02FD }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0209_3CCD }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x0209_3CCD }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0209_3CCD }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x0209_3CCD }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x0209_3CCD }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0209_3CCD }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0209_3CCD }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0209_3CCD }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0209_3CCD }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0209_3CCD }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0209_3CCD }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x02FF_FFFD }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xEE52_4111 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0209_3CCD }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_or_b32_vop3() {
    // V_OR_B32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        284,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0003 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0003 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEEF }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x5EAD_BEEF }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xDEAD_BEEF }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xDEAD_BEEF }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xDEAD_BEEF }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xDEAD_BEEF }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_subrev_f32_vop3() {
    // V_SUBREV_F32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f32(
        261,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4020_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFE0_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF7F_FFFF }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFD2_1FB6 }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC080_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FD2_1FB6 }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xBF80_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x3F80_0000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x3F80_0000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F80_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4100_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4180_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x4000_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Sgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4060_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_subrev_nc_u32_vop3() {
    // V_SUBREV_NC_U32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        295,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0004 }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0003 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0004 }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_0004 }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4114 }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFF3 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FF04 }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFD }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFC }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFD }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFC }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFC }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000D }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FC }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0xA152_4114 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0x2152_4114 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xA152_4114 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xA152_4114 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xA152_4114 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x2152_4114 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x2152_4114 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x2152_4114 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4114 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4114 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0004 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4110 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4114 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_sub_f32_vop3() {
    // V_SUB_F32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_f32(
        260,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // +0 in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // -0 in src0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // 1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // -1.0 in src0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf in src0
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf in src0
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // min denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // max -denorm in src0
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFC0_0000 }, // min normal in src0
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal in src0
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // 0.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 2.0 in src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC080_0000 }, // -2.5 in src0
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FD2_1FB6 }, // pi in src0
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // +0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // -0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xBF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4020_0000 }, // -1.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // +inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // -inf in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // qNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7FA0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFE0_0000 }, // sNaN in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x807F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // max -denorm in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // min normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF7F_FFFF }, // max normal in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 0.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // 2.0 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0xC020_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // -2.5 in src1
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFD2_1FB6 }, // pi in src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xC080_0000 }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3F80_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0xBF80_0000 }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0xBF80_0000 }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC100_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC180_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xC000_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC080_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Sgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC080_0000 }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC060_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Inline(245), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0x3FC0_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC080_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_sub_nc_u32_vop3() {
    // V_SUB_NC_U32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        294,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFD }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFC }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFD }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFC }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFC }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_000D }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FC }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0004 }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0003 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0004 }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_0004 }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4114 }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFF3 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FF04 }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEEC }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x5EAD_BEEC }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x5EAD_BEEC }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x5EAD_BEEC }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xDEAD_BEEC }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xDEAD_BEEC }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xDEAD_BEEC }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xDEAD_BEEC }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFC }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEF0 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_xor_b32_vop3() {
    // V_XOR_B32 in the VOP3 encoding, with every operand class and modifier the
    // format has.
    check_vop3_u32(
        285,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 1 in src0
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFC }, // -1 / UINT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0003 }, // INT_MIN in src0
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFC }, // INT_MAX in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 2 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFC }, // 0xFFFF in src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // 0xDEADBEEF in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 16 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FC }, // 255 in src0
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // 0 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 1 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFC }, // -1 / UINT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0003 }, // INT_MIN in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x7FFF_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFC }, // INT_MAX in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0002), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 2 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_FFFF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFC }, // 0xFFFF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0xDEAD_BEEF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // 0xDEADBEEF in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_0010), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0013 }, // 16 in src1
            Vop3F32 { src0: Src::Vgpr(0x0000_0003), src1: Src::Vgpr(0x0000_00FF), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FC }, // 255 in src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEEC }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 2, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // abs on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x5EAD_BEEC }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 2, clamp: false, omod: 0, expected: 0x5EAD_BEEC }, // neg on src1
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 3, neg: 3, clamp: false, omod: 0, expected: 0x5EAD_BEEC }, // abs then neg on both
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xDEAD_BEEC }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xDEAD_BEEC }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xDEAD_BEEC }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xDEAD_BEEC }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // src0 from an SGPR
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Sgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // src1 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFC }, // src0 an inline constant
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Inline(193), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4110 }, // src1 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0x0000_0003), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEC }, // src0 a literal constant
        ],
    );
}
