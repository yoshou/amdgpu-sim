//! The VOP3 encoding of the instructions that read one source.

use super::*;
use crate::encoding::Src;

#[test]
pub(crate) fn v_bfrev_b32_vop3() {
    // V_BFREV_B32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_u32(
        440,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // 1
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // INT_MIN
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // INT_MAX
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_0000 }, // 0xFFFF
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xF77D_B57B }, // 0xDEADBEEF
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0800_0000 }, // 16
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF00_0000 }, // 255
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0xF77D_B57A }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xF77D_B57A }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xF77D_B57B }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xF77D_B57B }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xF77D_B57B }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xF77D_B57B }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xF77D_B57B }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xF77D_B57B }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xF77D_B57B }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_ceil_f32_vop3() {
    // V_CEIL_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        418,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4040_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC000_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC080_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC100_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBF80_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_clz_i32_u32_vop3() {
    // V_CLZ_I32_U32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_u32(
        441,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // 0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001F }, // 1
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1 / UINT_MAX
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // INT_MIN
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // INT_MAX
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001E }, // 2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 0xFFFF
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0xDEADBEEF
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_001B }, // 16
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0018 }, // 255
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0001 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cos_f32_vop3() {
    // V_COS_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        438,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F21_32CB }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xBF80_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xBF80_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC000_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC080_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBF00_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f16_f32_vop3() {
    // V_CVT_F16_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_u32(
        394,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_8000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_3C00 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_BC00 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_7C00 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FC00 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_7E00 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_7F00 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_8000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_7C00 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_3800 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_3E00 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_C100 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_4248 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_4100 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_4100 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x0000_C100 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_C500 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_C900 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_BD00 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_C100 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_C000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_C100 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f32_f16_vop3() {
    // V_CVT_F32_F16 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        395,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x0000_8000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x0000_3C00), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0x0000_BC00), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x0000_7C00), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0x0000_FC00), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x0000_7E00), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_7D00), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3380_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x0000_7BFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x477F_E000 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x0000_4000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0x0000_3800), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x0000_C000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0x0000_C000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4000_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0x0000_C000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC000_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0x0000_C000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0x0000_C000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC080_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0x0000_C000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC100_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0x0000_C000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBF80_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0x0000_C000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0x0000_C000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f32_f64_vop3() {
    // V_CVT_F32_F64 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        399,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC0_0000 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4049_0FDB }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4020_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4020_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC020_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC0A0_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC120_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFA0_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC020_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f32_i32_vop3() {
    // V_CVT_F32_I32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        389,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1 / UINT_MAX
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xCF00_0000 }, // INT_MIN
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4F00_0000 }, // INT_MAX
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x477F_FF00 }, // 0xFFFF
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xCE05_4904 }, // 0xDEADBEEF
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4180_0000 }, // 16
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x437F_0000 }, // 255
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4EBD_5B7E }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4EBD_5B7E }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xCE05_4904 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xCE85_4904 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xCF05_4904 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xCD85_4904 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xCE05_4904 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xCE05_4904 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f32_u32_vop3() {
    // V_CVT_F32_U32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        390,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4F80_0000 }, // -1 / UINT_MAX
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4F00_0000 }, // INT_MIN
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4F00_0000 }, // INT_MAX
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x477F_FF00 }, // 0xFFFF
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4F5E_ADBF }, // 0xDEADBEEF
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4180_0000 }, // 16
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x437F_0000 }, // 255
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4EBD_5B7E }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4EBD_5B7E }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x4F5E_ADBF }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3F80_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x4FDE_ADBF }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x505E_ADBF }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x4EDE_ADBF }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4F5E_ADBF }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4F80_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4F5E_ADBF }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f64_f32_vop3() {
    // V_CVT_F64_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f64(
        400,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0
            Vop3F64 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.0
            Vop3F64 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -1.0
            Vop3F64 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf
            Vop3F64 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf
            Vop3F64 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN
            Vop3F64 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN
            Vop3F64 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x36A0_0000_0000_0000 }, // min denorm
            Vop3F64 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xB80F_FFFF_C000_0000 }, // max -denorm
            Vop3F64 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3810_0000_0000_0000 }, // min normal
            Vop3F64 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x47EF_FFFF_E000_0000 }, // max normal
            Vop3F64 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 0.5
            Vop3F64 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF8_0000_0000_0000 }, // 1.5
            Vop3F64 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 2.0
            Vop3F64 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // -2.5
            Vop3F64 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4009_21FB_6000_0000 }, // pi
            Vop3F64 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4004_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4004_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC014_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC024_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFF4_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // src0 an inline constant
            Vop3F64 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC004_0000_0000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f64_i32_vop3() {
    // V_CVT_F64_I32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f64(
        388,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 0
            Vop3F64 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1
            Vop3F64 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -1 / UINT_MAX
            Vop3F64 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC1E0_0000_0000_0000 }, // INT_MIN
            Vop3F64 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x41DF_FFFF_FFC0_0000 }, // INT_MAX
            Vop3F64 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 2
            Vop3F64 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40EF_FFE0_0000_0000 }, // 0xFFFF
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC1C0_A920_8880_0000 }, // 0xDEADBEEF
            Vop3F64 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4030_0000_0000_0000 }, // 16
            Vop3F64 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x406F_E000_0000_0000 }, // 255
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x41D7_AB6F_BBC0_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x41D7_AB6F_BBC0_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC1C0_A920_8880_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC1D0_A920_8880_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC1E0_A920_8880_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xC1B0_A920_8880_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC1C0_A920_8880_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(193), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // src0 an inline constant
            Vop3F64 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC1C0_A920_8880_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_f64_u32_vop3() {
    // V_CVT_F64_U32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f64(
        406,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 0
            Vop3F64 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1
            Vop3F64 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x41EF_FFFF_FFE0_0000 }, // -1 / UINT_MAX
            Vop3F64 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x41E0_0000_0000_0000 }, // INT_MIN
            Vop3F64 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x41DF_FFFF_FFC0_0000 }, // INT_MAX
            Vop3F64 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 2
            Vop3F64 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x40EF_FFE0_0000_0000 }, // 0xFFFF
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x41EB_D5B7_DDE0_0000 }, // 0xDEADBEEF
            Vop3F64 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4030_0000_0000_0000 }, // 16
            Vop3F64 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x406F_E000_0000_0000 }, // 255
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x41D7_AB6F_BBC0_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x41D7_AB6F_BBC0_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x41EB_D5B7_DDE0_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x41FB_D5B7_DDE0_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x420B_D5B7_DDE0_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x41DB_D5B7_DDE0_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x41EB_D5B7_DDE0_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(193), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x41EF_FFFF_FFE0_0000 }, // src0 an inline constant
            Vop3F64 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x41EB_D5B7_DDE0_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_i32_f32_vop3() {
    // V_CVT_I32_F32 in the VOP3 encoding. ISA: "1ULP accuracy".
    check_vop3_u32_ulp(
        392,
        1,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0002 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xFFFF_FFFE }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xFFFF_FFFE }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xFFFF_FFFE }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xFFFF_FFFE }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_i32_f64_vop3() {
    // V_CVT_I32_F64 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_u32(
        387,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0002 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xFFFF_FFFE }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xFFFF_FFFE }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xFFFF_FFFE }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xFFFF_FFFE }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // src0 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_u32_f32_vop3() {
    // V_CVT_U32_F32 in the VOP3 encoding. ISA: "1ULP accuracy".
    check_vop3_u32_ulp(
        391,
        1,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0002 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_cvt_u32_f64_vop3() {
    // V_CVT_U32_F64 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_u32(
        405,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0003 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0002 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_exp_f32_vop3() {
    // V_EXP_F32 in the VOP3 encoding. ISA: "1ULP accuracy, denormals are flushed".
    check_vop3_f32_ulp(
        421,
        1,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FB5_04F3 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4035_04F3 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4080_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3E35_04F3 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x410D_331C }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x40B5_04F3 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x40B5_04F3 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x3E35_04F3 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3E35_04F3 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x3EB5_04F3 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x3F35_04F3 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3DB5_04F3 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3E35_04F3 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3E80_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3E35_04F3 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_floor_f32_vop3() {
    // V_FLOOR_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        420,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC040_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4000_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC040_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC0C0_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC140_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFC0_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC040_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC040_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_floor_f64_vop3() {
    // V_FLOOR_F64 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f64(
        410,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -1.0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // min denorm
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // max -denorm
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // min normal
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 0.5
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.5
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 2.0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC008_0000_0000_0000 }, // -2.5
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // pi
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC008_0000_0000_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC018_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC028_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFF8_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC008_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_fract_f64_vop3() {
    // V_FRACT_F64 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f64(
        446,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // -0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 1.0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // -1.0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // +inf
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // -inf
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0001 }, // min denorm
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FEF_FFFF_FFFF_FFFF }, // max -denorm
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0010_0000_0000_0000 }, // min normal
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // max normal
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 0.5
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 1.5
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 2.0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // -2.5
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FC2_1FB5_4442_D180 }, // pi
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x3FF0_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x4000_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x3FD0_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
#[ignore = "faults during execution: takes the test process down with SIGSEGV rather than reporting a wrong value"]
pub(crate) fn v_frexp_exp_i32_f32_vop3() {
    // V_FREXP_EXP_I32_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_u32(
        447,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FF6C }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FF82 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FF83 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0080 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0002 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x0000_0002 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0002 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0002 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0002 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0002 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // src0 a literal constant
        ],
    );
}

#[test]
#[ignore = "faults during execution: takes the test process down with SIGSEGV rather than reporting a wrong value"]
pub(crate) fn v_frexp_exp_i32_f64_vop3() {
    // V_FREXP_EXP_I32_F64 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_u32(
        444,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FBCF }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FC02 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FC03 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0400 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0002 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x0000_0002 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0002 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0002 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0002 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0002 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // src0 an inline constant
        ],
    );
}

#[test]
#[ignore = "the JIT faults on this one: it takes the test process down with SIGSEGV rather than reporting a wrong value"]
pub(crate) fn v_frexp_mant_f32_vop3() {
    // V_FREXP_MANT_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        448,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF7F_FFFE }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F7F_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F40_0000 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF20_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F49_0FDB }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3F20_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3F20_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xBF20_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xBFA0_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC020_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBEA0_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF20_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF20_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
#[ignore = "faults during execution: takes the test process down with SIGSEGV rather than reporting a wrong value"]
pub(crate) fn v_frexp_mant_f64_vop3() {
    // V_FREXP_MANT_F64 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f64(
        445,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 1.0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFE0_0000_0000_0000 }, // -1.0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // min denorm
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFEF_FFFF_FFFF_FFFE }, // max -denorm
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // min normal
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FEF_FFFF_FFFF_FFFF }, // max normal
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 0.5
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE8_0000_0000_0000 }, // 1.5
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 2.0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFE4_0000_0000_0000 }, // -2.5
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE9_21FB_5444_2D18 }, // pi
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FE4_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xBFE4_0000_0000_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xBFF4_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC004_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFD4_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFE4_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFE0_0000_0000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_log_f32_vop3() {
    // V_LOG_F32 in the VOP3 encoding. ISA: "1ULP accuracy, denormals are flushed".
    check_vop3_f32_ulp(
        423,
        1,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC2FC_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x42FF_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F15_C01A }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FD3_643A }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FA9_34F0 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FA9_34F0 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xFFC0_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xFFC0_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xFFC0_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_mov_b32_vop3() {
    // V_MOV_B32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_u32(
        385,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0001 }, // 1
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // INT_MIN
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MAX
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0002 }, // 2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_FFFF }, // 0xFFFF
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // 0xDEADBEEF
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0010 }, // 16
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_00FF }, // 255
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x5EAD_BEEF }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x5EAD_BEEF }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0xDEAD_BEEF }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xDEAD_BEEF }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xDEAD_BEEF }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xDEAD_BEEF }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xDEAD_BEEF }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_not_b32_vop3() {
    // V_NOT_B32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_u32(
        439,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFF }, // 0
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFE }, // 1
            Vop3F32 { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1 / UINT_MAX
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFF_FFFF }, // INT_MIN
            Vop3F32 { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // INT_MAX
            Vop3F32 { src0: Src::Vgpr(0x0000_0002), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFFD }, // 2
            Vop3F32 { src0: Src::Vgpr(0x0000_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_0000 }, // 0xFFFF
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4110 }, // 0xDEADBEEF
            Vop3F32 { src0: Src::Vgpr(0x0000_0010), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FFEF }, // 16
            Vop3F32 { src0: Src::Vgpr(0x0000_00FF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFFF_FF00 }, // 255
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0xA152_4110 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0xA152_4110 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x2152_4110 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x2152_4110 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x2152_4110 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x2152_4110 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x2152_4110 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4110 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(193), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xDEAD_BEEF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2152_4110 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_rcp_f32_vop3() {
    // V_RCP_F32 in the VOP3 encoding. ISA: "1ULP accuracy ... Denormals are flushed".
    check_vop3_f32_ulp(
        426,
        1,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7E80_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F2A_AAAA }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBECC_CCCD }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3EA2_F983 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3ECC_CCCD }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3ECC_CCCD }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xBECC_CCCD }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xBF4C_CCCD }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xBFCC_CCCD }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBE4C_CCCD }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBECC_CCCD }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBECC_CCCD }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_rcp_f64_vop3() {
    // V_RCP_F64 in the VOP3 encoding. ISA: "(2**29)ULP accuracy".
    check_vop3_f64_ulp(
        431,
        1 << 29,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -1.0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +inf
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -inf
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // min denorm
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFD0_0000_0AF8_0000 }, // max -denorm
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FD0_0000_0000_0000 }, // min normal
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0004_0000_0000_0001 }, // max normal
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 0.5
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE5_5555_5400_0000 }, // 1.5
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE0_0000_0000_0000 }, // 2.0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFD9_9999_9C00_0000 }, // -2.5
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FD4_5F30_6C40_0000 }, // pi
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FD9_9999_9C00_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FD9_9999_9C00_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xBFD9_9999_9C00_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xBFE9_9999_9C00_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xBFF9_9999_9C00_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFC9_9999_9C00_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFD9_9999_9C00_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFE0_0000_0000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_rcp_iflag_f32_vop3() {
    // V_RCP_IFLAG_F32 in the VOP3 encoding. measured: 1 ULP, as for V_RCP_F32.
    check_vop3_f32_ulp(
        427,
        1,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7E80_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F2A_AAAA }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F00_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBECC_CCCD }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3EA2_F983 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3ECC_CCCD }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3ECC_CCCD }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xBECC_CCCD }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xBF4C_CCCD }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xBFCC_CCCD }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBE4C_CCCD }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBECC_CCCD }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF00_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBECC_CCCD }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_rndne_f32_vop3() {
    // V_RNDNE_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        419,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4000_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC000_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC080_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC100_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBF80_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_rndne_f64_vop3() {
    // V_RNDNE_F64 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f64(
        409,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -1.0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // min denorm
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // max -denorm
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // min normal
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 0.5
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 1.5
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 2.0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // -2.5
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // pi
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC010_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC020_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFF0_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_rsq_f32_vop3() {
    // V_RSQ_F32 in the VOP3 encoding. ISA: "1ULP accuracy, denormals are flushed".
    check_vop3_f32_ulp(
        430,
        1,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5F00_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1F80_0000 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FB5_04F3 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F51_05EC }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F35_04F3 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F10_6EBA }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3F21_E89B }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3F21_E89B }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xFFC0_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xFFC0_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xFFC0_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_rsq_f64_vop3() {
    // V_RSQ_F64 in the VOP3 encoding. ISA: "(2**29)ULP accuracy".
    check_vop3_f64_ulp(
        433,
        1 << 29,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // -1.0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +inf
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // -inf
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x6180_0000_0000_0000 }, // min denorm
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // max -denorm
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5FE0_0000_0000_0000 }, // min normal
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1FF0_0000_019E_0000 }, // max normal
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF6_A09E_6000_0000 }, // 0.5
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FEA_20BD_7400_0000 }, // 1.5
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE6_A09E_6000_0000 }, // 2.0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // -2.5
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE2_0DD7_53BE_0000 }, // pi
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FE4_3D13_6400_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FE4_3D13_6400_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xFFF8_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xFFF8_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xFFF8_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_sin_f32_vop3() {
    // V_SIN_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        437,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0006 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x81C9_0FD3 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x01C9_0FD5 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F46_DFE0 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x0000_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0x0000_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0x0000_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0x0000_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0x0000_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_sqrt_f32_vop3() {
    // V_SQRT_F32 in the VOP3 encoding. ISA: "1ULP accuracy, denormals are flushed".
    check_vop3_f32_ulp(
        435,
        1,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5F7F_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F35_04F3 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F9C_C470 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FB5_04F3 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE2_DFC5 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FCA_62C2 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FCA_62C2 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xFFC0_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xFFC0_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xFFC0_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFC0_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_sqrt_f64_vop3() {
    // V_SQRT_F64 in the VOP3 encoding. ISA: "(2**29)ULP accuracy".
    check_vop3_f64_ulp(
        436,
        1 << 29,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0400_0000 }, // 1.0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // -1.0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // -inf
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x1E60_0000_0400_0000 }, // min denorm
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // max -denorm
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x2000_0000_0400_0000 }, // min normal
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x5FEF_FFFF_FC08_0000 }, // max normal
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FE6_A09E_6400_0000 }, // 0.5
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF3_988E_1400_0000 }, // 1.5
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF6_A09E_6400_0000 }, // 2.0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // -2.5
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FFC_5BF8_9518_0000 }, // pi
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x3FF9_4C58_3C00_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x3FF9_4C58_3C00_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xFFF8_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xFFF8_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xFFF8_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF8_0000_0000_0000 }, // src0 an inline constant
        ],
    );
}

#[test]
pub(crate) fn v_trunc_f32_vop3() {
    // V_TRUNC_F32 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f32(
        417,
        &[
            Vop3F32 { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // +0
            Vop3F32 { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // -0
            Vop3F32 { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.0
            Vop3F32 { src0: Src::Vgpr(0xBF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBF80_0000 }, // -1.0
            Vop3F32 { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F80_0000 }, // +inf
            Vop3F32 { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFF80_0000 }, // -inf
            Vop3F32 { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FC0_0000 }, // qNaN
            Vop3F32 { src0: Src::Vgpr(0x7FA0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FE0_0000 }, // sNaN
            Vop3F32 { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min denorm
            Vop3F32 { src0: Src::Vgpr(0x807F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000 }, // max -denorm
            Vop3F32 { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // min normal
            Vop3F32 { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7F7F_FFFF }, // max normal
            Vop3F32 { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000 }, // 0.5
            Vop3F32 { src0: Src::Vgpr(0x3FC0_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3F80_0000 }, // 1.5
            Vop3F32 { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // 2.0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // -2.5
            Vop3F32 { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4040_0000 }, // pi
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000 }, // abs on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4000_0000 }, // neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC000_0000 }, // abs then neg on src0
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000 }, // clamp
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC080_0000 }, // omod x2
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC100_0000 }, // omod x4
            Vop3F32 { src0: Src::Vgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBF80_0000 }, // omod /2
            Vop3F32 { src0: Src::Sgpr(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 from an SGPR
            Vop3F32 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 an inline constant
            Vop3F32 { src0: Src::Literal(0xC020_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000 }, // src0 a literal constant
        ],
    );
}

#[test]
pub(crate) fn v_trunc_f64_vop3() {
    // V_TRUNC_F64 in the VOP3 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop3_f64(
        407,
        &[
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // +0
            Vop3F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // -0
            Vop3F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.0
            Vop3F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xBFF0_0000_0000_0000 }, // -1.0
            Vop3F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF0_0000_0000_0000 }, // +inf
            Vop3F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xFFF0_0000_0000_0000 }, // -inf
            Vop3F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FF8_0000_0000_0000 }, // qNaN
            Vop3F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FFC_0000_0000_0000 }, // sNaN
            Vop3F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // min denorm
            Vop3F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x8000_0000_0000_0000 }, // max -denorm
            Vop3F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // min normal
            Vop3F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal
            Vop3F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x0000_0000_0000_0000 }, // 0.5
            Vop3F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x3FF0_0000_0000_0000 }, // 1.5
            Vop3F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // 2.0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // -2.5
            Vop3F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0x4008_0000_0000_0000 }, // pi
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 0, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // abs on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 1, clamp: false, omod: 0, expected: 0x4000_0000_0000_0000 }, // neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 1, neg: 1, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // abs then neg on src0
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: true, omod: 0, expected: 0x0000_0000_0000_0000 }, // clamp
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 1, expected: 0xC010_0000_0000_0000 }, // omod x2
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 2, expected: 0xC020_0000_0000_0000 }, // omod x4
            Vop3F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 3, expected: 0xBFF0_0000_0000_0000 }, // omod /2
            Vop3F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // src0 from an SGPR
            Vop3F64 { src0: Src::Inline(245), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0, neg: 0, clamp: false, omod: 0, expected: 0xC000_0000_0000_0000 }, // src0 an inline constant
        ],
    );
}
