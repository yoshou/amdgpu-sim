//! The VOP3 encoding of the instructions that write a scalar
//! destination as well as a register.
//!
//! The carry a wave's lanes produce is a mask, so each case checks
//! the whole 32-bit destination the instruction wrote, not just the
//! bit belonging to the lane the register result is read from.

use super::*;
use crate::encoding::Src;

#[test]
fn v_add_co_u32_vop3sd() {
    // V_ADD_CO_U32.
    // The carry or borrow out lands in the scalar destination.
    check_vop3sd(
        768,
        &[
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // 0x00000000 and 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0x0000_0000 }, // 0x00000001 and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // 0xffffffff and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // 0x00000001 and 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // 0x80000000 and 0x80000000
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0xFFFF_FFFF }, // 0xffffffff and 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0x0000_0000 }, // 0x00000000 and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_8000_0000, expected_sdst: 0x0000_0000 }, // 0x7fffffff and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_F0E2_1567, expected_sdst: 0x0000_0000 }, // 0xdeadbeef and 0x12345678
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0000 }, // 0x00000000 and 0xffffffff
        ],
    );
}

#[test]
fn v_sub_co_u32_vop3sd() {
    // V_SUB_CO_U32.
    // The carry or borrow out lands in the scalar destination.
    check_vop3sd(
        769,
        &[
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // 0x00000000 and 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // 0x00000001 and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0x0000_0000 }, // 0xffffffff and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0xFFFF_FFFF }, // 0x00000001 and 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // 0x80000000 and 0x80000000
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // 0xffffffff and 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // 0x00000000 and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_7FFF_FFFE, expected_sdst: 0x0000_0000 }, // 0x7fffffff and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_CC79_6877, expected_sdst: 0x0000_0000 }, // 0xdeadbeef and 0x12345678
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // 0x00000000 and 0xffffffff
        ],
    );
}

#[test]
fn v_subrev_co_u32_vop3sd() {
    // V_SUBREV_CO_U32.
    // The carry or borrow out lands in the scalar destination.
    check_vop3sd(
        770,
        &[
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // 0x00000000 and 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // 0x00000001 and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0xFFFF_FFFF }, // 0xffffffff and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0x0000_0000 }, // 0x00000001 and 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // 0x80000000 and 0x80000000
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // 0xffffffff and 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0x0000_0000 }, // 0x00000000 and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x7FFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_8000_0002, expected_sdst: 0xFFFF_FFFF }, // 0x7fffffff and 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xDEAD_BEEF), src1: Src::Vgpr(0x1234_5678), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_3386_9789, expected_sdst: 0xFFFF_FFFF }, // 0xdeadbeef and 0x12345678
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Inline(128), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0000 }, // 0x00000000 and 0xffffffff
        ],
    );
}

#[test]
fn v_add_co_ci_u32_vop3sd() {
    // V_ADD_CO_CI_U32.
    // The third source is the carry in, one bit per lane.
    check_vop3sd(
        288,
        &[
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0x0000_0000 }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0x0000_0000 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_0000_0003, expected_sdst: 0x0000_0000 }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0x0000_0000 }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_0000_0003, expected_sdst: 0x0000_0000 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0xFFFF_FFFF }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000001
        ],
    );
}

#[test]
fn v_sub_co_ci_u32_vop3sd() {
    // V_SUB_CO_CI_U32.
    // The third source is the carry in, one bit per lane.
    check_vop3sd(
        289,
        &[
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xAAAA_AAAA }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0001 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xAAAA_AAAA }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0001 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFD, expected_sdst: 0x0000_0000 }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0x0000_0000 }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFD, expected_sdst: 0x0000_0000 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0xFFFF_FFFF }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xAAAA_AAAA }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0001 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xAAAA_AAAA }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0001 }, // carry in 0x00000001
        ],
    );
}

#[test]
fn v_subrev_co_ci_u32_vop3sd() {
    // V_SUBREV_CO_CI_U32.
    // The third source is the carry in, one bit per lane.
    check_vop3sd(
        290,
        &[
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xAAAA_AAAA }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0001 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xAAAA_AAAA }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0001 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0xFFFF_FFFF }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0x0000_0001), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0xFFFF_FFFF }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFD, expected_sdst: 0x0000_0000 }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0x0000_0000 }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFD, expected_sdst: 0x0000_0000 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xAAAA_AAAA }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0001 }, // carry in 0x00000001
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // carry in 0x00000000
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xFFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // carry in 0xffffffff
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0xAAAA_AAAA), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xAAAA_AAAA }, // carry in 0xaaaaaaaa
            Vop3sdCase { src0: Src::Vgpr(0xFFFF_FFFF), src1: Src::Vgpr(0xFFFF_FFFF), src2: Src::Sgpr(0x0000_0001), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0001 }, // carry in 0x00000001
        ],
    );
}

#[test]
fn v_mad_co_u64_u32_vop3sd() {
    // V_MAD_CO_U64_U32.
    // A 32-bit product added to a 64-bit source, carrying out of 64 bits.
    check_vop3sd(
        766,
        &[
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_0000_0001), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0xFFFF_FFFF_FFFF_FFFF), neg: 0, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_sdst: 0x0000_0000 }, // addend 0xffffffffffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x0000_0000_FFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0000 }, // addend 0x00000000ffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x0000_0000_0000_0000), neg: 0, expected: 0x0000_0000_0000_0001, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x0000_0000_0000_0001), neg: 0, expected: 0x0000_0000_0000_0002, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0xFFFF_FFFF_FFFF_FFFF), neg: 0, expected: 0x0000_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // addend 0xffffffffffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x0000_0000_FFFF_FFFF), neg: 0, expected: 0x0000_0001_0000_0000, expected_sdst: 0x0000_0000 }, // addend 0x00000000ffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x0000_0000_0000_0000), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x0000_0000_0000_0001), neg: 0, expected: 0x0000_0001_0000_0000, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0xFFFF_FFFF_FFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0xFFFF_FFFF }, // addend 0xffffffffffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x0000_0000_FFFF_FFFF), neg: 0, expected: 0x0000_0001_FFFF_FFFE, expected_sdst: 0x0000_0000 }, // addend 0x00000000ffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_0000_0000), neg: 0, expected: 0x0000_0000_FFFF_FFFF, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_0000_0001), neg: 0, expected: 0x0000_0001_0000_0000, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_FFFF_FFFF), src2: Src::Vgpr(0xFFFF_FFFF_FFFF_FFFF), neg: 0, expected: 0x0000_0000_FFFF_FFFE, expected_sdst: 0xFFFF_FFFF }, // addend 0xffffffffffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_FFFF_FFFF), neg: 0, expected: 0x0000_0001_FFFF_FFFE, expected_sdst: 0x0000_0000 }, // addend 0x00000000ffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_8000_0000), src1: Src::Vgpr(0x0000_0000_8000_0000), src2: Src::Vgpr(0x0000_0000_0000_0000), neg: 0, expected: 0x4000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_8000_0000), src1: Src::Vgpr(0x0000_0000_8000_0000), src2: Src::Vgpr(0x0000_0000_0000_0001), neg: 0, expected: 0x4000_0000_0000_0001, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_8000_0000), src1: Src::Vgpr(0x0000_0000_8000_0000), src2: Src::Vgpr(0xFFFF_FFFF_FFFF_FFFF), neg: 0, expected: 0x3FFF_FFFF_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // addend 0xffffffffffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_8000_0000), src1: Src::Vgpr(0x0000_0000_8000_0000), src2: Src::Vgpr(0x0000_0000_FFFF_FFFF), neg: 0, expected: 0x4000_0000_FFFF_FFFF, expected_sdst: 0x0000_0000 }, // addend 0x00000000ffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_0000_0000), neg: 0, expected: 0xFFFF_FFFE_0000_0001, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000000
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_0000_0001), neg: 0, expected: 0xFFFF_FFFE_0000_0002, expected_sdst: 0x0000_0000 }, // addend 0x0000000000000001
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_FFFF_FFFF), src2: Src::Vgpr(0xFFFF_FFFF_FFFF_FFFF), neg: 0, expected: 0xFFFF_FFFE_0000_0000, expected_sdst: 0xFFFF_FFFF }, // addend 0xffffffffffffffff
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_FFFF_FFFF), src1: Src::Vgpr(0x0000_0000_FFFF_FFFF), src2: Src::Vgpr(0x0000_0000_FFFF_FFFF), neg: 0, expected: 0xFFFF_FFFF_0000_0000, expected_sdst: 0x0000_0000 }, // addend 0x00000000ffffffff
        ],
    );
}

#[test]
fn v_div_scale_f32_vop3sd() {
    // V_DIV_SCALE_F32.
    // S0 is the operand to scale, S1 the denominator and S2 the numerator.
    check_vop3sd(
        764,
        &[
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0x0000_0000 }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F80_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0x0000_0000 }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x4000_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_4000_0000, expected_sdst: 0x0000_0000 }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x4000_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0x0000_0000 }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_FFC0_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x0000_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_FFC0_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_FFC0_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x8000_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_FFC0_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x7F80_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_7F80_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7F80_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0xFF80_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_FF80_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0xFF80_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x7FC0_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_7FC0_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7FC0_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x0000_0001), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_1500_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x0000_0001), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x0080_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_2080_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x0080_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x7F7F_FFFF), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_5F7F_FFFF, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x7F7F_FFFF), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x3F00_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F00_0000, expected_sdst: 0x0000_0000 }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x3F00_0000), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0x0000_0000 }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x4049_0FDB), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_4049_0FDB, expected_sdst: 0x0000_0000 }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3F80_0000), src1: Src::Vgpr(0x4049_0FDB), src2: Src::Vgpr(0x3F80_0000), neg: 0, expected: 0x0000_0000_3F80_0000, expected_sdst: 0x0000_0000 }, // scaling the numerator
        ],
    );
}

#[test]
fn v_div_scale_f64_vop3sd() {
    // V_DIV_SCALE_F64.
    // S0 is the operand to scale, S1 the denominator and S2 the numerator.
    check_vop3sd(
        765,
        &[
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0x0000_0000 }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x3FF0_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0x0000_0000 }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x4000_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x4000_0000_0000_0000, expected_sdst: 0x0000_0000 }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x4000_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0x0000_0000 }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0xFFF8_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0xFFF8_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x8000_0000_0000_0000), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0xFFF8_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x8000_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0xFFF8_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x7FF0_0000_0000_0000), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x7FF0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x7FF0_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0xFFF0_0000_0000_0000), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0xFFF0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0xFFF0_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x7FF8_0000_0000_0000), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x7FF8_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x7FF8_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x0000_0000_0000_0001), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x04D0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x0000_0000_0000_0001), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x0010_0000_0000_0000), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x0810_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x0010_0000_0000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x77EF_FFFF_FFFF_FFFF, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x01A5_6E1F_C2F8_F359), src1: Src::Vgpr(0x01A5_6E1F_C2F8_F359), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x09A5_6E1F_C2F8_F359, expected_sdst: 0xFFFF_FFFF }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x01A5_6E1F_C2F8_F359), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0xFFFF_FFFF }, // scaling the numerator
            Vop3sdCase { src0: Src::Vgpr(0x4202_A05F_2000_0000), src1: Src::Vgpr(0x4202_A05F_2000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x4202_A05F_2000_0000, expected_sdst: 0x0000_0000 }, // scaling the denominator
            Vop3sdCase { src0: Src::Vgpr(0x3FF0_0000_0000_0000), src1: Src::Vgpr(0x4202_A05F_2000_0000), src2: Src::Vgpr(0x3FF0_0000_0000_0000), neg: 0, expected: 0x3FF0_0000_0000_0000, expected_sdst: 0x0000_0000 }, // scaling the numerator
        ],
    );
}
