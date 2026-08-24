//! RDNA4 instruction word layouts and operand fields.
//!
//! Nothing here interprets an instruction; these are the bit positions the ISA
//! manual gives for each format, and the encodings of the 9-bit operand field.


/// Bytes reserved for the patch slot: an 8-byte marker plus 30 `s_nop`. Enough
/// for setup instructions ahead of the instruction under test. Must match the
/// `.rept` count in tools/isa_probe/harness.hip.
pub(crate) const SLOT_BYTES: usize = 8 + 30 * 4;

/// Each harness opens its slot with `v_mov_b32 v6, <literal>`; the literal
/// distinguishes the formats, which all live in one code object.
pub(crate) fn slot_marker(literal: u32) -> [u8; 8] {
    let mut m = [0u8; 8];
    m[..4].copy_from_slice(&0x7E0C_02FFu32.to_le_bytes());
    m[4..].copy_from_slice(&literal.to_le_bytes());
    m
}

pub(crate) const S_NOP: u32 = 0xBF80_0000;

/// Source operand field: VGPR `n` (the 9-bit field encodes VGPRs at 256..511).
pub(crate) const fn vgpr(n: u32) -> u32 {
    256 + n
}

/// VOP1: [31:25] = 0111111, [24:17] = VDST, [16:9] = OP, [8:0] = SRC0.
pub(crate) const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    (0b0111111 << 25) | (vdst << 17) | (op << 9) | src0
}

/// VOP2: [31] = 0, [30:25] = OP, [24:17] = VDST, [16:9] = VSRC1, [8:0] = SRC0.
/// VSRC1 is a bare VGPR index, not a full operand field.
pub(crate) const fn vop2(op: u32, vdst: u32, vsrc1: u32, src0: u32) -> u32 {
    (op << 25) | (vdst << 17) | (vsrc1 << 9) | src0
}

/// VOP3: [7:0] VDST, [10:8] ABS, [14:11] OPSEL, [15] CLAMP, [25:16] OP,
/// [31:26] = 110101, [40:32] SRC0, [49:41] SRC1, [58:50] SRC2, [60:59] OMOD,
/// [63:61] NEG.
#[allow(clippy::too_many_arguments)]
pub(crate) const fn vop3(
    op: u32,
    vdst: u32,
    src0: u32,
    src1: u32,
    src2: u32,
    abs: u32,
    neg: u32,
    clamp: bool,
    omod: u32,
) -> [u32; 2] {
    let lo = vdst | (abs << 8) | (clamp as u32) << 15 | (op << 16) | (0b110101 << 26);
    let hi = src0 | (src1 << 9) | (src2 << 18) | (omod << 27) | (neg << 29);
    [lo, hi]
}

/// Where a source operand comes from. These are the encodings of the 9-bit
/// operand field, not a classification of our own. The value is 64 bits because
/// the harness always fills a register pair; a 32-bit instruction reads only the
/// low register of it.
#[derive(Clone, Copy)]
pub(crate) enum Src {
    /// Value placed in the VGPR pair this harness assigns to the position.
    Vgpr(u64),
    /// Value placed in the SGPR pair this harness assigns to the position.
    Sgpr(u64),
    /// Inline constant, named by its operand-field encoding (128 = 0,
    /// 240 = 0.5, 242 = 1.0, 246 = 4.0, ...). Carries no value of its own.
    Inline(u32),
    /// Literal constant: operand field 255 plus a following dword.
    Literal(u32),
}
