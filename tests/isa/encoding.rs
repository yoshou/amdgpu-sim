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

/// VOP3P: [31:24] = 11001100, [25:16] = OP, [15] = CLAMP, [14] = OPSEL_HI[2],
/// [13:11] = OPSEL, [10:8] = NEG_HI, [7:0] = VDST, then [40:32] SRC0,
/// [49:41] SRC1, [58:50] SRC2, [60:59] = OPSEL_HI[1:0], [63:61] = NEG.
pub(crate) const fn vop3p(
    op: u32,
    vdst: u32,
    src0: u32,
    src1: u32,
    src2: u32,
    opsel: u32,
    opsel_hi: u32,
    neg: u32,
    neg_hi: u32,
    clamp: bool,
) -> [u32; 2] {
    let lo = (0b1100_1100 << 24)
        | (op << 16)
        | ((clamp as u32) << 15)
        | (((opsel_hi >> 2) & 1) << 14)
        | ((opsel & 7) << 11)
        | ((neg_hi & 7) << 8)
        | (vdst & 0xFF);
    let hi = (src0 & 0x1FF)
        | ((src1 & 0x1FF) << 9)
        | ((src2 & 0x1FF) << 18)
        | ((opsel_hi & 3) << 27)
        | ((neg & 7) << 29);
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

/// VOPC: [31:25] = 0111110, [24:17] = OP, [16:9] = VSRC1, [8:0] = SRC0.
/// VSRC1 is a bare VGPR index.
pub(crate) const fn vopc(op: u32, vsrc1: u32, src0: u32) -> u32 {
    (0b0111110 << 25) | (op << 17) | (vsrc1 << 9) | src0
}

/// The VOP3 encoding of a compare. The destination is an SGPR named by the
/// same [7:0] field a VOP3 uses for its VGPR destination, and the opcode is the
/// VOPC opcode unchanged. There is no clamp or omod.
pub(crate) const fn vop3_sdst(
    op: u32,
    sdst: u32,
    src0: u32,
    src1: u32,
    abs: u32,
    neg: u32,
) -> [u32; 2] {
    let lo = sdst | (abs << 8) | (op << 16) | (0b110101 << 26);
    let hi = src0 | (src1 << 9) | (neg << 29);
    [lo, hi]
}

/// VOP3SD: the VOP3 encoding of the instructions that write a scalar
/// destination as well, which [14:8] names. There is no ABS or OPSEL, and the
/// opcode alone decides that the format is this one.
pub(crate) const fn vop3sd(
    op: u32,
    vdst: u32,
    sdst: u32,
    src0: u32,
    src1: u32,
    src2: u32,
    neg: u32,
) -> [u32; 2] {
    let lo = vdst | (sdst << 8) | (op << 16) | (0b110101 << 26);
    let hi = src0 | (src1 << 9) | (src2 << 18) | (neg << 29);
    [lo, hi]
}

/// VOPD: two operations in one instruction. [8:0] SRCX0, [16:9] VSRCX1,
/// [21:17] OPY, [25:22] OPX, [31:26] = 110010, [40:32] SRCY0, [48:41] VSRCY1,
/// [55:49] VDSTY without its low bit, which the hardware takes as the opposite
/// of VDSTX's, and [63:56] VDSTX.
pub(crate) const fn vopd(
    opx: u32,
    opy: u32,
    vdstx: u32,
    vdsty: u32,
    srcx0: u32,
    vsrcx1: u32,
    srcy0: u32,
    vsrcy1: u32,
) -> [u32; 2] {
    let lo = srcx0 | (vsrcx1 << 9) | (opy << 17) | (opx << 22) | (0b110010 << 26);
    let hi = srcy0 | (vsrcy1 << 9) | ((vdsty >> 1) << 17) | (vdstx << 24);
    [lo, hi]
}

/// SOP1: [31:23] = 101111101, [22:16] = SDST, [15:8] = OP, [7:0] = SSRC0.
pub(crate) const fn sop1(op: u32, sdst: u32, ssrc0: u32) -> u32 {
    (0b101111101 << 23) | (sdst << 16) | (op << 8) | ssrc0
}

/// SOP2: [31:30] = 10, [29:23] = OP, [22:16] = SDST, [15:8] = SSRC1,
/// [7:0] = SSRC0.
pub(crate) const fn sop2(op: u32, sdst: u32, ssrc1: u32, ssrc0: u32) -> u32 {
    (0b10 << 30) | (op << 23) | (sdst << 16) | (ssrc1 << 8) | ssrc0
}

/// SOPC: [31:23] = 101111110, [22:16] = OP, [15:8] = SSRC1, [7:0] = SSRC0.
/// There is no destination field: the result is SCC.
pub(crate) const fn sopc(op: u32, ssrc1: u32, ssrc0: u32) -> u32 {
    (0b101111110 << 23) | (op << 16) | (ssrc1 << 8) | ssrc0
}

/// SOPK: [31:28] = 1011, [27:23] = OP, [22:16] = SDST, [15:0] = SIMM16.
pub(crate) const fn sopk(op: u32, sdst: u32, simm16: u32) -> u32 {
    (0b1011 << 28) | (op << 23) | (sdst << 16) | simm16
}

/// The FLAT-family encodings, which differ only in these top bits.
pub(crate) const VFLAT: u32 = 0b1110_1100;
pub(crate) const VGLOBAL: u32 = 0b1110_1110;
pub(crate) const VSCRATCH: u32 = 0b1110_1101;

/// SADDR = NULL, meaning the address is the full 64 bits in the VGPR pair
/// rather than an SGPR base plus a 32-bit offset.
pub(crate) const SADDR_NULL: u32 = 0x7C;

/// A FLAT, GLOBAL or SCRATCH instruction: three dwords.
/// word0 [31:24] = encoding, [21:14] = OP, [6:0] = SADDR.
/// word1 [7:0] = VDST, [30:23] = VSRC.
/// word2 [7:0] = VADDR, [31:8] = IOFFSET (signed).
pub(crate) const fn vmem(
    enc: u32,
    op: u32,
    vdst: u32,
    vsrc: u32,
    vaddr: u32,
    saddr: u32,
    ioffset: i32,
) -> [u32; 3] {
    [
        (enc << 24) | (op << 14) | saddr,
        vdst | (vsrc << 23),
        (vaddr & 0xFF) | (((ioffset as u32) & 0x00FF_FFFF) << 8),
    ]
}

/// The same, with the cache fields the FLAT family carries: SCOPE and the
/// temporal hint. For an atomic, TH[0] is what asks for the value the memory
/// held before the operation.
/// word1 [19:18] = SCOPE, [22:20] = TH.
pub(crate) const fn vmem_hint(
    enc: u32,
    op: u32,
    vdst: u32,
    vsrc: u32,
    vaddr: u32,
    saddr: u32,
    ioffset: i32,
    th: u32,
    scope: u32,
) -> [u32; 3] {
    let [w0, w1, w2] = vmem(enc, op, vdst, vsrc, vaddr, saddr, ioffset);
    [w0, w1 | (scope << 18) | (th << 20), w2]
}

/// A SCRATCH instruction: three dwords, laid out like the FLAT family but with
/// an SVE bit that says whether VADDR takes part.
/// word0 [31:24] = encoding, [21:14] = OP, [6:0] = SADDR.
/// word1 [7:0] = VDST, [17] = SVE, [30:23] = VSRC.
/// word2 [7:0] = VADDR, [31:8] = IOFFSET (signed).
pub(crate) const fn vscratch(
    op: u32,
    vdst: u32,
    vsrc: u32,
    vaddr: u32,
    saddr: u32,
    ioffset: i32,
    sve: u32,
) -> [u32; 3] {
    [
        (VSCRATCH << 24) | (op << 14) | (saddr & 0x7F),
        (vdst & 0xFF) | (sve << 17) | ((vsrc & 0xFF) << 23),
        (vaddr & 0xFF) | (((ioffset as u32) & 0x00FF_FFFF) << 8),
    ]
}

/// SMEM: [31:26] = 111101, [18:13] = OP, [12:6] = SDATA, [5:0] = SBASE,
/// IOFFSET in [55:32] and SOFFSET in [63:57]. SBASE is a register *pair* index,
/// so s[10:11] is 5. SOFFSET must be NULL, or the SGPR it names is added to the
/// address.
pub(crate) const fn smem(op: u32, sdata: u32, sbase: u32, ioffset: i32) -> [u32; 2] {
    [
        (0b111101 << 26) | (op << 13) | (sdata << 6) | sbase,
        ((ioffset as u32) & 0x00FF_FFFF) | (SADDR_NULL << 25),
    ]
}

/// DS: [31:26] = 110110, [25:18] = OP, [15:8] = OFFSET1, [7:0] = OFFSET0, and
/// the second dword carries [7:0] ADDR, [15:8] DATA0, [23:16] DATA1,
/// [31:24] VDST.
pub(crate) const fn ds(
    op: u32,
    vdst: u32,
    addr: u32,
    data0: u32,
    data1: u32,
    offset0: u32,
    offset1: u32,
) -> [u32; 2] {
    [
        (0b110110 << 26) | (op << 18) | ((offset1 & 0xFF) << 8) | (offset0 & 0xFF),
        (addr & 0xFF) | ((data0 & 0xFF) << 8) | ((data1 & 0xFF) << 16) | ((vdst & 0xFF) << 24),
    ]
}

/// VIMAGE: the image format that takes no sampler, which the ray-tracing
/// instructions use. [2:0] DIM, [4] R128, [5] D16, [6] A16, [21:14] OP,
/// [25:22] DMASK, [31:26] = 110100, then [39:32] VDATA, [49:41] RSRC,
/// [51:50] SCOPE, [54:52] TH, [55] TFE, [63:56] VADDR4, and the third dword
/// holds VADDR0..VADDR3. The BVH opcodes read their address VGPRs in groups,
/// so each VADDR field names the first register of its group.
pub(crate) const fn vimage(
    op: u32,
    dim: u32,
    r128: u32,
    dmask: u32,
    vdata: u32,
    rsrc: u32,
    vaddr: [u32; 5],
) -> [u32; 3] {
    [
        (0b110100 << 26) | (dmask << 22) | (op << 14) | (r128 << 4) | dim,
        (vdata & 0xFF) | ((rsrc & 0x1FF) << 9) | (vaddr[4] << 24),
        vaddr[0] | (vaddr[1] << 8) | (vaddr[2] << 16) | (vaddr[3] << 24),
    ]
}

/// VSAMPLE: the image format that takes a sampler. Laid out like VIMAGE except
/// that [3] is TFE, [13] UNRM, [40] LWE and [63:55] SAMP, which names the S#.
pub(crate) const fn vsample(
    op: u32,
    dim: u32,
    dmask: u32,
    unrm: u32,
    vdata: u32,
    rsrc: u32,
    samp: u32,
    vaddr: [u32; 4],
) -> [u32; 3] {
    [
        (0b111001 << 26) | (dmask << 22) | (op << 14) | (unrm << 13) | dim,
        (vdata & 0xFF) | ((rsrc & 0x1FF) << 9) | ((samp & 0x1FF) << 23),
        vaddr[0] | (vaddr[1] << 8) | (vaddr[2] << 16) | (vaddr[3] << 24),
    ]
}
