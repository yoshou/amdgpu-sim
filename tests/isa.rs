//! Unit tests for the RDNA4 instruction implementations.
//!
//! Every expected value in this file was captured on an AMD gfx1200 part. The
//! comparison is bit-exact unless the ISA manual grants a tolerance for that
//! opcode, or the operation cannot be bit-exact by its nature -- in which case
//! the threshold is quoted from the manual or derived from measurement, and the
//! comment says which. Special values (NaN, +-0, denormals, +-inf) are always
//! compared bit-exactly, because the manual pins them down in every case.
//!
//! `tests/data/harness_gfx1200.o` is a fixed kernel with a 16-byte slot that
//! these tests patch with the instruction under test, so operand fields and
//! modifiers are reachable without recompiling anything.

use amdgpu_sim::buffer::*;
use amdgpu_sim::processor::*;
use amdgpu_sim::rdna_processor::*;
use object::*;
use std::fs::File;
use std::io::Read;

const LANES: usize = 32;

/// Bytes reserved for the patch slot: an 8-byte marker plus 30 `s_nop`. Enough
/// for setup instructions ahead of the instruction under test. Must match the
/// `.rept` count in tools/isa_probe/harness.hip.
const SLOT_BYTES: usize = 8 + 30 * 4;

/// Each harness opens its slot with `v_mov_b32 v6, <literal>`; the literal
/// distinguishes the formats, which all live in one code object.
fn slot_marker(literal: u32) -> [u8; 8] {
    let mut m = [0u8; 8];
    m[..4].copy_from_slice(&0x7E0C_02FFu32.to_le_bytes());
    m[4..].copy_from_slice(&literal.to_le_bytes());
    m
}

const S_NOP: u32 = 0xBF80_0000;

/// Source operand field: VGPR `n` (the 9-bit field encodes VGPRs at 256..511).
const fn vgpr(n: u32) -> u32 {
    256 + n
}

/// VOP1: [31:25] = 0111111, [24:17] = VDST, [16:9] = OP, [8:0] = SRC0.
const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    (0b0111111 << 25) | (vdst << 17) | (op << 9) | src0
}

/// VOP2: [31] = 0, [30:25] = OP, [24:17] = VDST, [16:9] = VSRC1, [8:0] = SRC0.
/// VSRC1 is a bare VGPR index, not a full operand field.
const fn vop2(op: u32, vdst: u32, vsrc1: u32, src0: u32) -> u32 {
    (op << 25) | (vdst << 17) | (vsrc1 << 9) | src0
}

/// VOP3: [7:0] VDST, [10:8] ABS, [14:11] OPSEL, [15] CLAMP, [25:16] OP,
/// [31:26] = 110101, [40:32] SRC0, [49:41] SRC1, [58:50] SRC2, [60:59] OMOD,
/// [63:61] NEG.
#[allow(clippy::too_many_arguments)]
const fn vop3(
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
/// operand field, not a classification of our own.
#[derive(Clone, Copy)]
enum Src {
    /// Value placed in the VGPR pair this harness assigns to the position.
    Vgpr(u32),
    /// Value placed in the SGPR pair this harness assigns to the position.
    Sgpr(u32),
    /// Inline constant, named by its operand-field encoding (128 = 0,
    /// 240 = 0.5, 242 = 1.0, 246 = 4.0, ...). Carries no value of its own.
    Inline(u32),
}

/// One instruction format's harness kernel: where it puts the sources, what it
/// reads back, and the slot it executes in between.
struct Harness {
    mem: Vec<u8>,
    slot: usize,
    kernel_addr: usize,
    kernarg_size: usize,
    /// Source dwords the kernel reads per lane.
    src_stride: usize,
    /// Result dwords the kernel writes per lane.
    out_stride: usize,
}

#[derive(serde::Deserialize)]
struct KernelMeta {
    #[serde(alias = ".kernarg_segment_size")]
    kernarg_segment_size: i64,
}

#[derive(serde::Deserialize)]
struct Meta {
    #[serde(alias = "amdhsa.kernels")]
    amdhsa_kernels: Vec<KernelMeta>,
}

fn align(value: usize, align: usize) -> usize {
    value.div_ceil(align) * align
}

fn kernarg_segment_size(note: &[u8]) -> usize {
    let mut pos = 0;
    while pos < note.len() {
        let name_size = get_u32(note, pos) as usize;
        let data_size = get_u32(note, pos + 4) as usize;
        let note_type = get_u32(note, pos + 8) as usize;
        pos = align(pos + 12 + name_size, 4);
        let data = get_bytes(note, pos, data_size);
        pos = align(pos + data_size, 4);
        if note_type == 32 {
            let map: Meta = rmp_serde::from_slice(&data).unwrap();
            return map.amdhsa_kernels[0].kernarg_segment_size as usize;
        }
    }
    panic!("no MessagePack metadata note in the harness object");
}

impl Harness {
    fn vop1() -> Self {
        Self::load("harness_vop1.kd", 0x1111_1111, 2, 2)
    }

    fn vop2() -> Self {
        Self::load("harness_vop2.kd", 0x2222_2222, 4, 4)
    }

    fn vop3() -> Self {
        Self::load("harness_vop3.kd", 0x3333_3333, 6, 2)
    }

    fn load(kernel: &str, marker_literal: u32, src_stride: usize, out_stride: usize) -> Self {
        let marker = slot_marker(marker_literal);
        let mut data = vec![];
        File::open("tests/data/harness_gfx1200.o")
            .expect("tests/data/harness_gfx1200.o")
            .read_to_end(&mut data)
            .unwrap();
        let elf = ElfFile::parse(&data).expect("failed to parse the harness object");

        let note = elf
            .sections()
            .find(|s| s.name() == Some(".note"))
            .expect("no .note section");
        let kernarg_size = kernarg_segment_size(note.data());

        let mut mem = Vec::<u8>::new();
        for segment in elf.segments() {
            let offset = segment.address() as usize;
            let size = segment.size() as usize;
            let new_size = mem.len().max(offset + size);
            mem.resize(new_size, 0);
            mem[offset..(offset + size.min(segment.data().len()))]
                .copy_from_slice(segment.data());
        }

        let slot = mem
            .windows(marker.len())
            .position(|w| w == marker)
            .unwrap_or_else(|| panic!("patch slot marker for {} not found", kernel));
        // The marker alone is not proof we found the right place; the padding
        // behind it must be there too, or the harness and this file disagree.
        for i in (marker.len()..SLOT_BYTES).step_by(4) {
            assert_eq!(
                get_u32(&mem, slot + i),
                S_NOP,
                "harness slot is shorter than SLOT_BYTES at +{}",
                i
            );
        }

        let kernel_addr = elf
            .symbols()
            .find(|sym| sym.name() == Some(kernel))
            .unwrap_or_else(|| panic!("{} not found", kernel))
            .address() as usize;

        Harness {
            mem,
            slot,
            kernel_addr,
            kernarg_size,
            src_stride,
            out_stride,
        }
    }

    /// Patch `words` into the slot, pad the rest with S_NOP, and run one wave.
    /// `src` holds `src_stride` dwords per lane and `uni` the values the kernel
    /// puts in SGPRs; the result is `out_stride` dwords per lane.
    fn run(&self, engine: Engine, words: &[u32], src: &[u32], uni: &[u32]) -> Vec<u32> {
        assert_eq!(src.len(), LANES * self.src_stride);
        assert!(words.len() * 4 <= SLOT_BYTES, "instruction sequence exceeds the slot");
        let mut mem = self.mem.clone();
        for i in 0..(SLOT_BYTES / 4) {
            let word = words.get(i).copied().unwrap_or(S_NOP);
            mem[self.slot + i * 4..self.slot + i * 4 + 4].copy_from_slice(&word.to_le_bytes());
        }

        let mut out = vec![0u32; LANES * self.out_stride];
        let mut arg_buffer = vec![0u8; self.kernarg_size];
        set_u64(&mut arg_buffer, 0, out.as_mut_ptr() as u64);
        set_u64(&mut arg_buffer, 8, src.as_ptr() as u64);
        set_u64(&mut arg_buffer, 16, uni.as_ptr() as u64);

        let aql = HsaKernelDispatchPacket {
            header: 0,
            setup: 0,
            workgroup_size_x: LANES as u16,
            workgroup_size_y: 1,
            workgroup_size_z: 1,
            grid_size_x: 1,
            grid_size_y: 1,
            grid_size_z: 1,
            private_segment_size: 0,
            group_segment_size: 0,
            kernel_object: Pointer::new(&mem, self.kernel_addr),
            kernarg_address: Pointer::new(&arg_buffer, 0),
        };

        let mut processor = RDNAProcessor::with_engine(&aql, 32, 32, &mem, engine);
        processor.execute();
        out
    }
}

fn engine_name(engine: Engine) -> &'static str {
    match engine {
        Engine::Interpreter => "interpreter",
        Engine::LlvmJit => "LLVM JIT",
    }
}

// ---------------------------------------------------------------- comparison

fn is_nan_f32(bits: u32) -> bool {
    (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0
}

fn is_inf_f32(bits: u32) -> bool {
    (bits & 0x7FFF_FFFF) == 0x7F80_0000
}

fn is_zero_f32(bits: u32) -> bool {
    (bits & 0x7FFF_FFFF) == 0
}

fn is_denorm_f32(bits: u32) -> bool {
    (bits & 0x7F80_0000) == 0 && (bits & 0x007F_FFFF) != 0
}

/// Monotone map from an f32 bit pattern to an ordered integer; the difference
/// between two such integers is the ULP distance.
fn ordered_f32(bits: u32) -> i64 {
    if bits & 0x8000_0000 != 0 {
        -((bits & 0x7FFF_FFFF) as i64)
    } else {
        bits as i64
    }
}

fn ulp_f32(a: u32, b: u32) -> i64 {
    (ordered_f32(a) - ordered_f32(b)).abs()
}

fn show_f32(bits: u32) -> String {
    format!("0x{:08X} ({:e})", bits, f32::from_bits(bits))
}

fn is_nan_f64(bits: u64) -> bool {
    (bits & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000 && (bits & 0x000F_FFFF_FFFF_FFFF) != 0
}

fn is_inf_f64(bits: u64) -> bool {
    (bits & 0x7FFF_FFFF_FFFF_FFFF) == 0x7FF0_0000_0000_0000
}

fn is_zero_f64(bits: u64) -> bool {
    (bits & 0x7FFF_FFFF_FFFF_FFFF) == 0
}

fn is_denorm_f64(bits: u64) -> bool {
    (bits & 0x7FF0_0000_0000_0000) == 0 && (bits & 0x000F_FFFF_FFFF_FFFF) != 0
}

fn ordered_f64(bits: u64) -> i128 {
    if bits & 0x8000_0000_0000_0000 != 0 {
        -((bits & 0x7FFF_FFFF_FFFF_FFFF) as i128)
    } else {
        bits as i128
    }
}

fn ulp_f64(a: u64, b: u64) -> i128 {
    (ordered_f64(a) - ordered_f64(b)).abs()
}

fn show_f64(bits: u64) -> String {
    format!("0x{:016X} ({:e})", bits, f64::from_bits(bits))
}

/// Bit-exact comparison of a VOP1 f32 instruction against captured hardware.
fn check_vop1_f32(op: u32, cases: &[(u32, u32)]) {
    check_vop1_f32_ulp(op, 0, cases);
}

/// As above, but finite non-zero results may differ from hardware by up to
/// `ulp`. NaN, +-0, +-inf and denormal results are still compared bit-exactly.
fn check_vop1_f32_ulp(op: u32, ulp: i64, cases: &[(u32, u32)]) {
    assert!(cases.len() <= LANES, "at most {} cases per call", LANES);
    let harness = Harness::vop1();

    let mut src = vec![0u32; LANES * harness.src_stride];
    for (i, (input, _)) in cases.iter().enumerate() {
        src[i * harness.src_stride] = *input;
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
            // tolerance never applies to them.
            let special = is_nan_f32(*hw)
                || is_nan_f32(got)
                || is_zero_f32(*hw)
                || is_zero_f32(got)
                || is_inf_f32(*hw)
                || is_inf_f32(got)
                || is_denorm_f32(*hw)
                || is_denorm_f32(got);
            let distance = ulp_f32(got, *hw);
            if !special && distance <= ulp {
                continue;
            }
            failures.push(format!(
                "  {:<11} src0={} hardware={} simulator={}{}",
                engine_name(engine),
                show_f32(*input),
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
fn check_vop1_f64(op: u32, cases: &[(u64, u64)]) {
    check_vop1_f64_ulp(op, 0, cases);
}

/// As above, but finite non-zero results may differ from hardware by up to
/// `ulp`. NaN, +-0, +-inf and denormal results are still compared bit-exactly.
fn check_vop1_f64_ulp(op: u32, ulp: i128, cases: &[(u64, u64)]) {
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
                || is_inf_f64(got)
                || is_denorm_f64(*hw)
                || is_denorm_f64(got);
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

/// One VOP2 case. src0 takes the full 9-bit operand field; vsrc1 can only name
/// a VGPR, which is the format's own asymmetry.
struct Vop2F32 {
    src0: Src,
    vsrc1: u32,
    expected: u32,
}

/// Bit-exact comparison of a VOP2 f32 instruction against captured hardware.
fn check_vop2_f32(op: u32, cases: &[Vop2F32]) {
    let harness = Harness::vop2();

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut src = vec![0u32; LANES * harness.src_stride];
        let mut uni = vec![0u32; 8];
        let field = match case.src0 {
            Src::Vgpr(value) => {
                for lane in 0..LANES {
                    src[lane * harness.src_stride] = value;
                }
                vgpr(0)
            }
            Src::Sgpr(value) => {
                uni[0] = value;
                10
            }
            Src::Inline(encoding) => encoding,
        };
        for lane in 0..LANES {
            src[lane * harness.src_stride + 2] = case.vsrc1;
        }
        let words = [vop2(op, 6, 2, field)];

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = harness.run(engine, &words, &src, &uni)[0];
            if got == case.expected {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} vsrc1={} hardware={} simulator={}",
                engine_name(engine),
                i,
                show_f32(case.vsrc1),
                show_f32(case.expected),
                show_f32(got),
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

/// One VOP3 case. Every operand and modifier the format has is a field, so a
/// test cannot leave one unstated.
struct Vop3F32 {
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
fn check_vop3_f32(op: u32, cases: &[Vop3F32]) {
    check_vop3_f32_ulp(op, 0, cases);
}

/// As above, with a tolerance on finite non-zero results.
fn check_vop3_f32_ulp(op: u32, ulp: i64, cases: &[Vop3F32]) {
    assert!(cases.len() <= LANES, "at most {} cases per call", LANES);
    let harness = Harness::vop3();

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        // One case per run: the SGPR sources are wave-uniform, so cases that
        // name an SGPR cannot share a wave with cases that use a different
        // value there.
        let mut src = vec![0u32; LANES * harness.src_stride];
        let mut uni = vec![0u32; 8];
        let mut field = [0u32; 3];
        for (position, s) in [case.src0, case.src1, case.src2].iter().enumerate() {
            field[position] = match s {
                Src::Vgpr(value) => {
                    for lane in 0..LANES {
                        src[lane * harness.src_stride + position * 2] = *value;
                    }
                    vgpr(position as u32 * 2)
                }
                Src::Sgpr(value) => {
                    uni[position * 2] = *value;
                    10 + position as u32 * 2
                }
                Src::Inline(encoding) => *encoding,
            };
        }
        let words = vop3(
            op, 6, field[0], field[1], field[2], case.abs, case.neg, case.clamp, case.omod,
        );

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = harness.run(engine, &words, &src, &uni)[0];
            if got == case.expected {
                continue;
            }
            let special = is_nan_f32(case.expected)
                || is_nan_f32(got)
                || is_zero_f32(case.expected)
                || is_zero_f32(got)
                || is_inf_f32(case.expected)
                || is_inf_f32(got)
                || is_denorm_f32(case.expected)
                || is_denorm_f32(got);
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

// -------------------------------------------------------------------- tests

#[test]
fn v_rcp_f32_vop1() {
    // ISA 16.8 V_RCP_F32: "1ULP accuracy ... Denormals are flushed."
    check_vop1_f32_ulp(
        42,
        1,
        &[
            (0x0000_0000, 0x7F80_0000), // +0           -> +inf
            (0x8000_0000, 0xFF80_0000), // -0           -> -inf
            (0x7F80_0000, 0x0000_0000), // +inf         -> +0
            (0xFF80_0000, 0x8000_0000), // -inf         -> -0
            (0x0000_0001, 0x7F80_0000), // min denorm   -> input flushed to +0
            (0x807F_FFFF, 0xFF80_0000), // max -denorm  -> input flushed to -0
            (0x7F7F_FFFF, 0x0000_0000), // FLT_MAX      -> denormal result flushed
            (0x3F80_0000, 0x3F80_0000), // 1.0          -> 1.0
            (0x4000_0000, 0x3F00_0000), // 2.0          -> 0.5
            (0x4265_2EE0, 0x3C8E_FA35), // 57.295776    -> within 1 ULP
        ],
    );
}

#[test]
fn v_rcp_f32_vop3() {
    // Same opcode in the VOP3 encoding, exercising every operand class and
    // modifier the format has. Values captured on gfx1200.
    const V2: u32 = 0x4000_0000; // 2.0
    const VM2: u32 = 0xC000_0000; // -2.0
    const V4: u32 = 0x4080_0000; // 4.0
    check_vop3_f32_ulp(
        426,
        1,
        &[
            Vop3F32 { src0: Src::Vgpr(V2),  src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b000, neg: 0b000, clamp: false, omod: 0, expected: 0x3F00_0000 },
            Vop3F32 { src0: Src::Vgpr(V2),  src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b000, neg: 0b001, clamp: false, omod: 0, expected: 0xBF00_0000 },
            Vop3F32 { src0: Src::Vgpr(VM2), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b001, neg: 0b000, clamp: false, omod: 0, expected: 0x3F00_0000 },
            Vop3F32 { src0: Src::Vgpr(VM2), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b001, neg: 0b001, clamp: false, omod: 0, expected: 0xBF00_0000 },
            Vop3F32 { src0: Src::Vgpr(V4),  src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b000, neg: 0b000, clamp: false, omod: 1, expected: 0x3F00_0000 },
            Vop3F32 { src0: Src::Vgpr(V4),  src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b000, neg: 0b000, clamp: false, omod: 3, expected: 0x3E00_0000 },
            Vop3F32 { src0: Src::Vgpr(VM2), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b000, neg: 0b000, clamp: true,  omod: 0, expected: 0x0000_0000 },
            Vop3F32 { src0: Src::Sgpr(V2),  src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b000, neg: 0b000, clamp: false, omod: 0, expected: 0x3F00_0000 },
            Vop3F32 { src0: Src::Inline(242), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b000, neg: 0b000, clamp: false, omod: 0, expected: 0x3F80_0000 },
            Vop3F32 { src0: Src::Inline(246), src1: Src::Vgpr(0), src2: Src::Vgpr(0), abs: 0b000, neg: 0b000, clamp: false, omod: 0, expected: 0x3E80_0000 },
        ],
    );
}

#[test]
fn v_rcp_iflag_f32_vop1() {
    // ISA 16.8 V_RCP_IFLAG_F32 gives the pseudo code `D0.f32 = 1.0F / S0.f32`
    // with no accuracy statement, but the hardware runs it on the reciprocal
    // unit: measured 1 ULP on normal inputs and denormals flushed, matching the
    // 1 ULP the manual states for V_RCP_F32.
    check_vop1_f32_ulp(
        43,
        1,
        &[
            (0x0000_0000, 0x7F80_0000), // +0           -> +inf
            (0x8000_0000, 0xFF80_0000), // -0           -> -inf
            (0x7F80_0000, 0x0000_0000), // +inf         -> +0
            (0xFF80_0000, 0x8000_0000), // -inf         -> -0
            (0x0000_0001, 0x7F80_0000), // min denorm   -> input flushed to +0
            (0x807F_FFFF, 0xFF80_0000), // max -denorm  -> input flushed to -0
            (0x7F7F_FFFF, 0x0000_0000), // FLT_MAX      -> denormal result flushed
            (0x3F80_0000, 0x3F80_0000), // 1.0          -> 1.0
            (0x4000_0000, 0x3F00_0000), // 2.0          -> 0.5
        ],
    );
}

#[test]
fn v_sqrt_f32_vop1() {
    // ISA 16.8 V_SQRT_F32: "1ULP accuracy, denormals are flushed."
    check_vop1_f32_ulp(
        51,
        1,
        &[
            (0x0000_0000, 0x0000_0000), // +0          -> +0
            (0x8000_0000, 0x8000_0000), // -0          -> -0
            (0x7F80_0000, 0x7F80_0000), // +inf        -> +inf
            (0xFF80_0000, 0xFFC0_0000), // -inf        -> NaN
            (0x0000_0001, 0x0000_0000), // min denorm  -> input flushed to +0
            (0x807F_FFFF, 0x8000_0000), // max -denorm -> input flushed to -0
            (0x3F80_0000, 0x3F80_0000), // 1.0         -> 1.0
            (0x4080_0000, 0x4000_0000), // 4.0         -> 2.0
            (0x4000_0000, 0x3FB5_04F3), // 2.0         -> sqrt(2), within 1 ULP
            (0x7FA0_0000, 0x7FE0_0000), // sNaN        -> quieted, payload kept
        ],
    );
}

#[test]
fn v_floor_f32_vop1() {
    // ISA 16.8 V_FLOOR_F32: exact pseudo code, no tolerance.
    check_vop1_f32(
        36,
        &[
            (0x0000_0000, 0x0000_0000), // +0          -> +0
            (0x8000_0000, 0x8000_0000), // -0          -> -0
            (0x3FC0_0000, 0x3F80_0000), // 1.5         -> 1.0
            (0xBFC0_0000, 0xC000_0000), // -1.5        -> -2.0
            (0xBF00_0000, 0xBF80_0000), // -0.5        -> -1.0
            (0x7F80_0000, 0x7F80_0000), // +inf        -> +inf
            (0x0000_0001, 0x0000_0000), // min denorm  -> +0
            (0x807F_FFFF, 0xBF80_0000), // max -denorm -> -1.0
            (0x7FA0_0000, 0x7FE0_0000), // sNaN        -> quieted, payload kept
        ],
    );
}

#[test]
fn v_rndne_f32_vop1() {
    // ISA 16.8 V_RNDNE_F32: exact pseudo code, no tolerance. Note the manual's
    // pseudo code returns +0 for a tiny negative input; the hardware returns -0.
    check_vop1_f32(
        35,
        &[
            (0x0000_0000, 0x0000_0000), // +0    -> +0
            (0x8000_0000, 0x8000_0000), // -0    -> -0
            (0x3F00_0000, 0x0000_0000), // 0.5   -> 0.0, ties to even
            (0x3FC0_0000, 0x4000_0000), // 1.5   -> 2.0
            (0x4020_0000, 0x4000_0000), // 2.5   -> 2.0, ties to even
            (0xBF00_0000, 0x8000_0000), // -0.5  -> -0.0
            (0xBFC0_0000, 0xC000_0000), // -1.5  -> -2.0
            (0xBC00_0000, 0x8000_0000), // -0.0078125 -> -0.0
            (0x7FA0_0000, 0x7FE0_0000), // sNaN  -> quieted, payload kept
        ],
    );
}

#[test]
fn v_frexp_mant_f32_vop1() {
    // ISA 16.8 V_FREXP_MANT_F32: exact pseudo code, no tolerance.
    check_vop1_f32(
        64,
        &[
            (0x3F80_0000, 0x3F00_0000), // 1.0   -> 0.5
            (0x4000_0000, 0x3F00_0000), // 2.0   -> 0.5
            (0xC000_0000, 0xBF00_0000), // -2.0  -> -0.5
            (0x3F40_0000, 0x3F40_0000), // 0.75  -> 0.75
            (0x0000_0000, 0x0000_0000), // +0    -> +0
            (0x8000_0000, 0x8000_0000), // -0    -> -0
            (0x7F80_0000, 0x7F80_0000), // +inf  -> +inf
            (0x7FA0_0000, 0x7FE0_0000), // sNaN  -> quieted, payload kept
        ],
    );
}

#[test]
fn v_rcp_f64_vop1() {
    // ISA 16.8 V_RCP_F64: "This opcode has (2**29)ULP accuracy and supports
    // denormals." The hardware value is a Newton-Raphson seed, so 1/3 lands
    // 112,984,064 ULP from the exact quotient -- inside the stated bound.
    check_vop1_f64_ulp(
        47,
        1 << 29,
        &[
            (0x0000_0000_0000_0000, 0x7FF0_0000_0000_0000), // +0   -> +inf
            (0x8000_0000_0000_0000, 0xFFF0_0000_0000_0000), // -0   -> -inf
            (0x7FF0_0000_0000_0000, 0x0000_0000_0000_0000), // +inf -> +0
            (0xFFF0_0000_0000_0000, 0x8000_0000_0000_0000), // -inf -> -0
            (0x3FF0_0000_0000_0000, 0x3FF0_0000_0000_0000), // 1.0  -> 1.0
            (0x4000_0000_0000_0000, 0x3FE0_0000_0000_0000), // 2.0  -> 0.5
            (0x4010_0000_0000_0000, 0x3FD0_0000_0000_0000), // 4.0  -> 0.25
            (0x3FD5_5555_5555_5555, 0x4008_0000_06BC_0000), // 1/3  -> seed near 3.0
        ],
    );
}

#[test]
fn v_rsq_f64_vop1() {
    // ISA 16.8 V_RSQ_F64: "This opcode has (2**29)ULP accuracy and supports
    // denormals."
    check_vop1_f64_ulp(
        49,
        1 << 29,
        &[
            (0x0000_0000_0000_0000, 0x7FF0_0000_0000_0000), // +0   -> +inf
            (0x8000_0000_0000_0000, 0xFFF0_0000_0000_0000), // -0   -> -inf
            (0x7FF0_0000_0000_0000, 0x0000_0000_0000_0000), // +inf -> +0
            (0x3FF0_0000_0000_0000, 0x3FF0_0000_0000_0000), // 1.0  -> 1.0
            (0x4010_0000_0000_0000, 0x3FE0_0000_0000_0000), // 4.0  -> 0.5
            (0xBFF0_0000_0000_0000, 0xFFF8_0000_0000_0000), // -1.0 -> NaN
        ],
    );
}

#[test]
fn v_rndne_f64_vop1() {
    // ISA 16.8 V_RNDNE_F64: exact pseudo code, no tolerance.
    check_vop1_f64(
        25,
        &[
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0   -> +0
            (0x8000_0000_0000_0000, 0x8000_0000_0000_0000), // -0   -> -0
            (0x3FE0_0000_0000_0000, 0x0000_0000_0000_0000), // 0.5  -> 0.0
            (0x3FF8_0000_0000_0000, 0x4000_0000_0000_0000), // 1.5  -> 2.0
            (0x4004_0000_0000_0000, 0x4000_0000_0000_0000), // 2.5  -> 2.0
            (0xBFE0_0000_0000_0000, 0x8000_0000_0000_0000), // -0.5 -> -0.0
            (0xBC00_0000_0000_0000, 0x8000_0000_0000_0000), // tiny negative -> -0.0
        ],
    );
}

#[test]
fn v_fract_f64_vop1() {
    // ISA 16.12 V_FRACT_F64: "0.5ULP accuracy" -- exact -- and the result is in
    // [0,1), so a tiny negative input gives the largest value below 1 rather
    // than rounding up to 1.0.
    check_vop1_f64(
        62,
        &[
            (0xBFF8_0000_0000_0000, 0x3FE0_0000_0000_0000), // -1.5 -> 0.5
            (0x3FF8_0000_0000_0000, 0x3FE0_0000_0000_0000), // 1.5  -> 0.5
            (0x0000_0000_0000_0000, 0x0000_0000_0000_0000), // +0   -> +0
            (0x8000_0000_0000_0000, 0x0000_0000_0000_0000), // -0   -> +0
            (0xBC00_0000_0000_0000, 0x3FEF_FFFF_FFFF_FFFF), // tiny negative -> largest < 1
            (0x7FF0_0000_0000_0000, 0xFFF8_0000_0000_0000), // +inf -> NaN
        ],
    );
}

#[test]
fn v_sub_f32_vop2() {
    // ISA 16.7 V_SUB_F32: "0.5ULP precision" -- exact. The hardware evaluates
    // S0 + (-S1), so a NaN coming from src1 is propagated with its sign flipped.
    const ONE: u32 = 0x3F80_0000;
    const HALF: u32 = 0x3F00_0000;
    check_vop2_f32(
        4,
        &[
            Vop2F32 { src0: Src::Vgpr(ONE),        vsrc1: HALF,        expected: 0x3F00_0000 },
            Vop2F32 { src0: Src::Vgpr(HALF),       vsrc1: ONE,         expected: 0xBF00_0000 },
            Vop2F32 { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x7F80_0000, expected: 0xFFC0_0000 },
            Vop2F32 { src0: Src::Vgpr(0x7FA0_0000), vsrc1: ONE,         expected: 0x7FE0_0000 },
            Vop2F32 { src0: Src::Vgpr(ONE),        vsrc1: 0x7FA0_0000, expected: 0xFFE0_0000 },
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0000, expected: 0x0000_0000 },
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0000, expected: 0x8000_0000 },
        ],
    );
}
