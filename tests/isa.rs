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
/// operand field, not a classification of our own. The value is 64 bits because
/// the harness always fills a register pair; a 32-bit instruction reads only the
/// low register of it.
#[derive(Clone, Copy)]
enum Src {
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
fn check_vop1_f32(op: u32, cases: &[(u64, u32)]) {
    check_vop1_f32_ulp(op, 0, cases);
}

/// As above, but finite non-zero results may differ from hardware by up to
/// `ulp`. NaN, +-0, +-inf and denormal results are still compared bit-exactly.
fn check_vop1_f32_ulp(op: u32, ulp: i64, cases: &[(u64, u32)]) {
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
    /// VSRC1 is a bare VGPR index in this format, so only a value goes here.
    vsrc1: u64,
    expected: u32,
}

/// A VOP2 case whose result is 64 bits wide.
struct Vop2F64 {
    src0: Src,
    vsrc1: u64,
    expected: u64,
}

/// A VOP2 case for the carry and select forms, which read VCC and may write it.
struct Vop2Vcc {
    src0: Src,
    vsrc1: u64,
    vcc_in: u32,
    expected: u32,
    expected_vcc: u32,
}

/// Bit-exact comparison of a VOP2 f32 instruction against captured hardware.
fn check_vop2_f32(op: u32, cases: &[Vop2F32]) {
    let harness = Harness::vop2();

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut src = vec![0u32; LANES * harness.src_stride];
        let mut uni = vec![0u32; 8];
        let mut literal = Vec::new();
        let field = match case.src0 {
            Src::Vgpr(value) => {
                for lane in 0..LANES {
                    src[lane * harness.src_stride] = value as u32;
                    src[lane * harness.src_stride + 1] = (value >> 32) as u32;
                }
                vgpr(0)
            }
            Src::Sgpr(value) => {
                uni[0] = value as u32;
                uni[1] = (value >> 32) as u32;
                10
            }
            Src::Inline(encoding) => encoding,
            Src::Literal(value) => {
                literal.push(value as u32);
                255
            }
        };
        for lane in 0..LANES {
            src[lane * harness.src_stride + 2] = case.vsrc1 as u32;
            src[lane * harness.src_stride + 3] = (case.vsrc1 >> 32) as u32;
        }
        let mut words = vec![vop2(op, 6, 2, field)];
        words.extend(literal);

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = harness.run(engine, &words, &src, &uni)[0];
            if got == case.expected {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} vsrc1={} hardware={} simulator={}",
                engine_name(engine),
                i,
                show_f32(case.vsrc1 as u32),
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

/// One VOP3 case with 64-bit operands.
struct Vop3F64 {
    src0: Src,
    src1: Src,
    src2: Src,
    abs: u32,
    neg: u32,
    clamp: bool,
    omod: u32,
    expected: u64,
}

/// Bit-exact comparison of a VOP1 instruction whose result is an integer or a
/// packed half, against captured hardware.
fn check_vop1_u32(op: u32, cases: &[(u64, u32)]) {
    check_vop1_f32_ulp(op, 0, cases);
}

/// Bit-exact comparison of a VOP3 f64 instruction against captured hardware.
fn check_vop3_f64(op: u32, cases: &[Vop3F64]) {
    check_vop3_f64_ulp(op, 0, cases);
}

/// As above, with a tolerance on finite non-zero results.
fn check_vop3_f64_ulp(op: u32, ulp: i128, cases: &[Vop3F64]) {
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
            let special = is_nan_f64(case.expected)
                || is_nan_f64(got)
                || is_zero_f64(case.expected)
                || is_zero_f64(got)
                || is_inf_f64(case.expected)
                || is_inf_f64(got)
                || is_denorm_f64(case.expected)
                || is_denorm_f64(got);
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
        failures.len(), cases.len() * 2, failures.join("\n"),
    );
}

/// Bit-exact comparison of a VOP3 instruction with an integer or packed-half
/// result, against captured hardware.
fn check_vop3_u32(op: u32, cases: &[Vop3F32]) {
    check_vop3_f32_ulp(op, 0, cases);
}

/// Integer results with a tolerance: the manual grants "1ULP accuracy" to the
/// float-to-integer conversions, which means the integer may be off by one --
/// a plain difference, not a float ULP distance.
fn check_vop1_u32_ulp(op: u32, tolerance: i64, cases: &[(u64, u32)]) {
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
        failures.len(), cases.len() * 2, failures.join("\n"),
    );
}

/// As above, for the VOP3 encoding.
fn check_vop3_u32_ulp(op: u32, tolerance: i64, cases: &[Vop3F32]) {
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
        failures.len(), cases.len() * 2, failures.join("\n"),
    );
}


/// Bit-exact comparison of a VOP2 f64 instruction against captured hardware.
fn check_vop2_f64(op: u32, cases: &[Vop2F64]) {
    let harness = Harness::vop2();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut src = vec![0u32; LANES * harness.src_stride];
        let mut uni = vec![0u32; 8];
        let mut literal = Vec::new();
        let field = match case.src0 {
            Src::Vgpr(value) => {
                for lane in 0..LANES {
                    src[lane * harness.src_stride] = value as u32;
                    src[lane * harness.src_stride + 1] = (value >> 32) as u32;
                }
                vgpr(0)
            }
            Src::Sgpr(value) => {
                uni[0] = value as u32;
                uni[1] = (value >> 32) as u32;
                10
            }
            Src::Inline(encoding) => encoding,
            Src::Literal(value) => {
                literal.push(value as u32);
                255
            }
        };
        for lane in 0..LANES {
            src[lane * harness.src_stride + 2] = case.vsrc1 as u32;
            src[lane * harness.src_stride + 3] = (case.vsrc1 >> 32) as u32;
        }
        let mut words = vec![vop2(op, 6, 2, field)];
        words.extend(literal);

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = harness.run(engine, &words, &src, &uni);
            let got = out[0] as u64 | ((out[1] as u64) << 32);
            if got == case.expected {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} vsrc1={} hardware={} simulator={}",
                engine_name(engine), i, show_f64(case.vsrc1),
                show_f64(case.expected), show_f64(got),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} case-results differ from hardware:\n{}",
        failures.len(), cases.len() * 2, failures.join("\n"),
    );
}

/// Bit-exact comparison of a VOP2 instruction with an integer result.
fn check_vop2_u32(op: u32, cases: &[Vop2F32]) {
    check_vop2_f32(op, cases);
}

/// V_FMAMK_F32 and V_FMAAK_F32 embed a 32-bit constant after the instruction
/// rather than taking it through an operand field, so they get their own shape.
struct Vop2Literal {
    src0: Src,
    vsrc1: u64,
    k: u32,
    expected: u32,
}

fn check_vop2_literal_f32(op: u32, cases: &[Vop2Literal]) {
    let harness = Harness::vop2();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut src = vec![0u32; LANES * harness.src_stride];
        let mut uni = vec![0u32; 8];
        let field = match case.src0 {
            Src::Vgpr(value) => {
                for lane in 0..LANES {
                    src[lane * harness.src_stride] = value as u32;
                }
                vgpr(0)
            }
            Src::Sgpr(value) => {
                uni[0] = value as u32;
                10
            }
            Src::Inline(encoding) => encoding,
            Src::Literal(_) => unreachable!("the constant is already the literal here"),
        };
        for lane in 0..LANES {
            src[lane * harness.src_stride + 2] = case.vsrc1 as u32;
        }
        let words = [vop2(op, 6, 2, field), case.k];

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = harness.run(engine, &words, &src, &uni)[0];
            if got == case.expected {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} k={} hardware={} simulator={}",
                engine_name(engine), i, show_f32(case.k),
                show_f32(case.expected), show_f32(got),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} case-results differ from hardware:\n{}",
        failures.len(), cases.len() * 2, failures.join("\n"),
    );
}

/// The carry and select forms read VCC, and the carry forms write it back, so
/// both the vector result and VCC are compared.
fn check_vop2_vcc(op: u32, cases: &[Vop2Vcc]) {
    let harness = Harness::vop2();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut src = vec![0u32; LANES * harness.src_stride];
        let mut uni = vec![0u32; 8];
        let field = match case.src0 {
            Src::Vgpr(value) => {
                for lane in 0..LANES {
                    src[lane * harness.src_stride] = value as u32;
                }
                vgpr(0)
            }
            Src::Sgpr(value) => {
                uni[0] = value as u32;
                10
            }
            Src::Inline(encoding) => encoding,
            Src::Literal(_) => unreachable!("literal sources are not used by these forms"),
        };
        for lane in 0..LANES {
            src[lane * harness.src_stride + 2] = case.vsrc1 as u32;
        }
        uni[4] = case.vcc_in;
        let words = [vop2(op, 6, 2, field)];

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = harness.run(engine, &words, &src, &uni);
            let (got, vcc) = (out[0], out[2]);
            if got == case.expected && vcc == case.expected_vcc {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} vcc_in=0x{:08X} hardware=(0x{:08X}, vcc 0x{:08X}) simulator=(0x{:08X}, vcc 0x{:08X})",
                engine_name(engine), i, case.vcc_in,
                case.expected, case.expected_vcc, got, vcc,
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} case-results differ from hardware:\n{}",
        failures.len(), cases.len() * 2, failures.join("\n"),
    );
}

// -------------------------------------------------------------------- tests
#[test]
fn v_bfrev_b32_vop1() {
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
fn v_bfrev_b32_vop3() {
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
fn v_ceil_f32_vop1() {
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
fn v_ceil_f32_vop3() {
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
fn v_clz_i32_u32_vop1() {
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
fn v_clz_i32_u32_vop3() {
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
fn v_cos_f32_vop1() {
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
fn v_cos_f32_vop3() {
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
fn v_cvt_f16_f32_vop1() {
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
fn v_cvt_f16_f32_vop3() {
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
fn v_cvt_f32_f16_vop1() {
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
fn v_cvt_f32_f16_vop3() {
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
fn v_cvt_f32_f64_vop1() {
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
fn v_cvt_f32_f64_vop3() {
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
fn v_cvt_f32_i32_vop1() {
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
fn v_cvt_f32_i32_vop3() {
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
fn v_cvt_f32_u32_vop1() {
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
fn v_cvt_f32_u32_vop3() {
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
fn v_cvt_f64_f32_vop1() {
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
fn v_cvt_f64_f32_vop3() {
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
fn v_cvt_f64_i32_vop1() {
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
fn v_cvt_f64_i32_vop3() {
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
fn v_cvt_f64_u32_vop1() {
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
fn v_cvt_f64_u32_vop3() {
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
fn v_cvt_i32_f32_vop1() {
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
fn v_cvt_i32_f32_vop3() {
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
fn v_cvt_i32_f64_vop1() {
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
fn v_cvt_i32_f64_vop3() {
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
fn v_cvt_u32_f32_vop1() {
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
fn v_cvt_u32_f32_vop3() {
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
fn v_cvt_u32_f64_vop1() {
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
fn v_cvt_u32_f64_vop3() {
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
fn v_exp_f32_vop1() {
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
fn v_exp_f32_vop3() {
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
fn v_floor_f32_vop1() {
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
fn v_floor_f32_vop3() {
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
fn v_floor_f64_vop1() {
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
fn v_floor_f64_vop3() {
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
fn v_fract_f64_vop1() {
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
fn v_fract_f64_vop3() {
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
fn v_frexp_exp_i32_f32_vop1() {
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
fn v_frexp_exp_i32_f32_vop3() {
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
fn v_frexp_exp_i32_f64_vop1() {
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
fn v_frexp_exp_i32_f64_vop3() {
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
fn v_frexp_mant_f32_vop1() {
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
fn v_frexp_mant_f32_vop3() {
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
fn v_frexp_mant_f64_vop1() {
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
fn v_frexp_mant_f64_vop3() {
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
fn v_log_f32_vop1() {
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
fn v_log_f32_vop3() {
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
fn v_mov_b32_vop1() {
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
fn v_mov_b32_vop3() {
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
fn v_not_b32_vop1() {
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
fn v_not_b32_vop3() {
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
fn v_rcp_f32_vop1() {
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
fn v_rcp_f32_vop3() {
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
fn v_rcp_f64_vop1() {
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
fn v_rcp_f64_vop3() {
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
fn v_rcp_iflag_f32_vop1() {
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
fn v_rcp_iflag_f32_vop3() {
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
fn v_rndne_f32_vop1() {
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
fn v_rndne_f32_vop3() {
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
fn v_rndne_f64_vop1() {
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
fn v_rndne_f64_vop3() {
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
fn v_rsq_f32_vop1() {
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
fn v_rsq_f32_vop3() {
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
fn v_rsq_f64_vop1() {
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
fn v_rsq_f64_vop3() {
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
fn v_sin_f32_vop1() {
    // V_SIN_F32 in the VOP1 encoding. No accuracy statement in the manual, so the pseudo
// code determines the result exactly.
    check_vop1_f32(
        53,
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
fn v_sin_f32_vop3() {
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
fn v_sqrt_f32_vop1() {
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
fn v_sqrt_f32_vop3() {
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
fn v_sqrt_f64_vop1() {
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
fn v_sqrt_f64_vop3() {
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
fn v_trunc_f32_vop1() {
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
fn v_trunc_f32_vop3() {
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
fn v_trunc_f64_vop1() {
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

#[test]
fn v_trunc_f64_vop3() {
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

#[test]
fn v_add_co_ci_u32_vop2() {
    // V_ADD_CO_CI_U32 reads VCC and, for the carry forms, writes it back. Both the
    // vector result and VCC are compared. vcc_in covers all lanes off, all on,
    // and a mixed mask.
    check_vop2_vcc(
        32,
        &[
            Vop2Vcc { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0002, vcc_in: 0x0000_0000, expected: 0x0000_0003, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0002, vcc_in: 0xFFFF_FFFF, expected: 0x0000_0004, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0x0000_0000, expected: 0x0000_0000, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0xFFFF_FFFF, expected: 0x0000_0001, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0000, vcc_in: 0x0000_0000, expected: 0x0000_0000, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x8000_0000, vcc_in: 0x0000_0000, expected: 0x0000_0000, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0005), vsrc1: 0x0000_0003, vcc_in: 0xAAAA_AAAA, expected: 0x0000_0008, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0x0000_0000, expected: 0x8000_0000, expected_vcc: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_add_f32_vop2() {
    // V_ADD_F32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f32(
        3,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // +0 in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // -0 in src0
            Vop2F32 { src0: Src::Vgpr(0x3F80_0000), vsrc1: 0x3FC0_0000, expected: 0x4020_0000 }, // 1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xBF80_0000), vsrc1: 0x3FC0_0000, expected: 0x3F00_0000 }, // -1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x3FC0_0000, expected: 0x7F80_0000 }, // +inf in src0
            Vop2F32 { src0: Src::Vgpr(0xFF80_0000), vsrc1: 0x3FC0_0000, expected: 0xFF80_0000 }, // -inf in src0
            Vop2F32 { src0: Src::Vgpr(0x7FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FA0_0000), vsrc1: 0x3FC0_0000, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // min denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x807F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // max -denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x0080_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // min normal in src0
            Vop2F32 { src0: Src::Vgpr(0x7F7F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x7F7F_FFFF }, // max normal in src0
            Vop2F32 { src0: Src::Vgpr(0x3F00_0000), vsrc1: 0x3FC0_0000, expected: 0x4000_0000 }, // 0.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x4040_0000 }, // 1.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4000_0000), vsrc1: 0x3FC0_0000, expected: 0x4060_0000 }, // 2.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xBF80_0000 }, // -2.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4049_0FDB), vsrc1: 0x3FC0_0000, expected: 0x4094_87EE }, // pi in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0000, expected: 0x3FC0_0000 }, // +0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x8000_0000, expected: 0x3FC0_0000 }, // -0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F80_0000, expected: 0x4020_0000 }, // 1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xBF80_0000, expected: 0x3F00_0000 }, // -1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F80_0000, expected: 0x7F80_0000 }, // +inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xFF80_0000, expected: 0xFF80_0000 }, // -inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FC0_0000, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FA0_0000, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0001, expected: 0x3FC0_0000 }, // min denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x807F_FFFF, expected: 0x3FC0_0000 }, // max -denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0080_0000, expected: 0x3FC0_0000 }, // min normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F7F_FFFF, expected: 0x7F7F_FFFF }, // max normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F00_0000, expected: 0x4000_0000 }, // 0.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x4040_0000 }, // 1.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4000_0000, expected: 0x4060_0000 }, // 2.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xC020_0000, expected: 0xBF80_0000 }, // -2.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4049_0FDB, expected: 0x4094_87EE }, // pi in src1
            Vop2F32 { src0: Src::Sgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xBF80_0000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(245), vsrc1: 0x3FC0_0000, expected: 0xBF00_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xBF80_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_add_f32_vop3() {
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
fn v_add_f64_vop2() {
    // V_ADD_F64 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f64(
        2,
        &[
            Vop2F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // +0 in src0
            Vop2F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -0 in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4004_0000_0000_0000 }, // 1.0 in src0
            Vop2F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FE0_0000_0000_0000 }, // -1.0 in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FF0_0000_0000_0000 }, // +inf in src0
            Vop2F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xFFF0_0000_0000_0000 }, // -inf in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src0
            Vop2F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // max -denorm in src0
            Vop2F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // min normal in src0
            Vop2F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src0
            Vop2F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4000_0000_0000_0000 }, // 0.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4008_0000_0000_0000 }, // 1.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x400C_0000_0000_0000 }, // 2.0 in src0
            Vop2F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xBFF0_0000_0000_0000 }, // -2.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4012_90FD_AA22_168C }, // pi in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0000_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // +0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x8000_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF0_0000_0000_0000, expected: 0x4004_0000_0000_0000 }, // 1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xBFF0_0000_0000_0000, expected: 0x3FE0_0000_0000_0000 }, // -1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF0_0000_0000_0000, expected: 0x7FF0_0000_0000_0000 }, // +inf in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xFFF0_0000_0000_0000, expected: 0xFFF0_0000_0000_0000 }, // -inf in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF8_0000_0000_0000, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF4_0000_0000_0000, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0000_0000_0000_0001, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x800F_FFFF_FFFF_FFFF, expected: 0x3FF8_0000_0000_0000 }, // max -denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0010_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // min normal in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FEF_FFFF_FFFF_FFFF, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FE0_0000_0000_0000, expected: 0x4000_0000_0000_0000 }, // 0.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4008_0000_0000_0000 }, // 1.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, expected: 0x400C_0000_0000_0000 }, // 2.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xC004_0000_0000_0000, expected: 0xBFF0_0000_0000_0000 }, // -2.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x4009_21FB_5444_2D18, expected: 0x4012_90FD_AA22_168C }, // pi in src1
            Vop2F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xBFF0_0000_0000_0000 }, // src0 from a sgpr
            Vop2F64 { src0: Src::Inline(245), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xBFE0_0000_0000_0000 }, // src0 from a inline
        ],
    );
}

#[test]
fn v_add_f64_vop3() {
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
fn v_add_nc_u32_vop2() {
    // V_ADD_NC_U32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        37,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0004 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0002 }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x8000_0003 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x8000_0002 }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0005 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0001_0002 }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEF2 }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0013 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_0102 }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0003 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0004 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0x0000_0002 }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x8000_0003 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x8000_0002 }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0005 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0001_0002 }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0xDEAD_BEF2 }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0013 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_0102 }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEF2 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0x0000_0002 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEF2 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_add_nc_u32_vop3() {
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
fn v_and_b32_vop2() {
    // V_AND_B32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        27,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0001 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0002 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0001 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0x0000_0003 }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x0000_0000 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x0000_0003 }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0002 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_0003 }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0x0000_0003 }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0000 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_0003 }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_and_b32_vop3() {
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
fn v_ashrrev_i32_vop2() {
    // V_ASHRREV_I32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        26,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0001 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0000 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0xF000_0000 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x0FFF_FFFF }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0000 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_1FFF }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0xFBD5_B7DD }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0002 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_001F }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_ashrrev_i32_vop3() {
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
fn v_cndmask_b32_vop2() {
    // V_CNDMASK_B32 reads VCC and, for the carry forms, writes it back. Both the
    // vector result and VCC are compared. vcc_in covers all lanes off, all on,
    // and a mixed mask.
    check_vop2_vcc(
        1,
        &[
            Vop2Vcc { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0002, vcc_in: 0x0000_0000, expected: 0x0000_0001, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0002, vcc_in: 0xFFFF_FFFF, expected: 0x0000_0002, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0x0000_0000, expected: 0xFFFF_FFFF, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0xFFFF_FFFF, expected: 0x0000_0001, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0000, vcc_in: 0x0000_0000, expected: 0x0000_0000, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x8000_0000, vcc_in: 0x0000_0000, expected: 0x8000_0000, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0005), vsrc1: 0x0000_0003, vcc_in: 0xAAAA_AAAA, expected: 0x0000_0005, expected_vcc: 0xAAAA_AAAA },
            Vop2Vcc { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0x0000_0000, expected: 0x7FFF_FFFF, expected_vcc: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_fmac_f32_vop2() {
    // V_FMAC_F32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f32(
        43,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3FC0_0000, expected: 0x0000_0000 }, // +0 in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x3FC0_0000, expected: 0x0000_0000 }, // -0 in src0
            Vop2F32 { src0: Src::Vgpr(0x3F80_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // 1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xBF80_0000), vsrc1: 0x3FC0_0000, expected: 0xBFC0_0000 }, // -1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x3FC0_0000, expected: 0x7F80_0000 }, // +inf in src0
            Vop2F32 { src0: Src::Vgpr(0xFF80_0000), vsrc1: 0x3FC0_0000, expected: 0xFF80_0000 }, // -inf in src0
            Vop2F32 { src0: Src::Vgpr(0x7FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FA0_0000), vsrc1: 0x3FC0_0000, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x3FC0_0000, expected: 0x0000_0002 }, // min denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x807F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x80BF_FFFE }, // max -denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x0080_0000), vsrc1: 0x3FC0_0000, expected: 0x00C0_0000 }, // min normal in src0
            Vop2F32 { src0: Src::Vgpr(0x7F7F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x7F80_0000 }, // max normal in src0
            Vop2F32 { src0: Src::Vgpr(0x3F00_0000), vsrc1: 0x3FC0_0000, expected: 0x3F40_0000 }, // 0.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x4010_0000 }, // 1.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4000_0000), vsrc1: 0x3FC0_0000, expected: 0x4040_0000 }, // 2.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC070_0000 }, // -2.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4049_0FDB), vsrc1: 0x3FC0_0000, expected: 0x4096_CBE4 }, // pi in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // +0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x8000_0000, expected: 0x0000_0000 }, // -0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F80_0000, expected: 0x3FC0_0000 }, // 1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xBF80_0000, expected: 0xBFC0_0000 }, // -1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F80_0000, expected: 0x7F80_0000 }, // +inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xFF80_0000, expected: 0xFF80_0000 }, // -inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FC0_0000, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FA0_0000, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0001, expected: 0x0000_0002 }, // min denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x807F_FFFF, expected: 0x80BF_FFFE }, // max -denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0080_0000, expected: 0x00C0_0000 }, // min normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F7F_FFFF, expected: 0x7F80_0000 }, // max normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F00_0000, expected: 0x3F40_0000 }, // 0.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x4010_0000 }, // 1.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4000_0000, expected: 0x4040_0000 }, // 2.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xC020_0000, expected: 0xC070_0000 }, // -2.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4049_0FDB, expected: 0x4096_CBE4 }, // pi in src1
            Vop2F32 { src0: Src::Sgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC070_0000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(245), vsrc1: 0x3FC0_0000, expected: 0xC040_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC070_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_fmac_f32_vop3() {
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
fn v_lshlrev_b32_vop2() {
    // V_LSHLREV_B32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        24,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0006 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0x8000_0000 }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x8000_0000 }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_000C }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x8000_0000 }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0001_8000 }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0003_0000 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x8000_0000 }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0008 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0xFFFF_FFF8 }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x0000_0000 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0xFFFF_FFF8 }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0010 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0007_FFF8 }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0xF56D_F778 }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0080 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_07F8 }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0001_8000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0x8000_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0001_8000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_lshlrev_b32_vop3() {
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
fn v_lshlrev_b64_vop2() {
    // V_LSHLREV_B64 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f64(
        31,
        &[
            Vop2F64 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 0 in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FF0_0000_0000_0000 }, // 1 in src0
            Vop2F64 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // -1 / UINT_MAX in src0
            Vop2F64 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // INT_MIN in src0
            Vop2F64 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // INT_MAX in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xFFE0_0000_0000_0000 }, // 2 in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // 0xFFFF in src0
            Vop2F64 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // 0xDEADBEEF in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // 16 in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // 255 in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // +0 in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // -0 in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x3FF0_0000_0000_0000, expected: 0xFF80_0000_0000_0000 }, // 1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xBFF0_0000_0000_0000, expected: 0xFF80_0000_0000_0000 }, // -1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FF0_0000_0000_0000, expected: 0xFF80_0000_0000_0000 }, // +inf in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFF0_0000_0000_0000, expected: 0xFF80_0000_0000_0000 }, // -inf in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FF8_0000_0000_0000, expected: 0xFFC0_0000_0000_0000 }, // qNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FF4_0000_0000_0000, expected: 0xFFA0_0000_0000_0000 }, // sNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000_0000_0001, expected: 0x0000_0000_0000_0008 }, // min denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x800F_FFFF_FFFF_FFFF, expected: 0x007F_FFFF_FFFF_FFF8 }, // max -denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0010_0000_0000_0000, expected: 0x0080_0000_0000_0000 }, // min normal in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FEF_FFFF_FFFF_FFFF, expected: 0xFF7F_FFFF_FFFF_FFF8 }, // max normal in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x3FE0_0000_0000_0000, expected: 0xFF00_0000_0000_0000 }, // 0.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xFFC0_0000_0000_0000 }, // 1.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x4000_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // 2.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xC004_0000_0000_0000, expected: 0x0020_0000_0000_0000 }, // -2.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x4009_21FB_5444_2D18, expected: 0x0049_0FDA_A221_68C0 }, // pi in src1
            Vop2F64 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // src0 from a sgpr
            Vop2F64 { src0: Src::Inline(193), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // src0 from a inline
            Vop2F64 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_lshlrev_b64_vop3() {
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
fn v_lshrrev_b32_vop2() {
    // V_LSHRREV_B32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        25,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0001 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0000 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0x1FFF_FFFF }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x1000_0000 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x0FFF_FFFF }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0000 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_1FFF }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0x1BD5_B7DD }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0002 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_001F }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_lshrrev_b32_vop3() {
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
fn v_max_i32_vop2() {
    // V_MAX_I32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        18,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x7FFF_FFFF }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_FFFF }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0010 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_00FF }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0003 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0003 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0x0000_0003 }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x0000_0003 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x7FFF_FFFF }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0003 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_FFFF }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0x0000_0003 }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0010 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_00FF }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_max_i32_vop3() {
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
fn v_max_num_f32_vop2() {
    // V_MAX_NUM_F32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f32(
        22,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // +0 in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // -0 in src0
            Vop2F32 { src0: Src::Vgpr(0x3F80_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // 1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xBF80_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // -1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x3FC0_0000, expected: 0x7F80_0000 }, // +inf in src0
            Vop2F32 { src0: Src::Vgpr(0xFF80_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // -inf in src0
            Vop2F32 { src0: Src::Vgpr(0x7FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // qNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FA0_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // sNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // min denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x807F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // max -denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x0080_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // min normal in src0
            Vop2F32 { src0: Src::Vgpr(0x7F7F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x7F7F_FFFF }, // max normal in src0
            Vop2F32 { src0: Src::Vgpr(0x3F00_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // 0.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // 1.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4000_0000), vsrc1: 0x3FC0_0000, expected: 0x4000_0000 }, // 2.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // -2.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4049_0FDB), vsrc1: 0x3FC0_0000, expected: 0x4049_0FDB }, // pi in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0000, expected: 0x3FC0_0000 }, // +0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x8000_0000, expected: 0x3FC0_0000 }, // -0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F80_0000, expected: 0x3FC0_0000 }, // 1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xBF80_0000, expected: 0x3FC0_0000 }, // -1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F80_0000, expected: 0x7F80_0000 }, // +inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xFF80_0000, expected: 0x3FC0_0000 }, // -inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FC0_0000, expected: 0x3FC0_0000 }, // qNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FA0_0000, expected: 0x3FC0_0000 }, // sNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0001, expected: 0x3FC0_0000 }, // min denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x807F_FFFF, expected: 0x3FC0_0000 }, // max -denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0080_0000, expected: 0x3FC0_0000 }, // min normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F7F_FFFF, expected: 0x7F7F_FFFF }, // max normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F00_0000, expected: 0x3FC0_0000 }, // 0.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // 1.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4000_0000, expected: 0x4000_0000 }, // 2.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xC020_0000, expected: 0x3FC0_0000 }, // -2.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4049_0FDB, expected: 0x4049_0FDB }, // pi in src1
            Vop2F32 { src0: Src::Sgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(245), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_max_num_f32_vop3() {
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
fn v_max_num_f64_vop2() {
    // V_MAX_NUM_F64 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f64(
        14,
        &[
            Vop2F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // +0 in src0
            Vop2F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -0 in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src0
            Vop2F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -1.0 in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FF0_0000_0000_0000 }, // +inf in src0
            Vop2F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -inf in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // qNaN in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // sNaN in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src0
            Vop2F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // max -denorm in src0
            Vop2F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // min normal in src0
            Vop2F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src0
            Vop2F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 0.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4000_0000_0000_0000 }, // 2.0 in src0
            Vop2F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -2.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4009_21FB_5444_2D18 }, // pi in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0000_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // +0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x8000_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF0_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xBFF0_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF0_0000_0000_0000, expected: 0x7FF0_0000_0000_0000 }, // +inf in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xFFF0_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -inf in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // qNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF4_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // sNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0000_0000_0000_0001, expected: 0x3FF8_0000_0000_0000 }, // min denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x800F_FFFF_FFFF_FFFF, expected: 0x3FF8_0000_0000_0000 }, // max -denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0010_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // min normal in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FEF_FFFF_FFFF_FFFF, expected: 0x7FEF_FFFF_FFFF_FFFF }, // max normal in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FE0_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 0.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, expected: 0x4000_0000_0000_0000 }, // 2.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xC004_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // -2.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x4009_21FB_5444_2D18, expected: 0x4009_21FB_5444_2D18 }, // pi in src1
            Vop2F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // src0 from a sgpr
            Vop2F64 { src0: Src::Inline(245), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // src0 from a inline
        ],
    );
}

#[test]
fn v_max_num_f64_vop3() {
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
fn v_max_u32_vop2() {
    // V_MAX_U32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        20,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x8000_0000 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x7FFF_FFFF }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_FFFF }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0010 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_00FF }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0003 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0003 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x8000_0000 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x7FFF_FFFF }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0003 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_FFFF }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0010 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_00FF }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEF }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFF }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEF }, // src0 from a literal
        ],
    );
}

#[test]
fn v_max_u32_vop3() {
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
fn v_min_i32_vop2() {
    // V_MIN_I32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        17,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0001 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x8000_0000 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0002 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0001 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x8000_0000 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x0000_0003 }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0002 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_0003 }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0003 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_0003 }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEF }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFF }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEF }, // src0 from a literal
        ],
    );
}

#[test]
fn v_min_i32_vop3() {
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
fn v_min_num_f32_vop2() {
    // V_MIN_NUM_F32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f32(
        21,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3FC0_0000, expected: 0x0000_0000 }, // +0 in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x3FC0_0000, expected: 0x8000_0000 }, // -0 in src0
            Vop2F32 { src0: Src::Vgpr(0x3F80_0000), vsrc1: 0x3FC0_0000, expected: 0x3F80_0000 }, // 1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xBF80_0000), vsrc1: 0x3FC0_0000, expected: 0xBF80_0000 }, // -1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // +inf in src0
            Vop2F32 { src0: Src::Vgpr(0xFF80_0000), vsrc1: 0x3FC0_0000, expected: 0xFF80_0000 }, // -inf in src0
            Vop2F32 { src0: Src::Vgpr(0x7FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // qNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FA0_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // sNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x3FC0_0000, expected: 0x0000_0001 }, // min denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x807F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x807F_FFFF }, // max -denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x0080_0000), vsrc1: 0x3FC0_0000, expected: 0x0080_0000 }, // min normal in src0
            Vop2F32 { src0: Src::Vgpr(0x7F7F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // max normal in src0
            Vop2F32 { src0: Src::Vgpr(0x3F00_0000), vsrc1: 0x3FC0_0000, expected: 0x3F00_0000 }, // 0.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // 1.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4000_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // 2.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC020_0000 }, // -2.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4049_0FDB), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // pi in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // +0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x8000_0000, expected: 0x8000_0000 }, // -0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F80_0000, expected: 0x3F80_0000 }, // 1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xBF80_0000, expected: 0xBF80_0000 }, // -1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F80_0000, expected: 0x3FC0_0000 }, // +inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xFF80_0000, expected: 0xFF80_0000 }, // -inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FC0_0000, expected: 0x3FC0_0000 }, // qNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FA0_0000, expected: 0x3FC0_0000 }, // sNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0001, expected: 0x0000_0001 }, // min denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x807F_FFFF, expected: 0x807F_FFFF }, // max -denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0080_0000, expected: 0x0080_0000 }, // min normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F7F_FFFF, expected: 0x3FC0_0000 }, // max normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F00_0000, expected: 0x3F00_0000 }, // 0.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // 1.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4000_0000, expected: 0x3FC0_0000 }, // 2.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xC020_0000, expected: 0xC020_0000 }, // -2.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4049_0FDB, expected: 0x3FC0_0000 }, // pi in src1
            Vop2F32 { src0: Src::Sgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC020_0000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(245), vsrc1: 0x3FC0_0000, expected: 0xC000_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC020_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_min_num_f32_vop3() {
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
fn v_min_num_f64_vop2() {
    // V_MIN_NUM_F64 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f64(
        13,
        &[
            Vop2F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // +0 in src0
            Vop2F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x8000_0000_0000_0000 }, // -0 in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF0_0000_0000_0000 }, // 1.0 in src0
            Vop2F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xBFF0_0000_0000_0000 }, // -1.0 in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // +inf in src0
            Vop2F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xFFF0_0000_0000_0000 }, // -inf in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // qNaN in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // sNaN in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0001 }, // min denorm in src0
            Vop2F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x800F_FFFF_FFFF_FFFF }, // max -denorm in src0
            Vop2F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0010_0000_0000_0000 }, // min normal in src0
            Vop2F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // max normal in src0
            Vop2F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FE0_0000_0000_0000 }, // 0.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 2.0 in src0
            Vop2F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xC004_0000_0000_0000 }, // -2.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // pi in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0000_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // +0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x8000_0000_0000_0000, expected: 0x8000_0000_0000_0000 }, // -0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF0_0000_0000_0000, expected: 0x3FF0_0000_0000_0000 }, // 1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xBFF0_0000_0000_0000, expected: 0xBFF0_0000_0000_0000 }, // -1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF0_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // +inf in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xFFF0_0000_0000_0000, expected: 0xFFF0_0000_0000_0000 }, // -inf in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // qNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF4_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // sNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0000_0000_0000_0001, expected: 0x0000_0000_0000_0001 }, // min denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x800F_FFFF_FFFF_FFFF, expected: 0x800F_FFFF_FFFF_FFFF }, // max -denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0010_0000_0000_0000, expected: 0x0010_0000_0000_0000 }, // min normal in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FEF_FFFF_FFFF_FFFF, expected: 0x3FF8_0000_0000_0000 }, // max normal in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FE0_0000_0000_0000, expected: 0x3FE0_0000_0000_0000 }, // 0.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 1.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 2.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xC004_0000_0000_0000, expected: 0xC004_0000_0000_0000 }, // -2.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x4009_21FB_5444_2D18, expected: 0x3FF8_0000_0000_0000 }, // pi in src1
            Vop2F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xC004_0000_0000_0000 }, // src0 from a sgpr
            Vop2F64 { src0: Src::Inline(245), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xC000_0000_0000_0000 }, // src0 from a inline
        ],
    );
}

#[test]
fn v_min_num_f64_vop3() {
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
fn v_min_u32_vop2() {
    // V_MIN_U32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        19,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0001 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0002 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0001 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0x0000_0003 }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x0000_0003 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x0000_0003 }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0002 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_0003 }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0x0000_0003 }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0003 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_0003 }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_min_u32_vop3() {
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
fn v_mul_f32_vop2() {
    // V_MUL_F32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f32(
        8,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3FC0_0000, expected: 0x0000_0000 }, // +0 in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x3FC0_0000, expected: 0x8000_0000 }, // -0 in src0
            Vop2F32 { src0: Src::Vgpr(0x3F80_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // 1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xBF80_0000), vsrc1: 0x3FC0_0000, expected: 0xBFC0_0000 }, // -1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x3FC0_0000, expected: 0x7F80_0000 }, // +inf in src0
            Vop2F32 { src0: Src::Vgpr(0xFF80_0000), vsrc1: 0x3FC0_0000, expected: 0xFF80_0000 }, // -inf in src0
            Vop2F32 { src0: Src::Vgpr(0x7FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FA0_0000), vsrc1: 0x3FC0_0000, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x3FC0_0000, expected: 0x0000_0002 }, // min denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x807F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x80BF_FFFE }, // max -denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x0080_0000), vsrc1: 0x3FC0_0000, expected: 0x00C0_0000 }, // min normal in src0
            Vop2F32 { src0: Src::Vgpr(0x7F7F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x7F80_0000 }, // max normal in src0
            Vop2F32 { src0: Src::Vgpr(0x3F00_0000), vsrc1: 0x3FC0_0000, expected: 0x3F40_0000 }, // 0.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x4010_0000 }, // 1.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4000_0000), vsrc1: 0x3FC0_0000, expected: 0x4040_0000 }, // 2.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC070_0000 }, // -2.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4049_0FDB), vsrc1: 0x3FC0_0000, expected: 0x4096_CBE4 }, // pi in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // +0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x8000_0000, expected: 0x8000_0000 }, // -0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F80_0000, expected: 0x3FC0_0000 }, // 1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xBF80_0000, expected: 0xBFC0_0000 }, // -1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F80_0000, expected: 0x7F80_0000 }, // +inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xFF80_0000, expected: 0xFF80_0000 }, // -inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FC0_0000, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FA0_0000, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0001, expected: 0x0000_0002 }, // min denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x807F_FFFF, expected: 0x80BF_FFFE }, // max -denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0080_0000, expected: 0x00C0_0000 }, // min normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F7F_FFFF, expected: 0x7F80_0000 }, // max normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F00_0000, expected: 0x3F40_0000 }, // 0.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x4010_0000 }, // 1.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4000_0000, expected: 0x4040_0000 }, // 2.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xC020_0000, expected: 0xC070_0000 }, // -2.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4049_0FDB, expected: 0x4096_CBE4 }, // pi in src1
            Vop2F32 { src0: Src::Sgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC070_0000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(245), vsrc1: 0x3FC0_0000, expected: 0xC040_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC070_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_mul_f32_vop3() {
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
fn v_mul_f64_vop2() {
    // V_MUL_F64 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f64(
        6,
        &[
            Vop2F64 { src0: Src::Vgpr(0x0000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // +0 in src0
            Vop2F64 { src0: Src::Vgpr(0x8000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x8000_0000_0000_0000 }, // -0 in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src0
            Vop2F64 { src0: Src::Vgpr(0xBFF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xBFF8_0000_0000_0000 }, // -1.0 in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FF0_0000_0000_0000 }, // +inf in src0
            Vop2F64 { src0: Src::Vgpr(0xFFF0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xFFF0_0000_0000_0000 }, // -inf in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src0
            Vop2F64 { src0: Src::Vgpr(0x7FF4_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src0
            Vop2F64 { src0: Src::Vgpr(0x0000_0000_0000_0001), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0000_0000_0000_0002 }, // min denorm in src0
            Vop2F64 { src0: Src::Vgpr(0x800F_FFFF_FFFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x8017_FFFF_FFFF_FFFE }, // max -denorm in src0
            Vop2F64 { src0: Src::Vgpr(0x0010_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x0018_0000_0000_0000 }, // min normal in src0
            Vop2F64 { src0: Src::Vgpr(0x7FEF_FFFF_FFFF_FFFF), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x7FF0_0000_0000_0000 }, // max normal in src0
            Vop2F64 { src0: Src::Vgpr(0x3FE0_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x3FE8_0000_0000_0000 }, // 0.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4002_0000_0000_0000 }, // 1.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x4000_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4008_0000_0000_0000 }, // 2.0 in src0
            Vop2F64 { src0: Src::Vgpr(0xC004_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xC00E_0000_0000_0000 }, // -2.5 in src0
            Vop2F64 { src0: Src::Vgpr(0x4009_21FB_5444_2D18), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4012_D97C_7F33_21D2 }, // pi in src0
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0000_0000_0000_0000, expected: 0x0000_0000_0000_0000 }, // +0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x8000_0000_0000_0000, expected: 0x8000_0000_0000_0000 }, // -0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF0_0000_0000_0000, expected: 0x3FF8_0000_0000_0000 }, // 1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xBFF0_0000_0000_0000, expected: 0xBFF8_0000_0000_0000 }, // -1.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF0_0000_0000_0000, expected: 0x7FF0_0000_0000_0000 }, // +inf in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xFFF0_0000_0000_0000, expected: 0xFFF0_0000_0000_0000 }, // -inf in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF8_0000_0000_0000, expected: 0x7FF8_0000_0000_0000 }, // qNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FF4_0000_0000_0000, expected: 0x7FFC_0000_0000_0000 }, // sNaN in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0000_0000_0000_0001, expected: 0x0000_0000_0000_0002 }, // min denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x800F_FFFF_FFFF_FFFF, expected: 0x8017_FFFF_FFFF_FFFE }, // max -denorm in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x0010_0000_0000_0000, expected: 0x0018_0000_0000_0000 }, // min normal in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x7FEF_FFFF_FFFF_FFFF, expected: 0x7FF0_0000_0000_0000 }, // max normal in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FE0_0000_0000_0000, expected: 0x3FE8_0000_0000_0000 }, // 0.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0x4002_0000_0000_0000 }, // 1.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, expected: 0x4008_0000_0000_0000 }, // 2.0 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0xC004_0000_0000_0000, expected: 0xC00E_0000_0000_0000 }, // -2.5 in src1
            Vop2F64 { src0: Src::Vgpr(0x3FF8_0000_0000_0000), vsrc1: 0x4009_21FB_5444_2D18, expected: 0x4012_D97C_7F33_21D2 }, // pi in src1
            Vop2F64 { src0: Src::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xC00E_0000_0000_0000 }, // src0 from a sgpr
            Vop2F64 { src0: Src::Inline(245), vsrc1: 0x3FF8_0000_0000_0000, expected: 0xC008_0000_0000_0000 }, // src0 from a inline
        ],
    );
}

#[test]
fn v_mul_f64_vop3() {
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
fn v_mul_i32_i24_vop2() {
    // V_MUL_I32_I24 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        9,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFD }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFD }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0006 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0002_FFFD }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xFF09_3CCD }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0030 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_02FD }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0003 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0xFFFF_FFFD }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x0000_0000 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0xFFFF_FFFD }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0006 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0002_FFFD }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0xFF09_3CCD }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0030 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_02FD }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xFF09_3CCD }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFD }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xFF09_3CCD }, // src0 from a literal
        ],
    );
}

#[test]
fn v_mul_i32_i24_vop3() {
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
fn v_mul_u32_u24_vop2() {
    // V_MUL_U32_U24 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        11,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0x02FF_FFFD }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0000 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x02FF_FFFD }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0006 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0002_FFFD }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0209_3CCD }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0030 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_02FD }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0000 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0003 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0x02FF_FFFD }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x0000_0000 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x02FF_FFFD }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0006 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0002_FFFD }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0x0209_3CCD }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0030 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_02FD }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0209_3CCD }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0x02FF_FFFD }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x0209_3CCD }, // src0 from a literal
        ],
    );
}

#[test]
fn v_mul_u32_u24_vop3() {
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
fn v_or_b32_vop2() {
    // V_OR_B32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        28,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x8000_0003 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x7FFF_FFFF }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_FFFF }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0013 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_00FF }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0003 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0003 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0xFFFF_FFFF }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x8000_0003 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x7FFF_FFFF }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0003 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_FFFF }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0xDEAD_BEEF }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0013 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_00FF }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEF }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFF }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEF }, // src0 from a literal
        ],
    );
}

#[test]
fn v_or_b32_vop3() {
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
fn v_subrev_co_ci_u32_vop2() {
    // V_SUBREV_CO_CI_U32 reads VCC and, for the carry forms, writes it back. Both the
    // vector result and VCC are compared. vcc_in covers all lanes off, all on,
    // and a mixed mask.
    check_vop2_vcc(
        34,
        &[
            Vop2Vcc { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0002, vcc_in: 0x0000_0000, expected: 0x0000_0001, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0002, vcc_in: 0xFFFF_FFFF, expected: 0x0000_0000, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0x0000_0000, expected: 0x0000_0002, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0xFFFF_FFFF, expected: 0x0000_0001, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0000, vcc_in: 0x0000_0000, expected: 0x0000_0000, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x8000_0000, vcc_in: 0x0000_0000, expected: 0x0000_0000, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0005), vsrc1: 0x0000_0003, vcc_in: 0xAAAA_AAAA, expected: 0xFFFF_FFFE, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0x0000_0000, expected: 0x8000_0002, expected_vcc: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_subrev_f32_vop2() {
    // V_SUBREV_F32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f32(
        5,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // +0 in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // -0 in src0
            Vop2F32 { src0: Src::Vgpr(0x3F80_0000), vsrc1: 0x3FC0_0000, expected: 0x3F00_0000 }, // 1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xBF80_0000), vsrc1: 0x3FC0_0000, expected: 0x4020_0000 }, // -1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x3FC0_0000, expected: 0xFF80_0000 }, // +inf in src0
            Vop2F32 { src0: Src::Vgpr(0xFF80_0000), vsrc1: 0x3FC0_0000, expected: 0x7F80_0000 }, // -inf in src0
            Vop2F32 { src0: Src::Vgpr(0x7FC0_0000), vsrc1: 0x3FC0_0000, expected: 0xFFC0_0000 }, // qNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FA0_0000), vsrc1: 0x3FC0_0000, expected: 0xFFE0_0000 }, // sNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // min denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x807F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // max -denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x0080_0000), vsrc1: 0x3FC0_0000, expected: 0x3FC0_0000 }, // min normal in src0
            Vop2F32 { src0: Src::Vgpr(0x7F7F_FFFF), vsrc1: 0x3FC0_0000, expected: 0xFF7F_FFFF }, // max normal in src0
            Vop2F32 { src0: Src::Vgpr(0x3F00_0000), vsrc1: 0x3FC0_0000, expected: 0x3F80_0000 }, // 0.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x0000_0000 }, // 1.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4000_0000), vsrc1: 0x3FC0_0000, expected: 0xBF00_0000 }, // 2.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0x4080_0000 }, // -2.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4049_0FDB), vsrc1: 0x3FC0_0000, expected: 0xBFD2_1FB6 }, // pi in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0000, expected: 0xBFC0_0000 }, // +0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x8000_0000, expected: 0xBFC0_0000 }, // -0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F80_0000, expected: 0xBF00_0000 }, // 1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xBF80_0000, expected: 0xC020_0000 }, // -1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F80_0000, expected: 0x7F80_0000 }, // +inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xFF80_0000, expected: 0xFF80_0000 }, // -inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FC0_0000, expected: 0x7FC0_0000 }, // qNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FA0_0000, expected: 0x7FE0_0000 }, // sNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0001, expected: 0xBFC0_0000 }, // min denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x807F_FFFF, expected: 0xBFC0_0000 }, // max -denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0080_0000, expected: 0xBFC0_0000 }, // min normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F7F_FFFF, expected: 0x7F7F_FFFF }, // max normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F00_0000, expected: 0xBF80_0000 }, // 0.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x0000_0000 }, // 1.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4000_0000, expected: 0x3F00_0000 }, // 2.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xC020_0000, expected: 0xC080_0000 }, // -2.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4049_0FDB, expected: 0x3FD2_1FB6 }, // pi in src1
            Vop2F32 { src0: Src::Sgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0x4080_0000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(245), vsrc1: 0x3FC0_0000, expected: 0x4060_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0x4080_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_subrev_f32_vop3() {
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
fn v_subrev_nc_u32_vop2() {
    // V_SUBREV_NC_U32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        39,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0002 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_0004 }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x8000_0003 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x8000_0004 }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0001 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0xFFFF_0004 }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x2152_4114 }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0xFFFF_FFF3 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0xFFFF_FF04 }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0xFFFF_FFFD }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0xFFFF_FFFE }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0xFFFF_FFFC }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x7FFF_FFFD }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x7FFF_FFFC }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0xFFFF_FFFF }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_FFFC }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0xDEAD_BEEC }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_000D }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_00FC }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x2152_4114 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0x0000_0004 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0x2152_4114 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_subrev_nc_u32_vop3() {
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
fn v_sub_co_ci_u32_vop2() {
    // V_SUB_CO_CI_U32 reads VCC and, for the carry forms, writes it back. Both the
    // vector result and VCC are compared. vcc_in covers all lanes off, all on,
    // and a mixed mask.
    check_vop2_vcc(
        33,
        &[
            Vop2Vcc { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0002, vcc_in: 0x0000_0000, expected: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0002, vcc_in: 0xFFFF_FFFF, expected: 0xFFFF_FFFE, expected_vcc: 0xFFFF_FFFF },
            Vop2Vcc { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0x0000_0000, expected: 0xFFFF_FFFE, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0xFFFF_FFFF, expected: 0xFFFF_FFFD, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0000, vcc_in: 0x0000_0000, expected: 0x0000_0000, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x8000_0000, vcc_in: 0x0000_0000, expected: 0x0000_0000, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x0000_0005), vsrc1: 0x0000_0003, vcc_in: 0xAAAA_AAAA, expected: 0x0000_0002, expected_vcc: 0x0000_0000 },
            Vop2Vcc { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0001, vcc_in: 0x0000_0000, expected: 0x7FFF_FFFE, expected_vcc: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_sub_f32_vop2() {
    // V_SUB_F32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_f32(
        4,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3FC0_0000, expected: 0xBFC0_0000 }, // +0 in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x3FC0_0000, expected: 0xBFC0_0000 }, // -0 in src0
            Vop2F32 { src0: Src::Vgpr(0x3F80_0000), vsrc1: 0x3FC0_0000, expected: 0xBF00_0000 }, // 1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xBF80_0000), vsrc1: 0x3FC0_0000, expected: 0xC020_0000 }, // -1.0 in src0
            Vop2F32 { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x3FC0_0000, expected: 0x7F80_0000 }, // +inf in src0
            Vop2F32 { src0: Src::Vgpr(0xFF80_0000), vsrc1: 0x3FC0_0000, expected: 0xFF80_0000 }, // -inf in src0
            Vop2F32 { src0: Src::Vgpr(0x7FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x7FC0_0000 }, // qNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FA0_0000), vsrc1: 0x3FC0_0000, expected: 0x7FE0_0000 }, // sNaN in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x3FC0_0000, expected: 0xBFC0_0000 }, // min denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x807F_FFFF), vsrc1: 0x3FC0_0000, expected: 0xBFC0_0000 }, // max -denorm in src0
            Vop2F32 { src0: Src::Vgpr(0x0080_0000), vsrc1: 0x3FC0_0000, expected: 0xBFC0_0000 }, // min normal in src0
            Vop2F32 { src0: Src::Vgpr(0x7F7F_FFFF), vsrc1: 0x3FC0_0000, expected: 0x7F7F_FFFF }, // max normal in src0
            Vop2F32 { src0: Src::Vgpr(0x3F00_0000), vsrc1: 0x3FC0_0000, expected: 0xBF80_0000 }, // 0.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x0000_0000 }, // 1.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4000_0000), vsrc1: 0x3FC0_0000, expected: 0x3F00_0000 }, // 2.0 in src0
            Vop2F32 { src0: Src::Vgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC080_0000 }, // -2.5 in src0
            Vop2F32 { src0: Src::Vgpr(0x4049_0FDB), vsrc1: 0x3FC0_0000, expected: 0x3FD2_1FB6 }, // pi in src0
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0000, expected: 0x3FC0_0000 }, // +0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x8000_0000, expected: 0x3FC0_0000 }, // -0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F80_0000, expected: 0x3F00_0000 }, // 1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xBF80_0000, expected: 0x4020_0000 }, // -1.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F80_0000, expected: 0xFF80_0000 }, // +inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xFF80_0000, expected: 0x7F80_0000 }, // -inf in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FC0_0000, expected: 0xFFC0_0000 }, // qNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7FA0_0000, expected: 0xFFE0_0000 }, // sNaN in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0000_0001, expected: 0x3FC0_0000 }, // min denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x807F_FFFF, expected: 0x3FC0_0000 }, // max -denorm in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x0080_0000, expected: 0x3FC0_0000 }, // min normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x7F7F_FFFF, expected: 0xFF7F_FFFF }, // max normal in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3F00_0000, expected: 0x3F80_0000 }, // 0.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x3FC0_0000, expected: 0x0000_0000 }, // 1.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4000_0000, expected: 0xBF00_0000 }, // 2.0 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0xC020_0000, expected: 0x4080_0000 }, // -2.5 in src1
            Vop2F32 { src0: Src::Vgpr(0x3FC0_0000), vsrc1: 0x4049_0FDB, expected: 0xBFD2_1FB6 }, // pi in src1
            Vop2F32 { src0: Src::Sgpr(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC080_0000 }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(245), vsrc1: 0x3FC0_0000, expected: 0xC060_0000 }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xC020_0000), vsrc1: 0x3FC0_0000, expected: 0xC080_0000 }, // src0 from a literal
        ],
    );
}

#[test]
fn v_sub_f32_vop3() {
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
fn v_sub_nc_u32_vop2() {
    // V_SUB_NC_U32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        38,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFD }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFE }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFC }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x7FFF_FFFD }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x7FFF_FFFC }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFF }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_FFFC }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEC }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_000D }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_00FC }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0003 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0002 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0x0000_0004 }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x8000_0003 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x8000_0004 }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0001 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0xFFFF_0004 }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0x2152_4114 }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0xFFFF_FFF3 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0xFFFF_FF04 }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEC }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFC }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEC }, // src0 from a literal
        ],
    );
}

#[test]
fn v_sub_nc_u32_vop3() {
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
fn v_xor_b32_vop2() {
    // V_XOR_B32 in the VOP2 encoding. Bit-exact: the manual states 0.5ULP or no
    // tolerance at all, so the result is uniquely determined.
    check_vop2_u32(
        29,
        &[
            Vop2F32 { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x0000_0003, expected: 0x0000_0003 }, // 0 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0001), vsrc1: 0x0000_0003, expected: 0x0000_0002 }, // 1 in src0
            Vop2F32 { src0: Src::Vgpr(0xFFFF_FFFF), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFC }, // -1 / UINT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x8000_0000), vsrc1: 0x0000_0003, expected: 0x8000_0003 }, // INT_MIN in src0
            Vop2F32 { src0: Src::Vgpr(0x7FFF_FFFF), vsrc1: 0x0000_0003, expected: 0x7FFF_FFFC }, // INT_MAX in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0002), vsrc1: 0x0000_0003, expected: 0x0000_0001 }, // 2 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_FFFF), vsrc1: 0x0000_0003, expected: 0x0000_FFFC }, // 0xFFFF in src0
            Vop2F32 { src0: Src::Vgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEC }, // 0xDEADBEEF in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0010), vsrc1: 0x0000_0003, expected: 0x0000_0013 }, // 16 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_00FF), vsrc1: 0x0000_0003, expected: 0x0000_00FC }, // 255 in src0
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0000, expected: 0x0000_0003 }, // 0 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0001, expected: 0x0000_0002 }, // 1 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xFFFF_FFFF, expected: 0xFFFF_FFFC }, // -1 / UINT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x8000_0000, expected: 0x8000_0003 }, // INT_MIN in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x7FFF_FFFF, expected: 0x7FFF_FFFC }, // INT_MAX in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0002, expected: 0x0000_0001 }, // 2 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_FFFF, expected: 0x0000_FFFC }, // 0xFFFF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0xDEAD_BEEF, expected: 0xDEAD_BEEC }, // 0xDEADBEEF in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_0010, expected: 0x0000_0013 }, // 16 in src1
            Vop2F32 { src0: Src::Vgpr(0x0000_0003), vsrc1: 0x0000_00FF, expected: 0x0000_00FC }, // 255 in src1
            Vop2F32 { src0: Src::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEC }, // src0 from a sgpr
            Vop2F32 { src0: Src::Inline(193), vsrc1: 0x0000_0003, expected: 0xFFFF_FFFC }, // src0 from a inline
            Vop2F32 { src0: Src::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, expected: 0xDEAD_BEEC }, // src0 from a literal
        ],
    );
}

#[test]
fn v_xor_b32_vop3() {
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

#[test]
fn v_fmamk_f32_vop2() {
    // V_FMAMK_F32 carries its constant in the dword after the instruction, so the
    // encoding is 8 bytes even though the operand fields name only registers.
    check_vop2_literal_f32(
        44,
        &[
            Vop2Literal { src0: Src::Vgpr(0x4000_0000), vsrc1: 0x3F80_0000, k: 0x3FC0_0000, expected: 0x4080_0000 },
            Vop2Literal { src0: Src::Vgpr(0x3F80_0000), vsrc1: 0x4000_0000, k: 0xBF80_0000, expected: 0x3F80_0000 },
            Vop2Literal { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3F80_0000, k: 0x3F80_0000, expected: 0x3F80_0000 },
            Vop2Literal { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x0000_0000, k: 0x3F80_0000, expected: 0x7F80_0000 },
            Vop2Literal { src0: Src::Vgpr(0xBF80_0000), vsrc1: 0xBF80_0000, k: 0x4000_0000, expected: 0xC040_0000 },
            Vop2Literal { src0: Src::Vgpr(0x7FA0_0000), vsrc1: 0x3F80_0000, k: 0x3F80_0000, expected: 0x7FE0_0000 },
        ],
    );
}

#[test]
fn v_fmaak_f32_vop2() {
    // V_FMAAK_F32 carries its constant in the dword after the instruction, so the
    // encoding is 8 bytes even though the operand fields name only registers.
    check_vop2_literal_f32(
        45,
        &[
            Vop2Literal { src0: Src::Vgpr(0x4000_0000), vsrc1: 0x3F80_0000, k: 0x3FC0_0000, expected: 0x4060_0000 },
            Vop2Literal { src0: Src::Vgpr(0x3F80_0000), vsrc1: 0x4000_0000, k: 0xBF80_0000, expected: 0x3F80_0000 },
            Vop2Literal { src0: Src::Vgpr(0x0000_0000), vsrc1: 0x3F80_0000, k: 0x3F80_0000, expected: 0x3F80_0000 },
            Vop2Literal { src0: Src::Vgpr(0x7F80_0000), vsrc1: 0x0000_0000, k: 0x3F80_0000, expected: 0xFFC0_0000 },
            Vop2Literal { src0: Src::Vgpr(0xBF80_0000), vsrc1: 0xBF80_0000, k: 0x4000_0000, expected: 0x4040_0000 },
            Vop2Literal { src0: Src::Vgpr(0x7FA0_0000), vsrc1: 0x3F80_0000, k: 0x3F80_0000, expected: 0x7FE0_0000 },
        ],
    );
}
