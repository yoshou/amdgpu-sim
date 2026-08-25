//! The scalar ALU formats: SOP1, SOP2, SOPC and SOPK.
//!
//! These write an SGPR, SCC, or both, and the SAVEEXEC forms write EXEC as
//! well. The harness reads all three back and drives SCC and EXEC beforehand,
//! because several of these instructions read what they also write.
//!
//! Operands here come from a 8-bit field, which -- unlike the vector formats --
//! cannot name a VGPR at all.

use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// The SGPR pair the harness gives these instructions as a destination.
const SDST: u32 = 16;

/// Where a scalar operand comes from. There is no VGPR case: the 8-bit field
/// has no encoding for one.
#[derive(Clone, Copy)]
pub(crate) enum SaluSrc {
    /// Value placed in the SGPR pair this harness assigns to the position.
    Sgpr(u64),
    /// Inline constant, named by its operand-field encoding.
    Inline(u32),
    /// Literal constant: operand field 255 plus a following dword.
    Literal(u32),
}

fn place(src: SaluSrc, position: usize, uni: &mut [u32], literal: &mut Vec<u32>) -> u32 {
    match src {
        SaluSrc::Sgpr(value) => {
            uni[position * 2] = value as u32;
            uni[position * 2 + 1] = (value >> 32) as u32;
            10 + position as u32 * 2
        }
        SaluSrc::Inline(encoding) => encoding,
        SaluSrc::Literal(value) => {
            literal.push(value);
            255
        }
    }
}

struct Observed {
    dst: u64,
    scc: u32,
    exec: u32,
}

fn run(harness: &Harness, engine: Engine, words: &[u32], uni: &[u32]) -> Observed {
    let src = vec![0u32; LANES * harness.src_stride];
    let out = harness.run(engine, words, &src, uni);
    Observed {
        dst: out[0] as u64 | ((out[1] as u64) << 32),
        scc: out[2],
        exec: out[3],
    }
}

fn report(failures: Vec<String>, total: usize) {
    assert!(
        failures.is_empty(),
        "{} of {} case-results differ from hardware:\n{}",
        failures.len(),
        total,
        failures.join("\n"),
    );
}

fn compare(
    engine: Engine,
    index: usize,
    got: &Observed,
    dst: u64,
    scc: u32,
    exec: u32,
    context: &str,
    failures: &mut Vec<String>,
) {
    if got.dst == dst && got.scc == scc && got.exec == exec {
        return;
    }
    failures.push(format!(
        "  {:<11} case {} {} hardware=(dst 0x{:016X}, scc {}, exec 0x{:08X}) simulator=(dst 0x{:016X}, scc {}, exec 0x{:08X})",
        engine_name(engine), index, context, dst, scc, exec, got.dst, got.scc, got.exec,
    ));
}

/// SOP1: one source, an SGPR destination.
pub(crate) struct Sop1Case {
    pub(crate) src0: SaluSrc,
    pub(crate) scc_in: u32,
    pub(crate) exec_in: u32,
    pub(crate) expected: u64,
    pub(crate) expected_scc: u32,
    pub(crate) expected_exec: u32,
}

pub(crate) fn check_sop1(op: u32, cases: &[Sop1Case]) {
    let harness = Harness::salu();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut uni = vec![0u32; 8];
        let mut literal = Vec::new();
        let field = place(case.src0, 0, &mut uni, &mut literal);
        uni[4] = case.scc_in;
        uni[5] = case.exec_in;
        let mut words = vec![sop1(op, SDST, field)];
        words.extend(literal);
        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = run(&harness, engine, &words, &uni);
            let ctx = format!("scc_in={} exec_in=0x{:08X}", case.scc_in, case.exec_in);
            compare(
                engine,
                i,
                &got,
                case.expected,
                case.expected_scc,
                case.expected_exec,
                &ctx,
                &mut failures,
            );
        }
    }
    report(failures, cases.len() * 2);
}

/// SOP2: two sources, an SGPR destination.
pub(crate) struct Sop2Case {
    pub(crate) src0: SaluSrc,
    pub(crate) src1: SaluSrc,
    pub(crate) scc_in: u32,
    pub(crate) expected: u64,
    pub(crate) expected_scc: u32,
}

pub(crate) fn check_sop2(op: u32, cases: &[Sop2Case]) {
    let harness = Harness::salu();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut uni = vec![0u32; 8];
        let mut literal = Vec::new();
        let f0 = place(case.src0, 0, &mut uni, &mut literal);
        let f1 = place(case.src1, 1, &mut uni, &mut literal);
        uni[4] = case.scc_in;
        uni[5] = 0xFFFF_FFFF;
        let mut words = vec![sop2(op, SDST, f1, f0)];
        words.extend(literal);
        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = run(&harness, engine, &words, &uni);
            let ctx = format!("scc_in={}", case.scc_in);
            compare(engine, i, &got, case.expected, case.expected_scc, 0xFFFF_FFFF,
                    &ctx, &mut failures);
        }
    }
    report(failures, cases.len() * 2);
}

/// SOPC: two sources and no destination field -- the result is SCC alone.
pub(crate) struct SopcCase {
    pub(crate) src0: SaluSrc,
    pub(crate) src1: SaluSrc,
    pub(crate) expected_scc: u32,
}

pub(crate) fn check_sopc(op: u32, cases: &[SopcCase]) {
    let harness = Harness::salu();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut uni = vec![0u32; 8];
        let mut literal = Vec::new();
        let f0 = place(case.src0, 0, &mut uni, &mut literal);
        let f1 = place(case.src1, 1, &mut uni, &mut literal);
        uni[5] = 0xFFFF_FFFF;
        let mut words = vec![sopc(op, f1, f0)];
        words.extend(literal);
        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = run(&harness, engine, &words, &uni);
            // The destination register must be left alone: this format has no
            // destination field, so a write there would be a decoding mistake.
            compare(engine, i, &got, 0, case.expected_scc, 0xFFFF_FFFF, "", &mut failures);
        }
    }
    report(failures, cases.len() * 2);
}

/// SOPK: a 16-bit immediate and an SGPR destination.
pub(crate) struct SopkCase {
    pub(crate) simm16: u32,
    /// The destination is read by S_CMOVK_I32 and S_MULK_I32, so it is seeded.
    pub(crate) dst_in: u32,
    pub(crate) scc_in: u32,
    pub(crate) expected: u64,
    pub(crate) expected_scc: u32,
}

pub(crate) fn check_sopk(op: u32, cases: &[SopkCase]) {
    let harness = Harness::salu();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut uni = vec![0u32; 8];
        uni[4] = case.scc_in;
        uni[5] = 0xFFFF_FFFF;
        // Seed the destination through a preceding s_mov, since these forms
        // read it.
        let words = vec![sop1(0, SDST, 255), case.dst_in, sopk(op, SDST, case.simm16)];
        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let got = run(&harness, engine, &words, &uni);
            let ctx = format!(
                "simm16=0x{:04X} dst_in=0x{:08X} scc_in={}",
                case.simm16, case.dst_in, case.scc_in
            );
            compare(
                engine,
                i,
                &got,
                case.expected,
                case.expected_scc,
                0xFFFF_FFFF,
                &ctx,
                &mut failures,
            );
        }
    }
    report(failures, cases.len() * 2);
}
#[test]
fn s_and_not1_saveexec_b32_sop1() {
    // S_AND_NOT1_SAVEEXEC_B32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        48,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_0000_FFFF, expected_scc: 1, expected_exec: 0xFFFF_0000 },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn s_and_saveexec_b32_sop1() {
    // S_AND_SAVEEXEC_B32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        32,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_0001 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_0001 },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x8000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x8000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x7FFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x7FFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_0002 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_0002 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xDEAD_BEEF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xDEAD_BEEF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_0010 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_0010 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_001F },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x0000_001F },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_0000_FFFF, expected_scc: 1, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xDEAD_BEEF },
        ],
    );
}

#[test]
fn s_ctz_i32_b32_sop1() {
    // S_CTZ_I32_B32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        8,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_001F, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_001F, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0001, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0001, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0004, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0004, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn s_cvt_f32_i32_sop1() {
    // S_CVT_F32_I32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        100,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_3F80_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_3F80_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_BF80_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_BF80_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_CF00_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_CF00_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F00_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F00_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_477F_FF00, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_477F_FF00, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_CE05_4904, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_CE05_4904, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4180_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4180_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_41F8_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_41F8_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_BF80_0000, expected_scc: 0, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_BF80_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_CE05_4904, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn s_cvt_f32_u32_sop1() {
    // S_CVT_F32_U32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        101,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_3F80_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_3F80_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F80_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F80_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F00_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F00_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F00_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F00_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_477F_FF00, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_477F_FF00, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F5E_ADBF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F5E_ADBF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4180_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4180_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_41F8_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_41F8_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_4F80_0000, expected_scc: 0, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F80_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_4F5E_ADBF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn s_cvt_i32_f32_sop1() {
    // S_CVT_I32_F32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        102,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_8000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_8000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_8000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn s_cvt_u32_f32_sop1() {
    // S_CVT_U32_F32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        103,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn s_mov_b32_sop1() {
    // S_MOV_B32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        0,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0001, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0001, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_8000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_8000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0002, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0002, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0010, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0010, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_001F, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_001F, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn s_mov_b64_sop1() {
    // S_MOV_B64.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        1,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0001, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0001, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x8000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x8000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x7FFF_FFFF_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x7FFF_FFFF_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0xDEAD_BEEF_CAFE_BABE, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0xDEAD_BEEF_CAFE_BABE, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0020, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0020, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn s_or_saveexec_b32_sop1() {
    // S_OR_SAVEEXEC_B32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        34,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_0000_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn s_sext_i32_i16_sop1() {
    // S_SEXT_I32_I16.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        15,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0001, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0001, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0000, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0002, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0002, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_BEEF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_BEEF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0010, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_0010, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_001F, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_0000_001F, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_FFFF },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_BEEF, expected_scc: 0, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn s_xor_saveexec_b32_sop1() {
    // S_XOR_SAVEEXEC_B32.
    // SOP1 writes an SGPR and may write SCC; the SAVEEXEC forms write EXEC
    // too, so all three are compared and EXEC is driven beforehand.
    check_sop1(
        36,
        &[
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFE },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0001), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFE },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x7FFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x8000_0000), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x7FFF_FFFF },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x8000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x8000_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFD },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0002), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFFD },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_0000 },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x2152_4110 },
            Sop1Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x2152_4110 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFEF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_0010), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFEF },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFE0 },
            Sop1Case { src0: SaluSrc::Sgpr(0x0000_001F), scc_in: 1, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0xFFFF_FFE0 },
            Sop1Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, exec_in: 0x0000_FFFF, expected: 0x0000_0000_0000_FFFF, expected_scc: 1, expected_exec: 0xFFFF_0000 },
            Sop1Case { src0: SaluSrc::Inline(193), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0, expected_exec: 0x0000_0000 },
            Sop1Case { src0: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, exec_in: 0xFFFF_FFFF, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1, expected_exec: 0x2152_4110 },
        ],
    );
}

#[test]
fn s_add_co_ci_u32_sop2() {
    // S_ADD_CO_CI_U32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        4,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0006, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0001_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0001_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_DEAD_BEF3, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0014, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0022, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0023, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0001_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_0022, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_add_co_i32_sop2() {
    // S_ADD_CO_I32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        2,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0001_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0001_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0013, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0022, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0022, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0001_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_0022, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_add_co_u32_sop2() {
    // S_ADD_CO_U32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        0,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0001_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0001_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0013, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0022, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0022, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0001_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_0022, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_add_nc_u64_sop2() {
    // S_ADD_NC_U64.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        83,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0004, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x8000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x8000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x8000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x8000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0001_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0001_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xDEAD_BEEF_CAFE_BAC1, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xDEAD_BEEF_CAFE_BAC1, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0023, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0023, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, expected: 0x8000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x8000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, expected: 0x0000_0001_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, expected: 0xDEAD_BEEF_CAFE_BAC1, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, expected: 0x0000_0000_0000_0023, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEF2, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_and_b32_sop2() {
    // S_AND_B32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        22,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_and_b64_sop2() {
    // S_AND_B64.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        23,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_and_not1_b32_sop2() {
    // S_AND_NOT1_B32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        34,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_7FFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_7FFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_001C, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_001C, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_ashr_i64_sop2() {
    // S_ASHR_I64.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        13,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xF000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xF000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_1FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_1FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFBD5_B7DD_F95F_D757, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xFBD5_B7DD_F95F_D757, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0004, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FBD5_B7DD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_bfe_u32_sop2() {
    // S_BFE_U32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        38,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_bfm_b32_sop2() {
    // S_BFM_B32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        42,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0008, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0008, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0018, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0018, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0003_FFF8, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0003_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0007_FFF8, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0007_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0007, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_000E, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_0000_0007, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_001C, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0003_8000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0007_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0003_FFF8, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0003_8000, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_cselect_b32_sop2() {
    // S_CSELECT_B32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        48,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_001F, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0000_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0010, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_001F, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_cselect_b64_sop2() {
    // S_CSELECT_B64.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        49,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x8000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x7FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xDEAD_BEEF_CAFE_BABE, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0020, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, expected: 0x8000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x7FFF_FFFF_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, expected: 0xDEAD_BEEF_CAFE_BABE, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, expected: 0x0000_0000_0000_0020, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_lshl_b32_sop2() {
    // S_LSHL_B32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        8,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0008, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0008, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0007_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0007_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_F56D_F778, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_F56D_F778, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0080, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0080, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_00F8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_00F8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0006, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_000C, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0001_8000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0003_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_F56D_F778, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0001_8000, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_lshl_b64_sop2() {
    // S_LSHL_B64.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        9,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0008, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0008, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xFFFF_FFFF_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xFFFF_FFFF_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0007_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0007_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xF56D_F77E_57F5_D5F0, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xF56D_F77E_57F5_D5F0, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0100, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0100, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0006, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x8000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x8000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, expected: 0x8000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, expected: 0xC000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, expected: 0x0000_0003_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFF8, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x8000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0006_F56D_F778, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0001_8000_0000_0000, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_lshr_b32_sop2() {
    // S_LSHR_B32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        10,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_1FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_1FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_1000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_1000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_1FFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_1FFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_1BD5_B7DD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_1BD5_B7DD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_1FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_1BD5_B7DD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_lshr_b64_sop2() {
    // S_LSHR_B64.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        11,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x1FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x1FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x1000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x1000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_1FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_1FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x1BD5_B7DD_F95F_D757, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x1BD5_B7DD_F95F_D757, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0004, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x1FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_1BD5_B7DD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_max_u32_sop2() {
    // S_MAX_U32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        21,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_001F, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_001F, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0000_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0010, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_001F, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_mul_hi_u32_sop2() {
    // S_MUL_HI_U32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        45,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_mul_i32_sop2() {
    // S_MUL_I32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        44,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_7FFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_7FFF_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0006, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0006, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0002_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0002_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_9C09_3CCD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_9C09_3CCD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0030, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0030, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_005D, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_005D, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_8000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_7FFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0006, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0002_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_9C09_3CCD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0030, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_005D, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_9C09_3CCD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_9C09_3CCD, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_mul_u64_sop2() {
    // S_MUL_U64.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        85,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xFFFF_FFFF_FFFF_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x8000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x8000_0000_0000_0000, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x7FFF_FFFF_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x7FFF_FFFF_FFFF_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0002_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0002_FFFF_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x9C09_3CCF_60FC_303A, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x9C09_3CCF_60FC_303A, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0060, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0060, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, expected: 0x8000_0000_0000_0000, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x7FFF_FFFF_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, expected: 0x0000_0002_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, expected: 0x9C09_3CCF_60FC_303A, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, expected: 0x0000_0000_0000_0060, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0002_9C09_3CCD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0002_9C09_3CCD, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_or_b32_sop2() {
    // S_OR_B32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        24,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0013, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_001F, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_001F, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0000_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_001F, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_or_b64_sop2() {
    // S_OR_B64.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        25,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x8000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x8000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x7FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x7FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xDEAD_BEEF_CAFE_BABF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xDEAD_BEEF_CAFE_BABF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0023, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0023, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, expected: 0x8000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x7FFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, expected: 0xDEAD_BEEF_CAFE_BABF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, expected: 0x0000_0000_0000_0023, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEF, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_or_not1_b32_sop2() {
    // S_OR_NOT1_B32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        36,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFE, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFE, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_7FFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_2152_4113, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_FFFF_FFEF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_FFFF_FFE3, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_2152_4113, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_sub_co_i32_sop2() {
    // S_SUB_CO_I32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        3,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFD, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFE, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFE, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_7FFF_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_7FFF_FFFD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_7FFF_FFFC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_7FFF_FFFC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_FFFC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_FFFC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_000D, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_000D, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_001C, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_001C, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_8000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_2152_4114, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_FFFF_FFF3, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_FFFF_FFE4, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_0000_0004, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 0 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_2152_4114, expected_scc: 0 },
        ],
    );
}

#[test]
fn s_xor_b32_sop2() {
    // S_XOR_B32.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        26,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_8000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_7FFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_7FFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0013, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_0000_001C, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 1, expected: 0x0000_0000_0000_001C, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), scc_in: 0, expected: 0x0000_0000_8000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), scc_in: 0, expected: 0x0000_0000_7FFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), scc_in: 0, expected: 0x0000_0000_0000_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), scc_in: 0, expected: 0x0000_0000_0000_0013, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), scc_in: 0, expected: 0x0000_0000_0000_001C, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_xor_b64_sop2() {
    // S_XOR_B64.
    // Both operand positions are swept, and SCC is driven beforehand because
    // the carry and select forms read it.
    check_sop2(
        27,
        &[
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xFFFF_FFFF_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x8000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x8000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x7FFF_FFFF_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x7FFF_FFFF_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xDEAD_BEEF_CAFE_BABD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0xDEAD_BEEF_CAFE_BABD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_0000_0023, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 1, expected: 0x0000_0000_0000_0023, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), scc_in: 0, expected: 0x0000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), scc_in: 0, expected: 0x0000_0000_0000_0002, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), scc_in: 0, expected: 0x8000_0000_0000_0003, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), scc_in: 0, expected: 0x7FFF_FFFF_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), scc_in: 0, expected: 0x0000_0000_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), scc_in: 0, expected: 0xDEAD_BEEF_CAFE_BABD, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), scc_in: 0, expected: 0x0000_0000_0000_0023, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Inline(193), scc_in: 0, expected: 0xFFFF_FFFF_FFFF_FFFC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
            Sop2Case { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Literal(0xDEAD_BEEF), scc_in: 0, expected: 0x0000_0000_DEAD_BEEC, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_cmp_eq_i32_sopc() {
    // S_CMP_EQ_I32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        0,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
        ],
    );
}

#[test]
fn s_cmp_eq_u32_sopc() {
    // S_CMP_EQ_U32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        6,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
        ],
    );
}

#[test]
fn s_cmp_eq_u64_sopc() {
    // S_CMP_EQ_U64.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        16,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 0 },
        ],
    );
}

#[test]
fn s_cmp_ge_i32_sopc() {
    // S_CMP_GE_I32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        3,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
        ],
    );
}

#[test]
fn s_cmp_ge_u32_sopc() {
    // S_CMP_GE_U32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        9,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
        ],
    );
}

#[test]
fn s_cmp_gt_i32_sopc() {
    // S_CMP_GT_I32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        2,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
        ],
    );
}

#[test]
fn s_cmp_gt_u32_sopc() {
    // S_CMP_GT_U32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        8,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
        ],
    );
}

#[test]
fn s_cmp_le_i32_sopc() {
    // S_CMP_LE_I32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        5,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
        ],
    );
}

#[test]
fn s_cmp_le_u32_sopc() {
    // S_CMP_LE_U32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        11,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
        ],
    );
}

#[test]
fn s_cmp_lg_i32_sopc() {
    // S_CMP_LG_I32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        1,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
        ],
    );
}

#[test]
fn s_cmp_lg_u32_sopc() {
    // S_CMP_LG_U32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        7,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
        ],
    );
}

#[test]
fn s_cmp_lg_u64_sopc() {
    // S_CMP_LG_U64.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        17,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0001), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000_0000_0000), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0020), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0001), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF_FFFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x8000_0000_0000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF_FFFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_FFFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF_CAFE_BABE), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000_0000_0003), src1: SaluSrc::Sgpr(0x0000_0000_0000_0020), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0000_0000_0003), expected_scc: 1 },
        ],
    );
}

#[test]
fn s_cmp_lt_i32_sopc() {
    // S_CMP_LT_I32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        4,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
        ],
    );
}

#[test]
fn s_cmp_lt_u32_sopc() {
    // S_CMP_LT_U32.
    // This format has no destination field, so the test also checks that the
    // destination register is left alone.
    check_sopc(
        10,
        &[
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0001), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0xFFFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x8000_0000), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x7FFF_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0002), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_FFFF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0010), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_001F), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0000), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0001), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xFFFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x8000_0000), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x7FFF_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0002), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_FFFF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0xDEAD_BEEF), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_0010), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Sgpr(0x0000_0003), src1: SaluSrc::Sgpr(0x0000_001F), expected_scc: 1 },
            SopcCase { src0: SaluSrc::Inline(193), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
            SopcCase { src0: SaluSrc::Literal(0xDEAD_BEEF), src1: SaluSrc::Sgpr(0x0000_0003), expected_scc: 0 },
        ],
    );
}

#[test]
fn s_cmovk_i32_sopk() {
    // S_CMOVK_I32.
    // The destination is seeded first, because these forms read it as well as
    // write it.
    check_sopk(
        2,
        &[
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_FFFF_8000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_FFFF_8000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_FFFF_8000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_7FFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_7FFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_7FFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_movk_i32_sopk() {
    // S_MOVK_I32.
    // The destination is seeded first, because these forms read it as well as
    // write it.
    check_sopk(
        0,
        &[
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_FFFF_8000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_FFFF_8000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_FFFF_8000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_FFFF_8000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_8000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_FFFF_8000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_7FFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_7FFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_7FFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_7FFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_0000_7FFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_7FFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0010, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0010, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_0000_0010, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_0010, expected_scc: 1 },
        ],
    );
}

#[test]
fn s_mulk_i32_sopk() {
    // S_MULK_I32.
    // The destination is seeded first, because these forms read it as well as
    // write it.
    check_sopk(
        16,
        &[
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0000, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0005, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_0005, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0001, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_FFFF_FFFF, expected_scc: 1 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_FFFF_FFFB, expected_scc: 0 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_FFFF_FFFB, expected_scc: 1 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_0000_0001, expected_scc: 0 },
            SopkCase { simm16: 0x0000_FFFF, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_0001, expected_scc: 1 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_FFFD_8000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_FFFD_8000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_0000_8000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_8000, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_0000_8000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0002_7FFB, expected_scc: 0 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0002_7FFB, expected_scc: 1 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_8001, expected_scc: 0 },
            SopkCase { simm16: 0x0000_7FFF, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_FFFF_8001, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0000, scc_in: 0, expected: 0x0000_0000_0000_0000, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0000, scc_in: 1, expected: 0x0000_0000_0000_0000, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0005, scc_in: 0, expected: 0x0000_0000_0000_0050, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0x0000_0005, scc_in: 1, expected: 0x0000_0000_0000_0050, expected_scc: 1 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0xFFFF_FFFF, scc_in: 0, expected: 0x0000_0000_FFFF_FFF0, expected_scc: 0 },
            SopkCase { simm16: 0x0000_0010, dst_in: 0xFFFF_FFFF, scc_in: 1, expected: 0x0000_0000_FFFF_FFF0, expected_scc: 1 },
        ],
    );
}
