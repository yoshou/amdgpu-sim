//! VOPC: two sources and a mask result.
//!
//! The VOPC encoding writes VCC, the VOP3 encoding writes a named SGPR, and the
//! CMPX forms write EXEC as well. The harness reads all three back, so a test
//! sees which of them the instruction actually touched -- writing the wrong one
//! is a failure mode a value comparison alone would miss.
//!
//! A compare is tested a whole wave at a time: the per-lane inputs are fixed and
//! the case states the mask the hardware produced. EXEC is varied too, because
//! an inactive lane must contribute a zero bit rather than its comparison.

use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// Per-lane src0 values, repeated to fill the wave. Chosen so that a compare
/// against a mid-range src1 splits the wave, and so that every special value is
/// on some lane.
pub(crate) const F32_LANES: [u64; 16] = [
    0x0000_0000, 0x8000_0000, 0x3F80_0000, 0xBF80_0000,
    0x7F80_0000, 0xFF80_0000, 0x7FC0_0000, 0x7FA0_0000,
    0x0000_0001, 0x807F_FFFF, 0x0080_0000, 0x7F7F_FFFF,
    0x3F00_0000, 0x3FC0_0000, 0x4000_0000, 0xC020_0000,
];

pub(crate) const F64_LANES: [u64; 16] = [
    0x0000_0000_0000_0000, 0x8000_0000_0000_0000, 0x3FF0_0000_0000_0000, 0xBFF0_0000_0000_0000,
    0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000, 0x7FF8_0000_0000_0000, 0x7FF4_0000_0000_0000,
    0x0000_0000_0000_0001, 0x800F_FFFF_FFFF_FFFF, 0x0010_0000_0000_0000, 0x7FEF_FFFF_FFFF_FFFF,
    0x3FE0_0000_0000_0000, 0x3FF8_0000_0000_0000, 0x4000_0000_0000_0000, 0xC004_0000_0000_0000,
];

pub(crate) const U32_LANES: [u64; 16] = [
    0x0000_0000, 0x0000_0001, 0xFFFF_FFFF, 0x8000_0000,
    0x7FFF_FFFF, 0x0000_0002, 0x0000_FFFF, 0xDEAD_BEEF,
    0x0000_0010, 0x0000_00FF, 0x0000_0003, 0x0000_0004,
    0xFFFF_FFFE, 0x8000_0001, 0x4000_0000, 0x0000_0005,
];

/// SRC0 is the only position in this format that takes the full operand field.
/// A VGPR there is what lets the wave see different values per lane; the other
/// classes are wave-uniform by construction.
#[derive(Clone, Copy)]
pub(crate) enum VopcSrc0 {
    /// Lane `i` takes `lanes[i % lanes.len()]` in the VGPR pair.
    Lanes(&'static [u64]),
    Sgpr(u64),
    Inline(u32),
    Literal(u32),
}

/// One compare, run over a whole wave.
pub(crate) struct VopcCase {
    pub(crate) src0: VopcSrc0,
    /// VSRC1 can only name a VGPR here, so only a value goes in it.
    pub(crate) vsrc1: u64,
    pub(crate) exec_in: u32,
    pub(crate) expected_vcc: u32,
    pub(crate) expected_exec: u32,
}

/// One compare in the VOP3 encoding, which names its own SGPR destination and
/// can apply abs and neg to either source.
pub(crate) struct Vopc3Case {
    pub(crate) src0: VopcSrc0,
    pub(crate) vsrc1: u64,
    pub(crate) abs: u32,
    pub(crate) neg: u32,
    pub(crate) exec_in: u32,
    pub(crate) expected_sdst: u32,
    pub(crate) expected_exec: u32,
}

/// The SGPR the harness gives the VOP3 compares as a destination.
const SDST: u32 = 16;

fn fill(
    harness: &Harness,
    src0: VopcSrc0,
    vsrc1: u64,
    uni: &mut [u32],
) -> (Vec<u32>, u32, Vec<u32>) {
    let mut src = vec![0u32; LANES * harness.src_stride];
    for lane in 0..LANES {
        src[lane * harness.src_stride + 2] = vsrc1 as u32;
        src[lane * harness.src_stride + 3] = (vsrc1 >> 32) as u32;
    }
    let mut literal = Vec::new();
    let field = match src0 {
        VopcSrc0::Lanes(values) => {
            for lane in 0..LANES {
                let value = values[lane % values.len()];
                src[lane * harness.src_stride] = value as u32;
                src[lane * harness.src_stride + 1] = (value >> 32) as u32;
            }
            vgpr(0)
        }
        VopcSrc0::Sgpr(value) => {
            uni[0] = value as u32;
            uni[1] = (value >> 32) as u32;
            10
        }
        VopcSrc0::Inline(encoding) => encoding,
        VopcSrc0::Literal(value) => {
            literal.push(value);
            255
        }
    };
    (src, field, literal)
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

/// A compare in the VOPC encoding, which writes VCC.
pub(crate) fn check_vopc(op: u32, cases: &[VopcCase]) {
    let harness = Harness::vopc();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut uni = vec![0u32; 8];
        let (src, field, literal) = fill(&harness, case.src0, case.vsrc1, &mut uni);
        uni[5] = case.exec_in;
        let mut words = vec![vopc(op, 2, field)];
        words.extend(literal);

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = harness.run(engine, &words, &src, &uni);
            let (vcc, exec) = (out[0], out[1]);
            if vcc == case.expected_vcc && exec == case.expected_exec {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} exec_in=0x{:08X} hardware=(vcc 0x{:08X}, exec 0x{:08X}) simulator=(vcc 0x{:08X}, exec 0x{:08X})",
                engine_name(engine), i, case.exec_in,
                case.expected_vcc, case.expected_exec, vcc, exec,
            ));
        }
    }
    report(failures, cases.len() * 2);
}

/// A compare in the VOP3 encoding, which writes the SGPR named in SDST.
pub(crate) fn check_vopc_vop3(op: u32, cases: &[Vopc3Case]) {
    let harness = Harness::vopc();
    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut uni = vec![0u32; 8];
        let (src, field, literal) = fill(&harness, case.src0, case.vsrc1, &mut uni);
        uni[5] = case.exec_in;
        let mut words = vop3_sdst(op, SDST, field, vgpr(2), case.abs, case.neg).to_vec();
        words.extend(literal);

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = harness.run(engine, &words, &src, &uni);
            let (exec, sdst) = (out[1], out[2]);
            if sdst == case.expected_sdst && exec == case.expected_exec {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} (abs={:#03b} neg={:#03b} exec_in=0x{:08X}) hardware=(sdst 0x{:08X}, exec 0x{:08X}) simulator=(sdst 0x{:08X}, exec 0x{:08X})",
                engine_name(engine), i, case.abs, case.neg, case.exec_in,
                case.expected_sdst, case.expected_exec, sdst, exec,
            ));
        }
    }
    report(failures, cases.len() * 2);
}
#[test]
fn v_cmpx_eq_f32_vopc() {
    // V_CMPX_EQ_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        146,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x4000_4000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0003_0003 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0003 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0002_0002 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_eq_f32_vop3() {
    // V_CMPX_EQ_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        146,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0003_0003 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0003 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_eq_f64_vopc() {
    // V_CMPX_EQ_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        162,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x4000_4000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0003_0003 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0003 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0002_0002 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_eq_f64_vop3() {
    // V_CMPX_EQ_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        162,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0003_0003 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0003 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_eq_i64_vopc() {
    // V_CMPX_EQ_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        210,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x4000_4000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0001_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0040_0040 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0040 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_eq_i64_vop3() {
    // V_CMPX_EQ_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        210,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0001_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_eq_u32_vopc() {
    // V_CMPX_EQ_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        202,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0400_0400 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0001_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0004_0004 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0004 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_eq_u32_vop3() {
    // V_CMPX_EQ_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        202,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0400_0400 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0001_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0400_0400 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0400_0400 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_eq_u64_vopc() {
    // V_CMPX_EQ_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        218,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x4000_4000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0001_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0040_0040 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0040 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_eq_u64_vop3() {
    // V_CMPX_EQ_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        218,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0001_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4000_4000 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ge_f32_vopc() {
    // V_CMPX_GE_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        150,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x4810_4810 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0010 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0800_0800 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7D17_7D17 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0017 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2802_2802 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_ge_f32_vop3() {
    // V_CMPX_GE_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        150,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4810_4810 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0010 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7D17_7D17 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0017 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xC830_C830 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4810_4810 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7F1F_7F1F }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ge_f64_vopc() {
    // V_CMPX_GE_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        166,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x4810_4810 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0010 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0800_0800 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7D17_7D17 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0017 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2802_2802 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_ge_f64_vop3() {
    // V_CMPX_GE_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        166,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4810_4810 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0010 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7D17_7D17 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0017 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xC830_C830 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x4810_4810 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7F1F_7F1F }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ge_i64_vopc() {
    // V_CMPX_GE_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        214,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x48D0_48D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0880_0880 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7DD5_7DD5 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D5 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2880_2880 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0040_0040 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0040 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_ge_i64_vop3() {
    // V_CMPX_GE_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        214,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x48D0_48D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7DD5_7DD5 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D5 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xC8F0_C8F0 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x48D0_48D0 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFDF5_FDF5 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ge_u32_vopc() {
    // V_CMPX_GE_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        206,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFDC_FFDC },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00DC },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAA88_AA88 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0004_0004 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0004 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_ge_u32_vop3() {
    // V_CMPX_GE_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        206,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFDC_FFDC }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00DC }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xDFD4_DFD4 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFDC_FFDC }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xDFF7_DFF7 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x1084_1084 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ge_u64_vopc() {
    // V_CMPX_GE_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        222,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xCAFA_CAFA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x8AAA_8AAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x826A_826A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_006A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x822A_822A },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_ge_u64_vop3() {
    // V_CMPX_GE_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        222,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xCAFA_CAFA }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FA }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xC8F0_C8F0 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xCAFA_CAFA }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFDF5_FDF5 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_gt_f32_vopc() {
    // V_CMPX_GT_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        148,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0810_0810 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0010 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0800_0800 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7D14_7D14 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0014 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2800_2800 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_gt_f32_vop3() {
    // V_CMPX_GT_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        148,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0810_0810 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0010 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7D14_7D14 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0014 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8830_8830 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0810_0810 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7F1F_7F1F }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_gt_f64_vopc() {
    // V_CMPX_GT_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        164,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0810_0810 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0010 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0800_0800 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7D14_7D14 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0014 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2800_2800 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_gt_f64_vop3() {
    // V_CMPX_GT_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        164,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0810_0810 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0010 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7D14_7D14 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0014 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8830_8830 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0810_0810 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7F1F_7F1F }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_gt_i32_vopc() {
    // V_CMPX_GT_I32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        196,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xCB50_CB50 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0050 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x8A00_8A00 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xCF72_CF72 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0072 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x8A22_8A22 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xCF73_CF73 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0073 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x8A22_8A22 },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_gt_i32_vop3() {
    // V_CMPX_GT_I32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        196,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xCB50_CB50 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0050 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xCF72_CF72 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0072 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xDBD4_DBD4 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xCB50_CB50 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x1084_1084 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xDFF7_DFF7 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_gt_i64_vopc() {
    // V_CMPX_GT_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        212,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x08D0_08D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0880_0880 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7DD4_7DD4 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D4 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2880_2880 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_gt_i64_vop3() {
    // V_CMPX_GT_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        212,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x08D0_08D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7DD4_7DD4 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D4 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x88F0_88F0 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x08D0_08D0 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFDF5_FDF5 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_gt_u32_vopc() {
    // V_CMPX_GT_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        204,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFBDC_FBDC },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00DC },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAA88_AA88 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFE_FFFE },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FE },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_gt_u32_vop3() {
    // V_CMPX_GT_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        204,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFBDC_FBDC }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00DC }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFE_FFFE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xDBD4_DBD4 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFBDC_FBDC }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xDFF7_DFF7 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x1084_1084 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_gt_u64_vopc() {
    // V_CMPX_GT_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        220,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x8AFA_8AFA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x8AAA_8AAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFE_FFFE },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FE },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x822A_822A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x822A_822A },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_gt_u64_vop3() {
    // V_CMPX_GT_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        220,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8AFA_8AFA }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FA }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFE_FFFE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x88F0_88F0 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8AFA_8AFA }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFDF5_FDF5 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_le_f32_vopc() {
    // V_CMPX_LE_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        147,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xF72F_F72F },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002F },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA22A_A22A },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x822B_822B },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002B },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x822A_822A },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_le_f32_vop3() {
    // V_CMPX_LE_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        147,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF72F_F72F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x822B_822B }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002B }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x770F_770F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF72F_F72F }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7F1F_7F1F }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_le_f64_vopc() {
    // V_CMPX_LE_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        163,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xF72F_F72F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA22A_A22A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x822B_822B },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002B },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x822A_822A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_le_f64_vop3() {
    // V_CMPX_LE_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        163,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF72F_F72F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x822B_822B }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002B }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x770F_770F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF72F_F72F }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7F1F_7F1F }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_le_i64_vopc() {
    // V_CMPX_LE_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        211,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xF72F_F72F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA22A_A22A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x822B_822B },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002B },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x822A_822A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_le_i64_vop3() {
    // V_CMPX_LE_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        211,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF72F_F72F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x822B_822B }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002B }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x770F_770F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF72F_F72F }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x020A_020A }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_le_u32_vopc() {
    // V_CMPX_LE_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        203,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0423_0423 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0023 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0022_0022 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0001_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_le_u32_vop3() {
    // V_CMPX_LE_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        203,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0423_0423 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0023 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0001_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x242B_242B }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0423_0423 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x2008_2008 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xEF7B_EF7B }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_le_u64_vopc() {
    // V_CMPX_LE_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        219,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7505_7505 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0005 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2000_2000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0001_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0001 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7DD5_7DD5 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D5 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2880_2880 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_le_u64_vop3() {
    // V_CMPX_LE_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        219,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7505_7505 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0005 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0001_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0001 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x770F_770F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7505_7505 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x020A_020A }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_lg_f32_vopc() {
    // V_CMPX_LG_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        149,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xBF3F_BF3F },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_003F },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAA2A_AA2A },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFF3C_FF3C },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_003C },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAA28_AA28 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_lg_f32_vop3() {
    // V_CMPX_LG_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        149,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBF3F_BF3F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_003F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFF3C_FF3C }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_003C }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBF3F_BF3F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBF3F_BF3F }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFF3F_FF3F }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFF3F_FF3F }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_lg_f64_vopc() {
    // V_CMPX_LG_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        165,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xBF3F_BF3F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_003F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAA2A_AA2A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFF3C_FF3C },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_003C },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAA28_AA28 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_lg_f64_vop3() {
    // V_CMPX_LG_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        165,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBF3F_BF3F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_003F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFF3C_FF3C }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_003C }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBF3F_BF3F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBF3F_BF3F }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFF3F_FF3F }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFF3F_FF3F }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_lt_f32_vopc() {
    // V_CMPX_LT_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        145,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xB72F_B72F },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002F },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA22A_A22A },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x8228_8228 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0028 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x8228_8228 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_lt_f32_vop3() {
    // V_CMPX_LT_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        145,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB72F_B72F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8228_8228 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0028 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x370F_370F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB72F_B72F }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7F1F_7F1F }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_lt_f64_vopc() {
    // V_CMPX_LT_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        161,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xB72F_B72F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA22A_A22A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x8228_8228 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0028 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x8228_8228 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_lt_f64_vop3() {
    // V_CMPX_LT_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        161,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB72F_B72F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8228_8228 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0028 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x370F_370F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB72F_B72F }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7F1F_7F1F }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x8020_8020 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_lt_i32_vopc() {
    // V_CMPX_LT_I32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        193,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x30AF_30AF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00AF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x20AA_20AA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x308C_308C },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_008C },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2088_2088 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x3088_3088 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0088 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2088_2088 },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_lt_i32_vop3() {
    // V_CMPX_LT_I32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        193,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x30AF_30AF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00AF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x308C_308C }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_008C }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x202B_202B }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x30AF_30AF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xEF7B_EF7B }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x2008_2008 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_lt_i64_vopc() {
    // V_CMPX_LT_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        209,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xB72F_B72F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002F },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA22A_A22A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x822A_822A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_002A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x822A_822A },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFBF_FFBF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00BF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_lt_i64_vop3() {
    // V_CMPX_LT_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        209,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB72F_B72F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002F }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x822A_822A }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_002A }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x370F_370F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB72F_B72F }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x020A_020A }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_lt_u32_vopc() {
    // V_CMPX_LT_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        201,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0023_0023 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0023 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0022_0022 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFB_FFFB },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FB },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_lt_u32_vop3() {
    // V_CMPX_LT_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        201,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0023_0023 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0023 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x202B_202B }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0023_0023 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x2008_2008 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xEF7B_EF7B }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_lt_u64_vopc() {
    // V_CMPX_LT_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        217,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x3505_3505 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0005 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2000_2000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7D95_7D95 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0095 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2880_2880 },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_lt_u64_vop3() {
    // V_CMPX_LT_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        217,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x3505_3505 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0005 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_0000 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x370F_370F }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x3505_3505 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x020A_020A }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_neq_f32_vopc() {
    // V_CMPX_NEQ_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        157,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xBFFF_BFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFC_FFFC },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FC },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAA8_AAA8 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_neq_f32_vop3() {
    // V_CMPX_NEQ_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        157,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFC_FFFC }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FC }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_neq_f64_vopc() {
    // V_CMPX_NEQ_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        173,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xBFFF_BFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFC_FFFC },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FC },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAA8_AAA8 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_neq_f64_vop3() {
    // V_CMPX_NEQ_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        173,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFC_FFFC }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FC }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ne_i64_vopc() {
    // V_CMPX_NE_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        213,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xBFFF_BFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFE_FFFE },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FE },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFBF_FFBF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00BF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_ne_i64_vop3() {
    // V_CMPX_NE_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        213,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFE_FFFE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ne_u32_vopc() {
    // V_CMPX_NE_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        205,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFBFF_FBFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFE_FFFE },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FE },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFB_FFFB },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FB },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_ne_u32_vop3() {
    // V_CMPX_NE_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        205,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFBFF_FBFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFE_FFFE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFBFF_FBFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFBFF_FBFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ne_u64_vopc() {
    // V_CMPX_NE_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        221,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xBFFF_BFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFE_FFFE },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FE },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFBF_FFBF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00BF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_ne_u64_vop3() {
    // V_CMPX_NE_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        221,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFE_FFFE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FE }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xBFFF_BFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_nge_f32_vopc() {
    // V_CMPX_NGE_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        153,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xB7EF_B7EF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00EF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA2AA_A2AA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x82E8_82E8 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00E8 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x82A8_82A8 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_nge_f32_vop3() {
    // V_CMPX_NGE_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        153,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB7EF_B7EF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00EF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x82E8_82E8 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00E8 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x37CF_37CF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB7EF_B7EF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x80E0_80E0 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_nge_f64_vopc() {
    // V_CMPX_NGE_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        169,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xB7EF_B7EF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00EF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA2AA_A2AA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x82E8_82E8 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00E8 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x82A8_82A8 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_nge_f64_vop3() {
    // V_CMPX_NGE_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        169,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB7EF_B7EF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00EF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x82E8_82E8 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00E8 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x37CF_37CF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xB7EF_B7EF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x80E0_80E0 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ngt_f32_vopc() {
    // V_CMPX_NGT_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        155,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xF7EF_F7EF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00EF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA2AA_A2AA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x82EB_82EB },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00EB },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x82AA_82AA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_ngt_f32_vop3() {
    // V_CMPX_NGT_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        155,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF7EF_F7EF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00EF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x82EB_82EB }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00EB }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x77CF_77CF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF7EF_F7EF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x80E0_80E0 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_ngt_f64_vopc() {
    // V_CMPX_NGT_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        171,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xF7EF_F7EF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00EF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xA2AA_A2AA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x82EB_82EB },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00EB },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x82AA_82AA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmpx_ngt_f64_vop3() {
    // V_CMPX_NGT_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        171,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF7EF_F7EF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00EF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x82EB_82EB }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00EB }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x77CF_77CF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xF7EF_F7EF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x80E0_80E0 }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_nle_f32_vopc() {
    // V_CMPX_NLE_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        156,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x08D0_08D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0880_0880 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7DD4_7DD4 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D4 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2880_2880 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_nle_f32_vop3() {
    // V_CMPX_NLE_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        156,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x08D0_08D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7DD4_7DD4 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D4 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x88F0_88F0 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x08D0_08D0 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x80E0_80E0 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_nle_f64_vopc() {
    // V_CMPX_NLE_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        172,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x08D0_08D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0880_0880 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7DD4_7DD4 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D4 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2880_2880 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_nle_f64_vop3() {
    // V_CMPX_NLE_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        172,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x08D0_08D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7DD4_7DD4 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D4 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x88F0_88F0 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x08D0_08D0 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x80E0_80E0 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_nlt_f32_vopc() {
    // V_CMPX_NLT_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        158,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x48D0_48D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0880_0880 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7DD7_7DD7 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D7 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2882_2882 },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_nlt_f32_vop3() {
    // V_CMPX_NLT_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        158,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x48D0_48D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7DD7_7DD7 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D7 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xC8F0_C8F0 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x48D0_48D0 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x80E0_80E0 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmpx_nlt_f64_vopc() {
    // V_CMPX_NLT_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        174,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x48D0_48D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D0 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x0880_0880 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x7DD7_7DD7 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00D7 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0x2882_2882 },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_0000 },
        ],
    );
}

#[test]
fn v_cmpx_nlt_f64_vop3() {
    // V_CMPX_NLT_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        174,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x48D0_48D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D0 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7DD7_7DD7 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00D7 }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xC8F0_C8F0 }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x48D0_48D0 }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x80E0_80E0 }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0x7FDF_7FDF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_class_f32_vopc() {
    // V_CMP_CLASS_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        126,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_class_f32_vop3() {
    // V_CMP_CLASS_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        126,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_class_f64_vopc() {
    // V_CMP_CLASS_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        127,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_class_f64_vop3() {
    // V_CMP_CLASS_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        127,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_eq_f32_vopc() {
    // V_CMP_EQ_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        18,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x4000_4000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0003_0003, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0003, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0002_0002, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_eq_f32_vop3() {
    // V_CMP_EQ_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        18,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0003_0003, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0003, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_eq_f64_vopc() {
    // V_CMP_EQ_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        34,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x4000_4000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0003_0003, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0003, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0002_0002, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_eq_f64_vop3() {
    // V_CMP_EQ_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        34,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0003_0003, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0003, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_eq_i64_vopc() {
    // V_CMP_EQ_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        82,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x4000_4000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0001_0001, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0001, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0040_0040, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0040, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_eq_i64_vop3() {
    // V_CMP_EQ_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        82,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0001_0001, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0001, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_eq_u16_vopc() {
    // V_CMP_EQ_U16 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        58,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0400_0400, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x4009_4009, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0009, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0008_0008, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0054_0054, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0054, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_eq_u16_vop3() {
    // V_CMP_EQ_U16 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        58,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0400_0400, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4009_4009, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0009, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0400_0400, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0400_0400, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_eq_u32_vopc() {
    // V_CMP_EQ_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        74,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0400_0400, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0001_0001, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0001, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0004_0004, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0004, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_eq_u32_vop3() {
    // V_CMP_EQ_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        74,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0400_0400, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0001_0001, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0001, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0400_0400, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0400_0400, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_eq_u64_vopc() {
    // V_CMP_EQ_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        90,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x4000_4000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0001_0001, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0001, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0040_0040, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0040, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_eq_u64_vop3() {
    // V_CMP_EQ_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        90,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0001_0001, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0001, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4000_4000, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ge_f32_vopc() {
    // V_CMP_GE_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        22,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x4810_4810, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0010, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0800_0800, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7D17_7D17, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0017, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2802_2802, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ge_f32_vop3() {
    // V_CMP_GE_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        22,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4810_4810, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0010, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7D17_7D17, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0017, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xC830_C830, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4810_4810, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7F1F_7F1F, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ge_f64_vopc() {
    // V_CMP_GE_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        38,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x4810_4810, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0010, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0800_0800, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7D17_7D17, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0017, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2802_2802, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ge_f64_vop3() {
    // V_CMP_GE_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        38,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4810_4810, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0010, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7D17_7D17, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0017, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xC830_C830, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x4810_4810, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7F1F_7F1F, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ge_i64_vopc() {
    // V_CMP_GE_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        86,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x48D0_48D0, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D0, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0880_0880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7DD5_7DD5, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D5, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2880_2880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0040_0040, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0040, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ge_i64_vop3() {
    // V_CMP_GE_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        86,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x48D0_48D0, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D0, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7DD5_7DD5, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D5, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xC8F0_C8F0, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x48D0_48D0, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFDF5_FDF5, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ge_u32_vopc() {
    // V_CMP_GE_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        78,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFDC_FFDC, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00DC, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA88_AA88, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0004_0004, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0004, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ge_u32_vop3() {
    // V_CMP_GE_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        78,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFDC_FFDC, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00DC, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xDFD4_DFD4, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFDC_FFDC, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xDFF7_DFF7, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x1084_1084, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ge_u64_vopc() {
    // V_CMP_GE_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        94,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xCAFA_CAFA, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FA, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x8AAA_8AAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x826A_826A, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_006A, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x822A_822A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ge_u64_vop3() {
    // V_CMP_GE_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        94,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xCAFA_CAFA, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FA, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xC8F0_C8F0, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xCAFA_CAFA, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFDF5_FDF5, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_gt_f32_vopc() {
    // V_CMP_GT_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        20,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0810_0810, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0010, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0800_0800, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7D14_7D14, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0014, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2800_2800, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_gt_f32_vop3() {
    // V_CMP_GT_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        20,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0810_0810, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0010, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7D14_7D14, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0014, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8830_8830, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0810_0810, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7F1F_7F1F, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_gt_f64_vopc() {
    // V_CMP_GT_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        36,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0810_0810, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0010, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0800_0800, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7D14_7D14, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0014, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2800_2800, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_gt_f64_vop3() {
    // V_CMP_GT_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        36,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0810_0810, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0010, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7D14_7D14, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0014, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8830_8830, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0810_0810, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7F1F_7F1F, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_gt_i32_vopc() {
    // V_CMP_GT_I32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        68,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xCB50_CB50, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0050, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x8A00_8A00, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xCF72_CF72, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0072, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x8A22_8A22, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0xCF73_CF73, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0073, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x8A22_8A22, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_gt_i32_vop3() {
    // V_CMP_GT_I32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        68,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xCB50_CB50, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0050, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xCF72_CF72, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0072, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xDBD4_DBD4, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xCB50_CB50, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x1084_1084, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xDFF7_DFF7, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_gt_i64_vopc() {
    // V_CMP_GT_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        84,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x08D0_08D0, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D0, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0880_0880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7DD4_7DD4, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D4, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2880_2880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_gt_i64_vop3() {
    // V_CMP_GT_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        84,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x08D0_08D0, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D0, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7DD4_7DD4, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D4, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x88F0_88F0, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x08D0_08D0, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFDF5_FDF5, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_gt_u16_vopc() {
    // V_CMP_GT_U16 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        60,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x9BD4_9BD4, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D4, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x8A80_8A80, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xBFF6_BFF6, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00F6, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAA2_AAA2, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_gt_u16_vop3() {
    // V_CMP_GT_U16 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        60,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x9BD4_9BD4, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D4, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFF6_BFF6, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00F6, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x9BD4_9BD4, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x9BD4_9BD4, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x10D4_10D4, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_gt_u32_vopc() {
    // V_CMP_GT_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        76,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFBDC_FBDC, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00DC, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA88_AA88, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FE, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_gt_u32_vop3() {
    // V_CMP_GT_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        76,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFBDC_FBDC, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00DC, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FE, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xDBD4_DBD4, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFBDC_FBDC, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xDFF7_DFF7, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x1084_1084, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_gt_u64_vopc() {
    // V_CMP_GT_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        92,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x8AFA_8AFA, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FA, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x8AAA_8AAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FE, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x822A_822A, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002A, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x822A_822A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_gt_u64_vop3() {
    // V_CMP_GT_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        92,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8AFA_8AFA, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FA, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FE, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x88F0_88F0, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8AFA_8AFA, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFDF5_FDF5, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_le_f32_vopc() {
    // V_CMP_LE_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        19,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xF72F_F72F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA22A_A22A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x822B_822B, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002B, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x822A_822A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_le_f32_vop3() {
    // V_CMP_LE_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        19,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF72F_F72F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x822B_822B, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002B, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x770F_770F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF72F_F72F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7F1F_7F1F, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_le_f64_vopc() {
    // V_CMP_LE_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        35,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xF72F_F72F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA22A_A22A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x822B_822B, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002B, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x822A_822A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_le_f64_vop3() {
    // V_CMP_LE_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        35,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF72F_F72F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x822B_822B, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002B, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x770F_770F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF72F_F72F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7F1F_7F1F, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_le_i64_vopc() {
    // V_CMP_LE_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        83,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xF72F_F72F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA22A_A22A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x822B_822B, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002B, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x822A_822A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_le_i64_vop3() {
    // V_CMP_LE_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        83,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF72F_F72F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x822B_822B, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002B, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x770F_770F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF72F_F72F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x020A_020A, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_le_u32_vopc() {
    // V_CMP_LE_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        75,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0423_0423, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0023, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0022_0022, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0001_0001, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0001, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_le_u32_vop3() {
    // V_CMP_LE_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        75,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0423_0423, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0023, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0001_0001, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0001, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x242B_242B, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0423_0423, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x2008_2008, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xEF7B_EF7B, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_le_u64_vopc() {
    // V_CMP_LE_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        91,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7505_7505, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0005, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2000_2000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0001_0001, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0001, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7DD5_7DD5, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D5, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2880_2880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_le_u64_vop3() {
    // V_CMP_LE_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        91,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7505_7505, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0005, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0001_0001, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0001, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x770F_770F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7505_7505, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x020A_020A, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_lg_f32_vopc() {
    // V_CMP_LG_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        21,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xBF3F_BF3F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_003F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA2A_AA2A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFF3C_FF3C, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_003C, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA28_AA28, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_lg_f32_vop3() {
    // V_CMP_LG_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        21,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBF3F_BF3F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_003F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3C_FF3C, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_003C, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBF3F_BF3F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBF3F_BF3F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_lg_f64_vopc() {
    // V_CMP_LG_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        37,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xBF3F_BF3F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_003F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA2A_AA2A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFF3C_FF3C, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_003C, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA28_AA28, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_lg_f64_vop3() {
    // V_CMP_LG_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        37,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBF3F_BF3F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_003F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3C_FF3C, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_003C, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBF3F_BF3F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBF3F_BF3F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_lt_f32_vopc() {
    // V_CMP_LT_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        17,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xB72F_B72F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA22A_A22A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x8228_8228, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0028, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x8228_8228, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_lt_f32_vop3() {
    // V_CMP_LT_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        17,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB72F_B72F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8228_8228, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0028, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x370F_370F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB72F_B72F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7F1F_7F1F, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_lt_f64_vopc() {
    // V_CMP_LT_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        33,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xB72F_B72F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA22A_A22A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x8228_8228, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0028, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x8228_8228, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_lt_f64_vop3() {
    // V_CMP_LT_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        33,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB72F_B72F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8228_8228, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0028, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x370F_370F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB72F_B72F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7F1F_7F1F, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x8020_8020, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_lt_i32_vopc() {
    // V_CMP_LT_I32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        65,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x30AF_30AF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00AF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x20AA_20AA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x308C_308C, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_008C, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2088_2088, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0x3088_3088, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0088, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2088_2088, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_lt_i32_vop3() {
    // V_CMP_LT_I32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        65,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x30AF_30AF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00AF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x308C_308C, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_008C, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x202B_202B, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x30AF_30AF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xEF7B_EF7B, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x2008_2008, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_lt_i64_vopc() {
    // V_CMP_LT_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        81,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xB72F_B72F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA22A_A22A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x822A_822A, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_002A, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x822A_822A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFBF_FFBF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00BF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_lt_i64_vop3() {
    // V_CMP_LT_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        81,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB72F_B72F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x822A_822A, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_002A, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x370F_370F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB72F_B72F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x020A_020A, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_lt_u32_vopc() {
    // V_CMP_LT_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        73,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0023_0023, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0023, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0022_0022, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFB_FFFB, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FB, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_lt_u32_vop3() {
    // V_CMP_LT_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        73,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0023_0023, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0023, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x202B_202B, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0023_0023, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x2008_2008, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xEF7B_EF7B, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_lt_u64_vopc() {
    // V_CMP_LT_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        89,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x3505_3505, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0005, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2000_2000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7D95_7D95, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0095, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2880_2880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_lt_u64_vop3() {
    // V_CMP_LT_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        89,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x3505_3505, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0005, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x0000_0000, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_0000, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x370F_370F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x3505_3505, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x020A_020A, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_neq_f32_vopc() {
    // V_CMP_NEQ_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        29,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFC_FFFC, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FC, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAA8_AAA8, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_neq_f32_vop3() {
    // V_CMP_NEQ_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        29,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFC_FFFC, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FC, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_neq_f64_vopc() {
    // V_CMP_NEQ_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        45,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFC_FFFC, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FC, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAA8_AAA8, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_neq_f64_vop3() {
    // V_CMP_NEQ_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        45,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFC_FFFC, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FC, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ne_i64_vopc() {
    // V_CMP_NE_I64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        85,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FE, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFBF_FFBF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00BF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ne_i64_vop3() {
    // V_CMP_NE_I64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        85,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FE, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ne_u32_vopc() {
    // V_CMP_NE_U32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        77,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFBFF_FBFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FE, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFB_FFFB, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FB, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0xFFFF_FFFF, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(193), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xDEAD_BEEF), vsrc1: 0x0000_0003, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ne_u32_vop3() {
    // V_CMP_NE_U32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        77,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFBFF_FBFF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FE, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFBFF_FBFF, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFBFF_FBFF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&U32_LANES), vsrc1: 0x0000_0003, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ne_u64_vopc() {
    // V_CMP_NE_U64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        93,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FE, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFBF_FFBF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00BF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ne_u64_vop3() {
    // V_CMP_NE_U64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        93,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFE_FFFE, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00FE, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xBFFF_BFFF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_nge_f32_vopc() {
    // V_CMP_NGE_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        25,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xB7EF_B7EF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00EF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA2AA_A2AA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x82E8_82E8, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00E8, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x82A8_82A8, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_nge_f32_vop3() {
    // V_CMP_NGE_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        25,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB7EF_B7EF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00EF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x82E8_82E8, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00E8, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x37CF_37CF, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB7EF_B7EF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x80E0_80E0, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_nge_f64_vopc() {
    // V_CMP_NGE_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        41,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xB7EF_B7EF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00EF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA2AA_A2AA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x82E8_82E8, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00E8, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x82A8_82A8, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_nge_f64_vop3() {
    // V_CMP_NGE_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        41,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB7EF_B7EF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00EF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x82E8_82E8, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00E8, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x37CF_37CF, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xB7EF_B7EF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x80E0_80E0, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ngt_f32_vopc() {
    // V_CMP_NGT_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        27,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xF7EF_F7EF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00EF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA2AA_A2AA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x82EB_82EB, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00EB, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x82AA_82AA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ngt_f32_vop3() {
    // V_CMP_NGT_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        27,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF7EF_F7EF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00EF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x82EB_82EB, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00EB, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x77CF_77CF, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF7EF_F7EF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x80E0_80E0, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_ngt_f64_vopc() {
    // V_CMP_NGT_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        43,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xF7EF_F7EF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00EF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xA2AA_A2AA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x82EB_82EB, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00EB, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x82AA_82AA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_ngt_f64_vop3() {
    // V_CMP_NGT_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        43,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF7EF_F7EF, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00EF, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x82EB_82EB, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00EB, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x77CF_77CF, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xF7EF_F7EF, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x80E0_80E0, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_nle_f32_vopc() {
    // V_CMP_NLE_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        28,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x08D0_08D0, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D0, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0880_0880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7DD4_7DD4, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D4, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2880_2880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_nle_f32_vop3() {
    // V_CMP_NLE_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        28,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x08D0_08D0, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D0, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7DD4_7DD4, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D4, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x88F0_88F0, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x08D0_08D0, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x80E0_80E0, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_nle_f64_vopc() {
    // V_CMP_NLE_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        44,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x08D0_08D0, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D0, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0880_0880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7DD4_7DD4, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D4, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2880_2880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_nle_f64_vop3() {
    // V_CMP_NLE_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        44,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x08D0_08D0, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D0, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7DD4_7DD4, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D4, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x88F0_88F0, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x08D0_08D0, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x80E0_80E0, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_nlt_f32_vopc() {
    // V_CMP_NLT_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        30,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x48D0_48D0, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D0, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0880_0880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7DD7_7DD7, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D7, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2882_2882, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_nlt_f32_vop3() {
    // V_CMP_NLT_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        30,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x48D0_48D0, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D0, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7DD7_7DD7, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D7, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xC8F0_C8F0, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x48D0_48D0, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x80E0_80E0, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_nlt_f64_vopc() {
    // V_CMP_NLT_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        46,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x48D0_48D0, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D0, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0880_0880, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x7DD7_7DD7, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00D7, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x2882_2882, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_00FF, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAAAA_AAAA, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_nlt_f64_vop3() {
    // V_CMP_NLT_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        46,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x48D0_48D0, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D0, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7DD7_7DD7, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_00D7, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xC8F0_C8F0, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0x48D0_48D0, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0x80E0_80E0, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0x7FDF_7FDF, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_o_f32_vopc() {
    // V_CMP_O_F32 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        23,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_003F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA2A_AA2A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_003F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA2A_AA2A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x7FC0_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Literal(0xC020_0000), vsrc1: 0x4000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_o_f32_vop3() {
    // V_CMP_O_F32 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        23,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_003F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_003F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F32_LANES), vsrc1: 0x4000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}

#[test]
fn v_cmp_o_f64_vopc() {
    // V_CMP_O_F64 in the VOPC encoding, which writes VCC. EXEC is varied so that an
    // inactive lane is seen to contribute a zero bit rather than its comparison.
    check_vopc(
        39,
        &[
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_003F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA2A_AA2A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_003F, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0xAA2A_AA2A, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0x0000_0000, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0x0000_00FF, expected_vcc: 0x0000_0000, expected_exec: 0x0000_00FF },
            VopcCase { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x7FF8_0000_0000_0000, exec_in: 0xAAAA_AAAA, expected_vcc: 0x0000_0000, expected_exec: 0xAAAA_AAAA },
            VopcCase { src0: VopcSrc0::Sgpr(0xC004_0000_0000_0000), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
            VopcCase { src0: VopcSrc0::Inline(245), vsrc1: 0x4000_0000_0000_0000, exec_in: 0xFFFF_FFFF, expected_vcc: 0xFFFF_FFFF, expected_exec: 0xFFFF_FFFF },
        ],
    );
}

#[test]
fn v_cmp_o_f64_vop3() {
    // V_CMP_O_F64 in the VOP3 encoding, which writes the SGPR named in [7:0] instead
    // of VCC, and can apply abs and neg to either source.
    check_vopc_vop3(
        39,
        &[
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_003F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x0000_0000_0000_0000, abs: 0, neg: 0, exec_in: 0x0000_00FF, expected_sdst: 0x0000_003F, expected_exec: 0x0000_00FF }, // no modifiers
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 1, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // abs on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 2, neg: 0, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // abs on src1
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 1, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // neg on src0
            Vopc3Case { src0: VopcSrc0::Lanes(&F64_LANES), vsrc1: 0x4000_0000_0000_0000, abs: 0, neg: 2, exec_in: 0xFFFF_FFFF, expected_sdst: 0xFF3F_FF3F, expected_exec: 0xFFFF_FFFF }, // neg on src1
        ],
    );
}
