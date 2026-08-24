//! VOP2: two sources, of which VSRC1 can only name a VGPR. The carry and
//! select forms additionally read VCC, and the carry forms write it back.

use crate::compare::*;
use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// One VOP2 case. src0 takes the full 9-bit operand field; vsrc1 can only name
/// a VGPR, which is the format's own asymmetry.
pub(crate) struct Vop2F32 {
    src0: Src,
    /// VSRC1 is a bare VGPR index in this format, so only a value goes here.
    vsrc1: u64,
    expected: u32,
}

/// A VOP2 case whose result is 64 bits wide.
pub(crate) struct Vop2F64 {
    src0: Src,
    vsrc1: u64,
    expected: u64,
}

/// A VOP2 case for the carry and select forms, which read VCC and may write it.
pub(crate) struct Vop2Vcc {
    src0: Src,
    vsrc1: u64,
    vcc_in: u32,
    expected: u32,
    expected_vcc: u32,
}

/// Bit-exact comparison of a VOP2 f32 instruction against captured hardware.
pub(crate) fn check_vop2_f32(op: u32, cases: &[Vop2F32]) {
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

/// Bit-exact comparison of a VOP2 f64 instruction against captured hardware.
pub(crate) fn check_vop2_f64(op: u32, cases: &[Vop2F64]) {
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
pub(crate) fn check_vop2_u32(op: u32, cases: &[Vop2F32]) {
    check_vop2_f32(op, cases);
}

/// V_FMAMK_F32 and V_FMAAK_F32 embed a 32-bit constant after the instruction
/// rather than taking it through an operand field, so they get their own shape.
pub(crate) struct Vop2Literal {
    src0: Src,
    vsrc1: u64,
    k: u32,
    expected: u32,
}

pub(crate) fn check_vop2_literal_f32(op: u32, cases: &[Vop2Literal]) {
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
pub(crate) fn check_vop2_vcc(op: u32, cases: &[Vop2Vcc]) {
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

#[test]
pub(crate) fn v_add_co_ci_u32_vop2() {
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
pub(crate) fn v_add_f32_vop2() {
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
pub(crate) fn v_add_f64_vop2() {
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
pub(crate) fn v_add_nc_u32_vop2() {
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
pub(crate) fn v_and_b32_vop2() {
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
pub(crate) fn v_ashrrev_i32_vop2() {
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
pub(crate) fn v_cndmask_b32_vop2() {
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
pub(crate) fn v_fmac_f32_vop2() {
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
pub(crate) fn v_lshlrev_b32_vop2() {
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
pub(crate) fn v_lshlrev_b64_vop2() {
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
pub(crate) fn v_lshrrev_b32_vop2() {
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
pub(crate) fn v_max_i32_vop2() {
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
pub(crate) fn v_max_num_f32_vop2() {
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
pub(crate) fn v_max_num_f64_vop2() {
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
pub(crate) fn v_max_u32_vop2() {
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
pub(crate) fn v_min_i32_vop2() {
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
pub(crate) fn v_min_num_f32_vop2() {
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
pub(crate) fn v_min_num_f64_vop2() {
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
pub(crate) fn v_min_u32_vop2() {
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
pub(crate) fn v_mul_f32_vop2() {
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
pub(crate) fn v_mul_f64_vop2() {
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
pub(crate) fn v_mul_i32_i24_vop2() {
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
pub(crate) fn v_mul_u32_u24_vop2() {
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
pub(crate) fn v_or_b32_vop2() {
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
pub(crate) fn v_subrev_co_ci_u32_vop2() {
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
pub(crate) fn v_subrev_f32_vop2() {
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
pub(crate) fn v_subrev_nc_u32_vop2() {
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
pub(crate) fn v_sub_co_ci_u32_vop2() {
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
pub(crate) fn v_sub_f32_vop2() {
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
pub(crate) fn v_sub_nc_u32_vop2() {
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
pub(crate) fn v_xor_b32_vop2() {
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
pub(crate) fn v_fmamk_f32_vop2() {
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
pub(crate) fn v_fmaak_f32_vop2() {
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
