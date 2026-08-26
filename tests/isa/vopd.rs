//! VOPD, the dual-issue format: one instruction, two operations.
//!
//! Each opcode is tested in both halves, since an engine implements the two
//! independently. The harness fixes the registers the format's rules leave it
//! no choice about: the X half reads v0 and v2 and writes v6, the Y half reads
//! v1 and v3 and writes v7, which puts the two sources of each port in
//! different banks and makes one destination even and the other odd.

use crate::compare::*;
use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// One VOPD case. Both halves are stated, because one instruction runs both.
pub(crate) struct VopdCase {
    /// The X half's first source, which may name an SGPR or a constant.
    src0x: Src,
    /// The value in v2, the X half's second source.
    vsrc1x: u32,
    /// The Y half's first source.
    src0y: Src,
    /// The value in v3, the Y half's second source.
    vsrc1y: u32,
    /// What the destinations hold before the instruction, which the
    /// accumulating forms read.
    dstx_in: u32,
    dsty_in: u32,
    vcc_in: u32,
    expected_x: u32,
    expected_y: u32,
}

/// Bit-exact comparison of a VOPD pair against captured hardware.
pub(crate) fn check_vopd(opx: u32, opy: u32, literal: Option<u32>, cases: &[VopdCase]) {
    let harness = Harness::vopd();

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut src = vec![0u32; LANES * harness.src_stride];
        let mut uni = vec![0u32; 8];
        uni[4] = case.vcc_in;
        for lane in 0..LANES {
            src[lane * harness.src_stride + 2] = case.vsrc1x;
            src[lane * harness.src_stride + 3] = case.vsrc1y;
            src[lane * harness.src_stride + 4] = case.dstx_in;
            src[lane * harness.src_stride + 5] = case.dsty_in;
        }
        // The two halves take their first source from their own register, or
        // from an SGPR of their own: at most one each is what the format allows.
        let mut field = [0u32; 2];
        for (half, s) in [case.src0x, case.src0y].iter().enumerate() {
            field[half] = match s {
                Src::Vgpr(value) => {
                    for lane in 0..LANES {
                        src[lane * harness.src_stride + half] = *value as u32;
                    }
                    vgpr(half as u32)
                }
                Src::Sgpr(value) => {
                    uni[half * 2] = *value as u32;
                    10 + half as u32 * 2
                }
                Src::Inline(encoding) => *encoding,
                Src::Literal(_) => 255,
            };
        }
        let mut words = vopd(opx, opy, 6, 7, field[0], 2, field[1], 3).to_vec();
        if let Some(value) = literal {
            words.push(value);
        }

        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = harness.run(engine, &words, &src, &uni);
            let (got_x, got_y) = (out[0], out[1]);
            if got_x == case.expected_x && got_y == case.expected_y {
                continue;
            }
            failures.push(format!(
                "  {:<11} case {} hardware=(x {}, y {}) simulator=(x {}, y {})",
                engine_name(engine),
                i,
                show_f32(case.expected_x),
                show_f32(case.expected_y),
                show_f32(got_x),
                show_f32(got_y),
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


#[test]
fn v_dual_fmac_f32_x_vopd() {
    // V_DUAL_FMAC_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        0,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0002, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0100_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4080_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC0A0_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40C9_0FDB, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x8000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x3F80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4040_0000, expected_y: 0x3F80_0000 }, // 1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0xBF80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // -1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x7F80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0xFF80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in the destination
        ],
    );
}

#[test]
fn v_dual_fmac_f32_y_vopd() {
    // V_DUAL_FMAC_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        0,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0002 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0100_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4080_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC0A0_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40C9_0FDB }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x8000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x3F80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4040_0000 }, // 1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0xBF80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // -1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x7F80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0xFF80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in the destination
        ],
    );
}

#[test]
fn v_dual_fmaak_f32_x_vopd() {
    // V_DUAL_FMAAK_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    // The pair shares one literal constant, 0x40490fdb.
    check_vopd(
        1,
        8,
        Some(0x40490FDB),
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4049_0FDB, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4049_0FDB, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40A4_87EE, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F92_1FB6, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4049_0FDB, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4049_0FDB, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4084_87EE, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40E4_87EE, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBFED_E04A, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4116_CBE4, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4049_0FDB, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4049_0FDB, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40A4_87EE, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F92_1FB6, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40A4_87EE, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4084_87EE, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_fmaak_f32_y_vopd() {
    // V_DUAL_FMAAK_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    // The pair shares one literal constant, 0x40490fdb.
    check_vopd(
        8,
        1,
        Some(0x40490FDB),
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4049_0FDB }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4049_0FDB }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40A4_87EE }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F92_1FB6 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4049_0FDB }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4049_0FDB }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4084_87EE }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40E4_87EE }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBFED_E04A }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4116_CBE4 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4049_0FDB }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4049_0FDB }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40A4_87EE }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F92_1FB6 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40A4_87EE }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4084_87EE }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_fmamk_f32_x_vopd() {
    // V_DUAL_FMAMK_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    // The pair shares one literal constant, 0x40490fdb.
    check_vopd(
        2,
        8,
        Some(0x40490FDB),
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40A4_87EE, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBF92_1FB6, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4064_87EE, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4104_87EE, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC0BB_53D2, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x413D_E9E7, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40C9_0FDB, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40C9_0FDB, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40E9_0FDB, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40A9_0FDB, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40E9_0FDB, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4084_87EE, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_fmamk_f32_y_vopd() {
    // V_DUAL_FMAMK_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    // The pair shares one literal constant, 0x40490fdb.
    check_vopd(
        8,
        2,
        Some(0x40490FDB),
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40A4_87EE }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBF92_1FB6 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4064_87EE }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4104_87EE }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC0BB_53D2 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x413D_E9E7 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40C9_0FDB }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40C9_0FDB }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40E9_0FDB }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40A9_0FDB }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40E9_0FDB }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4084_87EE }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_mul_f32_x_vopd() {
    // V_DUAL_MUL_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        3,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x8000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0002, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0100_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4080_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC0A0_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40C9_0FDB, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x8000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_mul_f32_y_vopd() {
    // V_DUAL_MUL_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        3,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x8000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0002 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0100_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4080_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC0A0_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40C9_0FDB }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x8000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_add_f32_x_vopd() {
    // V_DUAL_ADD_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        4,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4040_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F7F_FFFF, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4020_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4080_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBF00_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40A4_87EE, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4040_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4040_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_add_f32_y_vopd() {
    // V_DUAL_ADD_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        4,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4040_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F7F_FFFF }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4020_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4080_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBF00_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40A4_87EE }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4040_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4040_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_sub_f32_x_vopd() {
    // V_DUAL_SUB_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        5,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBF80_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC040_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F7F_FFFF, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBFC0_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC090_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F92_1FB6, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4040_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFFC0_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFFE0_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_sub_f32_y_vopd() {
    // V_DUAL_SUB_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        5,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBF80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC040_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F7F_FFFF }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBFC0_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC090_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F92_1FB6 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4040_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFFC0_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFFE0_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_subrev_f32_x_vopd() {
    // V_DUAL_SUBREV_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        6,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4040_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFFC0_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFFE0_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF7F_FFFF, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3FC0_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4090_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBF92_1FB6, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBF80_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC040_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBF80_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_subrev_f32_y_vopd() {
    // V_DUAL_SUBREV_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        6,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4040_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFFC0_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFFE0_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF7F_FFFF }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3FC0_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4090_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBF92_1FB6 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBF80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC040_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBF80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_mul_dx9_zero_f32_x_vopd() {
    // V_DUAL_MUL_DX9_ZERO_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        7,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0002, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0100_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4080_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC0A0_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40C9_0FDB, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FC0_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7FE0_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_mul_dx9_zero_f32_y_vopd() {
    // V_DUAL_MUL_DX9_ZERO_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        7,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0002 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0100_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4080_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC0A0_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40C9_0FDB }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FC0_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7FE0_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_mov_b32_x_vopd() {
    // V_DUAL_MOV_B32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x0000_000F }, // 0 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0001, expected_y: 0x0000_000F }, // 1 in src0
            VopdCase { src0x: Src::Vgpr(0xFFFF_FFFF), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFFFF_FFFF, expected_y: 0x0000_000F }, // -1 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x8000_0000, expected_y: 0x0000_000F }, // INT_MIN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_FFFF), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_FFFF, expected_y: 0x0000_000F }, // 0xFFFF in src0
            VopdCase { src0x: Src::Vgpr(0xDEAD_BEEF), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xDEAD_BEEF, expected_y: 0x0000_000F }, // 0xDEADBEEF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 3 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0020), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0020, expected_y: 0x0000_000F }, // 32 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0001, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0xFFFF_FFFF, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // -1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // INT_MIN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_FFFF, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 0xFFFF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0xDEAD_BEEF, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 0xDEADBEEF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 3 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0020, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 32 in vsrc1
            VopdCase { src0x: Src::Sgpr(0x0000_0003), vsrc1x: 0x0000_000F, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x0000_000F, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_000F }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_mov_b32_y_vopd() {
    // V_DUAL_MOV_B32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // 0 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0001 }, // 1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFFFF_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0xFFFF_FFFF }, // -1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x8000_0000 }, // INT_MIN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_FFFF }, // 0xFFFF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xDEAD_BEEF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0xDEAD_BEEF }, // 0xDEADBEEF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 3 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0020), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0020 }, // 32 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0001, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xFFFF_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // -1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // INT_MIN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0xFFFF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xDEAD_BEEF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0xDEADBEEF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 3 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0020, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 32 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x0000_0003), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_cndmask_b32_x_vopd() {
    // V_DUAL_CNDMASK_B32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        9,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x0000_000F }, // 0 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0001, expected_y: 0x0000_000F }, // 1 in src0
            VopdCase { src0x: Src::Vgpr(0xFFFF_FFFF), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFFFF_FFFF, expected_y: 0x0000_000F }, // -1 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x8000_0000, expected_y: 0x0000_000F }, // INT_MIN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_FFFF), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_FFFF, expected_y: 0x0000_000F }, // 0xFFFF in src0
            VopdCase { src0x: Src::Vgpr(0xDEAD_BEEF), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xDEAD_BEEF, expected_y: 0x0000_000F }, // 0xDEADBEEF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 3 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0020), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0020, expected_y: 0x0000_000F }, // 32 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0001, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0xFFFF_FFFF, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // -1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // INT_MIN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_FFFF, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 0xFFFF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0xDEAD_BEEF, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 0xDEADBEEF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0003, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 3 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_0020, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // 32 in vsrc1
            VopdCase { src0x: Src::Sgpr(0x0000_0003), vsrc1x: 0x0000_000F, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x0000_000F, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_000F }, // the inline 1.0
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_000F, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // VCC 0x00000000
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_000F, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0xFFFF_FFFF, expected_x: 0x0000_000F, expected_y: 0x0000_000F }, // VCC 0xffffffff
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_000F, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0xAAAA_AAAA, expected_x: 0x0000_0003, expected_y: 0x0000_000F }, // VCC 0xaaaaaaaa
            VopdCase { src0x: Src::Vgpr(0x0000_0003), vsrc1x: 0x0000_000F, src0y: Src::Vgpr(0x0000_000F), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0001, expected_x: 0x0000_000F, expected_y: 0x0000_000F }, // VCC 0x00000001
        ],
    );
}

#[test]
fn v_dual_cndmask_b32_y_vopd() {
    // V_DUAL_CNDMASK_B32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        9,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // 0 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0001 }, // 1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFFFF_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0xFFFF_FFFF }, // -1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x8000_0000 }, // INT_MIN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_FFFF }, // 0xFFFF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xDEAD_BEEF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0xDEAD_BEEF }, // 0xDEADBEEF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 3 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0020), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0020 }, // 32 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0001, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xFFFF_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // -1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // INT_MIN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0xFFFF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xDEAD_BEEF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0xDEADBEEF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 3 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0020, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 32 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x0000_0003), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x3F80_0000 }, // the inline 1.0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // VCC 0x00000000
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0xFFFF_FFFF, expected_x: 0x0000_000F, expected_y: 0x0000_000F }, // VCC 0xffffffff
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0xAAAA_AAAA, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // VCC 0xaaaaaaaa
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0001, expected_x: 0x0000_000F, expected_y: 0x0000_000F }, // VCC 0x00000001
        ],
    );
}

#[test]
fn v_dual_max_num_f32_x_vopd() {
    // V_DUAL_MAX_NUM_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        10,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F7F_FFFF, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4049_0FDB, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_max_num_f32_y_vopd() {
    // V_DUAL_MAX_NUM_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        10,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F7F_FFFF }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4049_0FDB }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_min_num_f32_x_vopd() {
    // V_DUAL_MIN_NUM_F32 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        11,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x8000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBF80_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x7FC0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x7FA0_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0001, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0080_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x7F7F_FFFF), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F00_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC020_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x8000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xBF80_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FC0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7FA0_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_min_num_f32_y_vopd() {
    // V_DUAL_MIN_NUM_F32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        11,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x8000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBF80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FC0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // qNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7FA0_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // sNaN in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0001 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0080_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F7F_FFFF), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // max normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F00_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC020_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x8000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xBF80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FC0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // qNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7FA0_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // sNaN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_dot2acc_f32_f16_x_vopd() {
    // V_DUAL_DOT2ACC_F32_F16 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        12,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4070_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC070_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3780_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4060_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4080_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC084_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4089_2000, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4070_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC070_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4070_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3FF0_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4070_0000, expected_y: 0x3F80_0000 }, // +0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x8000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4070_0000, expected_y: 0x3F80_0000 }, // -0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x3F80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4098_0000, expected_y: 0x3F80_0000 }, // 1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0xBF80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4030_0000, expected_y: 0x3F80_0000 }, // -1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x7F80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0xFF80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in the destination
        ],
    );
}

#[test]
fn v_dual_dot2acc_f32_f16_y_vopd() {
    // V_DUAL_DOT2ACC_F32_F16 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        12,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4070_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC070_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3780_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4060_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4080_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC084_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4089_2000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4070_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC070_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4070_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3FF0_0000 }, // the inline 1.0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4070_0000 }, // +0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x8000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4070_0000 }, // -0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x3F80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4098_0000 }, // 1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0xBF80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4030_0000 }, // -1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x7F80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0xFF80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in the destination
        ],
    );
}

#[test]
fn v_dual_dot2acc_f32_bf16_x_vopd() {
    // V_DUAL_DOT2ACC_F32_BF16 in the X half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        13,
        8,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x8000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0xBF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x7F80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0xFF80_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x0000_0001), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x0080_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0100_0000, expected_y: 0x3F80_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F00_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4080_0000, expected_y: 0x3F80_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0xC020_0000), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC0A0_0000, expected_y: 0x3F80_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x4049_0FDB), vsrc1x: 0x4000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x40C9_0000, expected_y: 0x3F80_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x8000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_0000, expected_y: 0x3F80_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xBF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xC000_0000, expected_y: 0x3F80_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x7F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0xFF80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Sgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // an SGPR source
            VopdCase { src0x: Src::Inline(242), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // +0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x8000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4000_0000, expected_y: 0x3F80_0000 }, // -0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x3F80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x4040_0000, expected_y: 0x3F80_0000 }, // 1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0xBF80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // -1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0x7F80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x7F80_0000, expected_y: 0x3F80_0000 }, // +inf in the destination
            VopdCase { src0x: Src::Vgpr(0x4000_0000), vsrc1x: 0x3F80_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x0000_0000, dstx_in: 0xFF80_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0xFF80_0000, expected_y: 0x3F80_0000 }, // -inf in the destination
        ],
    );
}

#[test]
fn v_dual_dot2acc_f32_bf16_y_vopd() {
    // V_DUAL_DOT2ACC_F32_BF16 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        13,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // -0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xBF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -1.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x7F80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFF80_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // min denorm in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0080_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0100_0000 }, // min normal in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x3F00_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // 0.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4080_0000 }, // 2.0 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xC020_0000), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC0A0_0000 }, // -2.5 in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4049_0FDB), vsrc1y: 0x4000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x40C9_0000 }, // pi in src0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // +0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x0000_0000 }, // -0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // 1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xBF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xC000_0000 }, // -1.0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x7F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0xFF80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in vsrc1
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // the inline 1.0
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // +0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x8000_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4000_0000 }, // -0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x3F80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x4040_0000 }, // 1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0xBF80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x3F80_0000 }, // -1.0 in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0x7F80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0x7F80_0000 }, // +inf in the destination
            VopdCase { src0x: Src::Vgpr(0x3F80_0000), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x4000_0000), vsrc1y: 0x3F80_0000, dstx_in: 0x0000_0000, dsty_in: 0xFF80_0000, vcc_in: 0x0000_0000, expected_x: 0x3F80_0000, expected_y: 0xFF80_0000 }, // -inf in the destination
        ],
    );
}

#[test]
fn v_dual_add_nc_u32_y_vopd() {
    // V_DUAL_ADD_NC_U32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        16,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0004 }, // 1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFFFF_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0002 }, // -1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x8000_0003 }, // INT_MIN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0001_0002 }, // 0xFFFF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xDEAD_BEEF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0xDEAD_BEF2 }, // 0xDEADBEEF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0006 }, // 3 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0020), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0023 }, // 32 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0001, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0004 }, // 1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xFFFF_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0002 }, // -1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x8000_0003 }, // INT_MIN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0001_0002 }, // 0xFFFF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xDEAD_BEEF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0xDEAD_BEF2 }, // 0xDEADBEEF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0006 }, // 3 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0020, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0023 }, // 32 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x0000_0003), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0012 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x3F80_000F }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_lshlrev_b32_y_vopd() {
    // V_DUAL_LSHLREV_B32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        17,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0006 }, // 1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFFFF_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x8000_0000 }, // -1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // INT_MIN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x8000_0000 }, // 0xFFFF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xDEAD_BEEF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0001_8000 }, // 0xDEADBEEF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0018 }, // 3 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0020), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 32 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // 0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0001, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0008 }, // 1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xFFFF_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0xFFFF_FFF8 }, // -1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // INT_MIN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0007_FFF8 }, // 0xFFFF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xDEAD_BEEF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0xF56D_F778 }, // 0xDEADBEEF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0018 }, // 3 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0020, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0100 }, // 32 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x0000_0003), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0078 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_000F }, // the inline 1.0
        ],
    );
}

#[test]
fn v_dual_and_b32_y_vopd() {
    // V_DUAL_AND_B32 in the Y half, against a V_DUAL_MOV_B32 in the other.
    check_vopd(
        8,
        18,
        None,
        &[
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // 0 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0001), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0001 }, // 1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xFFFF_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // -1 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x8000_0000), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // INT_MIN in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_FFFF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0xFFFF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0xDEAD_BEEF), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0xDEADBEEF in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 3 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0020), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // 32 in src0
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // 0 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0001, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0001 }, // 1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xFFFF_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // -1 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x8000_0000, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // INT_MIN in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_FFFF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0xFFFF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0xDEAD_BEEF, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 0xDEADBEEF in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0003, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // 3 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Vgpr(0x0000_0003), vsrc1y: 0x0000_0020, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // 32 in vsrc1
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Sgpr(0x0000_0003), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0003 }, // an SGPR source
            VopdCase { src0x: Src::Vgpr(0x0000_000F), vsrc1x: 0x0000_0000, src0y: Src::Inline(242), vsrc1y: 0x0000_000F, dstx_in: 0x0000_0000, dsty_in: 0x0000_0000, vcc_in: 0x0000_0000, expected_x: 0x0000_000F, expected_y: 0x0000_0000 }, // the inline 1.0
        ],
    );
}
