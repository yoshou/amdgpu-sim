//! Decode the existing EXEC proof restrictions into SSA definition policies.
//! The policies describe permitted proofs, not alternate execution semantics.
use super::*;

const EXEC: u32 = 126;
/// SGPRs written by an instruction (over-approximated — used to *clear* tracked
/// saved-active bits, so over-clearing is always sound).
pub(super) fn scalar_dests(inst: &InstFormat) -> u128 {
    let bit = |r: u32| 1u128 << (r & 127);
    let pair = |r: u32| bit(r) | bit(r + 1);
    match inst {
        InstFormat::SOP1(i) => match i.op {
            I::S_MOV_B64 => pair(i.sdst as u32),
            _ => bit(i.sdst as u32),
        },
        InstFormat::SOP2(i) => match i.op {
            I::S_ADD_NC_U64 | I::S_MUL_U64 | I::S_LSHL_B64 | I::S_AND_B64 | I::S_OR_B64 => {
                pair(i.sdst as u32)
            }
            _ => bit(i.sdst as u32),
        },
        InstFormat::SOPK(i) => bit(i.sdst as u32),
        InstFormat::SMEM(i) => {
            let words = match i.op {
                I::S_LOAD_B32 => 1,
                I::S_LOAD_B64 => 2,
                I::S_LOAD_B96 => 3,
                I::S_LOAD_B128 => 4,
                I::S_LOAD_B256 => 8,
                I::S_LOAD_B512 => 16,
                _ => 1,
            };
            (0..words).fold(0u128, |m, k| m | bit(i.sdata as u32 + k))
        }
        InstFormat::VOP3SD(i) => bit(i.sdst as u32),
        InstFormat::VOP1(i) if matches!(i.op,I::V_READFIRSTLANE_B32) => bit(i.vdst as u32),
        InstFormat::VOP3(i) if matches!(i.op,I::V_READLANE_B32) => bit(i.vdst as u32),
        // VOPC / VOP3 compares write a lane mask (VCC or, for V_CMPX, EXEC).
        InstFormat::VOPC(i) => {
            if format!("{:?}", i.op).starts_with("V_CMPX") { bit(EXEC) } else { bit(106) }
        }
        InstFormat::VOP3(i) if format!("{:?}", i.op).contains("V_CMP") => bit(i.vdst as u32),
        _ => 0,
    }
}

/// Includes implicit EXEC writes (saveexec and cmpx), not just scalar destinations.
pub(in crate::rdna_spmd) fn writes_exec(inst: &InstFormat) -> bool {
    scalar_dests(inst) & (1u128 << EXEC) != 0 || matches!(inst,
        InstFormat::SOP1(i) if matches!(i.op,
            I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 |
            I::S_OR_SAVEEXEC_B32 | I::S_XOR_SAVEEXEC_B32))
}

fn scalar_reg(op: &SourceOperand) -> Option<u32> {
    match op {
        SourceOperand::ScalarRegister(reg) => Some(*reg as u32),
        _ => None,
    }
}

fn mask_logic_op(op: I) -> bool {
    matches!(op,
        I::S_AND_B32 | I::S_OR_B32 | I::S_XOR_B32 |
        I::S_AND_NOT1_B32 | I::S_OR_NOT1_B32)
}

fn saveexec_op(op: I) -> bool {
    matches!(op,
        I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 |
        I::S_OR_SAVEEXEC_B32 | I::S_XOR_SAVEEXEC_B32)
}


fn operand_is_reg(op: &SourceOperand, reg: u32) -> bool {
    scalar_reg(op) == Some(reg)
}

fn range_contains(first: u32, words: u32, reg: u32) -> bool {
    (first..first.saturating_add(words)).contains(&reg)
}

/// Whether `inst` reads `reg` as an ordinary scalar value rather than as a
/// packed lane mask. Destinations are deliberately absent: a scalar
/// redefinition after the mask value is dead is harmless, and the reaching
/// definition transfer below kills it before a later read.
fn scalar_mask_read(inst: &InstFormat, reg: u32) -> bool {
    match inst {
        InstFormat::SOP1(i) => !(matches!(i.op, I::S_MOV_B32) || saveexec_op(i.op))
            && operand_is_reg(&i.ssrc0, reg),
        InstFormat::SOP2(i) => !mask_logic_op(i.op)
            && (operand_is_reg(&i.ssrc0, reg) || operand_is_reg(&i.ssrc1, reg)),
        InstFormat::SOPK(_) => false,
        InstFormat::SOPC(i) => operand_is_reg(&i.ssrc0, reg) || operand_is_reg(&i.ssrc1, reg),
        InstFormat::SOPP(_) => false,
        InstFormat::SMEM(i) => {
            range_contains(i.sbase as u32, 2, reg)
                || i.soffset as u32 == reg
        }
        InstFormat::VOP1(i) => operand_is_reg(&i.src0, reg),
        InstFormat::VOP2(i) => operand_is_reg(&i.src0, reg),
        InstFormat::VOP3(i) => {
            // V_CMP writes EXEC/VCC implicitly; its vector operands do not
            // alias an SGPR unless one is explicitly encoded as a source.
            operand_is_reg(&i.src0, reg) || operand_is_reg(&i.src1, reg) || operand_is_reg(&i.src2, reg)
        }
        InstFormat::VOP3SD(i) => operand_is_reg(&i.src0, reg)
            || operand_is_reg(&i.src1, reg) || operand_is_reg(&i.src2, reg),
        InstFormat::VOP3P(i) => operand_is_reg(&i.src0, reg)
            || operand_is_reg(&i.src1, reg) || operand_is_reg(&i.src2, reg),
        InstFormat::VOPC(i) => operand_is_reg(&i.src0, reg),
        InstFormat::VOPD(i) => operand_is_reg(&i.src0x, reg) || operand_is_reg(&i.src0y, reg),
        InstFormat::VFLAT(i) => {
            i.saddr != 124 && i.saddr != 127 && range_contains(i.saddr as u32, 2, reg)
        }
        InstFormat::VGLOBAL(i) => {
            i.saddr != 124 && i.saddr != 127 && range_contains(i.saddr as u32, 2, reg)
        }
        InstFormat::VSCRATCH(i) => {
            i.saddr != 124 && i.saddr != 127 && range_contains(i.saddr as u32, 2, reg)
        }
        InstFormat::VIMAGE(i) => range_contains(i.rsrc as u32, 4, reg),
        InstFormat::VSAMPLE(i) => range_contains(i.rsrc as u32, 4, reg) || range_contains(i.samp as u32, 4, reg),
        InstFormat::DS(_) => false,
    }
}

pub(super) fn mask_scalar_reads(inst: &InstFormat) -> Vec<u32> {
    (0..128).filter(|&r| scalar_mask_read(inst, r)).collect()
}
