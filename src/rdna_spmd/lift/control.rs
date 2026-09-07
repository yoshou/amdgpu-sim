//! Decode the existing EXEC proof restrictions into SSA definition policies.
//! The policies describe permitted proofs, not alternate execution semantics.
use super::*;
use crate::rdna_spmd::analysis::state::{Activation, ExecPolicy};
use super::state::{Word, Words};

pub(super) fn policy(inst: &InstFormat, before: &Words, after: &Words) -> ExecPolicy {
    let old = before[&Word::Mask(126)];
    let next = after[&Word::Mask(126)];
    let scalar = |r: u8| Word::scalar(r as u32).and_then(|w| before.get(&w).copied());
    let mut update = if writes_exec(inst) { Activation::Unknown } else { Activation::Copy(old) };
    let mut saved = None;
    let mut constant = None;
    let mut preserves_nonempty = false;
    match inst {
        InstFormat::SOP1(i) if matches!(i.op, I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 | I::S_OR_SAVEEXEC_B32 | I::S_XOR_SAVEEXEC_B32) => {
            saved = Word::scalar(i.sdst as u32).and_then(|w| after.get(&w).copied());
            update = Activation::Unknown;
        },
        InstFormat::SOP1(i) if i.sdst == 126 && matches!(i.op, I::S_MOV_B32 | I::S_MOV_B64) => {
            constant = match i.ssrc0 { SourceOperand::IntegerConstant(v) => Some(v), SourceOperand::LiteralConstant(v) => Some(v as u64), _ => None };
            update = constant.map_or(Activation::Unknown, |v| Activation::Constant(v & 1 != 0));
        },
        InstFormat::SOP2(i) if i.sdst == 126 && matches!(i.op, I::S_OR_B32 | I::S_OR_B64) => {
            let reads_exec = matches!(i.ssrc0, SourceOperand::ScalarRegister(126)) || matches!(i.ssrc1, SourceOperand::ScalarRegister(126));
            preserves_nonempty = reads_exec;
            if matches!(i.op, I::S_OR_B32) {
                let saved_reg = match i.ssrc1 { SourceOperand::ScalarRegister(r) => Some(r), _ => match i.ssrc0 { SourceOperand::ScalarRegister(r) => Some(r), _ => None } };
                update = Activation::Or { exec: reads_exec.then_some(old), saved: saved_reg.filter(|&r| r != 126).and_then(scalar) };
            }
        },
        _ => {},
    }
    let writes = scalar_dests(inst);
    let killed = (0..128).filter(|r| writes & (1u128 << r) != 0)
        .filter_map(|r| Word::scalar(r).and_then(|w| after.get(&w).copied())).collect();
    let writes_exec = writes_exec(inst);
    let wave_writes_exec = super::wave::instruction(inst).is_some_and(|a| a.io().writes.has_sgpr(126));
    ExecPolicy { before: old, after: next, update, saved, killed, constant,
        preserves_nonempty, writes: writes_exec, resets_nonempty: writes_exec || wave_writes_exec }
}

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
pub(super) fn writes_exec(inst: &InstFormat) -> bool {
    scalar_dests(inst) & (1u128 << EXEC) != 0 || matches!(inst,
        InstFormat::SOP1(i) if matches!(i.op,
            I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 |
            I::S_OR_SAVEEXEC_B32 | I::S_XOR_SAVEEXEC_B32))
}

/// Only mask widening can expose a value from a previously inactive lane.
/// Unknown writes remain widening; recognize only subset-of-old-EXEC forms.
pub(super) fn may_enable_lanes(inst: &InstFormat) -> bool {
    if !writes_exec(inst) { return false; }
    let exec = |s: &SourceOperand| matches!(s, SourceOperand::ScalarRegister(126));
    match inst {
        InstFormat::VOPC(_) => false, // st_cmp: comparison & old EXEC
        InstFormat::VOP3(i) if format!("{:?}", i.op).starts_with("V_CMP") => false,
        InstFormat::SOP1(i) if matches!(i.op, I::S_AND_SAVEEXEC_B32) => false,
        InstFormat::SOP1(i) if i.sdst == 126 && matches!(i.op, I::S_MOV_B32 | I::S_MOV_B64) => {
            !(exec(&i.ssrc0) || matches!(i.ssrc0,
                SourceOperand::IntegerConstant(0) | SourceOperand::LiteralConstant(0)))
        }
        InstFormat::SOP2(i) if i.sdst == 126 && matches!(i.op, I::S_AND_B32 | I::S_AND_B64) => {
            !(exec(&i.ssrc0) || exec(&i.ssrc1))
        }
        InstFormat::SOP2(i) if matches!(i.op, I::S_AND_NOT1_B32) => !exec(&i.ssrc0),
        _ => true,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_instructions::{SOP1, SOP2};
    #[test]
    fn distinguishes_exec_subsets_from_reactivation() {
        for (op, widening) in [(I::S_AND_B32, false), (I::S_AND_NOT1_B32, false), (I::S_OR_B32, true), (I::S_XOR_B32, true)] {
            let inst = InstFormat::SOP2(SOP2 { op, sdst: 126,
                ssrc0: SourceOperand::ScalarRegister(126), ssrc1: SourceOperand::ScalarRegister(4) });
            assert!(writes_exec(&inst));
            assert_eq!(may_enable_lanes(&inst), widening);
        }
        // This ISA form is src & !old_exec in the existing emitter and can
        // enable lanes; do not confuse it with old_exec & !src.
        let inst = InstFormat::SOP1(SOP1 { op: I::S_AND_NOT1_SAVEEXEC_B32, sdst: 4,
            ssrc0: SourceOperand::ScalarRegister(6) });
        assert!(may_enable_lanes(&inst));
    }
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

pub(super) fn scalar_write_regs(inst: &InstFormat) -> Vec<u32> {
    let pair = |reg: u8| vec![reg as u32, reg as u32 + 1];
    let one = |reg: u8| vec![reg as u32];
    match inst {
        InstFormat::SOP1(i) => if matches!(i.op, I::S_MOV_B64) { pair(i.sdst) } else { one(i.sdst) },
        InstFormat::SOP2(i) if matches!(i.op,
            I::S_ADD_NC_U64 | I::S_MUL_U64 | I::S_LSHL_B64 | I::S_LSHR_B64 |
            I::S_ASHR_I64 | I::S_AND_B64 | I::S_OR_B64 | I::S_XOR_B64 | I::S_CSELECT_B64) => pair(i.sdst),
        InstFormat::SOP2(i) => one(i.sdst),
        InstFormat::SOPK(i) => one(i.sdst),
        InstFormat::VOP3SD(i) => one(i.sdst),
        InstFormat::SMEM(i) => {
            let words = match i.op {
                I::S_LOAD_B32 | I::S_LOAD_U16 => 1,
                I::S_LOAD_B64 => 2,
                I::S_LOAD_B96 => 3,
                I::S_LOAD_B128 => 4,
                I::S_LOAD_B256 => 8,
                I::S_LOAD_B512 => 16,
                _ => 1,
            };
            (0..words).map(|offset| i.sdata as u32 + offset).collect()
        }
        _ => vec![],
    }
}


/// Native mask eligibility restrictions attached to the actual SSA definitions.
pub(super) fn mask_policy(inst: &InstFormat, before: &Words, after: &Words) -> crate::rdna_spmd::analysis::state::MaskPolicy {
    use crate::rdna_spmd::analysis::state::{MaskPolicy, MaskEvent};
    let mut p = MaskPolicy::default();
    p.reads = before.iter().filter_map(|(word, &value)| match word {
        Word::Sgpr(r) | Word::Mask(r) if scalar_mask_read(inst, *r) => Some((*r, value)), _ => None,
    }).collect();
    let mut rules = std::collections::BTreeMap::<u32, (Vec<u32>, bool)>::new();
    let src = |ops: &[SourceOperand]| ops.iter().filter_map(scalar_reg).collect::<Vec<_>>();
    match inst {
        InstFormat::SOP1(i) if saveexec_op(i.op) => {
            p.seed = vec![i.sdst as u32]; p.seed.extend(src(&[i.ssrc0]));
            rules.insert(i.sdst as u32, (vec![], true)); rules.insert(126, (vec![], true));
            p.event = Some(MaskEvent::Save(i.sdst as u32));
        },
        InstFormat::SOP1(i) if matches!(i.op, I::S_MOV_B32) => {
            let sources = src(&[i.ssrc0]);
            p.closure = vec![i.sdst as u32]; p.closure.extend(&sources);
            rules.insert(i.sdst as u32, (sources, false));
            if scalar_reg(&i.ssrc0) == Some(126) { p.event = Some(MaskEvent::Copy(i.sdst as u32)); }
        },
        InstFormat::SOP2(i) if mask_logic_op(i.op) => {
            let sources = src(&[i.ssrc0, i.ssrc1]);
            p.closure = vec![i.sdst as u32]; p.closure.extend(&sources);
            rules.insert(i.sdst as u32, (sources, false));
        },
        InstFormat::VOPC(i) if format!("{:?}", i.op).starts_with("V_CMP") => {
            let r = if format!("{:?}", i.op).starts_with("V_CMPX") { 126 } else { 106 };
            p.seed.push(r); rules.insert(r, (vec![], true));
            if r == 126 { p.event = Some(MaskEvent::Compare); }
        },
        InstFormat::VOP3(i) if format!("{:?}", i.op).starts_with("V_CMP") => {
            let r = if format!("{:?}", i.op).starts_with("V_CMPX") { 126 } else { 106 };
            p.seed.push(r); rules.insert(r, (vec![], true));
            if r == 126 && i.vdst == 126 { p.event = Some(MaskEvent::Compare); }
        },
        InstFormat::VOP3SD(i) => { rules.insert(i.sdst as u32, (vec![], true)); },
        _ => { for r in scalar_write_regs(inst) { rules.insert(r, (vec![], r == 126 || r == 106)); } },
    }
    if let InstFormat::SOP2(i) = inst {
        if i.sdst == 126 { p.event = Some(MaskEvent::Logic { restore: matches!(i.op, I::S_OR_B32), sources: src(&[i.ssrc0, i.ssrc1]) }); }
    }
    for (word, &value) in after {
        let slot = match word { Word::Sgpr(r) | Word::Mask(r) => *r, _ => continue };
        if let Some((sources, fixed)) = rules.remove(&slot) {
            let inputs = sources.into_iter().filter_map(|r| Word::scalar(r).and_then(|w| before.get(&w).copied())).collect();
            p.definitions.push((value, inputs, fixed));
        } else if before[word] != value { p.definitions.push((value, vec![before[word]], false)); }
    }
    p.cross_lane = matches!(inst,
        InstFormat::VOP3(i) if matches!(i.op, I::V_READLANE_B32 | I::V_WRITELANE_B32))
        || matches!(inst, InstFormat::VOP3P(i) if matches!(i.op, I::V_WMMA_F32_16X16X16_F16))
        || matches!(inst, InstFormat::DS(i) if matches!(i.op, I::DS_BPERMUTE_B32 | I::DS_BPERMUTE_FI_B32));
    p
}

pub(super) fn mask_scalar_reads(inst: &InstFormat) -> Vec<u32> {
    (0..128).filter(|&r| scalar_mask_read(inst, r)).collect()
}
