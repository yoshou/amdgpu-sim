//! Decode conservative rewrite eligibility and attach SSA observations.
use super::*;
use super::state::{Word, Words};
// SGPRs live at 0.., VGPRs at VGPR_BASE.. in the pass's register numbering.
pub(super) const VGPR_BASE: u32 = 512;

const SGPR_NULL: u32 = 124;

pub(super) struct InstEffects {
    pub reads: Vec<u32>,
    // Registers fully written by the instruction. Must be exact or
    // under-approximated: a partial write (e.g. 16-bit halves) must not be
    // listed here.
    pub kills: Vec<u32>,
    // Pure VALU instruction whose only effect is writing `kills`.
    pub removable: bool,
    // false: unknown effects; treated as reading every register.
    pub known: bool,
}

impl InstEffects {
    fn unknown() -> Self {
        InstEffects {
            reads: Vec::new(),
            kills: Vec::new(),
            removable: false,
            known: false,
        }
    }

    fn known(reads: Vec<u32>, kills: Vec<u32>, removable: bool) -> Self {
        InstEffects {
            reads,
            kills,
            removable,
            known: true,
        }
    }
}

fn read_sgpr(reg: u32, words: u32, reads: &mut Vec<u32>) {
    for i in 0..words {
        if reg + i != SGPR_NULL {
            reads.push(reg + i);
        }
    }
}

fn read_vgpr(reg: u32, words: u32, reads: &mut Vec<u32>) {
    for i in 0..words {
        reads.push(VGPR_BASE + reg + i);
    }
}

fn read_src(op: &SourceOperand, words: u32, reads: &mut Vec<u32>) {
    match op {
        SourceOperand::ScalarRegister(reg) => read_sgpr(*reg as u32, words, reads),
        SourceOperand::VectorRegister(reg) => read_vgpr(*reg as u32, words, reads),
        _ => {}
    }
}

fn kill_sgpr(reg: u32, words: u32, kills: &mut Vec<u32>) {
    for i in 0..words {
        if reg + i != SGPR_NULL {
            kills.push(reg + i);
        }
    }
}

fn kill_vgpr(reg: u32, words: u32, kills: &mut Vec<u32>) {
    for i in 0..words {
        kills.push(VGPR_BASE + reg + i);
    }
}

// Width of each operand in 32-bit words for instructions the pass
// understands; (src_words, dst_words). Reads may be over-approximated,
// kills must not be.
fn vop3_arith_widths(op: &I) -> Option<(u32, u32)> {
    match op {
        I::V_FMA_F64
        | I::V_MUL_F64
        | I::V_ADD_F64
        | I::V_MAX_NUM_F64
        | I::V_MIN_NUM_F64
        | I::V_DIV_FMAS_F64
        | I::V_DIV_FIXUP_F64 => Some((2, 2)),
        I::V_LDEXP_F64 | I::V_TRIG_PREOP_F64 => Some((2, 2)),
        I::V_CVT_U32_F64 | I::V_CVT_I32_F64 | I::V_CVT_F32_F64 => Some((2, 1)),
        I::V_CVT_F64_U32 | I::V_CVT_F64_I32 | I::V_CVT_F64_F32 => Some((1, 2)),
        I::V_CNDMASK_B32
        | I::V_MOV_B32
        | I::V_XOR_B32
        | I::V_AND_B32
        | I::V_OR_B32
        | I::V_XOR3_B32
        | I::V_ADD3_U32
        | I::V_LSHLREV_B32
        | I::V_LSHRREV_B32
        | I::V_ASHRREV_I32
        | I::V_ADD_NC_U32
        | I::V_SUB_NC_U32
        | I::V_MUL_LO_U32
        | I::V_FMA_F32
        | I::V_ADD_F32
        | I::V_MUL_F32 => Some((1, 1)),
        _ => None,
    }
}

fn vop1_widths(op: &I) -> Option<(u32, u32)> {
    match op {
        I::V_RCP_F64 | I::V_RSQ_F64 | I::V_SQRT_F64 | I::V_RNDNE_F64 | I::V_FREXP_MANT_F64 => {
            Some((2, 2))
        }
        I::V_FRACT_F64 => Some((2, 2)),
        I::V_FREXP_EXP_I32_F64 | I::V_CVT_U32_F64 | I::V_CVT_I32_F64 | I::V_CVT_F32_F64 => {
            Some((2, 1))
        }
        I::V_CVT_F64_U32 | I::V_CVT_F64_I32 | I::V_CVT_F64_F32 => Some((1, 2)),
        I::V_MOV_B32 => Some((1, 1)),
        _ => None,
    }
}

pub(super) fn effects_of(inst: &InstFormat) -> InstEffects {
    match inst {
        InstFormat::SOPP(inst) => match inst.op {
            I::S_DELAY_ALU
            | I::S_WAIT_ALU
            | I::S_WAIT_LOADCNT
            | I::S_WAIT_LOADCNT_DSCNT
            | I::S_WAIT_DSCNT
            | I::S_WAIT_KMCNT
            | I::S_WAIT_STORECNT
            | I::S_WAIT_BVHCNT
            | I::S_WAIT_SAMPLECNT
            | I::S_CLAUSE
            | I::S_NOP
            | I::S_BRANCH
            | I::S_ENDPGM => InstEffects::known(Vec::new(), Vec::new(), false),
            I::S_CBRANCH_VCCZ | I::S_CBRANCH_VCCNZ => {
                InstEffects::known(vec![106], Vec::new(), false)
            }
            I::S_CBRANCH_EXECZ | I::S_CBRANCH_EXECNZ => {
                InstEffects::known(vec![126], Vec::new(), false)
            }
            I::S_CBRANCH_SCC0 | I::S_CBRANCH_SCC1 => {
                InstEffects::known(Vec::new(), Vec::new(), false)
            }
            _ => InstEffects::unknown(),
        },
        InstFormat::SOP1(inst) => match inst.op {
            I::S_MOV_B32 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 1, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            I::S_MOV_B64 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 2, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 2, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 | I::S_OR_SAVEEXEC_B32 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 1, &mut reads);
                read_sgpr(126, 1, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                kill_sgpr(126, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            _ => InstEffects::unknown(),
        },
        InstFormat::SOP2(inst) => match inst.op {
            I::S_AND_B32
            | I::S_OR_B32
            | I::S_XOR_B32
            | I::S_AND_NOT1_B32
            | I::S_CSELECT_B32
            | I::S_LSHL_B32
            | I::S_LSHR_B32
            | I::S_MUL_I32 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 1, &mut reads);
                read_src(&inst.ssrc1, 1, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            I::S_ADD_NC_U64 => {
                let mut reads = Vec::new();
                read_src(&inst.ssrc0, 2, &mut reads);
                read_src(&inst.ssrc1, 2, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.sdst as u32, 2, &mut kills);
                InstEffects::known(reads, kills, false)
            }
            _ => InstEffects::unknown(),
        },
        InstFormat::SOPC(inst) => {
            // SOPC compares only write SCC, which this pass does not model.
            let mut reads = Vec::new();
            read_src(&inst.ssrc0, 2, &mut reads);
            read_src(&inst.ssrc1, 2, &mut reads);
            InstEffects::known(reads, Vec::new(), false)
        }
        InstFormat::SOPK(inst) => {
            // Conservatively treat the destination as read.
            let mut reads = Vec::new();
            read_sgpr(inst.sdst as u32, 2, &mut reads);
            InstEffects::known(reads, Vec::new(), false)
        }
        InstFormat::VOPC(inst) => {
            let name = format!("{:?}", inst.op);
            let mut reads = Vec::new();
            read_src(&inst.src0, 2, &mut reads);
            read_vgpr(inst.vsrc1 as u32, 2, &mut reads);
            read_sgpr(126, 1, &mut reads);
            let mut kills = Vec::new();
            if name.starts_with("V_CMPX_") {
                kill_sgpr(126, 1, &mut kills);
            } else {
                kill_sgpr(106, 1, &mut kills);
            }
            InstEffects::known(reads, kills, false)
        }
        InstFormat::VOP1(inst) => match vop1_widths(&inst.op) {
            Some((src_words, dst_words)) => {
                let mut reads = Vec::new();
                read_src(&inst.src0, src_words, &mut reads);
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, dst_words, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            None => InstEffects::unknown(),
        },
        InstFormat::VOP2(inst) => match vop3_arith_widths(&inst.op) {
            Some((src_words, dst_words)) => {
                let mut reads = Vec::new();
                read_src(&inst.src0, src_words, &mut reads);
                read_vgpr(inst.vsrc1 as u32, src_words, &mut reads);
                if let I::V_CNDMASK_B32 = inst.op {
                    read_sgpr(106, 1, &mut reads);
                }
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, dst_words, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            None => InstEffects::unknown(),
        },
        InstFormat::VOP3(inst) => {
            let name = format!("{:?}", inst.op);
            if name.starts_with("V_CMPX_") {
                let mut reads = Vec::new();
                read_src(&inst.src0, 2, &mut reads);
                read_src(&inst.src1, 2, &mut reads);
                read_sgpr(126, 1, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(126, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            } else if name.starts_with("V_CMP_") {
                // VOP3-encoded compare: vdst is the destination SGPR.
                let mut reads = Vec::new();
                read_src(&inst.src0, 2, &mut reads);
                read_src(&inst.src1, 2, &mut reads);
                let mut kills = Vec::new();
                kill_sgpr(inst.vdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, false)
            } else {
                match vop3_arith_widths(&inst.op) {
                    Some((src_words, dst_words)) => {
                        let mut reads = Vec::new();
                        let src1_words = match inst.op {
                            I::V_LDEXP_F64 | I::V_TRIG_PREOP_F64 => 1,
                            _ => src_words,
                        };
                        read_src(&inst.src0, src_words, &mut reads);
                        read_src(&inst.src1, src1_words, &mut reads);
                        read_src(&inst.src2, src_words, &mut reads);
                        if let I::V_DIV_FMAS_F64 = inst.op {
                            read_sgpr(106, 1, &mut reads);
                        }
                        let mut kills = Vec::new();
                        kill_vgpr(inst.vdst as u32, dst_words, &mut kills);
                        InstEffects::known(reads, kills, true)
                    }
                    None => InstEffects::unknown(),
                }
            }
        }
        InstFormat::VOP3SD(inst) => match inst.op {
            I::V_DIV_SCALE_F64 => {
                let mut reads = Vec::new();
                read_src(&inst.src0, 2, &mut reads);
                read_src(&inst.src1, 2, &mut reads);
                read_src(&inst.src2, 2, &mut reads);
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, 2, &mut kills);
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            I::V_MAD_CO_U64_U32 => {
                let mut reads = Vec::new();
                read_src(&inst.src0, 1, &mut reads);
                read_src(&inst.src1, 1, &mut reads);
                read_src(&inst.src2, 2, &mut reads);
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, 2, &mut kills);
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            I::V_ADD_CO_CI_U32 | I::V_SUB_CO_CI_U32 => {
                let mut reads = Vec::new();
                read_src(&inst.src0, 1, &mut reads);
                read_src(&inst.src1, 1, &mut reads);
                read_src(&inst.src2, 1, &mut reads);
                let mut kills = Vec::new();
                kill_vgpr(inst.vdst as u32, 1, &mut kills);
                kill_sgpr(inst.sdst as u32, 1, &mut kills);
                InstEffects::known(reads, kills, true)
            }
            _ => InstEffects::unknown(),
        },
        InstFormat::VGLOBAL(inst) => {
            let name = format!("{:?}", inst.op);
            let data_words = if name.ends_with("_B128") {
                4
            } else if name.ends_with("_B96") {
                3
            } else if name.ends_with("_B64") {
                2
            } else {
                1
            };
            let mut reads = Vec::new();
            read_vgpr(inst.vaddr as u32, 2, &mut reads);
            if inst.saddr != SGPR_NULL as u8 {
                read_sgpr(inst.saddr as u32, 2, &mut reads);
            }
            read_sgpr(126, 1, &mut reads);
            if name.starts_with("GLOBAL_LOAD") {
                // Loads only write lanes with EXEC set, so the destination is
                // a partial write: model it as a read, never a kill.
                read_vgpr(inst.vdst as u32, data_words, &mut reads);
                InstEffects::known(reads, Vec::new(), false)
            } else if name.starts_with("GLOBAL_STORE") {
                read_vgpr(inst.vsrc as u32, data_words, &mut reads);
                InstEffects::known(reads, Vec::new(), false)
            } else {
                InstEffects::unknown()
            }
        }
        InstFormat::VOPD(inst) => {
            let half =
                |op: &I, src0: &SourceOperand, vsrc1: u8, vdst: u32| -> Option<(Vec<u32>, Vec<u32>)> {
                    let mut reads = Vec::new();
                    let mut kills = Vec::new();
                    match op {
                        I::V_DUAL_MOV_B32 => {
                            read_src(src0, 1, &mut reads);
                        }
                        I::V_DUAL_CNDMASK_B32 => {
                            read_src(src0, 1, &mut reads);
                            read_vgpr(vsrc1 as u32, 1, &mut reads);
                            read_sgpr(106, 1, &mut reads);
                        }
                        I::V_DUAL_ADD_F32
                        | I::V_DUAL_MUL_F32
                        | I::V_DUAL_AND_B32
                        | I::V_DUAL_ADD_NC_U32
                        | I::V_DUAL_LSHLREV_B32 => {
                            read_src(src0, 1, &mut reads);
                            read_vgpr(vsrc1 as u32, 1, &mut reads);
                        }
                        I::V_DUAL_FMAC_F32 => {
                            read_src(src0, 1, &mut reads);
                            read_vgpr(vsrc1 as u32, 1, &mut reads);
                            read_vgpr(vdst as u32, 1, &mut reads);
                        }
                        _ => return None,
                    }
                    kill_vgpr(vdst as u32, 1, &mut kills);
                    Some((reads, kills))
                };

            // VOPD Y-op's real VGPR is (vdsty << 1) | ((vdstx & 1) ^ 1) — opposite
            // parity of X — not vdsty directly. Using vdsty here made the DCE treat
            // the wrong register as killed and drop live producers of the real one.
            let dy = ((inst.vdsty as u32) << 1) | (((inst.vdstx as u32) & 1) ^ 1);
            match (
                half(&inst.opx, &inst.src0x, inst.vsrc1x, inst.vdstx as u32),
                half(&inst.opy, &inst.src0y, inst.vsrc1y, dy),
            ) {
                (Some((rx, kx)), Some((ry, ky))) => {
                    let mut reads = rx;
                    reads.extend(ry);
                    let mut kills = kx;
                    kills.extend(ky);
                    InstEffects::known(reads, kills, false)
                }
                _ => InstEffects::unknown(),
            }
        }
        _ => InstEffects::unknown(),
    }
}


fn operand(source: SourceOperand, words: &Words) -> crate::rdna_spmd::analysis::rewrite::Operand {
    use crate::rdna_spmd::analysis::rewrite::{Operand, ConstantEncoding::*};
    let mut out = Operand { slot: None, scalar: false, constant: None, words: [None; 2] };
    match source {
        SourceOperand::VectorRegister(r) => {
            out.slot = Some(r as u32);
            out.words = [words.get(&Word::Vgpr(r as u32)).copied(), words.get(&Word::Vgpr(r as u32+1)).copied()];
        },
        SourceOperand::ScalarRegister(r) => {
            out.slot = Some(r as u32); out.scalar = true;
            out.words = [r as u32, r as u32+1].map(|r| Word::scalar(r).and_then(|w| words.get(&w).copied()));
        },
        SourceOperand::FloatConstant(v) => out.constant = Some((Float, v.to_bits())),
        SourceOperand::IntegerConstant(v) => out.constant = Some((Integer, v)),
        SourceOperand::LiteralConstant(v) => out.constant = Some((Literal, v as u64)),
        SourceOperand::PrivateBase => {},
    }
    out
}
fn math(inst: &InstFormat, before: &Words) -> Option<crate::rdna_spmd::analysis::rewrite::Math> {
    use crate::rdna_spmd::analysis::rewrite::{Math, Kind};
    let (op, destination, scalar_destination, inputs, neg, abs, omod, cm, opsel, form) = match inst {
        InstFormat::VOP1(i) => (i.op, i.vdst, None, [i.src0, SourceOperand::IntegerConstant(0), SourceOperand::IntegerConstant(0)], 0, 0, 0, 0, 0, 1),
        InstFormat::VOP2(i) => (i.op, i.vdst, None, [i.src0, SourceOperand::VectorRegister(i.vsrc1), SourceOperand::IntegerConstant(0)], 0, 0, 0, 0, 0, 2),
        InstFormat::VOP3(i) => (i.op, i.vdst, None, [i.src0, i.src1, i.src2], i.neg, i.abs, i.omod, i.cm, i.opsel, 3),
        InstFormat::VOP3SD(i) => (i.op, i.vdst, Some(i.sdst as u32), [i.src0, i.src1, i.src2], i.neg, 0, i.omod, i.cm, 0, 4),
        _ => return None,
    };
    let kind = match op {
        I::V_RSQ_F64 => Kind::Rsq, I::V_RCP_F64 => Kind::Rcp, I::V_MUL_F64 => Kind::Mul,
        I::V_FMA_F64 => Kind::Fma, I::V_DIV_SCALE_F64 => Kind::DivScale,
        I::V_DIV_FMAS_F64 => Kind::DivFmas, I::V_DIV_FIXUP_F64 => Kind::DivFixup,
        _ => return None,
    };
    let local_sqrt = match kind {
        Kind::Rsq => form == 1 || (form == 3 && neg == 0 && abs == 0 && omod == 0),
        Kind::Mul => form == 2 || (form == 3 && neg == 0 && abs == 0 && omod == 0 && cm == 0 && opsel == 0),
        Kind::Fma => form == 3 && abs == 0 && omod == 0 && cm == 0 && opsel == 0,
        _ => false,
    };
    let local_div = match kind {
        Kind::Rcp => form == 1 || (form == 3 && neg == 0 && abs == 0 && omod == 0),
        Kind::Mul => form == 2 || (form == 3 && neg == 0 && abs == 0 && omod == 0 && cm == 0),
        Kind::Fma => form == 3 && omod == 0 && cm == 0 && opsel == 0,
        Kind::DivScale => form == 4 && neg == 0 && omod == 0 && cm == 0,
        Kind::DivFmas => form == 3 && neg == 0 && abs == 0,
        Kind::DivFixup => form == 3,
        _ => false,
    };
    let cross_sqrt = match kind { Kind::Rsq => form == 1, Kind::Mul => form == 2, Kind::Fma => form == 3, _ => false };
    Some(Math { kind, destination: destination as u32, scalar_destination, inputs: inputs.map(|s| operand(s, before)), neg, abs,
        local_sqrt, local_div, cross_sqrt })
}
pub(super) fn word(slot: u32) -> Option<Word> {
    if slot >= VGPR_BASE { Some(Word::Vgpr(slot-VGPR_BASE)) } else { Word::scalar(slot) }
}
pub(in crate::rdna_spmd) fn observation(inst: &InstFormat, before: &Words, after: &Words) -> crate::rdna_spmd::analysis::rewrite::Observation {
    let effects = effects_of(inst);
    crate::rdna_spmd::analysis::rewrite::Observation {
        reads: effects.reads.iter().filter_map(|&r| word(r).and_then(|w| before.get(&w).copied())).collect(),
        defined_words: after.iter().filter_map(|(word, &value)| {
            if before[word] == value { return None; }
            let slot = match word { Word::Vgpr(r) => VGPR_BASE+r, Word::Sgpr(r) | Word::Mask(r) => *r };
            Some((slot, value))
        }).collect(),
        definitions: effects.kills.iter().filter_map(|&r| Some((r, *after.get(&word(r)?)?))).collect(),
        replaced: effects.kills.iter().filter_map(|&r| word(r).and_then(|w| before.get(&w).copied())).collect(),
        known: effects.known, removable: effects.removable, math: math(inst, before),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_instructions::{VOP1, VOP2, VOP3, VOP3SD};

    #[test]
    fn interleaved_division_uses_the_definition_before_numerator_scaling() {
        let v = SourceOperand::VectorRegister;
        let one = SourceOperand::FloatConstant(1.0);
        let scale = |dst, flag, src0, src1, src2| InstFormat::VOP3SD(VOP3SD {
            op: I::V_DIV_SCALE_F64, vdst: dst, sdst: flag, src0, src1, src2, cm: 0, omod: 0, neg: 0,
        });
        let rcp = |dst, src| InstFormat::VOP1(VOP1 { op: I::V_RCP_F64, vdst: dst, src0: v(src) });
        let alu = |op, dst, src0, src1, src2, neg| InstFormat::VOP3(VOP3 {
            op, vdst: dst, src0, src1, src2, neg, abs: 0, cm: 0, omod: 0, opsel: 0,
        });
        // Two independent refinements are interleaved. Only the second one
        // reaches a fixup here. Its first scheduling point is numerator scaling,
        // which does not itself read the already-scaled denominator v2:v3.
        let body = vec![
            scale(16,124,v(14),v(14),one), scale(2,124,v(22),v(22),v(0)),
            scale(10,106,v(0),v(22),v(0)), rcp(18,16), rcp(6,2),
            alu(I::V_FMA_F64,32,v(16),v(18),one,1), alu(I::V_FMA_F64,8,v(2),v(6),one,1),
            alu(I::V_FMA_F64,18,v(18),v(32),v(18),0), alu(I::V_FMA_F64,6,v(6),v(8),v(6),0),
            alu(I::V_FMA_F64,8,v(2),v(6),one,1), alu(I::V_FMA_F64,6,v(6),v(8),v(6),0),
            InstFormat::VOP2(VOP2 { op: I::V_MUL_F64, vdst: 8, src0: v(10), vsrc1: 6, literal_constant: None }),
            alu(I::V_FMA_F64,2,v(2),v(8),v(10),1), alu(I::V_DIV_FMAS_F64,2,v(2),v(6),v(8),0),
            alu(I::V_DIV_FIXUP_F64,24,v(2),v(22),v(0),0),
        ];
        use crate::rdna_spmd::{CompilationInput,ScalarProgram,ScalarBlock,Terminator};
        use std::collections::BTreeMap;
        let program=ScalarProgram {entry_pc:0,blocks:BTreeMap::from([(0,ScalarBlock {pc:0,body,term:Terminator::Return})])};
        let mut f=program.to_ssa().function;
        let observations:Vec<_>=f.state.sites[&0].iter().map(|s|s.rewrite.clone()).collect();
        let expected:Vec<_>=[0,3,5,7,14].map(|i|observations[i].math.as_ref().map(|m|(m.kind,m.destination))).into();
        let remove=crate::rdna_spmd::combine::divisions(&observations);
        assert_eq!(remove.iter().filter(|&&v|v).count(),10);
        f.remove_sites(0,&remove);
        assert_eq!(f.state.sites[&0].iter().map(|s|s.rewrite.math.as_ref().map(|m|(m.kind,m.destination))).collect::<Vec<_>>(),expected);
        f.compact();
        f.ir.verify_with(&f.registry).unwrap();
    }
}
