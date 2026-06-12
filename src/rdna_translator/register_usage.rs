use std::collections::HashSet;

use super::*;

#[derive(Debug, Clone)]
pub(crate) struct RegisterUsage {
    pub(crate) use_sgprs: HashSet<u32>,
    pub(crate) use_vgprs: HashSet<u32>,
    pub(crate) def_sgprs: HashSet<u32>,
    pub(crate) def_vgprs: HashSet<u32>,
    pub(crate) incomming_sgprs: HashSet<u32>,
    pub(crate) incomming_vgprs: HashSet<u32>,
}

impl RegisterUsage {
    pub(crate) fn new() -> Self {
        RegisterUsage {
            use_sgprs: HashSet::new(),
            use_vgprs: HashSet::new(),
            def_sgprs: HashSet::new(),
            def_vgprs: HashSet::new(),
            incomming_sgprs: HashSet::new(),
            incomming_vgprs: HashSet::new(),
        }
    }

    pub(crate) fn use_sgpr_u32(&mut self, reg: u32) {
        if reg == 124 {
            return;
        }
        if !self.def_sgprs.contains(&reg) {
            self.incomming_sgprs.insert(reg);
        }
        self.use_sgprs.insert(reg);
    }

    pub(crate) fn use_sgpr_u64(&mut self, reg: u32) {
        self.use_sgpr_u32(reg);
        self.use_sgpr_u32(reg + 1);
    }

    pub(crate) fn _use_sgpr_f64(&mut self, reg: u32) {
        self.use_sgpr_u32(reg);
        self.use_sgpr_u32(reg + 1);
    }

    pub(crate) fn use_vgpr_u32(&mut self, reg: u32) {
        if !self.def_vgprs.contains(&reg) {
            self.incomming_vgprs.insert(reg);
        }
        self.use_vgprs.insert(reg);
    }

    pub(crate) fn use_vgpr_f32(&mut self, reg: u32) {
        if !self.def_vgprs.contains(&reg) {
            self.incomming_vgprs.insert(reg);
        }
        self.use_vgprs.insert(reg);
    }

    pub(crate) fn use_vgpr_u64(&mut self, reg: u32) {
        self.use_vgpr_u32(reg);
        self.use_vgpr_u32(reg + 1);
    }

    pub(crate) fn use_vgpr_f64(&mut self, reg: u32) {
        self.use_vgpr_u32(reg);
        self.use_vgpr_u32(reg + 1);
    }

    pub(crate) fn use_operand_u16(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => self.use_sgpr_u32(*reg as u32),
            SourceOperand::VectorRegister(reg) => self.use_vgpr_u32(*reg as u32),
            _ => {}
        };
    }

    pub(crate) fn use_operand_u32(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => self.use_sgpr_u32(*reg as u32),
            SourceOperand::VectorRegister(reg) => self.use_vgpr_u32(*reg as u32),
            _ => {}
        };
    }

    pub(crate) fn use_operand_f32(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => self.use_sgpr_u32(*reg as u32),
            SourceOperand::VectorRegister(reg) => self.use_vgpr_u32(*reg as u32),
            _ => {}
        };
    }

    pub(crate) fn use_operand_f16_vec<const N: usize>(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => {
                for i in (0..N).step_by(2) {
                    self.use_sgpr_u32(*reg as u32 + (i / 2) as u32);
                }
            }
            SourceOperand::VectorRegister(reg) => {
                for i in (0..N).step_by(2) {
                    self.use_vgpr_u32(*reg as u32 + (i / 2) as u32);
                }
            }
            _ => {}
        };
    }

    pub(crate) fn use_operand_f32_vec<const N: usize>(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => {
                for i in 0..N {
                    self.use_sgpr_u32(*reg as u32 + i as u32);
                }
            }
            SourceOperand::VectorRegister(reg) => {
                for i in 0..N {
                    self.use_vgpr_u32(*reg as u32 + i as u32);
                }
            }
            _ => {}
        };
    }

    pub(crate) fn use_operand_u64(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => {
                self.use_sgpr_u32(*reg as u32);
                self.use_sgpr_u32((*reg + 1) as u32);
            }
            SourceOperand::VectorRegister(reg) => {
                self.use_vgpr_u32(*reg as u32);
                self.use_vgpr_u32((*reg + 1) as u32);
            }
            _ => {}
        };
    }

    pub(crate) fn use_operand_f64(&mut self, operand: &SourceOperand) {
        match operand {
            SourceOperand::ScalarRegister(reg) => {
                self.use_sgpr_u32(*reg as u32);
                self.use_sgpr_u32((*reg + 1) as u32);
            }
            SourceOperand::VectorRegister(reg) => {
                self.use_vgpr_u32(*reg as u32);
                self.use_vgpr_u32((*reg + 1) as u32);
            }
            _ => {}
        };
    }

    pub(crate) fn def_sgpr_u32(&mut self, reg: u32) {
        if reg == 124 {
            return;
        }
        self.def_sgprs.insert(reg);
    }

    pub(crate) fn def_sgpr_f32(&mut self, reg: u32) {
        self.def_sgpr_u32(reg);
    }

    pub(crate) fn def_sgpr_u64(&mut self, reg: u32) {
        self.def_sgpr_u32(reg);
        self.def_sgpr_u32(reg + 1);
    }

    pub(crate) fn _def_sgpr_f64(&mut self, reg: u32) {
        self.def_sgpr_u32(reg);
        self.def_sgpr_u32(reg + 1);
    }

    pub(crate) fn def_vgpr_u32(&mut self, reg: u32) {
        self.use_vgpr_u32(reg);
        self.def_vgprs.insert(reg);
    }

    pub(crate) fn def_vgpr_u16(&mut self, reg: u32) {
        self.def_vgpr_u32(reg);
    }

    pub(crate) fn def_vgpr_f32(&mut self, reg: u32) {
        self.def_vgpr_u32(reg);
    }

    pub(crate) fn def_vgpr_f32_vec<const N: usize>(&mut self, reg: u32) {
        for i in 0..N {
            self.def_vgpr_u32(reg + i as u32);
        }
    }

    pub(crate) fn def_vgpr_u64(&mut self, reg: u32) {
        self.def_vgpr_u32(reg);
        self.def_vgpr_u32(reg + 1);
    }

    pub(crate) fn def_vgpr_f64(&mut self, reg: u32) {
        self.def_vgpr_u32(reg);
        self.def_vgpr_u32(reg + 1);
    }
}

impl RDNATranslator {
    pub(crate) fn analyze_instructions(inst: &InstFormat, reg_usage: &mut RegisterUsage) {
        match inst {
            InstFormat::SOPP(inst) => match inst.op {
                I::S_CLAUSE => {}
                I::S_WAIT_KMCNT => {}
                I::S_DELAY_ALU => {}
                I::S_WAIT_ALU => {}
                I::S_WAIT_LOADCNT => {}
                I::S_CBRANCH_SCC0 => {}
                I::S_CBRANCH_SCC1 => {}
                I::S_BRANCH => {}
                I::S_ENDPGM => {}
                I::S_WAIT_BVHCNT => {}
                I::S_WAIT_SAMPLECNT => {}
                I::S_WAIT_STORECNT => {}
                I::S_WAIT_LOADCNT_DSCNT => {}
                I::S_WAIT_DSCNT => {}
                I::S_BARRIER_WAIT => {}
                I::S_NOP => {}
                I::S_SENDMSG => {}
                I::S_CBRANCH_EXECZ => {
                    reg_usage.use_sgpr_u32(126);
                }
                I::S_CBRANCH_EXECNZ => {
                    reg_usage.use_sgpr_u32(126);
                }
                I::S_CBRANCH_VCCZ => {
                    reg_usage.use_sgpr_u32(106);
                }
                I::S_CBRANCH_VCCNZ => {
                    reg_usage.use_sgpr_u32(106);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOPC(inst) => match inst.op {
                I::V_CMP_GT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_EQ_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_NE_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_LT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMPX_NE_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_GT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMP_GT_U64 => {
                    reg_usage.use_operand_u64(&inst.src0);
                    reg_usage.use_vgpr_u64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_EQ_U64 => {
                    reg_usage.use_operand_u64(&inst.src0);
                    reg_usage.use_vgpr_u64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_GT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_LT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_LE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_NLT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_NGT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMPX_NGT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_NGE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_LT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_EQ_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_LT_I32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMP_GT_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_LE_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_LT_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_GE_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_CMP_NGT_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP1(inst) => match inst.op {
                I::V_CVT_F64_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MOV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_NOT_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CLZ_I32_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_RCP_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_RSQ_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_RNDNE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_FRACT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_CVT_I32_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CVT_F64_I32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_READFIRSTLANE_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CVT_F32_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_CVT_U32_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CVT_I32_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_RCP_IFLAG_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_RCP_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_SQRT_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_FREXP_MANT_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_FREXP_EXP_I32_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_RNDNE_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP2(inst) => match inst.op {
                I::V_ADD_NC_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_SUBREV_NC_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ADD_CO_CI_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.use_sgpr_u32(106);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(106);
                }
                I::V_SUB_NC_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_MIN_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_MAX_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_MUL_U32_U24 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_AND_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_OR_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_XOR_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHLREV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHRREV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CNDMASK_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u32(inst.vsrc1 as u32);
                    reg_usage.use_sgpr_u32(106);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHLREV_B64 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_vgpr_u64(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_u64(inst.vdst as u32);
                }
                I::V_ADD_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MUL_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MAX_NUM_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_vgpr_f64(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_ADD_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_SUB_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_MUL_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_FMAC_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.use_vgpr_f32(inst.vdst as u32);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_FMAMK_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_FMAAK_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_vgpr_f32(inst.vsrc1 as u32);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP3(inst) => match inst.op {
                I::V_MAD_U32_U24 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_BFE_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CNDMASK_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_AND_OR_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ADD_NC_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_GT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_EQ_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_NE_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMPX_NE_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMPX_GT_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(126);
                }
                I::V_CMP_EQ_U64 => {
                    reg_usage.use_operand_u64(&inst.src0);
                    reg_usage.use_operand_u64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_NLT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_NGT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_GT_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LG_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_NEQ_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_ADD_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MUL_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_FMA_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.use_operand_f64(&inst.src2);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_DIV_FMAS_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.use_operand_f64(&inst.src2);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_DIV_FIXUP_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.use_operand_f64(&inst.src2);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_LDEXP_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_CMP_CLASS_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_OR3_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_XAD_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_XOR3_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ADD3_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_MUL_LO_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_MAX_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_MIN_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_TRIG_PREOP_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_MAX_NUM_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                }
                I::V_READLANE_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_WRITELANE_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_AND_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_XOR_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ALIGNBIT_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_EQ_U16 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_GT_U16 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_LSHRREV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHLREV_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHL_OR_B32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ADD_LSHL_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHL_ADD_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_ASHRREV_I32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHLREV_B16 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u16(&inst.src1);
                    reg_usage.def_vgpr_u16(inst.vdst as u32);
                }
                I::V_ADD_NC_U16 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_LSHLREV_B64 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u64(&inst.src1);
                    reg_usage.def_vgpr_u64(inst.vdst as u32);
                }
                I::V_LSHRREV_B64 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u64(&inst.src1);
                    reg_usage.def_vgpr_u64(inst.vdst as u32);
                }
                I::V_CVT_F32_F16 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_CVT_F32_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_MUL_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_ADD_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_FMAC_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.use_vgpr_f32(inst.vdst as u32);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_RCP_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_FLOOR_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_SUB_NC_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_MUL_HI_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_GE_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_S_RCP_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.def_sgpr_f32(inst.vdst as u32);
                }
                I::V_FMA_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.use_operand_f32(&inst.src2);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_DIV_FMAS_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.use_operand_f32(&inst.src2);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_DIV_FIXUP_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.use_operand_f32(&inst.src2);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                I::V_CMP_GE_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LT_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LE_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_GT_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_LG_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_CMP_CLASS_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_sgpr_u32(inst.vdst as u32);
                }
                I::V_LDEXP_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP3SD(inst) => match inst.op {
                I::V_DIV_SCALE_F32 => {
                    reg_usage.use_operand_f32(&inst.src0);
                    reg_usage.use_operand_f32(&inst.src1);
                    reg_usage.use_operand_f32(&inst.src2);
                    reg_usage.def_vgpr_f32(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::V_MAD_CO_U64_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u64(&inst.src2);
                    reg_usage.def_vgpr_u64(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::V_DIV_SCALE_F64 => {
                    reg_usage.use_operand_f64(&inst.src0);
                    reg_usage.use_operand_f64(&inst.src1);
                    reg_usage.use_operand_f64(&inst.src2);
                    reg_usage.def_vgpr_f64(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::V_ADD_CO_CI_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::V_ADD_CO_U32 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOP3P(inst) => match inst.op {
                I::V_WMMA_F32_16X16X16_F16 => {
                    reg_usage.use_operand_f16_vec::<8>(&inst.src0);
                    reg_usage.use_operand_f16_vec::<8>(&inst.src1);
                    reg_usage.use_operand_f32_vec::<8>(&inst.src2);
                    reg_usage.def_vgpr_f32_vec::<8>(inst.vdst as u32);
                }
                I::V_FMA_MIXLO_F16 => {
                    reg_usage.use_operand_u32(&inst.src0);
                    reg_usage.use_operand_u32(&inst.src1);
                    reg_usage.use_operand_u32(&inst.src2);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VOPD(inst) => {
                let vdstx = inst.vdstx as u32;
                let vdsty = ((inst.vdsty << 1) | ((inst.vdstx & 1) ^ 1)) as u32;
                match inst.opx {
                    I::V_DUAL_CNDMASK_B32 => {
                        reg_usage.use_operand_u32(&inst.src0x);
                        reg_usage.use_vgpr_u32(inst.vsrc1x as u32);
                        reg_usage.use_sgpr_u32(106);
                        reg_usage.def_vgpr_u32(vdstx);
                    }
                    I::V_DUAL_MOV_B32 => {
                        reg_usage.use_operand_u32(&inst.src0x);
                        reg_usage.def_vgpr_u32(vdstx);
                    }
                    I::V_DUAL_FMAC_F32 => {
                        reg_usage.use_operand_f32(&inst.src0x);
                        reg_usage.use_vgpr_f32(inst.vsrc1x as u32);
                        reg_usage.use_vgpr_f32(vdstx);
                        reg_usage.def_vgpr_f32(vdstx);
                    }
                    I::V_DUAL_ADD_F32 => {
                        reg_usage.use_operand_f32(&inst.src0x);
                        reg_usage.use_vgpr_f32(inst.vsrc1x as u32);
                        reg_usage.def_vgpr_f32(vdstx);
                    }
                    I::V_DUAL_SUB_F32 => {
                        reg_usage.use_operand_f32(&inst.src0x);
                        reg_usage.use_vgpr_f32(inst.vsrc1x as u32);
                        reg_usage.def_vgpr_f32(vdstx);
                    }
                    I::V_DUAL_MUL_F32 => {
                        reg_usage.use_operand_f32(&inst.src0x);
                        reg_usage.use_vgpr_f32(inst.vsrc1x as u32);
                        reg_usage.def_vgpr_f32(vdstx);
                    }
                    I::V_DUAL_FMAAK_F32 => {
                        reg_usage.use_operand_f32(&inst.src0x);
                        reg_usage.use_vgpr_f32(inst.vsrc1x as u32);
                        reg_usage.def_vgpr_f32(vdstx);
                    }
                    _ => {
                        panic!("Unsupported instruction: {:?}", inst);
                    }
                }
                match inst.opy {
                    I::V_DUAL_CNDMASK_B32 => {
                        reg_usage.use_operand_u32(&inst.src0y);
                        reg_usage.use_vgpr_u32(inst.vsrc1y as u32);
                        reg_usage.use_sgpr_u32(106);
                        reg_usage.def_vgpr_u32(vdsty);
                    }
                    I::V_DUAL_MOV_B32 => {
                        reg_usage.use_operand_u32(&inst.src0y);
                        reg_usage.def_vgpr_u32(vdsty);
                    }
                    I::V_DUAL_ADD_NC_U32 => {
                        reg_usage.use_operand_u32(&inst.src0y);
                        reg_usage.use_vgpr_u32(inst.vsrc1y as u32);
                        reg_usage.def_vgpr_u32(vdsty);
                    }
                    I::V_DUAL_LSHLREV_B32 => {
                        reg_usage.use_operand_u32(&inst.src0y);
                        reg_usage.use_vgpr_u32(inst.vsrc1y as u32);
                        reg_usage.def_vgpr_u32(vdsty);
                    }
                    I::V_DUAL_AND_B32 => {
                        reg_usage.use_operand_u32(&inst.src0y);
                        reg_usage.use_vgpr_u32(inst.vsrc1y as u32);
                        reg_usage.def_vgpr_u32(vdsty);
                    }
                    I::V_DUAL_ADD_F32 => {
                        reg_usage.use_operand_f32(&inst.src0y);
                        reg_usage.use_vgpr_f32(inst.vsrc1y as u32);
                        reg_usage.def_vgpr_f32(vdsty);
                    }
                    I::V_DUAL_SUB_F32 => {
                        reg_usage.use_operand_f32(&inst.src0y);
                        reg_usage.use_vgpr_f32(inst.vsrc1y as u32);
                        reg_usage.def_vgpr_f32(vdsty);
                    }
                    I::V_DUAL_MUL_F32 => {
                        reg_usage.use_operand_f32(&inst.src0y);
                        reg_usage.use_vgpr_f32(inst.vsrc1y as u32);
                        reg_usage.def_vgpr_f32(vdsty);
                    }
                    I::V_DUAL_FMAC_F32 => {
                        reg_usage.use_operand_f32(&inst.src0y);
                        reg_usage.use_vgpr_f32(inst.vsrc1y as u32);
                        reg_usage.use_vgpr_f32(vdsty);
                        reg_usage.def_vgpr_f32(vdsty);
                    }
                    I::V_DUAL_FMAAK_F32 => {
                        reg_usage.use_operand_f32(&inst.src0y);
                        reg_usage.use_vgpr_f32(inst.vsrc1y as u32);
                        reg_usage.def_vgpr_f32(vdsty);
                    }
                    _ => {
                        panic!("Unsupported instruction: {:?}", inst);
                    }
                }
            }
            InstFormat::SMEM(inst) => match inst.op {
                I::S_LOAD_U16 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    reg_usage.def_sgpr_u32(inst.sdata as u32);
                }
                I::S_LOAD_B32 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    reg_usage.def_sgpr_u32(inst.sdata as u32);
                }
                I::S_LOAD_B64 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    for i in 0..2 {
                        reg_usage.def_sgpr_u32(inst.sdata as u32 + i);
                    }
                }
                I::S_LOAD_B96 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    for i in 0..3 {
                        reg_usage.def_sgpr_u32(inst.sdata as u32 + i);
                    }
                }
                I::S_LOAD_B128 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    for i in 0..4 {
                        reg_usage.def_sgpr_u32(inst.sdata as u32 + i);
                    }
                }
                I::S_LOAD_B256 => {
                    reg_usage.use_sgpr_u64(inst.sbase as u32 * 2);
                    for i in 0..8 {
                        reg_usage.def_sgpr_u32(inst.sdata as u32 + i);
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOP1(inst) => match inst.op {
                I::S_BARRIER_SIGNAL => {}
                I::S_AND_NOT1_SAVEEXEC_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_sgpr_u32(126);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::S_AND_SAVEEXEC_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_sgpr_u32(126);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::S_OR_SAVEEXEC_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_sgpr_u32(126);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                    reg_usage.def_sgpr_u32(126);
                }
                I::S_MOV_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_MOV_B64 => {
                    reg_usage.use_operand_u64(&inst.ssrc0);
                    reg_usage.def_sgpr_u64(inst.sdst as u32);
                }
                I::S_CTZ_I32_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_CVT_F32_I32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.def_sgpr_f32(inst.sdst as u32);
                }
                I::S_CVT_F32_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.def_sgpr_f32(inst.sdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOP2(inst) => match inst.op {
                I::S_AND_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_OR_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_XOR_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_AND_NOT1_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_OR_NOT1_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_LSHR_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_CSELECT_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_BFM_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_ADD_NC_U64 => {
                    reg_usage.use_operand_u64(&inst.ssrc0);
                    reg_usage.use_operand_u64(&inst.ssrc1);
                    reg_usage.def_sgpr_u64(inst.sdst as u32);
                }
                I::S_ADD_CO_I32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_SUB_CO_I32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_MUL_I32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_MUL_HI_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_BFE_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_LSHL_B32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_MAX_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                I::S_LSHL_B64 => {
                    reg_usage.use_operand_u64(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                    reg_usage.def_sgpr_u64(inst.sdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOPC(inst) => match inst.op {
                I::S_CMP_LG_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_EQ_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_LT_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_GE_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_GT_U32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_LT_I32 => {
                    reg_usage.use_operand_u32(&inst.ssrc0);
                    reg_usage.use_operand_u32(&inst.ssrc1);
                }
                I::S_CMP_LG_U64 => {
                    reg_usage.use_operand_u64(&inst.ssrc0);
                    reg_usage.use_operand_u64(&inst.ssrc1);
                }
                I::S_CMP_EQ_U64 => {
                    reg_usage.use_operand_u64(&inst.ssrc0);
                    reg_usage.use_operand_u64(&inst.ssrc1);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::SOPK(inst) => match inst.op {
                I::S_MOVK_I32 => {
                    reg_usage.def_sgpr_u32(inst.sdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VFLAT(inst) => match inst.op {
                I::FLAT_LOAD_B32 => {
                    reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::FLAT_LOAD_B64 => {
                    reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    for i in 0..2 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::FLAT_LOAD_B128 => {
                    reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    for i in 0..4 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::FLAT_STORE_B32 => {
                    reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    reg_usage.use_vgpr_u32(inst.vsrc as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VGLOBAL(inst) => match inst.op {
                I::GLOBAL_LOAD_U8 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..1 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_LOAD_U16 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..1 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_LOAD_B32 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..1 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_LOAD_B64 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..2 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_LOAD_B128 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..4 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::GLOBAL_STORE_B16 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    reg_usage.use_vgpr_u32(inst.vsrc as u32);
                }
                I::GLOBAL_STORE_B32 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    reg_usage.use_vgpr_u32(inst.vsrc as u32);
                }
                I::GLOBAL_STORE_B64 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..2 {
                        reg_usage.use_vgpr_u32(inst.vsrc as u32 + i);
                    }
                }
                I::GLOBAL_STORE_B128 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    for i in 0..4 {
                        reg_usage.use_vgpr_u32(inst.vsrc as u32 + i);
                    }
                }
                I::GLOBAL_ATOMIC_ADD_U32 => {
                    if inst.saddr != 124 {
                        reg_usage.use_sgpr_u64(inst.saddr as u32);
                        reg_usage.use_vgpr_u32(inst.vaddr as u32);
                    } else {
                        reg_usage.use_vgpr_u64(inst.vaddr as u32);
                    }

                    reg_usage.use_vgpr_u32(inst.vsrc as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::GLOBAL_WB => {}
                I::GLOBAL_INV => {}
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VSCRATCH(inst) => match inst.op {
                I::SCRATCH_LOAD_B32 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::SCRATCH_LOAD_B64 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    for i in 0..2 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::SCRATCH_LOAD_B96 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    for i in 0..3 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::SCRATCH_LOAD_B128 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    for i in 0..4 {
                        reg_usage.def_vgpr_u32(inst.vdst as u32 + i);
                    }
                }
                I::SCRATCH_STORE_B32 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    reg_usage.use_vgpr_u32(inst.vsrc as u32);
                }
                I::SCRATCH_STORE_B64 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    for i in 0..2 {
                        reg_usage.use_vgpr_u32(inst.vsrc as u32 + i);
                    }
                }
                I::SCRATCH_STORE_B128 => {
                    reg_usage.use_sgpr_u32(inst.saddr as u32);
                    for i in 0..4 {
                        reg_usage.use_vgpr_u32(inst.vsrc as u32 + i);
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::DS(inst) => match inst.op {
                I::DS_LOAD_U8 => {
                    reg_usage.use_vgpr_u32(inst.addr as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                I::DS_STORE_B8 => {
                    reg_usage.use_vgpr_u32(inst.addr as u32);
                    reg_usage.use_vgpr_u32(inst.data0 as u32);
                }
                I::DS_BPERMUTE_B32 => {
                    reg_usage.use_vgpr_u32(inst.addr as u32);
                    reg_usage.use_vgpr_u32(inst.data0 as u32);
                    reg_usage.def_vgpr_u32(inst.vdst as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VIMAGE(inst) => match inst.op {
                I::IMAGE_BVH64_INTERSECT_RAY => {
                    reg_usage.use_vgpr_u64(inst.vaddr0 as u32);
                    reg_usage.use_vgpr_f32(inst.vaddr1 as u32);
                    for i in 0..3 {
                        reg_usage.use_vgpr_f32(inst.vaddr2 as u32 + i);
                    }
                    for i in 0..3 {
                        reg_usage.use_vgpr_f32(inst.vaddr3 as u32 + i);
                    }
                    for i in 0..3 {
                        reg_usage.use_vgpr_f32(inst.vaddr4 as u32 + i);
                    }
                    for i in 0..4 {
                        reg_usage.def_vgpr_u32(inst.vdata as u32 + i);
                    }
                }
                I::IMAGE_BVH8_INTERSECT_RAY => {
                    reg_usage.use_vgpr_u64(inst.vaddr0 as u32);
                    reg_usage.use_vgpr_f32(inst.vaddr1 as u32);
                    reg_usage.use_vgpr_u32(inst.vaddr1 as u32 + 1);
                    for i in 0..3 {
                        reg_usage.use_vgpr_f32(inst.vaddr2 as u32 + i);
                    }
                    for i in 0..3 {
                        reg_usage.use_vgpr_f32(inst.vaddr3 as u32 + i);
                    }
                    reg_usage.use_vgpr_u32(inst.vaddr4 as u32);
                    for i in 0..10 {
                        reg_usage.def_vgpr_u32(inst.vdata as u32 + i);
                    }
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
            InstFormat::VSAMPLE(inst) => match inst.op {
                I::IMAGE_SAMPLE_LZ => {
                    reg_usage.use_vgpr_f32(inst.vaddr0 as u32);
                    reg_usage.use_vgpr_f32(inst.vaddr1 as u32);
                    for i in 0..8 {
                        reg_usage.use_sgpr_u32(inst.rsrc as u32 + i);
                    }
                    for i in 0..4 {
                        reg_usage.use_sgpr_u32(inst.samp as u32 + i);
                    }
                    reg_usage.def_vgpr_u32(inst.vdata as u32);
                }
                _ => {
                    panic!("Unsupported instruction: {:?}", inst);
                }
            },
        }
    }
}
