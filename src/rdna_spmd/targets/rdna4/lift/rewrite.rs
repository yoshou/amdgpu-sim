use super::regs::Word;
use super::*;

pub(super) const VGPR_BASE: u32 = 512;

const SGPR_NULL: u32 = 124;

pub(super) struct InstEffects {
    pub reads: Vec<u32>,

    pub kills: Vec<u32>,
}

impl InstEffects {
    fn unknown() -> Self {
        InstEffects {
            reads: Vec::new(),
            kills: Vec::new(),
        }
    }

    fn known(reads: Vec<u32>, kills: Vec<u32>) -> Self {
        InstEffects { reads, kills }
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

#[derive(Default)]
struct Effects {
    reads: Vec<u32>,
    kills: Vec<u32>,
}

impl Effects {
    fn read(mut self, op: &SourceOperand, words: u32) -> Self {
        read_src(op, words, &mut self.reads);
        self
    }
    fn read_sgpr(mut self, reg: u32, words: u32) -> Self {
        read_sgpr(reg, words, &mut self.reads);
        self
    }
    fn read_vgpr(mut self, reg: u32, words: u32) -> Self {
        read_vgpr(reg, words, &mut self.reads);
        self
    }
    fn read_exec(self) -> Self {
        self.read_sgpr(126, 1)
    }
    fn read_vcc(self) -> Self {
        self.read_sgpr(106, 1)
    }
    fn kill_sgpr(mut self, reg: u32, words: u32) -> Self {
        kill_sgpr(reg, words, &mut self.kills);
        self
    }
    fn kill_vgpr(mut self, reg: u32, words: u32) -> Self {
        kill_vgpr(reg, words, &mut self.kills);
        self
    }
    fn known(self) -> InstEffects {
        InstEffects::known(self.reads, self.kills)
    }
}

fn sopp_effects(op: I) -> InstEffects {
    match op {
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
        | I::S_ENDPGM => Effects::default().known(),
        I::S_CBRANCH_VCCZ | I::S_CBRANCH_VCCNZ => Effects::default().read_vcc().known(),
        I::S_CBRANCH_EXECZ | I::S_CBRANCH_EXECNZ => Effects::default().read_exec().known(),
        I::S_CBRANCH_SCC0 | I::S_CBRANCH_SCC1 => Effects::default().known(),
        _ => InstEffects::unknown(),
    }
}

fn sop1_effects(inst: &crate::rdna_instructions::SOP1) -> InstEffects {
    match inst.op {
        I::S_MOV_B32 => Effects::default()
            .read(&inst.ssrc0, 1)
            .kill_sgpr(inst.sdst as u32, 1)
            .known(),
        I::S_MOV_B64 => Effects::default()
            .read(&inst.ssrc0, 2)
            .kill_sgpr(inst.sdst as u32, 2)
            .known(),
        I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 | I::S_OR_SAVEEXEC_B32 => {
            Effects::default()
                .read(&inst.ssrc0, 1)
                .read_exec()
                .kill_sgpr(inst.sdst as u32, 1)
                .kill_sgpr(126, 1)
                .known()
        }
        _ => InstEffects::unknown(),
    }
}

fn sop2_effects(inst: &crate::rdna_instructions::SOP2) -> InstEffects {
    match inst.op {
        I::S_AND_B32
        | I::S_OR_B32
        | I::S_XOR_B32
        | I::S_AND_NOT1_B32
        | I::S_CSELECT_B32
        | I::S_LSHL_B32
        | I::S_LSHR_B32
        | I::S_MUL_I32 => Effects::default()
            .read(&inst.ssrc0, 1)
            .read(&inst.ssrc1, 1)
            .kill_sgpr(inst.sdst as u32, 1)
            .known(),
        I::S_ADD_NC_U64 => Effects::default()
            .read(&inst.ssrc0, 2)
            .read(&inst.ssrc1, 2)
            .kill_sgpr(inst.sdst as u32, 2)
            .known(),
        _ => InstEffects::unknown(),
    }
}

fn vopc_effects(inst: &crate::rdna_instructions::VOPC) -> InstEffects {
    let name = format!("{:?}", inst.op);
    let effects = Effects::default()
        .read(&inst.src0, 2)
        .read_vgpr(inst.vsrc1 as u32, 2)
        .read_exec();
    if name.starts_with("V_CMPX_") {
        effects.kill_sgpr(126, 1).known()
    } else {
        effects.kill_sgpr(106, 1).known()
    }
}

fn vop1_effects(inst: &crate::rdna_instructions::VOP1) -> InstEffects {
    match vop1_widths(&inst.op) {
        Some((src_words, dst_words)) => Effects::default()
            .read(&inst.src0, src_words)
            .kill_vgpr(inst.vdst as u32, dst_words)
            .known(),
        None => InstEffects::unknown(),
    }
}

fn vop2_effects(inst: &crate::rdna_instructions::VOP2) -> InstEffects {
    match vop3_arith_widths(&inst.op) {
        Some((src_words, dst_words)) => {
            let mut effects = Effects::default()
                .read(&inst.src0, src_words)
                .read_vgpr(inst.vsrc1 as u32, src_words);
            if let I::V_CNDMASK_B32 = inst.op {
                effects = effects.read_vcc();
            }
            effects.kill_vgpr(inst.vdst as u32, dst_words).known()
        }
        None => InstEffects::unknown(),
    }
}

fn vop3_effects(inst: &crate::rdna_instructions::VOP3) -> InstEffects {
    let name = format!("{:?}", inst.op);
    if name.starts_with("V_CMPX_") {
        return Effects::default()
            .read(&inst.src0, 2)
            .read(&inst.src1, 2)
            .read_exec()
            .kill_sgpr(126, 1)
            .known();
    }
    if name.starts_with("V_CMP_") {

        return Effects::default()
            .read(&inst.src0, 2)
            .read(&inst.src1, 2)
            .kill_sgpr(inst.vdst as u32, 1)
            .known();
    }
    match vop3_arith_widths(&inst.op) {
        Some((src_words, dst_words)) => {
            let src1_words = match inst.op {
                I::V_LDEXP_F64 | I::V_TRIG_PREOP_F64 => 1,
                _ => src_words,
            };
            let mut effects = Effects::default()
                .read(&inst.src0, src_words)
                .read(&inst.src1, src1_words)
                .read(&inst.src2, src_words);
            if let I::V_DIV_FMAS_F64 = inst.op {
                effects = effects.read_vcc();
            }
            effects.kill_vgpr(inst.vdst as u32, dst_words).known()
        }
        None => InstEffects::unknown(),
    }
}

fn vop3sd_effects(inst: &crate::rdna_instructions::VOP3SD) -> InstEffects {
    let words = match inst.op {
        I::V_DIV_SCALE_F64 => [2, 2, 2, 2],
        I::V_MAD_CO_U64_U32 => [1, 1, 2, 2],
        I::V_ADD_CO_CI_U32 | I::V_SUB_CO_CI_U32 => [1, 1, 1, 1],
        _ => return InstEffects::unknown(),
    };
    Effects::default()
        .read(&inst.src0, words[0])
        .read(&inst.src1, words[1])
        .read(&inst.src2, words[2])
        .kill_vgpr(inst.vdst as u32, words[3])
        .kill_sgpr(inst.sdst as u32, 1)
        .known()
}

fn vglobal_effects(inst: &crate::rdna_instructions::VGLOBAL) -> InstEffects {
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
    let mut effects = Effects::default().read_vgpr(inst.vaddr as u32, 2);
    if inst.saddr != SGPR_NULL as u8 {
        effects = effects.read_sgpr(inst.saddr as u32, 2);
    }
    let effects = effects.read_exec();
    if name.starts_with("GLOBAL_LOAD") {

        effects.read_vgpr(inst.vdst as u32, data_words).known()
    } else if name.starts_with("GLOBAL_STORE") {
        effects.read_vgpr(inst.vsrc as u32, data_words).known()
    } else {
        InstEffects::unknown()
    }
}

fn vopd_half(op: &I, src0: &SourceOperand, vsrc1: u8, vdst: u32) -> Option<Effects> {
    let effects = match op {
        I::V_DUAL_MOV_B32 => Effects::default().read(src0, 1),
        I::V_DUAL_CNDMASK_B32 => Effects::default()
            .read(src0, 1)
            .read_vgpr(vsrc1 as u32, 1)
            .read_vcc(),
        I::V_DUAL_ADD_F32
        | I::V_DUAL_MUL_F32
        | I::V_DUAL_AND_B32
        | I::V_DUAL_ADD_NC_U32
        | I::V_DUAL_LSHLREV_B32 => Effects::default().read(src0, 1).read_vgpr(vsrc1 as u32, 1),
        I::V_DUAL_FMAC_F32 => Effects::default()
            .read(src0, 1)
            .read_vgpr(vsrc1 as u32, 1)
            .read_vgpr(vdst, 1),
        _ => return None,
    };
    Some(effects.kill_vgpr(vdst, 1))
}

fn vopd_effects(inst: &crate::rdna_instructions::VOPD) -> InstEffects {

    let dy = ((inst.vdsty as u32) << 1) | (((inst.vdstx as u32) & 1) ^ 1);
    match (
        vopd_half(&inst.opx, &inst.src0x, inst.vsrc1x, inst.vdstx as u32),
        vopd_half(&inst.opy, &inst.src0y, inst.vsrc1y, dy),
    ) {
        (Some(x), Some(y)) => {
            let mut reads = x.reads;
            reads.extend(y.reads);
            let mut kills = x.kills;
            kills.extend(y.kills);
            InstEffects::known(reads, kills)
        }
        _ => InstEffects::unknown(),
    }
}

pub(super) fn effects_of(inst: &InstFormat) -> InstEffects {
    match inst {
        InstFormat::SOPP(inst) => sopp_effects(inst.op),
        InstFormat::SOP1(inst) => sop1_effects(inst),
        InstFormat::SOP2(inst) => sop2_effects(inst),

        InstFormat::SOPC(inst) => Effects::default()
            .read(&inst.ssrc0, 2)
            .read(&inst.ssrc1, 2)
            .known(),

        InstFormat::SOPK(inst) => Effects::default().read_sgpr(inst.sdst as u32, 2).known(),
        InstFormat::VOPC(inst) => vopc_effects(inst),
        InstFormat::VOP1(inst) => vop1_effects(inst),
        InstFormat::VOP2(inst) => vop2_effects(inst),
        InstFormat::VOP3(inst) => vop3_effects(inst),
        InstFormat::VOP3SD(inst) => vop3sd_effects(inst),
        InstFormat::VGLOBAL(inst) => vglobal_effects(inst),
        InstFormat::VOPD(inst) => vopd_effects(inst),
        _ => InstEffects::unknown(),
    }
}

pub(super) fn word(slot: u32) -> Option<Word> {
    if slot >= VGPR_BASE {
        Some(Word::Vgpr(slot - VGPR_BASE))
    } else {
        Word::scalar(slot)
    }
}
