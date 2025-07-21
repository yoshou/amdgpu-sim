use crate::bit::*;
use crate::instructions::*;
use crate::rdna_instructions::*;

fn decode_sop1_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        0 => Ok((I::S_MOV_B32, 4)),
        1 => Ok((I::S_MOV_B64, 4)),
        _ => Err(()),
    }
}

fn decode_sop2_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        0 => Ok((I::S_ADD_CO_U32, 4)),
        1 => Ok((I::S_SUB_CO_U32, 4)),
        2 => Ok((I::S_ADD_CO_I32, 4)),
        3 => Ok((I::S_SUB_CO_I32, 4)),
        4 => Ok((I::S_ADD_CO_CI_U32, 4)),
        5 => Ok((I::S_SUB_CO_CI_U32, 4)),
        6 => Ok((I::S_ABSDIFF_I32, 4)),
        8 => Ok((I::S_LSHL_B32, 4)),
        9 => Ok((I::S_LSHL_B64, 4)),
        10 => Ok((I::S_LSHR_B32, 4)),
        11 => Ok((I::S_LSHR_B64, 4)),
        12 => Ok((I::S_ASHR_I32, 4)),
        13 => Ok((I::S_ASHR_I64, 4)),
        14 => Ok((I::S_LSHL1_ADD_U32, 4)),
        15 => Ok((I::S_LSHL2_ADD_U32, 4)),
        16 => Ok((I::S_LSHL3_ADD_U32, 4)),
        17 => Ok((I::S_LSHL4_ADD_U32, 4)),
        18 => Ok((I::S_MIN_I32, 4)),
        19 => Ok((I::S_MIN_U32, 4)),
        20 => Ok((I::S_MAX_I32, 4)),
        21 => Ok((I::S_MAX_U32, 4)),
        22 => Ok((I::S_AND_B32, 4)),
        23 => Ok((I::S_AND_B64, 4)),
        24 => Ok((I::S_OR_B32, 4)),
        25 => Ok((I::S_OR_B64, 4)),
        26 => Ok((I::S_XOR_B32, 4)),
        27 => Ok((I::S_XOR_B64, 4)),
        28 => Ok((I::S_NAND_B32, 4)),
        29 => Ok((I::S_NAND_B64, 4)),
        30 => Ok((I::S_NOR_B32, 4)),
        31 => Ok((I::S_NOR_B64, 4)),
        32 => Ok((I::S_XNOR_B32, 4)),
        33 => Ok((I::S_XNOR_B64, 4)),
        34 => Ok((I::S_AND_NOT1_B32, 4)),
        35 => Ok((I::S_AND_NOT1_B64, 4)),
        36 => Ok((I::S_OR_NOT1_B32, 4)),
        37 => Ok((I::S_OR_NOT1_B64, 4)),
        38 => Ok((I::S_BFE_U32, 4)),
        39 => Ok((I::S_BFE_I32, 4)),
        40 => Ok((I::S_BFE_U64, 4)),
        41 => Ok((I::S_BFE_I64, 4)),
        42 => Ok((I::S_BFM_B32, 4)),
        43 => Ok((I::S_BFM_B64, 4)),
        44 => Ok((I::S_MUL_I32, 4)),
        45 => Ok((I::S_MUL_HI_U32, 4)),
        46 => Ok((I::S_MUL_HI_I32, 4)),
        48 => Ok((I::S_CSELECT_B32, 4)),
        49 => Ok((I::S_CSELECT_B64, 4)),
        50 => Ok((I::S_PACK_LL_B32_B16, 4)),
        51 => Ok((I::S_PACK_LH_B32_B16, 4)),
        52 => Ok((I::S_PACK_HH_B32_B16, 4)),
        53 => Ok((I::S_PACK_HL_B32_B16, 4)),
        64 => Ok((I::S_ADD_F32, 4)),
        65 => Ok((I::S_SUB_F32, 4)),
        66 => Ok((I::S_MIN_NUM_F32, 4)),
        67 => Ok((I::S_MAX_NUM_F32, 4)),
        68 => Ok((I::S_MUL_F32, 4)),
        69 => Ok((I::S_FMAAK_F32, 4)),
        70 => Ok((I::S_FMAMK_F32, 4)),
        71 => Ok((I::S_FMAC_F32, 4)),
        72 => Ok((I::S_CVT_PK_RTZ_F16_F32, 4)),
        73 => Ok((I::S_ADD_F16, 4)),
        74 => Ok((I::S_SUB_F16, 4)),
        75 => Ok((I::S_MIN_NUM_F16, 4)),
        76 => Ok((I::S_MAX_NUM_F16, 4)),
        77 => Ok((I::S_MUL_F16, 4)),
        78 => Ok((I::S_FMAC_F16, 4)),
        79 => Ok((I::S_MINIMUM_F32, 4)),
        80 => Ok((I::S_MAXIMUM_F32, 4)),
        81 => Ok((I::S_MINIMUM_F16, 4)),
        82 => Ok((I::S_MAXIMUM_F16, 4)),
        83 => Ok((I::S_ADD_NC_U64, 4)),
        84 => Ok((I::S_SUB_NC_U64, 4)),
        85 => Ok((I::S_MUL_U64, 4)),
        _ => Err(()),
    }
}

fn decode_sopk_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        _ => Err(()),
    }
}

fn decode_sopc_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        _ => Err(()),
    }
}

fn decode_sopp_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        0 => Ok((I::S_NOP, 4)),
        1 => Ok((I::S_SETKILL, 4)),
        2 => Ok((I::S_SETHALT, 4)),
        3 => Ok((I::S_SLEEP, 4)),
        5 => Ok((I::S_CLAUSE, 4)),
        7 => Ok((I::S_DELAY_ALU, 4)),
        8 => Ok((I::S_WAIT_ALU, 4)),
        9 => Ok((I::S_WAITCNT, 4)),
        10 => Ok((I::S_WAIT_IDLE, 4)),
        11 => Ok((I::S_WAIT_EVENT, 4)),
        16 => Ok((I::S_TRAP, 4)),
        17 => Ok((I::S_ROUND_MODE, 4)),
        18 => Ok((I::S_DENORM_MODE, 4)),
        20 => Ok((I::S_BARRIER_WAIT, 4)),
        31 => Ok((I::S_CODE_END, 4)),
        32 => Ok((I::S_BRANCH, 4)),
        33 => Ok((I::S_CBRANCH_SCC0, 4)),
        34 => Ok((I::S_CBRANCH_SCC1, 4)),
        35 => Ok((I::S_CBRANCH_VCCZ, 4)),
        36 => Ok((I::S_CBRANCH_VCCNZ, 4)),
        37 => Ok((I::S_CBRANCH_EXECZ, 4)),
        38 => Ok((I::S_CBRANCH_EXECNZ, 4)),
        48 => Ok((I::S_ENDPGM, 4)),
        49 => Ok((I::S_ENDPGM_SAVED, 4)),
        52 => Ok((I::S_WAKEUP, 4)),
        53 => Ok((I::S_SETPRIO, 4)),
        54 => Ok((I::S_SENDMSG, 4)),
        55 => Ok((I::S_SENDMSGHALT, 4)),
        56 => Ok((I::S_INCPERFLEVEL, 4)),
        57 => Ok((I::S_DECPERFLEVEL, 4)),
        60 => Ok((I::S_ICACHE_INV, 4)),
        64 => Ok((I::S_WAIT_LOADCNT, 4)),
        65 => Ok((I::S_WAIT_STORECNT, 4)),
        66 => Ok((I::S_WAIT_SAMPLECNT, 4)),
        67 => Ok((I::S_WAIT_BVHCNT, 4)),
        68 => Ok((I::S_WAIT_EXPCNT, 4)),
        70 => Ok((I::S_WAIT_DSCNT, 4)),
        71 => Ok((I::S_WAIT_KMCNT, 4)),
        72 => Ok((I::S_WAIT_LOADCNT_DSCNT, 4)),
        73 => Ok((I::S_WAIT_STORECNT_DSCNT, 4)),
        _ => Err(()),
    }
}

fn decode_vop1_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        _ => Err(()),
    }
}

fn decode_vop2_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        _ => Err(()),
    }
}

fn decode_vop3_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        283 => Ok((I::V_AND_B32, 8)),
        864 => Ok((I::V_READLANE_B32, 8)),
        865 => Ok((I::V_WRITELANE_B32, 8)),
        _ => Err(()),
    }
}

fn decode_vopc_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        _ => Err(()),
    }
}

fn decode_smem_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        0 => Ok((I::S_LOAD_B32, 8)),
        1 => Ok((I::S_LOAD_B64, 8)),
        2 => Ok((I::S_LOAD_B128, 8)),
        3 => Ok((I::S_LOAD_B256, 8)),
        4 => Ok((I::S_LOAD_B512, 8)),
        5 => Ok((I::S_LOAD_B96, 8)),
        8 => Ok((I::S_LOAD_I8, 8)),
        9 => Ok((I::S_LOAD_U8, 8)),
        10 => Ok((I::S_LOAD_I16, 8)),
        11 => Ok((I::S_LOAD_U16, 8)),
        16 => Ok((I::S_BUFFER_LOAD_B32, 8)),
        17 => Ok((I::S_BUFFER_LOAD_B64, 8)),
        18 => Ok((I::S_BUFFER_LOAD_B128, 8)),
        19 => Ok((I::S_BUFFER_LOAD_B256, 8)),
        20 => Ok((I::S_BUFFER_LOAD_B512, 8)),
        21 => Ok((I::S_BUFFER_LOAD_B96, 8)),
        24 => Ok((I::S_BUFFER_LOAD_I8, 8)),
        25 => Ok((I::S_BUFFER_LOAD_U8, 8)),
        26 => Ok((I::S_BUFFER_LOAD_I16, 8)),
        27 => Ok((I::S_BUFFER_LOAD_U16, 8)),
        33 => Ok((I::S_DCACHE_INV, 8)),
        36 => Ok((I::S_PREFETCH_INST, 8)),
        37 => Ok((I::S_PREFETCH_INST_PC_REL, 8)),
        38 => Ok((I::S_PREFETCH_DATA, 8)),
        39 => Ok((I::S_BUFFER_PREFETCH_DATA, 8)),
        40 => Ok((I::S_PREFETCH_DATA_PC_REL, 8)),
        _ => Err(()),
    }
}

fn max<T: std::cmp::Ord>(a: T, b: T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

pub fn decode_rdna4(inst: u64) -> Result<(InstFormat, usize), ()> {
    if (get_bits(inst, 31, 23) as u32) == SOP1_ENCODE {
        let ssrc0 = get_bits(inst, 7, 0) as u8;
        let (op, size) = decode_sop1_opcode_rdna4(get_bits(inst, 15, 8) as u32)?;
        Ok((
            InstFormat::SOP1(SOP1 {
                ssrc0,
                op,
                sdst: get_bits(inst, 22, 16) as u8,
            }),
            if ssrc0 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 23) as u32) == SOPC_ENCODE {
        let (op, size) = decode_sopc_opcode_rdna4(get_bits(inst, 22, 16) as u32)?;
        Ok((
            InstFormat::SOPC(SOPC {
                ssrc0: get_bits(inst, 7, 0) as u8,
                ssrc1: get_bits(inst, 15, 8) as u8,
                op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 23) as u32) == SOPP_ENCODE {
        let (op, size) = decode_sopp_opcode_rdna4(get_bits(inst, 22, 16) as u32)?;
        Ok((
            InstFormat::SOPP(SOPP {
                simm16: get_bits(inst, 15, 0) as u16,
                op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 28) as u32) == SOPK_ENCODE {
        let (op, size) = decode_sopk_opcode_rdna4(get_bits(inst, 27, 23) as u32)?;
        Ok((
            InstFormat::SOPK(SOPK {
                simm16: get_bits(inst, 15, 0) as u16,
                sdst: get_bits(inst, 22, 16) as u8,
                op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 30) as u32) == SOP2_ENCODE {
        let ssrc1 = get_bits(inst, 15, 8) as u8;
        let (op, size) = decode_sop2_opcode_rdna4(get_bits(inst, 29, 23) as u32)?;
        Ok((
            InstFormat::SOP2(SOP2 {
                ssrc0: get_bits(inst, 7, 0) as u8,
                ssrc1,
                sdst: get_bits(inst, 22, 16) as u8,
                op,
            }),
            if ssrc1 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 25) as u32) == VOPC_ENCODE {
        let (op, size) = decode_vopc_opcode_rdna4(get_bits(inst, 24, 17) as u32)?;
        Ok((
            InstFormat::VOPC(VOPC {
                src0: get_bits(inst, 8, 0) as u16,
                vsrc1: get_bits(inst, 16, 9) as u8,
                op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 25) as u32) == VOP1_ENCODE {
        let src0 = get_bits(inst, 8, 0) as u16;
        let (op, size) = decode_vop1_opcode_rdna4(get_bits(inst, 16, 9) as u32)?;
        Ok((
            InstFormat::VOP1(VOP1 {
                src0,
                op,
                vdst: get_bits(inst, 24, 17) as u8,
            }),
            if src0 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 31) as u32) == VOP2_ENCODE {
        let src0 = get_bits(inst, 8, 0) as u16;
        let (op, size) = decode_vop2_opcode_rdna4(get_bits(inst, 30, 25) as u32)?;
        Ok((
            InstFormat::VOP2(VOP2 {
                src0,
                vsrc1: get_bits(inst, 16, 9) as u8,
                vdst: get_bits(inst, 24, 17) as u8,
                op,
            }),
            if src0 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 26) as u32) == VOP3_ENCODE {
        let (op, size) = decode_vop3_opcode_rdna4(get_bits(inst, 25, 16) as u32)?;
        Ok((
            InstFormat::VOP3(VOP3 {
                vdst: get_bits(inst, 7, 0) as u8,
                abs: get_bits(inst, 10, 8) as u8,
                opsel: get_bits(inst, 14, 1) as u8,
                cm: get_bits(inst, 15, 15) as u8,
                op,
                src0: get_bits(inst, 40, 32) as u16,
                src1: get_bits(inst, 49, 41) as u16,
                src2: get_bits(inst, 58, 50) as u16,
                omod: get_bits(inst, 60, 59) as u8,
                neg: get_bits(inst, 63, 61) as u8,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 26) as u32) == SMEM_ENCODE {
        let (op, size) = decode_smem_opcode_rdna4(get_bits(inst, 18, 13) as u32)?;
        Ok((
            InstFormat::SMEM(SMEM {
                op,
                sdata: get_bits(inst, 12, 6) as u8,
                sbase: get_bits(inst, 5, 0) as u8,
                ioffset: get_bits(inst, 55, 32) as u32,
                soffset: get_bits(inst, 63, 57) as u8,
                scope: get_bits(inst, 22, 21) as u8,
                th: get_bits(inst, 24, 23) as u8,
            }),
            size,
        ))
    } else {
        Err(())
    }
}
