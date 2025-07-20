use crate::gcn_instructions::*;
use crate::instructions::*;

fn decode_sop1_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        0 => Ok((I::S_MOV_B32, 4)),
        1 => Ok((I::S_MOV_B64, 4)),
        2 => Ok((I::S_CMOV_B32, 4)),
        3 => Ok((I::S_CMOV_B64, 4)),
        4 => Ok((I::S_NOT_B32, 4)),
        5 => Ok((I::S_NOT_B64, 4)),
        6 => Ok((I::S_WQM_B32, 4)),
        7 => Ok((I::S_WQM_B64, 4)),
        8 => Ok((I::S_BREV_B32, 4)),
        9 => Ok((I::S_BREV_B64, 4)),
        10 => Ok((I::S_BCNT0_I32_B32, 4)),
        11 => Ok((I::S_BCNT0_I32_B64, 4)),
        12 => Ok((I::S_BCNT1_I32_B32, 4)),
        13 => Ok((I::S_BCNT1_I32_B64, 4)),
        14 => Ok((I::S_FF0_I32_B32, 4)),
        15 => Ok((I::S_FF0_I32_B64, 4)),
        16 => Ok((I::S_FF1_I32_B32, 4)),
        17 => Ok((I::S_FF1_I32_B64, 4)),
        18 => Ok((I::S_FLBIT_I32_B32, 4)),
        19 => Ok((I::S_FLBIT_I32_B64, 4)),
        20 => Ok((I::S_FLBIT_I32, 4)),
        21 => Ok((I::S_FLBIT_I32_I64, 4)),
        22 => Ok((I::S_SEXT_I32_I8, 4)),
        23 => Ok((I::S_SEXT_I32_I16, 4)),
        24 => Ok((I::S_BITSET0_B32, 4)),
        25 => Ok((I::S_BITSET0_B64, 4)),
        26 => Ok((I::S_BITSET1_B32, 4)),
        27 => Ok((I::S_BITSET1_B64, 4)),
        28 => Ok((I::S_GETPC_B64, 4)),
        29 => Ok((I::S_SETPC_B64, 4)),
        30 => Ok((I::S_SWAPPC_B64, 4)),
        31 => Ok((I::S_RFE_B64, 4)),
        32 => Ok((I::S_AND_SAVEEXEC_B64, 4)),
        33 => Ok((I::S_OR_SAVEEXEC_B64, 4)),
        34 => Ok((I::S_XOR_SAVEEXEC_B64, 4)),
        35 => Ok((I::S_ANDN2_SAVEEXEC_B64, 4)),
        36 => Ok((I::S_ORN2_SAVEEXEC_B64, 4)),
        37 => Ok((I::S_NAND_SAVEEXEC_B64, 4)),
        38 => Ok((I::S_NOR_SAVEEXEC_B64, 4)),
        39 => Ok((I::S_XNOR_SAVEEXEC_B64, 4)),
        40 => Ok((I::S_QUADMASK_B32, 4)),
        41 => Ok((I::S_QUADMASK_B64, 4)),
        42 => Ok((I::S_MOVRELS_B32, 4)),
        43 => Ok((I::S_MOVRELS_B64, 4)),
        44 => Ok((I::S_MOVRELD_B32, 4)),
        45 => Ok((I::S_MOVRELD_B64, 4)),
        46 => Ok((I::S_CBRANCH_JOIN, 4)),
        48 => Ok((I::S_ABS_I32, 4)),
        49 => Ok((I::S_SET_GPR_IDX_IDX, 4)),
        _ => Err(()),
    }
}

fn decode_sop2_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        0 => Ok((I::S_ADD_U32, 4)),
        1 => Ok((I::S_SUB_U32, 4)),
        2 => Ok((I::S_ADD_I32, 4)),
        3 => Ok((I::S_SUB_I32, 4)),
        4 => Ok((I::S_ADDC_U32, 4)),
        5 => Ok((I::S_SUBB_U32, 4)),
        6 => Ok((I::S_MIN_I32, 4)),
        7 => Ok((I::S_MIN_U32, 4)),
        8 => Ok((I::S_MAX_I32, 4)),
        9 => Ok((I::S_MAX_U32, 4)),
        10 => Ok((I::S_CSELECT_B32, 4)),
        11 => Ok((I::S_CSELECT_B64, 4)),
        12 => Ok((I::S_AND_B32, 4)),
        13 => Ok((I::S_AND_B64, 4)),
        14 => Ok((I::S_OR_B32, 4)),
        15 => Ok((I::S_OR_B64, 4)),
        16 => Ok((I::S_XOR_B32, 4)),
        17 => Ok((I::S_XOR_B64, 4)),
        18 => Ok((I::S_ANDN2_B32, 4)),
        19 => Ok((I::S_ANDN2_B64, 4)),
        20 => Ok((I::S_ORN2_B32, 4)),
        21 => Ok((I::S_ORN2_B64, 4)),
        22 => Ok((I::S_NAND_B32, 4)),
        23 => Ok((I::S_NAND_B64, 4)),
        24 => Ok((I::S_NOR_B32, 4)),
        25 => Ok((I::S_NOR_B64, 4)),
        26 => Ok((I::S_XNOR_B32, 4)),
        27 => Ok((I::S_XNOR_B64, 4)),
        28 => Ok((I::S_LSHL_B32, 4)),
        29 => Ok((I::S_LSHL_B64, 4)),
        30 => Ok((I::S_LSHR_B32, 4)),
        31 => Ok((I::S_LSHR_B64, 4)),
        32 => Ok((I::S_ASHR_I32, 4)),
        33 => Ok((I::S_ASHR_I64, 4)),
        34 => Ok((I::S_BFM_B32, 4)),
        35 => Ok((I::S_BFM_B64, 4)),
        36 => Ok((I::S_MUL_I32, 4)),
        37 => Ok((I::S_BFE_U32, 4)),
        38 => Ok((I::S_BFE_I32, 4)),
        39 => Ok((I::S_BFE_U64, 4)),
        40 => Ok((I::S_BFE_I64, 4)),
        41 => Ok((I::S_CBRANCH_G_FORK, 4)),
        42 => Ok((I::S_ABSDIFF_I32, 4)),
        43 => Ok((I::S_RFE_RESTORE_B64, 4)),
        _ => Err(()),
    }
}

fn decode_sopk_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        0 => Ok((I::S_MOVK_I32, 4)),
        1 => Ok((I::S_CMOVK_I32, 4)),
        2 => Ok((I::S_CMPK_EQ_I32, 4)),
        3 => Ok((I::S_CMPK_LG_I32, 4)),
        4 => Ok((I::S_CMPK_GT_I32, 4)),
        5 => Ok((I::S_CMPK_GE_I32, 4)),
        6 => Ok((I::S_CMPK_LT_I32, 4)),
        7 => Ok((I::S_CMPK_LE_I32, 4)),
        8 => Ok((I::S_CMPK_EQ_U32, 4)),
        9 => Ok((I::S_CMPK_LG_U32, 4)),
        10 => Ok((I::S_CMPK_GT_U32, 4)),
        11 => Ok((I::S_CMPK_GE_U32, 4)),
        12 => Ok((I::S_CMPK_LT_U32, 4)),
        13 => Ok((I::S_CMPK_LE_U32, 4)),
        14 => Ok((I::S_ADDK_I32, 4)),
        15 => Ok((I::S_MULK_I32, 4)),
        16 => Ok((I::S_CBRANCH_I_FORK, 4)),
        17 => Ok((I::S_GETREG_B32, 4)),
        18 => Ok((I::S_SETREG_B32, 4)),
        20 => Ok((I::S_SETREG_IMM32_B32, 8)),
        _ => Err(()),
    }
}

fn decode_sopc_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        0 => Ok((I::S_CMP_EQ_I32, 4)),
        1 => Ok((I::S_CMP_LG_I32, 4)),
        2 => Ok((I::S_CMP_GT_I32, 4)),
        3 => Ok((I::S_CMP_GE_I32, 4)),
        4 => Ok((I::S_CMP_LT_I32, 4)),
        5 => Ok((I::S_CMP_LE_I32, 4)),
        6 => Ok((I::S_CMP_EQ_U32, 4)),
        7 => Ok((I::S_CMP_LG_U32, 4)),
        8 => Ok((I::S_CMP_GT_U32, 4)),
        9 => Ok((I::S_CMP_GE_U32, 4)),
        10 => Ok((I::S_CMP_LT_U32, 4)),
        11 => Ok((I::S_CMP_LE_U32, 4)),
        12 => Ok((I::S_BITCMP0_B32, 4)),
        13 => Ok((I::S_BITCMP1_B32, 4)),
        14 => Ok((I::S_BITCMP0_B64, 4)),
        15 => Ok((I::S_BITCMP1_B64, 4)),
        16 => Ok((I::S_SETVSKIP, 4)),
        17 => Ok((I::S_SET_GPR_IDX_ON, 4)),
        18 => Ok((I::S_CMP_EQ_U64, 4)),
        19 => Ok((I::S_CMP_NE_U64, 4)),
        _ => Err(()),
    }
}

fn decode_sopp_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        0 => Ok((I::S_NOP, 4)),
        1 => Ok((I::S_ENDPGM, 4)),
        2 => Ok((I::S_BRANCH, 4)),
        4 => Ok((I::S_CBRANCH_SCC0, 4)),
        5 => Ok((I::S_CBRANCH_SCC1, 4)),
        6 => Ok((I::S_CBRANCH_VCCZ, 4)),
        7 => Ok((I::S_CBRANCH_VCCNZ, 4)),
        8 => Ok((I::S_CBRANCH_EXECZ, 4)),
        9 => Ok((I::S_CBRANCH_EXECNZ, 4)),
        10 => Ok((I::S_BARRIER, 4)),
        11 => Ok((I::S_SETKILL, 4)),
        12 => Ok((I::S_WAITCNT, 4)),
        13 => Ok((I::S_SETHALT, 4)),
        14 => Ok((I::S_SLEEP, 4)),
        15 => Ok((I::S_SETPRIO, 4)),
        16 => Ok((I::S_SENDMSG, 4)),
        17 => Ok((I::S_SENDMSGHALT, 4)),
        18 => Ok((I::S_TRAP, 4)),
        19 => Ok((I::S_ICACHE_INV, 4)),
        20 => Ok((I::S_INCPERFLEVEL, 4)),
        21 => Ok((I::S_DECPERFLEVEL, 4)),
        22 => Ok((I::S_TTRACEDATA, 4)),
        23 => Ok((I::S_CBRANCH_CDBGSYS, 4)),
        24 => Ok((I::S_CBRANCH_CDBGUSER, 4)),
        25 => Ok((I::S_CBRANCH_CDBGSYS_OR_USER, 4)),
        26 => Ok((I::S_CBRANCH_CDBGSYS_AND_USER, 4)),
        27 => Ok((I::S_ENDPGM_SAVED, 4)),
        28 => Ok((I::S_SET_GPR_IDX_OFF, 4)),
        29 => Ok((I::S_SET_GPR_IDX_MODE, 4)),
        _ => Err(()),
    }
}

fn decode_vop1_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        0 => Ok((I::V_NOP, 4)),
        1 => Ok((I::V_MOV_B32, 4)),
        2 => Ok((I::V_READFIRSTLANE_B32, 4)),
        3 => Ok((I::V_CVT_I32_F64, 4)),
        4 => Ok((I::V_CVT_F64_I32, 4)),
        5 => Ok((I::V_CVT_F32_I32, 4)),
        6 => Ok((I::V_CVT_F32_U32, 4)),
        7 => Ok((I::V_CVT_U32_F32, 4)),
        8 => Ok((I::V_CVT_I32_F32, 4)),
        10 => Ok((I::V_CVT_F16_F32, 4)),
        11 => Ok((I::V_CVT_F32_F16, 4)),
        12 => Ok((I::V_CVT_RPI_I32_F32, 4)),
        13 => Ok((I::V_CVT_FLR_I32_F32, 4)),
        14 => Ok((I::V_CVT_OFF_F32_I4, 4)),
        15 => Ok((I::V_CVT_F32_F64, 4)),
        16 => Ok((I::V_CVT_F64_F32, 4)),
        17 => Ok((I::V_CVT_F32_UBYTE0, 4)),
        18 => Ok((I::V_CVT_F32_UBYTE1, 4)),
        19 => Ok((I::V_CVT_F32_UBYTE2, 4)),
        20 => Ok((I::V_CVT_F32_UBYTE3, 4)),
        21 => Ok((I::V_CVT_U32_F64, 4)),
        22 => Ok((I::V_CVT_F64_U32, 4)),
        23 => Ok((I::V_TRUNC_F64, 4)),
        24 => Ok((I::V_CEIL_F64, 4)),
        25 => Ok((I::V_RNDNE_F64, 4)),
        26 => Ok((I::V_FLOOR_F64, 4)),
        27 => Ok((I::V_FRACT_F32, 4)),
        28 => Ok((I::V_TRUNC_F32, 4)),
        29 => Ok((I::V_CEIL_F32, 4)),
        30 => Ok((I::V_RNDNE_F32, 4)),
        31 => Ok((I::V_FLOOR_F32, 4)),
        32 => Ok((I::V_EXP_F32, 4)),
        33 => Ok((I::V_LOG_F32, 4)),
        34 => Ok((I::V_RCP_F32, 4)),
        35 => Ok((I::V_RCP_IFLAG_F32, 4)),
        36 => Ok((I::V_RSQ_F32, 4)),
        37 => Ok((I::V_RCP_F64, 4)),
        38 => Ok((I::V_RSQ_F64, 4)),
        39 => Ok((I::V_SQRT_F32, 4)),
        40 => Ok((I::V_SQRT_F64, 4)),
        41 => Ok((I::V_SIN_F32, 4)),
        42 => Ok((I::V_COS_F32, 4)),
        43 => Ok((I::V_NOT_B32, 4)),
        44 => Ok((I::V_BFREV_B32, 4)),
        45 => Ok((I::V_FFBH_U32, 4)),
        46 => Ok((I::V_FFBL_B32, 4)),
        47 => Ok((I::V_FFBH_I32, 4)),
        48 => Ok((I::V_FREXP_EXP_I32_F64, 4)),
        49 => Ok((I::V_FREXP_MANT_F64, 4)),
        50 => Ok((I::V_FRACT_F64, 4)),
        51 => Ok((I::V_FREXP_EXP_I32_F32, 4)),
        52 => Ok((I::V_FREXP_MANT_F32, 4)),
        53 => Ok((I::V_CLREXCP, 4)),
        54 => Ok((I::V_MOVRELD_B32, 4)),
        55 => Ok((I::V_MOVRELS_B32, 4)),
        56 => Ok((I::V_MOVRELSD_B32, 4)),
        57 => Ok((I::V_CVT_F16_U16, 4)),
        58 => Ok((I::V_CVT_F16_I16, 4)),
        59 => Ok((I::V_CVT_U16_F16, 4)),
        60 => Ok((I::V_CVT_I16_F16, 4)),
        61 => Ok((I::V_RCP_F16, 4)),
        62 => Ok((I::V_SQRT_F16, 4)),
        63 => Ok((I::V_RSQ_F16, 4)),
        64 => Ok((I::V_LOG_F16, 4)),
        65 => Ok((I::V_EXP_F16, 4)),
        66 => Ok((I::V_FREXP_MANT_F16, 4)),
        67 => Ok((I::V_FREXP_EXP_I16_F16, 4)),
        68 => Ok((I::V_FLOOR_F16, 4)),
        69 => Ok((I::V_CEIL_F16, 4)),
        70 => Ok((I::V_TRUNC_F16, 4)),
        71 => Ok((I::V_RNDNE_F16, 4)),
        72 => Ok((I::V_FRACT_F16, 4)),
        73 => Ok((I::V_SIN_F16, 4)),
        74 => Ok((I::V_COS_F16, 4)),
        75 => Ok((I::V_EXP_LEGACY_F32, 4)),
        76 => Ok((I::V_LOG_LEGACY_F32, 4)),
        _ => Err(()),
    }
}

fn decode_vop2_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        0 => Ok((I::V_CNDMASK_B32, 4)),
        1 => Ok((I::V_ADD_F32, 4)),
        2 => Ok((I::V_SUB_F32, 4)),
        3 => Ok((I::V_SUBREV_F32, 4)),
        4 => Ok((I::V_MUL_LEGACY_F32, 4)),
        5 => Ok((I::V_MUL_F32, 4)),
        6 => Ok((I::V_MUL_I32_I24, 4)),
        7 => Ok((I::V_MUL_HI_I32_I24, 4)),
        8 => Ok((I::V_MUL_U32_U24, 4)),
        9 => Ok((I::V_MUL_HI_U32_U24, 4)),
        10 => Ok((I::V_MIN_F32, 4)),
        11 => Ok((I::V_MAX_F32, 4)),
        12 => Ok((I::V_MIN_I32, 4)),
        13 => Ok((I::V_MAX_I32, 4)),
        14 => Ok((I::V_MIN_U32, 4)),
        15 => Ok((I::V_MAX_U32, 4)),
        16 => Ok((I::V_LSHRREV_B32, 4)),
        17 => Ok((I::V_ASHRREV_I32, 4)),
        18 => Ok((I::V_LSHLREV_B32, 4)),
        19 => Ok((I::V_AND_B32, 4)),
        20 => Ok((I::V_OR_B32, 4)),
        21 => Ok((I::V_XOR_B32, 4)),
        22 => Ok((I::V_MAC_F32, 4)),
        23 => Ok((I::V_MADMK_F32, 4)),
        24 => Ok((I::V_MADAK_F32, 8)),
        25 => Ok((I::V_ADD_U32, 4)),
        26 => Ok((I::V_SUB_U32, 4)),
        27 => Ok((I::V_SUBREV_U32, 4)),
        28 => Ok((I::V_ADDC_U32, 4)),
        29 => Ok((I::V_SUBB_U32, 4)),
        30 => Ok((I::V_SUBBREV_U32, 4)),
        31 => Ok((I::V_ADD_F16, 4)),
        32 => Ok((I::V_SUB_F16, 4)),
        33 => Ok((I::V_SUBREV_F16, 4)),
        34 => Ok((I::V_MUL_F16, 4)),
        35 => Ok((I::V_MAC_F16, 4)),
        36 => Ok((I::V_MADMK_F16, 4)),
        37 => Ok((I::V_MADAK_F16, 4)),
        38 => Ok((I::V_ADD_U16, 4)),
        39 => Ok((I::V_SUB_U16, 4)),
        40 => Ok((I::V_SUBREV_U16, 4)),
        41 => Ok((I::V_MUL_LO_U16, 4)),
        42 => Ok((I::V_LSHLREV_B16, 4)),
        43 => Ok((I::V_LSHRREV_B16, 4)),
        44 => Ok((I::V_ASHRREV_I16, 4)),
        45 => Ok((I::V_MAX_F16, 4)),
        46 => Ok((I::V_MIN_F16, 4)),
        47 => Ok((I::V_MAX_U16, 4)),
        48 => Ok((I::V_MAX_I16, 4)),
        49 => Ok((I::V_MIN_U16, 4)),
        50 => Ok((I::V_MIN_I16, 4)),
        51 => Ok((I::V_LDEXP_F16, 4)),
        _ => Err(()),
    }
}

fn decode_op16(opcode: u32) -> Option<OP16> {
    match opcode {
        0 => Some(OP16::F),
        1 => Some(OP16::LT),
        2 => Some(OP16::EQ),
        3 => Some(OP16::LE),
        4 => Some(OP16::GT),
        5 => Some(OP16::LG),
        6 => Some(OP16::GE),
        7 => Some(OP16::O),
        8 => Some(OP16::U),
        9 => Some(OP16::NGE),
        10 => Some(OP16::NLG),
        11 => Some(OP16::NGT),
        12 => Some(OP16::NLE),
        13 => Some(OP16::NEQ),
        14 => Some(OP16::NLT),
        15 => Some(OP16::TRU),
        _ => None,
    }
}

fn decode_op8(opcode: u32) -> Option<OP8> {
    match opcode {
        0 => Some(OP8::F),
        1 => Some(OP8::LT),
        2 => Some(OP8::EQ),
        3 => Some(OP8::LE),
        4 => Some(OP8::GT),
        5 => Some(OP8::LG),
        6 => Some(OP8::GE),
        7 => Some(OP8::TRU),
        _ => None,
    }
}

fn decode_vopc_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        0x10 => Ok((I::V_CMP_CLASS_F32, 4)),
        0x11 => Ok((I::V_CMPX_CLASS_F32, 4)),
        0x12 => Ok((I::V_CMP_CLASS_F64, 4)),
        0x13 => Ok((I::V_CMPX_CLASS_F64, 4)),
        0x14 => Ok((I::V_CMP_CLASS_F16, 4)),
        0x15 => Ok((I::V_CMPX_CLASS_F16, 4)),
        0x20..=0x2F => Ok((I::V_CMP_F16(decode_op16(opcode - 0x20).unwrap()), 4)),
        0x30..=0x3F => Ok((I::V_CMPX_F16(decode_op16(opcode - 0x30).unwrap()), 4)),
        0x40..=0x4F => Ok((I::V_CMP_F32(decode_op16(opcode - 0x40).unwrap()), 4)),
        0x50..=0x5F => Ok((I::V_CMPX_F32(decode_op16(opcode - 0x50).unwrap()), 4)),
        0x60..=0x6F => Ok((I::V_CMP_F64(decode_op16(opcode - 0x60).unwrap()), 4)),
        0x70..=0x7F => Ok((I::V_CMPX_F64(decode_op16(opcode - 0x70).unwrap()), 4)),
        0xA0..=0xA7 => Ok((I::V_CMP_I16(decode_op8(opcode - 0xA0).unwrap()), 4)),
        0xA8..=0xAF => Ok((I::V_CMP_U16(decode_op8(opcode - 0xA8).unwrap()), 4)),
        0xB0..=0xB7 => Ok((I::V_CMPX_I16(decode_op8(opcode - 0xB0).unwrap()), 4)),
        0xB8..=0xBF => Ok((I::V_CMPX_U16(decode_op8(opcode - 0xB8).unwrap()), 4)),
        0xC0..=0xC7 => Ok((I::V_CMP_I32(decode_op8(opcode - 0xC0).unwrap()), 4)),
        0xC8..=0xCF => Ok((I::V_CMP_U32(decode_op8(opcode - 0xC8).unwrap()), 4)),
        0xD0..=0xD7 => Ok((I::V_CMPX_I32(decode_op8(opcode - 0xD0).unwrap()), 4)),
        0xD8..=0xDF => Ok((I::V_CMPX_U32(decode_op8(opcode - 0xD8).unwrap()), 4)),
        0xE0..=0xE7 => Ok((I::V_CMP_I64(decode_op8(opcode - 0xE0).unwrap()), 4)),
        0xE8..=0xEF => Ok((I::V_CMP_U64(decode_op8(opcode - 0xE8).unwrap()), 4)),
        0xF0..=0xF7 => Ok((I::V_CMPX_I64(decode_op8(opcode - 0xF0).unwrap()), 4)),
        0xF8..=0xFF => Ok((I::V_CMPX_U64(decode_op8(opcode - 0xF8).unwrap()), 4)),
        _ => Err(()),
    }
}

fn decode_vop3a_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        0..=255 => {
            let (op, _) = decode_vopc_opcode_gcn3(opcode)?;
            Ok((op, 8))
        }
        256..=319 => {
            let (op, _) = decode_vop2_opcode_gcn3(opcode - 256)?;
            Ok((op, 8))
        }
        320..=447 => {
            let (op, _) = decode_vop1_opcode_gcn3(opcode - 320)?;
            Ok((op, 8))
        }
        448 => Ok((I::V_MAD_LEGACY_F32, 8)),
        449 => Ok((I::V_MAD_F32, 8)),
        450 => Ok((I::V_MAD_I32_I24, 8)),
        451 => Ok((I::V_MAD_U32_U24, 8)),
        452 => Ok((I::V_CUBEID_F32, 8)),
        453 => Ok((I::V_CUBESC_F32, 8)),
        454 => Ok((I::V_CUBETC_F32, 8)),
        455 => Ok((I::V_CUBEMA_F32, 8)),
        456 => Ok((I::V_BFE_U32, 8)),
        457 => Ok((I::V_BFE_I32, 8)),
        458 => Ok((I::V_BFI_B32, 8)),
        459 => Ok((I::V_FMA_F32, 8)),
        460 => Ok((I::V_FMA_F64, 8)),
        461 => Ok((I::V_LERP_U8, 8)),
        462 => Ok((I::V_ALIGNBIT_B32, 8)),
        463 => Ok((I::V_ALIGNBYTE_B32, 8)),
        464 => Ok((I::V_MIN3_F32, 8)),
        465 => Ok((I::V_MIN3_I32, 8)),
        466 => Ok((I::V_MIN3_U32, 8)),
        467 => Ok((I::V_MAX3_F32, 8)),
        468 => Ok((I::V_MAX3_I32, 8)),
        469 => Ok((I::V_MAX3_U32, 8)),
        470 => Ok((I::V_MED3_F32, 8)),
        471 => Ok((I::V_MED3_I32, 8)),
        472 => Ok((I::V_MED3_U32, 8)),
        473 => Ok((I::V_SAD_U8, 8)),
        474 => Ok((I::V_SAD_HI_U8, 8)),
        475 => Ok((I::V_SAD_U16, 8)),
        476 => Ok((I::V_SAD_U32, 8)),
        477 => Ok((I::V_CVT_PK_U8_F32, 8)),
        478 => Ok((I::V_DIV_FIXUP_F32, 8)),
        479 => Ok((I::V_DIV_FIXUP_F64, 8)),
        482 => Ok((I::V_DIV_FMAS_F32, 8)),
        483 => Ok((I::V_DIV_FMAS_F64, 8)),
        484 => Ok((I::V_MSAD_U8, 8)),
        485 => Ok((I::V_QSAD_PK_U16_U8, 8)),
        486 => Ok((I::V_MQSAD_PK_U16_U8, 8)),
        487 => Ok((I::V_MQSAD_U32_U8, 8)),
        488 => Ok((I::V_MAD_U64_U32, 8)),
        489 => Ok((I::V_MAD_I64_I32, 8)),
        490 => Ok((I::V_MAD_F16, 8)),
        491 => Ok((I::V_MAD_U16, 8)),
        492 => Ok((I::V_MAD_I16, 8)),
        493 => Ok((I::V_PERM_B32, 8)),
        494 => Ok((I::V_FMA_F16, 8)),
        495 => Ok((I::V_DIV_FIXUP_F16, 8)),
        496 => Ok((I::V_CVT_PKACCUM_U8_F32, 8)),
        624 => Ok((I::V_INTERP_P1_F32, 8)),
        625 => Ok((I::V_INTERP_P2_F32, 8)),
        626 => Ok((I::V_INTERP_MOV_F32, 8)),
        628 => Ok((I::V_INTERP_P1LL_F16, 8)),
        629 => Ok((I::V_INTERP_P1LV_F16, 8)),
        630 => Ok((I::V_INTERP_P2_F16, 8)),
        640 => Ok((I::V_ADD_F64, 8)),
        641 => Ok((I::V_MUL_F64, 8)),
        642 => Ok((I::V_MIN_F64, 8)),
        643 => Ok((I::V_MAX_F64, 8)),
        644 => Ok((I::V_LDEXP_F64, 8)),
        645 => Ok((I::V_MUL_LO_U32, 8)),
        646 => Ok((I::V_MUL_HI_U32, 8)),
        647 => Ok((I::V_MUL_HI_I32, 8)),
        648 => Ok((I::V_LDEXP_F32, 8)),
        649 => Ok((I::V_READLANE_B32, 8)),
        650 => Ok((I::V_WRITELANE_B32, 8)),
        651 => Ok((I::V_BCNT_U32_B32, 8)),
        652 => Ok((I::V_MBCNT_LO_U32_B32, 8)),
        653 => Ok((I::V_MBCNT_HI_U32_B32, 8)),
        655 => Ok((I::V_LSHLREV_B64, 8)),
        656 => Ok((I::V_LSHRREV_B64, 8)),
        657 => Ok((I::V_ASHRREV_I64, 8)),
        658 => Ok((I::V_TRIG_PREOP_F64, 8)),
        659 => Ok((I::V_BFM_B32, 8)),
        660 => Ok((I::V_CVT_PKNORM_I16_F32, 8)),
        661 => Ok((I::V_CVT_PKNORM_U16_F32, 8)),
        662 => Ok((I::V_CVT_PKRTZ_F16_F32, 8)),
        663 => Ok((I::V_CVT_PK_U16_U32, 8)),
        664 => Ok((I::V_CVT_PK_I16_I32, 8)),
        _ => Err(()),
    }
}

fn decode_vop3b_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        0x10 => Ok((I::V_CMP_CLASS_F32, 4)),
        0x11 => Ok((I::V_CMPX_CLASS_F32, 4)),
        0x12 => Ok((I::V_CMP_CLASS_F64, 4)),
        0x13 => Ok((I::V_CMPX_CLASS_F64, 4)),
        0x14 => Ok((I::V_CMP_CLASS_F16, 4)),
        0x15 => Ok((I::V_CMPX_CLASS_F16, 4)),
        0x20..=0x2F => Ok((I::V_CMP_F16(decode_op16(opcode - 0x20).unwrap()), 4)),
        0x30..=0x3F => Ok((I::V_CMPX_F16(decode_op16(opcode - 0x30).unwrap()), 4)),
        0x40..=0x4F => Ok((I::V_CMP_F32(decode_op16(opcode - 0x40).unwrap()), 4)),
        0x50..=0x5F => Ok((I::V_CMPX_F32(decode_op16(opcode - 0x50).unwrap()), 4)),
        0x60..=0x6F => Ok((I::V_CMP_F64(decode_op16(opcode - 0x60).unwrap()), 4)),
        0x70..=0x7F => Ok((I::V_CMPX_F64(decode_op16(opcode - 0x70).unwrap()), 4)),
        0xA0..=0xA7 => Ok((I::V_CMP_I16(decode_op8(opcode - 0xA0).unwrap()), 4)),
        0xA8..=0xAF => Ok((I::V_CMP_U16(decode_op8(opcode - 0xA8).unwrap()), 4)),
        0xB0..=0xB7 => Ok((I::V_CMPX_I16(decode_op8(opcode - 0xB0).unwrap()), 4)),
        0xB8..=0xBF => Ok((I::V_CMPX_U16(decode_op8(opcode - 0xB8).unwrap()), 4)),
        0xC0..=0xC7 => Ok((I::V_CMP_I32(decode_op8(opcode - 0xC0).unwrap()), 4)),
        0xC8..=0xCF => Ok((I::V_CMP_U32(decode_op8(opcode - 0xC8).unwrap()), 4)),
        0xD0..=0xD7 => Ok((I::V_CMPX_I32(decode_op8(opcode - 0xD0).unwrap()), 4)),
        0xD8..=0xDF => Ok((I::V_CMPX_U32(decode_op8(opcode - 0xD8).unwrap()), 4)),
        0xE0..=0xE7 => Ok((I::V_CMP_I64(decode_op8(opcode - 0xE0).unwrap()), 4)),
        0xE8..=0xEF => Ok((I::V_CMP_U64(decode_op8(opcode - 0xE8).unwrap()), 4)),
        0xF0..=0xF7 => Ok((I::V_CMPX_I64(decode_op8(opcode - 0xF0).unwrap()), 4)),
        0xF8..=0xFF => Ok((I::V_CMPX_U64(decode_op8(opcode - 0xF8).unwrap()), 4)),
        281 => Ok((I::V_ADD_U32, 8)),
        282 => Ok((I::V_SUB_U32, 8)),
        283 => Ok((I::V_SUBREV_U32, 8)),
        284 => Ok((I::V_ADDC_U32, 8)),
        285 => Ok((I::V_SUBB_U32, 8)),
        286 => Ok((I::V_SUBBREV_U32, 8)),
        480 => Ok((I::V_DIV_SCALE_F32, 8)),
        481 => Ok((I::V_DIV_SCALE_F64, 8)),
        _ => Err(()),
    }
}

fn decode_smem_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        0 => Ok((I::S_LOAD_DWORD, 8)),
        1 => Ok((I::S_LOAD_DWORDX2, 8)),
        2 => Ok((I::S_LOAD_DWORDX4, 8)),
        3 => Ok((I::S_LOAD_DWORDX8, 8)),
        4 => Ok((I::S_LOAD_DWORDX16, 8)),
        8 => Ok((I::S_BUFFER_LOAD_DWORD, 8)),
        9 => Ok((I::S_BUFFER_LOAD_DWORDX2, 8)),
        10 => Ok((I::S_BUFFER_LOAD_DWORDX4, 8)),
        11 => Ok((I::S_BUFFER_LOAD_DWORDX8, 8)),
        12 => Ok((I::S_BUFFER_LOAD_DWORDX16, 8)),
        16 => Ok((I::S_STORE_DWORD, 8)),
        17 => Ok((I::S_STORE_DWORDX2, 8)),
        18 => Ok((I::S_STORE_DWORDX4, 8)),
        24 => Ok((I::S_BUFFER_STORE_DWORD, 8)),
        25 => Ok((I::S_BUFFER_STORE_DWORDX2, 8)),
        26 => Ok((I::S_BUFFER_STORE_DWORDX4, 8)),
        32 => Ok((I::S_DCACHE_INV, 8)),
        33 => Ok((I::S_DCACHE_WB, 8)),
        34 => Ok((I::S_DCACHE_INV_VOL, 8)),
        35 => Ok((I::S_DCACHE_WB_VOL, 8)),
        36 => Ok((I::S_MEMTIME, 8)),
        37 => Ok((I::S_MEMREALTIME, 8)),
        38 => Ok((I::S_ATC_PROBE, 8)),
        39 => Ok((I::S_ATC_PROBE_BUFFER, 8)),
        _ => Err(()),
    }
}

fn decode_flat_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        16 => Ok((I::FLAT_LOAD_UBYTE, 8)),
        17 => Ok((I::FLAT_LOAD_SBYTE, 8)),
        18 => Ok((I::FLAT_LOAD_USHORT, 8)),
        19 => Ok((I::FLAT_LOAD_SSHORT, 8)),
        20 => Ok((I::FLAT_LOAD_DWORD, 8)),
        21 => Ok((I::FLAT_LOAD_DWORDX2, 8)),
        22 => Ok((I::FLAT_LOAD_DWORDX3, 8)),
        23 => Ok((I::FLAT_LOAD_DWORDX4, 8)),
        24 => Ok((I::FLAT_STORE_BYTE, 8)),
        26 => Ok((I::FLAT_STORE_SHORT, 8)),
        28 => Ok((I::FLAT_STORE_DWORD, 8)),
        29 => Ok((I::FLAT_STORE_DWORDX2, 8)),
        30 => Ok((I::FLAT_STORE_DWORDX3, 8)),
        31 => Ok((I::FLAT_STORE_DWORDX4, 8)),
        64 => Ok((I::FLAT_ATOMIC_SWAP, 8)),
        65 => Ok((I::FLAT_ATOMIC_CMPSWAP, 8)),
        66 => Ok((I::FLAT_ATOMIC_ADD, 8)),
        67 => Ok((I::FLAT_ATOMIC_SUB, 8)),
        53 => Ok((I::FLAT_ATOMIC_SMIN, 8)),
        69 => Ok((I::FLAT_ATOMIC_UMIN, 8)),
        70 => Ok((I::FLAT_ATOMIC_SMAX, 8)),
        71 => Ok((I::FLAT_ATOMIC_UMAX, 8)),
        72 => Ok((I::FLAT_ATOMIC_AND, 8)),
        73 => Ok((I::FLAT_ATOMIC_OR, 8)),
        74 => Ok((I::FLAT_ATOMIC_XOR, 8)),
        75 => Ok((I::FLAT_ATOMIC_INC, 8)),
        76 => Ok((I::FLAT_ATOMIC_DEC, 8)),
        96 => Ok((I::FLAT_ATOMIC_SWAP_X2, 8)),
        97 => Ok((I::FLAT_ATOMIC_CMPSWAP_X2, 8)),
        98 => Ok((I::FLAT_ATOMIC_ADD_X2, 8)),
        99 => Ok((I::FLAT_ATOMIC_SUB_X2, 8)),
        100 => Ok((I::FLAT_ATOMIC_SMIN_X2, 8)),
        101 => Ok((I::FLAT_ATOMIC_UMIN_X2, 8)),
        102 => Ok((I::FLAT_ATOMIC_SMAX_X2, 8)),
        103 => Ok((I::FLAT_ATOMIC_UMAX_X2, 8)),
        104 => Ok((I::FLAT_ATOMIC_AND_X2, 8)),
        105 => Ok((I::FLAT_ATOMIC_OR_X2, 8)),
        106 => Ok((I::FLAT_ATOMIC_XOR_X2, 8)),
        107 => Ok((I::FLAT_ATOMIC_INC_X2, 8)),
        108 => Ok((I::FLAT_ATOMIC_DEC_X2, 8)),
        _ => Err(()),
    }
}

fn decode_mubuf_opcode_gcn3(opcode: u32) -> Result<(I, usize), ()> {
    match opcode {
        0 => Ok((I::BUFFER_LOAD_FORMAT_X, 8)),
        1 => Ok((I::BUFFER_LOAD_FORMAT_XY, 8)),
        2 => Ok((I::BUFFER_LOAD_FORMAT_XYZ, 8)),
        3 => Ok((I::BUFFER_LOAD_FORMAT_XYZW, 8)),
        4 => Ok((I::BUFFER_STORE_FORMAT_X, 8)),
        5 => Ok((I::BUFFER_STORE_FORMAT_XY, 8)),
        6 => Ok((I::BUFFER_STORE_FORMAT_XYZ, 8)),
        7 => Ok((I::BUFFER_STORE_FORMAT_XYZW, 8)),
        8 => Ok((I::BUFFER_LOAD_FORMAT_D16_X, 8)),
        9 => Ok((I::BUFFER_LOAD_FORMAT_D16_XY, 8)),
        10 => Ok((I::BUFFER_LOAD_FORMAT_D16_XYZ, 8)),
        11 => Ok((I::BUFFER_LOAD_FORMAT_D16_XYZW, 8)),
        12 => Ok((I::BUFFER_STORE_FORMAT_D16_X, 8)),
        13 => Ok((I::BUFFER_STORE_FORMAT_D16_XY, 8)),
        14 => Ok((I::BUFFER_STORE_FORMAT_D16_XYZ, 8)),
        15 => Ok((I::BUFFER_STORE_FORMAT_D16_XYZW, 8)),
        16 => Ok((I::BUFFER_LOAD_UBYTE, 8)),
        17 => Ok((I::BUFFER_LOAD_SBYTE, 8)),
        18 => Ok((I::BUFFER_LOAD_USHORT, 8)),
        19 => Ok((I::BUFFER_LOAD_SSHORT, 8)),
        20 => Ok((I::BUFFER_LOAD_DWORD, 8)),
        21 => Ok((I::BUFFER_LOAD_DWORDX2, 8)),
        22 => Ok((I::BUFFER_LOAD_DWORDX3, 8)),
        23 => Ok((I::BUFFER_LOAD_DWORDX4, 8)),
        24 => Ok((I::BUFFER_STORE_BYTE, 8)),
        26 => Ok((I::BUFFER_STORE_SHORT, 8)),
        28 => Ok((I::BUFFER_STORE_DWORD, 8)),
        29 => Ok((I::BUFFER_STORE_DWORDX2, 8)),
        30 => Ok((I::BUFFER_STORE_DWORDX3, 8)),
        31 => Ok((I::BUFFER_STORE_DWORDX4, 8)),
        61 => Ok((I::BUFFER_STORE_LDS_DWORD, 8)),
        62 => Ok((I::BUFFER_WBINVL1, 8)),
        63 => Ok((I::BUFFER_WBINVL1_VOL, 8)),
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

fn get_bits(value: u64, to: usize, from: usize) -> u64 {
    let num = to + 1 - from;
    (value >> from) & ((1u64 << num) - 1)
}

pub fn decode_gcn3(inst: u64) -> Result<(InstFormat, usize), ()> {
    if (get_bits(inst, 31, 23) as u32) == SOP1_ENCODE {
        let ssrc0 = get_bits(inst, 7, 0) as u8;
        let (op, size) = decode_sop1_opcode_gcn3(get_bits(inst, 15, 8) as u32)?;
        Ok((
            InstFormat::SOP1(SOP1 {
                ssrc0,
                op,
                sdst: get_bits(inst, 22, 16) as u8,
            }),
            if ssrc0 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 23) as u32) == SOPC_ENCODE {
        let (op, size) = decode_sopc_opcode_gcn3(get_bits(inst, 22, 16) as u32)?;
        Ok((
            InstFormat::SOPC(SOPC {
                ssrc0: get_bits(inst, 7, 0) as u8,
                ssrc1: get_bits(inst, 15, 8) as u8,
                op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 23) as u32) == SOPP_ENCODE {
        let (op, size) = decode_sopp_opcode_gcn3(get_bits(inst, 22, 16) as u32)?;
        Ok((
            InstFormat::SOPP(SOPP {
                simm16: get_bits(inst, 15, 0) as u16,
                op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 28) as u32) == SOPK_ENCODE {
        let (op, size) = decode_sopk_opcode_gcn3(get_bits(inst, 27, 23) as u32)?;
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
        let (op, size) = decode_sop2_opcode_gcn3(get_bits(inst, 29, 23) as u32)?;
        Ok((
            InstFormat::SOP2(SOP2 {
                ssrc0: get_bits(inst, 7, 0) as u8,
                ssrc1,
                sdst: get_bits(inst, 22, 16) as u8,
                op,
            }),
            if ssrc1 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 26) as u32) == VOP3AB_ENCODE {
        let op = get_bits(inst, 25, 16) as u32;
        match op {
            (281..=286) | (480..=481) => {
                let (op, size) = decode_vop3b_opcode_gcn3(op)?;
                Ok((
                    InstFormat::VOP3B(VOP3B {
                        vdst: get_bits(inst, 7, 0) as u8,
                        sdst: get_bits(inst, 14, 8) as u8,
                        clamp: get_bits(inst, 15, 15) as u8,
                        op,
                        src0: get_bits(inst, 40, 32) as u16,
                        src1: get_bits(inst, 49, 41) as u16,
                        src2: get_bits(inst, 58, 50) as u16,
                        omod: get_bits(inst, 60, 59) as u8,
                        neg: get_bits(inst, 63, 61) as u8,
                    }),
                    size,
                ))
            }
            _ => {
                let (op, size) = decode_vop3a_opcode_gcn3(op)?;
                Ok((
                    InstFormat::VOP3A(VOP3A {
                        vdst: get_bits(inst, 7, 0) as u8,
                        abs: get_bits(inst, 10, 8) as u8,
                        clamp: get_bits(inst, 15, 15) as u8,
                        op,
                        src0: get_bits(inst, 40, 32) as u16,
                        src1: get_bits(inst, 49, 41) as u16,
                        src2: get_bits(inst, 58, 50) as u16,
                        omod: get_bits(inst, 60, 59) as u8,
                        neg: get_bits(inst, 63, 61) as u8,
                    }),
                    size,
                ))
            }
        }
    } else if (get_bits(inst, 31, 25) as u32) == VOPC_ENCODE {
        let (op, size) = decode_vopc_opcode_gcn3(get_bits(inst, 24, 17) as u32)?;
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
        let (op, size) = decode_vop1_opcode_gcn3(get_bits(inst, 16, 9) as u32)?;
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
        let (op, size) = decode_vop2_opcode_gcn3(get_bits(inst, 30, 25) as u32)?;
        Ok((
            InstFormat::VOP2(VOP2 {
                src0,
                vsrc1: get_bits(inst, 16, 9) as u8,
                vdst: get_bits(inst, 24, 17) as u8,
                op,
            }),
            if src0 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 26) as u32) == SMEM_ENCODE {
        let (op, size) = decode_smem_opcode_gcn3(get_bits(inst, 25, 18) as u32)?;
        Ok((
            InstFormat::SMEM(SMEM {
                sbase: get_bits(inst, 5, 0) as u8,
                sdata: get_bits(inst, 12, 6) as u8,
                glc: get_bits(inst, 16, 16) as u8,
                imm: get_bits(inst, 17, 17) as u8,
                op,
                offset: get_bits(inst, 51, 32) as u8,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 26) as u32) == FLAT_ENCODE {
        let (op, size) = decode_flat_opcode_gcn3(get_bits(inst, 24, 18) as u32)?;
        Ok((
            InstFormat::FLAT(FLAT {
                glc: get_bits(inst, 16, 16) as u8,
                slc: get_bits(inst, 17, 17) as u8,
                op,
                addr: get_bits(inst, 39, 32) as u8,
                data: get_bits(inst, 47, 40) as u8,
                tfe: get_bits(inst, 55, 55) as u8,
                vdst: get_bits(inst, 63, 56) as u8,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 26) as u32) == MUBUF_ENCODE {
        let (op, size) = decode_mubuf_opcode_gcn3(get_bits(inst, 24, 18) as u32)?;
        Ok((
            InstFormat::MUBUF(MUBUF {
                offset: get_bits(inst, 11, 0) as u16,
                offen: get_bits(inst, 12, 12) as u8,
                idxen: get_bits(inst, 13, 13) as u8,
                glc: get_bits(inst, 14, 14) as u8,
                lds: get_bits(inst, 16, 16) as u8,
                slc: get_bits(inst, 17, 17) as u8,
                op,
                vaddr: get_bits(inst, 39, 32) as u8,
                vdata: get_bits(inst, 47, 40) as u8,
                srsrc: get_bits(inst, 52, 48) as u8,
                tfe: get_bits(inst, 55, 55) as u8,
                soffset: get_bits(inst, 63, 56) as u8,
            }),
            size,
        ))
    } else {
        Err(())
    }
}
