use crate::instructions::*;

fn decode_sop1_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
        0 => Ok((I::S_MOV_B32, 4)),
        1 => Ok((I::S_MOV_B64, 4)),
        _ => Err(()),
    }
}

fn decode_sop2_opcode_rdna4(opcode: u32) -> Result<(I, usize), ()> {
    match opcode as u8 {
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

pub fn decode_rdna4(inst: u64) -> Result<(InstFormat, usize), ()> {
    if (get_bits(inst, 31, 23) as u32) == SOP1_ENCODE {
        let ssrc0 = get_bits(inst, 7, 0) as u8;
        let (op, size) = decode_sop1_opcode_rdna4(get_bits(inst, 15, 8) as u32)?;
        Ok((
            InstFormat::SOP1(SOP1 {
                SSRC0: ssrc0,
                OP: op,
                SDST: get_bits(inst, 22, 16) as u8,
            }),
            if ssrc0 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 23) as u32) == SOPC_ENCODE {
        let (op, size) = decode_sopc_opcode_rdna4(get_bits(inst, 22, 16) as u32)?;
        Ok((
            InstFormat::SOPC(SOPC {
                SSRC0: get_bits(inst, 7, 0) as u8,
                SSRC1: get_bits(inst, 15, 8) as u8,
                OP: op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 23) as u32) == SOPP_ENCODE {
        let (op, size) = decode_sopp_opcode_rdna4(get_bits(inst, 22, 16) as u32)?;
        Ok((
            InstFormat::SOPP(SOPP {
                SIMM16: get_bits(inst, 15, 0) as u16,
                OP: op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 28) as u32) == SOPK_ENCODE {
        let (op, size) = decode_sopk_opcode_rdna4(get_bits(inst, 27, 23) as u32)?;
        Ok((
            InstFormat::SOPK(SOPK {
                SIMM16: get_bits(inst, 15, 0) as u16,
                SDST: get_bits(inst, 22, 16) as u8,
                OP: op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 30) as u32) == SOP2_ENCODE {
        let ssrc1 = get_bits(inst, 15, 8) as u8;
        let (op, size) = decode_sop2_opcode_rdna4(get_bits(inst, 29, 23) as u32)?;
        Ok((
            InstFormat::SOP2(SOP2 {
                SSRC0: get_bits(inst, 7, 0) as u8,
                SSRC1: ssrc1,
                SDST: get_bits(inst, 22, 16) as u8,
                OP: op,
            }),
            if ssrc1 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 25) as u32) == VOPC_ENCODE {
        let (op, size) = decode_vopc_opcode_rdna4(get_bits(inst, 24, 17) as u32)?;
        Ok((
            InstFormat::VOPC(VOPC {
                SRC0: get_bits(inst, 8, 0) as u16,
                VSRC1: get_bits(inst, 16, 9) as u8,
                OP: op,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 25) as u32) == VOP1_ENCODE {
        let src0 = get_bits(inst, 8, 0) as u16;
        let (op, size) = decode_vop1_opcode_rdna4(get_bits(inst, 16, 9) as u32)?;
        Ok((
            InstFormat::VOP1(VOP1 {
                SRC0: src0,
                OP: op,
                VDST: get_bits(inst, 24, 17) as u8,
            }),
            if src0 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 31) as u32) == VOP2_ENCODE {
        let src0 = get_bits(inst, 8, 0) as u16;
        let (op, size) = decode_vop2_opcode_rdna4(get_bits(inst, 30, 25) as u32)?;
        Ok((
            InstFormat::VOP2(VOP2 {
                SRC0: src0,
                VSRC1: get_bits(inst, 16, 9) as u8,
                VDST: get_bits(inst, 24, 17) as u8,
                OP: op,
            }),
            if src0 == 255 { max(8, size) } else { size },
        ))
    } else if (get_bits(inst, 31, 26) as u32) == VOP3_ENCODE {
        let src0 = get_bits(inst, 8, 0) as u16;
        let (op, size) = decode_vop3_opcode_rdna4(get_bits(inst, 25, 16) as u32)?;
        Ok((
            InstFormat::VOP3(VOP3 {
                VDST: get_bits(inst, 7, 0) as u8,
                ABS: get_bits(inst, 10, 8) as u8,
                OPSEL: get_bits(inst, 14, 1) as u8,
                CM: get_bits(inst, 15, 15) as u8,
                OP: op,
                SRC0: get_bits(inst, 40, 32) as u16,
                SRC1: get_bits(inst, 49, 41) as u16,
                SRC2: get_bits(inst, 58, 50) as u16,
                OMOD: get_bits(inst, 60, 59) as u8,
                NEG: get_bits(inst, 63, 61) as u8,
            }),
            size,
        ))
    } else if (get_bits(inst, 31, 26) as u32) == SMEM_ENCODE {
        let (op, size) = decode_smem_opcode_rdna4(get_bits(inst, 25, 18) as u32)?;
        Ok((
            InstFormat::SMEM(SMEM {
                SBASE: get_bits(inst, 5, 0) as u8,
                SDATA: get_bits(inst, 12, 6) as u8,
                GLC: get_bits(inst, 16, 16) as u8,
                IMM: get_bits(inst, 17, 17) as u8,
                OP: op,
                OFFSET: get_bits(inst, 51, 32) as u8,
            }),
            size,
        ))
    } else {
        Err(())
    }
}
