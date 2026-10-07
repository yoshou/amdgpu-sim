use crate::bit::get_bits;
use crate::gcn3_decoder as table;
use crate::instructions::I;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sdwa {
    pub dst_sel: u8,
    pub dst_unused: u8,
    pub clamp: bool,
    pub src0_sel: u8,
    pub src0_sext: bool,
    pub src0_neg: bool,
    pub src0_abs: bool,
    pub src1_sel: u8,
    pub src1_sext: bool,
    pub src1_neg: bool,
    pub src1_abs: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dpp {
    pub ctrl: u16,
    pub bound_ctrl: bool,
    pub src0_neg: bool,
    pub src0_abs: bool,
    pub src1_neg: bool,
    pub src1_abs: bool,
    pub bank_mask: u8,
    pub row_mask: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ext {
    Plain,
    Sdwa(Sdwa),
    Dpp(Dpp),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    Sop2 {
        sdst: u8,
        ssrc0: u16,
        ssrc1: u16,
    },
    Sopk {
        sdst: u8,
        simm16: u16,
    },
    Sop1 {
        sdst: u8,
        ssrc0: u16,
    },
    Sopc {
        ssrc0: u16,
        ssrc1: u16,
    },
    Sopp {
        simm16: u16,
    },
    Smem {
        sdata: u8,
        sbase: u8,
        offset: u32,
        imm: bool,
        glc: bool,
    },
    Vop1 {
        vdst: u8,
        src0: u16,
        ext: Ext,
    },
    Vop2 {
        vdst: u8,
        src0: u16,
        vsrc1: u8,
        ext: Ext,
    },
    Vopc {
        src0: u16,
        vsrc1: u8,
        ext: Ext,
    },
    Vop3 {
        vdst: u8,
        sdst: Option<u8>,
        src: [u16; 3],
        abs: u8,
        neg: u8,
        clamp: bool,
        omod: u8,
    },
    Ds {
        offset0: u8,
        offset1: u8,
        gds: bool,
        addr: u8,
        data0: u8,
        data1: u8,
        vdst: u8,
    },
    Mubuf {
        offset: u16,
        offen: bool,
        idxen: bool,
        glc: bool,
        lds: bool,
        slc: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        tfe: bool,
        soffset: u16,
    },
    Mimg {
        dmask: u8,
        unorm: bool,
        glc: bool,
        da: bool,
        r128: bool,
        tfe: bool,
        lwe: bool,
        slc: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        ssamp: u8,
        d16: bool,
    },
    Flat {
        glc: bool,
        slc: bool,
        addr: u8,
        data: u8,
        tfe: bool,
        vdst: u8,
    },
}

#[derive(Clone, Copy, Debug)]
pub struct Inst {
    pub op: I,
    pub size: usize,
    pub literal: Option<u32>,
    pub form: Form,
}

pub const LITERAL: u16 = 255;

pub fn wide(op: I) -> bool {
    let n = format!("{:?}", op);
    n.ends_with("_B64") || n.ends_with("_U64") || n.ends_with("_I64") || n.ends_with("_F64") || n.contains("_F64(") || n.contains("_I64(") || n.contains("_U64(")
}

pub fn vector_words(op: I) -> (u32, [u32; 3]) {
    match op {
        I::V_LDEXP_F64 | I::V_TRIG_PREOP_F64 => (2, [2, 1, 1]),
        I::V_FREXP_EXP_I32_F64 | I::V_CVT_I32_F64 | I::V_CVT_U32_F64 | I::V_CVT_F32_F64 => (1, [2, 1, 1]),
        I::V_CVT_F64_I32 | I::V_CVT_F64_U32 | I::V_CVT_F64_F32 => (2, [1, 1, 1]),
        I::V_CMP_CLASS_F64 | I::V_CMPX_CLASS_F64 => (2, [2, 1, 1]),
        I::V_LSHLREV_B64 | I::V_LSHRREV_B64 | I::V_ASHRREV_I64 => (2, [1, 2, 1]),
        I::V_MAD_U64_U32 | I::V_MAD_I64_I32 => (2, [1, 1, 2]),
        I::V_CMP_F64(_) | I::V_CMPX_F64(_) | I::V_CMP_I64(_) | I::V_CMPX_I64(_) | I::V_CMP_U64(_) | I::V_CMPX_U64(_) => (2, [2, 2, 1]),
        op if wide(op) => (2, [2, 2, 2]),
        _ => (1, [1, 1, 1]),
    }
}

pub fn memory_words(op: I) -> u32 {
    let n = format!("{:?}", op);
    if n.ends_with("X16") {
        16
    } else if n.ends_with("X8") {
        8
    } else if n.ends_with("X4") || n.ends_with("XYZW") || n.ends_with("B128") {
        4
    } else if n.ends_with("X3") || n.ends_with("XYZ") || n.ends_with("B96") {
        3
    } else if n.ends_with("X2") || n.ends_with("XY") || n.ends_with("B64") {
        2
    } else {
        1
    }
}

const SDWA: u16 = 249;
const DPP: u16 = 250;

fn opcode(found: Result<(I, usize), ()>) -> Option<I> {
    found.ok().map(|(op, _)| op)
}

fn mimg_opcode(op: u32) -> Option<I> {
    Some(match op {
        0 => I::IMAGE_LOAD,
        1 => I::IMAGE_LOAD_MIP,
        2 => I::IMAGE_LOAD_PCK,
        3 => I::IMAGE_LOAD_PCK_SGN,
        4 => I::IMAGE_LOAD_MIP_PCK,
        5 => I::IMAGE_LOAD_MIP_PCK_SGN,
        8 => I::IMAGE_STORE,
        9 => I::IMAGE_STORE_MIP,
        10 => I::IMAGE_STORE_PCK,
        11 => I::IMAGE_STORE_MIP_PCK,
        14 => I::IMAGE_GET_RESINFO,
        32 => I::IMAGE_SAMPLE,
        36 => I::IMAGE_SAMPLE_L,
        39 => I::IMAGE_SAMPLE_LZ,
        _ => return None,
    })
}

fn vop3b(op: u32) -> Option<I> {
    match op {
        281..=286 | 488 | 489 => opcode(table::decode_vop3a_opcode_gcn3(op)),
        480 => Some(I::V_DIV_SCALE_F32),
        481 => Some(I::V_DIV_SCALE_F64),
        _ => None,
    }
}

fn takes_constant(op: I) -> bool {
    matches!(
        op,
        I::V_MADMK_F32 | I::V_MADAK_F32 | I::V_MADMK_F16 | I::V_MADAK_F16
    )
}

fn sdwa(second: u64) -> Sdwa {
    Sdwa {
        dst_sel: get_bits(second, 10, 8) as u8,
        dst_unused: get_bits(second, 12, 11) as u8,
        clamp: get_bits(second, 13, 13) != 0,
        src0_sel: get_bits(second, 18, 16) as u8,
        src0_sext: get_bits(second, 19, 19) != 0,
        src0_neg: get_bits(second, 20, 20) != 0,
        src0_abs: get_bits(second, 21, 21) != 0,
        src1_sel: get_bits(second, 26, 24) as u8,
        src1_sext: get_bits(second, 27, 27) != 0,
        src1_neg: get_bits(second, 28, 28) != 0,
        src1_abs: get_bits(second, 29, 29) != 0,
    }
}

fn dpp(second: u64) -> Dpp {
    Dpp {
        ctrl: get_bits(second, 16, 8) as u16,
        bound_ctrl: get_bits(second, 19, 19) != 0,
        src0_neg: get_bits(second, 20, 20) != 0,
        src0_abs: get_bits(second, 21, 21) != 0,
        src1_neg: get_bits(second, 22, 22) != 0,
        src1_abs: get_bits(second, 23, 23) != 0,
        bank_mask: get_bits(second, 27, 24) as u8,
        row_mask: get_bits(second, 31, 28) as u8,
    }
}

pub fn decode(memory: &[u8], pc: usize) -> Result<Inst, String> {
    let word = |at: usize| -> Result<u64, String> {
        memory
            .get(at..at + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as u64)
            .ok_or_else(|| format!("instruction at {pc:#x} is outside the loaded object"))
    };
    let first = word(pc)?;
    let undecodable = || format!("undecodable instruction {first:#010x} at {pc:#x}");
    let found = |op: Option<I>| op.ok_or_else(undecodable);
    let field = |high, low| get_bits(first, high, low);
    let literal = |used: bool| -> Result<Option<u32>, String> {
        if used {
            word(pc + 4).map(|w| Some(w as u32))
        } else {
            Ok(None)
        }
    };
    let short = |op: I, form: Form, uses: bool| -> Result<Inst, String> {
        let literal = literal(uses)?;
        Ok(Inst {
            op,
            size: if literal.is_some() { 8 } else { 4 },
            literal,
            form,
        })
    };
    let long = |op: I, form: Form| -> Inst {
        Inst {
            op,
            size: 8,
            literal: None,
            form,
        }
    };
    if field(31, 23) == 0b1_0111_1101 {
        let ssrc0 = field(7, 0) as u16;
        let op = found(opcode(table::decode_sop1_opcode_gcn3(field(15, 8) as u32)))?;
        let form = Form::Sop1 {
            sdst: field(22, 16) as u8,
            ssrc0,
        };
        return short(op, form, ssrc0 == LITERAL);
    }
    if field(31, 23) == 0b1_0111_1110 {
        let (ssrc0, ssrc1) = (field(7, 0) as u16, field(15, 8) as u16);
        let op = found(opcode(table::decode_sopc_opcode_gcn3(field(22, 16) as u32)))?;
        return short(op, Form::Sopc { ssrc0, ssrc1 }, ssrc0 == LITERAL || ssrc1 == LITERAL);
    }
    if field(31, 23) == 0b1_0111_1111 {
        let op = found(opcode(table::decode_sopp_opcode_gcn3(field(22, 16) as u32)))?;
        return short(
            op,
            Form::Sopp {
                simm16: field(15, 0) as u16,
            },
            false,
        );
    }
    if field(31, 28) == 0b1011 {
        let op = found(opcode(table::decode_sopk_opcode_gcn3(field(27, 23) as u32)))?;
        let form = Form::Sopk {
            sdst: field(22, 16) as u8,
            simm16: field(15, 0) as u16,
        };
        return short(op, form, matches!(op, I::S_SETREG_IMM32_B32));
    }
    if field(31, 30) == 0b10 {
        let (ssrc0, ssrc1) = (field(7, 0) as u16, field(15, 8) as u16);
        let op = found(opcode(table::decode_sop2_opcode_gcn3(field(29, 23) as u32)))?;
        let form = Form::Sop2 {
            sdst: field(22, 16) as u8,
            ssrc0,
            ssrc1,
        };
        return short(op, form, ssrc0 == LITERAL || ssrc1 == LITERAL);
    }
    if field(31, 25) == 0b011_1110 || field(31, 25) == 0b011_1111 || field(31, 31) == 0 {
        let src0 = field(8, 0) as u16;
        let vsrc1 = field(16, 9) as u8;
        let vdst = field(24, 17) as u8;
        let op = found(match field(31, 25) {
            0b011_1110 => opcode(table::decode_vopc_opcode_gcn3(field(24, 17) as u32)),
            0b011_1111 => opcode(table::decode_vop1_opcode_gcn3(field(16, 9) as u32)),
            _ => opcode(table::decode_vop2_opcode_gcn3(field(30, 25) as u32)),
        })?;
        let (src0, ext) = match src0 {
            SDWA | DPP => {
                if takes_constant(op) {
                    return Err(undecodable());
                }
                let second = word(pc + 4)?;
                let register = 256 + get_bits(second, 7, 0) as u16;
                let ext = if src0 == SDWA {
                    Ext::Sdwa(sdwa(second))
                } else {
                    Ext::Dpp(dpp(second))
                };
                (register, ext)
            }
            _ => (src0, Ext::Plain),
        };
        let form = match field(31, 25) {
            0b011_1110 => Form::Vopc { src0, vsrc1, ext },
            0b011_1111 => Form::Vop1 { vdst, src0, ext },
            _ => Form::Vop2 {
                vdst,
                src0,
                vsrc1,
                ext,
            },
        };
        if ext != Ext::Plain {
            return Ok(Inst {
                op,
                size: 8,
                literal: None,
                form,
            });
        }
        if src0 == LITERAL && takes_constant(op) {
            return Err(undecodable());
        }
        return short(op, form, src0 == LITERAL || takes_constant(op));
    }
    match field(31, 26) {
        0b11_0000 => {
            let second = word(pc + 4)?;
            let op = found(opcode(table::decode_smem_opcode_gcn3(field(25, 18) as u32)))?;
            let imm = field(17, 17) != 0;
            Ok(long(
                op,
                Form::Smem {
                    sdata: field(12, 6) as u8,
                    sbase: field(5, 0) as u8,
                    offset: if imm {
                        get_bits(second, 19, 0) as u32
                    } else {
                        get_bits(second, 6, 0) as u32
                    },
                    imm,
                    glc: field(16, 16) != 0,
                },
            ))
        }
        0b11_0100 => {
            let second = word(pc + 4)?;
            let number = field(25, 16) as u32;
            let (op, sdst) = match vop3b(number) {
                Some(op) => (op, Some(field(14, 8) as u8)),
                None => (found(opcode(table::decode_vop3a_opcode_gcn3(number)))?, None),
            };
            if matches!(op, I::V_MADMK_F32 | I::V_MADAK_F32 | I::V_MADMK_F16 | I::V_MADAK_F16) {
                return Err(undecodable());
            }
            Ok(long(
                op,
                Form::Vop3 {
                    vdst: field(7, 0) as u8,
                    sdst,
                    src: [
                        get_bits(second, 8, 0) as u16,
                        get_bits(second, 17, 9) as u16,
                        get_bits(second, 26, 18) as u16,
                    ],
                    abs: if sdst.is_some() { 0 } else { field(10, 8) as u8 },
                    neg: get_bits(second, 31, 29) as u8,
                    clamp: field(15, 15) != 0,
                    omod: get_bits(second, 28, 27) as u8,
                },
            ))
        }
        0b11_0110 => {
            let second = word(pc + 4)?;
            let op = found(opcode(table::decode_ds_opcode_gcn3(field(24, 17) as u32)))?;
            Ok(long(
                op,
                Form::Ds {
                    offset0: field(7, 0) as u8,
                    offset1: field(15, 8) as u8,
                    gds: field(16, 16) != 0,
                    addr: get_bits(second, 7, 0) as u8,
                    data0: get_bits(second, 15, 8) as u8,
                    data1: get_bits(second, 23, 16) as u8,
                    vdst: get_bits(second, 31, 24) as u8,
                },
            ))
        }
        0b11_0111 => {
            let second = word(pc + 4)?;
            let op = found(opcode(table::decode_flat_opcode_gcn3(field(24, 18) as u32)))?;
            Ok(long(
                op,
                Form::Flat {
                    glc: field(16, 16) != 0,
                    slc: field(17, 17) != 0,
                    addr: get_bits(second, 7, 0) as u8,
                    data: get_bits(second, 15, 8) as u8,
                    tfe: get_bits(second, 23, 23) != 0,
                    vdst: get_bits(second, 31, 24) as u8,
                },
            ))
        }
        0b11_1000 => {
            let second = word(pc + 4)?;
            let op = found(opcode(table::decode_mubuf_opcode_gcn3(field(24, 18) as u32)))?;
            Ok(long(
                op,
                Form::Mubuf {
                    offset: field(11, 0) as u16,
                    offen: field(12, 12) != 0,
                    idxen: field(13, 13) != 0,
                    glc: field(14, 14) != 0,
                    lds: field(16, 16) != 0,
                    slc: field(17, 17) != 0,
                    vaddr: get_bits(second, 7, 0) as u8,
                    vdata: get_bits(second, 15, 8) as u8,
                    srsrc: get_bits(second, 20, 16) as u8,
                    tfe: get_bits(second, 23, 23) != 0,
                    soffset: get_bits(second, 31, 24) as u16,
                },
            ))
        }
        0b11_1100 => {
            let second = word(pc + 4)?;
            let op = found(mimg_opcode(field(24, 18) as u32))?;
            Ok(long(
                op,
                Form::Mimg {
                    dmask: field(11, 8) as u8,
                    unorm: field(12, 12) != 0,
                    glc: field(13, 13) != 0,
                    da: field(14, 14) != 0,
                    r128: field(15, 15) != 0,
                    tfe: field(16, 16) != 0,
                    lwe: field(17, 17) != 0,
                    slc: field(25, 25) != 0,
                    vaddr: get_bits(second, 7, 0) as u8,
                    vdata: get_bits(second, 15, 8) as u8,
                    srsrc: get_bits(second, 20, 16) as u8,
                    ssamp: get_bits(second, 25, 21) as u8,
                    d16: get_bits(second, 31, 31) != 0,
                },
            ))
        }
        _ => Err(undecodable()),
    }
}
