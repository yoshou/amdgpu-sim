use super::decode::{decode, memory_words, vector_words, wide, Dpp, Ext, Form, Inst, Sdwa, LITERAL};
use crate::bit::get_bits;
use crate::instructions::I;
use object::{ElfFile, Object, ObjectSection, ObjectSegment};

fn name(op: I) -> String {
    if matches!(op, I::S_CMP_NE_U64) {
        return "s_cmp_lg_u64".to_string();
    }
    let debug = format!("{:?}", op);
    if let Some((base, predicate)) = debug.strip_suffix(')').and_then(|d| d.split_once('(')) {
        let (prefix, ty) = base.rsplit_once('_').unwrap();
        let predicate = predicate.to_lowercase();
        let predicate = if ty.starts_with('F') {
            predicate
        } else {
            match predicate.as_str() {
                "lg" => "ne".to_string(),
                "tru" => "t".to_string(),
                _ => predicate,
            }
        };
        return format!("{}_{}_{}", prefix.to_lowercase(), predicate, ty.to_lowercase());
    }
    debug.to_lowercase()
}

fn range(prefix: &str, first: u16, words: u32) -> String {
    if words == 1 {
        format!("{prefix}{first}")
    } else {
        format!("{prefix}[{}:{}]", first, first as u32 + words - 1)
    }
}

fn special(code: u16, words: u32, name: &str) -> String {
    match words {
        1 => format!("{name}_{}", if code % 2 == 0 { "lo" } else { "hi" }),
        _ => name.to_string(),
    }
}

fn operand(code: u16, words: u32, literal: Option<u32>, double: bool) -> String {
    match code {
        0..=101 => range("s", code, words),
        102 | 103 => special(code, words, "flat_scratch"),
        104 | 105 => special(code, words, "xnack_mask"),
        106 | 107 => special(code, words, "vcc"),
        108 | 109 => special(code, words, "tba"),
        110 | 111 => special(code, words, "tma"),
        112..=123 => range("ttmp", code - 112, words),
        124 => "m0".to_string(),
        126 | 127 => special(code, words, "exec"),
        128..=192 => format!("{}", code - 128),
        193..=208 => format!("-{}", code - 192),
        240 => "0.5".to_string(),
        241 => "-0.5".to_string(),
        242 => "1.0".to_string(),
        243 => "-1.0".to_string(),
        244 => "2.0".to_string(),
        245 => "-2.0".to_string(),
        246 => "4.0".to_string(),
        247 => "-4.0".to_string(),
        248 if double => "0.15915494309189532".to_string(),
        248 => "0.15915494".to_string(),
        251 => "src_vccz".to_string(),
        252 => "src_execz".to_string(),
        253 => "src_scc".to_string(),
        254 => "src_lds_direct".to_string(),
        LITERAL => {
            let value = literal.unwrap();
            let inline = (-16..=64).contains(&(value as i32))
                || [0.5f32, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0, -4.0].iter().any(|f| f.to_bits() == value)
                || value == 0x3e22_f983;
            if inline {
                format!("lit(0x{value:x})")
            } else {
                format!("0x{value:x}")
            }
        }
        256..=511 => range("v", code - 256, words),
        _ => format!("<{code}>"),
    }
}

fn vgpr(r: u8, words: u32) -> String {
    range("v", r as u16, words)
}

fn modified(text: String, abs: bool, neg: bool) -> String {
    let text = if abs { format!("|{text}|") } else { text };
    if neg {
        format!("-{text}")
    } else {
        text
    }
}

fn scalar_words(op: I) -> (u32, [u32; 2]) {
    match op {
        I::S_LSHL_B64 | I::S_LSHR_B64 | I::S_ASHR_I64 | I::S_BFE_U64 | I::S_BFE_I64 => (2, [2, 1]),
        I::S_BFM_B64 => (2, [1, 1]),
        I::S_BCNT0_I32_B64 | I::S_BCNT1_I32_B64 | I::S_FF0_I32_B64 | I::S_FF1_I32_B64 | I::S_FLBIT_I32_B64 | I::S_FLBIT_I32_I64 => (1, [2, 2]),
        I::S_BITCMP0_B64 | I::S_BITCMP1_B64 => (1, [2, 1]),
        I::S_BITSET0_B64 | I::S_BITSET1_B64 => (2, [1, 1]),
        op if wide(op) => (2, [2, 2]),
        _ => (1, [1, 1]),
    }
}

fn compare(op: I) -> bool {
    format!("{:?}", op).starts_with("V_CMP")
}

fn carry(op: I) -> Option<bool> {
    match op {
        I::V_ADD_U32 | I::V_SUB_U32 | I::V_SUBREV_U32 => Some(false),
        I::V_ADDC_U32 | I::V_SUBB_U32 | I::V_SUBBREV_U32 => Some(true),
        _ => None,
    }
}

fn arity(op: I, number: u32) -> usize {
    match number {
        0..=255 => 2,
        256..=319 => match op {
            I::V_CNDMASK_B32 | I::V_ADDC_U32 | I::V_SUBB_U32 | I::V_SUBBREV_U32 => 3,
            _ => 2,
        },
        320..=447 => match op {
            I::V_NOP | I::V_CLREXCP => 0,
            _ => 1,
        },
        448..=511 => 3,
        _ => match op {
            I::V_INTERP_MOV_F32 => 1,
            _ => 2,
        },
    }
}

fn select(sel: u8) -> &'static str {
    ["BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD", "<7>"][sel as usize]
}

fn sdwa_source(code: u16, sext: bool, abs: bool, neg: bool) -> String {
    let text = operand(code, 1, None, false);
    let text = if sext { format!("sext({text})") } else { text };
    modified(text, abs, neg)
}

fn sdwa_suffix(s: &Sdwa, dst: bool, sources: usize) -> String {
    let mut text = String::new();
    if s.clamp {
        text += " clamp";
    }
    if dst {
        text += &format!(" dst_sel:{}", select(s.dst_sel));
        text += &format!(
            " dst_unused:{}",
            ["UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE", "<3>"][s.dst_unused as usize]
        );
    }
    text += &format!(" src0_sel:{}", select(s.src0_sel));
    if sources > 1 {
        text += &format!(" src1_sel:{}", select(s.src1_sel));
    }
    text
}

fn dpp_suffix(d: &Dpp) -> String {
    let control = match d.ctrl {
        0x000..=0x0ff => format!(
            "quad_perm:[{},{},{},{}]",
            d.ctrl & 3,
            d.ctrl >> 2 & 3,
            d.ctrl >> 4 & 3,
            d.ctrl >> 6 & 3
        ),
        0x101..=0x10f => format!("row_shl:{}", d.ctrl & 15),
        0x111..=0x11f => format!("row_shr:{}", d.ctrl & 15),
        0x121..=0x12f => format!("row_ror:{}", d.ctrl & 15),
        0x130 => "wave_shl:1".to_string(),
        0x134 => "wave_rol:1".to_string(),
        0x138 => "wave_shr:1".to_string(),
        0x13c => "wave_ror:1".to_string(),
        0x140 => "row_mirror".to_string(),
        0x141 => "row_half_mirror".to_string(),
        0x142 => "row_bcast:15".to_string(),
        0x143 => "row_bcast:31".to_string(),
        c => format!("<{c:#x}>"),
    };
    format!(
        " {control} row_mask:0x{:x} bank_mask:0x{:x}{}",
        d.row_mask,
        d.bank_mask,
        if d.bound_ctrl { " bound_ctrl:1" } else { "" }
    )
}

fn hwreg(simm16: u16) -> String {
    let id = simm16 & 63;
    let offset = simm16 >> 6 & 31;
    let size = (simm16 >> 11) + 1;
    let name = match id {
        1 => "HW_REG_MODE".to_string(),
        2 => "HW_REG_STATUS".to_string(),
        3 => "HW_REG_TRAPSTS".to_string(),
        4 => "HW_REG_HW_ID".to_string(),
        5 => "HW_REG_GPR_ALLOC".to_string(),
        6 => "HW_REG_LDS_ALLOC".to_string(),
        7 => "HW_REG_IB_STS".to_string(),
        n => format!("{n}"),
    };
    if offset == 0 && size == 32 {
        format!("hwreg({name})")
    } else {
        format!("hwreg({name}, {offset}, {size})")
    }
}

fn waitcnt(simm16: u16) -> String {
    let mut parts = vec![];
    let vm = simm16 & 15;
    let exp = simm16 >> 4 & 7;
    let lgkm = simm16 >> 8 & 15;
    if vm != 15 {
        parts.push(format!("vmcnt({vm})"));
    }
    if exp != 7 {
        parts.push(format!("expcnt({exp})"));
    }
    if lgkm != 15 {
        parts.push(format!("lgkmcnt({lgkm})"));
    }
    if parts.is_empty() {
        format!("{simm16}")
    } else {
        parts.join(" ")
    }
}

fn smem(i: &Inst, sdata: u8, sbase: u8, offset: u32, imm: bool, glc: bool) -> String {
    let n = name(i.op);
    let buffer = n.starts_with("s_buffer");
    let base = range("s", sbase as u16 * 2, if buffer { 4 } else { 2 });
    let mut text = match i.op {
        I::S_DCACHE_INV | I::S_DCACHE_WB | I::S_DCACHE_INV_VOL | I::S_DCACHE_WB_VOL => return n,
        I::S_MEMTIME | I::S_MEMREALTIME => format!("{n} {}", range("s", sdata as u16, 2)),
        _ => {
            let at = if imm {
                format!("0x{offset:x}")
            } else {
                operand(offset as u16, 1, None, false)
            };
            format!("{n} {}, {base}, {at}", operand(sdata as u16, memory_words(i.op), None, false))
        }
    };
    if glc {
        text += " glc";
    }
    text
}

fn ds(i: &Inst, offset0: u8, offset1: u8, gds: bool, addr: u8, data0: u8, data1: u8, vdst: u8) -> String {
    let n = name(i.op);
    let element = memory_words(i.op);
    let two = n.contains("read2") || n.contains("write2") || n.contains("wrxchg2");
    let words = if two { 2 * element } else { element };
    let mut operands = vec![];
    let returns = n.contains("read") || n.contains("rtn") || n.contains("bpermute") || n.contains("permute") || n.contains("swizzle") || n.contains("consume") || n.contains("append");
    if returns {
        operands.push(vgpr(vdst, words));
    }
    if !matches!(i.op, I::DS_CONSUME | I::DS_APPEND) {
        operands.push(vgpr(addr, 1));
    }
    let writes = !(n.contains("read") || matches!(i.op, I::DS_CONSUME | I::DS_APPEND));
    if writes {
        operands.push(vgpr(data0, element));
        if two || n.contains("cmpst") || n.contains("mskor") {
            operands.push(vgpr(data1, element));
        }
    }
    let mut text = format!("{n} {}", operands.join(", "));
    if two {
        if offset0 != 0 {
            text += &format!(" offset0:{offset0}");
        }
        if offset1 != 0 {
            text += &format!(" offset1:{offset1}");
        }
    } else {
        let offset = offset0 as u32 | (offset1 as u32) << 8;
        if offset != 0 {
            text += &format!(" offset:{offset}");
        }
    }
    if gds {
        text += " gds";
    }
    text
}

fn flat(i: &Inst, glc: bool, slc: bool, addr: u8, data: u8, tfe: bool, vdst: u8) -> String {
    let n = name(i.op);
    let words = memory_words(i.op);
    let atomic = n.contains("atomic");
    let operands = if n.contains("load") {
        format!("{}, {}", vgpr(vdst, words + tfe as u32), vgpr(addr, 2))
    } else if atomic {
        let pair = if n.contains("_x2") { 2 } else { 1 };
        let given = if n.contains("cmpswap") { pair * 2 } else { pair };
        if glc {
            format!("{}, {}, {}", vgpr(vdst, pair), vgpr(addr, 2), vgpr(data, given))
        } else {
            format!("{}, {}", vgpr(addr, 2), vgpr(data, given))
        }
    } else {
        format!("{}, {}", vgpr(addr, 2), vgpr(data, words))
    };
    let mut text = format!("{n} {operands}");
    if glc {
        text += " glc";
    }
    if slc {
        text += " slc";
    }
    if tfe {
        text += " tfe";
    }
    text
}

#[allow(clippy::too_many_arguments)]
fn mubuf(i: &Inst, offset: u16, offen: bool, idxen: bool, glc: bool, lds: bool, slc: bool, vaddr: u8, vdata: u8, srsrc: u8, tfe: bool, soffset: u16) -> String {
    let n = name(i.op);
    if matches!(i.op, I::BUFFER_WBINVL1 | I::BUFFER_WBINVL1_VOL) {
        return n;
    }
    let words = memory_words(i.op);
    let address = match (offen, idxen) {
        (false, false) => "off".to_string(),
        (true, true) => vgpr(vaddr, 2),
        _ => vgpr(vaddr, 1),
    };
    let mut text = format!(
        "{n} {}, {address}, {}, {}",
        vgpr(vdata, words + tfe as u32),
        range("s", srsrc as u16 * 4, 4),
        operand(soffset, 1, None, false)
    );
    if idxen {
        text += " idxen";
    }
    if offen {
        text += " offen";
    }
    if offset != 0 {
        text += &format!(" offset:{offset}");
    }
    if glc {
        text += " glc";
    }
    if slc {
        text += " slc";
    }
    if lds {
        text += " lds";
    }
    if tfe {
        text += " tfe";
    }
    text
}

fn sopp(op: I, simm16: u16) -> String {
    let n = name(op);
    match op {
        I::S_ENDPGM | I::S_BARRIER | I::S_ICACHE_INV | I::S_TTRACEDATA | I::S_ENDPGM_SAVED | I::S_SET_GPR_IDX_OFF => n,
        I::S_WAITCNT => format!("{n} {}", waitcnt(simm16)),
        _ => format!("{n} {simm16}"),
    }
}

fn vop3(i: &Inst, memory: &[u8], pc: usize, vdst: u8, sdst: Option<u8>, src: [u16; 3], abs: u8, neg: u8, clamp: bool, omod: u8) -> String {
    let number = get_bits(u32::from_le_bytes([memory[pc], memory[pc + 1], memory[pc + 2], memory[pc + 3]]) as u64, 25, 16) as u32;
    let suffix = if number < 448 { "_e64" } else { "" };
    let n = format!("{}{suffix}", name(i.op));
    let (dst_words, src_words) = vector_words(i.op);
    let count = arity(i.op, number);
    let double = format!("{:?}", i.op).contains("F64");
    let mut operands = vec![];
    let source = |k: usize, words: u32| modified(operand(src[k], words, None, double), abs >> k & 1 != 0, neg >> k & 1 != 0);
    match i.op {
        I::V_READLANE_B32 => {
            operands.push(operand(vdst as u16, 1, None, false));
            operands.push(source(0, 1));
            operands.push(source(1, 1));
        }
        I::V_WRITELANE_B32 => {
            operands.push(vgpr(vdst, 1));
            operands.push(source(0, 1));
            operands.push(source(1, 1));
        }
        I::V_READFIRSTLANE_B32 => {
            operands.push(operand(vdst as u16, 1, None, false));
            operands.push(source(0, 1));
        }
        op if compare(op) && number < 256 => {
            operands.push(operand(vdst as u16, 2, None, false));
            for k in 0..count {
                operands.push(source(k, src_words[k]));
            }
        }
        _ => {
            operands.push(vgpr(vdst, dst_words));
            if let Some(sdst) = sdst {
                operands.push(operand(sdst as u16, 2, None, false));
            }
            for k in 0..count {
                let words = if k == 2 && matches!(i.op, I::V_CNDMASK_B32 | I::V_ADDC_U32 | I::V_SUBB_U32 | I::V_SUBBREV_U32) {
                    2
                } else {
                    src_words[k]
                };
                operands.push(source(k, words));
            }
        }
    }
    let mut text = if operands.is_empty() { n } else { format!("{n} {}", operands.join(", ")) };
    if clamp {
        text += " clamp";
    }
    match omod {
        1 => text += " mul:2",
        2 => text += " mul:4",
        3 => text += " div:2",
        _ => {}
    }
    text
}

fn short_vector(i: &Inst, form: &Form) -> String {
    let (dst_words, src_words) = vector_words(i.op);
    let double = format!("{:?}", i.op).contains("F64");
    let (src0, ext) = match *form {
        Form::Vop1 { src0, ext, .. } | Form::Vop2 { src0, ext, .. } | Form::Vopc { src0, ext, .. } => (src0, ext),
        _ => unreachable!(),
    };
    let suffix = match ext {
        Ext::Plain if matches!(i.op, I::V_MADMK_F32 | I::V_MADAK_F32 | I::V_MADMK_F16 | I::V_MADAK_F16 | I::V_READFIRSTLANE_B32) => "",
        Ext::Plain => "_e32",
        Ext::Sdwa(_) if matches!(form, Form::Vopc { .. }) => "",
        Ext::Sdwa(_) => "_sdwa",
        Ext::Dpp(_) => "_dpp",
    };
    let n = format!("{}{suffix}", name(i.op));
    let (abs, neg, sext) = match ext {
        Ext::Sdwa(s) => ([s.src0_abs, s.src1_abs], [s.src0_neg, s.src1_neg], [s.src0_sext, s.src1_sext]),
        Ext::Dpp(d) => ([d.src0_abs, d.src1_abs], [d.src0_neg, d.src1_neg], [false, false]),
        Ext::Plain => ([false; 2], [false; 2], [false; 2]),
    };
    let first = if matches!(ext, Ext::Sdwa(_)) {
        sdwa_source(src0, sext[0], abs[0], neg[0])
    } else {
        modified(operand(src0, src_words[0], i.literal, double), abs[0], neg[0])
    };
    let second = |vsrc1: u8| {
        if matches!(ext, Ext::Sdwa(_)) {
            sdwa_source(256 + vsrc1 as u16, sext[1], abs[1], neg[1])
        } else {
            modified(vgpr(vsrc1, src_words[1]), abs[1], neg[1])
        }
    };
    let (operands, sources) = match *form {
        Form::Vop1 { vdst, .. } => {
            let dst = if matches!(i.op, I::V_READFIRSTLANE_B32) {
                operand(vdst as u16, 1, None, false)
            } else {
                vgpr(vdst, dst_words)
            };
            if matches!(i.op, I::V_NOP | I::V_CLREXCP) {
                (vec![], 0)
            } else {
                (vec![dst, first], 1)
            }
        }
        Form::Vop2 { vdst, vsrc1, .. } => {
            let mut operands = vec![vgpr(vdst, dst_words)];
            if carry(i.op).is_some() {
                operands.push("vcc".to_string());
            }
            match i.op {
                I::V_MADMK_F32 | I::V_MADMK_F16 => {
                    operands.push(first);
                    operands.push(format!("0x{:x}", i.literal.unwrap()));
                    operands.push(second(vsrc1));
                }
                I::V_MADAK_F32 | I::V_MADAK_F16 => {
                    operands.push(first);
                    operands.push(second(vsrc1));
                    operands.push(format!("0x{:x}", i.literal.unwrap()));
                }
                _ => {
                    operands.push(first);
                    operands.push(second(vsrc1));
                }
            }
            if carry(i.op) == Some(true) || matches!(i.op, I::V_CNDMASK_B32) {
                operands.push("vcc".to_string());
            }
            (operands, 2)
        }
        Form::Vopc { vsrc1, .. } => (vec!["vcc".to_string(), first, second(vsrc1)], 2),
        _ => unreachable!(),
    };
    let mut text = if operands.is_empty() { n } else { format!("{n} {}", operands.join(", ")) };
    match ext {
        Ext::Sdwa(s) => text += &sdwa_suffix(&s, !matches!(form, Form::Vopc { .. }), sources),
        Ext::Dpp(d) => text += &dpp_suffix(&d),
        Ext::Plain => {}
    }
    text
}

fn text(memory: &[u8], pc: usize, i: &Inst) -> String {
    let n = name(i.op);
    match i.form {
        Form::Sop2 { sdst, ssrc0, ssrc1 } => {
            let (d, s) = scalar_words(i.op);
            format!(
                "{n} {}, {}, {}",
                operand(sdst as u16, d, None, false),
                operand(ssrc0, s[0], i.literal, false),
                operand(ssrc1, s[1], i.literal, false)
            )
        }
        Form::Sop1 { sdst, ssrc0 } => {
            let (d, s) = scalar_words(i.op);
            match i.op {
                I::S_GETPC_B64 => format!("{n} {}", operand(sdst as u16, 2, None, false)),
                I::S_SETPC_B64 | I::S_RFE_B64 => format!("{n} {}", operand(ssrc0, 2, i.literal, false)),
                I::S_CBRANCH_JOIN | I::S_SET_GPR_IDX_IDX => format!("{n} {}", operand(ssrc0, 1, i.literal, false)),
                _ => format!(
                    "{n} {}, {}",
                    operand(sdst as u16, d, None, false),
                    operand(ssrc0, s[0], i.literal, false)
                ),
            }
        }
        Form::Sopc { ssrc0, ssrc1 } => {
            let (_, s) = scalar_words(i.op);
            format!(
                "{n} {}, {}",
                operand(ssrc0, s[0], i.literal, false),
                operand(ssrc1, s[1], i.literal, false)
            )
        }
        Form::Sopk { sdst, simm16 } => match i.op {
            I::S_GETREG_B32 => format!("{n} {}, {}", operand(sdst as u16, 1, None, false), hwreg(simm16)),
            I::S_SETREG_B32 => format!("{n} {}, {}", hwreg(simm16), operand(sdst as u16, 1, None, false)),
            I::S_SETREG_IMM32_B32 => {
                let value = i.literal.unwrap();
                if (-16..=64).contains(&(value as i32)) {
                    format!("{n} {}, {}", hwreg(simm16), value as i32)
                } else {
                    format!("{n} {}, 0x{value:x}", hwreg(simm16))
                }
            }
            _ => format!("{n} {}, 0x{:x}", operand(sdst as u16, 1, None, false), simm16),
        },
        Form::Sopp { simm16 } => sopp(i.op, simm16),
        Form::Smem { sdata, sbase, offset, imm, glc } => smem(i, sdata, sbase, offset, imm, glc),
        Form::Vop1 { .. } | Form::Vop2 { .. } | Form::Vopc { .. } => short_vector(i, &i.form),
        Form::Vop3 { vdst, sdst, src, abs, neg, clamp, omod } => vop3(i, memory, pc, vdst, sdst, src, abs, neg, clamp, omod),
        Form::Ds { offset0, offset1, gds, addr, data0, data1, vdst } => ds(i, offset0, offset1, gds, addr, data0, data1, vdst),
        Form::Flat { glc, slc, addr, data, tfe, vdst } => flat(i, glc, slc, addr, data, tfe, vdst),
        Form::Mubuf { offset, offen, idxen, glc, lds, slc, vaddr, vdata, srsrc, tfe, soffset } => {
            mubuf(i, offset, offen, idxen, glc, lds, slc, vaddr, vdata, srsrc, tfe, soffset)
        }
        Form::Mimg { dmask, unorm, glc, da, r128, tfe, lwe, slc, vaddr, vdata, srsrc, ssamp, d16 } => {
            let words = (dmask as u32).count_ones().max(1) + tfe as u32;
            let mut text = format!("{n} {}, {}, {}", vgpr(vdata, words), vgpr(vaddr, 1), range("s", srsrc as u16 * 4, if r128 { 4 } else { 8 }));
            if n.contains("sample") || n.contains("gather") {
                text += &format!(", {}", range("s", ssamp as u16 * 4, 4));
            }
            text += &format!(" dmask:0x{dmask:x}");
            for (set, flag) in [(unorm, "unorm"), (glc, "glc"), (slc, "slc"), (r128, "r128"), (tfe, "tfe"), (lwe, "lwe"), (da, "da"), (d16, "d16")] {
                if set {
                    text += &format!(" {flag}");
                }
            }
            text
        }
    }
}

fn image(path: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let elf = ElfFile::parse(&bytes).unwrap();
    let mut memory = Vec::new();
    let mut place = |start: usize, data: &[u8]| {
        if memory.len() < start + data.len() {
            memory.resize(start + data.len(), 0);
        }
        memory[start..start + data.len()].copy_from_slice(data);
    };
    let mut segments = 0;
    for segment in elf.segments() {
        segments += 1;
        let data = segment.data();
        place(segment.address() as usize, &data[..data.len().min(segment.size() as usize)]);
    }
    if segments == 0 {
        for section in elf.sections().filter(|s| s.name() == Some(".text")) {
            place(section.address() as usize, section.data());
        }
    }
    memory
}

fn tool(name: &str) -> std::path::PathBuf {
    let prefix = std::env::var("LLVM_SYS_221_PREFIX").unwrap_or_else(|_| "/usr/lib/llvm-22".to_string());
    std::path::Path::new(&prefix).join("bin").join(name)
}

fn disassembly(path: &str) -> Vec<(usize, usize, String)> {
    let output = std::process::Command::new(tool("llvm-objdump"))
        .args(["-d", "--mcpu=gfx803", path])
        .output()
        .expect("llvm-objdump runs");
    assert!(output.status.success());
    let text = String::from_utf8(output.stdout).unwrap();
    let mut function = 0;
    text.lines()
        .filter_map(|line| {
            if line.ends_with(">:") {
                function += 1;
                return None;
            }
            if !line.starts_with('\t') {
                return None;
            }
            let (code, comment) = line.split_once("//")?;
            let address = comment.trim().split(':').next()?;
            let address = usize::from_str_radix(address, 16).ok()?;
            Some((function, address, code.split_whitespace().collect::<Vec<_>>().join(" ")))
        })
        .collect()
}

const OBJECTS: [&str; 5] = [
    "tests/data/kernels_gfx803.o",
    "examples/bitonic_sort/kernel_gfx803.o",
    "examples/histogram/kernel_gfx803.o",
    "examples/smallpt/kernel_gfx803.o",
    "examples/texture/kernel_gfx803.o",
];

fn reproduce(paths: &[&str]) -> (usize, Vec<String>) {
    let mut mismatches = vec![];
    let mut count = 0;
    for &path in paths {
        let memory = image(path);
        let lines = disassembly(path);
        for (k, (function, pc, expected)) in lines.iter().enumerate() {
            count += 1;
            let decoded = match decode(&memory, *pc) {
                Ok(i) => i,
                Err(e) => {
                    mismatches.push(format!("{path} {pc:#x}: {e}; llvm reads {expected}"));
                    continue;
                }
            };
            if let Some((same, next, _)) = lines.get(k + 1) {
                if same == function && next - pc != decoded.size {
                    mismatches.push(format!("{path} {pc:#x}: size {} but the next instruction is at {next:#x} ({expected})", decoded.size));
                }
            }
            let printed = text(&memory, *pc, &decoded).split_whitespace().collect::<Vec<_>>().join(" ");
            if printed != *expected {
                mismatches.push(format!("{path} {pc:#x}: {printed}  !=  {expected}"));
            }
        }
    }
    (count, mismatches)
}

#[test]
fn decoding_reproduces_the_llvm_disassembly_of_every_gfx803_object() {
    let (count, mismatches) = reproduce(&OBJECTS);
    assert!(count > 9000, "only {} instructions", count);
    assert!(mismatches.is_empty(), "{} of {count} differ:\n{}", mismatches.len(), mismatches.iter().take(60).cloned().collect::<Vec<_>>().join("\n"));
}

const FIELDS: &str = "\
v_mov_b32_dpp v0, v1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf
v_add_f32_dpp v0, -v1, |v2| row_shl:1 row_mask:0xa bank_mask:0x5 bound_ctrl:0
v_add_u32_dpp v0, vcc, v1, v2 row_shr:3 row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v3, v4 wave_shl:1 row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v3, v4 wave_ror:1 row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v3, v4 row_bcast:15 row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v3, v4 row_mirror row_mask:0xf bank_mask:0xf
v_mov_b32_dpp v3, v4 row_ror:7 row_mask:0x3 bank_mask:0xc
v_mov_b32_sdwa v0, sext(v1) dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_2
v_add_f32_sdwa v0, -v1, |v2| clamp dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:WORD_0 src1_sel:BYTE_3
v_cmp_eq_u32_sdwa vcc, v1, v2 src0_sel:WORD_1 src1_sel:BYTE_0
v_add_u32_sdwa v1, vcc, v2, v3 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:WORD_1
v_add_f32_e64 v0, -|v1|, s2 clamp mul:2
v_fma_f32 v0, v1, -v2, |v3| div:2
v_mul_f64 v[0:1], -v[2:3], 4.0 mul:4
v_mad_u64_u32 v[0:1], s[2:3], v4, v5, v[6:7]
v_div_scale_f32 v0, vcc, v1, v2, v3
v_div_scale_f64 v[0:1], s[4:5], v[2:3], v[2:3], 1.0
v_cndmask_b32_e64 v0, 0, 1.0, s[4:5]
v_cmpx_lt_f32_e32 vcc, v0, v1
v_cmpx_lt_f32_e64 s[0:1], v0, v1
v_cmp_class_f32_e64 s[2:3], -v0, s5
v_cmp_t_i32_e32 vcc, 0, v1
v_cmp_tru_f32_e32 vcc, v0, v1
v_cmp_f_u64_e64 s[6:7], v[0:1], s[2:3]
v_subbrev_u32_e64 v0, s[0:1], v1, v2, s[2:3]
v_addc_u32_e32 v0, vcc, v3, v1, vcc
v_trig_preop_f64 v[0:1], |v[2:3]|, v4
v_ldexp_f64 v[0:1], 0.15915494309189532, v2
v_rcp_f32_e32 v0, 0.15915494
v_readlane_b32 s5, v6, 63
v_writelane_b32 v1, m0, 5
v_mbcnt_hi_u32_b32 v0, exec_hi, v0
v_bfe_i32 v0, v1, 0x10, 5
v_alignbit_b32 v0, v1, v2, 7
v_cvt_pkrtz_f16_f32 v0, v1, v2
v_mac_f32_e64 v0, v1, v2 clamp
v_madmk_f32 v0, v1, 0x40490fdb, v2
v_madak_f16 v0, v1, v2, 0x3c00
v_mov_b32_e32 v0, src_lds_direct
v_mov_b32_e32 v0, src_vccz
v_mov_b32_e32 v0, src_execz
v_mov_b32_e32 v0, src_scc
v_mov_b32_e32 v0, flat_scratch_hi
v_mov_b32_e32 v0, vcc_hi
v_mov_b32_e32 v0, ttmp3
v_mov_b32_e32 v0, tba_lo
s_mov_b64 s[0:1], flat_scratch
s_mov_b64 ttmp[4:5], s[2:3]
s_mov_b32 s0, 0x12345678
s_mov_b32 s0, lit(0x40)
s_and_b32 s0, 0x80000000, 0x80000000
s_cmp_eq_u64 s[0:1], s[2:3]
s_bitcmp1_b64 s[2:3], 63
s_lshr_b64 s[0:1], s[2:3], 63
s_bfe_u64 s[0:1], s[2:3], 0x80000
s_bcnt0_i32_b64 s0, exec
s_flbit_i32_i64 s0, s[2:3]
s_getpc_b64 s[0:1]
s_swappc_b64 s[30:31], s[4:5]
s_setpc_b64 s[30:31]
s_cbranch_join s4
s_movk_i32 s0, 0xffff
s_cmpk_lg_u32 s1, 0x8000
s_mulk_i32 s2, 0x7
s_getreg_b32 s0, hwreg(HW_REG_MODE)
s_getreg_b32 s0, hwreg(HW_REG_STATUS, 3, 4)
s_setreg_b32 hwreg(HW_REG_TRAPSTS, 8, 3), s7
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 4), 0x12345
s_waitcnt vmcnt(1) expcnt(2) lgkmcnt(3)
s_waitcnt expcnt(0)
s_nop 7
s_sleep 2
s_setprio 3
s_trap 2
s_cbranch_execz 3
s_cbranch_vccz -5
s_branch 0
s_barrier
s_endpgm
s_load_dwordx4 s[0:3], s[4:5], s6
s_load_dword s0, s[4:5], 0xfffff glc
s_load_dwordx16 s[16:31], s[2:3], 0x40
s_buffer_load_dwordx2 s[0:1], s[4:7], 0x10
s_buffer_load_dwordx8 s[8:15], s[12:15], m0
s_store_dword s1, s[2:3], 0x8 glc
s_memtime s[0:1]
s_dcache_inv
buffer_load_dword v1, v[2:3], s[4:7], s8 idxen offen offset:4095 glc slc
buffer_store_dword v1, v2, s[8:11], 0 idxen offset:12
buffer_load_ubyte v[1:2], off, s[0:3], s4 offset:7 tfe
buffer_load_dwordx4 v[4:7], v1, s[12:15], m0 offen
buffer_store_dwordx2 v[1:2], off, s[96:99], 64 glc
buffer_wbinvl1_vol
ds_write_b32 v1, v2 offset:65535 gds
ds_read2st64_b64 v[0:3], v4 offset0:1 offset1:255
ds_add_rtn_u32 v0, v1, v2 offset:8
ds_cmpst_rtn_b32 v0, v1, v2, v3
ds_wrxchg2_rtn_b32 v[0:1], v1, v2, v3 offset0:3 offset1:4
ds_write2_b64 v1, v[2:3], v[4:5] offset0:7
ds_read_u16 v5, v6
ds_permute_b32 v0, v1, v2 offset:4
flat_load_dword v0, v[1:2] glc slc
flat_load_dwordx3 v[0:2], v[1:2]
flat_atomic_cmpswap_x2 v[0:1], v[2:3], v[4:7] glc
flat_atomic_inc v[2:3], v4 slc
flat_store_short v[0:1], v2 glc
image_sample_lz v[0:3], v0, s[8:15], s[16:19] dmask:0xf unorm glc da
image_load v[0:1], v[2:5], s[8:15] dmask:0x3 slc
image_sample v0, v[1:2], s[4:11], s[16:19] dmask:0x1 lwe
";

#[test]
fn decoding_reproduces_the_llvm_disassembly_of_every_field_the_encodings_carry() {
    let directory = std::env::temp_dir().join(format!("gcn3-fields-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("fields.s");
    let object = directory.join("fields.o");
    std::fs::write(&source, FIELDS).unwrap();
    let status = std::process::Command::new(tool("llvm-mc"))
        .args(["-arch=amdgcn", "-mcpu=gfx803", "-filetype=obj"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("llvm-mc runs");
    assert!(status.success());
    let (count, mismatches) = reproduce(&[object.to_str().unwrap()]);
    std::fs::remove_dir_all(&directory).unwrap();
    assert_eq!(count, FIELDS.lines().count());
    assert!(mismatches.is_empty(), "{} of {count} differ:\n{}", mismatches.len(), mismatches.join("\n"));
}


fn assemble(text: &str) -> Vec<u8> {
    let directory = std::env::temp_dir().join(format!("gcn3-unit-{}-{}", std::process::id(), text.len()));
    std::fs::create_dir_all(&directory).unwrap();
    let source = directory.join("unit.s");
    let object = directory.join("unit.o");
    std::fs::write(&source, text).unwrap();
    let status = std::process::Command::new(tool("llvm-mc"))
        .args(["-arch=amdgcn", "-mcpu=gfx803", "-filetype=obj"])
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("llvm-mc runs");
    assert!(status.success());
    let memory = image(object.to_str().unwrap());
    std::fs::remove_dir_all(&directory).unwrap();
    memory
}

const LOOP: &str = "\
s_mov_b32 s0, 0
v_mov_b32 v1, 0
loop:
v_add_u32 v1, vcc, 1, v1
s_add_u32 s0, s0, 1
s_cmp_lt_u32 s0, 10
s_cbranch_scc1 loop
v_cmp_gt_u32 vcc, 5, v1
s_and_saveexec_b64 s[2:3], vcc
v_mov_b32 v2, 1
s_or_b64 exec, exec, s[2:3]
s_barrier
s_endpgm
";

#[test]
fn discovery_splits_blocks_at_branches_targets_exec_writes_and_barriers() {
    let memory = assemble(LOOP);
    let graph = super::graph::discover(&[0], &memory).unwrap();
    let starts: Vec<usize> = graph.blocks.keys().copied().collect();
    assert_eq!(starts, vec![0x0, 0x8, 0x18, 0x20, 0x28, 0x2c]);
    let next = |start: usize| graph.blocks[&start].next.clone();
    assert_eq!(next(0x0), vec![0x8]);
    assert_eq!(next(0x8), vec![0x18, 0x8]);
    assert_eq!(next(0x18), vec![0x20]);
    assert_eq!(next(0x20), vec![0x28]);
    assert_eq!(next(0x28), vec![0x2c]);
    assert_eq!(next(0x2c), Vec::<usize>::new());
    assert!(matches!(graph.blocks[&0x28].insts.last().unwrap().1.op, I::S_BARRIER));
}

#[test]
fn an_exec_write_before_an_scc_branch_stays_in_its_block() {
    let memory = assemble("v_cmp_eq_u32 vcc, 0, v0\ns_and_b64 exec, exec, vcc\ns_nop 0\ns_cbranch_scc0 done\nv_mov_b32 v1, 0\ndone:\ns_endpgm\n");
    let graph = super::graph::discover(&[0], &memory).unwrap();
    assert_eq!(graph.blocks.keys().copied().collect::<Vec<_>>(), vec![0x0, 0x10, 0x14]);
}

const SCALARS: &str = "\
s_mov_b32 m0, -1
s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 0
s_add_u32 s0, s0, s9
s_addc_u32 s1, s1, 0
s_cbranch_scc1 skip
s_mov_b32 m0, 0x10000
skip:
s_mov_b32 s5, 3
s_setreg_b32 hwreg(HW_REG_MODE, 4, 2), s5
s_getreg_b32 s6, hwreg(HW_REG_MODE, 4, 2)
s_mov_b64 s[10:11], -2
s_movk_i32 s12, 0x8000
v_readfirstlane_b32 s13, v0
s_setreg_b32 hwreg(HW_REG_MODE, 6, 2), s13
s_endpgm
";

#[test]
fn the_scalar_state_follows_m0_the_mode_and_the_descriptor() {
    use super::state::{State, Value};
    let memory = assemble(SCALARS);
    let graph = super::graph::discover(&[0], &memory).unwrap();
    let traced = super::trace::trace(&graph, State::entry(0x3f0, vec![(9, 0)])).unwrap();
    let states: std::collections::BTreeMap<usize, State> = traced
        .states
        .iter()
        .flat_map(|(&(_, start), states)| graph.blocks[&start].insts.iter().map(|(pc, _)| *pc).zip(states.iter().cloned()))
        .collect();
    let pcs: Vec<usize> = graph.blocks.values().flat_map(|b| b.insts.iter().map(|(pc, _)| *pc)).collect();
    let at = |k: usize| &states[&pcs[k]];
    assert_eq!(at(0).sgprs[124], Value::Unknown);
    assert_eq!(at(1).sgprs[124], Value::Known(0xffff_ffff));
    assert_eq!(at(1).mode.field(4, 2), Some(3));
    assert_eq!(at(2).mode.field(4, 2), Some(0));
    assert_eq!(at(3).sgprs[0], Value::Entry(0));
    assert_eq!(at(3).scc, Value::Known(0));
    assert_eq!(at(4).sgprs[1], Value::Entry(1));
    assert_eq!(at(5).sgprs[124], Value::Known(0xffff_ffff));
    let skip = pcs[6];
    assert_eq!(states[&skip].sgprs[124], Value::Unknown, "the paths into skip disagree on M0");
    assert_eq!(states[&skip].sgprs[0], Value::Entry(0));
    assert_eq!(at(7).mode.field(4, 2), Some(0));
    assert_eq!(at(8).mode.field(4, 2), Some(3));
    assert_eq!(at(9).sgprs[6], Value::Known(3));
    assert_eq!((at(10).sgprs[10], at(10).sgprs[11]), (Value::Known(0xffff_fffe), Value::Known(0xffff_ffff)));
    assert_eq!(at(11).sgprs[12], Value::Known(0xffff_8000));
    assert_eq!(at(12).sgprs[13], Value::Unknown);
    assert_eq!(at(13).mode.field(6, 2), None);
    assert_eq!(at(13).mode.field(4, 2), Some(3));
}

const CALLS: &str = "\
s_getpc_b64 s[4:5]
first:
s_add_u32 s4, s4, callee-first
s_addc_u32 s5, s5, 0
s_swappc_b64 s[30:31], s[4:5]
s_getpc_b64 s[4:5]
second:
s_add_u32 s4, s4, callee-second
s_addc_u32 s5, s5, 0
s_swappc_b64 s[30:31], s[4:5]
s_endpgm
callee:
v_add_f32 v0, 1.0, v0
";

#[test]
fn each_call_site_traces_its_own_copy_of_the_callee_under_distinct_dense_ids() {
    use super::state::State;
    let memory = assemble(&format!("{CALLS}s_setpc_b64 s[30:31]\n"));
    let (graph, traced) = super::trace::explore(0, &memory, State::entry(0x3f0, vec![])).unwrap();
    assert_eq!(graph.blocks.keys().copied().collect::<Vec<_>>(), vec![0x0, 0x14, 0x28, 0x2c]);
    assert_eq!(graph.span(), 0x34);
    let contexts: Vec<_> = traced.contexts.iter().map(|c| (c.parent, c.site, c.entry, c.ret)).collect();
    assert_eq!(
        contexts,
        vec![(None, usize::MAX, 0, usize::MAX), (Some(0), 0x10, 0x2c, 0x14), (Some(0), 0x24, 0x2c, 0x28)]
    );
    assert_eq!(traced.calls.iter().map(|(&k, &v)| (k, v)).collect::<Vec<_>>(), vec![((0, 0x0), 1), ((0, 0x14), 2)]);
    let traced_blocks: Vec<_> = traced.states.keys().copied().collect();
    assert_eq!(traced_blocks, vec![(0, 0x0), (0, 0x14), (0, 0x28), (1, 0x2c), (2, 0x2c)]);
    let ids: Vec<usize> = traced_blocks.iter().map(|&(c, pc)| super::trace::id(graph.span(), c, pc)).collect();
    assert_eq!(ids, vec![0x0, 0x14, 0x28, 0x34 + 0x2c, 2 * 0x34 + 0x2c]);
}

#[test]
fn calls_that_recurse_return_elsewhere_or_go_nowhere_known_are_refused() {
    use super::state::State;
    let explore = |text: String| super::trace::explore(0, &assemble(&text), State::entry(0x3f0, vec![])).err().unwrap();
    let recursive = explore(format!("{CALLS}s_getpc_b64 s[6:7]\nagain:\ns_add_u32 s6, s6, callee-again\ns_addc_u32 s7, s7, -1\ns_swappc_b64 s[30:31], s[6:7]\ns_endpgm\n"));
    assert_eq!(recursive, "0x40: a recursive call to 0x2c");
    let elsewhere = explore(format!("{CALLS}s_add_u32 s30, s30, 4\ns_setpc_b64 s[30:31]\n"));
    assert_eq!(elsewhere, "0x34: a function that may return somewhere other than 0x14");
    let unknown = explore(format!("{CALLS}v_readfirstlane_b32 s30, v0\ns_setpc_b64 s[30:31]\n"));
    assert_eq!(unknown, "0x34: a function that may return somewhere other than 0x14");
    let jump = explore("s_getpc_b64 s[4:5]\ns_setpc_b64 s[4:5]\n".to_string());
    assert_eq!(jump, "0x4: a kernel that jumps through a register");
    let nowhere = explore("v_readfirstlane_b32 s4, v0\ns_mov_b32 s5, 0\ns_swappc_b64 s[30:31], s[4:5]\ns_endpgm\n".to_string());
    assert_eq!(nowhere, "0x8: a call to an address the program does not determine");
}

const LANES: &str = "\
s_mov_b32 s4, 7
v_writelane_b32 v24, s4, 13
s_mov_b32 s5, 9
v_writelane_b32 v24, s5, 14
v_readlane_b32 s6, v24, 13
v_readlane_b32 s7, v24, 14
v_mov_b32 v23, 0
v_readlane_b32 s8, v24, 13
v_add_f64 v[22:23], v[0:1], v[2:3]
v_readlane_b32 s9, v24, 13
v_lshlrev_b64 v[23:24], 1, v[0:1]
v_readlane_b32 s10, v24, 13
v_writelane_b32 v24, s4, 13
v_readfirstlane_b32 s11, v0
s_mov_b32 m0, s11
v_writelane_b32 v24, s4, m0
v_readlane_b32 s12, v24, 13
v_writelane_b32 v25, s4, 3
flat_load_dwordx2 v[24:25], v[0:1]
v_readlane_b32 s13, v25, 3
s_mov_b32 m0, 67
v_writelane_b32 v26, s4, m0
v_readlane_b32 s14, v26, 3
s_endpgm
";

#[test]
fn lanes_written_by_writelane_read_back_until_a_write_reaches_their_register() {
    use super::state::{State, Value};
    let memory = assemble(LANES);
    let (graph, traced) = super::trace::explore(0, &memory, State::entry(0x3f0, vec![])).unwrap();
    let block = graph.blocks.keys().copied().next().unwrap();
    let last = traced.states[&(0, block)].last().unwrap();
    let read: Vec<Value> = (6..=14).map(|r| last.sgprs[r]).collect();
    assert_eq!(
        read,
        vec![
            Value::Known(7),
            Value::Known(9),
            Value::Known(7),
            Value::Known(7),
            Value::Unknown,
            Value::Unknown,
            Value::Unknown,
            Value::Unknown,
            Value::Known(7),
        ]
    );
    assert_eq!(last.lanes.keys().copied().collect::<Vec<_>>(), vec![(26, 3)]);
}

#[test]
fn paths_that_disagree_on_a_lane_forget_it_where_they_join() {
    use super::state::{State, Value};
    let memory = assemble(
        "s_mov_b32 s4, 7
s_mov_b32 s5, 9
v_writelane_b32 v30, s4, 1
v_writelane_b32 v30, s4, 2
s_cmp_eq_u32 s6, 0
s_cbranch_scc1 other
v_writelane_b32 v30, s5, 1
s_branch join
other:
v_writelane_b32 v30, s4, 1
join:
v_readlane_b32 s7, v30, 1
v_readlane_b32 s8, v30, 2
s_endpgm
",
    );
    let (graph, traced) = super::trace::explore(0, &memory, State::entry(0x3f0, vec![])).unwrap();
    let join = *graph.blocks.keys().last().unwrap();
    let last = traced.states[&(0, join)].last().unwrap();
    assert_eq!((last.sgprs[7], last.sgprs[8]), (Value::Unknown, Value::Known(7)));
}

#[test]
fn a_call_finds_a_target_restored_from_a_lane() {
    use super::state::State;
    let memory = assemble(
        "s_getpc_b64 s[4:5]
first:
s_add_u32 s4, s4, callee-first
s_addc_u32 s5, s5, 0
v_writelane_b32 v24, s4, 13
v_writelane_b32 v24, s5, 14
v_mov_b32 v23, 0
v_readlane_b32 s4, v24, 13
v_readlane_b32 s5, v24, 14
s_swappc_b64 s[30:31], s[4:5]
s_endpgm
callee:
s_setpc_b64 s[30:31]
",
    );
    let (_, traced) = super::trace::explore(0, &memory, State::entry(0x3f0, vec![])).unwrap();
    assert_eq!(traced.contexts.len(), 2);
    assert_eq!(traced.contexts[1].entry, 0x3c);
}

#[test]
fn vector_writes_name_every_register_an_instruction_may_define() {
    use super::state::vector_writes;
    let memory = assemble(
        "v_mov_b32 v3, v1
v_add_f64 v[4:5], v[0:1], v[2:3]
v_cvt_f32_f64 v6, v[0:1]
v_cvt_f64_f32 v[7:8], v0
v_mad_u64_u32 v[9:10], s[0:1], v0, v1, v[2:3]
v_lshlrev_b64 v[11:12], 1, v[0:1]
v_cmp_gt_u32 vcc, 5, v1
v_cmp_gt_u32_e64 s[4:5], 5, v1
v_readlane_b32 s7, v1, 3
v_readfirstlane_b32 s8, v1
v_writelane_b32 v13, s7, 3
v_add_u32_e64 v14, s[2:3], v1, v2
flat_load_dwordx4 v[15:18], v[0:1]
flat_store_dwordx2 v[0:1], v[2:3]
flat_atomic_cmpswap v19, v[0:1], v[2:3] glc
buffer_load_dword v20, off, s[0:3], 0 offset:4
buffer_store_dword v20, off, s[0:3], 0 offset:4
ds_read2_b64 v[21:24], v0 offset1:1
ds_write_b32 v0, v1
ds_add_rtn_u32 v25, v0, v1
ds_bpermute_b32 v26, v0, v1
image_sample_lz v[27:30], v0, s[8:15], s[16:19] dmask:0xf
v_movrels_b32 v31, v1
s_mov_b32 s9, s10
s_endpgm
",
    );
    let mut pc = 0;
    let mut found = vec![];
    loop {
        let inst = decode(&memory, pc).unwrap();
        if matches!(inst.op, I::S_ENDPGM) {
            break;
        }
        found.push(vector_writes(&inst));
        pc += inst.size;
    }
    let span = |first: u16, count: u16| Some((first..first + count).collect::<Vec<u16>>());
    assert_eq!(
        found,
        vec![
            span(3, 1),
            span(4, 2),
            span(6, 1),
            span(7, 2),
            span(9, 2),
            span(11, 2),
            span(0, 0),
            span(0, 0),
            span(0, 0),
            span(0, 0),
            span(13, 1),
            span(14, 1),
            span(15, 4),
            span(0, 0),
            span(19, 1),
            span(20, 1),
            span(0, 0),
            span(21, 4),
            span(0, 0),
            span(25, 1),
            span(26, 1),
            span(27, 4),
            None,
            span(0, 0),
        ]
    );
}

#[test]
fn scalar_writes_name_every_register_an_instruction_defines() {
    use super::state::{scalar_writes, writes_exec, writes_scc};
    let memory = assemble(
        "v_cmp_gt_u32 vcc, 5, v1
v_cmpx_gt_u32 vcc, 5, v1
v_cmp_gt_u32_e64 s[4:5], 5, v1
s_and_saveexec_b64 s[2:3], vcc
s_load_dwordx4 s[8:11], s[4:5], 0x0
v_readlane_b32 s7, v1, 3
v_add_u32_e64 v1, s[12:13], v1, v2
v_add_u32 v1, vcc, v1, v2
s_bcnt1_i32_b64 s14, s[2:3]
s_mov_b32 s15, s16
s_cselect_b64 s[16:17], s[2:3], 0
s_endpgm
",
    );
    let mut pc = 0;
    let mut found = vec![];
    loop {
        let inst = decode(&memory, pc).unwrap();
        if matches!(inst.op, I::S_ENDPGM) {
            break;
        }
        found.push((scalar_writes(&inst), writes_exec(&inst), writes_scc(&inst)));
        pc += inst.size;
    }
    assert_eq!(
        found,
        vec![
            (vec![106, 107], false, false),
            (vec![106, 107, 126, 127], true, false),
            (vec![4, 5], false, false),
            (vec![2, 3, 126, 127], true, true),
            (vec![8, 9, 10, 11], false, false),
            (vec![7], false, false),
            (vec![12, 13], false, false),
            (vec![106, 107], false, false),
            (vec![14], false, true),
            (vec![15], false, false),
            (vec![16, 17], false, false),
        ]
    );
}

#[test]
fn operands_map_to_their_rdna_counterparts() {
    use super::lower::{scalar_dst, source};
    use crate::rdna_instructions::SourceOperand as S;
    let shown = |code: u16| format!("{:?}", source(code, Some(0x1234)).unwrap());
    assert_eq!(shown(5), format!("{:?}", S::ScalarRegister(5)));
    assert_eq!(shown(124), format!("{:?}", S::ScalarRegister(125)));
    assert_eq!(shown(106), format!("{:?}", S::ScalarRegister(106)));
    assert_eq!(shown(126), format!("{:?}", S::ScalarRegister(126)));
    assert_eq!(shown(192), format!("{:?}", S::IntegerConstant(64)));
    assert_eq!(shown(193), format!("{:?}", S::IntegerConstant(u64::MAX)));
    assert_eq!(shown(208), format!("{:?}", S::IntegerConstant(-16i64 as u64)));
    assert_eq!(shown(242), format!("{:?}", S::FloatConstant(1.0)));
    assert_eq!(shown(255), format!("{:?}", S::LiteralConstant(0x1234)));
    assert_eq!(shown(300), format!("{:?}", S::VectorRegister(44)));
    match source(248, None).unwrap() {
        S::FloatConstant(v) => {
            assert_eq!(v.to_bits(), 0x3fc4_5f30_6dc9_c882);
            assert_eq!((v as f32).to_bits(), 0x3e22_f983);
        }
        other => panic!("{:?}", other),
    }
    for refused in [104u16, 105, 108, 123, 125, 251, 252, 253, 254] {
        assert!(source(refused, None).is_err(), "operand {}", refused);
    }
    assert_eq!(scalar_dst(124).unwrap(), 125);
    assert_eq!(scalar_dst(103).unwrap(), 103);
    assert!(scalar_dst(112).is_err());
}

#[test]
fn flushing_wraps_only_the_operations_that_honor_the_mode() {
    use super::flush::flushed;
    use crate::rdna_instructions::{InstFormat, SourceOperand as S, VOP3};
    use crate::rdna_spmd::rdna4::lift::{instruction_with_registry, Lowering};
    let registry = crate::rdna_spmd::rdna4::dialect().registry;
    let lowering = |op| {
        instruction_with_registry(
            &InstFormat::VOP3(VOP3 {
                vdst: 0,
                abs: 0,
                opsel: 0,
                cm: 0,
                op,
                src0: S::VectorRegister(1),
                src1: S::VectorRegister(2),
                src2: S::VectorRegister(3),
                omod: 0,
                neg: 0,
            }),
            &registry,
            64,
        )
    };
    let size = |l: &Lowering| match l {
        Lowering::TypedAlu { expr, .. } => expr.expr().insts.len(),
        _ => unreachable!(),
    };
    let add = lowering(I::V_ADD_F32);
    let before = size(&add);
    assert_eq!(size(&flushed(lowering(I::V_ADD_F32), &registry, true, true)), before + 27);
    assert_eq!(size(&flushed(lowering(I::V_ADD_F32), &registry, true, false)), before + 18);
    assert_eq!(size(&flushed(lowering(I::V_ADD_F32), &registry, false, true)), before + 9);
    assert_eq!(size(&flushed(lowering(I::V_ADD_F32), &registry, false, false)), before);
    let max = lowering(I::V_MAX_F32);
    assert_eq!(size(&flushed(lowering(I::V_MAX_F32), &registry, true, true)), size(&max));
    let fma = lowering(I::V_FMA_F32);
    assert_eq!(size(&flushed(lowering(I::V_FMA_F32), &registry, true, true)), size(&fma) + 36);
    let floor = lowering(I::V_FLOOR_F32);
    assert_eq!(size(&flushed(lowering(I::V_FLOOR_F32), &registry, true, true)), size(&floor) + 18);
    let reciprocal = lowering(I::V_RCP_F32);
    assert_eq!(size(&flushed(lowering(I::V_RCP_F32), &registry, true, true)), size(&reciprocal));
}
