use crate::probe::Probe;
use crate::vector::{lanes, word};

const SELECTS: [&str; 7] = ["BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD"];

fn select(value: u32, sel: usize, sext: bool) -> u32 {
    let (shift, bits) = match sel {
        0..=3 => (8 * sel as u32, 8),
        4 | 5 => (16 * (sel as u32 - 4), 16),
        _ => return value,
    };
    let field = value >> shift << (32 - bits);
    if sext {
        ((field as i32) >> (32 - bits)) as u32
    } else {
        field >> (32 - bits)
    }
}

fn place(value: u32, old: u32, sel: usize, unused: usize) -> u32 {
    let (shift, bits) = match sel {
        0..=3 => (8 * sel as u32, 8),
        4 | 5 => (16 * (sel as u32 - 4), 16),
        _ => return value,
    };
    let mask = (((1u64 << bits) - 1) as u32) << shift;
    let field = value << (32 - bits);
    match unused {
        0 => (field >> (32 - bits)) << shift,
        1 => (((field as i32) >> (32 - bits)) << shift) as u32,
        _ => old & !mask | (field >> (32 - bits)) << shift,
    }
}

#[test]
fn sources_are_selected_and_extended() {
    let inputs = lanes(3, word);
    for (s0, name0) in SELECTS.iter().enumerate() {
        for (s1, name1) in SELECTS.iter().enumerate() {
            Probe::new(3, 1, &format!("v_or_b32_sdwa v20, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:{name0} src1_sel:{name1}"))
                .check(&inputs, |_, x| vec![select(x[0], s0, false) | select(x[1], s1, false)]);
        }
        Probe::new(3, 1, &format!("v_add_u32_sdwa v20, vcc, sext(v1), v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:{name0} src1_sel:DWORD"))
            .check(&inputs, |_, x| vec![select(x[0], s0, true).wrapping_add(x[1])]);
    }
}

#[test]
fn destinations_pad_extend_or_preserve() {
    let inputs = lanes(3, word);
    for (sel, name) in SELECTS.iter().enumerate() {
        for (unused, kind) in ["UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE"].iter().enumerate() {
            Probe::new(3, 1, &format!("v_mov_b32 v20, v3\nv_xor_b32_sdwa v20, v1, v2 dst_sel:{name} dst_unused:{kind} src0_sel:DWORD src1_sel:DWORD"))
                .check(&inputs, |_, x| vec![place(x[0] ^ x[1], x[2], sel, unused)]);
        }
    }
}

#[test]
fn other_formats_and_widths_compose() {
    let inputs = lanes(3, word);
    Probe::new(3, 1, "v_lshlrev_b16_sdwa v20, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_1")
        .check(&inputs, |_, x| vec![((select(x[1], 1, false) << (x[0] & 15)) & 0xffff)]);
    Probe::new(3, 1, "v_mul_u32_u24_sdwa v20, v1, v2 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:WORD_1")
        .check(&inputs, |_, x| vec![place(select(x[0], 0, false) * select(x[1], 5, false), 0, 5, 0)]);
    Probe::new(3, 1, "v_mov_b32_sdwa v20, v1 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:BYTE_3")
        .check(&inputs, |_, x| vec![place(select(x[0], 3, false), 0, 2, 1)]);
    Probe::new(3, 2, "v_add_u32_sdwa v20, vcc, v1, v2 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0\nv_cndmask_b32 v21, 0, 1, vcc")
        .check(&inputs, |_, x| {
            let sum = select(x[0], 5, false) as u64 + select(x[1], 4, false) as u64;
            vec![sum as u32, (sum >> 32) as u32]
        });
    Probe::new(3, 1, "v_cmp_eq_u32_sdwa vcc, v1, v2 src0_sel:BYTE_1 src1_sel:BYTE_1\nv_cndmask_b32 v20, 0, 1, vcc")
        .check(&inputs, |_, x| vec![(select(x[0], 1, false) == select(x[1], 1, false)) as u32]);
}
