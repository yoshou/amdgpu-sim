use crate::probe::{Probe, LANES};

fn waves(rows: &[Vec<u32>]) -> Vec<u32> {
    rows.iter()
        .flat_map(|row| (0..LANES).flat_map(move |_| row.clone()))
        .collect()
}

const PAIRS: [(u32, u32); 10] = [
    (0, 0),
    (1, 0xffff_ffff),
    (0x8000_0000, 0x8000_0000),
    (0x7fff_ffff, 1),
    (5, 9),
    (0xdead_beef, 0x1234_5678),
    (0xffff_fffe, 0xffff_fffe),
    (17, 3),
    (0x0001_0010, 0x0005_0004),
    (0x8000, 0x21),
];

fn pairs() -> Vec<u32> {
    waves(&PAIRS.iter().map(|&(a, b)| vec![a, b]).collect::<Vec<_>>())
}

fn binary(op: &str) -> Probe {
    Probe::new(
        2,
        2,
        &format!(
            "v_readfirstlane_b32 s12, v1
             v_readfirstlane_b32 s13, v2
             s_cmp_eq_u32 s12, s13
             {op} s14, s12, s13
             s_cselect_b32 s15, 1, 0
             v_mov_b32 v20, s14
             v_mov_b32 v21, s15"
        ),
    )
}

fn unary(op: &str) -> Probe {
    Probe::new(
        2,
        2,
        &format!(
            "v_readfirstlane_b32 s12, v1
             v_readfirstlane_b32 s14, v2
             s_cmp_eq_u32 s12, s14
             {op} s14, s12
             s_cselect_b32 s15, 1, 0
             v_mov_b32 v20, s14
             v_mov_b32 v21, s15"
        ),
    )
}

fn scc(a: u32, b: u32) -> u32 {
    (a == b) as u32
}

#[test]
fn add_and_subtract_set_the_carry_and_borrow() {
    let inputs = pairs();
    binary("s_add_u32").check(&inputs, |_, x| {
        let sum = x[0] as u64 + x[1] as u64;
        vec![sum as u32, (sum >> 32) as u32]
    });
    binary("s_sub_u32").check(&inputs, |_, x| vec![x[0].wrapping_sub(x[1]), (x[1] > x[0]) as u32]);
    binary("s_add_i32").check(&inputs, |_, x| {
        let (r, o) = (x[0] as i32).overflowing_add(x[1] as i32);
        vec![r as u32, o as u32]
    });
    binary("s_sub_i32").check(&inputs, |_, x| {
        let (r, o) = (x[0] as i32).overflowing_sub(x[1] as i32);
        vec![r as u32, o as u32]
    });
}

#[test]
fn add_and_subtract_with_carry_take_the_carry_from_scc() {
    let inputs = pairs();
    binary("s_addc_u32").check(&inputs, |_, x| {
        let sum = x[0] as u64 + x[1] as u64 + scc(x[0], x[1]) as u64;
        vec![sum as u32, (sum >> 32) as u32]
    });
    binary("s_subb_u32").check(&inputs, |_, x| {
        let borrow = scc(x[0], x[1]) as u64;
        let difference = (x[0] as u64).wrapping_sub(x[1] as u64 + borrow);
        vec![difference as u32, ((x[1] as u64 + borrow) > x[0] as u64) as u32]
    });
}

#[test]
fn logic_sets_scc_when_the_result_is_nonzero() {
    let inputs = pairs();
    let ops: [(&str, fn(u32, u32) -> u32); 8] = [
        ("s_and_b32", |a, b| a & b),
        ("s_or_b32", |a, b| a | b),
        ("s_xor_b32", |a, b| a ^ b),
        ("s_andn2_b32", |a, b| a & !b),
        ("s_orn2_b32", |a, b| a | !b),
        ("s_nand_b32", |a, b| !(a & b)),
        ("s_nor_b32", |a, b| !(a | b)),
        ("s_xnor_b32", |a, b| !(a ^ b)),
    ];
    for (op, f) in ops {
        binary(op).check(&inputs, |_, x| {
            let r = f(x[0], x[1]);
            vec![r, (r != 0) as u32]
        });
    }
}

#[test]
fn shifts_take_the_low_five_bits_of_the_amount() {
    let inputs = pairs();
    binary("s_lshl_b32").check(&inputs, |_, x| {
        let r = x[0] << (x[1] & 31);
        vec![r, (r != 0) as u32]
    });
    binary("s_lshr_b32").check(&inputs, |_, x| {
        let r = x[0] >> (x[1] & 31);
        vec![r, (r != 0) as u32]
    });
    binary("s_ashr_i32").check(&inputs, |_, x| {
        let r = ((x[0] as i32) >> (x[1] & 31)) as u32;
        vec![r, (r != 0) as u32]
    });
}

#[test]
fn bit_fields_multiplies_maxima_and_selects() {
    let inputs = pairs();
    binary("s_bfm_b32").check(&inputs, |_, x| {
        let mask = ((1u64 << (x[0] & 31)) - 1) as u32;
        vec![mask << (x[1] & 31), scc(x[0], x[1])]
    });
    binary("s_bfe_u32").check(&inputs, |_, x| {
        let width = (x[1] >> 16) & 0x7f;
        let mask = if width >= 32 { u32::MAX } else { (1u32 << width) - 1 };
        let r = (x[0] >> (x[1] & 31)) & mask;
        vec![r, (r != 0) as u32]
    });
    binary("s_mul_i32").check(&inputs, |_, x| vec![x[0].wrapping_mul(x[1]), scc(x[0], x[1])]);
    binary("s_max_u32").check(&inputs, |_, x| vec![x[0].max(x[1]), (x[0] > x[1]) as u32]);
    binary("s_cselect_b32").check(&inputs, |_, x| {
        vec![if x[0] == x[1] { x[0] } else { x[1] }, scc(x[0], x[1])]
    });
}

#[test]
fn single_operand_operations() {
    let inputs = pairs();
    unary("s_mov_b32").check(&inputs, |_, x| vec![x[0], scc(x[0], x[1])]);
    unary("s_cmov_b32").check(&inputs, |_, x| vec![if x[0] == x[1] { x[0] } else { x[1] }, scc(x[0], x[1])]);
    unary("s_not_b32").check(&inputs, |_, x| vec![!x[0], (!x[0] != 0) as u32]);
    unary("s_brev_b32").check(&inputs, |_, x| vec![x[0].reverse_bits(), scc(x[0], x[1])]);
    unary("s_bcnt0_i32_b32").check(&inputs, |_, x| vec![x[0].count_zeros(), (x[0].count_zeros() != 0) as u32]);
    unary("s_bcnt1_i32_b32").check(&inputs, |_, x| vec![x[0].count_ones(), (x[0].count_ones() != 0) as u32]);
    unary("s_ff1_i32_b32").check(&inputs, |_, x| {
        vec![if x[0] == 0 { u32::MAX } else { x[0].trailing_zeros() }, scc(x[0], x[1])]
    });
    unary("s_sext_i32_i16").check(&inputs, |_, x| vec![x[0] as i16 as i32 as u32, scc(x[0], x[1])]);
}

fn wide(op: &str, source: &str) -> Probe {
    Probe::new(
        4,
        3,
        &format!(
            "v_readfirstlane_b32 s12, v1
             v_readfirstlane_b32 s13, v2
             v_readfirstlane_b32 s14, v3
             v_readfirstlane_b32 s15, v4
             s_cmp_eq_u32 s12, s14
             {op} s[16:17], s[12:13], {source}
             s_cselect_b32 s18, 1, 0
             v_mov_b32 v20, s16
             v_mov_b32 v21, s17
             v_mov_b32 v22, s18"
        ),
    )
}

fn quads() -> Vec<u32> {
    let rows: Vec<Vec<u32>> = PAIRS
        .iter()
        .zip(PAIRS.iter().rev())
        .map(|(&(a, b), &(c, d))| vec![a, b, c, d])
        .chain([vec![0, 0, 0, 0], vec![0, 1, 0, 1], vec![u32::MAX, u32::MAX, 63, 0]])
        .collect();
    waves(&rows)
}

fn split(v: u64) -> [u32; 2] {
    [v as u32, (v >> 32) as u32]
}

#[test]
fn sixty_four_bit_logic_and_shifts() {
    let inputs = quads();
    let pair = |x: &[u32], k: usize| x[k] as u64 | (x[k + 1] as u64) << 32;
    let ops: [(&str, fn(u64, u64) -> u64); 5] = [
        ("s_and_b64", |a, b| a & b),
        ("s_or_b64", |a, b| a | b),
        ("s_xor_b64", |a, b| a ^ b),
        ("s_andn2_b64", |a, b| a & !b),
        ("s_orn2_b64", |a, b| a | !b),
    ];
    for (op, f) in ops {
        wide(op, "s[14:15]").check(&inputs, |_, x| {
            let r = f(pair(x, 0), pair(x, 2));
            let [lo, hi] = split(r);
            vec![lo, hi, (r != 0) as u32]
        });
    }
    let shifts: [(&str, fn(u64, u32) -> u64); 3] = [
        ("s_lshl_b64", |a, n| a << (n & 63)),
        ("s_lshr_b64", |a, n| a >> (n & 63)),
        ("s_ashr_i64", |a, n| ((a as i64) >> (n & 63)) as u64),
    ];
    for (op, f) in shifts {
        wide(op, "s14").check(&inputs, |_, x| {
            let r = f(pair(x, 0), x[2]);
            let [lo, hi] = split(r);
            vec![lo, hi, (r != 0) as u32]
        });
    }
    wide("s_cselect_b64", "s[14:15]").check(&inputs, |_, x| {
        let r = if x[0] == x[2] { pair(x, 0) } else { pair(x, 2) };
        let [lo, hi] = split(r);
        vec![lo, hi, scc(x[0], x[2])]
    });
}

fn wide_unary(op: &str, words: usize) -> Probe {
    let store = if words == 2 { "v_mov_b32 v21, s17" } else { "v_mov_b32 v21, 0" };
    let dst = if words == 2 { "s[16:17]" } else { "s16" };
    Probe::new(
        4,
        3,
        &format!(
            "v_readfirstlane_b32 s12, v1
             v_readfirstlane_b32 s13, v2
             v_readfirstlane_b32 s14, v3
             v_readfirstlane_b32 s15, v4
             s_mov_b64 s[16:17], s[14:15]
             s_cmp_eq_u32 s12, s14
             {op} {dst}, s[12:13]
             s_cselect_b32 s18, 1, 0
             v_mov_b32 v20, s16
             {store}
             v_mov_b32 v22, s18"
        ),
    )
}

#[test]
fn sixty_four_bit_single_operand_operations() {
    let inputs = quads();
    let pair = |x: &[u32], k: usize| x[k] as u64 | (x[k + 1] as u64) << 32;
    wide_unary("s_mov_b64", 2).check(&inputs, |_, x| vec![x[0], x[1], scc(x[0], x[2])]);
    wide_unary("s_cmov_b64", 2).check(&inputs, |_, x| {
        let [lo, hi] = split(if x[0] == x[2] { pair(x, 0) } else { pair(x, 2) });
        vec![lo, hi, scc(x[0], x[2])]
    });
    wide_unary("s_not_b64", 2).check(&inputs, |_, x| {
        let r = !pair(x, 0);
        let [lo, hi] = split(r);
        vec![lo, hi, (r != 0) as u32]
    });
    wide_unary("s_bcnt1_i32_b64", 1).check(&inputs, |_, x| {
        let r = pair(x, 0).count_ones();
        vec![r, 0, (r != 0) as u32]
    });
    wide_unary("s_ff1_i32_b64", 1).check(&inputs, |_, x| {
        let v = pair(x, 0);
        vec![if v == 0 { u32::MAX } else { v.trailing_zeros() }, 0, scc(x[0], x[2])]
    });
}

fn compare(op: &str, wide: bool) -> Probe {
    let operands = if wide { "s[12:13], s[14:15]" } else { "s12, s14" };
    Probe::new(
        4,
        1,
        &format!(
            "v_readfirstlane_b32 s12, v1
             v_readfirstlane_b32 s13, v2
             v_readfirstlane_b32 s14, v3
             v_readfirstlane_b32 s15, v4
             {op} {operands}
             s_cselect_b32 s16, 1, 0
             v_mov_b32 v20, s16"
        ),
    )
}

#[test]
fn compares_set_scc() {
    let inputs = quads();
    let ops: [(&str, fn(u32, u32) -> bool); 10] = [
        ("s_cmp_eq_i32", |a, b| a == b),
        ("s_cmp_lg_i32", |a, b| a != b),
        ("s_cmp_gt_i32", |a, b| (a as i32) > b as i32),
        ("s_cmp_ge_i32", |a, b| (a as i32) >= b as i32),
        ("s_cmp_lt_i32", |a, b| (a as i32) < (b as i32)),
        ("s_cmp_le_i32", |a, b| a as i32 <= b as i32),
        ("s_cmp_gt_u32", |a, b| a > b),
        ("s_cmp_ge_u32", |a, b| a >= b),
        ("s_cmp_lt_u32", |a, b| a < b),
        ("s_cmp_le_u32", |a, b| a <= b),
    ];
    for (op, f) in ops {
        compare(op, false).check(&inputs, |_, x| vec![f(x[0], x[2]) as u32]);
    }
    let pair = |x: &[u32], k: usize| x[k] as u64 | (x[k + 1] as u64) << 32;
    compare("s_cmp_eq_u64", true).check(&inputs, |_, x| vec![(pair(x, 0) == pair(x, 2)) as u32]);
    compare("s_cmp_lg_u64", true).check(&inputs, |_, x| vec![(pair(x, 0) != pair(x, 2)) as u32]);
}

fn immediate(op: &str, k: u16) -> Probe {
    Probe::new(
        2,
        2,
        &format!(
            "v_readfirstlane_b32 s12, v1
             v_readfirstlane_b32 s13, v2
             s_cmp_eq_u32 s12, s13
             {op} s12, {k:#x}
             s_cselect_b32 s15, 1, 0
             v_mov_b32 v20, s12
             v_mov_b32 v21, s15"
        ),
    )
}

#[test]
fn immediates_extend_by_signedness() {
    let inputs = waves(&[vec![0x9000, 0], vec![0xffff_8000, 0], vec![5, 5], vec![0x7fff_fff0, 1], vec![0xffff_ffff, 2]]);
    for k in [0x8000u16, 0x7fff, 0xfff0, 0x10] {
        let signed = k as i16 as i32 as u32;
        let unsigned = k as u32;
        immediate("s_movk_i32", k).check(&inputs, |_, x| vec![signed, scc(x[0], x[1])]);
        immediate("s_cmovk_i32", k).check(&inputs, |_, x| vec![if x[0] == x[1] { signed } else { x[0] }, scc(x[0], x[1])]);
        immediate("s_mulk_i32", k).check(&inputs, |_, x| vec![x[0].wrapping_mul(signed), scc(x[0], x[1])]);
        immediate("s_addk_i32", k).check(&inputs, |_, x| {
            let (r, o) = (x[0] as i32).overflowing_add(signed as i32);
            vec![r as u32, o as u32]
        });
        let compares: [(&str, bool, fn(u32, u32, bool) -> bool); 12] = [
            ("s_cmpk_eq_i32", true, |a, b, _| a == b),
            ("s_cmpk_lg_i32", true, |a, b, _| a != b),
            ("s_cmpk_gt_i32", true, |a, b, _| a as i32 > b as i32),
            ("s_cmpk_ge_i32", true, |a, b, _| a as i32 >= b as i32),
            ("s_cmpk_lt_i32", true, |a, b, _| (a as i32) < b as i32),
            ("s_cmpk_le_i32", true, |a, b, _| a as i32 <= b as i32),
            ("s_cmpk_eq_u32", false, |a, b, _| a == b),
            ("s_cmpk_lg_u32", false, |a, b, _| a != b),
            ("s_cmpk_gt_u32", false, |a, b, _| a > b),
            ("s_cmpk_ge_u32", false, |a, b, _| a >= b),
            ("s_cmpk_lt_u32", false, |a, b, _| a < b),
            ("s_cmpk_le_u32", false, |a, b, _| a <= b),
        ];
        for (op, is_signed, f) in compares {
            let value = if is_signed { signed } else { unsigned };
            Probe::new(
                2,
                1,
                &format!(
                    "v_readfirstlane_b32 s12, v1
                     {op} s12, {k:#x}
                     s_cselect_b32 s15, 1, 0
                     v_mov_b32 v20, s15"
                ),
            )
            .check(&inputs, |_, x| vec![f(x[0], value, is_signed) as u32]);
        }
    }
}

fn saveexec(op: &str) -> Probe {
    Probe::new(
        1,
        6,
        &format!(
            "v_mov_b32 v20, 0
             s_mov_b64 s[20:21], exec
             v_cmp_ne_u32 s[22:23], 2, v1
             v_cmp_lt_u32 vcc, 1, v1
             s_mov_b64 exec, s[22:23]
             {op} s[12:13], vcc
             s_cselect_b32 s16, 1, 0
             s_mov_b64 s[14:15], exec
             v_mov_b32 v20, 1
             s_mov_b64 exec, s[20:21]
             v_mov_b32 v21, s12
             v_mov_b32 v22, s13
             v_mov_b32 v23, s14
             v_mov_b32 v24, s15
             v_mov_b32 v25, s16"
        ),
    )
}

#[test]
fn saveexec_saves_exec_and_combines_it_with_the_source() {
    let mask = |wave: &[u32], f: &dyn Fn(u32) -> bool| (0..LANES).fold(0u64, |m, l| m | (f(wave[l]) as u64) << l);
    let ops: [(&str, fn(u64, u64) -> u64); 8] = [
        ("s_and_saveexec_b64", |s, e| s & e),
        ("s_or_saveexec_b64", |s, e| s | e),
        ("s_xor_saveexec_b64", |s, e| s ^ e),
        ("s_andn2_saveexec_b64", |s, e| s & !e),
        ("s_orn2_saveexec_b64", |s, e| s | !e),
        ("s_nand_saveexec_b64", |s, e| !(s & e)),
        ("s_nor_saveexec_b64", |s, e| !(s | e)),
        ("s_xnor_saveexec_b64", |s, e| !(s ^ e)),
    ];
    let first: Vec<u32> = (0..LANES as u32).map(|l| l % 4).collect();
    let second: Vec<u32> = (0..LANES as u32).map(|l| [2, 3, 0, 9, 2, 1][(l % 6) as usize] + (l >= 40) as u32 * 4).collect();
    for wave in [first, second] {
        let exec = mask(&wave, &|v| v != 2);
        let vcc = mask(&wave, &|v| v > 1);
        for (op, f) in ops {
            let next = f(vcc, exec);
            saveexec(op).check(&wave, |lane, _| {
                vec![
                    (next >> lane & 1) as u32,
                    exec as u32,
                    (exec >> 32) as u32,
                    next as u32,
                    (next >> 32) as u32,
                    (next != 0) as u32,
                ]
            });
        }
    }
}

#[test]
fn getreg_reads_the_mode_the_program_sets() {
    let probe = Probe::new(
        0,
        3,
        "s_getreg_b32 s12, hwreg(HW_REG_MODE, 0, 10)
         s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 2), 1
         s_getreg_b32 s13, hwreg(HW_REG_MODE, 0, 10)
         s_mov_b32 s14, 2
         s_setreg_b32 hwreg(HW_REG_MODE, 4, 2), s14
         s_getreg_b32 s14, hwreg(HW_REG_MODE, 4, 4)
         v_mov_b32 v20, s12
         v_mov_b32 v21, s13
         v_mov_b32 v22, s14",
    );
    probe.check(&[], |_, _| vec![0x3f0, 0x3d0, 0xe]);
    probe.flushing().check(&[], |_, _| vec![0x3c0, 0x3d0, 0xe]);
}
