use crate::probe::{Probe, LANES};

pub fn lanes(width: usize, f: impl Fn(usize, usize) -> u32) -> Vec<u32> {
    (0..LANES).flat_map(|l| (0..width).map(move |k| (l, k))).map(|(l, k)| f(l, k)).collect()
}

pub const WORDS: [u32; 16] = [
    0,
    1,
    2,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_ffff,
    0xffff_fffe,
    0x1234_5678,
    0xdead_beef,
    0x0000_ffff,
    0x0001_0000,
    0x00ff_ff00,
    31,
    32,
    0x8000_0001,
    0x7fff_8000,
];

pub fn word(l: usize, k: usize) -> u32 {
    WORDS[(l * (k + 3) + k * 5 + l / 16) % WORDS.len()] ^ if k == 2 { (l as u32) << 3 } else { 0 }
}

fn mask_of(inputs: &[u32], width: usize, f: impl Fn(&[u32]) -> bool) -> u64 {
    (0..LANES).fold(0u64, |m, l| m | (f(&inputs[l * width..(l + 1) * width]) as u64) << l)
}

#[test]
fn carries_flow_through_vcc_and_scalar_pairs() {
    let inputs = lanes(3, word);
    let carry = |a: u32, b: u32, c: u32, sub: bool| -> (u32, u32) {
        let wide = if sub {
            (a as u64).wrapping_sub(b as u64 + c as u64)
        } else {
            a as u64 + b as u64 + c as u64
        };
        (wide as u32, (wide >> 32 != 0) as u32)
    };
    let cases: [(&str, bool, bool, bool); 6] = [
        ("v_add_u32", false, false, false),
        ("v_sub_u32", true, false, false),
        ("v_subrev_u32", true, true, false),
        ("v_addc_u32", false, false, true),
        ("v_subb_u32", true, false, true),
        ("v_subbrev_u32", true, true, true),
    ];
    for (op, sub, reverse, carry_in) in cases {
        let short = if carry_in {
            format!("v_cmp_gt_u32 vcc, v3, v1\n{op}_e32 v20, vcc, v1, v2, vcc\nv_cndmask_b32 v21, 0, 1, vcc")
        } else {
            format!("{op}_e32 v20, vcc, v1, v2\nv_cndmask_b32 v21, 0, 1, vcc")
        };
        let long = if carry_in {
            format!("v_cmp_gt_u32 s[12:13], v3, v1\n{op}_e64 v20, s[14:15], v1, v2, s[12:13]\nv_cndmask_b32 v21, 0, 1, s[14:15]")
        } else {
            format!("{op}_e64 v20, s[14:15], v1, v2\nv_cndmask_b32 v21, 0, 1, s[14:15]")
        };
        for body in [short, long] {
            Probe::new(3, 2, &body).check(&inputs, |_, x| {
                let c = if carry_in { (x[2] > x[0]) as u32 } else { 0 };
                let (a, b) = if reverse { (x[1], x[0]) } else { (x[0], x[1]) };
                let (r, flag) = carry(a, b, c, sub);
                vec![r, flag]
            });
        }
    }
}

#[test]
fn carries_compose_into_sixty_four_bit_arithmetic() {
    let inputs = lanes(4, word);
    Probe::new(
        4,
        2,
        "v_add_u32 v20, vcc, v1, v3
         v_addc_u32 v21, vcc, v2, v4, vcc",
    )
    .check(&inputs, |_, x| {
        let sum = (x[0] as u64 | (x[1] as u64) << 32).wrapping_add(x[2] as u64 | (x[3] as u64) << 32);
        vec![sum as u32, (sum >> 32) as u32]
    });
}

#[test]
fn integer_arithmetic_and_bit_operations() {
    let inputs = lanes(3, word);
    let binary: [(&str, fn(u32, u32) -> u32); 15] = [
        ("v_lshlrev_b32", |a, b| b << (a & 31)),
        ("v_lshrrev_b32", |a, b| b >> (a & 31)),
        ("v_ashrrev_i32", |a, b| ((b as i32) >> (a & 31)) as u32),
        ("v_and_b32", |a, b| a & b),
        ("v_or_b32", |a, b| a | b),
        ("v_xor_b32", |a, b| a ^ b),
        ("v_min_i32", |a, b| (a as i32).min(b as i32) as u32),
        ("v_max_i32", |a, b| (a as i32).max(b as i32) as u32),
        ("v_min_u32", |a, b| a.min(b)),
        ("v_max_u32", |a, b| a.max(b)),
        ("v_mul_u32_u24", |a, b| (a & 0xff_ffff).wrapping_mul(b & 0xff_ffff)),
        ("v_mul_i32_i24", |a, b| (((a << 8) as i32 >> 8).wrapping_mul((b << 8) as i32 >> 8)) as u32),
        ("v_mul_lo_u32", |a, b| a.wrapping_mul(b)),
        ("v_mul_hi_u32", |a, b| ((a as u64 * b as u64) >> 32) as u32),
        ("v_bcnt_u32_b32", |a, b| a.count_ones().wrapping_add(b)),
    ];
    for (op, f) in binary {
        let three = matches!(op, "v_mul_lo_u32" | "v_mul_hi_u32" | "v_bcnt_u32_b32");
        let forms: Vec<String> = if three {
            vec![format!("{op} v20, v1, v2")]
        } else {
            vec![format!("{op}_e32 v20, v1, v2"), format!("{op}_e64 v20, v1, v2")]
        };
        for body in forms {
            Probe::new(3, 1, &body).check(&inputs, |_, x| vec![f(x[0], x[1])]);
        }
    }
    let ternary: [(&str, fn(u32, u32, u32) -> u32); 11] = [
        ("v_mad_u32_u24", |a, b, c| (a & 0xff_ffff).wrapping_mul(b & 0xff_ffff).wrapping_add(c)),
        ("v_mad_i32_i24", |a, b, c| (((a << 8) as i32 >> 8).wrapping_mul((b << 8) as i32 >> 8) as u32).wrapping_add(c)),
        ("v_bfe_u32", |a, b, c| (a >> (b & 31)) & if c & 31 == 0 { 0 } else { u32::MAX >> (32 - (c & 31)) }),
        ("v_bfi_b32", |a, b, c| (a & b) | (!a & c)),
        ("v_alignbit_b32", |a, b, c| (((a as u64) << 32 | b as u64) >> (c & 31)) as u32),
        ("v_min3_i32", |a, b, c| (a as i32).min(b as i32).min(c as i32) as u32),
        ("v_max3_i32", |a, b, c| (a as i32).max(b as i32).max(c as i32) as u32),
        ("v_min3_u32", |a, b, c| a.min(b).min(c)),
        ("v_max3_u32", |a, b, c| a.max(b).max(c)),
        ("v_med3_i32", |a, b, c| {
            let mut v = [a as i32, b as i32, c as i32];
            v.sort();
            v[1] as u32
        }),
        ("v_med3_u32", |a, b, c| {
            let mut v = [a, b, c];
            v.sort();
            v[1]
        }),
    ];
    for (op, f) in ternary {
        Probe::new(3, 1, &format!("{op} v20, v1, v2, v3")).check(&inputs, |_, x| vec![f(x[0], x[1], x[2])]);
    }
    let unary: [(&str, fn(u32) -> u32); 4] = [
        ("v_not_b32", |a| !a),
        ("v_bfrev_b32", |a| a.reverse_bits()),
        ("v_ffbh_u32", |a| if a == 0 { u32::MAX } else { a.leading_zeros() }),
        ("v_mov_b32", |a| a),
    ];
    for (op, f) in unary {
        for body in [format!("{op}_e32 v20, v1"), format!("{op}_e64 v20, v1")] {
            Probe::new(3, 1, &body).check(&inputs, |_, x| vec![f(x[0])]);
        }
    }
}

#[test]
fn sixty_four_bit_shifts_and_wide_multiply_add() {
    let inputs = lanes(4, word);
    let pair = |x: &[u32], k: usize| x[k] as u64 | (x[k + 1] as u64) << 32;
    let shifts: [(&str, fn(u32, u64) -> u64); 3] = [
        ("v_lshlrev_b64", |n, v| v << (n & 63)),
        ("v_lshrrev_b64", |n, v| v >> (n & 63)),
        ("v_ashrrev_i64", |n, v| ((v as i64) >> (n & 63)) as u64),
    ];
    for (op, f) in shifts {
        Probe::new(4, 2, &format!("{op} v[20:21], v4, v[1:2]")).check(&inputs, |_, x| {
            let r = f(x[3], pair(x, 0));
            vec![r as u32, (r >> 32) as u32]
        });
    }
    Probe::new(
        4,
        3,
        "v_mad_u64_u32 v[20:21], s[12:13], v1, v2, v[3:4]
         v_cndmask_b32 v22, 0, 1, s[12:13]",
    )
    .check(&inputs, |_, x| {
        let product = x[0] as u64 * x[1] as u64;
        let (sum, over) = product.overflowing_add(pair(x, 2));
        vec![sum as u32, (sum >> 32) as u32, over as u32]
    });
}

#[test]
fn conditional_moves_follow_vcc_and_scalar_masks() {
    let inputs = lanes(3, word);
    Probe::new(3, 1, "v_cmp_gt_u32 vcc, v1, v2\nv_cndmask_b32_e32 v20, v3, v2, vcc").check(&inputs, |_, x| {
        vec![if x[0] > x[1] { x[1] } else { x[2] }]
    });
    Probe::new(3, 1, "v_cmp_lt_i32 s[14:15], v1, v2\nv_cndmask_b32_e64 v20, v3, -1, s[14:15]").check(&inputs, |_, x| {
        vec![if (x[0] as i32) < x[1] as i32 { u32::MAX } else { x[2] }]
    });
}

#[test]
fn mbcnt_counts_the_lanes_below_in_a_wave_of_64() {
    let inputs = lanes(1, |l, _| (l % 3 != 0) as u32);
    let probe = Probe::new(
        1,
        3,
        "v_cmp_ne_u32 s[12:13], 0, v1
         v_mbcnt_lo_u32_b32 v20, s12, 0
         v_mbcnt_hi_u32_b32 v21, s13, v20
         v_mbcnt_lo_u32_b32 v22, -1, 0
         v_mbcnt_hi_u32_b32 v22, -1, v22",
    );
    let mask = mask_of(&inputs, 1, |x| x[0] != 0);
    probe.check(&inputs, |l, _| {
        let below = if l == 0 { 0 } else { u64::MAX >> (64 - l) };
        let low = (mask as u32 & below as u32).count_ones();
        vec![low, (mask & below).count_ones(), l as u32]
    });
}
