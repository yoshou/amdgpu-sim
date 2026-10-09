use crate::probe::Probe;
use crate::vector::lanes;

const SHORTS: [u32; 16] = [
    0x0000_0000,
    0xffff_0001,
    0x1234_7fff,
    0xabcd_8000,
    0x0000_ffff,
    0x5555_fffe,
    0x0001_0010,
    0x8000_000f,
    0x0000_0011,
    0xdead_beef,
    0x0000_00ff,
    0xffff_ff00,
    0x7fff_1234,
    0x0000_4000,
    0x0101_c000,
    0x0000_0003,
];

fn short(l: usize, k: usize) -> u32 {
    SHORTS[(l * (k + 1) + 3 * k + l / 16) % SHORTS.len()]
}

fn low(x: u32) -> u32 {
    x & 0xffff
}

fn signed(x: u32) -> i32 {
    x as u16 as i16 as i32
}

#[test]
fn sixteen_bit_integer_operations_zero_the_high_half() {
    let inputs = lanes(3, short);
    let ops: [(&str, fn(u32, u32) -> u32); 11] = [
        ("v_add_u16", |a, b| low(a.wrapping_add(b))),
        ("v_sub_u16", |a, b| low(a.wrapping_sub(b))),
        ("v_subrev_u16", |a, b| low(b.wrapping_sub(a))),
        ("v_mul_lo_u16", |a, b| low(a.wrapping_mul(b))),
        ("v_lshlrev_b16", |a, b| low(low(b) << (a & 15))),
        ("v_lshrrev_b16", |a, b| low(b) >> (a & 15)),
        ("v_ashrrev_i16", |a, b| low((signed(b) >> (a & 15)) as u32)),
        ("v_max_u16", |a, b| low(a).max(low(b))),
        ("v_min_u16", |a, b| low(a).min(low(b))),
        ("v_max_i16", |a, b| low(signed(a).max(signed(b)) as u32)),
        ("v_min_i16", |a, b| low(signed(a).min(signed(b)) as u32)),
    ];
    for (op, f) in ops {
        for body in [format!("v_mov_b32 v20, -1\n{op}_e32 v20, v1, v2"), format!("v_mov_b32 v20, -1\n{op}_e64 v20, v1, v2")] {
            Probe::new(3, 1, &body).check(&inputs, |_, x| vec![f(x[0], x[1])]);
        }
    }
    Probe::new(3, 2, "v_mad_u16 v20, v1, v2, v3\nv_mad_i16 v21, v1, v2, v3").check(&inputs, |_, x| {
        vec![
            low(low(x[0]) * low(x[1]) + low(x[2])),
            low((signed(x[0]) * signed(x[1]) + signed(x[2])) as u32),
        ]
    });
}

#[test]
fn sixteen_bit_clamps_saturate() {
    let inputs = lanes(3, short);
    Probe::new(3, 4, "v_add_u16_e64 v20, v1, v2 clamp\nv_sub_u16_e64 v21, v1, v2 clamp\nv_mad_u16 v22, v1, v2, v3 clamp\nv_mad_i16 v23, v1, v2, v3 clamp").check(&inputs, |_, x| {
        vec![
            (low(x[0]) + low(x[1])).min(0xffff),
            low(x[0]).saturating_sub(low(x[1])),
            (low(x[0]) * low(x[1]) + low(x[2])).min(0xffff),
            low((signed(x[0]) * signed(x[1]) + signed(x[2])).clamp(-32768, 32767) as u32),
        ]
    });
}

#[test]
fn sixteen_bit_compares_read_the_low_half() {
    let inputs = lanes(2, short);
    let ops: [(&str, fn(u32, u32) -> bool); 6] = [
        ("v_cmp_gt_u16", |a, b| low(a) > low(b)),
        ("v_cmp_le_u16", |a, b| low(a) <= low(b)),
        ("v_cmp_eq_u16", |a, b| low(a) == low(b)),
        ("v_cmp_lt_i16", |a, b| signed(a) < signed(b)),
        ("v_cmp_ge_i16", |a, b| signed(a) >= signed(b)),
        ("v_cmp_ne_i16", |a, b| signed(a) != signed(b)),
    ];
    for (op, f) in ops {
        Probe::new(2, 1, &format!("{op}_e32 vcc, v1, v2\nv_cndmask_b32 v20, 0, 1, vcc")).check(&inputs, |_, x| vec![f(x[0], x[1]) as u32]);
        Probe::new(2, 1, &format!("{op}_e64 s[12:13], v1, v2\nv_cndmask_b32 v20, 0, 1, s[12:13]")).check(&inputs, |_, x| vec![f(x[0], x[1]) as u32]);
    }
}

const HALVES: [u16; 24] = [
    0x0000, 0x8000, 0x3c00, 0xbc00, 0x0001, 0x8001, 0x03ff, 0x0400, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x3555, 0x3800, 0x4248, 0xc248, 0x1400,
    0x0200, 0x5640, 0xd640, 0x3bff, 0x0401, 0x7400, 0x87ff,
];

fn half(l: usize, k: usize) -> u32 {
    let h = HALVES[(l * (2 * k + 1) + 5 * k + l / 24) % HALVES.len()] as u32;
    h | (0x5a5a_0000 ^ (l as u32) << 20)
}

fn decode(h: u32) -> (bool, i128, i32) {
    let h = h & 0xffff;
    let sign = h & 0x8000 != 0;
    let exponent = (h >> 10 & 31) as i32;
    let fraction = (h & 0x3ff) as i128;
    if exponent == 0 {
        (sign, fraction, -24)
    } else {
        (sign, fraction | 0x400, exponent - 25)
    }
}

fn encode(negative: bool, magnitude: i128, scale: i32) -> u32 {
    let sign = if negative { 0x8000 } else { 0 };
    if magnitude == 0 {
        return sign;
    }
    let mut exponent = -24;
    while magnitude >> (exponent - scale) >= 2048 {
        exponent += 1;
    }
    let shift = exponent - scale;
    let (mut m, remainder, half) = if shift > 0 {
        (magnitude >> shift, magnitude & ((1 << shift) - 1), 1i128 << (shift - 1))
    } else {
        (magnitude << -shift, 0, 1)
    };
    if remainder > half || remainder == half && shift > 0 && m & 1 == 1 {
        m += 1;
    }
    if m == 2048 {
        m = 1024;
        exponent += 1;
    }
    if exponent > 5 {
        return sign | 0x7c00;
    }
    if m >= 1024 {
        sign | ((exponent + 25) as u32) << 10 | (m as u32 - 1024)
    } else {
        sign | m as u32
    }
}

fn special(h: u32) -> bool {
    h & 0x7c00 == 0x7c00
}

fn fma_half(a: u32, b: u32, c: u32) -> u32 {
    if special(a) || special(b) || special(c) {
        let v = half::f16::from_bits(a as u16).to_f64() * half::f16::from_bits(b as u16).to_f64() + half::f16::from_bits(c as u16).to_f64();
        return half::f16::from_f64(v).to_bits() as u32;
    }
    let (sa, ma, ea) = decode(a);
    let (sb, mb, eb) = decode(b);
    let (sc, mc, ec) = decode(c);
    let product = ma * mb << (ea + eb + 48);
    let product = if sa != sb { -product } else { product };
    let addend = mc << (ec + 48);
    let addend = if sc { -addend } else { addend };
    let sum = product + addend;
    if sum == 0 {
        let negative_zeros = product == 0 && addend == 0 && sa != sb && sc;
        return if negative_zeros { 0x8000 } else { 0 };
    }
    encode(sum < 0, sum.abs(), -48)
}

#[test]
fn half_precision_fused_multiply_add_rounds_once() {
    let inputs = lanes(3, half);
    Probe::new(3, 1, "v_fma_f16 v20, v1, v2, v3").check_float(&inputs, |_, x| vec![fma_half(x[0], x[1], x[2])]);
    let tricky = lanes(3, |l, k| {
        let rows: [[u32; 3]; 8] = [
            [0x6400, 0x6400, 0x0001],
            [0x3c01, 0x3c01, 0x8001],
            [0x7bff, 0x3c00, 0x0001],
            [0x5bff, 0x5bff, 0xd7ff],
            [0x8000, 0x3c00, 0x8000],
            [0x8000, 0x3c00, 0x0000],
            [0x3c00, 0x3c00, 0xbc00],
            [0x0001, 0x0001, 0x7bff],
        ];
        rows[l % 8][k]
    });
    Probe::new(3, 1, "v_fma_f16 v20, v1, v2, v3").check_float(&tricky, |_, x| vec![fma_half(x[0], x[1], x[2])]);
}

#[test]
fn half_conversions_write_a_zero_high_half() {
    let singles = lanes(1, |l, _| [0x3f80_0000u32, 0x3380_0000, 0x3380_0001, 0x477f_f000, 0x477f_ef00, 0x3555_5555, 0x8000_0001, 0x7f80_0000, 0x3a80_0000, 0xc77f_e000][l % 10]);
    Probe::new(1, 1, "v_mov_b32 v20, -1\nv_cvt_f16_f32 v20, v1").check_float(&singles, |_, x| vec![half::f16::from_f32(f32::from_bits(x[0])).to_bits() as u32]);
    let inputs = lanes(1, half);
    Probe::new(1, 1, "v_cvt_f32_f16 v20, v1").check_float(&inputs, |_, x| vec![half::f16::from_bits(x[0] as u16).to_f32().to_bits()]);
}
