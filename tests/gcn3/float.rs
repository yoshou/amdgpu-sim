use crate::probe::{Probe, LANES};
use crate::vector::lanes;

const SINGLES: [u32; 32] = [
    0x0000_0000,
    0x8000_0000,
    0x3f80_0000,
    0xbf80_0000,
    0x3f00_0000,
    0x3fc0_0000,
    0x0000_0001,
    0x8000_0001,
    0x007f_ffff,
    0x0040_0000,
    0x8040_0000,
    0x0080_0000,
    0x0080_0001,
    0x8080_0000,
    0x1e3c_e508,
    0x1f5e_e8f5,
    0x7f7f_ffff,
    0xff7f_ffff,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7fc1_2345,
    0x4000_0000,
    0x4040_0000,
    0xc0a0_0000,
    0x3e4c_cccd,
    0x00c0_0000,
    0x0100_0000,
    0x3f7f_ffff,
    0xbf00_0001,
    0x4b80_0001,
    0xcb00_0003,
];

fn single(l: usize, k: usize) -> u32 {
    SINGLES[(l * (2 * k + 1) + 7 * k + l / 32) % SINGLES.len()]
}

fn f(bits: u32) -> f32 {
    f32::from_bits(bits)
}

fn ftz(x: f32) -> f32 {
    if x.is_subnormal() {
        if x.is_sign_negative() {
            -0.0
        } else {
            0.0
        }
    } else {
        x
    }
}

#[derive(Clone, Copy)]
struct Mode {
    inputs: bool,
    outputs: bool,
}

impl Mode {
    fn of(flush: bool) -> Self {
        Mode {
            inputs: flush,
            outputs: flush,
        }
    }
    fn i(self, x: f32) -> f32 {
        if self.inputs {
            ftz(x)
        } else {
            x
        }
    }
    fn o(self, x: f32) -> f32 {
        if self.outputs {
            ftz(x)
        } else {
            x
        }
    }
}

fn probe(inputs: usize, outputs: usize, body: &str, flush: bool) -> Probe {
    let probe = Probe::new(inputs, outputs, body);
    if flush {
        probe.flushing()
    } else {
        probe
    }
}

fn subtract(a: f32, b: f32) -> f32 {
    if b.is_nan() && !a.is_nan() {
        return f32::from_bits((b.to_bits() | 0x0040_0000) ^ 0x8000_0000);
    }
    a - b
}

#[test]
fn arithmetic_flushes_denormals_by_the_mode() {
    let inputs = lanes(3, single);
    for flush in [false, true] {
        let m = Mode::of(flush);
        let ops: [(&str, fn(Mode, f32, f32) -> f32); 4] = [
            ("v_add_f32", |m, a, b| m.o(m.i(a) + m.i(b))),
            ("v_sub_f32", |m, a, b| m.o(subtract(m.i(a), m.i(b)))),
            ("v_subrev_f32", |m, a, b| m.o(subtract(m.i(b), m.i(a)))),
            ("v_mul_f32", |m, a, b| m.o(m.i(a) * m.i(b))),
        ];
        for (op, g) in ops {
            for form in ["e32", "e64"] {
                probe(3, 1, &format!("{op}_{form} v20, v1, v2"), flush).check_float(&inputs, |_, x| vec![g(m, f(x[0]), f(x[1])).to_bits()]);
            }
        }
        probe(3, 1, "v_fma_f32 v20, v1, v2, v3", flush).check_float(&inputs, |_, x| {
            vec![m.o(m.i(f(x[0])).mul_add(m.i(f(x[1])), m.i(f(x[2])))).to_bits()]
        });
    }
}

fn min_max(a: f32, b: f32, max: bool) -> f32 {
    if a.is_nan() {
        return b;
    }
    if b.is_nan() {
        return a;
    }
    if a == b {
        return if max == (a.is_sign_positive()) { a } else { b };
    }
    if (a > b) == max {
        a
    } else {
        b
    }
}

#[test]
fn minimum_and_maximum_pass_denormals_through() {
    let inputs = lanes(3, single);
    for flush in [false, true] {
        for (op, max) in [("v_min_f32", false), ("v_max_f32", true)] {
            probe(3, 1, &format!("{op} v20, v1, v2"), flush).check_float(&inputs, |_, x| vec![min_max(f(x[0]), f(x[1]), max).to_bits()]);
        }
    }
}

#[test]
fn multiply_add_rounds_the_product_and_always_flushes() {
    let inputs = lanes(3, single);
    let mad = |a: f32, b: f32, c: f32| ftz(ftz(ftz(a) * ftz(b)) + ftz(c));
    for flush in [false, true] {
        probe(3, 1, "v_mad_f32 v20, v1, v2, v3", flush).check_float(&inputs, |_, x| vec![mad(f(x[0]), f(x[1]), f(x[2])).to_bits()]);
        probe(3, 1, "v_mov_b32 v20, v3\nv_mac_f32 v20, v1, v2", flush).check_float(&inputs, |_, x| vec![mad(f(x[0]), f(x[1]), f(x[2])).to_bits()]);
        probe(3, 1, "v_mov_b32 v20, v3\nv_mac_f32_e64 v20, v1, v2", flush).check_float(&inputs, |_, x| vec![mad(f(x[0]), f(x[1]), f(x[2])).to_bits()]);
        probe(3, 1, "v_madmk_f32 v20, v1, 0x3e4ccccd, v2", flush).check_float(&inputs, |_, x| vec![mad(f(x[0]), f(0x3e4c_cccd), f(x[1])).to_bits()]);
        probe(3, 1, "v_madak_f32 v20, v1, v2, 0xbf000001", flush).check_float(&inputs, |_, x| vec![mad(f(x[0]), f(x[1]), f(0xbf00_0001)).to_bits()]);
    }
}

#[test]
fn rounding_operations_see_flushed_inputs() {
    let inputs = lanes(1, single);
    for flush in [false, true] {
        let m = Mode::of(flush);
        let ops: [(&str, fn(f32) -> f32); 4] = [
            ("v_floor_f32", f32::floor),
            ("v_ceil_f32", f32::ceil),
            ("v_trunc_f32", f32::trunc),
            ("v_rndne_f32", |x| {
                let r = x.round();
                if (x - x.trunc()).abs() == 0.5 {
                    2.0 * (x / 2.0).round()
                } else {
                    r
                }
            }),
        ];
        for (op, g) in ops {
            probe(1, 1, &format!("{op}_e32 v20, v1"), flush).check_float(&inputs, |_, x| {
                let r = g(m.i(f(x[0])));
                vec![if r.is_nan() { f(x[0]).to_bits() | 0x0040_0000 } else { m.o(r).to_bits() }]
            });
        }
    }
}

#[test]
fn compares_flush_their_inputs_but_classes_see_denormals() {
    let inputs = lanes(2, single);
    for flush in [false, true] {
        let m = Mode::of(flush);
        let ops: [(&str, fn(f32, f32) -> bool); 6] = [
            ("v_cmp_lt_f32", |a, b| a < b),
            ("v_cmp_eq_f32", |a, b| a == b),
            ("v_cmp_le_f32", |a, b| a <= b),
            ("v_cmp_neq_f32", |a, b| !(a == b)),
            ("v_cmp_nlt_f32", |a, b| !(a < b)),
            ("v_cmp_u_f32", |a, b| a.is_nan() || b.is_nan()),
        ];
        for (op, g) in ops {
            probe(2, 1, &format!("{op}_e32 vcc, v1, v2\nv_cndmask_b32 v20, 0, 1, vcc"), flush)
                .check(&inputs, |_, x| vec![g(m.i(f(x[0])), m.i(f(x[1]))) as u32]);
        }
        probe(2, 1, "s_movk_i32 s14, 0x90\nv_cmp_class_f32_e64 s[12:13], v1, s14\nv_cndmask_b32 v20, 0, 1, s[12:13]", flush)
            .check(&inputs, |_, x| vec![f(x[0]).is_subnormal() as u32]);
    }
}

#[test]
fn conversions_flush_the_single_precision_side() {
    let inputs = lanes(2, single);
    for flush in [false, true] {
        let m = Mode::of(flush);
        probe(2, 2, "v_cvt_f64_f32 v[20:21], v1", flush).check(&inputs, |_, x| {
            let wide = m.i(f(x[0])) as f64;
            let bits = if f(x[0]).is_nan() { (wide.to_bits()) | 0x0008_0000_0000_0000 } else { wide.to_bits() };
            vec![bits as u32, (bits >> 32) as u32]
        });
        probe(2, 1, "v_cvt_f32_f64 v20, v[1:2]", flush).check(&inputs, |_, x| {
            let wide = f64::from_bits(x[0] as u64 | (x[1] as u64) << 32);
            let narrow = wide as f32;
            vec![if wide.is_nan() { narrow.to_bits() | 0x0040_0000 } else { m.o(narrow).to_bits() }]
        });
        probe(2, 2, "v_cvt_i32_f32 v20, v1\nv_cvt_u32_f32 v21, v1", flush).check(&inputs, |_, x| {
            let v = f(x[0]);
            let signed = if v.is_nan() { 0 } else { v as i32 as u32 };
            let unsigned = if v.is_nan() { 0 } else { v as u32 };
            vec![signed, unsigned]
        });
        probe(2, 2, "v_cvt_f32_i32 v20, v1\nv_cvt_f32_u32 v21, v1", flush)
            .check(&inputs, |_, x| vec![(x[0] as i32 as f32).to_bits(), (x[0] as f32).to_bits()]);
    }
}

#[test]
fn modifiers_scale_clamp_and_negate() {
    let inputs = lanes(3, single);
    let output = |v: f32, clamp: bool, factor: f32, flush: bool| {
        let m = Mode::of(flush);
        let mut v = v;
        if factor != 1.0 {
            v = m.o(m.i(v) * factor);
            if v.to_bits() & 0x7f80_0000 == 0 {
                v = 0.0;
            }
        }
        if clamp {
            v = if m.i(v) > 0.0 { v.min(1.0) } else { 0.0 };
        }
        v
    };
    for flush in [false, true] {
        let m = Mode::of(flush);
        let cases: [(&str, bool, f32); 4] = [("clamp", true, 1.0), ("mul:2", false, 2.0), ("mul:4", false, 4.0), ("clamp div:2", true, 0.5)];
        for (suffix, clamp, factor) in cases {
            probe(3, 1, &format!("v_add_f32_e64 v20, -v1, |v2| {suffix}"), flush).check_float(&inputs, |_, x| {
                let sum = m.o(m.i(-f(x[0])) + m.i(f(x[1]).abs()));
                vec![output(sum, clamp, factor, flush).to_bits()]
            });
        }
    }
}

#[test]
fn reciprocal_and_square_roots_always_flush() {
    let inputs = lanes(1, single);
    for flush in [false, true] {
        probe(1, 3, "v_rcp_f32 v20, v1\nv_sqrt_f32 v21, v1\nv_rsq_f32 v22, v1", flush).check_float(&inputs, |_, x| {
            let v = ftz(f(x[0]));
            vec![ftz(1.0 / v).to_bits(), ftz(v.sqrt()).to_bits(), ftz(1.0 / v.sqrt()).to_bits()]
        });
    }
}

#[test]
fn scaling_by_powers_of_two_flushes_both_ends() {
    let inputs = lanes(2, |l, k| if k == 1 { [0u32, 1, 0xffff_ffff, 127, 0xffff_ff81, 200, 0xffff_ff00, 3][l % 8] } else { single(l, k) });
    for flush in [false, true] {
        let m = Mode::of(flush);
        probe(2, 1, "v_ldexp_f32 v20, v1, v2", flush).check_float(&inputs, |_, x| {
            let v = m.i(f(x[0]));
            let n = (x[1] as i32).clamp(-400, 400);
            let r = if v.is_nan() { f32::from_bits(v.to_bits() | 0x0040_0000) } else { ((v as f64) * 2f64.powi(n)) as f32 };
            vec![m.o(r).to_bits()]
        });
    }
}

#[test]
fn double_precision_keeps_its_denormals_in_both_modes() {
    let doubles: [u64; 8] = [0x0000_0000_0000_0001, 0x8000_0000_0000_0003, 0x3ff0_0000_0000_0000, 0x0010_0000_0000_0000, 0xbfe0_0000_0000_0001, 0x7ff0_0000_0000_0000, 0x0008_0000_0000_0000, 0x4005_bf0a_8b14_5769];
    let inputs = lanes(4, |l, k| {
        let d = doubles[(l * (k / 2 + 1) + k) % doubles.len()];
        if k % 2 == 0 { d as u32 } else { (d >> 32) as u32 }
    });
    let pair = |x: &[u32], k: usize| f64::from_bits(x[k] as u64 | (x[k + 1] as u64) << 32);
    for flush in [false, true] {
        probe(4, 6, "v_add_f64 v[20:21], v[1:2], v[3:4]\nv_mul_f64 v[22:23], v[1:2], v[3:4]\nv_fma_f64 v[24:25], v[1:2], v[3:4], v[1:2]", flush).check(&inputs, |_, x| {
            let (a, b) = (pair(x, 0), pair(x, 2));
            let words = |v: f64| [v.to_bits() as u32, (v.to_bits() >> 32) as u32];
            let mut out = vec![];
            out.extend(words(a + b));
            out.extend(words(a * b));
            out.extend(words(a.mul_add(b, a)));
            out
        });
    }
}

#[test]
fn every_lane_of_a_wave_of_64_computes() {
    let inputs = lanes(2, single);
    assert_eq!(inputs.len(), 2 * LANES);
    probe(2, 1, "v_mul_f32 v20, v1, v2", true).check_float(&inputs, |_, x| vec![ftz(ftz(f(x[0])) * ftz(f(x[1]))).to_bits()]);
}

