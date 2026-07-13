//! Runtime helper functions called from JIT-generated scalar code, resolved
//! via the process symbol table (same mechanism as the masked backend's
//! `image_*` intrinsics).

/// The shared 1280-bit 2/π fraction table (also inlined as an LLVM constant by
/// the SPMD backend's vectorized `V_TRIG_PREOP_F64`, [super::emit_vec]).
pub(crate) const TWO_OVER_PI_FRACTION: [u64; 20] = [
    0xBA10AC06608DF8F6,
    0x25D4D7F6BF623F1A,
    0xE2F67A0E73EF14A5,
    0xD45AEA4F758FD7CB,
    0x136E9E8C7ECD3CBF,
    0xDA3EDA6CFD9E4F96,
    0x301FDE5E2316B414,
    0x50763FF12FFFBC0B,
    0x73E93908BF177BF2,
    0xFC827323AC7306A6,
    0x8909D338E04D68BE,
    0x4E7DD1046BEA5D76,
    0x2439FC3BD6396253,
    0xA5C00C925DD413A3,
    0x8AC36E48DC74849B,
    0x2083FCA2C757BD77,
    0xBB81B6C52B327887,
    0x2A53F84EAFA3EA69,
    0x000145F306DC9C88,
    0x0000000000000000,
];

fn rt_get_exp_f32(val: f32) -> i16 {
    ((val.to_bits() >> 23) & 0xff) as i16
}

/// `V_DIV_SCALE_F32`: bit-exact port of `RDNAProcessor::div_scale_f32`. Returns
/// the scaled value in the low 32 bits and the VCC scale flag in bit 32.
#[unsafe(no_mangle)]
pub extern "C" fn scalar_div_scale_f32(s0: f32, s1: f32, s2: f32) -> u64 {
    let mut vcc = false;
    let mut d = s0 * s2 / s1;
    let s1_exp = rt_get_exp_f32(s1);
    let s2_exp = rt_get_exp_f32(s2);
    if s2 == 0.0 || s1 == 0.0 {
        d = f32::NAN;
    } else if s2_exp - s1_exp >= 96 {
        vcc = true;
        if s0 == s1 { d = s0 * 64f32.exp2(); }
    } else if !s1.is_normal() {
        d = s0 * 64f32.exp2();
    } else if (!(1.0 / s1).is_normal()) && (!(s2 / s1).is_normal()) {
        vcc = true;
        if s0 == s1 { d = s0 * 64f32.exp2(); }
    } else if !(1.0 / s1).is_normal() {
        d = s0 * (-64f32).exp2();
    } else if !(s2 / s1).is_normal() {
        vcc = true;
        if s0 == s2 { d = s0 * 64f32.exp2(); }
    } else if s2_exp <= 23 {
        d = s0 * 64f32.exp2();
    }
    (d.to_bits() as u64) | ((vcc as u64) << 32)
}

/// `V_DIV_FIXUP_F32`: bit-exact port of `RDNAProcessor::div_fixup_f32`. `s0` is
/// the Newton-Raphson quotient approximation from the divide chain.
#[unsafe(no_mangle)]
pub extern "C" fn scalar_div_fixup_f32(s0: f32, s1: f32, s2: f32) -> f32 {
    let sign_out = s1.is_sign_negative() != s2.is_sign_negative();
    if s2.is_nan() {
        s2
    } else if s1.is_nan() {
        s1
    } else if s1 == 0.0 && s2 == 0.0 {
        f32::from_bits(0xffc00000)
    } else if s1.is_infinite() && s2.is_infinite() {
        f32::from_bits(0xffc00000)
    } else if s1 == 0.0 || s2.is_infinite() {
        if sign_out { f32::NEG_INFINITY } else { f32::INFINITY }
    } else if s1.is_infinite() || s2 == 0.0 {
        if sign_out { -0.0 } else { 0.0 }
    } else if (rt_get_exp_f32(s2) - rt_get_exp_f32(s1)) < -150 {
        if sign_out { -0.0 } else { 0.0 }
    } else if rt_get_exp_f32(s1) == 255 {
        if sign_out { f32::NEG_INFINITY } else { f32::INFINITY }
    } else {
        if sign_out { -s0.abs() } else { s0.abs() }
    }
}

/// `V_FREXP_MANT_F32`: mantissa in [0.5, 1) from `libm::frexpf`, passing NaN/Inf
/// through unchanged (mirrors `RDNAProcessor::v_frexp_mant_f32_e32`).
#[unsafe(no_mangle)]
pub extern "C" fn scalar_frexp_mant_f32(x: f32) -> f32 {
    if x.is_nan() || x.is_infinite() { x } else { libm::frexpf(x).0 }
}

/// `V_FREXP_EXP_I32_F32`: binary exponent from `libm::frexpf`; 0 for NaN/Inf
/// (mirrors `RDNAProcessor::v_frexp_exp_i32_f32_e32`).
#[unsafe(no_mangle)]
pub extern "C" fn scalar_frexp_exp_f32(x: f32) -> i32 {
    if x.is_nan() || x.is_infinite() { 0 } else { libm::frexpf(x).1 }
}

/// `V_TRIG_PREOP_F64`: Payne–Hanek argument-reduction preprocessing — returns a
/// 53-bit segment of the high-precision 2/π expansion selected by the exponent
/// of `src0` and the segment index `src1`. Bit-exact port of the masked
/// reference (`RDNAProcessor::v_trig_preop_f64`); shares the same 1280-bit 2/π
/// fraction table so trig-heavy paths match the masked emulator exactly.
#[unsafe(no_mangle)]
pub extern "C" fn v_trig_preop_f64(src0: f64, src1: u32) -> f64 {
    // Biased exponent of |src0|.
    let exp = ((src0.to_bits() >> 52) & 0x7ff) as i32;

    let mut shift = (src1 & 0x1f) as i32 * 53;
    if exp > 1077 {
        shift += exp - 1077;
    }

    let bit_offset = 1201 - 53 - shift;
    let mut scale = -53 - shift;
    if exp >= 1968 {
        scale += 128;
    }

    // Out-of-range segment ⇒ the fraction is exhausted (returns 0).
    if bit_offset < 0 || (bit_offset as usize) + 53 > TWO_OVER_PI_FRACTION.len() * 64 {
        return 0.0;
    }
    let result = get_bits_u64(&TWO_OVER_PI_FRACTION, bit_offset as usize, 53);
    libm::ldexp(result as f64, scale)
}

/// Extract `bit_size` bits starting at `bit_offset` from a little-endian array
/// of u64 words (mirrors `crate::buffer::get_bits_u64`).
fn get_bits_u64(buffer: &[u64], bit_offset: usize, bit_size: usize) -> u64 {
    let word = bit_offset / 64;
    let bit = bit_offset % 64;
    if bit + bit_size <= 64 {
        let mask = if bit_size == 64 { u64::MAX } else { (1u64 << bit_size) - 1 };
        (buffer[word] >> bit) & mask
    } else {
        let lo_bits = 64 - bit;
        let hi_bits = bit_size - lo_bits;
        ((buffer[word] >> bit) & ((1u64 << lo_bits) - 1))
            | ((buffer[word + 1] & ((1u64 << hi_bits) - 1)) << lo_bits)
    }
}
