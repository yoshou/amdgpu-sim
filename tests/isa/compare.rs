//! Comparing a result against the value the hardware produced.
//!
//! Split by result type, because the bit layout, the ULP distance and the way a
//! failure needs to be printed all differ.

// ---------------------------------------------------------------- comparison
pub(crate) fn is_nan_f32(bits: u32) -> bool {
    (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0
}

pub(crate) fn is_inf_f32(bits: u32) -> bool {
    (bits & 0x7FFF_FFFF) == 0x7F80_0000
}

pub(crate) fn is_zero_f32(bits: u32) -> bool {
    (bits & 0x7FFF_FFFF) == 0
}

/// Monotone map from an f32 bit pattern to an ordered integer; the difference
/// between two such integers is the ULP distance.
pub(crate) fn ordered_f32(bits: u32) -> i64 {
    if bits & 0x8000_0000 != 0 {
        -((bits & 0x7FFF_FFFF) as i64)
    } else {
        bits as i64
    }
}

pub(crate) fn ulp_f32(a: u32, b: u32) -> i64 {
    (ordered_f32(a) - ordered_f32(b)).abs()
}

pub(crate) fn show_f32(bits: u32) -> String {
    format!("0x{:08X} ({:e})", bits, f32::from_bits(bits))
}

pub(crate) fn is_nan_f64(bits: u64) -> bool {
    (bits & 0x7FF0_0000_0000_0000) == 0x7FF0_0000_0000_0000 && (bits & 0x000F_FFFF_FFFF_FFFF) != 0
}

pub(crate) fn is_inf_f64(bits: u64) -> bool {
    (bits & 0x7FFF_FFFF_FFFF_FFFF) == 0x7FF0_0000_0000_0000
}

pub(crate) fn is_zero_f64(bits: u64) -> bool {
    (bits & 0x7FFF_FFFF_FFFF_FFFF) == 0
}

pub(crate) fn ordered_f64(bits: u64) -> i128 {
    if bits & 0x8000_0000_0000_0000 != 0 {
        -((bits & 0x7FFF_FFFF_FFFF_FFFF) as i128)
    } else {
        bits as i128
    }
}

pub(crate) fn ulp_f64(a: u64, b: u64) -> i128 {
    (ordered_f64(a) - ordered_f64(b)).abs()
}

pub(crate) fn show_f64(bits: u64) -> String {
    format!("0x{:016X} ({:e})", bits, f64::from_bits(bits))
}
