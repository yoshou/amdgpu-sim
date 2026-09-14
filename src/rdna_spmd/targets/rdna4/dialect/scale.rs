//! ISA §16.12 LDEXP: x * 2^n with one final rounding. AVX-512 SCALEF
//! retains the existing native vector implementation. Portable prescaling
//! prevents premature underflow: negative chunks are applied only below Emin
//! and retain a full significand's headroom before the final multiplication.
use super::*;

pub(super) fn f32(e: &Emitter, a: &[Value]) -> Value {
    scale(e, Ty::F32, a[0], a[1])
}
pub(super) fn f64(e: &Emitter, a: &[Value]) -> Value {
    scale(e, Ty::F64, a[0], a[1])
}

fn scale(e: &Emitter, ty: Ty, value: Value, exponent: Value) -> Value {
    #[cfg(target_arch = "x86_64")]
    if value.is_vector() && std::arch::is_x86_feature_detected!("avx512f") {
        let w = value.ty().vector_size();
        let native = if ty == Ty::F32 { 16 } else { 8 };
        if w >= native || (w == native / 2 && std::arch::is_x86_feature_detected!("avx512vl")) {
            return native_scale(e, ty, value, exponent, w, native);
        }
    }
    portable(e, ty, value, exponent)
}

fn power(e: &Emitter, ty: Ty, exponent: Value) -> Value {
    let ir = e.ir;
    let (it, bias, fraction) = if ty == Ty::F32 {
        (Ty::I32, 127, 23)
    } else {
        (Ty::I64, 1023, 52)
    };
    let exponent = if it == Ty::I64 {
        ir.sext(exponent, e.ty(it))
    } else {
        exponent
    };
    let biased = ir.add(exponent, e.constant(it, bias));
    let bits = ir.shl(biased, e.constant(it, fraction));
    ir.bitcast(bits, e.ty(ty))
}

fn portable(e: &Emitter, ty: Ty, mut value: Value, mut exp: Value) -> Value {
    let ir = e.ir;
    let (max, min, precision) = if ty == Ty::F32 {
        (127i32, -126i32, 24)
    } else {
        (1023, -1022, 53)
    };
    let k = |v: i32| e.constant(Ty::I32, v as u32 as u64);
    for _ in 0..2 {
        let high = ir.icmp(IntPred::Sgt, exp, k(max));
        let low = ir.icmp(IntPred::Slt, exp, k(min));
        let step = ir.select(low, k(min + precision), k(0));
        let step = ir.select(high, k(max), step);
        exp = ir.sub(exp, step);
        value = ir.fmul(value, power(e, ty, step));
    }
    let high = ir.icmp(IntPred::Sgt, exp, k(max));
    exp = ir.select(high, k(max), exp);
    let low = ir.icmp(IntPred::Slt, exp, k(min));
    exp = ir.select(low, k(min), exp);
    ir.fmul(value, power(e, ty, exp))
}

fn native_scale(e: &Emitter, ty: Ty, value: Value, exponent: Value, w: u32, native: u32) -> Value {
    let ir = e.ir;
    let lanes = w.min(native);
    let scalar = if ty == Ty::F32 { ir.f32() } else { ir.f64() };
    let chunk_ty = scalar.vector(lanes);
    let name = format!(
        "llvm.x86.avx512.mask.scalef.{}.{}",
        if ty == Ty::F32 { "ps" } else { "pd" },
        if lanes == native { 512 } else { 256 }
    );
    let mut parts = Vec::new();
    for index in 0..w / lanes {
        let mut a = value;
        let mut b = exponent;
        if w != lanes {
            let mask: Vec<u32> = (0..lanes).map(|lane| index * lanes + lane).collect();
            a = ir.shuffle_by(a, a.ty().poison(), &mask);
            b = ir.shuffle_by(b, b.ty().poison(), &mask);
        }
        let b = ir.sitofp(b, chunk_ty);
        let mask = ir
            .int(if lanes == 16 { 16 } else { 8 })
            .const_int((1 << lanes) - 1);
        let mut args = vec![a, b, a, mask];
        if lanes == native {
            args.push(ir.ci32(4));
        }
        let types = args.iter().map(|a| a.ty()).collect::<Vec<_>>();
        parts.push(ir.call_named(&name, chunk_ty, &types, &args));
    }
    if parts.len() == 1 {
        parts[0]
    } else {
        assert_eq!(parts.len(), 2);
        let mask: Vec<u32> = (0..w).collect();
        ir.shuffle_by(parts[0], parts[1], &mask)
    }
}
