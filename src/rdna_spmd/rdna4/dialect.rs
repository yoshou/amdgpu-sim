use crate::instructions::I;
use crate::rdna_spmd::dialect::{
    Dialect, DialectRegistry, Effect, Implementation, Operation, TargetOp,
};
use crate::rdna_spmd::ir::{FloatPred, IntPred};
use crate::rdna_spmd::native::{Type, Value};
use crate::rdna_spmd::{codegen::Emitter, ir::Ty};
mod bvh;
mod division;
mod image;
mod reduction;
mod scale;

pub const ID: u32 = 0x52444e34;
mod idioms;

pub use bvh::lowering_state;
pub use idioms::{DivisionIdioms, SqrtIdioms};
pub const REGISTERS: crate::rdna_spmd::dialect::Registers =
    crate::rdna_spmd::dialect::Registers {
        exec: 126,
        vcc: 106,
        null: 124,
        scc_slot: 128,
        sgprs: 128,
        vgprs: 256,
    };

pub fn register(registry: &mut Dialect) -> Result<(), &'static str> {
    registry.register(
        ID,
        36,
        Operation {
            name: "image_sample_lz",
            effect: Effect::ReadGlobal { every_lane: true },
            immediates: &[(12, 3)],
            inputs: &[
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I1,
                Ty::F32,
                Ty::F32,
            ],
            outputs: vec![Ty::I32],
            lower: Implementation::Single(image::sample),
        },
    )?;
    for (id, name, ty, lower) in [
        (
            1,
            "rcp.f32",
            Ty::F32,
            rcp_f32 as fn(&Emitter, &[Value]) -> Value,
        ),
        (2, "rcp.f64", Ty::F64, rcp_f64),
        (3, "rsq.f32", Ty::F32, rsq_f32),
        (4, "rsq.f64", Ty::F64, rsq_f64),
        (5, "sqrt.f32", Ty::F32, sqrt_f32),
        (6, "sqrt.f64", Ty::F64, sqrt_f64),
        (20, "floor.f32", Ty::F32, floor_f32),
        (21, "ceil.f32", Ty::F32, ceil_f32),
        (22, "trunc.f32", Ty::F32, trunc_f32),
        (23, "rndne.f32", Ty::F32, round_f32),
        (24, "rndne.f64", Ty::F64, round_f64),
        (25, "fract.f64", Ty::F64, fract_f64),
        (26, "floor.f64", Ty::F64, floor_f64),
        (27, "trunc.f64", Ty::F64, trunc_f64),
        (28, "exp2.f32", Ty::F32, exp2_f32),
        (29, "log2.f32", Ty::F32, log2_f32),
        (30, "sin.f32", Ty::F32, sin_f32),
        (31, "cos.f32", Ty::F32, cos_f32),
    ] {
        registry.register(
            ID,
            id,
            Operation {
                effect: Effect::Pure,
                immediates: &[],
                name,
                inputs: if ty == Ty::F32 {
                    &[Ty::F32]
                } else {
                    &[Ty::F64]
                },
                outputs: vec![ty],
                lower: Implementation::Single(lower),
            },
        )?;
    }
    for (id, name, input, output, lower) in [
        (
            9,
            "frexp_mant.f32",
            Ty::F32,
            Ty::F32,
            mant_f32 as fn(&Emitter, &[Value]) -> Value,
        ),
        (10, "frexp_mant.f64", Ty::F64, Ty::F64, mant_f64),
        (11, "frexp_exp.f32", Ty::F32, Ty::I32, exp_f32),
        (12, "frexp_exp.f64", Ty::F64, Ty::I32, exp_f64),
    ] {
        registry.register(
            ID,
            id,
            Operation {
                effect: Effect::Pure,
                immediates: &[],
                name,
                inputs: if input == Ty::F32 {
                    &[Ty::F32]
                } else {
                    &[Ty::F64]
                },
                outputs: vec![output],
                lower: Implementation::Single(lower),
            },
        )?;
    }
    registry.register(
        ID,
        32,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "cmp_class.f32",
            inputs: &[Ty::F32, Ty::I32],
            outputs: vec![Ty::I1],
            lower: Implementation::Single(class_f32),
        },
    )?;
    registry.register(
        ID,
        33,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "cmp_class.f64",
            inputs: &[Ty::F64, Ty::I32],
            outputs: vec![Ty::I1],
            lower: Implementation::Single(class_f64),
        },
    )?;
    registry.register(
        ID,
        34,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "cvt.f32.f16",
            inputs: &[Ty::I32],
            outputs: vec![Ty::F32],
            lower: Implementation::Single(from_half),
        },
    )?;
    registry.register(
        ID,
        35,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "cvt.f16.f32",
            inputs: &[Ty::F32],
            outputs: vec![Ty::I32],
            lower: Implementation::Single(to_half),
        },
    )?;
    registry.register(
        ID,
        7,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "ldexp.f32",
            inputs: &[Ty::F32, Ty::I32],
            outputs: vec![Ty::F32],
            lower: Implementation::Single(scale::f32),
        },
    )?;
    registry.register(
        ID,
        8,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "ldexp.f64",
            inputs: &[Ty::F64, Ty::I32],
            outputs: vec![Ty::F64],
            lower: Implementation::Single(scale::f64),
        },
    )?;
    registry.register(
        ID,
        13,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "trig_preop.f64",
            inputs: &[Ty::F64, Ty::I32],
            outputs: vec![Ty::F64],
            lower: Implementation::Single(reduction::lower),
        },
    )?;
    registry.register(
        ID,
        17,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "div_fixup.f64",
            inputs: &[Ty::F64, Ty::F64, Ty::F64],
            outputs: vec![Ty::F64],
            lower: Implementation::Single(division::fixup_f64),
        },
    )?;
    registry.register(
        ID,
        16,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "div_fixup.f32",
            inputs: &[Ty::F32, Ty::F32, Ty::F32],
            outputs: vec![Ty::F32],
            lower: Implementation::Single(division::fixup_f32),
        },
    )?;
    registry.register(
        ID,
        18,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "div_fmas.f32",
            inputs: &[Ty::F32, Ty::F32, Ty::F32, Ty::I1],
            outputs: vec![Ty::F32],
            lower: Implementation::Single(division::fmas_f32),
        },
    )?;
    registry.register(
        ID,
        19,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "div_fmas.f64",
            inputs: &[Ty::F64, Ty::F64, Ty::F64, Ty::I1],
            outputs: vec![Ty::F64],
            lower: Implementation::Single(division::fmas_f64),
        },
    )?;
    registry.register(
        ID,
        14,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "div_scale.f32",
            inputs: &[Ty::F32, Ty::F32, Ty::F32],
            outputs: vec![Ty::F32, Ty::I1],
            lower: Implementation::Multiple(division::scale_f32),
        },
    )?;
    registry.register(
        ID,
        15,
        Operation {
            effect: Effect::Pure,
            immediates: &[],
            name: "div_scale.f64",
            inputs: &[Ty::F64, Ty::F64, Ty::F64],
            outputs: vec![Ty::F64, Ty::I1],
            lower: Implementation::Multiple(division::scale_f64),
        },
    )?;
    registry.register(
        ID,
        38,
        Operation {
            effect: Effect::ReadGlobal { every_lane: false },
            immediates: &[],
            name: "image_bvh8_intersect_ray",
            inputs: &[
                Ty::I32,
                Ty::I32,
                Ty::I64,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I1,
            ],
            outputs: vec![Ty::I32; 10],
            lower: Implementation::Multiple(bvh::lower8),
        },
    )?;
    registry.register(
        ID,
        37,
        Operation {
            effect: Effect::ReadGlobal { every_lane: false },
            immediates: &[],
            name: "image_bvh64_intersect_ray",
            inputs: &[
                Ty::I32,
                Ty::I32,
                Ty::I64,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I32,
                Ty::I1,
            ],
            outputs: vec![Ty::I32; 4],
            lower: Implementation::Multiple(bvh::lower),
        },
    )?;
    Ok(())
}

pub fn image_sample(registry: &DialectRegistry) -> TargetOp {
    registry
        .lookup(ID, "image_sample_lz")
        .expect("missing RDNA4 image provider")
}

pub fn comparison(registry: &DialectRegistry, ty: Ty) -> TargetOp {
    registry
        .lookup(
            ID,
            match ty {
                Ty::F32 => "cmp_class.f32",
                Ty::F64 => "cmp_class.f64",
                _ => unreachable!(),
            },
        )
        .expect("missing RDNA4 comparison provider")
}

fn classify(e: &Emitter, ty: Ty, value: Value, selector: Value) -> Value {
    let ir = e.ir();
    let mut result = e.constant(Ty::I1, 0);
    for index in 0..10 {
        let class = e.call(
            &format!("llvm.is.fpclass.{}", e.suffix(ty)),
            Ty::I1,
            &[value, ir.ci32(1 << index)],
        );
        let requested = ir.and(selector, e.constant(Ty::I32, 1 << index));
        let requested = ir.icmp(IntPred::Ne, requested, e.constant(Ty::I32, 0));
        result = ir.or(result, ir.and(class, requested));
    }
    result
}
fn class_f32(e: &Emitter, a: &[Value]) -> Value {
    classify(e, Ty::F32, a[0], a[1])
}
fn class_f64(e: &Emitter, a: &[Value]) -> Value {
    classify(e, Ty::F64, a[0], a[1])
}

pub fn unary(registry: &DialectRegistry, op: I) -> Option<TargetOp> {
    let name = match op {
        I::V_RCP_F32 | I::V_RCP_IFLAG_F32 | I::V_S_RCP_F32 => "rcp.f32",
        I::V_RCP_F64 => "rcp.f64",
        I::V_RSQ_F32 | I::V_S_RSQ_F32 => "rsq.f32",
        I::V_RSQ_F64 => "rsq.f64",
        I::V_SQRT_F32 | I::V_S_SQRT_F32 => "sqrt.f32",
        I::V_SQRT_F64 => "sqrt.f64",
        I::V_FLOOR_F32 => "floor.f32",
        I::V_FLOOR_F64 => "floor.f64",
        I::V_CEIL_F32 => "ceil.f32",
        I::V_TRUNC_F32 => "trunc.f32",
        I::V_TRUNC_F64 => "trunc.f64",
        I::V_RNDNE_F32 => "rndne.f32",
        I::V_RNDNE_F64 => "rndne.f64",
        I::V_FRACT_F64 => "fract.f64",
        I::V_FREXP_MANT_F32 => "frexp_mant.f32",
        I::V_FREXP_MANT_F64 => "frexp_mant.f64",
        I::V_FREXP_EXP_I32_F32 => "frexp_exp.f32",
        I::V_FREXP_EXP_I32_F64 => "frexp_exp.f64",
        I::V_CVT_F32_F16 => "cvt.f32.f16",
        I::V_CVT_F16_F32 => "cvt.f16.f32",
        I::V_EXP_F32 | I::V_S_EXP_F32 => "exp2.f32",
        I::V_LOG_F32 | I::V_S_LOG_F32 => "log2.f32",
        I::V_SIN_F32 => "sin.f32",
        I::V_COS_F32 => "cos.f32",
        _ => return None,
    };
    Some(registry.lookup(ID, name).expect("missing RDNA4 provider"))
}

pub fn binary(registry: &DialectRegistry, op: I) -> Option<TargetOp> {
    let name = match op {
        I::V_LDEXP_F32 => "ldexp.f32",
        I::V_LDEXP_F64 => "ldexp.f64",
        I::V_TRIG_PREOP_F64 => "trig_preop.f64",
        _ => return None,
    };
    Some(registry.lookup(ID, name).expect("missing RDNA4 provider"))
}

pub fn division(registry: &DialectRegistry, op: I) -> Option<TargetOp> {
    let name = match op {
        I::V_DIV_FIXUP_F32 => "div_fixup.f32",
        I::V_DIV_FIXUP_F64 => "div_fixup.f64",
        I::V_DIV_SCALE_F32 => "div_scale.f32",
        I::V_DIV_SCALE_F64 => "div_scale.f64",
        I::V_DIV_FMAS_F32 => "div_fmas.f32",
        I::V_DIV_FMAS_F64 => "div_fmas.f64",
        _ => return None,
    };
    Some(registry.lookup(ID, name).expect("missing RDNA4 provider"))
}

fn flush(e: &Emitter, value: Value) -> Value {
    let ir = e.ir();

    let tiny = e.call(
        &format!("llvm.is.fpclass.{}", e.suffix(Ty::F32)),
        Ty::I1,
        &[value, ir.ci32(0x90)],
    );
    let bits = ir.bitcast(value, e.ty(Ty::I32));
    let sign = ir.and(bits, e.constant(Ty::I32, 0x8000_0000));
    ir.bitcast(ir.select(tiny, sign, bits), e.ty(Ty::F32))
}

fn math(e: &Emitter, ty: Ty, mut value: Value, sqrt: bool, reciprocal: bool) -> Value {
    if ty == Ty::F32 {
        value = flush(e, value);
    }
    if sqrt {
        value = e.call(&format!("llvm.sqrt.{}", e.suffix(ty)), ty, &[value]);
    }
    if reciprocal {
        let one = e.constant(
            ty,
            if ty == Ty::F32 {
                1f32.to_bits() as u64
            } else {
                1f64.to_bits()
            },
        );
        value = e.ir().fdiv(one, value);
    }
    if ty == Ty::F32 {
        value = flush(e, value);
    }
    value
}
fn rcp_f32(e: &Emitter, a: &[Value]) -> Value {
    math(e, Ty::F32, a[0], false, true)
}
fn rcp_f64(e: &Emitter, a: &[Value]) -> Value {
    math(e, Ty::F64, a[0], false, true)
}
fn rsq_f32(e: &Emitter, a: &[Value]) -> Value {
    math(e, Ty::F32, a[0], true, true)
}
fn rsq_f64(e: &Emitter, a: &[Value]) -> Value {
    math(e, Ty::F64, a[0], true, true)
}
fn sqrt_f32(e: &Emitter, a: &[Value]) -> Value {
    math(e, Ty::F32, a[0], true, false)
}
fn sqrt_f64(e: &Emitter, a: &[Value]) -> Value {
    math(e, Ty::F64, a[0], true, false)
}

fn exp2_f32(e: &Emitter, a: &[Value]) -> Value {
    flush(
        e,
        e.call(
            &format!("llvm.exp2.{}", e.suffix(Ty::F32)),
            Ty::F32,
            &[flush(e, a[0])],
        ),
    )
}
fn log2_f32(e: &Emitter, a: &[Value]) -> Value {
    let input = flush(e, a[0]);
    let result = flush(
        e,
        e.call(
            &format!("llvm.log2.{}", e.suffix(Ty::F32)),
            Ty::F32,
            &[input],
        ),
    );

    let negative = e.ir().fcmp(FloatPred::Olt, input, e.constant(Ty::F32, 0));
    e.ir().select(negative, e.constant(Ty::F32, 0xffc0_0000), result)
}

fn trig(e: &Emitter, value: Value, cosine: bool) -> Value {
    let ir = e.ir();
    let rounded = e.call(
        &format!("llvm.roundeven.{}", e.suffix(Ty::F32)),
        Ty::F32,
        &[value],
    );
    let reduced = ir.fsub(value, rounded);
    let wide = ir.fpext(reduced, e.ty(Ty::F64));
    let angle = ir.fmul(wide, e.constant(Ty::F64, std::f64::consts::TAU.to_bits()));
    let op = if cosine { "cos" } else { "sin" };
    let result = e.call(
        &format!("llvm.{op}.{}", e.suffix(Ty::F64)),
        Ty::F64,
        &[angle],
    );
    let mut result = ir.fptrunc(result, e.ty(Ty::F32));
    let abs = e.call(
        &format!("llvm.fabs.{}", e.suffix(Ty::F32)),
        Ty::F32,
        &[reduced],
    );
    let zero = e.constant(Ty::F32, 0);
    let exact_zero = ir.fcmp(
        FloatPred::Oeq,
        abs,
        e.constant(
            Ty::F32,
            if cosine { 0.25f32 } else { 0.5f32 }.to_bits() as u64,
        ),
    );
    result = ir.select(exact_zero, zero, result);
    if !cosine {
        let integral = ir.fcmp(FloatPred::Oeq, abs, zero);
        result = ir.select(integral, zero, result);
        let input_zero = ir.fcmp(FloatPred::Oeq, value, zero);
        result = ir.select(input_zero, value, result);
    }
    result
}
fn sin_f32(e: &Emitter, a: &[Value]) -> Value {
    trig(e, a[0], false)
}
fn cos_f32(e: &Emitter, a: &[Value]) -> Value {
    trig(e, a[0], true)
}

fn rounding(e: &Emitter, ty: Ty, op: &str, a: Value) -> Value {
    e.call(&format!("llvm.{op}.{}", e.suffix(ty)), ty, &[a])
}
fn floor_f32(e: &Emitter, a: &[Value]) -> Value {
    rounding(e, Ty::F32, "floor", a[0])
}
fn floor_f64(e: &Emitter, a: &[Value]) -> Value {
    rounding(e, Ty::F64, "floor", a[0])
}
fn ceil_f32(e: &Emitter, a: &[Value]) -> Value {
    rounding(e, Ty::F32, "ceil", a[0])
}
fn trunc_f32(e: &Emitter, a: &[Value]) -> Value {
    rounding(e, Ty::F32, "trunc", a[0])
}
fn trunc_f64(e: &Emitter, a: &[Value]) -> Value {
    rounding(e, Ty::F64, "trunc", a[0])
}
fn round_f32(e: &Emitter, a: &[Value]) -> Value {
    rounding(e, Ty::F32, "roundeven", a[0])
}
fn round_f64(e: &Emitter, a: &[Value]) -> Value {
    rounding(e, Ty::F64, "roundeven", a[0])
}
fn fract_f64(e: &Emitter, a: &[Value]) -> Value {
    let floor = floor_f64(e, a);
    let value = e.ir().fsub(a[0], floor);

    let cap = e.constant(Ty::F64, 0x3fef_ffff_ffff_ffff);
    let over = e.ir().fcmp(FloatPred::Ogt, value, cap);
    e.ir().select(over, cap, value)
}

fn frexp(e: &Emitter, ty: Ty, input: Value) -> (Value, Value) {
    let ir = e.ir();
    let (word, fraction_bits, exponent_bits, bias) = if ty == Ty::F32 {
        (Ty::I32, 23, 8, 127)
    } else {
        (Ty::I64, 52, 11, 1023)
    };
    let k = |x| e.constant(word, x);
    let eq = |a, b| ir.icmp(IntPred::Eq, a, b);
    let bits = ir.bitcast(input, e.ty(word));
    let sign = ir.and(bits, k(1u64 << (word.bits() - 1)));
    let fraction = ir.and(bits, k((1u64 << fraction_bits) - 1));
    let exponent = ir.and(ir.lshr(bits, k(fraction_bits)), k((1 << exponent_bits) - 1));
    let exp_zero = eq(exponent, k(0));
    let special = eq(exponent, k((1 << exponent_bits) - 1));
    let fraction_zero = eq(fraction, k(0));
    let zero = ir.and(exp_zero, fraction_zero);
    let lz = e.call(
        &format!("llvm.ctlz.{}", e.suffix(word)),
        word,
        &[fraction, ir.ci1(false)],
    );
    let shift = ir.sub(lz, k(exponent_bits));

    let safe_shift = ir.and(shift, k(word.bits() as u64 - 1));
    let normalized = ir.and(ir.shl(fraction, safe_shift), k((1u64 << fraction_bits) - 1));
    let mantissa = ir.select(exp_zero, normalized, fraction);
    let mantissa = ir.or(ir.or(sign, mantissa), k((bias - 1) << fraction_bits));
    let nan = ir.and(special, ir.not(fraction_zero));
    let quiet = ir.or(bits, k(1 << (fraction_bits - 1)));
    let special_bits = ir.select(nan, quiet, bits);
    let mantissa = ir.select(ir.or(zero, special), special_bits, mantissa);
    let normal_exp = ir.sub(exponent, k(bias - 1));
    let sub_exp = ir.sub(ir.sub(k(2), k(bias)), shift);
    let exp = ir.select(exp_zero, sub_exp, normal_exp);
    let exp = ir.select(ir.or(zero, special), k(0), exp);
    let exp = if word == Ty::I64 {
        ir.trunc(exp, e.ty(Ty::I32))
    } else {
        exp
    };
    (ir.bitcast(mantissa, e.ty(ty)), exp)
}
fn mant_f32(e: &Emitter, a: &[Value]) -> Value {
    frexp(e, Ty::F32, a[0]).0
}
fn mant_f64(e: &Emitter, a: &[Value]) -> Value {
    frexp(e, Ty::F64, a[0]).0
}
fn exp_f32(e: &Emitter, a: &[Value]) -> Value {
    frexp(e, Ty::F32, a[0]).1
}
fn exp_f64(e: &Emitter, a: &[Value]) -> Value {
    frexp(e, Ty::F64, a[0]).1
}

fn from_half(e: &Emitter, a: &[Value]) -> Value {
    let ir = e.ir();
    let i16 = e.shaped(ir.i16());
    let f16 = e.shaped(ir.f16());
    let bits = ir.trunc(a[0], i16);
    let value = ir.bitcast(bits, f16);
    ir.fpext(value, e.ty(Ty::F32))
}
fn to_half(e: &Emitter, a: &[Value]) -> Value {
    let ir = e.ir();
    let i16 = e.shaped(ir.i16());
    let f16 = e.shaped(ir.f16());
    let value = ir.fptrunc(a[0], f16);
    let bits = ir.bitcast(value, i16);
    ir.zext(bits, e.ty(Ty::I32))
}

pub fn bvh(registry: &DialectRegistry) -> TargetOp {
    registry
        .lookup(ID, "image_bvh64_intersect_ray")
        .expect("missing BVH provider")
}
pub fn bvh8(registry: &DialectRegistry) -> TargetOp {
    registry
        .lookup(ID, "image_bvh8_intersect_ray")
        .expect("missing BVH8 provider")
}
