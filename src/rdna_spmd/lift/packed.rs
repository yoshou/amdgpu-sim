//! Packed integer semantics (RDNA4 ISA §16.10), expressed as word SSA.
//! Both selected source halves are captured before the packed destination write.
use super::*;

pub(super) fn instruction(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering<'static>> {
    let InstFormat::VOP3P(i) = inst else { return None; };
    if let Some(dot) = dot_integer(i, registry) { return Some(dot); }
    if let Some(float) = packed_float(i, registry) { return Some(float); }
    if let Some(mixed) = mixed_float(i, registry) { return Some(mixed); }
    if let Some(dot) = dot_float(i, registry) { return Some(dot); }
    let (arithmetic, signed, arity, saturates) = match i.op {
        I::V_PK_ADD_I16 => (IntOp::Add, true, 2, true),
        I::V_PK_SUB_I16 => (IntOp::Sub, true, 2, true),
        I::V_PK_MAD_I16 => (IntOp::Mul, true, 3, true),
        I::V_PK_ADD_U16 => (IntOp::Add, false, 2, true),
        I::V_PK_SUB_U16 => (IntOp::Sub, false, 2, true),
        I::V_PK_MAD_U16 => (IntOp::Mul, false, 3, true),
        I::V_PK_MUL_LO_U16 => (IntOp::Mul, false, 2, false),
        I::V_PK_LSHLREV_B16 => (IntOp::Shl, false, 2, false),
        I::V_PK_LSHRREV_B16 => (IntOp::LShr, false, 2, false),
        I::V_PK_ASHRREV_I16 => (IntOp::AShr, true, 2, false),
        I::V_PK_MIN_I16 | I::V_PK_MAX_I16 => (IntOp::Add, true, 2, false),
        I::V_PK_MIN_U16 | I::V_PK_MAX_U16 => (IntOp::Add, false, 2, false),
        _ => return None,
    };
    let mut b = Builder::new(registry, [&i.src0, &i.src1, &i.src2].iter().take(arity)
        .map(|s| input((*s).clone(), Ty::I32)).collect());
    let mut halves = vec![];
    for (select, neg) in [(i.opsel, i.neg), (i.opsel_hi | (i.opsel_hi2 << 2), i.neg_hi)] {
        let shift = b.k(Ty::I32, 16);
        let mask = b.k(Ty::I32, 0xffff);
        let mut args = vec![];
        for source in 0..arity {
            let raw = ValueId(source);
            let selected = if (select >> source) & 1 != 0 { b.int(IntOp::LShr, raw, shift) } else { raw };
            let selected = b.int(IntOp::And, selected, mask);
            // Preserve hardware-captured NEG/NEG_HI behavior on packed integer
            // encodings: toggle bit 15 before interpreting the signed half.
            let selected = b.bits_mod(Ty::I32, 16, selected, 0, neg, source);
            let selected = if signed {
                let shifted = b.int(IntOp::Shl, selected, shift);
                b.int(IntOp::AShr, shifted, shift)
            } else { selected };
            args.push(selected);
        }
        let mut result = match i.op {
            I::V_PK_MIN_I16 | I::V_PK_MIN_U16 | I::V_PK_MAX_I16 | I::V_PK_MAX_U16 => {
                let minimum = matches!(i.op, I::V_PK_MIN_I16 | I::V_PK_MIN_U16);
                let predicate = match (minimum, signed) {
                    (true, true) => IntPred::Slt, (true, false) => IntPred::Ult,
                    (false, true) => IntPred::Sgt, (false, false) => IntPred::Ugt,
                };
                let condition = b.push(Ty::I1, Op::Cmp(predicate, args[0], args[1]));
                b.push(Ty::I32, Op::Select(condition, args[0], args[1]))
            }
            I::V_PK_LSHLREV_B16 | I::V_PK_LSHRREV_B16 | I::V_PK_ASHRREV_I16 => {
                let mask = b.k(Ty::I32, 15);
                let amount = b.int(IntOp::And, args[0], mask);
                b.int(arithmetic, args[1], amount)
            }
            _ => {
                let value = b.int(arithmetic, args[0], args[1]);
                if arity == 3 { b.int(IntOp::Add, value, args[2]) } else { value }
            }
        };
        if saturates && i.cm != 0 {
            let low = b.k(Ty::I32, if signed { 0xffff_8000 } else { 0 });
            let high = b.k(Ty::I32, if signed { 0x7fff } else { 0xffff });
            // Unsigned multiplication plus an unsigned half still fits u32.
            // Unsigned subtraction alone needs a borrow check before clipping.
            if !signed && arithmetic == IntOp::Sub {
                let borrow = b.push(Ty::I1, Op::Cmp(IntPred::Ult, args[0], args[1]));
                result = b.push(Ty::I32, Op::Select(borrow, low, result));
            } else if signed {
                let below = b.push(Ty::I1, Op::Cmp(IntPred::Slt, result, low));
                result = b.push(Ty::I32, Op::Select(below, low, result));
            }
            let above = b.push(Ty::I1, Op::Cmp(if signed { IntPred::Sgt } else { IntPred::Ugt }, result, high));
            result = b.push(Ty::I32, Op::Select(above, high, result));
        }
        halves.push(b.int(IntOp::And, result, mask));
    }
    let shift = b.k(Ty::I32, 16);
    let high = b.int(IntOp::Shl, halves[1], shift);
    let packed = b.int(IntOp::Or, halves[0], high);
    Some(b.finish(Output::Vgpr(i.vdst as u32, Ty::I32), packed))
}

// §16.10: IU8/IU4 use NEG[0:1] as signedness selectors. OPSEL selects
// halves before unpacking; the accumulator is a complete, unmodified word.
fn dot_integer(i: &crate::rdna_instructions::VOP3P, registry: &DialectRegistry) -> Option<Lowering<'static>> {
    let (bits, signed) = match i.op {
        I::V_DOT4_I32_IU8 => (8, true), I::V_DOT4_U32_U8 => (8, false),
        I::V_DOT8_I32_IU4 => (4, true), I::V_DOT8_U32_U4 => (4, false),
        _ => return None,
    };
    let mut b = Builder::new(registry, vec![input(i.src0, Ty::I32), input(i.src1, Ty::I32), input(i.src2, Ty::I32)]);
    let mut sum = b.push(Ty::I64, Op::Convert(if signed { Cvt::SExt } else { Cvt::ZExt }, Ty::I64, ValueId(2)));
    for select in [i.opsel, i.opsel_hi | i.opsel_hi2 << 2] {
        for component in 0..16 / bits {
            let mut values = Vec::new();
            for source in 0..2 {
                let shift = b.k(Ty::I32, ((select >> source & 1) as u64) * 16 + component * bits);
                let value = b.int(IntOp::LShr, ValueId(source), shift);
                let mask = b.k(Ty::I32, (1 << bits) - 1);
                let mut value = b.int(IntOp::And, value, mask);
                let negative = signed && i.neg >> source & 1 != 0;
                if negative {
                    let shift = b.k(Ty::I32, 32 - bits);
                    value = b.int(IntOp::Shl, value, shift);
                    value = b.int(IntOp::AShr, value, shift);
                }
                values.push(b.push(Ty::I64, Op::Convert(if negative { Cvt::SExt } else { Cvt::ZExt }, Ty::I64, value)));
            }
            let product = b.push(Ty::I64, Op::Int(IntOp::Mul, values[0], values[1]));
            sum = b.push(Ty::I64, Op::Int(IntOp::Add, sum, product));
        }
    }
    if i.cm != 0 {
        let high = b.k(Ty::I64, if signed { i32::MAX as u64 } else { u32::MAX as u64 });
        let above = b.push(Ty::I1, Op::Cmp(if signed { IntPred::Sgt } else { IntPred::Ugt }, sum, high));
        sum = b.push(Ty::I64, Op::Select(above, high, sum));
        if signed {
            let low = b.k(Ty::I64, i32::MIN as i64 as u64);
            let below = b.push(Ty::I1, Op::Cmp(IntPred::Slt, sum, low));
            sum = b.push(Ty::I64, Op::Select(below, low, sum));
        }
    }
    let value = b.push(Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, sum));
    Some(b.finish(Output::Vgpr(i.vdst as u32, Ty::I32), value))
}

// F16 products have at most 22 significant bits. F64 retains every bit
// relevant to a finite F16 fused result. Jam an inexact intermediate F32
// conversion to odd before the provider's RNE half conversion, avoiding
// double rounding at half-precision midpoints.
fn narrow_wide_half(b: &mut Builder<'_>, value: ValueId) -> ValueId {
    let single = b.push(Ty::F32, Op::Convert(Cvt::FloatResizeRte, Ty::F32, value));
    let widened = b.push(Ty::F64, Op::Convert(Cvt::FloatResizeRte, Ty::F64, single));
    let magnitude = b.push(Ty::F64, Op::Unary(FloatUnary::Abs, value));
    let rounded = b.push(Ty::F64, Op::Unary(FloatUnary::Abs, widened));
    let rounded_up = b.push(Ty::I1, Op::FCmp(FloatPred::Ogt, rounded, magnitude));
    let inexact = b.push(Ty::I1, Op::FCmp(FloatPred::One, widened, value));
    let bits = b.push(Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, single));
    let one = b.k(Ty::I32, 1);
    let down = b.int(IntOp::Sub, bits, one);
    let truncated = b.push(Ty::I32, Op::Select(rounded_up, down, bits));
    let odd = b.int(IntOp::Or, truncated, one);
    let bits = b.push(Ty::I32, Op::Select(inexact, odd, bits));
    let single = b.push(Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, bits));
    let target = crate::rdna_spmd::dialect::rdna4::unary(b.registry, I::V_CVT_F16_F32).unwrap();
    b.target_one(target, Arguments::Unary(single))
}

fn packed_float(i: &crate::rdna_instructions::VOP3P, registry: &DialectRegistry) -> Option<Lowering<'static>> {
    let arity = match i.op {
        I::V_PK_FMA_F16 => 3,
        I::V_PK_ADD_F16 | I::V_PK_MUL_F16 | I::V_PK_MIN_NUM_F16 | I::V_PK_MAX_NUM_F16 |
        I::V_PK_MINIMUM_F16 | I::V_PK_MAXIMUM_F16 => 2,
        _ => return None,
    };
    let mut b = Builder::new(registry, [i.src0, i.src1, i.src2].iter().copied().take(arity)
        .map(|s| input(half::source(s), Ty::I32)).collect());
    let widen = crate::rdna_spmd::dialect::rdna4::unary(registry, I::V_CVT_F32_F16).unwrap();
    let narrow = crate::rdna_spmd::dialect::rdna4::unary(registry, I::V_CVT_F16_F32).unwrap();
    let mut results = Vec::new();
    for (select, neg) in [(i.opsel, i.neg), (i.opsel_hi | i.opsel_hi2 << 2, i.neg_hi)] {
        let mut values = Vec::new();
        for index in 0..arity {
            let mut value = ValueId(index);
            if select >> index & 1 != 0 { let shift = b.k(Ty::I32, 16); value = b.int(IntOp::LShr, value, shift); }
            value = b.bits_mod(Ty::I32, 16, value, 0, neg, index);
            values.push(b.target_one(widen, Arguments::Unary(value)));
        }
        let wide = matches!(i.op, I::V_PK_FMA_F16);
        let mut value = if wide {
            let args = values.iter().map(|&v| b.push(Ty::F64, Op::Convert(Cvt::FloatResizeRte, Ty::F64, v))).collect::<Vec<_>>();
            b.push(Ty::F64, Op::Fma(args[0], args[1], args[2]))
        } else {
            let op = match i.op { I::V_PK_ADD_F16 => FloatOp::Add, I::V_PK_MUL_F16 => FloatOp::Mul,
                I::V_PK_MIN_NUM_F16 | I::V_PK_MINIMUM_F16 => FloatOp::MinNum, _ => FloatOp::MaxNum };
            let mut value = b.push(Ty::F32, Op::Float(op, values[0], values[1]));
            if matches!(i.op, I::V_PK_MINIMUM_F16 | I::V_PK_MAXIMUM_F16) {
                for &arg in values.iter().rev() {
                    let nan = b.push(Ty::I1, Op::FCmp(FloatPred::Uno, arg, arg));
                    let bits = b.push(Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, arg));
                    let quiet = b.k(Ty::I32, 0x0040_0000);
                    let bits = b.int(IntOp::Or, bits, quiet);
                    let quiet = b.push(Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, bits));
                    value = b.push(Ty::F32, Op::Select(nan, quiet, value));
                }
            }
            value
        };
        // Clipping commutes with half RNE because 0 and 1 are exactly
        // representable, including the ISA's NaN-to-zero clamp rule.
        value = b.output_mod(if wide { Ty::F64 } else { Ty::F32 }, value, i.cm, 0);
        results.push(if wide { narrow_wide_half(&mut b, value) } else { b.target_one(narrow, Arguments::Unary(value)) });
    }
    let shift = b.k(Ty::I32, 16);
    let high = b.int(IntOp::Shl, results[1], shift);
    let value = b.int(IntOp::Or, results[0], high);
    Some(b.finish(Output::Vgpr(i.vdst as u32, Ty::I32), value))
}

// §16.10 MIX: OPSEL_HI chooses precision; OPSEL chooses the half only
// for F16 inputs. NEG_HI is ABS, and a half destination preserves its sibling.
fn mixed_float(i: &crate::rdna_instructions::VOP3P, registry: &DialectRegistry) -> Option<Lowering<'static>> {
    let (partial, high) = match i.op {
        I::V_FMA_MIX_F32 => (false, false), I::V_FMA_MIXLO_F16 => (true, false),
        I::V_FMA_MIXHI_F16 => (true, true), _ => return None,
    };
    let precision = i.opsel_hi | i.opsel_hi2 << 2;
    let mut inputs = [i.src0, i.src1, i.src2].iter().copied().enumerate().map(|(index, s)| {
        input(if precision >> index & 1 != 0 { half::source(s) } else { s }, Ty::I32)
    }).collect::<Vec<_>>();
    if partial { inputs.push(input(SourceOperand::VectorRegister(i.vdst), Ty::I32)); }
    let mut b = Builder::new(registry, inputs);
    let widen = crate::rdna_spmd::dialect::rdna4::unary(registry, I::V_CVT_F32_F16).unwrap();
    let mut values = Vec::new();
    for index in 0..3 {
        let mut value = ValueId(index);
        if precision >> index & 1 != 0 {
            if i.opsel >> index & 1 != 0 { let shift = b.k(Ty::I32, 16); value = b.int(IntOp::LShr, value, shift); }
            value = b.target_one(widen, Arguments::Unary(value));
        } else { value = b.push(Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, value)); }
        values.push(b.float_mod(Ty::F32, value, i.neg_hi, i.neg, index));
    }
    let value = b.push(Ty::F32, Op::Fma(values[0], values[1], values[2]));
    let mut value = b.output_mod(Ty::F32, value, i.cm, 0);
    if partial {
        let narrow = crate::rdna_spmd::dialect::rdna4::unary(registry, I::V_CVT_F16_F32).unwrap();
        value = b.target_one(narrow, Arguments::Unary(value));
        if high { let shift = b.k(Ty::I32, 16); value = b.int(IntOp::Shl, value, shift); }
        let keep = b.k(Ty::I32, if high { 0xffff } else { 0xffff_0000 });
        let old = b.int(IntOp::And, ValueId(3), keep);
        value = b.int(IntOp::Or, old, value);
    }
    Some(b.finish(Output::Vgpr(i.vdst as u32, if partial { Ty::I32 } else { Ty::F32 }), value))
}

// §16.10 dot products accumulate two exact short-format products and F32 C.
// Retaining the F64 summation residual avoids losing C when two large
// products cancel. Captured F32 results allow two ULP; CLAMP is ignored.
fn dot_float(i: &crate::rdna_instructions::VOP3P, registry: &DialectRegistry) -> Option<Lowering<'static>> {
    let bf16 = match i.op { I::V_DOT2_F32_F16 => false, I::V_DOT2_F32_BF16 => true, _ => return None };
    let mut b = Builder::new(registry, vec![input(i.src0, Ty::I32), input(i.src1, Ty::I32), input(i.src2, Ty::F32)]);
    let widen = crate::rdna_spmd::dialect::rdna4::unary(registry, I::V_CVT_F32_F16).unwrap();
    let mut products = Vec::new();
    for (select, neg) in [(i.opsel, i.neg), (i.opsel_hi | i.opsel_hi2 << 2, i.neg_hi)] {
        let mut values = Vec::new();
        for index in 0..2 {
            let mut value = ValueId(index);
            // BF16 floating inlines always select the FP32 upper half (§7.7.2).
            let source = if index == 0 { i.src0 } else { i.src1 };
            let high = if bf16 && matches!(source, SourceOperand::FloatConstant(_)) { true } else { select >> index & 1 != 0 };
            if high { let shift = b.k(Ty::I32, 16); value = b.int(IntOp::LShr, value, shift); }
            value = b.bits_mod(Ty::I32, 16, value, 0, neg, index);
            value = if bf16 {
                let shift = b.k(Ty::I32, 16);
                let bits = b.int(IntOp::Shl, value, shift);
                b.push(Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, bits))
            } else { b.target_one(widen, Arguments::Unary(value)) };
            values.push(b.push(Ty::F64, Op::Convert(Cvt::FloatResizeRte, Ty::F64, value)));
        }
        products.push(b.push(Ty::F64, Op::Float(FloatOp::Mul, values[0], values[1])));
    }
    let c = b.float_mod(Ty::F32, ValueId(2), 0, i.neg | i.neg_hi, 2);
    let c = b.push(Ty::F64, Op::Convert(Cvt::FloatResizeRte, Ty::F64, c));
    let (sum, residual) = two_sum(&mut b, products[0], products[1]);
    let (total, error) = two_sum(&mut b, sum, c);
    let error = b.push(Ty::F64, Op::Float(FloatOp::Add, error, residual));
    let corrected = b.push(Ty::F64, Op::Float(FloatOp::Add, total, error));
    // Residual arithmetic is undefined for infinities; preserve the ordinary
    // IEEE sum in that case, including NaN payload propagation.
    let magnitude = b.push(Ty::F64, Op::Unary(FloatUnary::Abs, total));
    let bound = b.k(Ty::F64, f64::MAX.to_bits());
    let finite = b.push(Ty::I1, Op::FCmp(FloatPred::Ole, magnitude, bound));
    let result = b.push(Ty::F64, Op::Select(finite, corrected, total));
    let result = b.push(Ty::F32, Op::Convert(Cvt::FloatResizeRte, Ty::F32, result));
    Some(b.finish(Output::Vgpr(i.vdst as u32, Ty::F32), result))
}

fn two_sum(b: &mut Builder<'_>, a: ValueId, c: ValueId) -> (ValueId, ValueId) {
    let sum = b.push(Ty::F64, Op::Float(FloatOp::Add, a, c));
    let c_virtual = b.push(Ty::F64, Op::Float(FloatOp::Sub, sum, a));
    let a_virtual = b.push(Ty::F64, Op::Float(FloatOp::Sub, sum, c_virtual));
    let a_error = b.push(Ty::F64, Op::Float(FloatOp::Sub, a, a_virtual));
    let c_error = b.push(Ty::F64, Op::Float(FloatOp::Sub, c, c_virtual));
    let error = b.push(Ty::F64, Op::Float(FloatOp::Add, a_error, c_error));
    (sum, error)
}
