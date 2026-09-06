//! Comparison semantics shared by VOPC and VOP3, before mask writeback.
use super::*;
use crate::instructions::{OP8, OP16};

#[derive(Clone, Copy)]
enum Predicate { Float(OP16), Integer(OP8, bool), Class }
#[derive(Clone, Copy)]
struct Comparison { bits: u32, predicate: Predicate, exec: bool }

fn decode(op: I) -> Option<Comparison> {
    use Predicate::*;
    let (bits, predicate, exec) = match op {
        I::V_CMP_F32(p) => (32, Float(p), false),
        I::V_CMP_LT_F32 => (32, Float(OP16::LT), false),
        I::V_CMP_EQ_F32 => (32, Float(OP16::EQ), false),
        I::V_CMP_LE_F32 => (32, Float(OP16::LE), false),
        I::V_CMP_GT_F32 => (32, Float(OP16::GT), false),
        I::V_CMP_LG_F32 => (32, Float(OP16::LG), false),
        I::V_CMP_GE_F32 => (32, Float(OP16::GE), false),
        I::V_CMP_O_F32 => (32, Float(OP16::O), false),
        I::V_CMP_U_F32 => (32, Float(OP16::U), false),
        I::V_CMP_NGE_F32 => (32, Float(OP16::NGE), false),
        I::V_CMP_NLG_F32 => (32, Float(OP16::NLG), false),
        I::V_CMP_NGT_F32 => (32, Float(OP16::NGT), false),
        I::V_CMP_NLE_F32 => (32, Float(OP16::NLE), false),
        I::V_CMP_NEQ_F32 => (32, Float(OP16::NEQ), false),
        I::V_CMP_NLT_F32 => (32, Float(OP16::NLT), false),
        I::V_CMPX_F32(p) => (32, Float(p), true),
        I::V_CMPX_LT_F32 => (32, Float(OP16::LT), true),
        I::V_CMPX_EQ_F32 => (32, Float(OP16::EQ), true),
        I::V_CMPX_LE_F32 => (32, Float(OP16::LE), true),
        I::V_CMPX_GT_F32 => (32, Float(OP16::GT), true),
        I::V_CMPX_LG_F32 => (32, Float(OP16::LG), true),
        I::V_CMPX_GE_F32 => (32, Float(OP16::GE), true),
        I::V_CMPX_O_F32 => (32, Float(OP16::O), true),
        I::V_CMPX_U_F32 => (32, Float(OP16::U), true),
        I::V_CMPX_NGE_F32 => (32, Float(OP16::NGE), true),
        I::V_CMPX_NLG_F32 => (32, Float(OP16::NLG), true),
        I::V_CMPX_NGT_F32 => (32, Float(OP16::NGT), true),
        I::V_CMPX_NLE_F32 => (32, Float(OP16::NLE), true),
        I::V_CMPX_NEQ_F32 => (32, Float(OP16::NEQ), true),
        I::V_CMPX_NLT_F32 => (32, Float(OP16::NLT), true),
        I::V_CMP_F64(p) => (64, Float(p), false),
        I::V_CMP_LT_F64 => (64, Float(OP16::LT), false),
        I::V_CMP_EQ_F64 => (64, Float(OP16::EQ), false),
        I::V_CMP_LE_F64 => (64, Float(OP16::LE), false),
        I::V_CMP_GT_F64 => (64, Float(OP16::GT), false),
        I::V_CMP_LG_F64 => (64, Float(OP16::LG), false),
        I::V_CMP_GE_F64 => (64, Float(OP16::GE), false),
        I::V_CMP_O_F64 => (64, Float(OP16::O), false),
        I::V_CMP_U_F64 => (64, Float(OP16::U), false),
        I::V_CMP_NGE_F64 => (64, Float(OP16::NGE), false),
        I::V_CMP_NLG_F64 => (64, Float(OP16::NLG), false),
        I::V_CMP_NGT_F64 => (64, Float(OP16::NGT), false),
        I::V_CMP_NLE_F64 => (64, Float(OP16::NLE), false),
        I::V_CMP_NEQ_F64 => (64, Float(OP16::NEQ), false),
        I::V_CMP_NLT_F64 => (64, Float(OP16::NLT), false),
        I::V_CMPX_F64(p) => (64, Float(p), true),
        I::V_CMPX_LT_F64 => (64, Float(OP16::LT), true),
        I::V_CMPX_EQ_F64 => (64, Float(OP16::EQ), true),
        I::V_CMPX_LE_F64 => (64, Float(OP16::LE), true),
        I::V_CMPX_GT_F64 => (64, Float(OP16::GT), true),
        I::V_CMPX_LG_F64 => (64, Float(OP16::LG), true),
        I::V_CMPX_GE_F64 => (64, Float(OP16::GE), true),
        I::V_CMPX_O_F64 => (64, Float(OP16::O), true),
        I::V_CMPX_U_F64 => (64, Float(OP16::U), true),
        I::V_CMPX_NGE_F64 => (64, Float(OP16::NGE), true),
        I::V_CMPX_NLG_F64 => (64, Float(OP16::NLG), true),
        I::V_CMPX_NGT_F64 => (64, Float(OP16::NGT), true),
        I::V_CMPX_NLE_F64 => (64, Float(OP16::NLE), true),
        I::V_CMPX_NEQ_F64 => (64, Float(OP16::NEQ), true),
        I::V_CMPX_NLT_F64 => (64, Float(OP16::NLT), true),
        I::V_CMP_I16(p) => (16, Integer(p, true), false),
        I::V_CMP_LT_I16 => (16, Integer(OP8::LT, true), false),
        I::V_CMP_EQ_I16 => (16, Integer(OP8::EQ, true), false),
        I::V_CMP_LE_I16 => (16, Integer(OP8::LE, true), false),
        I::V_CMP_GT_I16 => (16, Integer(OP8::GT, true), false),
        I::V_CMP_NE_I16 => (16, Integer(OP8::LG, true), false),
        I::V_CMP_GE_I16 => (16, Integer(OP8::GE, true), false),
        I::V_CMPX_I16(p) => (16, Integer(p, true), true),
        I::V_CMPX_LT_I16 => (16, Integer(OP8::LT, true), true),
        I::V_CMPX_EQ_I16 => (16, Integer(OP8::EQ, true), true),
        I::V_CMPX_LE_I16 => (16, Integer(OP8::LE, true), true),
        I::V_CMPX_GT_I16 => (16, Integer(OP8::GT, true), true),
        I::V_CMPX_NE_I16 => (16, Integer(OP8::LG, true), true),
        I::V_CMPX_GE_I16 => (16, Integer(OP8::GE, true), true),
        I::V_CMP_U16(p) => (16, Integer(p, false), false),
        I::V_CMP_LT_U16 => (16, Integer(OP8::LT, false), false),
        I::V_CMP_EQ_U16 => (16, Integer(OP8::EQ, false), false),
        I::V_CMP_LE_U16 => (16, Integer(OP8::LE, false), false),
        I::V_CMP_GT_U16 => (16, Integer(OP8::GT, false), false),
        I::V_CMP_NE_U16 => (16, Integer(OP8::LG, false), false),
        I::V_CMP_GE_U16 => (16, Integer(OP8::GE, false), false),
        I::V_CMPX_U16(p) => (16, Integer(p, false), true),
        I::V_CMPX_LT_U16 => (16, Integer(OP8::LT, false), true),
        I::V_CMPX_EQ_U16 => (16, Integer(OP8::EQ, false), true),
        I::V_CMPX_LE_U16 => (16, Integer(OP8::LE, false), true),
        I::V_CMPX_GT_U16 => (16, Integer(OP8::GT, false), true),
        I::V_CMPX_NE_U16 => (16, Integer(OP8::LG, false), true),
        I::V_CMPX_GE_U16 => (16, Integer(OP8::GE, false), true),
        I::V_CMP_I32(p) => (32, Integer(p, true), false),
        I::V_CMP_LT_I32 => (32, Integer(OP8::LT, true), false),
        I::V_CMP_EQ_I32 => (32, Integer(OP8::EQ, true), false),
        I::V_CMP_LE_I32 => (32, Integer(OP8::LE, true), false),
        I::V_CMP_GT_I32 => (32, Integer(OP8::GT, true), false),
        I::V_CMP_NE_I32 => (32, Integer(OP8::LG, true), false),
        I::V_CMP_GE_I32 => (32, Integer(OP8::GE, true), false),
        I::V_CMPX_I32(p) => (32, Integer(p, true), true),
        I::V_CMPX_LT_I32 => (32, Integer(OP8::LT, true), true),
        I::V_CMPX_EQ_I32 => (32, Integer(OP8::EQ, true), true),
        I::V_CMPX_LE_I32 => (32, Integer(OP8::LE, true), true),
        I::V_CMPX_GT_I32 => (32, Integer(OP8::GT, true), true),
        I::V_CMPX_NE_I32 => (32, Integer(OP8::LG, true), true),
        I::V_CMPX_GE_I32 => (32, Integer(OP8::GE, true), true),
        I::V_CMP_U32(p) => (32, Integer(p, false), false),
        I::V_CMP_LT_U32 => (32, Integer(OP8::LT, false), false),
        I::V_CMP_EQ_U32 => (32, Integer(OP8::EQ, false), false),
        I::V_CMP_LE_U32 => (32, Integer(OP8::LE, false), false),
        I::V_CMP_GT_U32 => (32, Integer(OP8::GT, false), false),
        I::V_CMP_NE_U32 => (32, Integer(OP8::LG, false), false),
        I::V_CMP_GE_U32 => (32, Integer(OP8::GE, false), false),
        I::V_CMPX_U32(p) => (32, Integer(p, false), true),
        I::V_CMPX_LT_U32 => (32, Integer(OP8::LT, false), true),
        I::V_CMPX_EQ_U32 => (32, Integer(OP8::EQ, false), true),
        I::V_CMPX_LE_U32 => (32, Integer(OP8::LE, false), true),
        I::V_CMPX_GT_U32 => (32, Integer(OP8::GT, false), true),
        I::V_CMPX_NE_U32 => (32, Integer(OP8::LG, false), true),
        I::V_CMPX_GE_U32 => (32, Integer(OP8::GE, false), true),
        I::V_CMP_I64(p) => (64, Integer(p, true), false),
        I::V_CMP_LT_I64 => (64, Integer(OP8::LT, true), false),
        I::V_CMP_EQ_I64 => (64, Integer(OP8::EQ, true), false),
        I::V_CMP_LE_I64 => (64, Integer(OP8::LE, true), false),
        I::V_CMP_GT_I64 => (64, Integer(OP8::GT, true), false),
        I::V_CMP_NE_I64 => (64, Integer(OP8::LG, true), false),
        I::V_CMP_GE_I64 => (64, Integer(OP8::GE, true), false),
        I::V_CMPX_I64(p) => (64, Integer(p, true), true),
        I::V_CMPX_LT_I64 => (64, Integer(OP8::LT, true), true),
        I::V_CMPX_EQ_I64 => (64, Integer(OP8::EQ, true), true),
        I::V_CMPX_LE_I64 => (64, Integer(OP8::LE, true), true),
        I::V_CMPX_GT_I64 => (64, Integer(OP8::GT, true), true),
        I::V_CMPX_NE_I64 => (64, Integer(OP8::LG, true), true),
        I::V_CMPX_GE_I64 => (64, Integer(OP8::GE, true), true),
        I::V_CMP_U64(p) => (64, Integer(p, false), false),
        I::V_CMP_LT_U64 => (64, Integer(OP8::LT, false), false),
        I::V_CMP_EQ_U64 => (64, Integer(OP8::EQ, false), false),
        I::V_CMP_LE_U64 => (64, Integer(OP8::LE, false), false),
        I::V_CMP_GT_U64 => (64, Integer(OP8::GT, false), false),
        I::V_CMP_NE_U64 => (64, Integer(OP8::LG, false), false),
        I::V_CMP_GE_U64 => (64, Integer(OP8::GE, false), false),
        I::V_CMPX_U64(p) => (64, Integer(p, false), true),
        I::V_CMPX_LT_U64 => (64, Integer(OP8::LT, false), true),
        I::V_CMPX_EQ_U64 => (64, Integer(OP8::EQ, false), true),
        I::V_CMPX_LE_U64 => (64, Integer(OP8::LE, false), true),
        I::V_CMPX_GT_U64 => (64, Integer(OP8::GT, false), true),
        I::V_CMPX_NE_U64 => (64, Integer(OP8::LG, false), true),
        I::V_CMPX_GE_U64 => (64, Integer(OP8::GE, false), true),
        I::V_CMP_CLASS_F16 => (16, Class, false),
        I::V_CMPX_CLASS_F16 => (16, Class, true),
        I::V_CMP_CLASS_F32 => (32, Class, false),
        I::V_CMPX_CLASS_F32 => (32, Class, true),
        I::V_CMP_CLASS_F64 => (64, Class, false),
        I::V_CMPX_CLASS_F64 => (64, Class, true),
        _ => return None,
    };
    Some(Comparison { bits, predicate, exec })
}

pub(super) fn instruction(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering<'static>> {
    let (op, sources, dst, abs, neg, opsel) = match inst {
        InstFormat::VOPC(i) => (i.op, [i.src0.clone(), SourceOperand::VectorRegister(i.vsrc1)], 106, 0, 0, 0),
        InstFormat::VOP3(i) => (i.op, [i.src0.clone(), i.src1.clone()], i.vdst as u32, i.abs, i.neg, i.opsel),
        _ => return None,
    };
    let comparison = decode(op)?;
    let ty = match comparison.predicate {
        Predicate::Float(_) => if comparison.bits == 64 { Ty::F64 } else { Ty::F32 },
        _ => if comparison.bits == 64 { Ty::I64 } else { Ty::I32 },
    };
    let mut b = Builder::new(registry, IntoIterator::into_iter(sources).enumerate().map(|(i, source)| {
        let ty = if i == 1 && matches!(comparison.predicate, Predicate::Class) { Ty::I32 } else { ty };
        if matches!(comparison.predicate, Predicate::Integer(_, true)) { signed_input(source, ty) }
        else { input(source, ty) }
    }).collect());
    let (mut a, mut c) = (ValueId(0), ValueId(1));
    let result = match comparison.predicate {
        Predicate::Float(p) => {
            a = b.float_mod(ty, a, abs, neg, 0);
            c = b.float_mod(ty, c, abs, neg, 1);
            let pred = match p {
                OP16::F => { let result = b.k(Ty::I1, 0); return Some(b.finish(Output::Compare(if comparison.exec {126} else {dst}), result)); },
                OP16::TRU => { let result = b.k(Ty::I1, 1); return Some(b.finish(Output::Compare(if comparison.exec {126} else {dst}), result)); },
                OP16::LT => FloatPred::Olt, OP16::EQ => FloatPred::Oeq,
                OP16::LE => FloatPred::Ole, OP16::GT => FloatPred::Ogt,
                OP16::LG => FloatPred::One, OP16::GE => FloatPred::Oge,
                OP16::O => FloatPred::Ord, OP16::U => FloatPred::Uno,
                OP16::NGE => FloatPred::Ult, OP16::NLG => FloatPred::Ueq,
                OP16::NGT => FloatPred::Ule, OP16::NLE => FloatPred::Ugt,
                OP16::NEQ => FloatPred::Une, OP16::NLT => FloatPred::Uge,
            };
            b.push(Ty::I1, Op::FCmp(pred, a, c))
        }
        Predicate::Integer(p, signed) => {
            if comparison.bits == 16 {
                a = half_word(&mut b, a, opsel & 1 != 0, signed);
                c = half_word(&mut b, c, opsel & 2 != 0, signed);
            }
            a = b.bits_mod(ty, comparison.bits, a, abs, neg, 0);
            c = b.bits_mod(ty, comparison.bits, c, abs, neg, 1);
            if comparison.bits == 16 {
                // Restore sign extension after changing bit 15.
                a = half_word(&mut b, a, false, signed);
                c = half_word(&mut b, c, false, signed);
            }
            match p {
                OP8::F => b.k(Ty::I1, 0), OP8::TRU => b.k(Ty::I1, 1),
                _ => {
                    let pred = match (p, signed) {
                        (OP8::EQ, _) => IntPred::Eq, (OP8::LG, _) => IntPred::Ne,
                        (OP8::LT, true) => IntPred::Slt, (OP8::LE, true) => IntPred::Sle,
                        (OP8::GT, true) => IntPred::Sgt, (OP8::GE, true) => IntPred::Sge,
                        (OP8::LT, false) => IntPred::Ult, (OP8::LE, false) => IntPred::Ule,
                        (OP8::GT, false) => IntPred::Ugt, (OP8::GE, false) => IntPred::Uge,
                        _ => unreachable!(),
                    };
                    b.push(Ty::I1, Op::Cmp(pred, a, c))
                }
            }
        }
        Predicate::Class => {
            if comparison.bits == 16 {
                a = half_word(&mut b, a, opsel & 1 != 0, false);
                classify(&mut b, ty, comparison.bits, a, c)
            } else {
                let float = if comparison.bits == 32 { Ty::F32 } else { Ty::F64 };
                let a = b.push(float, Op::Convert(Cvt::Bitcast, float, a));
                let target = crate::rdna_spmd::dialect::rdna4::comparison(registry, float);
                b.target_one(target, Arguments::Binary([a, c]))
            }
        }
    };
    Some(b.finish(Output::Compare(if comparison.exec {126} else {dst}), result))
}

fn half_word(b: &mut Builder, value: ValueId, high: bool, signed: bool) -> ValueId {
    let shift = b.k(Ty::I32, 16);
    let low = if high { b.int(IntOp::LShr, value, shift) } else { value };
    if signed {
        let shifted = b.int(IntOp::Shl, low, shift);
        b.int(IntOp::AShr, shifted, shift)
    } else {
        let mask = b.k(Ty::I32, 0xffff);
        b.int(IntOp::And, low, mask)
    }
}

/// ISA class bits are sNaN, qNaN, -inf, -normal, -subnormal, -zero,
/// +zero, +subnormal, +normal, +inf. The class selector is a runtime word,
/// including per-lane selectors; LLVM's immarg fpclass cannot express it.
fn classify(b: &mut Builder, ty: Ty, bits: u32, value: ValueId, selector: ValueId) -> ValueId {
    let fraction_bits = match bits { 16 => 10, 32 => 23, 64 => 52, _ => unreachable!() };
    let exponent_bits = bits - fraction_bits - 1;
    let mask = b.k(ty, (1u64 << fraction_bits) - 1);
    let fraction = b.push(ty, Op::Int(IntOp::And, value, mask));
    let exponent_mask = b.k(ty, ((1u64 << exponent_bits) - 1) << fraction_bits);
    let exponent = b.push(ty, Op::Int(IntOp::And, value, exponent_mask));
    let zero = b.k(ty, 0);
    let exponent_zero = b.push(Ty::I1, Op::Cmp(IntPred::Eq, exponent, zero));
    let exponent_full = b.push(Ty::I1, Op::Cmp(IntPred::Eq, exponent, exponent_mask));
    let fraction_zero = b.push(Ty::I1, Op::Cmp(IntPred::Eq, fraction, zero));
    let sign_mask = b.k(ty, 1u64 << (bits - 1));
    let sign = b.push(ty, Op::Int(IntOp::And, value, sign_mask));
    let negative = b.push(Ty::I1, Op::Cmp(IntPred::Ne, sign, zero));
    let quiet_mask = b.k(ty, 1u64 << (fraction_bits - 1));
    let quiet = b.push(ty, Op::Int(IntOp::And, value, quiet_mask));
    let quiet = b.push(Ty::I1, Op::Cmp(IntPred::Ne, quiet, zero));
    let negative_class = [2, 3, 4, 5].map(|n| b.k(Ty::I32, 1 << n));
    let positive_class = [9, 8, 7, 6].map(|n| b.k(Ty::I32, 1 << n));
    let classes: Vec<_> = negative_class.iter().zip(positive_class).map(|(&n, p)| b.push(Ty::I32, Op::Select(negative, n, p))).collect();
    let snan = b.k(Ty::I32, 1); let qnan = b.k(Ty::I32, 2);
    let nan = b.push(Ty::I32, Op::Select(quiet, qnan, snan));
    let special = b.push(Ty::I32, Op::Select(fraction_zero, classes[0], nan));
    let tiny = b.push(Ty::I32, Op::Select(fraction_zero, classes[3], classes[2]));
    let finite = b.push(Ty::I32, Op::Select(exponent_zero, tiny, classes[1]));
    let class = b.push(Ty::I32, Op::Select(exponent_full, special, finite));
    let selected = b.int(IntOp::And, class, selector);
    let zero = b.k(Ty::I32, 0);
    b.push(Ty::I1, Op::Cmp(IntPred::Ne, selected, zero))
}
