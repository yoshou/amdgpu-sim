//! Half conversion operand selection and partial-register destinations (§7.4).
//! Numeric conversions are supplied by the RDNA4 provider.
use super::*;

pub(super) fn source(mut source: SourceOperand) -> SourceOperand {
    // §4.1: inline floating constants use the operand's precision and occupy
    // the low half, while integer and literal constants retain their bits.
    if let SourceOperand::FloatConstant(value) = source {
        source = SourceOperand::LiteralConstant(::half::f16::from_f32(value as f32).to_bits() as u32);
    }
    source
}

struct Operands {
    op: I, src: SourceOperand, dst: u8, abs: u8, neg: u8, clamp: u8,
    omod: u8, src_high: bool, dst_high: bool, widening: bool,
}

fn operands(inst: &InstFormat) -> Option<Operands> {
    let (op, mut src, mut dst, abs, neg, clamp, omod, mut src_high, mut dst_high, short) = match inst {
        InstFormat::VOP1(i) => (i.op, i.src0.clone(), i.vdst, 0, 0, 0, 0, false, false, true),
        InstFormat::VOP3(i) => (i.op, i.src0.clone(), i.vdst, i.abs, i.neg, i.cm, i.omod, i.opsel & 1 != 0, i.opsel & 8 != 0, false),
        _ => return None,
    };
    let widening = match op { I::V_CVT_F32_F16 => true, I::V_CVT_F16_F32 => false, _ => return None };
    if short {
        if widening {
            if let SourceOperand::VectorRegister(ref mut r) = src { src_high = *r & 128 != 0; *r &= 127; }
        } else { dst_high = dst & 128 != 0; dst &= 127; }
    }
    Some(Operands { op, src, dst, abs, neg, clamp, omod, src_high, dst_high, widening })
}

/// The preserved half is an input, including when the numeric input is uniform.
/// Share operand decoding with liveness, divergence and register-view analysis.
pub(in crate::rdna_spmd) fn registers(inst: &InstFormat) -> Option<(Vec<u32>, u32)> {
    if let InstFormat::VOP3P(i) = inst {
        if matches!(i.op, I::V_FMA_MIXLO_F16 | I::V_FMA_MIXHI_F16) {
            let mut reads: Vec<_> = [i.src0, i.src1, i.src2].iter().filter_map(|s| {
                if let SourceOperand::VectorRegister(r) = s { Some(*r as u32) } else { None }
            }).collect();
            reads.push(i.vdst as u32);
            return Some((reads, i.vdst as u32));
        }
    }
    let o = operands(inst)?;
    let mut reads = Vec::new();
    if let SourceOperand::VectorRegister(r) = o.src { reads.push(r as u32); }
    if !o.widening { reads.push(o.dst as u32); }
    Some((reads, o.dst as u32))
}

pub(super) fn instruction(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let Operands { op, src, dst, abs, neg, clamp, omod, src_high, dst_high, widening } = operands(inst)?;
    let target = crate::rdna_spmd::dialect::rdna4::unary(registry, op).unwrap();
    if widening {
        let mut b = Builder::new(registry, vec![input(source(src), Ty::I32)]);
        let mut value = ValueId(0);
        if src_high { let shift = b.k(Ty::I32, 16); value = b.int(IntOp::LShr, value, shift); }
        value = b.bits_mod(Ty::I32, 16, value, abs, neg, 0);
        value = b.target_one(target, Arguments::Unary(value));
        value = b.output_mod(Ty::F32, value, clamp, omod);
        Some(b.finish(Output::Vgpr(dst as u32, Ty::F32), value))
    } else {
        let mut b = Builder::new(registry, vec![input(src, Ty::F32), input(SourceOperand::VectorRegister(dst), Ty::I32)]);
        let mut value = b.float_mod(Ty::F32, ValueId(0), abs, neg, 0);
        value = b.output_mod(Ty::F32, value, clamp, omod);
        value = b.target_one(target, Arguments::Unary(value));
        if omod != 0 {
            let mask = b.k(Ty::I32, 0x7c00); let exponent = b.int(IntOp::And, value, mask);
            let zero = b.k(Ty::I32, 0); let tiny = b.push(Ty::I1, Op::Cmp(IntPred::Eq, exponent, zero));
            value = b.push(Ty::I32, Op::Select(tiny, zero, value));
        }
        if dst_high { let shift = b.k(Ty::I32, 16); value = b.int(IntOp::Shl, value, shift); }
        let keep = b.k(Ty::I32, if dst_high { 0xffff } else { 0xffff_0000 });
        let old = b.int(IntOp::And, ValueId(1), keep);
        let result = b.int(IntOp::Or, old, value);
        Some(b.finish(Output::Vgpr(dst as u32, Ty::I32), result))
    }
}
