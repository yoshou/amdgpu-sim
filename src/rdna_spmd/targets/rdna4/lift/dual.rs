//! Dual issue reads both operand sets before either architectural write.
use super::*;
use crate::rdna_instructions::{VOP1, VOP2, VOP3P};

pub(super) fn instruction(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let InstFormat::VOPD(i) = inst else { return None; };
    let destinations = [i.vdstx, (i.vdsty << 1) | ((i.vdstx & 1) ^ 1)];
    let mut halves = Vec::new();
    for (op, source, reg, dst) in [
        (i.opx, &i.src0x, i.vsrc1x, destinations[0]),
        (i.opy, &i.src0y, i.vsrc1y, destinations[1]),
    ] {
        if matches!(op, I::V_DUAL_DOT2ACC_F32_F16 | I::V_DUAL_DOT2ACC_F32_BF16 | I::V_DUAL_MUL_DX9_ZERO_F32) {
            let lowering = if matches!(op, I::V_DUAL_MUL_DX9_ZERO_F32) {
                let mut b = Builder::new(registry, vec![input(*source, Ty::F32), input(SourceOperand::VectorRegister(reg), Ty::F32)]);
                let zero = b.k(Ty::F32, 0);
                let az = b.push(Ty::I1, Op::FCmp(FloatPred::Oeq, ValueId(0), zero));
                let bz = b.push(Ty::I1, Op::FCmp(FloatPred::Oeq, ValueId(1), zero));
                let any = b.push(Ty::I1, Op::Int(IntOp::Or, az, bz));
                let product = b.push(Ty::F32, Op::Float(FloatOp::Mul, ValueId(0), ValueId(1)));
                let value = b.push(Ty::F32, Op::Select(any, zero, product));
                b.finish(Output::Vgpr(dst as u32, Ty::F32), value)
            } else {
                let bf16 = matches!(op, I::V_DUAL_DOT2ACC_F32_BF16);
                // §7.7.2 DOT2ACC inlines replicate the short-format constant.
                let source = match *source {
                    SourceOperand::FloatConstant(x) => {
                        let bits = if bf16 { (x as f32).to_bits() >> 16 }
                            else { ::half::f16::from_f32(x as f32).to_bits() as u32 };
                        SourceOperand::LiteralConstant(bits | bits << 16)
                    }
                    SourceOperand::IntegerConstant(x) => {
                        let bits = x as u32 & 0xffff;
                        SourceOperand::LiteralConstant(bits | bits << 16)
                    }
                    s => s,
                };
                let packed = InstFormat::VOP3P(VOP3P {
                    op: if bf16 { I::V_DOT2_F32_BF16 } else { I::V_DOT2_F32_F16 },
                    src0: source, src1: SourceOperand::VectorRegister(reg), src2: SourceOperand::VectorRegister(dst),
                    vdst: dst, opsel: 0, opsel_hi: 3, opsel_hi2: 1, neg: 0, neg_hi: 0, cm: 0,
                });
                super::packed::instruction(&packed, registry).unwrap()
            };
            let Lowering::TypedAlu { inputs, outputs, expr, .. } = lowering else { unreachable!() };
            halves.push((inputs, outputs, expr));
            continue;
        }
        let op = match op {
            I::V_DUAL_MOV_B32 => I::V_MOV_B32,
            I::V_DUAL_AND_B32 => I::V_AND_B32,
            I::V_DUAL_ADD_NC_U32 => I::V_ADD_NC_U32,
            I::V_DUAL_LSHLREV_B32 => I::V_LSHLREV_B32,
            I::V_DUAL_CNDMASK_B32 => I::V_CNDMASK_B32,
            I::V_DUAL_MUL_F32 => I::V_MUL_F32,
            I::V_DUAL_MAX_NUM_F32 => I::V_MAX_NUM_F32,
            I::V_DUAL_MIN_NUM_F32 => I::V_MIN_NUM_F32,
            I::V_DUAL_ADD_F32 => I::V_ADD_F32,
            I::V_DUAL_SUB_F32 => I::V_SUB_F32,
            I::V_DUAL_SUBREV_F32 => I::V_SUBREV_F32,
            I::V_DUAL_FMAC_F32 => I::V_FMAC_F32,
            I::V_DUAL_FMAMK_F32 => I::V_FMAMK_F32,
            I::V_DUAL_FMAAK_F32 => I::V_FMAAK_F32,
            _ => return None,
        };
        // Normalize the format at the ISA boundary; the arithmetic semantics
        // are the same shared lift used for individually encoded operations.
        let single = if matches!(op, I::V_MOV_B32) {
            InstFormat::VOP1(VOP1 { op, src0: source.clone(), vdst: dst })
        } else {
            InstFormat::VOP2(VOP2 { op, src0: source.clone(), vsrc1: reg,
                vdst: dst, literal_constant: i.literal_constant })
        };
        let Lowering::TypedAlu { inputs, outputs, expr, scalar: false } = super::instruction_with_registry(&single, registry)
            else { return None; };
        halves.push((inputs, outputs, expr));
    }
    let mut b = Builder::new(registry, halves.iter().flat_map(|h| h.0.iter().cloned()).collect());
    let mut base = 0;
    let mut outputs = Vec::new();
    for (inputs, destinations, expr) in halves {
        let mut values: Vec<_> = (base..base + inputs.len()).map(ValueId).collect();
        base += inputs.len();
        for inst in &expr.expr().insts {
            let results = match inst {
                ExprInst::Core(ty, op) => vec![b.push(*ty, op.map(|v| values[v.0]))],
                ExprInst::Target { op, args, .. } => b.target(*op, args.map(|v| values[v.0])),
            };
            values.extend(results);
        }
        outputs.extend(destinations.into_iter().zip(expr.expr().results.iter().map(|v| values[v.0])));
    }
    Some(b.finish_many(false, outputs))
}
