use super::*;

fn arithmetic(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let InstFormat::SOP2(i) = inst else {
        return None;
    };
    let ty = match i.op {
        I::S_ADD_U32
        | I::S_ADD_CO_U32
        | I::S_ADD_CO_CI_U32
        | I::S_SUB_CO_U32
        | I::S_SUB_CO_CI_U32
        | I::S_ADD_CO_I32
        | I::S_ADD_I32
        | I::S_SUB_CO_I32
        | I::S_MUL_I32
        | I::S_MUL_HI_U32
        | I::S_CSELECT_B32
        | I::S_LSHL_B32
        | I::S_LSHR_B32
        | I::S_ASHR_I32
        | I::S_BFM_B32
        | I::S_BFE_U32
        | I::S_MAX_U32 => Ty::I32,
        I::S_ADD_NC_U64
        | I::S_MUL_U64
        | I::S_LSHL_B64
        | I::S_LSHR_B64
        | I::S_ASHR_I64
        | I::S_AND_B64
        | I::S_OR_B64
        | I::S_XOR_B64
        | I::S_CSELECT_B64 => Ty::I64,
        _ => return None,
    };
    let shift64 = matches!(i.op, I::S_LSHL_B64 | I::S_LSHR_B64 | I::S_ASHR_I64);
    let mut inputs = vec![
        if matches!(i.op, I::S_ASHR_I32 | I::S_ASHR_I64) {
            signed_input(i.ssrc0, ty)
        } else {
            input(i.ssrc0, ty)
        },
        input(i.ssrc1, if shift64 { Ty::I32 } else { ty }),
    ];
    if matches!(
        i.op,
        I::S_CSELECT_B32 | I::S_CSELECT_B64 | I::S_ADD_CO_CI_U32 | I::S_SUB_CO_CI_U32
    ) {
        inputs.push(Input {
            source: InputSource::Scc,
            ty: Ty::I1,
        });
    }
    let mut b = Builder::new(registry, inputs);
    let (a, mut c) = (ValueId(0), ValueId(1));
    if shift64 {
        c = b.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, c));
    }
    let mut flag = None;
    let result = match i.op {
        I::S_ADD_U32
        | I::S_ADD_CO_U32
        | I::S_ADD_CO_CI_U32
        | I::S_SUB_CO_U32
        | I::S_SUB_CO_CI_U32
        | I::S_MUL_HI_U32 => {
            let a = b.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, a));
            let mut c = b.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, c));
            if matches!(i.op, I::S_ADD_CO_CI_U32 | I::S_SUB_CO_CI_U32) {
                let carry = b.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, ValueId(2)));
                c = b.push(Ty::I64, Op::Int(IntOp::Add, c, carry));
            }
            let subtract = matches!(i.op, I::S_SUB_CO_U32 | I::S_SUB_CO_CI_U32);
            let op = if subtract {
                IntOp::Sub
            } else if matches!(i.op, I::S_MUL_HI_U32) {
                IntOp::Mul
            } else {
                IntOp::Add
            };
            let wide = b.push(Ty::I64, Op::Int(op, a, c));
            let shift = b.k(Ty::I64, 32);
            let high = b.push(Ty::I64, Op::Int(IntOp::LShr, wide, shift));
            if !matches!(i.op, I::S_MUL_HI_U32) {
                let zero = b.k(Ty::I64, 0);
                flag = Some(b.push(Ty::I1, Op::Cmp(IntPred::Ne, high, zero)));
                b.push(Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, wide))
            } else {
                b.push(Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, high))
            }
        }
        I::S_ADD_CO_I32 | I::S_ADD_I32 | I::S_SUB_CO_I32 => {
            let sub = matches!(i.op, I::S_SUB_CO_I32);
            let result = b.int(if sub { IntOp::Sub } else { IntOp::Add }, a, c);
            let ar = b.int(IntOp::Xor, a, result);
            let other = b.int(
                IntOp::Xor,
                if sub { a } else { c },
                if sub { c } else { result },
            );
            let overflow = b.int(IntOp::And, ar, other);
            let zero = b.k(Ty::I32, 0);
            flag = Some(b.push(Ty::I1, Op::Cmp(IntPred::Slt, overflow, zero)));
            result
        }
        I::S_MUL_I32 | I::S_MUL_U64 => b.push(ty, Op::Int(IntOp::Mul, a, c)),
        I::S_ADD_NC_U64 => b.push(Ty::I64, Op::Int(IntOp::Add, a, c)),
        I::S_CSELECT_B32 | I::S_CSELECT_B64 => b.push(ty, Op::Select(ValueId(2), a, c)),
        I::S_AND_B64 | I::S_OR_B64 | I::S_XOR_B64 => {
            let op = match i.op {
                I::S_AND_B64 => IntOp::And,
                I::S_OR_B64 => IntOp::Or,
                _ => IntOp::Xor,
            };
            let value = b.push(ty, Op::Int(op, a, c));
            let zero = b.k(ty, 0);
            flag = Some(b.push(Ty::I1, Op::Cmp(IntPred::Ne, value, zero)));
            value
        }
        I::S_LSHL_B32
        | I::S_LSHR_B32
        | I::S_ASHR_I32
        | I::S_LSHL_B64
        | I::S_LSHR_B64
        | I::S_ASHR_I64 => {
            let op = match i.op {
                I::S_LSHR_B32 | I::S_LSHR_B64 => IntOp::LShr,
                I::S_ASHR_I32 | I::S_ASHR_I64 => IntOp::AShr,
                _ => IntOp::Shl,
            };
            let value = b.push(ty, Op::Int(op, a, c));
            let zero = b.k(ty, 0);
            flag = Some(b.push(Ty::I1, Op::Cmp(IntPred::Ne, value, zero)));
            value
        }
        I::S_BFM_B32 => {
            let one = b.k(Ty::I32, 1);
            let power = b.int(IntOp::Shl, one, a);
            let ones = b.int(IntOp::Sub, power, one);
            b.int(IntOp::Shl, ones, c)
        }
        I::S_BFE_U32 => {
            let sixteen = b.k(Ty::I32, 16);
            let width = b.int(IntOp::LShr, c, sixteen);
            let field_mask = b.k(Ty::I32, 0x7f);
            let width = b.int(IntOp::And, width, field_mask);
            let one = b.k(Ty::I32, 1);
            let power = b.int(IntOp::Shl, one, width);
            let low_mask = b.int(IntOp::Sub, power, one);
            let thirty_two = b.k(Ty::I32, 32);
            let wide = b.push(Ty::I1, Op::Cmp(IntPred::Uge, width, thirty_two));
            let all = b.k(Ty::I32, u32::MAX as u64);
            let mask = b.push(Ty::I32, Op::Select(wide, all, low_mask));
            let shifted = b.int(IntOp::LShr, a, c);
            let value = b.int(IntOp::And, shifted, mask);
            let zero = b.k(Ty::I32, 0);
            flag = Some(b.push(Ty::I1, Op::Cmp(IntPred::Ne, value, zero)));
            value
        }
        I::S_MAX_U32 => {
            let greater = b.push(Ty::I1, Op::Cmp(IntPred::Ugt, a, c));
            flag = Some(greater);
            b.push(Ty::I32, Op::Select(greater, a, c))
        }
        _ => unreachable!(),
    };
    let mut results = vec![(Output::Scalar(i.sdst as u32, ty), result)];
    if let Some(flag) = flag {
        results.push((Output::Scc, flag));
    }
    Some(b.finish_many(true, results))
}

fn compare(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let InstFormat::SOPC(i) = inst else {
        return None;
    };
    let (ty, predicate) = match i.op {
        I::S_CMP_EQ_U32 | I::S_CMP_EQ_I32 => (Ty::I32, IntPred::Eq),
        I::S_CMP_LG_U32 | I::S_CMP_LG_I32 => (Ty::I32, IntPred::Ne),
        I::S_CMP_GT_U32 => (Ty::I32, IntPred::Ugt),
        I::S_CMP_GE_U32 => (Ty::I32, IntPred::Uge),
        I::S_CMP_LT_U32 => (Ty::I32, IntPred::Ult),
        I::S_CMP_LE_U32 => (Ty::I32, IntPred::Ule),
        I::S_CMP_GT_I32 => (Ty::I32, IntPred::Sgt),
        I::S_CMP_GE_I32 => (Ty::I32, IntPred::Sge),
        I::S_CMP_LT_I32 => (Ty::I32, IntPred::Slt),
        I::S_CMP_LE_I32 => (Ty::I32, IntPred::Sle),
        I::S_CMP_EQ_U64 => (Ty::I64, IntPred::Eq),
        I::S_CMP_LG_U64 => (Ty::I64, IntPred::Ne),
        _ => return float_compare(i, registry),
    };
    let mut b = Builder::new(
        registry,
        vec![input(i.ssrc0.clone(), ty), input(i.ssrc1.clone(), ty)],
    );
    let flag = b.push(Ty::I1, Op::Cmp(predicate, ValueId(0), ValueId(1)));
    Some(b.finish_many(true, vec![(Output::Scc, flag)]))
}

fn float_compare(i: &crate::rdna_instructions::SOPC, registry: &DialectRegistry) -> Option<Lowering> {
    let predicate = match i.op {
        I::S_CMP_LT_F32 => FloatPred::Olt,
        I::S_CMP_EQ_F32 => FloatPred::Oeq,
        I::S_CMP_LE_F32 => FloatPred::Ole,
        I::S_CMP_GT_F32 => FloatPred::Ogt,
        I::S_CMP_LG_F32 => FloatPred::One,
        I::S_CMP_GE_F32 => FloatPred::Oge,
        I::S_CMP_O_F32 => FloatPred::Ord,
        I::S_CMP_U_F32 => FloatPred::Uno,
        I::S_CMP_NGE_F32 => FloatPred::Ult,
        I::S_CMP_NLG_F32 => FloatPred::Ueq,
        I::S_CMP_NGT_F32 => FloatPred::Ule,
        I::S_CMP_NLE_F32 => FloatPred::Ugt,
        I::S_CMP_NEQ_F32 => FloatPred::Une,
        I::S_CMP_NLT_F32 => FloatPred::Uge,
        _ => return None,
    };
    let ty = Ty::F32;
    let mut b = Builder::new(
        registry,
        vec![input(i.ssrc0.clone(), ty), input(i.ssrc1.clone(), ty)],
    );
    let flag = b.push(Ty::I1, Op::FCmp(predicate, ValueId(0), ValueId(1)));
    Some(b.finish_many(true, vec![(Output::Scc, flag)]))
}

fn unary(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let InstFormat::SOP1(i) = inst else {
        return None;
    };
    let (from, to, cvt) = match i.op {
        I::S_MOV_B32 => (Ty::I32, Ty::I32, None),
        I::S_MOV_B64 => (Ty::I64, Ty::I64, None),
        I::S_CTZ_I32_B32 | I::S_SEXT_I32_I16 | I::S_BREV_B32 => (Ty::I32, Ty::I32, None),
        I::S_BREV_B64 => (Ty::I64, Ty::I64, None),
        I::S_CVT_F32_I32 => (Ty::I32, Ty::F32, Some(Cvt::SignedToFloatRte)),
        I::S_CVT_F32_U32 => (Ty::I32, Ty::F32, Some(Cvt::UnsignedToFloatRte)),
        I::S_CVT_I32_F32 => (Ty::F32, Ty::I32, Some(Cvt::FloatToSignedSatRtz)),
        I::S_CVT_U32_F32 => (Ty::F32, Ty::I32, Some(Cvt::FloatToUnsignedSatRtz)),
        I::S_CVT_F16_F32 => (Ty::F32, Ty::I32, None),
        _ => return None,
    };
    let mut b = Builder::new(registry, vec![input(i.ssrc0.clone(), from)]);
    let result = if matches!(i.op, I::S_CVT_F16_F32) {
        let target = crate::rdna_spmd::rdna4::dialect::unary(registry, I::V_CVT_F16_F32).unwrap();
        b.target_one(target, Arguments::Unary(ValueId(0)))
    } else if let Some(cvt) = cvt {
        b.push(to, Op::Convert(cvt, to, ValueId(0)))
    } else if matches!(i.op, I::S_SEXT_I32_I16) {
        let shift = b.k(Ty::I32, 16);
        let high = b.int(IntOp::Shl, ValueId(0), shift);
        b.int(IntOp::AShr, high, shift)
    } else if matches!(i.op, I::S_BREV_B32 | I::S_BREV_B64) {
        b.push(to, Op::ReverseBits(ValueId(0)))
    } else if matches!(i.op, I::S_CTZ_I32_B32) {
        let count = b.push(Ty::I32, Op::TrailingZeros(ValueId(0)));
        let zero = b.k(Ty::I32, 0);
        let empty = b.push(Ty::I1, Op::Cmp(IntPred::Eq, ValueId(0), zero));
        let missing = b.k(Ty::I32, u32::MAX as u64);
        b.push(Ty::I32, Op::Select(empty, missing, count))
    } else {
        ValueId(0)
    };
    Some(b.finish_many(true, vec![(Output::Scalar(i.sdst as u32, to), result)]))
}

fn immediate(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let InstFormat::SOPK(i) = inst else {
        return None;
    };
    let predicate = match i.op {
        I::S_CMPK_EQ_I32 | I::S_CMPK_EQ_U32 => Some(IntPred::Eq),
        I::S_CMPK_LG_I32 | I::S_CMPK_LG_U32 => Some(IntPred::Ne),
        I::S_CMPK_GT_I32 => Some(IntPred::Sgt),
        I::S_CMPK_GE_I32 => Some(IntPred::Sge),
        I::S_CMPK_LT_I32 => Some(IntPred::Slt),
        I::S_CMPK_LE_I32 => Some(IntPred::Sle),
        I::S_CMPK_GT_U32 => Some(IntPred::Ugt),
        I::S_CMPK_GE_U32 => Some(IntPred::Uge),
        I::S_CMPK_LT_U32 => Some(IntPred::Ult),
        I::S_CMPK_LE_U32 => Some(IntPred::Ule),
        I::S_MOVK_I32 | I::S_CMOVK_I32 | I::S_ADDK_I32 | I::S_ADDK_CO_I32 | I::S_MULK_I32 => None,
        _ => return None,
    };
    let mut inputs = if matches!(i.op, I::S_MOVK_I32) {
        vec![]
    } else {
        vec![input(SourceOperand::ScalarRegister(i.sdst), Ty::I32)]
    };
    if matches!(i.op, I::S_CMOVK_I32) {
        inputs.push(Input {
            source: InputSource::Scc,
            ty: Ty::I1,
        });
    }
    let mut b = Builder::new(registry, inputs);
    let imm = b.k(Ty::I32, i.simm16 as i16 as i32 as u32 as u64);
    if let Some(predicate) = predicate {
        let flag = b.push(Ty::I1, Op::Cmp(predicate, ValueId(0), imm));
        return Some(b.finish_many(true, vec![(Output::Scc, flag)]));
    }
    let mut flag = None;
    let result = match i.op {
        I::S_MOVK_I32 => imm,
        I::S_CMOVK_I32 => b.push(Ty::I32, Op::Select(ValueId(1), imm, ValueId(0))),
        I::S_ADDK_I32 | I::S_ADDK_CO_I32 => {
            let result = b.int(IntOp::Add, ValueId(0), imm);
            let x = b.int(IntOp::Xor, ValueId(0), result);
            let y = b.int(IntOp::Xor, imm, result);
            let sign = b.int(IntOp::And, x, y);
            let zero = b.k(Ty::I32, 0);
            flag = Some(b.push(Ty::I1, Op::Cmp(IntPred::Slt, sign, zero)));
            result
        }
        I::S_MULK_I32 => b.int(IntOp::Mul, ValueId(0), imm),
        _ => unreachable!(),
    };
    let mut outputs = vec![(Output::Scalar(i.sdst as u32, Ty::I32), result)];
    if let Some(flag) = flag {
        outputs.push((Output::Scc, flag));
    }
    Some(b.finish_many(true, outputs))
}

fn cmov(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let InstFormat::SOP1(i) = inst else {
        return None;
    };
    let ty = match i.op {
        I::S_CMOV_B32 => Ty::I32,
        I::S_CMOV_B64 => Ty::I64,
        _ => return None,
    };
    let mut b = Builder::new(
        registry,
        vec![
            input(i.ssrc0.clone(), ty),
            input(SourceOperand::ScalarRegister(i.sdst), ty),
            Input {
                source: InputSource::Scc,
                ty: Ty::I1,
            },
        ],
    );
    let result = b.push(ty, Op::Select(ValueId(2), ValueId(0), ValueId(1)));
    Some(b.finish_many(true, vec![(Output::Scalar(i.sdst as u32, ty), result)]))
}

pub fn instruction(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    masks(inst, registry)
        .or_else(|| cmov(inst, registry))
        .or_else(|| arithmetic(inst, registry))
        .or_else(|| float(inst, registry))
        .or_else(|| bit_count(inst, registry))
        .or_else(|| compare(inst, registry))
        .or_else(|| unary(inst, registry))
        .or_else(|| immediate(inst, registry))
}

fn bit_count(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let InstFormat::SOP1(i) = inst else {
        return None;
    };
    let (ty, ones) = match i.op {
        I::S_BCNT1_I32_B32 => (Ty::I32, true),
        I::S_BCNT0_I32_B32 => (Ty::I32, false),
        I::S_BCNT1_I32_B64 => (Ty::I64, true),
        I::S_BCNT0_I32_B64 => (Ty::I64, false),
        _ => return None,
    };
    let mut b = Builder::new(registry, vec![input(i.ssrc0.clone(), ty)]);
    let word = if ones {
        ValueId(0)
    } else {
        let all = b.k(
            ty,
            if ty == Ty::I64 {
                u64::MAX
            } else {
                u32::MAX as u64
            },
        );
        b.push(ty, Op::Int(IntOp::Xor, ValueId(0), all))
    };
    let count = b.push(ty, Op::PopulationCount(word));
    let count = if ty == Ty::I64 {
        b.push(Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, count))
    } else {
        count
    };
    let zero = b.k(Ty::I32, 0);
    let flag = b.push(Ty::I1, Op::Cmp(IntPred::Ne, count, zero));
    Some(b.finish_many(
        true,
        vec![
            (Output::Scalar(i.sdst as u32, Ty::I32), count),
            (Output::Scc, flag),
        ],
    ))
}

fn float(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let InstFormat::SOP2(i) = inst else {
        return None;
    };
    let op = match i.op {
        I::S_ADD_F32 => Some(FloatOp::Add),
        I::S_SUB_F32 => Some(FloatOp::Sub),
        I::S_MUL_F32 => Some(FloatOp::Mul),
        I::S_MIN_NUM_F32 => Some(FloatOp::MinNum),
        I::S_MAX_NUM_F32 => Some(FloatOp::MaxNum),
        I::S_FMAC_F32 => None,
        _ => return None,
    };
    let ty = Ty::F32;
    let mut inputs = vec![input(i.ssrc0, ty), input(i.ssrc1, ty)];
    if op.is_none() {
        inputs.push(input(SourceOperand::ScalarRegister(i.sdst), ty));
    }
    let mut b = Builder::new(registry, inputs);
    let mut result = match op {
        Some(op) => b.push(ty, Op::Float(op, ValueId(0), ValueId(1))),
        None => b.push(ty, Op::Fma(ValueId(0), ValueId(1), ValueId(2))),
    };
    if matches!(i.op, I::S_SUB_F32) {
        let nan_a = b.push(Ty::I1, Op::FCmp(FloatPred::Uno, ValueId(0), ValueId(0)));
        let nan_b = b.push(Ty::I1, Op::FCmp(FloatPred::Uno, ValueId(1), ValueId(1)));
        let word = b.push(Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, ValueId(1)));
        let quiet = b.k(Ty::I32, 0x0040_0000);
        let word = b.int(IntOp::Or, word, quiet);
        let sign = b.k(Ty::I32, 0x8000_0000);
        let word = b.int(IntOp::Xor, word, sign);
        let nan = b.push(Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, word));
        let right_nan = b.push(Ty::F32, Op::Select(nan_b, nan, result));
        result = b.push(Ty::F32, Op::Select(nan_a, result, right_nan));
    }
    Some(b.finish_many(true, vec![(Output::Scalar(i.sdst as u32, ty), result)]))
}

fn masks(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    if let InstFormat::SOP1(i) = inst {
        let op = match i.op {
            I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 => IntOp::And,
            I::S_OR_SAVEEXEC_B32 => IntOp::Or,
            I::S_XOR_SAVEEXEC_B32 => IntOp::Xor,
            _ => return None,
        };
        let mut b = Builder::new(
            registry,
            vec![
                input(i.ssrc0, Ty::I32),
                input(SourceOperand::ScalarRegister(126), Ty::I32),
            ],
        );
        let old = ValueId(1);
        let rhs = if matches!(i.op, I::S_AND_NOT1_SAVEEXEC_B32) {
            let all = b.k(Ty::I32, u32::MAX as u64);
            b.push(Ty::I32, Op::Int(IntOp::Xor, old, all))
        } else {
            old
        };
        let next = b.push(Ty::I32, Op::Int(op, ValueId(0), rhs));
        let zero = b.k(Ty::I32, 0);
        let flag = b.push(Ty::I1, Op::Cmp(IntPred::Ne, next, zero));
        return Some(b.finish_many(
            true,
            vec![
                (Output::Scalar(i.sdst as u32, Ty::I32), old),
                (Output::Scalar(126, Ty::I32), next),
                (Output::Scc, flag),
            ],
        ));
    }
    let InstFormat::SOP2(i) = inst else {
        return None;
    };
    let op = match i.op {
        I::S_AND_B32 | I::S_AND_NOT1_B32 => IntOp::And,
        I::S_OR_B32 | I::S_OR_NOT1_B32 => IntOp::Or,
        I::S_XOR_B32 => IntOp::Xor,
        _ => return None,
    };
    let mut b = Builder::new(
        registry,
        vec![input(i.ssrc0, Ty::I32), input(i.ssrc1, Ty::I32)],
    );
    let rhs = if matches!(i.op, I::S_AND_NOT1_B32 | I::S_OR_NOT1_B32) {
        let all = b.k(Ty::I32, u32::MAX as u64);
        b.int(IntOp::Xor, ValueId(1), all)
    } else {
        ValueId(1)
    };
    let value = b.int(op, ValueId(0), rhs);
    let zero = b.k(Ty::I32, 0);
    let flag = b.push(Ty::I1, Op::Cmp(IntPred::Ne, value, zero));
    Some(b.finish_many(
        true,
        vec![
            (Output::Scalar(i.sdst as u32, Ty::I32), value),
            (Output::Scc, flag),
        ],
    ))
}
