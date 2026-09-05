//! Incremental ISA lift. Pure ALU semantics live in typed SSA; register access,
//! mask-word conventions and the remaining effectful instructions stay in this
//! explicit migration adapter. Nothing is scheduled across a legacy boundary.
use super::ir::typed::*;
use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand};
pub(super) mod memory;
pub(super) mod wave;

pub(super) enum Lowering<'a> {
    Memory(memory::Memory),
    Wave(wave::YieldAction),
    TypedAlu {
        inputs: Vec<Input>,
        output: Output,
        expr: VerifiedExpr,
    },
    Legacy(&'a InstFormat),
}
#[derive(Clone, Debug)]
pub(super) struct Input {
    pub source: SourceOperand,
    pub ty: Ty,
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Output {
    Vgpr(u32, Ty),
    Compare(u32),
}
impl Output {
    pub fn ty(self) -> Ty {
        match self {
            Self::Vgpr(_, t) => t,
            Self::Compare(_) => Ty::I1,
        }
    }
}
struct Builder {
    inputs: Vec<Input>,
    insts: Vec<(Ty, Op)>,
}
impl Builder {
    fn new(inputs: Vec<Input>) -> Self {
        Self {
            inputs,
            insts: vec![],
        }
    }
    fn push(&mut self, ty: Ty, op: Op) -> ValueId {
        let id = ValueId(self.inputs.len() + self.insts.len());
        self.insts.push((ty, op));
        id
    }
    fn k(&mut self, ty: Ty, bits: u64) -> ValueId {
        self.push(ty, Op::Const(ty, bits))
    }
    fn int(&mut self, op: IntOp, a: ValueId, b: ValueId) -> ValueId {
        self.push(Ty::I32, Op::Int(op, a, b))
    }
    fn float_mod(&mut self, ty: Ty, mut a: ValueId, abs: u8, neg: u8, index: usize) -> ValueId {
        if abs >> index & 1 != 0 {
            a = self.push(ty, Op::Unary(FloatUnary::Abs, a));
        }
        if neg >> index & 1 != 0 {
            a = self.push(ty, Op::Unary(FloatUnary::Neg, a));
        }
        a
    }
    fn finish(self, output: Output, result: ValueId) -> Lowering<'static> {
        let expr = Expr {
            params: self.inputs.iter().map(|i| i.ty).collect(),
            insts: self.insts,
            result,
        }
        .verify()
        .expect("invalid ALU lift");
        Lowering::TypedAlu {
            inputs: self.inputs,
            output,
            expr,
        }
    }
}
fn input(source: SourceOperand, ty: Ty) -> Input {
    Input { source, ty }
}

pub(super) fn instruction(inst: &InstFormat) -> Lowering<'_> {
    if let Some(action) = wave::instruction(inst) { return Lowering::Wave(action); }
    if let Some(memory) = memory::instruction(inst) { return Lowering::Memory(memory); }
    let (op, src, dst, abs, neg, cm) = match inst {
        InstFormat::VOP1(i) => (i.op, vec![i.src0.clone()], i.vdst, 0, 0, 0),
        InstFormat::VOP2(i) => (
            i.op,
            vec![i.src0.clone(), SourceOperand::VectorRegister(i.vsrc1)],
            i.vdst,
            0,
            0,
            0,
        ),
        InstFormat::VOP3(i) => (
            i.op,
            vec![i.src0.clone(), i.src1.clone(), i.src2.clone()],
            i.vdst,
            i.abs,
            i.neg,
            i.cm,
        ),
        InstFormat::VOPC(i) => (
            i.op,
            vec![i.src0.clone(), SourceOperand::VectorRegister(i.vsrc1)],
            if is_cmpx(i.op) { 126 } else { 106 },
            0,
            0,
            0,
        ),
        _ => return Lowering::Legacy(inst),
    };
    let a = ValueId(0);
    let b = ValueId(1);
    let c = ValueId(2);
    // Decide input types/arity before assigning IDs. Literals retain their ISA
    // interpretation in Input, while constants introduced by the lift are bits.
    let cmp = compare(op);
    let cvt = conversion(op);
    let float_ty = match op {
        I::V_ADD_F64
        | I::V_MUL_F64
        | I::V_FMA_F64
        | I::V_MIN_NUM_F64
        | I::V_MAX_NUM_F64
        | I::V_RCP_F64
        | I::V_RSQ_F64
        | I::V_SQRT_F64
        | I::V_RNDNE_F64 => Some(Ty::F64),
        I::V_ADD_F32
        | I::V_SUB_F32
        | I::V_SUBREV_F32
        | I::V_MUL_F32
        | I::V_FMA_F32
        | I::V_FMAC_F32
        | I::V_FMAMK_F32
        | I::V_FMAAK_F32
        | I::V_MIN_F32
        | I::V_MAX_F32
        | I::V_MIN_NUM_F32
        | I::V_MAX_NUM_F32
        | I::V_RCP_F32
        | I::V_RCP_IFLAG_F32
        | I::V_RSQ_F32
        | I::V_SQRT_F32
        | I::V_FLOOR_F32
        | I::V_CEIL_F32
        | I::V_TRUNC_F32
        | I::V_RNDNE_F32 => Some(Ty::F32),
        _ => None,
    };
    let ty = cmp
        .map(|x| x.0)
        .or(cvt.map(|x| x.0))
        .or(float_ty)
        .unwrap_or(Ty::I32);
    let arity = if cmp.is_some() {
        2
    } else if cvt.is_some() {
        1
    } else {
        match op {
            I::V_MOV_B32
            | I::V_RCP_F32
            | I::V_RCP_IFLAG_F32
            | I::V_RSQ_F32
            | I::V_SQRT_F32
            | I::V_RCP_F64
            | I::V_RSQ_F64
            | I::V_SQRT_F64
            | I::V_RNDNE_F64
            | I::V_FLOOR_F32
            | I::V_CEIL_F32
            | I::V_TRUNC_F32
            | I::V_RNDNE_F32 => 1,
            I::V_ADD3_U32
            | I::V_XOR3_B32
            | I::V_XAD_U32
            | I::V_AND_OR_B32
            | I::V_OR3_B32
            | I::V_LSHL_OR_B32
            | I::V_LSHL_ADD_U32
            | I::V_ADD_LSHL_U32
            | I::V_BFE_U32
            | I::V_BFI_B32
            | I::V_ALIGNBIT_B32
            | I::V_FMA_F32
            | I::V_FMA_F64 => 3,
            _ => 2,
        }
    };
    if src.len() < arity {
        return Lowering::Legacy(inst);
    }
    let mut inputs: Vec<_> = src
        .iter()
        .take(arity)
        .cloned()
        .map(|s| input(s, ty))
        .collect();
    if matches!(op, I::V_FMAC_F32) {
        inputs.push(input(SourceOperand::VectorRegister(dst), Ty::F32));
    }
    if matches!(op, I::V_FMAMK_F32 | I::V_FMAAK_F32) {
        let literal = match inst {
            InstFormat::VOP2(i) => i.literal_constant,
            _ => None,
        };
        let Some(literal) = literal else {
            return Lowering::Legacy(inst);
        };
        inputs.push(input(SourceOperand::LiteralConstant(literal), Ty::F32));
    }
    if matches!(op, I::V_CNDMASK_B32) {
        // I1 inputs are lane-mask projections, not integer conversions.
        inputs.push(input(
            src.get(2)
                .cloned()
                .unwrap_or(SourceOperand::ScalarRegister(106)),
            Ty::I1,
        ));
    }
    if matches!(op, I::V_LSHLREV_B64 | I::V_LSHRREV_B64) {
        inputs[1].ty = Ty::I64;
    }
    let mut q = Builder::new(inputs);
    let (mut a, mut b, mut c) = (a, b, c);
    if !ty.integer() && cvt.is_none() {
        a = q.float_mod(ty, a, abs, neg, 0);
        if arity >= 2 {
            b = q.float_mod(ty, b, abs, neg, 1);
        }
        if arity >= 3 {
            c = q.float_mod(ty, c, abs, neg, 2);
        }
    }
    let mut output = Output::Vgpr(dst as u32, float_ty.unwrap_or(Ty::I32));
    let result = if let Some((_, cmp)) = cmp {
        output = Output::Compare(dst as u32);
        q.push(Ty::I1, cmp.map(|id| if id.0 == 0 { a } else { b }))
    } else if let Some((_, to, kind)) = cvt {
        output = Output::Vgpr(dst as u32, to);
        q.push(to, Op::Convert(kind, to, a))
    } else {
        match op {
            I::V_MOV_B32 => a,
            I::V_ADD_NC_U32 => q.int(IntOp::Add, a, b),
            I::V_SUB_NC_U32 => q.int(IntOp::Sub, a, b),
            I::V_SUBREV_NC_U32 => q.int(IntOp::Sub, b, a),
            I::V_AND_B32 => q.int(IntOp::And, a, b),
            I::V_OR_B32 => q.int(IntOp::Or, a, b),
            I::V_XOR_B32 => q.int(IntOp::Xor, a, b),
            I::V_MUL_LO_U32 => q.int(IntOp::Mul, a, b),
            I::V_LSHLREV_B32 => q.int(IntOp::Shl, b, a),
            I::V_LSHRREV_B32 => q.int(IntOp::LShr, b, a),
            I::V_ASHRREV_I32 => q.int(IntOp::AShr, b, a),
            I::V_MIN_U32 | I::V_MAX_U32 | I::V_MIN_I32 | I::V_MAX_I32 => {
                let pred = match op {
                    I::V_MIN_U32 => IntPred::Ult,
                    I::V_MAX_U32 => IntPred::Ugt,
                    I::V_MIN_I32 => IntPred::Slt,
                    _ => IntPred::Sgt,
                };
                let cond = q.push(Ty::I1, Op::Cmp(pred, a, b));
                q.push(Ty::I32, Op::Select(cond, a, b))
            }
            I::V_CNDMASK_B32 => {
                if abs != 0 || neg != 0 {
                    a = q.push(Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, a));
                    b = q.push(Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, b));
                    a = q.float_mod(Ty::F32, a, abs, neg, 0);
                    b = q.float_mod(Ty::F32, b, abs, neg, 1);
                    a = q.push(Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, a));
                    b = q.push(Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, b));
                }
                q.push(Ty::I32, Op::Select(c, b, a))
            }
            I::V_ADD3_U32
            | I::V_XOR3_B32
            | I::V_XAD_U32
            | I::V_AND_OR_B32
            | I::V_OR3_B32
            | I::V_LSHL_OR_B32
            | I::V_LSHL_ADD_U32
            | I::V_ADD_LSHL_U32 => {
                let (first, second) = match op {
                    I::V_ADD3_U32 => (IntOp::Add, IntOp::Add),
                    I::V_XOR3_B32 => (IntOp::Xor, IntOp::Xor),
                    I::V_XAD_U32 => (IntOp::Xor, IntOp::Add),
                    I::V_AND_OR_B32 => (IntOp::And, IntOp::Or),
                    I::V_OR3_B32 => (IntOp::Or, IntOp::Or),
                    I::V_LSHL_OR_B32 => (IntOp::Shl, IntOp::Or),
                    I::V_LSHL_ADD_U32 => (IntOp::Shl, IntOp::Add),
                    _ => (IntOp::Add, IntOp::Shl),
                };
                let t = q.int(first, a, b);
                q.int(second, t, c)
            }
            I::V_BFE_U32 => {
                let mask = q.k(Ty::I32, 31);
                let w = q.int(IntOp::And, c, mask);
                let one = q.k(Ty::I32, 1);
                let mask = q.int(IntOp::Shl, one, w);
                let mask = q.int(IntOp::Sub, mask, one);
                let x = q.int(IntOp::LShr, a, b);
                q.int(IntOp::And, x, mask)
            }
            I::V_BFI_B32 => {
                let ones = q.k(Ty::I32, u32::MAX as u64);
                let inv = q.int(IntOp::Xor, a, ones);
                let x = q.int(IntOp::And, a, b);
                let y = q.int(IntOp::And, inv, c);
                q.int(IntOp::Or, x, y)
            }
            I::V_MUL_HI_U32 => {
                let a = q.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, a));
                let b = q.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, b));
                let p = q.push(Ty::I64, Op::Int(IntOp::Mul, a, b));
                let k = q.k(Ty::I64, 32);
                let h = q.push(Ty::I64, Op::Int(IntOp::LShr, p, k));
                q.push(Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, h))
            }
            I::V_LSHLREV_B64 | I::V_LSHRREV_B64 => {
                let a = q.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, a));
                output = Output::Vgpr(dst as u32, Ty::I64);
                q.push(
                    Ty::I64,
                    Op::Int(
                        if matches!(op, I::V_LSHLREV_B64) {
                            IntOp::Shl
                        } else {
                            IntOp::LShr
                        },
                        b,
                        a,
                    ),
                )
            }
            I::V_ADD_F32
            | I::V_ADD_F64
            | I::V_SUB_F32
            | I::V_SUBREV_F32
            | I::V_MUL_F32
            | I::V_MUL_F64
            | I::V_MIN_F32
            | I::V_MIN_NUM_F32
            | I::V_MIN_NUM_F64
            | I::V_MAX_F32
            | I::V_MAX_NUM_F32
            | I::V_MAX_NUM_F64 => {
                let f = match op {
                    I::V_ADD_F32 | I::V_ADD_F64 => FloatOp::Add,
                    I::V_SUB_F32 | I::V_SUBREV_F32 => FloatOp::Sub,
                    I::V_MUL_F32 | I::V_MUL_F64 => FloatOp::Mul,
                    I::V_MIN_F32 | I::V_MIN_NUM_F32 | I::V_MIN_NUM_F64 => FloatOp::MinNum,
                    _ => FloatOp::MaxNum,
                };
                let (a, b) = if matches!(op, I::V_SUBREV_F32) {
                    (b, a)
                } else {
                    (a, b)
                };
                let mut r = q.push(ty, Op::Float(f, a, b));
                // This is the existing V_MAX_NUM_F64 clamp contract. Other output
                // modifiers remain as in the current implementation.
                if matches!(op, I::V_MAX_NUM_F64) && cm & 1 != 0 {
                    let zero = q.k(ty, 0);
                    let one = q.k(ty, 1f64.to_bits());
                    r = q.push(ty, Op::Float(FloatOp::MinNum, r, one));
                    r = q.push(ty, Op::Float(FloatOp::MaxNum, r, zero));
                }
                r
            }
            I::V_FMA_F64 => q.push(ty, Op::MulAdd(a, b, c)),
            I::V_FMA_F32 | I::V_FMAC_F32 | I::V_FMAAK_F32 => q.push(ty, Op::Fma(a, b, c)),
            I::V_FMAMK_F32 => q.push(ty, Op::Fma(a, c, b)),
            I::V_RCP_F32 | I::V_RCP_IFLAG_F32 | I::V_RCP_F64 | I::V_RSQ_F32 | I::V_RSQ_F64 => {
                let one = q.k(
                    ty,
                    if ty == Ty::F32 {
                        1f32.to_bits() as u64
                    } else {
                        1f64.to_bits()
                    },
                );
                let a = if matches!(op, I::V_RSQ_F32 | I::V_RSQ_F64) {
                    q.push(ty, Op::Unary(FloatUnary::Sqrt, a))
                } else {
                    a
                };
                q.push(ty, Op::Float(FloatOp::Div, one, a))
            }
            I::V_SQRT_F32
            | I::V_SQRT_F64
            | I::V_FLOOR_F32
            | I::V_CEIL_F32
            | I::V_TRUNC_F32
            | I::V_RNDNE_F32
            | I::V_RNDNE_F64 => {
                let f = match op {
                    I::V_SQRT_F32 | I::V_SQRT_F64 => FloatUnary::Sqrt,
                    I::V_FLOOR_F32 => FloatUnary::Floor,
                    I::V_CEIL_F32 => FloatUnary::Ceil,
                    I::V_TRUNC_F32 => FloatUnary::Trunc,
                    _ => FloatUnary::RoundEven,
                };
                q.push(ty, Op::Unary(f, a))
            }
            _ => return Lowering::Legacy(inst),
        }
    };
    q.finish(output, result)
}
fn conversion(op: I) -> Option<(Ty, Ty, Cvt)> {
    Some(match op {
        I::V_CVT_F32_I32 => (Ty::I32, Ty::F32, Cvt::SignedToFloatRte),
        I::V_CVT_F32_U32 => (Ty::I32, Ty::F32, Cvt::UnsignedToFloatRte),
        I::V_CVT_F64_I32 => (Ty::I32, Ty::F64, Cvt::SignedToFloatRte),
        I::V_CVT_F64_U32 => (Ty::I32, Ty::F64, Cvt::UnsignedToFloatRte),
        I::V_CVT_I32_F32 => (Ty::F32, Ty::I32, Cvt::FloatToSignedSatRtz),
        I::V_CVT_U32_F32 => (Ty::F32, Ty::I32, Cvt::FloatToUnsignedSatRtz),
        I::V_CVT_I32_F64 => (Ty::F64, Ty::I32, Cvt::FloatToSignedSatRtz),
        _ => return None,
    })
}
fn is_cmpx(op: I) -> bool {
    format!("{op:?}").starts_with("V_CMPX")
}

fn compare(op: I) -> Option<(Ty, Op)> {
    let a = ValueId(0);
    let b = ValueId(1);
    Some(match op {
        I::V_CMP_GT_F32 | I::V_CMPX_GT_F32 => (Ty::F32, Op::FCmp(FloatPred::Ogt, a, b)),
        I::V_CMP_LT_F32 | I::V_CMPX_LT_F32 => (Ty::F32, Op::FCmp(FloatPred::Olt, a, b)),
        I::V_CMP_LE_F32 | I::V_CMPX_LE_F32 => (Ty::F32, Op::FCmp(FloatPred::Ole, a, b)),
        I::V_CMP_GE_F32 | I::V_CMPX_GE_F32 => (Ty::F32, Op::FCmp(FloatPred::Oge, a, b)),
        I::V_CMP_EQ_F32 | I::V_CMPX_EQ_F32 => (Ty::F32, Op::FCmp(FloatPred::Oeq, a, b)),
        I::V_CMP_LG_F32 | I::V_CMPX_LG_F32 => (Ty::F32, Op::FCmp(FloatPred::One, a, b)),
        I::V_CMP_NLT_F32 | I::V_CMPX_NLT_F32 => (Ty::F32, Op::FCmp(FloatPred::Uge, a, b)),
        I::V_CMP_NGT_F32 | I::V_CMPX_NGT_F32 => (Ty::F32, Op::FCmp(FloatPred::Ule, a, b)),
        I::V_CMP_NGE_F32 | I::V_CMPX_NGE_F32 => (Ty::F32, Op::FCmp(FloatPred::Ult, a, b)),
        I::V_CMP_NLE_F32 | I::V_CMPX_NLE_F32 => (Ty::F32, Op::FCmp(FloatPred::Ugt, a, b)),
        I::V_CMP_NEQ_F32 | I::V_CMPX_NEQ_F32 => (Ty::F32, Op::FCmp(FloatPred::Une, a, b)),
        I::V_CMP_GT_F64 | I::V_CMPX_GT_F64 => (Ty::F64, Op::FCmp(FloatPred::Ogt, a, b)),
        I::V_CMP_LT_F64 | I::V_CMPX_LT_F64 => (Ty::F64, Op::FCmp(FloatPred::Olt, a, b)),
        I::V_CMP_LE_F64 | I::V_CMPX_LE_F64 => (Ty::F64, Op::FCmp(FloatPred::Ole, a, b)),
        I::V_CMP_GE_F64 | I::V_CMPX_GE_F64 => (Ty::F64, Op::FCmp(FloatPred::Oge, a, b)),
        I::V_CMP_EQ_F64 | I::V_CMPX_EQ_F64 => (Ty::F64, Op::FCmp(FloatPred::Oeq, a, b)),
        I::V_CMP_LG_F64 | I::V_CMPX_LG_F64 => (Ty::F64, Op::FCmp(FloatPred::One, a, b)),
        I::V_CMP_NLT_F64 | I::V_CMPX_NLT_F64 => (Ty::F64, Op::FCmp(FloatPred::Uge, a, b)),
        I::V_CMP_NGT_F64 | I::V_CMPX_NGT_F64 => (Ty::F64, Op::FCmp(FloatPred::Ule, a, b)),
        I::V_CMP_NGE_F64 | I::V_CMPX_NGE_F64 => (Ty::F64, Op::FCmp(FloatPred::Ult, a, b)),
        I::V_CMP_NLE_F64 | I::V_CMPX_NLE_F64 => (Ty::F64, Op::FCmp(FloatPred::Ugt, a, b)),
        I::V_CMP_NEQ_F64 | I::V_CMPX_NEQ_F64 => (Ty::F64, Op::FCmp(FloatPred::Une, a, b)),
        I::V_CMP_EQ_U64 | I::V_CMPX_EQ_U64 | I::V_CMP_EQ_I64 | I::V_CMPX_EQ_I64 => {
            (Ty::I64, Op::Cmp(IntPred::Eq, a, b))
        }
        I::V_CMP_NE_U64 | I::V_CMPX_NE_U64 | I::V_CMP_NE_I64 | I::V_CMPX_NE_I64 => {
            (Ty::I64, Op::Cmp(IntPred::Ne, a, b))
        }
        I::V_CMP_GT_U64 | I::V_CMPX_GT_U64 => (Ty::I64, Op::Cmp(IntPred::Ugt, a, b)),
        I::V_CMP_LT_U64 | I::V_CMPX_LT_U64 => (Ty::I64, Op::Cmp(IntPred::Ult, a, b)),
        I::V_CMP_GE_U64 | I::V_CMPX_GE_U64 => (Ty::I64, Op::Cmp(IntPred::Uge, a, b)),
        I::V_CMP_LE_U64 | I::V_CMPX_LE_U64 => (Ty::I64, Op::Cmp(IntPred::Ule, a, b)),
        I::V_CMP_GT_I64 | I::V_CMPX_GT_I64 => (Ty::I64, Op::Cmp(IntPred::Sgt, a, b)),
        I::V_CMP_LT_I64 | I::V_CMPX_LT_I64 => (Ty::I64, Op::Cmp(IntPred::Slt, a, b)),
        I::V_CMP_GE_I64 | I::V_CMPX_GE_I64 => (Ty::I64, Op::Cmp(IntPred::Sge, a, b)),
        I::V_CMP_LE_I64 | I::V_CMPX_LE_I64 => (Ty::I64, Op::Cmp(IntPred::Sle, a, b)),
        I::V_CMP_EQ_U32 | I::V_CMPX_EQ_U32 => (Ty::I32, Op::Cmp(IntPred::Eq, a, b)),
        I::V_CMP_NE_U32 | I::V_CMPX_NE_U32 => (Ty::I32, Op::Cmp(IntPred::Ne, a, b)),
        I::V_CMP_GT_U32 | I::V_CMPX_GT_U32 => (Ty::I32, Op::Cmp(IntPred::Ugt, a, b)),
        I::V_CMP_LT_U32 | I::V_CMPX_LT_U32 => (Ty::I32, Op::Cmp(IntPred::Ult, a, b)),
        I::V_CMP_GE_U32 | I::V_CMPX_GE_U32 => (Ty::I32, Op::Cmp(IntPred::Uge, a, b)),
        I::V_CMP_LE_U32 | I::V_CMPX_LE_U32 => (Ty::I32, Op::Cmp(IntPred::Ule, a, b)),
        I::V_CMP_LT_I32 | I::V_CMPX_LT_I32 => (Ty::I32, Op::Cmp(IntPred::Slt, a, b)),
        I::V_CMP_GT_I32 | I::V_CMPX_GT_I32 => (Ty::I32, Op::Cmp(IntPred::Sgt, a, b)),
        _ => return None,
    })
}

#[cfg(test)]
mod tests;

pub(super) mod function;
