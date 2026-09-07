//! Lift ISA operands, state updates and effects into typed SSA.
use super::ir::typed::*;
use super::dialect::{DialectRegistry, Arguments};
use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand};
pub(super) mod memory;
pub(super) mod wave;
mod scalar;
mod compare;
mod dual;
mod packed;
mod division;
mod image;
pub(super) mod half;
mod state;

pub(super) enum Lowering {
    Memory(memory::Memory),
    Wave(wave::YieldAction),
    TypedAlu {
        inputs: Vec<Input>,
        outputs: Vec<Output>,
        scalar: bool,
        expr: VerifiedExpr,
    },
}
#[derive(Clone, Debug)]
pub(super) struct Input {
    pub source: InputSource,
    pub ty: Ty,
}
impl Input {
    /// Resolve encoded constant precision at the ISA boundary. Mask constants
    /// still require lane extraction, so they are not plain I1 constants.
    fn constant_bits(&self) -> Option<u64> {
        if self.ty == Ty::I1 { return None; }
        let InputSource::Operand(source) = self.source else { return None; };
        let bits = match source {
            SourceOperand::LiteralConstant(value) => if self.ty == Ty::F64 { (value as u64) << 32 } else { value as u64 },
            SourceOperand::IntegerConstant(value) => if self.ty == Ty::F64 { value << 32 } else { value },
            SourceOperand::FloatConstant(value) => if self.ty.bits() == 64 { value.to_bits() } else { (value as f32).to_bits() as u64 },
            SourceOperand::ScalarRegister(124) if self.ty.bits() == 32 => 0,
            _ => return None,
        };
        Some(if self.ty.bits() == 32 { bits as u32 as u64 } else { bits })
    }
}
#[derive(Clone, Debug)]
pub(super) enum InputSource {
    Operand(SourceOperand),
    Scc,
    /// Architectural per-work-item state; never a wave reduction.
    MaskBit(u32),
    /// EXEC used by an architectural write, in the existing native predicate
    /// representation rather than an ISA numeric source view.
    ExecPredicate,
}
impl InputSource {
    pub fn operand(&self) -> &SourceOperand {
        match self { Self::Operand(source) => source, _ => panic!("control state is not an ISA source operand") }
    }
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Output {
    Vgpr(u32, Ty),
    Compare(u32),
    Mask(u32),
    /// A fully lifted per-lane mask assignment (predication is already in SSA).
    MaskBit(u32),
    Scalar(u32, Ty),
    Scc,
}
impl Output {
    pub fn ty(self) -> Ty {
        match self {
            Self::Vgpr(_, t) | Self::Scalar(_, t) => t,
            Self::Compare(_) | Self::Mask(_) | Self::MaskBit(_) | Self::Scc => Ty::I1,
        }
    }
}
struct Builder<'r> {
    registry: &'r DialectRegistry,
    inputs: Vec<Input>,
    insts: Vec<ExprInst>,
    next_value: usize,
}
impl<'r> Builder<'r> {
    fn new(registry: &'r DialectRegistry, inputs: Vec<Input>) -> Self {
        Self {
            registry,
            next_value: inputs.len(),
            inputs,
            insts: vec![],
        }
    }
    fn push(&mut self, ty: Ty, op: Op) -> ValueId {
        let id = ValueId(self.next_value);
        self.next_value += 1;
        self.insts.push(ExprInst::Core(ty, op));
        id
    }
    fn target(&mut self, op: super::dialect::TargetOp, args: Arguments) -> Vec<ValueId> {
        let outputs = self.registry.operation(op).expect("missing target provider").outputs.clone();
        let values = (self.next_value..self.next_value + outputs.len()).map(ValueId).collect();
        self.next_value += outputs.len();
        self.insts.push(ExprInst::Target { op, args, outputs });
        values
    }
    fn target_one(&mut self, op: super::dialect::TargetOp, args: Arguments) -> ValueId {
        let values = self.target(op, args);
        assert_eq!(values.len(), 1);
        values[0]
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
    /// Preserve captured VOP3 bit-modifier behavior also on integer encodings
    /// (ISA §7.2.2.1 leaves that combination undefined). This is not integer negation.
    fn bits_mod(&mut self, ty: Ty, bits: u32, mut value: ValueId, abs: u8, neg: u8, index: usize) -> ValueId {
        if abs >> index & 1 != 0 {
            let mask = self.k(ty, if ty == Ty::I64 { u64::MAX } else { u32::MAX as u64 } ^ (1u64 << (bits - 1)));
            value = self.push(ty, Op::Int(IntOp::And, value, mask));
        }
        if neg >> index & 1 != 0 {
            let sign = self.k(ty, 1u64 << (bits - 1));
            value = self.push(ty, Op::Int(IntOp::Xor, value, sign));
        }
        value
    }
    fn output_mod(&mut self, ty: Ty, mut value: ValueId, clamp: u8, omod: u8) -> ValueId {
        if omod != 0 {
            let factor: f64 = match omod { 1 => 2.0, 2 => 4.0, 3 => 0.5, _ => unreachable!() };
            let factor = self.k(ty, if ty == Ty::F64 { factor.to_bits() } else { (factor as f32).to_bits() as u64 });
            value = self.push(ty, Op::Float(FloatOp::Mul, value, factor));
            // ISA §7.2.3.1: OMOD flushes output subnormals and maps -0 to +0.
            let word_ty = if ty == Ty::F64 { Ty::I64 } else { Ty::I32 };
            let word = self.push(word_ty, Op::Convert(Cvt::Bitcast, word_ty, value));
            let mask = self.k(word_ty, if ty == Ty::F64 { 0x7ff0_0000_0000_0000 } else { 0x7f80_0000 });
            let exponent = self.push(word_ty, Op::Int(IntOp::And, word, mask));
            let zero = self.k(word_ty, 0);
            let tiny = self.push(Ty::I1, Op::Cmp(IntPred::Eq, exponent, zero));
            let zero = self.k(ty, 0);
            value = self.push(ty, Op::Select(tiny, zero, value));
        }
        if clamp != 0 {
            let zero = self.k(ty, 0);
            let one = self.k(ty, if ty == Ty::F64 { 1f64.to_bits() } else { 1f32.to_bits() as u64 });
            // Ordered > maps NaNs, negative values and both zeros to +0.
            let positive = self.push(Ty::I1, Op::FCmp(FloatPred::Ogt, value, zero));
            value = self.push(ty, Op::Float(FloatOp::MinNum, value, one));
            value = self.push(ty, Op::Select(positive, value, zero));
        }
        value
    }
    fn finish(self, output: Output, result: ValueId) -> Lowering {
        self.finish_many(false, vec![(output, result)])
    }
    fn finish_many(self, scalar: bool, results: Vec<(Output, ValueId)>) -> Lowering {
        let types: Vec<_> = self.inputs.iter().map(|i| i.ty)
            .chain(self.insts.iter().flat_map(|i| i.result_types().iter().copied())).collect();
        for &(output, value) in &results {
            assert_eq!(types.get(value.0), Some(&output.ty()), "incorrect ALU output type");
        }
        let expr = Expr {
            params: self.inputs.iter().map(|i| i.ty).collect(),
            insts: self.insts,
            results: results.iter().map(|x| x.1).collect(),
        }
        .verify_with(self.registry)
        .expect("invalid ALU lift");
        Lowering::TypedAlu {
            inputs: self.inputs,
            outputs: results.into_iter().map(|x| x.0).collect(),
            scalar,
            expr,
        }
    }
}
fn input(source: SourceOperand, ty: Ty) -> Input {
    Input { source: InputSource::Operand(source), ty }
}
/// Literal words in signed 64-bit operand positions are sign-extended;
/// unsigned and bit-pattern positions retain their zero-extension rule.
fn signed_input(source: SourceOperand, ty: Ty) -> Input {
    input(match (source, ty) {
        (SourceOperand::LiteralConstant(word), Ty::I64) => SourceOperand::IntegerConstant(word as i32 as i64 as u64),
        _ => source,
    }, ty)
}

/// Arithmetic results and flags are separate SSA values. All operands are
/// bound before either destination is written, including aliased carry words.
fn carry(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    let (op, mut sources, dst, mask) = match inst {
        InstFormat::VOP2(i) if matches!(i.op, I::V_ADD_CO_CI_U32 | I::V_SUB_CO_CI_U32 | I::V_SUBREV_CO_CI_U32) =>
            (i.op, vec![i.src0.clone(), SourceOperand::VectorRegister(i.vsrc1), SourceOperand::ScalarRegister(106)], i.vdst, 106),
        InstFormat::VOP3SD(i) if matches!(i.op, I::V_ADD_CO_U32 | I::V_ADD_CO_CI_U32 | I::V_SUB_CO_U32 | I::V_SUBREV_CO_U32 | I::V_SUB_CO_CI_U32 | I::V_SUBREV_CO_CI_U32 | I::V_MAD_CO_U64_U32) =>
            (i.op, vec![i.src0.clone(), i.src1.clone(), i.src2.clone()], i.vdst, i.sdst),
        _ => return None,
    };
    let mad = matches!(op, I::V_MAD_CO_U64_U32);
    let carry_in = matches!(op, I::V_ADD_CO_CI_U32 | I::V_SUB_CO_CI_U32 | I::V_SUBREV_CO_CI_U32);
    if !carry_in && !mad { sources.truncate(2); }
    let mut b = Builder::new(registry, sources.into_iter().enumerate()
        .map(|(i, s)| input(s, if i < 2 { Ty::I32 } else if mad { Ty::I64 } else { Ty::I1 })).collect());
    let neg = if let InstFormat::VOP3SD(i) = inst { i.neg } else { 0 };
    let a = b.bits_mod(Ty::I32, 32, ValueId(0), 0, neg, 0);
    let c = b.bits_mod(Ty::I32, 32, ValueId(1), 0, neg, 1);
    let a = b.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, a));
    let c = b.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, c));
    let (result, flag, ty) = if mad {
        let product = b.push(Ty::I64, Op::Int(IntOp::Mul, a, c));
        let sum = b.push(Ty::I64, Op::Int(IntOp::Add, product, ValueId(2)));
        let flag = b.push(Ty::I1, Op::Cmp(IntPred::Ult, sum, product));
        (sum, flag, Ty::I64)
    } else {
        let (a, c) = if matches!(op, I::V_SUBREV_CO_U32 | I::V_SUBREV_CO_CI_U32) { (c, a) } else { (a, c) };
        let arithmetic = if matches!(op, I::V_SUB_CO_U32 | I::V_SUBREV_CO_U32 | I::V_SUB_CO_CI_U32 | I::V_SUBREV_CO_CI_U32) { IntOp::Sub } else { IntOp::Add };
        let mut wide = b.push(Ty::I64, Op::Int(arithmetic, a, c));
        if carry_in {
            let cin = b.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, ValueId(2)));
            wide = b.push(Ty::I64, Op::Int(arithmetic, wide, cin));
        }
        let result = b.push(Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, wide));
        let shift = b.k(Ty::I64, 32);
        let high = b.push(Ty::I64, Op::Int(IntOp::LShr, wide, shift));
        let zero = b.k(Ty::I64, 0);
        let flag = b.push(Ty::I1, Op::Cmp(IntPred::Ne, high, zero));
        (result, flag, Ty::I32)
    };
    let mut results = vec![(Output::Vgpr(dst as u32, ty), result)];
    if mask != 124 { results.push((Output::Mask(mask as u32), flag)); }
    Some(b.finish_many(false, results))
}

#[cfg(test)]
pub(super) fn instruction(inst: &InstFormat) -> Lowering { instruction_with_registry(inst, &DialectRegistry::rdna4()) }

pub(super) fn instruction_with_registry(inst: &InstFormat, registry: &DialectRegistry) -> Lowering {
    if let Some(action) = wave::instruction(inst) { return Lowering::Wave(action); }
    if let Some(memory) = memory::instruction(inst) { return Lowering::Memory(memory); }
    if let Some(image) = image::instruction(inst, registry) { return image; }
    if let Some(alu) = half::instruction(inst, registry) { return alu; }
    if let Some(alu) = division::instruction(inst, registry) { return alu; }
    if let Some(alu) = carry(inst, registry).or_else(|| scalar::instruction(inst, registry)) { return alu; }
    if let Some(alu) = dual::instruction(inst, registry) { return alu; }
    if let Some(alu) = compare::instruction(inst, registry) { return alu; }
    if let Some(alu) = packed::instruction(inst, registry) { return alu; }
    let (op, src, dst, abs, neg, cm, omod) = match inst {
        InstFormat::VOP1(i) => (i.op, vec![i.src0.clone()], i.vdst, 0, 0, 0, 0),
        InstFormat::VOP2(i) => (
            i.op,
            vec![i.src0.clone(), SourceOperand::VectorRegister(i.vsrc1)],
            i.vdst,
            0,
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
            i.omod,
        ),
        _ => panic!("instruction has no typed IR lowering: {:?}", inst),
    };
    let a = ValueId(0);
    let b = ValueId(1);
    let c = ValueId(2);
    // Decide input types/arity before assigning IDs. Literals retain their ISA
    // interpretation in Input, while constants introduced by the lift are bits.
    let cvt = conversion(op);
    let target = super::dialect::rdna4::unary(registry, op)
        .or_else(|| super::dialect::rdna4::binary(registry, op));
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
    }.or_else(|| target.map(|id| registry.operation(id).unwrap().inputs[0]));
    let ty = cvt.map(|x| x.0)
        .or(float_ty)
        .unwrap_or(Ty::I32);
    let arity = if let Some(target) = target {
        registry.operation(target).unwrap().inputs.len()
    } else if cvt.is_some() {
        1
    } else {
        match op {
            I::V_MOV_B32 | I::V_NOT_B32 | I::V_BFREV_B32 | I::V_CLZ_I32_U32
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
            | I::V_MAD_U32_U24 | I::V_MAD_I32_I24
            | I::V_BFE_U32
            | I::V_BFI_B32
            | I::V_ALIGNBIT_B32
            | I::V_FMA_F32
            | I::V_FMA_F64 => 3,
            _ => 2,
        }
    };
    if src.len() < arity {
        panic!("instruction has no typed IR lowering: {:?}", inst);
    }
    let mut inputs: Vec<_> = src
        .iter()
        .take(arity)
        .cloned()
        .map(|s| input(s, ty))
        .collect();
    if let Some(target) = target {
        for (input, &ty) in inputs.iter_mut().zip(registry.operation(target).unwrap().inputs) { input.ty = ty; }
    }
    if matches!(op, I::V_FMAC_F32) {
        inputs.push(input(SourceOperand::VectorRegister(dst), Ty::F32));
    }
    if matches!(op, I::V_FMAMK_F32 | I::V_FMAAK_F32) {
        let literal = match inst {
            InstFormat::VOP2(i) => i.literal_constant,
            _ => None,
        };
        let Some(literal) = literal else {
            panic!("instruction has no typed IR lowering: {:?}", inst);
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
    if matches!(op, I::V_LSHLREV_B64 | I::V_LSHRREV_B64 | I::V_ASHRREV_I64) {
        inputs[1].ty = Ty::I64;
    }
    let mut q = Builder::new(registry, inputs);
    let mut values = vec![a, b, c];
    for index in 0..arity {
        let input_ty = q.inputs[index].ty;
        values[index] = if input_ty.integer() {
            let bits = if matches!(op, I::V_ADD_NC_U16 | I::V_LSHLREV_B16 | I::V_LSHRREV_B16) { 16 } else { input_ty.bits() };
            q.bits_mod(input_ty, bits, values[index], abs, neg, index)
        } else {
            q.float_mod(input_ty, values[index], abs, neg, index)
        };
    }

    let (a, b, c) = (values[0], values[1], values[2]);
    let mut output = Output::Vgpr(dst as u32, float_ty.unwrap_or(Ty::I32));
    let result = if let Some((_, to, kind)) = cvt {
        output = Output::Vgpr(dst as u32, to);
        q.push(to, Op::Convert(kind, to, a))
    } else if let Some(target) = target {
        let to = registry.operation(target).unwrap().outputs[0];
        output = if matches!(op, I::V_S_RCP_F32) { Output::Scalar(dst as u32, to) } else { Output::Vgpr(dst as u32, to) };
        q.target_one(target, if arity == 1 { Arguments::Unary(a) } else { Arguments::Binary([a, b]) })
    } else {
        match op {
            I::V_MOV_B32 => a,
            I::V_NOT_B32 => { let mask = q.k(Ty::I32, u32::MAX as u64); q.int(IntOp::Xor, a, mask) }
            I::V_BFREV_B32 => q.push(Ty::I32, Op::ReverseBits(a)),
            I::V_BCNT_U32_B32 => { let count = q.push(Ty::I32, Op::PopulationCount(a)); q.int(IntOp::Add, count, b) }
            I::V_MAD_U32_U24 | I::V_MAD_I32_I24 => {
                let mut extend = |value| {
                    if matches!(op, I::V_MAD_I32_I24) {
                        let shift = q.k(Ty::I32, 8);
                        let left = q.int(IntOp::Shl, value, shift);
                        q.int(IntOp::AShr, left, shift)
                    } else {
                        let mask = q.k(Ty::I32, 0x00ff_ffff);
                        q.int(IntOp::And, value, mask)
                    }
                };
                let a = extend(a); let b = extend(b);
                let product = q.int(IntOp::Mul, a, b);
                q.int(IntOp::Add, product, c)
            },
            I::V_CLZ_I32_U32 => {
                let count = q.push(Ty::I32, Op::LeadingZeros(a));
                let zero = q.k(Ty::I32, 0);
                let empty = q.push(Ty::I1, Op::Cmp(IntPred::Eq, a, zero));
                let missing = q.k(Ty::I32, u32::MAX as u64);
                q.push(Ty::I32, Op::Select(empty, missing, count))
            }
            I::V_ADD_NC_U32 => {
                let sum = q.int(IntOp::Add, a, b);
                if cm != 0 {
                    let overflow = q.push(Ty::I1, Op::Cmp(IntPred::Ult, sum, a));
                    let max = q.k(Ty::I32, u32::MAX as u64);
                    q.push(Ty::I32, Op::Select(overflow, max, sum))
                } else { sum }
            },
            I::V_ALIGNBIT_B32 => {
                let high = q.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, a));
                let low = q.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, b));
                let thirty_two = q.k(Ty::I64, 32);
                let high = q.push(Ty::I64, Op::Int(IntOp::Shl, high, thirty_two));
                let joined = q.push(Ty::I64, Op::Int(IntOp::Or, high, low));
                let mask = q.k(Ty::I32, 31);
                let amount = q.int(IntOp::And, c, mask);
                let amount = q.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, amount));
                let value = q.push(Ty::I64, Op::Int(IntOp::LShr, joined, amount));
                q.push(Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, value))
            }
            I::V_ADD_NC_U16 => {
                let mask = q.k(Ty::I32, 0xffff);
                let a = q.int(IntOp::And, a, mask);
                let b = q.int(IntOp::And, b, mask);
                let sum = q.int(IntOp::Add, a, b);
                if cm != 0 {
                    let overflow = q.push(Ty::I1, Op::Cmp(IntPred::Ugt, sum, mask));
                    q.push(Ty::I32, Op::Select(overflow, mask, sum))
                } else { q.int(IntOp::And, sum, mask) }
            }
            I::V_LSHLREV_B16 | I::V_LSHRREV_B16 => {
                let mask = q.k(Ty::I32, 0xffff);
                let data = q.int(IntOp::And, b, mask);
                let low_four = q.k(Ty::I32, 15);
                let amount = q.int(IntOp::And, a, low_four);
                let shift = if matches!(op, I::V_LSHLREV_B16) { IntOp::Shl } else { IntOp::LShr };
                let value = q.int(shift, data, amount);
                q.int(IntOp::And, value, mask)
            }
            I::V_SUB_NC_U32 | I::V_SUBREV_NC_U32 => {
                let (a, b) = if matches!(op, I::V_SUBREV_NC_U32) { (b, a) } else { (a, b) };
                let difference = q.int(IntOp::Sub, a, b);
                if cm != 0 {
                    let borrow = q.push(Ty::I1, Op::Cmp(IntPred::Ult, a, b));
                    let zero = q.k(Ty::I32, 0);
                    q.push(Ty::I32, Op::Select(borrow, zero, difference))
                } else { difference }
            },
            I::V_AND_B32 => q.int(IntOp::And, a, b),
            I::V_OR_B32 => q.int(IntOp::Or, a, b),
            I::V_XOR_B32 => q.int(IntOp::Xor, a, b),
            I::V_MUL_LO_U32 => q.int(IntOp::Mul, a, b),
            I::V_MUL_U32_U24 | I::V_MUL_I32_I24 => {
                let (a, b) = if matches!(op, I::V_MUL_I32_I24) {
                    let shift = q.k(Ty::I32, 8);
                    let a = q.int(IntOp::Shl, a, shift);
                    let a = q.int(IntOp::AShr, a, shift);
                    let b = q.int(IntOp::Shl, b, shift);
                    let b = q.int(IntOp::AShr, b, shift);
                    (a, b)
                } else {
                    let mask = q.k(Ty::I32, 0x00ff_ffff);
                    (q.int(IntOp::And, a, mask), q.int(IntOp::And, b, mask))
                };
                q.int(IntOp::Mul, a, b)
            }
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
            I::V_CNDMASK_B32 => q.push(Ty::I32, Op::Select(c, b, a)),
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
            I::V_LSHLREV_B64 | I::V_LSHRREV_B64 | I::V_ASHRREV_I64 => {
                let a = q.push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, a));
                output = Output::Vgpr(dst as u32, Ty::I64);
                q.push(
                    Ty::I64,
                    Op::Int(
                        if matches!(op, I::V_LSHLREV_B64) {
                            IntOp::Shl
                        } else if matches!(op, I::V_ASHRREV_I64) {
                            IntOp::AShr
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
                let mut result = q.push(ty, Op::Float(f, a, b));
                if matches!(op, I::V_SUB_F32 | I::V_SUBREV_F32) {
                    // Captured subtraction cases preserve the NaN operand's
                    // payload and negate the subtrahend's NaN sign.
                    let nan_a = q.push(Ty::I1, Op::FCmp(FloatPred::Uno, a, a));
                    let nan_b = q.push(Ty::I1, Op::FCmp(FloatPred::Uno, b, b));
                    let word = q.push(Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, b));
                    let quiet = q.k(Ty::I32, 0x0040_0000);
                    let word = q.int(IntOp::Or, word, quiet);
                    let sign = q.k(Ty::I32, 0x8000_0000);
                    let word = q.int(IntOp::Xor, word, sign);
                    let nan = q.push(Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, word));
                    let right_nan = q.push(Ty::F32, Op::Select(nan_b, nan, result));
                    result = q.push(Ty::F32, Op::Select(nan_a, result, right_nan));
                }
                result
            }
            I::V_FMA_F64 => q.push(ty, Op::MulAdd(a, b, c)),
            I::V_FMA_F32 | I::V_FMAC_F32 | I::V_FMAAK_F32 => q.push(ty, Op::Fma(a, b, c)),
            I::V_FMAMK_F32 => q.push(ty, Op::Fma(a, c, b)),
            _ => panic!("instruction has no typed IR lowering: {:?}", inst),
        }
    };

    let result = if !output.ty().integer() { q.output_mod(output.ty(), result, cm, omod) } else { result };
    if matches!(op, I::V_S_RCP_F32) { q.finish_many(true, vec![(output, result)]) } else { q.finish(output, result) }
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
        I::V_CVT_U32_F64 => (Ty::F64, Ty::I32, Cvt::FloatToUnsignedSatRtz),
        I::V_CVT_F32_F64 => (Ty::F64, Ty::F32, Cvt::FloatResizeRte),
        I::V_CVT_F64_F32 => (Ty::F32, Ty::F64, Cvt::FloatResizeRte),
        _ => return None,
    })
}
#[cfg(test)]
mod tests;

pub(super) mod function;
