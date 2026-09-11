//! Immutable, compiler-owned target operation registry. Core IR only retains
//! stable opaque IDs; signatures and lowering belong to the provider.
use super::ir::{Ty, ValueId};
use std::collections::BTreeMap;
use llvm_sys::prelude::LLVMValueRef;


#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TargetOp { dialect: u32, operation: u32 }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Arguments {
    Unary(ValueId),
    Binary([ValueId; 2]),
    Ternary([ValueId; 3]),
    Quaternary([ValueId; 4]),
    Thirteen([ValueId; 13]),
    Fourteen([ValueId; 14]),
    Sixteen([ValueId; 16]),
    Fifteen([ValueId; 15]),
}
impl Arguments {
    pub fn map(self, mut f: impl FnMut(ValueId) -> ValueId) -> Self {
        match self {
            Self::Unary(a) => Self::Unary(f(a)), Self::Binary(a) => Self::Binary(a.map(f)),
            Self::Ternary(a) => Self::Ternary(a.map(f)), Self::Quaternary(a) => Self::Quaternary(a.map(f)),
            Self::Thirteen(a) => Self::Thirteen(a.map(f)), Self::Fourteen(a) => Self::Fourteen(a.map(f)),
            Self::Sixteen(a) => Self::Sixteen(a.map(f)), Self::Fifteen(a) => Self::Fifteen(a.map(f)),
        }
    }
    pub fn values(&self) -> &[ValueId] {
        match self { Self::Unary(a) => std::slice::from_ref(a), Self::Binary(a) => a,
            Self::Ternary(a) => a, Self::Quaternary(a) => a, Self::Thirteen(a) => a, Self::Fourteen(a) => a,
            Self::Sixteen(a) => a, Self::Fifteen(a) => a }
    }
}

pub(crate) struct Operation {
    pub name: &'static str,
    pub inputs: &'static [Ty],
    pub outputs: Vec<Ty>,
    pub lower: Implementation,
    pub effect: Effect,
    /// Operand index and largest accepted immediate. The provider owns these
    /// constraints; callers cannot weaken them by supplying another signature.
    pub immediates: &'static [(usize, u64)],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Effect { Pure, ReadGlobal { every_lane: bool } }
pub(crate) enum Implementation {
    Single(unsafe fn(&super::codegen::ops::Emitter, &[LLVMValueRef]) -> LLVMValueRef),
    Multiple(unsafe fn(&super::codegen::ops::Emitter, &[LLVMValueRef]) -> Vec<LLVMValueRef>),
}
impl Operation {
    pub fn verify_immediates(&self, args: Arguments, constant: impl Fn(ValueId) -> Option<u64>) -> Result<(), &'static str> {
        for &(index, maximum) in self.immediates {
            if !args.values().get(index).and_then(|&v| constant(v)).is_some_and(|v| v <= maximum) {
                return Err("target requires an in-range constant operand");
            }
        }
        Ok(())
    }
    pub unsafe fn emit(&self, emitter: &super::codegen::ops::Emitter, args: &[LLVMValueRef]) -> Vec<LLVMValueRef> {
        let values = match self.lower {
            Implementation::Single(lower) => vec![lower(emitter, args)],
            Implementation::Multiple(lower) => lower(emitter, args),
        };
        assert_eq!(values.len(), self.outputs.len(), "provider result count mismatch");
        values
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Registers {
    pub exec: u32,
    pub vcc: u32,
    pub null: u32,
    pub scc_slot: u32,
    pub sgprs: u32,
    pub vgprs: u32,
}

#[derive(Default)]
pub(crate) struct DialectRegistry {
    operations: BTreeMap<TargetOp, Operation>,
    dialects: BTreeMap<u32, &'static str>,
    registers: Registers,
    state: Option<unsafe fn(&super::codegen::ops::Emitter, LLVMValueRef) -> Box<dyn std::any::Any>>,
    idioms: Vec<Box<dyn super::pass::idioms::Idiom>>,
}
impl TargetOp {
    pub fn dialect(self) -> u32 { self.dialect }
}
impl DialectRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn add_dialect(&mut self, id: u32, name: &'static str) { self.dialects.insert(id, name); }
    pub fn set_registers(&mut self, registers: Registers) { self.registers = registers; }
    pub fn set_lowering_state(&mut self, prepare: unsafe fn(&super::codegen::ops::Emitter, LLVMValueRef) -> Box<dyn std::any::Any>) { self.state = Some(prepare); }
    pub fn add_idiom(&mut self, idiom: Box<dyn super::pass::idioms::Idiom>) { self.idioms.push(idiom); }
    pub fn registers(&self) -> Registers { self.registers }
    pub fn idioms(&self) -> &[Box<dyn super::pass::idioms::Idiom>] { &self.idioms }
    pub unsafe fn lowering_state(&self, emitter: &super::codegen::ops::Emitter, sink: LLVMValueRef) -> Option<Box<dyn std::any::Any>> {
        self.state.map(|prepare| prepare(emitter, sink))
    }
    pub fn dialect_name(&self, dialect: u32) -> Option<&'static str> { self.dialects.get(&dialect).copied() }
    #[cfg(test)]
    pub fn dialect_id(&self, name: &str) -> Option<u32> { self.dialects.iter().find_map(|(&id, &n)| (n == name).then_some(id)) }
    pub fn register(&mut self, dialect: u32, operation: u32, spec: Operation) -> Result<TargetOp, &'static str> {
        let op = TargetOp { dialect, operation };
        if self.operations.contains_key(&op) || self.operations.iter().any(|(id, other)| id.dialect == dialect && other.name == spec.name) {
            return Err("duplicate target ID or mnemonic");
        }
        self.operations.insert(op, spec);
        Ok(op)
    }
    pub fn lookup(&self, dialect: u32, name: &str) -> Result<TargetOp, &'static str> {
        self.operations.iter().find_map(|(id, spec)| (id.dialect == dialect && spec.name == name).then_some(*id))
            .ok_or("unregistered target operation")
    }
    pub fn operation(&self, op: TargetOp) -> Result<&Operation, &'static str> {
        self.operations.get(&op).ok_or("unregistered target operation")
    }
    pub fn result_types(&self, op: TargetOp, args: Arguments, types: &[Ty]) -> Result<&[Ty], &'static str> {
        let spec = self.operation(op)?;
        let args = args.values();
        if args.len() != spec.inputs.len() || args.iter().zip(spec.inputs).any(|(id, ty)| types.get(id.0) != Some(ty)) {
            return Err("target signature mismatch");
        }
        Ok(&spec.outputs)
    }
    #[cfg(test)]
    pub fn result_type(&self, op: TargetOp, args: Arguments, types: &[Ty]) -> Result<Ty, &'static str> {
        match self.result_types(op, args, types)? { [ty] => Ok(*ty), _ => Err("operation has multiple results") }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::targets::rdna4::dialect as rdna4;
    #[test]
    fn registry_rejects_collisions_missing_providers_and_bad_types() {
        let mut registry = crate::rdna_spmd::targets::rdna4::registry();
        let id = rdna4::unary(&registry, crate::instructions::I::V_RCP_F32).unwrap();
        assert!(rdna4::register(&mut registry).is_err());
        assert!(DialectRegistry::default().operation(id).is_err());
        assert!(registry.lookup(id.dialect, "unknown").is_err());
        assert_eq!(registry.result_type(id, Arguments::Unary(ValueId(0)), &[Ty::F32]), Ok(Ty::F32));
        assert!(registry.result_type(id, Arguments::Unary(ValueId(0)), &[Ty::I32]).is_err());
        assert!(registry.result_type(id, Arguments::Unary(ValueId(1)), &[Ty::F32]).is_err());
    }

    #[test]
    fn target_provider_matches_reference_at_all_widths() {
        use crate::instructions::I;
        use crate::rdna_spmd::{jit, codegen::ops::Emitter};
        use llvm_sys::core::*;
        use std::sync::Arc;
        let registry = Arc::new(crate::rdna_spmd::targets::rdna4::registry());
        for opcode in [I::V_RCP_F32, I::V_RCP_F64, I::V_RSQ_F32, I::V_RSQ_F64,
            I::V_SQRT_F32, I::V_SQRT_F64, I::V_FLOOR_F32, I::V_FLOOR_F64,
            I::V_CEIL_F32, I::V_TRUNC_F32, I::V_TRUNC_F64, I::V_RNDNE_F32,
            I::V_RNDNE_F64, I::V_FRACT_F64, I::V_FREXP_MANT_F32,
            I::V_FREXP_MANT_F64, I::V_FREXP_EXP_I32_F32, I::V_FREXP_EXP_I32_F64,
            I::V_CVT_F32_F16, I::V_CVT_F16_F32,
            I::V_EXP_F32, I::V_LOG_F32, I::V_SIN_F32, I::V_COS_F32,
            I::V_LDEXP_F32, I::V_LDEXP_F64, I::V_TRIG_PREOP_F64] {
            let target = rdna4::unary(&registry, opcode).or_else(|| rdna4::binary(&registry, opcode)).unwrap();
            let spec = registry.operation(target).unwrap();
            let input_ty = spec.inputs[0]; let output_ty = spec.outputs[0];
            let inputs: Vec<u64> = if input_ty == Ty::I32 {
                vec![0, 0x8000, 1, 0x83ff, 0x0400, 0x3c00, 0x3e00, 0xbc00,
                    0x7bff, 0x7c00, 0xfc00, 0x7d23, 0xfe45, 0xabcd3c00]
            } else if input_ty == Ty::F32 {
                vec![0, 0x80000000, 1, 0x807fffff, 0x800000, 0x3f800000,
                    0x3fc00000, 0x40200000, 0xbf000000, 0xc0200000, 0x7f7fffff,
                    0x7f800000, 0xff800000, 0x7fa12345, 0xffc12345,
                    0x3e800000, 0xbe800000, 0x3f000000, 0x40490fdb]
            } else {
                vec![0, 0x8000000000000000, 1, 0x800fffffffffffff, 0x0010000000000000,
                    0x3ff0000000000000, 0x3ff8000000000000, 0x4004000000000000,
                    0xbfe0000000000000, 0xc004000000000000, 0x7fefffffffffffff,
                    0x7ff0000000000000, 0xfff0000000000000, 0x7ff1234512341234, 0xfff8234512341234,
                    0x3fd000000000000b] // Witness against successive subnormal rounding.
            };
            let binary = spec.inputs.len() == 2;
            let exponents = if binary { vec![i32::MIN, -2200, -1075, -1023, -1022, -1021,
                -970, -150, -127, -126, -125, -1, 0, 1, 1023, 1024, 2100, i32::MAX] } else { vec![0] };
            let cases: Vec<_> = inputs.iter().flat_map(|&bits| exponents.iter().map(move |&exp| (bits, exp))).collect();
            for width in [0, 1, 2, 4, 8, 16] {
                unsafe {
                    let module = jit::Module::new("target_reference");
                    let b = module.builder; let ctx = module.ctx; let n = b"\0".as_ptr().cast();
                    let pointer = LLVMPointerTypeInContext(ctx, 0);
                    let ft = LLVMFunctionType(LLVMVoidTypeInContext(ctx), [pointer, pointer, pointer].as_mut_ptr(), 3, 0);
                    let f = LLVMAddFunction(module.module, b"kernel\0".as_ptr().cast(), ft);
                    let entry = LLVMAppendBasicBlockInContext(ctx, f, n);
                    LLVMPositionBuilderAtEnd(b, entry);
                    let emitter = Emitter::new(b, (width != 0).then_some(width), registry.clone());
                    let input = LLVMBuildLoad2(b, emitter.ty(input_ty), LLVMGetParam(f, 0), n);
                    LLVMSetAlignment(input, 4);
                    let mut args = Arguments::Unary(ValueId(0));
                    let mut values = vec![input]; let mut types = vec![input_ty];
                    if binary {
                        let exponent = LLVMBuildLoad2(b, emitter.ty(Ty::I32), LLVMGetParam(f, 1), n);
                        LLVMSetAlignment(exponent, 4); values.push(exponent); types.push(Ty::I32);
                        args = Arguments::Binary([ValueId(0), ValueId(1)]);
                    }
                    assert_eq!(registry.result_type(target, args, &types), Ok(output_ty));
                    let output = emitter.target(target, args, &values)[0];
                    let store = LLVMBuildStore(b, output, LLVMGetParam(f, 2)); LLVMSetAlignment(store, 4);
                    LLVMBuildRetVoid(b);
                    let code = module.finish(if width == 0 { jit::Mode::Scalar } else { jit::Mode::Packet });
                    let run: unsafe extern "C" fn(*const u32, *const i32, *mut u32) = std::mem::transmute(code.address() as usize);
                    let w = width.max(1) as usize;
                    let input_words = input_ty.bits() as usize / 32; let output_words = output_ty.bits() as usize / 32;
                    let mut src = vec![0u32; w * input_words]; let mut dst = vec![0u32; w * output_words];
                    let mut exp = vec![0i32; w];
                    for start in (0..cases.len()).step_by(w) {
                        for lane in 0..w {
                            let (bits, exponent) = cases[(start + lane) % cases.len()];
                            exp[lane] = exponent;
                            src[lane * input_words] = bits as u32;
                            if input_words == 2 { src[lane * input_words + 1] = (bits >> 32) as u32; }
                        }
                        run(src.as_ptr(), exp.as_ptr(), dst.as_mut_ptr());
                        for lane in 0..w {
                            let got = dst[lane * output_words] as u64 | if output_words == 2 { (dst[lane * output_words + 1] as u64) << 32 } else { 0 };
                            let (bits, exponent) = cases[(start + lane) % cases.len()];
                            // ISA §16.12 specifies scalbn/ldexp, including one
                            // final rounding. libm is independent of native SSA.
                            let expected = if matches!(opcode, I::V_TRIG_PREOP_F64) {
                                rdna4::reference_reduction(bits, exponent as u32)
                            } else if binary {
                                if input_ty == Ty::F32 { libm::scalbnf(f32::from_bits(bits as u32), exponent).to_bits() as u64 }
                                else { libm::scalbn(f64::from_bits(bits), exponent).to_bits() }
                            } else { rdna4::reference(opcode, bits) };
                            let nan = |bits| match output_ty {
                                Ty::F32 => f32::from_bits(bits as u32).is_nan(), Ty::F64 => f64::from_bits(bits).is_nan(), _ => false,
                            };
                            assert!(got == expected || nan(got) && nan(expected), "op={:?} width={} lane={} got={:x} expected={:x}", opcode, width, lane, got, expected);
                        }
                    }
                }
            }
        }
    }
}
