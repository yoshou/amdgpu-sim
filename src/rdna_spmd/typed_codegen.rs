//! Shared scalar/packet LLVM lowering of the core SSA operations.
//! No ISA opcodes, register files or per-lane runtime calls.
use super::ir::typed::*;
use llvm::core::*;
use llvm::prelude::*;
use llvm_sys as llvm;

pub(super) struct Emitter {
    pub b: LLVMBuilderRef,
    module: LLVMModuleRef,
    ctx: LLVMContextRef,
    width: Option<u32>,
}
impl Emitter {
    pub unsafe fn new(b: LLVMBuilderRef, width: Option<u32>) -> Self {
        let module = LLVMGetGlobalParent(LLVMGetBasicBlockParent(LLVMGetInsertBlock(b)));
        Self {
            b,
            module,
            ctx: LLVMGetModuleContext(module),
            width,
        }
    }
    unsafe fn ty(&self, t: Ty) -> LLVMTypeRef {
        let t = match t {
            Ty::I1 => LLVMInt1TypeInContext(self.ctx),
            Ty::I32 => LLVMInt32TypeInContext(self.ctx),
            Ty::I64 => LLVMInt64TypeInContext(self.ctx),
            Ty::F32 => LLVMFloatTypeInContext(self.ctx),
            Ty::F64 => LLVMDoubleTypeInContext(self.ctx),
        };
        self.width.map_or(t, |w| LLVMVectorType(t, w))
    }
    unsafe fn constant(&self, ty: Ty, bits: u64) -> LLVMValueRef {
        let t = LLVMIntTypeInContext(self.ctx, ty.bits());
        let v = LLVMConstInt(t, bits, 0);
        let v = if let Some(w) = self.width {
            LLVMConstVector(vec![v; w as usize].as_mut_ptr(), w)
        } else {
            v
        };
        if ty.integer() {
            v
        } else {
            LLVMConstBitCast(v, self.ty(ty))
        }
    }
    fn suffix(&self, t: Ty) -> String {
        let t = match t {
            Ty::I1 => "i1",
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
        };
        self.width.map_or_else(|| t.into(), |w| format!("v{w}{t}"))
    }
    unsafe fn call(&self, name: &str, ret: Ty, args: &[LLVMValueRef]) -> LLVMValueRef {
        let name = std::ffi::CString::new(name).unwrap();
        let mut types: Vec<_> = args.iter().map(|&v| LLVMTypeOf(v)).collect();
        let ft = LLVMFunctionType(self.ty(ret), types.as_mut_ptr(), types.len() as u32, 0);
        let mut f = LLVMGetNamedFunction(self.module, name.as_ptr());
        if f.is_null() {
            f = LLVMAddFunction(self.module, name.as_ptr(), ft);
        }
        LLVMBuildCall2(
            self.b,
            ft,
            f,
            args.to_vec().as_mut_ptr(),
            args.len() as u32,
            b"\0".as_ptr().cast(),
        )
    }
    pub unsafe fn op(&self, ty: Ty, op: Op, values: &[LLVMValueRef]) -> LLVMValueRef {
        let b = self.b;
        let n = b"\0".as_ptr().cast();
        let v = |id: ValueId| values[id.0];
        match op {
            Op::Const(t, bits) => self.constant(t, bits),
            Op::Int(op, a, c) => {
                let (a, mut c) = (v(a), v(c));
                if matches!(op, IntOp::Shl | IntOp::LShr | IntOp::AShr) {
                    c = LLVMBuildAnd(b, c, self.constant(ty, (ty.bits() - 1) as u64), n);
                }
                match op {
                    IntOp::Add => LLVMBuildAdd(b, a, c, n),
                    IntOp::Sub => LLVMBuildSub(b, a, c, n),
                    IntOp::Mul => LLVMBuildMul(b, a, c, n),
                    IntOp::And => LLVMBuildAnd(b, a, c, n),
                    IntOp::Or => LLVMBuildOr(b, a, c, n),
                    IntOp::Xor => LLVMBuildXor(b, a, c, n),
                    IntOp::Shl => LLVMBuildShl(b, a, c, n),
                    IntOp::LShr => LLVMBuildLShr(b, a, c, n),
                    IntOp::AShr => LLVMBuildAShr(b, a, c, n),
                }
            }
            Op::Cmp(p, a, c) => LLVMBuildICmp(
                b,
                match p {
                    IntPred::Eq => llvm::LLVMIntPredicate::LLVMIntEQ,
                    IntPred::Ne => llvm::LLVMIntPredicate::LLVMIntNE,
                    IntPred::Ult => llvm::LLVMIntPredicate::LLVMIntULT,
                    IntPred::Ugt => llvm::LLVMIntPredicate::LLVMIntUGT,
                    IntPred::Ule => llvm::LLVMIntPredicate::LLVMIntULE,
                    IntPred::Uge => llvm::LLVMIntPredicate::LLVMIntUGE,
                    IntPred::Slt => llvm::LLVMIntPredicate::LLVMIntSLT,
                    IntPred::Sgt => llvm::LLVMIntPredicate::LLVMIntSGT,
                    IntPred::Sle => llvm::LLVMIntPredicate::LLVMIntSLE,
                    IntPred::Sge => llvm::LLVMIntPredicate::LLVMIntSGE,
                },
                v(a),
                v(c),
                n,
            ),
            Op::FCmp(p, a, c) => LLVMBuildFCmp(
                b,
                match p {
                    FloatPred::Oeq => llvm::LLVMRealPredicate::LLVMRealOEQ,
                    FloatPred::Ogt => llvm::LLVMRealPredicate::LLVMRealOGT,
                    FloatPred::Oge => llvm::LLVMRealPredicate::LLVMRealOGE,
                    FloatPred::Olt => llvm::LLVMRealPredicate::LLVMRealOLT,
                    FloatPred::Ole => llvm::LLVMRealPredicate::LLVMRealOLE,
                    FloatPred::One => llvm::LLVMRealPredicate::LLVMRealONE,
                    FloatPred::Ord => llvm::LLVMRealPredicate::LLVMRealORD,
                    FloatPred::Uno => llvm::LLVMRealPredicate::LLVMRealUNO,
                    FloatPred::Ueq => llvm::LLVMRealPredicate::LLVMRealUEQ,
                    FloatPred::Ugt => llvm::LLVMRealPredicate::LLVMRealUGT,
                    FloatPred::Uge => llvm::LLVMRealPredicate::LLVMRealUGE,
                    FloatPred::Ult => llvm::LLVMRealPredicate::LLVMRealULT,
                    FloatPred::Ule => llvm::LLVMRealPredicate::LLVMRealULE,
                    FloatPred::Une => llvm::LLVMRealPredicate::LLVMRealUNE,
                },
                v(a),
                v(c),
                n,
            ),
            Op::Select(c, a, d) => LLVMBuildSelect(b, v(c), v(a), v(d), n),
            Op::Float(op, a, c) => match op {
                FloatOp::Add => LLVMBuildFAdd(b, v(a), v(c), n),
                FloatOp::Sub => LLVMBuildFSub(b, v(a), v(c), n),
                FloatOp::Mul => LLVMBuildFMul(b, v(a), v(c), n),
                FloatOp::Div => LLVMBuildFDiv(b, v(a), v(c), n),
                FloatOp::MinNum | FloatOp::MaxNum => self.call(
                    &format!(
                        "llvm.{}.{}",
                        if op == FloatOp::MinNum {
                            "minnum"
                        } else {
                            "maxnum"
                        },
                        self.suffix(ty)
                    ),
                    ty,
                    &[v(a), v(c)],
                ),
            },
            Op::Unary(op, a) => {
                if op == FloatUnary::Neg {
                    return LLVMBuildFNeg(b, v(a), n);
                }
                let name = match op {
                    FloatUnary::Abs => "fabs",
                    FloatUnary::Sqrt => "sqrt",
                    FloatUnary::Floor => "floor",
                    FloatUnary::Ceil => "ceil",
                    FloatUnary::Trunc => "trunc",
                    FloatUnary::RoundEven => "roundeven",
                    FloatUnary::Neg => unreachable!(),
                };
                self.call(&format!("llvm.{name}.{}", self.suffix(ty)), ty, &[v(a)])
            }
            Op::Fma(a, c, d) | Op::MulAdd(a, c, d) => self.call(
                &format!(
                    "llvm.{}.{}",
                    if matches!(op, Op::Fma(..)) {
                        "fma"
                    } else {
                        "fmuladd"
                    },
                    self.suffix(ty)
                ),
                ty,
                &[v(a), v(c), v(d)],
            ),
            Op::Convert(op, to, a) => {
                let a = v(a);
                let t = self.ty(to);
                match op {
                    Cvt::Bitcast => LLVMBuildBitCast(b, a, t, n),
                    Cvt::ZExt => LLVMBuildZExt(b, a, t, n),
                    Cvt::SExt => LLVMBuildSExt(b, a, t, n),
                    Cvt::Trunc => LLVMBuildTrunc(b, a, t, n),
                    Cvt::SignedToFloatRte => LLVMBuildSIToFP(b, a, t, n),
                    Cvt::UnsignedToFloatRte => LLVMBuildUIToFP(b, a, t, n),
                    Cvt::FloatResizeRte => {
                        if to == Ty::F64 {
                            LLVMBuildFPExt(b, a, t, n)
                        } else {
                            LLVMBuildFPTrunc(b, a, t, n)
                        }
                    }
                    Cvt::FloatToSignedSatRtz | Cvt::FloatToUnsignedSatRtz => {
                        let at = LLVMTypeOf(a);
                        let at = if self.width.is_some() {
                            LLVMGetElementType(at)
                        } else {
                            at
                        };
                        let from = if LLVMGetTypeKind(at) == llvm::LLVMTypeKind::LLVMFloatTypeKind {
                            Ty::F32
                        } else {
                            Ty::F64
                        };
                        self.call(
                            &format!(
                                "llvm.{}.sat.{}.{}",
                                if op == Cvt::FloatToSignedSatRtz {
                                    "fptosi"
                                } else {
                                    "fptoui"
                                },
                                self.suffix(to),
                                self.suffix(from)
                            ),
                            to,
                            &[a],
                        )
                    }
                }
            }
        }
    }
}

/// Values are addressed by the verified function's SSA IDs. Block arguments
/// lower through the existing boundary allocas, which LLVM promotes to phis;
/// within an ALU sequence the effective SSA value is reused directly.
pub(super) struct Values {
    emitter: Emitter,
    values: Vec<LLVMValueRef>,
}
impl Values {
    pub unsafe fn prepare_memory(
        &mut self,
        f: &super::lift::function::Function,
        pc: usize,
        index: usize,
        mut read: impl FnMut(&super::lift::memory::Parameter, bool) -> LLVMValueRef,
    ) {
        use super::ir::typed::cfg::*;
        let plan = &f.blocks[&pc].memory[&index];
        let scalar = plan.memory.scalar();
        for (parameter, id) in &plan.parameters {
            if self.emitter.width.is_none() && matches!(parameter,super::lift::memory::Parameter::ScratchBase|super::lift::memory::Parameter::ScratchSize) {continue;}
            self.values[id.0] = read(parameter, scalar);
        }
        let emitter = Emitter::new(self.emitter.b, if scalar { None } else { self.emitter.width });
        for inst in &f.ir.func().blocks[&BlockId(pc)].insts[plan.core.clone()] {
            let Inst::Core { value, ty, op } = *inst else { unreachable!() };
            self.values[value.0] = emitter.op(ty, op, &self.values);
        }
        if let Some((range,_,_,_))=&plan.flat {
            if self.emitter.width.is_some() {
                for inst in &f.ir.func().blocks[&BlockId(pc)].insts[range.clone()] {
                    let Inst::Core{value,ty,op}=*inst else{unreachable!()};
                    self.values[value.0]=emitter.op(ty,op,&self.values);
                }
            }
        }
    }
    pub fn value(&self, value: ValueId) -> LLVMValueRef {
        let result = self.values[value.0];
        assert!(!result.is_null(), "unmaterialized memory SSA value");
        result
    }
    pub fn memory_data(&self, f: &super::lift::function::Function, pc: usize, index: usize, word: u32) -> LLVMValueRef {
        use super::ir::typed::cfg::*;
        let plan = &f.blocks[&pc].memory[&index];
        let Inst::Effect { inputs, .. } = &f.ir.func().blocks[&BlockId(pc)].insts[plan.effects[word as usize]] else { unreachable!() };
        self.value(inputs[1])
    }
    pub unsafe fn new(
        f: &super::lift::function::Function,
        b: LLVMBuilderRef,
        width: Option<u32>,
    ) -> Self {
        Self {
            emitter: Emitter::new(b, width),
            values: vec![std::ptr::null_mut(); f.ir.func().types.len()],
        }
    }
    pub fn begin_block(&mut self, f: &super::lift::function::Function, pc: usize) {
        use super::ir::typed::cfg::*;
        let block = &f.ir.func().blocks[&BlockId(pc)];
        for &(v, _) in &block.params {
            self.values[v.0] = std::ptr::null_mut();
        }
        for inst in &block.insts {
            match inst {
                Inst::Core { value, .. } => self.values[value.0] = std::ptr::null_mut(),
                Inst::Boundary { outputs, .. } | Inst::Effect { outputs, .. } => {
                    for &(v, _) in outputs {
                        self.values[v.0] = std::ptr::null_mut();
                    }
                }
            }
        }
    }
    pub unsafe fn emit(
        &mut self,
        f: &super::lift::function::Function,
        pc: usize,
        index: usize,
        mut read: impl FnMut(&super::lift::Input) -> LLVMValueRef,
        mut write: impl FnMut(super::lift::Output, LLVMValueRef),
    ) {
        use super::ir::typed::cfg::*;
        let alu = f.blocks[&pc].instructions[index]
            .as_ref()
            .expect("missing typed lowering");
        for (input, id) in &alu.inputs {
            if self.values[id.0].is_null() {
                self.values[id.0] = read(input);
            }
        }
        let block = &f.ir.func().blocks[&BlockId(pc)];
        for inst in &block.insts[alu.core.clone()] {
            let Inst::Core { value, ty, op } = *inst else {
                unreachable!("adapter in core range")
            };
            self.values[value.0] = self.emitter.op(ty, op, &self.values);
        }
        // Keep the effective value in its coalesced boundary slot until its
        // first use, matching the existing emitter's load placement. Loading
        // immediately after every write perturbs LLVM's optimization order
        // even when that SSA definition is never read. Later uses still share
        // the same SSA ID; the lifter invalidates views on overlapping writes.
        write(alu.output, self.values[alu.result.0]);
    }
}
