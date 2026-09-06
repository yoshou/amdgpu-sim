//! Shared scalar/packet LLVM lowering of the core SSA operations.
//! No ISA opcodes, register files or per-lane runtime calls.
use super::ir::typed::*;
use llvm::core::*;
use llvm::prelude::*;
use llvm_sys as llvm;

pub(super) struct Emitter {
    registry: std::sync::Arc<super::dialect::DialectRegistry>,
    pub b: LLVMBuilderRef,
    module: LLVMModuleRef,
    pub(super) ctx: LLVMContextRef,
    width: Option<u32>,
    lane_id: Option<LLVMValueRef>,
}
impl Emitter {
    pub unsafe fn new(b: LLVMBuilderRef, width: Option<u32>, registry: std::sync::Arc<super::dialect::DialectRegistry>) -> Self {
        let module = LLVMGetGlobalParent(LLVMGetBasicBlockParent(LLVMGetInsertBlock(b)));
        Self {
            registry,
            b,
            module,
            ctx: LLVMGetModuleContext(module),
            width,
            lane_id: None,
        }
    }
    pub(super) unsafe fn ty(&self, t: Ty) -> LLVMTypeRef {
        let t = match t {
            Ty::I1 => LLVMInt1TypeInContext(self.ctx),
            Ty::I32 => LLVMInt32TypeInContext(self.ctx),
            Ty::I64 => LLVMInt64TypeInContext(self.ctx),
            Ty::F32 => LLVMFloatTypeInContext(self.ctx),
            Ty::F64 => LLVMDoubleTypeInContext(self.ctx),
        };
        self.shaped(t)
    }
    pub(super) unsafe fn shaped(&self, scalar: LLVMTypeRef) -> LLVMTypeRef {
        self.width.map_or(scalar, |w| LLVMVectorType(scalar, w))
    }
    pub(super) unsafe fn constant(&self, ty: Ty, bits: u64) -> LLVMValueRef {
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
    pub(super) fn suffix(&self, t: Ty) -> String {
        let t = match t {
            Ty::I1 => "i1",
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
        };
        self.width.map_or_else(|| t.into(), |w| format!("v{w}{t}"))
    }
    pub(super) fn width(&self) -> Option<u32> { self.width }
    pub(super) unsafe fn call(&self, name: &str, ret: Ty, args: &[LLVMValueRef]) -> LLVMValueRef {
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
    pub unsafe fn target(&self, op: super::dialect::TargetOp, args: super::dialect::Arguments, values: &[LLVMValueRef]) -> Vec<LLVMValueRef> {
        let spec = self.registry.operation(op).expect("unverified target");
        let args = args.values().iter().map(|id| values[id.0]).collect::<Vec<_>>();
        spec.emit(self, &args)
    }
    pub unsafe fn op(&self, ty: Ty, op: Op, values: &[LLVMValueRef]) -> LLVMValueRef {
        let b = self.b;
        let n = b"\0".as_ptr().cast();
        let v = |id: ValueId| values[id.0];
        match op {
            Op::Env(Env::LaneId) => self.lane_id.expect("LaneId requires the invocation environment"),
            Op::Const(t, bits) => self.constant(t, bits),
            Op::Pack64(a, c) => {
                let lo = LLVMBuildZExt(b, v(a), self.ty(Ty::I64), n);
                let hi = LLVMBuildZExt(b, v(c), self.ty(Ty::I64), n);
                let hi = LLVMBuildShl(b, hi, self.constant(Ty::I64, 32), n);
                LLVMBuildOr(b, lo, hi, n)
            }
            Op::UnpackLo(a) | Op::UnpackHi(a) => {
                let wide = if matches!(op, Op::UnpackHi(_)) {
                    LLVMBuildLShr(b, v(a), self.constant(Ty::I64, 32), n)
                } else { v(a) };
                LLVMBuildTrunc(b, wide, self.ty(Ty::I32), n)
            }
            Op::TrailingZeros(a) => self.call(&format!("llvm.cttz.{}", self.suffix(ty)), ty,
                &[v(a), LLVMConstInt(LLVMInt1TypeInContext(self.ctx), 0, 0)]),
            Op::LeadingZeros(a) => self.call(&format!("llvm.ctlz.{}", self.suffix(ty)), ty,
                &[v(a), LLVMConstInt(LLVMInt1TypeInContext(self.ctx), 0, 0)]),
            Op::PopulationCount(a) => self.call(&format!("llvm.ctpop.{}", self.suffix(ty)), ty, &[v(a)]),
            Op::ReverseBits(a) => self.call(&format!("llvm.bitreverse.{}", self.suffix(ty)), ty, &[v(a)]),
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
/// lower through the register representations' explicit native phis;
/// within an ALU sequence the effective SSA value is reused directly.
pub(super) struct Values {
    emitter: Emitter,
    scalar_emitter: Emitter,
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
        let emitter = Emitter::new(self.emitter.b, if scalar { None } else { self.emitter.width }, self.emitter.registry.clone());
        for &(pair, view) in &plan.pairs {
            self.values[pair.0] = LLVMBuildBitCast(self.emitter.b, self.values[view.0], emitter.ty(Ty::I64), b"\0".as_ptr().cast());
        }
        for inst in &f.ir.func().blocks[&BlockId(pc)].insts[plan.core.clone()] {
            let Inst::Core { value, ty, op } = *inst else { unreachable!() };
            if self.values[value.0].is_null() { self.values[value.0] = emitter.op(ty, op, &self.values); }
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
    pub unsafe fn yield_values(
        &mut self, plan: &super::lift::wave::Plan, frame: LLVMValueRef,
        context: LLVMValueRef, resume: usize,
        mut write: impl FnMut(super::lift::wave::Destination, Ty, LLVMValueRef, LLVMValueRef),
    ) {
        let b = self.emitter.b; let ctx = self.emitter.ctx;
        if plan.local {
            for (&(id,ty), &destination) in plan.results.iter().zip(&plan.destinations) {
                assert_eq!(ty,Ty::I32);
                let value = self.values[id.0];
                write(destination,ty,value,value);
            }
            return;
        }
        let n = b"\0".as_ptr().cast();
        let i32t = LLVMInt32TypeInContext(ctx); let i64t = LLVMInt64TypeInContext(ctx);
        let ptr = LLVMPointerTypeInContext(ctx, 0);
        let width = self.emitter.width.unwrap_or(1) as u64;
        let pointer = |slot: usize| LLVMBuildGEP2(b, i32t, frame,
            [LLVMConstInt(i32t, slot as u64 * width, 0)].as_mut_ptr(), 1, n);
        for (index, (&id, &ty)) in plan.arguments.iter().zip(&plan.layout.inputs).enumerate() {
            use super::yield_values::Argument;
            if matches!(plan.layout.arguments[index], Argument::Constant(_)) { continue; }
            let value = self.values[id.0];
            let bits = match ty {
                Ty::I1 => LLVMBuildZExt(b, value, self.emitter.ty(Ty::I32), n),
                Ty::F32 => LLVMBuildBitCast(b, value, self.emitter.ty(Ty::I32), n),
                Ty::I32 => value,
                _ => unreachable!("wide wave operand"),
            };
            let bits = if matches!(plan.layout.arguments[index], Argument::Uniform) && self.emitter.width.is_some() {
                LLVMBuildExtractElement(b, bits, LLVMConstInt(i32t, 0, 0), n)
            } else { bits };
            LLVMSetAlignment(LLVMBuildStore(b, bits, pointer(index)), 4);
        }
        let ty = LLVMFunctionType(LLVMVoidTypeInContext(ctx), [ptr, i64t, ptr].as_mut_ptr(), 3, 0);
        let name = b"amdgpu_sim_fiber_yield_values\0".as_ptr().cast();
        let mut function = LLVMGetNamedFunction(self.emitter.module, name);
        if function.is_null() { function = LLVMAddFunction(self.emitter.module, name, ty); }
        LLVMBuildCall2(b, ty, function,
            [context, LLVMConstInt(i64t, resume as u64, 0), frame].as_mut_ptr(), 3, n);
        for (index, (&(id, ty), &destination)) in plan.results.iter().zip(&plan.destinations).enumerate() {
            let uniform = plan.layout.uniform_result();
            let ty_bits = if uniform { i32t } else { self.emitter.ty(Ty::I32) };
            let bits = LLVMBuildLoad2(b, ty_bits, pointer(plan.layout.output_base + index), n);
            LLVMSetAlignment(bits, 4);
            let bits = if uniform && self.emitter.width.is_some() {
                let vector = LLVMBuildInsertElement(b, LLVMGetPoison(self.emitter.ty(Ty::I32)), bits, LLVMConstInt(i32t,0,0), n);
                let mask = LLVMConstNull(LLVMVectorType(i32t, width as u32));
                LLVMBuildShuffleVector(b, vector, LLVMGetPoison(self.emitter.ty(Ty::I32)), mask, n)
            } else { bits };
            let result = match ty {
                Ty::I1 => LLVMBuildTrunc(b, bits, self.emitter.ty(Ty::I1), n),
                Ty::F32 => LLVMBuildBitCast(b, bits, self.emitter.ty(Ty::F32), n),
                Ty::I32 => bits,
                _ => unreachable!("wide wave result"),
            };
            self.values[id.0] = result;
            write(destination, ty, result, bits);
        }
    }
    pub unsafe fn prepare_yield(
        &mut self, f: &super::lift::function::Function, pc: usize,
        lane_base: LLVMValueRef,
        mut read: impl FnMut(&super::lift::Input) -> LLVMValueRef,
    ) {
        use super::ir::typed::cfg::*;
        let plan = f.blocks[&pc].yield_values.as_ref().expect("yield has no SSA operands");
        if plan.local {
            let e = &mut self.emitter;
            let n = b"\0".as_ptr().cast(); let i32t = LLVMInt32TypeInContext(e.ctx);
            let base = LLVMBuildTrunc(e.b,lane_base,i32t,n);
            e.lane_id = Some(if let Some(width) = e.width {
                let mut lanes: Vec<_> = (0..width).map(|lane| LLVMConstInt(i32t,lane as u64,0)).collect();
                let initial = LLVMBuildInsertElement(e.b,LLVMGetPoison(e.ty(Ty::I32)),base,LLVMConstInt(i32t,0,0),n);
                let broadcast = LLVMBuildShuffleVector(e.b,initial,LLVMGetPoison(e.ty(Ty::I32)),LLVMConstNull(e.ty(Ty::I32)),n);
                LLVMBuildAdd(e.b,broadcast,LLVMConstVector(lanes.as_mut_ptr(),width),n)
            } else { base });
        }
        for (input, id) in &plan.parameters {
            if self.values[id.0].is_null() || LLVMTypeOf(self.values[id.0]) != self.emitter.ty(input.ty) {
                self.values[id.0] = read(input);
            }
        }
        for inst in &f.ir.func().blocks[&BlockId(pc)].insts[plan.core.clone()] {
            if let Inst::Core { value, ty, op } = inst {
                if self.values[value.0].is_null() {
                    self.values[value.0] = self.emitter.op(*ty, *op, &self.values);
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
            emitter: Emitter::new(b, width, f.registry.clone()),
            scalar_emitter: Emitter::new(b, None, f.registry.clone()),
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
                Inst::Boundary { outputs, .. } | Inst::Effect { outputs, .. } | Inst::Target { outputs, .. } => {
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
        mut read: impl FnMut(&super::lift::Input, bool) -> LLVMValueRef,
        mut write: impl FnMut(super::lift::Output, LLVMValueRef),
    ) {
        use super::ir::typed::cfg::*;
        let alu = f.blocks[&pc].instructions[index]
            .as_ref()
            .expect("missing typed lowering");
        for (input, id) in &alu.inputs {
            let emitter = if alu.scalar { &self.scalar_emitter } else { &self.emitter };
            // One ordinary SGPR definition can have both uniform scalar and
            // broadcast packet uses. Rebind its native representation when
            // the consumer changes shape; its architectural SSA ID is shared.
            if self.values[id.0].is_null() || LLVMTypeOf(self.values[id.0]) != emitter.ty(input.ty) {
                self.values[id.0] = read(input, alu.scalar);
            }
        }
        let block = &f.ir.func().blocks[&BlockId(pc)];
        for &(pair, view) in &alu.pairs {
            let emitter = if alu.scalar { &self.scalar_emitter } else { &self.emitter };
            self.values[pair.0] = LLVMBuildBitCast(self.emitter.b, self.values[view.0], emitter.ty(Ty::I64), b"\0".as_ptr().cast());
        }
        for inst in &block.insts[alu.core.clone()] {
            let emitter = if alu.scalar { &self.scalar_emitter } else { &self.emitter };
            match inst {
                Inst::Core { value, ty, op } => {
                    if self.values[value.0].is_null() { self.values[value.0] = emitter.op(*ty, *op, &self.values); }
                }
                Inst::Target { op, args, outputs, .. } => {
                    let results = emitter.target(*op, *args, &self.values);
                    for (&(id, _), value) in outputs.iter().zip(results) { self.values[id.0] = value; }
                }
                _ => unreachable!("adapter in typed instruction range"),
            }
        }
        // Keep the effective value in its coalesced boundary slot until its
        // first use, matching the existing emitter's load placement. Loading
        // immediately after every write perturbs LLVM's optimization order
        // even when that SSA definition is never read. Later uses still share
        // the same SSA ID; the lifter invalidates views on overlapping writes.
        for &(output, result) in &alu.outputs {
            write(output, self.values[result.0]);
        }
    }
    pub unsafe fn condition(
        &mut self,
        f: &super::lift::function::Function,
        pc: usize,
        mut read: impl FnMut(&super::lift::Input) -> LLVMValueRef,
    ) {
        use super::ir::typed::cfg::*;
        let Some(condition) = &f.blocks[&pc].condition else { return; };
        if self.values[condition.input_value.0].is_null()
            || LLVMTypeOf(self.values[condition.input_value.0]) != self.scalar_emitter.ty(Ty::I1) {
            self.values[condition.input_value.0] = read(&condition.input);
        }
        for inst in &f.ir.func().blocks[&BlockId(pc)].insts[condition.core.clone()] {
            let Inst::Core { value, ty, op } = *inst else { unreachable!() };
            self.values[value.0] = self.scalar_emitter.op(ty, op, &self.values);
        }
    }
}
