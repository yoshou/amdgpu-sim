//! Shared scalar/packet LLVM lowering of the core SSA operations.
//! No ISA opcodes, register files or per-lane runtime calls.
use super::ir::typed::*;
use llvm::core::*;
use llvm::prelude::*;
use llvm_sys as llvm;

pub(super) struct Emitter {
    pub(super) bvh: Option<super::dialect::rdna4::bvh::Storage>,
    registry: std::sync::Arc<super::dialect::DialectRegistry>,
    pub b: LLVMBuilderRef,
    module: LLVMModuleRef,
    pub(super) ctx: LLVMContextRef,
    width: Option<u32>,
    lane_id: Option<LLVMValueRef>,
    valid_lane: Option<LLVMValueRef>,
    scratch: Option<(LLVMValueRef, LLVMValueRef)>,
}
impl Emitter {
    unsafe fn set_lane_id(&mut self, lane_base: LLVMValueRef) {
        let n = b"\0".as_ptr().cast();
        let i32t = LLVMInt32TypeInContext(self.ctx);
        let base = LLVMBuildTrunc(self.b,lane_base,i32t,n);
        self.lane_id = Some(if let Some(width) = self.width {
            let mut lanes: Vec<_> = (0..width).map(|lane| LLVMConstInt(i32t,lane as u64,0)).collect();
            let initial = LLVMBuildInsertElement(self.b,LLVMGetPoison(self.ty(Ty::I32)),base,LLVMConstInt(i32t,0,0),n);
            let broadcast = LLVMBuildShuffleVector(self.b,initial,LLVMGetPoison(self.ty(Ty::I32)),LLVMConstNull(self.ty(Ty::I32)),n);
            LLVMBuildAdd(self.b,broadcast,LLVMConstVector(lanes.as_mut_ptr(),width),n)
        } else { base });
    }
    pub unsafe fn new(b: LLVMBuilderRef, width: Option<u32>, registry: std::sync::Arc<super::dialect::DialectRegistry>) -> Self {
        let module = LLVMGetGlobalParent(LLVMGetBasicBlockParent(LLVMGetInsertBlock(b)));
        Self {
            registry,
            bvh: None,
            b,
            module,
            ctx: LLVMGetModuleContext(module),
            width,
            lane_id: None,
            valid_lane: None,
            scratch: None,
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
            Op::Env(Env::PacketLaneId) => {
                if let Some(width)=self.width {
                    let i32t=LLVMInt32TypeInContext(self.ctx);
                    let mut lanes:Vec<_>=(0..width).map(|lane|LLVMConstInt(i32t,lane as u64,0)).collect();
                    LLVMConstVector(lanes.as_mut_ptr(),width)
                } else {self.constant(Ty::I32,0)}
            },
            Op::Env(env @ (Env::ScratchBase | Env::ScratchSize)) => {
                let (base, size) = self.scratch.expect("scratch requires the invocation environment");
                let value = if env == Env::ScratchBase { base } else { size };
                if let Some(width) = self.width {
                    let i32t = LLVMInt32TypeInContext(self.ctx);
                    let vector = LLVMBuildInsertElement(b, LLVMGetPoison(self.ty(Ty::I64)), value, LLVMConstInt(i32t, 0, 0), n);
                    LLVMBuildShuffleVector(b, vector, LLVMGetPoison(self.ty(Ty::I64)), LLVMConstNull(LLVMVectorType(i32t, width)), n)
                } else { value }
            },
            Op::Env(Env::ValidLane) => self.valid_lane.expect("ValidLane requires the invocation environment"),
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
    mask_words: Vec<LLVMValueRef>,
    narrow_words: Vec<Option<LLVMValueRef>>,
    wide_words: Vec<Option<(LLVMValueRef,LLVMValueRef)>>,
    cooperative: bool,
    lane_base: LLVMValueRef,
    valid_mask: LLVMValueRef,
    live: Vec<bool>,
    retained: Vec<bool>,
    types: Vec<Ty>,
    definitions: Vec<Option<Op>>,
}
impl Values {
    /// Lower the pure SSA updates after a native effect produces its value.
    /// The effect result is the only additional binding; conversions belong
    /// to the verified IR, including lane projection and lane validity.
    pub unsafe fn effect_result(&self, raw: ValueId, result: ValueId, value: LLVMValueRef) -> (Ty,LLVMValueRef) {
        if raw==result {return (self.types[result.0],value);}
        unsafe fn build(s:&Values,id:ValueId,cache:&mut [LLVMValueRef],scalar:bool)->LLVMValueRef {
            if !cache[id.0].is_null() {return cache[id.0];}
            let op=s.definitions[id.0].expect("unbound effect result dependency");
            let mut args=vec![];
            op.map(|arg|{if !args.contains(&arg) {args.push(arg);}arg});
            for arg in args {let v=build(s,arg,cache,scalar);cache[arg.0]=s.shape(v,s.types[arg.0],scalar);}
            let v=if matches!(op,Op::Env(Env::ValidLane)) {
                let e=&s.emitter;let n=b"\0".as_ptr().cast();
                let bits=LLVMBuildTrunc(e.b,s.valid_mask,LLVMIntTypeInContext(e.ctx,e.width.unwrap_or(1)),n);
                LLVMBuildBitCast(e.b,bits,e.ty(Ty::I1),n)
            } else {s.emitter.op(s.types[id.0],op,cache)};
            cache[id.0]=v;v
        }
        let mut cache=self.values.clone();
        cache[raw.0]=value;
        let scalar=self.emitter.width.is_none();
        // Lane environment values use the same shape as the result packet.
        let value=build(self,result,&mut cache,scalar);
        (self.types[result.0],value)
    }
    unsafe fn shape(&self, value: LLVMValueRef, ty: Ty, scalar: bool) -> LLVMValueRef {
        assert!(!value.is_null(),"undefined SSA operand in native lowering");
        let target = if scalar { self.scalar_emitter.ty(ty) } else { self.emitter.ty(ty) };
        if LLVMTypeOf(value) == target { return value; }
        let n = b"\0".as_ptr().cast(); let b = self.emitter.b;
        let zero = LLVMConstInt(LLVMInt32TypeInContext(self.emitter.ctx),0,0);
        if scalar { LLVMBuildExtractElement(b,value,zero,n) }
        else {
            let initial = LLVMBuildInsertElement(b,LLVMGetPoison(target),value,zero,n);
            let mask = LLVMConstNull(LLVMVectorType(LLVMInt32TypeInContext(self.emitter.ctx),self.emitter.width.unwrap()));
            LLVMBuildShuffleVector(b,initial,LLVMGetPoison(target),mask,n)
        }
    }
    unsafe fn emit_inst(&mut self, inst: &cfg::Inst, scalar: bool) {
        use cfg::Inst;
        match inst {
            Inst::Core {value,..} if !self.live[value.0] => return,
            Inst::Target {provenance:None,outputs,..} if outputs.iter().all(|p|!self.live[p.0.0]) => return,
            Inst::Packet {output,..} if !self.live[output.0]=>return,
            _=>{}
        }
        match inst {
            Inst::Core { value,ty,op } => {
                if !self.values[value.0].is_null() { return; }
                if let Op::Convert(Cvt::Bitcast,to,source)=op {
                    if *to==self.types[source.0] {self.mask_words[value.0]=self.mask_words[source.0];}
                }
                if matches!(op,Op::Env(Env::LaneId))&&self.emitter.lane_id.is_none() {self.emitter.set_lane_id(self.lane_base);}
                if matches!(op,Op::Env(Env::ValidLane)) {
                    self.mask_words[value.0]=self.valid_mask;
                    let e=&mut self.emitter;
                    let n=b"\0".as_ptr().cast();
                    let bits=LLVMBuildTrunc(e.b,self.valid_mask,LLVMIntTypeInContext(e.ctx,e.width.unwrap_or(1)),n);
                    e.valid_lane=Some(LLVMBuildBitCast(e.b,bits,e.ty(Ty::I1),n));
                }
                // The existing packed mask conversion is trunc-to-iW followed
                // by a bitcast to <W x i1>. Keep that representation for the
                // explicit SSA spelling `(word >> PacketLaneId) & 1`.
                if let Op::Convert(Cvt::Trunc,Ty::I1,shift)=*op {
                    if let Some(Op::Int(IntOp::LShr,word,lane))=self.definitions[shift.0] {
                        let raw=self.values[word.0];
                        if matches!(self.definitions[lane.0],Some(Op::Env(Env::PacketLaneId))) && !raw.is_null()
                            && LLVMTypeOf(raw)==self.scalar_emitter.ty(Ty::I32) {
                            if self.emitter.width.is_some()&&!self.retained[value.0] {
                                self.mask_words[value.0]=raw;
                                return;
                            }
                            let bits=LLVMBuildTrunc(self.emitter.b,raw,LLVMIntTypeInContext(self.emitter.ctx,self.emitter.width.unwrap_or(1)),b"\0".as_ptr().cast());
                            self.values[value.0]=LLVMBuildBitCast(self.emitter.b,bits,self.emitter.ty(Ty::I1),b"\0".as_ptr().cast());
                            if self.emitter.width.is_some() {self.mask_words[value.0]=raw;}
                            return;
                        }
                    }
                }
                let mut args = vec![];
                op.map(|id| { if !args.contains(&id) { args.push(id); } id });
                let scalar = scalar && !matches!(op,Op::Env(_)) && args.iter().all(|id|
                    !self.values[id.0].is_null() && LLVMGetTypeKind(LLVMTypeOf(self.values[id.0])) != llvm::LLVMTypeKind::LLVMVectorTypeKind);
                let previous: Vec<_> = args.iter().map(|id| (*id,self.values[id.0])).collect();
                for &(id,v) in &previous {
                    assert!(!v.is_null(),"undefined SSA value {} for {:?}",id.0,op);
                    let raw = LLVMTypeOf(v);
                    let elem = if LLVMGetTypeKind(raw)==llvm::LLVMTypeKind::LLVMVectorTypeKind { LLVMGetElementType(raw) } else {raw};
                    let t = match LLVMGetTypeKind(elem) {
                        llvm::LLVMTypeKind::LLVMFloatTypeKind => Ty::F32,
                        llvm::LLVMTypeKind::LLVMDoubleTypeKind => Ty::F64,
                        _ => match LLVMGetIntTypeWidth(elem) { 1=>Ty::I1,32=>Ty::I32,64=>Ty::I64,_=>unreachable!() },
                    };
                    self.values[id.0] = self.shape(v,t,scalar);
                }
                let emitter = if scalar { &self.scalar_emitter } else { &self.emitter };
                let result = emitter.op(*ty,*op,&self.values);
                for (id,v) in previous { self.values[id.0]=v; }
                self.values[value.0]=result;
            }
            Inst::Target { op,args,outputs,.. } => {
                let emitter = if scalar { &self.scalar_emitter } else { &self.emitter };
                let results = emitter.target(*op,*args,&self.values);
                for (&(id,_),v) in outputs.iter().zip(results) {self.values[id.0]=v;}
            }
            Inst::Packet { op,input,output } => {
                // Architectural mask queries use the packet ABI. Explicit
                // wave operations are emitted by the existing yield plan.
                let b = self.emitter.b; let ctx=self.emitter.ctx; let n=b"\0".as_ptr().cast();
                let i32t=LLVMInt32TypeInContext(ctx);
                let bits=if !self.mask_words[input.0].is_null() {
                    let word=LLVMBuildZExt(b,self.mask_words[input.0],i32t,n);
                    if *op==cfg::PacketOp::Any {
                        LLVMBuildAnd(b,word,LLVMConstInt(i32t,(1u64<<self.emitter.width.unwrap_or(1))-1,0),n)
                    } else {word}
                } else {
                    let input=self.shape(self.values[input.0],Ty::I1,false);
                    let bits=if let Some(width)=self.emitter.width {
                        LLVMBuildBitCast(b,input,LLVMIntTypeInContext(ctx,width),n)
                    } else {input};
                    LLVMBuildZExt(b,bits,i32t,n)
                };
                let raw=if *op==cfg::PacketOp::Any {
                    LLVMBuildICmp(b,llvm::LLVMIntPredicate::LLVMIntNE,bits,LLVMConstNull(i32t),n)
                } else {bits};
                self.values[output.0]=raw;
            }
            Inst::Effect {..}=>unreachable!("effects require their native action plan"),
        }
    }
    pub unsafe fn set_scratch_environment(&mut self, base: LLVMValueRef, size: LLVMValueRef) {
        self.emitter.scratch = Some((base, size));
        self.scalar_emitter.scratch = Some((base, size));
    }
    pub unsafe fn set_bvh_storage(&mut self,storage:super::dialect::rdna4::bvh::Storage) {
        self.emitter.bvh=Some(storage);self.scalar_emitter.bvh=Some(storage);
    }
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
            self.values[id.0] = read(parameter, scalar);
        }
        let emitter = Emitter::new(self.emitter.b, if scalar { None } else { self.emitter.width }, self.emitter.registry.clone());
        for &(pair, view) in &plan.pairs {
            self.values[pair.0] = LLVMBuildBitCast(self.emitter.b, self.values[view.0], emitter.ty(Ty::I64), b"\0".as_ptr().cast());
        }
        for inst in &f.ir.func().blocks[&BlockId(pc)].insts[plan.core.clone()] {
            self.emit_inst(inst,scalar);
        }
        if let Some((range,_,_,_))=&plan.flat {
            for inst in &f.ir.func().blocks[&BlockId(pc)].insts[range.clone()] {self.emit_inst(inst,scalar);}
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
            let architectural=plan.definitions[index].1;
            if matches!(destination,super::lift::wave::Destination::Sgpr(_)) {
                let (ty,value)=self.effect_result(id,architectural,result);
                self.values[architectural.0]=value;
                write(destination,ty,value,value);
            } else {write(destination, ty, result, bits);}
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
            self.emit_inst(inst,false);
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
        let function = LLVMGetBasicBlockParent(LLVMGetInsertBlock(b));
        let cooperative = LLVMCountParams(function) == 9;
        let emitter = Emitter::new(b,width,f.registry.clone());
        let lane_base = if cooperative { LLVMGetParam(function,6) } else { LLVMConstInt(LLVMInt64TypeInContext(emitter.ctx),0,0) };
        // The scalar scheduler starts fibers only for an allocated lane.
        // Padding is possible inside a vector packet, as in the existing ABI.
        let valid_mask = if cooperative && width.is_some() { LLVMGetParam(function,8) } else { LLVMConstInt(LLVMInt32TypeInContext(emitter.ctx),u32::MAX as u64,0) };
        let mut definitions=vec![None;f.ir.func().types.len()];
        for block in f.ir.func().blocks.values() {
            for inst in &block.insts {if let cfg::Inst::Core {value,op,..}=inst {definitions[value.0]=Some(*op);}}
        }

        Self {
            emitter,
            lane_base,
            valid_mask,
            live: f.native_live.clone(),
            retained: f.retained.clone(),
            types: f.ir.func().types.clone(),
            definitions,
            cooperative,
            scalar_emitter: Emitter::new(b, None, f.registry.clone()),
            values: vec![std::ptr::null_mut(); f.ir.func().types.len()],
            mask_words: vec![std::ptr::null_mut(); f.ir.func().types.len()],
            narrow_words:vec![None;f.ir.func().types.len()],
            wide_words: vec![None;f.ir.func().types.len()],
        }
    }
    pub fn begin_block(&mut self, f: &super::lift::function::Function, pc: usize) {
        unsafe {
            self.emitter.lane_id=None;
            self.emitter.valid_lane=None;
        }
        use super::ir::typed::cfg::*;
        let block = &f.ir.func().blocks[&BlockId(pc)];
        for &(v, _) in &block.params {
            self.values[v.0] = std::ptr::null_mut();
            self.mask_words[v.0] = std::ptr::null_mut();
            self.narrow_words[v.0]=None;
            self.wide_words[v.0]=None;
        }
        for inst in &block.insts {
            match inst {
                Inst::Core { value, .. }|Inst::Packet {output:value,..} => {self.values[value.0] = std::ptr::null_mut();self.mask_words[value.0]=std::ptr::null_mut();self.narrow_words[value.0]=None;self.wide_words[value.0]=None;},
                Inst::Effect { outputs, .. } | Inst::Target { outputs, .. } => {
                    for &(v, _) in outputs {
                        self.values[v.0] = std::ptr::null_mut();
                        self.mask_words[v.0] = std::ptr::null_mut();
                        self.narrow_words[v.0]=None;
                        self.wide_words[v.0]=None;
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
        unpredicated: bool,
        packed_mask: impl Fn(u32)->bool,
        read_mask_word: impl Fn(u32)->LLVMValueRef,
        mut read: impl FnMut(&super::lift::Input, bool) -> LLVMValueRef,
        mut write: impl FnMut(super::lift::Output, LLVMValueRef, Option<LLVMValueRef>),
    ) {
        use super::ir::typed::cfg::*;
        let alu = f.blocks[&pc].instructions[index]
            .as_ref()
            .expect("missing typed lowering");
        let structured_logic=self.emitter.width.is_some()&&alu.mask_logic&&alu.outputs.iter().any(|&(out,_)|match out {
            super::lift::Output::MaskBit(reg)|super::lift::Output::Scalar(reg,Ty::I32)=>!packed_mask(reg),
            _=>false,
        });
        if structured_logic&&alu.outputs.iter().all(|&(out,_)|match out {
            super::lift::Output::MaskBit(reg)|super::lift::Output::Scalar(reg,Ty::I32)=>!packed_mask(reg),
            super::lift::Output::Scc=>true,
            _=>false,
        }) {
            self.emit_structured_logic(f,pc,alu,&mut read,&mut write);
            return;
        }
        let mut word_views=Vec::new();
        for (input,id) in &alu.inputs[..alu.source_inputs] {
            let reg=match input.source {
                super::lift::InputSource::MaskBit(reg)=>Some(reg),
                _=>None,
            };
            if let Some(reg)=reg {
                if self.mask_words[id.0].is_null() {self.mask_words[id.0]=read_mask_word(reg);}
            }
        }
        for (input, id) in &alu.inputs[..alu.source_inputs] {
            if matches!(input.source,super::lift::InputSource::MaskBit(_))&&!self.mask_words[id.0].is_null()&&!self.retained[id.0] {continue;}
            let emitter = if alu.scalar { &self.scalar_emitter } else { &self.emitter };
            // One ordinary SGPR definition can have both uniform scalar and
            // broadcast packet uses. Rebind its native representation when
            // the consumer changes shape; its architectural SSA ID is shared.
            if self.values[id.0].is_null() || LLVMTypeOf(self.values[id.0]) != emitter.ty(input.ty) {
                self.values[id.0] = read(input, alu.scalar && !matches!(input.source,super::lift::InputSource::MaskBit(_)));
            }
            if structured_logic&&input.ty==Ty::I32&&!word_views.iter().any(|&(saved,_)|saved==*id) {
                let raw=self.values[id.0];word_views.push((*id,raw));
                self.values[id.0]=LLVMBuildAnd(self.emitter.b,raw,self.scalar_emitter.constant(Ty::I32,(1u64<<self.emitter.width.unwrap())-1),b"\0".as_ptr().cast());
            }
        }
        let block = &f.ir.func().blocks[&BlockId(pc)];
        let mut next_output=0;
        let mut deferred_scc=None;
        // Architectural words are coalesced into the existing native register
        // cells by `write`. Materialize only the expressions feeding those
        // writes here; a later word use binds the corresponding native cell.
        let mut needed: Vec<_>=alu.outputs.iter()
            .filter_map(|&(_,id)|self.live[id.0].then_some(id)).collect();
        for inst in &block.insts[alu.core.clone()] {
            match inst {
                Inst::Core {value,..}|Inst::Packet {output:value,..}=>{if self.retained[value.0] {needed.push(*value);}},
                Inst::Effect {outputs,..}|Inst::Target {outputs,..}=>{
                    needed.extend(outputs.iter().filter_map(|&(id,_)|self.retained[id.0].then_some(id)));
                },
            }
        }
        let mut native=vec![false;alu.core.len()];
        let mut source_definitions:Vec<_>=block.insts[alu.core.start..alu.previous_core.start].iter().filter_map(|inst|match inst {
            Inst::Core {value,..}=>Some(*value),_=>None,
        }).collect();
        source_definitions.retain(|id|!alu.inputs[..alu.source_inputs].iter().any(|(_,input)|input==id));
        let bound:Vec<_>=alu.inputs.iter().map(|(_,id)|*id).filter(|id|!source_definitions.contains(id))
            .chain(alu.pairs.iter().map(|&(id,_)|id)).collect();
        for (offset,inst) in block.insts[alu.core.clone()].iter().enumerate().rev() {
            match inst {
                Inst::Core {value,..} if bound.contains(value)=>{},
                Inst::Core {value,op:Op::Convert(Cvt::Trunc,Ty::I1,shift),..} if needed.contains(value)
                    &&matches!(self.definitions[shift.0],Some(Op::Int(IntOp::LShr,_,lane)) if matches!(self.definitions[lane.0],Some(Op::Env(Env::PacketLaneId))))=>{
                    native[offset]=true;
                    let Some(Op::Int(_,word,_))=self.definitions[shift.0] else {unreachable!()};
                    needed.push(word);
                },
                Inst::Core {value,op,..} if needed.contains(value)=>{
                    native[offset]=true;op.map(|id|{needed.push(id);id});
                },
                Inst::Target {outputs,args,provenance,..} if provenance.is_some()||outputs.iter().any(|(id,_)|needed.contains(id))=>{
                    native[offset]=true;needed.extend(args.values());
                },
                Inst::Effect {outputs,inputs,..} if outputs.iter().any(|(id,_)|needed.contains(id))=>{
                    native[offset]=true;needed.extend(inputs);
                },
                Inst::Packet {input,output,..} if needed.contains(output)=>{
                    native[offset]=true;needed.push(*input);
                },
                _=>{},
            }
        }
        for &(pair, view) in &alu.pairs {
            if self.values[view.0].is_null() {continue;}
            let emitter = if alu.scalar { &self.scalar_emitter } else { &self.emitter };
            self.values[pair.0] = LLVMBuildBitCast(self.emitter.b, self.values[view.0], emitter.ty(Ty::I64), b"\0".as_ptr().cast());
        }
        for (offset,original) in block.insts[alu.core.clone()].iter().enumerate() {
            if !native[offset] {continue;}
            if alu.mask_logic {
                if let Inst::Core {value,..}=original {
                    if alu.outputs.iter().any(|&(out,id)|matches!(out,super::lift::Output::Scc)&&id==*value) {
                        deferred_scc=Some(original);
                        continue;
                    }
                }
            }
            if alu.previous_core.contains(&(alu.core.start+offset)) {continue;}
            if let Inst::Packet {op:cfg::PacketOp::Ballot,input,output}=original {
                if let Some(&(super::lift::Output::Scalar(reg,Ty::I32),id))=alu.outputs.get(next_output) {
                    if id==*output&&!packed_mask(reg)&&alu.mask_updates.iter().any(|&(_,r)|r==reg) {
                        // Compare/carry outputs use the region's boolean cell,
                        // including masks held in otherwise ordinary SGPRs.
                        write(super::lift::Output::MaskBit(reg),self.values[input.0],None);
                        next_output+=1;
                        continue;
                    }
                }
            }
            if let Inst::Core {value,..}=original {
                if !self.live[value.0] || !self.values[value.0].is_null() {continue;}
                if let Some((input,_))=alu.inputs.iter().find(|(_,id)|id==value&&!source_definitions.contains(id)) {
                    if alu.outputs.iter().any(|&(_,id)|matches!(self.definitions[id.0],Some(Op::Select(p,_,_)) if p==*value)) {continue;}
                    if input.ty==Ty::I1&&!self.mask_words[value.0].is_null()&&!self.retained[value.0] {continue;}
                    self.values[value.0]=read(input,alu.scalar&&!matches!(input.ty,Ty::I1));
                    continue;
                }
            }
            let mut inst=original.clone();
            if structured_logic {
                if let Inst::Core {op:Op::Const(Ty::I32,bits),..}=&mut inst {*bits&=(1u64<<self.emitter.width.unwrap())-1;}
            }
            // The existing structured-mask region keeps EXEC in its boolean
            // cell. Its saved/restored masks are already bounded at entry;
            // packed SGPR stores perform the separate validity intersection.
            if let Inst::Core {value,ty:Ty::I1,op:Op::Int(IntOp::And,bit,valid)}=&inst {
                if alu.mask_updates.contains(&(*value,126))&&(self.emitter.width.is_none()||!packed_mask(126)||!self.cooperative)
                    &&matches!(self.definitions[valid.0],Some(Op::Env(Env::ValidLane))) {
                    inst=Inst::Core {value:*value,ty:Ty::I1,op:Op::Convert(Cvt::Bitcast,Ty::I1,*bit)};
                }
            }
            if unpredicated {
                if let Inst::Core {value,ty,op}=&mut inst {
                    if alu.packet_elidable.contains(value) {
                        let raw=alu.predicated.iter().find(|(id,_)|id==value).unwrap().1;
                        *op=match *op {
                            Op::Select(..)|Op::Int(IntOp::And,_,_)=>Op::Convert(Cvt::Bitcast,*ty,raw),
                            other=>other,
                        };
                    }
                }
            }
            let destination=if let Inst::Core {value,op:Op::Select(..),..}=&inst {
                alu.outputs.iter().find_map(|&(out,id)|if id==*value {if let super::lift::Output::Vgpr(reg,_)=out {Some(reg)} else {None}} else {None})
            } else {None};
            if destination.is_none() {if let Inst::Core {op:Op::Select(pred,yes,no),..}=&mut inst {
                if self.values[pred.0].is_null() {
                    if let Some((input,_))=alu.inputs.iter().find(|(_,v)|v==pred) {
                        self.values[pred.0]=read(input,alu.scalar&&!matches!(input.source,super::lift::InputSource::MaskBit(_)));
                    }
                }
                let p=self.values[pred.0];
                if !p.is_null() && LLVMIsConstant(p)!=0 {
                    let width=if LLVMGetTypeKind(LLVMTypeOf(p))==llvm::LLVMTypeKind::LLVMVectorTypeKind {LLVMGetVectorSize(LLVMTypeOf(p))} else {1};
                    let bits=LLVMConstBitCast(p,LLVMIntTypeInContext(self.emitter.ctx,width));
                    if !LLVMIsAConstantInt(bits).is_null() {
                        let value=LLVMConstIntGetZExtValue(bits);
                        if value==0 {*yes=*no;}
                        else if value==(1u64<<width)-1 {*no=*yes;}
                    }
                }
            }}
            if let Inst::Core {value,ty:Ty::I1,op:Op::Int(_,a,b)}=&inst {
                if alu.mask_updates.iter().any(|&(id,reg)|id==*value&&packed_mask(reg)) {
                    for id in [*a,*b] {
                        if !self.mask_words[id.0].is_null() {continue;}
                        if let Some((input,_))=alu.inputs.iter().find(|(_,bound)|*bound==id) {
                            let reg=match input.source {
                                super::lift::InputSource::MaskBit(reg)=>Some(reg),
                                super::lift::InputSource::ExecPredicate=>Some(126),
                                super::lift::InputSource::Operand(crate::rdna_instructions::SourceOperand::ScalarRegister(reg)) if input.ty==Ty::I1=>Some(reg as u32),
                                _=>None,
                            };
                            if let Some(reg)=reg {self.mask_words[id.0]=read_mask_word(reg);}
                        }
                    }
                }
            }
            let args=match &inst {
                Inst::Core {op:Op::Select(_,raw,_),..} if destination.is_some()=>vec![*raw],
                Inst::Core {op:Op::Convert(Cvt::Trunc,Ty::I1,shift),..}
                    if matches!(self.definitions[shift.0],Some(Op::Int(IntOp::LShr,_,lane)) if matches!(self.definitions[lane.0],Some(Op::Env(Env::PacketLaneId))))=>{
                    let Some(Op::Int(_,word,_))=self.definitions[shift.0] else {unreachable!()};vec![word]
                },
                Inst::Core {ty:Ty::I1,op:Op::Convert(Cvt::Bitcast,Ty::I1,raw),..} if !self.mask_words[raw.0].is_null()=>vec![],
                Inst::Core {value,ty:Ty::I1,op:Op::Int(_,a,b)} if alu.mask_updates.iter().any(|&(id,reg)|id==*value&&packed_mask(reg))=>
                    [*a,*b].iter().copied().filter(|id|self.mask_words[id.0].is_null()).collect(),
                Inst::Core {op,..}=>{let mut a=vec![];op.map(|v|{a.push(v);v});a},
                Inst::Packet {input,..} if !self.mask_words[input.0].is_null()=>vec![],
                Inst::Packet {input,..}=>vec![*input],
                Inst::Effect {inputs,..}=>inputs.clone(),
                _=>vec![],
            };
            for id in args {
                if self.values[id.0].is_null() {
                    if let Some((input,_))=alu.inputs.iter().find(|(_,v)|*v==id) {
                        self.values[id.0]=read(input,alu.scalar&&!matches!(input.source,super::lift::InputSource::MaskBit(_)));
                    } else if self.types[id.0]==Ty::I1&&!self.mask_words[id.0].is_null() {
                        let e=&self.emitter;let n=b"\0".as_ptr().cast();
                        let bits=LLVMBuildTrunc(e.b,self.mask_words[id.0],LLVMIntTypeInContext(e.ctx,e.width.unwrap_or(1)),n);
                        self.values[id.0]=LLVMBuildBitCast(e.b,bits,e.ty(Ty::I1),n);
                    } else if let Some(bits)=self.narrow_words[id.0] {
                        self.values[id.0]=LLVMBuildBitCast(self.emitter.b,bits,self.emitter.ty(self.types[id.0]),b"\0".as_ptr().cast());
                    } else if let Some((lo,hi))=self.wide_words[id.0] {
                        let e=&self.emitter;let n=b"\0".as_ptr().cast();
                        let lo=LLVMBuildZExt(e.b,lo,e.ty(Ty::I64),n);
                        let hi=LLVMBuildZExt(e.b,hi,e.ty(Ty::I64),n);
                        let hi=LLVMBuildShl(e.b,hi,e.constant(Ty::I64,32),n);
                        let bits=LLVMBuildOr(e.b,hi,lo,n);
                        self.values[id.0]=LLVMBuildBitCast(e.b,bits,e.ty(self.types[id.0]),n);
                    }
                }
            }
            if let Inst::Core {value,ty:Ty::I1,op:Op::Int(op @ (IntOp::And|IntOp::Or|IntOp::Xor),a,b)}=inst {
                if alu.mask_updates.iter().any(|&(id,reg)|id==value&&packed_mask(reg)) {
                    let e=&self.emitter;let n=b"\0".as_ptr().cast();
                    let iw=LLVMIntTypeInContext(e.ctx,e.width.unwrap_or(1));let i32t=LLVMInt32TypeInContext(e.ctx);
                    let pack=|v|LLVMBuildZExt(e.b,LLVMBuildBitCast(e.b,v,iw,n),i32t,n);
                    let a=if self.mask_words[a.0].is_null() {pack(self.shape(self.values[a.0],Ty::I1,false))} else {LLVMBuildZExt(e.b,self.mask_words[a.0],i32t,n)};
                    let b=if self.mask_words[b.0].is_null() {pack(self.shape(self.values[b.0],Ty::I1,false))} else {LLVMBuildZExt(e.b,self.mask_words[b.0],i32t,n)};
                    let b=if e.width.is_none() {LLVMBuildAnd(e.b,b,LLVMConstInt(i32t,1,0),n)} else {b};
                    let bits=match op {IntOp::And=>LLVMBuildAnd(e.b,a,b,n),IntOp::Or=>LLVMBuildOr(e.b,a,b,n),IntOp::Xor=>LLVMBuildXor(e.b,a,b,n),_=>unreachable!()};
                    // Ordinary mask destinations store the packed SGPR word.
                    // Their following ballot consumes that word directly; a
                    // boolean projection belongs at a later mask operand read.
                    let word_only=e.width.is_some()&&!self.retained[value.0];
                    if !word_only {self.values[value.0]=LLVMBuildBitCast(e.b,LLVMBuildTrunc(e.b,bits,iw,n),e.ty(Ty::I1),n);}
                    self.mask_words[value.0]=bits;
                } else {self.emit_inst(&inst,alu.scalar);}
            } else if let Inst::Core {value,ty:Ty::I1,op:Op::Convert(Cvt::Bitcast,Ty::I1,raw)}=&inst {
                self.mask_words[value.0]=self.mask_words[raw.0];
                self.values[value.0]=self.values[raw.0];
            } else if let (Inst::Core {value,ty,op:Op::Select(p,a,_)},Some(reg))=(&inst,destination) {
                    let (value,ty,p,a)=(*value,*ty,*p,*a);
                    let raw=self.shape(self.values[a.0],ty,false);
                    let e=&self.emitter;let n=b"\0".as_ptr().cast();
                    let predicate=&alu.inputs.iter().find(|(_,id)|*id==p).expect("architectural predicate binding").0;
                    let input=|reg,ty|super::lift::Input {source:super::lift::InputSource::Operand(crate::rdna_instructions::SourceOperand::VectorRegister(reg as u8)),ty};
                    self.values[value.0]=if ty==Ty::F32 {
                        let a=LLVMBuildBitCast(e.b,raw,e.ty(Ty::I32),n);
                        let old=read(&input(reg,Ty::I32),false);
                        let p=read(predicate,false);
                        let bits=LLVMBuildSelect(e.b,p,a,old,n);
                        self.narrow_words[value.0]=Some(bits);
                        std::ptr::null_mut()
                    } else if ty==Ty::I64||(ty==Ty::F64&&e.width.is_none()) {
                        let a=LLVMBuildBitCast(e.b,raw,e.ty(Ty::I64),n);
                        let lo_a=LLVMBuildTrunc(e.b,a,e.ty(Ty::I32),n);
                        let hi_a=LLVMBuildLShr(e.b,a,e.constant(Ty::I64,32),n);
                        let hi_a=LLVMBuildTrunc(e.b,hi_a,e.ty(Ty::I32),n);
                        assert_eq!(alu.outputs[next_output].1,value);
                        let lo_b=read(&input(reg,Ty::I32),false);
                        let p=read(predicate,false);
                        let lo=LLVMBuildSelect(e.b,p,lo_a,lo_b,n);
                        write(super::lift::Output::Vgpr(reg,Ty::I32),lo,None);
                        let hi_b=read(&input(reg+1,Ty::I32),false);
                        let p=read(predicate,false);
                        let hi=LLVMBuildSelect(e.b,p,hi_a,hi_b,n);
                        write(super::lift::Output::Vgpr(reg+1,Ty::I32),hi,None);
                        next_output+=1;
                        self.wide_words[value.0]=Some((lo,hi));
                        std::ptr::null_mut()
                    } else {
                        let old=read(&input(reg,ty),false);
                        let p=read(predicate,false);
                        LLVMBuildSelect(e.b,p,raw,old,n)
                    };
            } else {self.emit_inst(&inst,alu.scalar);}
            if alu.core.start+offset+1>=alu.updates_start {
                self.write_ready(alu,&mut next_output,&mut write);
            }
        }
        // Keep the effective value in its coalesced boundary slot until its
        // first use, matching the existing emitter's load placement. Loading
        // immediately after every write perturbs LLVM's optimization order
        // even when that SSA definition is never read. Later uses still share
        // the same SSA ID; the lifter invalidates views on overlapping writes.
        if let Some(inst)=deferred_scc {self.emit_inst(inst,true);}
        self.write_ready(alu,&mut next_output,&mut write);
        assert_eq!(next_output,alu.outputs.len(),"unmaterialized architectural SSA output");
        for (id,raw) in word_views {self.values[id.0]=raw;}
        // Match the existing word/pair coalescing boundary. A subsequent word
        // consumer reads the representation written above, rather than keeping
        // the producer's wider expression alive across overlapping pair writes.
        for &id in &alu.vector_words {if !self.retained[id.0] {self.values[id.0]=std::ptr::null_mut();}}
        for &(output,id) in &alu.outputs {
            if matches!(output,super::lift::Output::Vgpr(..))&&!self.retained[id.0] {self.values[id.0]=std::ptr::null_mut();}
        }
    }
    /// The existing structured region represents mask words as boolean
    /// packets. Lower their explicit SSA bitwise operations in that same
    /// representation, including the saved EXEC write before its replacement.
    unsafe fn emit_structured_logic(
        &mut self,f:&super::lift::function::Function,pc:usize,alu:&super::lift::function::Alu,
        read:&mut impl FnMut(&super::lift::Input,bool)->LLVMValueRef,
        write:&mut impl FnMut(super::lift::Output,LLVMValueRef,Option<LLVMValueRef>),
    ) {
        unsafe fn eval(
            this:&Values,f:&super::lift::function::Function,pc:usize,alu:&super::lift::function::Alu,id:ValueId,
            cache:&mut std::collections::BTreeMap<ValueId,LLVMValueRef>,
            read:&mut impl FnMut(&super::lift::Input,bool)->LLVMValueRef,
        )->LLVMValueRef {
            if let Some(&value)=cache.get(&id) {return value;}
            let e=&this.emitter;let n=b"\0".as_ptr().cast();
            let value=if let Some((input,_))=alu.inputs.iter().find(|(_,value)|*value==id) {
                let mut input=input.clone();input.ty=Ty::I1;
                if let super::lift::InputSource::Operand(crate::rdna_instructions::SourceOperand::ScalarRegister(reg))=input.source {
                    input.source=super::lift::InputSource::MaskBit(reg as u32);
                }
                read(&input,false)
            } else if let Some(op)=this.definitions[id.0] {
                match op {
                    Op::Const(Ty::I32,bits)=>{
                        let packed=LLVMConstInt(LLVMIntTypeInContext(e.ctx,e.width.unwrap()),bits,0);
                        LLVMConstBitCast(packed,e.ty(Ty::I1))
                    },
                    Op::Convert(Cvt::Bitcast,Ty::I1,raw)=>eval(this,f,pc,alu,raw,cache,read),
                    Op::Convert(Cvt::Trunc,Ty::I1,shift)=>{
                        let Some(Op::Int(IntOp::LShr,word,lane))=this.definitions[shift.0] else {panic!("structured mask projection");};
                        assert!(matches!(this.definitions[lane.0],Some(Op::Env(Env::PacketLaneId))));
                        eval(this,f,pc,alu,word,cache,read)
                    },
                    Op::Int(op @ (IntOp::And|IntOp::Or|IntOp::Xor),a,b)=>{
                        let a=eval(this,f,pc,alu,a,cache,read);
                        if matches!(this.definitions[b.0],Some(Op::Env(Env::ValidLane))) {a}
                        else {
                            let b=eval(this,f,pc,alu,b,cache,read);
                            match op {IntOp::And=>LLVMBuildAnd(e.b,a,b,n),IntOp::Or=>LLVMBuildOr(e.b,a,b,n),IntOp::Xor=>LLVMBuildXor(e.b,a,b,n),_=>unreachable!()}
                        }
                    },
                    Op::Cmp(IntPred::Ne,a,b)=>{
                        assert!(matches!(this.definitions[b.0],Some(Op::Const(Ty::I32,0))));
                        let mask=eval(this,f,pc,alu,a,cache,read);
                        let packed=LLVMBuildBitCast(e.b,mask,LLVMIntTypeInContext(e.ctx,e.width.unwrap()),n);
                        let packed=LLVMBuildZExt(e.b,packed,this.scalar_emitter.ty(Ty::I32),n);
                        LLVMBuildICmp(e.b,llvm::LLVMIntPredicate::LLVMIntNE,packed,this.scalar_emitter.constant(Ty::I32,0),n)
                    },
                    _=>panic!("unsupported structured mask expression {:?}",op),
                }
            } else {
                let input=f.ir.func().blocks[&cfg::BlockId(pc)].insts.iter().find_map(|inst|match inst {
                    cfg::Inst::Packet {op:cfg::PacketOp::Ballot,input,output} if *output==id=>Some(*input),
                    _=>None,
                }).expect("structured mask word must have an SSA definition");
                eval(this,f,pc,alu,input,cache,read)
            };
            cache.insert(id,value);value
        }
        let mut cache=std::collections::BTreeMap::new();
        for &(output,id) in &alu.outputs {
            let value=eval(self,f,pc,alu,id,&mut cache,read);
            match output {
                super::lift::Output::Scalar(reg,Ty::I32)|super::lift::Output::MaskBit(reg)=>write(super::lift::Output::MaskBit(reg),value,None),
                super::lift::Output::Scc=>write(output,value,None),
                _=>unreachable!(),
            }
        }
        for (id,value) in cache {
            if self.types[id.0]==Ty::I1 {self.values[id.0]=value;}
            else {self.values[id.0]=std::ptr::null_mut();}
            self.mask_words[id.0]=std::ptr::null_mut();
        }
    }
    unsafe fn write_ready(&self,alu:&super::lift::function::Alu,next:&mut usize,write:&mut impl FnMut(super::lift::Output,LLVMValueRef,Option<LLVMValueRef>)) {
        while let Some(&(output,id))=alu.outputs.get(*next) {
            if let (super::lift::Output::Vgpr(reg,Ty::F32),Some(bits))=(output,self.narrow_words[id.0]) {
                write(super::lift::Output::Vgpr(reg,Ty::I32),bits,None);
                *next+=1;
                continue;
            }
            if let (super::lift::Output::Vgpr(reg,Ty::I64|Ty::F64),Some((lo,hi)))=(output,self.wide_words[id.0]) {
                write(super::lift::Output::Vgpr(reg,Ty::I32),lo,None);
                write(super::lift::Output::Vgpr(reg+1,Ty::I32),hi,None);
                *next+=1;
                continue;
            }
            if self.values[id.0].is_null() {
                if matches!(output,super::lift::Output::MaskBit(_))&&!self.mask_words[id.0].is_null() {
                    write(output,std::ptr::null_mut(),Some(self.mask_words[id.0]));
                    *next+=1;
                    continue;
                }
                break;
            }
            let scalar=matches!(output,super::lift::Output::Scalar(..)|super::lift::Output::Scc);
            let word=(matches!(output,super::lift::Output::MaskBit(_))&&!self.mask_words[id.0].is_null()).then_some(self.mask_words[id.0]);
            write(output,self.shape(self.values[id.0],output.ty(),scalar),word);
            *next+=1;
        }
    }
    pub unsafe fn condition(
        &mut self,
        f: &super::lift::function::Function,
        pc: usize,
        read_mask_word: impl Fn(u32)->LLVMValueRef,
        mut read: impl FnMut(&super::lift::Input) -> LLVMValueRef,
    ) {
        use super::ir::typed::cfg::*;
        let Some(condition) = &f.blocks[&pc].condition else { return; };
        if let super::lift::InputSource::MaskBit(reg)=condition.input.source {
            self.mask_words[condition.input_value.0]=read_mask_word(reg);
        }
        if self.mask_words[condition.input_value.0].is_null()&&(self.values[condition.input_value.0].is_null()
            || LLVMTypeOf(self.values[condition.input_value.0]) != if matches!(condition.input.source,super::lift::InputSource::MaskBit(_)) { self.emitter.ty(Ty::I1) } else { self.scalar_emitter.ty(Ty::I1) }
            ) {
            self.values[condition.input_value.0] = read(&condition.input);
        }
        for inst in &f.ir.func().blocks[&BlockId(pc)].insts[condition.core.clone()] {
            self.emit_inst(inst,true);
        }
    }
}
