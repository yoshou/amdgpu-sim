//! Shared scalar/packet LLVM lowering of the core SSA operations.
//! No ISA opcodes, register files or per-lane runtime calls.
use super::super::ir::*;
use llvm::core::*;
use llvm::prelude::*;
use llvm_sys as llvm;

pub(in crate::rdna_spmd) struct Emitter {
    pub(in crate::rdna_spmd) bvh: Option<super::super::dialect::rdna4::bvh::Storage>,
    registry: std::sync::Arc<super::super::dialect::DialectRegistry>,
    pub b: LLVMBuilderRef,
    module: LLVMModuleRef,
    pub(in crate::rdna_spmd) ctx: LLVMContextRef,
    width: Option<u32>,
    pub(in crate::rdna_spmd) valid_lane: Option<LLVMValueRef>,
    pub(in crate::rdna_spmd) scratch: Option<(LLVMValueRef, LLVMValueRef)>,
    pub(in crate::rdna_spmd) lane_id: Option<LLVMValueRef>,
}
impl Emitter {
    pub(in crate::rdna_spmd) unsafe fn set_lane_id(&mut self, lane_base: LLVMValueRef) {
        let n = b"\0".as_ptr().cast();
        let i32t = LLVMInt32TypeInContext(self.ctx);
        let base = LLVMBuildTrunc(self.b, lane_base, i32t, n);
        self.lane_id = Some(if let Some(width) = self.width {
            let mut lanes: Vec<_> = (0..width).map(|lane| LLVMConstInt(i32t, lane as u64, 0)).collect();
            let initial = LLVMBuildInsertElement(self.b, LLVMGetPoison(self.ty(Ty::I32)), base, LLVMConstInt(i32t, 0, 0), n);
            let broadcast = LLVMBuildShuffleVector(self.b, initial, LLVMGetPoison(self.ty(Ty::I32)), LLVMConstNull(self.ty(Ty::I32)), n);
            LLVMBuildAdd(self.b, broadcast, LLVMConstVector(lanes.as_mut_ptr(), width), n)
        } else { base });
    }
    pub unsafe fn new(b: LLVMBuilderRef, width: Option<u32>, registry: std::sync::Arc<super::super::dialect::DialectRegistry>) -> Self {
        let module = LLVMGetGlobalParent(LLVMGetBasicBlockParent(LLVMGetInsertBlock(b)));
        Self {
            registry,
            bvh: None,
            b,
            module,
            ctx: LLVMGetModuleContext(module),
            width,
            valid_lane: None,
            scratch: None,
            lane_id: None,
        }
    }
    pub(in crate::rdna_spmd) unsafe fn ty(&self, t: Ty) -> LLVMTypeRef {
        let t = match t {
            Ty::I1 => LLVMInt1TypeInContext(self.ctx),
            Ty::I32 => LLVMInt32TypeInContext(self.ctx),
            Ty::I64 => LLVMInt64TypeInContext(self.ctx),
            Ty::F32 => LLVMFloatTypeInContext(self.ctx),
            Ty::F64 => LLVMDoubleTypeInContext(self.ctx),
        };
        self.shaped(t)
    }
    pub(in crate::rdna_spmd) unsafe fn shaped(&self, scalar: LLVMTypeRef) -> LLVMTypeRef {
        self.width.map_or(scalar, |w| LLVMVectorType(scalar, w))
    }
    pub(in crate::rdna_spmd) unsafe fn constant(&self, ty: Ty, bits: u64) -> LLVMValueRef {
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
    pub(in crate::rdna_spmd) fn suffix(&self, t: Ty) -> String {
        let t = match t {
            Ty::I1 => "i1",
            Ty::I32 => "i32",
            Ty::I64 => "i64",
            Ty::F32 => "f32",
            Ty::F64 => "f64",
        };
        self.width.map_or_else(|| t.into(), |w| format!("v{w}{t}"))
    }
    pub(in crate::rdna_spmd) fn width(&self) -> Option<u32> { self.width }
    pub(in crate::rdna_spmd) unsafe fn call(&self, name: &str, ret: Ty, args: &[LLVMValueRef]) -> LLVMValueRef {
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
    pub unsafe fn target(&self, op: super::super::dialect::TargetOp, args: super::super::dialect::Arguments, values: &[LLVMValueRef]) -> Vec<LLVMValueRef> {
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
