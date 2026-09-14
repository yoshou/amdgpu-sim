//! Shared scalar/packet LLVM lowering of the core SSA operations.
//! No ISA opcodes, register files or per-lane runtime calls.
use super::super::ir::*;
use super::super::native::{Builder, Type, Value};

pub(crate) struct Emitter {
    pub(in crate::rdna_spmd) state: Option<Box<dyn std::any::Any>>,
    registry: std::sync::Arc<super::super::dialect::DialectRegistry>,
    pub(in crate::rdna_spmd) ir: Builder,
    width: Option<u32>,
    wide: std::cell::Cell<bool>,
    pub(in crate::rdna_spmd) valid_lane: Option<Value>,
    pub(in crate::rdna_spmd) outside_lanes: Option<Value>,
    pub(in crate::rdna_spmd) scratch: Option<(Value, Value)>,
    pub(in crate::rdna_spmd) lane_id: Option<Value>,
}
impl Emitter {
    pub(in crate::rdna_spmd) fn set_lane_id(&mut self, lane_base: Value) {
        let ir = self.ir;
        let base = ir.trunc(lane_base, ir.i32());
        self.lane_id = Some(match self.width {
            Some(width) => {
                let lanes: Vec<u32> = (0..width).collect();
                let broadcast = ir.splat(base, width);
                ir.add(broadcast, ir.const_i32_vector(&lanes))
            }
            None => base,
        });
    }
    pub fn new(ir: Builder, width: Option<u32>, registry: std::sync::Arc<super::super::dialect::DialectRegistry>) -> Self {
        Self {
            registry,
            state: None,
            ir,
            wide: std::cell::Cell::new(false),
            width,
            valid_lane: None,
            outside_lanes: None,
            scratch: None,
            lane_id: None,
        }
    }
    pub(in crate::rdna_spmd) fn ty(&self, t: Ty) -> Type {
        let t = match t {
            Ty::I1 if self.wide.get() => self.ir.i64(),
            Ty::I1 => self.ir.i1(),
            Ty::I32 => self.ir.i32(),
            Ty::I64 => self.ir.i64(),
            Ty::F32 => self.ir.f32(),
            Ty::F64 => self.ir.f64(),
        };
        self.shaped(t)
    }
    pub(in crate::rdna_spmd) fn wide(&self) -> bool { self.wide.get() }
    pub(in crate::rdna_spmd) fn set_wide(&mut self, wide: bool) { self.wide.set(wide && self.width.is_some()); }
    pub(in crate::rdna_spmd) fn to_bool(&self, v: Value) -> Value {
        if !self.wide.get() || !v.is_vector() { return v; }
        self.ir.icmp(IntPred::Slt, v, v.ty().null())
    }
    pub(in crate::rdna_spmd) fn from_bool(&self, v: Value) -> Value {
        if !self.wide.get() || !v.is_vector() { return v; }
        self.ir.sext(v, self.ty(Ty::I1))
    }
    pub(in crate::rdna_spmd) fn shaped(&self, scalar: Type) -> Type {
        self.width.map_or(scalar, |w| scalar.vector(w))
    }
    pub(in crate::rdna_spmd) fn constant(&self, ty: Ty, bits: u64) -> Value {
        let (t, bits) = if ty == Ty::I1 && self.wide.get() { (self.ir.i64(), if bits & 1 == 1 { u64::MAX } else { 0 }) } else { (self.ir.int(ty.bits()), bits) };
        let v = t.const_int(bits);
        let v = match self.width {
            Some(w) => self.ir.const_vector(&vec![v; w as usize]),
            None => v,
        };
        if ty.integer() { v } else { self.ir.const_bitcast(v, self.ty(ty)) }
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
    pub(in crate::rdna_spmd) fn state<T: 'static>(&self) -> Option<&T> { self.state.as_ref().and_then(|s| s.downcast_ref::<T>()) }
    pub(in crate::rdna_spmd) fn call(&self, name: &str, ret: Ty, args: &[Value]) -> Value {
        let types: Vec<Type> = args.iter().map(|v| v.ty()).collect();
        self.ir.call_named(name, self.ty(ret), &types, args)
    }
    pub fn target(&self, op: super::super::dialect::TargetOp, args: super::super::dialect::Arguments, values: &[Value]) -> Vec<Value> {
        let spec = self.registry.operation(op).expect("unverified target");
        let args = args.values().iter().enumerate().map(|(i, id)| if spec.inputs.get(i) == Some(&Ty::I1) { self.to_bool(values[id.0]) } else { values[id.0] }).collect::<Vec<_>>();
        let wide = self.wide.replace(false);
        let results = spec.emit(self, &args);
        self.wide.set(wide);
        results.into_iter().zip(&spec.outputs).map(|(v, &t)| if t == Ty::I1 { self.from_bool(v) } else { v }).collect()
    }
    pub fn op(&self, ty: Ty, op: Op, values: &[Value]) -> Value {
        let ir = self.ir;
        let v = |id: ValueId| values[id.0];
        match op {
            Op::Env(Env::LaneId) => self.lane_id.expect("LaneId requires the invocation environment"),
            Op::Env(Env::PacketLaneId) => match self.width {
                Some(width) => ir.const_i32_vector(&(0..width).collect::<Vec<u32>>()),
                None => self.constant(Ty::I32, 0),
            },
            Op::Env(env @ (Env::ScratchBase | Env::ScratchSize)) => {
                let (base, size) = self.scratch.expect("scratch requires the invocation environment");
                let value = if env == Env::ScratchBase { base } else { size };
                match self.width {
                    Some(width) => ir.splat(value, width),
                    None => value,
                }
            }
            Op::Env(Env::ValidLane) => self.valid_lane.expect("ValidLane requires the invocation environment"),
            Op::Env(Env::OutsideLanes) => self.outside_lanes.expect("OutsideLanes requires the invocation environment"),
            Op::Const(t, bits) => self.constant(t, bits),
            Op::Pack64(a, c) => {
                let lo = ir.zext(v(a), self.ty(Ty::I64));
                let hi = ir.zext(v(c), self.ty(Ty::I64));
                let hi = ir.shl(hi, self.constant(Ty::I64, 32));
                ir.or(lo, hi)
            }
            Op::UnpackLo(a) | Op::UnpackHi(a) => {
                let wide = if matches!(op, Op::UnpackHi(_)) { ir.lshr(v(a), self.constant(Ty::I64, 32)) } else { v(a) };
                ir.trunc(wide, self.ty(Ty::I32))
            }
            Op::TrailingZeros(a) => self.call(&format!("llvm.cttz.{}", self.suffix(ty)), ty, &[v(a), ir.ci1(false)]),
            Op::LeadingZeros(a) => self.call(&format!("llvm.ctlz.{}", self.suffix(ty)), ty, &[v(a), ir.ci1(false)]),
            Op::PopulationCount(a) => self.call(&format!("llvm.ctpop.{}", self.suffix(ty)), ty, &[v(a)]),
            Op::ReverseBits(a) => self.call(&format!("llvm.bitreverse.{}", self.suffix(ty)), ty, &[v(a)]),
            Op::Int(op, a, c) => {
                let (a, mut c) = (v(a), v(c));
                if matches!(op, IntOp::Shl | IntOp::LShr | IntOp::AShr) {
                    c = ir.and(c, self.constant(ty, (ty.bits() - 1) as u64));
                }
                match op {
                    IntOp::Add => ir.add(a, c),
                    IntOp::Sub => ir.sub(a, c),
                    IntOp::Mul => ir.mul(a, c),
                    IntOp::And => ir.and(a, c),
                    IntOp::Or => ir.or(a, c),
                    IntOp::Xor => ir.xor(a, c),
                    IntOp::Shl => ir.shl(a, c),
                    IntOp::LShr => ir.lshr(a, c),
                    IntOp::AShr => ir.ashr(a, c),
                }
            }
            Op::Cmp(p, a, c) => {
                let predicate = match p {
                    IntPred::Eq => IntPred::Eq,
                    IntPred::Ne => IntPred::Ne,
                    IntPred::Ult => IntPred::Ult,
                    IntPred::Ugt => IntPred::Ugt,
                    IntPred::Ule => IntPred::Ule,
                    IntPred::Uge => IntPred::Uge,
                    IntPred::Slt => IntPred::Slt,
                    IntPred::Sgt => IntPred::Sgt,
                    IntPred::Sle => IntPred::Sle,
                    IntPred::Sge => IntPred::Sge,
                };
                self.from_bool(ir.icmp(predicate, v(a), v(c)))
            }
            Op::FCmp(p, a, c) => {
                let predicate = match p {
                    FloatPred::Oeq => FloatPred::Oeq,
                    FloatPred::Ogt => FloatPred::Ogt,
                    FloatPred::Oge => FloatPred::Oge,
                    FloatPred::Olt => FloatPred::Olt,
                    FloatPred::Ole => FloatPred::Ole,
                    FloatPred::One => FloatPred::One,
                    FloatPred::Ord => FloatPred::Ord,
                    FloatPred::Uno => FloatPred::Uno,
                    FloatPred::Ueq => FloatPred::Ueq,
                    FloatPred::Ugt => FloatPred::Ugt,
                    FloatPred::Uge => FloatPred::Uge,
                    FloatPred::Ult => FloatPred::Ult,
                    FloatPred::Ule => FloatPred::Ule,
                    FloatPred::Une => FloatPred::Une,
                };
                self.from_bool(ir.fcmp(predicate, v(a), v(c)))
            }
            Op::Select(c, a, d) => ir.select(self.to_bool(v(c)), v(a), v(d)),
            Op::Float(op, a, c) => match op {
                FloatOp::Add => ir.fadd(v(a), v(c)),
                FloatOp::Sub => ir.fsub(v(a), v(c)),
                FloatOp::Mul => ir.fmul(v(a), v(c)),
                FloatOp::Div => ir.fdiv(v(a), v(c)),
                FloatOp::MinNum | FloatOp::MaxNum => {
                    let name = if op == FloatOp::MinNum { "minnum" } else { "maxnum" };
                    self.call(&format!("llvm.{name}.{}", self.suffix(ty)), ty, &[v(a), v(c)])
                }
            },
            Op::Unary(op, a) => match op {
                FloatUnary::Neg => ir.fneg(v(a)),
                FloatUnary::Abs => self.call(&format!("llvm.fabs.{}", self.suffix(ty)), ty, &[v(a)]),
            },
            Op::Fma(a, c, d) | Op::MulAdd(a, c, d) => {
                let name = if matches!(op, Op::Fma(..)) { "fma" } else { "fmuladd" };
                self.call(&format!("llvm.{name}.{}", self.suffix(ty)), ty, &[v(a), v(c), v(d)])
            }
            Op::Convert(op, to, a) => {
                let a = v(a);
                let t = self.ty(to);
                let converted = match op {
                    Cvt::Bitcast => ir.bitcast(a, t),
                    Cvt::ZExt => ir.zext(a, t),
                    Cvt::SExt => ir.sext(a, t),
                    Cvt::Trunc => ir.trunc(a, t),
                    Cvt::SignedToFloatRte => ir.sitofp(a, t),
                    Cvt::UnsignedToFloatRte => ir.uitofp(a, t),
                    Cvt::FloatResizeRte => if to == Ty::F64 { ir.fpext(a, t) } else { ir.fptrunc(a, t) },
                    Cvt::FloatToSignedSatRtz | Cvt::FloatToUnsignedSatRtz => {
                        let at = a.ty();
                        let at = if self.width.is_some() { at.element() } else { at };
                        let from = if at.is_float() { Ty::F32 } else { Ty::F64 };
                        let name = if op == Cvt::FloatToSignedSatRtz { "fptosi" } else { "fptoui" };
                        self.call(&format!("llvm.{name}.sat.{}.{}", self.suffix(to), self.suffix(from)), to, &[a])
                    }
                };
                if to == Ty::I1 { self.from_bool(converted) } else { converted }
            }
        }
    }
}
