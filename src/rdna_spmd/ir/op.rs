//! Typed, width-independent SSA operations. ISA registers and lane-mask word
//! conventions are resolved by the ISA lifter, outside the core ops.

use super::{Ty, ValueId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum IntPred {
    Eq,
    Ne,
    Ult,
    Ugt,
    Ule,
    Uge,
    Slt,
    Sgt,
    Sle,
    Sge,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloatOp {
    Add,
    Sub,
    Mul,
    Div,
    MinNum,
    MaxNum,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloatUnary {
    Neg,
    Abs,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Complete predicate set, including forms not used by current ISA input.
pub(crate) enum FloatPred {
    Oeq,
    Ogt,
    Oge,
    Olt,
    Ole,
    One,
    Ord,
    Uno,
    Ueq,
    Ugt,
    Uge,
    Ult,
    Ule,
    Une,
}
/// The current ISA lift uses rte for integer -> float and saturating rtz for
/// float -> integer (including NaN -> 0), retaining those semantics explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Core conversion forms are independent of the initial ISA coverage.
pub(crate) enum Cvt {
    SignedToFloatRte,
    UnsignedToFloatRte,
    FloatToSignedSatRtz,
    FloatToUnsignedSatRtz,
    FloatResizeRte,
    ZExt,
    SExt,
    Trunc,
    Bitcast,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Env { LaneId, PacketLaneId, ValidLane, OutsideLanes, ScratchBase, ScratchSize }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Env(Env),
    Int(IntOp, ValueId, ValueId),
    /// Integer width for zero, otherwise the number of trailing zero bits.
    TrailingZeros(ValueId),
    /// Integer width for zero, otherwise the number of leading zero bits.
    LeadingZeros(ValueId),
    PopulationCount(ValueId),
    ReverseBits(ValueId),
    Pack64(ValueId, ValueId),
    UnpackLo(ValueId),
    UnpackHi(ValueId),
    Cmp(IntPred, ValueId, ValueId),
    Float(FloatOp, ValueId, ValueId),
    Unary(FloatUnary, ValueId),
    FCmp(FloatPred, ValueId, ValueId),
    Fma(ValueId, ValueId, ValueId),
    /// Optional contraction, kept distinct from fused fma to preserve the
    /// existing f64 lowering's llvm.fmuladd contract.
    MulAdd(ValueId, ValueId, ValueId),
    Convert(Cvt, Ty, ValueId),
    Const(Ty, u64),
    Select(ValueId, ValueId, ValueId),
}
impl Op {
    pub fn map(self, mut f: impl FnMut(ValueId) -> ValueId) -> Self {
        match self {
            Self::Int(o, a, b) => Self::Int(o, f(a), f(b)),
            Self::TrailingZeros(a) => Self::TrailingZeros(f(a)),
            Self::LeadingZeros(a) => Self::LeadingZeros(f(a)),
            Self::PopulationCount(a) => Self::PopulationCount(f(a)),
            Self::ReverseBits(a) => Self::ReverseBits(f(a)),
            Self::Pack64(a, b) => Self::Pack64(f(a), f(b)),
            Self::UnpackLo(a) => Self::UnpackLo(f(a)),
            Self::UnpackHi(a) => Self::UnpackHi(f(a)),
            Self::Cmp(o, a, b) => Self::Cmp(o, f(a), f(b)),
            Self::Float(o, a, b) => Self::Float(o, f(a), f(b)),
            Self::FCmp(o, a, b) => Self::FCmp(o, f(a), f(b)),
            Self::Unary(o, a) => Self::Unary(o, f(a)),
            Self::Convert(o, t, a) => Self::Convert(o, t, f(a)),
            Self::Fma(a, b, c) => Self::Fma(f(a), f(b), f(c)),
            Self::MulAdd(a, b, c) => Self::MulAdd(f(a), f(b), f(c)),
            Self::Select(a, b, c) => Self::Select(f(a), f(b), f(c)),
            Self::Const(..) | Self::Env(..) => self,
        }
    }
    pub fn result_type(self, types: &[Ty]) -> Result<Ty, &'static str> {
        let ty = |v: ValueId| {
            types
                .get(v.0)
                .copied()
                .ok_or("undefined or non-dominating value")
        };
        let pair = |a, b| {
            let a = ty(a)?;
            if a != ty(b)? {
                Err("operand type mismatch")
            } else {
                Ok(a)
            }
        };
        match self {
            Self::Env(Env::LaneId | Env::PacketLaneId | Env::OutsideLanes) => Ok(Ty::I32),
            Self::Env(Env::ValidLane) => Ok(Ty::I1),
            Self::Env(Env::ScratchBase | Env::ScratchSize) => Ok(Ty::I64),
            Self::Pack64(a, b) => {
                if pair(a, b)? != Ty::I32 { return Err("pack64 requires two i32 words"); }
                Ok(Ty::I64)
            }
            Self::UnpackLo(a) | Self::UnpackHi(a) => {
                if ty(a)? != Ty::I64 { return Err("unpack requires i64"); }
                Ok(Ty::I32)
            }
            Self::TrailingZeros(a) | Self::LeadingZeros(a) | Self::PopulationCount(a) | Self::ReverseBits(a) => {
                let t = ty(a)?;
                if !matches!(t, Ty::I32 | Ty::I64) { return Err("bit count requires an integer word"); }
                Ok(t)
            }
            Self::Int(op, a, b) => {
                let t = pair(a, b)?;
                if !t.integer()
                    || (t == Ty::I1 && !matches!(op, IntOp::And | IntOp::Or | IntOp::Xor))
                {
                    return Err("invalid integer arithmetic");
                }
                Ok(t)
            }
            Self::Cmp(_, a, b) => {
                if !pair(a, b)?.integer() {
                    return Err("integer comparison requires integers");
                }
                Ok(Ty::I1)
            }
            Self::Float(_, a, b) | Self::FCmp(_, a, b) => {
                let t = pair(a, b)?;
                if t.integer() {
                    return Err("float operation requires floats");
                }
                Ok(if matches!(self, Self::FCmp(..)) {
                    Ty::I1
                } else {
                    t
                })
            }
            Self::Unary(_, a) => {
                let t = ty(a)?;
                if t.integer() {
                    Err("float operation requires floats")
                } else {
                    Ok(t)
                }
            }
            Self::Fma(a, b, c) | Self::MulAdd(a, b, c) => {
                let t = pair(a, b)?;
                if t.integer() || t != ty(c)? {
                    Err("invalid multiply-add")
                } else {
                    Ok(t)
                }
            }
            Self::Const(t, bits) => {
                if t.bits() < 64 && bits >> t.bits() != 0 {
                    Err("constant exceeds type width")
                } else {
                    Ok(t)
                }
            }
            Self::Convert(op, to, a) => {
                let from = ty(a)?;
                let valid = match op {
                    Cvt::SignedToFloatRte | Cvt::UnsignedToFloatRte => {
                        from.integer() && !to.integer()
                    }
                    Cvt::FloatToSignedSatRtz | Cvt::FloatToUnsignedSatRtz => {
                        !from.integer() && to.integer()
                    }
                    Cvt::FloatResizeRte => !from.integer() && !to.integer(),
                    Cvt::ZExt | Cvt::SExt => {
                        from.integer() && to.integer() && from.bits() < to.bits()
                    }
                    Cvt::Trunc => from.integer() && to.integer() && from.bits() > to.bits(),
                    Cvt::Bitcast => from.bits() == to.bits(),
                };
                if valid {
                    Ok(to)
                } else {
                    Err("invalid conversion")
                }
            }
            Self::Select(c, a, b) => {
                if ty(c)? != Ty::I1 {
                    return Err("select condition requires i1");
                }
                pair(a, b)
            }
        }
    }
}
