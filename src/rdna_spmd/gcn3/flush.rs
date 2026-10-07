use crate::rdna_spmd::dialect::DialectRegistry;
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::rdna4::lift::{Builder, Lowering};

const HONORING: [&str; 10] = [
    "floor.f32",
    "ceil.f32",
    "trunc.f32",
    "rndne.f32",
    "frexp_mant.f32",
    "frexp_exp.f32",
    "ldexp.f32",
    "div_scale.f32",
    "div_fmas.f32",
    "div_fixup.f32",
];

pub fn zeroed(b: &mut Builder, value: ValueId) -> ValueId {
    let bits = b.push(Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, value));
    let exponent_mask = b.k(Ty::I32, 0x7f80_0000);
    let exponent = b.int(IntOp::And, bits, exponent_mask);
    let zero = b.k(Ty::I32, 0);
    let tiny = b.push(Ty::I1, Op::Cmp(IntPred::Eq, exponent, zero));
    let sign_mask = b.k(Ty::I32, 0x8000_0000);
    let sign = b.int(IntOp::And, bits, sign_mask);
    let kept = b.push(Ty::I32, Op::Select(tiny, sign, bits));
    b.push(Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, kept))
}

fn honors(op: &Op) -> (bool, bool) {
    match op {
        Op::Float(FloatOp::Add | FloatOp::Sub | FloatOp::Mul | FloatOp::Div, ..) => (true, true),
        Op::Fma(..) | Op::MulAdd(..) => (true, true),
        Op::FCmp(..) => (true, false),
        Op::Convert(Cvt::FloatResizeRte, ..) => (true, true),
        Op::Convert(Cvt::FloatToSignedSatRtz | Cvt::FloatToUnsignedSatRtz, ..) => (true, false),
        _ => (false, false),
    }
}

pub fn flushed(lowering: Lowering, registry: &DialectRegistry, inputs: bool, outputs: bool) -> Lowering {
    if !inputs && !outputs {
        return lowering;
    }
    let Lowering::TypedAlu {
        inputs: given,
        outputs: written,
        scalar,
        expr,
    } = lowering
    else {
        return lowering;
    };
    let expr = expr.expr();
    let mut types = expr.params.clone();
    let mut b = Builder::new(registry, given);
    let mut map: Vec<ValueId> = (0..expr.params.len()).map(ValueId).collect();
    for inst in &expr.insts {
        match inst {
            ExprInst::Core(ty, op) => {
                let (operands, result) = honors(op);
                let op = op.map(|v| {
                    if operands && inputs && types[v.0] == Ty::F32 {
                        zeroed(&mut b, map[v.0])
                    } else {
                        map[v.0]
                    }
                });
                let mut value = b.push(*ty, op);
                if result && outputs && *ty == Ty::F32 {
                    value = zeroed(&mut b, value);
                }
                map.push(value);
                types.push(*ty);
            }
            ExprInst::Target { op, args, outputs: tys } => {
                let honoring = HONORING.contains(&registry.operation(*op).unwrap().name);
                let args = args.map(|v| {
                    if honoring && inputs && types[v.0] == Ty::F32 {
                        zeroed(&mut b, map[v.0])
                    } else {
                        map[v.0]
                    }
                });
                let values = b.target(*op, args);
                for (&value, &ty) in values.iter().zip(tys) {
                    map.push(if honoring && outputs && ty == Ty::F32 {
                        zeroed(&mut b, value)
                    } else {
                        value
                    });
                    types.push(ty);
                }
            }
        }
    }
    let results = written
        .iter()
        .zip(&expr.results)
        .map(|(&output, result)| (output, map[result.0]))
        .collect();
    b.finish_many(scalar, results)
}
