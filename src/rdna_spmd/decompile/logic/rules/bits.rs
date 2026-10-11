use super::super::kernel::{constant_choices, projected_word, Atom, Choice, Queries};
use super::answers::{wave_answer, HasAnswers};
use super::cells::{cell_bit, HasCells};
use super::floats::{float_order, float_relation, float_within};
use super::lanes::{lane_values, HasLanes};
use super::words::small_comparison;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) fn compute_bit<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, v: ValueId) -> Bdd
where
    Q::State: HasLanes + HasCells + HasAnswers,
{
    let opaque = Atom::Bit(v);
    let Some(inst) = facts.inst(f, v) else {
        return q.atom(opaque);
    };
    match inst {
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::Any),
            inputs,
            outputs,
            ..
        } => {
            let local = q.local(Choice::Query(outputs[0].0));
            if local == Bdd::FALSE {
                return wave_answer(q, f, facts, v);
            }
            let bit = q.bit(f, facts, inputs[0]);
            if local == Bdd::TRUE {
                return bit;
            }
            let wave = wave_answer(q, f, facts, v);
            q.m().ite(local, bit, wave)
        }
        Inst::Core { op, .. } => match *op {
            Op::Const(_, k) => Manager::constant(k != 0),
            Op::Env(Env::ValidLane) => Bdd::TRUE,
            Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                let (a, b) = (q.bit(f, facts, a), q.bit(f, facts, b));
                match k {
                    IntOp::And => q.m().and(a, b),
                    IntOp::Or => q.m().or(a, b),
                    _ => q.m().xor(a, b),
                }
            }
            Op::Select(c, a, b) => {
                let c = q.bit(f, facts, c);
                let a = q.bit(f, facts, a);
                let b = q.bit(f, facts, b);
                q.m().ite(c, a, b)
            }
            Op::Convert(Cvt::Bitcast, Ty::I1, a) => q.bit(f, facts, a),
            Op::Convert(Cvt::Trunc, Ty::I1, s) => match projected_word(f, facts, s) {
                Some(w) => q.view(f, facts, w),
                None => q.atom(opaque),
            },
            _ if lane_values(q, f, facts, v).is_some() => {
                let values = lane_values(q, f, facts, v).unwrap();
                q.lanes(|l| values[l as usize] & 1 == 1)
            }
            Op::Cmp(p, a, b) if small_comparison(q, f, facts, p, a, b).is_some() => small_comparison(q, f, facts, p, a, b).unwrap(),
            Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b) if f.types[a.0] == Ty::I1 => {
                let (x, y) = (q.bit(f, facts, a), q.bit(f, facts, b));
                if p == IntPred::Ne {
                    q.m().xor(x, y)
                } else {
                    q.m().iff(x, y)
                }
            }
            Op::Cmp(..) => {
                let cell = cell_bit(q, f, facts, v);
                let own = q.atom(opaque);
                let own_var = q.m().decompose(own).map(|(var, _, _)| var);
                let decided = cell.filter(|&bit| !q.support(bit).iter().any(|&var| Some(var) == own_var));
                match decided.or_else(|| word_compare(q, f, facts, v)).or(cell) {
                    Some(bit) => bit,
                    None => own,
                }
            }
            Op::FCmp(p, a, b) => {
                let within = float_within(q, f, facts, v, p, a, b);
                let leaf = if within == q.atom(Atom::Bit(v)) { float_relation(q, f, facts, v, p, a, b) } else { within };
                float_order(q, f, facts, p, a, b, leaf, &mut HashMap::default())
            }
            _ => q.atom(opaque),
        },
        _ => q.atom(opaque),
    }
}

fn word_compare<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, v: ValueId) -> Option<Bdd> {
    let Some(Op::Cmp(p, a, b)) = facts.op(f, v) else {
        return None;
    };
    let (w, k, flipped) = match (facts.constant(f, a), facts.constant(f, b)) {
        (None, Some(k)) => (a, k, false),
        (Some(k), None) => (b, k, true),
        _ => return None,
    };
    if f.types[w.0] != Ty::I32 || facts.compared[w.0] < 2 {
        return None;
    }
    let signed = matches!(p, IntPred::Slt | IntPred::Sgt | IntPred::Sle | IntPred::Sge);
    let target = if signed { k as u32 ^ 0x8000_0000 } else { k as u32 };
    let mut equal = Bdd::TRUE;
    let mut below = Bdd::FALSE;
    for i in (0..32u8).rev() {
        let raw = q.atom(Atom::WordBit(w, i));
        let x = if signed && i == 31 { q.m().not(raw) } else { raw };
        let clear = q.m().not(x);
        if target >> i & 1 == 1 {
            let here = q.m().and(equal, clear);
            below = q.m().or(below, here);
            equal = q.m().and(equal, x);
        } else {
            equal = q.m().and(equal, clear);
        }
    }
    let (lhs_below, lhs_equal) = if flipped {
        let at_most = q.m().or(below, equal);
        (q.m().not(at_most), equal)
    } else {
        (below, equal)
    };
    Some(match p {
        IntPred::Eq => lhs_equal,
        IntPred::Ne => q.m().not(lhs_equal),
        IntPred::Ult | IntPred::Slt => lhs_below,
        IntPred::Ule | IntPred::Sle => q.m().or(lhs_below, lhs_equal),
        IntPred::Uge | IntPred::Sge => q.m().not(lhs_below),
        IntPred::Ugt | IntPred::Sgt => {
            let at_most = q.m().or(lhs_below, lhs_equal);
            q.m().not(at_most)
        }
    })
}

pub(super) fn compute_view<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, w: ValueId) -> Bdd
where
    Q::State: HasLanes,
{
    let opaque = Atom::View(w);
    let Some(inst) = facts.inst(f, w) else {
        return q.atom(opaque);
    };
    match inst {
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::Ballot { high }),
            inputs,
            ..
        } => {
            let own = q.bit(f, facts, inputs[0]);
            if f.lanes == 32 {
                return own;
            }
            let other = q.atom(opaque);
            q.half(*high, own, other)
        }
        Inst::Core { op, .. } => match *op {
            _ if lane_values(q, f, facts, w).is_some() => {
                let values = lane_values(q, f, facts, w).unwrap();
                q.lanes(|l| values[l as usize] >> (l & 31) & 1 == 1)
            }
            Op::Const(ty, k) => q.word_of(ty, k),
            Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                if constant_choices(f, facts, a).is_some() && constant_choices(f, facts, b).is_some() {
                    return constant_word(q, f, facts, k, a, b);
                }
                let (a, b) = (q.view(f, facts, a), q.view(f, facts, b));
                match k {
                    IntOp::And => q.m().and(a, b),
                    IntOp::Or => q.m().or(a, b),
                    _ => q.m().xor(a, b),
                }
            }
            Op::Select(c, a, b) => {
                let c = q.bit(f, facts, c);
                let a = q.view(f, facts, a);
                let b = q.view(f, facts, b);
                q.m().ite(c, a, b)
            }
            Op::Convert(Cvt::Bitcast, to @ (Ty::I32 | Ty::I64), a) if f.types[a.0] == to => {
                q.view(f, facts, a)
            }
            Op::Pack64(a, b) => {
                let low = q.view(f, facts, a);
                if f.lanes == 32 {
                    return low;
                }
                let high = q.view(f, facts, b);
                let upper = q.atom(Atom::Lane(5));
                q.m().ite(upper, high, low)
            }
            Op::UnpackLo(a) => {
                let own = q.view(f, facts, a);
                if f.lanes == 32 {
                    return own;
                }
                let other = q.atom(opaque);
                q.half(false, own, other)
            }
            Op::UnpackHi(a) if f.lanes == 64 => {
                let own = q.view(f, facts, a);
                let other = q.atom(opaque);
                q.half(true, own, other)
            }
            _ => q.atom(opaque),
        },
        _ => q.atom(opaque),
    }
}

fn constant_word<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, k: IntOp, a: ValueId, b: ValueId) -> Bdd {
    for (x, other) in [(a, b), (b, a)] {
        if let Some(Op::Select(c, p, r)) = facts.op(f, x) {
            let c = q.bit(f, facts, c);
            let p = constant_word(q, f, facts, k, p, other);
            let r = constant_word(q, f, facts, k, r, other);
            return q.m().ite(c, p, r);
        }
    }
    let (x, y) = (facts.constant(f, a).unwrap(), facts.constant(f, b).unwrap());
    let word = match k {
        IntOp::And => x & y,
        IntOp::Or => x | y,
        _ => x ^ y,
    };
    q.word_of(f.types[a.0], word)
}
