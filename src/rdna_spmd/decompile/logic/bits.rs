use super::super::address::compare;
use super::super::terms::interval;
use super::answers::wave_answer;
use super::atoms::{Atom, Choice};
use super::cells::cell_bit;
use super::floats::{float_order, float_relation, float_within};
use super::patterns::{constant_choices, projected_word};
use super::queries::Queries;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) fn compute_bit<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, v: ValueId) -> Bdd {
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
            _ if q.lane_values(f, facts, v).is_some() => {
                let values = q.lane_values(f, facts, v).unwrap();
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
            Op::Cmp(..) => match cell_bit(q, f, facts, v) {
                Some(bit) => bit,
                None => q.atom(opaque),
            },
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

pub(super) fn compute_view<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, w: ValueId) -> Bdd {
    let opaque = Atom::View(w);
    let Some(inst) = facts.inst(f, w) else {
        return q.atom(opaque);
    };
    match inst {
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::Ballot),
            inputs,
            ..
        } => q.bit(f, facts, inputs[0]),
        Inst::Core { op, .. } => match *op {
            _ if q.lane_values(f, facts, w).is_some() => {
                let values = q.lane_values(f, facts, w).unwrap();
                q.lanes(|l| values[l as usize] >> l & 1 == 1)
            }
            Op::Const(_, k) => q.word(k as u32),
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
            Op::Convert(Cvt::Bitcast, Ty::I32, a) if f.types[a.0] == Ty::I32 => {
                q.view(f, facts, a)
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
    } as u32;
    q.word(word)
}

fn small_word(f: &Func, facts: &Facts, w: ValueId) -> Option<(u32, u32)> {
    if f.types[w.0] != Ty::I32 || !facts.uniform[w.0] || !matches!(facts.site[w.0], Site::Inst { .. }) || facts.constant(f, w).is_some() {
        return None;
    }
    interval(f, facts, w, 0).filter(|&(_, high)| high < 32)
}

fn small_comparison<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, p: IntPred, a: ValueId, b: ValueId) -> Option<Bdd> {
    let (w, other, flipped) = match (small_word(f, facts, a), small_word(f, facts, b)) {
        (Some(_), None) => (a, b, false),
        (None, Some(_)) => (b, a, true),
        _ => return None,
    };
    if facts.constant(f, other).is_some() || f.types[other.0] != Ty::I32 {
        return None;
    }
    let (low, high) = small_word(f, facts, w)?;
    let values = q.lane_values(f, facts, other)?;
    let mut g = Bdd::FALSE;
    for x in low..=high {
        let is = q.word_is(w, x);
        let lanes = q.lanes(|l| {
            let o = values[l as usize];
            if flipped {
                compare(p, o, x)
            } else {
                compare(p, x, o)
            }
        });
        let both = q.m().and(is, lanes);
        g = q.m().or(g, both);
    }
    Some(g)
}

pub(super) fn lane_is<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, block: BlockId, target: ValueId) -> Option<Bdd> {
    if let Some((low, high)) = small_word(f, facts, target) {
        let mut g = Bdd::FALSE;
        for x in low..=high {
            let is = q.word_is(target, x);
            let lanes = q.lanes(|l| l == x);
            let both = q.m().and(is, lanes);
            g = q.m().or(g, both);
        }
        return Some(g);
    }
    if !interval(f, facts, target, 0).is_some_and(|(_, high)| high < 32) {
        return None;
    }
    for inst in &f.blocks[&block].insts {
        let Inst::Core {
            value,
            op: Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b),
            ..
        } = inst
        else {
            continue;
        };
        let other = if facts.is_lane_id(f, *a) {
            *b
        } else if facts.is_lane_id(f, *b) {
            *a
        } else {
            continue;
        };
        if other != target {
            continue;
        }
        let bit = q.bit(f, facts, *value);
        return Some(if *p == IntPred::Ne { q.m().not(bit) } else { bit });
    }
    None
}
