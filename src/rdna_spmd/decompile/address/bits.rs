use super::super::encoding::Encoding;
use super::control::{can_take, incoming};
use super::form::*;
use super::limits::{decide, offset};
use super::queries::*;
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::ir::*;

pub(super) fn compute_bit<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
    let (f, facts) = (q.program().f, q.program().facts);
    if let Some(&root) = q.program().copies.get(&v) {
        return q.bit(root, lane, assume);
    }
    match facts.site[v.0] {
        Site::Param { block, index } if block == f.entry => {
            let valid = q.symbols().valid(lane);
            match q.program().inputs[index].source {
                ParameterSource::MaskBit(r) if r == q.program().exec => unassumed(Some(valid)),
                _ => unassumed(None),
            }
        }
        Site::Param { block, index } => {
            let Some(arguments) = incoming(q, v, block, index) else {
                if let Some(bit) = q.loop_bit(block, index, lane) {
                    return unassumed(Some(bit));
                }
                return unassumed(narrowed(q, v, block, index, lane));
            };
            let mut joined: Option<Option<bool>> = None;
            for a in arguments {
                let (bit, _) = q.bit(a, lane, None);
                match joined {
                    None => joined = Some(bit),
                    Some(old) if old == bit => {}
                    Some(_) => return unassumed(None),
                }
            }
            unassumed(joined.flatten())
        }
        Site::Inst { block, index } => match f.blocks[&block].insts[index] {
            Inst::Core { op, .. } => core_bit(q, op, block, lane, assume),
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::Any),
                ref inputs,
                ..
            } => {
                let input = inputs[0];
                let mut unknown = false;
                for l in 0..LANES {
                    if !q.symbols().valid(l) {
                        continue;
                    }
                    match q.bit(input, l, None).0 {
                        Some(true) => return unassumed(Some(true)),
                        Some(false) => {}
                        None => unknown = true,
                    }
                }
                unassumed((!unknown).then_some(false))
            }
            _ => unassumed(None),
        },
        Site::Unreached => unassumed(None),
    }
}

fn core_bit<'a, Q: Queries<'a>>(q: &mut Q, op: Op, block: BlockId, lane: usize, assume: Option<ValueId>) -> Assumed<Option<bool>> {
    let mut used = Reliance::default();
    macro_rules! bit {
        ($this:expr, $x:expr) => {{
            let (b, u) = $this.bit($x, lane, assume);
            used |= u;
            b
        }};
    }
    let result = match op {
        Op::Const(_, k) => Some(k & 1 != 0),
        Op::Env(Env::ValidLane) => Some(q.symbols().valid(lane)),
        Op::Int(IntOp::And, a, b) if aperture(q, a, b, lane).is_some() => {
            aperture(q, a, b, lane)
        }
        Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
            let x = bit!(q, a);
            if k == IntOp::And && x == Some(false) {
                Some(false)
            } else if k == IntOp::Or && x == Some(true) {
                Some(true)
            } else {
                let y = bit!(q, b);
                match (k, x, y) {
                    (IntOp::And, _, Some(false)) => Some(false),
                    (IntOp::Or, _, Some(true)) => Some(true),
                    (_, Some(x), Some(y)) => Some(match k {
                        IntOp::And => x && y,
                        IntOp::Or => x || y,
                        _ => x != y,
                    }),
                    _ => None,
                }
            }
        }
        Op::Select(c, a, b) => match bit!(q, c) {
            Some(true) => bit!(q, a),
            Some(false) => bit!(q, b),
            None => {
                let (x, y) = (bit!(q, a), bit!(q, b));
                if x == y {
                    x
                } else {
                    None
                }
            }
        },
        Op::Convert(Cvt::Trunc, Ty::I1, x) | Op::Convert(Cvt::Bitcast, Ty::I1, x)
            if q.program().f.types[x.0] != Ty::I1 =>
        {
            let (value, u) = q.value(x, lane, assume);
            used |= u;
            match value.form.as_constant() {
                Some(k) => Some(k & 1 != 0),
                None => match q.program().facts.op(q.program().f, x) {
                    Some(Op::Int(IntOp::LShr, w, s)) => {
                        match q.value(s, lane, None).0.form.as_constant() {
                            Some(k) if k < 32 && (q.program().facts.uniform[w.0] || k as usize == lane) => {
                                q.word_bit(w, k as usize)
                            }
                            _ => None,
                        }
                    }
                    _ => None,
                },
            }
        }
        Op::Convert(k @ (Cvt::FloatToSignedSatRtz | Cvt::FloatToUnsignedSatRtz), Ty::I1, x) => {
            q.program().converted(k, Ty::I1, x).map(|b| b != 0)
        }
        Op::Convert(_, Ty::I1, x) => bit!(q, x),
        Op::Cmp(pred, a, b) if q.program().f.types[a.0] == Ty::I1 => {
            match (bit!(q, a), bit!(q, b)) {
                (Some(x), Some(y)) => match pred {
                    IntPred::Eq => Some(x == y),
                    IntPred::Ne => Some(x != y),
                    _ => None,
                },
                _ => None,
            }
        }
        Op::Cmp(pred, a, b) => {
            let (x, u) = q.value(a, lane, assume);
            used |= u;
            let (y, u) = q.value(b, lane, assume);
            used |= u;
            let wide = q.program().f.types[a.0] == Ty::I64;
            let difference = x.form.sub(&y.form);
            let plain = match (pred, difference.as_constant()) {
                (IntPred::Ne, Some(d)) if d != 0 => Some(true),
                (IntPred::Eq, Some(d)) if d != 0 => Some(false),
                _ if wide => None,
                (IntPred::Eq | IntPred::Ule | IntPred::Uge | IntPred::Sle | IntPred::Sge, Some(0)) => Some(true),
                (IntPred::Ne | IntPred::Ult | IntPred::Ugt | IntPred::Slt | IntPred::Sgt, Some(0)) => Some(false),
                (_, Some(d)) => match (q.symbols().bounds(&x.form), q.symbols().bounds(&y.form)) {
                    (Some(bx), Some(by)) => decide(pred, bx, by).or_else(|| offset(pred, d, by)),
                    (_, Some(by)) => offset(pred, d, by),
                    _ => None,
                },
                _ => match (q.symbols().bounds(&x.form), q.symbols().bounds(&y.form)) {
                    (Some(bx), Some(by)) => decide(pred, bx, by),
                    _ => None,
                },
            };
            if plain.is_some() || wide {
                plain
            } else {
                let limits = q.limits(block);
                let mut encoding = Encoding::new(&q.symbols().unknowns, &|_: &UnknownInfo| false);
                encoding.decide(pred, &x.form, &y.form, &limits.classes, &limits.orders)
            }
        }
        _ => None,
    };
    (result, used)
}

fn aperture<'a, Q: Queries<'a>>(q: &mut Q, a: ValueId, b: ValueId, lane: usize) -> Option<bool> {
    let (f, facts) = (q.program().f, q.program().facts);
    let (Some(Op::Cmp(IntPred::Uge, base, low)), Some(Op::Cmp(IntPred::Ult, again, high))) = (facts.op(f, a), facts.op(f, b))
    else {
        return None;
    };
    let env = |x: ValueId| match facts.op(f, x) {
        Some(Op::Env(e)) => Some(e),
        _ => None,
    };
    let bounded = match facts.op(f, high) {
        Some(Op::Int(IntOp::Add, x, y)) => matches!(
            (env(x), env(y)),
            (Some(Env::ScratchBase), Some(Env::ScratchSize)) | (Some(Env::ScratchSize), Some(Env::ScratchBase))
        ),
        _ => false,
    };
    if base != again || env(low) != Some(Env::ScratchBase) || !bounded {
        return None;
    }
    let (pointer, _) = q.value(base, lane, None);
    pointer.region.map(|r| r == Region::Private)
}

pub(super) fn compute_word_bit<'a, Q: Queries<'a>>(q: &mut Q, w: ValueId, bit: usize) -> Option<bool> {
    if let Some(k) = q.value(w, bit, None).0.form.as_constant() {
        return Some(k >> bit & 1 != 0);
    }
    if let Site::Param { block, index } = q.program().facts.site[w.0] {
        if block == q.program().f.entry {
            return None;
        }
        let own = q.program().rank[&block];
        let mut entering: Option<Option<bool>> = None;
        for &(pred, slot) in &q.program().facts.incoming[&block].clone() {
            if !can_take(q, pred, slot, block) {
                continue;
            }
            let arg = q.program().f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
            if q.program().headers.contains(&block) && q.program().rank[&pred] >= own {
                let edge = q.program().edge_condition(pred, slot);
                if !q.reason(|conditions, program| conditions.word_implies(program, arg, w, edge)) {
                    return None;
                }
                continue;
            }
            let b = q.word_bit(arg, bit);
            match entering {
                None => entering = Some(b),
                Some(old) if old == b => {}
                Some(_) => return None,
            }
        }
        let entering = entering.flatten();
        return if q.program().headers.contains(&block) {
            (entering == Some(false)).then_some(false)
        } else {
            entering
        };
    }
    match q.program().facts.inst(q.program().f, w) {
        Some(Inst::Core { op, .. }) => match *op {
            Op::Const(_, k) => Some(k >> bit & 1 != 0),
            Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                let x = q.word_bit(a, bit);
                match (k, x) {
                    (IntOp::And, Some(false)) => return Some(false),
                    (IntOp::Or, Some(true)) => return Some(true),
                    _ => {}
                }
                let y = q.word_bit(b, bit);
                match (k, x, y) {
                    (IntOp::And, _, Some(false)) => Some(false),
                    (IntOp::Or, _, Some(true)) => Some(true),
                    (_, Some(x), Some(y)) => Some(match k {
                        IntOp::And => x && y,
                        IntOp::Or => x || y,
                        _ => x != y,
                    }),
                    _ => None,
                }
            }
            Op::Select(c, a, b) => match q.bit(c, bit, None).0 {
                Some(true) => q.word_bit(a, bit),
                Some(false) => q.word_bit(b, bit),
                None => {
                    let (x, y) = (q.word_bit(a, bit), q.word_bit(b, bit));
                    if x == y {
                        x
                    } else {
                        None
                    }
                }
            },
            Op::Convert(Cvt::Bitcast, _, a) => q.word_bit(a, bit),
            _ => None,
        },
        Some(Inst::Effect {
            op: EffectOp::Wave(WaveOp::Ballot),
            inputs,
            ..
        }) => {
            if !q.symbols().valid(bit) {
                return Some(false);
            }
            q.bit(inputs[0], bit, None).0
        }
        _ => None,
    }
}

fn narrowed<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, block: BlockId, index: usize, lane: usize) -> Option<bool> {
    let own = q.program().rank[&block];
    let mut entering = Vec::new();
    for &(pred, slot) in &q.program().facts.incoming[&block].clone() {
        if !can_take(q, pred, slot, block) {
            continue;
        }
        let arg = q.program().f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
        if q.program().rank[&pred] >= own {
            let edge = q.program().edge_condition(pred, slot);
            if !q.reason(|conditions, program| conditions.implies(program, arg, v, edge)) {
                return None;
            }
        } else {
            entering.push(arg);
        }
    }
    for a in entering {
        if q.bit(a, lane, None).0 != Some(false) {
            return None;
        }
    }
    Some(false)
}
