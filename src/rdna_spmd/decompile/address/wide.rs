use super::control::{bounds_at, incoming};
use super::fields::shifted_part;
use super::form::*;
use super::queries::*;
use super::symbols::Key;
use crate::rdna_spmd::analysis::facts::Site;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) fn compute_high<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, lane: usize) -> Option<Form> {
    let f = q.program().f;
    if f.types[v.0] != Ty::I64 {
        return None;
    }
    let low = |this: &mut Q, x: ValueId, at: BlockId| this.operand(x, at, lane, None).0.form;
    match q.program().facts.site[v.0] {
        Site::Param { block, .. } if block == f.entry || q.program().headers.contains(&block) => None,
        Site::Param { block, index } => {
            let mut joined: Option<Form> = None;
            for a in incoming(q, v, block, index)? {
                let high = high_operand(q, a, block, lane);
                match &joined {
                    None => joined = Some(high),
                    Some(old) if *old == high => {}
                    Some(_) => return None,
                }
            }
            joined
        }
        Site::Inst { block, index } => match &f.blocks[&block].insts[index] {
            Inst::Core { op, .. } => match *op {
                Op::Const(_, k) => Some(Form::constant((k >> 32) as u32)),
                Op::Pack64(_, hi) => Some(low(q, hi, block)),
                Op::Convert(Cvt::ZExt, _, _) => Some(Form::constant(0)),
                Op::Convert(Cvt::SExt, _, a) => {
                    let negative = if f.types[a.0] == Ty::I1 {
                        q.bit(a, lane, None).0?
                    } else {
                        let word = low(q, a, block);
                        let (lowest, highest) = q.symbols().bounds(&word)?;
                        if highest < 1 << 31 {
                            false
                        } else if lowest >= 1 << 31 {
                            true
                        } else {
                            return None;
                        }
                    };
                    Some(Form::constant(if negative { u32::MAX } else { 0 }))
                }
                Op::Int(k @ (IntOp::Add | IntOp::Sub), a, b) => {
                    let (la, lb) = (low(q, a, block), low(q, b, block));
                    let (ha, hb) = (high_operand(q, a, block, lane), high_operand(q, b, block, lane));
                    let carry = word_carry(q, v, lane, &la, &lb, k == IntOp::Sub);
                    Some(if k == IntOp::Add {
                        ha.add(&hb).add(&carry)
                    } else {
                        ha.sub(&hb).sub(&carry)
                    })
                }
                Op::Int(k @ (IntOp::Shl | IntOp::LShr | IntOp::AShr), a, s) => {
                    let amount = low(q, s, block).as_constant()? & 63;
                    let (la, ha) = (low(q, a, block), high_operand(q, a, block, lane));
                    match (k, amount) {
                        (_, 0) => Some(ha),
                        (IntOp::Shl, k) if k >= 32 => Some(la.scale(1u32 << (k - 32))),
                        (IntOp::Shl, k) => {
                            let spilled = shifted_part(q, v, &la, 32 - k, lane)?;
                            Some(ha.scale(1u32 << k).add(&spilled))
                        }
                        (IntOp::LShr, k) if k >= 32 => Some(Form::constant(0)),
                        (_, k) if q.symbols().bounds(&ha).is_some_and(|(_, top)| top < 1 << 31) => {
                            if k >= 32 {
                                Some(Form::constant(0))
                            } else {
                                shifted_part(q, v, &ha, k, lane)
                            }
                        }
                        (IntOp::LShr, k) => shifted_part(q, v, &ha, k, lane),
                        _ => None,
                    }
                }
                Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                    let (m, other) = match (q.program().facts.constant(f, a), q.program().facts.constant(f, b)) {
                        (_, Some(m)) => (m, a),
                        (Some(m), _) => (m, b),
                        _ => return None,
                    };
                    let m = (m >> 32) as u32;
                    let high = high_operand(q, other, block, lane);
                    match (k, m, high.as_constant()) {
                        (_, _, Some(h)) => Some(Form::constant(match k {
                            IntOp::And => h & m,
                            IntOp::Or => h | m,
                            _ => h ^ m,
                        })),
                        (IntOp::And, 0, _) => Some(Form::constant(0)),
                        (IntOp::And, u32::MAX, _) | (IntOp::Or | IntOp::Xor, 0, _) => Some(high),
                        _ => None,
                    }
                }
                Op::Select(c, a, b) => match q.bit(c, lane, None).0 {
                    Some(true) => Some(high_operand(q, a, block, lane)),
                    Some(false) => Some(high_operand(q, b, block, lane)),
                    None => {
                        let (x, y) = (high_operand(q, a, block, lane), high_operand(q, b, block, lane));
                        (x == y).then_some(x)
                    }
                },
                _ => None,
            },
            Inst::Effect {
                op:
                    EffectOp::Memory {
                        op: MemoryOp::Load(MemSize::B64),
                        ..
                    },
                inputs,
                ..
            } => {
                let address = q.operand(inputs[0], block, lane, None).0;
                if address.region != Some(Region::Kernarg) {
                    return None;
                }
                let at = address.form.sub(&q.symbols_mut().base(Region::Kernarg).form).as_constant()?;
                Some(Form::constant(match q.program().env.binding(at) {
                    Some(binding) => (binding.pointer >> 32) as u32,
                    None => q.program().env.kernarg_word(at + 4, 4),
                }))
            }
            _ => None,
        },
        Site::Unreached => None,
    }
}

fn high_operand<'a, Q: Queries<'a>>(q: &mut Q, x: ValueId, at: BlockId, lane: usize) -> Form {
    let high = q.high(x, lane);
    q.symbols_mut().leave(Value::of(high), at, lane).form
}

pub(super) fn known_high<'a, Q: Queries<'a>>(q: &mut Q, x: ValueId, at: BlockId, lane: usize) -> Option<Form> {
    let x = q.program().copies.get(&x).copied().unwrap_or(x);
    if !q.program().summed_high(x, &mut HashMap::default()) {
        return None;
    }
    let high = high_operand(q, x, at, lane);
    (!high.terms.iter().any(|(u, _)| q.symbols().opaque_highs.contains(u))).then_some(high)
}

fn word_carry<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, lane: usize, la: &Form, lb: &Form, borrow: bool) -> Form {
    let at = q.program().block_of(v);
    let (bounds_a, bounds_b) = (bounds_at(q, la, at), bounds_at(q, lb, at));
    let decided = if borrow {
        if lb.as_constant() == Some(0) || la == lb {
            Some(false)
        } else {
            match (bounds_a, bounds_b) {
                (Some((low_a, high_a)), Some((low_b, high_b))) if low_a >= high_b || high_a < low_b => Some(high_a < low_b),
                _ => None,
            }
        }
    } else if la.as_constant() == Some(0) || lb.as_constant() == Some(0) {
        Some(false)
    } else {
        match (bounds_a, bounds_b) {
            (Some((low_a, high_a)), Some((low_b, high_b))) if high_a + high_b < 1 << 32 || low_a + low_b >= 1 << 32 => {
                Some(low_a + low_b >= 1 << 32)
            }
            _ => None,
        }
    };
    if let Some(c) = decided {
        return Form::constant(c as u32);
    }
    let shared = q.program().facts.uniform[v.0];
    let block = q.program().block_of(v);
    let u = q.symbols_mut().intern(
        Key::Carry(v, if shared { ALL } else { lane as u8 }),
        UnknownInfo {
            rank: 0,
            shared,
            block,
            range: Some((0, 1)),
            through: Vec::new(),
            values: None,
        },
    );
    Form::unknown(u)
}
