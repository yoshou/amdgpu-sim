use super::control::bounds_at;
use super::form::*;
use super::queries::*;
use super::symbols::Key;
use crate::rdna_spmd::ir::*;

pub(super) fn shift<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, form: &Form, k: u32, lane: usize) -> Value {
    if let Some(x) = form.as_constant() {
        return Value::constant(x >> k);
    }
    if k == 0 {
        return Value::of(form.clone());
    }
    let at = q.program().block_of(v);
    if let Some((low, high)) = bounds_at(q, form, at) {
        if low >> k == high >> k {
            return Value::constant((low >> k) as u32);
        }
    }
    if form.terms.iter().any(|&(u, _)| !q.symbols().unknowns[u as usize].shared) {
        return q.symbols_mut().opaque(v, lane, None);
    }
    let terms = Form {
        constant: 0,
        terms: form.terms.clone(),
    };
    let bounds = q.symbols().bounds(&terms);
    let c = form.constant;
    let (block, key) = q.symbols().variance(&[&terms], q.program().block_of(v));
    let mut wrap = 0u32;
    let mut result = if k <= terms.alignment() {
        if bounds.is_none() {
            wrap = (1u32 << k) - 1;
        }
        Form {
            constant: 0,
            terms: form.terms.iter().map(|&(u, c)| (u, c >> k)).collect(),
        }
    } else {
        let (low, high) = bounds.unwrap_or((0, u32::MAX as u64));
        let through = q.symbols().through(&terms);
        let u = q.symbols_mut().intern(
            Key::Shifted(key, form.terms.clone(), k),
            UnknownInfo {
                rank: 0,
                shared: true,
                block,
                range: Some(((low >> k) as u32, (high >> k) as u32)),
                through,
                values: None,
            },
        );
        q.symbols_mut().derived.entry(u).or_default().push(v);
        let mut high_part = Form::unknown(u);
        let below = c & ((1u32 << k) - 1);
        if below != 0 {
            let carry = q.symbols_mut().intern(
                Key::ShiftCarry(key, form.terms.clone(), k, below),
                UnknownInfo {
                    rank: 0,
                    shared: true,
                    block,
                    range: Some((0, 1)),
                    through: Vec::new(),
                    values: None,
                },
            );
            high_part = high_part.add(&Form::unknown(carry));
        }
        high_part
    };
    result = result.add(&Form::constant(c >> k));
    let wraps = c != 0 && bounds.is_none_or(|(_, high)| high + c as u64 >= 1 << 32);
    if wraps {
        wrap += 1;
    }
    if wrap > 0 {
        let w = q.symbols_mut().intern(
            Key::ShiftWrap(key, form.terms.clone(), k, c),
            UnknownInfo {
                rank: 0,
                shared: true,
                block,
                range: Some((0, wrap)),
                through: Vec::new(),
                values: None,
            },
        );
        result = result.sub(&Form::unknown(w).scale(1u32 << (32 - k)));
    }
    Value::of(result)
}

pub(super) fn mask<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, form: &Form, m: u32, lane: usize) -> Value {
    if m == u32::MAX {
        return Value::of(form.clone());
    }
    let at = q.program().block_of(v);
    if let Some((_, high)) = bounds_at(q, form, at) {
        let j = form.alignment().min(32);
        let low = if j >= 32 { form.constant } else { form.constant & ((1u32 << j) - 1) };
        let top = 64 - high.leading_zeros();
        let upto = if top >= 32 { u32::MAX } else { (1u32 << top) - 1 };
        let above = if j >= 32 { 0 } else { upto & !((1u32 << j) - 1) };
        if above & !m == 0 {
            return Value::of(form.sub(&Form::constant(low & !m)));
        }
    }
    let low = m.wrapping_add(1).is_power_of_two() || m == u32::MAX;
    if low {
        let k = m.count_ones();
        let kept: Vec<(Unknown, u32)> = form
            .terms
            .iter()
            .copied()
            .filter(|&(_, c)| c.trailing_zeros() < k)
            .collect();
        let rest = Form {
            constant: form.constant,
            terms: kept,
        };
        if let Some(x) = rest.as_constant() {
            return Value::constant(x & m);
        }
        if let Some((low, high)) = q.symbols().bounds(&rest) {
            if high <= u32::MAX as u64 && low & !(m as u64) == high & !(m as u64) {
                return Value::of(rest.sub(&Form::constant((low & !(m as u64)) as u32)));
            }
        }
        let j = rest.alignment();
        if rest.terms.iter().all(|&(u, _)| q.symbols().unknowns[u as usize].shared) {
            let (block, key) = q.symbols().variance(&[&rest], q.program().block_of(v));
            let through = q.symbols().through(&rest);
            let terms = Form {
                constant: 0,
                terms: rest.terms.clone(),
            };
            let whole = match q.symbols().bounds(&terms) {
                Some((low, high)) if low & !(m as u64) == high & !(m as u64) => {
                    Some((terms.sub(&Form::constant((low & !(m as u64)) as u32)), high - (low & !(m as u64))))
                }
                _ => None,
            };
            let (masked, top) = match whole {
                Some(exact) => exact,
                None => {
                    let u = q.symbols_mut().intern(
                        Key::Masked(key, rest.terms.clone(), m),
                        UnknownInfo {
                            rank: 0,
                            shared: true,
                            block,
                            range: Some((0, m >> j)),
                            through,
                            values: None,
                        },
                    );
                    q.symbols_mut().derived.entry(u).or_default().push(v);
                    (Form::unknown(u).scale(1 << j), m as u64)
                }
            };
            let below = form.constant & ((1u32 << j) - 1);
            let carried = (form.constant - below) & m;
            let mut result = masked.add(&Form::constant(below));
            if carried != 0 && top + carried as u64 + below as u64 > m as u64 {
                let carry = q.symbols_mut().intern(
                    Key::MaskCarry(key, rest.terms.clone(), m, carried + below),
                    UnknownInfo {
                        rank: 0,
                        shared: true,
                        block,
                        range: Some((0, 1)),
                        through: Vec::new(),
                        values: None,
                    },
                );
                result = result.add(&Form::constant(carried)).sub(&Form::unknown(carry).scale(m + 1));
            } else {
                result = result.add(&Form::constant(carried));
            }
            return Value::of(result);
        }
        return q.symbols_mut().opaque(v, lane, Some((0, m)));
    }
    let high = !m;
    if high.wrapping_add(1).is_power_of_two() {
        let k = high.count_ones();
        if form.alignment() >= k {
            return Value::of(form.sub(&Form::constant(form.constant & high)));
        }
        if form.terms.iter().all(|&(u, _)| q.symbols().unknowns[u as usize].shared) {
            let above = shift(q, v, form, k, lane);
            return Value::of(above.form.scale(1 << k));
        }
    }
    let s = m.trailing_zeros();
    let field = m >> s;
    if s > 0 && field.wrapping_add(1).is_power_of_two() && form.terms.iter().all(|&(u, _)| q.symbols().unknowns[u as usize].shared) {
        let above = shift(q, v, form, s, lane);
        let inner = mask(q, v, &above.form, field, lane);
        return Value::of(inner.form.scale(1 << s));
    }
    q.symbols_mut().opaque(v, lane, Some((0, m)))
}

pub(super) fn split_low_bits<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, form: &Form, c: u32, op: IntOp, lane: usize) -> Option<Value> {
    if form.terms.iter().any(|&(u, _)| !q.symbols().unknowns[u as usize].shared) {
        return None;
    }
    let j = 32 - c.leading_zeros();
    let top = if j == 32 { u32::MAX } else { (1u32 << j) - 1 };
    let above = if j == 32 { Form::constant(0) } else { shift(q, v, form, j, lane).form.scale(1 << j) };
    if op == IntOp::Or && c == top {
        return Some(Value::of(above.add(&Form::constant(c))));
    }
    let range = if op == IntOp::Or { (c, top) } else { (0, top) };
    let block = q.program().block_of(v);
    let through = q.symbols().through(form);
    let low = q.symbols_mut().intern(
        Key::Low(v, form.clone(), c),
        UnknownInfo {
            rank: 0,
            shared: true,
            block,
            range: Some(range),
            through,
            values: None,
        },
    );
    Some(Value::of(above.add(&Form::unknown(low))))
}

pub(super) fn shifted_part<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, form: &Form, k: u32, lane: usize) -> Option<Form> {
    if let Some(x) = form.as_constant() {
        return Some(Form::constant(x >> k));
    }
    if k == 0 {
        return Some(form.clone());
    }
    form.terms
        .iter()
        .all(|&(u, _)| q.symbols().unknowns[u as usize].shared)
        .then(|| shift(q, v, form, k, lane).form)
}
