use super::super::hazard::Reach;
use super::control::condition_limits;
use super::fields::mask;
use super::form::*;
use super::limits::{both_limits, Classes};
use super::queries::*;
use crate::rdna_spmd::ir::*;

pub(super) fn access_classes<'a, Q: Queries<'a>>(q: &mut Q, block: BlockId, predicate: Option<ValueId>, lane: usize) -> Classes {
    let mut limits = (*q.limits(block)).clone();
    if let Some(p) = predicate {
        let own = condition_limits(q, p, true, Some((block, lane)));
        limits = both_limits(limits, own);
    }
    limits.classes
}

pub(super) fn wide_at<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, at: BlockId, lane: usize, depth: usize) -> Option<Wide> {
    if depth > 12 {
        return None;
    }
    let v = q.program().copies.get(&v).copied().unwrap_or(v);
    if q.program().f.types[v.0] != Ty::I64 {
        return None;
    }
    if let Some(low) = q.operand(v, at, lane, None).0.form.as_constant() {
        if let Some(high) = q.high(v, lane).as_constant() {
            return Some(Wide {
                constant: (high as i128) << 32 | low as i128,
                terms: Vec::new(),
                words: Vec::new(),
            });
        }
    }
    let wide = match q.program().facts.op(q.program().f, v)? {
        Op::Convert(Cvt::ZExt, Ty::I64, x) if q.program().f.types[x.0] == Ty::I32 => {
            let form = q.operand(x, at, lane, None).0.form;
            if form.as_constant().is_none() && !(form.constant == 0 && matches!(form.terms.as_slice(), [(_, 1)])) && q.symbols().bounds(&form).is_none() {
                Wide {
                    constant: 0,
                    terms: Vec::new(),
                    words: vec![(form, 1)],
                }
            } else {
                Wide {
                    constant: form.constant as i128,
                    terms: form.terms.iter().map(|&(u, c)| (u, c as i128)).collect(),
                    words: Vec::new(),
                }
            }
        }
        Op::Int(k @ (IntOp::Add | IntOp::Sub), a, b) => {
            let x = wide_at(q, a, at, lane, depth + 1)?;
            let y = wide_at(q, b, at, lane, depth + 1)?;
            x.plus(&y, if k == IntOp::Add { 1 } else { -1 })
        }
        Op::Int(IntOp::Mul, a, b) => match (q.program().facts.constant(q.program().f, a), q.program().facts.constant(q.program().f, b)) {
            (_, Some(k)) => wide_at(q, a, at, lane, depth + 1)?.times(k as i128),
            (Some(k), _) => wide_at(q, b, at, lane, depth + 1)?.times(k as i128),
            _ => return None,
        },
        Op::Int(IntOp::Shl, a, s) => {
            let k = q.program().facts.constant(q.program().f, s)?;
            if k >= 64 {
                return None;
            }
            wide_at(q, a, at, lane, depth + 1)?.times(1i128 << k)
        }
        _ => return None,
    };
    let (mut low, mut high) = (wide.constant, wide.constant);
    let ranges = wide
        .terms
        .iter()
        .map(|&(u, c)| (c, q.symbols().range(u).unwrap_or((0, u32::MAX))))
        .chain(wide.words.iter().map(|&(_, c)| (c, (0, u32::MAX))))
        .collect::<Vec<_>>();
    for (c, (l, h)) in ranges {
        let (a, b) = (c * l as i128, c * h as i128);
        low += a.min(b);
        high += a.max(b);
    }
    (low >= 0 && high < 1 << 64).then_some(wide)
}

pub(super) fn resource_span<'a, Q: Queries<'a>>(q: &mut Q, at: (BlockId, usize), reach: Reach, lane: usize) -> Option<(Value, u32, Option<Form>)> {
    let Inst::Target { args, outputs, .. } = &q.program().f.blocks[&at.0].insts[at.1] else {
        return None;
    };
    let (args, v) = (args.values().to_vec(), outputs.first()?.0);
    let word = |this: &mut Q, index: usize| -> Option<Form> {
        let x = *args.get(index)?;
        Some(this.operand(x, at.0, lane, None).0.form)
    };
    let base = word(q, 0)?.scale(256);
    match reach {
        Reach::Anywhere => None,
        Reach::Node { offset, shift, bytes, kinds } => {
            let mut sum = Form::constant(0);
            let mut exact: Option<u64> = Some(0);
            for &(index, mask) in offset {
                let arg = *args.get(index)?;
                let x = word(q, index)?;
                sum = sum.add(&masked_part(q, v, &x, mask as u32, lane)?);
                exact = match (exact, x.as_constant()) {
                    (Some(acc), Some(low)) => {
                        let high = if q.program().f.types[arg.0] == Ty::I64 { q.high(arg, lane).as_constant() } else { Some(0) };
                        high.map(|high| acc.wrapping_add((((high as u64) << 32) | low as u64) & mask))
                    }
                    _ => None,
                };
            }
            let bytes = match kinds {
                Some((index, mask, table)) => {
                    let x = word(q, index)?;
                    let known = x.terms.iter().all(|&(_, c)| (c as u64) & mask == 0);
                    match known {
                        true => table.iter().find(|&&(kind, _)| kind == (x.constant as u64) & mask).map_or(bytes, |&(_, b)| b),
                        false => bytes,
                    }
                }
                None => bytes,
            };
            let narrow = offset.iter().all(|&(index, mask)| {
                let arg = args[index];
                q.program().f.types[arg.0] != Ty::I64 || mask >> 32 == 0 || q.high(arg, lane).as_constant().is_some_and(|h| (h as u64) & (mask >> 32) == 0)
            });
            let high = match (word(q, 0)?.as_constant(), word(q, 1)?.as_constant(), exact) {
                (Some(r0), Some(r1), Some(offset)) => {
                    let start = ((((r1 & 0xff) as u64) << 40) | ((r0 as u64) << 8)).wrapping_add(offset << shift);
                    Some(Form::constant((start >> 32) as u32))
                }
                (Some(r0), Some(r1), None) if narrow => q.symbols().bounds(&sum).and_then(|(low, high)| {
                    let origin = (((r1 & 0xff) as u64) << 40) | ((r0 as u64) << 8);
                    let first = origin.wrapping_add(low << shift) >> 32;
                    let last = origin.wrapping_add(high << shift) >> 32;
                    (high < 1 << 32 && high << shift >> shift == high && first == last).then(|| Form::constant(first as u32))
                }),
                _ => None,
            };
            Some((Value::of(base.add(&sum.scale(1u32 << shift))), bytes, high))
        }
        Reach::Image => {
            let (w1, w2, w4) = (word(q, 1)?.as_constant()?, word(q, 2)?.as_constant()?, word(q, 4)?.as_constant()?);
            let width = ((w1 >> 30) | (w2 & 0x3fff) << 2) as u64 + 1;
            let height = ((w2 >> 14) & 0xffff) as u64 + 1;
            let pitch = (w4 & 0xffff) as u64;
            let row = if pitch != 0 { pitch + 1 } else { width }.div_ceil(128) * 128;
            let constant = |this: &Q, index: usize| args.get(index).and_then(|&x| this.program().facts.constant(this.program().f, x));
            let sampler: Option<Vec<u32>> = (8..12).map(|i| constant(q, i).map(|k| k as u32)).collect();
            let point = sampler.as_ref().is_some_and(|s| crate::buffer::get_bits_u32(s, 84, 2) == 0);
            let coordinates = [coordinate(q, args.get(14).copied(), at.0, lane), coordinate(q, args.get(15).copied(), at.0, lane)];
            let axis = |coord: usize, size: u64, index: usize| -> Option<Option<(u64, u64)>> {
                let full = Some(Some((0, size - 1)));
                let (Some(sampler), Some(unrm), Some((low, high))) = (sampler.as_ref(), constant(q, 13), coordinates[coord - 14]) else {
                    return full;
                };
                if !point {
                    return full;
                }
                let unnormalized = unrm != 0 || crate::buffer::get_bits_u32(sampler, 15, 1) != 0;
                let texel = |c: f32| if unnormalized { c } else { c * size as f32 }.floor();
                let (first, last) = (texel(low), texel(high));
                if !(first >= i32::MIN as f32 && last <= i32::MAX as f32 && last - first <= 4.0 * size as f32) {
                    return full;
                }
                let mode = match crate::buffer::get_bits_u32(sampler, index * 3, 3) {
                    0 if unnormalized => 2,
                    1 if unnormalized => 3,
                    mode => mode,
                };
                let mut span: Option<(u64, u64)> = None;
                for t in first as i32..=last as i32 {
                    if let Some(x) = crate::rdna_translator::bvh::clamp_texel(t, size as i32, mode) {
                        let x = x as u64;
                        span = Some(span.map_or((x, x), |(a, b)| (a.min(x), b.max(x))));
                    }
                }
                Some(span)
            };
            let (Some(x), Some(y)) = (axis(14, width, 0)?, axis(15, height, 1)?) else {
                return Some((Value::of(base), 0, None));
            };
            let (start, end) = (y.0 * row + x.0, y.1 * row + x.1 + 1);
            (end < 1 << 31).then(|| (Value::of(base.add(&Form::constant(start as u32))), (end - start) as u32, None))
        }
    }
}

fn coordinate<'a, Q: Queries<'a>>(q: &mut Q, x: Option<ValueId>, at: BlockId, lane: usize) -> Option<(f32, f32)> {
    let x = x?;
    if let Some(k) = q.program().facts.constant(q.program().f, x) {
        let c = f32::from_bits(k as u32);
        return (!c.is_nan()).then_some((c, c));
    }
    let Some(Op::Convert(cvt @ (Cvt::UnsignedToFloatRte | Cvt::SignedToFloatRte), Ty::F32, i)) = q.program().facts.op(q.program().f, x) else {
        return None;
    };
    if q.program().f.types[i.0] != Ty::I32 {
        return None;
    }
    let form = q.operand(i, at, lane, None).0.form;
    let (low, high) = match form.as_constant() {
        Some(k) => (k as u64, k as u64),
        None => q.symbols().bounds(&form)?,
    };
    if high >= 1 << 32 || (cvt == Cvt::SignedToFloatRte && high >= 1 << 31) {
        return None;
    }
    Some((low as f32, high as f32))
}

fn masked_part<'a, Q: Queries<'a>>(q: &mut Q, v: ValueId, form: &Form, m: u32, lane: usize) -> Option<Form> {
    if m == u32::MAX {
        return Some(form.clone());
    }
    if let Some(x) = form.as_constant() {
        return Some(Form::constant(x & m));
    }
    if let Some(low) = low_bits(form, m, IntOp::And) {
        return Some(low);
    }
    let aligning = (!m).wrapping_add(1).is_power_of_two();
    let shared = form.terms.iter().all(|&(u, _)| q.symbols().unknowns[u as usize].shared);
    (aligning && shared).then(|| mask(q, v, form, m, lane).form)
}
