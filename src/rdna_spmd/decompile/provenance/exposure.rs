use super::super::address::{aligns, Region};
use super::demand::{demands, uses, Demand};
use super::layout::{merge, pointers, points};
use super::program::Program;
use super::sets::Sets;
use super::{Exposure, Spill};
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) fn exposures(sets: &Sets, env: &Environment) -> (Vec<Exposure>, Vec<Spill>) {
    let (program, layout) = (sets.program(), sets.layout());
    let (f, facts) = (program.f, program.facts);
    let used = uses(program);
    let mut demand: Option<Vec<Demand>> = None;
    let mut wanted = vec![0; sets.words()];
    for &id in layout.allocations() {
        if !env.exposed.contains(&id) {
            layout.mark(&mut wanted, Some(Region::Allocation(id)));
        }
    }
    let mut memos: HashMap<ValueId, (Vec<ValueId>, HashMap<ValueId, Vec<u64>>)> = HashMap::default();
    let mut exposing = Vec::new();
    let mut spills: Vec<Spill> = Vec::new();
    let mut buffer = [ValueId(0); 3];
    let mut acc = vec![0; sets.words()];
    for &b in &facts.order {
        for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
            let (operands, lanewise, assume, scratch): (&[ValueId], _, _, _) = match inst {
                Inst::Core { value, op, .. } => {
                    let n = exposed(program, *value, *op, &mut buffer);
                    (&buffer[..n], true, None, None)
                }
                Inst::Target { op, args, .. } if program.pure(*op) => (args.values(), true, None, None),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Wmma),
                    inputs,
                    ..
                } => (&inputs[..], false, None, None),
                Inst::Effect {
                    op: EffectOp::Memory { op, space, .. },
                    inputs,
                    ..
                } if !matches!(op, MemoryOp::Load(_) | MemoryOp::Fence) => {
                    let m = op.mask_input();
                    let bytes = match op {
                        MemoryOp::Store(size) => size.bytes(),
                        _ => 4,
                    };
                    let scratch = (*space == Space::Scratch).then_some((inputs[0], bytes));
                    (&inputs[1..m], false, Some(inputs[m]), scratch)
                }
                _ => continue,
            };
            let reaching = |set: &[u64]| set.iter().zip(&wanted).any(|(x, y)| x & y != 0);
            let relevant = match scratch {
                Some(_) => operands.iter().any(|&x| points(sets.of(x))),
                None => operands.iter().any(|&x| reaching(sets.of(x))),
            };
            if !relevant {
                continue;
            }
            let mut assume = assume;
            if !matches!(inst, Inst::Effect { op: EffectOp::Memory { .. }, .. }) {
                let mut data = false;
                inst.for_each_output(|o| data |= used[o.0]);
                if !data {
                    continue;
                }
                let demand = demand.get_or_insert_with(|| demands(program));
                let mut out = Demand::Dead;
                inst.for_each_output(|o| out = out.meet(demand[o.0]));
                match (out, lanewise) {
                    (Demand::Dead, _) => continue,
                    (Demand::Under(c), true) => assume = Some(c),
                    _ => {}
                }
            }
            acc.fill(0);
            for &x in operands {
                match assume {
                    Some(c) => {
                        let (held, memo) = memos.entry(c).or_insert_with(|| (program.conjuncts(c), HashMap::default()));
                        merge(&mut acc, &under(sets, x, held, memo));
                    }
                    None => {
                        merge(&mut acc, sets.of(x));
                    }
                }
            }
            if let Some((address, bytes)) = scratch {
                acc[0] &= !1;
                if !points(&acc) {
                    continue;
                }
                let words = program.words(address, bytes);
                let parts = if operands.iter().any(|&x| sets.has_parts(x)) {
                    (0..program.lanes())
                        .map(|lane| {
                            let mut part = vec![0; sets.words()];
                            for &x in operands {
                                merge(&mut part, sets.part(x, lane));
                            }
                            for (p, a) in part.iter_mut().zip(&acc) {
                                *p &= a;
                            }
                            layout.regions(&part)
                        })
                        .collect()
                } else {
                    vec![layout.regions(&acc)]
                };
                spills.push(Spill {
                    at: (b, index),
                    words,
                    data: operands.to_vec(),
                    mask: assume.unwrap(),
                    parts,
                });
                continue;
            }
            let candidates: Vec<u64> = layout
                .allocations()
                .iter()
                .copied()
                .filter(|&id| layout.has(&acc, Some(Region::Allocation(id))) && layout.has(&wanted, Some(Region::Allocation(id))))
                .collect();
            if !candidates.is_empty() {
                exposing.push(Exposure {
                    operands: operands.to_vec(),
                    assume,
                    candidates,
                });
            }
        }
    }
    (exposing, spills)
}

fn exposed(program: &Program, v: ValueId, op: Op, out: &mut [ValueId; 3]) -> usize {
    let wide = program.f.types[v.0] == Ty::I64;
    let high = |s: ValueId| wide && program.constant(s).is_some_and(|k| k >= 32);
    let list: &[ValueId] = match op {
        Op::Int(IntOp::And, a, b) => match (program.constant(a), program.constant(b)) {
            (None, Some(m)) | (Some(m), None) if aligns(m) => &[],
            _ => &[a, b],
        },
        Op::Int(IntOp::Mul | IntOp::Shl, a, b) | Op::Float(_, a, b) => &[a, b],
        Op::Int(IntOp::AShr, a, b) if !high(b) => &[a, b],
        Op::Int(IntOp::LShr, a, b) if wide && !high(b) => &[a, b],
        Op::Pack64(_, a) | Op::ReverseBits(a) | Op::Unary(_, a) => &[a],
        Op::Fma(a, b, c) | Op::MulAdd(a, b, c) => &[a, b, c],
        Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, ..) => &[],
        Op::Convert(_, _, a) => &[a],
        _ => &[],
    };
    out[..list.len()].copy_from_slice(list);
    list.len()
}

fn under(sets: &Sets, x: ValueId, held: &[ValueId], memo: &mut HashMap<ValueId, Vec<u64>>) -> Vec<u64> {
    let program = sets.program();
    let x = program.copies.get(&x).copied().unwrap_or(x);
    if !points(sets.of(x)) {
        return sets.of(x).to_vec();
    }
    if let Some(found) = memo.get(&x) {
        return found.clone();
    }
    let (f, facts) = (program.f, program.facts);
    let mut acc = vec![0; sets.words()];
    match facts.op(f, x) {
        Some(Op::Select(c, a, b)) => {
            let c = program.copies.get(&c).copied().unwrap_or(c);
            if held.contains(&c) {
                acc = under(sets, a, held, memo);
            } else {
                merge(&mut acc, &under(sets, a, held, memo));
                merge(&mut acc, &under(sets, b, held, memo));
            }
        }
        Some(Op::Int(IntOp::Or | IntOp::Xor, a, b)) => {
            let (y, z) = (under(sets, a, held, memo), under(sets, b, held, memo));
            pointers(&mut acc, &y);
            pointers(&mut acc, &z);
            if y[0] & z[0] & 1 != 0 {
                acc[0] |= 1;
            }
        }
        Some(Op::Int(IntOp::Add, a, b)) => {
            let (y, z) = (under(sets, a, held, memo), under(sets, b, held, memo));
            if z[0] & 1 != 0 {
                pointers(&mut acc, &y);
            }
            if y[0] & 1 != 0 {
                pointers(&mut acc, &z);
            }
            if y[0] & z[0] & 1 != 0 || (points(&y) && points(&z)) {
                acc[0] |= 1;
            }
        }
        Some(Op::Int(IntOp::Sub, a, b)) => {
            let (y, z) = (under(sets, a, held, memo), under(sets, b, held, memo));
            if z[0] & 1 != 0 {
                pointers(&mut acc, &y);
            }
            if y[0] & 1 != 0 || (points(&y) && points(&z)) {
                acc[0] |= 1;
            }
        }
        Some(Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, to, a))
            if to.bits() >= 32 && f.types[a.0].bits() >= 32 =>
        {
            acc = under(sets, a, held, memo);
        }
        Some(Op::Pack64(lo, _) | Op::UnpackLo(lo)) => acc = under(sets, lo, held, memo),
        Some(Op::UnpackHi(y)) => match facts.op(f, y) {
            Some(Op::Pack64(_, hi)) => acc = under(sets, hi, held, memo),
            _ => acc[0] |= 1,
        },
        Some(Op::Int(IntOp::And, a, b)) => match (program.constant(a), program.constant(b)) {
            (None, Some(m)) if aligns(m) => acc = under(sets, a, held, memo),
            (Some(m), None) if aligns(m) => acc = under(sets, b, held, memo),
            (None, None) => {
                let (y, z) = (under(sets, a, held, memo), under(sets, b, held, memo));
                if z[0] & 1 != 0 {
                    pointers(&mut acc, &y);
                }
                if y[0] & 1 != 0 {
                    pointers(&mut acc, &z);
                }
                acc[0] |= 1;
            }
            _ => acc[0] |= 1,
        },
        Some(Op::Int(IntOp::LShr, a, _)) if f.types[x.0] != Ty::I64 => {
            merge(&mut acc, &under(sets, a, held, memo));
            acc[0] |= 1;
        }
        _ => acc.copy_from_slice(sets.of(x)),
    }
    memo.insert(x, acc.clone());
    acc
}
