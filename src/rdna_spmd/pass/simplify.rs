use super::super::analysis::Analyses;
use super::super::ir::{Cvt, IntOp, Op, Ty, ValueId, *};
use std::collections::{BTreeMap, BTreeSet};

pub struct Simplify;
impl super::Pass for Simplify {
    fn name(&self) -> &str {
        "simplify"
    }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool {
        run(f) > 0
    }
}

fn half(defs: &[Option<Op>], v: ValueId) -> Option<(ValueId, bool)> {
    match defs[v.0] {
        Some(Op::UnpackLo(x)) => Some((x, false)),
        Some(Op::UnpackHi(x)) => Some((x, true)),
        _ => None,
    }
}

fn other_half(defs: &[Option<Op>], v: ValueId, x: ValueId, hi: bool) -> bool {
    half(defs, v) == Some((x, hi))
}

struct Rewrite {
    fresh: Vec<(Ty, Op)>,
    last: Op,
}

fn slot(i: usize) -> ValueId {
    ValueId(usize::MAX - i)
}

fn lift(defs: &[Option<Op>], lo: ValueId, hi: ValueId) -> Option<Rewrite> {
    let bitwise = |v: ValueId| match defs[v.0] {
        Some(Op::Int(op @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b)) => Some((op, a, b)),
        _ => None,
    };
    let identity = |op: IntOp| Op::Const(Ty::I32, if op == IntOp::And { 0xffff_ffff } else { 0 });
    if let Some((x, false)) = half(defs, lo) {
        if let Some((op, a, b)) = bitwise(hi) {
            let k = if other_half(defs, a, x, true) {
                b
            } else if other_half(defs, b, x, true) {
                a
            } else {
                return None;
            };
            return Some(Rewrite {
                fresh: vec![(Ty::I32, identity(op)), (Ty::I64, Op::Pack64(slot(0), k))],
                last: Op::Int(op, x, slot(1)),
            });
        }
    }
    if let Some((x, true)) = half(defs, hi) {
        if let Some((op, a, b)) = bitwise(lo) {
            let k = if other_half(defs, a, x, false) {
                b
            } else if other_half(defs, b, x, false) {
                a
            } else {
                return None;
            };
            return Some(Rewrite {
                fresh: vec![(Ty::I32, identity(op)), (Ty::I64, Op::Pack64(k, slot(0)))],
                last: Op::Int(op, x, slot(1)),
            });
        }
    }
    None
}

fn hoist(defs: &[Option<Op>], lo: ValueId, hi: ValueId) -> Option<Rewrite> {
    if let (Some((x, false)), Some(Op::Select(c, p, q))) = (half(defs, lo), defs[hi.0]) {
        if other_half(defs, q, x, true) {
            return Some(Rewrite {
                fresh: vec![(Ty::I64, Op::Pack64(lo, p))],
                last: Op::Select(c, slot(0), x),
            });
        }
        if other_half(defs, p, x, true) {
            return Some(Rewrite {
                fresh: vec![(Ty::I64, Op::Pack64(lo, q))],
                last: Op::Select(c, x, slot(0)),
            });
        }
    }
    if let (Some((x, true)), Some(Op::Select(c, p, q))) = (half(defs, hi), defs[lo.0]) {
        if other_half(defs, q, x, false) {
            return Some(Rewrite {
                fresh: vec![(Ty::I64, Op::Pack64(p, hi))],
                last: Op::Select(c, slot(0), x),
            });
        }
        if other_half(defs, p, x, false) {
            return Some(Rewrite {
                fresh: vec![(Ty::I64, Op::Pack64(q, hi))],
                last: Op::Select(c, x, slot(0)),
            });
        }
    }
    None
}

fn insert(defs: &[Option<Op>], lo: ValueId, hi: ValueId) -> Option<Rewrite> {
    let (low, high) = (half(defs, lo), half(defs, hi));
    if low.is_none() && high.is_none() {
        return None;
    }
    let mut fresh: Vec<(Ty, Op)> = Vec::new();
    let mut push = |ty: Ty, op: Op| -> ValueId {
        fresh.push((ty, op));
        slot(fresh.len() - 1)
    };
    let low_word = match low {
        Some((x, false)) => {
            let mask = push(Ty::I64, Op::Const(Ty::I64, 0xffff_ffff));
            push(Ty::I64, Op::Int(IntOp::And, x, mask))
        }
        Some((x, true)) => {
            let shift = push(Ty::I64, Op::Const(Ty::I64, 32));
            push(Ty::I64, Op::Int(IntOp::LShr, x, shift))
        }
        None => push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lo)),
    };
    let high_word = match high {
        Some((x, true)) => {
            let mask = push(Ty::I64, Op::Const(Ty::I64, 0xffff_ffff_0000_0000));
            push(Ty::I64, Op::Int(IntOp::And, x, mask))
        }
        Some((x, false)) => {
            let shift = push(Ty::I64, Op::Const(Ty::I64, 32));
            push(Ty::I64, Op::Int(IntOp::Shl, x, shift))
        }
        None => {
            let wide = push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, hi));
            let shift = push(Ty::I64, Op::Const(Ty::I64, 32));
            push(Ty::I64, Op::Int(IntOp::Shl, wide, shift))
        }
    };
    Some(Rewrite {
        fresh,
        last: Op::Int(IntOp::Or, low_word, high_word),
    })
}

fn plan(f: &Func, defs: &[Option<Op>], ty: Ty, op: Op) -> Option<Rewrite> {
    match op {
        Op::Pack64(lo, hi) => match (half(defs, lo), half(defs, hi)) {
            (Some((x, false)), Some((y, true))) if x == y => None,
            _ => hoist(defs, lo, hi)
                .or_else(|| lift(defs, lo, hi))
                .or_else(|| insert(defs, lo, hi)),
        },
        Op::Select(c, a, b) => match (half(defs, a), half(defs, b)) {
            (Some((x, false)), Some((y, false))) if f.types[x.0] == f.types[y.0] => Some(Rewrite {
                fresh: vec![(f.types[x.0], Op::Select(c, x, y))],
                last: Op::UnpackLo(slot(0)),
            }),
            (Some((x, true)), Some((y, true))) if f.types[x.0] == f.types[y.0] => Some(Rewrite {
                fresh: vec![(f.types[x.0], Op::Select(c, x, y))],
                last: Op::UnpackHi(slot(0)),
            }),
            _ => None,
        },
        Op::Convert(Cvt::ZExt, Ty::I64, a) if ty == Ty::I64 => match half(defs, a) {
            Some((x, false)) if f.types[x.0] == Ty::I64 => Some(Rewrite {
                fresh: vec![(Ty::I64, Op::Const(Ty::I64, 0xffff_ffff))],
                last: Op::Int(IntOp::And, x, slot(0)),
            }),
            Some((x, true)) if f.types[x.0] == Ty::I64 => Some(Rewrite {
                fresh: vec![(Ty::I64, Op::Const(Ty::I64, 32))],
                last: Op::Int(IntOp::LShr, x, slot(0)),
            }),
            _ => None,
        },
        _ => None,
    }
}

fn expand(f: &mut Func) -> usize {
    let defs = f.definitions();
    let mut plans: Vec<(BlockId, usize, Rewrite)> = Vec::new();
    for (&id, block) in &f.blocks {
        for (index, inst) in block.insts.iter().enumerate() {
            let Inst::Core { ty, op, .. } = inst else {
                continue;
            };
            if let Some(rewrite) = plan(f, &defs, *ty, *op) {
                plans.push((id, index, rewrite));
            }
        }
    }
    let count = plans.len();
    for (id, index, rewrite) in plans.into_iter().rev() {
        let fresh: Vec<ValueId> = rewrite.fresh.iter().map(|(ty, _)| f.value(*ty)).collect();
        let place = |op: Op| {
            op.map(|v| {
                if v.0 > usize::MAX - fresh.len() - 1 {
                    fresh[usize::MAX - v.0]
                } else {
                    v
                }
            })
        };
        let insts: Vec<Inst> = rewrite
            .fresh
            .iter()
            .enumerate()
            .map(|(i, (ty, op))| Inst::Core {
                value: fresh[i],
                ty: *ty,
                op: place(*op),
            })
            .collect();
        let block = f.blocks.get_mut(&id).unwrap();
        if let Inst::Core { op, .. } = &mut block.insts[index] {
            *op = place(rewrite.last);
        }
        block.insts.splice(index..index, insts);
    }
    count
}

fn common(f: &mut Func) -> usize {
    let mut map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    for block in f.blocks.values() {
        let mut seen: Vec<(Ty, Op, ValueId)> = Vec::new();
        for inst in &block.insts {
            let Inst::Core { value, ty, op } = inst else {
                continue;
            };
            match seen.iter().find(|(t, o, _)| t == ty && o == op) {
                Some(&(_, _, previous)) => {
                    map.insert(*value, previous);
                }
                None => seen.push((*ty, *op, *value)),
            }
        }
    }
    if map.is_empty() {
        return 0;
    }
    f.rename(&map);
    for block in f.blocks.values_mut() {
        block
            .insts
            .retain(|inst| !matches!(inst, Inst::Core { value, .. } if map.contains_key(value)));
    }
    map.len()
}

fn run(f: &mut Func) -> usize {
    let mut folded = expand(f);
    let known = f.definitions();
    for block in f.blocks.values_mut() {
        for inst in &mut block.insts {
            let Inst::Core { op, .. } = inst else {
                continue;
            };
            let constant = match *op {
                Op::Pack64(a, b) => match (known[a.0], known[b.0]) {
                    (Some(Op::Const(_, lo)), Some(Op::Const(_, hi))) => {
                        Some(Op::Const(Ty::I64, (lo & 0xffff_ffff) | (hi << 32)))
                    }
                    _ => None,
                },
                Op::UnpackLo(a) => match known[a.0] {
                    Some(Op::Const(_, bits)) => Some(Op::Const(Ty::I32, bits & 0xffff_ffff)),
                    _ => None,
                },
                Op::UnpackHi(a) => match known[a.0] {
                    Some(Op::Const(_, bits)) => Some(Op::Const(Ty::I32, bits >> 32)),
                    _ => None,
                },
                _ => None,
            };
            if let Some(constant) = constant {
                *op = constant;
                folded += 1;
            }
        }
    }
    let defs = f.definitions();
    let mut map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            let Inst::Core { value, ty, op } = inst else {
                continue;
            };
            let replacement = match *op {
                Op::Pack64(a, b) => match (defs[a.0], defs[b.0]) {
                    (Some(Op::UnpackLo(x)), Some(Op::UnpackHi(y)))
                        if x == y && f.types[x.0] == *ty =>
                    {
                        Some(x)
                    }
                    _ => None,
                },
                Op::UnpackLo(a) => match defs[a.0] {
                    Some(Op::Pack64(lo, _)) => Some(lo),
                    _ => None,
                },
                Op::UnpackHi(a) => match defs[a.0] {
                    Some(Op::Pack64(_, hi)) => Some(hi),
                    _ => None,
                },
                Op::Convert(Cvt::Bitcast, to, a) => {
                    if f.types[a.0] == to {
                        Some(a)
                    } else if let Some(Op::Convert(Cvt::Bitcast, _, z)) = defs[a.0] {
                        (f.types[z.0] == to).then_some(z)
                    } else {
                        None
                    }
                }
                Op::Select(_, a, b) if a == b => Some(a),
                _ => None,
            };
            if let Some(target) = replacement {
                if target != *value {
                    map.insert(*value, target);
                }
            }
        }
    }
    let mut count = map.len() + folded;
    if !map.is_empty() {
        f.rename(&map);
        for block in f.blocks.values_mut() {
            block.insts.retain(
                |inst| !matches!(inst, Inst::Core { value, .. } if map.contains_key(value)),
            );
        }
    }
    count += distribute_packs(f);
    count += common(f);
    if count != 0 {
        f.compact();
    }
    count
}

fn cancels(mut a: ValueId, mut b: ValueId, defs: &[Option<Op>]) -> bool {
    loop {
        match (defs[a.0], defs[b.0]) {
            (Some(Op::UnpackLo(x)), Some(Op::UnpackHi(y))) => return x == y,
            (Some(Op::Select(c, _, x)), Some(Op::Select(d, _, y))) if c == d => {
                a = x;
                b = y;
            }
            _ => return false,
        }
    }
}

fn distribute_packs(f: &mut Func) -> usize {
    let defs = f.definitions();
    let mut float: BTreeSet<ValueId> = BTreeSet::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            if let Inst::Core {
                op: Op::Convert(Cvt::Bitcast, Ty::F64, a),
                ..
            } = inst
            {
                float.insert(*a);
            }
        }
    }
    let mut count = 0;
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for id in ids {
        let mut insts = Vec::new();
        let old = std::mem::take(&mut f.blocks.get_mut(&id).unwrap().insts);
        for inst in old {
            if let Inst::Core {
                value,
                ty,
                op: Op::Pack64(a, b),
            } = inst
            {
                if let (Some(Op::Select(c, x, y)), Some(Op::Select(d, z, w))) =
                    (defs[a.0], defs[b.0])
                {
                    let loose = |a: ValueId, b: ValueId| {
                        matches!(
                            (defs[a.0], defs[b.0]),
                            (None, None) | (Some(Op::Const(..)), Some(Op::Const(..)))
                        )
                    };
                    if c == d
                        && (float.contains(&value)
                            || cancels(x, z, &defs)
                            || cancels(y, w, &defs)
                            || loose(y, w))
                    {
                        let lo = f.value(ty);
                        let hi = f.value(ty);
                        insts.push(Inst::Core {
                            value: lo,
                            ty,
                            op: Op::Pack64(x, z),
                        });
                        insts.push(Inst::Core {
                            value: hi,
                            ty,
                            op: Op::Pack64(y, w),
                        });
                        insts.push(Inst::Core {
                            value,
                            ty,
                            op: Op::Select(c, lo, hi),
                        });
                        count += 1;
                        continue;
                    }
                }
            }
            insts.push(inst);
        }
        f.blocks.get_mut(&id).unwrap().insts = insts;
    }
    count
}
