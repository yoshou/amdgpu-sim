use super::super::ir::{*, Cvt, Op, Ty, ValueId};
use std::collections::{BTreeMap, BTreeSet};

fn definitions(f: &Func) -> Vec<Option<Op>> {
    let mut out = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts { if let Inst::Core { value, op, .. } = inst { out[value.0] = Some(*op); } }
    }
    out
}

pub(crate) fn rename(f: &mut Func, map: &BTreeMap<ValueId, ValueId>) {
    let m = |v: ValueId| { let mut v = v; while let Some(&next) = map.get(&v) { v = next; } v };
    for block in f.blocks.values_mut() {
        for inst in &mut block.insts {
            match inst {
                Inst::Core { op, .. } => *op = op.map(m),
                Inst::Packet { input, .. } => *input = m(*input),
                Inst::Target { args, .. } => *args = args.map(m),
                Inst::Effect { inputs, .. } => for v in inputs { *v = m(*v); },
            }
        }
        match &mut block.term {
            Term::Br(e) => for v in &mut e.args { *v = m(*v); },
            Term::CondBr { cond, yes, no } => { *cond = m(*cond); for v in yes.args.iter_mut().chain(&mut no.args) { *v = m(*v); } }
            Term::Ret(args) => for v in args { *v = m(*v); },
        }
    }
}

pub(crate) fn run(f: &mut Func) -> usize {
    let defs = definitions(f);
    let mut map: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            let Inst::Core { value, ty, op } = inst else { continue; };
            let replacement = match *op {
                Op::Pack64(a, b) => match (defs[a.0], defs[b.0]) {
                    (Some(Op::UnpackLo(x)), Some(Op::UnpackHi(y))) if x == y && f.types[x.0] == *ty => Some(x),
                    _ => None,
                },
                Op::UnpackLo(a) => match defs[a.0] { Some(Op::Pack64(lo, _)) => Some(lo), _ => None },
                Op::UnpackHi(a) => match defs[a.0] { Some(Op::Pack64(_, hi)) => Some(hi), _ => None },
                Op::Convert(Cvt::Bitcast, to, a) => {
                    if f.types[a.0] == to { Some(a) }
                    else if let Some(Op::Convert(Cvt::Bitcast, _, z)) = defs[a.0] { (f.types[z.0] == to).then_some(z) }
                    else { None }
                }
                Op::Select(_, a, b) if a == b => Some(a),
                _ => None,
            };
            if let Some(target) = replacement {
                if target != *value { map.insert(*value, target); }
            }
        }
    }
    let mut count = map.len();
    if !map.is_empty() {
        rename(f, &map);
        for block in f.blocks.values_mut() {
            block.insts.retain(|inst| !matches!(inst, Inst::Core { value, .. } if map.contains_key(value)));
        }
    }
    count += distribute_packs(f);
    if count != 0 { super::compact(f); }
    count
}

fn cancels(mut a: ValueId, mut b: ValueId, defs: &[Option<Op>]) -> bool {
    loop {
        match (defs[a.0], defs[b.0]) {
            (Some(Op::UnpackLo(x)), Some(Op::UnpackHi(y))) => return x == y,
            (Some(Op::Select(c, _, x)), Some(Op::Select(d, _, y))) if c == d => { a = x; b = y; }
            _ => return false,
        }
    }
}

fn distribute_packs(f: &mut Func) -> usize {
    let defs = definitions(f);
    let mut float: BTreeSet<ValueId> = BTreeSet::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            if let Inst::Core { op: Op::Convert(Cvt::Bitcast, Ty::F64, a), .. } = inst { float.insert(*a); }
        }
    }
    let mut count = 0;
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    for id in ids {
        let mut insts = Vec::new();
        let old = std::mem::take(&mut f.blocks.get_mut(&id).unwrap().insts);
        for inst in old {
            if let Inst::Core { value, ty, op: Op::Pack64(a, b) } = inst {
                if let (Some(Op::Select(c, x, y)), Some(Op::Select(d, z, w))) = (defs[a.0], defs[b.0]) {
                    let loose = |a: ValueId, b: ValueId| matches!((defs[a.0], defs[b.0]), (None, None) | (Some(Op::Const(..)), Some(Op::Const(..))));
                    if c == d && (float.contains(&value) || cancels(x, z, &defs) || cancels(y, w, &defs) || loose(y, w)) {
                        let lo = f.value(ty);
                        let hi = f.value(ty);
                        insts.push(Inst::Core { value: lo, ty, op: Op::Pack64(x, z) });
                        insts.push(Inst::Core { value: hi, ty, op: Op::Pack64(y, w) });
                        insts.push(Inst::Core { value, ty, op: Op::Select(c, lo, hi) });
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
