use super::super::host::Vectors;
use super::super::ir::{BlockId, Cvt, Func, Inst, IntOp, Op, Term, Ty, ValueId};
use std::collections::{BTreeMap, BTreeSet};

fn loop_depths(f: &Func) -> BTreeMap<BlockId, u32> {
    let mut state: BTreeMap<BlockId, u8> = BTreeMap::new();
    let mut stack: Vec<(BlockId, usize)> = vec![(f.entry, 0)];
    state.insert(f.entry, 1);
    let mut back: Vec<(BlockId, BlockId)> = Vec::new();
    while let Some((id, next)) = stack.last_mut() {
        let edges: Vec<BlockId> = f.blocks[id].term.edges().map(|e| e.dst).collect();
        if *next < edges.len() {
            let dst = edges[*next];
            *next += 1;
            match state.get(&dst) {
                Some(1) => back.push((*id, dst)),
                Some(_) => {}
                None => { state.insert(dst, 1); stack.push((dst, 0)); }
            }
        } else { state.insert(*id, 2); stack.pop(); }
    }
    let mut preds: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
    for (&id, block) in &f.blocks { for e in block.term.edges() { preds.entry(e.dst).or_default().push(id); } }
    let mut depth: BTreeMap<BlockId, u32> = f.blocks.keys().map(|&b| (b, 0)).collect();
    for (latch, header) in back {
        let mut members = BTreeSet::new();
        members.insert(header);
        let mut work = vec![latch];
        while let Some(x) = work.pop() {
            if !members.insert(x) { continue; }
            if let Some(ps) = preds.get(&x) { for &p in ps { if !members.contains(&p) { work.push(p); } } }
        }
        for m in members { *depth.get_mut(&m).unwrap() += 1; }
    }
    depth
}

pub(crate) fn costs(f: &Func, width: u32, host: &Vectors) -> (f64, f64) {
    let ops = |bits: u32| ((width * bits + host.bits - 1) / host.bits).max(1) as f64;
    let vector = width * 64 >= host.scalar_below();
    let (narrow_pack, narrow_unpack, narrow_logic, narrow_regs) =
        if !vector { (0.0, 0.0, 1.0, 1.0) } else if host.native_masks { (0.0, 0.0, 1.0, 0.0) } else { (2.0 * ops(64), 2.0 * ops(64), ops(32), ops(32)) };
    let (wide_widen, wide_pack, wide_logic, wide_regs) =
        if !vector { (0.0, 0.0, 1.0, 1.0) } else if host.native_masks { (ops(64), ops(64), ops(64), ops(64)) } else { (ops(64), 2.0 * ops(64), ops(64), ops(64)) };
    let depths = loop_depths(f);
    let mut produced: Vec<Option<u32>> = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            if let Inst::Core { value, op: Op::Cmp(_, a, _) | Op::FCmp(_, a, _), .. } = inst { produced[value.0] = Some(f.types[a.0].bits()); }
        }
    }
    let (mut narrow, mut wide) = (0.0f64, 0.0f64);
    for (&id, block) in &f.blocks {
        let weight = 31f64.powi(depths.get(&id).copied().unwrap_or(0).min(4) as i32);
        let mut direct: BTreeMap<ValueId, usize> = BTreeMap::new();
        let mut selects64: BTreeSet<ValueId> = BTreeSet::new();
        let mut selects32: BTreeSet<ValueId> = BTreeSet::new();
        let mut logic = 0usize;
        let mut queries = 0usize;
        for inst in &block.insts {
            match inst {
                Inst::Core { ty, op: Op::Select(c, ..), .. } if f.types[c.0] == Ty::I1 => {
                    if ty.bits() == 64 { selects64.insert(*c); } else { selects32.insert(*c); }
                    *direct.entry(*c).or_default() += 1;
                }
                Inst::Core { ty: Ty::I1, op: Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, ..) | Op::Select(..), .. } => logic += 1,
                Inst::Core { ty: Ty::I1, op: Op::Convert(Cvt::Trunc, ..), .. } => {}
                Inst::Packet { .. } => queries += 1,
                _ => {}
            }
        }
        let mut uses: BTreeMap<ValueId, usize> = BTreeMap::new();
        for inst in &block.insts { super::super::pass::dce::operands(inst, |v| *uses.entry(v).or_default() += 1); }
        for e in block.term.edges() { for &v in &e.args { *uses.entry(v).or_default() += 1; } }
        if let Term::CondBr { cond, .. } = &block.term { *uses.entry(*cond).or_default() += 1; }
        let (mut n, mut w) = (0.0f64, 0.0f64);
        for (&v, &count) in &uses {
            let Some(bits) = produced[v.0] else { continue };
            let same = direct.get(&v).copied().unwrap_or(0);
            let escapes = count > same || (bits == 64 && selects32.contains(&v)) || (bits == 32 && selects64.contains(&v));
            if !escapes { continue; }
            if bits == 64 { n += narrow_pack; } else { w += wide_widen; }
        }
        n += selects64.len() as f64 * narrow_unpack;
        w += selects32.len() as f64 * wide_pack;
        n += (logic + queries) as f64 * narrow_logic;
        w += (logic + queries) as f64 * wide_logic;
        let carried = block.params.iter().filter(|p| p.1 == Ty::I1).count() as f64;
        w += carried * (wide_regs - narrow_regs) * 2.0;
        narrow += weight * n;
        wide += weight * w;
    }
    (narrow, wide)
}
