use super::super::ir::{*, EffectOp, WaveOp, Cvt, Env, IntOp, Op, Ty, ValueId};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct Masks {
    #[cfg_attr(not(test), allow(dead_code))]
    pub reactivation: Vec<(BlockId, usize)>,
    pub full: Vec<bool>,
    pub guarded: Vec<bool>,
    pub predicated: Vec<Option<(ValueId, ValueId)>>,
    pub masked: Vec<bool>,
    pub masked_result: Vec<Option<ValueId>>,
    pub exposed: Vec<u8>,
    pub chain: BTreeMap<BlockId, Vec<(usize, ValueId)>>,
}

impl Masks {
    pub fn at(&self, block: BlockId, index: usize) -> ValueId {
        let chain = &self.chain[&block];
        chain.iter().rev().find(|(at, _)| *at <= index).map(|(_, v)| *v).unwrap_or(chain[0].1)
    }
}

pub(crate) fn all_active_guard(f: &Func, cond: ValueId, exec: ValueId, defs: &[Option<Op>]) -> bool {
    let any_input = f.blocks.values().flat_map(|b| &b.insts).find_map(|inst| match inst {
        Inst::Packet { op: PacketOp::Any, input, output } if *output == cond => Some(*input),
        Inst::Effect { op: EffectOp::Wave(WaveOp::Any), inputs, outputs, .. } if outputs[0].0 == cond => Some(inputs[0]),
        _ => None,
    });
    let Some(inactive) = any_input else { return false; };
    let Some(Op::Int(IntOp::And, a, b)) = defs[inactive.0] else { return false; };
    let (negated, valid) = if matches!(defs[b.0], Some(Op::Env(Env::ValidLane))) { (a, b) } else { (b, a) };
    if !matches!(defs[valid.0], Some(Op::Env(Env::ValidLane))) { return false; }
    let Some(Op::Int(IntOp::Xor, x, one)) = defs[negated.0] else { return false; };
    let flipped = |v: ValueId| matches!(defs[v.0], Some(Op::Const(Ty::I1, 1)));
    let bit = if flipped(one) { x } else if flipped(x) { one } else { return false; };
    same_bit(bit, exec, defs) || same_bit(exec, bit, defs)
}

fn exec_param(block: &Block, exec_index: usize, f: &Func) -> ValueId {
    let entry = &f.blocks[&f.entry];
    let k = entry.params[..exec_index].iter().filter(|p| p.1 == Ty::I1).count();
    block.params.iter().filter(|p| p.1 == Ty::I1).nth(k).expect("block lacks its EXEC parameter").0
}

fn definitions(f: &Func) -> Vec<Option<Op>> {
    let mut out = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts { if let Inst::Core { value, op, .. } = inst { out[value.0] = Some(*op); } }
    }
    out
}

fn ballot_of(f: &Func) -> BTreeMap<ValueId, ValueId> {
    let mut out = BTreeMap::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Packet { op: PacketOp::Ballot, input, output } => { out.insert(*output, *input); }
                Inst::Effect { op: EffectOp::Wave(WaveOp::Ballot), inputs, outputs, .. } => { out.insert(outputs[0].0, inputs[0]); }
                _ => {}
            }
        }
    }
    out
}

fn same_bit(value: ValueId, exec: ValueId, defs: &[Option<Op>]) -> bool {
    let mut value = value;
    loop {
        if value == exec { return true; }
        match defs[value.0].as_ref() {
            Some(Op::Convert(Cvt::Bitcast, Ty::I1, inner)) => value = *inner,
            _ => return false,
        }
    }
}

fn subset(bit: ValueId, exec: ValueId, defs: &[Option<Op>], ballots: &BTreeMap<ValueId, ValueId>, constants: &[Option<u64>]) -> bool {
    if same_bit(bit, exec, defs) { return true; }
    if constants[bit.0] == Some(0) { return true; }
    match defs[bit.0].as_ref() {
        Some(Op::Int(IntOp::And, a, b)) => same_bit(*a, exec, defs) || same_bit(*b, exec, defs),
        Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => match defs[shifted.0].as_ref() {
            Some(Op::Int(IntOp::LShr, word, lane)) if matches!(defs[lane.0].as_ref(), Some(Op::Env(Env::PacketLaneId))) => {
                if constants[word.0] == Some(0) { return true; }
                match defs[word.0].as_ref() {
                    Some(Op::Int(IntOp::And, a, b)) => [a, b].iter().any(|w| ballots.get(w).is_some_and(|bit| same_bit(*bit, exec, defs))),
                    _ => ballots.get(word).is_some_and(|bit| same_bit(*bit, exec, defs)),
                }
            }
            _ => false,
        },
        _ => false,
    }
}

fn full_word(word: ValueId, old: ValueId, full: &[bool], defs: &[Option<Op>], ballots: &BTreeMap<ValueId, ValueId>, constants: &[Option<u64>], lane_mask: u64) -> bool {
    if let Some(k) = constants[word.0] { return k & lane_mask == lane_mask; }
    if let Some(bit) = ballots.get(&word) { return full[bit.0] || same_bit(*bit, old, defs) && full[old.0]; }
    match defs[word.0].as_ref() {
        Some(Op::Int(IntOp::Or, a, b)) => full_word(*a, old, full, defs, ballots, constants, lane_mask) || full_word(*b, old, full, defs, ballots, constants, lane_mask),
        Some(Op::Int(IntOp::And, a, b)) => full_word(*a, old, full, defs, ballots, constants, lane_mask) && full_word(*b, old, full, defs, ballots, constants, lane_mask),
        Some(Op::Convert(Cvt::Bitcast, _, a)) => full_word(*a, old, full, defs, ballots, constants, lane_mask),
        _ => false,
    }
}

fn full_bit(bit: ValueId, old: ValueId, full: &[bool], defs: &[Option<Op>], ballots: &BTreeMap<ValueId, ValueId>, constants: &[Option<u64>], lane_mask: u64) -> bool {
    if same_bit(bit, old, defs) { return full[old.0]; }
    if let Some(k) = constants[bit.0] { return k & 1 != 0; }
    match defs[bit.0].as_ref() {
        Some(Op::Convert(Cvt::Bitcast, Ty::I1, inner)) => full_bit(*inner, old, full, defs, ballots, constants, lane_mask),
        Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => match defs[shifted.0].as_ref() {
            Some(Op::Int(IntOp::LShr, word, lane)) if matches!(defs[lane.0].as_ref(), Some(Op::Env(Env::PacketLaneId))) =>
                full_word(*word, old, full, defs, ballots, constants, lane_mask),
            _ => false,
        },
        Some(Op::Int(IntOp::And, a, b)) => full_bit(*a, old, full, defs, ballots, constants, lane_mask) && full_bit(*b, old, full, defs, ballots, constants, lane_mask),
        Some(Op::Int(IntOp::Or, a, b)) => full_bit(*a, old, full, defs, ballots, constants, lane_mask) || full_bit(*b, old, full, defs, ballots, constants, lane_mask),
        _ => false,
    }
}

pub(crate) fn analyze(registry: &super::super::dialect::DialectRegistry, f: &Func, exec_index: usize, constants: &[Option<u64>], width: u32, entry_full: bool) -> Masks {
    let every_lane = |op: super::super::dialect::TargetOp| registry.operation(op).is_ok_and(|o| o.effect == super::super::dialect::Effect::ReadGlobal { every_lane: true });
    let defs = definitions(f);
    let ballots = ballot_of(f);
    let lane_mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
    let mut reactivation = Vec::new();
    let mut updates: Vec<(ValueId, ValueId, ValueId)> = Vec::new();
    let mut predicated: Vec<Option<(ValueId, ValueId)>> = vec![None; f.types.len()];
    let mut masked = vec![false; f.types.len()];
    let mut masked_result = vec![None; f.types.len()];
    let mut chain: BTreeMap<BlockId, Vec<(usize, ValueId)>> = BTreeMap::new();
    for (&id, block) in &f.blocks {
        let mut current = exec_param(block, exec_index, f);
        let mut entries = vec![(0usize, current)];
        let query = match &block.term {
            Term::CondBr { cond, .. } => block.insts.iter().find_map(|inst| match inst {
                Inst::Packet { op: PacketOp::Any, input, output } if output == cond => Some(*input),
                _ => None,
            }).filter(|input| !block.term.edges().iter().any(|e| e.args.contains(input))),
            _ => None,
        };
        for (index, inst) in block.insts.iter().enumerate() {
            match inst {
                Inst::Core { value, op: Op::Int(IntOp::And, a, b), .. }
                    if query != Some(*value) && matches!(defs[b.0].as_ref(), Some(Op::Env(Env::ValidLane))) || query != Some(*value) && matches!(defs[a.0].as_ref(), Some(Op::Env(Env::ValidLane))) => {
                    let bit = if matches!(defs[b.0].as_ref(), Some(Op::Env(Env::ValidLane))) { *a } else { *b };
                    if !subset(bit, current, &defs, &ballots, constants) { reactivation.push((id, index)); }
                    updates.push((*value, current, bit));
                    current = *value;
                    entries.push((index + 1, current));
                }
                Inst::Core { value, op: Op::Select(c, new, _), .. } if same_bit(*c, current, &defs) => predicated[value.0] = Some((*new, current)),
                Inst::Core { value, ty: Ty::I1, op: Op::Int(IntOp::And, a, b) } if same_bit(*a, current, &defs) || same_bit(*b, current, &defs) => { masked[value.0] = true; masked_result[value.0] = Some(if same_bit(*b, current, &defs) { *a } else { *b }); }
                _ => {}
            }
        }
        chain.insert(id, entries);
    }
    let mut full = vec![true; f.types.len()];
    let entry_exec = exec_param(&f.blocks[&f.entry], exec_index, f);
    full[entry_exec.0] = entry_full;
    loop {
        let before = full.clone();
        for &(value, old, bit) in &updates {
            full[value.0] = full_bit(bit, old, &full, &defs, &ballots, constants, lane_mask);
        }
        let mut incoming = BTreeMap::from([(entry_exec, entry_full)]);
        for (&id, block) in &f.blocks {
            let outgoing = chain[&id].last().unwrap().1;
            let guarded_edge = match &block.term {
                Term::CondBr { cond, no, .. } if all_active_guard(f, *cond, outgoing, &defs) => Some(no.dst),
                _ => None,
            };
            for edge in block.term.edges() {
                let param = exec_param(&f.blocks[&edge.dst], exec_index, f);
                let fact = full[outgoing.0] || guarded_edge == Some(edge.dst);
                incoming.entry(param).and_modify(|v| *v &= fact).or_insert(fact);
            }
        }
        for (value, fact) in incoming { full[value.0] = fact; }
        if full == before { break; }
    }
    let live = live_across(f, &reactivation, &predicated);
    let internal = exposure(f, &predicated, &masked, &live, false, &every_lane);
    let exposed = exposure(f, &predicated, &masked, &live, true, &every_lane);
    let guarded = predicated.iter().enumerate().map(|(id, p)| p.is_some_and(|(_, exec)| full[exec.0] || internal[id] == 0)).collect();
    Masks { reactivation, full, guarded, predicated, masked, masked_result, exposed, chain }
}

fn exposure(f: &Func, predicated: &[Option<(ValueId, ValueId)>], masked: &[bool], live: &[bool], returns: bool, every_lane: &dyn Fn(super::super::dialect::TargetOp) -> bool) -> Vec<u8> {
    let words = |v: ValueId| if f.types[v.0].bits() == 64 { 3u8 } else { 1u8 };
    let mut producers: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut params: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut incoming: BTreeMap<BlockId, Vec<&Edge>> = BTreeMap::new();
    let mut pending: Vec<(ValueId, u8)> = live.iter().enumerate().filter_map(|(id, &live)| live.then_some((ValueId(id), 3))).collect();
    for (&id, block) in &f.blocks {
        for (index, &(p, _)) in block.params.iter().enumerate() { params[p.0] = Some((id, index)); }
        for (index, inst) in block.insts.iter().enumerate() {
            for v in outputs(inst) { producers[v.0] = Some((id, index)); }
            match inst {
                Inst::Effect { op: EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadLane | WaveOp::WriteLane | WaveOp::BpermuteFi | WaveOp::Wmma), inputs, .. } => pending.extend(inputs.iter().map(|&v| (v, 3))),
                Inst::Packet { input, .. } => pending.push((*input, 3)),
                Inst::Target { op, args, provenance: Some(_), .. } if every_lane(*op) => pending.extend(args.values().iter().map(|&v| (v, 3))),
                _ => {}
            }
        }
        if let (true, Term::Ret(args)) = (returns, &block.term) { pending.extend(args.iter().map(|&v| (v, 3))); }
        for edge in block.term.edges() { incoming.entry(edge.dst).or_default().push(edge); }
    }
    let mut exposed = vec![0u8; f.types.len()];
    while let Some((v, mask)) = pending.pop() {
        let added = mask & words(v) & !exposed[v.0];
        if added == 0 { continue; }
        exposed[v.0] |= added;
        if let Some((block, index)) = producers[v.0] {
            match &f.blocks[&block].insts[index] {
                Inst::Core { op: Op::Select(_, _, old), .. } if predicated[v.0].is_some() => pending.push((*old, added)),
                Inst::Core { op: Op::Int(..), .. } if masked[v.0] => {}
                Inst::Core { op: Op::UnpackLo(x), .. } => pending.push((*x, 1)),
                Inst::Core { op: Op::UnpackHi(x), .. } => pending.push((*x, 2)),
                Inst::Core { op: Op::Pack64(lo, hi), .. } => {
                    if added & 1 != 0 { pending.push((*lo, 1)); }
                    if added & 2 != 0 { pending.push((*hi, 1)); }
                }
                Inst::Core { op: Op::Convert(Cvt::Bitcast, _, a), .. } => pending.push((*a, added)),
                Inst::Core { op: Op::Convert(Cvt::Trunc | Cvt::ZExt | Cvt::SExt, _, a), .. } => pending.push((*a, 1)),
                Inst::Core { op, .. } => { op.map(|a| { pending.push((a, 3)); a }); }
                Inst::Target { args, .. } => pending.extend(args.values().iter().map(|&a| (a, 3))),
                Inst::Packet { input, .. } => pending.push((*input, 3)),
                Inst::Effect { .. } => {}
            }
        } else if let Some((block, index)) = params[v.0] {
            for edge in incoming.get(&block).into_iter().flatten() { pending.push((edge.args[index], added)); }
        }
    }
    exposed
}

fn uses(inst: &Inst) -> Vec<ValueId> {
    match inst {
        Inst::Core { op, .. } => { let mut out = Vec::new(); op.map(|v| { out.push(v); v }); out }
        Inst::Packet { input, .. } => vec![*input],
        Inst::Target { args, .. } => args.values().to_vec(),
        Inst::Effect { inputs, .. } => inputs.clone(),
    }
}

fn outputs(inst: &Inst) -> Vec<ValueId> {
    match inst {
        Inst::Core { value, .. } | Inst::Packet { output: value, .. } => vec![*value],
        Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => outputs.iter().map(|o| o.0).collect(),
    }
}

fn reads(inst: &Inst, predicated: &[Option<(ValueId, ValueId)>]) -> Vec<ValueId> {
    match inst {
        Inst::Core { value, op: Op::Select(c, new, _), .. } if predicated[value.0].is_some() => vec![*c, *new],
        _ => uses(inst),
    }
}

fn needed(f: &Func, predicated: &[Option<(ValueId, ValueId)>]) -> Vec<bool> {
    let mut params: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut producers: Vec<Option<(BlockId, usize)>> = vec![None; f.types.len()];
    let mut incoming: BTreeMap<BlockId, Vec<&Edge>> = BTreeMap::new();
    let mut pending = Vec::new();
    for (&id, block) in &f.blocks {
        for (index, &(p, _)) in block.params.iter().enumerate() { params[p.0] = Some((id, index)); }
        for (index, inst) in block.insts.iter().enumerate() {
            match inst {
                Inst::Core { value, .. } | Inst::Packet { output: value, .. } => producers[value.0] = Some((id, index)),
                Inst::Effect { inputs, outputs, .. } => {
                    pending.extend(inputs.iter().copied());
                    for &(v, _) in outputs { producers[v.0] = Some((id, index)); }
                }
                Inst::Target { args, outputs, .. } => {
                    pending.extend(args.values().iter().copied());
                    for &(v, _) in outputs { producers[v.0] = Some((id, index)); }
                }
            }
        }
        match &block.term { Term::CondBr { cond, .. } => pending.push(*cond), Term::Ret(args) => pending.extend(args.iter().copied()), Term::Br(_) => {} }
        for edge in block.term.edges() { incoming.entry(edge.dst).or_default().push(edge); }
    }
    let mut needed = vec![false; f.types.len()];
    while let Some(v) = pending.pop() {
        if needed[v.0] { continue; }
        needed[v.0] = true;
        if let Some((block, index)) = producers[v.0] {
            if let inst @ (Inst::Core { .. } | Inst::Packet { .. }) = &f.blocks[&block].insts[index] { pending.extend(reads(inst, predicated)); }
        } else if let Some((block, index)) = params[v.0] {
            for edge in incoming.get(&block).into_iter().flatten() { pending.push(edge.args[index]); }
        }
    }
    needed
}

fn active(inst: &Inst, needed: &[bool]) -> bool {
    match inst {
        Inst::Core { value, .. } | Inst::Packet { output: value, .. } => needed[value.0],
        Inst::Effect { .. } | Inst::Target { .. } => true,
    }
}

fn live_across(f: &Func, points: &[(BlockId, usize)], predicated: &[Option<(ValueId, ValueId)>]) -> Vec<bool> {
    let needed = needed(f, predicated);
    let mut live_in: BTreeMap<BlockId, BTreeSet<ValueId>> = f.blocks.keys().map(|&id| (id, BTreeSet::new())).collect();
    loop {
        let mut changed = false;
        for (&id, block) in &f.blocks {
            let mut live = BTreeSet::new();
            for edge in block.term.edges() {
                for (&arg, &(param, _)) in edge.args.iter().zip(&f.blocks[&edge.dst].params) {
                    if needed[param.0] && live_in[&edge.dst].contains(&param) { live.insert(arg); }
                }
            }
            match &block.term { Term::CondBr { cond, .. } => { live.insert(*cond); } Term::Ret(args) => live.extend(args.iter().copied()), Term::Br(_) => {} }
            for inst in block.insts.iter().rev() {
                for v in outputs(inst) { live.remove(&v); }
                if active(inst, &needed) { for v in reads(inst, predicated) { live.insert(v); } }
            }
            if live != live_in[&id] { live_in.insert(id, live); changed = true; }
        }
        if !changed { break; }
    }
    let mut across = vec![false; f.types.len()];
    for &(id, index) in points {
        let block = &f.blocks[&id];
        let mut live = BTreeSet::new();
        for edge in block.term.edges() {
            for (&arg, &(param, _)) in edge.args.iter().zip(&f.blocks[&edge.dst].params) {
                if needed[param.0] && live_in[&edge.dst].contains(&param) { live.insert(arg); }
            }
        }
        match &block.term { Term::CondBr { cond, .. } => { live.insert(*cond); } Term::Ret(args) => live.extend(args.iter().copied()), Term::Br(_) => {} }
        for inst in block.insts[index + 1..].iter().rev() {
            for v in outputs(inst) { live.remove(&v); }
            if active(inst, &needed) { for v in reads(inst, predicated) { live.insert(v); } }
        }
        let defined_after: BTreeSet<_> = block.insts[index..].iter().flat_map(outputs).collect();
        for v in live { if !defined_after.contains(&v) { across[v.0] = true; } }
    }
    across
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn narrowing_exec_keeps_predicated_writes_guarded_and_widening_does_not() {
        let mut f = Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] };
        let exec = f.value(Ty::I1); let x = f.value(Ty::I32); let saved = f.value(Ty::I32);
        let mut insts = Vec::new();
        let mut core = |f: &mut Func, ty, op| { let v = f.value(ty); insts.push(Inst::Core { value: v, ty, op }); v };
        let valid = core(&mut f, Ty::I1, Op::Env(Env::ValidLane));
        let pred = core(&mut f, Ty::I1, Op::Convert(Cvt::Bitcast, Ty::I1, exec));
        let first = core(&mut f, Ty::I32, Op::Select(pred, x, x));
        let cmp = core(&mut f, Ty::I1, Op::Int(IntOp::And, exec, valid));
        let narrowed = core(&mut f, Ty::I1, Op::Int(IntOp::And, cmp, valid));
        let pred2 = core(&mut f, Ty::I1, Op::Convert(Cvt::Bitcast, Ty::I1, narrowed));
        let second = core(&mut f, Ty::I32, Op::Select(pred2, x, first));
        let lane = core(&mut f, Ty::I32, Op::Env(Env::PacketLaneId));
        let shifted = core(&mut f, Ty::I32, Op::Int(IntOp::LShr, saved, lane));
        let restored_bit = core(&mut f, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let restored = core(&mut f, Ty::I1, Op::Int(IntOp::And, restored_bit, valid));
        let pred3 = core(&mut f, Ty::I1, Op::Convert(Cvt::Bitcast, Ty::I1, restored));
        let third = core(&mut f, Ty::I32, Op::Select(pred3, x, second));
        let sum = core(&mut f, Ty::I32, Op::Int(IntOp::Add, first, third));
        f.blocks.insert(BlockId(0), Block { params: vec![(exec, Ty::I1), (x, Ty::I32), (saved, Ty::I32)], insts, term: Term::Br(Edge { dst: BlockId(1), args: vec![restored, sum, second] }) });
        let e1 = f.value(Ty::I1); let p = f.value(Ty::I32); let q = f.value(Ty::I32); let always = f.value(Ty::I1);
        let semantics = super::super::super::ir::MemorySemantics { scope: super::super::super::ir::Scope::WorkItem,
            ordering: super::super::super::ir::Ordering::Relaxed, cache_policy: super::super::super::ir::CachePolicy::Temporal, volatile: false, deferred_scope: false };
        f.blocks.insert(BlockId(1), Block { params: vec![(e1, Ty::I1), (p, Ty::I32), (q, Ty::I32)], insts: vec![
            Inst::Core { value: always, ty: Ty::I1, op: Op::Const(Ty::I1, 1) },
            Inst::Effect { provenance: 0, op: EffectOp::Memory { space: super::super::super::ir::Space::Lds, op: super::super::super::ir::MemoryOp::Store(super::super::super::ir::MemSize::B32), semantics }, inputs: vec![q, p, always], outputs: vec![] },
        ], term: Term::Ret(vec![]) });
        let constants = super::super::constants(&f);
        let masks = analyze(&crate::rdna_spmd::targets::rdna4::registry(), &f, 0, &constants, 16, false);
        assert_eq!(masks.reactivation, vec![(BlockId(0), 10)]);
        assert!(!masks.guarded[first.0]);
        assert!(!masks.guarded[second.0]);
        assert!(masks.guarded[third.0]);
        let masks = analyze(&crate::rdna_spmd::targets::rdna4::registry(), &f, 0, &constants, 16, true);
        assert!(masks.full[exec.0]);
        assert!(masks.full[cmp.0]);
        assert!(!masks.full[restored.0]);
        assert!(masks.guarded[first.0]);
        assert!(masks.guarded[second.0]);
        assert!(masks.guarded[third.0]);
    }
}

pub(crate) struct Exec {
    chain: BTreeMap<BlockId, Vec<(usize, ValueId)>>,
    nonempty: Vec<bool>,
    active: Vec<bool>,
}

impl Exec {
    pub fn at(&self, block: BlockId, index: usize) -> ValueId {
        let chain = &self.chain[&block];
        chain.iter().rev().find(|(at, _)| *at <= index).map(|(_, v)| *v).unwrap_or(chain[0].1)
    }
    pub fn nonempty_at(&self, block: BlockId, index: usize) -> bool { self.nonempty[self.at(block, index).0] }
    pub fn active_at(&self, block: BlockId, index: usize) -> bool { self.active[self.at(block, index).0] }
}

enum Word { Constant, Ballot(ValueId), Or(ValueId, ValueId), And, Other }

fn word_of(value: ValueId, defs: &[Option<Op>], ballots: &BTreeMap<ValueId, ValueId>, constants: &[Option<u64>]) -> Word {
    if constants[value.0].is_some() { return Word::Constant; }
    if let Some(bit) = ballots.get(&value) { return Word::Ballot(*bit); }
    match defs[value.0].as_ref() {
        Some(Op::Int(IntOp::Or, a, b)) => Word::Or(*a, *b),
        Some(Op::Int(IntOp::And, ..)) => Word::And,
        Some(Op::Convert(Cvt::Bitcast, _, a)) => word_of(*a, defs, ballots, constants),
        _ => Word::Other,
    }
}

enum Update { Copy(ValueId), Constant(u64), Word(ValueId), Bit, Unknown }

fn update_of(bit: ValueId, defs: &[Option<Op>], constants: &[Option<u64>]) -> Update {
    if let Some(k) = constants[bit.0] { return Update::Constant(if k & 1 != 0 { u64::MAX } else { 0 }); }
    match defs[bit.0].as_ref() {
        Some(Op::Convert(Cvt::Bitcast, Ty::I1, inner)) => update_of(*inner, defs, constants),
        Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => match defs[shifted.0].as_ref() {
            Some(Op::Int(IntOp::LShr, word, lane)) if matches!(defs[lane.0].as_ref(), Some(Op::Env(Env::PacketLaneId))) => {
                if let Some(k) = constants[word.0] { Update::Constant(k) } else { Update::Word(*word) }
            }
            _ => Update::Unknown,
        },
        Some(Op::Int(IntOp::And, ..)) => Update::Bit,
        _ => if defs[bit.0].is_some() { Update::Unknown } else { Update::Copy(bit) },
    }
}

pub(crate) fn exec(f: &Func, exec_index: usize, constants: &[Option<u64>], width: u32, initial: bool, packed: bool) -> Exec {
    let defs = definitions(f);
    let ballots = ballot_of(f);
    let mut chain: BTreeMap<BlockId, Vec<(usize, ValueId)>> = BTreeMap::new();
    let mut updates: Vec<(ValueId, ValueId, Update)> = Vec::new();
    let mut barrier = BTreeSet::new();
    for (&id, block) in &f.blocks {
        let mut current = exec_param(block, exec_index, f);
        let mut entries = vec![(0usize, current)];
        for (index, inst) in block.insts.iter().enumerate() {
            match inst {
                Inst::Core { value, op: Op::Int(IntOp::And, a, b), .. }
                    if matches!(defs[b.0].as_ref(), Some(Op::Env(Env::ValidLane))) || matches!(defs[a.0].as_ref(), Some(Op::Env(Env::ValidLane))) => {
                    let bit = if matches!(defs[b.0].as_ref(), Some(Op::Env(Env::ValidLane))) { *a } else { *b };
                    updates.push((*value, current, update_of(bit, &defs, constants)));
                    current = *value;
                    entries.push((index + 1, current));
                }
                Inst::Effect { op: EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait, .. } => { barrier.insert(id); }
                _ => {}
            }
        }
        chain.insert(id, entries);
    }
    let lane_mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
    let mut nonempty = vec![true; f.types.len()];
    let mut active = vec![true; f.types.len()];
    let mut saved = vec![true; f.types.len()];
    let entry_exec = exec_param(&f.blocks[&f.entry], exec_index, f);
    let mut any_of: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Packet { op: PacketOp::Any, input, output } => { any_of.insert(*output, *input); }
                Inst::Effect { op: EffectOp::Wave(WaveOp::Any), inputs, outputs, .. } => { any_of.insert(outputs[0].0, inputs[0]); }
                _ => {}
            }
        }
    }
    loop {
        let mut changed = false;
        let mut set = |table: &mut Vec<bool>, id: ValueId, fact: bool| { if table[id.0] != fact { table[id.0] = fact; changed = true; } };
        for block in f.blocks.values() {
            for inst in &block.insts {
                match inst {
                    Inst::Packet { op: PacketOp::Ballot, input, output } => set(&mut saved, *output, active[input.0]),
                    Inst::Effect { op: EffectOp::Wave(WaveOp::Ballot), inputs, outputs, .. } => set(&mut saved, outputs[0].0, active[inputs[0].0]),
                    Inst::Core { value, op, .. } => {
                        let fact = match op {
                            Op::Convert(Cvt::Bitcast, _, a) => saved[a.0],
                            _ => false,
                        };
                        set(&mut saved, *value, fact);
                    }
                    _ => {}
                }
            }
        }
        for &(value, old, ref update) in &updates {
            let (n, a) = match update {
                Update::Copy(v) => (nonempty[v.0], active[v.0]),
                Update::Constant(k) => (initial && k & lane_mask != 0, k & 1 != 0),
                Update::Word(word) => match word_of(*word, &defs, &ballots, constants) {
                    Word::Or(x, y) => {
                        let reads_old = |w: ValueId| matches!(word_of(w, &defs, &ballots, constants), Word::Ballot(bit) if same_bit(bit, old, &defs));
                        let preserves = reads_old(x) || reads_old(y);
                        (preserves && nonempty[old.0], (reads_old(x) && active[old.0]) || (reads_old(y) && active[old.0]) || saved[x.0] || saved[y.0])
                    }
                    Word::Ballot(bit) => (nonempty[bit.0], active[bit.0]),
                    _ => (false, false),
                },
                Update::Bit | Update::Unknown => (false, false),
            };
            set(&mut nonempty, value, n);
            set(&mut active, value, a);
        }
        let mut incoming_nonempty: Vec<Option<bool>> = vec![None; f.types.len()];
        let mut incoming_active: Vec<Option<bool>> = vec![None; f.types.len()];
        let mut incoming_saved: Vec<Option<bool>> = vec![None; f.types.len()];
        incoming_nonempty[entry_exec.0] = Some(initial);
        incoming_active[entry_exec.0] = Some(true);
        for &(id, _) in &f.blocks[&f.entry].params { incoming_saved[id.0] = Some(false); }
        let meet = |table: &mut Vec<Option<bool>>, id: ValueId, fact: bool| { table[id.0] = Some(table[id.0].unwrap_or(true) & fact); };
        for (&id, block) in &f.blocks {
            let outgoing = chain[&id].last().unwrap().1;
            let pins: Vec<(BlockId, bool)> = match &block.term {
                Term::CondBr { cond, yes, no } => {
                    let (query, negated) = match defs[cond.0].as_ref() {
                        Some(Op::Cmp(super::super::ir::IntPred::Eq, q, zero)) if constants[zero.0] == Some(0) => (*q, true),
                        _ => (*cond, false),
                    };
                    match any_of.get(&query).copied() {
                        Some(bit) if same_bit(bit, outgoing, &defs) || same_bit(outgoing, bit, &defs) => vec![(yes.dst, !negated), (no.dst, negated)],
                        _ if !negated && all_active_guard(f, *cond, outgoing, &defs) => vec![(no.dst, true)],
                        _ => vec![],
                    }
                }
                _ => vec![],
            };
            for edge in block.term.edges() {
                let pinned = pins.iter().find(|(dst, _)| *dst == edge.dst).map(|(_, v)| *v);
                let param = exec_param(&f.blocks[&edge.dst], exec_index, f);
                let fact = if barrier.contains(&id) { false } else { pinned.unwrap_or(nonempty[outgoing.0]) };
                meet(&mut incoming_nonempty, param, fact);
                if !packed && (edge.dst <= id || pinned == Some(false)) { continue; }
                let fact = pinned.unwrap_or(active[outgoing.0]);
                meet(&mut incoming_active, param, fact);
                for (&arg, &(p, _)) in edge.args.iter().zip(&f.blocks[&edge.dst].params) {
                    meet(&mut incoming_saved, p, saved[arg.0]);
                }
            }
        }
        for (value, fact) in incoming_nonempty.iter().enumerate() { if let Some(fact) = *fact { set(&mut nonempty, ValueId(value), fact); } }
        for (value, fact) in incoming_active.iter().enumerate() { if let Some(fact) = *fact { set(&mut active, ValueId(value), fact); } }
        for (value, fact) in incoming_saved.iter().enumerate() { if let Some(fact) = *fact { set(&mut saved, ValueId(value), fact); } }
        if !changed { break; }
    }
    Exec { chain, nonempty, active }
}

