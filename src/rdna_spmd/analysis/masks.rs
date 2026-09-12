use super::super::ir::{*, EffectOp, WaveOp, Cvt, Env, IntOp, Op, Ty, ValueId};
use super::{Analyses, Analysis, Constants};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

#[derive(PartialEq)]
pub(crate) struct Masks {
    #[cfg_attr(not(test), allow(dead_code))]
    pub reactivation: Rc<Vec<(BlockId, usize)>>,
    pub full: Vec<bool>,
    pub guarded: Vec<bool>,
    pub predicated: Rc<Vec<Option<(ValueId, ValueId)>>>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub masked: Rc<Vec<bool>>,
    pub exposed: Vec<u8>,
    pub chain: Rc<BTreeMap<BlockId, Vec<(usize, ValueId)>>>,
}

impl Masks {
    pub fn at(&self, block: BlockId, index: usize) -> ValueId {
        let chain = &self.chain[&block];
        chain.iter().rev().find(|(at, _)| *at <= index).map(|(_, v)| *v).unwrap_or(chain[0].1)
    }
}

impl Analysis for Predication {
    type Result = Predication;
    const NAME: &'static str = "predication";
    fn compute(f: &Func, analyses: &Analyses) -> Self {
        let constants = analyses.get::<Constants>(f);
        predication(f, analyses.context().exec_index, &constants)
    }
}

impl Analysis for Masks {
    type Result = Masks;
    const NAME: &'static str = "masks";
    fn compute(f: &Func, analyses: &Analyses) -> Self {
        let ctx = analyses.context();
        let (predication, constants) = (analyses.get::<Predication>(f), analyses.get::<Constants>(f));
        analyze_from(ctx.registry, f, &predication, ctx.exec_index, &constants, ctx.lanes, ctx.entry_full)
    }
}

impl Analysis for Exec {
    type Result = Exec;
    const NAME: &'static str = "exec";
    fn compute(f: &Func, analyses: &Analyses) -> Self {
        let ctx = analyses.context();
        let constants = analyses.get::<Constants>(f);
        exec(f, ctx.exec_index, &constants, ctx.lanes, ctx.exec_initial, ctx.lanes > 1)
    }
}

fn any_of(f: &Func) -> Vec<Option<ValueId>> {
    let mut out = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Packet { op: PacketOp::Any, input, output } => out[output.0] = Some(*input),
                Inst::Effect { op: EffectOp::Wave(WaveOp::Any), inputs, outputs, .. } => out[outputs[0].0 .0] = Some(inputs[0]),
                _ => {}
            }
        }
    }
    out
}

pub(crate) fn all_active_guard(any: &[Option<ValueId>], cond: ValueId, exec: ValueId, defs: &[Option<Op>]) -> bool {
    let Some(inactive) = any[cond.0] else { return false; };
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

fn ballot_of(f: &Func) -> Vec<Option<ValueId>> {
    let mut out = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Packet { op: PacketOp::Ballot, input, output } => out[output.0] = Some(*input),
                Inst::Effect { op: EffectOp::Wave(WaveOp::Ballot), inputs, outputs, .. } => out[outputs[0].0 .0] = Some(inputs[0]),
                _ => {}
            }
        }
    }
    out
}

fn block_index(f: &Func) -> Vec<usize> {
    let mut out = vec![usize::MAX; f.blocks.keys().map(|b| b.0 + 1).max().unwrap_or(0)];
    for (index, id) in f.blocks.keys().enumerate() { out[id.0] = index; }
    out
}

struct Layout<'f> {
    blocks: Vec<&'f Block>,
    index: Vec<usize>,
    incoming: Vec<Vec<&'f Edge>>,
    producers: Vec<Option<(usize, usize)>>,
    params: Vec<Option<(usize, usize)>>,
}

fn layout(f: &Func) -> Layout<'_> {
    let index = block_index(f);
    let blocks: Vec<&Block> = f.blocks.values().collect();
    let incoming = incoming_edges(f, &index);
    let mut producers: Vec<Option<(usize, usize)>> = vec![None; f.types.len()];
    let mut params: Vec<Option<(usize, usize)>> = vec![None; f.types.len()];
    for (at, block) in blocks.iter().enumerate() {
        for (position, &(p, _)) in block.params.iter().enumerate() { params[p.0] = Some((at, position)); }
        for (position, inst) in block.insts.iter().enumerate() {
            for_each_output(inst, |v| producers[v.0] = Some((at, position)));
        }
    }
    Layout { blocks, index, incoming, producers, params }
}

fn incoming_edges<'f>(f: &'f Func, index: &[usize]) -> Vec<Vec<&'f Edge>> {
    let mut out: Vec<Vec<&Edge>> = (0..f.blocks.len()).map(|_| Vec::new()).collect();
    for block in f.blocks.values() {
        for edge in block.term.edges() { out[index[edge.dst.0]].push(edge); }
    }
    out
}

struct Bits(Vec<u64>);
impl Bits {
    fn new(values: usize) -> Self { Self(vec![0; values.div_ceil(64)]) }
    fn clear(&mut self) { self.0.fill(0); }
    fn insert(&mut self, v: ValueId) { self.0[v.0 / 64] |= 1 << (v.0 % 64); }
    fn remove(&mut self, v: ValueId) { self.0[v.0 / 64] &= !(1 << (v.0 % 64)); }
    fn contains(&self, v: ValueId) -> bool { self.0[v.0 / 64] >> (v.0 % 64) & 1 != 0 }
    fn for_each(&self, mut f: impl FnMut(ValueId)) {
        for (w, &word) in self.0.iter().enumerate() {
            let mut bits = word;
            while bits != 0 { let b = bits.trailing_zeros() as usize; bits &= bits - 1; f(ValueId(w * 64 + b)); }
        }
    }
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

fn subset(bit: ValueId, exec: ValueId, defs: &[Option<Op>], ballots: &[Option<ValueId>], constants: &[Option<u64>]) -> bool {
    if same_bit(bit, exec, defs) { return true; }
    if constants[bit.0] == Some(0) { return true; }
    match defs[bit.0].as_ref() {
        Some(Op::Int(IntOp::And, a, b)) => same_bit(*a, exec, defs) || same_bit(*b, exec, defs),
        Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => match defs[shifted.0].as_ref() {
            Some(Op::Int(IntOp::LShr, word, lane)) if matches!(defs[lane.0].as_ref(), Some(Op::Env(Env::LaneId))) => {
                if constants[word.0] == Some(0) { return true; }
                match defs[word.0].as_ref() {
                    Some(Op::Int(IntOp::And, a, b)) => [a, b].iter().any(|w| ballots[w.0].is_some_and(|bit| same_bit(bit, exec, defs))),
                    _ => ballots[word.0].is_some_and(|bit| same_bit(bit, exec, defs)),
                }
            }
            _ => false,
        },
        _ => false,
    }
}

fn full_word(word: ValueId, old: ValueId, full: &[bool], defs: &[Option<Op>], ballots: &[Option<ValueId>], constants: &[Option<u64>], lane_mask: u64) -> bool {
    if let Some(k) = constants[word.0] { return k & lane_mask == lane_mask; }
    if let Some(bit) = ballots[word.0] { return full[bit.0] || same_bit(bit, old, defs) && full[old.0]; }
    match defs[word.0].as_ref() {
        Some(Op::Int(IntOp::Or, a, b)) => full_word(*a, old, full, defs, ballots, constants, lane_mask) || full_word(*b, old, full, defs, ballots, constants, lane_mask),
        Some(Op::Int(IntOp::And, a, b)) => full_word(*a, old, full, defs, ballots, constants, lane_mask) && full_word(*b, old, full, defs, ballots, constants, lane_mask),
        Some(Op::Convert(Cvt::Bitcast, _, a)) => full_word(*a, old, full, defs, ballots, constants, lane_mask),
        _ => false,
    }
}

fn full_bit(bit: ValueId, old: ValueId, full: &[bool], defs: &[Option<Op>], ballots: &[Option<ValueId>], constants: &[Option<u64>], lane_mask: u64) -> bool {
    if same_bit(bit, old, defs) { return full[old.0]; }
    if let Some(k) = constants[bit.0] { return k & 1 != 0; }
    match defs[bit.0].as_ref() {
        Some(Op::Convert(Cvt::Bitcast, Ty::I1, inner)) => full_bit(*inner, old, full, defs, ballots, constants, lane_mask),
        Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => match defs[shifted.0].as_ref() {
            Some(Op::Int(IntOp::LShr, word, lane)) if matches!(defs[lane.0].as_ref(), Some(Op::Env(Env::LaneId))) =>
                full_word(*word, old, full, defs, ballots, constants, lane_mask),
            _ => false,
        },
        Some(Op::Int(IntOp::And, a, b)) => full_bit(*a, old, full, defs, ballots, constants, lane_mask) && full_bit(*b, old, full, defs, ballots, constants, lane_mask),
        Some(Op::Int(IntOp::Or, a, b)) => full_bit(*a, old, full, defs, ballots, constants, lane_mask) || full_bit(*b, old, full, defs, ballots, constants, lane_mask),
        _ => false,
    }
}

#[derive(PartialEq)]
pub(crate) struct Predication {
    pub reactivation: Rc<Vec<(BlockId, usize)>>,
    pub predicated: Rc<Vec<Option<(ValueId, ValueId)>>>,
    pub masked: Rc<Vec<bool>>,
    pub masked_result: Vec<Option<ValueId>>,
    pub chain: Rc<BTreeMap<BlockId, Vec<(usize, ValueId)>>>,
    updates: Vec<(ValueId, ValueId, ValueId)>,
    defs: Vec<Option<Op>>,
    ballots: Vec<Option<ValueId>>,
}

fn predication(f: &Func, exec_index: usize, constants: &[Option<u64>]) -> Predication {
    let defs = f.definitions();
    let ballots = ballot_of(f);
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
                Inst::Effect { op: EffectOp::Wave(WaveOp::Any), inputs, outputs, .. } if outputs[0].0 == *cond => Some(inputs[0]),
                _ => None,
            }).filter(|input| !block.term.edges().any(|e| e.args.contains(input))),
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
    Predication { reactivation: Rc::new(reactivation), predicated: Rc::new(predicated), masked: Rc::new(masked), masked_result, chain: Rc::new(chain), updates, defs, ballots }
}

#[cfg(test)]
fn analyze(registry: &super::super::dialect::DialectRegistry, f: &Func, exec_index: usize, constants: &[Option<u64>], width: u32, entry_full: bool) -> Masks {
    let predication = predication(f, exec_index, constants);
    analyze_from(registry, f, &predication, exec_index, constants, width, entry_full)
}

fn analyze_from(registry: &super::super::dialect::DialectRegistry, f: &Func, p: &Predication, exec_index: usize, constants: &[Option<u64>], width: u32, entry_full: bool) -> Masks {
    let every_lane = |op: super::super::dialect::TargetOp| registry.operation(op).is_ok_and(|o| o.effect == super::super::dialect::Effect::ReadGlobal { every_lane: true });
    let (defs, ballots) = (&p.defs, &p.ballots);
    let any = any_of(f);
    let lane_mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
    let (reactivation, predicated, masked, chain, updates) = (Rc::clone(&p.reactivation), Rc::clone(&p.predicated), Rc::clone(&p.masked), Rc::clone(&p.chain), &p.updates);
    let mut full = vec![true; f.types.len()];
    let entry_exec = exec_param(&f.blocks[&f.entry], exec_index, f);
    full[entry_exec.0] = entry_full;
    let layout = layout(f);
    let index = &layout.index;
    let execs: Vec<ValueId> = {
        let entry = &f.blocks[&f.entry];
        let k = entry.params[..exec_index].iter().filter(|p| p.1 == Ty::I1).count();
        f.blocks.values().map(|block| block.params.iter().filter(|p| p.1 == Ty::I1).nth(k).expect("block lacks its EXEC parameter").0).collect()
    };
    let transfers: Vec<(ValueId, ValueId, bool)> = {
        let mut out = Vec::new();
        for (&id, block) in &f.blocks {
            let outgoing = chain[&id].last().unwrap().1;
            let guarded_edge = match &block.term {
                Term::CondBr { cond, no, .. } if all_active_guard(&any, *cond, outgoing, defs) => Some(no.dst),
                _ => None,
            };
            for edge in block.term.edges() { out.push((execs[index[edge.dst.0]], outgoing, guarded_edge == Some(edge.dst))); }
        }
        out
    };
    let mut incoming: Vec<Option<bool>> = vec![None; f.types.len()];
    loop {
        let mut changed = false;
        for &(value, old, bit) in updates.iter() {
            let fact = full_bit(bit, old, &full, defs, ballots, constants, lane_mask);
            if full[value.0] != fact { full[value.0] = fact; changed = true; }
        }
        for &param in &execs { incoming[param.0] = None; }
        incoming[entry_exec.0] = Some(entry_full);
        for &(param, outgoing, guarded) in &transfers {
            let fact = full[outgoing.0] || guarded;
            incoming[param.0] = Some(incoming[param.0].map_or(fact, |v| v & fact));
        }
        for &param in &execs {
            if let Some(fact) = incoming[param.0] { if full[param.0] != fact { full[param.0] = fact; changed = true; } }
        }
        if let Some(fact) = incoming[entry_exec.0] { if full[entry_exec.0] != fact { full[entry_exec.0] = fact; changed = true; } }
        if !changed { break; }
    }
    let live = live_across(f, &reactivation, &predicated, &layout);
    let internal = exposure(f, &predicated, &masked, &live, false, &every_lane, &layout);
    let exposed = exposure(f, &predicated, &masked, &live, true, &every_lane, &layout);
    let guarded = predicated.iter().enumerate().map(|(id, p)| p.is_some_and(|(_, exec)| full[exec.0] || internal[id] == 0)).collect();
    Masks { reactivation, full, guarded, predicated, masked, exposed, chain }
}

fn for_each_use(inst: &Inst, mut f: impl FnMut(ValueId)) {
    match inst {
        Inst::Core { op, .. } => { op.map(|v| { f(v); v }); }
        Inst::Packet { input, .. } => f(*input),
        Inst::Target { args, .. } => for &v in args.values() { f(v); },
        Inst::Effect { inputs, .. } => for &v in inputs { f(v); },
    }
}

fn for_each_output(inst: &Inst, mut f: impl FnMut(ValueId)) {
    match inst {
        Inst::Core { value, .. } | Inst::Packet { output: value, .. } => f(*value),
        Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => for o in outputs { f(o.0); },
    }
}

fn for_each_read(inst: &Inst, predicated: &[Option<(ValueId, ValueId)>], mut f: impl FnMut(ValueId)) {
    match inst {
        Inst::Core { value, op: Op::Select(c, new, _), .. } if predicated[value.0].is_some() => { f(*c); f(*new); }
        _ => for_each_use(inst, f),
    }
}

fn exposure(f: &Func, predicated: &[Option<(ValueId, ValueId)>], masked: &[bool], live: &[bool], returns: bool, every_lane: &dyn Fn(super::super::dialect::TargetOp) -> bool, layout: &Layout) -> Vec<u8> {
    let words = |v: ValueId| if f.types[v.0].bits() == 64 { 3u8 } else { 1u8 };
    let (producers, params) = (&layout.producers, &layout.params);
    let mut pending: Vec<(ValueId, u8)> = live.iter().enumerate().filter_map(|(id, &live)| live.then_some((ValueId(id), 3))).collect();
    for block in &layout.blocks {
        for inst in block.insts.iter() {
            match inst {
                Inst::Effect { op: EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane | WaveOp::ReadLane | WaveOp::WriteLane | WaveOp::BpermuteFi | WaveOp::Wmma), inputs, .. } => pending.extend(inputs.iter().map(|&v| (v, 3))),
                Inst::Packet { input, .. } => pending.push((*input, 3)),
                Inst::Target { op, args, provenance: Some(_), .. } if every_lane(*op) => pending.extend(args.values().iter().map(|&v| (v, 3))),
                _ => {}
            }
        }
        if let (true, Term::Ret(args)) = (returns, &block.term) { pending.extend(args.iter().map(|&v| (v, 3))); }
    }
    let mut exposed = vec![0u8; f.types.len()];
    while let Some((v, mask)) = pending.pop() {
        let added = mask & words(v) & !exposed[v.0];
        if added == 0 { continue; }
        exposed[v.0] |= added;
        if let Some((block, index)) = producers[v.0] {
            match &layout.blocks[block].insts[index] {
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
            for edge in &layout.incoming[block] { pending.push((edge.args[index], added)); }
        }
    }
    exposed
}

fn needed(f: &Func, predicated: &[Option<(ValueId, ValueId)>], layout: &Layout) -> Vec<bool> {
    let (params, producers) = (&layout.params, &layout.producers);
    let mut pending = Vec::new();
    for block in &layout.blocks {
        for inst in block.insts.iter() {
            match inst {
                Inst::Core { .. } | Inst::Packet { .. } => {}
                Inst::Effect { inputs, .. } => pending.extend(inputs.iter().copied()),
                Inst::Target { args, .. } => pending.extend(args.values().iter().copied()),
            }
        }
        match &block.term { Term::CondBr { cond, .. } => pending.push(*cond), Term::Ret(args) => pending.extend(args.iter().copied()), Term::Br(_) => {} }
    }
    let mut needed = vec![false; f.types.len()];
    while let Some(v) = pending.pop() {
        if needed[v.0] { continue; }
        needed[v.0] = true;
        if let Some((block, position)) = producers[v.0] {
            if let inst @ (Inst::Core { .. } | Inst::Packet { .. }) = &layout.blocks[block].insts[position] { for_each_read(inst, predicated, |r| pending.push(r)); }
        } else if let Some((block, position)) = params[v.0] {
            for edge in &layout.incoming[block] { pending.push(edge.args[position]); }
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

fn live_out(blocks: &[&Block], block: &Block, needed: &[bool], index: &[usize], live_in: &[Bits], live: &mut Bits) {
    live.clear();
    for edge in block.term.edges() {
        let dst = &live_in[index[edge.dst.0]];
        for (&arg, &(param, _)) in edge.args.iter().zip(&blocks[index[edge.dst.0]].params) {
            if needed[param.0] && dst.contains(param) { live.insert(arg); }
        }
    }
    match &block.term { Term::CondBr { cond, .. } => live.insert(*cond), Term::Ret(args) => for &a in args { live.insert(a); }, Term::Br(_) => {} }
}

fn live_across(f: &Func, points: &[(BlockId, usize)], predicated: &[Option<(ValueId, ValueId)>], layout: &Layout) -> Vec<bool> {
    if points.is_empty() { return vec![false; f.types.len()]; }
    let needed = needed(f, predicated, layout);
    let values = f.types.len();
    let index = &layout.index;
    let blocks = &layout.blocks;
    let mut live_in: Vec<Bits> = (0..f.blocks.len()).map(|_| Bits::new(values)).collect();
    let mut live = Bits::new(values);
    loop {
        let mut changed = false;
        for (b, block) in blocks.iter().enumerate().rev() {
            live_out(blocks, block, &needed, index, &live_in, &mut live);
            for inst in block.insts.iter().rev() {
                for_each_output(inst, |v| live.remove(v));
                if active(inst, &needed) { for_each_read(inst, predicated, |v| live.insert(v)); }
            }
            if live.0 != live_in[b].0 { std::mem::swap(&mut live_in[b].0, &mut live.0); changed = true; }
        }
        if !changed { break; }
    }
    let mut across = vec![false; values];
    let mut defined_after = Bits::new(values);
    for &(id, position) in points {
        let block = blocks[index[id.0]];
        live_out(blocks, block, &needed, index, &live_in, &mut live);
        for inst in block.insts[position + 1..].iter().rev() {
            for_each_output(inst, |v| live.remove(v));
            if active(inst, &needed) { for_each_read(inst, predicated, |v| live.insert(v)); }
        }
        defined_after.clear();
        for inst in &block.insts[position..] { for_each_output(inst, |v| defined_after.insert(v)); }
        live.for_each(|v| if !defined_after.contains(v) { across[v.0] = true; });
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
        let constants = super::super::constant::constants(&f);
        let masks = analyze(&crate::rdna_spmd::targets::rdna4::registry(), &f, 0, &constants, 16, false);
        assert_eq!(*masks.reactivation, vec![(BlockId(0), 10)]);
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

#[derive(PartialEq)]
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

fn word_of(value: ValueId, defs: &[Option<Op>], ballots: &[Option<ValueId>], constants: &[Option<u64>]) -> Word {
    if constants[value.0].is_some() { return Word::Constant; }
    if let Some(bit) = ballots[value.0] { return Word::Ballot(bit); }
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
            Some(Op::Int(IntOp::LShr, word, lane)) if matches!(defs[lane.0].as_ref(), Some(Op::Env(Env::LaneId))) => {
                if let Some(k) = constants[word.0] { Update::Constant(k) } else { Update::Word(*word) }
            }
            _ => Update::Unknown,
        },
        Some(Op::Int(IntOp::And, ..)) => Update::Bit,
        _ => if defs[bit.0].is_some() { Update::Unknown } else { Update::Copy(bit) },
    }
}

fn exec(f: &Func, exec_index: usize, constants: &[Option<u64>], width: u32, initial: bool, packed: bool) -> Exec {
    let defs = f.definitions();
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
    let any_of = any_of(f);
    let index = block_index(f);
    let blocks: Vec<&Block> = f.blocks.values().collect();
    let execs: Vec<ValueId> = {
        let entry = &f.blocks[&f.entry];
        let k = entry.params[..exec_index].iter().filter(|p| p.1 == Ty::I1).count();
        blocks.iter().map(|block| block.params.iter().filter(|p| p.1 == Ty::I1).nth(k).expect("block lacks its EXEC parameter").0).collect()
    };
    enum Resolved {
        Copy(ValueId),
        Constant(u64),
        Or { x: ValueId, y: ValueId, reads_old_x: bool, reads_old_y: bool },
        Ballot(ValueId),
        Empty,
    }
    let resolved: Vec<(ValueId, ValueId, Resolved)> = updates.iter().map(|&(value, old, ref update)| {
        let kind = match update {
            Update::Copy(v) => Resolved::Copy(*v),
            Update::Constant(k) => Resolved::Constant(*k),
            Update::Word(word) => match word_of(*word, &defs, &ballots, constants) {
                Word::Or(x, y) => {
                    let reads_old = |w: ValueId| matches!(word_of(w, &defs, &ballots, constants), Word::Ballot(bit) if same_bit(bit, old, &defs));
                    Resolved::Or { x, y, reads_old_x: reads_old(x), reads_old_y: reads_old(y) }
                }
                Word::Ballot(bit) => Resolved::Ballot(bit),
                _ => Resolved::Empty,
            },
            Update::Bit | Update::Unknown => Resolved::Empty,
        };
        (value, old, kind)
    }).collect();
    struct Transfer<'f> { param: ValueId, outgoing: ValueId, stops: bool, pinned: Option<bool>, carries: bool, edge: &'f Edge, params: &'f [(ValueId, Ty)] }
    let transfers: Vec<Transfer> = {
        let mut out = Vec::new();
        for (&id, block) in &f.blocks {
            let outgoing = chain[&id].last().unwrap().1;
            let pins: Vec<(BlockId, bool)> = match &block.term {
                Term::CondBr { cond, yes, no } => {
                    let (query, negated) = match defs[cond.0].as_ref() {
                        Some(Op::Cmp(super::super::ir::IntPred::Eq, q, zero)) if constants[zero.0] == Some(0) => (*q, true),
                        _ => (*cond, false),
                    };
                    match any_of[query.0] {
                        Some(bit) if same_bit(bit, outgoing, &defs) || same_bit(outgoing, bit, &defs) => vec![(yes.dst, !negated), (no.dst, negated)],
                        _ if !negated && all_active_guard(&any_of, *cond, outgoing, &defs) => vec![(no.dst, true)],
                        _ => vec![],
                    }
                }
                _ => vec![],
            };
            let stops = barrier.contains(&id);
            for edge in block.term.edges() {
                let pinned = pins.iter().find(|(dst, _)| *dst == edge.dst).map(|(_, v)| *v);
                let dst = index[edge.dst.0];
                out.push(Transfer {
                    param: execs[dst], outgoing, stops, pinned,
                    carries: packed || !(edge.dst <= id || pinned == Some(false)),
                    edge, params: &blocks[dst].params,
                });
            }
        }
        out
    };
    let mut incoming_nonempty: Vec<Option<bool>> = vec![None; f.types.len()];
    let mut incoming_active: Vec<Option<bool>> = vec![None; f.types.len()];
    let mut incoming_saved: Vec<Option<bool>> = vec![None; f.types.len()];
    let all_params: Vec<ValueId> = blocks.iter().flat_map(|b| b.params.iter().map(|p| p.0)).collect();
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
        for &(value, old, ref update) in &resolved {
            let (n, a) = match *update {
                Resolved::Copy(v) => (nonempty[v.0], active[v.0]),
                Resolved::Constant(k) => (initial && k & lane_mask != 0, k & 1 != 0),
                Resolved::Or { x, y, reads_old_x, reads_old_y } => {
                    let preserves = reads_old_x || reads_old_y;
                    (preserves && nonempty[old.0], (reads_old_x && active[old.0]) || (reads_old_y && active[old.0]) || saved[x.0] || saved[y.0])
                }
                Resolved::Ballot(bit) => (nonempty[bit.0], active[bit.0]),
                Resolved::Empty => (false, false),
            };
            set(&mut nonempty, value, n);
            set(&mut active, value, a);
        }
        for &p in &all_params { incoming_nonempty[p.0] = None; incoming_active[p.0] = None; incoming_saved[p.0] = None; }
        incoming_nonempty[entry_exec.0] = Some(initial);
        incoming_active[entry_exec.0] = Some(true);
        for &(id, _) in &f.blocks[&f.entry].params { incoming_saved[id.0] = Some(false); }
        let meet = |table: &mut Vec<Option<bool>>, id: ValueId, fact: bool| { table[id.0] = Some(table[id.0].unwrap_or(true) & fact); };
        for t in &transfers {
            let fact = if t.stops { false } else { t.pinned.unwrap_or(nonempty[t.outgoing.0]) };
            meet(&mut incoming_nonempty, t.param, fact);
            if !t.carries { continue; }
            let fact = t.pinned.unwrap_or(active[t.outgoing.0]);
            meet(&mut incoming_active, t.param, fact);
            for (&arg, &(p, _)) in t.edge.args.iter().zip(t.params) {
                meet(&mut incoming_saved, p, saved[arg.0]);
            }
        }
        for &p in &all_params {
            if let Some(fact) = incoming_nonempty[p.0] { set(&mut nonempty, p, fact); }
            if let Some(fact) = incoming_active[p.0] { set(&mut active, p, fact); }
            if let Some(fact) = incoming_saved[p.0] { set(&mut saved, p, fact); }
        }
        if !changed { break; }
    }
    Exec { chain, nonempty, active }
}

