use super::super::ir::{Cvt, EffectOp, Env, IntOp, Op, Ty, ValueId, WaveOp, *};
use super::dataflow::{for_each_output, Backward, Cfg, Lattice, Sparse};
use super::lanes::{ballot_of, lane_word, valid_masked, Lanes, Region};
use super::{Analyses, Analysis, Constants};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

#[derive(PartialEq)]
pub(crate) struct Masks {
    #[cfg_attr(not(test), allow(dead_code))]
    pub reactivation: Rc<Vec<(BlockId, usize)>>,
    pub full: Vec<bool>,
    pub guarded: Vec<bool>,
    pub observed: Vec<bool>,
    pub predicated: Rc<Vec<Option<(ValueId, ValueId)>>>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub masked: Rc<Vec<bool>>,
    pub chain: Rc<BTreeMap<BlockId, Vec<(usize, ValueId)>>>,
}

impl Masks {
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
        let (predication, constants) =
            (analyses.get::<Predication>(f), analyses.get::<Constants>(f));
        analyze_from(
            ctx.registry,
            f,
            &predication,
            ctx.exec_index,
            &constants,
            ctx.lanes,
            ctx.entry_full,
        )
    }
}

impl Analysis for Exec {
    type Result = Exec;
    const NAME: &'static str = "exec";
    fn compute(f: &Func, analyses: &Analyses) -> Self {
        let ctx = analyses.context();
        let constants = analyses.get::<Constants>(f);
        exec(f, ctx.exec_index, &constants, ctx.lanes, ctx.exec_initial)
    }
}

fn wave_any_of(f: &Func) -> Vec<bool> {
    let mut out = vec![false; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            if let Inst::Effect {
                op: EffectOp::Wave(WaveOp::Any),
                outputs,
                ..
            } = inst
            {
                out[outputs[0].0 .0] = true;
            }
        }
    }
    out
}

pub(super) fn any_of(f: &Func) -> Vec<Option<ValueId>> {
    let mut out = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Packet {
                    op: PacketOp::Any,
                    input,
                    output,
                } => out[output.0] = Some(*input),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any),
                    inputs,
                    outputs,
                    ..
                } => out[outputs[0].0 .0] = Some(inputs[0]),
                _ => {}
            }
        }
    }
    out
}

pub(crate) fn all_active_guard(
    any: &[Option<ValueId>],
    cond: ValueId,
    exec: ValueId,
    defs: &[Option<Op>],
) -> bool {
    let Some(inactive) = any[cond.0] else {
        return false;
    };
    let Some(negated) = valid_masked(defs, inactive) else {
        return false;
    };
    let Some(Op::Int(IntOp::Xor, x, one)) = defs[negated.0] else {
        return false;
    };
    let flipped = |v: ValueId| matches!(defs[v.0], Some(Op::Const(Ty::I1, 1)));
    let bit = if flipped(one) {
        x
    } else if flipped(x) {
        one
    } else {
        return false;
    };
    same_bit(bit, exec, defs) || same_bit(exec, bit, defs)
}

fn exec_param(block: &Block, exec_index: usize, f: &Func) -> ValueId {
    let entry = &f.blocks[&f.entry];
    let k = entry.params[..exec_index]
        .iter()
        .filter(|p| p.1 == Ty::I1)
        .count();
    block
        .params
        .iter()
        .filter(|p| p.1 == Ty::I1)
        .nth(k)
        .expect("block lacks its EXEC parameter")
        .0
}

fn block_index(f: &Func) -> Vec<usize> {
    let mut out = vec![usize::MAX; f.blocks.keys().map(|b| b.0 + 1).max().unwrap_or(0)];
    for (index, id) in f.blocks.keys().enumerate() {
        out[id.0] = index;
    }
    out
}

struct Layout<'f> {
    blocks: Vec<&'f Block>,
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
        for (position, &(p, _)) in block.params.iter().enumerate() {
            params[p.0] = Some((at, position));
        }
        for (position, inst) in block.insts.iter().enumerate() {
            for_each_output(inst, |v| producers[v.0] = Some((at, position)));
        }
    }
    Layout {
        blocks,
        incoming,
        producers,
        params,
    }
}

fn incoming_edges<'f>(f: &'f Func, index: &[usize]) -> Vec<Vec<&'f Edge>> {
    let mut out: Vec<Vec<&Edge>> = (0..f.blocks.len()).map(|_| Vec::new()).collect();
    for block in f.blocks.values() {
        for edge in block.term.edges() {
            out[index[edge.dst.0]].push(edge);
        }
    }
    out
}

#[derive(Clone, PartialEq)]
struct Bits(Vec<u64>);
impl Lattice for Bits {
    fn meet(&self, other: &Self) -> Self {
        Bits(self.0.iter().zip(&other.0).map(|(a, b)| a | b).collect())
    }
}
impl Bits {
    fn new(values: usize) -> Self {
        Self(vec![0; values.div_ceil(64)])
    }
    fn insert(&mut self, v: ValueId) {
        self.0[v.0 / 64] |= 1 << (v.0 % 64);
    }
    fn remove(&mut self, v: ValueId) {
        self.0[v.0 / 64] &= !(1 << (v.0 % 64));
    }
    fn contains(&self, v: ValueId) -> bool {
        self.0[v.0 / 64] >> (v.0 % 64) & 1 != 0
    }
    fn for_each(&self, mut f: impl FnMut(ValueId)) {
        for (w, &word) in self.0.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                f(ValueId(w * 64 + b));
            }
        }
    }
}

fn same_bit(value: ValueId, exec: ValueId, defs: &[Option<Op>]) -> bool {
    let mut value = value;
    loop {
        if value == exec {
            return true;
        }
        match defs[value.0].as_ref() {
            Some(Op::Convert(Cvt::Bitcast, Ty::I1, inner)) => value = *inner,
            _ => return false,
        }
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
    lanes: Lanes,
}

fn predication(f: &Func, exec_index: usize, constants: &[Option<u64>]) -> Predication {
    let lanes = Lanes::new(f, constants);
    let defs = &lanes.defs;
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
            Term::CondBr { cond, .. } => block
                .insts
                .iter()
                .find_map(|inst| match inst {
                    Inst::Packet {
                        op: PacketOp::Any,
                        input,
                        output,
                    } if output == cond => Some(*input),
                    Inst::Effect {
                        op: EffectOp::Wave(WaveOp::Any),
                        inputs,
                        outputs,
                        ..
                    } if outputs[0].0 == *cond => Some(inputs[0]),
                    _ => None,
                })
                .filter(|input| !block.term.edges().any(|e| e.args.contains(input))),
            _ => None,
        };
        for (index, inst) in block.insts.iter().enumerate() {
            let Inst::Core { value, ty, op } = inst else {
                continue;
            };
            if let Some(bit) = lanes.valid_masked(*value).filter(|_| query != Some(*value)) {
                if !lanes.within(bit, &lanes.class(current)) {
                    reactivation.push((id, index));
                }
                updates.push((*value, current, bit));
                current = *value;
                entries.push((index + 1, current));
                continue;
            }
            match *op {
                Op::Select(c, new, _) if same_bit(c, current, defs) => {
                    predicated[value.0] = Some((new, current))
                }
                Op::Int(IntOp::And, a, b)
                    if *ty == Ty::I1
                        && (same_bit(a, current, defs) || same_bit(b, current, defs)) =>
                {
                    masked[value.0] = true;
                    masked_result[value.0] = Some(if same_bit(b, current, defs) { a } else { b });
                }
                _ => {}
            }
        }
        chain.insert(id, entries);
    }
    Predication {
        reactivation: Rc::new(reactivation),
        predicated: Rc::new(predicated),
        masked: Rc::new(masked),
        masked_result,
        chain: Rc::new(chain),
        updates,
        lanes,
    }
}

fn analyze_from(
    registry: &super::super::dialect::DialectRegistry,
    f: &Func,
    p: &Predication,
    exec_index: usize,
    constants: &[Option<u64>],
    width: u32,
    entry_full: bool,
) -> Masks {
    let every_lane = |op: super::super::dialect::TargetOp| {
        registry.operation(op).is_ok_and(|o| {
            o.effect == super::super::dialect::Effect::ReadGlobal { every_lane: true }
        })
    };
    let defs = &p.lanes.defs;
    let any = any_of(f);
    let lane_mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let (reactivation, predicated, masked, chain, updates) = (
        Rc::clone(&p.reactivation),
        Rc::clone(&p.predicated),
        Rc::clone(&p.masked),
        Rc::clone(&p.chain),
        &p.updates,
    );
    let layout = layout(f);
    let cfg = Cfg::new(f);
    let execs: Vec<ValueId> = cfg
        .blocks
        .iter()
        .map(|block| exec_param(block, exec_index, f))
        .collect();
    let entry_exec = execs[cfg.entry];
    let guarded_edge: Vec<Option<BlockId>> = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(at, block)| match &block.term {
            Term::CondBr { cond, no, .. }
                if all_active_guard(&any, *cond, chain[&cfg.ids[at]].last().unwrap().1, defs) =>
            {
                Some(no.dst)
            }
            _ => None,
        })
        .collect();
    let bit_update: Vec<Option<(ValueId, ValueId)>> = {
        let mut out = vec![None; f.types.len()];
        for &(value, old, bit) in updates.iter() {
            out[value.0] = Some((old, bit));
        }
        out
    };
    let known = |v: ValueId| {
        constants[v.0].map(|k| {
            if f.types[v.0] == Ty::I1 {
                k & 1 != 0
            } else {
                k & lane_mask == lane_mask
            }
        })
    };
    let boundary = |p: ValueId| {
        if p == entry_exec {
            entry_full
        } else {
            known(p).unwrap_or(false)
        }
    };
    let edge = |edge: &Edge, src: usize, position: usize, facts: &[bool]| {
        let dst = cfg.index[edge.dst.0];
        let param = cfg.blocks[dst].params[position].0;
        if param != execs[dst] {
            return known(param).unwrap_or(false);
        }
        facts[edge.args[position].0] || guarded_edge[src] == Some(edge.dst)
    };
    let transfer = |inst: &Inst, v: ValueId, facts: &[bool]| -> bool {
        if let Some((old, bit)) = bit_update[v.0] {
            return facts[bit.0] || same_bit(bit, old, defs) && facts[old.0];
        }
        if let Some(fact) = known(v) {
            return fact;
        }
        match inst {
            Inst::Packet {
                op: PacketOp::Ballot,
                input,
                ..
            } => facts[input.0],
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::Ballot),
                inputs,
                ..
            } => facts[inputs[0].0],
            Inst::Core { op, .. } => match *op {
                Op::Env(Env::ValidLane) => true,
                Op::Convert(Cvt::Bitcast, _, a) => facts[a.0],
                Op::Convert(Cvt::Trunc, Ty::I1, _) => {
                    lane_word(defs, v).is_some_and(|word| facts[word.0])
                }
                Op::Int(IntOp::And, a, b) => facts[a.0] && facts[b.0],
                Op::Int(IntOp::Or, a, b) => facts[a.0] || facts[b.0],
                _ => false,
            },
            _ => false,
        }
    };
    let full = Sparse {
        cfg: &cfg,
        start: true,
        boundary: &boundary,
        edge: &edge,
        transfer: &transfer,
    }
    .solve(f.types.len());
    let live = live_across(f, &reactivation, &predicated, &layout);
    let sets = exposure_sets(f, &predicated, &masked, &live, &every_lane, &layout);
    let current = defining_execs(f, &chain);
    let region = Region::new(f, &p.lanes, &cfg, &chain);
    let mut settled: BTreeMap<ValueId, bool> = BTreeMap::new();
    let residual: Vec<u8> = sets
        .iter()
        .zip(&current)
        .map(|(&[low, high], exec)| {
            let roots = low | high;
            let Some(exec) = *exec else {
                return roots;
            };
            if full[exec.0] {
                return 0;
            }
            if roots & POINT == 0 || roots & STATIC != 0 {
                return roots;
            }
            let never_widens = *settled
                .entry(exec)
                .or_insert_with(|| region.never_widens(exec));
            if never_widens {
                roots & !POINT
            } else {
                roots
            }
        })
        .collect();
    let guarded = predicated
        .iter()
        .zip(&residual)
        .map(|(p, roots)| p.is_some() && roots & (POINT | STATIC) == 0)
        .collect();
    let observed = residual.iter().map(|&roots| roots != 0).collect();
    Masks {
        reactivation,
        full,
        guarded,
        observed,
        predicated,
        masked,
        chain,
    }
}

fn defining_execs(
    f: &Func,
    chain: &BTreeMap<BlockId, Vec<(usize, ValueId)>>,
) -> Vec<Option<ValueId>> {
    let mut out = vec![None; f.types.len()];
    for (id, block) in &f.blocks {
        let entries = &chain[id];
        for &(param, _) in &block.params {
            out[param.0] = Some(entries[0].1);
        }
        let mut next = 0;
        for (index, inst) in block.insts.iter().enumerate() {
            while next + 1 < entries.len() && entries[next + 1].0 <= index {
                next += 1;
            }
            for_each_output(inst, |value| out[value.0] = Some(entries[next].1));
        }
    }
    out
}

const POINT: u8 = 1;
const RET: u8 = 2;
const STATIC: u8 = 4;

fn for_each_use(inst: &Inst, mut f: impl FnMut(ValueId)) {
    match inst {
        Inst::Core { op, .. } => {
            op.map(|v| {
                f(v);
                v
            });
        }
        Inst::Packet { input, .. } => f(*input),
        Inst::Target { args, .. } => {
            for &v in args.values() {
                f(v);
            }
        }
        Inst::Effect { inputs, .. } => {
            for &v in inputs {
                f(v);
            }
        }
    }
}

fn for_each_read(
    inst: &Inst,
    predicated: &[Option<(ValueId, ValueId)>],
    mut f: impl FnMut(ValueId),
) {
    match inst {
        Inst::Core {
            value,
            op: Op::Select(c, new, _),
            ..
        } if predicated[value.0].is_some() => {
            f(*c);
            f(*new);
        }
        _ => for_each_use(inst, f),
    }
}

fn exposure_sets(
    f: &Func,
    predicated: &[Option<(ValueId, ValueId)>],
    masked: &[bool],
    live: &[bool],
    every_lane: &dyn Fn(super::super::dialect::TargetOp) -> bool,
    layout: &Layout,
) -> Vec<[u8; 2]> {
    let wide = |v: ValueId| f.types[v.0].bits() == 64;
    let (producers, params) = (&layout.producers, &layout.params);
    let mut seeds = vec![[0u8; 2]; f.types.len()];
    let seed = |seeds: &mut Vec<[u8; 2]>, v: ValueId, root: u8| {
        seeds[v.0] = [seeds[v.0][0] | root, seeds[v.0][1] | root];
    };
    for (id, &live) in live.iter().enumerate() {
        if live {
            seed(&mut seeds, ValueId(id), POINT);
        }
    }
    for block in &layout.blocks {
        for inst in block.insts.iter() {
            match inst {
                Inst::Effect {
                    op:
                        EffectOp::Wave(
                            WaveOp::Any
                            | WaveOp::Ballot
                            | WaveOp::ReadFirstLane
                            | WaveOp::ReadLane
                            | WaveOp::WriteLane
                            | WaveOp::BpermuteFi
                            | WaveOp::Wmma,
                        ),
                    inputs,
                    ..
                } => {
                    for &v in inputs {
                        seed(&mut seeds, v, STATIC);
                    }
                }
                Inst::Packet { input, .. } => seed(&mut seeds, *input, STATIC),
                Inst::Target {
                    op,
                    args,
                    provenance: Some(_),
                    ..
                } if every_lane(*op) => {
                    for &v in args.values() {
                        seed(&mut seeds, v, STATIC);
                    }
                }
                _ => {}
            }
        }
        if let Term::Ret(args) = &block.term {
            for &v in args {
                seed(&mut seeds, v, RET);
            }
        }
    }
    let mut pending: Vec<(ValueId, [u8; 2])> = seeds
        .iter()
        .enumerate()
        .filter(|(_, &roots)| roots != [0, 0])
        .map(|(id, &roots)| (ValueId(id), roots))
        .collect();
    let mut sets = vec![[0u8; 2]; f.types.len()];
    while let Some((v, want)) = pending.pop() {
        let mut added = [0u8; 2];
        for (half, slot) in added.iter_mut().enumerate() {
            if half == 0 || wide(v) {
                *slot = want[half] & !sets[v.0][half];
                sets[v.0][half] |= *slot;
            }
        }
        if added == [0, 0] {
            continue;
        }
        let both = added[0] | added[1];
        let mut push = |target: ValueId, roots: [u8; 2]| {
            if roots != [0, 0] {
                pending.push((target, roots));
            }
        };
        if let Some((block, index)) = producers[v.0] {
            match &layout.blocks[block].insts[index] {
                Inst::Core {
                    op: Op::Select(_, _, old),
                    ..
                } if predicated[v.0].is_some() => push(*old, added),
                Inst::Core {
                    op: Op::Int(..), ..
                } if masked[v.0] => {}
                Inst::Core {
                    op: Op::UnpackLo(x),
                    ..
                } => push(*x, [both, 0]),
                Inst::Core {
                    op: Op::UnpackHi(x),
                    ..
                } => push(*x, [0, both]),
                Inst::Core {
                    op: Op::Pack64(lo, hi),
                    ..
                } => {
                    push(*lo, [added[0], 0]);
                    push(*hi, [added[1], 0]);
                }
                Inst::Core {
                    op: Op::Convert(Cvt::Bitcast, _, a),
                    ..
                } => push(*a, added),
                Inst::Core {
                    op: Op::Convert(Cvt::Trunc | Cvt::ZExt | Cvt::SExt, _, a),
                    ..
                } => push(*a, [both, 0]),
                Inst::Core { op, .. } => {
                    op.map(|a| {
                        push(a, [both, both]);
                        a
                    });
                }
                Inst::Target { args, .. } => {
                    for &a in args.values() {
                        push(a, [both, both]);
                    }
                }
                Inst::Packet { input, .. } => push(*input, [both, both]),
                Inst::Effect { .. } => {}
            }
        } else if let Some((block, index)) = params[v.0] {
            for edge in &layout.incoming[block] {
                push(edge.args[index], added);
            }
        }
    }
    sets
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
        match &block.term {
            Term::CondBr { cond, .. } => pending.push(*cond),
            Term::Ret(args) => pending.extend(args.iter().copied()),
            Term::Br(_) => {}
        }
    }
    let mut needed = vec![false; f.types.len()];
    while let Some(v) = pending.pop() {
        if needed[v.0] {
            continue;
        }
        needed[v.0] = true;
        if let Some((block, position)) = producers[v.0] {
            if let inst @ (Inst::Core { .. } | Inst::Packet { .. }) =
                &layout.blocks[block].insts[position]
            {
                for_each_read(inst, predicated, |r| pending.push(r));
            }
        } else if let Some((block, position)) = params[v.0] {
            for edge in &layout.incoming[block] {
                pending.push(edge.args[position]);
            }
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

fn live_across(
    f: &Func,
    points: &[(BlockId, usize)],
    predicated: &[Option<(ValueId, ValueId)>],
    layout: &Layout,
) -> Vec<bool> {
    if points.is_empty() {
        return vec![false; f.types.len()];
    }
    let needed = needed(f, predicated, layout);
    let values = f.types.len();
    let cfg = Cfg::new(f);
    let with_terminator = |at: usize, exit: &Bits| {
        let mut live = exit.clone();
        match &cfg.blocks[at].term {
            Term::CondBr { cond, .. } => live.insert(*cond),
            Term::Ret(args) => {
                for &a in args {
                    live.insert(a);
                }
            }
            Term::Br(_) => {}
        }
        live
    };
    let edge = |_: usize, edge: &Edge, entry: &Bits| {
        let mut out = Bits::new(values);
        for (&arg, &(param, _)) in edge
            .args
            .iter()
            .zip(&cfg.blocks[cfg.index[edge.dst.0]].params)
        {
            if needed[param.0] && entry.contains(param) {
                out.insert(arg);
            }
        }
        out
    };
    let transfer = |at: usize, exit: &Bits| {
        let mut live = with_terminator(at, exit);
        for inst in cfg.blocks[at].insts.iter().rev() {
            for_each_output(inst, |v| live.remove(v));
            if active(inst, &needed) {
                for_each_read(inst, predicated, |v| live.insert(v));
            }
        }
        live
    };
    let (_, exit) = Backward {
        cfg: &cfg,
        start: Bits::new(values),
        edge: &edge,
        transfer: &transfer,
    }
    .solve();
    let mut across = vec![false; values];
    for &(id, position) in points {
        let at = cfg.index[id.0];
        let block = cfg.blocks[at];
        let mut live = with_terminator(at, &exit[at]);
        for inst in block.insts[position + 1..].iter().rev() {
            for_each_output(inst, |v| live.remove(v));
            if active(inst, &needed) {
                for_each_read(inst, predicated, |v| live.insert(v));
            }
        }
        for inst in &block.insts[position..] {
            for_each_output(inst, |v| live.remove(v));
        }
        live.for_each(|v| across[v.0] = true);
    }
    across
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
        chain
            .iter()
            .rev()
            .find(|(at, _)| *at <= index)
            .map(|(_, v)| *v)
            .unwrap_or(chain[0].1)
    }
    pub fn nonempty_at(&self, block: BlockId, index: usize) -> bool {
        self.nonempty[self.at(block, index).0]
    }
    pub fn active_at(&self, block: BlockId, index: usize) -> bool {
        self.active[self.at(block, index).0]
    }
}

enum Word {
    Constant,
    Ballot(ValueId),
    Or(ValueId, ValueId),
    And,
    Other,
}

fn word_of(
    value: ValueId,
    defs: &[Option<Op>],
    ballots: &[Option<ValueId>],
    constants: &[Option<u64>],
) -> Word {
    if constants[value.0].is_some() {
        return Word::Constant;
    }
    if let Some(bit) = ballots[value.0] {
        return Word::Ballot(bit);
    }
    match defs[value.0].as_ref() {
        Some(Op::Int(IntOp::Or, a, b)) => Word::Or(*a, *b),
        Some(Op::Int(IntOp::And, ..)) => Word::And,
        Some(Op::Convert(Cvt::Bitcast, _, a)) => word_of(*a, defs, ballots, constants),
        _ => Word::Other,
    }
}

enum Update {
    Copy(ValueId),
    Constant(u64),
    Word(ValueId),
    Bit,
    Unknown,
}

fn update_of(bit: ValueId, defs: &[Option<Op>], constants: &[Option<u64>]) -> Update {
    if let Some(k) = constants[bit.0] {
        return Update::Constant(if k & 1 != 0 { u64::MAX } else { 0 });
    }
    match defs[bit.0].as_ref() {
        Some(Op::Convert(Cvt::Bitcast, Ty::I1, inner)) => update_of(*inner, defs, constants),
        Some(Op::Convert(Cvt::Trunc, Ty::I1, _)) => match lane_word(defs, bit) {
            Some(word) => match constants[word.0] {
                Some(k) => Update::Constant(k),
                None => Update::Word(word),
            },
            None => Update::Unknown,
        },
        Some(Op::Int(IntOp::And, ..)) => Update::Bit,
        _ => {
            if defs[bit.0].is_some() {
                Update::Unknown
            } else {
                Update::Copy(bit)
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct ExecFact {
    nonempty: bool,
    active: bool,
    saved: bool,
}
impl Lattice for ExecFact {
    fn meet(&self, other: &Self) -> Self {
        ExecFact {
            nonempty: self.nonempty && other.nonempty,
            active: self.active && other.active,
            saved: self.saved && other.saved,
        }
    }
}

fn exec(f: &Func, exec_index: usize, constants: &[Option<u64>], width: u32, initial: bool) -> Exec {
    let defs = f.definitions();
    let ballots = ballot_of(f);
    let mut chain: BTreeMap<BlockId, Vec<(usize, ValueId)>> = BTreeMap::new();
    let mut updates: Vec<(ValueId, ValueId, Update)> = Vec::new();
    let mut barrier = BTreeSet::new();
    for (&id, block) in &f.blocks {
        let mut current = exec_param(block, exec_index, f);
        let mut entries = vec![(0usize, current)];
        for (index, inst) in block.insts.iter().enumerate() {
            let update = match inst {
                Inst::Core { value, .. } => valid_masked(&defs, *value).map(|bit| (*value, bit)),
                _ => None,
            };
            if let Some((value, bit)) = update {
                updates.push((value, current, update_of(bit, &defs, constants)));
                current = value;
                entries.push((index + 1, current));
                continue;
            }
            match inst {
                Inst::Core {
                    value,
                    ty: Ty::I1,
                    op: Op::Int(IntOp::And, a, b),
                    ..
                } if same_bit(*a, current, &defs) || same_bit(*b, current, &defs) => {
                    updates.push((*value, current, Update::Bit));
                    current = *value;
                    entries.push((index + 1, current));
                }
                Inst::Effect {
                    op: EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait,
                    ..
                } => {
                    barrier.insert(id);
                }
                _ => {}
            }
        }
        chain.insert(id, entries);
    }
    let lane_mask = if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let any_of = any_of(f);
    let wave_any = wave_any_of(f);
    let cfg = Cfg::new(f);
    let execs: Vec<ValueId> = cfg
        .blocks
        .iter()
        .map(|block| exec_param(block, exec_index, f))
        .collect();
    enum Resolved {
        Copy(ValueId),
        Constant(u64),
        Or {
            x: ValueId,
            y: ValueId,
            reads_old_x: bool,
            reads_old_y: bool,
        },
        Ballot(ValueId),
        Empty,
    }
    let mut resolved: Vec<Option<(ValueId, Resolved)>> = (0..f.types.len()).map(|_| None).collect();
    for &(value, old, ref update) in &updates {
        let kind = match update {
            Update::Copy(v) => Resolved::Copy(*v),
            Update::Constant(k) => Resolved::Constant(*k),
            Update::Word(word) => match word_of(*word, &defs, &ballots, constants) {
                Word::Or(x, y) => {
                    let reads_old = |w: ValueId| matches!(word_of(w, &defs, &ballots, constants), Word::Ballot(bit) if same_bit(bit, old, &defs));
                    Resolved::Or {
                        x,
                        y,
                        reads_old_x: reads_old(x),
                        reads_old_y: reads_old(y),
                    }
                }
                Word::Ballot(bit) => Resolved::Ballot(bit),
                _ => Resolved::Empty,
            },
            Update::Bit | Update::Unknown => Resolved::Empty,
        };
        resolved[value.0] = Some((old, kind));
    }
    let mut tracked = vec![false; f.types.len()];
    for &p in &execs {
        tracked[p.0] = true;
    }
    for &(value, ..) in &updates {
        tracked[value.0] = true;
    }
    let pins: Vec<Vec<(BlockId, bool)>> = cfg
        .blocks
        .iter()
        .enumerate()
        .map(|(at, block)| {
            let outgoing = chain[&cfg.ids[at]].last().unwrap().1;
            match &block.term {
                Term::CondBr { cond, yes, no } => {
                    let (query, negated) = match defs[cond.0].as_ref() {
                        Some(Op::Cmp(super::super::ir::IntPred::Eq, q, zero))
                            if constants[zero.0] == Some(0) =>
                        {
                            (*q, true)
                        }
                        _ => (*cond, false),
                    };
                    let pins = match any_of[query.0] {
                        Some(bit)
                            if same_bit(bit, outgoing, &defs) || same_bit(outgoing, bit, &defs) =>
                        {
                            vec![(yes.dst, !negated), (no.dst, negated)]
                        }
                        _ if !negated && all_active_guard(&any_of, *cond, outgoing, &defs) => {
                            vec![(no.dst, true)]
                        }
                        _ => vec![],
                    };
                    if wave_any[query.0] {
                        pins.into_iter().filter(|(_, held)| !held).collect()
                    } else {
                        pins
                    }
                }
                _ => vec![],
            }
        })
        .collect();
    let entry_exec = execs[cfg.entry];
    let boundary = |p: ValueId| ExecFact {
        nonempty: p == entry_exec && initial,
        active: p == entry_exec,
        saved: p == entry_exec,
    };
    let edge = |edge: &Edge, src: usize, position: usize, facts: &[ExecFact]| {
        let dst = cfg.index[edge.dst.0];
        let arg = edge.args[position];
        if cfg.blocks[dst].params[position].0 != execs[dst] {
            return ExecFact {
                nonempty: false,
                active: false,
                saved: facts[arg.0].saved,
            };
        }
        let pinned = pins[src]
            .iter()
            .find(|(d, _)| *d == edge.dst)
            .map(|(_, v)| *v);
        let nonempty = !barrier.contains(&cfg.ids[src]) && pinned.unwrap_or(facts[arg.0].nonempty);
        let active = pinned.unwrap_or(facts[arg.0].active);
        ExecFact {
            nonempty,
            active,
            saved: active,
        }
    };
    let transfer = |inst: &Inst, v: ValueId, facts: &[ExecFact]| {
        let none = ExecFact {
            nonempty: false,
            active: false,
            saved: false,
        };
        if let Some((old, kind)) = &resolved[v.0] {
            let held = |x: ValueId| {
                if tracked[x.0] {
                    (facts[x.0].nonempty, facts[x.0].active)
                } else {
                    (facts[x.0].saved, facts[x.0].saved)
                }
            };
            let (nonempty, active) = match *kind {
                Resolved::Copy(x) => held(x),
                Resolved::Constant(k) => (initial && k & lane_mask != 0, k & 1 != 0),
                Resolved::Or {
                    x,
                    y,
                    reads_old_x,
                    reads_old_y,
                } => (
                    (reads_old_x || reads_old_y) && facts[old.0].nonempty,
                    (reads_old_x || reads_old_y) && facts[old.0].active
                        || facts[x.0].saved
                        || facts[y.0].saved,
                ),
                Resolved::Ballot(bit) => held(bit),
                Resolved::Empty => (false, false),
            };
            return ExecFact {
                nonempty,
                active,
                saved: active,
            };
        }
        match inst {
            Inst::Packet {
                op: PacketOp::Ballot,
                input,
                ..
            } => ExecFact {
                saved: facts[input.0].saved,
                ..none
            },
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::Ballot),
                inputs,
                ..
            } => ExecFact {
                saved: facts[inputs[0].0].saved,
                ..none
            },
            Inst::Core {
                op: Op::Convert(Cvt::Bitcast, _, a),
                ..
            } => ExecFact {
                saved: facts[a.0].saved,
                ..none
            },
            _ => none,
        }
    };
    let start = ExecFact {
        nonempty: true,
        active: true,
        saved: true,
    };
    let facts = Sparse {
        cfg: &cfg,
        start,
        boundary: &boundary,
        edge: &edge,
        transfer: &transfer,
    }
    .solve(f.types.len());
    Exec {
        chain,
        nonempty: facts.iter().map(|x| x.nonempty).collect(),
        active: facts.iter().map(|x| x.active).collect(),
    }
}
