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
    pub fn at(&self, block: BlockId, index: usize) -> ValueId {
        let chain = &self.chain[&block];
        chain
            .iter()
            .rev()
            .find(|(at, _)| *at <= index)
            .map(|(_, v)| *v)
            .unwrap_or(chain[0].1)
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

#[cfg(test)]
fn analyze(
    registry: &super::super::dialect::DialectRegistry,
    f: &Func,
    exec_index: usize,
    constants: &[Option<u64>],
    width: u32,
    entry_full: bool,
) -> Masks {
    let predication = predication(f, exec_index, constants);
    analyze_from(
        registry,
        f,
        &predication,
        exec_index,
        constants,
        width,
        entry_full,
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Builder {
        f: Func,
        insts: Vec<Inst>,
    }

    impl Builder {
        fn new() -> Self {
            Builder {
                f: Func {
                    entry: BlockId(0),
                    blocks: BTreeMap::new(),
                    types: vec![],
                },
                insts: vec![],
            }
        }

        fn param(&mut self, ty: Ty) -> ValueId {
            self.f.value(ty)
        }

        fn core(&mut self, ty: Ty, op: Op) -> ValueId {
            let value = self.f.value(ty);
            self.insts.push(Inst::Core { value, ty, op });
            value
        }

        fn packet(&mut self, op: PacketOp, input: ValueId) -> ValueId {
            let ty = if op == PacketOp::Ballot {
                Ty::I32
            } else {
                Ty::I1
            };
            let output = self.f.value(ty);
            self.insts.push(Inst::Packet { op, input, output });
            output
        }

        fn lane_bit(&mut self, word: ValueId) -> ValueId {
            let lane = self.core(Ty::I32, Op::Env(Env::LaneId));
            let shifted = self.core(Ty::I32, Op::Int(IntOp::LShr, word, lane));
            self.core(Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted))
        }

        fn restore(&mut self, bit: ValueId) -> ValueId {
            let valid = self.core(Ty::I1, Op::Env(Env::ValidLane));
            self.core(Ty::I1, Op::Int(IntOp::And, bit, valid))
        }

        fn block(&mut self, id: usize, params: Vec<(ValueId, Ty)>, term: Term) {
            let insts = std::mem::take(&mut self.insts);
            self.f.blocks.insert(
                BlockId(id),
                Block {
                    params,
                    insts,
                    term,
                },
            );
        }

        fn masks(&self) -> Masks {
            let constants = super::super::constant::constants(&self.f);
            analyze(
                &crate::rdna_spmd::targets::rdna4::registry(),
                &self.f,
                0,
                &constants,
                4,
                false,
            )
        }
    }

    #[test]
    fn a_restore_rebuilt_from_a_word_the_loop_changes_keeps_the_predicate() {
        let mut b = Builder::new();
        let (e0, x, m0) = (b.param(Ty::I1), b.param(Ty::I32), b.param(Ty::I32));
        b.block(
            0,
            vec![(e0, Ty::I1), (x, Ty::I32), (m0, Ty::I32)],
            Term::Br(Edge {
                dst: BlockId(1),
                args: vec![e0, x, m0],
            }),
        );
        let (e1, previous, m) = (b.param(Ty::I1), b.param(Ty::I32), b.param(Ty::I32));
        let bit = b.lane_bit(m);
        let restored = b.restore(bit);
        let one = b.core(Ty::I32, Op::Const(Ty::I32, 1));
        let next = b.core(Ty::I32, Op::Int(IntOp::Add, previous, one));
        let written = b.core(Ty::I32, Op::Select(restored, next, previous));
        let eight = b.core(Ty::I32, Op::Const(Ty::I32, 8));
        let below = b.core(Ty::I1, Op::Cmp(IntPred::Ult, written, eight));
        let taken = b.core(Ty::I1, Op::Int(IntOp::And, below, restored));
        let narrowed = b.restore(taken);
        let flip = b.core(Ty::I32, Op::Const(Ty::I32, 0xf));
        let flipped = b.core(Ty::I32, Op::Int(IntOp::Xor, m, flip));
        let again = b.packet(PacketOp::Any, narrowed);
        b.block(
            1,
            vec![(e1, Ty::I1), (previous, Ty::I32), (m, Ty::I32)],
            Term::CondBr {
                cond: again,
                yes: Edge {
                    dst: BlockId(1),
                    args: vec![narrowed, written, flipped],
                },
                no: Edge {
                    dst: BlockId(2),
                    args: vec![narrowed],
                },
            },
        );
        let e2 = b.param(Ty::I1);
        b.block(2, vec![(e2, Ty::I1)], Term::Ret(vec![]));
        let masks = b.masks();
        assert!(masks.predicated[written.0].is_some());
        assert!(masks.observed[written.0]);
        assert!(!masks.guarded[written.0]);
    }

    #[test]
    fn a_restore_from_a_bit_saved_in_the_previous_iteration_keeps_the_predicate() {
        let mut b = Builder::new();
        let (e0, x) = (b.param(Ty::I1), b.param(Ty::I32));
        let zero = b.core(Ty::I32, Op::Const(Ty::I32, 0));
        b.block(
            0,
            vec![(e0, Ty::I1), (x, Ty::I32)],
            Term::Br(Edge {
                dst: BlockId(1),
                args: vec![e0, x, zero],
            }),
        );
        let (e1, previous, earlier) = (b.param(Ty::I1), b.param(Ty::I32), b.param(Ty::I32));
        let one = b.core(Ty::I32, Op::Const(Ty::I32, 1));
        let next = b.core(Ty::I32, Op::Int(IntOp::Add, previous, one));
        let written = b.core(Ty::I32, Op::Select(e1, next, previous));
        let eight = b.core(Ty::I32, Op::Const(Ty::I32, 8));
        let below = b.core(Ty::I1, Op::Cmp(IntPred::Ult, written, eight));
        let taken = b.core(Ty::I1, Op::Int(IntOp::And, below, e1));
        let narrowed = b.restore(taken);
        let saved = b.packet(PacketOp::Ballot, e1);
        let bit = b.lane_bit(earlier);
        b.restore(bit);
        let read = b.core(Ty::I32, Op::Int(IntOp::Add, written, one));
        let again = b.packet(PacketOp::Any, narrowed);
        b.block(
            1,
            vec![(e1, Ty::I1), (previous, Ty::I32), (earlier, Ty::I32)],
            Term::CondBr {
                cond: again,
                yes: Edge {
                    dst: BlockId(1),
                    args: vec![narrowed, read, saved],
                },
                no: Edge {
                    dst: BlockId(2),
                    args: vec![narrowed],
                },
            },
        );
        let e2 = b.param(Ty::I1);
        b.block(2, vec![(e2, Ty::I1)], Term::Ret(vec![]));
        let masks = b.masks();
        assert!(masks.predicated[written.0].is_some());
        assert!(masks.observed[written.0]);
        assert!(!masks.guarded[written.0]);
    }

    fn loop_restoring_from(saved_from_loop: bool) -> (Func, ValueId, ValueId, ValueId) {
        let mut f = Func {
            entry: BlockId(0),
            blocks: BTreeMap::new(),
            types: vec![],
        };
        let exec = f.value(Ty::I1);
        let x = f.value(Ty::I32);
        let outer = f.value(Ty::I32);
        let core = |f: &mut Func, insts: &mut Vec<Inst>, ty, op| {
            let v = f.value(ty);
            insts.push(Inst::Core { value: v, ty, op });
            v
        };
        let mut entry = Vec::new();
        let bound = core(&mut f, &mut entry, Ty::I32, Op::Const(Ty::I32, 8));
        let inside = core(&mut f, &mut entry, Ty::I1, Op::Cmp(IntPred::Ult, x, bound));
        let checked = core(
            &mut f,
            &mut entry,
            Ty::I1,
            Op::Int(IntOp::And, inside, exec),
        );
        let word = f.value(Ty::I32);
        entry.push(Inst::Packet {
            op: PacketOp::Ballot,
            input: checked,
            output: word,
        });
        let exec_word = f.value(Ty::I32);
        entry.push(Inst::Packet {
            op: PacketOp::Ballot,
            input: exec,
            output: exec_word,
        });
        let both = core(
            &mut f,
            &mut entry,
            Ty::I32,
            Op::Int(IntOp::And, word, exec_word),
        );
        let lane = core(&mut f, &mut entry, Ty::I32, Op::Env(Env::LaneId));
        let shifted = core(
            &mut f,
            &mut entry,
            Ty::I32,
            Op::Int(IntOp::LShr, both, lane),
        );
        let bit = core(
            &mut f,
            &mut entry,
            Ty::I1,
            Op::Convert(Cvt::Trunc, Ty::I1, shifted),
        );
        let valid = core(&mut f, &mut entry, Ty::I1, Op::Env(Env::ValidLane));
        let narrowed = core(&mut f, &mut entry, Ty::I1, Op::Int(IntOp::And, bit, valid));
        f.blocks.insert(
            BlockId(0),
            Block {
                params: vec![(exec, Ty::I1), (x, Ty::I32), (outer, Ty::I32)],
                insts: entry,
                term: Term::Br(Edge {
                    dst: BlockId(1),
                    args: vec![narrowed, x, outer],
                }),
            },
        );
        let e1 = f.value(Ty::I1);
        let y = f.value(Ty::I32);
        let o1 = f.value(Ty::I32);
        let mut header = Vec::new();
        let k = core(&mut f, &mut header, Ty::I32, Op::Const(Ty::I32, 5));
        let written = core(&mut f, &mut header, Ty::I32, Op::Select(e1, k, y));
        let saved = f.value(Ty::I32);
        header.push(Inst::Packet {
            op: PacketOp::Ballot,
            input: e1,
            output: saved,
        });
        let odd = core(
            &mut f,
            &mut header,
            Ty::I32,
            Op::Int(IntOp::And, written, k),
        );
        let cond = core(&mut f, &mut header, Ty::I1, Op::Cmp(IntPred::Ne, odd, k));
        let taken = core(&mut f, &mut header, Ty::I1, Op::Int(IntOp::And, cond, e1));
        let body_word = f.value(Ty::I32);
        header.push(Inst::Packet {
            op: PacketOp::Ballot,
            input: taken,
            output: body_word,
        });
        let both1 = core(
            &mut f,
            &mut header,
            Ty::I32,
            Op::Int(IntOp::And, body_word, saved),
        );
        let lane1 = core(&mut f, &mut header, Ty::I32, Op::Env(Env::LaneId));
        let sh1 = core(
            &mut f,
            &mut header,
            Ty::I32,
            Op::Int(IntOp::LShr, both1, lane1),
        );
        let bit1 = core(
            &mut f,
            &mut header,
            Ty::I1,
            Op::Convert(Cvt::Trunc, Ty::I1, sh1),
        );
        let valid1 = core(&mut f, &mut header, Ty::I1, Op::Env(Env::ValidLane));
        let body_exec = core(
            &mut f,
            &mut header,
            Ty::I1,
            Op::Int(IntOp::And, bit1, valid1),
        );
        let restore_word = if saved_from_loop { saved } else { o1 };
        f.blocks.insert(
            BlockId(1),
            Block {
                params: vec![(e1, Ty::I1), (y, Ty::I32), (o1, Ty::I32)],
                insts: header,
                term: Term::Br(Edge {
                    dst: BlockId(2),
                    args: vec![body_exec, written, restore_word],
                }),
            },
        );
        let e2 = f.value(Ty::I1);
        let z = f.value(Ty::I32);
        let s2 = f.value(Ty::I32);
        let mut body = Vec::new();
        let k2 = core(&mut f, &mut body, Ty::I32, Op::Const(Ty::I32, 9));
        let written2 = core(&mut f, &mut body, Ty::I32, Op::Select(e2, k2, z));
        let done = core(
            &mut f,
            &mut body,
            Ty::I1,
            Op::Cmp(IntPred::Eq, written2, k2),
        );
        let done_masked = core(&mut f, &mut body, Ty::I1, Op::Int(IntOp::And, done, e2));
        let done_word = f.value(Ty::I32);
        body.push(Inst::Packet {
            op: PacketOp::Ballot,
            input: done_masked,
            output: done_word,
        });
        let joined = core(
            &mut f,
            &mut body,
            Ty::I32,
            Op::Int(IntOp::Or, done_word, s2),
        );
        let lane2 = core(&mut f, &mut body, Ty::I32, Op::Env(Env::LaneId));
        let sh2 = core(
            &mut f,
            &mut body,
            Ty::I32,
            Op::Int(IntOp::LShr, joined, lane2),
        );
        let bit2 = core(
            &mut f,
            &mut body,
            Ty::I1,
            Op::Convert(Cvt::Trunc, Ty::I1, sh2),
        );
        let valid2 = core(&mut f, &mut body, Ty::I1, Op::Env(Env::ValidLane));
        let restored = core(&mut f, &mut body, Ty::I1, Op::Int(IntOp::And, bit2, valid2));
        let again = f.value(Ty::I1);
        body.push(Inst::Packet {
            op: PacketOp::Any,
            input: restored,
            output: again,
        });
        f.blocks.insert(
            BlockId(2),
            Block {
                params: vec![(e2, Ty::I1), (z, Ty::I32), (s2, Ty::I32)],
                insts: body,
                term: Term::CondBr {
                    cond: again,
                    yes: Edge {
                        dst: BlockId(1),
                        args: vec![restored, written2, s2],
                    },
                    no: Edge {
                        dst: BlockId(3),
                        args: vec![restored, written2],
                    },
                },
            },
        );
        let e3 = f.value(Ty::I1);
        let w3 = f.value(Ty::I32);
        let mut exit = Vec::new();
        let addr = core(&mut f, &mut exit, Ty::I32, Op::Const(Ty::I32, 64));
        let semantics = super::super::super::ir::MemorySemantics {
            scope: super::super::super::ir::Scope::WorkItem,
            ordering: super::super::super::ir::Ordering::Relaxed,
            cache_policy: super::super::super::ir::CachePolicy::Temporal,
            volatile: false,
            deferred_scope: false,
        };
        exit.push(Inst::Effect {
            provenance: 0,
            op: EffectOp::Memory {
                space: super::super::super::ir::Space::Lds,
                op: super::super::super::ir::MemoryOp::Store(super::super::super::ir::MemSize::B32),
                semantics,
            },
            inputs: vec![addr, w3, e3],
            outputs: vec![],
        });
        f.blocks.insert(
            BlockId(3),
            Block {
                params: vec![(e3, Ty::I1), (w3, Ty::I32)],
                insts: exit,
                term: Term::Ret(vec![]),
            },
        );
        (f, written, written2, e1)
    }

    #[test]
    fn a_write_under_an_exec_that_every_later_restore_stays_within_needs_no_predicate() {
        let (f, written, written2, e1) = loop_restoring_from(true);
        let constants = super::super::constant::constants(&f);
        let masks = analyze(
            &crate::rdna_spmd::targets::rdna4::registry(),
            &f,
            0,
            &constants,
            4,
            false,
        );
        assert_eq!(masks.reactivation.len(), 1);
        assert!(!masks.full[e1.0]);
        assert!(masks.guarded[written.0]);
        assert!(!masks.observed[written.0]);
        assert!(!masks.guarded[written2.0]);
        assert!(masks.observed[written2.0]);
    }

    #[test]
    fn a_restore_from_a_mask_saved_before_the_exec_keeps_the_predicate() {
        let (f, written, written2, e1) = loop_restoring_from(false);
        let constants = super::super::constant::constants(&f);
        let masks = analyze(
            &crate::rdna_spmd::targets::rdna4::registry(),
            &f,
            0,
            &constants,
            4,
            false,
        );
        assert_eq!(masks.reactivation.len(), 1);
        assert!(!masks.full[e1.0]);
        assert!(!masks.guarded[written.0]);
        assert!(masks.observed[written.0]);
        assert!(!masks.guarded[written2.0]);
    }

    #[test]
    fn wave_level_queries_never_prove_a_lane_active_and_empty_exec_edges_carry() {
        fn build(wave: bool) -> (Func, ValueId) {
            let mut f = Func {
                entry: BlockId(0),
                blocks: BTreeMap::new(),
                types: vec![],
            };
            let exec = f.value(Ty::I1);
            let saved = f.value(Ty::I1);
            let flag = f.value(Ty::I1);
            let narrowed = f.value(Ty::I1);
            let any = f.value(Ty::I1);
            let query = if wave {
                Inst::Effect {
                    provenance: 7,
                    op: EffectOp::Wave(WaveOp::Any),
                    inputs: vec![narrowed],
                    outputs: vec![(any, Ty::I1)],
                }
            } else {
                Inst::Packet {
                    op: PacketOp::Any,
                    input: narrowed,
                    output: any,
                }
            };
            f.blocks.insert(
                BlockId(0),
                Block {
                    params: vec![(exec, Ty::I1), (saved, Ty::I1), (flag, Ty::I1)],
                    insts: vec![
                        Inst::Core {
                            value: narrowed,
                            ty: Ty::I1,
                            op: Op::Int(IntOp::And, flag, exec),
                        },
                        query,
                    ],
                    term: Term::CondBr {
                        cond: any,
                        yes: Edge {
                            dst: BlockId(1),
                            args: vec![narrowed, saved],
                        },
                        no: Edge {
                            dst: BlockId(2),
                            args: vec![narrowed, saved],
                        },
                    },
                },
            );
            let e1 = f.value(Ty::I1);
            let s1 = f.value(Ty::I1);
            f.blocks.insert(
                BlockId(1),
                Block {
                    params: vec![(e1, Ty::I1), (s1, Ty::I1)],
                    insts: vec![],
                    term: Term::Ret(vec![]),
                },
            );
            let e2 = f.value(Ty::I1);
            let s2 = f.value(Ty::I1);
            let valid = f.value(Ty::I1);
            let restored = f.value(Ty::I1);
            f.blocks.insert(
                BlockId(2),
                Block {
                    params: vec![(e2, Ty::I1), (s2, Ty::I1)],
                    insts: vec![
                        Inst::Core {
                            value: valid,
                            ty: Ty::I1,
                            op: Op::Env(Env::ValidLane),
                        },
                        Inst::Core {
                            value: restored,
                            ty: Ty::I1,
                            op: Op::Int(IntOp::And, s2, valid),
                        },
                    ],
                    term: Term::Ret(vec![]),
                },
            );
            (f, narrowed)
        }
        for wave in [true, false] {
            let (f, _) = build(wave);
            let constants = super::super::constant::constants(&f);
            let facts = exec(&f, 0, &constants, 1, true);
            assert!(
                facts.active_at(BlockId(0), 0),
                "the launched lane is active at entry"
            );
            assert!(
                !facts.active_at(BlockId(0), 1),
                "a compare masked by EXEC narrows it without a valid-lane operand"
            );
            assert_eq!(
                facts.active_at(BlockId(1), 0),
                !wave,
                "a wave's Any does not speak for this lane, a packet's Any does"
            );
            assert!(
                !facts.active_at(BlockId(2), 0) && !facts.nonempty_at(BlockId(2), 0),
                "the exec-zero edge carries the empty EXEC"
            );
            assert!(
                !facts.active_at(BlockId(2), 2),
                "a restore from a saved bit of unknown activeness proves nothing"
            );
        }
    }

    #[test]
    fn a_bit_saved_from_the_entry_exec_restores_an_active_lane() {
        let mut f = Func {
            entry: BlockId(0),
            blocks: BTreeMap::new(),
            types: vec![],
        };
        let e0 = f.value(Ty::I1);
        let saved = f.value(Ty::I1);
        f.blocks.insert(
            BlockId(0),
            Block {
                params: vec![(e0, Ty::I1)],
                insts: vec![Inst::Core {
                    value: saved,
                    ty: Ty::I1,
                    op: Op::Convert(Cvt::Bitcast, Ty::I1, e0),
                }],
                term: Term::Br(Edge {
                    dst: BlockId(1),
                    args: vec![e0, saved],
                }),
            },
        );
        let e1 = f.value(Ty::I1);
        let s1 = f.value(Ty::I1);
        let valid = f.value(Ty::I1);
        let restored = f.value(Ty::I1);
        f.blocks.insert(
            BlockId(1),
            Block {
                params: vec![(e1, Ty::I1), (s1, Ty::I1)],
                insts: vec![
                    Inst::Core {
                        value: valid,
                        ty: Ty::I1,
                        op: Op::Env(Env::ValidLane),
                    },
                    Inst::Core {
                        value: restored,
                        ty: Ty::I1,
                        op: Op::Int(IntOp::And, s1, valid),
                    },
                ],
                term: Term::Ret(vec![]),
            },
        );
        let constants = super::super::constant::constants(&f);
        let facts = exec(&f, 0, &constants, 1, true);
        assert!(
            facts.active_at(BlockId(1), 2),
            "the saved bit was taken while the lane was active"
        );
    }

    #[test]
    fn narrowing_exec_keeps_predicated_writes_guarded_and_widening_does_not() {
        let mut f = Func {
            entry: BlockId(0),
            blocks: BTreeMap::new(),
            types: vec![],
        };
        let exec = f.value(Ty::I1);
        let x = f.value(Ty::I32);
        let saved = f.value(Ty::I32);
        let mut insts = Vec::new();
        let mut core = |f: &mut Func, ty, op| {
            let v = f.value(ty);
            insts.push(Inst::Core { value: v, ty, op });
            v
        };
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
        f.blocks.insert(
            BlockId(0),
            Block {
                params: vec![(exec, Ty::I1), (x, Ty::I32), (saved, Ty::I32)],
                insts,
                term: Term::Br(Edge {
                    dst: BlockId(1),
                    args: vec![restored, sum, second],
                }),
            },
        );
        let e1 = f.value(Ty::I1);
        let p = f.value(Ty::I32);
        let q = f.value(Ty::I32);
        let always = f.value(Ty::I1);
        let semantics = super::super::super::ir::MemorySemantics {
            scope: super::super::super::ir::Scope::WorkItem,
            ordering: super::super::super::ir::Ordering::Relaxed,
            cache_policy: super::super::super::ir::CachePolicy::Temporal,
            volatile: false,
            deferred_scope: false,
        };
        f.blocks.insert(
            BlockId(1),
            Block {
                params: vec![(e1, Ty::I1), (p, Ty::I32), (q, Ty::I32)],
                insts: vec![
                    Inst::Core {
                        value: always,
                        ty: Ty::I1,
                        op: Op::Const(Ty::I1, 1),
                    },
                    Inst::Effect {
                        provenance: 0,
                        op: EffectOp::Memory {
                            space: super::super::super::ir::Space::Lds,
                            op: super::super::super::ir::MemoryOp::Store(
                                super::super::super::ir::MemSize::B32,
                            ),
                            semantics,
                        },
                        inputs: vec![q, p, always],
                        outputs: vec![],
                    },
                ],
                term: Term::Ret(vec![]),
            },
        );
        let constants = super::super::constant::constants(&f);
        let masks = analyze(
            &crate::rdna_spmd::targets::rdna4::registry(),
            &f,
            0,
            &constants,
            16,
            false,
        );
        assert_eq!(*masks.reactivation, vec![(BlockId(0), 10)]);
        assert!(!masks.guarded[first.0]);
        assert!(!masks.guarded[second.0]);
        assert!(masks.guarded[third.0]);
        let masks = analyze(
            &crate::rdna_spmd::targets::rdna4::registry(),
            &f,
            0,
            &constants,
            16,
            true,
        );
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
