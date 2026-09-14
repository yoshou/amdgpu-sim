use super::super::ir::{
    BlockId, Cvt, Edge, EffectOp, Env, Func, Inst, IntOp, Op, PacketOp, Ty, ValueId, WaveOp,
};
use super::dataflow::{for_each_output, Cfg, Forward, Lattice};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn valid_masked(defs: &[Option<Op>], value: ValueId) -> Option<ValueId> {
    let valid = |v: ValueId| matches!(defs[v.0], Some(Op::Env(Env::ValidLane)));
    match defs[value.0] {
        Some(Op::Int(IntOp::And, a, b)) if valid(b) => Some(a),
        Some(Op::Int(IntOp::And, a, b)) if valid(a) => Some(b),
        _ => None,
    }
}

pub(super) fn lane_word(defs: &[Option<Op>], value: ValueId) -> Option<ValueId> {
    let Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) = defs[value.0] else {
        return None;
    };
    match defs[shifted.0] {
        Some(Op::Int(IntOp::LShr, word, lane))
            if matches!(defs[lane.0], Some(Op::Env(Env::LaneId))) =>
        {
            Some(word)
        }
        _ => None,
    }
}

pub(super) fn ballot_of(f: &Func) -> Vec<Option<ValueId>> {
    let mut out = vec![None; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            match inst {
                Inst::Packet {
                    op: PacketOp::Ballot,
                    input,
                    output,
                } => out[output.0] = Some(*input),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Ballot),
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

fn sole_incoming(f: &Func) -> Vec<Option<ValueId>> {
    let mut incoming: BTreeMap<BlockId, Vec<(BlockId, &Edge)>> = BTreeMap::new();
    for (&id, block) in &f.blocks {
        for edge in block.term.edges() {
            incoming.entry(edge.dst).or_default().push((id, edge));
        }
    }
    let mut out = vec![None; f.types.len()];
    for (&id, block) in &f.blocks {
        if id == f.entry {
            continue;
        }
        if let Some([(source, edge)]) = incoming.get(&id).map(Vec::as_slice) {
            if *source != id {
                for (&(param, _), &arg) in block.params.iter().zip(&edge.args) {
                    out[param.0] = Some(arg);
                }
            }
        }
    }
    out
}

#[derive(Clone, Copy)]
pub(super) enum Rule {
    Always,
    Never,
    Same(ValueId),
    Either(ValueId, ValueId),
    Both(ValueId, ValueId),
}

impl Rule {
    fn operands(self) -> impl Iterator<Item = ValueId> {
        let (first, second) = match self {
            Rule::Same(a) => (Some(a), None),
            Rule::Either(a, b) | Rule::Both(a, b) => (Some(a), Some(b)),
            Rule::Always | Rule::Never => (None, None),
        };
        first.into_iter().chain(second)
    }

    fn holds(self, within: impl Fn(ValueId) -> bool) -> bool {
        match self {
            Rule::Always => true,
            Rule::Never => false,
            Rule::Same(a) => within(a),
            Rule::Either(a, b) => within(a) || within(b),
            Rule::Both(a, b) => within(a) && within(b),
        }
    }
}

#[derive(PartialEq)]
pub(super) struct Lanes {
    pub defs: Vec<Option<Op>>,
    ballots: Vec<Option<ValueId>>,
    zero: Vec<bool>,
    sole: Vec<Option<ValueId>>,
}

impl Lanes {
    pub fn new(f: &Func, constants: &[Option<u64>]) -> Self {
        Lanes {
            defs: f.definitions(),
            ballots: ballot_of(f),
            zero: constants.iter().map(|k| *k == Some(0)).collect(),
            sole: sole_incoming(f),
        }
    }

    pub fn valid_masked(&self, value: ValueId) -> Option<ValueId> {
        valid_masked(&self.defs, value)
    }

    fn rule(&self, value: ValueId) -> Rule {
        if self.zero[value.0] {
            return Rule::Always;
        }
        if let Some(bit) = self.ballots[value.0] {
            return Rule::Same(bit);
        }
        if let Some(word) = lane_word(&self.defs, value) {
            return Rule::Same(word);
        }
        match self.defs[value.0] {
            Some(Op::Convert(Cvt::Bitcast, _, a)) => Rule::Same(a),
            Some(Op::Int(IntOp::And, a, b)) => Rule::Either(a, b),
            Some(Op::Int(IntOp::Or | IntOp::Xor, a, b)) | Some(Op::Select(_, a, b)) => {
                Rule::Both(a, b)
            }
            _ => Rule::Never,
        }
    }

    fn equal(&self, value: ValueId) -> Option<ValueId> {
        match self.rule(value) {
            Rule::Same(source) => Some(source),
            _ => self.valid_masked(value).or(self.sole[value.0]),
        }
    }

    pub fn class(&self, exec: ValueId) -> BTreeSet<ValueId> {
        let mut class = BTreeSet::new();
        let mut pending = vec![exec];
        while let Some(value) = pending.pop() {
            if class.insert(value) {
                pending.extend(self.equal(value));
            }
        }
        class
    }

    pub fn within(&self, value: ValueId, class: &BTreeSet<ValueId>) -> bool {
        let mut known: BTreeMap<ValueId, bool> = class.iter().map(|&v| (v, true)).collect();
        let mut pending = vec![value];
        while let Some(&top) = pending.last() {
            if known.contains_key(&top) {
                pending.pop();
                continue;
            }
            let rule = self.rule(top);
            let unknown: Vec<ValueId> = rule
                .operands()
                .filter(|operand| !known.contains_key(operand))
                .collect();
            if unknown.is_empty() {
                pending.pop();
                let holds = rule.holds(|operand| known[&operand]);
                known.insert(top, holds);
            } else {
                pending.extend(unknown);
            }
        }
        known[&value]
    }
}

#[derive(Clone, PartialEq)]
enum Flow {
    Unreached,
    Within(Vec<u64>),
}

impl Lattice for Flow {
    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Flow::Unreached, flow) | (flow, Flow::Unreached) => flow.clone(),
            (Flow::Within(a), Flow::Within(b)) => {
                Flow::Within(a.iter().zip(b).map(|(x, y)| x & y).collect())
            }
        }
    }
}

struct Step {
    index: usize,
    value: ValueId,
    rule: Rule,
    exec: bool,
}

pub(super) struct Region<'a, 'f> {
    lanes: &'a Lanes,
    cfg: &'a Cfg<'f>,
    execs: Vec<ValueId>,
    homes: BTreeMap<ValueId, (usize, usize)>,
    steps: Vec<Vec<Step>>,
    slot: Vec<Option<usize>>,
    words: usize,
}

impl<'a, 'f> Region<'a, 'f> {
    pub fn new(
        f: &Func,
        lanes: &'a Lanes,
        cfg: &'a Cfg<'f>,
        chain: &BTreeMap<BlockId, Vec<(usize, ValueId)>>,
    ) -> Self {
        let mut homes = BTreeMap::new();
        let mut updates = BTreeSet::new();
        for (at, id) in cfg.ids.iter().enumerate() {
            for (k, &(from, exec)) in chain[id].iter().enumerate() {
                homes.insert(exec, (at, from));
                if k > 0 {
                    updates.insert(exec);
                }
            }
        }
        let execs: Vec<ValueId> = cfg.ids.iter().map(|id| chain[id][0].1).collect();
        let mut params = vec![None; f.types.len()];
        for (at, block) in cfg.blocks.iter().enumerate() {
            for (position, &(param, _)) in block.params.iter().enumerate() {
                params[param.0] = Some((at, position));
            }
        }
        let mut slot = vec![None; f.types.len()];
        let mut slots = 0;
        let mut pending: Vec<ValueId> = homes.keys().copied().collect();
        while let Some(value) = pending.pop() {
            if slot[value.0].is_some() {
                continue;
            }
            slot[value.0] = Some(slots);
            slots += 1;
            pending.extend(lanes.rule(value).operands());
            if let Some((at, position)) = params[value.0] {
                pending.extend(cfg.incoming[at].iter().map(|(_, edge)| edge.args[position]));
            }
        }
        let steps = cfg
            .blocks
            .iter()
            .map(|block| {
                let mut steps = Vec::new();
                for (index, inst) in block.insts.iter().enumerate() {
                    for_each_output(inst, |value| {
                        if slot[value.0].is_some() {
                            steps.push(Step {
                                index,
                                value,
                                rule: lanes.rule(value),
                                exec: updates.contains(&value),
                            });
                        }
                    });
                }
                steps
            })
            .collect();
        Region {
            lanes,
            cfg,
            execs,
            homes,
            steps,
            slot,
            words: slots.div_ceil(64),
        }
    }

    pub fn never_widens(&self, exec: ValueId) -> bool {
        let Some(&home) = self.homes.get(&exec) else {
            return false;
        };
        let mut seeds = vec![0u64; self.words];
        for value in self.lanes.class(exec) {
            self.set(&mut seeds, value, true);
        }
        let widened = Cell::new(false);
        let edge = |_: usize, edge: &Edge, exit: &Flow| {
            if widened.get() {
                return Flow::Unreached;
            }
            self.enter(edge, exit)
        };
        let transfer = |at: usize, entry: &Flow| {
            if widened.get() {
                return Flow::Unreached;
            }
            self.walk(at, entry, home, &seeds, &mut |within| {
                if !within {
                    widened.set(true);
                }
            })
        };
        Forward {
            cfg: self.cfg,
            start: Flow::Unreached,
            edge: &edge,
            transfer: &transfer,
        }
        .solve();
        !widened.get()
    }

    fn walk(
        &self,
        at: usize,
        entry: &Flow,
        home: (usize, usize),
        seeds: &[u64],
        visit: &mut dyn FnMut(bool),
    ) -> Flow {
        let mut bits = match entry {
            Flow::Within(bits) => Some(bits.clone()),
            Flow::Unreached if at == home.0 => None,
            Flow::Unreached => return Flow::Unreached,
        };
        if let Some(bits) = &bits {
            visit(self.has(bits, self.execs[at]));
        }
        let mut seeded = at != home.0;
        for step in &self.steps[at] {
            if !seeded && step.index >= home.1 {
                bits = Some(self.seed(bits, seeds));
                seeded = true;
            }
            if let Some(bits) = &mut bits {
                let within = step.rule.holds(|operand| self.has(bits, operand));
                self.set(bits, step.value, within);
                if step.exec {
                    visit(within);
                }
            }
        }
        if !seeded {
            bits = Some(self.seed(bits, seeds));
        }
        bits.map_or(Flow::Unreached, Flow::Within)
    }

    fn seed(&self, bits: Option<Vec<u64>>, seeds: &[u64]) -> Vec<u64> {
        match bits {
            Some(bits) => bits.iter().zip(seeds).map(|(x, y)| x & y).collect(),
            None => seeds.to_vec(),
        }
    }

    fn enter(&self, edge: &Edge, exit: &Flow) -> Flow {
        let Flow::Within(bits) = exit else {
            return Flow::Unreached;
        };
        let dst = self.cfg.blocks[self.cfg.index[edge.dst.0]];
        let mut entered = bits.clone();
        for (&(param, _), &arg) in dst.params.iter().zip(&edge.args) {
            if self.slot[param.0].is_some() {
                self.set(&mut entered, param, self.has(bits, arg));
            }
        }
        Flow::Within(entered)
    }

    fn has(&self, bits: &[u64], value: ValueId) -> bool {
        self.lanes.zero[value.0]
            || self.slot[value.0].is_some_and(|slot| bits[slot / 64] >> (slot % 64) & 1 != 0)
    }

    fn set(&self, bits: &mut [u64], value: ValueId, within: bool) {
        if let Some(slot) = self.slot[value.0] {
            let bit = 1u64 << (slot % 64);
            if within {
                bits[slot / 64] |= bit;
            } else {
                bits[slot / 64] &= !bit;
            }
        }
    }
}
