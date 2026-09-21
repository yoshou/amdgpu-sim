use super::super::ir::{
    BlockId, Cvt, Edge, EffectOp, Env, Func, Inst, IntOp, Op, PacketOp, Ty, ValueId, WaveOp,
};
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