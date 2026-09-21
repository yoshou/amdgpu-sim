use super::super::ir::{Cvt, EffectOp, IntOp, Op, Ty, ValueId, WaveOp, *};
use super::dataflow::{Cfg, Lattice, Sparse};
use super::lanes::{ballot_of, lane_word, valid_masked, Lanes};
use super::{Analyses, Analysis, Constants};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

impl Analysis for Predication {
    type Result = Predication;
    const NAME: &'static str = "predication";
    fn compute(f: &Func, analyses: &Analyses) -> Self {
        let constants = analyses.get::<Constants>(f);
        predication(f, analyses.context().exec_index, &constants)
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
