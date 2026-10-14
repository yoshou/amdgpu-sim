use super::atoms::{exists, Atom, Atoms, Choice};
use super::patterns::lane_test;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site, Use};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

#[derive(Default, Clone, PartialEq)]
pub struct Kept {
    pub queries: BTreeSet<ValueId>,

    pub words: BTreeSet<ValueId>,

    pub meets: BTreeSet<usize>,
}

impl Kept {
    pub fn insert(&mut self, choice: Choice) {
        match choice {
            Choice::Query(v) => self.queries.insert(v),
            Choice::Word(v) => self.words.insert(v),
            Choice::Meet(i) => self.meets.insert(i),
        };
    }

    pub fn remove(&mut self, choice: Choice) {
        match choice {
            Choice::Query(v) => self.queries.remove(&v),
            Choice::Word(v) => self.words.remove(&v),
            Choice::Meet(i) => self.meets.remove(&i),
        };
    }

    pub fn choices(&self) -> BTreeSet<Choice> {
        let queries = self.queries.iter().map(|&v| Choice::Query(v));
        let words = self.words.iter().map(|&v| Choice::Word(v));
        let meets = self.meets.iter().map(|&i| Choice::Meet(i));
        queries.chain(words).chain(meets).collect()
    }
}

enum Choices {
    Fixed {
        kept: BTreeSet<Choice>,
        read: Option<Vec<bool>>,
    },

    Open {
        whole: Vec<Bdd>,
        live: Vec<bool>,
        read: Vec<bool>,
        all_local: Bdd,
    },
}

pub(super) struct Policy {
    choices: Choices,
    listed: Vec<Choice>,
}

impl Policy {
    pub(super) fn fixed(kept: &BTreeSet<Choice>, listed: &[Choice], read: Option<Vec<bool>>) -> Self {
        Self {
            choices: Choices::Fixed { kept: kept.clone(), read },
            listed: listed.to_vec(),
        }
    }

    pub(super) fn open(f: &Func, facts: &Facts, listed: &[Choice], atoms: &mut Atoms, m: &mut Manager) -> Self {
        let mut whole: Vec<Bdd> = facts
            .materialized
            .iter()
            .map(|&v| Manager::constant(v))
            .collect();
        let mut pending = Vec::new();
        let mut all_local = Bdd::TRUE;
        for &c in listed {
            let local = atoms.atom(m, Atom::Marker(c));
            all_local = m.and(all_local, local);
            if let Choice::Word(v) = c {
                let kept = m.not(local);
                whole[v.0] = m.or(whole[v.0], kept);
                pending.push(v);
            }
        }
        while let Some(v) = pending.pop() {
            for a in facts.materialized_sources(f, v) {
                if !facts.lane_word[a.0] {
                    continue;
                }
                let joined = m.or(whole[a.0], whole[v.0]);
                if joined != whole[a.0] {
                    whole[a.0] = joined;
                    pending.push(a);
                }
            }
        }
        for &id in &facts.order {
            for inst in &f.blocks[&id].insts {
                let Inst::Core { value, .. } = *inst else {
                    continue;
                };
                if !facts.lane_word[value.0] {
                    continue;
                }
                let words: Vec<ValueId> = facts
                    .materialized_sources(f, value)
                    .into_iter()
                    .filter(|a| facts.lane_word[a.0])
                    .collect();
                if words.is_empty() {
                    continue;
                }
                let mut all = Bdd::TRUE;
                for a in words {
                    all = m.and(all, whole[a.0]);
                }
                whole[value.0] = m.or(whole[value.0], all);
            }
        }
        Self {
            choices: Choices::Open {
                whole,
                live: live_values(f, facts),
                read: read_values(f, facts),
                all_local,
            },
            listed: listed.to_vec(),
        }
    }

    pub(super) fn keep(&mut self, kept: &BTreeSet<Choice>) {
        if let Choices::Fixed { kept: fixed, .. } = &mut self.choices {
            let conversions = |s: &BTreeSet<Choice>| s.iter().filter(|c| !matches!(c, Choice::Meet(_))).copied().collect::<Vec<_>>();
            assert!(conversions(fixed) == conversions(kept), "keep changes only which meetings are kept");
            *fixed = kept.clone();
        }
    }

    pub(super) fn is_open(&self) -> bool {
        matches!(self.choices, Choices::Open { .. })
    }

    pub(super) fn local(&self, atoms: &mut Atoms, m: &mut Manager, c: Choice) -> Bdd {
        if let Choices::Fixed { kept, .. } = &self.choices {
            return Manager::constant(!kept.contains(&c));
        }
        if atoms.marks(c) {
            atoms.atom(m, Atom::Marker(c))
        } else {
            Bdd::TRUE
        }
    }

    pub(super) fn materialized(&self, facts: &Facts, v: ValueId) -> Bdd {
        match &self.choices {
            Choices::Fixed { .. } => Manager::constant(facts.materialized[v.0]),
            Choices::Open { whole, .. } => whole[v.0],
        }
    }

    pub(super) fn tag(&self, atoms: &mut Atoms, m: &mut Manager, c: Choice) -> Bdd {
        if matches!(self.choices, Choices::Fixed { .. }) && atoms.marks(c) {
            atoms.atom(m, Atom::Marker(c))
        } else {
            Bdd::TRUE
        }
    }

    pub(super) fn carried(&self, param: ValueId) -> bool {
        match &self.choices {
            Choices::Fixed { .. } => true,
            Choices::Open { live, .. } => live[param.0],
        }
    }

    pub(super) fn tracked(&self, v: ValueId) -> bool {
        match &self.choices {
            Choices::Fixed { read, .. } => read.as_ref().map_or(true, |read| read[v.0]),
            Choices::Open { read, .. } => read[v.0],
        }
    }

    pub(super) fn all_local(&self) -> Bdd {
        match self.choices {
            Choices::Fixed { .. } => Bdd::TRUE,
            Choices::Open { all_local, .. } => all_local,
        }
    }

    pub(super) fn choose(&self, atoms: &Atoms, m: &mut Manager, mut safe: Bdd) -> Kept {
        assert_ne!(safe, Bdd::FALSE);
        let mut kept = Kept::default();
        for &c in &self.listed {
            let var = atoms.var(Atom::Marker(c)).unwrap();
            let local = m.cofactor(safe, var, true);
            if local != Bdd::FALSE {
                safe = local;
                continue;
            }
            safe = m.cofactor(safe, var, false);
            kept.insert(c);
        }
        assert_eq!(safe, Bdd::TRUE);
        kept
    }
}

pub(super) fn possible_policies(atoms: &mut Atoms, m: &mut Manager, condition: Bdd) -> Bdd {
    let varying: Vec<u32> = atoms
        .support(m, condition)
        .iter()
        .copied()
        .filter(|&var| !matches!(atoms.of(var), Atom::Marker(_)))
        .collect();
    exists(m, &varying, condition)
}

pub(super) fn settled(atoms: &mut Atoms, m: &mut Manager, f: Bdd, kept: &BTreeSet<Choice>) -> Bdd {
    let markers: HashMap<u32, Bdd> = atoms
        .support(m, f)
        .iter()
        .filter_map(|&var| match atoms.of(var) {
            Atom::Marker(v) => Some((var, Manager::constant(!kept.contains(&v)))),
            _ => None,
        })
        .collect();
    if markers.is_empty() {
        return f;
    }
    m.compose(f, &|var| markers.get(&var).copied())
}

pub fn live_values(f: &Func, facts: &Facts) -> Vec<bool> {
    let mut pending = Vec::new();
    for &id in &facts.order {
        let block = &f.blocks[&id];
        if let Term::CondBr { cond, .. } = block.term {
            pending.push(cond);
        }
        for inst in &block.insts {
            if let Inst::Effect { op, inputs, .. } = inst {
                if !matches!(
                    op,
                    EffectOp::Wave(WaveOp::Any | WaveOp::Ballot { .. } | WaveOp::ReadFirstLane)
                ) {
                    pending.extend(inputs);
                }
            }
        }
    }
    let mut live = vec![false; f.types.len()];
    while let Some(v) = pending.pop() {
        if live[v.0] {
            continue;
        }
        live[v.0] = true;
        match facts.site[v.0] {
            Site::Param { block, index } if block != f.entry => {
                pending.extend(facts.arguments(f, block, index));
            }
            Site::Inst { block, index } => {
                pending.extend(f.blocks[&block].insts[index].operands());
            }
            _ => {}
        }
    }
    live
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Read {
    Never,
    Under(ValueId, bool),
    Always,
}

impl Read {
    fn or(self, other: Read) -> Read {
        match (self, other) {
            (Read::Never, read) | (read, Read::Never) => read,
            (a, b) if a == b => a,
            _ => Read::Always,
        }
    }

    fn under(self, c: ValueId, taken: bool) -> Read {
        match self {
            Read::Never => Read::Never,
            Read::Under(d, t) if d == c && t != taken => Read::Never,
            _ => Read::Under(c, taken),
        }
    }
}

/// The values whose differences some use can read. These are the live values, except a value that
/// reaches its readers only through an arm of a select whose result is read only where the select's
/// condition takes the other arm. A select reads each arm only where its condition takes that arm,
/// and a condition on a block parameter passes to the argument of each edge into the block. A branch
/// and every effect but a wave query read their inputs in every lane. A wave query, a target op and a
/// test of a lane word against zero read their inputs in every lane wherever their results are read.
pub fn read_values(f: &Func, facts: &Facts) -> Vec<bool> {
    let n = f.types.len();
    let mut reads = vec![Read::Never; n];
    let mut pending: Vec<ValueId> = (0..n).map(ValueId).collect();
    while let Some(v) = pending.pop() {
        let read = facts.uses[v.0].iter().fold(reads[v.0], |read, &u| read.or(read_by(f, facts, &reads, v, u)));
        if read == reads[v.0] {
            continue;
        }
        reads[v.0] = read;
        match facts.site[v.0] {
            Site::Param { block, index } if block != f.entry => pending.extend(facts.arguments(f, block, index)),
            Site::Inst { block, index } => pending.extend(f.blocks[&block].insts[index].operands()),
            _ => {}
        }
    }
    reads.into_iter().map(|read| read != Read::Never).collect()
}

fn read_by(f: &Func, facts: &Facts, reads: &[Read], v: ValueId, u: Use) -> Read {
    match u {
        Use::Arg { block, edge, index } => {
            let e = f.blocks[&block].term.edges().nth(edge).unwrap();
            match reads[f.blocks[&e.dst].params[index].0 .0] {
                Read::Under(c, taken) => match facts.site[c.0] {
                    Site::Param { block, index } if block == e.dst => Read::Under(e.args[index], taken),
                    _ => Read::Always,
                },
                read => read,
            }
        }
        Use::Inst { block, index } => {
            let inst = &f.blocks[&block].insts[index];
            let read = inst.outputs().iter().fold(Read::Never, |read, o| read.or(reads[o.0]));
            match inst {
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any | WaveOp::Ballot { .. } | WaveOp::ReadFirstLane),
                    ..
                }
                | Inst::Target { .. } => {
                    if read == Read::Never {
                        Read::Never
                    } else {
                        Read::Always
                    }
                }
                Inst::Effect { .. } => Read::Always,
                Inst::Core { op: Op::Select(c, a, b), .. } => {
                    let condition = if *c == v { read } else { Read::Never };
                    let taken = if *a == v { read.under(*c, true) } else { Read::Never };
                    let other = if *b == v { read.under(*c, false) } else { Read::Never };
                    condition.or(taken).or(other)
                }
                Inst::Core { op: Op::Cmp(IntPred::Eq | IntPred::Ne, a, b), .. }
                    if lane_test(f, facts, *a, *b) == Some(v) && read != Read::Never =>
                {
                    Read::Always
                }
                _ => read,
            }
        }
        Use::Cond(_) => Read::Always,
        Use::Ret(_) => Read::Never,
    }
}

pub fn choices(f: &Func, facts: &Facts) -> Vec<Choice> {
    let live = live_values(f, facts);
    let mut out = Vec::new();
    let mut words = BTreeSet::new();
    for &id in &facts.order {
        for inst in &f.blocks[&id].insts {
            match inst {
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any),
                    outputs,
                    ..
                } if live[outputs[0].0 .0] => out.push(Choice::Query(outputs[0].0)),
                Inst::Core {
                    value,
                    op: Op::Cmp(IntPred::Eq | IntPred::Ne, a, b),
                    ..
                } if live[value.0] => {
                    if let Some(w) = lane_test(f, facts, *a, *b) {
                        if !facts.materialized[w.0] && words.insert(w) {
                            out.push(Choice::Word(w));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}
