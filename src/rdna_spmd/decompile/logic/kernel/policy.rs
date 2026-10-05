use super::atoms::{exists, Atom, Atoms, Choice};
use super::patterns::lane_test;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

#[derive(Default, Clone)]
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
    },

    Open {
        whole: Vec<Bdd>,
        live: Vec<bool>,
        all_local: Bdd,
    },
}

pub(in super::super) struct Policy {
    choices: Choices,
    listed: Vec<Choice>,
}

impl Policy {
    pub(in super::super) fn fixed(kept: &BTreeSet<Choice>, listed: &[Choice]) -> Self {
        Self {
            choices: Choices::Fixed { kept: kept.clone() },
            listed: listed.to_vec(),
        }
    }

    pub(in super::super) fn open(f: &Func, facts: &Facts, listed: &[Choice], atoms: &mut Atoms, m: &mut Manager) -> Self {
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
            for a in sources(f, facts, v) {
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
        Self {
            choices: Choices::Open {
                whole,
                live: live_values(f, facts),
                all_local,
            },
            listed: listed.to_vec(),
        }
    }

    pub(in super::super) fn keep(&mut self, kept: &BTreeSet<Choice>) {
        if let Choices::Fixed { kept: fixed } = &mut self.choices {
            *fixed = kept.clone();
        }
    }

    pub(in super::super) fn is_open(&self) -> bool {
        matches!(self.choices, Choices::Open { .. })
    }

    pub(in super::super) fn local(&self, atoms: &mut Atoms, m: &mut Manager, c: Choice) -> Bdd {
        if let Choices::Fixed { kept } = &self.choices {
            return Manager::constant(!kept.contains(&c));
        }
        if atoms.marks(c) {
            atoms.atom(m, Atom::Marker(c))
        } else {
            Bdd::TRUE
        }
    }

    pub(in super::super) fn materialized(&self, facts: &Facts, v: ValueId) -> Bdd {
        match &self.choices {
            Choices::Fixed { .. } => Manager::constant(facts.materialized[v.0]),
            Choices::Open { whole, .. } => whole[v.0],
        }
    }

    pub(in super::super) fn tag(&self, atoms: &mut Atoms, m: &mut Manager, c: Choice) -> Bdd {
        if matches!(self.choices, Choices::Fixed { .. }) && atoms.marks(c) {
            atoms.atom(m, Atom::Marker(c))
        } else {
            Bdd::TRUE
        }
    }

    pub(in super::super) fn carried(&self, param: ValueId) -> bool {
        match &self.choices {
            Choices::Fixed { .. } => true,
            Choices::Open { live, .. } => live[param.0],
        }
    }

    pub(in super::super) fn all_local(&self) -> Bdd {
        match self.choices {
            Choices::Fixed { .. } => Bdd::TRUE,
            Choices::Open { all_local, .. } => all_local,
        }
    }

    pub(in super::super) fn choose(&self, atoms: &Atoms, m: &mut Manager, mut safe: Bdd) -> Kept {
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

pub(in super::super) fn possible_policies(atoms: &mut Atoms, m: &mut Manager, condition: Bdd) -> Bdd {
    let varying: Vec<u32> = atoms
        .support(m, condition)
        .iter()
        .copied()
        .filter(|&var| !matches!(atoms.of(var), Atom::Marker(_)))
        .collect();
    exists(m, &varying, condition)
}

pub(in super::super) fn settled(atoms: &mut Atoms, m: &mut Manager, f: Bdd, kept: &BTreeSet<Choice>) -> Bdd {
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

fn sources(f: &Func, facts: &Facts, v: ValueId) -> Vec<ValueId> {
    match facts.site[v.0] {
        Site::Param { block, index } if block != f.entry => {
            facts.arguments(f, block, index).collect()
        }
        Site::Inst { .. } => match facts.op(f, v) {
            Some(Op::Int(_, a, b)) | Some(Op::Select(_, a, b)) => vec![a, b],
            Some(Op::Convert(_, _, a)) => vec![a],
            _ => vec![],
        },
        _ => vec![],
    }
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
                    EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane)
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
