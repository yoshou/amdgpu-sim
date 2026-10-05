use super::super::hazard::Kind;
use super::super::logic::{Choice, Logic};
use super::program::Program;
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Default, PartialEq)]
pub struct Orderings {
    loaded: BTreeMap<(BlockId, usize), Bdd>,
    reordered: BTreeMap<(BlockId, usize), Bdd>,
}

impl Orderings {
    pub(super) fn new(program: &Program, logic: &mut Logic) -> Self {
        let mut this = Self::default();
        let hazards = program.hazards;
        let conflicts = hazards.conflicts();
        if conflicts.is_empty() {
            return this;
        }
        let (f, facts) = (program.f, program.facts);
        let mut partners: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
        for &(p, q) in &conflicts {
            partners.entry(p).or_default().insert(q);
            partners.entry(q).or_default().insert(p);
        }
        let pair = |p: usize, q: usize| (p.min(q), p.max(q));
        let at: BTreeMap<(BlockId, usize), usize> = hazards
            .accesses
            .iter()
            .enumerate()
            .map(|(i, a)| ((a.block, a.index), i))
            .collect();
        let n = facts.order.len();
        for (&s, targets) in &partners {
            let source = &hazards.accesses[s];
            let source_rank = program.rank[&source.block];
            let mut entry = vec![[Bdd::FALSE; 2]; n];
            let mut reaches: BTreeMap<usize, [Bdd; 2]> = BTreeMap::new();
            let mut work: BTreeSet<usize> = BTreeSet::from([source_rank]);
            while let Some(r) = work.pop_first() {
                let b = facts.order[r];
                let [mut within, mut around] = entry[r];
                let mut fresh = Bdd::FALSE;
                for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
                    if let Some(&m) = program.meetings.get(&(b, index)) {
                        let local = logic.local(Choice::Meet(m));
                        let tag = logic.tag(Choice::Meet(m));
                        let left = logic.m.and(local, tag);
                        within = logic.m.and(within, left);
                        around = logic.m.and(around, left);
                        fresh = logic.m.and(fresh, left);
                    }
                    if meets_every_lane(inst) || reads_a_varying_first_lane(program, inst) {
                        within = Bdd::FALSE;
                        around = Bdd::FALSE;
                        fresh = Bdd::FALSE;
                    }
                    if let Some(kept) = collective(program, logic, inst) {
                        let converted = logic.m.not(kept);
                        within = logic.m.and(within, converted);
                        around = logic.m.and(around, converted);
                        fresh = logic.m.and(fresh, converted);
                    }
                    if let Some(&t) = at.get(&(b, index)) {
                        if targets.contains(&t) {
                            let key = pair(s, t);
                            let again = hazards.accesses[t].instruction == source.instruction;
                            let mut pending = [Bdd::FALSE; 2];
                            if hazards.together.contains(&key) && !again {
                                pending[0] = logic.m.or(within, fresh);
                            }
                            if hazards.apart.contains(&key) {
                                pending[1] = around;
                            }
                            if pending != [Bdd::FALSE; 2] {
                                let old = reaches.get(&t).copied().unwrap_or([Bdd::FALSE; 2]);
                                let joined = [logic.m.or(old[0], pending[0]), logic.m.or(old[1], pending[1])];
                                reaches.insert(t, joined);
                            }
                        }
                    }
                    if (b, index) == (source.block, source.index) {
                        fresh = Bdd::TRUE;
                    }
                }
                let within = logic.m.or(within, fresh);
                if within == Bdd::FALSE && around == Bdd::FALSE {
                    continue;
                }
                for e in f.blocks[&b].term.edges() {
                    let d = program.rank[&e.dst];
                    let back = d <= r
                        && (0..program.loops.count()).any(|l| {
                            program.loops.header(l) == d && program.loops.contains(l, source_rank)
                        });
                    let (w, a) = if back {
                        (Bdd::FALSE, logic.m.or(within, around))
                    } else {
                        (within, around)
                    };
                    let [ow, oa] = entry[d];
                    let (nw, na) = (logic.m.or(ow, w), logic.m.or(oa, a));
                    if (nw, na) != (ow, oa) {
                        entry[d] = [nw, na];
                        work.insert(d);
                    }
                }
            }
            for (t, parts) in reaches {
                let target = &hazards.accesses[t];
                let key = pair(s, t);
                let reads = |logic: &mut Logic, reader: usize| -> Bdd {
                    let side = if reader == key.0 { 0 } else { 1 };
                    let mut joined = Bdd::FALSE;
                    for (k, apart) in [(0, false), (1, true)] {
                        let mut part = parts[k];
                        if part == Bdd::FALSE {
                            continue;
                        }
                        if hazards.idle.contains(&(key.0, key.1, apart, side)) {
                            if let Some(e) = hazards.accesses[reader].exec {
                                let active = logic.bit(f, facts, e);
                                let idle = logic.m.not(active);
                                part = logic.m.and(part, idle);
                            }
                        }
                        joined = logic.m.or(joined, part);
                    }
                    joined
                };
                let add = |logic: &mut Logic, map: &mut BTreeMap<(BlockId, usize), Bdd>, at: (BlockId, usize), extra: Bdd| {
                    let old = map.get(&at).copied().unwrap_or(Bdd::FALSE);
                    let joined = logic.m.or(old, extra);
                    map.insert(at, joined);
                };
                if source.kind.writes() && target.kind.reads() && !reads_back_its_own_constant(program, s, t) {
                    let extra = reads(logic, t);
                    add(logic, &mut this.loaded, (target.block, target.index), extra);
                }
                if source.kind.reads() && target.kind.writes() {
                    let extra = reads(logic, s);
                    add(logic, &mut this.loaded, (source.block, source.index), extra);
                }
                if source.kind.writes() && target.kind.writes() {
                    let extra = logic.m.or(parts[0], parts[1]);
                    add(logic, &mut this.reordered, (target.block, target.index), extra);
                }
            }
        }
        this
    }

    #[inline]
    pub(super) fn loaded(&self, at: (BlockId, usize)) -> Option<Bdd> {
        self.loaded.get(&at).copied()
    }

    #[inline]
    pub(super) fn reordered(&self, at: (BlockId, usize)) -> Option<Bdd> {
        self.reordered.get(&at).copied()
    }
}

fn collective(program: &Program, logic: &mut Logic, inst: &Inst) -> Option<Bdd> {
    let Inst::Effect {
        op: EffectOp::Wave(op),
        outputs,
        ..
    } = inst
    else {
        return None;
    };
    let out = outputs[0].0;
    match op {
        WaveOp::Any => {
            let local = logic.local(Choice::Query(out));
            Some(logic.m.not(local))
        }
        WaveOp::Ballot => Some(logic.materialized(program.facts, out)),
        _ => None,
    }
}

fn reads_a_varying_first_lane(program: &Program, inst: &Inst) -> bool {
    match inst {
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::ReadFirstLane),
            inputs,
            ..
        } => !program.facts.uniform[inputs[0].0],
        _ => false,
    }
}

fn reads_back_its_own_constant(program: &Program, s: usize, t: usize) -> bool {
    let accesses = &program.hazards.accesses;
    let (w, r) = (&accesses[s], &accesses[t]);
    let same = w.block == r.block
        && w.index < r.index
        && w.kind == Kind::Write
        && r.kind == Kind::Read
        && w.space == r.space
        && w.address.is_some()
        && w.address == r.address
        && w.predicate == r.predicate
        && w.bytes >= r.bytes;
    if !same || !w.address.is_some_and(|a| program.facts.uniform[a.0]) {
        return false;
    }
    let Inst::Effect {
        op: EffectOp::Memory {
            op: MemoryOp::Store(_),
            ..
        },
        inputs,
        ..
    } = &program.f.blocks[&w.block].insts[w.index]
    else {
        return false;
    };
    program.facts.constant(program.f, inputs[1]).is_some()
        && !accesses
            .iter()
            .any(|a| a.block == w.block && a.index > w.index && a.index < r.index && a.kind != Kind::Read)
}

fn meets_every_lane(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Effect {
            op: EffectOp::BarrierSignal { .. }
                | EffectOp::BarrierWait
                | EffectOp::Wave(
                    WaveOp::ReadLane
                        | WaveOp::WriteLane
                        | WaveOp::Bpermute
                        | WaveOp::BpermuteFi
                        | WaveOp::Wmma
                ),
            ..
        }
    )
}
