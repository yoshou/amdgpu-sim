use super::super::kernel::{Atom, Queries};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

#[derive(Default)]
struct BlockAnswers {
    residuals: Vec<(Bdd, ValueId)>,
    relation: Option<(usize, Bdd)>,
}

const ANSWERS: usize = 1 << 10;

#[derive(Default)]
pub(super) struct Answers(HashMap<BlockId, BlockAnswers>);

impl Answers {
    pub(super) fn sources(&self, block: BlockId) -> Vec<ValueId> {
        self.0.get(&block).map_or_else(Vec::new, |a| a.residuals.iter().map(|&(_, v)| v).collect())
    }
}

pub(super) trait HasAnswers {
    fn answers(&self) -> &Answers;
    fn answers_mut(&mut self) -> &mut Answers;
}

fn mentions_answers<Q: Queries>(q: &mut Q, f: Bdd) -> bool {
    let support = q.support(f);
    support.iter().any(|&v| matches!(q.atoms().get(v), Some(Atom::Some(..))))
}

pub(super) fn consistent<Q: Queries>(q: &mut Q, x: Bdd) -> Bdd
where
    Q::State: HasAnswers,
{
    if !mentions_answers(q, x) {
        return x;
    }
    let blocks: BTreeSet<BlockId> = q
        .support(x)
        .iter()
        .filter_map(|&v| match q.atoms().of(v) {
            Atom::Some(block, _) => Some(block),
            _ => None,
        })
        .collect();
    let mut result = x;
    for block in blocks {
        let relation = answer_relation(q, block);
        result = q.m().and(result, relation);
    }
    result
}

fn answer_relation<Q: Queries>(q: &mut Q, block: BlockId) -> Bdd
where
    Q::State: HasAnswers,
{
    let residuals: Vec<Bdd> = q.state().answers().0[&block].residuals.iter().map(|&(g, _)| g).collect();
    if let Some((count, relation)) = q.state().answers().0[&block].relation {
        if count == residuals.len() {
            return relation;
        }
    }
    let mut relation = Bdd::TRUE;
    if let Some(worlds) = answer_worlds(q, &residuals) {
        let mut any = Bdd::FALSE;
        for world in &worlds {
            let mut minterm = Bdd::TRUE;
            for (k, &held) in world.iter().enumerate() {
                let var = q.atom(Atom::Some(block, k as u16));
                let literal = if held { var } else { q.m().not(var) };
                minterm = q.m().and(minterm, literal);
            }
            any = q.m().or(any, minterm);
        }
        relation = any;
    }
    for (k, &g) in residuals.iter().enumerate() {
        let some = q.atom(Atom::Some(block, k as u16));
        let not_g = q.m().not(g);
        let implied = q.m().or(not_g, some);
        relation = q.m().and(relation, implied);
        if let Some(lane) = only_lane(q, g) {
            let here = q.lanes(|l| l == lane);
            let not_some = q.m().not(some);
            let back = q.m().or(not_some, g);
            let tie = q.m().ite(here, back, Bdd::TRUE);
            relation = q.m().and(relation, tie);
        }
    }
    q.state_mut().answers_mut().0.get_mut(&block).unwrap().relation = Some((residuals.len(), relation));
    relation
}

fn only_lane<Q: Queries>(q: &mut Q, g: Bdd) -> Option<u32> {
    if !q.lane_dependent(g) {
        return None;
    }
    let mut found = None;
    for lane in 0..q.atoms().lanes() {
        if q.at_lane(g, lane) != Bdd::FALSE {
            if found.is_some() {
                return None;
            }
            found = Some(lane);
        }
    }
    found
}

pub(super) fn wave_answer<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, out: ValueId) -> Bdd
where
    Q::State: HasAnswers,
{
    let Some(Inst::Effect { inputs, .. }) = facts.inst(f, out) else {
        return q.atom(Atom::Bit(out));
    };
    let bits = q.bit(f, facts, inputs[0]);
    answer(q, f, facts, out, bits)
}

pub(super) fn answer<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, out: ValueId, g: Bdd) -> Bdd
where
    Q::State: HasAnswers,
{
    let (Site::Inst { block, .. }, Some(Inst::Effect { inputs, .. })) = (facts.site[out.0], facts.inst(f, out)) else {
        return q.atom(Atom::Bit(out));
    };
    let some = some_lane_holds(q, facts, block, g, inputs[0], &mut HashMap::default());
    q.m().or(g, some)
}

fn answer_worlds<Q: Queries>(q: &mut Q, residuals: &[Bdd]) -> Option<Vec<Vec<bool>>> {
    let mut out = Vec::new();
    let mut chosen = Vec::with_capacity(residuals.len());
    enumerate_answers(q, residuals, &mut chosen, Bdd::FALSE, &mut out)?;
    Some(out)
}

fn enumerate_answers<Q: Queries>(q: &mut Q, residuals: &[Bdd], chosen: &mut Vec<bool>, denied: Bdd, out: &mut Vec<Vec<bool>>) -> Option<()> {
    let k = chosen.len();
    if k == residuals.len() {
        if out.len() >= ANSWERS {
            return None;
        }
        out.push(chosen.clone());
        return Some(());
    }
    for value in [true, false] {
        let denied = if value { denied } else { q.m().or(denied, residuals[k]) };
        chosen.push(value);
        if answers_possible(q, residuals, chosen, denied) {
            let r = enumerate_answers(q, residuals, chosen, denied, out);
            if r.is_none() {
                chosen.pop();
                return None;
            }
        }
        chosen.pop();
    }
    Some(())
}

fn answers_possible<Q: Queries>(q: &mut Q, residuals: &[Bdd], chosen: &[bool], denied: Bdd) -> bool {
    let open = q.m().not(denied);
    let mut at_lane: HashMap<u32, Bdd> = HashMap::default();
    for (i, &held) in chosen.iter().enumerate() {
        if !held {
            continue;
        }
        let witness = q.m().and(residuals[i], open);
        if witness == Bdd::FALSE {
            return false;
        }
        if let Some(lane) = only_lane(q, residuals[i]) {
            let joint = at_lane.get(&lane).copied().unwrap_or(open);
            let joint = q.m().and(joint, residuals[i]);
            let joint = q.at_lane(joint, lane);
            if joint == Bdd::FALSE {
                return false;
            }
            at_lane.insert(lane, joint);
        }
    }
    true
}

fn some_lane_holds<Q: Queries>(q: &mut Q, facts: &Facts, block: BlockId, g: Bdd, source: ValueId, done: &mut HashMap<Bdd, Bdd>) -> Bdd
where
    Q::State: HasAnswers,
{
    if g == Bdd::FALSE || g == Bdd::TRUE {
        return g;
    }
    if let Some(&b) = done.get(&g) {
        return b;
    }
    let support = q.support(g);
    let result = match support.iter().copied().find(|&v| q.uniform_atom(facts, v)) {
        Some(var) => {
            let (low, high) = (q.m().cofactor(g, var, false), q.m().cofactor(g, var, true));
            let no = some_lane_holds(q, facts, block, low, source, done);
            let yes = some_lane_holds(q, facts, block, high, source, done);
            let x = q.m().var(var);
            q.m().ite(x, yes, no)
        }
        None => {
            let answers = q.state_mut().answers_mut().0.entry(block).or_default();
            let k = match answers.residuals.iter().position(|&(x, _)| x == g) {
                Some(k) => k,
                None => {
                    answers.residuals.push((g, source));
                    answers.residuals.len() - 1
                }
            };
            q.atom(Atom::Some(block, k as u16))
        }
    };
    done.insert(g, result);
    result
}
