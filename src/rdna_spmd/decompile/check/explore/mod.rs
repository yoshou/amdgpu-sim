mod decide;
mod eval;
mod forms;
mod joint;

use super::super::logic::Atom;
use super::program::Program;
use super::queries::Queries;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use eval::{Eval, Evaluation, Write};
use joint::*;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

#[derive(Default)]
pub(super) struct Memo {
    positions: BTreeMap<BlockId, Rc<HashMap<ValueId, usize>>>,
    fresh: Fresh,
}

struct Pair {
    cond: Bdd,
    sides: Sides,
    writes: [Vec<Write>; 2],
    entries: BTreeMap<(Key, usize), (Bdd, Sides)>,
}

pub(super) struct Explore<'c, 'a, Q: Queries<'a>> {
    eval: Eval<'c, 'a, Q>,
    memo: &'c mut Memo,
    branch: BlockId,
    visited: BTreeSet<BlockId>,
    headers: BTreeSet<BlockId>,
    paths: u32,
}

impl<'c, 'a, Q: Queries<'a>> Explore<'c, 'a, Q> {
    pub(super) fn new(q: &'c mut Q, memo: &'c mut Memo, branch: BlockId, assume: Bdd) -> Self {
        let program = q.program();
        let headers = (0..program.loops.count()).map(|l| program.facts.order[program.loops.header(l)]).collect();
        Self {
            eval: Eval::new(q, assume),
            memo,
            branch,
            visited: BTreeSet::new(),
            headers,
            paths: PATHS,
        }
    }

    fn merge(&mut self, side: usize, param: ValueId, cond: Bdd, old: Desc, new: Desc) -> Desc {
        let same = if old.same == new.same { old.same } else { None };
        let bits = match (old.bits, new.bits) {
            (Some(a), Some(b)) if a == b => Some(a),
            (Some(a), Some(b)) => match (self.eval.decide(a, cond, side), self.eval.decide(b, cond, side)) {
                (Some(x), Some(y)) if x == y => Some(Manager::constant(x)),
                _ => Some(between(self.eval.logic(), &mut self.memo.fresh, side, param, a, b)),
            },
            _ => None,
        };
        Desc { same, bits }
    }

    fn positions(&mut self, x: BlockId) -> Rc<HashMap<ValueId, usize>> {
        if let Some(p) = self.memo.positions.get(&x) {
            return p.clone();
        }
        let p: Rc<HashMap<ValueId, usize>> = Rc::new(
            self.eval.q.program().f.blocks[&x]
                .params
                .iter()
                .enumerate()
                .map(|(j, &(v, _))| (v, j))
                .collect(),
        );
        self.memo.positions.insert(x, p.clone());
        p
    }

    pub(super) fn run(&mut self, wave: usize, lane: usize, arrivals: &mut BTreeMap<BlockId, Vec<Bdd>>) {
        let f = self.eval.q.program().f;
        let assume = self.eval.assume();
        let edges: Vec<&Edge> = f.blocks[&self.branch].term.edges().collect();
        let (e0, e1) = (edges[wave], edges[lane]);
        let sides = [
            e0.args.iter().map(|&a| self.eval.start(a)).collect(),
            e1.args.iter().map(|&a| self.eval.start(a)).collect(),
        ];
        let first: Key = [Some(e0.dst), Some(e1.dst)];
        let exit = f.blocks.len();
        let priority = |program: &Program, key: &Key| -> usize {
            key.iter().map(|b| b.map_or(exit, |b| program.rank[&b])).sum()
        };
        let mut pairs: BTreeMap<Key, Pair> = BTreeMap::from([(
            first,
            Pair {
                cond: assume,
                sides: sides.clone(),
                writes: [Vec::new(), Vec::new()],
                entries: BTreeMap::from([(([None, None], 0), (assume, sides))]),
            },
        )]);
        let mut worklist = BTreeSet::from([(priority(self.eval.q.program(), &first), first)]);
        while let Some((_, key)) = worklist.pop_first() {
            let pair = &pairs[&key];
            let (cond, sides, writes) = (pair.cond, pair.sides.clone(), pair.writes.clone());
            let cond = self.eval.q.and(cond, self.eval.q.safe());
            if cond == Bdd::FALSE {
                continue;
            }
            let side = match key {
                [None, None] => {
                    self.match_writes(cond, &writes);
                    if self.eval.q.stopped() {
                        return;
                    }
                    continue;
                }
                [Some(a), Some(b)] if a == b => {
                    self.match_writes(cond, &writes);
                    if self.eval.q.stopped() {
                        return;
                    }
                    self.meet(a, cond, &sides, arrivals);
                    continue;
                }
                [Some(a), Some(b)] => {
                    let program = self.eval.q.program();
                    if program.loops.before(program.rank[&a], program.rank[&b]) {
                        WAVE
                    } else {
                        LANE
                    }
                }
                [Some(_), None] => WAVE,
                [None, Some(_)] => LANE,
            };
            let block = key[side].unwrap();
            let steps = self.step(side, block, &sides[side], cond, !writes[side].is_empty());
            if self.eval.q.stopped() {
                return;
            }
            for (slot, (dst, descs, constraint, done)) in steps.into_iter().enumerate() {
                let c = self.eval.q.and(cond, constraint);
                let c = self.eval.q.and(c, self.eval.q.safe());
                if c == Bdd::FALSE {
                    continue;
                }
                let mut next = key;
                next[side] = dst;
                let incoming = if side == WAVE {
                    [descs, sides[LANE].clone()]
                } else {
                    [sides[WAVE].clone(), descs]
                };
                let looped = next.iter().flatten().any(|b| self.headers.contains(b));
                let (c, incoming) = if looped {
                    canonical(self.eval.logic(), &mut self.memo.fresh, c, incoming)
                } else {
                    (c, incoming)
                };
                let mut traces = writes.clone();
                traces[side].extend(done);
                let merged = match pairs.get(&next) {
                    None => Pair {
                        cond: c,
                        sides: incoming.clone(),
                        writes: traces,
                        entries: BTreeMap::from([((key, slot), (c, incoming))]),
                    },
                    Some(old) => {
                        let (ocond, osides) = (old.cond, old.sides.clone());
                        if old.writes != traces {
                            for s in [WAVE, LANE] {
                                self.unmatched(c, &traces[s][..], s);
                                self.unmatched(ocond, &old.writes[s][..], s);
                            }
                            if self.eval.q.stopped() {
                                return;
                            }
                        }
                        let kept = old.writes.clone();
                        if !looped {
                            let mut entries = old.entries.clone();
                            let arrival = (c, incoming);
                            if entries.get(&(key, slot)) == Some(&arrival) {
                                continue;
                            }
                            let (mcond, msides) = match entries.insert((key, slot), arrival.clone()) {
                                None => join(self.eval.logic(), &mut self.paths, (ocond, &osides), (arrival.0, &arrival.1)),
                                Some(_) => joined(self.eval.logic(), &mut self.paths, &entries),
                            };
                            let changed = mcond != ocond || msides != osides;
                            pairs.insert(
                                next,
                                Pair {
                                    cond: mcond,
                                    sides: msides,
                                    writes: kept,
                                    entries,
                                },
                            );
                            if changed {
                                worklist.insert((priority(self.eval.q.program(), &next), next));
                            }
                            continue;
                        }
                        let mcond = self.eval.q.or(ocond, c);
                        let mut grew = mcond != ocond;
                        let mut merged = osides.clone();
                        for s in [WAVE, LANE] {
                            let Some(blk) = next[s] else { continue };
                            if osides[s] == incoming[s] {
                                continue;
                            }
                            let mut classes: Vec<((Bdd, Bdd), Bdd)> = Vec::new();
                            for (k, &(param, _)) in f.blocks[&blk].params.iter().enumerate() {
                                let (o, n) = (osides[s][k], incoming[s][k]);
                                if o == n {
                                    continue;
                                }
                                let shared = match (o.bits, n.bits) {
                                    (Some(ob), Some(nb)) => {
                                        let m = &mut self.eval.logic().m;
                                        let (no, nn) = (m.not(ob), m.not(nb));
                                        classes.iter().find_map(|&(pair, bits)| {
                                            if pair == (ob, nb) {
                                                Some(bits)
                                            } else if pair == (no, nn) {
                                                Some(m.not(bits))
                                            } else {
                                                None
                                            }
                                        })
                                    }
                                    _ => None,
                                };
                                let m = match shared {
                                    Some(bits) => Desc {
                                        same: if o.same == n.same { o.same } else { None },
                                        bits: Some(bits),
                                    },
                                    None => {
                                        let m = self.merge(s, param, mcond, o, n);
                                        if let (Some(ob), Some(nb), Some(mb)) = (o.bits, n.bits, m.bits) {
                                            classes.push(((ob, nb), mb));
                                        }
                                        m
                                    }
                                };
                                if m != o {
                                    merged[s][k] = m;
                                    grew = true;
                                }
                            }
                        }
                        if !grew {
                            continue;
                        }
                        let (mcond, merged) = canonical(self.eval.logic(), &mut self.memo.fresh, mcond, merged);
                        if merged == osides && mcond == ocond {
                            continue;
                        }
                        Pair {
                            cond: mcond,
                            sides: merged,
                            writes: kept,
                            entries: BTreeMap::new(),
                        }
                    }
                };
                pairs.insert(next, merged);
                worklist.insert((priority(self.eval.q.program(), &next), next));
            }
        }
    }

    fn step(
        &mut self,
        side: usize,
        x: BlockId,
        params: &[Desc],
        cond: Bdd,
        stored: bool,
    ) -> Vec<(Option<BlockId>, Vec<Desc>, Bdd, Vec<Write>)> {
        let f = self.eval.q.program().f;
        let block = &f.blocks[&x];
        let mut ev = Evaluation {
            side,
            block: x,
            cond,
            descs: block
                .params
                .iter()
                .map(|&(v, _)| v)
                .zip(params.iter().copied())
                .collect(),
            writes: Vec::new(),
            stored,
        };
        self.visited.insert(ev.block);
        self.eval.check_effects(&mut ev);
        if self.eval.q.stopped() {
            return Vec::new();
        }
        let followed: Vec<(usize, Bdd)> = match &block.term {
            Term::Ret(_) => return vec![(None, Vec::new(), Bdd::TRUE, ev.writes)],
            Term::Br(_) => vec![(0, Bdd::TRUE)],
            Term::CondBr { cond: c, .. } => {
                let g = match self.eval.get(&mut ev, *c).bits {
                    Some(g) => g,
                    None => self.eval.fresh(side, *c, 0),
                };
                match self.eval.decide(g, cond, side) {
                    Some(true) => vec![(0, Bdd::TRUE)],
                    Some(false) => vec![(1, Bdd::TRUE)],
                    None => {
                        let ng = self.eval.logic().m.not(g);
                        vec![(0, self.eval.weaken(g, side)), (1, self.eval.weaken(ng, side))]
                    }
                }
            }
        };
        let position = self.positions(x);
        let mut out = Vec::new();
        for (slot, constraint) in followed {
            let edge = block.term.edges().nth(slot).unwrap();
            let mut args = Vec::with_capacity(edge.args.len());
            for &arg in &edge.args {
                let d = match position.get(&arg) {
                    Some(&j) => params[j],
                    None => self.eval.get(&mut ev, arg),
                };
                args.push(d);
            }
            out.push((Some(edge.dst), args, constraint, ev.writes.clone()));
        }
        out
    }

    fn unmatched(&mut self, cond: Bdd, writes: &[Write], side: usize) {
        let reason = if side == WAVE {
            "the wave program may store while the programs are apart"
        } else {
            "the lane program may store while the programs are apart"
        };
        for w in writes {
            self.eval.q.require(w.at.0, w.at.1, reason, cond);
            if self.eval.q.stopped() {
                return;
            }
        }
    }

    fn match_writes(&mut self, cond: Bdd, writes: &[Vec<Write>; 2]) {
        let (wave, lane) = (&writes[WAVE], &writes[LANE]);
        let same = |w: &Write, l: &Write| {
            w.space == l.space
                && w.op == l.op
                && w.address.is_some()
                && w.address == l.address
                && w.data.is_some()
                && w.data == l.data
                && w.mask == l.mask
        };
        let prefix = wave.iter().zip(lane).take_while(|(w, l)| same(w, l)).count();
        let mut pairs: Vec<(usize, usize)> = (0..prefix).map(|i| (i, i)).collect();
        let (rest_wave, rest_lane) = (&wave[prefix..], &lane[prefix..]);
        if rest_wave.len() == rest_lane.len() && self.pairwise_apart(rest_wave) && self.pairwise_apart(rest_lane) {
            let mut taken = vec![false; rest_lane.len()];
            for (i, w) in rest_wave.iter().enumerate() {
                if let Some(j) = (0..rest_lane.len()).find(|&j| !taken[j] && same(w, &rest_lane[j])) {
                    taken[j] = true;
                    pairs.push((prefix + i, prefix + j));
                }
            }
        }
        pairs.retain(|&(i, j)| {
            let (w, l) = (&wave[i], &lane[j]);
            (!w.partnered || self.partners_outside(w.at)) && (!l.partnered || self.partners_outside(l.at))
        });
        for &(i, _) in &pairs {
            let w = wave[i];
            let (address, data) = (self.eval.leaves(w.address.unwrap()), self.eval.leaves(w.data.unwrap()));
            let q = &mut *self.eval.q;
            let mut differs = q.or(address, data);
            let atoms: Vec<u32> = q.logic().support(w.mask).iter().copied().collect();
            for var in atoms {
                if let Atom::Bit(v) | Atom::View(v) | Atom::WordBit(v, _) = q.logic().atom_of(var) {
                    differs = q.or(differs, q.h(v));
                } else if !matches!(q.logic().atom_of(var), Atom::Lane(_) | Atom::Marker(_)) {
                    differs = Bdd::TRUE;
                }
            }
            let differs = q.and(cond, differs);
            q.require(w.at.0, w.at.1, "the programs may store different words while apart", differs);
            if q.stopped() {
                return;
            }
        }
        let left_wave: Vec<Write> = (0..wave.len()).filter(|&i| !pairs.iter().any(|&(x, _)| x == i)).map(|i| wave[i]).collect();
        let left_lane: Vec<Write> = (0..lane.len()).filter(|&j| !pairs.iter().any(|&(_, y)| y == j)).map(|j| lane[j]).collect();
        self.unmatched(cond, &left_wave, WAVE);
        if self.eval.q.stopped() {
            return;
        }
        self.unmatched(cond, &left_lane, LANE);
    }

    fn pairwise_apart(&self, writes: &[Write]) -> bool {
        writes.iter().enumerate().all(|(i, a)| writes[i + 1..].iter().all(|b| self.apart(a, b) || self.alike(a, b)))
    }

    fn alike(&self, a: &Write, b: &Write) -> bool {
        let MemoryOp::Store(size) = a.op else {
            return false;
        };
        if a.op != b.op || a.space != b.space || a.data.is_none() || a.data != b.data {
            return false;
        }
        let (Some(x), Some(y)) = (a.address, b.address) else {
            return false;
        };
        let bytes = size.bytes() as u64;
        let (terms, constant) = self.eval.forms.difference(x, y);
        let modulus: u128 = 1 << self.eval.forms.ty(x).bits();
        let aligned = |c: u64| (c as u128 % modulus) % bytes as u128 == 0;
        modulus % bytes as u128 == 0 && aligned(constant) && terms.values().all(|&c| aligned(c))
    }

    fn apart(&self, a: &Write, b: &Write) -> bool {
        if a.space != b.space {
            return true;
        }
        let (Some(x), Some(y)) = (a.address, b.address) else {
            return false;
        };
        let (terms, constant) = self.eval.forms.difference(x, y);
        if !terms.is_empty() {
            return false;
        }
        let bits = self.eval.forms.ty(x).bits();
        let modulus: u128 = 1 << bits;
        let d = constant as u128 % modulus;
        let bytes = |w: &Write| match w.op {
            MemoryOp::Store(size) => size.bytes() as u128,
            _ => 4,
        };
        d >= bytes(b) && modulus - d >= bytes(a)
    }

    fn partners_outside(&self, at: (BlockId, usize)) -> bool {
        let hazards = self.eval.q.program().hazards;
        let Some(i) = hazards.accesses.iter().position(|a| (a.block, a.index) == at) else {
            return true;
        };
        hazards
            .together
            .iter()
            .chain(&hazards.apart)
            .filter_map(|&(p, q)| if p == i { Some(q) } else if q == i { Some(p) } else { None })
            .all(|j| !self.visited.contains(&hazards.accesses[j].block))
    }

    fn meet(
        &mut self,
        block: BlockId,
        cond: Bdd,
        sides: &Sides,
        arrivals: &mut BTreeMap<BlockId, Vec<Bdd>>,
    ) {
        let (f, facts) = (self.eval.q.program().f, self.eval.q.program().facts);
        let dst = &f.blocks[&block];
        let mut relation = Bdd::TRUE;
        for (k, &(param, ty)) in dst.params.iter().enumerate() {
            if !self.eval.logic().carried(param) {
                continue;
            }
            let atom = match ty {
                Ty::I1 => Atom::Bit(param),
                Ty::I32 | Ty::I64 if facts.viewed[param.0] => Atom::View(param),
                _ => continue,
            };
            if let Some(g) = sides[LANE][k].bits {
                let a = self.eval.logic().atom(atom);
                let link = self.eval.logic().m.iff(a, g);
                relation = self.eval.q.and(relation, link);
            }
        }
        let mut restatements: HashMap<Bdd, Bdd> = HashMap::default();
        for (k, &(param, ty)) in dst.params.iter().enumerate() {
            if !self.eval.logic().carried(param) {
                continue;
            }
            let (w, l) = (sides[WAVE][k], sides[LANE][k]);
            let boolean =
                ty == Ty::I1 || (facts.lane_word[param.0] && !facts.materialized[param.0]);
            let same = w.same.is_some() && w.same == l.same;
            let mut differs = if same {
                let leaves = self.eval.leaves(w.same.unwrap());
                self.eval.q.and(cond, leaves)
            } else if let (true, Some(gw), Some(gl)) = (boolean, w.bits, l.bits) {
                let equal = self.eval.logic().m.iff(gw, gl);
                if self.eval.decide(equal, cond, WAVE) == Some(true) {
                    let q = &mut *self.eval.q;
                    let mut atoms: BTreeSet<u32> =
                        q.logic().support(gw).iter().copied().collect();
                    atoms.extend(q.logic().support(gl).iter().copied());
                    let mut hs = Bdd::FALSE;
                    for var in atoms {
                        if let Atom::Bit(v) | Atom::View(v) | Atom::WordBit(v, _) = q.logic().atom_of(var) {
                            hs = q.or(hs, q.h(v));
                        }
                    }
                    q.and(cond, hs)
                } else {
                    cond
                }
            } else {
                cond
            };
            let q = &mut *self.eval.q;
            if matches!(ty, Ty::I32 | Ty::I64) && facts.lane_word[param.0] && !same {
                let mode = q.logic().materialized(facts, param);
                let full = q.and(cond, mode);
                differs = q.or(differs, full);
            }
            if differs == Bdd::FALSE {
                continue;
            }
            let restated = match restatements.get(&differs) {
                Some(&r) => r,
                None => {
                    let logic = q.logic();
                    let mut foreign: HashMap<u32, ()> = HashMap::default();
                    let supports = [logic.support(differs), logic.support(relation)];
                    for &v in supports.iter().flat_map(|s| s.iter()) {
                        if !matches!(logic.atom_of(v), Atom::Marker(_)) && logic.scope(facts, v) != Some(block) {
                            foreign.insert(v, ());
                        }
                    }
                    let r = logic.m.and_exists(differs, relation, &|v| foreign.contains_key(&v));
                    restatements.insert(differs, r);
                    r
                }
            };
            let logic = q.logic();
            if restated == Bdd::FALSE {
                continue;
            }
            let entry = arrivals
                .entry(block)
                .or_insert_with(|| vec![Bdd::FALSE; dst.params.len()]);
            entry[k] = logic.m.or(entry[k], restated);
        }
    }
}
