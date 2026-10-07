use super::atoms::Atom;
use super::queries::{Queries, Rules};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

#[derive(Clone)]
pub struct Binding {
    pub atom: Bdd,
    pub bound: Bdd,
    pub support: Vec<u32>,
}

#[derive(Default)]
struct EdgeIndex {
    by_var: HashMap<u32, Vec<usize>>,
    words: HashMap<ValueId, Vec<usize>>,
}

#[derive(Default)]
pub(super) struct Edges {
    edges: BTreeMap<(BlockId, usize), Rc<EdgeIndex>>,
    relations: BTreeMap<(BlockId, usize), Rc<Vec<Binding>>>,
    images: HashMap<(usize, usize, Bdd), Bdd>,
    bridges: HashMap<(BlockId, usize), Rc<Vec<Binding>>>,
}

impl Edges {
    fn bridges<Q: Queries>(&mut self, q: &mut Q, f: &Func, facts: &Facts, src: BlockId, slot: usize) -> Rc<Vec<Binding>> {
        if let Some(b) = self.bridges.get(&(src, slot)) {
            return b.clone();
        }
        let links = Rc::new(Q::State::bridges(q, f, facts, src, slot));
        self.bridges.insert((src, slot), links.clone());
        links
    }

    fn edge_index<Q: Queries>(&mut self, q: &mut Q, f: &Func, facts: &Facts, src: BlockId, slot: usize) -> Rc<EdgeIndex> {
        if let Some(index) = self.edges.get(&(src, slot)) {
            return index.clone();
        }
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = &f.blocks[&edge.dst];
        let mut index = EdgeIndex::default();
        for (k, (&(param, ty), &arg)) in dst.params.iter().zip(&edge.args).enumerate() {
            if !q.carried(param) {
                continue;
            }
            let bound = match ty {
                Ty::I1 => q.bit(f, facts, arg),
                Ty::I32 | Ty::I64 => {
                    index.words.entry(arg).or_default().push(k);
                    if !(facts.lane_word[param.0] || facts.lane_word[arg.0]) {
                        continue;
                    }
                    q.view(f, facts, arg)
                }
                _ => continue,
            };
            for &var in q.support(bound).iter() {
                index.by_var.entry(var).or_default().push(k);
            }
        }
        let index = Rc::new(index);
        self.edges.insert((src, slot), index.clone());
        index
    }

    pub(super) fn image<Q: Queries>(
        &mut self,
        q: &mut Q,
        f: &Func,
        facts: &Facts,
        src: BlockId,
        slot: usize,
        formula: Bdd,
    ) -> Bdd {
        if formula.constant().is_some() {
            return formula;
        }
        if let Some(&r) = self.images.get(&(src.0, slot, formula)) {
            return r;
        }
        let r = self.compute_image(q, f, facts, src, slot, formula);
        self.images.insert((src.0, slot, formula), r);
        r
    }

    fn compute_image<Q: Queries>(
        &mut self,
        q: &mut Q,
        f: &Func,
        facts: &Facts,
        src: BlockId,
        slot: usize,
        formula: Bdd,
    ) -> Bdd {
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = &f.blocks[&edge.dst];
        let index = self.edge_index(q, f, facts, src, slot);
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let mut pending = q.scoped(facts, formula, src);
        let mut links = Vec::new();
        let mut used = vec![false; dst.params.len()];
        while let Some(var) = pending.pop() {
            if !seen.insert(var) {
                continue;
            }
            let mut bound: Vec<usize> = index.by_var.get(&var).cloned().unwrap_or_default();
            if let Atom::View(w) = q.atoms().of(var) {
                bound.extend(index.words.get(&w).into_iter().flatten().copied());
            }
            for k in bound {
                if used[k] {
                    continue;
                }
                used[k] = true;
                let (param, ty) = dst.params[k];
                let arg = edge.args[k];
                let (atom, bound) = if ty == Ty::I1 {
                    (Atom::Bit(param), q.bit(f, facts, arg))
                } else {
                    (Atom::View(param), q.view(f, facts, arg))
                };
                let link = binding(q, facts, src, arriving(src, edge.dst, atom), bound);
                pending.extend(link.support.iter().copied().filter(|v| !seen.contains(v)));
                links.push(link);
            }
        }
        for link in self.bridges(q, f, facts, src, slot).iter() {
            if link.support.iter().any(|v| seen.contains(v)) {
                links.push(link.clone());
            }
        }
        let r = project(q, facts, src, formula, &links);
        arrived(q, src, edge.dst, r)
    }

    fn bindings<Q: Queries>(&mut self, q: &mut Q, f: &Func, facts: &Facts, src: BlockId, slot: usize) -> Rc<Vec<Binding>> {
        if let Some(b) = self.relations.get(&(src, slot)) {
            return b.clone();
        }
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = &f.blocks[&edge.dst];
        let mut links = Vec::new();
        for (&(param, ty), &arg) in dst.params.iter().zip(&edge.args) {
            if !q.carried(param) {
                continue;
            }
            let (atom, bound) = match ty {
                Ty::I1 => (Atom::Bit(param), q.bit(f, facts, arg)),
                Ty::I32 | Ty::I64 if facts.viewed[param.0] && !facts.materialized[param.0] => {
                    (Atom::View(param), q.view(f, facts, arg))
                }
                _ => continue,
            };
            links.push(binding(q, facts, src, arriving(src, edge.dst, atom), bound));
        }
        links.extend(self.bridges(q, f, facts, src, slot).iter().cloned());
        let links = Rc::new(links);
        self.relations.insert((src, slot), links.clone());
        links
    }

    pub(super) fn post<Q: Queries>(&mut self, q: &mut Q, f: &Func, facts: &Facts, src: BlockId, slot: usize, formula: Bdd) -> Bdd {
        if formula == Bdd::FALSE {
            return formula;
        }
        let links = self.bindings(q, f, facts, src, slot);
        let r = project(q, facts, src, formula, &links);
        let dst = f.blocks[&src].term.edges().nth(slot).unwrap().dst;
        arrived(q, src, dst, r)
    }

    pub(super) fn reach<Q: Queries>(
        &mut self,
        q: &mut Q,
        f: &Func,
        facts: &Facts,
        start: BlockId,
        formula: Bdd,
    ) -> BTreeMap<BlockId, Bdd> {
        let rank: BTreeMap<BlockId, usize> = facts
            .order
            .iter()
            .enumerate()
            .map(|(r, &b)| (b, r))
            .collect();
        let mut reach = BTreeMap::from([(start, formula)]);
        let mut sent: BTreeMap<BlockId, Bdd> = BTreeMap::new();
        let mut worklist: BTreeSet<(usize, BlockId)> = BTreeSet::from([(rank[&start], start)]);
        while let Some((_, x)) = worklist.pop_first() {
            let whole = reach[&x];
            let before = sent.insert(x, whole).unwrap_or(Bdd::FALSE);
            let r = if before == Bdd::FALSE {
                whole
            } else {
                let unsent = q.m().not(before);
                q.m().restrict(whole, unsent)
            };
            let block = &f.blocks[&x];
            let conditions: Vec<(usize, Bdd)> = match &block.term {
                Term::Ret(_) => vec![],
                Term::Br(_) => vec![(0, r)],
                Term::CondBr { cond, .. } => {
                    let c = q.bit(f, facts, *cond);
                    let nc = q.m().not(c);
                    vec![(0, q.m().and(r, c)), (1, q.m().and(r, nc))]
                }
            };
            for (slot, g) in conditions {
                if g == Bdd::FALSE {
                    continue;
                }
                let dst = block.term.edges().nth(slot).unwrap().dst;
                let image = self.post(q, f, facts, x, slot, g);
                let old = reach.get(&dst).copied().unwrap_or(Bdd::FALSE);
                let joined = q.m().or(old, image);
                if joined != old {
                    reach.insert(dst, joined);
                    worklist.insert((rank[&dst], dst));
                }
            }
        }
        reach
    }
}

fn arrived<Q: Queries>(q: &mut Q, src: BlockId, dst: BlockId, r: Bdd) -> Bdd {
    if src != dst || r.constant().is_some() {
        return r;
    }
    let next: Vec<(u32, Atom)> = q
        .support(r)
        .iter()
        .filter_map(|&var| match q.atoms().of(var) {
            Atom::Next(v, false) => Some((var, Atom::Bit(v))),
            Atom::Next(v, true) => Some((var, Atom::View(v))),
            _ => None,
        })
        .collect();
    if next.is_empty() {
        return r;
    }
    let renamed: HashMap<u32, Bdd> = next.into_iter().map(|(var, atom)| (var, q.atom(atom))).collect();
    q.m().compose(r, &|v| renamed.get(&v).copied())
}

fn binding<Q: Queries>(q: &mut Q, facts: &Facts, src: BlockId, atom: Atom, bound: Bdd) -> Binding {
    let atom = q.atom(atom);
    let support = q.scoped(facts, bound, src);
    Binding {
        atom,
        bound,
        support,
    }
}

fn project<Q: Queries>(q: &mut Q, facts: &Facts, src: BlockId, formula: Bdd, links: &[Binding]) -> Bdd {
    let mut renamed: HashMap<u32, Bdd> = HashMap::default();
    let mut equalities = Bdd::TRUE;
    let mut rest: Vec<usize> = Vec::new();
    for (i, link) in links.iter().enumerate() {
        let literal = match link.support.as_slice() {
            [v] => {
                let var = q.m().var(*v);
                if link.bound == var {
                    Some((*v, link.atom))
                } else if link.bound == q.m().not(var) {
                    Some((*v, q.m().not(link.atom)))
                } else {
                    None
                }
            }
            _ => None,
        };
        match literal {
            Some((v, stand_in)) => match renamed.get(&v) {
                None => {
                    renamed.insert(v, stand_in);
                }
                Some(&first) => {
                    let same = q.m().iff(stand_in, first);
                    equalities = q.m().and(equalities, same);
                }
            },
            None => rest.push(i),
        }
    }
    if renamed.is_empty() {
        return schedule(q, facts, src, formula, links);
    }
    let formula = q.m().compose(formula, &|v| renamed.get(&v).copied());
    let formula = q.m().and(formula, equalities);
    let rest: Vec<Binding> = rest
        .into_iter()
        .map(|i| {
            let link = &links[i];
            Binding {
                atom: link.atom,
                bound: q.m().compose(link.bound, &|v| renamed.get(&v).copied()),
                support: link
                    .support
                    .iter()
                    .copied()
                    .filter(|v| !renamed.contains_key(v))
                    .collect(),
            }
        })
        .collect();
    schedule(q, facts, src, formula, &rest)
}

fn schedule<Q: Queries>(q: &mut Q, facts: &Facts, src: BlockId, formula: Bdd, links: &[Binding]) -> Bdd {
    let formula_atoms: BTreeSet<u32> = q.scoped(facts, formula, src).into_iter().collect();
    let mut occurrences: HashMap<u32, usize> = HashMap::default();
    for link in links {
        for &v in &link.support {
            *occurrences.entry(v).or_default() += 1;
        }
    }
    let foreign: Vec<bool> = links
        .iter()
        .map(|link| q.support(link.bound).iter().any(|v| !link.support.contains(v)))
        .collect();
    let mut pending: Vec<&Binding> = links
        .iter()
        .zip(&foreign)
        .filter(|&(link, &foreign)| {
            foreign
                || link.support.is_empty()
                || link.bound.constant().is_some()
                || link
                    .support
                    .iter()
                    .any(|v| occurrences[v] > 1 || formula_atoms.contains(v))
        })
        .map(|(link, _)| link)
        .collect();
    occurrences.clear();
    for link in &pending {
        for &v in &link.support {
            *occurrences.entry(v).or_default() += 1;
        }
    }
    let lone: Vec<u32> = formula_atoms
        .iter()
        .copied()
        .filter(|v| !occurrences.contains_key(v))
        .collect();
    let mut acc = q.exists(&lone, formula);
    let mut built: BTreeSet<u32> = q.support(acc).iter().copied().collect();
    while !pending.is_empty() {
        let mut best = 0;
        let mut best_key = None;
        for (j, link) in pending.iter().enumerate() {
            let shared = link.support.iter().filter(|v| built.contains(v)).count();
            let key = (shared, std::cmp::Reverse(link.support.len()));
            if best_key.map_or(true, |k| key > k) {
                best = j;
                best_key = Some(key);
            }
        }
        let link = pending.swap_remove(best);
        let relation = q.m().iff(link.atom, link.bound);
        let mut finished = Vec::new();
        for &v in &link.support {
            let count = occurrences.get_mut(&v).unwrap();
            *count -= 1;
            if *count == 0 {
                finished.push(v);
            }
        }
        acc = if finished.is_empty() {
            q.m().and(acc, relation)
        } else {
            finished.sort_unstable();
            q.m()
                .and_exists(acc, relation, &|v| finished.binary_search(&v).is_ok())
        };
        if acc == Bdd::FALSE {
            return acc;
        }
        built.extend(link.support.iter().copied());
        for v in &finished {
            built.remove(v);
        }
    }
    acc
}

fn arriving(src: BlockId, dst: BlockId, atom: Atom) -> Atom {
    match atom {
        Atom::Bit(v) if src == dst => Atom::Next(v, false),
        Atom::View(v) if src == dst => Atom::Next(v, true),
        other => other,
    }
}
