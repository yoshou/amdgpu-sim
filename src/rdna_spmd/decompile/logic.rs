use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Atom {
    Bit(ValueId),
    View(ValueId),
    Constant(u32),
    Fresh(usize, ValueId, u32),
    Term(usize, bool),
    Marker(ValueId),
}

#[derive(Default)]
pub struct Kept {
    pub queries: BTreeSet<ValueId>,

    pub words: BTreeSet<ValueId>,
}

enum Choices {
    Fixed {
        kept: BTreeSet<ValueId>,
    },

    Open {
        whole: Vec<Bdd>,
        live: Vec<bool>,
        all_local: Bdd,
    },
}

const MARKERS_LAST: u32 = 0xfffe_0000;

pub struct Logic {
    pub m: Manager,
    choices: Choices,
    markers_first: bool,
    vars: HashMap<Atom, u32>,
    atoms: HashMap<u32, Atom>,
    params: HashMap<ValueId, (usize, usize)>,
    constants: HashMap<u32, u32>,
    markers: HashMap<ValueId, u32>,
    listed: Vec<(ValueId, bool)>,
    detour: HashMap<Atom, u32>,
    bits: HashMap<ValueId, Bdd>,
    views: HashMap<ValueId, Bdd>,
    supports: HashMap<Bdd, Rc<Vec<u32>>>,
    edges: BTreeMap<(BlockId, usize), Rc<EdgeIndex>>,
    relations: BTreeMap<(BlockId, usize), Rc<Vec<Binding>>>,
    images: HashMap<(usize, usize, Bdd), Bdd>,
}

impl Logic {
    fn with(
        f: &Func,
        facts: &Facts,
        choices: Choices,
        markers_first: bool,
        listed: &[(ValueId, bool)],
    ) -> Self {
        let mut params = HashMap::default();
        for (rank, id) in facts.order.iter().enumerate() {
            for (index, &(v, _)) in f.blocks[id].params.iter().enumerate() {
                params.insert(v, (rank, index));
            }
        }
        assert!(listed.len() < 1 << 16, "too many conversion choices");
        let markers = listed
            .iter()
            .enumerate()
            .map(|(i, &(v, _))| (v, i as u32))
            .collect();
        Self {
            m: Manager::new(),
            choices,
            markers_first,
            vars: HashMap::default(),
            atoms: HashMap::default(),
            params,
            constants: HashMap::default(),
            markers,
            listed: listed.to_vec(),
            detour: HashMap::default(),
            bits: HashMap::default(),
            views: HashMap::default(),
            supports: HashMap::default(),
            edges: BTreeMap::new(),
            relations: BTreeMap::new(),
            images: HashMap::default(),
        }
    }

    pub fn fixed(
        f: &Func,
        facts: &Facts,
        kept: &BTreeSet<ValueId>,
        tags: &[(ValueId, bool)],
    ) -> Self {
        let choices = Choices::Fixed { kept: kept.clone() };
        Self::with(f, facts, choices, false, tags)
    }

    pub fn open(f: &Func, facts: &Facts, listed: &[(ValueId, bool)]) -> Self {
        let placeholder = Choices::Fixed {
            kept: BTreeSet::new(),
        };
        let mut logic = Self::with(f, facts, placeholder, true, listed);
        let mut whole: Vec<Bdd> = facts
            .materialized
            .iter()
            .map(|&v| Manager::constant(v))
            .collect();
        let mut pending = Vec::new();
        let mut all_local = Bdd::TRUE;
        for &(v, word) in listed {
            let local = logic.atom(Atom::Marker(v));
            all_local = logic.m.and(all_local, local);
            if word {
                let kept = logic.m.not(local);
                whole[v.0] = logic.m.or(whole[v.0], kept);
                pending.push(v);
            }
        }
        while let Some(v) = pending.pop() {
            for a in sources(f, facts, v) {
                if !facts.lane_word[a.0] {
                    continue;
                }
                let joined = logic.m.or(whole[a.0], whole[v.0]);
                if joined != whole[a.0] {
                    whole[a.0] = joined;
                    pending.push(a);
                }
            }
        }
        logic.choices = Choices::Open {
            whole,
            live: live_values(f, facts),
            all_local,
        };
        logic
    }

    pub fn local(&mut self, v: ValueId) -> Bdd {
        if let Choices::Fixed { kept } = &self.choices {
            return Manager::constant(!kept.contains(&v));
        }
        if self.markers.contains_key(&v) {
            self.atom(Atom::Marker(v))
        } else {
            Bdd::TRUE
        }
    }

    pub fn materialized(&self, facts: &Facts, v: ValueId) -> Bdd {
        match &self.choices {
            Choices::Fixed { .. } => Manager::constant(facts.materialized[v.0]),
            Choices::Open { whole, .. } => whole[v.0],
        }
    }

    pub fn tag(&mut self, v: ValueId) -> Bdd {
        if matches!(self.choices, Choices::Fixed { .. }) && self.markers.contains_key(&v) {
            self.atom(Atom::Marker(v))
        } else {
            Bdd::TRUE
        }
    }

    pub fn carried(&self, param: ValueId) -> bool {
        match &self.choices {
            Choices::Fixed { .. } => true,
            Choices::Open { live, .. } => live[param.0],
        }
    }

    pub fn all_local(&self) -> Bdd {
        match self.choices {
            Choices::Fixed { .. } => Bdd::TRUE,
            Choices::Open { all_local, .. } => all_local,
        }
    }

    pub fn possible_policies(&mut self, condition: Bdd) -> Bdd {
        let varying: Vec<u32> = self
            .support(condition)
            .iter()
            .copied()
            .filter(|&var| !matches!(self.atoms[&var], Atom::Marker(_)))
            .collect();
        self.exists(&varying, condition)
    }

    pub fn choose(&mut self, mut safe: Bdd) -> Kept {
        assert_ne!(safe, Bdd::FALSE);
        let mut kept = Kept::default();
        for (v, word) in self.listed.clone() {
            let var = self.vars[&Atom::Marker(v)];
            let local = self.m.cofactor(safe, var, true);
            if local != Bdd::FALSE {
                safe = local;
                continue;
            }
            safe = self.m.cofactor(safe, var, false);
            if word {
                kept.words.insert(v);
            } else {
                kept.queries.insert(v);
            }
        }
        assert_eq!(safe, Bdd::TRUE);
        kept
    }

    pub fn settled(&mut self, f: Bdd, kept: &BTreeSet<ValueId>) -> Bdd {
        let markers: HashMap<u32, Bdd> = self
            .support(f)
            .iter()
            .filter_map(|&var| match self.atoms[&var] {
                Atom::Marker(v) => Some((var, Manager::constant(!kept.contains(&v)))),
                _ => None,
            })
            .collect();
        if markers.is_empty() {
            return f;
        }
        self.m.compose(f, &|var| markers.get(&var).copied())
    }

    pub fn exists(&mut self, vars: &[u32], f: Bdd) -> Bdd {
        if vars.is_empty() {
            return f;
        }
        let mut vars = vars.to_vec();
        vars.sort_unstable();
        self.m.exists(f, &|v| vars.binary_search(&v).is_ok())
    }

    pub fn forall(&mut self, vars: &[u32], f: Bdd) -> Bdd {
        if vars.is_empty() {
            return f;
        }
        let mut vars = vars.to_vec();
        vars.sort_unstable();
        self.m.forall(f, &|v| vars.binary_search(&v).is_ok())
    }

    pub fn atom(&mut self, atom: Atom) -> Bdd {
        let var = match self.vars.get(&atom) {
            Some(&var) => var,
            None => {
                let var = self.number(atom);
                self.atoms.insert(var, atom);
                self.vars.insert(atom, var);
                var
            }
        };
        self.m.var(var)
    }

    fn number(&mut self, atom: Atom) -> u32 {
        let var = match atom {
            Atom::Marker(v) => {
                let i = self.markers[&v];
                return if self.markers_first {
                    i
                } else {
                    MARKERS_LAST + i
                };
            }
            Atom::Bit(v) | Atom::View(v) => {
                let view = matches!(atom, Atom::View(_)) as u32;
                match self.params.get(&v) {
                    Some(&(rank, index)) => {
                        assert!(
                            index < 1 << 13 && rank < 1 << 16,
                            "register layout too large"
                        );
                        ((index as u32) << 17) | (view << 16) | rank as u32
                    }
                    None => {
                        assert!(v.0 < 1 << 28, "function too large");
                        (1 << 30) | ((v.0 as u32) << 1) | view
                    }
                }
            }
            Atom::Constant(k) => {
                let next = self.constants.len() as u32;
                (2 << 30) | *self.constants.entry(k).or_insert(next)
            }
            Atom::Fresh(..) | Atom::Term(..) => {
                let next = self.detour.len() as u32;
                assert!(next < 1 << 29, "too many detour values");
                (3 << 30) | *self.detour.entry(atom).or_insert(next)
            }
        };
        var + (1 << 16)
    }

    pub fn atom_of(&self, var: u32) -> Atom {
        self.atoms[&var]
    }

    pub fn support(&mut self, f: Bdd) -> Rc<Vec<u32>> {
        if let Some(s) = self.supports.get(&f) {
            return s.clone();
        }
        let s: Rc<Vec<u32>> = Rc::new(self.m.support(f).into_iter().collect());
        self.supports.insert(f, s.clone());
        s
    }

    pub fn uniform_atom(&self, facts: &Facts, var: u32) -> bool {
        match self.atoms[&var] {
            Atom::Bit(v) => facts.uniform[v.0],
            Atom::View(v) => facts.saturated[v.0],
            Atom::Marker(_) => true,
            Atom::Constant(_) | Atom::Fresh(..) | Atom::Term(..) => false,
        }
    }

    pub fn scope(&self, facts: &Facts, var: u32) -> Option<BlockId> {
        match self.atoms[&var] {
            Atom::Bit(v) | Atom::View(v) => match facts.site[v.0] {
                Site::Param { block, .. } | Site::Inst { block, .. } => Some(block),
                Site::Unreached => None,
            },
            Atom::Constant(_) | Atom::Marker(_) | Atom::Fresh(..) | Atom::Term(..) => None,
        }
    }

    fn scoped(&mut self, facts: &Facts, f: Bdd, block: BlockId) -> Vec<u32> {
        self.support(f)
            .iter()
            .copied()
            .filter(|&v| self.scope(facts, v) == Some(block))
            .collect()
    }

    pub fn bit(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Bdd {
        if let Some(&b) = self.bits.get(&v) {
            return b;
        }
        let b = self.compute_bit(f, facts, v);
        self.bits.insert(v, b);
        b
    }

    fn compute_bit(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Bdd {
        let opaque = Atom::Bit(v);
        let Some(inst) = facts.inst(f, v) else {
            return self.atom(opaque);
        };
        match inst {
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::Any),
                inputs,
                outputs,
                ..
            } => {
                let local = self.local(outputs[0].0);
                if local == Bdd::FALSE {
                    return self.atom(opaque);
                }
                let bit = self.bit(f, facts, inputs[0]);
                if local == Bdd::TRUE {
                    return bit;
                }
                let wave = self.atom(opaque);
                self.m.ite(local, bit, wave)
            }
            Inst::Core { op, .. } => match *op {
                Op::Const(_, k) => Manager::constant(k != 0),
                Op::Env(Env::ValidLane) => Bdd::TRUE,
                Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                    let (a, b) = (self.bit(f, facts, a), self.bit(f, facts, b));
                    match k {
                        IntOp::And => self.m.and(a, b),
                        IntOp::Or => self.m.or(a, b),
                        _ => self.m.xor(a, b),
                    }
                }
                Op::Select(c, a, b) => {
                    let c = self.bit(f, facts, c);
                    let a = self.bit(f, facts, a);
                    let b = self.bit(f, facts, b);
                    self.m.ite(c, a, b)
                }
                Op::Convert(Cvt::Bitcast, Ty::I1, a) => self.bit(f, facts, a),
                Op::Convert(Cvt::Trunc, Ty::I1, s) => match projected_word(f, facts, s) {
                    Some(w) => self.view(f, facts, w),
                    None => self.atom(opaque),
                },
                Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b) => {
                    self.compare(f, facts, p, a, b, opaque, &mut HashMap::default())
                }
                _ => self.atom(opaque),
            },
            _ => self.atom(opaque),
        }
    }

    fn compare(
        &mut self,
        f: &Func,
        facts: &Facts,
        pred: IntPred,
        a: ValueId,
        b: ValueId,
        opaque: Atom,
        memo: &mut HashMap<(ValueId, ValueId), Bdd>,
    ) -> Bdd {
        let test = lane_test(f, facts, a, b);
        if let Some(w) = test {
            if self.materialized(facts, w) == Bdd::FALSE {
                let bit = self.view(f, facts, w);
                return if pred == IntPred::Ne {
                    bit
                } else {
                    self.m.not(bit)
                };
            }
        }
        if let Some(&g) = memo.get(&(a, b)) {
            return g;
        }
        let g = if a == b {
            Manager::constant(pred == IntPred::Eq)
        } else if let (Some(x), Some(y)) = (facts.constant(f, a), facts.constant(f, b)) {
            Manager::constant((x == y) == (pred == IntPred::Eq))
        } else if f.types[a.0] == Ty::I1 {
            let (x, y) = (self.bit(f, facts, a), self.bit(f, facts, b));
            if pred == IntPred::Ne {
                self.m.xor(x, y)
            } else {
                self.m.iff(x, y)
            }
        } else if let Some(Op::Select(c, yes, no)) = facts.op(f, a) {
            let c = self.bit(f, facts, c);
            let yes = self.compare(f, facts, pred, yes, b, opaque, memo);
            let no = self.compare(f, facts, pred, no, b, opaque, memo);
            self.m.ite(c, yes, no)
        } else if let Some(Op::Select(c, yes, no)) = facts.op(f, b) {
            let c = self.bit(f, facts, c);
            let yes = self.compare(f, facts, pred, a, yes, opaque, memo);
            let no = self.compare(f, facts, pred, a, no, opaque, memo);
            self.m.ite(c, yes, no)
        } else {
            self.atom(opaque)
        };
        let g = match test {
            Some(w) => {
                let mode = self.materialized(facts, w);
                if mode == Bdd::TRUE {
                    g
                } else {
                    let bit = self.view(f, facts, w);
                    let lane = if pred == IntPred::Ne {
                        bit
                    } else {
                        self.m.not(bit)
                    };
                    self.m.ite(mode, g, lane)
                }
            }
            None => g,
        };
        memo.insert((a, b), g);
        g
    }

    pub fn view(&mut self, f: &Func, facts: &Facts, w: ValueId) -> Bdd {
        if let Some(&b) = self.views.get(&w) {
            return b;
        }
        let b = self.compute_view(f, facts, w);
        self.views.insert(w, b);
        b
    }

    fn compute_view(&mut self, f: &Func, facts: &Facts, w: ValueId) -> Bdd {
        let opaque = Atom::View(w);
        let Some(inst) = facts.inst(f, w) else {
            return self.atom(opaque);
        };
        match inst {
            Inst::Effect {
                op: EffectOp::Wave(WaveOp::Ballot),
                inputs,
                ..
            } => self.bit(f, facts, inputs[0]),
            Inst::Core { op, .. } => match *op {
                Op::Const(_, k) => match k as u32 {
                    0 => Bdd::FALSE,
                    u32::MAX => Bdd::TRUE,
                    k => self.atom(Atom::Constant(k)),
                },
                Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                    let (a, b) = (self.view(f, facts, a), self.view(f, facts, b));
                    match k {
                        IntOp::And => self.m.and(a, b),
                        IntOp::Or => self.m.or(a, b),
                        _ => self.m.xor(a, b),
                    }
                }
                Op::Select(c, a, b) => {
                    let c = self.bit(f, facts, c);
                    let a = self.view(f, facts, a);
                    let b = self.view(f, facts, b);
                    self.m.ite(c, a, b)
                }
                Op::Convert(Cvt::Bitcast, Ty::I32, a) if f.types[a.0] == Ty::I32 => {
                    self.view(f, facts, a)
                }
                _ => self.atom(opaque),
            },
            _ => self.atom(opaque),
        }
    }

    fn edge_index(&mut self, f: &Func, facts: &Facts, src: BlockId, slot: usize) -> Rc<EdgeIndex> {
        if let Some(index) = self.edges.get(&(src, slot)) {
            return index.clone();
        }
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = &f.blocks[&edge.dst];
        let mut index = EdgeIndex::default();
        for (k, (&(param, ty), &arg)) in dst.params.iter().zip(&edge.args).enumerate() {
            if !self.carried(param) {
                continue;
            }
            let bound = match ty {
                Ty::I1 => self.bit(f, facts, arg),
                Ty::I32 => {
                    index.words.entry(arg).or_default().push(k);
                    if !(facts.lane_word[param.0] || facts.lane_word[arg.0]) {
                        continue;
                    }
                    self.view(f, facts, arg)
                }
                _ => continue,
            };
            for &var in self.support(bound).iter() {
                index.by_var.entry(var).or_default().push(k);
            }
        }
        let index = Rc::new(index);
        self.edges.insert((src, slot), index.clone());
        index
    }

    pub fn image(
        &mut self,
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
        let r = self.compute_image(f, facts, src, slot, formula);
        self.images.insert((src.0, slot, formula), r);
        r
    }

    fn compute_image(
        &mut self,
        f: &Func,
        facts: &Facts,
        src: BlockId,
        slot: usize,
        formula: Bdd,
    ) -> Bdd {
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = &f.blocks[&edge.dst];
        let index = self.edge_index(f, facts, src, slot);
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let mut pending = self.scoped(facts, formula, src);
        let mut links = Vec::new();
        let mut used = vec![false; dst.params.len()];
        while let Some(var) = pending.pop() {
            if !seen.insert(var) {
                continue;
            }
            let mut bound: Vec<usize> = index.by_var.get(&var).cloned().unwrap_or_default();
            if let Atom::View(w) = self.atoms[&var] {
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
                    (Atom::Bit(param), self.bit(f, facts, arg))
                } else {
                    (Atom::View(param), self.view(f, facts, arg))
                };
                let link = self.binding(facts, src, atom, bound);
                pending.extend(link.support.iter().copied().filter(|v| !seen.contains(v)));
                links.push(link);
            }
        }
        self.project(facts, src, formula, &links)
    }

    fn binding(&mut self, facts: &Facts, src: BlockId, atom: Atom, bound: Bdd) -> Binding {
        let atom = self.atom(atom);
        let support = self.scoped(facts, bound, src);
        Binding {
            atom,
            bound,
            support,
        }
    }

    fn project(&mut self, facts: &Facts, src: BlockId, formula: Bdd, links: &[Binding]) -> Bdd {
        let mut renamed: HashMap<u32, Bdd> = HashMap::default();
        let mut equalities = Bdd::TRUE;
        let mut rest: Vec<usize> = Vec::new();
        for (i, link) in links.iter().enumerate() {
            let literal = match link.support.as_slice() {
                [v] => {
                    let var = self.m.var(*v);
                    if link.bound == var {
                        Some((*v, link.atom))
                    } else if link.bound == self.m.not(var) {
                        Some((*v, self.m.not(link.atom)))
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
                        let same = self.m.iff(stand_in, first);
                        equalities = self.m.and(equalities, same);
                    }
                },
                None => rest.push(i),
            }
        }
        if renamed.is_empty() {
            return self.schedule(facts, src, formula, links);
        }
        let formula = self.m.compose(formula, &|v| renamed.get(&v).copied());
        let formula = self.m.and(formula, equalities);
        let rest: Vec<Binding> = rest
            .into_iter()
            .map(|i| {
                let link = &links[i];
                Binding {
                    atom: link.atom,
                    bound: self.m.compose(link.bound, &|v| renamed.get(&v).copied()),
                    support: link
                        .support
                        .iter()
                        .copied()
                        .filter(|v| !renamed.contains_key(v))
                        .collect(),
                }
            })
            .collect();
        self.schedule(facts, src, formula, &rest)
    }

    fn schedule(&mut self, facts: &Facts, src: BlockId, formula: Bdd, links: &[Binding]) -> Bdd {
        let formula_atoms: BTreeSet<u32> = self.scoped(facts, formula, src).into_iter().collect();
        let mut occurrences: HashMap<u32, usize> = HashMap::default();
        for link in links {
            for &v in &link.support {
                *occurrences.entry(v).or_default() += 1;
            }
        }
        let mut pending: Vec<&Binding> = links
            .iter()
            .filter(|link| {
                link.support.is_empty()
                    || link.bound.constant().is_some()
                    || link
                        .support
                        .iter()
                        .any(|v| occurrences[v] > 1 || formula_atoms.contains(v))
            })
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
        let mut acc = self.exists(&lone, formula);
        let mut built: BTreeSet<u32> = self.support(acc).iter().copied().collect();
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
            let relation = self.m.iff(link.atom, link.bound);
            let mut finished = Vec::new();
            for &v in &link.support {
                let count = occurrences.get_mut(&v).unwrap();
                *count -= 1;
                if *count == 0 {
                    finished.push(v);
                }
            }
            acc = if finished.is_empty() {
                self.m.and(acc, relation)
            } else {
                finished.sort_unstable();
                self.m
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

    fn bindings(&mut self, f: &Func, facts: &Facts, src: BlockId, slot: usize) -> Rc<Vec<Binding>> {
        if let Some(b) = self.relations.get(&(src, slot)) {
            return b.clone();
        }
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = &f.blocks[&edge.dst];
        let mut links = Vec::new();
        for (&(param, ty), &arg) in dst.params.iter().zip(&edge.args) {
            if !self.carried(param) {
                continue;
            }
            let (atom, bound) = match ty {
                Ty::I1 => (Atom::Bit(param), self.bit(f, facts, arg)),
                Ty::I32 if facts.viewed[param.0] && !facts.materialized[param.0] => {
                    (Atom::View(param), self.view(f, facts, arg))
                }
                _ => continue,
            };
            links.push(self.binding(facts, src, atom, bound));
        }
        let links = Rc::new(links);
        self.relations.insert((src, slot), links.clone());
        links
    }

    fn post(&mut self, f: &Func, facts: &Facts, src: BlockId, slot: usize, formula: Bdd) -> Bdd {
        if formula == Bdd::FALSE {
            return formula;
        }
        let links = self.bindings(f, facts, src, slot);
        self.project(facts, src, formula, &links)
    }

    pub fn reach(
        &mut self,
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
                let unsent = self.m.not(before);
                self.m.restrict(whole, unsent)
            };
            let block = &f.blocks[&x];
            let conditions: Vec<(usize, Bdd)> = match &block.term {
                Term::Ret(_) => vec![],
                Term::Br(_) => vec![(0, r)],
                Term::CondBr { cond, .. } => {
                    let c = self.bit(f, facts, *cond);
                    let nc = self.m.not(c);
                    vec![(0, self.m.and(r, c)), (1, self.m.and(r, nc))]
                }
            };
            for (slot, g) in conditions {
                if g == Bdd::FALSE {
                    continue;
                }
                let dst = block.term.edges().nth(slot).unwrap().dst;
                let image = self.post(f, facts, x, slot, g);
                let old = reach.get(&dst).copied().unwrap_or(Bdd::FALSE);
                let joined = self.m.or(old, image);
                if joined != old {
                    reach.insert(dst, joined);
                    worklist.insert((rank[&dst], dst));
                }
            }
        }
        reach
    }
}

struct Binding {
    atom: Bdd,
    bound: Bdd,
    support: Vec<u32>,
}

#[derive(Default)]
struct EdgeIndex {
    by_var: HashMap<u32, Vec<usize>>,
    words: HashMap<ValueId, Vec<usize>>,
}

pub fn lane_test(f: &Func, facts: &Facts, a: ValueId, b: ValueId) -> Option<ValueId> {
    if facts.lane_word[a.0] && facts.constant(f, b) == Some(0) {
        Some(a)
    } else if facts.lane_word[b.0] && facts.constant(f, a) == Some(0) {
        Some(b)
    } else {
        None
    }
}

pub fn projected_word(f: &Func, facts: &Facts, s: ValueId) -> Option<ValueId> {
    match facts.op(f, s) {
        Some(Op::Int(IntOp::LShr, w, lane)) if facts.is_lane_id(f, lane) => Some(w),
        _ => None,
    }
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

pub fn choices(f: &Func, facts: &Facts) -> Vec<(ValueId, bool)> {
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
                } if live[outputs[0].0 .0] => out.push((outputs[0].0, false)),
                Inst::Core {
                    value,
                    op: Op::Cmp(IntPred::Eq | IntPred::Ne, a, b),
                    ..
                } if live[value.0] => {
                    if let Some(w) = lane_test(f, facts, *a, *b) {
                        if !facts.materialized[w.0] && words.insert(w) {
                            out.push((w, true));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    out
}
