use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::analysis::bdd::HashMap;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Atom {
    Bit(ValueId),
    View(ValueId),
    Constant(u32),
    Fresh(usize, ValueId, u32),
    Term(usize, usize, bool),
    /// A conversion choice: true answers an `any` locally, or allows a
    /// lane word to be represented by its own bit. False keeps the query or
    /// requires the word to remain whole.
    Marker(ValueId),
}

/// The wave operations the lane program keeps as operations over the lanes
/// that are at them, instead of answering from the lane's own values.
#[derive(Default)]
pub(super) struct Kept {
    /// `any` queries kept as operations over the lanes at them.
    pub queries: BTreeSet<ValueId>,
    /// Lane words kept as the words the wave computes, because a query kept
    /// over them or an operation reads them whole.
    pub words: BTreeSet<ValueId>,
}

pub(super) struct Logic {
    pub m: Manager,
    vars: HashMap<Atom, u32>,
    atoms: HashMap<u32, Atom>,
    params: HashMap<ValueId, (usize, usize)>,
    constants: HashMap<u32, u32>,
    markers: HashMap<ValueId, u32>,
    fresh: HashMap<(usize, ValueId, u32), u32>,
    kept: BTreeSet<ValueId>,
    bits: HashMap<ValueId, Bdd>,
    views: HashMap<ValueId, Bdd>,
    supports: HashMap<Bdd, Vec<u32>>,
    policy: Option<Vec<Bdd>>,
    choices: Vec<(ValueId, bool)>,
    abstract_queries: bool,
    /// Leaves the words kept whole out of the relation reach follows along an
    /// edge, so reach takes their bits to be anything. Relating them is most
    /// of what reach costs where queries are answered by the wave, since the
    /// lanes a query brings along hold them in every combination.
    open_words: bool,
    pub all_local: Bdd,
    edges: std::collections::BTreeMap<(BlockId, usize), std::rc::Rc<EdgeIndex>>,
    relations: std::collections::BTreeMap<(BlockId, usize), std::rc::Rc<Vec<Binding>>>,
}

impl Logic {
    /// `kept` holds the `any` queries the lane program keeps, by the value
    /// each defines; a test of a lane word is kept with the word.
    pub fn new(f: &Func, facts: &Facts, kept: &BTreeSet<ValueId>) -> Self {
        let mut params = HashMap::default();
        for (rank, id) in facts.order.iter().enumerate() {
            for (index, &(v, _)) in f.blocks[id].params.iter().enumerate() {
                params.insert(v, (rank, index));
            }
        }
        Self {
            m: Manager::new(),
            vars: HashMap::default(),
            atoms: HashMap::default(),
            params,
            constants: HashMap::default(),
            markers: HashMap::default(),
            fresh: HashMap::default(),
            kept: kept.clone(),
            bits: HashMap::default(),
            views: HashMap::default(),
            supports: HashMap::default(),
            policy: None,
            choices: Vec::new(),
            abstract_queries: false,
            open_words: false,
            all_local: Bdd::TRUE,
            edges: std::collections::BTreeMap::new(),
            relations: std::collections::BTreeMap::new(),
        }
    }

    /// Interpret every conversion policy in the same formulas. A marker is
    /// true when a query is answered locally; word materialization propagates
    /// backwards from the tests that choose to read the whole word.
    pub fn policies(f: &Func, facts: &Facts) -> Self {
        let mut logic = Self::new(f, facts, &BTreeSet::new());
        let mut whole: Vec<Bdd> = facts
            .materialized
            .iter()
            .map(|&v| Manager::constant(v))
            .collect();
        let live = super::policy::live_values(f, facts);
        let mut pending = Vec::new();
        let mut words = BTreeSet::new();
        for &id in &facts.order {
            for inst in &f.blocks[&id].insts {
                match inst {
                    Inst::Effect {
                        op: EffectOp::Wave(WaveOp::Any),
                        outputs,
                        ..
                    } if live[outputs[0].0 .0] => {
                        let selector = outputs[0].0;
                        logic.atom(Atom::Marker(selector));
                        logic.choices.push((outputs[0].0, false));
                    }
                    Inst::Core {
                        value,
                        op: Op::Cmp(IntPred::Eq | IntPred::Ne, a, b),
                        ..
                    } if live[value.0] => {
                        if let Some(w) = lane_test(f, facts, *a, *b) {
                            if !facts.materialized[w.0] && words.insert(w) {
                                let selector = w;
                                let local = logic.atom(Atom::Marker(selector));
                                let kept = logic.m.not(local);
                                whole[w.0] = logic.m.or(whole[w.0], kept);
                                logic.choices.push((w, true));
                                pending.push(w);
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        while let Some(v) = pending.pop() {
            let sources: Vec<ValueId> = match facts.site[v.0] {
                Site::Param { block, index } if block != f.entry => {
                    facts.arguments(f, block, index).collect()
                }
                Site::Inst { .. } => match facts.op(f, v) {
                    Some(Op::Int(_, a, b)) | Some(Op::Select(_, a, b)) => vec![a, b],
                    Some(Op::Convert(_, _, a)) => vec![a],
                    _ => vec![],
                },
                _ => vec![],
            };
            for a in sources.into_iter().filter(|a| facts.lane_word[a.0]) {
                let next = logic.m.or(whole[a.0], whole[v.0]);
                if next != whole[a.0] {
                    whole[a.0] = next;
                    pending.push(a);
                }
            }
        }
        for &(v, _) in &logic.choices.clone() {
            let local = logic.atom(Atom::Marker(v));
            logic.all_local = logic.m.and(logic.all_local, local);
        }
        logic.policy = Some(whole);
        logic
    }

    pub fn materialized(&self, facts: &Facts, v: ValueId) -> Bdd {
        self.policy
            .as_ref()
            .map_or_else(|| Manager::constant(facts.materialized[v.0]), |p| p[v.0])
    }

    pub fn local(&mut self, v: ValueId) -> Bdd {
        if self.policy.is_some() {
            if self.markers.contains_key(&v) {
                self.atom(Atom::Marker(v))
            } else {
                Bdd::TRUE
            }
        } else {
            Manager::constant(!self.kept.contains(&v))
        }
    }

    /// Eliminate execution-state variables, retaining only policy choices.
    pub fn possible_policies(&mut self, condition: Bdd) -> Bdd {
        let varying: BTreeSet<_> = self
            .support(condition)
            .into_iter()
            .filter(|&var| !matches!(self.atom_of(var), Atom::Marker(_)))
            .collect();
        self.m.exists(condition, &|var| varying.contains(&var))
    }

    /// Prefer a local answer whenever the remaining safety conditions allow
    /// it. This chooses a model of an already proved formula, not another
    /// invocation of the program proof.
    pub fn choose(&mut self, mut safe: Bdd) -> Kept {
        assert_ne!(safe, Bdd::FALSE);
        let mut kept = Kept::default();
        for &(v, word) in &self.choices {
            let var = self.vars[&Atom::Marker(v)];
            let local = self.m.cofactor(safe, var, true);
            let local = if local != Bdd::FALSE {
                safe = local;
                true
            } else {
                safe = self.m.cofactor(safe, var, false);
                false
            };
            if !local {
                if word {
                    kept.words.insert(v);
                } else {
                    kept.queries.insert(v);
                }
            }
        }
        assert_eq!(safe, Bdd::TRUE);
        kept
    }

    fn clear_values(&mut self) {
        self.bits.clear();
        self.views.clear();
        self.edges.clear();
        self.relations.clear();
    }

    /// Reachability under the policies this logic interprets. It is exact
    /// for the policy that answers everything locally. For the others,
    /// unconstrained query results and words kept whole safely overapproximate
    /// reach. This avoids
    /// enumerating combinations of unrelated branch choices in the
    /// reachability relation; value differences still carry their choices.
    pub fn policy_reach(
        &mut self,
        f: &Func,
        facts: &Facts,
        start: BlockId,
        initial: Bdd,
    ) -> std::collections::BTreeMap<BlockId, Bdd> {
        if self.policy.is_none() {
            return self.reach(f, facts, start, initial);
        }
        let policy = self.policy.take();
        self.clear_values();
        self.abstract_queries = true;
        self.open_words = true;
        let general = self.reach(f, facts, start, initial);
        self.abstract_queries = false;
        self.open_words = false;
        self.policy = policy;
        self.clear_values();
        general
    }

    /// Reach that takes the bits of the words kept whole to be anything.
    pub fn open_reach(
        &mut self,
        f: &Func,
        facts: &Facts,
        start: BlockId,
        initial: Bdd,
    ) -> std::collections::BTreeMap<BlockId, Bdd> {
        self.open_words = true;
        let reach = self.reach(f, facts, start, initial);
        self.open_words = false;
        self.relations.clear();
        reach
    }

    pub fn atom(&mut self, atom: Atom) -> Bdd {
        let var = match self.vars.get(&atom) {
            Some(&var) => var,
            None => {
                let var = match atom {
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
                                assert!(v.0 < 1 << 29, "function too large");
                                (1 << 30) | ((v.0 as u32) << 1) | view
                            }
                        }
                    }
                    Atom::Constant(k) => {
                        let next = self.constants.len() as u32;
                        (2 << 30) | *self.constants.entry(k).or_insert(next)
                    }
                    Atom::Marker(v) => {
                        let next = self.markers.len() as u32;
                        assert!(next < 1 << 16, "too many conversion choices");
                        *self.markers.entry(v).or_insert(next)
                    }
                    Atom::Fresh(d, v, position) => {
                        let next = self.fresh.len() as u32;
                        assert!(next < (1 << 30) - (1 << 16) - 1, "too many detour values");
                        (3 << 30) | *self.fresh.entry((d, v, position)).or_insert(next)
                    }
                    Atom::Term(d, t, view) => {
                        let next = self.fresh.len() as u32;
                        assert!(next < (1 << 30) - (1 << 16) - 1, "too many detour values");
                        let key = (usize::MAX - d, ValueId(t), view as u32);
                        (3 << 30) | *self.fresh.entry(key).or_insert(next)
                    }
                };
                // Put policy choices before the execution-state variables:
                // mux selectors after their inputs duplicate large subgraphs.
                let var = if matches!(atom, Atom::Marker(_)) {
                    var
                } else {
                    var.checked_add(1 << 16).expect("too many BDD variables")
                };
                self.atoms.insert(var, atom);
                self.vars.insert(atom, var);
                var
            }
        };
        self.m.var(var)
    }

    pub fn atom_of(&self, var: u32) -> Atom {
        self.atoms[&var]
    }

    /// The variables a function tests, in their order.
    pub fn support(&mut self, f: Bdd) -> Vec<u32> {
        if let Some(s) = self.supports.get(&f) {
            return s.clone();
        }
        let s: Vec<u32> = self.m.support(f).into_iter().collect();
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

    /// A formula with every marker settled: the queries in `kept` answered by
    /// the wave, every other query by the lane.
    pub fn settled(&mut self, f: Bdd, kept: &[ValueId]) -> Bdd {
        let markers: HashMap<u32, Bdd> = self
            .support(f)
            .into_iter()
            .filter_map(|var| match self.atoms[&var] {
                Atom::Marker(v) => Some((var, Manager::constant(!kept.contains(&v)))),
                _ => None,
            })
            .collect();
        if markers.is_empty() {
            return f;
        }
        self.m.compose(f, &|var| markers.get(&var).copied())
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
                if self.abstract_queries {
                    return self.atom(opaque);
                }
                let local = self.local(outputs[0].0);
                let bit = self.bit(f, facts, inputs[0]);
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
                Op::Convert(Cvt::Trunc, Ty::I1, s) => match facts.op(f, s) {
                    Some(Op::Int(IntOp::LShr, w, lane)) if facts.is_lane_id(f, lane) => {
                        self.view(f, facts, w)
                    }
                    _ => self.atom(opaque),
                },
                Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b) => {
                    self.compare(f, facts, p, a, b, opaque, &mut HashMap::default())
                }
                _ => self.atom(opaque),
            },
            _ => self.atom(opaque),
        }
    }

    /// Interpret equality through selects rather than inventing an unrelated
    /// bit for a boolean encoded as an integer. On arms we cannot interpret,
    /// the original comparison's bit remains an unknown answer.
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
        if self.abstract_queries && lane_test(f, facts, a, b).is_some() {
            return self.atom(opaque);
        }
        if let Some(w) = lane_test(f, facts, a, b) {
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
        } else if let (Some(a), Some(b)) = (facts.constant(f, a), facts.constant(f, b)) {
            Manager::constant((a == b) == (pred == IntPred::Eq))
        } else if f.types[a.0] == Ty::I1 {
            let (a, b) = (self.bit(f, facts, a), self.bit(f, facts, b));
            if pred == IntPred::Ne {
                self.m.xor(a, b)
            } else {
                self.m.iff(a, b)
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
        let g = if let Some(w) = lane_test(f, facts, a, b) {
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
        } else {
            g
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

    fn edge_index(
        &mut self,
        f: &Func,
        facts: &Facts,
        src: BlockId,
        slot: usize,
    ) -> std::rc::Rc<EdgeIndex> {
        if let Some(index) = self.edges.get(&(src, slot)) {
            return index.clone();
        }
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = &f.blocks[&edge.dst];
        let mut index = EdgeIndex::default();
        for (k, (&(param, ty), &arg)) in dst.params.iter().zip(&edge.args).enumerate() {
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
            for var in self.support(bound) {
                index.by_var.entry(var).or_default().push(k);
            }
        }
        let index = std::rc::Rc::new(index);
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
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = &f.blocks[&edge.dst];
        let index = self.edge_index(f, facts, src, slot);
        let mut seen: BTreeSet<u32> = BTreeSet::new();
        let mut pending: Vec<u32> = self
            .support(formula)
            .into_iter()
            .filter(|&v| self.scope(facts, v) == Some(src))
            .collect();
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
                let (atom, formula) = if ty == Ty::I1 {
                    (Atom::Bit(param), self.bit(f, facts, arg))
                } else {
                    (Atom::View(param), self.view(f, facts, arg))
                };
                let link = self.binding(facts, src, atom, formula);
                for &v in &link.support {
                    if !seen.contains(&v) {
                        pending.push(v);
                    }
                }
                links.push(link);
            }
        }
        self.project(facts, src, formula, &links)
    }

    fn binding(&mut self, facts: &Facts, src: BlockId, atom: Atom, bound: Bdd) -> Binding {
        let atom = self.atom(atom);
        let support = self
            .support(bound)
            .into_iter()
            .filter(|&v| self.scope(facts, v) == Some(src))
            .collect();
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
                    let negated = self.m.not(var);
                    if link.bound == var {
                        Some((*v, link.atom))
                    } else if link.bound == negated {
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
        if !renamed.is_empty() {
            let formula = self.m.compose(formula, &|v| renamed.get(&v).copied());
            let formula = self.m.and(formula, equalities);
            let rest: Vec<Binding> = rest
                .into_iter()
                .map(|i| {
                    let link = &links[i];
                    let bound = self.m.compose(link.bound, &|v| renamed.get(&v).copied());
                    Binding {
                        atom: link.atom,
                        bound,
                        support: link
                            .support
                            .iter()
                            .copied()
                            .filter(|v| !renamed.contains_key(v))
                            .collect(),
                    }
                })
                .collect();
            return self.schedule(facts, src, formula, &rest);
        }
        self.schedule(facts, src, formula, links)
    }

    fn schedule(&mut self, facts: &Facts, src: BlockId, formula: Bdd, links: &[Binding]) -> Bdd {
        let formula_atoms: BTreeSet<u32> = self
            .support(formula)
            .into_iter()
            .filter(|&v| self.scope(facts, v) == Some(src))
            .collect();
        let mut occurrences: HashMap<u32, usize> = HashMap::default();
        for link in links {
            for &v in &link.support {
                *occurrences.entry(v).or_default() += 1;
            }
        }
        let mut pending: Vec<usize> = (0..links.len())
            .filter(|&i| {
                let link = &links[i];
                link.support.is_empty()
                    || link.bound.constant().is_some()
                    || link
                        .support
                        .iter()
                        .any(|v| occurrences[v] > 1 || formula_atoms.contains(v))
            })
            .collect();
        occurrences.clear();
        for &i in &pending {
            for &v in &links[i].support {
                *occurrences.entry(v).or_default() += 1;
            }
        }
        let lone: Vec<u32> = formula_atoms
            .iter()
            .copied()
            .filter(|v| !occurrences.contains_key(v))
            .collect();
        let mut acc = self.m.exists(formula, &|v| lone.contains(&v));
        while !pending.is_empty() {
            let built = self.m.support(acc);
            let (position, _) = pending
                .iter()
                .enumerate()
                .max_by_key(|(_, &i)| {
                    let support = &links[i].support;
                    let shared = support.iter().filter(|v| built.contains(v)).count();
                    (shared, std::cmp::Reverse(support.len()))
                })
                .unwrap();
            let i = pending.swap_remove(position);
            let link = self.m.iff(links[i].atom, links[i].bound);
            let mut finished = Vec::new();
            for &v in &links[i].support {
                let count = occurrences.get_mut(&v).unwrap();
                *count -= 1;
                if *count == 0 {
                    finished.push(v);
                }
            }
            acc = if finished.is_empty() {
                self.m.and(acc, link)
            } else {
                self.m.and_exists(acc, link, &|v| finished.contains(&v))
            };
            if acc == Bdd::FALSE {
                return acc;
            }
        }
        acc
    }
}

struct Binding {
    atom: Bdd,
    bound: Bdd,
    support: Vec<u32>,
}

impl Logic {
    fn bindings(
        &mut self,
        f: &Func,
        facts: &Facts,
        src: BlockId,
        slot: usize,
    ) -> std::rc::Rc<Vec<Binding>> {
        if let Some(b) = self.relations.get(&(src, slot)) {
            return b.clone();
        }
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = &f.blocks[&edge.dst];
        let mut links = Vec::new();
        for (&(param, ty), &arg) in dst.params.iter().zip(&edge.args) {
            let (atom, bound) = match ty {
                Ty::I1 => (Atom::Bit(param), self.bit(f, facts, arg)),
                Ty::I32
                    if facts.viewed[param.0]
                        && !(self.open_words && facts.materialized[param.0]) =>
                {
                    (Atom::View(param), self.view(f, facts, arg))
                }
                _ => continue,
            };
            links.push(self.binding(facts, src, atom, bound));
        }
        let links = std::rc::Rc::new(links);
        self.relations.insert((src, slot), links.clone());
        links
    }

    pub fn reach(
        &mut self,
        f: &Func,
        facts: &Facts,
        start: BlockId,
        formula: Bdd,
    ) -> std::collections::BTreeMap<BlockId, Bdd> {
        let rank: std::collections::BTreeMap<BlockId, usize> = facts
            .order
            .iter()
            .enumerate()
            .map(|(r, &b)| (b, r))
            .collect();
        let mut reach = std::collections::BTreeMap::from([(start, formula)]);
        // What each block has already passed on. An image distributes over a
        // union, so only what a block gained since is passed on again; any
        // function between the gain and the whole serves, and the smallest
        // such is cheapest to pass on.
        let mut sent: std::collections::BTreeMap<BlockId, Bdd> = Default::default();
        let mut worklist: BTreeSet<(usize, BlockId)> = BTreeSet::from([(rank[&start], start)]);
        while let Some((_, x)) = worklist.pop_first() {
            let whole = reach[&x];
            let before = sent.insert(x, whole).unwrap_or(Bdd::FALSE);
            let unsent = self.m.not(before);
            let r = self.m.restrict(whole, unsent);
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

    pub fn post(
        &mut self,
        f: &Func,
        facts: &Facts,
        src: BlockId,
        slot: usize,
        formula: Bdd,
    ) -> Bdd {
        if formula == Bdd::FALSE {
            return formula;
        }
        let links = self.bindings(f, facts, src, slot);
        self.project(facts, src, formula, &links)
    }
}

#[derive(Default)]
struct EdgeIndex {
    by_var: HashMap<u32, Vec<usize>>,
    words: HashMap<ValueId, Vec<usize>>,
}

pub(super) fn lane_test(f: &Func, facts: &Facts, a: ValueId, b: ValueId) -> Option<ValueId> {
    if facts.lane_word[a.0] && facts.constant(f, b) == Some(0) {
        Some(a)
    } else if facts.lane_word[b.0] && facts.constant(f, a) == Some(0) {
        Some(b)
    } else {
        None
    }
}
