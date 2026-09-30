use super::address::compare;
use super::check::interval;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::rdna_spmd::hash::Mix>>;
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Atom {
    Bit(ValueId),
    View(ValueId),
    Lane(u8),
    Fresh(usize, ValueId, u32),
    Term(usize, bool),
    Next(ValueId, bool),
    Marker(Choice),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Choice {
    Query(ValueId),
    Word(ValueId),
    Meet(usize),
}

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

const MARKERS_LAST: u32 = 0xfffe_0000;
pub const PATH: usize = 3;

pub struct Logic {
    pub m: Manager,
    choices: Choices,
    markers_first: bool,
    vars: HashMap<Atom, u32>,
    atoms: HashMap<u32, Atom>,
    params: HashMap<ValueId, (usize, usize)>,
    markers: HashMap<Choice, u32>,
    listed: Vec<Choice>,
    detour: HashMap<Atom, u32>,
    bits: HashMap<ValueId, Bdd>,
    views: HashMap<ValueId, Bdd>,
    lane_values: HashMap<ValueId, Option<[u32; 32]>>,
    orders: HashMap<BlockId, Rc<HashMap<(IntPred, ValueId, ValueId), ValueId>>>,
    thresholds: HashMap<BlockId, Rc<Vec<Threshold>>>,
    supports: HashMap<Bdd, Rc<Vec<u32>>>,
    edges: BTreeMap<(BlockId, usize), Rc<EdgeIndex>>,
    relations: BTreeMap<(BlockId, usize), Rc<Vec<Binding>>>,
    images: HashMap<(usize, usize, Bdd), Bdd>,
    uniform_tests: HashSet<ValueId>,
    comparisons: HashMap<BlockId, Rc<Vec<(usize, ValueId, IntPred, ValueId, ValueId)>>>,
}

impl Logic {
    fn with(
        f: &Func,
        facts: &Facts,
        choices: Choices,
        markers_first: bool,
        listed: &[Choice],
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
            .map(|(i, &c)| (c, i as u32))
            .collect();
        Self {
            m: Manager::new(),
            choices,
            markers_first,
            vars: HashMap::default(),
            atoms: HashMap::default(),
            params,
            markers,
            listed: listed.to_vec(),
            detour: HashMap::default(),
            bits: HashMap::default(),
            views: HashMap::default(),
            lane_values: HashMap::default(),
            orders: HashMap::default(),
            thresholds: HashMap::default(),
            supports: HashMap::default(),
            edges: BTreeMap::new(),
            relations: BTreeMap::new(),
            images: HashMap::default(),
            uniform_tests: HashSet::default(),
            comparisons: HashMap::default(),
        }
    }

    pub fn fixed(f: &Func, facts: &Facts, kept: &BTreeSet<Choice>, tags: &[Choice]) -> Self {
        let choices = Choices::Fixed { kept: kept.clone() };
        Self::with(f, facts, choices, false, tags)
    }

    pub fn open(f: &Func, facts: &Facts, listed: &[Choice]) -> Self {
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
        for &c in listed {
            let local = logic.atom(Atom::Marker(c));
            all_local = logic.m.and(all_local, local);
            if let Choice::Word(v) = c {
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

    pub fn keep(&mut self, kept: &BTreeSet<Choice>) {
        if let Choices::Fixed { kept: fixed } = &mut self.choices {
            *fixed = kept.clone();
        }
    }

    pub fn is_open(&self) -> bool {
        matches!(self.choices, Choices::Open { .. })
    }

    pub fn local(&mut self, c: Choice) -> Bdd {
        if let Choices::Fixed { kept } = &self.choices {
            return Manager::constant(!kept.contains(&c));
        }
        if self.markers.contains_key(&c) {
            self.atom(Atom::Marker(c))
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

    pub fn tag(&mut self, c: Choice) -> Bdd {
        if matches!(self.choices, Choices::Fixed { .. }) && self.markers.contains_key(&c) {
            self.atom(Atom::Marker(c))
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
        for c in self.listed.clone() {
            let var = self.vars[&Atom::Marker(c)];
            let local = self.m.cofactor(safe, var, true);
            if local != Bdd::FALSE {
                safe = local;
                continue;
            }
            safe = self.m.cofactor(safe, var, false);
            kept.insert(c);
        }
        assert_eq!(safe, Bdd::TRUE);
        kept
    }

    pub fn settled(&mut self, f: Bdd, kept: &BTreeSet<Choice>) -> Bdd {
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
            Atom::Marker(c) => {
                let i = self.markers[&c];
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
            Atom::Lane(i) => (2 << 30) | i as u32,
            Atom::Fresh(PATH, _, i) => {
                assert!(i < 1 << 16, "too many paths");
                return (1 << 16) + i;
            }
            Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => {
                let next = self.detour.len() as u32;
                assert!(next < 1 << 29, "too many detour values");
                (3 << 30) | *self.detour.entry(atom).or_insert(next)
            }
        };
        var + (1 << 17)
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
            Atom::Bit(v) => facts.uniform[v.0] || self.uniform_tests.contains(&v),
            Atom::View(v) => facts.saturated[v.0],
            Atom::Marker(_) => true,
            Atom::Lane(_) | Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => false,
        }
    }

    pub fn scope(&self, facts: &Facts, var: u32) -> Option<BlockId> {
        match self.atoms[&var] {
            Atom::Bit(v) | Atom::View(v) => match facts.site[v.0] {
                Site::Param { block, .. } | Site::Inst { block, .. } => Some(block),
                Site::Unreached => None,
            },
            Atom::Lane(_) | Atom::Marker(_) | Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => None,
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
                let local = self.local(Choice::Query(outputs[0].0));
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
                _ if self.lane_values(f, facts, v, 0).is_some() => {
                    let values = self.lane_values(f, facts, v, 0).unwrap();
                    self.lanes(|l| values[l as usize] & 1 == 1)
                }
                Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b) => {
                    let leaf = match (equality(f, facts, p, a, b), facts.site[v.0]) {
                        (Some(e), Site::Inst { block, index }) => {
                            let equal = self.equal(f, facts, block, index, v, e);
                            if e.flip {
                                self.m.not(equal)
                            } else {
                                equal
                            }
                        }
                        (None, Site::Inst { .. })
                            if f.types[a.0] == Ty::I32 && a != b && facts.constant(f, a).is_none() && facts.constant(f, b).is_none() =>
                        {
                            if !facts.uniform[v.0] && uniform_difference(f, facts, a, b) {
                                self.uniform_tests.insert(v);
                            }
                            let equal = self.value_equality(f, facts, v, a, b, p == IntPred::Ne);
                            if p == IntPred::Ne {
                                self.m.not(equal)
                            } else {
                                equal
                            }
                        }
                        _ => self.atom(opaque),
                    };
                    self.compare(f, facts, p, a, b, leaf, &mut HashMap::default())
                }
                Op::FCmp(p, a, b) => {
                    let leaf = self.float_within(f, facts, v, p, a, b);
                    self.float_order(f, facts, p, a, b, leaf, &mut HashMap::default())
                }
                Op::Cmp(p, a, b) => {
                    let uniform = !facts.uniform[v.0] && uniform_order(f, facts, p, a, b);
                    if uniform {
                        self.uniform_tests.insert(v);
                    }
                    let leaf = match (threshold(f, facts, p, a, b), facts.site[v.0]) {
                        (Some(t), Site::Inst { block, index }) => {
                            let below = self.below(f, facts, block, index, v, t);
                            if t.flip {
                                self.m.not(below)
                            } else {
                                below
                            }
                        }
                        _ => {
                            let (q, x, y, negated) = ordered(p, a, b);
                            let first = match facts.site[v.0] {
                                Site::Inst { block, .. } => self.first_order(f, block)[&(q, x, y)],
                                _ => v,
                            };
                            if uniform {
                                self.uniform_tests.insert(first);
                            }
                            let relation = self.relation(f, facts, first, (q, x, y));
                            if negated {
                                self.m.not(relation)
                            } else {
                                relation
                            }
                        }
                    };
                    self.order(f, facts, p, a, b, leaf, &mut HashMap::default())
                }
                _ => self.atom(opaque),
            },
            _ => self.atom(opaque),
        }
    }

    fn below(&mut self, f: &Func, facts: &Facts, block: BlockId, index: usize, v: ValueId, t: Threshold) -> Bdd {
        self.within(f, facts, block, index, v, t)
    }

    fn equal(&mut self, f: &Func, facts: &Facts, block: BlockId, index: usize, v: ValueId, e: Threshold) -> Bdd {
        self.within(f, facts, block, index, v, e)
    }

    fn within(&mut self, f: &Func, facts: &Facts, block: BlockId, index: usize, v: ValueId, t: Threshold) -> Bdd {
        let target = t.truth();
        if target.is_empty() {
            return Bdd::FALSE;
        }
        if target == [(0, u32::MAX as u64)] {
            return Bdd::TRUE;
        }
        let list = self.block_thresholds(f, facts, block);
        let earlier: Vec<Threshold> = list.iter().filter(|e| e.index < index && e.value == t.value).copied().collect();
        let mut sets: Vec<(Bdd, Vec<(u64, u64)>)> = Vec::new();
        for e in &earlier {
            let bit = self.bit(f, facts, e.of);
            let holds = if e.flip { self.m.not(bit) } else { bit };
            let set = e.truth();
            if set == target {
                return holds;
            }
            sets.push((holds, set));
        }
        let atom = self.atom(Atom::Bit(v));
        let free = if t.flip { self.m.not(atom) } else { atom };
        self.cells(&sets, &target, free, u32::MAX as u64)
    }

    fn float_within(&mut self, f: &Func, facts: &Facts, v: ValueId, p: FloatPred, a: ValueId, b: ValueId) -> Bdd {
        let atom = self.atom(Atom::Bit(v));
        let (Some((x, target)), Site::Inst { block, index }) = (float_test(f, facts, p, a, b), facts.site[v.0]) else {
            return atom;
        };
        if target.is_empty() {
            return Bdd::FALSE;
        }
        if target == [(0, FLOAT_TOP), (FLOAT_NAN, FLOAT_NAN)] {
            return Bdd::TRUE;
        }
        let mut sets = Vec::new();
        for inst in &f.blocks[&block].insts[..index] {
            let Inst::Core { value, op: Op::FCmp(q, c, d), .. } = inst else {
                continue;
            };
            let Some((y, set)) = float_test(f, facts, *q, *c, *d) else {
                continue;
            };
            if y != x {
                continue;
            }
            let holds = self.bit(f, facts, *value);
            if set == target {
                return holds;
            }
            sets.push((holds, set));
        }
        if sets.is_empty() {
            return atom;
        }
        self.cells(&sets, &target, atom, FLOAT_NAN)
    }

    fn cells(&mut self, sets: &[(Bdd, Vec<(u64, u64)>)], target: &[(u64, u64)], free: Bdd, top: u64) -> Bdd {
        let mut points: Vec<u64> = vec![0, top + 1];
        for (_, set) in sets {
            for &(low, high) in set {
                points.push(low);
                points.push(high + 1);
            }
        }
        points.sort_unstable();
        points.dedup();
        let member = |set: &[(u64, u64)], x: u64| set.iter().any(|&(low, high)| low <= x && x <= high);
        let mut regions: Vec<(Vec<bool>, bool, bool)> = Vec::new();
        for w in points.windows(2) {
            let (low, high) = (w[0], w[1] - 1);
            let covered: u64 = target
                .iter()
                .map(|&(a, b)| {
                    let (a, b) = (a.max(low), b.min(high));
                    if a <= b {
                        b - a + 1
                    } else {
                        0
                    }
                })
                .sum();
            let (some, all) = (covered > 0, covered == high - low + 1);
            let key: Vec<bool> = sets.iter().map(|(_, set)| member(set, low)).collect();
            match regions.iter_mut().find(|(k, _, _)| *k == key) {
                Some(region) => {
                    region.1 |= some;
                    region.2 &= all;
                }
                None => regions.push((key, some, all)),
            }
        }
        let mut result = Bdd::FALSE;
        for (key, some, all) in regions {
            if !some {
                continue;
            }
            let mut cell = if all { Bdd::TRUE } else { free };
            for ((holds, _), inside) in sets.iter().zip(key) {
                let literal = if inside { *holds } else { self.m.not(*holds) };
                cell = self.m.and(cell, literal);
                if cell == Bdd::FALSE {
                    break;
                }
            }
            result = self.m.or(result, cell);
        }
        result
    }

    fn block_thresholds(&mut self, f: &Func, facts: &Facts, block: BlockId) -> Rc<Vec<Threshold>> {
        if let Some(list) = self.thresholds.get(&block) {
            return list.clone();
        }
        let mut list = Vec::new();
        for (index, inst) in f.blocks[&block].insts.iter().enumerate() {
            if let Inst::Core {
                value,
                op: Op::Cmp(p, a, b),
                ..
            } = inst
            {
                if let Some(t) = threshold(f, facts, *p, *a, *b).or_else(|| equality(f, facts, *p, *a, *b)) {
                    list.push(Threshold { index, of: *value, ..t });
                }
            }
        }
        let list = Rc::new(list);
        self.thresholds.insert(block, list.clone());
        list
    }

    fn relation(&mut self, f: &Func, facts: &Facts, first: ValueId, (q, x, y): (IntPred, ValueId, ValueId)) -> Bdd {
        let atom = self.atom(Atom::Bit(first));
        let Some(Op::Cmp(p, a, b)) = facts.op(f, first) else {
            unreachable!("the first order of a block is a comparison")
        };
        let holds = if ordered(p, a, b).3 { self.m.not(atom) } else { atom };
        let Site::Inst { block, index } = facts.site[first.0] else {
            return holds;
        };
        let edges = self.order_edges(f, facts, block, index, Some(q));
        if edges.is_empty() {
            return holds;
        }
        let (back, _) = self.paths(&edges, y, x);
        let (_, forward) = self.paths(&edges, x, y);
        let open = self.m.not(back);
        let allowed = self.m.and(holds, open);
        self.m.or(allowed, forward)
    }

    fn value_equality(&mut self, f: &Func, facts: &Facts, v: ValueId, x: ValueId, y: ValueId, flip: bool) -> Bdd {
        let atom = self.atom(Atom::Bit(v));
        let holds = if flip { self.m.not(atom) } else { atom };
        let Site::Inst { block, index } = facts.site[v.0] else {
            return holds;
        };
        let mut must = Bdd::FALSE;
        let mut apart = Bdd::FALSE;
        for q in [IntPred::Ult, IntPred::Slt] {
            let edges = self.order_edges(f, facts, block, index, Some(q));
            if edges.is_empty() {
                continue;
            }
            let (up, above) = self.paths(&edges, x, y);
            let (down, below) = self.paths(&edges, y, x);
            let both = self.m.and(up, down);
            must = self.m.or(must, both);
            let strict = self.m.or(above, below);
            apart = self.m.or(apart, strict);
        }
        let edges = self.order_edges(f, facts, block, index, None);
        if !edges.is_empty() {
            let (joined, _) = self.paths(&edges, x, y);
            must = self.m.or(must, joined);
            let unequal: Vec<(ValueId, ValueId, Bdd)> = edges.iter().step_by(2).map(|&(a, b, _, equal)| (a, b, self.m.not(equal))).collect();
            for (a, b, differ) in unequal {
                let (xa, _) = self.paths(&edges, x, a);
                let (by, _) = self.paths(&edges, b, y);
                let (xb, _) = self.paths(&edges, x, b);
                let (ay, _) = self.paths(&edges, a, y);
                let direct = self.m.and(xa, by);
                let crossed = self.m.and(xb, ay);
                let linked = self.m.or(direct, crossed);
                let split = self.m.and(differ, linked);
                apart = self.m.or(apart, split);
            }
        }
        let open = self.m.not(apart);
        let allowed = self.m.and(holds, open);
        self.m.or(allowed, must)
    }

    fn block_comparisons(&mut self, f: &Func, facts: &Facts, block: BlockId) -> Rc<Vec<(usize, ValueId, IntPred, ValueId, ValueId)>> {
        if let Some(list) = self.comparisons.get(&block) {
            return list.clone();
        }
        let list: Vec<(usize, ValueId, IntPred, ValueId, ValueId)> = f.blocks[&block]
            .insts
            .iter()
            .enumerate()
            .filter_map(|(i, inst)| match inst {
                Inst::Core {
                    value,
                    op: Op::Cmp(p, a, b),
                    ..
                } if f.types[a.0] == Ty::I32 && facts.constant(f, *a).is_none() && facts.constant(f, *b).is_none() && a != b => {
                    Some((i, *value, *p, *a, *b))
                }
                _ => None,
            })
            .collect();
        let list = Rc::new(list);
        self.comparisons.insert(block, list.clone());
        list
    }

    fn order_edges(&mut self, f: &Func, facts: &Facts, block: BlockId, index: usize, family: Option<IntPred>) -> Vec<(ValueId, ValueId, bool, Bdd)> {
        let list = self.block_comparisons(f, facts, block);
        let mut edges = Vec::new();
        for &(i, c, p, a, b) in list.iter() {
            if i >= index {
                break;
            }
            let bit = self.bit(f, facts, c);
            match p {
                IntPred::Eq | IntPred::Ne => {
                    let equal = if p == IntPred::Ne { self.m.not(bit) } else { bit };
                    edges.push((a, b, false, equal));
                    edges.push((b, a, false, equal));
                }
                _ => {
                    let (q, u, w, negated) = ordered(p, a, b);
                    if family.is_some_and(|family| family != q) || family.is_none() {
                        continue;
                    }
                    let less = if negated { self.m.not(bit) } else { bit };
                    let not_less = self.m.not(less);
                    edges.push((u, w, true, less));
                    edges.push((w, u, false, not_less));
                }
            }
        }
        edges
    }

    fn paths(&mut self, edges: &[(ValueId, ValueId, bool, Bdd)], from: ValueId, to: ValueId) -> (Bdd, Bdd) {
        let mut any: HashMap<ValueId, Bdd> = HashMap::default();
        let mut strict: HashMap<ValueId, Bdd> = HashMap::default();
        any.insert(from, Bdd::TRUE);
        loop {
            let mut changed = false;
            for &(a, b, is_strict, guard) in edges {
                let reach = any.get(&a).copied().unwrap_or(Bdd::FALSE);
                let sharp = strict.get(&a).copied().unwrap_or(Bdd::FALSE);
                if reach == Bdd::FALSE && sharp == Bdd::FALSE {
                    continue;
                }
                let step = self.m.and(reach, guard);
                let old = any.get(&b).copied().unwrap_or(Bdd::FALSE);
                let new = self.m.or(old, step);
                if new != old {
                    any.insert(b, new);
                    changed = true;
                }
                let through = if is_strict { step } else { self.m.and(sharp, guard) };
                let old = strict.get(&b).copied().unwrap_or(Bdd::FALSE);
                let new = self.m.or(old, through);
                if new != old {
                    strict.insert(b, new);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let reach = if from == to { Bdd::TRUE } else { any.get(&to).copied().unwrap_or(Bdd::FALSE) };
        (reach, strict.get(&to).copied().unwrap_or(Bdd::FALSE))
    }

    fn float_order(
        &mut self,
        f: &Func,
        facts: &Facts,
        pred: FloatPred,
        a: ValueId,
        b: ValueId,
        leaf: Bdd,
        memo: &mut HashMap<(ValueId, ValueId), Bdd>,
    ) -> Bdd {
        if let Some(&g) = memo.get(&(a, b)) {
            return g;
        }
        let float = |v: ValueId| -> Option<f64> {
            let k = facts.constant(f, v)?;
            match f.types[v.0] {
                Ty::F32 => Some(f32::from_bits(k as u32) as f64),
                Ty::F64 => Some(f64::from_bits(k)),
                _ => None,
            }
        };
        let g = if let (Some(x), Some(y)) = (float(a), float(b)) {
            Manager::constant(float_compare(pred, x, y))
        } else if let Some(Op::Select(c, yes, no)) = facts.op(f, a) {
            let c = self.bit(f, facts, c);
            let yes = self.float_order(f, facts, pred, yes, b, leaf, memo);
            let no = self.float_order(f, facts, pred, no, b, leaf, memo);
            self.m.ite(c, yes, no)
        } else if let Some(Op::Select(c, yes, no)) = facts.op(f, b) {
            let c = self.bit(f, facts, c);
            let yes = self.float_order(f, facts, pred, a, yes, leaf, memo);
            let no = self.float_order(f, facts, pred, a, no, leaf, memo);
            self.m.ite(c, yes, no)
        } else {
            leaf
        };
        memo.insert((a, b), g);
        g
    }

    fn order(
        &mut self,
        f: &Func,
        facts: &Facts,
        pred: IntPred,
        a: ValueId,
        b: ValueId,
        leaf: Bdd,
        memo: &mut HashMap<(ValueId, ValueId), Bdd>,
    ) -> Bdd {
        if let Some(&g) = memo.get(&(a, b)) {
            return g;
        }
        let g = if a == b {
            Manager::constant(compare(pred, 0, 0))
        } else if let (Some(x), Some(y), Ty::I32) = (facts.constant(f, a), facts.constant(f, b), f.types[a.0]) {
            Manager::constant(compare(pred, x as u32, y as u32))
        } else if let Some(Op::Select(c, yes, no)) = facts.op(f, a) {
            let c = self.bit(f, facts, c);
            let yes = self.order(f, facts, pred, yes, b, leaf, memo);
            let no = self.order(f, facts, pred, no, b, leaf, memo);
            self.m.ite(c, yes, no)
        } else if let Some(Op::Select(c, yes, no)) = facts.op(f, b) {
            let c = self.bit(f, facts, c);
            let yes = self.order(f, facts, pred, a, yes, leaf, memo);
            let no = self.order(f, facts, pred, a, no, leaf, memo);
            self.m.ite(c, yes, no)
        } else {
            leaf
        };
        memo.insert((a, b), g);
        g
    }

    fn first_order(&mut self, f: &Func, block: BlockId) -> Rc<HashMap<(IntPred, ValueId, ValueId), ValueId>> {
        if let Some(index) = self.orders.get(&block) {
            return index.clone();
        }
        let mut index = HashMap::default();
        for inst in &f.blocks[&block].insts {
            if let Inst::Core {
                value,
                op: Op::Cmp(p, a, b),
                ..
            } = inst
            {
                if !matches!(p, IntPred::Eq | IntPred::Ne) {
                    let (q, x, y, _) = ordered(*p, *a, *b);
                    index.entry((q, x, y)).or_insert(*value);
                }
            }
        }
        let index = Rc::new(index);
        self.orders.insert(block, index.clone());
        index
    }

    fn compare(
        &mut self,
        f: &Func,
        facts: &Facts,
        pred: IntPred,
        a: ValueId,
        b: ValueId,
        leaf: Bdd,
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
            let yes = self.compare(f, facts, pred, yes, b, leaf, memo);
            let no = self.compare(f, facts, pred, no, b, leaf, memo);
            self.m.ite(c, yes, no)
        } else if let Some(Op::Select(c, yes, no)) = facts.op(f, b) {
            let c = self.bit(f, facts, c);
            let yes = self.compare(f, facts, pred, a, yes, leaf, memo);
            let no = self.compare(f, facts, pred, a, no, leaf, memo);
            self.m.ite(c, yes, no)
        } else {
            leaf
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
                _ if self.lane_values(f, facts, w, 0).is_some() => {
                    let values = self.lane_values(f, facts, w, 0).unwrap();
                    self.lanes(|l| values[l as usize] >> l & 1 == 1)
                }
                Op::Const(_, k) => self.word(k as u32),
                Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b) => {
                    if constant_choices(f, facts, a).is_some() && constant_choices(f, facts, b).is_some() {
                        return self.constant_word(f, facts, k, a, b);
                    }
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

    fn constant_word(&mut self, f: &Func, facts: &Facts, k: IntOp, a: ValueId, b: ValueId) -> Bdd {
        for (x, other) in [(a, b), (b, a)] {
            if let Some(Op::Select(c, p, q)) = facts.op(f, x) {
                let c = self.bit(f, facts, c);
                let p = self.constant_word(f, facts, k, p, other);
                let q = self.constant_word(f, facts, k, q, other);
                return self.m.ite(c, p, q);
            }
        }
        let (x, y) = (facts.constant(f, a).unwrap(), facts.constant(f, b).unwrap());
        let word = match k {
            IntOp::And => x & y,
            IntOp::Or => x | y,
            _ => x ^ y,
        } as u32;
        self.word(word)
    }

    pub fn word(&mut self, k: u32) -> Bdd {
        self.lanes(|l| k >> l & 1 == 1)
    }

    pub fn lane_bits(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Option<[(u32, u32); 32]> {
        if f.types[v.0] != Ty::I32 {
            return None;
        }
        Some(self.known_bits(f, facts, v, 0))
    }

    fn known_bits(&mut self, f: &Func, facts: &Facts, v: ValueId, depth: u32) -> [(u32, u32); 32] {
        if let Some(values) = self.lane_values(f, facts, v, 0) {
            return std::array::from_fn(|l| (u32::MAX, values[l]));
        }
        if depth > 16 || f.types[v.0] != Ty::I32 {
            return [(0, 0); 32];
        }
        let constant = |x: ValueId| facts.constant(f, x).map(|k| k as u32);
        match facts.op(f, v) {
            Some(Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b)) => {
                let (x, y) = (self.known_bits(f, facts, a, depth + 1), self.known_bits(f, facts, b, depth + 1));
                std::array::from_fn(|l| {
                    let ((mx, vx), (my, vy)) = (x[l], y[l]);
                    let mask = match k {
                        IntOp::And => (mx & my) | (mx & !vx) | (my & !vy),
                        IntOp::Or => (mx & my) | (mx & vx) | (my & vy),
                        _ => mx & my,
                    };
                    let value = match k {
                        IntOp::And => vx & vy,
                        IntOp::Or => vx | vy,
                        _ => vx ^ vy,
                    };
                    (mask, value & mask)
                })
            }
            Some(Op::Int(k @ (IntOp::Add | IntOp::Sub), a, b)) => {
                let (x, y) = (self.known_bits(f, facts, a, depth + 1), self.known_bits(f, facts, b, depth + 1));
                std::array::from_fn(|l| {
                    let ((mx, vx), (my, vy)) = (x[l], y[l]);
                    let (zero_x, one_x) = (mx & !vx, vx);
                    let (zero_y, one_y, carry) = match k {
                        IntOp::Add => (my & !vy, vy, 0u32),
                        _ => (vy, my & !vy, 1u32),
                    };
                    let sum_zero = (!zero_x).wrapping_add(!zero_y).wrapping_add(carry);
                    let sum_one = one_x.wrapping_add(one_y).wrapping_add(carry);
                    let carry_zero = !(sum_zero ^ zero_x ^ zero_y);
                    let carry_one = sum_one ^ one_x ^ one_y;
                    let known = (zero_x | one_x) & (zero_y | one_y) & (carry_zero | carry_one);
                    (known, sum_one & known)
                })
            }
            Some(Op::Int(IntOp::Shl, a, s)) if constant(s).is_some() => {
                let k = constant(s).unwrap() & 31;
                let x = self.known_bits(f, facts, a, depth + 1);
                std::array::from_fn(|l| ((x[l].0 << k) | ((1u32 << k) - 1), x[l].1 << k))
            }
            Some(Op::Int(IntOp::Mul, a, s)) if constant(s).is_some_and(|k| k.is_power_of_two()) => {
                let k = constant(s).unwrap().trailing_zeros();
                let x = self.known_bits(f, facts, a, depth + 1);
                std::array::from_fn(|l| ((x[l].0 << k) | ((1u32 << k) - 1), x[l].1 << k))
            }
            Some(Op::Int(IntOp::LShr, a, s)) if constant(s).is_some() => {
                let k = constant(s).unwrap() & 31;
                let x = self.known_bits(f, facts, a, depth + 1);
                std::array::from_fn(|l| ((x[l].0 >> k) | !(u32::MAX >> k), x[l].1 >> k))
            }
            _ => [(0, 0); 32],
        }
    }

    pub fn lane_function(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Option<[u32; 32]> {
        self.lane_values(f, facts, v, 0)
    }

    fn lane_values(&mut self, f: &Func, facts: &Facts, v: ValueId, depth: u32) -> Option<[u32; 32]> {
        if let Some(&r) = self.lane_values.get(&v) {
            return r;
        }
        if depth > 64 {
            return None;
        }
        let r = self.compute_lane_values(f, facts, v, depth);
        self.lane_values.insert(v, r);
        r
    }

    fn compute_lane_values(&mut self, f: &Func, facts: &Facts, v: ValueId, depth: u32) -> Option<[u32; 32]> {
        let bits = match f.types[v.0] {
            Ty::I1 => 1,
            Ty::I32 => u32::MAX,
            _ => return None,
        };
        let mut out = [0u32; 32];
        match facts.op(f, v)? {
            Op::Const(_, k) => out = [k as u32; 32],
            Op::Env(Env::LaneId) => out = std::array::from_fn(|l| l as u32),
            Op::Int(k, a, b) => {
                let a = self.lane_values(f, facts, a, depth + 1)?;
                let b = self.lane_values(f, facts, b, depth + 1)?;
                for l in 0..32 {
                    let (x, y) = (a[l], b[l]);
                    out[l] = match k {
                        IntOp::Add => x.wrapping_add(y),
                        IntOp::Sub => x.wrapping_sub(y),
                        IntOp::Mul => x.wrapping_mul(y),
                        IntOp::And => x & y,
                        IntOp::Or => x | y,
                        IntOp::Xor => x ^ y,
                        IntOp::Shl | IntOp::LShr | IntOp::AShr if y >= 32 => return None,
                        IntOp::Shl => x << y,
                        IntOp::LShr => x >> y,
                        IntOp::AShr => ((x as i32) >> y) as u32,
                    };
                }
            }
            Op::Cmp(p, a, b) if f.types[a.0] == Ty::I32 => {
                let a = self.lane_values(f, facts, a, depth + 1)?;
                let b = self.lane_values(f, facts, b, depth + 1)?;
                out = std::array::from_fn(|l| compare(p, a[l], b[l]) as u32);
            }
            Op::Select(c, a, b) => {
                let c = self.lane_values(f, facts, c, depth + 1)?;
                let a = self.lane_values(f, facts, a, depth + 1)?;
                let b = self.lane_values(f, facts, b, depth + 1)?;
                out = std::array::from_fn(|l| if c[l] & 1 == 1 { a[l] } else { b[l] });
            }
            _ => return None,
        }
        Some(out.map(|x| x & bits))
    }

    pub fn lane_dependent(&mut self, f: Bdd) -> bool {
        let support = self.support(f);
        support.iter().any(|v| matches!(self.atoms.get(v), Some(Atom::Lane(_))))
    }

    pub fn at_lane(&mut self, f: Bdd, lane: u32) -> Bdd {
        let mut f = f;
        for i in 0..5u8 {
            if let Some(&var) = self.vars.get(&Atom::Lane(i)) {
                f = self.m.cofactor(f, var, lane >> i & 1 == 1);
            }
        }
        f
    }

    pub fn lanes(&mut self, holds: impl Fn(u32) -> bool) -> Bdd {
        let bits: Vec<Bdd> = (0..5).map(|i| self.atom(Atom::Lane(i))).collect();
        let mut any = Bdd::FALSE;
        for l in (0..32u32).filter(|&l| holds(l)) {
            let mut one = Bdd::TRUE;
            for (i, &bit) in bits.iter().enumerate() {
                let literal = if l >> i & 1 == 1 { bit } else { self.m.not(bit) };
                one = self.m.and(one, literal);
            }
            any = self.m.or(any, one);
        }
        any
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
                let link = self.binding(facts, src, arriving(src, edge.dst, atom), bound);
                pending.extend(link.support.iter().copied().filter(|v| !seen.contains(v)));
                links.push(link);
            }
        }
        for (atom, bound) in self.passed_tests(f, facts, src, slot) {
            let link = self.binding(facts, src, atom, bound);
            if link.support.iter().any(|v| seen.contains(v)) {
                links.push(link);
            }
        }
        let r = self.project(facts, src, formula, &links);
        self.arrived(src, edge.dst, r)
    }

    fn arrived(&mut self, src: BlockId, dst: BlockId, r: Bdd) -> Bdd {
        if src != dst || r.constant().is_some() {
            return r;
        }
        let next: Vec<(u32, Atom)> = self
            .support(r)
            .iter()
            .filter_map(|&var| match self.atoms[&var] {
                Atom::Next(v, false) => Some((var, Atom::Bit(v))),
                Atom::Next(v, true) => Some((var, Atom::View(v))),
                _ => None,
            })
            .collect();
        if next.is_empty() {
            return r;
        }
        let renamed: HashMap<u32, Bdd> = next.into_iter().map(|(var, atom)| (var, self.atom(atom))).collect();
        self.m.compose(r, &|v| renamed.get(&v).copied())
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
        let foreign: Vec<bool> = links
            .iter()
            .map(|link| self.support(link.bound).iter().any(|v| !link.support.contains(v)))
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
            links.push(self.binding(facts, src, arriving(src, edge.dst, atom), bound));
        }
        for (atom, bound) in self.passed_tests(f, facts, src, slot) {
            links.push(self.binding(facts, src, atom, bound));
        }
        let links = Rc::new(links);
        self.relations.insert((src, slot), links.clone());
        links
    }

    fn passed_tests(&mut self, f: &Func, facts: &Facts, src: BlockId, slot: usize) -> Vec<(Atom, Bdd)> {
        let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
        let dst = edge.dst;
        let mut out = Vec::new();
        let targets = self.block_thresholds(f, facts, dst);
        if !targets.is_empty() {
            let sources = self.block_thresholds(f, facts, src);
            let mut seen: Vec<(ValueId, bool, bool, i64)> = Vec::new();
            for t in targets.iter() {
                let Site::Param { block, index } = facts.site[t.value.0] else {
                    continue;
                };
                let test = (t.value, t.equal, t.signed, t.at);
                if block != dst || !self.carried(t.value) || seen.contains(&test) {
                    continue;
                }
                seen.push(test);
                let arg = f.blocks[&src].term.edges().nth(slot).unwrap().args[index];
                let relevant: Vec<Threshold> = sources.iter().filter(|e| e.value == arg).copied().collect();
                if relevant.is_empty() {
                    continue;
                }
                let mut sets = Vec::new();
                for e in &relevant {
                    let bit = self.bit(f, facts, e.of);
                    let holds = if e.flip { self.m.not(bit) } else { bit };
                    sets.push((holds, e.truth()));
                }
                let atom = arriving(src, dst, Atom::Bit(t.of));
                let own = self.atom(atom);
                let free = if t.flip { self.m.not(own) } else { own };
                let truth = self.cells(&sets, &t.truth(), free, u32::MAX as u64);
                let bound = if t.flip { self.m.not(truth) } else { truth };
                out.push((atom, bound));
            }
        }
        let args: Vec<ValueId> = f.blocks[&src].term.edges().nth(slot).unwrap().args.clone();
        let orders = self.first_order(f, dst);
        if !orders.is_empty() {
            let at = |p: ValueId| match facts.site[p.0] {
                Site::Param { block, index } if block == dst => Some(index),
                _ => None,
            };
            let sources = self.first_order(f, src);
            let mut pairs: Vec<((IntPred, ValueId, ValueId), ValueId)> = orders.iter().map(|(&k, &v)| (k, v)).collect();
            pairs.sort_by_key(|&(_, v)| v.0);
            for ((q, x, y), t) in pairs {
                let (Some(i), Some(j)) = (at(x), at(y)) else {
                    continue;
                };
                if !self.carried(x) || !self.carried(y) {
                    continue;
                }
                let Some(&e) = sources.get(&(q, args[i], args[j])) else {
                    continue;
                };
                let Some(Op::Cmp(pe, ae, be)) = facts.op(f, e) else {
                    continue;
                };
                let Some(Op::Cmp(pt, at_, bt)) = facts.op(f, t) else {
                    continue;
                };
                let bit = self.bit(f, facts, e);
                let canonical = if ordered(pe, ae, be).3 { self.m.not(bit) } else { bit };
                let bound = if ordered(pt, at_, bt).3 { self.m.not(canonical) } else { canonical };
                out.push((arriving(src, dst, Atom::Bit(t)), bound));
            }
        }
        out
    }

    fn post(&mut self, f: &Func, facts: &Facts, src: BlockId, slot: usize, formula: Bdd) -> Bdd {
        if formula == Bdd::FALSE {
            return formula;
        }
        let links = self.bindings(f, facts, src, slot);
        let r = self.project(facts, src, formula, &links);
        let dst = f.blocks[&src].term.edges().nth(slot).unwrap().dst;
        self.arrived(src, dst, r)
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

fn arriving(src: BlockId, dst: BlockId, atom: Atom) -> Atom {
    match atom {
        Atom::Bit(v) if src == dst => Atom::Next(v, false),
        Atom::View(v) if src == dst => Atom::Next(v, true),
        other => other,
    }
}

pub(super) fn float_compare(pred: FloatPred, x: f64, y: f64) -> bool {
    let unordered = x.is_nan() || y.is_nan();
    match pred {
        FloatPred::Oeq => !unordered && x == y,
        FloatPred::Ogt => !unordered && x > y,
        FloatPred::Oge => !unordered && x >= y,
        FloatPred::Olt => !unordered && x < y,
        FloatPred::Ole => !unordered && x <= y,
        FloatPred::One => !unordered && x != y,
        FloatPred::Ord => !unordered,
        FloatPred::Uno => unordered,
        FloatPred::Ueq => unordered || x == y,
        FloatPred::Ugt => unordered || x > y,
        FloatPred::Uge => unordered || x >= y,
        FloatPred::Ult => unordered || x < y,
        FloatPred::Ule => unordered || x <= y,
        FloatPred::Une => unordered || x != y,
    }
}

#[derive(Clone, Copy)]
struct Threshold {
    value: ValueId,
    signed: bool,
    at: i64,
    flip: bool,
    index: usize,
    of: ValueId,
    equal: bool,
}

impl Threshold {
    fn truth(&self) -> Vec<(u64, u64)> {
        let top = u32::MAX as u64;
        if self.equal {
            let k = self.at as u32 as u64;
            return vec![(k, k)];
        }
        if !self.signed {
            return if self.at <= 0 {
                Vec::new()
            } else if self.at > top as i64 {
                vec![(0, top)]
            } else {
                vec![(0, self.at as u64 - 1)]
            };
        }
        let half = 1u64 << 31;
        if self.at <= i32::MIN as i64 {
            Vec::new()
        } else if self.at > i32::MAX as i64 {
            vec![(0, top)]
        } else if self.at <= 0 {
            vec![(half, (self.at - 1) as i32 as u32 as u64)]
        } else {
            vec![(0, self.at as u64 - 1), (half, top)]
        }
    }

}

const FLOAT_OFFSET: u64 = 0x7f80_0000;
const FLOAT_TOP: u64 = 2 * FLOAT_OFFSET;
const FLOAT_NAN: u64 = FLOAT_TOP + 1;

fn float_key(bits: u32) -> Option<u64> {
    if f32::from_bits(bits).is_nan() {
        return None;
    }
    let magnitude = (bits & 0x7fff_ffff) as u64;
    Some(if bits >> 31 == 1 { FLOAT_OFFSET - magnitude } else { FLOAT_OFFSET + magnitude })
}

fn float_test(f: &Func, facts: &Facts, p: FloatPred, a: ValueId, b: ValueId) -> Option<(ValueId, Vec<(u64, u64)>)> {
    if f.types[a.0] != Ty::F32 {
        return None;
    }
    let (x, k, p) = match (facts.constant(f, a), facts.constant(f, b)) {
        (None, Some(k)) => (a, k, p),
        (Some(k), None) => {
            let swapped = match p {
                FloatPred::Olt => FloatPred::Ogt,
                FloatPred::Ogt => FloatPred::Olt,
                FloatPred::Ole => FloatPred::Oge,
                FloatPred::Oge => FloatPred::Ole,
                FloatPred::Ult => FloatPred::Ugt,
                FloatPred::Ugt => FloatPred::Ult,
                FloatPred::Ule => FloatPred::Uge,
                FloatPred::Uge => FloatPred::Ule,
                other => other,
            };
            (b, k, swapped)
        }
        _ => return None,
    };
    let unordered = matches!(p, FloatPred::Uno | FloatPred::Ueq | FloatPred::Ugt | FloatPred::Uge | FloatPred::Ult | FloatPred::Ule | FloatPred::Une);
    let mut set = match float_key(k as u32) {
        None if unordered => vec![(0, FLOAT_TOP)],
        None => Vec::new(),
        Some(t) => {
            let below = if t == 0 { Vec::new() } else { vec![(0, t - 1)] };
            let above = if t == FLOAT_TOP { Vec::new() } else { vec![(t + 1, FLOAT_TOP)] };
            match p {
                FloatPred::Olt | FloatPred::Ult => below,
                FloatPred::Ole | FloatPred::Ule => vec![(0, t)],
                FloatPred::Ogt | FloatPred::Ugt => above,
                FloatPred::Oge | FloatPred::Uge => vec![(t, FLOAT_TOP)],
                FloatPred::Oeq | FloatPred::Ueq => vec![(t, t)],
                FloatPred::One | FloatPred::Une => below.into_iter().chain(above).collect(),
                FloatPred::Ord => vec![(0, FLOAT_TOP)],
                FloatPred::Uno => Vec::new(),
            }
        }
    };
    if unordered {
        set.push((FLOAT_NAN, FLOAT_NAN));
    }
    Some((x, set))
}

fn uniform_order(f: &Func, facts: &Facts, p: IntPred, a: ValueId, b: ValueId) -> bool {
    if f.types[a.0] != Ty::I32 || a == b || facts.constant(f, a).is_some() || facts.constant(f, b).is_some() || !uniform_difference(f, facts, a, b) {
        return false;
    }
    let signed = matches!(p, IntPred::Slt | IntPred::Sle | IntPred::Sgt | IntPred::Sge);
    match (interval(f, facts, a, 0), interval(f, facts, b, 0)) {
        (Some(x), Some(y)) => !signed || (x.1 < 1 << 31 && y.1 < 1 << 31),
        _ => false,
    }
}

fn uniform_difference(f: &Func, facts: &Facts, x: ValueId, y: ValueId) -> bool {
    let mut terms: HashMap<ValueId, u32> = HashMap::default();
    linear_terms(f, facts, x, 1, &mut terms, 0);
    linear_terms(f, facts, y, u32::MAX, &mut terms, 0);
    terms.iter().all(|(&leaf, &c)| c == 0 || facts.uniform[leaf.0])
}

fn linear_terms(f: &Func, facts: &Facts, v: ValueId, scale: u32, terms: &mut HashMap<ValueId, u32>, depth: u32) {
    let constant = |x: ValueId| facts.constant(f, x).map(|k| k as u32);
    if constant(v).is_some() {
        return;
    }
    let op = if depth > 16 { None } else { facts.op(f, v) };
    match op {
        Some(Op::Int(IntOp::Add, a, b)) => {
            linear_terms(f, facts, a, scale, terms, depth + 1);
            linear_terms(f, facts, b, scale, terms, depth + 1);
        }
        Some(Op::Int(IntOp::Sub, a, b)) => {
            linear_terms(f, facts, a, scale, terms, depth + 1);
            linear_terms(f, facts, b, scale.wrapping_neg(), terms, depth + 1);
        }
        Some(Op::Int(IntOp::Mul, a, b)) if constant(b).is_some() => {
            linear_terms(f, facts, a, scale.wrapping_mul(constant(b).unwrap()), terms, depth + 1);
        }
        Some(Op::Int(IntOp::Mul, a, b)) if constant(a).is_some() => {
            linear_terms(f, facts, b, scale.wrapping_mul(constant(a).unwrap()), terms, depth + 1);
        }
        Some(Op::Int(IntOp::Shl, a, b)) if constant(b).is_some() => {
            linear_terms(f, facts, a, scale.wrapping_mul(1u32 << (constant(b).unwrap() & 31)), terms, depth + 1);
        }
        _ => {
            let c = terms.entry(v).or_insert(0);
            *c = c.wrapping_add(scale);
        }
    }
}

fn equality(f: &Func, facts: &Facts, p: IntPred, a: ValueId, b: ValueId) -> Option<Threshold> {
    if !matches!(p, IntPred::Eq | IntPred::Ne) || f.types[a.0] != Ty::I32 {
        return None;
    }
    let (value, k) = match (facts.constant(f, a), facts.constant(f, b)) {
        (None, Some(k)) => (a, k),
        (Some(k), None) => (b, k),
        _ => return None,
    };
    Some(Threshold {
        value,
        signed: false,
        at: k as u32 as i64,
        flip: p == IntPred::Ne,
        index: 0,
        of: value,
        equal: true,
    })
}

fn threshold(f: &Func, facts: &Facts, p: IntPred, a: ValueId, b: ValueId) -> Option<Threshold> {
    if matches!(p, IntPred::Eq | IntPred::Ne) || f.types[a.0] != Ty::I32 {
        return None;
    }
    let (q, x, y, negated) = ordered(p, a, b);
    let signed = q == IntPred::Slt;
    let at = |k: u64| if signed { k as u32 as i32 as i64 } else { k as u32 as i64 };
    let (value, at, flip) = match (facts.constant(f, x), facts.constant(f, y)) {
        (None, Some(k)) => (x, at(k), negated),
        (Some(k), None) => (y, at(k) + 1, !negated),
        _ => return None,
    };
    Some(Threshold {
        value,
        signed,
        at,
        flip,
        index: 0,
        of: value,
        equal: false,
    })
}

fn ordered(p: IntPred, a: ValueId, b: ValueId) -> (IntPred, ValueId, ValueId, bool) {
    match p {
        IntPred::Uge => (IntPred::Ult, a, b, true),
        IntPred::Ugt => (IntPred::Ult, b, a, false),
        IntPred::Ule => (IntPred::Ult, b, a, true),
        IntPred::Sge => (IntPred::Slt, a, b, true),
        IntPred::Sgt => (IntPred::Slt, b, a, false),
        IntPred::Sle => (IntPred::Slt, b, a, true),
        _ => (p, a, b, false),
    }
}

pub fn constant_choices(f: &Func, facts: &Facts, v: ValueId) -> Option<Vec<u64>> {
    let mut found = Vec::new();
    let mut stack = vec![v];
    while let Some(v) = stack.pop() {
        match facts.op(f, v)? {
            Op::Const(_, k) => {
                if !found.contains(&k) {
                    found.push(k);
                }
            }
            Op::Select(_, a, b) => stack.extend([a, b]),
            _ => return None,
        }
        if found.len() + stack.len() > 8 {
            return None;
        }
    }
    Some(found)
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

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    type Word = std::rc::Rc<dyn Fn(u32, u32, u32) -> u32>;

    fn lane_words() -> (Build, Vec<(String, ValueId, Word)>) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let yes = b.constant(e, Ty::I1, 1);
        let table = k.buffer(&mut b, e, 8);
        let u = b.load(e, Space::Global, MemSize::B32, table, yes);
        let at = b.constant(e, Ty::I64, 4);
        let second = b.int(e, IntOp::Add, table, at);
        let v = b.load(e, Space::Global, MemSize::B32, second, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let mut words: Vec<(String, ValueId, Word)> = vec![
            ("lane".into(), lane, std::rc::Rc::new(|_, _, l| l)),
            ("u".into(), u, std::rc::Rc::new(|u, _, _| u)),
            ("v".into(), v, std::rc::Rc::new(|_, v, _| v)),
        ];
        let mut r = Random::new(53);
        for _ in 0..60 {
            let (i, j) = (r.below(words.len() as u64) as usize, r.below(words.len() as u64) as usize);
            let k = [1u32, 2, 3, 4, 5, 8, 31, 0x80, 0xf0f0][r.below(9) as usize];
            let ((nx, x, fx), (ny, y, fy)) = (words[i].clone(), words[j].clone());
            let kc = b.constant(e, Ty::I32, k as u64);
            let (name, value, truth): (String, ValueId, Word) = match r.below(9) {
                0 => (format!("({} & {})", nx, ny), b.int(e, IntOp::And, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l) & fy(u, v, l))),
                1 => (format!("({} | {})", nx, ny), b.int(e, IntOp::Or, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l) | fy(u, v, l))),
                2 => (format!("({} ^ {})", nx, ny), b.int(e, IntOp::Xor, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l) ^ fy(u, v, l))),
                3 => (format!("({} & {})", nx, k), b.int(e, IntOp::And, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l) & k)),
                4 => (format!("({} | {})", nx, k), b.int(e, IntOp::Or, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l) | k)),
                5 => (format!("({} << {})", nx, k & 31), b.int(e, IntOp::Shl, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l) << (k & 31))),
                6 => (format!("({} >> {})", nx, k & 31), b.int(e, IntOp::LShr, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l) >> (k & 31))),
                7 => (format!("({} + {})", nx, ny), b.int(e, IntOp::Add, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_add(fy(u, v, l)))),
                _ => (format!("({} - {})", nx, ny), b.int(e, IntOp::Sub, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_sub(fy(u, v, l)))),
            };
            words.push((name, value, truth));
        }
        (b, words)
    }

    #[test]
    fn known_bits_hold_the_bits_of_every_lane_value() {
        let (b, words) = lane_words();
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(&b.f, &facts, &BTreeSet::new(), &[]);
        let mut r = Random::new(59);
        let mut wrong = Vec::new();
        for (name, v, truth) in &words {
            let Some(bits) = logic.lane_bits(&b.f, &facts, *v) else {
                continue;
            };
            for _ in 0..40 {
                let (u, w) = (r.next() as u32, r.next() as u32);
                for l in 0..32u32 {
                    let (mask, value) = bits[l as usize];
                    let t = truth(u, w, l);
                    if t & mask != value {
                        wrong.push(format!("{} lane {} is {:#x}, not {:#x} under {:#x}", name, l, t, value, mask));
                    }
                }
            }
        }
        wrong.truncate(5);
        assert!(wrong.is_empty(), "{:?}", wrong);
    }

    #[test]
    fn uniform_tests_are_equal_in_every_lane() {
        let (mut b, mut words) = lane_words();
        let e = BlockId(0);
        let (lane, u) = (words[0].1, words[1].1);
        let difference = b.int(e, IntOp::Sub, u, lane);
        words.push(("(u - lane)".into(), difference, std::rc::Rc::new(|u, _, l| u.wrapping_sub(l))));
        let doubled = b.int(e, IntOp::Add, lane, lane);
        words.push(("(lane + lane)".into(), doubled, std::rc::Rc::new(|_, _, l| l.wrapping_add(l))));
        let mut tests = Vec::new();
        let mut r = Random::new(61);
        for _ in 0..3000 {
            let i = r.below(words.len() as u64) as usize;
            let j = r.below(words.len() as u64) as usize;
            let t = b.cmp(e, IntPred::Eq, words[i].1, words[j].1);
            tests.push((i, j, t));
        }
        let n = words.len();
        for (i, j) in [(n - 2, 0), (n - 2, n - 1), (0, n - 2)] {
            let t = b.cmp(e, IntPred::Eq, words[i].1, words[j].1);
            tests.push((i, j, t));
        }
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let mut wrong = Vec::new();
        let mut decided = 0;
        for &(i, j, t) in &tests {
            if facts.uniform[t.0] || !uniform_difference(&b.f, &facts, words[i].1, words[j].1) {
                continue;
            }
            decided += 1;
            for _ in 0..40 {
                let mut pick = |r: &mut Random| if r.below(2) == 0 { r.below(64) as u32 } else { r.next() as u32 };
                let (u, w) = (pick(&mut r), pick(&mut r));
                let first = (words[i].2)(u, w, 0) == (words[j].2)(u, w, 0);
                if (1..32).any(|l| ((words[i].2)(u, w, l) == (words[j].2)(u, w, l)) != first) {
                    wrong.push(format!("{} == {}", words[i].0, words[j].0));
                    break;
                }
            }
        }
        wrong.truncate(5);
        assert!(decided > 0 && wrong.is_empty(), "{} uniform, wrong {:?}", decided, wrong);
    }

    #[test]
    fn uniform_orders_are_equal_in_every_lane() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let yes = b.constant(e, Ty::I1, 1);
        let table = k.buffer(&mut b, e, 8);
        let byte = b.load(e, Space::Global, MemSize::U8, table, yes);
        let at = b.constant(e, Ty::I64, 4);
        let second = b.int(e, IntOp::Add, table, at);
        let half = b.load(e, Space::Global, MemSize::U16, second, yes);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let mut words: Vec<(String, ValueId, Word)> = vec![
            ("lane".into(), lane, std::rc::Rc::new(|_, _, l| l)),
            ("b".into(), byte, std::rc::Rc::new(|u, _, _| u & 0xff)),
            ("h".into(), half, std::rc::Rc::new(|_, v, _| v & 0xffff)),
        ];
        let mut r = Random::new(67);
        for _ in 0..40 {
            let (i, j) = (r.below(words.len() as u64) as usize, r.below(words.len() as u64) as usize);
            let k = [1u32, 2, 3, 5, 31, 0x80][r.below(6) as usize];
            let ((nx, x, fx), (ny, y, fy)) = (words[i].clone(), words[j].clone());
            let kc = b.constant(e, Ty::I32, k as u64);
            let (name, value, truth): (String, ValueId, Word) = match r.below(4) {
                0 => (format!("({} + {})", nx, ny), b.int(e, IntOp::Add, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_add(fy(u, v, l)))),
                1 => (format!("({} - {})", nx, ny), b.int(e, IntOp::Sub, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_sub(fy(u, v, l)))),
                2 => (format!("({} * {})", nx, k), b.int(e, IntOp::Mul, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_mul(k))),
                _ => (format!("({} + {})", nx, k), b.int(e, IntOp::Add, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_add(k))),
            };
            words.push((name, value, truth));
        }
        let step = b.constant(e, Ty::I32, 0x400_0000);
        let wide = b.int(e, IntOp::Mul, lane, step);
        let small = b.constant(e, Ty::I32, 16);
        let large = b.constant(e, Ty::I32, 0x500_0000);
        let low = b.int(e, IntOp::Add, wide, small);
        let high = b.int(e, IntOp::Add, wide, large);
        words.push(("lane * 2^26 + 16".into(), low, std::rc::Rc::new(|_, _, l| l.wrapping_mul(0x400_0000).wrapping_add(16))));
        words.push(("lane * 2^26 + 0x5000000".into(), high, std::rc::Rc::new(|_, _, l| l.wrapping_mul(0x400_0000).wrapping_add(0x500_0000))));
        let predicates = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge];
        let mut tests = Vec::new();
        let n = words.len();
        for p in predicates {
            tests.push((n - 2, n - 1, p));
        }
        for _ in 0..3000 {
            let i = r.below(words.len() as u64) as usize;
            let j = r.below(words.len() as u64) as usize;
            let p = predicates[r.below(predicates.len() as u64) as usize];
            tests.push((i, j, p));
        }
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let mut wrong = Vec::new();
        let mut decided = 0;
        for &(i, j, p) in &tests {
            if !uniform_order(&b.f, &facts, p, words[i].1, words[j].1) {
                continue;
            }
            decided += 1;
            for _ in 0..40 {
                let (u, v) = (r.next() as u32, r.next() as u32);
                let holds = |l: u32| compare(p, (words[i].2)(u, v, l), (words[j].2)(u, v, l));
                let first = holds(0);
                if (1..32).any(|l| holds(l) != first) {
                    wrong.push(format!("{} {:?} {}", words[i].0, p, words[j].0));
                    break;
                }
            }
        }
        wrong.truncate(5);
        assert!(decided > 0 && wrong.is_empty(), "{} uniform, wrong {:?}", decided, wrong);
    }

    struct SelfLoop {
        b: Build,
        body: BlockId,
        exec: ValueId,
        carried: ValueId,
    }

    fn self_loop() -> SelfLoop {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
        let e = BlockId(0);
        let yes = b.constant(e, Ty::I1, 1);
        let (body, q) = b.block(&[Ty::I1, Ty::I1]);
        let (exit, _) = b.block(&[]);
        b.br(e, body, vec![p[0], yes]);
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        let five = b.constant(body, Ty::I32, 5);
        let fresh = b.cmp(body, IntPred::Ult, lane, five);
        let one = b.constant(body, Ty::I1, 1);
        let stale = b.int(body, IntOp::Xor, fresh, one);
        let conjunction = b.int(body, IntOp::And, q[1], stale);
        b.cond_br(body, conjunction, (body, vec![q[0], fresh]), (exit, vec![]));
        SelfLoop {
            b,
            body,
            exec: q[0],
            carried: q[1],
        }
    }

    #[test]
    fn reach_keeps_the_values_a_self_loop_carries_around() {
        let SelfLoop {
            b,
            body,
            exec,
            carried,
            ..
        } = self_loop();
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let start = logic.atom(Atom::Bit(f.blocks[&f.entry].params[0].0));
        let reach = logic.reach(f, &facts, f.entry, start);
        let exec = logic.atom(Atom::Bit(exec));
        let carried = logic.atom(Atom::Bit(carried));
        let lost = logic.m.not(carried);
        let second = logic.m.and(exec, lost);
        let state = logic.m.and(reach[&body], second);
        assert_ne!(
            state,
            Bdd::FALSE,
            "the second iteration starts with the carried bit false, which the first iteration sends when its fresh bit is false"
        );
    }

    #[test]
    fn bit_and_view_follow_the_operations() {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)]);
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let item = p[1];
        let three = b.constant(e, Ty::I32, 3);
        let seven = b.constant(e, Ty::I32, 7);
        let c = b.cmp(e, IntPred::Ult, item, three);
        let d = b.cmp(e, IntPred::Ult, item, seven);
        let one = b.constant(e, Ty::I32, 1);
        let two = b.constant(e, Ty::I32, 2);
        let x = b.core(e, Ty::I32, Op::Select(c, one, two));
        let y = b.core(e, Ty::I32, Op::Select(d, two, one));
        let crossed = b.cmp(e, IntPred::Eq, x, y);
        let picked = b.cmp(e, IntPred::Eq, x, one);
        let same = b.cmp(e, IntPred::Ne, lane, lane);
        let constants = b.cmp(e, IntPred::Eq, three, seven);
        let bits = b.cmp(e, IntPred::Ne, c, d);
        let w = b.wave(e, WaveOp::Ballot, vec![c]);
        let v = b.wave(e, WaveOp::Ballot, vec![d]);
        let own = b.int(e, IntOp::LShr, w, lane);
        let own = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, own));
        let zero = b.constant(e, Ty::I32, 0);
        let ones = b.constant(e, Ty::I32, 0xffff_ffff);
        let nothing = b.int(e, IntOp::And, w, zero);
        let everything = b.int(e, IntOp::Or, v, ones);
        let differ = b.int(e, IntOp::Xor, w, v);
        let chosen = b.core(e, Ty::I32, Op::Select(c, w, v));
        let cast = b.core(e, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, differ));
        let valid = b.core(e, Ty::I1, Op::Env(Env::ValidLane));
        let exec = p[0];
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let four = b.constant(e, Ty::I32, 4);
        let apart = b.int(e, IntOp::And, one, two);
        let other = b.core(e, Ty::I32, Op::Select(c, two, four));
        let missed = b.int(e, IntOp::And, one, other);
        let kept = b.int(e, IntOp::And, three, other);
        let z = b.core(e, Ty::I32, Op::Select(d, one, two));
        let paired = b.int(e, IntOp::Xor, x, z);
        let not_c = b.cmp(e, IntPred::Uge, item, three);
        let swapped_c = b.cmp(e, IntPred::Ugt, three, item);
        let not_d = b.cmp(e, IntPred::Ule, seven, item);
        let signed = b.cmp(e, IntPred::Slt, item, three);
        let not_signed = b.cmp(e, IntPred::Sge, item, three);
        let far = b.constant(e, Ty::I32, 99);
        let low_lanes = b.cmp(e, IntPred::Ult, lane, three);
        let high_lanes = b.cmp(e, IntPred::Ugt, lane, seven);
        let no_lane = b.cmp(e, IntPred::Eq, lane, far);
        let every_lane = b.cmp(e, IntPred::Ne, far, lane);
        let minus = b.constant(e, Ty::I32, 0xffff_fffe);
        let signed_lanes = b.cmp(e, IntPred::Sgt, lane, minus);
        let hundred = b.constant(e, Ty::I32, 100);
        let five = b.constant(e, Ty::I32, 5);
        let kept_item = b.core(e, Ty::I32, Op::Select(c, item, hundred));
        let small = b.cmp(e, IntPred::Ult, kept_item, five);
        let large = b.cmp(e, IntPred::Ugt, five, kept_item);
        let chosen_small = b.cmp(e, IntPred::Ult, x, two);
        let chosen_any = b.cmp(e, IntPred::Ule, x, two);
        let odd_bit = b.int(e, IntOp::And, lane, one);
        let odd = b.cmp(e, IntPred::Eq, odd_bit, one);
        let group = b.int(e, IntOp::LShr, lane, three);
        let third_group = b.cmp(e, IntPred::Eq, group, two);
        let own_bit = b.int(e, IntOp::Shl, one, lane);
        let thirty_two = b.constant(e, Ty::I32, 32);
        let gone = b.int(e, IntOp::Shl, lane, thirty_two);
        let undefined = b.cmp(e, IntPred::Eq, gone, zero);
        let mixed = b.int(e, IntOp::Add, lane, item);
        let with_item = b.cmp(e, IntPred::Ult, mixed, three);
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let bc = logic.atom(Atom::Bit(c));
        let own_d = logic.atom(Atom::Bit(d));
        let bd = logic.m.or(own_d, bc);
        let _ = exec;
        let mut expect = Vec::new();
        let xor = logic.m.xor(bc, bd);
        let ndd = logic.m.not(bd);
        let crossed_formula = logic.m.ite(bc, ndd, bd);
        expect.push(("item < 7 after item < 3", logic.bit(f, &facts, d), bd));
        expect.push(("select(c, 1, 2) == select(d, 2, 1)", logic.bit(f, &facts, crossed), crossed_formula));
        expect.push(("select(c, 1, 2) == 1", logic.bit(f, &facts, picked), bc));
        expect.push(("lane != lane", logic.bit(f, &facts, same), Bdd::FALSE));
        expect.push(("3 == 7", logic.bit(f, &facts, constants), Bdd::FALSE));
        expect.push(("c != d on bits", logic.bit(f, &facts, bits), xor));
        expect.push(("trunc(ballot(c) >> lane)", logic.bit(f, &facts, own), bc));
        expect.push(("ballot(c) & 0", logic.view(f, &facts, nothing), Bdd::FALSE));
        expect.push(("ballot(d) | ~0", logic.view(f, &facts, everything), Bdd::TRUE));
        expect.push(("ballot(c) ^ ballot(d)", logic.view(f, &facts, differ), xor));
        let chosen_formula = logic.m.ite(bc, bc, bd);
        expect.push(("select(c, ballot(c), ballot(d))", logic.view(f, &facts, chosen), chosen_formula));
        expect.push(("bitcast of the xor", logic.view(f, &facts, cast), xor));
        expect.push(("valid lane", logic.bit(f, &facts, valid), Bdd::TRUE));
        expect.push(("converted any(c)", logic.bit(f, &facts, q), bc));
        expect.push(("1 & 2", logic.view(f, &facts, apart), Bdd::FALSE));
        expect.push(("1 & select(c, 2, 4)", logic.view(f, &facts, missed), Bdd::FALSE));
        let k2 = logic.word(2);
        let kept_formula = logic.m.and(bc, k2);
        expect.push(("3 & select(c, 2, 4)", logic.view(f, &facts, kept), kept_formula));
        let k3 = logic.word(3);
        let unequal = logic.m.and(xor, k3);
        expect.push(("select(c, 1, 2) ^ select(d, 1, 2)", logic.view(f, &facts, paired), unequal));
        let (nbc, nbd) = (logic.m.not(bc), logic.m.not(bd));
        expect.push(("item >= 3", logic.bit(f, &facts, not_c), nbc));
        expect.push(("3 > item", logic.bit(f, &facts, swapped_c), bc));
        expect.push(("7 <= item", logic.bit(f, &facts, not_d), nbd));
        let bs = logic.atom(Atom::Bit(signed));
        let negative = logic.m.and(bs, nbd);
        let signed_formula = logic.m.or(bc, negative);
        let not_signed_formula = logic.m.not(signed_formula);
        expect.push(("item s< 3", logic.bit(f, &facts, signed), signed_formula));
        expect.push(("item s>= 3", logic.bit(f, &facts, not_signed), not_signed_formula));
        let low = logic.lanes(|l| l < 3);
        expect.push(("lane < 3", logic.bit(f, &facts, low_lanes), low));
        let high = logic.lanes(|l| l > 7);
        expect.push(("lane > 7", logic.bit(f, &facts, high_lanes), high));
        expect.push(("lane == 99", logic.bit(f, &facts, no_lane), Bdd::FALSE));
        expect.push(("99 != lane", logic.bit(f, &facts, every_lane), Bdd::TRUE));
        expect.push(("lane s> -2", logic.bit(f, &facts, signed_lanes), Bdd::TRUE));
        let small_leaf = logic.atom(Atom::Bit(small));
        let small_formula = logic.m.and(bc, small_leaf);
        expect.push(("select(c, item, 100) < 5", logic.bit(f, &facts, small), small_formula));
        expect.push(("5 > select(c, item, 100)", logic.bit(f, &facts, large), small_formula));
        expect.push(("select(c, 1, 2) < 2", logic.bit(f, &facts, chosen_small), bc));
        expect.push(("select(c, 1, 2) <= 2", logic.bit(f, &facts, chosen_any), Bdd::TRUE));
        let odd_lanes = logic.lanes(|l| l & 1 == 1);
        expect.push(("(lane & 1) == 1", logic.bit(f, &facts, odd), odd_lanes));
        let lanes_16_to_23 = logic.lanes(|l| (16..24).contains(&l));
        expect.push(("lane >> 3 == 2", logic.bit(f, &facts, third_group), lanes_16_to_23));
        expect.push(("the lane bit of 1 << lane", logic.view(f, &facts, own_bit), Bdd::TRUE));
        let undefined_atom = logic.atom(Atom::Bit(undefined));
        expect.push(("(lane << 32) == 0", logic.bit(f, &facts, undefined), undefined_atom));
        let with_item_atom = logic.atom(Atom::Bit(with_item));
        expect.push(("lane + item < 3", logic.bit(f, &facts, with_item), with_item_atom));
        let lane_one = logic.lanes(|l| l == 1);
        let bits_of_two = logic.word(2);
        expect.push(("the lanes of 2", bits_of_two, lane_one));
        let five = logic.word(5);
        let seven_bits = logic.word(7);
        let both = logic.m.and(five, seven_bits);
        expect.push(("5 & 7 by lanes", both, five));
        let wrong: Vec<&str> = expect.iter().filter(|(_, got, want)| got != want).map(|(name, ..)| *name).collect();
        assert!(wrong.is_empty(), "{:?}", wrong);
        let _ = crossed_formula;
        assert_eq!(xor, logic.m.xor(bc, bd));
    }

    struct Edge2 {
        b: Build,
        src: BlockId,
        bits: Vec<ValueId>,
    }

    fn edge(r: &mut Random, self_loop: bool) -> Edge2 {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)]);
        let e = BlockId(0);
        let n = 4;
        let (src, s) = b.block(&vec![Ty::I1; n]);
        let seeds: Vec<ValueId> = (0..n).map(|_| b.constant(e, Ty::I1, r.below(2))).collect();
        b.br(e, src, seeds);
        let mut bits: Vec<ValueId> = s.clone();
        for k in 0..3 {
            let bound = b.constant(src, Ty::I32, k + 1);
            bits.push(b.cmp(src, IntPred::Ult, p[1], bound));
        }
        let one = b.constant(src, Ty::I1, 1);
        let zero = b.constant(src, Ty::I1, 0);
        let pick = |r: &mut Random, bits: &[ValueId]| bits[r.below(bits.len() as u64) as usize];
        let args: Vec<ValueId> = (0..n)
            .map(|_| match r.below(6) {
                0 => one,
                1 => zero,
                2 => {
                    let x = pick(r, &bits);
                    b.int(src, IntOp::Xor, x, one)
                }
                3 => {
                    let (x, y) = (pick(r, &bits), pick(r, &bits));
                    b.int(src, IntOp::And, x, y)
                }
                _ => pick(r, &bits),
            })
            .collect();
        let (exit, _) = b.block(&[]);
        let dst = if self_loop {
            src
        } else {
            let (dst, _) = b.block(&vec![Ty::I1; n]);
            dst
        };
        let cond = pick(r, &bits);
        b.cond_br(src, cond, (dst, args), (exit, vec![]));
        Edge2 { b, src, bits }
    }

    fn images(seed: u64, rounds: usize, check: impl Fn(&mut Logic, Bdd, Bdd, Bdd) -> bool) -> Vec<(usize, &'static str)> {
        crossings(seed, rounds, false, check)
    }

    fn projection(logic: &mut Logic, f: &Func, facts: &Facts, src: BlockId, formula: Bdd, linked: &[bool], self_loop: bool) -> Bdd {
        let edge = f.blocks[&src].term.edges().next().unwrap().clone();
        let dst = &f.blocks[&edge.dst];
        let mut fresh = Vec::new();
        let mut relation = formula;
        for (k, (&(param, _), &arg)) in dst.params.iter().zip(&edge.args).enumerate() {
            if !linked[k] {
                continue;
            }
            let target = if self_loop {
                let t = logic.atom(Atom::Fresh(7, param, 0));
                fresh.push((t, param));
                t
            } else {
                logic.atom(Atom::Bit(param))
            };
            let bound = logic.bit(f, facts, arg);
            let link = logic.m.iff(target, bound);
            relation = logic.m.and(relation, link);
        }
        let scoped: Vec<u32> = logic.support(relation).iter().copied().filter(|&v| logic.scope(facts, v) == Some(src)).collect();
        let mut want = logic.exists(&scoped, relation);
        if self_loop {
            let renamed: HashMap<u32, Bdd> = fresh.iter().map(|&(t, param)| (logic.support(t)[0], logic.atom(Atom::Bit(param)))).collect();
            want = logic.m.compose(want, &|v| renamed.get(&v).copied());
        }
        want
    }

    fn crossings(seed: u64, rounds: usize, difference: bool, check: impl Fn(&mut Logic, Bdd, Bdd, Bdd) -> bool) -> Vec<(usize, &'static str)> {
        let mut r = Random::new(seed);
        let mut wrong = Vec::new();
        for round in 0..rounds {
            let self_loop = round % 2 == 1;
            let Edge2 { b, src, bits } = edge(&mut r, self_loop);
            let f = &b.f;
            let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
            let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
            let mut formula = Bdd::FALSE;
            for _ in 0..3 {
                let mut term = Bdd::TRUE;
                for _ in 0..2 {
                    let v = bits[r.below(bits.len() as u64) as usize];
                    let a = logic.bit(f, &facts, v);
                    let a = if r.below(2) == 0 { a } else { logic.m.not(a) };
                    term = logic.m.and(term, a);
                }
                formula = logic.m.or(formula, term);
            }
            let got = if difference {
                logic.image(f, &facts, src, 0, formula)
            } else {
                logic.post(f, &facts, src, 0, formula)
            };
            let edge = f.blocks[&src].term.edges().next().unwrap().clone();
            let every = vec![true; edge.args.len()];
            let full = projection(&mut logic, f, &facts, src, formula, &every, self_loop);
            let mut linked = vec![!difference; edge.args.len()];
            if difference {
                let mut reached: BTreeSet<u32> = logic.support(formula).iter().copied().filter(|&v| logic.scope(&facts, v) == Some(src)).collect();
                loop {
                    let mut grew = false;
                    for (k, &arg) in edge.args.iter().enumerate() {
                        let bound = logic.bit(f, &facts, arg);
                        let support: Vec<u32> = logic.support(bound).iter().copied().filter(|&v| logic.scope(&facts, v) == Some(src)).collect();
                        if !linked[k] && support.iter().any(|v| reached.contains(v)) {
                            linked[k] = true;
                            reached.extend(support);
                            grew = true;
                        }
                    }
                    if !grew {
                        break;
                    }
                }
            }
            let reachable = projection(&mut logic, f, &facts, src, formula, &linked, self_loop);
            if !check(&mut logic, got, full, reachable) {
                wrong.push((round, if self_loop { "self-loop" } else { "edge" }));
            }
        }
        wrong
    }

    #[test]
    fn image_keeps_every_state_the_argument_relations_allow() {
        let wrong = crossings(31, 400, true, |logic, got, full, _| logic.m.implies(full, got));
        assert!(wrong.is_empty(), "rounds whose difference image drops a state the edge can reach: {:?}", wrong);
    }

    #[test]
    fn image_equals_the_projection_of_the_formula_and_the_relations_it_reaches() {
        let wrong = crossings(31, 400, true, |_, got, _, reachable| got == reachable);
        assert!(wrong.is_empty(), "rounds whose difference image differs from the projection over the relations it reaches: {:?}", wrong);
    }

    #[test]
    fn post_keeps_every_state_the_argument_relations_allow() {
        let wrong = images(29, 400, |logic, got, full, _| logic.m.implies(full, got));
        assert!(wrong.is_empty(), "rounds whose image drops a state the edge can reach: {:?}", wrong);
    }

    #[test]
    fn post_equals_the_projection_of_the_formula_and_every_argument_relation() {
        let wrong = images(29, 400, |_, got, full, _| got == full);
        assert!(wrong.is_empty(), "rounds whose image differs from the projection: {:?}", wrong);
    }

    #[test]
    fn choose_keeps_the_first_choices_it_can_convert_and_nothing_it_need_not_keep() {
        let (b, _) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let n = 6;
        let listed: Vec<Choice> = (0..n).map(Choice::Meet).collect();
        let mut r = Random::new(3);
        for _ in 0..300 {
            let mut logic = Logic::open(f, &facts, &listed);
            let markers: Vec<Bdd> = listed.iter().map(|&c| logic.atom(Atom::Marker(c))).collect();
            let table: Vec<bool> = (0..1u32 << n).map(|_| r.below(3) == 0).collect();
            if !table.iter().any(|&x| x) {
                continue;
            }
            let mut safe = Bdd::FALSE;
            for (assignment, &allowed) in table.iter().enumerate() {
                if !allowed {
                    continue;
                }
                let mut term = Bdd::TRUE;
                for (i, &m) in markers.iter().enumerate() {
                    let literal = if assignment >> i & 1 != 0 { m } else { logic.m.not(m) };
                    term = logic.m.and(term, literal);
                }
                safe = logic.m.or(safe, term);
            }
            let kept = logic.choose(safe);
            let local = |kept: &BTreeSet<usize>| (0..n).filter(|i| !kept.contains(i)).fold(0usize, |a, i| a | 1 << i);
            let chosen = local(&kept.meets);
            assert!(table[chosen], "the choice lies outside the safe set");
            for &i in &kept.meets {
                assert!(!table[chosen | 1 << i], "converting Meet({}) alone stays safe", i);
            }
            let mut greedy = 0usize;
            for i in 0..n {
                let with = greedy | 1 << i;
                let fits = (0..1usize << n).any(|a| table[a] && a & ((1 << (i + 1)) - 1) == with);
                if fits {
                    greedy = with;
                }
            }
            assert_eq!(chosen, greedy, "the choice is not the greedy one in listed order");
        }
    }
}
