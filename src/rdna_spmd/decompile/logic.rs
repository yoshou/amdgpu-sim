use super::address::compare;
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

#[derive(Default)]
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
            Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => {
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
                        _ => self.atom(opaque),
                    };
                    self.compare(f, facts, p, a, b, leaf, &mut HashMap::default())
                }
                Op::FCmp(p, a, b) => {
                    let leaf = self.atom(opaque);
                    self.float_order(f, facts, p, a, b, leaf, &mut HashMap::default())
                }
                Op::Cmp(p, a, b) => {
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
        let (min, max) = if t.signed {
            (i32::MIN as i64, i32::MAX as i64)
        } else {
            (0, u32::MAX as i64)
        };
        if t.at <= min {
            return Bdd::FALSE;
        }
        if t.at > max {
            return Bdd::TRUE;
        }
        let list = self.block_thresholds(f, facts, block);
        let earlier = list.iter().filter(|e| e.index < index && e.value == t.value && !e.equal && e.signed == t.signed);
        let (mut lower, mut upper): (Option<Threshold>, Option<Threshold>) = (None, None);
        for e in earlier {
            if e.at == t.at {
                let bit = self.bit(f, facts, e.of);
                return if e.flip { self.m.not(bit) } else { bit };
            }
            if e.at < t.at && lower.is_none_or(|l| l.at < e.at) {
                lower = Some(*e);
            }
            if e.at > t.at && upper.is_none_or(|u| u.at > e.at) {
                upper = Some(*e);
            }
        }
        let atom = self.atom(Atom::Bit(v));
        let mut holds = if t.flip { self.m.not(atom) } else { atom };
        if let Some(u) = upper {
            let bit = self.bit(f, facts, u.of);
            let upper = if u.flip { self.m.not(bit) } else { bit };
            holds = self.m.and(holds, upper);
        }
        if let Some(l) = lower {
            let bit = self.bit(f, facts, l.of);
            let lower = if l.flip { self.m.not(bit) } else { bit };
            holds = self.m.or(holds, lower);
        }
        let equalities: Vec<Threshold> = list.iter().filter(|e| e.index < index && e.value == t.value && e.equal).copied().collect();
        for e in equalities {
            let k = e.constant(t.signed);
            let bit = self.bit(f, facts, e.of);
            let equal = if e.flip { self.m.not(bit) } else { bit };
            holds = if k < t.at {
                self.m.or(holds, equal)
            } else {
                let unequal = self.m.not(equal);
                self.m.and(holds, unequal)
            };
        }
        holds
    }

    fn equal(&mut self, f: &Func, facts: &Facts, block: BlockId, index: usize, v: ValueId, e: Threshold) -> Bdd {
        let list = self.block_thresholds(f, facts, block);
        let earlier: Vec<Threshold> = list.iter().filter(|x| x.index < index && x.value == e.value).copied().collect();
        if let Some(same) = earlier.iter().find(|x| x.equal && x.at == e.at) {
            let bit = self.bit(f, facts, same.of);
            return if same.flip { self.m.not(bit) } else { bit };
        }
        let atom = self.atom(Atom::Bit(v));
        let mut holds = if e.flip { self.m.not(atom) } else { atom };
        for x in earlier {
            let bit = self.bit(f, facts, x.of);
            let other = if x.flip { self.m.not(bit) } else { bit };
            let implied = if x.equal {
                self.m.not(other)
            } else if e.constant(x.signed) < x.at {
                other
            } else {
                self.m.not(other)
            };
            holds = self.m.and(holds, implied);
        }
        holds
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
        let reversed = self.first_order(f, block).get(&(q, y, x)).copied();
        match reversed {
            Some(r) if matches!(facts.site[r.0], Site::Inst { index: i, .. } if i < index) => {
                let other = self.relation(f, facts, r, (q, y, x));
                let not_other = self.m.not(other);
                self.m.and(holds, not_other)
            }
            _ => holds,
        }
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
        let links = Rc::new(links);
        self.relations.insert((src, slot), links.clone());
        links
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

fn float_compare(pred: FloatPred, x: f64, y: f64) -> bool {
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
    fn constant(&self, signed: bool) -> i64 {
        if signed {
            self.at as u32 as i32 as i64
        } else {
            self.at as u32 as i64
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
        let nbs = logic.m.not(bs);
        expect.push(("item s< 3", logic.bit(f, &facts, signed), bs));
        expect.push(("item s>= 3", logic.bit(f, &facts, not_signed), nbs));
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
