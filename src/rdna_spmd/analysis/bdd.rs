use std::collections::BTreeSet;
use crate::rdna_spmd::hash::{HashMap, Mix};
use std::hash::Hasher;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Bdd(u32);

impl Bdd {
    pub const FALSE: Bdd = Bdd(0);
    pub const TRUE: Bdd = Bdd(1);
    pub fn constant(self) -> Option<bool> {
        match self {
            Self::FALSE => Some(false),
            Self::TRUE => Some(true),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct Node {
    var: u32,
    low: Bdd,
    high: Bdd,
}

const TERMINAL: u32 = u32::MAX;

pub struct Manager {
    nodes: Vec<Node>,
    unique: HashMap<(u32, Bdd, Bdd), Bdd>,

    ite: Vec<(Bdd, Bdd, Bdd, Bdd)>,
    restrict: HashMap<(Bdd, Bdd), Bdd>,
    implications: HashMap<(Bdd, Bdd), bool>,
}

impl Manager {
    pub fn new() -> Self {
        let terminal = |value| Node {
            var: TERMINAL,
            low: Bdd(value),
            high: Bdd(value),
        };
        Self {
            nodes: vec![terminal(0), terminal(1)],
            unique: HashMap::default(),
            ite: Vec::new(),
            restrict: HashMap::default(),
            implications: HashMap::default(),
        }
    }

    pub fn decompose(&self, f: Bdd) -> Option<(u32, Bdd, Bdd)> {
        let node = self.nodes[f.0 as usize];
        (node.var != TERMINAL).then_some((node.var, node.low, node.high))
    }

    pub fn cofactor(&mut self, f: Bdd, var: u32, value: bool) -> Bdd {
        let constant = Manager::constant(value);
        self.compose(f, &|v| (v == var).then_some(constant))
    }

    pub fn restrict(&mut self, f: Bdd, care: Bdd) -> Bdd {
        if care == Bdd::FALSE || care == Bdd::TRUE || f.constant().is_some() {
            return f;
        }
        if let Some(&r) = self.restrict.get(&(f, care)) {
            return r;
        }
        let var = self.top(f).min(self.top(care));
        let (f0, f1) = self.cofactors(f, var);
        let (c0, c1) = self.cofactors(care, var);
        let r = if c0 == Bdd::FALSE {
            self.restrict(f1, c1)
        } else if c1 == Bdd::FALSE {
            self.restrict(f0, c0)
        } else if f0 == f1 {
            let merged = self.or(c0, c1);
            self.restrict(f0, merged)
        } else {
            let low = self.restrict(f0, c0);
            let high = self.restrict(f1, c1);
            self.node(var, low, high)
        };
        self.restrict.insert((f, care), r);
        r
    }

    fn top(&self, f: Bdd) -> u32 {
        self.nodes[f.0 as usize].var
    }

    fn cofactors(&self, f: Bdd, var: u32) -> (Bdd, Bdd) {
        let node = self.nodes[f.0 as usize];
        if node.var == var {
            (node.low, node.high)
        } else {
            (f, f)
        }
    }

    fn node(&mut self, var: u32, low: Bdd, high: Bdd) -> Bdd {
        if low == high {
            return low;
        }
        if let Some(&id) = self.unique.get(&(var, low, high)) {
            return id;
        }
        let id = Bdd(self.nodes.len() as u32);
        self.nodes.push(Node { var, low, high });
        self.unique.insert((var, low, high), id);
        id
    }

    pub fn var(&mut self, var: u32) -> Bdd {
        assert!(var != TERMINAL, "variable number reserved for terminals");
        self.node(var, Bdd::FALSE, Bdd::TRUE)
    }

    pub fn constant(value: bool) -> Bdd {
        if value {
            Bdd::TRUE
        } else {
            Bdd::FALSE
        }
    }

    fn slot(&self, f: Bdd, g: Bdd, h: Bdd) -> usize {
        let mut hash = Mix::default();
        hash.write_u64(f.0 as u64 | (g.0 as u64) << 32);
        hash.write_u64(h.0 as u64);
        hash.finish() as usize & (self.ite.len() - 1)
    }

    pub fn ite(&mut self, f: Bdd, g: Bdd, h: Bdd) -> Bdd {
        match f.constant() {
            Some(true) => return g,
            Some(false) => return h,
            None => {}
        }
        if g == h {
            return g;
        }
        if g == Bdd::TRUE && h == Bdd::FALSE {
            return f;
        }

        let (f, g, h) = if h == Bdd::FALSE && g.0 < f.0 {
            (g, f, h)
        } else if g == Bdd::TRUE && h.0 < f.0 {
            (h, g, f)
        } else {
            (f, g, h)
        };
        if self.ite.len() < self.nodes.len() && self.ite.len() < 1 << 22 {
            let slots = (self.nodes.len() * 2).next_power_of_two().max(1 << 10);
            self.ite = vec![(Bdd::FALSE, Bdd::FALSE, Bdd::FALSE, Bdd::FALSE); slots];
        }
        let slot = self.slot(f, g, h);
        let (cf, cg, ch, cached) = self.ite[slot];

        if (cf, cg, ch) == (f, g, h) {
            return cached;
        }
        let var = self.top(f).min(self.top(g)).min(self.top(h));
        let (f0, f1) = self.cofactors(f, var);
        let (g0, g1) = self.cofactors(g, var);
        let (h0, h1) = self.cofactors(h, var);
        let low = self.ite(f0, g0, h0);
        let high = self.ite(f1, g1, h1);
        let r = self.node(var, low, high);

        let slot = self.slot(f, g, h);
        self.ite[slot] = (f, g, h, r);
        r
    }

    pub fn not(&mut self, f: Bdd) -> Bdd {
        self.ite(f, Bdd::FALSE, Bdd::TRUE)
    }
    pub fn and(&mut self, f: Bdd, g: Bdd) -> Bdd {
        self.ite(f, g, Bdd::FALSE)
    }
    pub fn or(&mut self, f: Bdd, g: Bdd) -> Bdd {
        self.ite(f, Bdd::TRUE, g)
    }
    pub fn xor(&mut self, f: Bdd, g: Bdd) -> Bdd {
        let ng = self.not(g);
        self.ite(f, ng, g)
    }
    pub fn iff(&mut self, f: Bdd, g: Bdd) -> Bdd {
        let ng = self.not(g);
        self.ite(f, g, ng)
    }

    pub fn implies(&mut self, f: Bdd, g: Bdd) -> bool {
        if f == Bdd::FALSE || g == Bdd::TRUE || f == g {
            return true;
        }
        if f == Bdd::TRUE || g == Bdd::FALSE {
            return false;
        }
        if let Some(&result) = self.implications.get(&(f, g)) {
            return result;
        }

        let var = self.top(f).min(self.top(g));
        let (f0, f1) = self.cofactors(f, var);
        let (g0, g1) = self.cofactors(g, var);
        let result = self.implies(f0, g0) && self.implies(f1, g1);
        self.implications.insert((f, g), result);
        result
    }

    fn quantify(&mut self, f: Bdd, chosen: &dyn Fn(u32) -> bool, any: bool) -> Bdd {
        let mut memo = HashMap::default();
        self.quantify_memo(f, chosen, any, &mut memo)
    }

    fn quantify_memo(
        &mut self,
        f: Bdd,
        chosen: &dyn Fn(u32) -> bool,
        any: bool,
        memo: &mut HashMap<Bdd, Bdd>,
    ) -> Bdd {
        if f.constant().is_some() {
            return f;
        }
        if let Some(&r) = memo.get(&f) {
            return r;
        }
        let node = self.nodes[f.0 as usize];
        let low = self.quantify_memo(node.low, chosen, any, memo);
        let high = self.quantify_memo(node.high, chosen, any, memo);
        let r = if chosen(node.var) {
            if any {
                self.or(low, high)
            } else {
                self.and(low, high)
            }
        } else {
            let v = self.var(node.var);
            self.ite(v, high, low)
        };
        memo.insert(f, r);
        r
    }

    pub fn and_exists(&mut self, f: Bdd, g: Bdd, chosen: &dyn Fn(u32) -> bool) -> Bdd {
        let mut memo = HashMap::default();
        self.and_exists_memo(f, g, chosen, &mut memo)
    }

    fn and_exists_memo(
        &mut self,
        f: Bdd,
        g: Bdd,
        chosen: &dyn Fn(u32) -> bool,
        memo: &mut HashMap<(Bdd, Bdd), Bdd>,
    ) -> Bdd {
        if f == Bdd::FALSE || g == Bdd::FALSE {
            return Bdd::FALSE;
        }
        if f == Bdd::TRUE && g == Bdd::TRUE {
            return Bdd::TRUE;
        }
        let (f, g) = if g.0 < f.0 { (g, f) } else { (f, g) };
        if let Some(&r) = memo.get(&(f, g)) {
            return r;
        }
        let var = self.top(f).min(self.top(g));
        let (f0, f1) = self.cofactors(f, var);
        let (g0, g1) = self.cofactors(g, var);
        let low = self.and_exists_memo(f0, g0, chosen, memo);
        let r = if chosen(var) {
            if low == Bdd::TRUE {
                low
            } else {
                let high = self.and_exists_memo(f1, g1, chosen, memo);
                self.or(low, high)
            }
        } else {
            let high = self.and_exists_memo(f1, g1, chosen, memo);
            self.node(var, low, high)
        };
        memo.insert((f, g), r);
        r
    }

    pub fn exists(&mut self, f: Bdd, chosen: &dyn Fn(u32) -> bool) -> Bdd {
        self.quantify(f, chosen, true)
    }

    pub fn forall(&mut self, f: Bdd, chosen: &dyn Fn(u32) -> bool) -> Bdd {
        self.quantify(f, chosen, false)
    }

    pub fn compose(&mut self, f: Bdd, map: &dyn Fn(u32) -> Option<Bdd>) -> Bdd {
        let mut memo = HashMap::default();
        self.compose_memo(f, map, &mut memo)
    }

    fn compose_memo(
        &mut self,
        f: Bdd,
        map: &dyn Fn(u32) -> Option<Bdd>,
        memo: &mut HashMap<Bdd, Bdd>,
    ) -> Bdd {
        if f.constant().is_some() {
            return f;
        }
        if let Some(&r) = memo.get(&f) {
            return r;
        }
        let node = self.nodes[f.0 as usize];
        let low = self.compose_memo(node.low, map, memo);
        let high = self.compose_memo(node.high, map, memo);
        let v = match map(node.var) {
            Some(g) => g,
            None => self.var(node.var),
        };
        let r = self.ite(v, high, low);
        memo.insert(f, r);
        r
    }

    pub fn support(&self, f: Bdd) -> BTreeSet<u32> {
        let mut out = BTreeSet::new();
        let mut stack = vec![f];
        let mut seen: HashMap<Bdd, ()> = HashMap::default();
        while let Some(g) = stack.pop() {
            if g.constant().is_some() || seen.insert(g, ()).is_some() {
                continue;
            }
            let node = self.nodes[g.0 as usize];
            out.insert(node.var);
            stack.push(node.low);
            stack.push(node.high);
        }
        out
    }
}
