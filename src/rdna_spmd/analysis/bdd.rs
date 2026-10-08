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

const ITE_SLOTS: usize = 1 << 19;

pub struct Manager {
    nodes: Vec<Node>,
    unique: HashMap<(u32, Bdd, Bdd), Bdd>,

    ite: Vec<(Bdd, Bdd, Bdd, Bdd)>,
    restrict: HashMap<(Bdd, Bdd), Bdd>,
    implications: HashMap<(Bdd, Bdd), bool>,
    memo: HashMap<Bdd, Bdd>,
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
            memo: HashMap::default(),
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
        if self.ite.len() < self.nodes.len().min(ITE_SLOTS) {
            let slots = (self.nodes.len() * 2).next_power_of_two().clamp(1 << 10, ITE_SLOTS);
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
        let mut memo = self.take_memo();
        let r = self.quantify_memo(f, chosen, any, &mut memo);
        self.memo = memo;
        r
    }

    fn take_memo(&mut self) -> HashMap<Bdd, Bdd> {
        let mut memo = std::mem::take(&mut self.memo);
        if memo.capacity() > 4 * memo.len().max(1 << 10) {
            memo = HashMap::with_capacity_and_hasher(memo.len().max(1 << 10), Default::default());
        } else {
            memo.clear();
        }
        memo
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
        let mut memo = self.take_memo();
        let r = self.compose_memo(f, map, u32::MAX, &mut memo);
        self.memo = memo;
        r
    }

    pub fn compose_many(&mut self, fs: &[Bdd], map: &dyn Fn(u32) -> Option<Bdd>, last: u32) -> Vec<Bdd> {
        let mut memo = HashMap::default();
        fs.iter().map(|&f| self.compose_memo(f, map, last, &mut memo)).collect()
    }

    fn compose_memo(
        &mut self,
        f: Bdd,
        map: &dyn Fn(u32) -> Option<Bdd>,
        last: u32,
        memo: &mut HashMap<Bdd, Bdd>,
    ) -> Bdd {
        if f.constant().is_some() || self.top(f) > last {
            return f;
        }
        if let Some(&r) = memo.get(&f) {
            return r;
        }
        let node = self.nodes[f.0 as usize];
        let low = self.compose_memo(node.low, map, last, memo);
        let high = self.compose_memo(node.high, map, last, memo);
        let v = match map(node.var) {
            Some(g) => g,
            None => self.var(node.var),
        };
        let n = self.nodes[v.0 as usize];
        let r = if n.low == Bdd::FALSE && n.high == Bdd::TRUE && n.var < self.top(low) && n.var < self.top(high) {
            self.node(n.var, low, high)
        } else {
            self.ite(v, high, low)
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    fn random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn random_function(m: &mut Manager, state: &mut u64, vars: u32) -> Bdd {
        let mut g = Manager::constant(random(state) & 1 == 1);
        for _ in 0..6 {
            let v = m.var(random(state) as u32 % vars);
            let w = m.var(random(state) as u32 % vars);
            g = match random(state) % 3 {
                0 => {
                    let both = m.and(v, w);
                    m.xor(g, both)
                }
                1 => m.or(g, v),
                _ => {
                    let not = m.not(w);
                    m.and(g, not)
                }
            };
        }
        g
    }

    fn evaluate(m: &Manager, mut g: Bdd, value: &dyn Fn(u32) -> bool) -> bool {
        while let Some((var, low, high)) = m.decompose(g) {
            g = if value(var) { high } else { low };
        }
        g == Bdd::TRUE
    }

    #[test]
    fn each_quantification_reads_only_its_own_variables() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        let mut m = Manager::new();
        let vars = 8;
        for round in 0..600 {
            let f = random_function(&mut m, &mut state, vars);
            let chosen: Vec<u32> = (0..vars).filter(|_| random(&mut state) % 3 == 0).collect();
            let any = round % 2 == 0;
            let g = if any { m.exists(f, &|v| chosen.contains(&v)) } else { m.forall(f, &|v| chosen.contains(&v)) };
            for row in 0..1u32 << vars {
                let mut outcomes = (0..1u32 << chosen.len()).map(|pick| {
                    let value = |v: u32| match chosen.iter().position(|&c| c == v) {
                        Some(i) => pick >> i & 1 == 1,
                        None => row >> v & 1 == 1,
                    };
                    evaluate(&m, f, &value)
                });
                let want = if any { outcomes.any(|x| x) } else { outcomes.all(|x| x) };
                assert_eq!(evaluate(&m, g, &|v| row >> v & 1 == 1), want, "quantifying over {:?}", chosen);
            }
        }
    }

    #[test]
    fn compose_renames_variables_to_others_above_and_below_them() {
        let mut state = 0x2545_f491_4f6c_dd1d;
        let mut m = Manager::new();
        let vars = 10;
        let mut moved = 0;
        for _ in 0..400 {
            let f = random_function(&mut m, &mut state, vars);
            let mut map: HashMap<u32, Bdd> = HashMap::default();
            for _ in 0..1 + random(&mut state) % 5 {
                let v = random(&mut state) as u32 % vars;
                let w = random(&mut state) as u32 % vars;
                moved += (v != w) as usize;
                let target = m.var(w);
                map.insert(v, if random(&mut state) % 4 == 0 { m.not(target) } else { target });
            }
            let g = m.compose(f, &|v| map.get(&v).copied());
            let mut stack = vec![g];
            while let Some((var, low, high)) = stack.pop().and_then(|x| m.decompose(x)) {
                for child in [low, high] {
                    assert!(m.decompose(child).is_none_or(|(below, _, _)| var < below), "a node must test its variable before every variable below it");
                    stack.push(child);
                }
            }
            for row in 0..1u32 << vars {
                let before = |v: u32| row >> v & 1 == 1;
                let after = |v: u32| match map.get(&v) {
                    Some(&target) => evaluate(&m, target, &before),
                    None => before(v),
                };
                assert_eq!(evaluate(&m, g, &before), evaluate(&m, f, &after), "renaming must read each mapped variable as its target");
            }
        }
        assert!(moved > 1000);
    }

    #[test]
    fn compose_many_substitutes_every_mapped_variable_in_every_root() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        let mut m = Manager::new();
        let vars = 10;
        for _ in 0..300 {
            let roots: Vec<Bdd> = (0..3).map(|_| random_function(&mut m, &mut state, vars)).collect();
            let mut map: HashMap<u32, Bdd> = HashMap::default();
            for _ in 0..1 + random(&mut state) % 4 {
                let v = random(&mut state) as u32 % vars;
                let target = random_function(&mut m, &mut state, vars);
                map.insert(v, target);
            }
            let last = *map.keys().max().unwrap();
            let many = m.compose_many(&roots, &|v| map.get(&v).copied(), last);
            let one: Vec<Bdd> = roots.iter().map(|&f| m.compose(f, &|v| map.get(&v).copied())).collect();
            assert_eq!(many, one, "composing the roots together must give what composing each gives");
            for row in 0..1u32 << vars {
                let before = |v: u32| row >> v & 1 == 1;
                let after = |v: u32| match map.get(&v) {
                    Some(&target) => evaluate(&m, target, &before),
                    None => before(v),
                };
                for (&f, &g) in roots.iter().zip(&many) {
                    assert_eq!(evaluate(&m, g, &before), evaluate(&m, f, &after), "the result must read each mapped variable as its function");
                }
            }
        }
    }
}
