use std::collections::{BTreeSet, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Bdd(u32);

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

pub(super) struct Manager {
    nodes: Vec<Node>,
    unique: HashMap<(u32, Bdd, Bdd), Bdd>,
    ite: HashMap<(Bdd, Bdd, Bdd), Bdd>,
    restrict: HashMap<(Bdd, Bdd), Bdd>,
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
            unique: HashMap::new(),
            ite: HashMap::new(),
            restrict: HashMap::new(),
        }
    }

    /// The variable a function tests first and its two cofactors on it.
    pub fn decompose(&self, f: Bdd) -> Option<(u32, Bdd, Bdd)> {
        let node = self.nodes[f.0 as usize];
        (node.var != TERMINAL).then_some((node.var, node.low, node.high))
    }

    pub fn cofactor(&mut self, f: Bdd, var: u32, value: bool) -> Bdd {
        let constant = Manager::constant(value);
        self.compose(f, &|v| (v == var).then_some(constant))
    }

    /// A function that agrees with `f` wherever `care` holds, found by
    /// dropping every test that `care` already decides (Coudert and Madre's
    /// restrict). Outside `care` the result is unspecified.
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
        if let Some(&r) = self.ite.get(&(f, g, h)) {
            return r;
        }
        let var = self.top(f).min(self.top(g)).min(self.top(h));
        let (f0, f1) = self.cofactors(f, var);
        let (g0, g1) = self.cofactors(g, var);
        let (h0, h1) = self.cofactors(h, var);
        let low = self.ite(f0, g0, h0);
        let high = self.ite(f1, g1, h1);
        let r = self.node(var, low, high);
        self.ite.insert((f, g, h), r);
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
        let ng = self.not(g);
        self.and(f, ng) == Bdd::FALSE
    }

    fn quantify(&mut self, f: Bdd, chosen: &dyn Fn(u32) -> bool, any: bool) -> Bdd {
        let mut memo = HashMap::new();
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

    pub fn exists(&mut self, f: Bdd, chosen: &dyn Fn(u32) -> bool) -> Bdd {
        self.quantify(f, chosen, true)
    }

    pub fn forall(&mut self, f: Bdd, chosen: &dyn Fn(u32) -> bool) -> Bdd {
        self.quantify(f, chosen, false)
    }

    pub fn compose(&mut self, f: Bdd, map: &dyn Fn(u32) -> Option<Bdd>) -> Bdd {
        let mut memo = HashMap::new();
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
        let mut seen = BTreeSet::new();
        while let Some(g) = stack.pop() {
            if g.constant().is_some() || !seen.insert(g) {
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

    #[test]
    fn equal_functions_share_one_handle() {
        let mut m = Manager::new();
        let (a, b) = (m.var(0), m.var(1));
        let nb = m.not(b);
        let ab = m.and(a, b);
        let anb = m.and(a, nb);
        assert_eq!(m.or(ab, anb), a);
        let na = m.not(a);
        assert_eq!(m.and(a, na), Bdd::FALSE);
        assert_eq!(m.or(a, na), Bdd::TRUE);
        let x = m.xor(a, b);
        let e = m.iff(a, b);
        assert_eq!(m.not(x), e);
    }

    #[test]
    fn a_saved_mask_restored_by_or_implies_the_mask_it_saved() {
        let mut m = Manager::new();
        let (exec, c) = (m.var(0), m.var(1));
        let pushed = m.and(exec, c);
        let not_pushed = m.not(pushed);
        let other = m.and(exec, not_pushed);
        let restored = m.or(pushed, other);
        assert_eq!(restored, exec);
        assert!(m.implies(pushed, exec));
        assert!(!m.implies(exec, pushed));
    }

    #[test]
    fn quantifying_a_variable_removes_it_from_the_support() {
        let mut m = Manager::new();
        let (a, b, c) = (m.var(0), m.var(1), m.var(2));
        let ab = m.and(a, b);
        let f = m.or(ab, c);
        let e = m.exists(f, &|v| v == 1);
        assert_eq!(e, m.or(a, c));
        assert!(!m.support(e).contains(&1));
        let u = m.forall(f, &|v| v == 1);
        assert_eq!(u, c);
    }

    #[test]
    fn restricting_to_a_care_set_drops_the_tests_the_care_set_decides() {
        let mut m = Manager::new();
        let (mask, c, d) = (m.var(0), m.var(1), m.var(2));
        let nc = m.not(c);
        let taken = m.and(mask, c);
        let other = m.and(mask, nc);
        assert_eq!(m.restrict(taken, mask), c, "the two arms of a branch differ only in c");
        assert_eq!(m.restrict(other, mask), nc);
        let cd = m.and(c, d);
        let inner = m.and(mask, cd);
        let rest = m.not(inner);
        let care = m.and(mask, rest);
        let outer = m.and(mask, nc);
        let chosen = m.restrict(outer, care);
        let agreed = m.and(chosen, care);
        let expected = m.and(outer, care);
        assert_eq!(agreed, expected, "restrict agrees with the function on the care set");
        assert!(!m.support(chosen).contains(&0));
        assert_eq!(m.restrict(taken, Bdd::FALSE), taken);
    }

    #[test]
    fn composition_substitutes_a_formula_or_a_constant_for_a_variable() {
        let mut m = Manager::new();
        let (a, b, c) = (m.var(0), m.var(1), m.var(2));
        let f = m.xor(a, b);
        let bc = m.and(b, c);
        let g = m.compose(f, &|v| (v == 0).then_some(bc));
        let expected = m.xor(bc, b);
        assert_eq!(g, expected);
        let set = m.compose(f, &|v| (v == 0).then_some(Bdd::TRUE));
        let nb = m.not(b);
        assert_eq!(set, nb);
    }
}
