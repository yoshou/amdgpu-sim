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
    next: u32,
}

const PAGE: usize = 2 << 20;

#[cfg(target_os = "linux")]
fn advise<T>(v: &Vec<T>) {
    let bytes = v.capacity() * std::mem::size_of::<T>();
    let start = (v.as_ptr() as usize).next_multiple_of(PAGE);
    let end = (v.as_ptr() as usize + bytes) / PAGE * PAGE;
    if start < end {
        unsafe {
            libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn advise<T>(_: &Vec<T>) {}

fn paged<T>(capacity: usize) -> Vec<T> {
    let v = Vec::with_capacity(capacity);
    advise(&v);
    v
}

fn reserve<T: Copy>(v: &mut Vec<T>, len: usize) {
    if len > v.capacity() {
        let mut grown = paged(len.max(2 * v.capacity()));
        grown.extend_from_slice(v);
        *v = grown;
    }
}

fn place(var: u32, low: Bdd, high: Bdd) -> usize {
    let key = (low.0 as u64 | (high.0 as u64) << 32) ^ (var as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let h = key.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    (h ^ h >> 31) as usize
}

const TERMINAL: u32 = u32::MAX;

const ITE_SLOTS: usize = 1 << 16;

pub struct Manager {
    nodes: Vec<Node>,
    heads: Vec<u32>,

    ite: Vec<(Bdd, Bdd, Bdd, Bdd)>,
    restrict: HashMap<(Bdd, Bdd), Bdd>,
    implications: HashMap<(Bdd, Bdd), bool>,
    marks: Vec<(u32, Bdd)>,
    generation: u32,
    pairs: HashMap<(Bdd, Bdd), Bdd>,
}

impl Manager {
    pub fn new() -> Self {
        let terminal = |value| Node {
            var: TERMINAL,
            low: Bdd(value),
            high: Bdd(value),
            next: 0,
        };
        Self {
            nodes: vec![terminal(0), terminal(1)],
            heads: vec![0; 1 << 10],
            ite: Vec::new(),
            restrict: HashMap::default(),
            implications: HashMap::default(),
            marks: Vec::new(),
            generation: 0,
            pairs: HashMap::default(),
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
        let slot = place(var, low, high) & (self.heads.len() - 1);
        let mut id = self.heads[slot];
        while id != 0 {
            let node = self.nodes[id as usize];
            if node.var == var && node.low == low && node.high == high {
                return Bdd(id);
            }
            id = node.next;
        }
        let id = self.nodes.len() as u32;
        reserve(&mut self.nodes, id as usize + 1);
        self.nodes.push(Node {
            var,
            low,
            high,
            next: self.heads[slot],
        });
        self.heads[slot] = id;
        if self.nodes.len() > self.heads.len() {
            self.rehash();
        }
        Bdd(id)
    }

    fn rehash(&mut self) {
        let size = 2 * self.heads.len();
        let mut heads = paged(size);
        heads.resize(size, 0);
        for id in 2..self.nodes.len() {
            let node = self.nodes[id];
            let slot = place(node.var, node.low, node.high) & (size - 1);
            self.nodes[id].next = heads[slot];
            heads[slot] = id as u32;
        }
        self.heads = heads;
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
            self.ite = paged(slots);
            self.ite.resize(slots, (Bdd::FALSE, Bdd::FALSE, Bdd::FALSE, Bdd::FALSE));
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
        let r = if (low, high) == (g0, g1) {
            g
        } else if (low, high) == (h0, h1) {
            h
        } else if (low, high) == (f0, f1) {
            f
        } else {
            self.node(var, low, high)
        };

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

    fn generation(&mut self) -> u32 {
        if self.marks.len() < self.nodes.len() {
            reserve(&mut self.marks, self.nodes.len());
            self.marks.resize(self.nodes.len(), (0, Bdd::FALSE));
        }
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.marks.fill((0, Bdd::FALSE));
            self.generation = 1;
        }
        self.generation
    }

    fn quantify(&mut self, f: Bdd, chosen: &dyn Fn(u32) -> bool, any: bool) -> Bdd {
        let generation = self.generation();
        self.quantify_memo(f, chosen, any, generation)
    }

    fn quantify_memo(&mut self, f: Bdd, chosen: &dyn Fn(u32) -> bool, any: bool, generation: u32) -> Bdd {
        if f.constant().is_some() {
            return f;
        }
        let (seen, r) = self.marks[f.0 as usize];
        if seen == generation {
            return r;
        }
        let node = self.nodes[f.0 as usize];
        let low = self.quantify_memo(node.low, chosen, any, generation);
        let high = self.quantify_memo(node.high, chosen, any, generation);
        let r = if chosen(node.var) {
            if any {
                self.or(low, high)
            } else {
                self.and(low, high)
            }
        } else if (low, high) == (node.low, node.high) {
            f
        } else {
            self.node(node.var, low, high)
        };
        self.marks[f.0 as usize] = (generation, r);
        r
    }

    pub fn and_exists(&mut self, f: Bdd, g: Bdd, chosen: &dyn Fn(u32) -> bool) -> Bdd {
        let mut memo = std::mem::take(&mut self.pairs);
        if memo.capacity() > 4 * memo.len().max(1 << 10) {
            memo = HashMap::with_capacity_and_hasher(memo.len().max(1 << 10), Default::default());
        } else {
            memo.clear();
        }
        let r = self.and_exists_memo(f, g, chosen, &mut memo);
        self.pairs = memo;
        r
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
            if (low, high) == (f0, f1) {
                f
            } else if (low, high) == (g0, g1) {
                g
            } else {
                self.node(var, low, high)
            }
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
        let generation = self.generation();
        self.compose_memo(f, map, u32::MAX, generation)
    }

    pub fn compose_many(&mut self, fs: &[Bdd], map: &dyn Fn(u32) -> Option<Bdd>, last: u32) -> Vec<Bdd> {
        let generation = self.generation();
        fs.iter().map(|&f| self.compose_memo(f, map, last, generation)).collect()
    }

    fn compose_memo(&mut self, f: Bdd, map: &dyn Fn(u32) -> Option<Bdd>, last: u32, generation: u32) -> Bdd {
        if f.constant().is_some() || self.top(f) > last {
            return f;
        }
        let (seen, r) = self.marks[f.0 as usize];
        if seen == generation {
            return r;
        }
        let node = self.nodes[f.0 as usize];
        let low = self.compose_memo(node.low, map, last, generation);
        let high = self.compose_memo(node.high, map, last, generation);
        let r = match map(node.var) {
            Some(v) => {
                let n = self.nodes[v.0 as usize];
                if n.low == Bdd::FALSE && n.high == Bdd::TRUE && n.var < self.top(low) && n.var < self.top(high) {
                    self.node(n.var, low, high)
                } else {
                    self.ite(v, high, low)
                }
            }
            None if (low, high) == (node.low, node.high) => f,
            None if node.var < self.top(low) && node.var < self.top(high) => self.node(node.var, low, high),
            None => {
                let v = self.var(node.var);
                self.ite(v, high, low)
            }
        };
        self.marks[f.0 as usize] = (generation, r);
        r
    }

    pub fn size(&mut self, f: Bdd) -> usize {
        let generation = self.generation();
        let mut count = 0;
        let mut stack = vec![f];
        while let Some(g) = stack.pop() {
            if g.constant().is_some() || self.marks[g.0 as usize].0 == generation {
                continue;
            }
            self.marks[g.0 as usize].0 = generation;
            count += 1;
            let node = self.nodes[g.0 as usize];
            stack.push(node.low);
            stack.push(node.high);
        }
        count
    }

    pub fn support(&mut self, f: Bdd) -> Vec<u32> {
        let generation = self.generation();
        let mut out = Vec::new();
        let mut recent = [TERMINAL; 256];
        let mut stack = vec![f];
        while let Some(g) = stack.pop() {
            if g.constant().is_some() || self.marks[g.0 as usize].0 == generation {
                continue;
            }
            self.marks[g.0 as usize].0 = generation;
            let node = self.nodes[g.0 as usize];
            let slot = (node.var.wrapping_mul(0x9e37_79b9) >> 24) as usize;
            if recent[slot] != node.var {
                recent[slot] = node.var;
                out.push(node.var);
            }
            stack.push(node.low);
            stack.push(node.high);
        }
        out.sort_unstable();
        out.dedup();
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
    fn support_lists_exactly_the_variables_a_function_reads_in_order() {
        let mut state = 0x2545_f491_4f6c_dd1d;
        let mut m = Manager::new();
        let vars = 10;
        for _ in 0..400 {
            let f = random_function(&mut m, &mut state, vars);
            let want: Vec<u32> = (0..vars).filter(|&v| m.cofactor(f, v, false) != m.cofactor(f, v, true)).collect();
            assert_eq!(m.support(f), want, "the support must be the sorted variables whose cofactors differ");
            let g = random_function(&mut m, &mut state, vars);
            let both = m.and(f, g);
            let mut joint: Vec<u32> = m.support(f).into_iter().chain(m.support(g)).collect();
            joint.sort_unstable();
            joint.dedup();
            assert!(m.support(both).iter().all(|v| joint.contains(v)));
            assert_eq!(m.support(f), want, "asking again must give the same support");
        }
    }

    #[test]
    fn support_keeps_each_of_many_scattered_variables_once() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        let mut m = Manager::new();
        let vars: Vec<u32> = (0..700u32).map(|i| (1 << 30) | (i << 7) | (i % 5)).collect();
        for round in 0..40 {
            let mut f = Bdd::FALSE;
            let mut picked = Vec::new();
            for _ in 0..40 + round * 15 {
                let v = vars[random(&mut state) as usize % vars.len()];
                let x = m.var(v);
                f = m.xor(f, x);
                picked.push(v);
            }
            picked.sort_unstable();
            picked.dedup();
            let want: Vec<u32> = picked.into_iter().filter(|&v| m.cofactor(f, v, false) != m.cofactor(f, v, true)).collect();
            assert!(want.len() > 20 + round * 5);
            assert_eq!(m.support(f), want, "the support must list every variable whose cofactors differ once, in order");
        }
    }

    #[test]
    fn ite_follows_its_operands_row_by_row() {
        let mut state = 0x2545_f491_4f6c_dd1d;
        let mut m = Manager::new();
        let vars = 8;
        for _ in 0..500 {
            let f = random_function(&mut m, &mut state, vars);
            let g = random_function(&mut m, &mut state, vars);
            let h = if random(&mut state) % 4 == 0 { m.and(g, f) } else { random_function(&mut m, &mut state, vars) };
            let r = m.ite(f, g, h);
            for row in 0..1u32 << vars {
                let value = |v: u32| row >> v & 1 == 1;
                let want = if evaluate(&m, f, &value) { evaluate(&m, g, &value) } else { evaluate(&m, h, &value) };
                assert_eq!(evaluate(&m, r, &value), want, "ite must take g where f holds and h elsewhere");
            }
        }
    }

    #[test]
    fn and_exists_conjoins_before_quantifying() {
        let mut state = 0x9e37_79b9_7f4a_7c15;
        let mut m = Manager::new();
        let vars = 8;
        for _ in 0..500 {
            let f = random_function(&mut m, &mut state, vars);
            let g = if random(&mut state) % 3 == 0 { m.or(f, Bdd::FALSE) } else { random_function(&mut m, &mut state, vars) };
            let chosen: Vec<u32> = (0..vars).filter(|_| random(&mut state) % 3 == 0).collect();
            let r = m.and_exists(f, g, &|v| chosen.contains(&v));
            let both = m.and(f, g);
            assert_eq!(r, m.exists(both, &|v| chosen.contains(&v)), "quantifying {:?} out of the conjunction", chosen);
            for row in 0..1u32 << vars {
                let want = (0..1u32 << chosen.len()).any(|pick| {
                    let value = |v: u32| match chosen.iter().position(|&c| c == v) {
                        Some(i) => pick >> i & 1 == 1,
                        None => row >> v & 1 == 1,
                    };
                    evaluate(&m, f, &value) && evaluate(&m, g, &value)
                });
                assert_eq!(evaluate(&m, r, &|v| row >> v & 1 == 1), want);
            }
        }
    }

    #[test]
    fn equal_functions_keep_one_node_while_the_table_grows() {
        let mut state = 0x2545_f491_4f6c_dd1d;
        let mut m = Manager::new();
        let vars = 9;
        let mut built: Vec<(Bdd, Vec<bool>)> = Vec::new();
        let rows = |m: &Manager, f: Bdd| -> Vec<bool> { (0..1u32 << vars).map(|row| evaluate(m, f, &|v| row >> v & 1 == 1)).collect() };
        let start = m.heads.len();
        while m.nodes.len() < 40 * start {
            let f = random_function(&mut m, &mut state, vars);
            let table = rows(&m, f);
            built.push((f, table));
        }
        assert!(m.heads.len() >= 32 * start, "the table must have grown several times");
        let mut first: HashMap<Vec<bool>, Bdd> = HashMap::default();
        for (f, table) in &built {
            assert_eq!(*first.entry(table.clone()).or_insert(*f), *f, "equal functions must be one node");
        }
        let count = m.nodes.len();
        let mut again = 0x2545_f491_4f6c_dd1d;
        for (f, _) in &built {
            assert_eq!(random_function(&mut m, &mut again, vars), *f, "building a function again must find its node");
        }
        assert_eq!(m.nodes.len(), count, "building functions again must create no node");
        for id in 2..m.nodes.len() {
            let node = m.nodes[id];
            assert_eq!(m.node(node.var, node.low, node.high), Bdd(id as u32), "every node must be found by its own fields");
        }
    }

    #[test]
    fn reserve_keeps_contents_and_room() {
        for len in [0usize, 1, 1000, 3 << 20] {
            let mut v: Vec<u32> = (0..len as u32).collect();
            v.shrink_to_fit();
            let full = v.capacity();
            reserve(&mut v, len + 1);
            assert!(v.capacity() > len && v.capacity() >= 2 * full, "growing must at least double the room");
            assert!(v.iter().copied().eq(0..len as u32), "growing must keep every element");
            let before = v.capacity();
            reserve(&mut v, len);
            assert_eq!(v.capacity(), before, "room already there must not move the elements");
            let w: Vec<u64> = paged(len);
            assert!(w.capacity() >= len && w.is_empty());
        }
    }

    #[test]
    fn advice_leaves_every_element_in_place() {
        for len in [0usize, 10, 1 << 20, 5 << 20] {
            let v: Vec<u64> = (0..len as u64).map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15)).collect();
            advise(&v);
            assert!(v.iter().enumerate().all(|(i, &x)| x == (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)));
        }
    }

    #[test]
    fn size_counts_every_node_a_function_reaches_once() {
        let mut state = 0x2545_f491_4f6c_dd1d;
        let mut m = Manager::new();
        for _ in 0..300 {
            let f = random_function(&mut m, &mut state, 8);
            let mut seen = std::collections::BTreeSet::new();
            let mut stack = vec![f];
            while let Some(g) = stack.pop() {
                if let Some((_, low, high)) = m.decompose(g) {
                    if seen.insert(g) {
                        stack.push(low);
                        stack.push(high);
                    }
                }
            }
            assert_eq!(m.size(f), seen.len());
            assert_eq!(m.size(f), seen.len(), "counting again must give the same size");
        }
    }

    #[test]
    fn memo_generations_wrap_without_reading_old_marks() {
        let mut m = Manager::new();
        let (x, y, z) = (m.var(0), m.var(1), m.var(2));
        let xy = m.and(x, y);
        let f = m.or(xy, z);
        let g = {
            let (u, w) = (m.var(5), m.var(6));
            m.xor(u, w)
        };
        assert_eq!(m.support(f), vec![0, 1, 2]);
        m.generation = u32::MAX - 1;
        assert_eq!(m.support(g), vec![5, 6]);
        assert_eq!(m.support(g), vec![5, 6]);
        assert_eq!(m.support(f), vec![0, 1, 2], "a generation that wraps around must not see the marks of earlier ones");
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
