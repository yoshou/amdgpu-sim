use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Unit {
    Block(BlockId),
    Loop(usize),
}

pub struct Region {
    pub units: Vec<Unit>,

    pub children: Vec<Vec<usize>>,

    pub span: Vec<Vec<BlockId>>,
}

pub struct Structure {
    pub facts: Facts,
    pub loops: Loops,
    pub rank: BTreeMap<BlockId, usize>,

    pub regions: Vec<Region>,

    pub live: Vec<bool>,

    pub same: Vec<Option<ValueId>>,
}

impl Structure {
    pub fn new(f: &Func, facts: Facts) -> Self {
        let loops = Loops::new(f, &facts).expect("a lane program is reducible");
        let rank: BTreeMap<BlockId, usize> = facts
            .order
            .iter()
            .enumerate()
            .map(|(r, &b)| (b, r))
            .collect();
        let live = live(f, &facts);
        let mut s = Structure {
            facts,
            loops,
            rank,
            regions: Vec::new(),
            live,
            same: Vec::new(),
        };
        s.same = s.same_values(f);
        s.regions = (0..=s.loops.count())
            .map(|r| s.region(f, r.checked_sub(1)))
            .collect();
        s
    }

    pub fn within(&self, l: Option<usize>, block: BlockId) -> bool {
        l.map_or(true, |l| self.loops.contains(l, self.rank[&block]))
    }

    pub fn header(&self, l: usize) -> BlockId {
        self.facts.order[self.loops.header(l)]
    }

    fn unit(&self, l: Option<usize>, block: BlockId) -> Option<Unit> {
        if !self.within(l, block) {
            return None;
        }
        let mut inner = self.loops.innermost(self.rank[&block]);
        if inner == l {
            return Some(Unit::Block(block));
        }
        while let Some(i) = inner {
            if self.loops.parent(i) == l {
                return Some(Unit::Loop(i));
            }
            inner = self.loops.parent(i);
        }
        unreachable!("a block inside a region is in it or in one of its loops")
    }

    pub fn resolve(&self, mut v: ValueId) -> ValueId {
        while let Some(next) = self.same[v.0] {
            v = next;
        }
        v
    }

    pub fn defined_in(&self, v: ValueId) -> Option<BlockId> {
        match self.facts.site[v.0] {
            Site::Param { block, .. } | Site::Inst { block, .. } => Some(block),
            Site::Unreached => None,
        }
    }

    fn same_values(&self, f: &Func) -> Vec<Option<ValueId>> {
        let mut same: Vec<Option<ValueId>> = vec![None; f.types.len()];
        let find = |same: &[Option<ValueId>], mut v: ValueId| {
            while let Some(next) = same[v.0] {
                v = next;
            }
            v
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &id in &self.facts.order {
                if id == f.entry {
                    continue;
                }
                let r = self.rank[&id];
                for (index, &(p, _)) in f.blocks[&id].params.iter().enumerate() {
                    if same[p.0].is_some() || !self.live[p.0] {
                        continue;
                    }
                    let mut unique = None;
                    let mut forwards = true;
                    for arg in self.facts.arguments(f, id, index) {
                        let a = find(&same, arg);
                        if a == p {
                            continue;
                        }
                        match unique {
                            None => unique = Some(a),
                            Some(u) if u == a => {}
                            Some(_) => forwards = false,
                        }
                    }
                    let Some(value) = unique.filter(|_| forwards) else {
                        continue;
                    };
                    let Some(def) = self.defined_in(value) else {
                        continue;
                    };
                    let inside = (0..self.loops.count())
                        .filter(|&l| self.loops.contains(l, self.rank[&def]))
                        .all(|l| self.loops.contains(l, r));
                    if inside {
                        same[p.0] = Some(value);
                        changed = true;
                    }
                }
            }
        }
        same
    }

    fn region(&self, f: &Func, l: Option<usize>) -> Region {
        let head = match l {
            Some(l) => Unit::Block(self.header(l)),
            None => self.unit(None, f.entry).unwrap(),
        };
        let mut units = vec![head];
        for (r, &b) in self.facts.order.iter().enumerate() {
            if self.loops.innermost(r) == l && Unit::Block(b) != head {
                units.push(Unit::Block(b));
            }
        }
        for c in 0..self.loops.count() {
            if self.loops.parent(c) == l && Unit::Loop(c) != head {
                units.push(Unit::Loop(c));
            }
        }
        let index: BTreeMap<Unit, usize> = units.iter().enumerate().map(|(i, &u)| (u, i)).collect();
        let n = units.len();
        let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut own: Vec<Vec<BlockId>> = Vec::with_capacity(n);
        for (i, &u) in units.iter().enumerate() {
            let blocks: Vec<BlockId> = match u {
                Unit::Block(b) => vec![b],
                Unit::Loop(c) => self
                    .facts
                    .order
                    .iter()
                    .enumerate()
                    .filter(|&(r, _)| self.loops.contains(c, r))
                    .map(|(_, &b)| b)
                    .collect(),
            };
            own.push(blocks.clone());
            for &b in &blocks {
                for e in f.blocks[&b].term.edges() {
                    if let Unit::Loop(c) = u {
                        if self.loops.contains(c, self.rank[&e.dst]) {
                            continue;
                        }
                    }
                    if l.is_some() && Unit::Block(e.dst) == head {
                        continue;
                    }
                    if let Some(target) = self.unit(l, e.dst) {
                        if target != head {
                            successors[i].push(index[&target]);
                        }
                    }
                }
            }
        }
        let mut post = Vec::with_capacity(n);
        let mut seen = vec![false; n];
        let mut stack = vec![(0usize, 0usize)];
        seen[0] = true;
        while let Some((u, next)) = stack.last_mut() {
            if let Some(&v) = successors[*u].get(*next) {
                *next += 1;
                if !seen[v] {
                    seen[v] = true;
                    stack.push((v, 0));
                }
            } else {
                post.push(*u);
                stack.pop();
            }
        }
        let rpo: Vec<usize> = post.into_iter().rev().collect();
        let mut position = vec![usize::MAX; n];
        for (p, &u) in rpo.iter().enumerate() {
            position[u] = p;
        }
        let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (u, succ) in successors.iter().enumerate() {
            for &v in succ {
                predecessors[v].push(u);
            }
        }
        let mut idom: Vec<Option<usize>> = vec![None; n];
        idom[0] = Some(0);
        let mut changed = true;
        while changed {
            changed = false;
            for &v in rpo.iter().skip(1) {
                let mut new: Option<usize> = None;
                for &p in &predecessors[v] {
                    if idom[p].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(q) => intersect(&idom, &position, p, q),
                    });
                }
                if new.is_some() && new != idom[v] {
                    idom[v] = new;
                    changed = true;
                }
            }
        }
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &v in rpo.iter().skip(1) {
            children[idom[v].unwrap()].push(v);
        }
        let mut span = own;
        for &v in rpo.iter().skip(1).rev() {
            let parent = idom[v].unwrap();
            let below = span[v].clone();
            span[parent].extend(below);
        }
        let mut order = Vec::with_capacity(n);
        let mut stack = vec![0usize];
        while let Some(u) = stack.pop() {
            order.push(u);
            stack.extend(children[u].iter().rev().copied());
        }
        let renumber: BTreeMap<usize, usize> = order
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new))
            .collect();
        Region {
            units: order.iter().map(|&u| units[u]).collect(),
            children: order
                .iter()
                .map(|&u| children[u].iter().map(|c| renumber[c]).collect())
                .collect(),
            span: order.iter().map(|&u| span[u].clone()).collect(),
        }
    }
}

fn intersect(idom: &[Option<usize>], position: &[usize], mut a: usize, mut b: usize) -> usize {
    while a != b {
        while position[a] > position[b] {
            a = idom[a].unwrap();
        }
        while position[b] > position[a] {
            b = idom[b].unwrap();
        }
    }
    a
}

fn live(f: &Func, facts: &Facts) -> Vec<bool> {
    let mut used = vec![false; f.types.len()];
    for block in f.blocks.values() {
        for inst in &block.insts {
            for v in inst.operands() {
                used[v.0] = true;
            }
        }
        match &block.term {
            Term::CondBr { cond, .. } => used[cond.0] = true,
            Term::Ret(args) => {
                for v in args {
                    used[v.0] = true;
                }
            }
            Term::Br(_) => {}
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for &id in &facts.order {
            for e in f.blocks[&id].term.edges() {
                for (&arg, &(param, _)) in e.args.iter().zip(&f.blocks[&e.dst].params) {
                    if used[param.0] && !used[arg.0] {
                        used[arg.0] = true;
                        changed = true;
                    }
                }
            }
        }
    }
    used
}
