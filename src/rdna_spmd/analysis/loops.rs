use super::facts::Facts;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeMap;

pub(crate) struct Loops {
    innermost: Vec<Option<usize>>,
    parent: Vec<Option<usize>>,
    header: Vec<usize>,
}

impl Loops {
    pub fn new(f: &Func, facts: &Facts) -> Result<Self, BlockId> {
        let order = &facts.order;
        let n = order.len();
        let rank: BTreeMap<BlockId, usize> = order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
        let successors: Vec<Vec<usize>> = order
            .iter()
            .map(|b| f.blocks[b].term.edges().map(|e| rank[&e.dst]).collect())
            .collect();
        let mut predecessors = vec![Vec::new(); n];
        for (u, succ) in successors.iter().enumerate() {
            for &v in succ {
                predecessors[v].push(u);
            }
        }
        let mut idom: Vec<Option<usize>> = vec![None; n];
        idom[0] = Some(0);
        let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| {
            while a != b {
                while a > b {
                    a = idom[a].unwrap();
                }
                while b > a {
                    b = idom[b].unwrap();
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for v in 1..n {
                let mut new: Option<usize> = None;
                for &p in &predecessors[v] {
                    if idom[p].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => p,
                        Some(q) => intersect(&idom, p, q),
                    });
                }
                if new.is_some() && new != idom[v] {
                    idom[v] = new;
                    changed = true;
                }
            }
        }
        let dominates = |a: usize, mut b: usize| loop {
            if a == b {
                return true;
            }
            if b == 0 {
                return false;
            }
            b = idom[b].unwrap();
        };
        let mut bodies: BTreeMap<usize, Vec<bool>> = BTreeMap::new();
        for u in 0..n {
            for &h in &successors[u] {
                if h > u {
                    continue;
                }
                if !dominates(h, u) {
                    return Err(order[h]);
                }
                let body = bodies.entry(h).or_insert_with(|| {
                    let mut b = vec![false; n];
                    b[h] = true;
                    b
                });
                let mut stack = vec![u];
                while let Some(x) = stack.pop() {
                    if body[x] {
                        continue;
                    }
                    body[x] = true;
                    stack.extend(predecessors[x].iter().copied());
                }
            }
        }
        let loops: Vec<(usize, Vec<bool>)> = bodies.into_iter().collect();
        let size = |body: &[bool]| body.iter().filter(|&&x| x).count();
        let header: Vec<usize> = loops.iter().map(|l| l.0).collect();
        let smallest_containing = |x: usize, except: Option<usize>| {
            loops
                .iter()
                .enumerate()
                .filter(|&(i, l)| Some(i) != except && l.1[x])
                .min_by_key(|(_, l)| size(&l.1))
                .map(|(i, _)| i)
        };
        let parent = (0..loops.len())
            .map(|i| smallest_containing(header[i], Some(i)))
            .collect();
        let innermost = (0..n).map(|x| smallest_containing(x, None)).collect();
        Ok(Self {
            innermost,
            parent,
            header,
        })
    }

    pub fn count(&self) -> usize {
        self.header.len()
    }

    pub fn header(&self, l: usize) -> usize {
        self.header[l]
    }

    pub fn parent(&self, l: usize) -> Option<usize> {
        self.parent[l]
    }

    pub fn innermost(&self, rank: usize) -> Option<usize> {
        self.innermost[rank]
    }

    pub fn contains(&self, l: usize, rank: usize) -> bool {
        self.chain(rank).contains(&Some(l))
    }

    fn chain(&self, x: usize) -> Vec<Option<usize>> {
        let mut out = vec![self.innermost[x]];
        while let Some(l) = out.last().copied().flatten() {
            out.push(self.parent[l]);
        }
        out
    }

    pub fn before(&self, a: usize, b: usize) -> bool {
        let (ca, cb) = (self.chain(a), self.chain(b));
        let common = ca.iter().find(|l| cb.contains(l)).copied().flatten();
        let representative = |x: usize, chain: &[Option<usize>]| {
            let position = chain.iter().position(|&l| l == common).unwrap();
            if position == 0 {
                x
            } else {
                self.header[chain[position - 1].unwrap()]
            }
        };
        let (ra, rb) = (representative(a, &ca), representative(b, &cb));
        ra < rb || (ra == rb && a < b)
    }
}
