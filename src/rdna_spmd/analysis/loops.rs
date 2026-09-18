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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::ir::parse;
    use crate::rdna_spmd::program::{Parameter, ParameterSource};

    fn loops(text: &str) -> (Func, Facts, Result<Loops, BlockId>) {
        let registry = crate::rdna_spmd::targets::rdna4::registry();
        let f = parse::func(&registry, text).unwrap();
        let inputs = vec![Parameter {
            source: ParameterSource::MaskBit(126),
            ty: Ty::I1,
        }];
        let facts = Facts::new(&f, &inputs, &Default::default());
        let loops = Loops::new(&f, &facts);
        (f, facts, loops)
    }

    #[test]
    fn a_block_after_a_nested_loop_comes_after_every_block_of_that_loop() {
        let (_, facts, loops) = loops(
            "func entry b0
             b0(v0: i1):
               br b1(v0)
             b1(v1: i1):
               br b2(v1)
             b2(v2: i1):
               br b3(v2)
             b3(v3: i1):
               condbr v3, b2(v3), b4(v3)
             b4(v4: i1):
               condbr v4, b1(v4), b5(v4)
             b5(v5: i1):
               ret",
        );
        let loops = loops.unwrap();
        let r = |b: usize| facts.order.iter().position(|&x| x == BlockId(b)).unwrap();
        assert!(loops.before(r(3), r(4)), "the inner latch precedes the outer latch");
        assert!(loops.before(r(2), r(4)));
        assert!(loops.before(r(4), r(5)), "the outer latch precedes the exit");
        assert!(!loops.before(r(4), r(3)));
    }

    #[test]
    fn an_edge_into_the_middle_of_a_cycle_is_irreducible() {
        let (_, _, loops) = loops(
            "func entry b0
             b0(v0: i1):
               condbr v0, b1(v0), b2(v0)
             b1(v1: i1):
               br b2(v1)
             b2(v2: i1):
               condbr v2, b1(v2), b3(v2)
             b3(v3: i1):
               ret",
        );
        assert!(loops.is_err());
    }
}
