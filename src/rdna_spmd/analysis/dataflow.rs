use super::super::ir::{ValueId, *};

pub(super) trait Lattice: Clone + PartialEq {
    fn meet(&self, other: &Self) -> Self;
}

impl Lattice for bool {
    fn meet(&self, other: &Self) -> Self {
        *self && *other
    }
}

pub(super) struct Cfg<'f> {
    pub blocks: Vec<&'f Block>,
    pub ids: Vec<BlockId>,
    pub index: Vec<usize>,
    pub order: Vec<usize>,
    pub incoming: Vec<Vec<(usize, &'f Edge)>>,
    pub entry: usize,
}

impl<'f> Cfg<'f> {
    pub fn new(f: &'f Func) -> Self {
        let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
        let blocks: Vec<&Block> = f.blocks.values().collect();
        let mut index = vec![usize::MAX; ids.iter().map(|b| b.0 + 1).max().unwrap_or(0)];
        for (at, id) in ids.iter().enumerate() {
            index[id.0] = at;
        }
        let mut incoming: Vec<Vec<(usize, &Edge)>> =
            (0..blocks.len()).map(|_| Vec::new()).collect();
        for (at, block) in blocks.iter().enumerate() {
            for edge in block.term.edges() {
                incoming[index[edge.dst.0]].push((at, edge));
            }
        }
        let entry = index[f.entry.0];
        let mut order = Vec::with_capacity(blocks.len());
        let mut seen = vec![false; blocks.len()];
        seen[entry] = true;
        let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
        while let Some((at, next)) = stack.last_mut() {
            match blocks[*at].term.edges().nth(*next) {
                Some(edge) => {
                    *next += 1;
                    let dst = index[edge.dst.0];
                    if !seen[dst] {
                        seen[dst] = true;
                        stack.push((dst, 0));
                    }
                }
                None => {
                    order.push(*at);
                    stack.pop();
                }
            }
        }
        order.reverse();
        for at in 0..blocks.len() {
            if !seen[at] {
                order.push(at);
            }
        }
        Cfg {
            blocks,
            ids,
            index,
            order,
            incoming,
            entry,
        }
    }
}

const ROUND_LIMIT: usize = 1 << 16;

fn descend<L: Lattice>(facts: &mut [L], v: ValueId, next: L, changed: &mut bool) {
    let next = facts[v.0].meet(&next);
    if facts[v.0] != next {
        facts[v.0] = next;
        *changed = true;
    }
}

pub(super) struct Sparse<'a, 'f, L: Lattice> {
    pub cfg: &'a Cfg<'f>,
    pub start: L,
    pub boundary: &'a dyn Fn(ValueId) -> L,
    pub edge: &'a dyn Fn(&Edge, usize, usize, &[L]) -> L,
    pub transfer: &'a dyn Fn(&Inst, ValueId, &[L]) -> L,
}

impl<'a, 'f, L: Lattice> Sparse<'a, 'f, L> {
    pub fn solve(&self, values: usize) -> Vec<L> {
        let cfg = self.cfg;
        let mut facts = vec![self.start.clone(); values];
        for round in 0.. {
            assert!(round < ROUND_LIMIT, "dataflow solver did not converge");
            let mut changed = false;
            for &at in &cfg.order {
                let block = cfg.blocks[at];
                for (position, &(param, _)) in block.params.iter().enumerate() {
                    let mut fact = if at == cfg.entry {
                        Some((self.boundary)(param))
                    } else {
                        None
                    };
                    for &(src, edge) in &cfg.incoming[at] {
                        let along = (self.edge)(edge, src, position, &facts);
                        fact = Some(match fact {
                            Some(current) => current.meet(&along),
                            None => along,
                        });
                    }
                    if let Some(fact) = fact {
                        descend(&mut facts, param, fact, &mut changed);
                    }
                }
                for inst in &block.insts {
                    let mut outputs = Vec::new();
                    for_each_output(inst, |v| outputs.push(v));
                    for v in outputs {
                        let fact = (self.transfer)(inst, v, &facts);
                        descend(&mut facts, v, fact, &mut changed);
                    }
                }
            }
            if !changed {
                break;
            }
        }
        facts
    }
}
pub(super) fn for_each_output(inst: &Inst, mut f: impl FnMut(ValueId)) {
    match inst {
        Inst::Core { value, .. } | Inst::Packet { output: value, .. } => f(*value),
        Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => {
            for o in outputs {
                f(o.0);
            }
        }
    }
}
