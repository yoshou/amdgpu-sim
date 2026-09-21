use super::{BlockId, Func};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Presence {

    Own,

    Packet,

    Wave,

    Workgroup,
}

pub(crate) struct Dominators {
    pub order: Vec<BlockId>,
    idom: BTreeMap<BlockId, BlockId>,
}

impl Dominators {
    pub fn of(f: &Func) -> Self {
        let order = reverse_postorder(f);
        let rank: BTreeMap<BlockId, usize> =
            order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); order.len()];
        for (&id, block) in &f.blocks {
            if !rank.contains_key(&id) {
                continue;
            }
            let u = rank[&id];
            for edge in block.term.edges() {
                if let Some(&v) = rank.get(&edge.dst) {
                    preds[v].push(u);
                }
            }
        }
        let mut idom: Vec<Option<usize>> = vec![None; order.len()];
        if !order.is_empty() {
            idom[0] = Some(0);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for v in 1..order.len() {
                let mut new: Option<usize> = None;
                for &u in &preds[v] {
                    if idom[u].is_none() {
                        continue;
                    }
                    new = Some(match new {
                        None => u,
                        Some(mut a) => {
                            let mut b = u;
                            while a != b {
                                while a > b {
                                    a = idom[a].expect("a reached block has a dominator");
                                }
                                while b > a {
                                    b = idom[b].expect("a reached block has a dominator");
                                }
                            }
                            a
                        }
                    });
                }
                if new.is_some() && new != idom[v] {
                    idom[v] = new;
                    changed = true;
                }
            }
        }
        Self {
            idom: (1..order.len())
                .filter_map(|v| idom[v].map(|d| (order[v], order[d])))
                .collect(),
            order,
        }
    }

    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        let mut at = b;
        loop {
            if at == a {
                return true;
            }
            match self.idom.get(&at) {
                Some(&d) => at = d,
                None => return false,
            }
        }
    }
}

pub(crate) fn reverse_postorder(f: &Func) -> Vec<BlockId> {
    let mut seen = BTreeSet::from([f.entry]);
    let mut order = Vec::new();
    let mut stack = vec![(f.entry, 0usize)];
    while let Some(&mut (id, ref mut next)) = stack.last_mut() {
        let edge = f.blocks[&id].term.edges().nth(*next).map(|e| e.dst);
        match edge {
            Some(dst) => {
                *next += 1;
                if seen.insert(dst) {
                    stack.push((dst, 0));
                }
            }
            None => {
                order.push(id);
                stack.pop();
            }
        }
    }
    order.reverse();
    order
}

impl Func {

    pub fn one_region(&mut self, presence: Presence) {
        self.regions = BTreeMap::from([(self.entry, presence)]);
    }

    pub fn lowered_to_packets(&mut self) {
        for present in self.regions.values_mut() {
            if *present == Presence::Own {
                *present = Presence::Packet;
            }
        }
    }

    pub fn presence_needed(&self) -> Presence {
        self.blocks
            .values()
            .flat_map(|b| b.insts.iter())
            .filter_map(super::verify::lanes_read)
            .max()
            .unwrap_or(Presence::Own)
    }

    pub fn reads_the_packet(&self) -> bool {
        self.blocks
            .values()
            .flat_map(|b| b.insts.iter())
            .any(|inst| super::verify::lanes_read(inst) == Some(Presence::Packet))
    }

    pub fn regions_of(&self, doms: &Dominators) -> BTreeMap<BlockId, BlockId> {
        let depth = |e: BlockId| self.regions.keys().filter(|&&o| doms.dominates(o, e)).count();
        let depths: BTreeMap<BlockId, usize> =
            self.regions.keys().map(|&e| (e, depth(e))).collect();
        let mut out = BTreeMap::new();
        for &b in &doms.order {
            let held = self
                .regions
                .keys()
                .filter(|&&e| doms.dominates(e, b))
                .max_by_key(|&&e| depths[&e]);
            if let Some(&e) = held {
                out.insert(b, e);
            }
        }
        out
    }

    pub fn encloses(&self, doms: &Dominators, outer: BlockId, inner: BlockId) -> bool {
        outer != inner && doms.dominates(outer, inner)
    }
}
