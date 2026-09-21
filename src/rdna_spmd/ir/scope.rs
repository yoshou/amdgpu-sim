use super::{BlockId, EffectOp, Func, Inst};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Presence {

    Own,

    Packet,

    Wave,

    Workgroup,
}

pub struct Dominators {
    order: Vec<BlockId>,
    idom: BTreeMap<BlockId, BlockId>,
}

impl Dominators {
    pub fn of(f: &Func) -> Self {
        let order = f.reverse_postorder();
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

    pub fn order(&self) -> &[BlockId] {
        &self.order
    }

    pub fn parent(&self, b: BlockId) -> Option<BlockId> {
        self.idom.get(&b).copied()
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

impl Inst {
    pub fn lanes_read(&self) -> Option<Presence> {
        match self {
            Inst::Packet { .. } => Some(Presence::Packet),
            Inst::Effect {
                op: EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait,
                ..
            } => Some(Presence::Workgroup),
            Inst::Effect {
                op: EffectOp::Wave(_),
                ..
            } => Some(Presence::Wave),
            _ => None,
        }
    }
}

impl Func {

    pub fn one_region(&mut self, presence: Presence) {
        self.regions = BTreeMap::from([(self.entry, presence)]);
    }

    pub fn enter_regions(&mut self) {
        let scope = |present: Presence| match present {
            Presence::Own | Presence::Packet => 0,
            Presence::Wave => 1,
            Presence::Workgroup => 2,
        };
        let doms = Dominators::of(self);
        let mut needed: BTreeMap<BlockId, Presence> = doms
            .order
            .iter()
            .map(|&id| {
                let own = self.blocks[&id]
                    .insts
                    .iter()
                    .filter_map(Inst::lanes_read)
                    .max()
                    .unwrap_or(Presence::Own);
                (id, own)
            })
            .collect();
        for &id in doms.order.iter().rev() {
            if let Some(parent) = doms.parent(id) {
                let below = needed[&id];
                let above = needed.get_mut(&parent).unwrap();
                *above = (*above).max(below);
            }
        }
        let mut held: BTreeMap<BlockId, Presence> = BTreeMap::new();
        let mut regions = BTreeMap::new();
        for &id in &doms.order {
            let present = match doms.parent(id) {
                None => needed[&id],
                Some(parent) if scope(needed[&id]) < scope(held[&parent]) => needed[&id],
                Some(parent) => {
                    held.insert(id, held[&parent]);
                    continue;
                }
            };
            regions.insert(id, present);
            held.insert(id, present);
        }
        self.regions = regions;
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
            .filter_map(Inst::lanes_read)
            .max()
            .unwrap_or(Presence::Own)
    }

    pub fn reads_the_packet(&self) -> bool {
        self.blocks
            .values()
            .flat_map(|b| b.insts.iter())
            .any(|inst| inst.lanes_read() == Some(Presence::Packet))
    }

    pub fn regions_of(&self, doms: &Dominators) -> BTreeMap<BlockId, BlockId> {
        let depth = |e: BlockId| self.regions.keys().filter(|&&o| doms.dominates(o, e)).count();
        let depths: BTreeMap<BlockId, usize> =
            self.regions.keys().map(|&e| (e, depth(e))).collect();
        let mut out = BTreeMap::new();
        for &b in doms.order() {
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
