//! Which lanes are present in a region of the function.
//!
//! A region's [`Presence`] is the set of lanes guaranteed to be at the region's
//! entry at the same time. `Own` promises nothing about the other lanes, so
//! nothing in an `Own` region may read them; `Packet` promises the lanes the
//! lowering runs in lockstep, which is what a query over the packet needs;
//! `Wave` promises every lane of the wave, the mask saying which of them are
//! active, which is what a wave operation needs since it reads the registers of
//! lanes EXEC leaves out; `Workgroup` promises the same of every wave of the
//! work group, which is what a barrier needs.
//!
//! An operation may always run where more lanes are present than it reads, so
//! one rule covers them all: an operation belongs in a region whose presence is
//! at least what it reads.
//!
//! The set does not grow as the regions nest. A `Wave` region inside an `Own`
//! region could not be entered: the lanes of an `Own` region are wherever their
//! own control flow took them, and nothing reconverges them part way through a
//! region. So a region that keeps lanes together forces every region enclosing
//! it to keep them together as well, while its siblings stay free.
//!
//! A region is named by the block it is entered at, and holds every block that
//! entry dominates and no inner region's entry does. Nesting is therefore
//! dominance between entries, and a region cannot be entered anywhere but at
//! its entry. Adding or removing ordinary blocks leaves the regions alone; only
//! removing an entry removes a region. The whole function at `Wave` -- one
//! region, entered at the function's entry -- is the wave program the lifter
//! makes.

use super::{BlockId, Func};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Presence {
    /// Only the lane running the code; the others are wherever they are.
    Own,
    /// Every lane of the packet, which runs them in lockstep. This is what a
    /// lowered `Own` region becomes: the lowering brings the packet's lanes
    /// together, so a query over the packet can be answered there.
    Packet,
    /// Every lane of the wave, the mask saying which are active.
    Wave,
    /// Every lane of every wave of the work group.
    Workgroup,
}

/// Each block's immediate dominator. A block the entry does not reach has none.
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

    /// Whether `a` dominates `b`, which it does to itself.
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

/// The blocks a function's entry reaches, in reverse postorder.
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
    /// Puts the whole function in one region with `presence` there.
    pub fn one_region(&mut self, presence: Presence) {
        self.regions = BTreeMap::from([(self.entry, presence)]);
    }

    /// Lowering to packets of lanes running in lockstep brings the lanes of a
    /// packet together, so a region that had only its own lane now has the
    /// packet. What already had the wave or the work group is unchanged.
    pub fn lowered_to_packets(&mut self) {
        for present in self.regions.values_mut() {
            if *present == Presence::Own {
                *present = Presence::Packet;
            }
        }
    }

    /// The least a region holding the whole function must have present: the
    /// lanes its operations read beyond the one running them. A function that
    /// reads no other lane needs none of them present.
    pub fn presence_needed(&self) -> Presence {
        self.blocks
            .values()
            .flat_map(|b| b.insts.iter())
            .filter_map(super::verify::lanes_read)
            .max()
            .unwrap_or(Presence::Own)
    }

    /// Which region each block belongs to, named by the region's entry: the
    /// innermost region whose entry dominates the block. Blocks the function's
    /// entry does not reach are left out.
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

    /// Whether the region entered at `outer` encloses the one entered at
    /// `inner`, which it does not do to itself.
    pub fn encloses(&self, doms: &Dominators, outer: BlockId, inner: BlockId) -> bool {
        outer != inner && doms.dominates(outer, inner)
    }
}
