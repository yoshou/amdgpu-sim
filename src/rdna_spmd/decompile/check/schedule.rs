use super::program::Program;
use super::Mode;
use crate::rdna_spmd::analysis::facts::Use;
use crate::rdna_spmd::ir::*;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

struct Worklist {
    dirty_params: BTreeMap<BlockId, BTreeSet<usize>>,
    dirty_insts: BTreeMap<BlockId, BTreeSet<usize>>,
    queue: BinaryHeap<Reverse<usize>>,
    queued: Vec<bool>,
}

impl Worklist {
    fn dirty(&mut self, program: &Program, b: BlockId, param: Option<usize>, inst: Option<usize>) {
        if let Some(k) = param {
            self.dirty_params.entry(b).or_default().insert(k);
        }
        if let Some(i) = inst {
            self.dirty_insts.entry(b).or_default().insert(i);
        }
        let r = program.rank[&b];
        if !self.queued[r] {
            self.queued[r] = true;
            self.queue.push(Reverse(r));
        }
    }
}

struct Sweep {
    stale: Vec<bool>,
    next: usize,
    changed: bool,
    rose: bool,
}

enum Order {
    Search(Worklist),
    Direct(Sweep),
}

pub(super) struct Schedule(Order);

impl Schedule {
    pub(super) fn new(mode: Mode, program: &Program) -> Self {
        let (f, facts) = (program.f, program.facts);
        let blocks = facts.order.len();
        Self(match mode {
            Mode::Search => Order::Search(Worklist {
                dirty_params: BTreeMap::new(),
                dirty_insts: facts
                    .order
                    .iter()
                    .map(|&b| (b, (0..f.blocks[&b].insts.len()).collect()))
                    .collect(),
                queue: (0..blocks).map(Reverse).collect(),
                queued: vec![true; blocks],
            }),
            Mode::Direct => Order::Direct(Sweep {
                stale: vec![true; blocks],
                next: 0,
                changed: false,
                rose: false,
            }),
        })
    }

    pub(super) fn next(&mut self) -> Option<usize> {
        match &mut self.0 {
            Order::Search(work) => {
                let Reverse(r) = work.queue.pop()?;
                work.queued[r] = false;
                Some(r)
            }
            Order::Direct(sweep) => loop {
                while sweep.next < sweep.stale.len() {
                    let i = sweep.next;
                    sweep.next += 1;
                    if std::mem::take(&mut sweep.stale[i]) {
                        sweep.rose = false;
                        return Some(i);
                    }
                }
                sweep.next = 0;
                if !std::mem::take(&mut sweep.changed) {
                    return None;
                }
            },
        }
    }

    pub(super) fn transferred(&mut self, program: &Program, b: BlockId, narrowed: bool) {
        let Order::Direct(sweep) = &mut self.0 else {
            return;
        };
        if narrowed {
            sweep.stale.fill(true);
            sweep.changed = true;
        }
        if sweep.rose {
            sweep.changed = true;
            for e in program.f.blocks[&b].term.edges() {
                sweep.stale[program.rank[&e.dst]] = true;
            }
        }
    }

    pub(super) fn arrived(&mut self, program: &Program, b: BlockId, k: usize) {
        match &mut self.0 {
            Order::Search(work) => work.dirty(program, b, Some(k), None),
            Order::Direct(sweep) => sweep.stale[program.rank[&b]] = true,
        }
    }

    pub(super) fn rose(&mut self, program: &Program, v: ValueId) {
        let work = match &mut self.0 {
            Order::Search(work) => work,
            Order::Direct(sweep) => {
                sweep.rose = true;
                return;
            }
        };
        let (f, facts) = (program.f, program.facts);
        for &u in &facts.uses[v.0] {
            match u {
                Use::Inst { block, index } => work.dirty(program, block, None, Some(index)),
                Use::Arg { block, edge, index } => {
                    let dst = f.blocks[&block].term.edges().nth(edge).unwrap().dst;
                    work.dirty(program, dst, Some(index), None);
                }
                Use::Cond(_) | Use::Ret(_) => {}
            }
        }
    }

    pub(super) fn params(&mut self, b: BlockId, count: usize) -> Vec<usize> {
        match &mut self.0 {
            Order::Search(work) => work
                .dirty_params
                .remove(&b)
                .map_or_else(Vec::new, |s| s.into_iter().collect()),
            Order::Direct(_) => (0..count).collect(),
        }
    }

    #[inline]
    pub(super) fn pending(&self, b: BlockId) -> bool {
        match &self.0 {
            Order::Search(work) => work.dirty_insts.get(&b).is_some_and(|s| !s.is_empty()),
            Order::Direct(_) => true,
        }
    }

    #[inline]
    pub(super) fn take(&mut self, b: BlockId, index: usize) -> bool {
        match &mut self.0 {
            Order::Search(work) => work.dirty_insts.get_mut(&b).is_some_and(|s| s.remove(&index)),
            Order::Direct(_) => true,
        }
    }
}
