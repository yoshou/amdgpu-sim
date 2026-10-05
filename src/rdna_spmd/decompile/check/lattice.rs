use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeMap;

pub(super) struct Lattice {
    safe: Bdd,
    h: Vec<Bdd>,
    words: Vec<Bdd>,
    arrivals: BTreeMap<BlockId, Vec<Bdd>>,
    reach: BTreeMap<BlockId, Bdd>,
    guards: HashMap<(usize, usize), Bdd>,
}

impl Lattice {
    pub(super) fn new(n: usize) -> Self {
        Self {
            safe: Bdd::TRUE,
            h: vec![Bdd::FALSE; n],
            words: vec![Bdd::FALSE; n],
            arrivals: BTreeMap::new(),
            reach: BTreeMap::new(),
            guards: HashMap::default(),
        }
    }

    #[inline]
    pub(super) fn safe(&self) -> Bdd {
        self.safe
    }

    #[inline]
    pub(super) fn h(&self, v: ValueId) -> Bdd {
        self.h[v.0]
    }

    #[inline]
    pub(super) fn word(&self, v: ValueId) -> Bdd {
        self.words[v.0]
    }

    #[inline]
    pub(super) fn reachable(&self, b: BlockId) -> Bdd {
        self.reach.get(&b).copied().unwrap_or(Bdd::FALSE)
    }

    #[inline]
    pub(super) fn arrivals(&self, b: BlockId) -> Option<&Vec<Bdd>> {
        self.arrivals.get(&b)
    }

    #[inline]
    pub(super) fn guard(&self, pred: BlockId, slot: usize) -> Option<Bdd> {
        self.guards.get(&(pred.0, slot)).copied()
    }

    #[inline]
    pub(super) fn keep_guard(&mut self, pred: BlockId, slot: usize, guard: Bdd) {
        self.guards.insert((pred.0, slot), guard);
    }

    pub(super) fn start(&mut self, reach: BTreeMap<BlockId, Bdd>) {
        self.reach = reach;
    }

    #[inline]
    pub(super) fn raise_h(&mut self, m: &mut Manager, v: ValueId, x: Bdd) -> bool {
        if x == Bdd::FALSE {
            return false;
        }
        let x = m.and(x, self.safe);
        let joined = m.or(self.h[v.0], x);
        if joined == self.h[v.0] {
            return false;
        }
        self.h[v.0] = joined;
        true
    }

    #[inline]
    pub(super) fn raise_word(&mut self, m: &mut Manager, v: ValueId, x: Bdd) -> bool {
        if x == Bdd::FALSE {
            return false;
        }
        let x = m.and(x, self.safe);
        let joined = m.or(self.words[v.0], x);
        if joined == self.words[v.0] {
            return false;
        }
        self.words[v.0] = joined;
        true
    }

    pub(super) fn arrive(&mut self, m: &mut Manager, b: BlockId, k: usize, width: usize, c: Bdd) -> bool {
        let c = m.and(c, self.safe);
        if c == Bdd::FALSE {
            return false;
        }
        let old = self
            .arrivals
            .entry(b)
            .or_insert_with(|| vec![Bdd::FALSE; width])[k];
        let joined = m.or(old, c);
        if joined == old {
            return false;
        }
        self.arrivals.get_mut(&b).unwrap()[k] = joined;
        true
    }

    pub(super) fn narrow(&mut self, m: &mut Manager, good: Bdd) -> bool {
        self.safe = m.and(self.safe, good);
        if self.safe == Bdd::FALSE {
            return false;
        }
        let safe = self.safe;
        for x in self.h.iter_mut().chain(self.words.iter_mut()) {
            if *x != Bdd::FALSE {
                *x = m.and(*x, safe);
            }
        }
        for r in self.reach.values_mut() {
            *r = m.and(*r, safe);
        }
        for contributions in self.arrivals.values_mut() {
            for x in contributions {
                *x = m.and(*x, safe);
            }
        }
        self.guards.clear();
        true
    }
}
