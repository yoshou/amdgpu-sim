use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeMap;

pub(super) type Sent = (BlockId, usize, BlockId, usize, bool);

pub(super) struct Lattice {
    safe: Bdd,
    h: Vec<Bdd>,
    words: Vec<Bdd>,
    arrivals: BTreeMap<BlockId, Vec<Bdd>>,
    reach: BTreeMap<BlockId, Bdd>,
    guards: HashMap<(usize, usize), Bdd>,
    sent: HashMap<Sent, Bdd>,
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
            sent: HashMap::default(),
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

    pub(super) fn unsent(&mut self, m: &mut Manager, key: Sent, x: Bdd) -> Bdd {
        match self.sent.insert(key, x) {
            Some(before) if before == x => Bdd::FALSE,
            Some(before) => m.ite(before, Bdd::FALSE, x),
            None => x,
        }
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
        self.sent.clear();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsent_passes_on_only_what_a_slot_has_not_sent_since_the_last_narrowing() {
        let mut m = Manager::new();
        let mut lattice = Lattice::new(1);
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let vars: Vec<Bdd> = (0..6).map(|v| m.var(v)).collect();
        let mut grew = 0;
        for round in 0..200 {
            let key = (BlockId(round), 0, BlockId(1), 0, false);
            let mut sent = Bdd::FALSE;
            for _ in 0..6 {
                let mut cube = Bdd::TRUE;
                for &v in &vars {
                    let literal = match next() % 3 {
                        0 => v,
                        1 => m.not(v),
                        _ => Bdd::TRUE,
                    };
                    cube = m.and(cube, literal);
                }
                let x = m.or(sent, cube);
                let delta = lattice.unsent(&mut m, key, x);
                grew += (x != sent) as usize;
                assert_eq!(m.and(delta, sent), Bdd::FALSE, "a slot passes on nothing it sent before");
                assert_eq!(m.or(delta, sent), x, "a slot passes on everything it has not sent");
                assert_eq!(lattice.unsent(&mut m, key, x), Bdd::FALSE, "a slot sends a difference once");
                sent = x;
            }
            let word = (BlockId(round), 0, BlockId(1), 0, true);
            assert_eq!(lattice.unsent(&mut m, word, sent), sent, "each slot keeps its own record");
            assert!(lattice.narrow(&mut m, Bdd::TRUE));
            assert_eq!(lattice.unsent(&mut m, key, sent), sent, "narrowing forgets what every slot sent");
        }
        assert!(grew > 600);
    }
}
