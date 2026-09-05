//! Workgroup barrier rounds, counted per wave rather than per work item.
use std::collections::{BTreeMap, BTreeSet};
#[derive(Default)]
struct Round {
    signalled: BTreeSet<usize>,
    consumed: BTreeSet<usize>,
}
struct Barrier {
    signals: Vec<u64>,
    waits: Vec<u64>,
    rounds: BTreeMap<u64, Round>,
}
pub(super) struct Barriers {
    waves: usize,
    barriers: BTreeMap<u32, Barrier>,
}
impl Barriers {
    pub fn new(waves: usize) -> Self {
        assert!(waves > 0);
        Self {
            waves,
            barriers: BTreeMap::new(),
        }
    }
    fn barrier(&mut self, id: u32) -> &mut Barrier {
        let waves = self.waves;
        self.barriers.entry(id).or_insert_with(|| Barrier {
            signals: vec![0; waves],
            waits: vec![0; waves],
            rounds: BTreeMap::new(),
        })
    }
    pub fn signal(&mut self, wave: usize, id: u32) -> bool {
        assert!(wave < self.waves);
        let b = self.barrier(id);
        let generation = b.signals[wave];
        b.signals[wave] += 1;
        let round = b.rounds.entry(generation).or_default();
        let first = round.signalled.is_empty();
        assert!(round.signalled.insert(wave));
        first
    }
    pub fn wait(&mut self, wave: usize, id: u32) -> bool {
        let waves = self.waves;
        assert!(wave < waves);
        let b = self.barrier(id);
        let generation = b.waits[wave];
        let Some(round) = b.rounds.get_mut(&generation) else {
            return false;
        };
        if round.signalled.len() != waves {
            return false;
        }
        assert!(round.consumed.insert(wave));
        b.waits[wave] += 1;
        if round.consumed.len() == waves {
            b.rounds.remove(&generation);
        }
        true
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounds_ids_and_first_wave_are_independent() {
        let mut b = Barriers::new(3);
        assert!(!b.wait(1, 7));
        assert!(b.signal(2, 7));
        assert!(!b.signal(0, 7));
        assert!(!b.wait(2, 7));
        assert!(b.signal(0, 9));
        assert!(!b.signal(1, 7));
        assert!(b.wait(1, 7));
        assert!(b.wait(2, 7));
        assert!(b.signal(1, 7));
        assert!(b.wait(0, 7));
        assert!(!b.wait(1, 7));
        assert!(!b.signal(0, 7));
        assert!(!b.signal(2, 7));
        for w in 0..3 {
            assert!(b.wait(w, 7));
        }
        assert!(!b.wait(0, 9));
    }
}
