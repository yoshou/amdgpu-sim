use super::super::address::{Region, Regions};
use crate::rdna_spmd::environment::Environment;

const FIXED: [Option<Region>; 4] = [None, Some(Region::Kernarg), Some(Region::Dispatch), Some(Region::Private)];

pub(super) struct Layout {
    allocations: Vec<u64>,
    words: usize,
}

impl Layout {
    pub(super) fn new(env: &Environment) -> Self {
        let mut allocations: Vec<u64> = env.bindings.iter().map(|b| b.allocation).collect();
        allocations.sort_unstable();
        allocations.dedup();
        let words = (FIXED.len() + allocations.len()).div_ceil(64);
        Self { allocations, words }
    }

    #[inline]
    pub(super) fn words(&self) -> usize {
        self.words
    }

    #[inline]
    pub(super) fn allocations(&self) -> &[u64] {
        &self.allocations
    }

    fn index(&self, r: Option<Region>) -> usize {
        match r {
            Some(Region::Allocation(id)) => FIXED.len() + self.allocations.binary_search(&id).unwrap(),
            r => FIXED.iter().position(|&x| x == r).unwrap(),
        }
    }

    pub(super) fn mark(&self, set: &mut [u64], r: Option<Region>) {
        let i = self.index(r);
        set[i / 64] |= 1 << (i % 64);
    }

    pub(super) fn has(&self, set: &[u64], r: Option<Region>) -> bool {
        let i = self.index(r);
        set[i / 64] >> (i % 64) & 1 != 0
    }

    pub(super) fn region(&self, i: usize) -> Option<Region> {
        match FIXED.get(i) {
            Some(&r) => r,
            None => Some(Region::Allocation(self.allocations[i - FIXED.len()])),
        }
    }

    pub(super) fn regions(&self, set: &[u64]) -> Regions {
        let mut out = Regions::default();
        for i in 0..FIXED.len() + self.allocations.len() {
            if set[i / 64] >> (i % 64) & 1 != 0 {
                out.add(self.region(i));
            }
        }
        out
    }
}

pub(super) fn merge(into: &mut [u64], from: &[u64]) -> bool {
    let mut changed = false;
    for (x, &y) in into.iter_mut().zip(from) {
        changed |= *x | y != *x;
        *x |= y;
    }
    changed
}

pub(super) fn pointers(into: &mut [u64], from: &[u64]) {
    into[0] |= from[0] & !1;
    merge(&mut into[1..], &from[1..]);
}

pub(super) fn points(set: &[u64]) -> bool {
    set[0] & !1 != 0 || set[1..].iter().any(|&w| w != 0)
}
