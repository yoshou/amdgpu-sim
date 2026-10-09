use crate::rdna_spmd::hash::HashMap;

pub(super) type Depth = (usize, usize);

pub(super) const FREE: Depth = (usize::MAX, 0);

pub(super) fn level(d: usize) -> Depth {
    (d, d)
}

pub(super) type Cached<K, V> = HashMap<K, (V, Depth)>;

#[derive(Default)]
pub(super) struct Trail {
    deps: std::cell::RefCell<Vec<Depth>>,
    pub(super) checking: usize,
}

pub(super) struct Journal<E> {
    entries: Vec<(E, Depth)>,
    spare: Vec<(E, Depth)>,
}

pub(super) fn evict<K: std::hash::Hash + Eq, V>(cache: &mut Cached<K, V>, key: K, depth: usize) {
    if let std::collections::hash_map::Entry::Occupied(found) = cache.entry(key) {
        if found.get().1 .1 >= depth {
            found.remove();
        }
    }
}

impl Trail {
    pub(super) fn depend(&self, depth: Depth) {
        if depth == FREE {
            return;
        }
        if let Some(top) = self.deps.borrow_mut().last_mut() {
            *top = (top.0.min(depth.0), top.1.max(depth.1));
        }
    }

    pub(super) fn current(&self) -> Depth {
        self.deps.borrow().last().copied().unwrap_or(FREE)
    }

    #[inline]
    pub(super) fn push(&self) {
        self.deps.borrow_mut().push(FREE);
    }

    #[inline]
    pub(super) fn pop(&self) -> Depth {
        let depth = self.deps.borrow_mut().pop().unwrap_or(FREE);
        self.depend(depth);
        depth
    }

    #[inline]
    pub(super) fn open(&mut self) -> usize {
        self.checking += 1;
        self.deps.borrow_mut().push(FREE);
        self.checking
    }

    #[inline]
    pub(super) fn close(&mut self, depth: usize) -> Depth {
        let rests = self.deps.borrow_mut().pop().unwrap_or(FREE);
        self.checking -= 1;
        let outer = if rests.1 < depth {
            rests
        } else if rests.0 >= depth {
            FREE
        } else {
            (rests.0, depth - 1)
        };
        self.depend(outer);
        outer
    }
}

impl<E> Default for Journal<E> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
            spare: Vec::new(),
        }
    }
}

impl<E> Journal<E> {
    #[inline]
    pub(super) fn note(&mut self, entry: E, depth: Depth) {
        if depth != FREE {
            self.entries.push((entry, depth));
        }
    }

    #[inline]
    pub(super) fn mark(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub(super) fn settle(&mut self, mark: usize, depth: usize, mut evict: impl FnMut(E)) {
        let mut kept = std::mem::take(&mut self.spare);
        for (entry, at) in self.entries.drain(mark..) {
            if at.1 < depth {
                kept.push((entry, at));
            } else {
                evict(entry);
            }
        }
        self.entries.append(&mut kept);
        self.spare = kept;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evict_drops_only_entries_stored_at_or_inside_the_closed_level() {
        let mut cache: Cached<u32, char> = HashMap::default();
        cache.insert(1, ('a', (0, 2)));
        cache.insert(2, ('b', (1, 3)));
        cache.insert(3, ('c', (2, 5)));
        evict(&mut cache, 1, 3);
        evict(&mut cache, 2, 3);
        evict(&mut cache, 3, 3);
        evict(&mut cache, 4, 3);
        assert_eq!(cache.get(&1).map(|e| e.0), Some('a'), "an entry from outside the level stays");
        assert!(!cache.contains_key(&2), "an entry stored at the level goes");
        assert!(!cache.contains_key(&3), "an entry stored inside the level goes");
        assert_eq!(cache.len(), 1, "evicting a missing key adds nothing");
    }

    #[test]
    fn settle_keeps_outer_entries_in_order_and_evicts_the_rest_in_order() {
        let mut journal: Journal<u32> = Journal::default();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut model: Vec<(u32, Depth)> = Vec::new();
        for round in 0..300u32 {
            for i in 0..next() % 12 {
                let low = (next() % 6) as usize;
                let depth = if next() % 5 == 0 { FREE } else { (low, low + (next() % 4) as usize) };
                journal.note(round * 100 + i as u32, depth);
                if depth != FREE {
                    model.push((round * 100 + i as u32, depth));
                }
            }
            let mark = (next() as usize) % (model.len() + 1);
            let depth = (next() % 8) as usize;
            let mut evicted = Vec::new();
            journal.settle(mark, depth, |e| evicted.push(e));
            let tail: Vec<(u32, Depth)> = model.drain(mark..).collect();
            let want: Vec<u32> = tail.iter().filter(|e| e.1 .1 >= depth).map(|e| e.0).collect();
            model.extend(tail.into_iter().filter(|e| e.1 .1 < depth));
            assert_eq!(evicted, want, "entries recorded at or inside the level leave in the order they came");
            assert_eq!(journal.mark(), model.len());
            assert_eq!(journal.entries, model, "entries from outside the level stay in their order");
        }
    }
}
