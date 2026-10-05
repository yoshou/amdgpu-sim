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

pub(super) struct Journal<E>(Vec<(E, Depth)>);

pub(super) fn evict<K: std::hash::Hash + Eq, V>(cache: &mut Cached<K, V>, key: &K, depth: usize) {
    if cache.get(key).is_some_and(|e| e.1 .1 >= depth) {
        cache.remove(key);
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
        Self(Vec::new())
    }
}

impl<E> Journal<E> {
    #[inline]
    pub(super) fn note(&mut self, entry: E, depth: Depth) {
        if depth != FREE {
            self.0.push((entry, depth));
        }
    }

    #[inline]
    pub(super) fn mark(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub(super) fn settle(&mut self, mark: usize, depth: usize, mut evict: impl FnMut(E)) {
        let entries: Vec<(E, Depth)> = self.0.drain(mark..).collect();
        for (entry, at) in entries {
            if at.1 < depth {
                self.0.push((entry, at));
            } else {
                evict(entry);
            }
        }
    }
}
