use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Atom {
    Bit(ValueId),
    View(ValueId),
    WordBit(ValueId, u8),
    Lane(u8),
    Fresh(usize, ValueId, u32),
    Term(usize, bool),
    Next(ValueId, bool),
    Marker(Choice),
    Cell(BlockId, u16, u8),
    Some(BlockId, u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Choice {
    Query(ValueId),
    Word(ValueId),
    Meet(usize),
}

const MARKERS_LAST: u32 = 0xfffe_0000;
pub const PATH: usize = 3;

pub(super) struct Atoms {
    markers_first: bool,
    vars: HashMap<Atom, u32>,
    atoms: HashMap<u32, Atom>,
    params: HashMap<ValueId, (usize, usize)>,
    markers: HashMap<Choice, u32>,
    detour: HashMap<Atom, u32>,
    supports: HashMap<Bdd, Rc<Vec<u32>>>,
}

impl Atoms {
    pub(super) fn new(f: &Func, facts: &Facts, markers_first: bool, listed: &[Choice]) -> Self {
        let mut params = HashMap::default();
        for (rank, id) in facts.order.iter().enumerate() {
            for (index, &(v, _)) in f.blocks[id].params.iter().enumerate() {
                params.insert(v, (rank, index));
            }
        }
        assert!(listed.len() < 1 << 16, "too many conversion choices");
        let markers = listed
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        Self {
            markers_first,
            vars: HashMap::default(),
            atoms: HashMap::default(),
            params,
            markers,
            detour: HashMap::default(),
            supports: HashMap::default(),
        }
    }

    pub(super) fn atom(&mut self, m: &mut Manager, atom: Atom) -> Bdd {
        let var = match self.vars.get(&atom) {
            Some(&var) => var,
            None => {
                let var = self.number(atom);
                self.atoms.insert(var, atom);
                self.vars.insert(atom, var);
                var
            }
        };
        m.var(var)
    }

    fn number(&mut self, atom: Atom) -> u32 {
        let var = match atom {
            Atom::Marker(c) => {
                let i = self.markers[&c];
                return if self.markers_first {
                    i
                } else {
                    MARKERS_LAST + i
                };
            }
            Atom::Bit(v) | Atom::View(v) => {
                let view = matches!(atom, Atom::View(_)) as u32;
                match self.params.get(&v) {
                    Some(&(rank, index)) => {
                        assert!(
                            index < 1 << 13 && rank < 1 << 16,
                            "register layout too large"
                        );
                        ((index as u32) << 17) | (view << 16) | rank as u32
                    }
                    None => {
                        assert!(v.0 < 1 << 28, "function too large");
                        (1 << 30) | ((v.0 as u32) << 1) | view
                    }
                }
            }
            Atom::Lane(i) => (2 << 30) | i as u32,
            Atom::Cell(block, group, bit) => {
                assert!(block.0 < 1 << 19 && group < 64 && bit < 16, "too many cells");
                (1 << 30) | (1 << 29) | ((block.0 as u32) << 10) | ((group as u32) << 4) | bit as u32
            }
            Atom::Some(block, k) => {
                assert!(block.0 < 1 << 19 && k < 1 << 10, "too many answers");
                (1 << 30) | (1 << 29) | (1 << 28) | ((block.0 as u32) << 10) | k as u32
            }
            Atom::WordBit(v, i) => {
                assert!(v.0 < 1 << 26, "function too large");
                (2 << 30) + 8 + ((v.0 as u32) << 3) + i as u32
            }
            Atom::Fresh(PATH, _, i) => {
                assert!(i < 1 << 16, "too many paths");
                return (1 << 16) + i;
            }
            Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => {
                let next = self.detour.len() as u32;
                assert!(next < 1 << 29, "too many detour values");
                (3 << 30) | *self.detour.entry(atom).or_insert(next)
            }
        };
        var + (1 << 17)
    }

    #[inline]
    pub(super) fn of(&self, var: u32) -> Atom {
        self.atoms[&var]
    }

    #[inline]
    pub(super) fn get(&self, var: u32) -> Option<Atom> {
        self.atoms.get(&var).copied()
    }

    #[inline]
    pub(super) fn var(&self, atom: Atom) -> Option<u32> {
        self.vars.get(&atom).copied()
    }

    #[inline]
    pub(super) fn marks(&self, c: Choice) -> bool {
        self.markers.contains_key(&c)
    }

    pub(super) fn support(&mut self, m: &mut Manager, f: Bdd) -> Rc<Vec<u32>> {
        if let Some(s) = self.supports.get(&f) {
            return s.clone();
        }
        let s: Rc<Vec<u32>> = Rc::new(m.support(f).into_iter().collect());
        self.supports.insert(f, s.clone());
        s
    }
}

pub(super) fn exists(m: &mut Manager, vars: &[u32], f: Bdd) -> Bdd {
    if vars.is_empty() {
        return f;
    }
    let mut vars = vars.to_vec();
    vars.sort_unstable();
    m.exists(f, &|v| vars.binary_search(&v).is_ok())
}

pub(super) fn forall(m: &mut Manager, vars: &[u32], f: Bdd) -> Bdd {
    if vars.is_empty() {
        return f;
    }
    let mut vars = vars.to_vec();
    vars.sort_unstable();
    m.forall(f, &|v| vars.binary_search(&v).is_ok())
}

pub(super) fn scope(atom: Atom, facts: &Facts) -> Option<BlockId> {
    match atom {
        Atom::Bit(v) | Atom::View(v) | Atom::WordBit(v, _) => match facts.site[v.0] {
            Site::Param { block, .. } | Site::Inst { block, .. } => Some(block),
            Site::Unreached => None,
        },
        Atom::Cell(block, ..) | Atom::Some(block, _) => Some(block),
        Atom::Lane(_) | Atom::Marker(_) | Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => None,
    }
}
