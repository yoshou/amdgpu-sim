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

pub struct Atoms {
    lanes: u32,
    markers_first: bool,
    vars: HashMap<Atom, u32>,
    atoms: HashMap<u32, Atom>,
    params: HashMap<ValueId, (usize, usize)>,
    markers: HashMap<Choice, u32>,
    blocks: HashMap<BlockId, u32>,
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
            lanes: f.lanes,
            markers_first,
            vars: HashMap::default(),
            atoms: HashMap::default(),
            params,
            markers,
            blocks: f.blocks.keys().enumerate().map(|(rank, &id)| (id, rank as u32)).collect(),
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
                            index < 1 << 13 && rank < 1 << 15,
                            "register layout too large"
                        );
                        return (1 << 23) + (((index as u32) << 16) | (view << 15) | rank as u32);
                    }
                    None => {
                        assert!(v.0 < 1 << 28, "function too large");
                        (1 << 30) | ((v.0 as u32) << 1) | view
                    }
                }
            }
            Atom::Lane(5) => return 1 << 17,
            Atom::Lane(i) => (2 << 30) | i as u32,
            Atom::Cell(block, group, bit) => {
                let rank = self.blocks[&block];
                assert!(rank < 1 << 12 && group < 1 << 12 && bit < 16, "too many cells");
                (1 << 30) | (1 << 29) | (rank << 16) | ((group as u32) << 4) | bit as u32
            }
            Atom::Some(block, k) => {
                let rank = self.blocks[&block];
                assert!(rank < 1 << 12 && k < 1 << 10, "too many answers");
                (1 << 30) | (1 << 29) | (1 << 28) | (rank << 10) | k as u32
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
                assert!(next < 1 << 22, "too many detour values");
                return (1 << 17) + 1 + *self.detour.entry(atom).or_insert(next);
            }
        };
        var + (1 << 17) + 1
    }

    #[inline]
    pub fn lanes(&self) -> u32 {
        self.lanes
    }

    #[inline]
    pub fn of(&self, var: u32) -> Atom {
        self.atoms[&var]
    }

    #[inline]
    pub fn get(&self, var: u32) -> Option<Atom> {
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
        let s: Rc<Vec<u32>> = Rc::new(m.support(f));
        self.supports.insert(f, s.clone());
        s
    }
}

pub fn exists(m: &mut Manager, vars: &[u32], f: Bdd) -> Bdd {
    if vars.is_empty() {
        return f;
    }
    let mut vars = vars.to_vec();
    vars.sort_unstable();
    m.exists(f, &|v| vars.binary_search(&v).is_ok())
}

pub fn forall(m: &mut Manager, vars: &[u32], f: Bdd) -> Bdd {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn cells_number_by_block_order_and_leave_room_for_many_groups() {
        let mut f = Func::new(BlockId(0x15208), Presence::Wave, 64);
        let far = BlockId(0x7_fff0);
        for (id, next) in [(BlockId(0x15208), Some(far)), (far, None)] {
            f.blocks.insert(
                id,
                Block {
                    params: vec![],
                    insts: vec![],
                    term: match next {
                        Some(dst) => Term::Br(Edge { dst, args: vec![] }),
                        None => Term::Ret(vec![]),
                    },
                },
            );
        }
        let facts = Facts::new(&f, &[], &BTreeSet::new());
        let mut atoms = Atoms::new(&f, &facts, false, &[]);
        let atoms_in_order = [
            Atom::Cell(BlockId(0x15208), 0, 0),
            Atom::Cell(BlockId(0x15208), 0, 15),
            Atom::Cell(BlockId(0x15208), 96, 0),
            Atom::Cell(BlockId(0x15208), 4095, 15),
            Atom::Cell(far, 0, 0),
            Atom::Cell(far, 4095, 15),
            Atom::Some(BlockId(0x15208), 0),
            Atom::Some(far, 1023),
        ];
        let numbers: Vec<u32> = atoms_in_order.iter().map(|&a| atoms.number(a)).collect();
        assert!(numbers.windows(2).all(|w| w[0] < w[1]), "{:x?}", numbers);
    }

    #[test]
    fn detour_values_and_the_upper_half_come_before_every_atom_of_the_program() {
        let mut f = Func::new(BlockId(0), Presence::Wave, 64);
        let (a, b) = (f.value(Ty::I1), f.value(Ty::I32));
        let (c, d) = (f.value(Ty::I1), f.value(Ty::I32));
        let inner = f.value(Ty::I32);
        let last = BlockId(1);
        f.blocks.insert(
            BlockId(0),
            Block {
                params: vec![(a, Ty::I1), (b, Ty::I32)],
                insts: vec![],
                term: Term::Br(Edge { dst: last, args: vec![a, b] }),
            },
        );
        f.blocks.insert(
            last,
            Block {
                params: vec![(c, Ty::I1), (d, Ty::I32)],
                insts: vec![],
                term: Term::Ret(vec![]),
            },
        );
        let facts = Facts::new(&f, &[], &BTreeSet::new());
        let listed = [Choice::Meet(0), Choice::Word(b)];
        let mut atoms = Atoms::new(&f, &facts, true, &listed);
        let atoms_in_order = [
            Atom::Marker(Choice::Meet(0)),
            Atom::Marker(Choice::Word(b)),
            Atom::Fresh(PATH, ValueId(0), 0),
            Atom::Fresh(PATH, ValueId(0), (1 << 16) - 1),
            Atom::Lane(5),
            Atom::Fresh(0, ValueId(7), 0),
            Atom::Term(3, true),
            Atom::Fresh(4, ValueId(9), 0),
            Atom::Bit(a),
            Atom::Bit(c),
            Atom::View(b),
            Atom::View(d),
            Atom::Bit(inner),
            Atom::Cell(last, 0, 0),
            Atom::Lane(0),
            Atom::Lane(4),
            Atom::WordBit(inner, 0),
        ];
        let numbers: Vec<u32> = atoms_in_order.iter().map(|&x| atoms.number(x)).collect();
        assert!(numbers.windows(2).all(|w| w[0] < w[1]), "{:x?}", numbers);
        let mut last_markers = Atoms::new(&f, &facts, false, &listed);
        let lane = last_markers.number(Atom::Lane(5));
        let marker = last_markers.number(Atom::Marker(Choice::Meet(0)));
        assert!(lane < last_markers.number(Atom::Fresh(1, ValueId(0), 0)) && marker > last_markers.number(Atom::WordBit(inner, 0)));
    }
}
