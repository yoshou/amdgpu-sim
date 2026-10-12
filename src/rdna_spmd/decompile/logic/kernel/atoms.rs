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
    params: HashMap<ValueId, u32>,
    markers: HashMap<Choice, u32>,
    blocks: HashMap<BlockId, u32>,
    detour: HashMap<Atom, u32>,
    supports: HashMap<Bdd, Rc<Vec<u32>>>,
}

impl Atoms {
    pub(super) fn new(f: &Func, facts: &Facts, markers_first: bool, listed: &[Choice]) -> Self {
        let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
        let mut origins: HashMap<ValueId, (usize, usize, usize)> = HashMap::default();
        let mut keyed = Vec::new();
        for (r, id) in facts.order.iter().enumerate() {
            for (index, &(v, _)) in f.blocks[id].params.iter().enumerate() {
                let mut found: Option<Option<(usize, usize, usize)>> = None;
                for &(pred, slot) in &facts.incoming[id] {
                    if rank[&pred] >= r {
                        continue;
                    }
                    let arg = f.blocks[&pred].term.edges().nth(slot).unwrap().args[index];
                    let key = origins.get(&arg).copied().unwrap_or_else(|| match facts.site[arg.0] {
                        Site::Inst { block, index } => (rank[&block], 1, index),
                        Site::Param { block, index } => (rank[&block], 0, index),
                        Site::Unreached => (0, 0, 0),
                    });
                    found = match found {
                        None => Some(Some(key)),
                        Some(Some(old)) if old == key => Some(Some(old)),
                        _ => Some(None),
                    };
                }
                let origin = found.flatten().unwrap_or((r, 0, index));
                origins.insert(v, origin);
                keyed.push((origin, r, index, v));
            }
        }
        keyed.sort_unstable();
        assert!(keyed.len() < 1 << 22, "register layout too large");
        let params = keyed.into_iter().enumerate().map(|(i, (.., v))| (v, i as u32)).collect();
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
                    Some(&order) => return (1 << 23) + (view << 22) + order,
                    None => {
                        assert!(v.0 < 1 << 28, "function too large");
                        (1 << 30) | (view << 28) | v.0 as u32
                    }
                }
            }
            Atom::Lane(5) => return 1 << 17,
            Atom::Lane(i) => return (1 << 17) + 1 + i as u32,
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
                assert!(v.0 < 1 << 23 && i < 32, "function too large");
                (1 << 29) + (((1 << 23) - 1 - v.0 as u32) << 5) + i as u32
            }
            Atom::Fresh(PATH, _, i) => {
                assert!(i < 1 << 16, "too many paths");
                return (1 << 16) + i;
            }
            Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => {
                let next = self.detour.len() as u32;
                assert!(next < 1 << 22, "too many detour values");
                return (1 << 17) + 8 + *self.detour.entry(atom).or_insert(next);
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
    fn lane_bits_and_detour_values_come_before_every_atom_of_the_program() {
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
            Atom::Lane(0),
            Atom::Lane(4),
            Atom::Fresh(0, ValueId(7), 0),
            Atom::Term(3, true),
            Atom::Fresh(4, ValueId(9), 0),
            Atom::Bit(a),
            Atom::Bit(c),
            Atom::View(b),
            Atom::View(d),
            Atom::WordBit(inner, 0),
            Atom::Bit(inner),
            Atom::Cell(last, 0, 0),
        ];
        let numbers: Vec<u32> = atoms_in_order.iter().map(|&x| atoms.number(x)).collect();
        assert!(numbers.windows(2).all(|w| w[0] < w[1]), "{:x?}", numbers);
        let mut last_markers = Atoms::new(&f, &facts, false, &listed);
        let lane = last_markers.number(Atom::Lane(5));
        let marker = last_markers.number(Atom::Marker(Choice::Meet(0)));
        assert!(lane < last_markers.number(Atom::Fresh(1, ValueId(0), 0)) && marker > last_markers.number(Atom::WordBit(inner, 0)));
    }

    #[test]
    fn param_atoms_follow_where_the_values_they_carry_are_defined() {
        let mut f = Func::new(BlockId(0), Presence::Wave, 64);
        let (x, y) = (f.value(Ty::I32), f.value(Ty::I32));
        let (p0, p1) = (f.value(Ty::I32), f.value(Ty::I32));
        let (q0, q1) = (f.value(Ty::I32), f.value(Ty::I32));
        let (b1, b2) = (BlockId(1), BlockId(2));
        f.blocks.insert(
            BlockId(0),
            Block {
                params: vec![],
                insts: vec![
                    Inst::Core { value: x, ty: Ty::I32, op: Op::Const(Ty::I32, 1) },
                    Inst::Core { value: y, ty: Ty::I32, op: Op::Const(Ty::I32, 2) },
                ],
                term: Term::Br(Edge { dst: b1, args: vec![y, x] }),
            },
        );
        f.blocks.insert(
            b1,
            Block {
                params: vec![(p0, Ty::I32), (p1, Ty::I32)],
                insts: vec![],
                term: Term::Br(Edge { dst: b2, args: vec![p0, p1] }),
            },
        );
        f.blocks.insert(
            b2,
            Block {
                params: vec![(q0, Ty::I32), (q1, Ty::I32)],
                insts: vec![],
                term: Term::Ret(vec![]),
            },
        );
        let facts = Facts::new(&f, &[], &BTreeSet::new());
        let mut atoms = Atoms::new(&f, &facts, false, &[]);
        let atoms_in_order = [
            Atom::Bit(p1),
            Atom::Bit(q1),
            Atom::Bit(p0),
            Atom::Bit(q0),
            Atom::View(p1),
            Atom::View(q1),
            Atom::View(p0),
            Atom::View(q0),
        ];
        let numbers: Vec<u32> = atoms_in_order.iter().map(|&a| atoms.number(a)).collect();
        assert!(numbers.windows(2).all(|w| w[0] < w[1]), "{:x?}", numbers);
    }
}
