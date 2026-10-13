use super::patterns::projected_word;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;
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
    Runs(u32),
    Holds(u32, u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Choice {
    Query(ValueId),
    Word(ValueId),
    Meet(usize),
}

const MARKERS_LAST: u32 = 0xfffe_0000;
const RUNS_LAST: u32 = 0xfffd_0000;
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
        let origin_of = |origins: &HashMap<ValueId, (usize, usize, usize)>, v: ValueId| {
            origins.get(&v).copied().unwrap_or_else(|| match facts.site[v.0] {
                Site::Inst { block, index } => (rank[&block], 1, index),
                Site::Param { block, index } => (rank[&block], 0, index),
                Site::Unreached => (0, 0, 0),
            })
        };
        for (r, id) in facts.order.iter().enumerate() {
            let preds: Vec<(BlockId, usize)> = facts.incoming[id].iter().copied().filter(|&(pred, _)| rank[&pred] < r).collect();
            let forward: Vec<Vec<(ValueId, Option<ValueId>)>> = preds
                .iter()
                .map(|&(pred, slot)| anchors(f, facts, pred, f.blocks[&pred].term.edges().nth(slot).unwrap(), preds.len() > 1))
                .collect();
            for (index, &(v, _)) in f.blocks[id].params.iter().enumerate() {
                let mut found: Option<Option<(usize, usize, usize)>> = None;
                let mut anchored: Option<(usize, usize, usize)> = None;
                for args in &forward {
                    let (arg, anchor) = args[index];
                    let key = origin_of(&origins, arg);
                    if let Some(anchor) = anchor {
                        let at = origin_of(&origins, anchor);
                        anchored = Some(anchored.map_or(at, |a| a.min(at)));
                    }
                    found = match found {
                        None => Some(Some(key)),
                        Some(Some(old)) if old == key => Some(Some(old)),
                        _ => Some(None),
                    };
                }
                let origin = found.flatten().or(anchored).unwrap_or((r, 0, index));
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
            Atom::Runs(i) => {
                assert!(i < 1 << 14, "too many accesses");
                return RUNS_LAST + 4 * i;
            }
            Atom::Holds(i, k) => {
                assert!(i < 1 << 14 && k < 2, "too many accesses");
                return RUNS_LAST + 4 * i + 1 + k as u32;
            }
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

fn carried(f: &Func, facts: &Facts, mut v: ValueId) -> ValueId {
    loop {
        v = match facts.op(f, v) {
            Some(Op::Convert(Cvt::Trunc, Ty::I1, s)) => match projected_word(f, facts, s) {
                Some(w) => w,
                None => return v,
            },
            Some(Op::Convert(Cvt::Bitcast, Ty::I1, a)) => a,
            _ => return v,
        };
    }
}

fn anchors(f: &Func, facts: &Facts, block: BlockId, edge: &Edge, merging: bool) -> Vec<(ValueId, Option<ValueId>)> {
    let computed = |v: ValueId| matches!(facts.site[v.0], Site::Inst { block: b, .. } if b == block);
    let bits = |v: ValueId| match f.types[v.0] {
        Ty::I1 => true,
        Ty::I32 | Ty::I64 => facts.viewed[v.0],
        _ => false,
    };
    let values: Vec<ValueId> = edge.args.iter().map(|&arg| carried(f, facts, arg)).collect();
    if !merging || !values.iter().any(|&v| computed(v) && bits(v)) {
        return values.into_iter().map(|v| (v, None)).collect();
    }
    let copied: BTreeSet<ValueId> = values.iter().copied().filter(|&v| !computed(v) && bits(v)).collect();
    let inputs: Vec<Vec<ValueId>> = values
        .iter()
        .map(|&value| {
            let mut found = Vec::new();
            if !computed(value) || !bits(value) {
                return found;
            }
            let mut seen = BTreeSet::new();
            let mut stack = vec![value];
            while let Some(v) = stack.pop() {
                if !bits(v) || !seen.insert(v) {
                    continue;
                }
                if !computed(v) {
                    if copied.contains(&v) {
                        found.push(v);
                    }
                    continue;
                }
                if let Some(inst @ Inst::Core { .. }) = facts.inst(f, v) {
                    stack.extend(inst.operands().into_iter().rev());
                }
            }
            found
        })
        .collect();
    let mut dependents: HashMap<ValueId, usize> = HashMap::default();
    for list in &inputs {
        for &v in list {
            *dependents.entry(v).or_default() += 1;
        }
    }
    values
        .iter()
        .zip(&inputs)
        .map(|(&value, list)| (value, list.iter().copied().min_by_key(|v| dependents[v])))
        .collect()
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
        Atom::Lane(_) | Atom::Marker(_) | Atom::Runs(_) | Atom::Holds(..) | Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            Atom::Lane(5),
            Atom::Lane(0),
            Atom::Lane(4),
            Atom::Fresh(0, ValueId(7), 0),
            Atom::Fresh(PATH, ValueId(2), 0),
            Atom::Term(3, true),
            Atom::Fresh(PATH, ValueId(0), (1 << 16) - 1),
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

    #[test]
    fn a_param_updated_from_another_carried_param_sits_beside_it() {
        let mut f = Func::new(BlockId(0), Presence::Wave, 64);
        let (a, q, c) = (f.value(Ty::I1), f.value(Ty::I1), f.value(Ty::I1));
        let (a1, q1, c1) = (f.value(Ty::I1), f.value(Ty::I1), f.value(Ty::I1));
        let (k, masked, updated) = (f.value(Ty::I1), f.value(Ty::I1), f.value(Ty::I1));
        let (p, r) = (f.value(Ty::I1), f.value(Ty::I1));
        let (b1, b2) = (BlockId(1), BlockId(2));
        f.blocks.insert(
            BlockId(0),
            Block {
                params: vec![(a, Ty::I1), (q, Ty::I1), (c, Ty::I1)],
                insts: vec![],
                term: Term::CondBr {
                    cond: c,
                    yes: Edge { dst: b1, args: vec![a, q, c] },
                    no: Edge { dst: b2, args: vec![a, q] },
                },
            },
        );
        f.blocks.insert(
            b1,
            Block {
                params: vec![(a1, Ty::I1), (q1, Ty::I1), (c1, Ty::I1)],
                insts: vec![
                    Inst::Core { value: k, ty: Ty::I1, op: Op::Const(Ty::I1, 1) },
                    Inst::Core { value: masked, ty: Ty::I1, op: Op::Int(IntOp::And, k, q1) },
                    Inst::Core { value: updated, ty: Ty::I1, op: Op::Select(c1, masked, a1) },
                ],
                term: Term::Br(Edge { dst: b2, args: vec![updated, q1] }),
            },
        );
        f.blocks.insert(
            b2,
            Block {
                params: vec![(p, Ty::I1), (r, Ty::I1)],
                insts: vec![],
                term: Term::Ret(vec![]),
            },
        );
        let facts = Facts::new(&f, &[], &BTreeSet::new());
        let mut atoms = Atoms::new(&f, &facts, false, &[]);
        let atoms_in_order = [
            Atom::Bit(a),
            Atom::Bit(a1),
            Atom::Bit(q),
            Atom::Bit(q1),
            Atom::Bit(p),
            Atom::Bit(r),
            Atom::Bit(c),
            Atom::Bit(c1),
        ];
        let numbers: Vec<u32> = atoms_in_order.iter().map(|&x| atoms.number(x)).collect();
        assert!(numbers.windows(2).all(|w| w[0] < w[1]), "{:x?}", numbers);
        assert_eq!(numbers[5], numbers[4] + 1);
    }

    #[test]
    fn a_param_that_takes_the_lane_bit_of_a_word_follows_the_word() {
        let mut f = Func::new(BlockId(0), Presence::Wave, 64);
        let (w, u) = (f.value(Ty::I32), f.value(Ty::I32));
        let (w1, u1) = (f.value(Ty::I32), f.value(Ty::I32));
        let (lane, shifted, bit) = (f.value(Ty::I32), f.value(Ty::I32), f.value(Ty::I1));
        let (x, y) = (f.value(Ty::I1), f.value(Ty::I32));
        let (b1, b2) = (BlockId(1), BlockId(2));
        f.blocks.insert(
            BlockId(0),
            Block {
                params: vec![(w, Ty::I32), (u, Ty::I32)],
                insts: vec![],
                term: Term::Br(Edge { dst: b1, args: vec![w, u] }),
            },
        );
        f.blocks.insert(
            b1,
            Block {
                params: vec![(w1, Ty::I32), (u1, Ty::I32)],
                insts: vec![
                    Inst::Core { value: lane, ty: Ty::I32, op: Op::Env(Env::LaneId) },
                    Inst::Core { value: shifted, ty: Ty::I32, op: Op::Int(IntOp::LShr, w1, lane) },
                    Inst::Core { value: bit, ty: Ty::I1, op: Op::Convert(Cvt::Trunc, Ty::I1, shifted) },
                ],
                term: Term::Br(Edge { dst: b2, args: vec![bit, u1] }),
            },
        );
        f.blocks.insert(
            b2,
            Block {
                params: vec![(x, Ty::I1), (y, Ty::I32)],
                insts: vec![],
                term: Term::Ret(vec![]),
            },
        );
        let facts = Facts::new(&f, &[], &BTreeSet::new());
        let mut atoms = Atoms::new(&f, &facts, false, &[]);
        let atoms_in_order = [Atom::Bit(w), Atom::Bit(w1), Atom::Bit(x), Atom::Bit(u), Atom::Bit(u1), Atom::Bit(y)];
        let numbers: Vec<u32> = atoms_in_order.iter().map(|&a| atoms.number(a)).collect();
        assert!(numbers.windows(2).all(|w| w[0] < w[1]), "{:x?}", numbers);
    }
}
