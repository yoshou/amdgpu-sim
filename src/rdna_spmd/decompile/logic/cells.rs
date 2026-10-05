use super::super::address::compare;
use super::super::terms::Terms;
use super::atoms::{Atom, Atoms};
use super::patterns::lane_test;
use super::queries::Queries;
use super::HashSet;
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

#[derive(Default)]
pub(super) struct BlockCells {
    of: HashMap<ValueId, Rc<Tree>>,
    opaque: HashMap<ValueId, (IntPred, ValueId, ValueId)>,
    placed: HashMap<usize, (u16, usize)>,
    pub(super) groups: Vec<CellGroup>,
}

#[derive(Debug)]
enum Tree {
    Const(bool),
    Leaf(usize),
    Pick(ValueId, Rc<Tree>, Rc<Tree>),
    Test(ValueId, bool, Rc<Tree>),
}

impl Tree {
    fn leaves(&self, out: &mut BTreeSet<usize>) {
        match self {
            Tree::Const(_) => {}
            Tree::Leaf(i) => {
                out.insert(*i);
            }
            Tree::Pick(_, a, b) => {
                a.leaves(out);
                b.leaves(out);
            }
            Tree::Test(_, _, inner) => inner.leaves(out),
        }
    }
}

#[derive(Default)]
struct Distributor {
    preds: Vec<(IntPred, ValueId, ValueId)>,
    trees: HashMap<(IntPred, ValueId, ValueId), Rc<Tree>>,
}

impl Distributor {
    fn distribute(&mut self, f: &Func, facts: &Facts, p: IntPred, a: ValueId, b: ValueId) -> Rc<Tree> {
        if let Some(tree) = self.trees.get(&(p, a, b)) {
            return tree.clone();
        }
        let select = |x: ValueId| match facts.op(f, x) {
            Some(Op::Select(c, u, v)) => Some((c, u, v)),
            _ => None,
        };
        let tree = if let Some((c, u, v)) = select(a) {
            let yes = self.distribute(f, facts, p, u, b);
            let no = self.distribute(f, facts, p, v, b);
            Tree::Pick(c, yes, no)
        } else if let Some((c, u, v)) = select(b) {
            let yes = self.distribute(f, facts, p, a, u);
            let no = self.distribute(f, facts, p, a, v);
            Tree::Pick(c, yes, no)
        } else if let Some(value) = decided(f, facts, p, a, b) {
            Tree::Const(value)
        } else {
            let index = match self.preds.iter().position(|&q| q == (p, a, b)) {
                Some(i) => i,
                None => {
                    self.preds.push((p, a, b));
                    self.preds.len() - 1
                }
            };
            Tree::Leaf(index)
        };
        let tree = match (p, lane_test(f, facts, a, b)) {
            (IntPred::Eq | IntPred::Ne, Some(w)) => Tree::Test(w, p == IntPred::Eq, Rc::new(tree)),
            _ => tree,
        };
        let tree = Rc::new(tree);
        self.trees.insert((p, a, b), tree.clone());
        tree
    }
}

fn decided(f: &Func, facts: &Facts, p: IntPred, a: ValueId, b: ValueId) -> Option<bool> {
    if a == b {
        return Some(compare(p, 0, 0));
    }
    if f.types[a.0] == Ty::I32 {
        return Terms::new(f, facts).decided(p, a, b);
    }
    match (p, facts.constant(f, a), facts.constant(f, b)) {
        (IntPred::Eq, Some(x), Some(y)) => Some(x == y),
        (IntPred::Ne, Some(x), Some(y)) => Some(x != y),
        _ => None,
    }
}

fn grouped(leaves: &[Option<BTreeSet<ValueId>>]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..leaves.len()).collect();
    fn root(parent: &mut Vec<usize>, i: usize) -> usize {
        let mut i = i;
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    let mut owner: HashMap<ValueId, usize> = HashMap::default();
    for (i, set) in leaves.iter().enumerate() {
        for &leaf in set.iter().flatten() {
            match owner.get(&leaf) {
                Some(&j) => {
                    let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                    parent[a] = b;
                }
                None => {
                    owner.insert(leaf, i);
                }
            }
        }
    }
    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..leaves.len() {
        if leaves[i].is_some() {
            let r = root(&mut parent, i);
            members.entry(r).or_default().push(i);
        }
    }
    members.into_values().collect()
}

pub(super) struct CellGroup {
    pub(super) worlds: Vec<Vec<bool>>,
    pub(super) preds: Vec<(IntPred, ValueId, ValueId)>,
    pub(super) leaves: BTreeSet<ValueId>,
    uniform: Vec<bool>,
    parts: Vec<(usize, usize)>,
    counts: (usize, usize),
}

impl CellGroup {
    fn new(worlds: Vec<Vec<bool>>, preds: Vec<(IntPred, ValueId, ValueId)>, leaves: BTreeSet<ValueId>, uniform: Vec<bool>) -> Self {
        let project = |world: &Vec<bool>, keep: bool| -> Vec<bool> { world.iter().zip(&uniform).filter(|&(_, &u)| u == keep).map(|(&w, _)| w).collect() };
        let index = |list: &mut Vec<Vec<bool>>, part: Vec<bool>| match list.iter().position(|p| *p == part) {
            Some(i) => i,
            None => {
                list.push(part);
                list.len() - 1
            }
        };
        let (mut firsts, mut seconds): (Vec<Vec<bool>>, Vec<Vec<bool>>) = (Vec::new(), Vec::new());
        let parts: Vec<(usize, usize)> = worlds.iter().map(|world| (index(&mut firsts, project(world, true)), index(&mut seconds, project(world, false)))).collect();
        let counts = (firsts.len(), seconds.len());
        CellGroup { worlds, preds, leaves, uniform, parts, counts }
    }

    fn bits(count: usize) -> u8 {
        (usize::BITS - count.saturating_sub(1).leading_zeros()) as u8
    }
}

const CELLS: usize = 64;
const GROUP: usize = 12;

#[derive(Default)]
pub(super) struct Cells {
    blocks: HashMap<BlockId, Rc<BlockCells>>,
    groups: HashMap<(BlockId, u16), (u8, Vec<ValueId>)>,
    uniform_tests: HashSet<ValueId>,
}

impl Cells {
    #[inline]
    pub(super) fn leaves(&self, block: BlockId, group: u16) -> &[ValueId] {
        &self.groups[&(block, group)].1
    }
}

pub(super) fn uniform_atom(atoms: &Atoms, cells: &Cells, facts: &Facts, var: u32) -> bool {
    match atoms.of(var) {
        Atom::Bit(v) => facts.uniform[v.0] || cells.uniform_tests.contains(&v),
        Atom::View(v) => facts.saturated[v.0],
        Atom::WordBit(v, _) => facts.uniform[v.0],
        Atom::Marker(_) => true,
        Atom::Cell(block, group, bit) => bit < cells.groups[&(block, group)].0,
        Atom::Some(..) => true,
        Atom::Lane(_) | Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => false,
    }
}

pub(super) fn block_cells<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, block: BlockId) -> Rc<BlockCells> {
    if let Some(cells) = q.cells().blocks.get(&block) {
        return cells.clone();
    }
    let mut distributor = Distributor::default();
    let mut trees: Vec<(ValueId, IntPred, ValueId, ValueId, Rc<Tree>)> = Vec::new();
    for inst in &f.blocks[&block].insts {
        if let Inst::Core { value, op: Op::Cmp(p, a, b), .. } = inst {
            trees.push((*value, *p, *a, *b, distributor.distribute(f, facts, *p, *a, *b)));
        }
    }
    let preds = distributor.preds;
    let terms = Terms::new(f, facts);
    let leaves_of = |a: ValueId, b: ValueId| {
        if f.types[a.0] != Ty::I32 {
            return None;
        }
        let (mut set, mut lane) = (BTreeSet::new(), false);
        terms.leaves(a, 0, &mut set, &mut lane);
        terms.leaves(b, 0, &mut set, &mut lane);
        (!set.is_empty() && set.iter().all(|l| !facts.lane_word[l.0])).then_some(set)
    };
    let leaves: Vec<Option<BTreeSet<ValueId>>> = preds.iter().map(|&(_, a, b)| leaves_of(a, b)).collect();
    let mut cells = BlockCells::default();
    for list in grouped(&leaves) {
        if list.len() > GROUP {
            continue;
        }
        let preds: Vec<(IntPred, ValueId, ValueId)> = list.iter().map(|&i| preds[i]).collect();
        let Some(worlds) = Terms::new(f, facts).worlds(&preds, CELLS) else {
            continue;
        };
        if worlds.is_empty() {
            continue;
        }
        let uniform: Vec<bool> = preds.iter().map(|&p| Terms::new(f, facts).uniform(p)).collect();
        let leaves: BTreeSet<ValueId> = list.iter().flat_map(|&i| leaves[i].iter().flatten().copied()).collect();
        let group = CellGroup::new(worlds, preds, leaves, uniform);
        let number = cells.groups.len() as u16;
        q.cells_mut().groups.insert((block, number), (CellGroup::bits(group.counts.0), group.leaves.iter().copied().collect()));
        for (k, &i) in list.iter().enumerate() {
            cells.placed.insert(i, (number, k));
        }
        cells.groups.push(group);
    }
    for (value, p, a, b, tree) in trees {
        let mut indices = BTreeSet::new();
        tree.leaves(&mut indices);
        if indices.iter().any(|i| !cells.placed.contains_key(i)) && !facts.uniform[value.0] && a != b && leaves_of(a, b).is_some() {
            cells.opaque.insert(value, (p, a, b));
        }
        cells.of.insert(value, tree);
    }
    let cells = Rc::new(cells);
    q.cells_mut().blocks.insert(block, cells.clone());
    cells
}

pub(super) fn cell_bit<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, v: ValueId) -> Option<Bdd> {
    let Site::Inst { block, .. } = facts.site[v.0] else {
        return None;
    };
    let cells = block_cells(q, f, facts, block);
    let tree = cells.of.get(&v)?.clone();
    if let Some(&(p, a, b)) = cells.opaque.get(&v) {
        let mut terms = Terms::new(f, facts);
        let (mut leaves, mut lane) = (BTreeSet::new(), false);
        terms.leaves(a, 0, &mut leaves, &mut lane);
        terms.leaves(b, 0, &mut leaves, &mut lane);
        if !lane && leaves.iter().all(|l| facts.uniform[l.0]) || terms.uniform((p, a, b)) {
            q.cells_mut().uniform_tests.insert(v);
        }
    }
    let mut done = HashMap::default();
    Some(tree_bit(q, f, facts, block, v, &tree, &cells, &mut done))
}

fn leaf_bit<Q: Queries>(q: &mut Q, block: BlockId, v: ValueId, index: usize, cells: &BlockCells) -> Bdd {
    let Some(&(group, k)) = cells.placed.get(&index) else {
        return q.atom(Atom::Bit(v));
    };
    let g = &cells.groups[group as usize];
    let mut result = Bdd::FALSE;
    if g.uniform[k] {
        let mut seen = BTreeSet::new();
        for (cell, world) in g.worlds.iter().enumerate() {
            if world[k] && seen.insert(g.parts[cell].0) {
                let minterm = part_minterm(q, block, group, 0, g.parts[cell].0, g.counts.0);
                result = q.m().or(result, minterm);
            }
        }
    } else {
        for (cell, world) in g.worlds.iter().enumerate() {
            if world[k] {
                let minterm = cell_minterm(q, block, group, cell, g);
                result = q.m().or(result, minterm);
            }
        }
    }
    result
}

fn tree_bit<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, block: BlockId, v: ValueId, tree: &Rc<Tree>, cells: &BlockCells, done: &mut HashMap<*const Tree, Bdd>) -> Bdd {
    if let Some(&b) = done.get(&Rc::as_ptr(tree)) {
        return b;
    }
    let result = match &**tree {
        Tree::Const(value) => Manager::constant(*value),
        Tree::Leaf(index) => leaf_bit(q, block, v, *index, cells),
        Tree::Test(w, zero, inner) => {
            let mode = q.materialized(facts, *w);
            let lane = |q: &mut Q| {
                let bit = q.view(f, facts, *w);
                if *zero {
                    q.m().not(bit)
                } else {
                    bit
                }
            };
            if mode == Bdd::FALSE {
                lane(q)
            } else {
                let general = tree_bit(q, f, facts, block, v, inner, cells, done);
                if mode == Bdd::TRUE {
                    general
                } else {
                    let lane = lane(q);
                    q.m().ite(mode, general, lane)
                }
            }
        }
        Tree::Pick(c, yes, no) => {
            let c = q.bit(f, facts, *c);
            let yes = tree_bit(q, f, facts, block, v, yes, cells, done);
            let no = tree_bit(q, f, facts, block, v, no, cells, done);
            q.m().ite(c, yes, no)
        }
    };
    done.insert(Rc::as_ptr(tree), result);
    result
}

pub(super) fn cell_minterm<Q: Queries>(q: &mut Q, block: BlockId, group: u16, cell: usize, g: &CellGroup) -> Bdd {
    let (u, v) = g.parts[cell];
    let first = part_minterm(q, block, group, 0, u, g.counts.0);
    let second = part_minterm(q, block, group, CellGroup::bits(g.counts.0), v, g.counts.1);
    q.m().and(first, second)
}

fn part_minterm<Q: Queries>(q: &mut Q, block: BlockId, group: u16, offset: u8, part: usize, count: usize) -> Bdd {
    index_minterm(q, &|i| Atom::Cell(block, group, offset + i), part, count)
}

fn index_minterm<Q: Queries>(q: &mut Q, atom: &dyn Fn(u8) -> Atom, part: usize, count: usize) -> Bdd {
    let bits = CellGroup::bits(count);
    let mut result = Bdd::FALSE;
    for index in 0..(1usize << bits) {
        if index.min(count - 1) != part {
            continue;
        }
        let mut minterm = Bdd::TRUE;
        for i in 0..bits {
            let var = q.atom(atom(i));
            let literal = if index >> i & 1 == 1 { var } else { q.m().not(var) };
            minterm = q.m().and(minterm, literal);
        }
        result = q.m().or(result, minterm);
    }
    result
}
