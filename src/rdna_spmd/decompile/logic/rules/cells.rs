use super::super::super::address::compare;
use super::super::super::terms::Terms;
use super::super::kernel::{lane_test, Atom, Binding, Queries};
use crate::rdna_spmd::analysis::bdd::{Bdd, Manager};
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

type HashSet<T> = std::collections::HashSet<T, std::hash::BuildHasherDefault<crate::rdna_spmd::hash::Mix>>;

#[derive(Default)]
struct BlockCells {
    of: HashMap<ValueId, Arc<Tree>>,
    opaque: HashMap<ValueId, (IntPred, ValueId, ValueId)>,
    placed: HashMap<usize, (u16, usize)>,
    groups: Vec<CellGroup>,
}

#[derive(Debug)]
enum Tree {
    Const(bool),
    Leaf(usize),
    Pick(ValueId, Arc<Tree>, Arc<Tree>),
    Test(ValueId, bool, Arc<Tree>),
}

impl Tree {
    fn leaves(&self, out: &mut BTreeSet<usize>) {
        self.leaves_once(out, &mut HashSet::default());
    }

    fn leaves_once(&self, out: &mut BTreeSet<usize>, seen: &mut HashSet<*const Tree>) {
        match self {
            Tree::Const(_) => {}
            Tree::Leaf(i) => {
                out.insert(*i);
            }
            Tree::Pick(_, a, b) => {
                for t in [a, b] {
                    if seen.insert(Arc::as_ptr(t)) {
                        t.leaves_once(out, seen);
                    }
                }
            }
            Tree::Test(_, _, inner) => {
                if seen.insert(Arc::as_ptr(inner)) {
                    inner.leaves_once(out, seen);
                }
            }
        }
    }
}

#[derive(Default)]
struct Distributor {
    preds: Vec<(IntPred, ValueId, ValueId)>,
    trees: HashMap<(IntPred, ValueId, ValueId), Arc<Tree>>,
}

impl Distributor {
    fn distribute(&mut self, f: &Func, facts: &Facts, p: IntPred, a: ValueId, b: ValueId) -> Arc<Tree> {
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
            (IntPred::Eq | IntPred::Ne, Some(w)) => Tree::Test(w, p == IntPred::Eq, Arc::new(tree)),
            _ => tree,
        };
        let tree = Arc::new(tree);
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

struct CellGroup {
    worlds: Vec<Vec<bool>>,
    preds: Vec<(IntPred, ValueId, ValueId)>,
    leaves: BTreeSet<ValueId>,
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
const GROUPS: usize = 1 << 12;
const JOINT: usize = 1 << 10;

#[derive(Default, Clone)]
pub struct Cells {
    blocks: HashMap<BlockId, Arc<BlockCells>>,
    groups: HashMap<(BlockId, u16), (u8, Vec<ValueId>)>,
    uniform_tests: HashSet<ValueId>,
    reads: HashMap<BlockId, Vec<(ValueId, bool)>>,
}

impl Cells {
    pub fn of(f: &Func, facts: &Facts) -> Self {
        let mut cells = Cells::default();
        for &block in &facts.order {
            cells.block(f, facts, block);
        }
        cells
    }

    pub fn agreeing(&self, facts: &Facts) -> Self {
        let mut cells = self.clone();
        for (&block, reads) in &self.reads {
            if reads.iter().any(|&(v, uniform)| facts.uniform[v.0] != uniform) {
                cells.blocks.remove(&block);
                cells.reads.remove(&block);
                cells.groups.retain(|&(b, _), _| b != block);
            }
        }
        cells
    }

    fn block(&mut self, f: &Func, facts: &Facts, block: BlockId) -> Arc<BlockCells> {
        if let Some(cells) = self.blocks.get(&block) {
            return cells.clone();
        }
        let mut reads: Vec<(ValueId, bool)> = Vec::new();
        let mut distributor = Distributor::default();
        let mut trees: Vec<(ValueId, IntPred, ValueId, ValueId, Arc<Tree>)> = Vec::new();
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
            if cells.groups.len() == GROUPS {
                break;
            }
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
            let uniform: Vec<bool> = preds.iter().map(|&p| Terms::new(f, facts).uniform_noting(p, &mut |v, u| reads.push((v, u)))).collect();
            let leaves: BTreeSet<ValueId> = list.iter().flat_map(|&i| leaves[i].iter().flatten().copied()).collect();
            let group = CellGroup::new(worlds, preds, leaves, uniform);
            let number = cells.groups.len() as u16;
            self.groups.insert((block, number), (CellGroup::bits(group.counts.0), group.leaves.iter().copied().collect()));
            for (k, &i) in list.iter().enumerate() {
                cells.placed.insert(i, (number, k));
            }
            cells.groups.push(group);
        }
        for (value, p, a, b, tree) in trees {
            let mut indices = BTreeSet::new();
            tree.leaves(&mut indices);
            if indices.iter().any(|i| !cells.placed.contains_key(i)) {
                reads.push((value, facts.uniform[value.0]));
                if !facts.uniform[value.0] && a != b && leaves_of(a, b).is_some() {
                    cells.opaque.insert(value, (p, a, b));
                }
            }
            cells.of.insert(value, tree);
        }
        reads.sort_unstable();
        reads.dedup();
        self.reads.insert(block, reads);
        let cells = Arc::new(cells);
        self.blocks.insert(block, cells.clone());
        cells
    }

    #[inline]
    pub(super) fn leaves(&self, block: BlockId, group: u16) -> &[ValueId] {
        &self.groups[&(block, group)].1
    }

    #[inline]
    pub(super) fn uniform_test(&self, v: ValueId) -> bool {
        self.uniform_tests.contains(&v)
    }

    #[inline]
    pub(super) fn uniform_cell(&self, block: BlockId, group: u16, bit: u8) -> bool {
        bit < self.groups[&(block, group)].0
    }
}

pub(super) trait HasCells {
    fn cells_mut(&mut self) -> &mut Cells;
}

fn block_cells<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, block: BlockId) -> Arc<BlockCells>
where
    Q::State: HasCells,
{
    q.state_mut().cells_mut().block(f, facts, block)
}

pub(super) fn cell_bit<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, v: ValueId) -> Option<Bdd>
where
    Q::State: HasCells,
{
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
            q.state_mut().cells_mut().uniform_tests.insert(v);
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

fn tree_bit<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, block: BlockId, v: ValueId, tree: &Arc<Tree>, cells: &BlockCells, done: &mut HashMap<*const Tree, Bdd>) -> Bdd {
    if let Some(&b) = done.get(&Arc::as_ptr(tree)) {
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
    done.insert(Arc::as_ptr(tree), result);
    result
}

fn cell_minterm<Q: Queries>(q: &mut Q, block: BlockId, group: u16, cell: usize, g: &CellGroup) -> Bdd {
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

pub(super) fn bridges<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, src: BlockId, slot: usize) -> Vec<Binding>
where
    Q::State: HasCells,
{
    let edge = f.blocks[&src].term.edges().nth(slot).unwrap();
    let dst = edge.dst;
    if dst == src {
        return Vec::new();
    }
    let substitution: HashMap<ValueId, ValueId> = f.blocks[&dst]
        .params
        .iter()
        .zip(&edge.args)
        .map(|(&(param, _), &arg)| (param, arg))
        .collect();
    let (from, to) = (block_cells(q, f, facts, src), block_cells(q, f, facts, dst));
    let mut out = Vec::new();
    for (g_to, target) in to.groups.iter().enumerate() {
        let params: Vec<ValueId> = target.leaves.iter().copied().filter(|l| substitution.contains_key(l)).collect();
        if params.is_empty() || params.iter().any(|p| !q.carried(*p)) {
            continue;
        }
        let mut mapped: BTreeSet<ValueId> = BTreeSet::new();
        let terms = Terms::substituting(f, facts, substitution.clone());
        for &(_, a, b) in &target.preds {
            let mut lane = false;
            terms.leaves(a, 0, &mut mapped, &mut lane);
            terms.leaves(b, 0, &mut mapped, &mut lane);
            if lane {
                mapped.clear();
                break;
            }
        }
        if mapped.is_empty() {
            continue;
        }
        for (g_from, source) in from.groups.iter().enumerate() {
            if source.leaves.is_disjoint(&mapped) {
                continue;
            }
            let preds: Vec<(IntPred, ValueId, ValueId)> = source.preds.iter().chain(&target.preds).copied().collect();
            let mut terms = Terms::substituting(f, facts, substitution.clone());
            let Some(worlds) = terms.worlds(&preds, JOINT) else {
                continue;
            };
            let mut relation = Bdd::FALSE;
            for world in &worlds {
                let (a, b) = world.split_at(source.preds.len());
                let (Some(i), Some(j)) = (source.worlds.iter().position(|w| w == a), target.worlds.iter().position(|w| w == b)) else {
                    continue;
                };
                let left = cell_minterm(q, src, g_from as u16, i, source);
                let right = cell_minterm(q, dst, g_to as u16, j, target);
                let both = q.m().and(left, right);
                relation = q.m().or(relation, both);
            }
            let support = q.scoped(facts, relation, src);
            out.push(Binding {
                atom: Bdd::TRUE,
                bound: relation,
                support,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::super::super::testing::*;
    use super::*;

    fn fingerprint(cells: &Cells) -> Vec<String> {
        let mut blocks: Vec<&BlockId> = cells.blocks.keys().collect();
        blocks.sort();
        let mut out: Vec<String> = blocks
            .into_iter()
            .map(|b| {
                let c = &cells.blocks[b];
                let groups: Vec<String> = c.groups.iter().map(|g| format!("{:?} {:?} {:?} {:?}", g.preds, g.uniform, g.counts, g.worlds)).collect();
                let mut opaque: Vec<String> = c.opaque.iter().map(|(v, o)| format!("{:?} {:?}", v, o)).collect();
                opaque.sort();
                let mut placed: Vec<(&usize, &(u16, usize))> = c.placed.iter().collect();
                placed.sort();
                format!("b{} {:?} {:?} {:?}", b.0, groups, opaque, placed)
            })
            .collect();
        let mut groups: Vec<String> = cells.groups.iter().map(|(k, v)| format!("{:?} {:?}", k, v)).collect();
        groups.sort();
        out.extend(groups);
        out
    }

    #[test]
    fn agreeing_keeps_every_block_whose_reads_hold_and_rebuilds_the_rest_as_new() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let yes = b.constant(e, Ty::I1, 1);
        let table = k.buffer(&mut b, e, 8);
        let u = b.load(e, Space::Global, MemSize::B32, table, yes);
        let at = b.constant(e, Ty::I64, 4);
        let second = b.int(e, IntOp::Add, table, at);
        let v = b.load(e, Space::Global, MemSize::B32, second, yes);
        let mut values = vec![u, v];
        for (pred, x, y) in [(IntPred::Ult, u, v), (IntPred::Eq, u, v)] {
            values.push(b.cmp(e, pred, x, y));
        }
        let (crowd, c) = b.block(&[Ty::I1, Ty::I32]);
        b.br(e, crowd, vec![k.exec, u]);
        for i in 0..GROUP as u64 + 2 {
            let bound = b.constant(crowd, Ty::I32, 3 + 5 * i);
            values.push(b.cmp(crowd, IntPred::Ult, c[1], bound));
        }
        values.push(c[1]);
        let (rest, r) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
        b.br(crowd, rest, vec![c[0], u, v]);
        for (pred, x, y) in [(IntPred::Ult, r[1], r[2]), (IntPred::Ne, r[1], r[2])] {
            values.push(b.cmp(rest, pred, x, y));
        }
        values.extend([r[1], r[2]]);
        let f = &b.f;
        let none = BTreeSet::new();
        let facts = Facts::new(f, &b.inputs, &none);
        let all = Cells::of(f, &facts);
        let (mut dropped, mut shared) = (0, 0);
        for &x in &values {
            let mut other = Facts::new(f, &b.inputs, &none);
            other.uniform[x.0] = !other.uniform[x.0];
            let agreed = all.agreeing(&other);
            for (block, cells) in &agreed.blocks {
                assert!(Arc::ptr_eq(cells, &all.blocks[block]), "b{} must stay shared when flipping v{}", block.0, x.0);
                shared += 1;
            }
            for block in all.blocks.keys().filter(|b| !agreed.blocks.contains_key(b)) {
                assert!(!agreed.groups.keys().any(|&(g, _)| g == *block), "b{} keeps groups after it is dropped", block.0);
                assert!(!agreed.reads.contains_key(block));
                dropped += 1;
            }
            let mut completed = agreed;
            for &block in &other.order {
                completed.block(f, &other, block);
            }
            assert_eq!(fingerprint(&completed), fingerprint(&Cells::of(f, &other)), "flipping the uniformity of v{}", x.0);
        }
        assert!(dropped > 0 && shared > 0, "dropped {} shared {}", dropped, shared);
    }
}
