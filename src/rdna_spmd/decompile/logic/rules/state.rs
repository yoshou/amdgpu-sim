use super::super::kernel::{Atom, Atoms, Binding, Queries, Rules};
use super::answers::{Answers, HasAnswers};
use super::bits::{compute_bit, compute_view};
use super::cells::{self, Cells, HasCells};
use super::lanes::{HasLanes, LaneValues, MAX_LANES};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::ir::*;

#[derive(Default)]
pub struct State {
    lanes: LaneValues,
    cells: Cells,
    answers: Answers,
}

impl State {
    pub fn with_cells(cells: Cells) -> Self {
        Self {
            cells,
            ..Self::default()
        }
    }

    pub fn lane_function(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Option<[u32; MAX_LANES]> {
        self.lanes.of(f, facts, v, 0)
    }

    pub fn lane_bits(&mut self, f: &Func, facts: &Facts, v: ValueId) -> Option<[(u32, u32); MAX_LANES]> {
        if f.types[v.0] != Ty::I32 {
            return None;
        }
        Some(self.lanes.known_bits(f, facts, v, 0))
    }

    pub fn cell_leaves(&self, block: BlockId, group: u16) -> &[ValueId] {
        self.cells.leaves(block, group)
    }

    pub fn answer_sources(&self, block: BlockId) -> Vec<ValueId> {
        self.answers.sources(block)
    }
}

impl HasLanes for State {
    #[inline]
    fn lanes_mut(&mut self) -> &mut LaneValues {
        &mut self.lanes
    }
}

impl HasCells for State {
    #[inline]
    fn cells_mut(&mut self) -> &mut Cells {
        &mut self.cells
    }
}

impl HasAnswers for State {
    #[inline]
    fn answers(&self) -> &Answers {
        &self.answers
    }

    #[inline]
    fn answers_mut(&mut self) -> &mut Answers {
        &mut self.answers
    }
}

impl Rules for State {
    #[inline]
    fn bit<Q: Queries<State = Self>>(q: &mut Q, f: &Func, facts: &Facts, v: ValueId) -> Bdd {
        compute_bit(q, f, facts, v)
    }

    #[inline]
    fn view<Q: Queries<State = Self>>(q: &mut Q, f: &Func, facts: &Facts, w: ValueId) -> Bdd {
        compute_view(q, f, facts, w)
    }

    fn uniform(&self, atoms: &Atoms, facts: &Facts, var: u32) -> bool {
        match atoms.of(var) {
            Atom::Bit(v) => facts.uniform[v.0] || self.cells.uniform_test(v),
            Atom::View(v) => facts.saturated[v.0],
            Atom::WordBit(v, _) => facts.uniform[v.0],
            Atom::Marker(_) => true,
            Atom::Cell(block, group, bit) => self.cells.uniform_cell(block, group, bit),
            Atom::Some(..) => true,
            Atom::Lane(_) | Atom::Fresh(..) | Atom::Term(..) | Atom::Next(..) => false,
        }
    }

    #[inline]
    fn bridges<Q: Queries<State = Self>>(q: &mut Q, f: &Func, facts: &Facts, src: BlockId, slot: usize) -> Vec<Binding> {
        cells::bridges(q, f, facts, src, slot)
    }
}
