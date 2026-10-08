mod answers;
mod bits;
mod cells;
mod floats;
mod lanes;
mod state;
mod words;

#[cfg(test)]
pub use floats::float_compare;
pub(super) use state::State;
pub use cells::Cells;
pub use lanes::MAX_LANES;

use super::kernel::Eval;
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::ir::*;

#[inline]
pub(super) fn consistent(q: &mut Eval<'_, State>, x: Bdd) -> Bdd {
    answers::consistent(q, x)
}

#[inline]
pub(super) fn answer(q: &mut Eval<'_, State>, f: &Func, facts: &Facts, out: ValueId, g: Bdd) -> Bdd {
    answers::answer(q, f, facts, out, g)
}

#[cfg(test)]
pub(super) fn wave_answer(q: &mut Eval<'_, State>, f: &Func, facts: &Facts, out: ValueId) -> Bdd {
    answers::wave_answer(q, f, facts, out)
}

#[inline]
pub(super) fn lane_is(q: &mut Eval<'_, State>, f: &Func, facts: &Facts, block: BlockId, target: ValueId) -> Option<Bdd> {
    words::lane_is(q, f, facts, block, target)
}
