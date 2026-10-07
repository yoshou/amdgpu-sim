use super::super::super::address::compare;
use super::super::super::terms::interval;
use super::super::kernel::{Atom, Queries};
use super::lanes::{lane_values, HasLanes};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::ir::*;

fn word_is<Q: Queries>(q: &mut Q, w: ValueId, value: u32) -> Bdd {
    let mut g = Bdd::TRUE;
    for i in 0..q.atoms().lanes().trailing_zeros() as u8 {
        let bit = q.atom(Atom::WordBit(w, i));
        let literal = if value >> i & 1 == 1 { bit } else { q.m().not(bit) };
        g = q.m().and(g, literal);
    }
    g
}

fn small_word(f: &Func, facts: &Facts, w: ValueId) -> Option<(u32, u32)> {
    if f.types[w.0] != Ty::I32 || !facts.uniform[w.0] || !matches!(facts.site[w.0], Site::Inst { .. }) || facts.constant(f, w).is_some() {
        return None;
    }
    interval(f, facts, w, 0).filter(|&(_, high)| high < f.lanes)
}

pub(super) fn small_comparison<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, p: IntPred, a: ValueId, b: ValueId) -> Option<Bdd>
where
    Q::State: HasLanes,
{
    let (w, other, flipped) = match (small_word(f, facts, a), small_word(f, facts, b)) {
        (Some(_), None) => (a, b, false),
        (None, Some(_)) => (b, a, true),
        _ => return None,
    };
    if facts.constant(f, other).is_some() || f.types[other.0] != Ty::I32 {
        return None;
    }
    let (low, high) = small_word(f, facts, w)?;
    let values = lane_values(q, f, facts, other)?;
    let mut g = Bdd::FALSE;
    for x in low..=high {
        let is = word_is(q, w, x);
        let lanes = q.lanes(|l| {
            let o = values[l as usize];
            if flipped {
                compare(p, o, x)
            } else {
                compare(p, x, o)
            }
        });
        let both = q.m().and(is, lanes);
        g = q.m().or(g, both);
    }
    Some(g)
}

pub(super) fn lane_is<Q: Queries>(q: &mut Q, f: &Func, facts: &Facts, block: BlockId, target: ValueId) -> Option<Bdd> {
    if let Some((low, high)) = small_word(f, facts, target) {
        let mut g = Bdd::FALSE;
        for x in low..=high {
            let is = word_is(q, target, x);
            let lanes = q.lanes(|l| l == x);
            let both = q.m().and(is, lanes);
            g = q.m().or(g, both);
        }
        return Some(g);
    }
    if !interval(f, facts, target, 0).is_some_and(|(_, high)| high < f.lanes) {
        return None;
    }
    for inst in &f.blocks[&block].insts {
        let Inst::Core {
            value,
            op: Op::Cmp(p @ (IntPred::Eq | IntPred::Ne), a, b),
            ..
        } = inst
        else {
            continue;
        };
        let other = if facts.is_lane_id(f, *a) {
            *b
        } else if facts.is_lane_id(f, *b) {
            *a
        } else {
            continue;
        };
        if other != target {
            continue;
        }
        let bit = q.bit(f, facts, *value);
        return Some(if *p == IntPred::Ne { q.m().not(bit) } else { bit });
    }
    None
}
