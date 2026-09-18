//! Which lane bits every lane agrees on.
//!
//! The lowering branches on a query that only tests uniform bits and lets no
//! lane stay behind in a loop whose exits are uniform, so it has to know which
//! bits are uniform before it emits the packet program -- and whether a bit is
//! uniform depends on what the lowering emitted around it. It starts from what
//! the packet analysis says of the lane program, where every merge is taken to
//! agree, lowers, and asks the same analysis of the packet program it made. A
//! bit it relied on that turns out to vary is no longer relied on, and the
//! lowering runs again; every round only gives up assumptions, so the rounds
//! end, and the program that comes out rests only on what the analysis the
//! code generator trusts confirms.

use super::mask::Masks;
use super::Packing;
use crate::rdna_spmd::analysis::uniformity::Fact;
use crate::rdna_spmd::analysis::{Analyses, Context, MaskValues, Packet, Uniformity};
use crate::rdna_spmd::compiler::exec_index;
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::program::LiftedFunction;
use std::collections::BTreeMap;

pub(super) fn context(lane: &LiftedFunction, packing: Packing) -> Context<'_> {
    Context {
        packet: Some(Packet {
            aligned: packing.aligned,
        }),
        ..Context::new(
            &lane.registry,
            &lane.parameter_inputs,
            exec_index(&lane.parameter_inputs, &lane.registry),
            packing.lanes,
        )
    }
}

/// What the packet analysis says of the lane program's own values.
pub(super) fn assume(lane: &LiftedFunction, packing: Packing) -> Vec<bool> {
    Analyses::new(context(lane, packing))
        .get::<Uniformity<MaskValues>>(&lane.ir)
        .uniform()
}

/// What the packet analysis refutes of the lowering's assumptions: the lane
/// values whose bits were taken to be uniform and are not, and whether a bit
/// the lowering made from other bits was refuted as well.
pub(super) fn refuted(
    lane: &LiftedFunction,
    packing: Packing,
    packet: &Func,
    masks: &Masks,
    folded: &BTreeMap<ValueId, ValueId>,
) -> (Vec<ValueId>, bool) {
    let facts = Analyses::new(context(lane, packing)).get::<Uniformity<MaskValues>>(packet);
    let mut origins = Vec::new();
    let mut derived = false;
    for atom in masks.atoms() {
        if !atom.uniform {
            continue;
        }
        let value = folded.get(&atom.value).copied().unwrap_or(atom.value);
        if facts.facts[value.0] == Fact::Uniform {
            continue;
        }
        match atom.origin {
            Some(origin) => origins.push(origin),
            None => derived = true,
        }
    }
    origins.sort();
    origins.dedup();
    (origins, derived)
}
