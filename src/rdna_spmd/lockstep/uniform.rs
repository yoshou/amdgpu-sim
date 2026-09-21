use super::mask::Masks;
use super::Packing;
use crate::rdna_spmd::analysis::uniformity::Fact;
use crate::rdna_spmd::analysis::{Analyses, Context, Packet, Uniformity};
use crate::rdna_spmd::ir::*;
use crate::rdna_spmd::program::Program;
use std::collections::BTreeMap;

pub(super) fn context(lane: &Program, packing: Packing) -> Context<'_> {
    Context {
        packet: Some(Packet {
            aligned: packing.aligned,
        }),
        ..Context::new(
            &lane.registry,
            &lane.parameter_inputs,
            exec_index(&lane.parameter_inputs, lane.registry.registers().exec),
            packing.lanes,
        )
    }
}

pub(super) fn assume(lane: &Program, packing: Packing) -> Vec<bool> {
    Analyses::new(context(lane, packing))
        .get::<Uniformity>(&lane.ir)
        .uniform()
}

pub(super) fn refuted(
    lane: &Program,
    packing: Packing,
    packet: &Func,
    masks: &Masks,
    folded: &BTreeMap<ValueId, ValueId>,
) -> (Vec<ValueId>, bool) {
    let facts = Analyses::new(context(lane, packing)).get::<Uniformity>(packet);
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
