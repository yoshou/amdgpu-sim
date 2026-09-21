use super::super::analysis::uniformity::Fact;
use super::super::analysis::{
    holds_a_lane, Accesses, Analyses, Constants, Context, Packet, Uniformity,
};
use super::super::ir::{Ty, ValueId};
use super::super::pass::{
    adjacency::Adjacency,
    dce::{Dce, DeadParams},
    entry::PacketState,
    pairs::{Pairs, WideMemory},
    simplify::Simplify,
    Driver,
};
use super::super::program::Program;
use super::Prepared;
use std::collections::BTreeMap;

pub(in crate::rdna_spmd) fn prepare(
    f: Program,
    packing: super::super::lockstep::Packing,
    num_vgprs: usize,
) -> Prepared {
    let Program {
        mut ir,
        parameter_inputs: inputs,
        registry,
    } = f;
    ir.lowered_to_packets();
    let driver = Driver::new();
    let mut an = Analyses::new(Context {
        exec_initial: true,
        packet: Some(Packet {
            aligned: packing.aligned,
        }),
        ..Context::of(&registry, &inputs, packing.lanes)
    });
    let limit = 1 + ir.types.len();
    driver
        .fixpoint(&mut ir, &mut an, "simplify", limit, &[&Simplify, &Dce])
        .unwrap();
    driver.pipeline(&mut ir, &mut an, &[&PacketState]).unwrap();
    if std::env::var("AMDGPU_SIM_PAIRS").map_or(true, |v| v != "0") {
        driver
            .pipeline(&mut ir, &mut an, &[&Simplify, &Dce])
            .unwrap();
        driver.pipeline(&mut ir, &mut an, &[&Pairs]).unwrap();
        let limit = 1 + ir.types.len();
        driver
            .fixpoint(
                &mut ir,
                &mut an,
                "simplify",
                limit,
                &[&Simplify, &Dce, &DeadParams, &WideMemory],
            )
            .unwrap();
    }
    driver.pipeline(&mut ir, &mut an, &[&Adjacency]).unwrap();
    let constants = an.get::<Constants>(&ir);
    let uniformity = an.get::<Uniformity>(&ir);
    let accesses = an.get::<Accesses>(&ir);
    let holds_a_lane = holds_a_lane(&ir, &an, &accesses);
    let uniform = uniformity.uniform();
    let affine: BTreeMap<ValueId, u32> = uniformity
        .facts
        .iter()
        .enumerate()
        .filter_map(|(v, fact)| match *fact {
            Fact::Affine { stride, .. }
                if stride > 0 && stride <= 256 && ir.types[v] == Ty::I64 =>
            {
                Some((ValueId(v), stride as u32))
            }
            _ => None,
        })
        .collect();
    let (yields, groups) = super::yields::layouts(&ir, &uniform, &constants);
    let shapes = accesses
        .iter()
        .map(|a| {
            super::memory::shape(
                a,
                packing.lanes,
                super::memory::global_load(a, &uniform, &affine),
                &constants,
            )
        })
        .collect();
    let clusters =
        super::memory::clusters(&ir, &accesses, packing.lanes, &uniform, &affine);
    let min_private_bytes = accesses
        .iter()
        .filter_map(|a| a.static_scratch_end(&constants))
        .max()
        .unwrap_or(0) as usize;
    let ir = ir
        .verify_with(&registry)
        .expect("invalid prepared function SSA");
    Prepared {
        registry,
        ir,
        inputs,
        width: packing.lanes,
        uniform,
        holds_a_lane,
        accesses,
        shapes,
        clusters,
        yields,
        groups,
        min_private_bytes,
        num_vgprs,
    }
}
