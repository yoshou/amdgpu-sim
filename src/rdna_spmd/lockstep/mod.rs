mod cost;
mod emit;
mod localize;
mod mask;
mod structure;
mod uniform;

use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::compiler::exec_index;
use crate::rdna_spmd::decompile::Lane;
use crate::rdna_spmd::program::LiftedFunction;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Packing {

    pub lanes: u32,

    pub aligned: bool,
}

fn represent(lane: &LiftedFunction, packing: Packing) -> LiftedFunction {
    use crate::rdna_spmd::analysis::{Analyses, MaskValues};
    use crate::rdna_spmd::pass::dce::{Dce, DeadParams};
    use crate::rdna_spmd::pass::pairs::{Pairs, WideMemory};
    use crate::rdna_spmd::pass::simplify::Simplify;
    use crate::rdna_spmd::pass::Driver;
    use std::marker::PhantomData;
    let mut ir = lane.ir.clone();
    let driver = Driver::new();
    let mut analyses = Analyses::new(uniform::context(lane, packing));
    driver
        .pipeline(
            &mut ir,
            &mut analyses,
            &[&Simplify, &Dce, &Pairs::<MaskValues>(PhantomData)],
        )
        .unwrap();
    let limit = 1 + ir.types.len();
    driver
        .fixpoint(
            &mut ir,
            &mut analyses,
            "represent",
            limit,
            &[
                &Simplify,
                &Dce,
                &DeadParams::<MaskValues>(PhantomData),
                &WideMemory,
            ],
        )
        .unwrap();
    LiftedFunction {
        registry: lane.registry.clone(),
        ir,
        parameter_inputs: lane.parameter_inputs.clone(),
        revision: lane.revision + 1,
    }
}

pub(crate) fn lockstep(lane: &Lane, packing: Packing) -> LiftedFunction {
    let everyone = &lane.everyone;
    let lane = &represent(&lane.function, packing);
    let facts = Facts::new(&lane.ir, &lane.parameter_inputs, &Default::default());
    let shape = structure::Structure::new(&lane.ir, facts);
    let costs = cost::Costs::new(packing.lanes, &crate::rdna_spmd::host::Vectors::detect());
    let exec = exec_index(&lane.parameter_inputs, &lane.registry);
    let mut uniform = uniform::assume(lane, packing);
    loop {
        let lowered = emit::lower(
            &lane.ir,
            &shape,
            &lane.registry,
            &uniform,
            exec,
            &costs,
            everyone,
        );
        let mut ir = lowered.ir;
        let folded = localize::fold_forwarding(&mut ir);
        localize::localize(&mut ir);
        let (refuted, derived) = uniform::refuted(lane, packing, &ir, &lowered.masks, &folded);
        if refuted.is_empty() {
            assert!(
                !derived,
                "the lowering took a bit it made from uniform bits to vary"
            );

            ir.compact();
            ir.enter_regions();
            if let Err(e) = ir.check(&lane.registry) {
                panic!(
                    "the lockstep lowering built an invalid packet program: {}",
                    e
                );
            }
            return LiftedFunction {
                registry: lane.registry.clone(),
                ir,
                parameter_inputs: lane.parameter_inputs.clone(),
                revision: lane.revision + 1,
            };
        }
        for v in refuted {
            uniform[v.0] = false;
        }
    }
}
