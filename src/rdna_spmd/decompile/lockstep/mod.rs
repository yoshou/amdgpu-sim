//! Running lane programs in lockstep, the way ISPC runs SPMD program
//! instances on SIMD hardware.
//!
//! A lane program describes one lane. The packet program built from it runs
//! a packet of lanes together: each operation once for all of them, each
//! control decision as a mask of the lanes it holds for.
//!
//! - Control flow the lanes disagree on is *varying*. Both sides run, each
//!   under its own mask; effects happen only for the lanes in the mask, and
//!   the values the two sides computed meet in a choice where the sides merge.
//!   The choice tests only what tells the arriving lanes apart, so the merge
//!   after a branch chooses on the branch's condition.
//! - Control flow every lane agrees on is *uniform*: a query over a mask whose
//!   only varying part is known to hold a lane is a scalar test, and the packet
//!   program branches on it as a single program would, keeping the values
//!   either side computes uniform.
//! - A span of work that an empty mask would waste is skipped by a query over
//!   its mask, as ISPC tests `any` before the arms of a varying `if`, where
//!   the operations the span is expected to save outweigh the ones the query
//!   costs.
//! - A loop carries the mask of the lanes still in it and goes around while the
//!   mask holds a lane. What a lane carries out is kept from the iteration it
//!   left on, the way a masked store keeps a variable; a loop no lane can leave
//!   while another stays carries nothing extra, and one whose trip count is
//!   uniform is an ordinary loop.
//!
//! [`structure`] fixes the order the regions are walked in, [`mask`] is the
//! algebra of masks, [`cost`] counts the operations a span and a query
//! perform, [`emit`] lowers, [`localize`] puts the result in the IR's
//! block-local form, and [`uniform`] settles which bits are uniform.

mod cost;
mod emit;
mod localize;
mod mask;
mod structure;
#[cfg(test)]
mod tests;
mod uniform;

use super::facts::Facts;
use super::Refusal;
use crate::rdna_spmd::compiler::exec_index;
use crate::rdna_spmd::program::LiftedFunction;

/// How the packets a program is lowered for are laid out.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Packing {
    /// Lanes in a packet.
    pub lanes: u32,
    /// Whether every packet lies within one row of a work group, which makes
    /// the row part of a work-item id uniform.
    pub aligned: bool,
}

/// Settles how the lane program represents its values before it is lowered:
/// a 64-bit value a lane carries as two words becomes one value, so the
/// packet program chooses between whole values rather than between halves.
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

pub(crate) fn lockstep(lane: &super::Lane, packing: Packing) -> Result<LiftedFunction, Refusal> {
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
        )?;
        let mut ir = lowered.ir;
        let folded = localize::fold_forwarding(&mut ir);
        localize::localize(&mut ir);
        let (refuted, derived) = uniform::refuted(lane, packing, &ir, &lowered.masks, &folded);
        if refuted.is_empty() {
            assert!(
                !derived,
                "the lowering took a bit it made from uniform bits to vary"
            );
            // Folded parameters left their values behind; renumbering drops
            // them, now that nothing needs the lowering's numbering.
            ir.compact();
            if let Err(e) = ir.check(&lane.registry) {
                panic!(
                    "the lockstep lowering built an invalid packet program: {}",
                    e
                );
            }
            return Ok(LiftedFunction {
                registry: lane.registry.clone(),
                ir,
                parameter_inputs: lane.parameter_inputs.clone(),
                revision: lane.revision + 1,
            });
        }
        for v in refuted {
            uniform[v.0] = false;
        }
    }
}
