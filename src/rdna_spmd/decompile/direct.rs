use super::check::Check;
use super::hazard::{Hazards, Judge};
use super::logic::{Kept, Logic};
use super::search::listed;
use crate::rdna_spmd::analysis::facts::{Conversion, Facts};
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

pub fn prove(
    f: &Func,
    inputs: &[Parameter],
    exec_index: Option<usize>,
    hazards: &Hazards,
    judge: &mut Judge,
) -> (Kept, BTreeSet<u64>) {
    let facts = Facts::converted(f, inputs, &BTreeSet::new(), Conversion::Open);
    let loops = Loops::new(f, &facts).unwrap_or_else(|block| {
        panic!(
            "b{}: control flow enters a cycle other than through its header",
            block.0
        )
    });
    let listed = listed(f, &facts, hazards);
    loop {
        let known = judge.known().clone();
        let logic = Logic::open(f, &facts, &listed);
        let mut check = Check::new(f, &facts, inputs, exec_index, &loops, hazards, Some(&known), logic);
        let proved = check.run();
        let pending = check.unresolved();
        if !pending.is_empty() {
            judge.judge(&pending);
            continue;
        }
        if !proved {
            let (block, index, reason) = check.exhausted().unwrap();
            panic!(
                "b{}:{}: {} under every conversion policy",
                block.0, index, reason
            );
        }
        let safe = check.safe();
        let kept = check.logic().choose(safe);
        let everyone = check.everyone(&kept.choices());
        return (kept, everyone);
    }
}
