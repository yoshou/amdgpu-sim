use super::check::{Check, Mode};
use super::logic::{choices, Kept, Logic};
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

pub fn prove(f: &Func, inputs: &[Parameter], exec_index: Option<usize>) -> (Kept, BTreeSet<u64>) {
    let facts = Facts::new(f, inputs, &BTreeSet::new());
    let loops = Loops::new(f, &facts).unwrap_or_else(|block| {
        panic!(
            "b{}: control flow enters a cycle other than through its header",
            block.0
        )
    });
    let listed = choices(f, &facts);
    let logic = Logic::open(f, &facts, &listed);
    let mut check = Check::new(f, &facts, inputs, exec_index, &loops, logic, Mode::Direct);
    if !check.run() {
        let (block, index, reason) = check.exhausted.unwrap();
        panic!(
            "b{}:{}: {} under every conversion policy",
            block.0, index, reason
        );
    }
    let kept = check.logic.choose(check.safe);
    let disabled: BTreeSet<ValueId> = kept.queries.iter().chain(&kept.words).copied().collect();
    let everyone = check.everyone(&disabled);
    (kept, everyone)
}
