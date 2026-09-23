use super::check::{Check, Mode};
use super::logic::{choices, Atom, Kept, Logic};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

pub fn prove(f: &Func, inputs: &[Parameter], exec_index: Option<usize>) -> (Kept, BTreeSet<u64>) {
    let base = Facts::new(f, inputs, &BTreeSet::new());
    let loops = Loops::new(f, &base).unwrap_or_else(|block| {
        panic!(
            "b{}: control flow enters a cycle other than through its header",
            block.0
        )
    });
    let listed = choices(f, &base);
    let order: HashMap<ValueId, usize> = listed
        .iter()
        .enumerate()
        .map(|(i, &(v, _))| (v, i))
        .collect();
    let mut kept = Kept::default();
    loop {
        let facts = Facts::new(f, inputs, &kept.words);
        let logic = Logic::fixed(f, &facts, &kept.queries, &listed);
        let mut check = Check::new(f, &facts, inputs, exec_index, &loops, logic, Mode::Search);
        if check.run() {
            let everyone = check.everyone(&BTreeSet::new());
            return (kept, everyone);
        }
        let mut blamed = BTreeSet::new();
        for i in 0..check.violations.len() {
            let condition = check.violations[i].condition;
            if let Some(v) = named(&mut check.logic, condition, &order) {
                blamed.insert(v);
            }
        }
        if blamed.is_empty() {
            let v = &check.violations[0];
            panic!(
                "b{}:{}: {} under every conversion policy",
                v.block.0, v.index, v.reason
            );
        }
        for v in blamed {
            if listed[order[&v]].1 {
                kept.words.insert(v);
            } else {
                kept.queries.insert(v);
            }
        }
    }
}

fn named(logic: &mut Logic, condition: Bdd, order: &HashMap<ValueId, usize>) -> Option<ValueId> {
    let mut marks: Vec<(usize, u32, ValueId)> = logic
        .support(condition)
        .iter()
        .filter_map(|&var| match logic.atom_of(var) {
            Atom::Marker(v) => Some((order[&v], var, v)),
            _ => None,
        })
        .collect();
    marks.sort_unstable();
    marks
        .into_iter()
        .find(|&(_, var, _)| logic.m.cofactor(condition, var, false) != condition)
        .map(|(_, _, v)| v)
}
