use super::check::Check;
use super::hazard::Hazards;
use super::logic::{choices, Atom, Choice, Kept, Logic};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

pub fn prove(
    f: &Func,
    inputs: &[Parameter],
    exec_index: Option<usize>,
    hazards: &Hazards,
) -> (Kept, BTreeSet<u64>) {
    let base = Facts::new(f, inputs, &BTreeSet::new());
    let loops = Loops::new(f, &base).unwrap_or_else(|block| {
        panic!(
            "b{}: control flow enters a cycle other than through its header",
            block.0
        )
    });
    let listed = listed(f, &base, hazards);
    let order: HashMap<Choice, usize> = listed
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let mut kept = Kept::default();
    loop {
        let facts = Facts::new(f, inputs, &kept.words);
        let logic = Logic::fixed(f, &facts, &kept.choices(), &listed);
        let mut check = Check::new(f, &facts, inputs, exec_index, &loops, hazards, logic);
        if check.run() {
            let everyone = check.everyone(&BTreeSet::new());
            return (kept, everyone);
        }
        let mut blamed = BTreeSet::new();
        for i in 0..check.violations.len() {
            let condition = check.violations[i].condition;
            if let Some(v) = named(&mut check.logic, condition, &order) {
                if std::env::var_os("AMDGPU_SIM_PRINT_MEETINGS").is_some() {
                    let violation = &check.violations[i];
                    eprintln!(
                        "; keeps {:?} for b{}:{}: {}",
                        v, violation.block.0, violation.index, violation.reason
                    );
                }
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
        for c in blamed {
            kept.insert(c);
        }
    }
}

pub fn listed(f: &Func, facts: &Facts, hazards: &Hazards) -> Vec<Choice> {
    let mut listed = choices(f, facts);
    listed.extend((0..hazards.meetings.len()).rev().map(Choice::Meet));
    listed
}

fn named(logic: &mut Logic, condition: Bdd, order: &HashMap<Choice, usize>) -> Option<Choice> {
    let mut marks: Vec<(usize, u32, Choice)> = logic
        .support(condition)
        .iter()
        .filter_map(|&var| match logic.atom_of(var) {
            Atom::Marker(c) => Some((order[&c], var, c)),
            _ => None,
        })
        .collect();
    marks.sort_unstable();
    marks
        .into_iter()
        .find(|&(_, var, _)| logic.m.cofactor(condition, var, false) != condition)
        .map(|(_, _, v)| v)
}
