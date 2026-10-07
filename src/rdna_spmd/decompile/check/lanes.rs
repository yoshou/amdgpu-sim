use super::super::logic::Logic;
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;

pub(super) fn some_lane(logic: &mut Logic, facts: &Facts, x: Bdd) -> Bdd {
    let x = logic.consistent(x);
    let varying: Vec<u32> = logic
        .support(x)
        .iter()
        .copied()
        .filter(|&n| !logic.uniform_atom(facts, n))
        .collect();
    logic.exists(&varying, x)
}

pub(super) fn at_lane(logic: &mut Logic, x: Bdd, lane: u32) -> Bdd {
    let x = logic.consistent(x);
    logic.at_lane(x, lane)
}

pub(super) fn read_from(logic: &mut Logic, facts: &Facts, h: Bdd, source: impl Fn(usize) -> u32) -> Bdd {
    let mut seen: HashMap<u32, Bdd> = HashMap::default();
    let mut any = Bdd::FALSE;
    for l in 0..logic.lane_count() as usize {
        let s = source(l);
        let there = match seen.get(&s) {
            Some(&there) => there,
            None => {
                let at = at_lane(logic, h, s);
                let there = some_lane(logic, facts, at);
                seen.insert(s, there);
                there
            }
        };
        let here = logic.lanes(|x| x == l as u32);
        let reads = logic.m.and(here, there);
        any = logic.m.or(any, reads);
    }
    any
}

pub(super) fn read_from_any(logic: &mut Logic, facts: &Facts, h: Bdd, sources: impl Fn(usize) -> u64) -> Bdd {
    let mut seen: HashMap<u64, Bdd> = HashMap::default();
    let mut any = Bdd::FALSE;
    let count = logic.lane_count();
    for l in 0..count as usize {
        let set = sources(l);
        let there = match seen.get(&set) {
            Some(&there) => there,
            None => {
                let mut there = Bdd::FALSE;
                for s in 0..count {
                    if set >> s & 1 == 1 {
                        let at = at_lane(logic, h, s);
                        let read = some_lane(logic, facts, at);
                        there = logic.m.or(there, read);
                    }
                }
                seen.insert(set, there);
                there
            }
        };
        let here = logic.lanes(|x| x == l as u32);
        let reads = logic.m.and(here, there);
        any = logic.m.or(any, reads);
    }
    any
}

fn other_lanes(logic: &mut Logic, facts: &Facts, x: Bdd) -> Bdd {
    if !logic.lane_dependent(x) {
        return some_lane(logic, facts, x);
    }
    let mut any = Bdd::FALSE;
    for l in 0..logic.lane_count() {
        let at = at_lane(logic, x, l);
        let there = some_lane(logic, facts, at);
        if there == Bdd::FALSE {
            continue;
        }
        let elsewhere = logic.lanes(|y| y != l);
        let reads = logic.m.and(elsewhere, there);
        any = logic.m.or(any, reads);
    }
    any
}

pub(super) fn query(logic: &mut Logic, facts: &Facts, x: Bdd, hx: Bdd, tag: Bdd) -> Bdd {
    let whole = logic.m.or(x, hx);
    let others = other_lanes(logic, facts, whole);
    let absent = logic.m.not(x);
    let differs = logic.m.and(absent, others);
    let differs = logic.m.and(differs, tag);
    if differs == Bdd::FALSE {
        return hx;
    }
    logic.m.or(hx, differs)
}

pub(super) fn absorbs(logic: &mut Logic, fx: Bdd, hx: Bdd, conjunction: bool) -> Bdd {
    let value = if conjunction { logic.m.not(fx) } else { fx };
    let settled = logic.m.not(hx);
    logic.m.and(value, settled)
}
