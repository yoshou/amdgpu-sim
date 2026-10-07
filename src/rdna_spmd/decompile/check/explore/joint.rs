use super::super::super::logic::{Atom, Logic, PATH};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeMap;

pub(super) const WAVE: usize = 0;
pub(super) const LANE: usize = 1;
pub(super) const JOINT: usize = 2;
pub(super) const COPY: usize = 4;
pub(super) const PATHS: u32 = 1 << 15;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Desc {
    pub(super) same: Option<usize>,
    pub(super) bits: Option<Bdd>,
}

pub(super) type Sides = [Vec<Desc>; 2];

pub(super) type Key = [Option<BlockId>; 2];

#[derive(Default)]
pub(super) struct Fresh(HashMap<Bdd, bool>);

impl Fresh {
    pub(super) fn within(&mut self, logic: &mut Logic, g: Bdd) -> bool {
        if let Some(&f) = self.0.get(&g) {
            return f;
        }
        let support = logic.support(g);
        let f = support
            .iter()
            .any(|&v| matches!(logic.atom_of(v), Atom::Fresh(..)));
        self.0.insert(g, f);
        f
    }
}

pub(super) fn fresh_support(logic: &mut Logic, g: Bdd) -> Vec<u32> {
    let support = logic.support(g);
    support
        .iter()
        .copied()
        .filter(|&v| matches!(logic.atom_of(v), Atom::Fresh(..)))
        .collect()
}

pub(super) fn canonical(logic: &mut Logic, fresh: &mut Fresh, cond: Bdd, sides: Sides) -> (Bdd, Sides) {
    let mut order: Vec<u32> = Vec::new();
    for g in sides.iter().flatten().filter_map(|d| d.bits) {
        if fresh.within(logic, g) {
            for v in fresh_support(logic, g) {
                if !order.contains(&v) {
                    order.push(v);
                }
            }
        }
    }
    let absent: Vec<u32> = fresh_support(logic, cond)
        .into_iter()
        .filter(|v| !order.contains(v))
        .collect();
    let cond = logic.exists(&absent, cond);
    let (paths, values): (Vec<u32>, Vec<u32>) = order
        .iter()
        .partition(|&&v| matches!(logic.atom_of(v), Atom::Fresh(PATH, ..)));
    assert!(paths.len() < PATHS as usize, "too many paths");
    let mut map: HashMap<u32, Bdd> = HashMap::default();
    let targets = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, JOINT, i as u32 + 1))
        .chain(paths.iter().enumerate().map(|(i, &v)| (v, PATH, PATHS + i as u32)));
    for (v, kind, position) in targets.collect::<Vec<_>>() {
        let atom = Atom::Fresh(kind, ValueId(0), position);
        if logic.atom_of(v) != atom {
            let target = logic.atom(atom);
            map.insert(v, target);
        }
    }
    let Some(&last) = map.keys().max() else {
        return (cond, sides);
    };
    let mut roots: Vec<Bdd> = sides.iter().flatten().filter_map(|d| d.bits).collect();
    roots.push(cond);
    let renamed = logic.m.compose_many(&roots, &|v| map.get(&v).copied(), last);
    let mut renamed = renamed.into_iter();
    let mut out = sides;
    for d in out.iter_mut().flatten() {
        if d.bits.is_some() {
            d.bits = renamed.next();
        }
    }
    (renamed.next().unwrap(), out)
}

pub(super) fn join(logic: &mut Logic, paths: &mut u32, (oc, old): (Bdd, &Sides), (nc, new): (Bdd, &Sides)) -> (Bdd, Sides) {
    let mut out = old.clone();
    let differ = old.iter().flatten().zip(new.iter().flatten()).any(|(o, n)| o.bits != n.bits);
    if !differ {
        for (d, n) in out.iter_mut().flatten().zip(new.iter().flatten()) {
            if d.same != n.same {
                d.same = None;
            }
        }
        return (logic.m.or(oc, nc), out);
    }
    assert!(*paths > 0, "too many joins");
    *paths -= 1;
    let path = logic.atom(Atom::Fresh(PATH, ValueId(0), *paths));
    for (d, n) in out.iter_mut().flatten().zip(new.iter().flatten()) {
        let same = if d.same == n.same { d.same } else { None };
        let bits = match (d.bits, n.bits) {
            (Some(a), Some(b)) if a == b => Some(a),
            (Some(a), Some(b)) => Some(logic.m.ite(path, a, b)),
            _ => None,
        };
        *d = Desc { same, bits };
    }
    (logic.m.ite(path, oc, nc), out)
}

pub(super) fn joined(logic: &mut Logic, paths: &mut u32, entries: &BTreeMap<(Key, usize), (Bdd, Sides)>) -> (Bdd, Sides) {
    let mut values = entries.values();
    let (c, s) = values.next().unwrap();
    let mut joined = (*c, s.clone());
    for (c, s) in values {
        joined = join(logic, paths, (joined.0, &joined.1), (*c, s));
    }
    joined
}

pub(super) fn between(logic: &mut Logic, fresh: &mut Fresh, side: usize, param: ValueId, a: Bdd, b: Bdd) -> Bdd {
    let mut unknown = Vec::new();
    for g in [a, b] {
        if fresh.within(logic, g) {
            for v in fresh_support(logic, g) {
                if !unknown.contains(&v) {
                    unknown.push(v);
                }
            }
        }
    }
    let both = logic.m.and(a, b);
    let either = logic.m.or(a, b);
    let low = logic.forall(&unknown, both);
    let high = logic.exists(&unknown, either);
    if low == Bdd::FALSE && high == Bdd::TRUE {
        return logic.atom(Atom::Fresh(side, param, 0));
    }
    let choice = logic.atom(Atom::Fresh(side, param, u32::MAX));
    let open = logic.m.and(choice, high);
    logic.m.or(low, open)
}

#[cfg(test)]
mod tests {
    use super::super::super::super::testing::*;
    use super::*;
    use crate::rdna_spmd::analysis::facts::Facts;
    use std::collections::BTreeSet;

    fn with_logic(test: impl FnOnce(&mut Logic)) {
        let (b, _) = Build::kernel();
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(&b.f, &facts, &BTreeSet::new(), &[]);
        test(&mut logic);
    }

    fn random_state(logic: &mut Logic, r: &mut Random, context: &[Atom], unknowns: &[Atom], paths: &[Atom]) -> (Bdd, Sides) {
        let tied: Vec<Atom> = context.iter().chain(paths).copied().collect();
        let all: Vec<Atom> = tied.iter().chain(unknowns).copied().collect();
        let cond = random_function(logic, r, &tied, 3);
        let desc = |logic: &mut Logic, r: &mut Random| Desc {
            same: None,
            bits: Some(random_function(logic, r, &all, 4)),
        };
        let wave = vec![desc(logic, r), desc(logic, r)];
        let lane = vec![desc(logic, r)];
        (cond, [wave, lane])
    }

    fn held(logic: &mut Logic, (cond, sides): (Bdd, &Sides), context: &[(u32, bool)]) -> BTreeSet<Vec<bool>> {
        let roots: Vec<Bdd> = sides.iter().flatten().filter_map(|d| d.bits).collect();
        let mut hidden: Vec<u32> = Vec::new();
        for &g in roots.iter().chain([cond].iter()) {
            for &v in logic.support(g).iter() {
                if matches!(logic.atom_of(v), Atom::Fresh(..)) && !hidden.contains(&v) {
                    hidden.push(v);
                }
            }
        }
        let mut out = BTreeSet::new();
        for row in 0..1u64 << hidden.len() {
            let value = |var: u32| match hidden.iter().position(|&h| h == var) {
                Some(i) => row >> i & 1 == 1,
                None => context.iter().find(|&&(v, _)| v == var).unwrap().1,
            };
            if evaluate(&logic.m, cond, &value) {
                out.insert(roots.iter().map(|&g| evaluate(&logic.m, g, &value)).collect());
            }
        }
        out
    }

    fn contexts(logic: &mut Logic, context: &[Atom]) -> Vec<Vec<(u32, bool)>> {
        let vars: Vec<u32> = context.iter().map(|&a| variable(logic, a)).collect();
        (0..1u32 << vars.len())
            .map(|row| vars.iter().enumerate().map(|(i, &v)| (v, row >> i & 1 == 1)).collect())
            .collect()
    }

    #[test]
    fn between_ranges_over_the_bounds_both_arrivals_share_for_every_unknown() {
        with_logic(|logic| {
            let mut r = Random::new(53);
            let mut fresh = Fresh::default();
            let context = [Atom::Lane(0), Atom::Lane(1), Atom::Lane(2)];
            let unknowns = [Atom::Fresh(WAVE, ValueId(1000), 0), Atom::Fresh(JOINT, ValueId(0), 1), Atom::Fresh(JOINT, ValueId(0), 2)];
            let all: Vec<Atom> = context.iter().chain(&unknowns).copied().collect();
            let unknown_vars: Vec<u32> = unknowns.iter().map(|&a| variable(logic, a)).collect();
            let contexts = contexts(logic, &context);
            let mut wrong = Vec::new();
            for trial in 0..300 {
                let a = random_function(logic, &mut r, if trial % 3 == 0 { &context[..] } else { &all[..] }, 4);
                let b = random_function(logic, &mut r, &all, 4);
                let merged = between(logic, &mut fresh, LANE, ValueId(77), a, b);
                let own = fresh_support(logic, merged);
                if own.iter().any(|&v| !matches!(logic.atom_of(v), Atom::Fresh(LANE, ValueId(77), _))) {
                    wrong.push(format!("trial {} keeps an arrival's unknowns", trial));
                    continue;
                }
                for context in &contexts {
                    let (mut low, mut high) = (true, false);
                    for hidden in 0..1u32 << unknown_vars.len() {
                        let value = |var: u32| match unknown_vars.iter().position(|&v| v == var) {
                            Some(i) => hidden >> i & 1 == 1,
                            None => context.iter().find(|&&(v, _)| v == var).unwrap().1,
                        };
                        let (x, y) = (evaluate(&logic.m, a, &value), evaluate(&logic.m, b, &value));
                        low &= x && y;
                        high |= x || y;
                    }
                    let mut taken = BTreeSet::new();
                    for choice in 0..1u32 << own.len() {
                        let value = |var: u32| match own.iter().position(|&v| v == var) {
                            Some(i) => choice >> i & 1 == 1,
                            None => context.iter().find(|&&(v, _)| v == var).unwrap().1,
                        };
                        taken.insert(evaluate(&logic.m, merged, &value));
                    }
                    if taken != BTreeSet::from([low, high]) {
                        wrong.push(format!("trial {} takes {:?} under {:?}, bounds {} {}", trial, taken, context, low, high));
                    }
                }
            }
            assert!(wrong.is_empty(), "the merge must range over exactly the shared bounds: {:?}", &wrong[..wrong.len().min(8)]);
        });
    }

    #[test]
    fn join_holds_exactly_the_states_of_every_arrival() {
        with_logic(|logic| {
            let mut r = Random::new(43);
            let context = [Atom::Lane(0), Atom::Lane(1)];
            let unknowns = [Atom::Fresh(WAVE, ValueId(1000), 0), Atom::Fresh(LANE, ValueId(1001), 0), Atom::Fresh(JOINT, ValueId(0), 1)];
            let mut left = PATHS - 8;
            let paths = [Atom::Fresh(PATH, ValueId(0), PATHS - 1), Atom::Fresh(PATH, ValueId(0), PATHS - 2), Atom::Fresh(PATH, ValueId(0), PATHS)];
            let contexts = contexts(logic, &context);
            let mut wrong = Vec::new();
            for trial in 0..300 {
                let mut arrivals = vec![random_state(logic, &mut r, &context, &unknowns, &paths)];
                for _ in 0..2 {
                    let mut next = random_state(logic, &mut r, &context, &unknowns, &paths);
                    let first = arrivals[0].1.clone();
                    match r.below(4) {
                        0 => next.1 = first,
                        1 => next.1[WAVE][0] = first[WAVE][0],
                        _ => {}
                    }
                    arrivals.push(next);
                }
                let two = join(logic, &mut left, (arrivals[0].0, &arrivals[0].1), (arrivals[1].0, &arrivals[1].1));
                let entries: BTreeMap<(Key, usize), (Bdd, Sides)> =
                    arrivals.iter().enumerate().map(|(i, a)| (([None, None], i), a.clone())).collect();
                let three = joined(logic, &mut left, &entries);
                for context in &contexts {
                    let each: Vec<BTreeSet<Vec<bool>>> = arrivals.iter().map(|a| held(logic, (a.0, &a.1), context)).collect();
                    let expected_two: BTreeSet<Vec<bool>> = each[0].union(&each[1]).cloned().collect();
                    let expected_three: BTreeSet<Vec<bool>> = expected_two.union(&each[2]).cloned().collect();
                    if held(logic, (two.0, &two.1), context) != expected_two {
                        wrong.push(format!("trial {} two arrivals under {:?}", trial, context));
                    }
                    if held(logic, (three.0, &three.1), context) != expected_three {
                        wrong.push(format!("trial {} three arrivals under {:?}", trial, context));
                    }
                }
            }
            assert!(wrong.is_empty(), "the join must hold the states of the arrivals and nothing else: {:?}", &wrong[..wrong.len().min(8)]);
        });
    }

    #[test]
    fn canonical_names_hold_the_same_states_in_normal_form() {
        with_logic(|logic| {
            let mut r = Random::new(47);
            let mut fresh = Fresh::default();
            let context = [Atom::Lane(0), Atom::Lane(1)];
            let unknowns = [
                Atom::Fresh(WAVE, ValueId(1000), 0),
                Atom::Fresh(LANE, ValueId(1001), 0),
                Atom::Fresh(WAVE, ValueId(1002), u32::MAX),
                Atom::Fresh(JOINT, ValueId(0), 3),
                Atom::Fresh(JOINT, ValueId(0), 1),
            ];
            let paths = [Atom::Fresh(PATH, ValueId(0), PATHS - 1), Atom::Fresh(PATH, ValueId(0), PATHS + 2), Atom::Fresh(PATH, ValueId(0), 5)];
            for position in 1..=unknowns.len() as u32 {
                logic.atom(Atom::Fresh(JOINT, ValueId(0), position));
            }
            let contexts = contexts(logic, &context);
            let mut wrong = Vec::new();
            for trial in 0..300 {
                let (cond, sides) = random_state(logic, &mut r, &context, &unknowns, &paths);
                let (cc, cs) = canonical(logic, &mut fresh, cond, sides.clone());
                for context in &contexts {
                    if held(logic, (cc, &cs), context) != held(logic, (cond, &sides), context) {
                        wrong.push(format!("trial {} changes the states under {:?}", trial, context));
                    }
                }
                let mut names: Vec<Atom> = Vec::new();
                for g in cs.iter().flatten().filter_map(|d| d.bits).chain([cc]) {
                    for v in fresh_support(logic, g) {
                        let a = logic.atom_of(v);
                        if !names.contains(&a) {
                            names.push(a);
                        }
                    }
                }
                let values = names.iter().filter(|a| matches!(a, Atom::Fresh(JOINT, ..))).count() as u32;
                let tied = names.iter().filter(|a| matches!(a, Atom::Fresh(PATH, ..))).count() as u32;
                let normal = names.iter().all(|a| match *a {
                    Atom::Fresh(JOINT, ValueId(0), p) => (1..=values).contains(&p),
                    Atom::Fresh(PATH, ValueId(0), p) => (PATHS..PATHS + tied).contains(&p),
                    _ => false,
                });
                if !normal {
                    wrong.push(format!("trial {} leaves names {:?}", trial, names));
                }
                if canonical(logic, &mut fresh, cc, cs.clone()) != (cc, cs) {
                    wrong.push(format!("trial {} is not settled by one pass", trial));
                }
            }
            assert!(wrong.is_empty(), "renaming must keep the states and settle the names: {:?}", &wrong[..wrong.len().min(8)]);
        });
    }
}
