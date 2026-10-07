use super::super::super::logic::{Atom, PATH};
use super::super::queries::Queries;
use super::joint::{COPY, WAVE};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) struct Decisions {
    assume: Bdd,
    unreliable: HashMap<u32, Bdd>,
    decided: HashMap<(Bdd, Bdd, usize), Option<bool>>,
}

impl Decisions {
    pub(super) fn new(assume: Bdd) -> Self {
        Self {
            assume,
            unreliable: HashMap::default(),
            decided: HashMap::default(),
        }
    }

    #[inline]
    pub(super) fn assume(&self) -> Bdd {
        self.assume
    }

    pub(super) fn reliable<'a, Q: Queries<'a>>(&self, q: &mut Q, leaves: Bdd) -> bool {
        if leaves == Bdd::FALSE {
            return true;
        }
        let assume = q.and(self.assume, q.safe());
        q.and(assume, leaves) == Bdd::FALSE
    }

    fn unreliable<'a, Q: Queries<'a>>(&mut self, q: &mut Q, var: u32) -> Bdd {
        if let Some(&u) = self.unreliable.get(&var) {
            return u;
        }
        let differs = |q: &mut Q, values: &[ValueId]| {
            values.iter().fold(Bdd::FALSE, |u, &v| {
                let h = q.h(v);
                q.or(u, h)
            })
        };
        let u = match q.logic().atom_of(var) {
            Atom::Bit(v) | Atom::View(v) | Atom::WordBit(v, _) => differs(q, &[v]),
            Atom::Cell(block, group, _) => {
                let leaves: Vec<ValueId> = q.logic().cell_leaves(block, group).to_vec();
                differs(q, &leaves)
            }
            Atom::Some(block, _) => {
                let sources = q.logic().answer_sources(block);
                differs(q, &sources)
            }
            _ => Bdd::FALSE,
        };
        let u = if q.and(self.assume, u) == Bdd::FALSE { Bdd::FALSE } else { u };
        self.unreliable.insert(var, u);
        u
    }

    fn unknowns<'a, Q: Queries<'a>>(&mut self, q: &mut Q, g: Bdd, side: usize, forms: bool) -> Vec<(u32, Bdd)> {
        let support = q.logic().support(g);
        support
            .iter()
            .filter_map(|&v| {
                let unknown = match q.logic().atom_of(v) {
                    Atom::Fresh(PATH, ..) => Bdd::FALSE,
                    Atom::Fresh(..) => Bdd::TRUE,
                    Atom::Term(..) if forms => Bdd::TRUE,
                    Atom::Term(..) => Bdd::FALSE,
                    _ if side == WAVE => self.unreliable(q, v),
                    _ => Bdd::FALSE,
                };
                (unknown != Bdd::FALSE).then_some((v, unknown))
            })
            .collect()
    }

    pub(super) fn decide<'a, Q: Queries<'a>>(&mut self, q: &mut Q, g: Bdd, cond: Bdd, side: usize) -> Option<bool> {
        if let Some(k) = g.constant() {
            return Some(k);
        }
        let cond = q.and(cond, q.safe());
        if let Some(&d) = self.decided.get(&(g, cond, side)) {
            return d;
        }
        let unknown = self.unknowns(q, g, side, true);
        let holds = quantify(q, &unknown, g, true);
        let d = if q.logic().m.implies(cond, holds) {
            Some(true)
        } else {
            let ng = q.logic().m.not(g);
            let fails = quantify(q, &unknown, ng, true);
            if q.logic().m.implies(cond, fails) {
                Some(false)
            } else {
                None
            }
        };
        self.decided.insert((g, cond, side), d);
        d
    }

    pub(super) fn weaken<'a, Q: Queries<'a>>(&mut self, q: &mut Q, g: Bdd, side: usize) -> Bdd {
        let unknown = self.unknowns(q, g, side, false);
        quantify(q, &unknown, g, false)
    }

    pub(super) fn varies<'a, Q: Queries<'a>>(&mut self, q: &mut Q, wave: Bdd, lane: Bdd) -> Bdd {
        let mut shared = Vec::new();
        let mut copies = Vec::new();
        let mut seen = wave;
        for (v, condition) in self.unknowns(q, wave, WAVE, true) {
            match q.logic().atom_of(v) {
                Atom::Fresh(..) | Atom::Term(..) => shared.push(v),
                _ => {
                    let copy = copy_of(q, v);
                    seen = rename(q, seen, v, copy);
                    copies.push((v, copy, condition));
                }
            }
        }
        let others = q.logic().support(lane);
        for &v in others.iter() {
            if matches!(q.logic().atom_of(v), Atom::Fresh(k, ..) if k != PATH) || matches!(q.logic().atom_of(v), Atom::Term(..)) {
                if !shared.contains(&v) {
                    shared.push(v);
                }
            }
        }
        let differ = q.logic().m.xor(seen, lane);
        let mut g = q.logic().exists(&shared, differ);
        for (v, copy, condition) in copies {
            let any = q.logic().exists(&[copy], g);
            let kept = rename(q, g, copy, v);
            g = q.logic().m.ite(condition, any, kept);
        }
        g
    }
}

fn rename<'a, Q: Queries<'a>>(q: &mut Q, g: Bdd, from: u32, to: u32) -> Bdd {
    let (high, low) = (q.logic().m.cofactor(g, from, true), q.logic().m.cofactor(g, from, false));
    let to = q.logic().m.var(to);
    q.logic().m.ite(to, high, low)
}

fn copy_of<'a, Q: Queries<'a>>(q: &mut Q, v: u32) -> u32 {
    let copy = q.logic().atom(Atom::Fresh(COPY, ValueId(v as usize), 0));
    q.logic().m.decompose(copy).expect("a variable").0
}

fn quantify<'a, Q: Queries<'a>>(q: &mut Q, unknown: &[(u32, Bdd)], g: Bdd, all: bool) -> Bdd {
    let over = |q: &mut Q, vars: &[u32], g: Bdd| {
        if all {
            q.logic().forall(vars, g)
        } else {
            q.logic().exists(vars, g)
        }
    };
    let everywhere: Vec<u32> = unknown.iter().filter(|&&(_, c)| c == Bdd::TRUE).map(|&(v, _)| v).collect();
    let mut copies = Vec::new();
    let mut g = g;
    for &(v, condition) in unknown.iter().filter(|&&(_, c)| c != Bdd::TRUE) {
        let copy = copy_of(q, v);
        g = rename(q, g, v, copy);
        copies.push((v, copy, condition));
    }
    g = over(q, &everywhere, g);
    for (v, copy, condition) in copies {
        let quantified = over(q, &[copy], g);
        let kept = rename(q, g, copy, v);
        g = q.logic().m.ite(condition, quantified, kept);
    }
    g
}

#[cfg(test)]
mod tests {
    use super::super::super::super::hazard::Hazards;
    use super::super::super::super::logic::Logic;
    use super::super::super::super::testing::*;
    use super::super::super::differences::Differences;
    use super::super::super::program::Program;
    use super::super::super::Mode;
    use super::super::joint::{COPY, JOINT, LANE, PATHS, WAVE};
    use super::*;
    use crate::rdna_spmd::analysis::facts::Facts;
    use crate::rdna_spmd::analysis::loops::Loops;
    use std::collections::BTreeSet;

    fn with_differences(test: impl FnOnce(&mut Differences)) {
        let (b, _) = Build::kernel();
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let loops = Loops::new(f, &facts).unwrap();
        let hazards = Hazards {
            accesses: Vec::new(),
            together: BTreeSet::new(),
            apart: BTreeSet::new(),
            idle: BTreeSet::new(),
            meetings: Vec::new(),
        };
        let logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let program = Program::new(f, &facts, &b.inputs, Some(0), &loops, &hazards);
        let mut differences = Differences::new(program, logic, Mode::Search);
        test(&mut differences);
    }

    #[test]
    fn decide_and_weaken_answer_over_the_paths_the_condition_allows() {
        with_differences(|q| {
            let mut decisions = Decisions::new(Bdd::TRUE);
            let mut r = Random::new(53);
            let context = [Atom::Lane(0), Atom::Lane(1)];
            let unknowns = [Atom::Fresh(LANE, ValueId(1001), 0), Atom::Fresh(JOINT, ValueId(0), 1)];
            let paths = [Atom::Fresh(PATH, ValueId(0), PATHS - 1), Atom::Fresh(PATH, ValueId(0), PATHS)];
            let tied: Vec<Atom> = context.iter().chain(&paths).copied().collect();
            let all: Vec<Atom> = tied.iter().chain(&unknowns).copied().collect();
            let tied_vars: Vec<u32> = tied.iter().map(|&a| variable(q.logic(), a)).collect();
            let unknown_vars: Vec<u32> = unknowns.iter().map(|&a| variable(q.logic(), a)).collect();
            let mut wrong = Vec::new();
            for trial in 0..400 {
                let cond = random_function(q.logic(), &mut r, &tied, 3);
                if cond == Bdd::FALSE {
                    continue;
                }
                let g = random_function(q.logic(), &mut r, &all, 4);
                let (mut every, mut none) = (true, true);
                let mut expected_weak = Vec::new();
                for row in 0..1u32 << tied_vars.len() {
                    let mut some = false;
                    for hidden in 0..1u32 << unknown_vars.len() {
                        let value = |var: u32| match tied_vars.iter().position(|&v| v == var) {
                            Some(i) => row >> i & 1 == 1,
                            None => hidden >> unknown_vars.iter().position(|&v| v == var).unwrap() & 1 == 1,
                        };
                        let holds = evaluate(&q.logic.m, g, &value);
                        some |= holds;
                        if evaluate(&q.logic.m, cond, &value) {
                            every &= holds;
                            none &= !holds;
                        }
                    }
                    expected_weak.push(some);
                }
                let expected = if every { Some(true) } else if none { Some(false) } else { None };
                if decisions.decide(q, g, cond, LANE) != expected {
                    wrong.push(format!("trial {} decides {:?}, expected {:?}", trial, decisions.decide(q, g, cond, LANE), expected));
                }
                let weak = decisions.weaken(q, g, LANE);
                for (row, &some) in expected_weak.iter().enumerate() {
                    let value = |var: u32| row >> tied_vars.iter().position(|&v| v == var).unwrap() & 1 == 1;
                    if evaluate(&q.logic.m, weak, &value) != some {
                        wrong.push(format!("trial {} weakens to {} at {}, expected {}", trial, !some, row, some));
                    }
                }
            }
            assert!(wrong.is_empty(), "decide and weaken must quantify the unknowns and keep the paths: {:?}", &wrong[..wrong.len().min(8)]);
        });
    }

    #[test]
    fn the_wave_side_leaves_an_atom_unknown_exactly_where_its_value_differs() {
        let mut wrong = Vec::new();
        let mut r = Random::new(71);
        for trial in 0..300 {
            with_differences(|q| {
                let context = [Atom::Lane(0), Atom::Lane(1), Atom::Fresh(PATH, ValueId(0), PATHS)];
                let differing = [Atom::Bit(ValueId(0)), Atom::View(ValueId(1)), Atom::View(ValueId(2))];
                let unknown = Atom::Fresh(LANE, ValueId(1001), 0);
                let world: Vec<Atom> = context.iter().chain(&differing).copied().collect();
                let mut h = Vec::new();
                for &atom in &differing {
                    let pool: Vec<Atom> = if trial % 3 == 0 {
                        context.to_vec()
                    } else {
                        world.iter().copied().filter(|&a| a != atom).collect()
                    };
                    let condition = random_function(q.logic(), &mut r, &pool, 3);
                    let v = match atom {
                        Atom::Bit(v) | Atom::View(v) => v,
                        _ => unreachable!(),
                    };
                    q.raise_h(v, condition);
                    h.push((variable(q.logic(), atom), condition));
                }
                let world_vars: Vec<u32> = world.iter().map(|&a| variable(q.logic(), a)).collect();
                let unknown_var = variable(q.logic(), unknown);
                let all: Vec<Atom> = world.iter().copied().chain([unknown]).collect();
                let g = random_function(q.logic(), &mut r, &all, 5);
                let cond = random_function(q.logic(), &mut r, &world, 3);
                if cond == Bdd::FALSE {
                    return;
                }
                let mut decisions = Decisions::new(Bdd::TRUE);
                let (mut every, mut none) = (true, true);
                let mut weak_rows = Vec::new();
                for row in 0..1u32 << world_vars.len() {
                    let lane_side = |var: u32| world_vars.iter().position(|&v| v == var).is_some_and(|i| row >> i & 1 == 1);
                    let free: Vec<u32> = h
                        .iter()
                        .filter(|&&(_, c)| evaluate(&q.logic.m, c, &lane_side))
                        .map(|&(v, _)| v)
                        .chain([unknown_var])
                        .collect();
                    let mut some = false;
                    for hidden in 0..1u32 << free.len() {
                        let wave_side = |var: u32| match free.iter().position(|&v| v == var) {
                            Some(i) => hidden >> i & 1 == 1,
                            None => lane_side(var),
                        };
                        let holds = evaluate(&q.logic.m, g, &wave_side);
                        some |= holds;
                        if evaluate(&q.logic.m, cond, &lane_side) {
                            every &= holds;
                            none &= !holds;
                        }
                    }
                    weak_rows.push(some);
                }
                let expected = if every { Some(true) } else if none { Some(false) } else { None };
                let decided = decisions.decide(q, g, cond, WAVE);
                if decided != expected {
                    wrong.push(format!("trial {} decides {:?}, expected {:?}", trial, decided, expected));
                }
                let weak = decisions.weaken(q, g, WAVE);
                for (row, &some) in weak_rows.iter().enumerate() {
                    let lane_side = |var: u32| world_vars.iter().position(|&v| v == var).is_some_and(|i| row >> i & 1 == 1);
                    if evaluate(&q.logic.m, weak, &lane_side) != some {
                        wrong.push(format!("trial {} weakens to {} at {}, expected {}", trial, !some, row, some));
                    }
                }
                let leftover: Vec<Atom> = q.logic().support(weak).iter().map(|&v| q.logic().atom_of(v)).filter(|a| matches!(a, Atom::Fresh(COPY, ..))).collect();
                if !leftover.is_empty() {
                    wrong.push(format!("trial {} leaves copies {:?}", trial, leftover));
                }
            });
        }
        assert!(wrong.is_empty(), "the wave side must quantify an atom only where it differs: {:?}", &wrong[..wrong.len().min(8)]);
    }

    #[test]
    fn varies_holds_exactly_where_some_differing_atom_can_tell_the_sides_apart() {
        let mut wrong = Vec::new();
        let mut r = Random::new(83);
        for trial in 0..300 {
            with_differences(|q| {
                let context = [Atom::Lane(0), Atom::Lane(1), Atom::Fresh(PATH, ValueId(0), PATHS)];
                let differing = [Atom::Bit(ValueId(0)), Atom::View(ValueId(1)), Atom::View(ValueId(2))];
                let shared = Atom::Fresh(JOINT, ValueId(1002), 0);
                let world: Vec<Atom> = context.iter().chain(&differing).copied().collect();
                let mut h = Vec::new();
                for &atom in &differing {
                    let pool: Vec<Atom> = if trial % 3 == 0 {
                        context.to_vec()
                    } else {
                        world.iter().copied().filter(|&a| a != atom).collect()
                    };
                    let condition = random_function(q.logic(), &mut r, &pool, 3);
                    let v = match atom {
                        Atom::Bit(v) | Atom::View(v) => v,
                        _ => unreachable!(),
                    };
                    q.raise_h(v, condition);
                    h.push((variable(q.logic(), atom), condition));
                }
                let world_vars: Vec<u32> = world.iter().map(|&a| variable(q.logic(), a)).collect();
                let shared_var = variable(q.logic(), shared);
                let all: Vec<Atom> = world.iter().copied().chain([shared]).collect();
                let wave = random_function(q.logic(), &mut r, &all, 5);
                let lane = if trial % 4 == 0 { wave } else { random_function(q.logic(), &mut r, &all, 5) };
                let mut decisions = Decisions::new(Bdd::TRUE);
                let varies = decisions.varies(q, wave, lane);
                for row in 0..1u32 << world_vars.len() {
                    let lane_side = |var: u32| world_vars.iter().position(|&v| v == var).is_some_and(|i| row >> i & 1 == 1);
                    let free: Vec<u32> = h
                        .iter()
                        .filter(|&&(_, c)| evaluate(&q.logic.m, c, &lane_side))
                        .map(|&(v, _)| v)
                        .collect();
                    let mut expected = false;
                    for unknown in 0..2u32 {
                        for hidden in 0..1u32 << free.len() {
                            let wave_side = |var: u32| match free.iter().position(|&v| v == var) {
                                Some(i) => hidden >> i & 1 == 1,
                                None if var == shared_var => unknown == 1,
                                None => lane_side(var),
                            };
                            let lane_value = |var: u32| if var == shared_var { unknown == 1 } else { lane_side(var) };
                            expected |= evaluate(&q.logic.m, wave, &wave_side) != evaluate(&q.logic.m, lane, &lane_value);
                        }
                    }
                    if evaluate(&q.logic.m, varies, &lane_side) != expected {
                        wrong.push(format!("trial {} row {} varies {}, expected {}", trial, row, !expected, expected));
                    }
                }
                let leftover: Vec<Atom> = q.logic().support(varies).iter().map(|&v| q.logic().atom_of(v)).filter(|a| matches!(a, Atom::Fresh(..))).collect();
                if leftover.iter().any(|a| !matches!(a, Atom::Fresh(PATH, ..))) {
                    wrong.push(format!("trial {} leaves {:?}", trial, leftover));
                }
            });
        }
        assert!(wrong.is_empty(), "varies must hold exactly where the wave side can differ: {:?}", &wrong[..wrong.len().min(8)]);
    }
}
