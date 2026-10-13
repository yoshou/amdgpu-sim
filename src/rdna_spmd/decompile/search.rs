use super::check::{Basis, Check};
use super::hazard::{Hazards, Judge, Known, Question};
use super::logic::{choices, Atom, Choice, Kept, Logic, Structure};
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::{Conversion, Facts, Site};
use crate::rdna_spmd::analysis::loops::Loops;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

pub fn prove(
    f: &Func,
    inputs: &[Parameter],
    exec_index: Option<usize>,
    hazards: &Hazards,
    judge: &mut Judge,
) -> (Kept, BTreeSet<u64>) {
    let base = Facts::new(f, inputs, &BTreeSet::new());
    let loops = Loops::new(f, &base).unwrap_or_else(|block| {
        panic!(
            "b{}: control flow enters a cycle other than through its header",
            block.0
        )
    });
    let listed = listed(f, &base, hazards);
    let structure = Structure::of(f, &base);
    let order: HashMap<Choice, usize> = listed
        .iter()
        .enumerate()
        .map(|(i, &c)| (c, i))
        .collect();
    let mut kept = Kept::default();
    let mut added = BTreeSet::new();
    let mut basis: Option<Basis> = None;
    loop {
        let facts = Facts::converted(f, inputs, &kept.words, Conversion::Keeping(&kept.queries));
        let known = judge.known().clone();
        let mut check = match basis.take() {
            Some(basis) => Check::resume(f, &facts, inputs, exec_index, &loops, hazards, Some(&known), basis, &kept.choices()),
            None => {
                let logic = Logic::structured(&structure, f, &facts, &kept.choices(), &listed);
                Check::new(f, &facts, inputs, exec_index, &loops, hazards, Some(&known), logic)
            }
        };
        let proved = check.run();
        let pending = check.unresolved();
        if !pending.is_empty() {
            judge.judge(&pending);
            if check.resolve(judge.known()) {
                basis = Some(check.retire());
                continue;
            }
        }
        if proved {
            let failed = if added.len() == 1 { added.first().copied() } else { None };
            let minimal = without_redundant_choices(f, inputs, exec_index, &loops, hazards, judge, &base, &structure, &listed, kept.clone(), failed);
            if minimal == kept {
                let everyone = check.everyone(&BTreeSet::new());
                return (without_covered_meetings(&mut check, kept), everyone);
            }
            kept = minimal;
            break;
        }
        let mut blamed = BTreeSet::new();
        for i in 0..check.violations().len() {
            let condition = check.violations()[i].condition;
            let mut choices = alone(check.logic(), condition, &order);
            if choices.is_empty() {
                choices.extend(named(check.logic(), condition, &order));
            }
            for v in choices {
                if std::env::var_os("AMDGPU_SIM_PRINT_MEETINGS").is_some() {
                    let violation = &check.violations()[i];
                    eprintln!(
                        "; keeps {:?} for b{}:{}: {}",
                        v, violation.block.0, violation.index, violation.reason
                    );
                }
                blamed.insert(v);
            }
        }
        if blamed.is_empty() {
            let v = &check.violations()[0];
            panic!(
                "b{}:{}: {} under every conversion policy",
                v.block.0, v.index, v.reason
            );
        }
        let before = kept.choices();
        added = blamed.into_iter().filter(|c| !before.contains(c)).collect();
        if added.iter().all(|c| matches!(c, Choice::Meet(_))) {
            basis = Some(check.retire());
        }
        for &c in &added {
            kept.insert(c);
        }
    }
    let facts = Facts::converted(f, inputs, &kept.words, Conversion::Keeping(&kept.queries));
    loop {
        let known = judge.known().clone();
        let logic = Logic::structured(&structure, f, &facts, &kept.choices(), &listed);
        let mut check = Check::new(f, &facts, inputs, exec_index, &loops, hazards, Some(&known), logic);
        let proved = check.run();
        let pending = check.unresolved();
        if !pending.is_empty() {
            judge.judge(&pending);
            if check.resolve(judge.known()) {
                continue;
            }
        }
        assert!(proved, "the conversion policy left after dropping redundant choices no longer proves");
        let everyone = check.everyone(&BTreeSet::new());
        let kept = without_covered_meetings(&mut check, kept);
        return (kept, everyone);
    }
}

fn proves(
    f: &Func,
    inputs: &[Parameter],
    exec_index: Option<usize>,
    loops: &Loops,
    hazards: &Hazards,
    judge: &mut Judge,
    structure: &Structure,
    listed: &[Choice],
    kept: &Kept,
) -> bool {
    loop {
        match trial(f, inputs, exec_index, loops, hazards, judge.known(), structure, listed, kept) {
            Ok(proved) => return proved,
            Err(pending) => judge.judge(&pending),
        }
    }
}

fn trial(
    f: &Func,
    inputs: &[Parameter],
    exec_index: Option<usize>,
    loops: &Loops,
    hazards: &Hazards,
    known: &Known,
    structure: &Structure,
    listed: &[Choice],
    kept: &Kept,
) -> Result<bool, Vec<Question>> {
    let facts = Facts::converted(f, inputs, &kept.words, Conversion::Keeping(&kept.queries));
    let logic = Logic::structured(structure, f, &facts, &kept.choices(), listed);
    let mut check = Check::new(f, &facts, inputs, exec_index, loops, hazards, Some(known), logic);
    check.eager();
    if check.run() {
        return Ok(true);
    }
    let pending = check.unresolved();
    if pending.is_empty() {
        Ok(false)
    } else {
        Err(pending)
    }
}

fn without_redundant_choices(
    f: &Func,
    inputs: &[Parameter],
    exec_index: Option<usize>,
    loops: &Loops,
    hazards: &Hazards,
    judge: &mut Judge,
    base: &Facts,
    structure: &Structure,
    listed: &[Choice],
    mut kept: Kept,
    failed: Option<Choice>,
) -> Kept {
    let candidates: Vec<Choice> = kept
        .queries
        .iter()
        .map(|&v| Choice::Query(v))
        .chain(kept.words.iter().map(|&v| Choice::Word(v)))
        .collect();
    let schedule = latest_first(base, &candidates);
    let without = |choice: Choice| {
        let mut trial = kept.clone();
        trial.remove(choice);
        trial
    };
    let mut alone = vec![false; candidates.len()];
    let mut schedule: Vec<usize> = schedule.into_iter().filter(|&i| Some(candidates[i]) != failed).collect();
    while !schedule.is_empty() {
        let known = judge.known();
        let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).min(schedule.len());
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut pending: BTreeSet<Question> = BTreeSet::new();
        let mut undecided = Vec::new();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..workers)
                .map(|_| {
                    let (candidates, schedule, without, next) = (&candidates, &schedule, &without, &next);
                    scope.spawn(move || {
                        let mut done = Vec::new();
                        loop {
                            let Some(&i) = schedule.get(next.fetch_add(1, std::sync::atomic::Ordering::Relaxed)) else {
                                return done;
                            };
                            done.push((i, trial(f, inputs, exec_index, loops, hazards, known, structure, listed, &without(candidates[i]))));
                        }
                    })
                })
                .collect();
            for handle in handles {
                for (i, outcome) in handle.join().expect("a redundancy trial panicked") {
                    match outcome {
                        Ok(redundant) => alone[i] = redundant,
                        Err(questions) => {
                            pending.extend(questions);
                            undecided.push(i);
                        }
                    }
                }
            }
        });
        judge.judge(&pending.into_iter().collect::<Vec<_>>());
        schedule.retain(|i| undecided.contains(i));
    }
    let mut first = true;
    for (choice, redundant) in candidates.into_iter().zip(alone) {
        if !redundant {
            continue;
        }
        let mut trial = kept.clone();
        trial.remove(choice);
        if first || proves(f, inputs, exec_index, loops, hazards, judge, structure, listed, &trial) {
            kept = trial;
        }
        first = false;
    }
    kept
}

fn latest_first(facts: &Facts, candidates: &[Choice]) -> Vec<usize> {
    let rank: HashMap<BlockId, usize> = facts.order.iter().enumerate().map(|(r, &b)| (b, r)).collect();
    let position = |c: Choice| {
        let (Choice::Query(v) | Choice::Word(v)) = c else {
            return None;
        };
        match facts.site[v.0] {
            Site::Param { block, index } => Some((rank[&block], 0, index)),
            Site::Inst { block, index } => Some((rank[&block], 1, index)),
            _ => None,
        }
    };
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(position(candidates[i])));
    order
}

fn without_covered_meetings(check: &mut Check, mut kept: Kept) -> Kept {
    if kept.meets.is_empty() {
        return kept;
    }
    let base = check.orderings();
    let meets: Vec<usize> = kept.meets.iter().copied().collect();
    for m in meets {
        let mut trial = kept.clone();
        trial.meets.remove(&m);
        check.logic().keep(&trial.choices());
        if check.orderings() == base {
            kept = trial;
        } else {
            check.logic().keep(&kept.choices());
        }
    }
    kept
}

pub fn listed(f: &Func, facts: &Facts, hazards: &Hazards) -> Vec<Choice> {
    let mut listed = choices(f, facts);
    listed.extend((0..hazards.meetings.len()).rev().map(Choice::Meet));
    listed
}

fn alone(logic: &mut Logic, condition: Bdd, order: &HashMap<Choice, usize>) -> Vec<Choice> {
    let mut marks: Vec<(usize, u32, Choice)> = logic
        .support(condition)
        .iter()
        .filter_map(|&var| match logic.atom_of(var) {
            Atom::Marker(c) => Some((order[&c], var, c)),
            _ => None,
        })
        .collect();
    marks.sort_unstable();
    let mut left = condition;
    for &(_, var, _) in &marks {
        left = logic.m.cofactor(left, var, false);
    }
    let fixable = logic.m.not(left);
    let mut out = Vec::new();
    for &(_, var, c) in &marks {
        let mut g = condition;
        for &(_, other, _) in &marks {
            if other != var {
                g = logic.m.cofactor(g, other, false);
            }
        }
        let g = logic.m.cofactor(g, var, true);
        if logic.m.and(g, fixable) != Bdd::FALSE {
            out.push(c);
        }
    }
    out
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

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    fn needed_and_needless_queries() -> (Build, ValueId, [ValueId; 2]) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let flags = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let at = byte_offset(&mut b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, at, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let needed = b.wave(e, WaveOp::Any, vec![c]);
        let five = b.constant(e, Ty::I32, 5);
        let small = b.cmp(e, IntPred::Ult, k.kernarg.0, five);
        let first_needless = b.wave(e, WaveOp::Any, vec![small]);
        let seven = b.constant(e, Ty::I32, 7);
        let large = b.cmp(e, IntPred::Ugt, k.kernarg.1, seven);
        let second_needless = b.wave(e, WaveOp::Any, vec![large]);
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let shape = [Ty::I1, Ty::I64, Ty::I1, Ty::I1];
        let (first, t) = b.block(&shape);
        let (middle, m) = b.block(&shape);
        let (second, s) = b.block(&shape);
        let (later, l) = b.block(&shape);
        let (third, r) = b.block(&[Ty::I1, Ty::I64]);
        let (join, _) = b.block(&[Ty::I1]);
        let pass = |exec, at| vec![exec, at, first_needless, second_needless];
        b.cond_br(e, needed, (first, pass(k.exec, own)), (middle, pass(k.exec, own)));
        let one = b.constant(first, Ty::I32, 1);
        b.store(first, Space::Global, MemSize::B32, t[1], one, t[0]);
        b.br(first, middle, t.clone());
        b.cond_br(middle, m[2], (second, m.clone()), (later, m.clone()));
        let far = b.constant(second, Ty::I64, 4096);
        let other = b.int(second, IntOp::Add, s[1], far);
        let two = b.constant(second, Ty::I32, 2);
        b.store(second, Space::Global, MemSize::B32, other, two, s[0]);
        b.br(second, later, s.clone());
        let farther = b.constant(later, Ty::I64, 8192);
        let last = b.int(later, IntOp::Add, l[1], farther);
        b.cond_br(later, l[3], (third, vec![l[0], last]), (join, vec![l[0]]));
        let three = b.constant(third, Ty::I32, 3);
        b.store(third, Space::Global, MemSize::B32, r[1], three, r[0]);
        b.br(third, join, vec![r[0]]);
        (b, needed, [first_needless, second_needless])
    }

    #[test]
    fn trials_start_from_the_choice_latest_in_the_program() {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let one = b.constant(e, Ty::I32, 1);
        let odd = b.int(e, IntOp::And, lane, one);
        let test = b.cmp(e, IntPred::Ne, odd, one);
        let early = b.wave(e, WaveOp::Any, vec![test]);
        let word = b.int(e, IntOp::Add, lane, one);
        let (later, l) = b.block(&[Ty::I1, Ty::I32]);
        b.br(e, later, vec![p[0], word]);
        let late = b.wave(later, WaveOp::Any, vec![l[0]]);
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let candidates = [Choice::Query(early), Choice::Word(l[1]), Choice::Query(late), Choice::Word(word)];
        assert_eq!(
            latest_first(&facts, &candidates),
            vec![2, 1, 3, 0],
            "the later block first, its parameters before its instructions, and later instructions before earlier ones"
        );
    }

    #[test]
    fn redundant_choices_go_and_needed_ones_stay() {
        let (b, needed, needless) = needed_and_needless_queries();
        let (f, inputs) = (&b.f, &b.inputs[..]);
        let hazards = Hazards {
            accesses: Vec::new(),
            together: BTreeSet::new(),
            apart: BTreeSet::new(),
            idle: BTreeSet::new(),
            meetings: Vec::new(),
        };
        let base = Facts::new(f, inputs, &BTreeSet::new());
        let loops = Loops::new(f, &base).unwrap();
        let listed = listed(f, &base, &hazards);
        let structure = Structure::of(f, &base);
        let mut all = Kept::default();
        for q in [needed, needless[0], needless[1]] {
            assert!(listed.contains(&Choice::Query(q)));
            all.insert(Choice::Query(q));
        }
        let judge = &mut Judge::assumed();
        assert!(proves(f, inputs, Some(0), &loops, &hazards, judge, &structure, &listed, &all));
        assert!(!proves(f, inputs, Some(0), &loops, &hazards, judge, &structure, &listed, &Kept::default()), "a lane without the flag skips the store the wave makes");
        let minimal = without_redundant_choices(f, inputs, Some(0), &loops, &hazards, judge, &base, &structure, &listed, all.clone(), None);
        assert_eq!(minimal.choices(), BTreeSet::from([Choice::Query(needed)]), "every lane computes the same tests the wave asks about");
        let known = without_redundant_choices(f, inputs, Some(0), &loops, &hazards, judge, &base, &structure, &listed, all, Some(Choice::Query(needed)));
        assert_eq!(known.choices(), minimal.choices(), "a choice whose removal the search saw fail stays without a trial");
        let (kept, _) = prove(f, inputs, Some(0), &hazards, judge);
        assert_eq!(kept.choices(), BTreeSet::from([Choice::Query(needed)]));
    }

    #[test]
    fn alone_blames_exactly_the_choices_that_violate_beyond_what_keeping_everything_leaves() {
        let (b, _, _) = needed_and_needless_queries();
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let listed: Vec<Choice> = (0..4).map(Choice::Meet).collect();
        let order: HashMap<Choice, usize> = listed.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &listed);
        let mut pool: Vec<Atom> = listed.iter().map(|&c| Atom::Marker(c)).collect();
        pool.extend([Atom::Lane(0), Atom::Lane(1)]);
        let vars: Vec<u32> = pool.iter().map(|&a| variable(&mut logic, a)).collect();
        let mut r = Random::new(61);
        let (mut some, mut wrong) = (0, Vec::new());
        for trial in 0..400 {
            let mut condition = Bdd::FALSE;
            for _ in 0..1 + r.below(3) {
                let mut cube = Bdd::TRUE;
                for (i, &var) in vars.iter().enumerate() {
                    let literal = match r.below(if i < listed.len() { 3 } else { 4 }) {
                        0 => logic.m.var(var),
                        1 if i >= listed.len() => {
                            let v = logic.m.var(var);
                            logic.m.not(v)
                        }
                        _ => Bdd::TRUE,
                    };
                    cube = logic.m.and(cube, literal);
                }
                condition = logic.m.or(condition, cube);
            }
            let got: BTreeSet<Choice> = alone(&mut logic, condition, &order).into_iter().collect();
            let (vars, count) = (&vars, listed.len());
            let at = |local: Option<usize>, lanes: u32| {
                move |var: u32| match vars.iter().position(|&v| v == var) {
                    Some(j) if j < count => Some(j) == local,
                    Some(j) => lanes >> (j - count) & 1 == 1,
                    None => false,
                }
            };
            let support = logic.support(condition);
            let expected: BTreeSet<Choice> = listed
                .iter()
                .enumerate()
                .filter(|&(i, _)| support.contains(&vars[i]))
                .filter(|&(i, _)| {
                    (0..1u32 << 2).any(|lanes| {
                        evaluate(&logic.m, condition, &at(Some(i), lanes)) && !evaluate(&logic.m, condition, &at(None, lanes))
                    })
                })
                .map(|(_, &c)| c)
                .collect();
            some += got.len();
            if got != expected {
                wrong.push(format!("trial {} blames {:?}, expected {:?}", trial, got, expected));
            }
        }
        assert!(some > 0, "some condition has a choice that violates alone");
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
    }

    fn odd_answer_stored_by_one_wave(first: bool, loaded: bool) -> (crate::rdna_spmd::program::Program, ValueId) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = if loaded { k.buffer(&mut b, e, 0) } else { b.constant(e, Ty::I64, 0x1000) };
        let field = b.constant(e, Ty::I32, 0x3ff);
        let x = b.int(e, IntOp::And, k.item, field);
        let one = b.constant(e, Ty::I32, 1);
        let zero = b.constant(e, Ty::I32, 0);
        let parity = b.int(e, IntOp::And, x, one);
        let odd = b.cmp(e, IntPred::Ne, parity, zero);
        let asked = b.int(e, IntOp::And, odd, k.exec);
        let any = b.wave(e, WaveOp::Any, vec![asked]);
        let value = b.core(e, Ty::I32, Op::Select(any, one, zero));
        let half = b.constant(e, Ty::I32, 32);
        let chosen = b.cmp(e, if first { IntPred::Ult } else { IntPred::Uge }, x, half);
        let runs = b.int(e, IntOp::And, chosen, k.exec);
        let own = byte_offset(&mut b, e, buf, x, 4);
        b.store(e, Space::Global, MemSize::B32, own, value, runs);
        (b.program(), any)
    }

    #[test]
    fn prove_keeps_a_query_whose_answer_only_one_wave_stores() {
        for first in [false, true] {
            for loaded in [false, true] {
                let (program, any) = odd_answer_stored_by_one_wave(first, loaded);
                let mut env = environment(32, &[(0, 1, 0x1000)]);
                env.block = [64, 1, 1];
                let hazards = Hazards::find(&program, &env);
                let base = Facts::new(&program.ir, &program.parameter_inputs, &BTreeSet::new());
                let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards, &mut Judge::new(&program, &base, &env, &hazards));
                assert!(
                    kept.queries.contains(&any),
                    "wave {} of two stores whether any of its lanes is odd, which an even lane cannot tell alone (a load pairs with the store: {})",
                    if first { 0 } else { 1 },
                    loaded
                );
            }
        }
    }

    fn odd_answer_stored(scratch: bool) -> (crate::rdna_spmd::program::Program, ValueId, (BlockId, usize)) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let field = b.constant(e, Ty::I32, 0x3ff);
        let x = b.int(e, IntOp::And, k.item, field);
        let one = b.constant(e, Ty::I32, 1);
        let zero = b.constant(e, Ty::I32, 0);
        let parity = b.int(e, IntOp::And, x, one);
        let odd = b.cmp(e, IntPred::Ne, parity, zero);
        let asked = b.int(e, IntOp::And, odd, k.exec);
        let any = b.wave(e, WaveOp::Any, vec![asked]);
        let value = b.core(e, Ty::I32, Op::Select(any, one, zero));
        let shared = crate::rdna_spmd::SHARED_APERTURE;
        let p = if scratch {
            let base = b.core(e, Ty::I64, Op::Env(Env::ScratchBase));
            let eight = b.constant(e, Ty::I64, 8);
            b.int(e, IntOp::Add, base, eight)
        } else {
            let low = b.core(e, Ty::I32, Op::Env(Env::LaneId));
            let high = b.constant(e, Ty::I32, shared >> 32);
            b.core(e, Ty::I64, Op::Pack64(low, high))
        };
        let low = b.constant(e, Ty::I64, shared);
        let high = b.constant(e, Ty::I64, shared + (1 << 32));
        let above = b.cmp(e, IntPred::Uge, p, low);
        let below = b.cmp(e, IntPred::Ult, p, high);
        let in_window = b.int(e, IntOp::And, above, below);
        let local_exec = b.int(e, IntOp::And, k.exec, in_window);
        let local = b.core(e, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, p));
        let s = b.here(e);
        b.store(e, Space::Lds, MemSize::B32, local, value, local_exec);
        (b.program(), any, s)
    }

    #[test]
    fn prove_judges_only_the_store_whose_running_decides_a_violation() {
        for scratch in [false, true] {
            let (program, any, s) = odd_answer_stored(scratch);
            let env = environment(32, &[]);
            let hazards = Hazards::find(&program, &env);
            let base = Facts::new(&program.ir, &program.parameter_inputs, &BTreeSet::new());
            let mut judge = Judge::new(&program, &base, &env, &hazards);
            let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards, &mut judge);
            let i = hazards.accesses.iter().position(|a| (a.block, a.index) == s).unwrap();
            assert_eq!(judge.known().runs, std::collections::BTreeMap::from([(i, !scratch)]), "the store's data depends on the query, so the proof asks whether the store runs (scratch pointer: {})", scratch);
            assert_eq!(kept.queries.contains(&any), !scratch, "a store that never runs needs no query (scratch pointer: {})", scratch);
        }
    }

    #[test]
    fn prove_judges_nothing_when_no_violation_depends_on_an_access() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, buf, lane, 4);
        let value = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        b.store(e, Space::Global, MemSize::B32, own, value, k.exec);
        let program = b.program();
        let env = environment(32, &[(0, 1, 0x1000)]);
        let hazards = Hazards::find(&program, &env);
        let base = Facts::new(&program.ir, &program.parameter_inputs, &BTreeSet::new());
        let mut judge = Judge::new(&program, &base, &env, &hazards);
        let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards, &mut judge);
        assert!(kept.choices().is_empty());
        assert!(judge.known().is_empty(), "each lane copies its own word, so nothing turns on whether the copy runs: {:?}", judge.known());
    }
}
