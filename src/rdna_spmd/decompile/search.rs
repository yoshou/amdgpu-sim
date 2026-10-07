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
            let minimal = without_redundant_choices(f, inputs, exec_index, &loops, hazards, &listed, kept.clone());
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
            if let Some(v) = named(check.logic(), condition, &order) {
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
        for c in blamed {
            kept.insert(c);
        }
    }
    let facts = Facts::new(f, inputs, &kept.words);
    let logic = Logic::fixed(f, &facts, &kept.choices(), &listed);
    let mut check = Check::new(f, &facts, inputs, exec_index, &loops, hazards, logic);
    assert!(check.run(), "the conversion policy left after dropping redundant choices no longer proves");
    let everyone = check.everyone(&BTreeSet::new());
    let kept = without_covered_meetings(&mut check, kept);
    (kept, everyone)
}

fn proves(
    f: &Func,
    inputs: &[Parameter],
    exec_index: Option<usize>,
    loops: &Loops,
    hazards: &Hazards,
    listed: &[Choice],
    kept: &Kept,
) -> bool {
    let facts = Facts::new(f, inputs, &kept.words);
    let logic = Logic::fixed(f, &facts, &kept.choices(), listed);
    let mut check = Check::new(f, &facts, inputs, exec_index, loops, hazards, logic);
    check.eager();
    check.run()
}

fn without_redundant_choices(
    f: &Func,
    inputs: &[Parameter],
    exec_index: Option<usize>,
    loops: &Loops,
    hazards: &Hazards,
    listed: &[Choice],
    mut kept: Kept,
) -> Kept {
    let candidates: Vec<Choice> = kept
        .queries
        .iter()
        .map(|&v| Choice::Query(v))
        .chain(kept.words.iter().map(|&v| Choice::Word(v)))
        .collect();
    let without = |choice: Choice| {
        let mut trial = kept.clone();
        trial.remove(choice);
        trial
    };
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get()).min(candidates.len().max(1));
    let alone: Vec<bool> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|w| {
                let (candidates, without) = (&candidates, &without);
                scope.spawn(move || {
                    (w..candidates.len())
                        .step_by(workers)
                        .map(|i| (i, proves(f, inputs, exec_index, loops, hazards, listed, &without(candidates[i]))))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        let mut alone = vec![false; candidates.len()];
        for handle in handles {
            for (i, redundant) in handle.join().expect("a redundancy trial panicked") {
                alone[i] = redundant;
            }
        }
        alone
    });
    let mut first = true;
    for (choice, redundant) in candidates.into_iter().zip(alone) {
        if !redundant {
            continue;
        }
        let mut trial = kept.clone();
        trial.remove(choice);
        if first || proves(f, inputs, exec_index, loops, hazards, listed, &trial) {
            kept = trial;
        }
        first = false;
    }
    kept
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
        let mut all = Kept::default();
        for q in [needed, needless[0], needless[1]] {
            assert!(listed.contains(&Choice::Query(q)));
            all.insert(Choice::Query(q));
        }
        assert!(proves(f, inputs, Some(0), &loops, &hazards, &listed, &all));
        assert!(!proves(f, inputs, Some(0), &loops, &hazards, &listed, &Kept::default()), "a lane without the flag skips the store the wave makes");
        let minimal = without_redundant_choices(f, inputs, Some(0), &loops, &hazards, &listed, all);
        assert_eq!(minimal.choices(), BTreeSet::from([Choice::Query(needed)]), "every lane computes the same tests the wave asks about");
        let (kept, _) = prove(f, inputs, Some(0), &hazards);
        assert_eq!(kept.choices(), BTreeSet::from([Choice::Query(needed)]));
    }
}
