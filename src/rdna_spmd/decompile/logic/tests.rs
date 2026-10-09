use super::super::address::compare;
use super::super::testing::*;
use super::*;
use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;
use std::collections::BTreeSet;

type Word = std::rc::Rc<dyn Fn(u32, u32, u32) -> u32>;

fn lane_words() -> (Build, Vec<(String, ValueId, Word)>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let yes = b.constant(e, Ty::I1, 1);
    let table = k.buffer(&mut b, e, 8);
    let u = b.load(e, Space::Global, MemSize::B32, table, yes);
    let at = b.constant(e, Ty::I64, 4);
    let second = b.int(e, IntOp::Add, table, at);
    let v = b.load(e, Space::Global, MemSize::B32, second, yes);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let mut words: Vec<(String, ValueId, Word)> = vec![
        ("lane".into(), lane, std::rc::Rc::new(|_, _, l| l)),
        ("u".into(), u, std::rc::Rc::new(|u, _, _| u)),
        ("v".into(), v, std::rc::Rc::new(|_, v, _| v)),
    ];
    let mut r = Random::new(53);
    for _ in 0..60 {
        let (i, j) = (r.below(words.len() as u64) as usize, r.below(words.len() as u64) as usize);
        let k = [1u32, 2, 3, 4, 5, 8, 31, 0x80, 0xf0f0][r.below(9) as usize];
        let ((nx, x, fx), (ny, y, fy)) = (words[i].clone(), words[j].clone());
        let kc = b.constant(e, Ty::I32, k as u64);
        let (name, value, truth): (String, ValueId, Word) = match r.below(9) {
            0 => (format!("({} & {})", nx, ny), b.int(e, IntOp::And, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l) & fy(u, v, l))),
            1 => (format!("({} | {})", nx, ny), b.int(e, IntOp::Or, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l) | fy(u, v, l))),
            2 => (format!("({} ^ {})", nx, ny), b.int(e, IntOp::Xor, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l) ^ fy(u, v, l))),
            3 => (format!("({} & {})", nx, k), b.int(e, IntOp::And, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l) & k)),
            4 => (format!("({} | {})", nx, k), b.int(e, IntOp::Or, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l) | k)),
            5 => (format!("({} << {})", nx, k & 31), b.int(e, IntOp::Shl, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l) << (k & 31))),
            6 => (format!("({} >> {})", nx, k & 31), b.int(e, IntOp::LShr, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l) >> (k & 31))),
            7 => (format!("({} + {})", nx, ny), b.int(e, IntOp::Add, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_add(fy(u, v, l)))),
            _ => (format!("({} - {})", nx, ny), b.int(e, IntOp::Sub, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_sub(fy(u, v, l)))),
        };
        words.push((name, value, truth));
    }
    (b, words)
}

#[test]
fn small_word_comparisons_hold_exactly_in_every_lane_and_word() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let yes = b.constant(e, Ty::I1, 1);
    let table = k.buffer(&mut b, e, 8);
    let u = b.load(e, Space::Global, MemSize::B32, table, yes);
    let c = |b: &mut Build, k: u64| b.constant(e, Ty::I32, k);
    let (k31, k7, k15, k16, k63) = (c(&mut b, 31), c(&mut b, 7), c(&mut b, 15), c(&mut b, 16), c(&mut b, 63));
    let low = b.int(e, IntOp::And, u, k15);
    let high = b.int(e, IntOp::Or, low, k16);
    let smalls: Vec<(ValueId, Box<dyn Fn(u32) -> bool>)> = vec![
        (b.int(e, IntOp::And, u, k31), Box::new(|x| x < 32)),
        (b.int(e, IntOp::And, u, k7), Box::new(|x| x < 8)),
        (high, Box::new(|x| (16..32).contains(&x))),
        (b.int(e, IntOp::And, u, k63), Box::new(|x| x < 64)),
    ];
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let (two, one, three) = (c(&mut b, 2), c(&mut b, 1), c(&mut b, 3));
    let doubled = b.int(e, IntOp::Mul, lane, two);
    let odd = b.int(e, IntOp::Add, doubled, one);
    let quarter = b.int(e, IntOp::And, lane, three);
    let mirrored = b.int(e, IntOp::Sub, k31, lane);
    let lanes: Vec<(ValueId, Box<dyn Fn(u32) -> u32>)> = vec![
        (lane, Box::new(|l| l)),
        (odd, Box::new(|l| 2 * l + 1)),
        (quarter, Box::new(|l| l & 3)),
        (mirrored, Box::new(|l| 31 - l)),
    ];
    let mut cases = Vec::new();
    for (i, _) in smalls.iter().enumerate() {
        for (j, _) in lanes.iter().enumerate() {
            for (pi, &p) in [IntPred::Eq, IntPred::Ne, IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sge].iter().enumerate() {
                let _ = pi;
                cases.push((i, j, p, false, b.cmp(e, p, smalls[i].0, lanes[j].0)));
                cases.push((i, j, p, true, b.cmp(e, p, lanes[j].0, smalls[i].0)));
            }
        }
    }
    let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(&b.f, &facts, &BTreeSet::new(), &[]);
    let evaluate = |logic: &Logic, g: Bdd, w: ValueId, x: u32, l: u32| {
        let mut g = g;
        while let Some((var, low, high)) = logic.m.decompose(g) {
            let bit = match logic.atom_of(var) {
                Atom::WordBit(v, i) if v == w => x >> i & 1 == 1,
                Atom::Lane(i) => l >> i & 1 == 1,
                _ => return None,
            };
            g = if bit { high } else { low };
        }
        Some(g == Bdd::TRUE)
    };
    let mut wrong = Vec::new();
    for &(i, j, p, flipped, cmp) in &cases {
        let g = logic.bit(&b.f, &facts, cmp);
        let w = smalls[i].0;
        for x in (0..64).filter(|&x| smalls[i].1(x)) {
            for l in 0..32u32 {
                let o = lanes[j].1(l);
                let truth = if flipped { compare(p, o, x) } else { compare(p, x, o) };
                if evaluate(&logic, g, w, x, l).is_some_and(|got| got != truth) {
                    wrong.push(format!("small {} lane word {} {:?} flipped {} at w = {} lane {}", i, j, p, flipped, x, l));
                }
            }
        }
    }
    for (i, (w, possible)) in smalls.iter().enumerate() {
        let Some(g) = logic.lane_is(&b.f, &facts, e, *w) else {
            continue;
        };
        for x in (0..64).filter(|&x| possible(x)) {
            for l in 0..32u32 {
                if evaluate(&logic, g, *w, x, l).is_some_and(|got| got != (l == x & 31)) {
                    wrong.push(format!("lane_is of small {} at w = {} lane {}", i, x, l));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(8)]);
}

#[test]
fn known_bits_hold_the_bits_of_every_lane_value() {
    let (b, words) = lane_words();
    let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(&b.f, &facts, &BTreeSet::new(), &[]);
    let mut r = Random::new(59);
    let mut wrong = Vec::new();
    for (name, v, truth) in &words {
        let Some(bits) = logic.lane_bits(&b.f, &facts, *v) else {
            continue;
        };
        for _ in 0..40 {
            let (u, w) = (r.next() as u32, r.next() as u32);
            for l in 0..32u32 {
                let (mask, value) = bits[l as usize];
                let t = truth(u, w, l);
                if t & mask != value {
                    wrong.push(format!("{} lane {} is {:#x}, not {:#x} under {:#x}", name, l, t, value, mask));
                }
            }
        }
    }
    wrong.truncate(5);
    assert!(wrong.is_empty(), "{:?}", wrong);
}

#[test]
fn uniform_tests_are_equal_in_every_lane() {
    let (mut b, mut words) = lane_words();
    let e = BlockId(0);
    let (lane, u) = (words[0].1, words[1].1);
    let difference = b.int(e, IntOp::Sub, u, lane);
    words.push(("(u - lane)".into(), difference, std::rc::Rc::new(|u, _, l| u.wrapping_sub(l))));
    let doubled = b.int(e, IntOp::Add, lane, lane);
    words.push(("(lane + lane)".into(), doubled, std::rc::Rc::new(|_, _, l| l.wrapping_add(l))));
    let mut tests = Vec::new();
    let mut r = Random::new(61);
    for _ in 0..3000 {
        let i = r.below(words.len() as u64) as usize;
        let j = r.below(words.len() as u64) as usize;
        let t = b.cmp(e, IntPred::Eq, words[i].1, words[j].1);
        tests.push((i, j, t));
    }
    let n = words.len();
    for (i, j) in [(n - 2, 0), (n - 2, n - 1), (0, n - 2)] {
        let t = b.cmp(e, IntPred::Eq, words[i].1, words[j].1);
        tests.push((i, j, t));
    }
    let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(&b.f, &facts, &BTreeSet::new(), &[]);
    let mut wrong = Vec::new();
    let mut decided = 0;
    for &(i, j, t) in &tests {
        let bit = logic.bit(&b.f, &facts, t);
        let uniform = logic.support(bit).iter().all(|&var| logic.uniform_atom(&facts, var));
        if facts.uniform[t.0] || !uniform {
            continue;
        }
        decided += 1;
        for _ in 0..40 {
            let pick = |r: &mut Random| if r.below(2) == 0 { r.below(64) as u32 } else { r.next() as u32 };
            let (u, w) = (pick(&mut r), pick(&mut r));
            let first = (words[i].2)(u, w, 0) == (words[j].2)(u, w, 0);
            if (1..32).any(|l| ((words[i].2)(u, w, l) == (words[j].2)(u, w, l)) != first) {
                wrong.push(format!("{} == {}", words[i].0, words[j].0));
                break;
            }
        }
    }
    wrong.truncate(5);
    assert!(decided > 0 && wrong.is_empty(), "{} uniform, wrong {:?}", decided, wrong);
}

#[test]
fn uniform_orders_are_equal_in_every_lane() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let yes = b.constant(e, Ty::I1, 1);
    let table = k.buffer(&mut b, e, 8);
    let byte = b.load(e, Space::Global, MemSize::U8, table, yes);
    let at = b.constant(e, Ty::I64, 4);
    let second = b.int(e, IntOp::Add, table, at);
    let half = b.load(e, Space::Global, MemSize::U16, second, yes);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let mut words: Vec<(String, ValueId, Word)> = vec![
        ("lane".into(), lane, std::rc::Rc::new(|_, _, l| l)),
        ("b".into(), byte, std::rc::Rc::new(|u, _, _| u & 0xff)),
        ("h".into(), half, std::rc::Rc::new(|_, v, _| v & 0xffff)),
    ];
    let mut r = Random::new(67);
    for _ in 0..40 {
        let (i, j) = (r.below(words.len() as u64) as usize, r.below(words.len() as u64) as usize);
        let k = [1u32, 2, 3, 5, 31, 0x80][r.below(6) as usize];
        let ((nx, x, fx), (ny, y, fy)) = (words[i].clone(), words[j].clone());
        let kc = b.constant(e, Ty::I32, k as u64);
        let (name, value, truth): (String, ValueId, Word) = match r.below(4) {
            0 => (format!("({} + {})", nx, ny), b.int(e, IntOp::Add, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_add(fy(u, v, l)))),
            1 => (format!("({} - {})", nx, ny), b.int(e, IntOp::Sub, x, y), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_sub(fy(u, v, l)))),
            2 => (format!("({} * {})", nx, k), b.int(e, IntOp::Mul, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_mul(k))),
            _ => (format!("({} + {})", nx, k), b.int(e, IntOp::Add, x, kc), std::rc::Rc::new(move |u, v, l| fx(u, v, l).wrapping_add(k))),
        };
        words.push((name, value, truth));
    }
    let step = b.constant(e, Ty::I32, 0x400_0000);
    let wide = b.int(e, IntOp::Mul, lane, step);
    let small = b.constant(e, Ty::I32, 16);
    let large = b.constant(e, Ty::I32, 0x500_0000);
    let low = b.int(e, IntOp::Add, wide, small);
    let high = b.int(e, IntOp::Add, wide, large);
    words.push(("lane * 2^26 + 16".into(), low, std::rc::Rc::new(|_, _, l| l.wrapping_mul(0x400_0000).wrapping_add(16))));
    words.push(("lane * 2^26 + 0x5000000".into(), high, std::rc::Rc::new(|_, _, l| l.wrapping_mul(0x400_0000).wrapping_add(0x500_0000))));
    let predicates = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge];
    let mut tests = Vec::new();
    let n = words.len();
    for p in predicates {
        tests.push((n - 2, n - 1, p));
    }
    for _ in 0..3000 {
        let i = r.below(words.len() as u64) as usize;
        let j = r.below(words.len() as u64) as usize;
        let p = predicates[r.below(predicates.len() as u64) as usize];
        tests.push((i, j, p));
    }
    let tests: Vec<(usize, usize, IntPred, ValueId)> = tests.iter().map(|&(i, j, p)| (i, j, p, b.cmp(e, p, words[i].1, words[j].1))).collect();
    let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(&b.f, &facts, &BTreeSet::new(), &[]);
    let mut wrong = Vec::new();
    let mut decided = 0;
    for &(i, j, p, t) in &tests {
        let bit = logic.bit(&b.f, &facts, t);
        let uniform = logic.support(bit).iter().all(|&var| logic.uniform_atom(&facts, var));
        if facts.uniform[t.0] || !uniform {
            continue;
        }
        decided += 1;
        for _ in 0..40 {
            let (u, v) = (r.next() as u32, r.next() as u32);
            let holds = |l: u32| compare(p, (words[i].2)(u, v, l), (words[j].2)(u, v, l));
            let first = holds(0);
            if (1..32).any(|l| holds(l) != first) {
                wrong.push(format!("{} {:?} {}", words[i].0, p, words[j].0));
                break;
            }
        }
    }
    wrong.truncate(5);
    assert!(decided > 0 && wrong.is_empty(), "{} uniform, wrong {:?}", decided, wrong);
}

struct SelfLoop {
    b: Build,
    body: BlockId,
    exec: ValueId,
    carried: ValueId,
}

fn self_loop() -> SelfLoop {
    let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
    let e = BlockId(0);
    let yes = b.constant(e, Ty::I1, 1);
    let (body, q) = b.block(&[Ty::I1, Ty::I1]);
    let (exit, _) = b.block(&[]);
    b.br(e, body, vec![p[0], yes]);
    let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
    let five = b.constant(body, Ty::I32, 5);
    let fresh = b.cmp(body, IntPred::Ult, lane, five);
    let one = b.constant(body, Ty::I1, 1);
    let stale = b.int(body, IntOp::Xor, fresh, one);
    let conjunction = b.int(body, IntOp::And, q[1], stale);
    b.cond_br(body, conjunction, (body, vec![q[0], fresh]), (exit, vec![]));
    SelfLoop {
        b,
        body,
        exec: q[0],
        carried: q[1],
    }
}

#[test]
fn reach_keeps_the_values_a_self_loop_carries_around() {
    let SelfLoop {
        b,
        body,
        exec,
        carried,
        ..
    } = self_loop();
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
    let start = logic.atom(Atom::Bit(f.blocks[&f.entry].params[0].0));
    let reach = logic.reach(f, &facts, f.entry, start);
    let exec = logic.atom(Atom::Bit(exec));
    let carried = logic.atom(Atom::Bit(carried));
    let lost = logic.m.not(carried);
    let second = logic.m.and(exec, lost);
    let state = logic.m.and(reach[&body], second);
    assert_ne!(
        state,
        Bdd::FALSE,
        "the second iteration starts with the carried bit false, which the first iteration sends when its fresh bit is false"
    );
}

fn nested_selects(loaded_arm: bool) -> (Build, ValueId, ValueId, ValueId) {
    let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32), (ParameterSource::Vgpr(2), Ty::I32)]);
    let e = BlockId(0);
    let item = p[1];
    let two = b.constant(e, Ty::I32, 2);
    let mut value = if loaded_arm { p[2] } else { item };
    for k in 0..8 {
        let bound = b.constant(e, Ty::I32, 10 + k);
        let c = b.cmp(e, IntPred::Ult, item, bound);
        value = b.core(e, Ty::I32, Op::Select(c, two, value));
    }
    let three = b.constant(e, Ty::I32, 3);
    let small = b.cmp(e, IntPred::Ult, value, three);
    let large = b.cmp(e, IntPred::Ugt, value, three);
    let seventeen = b.constant(e, Ty::I32, 17);
    let any = b.cmp(e, IntPred::Ult, item, seventeen);
    (b, small, large, any)
}

#[test]
fn bit_follows_selects_of_any_depth() {
    let (b, small, large, any) = nested_selects(false);
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
    let any = logic.bit(f, &facts, any);
    assert_eq!(logic.bit(f, &facts, small), any, "eight nested selects give 2 below 17 and the item itself from 17 on");
    assert_eq!(logic.bit(f, &facts, large), logic.m.not(any), "the value is above 3 exactly from 17 on");
}

#[test]
fn bit_keeps_the_loaded_arm_eight_selects_deep() {
    let (b, small, large, any) = nested_selects(true);
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
    let any = logic.bit(f, &facts, any);
    let (small, large) = (logic.bit(f, &facts, small), logic.bit(f, &facts, large));
    assert_eq!(logic.m.and(small, any), any, "below 17 the value is 2");
    let beyond = logic.m.not(any);
    assert_ne!(logic.m.and(small, beyond), Bdd::FALSE, "from 17 on the loaded word may be below 3");
    assert_eq!(logic.m.and(small, large), Bdd::FALSE, "no value is both below and above 3");
    assert_ne!(logic.m.or(small, large), Bdd::TRUE, "the loaded word may be 3");
}

#[test]
fn bit_and_view_follow_the_operations() {
    let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)]);
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let item = p[1];
    let three = b.constant(e, Ty::I32, 3);
    let seven = b.constant(e, Ty::I32, 7);
    let c = b.cmp(e, IntPred::Ult, item, three);
    let d = b.cmp(e, IntPred::Ult, item, seven);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let x = b.core(e, Ty::I32, Op::Select(c, one, two));
    let y = b.core(e, Ty::I32, Op::Select(d, two, one));
    let crossed = b.cmp(e, IntPred::Eq, x, y);
    let picked = b.cmp(e, IntPred::Eq, x, one);
    let same = b.cmp(e, IntPred::Ne, lane, lane);
    let constants = b.cmp(e, IntPred::Eq, three, seven);
    let bits = b.cmp(e, IntPred::Ne, c, d);
    let w = b.wave(e, WaveOp::Ballot { high: false }, vec![c]);
    let v = b.wave(e, WaveOp::Ballot { high: false }, vec![d]);
    let own = b.int(e, IntOp::LShr, w, lane);
    let own = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, own));
    let zero = b.constant(e, Ty::I32, 0);
    let ones = b.constant(e, Ty::I32, 0xffff_ffff);
    let nothing = b.int(e, IntOp::And, w, zero);
    let everything = b.int(e, IntOp::Or, v, ones);
    let differ = b.int(e, IntOp::Xor, w, v);
    let chosen = b.core(e, Ty::I32, Op::Select(c, w, v));
    let cast = b.core(e, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, differ));
    let valid = b.core(e, Ty::I1, Op::Env(Env::ValidLane));
    let exec = p[0];
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let four = b.constant(e, Ty::I32, 4);
    let apart = b.int(e, IntOp::And, one, two);
    let other = b.core(e, Ty::I32, Op::Select(c, two, four));
    let missed = b.int(e, IntOp::And, one, other);
    let kept = b.int(e, IntOp::And, three, other);
    let z = b.core(e, Ty::I32, Op::Select(d, one, two));
    let paired = b.int(e, IntOp::Xor, x, z);
    let not_c = b.cmp(e, IntPred::Uge, item, three);
    let swapped_c = b.cmp(e, IntPred::Ugt, three, item);
    let not_d = b.cmp(e, IntPred::Ule, seven, item);
    let signed = b.cmp(e, IntPred::Slt, item, three);
    let not_signed = b.cmp(e, IntPred::Sge, item, three);
    let far = b.constant(e, Ty::I32, 99);
    let low_lanes = b.cmp(e, IntPred::Ult, lane, three);
    let high_lanes = b.cmp(e, IntPred::Ugt, lane, seven);
    let no_lane = b.cmp(e, IntPred::Eq, lane, far);
    let every_lane = b.cmp(e, IntPred::Ne, far, lane);
    let minus = b.constant(e, Ty::I32, 0xffff_fffe);
    let signed_lanes = b.cmp(e, IntPred::Sgt, lane, minus);
    let hundred = b.constant(e, Ty::I32, 100);
    let five = b.constant(e, Ty::I32, 5);
    let kept_item = b.core(e, Ty::I32, Op::Select(c, item, hundred));
    let small = b.cmp(e, IntPred::Ult, kept_item, five);
    let large = b.cmp(e, IntPred::Ugt, five, kept_item);
    let chosen_small = b.cmp(e, IntPred::Ult, x, two);
    let chosen_any = b.cmp(e, IntPred::Ule, x, two);
    let odd_bit = b.int(e, IntOp::And, lane, one);
    let odd = b.cmp(e, IntPred::Eq, odd_bit, one);
    let group = b.int(e, IntOp::LShr, lane, three);
    let third_group = b.cmp(e, IntPred::Eq, group, two);
    let own_bit = b.int(e, IntOp::Shl, one, lane);
    let thirty_two = b.constant(e, Ty::I32, 32);
    let gone = b.int(e, IntOp::Shl, lane, thirty_two);
    let undefined = b.cmp(e, IntPred::Eq, gone, zero);
    let mixed = b.int(e, IntOp::Add, lane, item);
    let with_item = b.cmp(e, IntPred::Ult, mixed, three);
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
    let bc = logic.bit(f, &facts, c);
    let bd = logic.bit(f, &facts, d);
    assert!(bc != Bdd::FALSE && bc != Bdd::TRUE && bd != bc, "item < 3 and item < 7 are open and distinct");
    assert_eq!(logic.m.and(bc, bd), bc, "item < 3 implies item < 7");
    let _ = exec;
    let mut expect = Vec::new();
    let xor = logic.m.xor(bc, bd);
    let ndd = logic.m.not(bd);
    let crossed_formula = logic.m.ite(bc, ndd, bd);
    expect.push(("select(c, 1, 2) == select(d, 2, 1)", logic.bit(f, &facts, crossed), crossed_formula));
    expect.push(("select(c, 1, 2) == 1", logic.bit(f, &facts, picked), bc));
    expect.push(("lane != lane", logic.bit(f, &facts, same), Bdd::FALSE));
    expect.push(("3 == 7", logic.bit(f, &facts, constants), Bdd::FALSE));
    expect.push(("c != d on bits", logic.bit(f, &facts, bits), xor));
    expect.push(("trunc(ballot(c) >> lane)", logic.bit(f, &facts, own), bc));
    expect.push(("ballot(c) & 0", logic.view(f, &facts, nothing), Bdd::FALSE));
    expect.push(("ballot(d) | ~0", logic.view(f, &facts, everything), Bdd::TRUE));
    expect.push(("ballot(c) ^ ballot(d)", logic.view(f, &facts, differ), xor));
    let chosen_formula = logic.m.ite(bc, bc, bd);
    expect.push(("select(c, ballot(c), ballot(d))", logic.view(f, &facts, chosen), chosen_formula));
    expect.push(("bitcast of the xor", logic.view(f, &facts, cast), xor));
    expect.push(("valid lane", logic.bit(f, &facts, valid), Bdd::TRUE));
    expect.push(("converted any(c)", logic.bit(f, &facts, q), bc));
    expect.push(("1 & 2", logic.view(f, &facts, apart), Bdd::FALSE));
    expect.push(("1 & select(c, 2, 4)", logic.view(f, &facts, missed), Bdd::FALSE));
    let k2 = logic.word_of(Ty::I32, 2);
    let kept_formula = logic.m.and(bc, k2);
    expect.push(("3 & select(c, 2, 4)", logic.view(f, &facts, kept), kept_formula));
    let k3 = logic.word_of(Ty::I32, 3);
    let unequal = logic.m.and(xor, k3);
    expect.push(("select(c, 1, 2) ^ select(d, 1, 2)", logic.view(f, &facts, paired), unequal));
    let (nbc, nbd) = (logic.m.not(bc), logic.m.not(bd));
    expect.push(("item >= 3", logic.bit(f, &facts, not_c), nbc));
    expect.push(("3 > item", logic.bit(f, &facts, swapped_c), bc));
    expect.push(("7 <= item", logic.bit(f, &facts, not_d), nbd));
    let bs = logic.bit(f, &facts, signed);
    expect.push(("item s< 3 within item < 7", logic.m.and(bs, bd), bc));
    assert_ne!(logic.m.and(bs, nbd), Bdd::FALSE, "a negative item is s< 3 and not < 7");
    let not_signed_formula = logic.m.not(bs);
    expect.push(("item s>= 3", logic.bit(f, &facts, not_signed), not_signed_formula));
    let low = logic.lanes(|l| l < 3);
    expect.push(("lane < 3", logic.bit(f, &facts, low_lanes), low));
    let high = logic.lanes(|l| l > 7);
    expect.push(("lane > 7", logic.bit(f, &facts, high_lanes), high));
    expect.push(("lane == 99", logic.bit(f, &facts, no_lane), Bdd::FALSE));
    expect.push(("99 != lane", logic.bit(f, &facts, every_lane), Bdd::TRUE));
    expect.push(("lane s> -2", logic.bit(f, &facts, signed_lanes), Bdd::TRUE));
    expect.push(("select(c, item, 100) < 5", logic.bit(f, &facts, small), bc));
    expect.push(("5 > select(c, item, 100)", logic.bit(f, &facts, large), bc));
    expect.push(("select(c, 1, 2) < 2", logic.bit(f, &facts, chosen_small), bc));
    expect.push(("select(c, 1, 2) <= 2", logic.bit(f, &facts, chosen_any), Bdd::TRUE));
    let odd_lanes = logic.lanes(|l| l & 1 == 1);
    expect.push(("(lane & 1) == 1", logic.bit(f, &facts, odd), odd_lanes));
    let lanes_16_to_23 = logic.lanes(|l| (16..24).contains(&l));
    expect.push(("lane >> 3 == 2", logic.bit(f, &facts, third_group), lanes_16_to_23));
    expect.push(("the lane bit of 1 << lane", logic.view(f, &facts, own_bit), Bdd::TRUE));
    let undefined_bit = logic.bit(f, &facts, undefined);
    assert!(undefined_bit != Bdd::TRUE && undefined_bit != Bdd::FALSE, "(lane << 32) == 0 stays open");
    let with_item_bit = logic.bit(f, &facts, with_item);
    assert!(with_item_bit != Bdd::TRUE && with_item_bit != Bdd::FALSE, "lane + item < 3 stays open");
    let lane_one = logic.lanes(|l| l == 1);
    let bits_of_two = logic.word_of(Ty::I32, 2);
    expect.push(("the lanes of 2", bits_of_two, lane_one));
    let five = logic.word_of(Ty::I32, 5);
    let seven_bits = logic.word_of(Ty::I32, 7);
    let both = logic.m.and(five, seven_bits);
    expect.push(("5 & 7 by lanes", both, five));
    let wrong: Vec<&str> = expect.iter().filter(|(_, got, want)| got != want).map(|(name, ..)| *name).collect();
    assert!(wrong.is_empty(), "{:?}", wrong);
    let _ = crossed_formula;
    assert_eq!(xor, logic.m.xor(bc, bd));
}

struct Edge2 {
    b: Build,
    src: BlockId,
    bits: Vec<ValueId>,
}

fn edge(r: &mut Random, self_loop: bool) -> Edge2 {
    let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)]);
    let e = BlockId(0);
    let n = 4;
    let (src, s) = b.block(&vec![Ty::I1; n]);
    let seeds: Vec<ValueId> = (0..n).map(|_| b.constant(e, Ty::I1, r.below(2))).collect();
    b.br(e, src, seeds);
    let mut bits: Vec<ValueId> = s.clone();
    for k in 0..3 {
        let bound = b.constant(src, Ty::I32, k + 1);
        bits.push(b.cmp(src, IntPred::Ult, p[1], bound));
    }
    let one = b.constant(src, Ty::I1, 1);
    let zero = b.constant(src, Ty::I1, 0);
    let pick = |r: &mut Random, bits: &[ValueId]| bits[r.below(bits.len() as u64) as usize];
    let args: Vec<ValueId> = (0..n)
        .map(|_| match r.below(6) {
            0 => one,
            1 => zero,
            2 => {
                let x = pick(r, &bits);
                b.int(src, IntOp::Xor, x, one)
            }
            3 => {
                let (x, y) = (pick(r, &bits), pick(r, &bits));
                b.int(src, IntOp::And, x, y)
            }
            _ => pick(r, &bits),
        })
        .collect();
    let (exit, _) = b.block(&[]);
    let dst = if self_loop {
        src
    } else {
        let (dst, _) = b.block(&vec![Ty::I1; n]);
        dst
    };
    let cond = pick(r, &bits);
    b.cond_br(src, cond, (dst, args), (exit, vec![]));
    Edge2 { b, src, bits }
}

fn images(seed: u64, rounds: usize, check: impl Fn(&mut Logic, Bdd, Bdd, Bdd) -> bool) -> Vec<(usize, &'static str)> {
    crossings(seed, rounds, false, check)
}

fn projection(logic: &mut Logic, f: &Func, facts: &Facts, src: BlockId, formula: Bdd, linked: &[bool], self_loop: bool) -> Bdd {
    let edge = f.blocks[&src].term.edges().next().unwrap().clone();
    let dst = &f.blocks[&edge.dst];
    let mut fresh = Vec::new();
    let mut relation = formula;
    for (k, (&(param, _), &arg)) in dst.params.iter().zip(&edge.args).enumerate() {
        if !linked[k] {
            continue;
        }
        let target = if self_loop {
            let t = logic.atom(Atom::Fresh(7, param, 0));
            fresh.push((t, param));
            t
        } else {
            logic.atom(Atom::Bit(param))
        };
        let bound = logic.bit(f, facts, arg);
        let link = logic.m.iff(target, bound);
        relation = logic.m.and(relation, link);
    }
    let scoped: Vec<u32> = logic.support(relation).iter().copied().filter(|&v| logic.scope(facts, v) == Some(src)).collect();
    let mut want = logic.exists(&scoped, relation);
    if self_loop {
        let renamed: HashMap<u32, Bdd> = fresh.iter().map(|&(t, param)| (logic.support(t)[0], logic.atom(Atom::Bit(param)))).collect();
        want = logic.m.compose(want, &|v| renamed.get(&v).copied());
    }
    want
}

fn crossings(seed: u64, rounds: usize, difference: bool, check: impl Fn(&mut Logic, Bdd, Bdd, Bdd) -> bool) -> Vec<(usize, &'static str)> {
    let mut r = Random::new(seed);
    let mut wrong = Vec::new();
    for round in 0..rounds {
        let self_loop = round % 2 == 1;
        let Edge2 { b, src, bits } = edge(&mut r, self_loop);
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let mut formula = Bdd::FALSE;
        for _ in 0..3 {
            let mut term = Bdd::TRUE;
            for _ in 0..2 {
                let v = bits[r.below(bits.len() as u64) as usize];
                let a = logic.bit(f, &facts, v);
                let a = if r.below(2) == 0 { a } else { logic.m.not(a) };
                term = logic.m.and(term, a);
            }
            formula = logic.m.or(formula, term);
        }
        let got = if difference {
            logic.image(f, &facts, src, 0, formula)
        } else {
            logic.post(f, &facts, src, 0, formula)
        };
        let edge = f.blocks[&src].term.edges().next().unwrap().clone();
        let every = vec![true; edge.args.len()];
        let full = projection(&mut logic, f, &facts, src, formula, &every, self_loop);
        let mut linked = vec![!difference; edge.args.len()];
        if difference {
            let mut reached: BTreeSet<u32> = logic.support(formula).iter().copied().filter(|&v| logic.scope(&facts, v) == Some(src)).collect();
            loop {
                let mut grew = false;
                for (k, &arg) in edge.args.iter().enumerate() {
                    let bound = logic.bit(f, &facts, arg);
                    let support: Vec<u32> = logic.support(bound).iter().copied().filter(|&v| logic.scope(&facts, v) == Some(src)).collect();
                    if !linked[k] && support.iter().any(|v| reached.contains(v)) {
                        linked[k] = true;
                        reached.extend(support);
                        grew = true;
                    }
                }
                if !grew {
                    break;
                }
            }
        }
        let reachable = projection(&mut logic, f, &facts, src, formula, &linked, self_loop);
        if !check(&mut logic, got, full, reachable) {
            wrong.push((round, if self_loop { "self-loop" } else { "edge" }));
        }
    }
    wrong
}

#[test]
fn image_keeps_every_state_the_argument_relations_allow() {
    let wrong = crossings(31, 400, true, |logic, got, full, _| logic.m.implies(full, got));
    assert!(wrong.is_empty(), "rounds whose difference image drops a state the edge can reach: {:?}", wrong);
}

#[test]
fn image_equals_the_projection_of_the_formula_and_the_relations_it_reaches() {
    let wrong = crossings(31, 400, true, |_, got, _, reachable| got == reachable);
    assert!(wrong.is_empty(), "rounds whose difference image differs from the projection over the relations it reaches: {:?}", wrong);
}

#[test]
fn post_keeps_every_state_the_argument_relations_allow() {
    let wrong = images(29, 400, |logic, got, full, _| logic.m.implies(full, got));
    assert!(wrong.is_empty(), "rounds whose image drops a state the edge can reach: {:?}", wrong);
}

#[test]
fn post_equals_the_projection_of_the_formula_and_every_argument_relation() {
    let wrong = images(29, 400, |_, got, full, _| got == full);
    assert!(wrong.is_empty(), "rounds whose image differs from the projection: {:?}", wrong);
}

#[test]
fn restate_equals_the_projection_of_the_formula_and_every_link_onto_the_block() {
    for limit in [0, usize::MAX] {
        let projected = with_joint_limit(limit, restatements);
        assert!(projected > 100, "too few rounds had anything to project");
    }
}

#[test]
fn images_equal_their_projections_whether_or_not_links_join_before_the_product() {
    for limit in [0, usize::MAX] {
        with_joint_limit(limit, || {
            let wrong = crossings(31, 400, true, |_, got, _, reachable| got == reachable);
            assert!(wrong.is_empty(), "limit {}: difference images that differ from the projection: {:?}", limit, wrong);
            let wrong = images(29, 400, |_, got, full, _| got == full);
            assert!(wrong.is_empty(), "limit {}: images that differ from the projection: {:?}", limit, wrong);
        });
    }
}

fn restatements() -> usize {
    let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
    let e = BlockId(0);
    let (src, s) = b.block(&[Ty::I1; 4]);
    let (dst, d) = b.block(&[Ty::I1; 4]);
    b.br(e, src, vec![p[0]; 4]);
    b.br(src, dst, s.clone());
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let tags = [Choice::Meet(0)];
    let mut pool: Vec<Atom> = s.iter().chain(&d).map(|&v| Atom::Bit(v)).collect();
    pool.extend([Atom::Lane(0), Atom::Lane(5), Atom::Marker(Choice::Meet(0)), Atom::Fresh(1, d[0], 0)]);
    let mut r = Random::new(71);
    let mut projected = 0;
    for _ in 0..400 {
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &tags);
        let formula = random_function(&mut logic, &mut r, &pool, 4);
        let mut links = Vec::new();
        for _ in 0..r.below(5) {
            let atom = logic.atom(Atom::Bit(d[r.below(d.len() as u64) as usize]));
            let bound = random_function(&mut logic, &mut r, &pool, 3);
            links.push((atom, bound));
        }
        let mut whole = formula;
        for &(atom, bound) in &links {
            let link = logic.m.iff(atom, bound);
            whole = logic.m.and(whole, link);
        }
        let foreign: Vec<u32> = logic
            .support(whole)
            .iter()
            .copied()
            .filter(|&v| !matches!(logic.atom_of(v), Atom::Marker(_) | Atom::Lane(5)) && logic.scope(&facts, v) != Some(dst))
            .collect();
        let want = logic.exists(&foreign, whole);
        let got = logic.restate(&facts, dst, formula, &links);
        assert_eq!(got, want, "restating must project away every atom outside the block but the markers and the upper half");
        projected += (!foreign.is_empty() && want != whole) as usize;
    }
    projected
}

fn canonical(m: &crate::rdna_spmd::analysis::bdd::Manager, g: Bdd) -> String {
    fn walk(m: &crate::rdna_spmd::analysis::bdd::Manager, g: Bdd, seen: &mut HashMap<Bdd, usize>, out: &mut Vec<String>) -> String {
        match m.decompose(g) {
            None => format!("{}", g == Bdd::TRUE),
            Some((var, low, high)) => {
                if let Some(&i) = seen.get(&g) {
                    return format!("#{}", i);
                }
                let (low, high) = (walk(m, low, seen, out), walk(m, high, seen, out));
                out.push(format!("{}?{}:{}", var, high, low));
                seen.insert(g, out.len() - 1);
                format!("#{}", out.len() - 1)
            }
        }
    }
    let mut out = Vec::new();
    let root = walk(m, g, &mut HashMap::default(), &mut out);
    format!("{} {}", root, out.join(" "))
}

#[test]
fn a_structure_built_under_other_facts_reads_every_comparison_as_a_lazy_logic_with_the_checks_facts() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let yes = b.constant(e, Ty::I1, 1);
    let table = k.buffer(&mut b, e, 8);
    let u = b.load(e, Space::Global, MemSize::B32, table, yes);
    let at = b.constant(e, Ty::I64, 4);
    let second = b.int(e, IntOp::Add, table, at);
    let v = b.load(e, Space::Global, MemSize::B32, second, yes);
    let w = b.load(e, Space::Global, MemSize::B32, table, yes);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let mut tests = Vec::new();
    for (pred, x, y) in [(IntPred::Ult, u, v), (IntPred::Eq, u, v), (IntPred::Ugt, v, lane), (IntPred::Ult, v, w)] {
        tests.push(b.cmp(e, pred, x, y));
    }
    let (next, n) = b.block(&[Ty::I1, Ty::I32, Ty::I32]);
    b.br(e, next, vec![k.exec, v, w]);
    for (pred, x, y) in [(IntPred::Ult, n[1], n[2]), (IntPred::Ne, n[1], n[2])] {
        tests.push(b.cmp(next, pred, x, y));
    }
    let f = &b.f;
    let none = BTreeSet::new();
    let built_under = Facts::new(f, &b.inputs, &none);
    let mut checked = Facts::new(f, &b.inputs, &none);
    assert!(checked.uniform[u.0], "the load from one address must start uniform");
    checked.uniform[u.0] = false;
    let built = Structure::of(f, &built_under);
    let kept: BTreeSet<Choice> = BTreeSet::new();
    let mut under_built = Logic::fixed(f, &built_under, &kept, &[]);
    let mut lazy = Logic::fixed(f, &checked, &kept, &[]);
    let mut known = Logic::structured(&built, f, &checked, &kept, &[]);
    let read = |logic: &mut Logic, facts: &Facts, t: ValueId| -> (String, Vec<(Atom, bool)>) {
        let x = logic.bit(f, facts, t);
        let atoms = logic.support(x).iter().map(|&var| (logic.atom_of(var), logic.uniform_atom(facts, var))).collect();
        (canonical(&logic.m, x), atoms)
    };
    let mut differ = 0;
    for &t in &tests {
        let want = read(&mut lazy, &checked, t);
        assert_eq!(read(&mut known, &checked, t), want, "the bit of v{} and which of its atoms are uniform under the check's facts", t.0);
        differ += (read(&mut under_built, &built_under, t) != want) as usize;
    }
    assert!(differ > 0, "the facts must change some comparison for the test to check the structure");
}

#[test]
fn a_structured_logic_reads_every_comparison_and_crossing_as_a_lazy_one() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let yes = b.constant(e, Ty::I1, 1);
    let table = k.buffer(&mut b, e, 8);
    let u = b.load(e, Space::Global, MemSize::B32, table, yes);
    let at = b.constant(e, Ty::I64, 4);
    let second = b.int(e, IntOp::Add, table, at);
    let v = b.load(e, Space::Global, MemSize::B32, second, yes);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let mut tests = Vec::new();
    for (pred, x, y) in [(IntPred::Ult, u, v), (IntPred::Eq, u, v), (IntPred::Ult, lane, u), (IntPred::Ugt, v, lane)] {
        tests.push(b.cmp(e, pred, x, y));
    }
    let five = b.constant(e, Ty::I32, 5);
    tests.push(b.cmp(e, IntPred::Ult, u, five));
    let sum = b.int(e, IntOp::Add, u, lane);
    tests.push(b.cmp(e, IntPred::Ult, sum, v));
    let (next, n) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I1]);
    b.br(e, next, vec![k.exec, u, v, tests[0]]);
    for (pred, x, y) in [(IntPred::Ult, n[1], n[2]), (IntPred::Ugt, n[1], n[2]), (IntPred::Ne, n[1], n[2])] {
        tests.push(b.cmp(next, pred, x, y));
    }
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let structure = Structure::of(f, &facts);
    let mut lazy = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
    let mut known = Logic::structured(&structure, f, &facts, &BTreeSet::new(), &[]);
    let mut cells = 0;
    for &t in &tests {
        let (x, y) = (lazy.bit(f, &facts, t), known.bit(f, &facts, t));
        assert_eq!(canonical(&lazy.m, x), canonical(&known.m, y), "the bit of v{}", t.0);
        cells += lazy.support(x).iter().filter(|&&var| matches!(lazy.atom_of(var), Atom::Cell(..))).count();
    }
    let crossing = |logic: &mut Logic| {
        let x = logic.bit(f, &facts, tests[2]);
        let y = logic.bit(f, &facts, tests[5]);
        let both = logic.m.and(x, y);
        let crossed = logic.image(f, &facts, e, 0, both);
        assert_ne!(crossed, Bdd::FALSE);
        canonical(&logic.m, crossed)
    };
    assert_eq!(crossing(&mut lazy), crossing(&mut known), "the difference a crossing carries");
    assert!(cells > 0, "the comparisons must read cells for the test to check the structure");
}

#[test]
fn choose_keeps_the_first_choices_it_can_convert_and_nothing_it_need_not_keep() {
    let (b, _) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let n = 6;
    let listed: Vec<Choice> = (0..n).map(Choice::Meet).collect();
    let mut r = Random::new(3);
    for _ in 0..300 {
        let mut logic = Logic::open(f, &facts, &listed);
        let markers: Vec<Bdd> = listed.iter().map(|&c| logic.atom(Atom::Marker(c))).collect();
        let table: Vec<bool> = (0..1u32 << n).map(|_| r.below(3) == 0).collect();
        if !table.iter().any(|&x| x) {
            continue;
        }
        let mut safe = Bdd::FALSE;
        for (assignment, &allowed) in table.iter().enumerate() {
            if !allowed {
                continue;
            }
            let mut term = Bdd::TRUE;
            for (i, &m) in markers.iter().enumerate() {
                let literal = if assignment >> i & 1 != 0 { m } else { logic.m.not(m) };
                term = logic.m.and(term, literal);
            }
            safe = logic.m.or(safe, term);
        }
        let kept = logic.choose(safe);
        let local = |kept: &BTreeSet<usize>| (0..n).filter(|i| !kept.contains(i)).fold(0usize, |a, i| a | 1 << i);
        let chosen = local(&kept.meets);
        assert!(table[chosen], "the choice lies outside the safe set");
        for &i in &kept.meets {
            assert!(!table[chosen | 1 << i], "converting Meet({}) alone stays safe", i);
        }
        let mut greedy = 0usize;
        for i in 0..n {
            let with = greedy | 1 << i;
            let fits = (0..1usize << n).any(|a| table[a] && a & ((1 << (i + 1)) - 1) == with);
            if fits {
                greedy = with;
            }
        }
        assert_eq!(chosen, greedy, "the choice is not the greedy one in listed order");
    }
}

#[test]
fn views_in_a_wave_of_64_lanes_hold_the_bit_each_lane_reads() {
    let (mut b, p) = Build::with_lanes(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)], 64);
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let three = b.constant(e, Ty::I32, 3);
    let c = b.cmp(e, IntPred::Ult, p[1], three);
    let low = b.wave(e, WaveOp::Ballot { high: false }, vec![c]);
    let high = b.wave(e, WaveOp::Ballot { high: true }, vec![c]);
    let pair = b.core(e, Ty::I64, Op::Pack64(low, high));
    let wide_lane = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
    let shifted = b.int(e, IntOp::LShr, pair, wide_lane);
    let own = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
    let narrow = b.int(e, IntOp::LShr, low, lane);
    let narrow_own = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, narrow));
    let lo = b.core(e, Ty::I32, Op::UnpackLo(pair));
    let hi = b.core(e, Ty::I32, Op::UnpackHi(pair));
    let mask = b.constant(e, Ty::I64, 0x8000_0001_0000_0002);
    let masked = b.int(e, IntOp::And, pair, mask);
    let two = b.constant(e, Ty::I32, 2);
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    assert!(facts.lane_word[pair.0] && facts.lane_word[lo.0] && facts.lane_word[hi.0], "a pair of ballot halves is a word of lanes");
    let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
    let bc = logic.bit(f, &facts, c);
    let below = logic.lanes(|l| l < 32);
    let above = logic.lanes(|l| l >= 32);
    let restricted = |logic: &mut Logic, g: Bdd, half: Bdd| logic.m.and(g, half);
    let mut expect: Vec<(&str, Bdd, Bdd)> = Vec::new();
    expect.push(("the view of the pair", logic.view(f, &facts, pair), bc));
    expect.push(("trunc(pair >> zext(lane))", logic.bit(f, &facts, own), bc));
    let (vl, vh) = (logic.view(f, &facts, low), logic.view(f, &facts, high));
    let (want_low, want_high) = (restricted(&mut logic, bc, below), restricted(&mut logic, bc, above));
    expect.push(("the low ballot below lane 32", restricted(&mut logic, vl, below), want_low));
    expect.push(("the high ballot from lane 32", restricted(&mut logic, vh, above), want_high));
    let (ul, uh) = (logic.view(f, &facts, lo), logic.view(f, &facts, hi));
    expect.push(("the low word of the pair below lane 32", restricted(&mut logic, ul, below), want_low));
    expect.push(("the high word of the pair from lane 32", restricted(&mut logic, uh, above), want_high));
    let bits = logic.lanes(|l| l == 1 || l == 32 || l == 63);
    let and = logic.m.and(bc, bits);
    expect.push(("pair & 0x8000000100000002", logic.view(f, &facts, masked), and));
    let twos = logic.lanes(|l| l == 1 || l == 33);
    expect.push(("a word of 32 bits repeats in the upper lanes", logic.view(f, &facts, two), twos));
    let wrong: Vec<&str> = expect.iter().filter(|(_, got, want)| got != want).map(|(name, ..)| *name).collect();
    assert!(wrong.is_empty(), "{:?}", wrong);
    let upper_low = restricted(&mut logic, vl, above);
    assert_ne!(upper_low, want_high, "the low ballot does not hold the upper lanes' bits");
    let narrow_bit = logic.bit(f, &facts, narrow_own);
    let narrow_low = restricted(&mut logic, narrow_bit, below);
    assert_eq!(narrow_low, want_low, "the low ballot shifted by the lane holds the lower lanes' bits");
    let narrow_high = restricted(&mut logic, narrow_bit, above);
    assert_ne!(narrow_high, want_high, "shifting a 32-bit word by an upper lane reads a lower lane's bit");
}

#[test]
fn views_of_wide_words_in_a_wave_of_32_lanes_read_the_low_word() {
    let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)]);
    let e = BlockId(0);
    let three = b.constant(e, Ty::I32, 3);
    let c = b.cmp(e, IntPred::Ult, p[1], three);
    let low = b.wave(e, WaveOp::Ballot { high: false }, vec![c]);
    let junk = b.constant(e, Ty::I32, 0x1234_5678);
    let pair = b.core(e, Ty::I64, Op::Pack64(low, junk));
    let lo = b.core(e, Ty::I32, Op::UnpackLo(pair));
    let mask = b.constant(e, Ty::I64, 0x8000_0001_0000_0002);
    let masked = b.int(e, IntOp::And, pair, mask);
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
    let bc = logic.bit(f, &facts, c);
    assert_eq!(logic.view(f, &facts, pair), bc, "the view of a wide word is the view of its low word");
    assert_eq!(logic.view(f, &facts, lo), bc, "the low word of a wide word keeps its view");
    let lane_one = logic.lanes(|l| l == 1);
    let and = logic.m.and(bc, lane_one);
    assert_eq!(logic.view(f, &facts, masked), and, "only the low 32 bits of a wide constant reach a lane");
}

#[test]
fn facts_treat_wide_words_and_the_halves_of_a_wave_of_64_lanes_by_the_lanes_they_cover() {
    let (mut b, p) = Build::with_lanes(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)], 64);
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let three = b.constant(e, Ty::I32, 3);
    let c = b.cmp(e, IntPred::Ult, p[1], three);
    let low = b.wave(e, WaveOp::Ballot { high: false }, vec![c]);
    let high = b.wave(e, WaveOp::Ballot { high: true }, vec![c]);
    let pair = b.core(e, Ty::I64, Op::Pack64(low, high));
    let wide_lane = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
    let shifted = b.int(e, IntOp::LShr, pair, wide_lane);
    let _own = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
    let tested = b.wave(e, WaveOp::Ballot { high: false }, vec![c]);
    let zero = b.constant(e, Ty::I32, 0);
    let _any = b.cmp(e, IntPred::Ne, tested, zero);
    let projected = b.wave(e, WaveOp::Ballot { high: true }, vec![c]);
    let narrow = b.int(e, IntOp::LShr, projected, lane);
    let _bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, narrow));
    let hi = b.core(e, Ty::I32, Op::UnpackHi(pair));
    let ones = b.constant(e, Ty::I64, u64::MAX);
    let mixed = b.core(e, Ty::I64, Op::Select(c, ones, pair));
    let saturated = b.core(e, Ty::I32, Op::UnpackLo(ones));
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    for (name, v) in [("low", low), ("high", high), ("pair", pair), ("tested", tested), ("projected", projected), ("hi", hi)] {
        assert!(facts.lane_word[v.0], "{} is a word of lanes", name);
    }
    assert!(facts.viewed[pair.0] && facts.viewed[low.0] && facts.viewed[high.0], "a projected pair is viewed with its halves");
    assert!(!facts.materialized[pair.0] && !facts.materialized[low.0] && !facts.materialized[high.0], "a pair read only by its lanes stays a view");
    assert!(facts.materialized[tested.0], "a test of a half asks about lanes the half does not cover");
    assert!(facts.materialized[projected.0], "a half shifted by the lane reads another lane's bit above or below it");
    assert!(facts.saturated[ones.0] && facts.saturated[saturated.0], "all-ones words and their halves are saturated");
    assert!(!facts.saturated[mixed.0], "a select by a varying bit is not saturated");
}

#[test]
fn facts_keep_the_high_word_of_a_wide_word_in_a_wave_of_32_lanes_whole() {
    let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)]);
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let three = b.constant(e, Ty::I32, 3);
    let c = b.cmp(e, IntPred::Ult, p[1], three);
    let low = b.wave(e, WaveOp::Ballot { high: false }, vec![c]);
    let zero = b.constant(e, Ty::I32, 0);
    let pair = b.core(e, Ty::I64, Op::Pack64(low, zero));
    let hi = b.core(e, Ty::I32, Op::UnpackHi(pair));
    let shifted = b.int(e, IntOp::LShr, hi, lane);
    let _bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    assert!(facts.lane_word[pair.0], "the pair is a word of lanes");
    assert!(!facts.lane_word[hi.0], "no lane of 32 reads the high word, so it is a plain word");
    assert!(facts.materialized[pair.0], "the high word needs the whole pair");
}

#[test]
fn facts_saturate_a_pair_only_when_both_halves_are_one_saturated_word() {
    let (mut b, p) = Build::with_lanes(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Sgpr(0), Ty::I32)], 64);
    let e = BlockId(0);
    let three = b.constant(e, Ty::I32, 3);
    let c = b.cmp(e, IntPred::Ult, p[1], three);
    let ones = b.constant(e, Ty::I32, 0xffff_ffff);
    let zero = b.constant(e, Ty::I32, 0);
    let s = b.core(e, Ty::I32, Op::Select(c, ones, zero));
    let t = b.core(e, Ty::I32, Op::Select(c, ones, zero));
    let same = b.core(e, Ty::I64, Op::Pack64(s, s));
    let twins = b.core(e, Ty::I64, Op::Pack64(s, t));
    let skew = b.core(e, Ty::I64, Op::Pack64(s, zero));
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    assert!(facts.saturated[s.0] && facts.saturated[t.0], "a uniform choice between all ones and zero is saturated");
    assert!(facts.saturated[same.0], "a pair of one saturated word repeats its bit in every lane");
    assert!(!facts.saturated[twins.0], "two words are not known to agree unless they are one value");
    assert!(!facts.saturated[skew.0], "a pair whose halves may differ is not saturated");
}

fn halves_of_two_conditions() -> (Build, [ValueId; 7]) {
    let (mut b, p) = Build::with_lanes(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)], 64);
    let e = BlockId(0);
    let three = b.constant(e, Ty::I32, 3);
    let five = b.constant(e, Ty::I32, 5);
    let c = b.cmp(e, IntPred::Ult, p[1], three);
    let d = b.cmp(e, IntPred::Ult, p[1], five);
    let low = b.wave(e, WaveOp::Ballot { high: false }, vec![c]);
    let high = b.wave(e, WaveOp::Ballot { high: true }, vec![c]);
    let other = b.wave(e, WaveOp::Ballot { high: true }, vec![d]);
    let kept = b.core(e, Ty::I64, Op::Pack64(low, high));
    let same = b.core(e, Ty::I64, Op::Pack64(low, high));
    let partial = b.core(e, Ty::I64, Op::Pack64(low, other));
    let both = b.int(e, IntOp::And, same, kept);
    let zero = b.constant(e, Ty::I64, 0);
    for w in [kept, same, partial, both] {
        b.cmp(e, IntPred::Ne, w, zero);
    }
    (b, [low, high, other, kept, same, partial, both])
}

#[test]
fn facts_materialize_a_word_built_only_from_materialized_words() {
    let (b, [low, high, other, kept, same, partial, both]) = halves_of_two_conditions();
    let f = &b.f;
    let none = Facts::new(f, &b.inputs, &BTreeSet::new());
    for (name, v) in [("low", low), ("high", high), ("other", other), ("kept", kept), ("same", same), ("partial", partial), ("both", both)] {
        assert!(none.lane_word[v.0], "{} is a word of lanes", name);
        assert!(!none.materialized[v.0], "{} is only tested, so it stays a view while nothing is kept", name);
    }
    let facts = Facts::new(f, &b.inputs, &BTreeSet::from([kept]));
    assert!(facts.materialized[kept.0] && facts.materialized[low.0] && facts.materialized[high.0], "a kept pair needs its halves whole");
    assert!(facts.materialized[same.0], "another pair of the same whole halves is a whole word at no cost");
    assert!(facts.materialized[both.0], "a word of whole words is whole, also through a word made whole on the way");
    assert!(!facts.materialized[other.0], "nothing needs the other half whole");
    assert!(!facts.materialized[partial.0], "a pair with a viewed half stays a view");
}

#[test]
fn the_open_policy_materializes_exactly_what_the_facts_of_each_kept_set_do() {
    let (b, values) = halves_of_two_conditions();
    let f = &b.f;
    let base = Facts::new(f, &b.inputs, &BTreeSet::new());
    let words = values[3..].to_vec();
    let listed: Vec<Choice> = words.iter().map(|&w| Choice::Word(w)).collect();
    let mut logic = Logic::open(f, &base, &listed);
    for subset in 0..1usize << words.len() {
        let kept_words: BTreeSet<ValueId> = (0..words.len()).filter(|i| subset >> i & 1 == 1).map(|i| words[i]).collect();
        let kept: BTreeSet<Choice> = kept_words.iter().map(|&w| Choice::Word(w)).collect();
        let facts = Facts::new(f, &b.inputs, &kept_words);
        for v in values {
            let whole = logic.materialized(&base, v);
            let settled = logic.settled(whole, &kept);
            assert_eq!(settled.constant(), Some(facts.materialized[v.0]), "v{} with {:?} kept", v.0, kept_words);
        }
    }
}

#[derive(Clone)]
enum Float {
    Var(usize),
    Const(u32),
    Sel(usize, Box<Float>, Box<Float>),
}

fn float_value(e: &Float, xs: &[f32; 3], truths: &[bool]) -> f32 {
    match e {
        Float::Var(i) => xs[*i],
        Float::Const(k) => f32::from_bits(*k),
        Float::Sel(c, u, v) => {
            if truths[*c] {
                float_value(u, xs, truths)
            } else {
                float_value(v, xs, truths)
            }
        }
    }
}

fn oracle_name(e: &Float) -> String {
    match e {
        Float::Var(i) => ["x", "y", "z"][*i].to_string(),
        Float::Const(k) => format!("{:?}", f32::from_bits(*k)),
        Float::Sel(c, u, v) => format!("c{} ? {} : {}", c, oracle_name(u), oracle_name(v)),
    }
}

fn emit(b: &mut Build, e: BlockId, vars: &[ValueId], cmps: &[(FloatPred, Float, Float, ValueId)], x: &Float) -> ValueId {
    match x {
        Float::Var(i) => vars[*i],
        Float::Const(k) => b.constant(e, Ty::F32, *k as u64),
        Float::Sel(c, u, v) => {
            let (u, v) = (emit(b, e, vars, cmps, u), emit(b, e, vars, cmps, v));
            b.core(e, Ty::F32, Op::Select(cmps[*c].3, u, v))
        }
    }
}

const ORACLE_SPECIALS: [u32; 18] = [
    0x3f7f_ffff, 0x0080_0000, 0x007f_ffff, 0x8080_0000,
    0x0000_0000, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_0001, 0x0000_0001, 0x8000_0001, 0x7f7f_ffff, 0xff7f_ffff, 0x3f80_0001, 0x4000_0000,
];

#[test]
fn float_rules_hold_for_special_values() {
    use FloatPred::*;
    let preds = [Oeq, Ogt, Oge, Olt, Ole, One, Ord, Uno, Ueq, Ugt, Uge, Ult, Ule, Une];
    let mut r = Random::new(1001);
    let mut wrong = Vec::new();
    let (mut derived, mut total) = (0usize, 0usize);
    for round in 0..1000 {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32), (ParameterSource::Vgpr(2), Ty::I32), (ParameterSource::Vgpr(3), Ty::I32)]);
        let e = BlockId(0);
        let vars: Vec<ValueId> = (1..4).map(|i| b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, p[i]))).collect();
        let mut cmps: Vec<(FloatPred, Float, Float, ValueId)> = Vec::new();
        let mut index: HashMap<ValueId, usize> = HashMap::default();
        let special = |r: &mut Random| ORACLE_SPECIALS[r.below(ORACLE_SPECIALS.len() as u64) as usize];
        let var = |r: &mut Random| if r.below(3) == 0 { r.below(3) as usize } else { 0 };
        for _ in 0..10 {
            let pred = preds[r.below(14) as usize];
            let (a, c): (Float, Float) = match r.below(6) {
                0 | 1 => (Float::Var(var(&mut r)), Float::Const(special(&mut r))),
                2 => (Float::Const(special(&mut r)), Float::Var(var(&mut r))),
                3 => {
                    let i = r.below(3) as usize;
                    let j = (i + 1 + r.below(2) as usize) % 3;
                    (Float::Var(i), Float::Var(j))
                }
                _ if !cmps.is_empty() => {
                    let k = r.below(cmps.len() as u64) as usize;
                    let arm = |r: &mut Random| if r.below(3) == 0 { Float::Var(var(r)) } else { Float::Const(special(r)) };
                    let s = Float::Sel(k, Box::new(arm(&mut r)), Box::new(arm(&mut r)));
                    let other = arm(&mut r);
                    if r.below(2) == 0 {
                        (s, other)
                    } else {
                        (other, s)
                    }
                }
                _ => (Float::Var(0), Float::Const(special(&mut r))),
            };
            let (ea, ec) = (emit(&mut b, e, &vars, &cmps, &a), emit(&mut b, e, &vars, &cmps, &c));
            let v = b.core(e, Ty::I1, Op::FCmp(pred, ea, ec));
            index.insert(v, cmps.len());
            cmps.push((pred, a, c, v));
        }
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let bits: Vec<Bdd> = cmps.iter().map(|c| logic.bit(f, &facts, c.3)).collect();
        for (c, &g) in cmps.iter().zip(&bits) {
            total += 1;
            if g != logic.atom(Atom::Bit(c.3)) {
                derived += 1;
            }
        }
        let mut atoms: HashMap<u32, usize> = HashMap::default();
        for &g in &bits {
            for &var in logic.support(g).iter() {
                match logic.atom_of(var) {
                    Atom::Bit(w) if index.contains_key(&w) => {
                        atoms.insert(var, index[&w]);
                    }
                    other => panic!("unexpected atom {:?} in a float fact", other),
                }
            }
        }
        let mut reported = false;
        for &kx in &ORACLE_SPECIALS {
            for &ky in &ORACLE_SPECIALS {
                for &kz in &ORACLE_SPECIALS {
                    let xs = [f32::from_bits(kx), f32::from_bits(ky), f32::from_bits(kz)];
                    let mut truths: Vec<bool> = Vec::new();
                    for (pred, a, c, _) in &cmps {
                        let (x, y) = (float_value(a, &xs, &truths), float_value(c, &xs, &truths));
                        truths.push(float_compare(*pred, x as f64, y as f64));
                    }
                    for (i, &g) in bits.iter().enumerate() {
                        let got = evaluate(&logic.m, g, &|var| truths[atoms[&var]]);
                        if got != truths[i] && !reported {
                            reported = true;
                            let program: Vec<String> = cmps.iter().enumerate().map(|(j, (p, a, c, _))| format!("c{} = {:?}({}, {})", j, p, oracle_name(a), oracle_name(c))).collect();
                            wrong.push(format!("round {}: c{} is {} at x={:#x} y={:#x} z={:#x}, fact says {}; program {:?}", round, i, truths[i], kx, ky, kz, got, program));
                        }
                    }
                }
            }
        }
    }
    assert!(derived > total / 4, "only {} of {} float facts were derived", derived, total);
    assert!(wrong.is_empty(), "{} wrong: {:#?}", wrong.len(), &wrong[..wrong.len().min(4)]);
}

#[test]
fn a_block_with_many_independent_comparisons_gets_its_bits() {
    let n = 70;
    let mut sources = vec![(ParameterSource::MaskBit(EXEC), Ty::I1)];
    sources.extend((1..=n).map(|i| (ParameterSource::Vgpr(i as u32), Ty::I32)));
    let (mut b, p) = Build::new(&sources);
    let e = BlockId(0);
    let ten = b.constant(e, Ty::I32, 10);
    let tests: Vec<ValueId> = (1..=n).map(|i| b.cmp(e, IntPred::Ult, p[i], ten)).collect();
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
    for &t in &tests {
        let g = logic.bit(f, &facts, t);
        assert!(g != Bdd::TRUE && g != Bdd::FALSE, "v < 10 is open for a vector register");
    }
}

#[test]
fn kept_query_answers_admit_every_wave() {
    let mut r = Random::new(2027);
    let mut wrong = Vec::new();
    let mut nontrivial = 0;
    for round in 0..300 {
        let (mut b, p) = Build::new(&[
            (ParameterSource::MaskBit(EXEC), Ty::I1),
            (ParameterSource::Sgpr(10), Ty::I1),
            (ParameterSource::Sgpr(11), Ty::I1),
            (ParameterSource::MaskBit(106), Ty::I1),
            (ParameterSource::MaskBit(108), Ty::I1),
            (ParameterSource::Vgpr(1), Ty::I32),
            (ParameterSource::Sgpr(12), Ty::I32),
        ]);
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let mut leaves: Vec<ValueId> = vec![p[1], p[2], p[3], p[4]];
        for k in [1u64, 3, 16, 31] {
            let kc = b.constant(e, Ty::I32, k);
            leaves.push(b.cmp(e, IntPred::Ult, lane, kc));
            leaves.push(b.cmp(e, IntPred::Eq, lane, kc));
        }
        for k in [3u64, 5, 9] {
            let kc = b.constant(e, Ty::I32, k);
            leaves.push(b.cmp(e, IntPred::Ult, p[5], kc));
            leaves.push(b.cmp(e, IntPred::Ult, p[6], kc));
        }
        leaves.push(b.cmp(e, IntPred::Ult, p[5], p[6]));
        let mut queries = Vec::new();
        for _ in 0..4 {
            let mut x = leaves[r.below(leaves.len() as u64) as usize];
            for _ in 0..r.below(4) {
                let y = leaves[r.below(leaves.len() as u64) as usize];
                x = match r.below(4) {
                    0 => b.int(e, IntOp::And, x, y),
                    1 => b.int(e, IntOp::Or, x, y),
                    2 => b.int(e, IntOp::Xor, x, y),
                    _ => {
                        let z = leaves[r.below(leaves.len() as u64) as usize];
                        b.core(e, Ty::I1, Op::Select(z, x, y))
                    }
                };
            }
            let x = b.int(e, IntOp::And, x, p[0]);
            let q = b.wave(e, WaveOp::Any, vec![x]);
            queries.push((x, q));
        }
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let kept: BTreeSet<Choice> = queries.iter().map(|&(_, q)| Choice::Query(q)).collect();
        let mut logic = Logic::fixed(f, &facts, &kept, &[]);
        let inputs: Vec<Bdd> = queries.iter().map(|&(x, _)| logic.bit(f, &facts, x)).collect();
        let answers: Vec<Bdd> = queries.iter().map(|&(_, q)| logic.bit(f, &facts, q)).collect();
        let exec = logic.bit(f, &facts, p[0]);
        let mut roots = inputs.clone();
        roots.extend(answers.iter().copied());
        roots.push(exec);
        let mut vars: BTreeSet<u32> = BTreeSet::new();
        for &g in &roots {
            vars.extend(logic.support(g).iter().copied());
        }
        let mut uniform = Vec::new();
        let mut varying = Vec::new();
        for &v in &vars {
            match logic.atom_of(v) {
                Atom::Some(..) => {}
                Atom::Lane(_) => {}
                _ if logic.uniform_atom(&facts, v) => uniform.push(v),
                _ => varying.push(v),
            }
        }
        if answers.iter().any(|&g| logic.support(g).iter().any(|&v| matches!(logic.atom_of(v), Atom::Some(..)))) {
            nontrivial += 1;
        }
        let lanes: Vec<(u32, u8)> = vars.iter().filter_map(|&v| match logic.atom_of(v) { Atom::Lane(i) => Some((v, i)), _ => None }).collect();
        for state in 0..60 {
            let shared: HashMap<u32, bool> = uniform.iter().map(|&v| (v, r.below(2) == 0)).collect();
            let own: Vec<HashMap<u32, bool>> = (0..32u32)
                .map(|l| {
                    let mut m: HashMap<u32, bool> = varying.iter().map(|&v| (v, r.below(4) != 0)).collect();
                    for &(v, i) in &lanes {
                        m.insert(v, l >> i & 1 == 1);
                    }
                    m.extend(shared.iter().map(|(&k, &x)| (k, x)));
                    m
                })
                .collect();
            let value = |logic: &Logic, g: Bdd, l: usize| evaluate(&logic.m, g, &|v| own[l][&v]);
            let active: Vec<usize> = (0..32).filter(|&l| value(&logic, exec, l)).collect();
            if active.is_empty() {
                continue;
            }
            let mut joint = Bdd::TRUE;
            for (k, &g) in answers.iter().enumerate() {
                let truth = active.iter().any(|&l| value(&logic, inputs[k], l));
                let want = if truth { g } else { logic.m.not(g) };
                let want = logic.consistent(want);
                joint = logic.m.and(joint, want);
            }
            let mut admitted = Bdd::TRUE;
            for &l in &active {
                let mut at = joint;
                for (&v, &x) in &own[l] {
                    at = logic.m.cofactor(at, v, x);
                }
                admitted = logic.m.and(admitted, at);
            }
            if admitted == Bdd::FALSE {
                wrong.push(format!("round {} state {}: no answer atoms fit every active lane ({} active)", round, state, active.len()));
                break;
            }
        }
    }
    assert!(nontrivial > 0, "no answer used an answer atom");
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(6)]);
}

#[derive(Clone, Copy, Debug)]
enum Node {
    Param(usize),
    Lane,
    Const(u32),
    Int(IntOp, usize, usize),
    Sel(usize, usize, usize),
    Cmp(IntPred, usize, usize),
    Bool(IntOp, usize, usize),
}

fn integer_wave(r: &mut Random) -> (Build, Vec<(Node, ValueId)>) {
    let (mut b, p) = Build::new(&[
        (ParameterSource::MaskBit(EXEC), Ty::I1),
        (ParameterSource::Vgpr(1), Ty::I32),
        (ParameterSource::Vgpr(2), Ty::I32),
        (ParameterSource::Sgpr(10), Ty::I32),
        (ParameterSource::Sgpr(11), Ty::I32),
    ]);
    let e = BlockId(0);
    let mut nodes: Vec<(Node, ValueId)> = (1..5).map(|i| (Node::Param(i - 1), p[i])).collect();
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    nodes.push((Node::Lane, lane));
    for k in [0u32, 1, 3, 31, 32, 0x8000_0000, 0xffff_ffff] {
        let v = b.constant(e, Ty::I32, k as u64);
        nodes.push((Node::Const(k), v));
    }
    let preds = [IntPred::Eq, IntPred::Ne, IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge];
    for _ in 0..24 {
        let words: Vec<usize> = (0..nodes.len()).filter(|&i| !matches!(nodes[i].0, Node::Cmp(..) | Node::Bool(..))).collect();
        let bits: Vec<usize> = (0..nodes.len()).filter(|&i| matches!(nodes[i].0, Node::Cmp(..) | Node::Bool(..))).collect();
        let pick = |r: &mut Random, from: &[usize]| from[r.below(from.len() as u64) as usize];
        let consts: Vec<usize> = (0..nodes.len()).filter(|&i| matches!(nodes[i].0, Node::Const(_))).collect();
        let node = match r.below(10) {
            0 | 1 | 2 | 3 => Node::Cmp(preds[r.below(10) as usize], pick(r, &words), pick(r, &words)),
            4 if !bits.is_empty() => {
                let op = [IntOp::And, IntOp::Or, IntOp::Xor][r.below(3) as usize];
                Node::Bool(op, pick(r, &bits), pick(r, &bits))
            }
            5 if !bits.is_empty() => Node::Sel(pick(r, &bits), pick(r, &words), pick(r, &words)),
            6 => {
                let op = [IntOp::Mul, IntOp::Shl, IntOp::LShr, IntOp::And][r.below(4) as usize];
                Node::Int(op, pick(r, &words), pick(r, &consts))
            }
            _ => {
                let op = [IntOp::Add, IntOp::Sub, IntOp::Or, IntOp::Xor, IntOp::And][r.below(5) as usize];
                Node::Int(op, pick(r, &words), pick(r, &words))
            }
        };
        let v = match node {
            Node::Cmp(pr, a, c) => b.cmp(e, pr, nodes[a].1, nodes[c].1),
            Node::Bool(op, a, c) | Node::Int(op, a, c) => b.int(e, op, nodes[a].1, nodes[c].1),
            Node::Sel(c, a, d) => b.core(e, Ty::I32, Op::Select(nodes[c].1, nodes[a].1, nodes[d].1)),
            _ => unreachable!(),
        };
        nodes.push((node, v));
    }
    (b, nodes)
}

fn integer_values(nodes: &[(Node, ValueId)], params: &[u32; 4], lane: u32) -> Vec<u32> {
    let mut out: Vec<u32> = Vec::with_capacity(nodes.len());
    for &(node, _) in nodes {
        let x = match node {
            Node::Param(i) => params[i],
            Node::Lane => lane,
            Node::Const(k) => k,
            Node::Int(op, a, c) | Node::Bool(op, a, c) => {
                let (x, y) = (out[a], out[c]);
                match op {
                    IntOp::Add => x.wrapping_add(y),
                    IntOp::Sub => x.wrapping_sub(y),
                    IntOp::Mul => x.wrapping_mul(y),
                    IntOp::And => x & y,
                    IntOp::Or => x | y,
                    IntOp::Xor => x ^ y,
                    IntOp::Shl => x << (y & 31),
                    IntOp::LShr => x >> (y & 31),
                    IntOp::AShr => ((x as i32) >> (y & 31)) as u32,
                }
            }
            Node::Sel(c, a, d) => {
                if out[c] & 1 == 1 {
                    out[a]
                } else {
                    out[d]
                }
            }
            Node::Cmp(p, a, c) => compare(p, out[a], out[c]) as u32,
        };
        out.push(x);
    }
    out
}

#[test]
fn integer_facts_admit_every_wave() {
    let mut r = Random::new(4049);
    let mut wrong = Vec::new();
    let mut derived = 0;
    for round in 0..200 {
        let (b, nodes) = integer_wave(&mut r);
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let tests: Vec<usize> = (0..nodes.len()).filter(|&i| matches!(nodes[i].0, Node::Cmp(..) | Node::Bool(..))).collect();
        let bits: Vec<Bdd> = tests.iter().map(|&i| logic.bit(f, &facts, nodes[i].1)).collect();
        for (&i, &g) in tests.iter().zip(&bits) {
            if g != logic.atom(Atom::Bit(nodes[i].1)) {
                derived += 1;
            }
        }
        let mut vars: BTreeSet<u32> = BTreeSet::new();
        for &g in &bits {
            vars.extend(logic.support(g).iter().copied());
        }
        let lanes: Vec<(u32, u8)> = vars.iter().filter_map(|&v| match logic.atom_of(v) { Atom::Lane(i) => Some((v, i)), _ => None }).collect();
        let varying: Vec<u32> = vars.iter().copied().filter(|&v| !matches!(logic.atom_of(v), Atom::Lane(_)) && !logic.uniform_atom(&facts, v)).collect();
        for state in 0..30 {
            let (s1, s2) = (interesting(&mut r), interesting(&mut r));
            let mut admitted = Bdd::TRUE;
            for l in 0..32u32 {
                let params = [interesting(&mut r), interesting(&mut r), s1, s2];
                let values = integer_values(&nodes, &params, l);
                let mut holds = Bdd::TRUE;
                for (&i, &g) in tests.iter().zip(&bits) {
                    let literal = if values[i] & 1 == 1 { g } else { logic.m.not(g) };
                    holds = logic.m.and(holds, literal);
                }
                for &(v, i) in &lanes {
                    holds = logic.m.cofactor(holds, v, l >> i & 1 == 1);
                }
                let holds = logic.exists(&varying, holds);
                admitted = logic.m.and(admitted, holds);
                if admitted == Bdd::FALSE {
                    let program: Vec<String> = nodes.iter().enumerate().map(|(k, (n, _))| format!("n{} = {:?}", k, n)).collect();
                    wrong.push(format!("round {} state {}: no atoms fit lanes 0..={} (s = {:#x}, {:#x}); {:?}", round, state, l, s1, s2, program));
                    break;
                }
            }
            if admitted == Bdd::FALSE {
                break;
            }
        }
    }
    assert!(derived > 0);
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(2)]);
}

#[test]
fn bit_of_a_hashed_word_comparison_finishes() {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32)]);
        let e = BlockId(0);
        let hash = b.constant(e, Ty::I32, 0x27d4_eb2d);
        let hashed = b.int(e, IntOp::Mul, p[1], hash);
        let bound = b.constant(e, Ty::I32, 1000);
        let small = b.cmp(e, IntPred::Ult, hashed, bound);
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(&b.f, &facts, &BTreeSet::new(), &[]);
        let g = logic.bit(&b.f, &facts, small);
        let _ = done.send(g != Bdd::TRUE && g != Bdd::FALSE);
    });
    let open = finished.recv_timeout(std::time::Duration::from_secs(20)).expect("bit(v * 0x27d4eb2d < 1000) did not finish within 20 s");
    assert!(open, "the comparison stays open");
}

#[test]
fn bit_after_a_chain_of_compare_exchanges_finishes() {
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32), (ParameterSource::Vgpr(2), Ty::I32)]);
        let e = BlockId(0);
        let (mut lo, mut hi) = (p[1], p[2]);
        for _ in 0..24 {
            let c = b.cmp(e, IntPred::Ult, lo, hi);
            let (l, h) = (b.core(e, Ty::I32, Op::Select(c, lo, hi)), b.core(e, Ty::I32, Op::Select(c, hi, lo)));
            lo = l;
            hi = h;
        }
        let bound = b.constant(e, Ty::I32, 10);
        let small = b.cmp(e, IntPred::Ult, lo, bound);
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(&b.f, &facts, &BTreeSet::new(), &[]);
        let g = logic.bit(&b.f, &facts, small);
        let _ = done.send(g != Bdd::TRUE && g != Bdd::FALSE);
    });
    let open = finished.recv_timeout(std::time::Duration::from_secs(20)).expect("bit(lo < 10) after 24 compare-exchanges did not finish within 20 s");
    assert!(open, "the comparison stays open");
}

#[test]
fn post_keeps_the_successor_of_every_concrete_state() {
    let preds = [IntPred::Eq, IntPred::Ne, IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge];
    let mut r = Random::new(5051);
    let mut wrong = Vec::new();
    let mut bridged = 0;
    for round in 0..300 {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::Vgpr(1), Ty::I32), (ParameterSource::Vgpr(2), Ty::I32), (ParameterSource::Sgpr(10), Ty::I32)]);
        let e = BlockId(0);
        let one = b.constant(e, Ty::I32, 1);
        let x1 = b.int(e, IntOp::Add, p[1], one);
        let ys = b.int(e, IntOp::Sub, p[2], p[3]);
        let mut src_nodes: Vec<(Node, ValueId)> = vec![(Node::Param(0), p[1]), (Node::Param(1), p[2]), (Node::Param(2), p[3])];
        src_nodes.push((Node::Const(1), one));
        src_nodes.push((Node::Int(IntOp::Add, 0, 3), x1));
        src_nodes.push((Node::Int(IntOp::Sub, 1, 2), ys));
        for k in [5u32, 0x8000_0000] {
            let c = b.constant(e, Ty::I32, k as u64);
            src_nodes.push((Node::Const(k), c));
        }
        let words = src_nodes.len();
        for _ in 0..5 {
            let (i, j) = (r.below(words as u64) as usize, r.below(words as u64) as usize);
            let pr = preds[r.below(10) as usize];
            let v = b.cmp(e, pr, src_nodes[i].1, src_nodes[j].1);
            src_nodes.push((Node::Cmp(pr, i, j), v));
        }
        let (dst, q) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I32]);
        let picks: Vec<usize> = (0..3).map(|_| r.below(words as u64) as usize).collect();
        b.br(e, dst, vec![p[0], src_nodes[picks[0]].1, src_nodes[picks[1]].1, src_nodes[picks[2]].1]);
        let mut dst_nodes: Vec<(Node, ValueId)> = (0..3).map(|i| (Node::Param(i), q[i + 1])).collect();
        let d1 = b.constant(dst, Ty::I32, 1);
        dst_nodes.push((Node::Const(1), d1));
        let a1 = b.int(dst, IntOp::Add, q[1], d1);
        dst_nodes.push((Node::Int(IntOp::Add, 0, 3), a1));
        for k in [5u32, 0x8000_0000] {
            let c = b.constant(dst, Ty::I32, k as u64);
            dst_nodes.push((Node::Const(k), c));
        }
        let dwords = dst_nodes.len();
        for _ in 0..5 {
            let (i, j) = (r.below(dwords as u64) as usize, r.below(dwords as u64) as usize);
            let pr = preds[r.below(10) as usize];
            let v = b.cmp(dst, pr, dst_nodes[i].1, dst_nodes[j].1);
            dst_nodes.push((Node::Cmp(pr, i, j), v));
        }
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let src_tests: Vec<usize> = (words..src_nodes.len()).collect();
        let dst_tests: Vec<usize> = (dwords..dst_nodes.len()).collect();
        let src_bits: Vec<Bdd> = src_tests.iter().map(|&i| logic.bit(f, &facts, src_nodes[i].1)).collect();
        let dst_bits: Vec<Bdd> = dst_tests.iter().map(|&i| logic.bit(f, &facts, dst_nodes[i].1)).collect();
        let terms: Vec<Vec<(usize, bool)>> = (0..3).map(|_| (0..2).map(|_| (r.below(src_tests.len() as u64) as usize, r.below(2) == 0)).collect()).collect();
        let mut formula = Bdd::FALSE;
        for term in &terms {
            let mut t = Bdd::TRUE;
            for &(i, positive) in term {
                let g = if positive { src_bits[i] } else { logic.m.not(src_bits[i]) };
                t = logic.m.and(t, g);
            }
            formula = logic.m.or(formula, t);
        }
        let image = logic.post(f, &facts, e, 0, formula);
        if logic.support(image).iter().any(|&v| matches!(logic.atom_of(v), Atom::Cell(..))) {
            bridged += 1;
        }
        for _ in 0..200 {
            let params = [interesting(&mut r), interesting(&mut r), interesting(&mut r), 0];
            let src_values = integer_values(&src_nodes, &params, 0);
            let holds = terms.iter().any(|t| t.iter().all(|&(i, positive)| (src_values[src_tests[i]] & 1 == 1) == positive));
            if !holds {
                continue;
            }
            let args = [src_values[picks[0]], src_values[picks[1]], src_values[picks[2]], 0];
            let dst_values = integer_values(&dst_nodes, &args, 0);
            let mut state = image;
            for (k, &i) in dst_tests.iter().enumerate() {
                let g = if dst_values[i] & 1 == 1 { dst_bits[k] } else { logic.m.not(dst_bits[k]) };
                state = logic.m.and(state, g);
            }
            let exec = logic.atom(Atom::Bit(q[0]));
            state = logic.m.and(state, exec);
            if state == Bdd::FALSE {
                let program: Vec<String> = src_nodes.iter().chain(&dst_nodes).map(|(n, _)| format!("{:?}", n)).collect();
                wrong.push(format!("round {}: the successor of src {:?} (args {:?}) leaves the image; picks {:?} terms {:?}; {:?}", round, params, args, picks, terms, program));
                break;
            }
        }
    }
    assert!(bridged > 0, "no image related cells across the edge");
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(2)]);
}

#[test]
fn wave_answer_over_many_uniform_bits_does_not_panic() {
    let n = 11;
    let mut sources = vec![(ParameterSource::MaskBit(EXEC), Ty::I1)];
    sources.extend((0..n).map(|i| (ParameterSource::Sgpr(10 + i as u32), Ty::I1)));
    sources.extend((0..n).map(|i| (ParameterSource::MaskBit(20 + 2 * i as u32), Ty::I1)));
    let (mut b, p) = Build::new(&sources);
    let e = BlockId(0);
    let mut x = b.constant(e, Ty::I1, 0);
    for i in 0..n {
        let both = b.int(e, IntOp::And, p[1 + i], p[1 + n + i]);
        x = b.int(e, IntOp::Or, x, both);
    }
    let x = b.int(e, IntOp::And, x, p[0]);
    let q = b.wave(e, WaveOp::Any, vec![x]);
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let kept: BTreeSet<Choice> = std::iter::once(Choice::Query(q)).collect();
    let mut logic = Logic::fixed(f, &facts, &kept, &[]);
    let g = logic.bit(f, &facts, q);
    assert_ne!(g, Bdd::FALSE);
}

#[test]
#[should_panic(expected = "keep changes only which meetings are kept")]
fn keep_refuses_to_change_which_queries_are_kept() {
    let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::MaskBit(106), Ty::I1)]);
    let e = BlockId(0);
    let x = b.int(e, IntOp::And, p[1], p[0]);
    let q = b.wave(e, WaveOp::Any, vec![x]);
    let f = &b.f;
    let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
    let converted = logic.bit(f, &facts, q);
    assert_eq!(converted, logic.bit(f, &facts, x), "a converted query reads the lane's own bit");
    let kept: BTreeSet<Choice> = std::iter::once(Choice::Query(q)).collect();
    logic.keep(&kept);
}

#[test]
fn keep_changes_which_meetings_are_kept() {
    let (b, _) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1)]);
    let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
    let mut logic = Logic::fixed(&b.f, &facts, &BTreeSet::new(), &[Choice::Meet(0)]);
    assert_eq!(logic.local(Choice::Meet(0)), Bdd::TRUE);
    logic.keep(&std::iter::once(Choice::Meet(0)).collect());
    assert_eq!(logic.local(Choice::Meet(0)), Bdd::FALSE);
}

#[derive(Clone, Copy, Debug)]
enum OracleWord {
    Bit(usize),
    LaneBelow(u32),
    Not(usize),
    Ballot(usize),
    Const(u32),
    OneShl,
    Sel(usize, usize, usize),
    Logic(IntOp, usize, usize),
    Cast(usize),
    Own(usize),
}

#[test]
fn views_hold_in_every_lane() {
    let mut r = Random::new(6067);
    let mut wrong = Vec::new();
    for round in 0..400 {
        let (mut b, p) = Build::new(&[(ParameterSource::MaskBit(EXEC), Ty::I1), (ParameterSource::MaskBit(106), Ty::I1), (ParameterSource::MaskBit(108), Ty::I1), (ParameterSource::Sgpr(10), Ty::I1)]);
        let e = BlockId(0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let one = b.constant(e, Ty::I32, 1);
        let mut nodes: Vec<(OracleWord, ValueId, bool)> = (1..4).map(|i| (OracleWord::Bit(i - 1), p[i], false)).collect();
        for k in [3u32, 17] {
            let kc = b.constant(e, Ty::I32, k as u64);
            nodes.push((OracleWord::LaneBelow(k), b.cmp(e, IntPred::Ult, lane, kc), false));
        }
        let shl = b.int(e, IntOp::Shl, one, lane);
        nodes.push((OracleWord::OneShl, shl, true));
        for _ in 0..16 {
            let bits: Vec<usize> = (0..nodes.len()).filter(|&i| !nodes[i].2).collect();
            let words: Vec<usize> = (0..nodes.len()).filter(|&i| nodes[i].2).collect();
            let pick = |r: &mut Random, from: &[usize]| from[r.below(from.len() as u64) as usize];
            let (node, v, word) = match r.below(9) {
                0 | 1 => {
                    let x = pick(&mut r, &bits);
                    (OracleWord::Ballot(x), b.wave(e, WaveOp::Ballot { high: false }, vec![nodes[x].1]), true)
                }
                2 => {
                    let k = interesting(&mut r);
                    (OracleWord::Const(k), b.constant(e, Ty::I32, k as u64), true)
                }
                3 => {
                    let (c, x, y) = (pick(&mut r, &bits), pick(&mut r, &words), pick(&mut r, &words));
                    (OracleWord::Sel(c, x, y), b.core(e, Ty::I32, Op::Select(nodes[c].1, nodes[x].1, nodes[y].1)), true)
                }
                4 | 5 => {
                    let op = [IntOp::And, IntOp::Or, IntOp::Xor][r.below(3) as usize];
                    let (x, y) = (pick(&mut r, &words), pick(&mut r, &words));
                    (OracleWord::Logic(op, x, y), b.int(e, op, nodes[x].1, nodes[y].1), true)
                }
                6 => {
                    let x = pick(&mut r, &words);
                    (OracleWord::Cast(x), b.core(e, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, nodes[x].1)), true)
                }
                7 => {
                    let x = pick(&mut r, &words);
                    let s = b.int(e, IntOp::LShr, nodes[x].1, lane);
                    (OracleWord::Own(x), b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, s)), false)
                }
                _ => {
                    let x = pick(&mut r, &bits);
                    let t = b.constant(e, Ty::I1, 1);
                    (OracleWord::Not(x), b.int(e, IntOp::Xor, nodes[x].1, t), false)
                }
            };
            nodes.push((node, v, word));
        }
        let f = &b.f;
        let facts = Facts::new(f, &b.inputs, &BTreeSet::new());
        let mut logic = Logic::fixed(f, &facts, &BTreeSet::new(), &[]);
        let facts_of: Vec<Bdd> = nodes.iter().map(|&(_, v, word)| if word { logic.view(f, &facts, v) } else { logic.bit(f, &facts, v) }).collect();
        for _ in 0..30 {
            let own: Vec<[bool; 3]> = (0..32).map(|_| [r.below(2) == 0, r.below(2) == 0, false]).collect();
            let shared = r.below(2) == 0;
            let mut values: Vec<[u32; 32]> = Vec::new();
            for &(node, _, _) in &nodes {
                let at = |values: &Vec<[u32; 32]>, i: usize, l: usize| values[i][l];
                let mut out = [0u32; 32];
                for l in 0..32 {
                    out[l] = match node {
                        OracleWord::Bit(i) => (if i == 2 { shared } else { own[l][i] }) as u32,
                        OracleWord::LaneBelow(k) => ((l as u32) < k) as u32,
                        OracleWord::Not(x) => at(&values, x, l) ^ 1,
                        OracleWord::Ballot(x) => (0..32).filter(|&m| at(&values, x, m) == 1).fold(0u32, |w, m| w | 1 << m),
                        OracleWord::Const(k) => k,
                        OracleWord::OneShl => 1u32 << l,
                        OracleWord::Sel(c, x, y) => if at(&values, c, l) == 1 { at(&values, x, l) } else { at(&values, y, l) },
                        OracleWord::Logic(op, x, y) => match op {
                            IntOp::And => at(&values, x, l) & at(&values, y, l),
                            IntOp::Or => at(&values, x, l) | at(&values, y, l),
                            _ => at(&values, x, l) ^ at(&values, y, l),
                        },
                        OracleWord::Cast(x) => at(&values, x, l),
                        OracleWord::Own(x) => at(&values, x, l) >> l & 1,
                    };
                }
                values.push(out);
            }
            let mut reported = false;
            for l in 0..32usize {
                let assignment = |var: u32| match logic.atom_of(var) {
                    Atom::Bit(v) if v == p[1] => own[l][0],
                    Atom::Bit(v) if v == p[2] => own[l][1],
                    Atom::Bit(v) if v == p[3] => shared,
                    Atom::Lane(i) => l >> i & 1 == 1,
                    other => panic!("unexpected atom {:?}", other),
                };
                for (k, &(node, _, word)) in nodes.iter().enumerate() {
                    let truth = if word { values[k][l] >> l & 1 == 1 } else { values[k][l] == 1 };
                    let got = evaluate(&logic.m, facts_of[k], &assignment);
                    if got != truth && !reported {
                        reported = true;
                        wrong.push(format!("round {}: node {} = {:?} at lane {} is {}, fact says {}", round, k, node, l, truth, got));
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{} wrong: {:?}", wrong.len(), &wrong[..wrong.len().min(4)]);
}

fn no_hazards() -> super::super::hazard::Hazards {
    super::super::hazard::Hazards {
        accesses: Vec::new(),
        together: BTreeSet::new(),
        apart: BTreeSet::new(),
        idle: BTreeSet::new(),
        meetings: Vec::new(),
    }
}

fn query_fed_through_a_parameter() -> (Build, ValueId, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flags = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, flags, lane, 4);
    let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let a = b.wave(e, WaveOp::Any, vec![c]);
    let out = k.buffer(&mut b, e, 0);
    let addr = byte_offset(&mut b, e, out, lane, 8);
    let (next, p) = b.block(&[Ty::I1, Ty::I1, Ty::I64]);
    b.br(e, next, vec![k.exec, a, addr]);
    let m = b.int(next, IntOp::And, p[1], p[0]);
    let q = b.wave(next, WaveOp::Any, vec![m]);
    let x = b.int(next, IntOp::Xor, q, p[1]);
    let one = b.constant(next, Ty::I32, 1);
    let two = b.constant(next, Ty::I32, 2);
    let first = b.core(next, Ty::I32, Op::Select(x, one, two));
    b.store(next, Space::Global, MemSize::B32, p[2], first, p[0]);
    let four = b.constant(next, Ty::I64, 4);
    let second_addr = b.int(next, IntOp::Add, p[2], four);
    let second = b.core(next, Ty::I32, Op::Select(q, one, two));
    b.store(next, Space::Global, MemSize::B32, second_addr, second, p[0]);
    (b, a, q)
}

#[test]
fn prove_keeps_a_query_whose_answer_a_parameter_carries_into_a_kept_query() {
    let (b, a, q) = query_fed_through_a_parameter();
    let mut wrong = Vec::new();
    let searched = super::super::search::prove(&b.f, &b.inputs, Some(0), &no_hazards()).0;
    let direct = super::super::direct::prove(&b.f, &b.inputs, Some(0), &no_hazards()).0;
    for (name, kept) in [("search", searched), ("direct", direct)] {
        if !kept.queries.contains(&a) {
            wrong.push(format!("{}: keeps {:?} but not A = {:?} (B = {:?})", name, kept.queries, a, q));
        }
    }
    assert!(wrong.is_empty(), "{:?}", wrong);
}
