use super::super::logic::{float_compare, Atom, Kept};
use super::super::testing::*;
use super::super::{direct, search};
use super::*;
use crate::rdna_spmd::hash::HashMap;

fn no_hazards() -> Hazards {
    Hazards {
        accesses: Vec::new(),
        together: BTreeSet::new(),
        apart: BTreeSet::new(),
        idle: BTreeSet::new(),
        meetings: Vec::new(),
    }
}

#[derive(Clone, Copy, Debug)]
enum Reader {
    ReadLane,
    ReadFirstLane,
    WriteLane,
    Bpermute,
    BpermuteFi,
    Wmma,
}

fn exchange(b: &mut Build, e: BlockId, reader: Reader, x: ValueId, exec: ValueId) -> ValueId {
    let zero = b.constant(e, Ty::I32, 0);
    match reader {
        Reader::ReadLane => b.wave(e, WaveOp::ReadLane, vec![x, zero, zero]),
        Reader::ReadFirstLane => b.wave(e, WaveOp::ReadFirstLane, vec![x, exec]),
        Reader::WriteLane => {
            let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
            let one = b.constant(e, Ty::I32, 1);
            b.wave(e, WaveOp::WriteLane, vec![x, one, lane, zero])
        }
        Reader::Bpermute => b.wave(e, WaveOp::Bpermute, vec![zero, x, exec]),
        Reader::BpermuteFi => b.wave(e, WaveOp::BpermuteFi, vec![zero, x, exec]),
        Reader::Wmma => {
            let fzero = b.constant(e, Ty::F32, 0);
            let mut inputs = vec![x; 8];
            inputs.extend([fzero; 8]);
            let outputs = b.effect(e, EffectOp::Wave(WaveOp::Wmma), inputs);
            b.core(e, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, outputs[0]))
        }
    }
}

fn reads_another_lane(reader: Reader) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let flags = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, flags, lane, 4);
    let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
    let other = match reader {
        Reader::WriteLane => zero,
        _ => lane,
    };
    let x = b.core(e, Ty::I32, Op::Select(q, ones, other));
    let y = exchange(&mut b, e, reader, x, k.exec);
    let address = byte_offset(&mut b, e, buf, lane, 4);
    b.store(e, Space::Global, MemSize::B32, address, y, c);
    (b, q)
}

fn reads_outside_exec(reader: Reader) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
    let idle = b.core(e, Ty::I32, Op::Select(q, ones, lane));
    let x = b.core(e, Ty::I32, Op::Select(k.exec, lane, idle));
    let y = exchange(&mut b, e, reader, x, k.exec);
    store_own(&mut b, &k, e, y, k.exec);
    (b, q)
}

struct Flagged {
    b: Build,
    k: Kernel,
    buf: ValueId,
    lane: ValueId,
    c: ValueId,
}

fn flagged() -> Flagged {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let flags = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, flags, lane, 4);
    let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    Flagged {
        b,
        k,
        buf,
        lane,
        c,
    }
}

fn both(b: &Build) -> [Kept; 2] {
    [
        search::prove(&b.f, &b.inputs, Some(0), &no_hazards()).0,
        direct::prove(&b.f, &b.inputs, Some(0), &no_hazards()).0,
    ]
}

#[test]
fn prove_keeps_a_query_that_decides_whether_lanes_store() {
    let Flagged { mut b, k, buf, lane, c, .. } = flagged();
    let e = BlockId(0);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let (then, t) = b.block(&[Ty::I1, Ty::I64]);
    let (join, _) = b.block(&[Ty::I1]);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    b.cond_br(e, q, (then, vec![k.exec, own]), (join, vec![k.exec]));
    let one = b.constant(then, Ty::I32, 1);
    b.store(then, Space::Global, MemSize::B32, t[1], one, t[0]);
    b.br(then, join, vec![t[0]]);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag skips the store the wave makes", wrong);
}

#[test]
fn prove_keeps_a_query_that_moves_the_address() {
    let Flagged { mut b, k, buf, lane, c, .. } = flagged();
    let e = BlockId(0);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let near = b.constant(e, Ty::I64, 0);
    let far = b.constant(e, Ty::I64, 4);
    let shift = b.core(e, Ty::I64, Op::Select(q, near, far));
    let own = byte_offset(&mut b, e, buf, lane, 8);
    let address = b.int(e, IntOp::Add, own, shift);
    let one = b.constant(e, Ty::I32, 1);
    b.store(e, Space::Global, MemSize::B32, address, one, k.exec);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag stores four bytes further", wrong);
}

#[test]
fn prove_keeps_a_query_that_picks_the_word_a_lane_loads() {
    let Flagged { mut b, k, buf, lane, c, .. } = flagged();
    let e = BlockId(0);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let table = k.buffer(&mut b, e, 16);
    let four = b.constant(e, Ty::I64, 4);
    let second = b.int(e, IntOp::Add, table, four);
    let from = b.core(e, Ty::I64, Op::Select(q, table, second));
    let v = b.load(e, Space::Global, MemSize::B32, from, k.exec);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    b.store(e, Space::Global, MemSize::B32, own, v, k.exec);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag loads the second word of the table", wrong);
}

#[test]
fn prove_keeps_a_query_that_masks_a_store() {
    let Flagged { mut b, k, buf, lane, c, .. } = flagged();
    let e = BlockId(0);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let mask = b.int(e, IntOp::And, q, k.exec);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let one = b.constant(e, Ty::I32, 1);
    b.store(e, Space::Global, MemSize::B32, own, one, mask);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag skips its store", wrong);
}

#[test]
fn prove_keeps_a_ballot_whose_lane_test_masks_a_store() {
    let Flagged { mut b, k, buf, lane, c, .. } = flagged();
    let e = BlockId(0);
    let w = b.wave(e, WaveOp::Ballot, vec![c]);
    let zero = b.constant(e, Ty::I32, 0);
    let any = b.cmp(e, IntPred::Ne, w, zero);
    let mask = b.int(e, IntOp::And, any, k.exec);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let one = b.constant(e, Ty::I32, 1);
    b.store(e, Space::Global, MemSize::B32, own, one, mask);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.words.contains(&w))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag reads its own bit of the ballot as zero", wrong);
}

#[test]
fn prove_keeps_a_query_that_ends_a_loop() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let flags = k.buffer(&mut b, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64, Ty::I64]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, zero, buf, flags]);
    let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
    let wave = b.constant(body, Ty::I32, 32);
    let row = b.int(body, IntOp::Mul, p[1], wave);
    let item = b.int(body, IntOp::Add, row, lane);
    let own = byte_offset(&mut b, body, p[3], item, 4);
    let flag = b.load(body, Space::Global, MemSize::B32, own, p[0]);
    let z = b.constant(body, Ty::I32, 0);
    let set = b.cmp(body, IntPred::Ne, flag, z);
    let c = b.int(body, IntOp::And, set, p[0]);
    let out = byte_offset(&mut b, body, p[2], item, 4);
    let one = b.constant(body, Ty::I32, 1);
    b.store(body, Space::Global, MemSize::B32, out, one, p[0]);
    let next = b.int(body, IntOp::Add, p[1], one);
    let q = b.wave(body, WaveOp::Any, vec![c]);
    b.cond_br(body, q, (body, vec![p[0], next, p[2], p[3]]), (exit, vec![p[0]]));
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag leaves the loop while the wave stores another row", wrong);
}

fn carried_query(latch: bool) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let flags = k.buffer(&mut b, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let yes = b.constant(e, Ty::I1, 1);
    let shape = [Ty::I1, Ty::I1, Ty::I32, Ty::I64, Ty::I64];
    let (body, p) = b.block(&shape);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, yes, zero, buf, flags]);
    let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
    let wave = b.constant(body, Ty::I32, 32);
    let row = b.int(body, IntOp::Mul, p[2], wave);
    let item = b.int(body, IntOp::Add, row, lane);
    let own = byte_offset(&mut b, body, p[4], item, 4);
    let always = b.constant(body, Ty::I1, 1);
    let flag = b.load(body, Space::Global, MemSize::B32, own, always);
    let z = b.constant(body, Ty::I32, 0);
    let c = b.cmp(body, IntPred::Ne, flag, z);
    let q = b.wave(body, WaveOp::Any, vec![c]);
    let mask = b.int(body, IntOp::And, p[1], p[0]);
    let out = byte_offset(&mut b, body, p[3], item, 4);
    let one = b.constant(body, Ty::I32, 1);
    b.store(body, Space::Global, MemSize::B32, out, one, mask);
    let next = b.int(body, IntOp::Add, p[2], one);
    let four = b.constant(body, Ty::I32, 4);
    let below = b.cmp(body, IntPred::Ult, next, four);
    let again = b.int(body, IntOp::And, p[1], below);
    let back = vec![p[0], q, next, p[3], p[4]];
    if latch {
        let (latch, l) = b.block(&shape);
        b.cond_br(body, again, (latch, back), (exit, vec![p[0]]));
        b.br(latch, body, l);
    } else {
        b.cond_br(body, again, (body, back), (exit, vec![p[0]]));
    }
    (b, q)
}

#[test]
fn prove_keeps_a_query_a_self_loop_carries_into_its_own_condition() {
    let (b, q) = carried_query(false);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag in row 0 leaves the loop and skips row 1, which the wave stores", wrong);
}

#[test]
fn prove_keeps_a_query_a_loop_with_a_latch_carries_into_its_own_condition() {
    let (b, q) = carried_query(true);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag in row 0 leaves the loop and skips row 1, which the wave stores", wrong);
}

fn detour_home(join: bool) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let flags = k.buffer(&mut b, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let yes = b.constant(e, Ty::I1, 1);
    let head_shape = [Ty::I1, Ty::I1, Ty::I32, Ty::I64, Ty::I64];
    let arm_shape = [Ty::I1, Ty::I32, Ty::I64, Ty::I64];
    let (head, h) = b.block(&head_shape);
    let (taken, t) = b.block(&arm_shape);
    let (other, o) = b.block(&arm_shape);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, head, vec![k.exec, yes, zero, buf, flags]);
    let lane = b.core(head, Ty::I32, Op::Env(Env::LaneId));
    let wave = b.constant(head, Ty::I32, 32);
    let row = b.int(head, IntOp::Mul, h[2], wave);
    let item = b.int(head, IntOp::Add, row, lane);
    let out = byte_offset(&mut b, head, h[3], item, 4);
    let mask = b.int(head, IntOp::And, h[1], h[0]);
    let one = b.constant(head, Ty::I32, 1);
    b.store(head, Space::Global, MemSize::B32, out, one, mask);
    let own = byte_offset(&mut b, head, h[4], item, 4);
    let always = b.constant(head, Ty::I1, 1);
    let flag = b.load(head, Space::Global, MemSize::B32, own, always);
    let z = b.constant(head, Ty::I32, 0);
    let c = b.cmp(head, IntPred::Ne, flag, z);
    let q = b.wave(head, WaveOp::Any, vec![c]);
    let go = b.int(head, IntOp::And, q, h[1]);
    let next = b.int(head, IntOp::Add, h[2], one);
    b.cond_br(head, go, (taken, vec![h[0], next, h[3], h[4]]), (other, vec![h[0], next, h[3], h[4]]));
    let on = b.constant(taken, Ty::I1, 1);
    let off = b.constant(other, Ty::I1, 0);
    if join {
        let (meet, m) = b.block(&head_shape);
        b.br(taken, meet, vec![t[0], on, t[1], t[2], t[3]]);
        let four = b.constant(other, Ty::I32, 4);
        let below = b.cmp(other, IntPred::Ult, o[1], four);
        b.cond_br(other, below, (meet, vec![o[0], off, o[1], o[2], o[3]]), (exit, vec![o[0]]));
        b.br(meet, head, m);
    } else {
        b.br(taken, head, vec![t[0], on, t[1], t[2], t[3]]);
        let four = b.constant(other, Ty::I32, 4);
        let below = b.cmp(other, IntPred::Ult, o[1], four);
        b.cond_br(other, below, (head, vec![o[0], off, o[1], o[2], o[3]]), (exit, vec![o[0]]));
    }
    (b, q)
}

#[test]
fn prove_keeps_a_query_whose_detour_ends_where_it_began() {
    let (b, q) = detour_home(false);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag takes the other arm, clears its bit and skips row 1, which the wave stores", wrong);
}

#[test]
fn prove_keeps_a_query_whose_detour_ends_at_a_join_before_the_header() {
    let (b, q) = detour_home(true);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag takes the other arm, clears its bit and skips row 1, which the wave stores", wrong);
}

#[test]
fn prove_demands_every_lane_for_a_kept_query_over_unmasked_bits() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let flags = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, flags, lane, 4);
    let yes = b.constant(e, Ty::I1, 1);
    let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let at = b.here(e);
    let q = b.wave(e, WaveOp::Any, vec![set]);
    let one = b.constant(e, Ty::I32, 1);
    let data = b.core(e, Ty::I32, Op::Select(q, one, zero));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    b.store(e, Space::Global, MemSize::B32, own, data, k.exec);
    let Inst::Effect { provenance, .. } = b.f.blocks[&e].insts[at.1] else { unreachable!() };
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
        assert!(kept.queries.contains(&q), "{}: a lane without the flag stores zero", name);
        assert!(everyone.contains(&provenance), "{}: the query reads the flags of lanes whose exec is clear", name);
    }
}

type Position = (BlockId, usize);

fn meetings(program: &crate::rdna_spmd::program::Program, hazards: &Hazards) -> Vec<(&'static str, BTreeSet<Position>)> {
    let (f, inputs) = (&program.ir, &program.parameter_inputs);
    let search = search::prove(f, inputs, Some(0), hazards).0;
    let direct = direct::prove(f, inputs, Some(0), hazards).0;
    vec![("search", search), ("direct", direct)]
        .into_iter()
        .map(|(name, kept)| (name, kept.meets.iter().map(|&m| hazards.meetings[m]).collect()))
        .collect()
}

fn keeps_exactly(program: &crate::rdna_spmd::program::Program, hazards: &Hazards, expected: &[Position], why: &str) {
    let expected: BTreeSet<Position> = expected.iter().copied().collect();
    let wrong: Vec<(&str, BTreeSet<Position>)> =
        meetings(program, hazards).into_iter().filter(|(_, kept)| *kept != expected).collect();
    assert!(wrong.is_empty(), "{}: expected meetings before {:?}, kept {:?}", why, expected, wrong);
}

fn store(b: &mut Build, block: BlockId, address: ValueId, data: u64, mask: ValueId) -> Position {
    let value = b.constant(block, Ty::I32, data);
    let at = b.here(block);
    b.store(block, Space::Global, MemSize::B32, address, value, mask);
    at
}

#[test]
fn prove_orders_two_stores_to_one_word_with_one_meeting() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let s1 = store(&mut b, e, buf, 1, k.exec);
    let s2 = store(&mut b, e, buf, 2, k.exec);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    keeps_exactly(&program, &hazards, &[s2], "every lane's second store must follow every lane's first");
}

#[test]
fn prove_orders_three_stores_to_one_word_with_the_two_meetings_between_them() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let s1 = store(&mut b, e, buf, 1, k.exec);
    let s2 = store(&mut b, e, buf, 2, k.exec);
    let s3 = store(&mut b, e, buf, 3, k.exec);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(s1, s2), (s2, s3), (s1, s3)], &[], &[]);
    keeps_exactly(&program, &hazards, &[s2, s3], "each store to the word must follow the one before it");
}

#[test]
fn prove_orders_stores_of_disjoint_lane_halves() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let sixteen = b.constant(e, Ty::I32, 16);
    let low = b.cmp(e, IntPred::Ult, lane, sixteen);
    let high = b.cmp(e, IntPred::Uge, lane, sixteen);
    let low = b.int(e, IntOp::And, low, k.exec);
    let high = b.int(e, IntOp::And, high, k.exec);
    let s1 = store(&mut b, e, buf, 1, low);
    let s2 = store(&mut b, e, buf, 2, high);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    keeps_exactly(&program, &hazards, &[s2], "the upper half's store must follow the lower half's");
}

#[test]
fn prove_orders_a_store_in_a_later_block() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let s1 = store(&mut b, e, buf, 1, k.exec);
    let (next, n) = b.block(&[Ty::I1, Ty::I64]);
    b.br(e, next, vec![k.exec, buf]);
    let s2 = store(&mut b, next, n[1], 2, n[0]);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    keeps_exactly(&program, &hazards, &[s2], "the store in the next block must follow the first");
}

fn read_after(write: impl Fn(&mut Build, BlockId, ValueId) -> ValueId) -> (crate::rdna_spmd::program::Program, Position, Position) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let out = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let data = write(&mut b, e, lane);
    let s = b.here(e);
    b.store(e, Space::Global, MemSize::B32, buf, data, k.exec);
    let l = b.here(e);
    let v = b.load(e, Space::Global, MemSize::B32, buf, k.exec);
    let own = byte_offset(&mut b, e, out, lane, 4);
    b.store(e, Space::Global, MemSize::B32, own, v, k.exec);
    (b.program(), s, l)
}

#[test]
fn prove_orders_a_load_after_the_stores_whose_word_it_reads() {
    let (program, s, l) = read_after(|_, _, lane| lane);
    let hazards = Hazards::given(&program, &[(s, l)], &[], &[]);
    keeps_exactly(&program, &hazards, &[l], "every lane must read the one word the wave's store leaves, not its own");
}

#[test]
fn prove_orders_a_load_after_a_constant_store_only_some_lanes_make() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let out = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let sixteen = b.constant(e, Ty::I32, 16);
    let low = b.cmp(e, IntPred::Ult, lane, sixteen);
    let some = b.int(e, IntOp::And, low, k.exec);
    let seven = b.constant(e, Ty::I32, 7);
    let s = b.here(e);
    b.store(e, Space::Global, MemSize::B32, buf, seven, some);
    let l = b.here(e);
    let v = b.load(e, Space::Global, MemSize::B32, buf, k.exec);
    let own = byte_offset(&mut b, e, out, lane, 4);
    b.store(e, Space::Global, MemSize::B32, own, v, k.exec);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(s, l)], &[], &[]);
    keeps_exactly(&program, &hazards, &[l], "lanes 16 to 31 read the word without storing it, so they must wait for lanes 0 to 15");
}

#[test]
fn prove_needs_no_meeting_when_every_lane_stores_the_word_it_reads_back() {
    let (program, s, l) = read_after(|b, e, _| b.constant(e, Ty::I32, 7));
    let hazards = Hazards::given(&program, &[(s, l)], &[], &[]);
    keeps_exactly(&program, &hazards, &[], "every lane stores 7 before it reads, and no lane stores anything else there");
}

#[test]
fn prove_orders_a_store_after_the_loads_that_read_the_old_word() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let out = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let l = b.here(e);
    let v = b.load(e, Space::Global, MemSize::B32, buf, k.exec);
    let s = b.here(e);
    b.store(e, Space::Global, MemSize::B32, buf, lane, k.exec);
    let own = byte_offset(&mut b, e, out, lane, 4);
    b.store(e, Space::Global, MemSize::B32, own, v, k.exec);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(l, s)], &[], &[]);
    keeps_exactly(&program, &hazards, &[s], "no lane may overwrite the word before every lane has read it");
}

#[test]
fn prove_needs_no_meeting_for_a_load_whose_word_nothing_uses() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let s = b.here(e);
    b.store(e, Space::Global, MemSize::B32, buf, lane, k.exec);
    let l = b.here(e);
    b.load(e, Space::Global, MemSize::B32, buf, k.exec);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(s, l)], &[], &[]);
    keeps_exactly(&program, &hazards, &[], "the loaded word reaches no store");
}

fn with_between(between: impl Fn(&mut Build, BlockId, &Kernel)) -> (crate::rdna_spmd::program::Program, Position, Position) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let s1 = store(&mut b, e, buf, 1, k.exec);
    between(&mut b, e, &k);
    let s2 = store(&mut b, e, buf, 2, k.exec);
    (b.program(), s1, s2)
}

#[test]
fn prove_needs_no_meeting_across_a_barrier() {
    let (program, s1, s2) = with_between(|b, e, _| {
        let id = b.constant(e, Ty::I32, 0);
        b.effect(e, EffectOp::BarrierSignal { is_first: false }, vec![id]);
        b.effect(e, EffectOp::BarrierWait, vec![id]);
    });
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    keeps_exactly(&program, &hazards, &[], "the barrier already aligns every lane between the stores");
}

#[test]
fn prove_needs_no_meeting_across_a_lane_read() {
    let (program, s1, s2) = with_between(|b, e, k| {
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let zero = b.constant(e, Ty::I32, 0);
        let y = b.wave(e, WaveOp::ReadLane, vec![lane, zero, zero]);
        let out = k.buffer(b, e, 8);
        let own = byte_offset(b, e, out, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, y, k.exec);
    });
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    keeps_exactly(&program, &hazards, &[], "the lane read already aligns every lane between the stores");
}

#[test]
fn search_needs_no_meeting_a_query_kept_later_already_orders() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let s1 = store(&mut b, e, buf, 1, k.exec);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let flags = k.buffer(&mut b, e, 8);
    let own = byte_offset(&mut b, e, flags, lane, 4);
    let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let s2 = store(&mut b, e, buf, 2, k.exec);
    let one = b.constant(e, Ty::I32, 1);
    let data = b.core(e, Ty::I32, Op::Select(q, one, zero));
    let out = k.buffer(&mut b, e, 16);
    let slot = byte_offset(&mut b, e, out, lane, 4);
    b.store(e, Space::Global, MemSize::B32, slot, data, k.exec);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards);
        assert_eq!(kept.queries.len(), 1, "{}: a lane without the flag stores 0 unless the query stays", name);
    }
    keeps_exactly(&program, &hazards, &[], "the kept query between the stores already aligns every lane");
}

fn either_answer() -> (Build, ValueId, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let first = b.wave(e, WaveOp::Any, vec![c]);
    let second = b.wave(e, WaveOp::Any, vec![c]);
    let either = b.int(e, IntOp::Or, first, second);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let data = b.core(e, Ty::I32, Op::Select(either, one, two));
    store_own(&mut b, &k, e, data, k.exec);
    (b, first, second)
}

#[test]
fn prove_keeps_one_of_two_queries_either_of_which_answers_a_store() {
    let (b, first, second) = either_answer();
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&first) && !kept.queries.contains(&second))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag stores 2 unless one of the queries stays", wrong);
}

fn masked_answers(first_masked: bool) -> (Build, ValueId, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let zero = b.constant(e, Ty::I32, 0);
    let flag = per_lane(&mut b, &k, e, 8);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let word = uniform_load(&mut b, &k, e, 16);
    let on = b.cmp(e, IntPred::Ne, word, zero);
    let masked = b.int(e, IntOp::And, c, on);
    let (first, second) = if first_masked {
        let other = per_lane(&mut b, &k, e, 24);
        let other_set = b.cmp(e, IntPred::Ne, other, zero);
        let d = b.int(e, IntOp::And, other_set, k.exec);
        (b.wave(e, WaveOp::Any, vec![masked]), b.wave(e, WaveOp::Any, vec![d]))
    } else {
        (b.wave(e, WaveOp::Any, vec![c]), b.wave(e, WaveOp::Any, vec![masked]))
    };
    let either = b.int(e, IntOp::Or, first, second);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let data = b.core(e, Ty::I32, Op::Select(either, one, two));
    store_own(&mut b, &k, e, data, k.exec);
    (b, first, second)
}

#[test]
fn prove_keeps_only_one_of_two_queries_over_a_word_and_its_uniform_mask() {
    let (b, first, second) = masked_answers(false);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| kept.queries.contains(&first) && kept.queries.contains(&second))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: any(c & u) is u & any(c), so a kept any(c) answers both", wrong);
}

#[test]
fn prove_keeps_both_of_two_queries_over_a_masked_word_and_another() {
    let (b, first, second) = masked_answers(true);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&first) || !kept.queries.contains(&second))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: any(c & u) and any(d) answer for different lanes", wrong);
}

#[test]
fn prove_keeps_both_of_two_queries_over_different_words_either_of_which_answers_a_store() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let zero = b.constant(e, Ty::I32, 0);
    let flags: Vec<ValueId> = [8, 12]
        .iter()
        .map(|&at| {
            let flag = per_lane(&mut b, &k, e, at);
            let set = b.cmp(e, IntPred::Ne, flag, zero);
            let c = b.int(e, IntOp::And, set, k.exec);
            b.wave(e, WaveOp::Any, vec![c])
        })
        .collect();
    let either = b.int(e, IntOp::Or, flags[0], flags[1]);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let data = b.core(e, Ty::I32, Op::Select(either, one, two));
    store_own(&mut b, &k, e, data, k.exec);
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&flags[0]) || !kept.queries.contains(&flags[1]))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: any(c) | any(d) is 1 in a lane with neither bit only when some other lane has one", wrong);
}

#[test]
fn prove_keeps_only_one_of_two_queries_either_of_which_answers_a_store() {
    let (b, first, second) = either_answer();
    let wrong: Vec<&str> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| kept.queries.contains(&first) && kept.queries.contains(&second))
        .map(|(name, _)| *name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a kept query answers any(c), which already holds wherever the other query's own bit c does", wrong);
}

#[test]
fn prove_needs_no_meeting_across_a_kept_query() {
    let (program, s1, s2) = with_between(|b, e, k| {
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let flags = k.buffer(b, e, 8);
        let own = byte_offset(b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let one = b.constant(e, Ty::I32, 1);
        let data = b.core(e, Ty::I32, Op::Select(q, one, zero));
        let out = k.buffer(b, e, 16);
        let slot = byte_offset(b, e, out, lane, 4);
        b.store(e, Space::Global, MemSize::B32, slot, data, k.exec);
    });
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards);
        assert_eq!(kept.queries.len(), 1, "{}: a lane without the flag stores 0 unless the query stays", name);
    }
    keeps_exactly(&program, &hazards, &[], "the kept query already aligns every lane between the stores");
}

#[test]
fn prove_orders_two_stores_across_a_converted_query() {
    let (program, s1, s2) = with_between(|b, e, k| {
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let flags = k.buffer(b, e, 8);
        let own = byte_offset(b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let one = b.constant(e, Ty::I32, 1);
        let data = b.core(e, Ty::I32, Op::Select(q, one, zero));
        let out = k.buffer(b, e, 16);
        let slot = byte_offset(b, e, out, lane, 4);
        b.store(e, Space::Global, MemSize::B32, slot, data, c);
    });
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards);
        assert!(kept.queries.is_empty(), "{}: only lanes with the flag store the answer, and they see true either way", name);
    }
    keeps_exactly(&program, &hazards, &[s2], "the converted query aligns no lanes, so the second store needs its meeting");
}

fn flag_between(b: &mut Build, e: BlockId, k: &Kernel) -> (ValueId, ValueId, ValueId) {
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let flags = k.buffer(b, e, 8);
    let own = byte_offset(b, e, flags, lane, 4);
    let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let out = k.buffer(b, e, 16);
    let slot = byte_offset(b, e, out, lane, 4);
    (flag, c, slot)
}

#[test]
fn prove_needs_no_meeting_across_a_ballot_whose_whole_word_is_stored() {
    let (program, s1, s2) = with_between(|b, e, k| {
        let (_, c, slot) = flag_between(b, e, k);
        let w = b.wave(e, WaveOp::Ballot, vec![c]);
        b.store(e, Space::Global, MemSize::B32, slot, w, k.exec);
    });
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    keeps_exactly(&program, &hazards, &[], "the stored ballot word is computed by the whole wave, which aligns every lane between the stores");
}

#[test]
fn prove_orders_two_stores_across_a_ballot_whose_own_bit_alone_is_used() {
    let (program, s1, s2) = with_between(|b, e, k| {
        let (_, c, slot) = flag_between(b, e, k);
        let w = b.wave(e, WaveOp::Ballot, vec![c]);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let shifted = b.int(e, IntOp::LShr, w, lane);
        let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let one = b.constant(e, Ty::I32, 1);
        b.store(e, Space::Global, MemSize::B32, slot, one, bit);
    });
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    keeps_exactly(&program, &hazards, &[s2], "each lane computes its own bit of the ballot, which aligns no lanes");
}

#[test]
fn prove_needs_no_meeting_across_a_first_lane_read_of_a_varying_word() {
    let (program, s1, s2) = with_between(|b, e, k| {
        let (flag, _, slot) = flag_between(b, e, k);
        let x = b.wave(e, WaveOp::ReadFirstLane, vec![flag, k.exec]);
        b.store(e, Space::Global, MemSize::B32, slot, x, k.exec);
    });
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    keeps_exactly(&program, &hazards, &[], "reading the first lane's flag stays a wave operation, which aligns every lane between the stores");
}

#[test]
fn prove_orders_two_stores_across_a_first_lane_read_of_a_uniform_word() {
    let (program, s1, s2) = with_between(|b, e, k| {
        let (_, _, slot) = flag_between(b, e, k);
        let seven = b.constant(e, Ty::I32, 7);
        let x = b.wave(e, WaveOp::ReadFirstLane, vec![seven, k.exec]);
        b.store(e, Space::Global, MemSize::B32, slot, x, k.exec);
    });
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    keeps_exactly(&program, &hazards, &[s2], "every lane holds 7, so the read becomes the lane's own 7 and aligns no lanes");
}

#[test]
fn prove_needs_no_meeting_when_only_inactive_lanes_read_the_word() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let out = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let zero = b.constant(e, Ty::I32, 0);
    let first = b.cmp(e, IntPred::Eq, lane, zero);
    let exec = b.int(e, IntOp::And, first, k.exec);
    let own = byte_offset(&mut b, e, out, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    b.br(e, then, vec![exec, buf, own]);
    let s = store(&mut b, then, t[1], 7, t[0]);
    let yes = b.constant(then, Ty::I1, 1);
    let l = b.here(then);
    let v = b.load(then, Space::Global, MemSize::B32, t[1], yes);
    b.store(then, Space::Global, MemSize::B32, t[2], v, t[0]);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(s, l)], &[], &[(l, s, false)]);
    keeps_exactly(&program, &hazards, &[], "only lane 0 keeps what it loads, and only lanes with exec clear read lane 0's word");
}

fn sliding_loop(lane_step: bool) -> (crate::rdna_spmd::program::Program, Position) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let buf = k.buffer(&mut b, e, 0);
    let zero = b.constant(e, Ty::I32, 0);
    let (body, p) = b.block(&[Ty::I1, Ty::I32, Ty::I64]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.br(e, body, vec![k.exec, zero, buf]);
    let index = if lane_step {
        let lane = b.core(body, Ty::I32, Op::Env(Env::LaneId));
        b.int(body, IntOp::Add, p[1], lane)
    } else {
        p[1]
    };
    let address = byte_offset(&mut b, body, p[2], index, 4);
    let s = b.here(body);
    b.store(body, Space::Global, MemSize::B32, address, p[1], p[0]);
    let one = b.constant(body, Ty::I32, 1);
    let next = b.int(body, IntOp::Add, p[1], one);
    let four = b.constant(body, Ty::I32, 4);
    let again = b.cmp(body, IntPred::Ult, next, four);
    b.cond_br(body, again, (body, vec![p[0], next, p[2]]), (exit, vec![p[0]]));
    (b.program(), s)
}

#[test]
fn prove_orders_iterations_that_store_to_each_other_words() {
    let (program, s) = sliding_loop(true);
    let hazards = Hazards::given(&program, &[], &[(s, s)], &[]);
    keeps_exactly(&program, &hazards, &[s], "lane a's store in iteration i + 1 must follow lane a + 1's in iteration i");
}

#[test]
fn prove_needs_no_meeting_for_one_instruction_within_one_iteration() {
    let (program, s) = sliding_loop(false);
    let hazards = Hazards::given(&program, &[(s, s)], &[], &[]);
    keeps_exactly(&program, &hazards, &[], "lanes of one instruction have no order in the wave either");
}

fn converted(b: &Build) -> Vec<&'static str> {
    ["search", "direct"]
        .iter()
        .zip(both(b))
        .filter(|(_, kept)| !kept.queries.is_empty() || !kept.words.is_empty())
        .map(|(name, _)| *name)
        .collect()
}

fn uniform_load(b: &mut Build, k: &Kernel, e: BlockId, offset: u64) -> ValueId {
    let table = k.buffer(b, e, offset);
    let yes = b.constant(e, Ty::I1, 1);
    b.load(e, Space::Global, MemSize::B32, table, yes)
}

fn per_lane(b: &mut Build, k: &Kernel, e: BlockId, offset: u64) -> ValueId {
    let table = k.buffer(b, e, offset);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(b, e, table, lane, 4);
    b.load(e, Space::Global, MemSize::B32, own, k.exec)
}

fn query_data(b: &mut Build, k: &Kernel, e: BlockId) -> ValueId {
    let flag = per_lane(b, k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    b.core(e, Ty::I32, Op::Select(q, one, two))
}

fn store_own(b: &mut Build, k: &Kernel, e: BlockId, data: ValueId, mask: ValueId) {
    let buf = k.buffer(b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(b, e, buf, lane, 4);
    b.store(e, Space::Global, MemSize::B32, own, data, mask);
}

fn two_orders(first: (IntPred, bool), second: (IntPred, bool)) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let v = per_lane(&mut b, &k, e, 16);
    let w = per_lane(&mut b, &k, e, 24);
    let order = |b: &mut Build, (p, swap): (IntPred, bool)| if swap { b.cmp(e, p, w, v) } else { b.cmp(e, p, v, w) };
    let one = order(&mut b, first);
    let other = order(&mut b, second);
    let both = b.int(e, IntOp::And, one, other);
    let mask = b.int(e, IntOp::And, both, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_two_orders_that_may_both_hold_mask() {
    let b = two_orders((IntPred::Ult, false), (IntPred::Ule, false));
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: v < w and v <= w both hold when v < w, so lanes store the answer", wrong);
}

#[test]
fn prove_keeps_a_query_whose_store_an_order_and_its_swap_mask() {
    let b = two_orders((IntPred::Slt, false), (IntPred::Sgt, true));
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: w s> v is v s< w, so lanes with v s< w store the answer", wrong);
}

#[test]
fn prove_keeps_a_query_whose_store_two_opposite_orders_that_hold_at_equality_mask() {
    let b = two_orders((IntPred::Ule, false), (IntPred::Ule, true));
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: v <= w and w <= v both hold when v == w", wrong);
}

#[test]
fn prove_keeps_a_query_whose_store_opposite_orders_of_different_signedness_mask() {
    let b = two_orders((IntPred::Slt, false), (IntPred::Ult, true));
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: v = -1 and w = 0 give v s< w and w u< v", wrong);
}

#[test]
fn prove_keeps_a_query_whose_store_an_equality_and_an_order_that_hold_together_mask() {
    let b = two_orders((IntPred::Eq, false), (IntPred::Ule, false));
    assert!(keeps(&b).is_empty(), "{:?}: v == w gives v <= w", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_an_equality_and_a_strict_order_mask() {
    let b = two_orders((IntPred::Eq, false), (IntPred::Ult, false));
    assert!(converted(&b).is_empty(), "{:?}: v == w and v < w never hold together", converted(&b));
}

fn orders_in_two_blocks(opposite: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let v = per_lane(&mut b, &k, e, 16);
    let w = per_lane(&mut b, &k, e, 24);
    let first = b.cmp(e, IntPred::Ult, v, w);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32, Ty::I1]);
    b.br(e, next, vec![k.exec, own, data, v, w, first]);
    let second = b.cmp(next, if opposite { IntPred::Uge } else { IntPred::Ule }, p[3], p[4]);
    let both = b.int(next, IntOp::And, p[5], second);
    let mask = b.int(next, IntOp::And, both, p[0]);
    b.store(next, Space::Global, MemSize::B32, p[1], p[2], mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_orders_in_two_blocks_that_may_both_hold_mask() {
    let b = orders_in_two_blocks(false);
    assert!(keeps(&b).is_empty(), "{:?}: v < w gives v <= w in the next block", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_opposite_orders_in_two_blocks_mask() {
    let b = orders_in_two_blocks(true);
    assert!(converted(&b).is_empty(), "{:?}: v < w in the first block rules out v >= w in the next", converted(&b));
}

fn float_bounds(above: u64) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let word = per_lane(&mut b, &k, e, 16);
    let x = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, word));
    let five = b.constant(e, Ty::F32, 0x40a0_0000);
    let bound = b.constant(e, Ty::F32, above);
    let small = b.core(e, Ty::I1, Op::FCmp(FloatPred::Olt, x, five));
    let large = b.core(e, Ty::I1, Op::FCmp(FloatPred::Ogt, x, bound));
    let both = b.int(e, IntOp::And, small, large);
    let mask = b.int(e, IntOp::And, both, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_two_float_bounds_that_may_both_hold_mask() {
    let b = float_bounds(0x3f80_0000);
    assert!(keeps(&b).is_empty(), "{:?}: x = 2.0 is below 5.0 and above 1.0", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_disjoint_float_bounds_mask() {
    let b = float_bounds(0x4120_0000);
    assert!(converted(&b).is_empty(), "{:?}: no float is below 5.0 and above 10.0", converted(&b));
}

fn two_bounds(below: u64, above: u64) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let v = per_lane(&mut b, &k, e, 16);
    let below = b.constant(e, Ty::I32, below);
    let above = b.constant(e, Ty::I32, above);
    let small = b.cmp(e, IntPred::Ult, v, below);
    let large = b.cmp(e, IntPred::Ugt, v, above);
    let both = b.int(e, IntOp::And, small, large);
    let mask = b.int(e, IntOp::And, both, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

fn bounded(first: (IntPred, u64), second: (IntPred, u64)) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let v = per_lane(&mut b, &k, e, 16);
    let mut bits = Vec::new();
    for (p, bound) in [first, second] {
        let bound = b.constant(e, Ty::I32, bound);
        bits.push(b.cmp(e, p, v, bound));
    }
    let both = b.int(e, IntOp::And, bits[0], bits[1]);
    let mask = b.int(e, IntOp::And, both, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_follows_pairs_of_bounds_on_one_word() {
    let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
    let bounds = [0u32, 1, 4, 5, 6, 10, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
    let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
    let mut r = Random::new(29);
    let mut wrong = Vec::new();
    for _ in 0..400 {
        let (p1, k1) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
        let (p2, k2) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
        let mut candidates = vec![0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
        for k in [k1, k2] {
            candidates.extend([k.wrapping_sub(1), k, k.wrapping_add(1)]);
        }
        let possible = candidates.iter().any(|&v| holds(p1, v, k1) && holds(p2, v, k2));
        let b = bounded((p1, k1 as u64), (p2, k2 as u64));
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            let converted = kept.queries.is_empty();
            if possible && converted {
                wrong.push(format!("{}: v {:?} {} and v {:?} {} can both hold, yet the query was converted", name, p1, k1, p2, k2));
            }
            if !possible && !converted {
                wrong.push(format!("{}: v {:?} {} and v {:?} {} never both hold, yet the query stays", name, p1, k1, p2, k2));
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

#[test]
fn prove_follows_a_bound_into_a_block_and_a_bound_inside_it() {
    let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
    let bounds = [0u32, 1, 4, 5, 6, 10, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
    let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
    let mut r = Random::new(41);
    let mut wrong = Vec::new();
    for _ in 0..200 {
        let (p1, k1) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
        let (p2, k2) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
        let mut candidates = vec![0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
        for k in [k1, k2] {
            candidates.extend([k.wrapping_sub(1), k, k.wrapping_add(1)]);
        }
        let possible = candidates.iter().any(|&v| holds(p1, v, k1) && holds(p2, v, k2));
        let b = branch_then_store((p1, k1 as u64), (p2, k2 as u64));
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            let converted = kept.queries.is_empty();
            if possible && converted {
                wrong.push(format!("{}: x {:?} {} into the block and x {:?} {} inside it can both hold, yet the query was converted", name, p1, k1, p2, k2));
            }
            if !possible && !converted {
                wrong.push(format!("{}: x {:?} {} into the block and x {:?} {} inside it never both hold, yet the query stays", name, p1, k1, p2, k2));
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

fn ordered_words(tests: &[(IntPred, usize, usize)]) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let words: Vec<ValueId> = [16, 24, 32].iter().map(|&o| per_lane(&mut b, &k, e, o)).collect();
    let mut mask = k.exec;
    for &(p, i, j) in tests {
        let c = b.cmp(e, p, words[i], words[j]);
        mask = b.int(e, IntOp::And, mask, c);
    }
    store_own(&mut b, &k, e, data, mask);
    b
}

fn ordered_and_bounded_words(tests: &[(IntPred, usize, Option<usize>, u32)]) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let words: Vec<ValueId> = [16, 24].iter().map(|&o| per_lane(&mut b, &k, e, o)).collect();
    let mut mask = k.exec;
    for &(p, i, j, bound) in tests {
        let other = match j {
            Some(j) => words[j],
            None => b.constant(e, Ty::I32, bound as u64),
        };
        let c = b.cmp(e, p, words[i], other);
        mask = b.int(e, IntOp::And, mask, c);
    }
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_follows_orders_of_both_signs_among_words_bounded_by_constants() {
    let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
    let bounds = [0u32, 1, 5, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_ffff];
    let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
    let points = [0u32, 1, 2, 4, 5, 6, 0x7fff_fffe, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0x8000_0002, 0xffff_fffe, 0xffff_ffff];
    let mut r = Random::new(71);
    let mut wrong = Vec::new();
    for _ in 0..300 {
        let count = 2 + r.below(3) as usize;
        let tests: Vec<(IntPred, usize, Option<usize>, u32)> = (0..count)
            .map(|_| {
                let i = r.below(2) as usize;
                let p = preds[r.below(10) as usize];
                if r.below(2) == 0 {
                    (p, i, Some(1 - i), 0)
                } else {
                    (p, i, None, bounds[r.below(bounds.len() as u64) as usize])
                }
            })
            .collect();
        let possible = points.iter().any(|&x| {
            points.iter().any(|&y| {
                tests.iter().all(|&(p, i, j, bound)| {
                    let w = [x, y];
                    holds(p, w[i], j.map_or(bound, |j| w[j]))
                })
            })
        });
        let b = ordered_and_bounded_words(&tests);
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            if possible && kept.queries.is_empty() {
                wrong.push(format!("{}: {:?} can all hold, yet the query was converted", name, tests));
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

fn branch_then_store_offset(entry: (IntPred, u32), offset: u32, store: (IntPred, u32)) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let table = k.buffer(&mut b, e, 16);
    let yes = b.constant(e, Ty::I1, 1);
    let x = b.load(e, Space::Global, MemSize::B32, table, yes);
    let bound = b.constant(e, Ty::I32, entry.1 as u64);
    let enters = b.cmp(e, entry.0, x, bound);
    let shift = b.constant(e, Ty::I32, offset as u64);
    let moved = b.int(e, IntOp::Add, x, shift);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.cond_br(e, enters, (then, vec![k.exec, moved, data, own]), (exit, vec![k.exec]));
    let limit = b.constant(then, Ty::I32, store.1 as u64);
    let test = b.cmp(then, store.0, t[1], limit);
    let mask = b.int(then, IntOp::And, test, t[0]);
    b.store(then, Space::Global, MemSize::B32, t[3], t[2], mask);
    b
}

#[test]
fn prove_follows_a_bound_carried_through_an_offset_into_the_next_block() {
    let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
    let bounds = [0u32, 1, 4, 5, 6, 10, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
    let offsets = [0u32, 1, 5, 0x8000_0000, 0xffff_ffff];
    let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
    let mut r = Random::new(73);
    let mut wrong = Vec::new();
    for _ in 0..200 {
        let (p1, k1) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
        let (p2, k2) = (preds[r.below(10) as usize], bounds[r.below(9) as usize]);
        let c = offsets[r.below(offsets.len() as u64) as usize];
        let mut candidates = vec![0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0xffff_ffff];
        for k in [k1, k2.wrapping_sub(c)] {
            candidates.extend([k.wrapping_sub(1), k, k.wrapping_add(1)]);
        }
        for x in candidates.clone() {
            candidates.push(x.wrapping_sub(c));
        }
        let possible = candidates.iter().any(|&x| holds(p1, x, k1) && holds(p2, x.wrapping_add(c), k2));
        let b = branch_then_store_offset((p1, k1), c, (p2, k2));
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            let converted = kept.queries.is_empty();
            if possible && converted {
                wrong.push(format!("{}: x {:?} {} into the block and x + {} {:?} {} inside it can both hold, yet the query was converted", name, p1, k1, c, p2, k2));
            }
            if !possible && !converted {
                wrong.push(format!("{}: x {:?} {} into the block and x + {} {:?} {} inside it never both hold, yet the query stays", name, p1, k1, c, p2, k2));
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

fn orders_then_compare(first: &[(IntPred, bool)], second: (IntPred, bool)) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let v = per_lane(&mut b, &k, e, 16);
    let w = per_lane(&mut b, &k, e, 24);
    let mut known = b.constant(e, Ty::I1, 1);
    for &(p, swap) in first {
        let c = if swap { b.cmp(e, p, w, v) } else { b.cmp(e, p, v, w) };
        known = b.int(e, IntOp::And, known, c);
    }
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32, Ty::I1]);
    b.br(e, next, vec![k.exec, own, data, v, w, known]);
    let later = if second.1 { b.cmp(next, second.0, p[4], p[3]) } else { b.cmp(next, second.0, p[3], p[4]) };
    let both_hold = b.int(next, IntOp::And, p[5], later);
    let mask = b.int(next, IntOp::And, both_hold, p[0]);
    b.store(next, Space::Global, MemSize::B32, p[1], p[2], mask);
    b
}

#[test]
fn prove_follows_orders_of_two_words_into_the_next_block() {
    let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
    let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
    let points = [0u32, 1, 2, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_fffe, 0xffff_ffff];
    let family = |p: IntPred| match p {
        IntPred::Eq | IntPred::Ne => 0,
        IntPred::Ult | IntPred::Ule | IntPred::Ugt | IntPred::Uge => 1,
        _ => 2,
    };
    let mut r = Random::new(79);
    let mut wrong = Vec::new();
    for _ in 0..300 {
        let first: Vec<(IntPred, bool)> = (0..1 + r.below(2)).map(|_| (preds[r.below(10) as usize], r.below(2) == 0)).collect();
        let second = (preds[r.below(10) as usize], r.below(2) == 0);
        let test = |(p, swap): (IntPred, bool), v: u32, w: u32| if swap { holds(p, w, v) } else { holds(p, v, w) };
        let possible = points.iter().any(|&v| points.iter().any(|&w| first.iter().all(|&t| test(t, v, w)) && test(second, v, w)));
        let families: BTreeSet<i32> = first.iter().chain([&second]).map(|&(p, _)| family(p)).filter(|&f| f != 0).collect();
        let b = orders_then_compare(&first, second);
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            let converted = kept.queries.is_empty();
            if possible && converted {
                wrong.push(format!("{}: {:?} then {:?} can both hold, yet the query was converted", name, first, second));
            }
            if !possible && !converted && families.len() <= 1 {
                wrong.push(format!("{}: {:?} then {:?} never both hold, yet the query stays", name, first, second));
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

#[test]
fn prove_follows_orders_among_three_words() {
    let preds = [IntPred::Ult, IntPred::Ule, IntPred::Ugt, IntPred::Uge, IntPred::Slt, IntPred::Sle, IntPred::Sgt, IntPred::Sge, IntPred::Eq, IntPred::Ne];
    let holds = |p: IntPred, v: u32, k: u32| super::super::address::compare(p, v, k);
    let points = [0u32, 1, 2, 0x8000_0000, 0x8000_0001, 0x8000_0002, 0xffff_fffe, 0xffff_ffff];
    let family = |p: IntPred| match p {
        IntPred::Eq | IntPred::Ne => 0,
        IntPred::Ult | IntPred::Ule | IntPred::Ugt | IntPred::Uge => 1,
        _ => 2,
    };
    let mut r = Random::new(43);
    let mut wrong = Vec::new();
    for _ in 0..300 {
        let count = 2 + r.below(2) as usize;
        let tests: Vec<(IntPred, usize, usize)> = (0..count)
            .map(|_| {
                let i = r.below(3) as usize;
                let j = (i + 1 + r.below(2) as usize) % 3;
                (preds[r.below(10) as usize], i, j)
            })
            .collect();
        let possible = points.iter().any(|&x| {
            points.iter().any(|&y| points.iter().any(|&z| tests.iter().all(|&(p, i, j)| holds(p, [x, y, z][i], [x, y, z][j]))))
        });
        let families: BTreeSet<i32> = tests.iter().map(|&(p, _, _)| family(p)).filter(|&f| f != 0).collect();
        let b = ordered_words(&tests);
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            let converted = kept.queries.is_empty();
            if possible && converted {
                wrong.push(format!("{}: {:?} can all hold, yet the query was converted", name, tests));
            }
            if !possible && !converted && families.len() <= 1 {
                wrong.push(format!("{}: {:?} never all hold, yet the query stays", name, tests));
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

fn float_pair(first: (FloatPred, f32), second: (FloatPred, f32)) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let word = per_lane(&mut b, &k, e, 16);
    let x = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, word));
    let mut mask = k.exec;
    for (p, c) in [first, second] {
        let bound = b.constant(e, Ty::F32, c.to_bits() as u64);
        let t = b.core(e, Ty::I1, Op::FCmp(p, x, bound));
        mask = b.int(e, IntOp::And, mask, t);
    }
    store_own(&mut b, &k, e, data, mask);
    b
}

fn float_orders_of_two_words(tests: &[(FloatPred, bool)]) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let v = per_lane(&mut b, &k, e, 16);
    let w = per_lane(&mut b, &k, e, 24);
    let x = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, v));
    let y = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, w));
    let mut mask = k.exec;
    for &(p, swap) in tests {
        let t = if swap { b.core(e, Ty::I1, Op::FCmp(p, y, x)) } else { b.core(e, Ty::I1, Op::FCmp(p, x, y)) };
        mask = b.int(e, IntOp::And, mask, t);
    }
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_follows_orders_of_two_float_words() {
    use FloatPred::*;
    let preds = [Oeq, Ogt, Oge, Olt, Ole, One, Ord, Uno, Ueq, Ugt, Uge, Ult, Ule, Une];
    let points = [0.0f32, -0.0, 1.0, -1.0, 2.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let mut r = Random::new(83);
    let mut wrong = Vec::new();
    for _ in 0..300 {
        let tests: Vec<(FloatPred, bool)> = (0..2 + r.below(2)).map(|_| (preds[r.below(14) as usize], r.below(2) == 0)).collect();
        let holds = |(p, swap): (FloatPred, bool), x: f32, y: f32| {
            let (a, b) = if swap { (y, x) } else { (x, y) };
            super::super::logic::float_compare(p, a as f64, b as f64)
        };
        let possible = points.iter().any(|&x| points.iter().any(|&y| tests.iter().all(|&t| holds(t, x, y))));
        let ordered = tests.iter().all(|&(p, _)| matches!(p, Oeq | Ogt | Oge | Olt | Ole));
        let b = float_orders_of_two_words(&tests);
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            let converted = kept.queries.is_empty();
            if possible && converted {
                wrong.push(format!("{}: {:?} can all hold, yet the query was converted", name, tests));
            }
            if !possible && !converted && ordered {
                wrong.push(format!("{}: {:?} never all hold, yet the query stays", name, tests));
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

#[test]
fn prove_follows_pairs_of_float_bounds_on_one_word() {
    use FloatPred::*;
    let preds = [Oeq, Ogt, Oge, Olt, Ole, One, Ord, Uno, Ueq, Ugt, Uge, Ult, Ule, Une];
    let bounds = [0.0f32, -0.0, 1.0, -1.0, 5.0, 10.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let mut r = Random::new(47);
    let mut wrong = Vec::new();
    for _ in 0..300 {
        let (p1, k1) = (preds[r.below(14) as usize], bounds[r.below(9) as usize]);
        let (p2, k2) = (preds[r.below(14) as usize], bounds[r.below(9) as usize]);
        let mut candidates = vec![0.0f32, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX, f32::MIN];
        for k in [k1, k2] {
            if !k.is_nan() {
                candidates.extend([f32::from_bits(k.to_bits().wrapping_sub(1)), k, f32::from_bits(k.to_bits().wrapping_add(1))]);
                candidates.extend([k - 0.5, k + 0.5]);
            }
        }
        let possible = candidates.iter().any(|&v| float_compare(p1, v as f64, k1 as f64) && float_compare(p2, v as f64, k2 as f64));
        let b = float_pair((p1, k1), (p2, k2));
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            let converted = kept.queries.is_empty();
            if possible && converted {
                wrong.push(format!("{}: x {:?} {} and x {:?} {} can both hold, yet the query was converted", name, p1, k1, p2, k2));
            }
            if !possible && !converted {
                wrong.push(format!("{}: x {:?} {} and x {:?} {} never both hold, yet the query stays", name, p1, k1, p2, k2));
            }
        }
    }
    assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(8)]);
}

#[test]
fn prove_converts_a_query_whose_store_two_different_equalities_mask() {
    let b = bounded((IntPred::Eq, 5), (IntPred::Eq, 7));
    assert!(converted(&b).is_empty(), "{:?}: v == 5 and v == 7 never hold together", converted(&b));
}

#[test]
fn prove_converts_a_query_whose_store_an_equality_and_its_negation_mask() {
    let b = bounded((IntPred::Eq, 5), (IntPred::Ne, 5));
    assert!(converted(&b).is_empty(), "{:?}: v == 5 and v != 5 never hold together", converted(&b));
}

#[test]
fn prove_converts_a_query_whose_store_an_equality_outside_a_bound_masks() {
    let b = bounded((IntPred::Eq, 5), (IntPred::Ult, 3));
    assert!(converted(&b).is_empty(), "{:?}: v == 5 and v < 3 never hold together", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_store_an_equality_inside_a_bound_masks() {
    for second in [(IntPred::Ult, 6), (IntPred::Eq, 5), (IntPred::Ne, 7)] {
        let b = bounded((IntPred::Eq, 5), second);
        let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
        assert!(wrong.is_empty(), "{:?}: v == 5 and v {:?} {} both hold for v = 5", wrong, second.0, second.1);
    }
}

#[test]
fn prove_converts_a_query_whose_store_disjoint_ranges_mask() {
    let b = two_bounds(5, 10);
    assert!(converted(&b).is_empty(), "{:?}: v < 5 and v > 10 never hold together, so the store never runs", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_store_overlapping_ranges_mask() {
    let b = two_bounds(10, 5);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: v < 10 and v > 5 both hold for v = 7", wrong);
}

#[test]
fn prove_converts_a_query_whose_store_opposite_orders_mask() {
    let b = two_orders((IntPred::Ult, false), (IntPred::Ult, true));
    assert!(converted(&b).is_empty(), "{:?}: v < w and w < v never hold together, so the store never runs", converted(&b));
}

#[test]
fn prove_converts_a_query_whose_store_a_contradiction_masks() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let v = per_lane(&mut b, &k, e, 16);
    let w = per_lane(&mut b, &k, e, 24);
    let below = b.cmp(e, IntPred::Ult, v, w);
    let above = b.cmp(e, IntPred::Uge, v, w);
    let never = b.int(e, IntOp::And, below, above);
    let mask = b.int(e, IntOp::And, never, k.exec);
    store_own(&mut b, &k, e, data, mask);
    assert!(converted(&b).is_empty(), "{:?}: v < w and v >= w never hold together, so the store never runs", converted(&b));
}

#[test]
fn prove_converts_a_query_whose_store_disjoint_constants_mask() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let none = b.int(e, IntOp::And, one, two);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let shifted = b.int(e, IntOp::LShr, none, lane);
    let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
    let mask = b.int(e, IntOp::And, bit, k.exec);
    store_own(&mut b, &k, e, data, mask);
    assert!(converted(&b).is_empty(), "{:?}: 1 & 2 is 0, so the store never runs", converted(&b));
}

#[test]
fn prove_converts_a_query_whose_store_the_branch_into_its_block_rules_out() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let table = k.buffer(&mut b, e, 16);
    let yes = b.constant(e, Ty::I1, 1);
    let x = b.load(e, Space::Global, MemSize::B32, table, yes);
    let five = b.constant(e, Ty::I32, 5);
    let is_five = b.cmp(e, IntPred::Eq, x, five);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.cond_br(e, is_five, (then, vec![k.exec, x, data, own]), (exit, vec![k.exec]));
    let five = b.constant(then, Ty::I32, 5);
    let still = b.cmp(then, IntPred::Ne, t[1], five);
    let mask = b.int(then, IntOp::And, still, t[0]);
    b.store(then, Space::Global, MemSize::B32, t[3], t[2], mask);
    assert!(converted(&b).is_empty(), "{:?}: the block runs only when x is 5, where the store's mask x != 5 is false", converted(&b));
}

fn branch_then_store(entry: (IntPred, u64), store: (IntPred, u64)) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let table = k.buffer(&mut b, e, 16);
    let yes = b.constant(e, Ty::I1, 1);
    let x = b.load(e, Space::Global, MemSize::B32, table, yes);
    let bound = b.constant(e, Ty::I32, entry.1);
    let enters = b.cmp(e, entry.0, x, bound);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.cond_br(e, enters, (then, vec![k.exec, x, data, own]), (exit, vec![k.exec]));
    let limit = b.constant(then, Ty::I32, store.1);
    let test = b.cmp(then, store.0, t[1], limit);
    let mask = b.int(then, IntOp::And, test, t[0]);
    b.store(then, Space::Global, MemSize::B32, t[3], t[2], mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_the_branch_into_its_block_lets_through() {
    let b = branch_then_store((IntPred::Eq, 5), (IntPred::Eq, 5));
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: the block runs only when x is 5, where the store's mask x == 5 holds", wrong);
}

#[test]
fn prove_keeps_a_query_whose_store_a_branch_on_another_constant_lets_through() {
    let b = branch_then_store((IntPred::Eq, 6), (IntPred::Ne, 5));
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: x is 6 in the block, where x != 5 holds", wrong);
}

#[test]
fn prove_converts_a_query_whose_store_the_bound_into_its_block_rules_out() {
    let b = branch_then_store((IntPred::Ult, 10), (IntPred::Uge, 10));
    assert!(converted(&b).is_empty(), "{:?}: x < 10 in the block, where x >= 10 is false", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_store_the_bound_into_its_block_lets_through() {
    let b = branch_then_store((IntPred::Ult, 10), (IntPred::Ult, 10));
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: x < 10 in the block, where the store's mask x < 10 holds", wrong);
}

#[test]
fn prove_keeps_a_query_whose_store_a_bound_of_the_other_signedness_lets_through() {
    let b = branch_then_store((IntPred::Slt, 10), (IntPred::Uge, 10));
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: x = -1 is below 10 signed and at least 10 unsigned", wrong);
}

fn chosen_constants(low: u64) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let flag = per_lane(&mut b, &k, e, 16);
    let zero = b.constant(e, Ty::I32, 0);
    let pick = b.cmp(e, IntPred::Ne, flag, zero);
    let (two, four) = (b.constant(e, Ty::I32, 2), b.constant(e, Ty::I32, 4));
    let chosen = b.core(e, Ty::I32, Op::Select(pick, two, four));
    let low = b.constant(e, Ty::I32, low);
    let both = b.int(e, IntOp::And, low, chosen);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let shifted = b.int(e, IntOp::LShr, both, lane);
    let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
    let mask = b.int(e, IntOp::And, bit, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

fn constants_over_a_ballot(low: u64) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let flag = per_lane(&mut b, &k, e, 16);
    let zero = b.constant(e, Ty::I32, 0);
    let pick = b.cmp(e, IntPred::Ne, flag, zero);
    let pick = b.int(e, IntOp::And, pick, k.exec);
    let word = b.wave(e, WaveOp::Ballot, vec![pick]);
    let low = b.constant(e, Ty::I32, low);
    let two = b.constant(e, Ty::I32, 2);
    let inner = b.int(e, IntOp::And, word, low);
    let both = b.int(e, IntOp::And, inner, two);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let shifted = b.int(e, IntOp::LShr, both, lane);
    let bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
    let mask = b.int(e, IntOp::And, bit, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_converts_a_query_whose_store_two_constants_over_a_ballot_mask_off() {
    let b = constants_over_a_ballot(1);
    assert!(converted(&b).is_empty(), "{:?}: ballot & 1 & 2 is 0, so the store never runs", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_store_two_constants_over_a_ballot_may_mask_on() {
    let b = constants_over_a_ballot(3);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: ballot & 3 & 2 keeps bit 1, so lane 1 stores whenever its flag is set", wrong);
}

#[test]
fn prove_converts_a_query_whose_store_a_constant_and_a_chosen_constant_mask_off() {
    let b = chosen_constants(1);
    assert!(converted(&b).is_empty(), "{:?}: 1 & 2 and 1 & 4 are 0, so the store never runs", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_store_a_constant_and_a_chosen_constant_may_mask_on() {
    let b = chosen_constants(3);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: 3 & 2 sets bit 1, so lane 1 stores whenever it picks 2", wrong);
}

#[test]
fn prove_converts_a_query_whose_value_an_and_with_zero_drops() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let zero = b.constant(e, Ty::I32, 0);
    let nothing = b.int(e, IntOp::And, data, zero);
    store_own(&mut b, &k, e, nothing, k.exec);
    assert!(converted(&b).is_empty(), "{:?}: every lane stores 0 whatever the query answers", converted(&b));
}

fn written_first_lane(only: u64) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let target = b.constant(e, Ty::I32, only);
    let mine = b.cmp(e, IntPred::Eq, lane, target);
    let both = b.int(e, IntOp::And, set, mine);
    let c = b.int(e, IntOp::And, both, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let x = b.core(e, Ty::I32, Op::Select(q, one, two));
    let five = b.constant(e, Ty::I32, 5);
    let y = b.wave(e, WaveOp::WriteLane, vec![x, five, lane]);
    store_own(&mut b, &k, e, y, k.exec);
    b
}

#[test]
fn prove_converts_a_query_whose_word_a_lane_write_takes_from_the_one_lane_that_answers() {
    let b = written_first_lane(0);
    assert!(converted(&b).is_empty(), "{:?}: the write takes lane 0's word, and lane 0's own answer is the wave's", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_word_a_lane_write_takes_from_a_lane_that_cannot_answer() {
    let b = written_first_lane(1);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: lane 0 alone answers false while the wave answers lane 1's flag", wrong);
}

fn written_elsewhere(skip: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let x = b.core(e, Ty::I32, Op::Select(q, one, two));
    let five = b.constant(e, Ty::I32, 5);
    let y = b.wave(e, WaveOp::WriteLane, vec![x, five, lane]);
    let mask = if skip {
        let other = b.cmp(e, IntPred::Ne, lane, five);
        b.int(e, IntOp::And, other, k.exec)
    } else {
        k.exec
    };
    store_own(&mut b, &k, e, y, mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_word_a_lane_write_puts_into_a_lane_that_stores() {
    let b = written_elsewhere(false);
    assert!(keeps(&b).is_empty(), "{:?}: lane 5 stores lane 0's word, whose answer is lane 0's own flag", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_word_a_lane_write_puts_only_into_a_lane_that_never_stores() {
    let b = written_elsewhere(true);
    assert!(converted(&b).is_empty(), "{:?}: every storing lane keeps its own lane id", converted(&b));
}

fn offset_lanes(uniform: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let u = uniform_load(&mut b, &k, e, 16);
    let shifted = b.int(e, IntOp::Add, lane, u);
    let five = b.constant(e, Ty::I32, 5);
    let other = if uniform { b.int(e, IntOp::Add, lane, five) } else { five };
    let t = b.cmp(e, IntPred::Eq, shifted, other);
    let c = b.int(e, IntOp::And, t, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let data = b.core(e, Ty::I32, Op::Select(q, one, two));
    store_own(&mut b, &k, e, data, k.exec);
    b
}

#[test]
fn prove_keeps_a_query_whose_bit_one_lane_offset_by_a_word_sets() {
    let b = offset_lanes(false);
    assert!(keeps(&b).is_empty(), "{:?}: lane + u == 5 holds in lane 5 - u alone", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_bit_every_lane_offset_by_a_word_shares() {
    let b = offset_lanes(true);
    assert!(converted(&b).is_empty(), "{:?}: lane + u == lane + 5 is u == 5 in every lane", converted(&b));
}

fn permuted(op: WaveOp, index: impl Fn(&mut Build, BlockId, ValueId) -> ValueId) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let zero = b.constant(e, Ty::I32, 0);
    let fifteen = b.constant(e, Ty::I32, 15);
    let low = b.int(e, IntOp::And, lane, fifteen);
    let first = b.cmp(e, IntPred::Eq, low, zero);
    let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
    let picked = b.core(e, Ty::I32, Op::Select(q, ones, lane));
    let x = b.core(e, Ty::I32, Op::Select(first, lane, picked));
    let index = index(&mut b, e, lane);
    let y = match op {
        WaveOp::ReadLane => b.wave(e, WaveOp::ReadLane, vec![x, index, zero]),
        _ => b.wave(e, op, vec![index, x, k.exec]),
    };
    store_own(&mut b, &k, e, y, k.exec);
    b
}

#[test]
fn prove_converts_a_query_whose_word_only_lanes_a_permute_skips_hold() {
    let mut wrong = Vec::new();
    for op in [WaveOp::Bpermute, WaveOp::BpermuteFi] {
        let b = permuted(op, |b, e, _| b.constant(e, Ty::I32, 0));
        if !converted(&b).is_empty() {
            wrong.push(format!("{:?} of lane 0: {:?}", op, converted(&b)));
        }
        let b = permuted(op, |b, e, lane| {
            let sixteen = b.constant(e, Ty::I32, 16);
            let pick = b.int(e, IntOp::And, lane, sixteen);
            let two = b.constant(e, Ty::I32, 2);
            b.int(e, IntOp::Shl, pick, two)
        });
        if !converted(&b).is_empty() {
            wrong.push(format!("{:?} of lane 0 or 16: {:?}", op, converted(&b)));
        }
    }
    let b = permuted(WaveOp::ReadLane, |b, e, lane| {
        let sixteen = b.constant(e, Ty::I32, 16);
        b.int(e, IntOp::And, lane, sixteen)
    });
    if !converted(&b).is_empty() {
        wrong.push(format!("lane read of lane 0 or 16: {:?}", converted(&b)));
    }
    assert!(wrong.is_empty(), "lanes 0 and 16 hold their lane id whatever the query answers: {:?}", wrong);
}

#[test]
fn prove_keeps_a_query_whose_word_lane_zero_alone_reads_from_lane_one() {
    let mut wrong = Vec::new();
    for op in [WaveOp::Bpermute, WaveOp::BpermuteFi, WaveOp::ReadLane] {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let q = flag_query(&mut b, &k, e);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let (zero, one) = (b.constant(e, Ty::I32, 0), b.constant(e, Ty::I32, 1));
        let second = b.cmp(e, IntPred::Eq, lane, one);
        let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
        let picked = b.core(e, Ty::I32, Op::Select(q, ones, zero));
        let x = b.core(e, Ty::I32, Op::Select(second, picked, lane));
        let next = b.int(e, IntOp::Xor, lane, one);
        let y = match op {
            WaveOp::ReadLane => b.wave(e, WaveOp::ReadLane, vec![x, next, zero]),
            _ => {
                let two = b.constant(e, Ty::I32, 2);
                let byte = b.int(e, IntOp::Shl, next, two);
                b.wave(e, op, vec![byte, x, k.exec])
            }
        };
        let first = b.cmp(e, IntPred::Eq, lane, zero);
        let mask = b.int(e, IntOp::And, first, k.exec);
        store_own(&mut b, &k, e, y, mask);
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            if kept.queries.is_empty() {
                wrong.push(format!("{:?} {}", op, name));
            }
        }
    }
    assert!(wrong.is_empty(), "lane 0 stores lane 1's word, which depends on the query: {:?}", wrong);
}

#[test]
fn prove_keeps_a_query_whose_word_the_lane_a_permute_takes_holds() {
    let mut wrong = Vec::new();
    for op in [WaveOp::Bpermute, WaveOp::BpermuteFi, WaveOp::ReadLane] {
        let b = permuted(op, |b, e, lane| {
            let one = b.constant(e, Ty::I32, 1);
            let next = b.int(e, IntOp::Xor, lane, one);
            if op == WaveOp::ReadLane {
                next
            } else {
                let two = b.constant(e, Ty::I32, 2);
                b.int(e, IntOp::Shl, next, two)
            }
        });
        for (name, kept) in ["search", "direct"].iter().zip(both(&b)) {
            if kept.queries.is_empty() {
                wrong.push(format!("{:?} {}", op, name));
            }
        }
    }
    assert!(wrong.is_empty(), "lane 0 reads lane 1, whose word depends on the query: {:?}", wrong);
}

fn apart_merge(reach_store: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let u = uniform_load(&mut b, &k, e, 24);
    let five = b.constant(e, Ty::I32, 5);
    let uniform = b.cmp(e, IntPred::Eq, u, five);
    let flag = per_lane(&mut b, &k, e, 28);
    let zero = b.constant(e, Ty::I32, 0);
    let c = b.cmp(e, IntPred::Ne, flag, zero);
    let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (first, x) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
    let (second, y) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
    let (merged, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (arm, vec![k.exec, own, c, uniform]), (join, vec![k.exec]));
    b.cond_br(arm, a[3], (first, vec![a[0], a[1], a[2]]), (second, vec![a[0], a[1], a[2]]));
    b.br(first, merged, vec![x[0], x[1], x[2], x[2]]);
    let never = b.constant(second, Ty::I1, 0);
    b.br(second, merged, vec![y[0], y[1], y[2], never]);
    let mask = if reach_store {
        m[3]
    } else {
        let yes = b.constant(merged, Ty::I1, 1);
        let not_c = b.int(merged, IntOp::Xor, m[2], yes);
        let only = b.int(merged, IntOp::And, m[3], not_c);
        b.int(merged, IntOp::And, only, m[0])
    };
    let one = b.constant(merged, Ty::I32, 1);
    b.store(merged, Space::Global, MemSize::B32, m[1], one, mask);
    b.br(merged, join, vec![m[0]]);
    b
}

#[test]
fn prove_converts_a_query_whose_arm_merges_two_paths_that_never_store() {
    let b = apart_merge(false);
    assert!(converted(&b).is_empty(), "{:?}: z is c or false, so z & !c never holds and the arm never stores", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_arm_merges_two_paths_one_of_which_stores() {
    let b = apart_merge(true);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: z is c on the first path, so lanes with c store when the wave takes the arm", wrong);
}

fn apart_merge_answer(reach_store: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let u = uniform_load(&mut b, &k, e, 24);
    let five = b.constant(e, Ty::I32, 5);
    let uniform = b.cmp(e, IntPred::Eq, u, five);
    let w = uniform_load(&mut b, &k, e, 32);
    let seven = b.constant(e, Ty::I32, 7);
    let other = b.cmp(e, IntPred::Eq, w, seven);
    let flag = per_lane(&mut b, &k, e, 28);
    let zero = b.constant(e, Ty::I32, 0);
    let c = b.cmp(e, IntPred::Ne, flag, zero);
    let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I1]);
    let (first, x) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (second, y) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
    let (merged, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (arm, vec![k.exec, own, c, uniform, other]), (join, vec![k.exec]));
    b.cond_br(arm, a[3], (first, vec![a[0], a[1], a[2], a[4]]), (second, vec![a[0], a[1], a[2]]));
    let r = b.wave(first, WaveOp::Any, vec![x[3]]);
    let z = b.int(first, IntOp::And, x[2], r);
    b.br(first, merged, vec![x[0], x[1], x[2], z]);
    let never = b.constant(second, Ty::I1, 0);
    b.br(second, merged, vec![y[0], y[1], y[2], never]);
    let mask = if reach_store {
        b.int(merged, IntOp::And, m[3], m[0])
    } else {
        let yes = b.constant(merged, Ty::I1, 1);
        let not_c = b.int(merged, IntOp::Xor, m[2], yes);
        let only = b.int(merged, IntOp::And, m[3], not_c);
        b.int(merged, IntOp::And, only, m[0])
    };
    let one = b.constant(merged, Ty::I32, 1);
    b.store(merged, Space::Global, MemSize::B32, m[1], one, mask);
    b.br(merged, join, vec![m[0]]);
    b
}

#[test]
fn prove_converts_a_query_whose_arm_merges_an_answer_and_a_path_that_never_store() {
    let b = apart_merge_answer(false);
    assert!(converted(&b).is_empty(), "{:?}: z is c & any(v) or false, so z & !c never holds and the arm never stores", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_arm_merges_an_answer_that_stores_and_another_path() {
    let b = apart_merge_answer(true);
    assert!(keeps(&b).is_empty(), "{:?}: z is c & any(v) on the first path, so lanes with c store when the wave takes the arm and v holds", keeps(&b));
}

fn apart_merge_pair(differ: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let u = uniform_load(&mut b, &k, e, 24);
    let five = b.constant(e, Ty::I32, 5);
    let uniform = b.cmp(e, IntPred::Eq, u, five);
    let flag = per_lane(&mut b, &k, e, 28);
    let zero = b.constant(e, Ty::I32, 0);
    let c = b.cmp(e, IntPred::Ne, flag, zero);
    let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (first, x) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
    let (second, y) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
    let (merged, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (arm, vec![k.exec, own, c, uniform]), (join, vec![k.exec]));
    b.cond_br(arm, a[3], (first, vec![a[0], a[1], a[2]]), (second, vec![a[0], a[1], a[2]]));
    b.br(first, merged, vec![x[0], x[1], x[2], x[2]]);
    let yes = b.constant(second, Ty::I1, 1);
    let not_c = b.int(second, IntOp::Xor, y[2], yes);
    let other = if differ { y[2] } else { not_c };
    b.br(second, merged, vec![y[0], y[1], not_c, other]);
    let yes = b.constant(merged, Ty::I1, 1);
    let not_other = b.int(merged, IntOp::Xor, m[3], yes);
    let only = b.int(merged, IntOp::And, m[2], not_other);
    let mask = b.int(merged, IntOp::And, only, m[0]);
    let one = b.constant(merged, Ty::I32, 1);
    b.store(merged, Space::Global, MemSize::B32, m[1], one, mask);
    b.br(merged, join, vec![m[0]]);
    b
}

#[test]
fn prove_converts_a_query_whose_arm_merges_two_bits_that_every_path_keeps_equal() {
    let b = apart_merge_pair(false);
    assert!(converted(&b).is_empty(), "{:?}: the two bits are (c, c) on one path and (!c, !c) on the other, so s & !t never holds and the arm never stores", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_arm_merges_two_bits_that_one_path_sets_apart() {
    let b = apart_merge_pair(true);
    assert!(keeps(&b).is_empty(), "{:?}: the two bits are (!c, c) on the second path, so lanes without c store when the wave takes it", keeps(&b));
}

fn apart_merge_branch(reach_store: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let u = uniform_load(&mut b, &k, e, 24);
    let five = b.constant(e, Ty::I32, 5);
    let uniform = b.cmp(e, IntPred::Eq, u, five);
    let flag = per_lane(&mut b, &k, e, 28);
    let zero = b.constant(e, Ty::I32, 0);
    let c = b.cmp(e, IntPred::Ne, flag, zero);
    let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (first, x) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (second, y) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (merged, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (arm, vec![k.exec, own, c, uniform]), (join, vec![k.exec]));
    b.cond_br(arm, a[3], (first, vec![a[0], a[1], a[2], a[3]]), (second, vec![a[0], a[1], a[2], a[3]]));
    b.br(first, merged, vec![x[0], x[1], x[2], x[3]]);
    let never = b.constant(second, Ty::I1, 0);
    b.br(second, merged, vec![y[0], y[1], never, y[3]]);
    let branch = if reach_store {
        m[3]
    } else {
        let yes = b.constant(merged, Ty::I1, 1);
        b.int(merged, IntOp::Xor, m[3], yes)
    };
    let only = b.int(merged, IntOp::And, m[2], branch);
    let mask = b.int(merged, IntOp::And, only, m[0]);
    let one = b.constant(merged, Ty::I32, 1);
    b.store(merged, Space::Global, MemSize::B32, m[1], one, mask);
    b.br(merged, join, vec![m[0]]);
    b
}

#[test]
fn prove_converts_a_query_whose_arm_merges_a_bit_only_the_branch_to_it_sets() {
    let b = apart_merge_branch(false);
    assert!(converted(&b).is_empty(), "{:?}: z is c on the path taken when u holds and false on the other, so z & !u never holds and the arm never stores", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_arm_merges_a_bit_that_stores_where_the_branch_to_it_holds() {
    let b = apart_merge_branch(true);
    assert!(keeps(&b).is_empty(), "{:?}: z is c on the path taken when u holds, so lanes with c store when the wave takes the arm and u holds", keeps(&b));
}

fn apart_lane_mask(contradiction: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I64]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (then, vec![k.exec, own]), (join, vec![k.exec]));
    let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
    let sixteen = b.constant(then, Ty::I32, 16);
    let low = b.cmp(then, IntPred::Ult, lane, sixteen);
    let mask = if contradiction {
        let high = b.cmp(then, IntPred::Uge, lane, sixteen);
        b.int(then, IntOp::And, low, high)
    } else {
        low
    };
    let mask = b.int(then, IntOp::And, mask, t[0]);
    let one = b.constant(then, Ty::I32, 1);
    b.store(then, Space::Global, MemSize::B32, t[1], one, mask);
    b.br(then, join, vec![t[0]]);
    b
}

#[test]
fn prove_converts_a_query_whose_arm_stores_under_a_lane_contradiction() {
    let b = apart_lane_mask(true);
    assert!(converted(&b).is_empty(), "{:?}: lane < 16 and lane >= 16 never hold together, so the arm never stores", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_arm_stores_in_the_low_lanes() {
    let b = apart_lane_mask(false);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: lanes 0 to 15 store when the wave takes the arm", wrong);
}

fn lane_query(target: u64) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let target = b.constant(e, Ty::I32, target);
    let only = b.cmp(e, IntPred::Eq, lane, target);
    let c = b.int(e, IntOp::And, only, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let data = b.core(e, Ty::I32, Op::Select(q, one, two));
    store_own(&mut b, &k, e, data, k.exec);
    b
}

fn lane_bits_query(target: u64) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let one = b.constant(e, Ty::I32, 1);
    let odd = b.int(e, IntOp::And, lane, one);
    let target = b.constant(e, Ty::I32, target);
    let hit = b.cmp(e, IntPred::Eq, odd, target);
    let c = b.int(e, IntOp::And, hit, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let two = b.constant(e, Ty::I32, 2);
    let data = b.core(e, Ty::I32, Op::Select(q, one, two));
    store_own(&mut b, &k, e, data, k.exec);
    b
}

#[test]
fn prove_converts_a_query_whose_lane_bit_never_matches() {
    let b = lane_bits_query(2);
    assert!(converted(&b).is_empty(), "{:?}: lane & 1 is 0 or 1, never 2", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_lane_bit_matches_in_odd_lanes() {
    let b = lane_bits_query(1);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: odd lanes answer true, even lanes alone answer false", wrong);
}

#[test]
fn prove_converts_a_query_no_lane_can_answer_true() {
    let b = lane_query(99);
    assert!(converted(&b).is_empty(), "{:?}: no lane is lane 99, so the query is false in the wave and in every lane", converted(&b));
}

#[test]
fn prove_keeps_a_query_one_lane_can_answer_true() {
    let b = lane_query(5);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: when lane 5 is active the wave answers true, but every other lane alone answers false", wrong);
}

#[test]
fn prove_converts_a_query_whose_arms_store_the_same_word() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I64]);
    let (other, o) = b.block(&[Ty::I1, Ty::I64]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (then, vec![k.exec, own]), (other, vec![k.exec, own]));
    for (block, p) in [(then, t), (other, o)] {
        let one = b.constant(block, Ty::I32, 1);
        b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
        b.br(block, join, vec![p[0]]);
    }
    assert!(converted(&b).is_empty(), "{:?}: both arms store 1 to the lane's own word", converted(&b));
}

fn arms_store(arm: impl Fn(&mut Build, BlockId, usize, &[ValueId]) -> Option<ValueId>) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
    let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
    let (join, j) = b.block(&[Ty::I1, Ty::I32]);
    b.cond_br(e, q, (then, vec![k.exec, own, lane]), (other, vec![k.exec, own, lane]));
    for (side, (block, p)) in [(then, t), (other, o)].iter().enumerate() {
        let carried = arm(&mut b, *block, side, p).unwrap_or(p[2]);
        b.br(*block, join, vec![p[0], carried]);
    }
    let out = k.buffer(&mut b, join, 16);
    let target = byte_offset(&mut b, join, out, j[1], 4);
    let one = b.constant(join, Ty::I32, 1);
    b.store(join, Space::Global, MemSize::B32, target, one, j[0]);
    b
}

fn keeps(b: &Build) -> Vec<&'static str> {
    ["search", "direct"].iter().zip(both(b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect()
}

#[test]
fn prove_keeps_a_query_whose_arms_store_different_values() {
    let b = arms_store(|b, block, side, p| {
        let value = b.constant(block, Ty::I32, 1 + side as u64);
        b.store(block, Space::Global, MemSize::B32, p[1], value, p[0]);
        None
    });
    assert!(keeps(&b).is_empty(), "{:?}: one arm stores 1 and the other 2", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_arms_store_to_different_words() {
    let b = arms_store(|b, block, side, p| {
        let at = b.constant(block, Ty::I64, 4 * side as u64);
        let address = b.int(block, IntOp::Add, p[1], at);
        let one = b.constant(block, Ty::I32, 1);
        b.store(block, Space::Global, MemSize::B32, address, one, p[0]);
        None
    });
    assert!(keeps(&b).is_empty(), "{:?}: one arm stores to the lane's word and the other to the next", keeps(&b));
}

#[test]
fn prove_keeps_a_query_only_one_of_whose_arms_stores() {
    let b = arms_store(|b, block, side, p| {
        if side == 0 {
            let one = b.constant(block, Ty::I32, 1);
            b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
        }
        None
    });
    assert!(keeps(&b).is_empty(), "{:?}: only one arm stores", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_arms_store_under_different_masks() {
    let b = arms_store(|b, block, side, p| {
        let mask = if side == 0 {
            p[0]
        } else {
            let sixteen = b.constant(block, Ty::I32, 16);
            let low = b.cmp(block, IntPred::Ult, p[2], sixteen);
            b.int(block, IntOp::And, low, p[0])
        };
        let one = b.constant(block, Ty::I32, 1);
        b.store(block, Space::Global, MemSize::B32, p[1], one, mask);
        None
    });
    assert!(keeps(&b).is_empty(), "{:?}: lanes 16 to 31 store in one arm only", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_arms_load_the_stored_word_at_different_times() {
    let b = arms_store(|b, block, side, p| {
        let one = b.constant(block, Ty::I32, 1);
        if side == 0 {
            b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
            Some(b.load(block, Space::Global, MemSize::B32, p[1], p[0]))
        } else {
            let old = b.load(block, Space::Global, MemSize::B32, p[1], p[0]);
            b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
            Some(old)
        }
    });
    assert!(keeps(&b).is_empty(), "{:?}: one arm reads back 1 and the other the word before the store", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_arms_store_one_word_computed_in_each_arm() {
    let b = arms_store(|b, block, _, p| {
        let zero = b.constant(block, Ty::I64, 0);
        let address = b.int(block, IntOp::Add, p[1], zero);
        let one = b.constant(block, Ty::I32, 1);
        b.store(block, Space::Global, MemSize::B32, address, one, p[0]);
        None
    });
    assert!(converted(&b).is_empty(), "{:?}: both arms store 1 to the lane's own word", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_arms_add_different_amounts_atomically() {
    let b = arms_store(|b, block, side, p| {
        let amount = b.constant(block, Ty::I32, 1 + side as u64);
        b.effect(block, memory(Space::Global, MemoryOp::AtomicAdd(Numeric::Unsigned)), vec![p[1], amount, p[0]]);
        None
    });
    assert!(keeps(&b).is_empty(), "{:?}: one arm adds 1 and the other 2", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_arms_add_the_same_amount_atomically() {
    let b = arms_store(|b, block, _, p| {
        let one = b.constant(block, Ty::I32, 1);
        b.effect(block, memory(Space::Global, MemoryOp::AtomicAdd(Numeric::Unsigned)), vec![p[1], one, p[0]]);
        None
    });
    assert!(converted(&b).is_empty(), "{:?}: both arms add 1 to the lane's own word", converted(&b));
}

fn arms_store_for_neighbours(second: u64) -> Vec<&'static str> {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let one = b.constant(e, Ty::I32, 1);
    let next = b.int(e, IntOp::Xor, lane, one);
    let neighbour = byte_offset(&mut b, e, buf, next, 4);
    let out = k.buffer(&mut b, e, 16);
    let slot = byte_offset(&mut b, e, out, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64, Ty::I64]);
    let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I64, Ty::I64]);
    let (join, j) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    let args = vec![k.exec, own, neighbour, slot];
    b.cond_br(e, q, (then, args.clone()), (other, args));
    let mut stores = Vec::new();
    for (side, (block, p)) in vec![(then, t), (other, o)].into_iter().enumerate() {
        let value = b.constant(block, Ty::I32, if side == 0 { 1 } else { second });
        stores.push(b.here(block));
        b.store(block, Space::Global, MemSize::B32, p[1], value, p[0]);
        b.br(block, join, vec![p[0], p[2], p[3]]);
    }
    let l = b.here(join);
    let read = b.load(join, Space::Global, MemSize::B32, j[1], j[0]);
    b.store(join, Space::Global, MemSize::B32, j[2], read, j[0]);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(stores[0], l), (stores[1], l)], &[], &[]);
    let (f, inputs) = (&program.ir, &program.parameter_inputs);
    vec![("search", search::prove(f, inputs, Some(0), &hazards).0), ("direct", direct::prove(f, inputs, Some(0), &hazards).0)]
        .into_iter()
        .filter(|(_, kept)| kept.queries.contains(&q) == (second == 1))
        .map(|(name, _)| name)
        .collect()
}

#[test]
fn prove_keeps_a_query_whose_arms_store_the_same_words_a_neighbour_overwrites() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let one = b.constant(e, Ty::I32, 1);
    let next = b.int(e, IntOp::Xor, lane, one);
    let neighbour = byte_offset(&mut b, e, buf, next, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    let (join, _) = b.block(&[Ty::I1]);
    let args = vec![k.exec, own, neighbour];
    b.cond_br(e, q, (then, args.clone()), (other, args));
    let mut stores = Vec::new();
    for (block, p) in vec![(then, t), (other, o)] {
        let first = b.constant(block, Ty::I32, 1);
        let second = b.constant(block, Ty::I32, 2);
        let s1 = b.here(block);
        b.store(block, Space::Global, MemSize::B32, p[1], first, p[0]);
        let s2 = b.here(block);
        b.store(block, Space::Global, MemSize::B32, p[2], second, p[0]);
        stores.push((s1, s2));
        b.br(block, join, vec![p[0]]);
    }
    let program = b.program();
    let pairs = [(stores[0].0, stores[0].1), (stores[1].0, stores[1].1), (stores[0].0, stores[1].1), (stores[1].0, stores[0].1)];
    let hazards = Hazards::given(&program, &pairs, &[], &[]);
    let (f, inputs) = (&program.ir, &program.parameter_inputs);
    let wrong: Vec<&str> = vec![("search", search::prove(f, inputs, Some(0), &hazards).0), ("direct", direct::prove(f, inputs, Some(0), &hazards).0)]
        .into_iter()
        .filter(|(_, kept)| !kept.queries.contains(&q))
        .map(|(name, _)| name)
        .collect();
    assert!(wrong.is_empty(), "{:?}: a lane in one arm may write 2 into its neighbour's word before the neighbour, in the other arm, writes 1 into it", wrong);
}

#[test]
fn prove_keeps_a_query_whose_arms_store_different_values_a_neighbour_reads() {
    let wrong = arms_store_for_neighbours(2);
    assert!(wrong.is_empty(), "{:?}: one arm stores 1 and the other 2, which the neighbour reads back", wrong);
}

#[test]
fn prove_converts_a_query_whose_arms_store_the_same_word_a_neighbour_reads() {
    let wrong = arms_store_for_neighbours(1);
    assert!(wrong.is_empty(), "{:?}: both arms store 1 to the lane's own word, and a meeting before the neighbour's read orders either", wrong);
}

fn bound_into_a_difference(inside: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let x = per_lane(&mut b, &k, e, 16);
    let five = b.constant(e, Ty::I32, 5);
    let small = b.cmp(e, IntPred::Ult, x, five);
    let three = b.constant(e, Ty::I32, 3);
    let v = b.core(e, Ty::I32, Op::Select(small, data, three));
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
    b.br(e, next, vec![k.exec, own, x, v]);
    let five = b.constant(next, Ty::I32, 5);
    let test = b.cmp(next, if inside { IntPred::Ult } else { IntPred::Uge }, p[2], five);
    let mask = b.int(next, IntOp::And, test, p[0]);
    b.store(next, Space::Global, MemSize::B32, p[1], p[3], mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_word_a_bound_carried_into_the_next_block_lets_through() {
    let b = bound_into_a_difference(true);
    assert!(keeps(&b).is_empty(), "{:?}: lanes with x < 5 store the answer", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_word_a_bound_carried_into_the_next_block_rules_out() {
    let b = bound_into_a_difference(false);
    assert!(converted(&b).is_empty(), "{:?}: lanes with x >= 5 store 3, which the query never reaches", converted(&b));
}

fn sum_bound_into_a_difference(bytes: bool, bound: u64, next_block: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let size = if bytes { MemSize::U8 } else { MemSize::B32 };
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let (first, second) = (k.buffer(&mut b, e, 16), k.buffer(&mut b, e, 24));
    let (own_x, own_y) = (byte_offset(&mut b, e, first, lane, 4), byte_offset(&mut b, e, second, lane, 4));
    let x = b.load(e, Space::Global, size, own_x, k.exec);
    let y = b.load(e, Space::Global, size, own_y, k.exec);
    let sum = b.int(e, IntOp::Add, x, y);
    let ten = b.constant(e, Ty::I32, 10);
    let small = b.cmp(e, IntPred::Ult, sum, ten);
    let three = b.constant(e, Ty::I32, 3);
    let v = b.core(e, Ty::I32, Op::Select(small, data, three));
    let buf = k.buffer(&mut b, e, 0);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (block, exec, own, x, v) = if next_block {
        let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
        b.br(e, next, vec![k.exec, own, x, v]);
        (next, p[0], p[1], p[2], p[3])
    } else {
        (e, k.exec, own, x, v)
    };
    let bound = b.constant(block, Ty::I32, bound);
    let test = b.cmp(block, IntPred::Uge, x, bound);
    let mask = b.int(block, IntOp::And, test, exec);
    b.store(block, Space::Global, MemSize::B32, own, v, mask);
    b
}

#[test]
fn prove_converts_a_query_whose_word_a_bound_on_a_summand_rules_out_a_bound_on_the_sum() {
    let b = sum_bound_into_a_difference(true, 10, false);
    assert!(converted(&b).is_empty(), "{:?}: bytes with x >= 10 have x + y >= 10, so they store 3", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_word_a_looser_bound_on_a_summand_lets_the_sum_through() {
    let b = sum_bound_into_a_difference(true, 5, false);
    assert!(keeps(&b).is_empty(), "{:?}: x = 5 and y = 0 pass both tests and store the answer", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_word_a_bound_on_a_summand_lets_a_wrapping_sum_through() {
    let b = sum_bound_into_a_difference(false, 10, false);
    assert!(keeps(&b).is_empty(), "{:?}: x = 0xffffffff and y = 1 sum to 0 and store the answer", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_word_a_bound_on_a_summand_carried_into_the_next_block_rules_out_a_bound_on_the_sum() {
    let b = sum_bound_into_a_difference(true, 10, true);
    assert!(converted(&b).is_empty(), "{:?}: bytes with x >= 10 have x + y >= 10, so the next block stores 3", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_word_a_looser_bound_on_a_summand_carried_into_the_next_block_lets_the_sum_through() {
    let b = sum_bound_into_a_difference(true, 5, true);
    assert!(keeps(&b).is_empty(), "{:?}: x = 5 and y = 0 pass both tests and the next block stores the answer", keeps(&b));
}

fn word_test_into_a_query(uniform: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let w = if uniform { uniform_load(&mut b, &k, e, 16) } else { b.wave(e, WaveOp::Ballot, vec![c]) };
    let t = b.cmp(e, IntPred::Ne, w, zero);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
    b.br(e, next, vec![k.exec, own, t]);
    let q = b.wave(next, WaveOp::Any, vec![p[2]]);
    let one = b.constant(next, Ty::I32, 1);
    let zero = b.constant(next, Ty::I32, 0);
    let v = b.core(next, Ty::I32, Op::Select(q, one, zero));
    b.store(next, Space::Global, MemSize::B32, p[1], v, p[0]);
    b
}

#[test]
fn prove_keeps_the_word_or_the_query_when_a_test_of_a_ballot_feeds_a_query_in_the_next_block() {
    let b = word_test_into_a_query(false);
    let kept = converted(&b);
    assert_eq!(kept, ["search", "direct"], "converting both the word test and the query stores each lane's own bit instead of whether any lane is set");
}

#[test]
fn prove_converts_a_query_over_a_test_of_a_uniform_word_in_the_next_block() {
    let b = word_test_into_a_query(true);
    assert!(converted(&b).is_empty(), "{:?}: the test of a uniform word is the same in every lane", converted(&b));
}

fn arms_store_two(order: bool, second: u64) -> Build {
    arms_store(move |b, block, side, p| {
        let past = b.constant(block, Ty::I64, 128);
        let next = b.int(block, IntOp::Add, p[1], past);
        let one = b.constant(block, Ty::I32, 1);
        let other = b.constant(block, Ty::I32, if side == 0 { 2 } else { second });
        if side == 0 || !order {
            b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
            b.store(block, Space::Global, MemSize::B32, next, other, p[0]);
        } else {
            b.store(block, Space::Global, MemSize::B32, next, other, p[0]);
            b.store(block, Space::Global, MemSize::B32, p[1], one, p[0]);
        }
        None
    })
}

#[test]
fn prove_keeps_a_query_whose_arms_store_a_second_word_differently() {
    let b = arms_store_two(true, 3);
    assert!(keeps(&b).is_empty(), "{:?}: one arm stores 2 into the word 32 further and the other 3", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_arms_store_two_words_in_either_order() {
    let b = arms_store_two(true, 2);
    assert!(converted(&b).is_empty(), "{:?}: both arms store 1 into the lane's word and 2 into the word 32 further, which no other lane of the wave touches", converted(&b));
}

fn ordered_three(closed: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let x = per_lane(&mut b, &k, e, 16);
    let y = per_lane(&mut b, &k, e, 24);
    let z = per_lane(&mut b, &k, e, 32);
    let xy = b.cmp(e, IntPred::Ult, x, y);
    let yz = b.cmp(e, IntPred::Ult, y, z);
    let third = if closed { b.cmp(e, IntPred::Ult, z, x) } else { b.cmp(e, IntPred::Ult, x, z) };
    let both = b.int(e, IntOp::And, xy, yz);
    let all = b.int(e, IntOp::And, both, third);
    let mask = b.int(e, IntOp::And, all, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_three_ordered_words_mask() {
    let b = ordered_three(false);
    assert!(keeps(&b).is_empty(), "{:?}: x < y < z holds for x = 0, y = 1, z = 2", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_a_cycle_of_orders_masks() {
    let b = ordered_three(true);
    assert!(converted(&b).is_empty(), "{:?}: x < y < z < x never holds", converted(&b));
}

fn signed_and_unsigned(below: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let x = per_lane(&mut b, &k, e, 16);
    let zero = b.constant(e, Ty::I32, 0);
    let ten = b.constant(e, Ty::I32, 10);
    let negative = b.cmp(e, IntPred::Slt, x, zero);
    let small = b.cmp(e, if below { IntPred::Ult } else { IntPred::Uge }, x, ten);
    let both = b.int(e, IntOp::And, negative, small);
    let mask = b.int(e, IntOp::And, both, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_a_negative_word_at_least_ten_unsigned_masks() {
    let b = signed_and_unsigned(false);
    assert!(keeps(&b).is_empty(), "{:?}: -1 is below 0 signed and at least 10 unsigned", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_a_negative_word_below_ten_unsigned_masks() {
    let b = signed_and_unsigned(true);
    assert!(converted(&b).is_empty(), "{:?}: a word below 0 signed is at least 2^31 unsigned", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_store_a_bound_that_admits_the_constant_lets_through() {
    let b = branch_then_store((IntPred::Ult, 5), (IntPred::Eq, 3));
    assert!(keeps(&b).is_empty(), "{:?}: x = 3 is below 5", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_a_bound_into_its_block_rules_out_by_equality() {
    let b = branch_then_store((IntPred::Ult, 5), (IntPred::Eq, 7));
    assert!(converted(&b).is_empty(), "{:?}: x < 5 in the block, where x == 7 is false", converted(&b));
}

fn permuted_from(odd: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let zero = b.constant(e, Ty::I32, 0);
    let first = b.cmp(e, IntPred::Eq, lane, zero);
    let seven = b.constant(e, Ty::I32, 7);
    let x = b.core(e, Ty::I32, Op::Select(first, data, seven));
    let u = uniform_load(&mut b, &k, e, 16);
    let two = b.constant(e, Ty::I32, 2);
    let bit = b.int(e, IntOp::And, u, two);
    let one = b.constant(e, Ty::I32, 1);
    let source = if odd {
        let raised = b.int(e, IntOp::Or, lane, one);
        b.int(e, IntOp::Or, raised, bit)
    } else {
        b.int(e, IntOp::And, u, one)
    };
    let four = b.constant(e, Ty::I32, 4);
    let index = b.int(e, IntOp::Mul, source, four);
    let yes = b.constant(e, Ty::I1, 1);
    let read = b.wave(e, WaveOp::Bpermute, vec![index, x, yes]);
    store_own(&mut b, &k, e, read, k.exec);
    b
}

#[test]
fn prove_keeps_a_query_whose_word_a_permute_may_fetch_from_lane_zero() {
    let b = permuted_from(false);
    assert!(keeps(&b).is_empty(), "{:?}: when u is even every lane fetches lane 0's word, which the query picks", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_word_a_permute_never_fetches() {
    let b = permuted_from(true);
    assert!(converted(&b).is_empty(), "{:?}: every lane fetches from an odd lane, whose word is 7", converted(&b));
}

fn over_active_bits(outside: u64, every: bool) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let sixteen = b.constant(e, Ty::I32, 16);
    let low = b.cmp(e, IntPred::Ult, lane, sixteen);
    let exec = b.int(e, IntOp::And, low, k.exec);
    let flags = k.buffer(&mut b, e, 8);
    let buf = k.buffer(&mut b, e, 0);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    b.br(e, then, vec![exec, flags, buf]);
    let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, then, t[1], lane, 4);
    let yes = b.constant(then, Ty::I1, 1);
    let flag = b.load(then, Space::Global, MemSize::B32, own, yes);
    let other = b.constant(then, Ty::I32, outside);
    let v = b.core(then, Ty::I32, Op::Select(t[0], flag, other));
    let five = b.constant(then, Ty::I32, 5);
    let small = b.cmp(then, IntPred::Ult, v, five);
    let at = b.here(then);
    let q = b.wave(then, WaveOp::Any, vec![small]);
    let one = b.constant(then, Ty::I32, 1);
    let two = b.constant(then, Ty::I32, 2);
    let data = b.core(then, Ty::I32, Op::Select(q, one, two));
    let out = byte_offset(&mut b, then, t[2], lane, 4);
    b.store(then, Space::Global, MemSize::B32, out, data, t[0]);
    let Inst::Effect { provenance, .. } = b.f.blocks[&then].insts[at.1] else { unreachable!() };
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
        assert!(kept.queries.contains(&q), "{}: a lane whose flag is small stores 1, the others 2", name);
        assert_eq!(everyone.contains(&provenance), every, "{}: a lane with exec clear sees {}", name, outside);
    }
}

fn over_active_floats(outside: u64, every: bool) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let sixteen = b.constant(e, Ty::I32, 16);
    let low = b.cmp(e, IntPred::Ult, lane, sixteen);
    let exec = b.int(e, IntOp::And, low, k.exec);
    let flags = k.buffer(&mut b, e, 8);
    let buf = k.buffer(&mut b, e, 0);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    b.br(e, then, vec![exec, flags, buf]);
    let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, then, t[1], lane, 4);
    let yes = b.constant(then, Ty::I1, 1);
    let word = b.load(then, Space::Global, MemSize::B32, own, yes);
    let flag = b.core(then, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, word));
    let other = b.constant(then, Ty::F32, outside);
    let v = b.core(then, Ty::F32, Op::Select(t[0], flag, other));
    let five = b.constant(then, Ty::F32, 0x40a0_0000);
    let small = b.core(then, Ty::I1, Op::FCmp(FloatPred::Olt, v, five));
    let at = b.here(then);
    let q = b.wave(then, WaveOp::Any, vec![small]);
    let one = b.constant(then, Ty::I32, 1);
    let two = b.constant(then, Ty::I32, 2);
    let data = b.core(then, Ty::I32, Op::Select(q, one, two));
    let out = byte_offset(&mut b, then, t[2], lane, 4);
    b.store(then, Space::Global, MemSize::B32, out, data, t[0]);
    let Inst::Effect { provenance, .. } = b.f.blocks[&then].insts[at.1] else { unreachable!() };
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
        assert!(kept.queries.contains(&q), "{}: a lane whose flag is small stores 1, the others 2", name);
        assert_eq!(everyone.contains(&provenance), every, "{}: a lane with exec clear sees {:#x}", name, outside);
    }
}

#[test]
fn prove_demands_no_lane_for_a_kept_query_over_float_bits_only_active_lanes_set() {
    over_active_floats(0x42c8_0000, false);
}

#[test]
fn prove_demands_every_lane_for_a_kept_query_over_float_bits_inactive_lanes_set() {
    over_active_floats(0x4040_0000, true);
}

fn over_active_products(masked: bool, every: bool) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let sixteen = b.constant(e, Ty::I32, 16);
    let low = b.cmp(e, IntPred::Ult, lane, sixteen);
    let exec = b.int(e, IntOp::And, low, k.exec);
    let flags = k.buffer(&mut b, e, 8);
    let buf = k.buffer(&mut b, e, 0);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    b.br(e, then, vec![exec, flags, buf]);
    let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, then, t[1], lane, 4);
    let yes = b.constant(then, Ty::I1, 1);
    let flag = b.load(then, Space::Global, MemSize::B32, own, yes);
    let factor = b.core(then, Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, if masked { t[0] } else { yes }));
    let v = b.int(then, IntOp::Mul, flag, factor);
    let zero = b.constant(then, Ty::I32, 0);
    let set = b.cmp(then, IntPred::Ne, v, zero);
    let at = b.here(then);
    let q = b.wave(then, WaveOp::Any, vec![set]);
    let one = b.constant(then, Ty::I32, 1);
    let two = b.constant(then, Ty::I32, 2);
    let data = b.core(then, Ty::I32, Op::Select(q, one, two));
    let out = byte_offset(&mut b, then, t[2], lane, 4);
    b.store(then, Space::Global, MemSize::B32, out, data, t[0]);
    let Inst::Effect { provenance, .. } = b.f.blocks[&then].insts[at.1] else { unreachable!() };
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
        assert!(kept.queries.contains(&q), "{}: a lane whose flag is set stores 1, the others 2", name);
        assert_eq!(everyone.contains(&provenance), every, "{}: a lane with exec clear multiplies its flag by {}", name, !masked as u32);
    }
}

#[test]
fn prove_demands_no_lane_for_a_kept_query_over_products_with_the_exec_bit() {
    over_active_products(true, false);
}

#[test]
fn prove_demands_every_lane_for_a_kept_query_over_products_with_one() {
    over_active_products(false, true);
}

#[test]
fn prove_demands_no_lane_for_a_kept_query_over_bits_only_active_lanes_set() {
    over_active_bits(100, false);
}

#[test]
fn prove_demands_every_lane_for_a_kept_query_over_bits_inactive_lanes_set() {
    over_active_bits(3, true);
}

fn flag_query(b: &mut Build, k: &Kernel, e: BlockId) -> ValueId {
    let flag = per_lane(b, k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    b.wave(e, WaveOp::Any, vec![c])
}

#[test]
fn prove_keeps_a_query_whose_arm_holds_a_meeting() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let buf = k.buffer(&mut b, e, 16);
    let (then, t) = b.block(&[Ty::I1, Ty::I64]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (then, vec![k.exec, buf]), (join, vec![k.exec]));
    let s1 = store(&mut b, then, t[1], 1, t[0]);
    let s2 = store(&mut b, then, t[1], 2, t[0]);
    b.br(then, join, vec![t[0]]);
    let program = b.program();
    let hazards = Hazards::given(&program, &[(s1, s2)], &[], &[]);
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (kept, _) = prove(&program.ir, &program.parameter_inputs, Some(0), &hazards);
        assert!(kept.queries.contains(&q), "{}: a lane without the flag skips the arm whose meeting every lane must reach", name);
        let meets: BTreeSet<Position> = kept.meets.iter().map(|&m| hazards.meetings[m]).collect();
        assert_eq!(meets, BTreeSet::from([s2]), "{}: the second store must follow the first", name);
    }
}

fn arms_carry(first: u64, second: u64) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I64]);
    let (other, o) = b.block(&[Ty::I1, Ty::I64]);
    let (join, j) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
    b.cond_br(e, q, (then, vec![k.exec, own]), (other, vec![k.exec, own]));
    let x = b.constant(then, Ty::I32, first);
    b.br(then, join, vec![t[0], t[1], x]);
    let y = b.constant(other, Ty::I32, second);
    b.br(other, join, vec![o[0], o[1], y]);
    b.store(join, Space::Global, MemSize::B32, j[1], j[2], j[0]);
    (b, q)
}

#[test]
fn prove_keeps_a_query_whose_arms_carry_different_words_to_a_store() {
    let (b, q) = arms_carry(1, 2);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| !kept.queries.contains(&q)).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag stores 2 where the wave stores 1", wrong);
}

#[test]
fn prove_converts_a_query_whose_arms_carry_the_same_word() {
    let (b, _) = arms_carry(1, 1);
    assert!(converted(&b).is_empty(), "{:?}: both arms hand the store 1", converted(&b));
}

fn ballot_count(masked: bool) -> (Build, u64) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let five = b.constant(e, Ty::I32, 5);
    let low = b.cmp(e, IntPred::Ult, lane, five);
    let x = if masked { b.int(e, IntOp::And, low, k.exec) } else { low };
    let at = b.here(e);
    let w = b.wave(e, WaveOp::Ballot, vec![x]);
    let count = b.core(e, Ty::I32, Op::PopulationCount(w));
    store_own(&mut b, &k, e, count, k.exec);
    let Inst::Effect { provenance, .. } = b.f.blocks[&e].insts[at.1] else { unreachable!() };
    (b, provenance)
}

#[test]
fn prove_demands_every_lane_for_a_ballot_of_unmasked_bits() {
    let (b, provenance) = ballot_count(false);
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (_, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
        assert!(everyone.contains(&provenance), "{}: the ballot counts the bits of lanes whose exec is clear", name);
    }
}

#[test]
fn prove_demands_no_lane_for_a_ballot_of_masked_bits() {
    let (b, provenance) = ballot_count(true);
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (_, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
        assert!(!everyone.contains(&provenance), "{}: a lane with exec clear adds no bit", name);
    }
}

#[test]
fn prove_keeps_a_query_that_picks_which_whole_word_a_count_reads() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let four = b.constant(e, Ty::I32, 4);
    let low = b.cmp(e, IntPred::Ult, lane, four);
    let low = b.int(e, IntOp::And, low, k.exec);
    let w1 = b.wave(e, WaveOp::Ballot, vec![low]);
    let w2 = b.wave(e, WaveOp::Ballot, vec![k.exec]);
    let w = b.core(e, Ty::I32, Op::Select(q, w1, w2));
    let count = b.core(e, Ty::I32, Op::PopulationCount(w));
    store_own(&mut b, &k, e, count, k.exec);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| !kept.queries.contains(&q)).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: a lane without the flag counts 32 bits where the wave counts 4", wrong);
}

#[test]
fn prove_keeps_a_query_whose_answer_a_kept_query_gathers_from_other_lanes() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let first = flag_query(&mut b, &k, e);
    let flag = per_lane(&mut b, &k, e, 16);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let x = b.int(e, IntOp::And, first, c);
    let second = b.wave(e, WaveOp::Any, vec![x]);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let data = b.core(e, Ty::I32, Op::Select(second, one, two));
    store_own(&mut b, &k, e, data, k.exec);
    let wrong: Vec<(&str, usize)> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&first) || !kept.queries.contains(&second))
        .map(|(n, kept)| (*n, kept.queries.len()))
        .collect();
    assert!(
        wrong.is_empty(),
        "{:?}: with the first query converted, the second gathers d_i & c_i instead of any(d) & c_i, which differ when d and c are set in different lanes",
        wrong
    );
}

#[test]
fn prove_keeps_a_query_that_only_other_lanes_feed_into_a_kept_query() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let first = flag_query(&mut b, &k, e);
    let flag = per_lane(&mut b, &k, e, 16);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let x = b.int(e, IntOp::And, first, c);
    let second = b.wave(e, WaveOp::Any, vec![x]);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let data = b.core(e, Ty::I32, Op::Select(second, one, two));
    let yes = b.constant(e, Ty::I1, 1);
    let clear = b.int(e, IntOp::Xor, set, yes);
    let mask = b.int(e, IntOp::And, clear, k.exec);
    store_own(&mut b, &k, e, data, mask);
    let wrong: Vec<(&str, Vec<bool>)> = ["search", "direct"]
        .iter()
        .zip(both(&b))
        .filter(|(_, kept)| !kept.queries.contains(&first) || !kept.queries.contains(&second))
        .map(|(n, kept)| (*n, vec![kept.queries.contains(&first), kept.queries.contains(&second)]))
        .collect();
    assert!(
        wrong.is_empty(),
        "{:?} (first kept, second kept): the lanes that store have c clear, yet the second query gathers d_i & c_i from the others instead of any(d) & c_i",
        wrong
    );
}

#[test]
fn prove_converts_a_query_whose_arm_stores_only_for_lanes_that_hold_the_bit() {
    let Flagged { mut b, k, buf, lane, c } = flagged();
    let e = BlockId(0);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I1]);
    let (join, _) = b.block(&[Ty::I1]);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    b.cond_br(e, q, (then, vec![k.exec, own, c]), (join, vec![k.exec]));
    let one = b.constant(then, Ty::I32, 1);
    b.store(then, Space::Global, MemSize::B32, t[1], one, t[2]);
    b.br(then, join, vec![t[0]]);
    assert!(converted(&b).is_empty(), "{:?}: a lane that stores holds the bit, so it takes the arm in both programs", converted(&b));
}

#[test]
fn prove_converts_a_query_whose_moved_address_only_lanes_that_hold_the_bit_use() {
    let Flagged { mut b, buf, lane, c, .. } = flagged();
    let e = BlockId(0);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let near = b.constant(e, Ty::I64, 0);
    let far = b.constant(e, Ty::I64, 4);
    let shift = b.core(e, Ty::I64, Op::Select(q, near, far));
    let own = byte_offset(&mut b, e, buf, lane, 8);
    let address = b.int(e, IntOp::Add, own, shift);
    let one = b.constant(e, Ty::I32, 1);
    b.store(e, Space::Global, MemSize::B32, address, one, c);
    assert!(converted(&b).is_empty(), "{:?}: a lane that stores holds the bit and sees the query true in both programs", converted(&b));
}

#[test]
fn prove_converts_a_query_whose_picked_word_only_lanes_that_hold_the_bit_load() {
    let Flagged { mut b, k, buf, lane, c } = flagged();
    let e = BlockId(0);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let table = k.buffer(&mut b, e, 16);
    let four = b.constant(e, Ty::I64, 4);
    let second = b.int(e, IntOp::Add, table, four);
    let from = b.core(e, Ty::I64, Op::Select(q, table, second));
    let v = b.load(e, Space::Global, MemSize::B32, from, c);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    b.store(e, Space::Global, MemSize::B32, own, v, c);
    assert!(converted(&b).is_empty(), "{:?}: a lane that loads and stores holds the bit and picks the first word in both programs", converted(&b));
}

#[test]
fn prove_converts_a_query_that_masks_only_lanes_that_hold_the_bit() {
    let Flagged { mut b, buf, lane, c, .. } = flagged();
    let e = BlockId(0);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let mask = b.int(e, IntOp::And, q, c);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let one = b.constant(e, Ty::I32, 1);
    b.store(e, Space::Global, MemSize::B32, own, one, mask);
    assert!(converted(&b).is_empty(), "{:?}: q & c is c in both programs", converted(&b));
}

#[test]
fn prove_converts_a_ballot_whose_lane_test_only_lanes_that_hold_the_bit_use() {
    let Flagged { mut b, buf, lane, c, .. } = flagged();
    let e = BlockId(0);
    let w = b.wave(e, WaveOp::Ballot, vec![c]);
    let zero = b.constant(e, Ty::I32, 0);
    let any = b.cmp(e, IntPred::Ne, w, zero);
    let mask = b.int(e, IntOp::And, any, c);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let one = b.constant(e, Ty::I32, 1);
    b.store(e, Space::Global, MemSize::B32, own, one, mask);
    assert!(converted(&b).is_empty(), "{:?}: (ballot(c) != 0) & c is c in both programs", converted(&b));
}

#[test]
fn prove_converts_a_query_whose_whole_word_only_lanes_that_hold_the_bit_count() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let four = b.constant(e, Ty::I32, 4);
    let low = b.cmp(e, IntPred::Ult, lane, four);
    let low = b.int(e, IntOp::And, low, k.exec);
    let w1 = b.wave(e, WaveOp::Ballot, vec![low]);
    let w2 = b.wave(e, WaveOp::Ballot, vec![k.exec]);
    let w = b.core(e, Ty::I32, Op::Select(q, w1, w2));
    let count = b.core(e, Ty::I32, Op::PopulationCount(w));
    store_own(&mut b, &k, e, count, c);
    assert!(converted(&b).is_empty(), "{:?}: a lane that stores holds the bit and counts the first word in both programs", converted(&b));
}

fn arms_compute(then_arm: usize, other_arm: usize) -> (Build, ValueId) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let one = b.constant(e, Ty::I32, 1);
    let next = b.int(e, IntOp::Add, lane, one);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
    let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32]);
    let (join, j) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
    b.cond_br(e, q, (then, vec![k.exec, own, lane, next]), (other, vec![k.exec, own, lane, next]));
    for (block, p, arm) in [(then, t, then_arm), (other, o, other_arm)] {
        let (x, y) = (p[2], p[3]);
        let v = match arm {
            0 => x,
            1 => {
                let pair = b.core(block, Ty::I64, Op::Pack64(x, y));
                b.core(block, Ty::I32, Op::UnpackLo(pair))
            }
            2 => {
                let pair = b.core(block, Ty::I64, Op::Pack64(x, y));
                b.core(block, Ty::I32, Op::UnpackHi(pair))
            }
            3 => {
                let ones = b.constant(block, Ty::I32, 0xffff_ffff);
                b.int(block, IntOp::And, x, ones)
            }
            4 => {
                let zero = b.constant(block, Ty::I32, 0);
                b.int(block, IntOp::Xor, x, zero)
            }
            5 => {
                let float = b.core(block, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, x));
                b.core(block, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, float))
            }
            6 => {
                let pair = b.core(block, Ty::I64, Op::Pack64(x, y));
                let thirty_two = b.constant(block, Ty::I64, 32);
                let down = b.int(block, IntOp::LShr, pair, thirty_two);
                b.core(block, Ty::I32, Op::UnpackLo(down))
            }
            8 => b.int(block, IntOp::Add, x, x),
            9 => {
                let two = b.constant(block, Ty::I32, 2);
                b.int(block, IntOp::Mul, x, two)
            }
            10 => {
                let one = b.constant(block, Ty::I32, 1);
                b.int(block, IntOp::Shl, x, one)
            }
            11 => {
                let one = b.constant(block, Ty::I32, 1);
                let up = b.int(block, IntOp::Add, x, one);
                b.int(block, IntOp::Sub, up, one)
            }
            12 => {
                let three = b.constant(block, Ty::I32, 3);
                b.int(block, IntOp::Mul, x, three)
            }
            13 => {
                let two = b.constant(block, Ty::I32, 2);
                b.int(block, IntOp::Shl, x, two)
            }
            14 => b.int(block, IntOp::Mul, x, y),
            15 => b.int(block, IntOp::Mul, y, x),
            16 => b.int(block, IntOp::Mul, x, x),
            _ => y,
        };
        b.br(block, join, vec![p[0], p[1], v]);
    }
    b.store(join, Space::Global, MemSize::B32, j[1], j[2], j[0]);
    (b, q)
}

#[test]
fn prove_converts_a_query_whose_arms_compute_one_word_in_different_ways() {
    let cases = [
        ("lo(pack(x, y)) and x", 1, 0),
        ("x & ~0 and x", 3, 0),
        ("x ^ 0 and x", 4, 0),
        ("bitcast round trip and x", 5, 0),
        ("hi(pack(x, y)) and y", 2, 7),
        ("lo(pack(x, y) >> 32) and y", 6, 7),
        ("x + x and x * 2", 8, 9),
        ("x << 1 and x * 2", 10, 9),
        ("(x + 1) - 1 and x", 11, 0),
    ];
    let kept: Vec<(&str, Vec<&str>)> = cases
        .iter()
        .map(|&(name, a, b)| (name, converted(&arms_compute(a, b).0)))
        .filter(|(_, names)| !names.is_empty())
        .collect();
    assert!(kept.is_empty(), "{:?}: both arms hand the store the same word", kept);
}

#[test]
fn prove_keeps_a_query_whose_arms_multiply_different_words() {
    let (b, _) = arms_compute(14, 16);
    assert!(keeps(&b).is_empty(), "{:?}: one arm stores x * y and the other x * x", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_arms_multiply_two_words_in_either_order() {
    let (b, _) = arms_compute(14, 15);
    assert!(converted(&b).is_empty(), "{:?}: x * y and y * x are one word", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_arms_compute_different_words() {
    let cases = [
        ("hi(pack(x, y)) and x", 2, 0),
        ("lo(pack(x, y)) and y", 1, 7),
        ("lo(pack(x, y) >> 32) and x", 6, 0),
        ("x and y", 0, 7),
        ("x + x and x * 3", 8, 12),
        ("x << 1 and x << 2", 10, 13),
        ("(x + 1) - 1 and y", 11, 7),
    ];
    let wrong: Vec<(&str, Vec<&str>)> = cases
        .iter()
        .map(|&(name, a, b)| {
            let (program, q) = arms_compute(a, b);
            let names: Vec<&str> = ["search", "direct"]
                .iter()
                .zip(both(&program))
                .filter(|(_, kept)| !kept.queries.contains(&q))
                .map(|(n, _)| *n)
                .collect();
            (name, names)
        })
        .filter(|(_, names)| !names.is_empty())
        .collect();
    assert!(wrong.is_empty(), "{:?}: the arms hand the store the lane id and the lane id + 1", wrong);
}

#[test]
fn search_keeps_a_query_that_another_lane_reads_through_a_lane_exchange() {
    let converted: Vec<Reader> = [Reader::ReadLane, Reader::ReadFirstLane, Reader::WriteLane, Reader::Bpermute, Reader::BpermuteFi, Reader::Wmma]
        .iter()
        .copied()
        .filter(|&reader| {
            let (b, q) = reads_another_lane(reader);
            let (kept, _) = search::prove(&b.f, &b.inputs, Some(0), &no_hazards());
            !kept.queries.contains(&q)
        })
        .collect();
    assert!(
        converted.is_empty(),
        "{:?}: when lane 0 alone has a zero flag, lane 0 answers the query false and hands the storing lanes 0 instead of the pair of halves 1.0",
        converted
    );
}

#[test]
fn direct_keeps_a_query_that_another_lane_reads_through_a_lane_exchange() {
    let converted: Vec<Reader> = [Reader::ReadLane, Reader::ReadFirstLane, Reader::WriteLane, Reader::Bpermute, Reader::BpermuteFi, Reader::Wmma]
        .iter()
        .copied()
        .filter(|&reader| {
            let (b, q) = reads_another_lane(reader);
            let (kept, _) = direct::prove(&b.f, &b.inputs, Some(0), &no_hazards());
            !kept.queries.contains(&q)
        })
        .collect();
    assert!(
        converted.is_empty(),
        "{:?}: when lane 0 alone has a zero flag, lane 0 answers the query false and hands the storing lanes 0 instead of the pair of halves 1.0",
        converted
    );
}

#[test]
fn prove_keeps_a_query_whose_word_lanes_outside_exec_hand_to_a_read_that_ignores_exec() {
    let converted: Vec<(Reader, &str)> = [Reader::ReadLane, Reader::BpermuteFi, Reader::Wmma]
        .iter()
        .flat_map(|&reader| {
            let (b, q) = reads_outside_exec(reader);
            ["search", "direct"]
                .iter()
                .zip(both(&b))
                .filter(|(_, kept)| !kept.queries.contains(&q))
                .map(|(name, _)| (reader, *name))
                .collect::<Vec<_>>()
        })
        .collect();
    assert!(
        converted.is_empty(),
        "{:?}: when lane 0 has exec clear and lane 1 the flag, lane 0 answers the query false and hands the storing lanes 0 instead of the pair of halves 1.0",
        converted
    );
}

#[test]
fn prove_converts_a_query_whose_word_only_lanes_a_masked_read_skips_hold() {
    let kept: Vec<(Reader, Vec<&str>)> = [Reader::ReadFirstLane, Reader::Bpermute]
        .iter()
        .map(|&reader| (reader, converted(&reads_outside_exec(reader).0)))
        .filter(|(_, names)| !names.is_empty())
        .collect();
    assert!(
        kept.is_empty(),
        "{:?}: only lanes with exec clear hold a word the query picks, and these reads take no such lane's word while a lane stores",
        kept
    );
}

fn lane_read(selector: impl Fn(&mut Build, BlockId, &Kernel) -> ValueId) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let zero = b.constant(e, Ty::I32, 0);
    let first = b.cmp(e, IntPred::Eq, lane, zero);
    let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
    let picked = b.core(e, Ty::I32, Op::Select(q, ones, lane));
    let x = b.core(e, Ty::I32, Op::Select(first, lane, picked));
    let selector = selector(&mut b, e, &k);
    let y = b.wave(e, WaveOp::ReadLane, vec![x, selector, zero]);
    store_own(&mut b, &k, e, y, k.exec);
    b
}

#[test]
fn prove_converts_a_query_whose_word_only_lanes_a_lane_read_skips_hold() {
    let b = lane_read(|b, e, _| b.constant(e, Ty::I32, 32));
    assert!(converted(&b).is_empty(), "{:?}: the read takes lane 32 & 31 = 0, whose word is its lane id 0 whatever the query answers", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_word_the_lane_a_lane_read_takes_holds() {
    let b = lane_read(|b, e, _| b.constant(e, Ty::I32, 1));
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: lane 1's word is 0x3c003c00 or 1 by the query", wrong);
}

#[test]
fn prove_keeps_a_query_whose_word_one_of_the_lanes_a_lane_read_may_take_holds() {
    let b = lane_read(|b, e, k| {
        let flag = per_lane(b, k, e, 24);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let one = b.constant(e, Ty::I32, 1);
        b.core(e, Ty::I32, Op::Select(set, zero, one))
    });
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: lanes without the flag read lane 1, whose word depends on the query", wrong);
}

fn accumulator_query(stored: usize) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let fzero = b.constant(e, Ty::F32, 0);
    let fone = b.constant(e, Ty::F32, 0x3f80_0000);
    let second = b.core(e, Ty::F32, Op::Select(q, fone, fzero));
    let mut inputs = vec![lane; 8];
    inputs.extend([fzero, second]);
    inputs.extend([fzero; 6]);
    let outputs = b.effect(e, EffectOp::Wave(WaveOp::Wmma), inputs);
    let word = b.core(e, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, outputs[stored]));
    store_own(&mut b, &k, e, word, k.exec);
    b
}

#[test]
fn prove_converts_a_query_whose_word_only_an_accumulator_the_stored_output_skips_holds() {
    let b = accumulator_query(0);
    assert!(converted(&b).is_empty(), "{:?}: output 0 adds the products to accumulator 0 alone, and only accumulator 1 depends on the query", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_word_the_accumulator_of_the_stored_output_holds() {
    let b = accumulator_query(1);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| kept.queries.is_empty()).map(|(n, _)| *n).collect();
    assert!(wrong.is_empty(), "{:?}: output 1 starts from accumulator 1, which is 1 or 0 by the query", wrong);
}

#[test]
fn prove_keeps_a_query_that_lane_zero_hands_to_a_first_lane_read_over_an_empty_mask() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let flag = per_lane(&mut b, &k, e, 16);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let mask = b.int(e, IntOp::And, set, k.exec);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let first = b.cmp(e, IntPred::Eq, lane, zero);
    let ones = b.constant(e, Ty::I32, 0x3c00_3c00);
    let picked = b.core(e, Ty::I32, Op::Select(q, ones, lane));
    let idle = b.core(e, Ty::I32, Op::Select(first, picked, lane));
    let x = b.core(e, Ty::I32, Op::Select(mask, lane, idle));
    let y = b.wave(e, WaveOp::ReadFirstLane, vec![x, mask]);
    let yes = b.constant(e, Ty::I1, 1);
    let clear = b.int(e, IntOp::Xor, mask, yes);
    let others = b.int(e, IntOp::Xor, first, yes);
    let stores = b.int(e, IntOp::And, clear, others);
    let stores = b.int(e, IntOp::And, stores, k.exec);
    store_own(&mut b, &k, e, y, stores);
    let wrong: Vec<&str> = ["search", "direct"].iter().zip(both(&b)).filter(|(_, kept)| !kept.queries.contains(&q)).map(|(n, _)| *n).collect();
    assert!(
        wrong.is_empty(),
        "{:?}: when no lane sets the mask, the read takes lane 0's word, and with the flag clear in lane 0 and set in lane 1, lane 0 answers the query false and hands lane 1 its lane id 0 instead of the pair of halves 1.0",
        wrong
    );
}

fn orders_in_two_blocks_of_kinds(second: IntPred) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let v = per_lane(&mut b, &k, e, 16);
    let w = per_lane(&mut b, &k, e, 24);
    let first = b.cmp(e, IntPred::Ult, v, w);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (next, p) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32, Ty::I1]);
    b.br(e, next, vec![k.exec, own, data, v, w, first]);
    let later = b.cmp(next, second, p[4], p[3]);
    let both = b.int(next, IntOp::And, p[5], later);
    let mask = b.int(next, IntOp::And, both, p[0]);
    b.store(next, Space::Global, MemSize::B32, p[1], p[2], mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_an_order_and_its_swap_in_the_next_block_mask() {
    let b = orders_in_two_blocks_of_kinds(IntPred::Ugt);
    assert!(keeps(&b).is_empty(), "{:?}: v < w is w > v, so lanes with v < w store", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_an_order_and_an_equality_in_the_next_block_mask() {
    let b = orders_in_two_blocks_of_kinds(IntPred::Eq);
    assert!(converted(&b).is_empty(), "{:?}: v < w in the first block rules out w == v in the next", converted(&b));
}

fn orders_of_both_signs(bounded: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let x = per_lane(&mut b, &k, e, 16);
    let y = per_lane(&mut b, &k, e, 24);
    let half = b.constant(e, Ty::I32, 1 << 31);
    let signed = b.cmp(e, IntPred::Slt, x, y);
    let unsigned = b.cmp(e, IntPred::Ult, y, x);
    let y_small = b.cmp(e, IntPred::Ult, y, half);
    let orders = b.int(e, IntOp::And, signed, unsigned);
    let mut all = b.int(e, IntOp::And, orders, y_small);
    if bounded {
        let x_small = b.cmp(e, IntPred::Ult, x, half);
        all = b.int(e, IntOp::And, all, x_small);
    }
    let mask = b.int(e, IntOp::And, all, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_opposite_orders_of_both_signs_mask_for_a_negative_word() {
    let b = orders_of_both_signs(false);
    assert!(keeps(&b).is_empty(), "{:?}: x = -1 and y = 0 give x s< y and y u< x", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_opposite_orders_of_both_signs_mask_for_two_small_words() {
    let b = orders_of_both_signs(true);
    assert!(converted(&b).is_empty(), "{:?}: below 2^31 the signed and unsigned orders agree, so x s< y and y u< x never hold together", converted(&b));
}

fn float_orders(opposite: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let v = per_lane(&mut b, &k, e, 16);
    let w = per_lane(&mut b, &k, e, 24);
    let x = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, v));
    let y = b.core(e, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, w));
    let below = b.core(e, Ty::I1, Op::FCmp(FloatPred::Olt, x, y));
    let other = if opposite { b.core(e, Ty::I1, Op::FCmp(FloatPred::Olt, y, x)) } else { b.core(e, Ty::I1, Op::FCmp(FloatPred::Ogt, y, x)) };
    let both = b.int(e, IntOp::And, below, other);
    let mask = b.int(e, IntOp::And, both, k.exec);
    store_own(&mut b, &k, e, data, mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_a_float_order_and_its_swap_mask() {
    let b = float_orders(false);
    assert!(keeps(&b).is_empty(), "{:?}: x < y is y > x, so lanes with x < y store", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_opposite_float_orders_mask() {
    let b = float_orders(true);
    assert!(converted(&b).is_empty(), "{:?}: x < y and y < x never hold together", converted(&b));
}

fn bound_carried_through_a_sum(limit: u64) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let table = k.buffer(&mut b, e, 16);
    let yes = b.constant(e, Ty::I1, 1);
    let x = b.load(e, Space::Global, MemSize::B32, table, yes);
    let five = b.constant(e, Ty::I32, 5);
    let enters = b.cmp(e, IntPred::Ult, x, five);
    let one = b.constant(e, Ty::I32, 1);
    let next = b.int(e, IntOp::Add, x, one);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (then, t) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
    let (exit, _) = b.block(&[Ty::I1]);
    b.cond_br(e, enters, (then, vec![k.exec, next, data, own]), (exit, vec![k.exec]));
    let limit = b.constant(then, Ty::I32, limit);
    let test = b.cmp(then, IntPred::Eq, t[1], limit);
    let mask = b.int(then, IntOp::And, test, t[0]);
    b.store(then, Space::Global, MemSize::B32, t[3], t[2], mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_store_a_bound_carried_through_a_sum_lets_through() {
    let b = bound_carried_through_a_sum(3);
    assert!(keeps(&b).is_empty(), "{:?}: x = 2 is below 5 and x + 1 is 3", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_store_a_bound_carried_through_a_sum_rules_out() {
    let b = bound_carried_through_a_sum(7);
    assert!(converted(&b).is_empty(), "{:?}: x < 5 in the first block, so the x + 1 the block receives is never 7", converted(&b));
}

fn over_active_products_carried(masked: bool, every: bool) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let sixteen = b.constant(e, Ty::I32, 16);
    let low = b.cmp(e, IntPred::Ult, lane, sixteen);
    let exec = b.int(e, IntOp::And, low, k.exec);
    let flags = k.buffer(&mut b, e, 8);
    let buf = k.buffer(&mut b, e, 0);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    let (next, n) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
    b.br(e, then, vec![exec, flags, buf]);
    let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, then, t[1], lane, 4);
    let yes = b.constant(then, Ty::I1, 1);
    let flag = b.load(then, Space::Global, MemSize::B32, own, yes);
    let factor = b.core(then, Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, if masked { t[0] } else { yes }));
    let v = b.int(then, IntOp::Mul, flag, factor);
    b.br(then, next, vec![t[0], t[2], v]);
    let zero = b.constant(next, Ty::I32, 0);
    let set = b.cmp(next, IntPred::Ne, n[2], zero);
    let at = b.here(next);
    let q = b.wave(next, WaveOp::Any, vec![set]);
    let one = b.constant(next, Ty::I32, 1);
    let two = b.constant(next, Ty::I32, 2);
    let data = b.core(next, Ty::I32, Op::Select(q, one, two));
    let lane = b.core(next, Ty::I32, Op::Env(Env::LaneId));
    let out = byte_offset(&mut b, next, n[1], lane, 4);
    b.store(next, Space::Global, MemSize::B32, out, data, n[0]);
    let Inst::Effect { provenance, .. } = b.f.blocks[&next].insts[at.1] else { unreachable!() };
    for (name, prove) in [("search", search::prove as fn(&Func, &[Parameter], Option<usize>, &Hazards) -> (Kept, BTreeSet<u64>)), ("direct", direct::prove)] {
        let (kept, everyone) = prove(&b.f, &b.inputs, Some(0), &no_hazards());
        assert!(kept.queries.contains(&q), "{}: a lane whose flag is set stores 1, the others 2", name);
        assert_eq!(everyone.contains(&provenance), every, "{}: a lane with exec clear carries its flag times {}", name, !masked as u32);
    }
}

#[test]
fn prove_demands_no_lane_for_a_kept_query_over_products_with_the_exec_bit_carried_into_the_next_block() {
    over_active_products_carried(true, false);
}

#[test]
fn prove_demands_every_lane_for_a_kept_query_over_products_with_one_carried_into_the_next_block() {
    over_active_products_carried(false, true);
}

fn permuted_from_a_sum(odd: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let data = query_data(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let zero = b.constant(e, Ty::I32, 0);
    let first = b.cmp(e, IntPred::Eq, lane, zero);
    let seven = b.constant(e, Ty::I32, 7);
    let x = b.core(e, Ty::I32, Op::Select(first, data, seven));
    let one = b.constant(e, Ty::I32, 1);
    let doubled = b.int(e, IntOp::Shl, lane, one);
    let u = uniform_load(&mut b, &k, e, 16);
    let bit = b.int(e, IntOp::And, u, one);
    let two = b.constant(e, Ty::I32, 2);
    let far = b.int(e, IntOp::Shl, bit, two);
    let base = if odd { b.int(e, IntOp::Add, doubled, one) } else { doubled };
    let source = b.int(e, IntOp::Add, base, far);
    let four = b.constant(e, Ty::I32, 4);
    let index = b.int(e, IntOp::Mul, source, four);
    let yes = b.constant(e, Ty::I1, 1);
    let read = b.wave(e, WaveOp::Bpermute, vec![index, x, yes]);
    store_own(&mut b, &k, e, read, k.exec);
    b
}

#[test]
fn prove_keeps_a_query_whose_word_a_permute_of_doubled_lanes_fetches_from_lane_zero() {
    let b = permuted_from_a_sum(false);
    assert!(keeps(&b).is_empty(), "{:?}: when u is even lanes 0 and 16 fetch lane 0's word, which the query picks", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_word_a_permute_of_doubled_lanes_plus_one_never_fetches() {
    let b = permuted_from_a_sum(true);
    assert!(converted(&b).is_empty(), "{:?}: 2 lane + 1 + 4 (u & 1) is odd, so every lane fetches from an odd lane, whose word is 7", converted(&b));
}

fn written_to_a_loaded_lane(skip: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let x = b.core(e, Ty::I32, Op::Select(q, one, two));
    let u = uniform_load(&mut b, &k, e, 16);
    let thirty_one = b.constant(e, Ty::I32, 31);
    let target = b.int(e, IntOp::And, u, thirty_one);
    let y = b.wave(e, WaveOp::WriteLane, vec![x, target, lane]);
    let mask = if skip {
        let other = b.cmp(e, IntPred::Ne, lane, target);
        b.int(e, IntOp::And, other, k.exec)
    } else {
        k.exec
    };
    store_own(&mut b, &k, e, y, mask);
    b
}

#[test]
fn prove_keeps_a_query_whose_word_a_lane_write_puts_into_a_loaded_lane_that_stores() {
    let b = written_to_a_loaded_lane(false);
    assert!(keeps(&b).is_empty(), "{:?}: lane u & 31 stores lane 0's word, whose answer is lane 0's own flag", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_word_a_lane_write_puts_only_into_a_loaded_lane_that_never_stores() {
    let b = written_to_a_loaded_lane(true);
    assert!(converted(&b).is_empty(), "{:?}: every storing lane keeps its own lane id", converted(&b));
}

enum Written {
    OnlyTarget,
    SkipNext,
    SkipWide,
    Below,
    AtMost,
    Above,
    AtLeast,
}

fn lane_written_under(shape: Written) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let x = b.core(e, Ty::I32, Op::Select(q, one, two));
    let u = uniform_load(&mut b, &k, e, 16);
    let thirty_one = b.constant(e, Ty::I32, 31);
    let target = match shape {
        Written::SkipWide => u,
        _ => b.int(e, IntOp::And, u, thirty_one),
    };
    let y = b.wave(e, WaveOp::WriteLane, vec![x, target, lane]);
    let test = match shape {
        Written::OnlyTarget => b.cmp(e, IntPred::Eq, lane, target),
        Written::SkipNext => {
            let next = b.int(e, IntOp::Add, target, one);
            b.cmp(e, IntPred::Ne, lane, next)
        }
        Written::SkipWide => b.cmp(e, IntPred::Ne, target, lane),
        Written::Below => b.cmp(e, IntPred::Ult, lane, target),
        Written::AtMost => b.cmp(e, IntPred::Ule, lane, target),
        Written::Above => b.cmp(e, IntPred::Ugt, lane, target),
        Written::AtLeast => b.cmp(e, IntPred::Uge, lane, target),
    };
    let mask = b.int(e, IntOp::And, test, k.exec);
    store_own(&mut b, &k, e, y, mask);
    b
}

#[test]
fn prove_converts_a_query_whose_word_a_lane_write_puts_only_into_a_loaded_lane_above_the_storing_lanes() {
    let converted: Vec<(&str, Vec<&str>)> = vec![("below", Written::Below), ("above", Written::Above)]
        .into_iter()
        .map(|(name, shape)| (name, converted(&lane_written_under(shape))))
        .filter(|(_, names)| !names.is_empty())
        .collect();
    assert!(converted.is_empty(), "{:?}: lane u & 31 is not below or above itself, so it never stores", converted);
}

#[test]
fn prove_keeps_a_query_whose_word_a_lane_write_puts_into_a_loaded_lane_among_the_storing_lanes() {
    let kept: Vec<(&str, Vec<&str>)> = vec![("at most", Written::AtMost), ("at least", Written::AtLeast)]
        .into_iter()
        .map(|(name, shape)| (name, keeps(&lane_written_under(shape))))
        .filter(|(_, names)| !names.is_empty())
        .collect();
    assert!(kept.is_empty(), "{:?}: lane u & 31 stores the written word", kept);
}

#[test]
fn prove_keeps_a_query_whose_word_a_lane_write_puts_into_the_only_lane_that_stores() {
    let b = lane_written_under(Written::OnlyTarget);
    assert!(keeps(&b).is_empty(), "{:?}: only lane u & 31 stores, and it stores the written word", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_lane_write_target_differs_from_the_lane_the_store_skips() {
    let b = lane_written_under(Written::SkipNext);
    assert!(keeps(&b).is_empty(), "{:?}: the store skips lane u & 31 + 1, so lane u & 31 stores the written word", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_lane_write_target_may_pass_the_last_lane() {
    let b = lane_written_under(Written::SkipWide);
    assert!(keeps(&b).is_empty(), "{:?}: u = 33 writes lane 1, while the store skips no lane", keeps(&b));
}

fn offset_lanes_below(shared: bool) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let table = k.buffer(&mut b, e, 16);
    let yes = b.constant(e, Ty::I1, 1);
    let u = b.load(e, Space::Global, MemSize::U8, table, yes);
    let shifted = b.int(e, IntOp::Add, lane, u);
    let five = b.constant(e, Ty::I32, 5);
    let other = if shared { b.int(e, IntOp::Add, lane, five) } else { five };
    let t = b.cmp(e, IntPred::Ult, shifted, other);
    let c = b.int(e, IntOp::And, t, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let data = b.core(e, Ty::I32, Op::Select(q, one, two));
    store_own(&mut b, &k, e, data, k.exec);
    b
}

#[test]
fn prove_keeps_a_query_whose_bit_lanes_offset_by_a_byte_order_differently() {
    let b = offset_lanes_below(false);
    assert!(keeps(&b).is_empty(), "{:?}: lane + u < 5 holds in the lanes below 5 - u alone", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_bit_every_lane_offset_by_a_byte_orders_alike() {
    let b = offset_lanes_below(true);
    assert!(converted(&b).is_empty(), "{:?}: lane + u < lane + 5 is u < 5 in every lane, as neither side wraps", converted(&b));
}

fn apart_loop_pair(opposite: bool) -> Build {
    looped_pair(opposite, Flips::Both)
}

enum Flips {
    Both,
    OnlyFirst,
    SecondOnALoadedBit,
}

fn looped_pair(opposite: bool, flips: Flips) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let table = k.buffer(&mut b, e, 16);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let everywhere = b.constant(e, Ty::I1, 1);
    let word = b.load(e, Space::Global, MemSize::B32, own, everywhere);
    let nothing = b.constant(e, Ty::I32, 0);
    let d = b.cmp(e, IntPred::Ne, word, nothing);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let flag = per_lane(&mut b, &k, e, 28);
    let zero = b.constant(e, Ty::I32, 0);
    let c = b.cmp(e, IntPred::Ne, flag, zero);
    let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I32, Ty::I1]);
    let (after, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (arm, vec![k.exec, own, c, d]), (join, vec![k.exec]));
    let yes = b.constant(arm, Ty::I1, 1);
    let not_c = b.int(arm, IntOp::Xor, a[2], yes);
    let zero = b.constant(arm, Ty::I32, 0);
    let second = if opposite { not_c } else { a[2] };
    b.br(arm, body, vec![a[0], a[1], a[2], second, zero, a[3]]);
    let yes = b.constant(body, Ty::I1, 1);
    let s = b.int(body, IntOp::Xor, p[2], yes);
    let t = match flips {
        Flips::Both => b.int(body, IntOp::Xor, p[3], yes),
        Flips::OnlyFirst => p[3],
        Flips::SecondOnALoadedBit => b.int(body, IntOp::Xor, p[3], p[5]),
    };
    let one = b.constant(body, Ty::I32, 1);
    let next = b.int(body, IntOp::Add, p[4], one);
    let three = b.constant(body, Ty::I32, 3);
    let again = b.cmp(body, IntPred::Ult, next, three);
    b.cond_br(body, again, (body, vec![p[0], p[1], s, t, next, p[5]]), (after, vec![p[0], p[1], s, t]));
    let yes = b.constant(after, Ty::I1, 1);
    let not_t = b.int(after, IntOp::Xor, m[3], yes);
    let only = b.int(after, IntOp::And, m[2], not_t);
    let mask = b.int(after, IntOp::And, only, m[0]);
    let one = b.constant(after, Ty::I32, 1);
    b.store(after, Space::Global, MemSize::B32, m[1], one, mask);
    b.br(after, join, vec![m[0]]);
    b
}

#[test]
fn prove_keeps_a_query_whose_arm_loops_two_bits_that_stay_opposite() {
    let b = apart_loop_pair(true);
    assert!(keeps(&b).is_empty(), "{:?}: the loop carries (c, !c) flipped three times, so s & !t is !c and lanes without c store", keeps(&b));
}

fn while_pair(flips: Flips) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let table = k.buffer(&mut b, e, 16);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let slot = byte_offset(&mut b, e, table, lane, 4);
    let everywhere = b.constant(e, Ty::I1, 1);
    let word = b.load(e, Space::Global, MemSize::B32, slot, everywhere);
    let nothing = b.constant(e, Ty::I32, 0);
    let d = b.cmp(e, IntPred::Ne, word, nothing);
    let buf = k.buffer(&mut b, e, 0);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let flag = per_lane(&mut b, &k, e, 28);
    let c = b.cmp(e, IntPred::Ne, flag, nothing);
    let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (header, h) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I32, Ty::I1]);
    let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I32, Ty::I1]);
    let (after, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (arm, vec![k.exec, own, c, d]), (join, vec![k.exec]));
    let zero = b.constant(arm, Ty::I32, 0);
    b.br(arm, header, vec![a[0], a[1], a[2], a[2], zero, a[3]]);
    let three = b.constant(header, Ty::I32, 3);
    let again = b.cmp(header, IntPred::Ult, h[4], three);
    b.cond_br(header, again, (body, h.clone()), (after, vec![h[0], h[1], h[2], h[3]]));
    let yes = b.constant(body, Ty::I1, 1);
    let s = b.int(body, IntOp::Xor, p[2], yes);
    let t = match flips {
        Flips::Both => b.int(body, IntOp::Xor, p[3], yes),
        Flips::OnlyFirst => p[3],
        Flips::SecondOnALoadedBit => b.int(body, IntOp::Xor, p[3], p[5]),
    };
    let one = b.constant(body, Ty::I32, 1);
    let next = b.int(body, IntOp::Add, p[4], one);
    b.br(body, header, vec![p[0], p[1], s, t, next, p[5]]);
    let yes = b.constant(after, Ty::I1, 1);
    let not_t = b.int(after, IntOp::Xor, m[3], yes);
    let only = b.int(after, IntOp::And, m[2], not_t);
    let mask = b.int(after, IntOp::And, only, m[0]);
    let one = b.constant(after, Ty::I32, 1);
    b.store(after, Space::Global, MemSize::B32, m[1], one, mask);
    b.br(after, join, vec![m[0]]);
    b
}

#[test]
fn prove_keeps_a_query_whose_arm_loop_stores_one_iteration_after_its_counter_passes_one() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (arm, a) = b.block(&[Ty::I1, Ty::I64]);
    let (header, h) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I32]);
    let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I32]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (arm, vec![k.exec, own]), (join, vec![k.exec]));
    let no = b.constant(arm, Ty::I1, 0);
    let zero = b.constant(arm, Ty::I32, 0);
    b.br(arm, header, vec![a[0], a[1], no, zero]);
    let three = b.constant(header, Ty::I32, 3);
    let again = b.cmp(header, IntPred::Ult, h[3], three);
    b.cond_br(header, again, (body, h.clone()), (join, vec![h[0]]));
    let one = b.constant(body, Ty::I32, 1);
    let t = b.cmp(body, IntPred::Eq, p[3], one);
    let yes = b.constant(body, Ty::I1, 1);
    let not_t = b.int(body, IntOp::Xor, t, yes);
    let fresh = b.int(body, IntOp::And, p[2], not_t);
    let mask = b.int(body, IntOp::And, fresh, p[0]);
    let seven = b.constant(body, Ty::I32, 7);
    b.store(body, Space::Global, MemSize::B32, p[1], seven, mask);
    let next = b.int(body, IntOp::Add, p[3], one);
    b.br(body, header, vec![p[0], p[1], t, next]);
    assert!(keeps(&b).is_empty(), "{:?}: in the pass with counter 2 the bit carried from counter 1 is set while the counter is not 1, so the arm stores", keeps(&b));
}

enum Related {
    Stale,
    Loose,
    Unfixed,
}

fn related_pair(shape: Related) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let q = flag_query(&mut b, &k, e);
    let everywhere = b.constant(e, Ty::I1, 1);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let zero = b.constant(e, Ty::I32, 0);
    let bit_at = |b: &mut Build, offset: u64| {
        let table = k.buffer(b, e, offset);
        let slot = byte_offset(b, e, table, lane, 4);
        let word = b.load(e, Space::Global, MemSize::B32, slot, everywhere);
        b.cmp(e, IntPred::Ne, word, zero)
    };
    let c = bit_at(&mut b, 16);
    let d = bit_at(&mut b, 24);
    let buf = k.buffer(&mut b, e, 0);
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let (arm, a) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1]);
    let (header, h) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I32, Ty::I1]);
    let (body, p) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I32, Ty::I1]);
    let (after, m) = b.block(&[Ty::I1, Ty::I64, Ty::I1, Ty::I1, Ty::I1]);
    let (join, _) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (arm, vec![k.exec, own, c, d]), (join, vec![k.exec]));
    let parity = matches!(shape, Related::Unfixed);
    let start = match shape {
        Related::Loose | Related::Unfixed => a[2],
        Related::Stale => b.int(arm, IntOp::And, a[2], a[3]),
    };
    let other = if parity { b.constant(arm, Ty::I1, 0) } else { a[3] };
    let none = b.constant(arm, Ty::I32, 0);
    b.br(arm, header, vec![a[0], a[1], start, other, none, a[2]]);
    let three = b.constant(header, Ty::I32, 3);
    let again = b.cmp(header, IntPred::Ult, h[4], three);
    b.cond_br(header, again, (body, h.clone()), (after, vec![h[0], h[1], h[2], h[3], h[5]]));
    let yes = b.constant(body, Ty::I1, 1);
    let flipped = b.int(body, IntOp::Xor, p[3], yes);
    let s = match shape {
        Related::Stale => b.int(body, IntOp::And, p[5], p[3]),
        Related::Unfixed => b.int(body, IntOp::Xor, p[2], yes),
        Related::Loose => b.int(body, IntOp::And, p[5], flipped),
    };
    let one = b.constant(body, Ty::I32, 1);
    let next = b.int(body, IntOp::Add, p[4], one);
    b.br(body, header, vec![p[0], p[1], s, flipped, next, p[5]]);
    let yes = b.constant(after, Ty::I1, 1);
    let only = if parity {
        b.int(after, IntOp::Xor, m[2], m[3])
    } else {
        let not_t = b.int(after, IntOp::Xor, m[3], yes);
        b.int(after, IntOp::And, m[2], not_t)
    };
    let mask = b.int(after, IntOp::And, only, m[0]);
    let one = b.constant(after, Ty::I32, 1);
    b.store(after, Space::Global, MemSize::B32, m[1], one, mask);
    b.br(after, join, vec![m[0]]);
    b
}

#[test]
fn prove_keeps_a_query_whose_while_loop_stores_a_bit_xor_a_parity() {
    let b = related_pair(Related::Unfixed);
    assert!(keeps(&b).is_empty(), "{:?}: s ^ p is c, which some lanes hold", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_while_loop_ands_a_fixed_bit_with_the_other_bit_before_it_flips() {
    let b = related_pair(Related::Stale);
    assert!(keeps(&b).is_empty(), "{:?}: after one pass s is c & d and t is !d, so s & !t is c & d", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_while_loop_starts_without_the_and() {
    let b = related_pair(Related::Loose);
    assert!(keeps(&b).is_empty(), "{:?}: the loop may run no pass, and c & !d then stores", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_while_loop_flips_two_bits_that_stay_equal() {
    let b = while_pair(Flips::Both);
    assert!(converted(&b).is_empty(), "{:?}: the header always holds s = t, so s & !t never holds", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_while_loop_flips_one_of_two_equal_bits_on_a_loaded_bit() {
    let b = while_pair(Flips::SecondOnALoadedBit);
    assert!(keeps(&b).is_empty(), "{:?}: after one pass s is !c and t is c ^ d, so s & !t is !c & !d", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_arm_loops_two_equal_bits_and_flips_only_one() {
    let b = looped_pair(false, Flips::OnlyFirst);
    assert!(keeps(&b).is_empty(), "{:?}: after three flips s is !c while t stays c, so lanes without c store", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_arm_loops_two_equal_bits_and_flips_one_on_a_loaded_bit() {
    let b = looped_pair(false, Flips::SecondOnALoadedBit);
    assert!(keeps(&b).is_empty(), "{:?}: t flips only where the loaded bit d is set, so after three flips s & !t is !c & !d", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_arm_loops_two_bits_that_stay_equal() {
    let b = apart_loop_pair(false);
    assert!(converted(&b).is_empty(), "{:?}: the loop flips both bits together, so s & !t never holds and the arm never stores", converted(&b));
}

fn arms_regroup(cases: &[(usize, usize)]) -> Vec<Build> {
    cases
        .iter()
        .map(|&(then_arm, other_arm)| {
            let (mut b, k) = Build::kernel();
            let e = BlockId(0);
            let q = flag_query(&mut b, &k, e);
            let buf = k.buffer(&mut b, e, 0);
            let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
            let one = b.constant(e, Ty::I32, 1);
            let two = b.constant(e, Ty::I32, 2);
            let y = b.int(e, IntOp::Add, lane, one);
            let z = b.int(e, IntOp::Add, lane, two);
            let own = byte_offset(&mut b, e, buf, lane, 4);
            let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32]);
            let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I32, Ty::I32, Ty::I32]);
            let (join, j) = b.block(&[Ty::I1, Ty::I64, Ty::I32]);
            b.cond_br(e, q, (then, vec![k.exec, own, lane, y, z]), (other, vec![k.exec, own, lane, y, z]));
            for (block, p, arm) in [(then, t, then_arm), (other, o, other_arm)] {
                let (x, y, z) = (p[2], p[3], p[4]);
                let v = match arm {
                    0 => {
                        let xy = b.int(block, IntOp::Mul, x, y);
                        b.int(block, IntOp::Mul, xy, z)
                    }
                    1 => {
                        let yz = b.int(block, IntOp::Mul, y, z);
                        b.int(block, IntOp::Mul, x, yz)
                    }
                    2 => {
                        let sum = b.int(block, IntOp::Add, x, y);
                        b.int(block, IntOp::Mul, sum, z)
                    }
                    3 => {
                        let xz = b.int(block, IntOp::Mul, x, z);
                        let yz = b.int(block, IntOp::Mul, y, z);
                        b.int(block, IntOp::Add, xz, yz)
                    }
                    4 => {
                        let xy = b.int(block, IntOp::And, x, y);
                        b.int(block, IntOp::And, xy, z)
                    }
                    5 => {
                        let yz = b.int(block, IntOp::And, y, z);
                        b.int(block, IntOp::And, x, yz)
                    }
                    _ => {
                        let xz = b.int(block, IntOp::Mul, x, z);
                        b.int(block, IntOp::Add, xz, y)
                    }
                };
                b.br(block, join, vec![p[0], p[1], v]);
            }
            b.store(join, Space::Global, MemSize::B32, j[1], j[2], j[0]);
            b
        })
        .collect()
}

#[test]
fn prove_keeps_a_query_whose_arms_compute_different_products_of_sums() {
    let wrong: Vec<Vec<&str>> = arms_regroup(&[(2, 6)]).iter().map(keeps).filter(|w| !w.is_empty()).collect();
    assert!(wrong.is_empty(), "{:?}: (x + y) z and x z + y differ", wrong);
}

#[test]
fn prove_converts_a_query_whose_arms_regroup_or_distribute_one_word() {
    let names = ["(x y) z and x (y z)", "(x + y) z and x z + y z", "(x & y) & z and x & (y & z)"];
    let kept: Vec<(&str, Vec<&str>)> = names
        .iter()
        .zip(arms_regroup(&[(0, 1), (2, 3), (4, 5)]))
        .map(|(&name, b)| (name, converted(&b)))
        .filter(|(_, names)| !names.is_empty())
        .collect();
    assert!(kept.is_empty(), "{:?}: both arms hand the store the same word", kept);
}

#[test]
fn prove_keeps_a_query_whose_arms_swap_different_words_atomically() {
    let b = arms_store(|b, block, side, p| {
        let expected = b.constant(block, Ty::I32, 0);
        let desired = b.constant(block, Ty::I32, 1 + side as u64);
        b.effect(block, memory(Space::Global, MemoryOp::AtomicCmpSwap), vec![p[1], desired, expected, p[0]]);
        None
    });
    assert!(keeps(&b).is_empty(), "{:?}: one arm swaps in 1 and the other 2", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_arms_swap_the_same_word_atomically() {
    let b = arms_store(|b, block, _, p| {
        let expected = b.constant(block, Ty::I32, 0);
        let desired = b.constant(block, Ty::I32, 1);
        b.effect(block, memory(Space::Global, MemoryOp::AtomicCmpSwap), vec![p[1], desired, expected, p[0]]);
        None
    });
    assert!(converted(&b).is_empty(), "{:?}: both arms swap 1 into the lane's own word when it holds 0", converted(&b));
}

#[test]
fn prove_keeps_a_query_whose_arms_swap_the_same_word_against_different_expected_words() {
    let b = arms_store(|b, block, side, p| {
        let expected = b.constant(block, Ty::I32, 2 * side as u64);
        let desired = b.constant(block, Ty::I32, 1);
        b.effect(block, memory(Space::Global, MemoryOp::AtomicCmpSwap), vec![p[1], desired, expected, p[0]]);
        None
    });
    assert!(keeps(&b).is_empty(), "{:?}: one arm swaps 1 in over 0 and the other over 2", keeps(&b));
}

fn arms_store_two_that_may_meet(second: u64) -> Build {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let u = uniform_load(&mut b, &k, e, 16);
    let one = b.constant(e, Ty::I32, 1);
    let bit = b.int(e, IntOp::And, u, one);
    let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, bit));
    let row = b.constant(e, Ty::I64, 128);
    let step = b.int(e, IntOp::Mul, wide, row);
    let maybe = b.int(e, IntOp::Add, own, step);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    let (join, j) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (then, vec![k.exec, own, maybe]), (other, vec![k.exec, own, maybe]));
    for (side, (block, p)) in [(then, t), (other, o)].iter().enumerate() {
        let five = b.constant(*block, Ty::I32, 5);
        let later = b.constant(*block, Ty::I32, if side == 0 { 5 } else { second });
        if side == 0 {
            b.store(*block, Space::Global, MemSize::B32, p[1], five, p[0]);
            b.store(*block, Space::Global, MemSize::B32, p[2], later, p[0]);
        } else {
            b.store(*block, Space::Global, MemSize::B32, p[2], later, p[0]);
            b.store(*block, Space::Global, MemSize::B32, p[1], five, p[0]);
        }
        b.br(*block, join, vec![p[0]]);
    }
    let out = k.buffer(&mut b, join, 24);
    let lane = b.core(join, Ty::I32, Op::Env(Env::LaneId));
    let target = byte_offset(&mut b, join, out, lane, 4);
    let one = b.constant(join, Ty::I32, 1);
    b.store(join, Space::Global, MemSize::B32, target, one, j[0]);
    b
}

#[test]
fn prove_keeps_a_query_whose_arms_store_two_values_into_two_words_that_may_meet_in_either_order() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let flag = per_lane(&mut b, &k, e, 8);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let buf = k.buffer(&mut b, e, 0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, buf, lane, 4);
    let u = uniform_load(&mut b, &k, e, 16);
    let one = b.constant(e, Ty::I32, 1);
    let bit = b.int(e, IntOp::And, u, one);
    let wide = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, bit));
    let row = b.constant(e, Ty::I64, 128);
    let step = b.int(e, IntOp::Mul, wide, row);
    let maybe = b.int(e, IntOp::Add, own, step);
    let (then, t) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    let (other, o) = b.block(&[Ty::I1, Ty::I64, Ty::I64]);
    let (join, j) = b.block(&[Ty::I1]);
    b.cond_br(e, q, (then, vec![k.exec, own, maybe]), (other, vec![k.exec, own, maybe]));
    for (side, (block, p)) in [(then, t), (other, o)].iter().enumerate() {
        let five = b.constant(*block, Ty::I32, 5);
        let six = b.constant(*block, Ty::I32, 6);
        if side == 0 {
            b.store(*block, Space::Global, MemSize::B32, p[1], five, p[0]);
            b.store(*block, Space::Global, MemSize::B32, p[2], six, p[0]);
        } else {
            b.store(*block, Space::Global, MemSize::B32, p[2], six, p[0]);
            b.store(*block, Space::Global, MemSize::B32, p[1], five, p[0]);
        }
        b.br(*block, join, vec![p[0]]);
    }
    let out = k.buffer(&mut b, join, 24);
    let lane = b.core(join, Ty::I32, Op::Env(Env::LaneId));
    let target = byte_offset(&mut b, join, out, lane, 4);
    let one = b.constant(join, Ty::I32, 1);
    b.store(join, Space::Global, MemSize::B32, target, one, j[0]);
    assert!(keeps(&b).is_empty(), "{:?}: when u is even the two words are one, which the first arm leaves 6 and the second 5", keeps(&b));
}

#[test]
fn prove_keeps_a_query_whose_arms_store_a_word_that_may_meet_another_differently() {
    let b = arms_store_two_that_may_meet(6);
    assert!(keeps(&b).is_empty(), "{:?}: one arm stores 5 and the other 6 into the second word", keeps(&b));
}

#[test]
fn prove_converts_a_query_whose_arms_store_the_same_value_into_two_words_that_may_meet_in_either_order() {
    let b = arms_store_two_that_may_meet(5);
    assert!(converted(&b).is_empty(), "{:?}: both arms leave 5 in the lane's word and in the word u & 1 rows further, whether or not they are one word", converted(&b));
}

#[test]
fn masked_tests_of_words_hold_only_where_inactive_lanes_see_zero() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let sixteen = b.constant(e, Ty::I32, 16);
    let low = b.cmp(e, IntPred::Ult, lane, sixteen);
    let exec = b.int(e, IntOp::And, low, k.exec);
    let flags = k.buffer(&mut b, e, 8);
    let (then, t) = b.block(&[Ty::I1, Ty::I64]);
    b.br(e, then, vec![exec, flags]);
    let lane = b.core(then, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, then, t[1], lane, 4);
    let yes = b.constant(then, Ty::I1, 1);
    let flag = b.load(then, Space::Global, MemSize::B32, own, yes);
    let on = b.core(then, Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, t[0]));
    let all = b.core(then, Ty::I32, Op::Convert(Cvt::SExt, Ty::I32, t[0]));
    let zero = b.constant(then, Ty::I32, 0);
    let three = b.constant(then, Ty::I32, 3);
    let product = b.int(then, IntOp::Mul, flag, on);
    let masked = b.int(then, IntOp::And, all, flag);
    let shifted = b.int(then, IntOp::Shl, product, three);
    let sum = b.int(then, IntOp::Add, product, masked);
    let plus = b.int(then, IntOp::Add, flag, on);
    let chosen = b.core(then, Ty::I32, Op::Select(t[0], flag, zero));
    let other = b.core(then, Ty::I32, Op::Select(t[0], zero, flag));
    let words = [("flag * zext(exec)", product, true), ("sext(exec) & flag", masked, true), ("(flag * zext(exec)) << 3", shifted, true), ("sum of two masked words", sum, true), ("flag + zext(exec)", plus, false), ("select(exec, flag, 0)", chosen, true), ("select(exec, 0, flag)", other, false), ("flag", flag, false)];
    let mut tests = Vec::new();
    for &(name, w, expected) in &words {
        tests.push((name, b.cmp(then, IntPred::Ne, w, zero), expected));
        tests.push((name, b.cmp(then, IntPred::Ugt, w, zero), expected));
        tests.push((name, b.cmp(then, IntPred::Eq, w, zero), false));
    }
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
    let mut check = Check::new(f, &facts, &b.inputs, Some(0), &loops, &hazards, logic);
    assert!(check.run());
    let wrong: Vec<String> = tests
        .iter()
        .filter(|&&(_, v, expected)| check.differences.masks().masked(v) != expected)
        .map(|&(name, v, expected)| format!("{:?} over {}: masked {}, expected {}", f.types[v.0], name, check.differences.masks().masked(v), expected))
        .collect();
    assert!(wrong.is_empty(), "{:?}", wrong);
}

type Formula = Box<dyn Fn(&mut Logic, &dyn Fn(ValueId) -> Bdd) -> Bdd>;

struct Case {
    name: &'static str,
    value: ValueId,
    truth: Formula,
}

fn check_differences(b: &Build, kept: &[Choice], cases: &[Case], exact: bool) -> Vec<String> {
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
    let kept: BTreeSet<Choice> = kept.iter().copied().collect();
    let logic = Logic::fixed(f, &facts, &kept, &[]);
    let mut check = Check::new(f, &facts, &b.inputs, Some(0), &loops, &hazards, logic);
    assert!(check.run(), "the program has no store, so nothing can be violated");
    let mut wrong = Vec::new();
    for case in cases {
        let h = if facts.lane_word[case.value.0] && facts.materialized[case.value.0] {
            check.differences.word(case.value)
        } else {
            check.differences.h(case.value)
        };
        let logic = check.logic();
        let truth = {
            let mut atoms: HashMap<ValueId, Bdd> = HashMap::default();
            for v in 0..f.types.len() {
                if f.types[v] == Ty::I1 {
                    let bit = match (facts.op(f, ValueId(v)), facts.inst(f, ValueId(v))) {
                        (Some(Op::Cmp(..)), _) => logic.bit(f, &facts, ValueId(v)),
                        (_, Some(Inst::Effect { op: EffectOp::Wave(WaveOp::Any), .. })) => logic.wave_answer(f, &facts, ValueId(v)),
                        _ => logic.atom(Atom::Bit(ValueId(v))),
                    };
                    atoms.insert(ValueId(v), bit);
                }
            }
            (case.truth)(logic, &|v| atoms[&v])
        };
        let (h, truth) = (logic.consistent(h), logic.consistent(truth));
        let ok = if exact { logic.m.implies(h, truth) } else { logic.m.implies(truth, h) };
        if !ok {
            wrong.push(case.name.to_string());
        }
    }
    wrong
}

struct Program {
    b: Build,
    q: ValueId,
    cases: Vec<Case>,
}

fn program() -> Program {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let yes = b.constant(e, Ty::I1, 1);
    let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let other = b.load(e, Space::Global, MemSize::B32, table, yes);
    let d = b.cmp(e, IntPred::Ult, other, flag);
    let one = b.constant(e, Ty::I32, 1);
    let two = b.constant(e, Ty::I32, 2);
    let five = b.constant(e, Ty::I32, 5);
    let mut cases = Vec::new();
    let exec = k.exec;
    let differs = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
        let c = l.m.and(bit(set), bit(exec));
        let absent = l.m.not(c);
        l.m.and(absent, bit(q))
    };
    cases.push(Case { name: "any(c)", value: q, truth: Box::new(differs) });
    let v = b.core(e, Ty::I32, Op::Select(q, one, two));
    cases.push(Case { name: "select(q, 1, 2)", value: v, truth: Box::new(differs) });
    let v = b.core(e, Ty::I32, Op::Select(q, one, one));
    cases.push(Case { name: "select(q, 1, 1)", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let v = b.int(e, IntOp::And, q, d);
    cases.push(Case {
        name: "q & d",
        value: v,
        truth: Box::new(move |l, bit| {
            let x = differs(l, bit);
            l.m.and(x, bit(d))
        }),
    });
    let v = b.int(e, IntOp::Or, q, d);
    cases.push(Case {
        name: "q | d",
        value: v,
        truth: Box::new(move |l, bit| {
            let x = differs(l, bit);
            let nd = l.m.not(bit(d));
            l.m.and(x, nd)
        }),
    });
    let v = b.int(e, IntOp::Xor, q, d);
    cases.push(Case { name: "q ^ d", value: v, truth: Box::new(differs) });
    let s = b.core(e, Ty::I32, Op::Select(q, one, two));
    let v = b.cmp(e, IntPred::Ult, s, two);
    cases.push(Case { name: "select(q, 1, 2) < 2", value: v, truth: Box::new(differs) });
    let v = b.cmp(e, IntPred::Ult, s, five);
    cases.push(Case { name: "select(q, 1, 2) < 5", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let table2 = k.buffer(&mut b, e, 16);
    let address = b.core(e, Ty::I64, Op::Select(q, table, table2));
    let v = b.load(e, Space::Global, MemSize::B32, address, yes);
    cases.push(Case { name: "load from select(q, a, b)", value: v, truth: Box::new(differs) });
    let v = b.load(e, Space::Global, MemSize::B32, table, q);
    let unperformed = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
        let c = l.m.and(bit(set), bit(exec));
        l.m.not(c)
    };
    cases.push(Case { name: "load masked by q", value: v, truth: Box::new(unperformed) });
    let v = b.load(e, Space::Global, MemSize::B32, table, yes);
    cases.push(Case { name: "unmasked load of a fixed word", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let masked = b.int(e, IntOp::And, q, k.exec);
    let w = b.wave(e, WaveOp::Ballot, vec![masked]);
    let v = b.core(e, Ty::I32, Op::PopulationCount(w));
    cases.push(Case { name: "popcount(ballot(q & exec))", value: v, truth: Box::new(move |_, bit| bit(q)) });
    let kept = b.wave(e, WaveOp::Any, vec![masked]);
    cases.push(Case { name: "any(q & exec)", value: kept, truth: Box::new(differs) });
    let picked = b.core(e, Ty::I32, Op::Select(q, one, lane));
    let v = b.wave(e, WaveOp::ReadLane, vec![picked, zero, zero]);
    cases.push(Case { name: "readlane(select(q, 1, lane), 0)", value: v, truth: Box::new(move |_, bit| bit(q)) });
    let v = b.int(e, IntOp::Add, flag, one);
    cases.push(Case { name: "flag + 1", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let v = b.core(e, Ty::I32, Op::Select(d, s, s));
    cases.push(Case { name: "select(d, s, s) for s = select(q, 1, 2)", value: v, truth: Box::new(differs) });
    let v = b.int(e, IntOp::And, s, one);
    cases.push(Case { name: "select(q, 1, 2) & 1", value: v, truth: Box::new(differs) });
    let v = b.int(e, IntOp::And, s, zero);
    cases.push(Case { name: "select(q, 1, 2) & 0", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let all = b.constant(e, Ty::I32, 0xffff_ffff);
    let v = b.int(e, IntOp::Or, s, all);
    cases.push(Case { name: "select(q, 1, 2) | ~0", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let v = b.int(e, IntOp::Or, s, two);
    cases.push(Case { name: "select(q, 1, 2) | 2", value: v, truth: Box::new(differs) });
    let v = b.int(e, IntOp::Mul, s, zero);
    cases.push(Case { name: "select(q, 1, 2) * 0", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let v = b.int(e, IntOp::Mul, s, one);
    cases.push(Case { name: "select(q, 1, 2) * 1", value: v, truth: Box::new(differs) });
    Program { b, q, cases }
}

#[test]
fn differences_hold_every_state_where_the_programs_can_disagree() {
    let Program { b, cases, .. } = program();
    let wrong = check_differences(&b, &[], &cases, false);
    assert!(wrong.is_empty(), "differences that miss a disagreement: {:?}", wrong);
}

fn choices_program() -> (Build, Vec<Case>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let yes = b.constant(e, Ty::I1, 1);
    let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let exec = k.exec;
    let differs = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
        let c = l.m.and(bit(set), bit(exec));
        let absent = l.m.not(c);
        l.m.and(absent, bit(q))
    };
    let k = |b: &mut Build, x: u64| b.constant(e, Ty::I32, x);
    let mut cases = Vec::new();
    let (one, two, three, four, five) = (k(&mut b, 1), k(&mut b, 2), k(&mut b, 3), k(&mut b, 4), k(&mut b, 5));
    let s = b.core(e, Ty::I32, Op::Select(q, one, three));
    let v = b.int(e, IntOp::And, s, one);
    cases.push(Case { name: "select(q, 1, 3) & 1", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let s = b.core(e, Ty::I32, Op::Select(q, four, five));
    let v = b.int(e, IntOp::LShr, s, one);
    cases.push(Case { name: "select(q, 4, 5) >> 1", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let s = b.core(e, Ty::I32, Op::Select(q, one, two));
    let v = b.int(e, IntOp::And, s, one);
    cases.push(Case { name: "select(q, 1, 2) & 1", value: v, truth: Box::new(differs) });
    let s = b.core(e, Ty::I32, Op::Select(q, two, four));
    let v = b.int(e, IntOp::LShr, s, one);
    cases.push(Case { name: "select(q, 2, 4) >> 1", value: v, truth: Box::new(differs) });
    (b, cases)
}

fn ranges_program() -> (Build, Vec<Case>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let yes = b.constant(e, Ty::I1, 1);
    let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let exec = k.exec;
    let differs = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
        let c = l.m.and(bit(set), bit(exec));
        let absent = l.m.not(c);
        l.m.and(absent, bit(q))
    };
    let k = |b: &mut Build, x: u64| b.constant(e, Ty::I32, x);
    let mut cases = Vec::new();
    let (one, three, four, five) = (k(&mut b, 1), k(&mut b, 3), k(&mut b, 4), k(&mut b, 5));
    let next = b.int(e, IntOp::Add, flag, one);
    let x = b.core(e, Ty::I32, Op::Select(q, flag, next));
    let v = b.int(e, IntOp::Sub, x, x);
    cases.push(Case { name: "x - x for x = select(q, flag, flag + 1)", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let v = b.int(e, IntOp::Sub, x, flag);
    cases.push(Case { name: "x - flag for x = select(q, flag, flag + 1)", value: v, truth: Box::new(differs) });
    let low = b.int(e, IntOp::And, flag, three);
    let s = b.core(e, Ty::I32, Op::Select(q, low, four));
    let v = b.cmp(e, IntPred::Ult, s, five);
    cases.push(Case { name: "select(q, flag & 3, 4) < 5", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let v = b.cmp(e, IntPred::Ult, s, four);
    cases.push(Case { name: "select(q, flag & 3, 4) < 4", value: v, truth: Box::new(differs) });
    (b, cases)
}

fn more_ranges_program() -> (Build, Vec<Case>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let yes = b.constant(e, Ty::I1, 1);
    let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let exec = k.exec;
    let differs = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
        let c = l.m.and(bit(set), bit(exec));
        let absent = l.m.not(c);
        l.m.and(absent, bit(q))
    };
    let k = |b: &mut Build, x: u64| b.constant(e, Ty::I32, x);
    let (one, two, three, four, eight, nine) = (k(&mut b, 1), k(&mut b, 2), k(&mut b, 3), k(&mut b, 4), k(&mut b, 8), k(&mut b, 9));
    let low = b.int(e, IntOp::And, flag, three);
    let s = b.core(e, Ty::I32, Op::Select(q, low, four));
    let mut cases = Vec::new();
    let doubled = b.int(e, IntOp::Mul, s, two);
    let v = b.cmp(e, IntPred::Ult, doubled, nine);
    cases.push(Case { name: "select(q, flag & 3, 4) * 2 < 9", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let v = b.cmp(e, IntPred::Ult, doubled, eight);
    cases.push(Case { name: "select(q, flag & 3, 4) * 2 < 8", value: v, truth: Box::new(differs) });
    let shifted = b.int(e, IntOp::Shl, s, one);
    let v = b.cmp(e, IntPred::Ult, shifted, nine);
    cases.push(Case { name: "select(q, flag & 3, 4) << 1 < 9", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let flipped = b.int(e, IntOp::Xor, s, one);
    let v = b.cmp(e, IntPred::Ult, flipped, eight);
    cases.push(Case { name: "select(q, flag & 3, 4) ^ 1 < 8", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let left = b.int(e, IntOp::Sub, eight, s);
    let v = b.cmp(e, IntPred::Uge, left, four);
    cases.push(Case { name: "8 - select(q, flag & 3, 4) >= 4", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let v = b.cmp(e, IntPred::Ugt, left, four);
    cases.push(Case { name: "8 - select(q, flag & 3, 4) > 4", value: v, truth: Box::new(differs) });
    (b, cases)
}

#[test]
fn differences_vanish_for_more_operations_the_ranges_of_their_operands_decide() {
    let (b, cases) = more_ranges_program();
    let loose = check_differences(&b, &[], &cases, true);
    assert!(loose.is_empty(), "differences that claim a disagreement that cannot happen: {:?}", loose);
}

#[test]
fn differences_hold_for_more_operations_the_ranges_of_their_operands_leave_open() {
    let (b, cases) = more_ranges_program();
    let missed = check_differences(&b, &[], &cases, false);
    assert!(missed.is_empty(), "differences that miss a possible disagreement: {:?}", missed);
}

#[test]
fn differences_vanish_for_operations_the_ranges_of_their_operands_decide() {
    let (b, cases) = ranges_program();
    let loose = check_differences(&b, &[], &cases, true);
    assert!(loose.is_empty(), "differences that claim a disagreement that cannot happen: {:?}", loose);
}

#[test]
fn differences_hold_for_operations_the_ranges_of_their_operands_leave_open() {
    let (b, cases) = ranges_program();
    let missed = check_differences(&b, &[], &cases, false);
    assert!(missed.is_empty(), "differences that miss a possible disagreement: {:?}", missed);
}

#[test]
fn differences_vanish_for_operations_every_constant_choice_agrees_on() {
    let (b, cases) = choices_program();
    let loose = check_differences(&b, &[], &cases, true);
    assert!(loose.is_empty(), "differences that claim a disagreement that cannot happen: {:?}", loose);
}

#[test]
fn differences_hold_for_operations_the_constant_choices_split() {
    let (b, cases) = choices_program();
    let missed = check_differences(&b, &[], &cases, false);
    assert!(missed.is_empty(), "differences that miss a possible disagreement: {:?}", missed);
}

#[test]
fn differences_hold_only_states_where_the_programs_can_disagree() {
    let Program { b, cases, .. } = program();
    let cases: Vec<Case> = cases.into_iter().filter(|c| c.name != "any(q & exec)").collect();
    let loose = check_differences(&b, &[], &cases, true);
    assert!(loose.is_empty(), "differences that claim a disagreement that cannot happen: {:?}", loose);
}

#[test]
fn differences_of_a_converted_query_hold_where_another_conversion_changes_its_input_elsewhere() {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let yes = b.constant(e, Ty::I1, 1);
    let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let other = k.buffer(&mut b, e, 16);
    let slot = byte_offset(&mut b, e, other, lane, 4);
    let word = b.load(e, Space::Global, MemSize::B32, slot, yes);
    let marked = b.cmp(e, IntPred::Ne, word, zero);
    let d = b.int(e, IntOp::And, marked, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let v = b.int(e, IntOp::And, q, d);
    let first = b.wave(e, WaveOp::Any, vec![v]);
    let second = b.wave(e, WaveOp::Any, vec![v]);
    let any_d = b.wave(e, WaveOp::Any, vec![d]);
    let exec = k.exec;
    let cases = vec![Case {
        name: "any(q & d) beside a kept any(q & d)",
        value: second,
        truth: Box::new(move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
            let own_c = l.m.and(bit(set), bit(exec));
            let own_d = l.m.and(bit(marked), bit(exec));
            let (no_c, no_d, no_both) = (l.m.not(own_c), l.m.not(own_d), l.m.not(bit(first)));
            let lanes = l.m.and(no_c, no_d);
            let answers = l.m.and(bit(q), bit(any_d));
            let state = l.m.and(lanes, answers);
            l.m.and(state, no_both)
        }),
    }];
    let missed = check_differences(&b, &[Choice::Query(first)], &cases, false);
    assert!(missed.is_empty(), "a lane with neither bit sees any(q & d) true when some lane has c and another d, even if no lane has both under the conversion of q: {:?}", missed);
}

#[test]
fn differences_of_a_converted_query_over_another_converted_query_hold_only_where_they_can_disagree() {
    let Program { b, cases, .. } = program();
    let cases: Vec<Case> = cases.into_iter().filter(|c| c.name == "any(q & exec)").collect();
    let loose = check_differences(&b, &[], &cases, true);
    assert!(loose.is_empty(), "any(q & exec) is any(c) wherever a lane runs, so it differs only where c is clear and any(c) holds: {:?}", loose);
}

#[test]
fn differences_vanish_when_the_query_is_kept() {
    let Program { b, q, cases } = program();
    let (sound, exact): (Vec<String>, Vec<String>) = {
        let only: Vec<Case> = cases
            .into_iter()
            .filter(|c| matches!(c.name, "any(c)" | "select(q, 1, 2)" | "q & d" | "load masked by q"))
            .map(|c| Case {
                name: c.name,
                value: c.value,
                truth: if c.name == "load masked by q" {
                    Box::new(move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| l.m.not(bit(q)))
                } else {
                    Box::new(|_: &mut Logic, _: &dyn Fn(ValueId) -> Bdd| Bdd::FALSE)
                },
            })
            .collect();
        (
            check_differences(&b, &[Choice::Query(q)], &only, false),
            check_differences(&b, &[Choice::Query(q)], &only, true),
        )
    };
    assert!(
        sound.is_empty() && exact.is_empty(),
        "with the query kept only an unperformed load differs: {:?} {:?}",
        sound,
        exact
    );
}

fn relations_program() -> (Build, Vec<Case>) {
    let (mut b, k) = Build::kernel();
    let e = BlockId(0);
    let table = k.buffer(&mut b, e, 8);
    let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
    let own = byte_offset(&mut b, e, table, lane, 4);
    let yes = b.constant(e, Ty::I1, 1);
    let flag = b.load(e, Space::Global, MemSize::B32, own, yes);
    let zero = b.constant(e, Ty::I32, 0);
    let set = b.cmp(e, IntPred::Ne, flag, zero);
    let c = b.int(e, IntOp::And, set, k.exec);
    let q = b.wave(e, WaveOp::Any, vec![c]);
    let exec = k.exec;
    let differs = move |l: &mut Logic, bit: &dyn Fn(ValueId) -> Bdd| {
        let c = l.m.and(bit(set), bit(exec));
        let absent = l.m.not(c);
        l.m.and(absent, bit(q))
    };
    let k = |b: &mut Build, x: u64| b.constant(e, Ty::I32, x);
    let (one, two, three, four) = (k(&mut b, 1), k(&mut b, 2), k(&mut b, 3), k(&mut b, 4));
    let low = b.int(e, IntOp::And, flag, three);
    let s = b.core(e, Ty::I32, Op::Select(q, low, four));
    let next = b.int(e, IntOp::Add, s, one);
    let mut cases = Vec::new();
    let v = b.cmp(e, IntPred::Ult, s, next);
    cases.push(Case { name: "s < s + 1 for s = select(q, flag & 3, 4)", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    let doubled = b.int(e, IntOp::Mul, s, two);
    let v = b.cmp(e, IntPred::Ult, next, doubled);
    cases.push(Case { name: "s + 1 < s * 2 for s = select(q, flag & 3, 4)", value: v, truth: Box::new(differs) });
    let v = b.cmp(e, IntPred::Uge, doubled, s);
    cases.push(Case { name: "s * 2 >= s for s = select(q, flag & 3, 4)", value: v, truth: Box::new(|_, _| Bdd::FALSE) });
    (b, cases)
}

#[test]
fn differences_vanish_for_operations_the_relations_of_their_operands_decide() {
    let (b, cases) = relations_program();
    let loose = check_differences(&b, &[], &cases, true);
    assert!(loose.is_empty(), "differences that claim a disagreement that cannot happen: {:?}", loose);
}

#[test]
fn differences_hold_for_operations_the_relations_of_their_operands_leave_open() {
    let (b, cases) = relations_program();
    let missed = check_differences(&b, &[], &cases, false);
    assert!(missed.is_empty(), "differences that miss a possible disagreement: {:?}", missed);
}
