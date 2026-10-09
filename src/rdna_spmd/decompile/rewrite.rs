use crate::rdna_spmd::analysis::facts::Facts;
use super::logic::Kept;
use crate::rdna_spmd::ir::*;

struct Block<'a> {
    q: &'a mut Func,
    insts: Vec<Inst>,
    lane: Option<ValueId>,
}

impl Block<'_> {
    fn core(&mut self, ty: Ty, op: Op) -> ValueId {
        let value = self.q.value(ty);
        self.insts.push(Inst::Core { value, ty, op });
        value
    }

    fn lane_id(&mut self) -> ValueId {
        match self.lane {
            Some(v) => v,
            None => {
                let v = self.core(Ty::I32, Op::Env(Env::LaneId));
                self.lane = Some(v);
                v
            }
        }
    }

    fn bit(&mut self, p: &Func, facts: &Facts, v: ValueId) -> ValueId {
        if converted(facts, v) {
            return v;
        }
        let ty = p.types[v.0];
        let ones = if ty == Ty::I64 { u64::MAX } else { u32::MAX as u64 };
        match facts.constant(p, v) {
            Some(0) => self.core(Ty::I1, Op::Const(Ty::I1, 0)),
            Some(k) if k == ones => self.core(Ty::I1, Op::Const(Ty::I1, 1)),
            _ => {
                let mut lane = self.lane_id();
                if ty == Ty::I64 {
                    lane = self.core(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
                }
                let shifted = self.core(ty, Op::Int(IntOp::LShr, v, lane));
                self.core(Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted))
            }
        }
    }

    fn upper(&mut self) -> ValueId {
        let lane = self.lane_id();
        let half = self.core(Ty::I32, Op::Const(Ty::I32, 32));
        self.core(Ty::I1, Op::Cmp(IntPred::Uge, lane, half))
    }
}

pub fn rename(f: &mut Func, map: &std::collections::BTreeMap<ValueId, ValueId>) {
    if map.is_empty() {
        return;
    }
    let m = |v: ValueId| {
        let mut v = v;
        while let Some(&next) = map.get(&v) {
            v = next;
        }
        v
    };
    for block in f.blocks.values_mut() {
        for inst in &mut block.insts {
            match inst {
                Inst::Core { op, .. } => *op = op.map(m),
                Inst::Packet { input, .. } => *input = m(*input),
                Inst::Target { args, .. } => *args = args.map(m),
                Inst::Effect { inputs, .. } => {
                    for v in inputs {
                        *v = m(*v);
                    }
                }
            }
        }
        for edge in block.term.edges_mut() {
            for v in &mut edge.args {
                *v = m(*v);
            }
        }
        match &mut block.term {
            Term::CondBr { cond, .. } => *cond = m(*cond),
            Term::Ret(args) => {
                for v in args {
                    *v = m(*v);
                }
            }
            Term::Br(_) => {}
        }
    }
}

fn converted(facts: &Facts, v: ValueId) -> bool {
    facts.lane_word[v.0] && !facts.materialized[v.0]
}

fn projection(p: &Func, facts: &Facts, s: ValueId) -> Option<ValueId> {
    match facts.op(p, s) {
        Some(Op::Int(IntOp::LShr, w, lane)) if facts.is_lane_shift(p, w, lane) && converted(facts, w) => {
            Some(w)
        }
        _ => None,
    }
}

pub fn lane_program(
    p: &Func,
    facts: &Facts,
    kept: &Kept,
    meetings: &std::collections::BTreeMap<(BlockId, usize), u64>,
) -> Func {
    let reachable: std::collections::BTreeSet<BlockId> = facts.order.iter().copied().collect();
    let mut q = Func::new(p.entry, Presence::Wave, p.lanes);
    q.blocks = p
        .blocks
        .iter()
        .filter(|(id, _)| reachable.contains(id))
        .map(|(&id, b)| (id, b.clone()))
        .collect();
    q.types = p.types.clone();
    for v in 0..q.types.len() {
        if converted(facts, ValueId(v)) {
            q.types[v] = Ty::I1;
        }
    }
    for &id in &facts.order {
        let old = std::mem::take(&mut q.blocks.get_mut(&id).unwrap().insts);
        let mut b = Block {
            q: &mut q,
            insts: Vec::with_capacity(old.len()),
            lane: None,
        };
        for (index, inst) in old.into_iter().enumerate() {
            if let Some(&provenance) = meetings.get(&(id, index)) {
                b.insts.push(Inst::Effect {
                    provenance,
                    op: EffectOp::Wave(WaveOp::Meet),
                    inputs: vec![],
                    outputs: vec![],
                });
            }
            rewrite_inst(p, facts, kept, &mut b, inst);
        }
        let mut term = b.q.blocks[&id].term.clone();
        for edge in term.edges_mut() {
            let params: Vec<ValueId> = b.q.blocks[&edge.dst].params.iter().map(|x| x.0).collect();
            for (arg, param) in edge.args.iter_mut().zip(params) {
                if converted(facts, param) {
                    *arg = b.bit(p, facts, *arg);
                }
            }
        }
        if let Term::Ret(args) = &mut term {
            args.clear();
        }
        let insts = std::mem::take(&mut b.insts);
        drop(b);
        let block = q.blocks.get_mut(&id).unwrap();
        block.insts = insts;
        block.term = term;
        for (v, ty) in &mut block.params {
            *ty = q.types[v.0];
        }
    }
    q.one_region(q.presence_needed());
    q
}

fn rewrite_inst(p: &Func, facts: &Facts, kept: &Kept, b: &mut Block, inst: Inst) {
    match inst {
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::Any),
            ref inputs,
            ref outputs,
            ..
        } => {
            if kept.queries.contains(&outputs[0].0) {
                b.insts.push(inst);
            } else {
                b.insts.push(Inst::Core {
                    value: outputs[0].0,
                    ty: Ty::I1,
                    op: Op::Convert(Cvt::Bitcast, Ty::I1, inputs[0]),
                });
            }
        }
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::Ballot { .. }),
            ref inputs,
            ref outputs,
            ..
        } => {
            let value = outputs[0].0;
            if converted(facts, value) {
                b.insts.push(Inst::Core {
                    value,
                    ty: Ty::I1,
                    op: Op::Convert(Cvt::Bitcast, Ty::I1, inputs[0]),
                });
            } else {

                b.insts.push(inst);
            }
        }
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::ReadFirstLane),
            ref inputs,
            ref outputs,
            ..
        } => {
            if facts.uniform[inputs[0].0] {
                b.insts.push(Inst::Core {
                    value: outputs[0].0,
                    ty: Ty::I32,
                    op: Op::Convert(Cvt::Bitcast, Ty::I32, inputs[0]),
                });
            } else {
                b.insts.push(inst);
            }
        }
        Inst::Core {
            value,
            op: Op::Env(Env::ValidLane),
            ..
        } => b.insts.push(Inst::Core {
            value,
            ty: Ty::I1,
            op: Op::Const(Ty::I1, 1),
        }),
        Inst::Core {
            value,
            op: Op::Int(IntOp::LShr, ..),
            ..
        } if projection(p, facts, value).is_some() => {}
        Inst::Core {
            value,
            op: Op::Convert(Cvt::Trunc, Ty::I1, s),
            ..
        } if projection(p, facts, s).is_some() => b.insts.push(Inst::Core {
            value,
            ty: Ty::I1,
            op: Op::Convert(Cvt::Bitcast, Ty::I1, projection(p, facts, s).unwrap()),
        }),
        Inst::Core {
            value,
            op: Op::Cmp(pred @ (IntPred::Eq | IntPred::Ne), x, y),
            ..
        } if super::logic::lane_test(p, facts, x, y).is_some_and(|w| converted(facts, w)) => {
            let w = super::logic::lane_test(p, facts, x, y).unwrap();
            let bit = b.bit(p, facts, w);
            let op = if pred == IntPred::Ne {
                Op::Convert(Cvt::Bitcast, Ty::I1, bit)
            } else {
                let one = b.core(Ty::I1, Op::Const(Ty::I1, 1));
                Op::Int(IntOp::Xor, bit, one)
            };
            b.insts.push(Inst::Core {
                value,
                ty: Ty::I1,
                op,
            });
        }
        Inst::Core { value, op, .. } if converted(facts, value) => {
            let op = match op {
                Op::Int(k @ (IntOp::And | IntOp::Or | IntOp::Xor), x, y) => {
                    Op::Int(k, b.bit(p, facts, x), b.bit(p, facts, y))
                }
                Op::Select(c, x, y) => Op::Select(c, b.bit(p, facts, x), b.bit(p, facts, y)),
                Op::Convert(Cvt::Bitcast, Ty::I32 | Ty::I64, x) | Op::UnpackLo(x) | Op::UnpackHi(x) => {
                    Op::Convert(Cvt::Bitcast, Ty::I1, b.bit(p, facts, x))
                }
                Op::Pack64(x, _) if p.lanes == 32 => Op::Convert(Cvt::Bitcast, Ty::I1, b.bit(p, facts, x)),
                Op::Pack64(x, y) => {
                    let (low, high) = (b.bit(p, facts, x), b.bit(p, facts, y));
                    let upper = b.upper();
                    Op::Select(upper, high, low)
                }
                other => unreachable!("a lane word defined by {:?}", other),
            };
            b.insts.push(Inst::Core {
                value,
                ty: Ty::I1,
                op,
            });
        }
        other => b.insts.push(other),
    }
}

#[cfg(test)]
mod tests {
    use super::super::logic::{Atom, Choice, Logic};
    use super::super::testing::*;
    use super::*;
    use crate::rdna_spmd::analysis::bdd::Bdd;
    use std::collections::{BTreeMap, BTreeSet};

    fn canonical(l: &Lowered, atom: Atom) -> Atom {
        match atom {
            Atom::View(v) if l.lane.types[v.0] == Ty::I1 => Atom::Bit(v),
            other => other,
        }
    }

    fn value(l: &Lowered, logic: &Logic, f: Bdd, assignment: &std::collections::HashMap<Atom, bool>) -> bool {
        let mut f = f;
        while let Some((var, low, high)) = logic.m.decompose(f) {
            f = if assignment[&canonical(l, logic.atom_of(var))] { high } else { low };
        }
        f == Bdd::TRUE
    }

    fn atoms(l: &Lowered, logic: &mut Logic, f: Bdd) -> Vec<Atom> {
        logic.support(f).iter().map(|&var| canonical(l, logic.atom_of(var))).collect()
    }

    struct Lowered {
        wave: Func,
        wave_facts: Facts,
        lane: Func,
        kept: BTreeSet<Choice>,
        words: BTreeSet<ValueId>,
    }

    fn lower(b: &Build, kept: &Kept, meetings: &BTreeMap<(BlockId, usize), u64>) -> Lowered {
        let facts = Facts::new(&b.f, &b.inputs, &kept.words);
        let lane = lane_program(&b.f, &facts, kept, meetings);
        Lowered {
            wave: b.f.clone(),
            wave_facts: facts,
            lane,
            kept: kept.choices(),
            words: kept.words.clone(),
        }
    }

    fn valid(b: &Build, l: &Lowered) -> Result<(), &'static str> {
        let mut lane = l.lane.clone();
        lane.compact();
        lane.check(&b.registry)
    }

    fn mismatches(b: &Build, l: &Lowered) -> Vec<ValueId> {
        let mut wave = Logic::fixed(&l.wave, &l.wave_facts, &l.kept, &[]);
        let lane_facts = Facts::new(&l.lane, &b.inputs, &l.words);
        let queries: BTreeSet<Choice> = l
            .lane
            .blocks
            .values()
            .flat_map(|b| &b.insts)
            .filter_map(|inst| match inst {
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any),
                    outputs,
                    ..
                } => Some(Choice::Query(outputs[0].0)),
                _ => None,
            })
            .collect();
        let mut lane = Logic::fixed(&l.lane, &lane_facts, &queries, &[]);
        let mut wrong = Vec::new();
        let same = |l: &Lowered, wave: &mut Logic, lane: &mut Logic, expected: Bdd, got: Bdd| {
            let mut all: Vec<Atom> = atoms(l, wave, expected);
            all.extend(atoms(l, lane, got));
            all.sort_by_key(|a| format!("{:?}", a));
            all.dedup();
            assert!(all.len() <= 16, "too many atoms to enumerate");
            (0..1u32 << all.len()).all(|bits| {
                let assignment: std::collections::HashMap<Atom, bool> =
                    all.iter().enumerate().map(|(i, &a)| (a, bits >> i & 1 != 0)).collect();
                value(l, wave, expected, &assignment) == value(l, lane, got, &assignment)
            })
        };
        for &id in &l.wave_facts.order {
            let edges = l.wave.blocks[&id].term.edges().zip(l.lane.blocks[&id].term.edges());
            for (we, le) in edges.collect::<Vec<_>>() {
                for ((&wa, &la), &(param, _)) in we.args.iter().zip(&le.args).zip(&l.lane.blocks[&we.dst].params) {
                    if l.lane.types[param.0] != Ty::I1 {
                        continue;
                    }
                    let expected = if l.wave.types[wa.0] == Ty::I1 {
                        wave.bit(&l.wave, &l.wave_facts, wa)
                    } else {
                        wave.view(&l.wave, &l.wave_facts, wa)
                    };
                    let got = lane.bit(&l.lane, &lane_facts, la);
                    if !same(l, &mut wave, &mut lane, expected, got) {
                        eprintln!("b{} -> b{}: the argument for v{} differs", id.0, we.dst.0, param.0);
                        wrong.push(param);
                    }
                }
            }
        }
        for &id in &l.wave_facts.order {
            let block = &l.wave.blocks[&id];
            let values: Vec<ValueId> = block
                .params
                .iter()
                .map(|p| p.0)
                .chain(block.insts.iter().flat_map(Inst::outputs))
                .collect();
            for v in values {
                if lane_facts.site[v.0] == crate::rdna_spmd::analysis::facts::Site::Unreached || l.lane.types[v.0] != Ty::I1 {
                    continue;
                }
                let expected = if l.wave.types[v.0] == Ty::I1 {
                    wave.bit(&l.wave, &l.wave_facts, v)
                } else {
                    wave.view(&l.wave, &l.wave_facts, v)
                };
                let got = lane.bit(&l.lane, &lane_facts, v);
                let mut all: Vec<Atom> = atoms(l, &mut wave, expected);
                all.extend(atoms(l, &mut lane, got));
                all.sort_by_key(|a| format!("{:?}", a));
                all.dedup();
                assert!(all.len() <= 16, "too many atoms to enumerate");
                let same = (0..1u32 << all.len()).all(|bits| {
                    let assignment: std::collections::HashMap<Atom, bool> =
                        all.iter().enumerate().map(|(i, &a)| (a, bits >> i & 1 != 0)).collect();
                    value(l, &wave, expected, &assignment) == value(l, &lane, got, &assignment)
                });
                if !same {
                    eprintln!(
                        "v{}: wave {:?} lane {:?}\n  wave inst {:?}\n  lane inst {:?}",
                        v.0,
                        atoms(l, &mut wave, expected),
                        atoms(l, &mut lane, got),
                        l.wave_facts.inst(&l.wave, v),
                        lane_facts.inst(&l.lane, v)
                    );
                    wrong.push(v);
                }
            }
        }
        wrong
    }

    struct Words {
        b: Build,
        q: ValueId,
        kept_query: ValueId,
        w: ValueId,
        tests: Vec<ValueId>,
    }

    fn words() -> Words {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let flags = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let five = b.constant(e, Ty::I32, 5);
        let small = b.cmp(e, IntPred::Ult, flag, five);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        let kept_query = b.wave(e, WaveOp::Any, vec![small]);
        let w = b.wave(e, WaveOp::Ballot { high: false }, vec![c]);
        let v = b.wave(e, WaveOp::Ballot { high: false }, vec![small]);
        let shifted = b.int(e, IntOp::LShr, w, lane);
        let own_bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let any = b.cmp(e, IntPred::Ne, w, zero);
        let none = b.cmp(e, IntPred::Eq, zero, w);
        let mask = b.constant(e, Ty::I32, 0xf0);
        let masked = b.int(e, IntOp::And, w, mask);
        let masked_any = b.cmp(e, IntPred::Ne, masked, zero);
        let ones = b.constant(e, Ty::I32, 0xffff_ffff);
        let chosen = b.core(e, Ty::I32, Op::Select(small, w, ones));
        let chosen_any = b.cmp(e, IntPred::Ne, chosen, zero);
        let both = b.int(e, IntOp::Xor, w, v);
        let both_any = b.cmp(e, IntPred::Ne, both, zero);
        let cast = b.core(e, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, w));
        let cast_none = b.cmp(e, IntPred::Eq, cast, zero);
        let valid = b.core(e, Ty::I1, Op::Env(Env::ValidLane));
        let mixed = b.int(e, IntOp::And, q, kept_query);
        let mixed = b.int(e, IntOp::Or, mixed, valid);
        let tests = vec![own_bit, any, none, masked_any, chosen_any, both_any, cast_none, mixed];
        let buf = k.buffer(&mut b, e, 0);
        let out = byte_offset(&mut b, e, buf, lane, 4);
        let one = b.constant(e, Ty::I32, 1);
        for &t in &tests {
            b.store(e, Space::Global, MemSize::B32, out, one, t);
        }
        Words {
            b,
            q,
            kept_query,
            w,
            tests,
        }
    }

    #[test]
    fn lane_program_computes_the_bits_the_proof_assumed_when_it_converts_everything() {
        let Words { b, kept_query, .. } = words();
        let mut kept = Kept::default();
        kept.insert(Choice::Query(kept_query));
        let l = lower(&b, &kept, &BTreeMap::new());
        assert_eq!(mismatches(&b, &l), Vec::<ValueId>::new());
        assert_eq!(valid(&b, &l), Ok(()));
    }

    #[test]
    fn lane_program_computes_the_bits_the_proof_assumed_when_it_keeps_the_words() {
        let Words { b, q, kept_query, w, tests } = words();
        let mut kept = Kept::default();
        kept.insert(Choice::Query(q));
        kept.insert(Choice::Query(kept_query));
        kept.insert(Choice::Word(w));
        let l = lower(&b, &kept, &BTreeMap::new());
        assert_eq!(mismatches(&b, &l), Vec::<ValueId>::new());
        assert_eq!(valid(&b, &l), Ok(()));
        assert_eq!(l.lane.types[w.0], Ty::I32, "a kept word stays a word");
        assert!(!tests.is_empty());
    }

    #[test]
    fn lane_program_reads_a_pair_of_ballot_halves_as_each_lane_s_own_bit() {
        let (mut b, k, _) = Build::kernel_in(&[], 64);
        let e = BlockId(0);
        let flags = k.buffer(&mut b, e, 8);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let own = byte_offset(&mut b, e, flags, lane, 4);
        let flag = b.load(e, Space::Global, MemSize::B32, own, k.exec);
        let zero = b.constant(e, Ty::I32, 0);
        let set = b.cmp(e, IntPred::Ne, flag, zero);
        let c = b.int(e, IntOp::And, set, k.exec);
        let low = b.wave(e, WaveOp::Ballot { high: false }, vec![c]);
        let high = b.wave(e, WaveOp::Ballot { high: true }, vec![c]);
        let pair = b.core(e, Ty::I64, Op::Pack64(low, high));
        let wide_lane = b.core(e, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
        let shifted = b.int(e, IntOp::LShr, pair, wide_lane);
        let own_bit = b.core(e, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let wide_zero = b.constant(e, Ty::I64, 0);
        let any = b.cmp(e, IntPred::Ne, pair, wide_zero);
        let none = b.cmp(e, IntPred::Eq, wide_zero, pair);
        let mask = b.constant(e, Ty::I64, 0xf0_0000_00f0);
        let masked = b.int(e, IntOp::And, pair, mask);
        let masked_any = b.cmp(e, IntPred::Ne, masked, wide_zero);
        let lo = b.core(e, Ty::I32, Op::UnpackLo(pair));
        let hi = b.core(e, Ty::I32, Op::UnpackHi(pair));
        let repacked = b.core(e, Ty::I64, Op::Pack64(lo, hi));
        let repacked_any = b.cmp(e, IntPred::Ne, repacked, wide_zero);
        let tests = vec![own_bit, any, none, masked_any, repacked_any];
        let buf = k.buffer(&mut b, e, 0);
        let out = byte_offset(&mut b, e, buf, lane, 4);
        let one = b.constant(e, Ty::I32, 1);
        for &t in &tests {
            b.store(e, Space::Global, MemSize::B32, out, one, t);
        }
        let l = lower(&b, &Kept::default(), &BTreeMap::new());
        assert_eq!(valid(&b, &l), Ok(()));
        for &w in &[low, high, pair, lo, hi, repacked] {
            assert_eq!(l.lane.types[w.0], Ty::I1, "v{} is read only by lanes and tests, so it becomes a bit", w.0);
        }
        let halves = [low, high, lo, hi];
        let wrong: Vec<ValueId> = mismatches(&b, &l).into_iter().filter(|v| !halves.contains(v)).collect();
        assert_eq!(wrong, Vec::<ValueId>::new(), "every value read off the pair agrees with the wave program");
        assert!(l.lane.blocks[&e].insts.iter().all(|i| !matches!(i, Inst::Effect { op: EffectOp::Wave(_), .. })), "no lane asks the wave");
    }

    #[test]
    fn lane_program_hands_a_materialized_word_to_a_converted_parameter_as_its_own_bit() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let five = b.constant(e, Ty::I32, 5);
        let small = b.cmp(e, IntPred::Ult, lane, five);
        let w = b.wave(e, WaveOp::Ballot { high: false }, vec![small]);
        let count = b.core(e, Ty::I32, Op::PopulationCount(w));
        let out = byte_offset(&mut b, e, buf, lane, 4);
        b.store(e, Space::Global, MemSize::B32, out, count, k.exec);
        let (next, n) = b.block(&[Ty::I1, Ty::I32, Ty::I64]);
        b.br(e, next, vec![k.exec, w, out]);
        let zero = b.constant(next, Ty::I32, 0);
        let any = b.cmp(next, IntPred::Ne, n[1], zero);
        let one = b.constant(next, Ty::I32, 1);
        b.store(next, Space::Global, MemSize::B32, n[2], one, any);
        let l = lower(&b, &Kept::default(), &BTreeMap::new());
        assert_eq!(l.lane.types[w.0], Ty::I32, "the population count needs the whole word");
        assert_eq!(l.lane.types[n[1].0], Ty::I1, "the parameter is read only by a lane test");
        assert_eq!(valid(&b, &l), Ok(()));
        assert_eq!(mismatches(&b, &l), Vec::<ValueId>::new());
    }

    #[test]
    fn lane_program_puts_each_kept_meeting_right_before_its_instruction() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let one = b.constant(e, Ty::I32, 1);
        b.store(e, Space::Global, MemSize::B32, buf, one, k.exec);
        let second = b.here(e);
        b.store(e, Space::Global, MemSize::B32, buf, one, k.exec);
        let meetings = BTreeMap::from([(second, 0xc000_0000_0000_0005u64)]);
        let l = lower(&b, &Kept::default(), &meetings);
        let insts = &l.lane.blocks[&e].insts;
        let stores: Vec<usize> = insts
            .iter()
            .enumerate()
            .filter(|(_, i)| matches!(i, Inst::Effect { op: EffectOp::Memory { op: MemoryOp::Store(_), .. }, .. }))
            .map(|(k, _)| k)
            .collect();
        assert_eq!(stores.len(), 2);
        let meets: Vec<(usize, u64)> = insts
            .iter()
            .enumerate()
            .filter_map(|(k, i)| match i {
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Meet),
                    provenance,
                    ..
                } => Some((k, *provenance)),
                _ => None,
            })
            .collect();
        assert_eq!(meets, vec![(stores[1] - 1, 0xc000_0000_0000_0005)], "one meeting, between the stores");
        assert!(stores[0] < stores[1] - 1);
    }

    #[test]
    fn lane_program_reads_a_uniform_first_lane_locally_and_keeps_the_others() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lo = b.core(e, Ty::I32, Op::UnpackLo(buf));
        let uniform = b.wave(e, WaveOp::ReadFirstLane, vec![lo, k.exec]);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let varying = b.wave(e, WaveOp::ReadFirstLane, vec![lane, k.exec]);
        let sum = b.int(e, IntOp::Add, uniform, varying);
        let out = byte_offset(&mut b, e, buf, lane, 4);
        b.store(e, Space::Global, MemSize::B32, out, sum, k.exec);
        let l = lower(&b, &Kept::default(), &BTreeMap::new());
        let defined = |v: ValueId| l.lane.blocks[&e].insts.iter().find(|i| i.outputs().contains(&v)).cloned();
        assert!(
            matches!(defined(uniform), Some(Inst::Core { op: Op::Convert(Cvt::Bitcast, Ty::I32, x), .. }) if x == lo),
            "every lane holds the same word, so each reads its own: {:?}",
            defined(uniform)
        );
        assert!(
            matches!(defined(varying), Some(Inst::Effect { op: EffectOp::Wave(WaveOp::ReadFirstLane), .. })),
            "the lanes hold different words, so the read stays collective"
        );
        assert_eq!(valid(&b, &l), Ok(()));
    }

    #[test]
    fn lane_program_drops_blocks_the_entry_does_not_reach_and_the_values_ret_returns() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let (dead, _) = b.block(&[Ty::I1]);
        b.f.blocks.get_mut(&e).unwrap().term = Term::Ret(vec![k.exec]);
        let _ = dead;
        let l = lower(&b, &Kept::default(), &BTreeMap::new());
        assert!(!l.lane.blocks.contains_key(&dead));
        assert_eq!(l.lane.blocks[&e].term, Term::Ret(Vec::new()));
    }

    #[test]
    fn lane_program_converts_a_carried_word_whose_back_edge_brings_a_materialized_word() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let five = b.constant(e, Ty::I32, 5);
        let small = b.cmp(e, IntPred::Ult, lane, five);
        let w = b.wave(e, WaveOp::Ballot { high: false }, vec![small]);
        let zero = b.constant(e, Ty::I32, 0);
        let (head, h) = b.block(&[Ty::I1, Ty::I32, Ty::I32, Ty::I64]);
        let (exit, _) = b.block(&[Ty::I1]);
        b.br(e, head, vec![k.exec, w, zero, buf]);
        let lane_h = b.core(head, Ty::I32, Op::Env(Env::LaneId));
        let shifted = b.int(head, IntOp::LShr, h[1], lane_h);
        let own = b.core(head, Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        let mask = b.int(head, IntOp::And, own, h[0]);
        let one = b.constant(head, Ty::I32, 1);
        b.store(head, Space::Global, MemSize::B32, h[3], one, mask);
        let three = b.constant(head, Ty::I32, 3);
        let odd = b.cmp(head, IntPred::Ult, lane_h, three);
        let w2 = b.wave(head, WaveOp::Ballot { high: false }, vec![odd]);
        let count = b.core(head, Ty::I32, Op::PopulationCount(w2));
        b.store(head, Space::Global, MemSize::B32, h[3], count, h[0]);
        let next = b.int(head, IntOp::Add, h[2], one);
        let four = b.constant(head, Ty::I32, 4);
        let again = b.cmp(head, IntPred::Ult, next, four);
        b.cond_br(head, again, (head, vec![h[0], w2, next, h[3]]), (exit, vec![h[0]]));
        let l = lower(&b, &Kept::default(), &BTreeMap::new());
        assert_eq!(l.lane.types[h[1].0], Ty::I1, "the carried word is read only by a lane test");
        assert_eq!(l.lane.types[w2.0], Ty::I32, "the population count needs the whole word");
        assert_eq!(l.lane.types[w.0], Ty::I1, "the entering ballot is read only through the parameter");
        assert_eq!(valid(&b, &l), Ok(()));
        assert_eq!(mismatches(&b, &l), Vec::<ValueId>::new());
    }
}
