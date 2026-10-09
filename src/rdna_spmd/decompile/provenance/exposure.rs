use super::super::address::{aligns, Region};
use super::demand::{demands, uses, Demand};
use super::layout::{merge, pointers, points};
use super::sets::Sets;
use super::{Exposure, Spill};
use crate::rdna_spmd::environment::Environment;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

pub(super) fn exposures(sets: &Sets, env: &Environment) -> (Vec<Exposure>, Vec<Spill>) {
    let (program, layout) = (sets.program(), sets.layout());
    let (f, facts) = (program.f, program.facts);
    let used = uses(program);
    let mut demand: Option<Vec<Demand>> = None;
    let mut wanted = vec![0; sets.words()];
    for &id in layout.allocations() {
        if !env.exposed.contains(&id) {
            layout.mark(&mut wanted, Some(Region::Allocation(id)));
        }
    }
    let mut memos: HashMap<ValueId, (Vec<ValueId>, HashMap<ValueId, Vec<u64>>)> = HashMap::default();
    let mut exposing = Vec::new();
    let mut spills: Vec<Spill> = Vec::new();
    let mut buffer = [ValueId(0); 3];
    let mut acc = vec![0; sets.words()];
    for &b in &facts.order {
        for (index, inst) in f.blocks[&b].insts.iter().enumerate() {
            let (operands, lanewise, assume, scratch): (&[ValueId], _, _, _) = match inst {
                Inst::Core { value, op, .. } => {
                    let n = exposed(sets, *value, *op, &mut buffer);
                    (&buffer[..n], true, None, None)
                }
                Inst::Target { op, args, .. } if program.pure(*op) => (args.values(), true, None, None),
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Wmma),
                    inputs,
                    ..
                } => (&inputs[..], false, None, None),
                Inst::Effect {
                    op: EffectOp::Memory { op, space, .. },
                    inputs,
                    ..
                } if !matches!(op, MemoryOp::Load(_) | MemoryOp::Fence) => {
                    let m = op.mask_input();
                    let bytes = match op {
                        MemoryOp::Store(size) => size.bytes(),
                        _ => 4,
                    };
                    let scratch = (*space == Space::Scratch).then_some((inputs[0], bytes));
                    (&inputs[1..m], false, Some(inputs[m]), scratch)
                }
                _ => continue,
            };
            let holds = |set: &[u64]| layout.has(set, Some(Region::Kernarg)) || layout.has(set, Some(Region::Dispatch));
            let reaching = |set: &[u64]| holds(set) || set.iter().zip(&wanted).any(|(x, y)| x & y != 0);
            let relevant = match scratch {
                Some(_) => operands.iter().any(|&x| points(sets.of(x))),
                None => operands.iter().any(|&x| reaching(sets.of(x))),
            };
            if !relevant {
                continue;
            }
            let mut assume = assume;
            if !matches!(inst, Inst::Effect { op: EffectOp::Memory { .. }, .. }) {
                let mut data = false;
                inst.for_each_output(|o| data |= used[o.0]);
                if !data {
                    continue;
                }
                let demand = demand.get_or_insert_with(|| demands(program));
                let mut out = Demand::Dead;
                inst.for_each_output(|o| out = out.meet(demand[o.0]));
                match (out, lanewise) {
                    (Demand::Dead, _) => continue,
                    (Demand::Under(c), true) => assume = Some(c),
                    _ => {}
                }
            }
            acc.fill(0);
            for &x in operands {
                match assume {
                    Some(c) => {
                        let (held, memo) = memos.entry(c).or_insert_with(|| (program.conjuncts(c), HashMap::default()));
                        merge(&mut acc, &under(sets, x, held, memo));
                    }
                    None => {
                        merge(&mut acc, sets.of(x));
                    }
                }
            }
            if let Some((address, bytes)) = scratch {
                acc[0] &= !1;
                if !points(&acc) {
                    continue;
                }
                let words = program.words(address, bytes);
                let parts = if operands.iter().any(|&x| sets.has_parts(x)) {
                    (0..program.lanes())
                        .map(|lane| {
                            let mut part = vec![0; sets.words()];
                            for &x in operands {
                                merge(&mut part, sets.part(x, lane));
                            }
                            for (p, a) in part.iter_mut().zip(&acc) {
                                *p &= a;
                            }
                            layout.regions(&part)
                        })
                        .collect()
                } else {
                    vec![layout.regions(&acc)]
                };
                spills.push(Spill {
                    at: (b, index),
                    words,
                    data: operands.to_vec(),
                    mask: assume.unwrap(),
                    parts,
                });
                continue;
            }
            let candidates: Vec<u64> = layout
                .allocations()
                .iter()
                .copied()
                .filter(|&id| (holds(&acc) || layout.has(&acc, Some(Region::Allocation(id)))) && layout.has(&wanted, Some(Region::Allocation(id))))
                .collect();
            if !candidates.is_empty() {
                exposing.push(Exposure {
                    operands: operands.to_vec(),
                    assume,
                    candidates,
                });
            }
        }
    }
    (exposing, spills)
}

fn exposed(sets: &Sets, v: ValueId, op: Op, out: &mut [ValueId; 3]) -> usize {
    let program = sets.program();
    let pointing = |x: ValueId| points(sets.of(x));
    let wide = program.f.types[v.0] == Ty::I64;
    let high = |s: ValueId| wide && program.constant(s).is_some_and(|k| k & 63 >= 32);
    let list: &[ValueId] = match op {
        Op::Int(IntOp::And, a, b) => match (program.constant(a), program.constant(b)) {
            (None, Some(m)) | (Some(m), None) if aligns(m) => &[],
            _ => &[a, b],
        },
        Op::Int(IntOp::Mul | IntOp::Shl, a, b) | Op::Float(_, a, b) => &[a, b],
        Op::Int(IntOp::Add, a, b) if pointing(a) && pointing(b) => &[a, b],
        Op::Int(IntOp::Sub, a, b) if pointing(b) => &[a, b],
        Op::Int(IntOp::AShr, a, b) if !high(b) => &[a, b],
        Op::Int(IntOp::LShr, a, b) if wide && !high(b) => &[a, b],
        Op::Pack64(_, a) | Op::ReverseBits(a) | Op::Unary(_, a) => &[a],
        Op::Fma(a, b, c) | Op::MulAdd(a, b, c) => &[a, b, c],
        Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, ..) => &[],
        Op::Convert(_, _, a) => &[a],
        _ => &[],
    };
    out[..list.len()].copy_from_slice(list);
    list.len()
}

fn under(sets: &Sets, x: ValueId, held: &[ValueId], memo: &mut HashMap<ValueId, Vec<u64>>) -> Vec<u64> {
    let program = sets.program();
    let x = program.copies.get(&x).copied().unwrap_or(x);
    if !points(sets.of(x)) {
        return sets.of(x).to_vec();
    }
    if let Some(found) = memo.get(&x) {
        return found.clone();
    }
    let (f, facts) = (program.f, program.facts);
    let mut acc = vec![0; sets.words()];
    match facts.op(f, x) {
        Some(Op::Select(c, a, b)) => {
            let c = program.copies.get(&c).copied().unwrap_or(c);
            if held.contains(&c) {
                acc = under(sets, a, held, memo);
            } else {
                merge(&mut acc, &under(sets, a, held, memo));
                merge(&mut acc, &under(sets, b, held, memo));
            }
        }
        Some(Op::Int(IntOp::Or | IntOp::Xor, a, b)) => {
            let (y, z) = (under(sets, a, held, memo), under(sets, b, held, memo));
            pointers(&mut acc, &y);
            pointers(&mut acc, &z);
            if y[0] & z[0] & 1 != 0 {
                acc[0] |= 1;
            }
        }
        Some(Op::Int(IntOp::Add, a, b)) => {
            let (y, z) = (under(sets, a, held, memo), under(sets, b, held, memo));
            if z[0] & 1 != 0 {
                pointers(&mut acc, &y);
            }
            if y[0] & 1 != 0 {
                pointers(&mut acc, &z);
            }
            if y[0] & z[0] & 1 != 0 || (points(&y) && points(&z)) {
                acc[0] |= 1;
            }
        }
        Some(Op::Int(IntOp::Sub, a, b)) => {
            let (y, z) = (under(sets, a, held, memo), under(sets, b, held, memo));
            if z[0] & 1 != 0 || (points(&y) && points(&z)) {
                pointers(&mut acc, &y);
            }
            if y[0] & 1 != 0 || (points(&y) && points(&z)) {
                acc[0] |= 1;
            }
        }
        Some(Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Trunc | Cvt::Bitcast, to, a))
            if to.bits() >= 32 && f.types[a.0].bits() >= 32 =>
        {
            acc = under(sets, a, held, memo);
        }
        Some(Op::Pack64(lo, _) | Op::UnpackLo(lo)) => acc = under(sets, lo, held, memo),
        Some(Op::UnpackHi(y)) => match facts.op(f, y) {
            Some(Op::Pack64(_, hi)) => acc = under(sets, hi, held, memo),
            _ => acc[0] |= 1,
        },
        Some(Op::Int(IntOp::And, a, b)) => match (program.constant(a), program.constant(b)) {
            (None, Some(m)) if aligns(m) => acc = under(sets, a, held, memo),
            (Some(m), None) if aligns(m) => acc = under(sets, b, held, memo),
            (None, None) => {
                let (y, z) = (under(sets, a, held, memo), under(sets, b, held, memo));
                if z[0] & 1 != 0 {
                    pointers(&mut acc, &y);
                }
                if y[0] & 1 != 0 {
                    pointers(&mut acc, &z);
                }
                acc[0] |= 1;
            }
            _ => acc[0] |= 1,
        },
        Some(Op::Int(IntOp::LShr, a, _)) if f.types[x.0] != Ty::I64 => {
            merge(&mut acc, &under(sets, a, held, memo));
            acc[0] |= 1;
        }
        _ => acc.copy_from_slice(sets.of(x)),
    }
    memo.insert(x, acc.clone());
    acc
}

#[cfg(test)]
mod tests {
    use super::super::super::address::{Addresses, Region};
    use super::super::super::hazard::Hazards;
    use super::super::super::testing::*;
    use crate::rdna_spmd::analysis::facts::Facts;
    use crate::rdna_spmd::analysis::loops::Loops;
    use crate::rdna_spmd::environment::Environment;
    use crate::rdna_spmd::ir::*;
    use std::collections::BTreeSet;

    fn addresses<T>(b: &Build, env: &Environment, f: impl FnOnce(&mut Addresses) -> T) -> T {
        let facts = Facts::new(&b.f, &b.inputs, &BTreeSet::new());
        let loops = Loops::new(&b.f, &facts).expect("a reducible test program");
        let headers: BTreeSet<BlockId> = (0..loops.count()).map(|l| facts.order[loops.header(l)]).collect();
        let mut a = Addresses::new(&b.f, &facts, &b.inputs, EXEC, b.entry, env, headers, &b.registry);
        a.enter(0);
        f(&mut a)
    }

    fn store_at(b: &mut Build, block: BlockId, address: ValueId, mask: ValueId) -> (BlockId, usize) {
        let zero = b.constant(block, Ty::I32, 0);
        let at = b.here(block);
        b.store(block, Space::Global, MemSize::B32, address, zero, mask);
        at
    }

    fn pair(h: &Hazards, p: (BlockId, usize), q: (BlockId, usize)) -> (usize, usize) {
        let at = |x: (BlockId, usize)| h.accesses.iter().position(|a| (a.block, a.index) == x).unwrap();
        let (p, q) = (at(p), at(q));
        (p.min(q), p.max(q))
    }

    fn two_buffers() -> Environment {
        environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000)])
    }

    struct Moved {
        b: Build,
        moved: ValueId,
        second: ValueId,
        exec: ValueId,
    }

    fn moved_by(build: impl Fn(&mut Build, BlockId, ValueId, ValueId) -> ValueId) -> Moved {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let moved = build(&mut b, e, first, second);
        Moved { b, moved, second, exec: k.exec }
    }

    fn difference(b: &mut Build, e: BlockId, first: ValueId, second: ValueId) -> ValueId {
        let gap = b.int(e, IntOp::Sub, second, first);
        b.int(e, IntOp::Add, first, gap)
    }

    fn cancelled(b: &mut Build, e: BlockId, first: ValueId, second: ValueId) -> ValueId {
        let zero = b.int(e, IntOp::Xor, first, first);
        b.int(e, IntOp::Add, second, zero)
    }

    #[test]
    fn a_pointer_moved_by_the_difference_of_two_buffers_may_point_into_the_second() {
        let Moved { b, moved, .. } = moved_by(difference);
        let set = addresses(&b, &two_buffers(), |a| a.regions(moved, 0, None, true));
        assert!(
            set.reaches(Some(Region::Allocation(2)), |a, b| a == b) || set.lost(),
            "first + (second - first) is second, so it must reach second or be lost, but its regions are {:?}",
            set
        );
    }

    #[test]
    fn lanes_storing_through_the_difference_of_two_buffers_meet_at_the_second() {
        let Moved { mut b, moved, second, exec } = moved_by(difference);
        let e = BlockId(0);
        let s1 = store_at(&mut b, e, moved, exec);
        let s2 = store_at(&mut b, e, second, exec);
        let h = Hazards::find(&b.program(), &two_buffers());
        assert!(
            h.together.contains(&pair(&h, s1, s2)),
            "both stores write word 0 of the second buffer in every lane"
        );
    }

    #[test]
    fn lanes_storing_through_a_pointer_plus_a_cancelled_xor_meet_at_it() {
        let Moved { mut b, moved, second, exec } = moved_by(cancelled);
        let e = BlockId(0);
        let s1 = store_at(&mut b, e, moved, exec);
        let s2 = store_at(&mut b, e, second, exec);
        let h = Hazards::find(&b.program(), &two_buffers());
        assert!(h.together.contains(&pair(&h, s1, s2)), "both stores write word 0 of the second buffer in every lane");
    }

    fn meetings_before_the_reload(through_difference: bool) -> (Vec<(BlockId, usize)>, (BlockId, usize), usize) {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let first = k.buffer(&mut b, e, 0);
        let second = k.buffer(&mut b, e, 8);
        let out = k.buffer(&mut b, e, 16);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let target = if through_difference { difference(&mut b, e, first, second) } else { second };
        b.store(e, Space::Global, MemSize::B32, target, lane, k.exec);
        let load = b.here(e);
        let v = b.load(e, Space::Global, MemSize::B32, second, k.exec);
        let own = byte_offset(&mut b, e, out, lane, 4);
        b.store(e, Space::Global, MemSize::B32, own, v, k.exec);
        let program = b.program();
        let env = environment(32, &[(0, 1, 0x1000), (8, 2, 0x2000), (16, 3, 0x3000)]);
        let hazards = Hazards::find(&program, &env);
        let (kept, _) = super::super::super::search::prove(&program.ir, &program.parameter_inputs, Some(0), &hazards);
        let before = kept.meets.iter().map(|&m| hazards.meetings[m]).collect();
        (before, load, hazards.conflicts().len())
    }

    #[test]
    fn decompile_orders_a_load_after_stores_straight_through_the_second_buffer() {
        let (before, load, _) = meetings_before_the_reload(false);
        assert_eq!(before, vec![load], "the control case: the same program storing through second itself");
    }

    #[test]
    fn decompile_orders_a_load_after_stores_through_the_difference_of_two_buffers() {
        let (before, load, conflicts) = meetings_before_the_reload(true);
        assert!(
            before.contains(&load),
            "every lane must read the id the last lane stored, but the lane program keeps meetings only before {:?} ({} conflicts found)",
            before,
            conflicts
        );
    }

    #[test]
    fn lanes_storing_through_a_pointer_shifted_by_64_meet_at_it() {
        let mut missed = Vec::new();
        for op in [IntOp::LShr, IntOp::AShr] {
            let (mut b, k) = Build::kernel();
            let e = BlockId(0);
            let buf = k.buffer(&mut b, e, 0);
            let sixty_four = b.constant(e, Ty::I64, 64);
            let shifted = b.int(e, op, buf, sixty_four);
            let s1 = store_at(&mut b, e, shifted, k.exec);
            let s2 = store_at(&mut b, e, buf, k.exec);
            let set = addresses(&b, &two_buffers(), |a| a.regions(shifted, 0, None, true));
            let h = Hazards::find(&b.program(), &two_buffers());
            if !h.together.contains(&pair(&h, s1, s2)) {
                missed.push((op, set));
            }
        }
        assert!(missed.is_empty(), "buf shifted by 64 is buf, where every lane stores next, but these miss it: {:?}", missed);
    }

    #[test]
    fn lanes_storing_through_a_buffer_read_via_a_laundered_kernarg_pointer_meet_at_it() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let base = b.core(e, Ty::I64, Op::Pack64(k.kernarg.0, k.kernarg.1));
        let one = b.constant(e, Ty::I64, 1);
        let laundered = b.int(e, IntOp::Mul, base, one);
        let yes = b.constant(e, Ty::I1, 1);
        let reread = b.load(e, Space::Global, MemSize::B64, laundered, yes);
        let s1 = store_at(&mut b, e, reread, k.exec);
        let s2 = store_at(&mut b, e, buf, k.exec);
        let set = addresses(&b, &two_buffers(), |a| a.regions(reread, 0, None, true));
        let h = Hazards::find(&b.program(), &two_buffers());
        assert!(
            h.together.contains(&pair(&h, s1, s2)),
            "both stores write word 0 of the first buffer in every lane; the re-read pointer has regions {:?}",
            set
        );
    }
}
