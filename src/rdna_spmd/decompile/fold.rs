use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use super::logic::{Atom, Choice, Logic};
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};

pub fn fold(q: &mut Func, inputs: &[Parameter], words: &BTreeSet<ValueId>, exec: Option<usize>) {
    let decided = {
        let facts = Facts::new(q, inputs, words);

        let kept: BTreeSet<Choice> = q
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
        let mut logic = Logic::fixed(q, &facts, &kept, &[]);
        let start = match exec {
            Some(index) => logic.atom(Atom::Bit(q.blocks[&q.entry].params[index].0)),
            None => Bdd::TRUE,
        };

        let reach = logic.reach(q, &facts, q.entry, start);
        decisions(q, &facts, &mut logic, &reach)
    };
    apply(q, &decided);
    remove_unreachable(q);
    remove_dead(q);
}

struct Decided {
    constant: BTreeMap<ValueId, bool>,
    reachable: BTreeSet<BlockId>,
}

fn decisions(
    q: &Func,
    facts: &Facts,
    logic: &mut Logic,
    reach: &BTreeMap<BlockId, Bdd>,
) -> Decided {
    let mut constant = BTreeMap::new();
    for &id in &facts.order {
        let r = reach.get(&id).copied().unwrap_or(Bdd::FALSE);
        if r == Bdd::FALSE {
            continue;
        }
        let block = &q.blocks[&id];
        let values = block
            .params
            .iter()
            .map(|p| p.0)
            .chain(block.insts.iter().flat_map(Inst::outputs));
        for v in values {
            if q.types[v.0] != Ty::I1 {
                continue;
            }
            if matches!(facts.op(q, v), Some(Op::Const(..))) {
                continue;
            }
            let g = logic.bit(q, facts, v);
            if logic.m.implies(r, g) {
                constant.insert(v, true);
            } else {
                let ng = logic.m.not(g);
                if logic.m.implies(r, ng) {
                    constant.insert(v, false);
                }
            }
        }
    }
    let reachable = reach
        .iter()
        .filter(|(_, &r)| r != Bdd::FALSE)
        .map(|(&b, _)| b)
        .collect();
    Decided {
        constant,
        reachable,
    }
}

fn apply(q: &mut Func, decided: &Decided) {
    let mut renames: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let ids: Vec<BlockId> = q.blocks.keys().copied().collect();
    for id in ids {
        if !decided.reachable.contains(&id) {
            continue;
        }
        let mut block = q.blocks.remove(&id).unwrap();
        let mut insts = Vec::with_capacity(block.insts.len());
        let mut head = Vec::new();
        for &(param, _) in &block.params {
            if let Some(&k) = decided.constant.get(&param) {
                let value = q.value(Ty::I1);
                head.push(Inst::Core {
                    value,
                    ty: Ty::I1,
                    op: Op::Const(Ty::I1, k as u64),
                });
                renames.insert(param, value);
            }
        }
        insts.extend(head);
        for inst in block.insts.drain(..) {
            match inst {
                Inst::Core { value, ty, .. } if decided.constant.contains_key(&value) => {
                    insts.push(Inst::Core {
                        value,
                        ty,
                        op: Op::Const(Ty::I1, decided.constant[&value] as u64),
                    });
                }
                Inst::Effect { outputs, .. }
                    if outputs.len() == 1 && decided.constant.contains_key(&outputs[0].0) =>
                {
                    let value = outputs[0].0;
                    insts.push(Inst::Core {
                        value,
                        ty: Ty::I1,
                        op: Op::Const(Ty::I1, decided.constant[&value] as u64),
                    });
                }
                other => insts.push(other),
            }
        }
        block.insts = insts;
        q.blocks.insert(id, block);
    }
    super::rewrite::rename(q, &renames);
    simplify(q);
}

fn simplify(q: &mut Func) {
    let constants: BTreeMap<ValueId, u64> = q
        .blocks
        .values()
        .flat_map(|b| &b.insts)
        .filter_map(|inst| match inst {
            Inst::Core {
                value,
                op: Op::Const(_, k),
                ..
            } => Some((*value, *k)),
            _ => None,
        })
        .collect();
    let mut renames = BTreeMap::new();
    for block in q.blocks.values_mut() {
        block.insts.retain(|inst| match inst {
            Inst::Effect {
                op:
                    EffectOp::Memory {
                        op: MemoryOp::Store(_),
                        ..
                    },
                inputs,
                ..
            } => constants.get(&inputs[2]) != Some(&0),
            _ => true,
        });
        for inst in &block.insts {
            if let Inst::Core {
                value,
                op: Op::Select(c, a, b),
                ..
            } = inst
            {
                match constants.get(c) {
                    Some(0) => {
                        renames.insert(*value, *b);
                    }
                    Some(_) => {
                        renames.insert(*value, *a);
                    }
                    None if a == b => {
                        renames.insert(*value, *a);
                    }
                    None => {}
                }
            }
        }
        if let Term::CondBr { cond, yes, no } = &block.term {
            match constants.get(cond) {
                Some(0) => block.term = Term::Br(no.clone()),
                Some(_) => block.term = Term::Br(yes.clone()),
                None => {}
            }
        }
    }
    super::rewrite::rename(q, &renames);
}

fn remove_unreachable(q: &mut Func) {
    let reachable: BTreeSet<BlockId> = q.reverse_postorder().into_iter().collect();
    q.blocks.retain(|id, _| reachable.contains(id));
}

fn remove_dead(q: &mut Func) {
    loop {
        let mut used = vec![false; q.types.len()];
        for block in q.blocks.values() {
            for inst in &block.insts {
                for v in inst.operands() {
                    used[v.0] = true;
                }
            }
            match &block.term {
                Term::CondBr { cond, .. } => used[cond.0] = true,
                Term::Ret(args) => {
                    for v in args {
                        used[v.0] = true;
                    }
                }
                Term::Br(_) => {}
            }
            for edge in block.term.edges() {
                for v in &edge.args {
                    used[v.0] = true;
                }
            }
        }
        let mut removed = false;
        for block in q.blocks.values_mut() {
            let before = block.insts.len();
            block.insts.retain(|inst| {
                let observable = match inst {
                    Inst::Effect { op, .. } => !matches!(
                        op,
                        EffectOp::Memory {
                            op: MemoryOp::Load(_),
                            ..
                        }
                    ),
                    _ => false,
                };
                observable || inst.outputs().iter().any(|v| used[v.0])
            });
            removed |= block.insts.len() != before;
        }
        if !removed {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::testing::*;
    use super::*;

    struct Branched {
        f: Func,
        inputs: Vec<Parameter>,
        then: BlockId,
        other: BlockId,
        join: BlockId,
        carried: ValueId,
        joined: ValueId,
    }

    fn branched() -> Branched {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let five = b.constant(e, Ty::I32, 5);
        let c = b.cmp(e, IntPred::Ult, lane, five);
        let (then, t) = b.block(&[Ty::I1, Ty::I1, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I1, Ty::I64]);
        let (join, j) = b.block(&[Ty::I1, Ty::I1, Ty::I64]);
        b.cond_br(e, c, (then, vec![k.exec, c, buf]), (other, vec![k.exec, c, buf]));
        let one = b.constant(then, Ty::I32, 1);
        let two = b.constant(then, Ty::I32, 2);
        let data = b.core(then, Ty::I32, Op::Select(t[1], one, two));
        let mask = b.int(then, IntOp::And, t[1], t[0]);
        b.store(then, Space::Global, MemSize::B32, t[2], data, mask);
        b.br(then, join, vec![t[0], t[1], t[2]]);
        let three = b.constant(other, Ty::I32, 3);
        let mask = b.int(other, IntOp::And, o[1], o[0]);
        b.store(other, Space::Global, MemSize::B32, o[2], three, mask);
        b.br(other, join, vec![o[0], o[1], o[2]]);
        let four = b.constant(join, Ty::I32, 4);
        let mask = b.int(join, IntOp::And, j[1], j[0]);
        b.store(join, Space::Global, MemSize::B32, j[2], four, mask);
        Branched {
            f: b.f,
            inputs: b.inputs,
            then,
            other,
            join,
            carried: t[1],
            joined: j[1],
        }
    }

    fn stores(f: &Func, block: BlockId) -> Vec<(ValueId, ValueId)> {
        f.blocks[&block]
            .insts
            .iter()
            .filter_map(|inst| match inst {
                Inst::Effect {
                    op: EffectOp::Memory { op: MemoryOp::Store(_), .. },
                    inputs,
                    ..
                } => Some((inputs[1], inputs[2])),
                _ => None,
            })
            .collect()
    }

    fn constant(f: &Func, v: ValueId) -> Option<u64> {
        f.blocks.values().flat_map(|b| &b.insts).find_map(|inst| match inst {
            Inst::Core { value, op: Op::Const(_, k), .. } if *value == v => Some(*k),
            _ => None,
        })
    }

    #[test]
    fn fold_decides_the_condition_inside_each_arm() {
        let Branched { mut f, inputs, then, other, carried, .. } = branched();
        fold(&mut f, &inputs, &BTreeSet::new(), Some(0));
        let taken = stores(&f, then);
        assert_eq!(taken.len(), 1);
        assert_eq!(constant(&f, taken[0].0), Some(1), "the arm runs only when the condition holds, so the select picks 1");
        assert!(!f.blocks[&then].insts.iter().any(|i| i.operands().contains(&carried)), "nothing in the arm still reads the condition");
        assert!(stores(&f, other).is_empty(), "the other arm's store is masked by a condition that is false there");
    }

    #[test]
    fn fold_keeps_a_bit_that_differs_between_the_paths_into_a_join() {
        let Branched { mut f, inputs, join, joined, .. } = branched();
        fold(&mut f, &inputs, &BTreeSet::new(), Some(0));
        let kept = stores(&f, join);
        assert_eq!(kept.len(), 1, "the store after the join runs on one path only");
        let mask = kept[0].1;
        assert_eq!(constant(&f, mask), None);
        assert!(
            f.blocks[&join].insts.iter().any(|i| i.operands().contains(&joined)),
            "the joined condition is true on one path and false on the other"
        );
    }

    #[test]
    fn fold_removes_an_arm_the_entry_decides_against() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let (then, t) = b.block(&[Ty::I1, Ty::I64]);
        let (other, o) = b.block(&[Ty::I1, Ty::I64]);
        b.cond_br(e, k.exec, (then, vec![k.exec, buf]), (other, vec![k.exec, buf]));
        let one = b.constant(then, Ty::I32, 1);
        b.store(then, Space::Global, MemSize::B32, t[1], one, t[0]);
        let two = b.constant(other, Ty::I32, 2);
        b.store(other, Space::Global, MemSize::B32, o[1], two, o[0]);
        let mut f = b.f;
        fold(&mut f, &b.inputs, &BTreeSet::new(), Some(0));
        assert!(!f.blocks.contains_key(&other), "every lane that runs has exec set at the entry");
        assert!(matches!(f.blocks[&e].term, Term::Br(Edge { dst, .. }) if dst == then));
    }

    #[test]
    fn fold_removes_unused_values_and_loads_and_keeps_every_other_effect() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let unused = b.int(e, IntOp::Add, lane, lane);
        let loaded = b.load(e, Space::Global, MemSize::B32, buf, k.exec);
        let five = b.constant(e, Ty::I32, 5);
        let c = b.cmp(e, IntPred::Ult, lane, five);
        let q = b.wave(e, WaveOp::Any, vec![c]);
        b.effect(e, EffectOp::Wave(WaveOp::Meet), vec![]);
        b.store(e, Space::Global, MemSize::B32, buf, lane, k.exec);
        let mut f = b.f;
        fold(&mut f, &b.inputs, &BTreeSet::new(), Some(0));
        let outputs: Vec<ValueId> = f.blocks[&e].insts.iter().flat_map(Inst::outputs).collect();
        assert!(!outputs.contains(&unused));
        assert!(!outputs.contains(&loaded), "a load nothing reads has no effect on memory");
        assert!(outputs.contains(&q), "a kept query is a collective every lane meets at");
        assert!(f.blocks[&e].insts.iter().any(|i| matches!(i, Inst::Effect { op: EffectOp::Wave(WaveOp::Meet), .. })));
        assert_eq!(stores(&f, e).len(), 1);
    }

    #[test]
    fn fold_keeps_an_atomic_and_a_store_whose_masks_it_cannot_decide() {
        let (mut b, k) = Build::kernel();
        let e = BlockId(0);
        let buf = k.buffer(&mut b, e, 0);
        let lane = b.core(e, Ty::I32, Op::Env(Env::LaneId));
        let five = b.constant(e, Ty::I32, 5);
        let c = b.cmp(e, IntPred::Ult, lane, five);
        let mask = b.int(e, IntOp::And, c, k.exec);
        b.store(e, Space::Global, MemSize::B32, buf, lane, mask);
        b.effect(e, memory(Space::Global, MemoryOp::AtomicAdd(Numeric::Unsigned)), vec![buf, lane, mask]);
        let mut f = b.f;
        fold(&mut f, &b.inputs, &BTreeSet::new(), Some(0));
        assert_eq!(stores(&f, e).len(), 1);
        assert!(f.blocks[&e]
            .insts
            .iter()
            .any(|i| matches!(i, Inst::Effect { op: EffectOp::Memory { op: MemoryOp::AtomicAdd(_), .. }, .. })));
    }

    #[test]
    fn fold_keeps_a_bit_that_a_self_loop_changes() {
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
        let mut f = b.f;
        fold(&mut f, &b.inputs, &BTreeSet::new(), Some(0));
        let kept = f.blocks[&body].insts.iter().any(|inst| {
            matches!(inst, Inst::Core { op: Op::Int(IntOp::And, x, _), .. } if *x == q[1])
        });
        assert!(
            kept,
            "the carried bit is true on entry and false after an iteration whose fresh bit is false, but fold made it constant: {:?}",
            f.blocks[&body].insts
        );
    }
}
