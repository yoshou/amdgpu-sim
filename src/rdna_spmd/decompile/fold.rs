use crate::rdna_spmd::analysis::bdd::Bdd;
use crate::rdna_spmd::analysis::facts::Facts;
use super::logic::{Atom, Logic};
use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};

pub fn fold(q: &mut Func, inputs: &[Parameter], words: &BTreeSet<ValueId>, exec: Option<usize>) {
    let decided = {
        let facts = Facts::new(q, inputs, words);

        let kept: BTreeSet<ValueId> = q
            .blocks
            .values()
            .flat_map(|b| &b.insts)
            .filter_map(|inst| match inst {
                Inst::Effect {
                    op: EffectOp::Wave(WaveOp::Any),
                    outputs,
                    ..
                } => Some(outputs[0].0),
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
