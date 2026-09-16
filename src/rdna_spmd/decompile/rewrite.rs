use super::facts::Facts;
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
        match facts.constant(p, v).map(|k| k as u32) {
            Some(0) => self.core(Ty::I1, Op::Const(Ty::I1, 0)),
            Some(u32::MAX) => self.core(Ty::I1, Op::Const(Ty::I1, 1)),
            _ => {
                let lane = self.lane_id();
                let shifted = self.core(Ty::I32, Op::Int(IntOp::LShr, v, lane));
                self.core(Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted))
            }
        }
    }
}

pub(super) fn rename(f: &mut Func, map: &std::collections::BTreeMap<ValueId, ValueId>) {
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

pub(super) fn converted(facts: &Facts, v: ValueId) -> bool {
    facts.lane_word[v.0] && !facts.materialized[v.0]
}

fn projection(p: &Func, facts: &Facts, s: ValueId) -> Option<ValueId> {
    match facts.op(p, s) {
        Some(Op::Int(IntOp::LShr, w, lane)) if facts.is_lane_id(p, lane) && converted(facts, w) => {
            Some(w)
        }
        _ => None,
    }
}

pub(super) fn lane_program(p: &Func, facts: &Facts) -> Func {
    let reachable: std::collections::BTreeSet<BlockId> = facts.order.iter().copied().collect();
    let mut q = Func {
        entry: p.entry,
        blocks: p
            .blocks
            .iter()
            .filter(|(id, _)| reachable.contains(id))
            .map(|(&id, b)| (id, b.clone()))
            .collect(),
        types: p.types.clone(),
    };
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
        for inst in old {
            rewrite_inst(p, facts, &mut b, inst);
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
    q
}

fn rewrite_inst(p: &Func, facts: &Facts, b: &mut Block, inst: Inst) {
    match inst {
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::Any),
            inputs,
            outputs,
            ..
        } => b.insts.push(Inst::Core {
            value: outputs[0].0,
            ty: Ty::I1,
            op: Op::Convert(Cvt::Bitcast, Ty::I1, inputs[0]),
        }),
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::Ballot),
            inputs,
            outputs,
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
                let word = b.core(Ty::I32, Op::Convert(Cvt::ZExt, Ty::I32, inputs[0]));
                let lane = b.lane_id();
                b.insts.push(Inst::Core {
                    value,
                    ty: Ty::I32,
                    op: Op::Int(IntOp::Shl, word, lane),
                });
            }
        }
        Inst::Effect {
            op: EffectOp::Wave(WaveOp::ReadFirstLane),
            inputs,
            outputs,
            ..
        } => b.insts.push(Inst::Core {
            value: outputs[0].0,
            ty: Ty::I32,
            op: Op::Convert(Cvt::Bitcast, Ty::I32, inputs[0]),
        }),
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
        } if super::logic::lane_test(p, facts, x, y).is_some() => {
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
                Op::Convert(Cvt::Bitcast, Ty::I32, x) => {
                    Op::Convert(Cvt::Bitcast, Ty::I1, b.bit(p, facts, x))
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
