use super::program::Program;
use crate::rdna_spmd::hash::HashMap;
use crate::rdna_spmd::ir::*;

#[derive(Clone, Copy, PartialEq)]
pub(super) enum Demand {
    Dead,
    Under(ValueId),
    All,
}

impl Demand {
    pub(super) fn meet(self, other: Demand) -> Demand {
        match (self, other) {
            (Demand::Dead, x) | (x, Demand::Dead) => x,
            (Demand::Under(x), Demand::Under(y)) if x == y => self,
            _ => Demand::All,
        }
    }
}

pub(super) fn demands(program: &Program) -> Vec<Demand> {
    let (f, facts) = (program.f, program.facts);
    let mut demand = vec![Demand::Dead; f.types.len()];
    let mut held: HashMap<ValueId, Vec<ValueId>> = HashMap::default();
    loop {
        let mut changed = false;
        for b in facts.order.iter().rev() {
            let block = &f.blocks[b];
            for edge in block.term.edges() {
                for (&x, &(p, _)) in edge.args.iter().zip(&f.blocks[&edge.dst].params) {
                    let d = demand[p.0];
                    changed |= use_as(&mut demand, x, d);
                }
            }
            for inst in block.insts.iter().rev() {
                let mut out = Demand::Dead;
                inst.for_each_output(|o| out = out.meet(demand[o.0]));
                match inst {
                    Inst::Core { ty: Ty::I1, .. }
                    | Inst::Core {
                        op: Op::TrailingZeros(_) | Op::LeadingZeros(_) | Op::PopulationCount(_),
                        ..
                    } => {}
                    Inst::Core {
                        op: Op::Select(c, a, b),
                        ..
                    } => {
                        let arm = if out == Demand::Dead { out } else { Demand::Under(*c) };
                        let other = match out {
                            Demand::Under(k) => {
                                let root = program.copies.get(c).copied().unwrap_or(*c);
                                let conjuncts = held.entry(k).or_insert_with(|| program.conjuncts(k));
                                if conjuncts.contains(&root) {
                                    Demand::Dead
                                } else {
                                    out
                                }
                            }
                            _ => out,
                        };
                        changed |= use_as(&mut demand, *a, arm);
                        changed |= use_as(&mut demand, *b, other);
                    }
                    Inst::Core { .. } => inst.for_each_operand(|x| changed |= use_as(&mut demand, x, out)),
                    Inst::Target { op, .. } if program.pure(*op) => {
                        inst.for_each_operand(|x| changed |= use_as(&mut demand, x, out))
                    }
                    Inst::Effect {
                        op: EffectOp::Memory { op, .. },
                        inputs,
                        ..
                    } if inputs.len() > op.mask_input() => {
                        let m = op.mask_input();
                        for &x in &inputs[..m] {
                            changed |= use_as(&mut demand, x, Demand::Under(inputs[m]));
                        }
                    }
                    _ => inst.for_each_operand(|x| changed |= use_as(&mut demand, x, Demand::All)),
                }
            }
        }
        if !changed {
            return demand;
        }
    }
}

pub(super) fn uses(program: &Program) -> Vec<bool> {
    let (f, facts) = (program.f, program.facts);
    let mut used = vec![false; f.types.len()];
    for b in &facts.order {
        let block = &f.blocks[b];
        for edge in block.term.edges() {
            for &x in &edge.args {
                used[x.0] = true;
            }
        }
        for inst in &block.insts {
            match inst {
                Inst::Core { ty: Ty::I1, .. }
                | Inst::Core {
                    op: Op::TrailingZeros(_) | Op::LeadingZeros(_) | Op::PopulationCount(_),
                    ..
                } => {}
                Inst::Core {
                    op: Op::Select(_, a, b),
                    ..
                } => {
                    used[a.0] = true;
                    used[b.0] = true;
                }
                Inst::Effect {
                    op: EffectOp::Memory { op, .. },
                    inputs,
                    ..
                } if inputs.len() > op.mask_input() => {
                    for &x in &inputs[..op.mask_input()] {
                        used[x.0] = true;
                    }
                }
                _ => inst.for_each_operand(|x| used[x.0] = true),
            }
        }
    }
    used
}

fn use_as(demand: &mut [Demand], x: ValueId, d: Demand) -> bool {
    let met = demand[x.0].meet(d);
    let changed = met != demand[x.0];
    demand[x.0] = met;
    changed
}
