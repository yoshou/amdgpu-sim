use super::super::analysis::Analyses;
use super::super::ir::*;
use std::collections::{BTreeMap, BTreeSet};

pub struct BranchSelects;
impl super::Pass for BranchSelects {
    fn name(&self) -> &str {
        "branch_selects"
    }
    fn run(&self, f: &mut Func, _: &Analyses) -> bool {
        run(f) > 0
    }
}

fn root(defs: &BTreeMap<ValueId, Op>, mut v: ValueId, types: &[Ty]) -> (ValueId, bool) {
    let mut flipped = false;
    loop {
        let constant = |x: ValueId| match defs.get(&x) {
            Some(Op::Const(_, k)) => Some(*k),
            _ => None,
        };
        match defs.get(&v) {
            Some(Op::Convert(Cvt::Bitcast, Ty::I1, x)) if types[x.0] == Ty::I1 => v = *x,
            Some(Op::Cmp(pred @ (IntPred::Eq | IntPred::Ne), x, y)) if types[x.0] == Ty::I1 => {
                let (x, k) = match (constant(*x), constant(*y)) {
                    (_, Some(k)) => (*x, k),
                    (Some(k), _) => (*y, k),
                    _ => return (v, flipped),
                };
                flipped ^= (*pred == IntPred::Eq) == (k == 0);
                v = x;
            }
            Some(Op::Int(IntOp::Xor, x, y)) if types[v.0] == Ty::I1 => {
                let x = match (constant(*x), constant(*y)) {
                    (_, Some(1)) => *x,
                    (Some(1), _) => *y,
                    _ => return (v, flipped),
                };
                flipped = !flipped;
                v = x;
            }
            _ => return (v, flipped),
        }
    }
}

struct Edge<'a> {
    f: &'a mut Func,
    defs: &'a BTreeMap<ValueId, Op>,
    local: &'a BTreeSet<ValueId>,
    cond: (ValueId, bool),
    taken: bool,
    memo: BTreeMap<ValueId, ValueId>,
    fresh: Vec<Inst>,
}

impl Edge<'_> {
    fn resolve(&mut self, v: ValueId) -> ValueId {
        if let Some(&r) = self.memo.get(&v) {
            return r;
        }
        let result = match self.defs.get(&v) {
            Some(op) if self.local.contains(&v) => {
                let resolved = match *op {
                    Op::Select(c, a, b) => {
                        let (c, flipped) = root(self.defs, c, &self.f.types);
                        (c == self.cond.0)
                            .then(|| self.resolve(if (flipped ^ self.cond.1) != self.taken { a } else { b }))
                    }
                    _ => None,
                };
                match resolved {
                    Some(r) => r,
                    None => {
                        let mapped = op.map(|x| self.resolve(x));
                        if mapped == *op {
                            v
                        } else {
                            let ty = self.f.types[v.0];
                            let value = self.f.value(ty);
                            self.fresh.push(Inst::Core {
                                value,
                                ty,
                                op: mapped,
                            });
                            value
                        }
                    }
                }
            }
            _ => v,
        };
        self.memo.insert(v, result);
        result
    }
}

fn run(f: &mut Func) -> usize {
    let mut defs = BTreeMap::new();
    for block in f.blocks.values() {
        for inst in &block.insts {
            if let Inst::Core { value, op, .. } = inst {
                defs.insert(*value, *op);
            }
        }
    }
    let ids: Vec<BlockId> = f.blocks.keys().copied().collect();
    let mut changed = 0;
    for id in ids {
        let Term::CondBr { cond, yes, no } = f.blocks[&id].term.clone() else {
            continue;
        };
        let local: BTreeSet<ValueId> = f.blocks[&id]
            .insts
            .iter()
            .filter_map(|inst| match inst {
                Inst::Core { value, .. } => Some(*value),
                _ => None,
            })
            .collect();
        let cond = root(&defs, cond, &f.types);
        let mut fresh = Vec::new();
        let mut edges = [yes, no];
        for (edge, taken) in edges.iter_mut().zip([true, false]) {
            let mut e = Edge {
                f,
                defs: &defs,
                local: &local,
                cond,
                taken,
                memo: BTreeMap::new(),
                fresh: Vec::new(),
            };
            for arg in &mut edge.args {
                let r = e.resolve(*arg);
                if r != *arg {
                    *arg = r;
                    changed += 1;
                }
            }
            fresh.extend(e.fresh);
        }
        if fresh.is_empty() && changed == 0 {
            continue;
        }
        let block = f.blocks.get_mut(&id).unwrap();
        block.insts.extend(fresh);
        let [yes, no] = edges;
        block.term = Term::CondBr {
            cond: match block.term {
                Term::CondBr { cond, .. } => cond,
                _ => unreachable!(),
            },
            yes,
            no,
        };
    }
    changed
}
