//! Function SSA with explicit block arguments, including loop backedges.
//!
//! During incremental lifting, a Boundary records the typed inputs/outputs of
//! one adapter action (register view, predicated register update, or legacy
//! instruction). It is ordered and cannot be speculated, removed or reordered.
//! It is deliberately not a core Op or a target-dialect escape hatch. Its owner
//! must retain the matching adapter action until memory/wave lifting replaces it.
use super::{Op, Ty, ValueId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BlockId(pub usize);
#[derive(Clone, Debug)]
pub(crate) enum Inst {
    Core {
        value: ValueId,
        ty: Ty,
        op: Op,
    },
    Boundary {
        inputs: Vec<ValueId>,
        outputs: Vec<(ValueId, Ty)>,
    },
}
#[derive(Clone, Debug)]
pub(crate) struct Edge {
    pub dst: BlockId,
    pub args: Vec<ValueId>,
}
#[derive(Clone, Debug)]
pub(crate) enum Term {
    Br(Edge),
    CondBr { cond: ValueId, yes: Edge, no: Edge },
    Ret,
}
impl Term {
    pub fn edges(&self) -> Vec<&Edge> {
        match self {
            Self::Br(e) => vec![e],
            Self::CondBr { yes, no, .. } => vec![yes, no],
            Self::Ret => vec![],
        }
    }
}
#[derive(Clone, Debug)]
pub(crate) struct Block {
    pub params: Vec<(ValueId, Ty)>,
    pub insts: Vec<Inst>,
    pub term: Term,
}
#[derive(Clone, Debug)]
pub(crate) struct Func {
    pub entry: BlockId,
    pub blocks: BTreeMap<BlockId, Block>,
    pub types: Vec<Ty>,
}
pub(crate) struct VerifiedFunc(Func);
impl Func {
    pub fn value(&mut self, ty: Ty) -> ValueId {
        let v = ValueId(self.types.len());
        self.types.push(ty);
        v
    }
    pub fn verify(self) -> Result<VerifiedFunc, &'static str> {
        if !self.blocks.contains_key(&self.entry) {
            return Err("missing entry");
        }
        let mut definitions = BTreeSet::new();
        for block in self.blocks.values() {
            let mut local = BTreeSet::new();
            let define = |id: ValueId,
                          ty: Ty,
                          local: &mut BTreeSet<ValueId>,
                          all: &mut BTreeSet<ValueId>| {
                if self.types.get(id.0) != Some(&ty) {
                    return Err("invalid value type");
                }
                if !all.insert(id) {
                    return Err("duplicate SSA definition");
                }
                local.insert(id);
                Ok(())
            };
            for &(id, ty) in &block.params {
                define(id, ty, &mut local, &mut definitions)?;
            }
            for inst in &block.insts {
                match inst {
                    Inst::Core { value, ty, op } => {
                        let mut valid = true;
                        op.map(|v| {
                            valid &= local.contains(&v);
                            v
                        });
                        if !valid {
                            return Err("value does not dominate use; pass it as a block argument");
                        }
                        if op.result_type(&self.types)? != *ty {
                            return Err("incorrect core result type");
                        }
                        define(*value, *ty, &mut local, &mut definitions)?;
                    }
                    Inst::Boundary { inputs, outputs } => {
                        if inputs.iter().any(|v| !local.contains(v)) {
                            return Err("non-dominating boundary input");
                        }
                        for &(id, ty) in outputs {
                            define(id, ty, &mut local, &mut definitions)?;
                        }
                    }
                }
            }
            if let Term::CondBr { cond, .. } = block.term {
                if !local.contains(&cond) || self.types[cond.0] != Ty::I1 {
                    return Err("branch requires a dominating i1");
                }
            }
            for e in block.term.edges() {
                let dst = self.blocks.get(&e.dst).ok_or("missing branch target")?;
                if e.args.len() != dst.params.len() {
                    return Err("block argument count mismatch");
                }
                for (&arg, &(_, ty)) in e.args.iter().zip(&dst.params) {
                    if !local.contains(&arg) || self.types[arg.0] != ty {
                        return Err("invalid block argument");
                    }
                }
            }
        }
        if definitions.len() != self.types.len() {
            return Err("undefined SSA value");
        }
        Ok(VerifiedFunc(self))
    }
}
impl VerifiedFunc {
    pub fn func(&self) -> &Func {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::ir::typed::IntOp;
    fn loop_func() -> Func {
        Func {
            entry: BlockId(0),
            types: vec![Ty::I32, Ty::I32, Ty::I32, Ty::I1],
            blocks: BTreeMap::from([
                (
                    BlockId(0),
                    Block {
                        params: vec![(ValueId(0), Ty::I32)],
                        insts: vec![],
                        term: Term::Br(Edge {
                            dst: BlockId(1),
                            args: vec![ValueId(0)],
                        }),
                    },
                ),
                (
                    BlockId(1),
                    Block {
                        params: vec![(ValueId(1), Ty::I32)],
                        insts: vec![
                            Inst::Core {
                                value: ValueId(2),
                                ty: Ty::I32,
                                op: Op::Int(IntOp::Add, ValueId(1), ValueId(1)),
                            },
                            Inst::Core {
                                value: ValueId(3),
                                ty: Ty::I1,
                                op: Op::Const(Ty::I1, 1),
                            },
                        ],
                        term: Term::CondBr {
                            cond: ValueId(3),
                            yes: Edge {
                                dst: BlockId(1),
                                args: vec![ValueId(2)],
                            },
                            no: Edge {
                                dst: BlockId(2),
                                args: vec![],
                            },
                        },
                    },
                ),
                (
                    BlockId(2),
                    Block {
                        params: vec![],
                        insts: vec![],
                        term: Term::Ret,
                    },
                ),
            ]),
        }
    }
    #[test]
    fn verifies_loop_block_arguments() {
        loop_func().verify().unwrap();
    }
    #[test]
    fn rejects_broken_cfg_ssa() {
        for kind in 0..6 {
            let mut f = loop_func();
            let block = f.blocks.get_mut(&BlockId(1)).unwrap();
            match kind {
                0 => block.params[0].0 = ValueId(0),
                1 => {
                    block.insts[0] = Inst::Core {
                        value: ValueId(2),
                        ty: Ty::I32,
                        op: Op::Int(IntOp::Add, ValueId(0), ValueId(1)),
                    }
                }
                2 => {
                    block.term = Term::Br(Edge {
                        dst: BlockId(99),
                        args: vec![],
                    })
                }
                3 => {
                    block.term = Term::Br(Edge {
                        dst: BlockId(1),
                        args: vec![],
                    })
                }
                4 => {
                    block.term = Term::Br(Edge {
                        dst: BlockId(1),
                        args: vec![ValueId(3)],
                    })
                }
                _ => {
                    block.term = Term::CondBr {
                        cond: ValueId(2),
                        yes: Edge {
                            dst: BlockId(2),
                            args: vec![],
                        },
                        no: Edge {
                            dst: BlockId(2),
                            args: vec![],
                        },
                    }
                }
            }
            assert!(f.verify().is_err(), "case {}", kind);
        }
    }
}
