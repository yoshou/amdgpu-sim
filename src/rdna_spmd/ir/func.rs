//! Function SSA with explicit block arguments, including loop backedges.
use super::{Op, Ty, ValueId};
use super::effect::EffectOp;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BlockId(pub usize);
/// Pure mask reductions within the current packet. These are compiler IR
/// operations, not variants of the runtime's wave-effect protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PacketOp { Any, Ballot }
impl PacketOp {
    pub fn result_type(self) -> Ty { match self { Self::Any=>Ty::I1,Self::Ballot=>Ty::I32 } }
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Inst {
    Packet {
        op: PacketOp,
        input: ValueId,
        output: ValueId,
    },
    Target {
        provenance: Option<u64>,
        op: crate::rdna_spmd::dialect::TargetOp,
        args: crate::rdna_spmd::dialect::Arguments,
        outputs: Vec<(ValueId, Ty)>,
    },
    Effect {
        provenance: u64,
        op: EffectOp,
        inputs: Vec<ValueId>,
        outputs: Vec<(ValueId, Ty)>,
    },
    Core {
        value: ValueId,
        ty: Ty,
        op: Op,
    },
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Edge {
    pub dst: BlockId,
    pub args: Vec<ValueId>,
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Term {
    Br(Edge),
    CondBr { cond: ValueId, yes: Edge, no: Edge },
    Ret(Vec<ValueId>),
}
impl Term {
    pub fn edges(&self) -> impl Iterator<Item = &Edge> {
        let (first, second) = match self {
            Self::Br(e) => (Some(e), None),
            Self::CondBr { yes, no, .. } => (Some(yes), Some(no)),
            Self::Ret(_) => (None, None),
        };
        first.into_iter().chain(second)
    }
    pub fn edges_mut(&mut self) -> impl Iterator<Item = &mut Edge> {
        let (first, second) = match self {
            Self::Br(e) => (Some(e), None),
            Self::CondBr { yes, no, .. } => (Some(yes), Some(no)),
            Self::Ret(_) => (None, None),
        };
        first.into_iter().chain(second)
    }
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Block {
    pub params: Vec<(ValueId, Ty)>,
    pub insts: Vec<Inst>,
    pub term: Term,
}
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Func {
    pub entry: BlockId,
    pub blocks: BTreeMap<BlockId, Block>,
    pub types: Vec<Ty>,
}
impl Func {
    pub fn value(&mut self, ty: Ty) -> ValueId {
        let v = ValueId(self.types.len());
        self.types.push(ty);
        v
    }
    pub fn definitions(&self) -> Vec<Option<Op>> {
        let mut out = vec![None; self.types.len()];
        for block in self.blocks.values() {
            for inst in &block.insts { if let Inst::Core { value, op, .. } = inst { out[value.0] = Some(*op); } }
        }
        out
    }
    pub fn compact(&mut self) {
        let mut map: Vec<Option<ValueId>> = vec![None; self.types.len()];
        let mut types = Vec::new();
        let mut definitions: Vec<ValueId> = Vec::new();
        for block in self.blocks.values() {
            for &(p, _) in &block.params { definitions.push(p); }
            for inst in &block.insts {
                match inst {
                    Inst::Core { value, .. } | Inst::Packet { output: value, .. } => definitions.push(*value),
                    Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => definitions.extend(outputs.iter().map(|o| o.0)),
                }
            }
        }
        for v in definitions {
            map[v.0] = Some(ValueId(types.len()));
            types.push(self.types[v.0]);
        }
        let m = |v: ValueId| map[v.0].expect("use of a removed SSA value");
        for block in self.blocks.values_mut() {
            for (p, _) in &mut block.params { *p = m(*p); }
            for inst in &mut block.insts {
                match inst {
                    Inst::Core { value, op, .. } => { *op = op.map(m); *value = m(*value); }
                    Inst::Packet { input, output, .. } => { *input = m(*input); *output = m(*output); }
                    Inst::Target { args, outputs, .. } => { *args = args.map(m); for (v, _) in outputs { *v = m(*v); } }
                    Inst::Effect { inputs, outputs, .. } => { for v in inputs { *v = m(*v); } for (v, _) in outputs { *v = m(*v); } }
                }
            }
            match &mut block.term {
                Term::Br(e) => for v in &mut e.args { *v = m(*v); },
                Term::CondBr { cond, yes, no } => { *cond = m(*cond); for v in yes.args.iter_mut().chain(&mut no.args) { *v = m(*v); } }
                Term::Ret(args) => for v in args { *v = m(*v); },
            }
        }
        self.types = types;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::IntOp;
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
                        term: Term::Ret(vec![]),
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
