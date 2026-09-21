use super::effect::EffectOp;
use super::scope::Presence;
use super::{Op, Ty, ValueId};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BlockId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketOp {
    Any,
    Ballot,
}
impl PacketOp {
    pub fn result_type(self) -> Ty {
        match self {
            Self::Any => Ty::I1,
            Self::Ballot => Ty::I32,
        }
    }
}
#[derive(Clone, Debug, PartialEq)]
pub enum Inst {
    Packet {
        op: PacketOp,
        input: ValueId,
        output: ValueId,
    },
    Target {
        provenance: Option<u64>,
        op: crate::rdna_spmd::ir::TargetOp,
        args: crate::rdna_spmd::ir::Arguments,
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
impl Inst {
    pub fn for_each_operand(&self, mut f: impl FnMut(ValueId)) {
        match self {
            Inst::Core { op, .. } => {
                op.map(|v| {
                    f(v);
                    v
                });
            }
            Inst::Packet { input, .. } => f(*input),
            Inst::Target { args, .. } => args.values().iter().for_each(|&v| f(v)),
            Inst::Effect { inputs, .. } => inputs.iter().for_each(|&v| f(v)),
        }
    }
    pub fn for_each_output(&self, mut f: impl FnMut(ValueId)) {
        match self {
            Inst::Core { value, .. } | Inst::Packet { output: value, .. } => f(*value),
            Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => {
                outputs.iter().for_each(|o| f(o.0))
            }
        }
    }
    pub fn operands(&self) -> Vec<ValueId> {
        let mut out = Vec::new();
        self.for_each_operand(|v| out.push(v));
        out
    }
    pub fn outputs(&self) -> Vec<ValueId> {
        let mut out = Vec::new();
        self.for_each_output(|v| out.push(v));
        out
    }
}
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub dst: BlockId,
    pub args: Vec<ValueId>,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Term {
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
pub struct Block {
    pub params: Vec<(ValueId, Ty)>,
    pub insts: Vec<Inst>,
    pub term: Term,
}
#[derive(Clone, Debug, PartialEq)]
pub struct Func {
    pub entry: BlockId,
    pub blocks: BTreeMap<BlockId, Block>,
    pub types: Vec<Ty>,

    pub regions: BTreeMap<BlockId, Presence>,
}
impl Func {

    pub fn new(entry: BlockId, presence: Presence) -> Self {
        Self {
            entry,
            blocks: BTreeMap::new(),
            types: Vec::new(),
            regions: BTreeMap::from([(entry, presence)]),
        }
    }

    pub fn value(&mut self, ty: Ty) -> ValueId {
        let v = ValueId(self.types.len());
        self.types.push(ty);
        v
    }
    pub fn reverse_postorder(&self) -> Vec<BlockId> {
        let mut seen = BTreeSet::from([self.entry]);
        let mut order = Vec::new();
        let mut stack = vec![(self.entry, 0usize)];
        while let Some(&mut (id, ref mut next)) = stack.last_mut() {
            let edge = self.blocks[&id].term.edges().nth(*next).map(|e| e.dst);
            match edge {
                Some(dst) => {
                    *next += 1;
                    if seen.insert(dst) {
                        stack.push((dst, 0));
                    }
                }
                None => {
                    order.push(id);
                    stack.pop();
                }
            }
        }
        order.reverse();
        order
    }
    pub fn rename(&mut self, map: &BTreeMap<ValueId, ValueId>) {
        let m = |v: ValueId| {
            let mut v = v;
            while let Some(&next) = map.get(&v) {
                v = next;
            }
            v
        };
        for block in self.blocks.values_mut() {
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
            match &mut block.term {
                Term::Br(e) => {
                    for v in &mut e.args {
                        *v = m(*v);
                    }
                }
                Term::CondBr { cond, yes, no } => {
                    *cond = m(*cond);
                    for v in yes.args.iter_mut().chain(&mut no.args) {
                        *v = m(*v);
                    }
                }
                Term::Ret(args) => {
                    for v in args {
                        *v = m(*v);
                    }
                }
            }
        }
    }
    pub fn definitions(&self) -> Vec<Option<Op>> {
        let mut out = vec![None; self.types.len()];
        for block in self.blocks.values() {
            for inst in &block.insts {
                if let Inst::Core { value, op, .. } = inst {
                    out[value.0] = Some(*op);
                }
            }
        }
        out
    }
    pub fn compact(&mut self) {
        let mut map: Vec<Option<ValueId>> = vec![None; self.types.len()];
        let mut types = Vec::new();
        let mut definitions: Vec<ValueId> = Vec::new();
        for block in self.blocks.values() {
            for &(p, _) in &block.params {
                definitions.push(p);
            }
            for inst in &block.insts {
                match inst {
                    Inst::Core { value, .. } | Inst::Packet { output: value, .. } => {
                        definitions.push(*value)
                    }
                    Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => {
                        definitions.extend(outputs.iter().map(|o| o.0))
                    }
                }
            }
        }
        for v in definitions {
            map[v.0] = Some(ValueId(types.len()));
            types.push(self.types[v.0]);
        }
        let m = |v: ValueId| map[v.0].expect("use of a removed SSA value");
        for block in self.blocks.values_mut() {
            for (p, _) in &mut block.params {
                *p = m(*p);
            }
            for inst in &mut block.insts {
                match inst {
                    Inst::Core { value, op, .. } => {
                        *op = op.map(m);
                        *value = m(*value);
                    }
                    Inst::Packet { input, output, .. } => {
                        *input = m(*input);
                        *output = m(*output);
                    }
                    Inst::Target { args, outputs, .. } => {
                        *args = args.map(m);
                        for (v, _) in outputs {
                            *v = m(*v);
                        }
                    }
                    Inst::Effect {
                        inputs, outputs, ..
                    } => {
                        for v in inputs {
                            *v = m(*v);
                        }
                        for (v, _) in outputs {
                            *v = m(*v);
                        }
                    }
                }
            }
            match &mut block.term {
                Term::Br(e) => {
                    for v in &mut e.args {
                        *v = m(*v);
                    }
                }
                Term::CondBr { cond, yes, no } => {
                    *cond = m(*cond);
                    for v in yes.args.iter_mut().chain(&mut no.args) {
                        *v = m(*v);
                    }
                }
                Term::Ret(args) => {
                    for v in args {
                        *v = m(*v);
                    }
                }
            }
        }
        self.types = types;
    }
}
