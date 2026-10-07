use crate::rdna_spmd::ir::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    Param { block: BlockId, index: usize },
    Inst { block: BlockId, index: usize },
    Unreached,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Use {
    Inst { block: BlockId, index: usize },
    Arg { block: BlockId, edge: usize, index: usize },
    Cond(BlockId),
    Ret(BlockId),
}

pub fn is_word(ty: Ty) -> bool {
    matches!(ty, Ty::I32 | Ty::I64)
}

pub struct Facts {
    pub site: Vec<Site>,
    pub uses: Vec<Vec<Use>>,
    pub incoming: BTreeMap<BlockId, Vec<(BlockId, usize)>>,
    pub order: Vec<BlockId>,
    pub uniform: Vec<bool>,
    pub lane_word: Vec<bool>,
    pub materialized: Vec<bool>,
    pub saturated: Vec<bool>,
    pub viewed: Vec<bool>,
}

impl Facts {

    pub fn new(f: &Func, inputs: &[Parameter], kept: &BTreeSet<ValueId>) -> Self {
        let n = f.types.len();
        let order = f.reverse_postorder();
        let mut site = vec![Site::Unreached; n];
        let mut uses: Vec<Vec<Use>> = vec![Vec::new(); n];
        let mut incoming: BTreeMap<BlockId, Vec<(BlockId, usize)>> =
            order.iter().map(|&b| (b, Vec::new())).collect();
        for &id in &order {
            let block = &f.blocks[&id];
            for (index, &(v, _)) in block.params.iter().enumerate() {
                site[v.0] = Site::Param { block: id, index };
            }
            for (index, inst) in block.insts.iter().enumerate() {
                for v in inst.outputs() {
                    site[v.0] = Site::Inst { block: id, index };
                }
                for v in inst.operands() {
                    uses[v.0].push(Use::Inst { block: id, index });
                }
            }
            match &block.term {
                Term::CondBr { cond, .. } => uses[cond.0].push(Use::Cond(id)),
                Term::Ret(args) => {
                    for v in args {
                        uses[v.0].push(Use::Ret(id));
                    }
                }
                Term::Br(_) => {}
            }
            for (edge, e) in block.term.edges().enumerate() {
                incoming.get_mut(&e.dst).unwrap().push((id, edge));
                for (index, &v) in e.args.iter().enumerate() {
                    uses[v.0].push(Use::Arg {
                        block: id,
                        edge,
                        index,
                    });
                }
            }
        }
        let mut facts = Facts {
            site,
            uses,
            incoming,
            order,
            uniform: vec![true; n],
            lane_word: vec![false; n],
            materialized: vec![false; n],
            saturated: vec![false; n],
            viewed: vec![false; n],
        };
        facts.solve_uniform(f, inputs);
        facts.solve_lane_words(f);
        facts.solve_materialized(f, kept);
        facts.solve_saturated(f);
        facts.solve_viewed(f);
        facts
    }

    fn solve_viewed(&mut self, f: &Func) {
        let mut pending: Vec<ValueId> = Vec::new();
        for v in 0..f.types.len() {
            if !is_word(f.types[v]) {
                continue;
            }
            let projected = self.uses[v].iter().any(|&u| match u {
                Use::Inst { block, index } => match f.blocks[&block].insts[index] {
                    Inst::Core {
                        value,
                        op: Op::Int(IntOp::LShr, a, s),
                        ..
                    } => {
                        a.0 == v
                            && self.is_lane_shift(f, a, s)
                            && self.uses[value.0].iter().any(|&t| {
                                matches!(t, Use::Inst { block, index }
                                    if matches!(f.blocks[&block].insts[index],
                                        Inst::Core { op: Op::Convert(Cvt::Trunc, Ty::I1, _), .. }))
                            })
                    }
                    _ => false,
                },
                _ => false,
            });
            if self.lane_word[v] || projected {
                pending.push(ValueId(v));
            }
        }
        while let Some(v) = pending.pop() {
            if self.viewed[v.0] {
                continue;
            }
            self.viewed[v.0] = true;
            let sources: Vec<ValueId> = match self.site[v.0] {
                Site::Param { block, index } if block != f.entry => {
                    self.arguments(f, block, index).collect()
                }
                Site::Inst { .. } => self.word_sources(f, v),
                _ => vec![],
            };
            pending.extend(
                sources
                    .into_iter()
                    .filter(|s| is_word(f.types[s.0]) && !self.viewed[s.0]),
            );
        }
    }

    fn word_sources(&self, f: &Func, v: ValueId) -> Vec<ValueId> {
        match self.op(f, v) {
            Some(Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, a, b))
            | Some(Op::Select(_, a, b))
            | Some(Op::Pack64(a, b)) => vec![a, b],
            Some(Op::Convert(Cvt::Bitcast, Ty::I32 | Ty::I64, a))
            | Some(Op::UnpackLo(a))
            | Some(Op::UnpackHi(a)) => vec![a],
            _ => vec![],
        }
    }

    pub fn inst<'f>(&self, f: &'f Func, v: ValueId) -> Option<&'f Inst> {
        match self.site[v.0] {
            Site::Inst { block, index } => Some(&f.blocks[&block].insts[index]),
            _ => None,
        }
    }

    pub fn op(&self, f: &Func, v: ValueId) -> Option<Op> {
        match self.inst(f, v) {
            Some(Inst::Core { op, .. }) => Some(*op),
            _ => None,
        }
    }

    pub fn constant(&self, f: &Func, v: ValueId) -> Option<u64> {
        match self.op(f, v) {
            Some(Op::Const(_, k)) => Some(k),
            _ => None,
        }
    }

    pub fn is_lane_id(&self, f: &Func, v: ValueId) -> bool {
        matches!(self.op(f, v), Some(Op::Env(Env::LaneId)))
    }

    pub fn is_lane_shift(&self, f: &Func, w: ValueId, s: ValueId) -> bool {
        match f.types[w.0] {
            Ty::I32 => self.is_lane_id(f, s),
            Ty::I64 => matches!(self.op(f, s), Some(Op::Convert(Cvt::ZExt, Ty::I64, l)) if self.is_lane_id(f, l)),
            _ => false,
        }
    }

    pub fn arguments<'f>(
        &'f self,
        f: &'f Func,
        block: BlockId,
        index: usize,
    ) -> impl Iterator<Item = ValueId> + 'f {
        self.incoming[&block].iter().map(move |&(pred, edge)| {
            f.blocks[&pred].term.edges().nth(edge).unwrap().args[index]
        })
    }

    fn solve_uniform(&mut self, f: &Func, inputs: &[Parameter]) {
        let entry = &f.blocks[&f.entry];
        for (index, &(v, _)) in entry.params.iter().enumerate() {
            self.uniform[v.0] = matches!(
                inputs.get(index).map(|p| p.source),
                Some(ParameterSource::Sgpr(_) | ParameterSource::Scc)
            );
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &id in &self.order {
                let block = &f.blocks[&id];
                if id != f.entry {
                    for (index, &(v, _)) in block.params.iter().enumerate() {
                        let u = self.arguments(f, id, index).all(|a| self.uniform[a.0]);
                        if !u && self.uniform[v.0] {
                            self.uniform[v.0] = false;
                            changed = true;
                        }
                    }
                }
                for inst in &block.insts {
                    let all = inst.operands().iter().all(|a| self.uniform[a.0]);
                    let u = match inst {
                        Inst::Core { op, .. } => match op {
                            Op::Const(..) => true,
                            Op::Env(Env::ValidLane | Env::ScratchSize) => true,
                            Op::Env(_) => false,
                            _ => all,
                        },
                        Inst::Target { .. } => all,
                        Inst::Packet { .. } => false,
                        Inst::Effect { op, .. } => match op {
                            EffectOp::Memory {
                                space: Space::Global,
                                op: MemoryOp::Load(_),
                                ..
                            } => all,
                            EffectOp::Wave(WaveOp::Any | WaveOp::Ballot { .. } | WaveOp::ReadFirstLane) => {
                                true
                            }
                            _ => false,
                        },
                    };
                    for v in inst.outputs() {
                        if !u && self.uniform[v.0] {
                            self.uniform[v.0] = false;
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    fn solve_lane_words(&mut self, f: &Func) {
        let mut changed = true;
        while changed {
            changed = false;
            for &id in &self.order {
                let block = &f.blocks[&id];
                if id != f.entry {
                    for (index, &(v, ty)) in block.params.iter().enumerate() {
                        if is_word(ty)
                            && !self.lane_word[v.0]
                            && self.arguments(f, id, index).any(|a| self.lane_word[a.0])
                        {
                            self.lane_word[v.0] = true;
                            changed = true;
                        }
                    }
                }
                for inst in &block.insts {
                    let word = match inst {
                        Inst::Effect {
                            op: EffectOp::Wave(WaveOp::Ballot { .. }),
                            ..
                        } => true,
                        Inst::Core { ty, op, .. } if is_word(*ty) => match *op {
                            Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, a, b)
                            | Op::Select(_, a, b)
                            | Op::Pack64(a, b) => self.lane_word[a.0] || self.lane_word[b.0],
                            Op::Convert(Cvt::Bitcast, Ty::I32 | Ty::I64, a) | Op::UnpackLo(a) => {
                                self.lane_word[a.0]
                            }
                            Op::UnpackHi(a) => f.lanes == 64 && self.lane_word[a.0],
                            _ => false,
                        },
                        _ => false,
                    };
                    if word {
                        for v in inst.outputs() {
                            if !self.lane_word[v.0] {
                                self.lane_word[v.0] = true;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }
    }

    fn view_use(&self, f: &Func, w: ValueId, u: Use) -> bool {
        match u {
            Use::Ret(_) => true,
            Use::Cond(_) => false,
            Use::Arg { .. } => true,
            Use::Inst { block, index } => {
                let Inst::Core { value, ty, op } = &f.blocks[&block].insts[index] else {
                    return false;
                };
                let covers = f.types[w.0].bits() >= f.lanes;
                match *op {
                    Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, ..) => is_word(*ty),
                    Op::Select(c, _, _) => c != w,
                    Op::Convert(Cvt::Bitcast, Ty::I32 | Ty::I64, _) => true,
                    Op::Pack64(..) | Op::UnpackLo(_) => true,
                    Op::UnpackHi(_) => f.lanes == 64,
                    Op::Int(IntOp::LShr, a, s) => {
                        covers
                            && a == w
                            && s != w
                            && self.is_lane_shift(f, a, s)
                            && self.uses[value.0].iter().all(|&shifted| {
                                matches!(shifted, Use::Inst { block, index }
                                    if matches!(f.blocks[&block].insts[index],
                                        Inst::Core { op: Op::Convert(Cvt::Trunc, Ty::I1, _), .. }))
                            })
                    }
                    Op::Cmp(IntPred::Eq | IntPred::Ne, a, b) => {
                        let other = if a == w { b } else { a };
                        covers && other != w && self.constant(f, other) == Some(0)
                    }
                    _ => false,
                }
            }
        }
    }

    fn solve_materialized(&mut self, f: &Func, kept: &BTreeSet<ValueId>) {
        let n = f.types.len();
        let mut pending: Vec<ValueId> = kept.iter().copied().collect();
        for v in 0..n {
            if self.lane_word[v]
                && self.uses[v]
                    .iter()
                    .any(|&u| !self.view_use(f, ValueId(v), u))
            {
                pending.push(ValueId(v));
            }
        }
        while let Some(v) = pending.pop() {
            if self.materialized[v.0] {
                continue;
            }
            self.materialized[v.0] = true;
            let sources: Vec<ValueId> = match self.site[v.0] {
                Site::Param { block, index } if block != f.entry => {
                    self.arguments(f, block, index).collect()
                }
                Site::Inst { .. } => match self.op(f, v) {
                    Some(Op::Int(_, a, b)) | Some(Op::Select(_, a, b)) | Some(Op::Pack64(a, b)) => {
                        vec![a, b]
                    }
                    Some(Op::Convert(_, _, a)) | Some(Op::UnpackLo(a)) | Some(Op::UnpackHi(a)) => {
                        vec![a]
                    }
                    _ => vec![],
                },
                _ => vec![],
            };
            pending.extend(sources.into_iter().filter(|s| self.lane_word[s.0]));
        }
    }

    fn solve_saturated(&mut self, f: &Func) {
        for &id in &self.order {
            let block = &f.blocks[&id];
            if id != f.entry {
                for &(v, ty) in &block.params {
                    self.saturated[v.0] = is_word(ty);
                }
            }
            for inst in &block.insts {
                if let Inst::Core { value, ty, .. } = inst {
                    if is_word(*ty) {
                        self.saturated[value.0] = true;
                    }
                }
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for &id in &self.order {
                let block = &f.blocks[&id];
                if id != f.entry {
                    for (index, &(v, ty)) in block.params.iter().enumerate() {
                        if !is_word(ty) || !self.saturated[v.0] {
                            continue;
                        }
                        if !self.arguments(f, id, index).all(|a| self.saturated[a.0]) {
                            self.saturated[v.0] = false;
                            changed = true;
                        }
                    }
                }
                for inst in &block.insts {
                    let Inst::Core { value, ty, op } = inst else {
                        continue;
                    };
                    if !is_word(*ty) {
                        continue;
                    }
                    let ones = if *ty == Ty::I64 { u64::MAX } else { 0xffff_ffff };
                    let s = match *op {
                        Op::Const(_, k) => k == 0 || k == ones,
                        Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, a, b) => {
                            self.saturated[a.0] && self.saturated[b.0]
                        }
                        Op::Select(c, a, b) => {
                            self.uniform[c.0] && self.saturated[a.0] && self.saturated[b.0]
                        }
                        Op::Convert(Cvt::Bitcast, Ty::I32 | Ty::I64, a)
                        | Op::UnpackLo(a)
                        | Op::UnpackHi(a) => self.saturated[a.0],
                        _ => false,
                    };
                    if !s && self.saturated[value.0] {
                        self.saturated[value.0] = false;
                        changed = true;
                    }
                }
            }
        }
    }
}
