use super::super::analysis::{Analyses, Constants};
use super::super::ir::*;
use super::Pass;
use std::collections::{BTreeMap, BTreeSet};

enum Use {
    Word(ValueId),
    Extract(ValueId),
    Nonzero,
    Argument(BlockId, usize),
    Other,
}

struct Mask {
    repr: Vec<ValueId>,
    defs: Vec<Option<Op>>,
    constants: Vec<Option<u64>>,
    ballot: Vec<Option<ValueId>>,
    parameter: Vec<Option<(BlockId, usize)>>,
    uses: Vec<Vec<Use>>,
    truncated: Vec<bool>,
    residue: Vec<bool>,
    word: Vec<bool>,
}

fn for_each_operand(inst: &Inst, mut f: impl FnMut(ValueId)) {
    match inst {
        Inst::Core { op, .. } => { op.map(|v| { f(v); v }); }
        Inst::Packet { input, .. } => f(*input),
        Inst::Effect { inputs, .. } => for &v in inputs { f(v) },
        Inst::Target { args, .. } => for &v in args.values() { f(v) },
    }
}

fn result(inst: &Inst) -> Option<ValueId> {
    match inst {
        Inst::Core { value, .. } | Inst::Packet { output: value, .. } => Some(*value),
        Inst::Effect { outputs, .. } | Inst::Target { outputs, .. } => outputs.first().map(|o| o.0),
    }
}

fn free_provenance(f: &Func) -> u64 {
    let used = f.blocks.values().flat_map(|b| &b.insts)
        .filter_map(|inst| match inst { Inst::Effect { provenance, .. } => Some(*provenance & !(1u64 << 63) & !SCHEDULED), _ => None })
        .max().unwrap_or(0);
    used + 1
}

fn incoming<'a>(f: &'a Func, block: BlockId, index: usize) -> impl Iterator<Item = ValueId> + 'a {
    f.blocks.values().flat_map(|b| b.term.edges()).filter(move |e| e.dst == block).map(move |e| e.args[index])
}

impl Mask {
    fn of(&self, value: ValueId) -> usize { self.repr[value.0].0 }

    fn new(f: &Func, constants: &[Option<u64>]) -> Self {
        let defs = f.definitions();
        let constants = constants.to_vec();
        let mut repr: Vec<ValueId> = (0..f.types.len()).map(ValueId).collect();
        for block in f.blocks.values() {
            for inst in &block.insts {
                if let Inst::Core { value, op: Op::Convert(Cvt::Bitcast, ty, source), .. } = inst {
                    if ty.bits() == f.types[source.0].bits() { repr[value.0] = repr[source.0]; }
                }
            }
        }
        let mut m = Mask {
            repr,
            ballot: vec![None; f.types.len()],
            parameter: vec![None; f.types.len()],
            uses: (0..f.types.len()).map(|_| Vec::new()).collect(),
            truncated: vec![true; f.types.len()],
            residue: vec![false; f.types.len()],
            word: vec![false; f.types.len()],
            defs, constants,
        };
        for (&id, block) in &f.blocks {
            for (index, &(value, _)) in block.params.iter().enumerate() { m.parameter[value.0] = Some((id, index)); }
            for inst in &block.insts {
                match inst {
                    Inst::Effect { op: EffectOp::Wave(WaveOp::Ballot), inputs, outputs, .. } => m.ballot[m.repr[outputs[0].0 .0].0] = Some(inputs[0]),
                    Inst::Packet { op: PacketOp::Ballot, input, output } => m.ballot[m.repr[output.0].0] = Some(*input),
                    _ => {}
                }
            }
        }
        for block in f.blocks.values() {
            for inst in &block.insts {
                let core = match inst { Inst::Core { op, .. } => Some(*op), _ => None };
                if !matches!(core, Some(Op::Convert(Cvt::Trunc, Ty::I1, _))) {
                    for_each_operand(inst, |v| m.truncated[v.0] = false);
                }
                if let Some(value) = result(inst) { if m.repr[value.0] != value { continue; } }
                match core {
                    Some(Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, a, b)) => {
                        let value = result(inst).unwrap();
                        m.uses[m.repr[a.0].0].push(Use::Word(value));
                        m.uses[m.repr[b.0].0].push(Use::Word(value));
                    }
                    Some(Op::Select(c, a, b)) => {
                        let value = result(inst).unwrap();
                        m.uses[m.repr[c.0].0].push(Use::Other);
                        m.uses[m.repr[a.0].0].push(Use::Word(value));
                        m.uses[m.repr[b.0].0].push(Use::Word(value));
                    }
                    Some(Op::Int(IntOp::LShr, word, lane)) if matches!(m.defs[lane.0], Some(Op::Env(Env::LaneId))) => {
                        m.uses[m.repr[word.0].0].push(Use::Extract(result(inst).unwrap()));
                        m.uses[m.repr[lane.0].0].push(Use::Other);
                    }
                    Some(Op::Cmp(IntPred::Ne | IntPred::Eq, a, b)) if m.constants[a.0] == Some(0) || m.constants[b.0] == Some(0) => {
                        let (word, zero) = if m.constants[b.0] == Some(0) { (a, b) } else { (b, a) };
                        m.uses[m.repr[word.0].0].push(Use::Nonzero);
                        m.uses[m.repr[zero.0].0].push(Use::Other);
                    }
                    _ => for_each_operand(inst, |v| m.uses[m.repr[v.0].0].push(Use::Other)),
                }
            }
            match &block.term {
                Term::Br(e) => for (i, &v) in e.args.iter().enumerate() { m.uses[m.repr[v.0].0].push(Use::Argument(e.dst, i)); },
                Term::CondBr { cond, yes, no } => {
                    m.uses[m.repr[cond.0].0].push(Use::Other);
                    for e in [yes, no] { for (i, &v) in e.args.iter().enumerate() { m.uses[m.repr[v.0].0].push(Use::Argument(e.dst, i)); } }
                }
                Term::Ret(args) => for &v in args { m.uses[m.repr[v.0].0].push(Use::Other); },
            }
        }
        m
    }

    fn settle(&mut self, f: &Func) {
        let mut changed = true;
        while changed {
            changed = false;
            for (&id, block) in &f.blocks {
                for inst in &block.insts {
                    let Inst::Core { value, op, .. } = inst else { continue };
                    let residue = match *op {
                        Op::Env(Env::OutsideLanes) => true,
                        Op::Int(_, a, b) | Op::Select(_, a, b) | Op::Cmp(_, a, b) => self.residue[a.0] || self.residue[b.0],
                        Op::Convert(_, _, a) => self.residue[a.0],
                        _ => false,
                    };
                    if residue && !self.residue[value.0] { self.residue[value.0] = true; changed = true; }
                }
                for (index, &(value, _)) in block.params.iter().enumerate() {
                    if self.residue[value.0] || id == f.entry { continue; }
                    if incoming(f, id, index).any(|arg| self.residue[arg.0]) { self.residue[value.0] = true; changed = true; }
                }
            }
        }
        for (v, ty) in f.types.iter().enumerate() {
            if self.repr[v] != ValueId(v) || self.residue[v] { continue; }
            if *ty != Ty::I32 && self.ballot[v].is_none() && !matches!(self.defs[v], Some(Op::Select(..))) { continue; }
            self.word[v] = self.ballot[v].is_some()
                || matches!(self.constants[v], Some(0) | Some(0xFFFF_FFFF))
                || matches!(self.defs[v], Some(Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, ..)) | Some(Op::Select(..)))
                || self.parameter[v].is_some_and(|(block, _)| block != f.entry);
        }
        let mut changed = true;
        while changed {
            changed = false;
            for v in 0..f.types.len() {
                if self.word[v] && !self.holds(ValueId(v), f) { self.word[v] = false; changed = true; }
            }
        }
    }

    fn holds(&self, value: ValueId, f: &Func) -> bool {
        let v = value.0;
        self.uses[v].iter().all(|use_| match *use_ {
            Use::Word(value) => self.word[self.of(value)],
            Use::Extract(shifted) => self.truncated[shifted.0],
            Use::Nonzero => true,
            Use::Argument(block, index) => self.word[self.of(f.blocks[&block].params[index].0)],
            Use::Other => false,
        })
    }
}

struct Emit {
    next: usize,
    types: Vec<Ty>,
    inserted: BTreeMap<BlockId, BTreeMap<usize, Vec<Inst>>>,
    pending: Vec<Inst>,
    lane: Option<ValueId>,
    projected: BTreeMap<ValueId, ValueId>,
    masked: BTreeMap<ValueId, ValueId>,
    lanes: Option<ValueId>,
    bit: Vec<Option<ValueId>>,
    out: Vec<Option<ValueId>>,
    provenance: u64,
}

impl Emit {
    fn as_word(&mut self, value: ValueId, f: &Func) -> ValueId {
        if f.types[value.0] == Ty::I32 { return value; }
        self.push(Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, value))
    }
    fn operand(&mut self, value: ValueId, f: &Func) -> ValueId {
        if let Some(bit) = self.bit[value.0] { return bit; }
        if let Some(&bit) = self.projected.get(&value) { return bit; }
        let source = value;
        let value = self.as_word(value, f);
        let lane = match self.lane {
            Some(lane) => lane,
            None => { let lane = self.push(Ty::I32, Op::Env(Env::LaneId)); self.lane = Some(lane); lane }
        };
        let shifted = self.push(Ty::I32, Op::Int(IntOp::LShr, value, lane));
        let bit = self.push(Ty::I1, Op::Convert(Cvt::Trunc, Ty::I1, shifted));
        self.projected.insert(source, bit);
        bit
    }
    fn beyond(&mut self, value: ValueId, f: &Func) -> ValueId {
        if let Some(out) = self.out[value.0] { return out; }
        if let Some(&out) = self.masked.get(&value) { return out; }
        let value = self.as_word(value, f);
        let lanes = match self.lanes {
            Some(lanes) => lanes,
            None => { let lanes = self.push(Ty::I32, Op::Env(Env::OutsideLanes)); self.lanes = Some(lanes); lanes }
        };
        let out = self.push(Ty::I32, Op::Int(IntOp::And, value, lanes));
        self.masked.insert(value, out);
        out
    }
    fn value(&mut self, ty: Ty) -> ValueId {
        let id = ValueId(self.next);
        self.next += 1;
        self.types.push(ty);
        id
    }
    fn push(&mut self, ty: Ty, op: Op) -> ValueId {
        let value = self.value(ty);
        self.pending.push(Inst::Core { value, ty, op });
        value
    }
    fn flush(&mut self, block: BlockId, index: usize) {
        if self.pending.is_empty() { return; }
        let insts = std::mem::take(&mut self.pending);
        self.inserted.entry(block).or_default().entry(index).or_default().extend(insts);
    }
}

pub(crate) fn run(f: &mut Func, constants: &[Option<u64>]) -> usize {
    let mut mask = Mask::new(f, constants);
    if mask.ballot.iter().all(Option::is_none) { return 0; }
    mask.settle(f);
    let mut required = vec![false; f.types.len()];
    let mut beyond = vec![false; f.types.len()];
    let sites = |kinds: &[bool]| (0..f.types.len()).map(ValueId)
        .filter(|v| mask.repr[v.0] == *v && mask.word[v.0] && mask.uses[v.0].iter().any(|u| match u {
            Use::Extract(_) => kinds[0], Use::Nonzero => kinds[1], _ => false }))
        .collect::<Vec<_>>();
    let wanted = !sites(&[true, true]).is_empty();
    for (mark, mut pending, through) in [
        (&mut required, sites(&[true, true]), &mask.word),
        (&mut beyond, sites(&[false, true]), &mask.word),
    ] {
        while let Some(value) = pending.pop() {
            if mark[value.0] { continue; }
            mark[value.0] = true;
            let v = value.0;
            if mask.ballot[v].is_none() && !matches!(mask.constants[v], Some(0) | Some(0xFFFF_FFFF)) {
                if let Some(Op::Int(_, a, b)) | Some(Op::Select(_, a, b)) = mask.defs[v] {
                    for x in [a, b] { let x = mask.repr[x.0]; if through[x.0] { pending.push(x); } }
                }
            }
            if let Some((block, index)) = mask.parameter[v] {
                for arg in incoming(f, block, index).collect::<Vec<_>>() { let arg = mask.repr[arg.0]; if through[arg.0] { pending.push(arg); } }
            }
        }
    }
    if !wanted { return 0; }

    let mut emit = Emit { next: f.types.len(), types: f.types.clone(), inserted: BTreeMap::new(),
        pending: Vec::new(), lane: None, projected: BTreeMap::new(), masked: BTreeMap::new(), lanes: None,
        bit: vec![None; f.types.len()], out: vec![None; f.types.len()], provenance: free_provenance(f) };
    let mut parameters: BTreeMap<BlockId, Vec<(usize, bool)>> = BTreeMap::new();
    for (&id, block) in &f.blocks {
        for (index, &(value, ty)) in block.params.iter().enumerate() {
            if ty != Ty::I32 { continue; }
            if required[value.0] {
                let bit = emit.value(Ty::I1);
                emit.bit[value.0] = Some(bit);
                parameters.entry(id).or_default().push((index, false));
            }
            if beyond[value.0] {
                let out = emit.value(Ty::I32);
                emit.out[value.0] = Some(out);
                parameters.entry(id).or_default().push((index, true));
            }
        }
    }
    let mut renames: BTreeMap<ValueId, ValueId> = BTreeMap::new();
    let mut dropped: BTreeSet<ValueId> = BTreeSet::new();
    for (&id, block) in &f.blocks {
        emit.lane = None;
        emit.lanes = None;
        emit.projected.clear();
        emit.masked.clear();
        for (index, inst) in block.insts.iter().enumerate() {
            let Some(value) = result(inst) else { continue };
            if required[value.0] {
                let bit = match (mask.ballot[value.0], mask.constants[value.0], mask.defs[value.0]) {
                    (Some(source), ..) => source,
                    (None, Some(0), _) => emit.push(Ty::I1, Op::Const(Ty::I1, 0)),
                    (None, Some(_), _) => emit.push(Ty::I1, Op::Const(Ty::I1, 1)),
                    (None, None, Some(Op::Int(op, a, b))) => {
                        let (a, b) = (emit.operand(mask.repr[a.0], f), emit.operand(mask.repr[b.0], f));
                        emit.push(Ty::I1, Op::Int(op, a, b))
                    }
                    (None, None, Some(Op::Select(c, a, b))) => {
                        let (a, b) = (emit.operand(mask.repr[a.0], f), emit.operand(mask.repr[b.0], f));
                        emit.push(Ty::I1, Op::Select(c, a, b))
                    }
                    _ => unreachable!("a word without a bit-wise definition"),
                };
                emit.bit[value.0] = Some(bit);
                emit.flush(id, index + 1);
            }
            if beyond[value.0] {
                let out = match (mask.ballot[value.0], mask.defs[value.0]) {
                    (Some(_), _) => emit.push(Ty::I32, Op::Const(Ty::I32, 0)),
                    (None, Some(Op::Int(op @ (IntOp::And | IntOp::Or | IntOp::Xor), a, b))) => {
                        let (a, b) = (emit.beyond(mask.repr[a.0], f), emit.beyond(mask.repr[b.0], f));
                        emit.push(Ty::I32, Op::Int(op, a, b))
                    }
                    (None, Some(Op::Select(c, a, b))) => {
                        let (a, b) = (emit.beyond(mask.repr[a.0], f), emit.beyond(mask.repr[b.0], f));
                        emit.push(Ty::I32, Op::Select(c, a, b))
                    }
                    _ => emit.beyond(value, f),
                };
                emit.out[value.0] = Some(out);
                emit.flush(id, index + 1);
            }
            match mask.defs[value.0] {
                Some(Op::Convert(Cvt::Trunc, Ty::I1, shifted)) => {
                    let Some(Op::Int(IntOp::LShr, word, _)) = mask.defs[shifted.0] else { continue };
                    if !required[word.0] || !mask.truncated[shifted.0] { continue; }
                    renames.insert(value, emit.bit[word.0].expect("a word without its bit"));
                    dropped.insert(value);
                }
                Some(Op::Cmp(pred @ (IntPred::Ne | IntPred::Eq), a, b))
                    if mask.constants[a.0] == Some(0) || mask.constants[b.0] == Some(0) => {
                    let word = if mask.constants[b.0] == Some(0) { a } else { b };
                    if !required[word.0] { continue; }
                    let bit = emit.bit[word.0].expect("a word without its bit");
                    let any = emit.value(Ty::I1);
                    let provenance = emit.provenance;
                    emit.provenance += 1;
                    assert!(provenance & SCHEDULED == 0, "synthesized query provenance overflowed");
                    emit.pending.push(Inst::Effect { provenance: (1u64 << 63) | provenance,
                        op: EffectOp::Wave(WaveOp::Any), inputs: vec![bit], outputs: vec![(any, Ty::I1)] });
                    let out = emit.beyond(mask.repr[word.0], f);
                    let zero = emit.push(Ty::I32, Op::Const(Ty::I32, 0));
                    let set = emit.push(Ty::I1, Op::Cmp(IntPred::Ne, out, zero));
                    let any = emit.push(Ty::I1, Op::Int(IntOp::Or, any, set));
                    let result = if pred == IntPred::Ne { any } else {
                        let zero = emit.push(Ty::I1, Op::Const(Ty::I1, 0));
                        emit.push(Ty::I1, Op::Cmp(IntPred::Eq, any, zero))
                    };
                    emit.flush(id, index);
                    renames.insert(value, result);
                    dropped.insert(value);
                }
                _ => {}
            }
        }
    }

    for (&id, indices) in &parameters {
        let block = f.blocks.get_mut(&id).unwrap();
        for &(index, outside) in indices {
            let value = block.params[index].0;
            let (companion, ty) = if outside { (emit.out[value.0], Ty::I32) } else { (emit.bit[value.0], Ty::I1) };
            block.params.push((companion.unwrap(), ty));
        }
    }
    for id in f.blocks.keys().copied().collect::<Vec<_>>() {
        emit.lane = None;
        emit.lanes = None;
        emit.projected.clear();
        emit.masked.clear();
        let end = f.blocks[&id].insts.len();
        let carried: Vec<(usize, Vec<(ValueId, bool)>)> = f.blocks[&id].term.edges().enumerate()
            .filter_map(|(slot, edge)| parameters.get(&edge.dst).map(|indices| (slot, indices.iter().map(|&(i, outside)| (edge.args[i], outside)).collect())))
            .collect();
        let appended: Vec<(usize, Vec<ValueId>)> = carried.into_iter()
            .map(|(slot, args)| (slot, args.into_iter()
                .map(|(arg, outside)| if outside { emit.beyond(arg, f) } else { emit.operand(arg, f) })
                .collect()))
            .collect();
        emit.flush(id, end);
        let block = f.blocks.get_mut(&id).unwrap();
        for (slot, bits) in appended { block.term.edges_mut().nth(slot).unwrap().args.extend(bits); }
    }

    let count = renames.len();
    f.types = emit.types;
    for (block, sites) in emit.inserted {
        let b = f.blocks.get_mut(&block).unwrap();
        for (index, insts) in sites.into_iter().rev() { b.insts.splice(index..index, insts); }
    }
    for block in f.blocks.values_mut() {
        block.insts.retain(|inst| !matches!(inst, Inst::Core { value, .. } if dropped.contains(value)));
    }
    super::simplify::rename(f, &renames);
    f.compact();
    count
}

pub(crate) struct MaskProjection;
impl Pass for MaskProjection {
    fn name(&self) -> &str { "mask_projection" }
    fn run(&self, f: &mut Func, analyses: &Analyses) -> bool {
        let constants = analyses.get::<Constants>(f);
        run(f, &constants) > 0
    }
}
