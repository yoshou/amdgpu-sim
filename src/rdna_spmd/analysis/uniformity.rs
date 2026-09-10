use super::super::ir::{*, EffectOp, MemoryOp, MemSize, Space, WaveOp, Cvt, Env, IntOp, Op, Ty, ValueId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fact {
    Uniform,
    Affine { stride: i64, span: Option<(u32, u32)> },
    Varying,
}

impl Fact {
    fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Self::Affine { stride: a, span: x }, Self::Affine { stride: b, span: y }) if a == b =>
                Self::Affine { stride: a, span: if x == y { x } else { None } },
            _ if self == other => self,
            _ => Self::Varying,
        }
    }
    fn join_all(a: Self, b: Self) -> Self {
        match (a, b) {
            (Self::Uniform, Self::Uniform) => Self::Uniform,
            _ => Self::Varying,
        }
    }
    fn add(self, other: Self, negate: bool) -> Self {
        match (self, other) {
            (Self::Uniform, Self::Uniform) => Self::Uniform,
            (Self::Affine { stride, .. }, Self::Uniform) => Self::Affine { stride, span: None },
            (Self::Uniform, Self::Affine { stride, .. }) => Self::Affine { stride: if negate { -stride } else { stride }, span: None },
            (Self::Affine { stride: a, .. }, Self::Affine { stride: b, .. }) => {
                let stride = if negate { a - b } else { a + b };
                if stride == 0 { Self::Uniform } else { Self::Affine { stride, span: None } }
            }
            _ => Self::Varying,
        }
    }
    fn bits(start: u32, end: u32) -> u64 {
        let end = if end >= 64 { u64::MAX } else { (1u64 << end) - 1 };
        let start = if start >= 64 { u64::MAX } else { (1u64 << start) - 1 };
        end & !start
    }
    fn masked(self, mask: u64) -> Self {
        match self {
            Self::Uniform => Self::Uniform,
            Self::Affine { stride, span: Some((start, end)) } => {
                let lane = Self::bits(start, end);
                if mask & lane == lane { Self::Affine { stride, span: Some((start, end)) } }
                else if mask & lane == 0 { Self::Uniform }
                else { Self::Varying }
            }
            _ => Self::Varying,
        }
    }
    fn shifted_right(self, amount: u64) -> Self {
        match self {
            Self::Uniform => Self::Uniform,
            Self::Affine { span: Some((_, end)), .. } if amount >= end as u64 => Self::Uniform,
            Self::Affine { stride, span: Some((start, end)) } if amount <= start as u64 && stride.trailing_zeros() as u64 >= amount =>
                Self::Affine { stride: stride >> amount, span: Some((start - amount as u32, end - amount as u32)) },
            _ => Self::Varying,
        }
    }
    fn shifted_left(self, amount: u64) -> Self {
        match self {
            Self::Uniform => Self::Uniform,
            Self::Affine { stride, span } if amount < 63 => Self::Affine { stride: stride.wrapping_shl(amount as u32),
                span: span.map(|(start, end)| (start + amount as u32, end + amount as u32)) },
            _ => Self::Varying,
        }
    }
    fn word(self, bits: u32) -> Self {
        match self {
            Self::Affine { stride, .. } if stride.trailing_zeros() >= bits => Self::Uniform,
            Self::Affine { stride, span } => Self::Affine { stride, span: span.filter(|&(_, end)| end <= bits) },
            other => other,
        }
    }
}

pub(crate) struct Entry {
    pub uniform: Vec<ValueId>,
    pub affine: Vec<(ValueId, i64, Option<(u32, u32)>)>,
    pub varying: Vec<ValueId>,
}

pub(crate) struct Uniformity {
    pub facts: Vec<Fact>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub pairs: BTreeMap<(ValueId, ValueId), Fact>,
}

#[cfg(test)]
impl Uniformity {
    pub fn fact(&self, value: ValueId) -> Fact { self.facts[value.0] }
    pub fn pair(&self, lo: ValueId, hi: ValueId) -> Option<Fact> { self.pairs.get(&(lo, hi)).copied() }
}

fn effect_output(op: EffectOp, inputs: &[Fact]) -> Fact {
    match op {
        EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane | WaveOp::ReadLane) => Fact::Uniform,
        EffectOp::Wave(_) => Fact::Varying,
        EffectOp::Memory { space: Space::Global, op: MemoryOp::Load(MemSize::B32), .. } => {
            if inputs[0] == Fact::Uniform { Fact::Uniform } else { Fact::Varying }
        }
        EffectOp::Memory { .. } => Fact::Varying,
        EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => Fact::Uniform,
    }
}

pub(crate) fn packet(f: &Func, entry: &Entry, constants: &[Option<u64>], guarded: &[bool]) -> Uniformity {
    let mut facts: Vec<Option<Fact>> = vec![None; f.types.len()];
    let mut pairs: Vec<Option<(ValueId, Fact)>> = vec![None; f.types.len()];
    let mut definitions: Vec<Option<Op>> = vec![None; f.types.len()];
    let mut unpacked_pairs: Vec<(ValueId, ValueId, ValueId)> = Vec::new();
    for block in f.blocks.values() {
        let mut lows: Vec<(ValueId, ValueId)> = Vec::new();
        let mut highs: Vec<(ValueId, ValueId)> = Vec::new();
        for inst in &block.insts {
            if let Inst::Core { value, op, .. } = inst {
                definitions[value.0] = Some(*op);
                match op {
                    Op::UnpackLo(x) => lows.push((*x, *value)),
                    Op::UnpackHi(x) => highs.push((*x, *value)),
                    _ => {}
                }
            }
        }
        for &(x, lo) in &lows { for &(y, hi) in &highs { if y == x { unpacked_pairs.push((lo, hi, x)); } } }
    }
    let entry_block = &f.blocks[&f.entry];
    for &(id, ty) in &entry_block.params {
        facts[id.0] = Some(if entry.varying.contains(&id) { Fact::Varying }
            else if entry.uniform.contains(&id) { Fact::Uniform }
            else if let Some(&(_, stride, span)) = entry.affine.iter().find(|(v, _, _)| *v == id) { Fact::Affine { stride, span } }
            else if ty == Ty::I1 { Fact::Varying }
            else { Fact::Uniform });
    }
    let unpacked = |value: ValueId, definitions: &[Option<Op>]| -> Option<(ValueId, bool)> {
        match definitions[value.0] { Some(Op::UnpackLo(x)) => Some((x, false)), Some(Op::UnpackHi(x)) => Some((x, true)), _ => None }
    };
    let mut top: Vec<Option<ValueId>> = vec![None; f.types.len()];
    for (&id, block) in &f.blocks {
        if id != f.entry { for w in block.params.windows(2) { top[w[0].0 .0] = Some(w[1].0); } }
    }
    let blocks: Vec<&Block> = f.blocks.values().collect();
    let mut position = vec![usize::MAX; f.blocks.keys().map(|b| b.0 + 1).max().unwrap_or(0)];
    for (at, id) in f.blocks.keys().enumerate() { position[id.0] = at; }
    let mut incoming_edges: Vec<Vec<&Edge>> = (0..blocks.len()).map(|_| Vec::new()).collect();
    for block in &blocks {
        for edge in block.term.edges() { incoming_edges[position[edge.dst.0]].push(edge); }
    }
    let order = {
        let mut order = Vec::with_capacity(blocks.len());
        let mut seen = vec![false; blocks.len()];
        let entry = position[f.entry.0];
        seen[entry] = true;
        let mut stack: Vec<(usize, usize)> = vec![(entry, 0)];
        while let Some((at, next)) = stack.last_mut() {
            match blocks[*at].term.edges().nth(*next) {
                Some(edge) => {
                    *next += 1;
                    let dst = position[edge.dst.0];
                    if !seen[dst] { seen[dst] = true; stack.push((dst, 0)); }
                }
                None => { order.push(*at); stack.pop(); }
            }
        }
        order.reverse();
        for at in 0..blocks.len() { if !seen[at] { order.push(at); } }
        order
    };
    let entry_at = position[f.entry.0];
    loop {
        let mut changed = false;
        fn assign(facts: &mut [Option<Fact>], changed: &mut bool, id: ValueId, fact: Fact) { if facts[id.0] != Some(fact) { facts[id.0] = Some(fact); *changed = true; } }
        for &at in &order {
            let block = blocks[at];
            if at != entry_at {
                for (index, &(param, _)) in block.params.iter().enumerate() {
                    let high = block.params.get(index + 1).map(|p| p.0);
                    let mut merged: Option<Fact> = None;
                    let mut merged_pair: Option<Option<Fact>> = None;
                    for edge in &incoming_edges[at] {
                        let Some(&arg) = edge.args.get(index) else { continue };
                        if let Some(fact) = facts[arg.0] {
                            merged = Some(match merged { Some(current) => current.meet(fact), None => fact });
                        }
                        let Some(high) = high else { continue };
                        if index + 1 >= edge.args.len() { continue; }
                        let source = (arg, edge.args[index + 1]);
                        if top[source.0 .0] == Some(source.1) { continue; }
                        let fact = pairs[source.0 .0].filter(|p| p.0 == source.1).map(|p| p.1).or_else(|| {
                            let (a, b) = (unpacked(source.0, &definitions)?, unpacked(source.1, &definitions)?);
                            (a.0 == b.0 && !a.1 && b.1).then(|| facts[a.0 .0]).flatten().filter(|fact| matches!(fact, Fact::Affine { .. }))
                        });
                        let _ = high;
                        merged_pair = Some(match merged_pair {
                            Some(current) => match (current, fact) { (Some(a), Some(b)) if a == b => Some(a), _ => None },
                            None => fact,
                        });
                    }
                    if let Some(fact) = merged { assign(&mut facts, &mut changed, param, fact); }
                    if let (Some(fact), Some(high)) = (merged_pair, high) {
                        if top[param.0] == Some(high) { top[param.0] = None; changed = true; }
                        let next = fact.map(|fact| (high, fact));
                        if pairs[param.0] != next { pairs[param.0] = next; changed = true; }
                    }
                }
            }
            for inst in &block.insts {
                match inst {
                    Inst::Core { value, ty, op } => {
                        let v = |id: ValueId| facts[id.0].unwrap_or(Fact::Uniform);
                        let fact = match *op {
                            Op::Const(..) => Fact::Uniform,
                            Op::Env(Env::LaneId | Env::PacketLaneId) => Fact::Affine { stride: 1, span: None },
                            Op::Env(Env::ValidLane | Env::ScratchBase | Env::ScratchSize) => Fact::Varying,
                            Op::Int(IntOp::Add, a, b) => v(a).add(v(b), false),
                            Op::Int(IntOp::Sub, a, b) => v(a).add(v(b), true),
                            Op::Int(IntOp::Mul, a, b) => match (v(a), v(b), constants[a.0], constants[b.0]) {
                                (Fact::Uniform, Fact::Uniform, _, _) => Fact::Uniform,
                                (fact @ Fact::Affine { .. }, _, _, Some(c)) | (_, fact @ Fact::Affine { .. }, Some(c), _) => {
                                    if c != 0 && c.is_power_of_two() { fact.shifted_left(c.trailing_zeros() as u64) }
                                    else if let Fact::Affine { stride, .. } = fact { Fact::Affine { stride: stride.wrapping_mul(c as i64), span: None } }
                                    else { Fact::Varying }
                                }
                                _ => Fact::Varying,
                            },
                            Op::Int(IntOp::Shl, a, b) => match constants[b.0] {
                                Some(c) => v(a).shifted_left(c),
                                None => Fact::join_all(v(a), v(b)),
                            },
                            Op::Int(IntOp::LShr, a, b) => match constants[b.0] {
                                Some(c) => v(a).shifted_right(c),
                                None => Fact::join_all(v(a), v(b)),
                            },
                            Op::Int(IntOp::And, a, b) => match (constants[a.0], constants[b.0]) {
                                (_, Some(m)) => v(a).masked(m),
                                (Some(m), _) => v(b).masked(m),
                                _ => Fact::join_all(v(a), v(b)),
                            },
                            Op::Int(_, a, b) => Fact::join_all(v(a), v(b)),
                            Op::Convert(Cvt::Trunc, to, a) => v(a).word(to.bits()),
                            Op::Convert(Cvt::ZExt | Cvt::SExt | Cvt::Bitcast, _, a) => v(a),
                            Op::Convert(_, _, a) => if v(a) == Fact::Uniform { Fact::Uniform } else { Fact::Varying },
                            Op::Pack64(lo, hi) => pairs[lo.0].filter(|p| p.0 == hi).map(|p| p.1).unwrap_or_else(|| match (v(lo), v(hi)) {
                                (Fact::Affine { stride, span }, Fact::Uniform) => Fact::Affine { stride, span: span.filter(|&(_, end)| end <= 32) },
                                (Fact::Uniform, Fact::Affine { stride, span }) => Fact::Affine { stride: stride.wrapping_shl(32), span: span.map(|(a, b)| (a + 32, b + 32)) },
                                (a, b) => Fact::join_all(a, b),
                            }),
                            Op::UnpackLo(a) => v(a).word(32),
                            Op::UnpackHi(a) => match v(a) {
                                Fact::Uniform => Fact::Uniform,
                                Fact::Affine { span: Some((_, end)), .. } if end <= 32 => Fact::Uniform,
                                Fact::Affine { stride, span: Some((start, end)) } if start >= 32 => Fact::Affine { stride: stride >> 32, span: Some((start - 32, end - 32)) },
                                Fact::Affine { stride, span: None } if stride.trailing_zeros() >= 32 => Fact::Affine { stride: stride >> 32, span: None },
                                _ => Fact::Varying,
                            },
                            Op::Select(c, a, b) => {
                                if a == b || guarded[value.0] { v(a) }
                                else if v(c) == Fact::Uniform { v(a).meet(v(b)) }
                                else { Fact::Varying }
                            }
                            Op::TrailingZeros(a) | Op::LeadingZeros(a) | Op::PopulationCount(a) | Op::ReverseBits(a) | Op::Unary(_, a) =>
                                if v(a) == Fact::Uniform { Fact::Uniform } else { Fact::Varying },
                            Op::Cmp(_, a, b) | Op::FCmp(_, a, b) | Op::Float(_, a, b) => Fact::join_all(v(a), v(b)),
                            Op::Fma(a, b, c) | Op::MulAdd(a, b, c) => Fact::join_all(Fact::join_all(v(a), v(b)), v(c)),
                        };
                        let fact = if *ty == Ty::I1 && fact != Fact::Uniform { Fact::Varying } else { fact };
                        assign(&mut facts, &mut changed, *value, fact);
                    }
                    Inst::Packet { output, .. } => assign(&mut facts, &mut changed, *output, Fact::Uniform),
                    Inst::Target { provenance, args, outputs, .. } => {
                        let fact = if provenance.is_some() { Fact::Varying }
                            else if args.values().iter().all(|a| facts[a.0].unwrap_or(Fact::Uniform) == Fact::Uniform) { Fact::Uniform } else { Fact::Varying };
                        for &(id, _) in outputs { assign(&mut facts, &mut changed, id, fact); }
                    }
                    Inst::Effect { op, inputs, outputs, .. } => {
                        let input_facts: Vec<_> = inputs.iter().map(|id| facts[id.0].unwrap_or(Fact::Uniform)).collect();
                        let fact = effect_output(*op, &input_facts);
                        for &(id, _) in outputs { assign(&mut facts, &mut changed, id, fact); }
                    }
                }
            }
        }
        for &(lo, hi, x) in &unpacked_pairs {
            if let Some(fact @ Fact::Affine { .. }) = facts[x.0] {
                if pairs[lo.0] != Some((hi, fact)) { pairs[lo.0] = Some((hi, fact)); changed = true; }
            }
        }
        if !changed { break; }
    }
    let pairs = pairs.iter().enumerate().filter_map(|(lo, p)| p.map(|(hi, fact)| ((ValueId(lo), hi), fact))).collect();
    Uniformity { facts: facts.into_iter().map(|fact| fact.unwrap_or(Fact::Uniform)).collect(), pairs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::ir::{IntPred};
    use crate::rdna_spmd::{targets::rdna4::decode::{ScalarProgram, ScalarBlock, Terminator}, targets::rdna4::lift::wave::{YieldAction, Operand, Destination}, CompilationInput};
    use crate::rdna_instructions::{InstFormat, SourceOperand, VOP3SD, VSCRATCH, VGLOBAL};
    use crate::instructions::I;

    fn lifted(program: &ScalarProgram) -> (crate::rdna_spmd::program::LiftedFunction, Uniformity) {
        let f = program.to_ssa().function;
        let u = crate::rdna_spmd::compiler::packet_uniformity(&f, 16, true, true);
        (f, u)
    }
    fn slot(f: &crate::rdna_spmd::program::LiftedFunction, slot: u32) -> usize {
        f.parameter_inputs.iter().position(|p| matches!(p.source, crate::rdna_spmd::program::ParameterSource::Vgpr(r) if r == slot)).unwrap()
    }
    fn parameter(f: &crate::rdna_spmd::program::LiftedFunction, pc: usize, register: u32) -> ValueId {
        f.ir.blocks[&BlockId(pc)].params[slot(f, register)].0
    }
    fn returned(f: &crate::rdna_spmd::program::LiftedFunction, pc: usize, register: u32) -> ValueId {
        let Term::Ret(args) = &f.ir.blocks[&BlockId(pc)].term else { panic!("block does not return") };
        args[slot(f, register)]
    }
    fn frame() -> InstFormat {
        InstFormat::VOP3SD(VOP3SD { vdst: 10, sdst: 106, cm: 0, op: I::V_MAD_CO_U64_U32, src0: SourceOperand::VectorRegister(0),
            src1: SourceOperand::IntegerConstant(16), src2: SourceOperand::ScalarRegister(0), omod: 0, neg: 0 })
    }

    #[test]
    fn scratch_loads_are_varying_and_uniform_global_loads_are_uniform() {
        let scratch = InstFormat::VSCRATCH(VSCRATCH { op: I::SCRATCH_LOAD_B32, vaddr: 0, vsrc: 0, vdst: 7, scope: 0, th: 0, ioffset: 0, saddr: 124, sve: 0 });
        let global = InstFormat::VGLOBAL(VGLOBAL { op: I::GLOBAL_LOAD_B32, vaddr: 2, vsrc: 0, vdst: 8, scope: 0, th: 0, ioffset: 0, saddr: 4, sve: 0 });
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock { pc: 0, body: vec![scratch, global], term: Terminator::Return })]) };
        let (f, u) = lifted(&program);
        assert_eq!(u.fact(returned(&f, 0, 7)), Fact::Varying);
        assert_eq!(u.fact(returned(&f, 0, 8)), Fact::Uniform);
    }

    #[test]
    fn cross_lane_writes_on_resume_are_varying_and_break_frames() {
        let write = |reg| Box::new(YieldAction::new(crate::rdna_spmd::ir::EffectOp::Wave(crate::rdna_spmd::ir::WaveOp::WriteLane),
            vec![Operand::Source(SourceOperand::IntegerConstant(0)), Operand::Source(SourceOperand::IntegerConstant(0)), Operand::Source(SourceOperand::VectorRegister(reg)), Operand::Source(SourceOperand::IntegerConstant(reg as u64))],
            vec![Destination::Vgpr(reg as u32)]));
        let program = ScalarProgram { entry_pc: 1, blocks: BTreeMap::from([
            (1, ScalarBlock { pc: 1, body: vec![frame()], term: Terminator::Yield { resume: 2, action: write(23) } }),
            (2, ScalarBlock { pc: 2, body: vec![], term: Terminator::Return }),
        ]) };
        let (f, u) = lifted(&program);
        assert_eq!(u.fact(parameter(&f, 2, 23)), Fact::Varying);
        assert_eq!(u.pair(parameter(&f, 2, 10), parameter(&f, 2, 11)), Some(Fact::Affine { stride: 16, span: None }));
        let program = ScalarProgram { entry_pc: 1, blocks: BTreeMap::from([
            (1, ScalarBlock { pc: 1, body: vec![frame()], term: Terminator::Yield { resume: 2, action: write(10) } }),
            (2, ScalarBlock { pc: 2, body: vec![], term: Terminator::Return }),
        ]) };
        let (f, u) = lifted(&program);
        assert_eq!(u.pair(parameter(&f, 2, 10), parameter(&f, 2, 11)), None);
    }

    fn func() -> (Func, Vec<ValueId>) {
        let mut f = Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] };
        let lane = f.value(Ty::I32); let base = f.value(Ty::I32); let flag = f.value(Ty::I1);
        (f, vec![lane, base, flag])
    }

    #[test]
    fn affine_lane_indices_scale_by_constants_and_survive_uniform_offsets() {
        let (mut f, params) = func();
        let (lane, base, flag) = (params[0], params[1], params[2]);
        let mut insts = Vec::new();
        let mut core = |f: &mut Func, ty, op| { let v = f.value(ty); insts.push(Inst::Core { value: v, ty, op }); v };
        let wide = core(&mut f, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
        let sixteen = core(&mut f, Ty::I64, Op::Const(Ty::I64, 16));
        let scaled = core(&mut f, Ty::I64, Op::Int(IntOp::Mul, wide, sixteen));
        let base64 = core(&mut f, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, base));
        let sum = core(&mut f, Ty::I64, Op::Int(IntOp::Add, scaled, base64));
        let lo = core(&mut f, Ty::I32, Op::UnpackLo(sum));
        let hi = core(&mut f, Ty::I32, Op::UnpackHi(sum));
        let repacked = core(&mut f, Ty::I64, Op::Pack64(lo, hi));
        let swapped = core(&mut f, Ty::I64, Op::Pack64(hi, lo));
        let chosen = core(&mut f, Ty::I32, Op::Select(flag, base, base));
        let picked = core(&mut f, Ty::I32, Op::Select(flag, base, lane));
        let cmp = core(&mut f, Ty::I1, Op::Cmp(IntPred::Eq, base, base));
        let product = core(&mut f, Ty::I64, Op::Int(IntOp::Mul, wide, base64));
        f.blocks.insert(BlockId(0), Block { params: vec![(lane, Ty::I32), (base, Ty::I32), (flag, Ty::I1)], insts, term: Term::Ret(vec![]) });
        let constants = super::super::constants(&f);
        let u = packet(&f, &Entry { uniform: vec![base], affine: vec![(lane, 1, Some((0, 10)))] , varying: vec![] }, &constants, &vec![false; f.types.len()]);
        assert_eq!(u.fact(scaled), Fact::Affine { stride: 16, span: Some((4, 14)) });
        assert_eq!(u.fact(sum), Fact::Affine { stride: 16, span: None });
        assert_eq!(u.fact(repacked), Fact::Affine { stride: 16, span: None });
        assert_eq!(u.fact(swapped), Fact::Varying);
        assert_eq!(u.pair(lo, hi), Some(Fact::Affine { stride: 16, span: None }));
        assert_eq!(u.fact(chosen), Fact::Uniform);
        assert_eq!(u.fact(picked), Fact::Varying);
        assert_eq!(u.fact(cmp), Fact::Uniform);
        assert_eq!(u.fact(product), Fact::Varying);
        assert_eq!(u.fact(flag), Fact::Varying);
    }

    #[test]
    fn packed_work_item_ids_split_into_an_affine_x_and_uniform_y() {
        let mut f = Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] };
        let ids = f.value(Ty::I32); let other = f.value(Ty::I32);
        let mut insts = Vec::new();
        let mut core = |f: &mut Func, ty, op| { let v = f.value(ty); insts.push(Inst::Core { value: v, ty, op }); v };
        let low = core(&mut f, Ty::I32, Op::Const(Ty::I32, 0x3ff));
        let x = core(&mut f, Ty::I32, Op::Int(IntOp::And, ids, low));
        let ten = core(&mut f, Ty::I32, Op::Const(Ty::I32, 10));
        let shifted = core(&mut f, Ty::I32, Op::Int(IntOp::LShr, ids, ten));
        let y = core(&mut f, Ty::I32, Op::Int(IntOp::And, shifted, low));
        let high = core(&mut f, Ty::I32, Op::Const(Ty::I32, 0xffc00));
        let y_bits = core(&mut f, Ty::I32, Op::Int(IntOp::And, ids, high));
        let partial = core(&mut f, Ty::I32, Op::Const(Ty::I32, 0x3f));
        let cut = core(&mut f, Ty::I32, Op::Int(IntOp::And, ids, partial));
        let pair = core(&mut f, Ty::I64, Op::Pack64(x, other));
        let wide = core(&mut f, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, x));
        let scaled = core(&mut f, Ty::I64, Op::Int(IntOp::Shl, wide, ten));
        let scaled_x = core(&mut f, Ty::I32, Op::Int(IntOp::Shl, x, ten));
        let masked_scaled = core(&mut f, Ty::I32, Op::Int(IntOp::And, scaled_x, high));
        let back = core(&mut f, Ty::I32, Op::Int(IntOp::LShr, masked_scaled, ten));
        let high_pair = core(&mut f, Ty::I64, Op::Pack64(other, x));
        let other_wide = core(&mut f, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, other));
        let sum = core(&mut f, Ty::I64, Op::Int(IntOp::Add, high_pair, other_wide));
        let low_word = core(&mut f, Ty::I32, Op::UnpackLo(sum));
        let high_word = core(&mut f, Ty::I32, Op::UnpackHi(high_pair));
        f.blocks.insert(BlockId(0), Block { params: vec![(ids, Ty::I32), (other, Ty::I32)], insts, term: Term::Ret(vec![]) });
        let constants = super::super::constants(&f);
        let u = packet(&f, &Entry { uniform: vec![other], affine: vec![(ids, 1, Some((0, 10)))] , varying: vec![] }, &constants, &vec![false; f.types.len()]);
        assert_eq!(u.fact(x), Fact::Affine { stride: 1, span: Some((0, 10)) });
        assert_eq!(u.fact(shifted), Fact::Uniform);
        assert_eq!(u.fact(y), Fact::Uniform);
        assert_eq!(u.fact(y_bits), Fact::Uniform);
        assert_eq!(u.fact(cut), Fact::Varying);
        assert_eq!(u.fact(pair), Fact::Affine { stride: 1, span: Some((0, 10)) });
        assert_eq!(u.fact(scaled), Fact::Affine { stride: 1024, span: Some((10, 20)) });
        assert_eq!(u.fact(masked_scaled), Fact::Affine { stride: 1024, span: Some((10, 20)) });
        assert_eq!(u.fact(back), Fact::Affine { stride: 1, span: Some((0, 10)) });
        assert_eq!(u.fact(high_pair), Fact::Affine { stride: 1 << 32, span: Some((32, 42)) });
        assert_eq!(u.fact(low_word), Fact::Uniform);
        assert_eq!(u.fact(high_word), Fact::Affine { stride: 1, span: Some((0, 10)) });
    }

    #[test]
    fn block_arguments_meet_incoming_facts_including_split_pairs() {
        let mut f = Func { entry: BlockId(0), blocks: BTreeMap::new(), types: vec![] };
        let lane = f.value(Ty::I32); let base = f.value(Ty::I32);
        let mut insts = Vec::new();
        let mut core = |f: &mut Func, ty, op| { let v = f.value(ty); insts.push(Inst::Core { value: v, ty, op }); v };
        let wide = core(&mut f, Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, lane));
        let eight = core(&mut f, Ty::I64, Op::Const(Ty::I64, 8));
        let sum = core(&mut f, Ty::I64, Op::Int(IntOp::Mul, wide, eight));
        let lo = core(&mut f, Ty::I32, Op::UnpackLo(sum));
        let hi = core(&mut f, Ty::I32, Op::UnpackHi(sum));
        let one = core(&mut f, Ty::I32, Op::Const(Ty::I32, 1));
        let cond = core(&mut f, Ty::I1, Op::Cmp(IntPred::Eq, base, one));
        let p_lo = f.value(Ty::I32); let p_hi = f.value(Ty::I32); let p_base = f.value(Ty::I32); let p_mixed = f.value(Ty::I32);
        f.blocks.insert(BlockId(0), Block { params: vec![(lane, Ty::I32), (base, Ty::I32)], insts,
            term: Term::CondBr { cond, yes: Edge { dst: BlockId(1), args: vec![lo, hi, base, base] }, no: Edge { dst: BlockId(1), args: vec![lo, hi, one, lane] } } });
        let next = f.value(Ty::I32);
        f.blocks.insert(BlockId(1), Block { params: vec![(p_lo, Ty::I32), (p_hi, Ty::I32), (p_base, Ty::I32), (p_mixed, Ty::I32)],
            insts: vec![Inst::Core { value: next, ty: Ty::I32, op: Op::Int(IntOp::Add, p_base, one) }],
            term: Term::CondBr { cond, yes: Edge { dst: BlockId(1), args: vec![p_lo, p_hi, next, p_mixed] }, no: Edge { dst: BlockId(2), args: vec![] } } });
        f.blocks.insert(BlockId(2), Block { params: vec![], insts: vec![], term: Term::Ret(vec![]) });
        let constants = super::super::constants(&f);
        let u = packet(&f, &Entry { uniform: vec![base], affine: vec![(lane, 1, Some((0, 10)))] , varying: vec![] }, &constants, &vec![false; f.types.len()]);
        assert_eq!(u.pair(p_lo, p_hi), Some(Fact::Affine { stride: 8, span: Some((3, 13)) }));
        assert_eq!(u.fact(p_lo), Fact::Affine { stride: 8, span: Some((3, 13)) });
        assert_eq!(u.fact(p_base), Fact::Uniform);
        assert_eq!(u.fact(p_mixed), Fact::Varying);
    }
}
