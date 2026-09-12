use super::super::ir::{*, EffectOp, MemoryOp, MemSize, Space, WaveOp, Cvt, Env, IntOp, Op, Ty, ValueId};
use super::super::program::{Parameter, ParameterSource};
use super::dataflow::{Cfg, Lattice, Sparse};
use super::{Analyses, Analysis, Constants, Masks, Packet};
use std::collections::BTreeMap;

impl Analysis for Uniformity {
    type Result = Uniformity;
    const NAME: &'static str = "uniformity";
    fn compute(f: &Func, analyses: &Analyses) -> Self {
        let ctx = analyses.context();
        let Some(Packet { aligned }) = ctx.packet else {
            return Uniformity { facts: vec![Fact::Uniform; f.types.len()], pairs: BTreeMap::new() };
        };
        let (constants, masks) = (analyses.get::<Constants>(f), analyses.get::<Masks>(f));
        packet(f, &entry(f, ctx.inputs, aligned), &constants, &masks.guarded)
    }
}

fn entry(f: &Func, inputs: &[Parameter], aligned: bool) -> Entry {
    let block = &f.blocks[&f.entry];
    let mut uniform = Vec::new();
    let mut affine = Vec::new();
    let mut varying = Vec::new();
    for (input, &(id, _)) in inputs.iter().zip(&block.params) {
        match input.source {
            ParameterSource::Vgpr(0) => if aligned { affine.push((id, 1, Some((0, 10)))) } else { varying.push(id) },
            ParameterSource::Vgpr(_) | ParameterSource::Sgpr(_) | ParameterSource::Scc => uniform.push(id),
            _ => {}
        }
    }
    Entry { uniform, affine, varying }
}

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

struct Entry {
    uniform: Vec<ValueId>,
    affine: Vec<(ValueId, i64, Option<(u32, u32)>)>,
    varying: Vec<ValueId>,
}

#[derive(PartialEq)]
pub(crate) struct Uniformity {
    pub facts: Vec<Fact>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub pairs: BTreeMap<(ValueId, ValueId), Fact>,
}

impl Uniformity {
    pub fn uniform(&self) -> Vec<bool> { self.facts.iter().map(|&fact| fact == Fact::Uniform).collect() }
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

#[derive(Clone, Copy, PartialEq)]
enum Known { Top, Fact(Fact) }
impl Known {
    fn get(self) -> Fact { match self { Known::Top => Fact::Uniform, Known::Fact(fact) => fact } }
}
impl Lattice for Known {
    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Known::Top, x) | (x, Known::Top) => *x,
            (Known::Fact(a), Known::Fact(b)) => Known::Fact(a.meet(*b)),
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Pair { Unknown, Of(ValueId, Fact), Broken }
impl Lattice for Pair {
    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Pair::Unknown, x) | (x, Pair::Unknown) => *x,
            (Pair::Of(a, x), Pair::Of(b, y)) if a == b && x == y => *self,
            _ => Pair::Broken,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Lane { fact: Known, pair: Pair }
impl Lattice for Lane {
    fn meet(&self, other: &Self) -> Self { Lane { fact: self.fact.meet(&other.fact), pair: self.pair.meet(&other.pair) } }
}

fn packet(f: &Func, entry: &Entry, constants: &[Option<u64>], guarded: &[bool]) -> Uniformity {
    let definitions = f.definitions();
    let mut high_of: Vec<Option<ValueId>> = vec![None; f.types.len()];
    for block in f.blocks.values() {
        let mut lows: Vec<(ValueId, ValueId)> = Vec::new();
        let mut highs: Vec<(ValueId, ValueId)> = Vec::new();
        for inst in &block.insts {
            if let Inst::Core { value, op, .. } = inst {
                match op {
                    Op::UnpackLo(x) => lows.push((*x, *value)),
                    Op::UnpackHi(x) => highs.push((*x, *value)),
                    _ => {}
                }
            }
        }
        for &(x, lo) in &lows { for &(y, hi) in &highs { if y == x { high_of[lo.0] = Some(hi); } } }
    }
    let unpacked = |value: ValueId| -> Option<(ValueId, bool)> {
        match definitions[value.0] { Some(Op::UnpackLo(x)) => Some((x, false)), Some(Op::UnpackHi(x)) => Some((x, true)), _ => None }
    };
    let cfg = Cfg::new(f);
    let boundary = |id: ValueId| {
        let ty = f.types[id.0];
        let fact = if entry.varying.contains(&id) { Fact::Varying }
            else if entry.uniform.contains(&id) { Fact::Uniform }
            else if let Some(&(_, stride, span)) = entry.affine.iter().find(|(v, _, _)| *v == id) { Fact::Affine { stride, span } }
            else if ty == Ty::I1 { Fact::Varying }
            else { Fact::Uniform };
        Lane { fact: Known::Fact(fact), pair: Pair::Broken }
    };
    let edge = |edge: &Edge, _: usize, position: usize, facts: &[Lane]| {
        let dst = &cfg.blocks[cfg.index[edge.dst.0]];
        let arg = edge.args[position];
        let fact = facts[arg.0].fact;
        let known = |id: ValueId| facts[id.0].fact.get();
        let pair = match (dst.params.get(position + 1), edge.args.get(position + 1)) {
            (Some(&(high, _)), Some(&next)) => match facts[arg.0].pair {
                Pair::Unknown => Pair::Unknown,
                Pair::Of(h, fact) if h == next => Pair::Of(high, fact),
                _ => match (unpacked(arg), unpacked(next)) {
                    (Some((x, false)), Some((y, true))) if x == y && matches!(known(x), Fact::Affine { .. }) => Pair::Of(high, known(x)),
                    _ => Pair::Broken,
                },
            },
            _ => Pair::Broken,
        };
        Lane { fact, pair }
    };
    let transfer = |inst: &Inst, value: ValueId, facts: &[Lane]| {
        let v = |id: ValueId| facts[id.0].fact.get();
        let fact = match inst {
            Inst::Core { ty, op, .. } => {
                let fact = match *op {
                    Op::Const(..) => Fact::Uniform,
                    Op::Env(Env::LaneId | Env::PacketLaneId) => Fact::Affine { stride: 1, span: None },
                    Op::Env(Env::OutsideLanes) => Fact::Uniform,
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
                    Op::Pack64(lo, hi) => match facts[lo.0].pair {
                        Pair::Of(h, fact) if h == hi => fact,
                        _ => match (v(lo), v(hi)) {
                            (Fact::Affine { stride, span }, Fact::Uniform) => Fact::Affine { stride, span: span.filter(|&(_, end)| end <= 32) },
                            (Fact::Uniform, Fact::Affine { stride, span }) => Fact::Affine { stride: stride.wrapping_shl(32), span: span.map(|(a, b)| (a + 32, b + 32)) },
                            (a, b) => Fact::join_all(a, b),
                        },
                    },
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
                if *ty == Ty::I1 && fact != Fact::Uniform { Fact::Varying } else { fact }
            }
            Inst::Packet { .. } => Fact::Uniform,
            Inst::Target { provenance, args, .. } => {
                if provenance.is_some() { Fact::Varying }
                else if args.values().iter().all(|a| v(*a) == Fact::Uniform) { Fact::Uniform } else { Fact::Varying }
            }
            Inst::Effect { op, inputs, .. } => {
                let input_facts: Vec<_> = inputs.iter().map(|id| v(*id)).collect();
                effect_output(*op, &input_facts)
            }
        };
        let pair = match (definitions[value.0], high_of[value.0]) {
            (Some(Op::UnpackLo(x)), Some(hi)) => match v(x) { fact @ Fact::Affine { .. } => Pair::Of(hi, fact), _ => Pair::Broken },
            _ => Pair::Broken,
        };
        Lane { fact: Known::Fact(fact), pair }
    };
    let start = Lane { fact: Known::Top, pair: Pair::Unknown };
    let lanes = Sparse { cfg: &cfg, start, boundary: &boundary, edge: &edge, transfer: &transfer }.solve(f.types.len());
    let pairs = lanes.iter().enumerate().filter_map(|(lo, lane)| match lane.pair { Pair::Of(hi, fact) => Some(((ValueId(lo), hi), fact)), _ => None }).collect();
    Uniformity { facts: lanes.iter().map(|lane| lane.fact.get()).collect(), pairs }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::super::ir::{IntPred};
    use crate::rdna_spmd::{targets::rdna4::decode::{ScalarProgram, ScalarBlock, Terminator}, targets::rdna4::lift::wave::{YieldAction, Operand, Destination}, CompilationInput};
    use crate::rdna_instructions::{InstFormat, SourceOperand, VOP3SD, VSCRATCH, VGLOBAL};
    use crate::instructions::I;

    fn lifted(program: &ScalarProgram) -> (crate::rdna_spmd::program::LiftedFunction, std::rc::Rc<Uniformity>) {
        let f = program.to_ssa().function;
        let exec_index = crate::rdna_spmd::compiler::exec_index(&f.parameter_inputs, &f.registry);
        let ctx = super::super::Context { entry_full: true, packet: Some(Packet { aligned: true }), ..super::super::Context::new(&f.registry, &f.parameter_inputs, exec_index, 16) };
        let u = Analyses::new(ctx).get::<Uniformity>(&f.ir);
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
        let constants = super::super::constant::constants(&f);
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
        let constants = super::super::constant::constants(&f);
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
        let constants = super::super::constant::constants(&f);
        let u = packet(&f, &Entry { uniform: vec![base], affine: vec![(lane, 1, Some((0, 10)))] , varying: vec![] }, &constants, &vec![false; f.types.len()]);
        assert_eq!(u.pair(p_lo, p_hi), Some(Fact::Affine { stride: 8, span: Some((3, 13)) }));
        assert_eq!(u.fact(p_lo), Fact::Affine { stride: 8, span: Some((3, 13)) });
        assert_eq!(u.fact(p_base), Fact::Uniform);
        assert_eq!(u.fact(p_mixed), Fact::Varying);
    }
}
