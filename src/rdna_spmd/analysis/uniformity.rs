use super::super::ir::{
    Cvt, EffectOp, Env, IntOp, MemSize, MemoryOp, Op, Space, Ty, ValueId, WaveOp, *,
};
use super::dataflow::{Cfg, Lattice, Sparse};
use super::{Analyses, Analysis, Constants, Packet};
use std::collections::BTreeMap;

impl Analysis for Uniformity {
    type Result = Uniformity;
    const NAME: &'static str = "uniformity";
    fn compute(f: &Func, analyses: &Analyses) -> Self {
        let ctx = analyses.context();
        let Some(Packet { aligned }) = ctx.packet else {
            return Uniformity {
                facts: vec![Fact::Uniform; f.types.len()],
                pairs: BTreeMap::new(),
            };
        };
        let constants = analyses.get::<Constants>(f);
        packet(
            f,
            &entry(f, ctx.inputs, aligned),
            &constants,
        )
    }
}

fn entry(f: &Func, inputs: &[Parameter], aligned: bool) -> Entry {
    let block = &f.blocks[&f.entry];
    let mut uniform = Vec::new();
    let mut affine = Vec::new();
    let mut varying = Vec::new();
    for (input, &(id, _)) in inputs.iter().zip(&block.params) {
        match input.source {
            ParameterSource::Vgpr(0) => {
                if aligned {
                    affine.push((id, 1, Some((0, 10))))
                } else {
                    varying.push(id)
                }
            }
            ParameterSource::Vgpr(_) | ParameterSource::Sgpr(_) | ParameterSource::Scc => {
                uniform.push(id)
            }
            _ => {}
        }
    }
    Entry {
        uniform,
        affine,
        varying,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fact {
    Uniform,
    Affine {
        stride: i64,
        span: Option<(u32, u32)>,
    },
    Varying,
}

impl Fact {
    fn meet(self, other: Self) -> Self {
        match (self, other) {
            (Self::Affine { stride: a, span: x }, Self::Affine { stride: b, span: y })
                if a == b =>
            {
                Self::Affine {
                    stride: a,
                    span: if x == y { x } else { None },
                }
            }
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
            (Self::Uniform, Self::Affine { stride, .. }) => Self::Affine {
                stride: if negate {
                    stride.wrapping_neg()
                } else {
                    stride
                },
                span: None,
            },
            (Self::Affine { stride: a, .. }, Self::Affine { stride: b, .. }) => {

                let stride = if negate {
                    a.wrapping_sub(b)
                } else {
                    a.wrapping_add(b)
                };
                if stride == 0 {
                    Self::Uniform
                } else {
                    Self::Affine { stride, span: None }
                }
            }
            _ => Self::Varying,
        }
    }
    fn bits(start: u32, end: u32) -> u64 {
        let end = if end >= 64 {
            u64::MAX
        } else {
            (1u64 << end) - 1
        };
        let start = if start >= 64 {
            u64::MAX
        } else {
            (1u64 << start) - 1
        };
        end & !start
    }

    fn masked(self, mask: u64, bits: u32) -> Self {
        let whole = if bits >= 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        let mask = mask & whole;
        if mask == whole {
            return self;
        }
        let cleared = whole & !mask;
        match self {
            Self::Uniform => Self::Uniform,
            Self::Affine {
                stride,
                span: Some((start, end)),
            } => {
                let lane = Self::bits(start, end);
                if mask & lane == lane {
                    Self::Affine {
                        stride,
                        span: Some((start, end)),
                    }
                } else if mask & lane == 0 {
                    Self::Uniform
                } else {
                    Self::Varying
                }
            }
            Self::Affine { stride, span: None }
                if mask != 0
                    && (mask + 1).is_power_of_two()
                    && stride.trailing_zeros() >= mask.count_ones() =>
            {
                Self::Uniform
            }
            Self::Affine { stride, span: None }
                if mask != 0
                    && (cleared + 1).is_power_of_two()
                    && stride.trailing_zeros() >= cleared.count_ones() =>
            {
                Self::Affine { stride, span: None }
            }
            _ => Self::Varying,
        }
    }

    fn shifted_right(self, amount: u64, bits: u32) -> Self {
        let amount = amount & (bits as u64 - 1);
        if amount == 0 {
            return self;
        }
        match self {
            Self::Uniform => Self::Uniform,
            Self::Affine {
                span: Some((_, end)),
                ..
            } if amount >= end as u64 => Self::Uniform,
            Self::Affine {
                stride,
                span: Some((start, end)),
            } if amount <= start as u64 && stride.trailing_zeros() as u64 >= amount => {
                Self::Affine {
                    stride: stride >> amount,
                    span: Some((start - amount as u32, end - amount as u32)),
                }
            }
            _ => Self::Varying,
        }
    }

    fn shifted_left(self, amount: u64, bits: u32) -> Self {
        let amount = amount & (bits as u64 - 1);
        match self {
            Self::Affine { stride, span } => Self::Affine {
                stride: stride.wrapping_shl(amount as u32),
                span: span
                    .map(|(start, end)| (start + amount as u32, end + amount as u32))
                    .filter(|&(_, end)| end <= bits),
            },
            other => other,
        }
    }

    fn widened(self, bits: u32, signed: bool) -> Self {
        match self {
            Self::Affine {
                span: Some((_, end)),
                ..
            } if end < bits || !signed && end <= bits => self,
            Self::Affine { .. } => Self::Varying,
            other => other,
        }
    }
    fn word(self, bits: u32) -> Self {
        match self {
            Self::Affine { stride, .. } if stride.trailing_zeros() >= bits => Self::Uniform,
            Self::Affine { stride, span } => Self::Affine {
                stride,
                span: span.filter(|&(_, end)| end <= bits),
            },
            other => other,
        }
    }
}

struct Entry {
    uniform: Vec<ValueId>,
    affine: Vec<(ValueId, i64, Option<(u32, u32)>)>,
    varying: Vec<ValueId>,
}

pub(crate) struct Uniformity {
    pub facts: Vec<Fact>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub pairs: BTreeMap<(ValueId, ValueId), Fact>,
}

impl PartialEq for Uniformity {
    fn eq(&self, other: &Self) -> bool {
        self.facts == other.facts && self.pairs == other.pairs
    }
}

impl Uniformity {
    pub fn uniform(&self) -> Vec<bool> {
        self.facts
            .iter()
            .map(|&fact| fact == Fact::Uniform)
            .collect()
    }
}

fn effect_output(op: EffectOp, inputs: &[Fact]) -> Fact {
    match op {
        EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane | WaveOp::ReadLane) => {
            Fact::Uniform
        }
        EffectOp::Wave(_) => Fact::Varying,
        EffectOp::Memory {
            space: Space::Global,
            op: MemoryOp::Load(MemSize::B32 | MemSize::B64),
            ..
        } => {
            if inputs[0] == Fact::Uniform {
                Fact::Uniform
            } else {
                Fact::Varying
            }
        }
        EffectOp::Memory { .. } => Fact::Varying,
        EffectOp::BarrierSignal { .. } | EffectOp::BarrierWait => Fact::Uniform,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Known {
    Top,
    Fact(Fact),
}
impl Known {
    fn get(self) -> Fact {
        match self {
            Known::Top => Fact::Uniform,
            Known::Fact(fact) => fact,
        }
    }
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
enum Pair {
    Unknown,
    Of(ValueId, Fact),
    Broken,
}
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
struct Lane {
    fact: Known,
    pair: Pair,
}
impl Lattice for Lane {
    fn meet(&self, other: &Self) -> Self {
        Lane {
            fact: self.fact.meet(&other.fact),
            pair: self.pair.meet(&other.pair),
        }
    }
}

fn packet(
    f: &Func,
    entry: &Entry,
    constants: &[Option<u64>],
) -> Uniformity {
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
        for &(x, lo) in &lows {
            for &(y, hi) in &highs {
                if y == x {
                    high_of[lo.0] = Some(hi);
                }
            }
        }
    }
    let unpacked = |value: ValueId| -> Option<(ValueId, bool)> {
        match definitions[value.0] {
            Some(Op::UnpackLo(x)) => Some((x, false)),
            Some(Op::UnpackHi(x)) => Some((x, true)),
            _ => None,
        }
    };
    let cfg = Cfg::new(f);
    let boundary = |id: ValueId| {
        let ty = f.types[id.0];
        let fact = if entry.varying.contains(&id) {
            Fact::Varying
        } else if entry.uniform.contains(&id) {
            Fact::Uniform
        } else if let Some(&(_, stride, span)) = entry.affine.iter().find(|(v, _, _)| *v == id) {
            Fact::Affine { stride, span }
        } else if ty == Ty::I1 {
            Fact::Varying
        } else {
            Fact::Uniform
        };
        Lane {
            fact: Known::Fact(fact),
            pair: Pair::Broken,
        }
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
                    (Some((x, false)), Some((y, true)))
                        if x == y && matches!(known(x), Fact::Affine { .. }) =>
                    {
                        Pair::Of(high, known(x))
                    }
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
                    Op::Env(Env::LaneId) => Fact::Affine {
                        stride: 1,
                        span: None,
                    },
                    Op::Env(Env::ValidLane | Env::ScratchBase | Env::ScratchSize) => Fact::Varying,
                    Op::Int(IntOp::Add, a, b) => v(a).add(v(b), false),
                    Op::Int(IntOp::Sub, a, b) => v(a).add(v(b), true),
                    Op::Int(IntOp::Mul, a, b) => match (v(a), v(b), constants[a.0], constants[b.0])
                    {
                        (Fact::Uniform, Fact::Uniform, _, _) => Fact::Uniform,
                        (fact @ Fact::Affine { .. }, _, _, Some(c))
                        | (_, fact @ Fact::Affine { .. }, Some(c), _) => {
                            if c != 0 && c.is_power_of_two() {
                                fact.shifted_left(c.trailing_zeros() as u64, ty.bits())
                            } else if let Fact::Affine { stride, .. } = fact {
                                Fact::Affine {
                                    stride: stride.wrapping_mul(c as i64),
                                    span: None,
                                }
                            } else {
                                Fact::Varying
                            }
                        }
                        _ => Fact::Varying,
                    },
                    Op::Int(IntOp::Shl, a, b) => match constants[b.0] {
                        Some(c) => v(a).shifted_left(c, ty.bits()),
                        None => Fact::join_all(v(a), v(b)),
                    },
                    Op::Int(IntOp::LShr, a, b) => match constants[b.0] {
                        Some(c) => v(a).shifted_right(c, ty.bits()),
                        None => Fact::join_all(v(a), v(b)),
                    },
                    Op::Int(IntOp::And, a, b) => match (constants[a.0], constants[b.0]) {
                        (_, Some(m)) => v(a).masked(m, ty.bits()),
                        (Some(m), _) => v(b).masked(m, ty.bits()),
                        _ => Fact::join_all(v(a), v(b)),
                    },
                    Op::Int(_, a, b) => Fact::join_all(v(a), v(b)),
                    Op::Convert(Cvt::Trunc, to, a) => v(a).word(to.bits()),
                    Op::Convert(Cvt::Bitcast, _, a) => v(a),
                    Op::Convert(cvt @ (Cvt::ZExt | Cvt::SExt), _, a) => {
                        v(a).widened(f.types[a.0].bits(), cvt == Cvt::SExt)
                    }
                    Op::Convert(_, _, a) => {
                        if v(a) == Fact::Uniform {
                            Fact::Uniform
                        } else {
                            Fact::Varying
                        }
                    }
                    Op::Pack64(lo, hi) => match facts[lo.0].pair {
                        Pair::Of(h, fact) if h == hi => fact,
                        _ => match (v(lo), v(hi)) {

                            (
                                Fact::Affine {
                                    stride,
                                    span: Some(span),
                                },
                                Fact::Uniform,
                            ) => Fact::Affine {
                                stride,
                                span: Some(span),
                            },
                            (Fact::Uniform, Fact::Affine { stride, span }) => Fact::Affine {
                                stride: stride.wrapping_shl(32),
                                span: span.map(|(a, b)| (a + 32, b + 32)),
                            },
                            (a, b) => Fact::join_all(a, b),
                        },
                    },
                    Op::UnpackLo(a) => v(a).word(32),
                    Op::UnpackHi(a) => match v(a) {
                        Fact::Uniform => Fact::Uniform,
                        Fact::Affine {
                            span: Some((_, end)),
                            ..
                        } if end <= 32 => Fact::Uniform,
                        Fact::Affine {
                            stride,
                            span: Some((start, end)),
                        } if start >= 32 => Fact::Affine {
                            stride: stride >> 32,
                            span: Some((start - 32, end - 32)),
                        },
                        Fact::Affine { stride, span: None } if stride.trailing_zeros() >= 32 => {
                            Fact::Affine {
                                stride: stride >> 32,
                                span: None,
                            }
                        }
                        _ => Fact::Varying,
                    },
                    Op::Select(c, a, b) => {
                        if a == b {
                            v(a)
                        } else if v(c) == Fact::Uniform {
                            v(a).meet(v(b))
                        } else {
                            Fact::Varying
                        }
                    }
                    Op::TrailingZeros(a)
                    | Op::LeadingZeros(a)
                    | Op::PopulationCount(a)
                    | Op::ReverseBits(a)
                    | Op::Unary(_, a) => {
                        if v(a) == Fact::Uniform {
                            Fact::Uniform
                        } else {
                            Fact::Varying
                        }
                    }
                    Op::Cmp(_, a, b) | Op::FCmp(_, a, b) | Op::Float(_, a, b) => {
                        Fact::join_all(v(a), v(b))
                    }
                    Op::Fma(a, b, c) | Op::MulAdd(a, b, c) => {
                        Fact::join_all(Fact::join_all(v(a), v(b)), v(c))
                    }
                };
                if *ty == Ty::I1 && fact != Fact::Uniform {
                    Fact::Varying
                } else {
                    fact
                }
            }
            Inst::Packet { .. } => Fact::Uniform,
            Inst::Target {
                provenance, args, ..
            } => {
                if provenance.is_some() {
                    Fact::Varying
                } else if args.values().iter().all(|a| v(*a) == Fact::Uniform) {
                    Fact::Uniform
                } else {
                    Fact::Varying
                }
            }
            Inst::Effect { op, inputs, .. } => {
                let input_facts: Vec<_> = inputs.iter().map(|id| v(*id)).collect();
                effect_output(*op, &input_facts)
            }
        };
        let pair = match (definitions[value.0], high_of[value.0]) {
            (Some(Op::UnpackLo(x)), Some(hi)) => match v(x) {
                fact @ Fact::Affine { .. } => Pair::Of(hi, fact),
                _ => Pair::Broken,
            },
            _ => Pair::Broken,
        };
        Lane {
            fact: Known::Fact(fact),
            pair,
        }
    };
    let start = Lane {
        fact: Known::Top,
        pair: Pair::Unknown,
    };
    let lanes = Sparse {
        cfg: &cfg,
        start,
        boundary: &boundary,
        edge: &edge,
        transfer: &transfer,
    }
    .solve(f.types.len());
    let pairs = lanes
        .iter()
        .enumerate()
        .filter_map(|(lo, lane)| match lane.pair {
            Pair::Of(hi, fact) => Some(((ValueId(lo), hi), fact)),
            _ => None,
        })
        .collect();
    Uniformity {
        facts: lanes.iter().map(|lane| lane.fact.get()).collect(),
        pairs,
    }
}
