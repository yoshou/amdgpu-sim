use super::*;
use crate::rdna_spmd::dialect::{DialectRegistry, TargetOp};
use std::fmt::Write;

pub(crate) fn ty(t: Ty) -> &'static str {
    match t { Ty::I1 => "i1", Ty::I32 => "i32", Ty::I64 => "i64", Ty::F32 => "f32", Ty::F64 => "f64" }
}
pub(crate) fn int_op(op: IntOp) -> &'static str {
    match op {
        IntOp::Add => "add", IntOp::Sub => "sub", IntOp::Mul => "mul", IntOp::And => "and", IntOp::Or => "or",
        IntOp::Xor => "xor", IntOp::Shl => "shl", IntOp::LShr => "lshr", IntOp::AShr => "ashr",
    }
}
pub(crate) fn int_pred(p: IntPred) -> &'static str {
    match p {
        IntPred::Eq => "eq", IntPred::Ne => "ne", IntPred::Ult => "ult", IntPred::Ugt => "ugt", IntPred::Ule => "ule",
        IntPred::Uge => "uge", IntPred::Slt => "slt", IntPred::Sgt => "sgt", IntPred::Sle => "sle", IntPred::Sge => "sge",
    }
}
pub(crate) fn float_op(op: FloatOp) -> &'static str {
    match op { FloatOp::Add => "add", FloatOp::Sub => "sub", FloatOp::Mul => "mul", FloatOp::Div => "div", FloatOp::MinNum => "minnum", FloatOp::MaxNum => "maxnum" }
}
pub(crate) fn float_unary(op: FloatUnary) -> &'static str {
    match op { FloatUnary::Neg => "neg", FloatUnary::Abs => "abs" }
}
pub(crate) fn float_pred(p: FloatPred) -> &'static str {
    match p {
        FloatPred::Oeq => "oeq", FloatPred::Ogt => "ogt", FloatPred::Oge => "oge", FloatPred::Olt => "olt", FloatPred::Ole => "ole",
        FloatPred::One => "one", FloatPred::Ord => "ord", FloatPred::Uno => "uno", FloatPred::Ueq => "ueq", FloatPred::Ugt => "ugt",
        FloatPred::Uge => "uge", FloatPred::Ult => "ult", FloatPred::Ule => "ule", FloatPred::Une => "une",
    }
}
pub(crate) fn cvt(op: Cvt) -> &'static str {
    match op {
        Cvt::SignedToFloatRte => "sitofp.rte", Cvt::UnsignedToFloatRte => "uitofp.rte", Cvt::FloatToSignedSatRtz => "fptosi.sat.rtz",
        Cvt::FloatToUnsignedSatRtz => "fptoui.sat.rtz", Cvt::FloatResizeRte => "fpresize.rte", Cvt::ZExt => "zext",
        Cvt::SExt => "sext", Cvt::Trunc => "trunc", Cvt::Bitcast => "bitcast",
    }
}
pub(crate) fn env(e: Env) -> &'static str {
    match e { Env::LaneId => "lane_id", Env::PacketLaneId => "packet_lane_id", Env::ValidLane => "valid_lane", Env::ScratchBase => "scratch_base", Env::ScratchSize => "scratch_size" }
}
pub(crate) fn space(s: Space) -> &'static str {
    match s { Space::Global => "global", Space::Scratch => "scratch", Space::Lds => "lds" }
}
pub(crate) fn mem_size(s: MemSize) -> &'static str {
    match s { MemSize::U8 => "u8", MemSize::I8 => "i8", MemSize::U16 => "u16", MemSize::I16 => "i16", MemSize::B32 => "b32" }
}
pub(crate) fn scope(s: Scope) -> &'static str {
    match s { Scope::WorkItem => "workitem", Scope::ComputeUnit => "cu", Scope::ShaderEngine => "se", Scope::Device => "device", Scope::System => "system", Scope::Workgroup => "workgroup" }
}
pub(crate) fn ordering(o: Ordering) -> &'static str {
    match o { Ordering::Relaxed => "relaxed", Ordering::Acquire => "acquire", Ordering::Release => "release", Ordering::Sequential => "seqcst" }
}
pub(crate) fn cache_policy(c: CachePolicy) -> &'static str {
    match c {
        CachePolicy::Temporal => "temporal", CachePolicy::NonTemporal => "nontemporal", CachePolicy::HighPriority => "high",
        CachePolicy::LastUse => "lastuse", CachePolicy::WriteBack => "writeback", CachePolicy::NearNonTemporal => "near_nt",
        CachePolicy::FarNonTemporal => "far_nt", CachePolicy::NearNonTemporalFarHigh => "near_nt_far_high",
        CachePolicy::NearNonTemporalFarWriteBack => "near_nt_far_wb",
    }
}
pub(crate) fn wave_op(w: WaveOp) -> &'static str {
    match w {
        WaveOp::Any => "any", WaveOp::Ballot => "ballot", WaveOp::ReadFirstLane => "readfirstlane", WaveOp::ReadLane => "readlane",
        WaveOp::WriteLane => "writelane", WaveOp::Bpermute => "bpermute", WaveOp::BpermuteFi => "bpermute_fi", WaveOp::Wmma => "wmma",
    }
}

fn v(id: ValueId) -> String { format!("v{}", id.0) }
fn values(ids: &[ValueId]) -> String { ids.iter().map(|&id| v(id)).collect::<Vec<_>>().join(", ") }
fn edge(e: &Edge) -> String { format!("b{}({})", e.dst.0, values(&e.args)) }

pub(crate) fn effect_op(op: EffectOp) -> String {
    match op {
        EffectOp::Memory { space: s, op, semantics } => {
            let op = match op {
                MemoryOp::Load(size) => format!("load.{}", mem_size(size)),
                MemoryOp::Store(size) => format!("store.{}", mem_size(size)),
                MemoryOp::AtomicAdd => "atomic_add".into(),
                MemoryOp::Fence => "fence".into(),
            };
            format!("memory {} {} {} {} {} volatile={} deferred={}", op, space(s), scope(semantics.scope), ordering(semantics.ordering),
                cache_policy(semantics.cache_policy), semantics.volatile as u8, semantics.deferred_scope as u8)
        }
        EffectOp::Wave(w) => format!("wave {}", wave_op(w)),
        EffectOp::BarrierSignal { is_first } => format!("barrier signal first={}", is_first as u8),
        EffectOp::BarrierWait => "barrier wait".into(),
    }
}

pub(crate) fn target_name(registry: &DialectRegistry, op: TargetOp) -> String {
    let dialect = registry.dialect_name(op.dialect()).unwrap_or("unknown");
    let name = registry.operation(op).map(|spec| spec.name).unwrap_or("unknown");
    format!("{dialect}.{name}")
}

pub(crate) fn op(o: Op) -> String {
    match o {
        Op::Env(e) => format!("env {}", env(e)),
        Op::Int(k, a, b) => format!("int {} {}, {}", int_op(k), v(a), v(b)),
        Op::TrailingZeros(a) => format!("cttz {}", v(a)),
        Op::LeadingZeros(a) => format!("ctlz {}", v(a)),
        Op::PopulationCount(a) => format!("ctpop {}", v(a)),
        Op::ReverseBits(a) => format!("bitreverse {}", v(a)),
        Op::Pack64(a, b) => format!("pack64 {}, {}", v(a), v(b)),
        Op::UnpackLo(a) => format!("unpack lo {}", v(a)),
        Op::UnpackHi(a) => format!("unpack hi {}", v(a)),
        Op::Cmp(p, a, b) => format!("cmp {} {}, {}", int_pred(p), v(a), v(b)),
        Op::Float(k, a, b) => format!("float {} {}, {}", float_op(k), v(a), v(b)),
        Op::Unary(k, a) => format!("unary {} {}", float_unary(k), v(a)),
        Op::FCmp(p, a, b) => format!("fcmp {} {}, {}", float_pred(p), v(a), v(b)),
        Op::Fma(a, b, c) => format!("fma {}, {}, {}", v(a), v(b), v(c)),
        Op::MulAdd(a, b, c) => format!("muladd {}, {}, {}", v(a), v(b), v(c)),
        Op::Convert(k, t, a) => format!("convert {} {} {}", cvt(k), ty(t), v(a)),
        Op::Const(t, bits) => format!("const {} {:#x}", ty(t), bits),
        Op::Select(c, a, b) => format!("select {}, {}, {}", v(c), v(a), v(b)),
    }
}

pub(crate) fn inst(registry: &DialectRegistry, types: &[Ty], i: &Inst) -> String {
    match i {
        Inst::Core { value, ty: t, op: o } => format!("{}: {} = {}", v(*value), ty(*t), op(*o)),
        Inst::Packet { op: PacketOp::Any, input, output } => format!("{}: i1 = packet any {}", v(*output), v(*input)),
        Inst::Packet { op: PacketOp::Ballot, input, output } => format!("{}: i32 = packet ballot {}", v(*output), v(*input)),
        Inst::Target { provenance, op, args, outputs } => {
            let lhs = outputs.iter().map(|&(id, t)| format!("{}: {}", v(id), ty(t))).collect::<Vec<_>>().join(", ");
            let provenance = provenance.map_or(String::new(), |p| format!(" !p{p}"));
            format!("{lhs} = target {}({}){provenance}", target_name(registry, *op), values(args.values()))
        }
        Inst::Effect { provenance, op, inputs, outputs } => {
            let lhs = if outputs.is_empty() { String::new() } else {
                format!("{} = ", outputs.iter().map(|&(id, t)| format!("{}: {}", v(id), ty(t))).collect::<Vec<_>>().join(", "))
            };
            let _ = types;
            format!("{lhs}effect !p{provenance} {} ({})", effect_op(*op), values(inputs))
        }
    }
}

pub(crate) fn term(t: &Term) -> String {
    match t {
        Term::Ret(args) => if args.is_empty() { "ret".into() } else { format!("ret {}", args.iter().map(|v| format!("v{}", v.0)).collect::<Vec<_>>().join(", ")) },
        Term::Br(e) => format!("br {}", edge(e)),
        Term::CondBr { cond, yes, no } => format!("condbr {}, {}, {}", v(*cond), edge(yes), edge(no)),
    }
}

pub(crate) fn func(registry: &DialectRegistry, f: &Func) -> String {
    let mut out = String::new();
    writeln!(out, "func entry b{}", f.entry.0).unwrap();
    for (id, block) in &f.blocks {
        let params = block.params.iter().map(|&(id, t)| format!("{}: {}", v(id), ty(t))).collect::<Vec<_>>().join(", ");
        writeln!(out, "b{}({}):", id.0, params).unwrap();
        for i in &block.insts { writeln!(out, "  {}", inst(registry, &f.types, i)).unwrap(); }
        writeln!(out, "  {}", term(&block.term)).unwrap();
    }
    out
}
