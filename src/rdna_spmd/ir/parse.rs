use super::*;
use crate::rdna_spmd::dialect::{Arguments, DialectRegistry};
use std::collections::BTreeMap;
use std::convert::TryInto;

pub(crate) type Result<T> = std::result::Result<T, String>;

struct Tokens<'a> { line: usize, items: Vec<&'a str>, at: usize }
impl<'a> Tokens<'a> {
    fn new(line: usize, text: &'a str) -> Self {
        let mut items = Vec::new();
        let bytes = text.as_bytes();
        let mut start = None;
        for (index, &b) in bytes.iter().enumerate() {
            let punct = matches!(b, b',' | b'(' | b')' | b':' | b'=');
            if b.is_ascii_whitespace() || punct {
                if let Some(s) = start.take() { items.push(&text[s..index]); }
                if punct { items.push(&text[index..index + 1]); }
            } else if start.is_none() { start = Some(index); }
        }
        if let Some(s) = start { items.push(&text[s..]); }
        Self { line, items, at: 0 }
    }
    fn peek(&self) -> Option<&'a str> { self.items.get(self.at).copied() }
    fn next(&mut self) -> Result<&'a str> {
        let item = self.peek().ok_or_else(|| format!("line {}: unexpected end", self.line))?;
        self.at += 1;
        Ok(item)
    }
    fn expect(&mut self, want: &str) -> Result<()> {
        let got = self.next()?;
        if got == want { Ok(()) } else { Err(format!("line {}: expected {want:?}, found {got:?}", self.line)) }
    }
    fn accept(&mut self, want: &str) -> bool {
        if self.peek() == Some(want) { self.at += 1; true } else { false }
    }
    fn done(&self) -> Result<()> {
        if self.at == self.items.len() { Ok(()) } else { Err(format!("line {}: trailing tokens", self.line)) }
    }
    fn value(&mut self) -> Result<ValueId> {
        let token = self.next()?;
        token.strip_prefix('v').and_then(|n| n.parse().ok()).map(ValueId).ok_or_else(|| format!("line {}: expected value, found {token:?}", self.line))
    }
    fn block(&mut self) -> Result<BlockId> {
        let token = self.next()?;
        token.strip_prefix('b').and_then(|n| n.parse().ok()).map(BlockId).ok_or_else(|| format!("line {}: expected block, found {token:?}", self.line))
    }
    fn ty(&mut self) -> Result<Ty> {
        let token = self.next()?;
        ty(token).ok_or_else(|| format!("line {}: unknown type {token:?}", self.line))
    }
    fn word<T: Copy>(&mut self, table: &[(&str, T)]) -> Result<T> {
        let token = self.next()?;
        table.iter().find(|(name, _)| *name == token).map(|x| x.1).ok_or_else(|| format!("line {}: unknown keyword {token:?}", self.line))
    }
    fn number(&mut self) -> Result<u64> {
        let token = self.next()?;
        let parsed = if let Some(hex) = token.strip_prefix("0x") { u64::from_str_radix(hex, 16) } else { token.parse() };
        parsed.map_err(|_| format!("line {}: expected number, found {token:?}", self.line))
    }
    fn values_in_parens(&mut self) -> Result<Vec<ValueId>> {
        self.expect("(")?;
        let mut out = Vec::new();
        if self.accept(")") { return Ok(out); }
        loop {
            out.push(self.value()?);
            if self.accept(")") { return Ok(out); }
            self.expect(",")?;
        }
    }
    fn edge(&mut self) -> Result<Edge> {
        let dst = self.block()?;
        Ok(Edge { dst, args: self.values_in_parens()? })
    }
}

fn ty(token: &str) -> Option<Ty> {
    Some(match token { "i1" => Ty::I1, "i32" => Ty::I32, "i64" => Ty::I64, "f32" => Ty::F32, "f64" => Ty::F64, _ => return None })
}

const INT_OPS: &[(&str, IntOp)] = &[("add", IntOp::Add), ("sub", IntOp::Sub), ("mul", IntOp::Mul), ("and", IntOp::And), ("or", IntOp::Or),
    ("xor", IntOp::Xor), ("shl", IntOp::Shl), ("lshr", IntOp::LShr), ("ashr", IntOp::AShr)];
const INT_PREDS: &[(&str, IntPred)] = &[("eq", IntPred::Eq), ("ne", IntPred::Ne), ("ult", IntPred::Ult), ("ugt", IntPred::Ugt), ("ule", IntPred::Ule),
    ("uge", IntPred::Uge), ("slt", IntPred::Slt), ("sgt", IntPred::Sgt), ("sle", IntPred::Sle), ("sge", IntPred::Sge)];
const FLOAT_OPS: &[(&str, FloatOp)] = &[("add", FloatOp::Add), ("sub", FloatOp::Sub), ("mul", FloatOp::Mul), ("div", FloatOp::Div),
    ("minnum", FloatOp::MinNum), ("maxnum", FloatOp::MaxNum)];
const FLOAT_UNARY: &[(&str, FloatUnary)] = &[("neg", FloatUnary::Neg), ("abs", FloatUnary::Abs)];
const FLOAT_PREDS: &[(&str, FloatPred)] = &[("oeq", FloatPred::Oeq), ("ogt", FloatPred::Ogt), ("oge", FloatPred::Oge), ("olt", FloatPred::Olt),
    ("ole", FloatPred::Ole), ("one", FloatPred::One), ("ord", FloatPred::Ord), ("uno", FloatPred::Uno), ("ueq", FloatPred::Ueq),
    ("ugt", FloatPred::Ugt), ("uge", FloatPred::Uge), ("ult", FloatPred::Ult), ("ule", FloatPred::Ule), ("une", FloatPred::Une)];
const CVTS: &[(&str, Cvt)] = &[("sitofp.rte", Cvt::SignedToFloatRte), ("uitofp.rte", Cvt::UnsignedToFloatRte), ("fptosi.sat.rtz", Cvt::FloatToSignedSatRtz),
    ("fptoui.sat.rtz", Cvt::FloatToUnsignedSatRtz), ("fpresize.rte", Cvt::FloatResizeRte), ("zext", Cvt::ZExt), ("sext", Cvt::SExt),
    ("trunc", Cvt::Trunc), ("bitcast", Cvt::Bitcast)];
const ENVS: &[(&str, Env)] = &[("lane_id", Env::LaneId), ("packet_lane_id", Env::PacketLaneId), ("valid_lane", Env::ValidLane), ("outside_lanes", Env::OutsideLanes),
    ("scratch_base", Env::ScratchBase), ("scratch_size", Env::ScratchSize)];
const SPACES: &[(&str, Space)] = &[("global", Space::Global), ("scratch", Space::Scratch), ("lds", Space::Lds)];
const SIZES: &[(&str, MemSize)] = &[("u8", MemSize::U8), ("i8", MemSize::I8), ("u16", MemSize::U16), ("i16", MemSize::I16), ("b32", MemSize::B32)];
const SCOPES: &[(&str, Scope)] = &[("workitem", Scope::WorkItem), ("cu", Scope::ComputeUnit), ("se", Scope::ShaderEngine), ("device", Scope::Device),
    ("system", Scope::System), ("workgroup", Scope::Workgroup)];
const ORDERINGS: &[(&str, Ordering)] = &[("relaxed", Ordering::Relaxed), ("acquire", Ordering::Acquire), ("release", Ordering::Release),
    ("seqcst", Ordering::Sequential)];
const CACHE: &[(&str, CachePolicy)] = &[("temporal", CachePolicy::Temporal), ("nontemporal", CachePolicy::NonTemporal), ("high", CachePolicy::HighPriority),
    ("lastuse", CachePolicy::LastUse), ("writeback", CachePolicy::WriteBack), ("near_nt", CachePolicy::NearNonTemporal),
    ("far_nt", CachePolicy::FarNonTemporal), ("near_nt_far_high", CachePolicy::NearNonTemporalFarHigh), ("near_nt_far_wb", CachePolicy::NearNonTemporalFarWriteBack)];
const WAVES: &[(&str, WaveOp)] = &[("any", WaveOp::Any), ("ballot", WaveOp::Ballot), ("readfirstlane", WaveOp::ReadFirstLane), ("readlane", WaveOp::ReadLane),
    ("writelane", WaveOp::WriteLane), ("bpermute", WaveOp::Bpermute), ("bpermute_fi", WaveOp::BpermuteFi), ("wmma", WaveOp::Wmma)];

fn flag(t: &mut Tokens, name: &str) -> Result<bool> {
    t.expect(name)?;
    t.expect("=")?;
    match t.next()? {
        "0" => Ok(false), "1" => Ok(true),
        other => Err(format!("line {}: expected {name}=0|1, found {other:?}", t.line)),
    }
}

fn op(t: &mut Tokens) -> Result<Op> {
    let kind = t.next()?;
    Ok(match kind {
        "env" => Op::Env(t.word(ENVS)?),
        "int" => { let k = t.word(INT_OPS)?; let a = t.value()?; t.expect(",")?; Op::Int(k, a, t.value()?) }
        "cttz" => Op::TrailingZeros(t.value()?),
        "ctlz" => Op::LeadingZeros(t.value()?),
        "ctpop" => Op::PopulationCount(t.value()?),
        "bitreverse" => Op::ReverseBits(t.value()?),
        "pack64" => { let a = t.value()?; t.expect(",")?; Op::Pack64(a, t.value()?) }
        "unpack" => { let half = t.next()?; let a = t.value()?; match half { "lo" => Op::UnpackLo(a), "hi" => Op::UnpackHi(a), _ => return Err(format!("line {}: unknown half {half:?}", t.line)) } }
        "cmp" => { let p = t.word(INT_PREDS)?; let a = t.value()?; t.expect(",")?; Op::Cmp(p, a, t.value()?) }
        "float" => { let k = t.word(FLOAT_OPS)?; let a = t.value()?; t.expect(",")?; Op::Float(k, a, t.value()?) }
        "unary" => { let k = t.word(FLOAT_UNARY)?; Op::Unary(k, t.value()?) }
        "fcmp" => { let p = t.word(FLOAT_PREDS)?; let a = t.value()?; t.expect(",")?; Op::FCmp(p, a, t.value()?) }
        "fma" | "muladd" => {
            let a = t.value()?; t.expect(",")?; let b = t.value()?; t.expect(",")?; let c = t.value()?;
            if kind == "fma" { Op::Fma(a, b, c) } else { Op::MulAdd(a, b, c) }
        }
        "convert" => { let k = t.word(CVTS)?; let to = t.ty()?; Op::Convert(k, to, t.value()?) }
        "const" => { let to = t.ty()?; Op::Const(to, t.number()?) }
        "select" => { let c = t.value()?; t.expect(",")?; let a = t.value()?; t.expect(",")?; Op::Select(c, a, t.value()?) }
        _ => return Err(format!("line {}: unknown operation {kind:?}", t.line)),
    })
}

fn effect(t: &mut Tokens) -> Result<EffectOp> {
    Ok(match t.next()? {
        "memory" => {
            let opcode = t.next()?;
            let op = match opcode {
                "atomic_add" => MemoryOp::AtomicAdd,
                "fence" => MemoryOp::Fence,
                _ => {
                    let (kind, size) = opcode.split_once('.').ok_or_else(|| format!("line {}: bad memory op {opcode:?}", t.line))?;
                    let size = SIZES.iter().find(|(n, _)| *n == size).map(|x| x.1).ok_or_else(|| format!("line {}: bad size {size:?}", t.line))?;
                    match kind { "load" => MemoryOp::Load(size), "store" => MemoryOp::Store(size), _ => return Err(format!("line {}: bad memory op {opcode:?}", t.line)) }
                }
            };
            let space = t.word(SPACES)?;
            let scope = t.word(SCOPES)?;
            let ordering = t.word(ORDERINGS)?;
            let cache_policy = t.word(CACHE)?;
            let volatile = flag(t, "volatile")?;
            let deferred_scope = flag(t, "deferred")?;
            EffectOp::Memory { space, op, semantics: MemorySemantics { scope, ordering, cache_policy, volatile, deferred_scope } }
        }
        "wave" => EffectOp::Wave(t.word(WAVES)?),
        "barrier" => match t.next()? {
            "signal" => EffectOp::BarrierSignal { is_first: flag(t, "first")? },
            "wait" => EffectOp::BarrierWait,
            other => return Err(format!("line {}: unknown barrier {other:?}", t.line)),
        },
        other => return Err(format!("line {}: unknown effect {other:?}", t.line)),
    })
}

fn provenance(t: &mut Tokens) -> Result<u64> {
    let token = t.next()?;
    token.strip_prefix("!p").and_then(|n| n.parse().ok()).ok_or_else(|| format!("line {}: expected provenance, found {token:?}", t.line))
}

fn arguments(line: usize, values: Vec<ValueId>) -> Result<Arguments> {
    Ok(match values.len() {
        1 => Arguments::Unary(values[0]),
        2 => Arguments::Binary([values[0], values[1]]),
        3 => Arguments::Ternary([values[0], values[1], values[2]]),
        4 => Arguments::Quaternary([values[0], values[1], values[2], values[3]]),
        13 => Arguments::Thirteen(values.try_into().unwrap()),
        14 => Arguments::Fourteen(values.try_into().unwrap()),
        15 => Arguments::Fifteen(values.try_into().unwrap()),
        16 => Arguments::Sixteen(values.try_into().unwrap()),
        n => return Err(format!("line {line}: unsupported target arity {n}")),
    })
}

fn inst(registry: &DialectRegistry, t: &mut Tokens) -> Result<Inst> {
    let mut lhs = Vec::new();
    if t.peek().is_some_and(|s| s.starts_with('v')) {
        loop {
            let id = t.value()?; t.expect(":")?; let ty = t.ty()?; lhs.push((id, ty));
            if t.accept("=") { break; }
            t.expect(",")?;
        }
    }
    let kind = t.next()?;
    let inst = match kind {
        "packet" => {
            let op = match t.next()? { "any" => PacketOp::Any, "ballot" => PacketOp::Ballot, other => return Err(format!("line {}: unknown packet op {other:?}", t.line)) };
            if lhs.len() != 1 || lhs[0].1 != op.result_type() { return Err(format!("line {}: packet result type", t.line)); }
            Inst::Packet { op, input: t.value()?, output: lhs[0].0 }
        }
        "target" => {
            let name = t.next()?;
            let (dialect, operation) = name.split_once('.').ok_or_else(|| format!("line {}: bad target name {name:?}", t.line))?;
            let dialect = registry.dialect_id(dialect).ok_or_else(|| format!("line {}: unknown dialect {dialect:?}", t.line))?;
            let op = registry.lookup(dialect, operation).map_err(|e| format!("line {}: {e}", t.line))?;
            let args = arguments(t.line, t.values_in_parens()?)?;
            let provenance = if t.peek().is_some() { Some(provenance(t)?) } else { None };
            Inst::Target { provenance, op, args, outputs: lhs }
        }
        "effect" => {
            let provenance = provenance(t)?;
            let op = effect(t)?;
            Inst::Effect { provenance, op, inputs: t.values_in_parens()?, outputs: lhs }
        }
        _ => {
            t.at -= 1;
            if lhs.len() != 1 { return Err(format!("line {}: core instruction requires one result", t.line)); }
            Inst::Core { value: lhs[0].0, ty: lhs[0].1, op: op(t)? }
        }
    };
    t.done()?;
    Ok(inst)
}

pub(crate) fn func(registry: &DialectRegistry, text: &str) -> Result<Func> {
    let mut lines = text.lines().enumerate().map(|(n, l)| (n + 1, l.trim())).filter(|(_, l)| !l.is_empty());
    let (line, header) = lines.next().ok_or("empty function")?;
    let mut t = Tokens::new(line, header);
    t.expect("func")?; t.expect("entry")?;
    let entry = t.block()?; t.done()?;
    let mut blocks = BTreeMap::new();
    let mut types: BTreeMap<ValueId, Ty> = BTreeMap::new();
    let mut define = |id: ValueId, ty: Ty, line: usize| -> Result<()> {
        if types.insert(id, ty).is_some() { return Err(format!("line {line}: duplicate definition of v{}", id.0)); }
        Ok(())
    };
    let mut current: Option<(BlockId, Block)> = None;
    for (line, text) in lines {
        let mut t = Tokens::new(line, text);
        if text.starts_with('b') && text.ends_with("):") {
            if let Some((id, block)) = current.take() { blocks.insert(id, block); }
            let id = t.block()?;
            t.expect("(")?;
            let mut params = Vec::new();
            if !t.accept(")") {
                loop {
                    let v = t.value()?; t.expect(":")?; let ty = t.ty()?; define(v, ty, line)?; params.push((v, ty));
                    if t.accept(")") { break; }
                    t.expect(",")?;
                }
            }
            t.expect(":")?; t.done()?;
            current = Some((id, Block { params, insts: Vec::new(), term: Term::Ret(vec![]) }));
            continue;
        }
        let (_, block) = current.as_mut().ok_or_else(|| format!("line {line}: instruction outside a block"))?;
        match t.peek() {
            Some("ret") => { t.next()?; let mut args = Vec::new(); while t.peek().is_some() { if !args.is_empty() { t.expect(",")?; } args.push(t.value()?); } t.done()?; block.term = Term::Ret(args); }
            Some("br") => { t.next()?; block.term = Term::Br(t.edge()?); t.done()?; }
            Some("condbr") => {
                t.next()?;
                let cond = t.value()?; t.expect(",")?; let yes = t.edge()?; t.expect(",")?; let no = t.edge()?; t.done()?;
                block.term = Term::CondBr { cond, yes, no };
            }
            _ => {
                let i = inst(registry, &mut t)?;
                match &i {
                    Inst::Core { value, ty, .. } => define(*value, *ty, line)?,
                    Inst::Packet { op, output, .. } => define(*output, op.result_type(), line)?,
                    Inst::Target { outputs, .. } | Inst::Effect { outputs, .. } => for &(id, ty) in outputs { define(id, ty, line)?; },
                }
                block.insts.push(i);
            }
        }
    }
    if let Some((id, block)) = current.take() { blocks.insert(id, block); }
    let count = types.keys().next_back().map_or(0, |id| id.0 + 1);
    let mut all = vec![Ty::I32; count];
    for (id, ty) in types { all[id.0] = ty; }
    Ok(Func { entry, blocks, types: all })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::ir::print;

    fn registry() -> DialectRegistry { crate::rdna_spmd::targets::rdna4::registry() }

    #[test]
    fn every_operation_form_prints_and_parses_back_to_the_same_function() {
        let registry = registry();
        let rcp = registry.lookup(crate::rdna_spmd::targets::rdna4::dialect::ID, "rcp.f32").unwrap();
        let semantics = MemorySemantics { scope: Scope::Device, ordering: Ordering::Relaxed, cache_policy: CachePolicy::NearNonTemporalFarWriteBack, volatile: true, deferred_scope: false };
        let mut f = Func { entry: BlockId(4), blocks: BTreeMap::new(), types: vec![] };
        let p0 = f.value(Ty::I32); let p1 = f.value(Ty::I64); let p2 = f.value(Ty::I1); let p3 = f.value(Ty::F32);
        let mut insts = Vec::new();
        let mut core = |f: &mut Func, ty, op| { let v = f.value(ty); insts.push(Inst::Core { value: v, ty, op }); v };
        let k = core(&mut f, Ty::I32, Op::Const(Ty::I32, 0x1f));
        let add = core(&mut f, Ty::I32, Op::Int(IntOp::AShr, p0, k));
        let cmp = core(&mut f, Ty::I1, Op::Cmp(IntPred::Sge, add, k));
        let pair = core(&mut f, Ty::I64, Op::Pack64(p0, add));
        let lo = core(&mut f, Ty::I32, Op::UnpackLo(pair));
        let _hi = core(&mut f, Ty::I32, Op::UnpackHi(p1));
        let fl = core(&mut f, Ty::F32, Op::Float(FloatOp::MinNum, p3, p3));
        let neg = core(&mut f, Ty::F32, Op::Unary(FloatUnary::Neg, fl));
        let fc = core(&mut f, Ty::I1, Op::FCmp(FloatPred::Une, neg, fl));
        let fma = core(&mut f, Ty::F32, Op::Fma(fl, neg, p3));
        let _mad = core(&mut f, Ty::F32, Op::MulAdd(fl, neg, fma));
        let d = core(&mut f, Ty::F64, Op::Convert(Cvt::FloatResizeRte, Ty::F64, fma));
        let _bits = core(&mut f, Ty::I64, Op::Convert(Cvt::Bitcast, Ty::I64, d));
        let _sel = core(&mut f, Ty::I32, Op::Select(fc, lo, add));
        let _env = core(&mut f, Ty::I32, Op::Env(Env::PacketLaneId));
        let _sb = core(&mut f, Ty::I64, Op::Env(Env::ScratchSize));
        let _tz = core(&mut f, Ty::I32, Op::TrailingZeros(add));
        let _lz = core(&mut f, Ty::I64, Op::LeadingZeros(pair));
        let _pc = core(&mut f, Ty::I32, Op::PopulationCount(add));
        let _rb = core(&mut f, Ty::I32, Op::ReverseBits(add));
        let any = f.value(Ty::I1); insts.push(Inst::Packet { op: PacketOp::Any, input: cmp, output: any });
        let ballot = f.value(Ty::I32); insts.push(Inst::Packet { op: PacketOp::Ballot, input: p2, output: ballot });
        let t = f.value(Ty::F32); insts.push(Inst::Target { provenance: None, op: rcp, args: Arguments::Unary(p3), outputs: vec![(t, Ty::F32)] });
        let loaded = f.value(Ty::I32);
        insts.push(Inst::Effect { provenance: 7, op: EffectOp::Memory { space: Space::Global, op: MemoryOp::Load(MemSize::I16), semantics }, inputs: vec![p1, p2], outputs: vec![(loaded, Ty::I32)] });
        insts.push(Inst::Effect { provenance: 8, op: EffectOp::Memory { space: Space::Lds, op: MemoryOp::Store(MemSize::B32), semantics }, inputs: vec![p0, loaded, p2], outputs: vec![] });
        let w = f.value(Ty::I32); insts.push(Inst::Effect { provenance: 9, op: EffectOp::Wave(WaveOp::ReadLane), inputs: vec![p0, loaded, k], outputs: vec![(w, Ty::I32)] });
        let first = f.value(Ty::I1); insts.push(Inst::Effect { provenance: 10, op: EffectOp::BarrierSignal { is_first: true }, inputs: vec![p0], outputs: vec![(first, Ty::I1)] });
        insts.push(Inst::Effect { provenance: 11, op: EffectOp::BarrierWait, inputs: vec![w], outputs: vec![] });
        f.blocks.insert(BlockId(4), Block { params: vec![(p0, Ty::I32), (p1, Ty::I64), (p2, Ty::I1), (p3, Ty::F32)], insts,
            term: Term::CondBr { cond: first, yes: Edge { dst: BlockId(8), args: vec![w] }, no: Edge { dst: BlockId(4), args: vec![add, pair, any, fma] } } });
        let q = f.value(Ty::I32);
        f.blocks.insert(BlockId(8), Block { params: vec![(q, Ty::I32)], insts: vec![], term: Term::Ret(vec![q, q]) });
        f.clone().verify_with(&registry).unwrap();
        let text = print::func(&registry, &f);
        let parsed = func(&registry, &text).unwrap();
        assert_eq!(parsed, f);
        assert_eq!(print::func(&registry, &parsed), text);
    }

    #[test]
    fn malformed_text_is_rejected_with_a_line_number() {
        let registry = registry();
        for (text, needle) in [
            ("func entry b0\nb0():\n  v0: i32 = int add v1\n  ret\n", "line 3"),
            ("func entry b0\nb0():\n  v0: i32 = target rdna4.nothing(v0)\n  ret\n", "unregistered"),
            ("func entry b0\nb0(v0: i32):\n  v0: i32 = const i32 1\n  ret\n", "duplicate"),
            ("func entry b0\n  ret\n", "outside a block"),
            ("func entry b0\nb0():\n  v0: i32 = const i32 1 extra\n  ret\n", "trailing"),
        ] {
            let error = func(&registry, text).unwrap_err();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn lifted_kernel_objects_round_trip_through_the_printer() {
        let registry = registry();
        for &(path, symbol) in crate::rdna_spmd::targets::rdna4::decode::OBJECTS {
            let (entry, memory) = crate::rdna_spmd::targets::rdna4::decode::load_object(path, symbol);
            let program = crate::rdna_spmd::decode_program("gfx1200", entry, &memory).unwrap();
            let text = print::func(&registry, &program.function.ir);
            let parsed = func(&registry, &text).unwrap();
            assert_eq!(parsed, program.function.ir);
            parsed.verify_with(&registry).unwrap();
        }
    }
}
