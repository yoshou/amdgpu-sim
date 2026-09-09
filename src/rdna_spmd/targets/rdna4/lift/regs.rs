//! Architectural register-word dependencies and typed SSA operand views.
//! EXEC/VCC keep their explicit mask bindings; NULL is never an SSA variable.
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in crate::rdna_spmd) enum Word { Vgpr(u32), Sgpr(u32), Mask(u32) }
pub(in crate::rdna_spmd) type Words = BTreeMap<Word, ValueId>;

pub(super) fn core(f: &mut Func, insts: &mut Vec<Inst>, ty: Ty, op: Op) -> ValueId {
    let value = f.value(ty);
    insts.push(Inst::Core { value, ty, op });
    value
}
pub(super) fn query(f: &mut Func, insts: &mut Vec<Inst>, op: WaveOp, bit: ValueId) -> ValueId {
    let ty = if op == WaveOp::Any { Ty::I1 } else { Ty::I32 };
    let value = f.value(ty);
    insts.push(Inst::Effect { provenance: (1u64 << 63) | value.0 as u64,
        op: EffectOp::Wave(op), inputs: vec![bit], outputs: vec![(value,ty)] });
    value
}
/// E1/E5: extract the lane's bit in the existing packet-local mask word.
pub(super) fn project(f: &mut Func, insts: &mut Vec<Inst>, word: ValueId) -> ValueId {
    let lane = core(f,insts,Ty::I32,Op::Env(Env::PacketLaneId));
    let shifted = core(f,insts,Ty::I32,Op::Int(IntOp::LShr,word,lane));
    core(f,insts,Ty::I1,Op::Convert(Cvt::Trunc,Ty::I1,shifted))
}
pub(super) fn valid_exec(f: &mut Func, insts: &mut Vec<Inst>, bit: ValueId) -> ValueId {
    let valid=core(f,insts,Ty::I1,Op::Env(Env::ValidLane));
    core(f,insts,Ty::I1,Op::Int(IntOp::And,bit,valid))
}

impl Word {
    pub(in crate::rdna_spmd) fn scalar(r: u32) -> Option<Self> {
        if matches!(r, 106 | 126) { Some(Self::Mask(r)) }
        else { (r < 128 && r != 124).then_some(Self::Sgpr(r)) }
    }
    pub fn ty(self) -> Ty { if matches!(self, Self::Mask(_)) { Ty::I1 } else { Ty::I32 } }
    pub fn source(source: &SourceOperand) -> Option<Self> {
        match *source {
            SourceOperand::VectorRegister(r) => Some(Self::Vgpr(r as u32)),
            SourceOperand::ScalarRegister(r) => Self::scalar(r as u32),
            _ => None,
        }
    }
    pub fn offset(self, offset: u32) -> Option<Self> {
        match self {
            Self::Vgpr(r) => (r + offset < 256).then_some(Self::Vgpr(r + offset)),
            Self::Sgpr(r) | Self::Mask(r) => Self::scalar(r + offset),
        }
    }
    pub fn ordinary_span(self, ty: Ty) -> bool {
        (0..ty.bits().div_ceil(32)).all(|k| self.offset(k).is_some_and(|r| r.ty() == Ty::I32))
    }
}
pub(super) fn words(set: &RegSet) -> impl Iterator<Item = Word> + '_ {
    set.vgprs().map(Word::Vgpr).chain(set.sgprs().filter_map(Word::scalar))
}
/// Register views shared by ALU and memory consumers. The shape is a native
/// representation choice; ordinary word definitions remain width independent.
pub(super) type Views = BTreeMap<(Word, Ty, bool), ValueId>;
#[derive(Default)]
pub(super) struct Operands {
    pub bindings: Vec<(Input, ValueId)>,
    pub pairs: Vec<(ValueId, ValueId)>,
    pub core: Vec<Inst>,
}
impl Operands {
    pub fn read(&mut self, input: &Input, scalar: bool, scc: Option<ValueId>,
        f: &mut Func, block: &mut Block, words: &Words, views: &mut Views,
    ) -> ValueId {
        use crate::rdna_spmd::ir::Inst;
        if let Some(bits) = input.constant_bits() {
            let value = f.value(input.ty);
            self.core.push(Inst::Core { value, ty: input.ty, op: Op::Const(input.ty, bits) });
            return value;
        }
        if matches!(input.source, InputSource::Scc) {
            let value = scc.expect("SCC operand without its SSA definition");
            self.bindings.push((input.clone(), value));
            return query(f, &mut self.core, WaveOp::Any, value);
        }
        if let InputSource::MaskBit(r) = input.source {
            let value = words[&Word::Mask(r)];
            self.bindings.push((input.clone(), value));
            return value;
        }
        if matches!(input.source, InputSource::ExecPredicate) {
            let value = core(f, &mut self.core, Ty::I1,
                Op::Convert(Cvt::Bitcast, Ty::I1, words[&Word::Mask(126)]));
            self.bindings.push((input.clone(), value));
            return value;
        }
        if matches!(input.source, InputSource::Operand(SourceOperand::PrivateBase)) {
            let raw = core(f, &mut self.core, Ty::I64, Op::Env(Env::ScratchBase));
            let value = if input.ty == Ty::I64 { raw } else {
                let word = core(f, &mut self.core, Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, raw));
                match input.ty {
                    Ty::I32 => word,
                    Ty::I1 => project(f, &mut self.core, word),
                    Ty::F32 => core(f, &mut self.core, Ty::F32, Op::Convert(Cvt::Bitcast, Ty::F32, word)),
                    _ => panic!("invalid private-base operand type"),
                }
            };
            self.bindings.push((input.clone(), value));
            return value;
        }
        if let InputSource::Operand(source) = &input.source {
            if input.ty == Ty::I1 {
                if let SourceOperand::ScalarRegister(r @ (106 | 126)) = *source {
                    let bit=words[&Word::Mask(r as u32)];
                    let value=core(f,&mut self.core,Ty::I1,Op::Convert(Cvt::Bitcast,Ty::I1,bit));
                    self.bindings.push((input.clone(),value));
                    return value;
                }
                let key=Word::source(source).map(|word|(word,Ty::I1,scalar));
                if let Some(value)=key.and_then(|key|views.get(&key).copied()) {
                    self.bindings.push((input.clone(),value));
                    return value;
                }
                let word = self.read(&super::input(source.clone(), Ty::I32), scalar, scc, f, block, words, views);
                let bit=project(f, &mut self.core, word);
                if let Some(key)=key {views.insert(key,bit);}
                self.bindings.push((input.clone(),bit));
                return bit;
            }
            if let SourceOperand::ScalarRegister(r) = *source {
                let count = input.ty.bits().div_ceil(32);
                if (0..count).any(|k| matches!(r as u32 + k,106|124|126)) {
                    let mut parts = vec![];
                    for k in 0..count {
                        let reg = r as u32 + k;
                        if matches!(reg,106|126) {
                            let bit = words[&Word::Mask(reg)];
                            self.bindings.push((Input { source: InputSource::MaskBit(reg), ty: Ty::I1 },bit));
                            parts.push(query(f,&mut self.core,WaveOp::Ballot,bit));
                        } else {
                            parts.push(self.read(&super::input(SourceOperand::ScalarRegister(reg as u8),Ty::I32),scalar,scc,f,block,words,views));
                        }
                    }
                    let raw = if count == 1 { parts[0] } else {
                        core(f,&mut self.core,Ty::I64,Op::Pack64(parts[0],parts[1]))
                    };
                    return if input.ty.integer() { raw } else {
                        core(f,&mut self.core,input.ty,Op::Convert(Cvt::Bitcast,input.ty,raw))
                    };
                }
            }
        }
        let word = match &input.source {
            InputSource::Operand(source) => Word::source(source).filter(|r| r.ordinary_span(input.ty)),
            _ => None,
        };
        let key = word.map(|word| (word, input.ty, scalar));
        if input.ty != Ty::I32 {
            if let Some(value) = key.and_then(|key| views.get(&key).copied()) {
                self.bindings.push((input.clone(), value));
                return value;
            }
        }
        if let Some(word) = word {
            if input.ty.bits() == 32 {
                let raw = words[&word];
                self.bindings.push((Input { source: input.source.clone(), ty: Ty::I32 }, raw));
                if input.ty == Ty::I32 { return raw; }
                let value = f.value(Ty::F32);
                self.core.push(Inst::Core { value, ty: Ty::F32, op: Op::Convert(Cvt::Bitcast, Ty::F32, raw) });
                views.insert(key.unwrap(), value);
                return value;
            }
            if input.ty.bits() == 64 {
                let pair = f.value(Ty::I64);
                self.core.push(Inst::Core { value: pair, ty: Ty::I64,
                    op: Op::Pack64(words[&word], words[&word.offset(1).unwrap()]) });
                let value = if input.ty == Ty::F64 {
                    let value = f.value(Ty::F64);
                    self.core.push(Inst::Core { value, ty: Ty::F64, op: Op::Convert(Cvt::Bitcast, Ty::F64, pair) });
                    value
                } else { pair };
                self.bindings.push((input.clone(), value));
                self.pairs.push((pair, value));
                views.insert(key.unwrap(), value);
                return value;
            }
        }
        panic!("operand lacks an explicit SSA definition: {:?}", input)
    }
}

fn source(set: &mut RegSet, src: &SourceOperand, count: u32) {
    for k in 0..count {
        match *src {
            SourceOperand::ScalarRegister(r) => set.add_sgpr(r as u32 + k),
            SourceOperand::VectorRegister(r) => set.add_vgpr(r as u32 + k),
            _ => {}
        }
    }
}
pub(super) fn footprint(lowering: &Lowering) -> BoundaryIo {
    let mut io = BoundaryIo::default();
    match lowering {
        Lowering::TypedAlu { inputs, outputs, .. } => {
            for input in inputs {
                match &input.source {
                    InputSource::Operand(s) => source(&mut io.reads, s, input.ty.bits().div_ceil(32)),
                    InputSource::Scc => io.reads.scc = true,
                    InputSource::MaskBit(r) => io.reads.add_sgpr(*r),
                    InputSource::ExecPredicate => io.reads.add_sgpr(126),
                }
            }
            for output in outputs {
                match *output {
                    Output::Vgpr(r, t) => for k in 0..t.bits().div_ceil(32) { io.writes.add_vgpr(r + k); },
                    Output::Scalar(r, t) => for k in 0..t.bits().div_ceil(32) { io.writes.add_sgpr(r + k); },
                    Output::Compare(r) | Output::Mask(r) => io.writes.add_sgpr(r),
                    Output::Scc => io.writes.scc = true,
                }
            }
        }
        Lowering::Memory(m) => {
            use memory::Address;
            for r in m.reads() { io.reads.add_vgpr(r); }
            match m.address {
                Address::Global { scalar: Some(r), .. } | Address::Flat { scalar: Some(r), .. } => {
                    io.reads.add_sgpr(r); io.reads.add_sgpr(r + 1);
                }
                Address::Scalar { base, scalar_offset, .. } => {
                    io.reads.add_sgpr(base); io.reads.add_sgpr(base + 1);
                    if let Some(r) = scalar_offset { io.reads.add_sgpr(r); }
                }
                Address::Scratch { scalar: Some(r), .. } => io.reads.add_sgpr(r),
                _ => {}
            }
            if m.returns {
                for r in m.dest..m.dest + m.words {
                    if m.scalar() { io.writes.add_sgpr(r); } else { io.writes.add_vgpr(r); }
                }
            }
        }
        Lowering::Wave(action) => return action.io(),

    }
    io
}

pub(super) struct Definitions {
    pub stored: Vec<(ValueId, Ty)>,
}

pub(super) fn define(
    f: &mut Func, insts: &mut Vec<Inst>, words: &mut Words, results: &[(Output, ValueId)],
    previous: &[Option<ValueId>], scalar: bool, scc: &mut ValueId, first_value: usize, writes: &[Word],
) -> Definitions {
    use crate::rdna_spmd::ir::Inst;
    for &(output, value) in results {
        if matches!(output, Output::Scc) { *scc = if scalar { value } else { query(f, insts, WaveOp::Any, value) }; }
    }
    let exec = core(f, insts, Ty::I1, Op::Convert(Cvt::Bitcast, Ty::I1, words[&Word::Mask(126)]));
    let mut stored = Vec::new();
    let mut word_defs = BTreeMap::new();
    for (output_index, &(output, result)) in results.iter().enumerate() {
        let ty = output.ty();
        let value = match output {
            Output::Vgpr(_, _) => {
                let old = previous[output_index].unwrap();
                let value = f.value(ty);
                insts.push(Inst::Core { value, ty, op: Op::Select(exec, result, old) });
                value
            }
            Output::Scalar(..) | Output::Scc => result,
            Output::Compare(_) | Output::Mask(_) => core(f, insts, Ty::I1, Op::Int(IntOp::And, result, exec)),
        };
        stored.push((value, ty));
        if let Output::Compare(reg) | Output::Mask(reg) = output {
            if let Some(word) = Word::scalar(reg) {
                let raw = if matches!(word, Word::Mask(_)) { value } else { query(f, insts, WaveOp::Ballot, value) };
                let raw = if word == Word::Mask(126) { valid_exec(f, insts, raw) } else { raw };
                word_defs.insert(word, raw);
            }
            continue;
        }
        let (reg, scalar) = match output {
            Output::Vgpr(r, _) => (r, false),
            Output::Scalar(r, _) => (r, true),
            Output::Scc => continue,
            _ => unreachable!(),
        };
        let bits = if matches!(ty, Ty::F32 | Ty::F64) {
            let int_ty = if ty == Ty::F32 { Ty::I32 } else { Ty::I64 };
            core(f, insts, int_ty, Op::Convert(Cvt::Bitcast, int_ty, value))
        } else { value };
        for k in 0..ty.bits().div_ceil(32) {
            let word = if scalar { Word::scalar(reg + k) } else { Some(Word::Vgpr(reg + k)) };
            if let Some(word) = word {
                let raw = if ty.bits() == 32 { bits } else {
                    core(f, insts, Ty::I32, if k == 0 { Op::UnpackLo(bits) } else { Op::UnpackHi(bits) })
                };
                let raw = if matches!(word, Word::Mask(_)) { project(f, insts, raw) } else { raw };
                let raw = if word == Word::Mask(126) { valid_exec(f, insts, raw) } else { raw };
                word_defs.insert(word, raw);
            }
        }
    }
    for (word, raw) in &mut word_defs {
        if matches!(word, Word::Sgpr(_)) && raw.0 < first_value {
            *raw = core(f, insts, Ty::I32, Op::Convert(Cvt::Bitcast, Ty::I32, *raw));
        }
    }
    for &r in writes {
        words.insert(r, *word_defs.get(&r).expect("typed output lacks its architectural SSA definition"));
    }
    Definitions { stored }
}

pub(super) fn branch(
    f: &mut Func, insts: &mut Vec<Inst>, words: &Words, scc: ValueId, cond: crate::rdna_spmd::targets::rdna4::decode::Cond,
) -> ValueId {
    use crate::rdna_spmd::targets::rdna4::decode::Cond;
    use crate::rdna_spmd::ir::Inst;
    let input_value = match cond {
        Cond::Scc0 | Cond::Scc1 => scc,
        Cond::ExecZ | Cond::ExecNz => words[&Word::Mask(126)],
        Cond::VccZ | Cond::VccNz => words[&Word::Mask(106)],
    };
    let query = if matches!(cond, Cond::Scc0 | Cond::Scc1) { input_value } else {
        let query = f.value(Ty::I1);
        insts.push(Inst::Packet { op: PacketOp::Any, input: input_value, output: query });
        query
    };
    let result = if matches!(cond, Cond::Scc0 | Cond::ExecZ | Cond::VccZ) {
        let zero = f.value(Ty::I1);
        insts.push(Inst::Core { value: zero, ty: Ty::I1, op: Op::Const(Ty::I1, 0) });
        let result = f.value(Ty::I1);
        insts.push(Inst::Core { value: result, ty: Ty::I1, op: Op::Cmp(IntPred::Eq, query, zero) });
        result
    } else { query };
    result
}

#[derive(Clone, Copy, Default, PartialEq)]
pub(crate) struct RegSet {
    pub(crate) scc: bool,
    sgpr: u128,
    vgpr: [u128; 2],
}

impl RegSet {
    /// Registers outside the architectural files (128 SGPRs, 256 VGPRs) are
    /// dropped: operand encodings can name reserved indices, and aliasing one
    /// onto a real register would be worse than ignoring it.
    pub(crate) fn add_sgpr(&mut self, reg: u32) {
        if reg < 128 {
            self.sgpr |= 1 << reg;
        }
    }
    pub(crate) fn add_vgpr(&mut self, reg: u32) {
        if reg < 256 {
            self.vgpr[(reg >> 7) as usize] |= 1 << (reg & 127);
        }
    }
    pub(crate) fn has_sgpr(&self, reg: u32) -> bool {
        reg < 128 && self.sgpr & (1 << reg) != 0
    }
    pub(crate) fn has_vgpr(&self, reg: u32) -> bool {
        reg < 256 && self.vgpr[(reg >> 7) as usize] & (1 << (reg & 127)) != 0
    }
    pub(crate) fn vgprs(&self) -> impl Iterator<Item = u32> + '_ {
        (0..256u32).filter(move |&reg| self.has_vgpr(reg))
    }
    pub(crate) fn sgprs(&self) -> impl Iterator<Item = u32> + '_ {
        (0..128u32).filter(move |&reg| self.has_sgpr(reg))
    }
}

/// What the host-applied wave-level op at a boundary touches: the registers it
/// reads out of the packet (which the kernel stores before yielding) and the
/// ones it writes back (which the kernel reloads afterwards). A partial write
/// — writelane touches one lane of the packed vector — belongs in `reads` too,
/// so the lanes it leaves alone survive the round trip.
#[derive(Clone, Copy, Default)]
pub(crate) struct BoundaryIo {
    pub(crate) reads: RegSet,
    pub(crate) writes: RegSet,
}
