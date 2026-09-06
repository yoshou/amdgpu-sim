//! Register-word dependencies during the migration to architectural state SSA.
//! EXEC/VCC keep their explicit mask bindings; NULL is never an SSA variable.
use super::*;
use crate::rdna_spmd::boundary::{BoundaryIo, RegSet};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum Word { Vgpr(u32), Sgpr(u32) }
pub(super) type Words = BTreeMap<Word, ValueId>;

impl Word {
    pub fn scalar(r: u32) -> Option<Self> {
        (r < 128 && !matches!(r, 106 | 124 | 126)).then_some(Self::Sgpr(r))
    }
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
            Self::Sgpr(r) => Self::scalar(r + offset),
        }
    }
    pub fn ordinary_span(self, ty: Ty) -> bool {
        (0..ty.bits().div_ceil(32)).all(|k| self.offset(k).is_some())
    }
}
pub(super) fn words(set: &RegSet) -> impl Iterator<Item = Word> + '_ {
    set.vgprs().map(Word::Vgpr).chain(set.sgprs().filter_map(Word::scalar))
}
pub(super) fn dependencies(source: &SourceOperand, ty: Ty, state: &Words) -> Vec<ValueId> {
    // A pair may start at a special low word and still contain an ordinary high
    // word. Enumerate the encoded pair before filtering the special words.
    (0..ty.bits().div_ceil(32)).filter_map(|k| match *source {
        SourceOperand::VectorRegister(r) => Some(Word::Vgpr(r as u32 + k)),
        SourceOperand::ScalarRegister(r) => Word::scalar(r as u32 + k),
        _ => None,
    }).filter_map(|r| state.get(&r).copied()).collect()
}
/// Register views shared by ALU and memory consumers. The shape is a native
/// representation choice; ordinary word definitions remain width independent.
pub(super) type Views = BTreeMap<(Word, Ty, bool), ValueId>;
#[derive(Default)]
pub(super) struct Operands {
    pub bindings: Vec<(Input, ValueId)>,
    pub pairs: Vec<(ValueId, ValueId)>,
    pub core: Vec<cfg::Inst>,
}
impl Operands {
    pub fn read(&mut self, input: &Input, scalar: bool, scc: Option<ValueId>,
        f: &mut cfg::Func, block: &mut cfg::Block, words: &Words, views: &mut Views,
    ) -> ValueId {
        use cfg::Inst;
        if let Some(bits) = input.constant_bits() {
            let value = f.value(input.ty);
            self.core.push(Inst::Core { value, ty: input.ty, op: Op::Const(input.ty, bits) });
            return value;
        }
        if matches!(input.source, InputSource::Scc) {
            let value = scc.expect("SCC operand without its SSA definition");
            self.bindings.push((input.clone(), value));
            return value;
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
        // Special mask/word conversions stay at the coupled E1–E6 boundary.
        let deps = match &input.source {
            InputSource::Operand(source) => dependencies(source, input.ty, words),
            _ => vec![],
        };
        let value = f.value(input.ty);
        block.insts.push(Inst::Boundary { inputs: deps, outputs: vec![(value, input.ty)] });
        if let Some(key) = key { views.insert(key, value); }
        self.bindings.push((input.clone(), value));
        value
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
pub(super) fn footprint(lowering: &Lowering<'_>) -> BoundaryIo {
    let mut io = BoundaryIo::default();
    match lowering {
        Lowering::TypedAlu { inputs, outputs, .. } => {
            for input in inputs {
                match &input.source {
                    InputSource::Operand(s) => source(&mut io.reads, s, input.ty.bits().div_ceil(32)),
                    InputSource::Scc => io.reads.scc = true,
                    InputSource::PacketMaskAny(r) => io.reads.add_sgpr(*r),
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
        Lowering::Legacy(inst) => {
            // Only the remaining adapter instructions enter here. Wide source
            // reads are conservative; destinations must describe actual writes.
            for r in crate::rdna_spmd::vec_live::vgpr_reads(inst) { io.reads.add_vgpr(r); }
            for r in crate::rdna_spmd::freshness::vgpr_writes(inst) { io.writes.add_vgpr(r); }
            match inst {
                InstFormat::SOP1(i) => {
                    source(&mut io.reads, &i.ssrc0, 1);
                    io.writes.add_sgpr(i.sdst as u32);
                    if !matches!(i.op, I::S_MOV_B32) { io.reads.add_sgpr(126); io.writes.add_sgpr(126); io.writes.scc = true; }
                }
                InstFormat::SOP2(i) => {
                    source(&mut io.reads, &i.ssrc0, 1); source(&mut io.reads, &i.ssrc1, 1);
                    io.writes.add_sgpr(i.sdst as u32); io.writes.scc = true;
                }
                InstFormat::VOP1(i) => source(&mut io.reads, &i.src0, 2),
                InstFormat::VOP3(i) => {
                    for s in [&i.src0, &i.src1, &i.src2] { source(&mut io.reads, s, 2); }
                    if matches!(i.op, I::V_CMP_CLASS_F32 | I::V_CMP_CLASS_F64 | I::V_CMP_EQ_U16 | I::V_CMP_GT_U16 | I::V_S_RCP_F32) {
                        io.writes.add_sgpr(i.vdst as u32);
                    }
                }
                InstFormat::VOP3SD(i) => {
                    for s in [&i.src0, &i.src1, &i.src2] { source(&mut io.reads, s, 2); }
                    io.writes.add_sgpr(i.sdst as u32);
                }
                InstFormat::VOP3P(i) => for s in [&i.src0, &i.src1, &i.src2] { source(&mut io.reads, s, 1); },
                InstFormat::VIMAGE(i) => for r in i.rsrc as u32..i.rsrc as u32 + 2 { io.reads.add_sgpr(r); },
                InstFormat::VSAMPLE(i) => {
                    for r in i.rsrc as u32..i.rsrc as u32 + 8 { io.reads.add_sgpr(r); }
                    for r in i.samp as u32..i.samp as u32 + 4 { io.reads.add_sgpr(r); }
                }
                _ => panic!("missing state footprint for adapter instruction {:?}", inst),
            }
        }
    }
    io
}
