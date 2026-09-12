use super::super::ir::{*, Cvt, IntOp, Op, Ty, ValueId};
use super::{Analyses, Analysis, Constants, Uniformity};

pub(crate) struct Accesses;
impl Analysis for Accesses {
    type Result = Vec<Access>;
    const NAME: &'static str = "accesses";
    fn compute(f: &Func, analyses: &Analyses) -> Self::Result {
        let constants = analyses.get::<Constants>(f);
        let uniform = analyses.get::<Uniformity>(f).uniform();
        accesses(f, &constants, &uniform)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    Scalar,
    Global { scalar_base: bool },
    Flat,
    Scratch { uniform: bool },
    Lds,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Access {
    pub block: BlockId,
    pub start: usize,
    pub end: usize,
    pub effects: Vec<usize>,
    pub op: MemoryOp,
    pub space: Space,
    pub semantics: MemorySemantics,
    pub form: Form,
    pub words: u32,
    pub base: ValueId,
    pub offset: i64,
    pub address: ValueId,
    pub mask: ValueId,
    pub inside: Option<ValueId>,
    pub offsets: Vec<u32>,
    pub data: Vec<ValueId>,
    pub results: Vec<ValueId>,
    pub private: Vec<ValueId>,
    pub used: bool,
}

impl Access {
    pub fn size(&self) -> MemSize {
        match self.op { MemoryOp::Load(s) | MemoryOp::Store(s) => s, _ => MemSize::B32 }
    }
    pub fn returns(&self) -> bool { matches!(self.op, MemoryOp::Load(_)) || self.op == MemoryOp::AtomicAdd && self.used }
    pub fn stores(&self) -> bool { matches!(self.op, MemoryOp::Store(_)) }
    pub fn scalar(&self) -> bool { self.form == Form::Scalar }
    pub fn static_scratch_end(&self, constants: &[Option<u64>]) -> Option<u32> {
        if self.space != Space::Scratch || !matches!(self.op, MemoryOp::Load(MemSize::B32) | MemoryOp::Store(MemSize::B32)) || self.semantics.volatile { return None; }
        let offset = constants[self.address.0]? as u32;
        ((offset as i32) >= 0).then(|| offset + self.words * 4)
    }
}

fn added_constant(value: ValueId, defs: &[Option<Op>], constants: &[Option<u64>], ty: Ty) -> Option<(ValueId, i64)> {
    let Some(Op::Int(IntOp::Add, a, b)) = defs[value.0] else { return None; };
    let k = constants[b.0]?;
    Some((a, if ty == Ty::I32 { k as u32 as i32 as i64 } else { k as i64 }))
}

fn flat_parts(mask: ValueId, defs: &[Option<Op>], constants: &[Option<u64>]) -> Option<(ValueId, ValueId)> {
    let outside = |v: ValueId| match defs[v.0] {
        Some(Op::Int(IntOp::Xor, inside, one)) if constants[one.0] == Some(1) => Some((inside, one)),
        _ => None,
    };
    match defs[mask.0] {
        Some(Op::Int(IntOp::And, a, b)) => outside(b).map(|(inside, _)| (a, inside)).or_else(|| outside(a).map(|(inside, _)| (b, inside))),
        _ => outside(mask).map(|(inside, one)| (one, inside)),
    }
}

fn displacement(word: ValueId, address: ValueId, defs: &[Option<Op>], constants: &[Option<u64>], ty: Ty) -> Option<i64> {
    if word == address { return Some(0); }
    added_constant(word, defs, constants, ty).filter(|(a, _)| *a == address).map(|(_, k)| k)
}

struct Word { input: ValueId, data: Option<ValueId>, mask: ValueId, result: Option<ValueId> }

fn word(inputs: &[ValueId], outputs: &[(ValueId, Ty)]) -> Word {
    Word {
        input: inputs[0],
        data: (inputs.len() == 3).then(|| inputs[1]),
        mask: *inputs.last().unwrap(),
        result: outputs.first().map(|o| o.0),
    }
}

fn accesses(f: &Func, constants: &[Option<u64>], uniform: &[bool]) -> Vec<Access> {
    let mut uses = vec![false; f.types.len()];
    for b in f.blocks.values() {
        for inst in &b.insts {
            match inst {
                Inst::Core { op, .. } => { op.map(|v| { uses[v.0] = true; v }); }
                Inst::Packet { input, .. } => uses[input.0] = true,
                Inst::Target { args, .. } => for v in args.values() { uses[v.0] = true; },
                Inst::Effect { inputs, .. } => for v in inputs { uses[v.0] = true; },
            }
        }
        for edge in b.term.edges() { for v in &edge.args { uses[v.0] = true; } }
        match &b.term { Term::CondBr { cond, .. } => uses[cond.0] = true, Term::Ret(args) => for v in args { uses[v.0] = true; }, Term::Br(_) => {} }
    }
    let defs = f.definitions();
    let mut out = Vec::new();
    for (&block, b) in &f.blocks {
        let mut index = 0;
        while index < b.insts.len() {
            let Inst::Effect { provenance, op: EffectOp::Memory { space, op, semantics }, inputs, outputs } = &b.insts[index] else { index += 1; continue; };
            let (space, op, semantics, first) = (*space, *op, *semantics, *provenance >> 8);
            let mut effects = vec![index];
            let mut words: Vec<Word> = Vec::new();
            let mut private = Vec::new();
            let mut flat = false;
            if !inputs.is_empty() { words.push(word(inputs, outputs)); }
            let mut next = index + 1;
            while next < b.insts.len() {
                match &b.insts[next] {
                    Inst::Effect { provenance, op: EffectOp::Memory { space: s, op: o, semantics: m }, inputs, outputs }
                        if *o == op && *provenance >> 8 == first => {
                        if *s == Space::Scratch && space == Space::Global {
                            flat = true;
                            effects.push(next);
                            if let Some(&(v, _)) = outputs.first() { private.push(v); }
                        } else if *s == space && *m == semantics {
                            effects.push(next);
                            words.push(word(inputs, outputs));
                        } else { break; }
                        next += 1;
                    }
                    Inst::Core { .. } => next += 1,
                    _ => break,
                }
            }
            let ty = if space == Space::Global { Ty::I64 } else { Ty::I32 };
            let mut address = words.first().map_or(ValueId(0), |w| w.input);
            let mut offsets: Vec<i64> = Vec::new();
            for candidate in [address, added_constant(address, &defs, constants, ty).map_or(address, |x| x.0)] {
                if let Some(d) = words.iter().map(|w| displacement(w.input, candidate, &defs, constants, ty)).collect::<Option<Vec<_>>>() {
                    address = candidate;
                    offsets = d;
                    break;
                }
            }
            assert_eq!(offsets.len(), words.len(), "memory words share one address expression");
            let (base, offset) = added_constant(address, &defs, constants, ty).unwrap_or((address, 0));
            let (mask, inside) = words.first().map_or((ValueId(0), None), |w| if flat {
                let (mask, inside) = flat_parts(w.mask, &defs, constants).expect("flat access lacks its aperture test");
                (mask, Some(inside))
            } else { (w.mask, None) });
            let form = if flat { Form::Flat }
                else if space == Space::Lds { Form::Lds }
                else if space == Space::Scratch { Form::Scratch { uniform: uniform[address.0] } }
                else if words.first().is_some_and(|w| constants[w.mask.0] == Some(1)) { Form::Scalar }
                else {
                    let scalar_base = matches!(defs[base.0], Some(Op::Int(IntOp::Add, a, c))
                        if matches!(defs[c.0], Some(Op::Convert(Cvt::ZExt, Ty::I64, _))) && uniform[a.0]);
                    Form::Global { scalar_base }
                };
            let results: Vec<ValueId> = words.iter().filter_map(|w| w.result).collect();
            let merged: Vec<ValueId> = if flat {
                results.iter().map(|&r| b.insts.iter().find_map(|inst| match inst {
                    Inst::Core { value, op: Op::Select(_, _, g), .. } if *g == r => Some(*value),
                    _ => None,
                }).unwrap_or(r)).collect()
            } else { results.clone() };
            let used = merged.iter().any(|&r| uses[r.0]);
            out.push(Access {
                block, start: index, end: next, effects, op, space, semantics, form,
                words: words.len() as u32, base, offset, address, mask, inside,
                offsets: offsets.iter().map(|&d| d as u32).collect(),
                data: words.iter().filter_map(|w| w.data).collect(), results, private, used,
            });
            index = next;
        }
    }
    out
}
