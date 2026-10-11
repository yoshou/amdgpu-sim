use super::super::ir::{Cvt, IntOp, Op, Ty, ValueId, *};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    Scalar,
    Global { scalar_base: bool },
    Flat,
    Scratch { uniform: bool },
    Lds,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Access {
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
    pub shared: Option<ValueId>,
    pub offsets: Vec<u32>,
    pub data: Vec<ValueId>,
    pub results: Vec<ValueId>,
    pub private: Vec<ValueId>,
    pub used: bool,
}

impl Access {
    pub fn reads(&self) -> impl Iterator<Item = ValueId> + '_ {
        IntoIterator::into_iter([self.base, self.address, self.mask])
            .chain(self.inside)
            .chain(self.shared)
            .chain(self.data.iter().copied())
    }

    pub fn size(&self) -> MemSize {
        self.op.size()
    }
    pub fn returns(&self) -> bool {
        matches!(self.op, MemoryOp::Load(_))
            || self.op.atomic() && self.used
    }
    pub fn stores(&self) -> bool {
        matches!(self.op, MemoryOp::Store(_))
    }
    pub fn scalar(&self) -> bool {
        self.form == Form::Scalar
    }
    pub fn static_scratch_end(&self, constants: &[Option<u64>]) -> Option<u32> {
        if self.space != Space::Scratch
            || !matches!(
                self.op,
                MemoryOp::Load(MemSize::B32) | MemoryOp::Store(MemSize::B32)
            )
            || self.semantics.volatile
        {
            return None;
        }
        let offset = constants[self.address.0]? as u32;
        ((offset as i32) >= 0).then(|| offset + self.words * 4)
    }
}

fn added_constant(
    value: ValueId,
    defs: &[Option<Op>],
    constants: &[Option<u64>],
    ty: Ty,
) -> Option<(ValueId, i64)> {
    let Some(Op::Int(IntOp::Add, a, b)) = defs[value.0] else {
        return None;
    };
    let k = constants[b.0]?;
    Some((
        a,
        if ty == Ty::I32 {
            k as u32 as i32 as i64
        } else {
            k as i64
        },
    ))
}

fn private_test(v: ValueId, defs: &[Option<Op>]) -> bool {
    let Some(Op::Int(IntOp::And, x, _)) = defs[v.0] else {
        return false;
    };
    let Some(Op::Cmp(IntPred::Uge, _, low)) = defs[x.0] else {
        return false;
    };
    matches!(defs[low.0], Some(Op::Env(Env::ScratchBase)))
}

fn flat_parts(
    mask: ValueId,
    defs: &[Option<Op>],
    constants: &[Option<u64>],
) -> Option<(ValueId, ValueId, Option<ValueId>)> {
    let negated = |v: ValueId| match defs[v.0] {
        Some(Op::Int(IntOp::Xor, inside, one)) if constants[one.0] == Some(1) => {
            Some((inside, one))
        }
        _ => None,
    };
    let outside = |v: ValueId| -> Option<(ValueId, ValueId, Option<ValueId>)> {
        if let Some((inside, one)) = negated(v) {
            return Some((one, inside, None));
        }
        let Some(Op::Int(IntOp::And, a, b)) = defs[v.0] else {
            return None;
        };
        let (Some((p, one)), Some((q, _))) = (negated(a), negated(b)) else {
            return None;
        };
        match (private_test(p, defs), private_test(q, defs)) {
            (true, false) => Some((one, p, Some(q))),
            (false, true) => Some((one, q, Some(p))),
            _ => None,
        }
    };
    match defs[mask.0] {
        Some(Op::Int(IntOp::And, a, b)) => outside(b)
            .map(|(_, inside, shared)| (a, inside, shared))
            .or_else(|| outside(a).map(|(_, inside, shared)| (b, inside, shared))),
        _ => outside(mask),
    }
}

fn displacement(
    word: ValueId,
    address: ValueId,
    defs: &[Option<Op>],
    constants: &[Option<u64>],
    ty: Ty,
) -> Option<i64> {
    if word == address {
        return Some(0);
    }
    added_constant(word, defs, constants, ty)
        .filter(|(a, _)| *a == address)
        .map(|(_, k)| k)
}

struct Word {
    input: ValueId,
    data: Vec<ValueId>,
    mask: ValueId,
    results: Vec<ValueId>,
}

fn word(inputs: &[ValueId], outputs: &[(ValueId, Ty)]) -> Word {
    Word {
        input: inputs[0],
        data: inputs[1..inputs.len() - 1].to_vec(),
        mask: *inputs.last().unwrap(),
        results: outputs.iter().map(|o| o.0).collect(),
    }
}

pub fn accesses(f: &Func, constants: &[Option<u64>], uniform: &[bool]) -> Vec<Access> {
    let mut uses = vec![false; f.types.len()];
    for b in f.blocks.values() {
        for inst in &b.insts {
            match inst {
                Inst::Core { op, .. } => {
                    op.map(|v| {
                        uses[v.0] = true;
                        v
                    });
                }
                Inst::Packet { input, .. } => uses[input.0] = true,
                Inst::Target { args, .. } => {
                    for v in args.values() {
                        uses[v.0] = true;
                    }
                }
                Inst::Effect { inputs, .. } => {
                    for v in inputs {
                        uses[v.0] = true;
                    }
                }
            }
        }
        for edge in b.term.edges() {
            for v in &edge.args {
                uses[v.0] = true;
            }
        }
        match &b.term {
            Term::CondBr { cond, .. } => uses[cond.0] = true,
            Term::Ret(args) => {
                for v in args {
                    uses[v.0] = true;
                }
            }
            Term::Br(_) => {}
        }
    }
    let defs = f.definitions();
    let mut out = Vec::new();
    for (&block, b) in &f.blocks {
        let mut index = 0;
        while index < b.insts.len() {
            let Inst::Effect {
                provenance,
                op:
                    EffectOp::Memory {
                        space,
                        op,
                        semantics,
                    },
                inputs,
                outputs,
            } = &b.insts[index]
            else {
                index += 1;
                continue;
            };
            let (space, op, semantics, first) = (*space, *op, *semantics, *provenance >> 8);
            let mut effects = vec![index];
            let mut words: Vec<Word> = Vec::new();
            let mut private = Vec::new();
            let mut flat = false;
            if !inputs.is_empty() {
                words.push(word(inputs, outputs));
            }
            let mut next = index + 1;
            while next < b.insts.len() {
                match &b.insts[next] {
                    Inst::Effect {
                        provenance,
                        op:
                            EffectOp::Memory {
                                space: s,
                                op: o,
                                semantics: m,
                            },
                        inputs,
                        outputs,
                    } if *o == op && *provenance >> 8 == first => {
                        if *s == Space::Scratch && space == Space::Global {
                            flat = true;
                            effects.push(next);
                            if let Some(&(v, _)) = outputs.first() {
                                private.push(v);
                            }
                        } else if *s == space && *m == semantics {
                            effects.push(next);
                            words.push(word(inputs, outputs));
                        } else {
                            break;
                        }
                        next += 1;
                    }
                    Inst::Core { .. } => next += 1,
                    _ => break,
                }
            }
            let ty = if space == Space::Global {
                Ty::I64
            } else {
                Ty::I32
            };
            let mut address = words.first().map_or(ValueId(0), |w| w.input);
            let mut offsets: Vec<i64> = Vec::new();
            for candidate in [
                address,
                added_constant(address, &defs, constants, ty).map_or(address, |x| x.0),
            ] {
                if let Some(d) = words
                    .iter()
                    .map(|w| displacement(w.input, candidate, &defs, constants, ty))
                    .collect::<Option<Vec<_>>>()
                {
                    address = candidate;
                    offsets = d;
                    break;
                }
            }
            assert_eq!(
                offsets.len(),
                words.len(),
                "memory words share one address expression"
            );

            if !flat && effects.len() == words.len() {
                let size = match op {
                    MemoryOp::Load(s) | MemoryOp::Store(s) => s.bytes() as i64,
                    _ => 4,
                };
                if let Some(gap) = offsets.windows(2).position(|w| w[1] != w[0] + size) {
                    let run = gap + 1;
                    next = effects[run];
                    effects.truncate(run);
                    words.truncate(run);
                    offsets.truncate(run);
                }
            }
            let (base, offset) =
                added_constant(address, &defs, constants, ty).unwrap_or((address, 0));
            let (mask, inside, shared) = words.first().map_or((ValueId(0), None, None), |w| {
                if flat {
                    let (mask, inside, shared) = flat_parts(w.mask, &defs, constants)
                        .expect("flat access lacks its aperture test");
                    (mask, Some(inside), shared)
                } else {
                    (w.mask, None, None)
                }
            });
            let form = if flat {
                Form::Flat
            } else if space == Space::Lds {
                Form::Lds
            } else if space == Space::Scratch {
                Form::Scratch {
                    uniform: uniform[address.0],
                }
            } else {
                let scalar_base = matches!(defs[base.0], Some(Op::Int(IntOp::Add, a, c))
                        if matches!(defs[c.0], Some(Op::Convert(Cvt::ZExt, Ty::I64, _))) && uniform[a.0]);
                Form::Global { scalar_base }
            };
            let results: Vec<ValueId> = words.iter().flat_map(|w| w.results.iter().copied()).collect();
            let merged: Vec<ValueId> = if flat {
                results
                    .iter()
                    .map(|&r| {
                        b.insts
                            .iter()
                            .find_map(|inst| match inst {
                                Inst::Core {
                                    value,
                                    op: Op::Select(_, _, g),
                                    ..
                                } if *g == r => Some(*value),
                                _ => None,
                            })
                            .unwrap_or(r)
                    })
                    .collect()
            } else {
                results.clone()
            };
            let used = merged.iter().any(|&r| uses[r.0]);
            out.push(Access {
                block,
                start: index,
                end: next,
                effects,
                op,
                space,
                semantics,
                form,
                words: words.len() as u32,
                base,
                offset,
                address,
                mask,
                inside,
                shared,
                offsets: offsets.iter().map(|&d| d as u32).collect(),
                data: words.iter().flat_map(|w| w.data.iter().copied()).collect(),
                results,
                private,
                used,
            });
            index = next;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Defs {
        defs: Vec<Option<Op>>,
        constants: Vec<Option<u64>>,
    }

    impl Defs {
        fn push(&mut self, op: Op, k: Option<u64>) -> ValueId {
            self.defs.push(Some(op));
            self.constants.push(k);
            ValueId(self.defs.len() - 1)
        }
        fn parts(&self, mask: ValueId) -> Option<(ValueId, ValueId, Option<ValueId>)> {
            flat_parts(mask, &self.defs, &self.constants)
        }
    }

    #[test]
    fn flat_parts_find_the_exec_mask_and_the_aperture_tests_of_a_flat_access() {
        let mut d = Defs { defs: Vec::new(), constants: Vec::new() };
        let mut push = |op: Op, k: Option<u64>| d.push(op, k);
        let exec = push(Op::Env(Env::ValidLane), None);
        let base = push(Op::Env(Env::ScratchBase), None);
        let size = push(Op::Env(Env::ScratchSize), None);
        let end = push(Op::Int(IntOp::Add, base, size), None);
        let p = push(Op::Env(Env::LaneId), None);
        let ge = push(Op::Cmp(IntPred::Uge, p, base), None);
        let lt = push(Op::Cmp(IntPred::Ult, p, end), None);
        let inside = push(Op::Int(IntOp::And, ge, lt), None);
        let one = push(Op::Const(Ty::I1, 1), Some(1));
        let not_inside = push(Op::Int(IntOp::Xor, inside, one), None);
        let only_private = push(Op::Int(IntOp::And, exec, not_inside), None);
        let low = push(Op::Const(Ty::I64, 1 << 48), Some(1 << 48));
        let high = push(Op::Const(Ty::I64, (1 << 48) + (1 << 32)), Some((1 << 48) + (1 << 32)));
        let sge = push(Op::Cmp(IntPred::Uge, p, low), None);
        let slt = push(Op::Cmp(IntPred::Ult, p, high), None);
        let shared = push(Op::Int(IntOp::And, sge, slt), None);
        let not_shared = push(Op::Int(IntOp::Xor, shared, one), None);
        let neither = push(Op::Int(IntOp::And, not_inside, not_shared), None);
        let with_lds = push(Op::Int(IntOp::And, exec, neither), None);
        let swapped = push(Op::Int(IntOp::And, not_shared, not_inside), None);
        let with_lds_swapped = push(Op::Int(IntOp::And, swapped, exec), None);
        assert_eq!(d.parts(only_private), Some((exec, inside, None)));
        assert_eq!(d.parts(with_lds), Some((exec, inside, Some(shared))));
        assert_eq!(d.parts(with_lds_swapped), Some((exec, inside, Some(shared))));
    }
}
