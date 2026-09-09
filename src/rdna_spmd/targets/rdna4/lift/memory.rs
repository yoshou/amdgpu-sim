//! Memory decoding and address semantics; no LLVM or execution-width knowledge.
use crate::rdna_spmd::ir::{*};
use super::input;
use crate::{
    instructions::I,
    rdna_instructions::{InstFormat, SourceOperand},
};

#[derive(Clone, Debug)]
pub(in crate::rdna_spmd) enum Address {
    Global {
        scalar: Option<u32>,
        vector: u32,
        offset: i64,
    },
    Scalar {
        base: u32,
        offset: i64,
        scalar_offset: Option<u32>,
    },
    Scratch {
        scalar: Option<u32>,
        vector: Option<u32>,
        offset: i64,
    },
    Lds {
        vector: u32,
        offset: u32,
    },
    Flat {
        scalar: Option<u32>,
        vector: u32,
        offset: i64,
    },
}
#[derive(Clone, Debug)]
pub(in crate::rdna_spmd) struct Memory {
    pub address: Address,
    pub op: MemoryOp,
    pub words: u32,
    pub dest: u32,
    pub data: u32,
    pub returns: bool,
    pub semantics: MemorySemantics,
    pair: Option<LdsPair>,
}
#[derive(Clone, Debug)]
struct LdsPair {
    offsets: [u32; 2],
    second_data: u32,
}
impl Memory {
    /// Byte displacement of one word, including noncontiguous LDS addresses.
    pub fn word_offset(&self, word: u32) -> u32 {
        match &self.pair {
            Some(pair) => {
                let n = self.words / 2;
                pair.offsets[(word / n) as usize] + (word % n) * 4
            }
            None => word * 4,
        }
    }
    pub fn data_register(&self, word: u32) -> u32 {
        match &self.pair {
            Some(pair) if word >= self.words / 2 => pair.second_data + word - self.words / 2,
            _ => self.data + word,
        }
    }
    pub fn reads(&self) -> Vec<u32> {
        let mut regs = vec![];
        match self.address {
            Address::Global { scalar, vector, .. } | Address::Flat { scalar, vector, .. } => {
                regs.push(vector);
                if scalar.is_none() {
                    regs.push(vector + 1);
                }
            }
            Address::Scratch { vector, .. } => regs.extend(vector),
            Address::Lds { vector, .. } => regs.push(vector),
            _ => {}
        }
        if self.stores() || self.op == MemoryOp::AtomicAdd {
            regs.extend((0..self.words).map(|k| self.data_register(k)));
        }
        regs
    }
    pub fn space(&self) -> Space {
        match self.address {
            Address::Scratch { .. } => Space::Scratch,
            Address::Lds { .. } => Space::Lds,
            _ => Space::Global,
        }
    }
    pub fn scalar(&self) -> bool {
        matches!(self.address, Address::Scalar { .. })
    }
    pub fn stores(&self) -> bool {
        matches!(self.op, MemoryOp::Store(_))
    }
}
fn sext(v: u32) -> i64 {
    ((v << 8) as i32 >> 8) as i64
}

/// RDNA4 ISA §4.1.4, Tables 12–16. Reserved fields are rejected at lift.
/// https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content
fn semantics(scope: u8, th: u8, op: MemoryOp, scalar: bool) -> MemorySemantics {
    let scope = match scope {
        0 => Scope::ComputeUnit,
        1 => Scope::ShaderEngine,
        2 => Scope::Device,
        3 => Scope::System,
        _ => panic!("invalid memory scope {}", scope),
    };
    assert!(
        th < if scalar { 4 } else { 8 },
        "reserved memory temporal hint {}",
        th
    );
    let (cache_policy, deferred_scope) = if op == MemoryOp::AtomicAdd {
        (
            if th & 2 == 0 {
                CachePolicy::Temporal
            } else {
                CachePolicy::NonTemporal
            },
            th & 5 == 4,
        )
    } else {
        (
            match th {
                0 => CachePolicy::Temporal,
                1 => CachePolicy::NonTemporal,
                2 => CachePolicy::HighPriority,
                3 => {
                    if matches!(op, MemoryOp::Store(_)) {
                        CachePolicy::WriteBack
                    } else {
                        CachePolicy::LastUse
                    }
                }
                4 => CachePolicy::NearNonTemporal,
                5 => CachePolicy::FarNonTemporal,
                6 => CachePolicy::NearNonTemporalFarHigh,
                7 if matches!(op, MemoryOp::Store(_)) => CachePolicy::NearNonTemporalFarWriteBack,
                _ => panic!("reserved load temporal hint {}", th),
            },
            false,
        )
    };
    MemorySemantics {
        scope,
        cache_policy,
        deferred_scope,
        volatile: false,
        ordering: if op == MemoryOp::AtomicAdd {
            Ordering::Sequential
        } else {
            Ordering::Relaxed
        },
    }
}

pub(in crate::rdna_spmd) fn instruction(inst: &InstFormat) -> Option<Memory> {
    let (opcode, mut address, dest, data, scope, th) = match inst {
        InstFormat::SMEM(i) => (
            i.op,
            Address::Scalar {
                base: i.sbase as u32 * 2,
                offset: sext(i.ioffset),
                scalar_offset: if i.soffset == 124 {
                    None
                } else {
                    Some(i.soffset as u32)
                },
            },
            i.sdata,
            0,
            i.scope,
            i.th,
        ),
        InstFormat::VGLOBAL(i) => (
            i.op,
            Address::Global {
                scalar: (i.saddr != 124).then_some(i.saddr as u32),
                vector: i.vaddr as u32,
                offset: sext(i.ioffset),
            },
            i.vdst,
            i.vsrc,
            i.scope,
            i.th,
        ),
        InstFormat::VFLAT(i) => (
            i.op,
            Address::Flat {
                scalar: (i.saddr != 124).then_some(i.saddr as u32),
                vector: i.vaddr as u32,
                offset: sext(i.ioffset),
            },
            i.vdst,
            i.vsrc,
            i.scope,
            i.th,
        ),
        InstFormat::VSCRATCH(i) => (
            i.op,
            Address::Scratch {
                scalar: (!matches!(i.saddr, 124 | 127)).then_some(i.saddr as u32),
                vector: (i.sve != 0).then_some(i.vaddr as u32),
                offset: sext(i.ioffset),
            },
            i.vdst,
            i.vsrc,
            i.scope,
            i.th,
        ),
        InstFormat::DS(i) if !matches!(i.op, I::DS_BPERMUTE_B32 | I::DS_BPERMUTE_FI_B32) => (
            i.op,
            Address::Lds {
                vector: i.addr as u32,
                offset: i.offset0 as u32 | ((i.offset1 as u32) << 8),
            },
            i.vdst,
            i.data0,
            0,
            0,
        ),
        _ => return None,
    };
    use MemSize::*;
    use MemoryOp::*;
    let (op, words) = match opcode {
        I::S_LOAD_U16
        | I::GLOBAL_LOAD_U16
        | I::FLAT_LOAD_U16
        | I::SCRATCH_LOAD_U16
        | I::DS_LOAD_U16 => (Load(U16), 1),
        I::S_LOAD_I16 | I::GLOBAL_LOAD_I16 | I::FLAT_LOAD_I16 | I::SCRATCH_LOAD_I16 | I::DS_LOAD_I16 => {
            (Load(I16), 1)
        }
        I::S_LOAD_U8 | I::GLOBAL_LOAD_U8 | I::FLAT_LOAD_U8 | I::SCRATCH_LOAD_U8 | I::DS_LOAD_U8 => (Load(U8), 1),
        I::S_LOAD_I8 | I::GLOBAL_LOAD_I8 | I::FLAT_LOAD_I8 | I::SCRATCH_LOAD_I8 | I::DS_LOAD_I8 => (Load(I8), 1),
        I::S_LOAD_B32
        | I::GLOBAL_LOAD_B32
        | I::FLAT_LOAD_B32
        | I::SCRATCH_LOAD_B32
        | I::DS_LOAD_B32 => (Load(B32), 1),
        I::S_LOAD_B64
        | I::GLOBAL_LOAD_B64
        | I::FLAT_LOAD_B64
        | I::SCRATCH_LOAD_B64
        | I::DS_LOAD_B64 => (Load(B32), 2),
        I::S_LOAD_B96 | I::GLOBAL_LOAD_B96 | I::FLAT_LOAD_B96 | I::SCRATCH_LOAD_B96 | I::DS_LOAD_B96 => {
            (Load(B32), 3)
        }
        I::S_LOAD_B128 | I::GLOBAL_LOAD_B128 | I::FLAT_LOAD_B128 | I::SCRATCH_LOAD_B128 | I::DS_LOAD_B128 => {
            (Load(B32), 4)
        }
        I::S_LOAD_B256 => (Load(B32), 8),
        I::S_LOAD_B512 => (Load(B32), 16),
        I::DS_LOAD_2ADDR_B32 | I::DS_LOAD_2ADDR_STRIDE64_B32 => (Load(B32), 2),
        I::DS_LOAD_2ADDR_B64 | I::DS_LOAD_2ADDR_STRIDE64_B64 => (Load(B32), 4),
        I::DS_STORE_2ADDR_B32 | I::DS_STORE_2ADDR_STRIDE64_B32 => (Store(B32), 2),
        I::DS_STORE_2ADDR_B64 | I::DS_STORE_2ADDR_STRIDE64_B64 => (Store(B32), 4),
        I::GLOBAL_STORE_B8 | I::FLAT_STORE_B8 | I::SCRATCH_STORE_B8 | I::DS_STORE_B8 => {
            (Store(U8), 1)
        }
        I::GLOBAL_STORE_B16 | I::FLAT_STORE_B16 | I::SCRATCH_STORE_B16 | I::DS_STORE_B16 => {
            (Store(U16), 1)
        }
        I::GLOBAL_STORE_B32 | I::FLAT_STORE_B32 | I::SCRATCH_STORE_B32 | I::DS_STORE_B32 => {
            (Store(B32), 1)
        }
        I::GLOBAL_STORE_B64 | I::FLAT_STORE_B64 | I::SCRATCH_STORE_B64 | I::DS_STORE_B64 => {
            (Store(B32), 2)
        }
        I::GLOBAL_STORE_B96 | I::FLAT_STORE_B96 | I::SCRATCH_STORE_B96 | I::DS_STORE_B96 => (Store(B32), 3),
        I::GLOBAL_STORE_B128 | I::FLAT_STORE_B128 | I::SCRATCH_STORE_B128 | I::DS_STORE_B128 => (Store(B32), 4),
        I::GLOBAL_ATOMIC_ADD_U32 | I::DS_ADD_U32 | I::DS_ADD_RTN_U32 => (AtomicAdd, 1),
        I::GLOBAL_WB | I::GLOBAL_INV => (Fence, 0),
        _ => panic!("unsupported memory lift {:?}", opcode),
    };
    // RDNA4 ISA §16.15 DS_LOAD/STORE_2ADDR: each offset counts elements
    // (4 or 8 bytes); STRIDE64 multiplies that element stride by 64. DATA1
    // supplies the second store's source independently of DATA0.
    // https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content
    let pair = if matches!(opcode, I::DS_LOAD_2ADDR_B32 | I::DS_LOAD_2ADDR_B64 |
        I::DS_STORE_2ADDR_B32 | I::DS_STORE_2ADDR_B64 | I::DS_LOAD_2ADDR_STRIDE64_B32 |
        I::DS_LOAD_2ADDR_STRIDE64_B64 | I::DS_STORE_2ADDR_STRIDE64_B32 | I::DS_STORE_2ADDR_STRIDE64_B64) {
        let InstFormat::DS(i) = inst else { unreachable!() };
        let stride = words / 2 * 4 * if matches!(opcode, I::DS_LOAD_2ADDR_STRIDE64_B32 |
            I::DS_LOAD_2ADDR_STRIDE64_B64 | I::DS_STORE_2ADDR_STRIDE64_B32 | I::DS_STORE_2ADDR_STRIDE64_B64) { 64 } else { 1 };
        address = Address::Lds { vector: i.addr as u32, offset: 0 };
        Some(LdsPair { offsets: [i.offset0 as u32 * stride, i.offset1 as u32 * stride], second_data: i.data1 as u32 })
    } else { None };
    let returns = matches!(op, Load(_))
        || op == AtomicAdd && (th & 1 != 0 || matches!(opcode, I::DS_ADD_RTN_U32));
    let mut semantics = semantics(scope, th, op, matches!(address, Address::Scalar { .. }));
    if matches!(address, Address::Lds { .. }) {
        semantics.scope = Scope::Workgroup;
    }
    if matches!(address, Address::Scratch { .. }) {
        semantics.scope = Scope::WorkItem;
    }
    if op == Fence {
        semantics.ordering = if matches!(opcode, I::GLOBAL_WB) {
            Ordering::Release
        } else {
            Ordering::Acquire
        };
    }
    Some(Memory {
        address,
        op,
        words,
        dest: dest as u32,
        data: data as u32,
        returns,
        semantics,
        pair,
    })
}

impl Memory {
    pub(super) fn lift(
        &self,
        f: &mut Func,
        block: &mut Block,
        words: &mut super::regs::Words,
        views: &mut super::regs::Views,
        provenance: &mut u64,
    ) {
        use crate::rdna_spmd::ir::Inst;
        let mut operands = super::regs::Operands::default();
        let mut reg = |source: SourceOperand, ty| operands.read(
            &input(source, ty), self.scalar(), None, f, block, words, views,
        );
        // Collect register operands first, preserving their ISA widths.
        let (s, v, offset, ty) = match self.address {
            Address::Global {
                scalar,
                vector,
                offset,
            }
            | Address::Flat {
                scalar,
                vector,
                offset,
            } => (
                scalar.map(|r| reg(SourceOperand::ScalarRegister(r as u8), Ty::I64)),
                Some(reg(
                    SourceOperand::VectorRegister(vector as u8),
                    if scalar.is_some() { Ty::I32 } else { Ty::I64 },
                )),
                offset,
                Ty::I64,
            ),
            Address::Scalar {
                base,
                offset,
                scalar_offset,
            } => (
                Some(reg(SourceOperand::ScalarRegister(base as u8), Ty::I64)),
                scalar_offset.map(|r| reg(SourceOperand::ScalarRegister(r as u8), Ty::I32)),
                offset as i64,
                Ty::I64,
            ),
            Address::Scratch {
                scalar,
                vector,
                offset,
            } => (
                scalar.map(|r| reg(SourceOperand::ScalarRegister(r as u8), Ty::I32)),
                vector.map(|r| reg(SourceOperand::VectorRegister(r as u8), Ty::I32)),
                offset,
                Ty::I32,
            ),
            Address::Lds { vector, offset } => (
                None,
                Some(reg(SourceOperand::VectorRegister(vector as u8), Ty::I32)),
                offset as i64,
                Ty::I32,
            ),
        };
        let data: Vec<_> = if self.stores() || self.op == MemoryOp::AtomicAdd {
            (0..self.words)
                .map(|k| {
                    reg(
                        SourceOperand::VectorRegister(self.data_register(k) as u8),
                        Ty::I32,
                    )
                })
                .collect()
        } else {
            vec![]
        };
        let mut core = operands.core;
        let mask = if self.scalar() {
            super::regs::core(f,&mut core,Ty::I1,Op::Const(Ty::I1,1))
        } else {
            let exec=words[&super::regs::Word::Mask(126)];
            super::regs::core(f,&mut core,Ty::I1,Op::Convert(Cvt::Bitcast,Ty::I1,exec))
        };
        let mut push = |t, op| {
            let value = f.value(t);
            core.push(Inst::Core { value, ty: t, op });
            value
        };
        let v = v.map(|v| {
            if ty == Ty::I64 && (s.is_some()) {
                push(Ty::I64, Op::Convert(Cvt::ZExt, Ty::I64, v))
            } else {
                v
            }
        });
        let base = match (s, v) {
            (Some(s), Some(v)) => push(ty, Op::Int(IntOp::Add, s, v)),
            (Some(s), None) => s,
            (None, Some(v)) => v,
            _ => push(ty, Op::Const(ty, 0)),
        };
        let off = push(
            ty,
            Op::Const(
                ty,
                if ty == Ty::I32 {
                    offset as u32 as u64
                } else {
                    offset as u64
                },
            ),
        );
        let address = push(ty, Op::Int(IntOp::Add, base, off));
        block.insts.extend(core);
        let flat = if matches!(self.address, Address::Flat { .. }) {
            let sb = f.value(Ty::I64);
            let size = f.value(Ty::I64);
            block.insts.push(Inst::Core { value: sb, ty: Ty::I64, op: Op::Env(crate::rdna_spmd::ir::Env::ScratchBase) });
            block.insts.push(Inst::Core { value: size, ty: Ty::I64, op: Op::Env(crate::rdna_spmd::ir::Env::ScratchSize) });
            let mut push = |t, op| {
                let value = f.value(t);
                block.insts.push(Inst::Core { value, ty: t, op });
                value
            };
            let hi = push(Ty::I64, Op::Int(IntOp::Add, sb, size));
            // RDNA4 ISA section 11.2: classify the base before IOFFSET.
            // Keeping this separate also shares the aperture test across fields.
            let ge = push(Ty::I1, Op::Cmp(IntPred::Uge, base, sb));
            let lt = push(Ty::I1, Op::Cmp(IntPred::Ult, base, hi));
            let inside = push(Ty::I1, Op::Int(IntOp::And, ge, lt));
            let yes = push(Ty::I1, Op::Int(IntOp::And, mask, inside));
            let one = push(Ty::I1, Op::Const(Ty::I1, 1));
            let outside = push(Ty::I1, Op::Int(IntOp::Xor, inside, one));
            let no = push(Ty::I1, Op::Int(IntOp::And, mask, outside));
            let off = push(Ty::I64, Op::Int(IntOp::Sub, address, sb));
            let private = push(Ty::I32, Op::Convert(Cvt::Trunc, Ty::I32, off));
            Some((inside, private, (yes, no)))
        } else {
            None
        };
        let instruction = *provenance << 8;
        *provenance += 1;
        let mut sub = 0u64;
        let mut next_provenance = || { let id = instruction | sub; sub += 1; id };
        if self.op == MemoryOp::Fence {
            block.insts.push(Inst::Effect {
                provenance: next_provenance(),
                op: EffectOp::Memory {
                    space: self.space(),
                    op: self.op,
                    semantics: self.semantics,
                },
                inputs: vec![],
                outputs: vec![],
            });
        }
        for k in 0..self.words {
            let a = if self.word_offset(k) == 0 {
                address
            } else {
                let off = f.value(ty);
                block.insts.push(Inst::Core {
                    value: off,
                    ty,
                    op: Op::Const(ty, self.word_offset(k) as u64),
                });
                let a = f.value(ty);
                block.insts.push(Inst::Core {
                    value: a,
                    ty,
                    op: Op::Int(IntOp::Add, address, off),
                });
                a
            };
            let mut inputs = vec![a];
            if !data.is_empty() {
                inputs.push(data[k as usize]);
            }
            inputs.push(flat.as_ref().map_or(mask, |f| f.2 .1));
            let outputs = if self.stores() {
                vec![]
            } else {
                vec![(f.value(Ty::I32), Ty::I32)]
            };
            block.insts.push(Inst::Effect {
                provenance: next_provenance(),
                op: EffectOp::Memory {
                    space: self.space(),
                    op: self.op,
                    semantics: self.semantics,
                },
                inputs,
                outputs: outputs.clone(),
            });
            let mut result = outputs.first().map(|o| o.0);
            if let Some((inside, private, (yes, _))) = &flat {
                let a = if k == 0 {
                    *private
                } else {
                    let off = f.value(Ty::I32);
                    block.insts.push(Inst::Core {
                        value: off,
                        ty: Ty::I32,
                        op: Op::Const(Ty::I32, k as u64 * 4),
                    });
                    let a = f.value(Ty::I32);
                    block.insts.push(Inst::Core {
                        value: a,
                        ty: Ty::I32,
                        op: Op::Int(IntOp::Add, *private, off),
                    });
                    a
                };
                let mut inputs = vec![a];
                if !data.is_empty() {
                    inputs.push(data[k as usize]);
                }
                inputs.push(*yes);
                let outputs = if self.stores() {
                    vec![]
                } else {
                    vec![(f.value(Ty::I32), Ty::I32)]
                };
                block.insts.push(Inst::Effect {
                    provenance: next_provenance(),
                    op: EffectOp::Memory {
                        space: Space::Scratch,
                        op: self.op,
                        semantics: self.semantics,
                    },
                    inputs,
                    outputs: outputs.clone(),
                });
                if let Some(global) = result {
                    let merged = f.value(Ty::I32);
                    block.insts.push(Inst::Core {
                        value: merged,
                        ty: Ty::I32,
                        op: Op::Select(*inside, outputs[0].0, global),
                    });
                    result = Some(merged);
                }
            }
            if self.returns {
                let result = result.unwrap();
                let word = if self.scalar() { super::regs::Word::scalar(self.dest + k) }
                    else { Some(super::regs::Word::Vgpr(self.dest + k)) };
                if let Some(word) = word {
                    let stored = if matches!(word,super::regs::Word::Mask(_)) {
                        super::regs::project(f,&mut block.insts,result)
                    } else if self.scalar() { result } else {
                        let stored = f.value(Ty::I32);
                        block.insts.push(Inst::Core { value: stored, ty: Ty::I32,
                            op: Op::Select(mask, result, words[&word]) });
                        stored
                    };
                    let stored=if word==super::regs::Word::Mask(126) {super::regs::valid_exec(f,&mut block.insts,stored)} else {stored};
                    words.insert(word, stored);
                } else if self.dest + k != 124 {
                    panic!("invalid scalar memory destination: {}", self.dest + k);
                }
            }
        }
    }
}
