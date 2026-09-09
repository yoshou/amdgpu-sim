//! Ordered, typed effects. ISA operands are resolved by the lifter.
use super::{Ty, ValueId};

pub(crate) const SCHEDULED: u64 = 1 << 62;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Space {
    Global,
    Scratch,
    Lds,
}
impl Space {
    pub fn address_type(self) -> Ty {
        if self == Self::Global {
            Ty::I64
        } else {
            Ty::I32
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MemSize {
    U8,
    I8,
    U16,
    I16,
    B32,
}
impl MemSize {
    pub fn bytes(self) -> u32 {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::B32 => 4,
        }
    }
    pub fn signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    WorkItem,
    ComputeUnit,
    ShaderEngine,
    Device,
    System,
    Workgroup,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ordering {
    Relaxed,
    Acquire,
    Release,
    Sequential,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CachePolicy {
    Temporal,
    NonTemporal,
    HighPriority,
    LastUse,
    WriteBack,
    NearNonTemporal,
    FarNonTemporal,
    NearNonTemporalFarHigh,
    NearNonTemporalFarWriteBack,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MemorySemantics {
    pub scope: Scope,
    pub ordering: Ordering,
    pub cache_policy: CachePolicy,
    pub volatile: bool,
    pub deferred_scope: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MemoryOp {
    Load(MemSize),
    Store(MemSize),
    AtomicAdd,
    Fence,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WaveOp {
    Any,
    Ballot,
    ReadFirstLane,
    ReadLane,
    WriteLane,
    Bpermute,
    BpermuteFi,
    Wmma,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EffectOp {
    Memory {
        space: Space,
        op: MemoryOp,
        semantics: MemorySemantics,
    },
    Wave(WaveOp),
    BarrierSignal {
        is_first: bool,
    },
    BarrierWait,
}
impl EffectOp {
    pub fn signature(self) -> (Vec<Ty>, Vec<Ty>) {
        use Ty::*;
        match self {
            Self::Memory { space, op, .. } => match op {
                MemoryOp::Load(_) => (vec![space.address_type(), I1], vec![I32]),
                MemoryOp::Store(_) => (vec![space.address_type(), I32, I1], vec![]),
                MemoryOp::AtomicAdd => (vec![space.address_type(), I32, I1], vec![I32]),
                MemoryOp::Fence => (vec![], vec![]),
            },
            Self::Wave(op) => match op {
                WaveOp::Any => (vec![I1], vec![I1]),
                WaveOp::Ballot => (vec![I1], vec![I32]),
                WaveOp::ReadFirstLane => (vec![I32, I1], vec![I32]),
                WaveOp::ReadLane => (vec![I32, I32, I32], vec![I32]),
                WaveOp::WriteLane => (vec![I32, I32, I32, I32], vec![I32]),
                WaveOp::Bpermute | WaveOp::BpermuteFi => (vec![I32, I32, I1], vec![I32]),
                WaveOp::Wmma => ([vec![I32; 8], vec![F32; 8]].concat(), vec![F32; 8]),
            },
            Self::BarrierSignal { is_first } => {
                (vec![I32], if is_first { vec![I1] } else { vec![] })
            }
            Self::BarrierWait => (vec![I32], vec![]),
        }
    }
    pub fn verify(
        self,
        inputs: &[ValueId],
        outputs: &[(ValueId, Ty)],
        types: &[Ty],
    ) -> Result<(), &'static str> {
        let (args, results) = self.signature();
        if inputs.len() != args.len() || outputs.len() != results.len() {
            return Err("effect arity mismatch");
        }
        if inputs
            .iter()
            .zip(args)
            .any(|(v, t)| types.get(v.0) != Some(&t))
            || outputs.iter().zip(results).any(|((_, t), r)| *t != r)
        {
            return Err("effect type mismatch");
        }
        if let Self::Memory {
            space,
            op,
            semantics,
        } = self
        {
            if space == Space::Scratch && matches!(op, MemoryOp::AtomicAdd | MemoryOp::Fence) {
                return Err("invalid private-memory effect");
            }
            if matches!(op, MemoryOp::Store(MemSize::I8 | MemSize::I16)) {
                return Err("signed store size");
            }
            if matches!(op, MemoryOp::Load(_))
                && matches!(semantics.ordering, Ordering::Release)
            {
                return Err("release load");
            }
            if matches!(op, MemoryOp::Store(_))
                && matches!(semantics.ordering, Ordering::Acquire)
            {
                return Err("acquire store");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Block, BlockId, Func, Inst, Term};
    use super::*;
    use std::collections::BTreeMap;
    #[test]
    fn effects_reject_wrong_spaces_masks_results_and_duplicate_provenance() {
        let semantics = MemorySemantics {
            scope: Scope::Device,
            ordering: Ordering::Relaxed,
            cache_policy: CachePolicy::Temporal,
            volatile: false,
            deferred_scope: false,
        };
        let load = EffectOp::Memory {
            space: Space::Global,
            op: MemoryOp::Load(MemSize::B32),
            semantics,
        };
        let types = vec![Ty::I64, Ty::I1, Ty::I32];
        let inputs = [ValueId(0), ValueId(1)];
        let outputs = [(ValueId(2), Ty::I32)];
        assert!(load.verify(&inputs, &outputs, &types).is_ok());
        assert!(load
            .verify(&[ValueId(2), ValueId(1)], &outputs, &types)
            .is_err());
        assert!(load
            .verify(&[ValueId(0), ValueId(2)], &outputs, &types)
            .is_err());
        assert!(load
            .verify(&inputs, &[(ValueId(2), Ty::F32)], &types)
            .is_err());
        let private = EffectOp::Memory {
            space: Space::Scratch,
            op: MemoryOp::AtomicAdd,
            semantics,
        };
        assert!(private
            .verify(&[ValueId(2), ValueId(2), ValueId(1)], &outputs, &types)
            .is_err());
        let invalid_order = EffectOp::Memory {
            space: Space::Global,
            op: MemoryOp::Load(MemSize::B32),
            semantics: MemorySemantics {
                ordering: Ordering::Release,
                ..semantics
            },
        };
        assert!(invalid_order.verify(&inputs, &outputs, &types).is_err());
        let mut f = Func {
            entry: BlockId(0),
            types: vec![Ty::I64, Ty::I1, Ty::I32, Ty::I32],
            blocks: BTreeMap::from([(
                BlockId(0),
                Block {
                    params: vec![(ValueId(0), Ty::I64), (ValueId(1), Ty::I1)],
                    insts: vec![
                        Inst::Effect {
                            provenance: 7,
                            op: load,
                            inputs: inputs.to_vec(),
                            outputs: outputs.to_vec(),
                        },
                        Inst::Effect {
                            provenance: 7,
                            op: load,
                            inputs: inputs.to_vec(),
                            outputs: vec![(ValueId(3), Ty::I32)],
                        },
                    ],
                    term: Term::Ret(vec![]),
                },
            )]),
        };
        assert!(f.clone().verify().is_err());
        if let Inst::Effect { provenance, .. } =
            &mut f.blocks.get_mut(&BlockId(0)).unwrap().insts[1]
        {
            *provenance = 8;
        }
        assert!(f.verify().is_ok());
    }
}
