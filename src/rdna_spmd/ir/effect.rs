use super::{Ty, ValueId};

pub(crate) const SCHEDULED: u64 = 1 << 62;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MemSize {
    U8,
    I8,
    U16,
    I16,
    B32,
    B64,
}
impl MemSize {
    pub fn bytes(self) -> u32 {
        match self {
            Self::U8 | Self::I8 => 1,
            Self::U16 | Self::I16 => 2,
            Self::B32 => 4,
            Self::B64 => 8,
        }
    }
    pub fn signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Scope {
    WorkItem,
    ComputeUnit,
    ShaderEngine,
    Device,
    System,
    Workgroup,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Ordering {
    Relaxed,
    Acquire,
    Release,
    Sequential,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MemorySemantics {
    pub scope: Scope,
    pub ordering: Ordering,
    pub cache_policy: CachePolicy,
    pub volatile: bool,
    pub deferred_scope: bool,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum MemoryOp {
    Load(MemSize),
    Store(MemSize),
    AtomicAdd,
    Fence,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
                MemoryOp::Load(MemSize::B64) => (vec![space.address_type(), I1], vec![I64]),
                MemoryOp::Load(_) => (vec![space.address_type(), I1], vec![I32]),
                MemoryOp::Store(MemSize::B64) => (vec![space.address_type(), I64, I1], vec![]),
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
            if matches!(op, MemoryOp::Load(_)) && matches!(semantics.ordering, Ordering::Release) {
                return Err("release load");
            }
            if matches!(op, MemoryOp::Store(_)) && matches!(semantics.ordering, Ordering::Acquire) {
                return Err("acquire store");
            }
        }
        Ok(())
    }
}
