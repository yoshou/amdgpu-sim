use crate::rdna_spmd::host::Vectors;
use crate::rdna_spmd::ir::*;

pub(super) struct Costs {
    lanes: u64,
    bits: u64,
}

impl Costs {
    pub fn new(lanes: u32, host: &Vectors) -> Self {
        Self {
            lanes: lanes as u64,
            bits: host.bits as u64,
        }
    }

    pub fn lanes(&self) -> u64 {
        self.lanes
    }

    pub fn registers(&self, ty: Ty, uniform: bool) -> u64 {
        if uniform {
            return 1;
        }
        (self.lanes * ty.bits() as u64).div_ceil(self.bits).max(1)
    }

    pub fn instruction(&self, inst: &Inst, uniform: &[bool]) -> u64 {
        match inst {
            Inst::Core {
                op: Op::Const(..), ..
            } => 0,
            Inst::Core {
                ty: Ty::I1,
                op:
                    Op::Int(IntOp::And | IntOp::Or | IntOp::Xor, ..)
                    | Op::Select(..)
                    | Op::Convert(Cvt::Bitcast, ..),
                ..
            } => 0,
            Inst::Core { value, ty, .. } => self.registers(*ty, uniform[value.0]),
            Inst::Packet { .. } => 1,
            Inst::Target { outputs, .. } => outputs
                .iter()
                .map(|&(v, ty)| self.registers(ty, uniform[v.0]))
                .sum(),
            Inst::Effect {
                op: EffectOp::Memory { op, .. },
                inputs,
                ..
            } => {
                let shared = |index: usize| uniform[inputs[index].0];
                let alike = match op {
                    MemoryOp::Load(_) => shared(0),
                    MemoryOp::Store(_) | MemoryOp::AtomicAdd => shared(0) && shared(1),
                    MemoryOp::Fence => true,
                };
                if alike {
                    1
                } else {
                    self.lanes
                }
            }
            Inst::Effect { .. } => 1,
        }
    }
}
