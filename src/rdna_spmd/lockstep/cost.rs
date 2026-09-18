//! What skipping a span saves and what the query that skips it costs.
//!
//! Both are counted in the one unit the host does work in: an operation on a
//! register. A value the lanes share takes one register; a value per lane takes
//! as many vector registers as its lanes need bits. A memory access moves one
//! element for the packet when every lane names the same word and one element
//! per lane otherwise. Counts of a span are lower bounds on the work it does:
//! a loop counts one trip, a bit a mask can absorb counts nothing, and a target
//! operation counts only the results it must produce.

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

    /// Registers a value occupies.
    pub fn registers(&self, ty: Ty, uniform: bool) -> u64 {
        if uniform {
            return 1;
        }
        (self.lanes * ty.bits() as u64).div_ceil(self.bits).max(1)
    }

    /// Operations the lanes perform for one instruction of a lane program.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn avx512() -> Vectors {
        Vectors {
            bits: 512,
            native_masks: true,
        }
    }

    #[test]
    fn a_value_takes_the_registers_its_lanes_need() {
        let sixteen = Costs::new(16, &avx512());
        assert_eq!(
            sixteen.registers(Ty::F64, false),
            2,
            "16 doubles need 1024 bits"
        );
        assert_eq!(sixteen.registers(Ty::I32, false), 1);
        assert_eq!(sixteen.registers(Ty::I1, false), 1);
        assert_eq!(
            sixteen.registers(Ty::F64, true),
            1,
            "a shared value is one scalar"
        );
        let eight = Costs::new(8, &avx512());
        assert_eq!(eight.registers(Ty::F64, false), 1);
    }

    #[test]
    fn an_access_moves_an_element_per_lane_unless_every_lane_names_the_same_word() {
        let costs = Costs::new(16, &avx512());
        let semantics = MemorySemantics {
            scope: Scope::ComputeUnit,
            ordering: Ordering::Relaxed,
            cache_policy: CachePolicy::Temporal,
            volatile: false,
            deferred_scope: false,
        };
        let load = |address: usize| Inst::Effect {
            provenance: 0,
            op: EffectOp::Memory {
                space: Space::Global,
                op: MemoryOp::Load(MemSize::B32),
                semantics,
            },
            inputs: vec![ValueId(address), ValueId(2)],
            outputs: vec![(ValueId(3), Ty::I32)],
        };
        let uniform = [true, false, false, false];
        assert_eq!(costs.instruction(&load(0), &uniform), 1);
        assert_eq!(costs.instruction(&load(1), &uniform), 16);
    }
}
