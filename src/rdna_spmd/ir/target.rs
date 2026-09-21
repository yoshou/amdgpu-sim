use super::{Ty, ValueId};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TargetOp {
    dialect: u32,
    operation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arguments {
    Unary(ValueId),
    Binary([ValueId; 2]),
    Ternary([ValueId; 3]),
    Quaternary([ValueId; 4]),
    Thirteen([ValueId; 13]),
    Fourteen([ValueId; 14]),
    Sixteen([ValueId; 16]),
    Fifteen([ValueId; 15]),
}
impl Arguments {
    pub fn map(self, mut f: impl FnMut(ValueId) -> ValueId) -> Self {
        match self {
            Self::Unary(a) => Self::Unary(f(a)),
            Self::Binary(a) => Self::Binary(a.map(f)),
            Self::Ternary(a) => Self::Ternary(a.map(f)),
            Self::Quaternary(a) => Self::Quaternary(a.map(f)),
            Self::Thirteen(a) => Self::Thirteen(a.map(f)),
            Self::Fourteen(a) => Self::Fourteen(a.map(f)),
            Self::Sixteen(a) => Self::Sixteen(a.map(f)),
            Self::Fifteen(a) => Self::Fifteen(a.map(f)),
        }
    }
    pub fn values(&self) -> &[ValueId] {
        match self {
            Self::Unary(a) => std::slice::from_ref(a),
            Self::Binary(a) => a,
            Self::Ternary(a) => a,
            Self::Quaternary(a) => a,
            Self::Thirteen(a) => a,
            Self::Fourteen(a) => a,
            Self::Sixteen(a) => a,
            Self::Fifteen(a) => a,
        }
    }
}

pub struct Operation {
    pub name: &'static str,
    pub inputs: &'static [Ty],
    pub outputs: Vec<Ty>,
    pub effect: Effect,

    pub immediates: &'static [(usize, u64)],
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    Pure,
    ReadGlobal { every_lane: bool },
}
impl Operation {
    pub fn verify_immediates(
        &self,
        args: Arguments,
        constant: impl Fn(ValueId) -> Option<u64>,
    ) -> Result<(), &'static str> {
        for &(index, maximum) in self.immediates {
            if !args
                .values()
                .get(index)
                .and_then(|&v| constant(v))
                .is_some_and(|v| v <= maximum)
            {
                return Err("target requires an in-range constant operand");
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Registers {
    pub exec: u32,
    pub vcc: u32,
    pub null: u32,
    pub scc_slot: u32,
    pub sgprs: u32,
    pub vgprs: u32,
}

#[derive(Default)]
pub struct DialectRegistry {
    operations: BTreeMap<TargetOp, Operation>,
    dialects: BTreeMap<u32, &'static str>,
    registers: Registers,
}
impl TargetOp {
    pub fn dialect(self) -> u32 {
        self.dialect
    }
}
impl DialectRegistry {
    pub fn add_dialect(&mut self, id: u32, name: &'static str) {
        self.dialects.insert(id, name);
    }
    pub fn set_registers(&mut self, registers: Registers) {
        self.registers = registers;
    }
    pub fn registers(&self) -> Registers {
        self.registers
    }
    pub fn dialect_name(&self, dialect: u32) -> Option<&'static str> {
        self.dialects.get(&dialect).copied()
    }
    pub fn register(
        &mut self,
        dialect: u32,
        operation: u32,
        spec: Operation,
    ) -> Result<TargetOp, &'static str> {
        let op = TargetOp { dialect, operation };
        if self.operations.contains_key(&op)
            || self
                .operations
                .iter()
                .any(|(id, other)| id.dialect == dialect && other.name == spec.name)
        {
            return Err("duplicate target ID or mnemonic");
        }
        self.operations.insert(op, spec);
        Ok(op)
    }
    pub fn lookup(&self, dialect: u32, name: &str) -> Result<TargetOp, &'static str> {
        self.operations
            .iter()
            .find_map(|(id, spec)| (id.dialect == dialect && spec.name == name).then_some(*id))
            .ok_or("unregistered target operation")
    }
    pub fn operation(&self, op: TargetOp) -> Result<&Operation, &'static str> {
        self.operations
            .get(&op)
            .ok_or("unregistered target operation")
    }
    pub fn result_types(
        &self,
        op: TargetOp,
        args: Arguments,
        types: &[Ty],
    ) -> Result<&[Ty], &'static str> {
        let spec = self.operation(op)?;
        let args = args.values();
        if args.len() != spec.inputs.len()
            || args
                .iter()
                .zip(spec.inputs)
                .any(|(id, ty)| types.get(id.0) != Some(ty))
        {
            return Err("target signature mismatch");
        }
        Ok(&spec.outputs)
    }
}
