use super::codegen::ops::Lowerings;
use super::native::Value;
use super::pass::idioms::Idiom;

pub(crate) use super::codegen::ops::Implementation;
pub(crate) use super::ir::{Arguments, DialectRegistry, Effect, Registers, TargetOp};

pub(crate) struct Operation {
    pub name: &'static str,
    pub inputs: &'static [super::ir::Ty],
    pub outputs: Vec<super::ir::Ty>,
    pub lower: Implementation,
    pub effect: Effect,

    pub immediates: &'static [(usize, u64)],
}

#[derive(Default)]
pub(crate) struct Dialect {
    pub registry: DialectRegistry,
    pub lowerings: Lowerings,
    pub idioms: Vec<Box<dyn Idiom>>,
}

impl Dialect {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_dialect(&mut self, id: u32, name: &'static str) {
        self.registry.add_dialect(id, name);
    }
    pub fn set_registers(&mut self, registers: Registers) {
        self.registry.set_registers(registers);
    }
    pub fn set_lowering_state(
        &mut self,
        prepare: fn(&super::codegen::ops::Emitter, Value) -> Box<dyn std::any::Any>,
    ) {
        self.lowerings.set_state(prepare);
    }
    pub fn add_idiom(&mut self, idiom: Box<dyn Idiom>) {
        self.idioms.push(idiom);
    }
    pub fn register(
        &mut self,
        dialect: u32,
        operation: u32,
        spec: Operation,
    ) -> Result<TargetOp, &'static str> {
        let op = self.registry.register(
            dialect,
            operation,
            super::ir::Operation {
                name: spec.name,
                inputs: spec.inputs,
                outputs: spec.outputs,
                effect: spec.effect,
                immediates: spec.immediates,
            },
        )?;
        self.lowerings.add(op, spec.lower);
        Ok(op)
    }
}
