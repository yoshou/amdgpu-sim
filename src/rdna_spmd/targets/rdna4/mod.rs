use std::sync::Arc;
use crate::rdna_spmd::dialect::DialectRegistry;
use crate::rdna_spmd::program::{CompilationInput, LiftedFunction, Program};
use crate::rdna_spmd::target::Target;

pub(crate) mod decode;
pub(crate) mod dialect;
pub(crate) mod lift;

pub(crate) fn registry() -> DialectRegistry {
    let mut registry = DialectRegistry::new();
    registry.add_dialect(dialect::ID, "rdna4");
    registry.set_registers(dialect::REGISTERS);
    registry.set_lowering_state(dialect::bvh::lowering_state);
    dialect::register(&mut registry).expect("RDNA4 target registration conflict");
    let idioms = dialect::idioms::SqrtIdioms::new(&registry);
    registry.add_idiom(Box::new(idioms));
    let divisions = dialect::idioms::DivisionIdioms::new(&registry);
    registry.add_idiom(Box::new(divisions));
    registry
}

pub(crate) struct Rdna4 { registry: Arc<DialectRegistry> }

impl Rdna4 {
    pub fn new() -> Self { Self { registry: Arc::new(registry()) } }
}

impl Target for Rdna4 {
    fn supports(&self, arch: &str) -> bool { arch.starts_with("gfx12") }
    fn decode(&self, entry_pc: usize, memory: &[u8]) -> Result<LiftedFunction, String> {
        let decoded = decode::program(entry_pc, memory)?;
        let normalized = decode::ScalarProgram { entry_pc: decoded.entry_pc, blocks: decoded.blocks.iter().map(|(&pc, b)|
            (pc, decode::lower_block(pc, &b.insts, &b.next_pcs))).collect() };
        Ok(lift_program(&normalized, self.registry.clone()))
    }
}

pub(crate) fn lift_program(source: &decode::ScalarProgram, registry: Arc<DialectRegistry>) -> LiftedFunction {
    use lift::{Lowering, wave::YieldAction};
    use crate::rdna_spmd::ir::EffectOp;
    use crate::{instructions::I, rdna_instructions::{InstFormat, SourceOperand}};
    let mut normalized = source.clone();
    let lowered: std::collections::BTreeMap<_, Vec<_>> = normalized.blocks.iter_mut().map(|(&pc, b)| {
        let mut body = Vec::new();
        let mut lowerings = Vec::new();
        for inst in &b.body {
            if matches!(inst, InstFormat::SOPP(i) if matches!(i.op, I::S_BARRIER)) {
                for op in [EffectOp::BarrierSignal { is_first: false }, EffectOp::BarrierWait] {
                    body.push(inst.clone());
                    lowerings.push(Lowering::Wave(YieldAction::new(op,
                        vec![lift::wave::Operand::Source(SourceOperand::LiteralConstant(u32::MAX))], vec![])));
                }
            } else {
                body.push(inst.clone());
                lowerings.push(lift::instruction_with_registry(inst, &registry));
            }
        }
        b.body = body;
        (pc, lowerings)
    }).collect();
    let refs = lowered.iter().map(|(&pc, b)| (pc, b.iter().collect())).collect();
    lift::function::lift(registry, &normalized, &refs)
}

impl CompilationInput for decode::ScalarProgram {
    fn to_ssa(&self) -> Program { Program { function: lift_program(self, Arc::new(registry())) } }
}
