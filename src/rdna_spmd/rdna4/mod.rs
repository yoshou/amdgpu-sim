use crate::rdna_spmd::dialect::{Dialect, DialectRegistry};
use crate::rdna_spmd::program::Program;
use std::sync::Arc;

mod decode;
mod dialect;
mod lift;

pub fn dialect() -> Dialect {
    let mut rdna4 = Dialect::new();
    rdna4.add_dialect(dialect::ID, "rdna4");
    rdna4.set_registers(dialect::REGISTERS);
    rdna4.set_lowering_state(dialect::lowering_state);
    dialect::register(&mut rdna4).expect("RDNA4 target registration conflict");
    let idioms = dialect::SqrtIdioms::new(&rdna4.registry);
    rdna4.add_idiom(Box::new(idioms));
    let divisions = dialect::DivisionIdioms::new(&rdna4.registry);
    rdna4.add_idiom(Box::new(divisions));
    rdna4
}

pub fn supports(arch: &str) -> bool {
    arch.starts_with("gfx12")
}

pub fn decode(entry_pc: usize, memory: &[u8]) -> Result<Program, String> {
    let decoded = decode::program(entry_pc, memory)?;
    let normalized = decode::ScalarProgram {
        entry_pc: decoded.entry_pc,
        blocks: decoded
            .blocks
            .iter()
            .map(|(&pc, b)| (pc, decode::lower_block(pc, &b.insts, &b.next_pcs)))
            .collect(),
    };
    Ok(lift_program(&normalized, Arc::new(dialect().registry)))
}

fn lift_program(
    source: &decode::ScalarProgram,
    registry: Arc<DialectRegistry>,
) -> Program {
    use crate::rdna_spmd::ir::EffectOp;
    use crate::{
        instructions::I,
        rdna_instructions::{InstFormat, SourceOperand},
    };
    use lift::{Lowering, YieldAction};
    let mut normalized = source.clone();
    let lowered: std::collections::BTreeMap<_, Vec<_>> = normalized
        .blocks
        .iter_mut()
        .map(|(&pc, b)| {
            let mut body = Vec::new();
            let mut lowerings = Vec::new();
            for inst in &b.body {
                if matches!(inst, InstFormat::SOPP(i) if matches!(i.op, I::S_BARRIER)) {
                    for op in [
                        EffectOp::BarrierSignal { is_first: false },
                        EffectOp::BarrierWait,
                    ] {
                        body.push(inst.clone());
                        lowerings.push(Lowering::Wave(YieldAction::new(
                            op,
                            vec![lift::Operand::Source(SourceOperand::LiteralConstant(
                                u32::MAX,
                            ))],
                            vec![],
                        )));
                    }
                } else {
                    body.push(inst.clone());
                    lowerings.push(lift::instruction_with_registry(inst, &registry));
                }
            }
            b.body = body;
            (pc, lowerings)
        })
        .collect();
    let refs = lowered
        .iter()
        .map(|(&pc, b)| (pc, b.iter().collect()))
        .collect();
    lift::lift(registry, &normalized, &refs)
}
