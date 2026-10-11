use crate::rdna_spmd::dialect::{Dialect, DialectRegistry};
use crate::rdna_spmd::program::Program;
use std::sync::Arc;

pub(crate) mod decode;
pub(crate) mod dialect;
pub(crate) mod lift;

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

pub fn decode(entry_pc: usize, memory: &[u8], lanes: u32, lds: bool) -> Result<Program, String> {
    let decoded = decode::program(entry_pc, memory)?;
    let normalized = decode::ScalarProgram {
        entry_pc: decoded.entry_pc,
        blocks: decoded
            .blocks
            .iter()
            .map(|(&pc, b)| (pc, decode::lower_block(pc, &b.insts, &b.next_pcs)))
            .collect(),
    };
    lift_program(&normalized, Arc::new(dialect().registry), lanes, lds)
}

pub fn reserves_temporaries(footprints: impl IntoIterator<Item = lift::BoundaryIo>) -> Result<(), String> {
    for io in footprints {
        for r in lift::dpp::TEMPORARIES {
            if io.reads.vgprs().any(|v| v == r) || io.writes.vgprs().any(|v| v == r) {
                return Err(format!("v{r} is reserved for DPP, but the program uses it"));
            }
        }
    }
    Ok(())
}

fn permlane(inst: &crate::rdna_instructions::InstFormat) -> Option<&crate::rdna_instructions::VOP3> {
    use crate::instructions::I;
    match inst {
        crate::rdna_instructions::InstFormat::VOP3(i)
            if matches!(i.op, I::V_PERMLANE16_B32 | I::V_PERMLANEX16_B32) =>
        {
            Some(i)
        }
        _ => None,
    }
}

fn lift_program(
    source: &decode::ScalarProgram,
    registry: Arc<DialectRegistry>,
    lanes: u32,
    lds: bool,
) -> Result<Program, String> {
    use crate::rdna_spmd::ir::EffectOp;
    use crate::{
        instructions::I,
        rdna_instructions::{InstFormat, SourceOperand},
    };
    use lift::{Lowering, YieldAction};
    let mut normalized = source.clone();
    let mut plain = Vec::new();
    let lowered: std::collections::BTreeMap<_, Vec<_>> = normalized
        .blocks
        .iter_mut()
        .map(|(&pc, b)| -> Result<_, String> {
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
                } else if let InstFormat::DPP(d) = inst {
                    let expanded = lift::dpp::lowerings(d, &registry, lanes).map_err(|e| format!("{pc:#x}: {e}"))?;
                    for lowering in expanded {
                        body.push(inst.clone());
                        lowerings.push(lowering);
                    }
                } else if let Some(i) = permlane(inst) {
                    let expanded = lift::dpp::permlane_lowerings(i, &registry, lanes).map_err(|e| format!("{pc:#x}: {e}"))?;
                    for lowering in expanded {
                        body.push(inst.clone());
                        lowerings.push(lowering);
                    }
                } else {
                    body.push(inst.clone());
                    let lowering = lift::instruction_with_registry(inst, &registry, lanes)
                        .map_err(|e| format!("{pc:#x}: {e}"))?;
                    plain.push(lift::footprint(&lowering, lanes));
                    lowerings.push(lowering);
                }
            }
            b.body = body;
            Ok((pc, lowerings))
        })
        .collect::<Result<_, String>>()?;
    reserves_temporaries(plain)?;
    let refs = lowered
        .iter()
        .map(|(&pc, b)| (pc, b.iter().collect()))
        .collect();
    Ok(lift::lift(registry, &normalized, &refs, lanes, lds))
}
