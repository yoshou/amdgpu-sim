use crate::rdna_spmd::dialect::{Dialect, DialectRegistry};
use crate::rdna_spmd::program::Program;
use std::sync::Arc;

pub(crate) mod decode;
pub(crate) mod dialect;
pub(crate) mod lift;
mod trace;

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
    use crate::instructions::I;
    use crate::rdna_instructions::InstFormat;
    use decode::{ScalarBlock, Terminator};
    let registry = Arc::new(dialect().registry);
    let (decoded, traced) = trace::explore(entry_pc, memory, &registry, lanes)?;
    let span = decoded.span();
    let mut blocks = std::collections::BTreeMap::new();
    for &(context, start) in &traced.reached {
        let block = &decoded.blocks[&start];
        let last = block.insts.len() - 1;
        let scalar = match decode::call_op(&block.insts[last]) {
            Some(I::S_SWAPPC_B64) => {
                let InstFormat::SOP1(call) = &block.insts[last] else {
                    unreachable!()
                };
                let callee = traced.calls[&(context, start)];
                let mut body = decode::body(&block.insts[..last], &block.pcs, block.end);
                body.push(decode::link(call.sdst, block.end));
                let entry = traced.contexts[callee].entry;
                ScalarBlock {
                    pc: start,
                    body,
                    term: Terminator::Jump(trace::id(span, callee, entry)),
                }
            }
            Some(_) => {
                let frame = traced.contexts[context];
                let parent = frame.parent.expect("a return traced inside a call");
                ScalarBlock {
                    pc: start,
                    body: decode::body(&block.insts[..last], &block.pcs, block.end),
                    term: Terminator::Jump(trace::id(span, parent, frame.ret)),
                }
            }
            None => {
                let next: Vec<usize> = block.next_pcs.iter().map(|&pc| trace::id(span, context, pc)).collect();
                decode::lower_block(start, block, &next)
            }
        };
        blocks.insert(trace::id(span, context, start), scalar);
    }
    let normalized = decode::ScalarProgram {
        entry_pc: decoded.entry_pc,
        blocks,
    };
    lift_program(&normalized, registry, lanes, lds)
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
        .map(|(&id, b)| -> Result<_, String> {
            let pc = b.pc;
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
            Ok((id, lowerings))
        })
        .collect::<Result<_, String>>()?;
    reserves_temporaries(plain)?;
    let refs = lowered
        .iter()
        .map(|(&pc, b)| (pc, b.iter().collect()))
        .collect();
    Ok(lift::lift(registry, &normalized, &refs, lanes, lds))
}
