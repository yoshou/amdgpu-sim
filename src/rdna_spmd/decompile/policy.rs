use crate::rdna_spmd::analysis::facts::{Facts, Site};
use crate::rdna_spmd::ir::*;

pub fn live_values(f: &Func, facts: &Facts) -> Vec<bool> {
    let mut pending = Vec::new();
    for &id in &facts.order {
        let block = &f.blocks[&id];
        if let Term::CondBr { cond, .. } = block.term {
            pending.push(cond);
        }
        for inst in &block.insts {
            if let Inst::Effect { op, inputs, .. } = inst {
                if !matches!(
                    op,
                    EffectOp::Wave(WaveOp::Any | WaveOp::Ballot | WaveOp::ReadFirstLane)
                ) {
                    pending.extend(inputs);
                }
            }
        }
    }
    let mut live = vec![false; f.types.len()];
    while let Some(v) = pending.pop() {
        if live[v.0] {
            continue;
        }
        live[v.0] = true;
        match facts.site[v.0] {
            Site::Param { block, index } if block != f.entry => {
                pending.extend(facts.arguments(f, block, index));
            }
            Site::Inst { block, index } => {
                pending.extend((f.blocks[&block].insts[index]).operands());
            }
            _ => {}
        }
    }
    live
}
