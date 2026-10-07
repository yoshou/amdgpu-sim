use crate::processor::KernelDescriptor;
use crate::rdna_spmd::engine::EntryLayout;
use crate::rdna_spmd::program::Program;
use crate::rdna_spmd::rdna4::decode::{Cond, ScalarBlock, ScalarProgram, Terminator};
use crate::rdna_spmd::rdna4::lift;
use std::collections::BTreeMap;
use std::sync::Arc;

mod decode;
mod flush;
mod graph;
mod lower;
mod state;
mod trace;
#[cfg(test)]
mod tests;

pub fn supports(arch: &str) -> bool {
    arch == "gfx803"
}

fn terminator(inst: &decode::Inst, next: &[usize]) -> Terminator {
    use crate::instructions::I;
    let cond = match inst.op {
        I::S_ENDPGM => return Terminator::Return,
        I::S_BRANCH => return Terminator::Jump(next[0]),
        I::S_CBRANCH_SCC0 => Cond::Scc0,
        I::S_CBRANCH_SCC1 => Cond::Scc1,
        I::S_CBRANCH_VCCZ => Cond::VccZ,
        I::S_CBRANCH_VCCNZ => Cond::VccNz,
        I::S_CBRANCH_EXECZ => Cond::ExecZ,
        I::S_CBRANCH_EXECNZ => Cond::ExecNz,
        _ => return Terminator::Jump(next[0]),
    };
    Terminator::Branch {
        cond,
        taken: next[1],
        fallthrough: next[0],
    }
}

pub fn decode(descriptor: &KernelDescriptor, entry_pc: usize, memory: &[u8], lanes: u32) -> Result<(Program, EntryLayout), String> {
    if lanes != 64 {
        return Err(format!("gfx803 runs waves of 64 lanes, not {lanes}"));
    }
    if descriptor.enable_sgpr_queue_ptr {
        return Err("the kernel reads the queue, which may give it flat addresses into private memory".to_string());
    }
    let layout = EntryLayout::separate(descriptor)?;
    let mode = descriptor.float_mode as u32
        | (descriptor.enable_dx10_clamp as u32) << 8
        | (descriptor.enable_ieee_mode as u32) << 9;
    let known = layout
        .private_segment_wave_offset
        .map(|r| (r as u16, 0))
        .into_iter()
        .collect();
    let (graph, traced) = trace::explore(entry_pc, memory, state::State::entry(mode, known))?;
    let registry = Arc::new(crate::rdna_spmd::rdna4::dialect().registry);
    let target = lower::Target {
        registry: &registry,
        lanes,
        descriptor: layout.private_segment_buffer.map(|r| r as u16),
    };
    let span = graph.span();
    let mut blocks = BTreeMap::new();
    let mut lowerings = BTreeMap::new();
    for (&(context, start), states) in &traced.states {
        let block = &graph.blocks[&start];
        let (end, last) = block.insts.last().expect("a block without instructions");
        let here = trace::id(span, context, start);
        let body = if graph::terminates(last.op) {
            &block.insts[..block.insts.len() - 1]
        } else {
            &block.insts[..]
        };
        let mut scan = Vec::new();
        let mut lowered = Vec::new();
        for ((at, inst), state) in body.iter().zip(states) {
            let result = lower::lower(*at, inst, state, &target).map_err(|e| format!("{at:#x}: {e}"))?;
            scan.extend(result.scan);
            lowered.extend(result.lowerings);
        }
        let term = match (last.op, last.form) {
            (crate::instructions::I::S_SWAPPC_B64, decode::Form::Sop1 { sdst, .. }) => {
                let callee = traced.calls[&(context, start)];
                let ret = traced.contexts[callee].ret;
                lowered.extend(lower::link(sdst, ret, &target).map_err(|e| format!("{end:#x}: {e}"))?.lowerings);
                Terminator::Jump(trace::id(span, callee, traced.contexts[callee].entry))
            }
            (crate::instructions::I::S_SETPC_B64, _) => {
                let frame = traced.contexts[context];
                Terminator::Jump(trace::id(span, frame.parent.unwrap(), frame.ret))
            }
            _ => {
                let next: Vec<usize> = block.next.iter().map(|&pc| trace::id(span, context, pc)).collect();
                terminator(last, &next)
            }
        };
        blocks.insert(here, ScalarBlock { pc: here, body: scan, term });
        lowerings.insert(here, lowered);
    }
    let program = ScalarProgram { entry_pc, blocks };
    let refs = lowerings.iter().map(|(&pc, b)| (pc, b.iter().collect())).collect();
    Ok((lift::lift(registry.clone(), &program, &refs, lanes), layout))
}
