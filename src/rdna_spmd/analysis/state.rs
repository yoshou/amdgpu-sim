//! State observations expressed as SSA uses. Native storage slots only bound
//! the existing conservative predication policy; data flow follows SSA edges.
use super::*;

#[derive(Clone)]
pub(in crate::rdna_spmd) enum Activation {
    Unknown,
    Copy(ValueId),
    Constant(bool),
    Or { exec: Option<ValueId>, saved: Option<ValueId> },
}
#[derive(Clone)]
pub(in crate::rdna_spmd) struct ExecPolicy {
    pub before: ValueId,
    pub after: ValueId,
    pub update: Activation,
    pub saved: Option<ValueId>,
    pub killed: Vec<ValueId>,
    pub constant: Option<u64>,
    pub preserves_nonempty: bool,
    pub writes: bool,
    pub resets_nonempty: bool,
}
#[derive(Clone)]
pub(in crate::rdna_spmd) enum MaskEvent {
    Save(u32), Copy(u32), Logic { restore: bool, sources: Vec<u32> }, Compare,
}
#[derive(Clone, Default)]
pub(in crate::rdna_spmd) struct MaskPolicy {
    pub closure: Vec<u32>,
    pub seed: Vec<u32>,
    pub definitions: Vec<(ValueId, Vec<ValueId>, bool)>,
    pub reads: Vec<(u32, ValueId)>,
    pub event: Option<MaskEvent>,
    pub cross_lane: bool,
}
#[derive(Clone)]
pub(in crate::rdna_spmd) struct Site {
    pub rewrite: super::rewrite::Observation,
    pub math_reads: Vec<(u32, ValueId)>,
    pub sqrt: crate::rdna_spmd::sqrt_idiom::Policy,
    pub masks: MaskPolicy,
    pub f64_definitions: Vec<(ValueId, ValueId)>,
    pub u64_definitions: Vec<(ValueId, ValueId)>,
    pub f64_uses: Vec<(u32, ValueId, ValueId)>,
    pub exec: ExecPolicy,
    pub reads: Vec<ValueId>,
    pub varying_inputs: Vec<ValueId>,
    pub intrinsically_varying: bool,
    pub address: Vec<ValueId>,
    pub frame: Option<(u32, ValueId, ValueId, u32)>,
    pub writes: Vec<(u32, ValueId)>,
    pub reactivation: Vec<(u32, ValueId)>,
}
#[derive(Clone, Default)]
pub(in crate::rdna_spmd) struct StateGraph {
    pub vector_parameters: Vec<(u32, usize)>,
    pub scalar_parameters: Vec<(u32, usize)>,
    pub scalar_outgoing: BTreeMap<usize, BTreeMap<u32, ValueId>>,
    pub conditions: BTreeMap<usize, crate::rdna_spmd::ir::Cond>,
    pub exec_edges: BTreeMap<(usize, usize), bool>,
    pub sites: BTreeMap<usize, Vec<Site>>,
    pub wave_reads: Vec<ValueId>,
    pub outgoing: BTreeMap<usize, BTreeMap<u32, ValueId>>,
    pub yielding: std::collections::BTreeSet<usize>,
    pub barriers: std::collections::BTreeSet<usize>,
    pub yield_exec_writes: std::collections::BTreeSet<usize>,
    pub resume_observations: Vec<(ValueId, ValueId)>,
}

/// Preserve the existing rule: every source use is observed, and every native
/// slot live at a merge/reactivation remains predicated throughout the function.
/// This intentionally does not introduce dead-instruction or narrower-use proofs.
pub(in crate::rdna_spmd) fn predication(
    f: &Func,
    graph: &StateGraph,
    exit_slots: &[u32],
) -> BTreeMap<usize, Vec<bool>> {
    let mut deps = vec![Vec::new(); f.types.len()];
    let mut predecessors = BTreeMap::<BlockId, usize>::new();
    for block in f.blocks.values() {
        for edge in block.term.edges() {
            *predecessors.entry(edge.dst).or_default() += 1;
            for (&arg, &(param, _)) in edge.args.iter().zip(&f.blocks[&edge.dst].params) {
                deps[param.0].push(arg);
            }
        }
    }
    for &(output, previous) in &graph.resume_observations { deps[output.0].push(previous); }
    let mut last_use = vec![None; f.types.len()];
    for sites in graph.sites.values() {
        for (index, site) in sites.iter().enumerate() {
            for value in &site.reads { last_use[value.0] = Some(index); }
        }
    }
    for value in &graph.wave_reads { last_use[value.0] = Some(usize::MAX); }
    let mut pending: Vec<_> = graph.sites.values().flatten().flat_map(|s| &s.reads).copied()
        .chain(graph.wave_reads.iter().copied()).collect();
    // The outgoing definitions are recorded as the destination parameters on
    // return sites, so the exit roots use the same native slot map as merges.
    let mut observed = [false; 256];
    for &slot in exit_slots { observed[slot as usize] = true; }
    let mut live = vec![false; f.types.len()];
    while let Some(value) = pending.pop() {
        if !live[value.0] {
            live[value.0] = true;
            pending.extend(&deps[value.0]);
        }
    }
    for (&pc, block) in &f.blocks {
        if predecessors.get(&pc).copied().unwrap_or(0) >= 2 {
            for &(slot, index) in &graph.vector_parameters {
                if live[block.params[index].0.0] { observed[slot as usize] = true; }
            }
        }
    }
    let mut edge_live = vec![false; f.types.len()];
    for block in f.blocks.values() {
        for edge in block.term.edges() {
            for (&arg, &(param, _)) in edge.args.iter().zip(&f.blocks[&edge.dst].params) {
                if live[param.0] { edge_live[arg.0] = true; }
            }
        }
    }
    for &(output, previous) in &graph.resume_observations {
        if edge_live[output.0] { edge_live[previous.0] = true; }
    }
    for sites in graph.sites.values() {
        for (index, site) in sites.iter().enumerate() {
            for &(slot, value) in &site.reactivation {
                if last_use[value.0].is_some_and(|use_index| use_index > index) || edge_live[value.0] {
                    observed[slot as usize] = true;
                }
            }
        }
    }
    graph.sites.iter().map(|(&pc, sites)| (pc, sites.iter().map(|site|
        !site.writes.is_empty() && site.writes.iter().all(|&(slot, _)| !observed[slot as usize])
    ).collect())).collect()
}

/// The current packet policy treats each instruction's destinations as one
/// cohort. It proves no additional uniformity for partial/independent outputs.
pub(in crate::rdna_spmd) fn varying(
    f: &Func, graph: &StateGraph, boundary_writes: &BTreeMap<usize, Vec<u32>>,
) -> Vec<bool> {
    let mut varying = vec![false; f.types.len()];
    if let Some(&(_, index)) = graph.vector_parameters.iter().find(|p| p.0 == 0) {
        varying[f.blocks[&f.entry].params[index].0.0] = true;
    }
    loop {
        let mut changed = false;
        for sites in graph.sites.values() {
            for site in sites {
                let fact = site.intrinsically_varying || site.varying_inputs.iter().any(|v| varying[v.0]);
                for &(slot, value) in &site.writes {
                    if (fact || slot == 0) && !varying[value.0] { varying[value.0] = true; changed = true; }
                }
            }
        }
        for (&pc, block) in &f.blocks {
            for edge in block.term.edges() {
                for &(slot, index) in &graph.vector_parameters {
                    let value = f.blocks[&edge.dst].params[index].0;
                    let source = graph.outgoing[&pc.0][&slot];
                    let host_write = graph.yielding.contains(&pc.0)
                        && boundary_writes.get(&edge.dst.0).is_some_and(|writes| writes.contains(&slot));
                    if (varying[source.0] || host_write) && !varying[value.0] { varying[value.0] = true; changed = true; }
                }
            }
        }
        if !changed { return varying; }
    }
}

pub(in crate::rdna_spmd) fn frames(
    f: &Func, graph: &StateGraph, boundary_writes: &BTreeMap<usize, Vec<u32>>,
) -> BTreeMap<(ValueId, ValueId), u32> {
    let parameters: BTreeMap<_, _> = graph.vector_parameters.iter().copied().collect();
    let mut top = BTreeMap::new();
    let mut facts = BTreeMap::new();
    for site in graph.sites.values().flatten() {
        if let Some((slot, lo, hi, stride)) = site.frame { top.insert(slot, stride); facts.insert((lo, hi), stride); }
    }
    for (&pc, block) in &f.blocks {
        if pc != f.entry {
            for (&slot, &stride) in &top {
                facts.insert((block.params[parameters[&slot]].0, block.params[parameters[&(slot+1)]].0), stride);
            }
        }
    }
    loop {
        let mut incoming: BTreeMap<(ValueId, ValueId), Option<u32>> = BTreeMap::new();
        for (&pc, block) in &f.blocks {
            for edge in block.term.edges() {
                let dest = &f.blocks[&edge.dst];
                for &slot in top.keys() {
                    let pair = (dest.params[parameters[&slot]].0, dest.params[parameters[&(slot+1)]].0);
                    let previous = (graph.outgoing[&pc.0][&slot], graph.outgoing[&pc.0][&(slot+1)]);
                    let killed = graph.yielding.contains(&pc.0) && boundary_writes.get(&edge.dst.0)
                        .is_some_and(|writes| writes.contains(&slot) || writes.contains(&(slot+1)));
                    let fact = if killed || edge.dst == f.entry { None } else { facts.get(&previous).copied() };
                    incoming.entry(pair).and_modify(|v| { if *v != fact { *v = None; } }).or_insert(fact);
                }
            }
        }
        let mut changed = false;
        for (pair, fact) in incoming {
            if facts.get(&pair).copied() != fact {
                if let Some(stride) = fact { facts.insert(pair, stride); } else { facts.remove(&pair); }
                changed = true;
            }
        }
        if !changed { return facts; }
    }
}

/// Forward must-facts on explicit EXEC and saved-mask definitions. The scalar
/// policy retains its original exclusions for inactive edges and backedges.
pub(in crate::rdna_spmd) fn active(f: &Func, graph: &StateGraph, packed: bool) -> BTreeMap<usize, Vec<bool>> {
    let mut active = vec![true; f.types.len()];
    let mut saved = vec![true; f.types.len()];
    let exec_index = graph.scalar_parameters.iter().find(|p| p.0 == 126).unwrap().1;
    loop {
        let previous_active = active.clone();
        let previous_saved = saved.clone();
        for sites in graph.sites.values() {
            for site in sites {
                let p = &site.exec;
                for v in &p.killed { saved[v.0] = false; }
                if let Some(v) = p.saved { saved[v.0] = active[p.before.0]; }
                active[p.after.0] = match p.update {
                    Activation::Unknown => false,
                    Activation::Copy(v) => active[v.0],
                    Activation::Constant(v) => v,
                    Activation::Or { exec, saved: mask } => exec.is_some_and(|v| active[v.0]) || mask.is_some_and(|v| saved[v.0]),
                };
            }
        }
        let mut incoming = BTreeMap::<ValueId, bool>::new();
        let mut mask_incoming = BTreeMap::<ValueId, bool>::new();
        incoming.insert(f.blocks[&f.entry].params[exec_index].0, true);
        for &(_, index) in &graph.scalar_parameters { mask_incoming.insert(f.blocks[&f.entry].params[index].0, false); }
        for (&pc, block) in &f.blocks {
            for edge in block.term.edges() {
                let pinned = graph.exec_edges.get(&(pc.0, edge.dst.0)).copied();
                if !packed && (edge.dst <= pc || pinned == Some(false)) { continue; }
                let out = &graph.scalar_outgoing[&pc.0];
                let exec = pinned.unwrap_or(active[out[&126].0]);
                incoming.entry(f.blocks[&edge.dst].params[exec_index].0).and_modify(|v| *v &= exec).or_insert(exec);
                for &(slot, index) in &graph.scalar_parameters {
                    let fact = saved[out[&slot].0];
                    mask_incoming.entry(f.blocks[&edge.dst].params[index].0).and_modify(|v| *v &= fact).or_insert(fact);
                }
            }
        }
        for (value, fact) in incoming { active[value.0] = fact; }
        for (value, fact) in mask_incoming { saved[value.0] = fact; }
        if active == previous_active && saved == previous_saved { break; }
    }
    graph.sites.iter().map(|(&pc, sites)| (pc, sites.iter().map(|s| active[s.exec.before.0]).collect())).collect()
}

pub(in crate::rdna_spmd) fn nonempty(
    f: &Func, graph: &StateGraph, width: u32, initial: bool,
) -> BTreeMap<usize, Vec<bool>> {
    let exec_index = graph.scalar_parameters.iter().find(|p| p.0 == 126).unwrap().1;
    let mut facts = vec![true; f.types.len()];
    loop {
        let previous = facts.clone();
        for sites in graph.sites.values() {
            for site in sites {
                let p = &site.exec;
                facts[p.after.0] = if !p.resets_nonempty { facts[p.before.0] }
                    else if let Some(bits) = p.constant { initial && bits & ((1u64 << width)-1) != 0 }
                    else { p.preserves_nonempty && facts[p.before.0] };
            }
        }
        let mut incoming = BTreeMap::from([(f.blocks[&f.entry].params[exec_index].0, initial)]);
        for (&pc, block) in &f.blocks {
            for edge in block.term.edges() {
                let value = f.blocks[&edge.dst].params[exec_index].0;
                let fact = graph.exec_edges.get(&(pc.0, edge.dst.0)).copied().unwrap_or_else(||
                    !graph.barriers.contains(&pc.0) && !graph.yield_exec_writes.contains(&pc.0)
                    && facts[graph.scalar_outgoing[&pc.0][&126].0]);
                incoming.entry(value).and_modify(|v| *v &= fact).or_insert(fact);
            }
        }
        for (value, fact) in incoming { facts[value.0] = fact; }
        if facts == previous { break; }
    }
    graph.sites.iter().map(|(&pc, sites)| (pc, sites.iter().map(|s| facts[s.exec.before.0]).collect())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rdna_spmd::{ir::{ScalarProgram, ScalarBlock, Terminator}, lift};
    use crate::rdna_instructions::{InstFormat, SourceOperand, VOP3SD, VSCRATCH};
    use crate::instructions::I;

    fn lift(program: &ScalarProgram) -> lift::function::LiftedFunction {
        let instructions: BTreeMap<_, Vec<_>> = program.blocks.iter().map(|(&pc, block)|
            (pc, block.body.iter().map(lift::instruction).collect())).collect();
        let refs = instructions.iter().map(|(&pc, body)| (pc, body.iter().collect())).collect();
        lift::function::Function::lift(std::sync::Arc::new(crate::rdna_spmd::dialect::DialectRegistry::rdna4()), program, &refs)
    }
    fn parameter(f: &lift::function::LiftedFunction, pc: usize, slot: u32) -> ValueId {
        let index = f.state.vector_parameters.iter().find(|p| p.0 == slot).unwrap().1;
        f.ir.blocks[&BlockId(pc)].params[index].0
    }
    #[test]
    fn scratch_load_result_is_intrinsically_divergent() {
        let inst = InstFormat::VSCRATCH(VSCRATCH { op: I::SCRATCH_LOAD_B32,
            vaddr: 0, vsrc: 0, vdst: 7, scope: 0, th: 0, ioffset: 0, saddr: 124, sve: 0 });
        let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock {
            pc: 0, body: vec![inst], term: Terminator::Return,
        })]) };
        let f = lift(&program);
        let facts = varying(&f.ir, &f.state, &BTreeMap::new());
        assert!(facts[f.state.sites[&0][0].writes[0].1.0]);
    }
    #[test]
    fn cross_lane_write_is_divergent_on_resume() {
        use crate::rdna_spmd::{lift::wave::{YieldAction, Operand, Destination}, ir::typed::effect::{EffectOp, WaveOp}};
        let action = YieldAction::new(EffectOp::Wave(WaveOp::ReadLane),
            vec![Operand::Source(SourceOperand::IntegerConstant(0)), Operand::Source(SourceOperand::IntegerConstant(0))],
            vec![Destination::Vgpr(23)]);
        let program = ScalarProgram { entry_pc: 1, blocks: BTreeMap::from([
            (1, ScalarBlock { pc: 1, body: vec![], term: Terminator::Yield { resume: 2, action: Box::new(action) } }),
            (2, ScalarBlock { pc: 2, body: vec![], term: Terminator::Return }),
        ]) };
        let f = lift(&program);
        let facts = varying(&f.ir, &f.state, &BTreeMap::from([(2, vec![23])]));
        assert!(facts[parameter(&f, 2, 23).0]);
    }
    #[test]
    fn cross_lane_write_kills_affine_frame_fact() {
        let inst = InstFormat::VOP3SD(VOP3SD { vdst: 10, sdst: 106, cm: 0,
            op: I::V_MAD_CO_U64_U32, src0: SourceOperand::VectorRegister(0),
            src1: SourceOperand::IntegerConstant(16), src2: SourceOperand::ScalarRegister(0), omod: 0, neg: 0 });
        let program = ScalarProgram { entry_pc: 1, blocks: BTreeMap::from([
            (1, ScalarBlock { pc: 1, body: vec![inst], term: Terminator::Barrier { resume: 2 } }),
            (2, ScalarBlock { pc: 2, body: vec![], term: Terminator::Return }),
        ]) };
        let f = lift(&program);
        let pair = (parameter(&f, 2, 10), parameter(&f, 2, 11));
        assert_eq!(frames(&f.ir, &f.state, &BTreeMap::new()).get(&pair), Some(&16));
        assert!(!frames(&f.ir, &f.state, &BTreeMap::from([(2, vec![10])])).contains_key(&pair));
    }
}

/// Availability of a native 64-bit view is attached to both SSA words. A new
/// half definition cannot inherit that view. Block arguments meet all incoming
/// pairs; a scheduler yield discards the native view while preserving bits.
pub(in crate::rdna_spmd) fn native_pairs(
    f: &Func, graph: &StateGraph, scalar: bool,
) -> BTreeMap<usize, [u128; 2]> {
    let parameters: BTreeMap<_, _> = if scalar { &graph.scalar_parameters } else { &graph.vector_parameters }
        .iter().copied().collect();
    let outgoing = if scalar { &graph.scalar_outgoing } else { &graph.outgoing };
    let slots: Vec<_> = parameters.keys().copied().filter(|r| parameters.contains_key(&(r+1))).collect();
    let pair = |block: &crate::rdna_spmd::ir::typed::cfg::Block, slot: u32|
        (block.params[parameters[&slot]].0, block.params[parameters[&(slot+1)]].0);
    let definitions: std::collections::BTreeSet<_> = graph.sites.values().flatten().flat_map(|s|
        if scalar { &s.u64_definitions } else { &s.f64_definitions }).copied().collect();
    let mut facts = definitions.clone();
    for (&pc, block) in &f.blocks {
        if pc != f.entry { for &slot in &slots { facts.insert(pair(block, slot)); } }
    }
    loop {
        let mut incoming = BTreeMap::new();
        for (&pc, block) in &f.blocks {
            for edge in block.term.edges() {
                for &slot in &slots {
                    let source = (outgoing[&pc.0][&slot], outgoing[&pc.0][&(slot+1)]);
                    let available = edge.dst != f.entry && !graph.yielding.contains(&pc.0) && facts.contains(&source);
                    incoming.entry(pair(&f.blocks[&edge.dst], slot))
                        .and_modify(|v| *v &= available).or_insert(available);
                }
            }
        }
        let mut changed = false;
        for (value, available) in incoming {
            if !available { changed |= facts.remove(&value); }
        }
        if !changed { break; }
    }
    f.blocks.iter().map(|(&pc, block)| {
        let mut set = [0u128; 2];
        for &slot in &slots {
            if facts.contains(&pair(block, slot)) { set[(slot / 128) as usize] |= 1 << (slot % 128); }
        }
        (pc.0, set)
    }).collect()
}

/// Keep the original function-wide choice of disjoint f64 cells. Each request
/// now names a typed SSA word pair; native slots only constrain overlap.
pub(in crate::rdna_spmd) fn f64_cells(graph: &StateGraph) -> [u128; 2] {
    let requested: std::collections::BTreeSet<_> = graph.sites.values().flatten()
        .flat_map(|s| &s.f64_uses).map(|&(slot, _, _)| slot).collect();
    let mut set = [0u128; 2];
    for &slot in &requested {
        if slot < 255 && !(slot > 0 && requested.contains(&(slot-1))) && !requested.contains(&(slot+1)) {
            set[(slot / 128) as usize] |= 1 << (slot % 128);
        }
    }
    set
}
