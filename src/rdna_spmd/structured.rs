//! Conservative natural-loop and lane-mask analysis for vector code generation.
//!
//! Natural loops, carried values and saved-mask scopes are computed from the
//! typed CFG and SSA definitions. Native mask-cell eligibility retains the
//! existing conservative slot closure and single leaf-loop selection.

use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
use crate::instructions::I;
#[cfg(test)]
use crate::rdna_instructions::{InstFormat, SourceOperand};

#[cfg(test)]
use super::ir::{ScalarProgram, Terminator};
use super::analysis::state::{StateGraph, Site, MaskEvent};
use super::ir::typed::cfg::{Func, Block};

/// A control-flow feature that prevents the structured-loop optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredReject {
    Barrier { pc: usize },
    CrossLane { pc: usize, inst: usize },
    /// A cyclic SCC with more than one entry cannot be represented by a
    /// single-entry natural loop without changing the RDNA control semantics.
    IrreducibleScc { entries: Vec<usize>, nodes: Vec<usize> },
    /// A natural loop itself has an entry other than its header.
    MultiEntryLoop { header: usize, entries: Vec<usize> },
}

/// VGPR state that a region exit transfers to its successor.  The set is a
/// conservative liveness over-approximation: carrying an extra register only
/// reduces future optimization opportunity, while omitting one would be
/// incorrect.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredExitState {
    pub from: usize,
    pub to: usize,
    pub live_vgprs: Vec<u32>,
}

/// Branch predicates present inside one loop. EXEC/VCC predicates represent
/// per-lane conditions, while SCC predicates remain uniform scalar control.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredControl {
    pub branch_conditions: Vec<super::ir::Cond>,
    pub conditional_blocks: Vec<usize>,
}

/// One decoded instruction that changes or saves EXEC. Operations remain in
/// reverse-postorder traversal order so save-and-restore pairs can be matched
/// within the loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StructuredMaskOpKind {
    SaveExec,
    CopyExec,
    RestoreExec,
    ExecLogic,
    CmpxNarrowExec,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredMaskOp {
    pub pc: usize,
    pub instruction: usize,
    pub kind: StructuredMaskOpKind,
    /// Saved-mask SGPR for Save/Copy/Restore operations, where applicable.
    pub saved_sgpr: Option<u32>,
}

/// A region-local `saveexec` scope with one proven save/copy and one restore.
/// The scope can wrap across the loop latch; `save` and `restore` therefore
/// identify the semantic boundary rather than a source-address interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredMaskScope {
    pub saved_sgpr: u32,
    pub save: StructuredMaskOp,
    pub restore: StructuredMaskOp,
}

/// Decoded location where a value derived from a lane mask is read as ordinary
/// scalar data. Such a read prevents the optimization from replacing that SGPR
/// solely with a vector of per-lane booleans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredScalarMaskAlias {
    pub sgpr: u32,
    pub pc: usize,
    pub instruction: usize,
}

/// Path-insensitive summary of EXEC saves and restores. `unrestored_saved` is
/// diagnostic only because an enclosing loop may perform the restore. Only
/// entries in `local_scopes` have a save and restore owned by this loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredMaskStack {
    pub operations: Vec<StructuredMaskOp>,
    pub saved_sgprs: Vec<u32>,
    /// EXEC, VCC, and SGPRs that may contain values copied or computed from
    /// those lane masks. Eligible loops can represent these as `<W x i1>`.
    pub mask_sgprs: Vec<u32>,
    /// Members of `mask_sgprs` whose reaching mask value is read by an
    /// operation other than the recognized mask-copy and mask-logic patterns.
    /// Replacing such an SGPR solely with `<W x i1>` would change the scalar
    /// read, so these uses must remain on the ordinary representation. This
    /// decoded-instruction analysis intentionally reports uncertain cases.
    pub scalar_alias_sgprs: Vec<u32>,
    pub scalar_alias_sites: Vec<StructuredScalarMaskAlias>,
    pub unrestored_saved: Vec<u32>,
    /// Saved masks that may still be live at a latch or an outward edge.  They
    /// cannot become a lexical predicate scope in this region.
    pub boundary_live_saved: Vec<u32>,
    /// Save/restore pairs with one local owner and no boundary escape.
    pub local_scopes: Vec<StructuredMaskScope>,
}

/// One natural loop and the register state that crosses its back-edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredLoop {
    pub header: usize,
    /// Header of the immediately enclosing natural loop, if any.  The root
    /// loops have no parent.
    pub parent: Option<usize>,
    /// Headers of immediately nested loops, in deterministic address order.
    pub children: Vec<usize>,
    pub latches: Vec<usize>,
    pub body: Vec<usize>,
    /// Blocks owned by this loop after its immediately nested loop bodies are
    /// removed. This identifies which blocks belong directly to this loop.
    pub exclusive_body: Vec<usize>,
    /// Reverse postorder of the loop sub-CFG from `header`.  This is the
    /// deterministic decoded control-flow order used by mask save/restore
    /// analysis; it is deliberately not address order.
    pub rpo_body: Vec<usize>,
    pub exits: Vec<(usize, usize)>,
    /// VGPR values live when control enters the loop header.
    pub entry_vgprs: Vec<u32>,
    /// VGPR values live on each edge leaving the loop.
    pub exit_state: Vec<StructuredExitState>,
    /// ISA-derived control predicates that must be represented by the region.
    pub control: StructuredControl,
    /// EXEC save/restore operations recovered from decoded instructions.
    pub mask_stack: StructuredMaskStack,
    pub carried: Vec<u32>,
    /// Carried values live at every block in the loop body.  They must remain
    /// outer-loop state even after re-nesting.
    pub core_carried: Vec<u32>,
    /// Carried values with a liveness gap in the loop body.  These are the only
    /// values whose storage may be limited to a nested part of the loop.
    pub gapped_carried: Vec<u32>,
}

/// Read-only loop-analysis result. `supported()` means that no barrier,
/// cross-lane instruction, or irreducible control flow was found. Individual
/// optimizations apply additional checks before using a loop from this plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredPlan {
    pub reachable_blocks: Vec<usize>,
    pub loops: Vec<StructuredLoop>,
    pub rejects: Vec<StructuredReject>,
}

impl StructuredPlan {
    pub fn supported(&self) -> bool { self.rejects.is_empty() }
}

struct RegionBlock<'a> {
    ssa: &'a Block,
    body: &'a [Site],
    yielding: bool,
    condition: Option<super::ir::Cond>,
}
struct RegionGraph<'a> {
    entry_pc: usize,
    blocks: BTreeMap<usize, RegionBlock<'a>>,
    ir: &'a Func,
    state: &'a StateGraph,
}
fn succs(block: &RegionBlock<'_>) -> Vec<usize> {
    block.ssa.term.edges().iter().map(|edge| edge.dst.0).collect()
}

fn predecessors(prog: &RegionGraph<'_>, reachable: &BTreeSet<usize>) -> BTreeMap<usize, BTreeSet<usize>> {
    let mut out: BTreeMap<usize, BTreeSet<usize>> = reachable.iter()
        .map(|&pc| (pc, BTreeSet::new())).collect();
    for &pc in reachable {
        for s in succs(&prog.blocks[&pc]) {
            if reachable.contains(&s) { out.get_mut(&s).unwrap().insert(pc); }
        }
    }
    out
}

fn reachable(prog: &RegionGraph<'_>) -> BTreeSet<usize> {
    let mut seen = BTreeSet::new();
    let mut todo = vec![prog.entry_pc];
    while let Some(pc) = todo.pop() {
        if !seen.insert(pc) { continue; }
        if let Some(block) = prog.blocks.get(&pc) {
            todo.extend(succs(block).into_iter().filter(|s| prog.blocks.contains_key(s)));
        }
    }
    seen
}

fn dominators(
    prog: &RegionGraph<'_>,
    reachable: &BTreeSet<usize>,
    preds: &BTreeMap<usize, BTreeSet<usize>>,
) -> BTreeMap<usize, BTreeSet<usize>> {
    let mut dom: BTreeMap<usize, BTreeSet<usize>> = reachable.iter().map(|&pc| {
        let init = if pc == prog.entry_pc {
            std::iter::once(pc).collect()
        } else {
            reachable.clone()
        };
        (pc, init)
    }).collect();
    loop {
        let mut changed = false;
        for &pc in reachable {
            if pc == prog.entry_pc { continue; }
            let ps = &preds[&pc];
            let mut next = if let Some((&first, rest)) = ps.iter().next().map(|p| (p, ps.iter().skip(1))) {
                let mut d = dom[&first].clone();
                for p in rest { d = d.intersection(&dom[p]).copied().collect(); }
                d
            } else {
                BTreeSet::new()
            };
            next.insert(pc);
            if next != dom[&pc] { dom.insert(pc, next); changed = true; }
        }
        if !changed { return dom; }
    }
}

fn natural_loop(
    header: usize,
    latch: usize,
    preds: &BTreeMap<usize, BTreeSet<usize>>,
) -> BTreeSet<usize> {
    let mut body = BTreeSet::from([header, latch]);
    let mut todo = vec![latch];
    while let Some(pc) = todo.pop() {
        for &p in &preds[&pc] {
            if body.insert(p) && p != header { todo.push(p); }
        }
    }
    body
}

fn region_rpo(prog: &RegionGraph<'_>, header: usize, body: &BTreeSet<usize>) -> Vec<usize> {
    fn visit(
        pc: usize,
        prog: &RegionGraph<'_>,
        body: &BTreeSet<usize>,
        seen: &mut BTreeSet<usize>,
        postorder: &mut Vec<usize>,
    ) {
        if !seen.insert(pc) { return; }
        for succ in succs(&prog.blocks[&pc]) {
            if body.contains(&succ) { visit(succ, prog, body, seen, postorder); }
        }
        postorder.push(pc);
    }

    let mut seen = BTreeSet::new();
    let mut postorder = Vec::with_capacity(body.len());
    visit(header, prog, body, &mut seen, &mut postorder);
    // A well-formed natural loop has every body block reachable from its
    // header. Keep malformed input deterministic for diagnostics instead of
    // silently omitting a block from the reported traversal order.
    for &pc in body {
        visit(pc, prog, body, &mut seen, &mut postorder);
    }
    postorder.reverse();
    postorder
}

fn live_in(prog: &RegionGraph<'_>, reachable: &BTreeSet<usize>) -> BTreeMap<usize, BTreeSet<u32>> {
    let mut dependencies = vec![Vec::new(); prog.ir.types.len()];
    let mut pending = Vec::new();
    for &pc in reachable {
        pending.extend(prog.blocks[&pc].body.iter().flat_map(|site| site.reads.iter().copied()));
        for edge in prog.blocks[&pc].ssa.term.edges() {
            if !reachable.contains(&edge.dst.0) { continue; }
            for &(slot, index) in &prog.state.vector_parameters {
                let parameter = prog.ir.blocks[&edge.dst].params[index].0;
                dependencies[parameter.0].push(prog.state.outgoing[&pc][&slot]);
            }
        }
    }
    let mut live = vec![false; prog.ir.types.len()];
    while let Some(value) = pending.pop() {
        if !live[value.0] { live[value.0] = true; pending.extend(&dependencies[value.0]); }
    }
    reachable.iter().map(|&pc| (pc, prog.state.vector_parameters.iter().filter_map(|&(slot, index)|
        live[prog.blocks[&pc].ssa.params[index].0.0].then_some(slot)).collect())).collect()
}

fn mask_value_sgprs(prog: &RegionGraph<'_>, body: &BTreeSet<usize>, saved: &BTreeSet<u32>) -> BTreeSet<u32> {
    let mut masks = BTreeSet::from([106, 126]);
    masks.extend(saved);
    loop {
        let before = masks.len();
        for &pc in body {
            for site in prog.blocks[&pc].body {
                masks.extend(&site.masks.seed);
                if site.masks.closure.iter().any(|r| masks.contains(r)) { masks.extend(&site.masks.closure); }
            }
        }
        if before == masks.len() { return masks; }
    }
}

fn scalar_mask_alias_sites(
    prog: &RegionGraph<'_>, body: &BTreeSet<usize>, header: usize, masks: &BTreeSet<u32>,
) -> Vec<StructuredScalarMaskAlias> {
    let mut facts = vec![false; prog.ir.types.len()];
    let seeds: Vec<_> = prog.state.scalar_parameters.iter().filter_map(|&(slot, index)|
        matches!(slot, 106 | 126).then_some(prog.blocks[&header].ssa.params[index].0)).collect();
    loop {
        let previous = facts.clone();
        for &value in &seeds { facts[value.0] = true; }
        for &pc in body {
            for site in prog.blocks[&pc].body {
                for (value, inputs, fixed) in &site.masks.definitions {
                    facts[value.0] = *fixed || inputs.iter().any(|v| facts[v.0]);
                }
            }
        }
        for &pc in body {
            for edge in prog.blocks[&pc].ssa.term.edges() {
                if !body.contains(&edge.dst.0) { continue; }
                for &(slot, index) in &prog.state.scalar_parameters {
                    if masks.contains(&slot) {
                        let parameter = prog.ir.blocks[&edge.dst].params[index].0;
                        facts[parameter.0] |= facts[prog.state.scalar_outgoing[&pc][&slot].0];
                    }
                }
            }
        }
        if facts == previous { break; }
    }
    let mut result = Vec::new();
    for &pc in body {
        for (instruction, site) in prog.blocks[&pc].body.iter().enumerate() {
            for &(sgpr, value) in &site.masks.reads {
                if masks.contains(&sgpr) && facts[value.0] { result.push(StructuredScalarMaskAlias { sgpr, pc, instruction }); }
            }
        }
    }
    result
}

fn saves(site: &Site, slot: u32) -> bool {
    matches!(site.masks.event, Some(MaskEvent::Save(r) | MaskEvent::Copy(r)) if r == slot)
}
fn restores(site: &Site, slot: u32) -> bool {
    matches!(&site.masks.event, Some(MaskEvent::Logic { restore: true, sources }) if sources.contains(&slot))
}

fn saved_mask_reaches_boundary(
    prog: &RegionGraph<'_>,
    header: usize,
    body: &BTreeSet<usize>,
    latches: &BTreeSet<usize>,
    exits: &BTreeSet<(usize, usize)>,
    reg: u32,
) -> bool {
    let mut entry: BTreeMap<usize, bool> = body.iter().map(|&pc| (pc, false)).collect();
    loop {
        let mut changed = false;
        for &pc in body {
            let mut live = entry[&pc];
            for inst in prog.blocks[&pc].body {
                if saves(inst, reg) {
                    live = true;
                } else if restores(inst, reg) {
                    live = false;
                }
            }
            for succ in succs(&prog.blocks[&pc]) {
                if body.contains(&succ) && live && !entry[&succ] {
                    entry.insert(succ, true);
                    changed = true;
                }
            }
        }
        if !changed { break; }
    }

    for &pc in body {
        let mut live = entry[&pc];
        for inst in prog.blocks[&pc].body {
            if saves(inst, reg) {
                live = true;
            } else if restores(inst, reg) {
                live = false;
            }
        }
        if live && (latches.contains(&pc) || exits.iter().any(|&(from, _)| from == pc)) {
            return true;
        }
    }
    let _ = header; // Documents that the external entry state is false.
    false
}

fn mask_stack(
    prog: &RegionGraph<'_>,
    header: usize,
    body: &BTreeSet<usize>,
    latches: &BTreeSet<usize>,
    exits: &BTreeSet<(usize, usize)>,
    rpo_body: &[usize],
) -> StructuredMaskStack {
    let saved: BTreeSet<_> = rpo_body.iter().flat_map(|pc| prog.blocks[pc].body).filter_map(|site|
        match site.masks.event { Some(MaskEvent::Save(r) | MaskEvent::Copy(r)) => Some(r), _ => None }).collect();
    let mut operations = Vec::new();
    let mut restored = BTreeSet::new();
    for &pc in rpo_body {
        for (instruction, site) in prog.blocks[&pc].body.iter().enumerate() {
            let (kind, saved_sgpr) = match &site.masks.event {
                Some(MaskEvent::Save(r)) => (StructuredMaskOpKind::SaveExec, Some(*r)),
                Some(MaskEvent::Copy(r)) => (StructuredMaskOpKind::CopyExec, Some(*r)),
                Some(MaskEvent::Logic { restore, sources }) => {
                    let selected = sources.iter().copied().find(|r| saved.contains(r));
                    let kind = if *restore && selected.is_some() {
                        restored.insert(selected.unwrap()); StructuredMaskOpKind::RestoreExec
                    } else { StructuredMaskOpKind::ExecLogic };
                    (kind, selected)
                },
                Some(MaskEvent::Compare) => (StructuredMaskOpKind::CmpxNarrowExec, None),
                None => continue,
            };
            operations.push(StructuredMaskOp { pc, instruction, kind, saved_sgpr });
        }
    }
    let boundary_live_saved: Vec<u32> = saved.iter().copied()
        .filter(|&reg| saved_mask_reaches_boundary(prog, header, body, latches, exits, reg))
        .collect();
    let mut local_scopes = Vec::new();
    for &saved_sgpr in &saved {
        if boundary_live_saved.contains(&saved_sgpr) || !restored.contains(&saved_sgpr) {
            continue;
        }
        let saves: Vec<&StructuredMaskOp> = operations.iter().filter(|op| {
            op.saved_sgpr == Some(saved_sgpr)
                && matches!(op.kind, StructuredMaskOpKind::SaveExec | StructuredMaskOpKind::CopyExec)
        }).collect();
        let restores: Vec<&StructuredMaskOp> = operations.iter().filter(|op| {
            op.saved_sgpr == Some(saved_sgpr) && matches!(op.kind, StructuredMaskOpKind::RestoreExec)
        }).collect();
        if saves.len() == 1 && restores.len() == 1 {
            local_scopes.push(StructuredMaskScope {
                saved_sgpr,
                save: saves[0].clone(),
                restore: restores[0].clone(),
            });
        }
    }
    let mask_sgprs = mask_value_sgprs(prog, body, &saved);
    let scalar_alias_sites = scalar_mask_alias_sites(prog, body, header, &mask_sgprs);
    let scalar_alias_sgprs = scalar_alias_sites.iter().map(|site| site.sgpr).collect::<BTreeSet<_>>();
    StructuredMaskStack {
        operations,
        saved_sgprs: saved.iter().copied().collect(),
        mask_sgprs: mask_sgprs.into_iter().collect(),
        scalar_alias_sgprs: scalar_alias_sgprs.into_iter().collect(),
        scalar_alias_sites,
        unrestored_saved: saved.difference(&restored).copied().collect(),
        boundary_live_saved,
        local_scopes,
    }
}

fn finish_order(pc: usize, prog: &RegionGraph<'_>, reachable: &BTreeSet<usize>, seen: &mut BTreeSet<usize>, out: &mut Vec<usize>) {
    if !seen.insert(pc) { return; }
    for s in succs(&prog.blocks[&pc]) {
        if reachable.contains(&s) { finish_order(s, prog, reachable, seen, out); }
    }
    out.push(pc);
}

fn reverse_sccs(
    prog: &RegionGraph<'_>,
    reachable: &BTreeSet<usize>,
    preds: &BTreeMap<usize, BTreeSet<usize>>,
) -> Vec<BTreeSet<usize>> {
    let mut order = Vec::new();
    let mut seen = BTreeSet::new();
    finish_order(prog.entry_pc, prog, reachable, &mut seen, &mut order);
    let mut components = Vec::new();
    seen.clear();
    while let Some(root) = order.pop() {
        if !seen.insert(root) { continue; }
        let mut component = BTreeSet::from([root]);
        let mut todo = vec![root];
        while let Some(pc) = todo.pop() {
            for &p in &preds[&pc] {
                if seen.insert(p) { component.insert(p); todo.push(p); }
            }
        }
        components.push(component);
    }
    components
}

/// Analyze natural loops, live VGPRs, branch conditions, and lane-mask state
/// without rewriting the program.
pub(super) fn analyze(function: &super::lift::function::LiftedFunction) -> StructuredPlan {
    let graph = RegionGraph {
        entry_pc: function.ir.entry.0,
        blocks: function.ir.blocks.iter().map(|(&pc, block)| (pc.0, RegionBlock {
            ssa: block, body: &function.state.sites[&pc.0],
            yielding: function.state.yielding.contains(&pc.0), condition: function.state.conditions.get(&pc.0).copied(),
        })).collect(),
        ir: &function.ir, state: &function.state,
    };
    let prog = &graph;
    let reachable = reachable(prog);
    let preds = predecessors(prog, &reachable);
    let dom = dominators(prog, &reachable, &preds);
    let live = live_in(prog, &reachable);
    let mut rejects = Vec::new();

    for &pc in &reachable {
        let block = &prog.blocks[&pc];
        if block.yielding {
            rejects.push(StructuredReject::Barrier { pc });
        }
        if let Some(inst) = block.body.iter().position(|s| s.masks.cross_lane) {
            rejects.push(StructuredReject::CrossLane { pc, inst });
        }
    }

    for scc in reverse_sccs(prog, &reachable, &preds) {
        let cyclic = scc.len() > 1 || scc.iter().any(|pc| succs(&prog.blocks[pc]).contains(pc));
        if !cyclic { continue; }
        let entries: BTreeSet<usize> = scc.iter().filter(|&&pc| preds[&pc].iter().any(|p| !scc.contains(p))).copied().collect();
        if entries.len() > 1 {
            rejects.push(StructuredReject::IrreducibleScc {
                entries: entries.into_iter().collect(),
                nodes: scc.into_iter().collect(),
            });
        }
    }

    let mut latches: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    for &tail in &reachable {
        for head in succs(&prog.blocks[&tail]) {
            if reachable.contains(&head) && dom[&tail].contains(&head) {
                latches.entry(head).or_default().insert(tail);
            }
        }
    }

    let mut loops = Vec::new();
    for (header, tails) in latches {
        let mut body = BTreeSet::new();
        for &tail in &tails { body.extend(natural_loop(header, tail, &preds)); }
        let entries: BTreeSet<usize> = body.iter().filter(|&&pc| preds[&pc].iter().any(|p| !body.contains(p))).copied().collect();
        if entries != BTreeSet::from([header]) {
            rejects.push(StructuredReject::MultiEntryLoop {
                header,
                entries: entries.into_iter().collect(),
            });
        }
        let mut exits = BTreeSet::new();
        for &pc in &body {
            for s in succs(&prog.blocks[&pc]) {
                if !body.contains(&s) { exits.insert((pc, s)); }
            }
        }
        let writes: BTreeSet<u32> = body.iter().flat_map(|pc| {
            prog.blocks[pc].body.iter().flat_map(|s| s.writes.iter().map(|&(slot, _)| slot))
        }).collect();
        let entry_vgprs: Vec<u32> = live[&header].iter().copied().collect();
        let carried: BTreeSet<u32> = live[&header].intersection(&writes).copied().collect();
        let core: BTreeSet<u32> = carried.iter().filter(|&&r| body.iter().all(|pc| live[pc].contains(&r))).copied().collect();
        let gapped: Vec<u32> = carried.difference(&core).copied().collect();
        let exit_state = exits.iter().map(|&(from, to)| StructuredExitState {
            from,
            to,
            live_vgprs: live[&to].iter().copied().collect(),
        }).collect();
        let mut branch_conditions = BTreeSet::new();
        let mut conditional_blocks = Vec::new();
        for &pc in &body {
            if let Some(cond) = prog.blocks[&pc].condition {
                branch_conditions.insert(cond);
                conditional_blocks.push(pc);
            }
        }
        let rpo_body = region_rpo(prog, header, &body);
        let mask_stack = mask_stack(prog, header, &body, &tails, &exits, &rpo_body);
        loops.push(StructuredLoop {
            header,
            parent: None,
            children: Vec::new(),
            latches: tails.into_iter().collect(),
            body: body.into_iter().collect(),
            exclusive_body: Vec::new(),
            rpo_body,
            exits: exits.into_iter().collect(),
            entry_vgprs,
            exit_state,
            control: StructuredControl {
                branch_conditions: branch_conditions.into_iter().collect(),
                conditional_blocks,
            },
            mask_stack,
            carried: carried.into_iter().collect(),
            core_carried: core.into_iter().collect(),
            gapped_carried: gapped,
        });
    }
    loops.sort_by_key(|l| (l.body.len(), l.header));

    // Form the loop forest from set containment.  The nearest strict superset
    // is the immediate parent; a loop may have many children but has at most
    // one such parent in a reducible CFG.  Keeping this as analysis data (not
    // a rewrite) keeps the nesting relation independently testable.
    let bodies: Vec<BTreeSet<usize>> = loops.iter()
        .map(|lp| lp.body.iter().copied().collect())
        .collect();
    let parents: Vec<Option<usize>> = (0..loops.len()).map(|child| {
        (0..loops.len())
            .filter(|&candidate| {
                bodies[candidate].len() > bodies[child].len()
                    && bodies[child].is_subset(&bodies[candidate])
            })
            .min_by_key(|&candidate| (bodies[candidate].len(), loops[candidate].header))
    }).collect();
    for (child, parent) in parents.iter().copied().enumerate() {
        loops[child].parent = parent.map(|index| loops[index].header);
    }
    for (child, parent) in parents.iter().copied().enumerate() {
        if let Some(parent) = parent {
            let child_header = loops[child].header;
            loops[parent].children.push(child_header);
        }
    }
    for loop_index in 0..loops.len() {
        loops[loop_index].children.sort_unstable();
        let mut nested = BTreeSet::new();
        for &child_header in &loops[loop_index].children {
            let child = loops.iter().position(|lp| lp.header == child_header).unwrap();
            nested.extend(bodies[child].iter().copied());
        }
        loops[loop_index].exclusive_body = bodies[loop_index]
            .difference(&nested)
            .copied()
            .collect();
    }
    StructuredPlan {
        reachable_blocks: reachable.into_iter().collect(),
        loops,
        rejects,
    }
}

pub fn analyze_structured(program: &impl super::CompilationInput) -> StructuredPlan {
    analyze(&program.to_ssa().function)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::rdna_instructions::{SOP1, SOP2, SOPK};
    use crate::rdna_spmd::{Cond, ScalarBlock};

    fn block(pc: usize, term: Terminator) -> ScalarBlock {
        ScalarBlock { pc, body: vec![], term }
    }

    #[test]
    fn recovers_single_entry_natural_loop() {
        let mut blocks = BTreeMap::new();
        blocks.insert(0, block(0, Terminator::Jump(1)));
        blocks.insert(1, block(1, Terminator::Branch {
            cond: Cond::Scc1, taken: 3, fallthrough: 2,
        }));
        blocks.insert(2, block(2, Terminator::Jump(1)));
        blocks.insert(3, block(3, Terminator::Return));
        let plan = analyze_structured(&ScalarProgram { entry_pc: 0, blocks });
        assert!(plan.supported());
        assert_eq!(plan.loops.len(), 1);
        assert_eq!(plan.loops[0].header, 1);
        assert_eq!(plan.loops[0].parent, None);
        assert!(plan.loops[0].children.is_empty());
        assert_eq!(plan.loops[0].body, vec![1, 2]);
        assert_eq!(plan.loops[0].exclusive_body, vec![1, 2]);
        assert_eq!(plan.loops[0].rpo_body, vec![1, 2]);
        assert_eq!(plan.loops[0].latches, vec![2]);
        assert_eq!(plan.loops[0].exits, vec![(1, 3)]);
        assert_eq!(plan.loops[0].entry_vgprs, Vec::<u32>::new());
        assert_eq!(plan.loops[0].exit_state, vec![StructuredExitState {
            from: 1,
            to: 3,
            live_vgprs: vec![],
        }]);
        assert_eq!(plan.loops[0].control, StructuredControl {
            branch_conditions: vec![Cond::Scc1],
            conditional_blocks: vec![1],
        });
        assert!(plan.loops[0].mask_stack.operations.is_empty());
    }

    #[test]
    fn recovers_nested_loop_tree_and_exclusive_bodies() {
        let mut blocks = BTreeMap::new();
        blocks.insert(0, block(0, Terminator::Jump(1)));
        blocks.insert(1, block(1, Terminator::Branch {
            cond: Cond::Scc1, taken: 5, fallthrough: 2,
        }));
        blocks.insert(2, block(2, Terminator::Branch {
            cond: Cond::Scc1, taken: 4, fallthrough: 3,
        }));
        blocks.insert(3, block(3, Terminator::Jump(2)));
        blocks.insert(4, block(4, Terminator::Jump(1)));
        blocks.insert(5, block(5, Terminator::Return));
        let plan = analyze_structured(&ScalarProgram { entry_pc: 0, blocks });
        assert!(plan.supported());
        assert_eq!(plan.loops.len(), 2);
        let inner = plan.loops.iter().find(|lp| lp.header == 2).unwrap();
        let outer = plan.loops.iter().find(|lp| lp.header == 1).unwrap();
        assert_eq!(inner.parent, Some(1));
        assert_eq!(inner.exclusive_body, vec![2, 3]);
        assert_eq!(outer.children, vec![2]);
        assert_eq!(outer.exclusive_body, vec![1, 4]);
    }

    #[test]
    fn rejects_multi_entry_cycle() {
        let mut blocks = BTreeMap::new();
        blocks.insert(0, block(0, Terminator::Branch {
            cond: Cond::Scc1, taken: 1, fallthrough: 2,
        }));
        blocks.insert(1, block(1, Terminator::Jump(2)));
        blocks.insert(2, block(2, Terminator::Jump(1)));
        let plan = analyze_structured(&ScalarProgram { entry_pc: 0, blocks });
        assert!(plan.rejects.iter().any(|r| matches!(r, StructuredReject::IrreducibleScc { .. })));
    }

    #[test]
    fn reports_mask_value_reused_as_scalar() {
        let mut blocks = BTreeMap::new();
        blocks.insert(0, block(0, Terminator::Jump(1)));
        let mut header = block(1, Terminator::Branch {
            cond: Cond::Scc1, taken: 3, fallthrough: 2,
        });
        header.body = vec![
            InstFormat::SOP1(SOP1 {
                ssrc0: SourceOperand::ScalarRegister(106),
                sdst: 5,
                op: I::S_AND_SAVEEXEC_B32,
            }),
            InstFormat::SOP2(SOP2 {
                ssrc0: SourceOperand::ScalarRegister(5),
                ssrc1: SourceOperand::IntegerConstant(1),
                sdst: 6,
                op: I::S_ADD_U32,
            }),
            InstFormat::SOPK(SOPK { simm16: 7, sdst: 5, op: I::S_MOVK_I32 }),
        ];
        blocks.insert(1, header);
        blocks.insert(2, block(2, Terminator::Jump(1)));
        blocks.insert(3, block(3, Terminator::Return));

        let plan = analyze_structured(&ScalarProgram { entry_pc: 0, blocks });
        let masks = &plan.loops[0].mask_stack;
        assert!(masks.mask_sgprs.contains(&5));
        assert_eq!(masks.scalar_alias_sgprs, vec![5]);
        assert_eq!(masks.scalar_alias_sites, vec![StructuredScalarMaskAlias {
            sgpr: 5,
            pc: 1,
            instruction: 1,
        }]);
    }

}
