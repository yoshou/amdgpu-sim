//! Conservative natural-loop and lane-mask analysis for vector code generation.
//!
//! This module does not rewrite [`ScalarProgram`]. It describes natural loops,
//! values carried across their boundaries, and EXEC/VCC save-and-restore
//! patterns found in the decoded instructions. The vector emitter currently
//! uses a conservative subset of this result to keep lane masks in vector form
//! inside one eligible leaf loop. Other loops continue through the ordinary
//! packed-work-item code-generation path.

use std::collections::{BTreeMap, BTreeSet};

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand};

use super::freshness::vgpr_writes;
use super::ir::{ScalarBlock, ScalarProgram, Terminator};
use super::vec_live::vgpr_reads;

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

fn succs(block: &ScalarBlock) -> Vec<usize> {
    match block.term {
        Terminator::Return => vec![],
        Terminator::Jump(t) => vec![t],
        Terminator::Branch { taken, fallthrough, .. } => vec![taken, fallthrough],
        Terminator::Barrier { resume } => vec![resume],
    }
}

fn predecessors(prog: &ScalarProgram, reachable: &BTreeSet<usize>) -> BTreeMap<usize, BTreeSet<usize>> {
    let mut out: BTreeMap<usize, BTreeSet<usize>> = reachable.iter()
        .map(|&pc| (pc, BTreeSet::new())).collect();
    for &pc in reachable {
        for s in succs(&prog.blocks[&pc]) {
            if reachable.contains(&s) { out.get_mut(&s).unwrap().insert(pc); }
        }
    }
    out
}

fn reachable(prog: &ScalarProgram) -> BTreeSet<usize> {
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
    prog: &ScalarProgram,
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

fn region_rpo(prog: &ScalarProgram, header: usize, body: &BTreeSet<usize>) -> Vec<usize> {
    fn visit(
        pc: usize,
        prog: &ScalarProgram,
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

fn live_in(prog: &ScalarProgram, reachable: &BTreeSet<usize>) -> BTreeMap<usize, BTreeSet<u32>> {
    let mut live: BTreeMap<usize, BTreeSet<u32>> = reachable.iter()
        .map(|&pc| (pc, BTreeSet::new())).collect();
    loop {
        let mut changed = false;
        for &pc in reachable {
            let mut cur = BTreeSet::new();
            for s in succs(&prog.blocks[&pc]) {
                if let Some(v) = live.get(&s) { cur.extend(v.iter().copied()); }
            }
            for inst in prog.blocks[&pc].body.iter().rev() {
                for w in vgpr_writes(inst) { cur.remove(&w); }
                cur.extend(vgpr_reads(inst));
            }
            if cur != live[&pc] { live.insert(pc, cur); changed = true; }
        }
        if !changed { return live; }
    }
}

fn has_cross_lane(block: &ScalarBlock) -> Option<usize> {
    block.body.iter().position(|inst| match inst {
        crate::rdna_instructions::InstFormat::VOP3(i) =>
            matches!(i.op, I::V_READLANE_B32 | I::V_WRITELANE_B32),
        crate::rdna_instructions::InstFormat::VOP3P(i) =>
            matches!(i.op, I::V_WMMA_F32_16X16X16_F16),
        crate::rdna_instructions::InstFormat::DS(i) =>
            matches!(i.op, I::DS_BPERMUTE_B32 | I::DS_BPERMUTE_FI_B32),
        _ => false,
    })
}

fn scalar_reg(op: &SourceOperand) -> Option<u32> {
    match op {
        SourceOperand::ScalarRegister(reg) => Some(*reg as u32),
        _ => None,
    }
}

fn is_saveexec(inst: &InstFormat, reg: u32) -> bool {
    matches!(inst, InstFormat::SOP1(i)
        if i.sdst as u32 == reg && matches!(i.op,
            I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 |
            I::S_OR_SAVEEXEC_B32 | I::S_XOR_SAVEEXEC_B32))
}

fn is_copyexec(inst: &InstFormat, reg: u32) -> bool {
    matches!(inst, InstFormat::SOP1(i)
        if i.sdst as u32 == reg && matches!(i.op, I::S_MOV_B32)
            && scalar_reg(&i.ssrc0) == Some(126))
}

fn restores_exec(inst: &InstFormat, reg: u32) -> bool {
    matches!(inst, InstFormat::SOP2(i)
        if i.sdst as u32 == 126 && matches!(i.op, I::S_OR_B32)
            && (scalar_reg(&i.ssrc0) == Some(reg) || scalar_reg(&i.ssrc1) == Some(reg)))
}

fn mask_logic_op(op: I) -> bool {
    matches!(op,
        I::S_AND_B32 | I::S_OR_B32 | I::S_XOR_B32 |
        I::S_AND_NOT1_B32 | I::S_OR_NOT1_B32)
}

fn saveexec_op(op: I) -> bool {
    matches!(op,
        I::S_AND_SAVEEXEC_B32 | I::S_AND_NOT1_SAVEEXEC_B32 |
        I::S_OR_SAVEEXEC_B32 | I::S_XOR_SAVEEXEC_B32)
}

/// The same mask-value closure a direct emitter would need, expressed only in
/// terms of decoded ISA.  It deliberately admits a scalar operand of a B32
/// logic operation: bitwise operations on packed lane masks remain valid for
/// any 32-bit input, but a later non-mask use of that operand is reported by
/// [`scalar_mask_alias_sites`].
fn mask_value_sgprs(
    prog: &ScalarProgram,
    body: &BTreeSet<usize>,
    saved: &BTreeSet<u32>,
) -> BTreeSet<u32> {
    const EXEC: u32 = 126;
    const VCC: u32 = 106;
    let mut masks = BTreeSet::from([EXEC, VCC]);
    masks.extend(saved);
    loop {
        let before = masks.len();
        for &pc in body {
            for inst in &prog.blocks[&pc].body {
                match inst {
                    InstFormat::SOP1(i) if matches!(i.op, I::S_MOV_B32) => {
                        let dst = i.sdst as u32;
                        let src = scalar_reg(&i.ssrc0);
                        if masks.contains(&dst) || src.is_some_and(|r| masks.contains(&r)) {
                            masks.insert(dst);
                            if let Some(src) = src { masks.insert(src); }
                        }
                    }
                    InstFormat::SOP1(i) if saveexec_op(i.op) => {
                        masks.insert(i.sdst as u32);
                        if let Some(src) = scalar_reg(&i.ssrc0) { masks.insert(src); }
                    }
                    InstFormat::SOP2(i) if mask_logic_op(i.op) => {
                        let srcs = [scalar_reg(&i.ssrc0), scalar_reg(&i.ssrc1)];
                        if masks.contains(&(i.sdst as u32))
                            || srcs.iter().flatten().any(|r| masks.contains(r))
                        {
                            masks.insert(i.sdst as u32);
                            masks.extend(srcs.iter().flatten().copied());
                        }
                    }
                    InstFormat::VOPC(i) if format!("{:?}", i.op).starts_with("V_CMP") => {
                        masks.insert(if format!("{:?}", i.op).starts_with("V_CMPX") { EXEC } else { VCC });
                    }
                    InstFormat::VOP3(i)
                        if format!("{:?}", i.op).starts_with("V_CMP") =>
                    {
                        masks.insert(if format!("{:?}", i.op).starts_with("V_CMPX") { EXEC } else { VCC });
                    }
                    _ => {}
                }
            }
        }
        if masks.len() == before { return masks; }
    }
}

fn operand_is_reg(op: &SourceOperand, reg: u32) -> bool {
    scalar_reg(op) == Some(reg)
}

fn range_contains(first: u32, words: u32, reg: u32) -> bool {
    (first..first.saturating_add(words)).contains(&reg)
}

/// Whether `inst` reads `reg` as an ordinary scalar value rather than as a
/// packed lane mask. Destinations are deliberately absent: a scalar
/// redefinition after the mask value is dead is harmless, and the reaching
/// definition transfer below kills it before a later read.
fn scalar_mask_read(inst: &InstFormat, reg: u32) -> bool {
    match inst {
        InstFormat::SOP1(i) => !(matches!(i.op, I::S_MOV_B32) || saveexec_op(i.op))
            && operand_is_reg(&i.ssrc0, reg),
        InstFormat::SOP2(i) => !mask_logic_op(i.op)
            && (operand_is_reg(&i.ssrc0, reg) || operand_is_reg(&i.ssrc1, reg)),
        InstFormat::SOPK(_) => false,
        InstFormat::SOPC(i) => operand_is_reg(&i.ssrc0, reg) || operand_is_reg(&i.ssrc1, reg),
        InstFormat::SOPP(_) => false,
        InstFormat::SMEM(i) => {
            range_contains(i.sbase as u32, 2, reg)
                || i.soffset as u32 == reg
        }
        InstFormat::VOP1(i) => operand_is_reg(&i.src0, reg),
        InstFormat::VOP2(i) => operand_is_reg(&i.src0, reg),
        InstFormat::VOP3(i) => {
            // V_CMP writes EXEC/VCC implicitly; its vector operands do not
            // alias an SGPR unless one is explicitly encoded as a source.
            operand_is_reg(&i.src0, reg) || operand_is_reg(&i.src1, reg) || operand_is_reg(&i.src2, reg)
        }
        InstFormat::VOP3SD(i) => operand_is_reg(&i.src0, reg)
            || operand_is_reg(&i.src1, reg) || operand_is_reg(&i.src2, reg),
        InstFormat::VOP3P(i) => operand_is_reg(&i.src0, reg)
            || operand_is_reg(&i.src1, reg) || operand_is_reg(&i.src2, reg),
        InstFormat::VOPC(i) => operand_is_reg(&i.src0, reg),
        InstFormat::VOPD(i) => operand_is_reg(&i.src0x, reg) || operand_is_reg(&i.src0y, reg),
        InstFormat::VFLAT(i) => {
            i.saddr != 124 && i.saddr != 127 && range_contains(i.saddr as u32, 2, reg)
        }
        InstFormat::VGLOBAL(i) => {
            i.saddr != 124 && i.saddr != 127 && range_contains(i.saddr as u32, 2, reg)
        }
        InstFormat::VSCRATCH(i) => {
            i.saddr != 124 && i.saddr != 127 && range_contains(i.saddr as u32, 2, reg)
        }
        InstFormat::VIMAGE(i) => range_contains(i.rsrc as u32, 4, reg),
        InstFormat::VSAMPLE(i) => range_contains(i.rsrc as u32, 4, reg) || range_contains(i.samp as u32, 4, reg),
        InstFormat::DS(_) => false,
    }
}

pub(super) fn scalar_write_regs(inst: &InstFormat) -> Vec<u32> {
    let pair = |reg: u8| vec![reg as u32, reg as u32 + 1];
    let one = |reg: u8| vec![reg as u32];
    match inst {
        InstFormat::SOP1(i) => if matches!(i.op, I::S_MOV_B64) { pair(i.sdst) } else { one(i.sdst) },
        InstFormat::SOP2(i) if matches!(i.op,
            I::S_ADD_NC_U64 | I::S_MUL_U64 | I::S_LSHL_B64 | I::S_LSHR_B64 |
            I::S_ASHR_I64 | I::S_AND_B64 | I::S_OR_B64 | I::S_XOR_B64 | I::S_CSELECT_B64) => pair(i.sdst),
        InstFormat::SOP2(i) => one(i.sdst),
        InstFormat::SOPK(i) => one(i.sdst),
        InstFormat::VOP3SD(i) => one(i.sdst),
        InstFormat::SMEM(i) => {
            let words = match i.op {
                I::S_LOAD_B32 | I::S_LOAD_U16 => 1,
                I::S_LOAD_B64 => 2,
                I::S_LOAD_B96 => 3,
                I::S_LOAD_B128 => 4,
                I::S_LOAD_B256 => 8,
                I::S_LOAD_B512 => 16,
                _ => 1,
            };
            (0..words).map(|offset| i.sdata as u32 + offset).collect()
        }
        _ => vec![],
    }
}

fn update_mask_definition(
    masks: &mut BTreeSet<u32>,
    candidates: &BTreeSet<u32>,
    reg: u32,
    is_mask: bool,
) {
    if !candidates.contains(&reg) { return; }
    if is_mask { masks.insert(reg); } else { masks.remove(&reg); }
}

/// Transfer the may-reaching set of packed-mask values through one decoded
/// instruction. A union at CFG joins is required: if any path reaches an
/// ordinary scalar read with a mask-derived value, the SGPR cannot be replaced
/// solely by a vector of lane booleans.
fn transfer_reaching_masks(
    inst: &InstFormat,
    masks: &mut BTreeSet<u32>,
    candidates: &BTreeSet<u32>,
) {
    const EXEC: u32 = 126;
    const VCC: u32 = 106;
    let source_mask = |op: &SourceOperand, masks: &BTreeSet<u32>| {
        scalar_reg(op).is_some_and(|reg| masks.contains(&reg))
    };
    match inst {
        InstFormat::SOP1(i) if saveexec_op(i.op) => {
            update_mask_definition(masks, candidates, i.sdst as u32, true);
            masks.insert(EXEC);
        }
        InstFormat::SOP1(i) if matches!(i.op, I::S_MOV_B32) => {
            let is_mask = source_mask(&i.ssrc0, masks);
            update_mask_definition(masks, candidates, i.sdst as u32, is_mask);
        }
        InstFormat::SOP2(i) if mask_logic_op(i.op) => {
            let is_mask = source_mask(&i.ssrc0, masks) || source_mask(&i.ssrc1, masks);
            update_mask_definition(masks, candidates, i.sdst as u32, is_mask);
        }
        InstFormat::VOPC(i) if format!("{:?}", i.op).starts_with("V_CMP") => {
            masks.insert(if format!("{:?}", i.op).starts_with("V_CMPX") { EXEC } else { VCC });
        }
        InstFormat::VOP3(i) if format!("{:?}", i.op).starts_with("V_CMP") => {
            masks.insert(if format!("{:?}", i.op).starts_with("V_CMPX") { EXEC } else { VCC });
        }
        InstFormat::VOP3SD(i) => {
            // The scalar destination is the per-lane carry/borrow mask.
            update_mask_definition(masks, candidates, i.sdst as u32, true);
        }
        _ => {
            for reg in scalar_write_regs(inst) {
                // EXEC and VCC are architecturally packed lane masks even
                // when an unusual SALU instruction writes them.
                update_mask_definition(masks, candidates, reg, reg == EXEC || reg == VCC);
            }
        }
    }
}

fn reaching_mask_entry(
    prog: &ScalarProgram,
    body: &BTreeSet<usize>,
    header: usize,
    candidates: &BTreeSet<u32>,
) -> BTreeMap<usize, BTreeSet<u32>> {
    const EXEC: u32 = 126;
    const VCC: u32 = 106;
    let mut entry: BTreeMap<usize, BTreeSet<u32>> = body.iter()
        .map(|&pc| (pc, BTreeSet::new())).collect();
    entry.insert(header, BTreeSet::from([EXEC, VCC]));
    loop {
        let mut incoming: BTreeMap<usize, BTreeSet<u32>> = body.iter()
            .map(|&pc| (pc, if pc == header { BTreeSet::from([EXEC, VCC]) } else { BTreeSet::new() }))
            .collect();
        for &from in body {
            let mut out = entry[&from].clone();
            for inst in &prog.blocks[&from].body {
                transfer_reaching_masks(inst, &mut out, candidates);
            }
            for to in succs(&prog.blocks[&from]) {
                if body.contains(&to) { incoming.get_mut(&to).unwrap().extend(out.iter().copied()); }
            }
        }
        if incoming == entry { return entry; }
        entry = incoming;
    }
}

fn scalar_mask_alias_sites(
    prog: &ScalarProgram,
    body: &BTreeSet<usize>,
    header: usize,
    masks: &BTreeSet<u32>,
) -> Vec<StructuredScalarMaskAlias> {
    let entry = reaching_mask_entry(prog, body, header, masks);
    let mut sites = Vec::new();
    for &pc in body {
        let mut reaching = entry[&pc].clone();
        for (instruction, inst) in prog.blocks[&pc].body.iter().enumerate() {
            for &sgpr in &reaching {
                if scalar_mask_read(inst, sgpr) {
                    sites.push(StructuredScalarMaskAlias { sgpr, pc, instruction });
                }
            }
            transfer_reaching_masks(inst, &mut reaching, masks);
        }
    }
    sites
}

/// May-analysis of one saved mask through the loop CFG.  `true` means a save
/// performed in this loop may not yet have been restored on that path.
fn saved_mask_reaches_boundary(
    prog: &ScalarProgram,
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
            for inst in &prog.blocks[&pc].body {
                if is_saveexec(inst, reg) || is_copyexec(inst, reg) {
                    live = true;
                } else if restores_exec(inst, reg) {
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
        for inst in &prog.blocks[&pc].body {
            if is_saveexec(inst, reg) || is_copyexec(inst, reg) {
                live = true;
            } else if restores_exec(inst, reg) {
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
    prog: &ScalarProgram,
    header: usize,
    body: &BTreeSet<usize>,
    latches: &BTreeSet<usize>,
    exits: &BTreeSet<(usize, usize)>,
    rpo_body: &[usize],
) -> StructuredMaskStack {
    const EXEC: u32 = 126;
    let mut saved = BTreeSet::new();
    for &pc in rpo_body {
        for inst in &prog.blocks[&pc].body {
            match inst {
                InstFormat::SOP1(i)
                    if matches!(
                        i.op,
                        I::S_AND_SAVEEXEC_B32
                            | I::S_AND_NOT1_SAVEEXEC_B32
                            | I::S_OR_SAVEEXEC_B32
                            | I::S_XOR_SAVEEXEC_B32
                    ) =>
                {
                    saved.insert(i.sdst as u32);
                }
                InstFormat::SOP1(i)
                    if matches!(i.op, I::S_MOV_B32)
                        && scalar_reg(&i.ssrc0) == Some(EXEC) =>
                {
                    saved.insert(i.sdst as u32);
                }
                _ => {}
            }
        }
    }

    let mut operations = Vec::new();
    let mut restored = BTreeSet::new();
    for &pc in rpo_body {
        for (instruction, inst) in prog.blocks[&pc].body.iter().enumerate() {
            match inst {
                InstFormat::SOP1(i)
                    if matches!(
                        i.op,
                        I::S_AND_SAVEEXEC_B32
                            | I::S_AND_NOT1_SAVEEXEC_B32
                            | I::S_OR_SAVEEXEC_B32
                            | I::S_XOR_SAVEEXEC_B32
                    ) =>
                {
                    operations.push(StructuredMaskOp {
                        pc,
                        instruction,
                        kind: StructuredMaskOpKind::SaveExec,
                        saved_sgpr: Some(i.sdst as u32),
                    });
                }
                InstFormat::SOP1(i)
                    if matches!(i.op, I::S_MOV_B32)
                        && scalar_reg(&i.ssrc0) == Some(EXEC) =>
                {
                    operations.push(StructuredMaskOp {
                        pc,
                        instruction,
                        kind: StructuredMaskOpKind::CopyExec,
                        saved_sgpr: Some(i.sdst as u32),
                    });
                }
                InstFormat::SOP2(i) if i.sdst as u32 == EXEC => {
                    let saved_sgpr = [scalar_reg(&i.ssrc0), scalar_reg(&i.ssrc1)]
                        .iter()
                        .flatten()
                        .copied()
                        .find(|reg| saved.contains(reg));
                    let kind = if matches!(i.op, I::S_OR_B32) && saved_sgpr.is_some() {
                        restored.insert(saved_sgpr.unwrap());
                        StructuredMaskOpKind::RestoreExec
                    } else {
                        StructuredMaskOpKind::ExecLogic
                    };
                    operations.push(StructuredMaskOp { pc, instruction, kind, saved_sgpr });
                }
                InstFormat::VOPC(i) if format!("{:?}", i.op).starts_with("V_CMPX") => {
                    operations.push(StructuredMaskOp {
                        pc,
                        instruction,
                        kind: StructuredMaskOpKind::CmpxNarrowExec,
                        saved_sgpr: None,
                    });
                }
                InstFormat::VOP3(i)
                    if format!("{:?}", i.op).starts_with("V_CMPX")
                        && i.vdst as u32 == EXEC =>
                {
                    operations.push(StructuredMaskOp {
                        pc,
                        instruction,
                        kind: StructuredMaskOpKind::CmpxNarrowExec,
                        saved_sgpr: None,
                    });
                }
                _ => {}
            }
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

fn finish_order(pc: usize, prog: &ScalarProgram, reachable: &BTreeSet<usize>, seen: &mut BTreeSet<usize>, out: &mut Vec<usize>) {
    if !seen.insert(pc) { return; }
    for s in succs(&prog.blocks[&pc]) {
        if reachable.contains(&s) { finish_order(s, prog, reachable, seen, out); }
    }
    out.push(pc);
}

fn reverse_sccs(
    prog: &ScalarProgram,
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
pub fn analyze_structured(prog: &ScalarProgram) -> StructuredPlan {
    let reachable = reachable(prog);
    let preds = predecessors(prog, &reachable);
    let dom = dominators(prog, &reachable, &preds);
    let live = live_in(prog, &reachable);
    let mut rejects = Vec::new();

    for &pc in &reachable {
        let block = &prog.blocks[&pc];
        if matches!(block.term, Terminator::Barrier { .. }) {
            rejects.push(StructuredReject::Barrier { pc });
        }
        if let Some(inst) = has_cross_lane(block) {
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
            prog.blocks[pc].body.iter().flat_map(vgpr_writes)
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
            if let Terminator::Branch { cond, .. } = &prog.blocks[&pc].term {
                branch_conditions.insert(*cond);
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
