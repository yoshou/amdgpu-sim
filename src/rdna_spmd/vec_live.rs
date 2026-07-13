//! VGPR liveness for **sound** predication elision in the width-W SPMD path.
//!
//! In SIMT every VGPR write is predicated by EXEC so an inactive lane keeps its
//! old value. Dropping that predication (running the write unconditionally) lets
//! an inactive lane compute *garbage* into the destination. That is harmless iff
//! the garbage never escapes to somewhere a (later-reconverging) lane observes
//! it. The only escapes are (a) a register that is **live-out** of the block /
//! read later, and (b) memory. Memory stores are *always* masked by EXEC
//! (`masked_scatter`) in [`super::emit_vec`], and compares write EXEC/VCC masked
//! by EXEC (`st_cmp`). So the remaining escape is registers.
//!
//! The analysis below uses a coarser program-wide criterion: a VGPR write may
//! skip predication when none of its destinations is live at a CFG
//! reconvergence point (a block with multiple predecessors) or at a caller-
//! observed fragment exit. Such destinations are treated as transient values,
//! so their writes can omit unnecessary lane-preserving selects. Loop-carried
//! or reconvergence-visible state stays predicated.
//!
//! Soundness depends only on never *under*-counting reads: a missed read would
//! mark a live register dead and wrongly elide. Reads are therefore
//! over-approximated (a 32-bit VGPR source `r` is counted as reading the pair
//! `{r, r+1}`); over-counting only keeps more registers live (more predication),
//! which is always safe.

use std::collections::BTreeMap;

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand};

use super::freshness::vgpr_writes;
use super::ir::{ScalarBlock, ScalarProgram, Terminator};

fn src_vgpr(op: &SourceOperand, out: &mut Vec<u32>) {
    if let SourceOperand::VectorRegister(r) = op {
        out.push(*r as u32);
        out.push(*r as u32 + 1); // over-approx: covers f64/64-bit pair reads
    }
}

/// All VGPRs an instruction may read (over-approximated upward — never under).
pub fn vgpr_reads(inst: &InstFormat) -> Vec<u32> {
    let mut r = Vec::new();
    match inst {
        InstFormat::VOP1(i) => src_vgpr(&i.src0, &mut r),
        InstFormat::VOP2(i) => {
            src_vgpr(&i.src0, &mut r);
            r.push(i.vsrc1 as u32);
            r.push(i.vsrc1 as u32 + 1);
        }
        InstFormat::VOP3(i) => {
            src_vgpr(&i.src0, &mut r);
            src_vgpr(&i.src1, &mut r);
            src_vgpr(&i.src2, &mut r);
        }
        InstFormat::VOP3SD(i) => {
            src_vgpr(&i.src0, &mut r);
            src_vgpr(&i.src1, &mut r);
            src_vgpr(&i.src2, &mut r);
        }
        InstFormat::VOPC(i) => {
            src_vgpr(&i.src0, &mut r);
            r.push(i.vsrc1 as u32);
            r.push(i.vsrc1 as u32 + 1);
        }
        InstFormat::VOPD(i) => {
            src_vgpr(&i.src0x, &mut r);
            r.push(i.vsrc1x as u32);
            r.push(i.vsrc1x as u32 + 1);
            src_vgpr(&i.src0y, &mut r);
            r.push(i.vsrc1y as u32);
            r.push(i.vsrc1y as u32 + 1);
        }
        InstFormat::VGLOBAL(i) => {
            // address (may be a 64-bit VGPR pair) + store data words.
            r.push(i.vaddr as u32);
            r.push(i.vaddr as u32 + 1);
            let store_words = match i.op {
                I::GLOBAL_STORE_B32 => 1,
                I::GLOBAL_STORE_B64 => 2,
                I::GLOBAL_STORE_B96 => 3,
                I::GLOBAL_STORE_B128 => 4,
                _ => 0,
            };
            for k in 0..store_words {
                r.push(i.vsrc as u32 + k);
            }
        }
        _ => {} // SALU/SMEM/SOPC read SGPRs only
    }
    r
}

/// PRECISE VGPR reads for divergence analysis: a 32-bit operand reads only `r`
/// (not the over-approximated `{r,r+1}` used for liveness soundness), so the
/// 32-bit address arithmetic doesn't pick up a spurious divergent neighbour and
/// a uniform load address stays provably uniform. f64/64-bit operands read the
/// pair. (Over-counting here would only *add* divergence — conservative/sound —
/// but it loses the uniform-gather→broadcast optimization, so we read precisely.)
fn op_is_pair(op: I) -> bool {
    let s = format!("{:?}", op);
    (s.contains("F64") && !matches!(op, I::V_CVT_F64_U32 | I::V_CVT_F64_I32 | I::V_CVT_I32_F64))
        || matches!(op, I::V_MAD_CO_U64_U32 | I::V_LSHLREV_B64 | I::V_ASHR_I64 | I::V_ASHRREV_I64)
}
fn push_src(r: &mut Vec<u32>, o: &SourceOperand, pair: bool) {
    if let SourceOperand::VectorRegister(x) = o { r.push(*x as u32); if pair { r.push(*x as u32 + 1); } }
}
fn push_v(r: &mut Vec<u32>, idx: u8, pair: bool) { r.push(idx as u32); if pair { r.push(idx as u32 + 1); } }
pub fn div_reads(inst: &InstFormat) -> Vec<u32> {
    let mut r = Vec::new();
    match inst {
        InstFormat::VOP1(i) => push_src(&mut r, &i.src0, op_is_pair(i.op)),
        InstFormat::VOP2(i) => { let p = op_is_pair(i.op); push_src(&mut r, &i.src0, p); push_v(&mut r, i.vsrc1, p); }
        InstFormat::VOP3(i) => { let p = op_is_pair(i.op); push_src(&mut r, &i.src0, p); push_src(&mut r, &i.src1, p); push_src(&mut r, &i.src2, p); }
        InstFormat::VOP3SD(i) => { let p = op_is_pair(i.op); push_src(&mut r, &i.src0, p); push_src(&mut r, &i.src1, p); push_src(&mut r, &i.src2, p); }
        InstFormat::VOPC(i) => { let p = op_is_pair(i.op); push_src(&mut r, &i.src0, p); push_v(&mut r, i.vsrc1, p); }
        InstFormat::VOPD(i) => { push_src(&mut r, &i.src0x, false); push_v(&mut r, i.vsrc1x, false); push_src(&mut r, &i.src0y, false); push_v(&mut r, i.vsrc1y, false); }
        InstFormat::VGLOBAL(i) => {
            // address: 64-bit VGPR pair when saddr is null (124), else a 32-bit offset.
            r.push(i.vaddr as u32);
            if i.saddr == 124 { r.push(i.vaddr as u32 + 1); }
            let store_words = match i.op {
                I::GLOBAL_STORE_B32 => 1, I::GLOBAL_STORE_B64 => 2,
                I::GLOBAL_STORE_B96 => 3, I::GLOBAL_STORE_B128 => 4, _ => 0,
            };
            for k in 0..store_words { r.push(i.vsrc as u32 + k); }
        }
        _ => {}
    }
    r
}

/// Per-work-item AFFINE frame-pointer detection: a VGPR pair P = vgpr0*stride +
/// uniform_base (e.g. `V_MAD_CO_U64_U32 vdst, vgpr0, 24, sgpr8` — the de-recursed
/// kernel's per-work-item state frame at `base + work_item_id*24`). Because
/// consecutive packed lanes' records are contiguous (stride bytes apart), a load
/// of such a frame can be a **contiguous vector load + in-register transpose**
/// instead of a gather (gather/scatter coalescing, cf. TACO). The address is
/// vgpr0-derived ⇒ always valid (even for inactive lanes) ⇒ the contiguous load
/// can't fault. flow-sensitive, meet = intersection (same stride on all paths).
pub fn frame_transfer(inst: &InstFormat, map: &mut std::collections::HashMap<u32, u32>) {
    for w in vgpr_writes(inst) { map.remove(&w); if w > 0 { map.remove(&(w - 1)); } }
    if let InstFormat::VOP3SD(i) = inst {
        if matches!(i.op, I::V_MAD_CO_U64_U32) {
            let stride = match (&i.src0, &i.src1) {
                (SourceOperand::VectorRegister(0), SourceOperand::IntegerConstant(c)) => Some(*c as u32),
                (SourceOperand::IntegerConstant(c), SourceOperand::VectorRegister(0)) => Some(*c as u32),
                (SourceOperand::VectorRegister(0), SourceOperand::LiteralConstant(c)) => Some(*c),
                (SourceOperand::LiteralConstant(c), SourceOperand::VectorRegister(0)) => Some(*c),
                _ => None,
            };
            // base (src2) must be uniform (scalar reg / constant), and stride sane.
            let base_uniform = matches!(i.src2, SourceOperand::ScalarRegister(_) | SourceOperand::IntegerConstant(_) | SourceOperand::LiteralConstant(_));
            if let Some(s) = stride {
                if base_uniform && s > 0 && s <= 256 { map.insert(i.vdst as u32, s); }
            }
        }
    }
}
/// If this instruction defines an affine frame pointer, returns (vdst, stride).
pub fn frame_def(inst: &InstFormat) -> Option<(u32, u32)> {
    if let InstFormat::VOP3SD(i) = inst {
        if matches!(i.op, I::V_MAD_CO_U64_U32) {
            let stride = match (&i.src0, &i.src1) {
                (SourceOperand::VectorRegister(0), SourceOperand::IntegerConstant(c)) => Some(*c as u32),
                (SourceOperand::IntegerConstant(c), SourceOperand::VectorRegister(0)) => Some(*c as u32),
                (SourceOperand::VectorRegister(0), SourceOperand::LiteralConstant(c)) => Some(*c),
                (SourceOperand::LiteralConstant(c), SourceOperand::VectorRegister(0)) => Some(*c),
                _ => None,
            };
            let base_uniform = matches!(i.src2, SourceOperand::ScalarRegister(_) | SourceOperand::IntegerConstant(_) | SourceOperand::LiteralConstant(_));
            if let Some(s) = stride {
                if base_uniform && s > 0 && s <= 256 { return Some((i.vdst as u32, s)); }
            }
        }
    }
    None
}

pub fn frame_entry(prog: &ScalarProgram) -> BTreeMap<usize, std::collections::HashMap<u32, u32>> {
    use std::collections::HashMap;
    // All (reg -> stride) frame defs anywhere = the TOP element. A *must*-analysis
    // (meet = intersection) must initialise non-entry blocks to TOP, else the
    // intersection with an initially-empty loop back-edge permanently clears a
    // frame that is actually live through the loop (the def dominates it).
    let mut top: HashMap<u32, u32> = HashMap::new();
    for (_, block) in &prog.blocks {
        for inst in &block.body {
            if let Some((d, s)) = frame_def(inst) { top.insert(d, s); }
        }
    }
    let mut entry: BTreeMap<usize, HashMap<u32, u32>> =
        prog.blocks.keys().map(|&pc| (pc, if pc == prog.entry_pc { HashMap::new() } else { top.clone() })).collect();
    loop {
        let mut incoming: BTreeMap<usize, Option<HashMap<u32, u32>>> = BTreeMap::new();
        incoming.insert(prog.entry_pc, Some(HashMap::new())); // external edge: no frames at entry
        for (&pc, block) in &prog.blocks {
            let mut m = entry[&pc].clone();
            for inst in &block.body { frame_transfer(inst, &mut m); }
            for t in succs(block) {
                let e = incoming.entry(t).or_insert(None);
                *e = Some(match e.take() {
                    None => m.clone(),
                    Some(p) => p.into_iter().filter(|(k, v)| m.get(k) == Some(v)).collect(), // intersection
                });
            }
        }
        let mut changed = false;
        for (&pc, _) in &prog.blocks {
            if let Some(Some(v)) = incoming.get(&pc) {
                if &entry[&pc] != v { entry.insert(pc, v.clone()); changed = true; }
            }
        }
        if !changed { break; }
    }
    entry
}

/// True if this instruction's result is *intrinsically* divergent (reads the
/// per-lane scratch base), regardless of its register inputs.
pub fn uses_private(inst: &InstFormat) -> bool {
    use crate::rdna_instructions::InstFormat::*;
    let has = |o: &SourceOperand| matches!(o, SourceOperand::PrivateBase);
    match inst {
        VOP1(i) => has(&i.src0),
        VOP2(i) => has(&i.src0),
        VOP3(i) => has(&i.src0) || has(&i.src1) || has(&i.src2),
        VOP3SD(i) => has(&i.src0) || has(&i.src1) || has(&i.src2),
        VOPC(i) => has(&i.src0),
        _ => false,
    }
}

/// Flow-sensitive divergence transfer: a dest is divergent iff a source is
/// divergent here (or it reads scratch); a uniform redefinition *kills* prior
/// divergence (so a reg reused as uniform — e.g. an address — after being
/// divergent f64 data becomes uniform again).
pub fn div_transfer(inst: &InstFormat, d: &mut [u128; 2]) {
    let getb = |d: &[u128; 2], r: u32| (d[(r >> 7) as usize] >> (r & 127)) & 1 == 1;
    let div = uses_private(inst) || div_reads(inst).iter().any(|&r| getb(d, r));
    for w in vgpr_writes(inst) {
        let idx = (w >> 7) as usize;
        if div { d[idx] |= 1u128 << (w & 127); } else { d[idx] &= !(1u128 << (w & 127)); }
    }
    d[0] |= 1; // VGPR0 is always the per-lane id
}

/// Per-block-entry divergent VGPR set (forward, meet = union).
pub fn divergent_entry(prog: &ScalarProgram) -> BTreeMap<usize, [u128; 2]> {
    let mut seed = [0u128; 2];
    seed[0] |= 1;
    divergent_entry_with_seed(prog, seed)
}

/// Divergence analysis for a continuation fragment whose entry values were
/// defined by an earlier fragment. Callers may conservatively seed all live
/// VGPRs divergent; under-classifying one could incorrectly turn a gather into
/// a scalar broadcast.
pub fn divergent_entry_with_seed(
    prog: &ScalarProgram,
    seed: [u128; 2],
) -> BTreeMap<usize, [u128; 2]> {
    let mut entry: BTreeMap<usize, [u128; 2]> = prog.blocks.keys().map(|&pc| (pc, [0u128; 2])).collect();
    entry.insert(prog.entry_pc, seed);
    loop {
        let mut incoming: BTreeMap<usize, [u128; 2]> = BTreeMap::new();
        incoming.insert(prog.entry_pc, seed);
        for (&_pc, block) in &prog.blocks {
            let mut d = entry[&_pc];
            for inst in &block.body { div_transfer(inst, &mut d); }
            for t in succs(block) {
                let e = incoming.entry(t).or_insert([0u128; 2]);
                e[0] |= d[0]; e[1] |= d[1];
            }
        }
        let mut changed = false;
        for (&pc, _) in &prog.blocks {
            if let Some(v) = incoming.get(&pc) {
                if entry[&pc] != *v { entry.insert(pc, *v); changed = true; }
            }
        }
        if !changed { break; }
    }
    entry
}

fn succs(block: &ScalarBlock) -> Vec<usize> {
    match &block.term {
        Terminator::Return => vec![],
        Terminator::Jump(t) => vec![*t],
        Terminator::Branch { taken, fallthrough, .. } => vec![*taken, *fallthrough],
        Terminator::Barrier { resume } => vec![*resume],
    }
}

/// 256-bit VGPR set (regs 0..256).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Set([u128; 2]);
impl Set {
    fn empty() -> Set { Set([0, 0]) }
    fn get(&self, r: u32) -> bool { (self.0[(r >> 7) as usize] >> (r & 127)) & 1 == 1 }
    fn set(&mut self, r: u32) { self.0[(r >> 7) as usize] |= 1u128 << (r & 127); }
    fn clear(&mut self, r: u32) { self.0[(r >> 7) as usize] &= !(1u128 << (r & 127)); }
    fn union(&mut self, o: &Set) { self.0[0] |= o.0[0]; self.0[1] |= o.0[1]; }
}

fn apply_backward(inst: &InstFormat, live: &mut Set) {
    // live_before = (live_after \ defs) ∪ uses
    for d in vgpr_writes(inst) { live.clear(d); }
    for u in vgpr_reads(inst) { live.set(u); }
}

/// Per (block pc, instruction index): may this instruction's VGPR write skip
/// EXEC predication? True iff it writes ≥1 VGPR and none of its destinations
/// is live at a CFG reconvergence point or a caller-observed fragment exit.
/// Instructions that write no VGPR (compares -> mask, stores -> memory) are
/// never elided here — their masking is handled in emit
/// (`st_cmp`/`masked_scatter`).
/// Variant for a resume-capable fragment. `exit_live` is observed by the
/// scheduler after a synthetic Return, so writes to those VGPRs are not dead
/// even though the fragment CFG has no successor there.
pub fn analyze_with_exit_live(
    prog: &ScalarProgram,
    exit_live: &[u32],
) -> BTreeMap<usize, Vec<bool>> {
    let mut external = Set::empty();
    for &reg in exit_live { external.set(reg); }
    // Backward liveness fixpoint: live-out per block.
    let mut live_out: BTreeMap<usize, Set> = prog.blocks.keys().map(|&pc| (pc, Set::empty())).collect();
    loop {
        let mut changed = false;
        for (&pc, block) in &prog.blocks {
            let mut lo = if matches!(block.term, Terminator::Return) { external } else { Set::empty() };
            for s in succs(block) {
                // live-in[succ] = (live-out[succ] \ def[succ]) ∪ use[succ]
                if let Some(&succ_lo) = live_out.get(&s) {
                    let mut li = succ_lo;
                    if let Some(sb) = prog.blocks.get(&s) {
                        let mut tmp = li;
                        // recompute live-in of succ from its live-out
                        for inst in sb.body.iter().rev() {
                            apply_backward(inst, &mut tmp);
                        }
                        li = tmp;
                    }
                    lo.union(&li);
                }
            }
            if lo != live_out[&pc] {
                live_out.insert(pc, lo);
                changed = true;
            }
        }
        if !changed { break; }
    }

    // "State" registers = those live-in to a **reconvergence point** (a block with
    // ≥2 predecessors: an if/else merge, or a loop header = preheader+back-edge).
    // At such a point a previously-inactive lane rejoins and reads these registers,
    // so a write to one MUST stay predicated (else the lane reads clobbered state —
    // this is the de-recursed kernel's accumulator/RNG/ray state, e.g. reg 35).
    // Any register NOT live-in to any merge is never observed by a reactivated
    // lane, so its writes can omit lane-preserving selects. Garbage from
    // inactive lanes in such a temporary is confined: every escape is either a
    // state register (predicated) or memory (always EXEC-masked).
    let mut preds: BTreeMap<usize, u32> = prog.blocks.keys().map(|&pc| (pc, 0u32)).collect();
    for (_, block) in &prog.blocks {
        for s in succs(block) { *preds.entry(s).or_insert(0) += 1; }
    }
    let live_in = |pc: usize| -> Set {
        let mut li = live_out[&pc];
        for inst in prog.blocks[&pc].body.iter().rev() { apply_backward(inst, &mut li); }
        li
    };
    let mut state = external;
    for (&pc, _) in &prog.blocks {
        if preds[&pc] >= 2 { state.union(&live_in(pc)); }
    }

    let mut out: BTreeMap<usize, Vec<bool>> = BTreeMap::new();
    for (&pc, block) in &prog.blocks {
        let observed = state;
        let mut flags = vec![false; block.body.len()];
        for (idx, inst) in block.body.iter().enumerate() {
            let writes = vgpr_writes(inst);
            flags[idx] = !writes.is_empty() && writes.iter().all(|&d| !observed.get(d));
        }
        out.insert(pc, flags);
    }
    out
}
