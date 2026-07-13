//! Scalar-only math-idiom combine: collapse the f64 `rsq + Newton-Raphson`
//! square-root expansion to a single `V_SQRT_F64`.
//!
//! The shared `rdna_translator::combine` recognizes the same chain but rejects
//! it when a Newton temporary is live at the block end (a conservative
//! single-block liveness check). Our scalar emit lowers `V_SQRT_F64` to the
//! exact `llvm.sqrt`, and the chain's temporaries are in fact dead downstream,
//! so with a real cross-block liveness analysis the whole Newton chain (~8 FMAs
//! per sqrt) can be deleted. Measured: ~366 such chains, ~20% of dynamic f64
//! arithmetic.

use std::collections::{BTreeMap, BTreeSet};

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand, VOP1};

use super::freshness::vgpr_writes;
use super::ir::{ScalarProgram, Terminator};

fn is_f64_op(op: I) -> bool {
    // Ops whose vector source operands are read as f64 *pairs* (r, r+1). i32 ops
    // (V_CNDMASK/V_MOV/V_AND/...) read a single register. Over-including a pair
    // for an f64 op's occasional i32 sub-operand (e.g. V_LDEXP src1) is safe;
    // under-counting an i32 op's single read as a pair was the bug.
    let s = format!("{:?}", op);
    s.contains("F64") && !matches!(op, I::V_CVT_F64_U32 | I::V_CVT_F64_I32)
}

/// VGPR registers read by an instruction (source operands), at correct width.
fn vgpr_reads(inst: &InstFormat) -> Vec<u32> {
    let mut r = Vec::new();
    match inst {
        InstFormat::VOP1(i) => {
            if let SourceOperand::VectorRegister(x) = i.src0 {
                r.push(x as u32);
                if is_f64_op(i.op) { r.push(x as u32 + 1); }
            }
        }
        InstFormat::VOP2(i) => {
            let pair = is_f64_op(i.op);
            if let SourceOperand::VectorRegister(x) = i.src0 { r.push(x as u32); if pair { r.push(x as u32 + 1); } }
            r.push(i.vsrc1 as u32);
            if pair { r.push(i.vsrc1 as u32 + 1); }
        }
        InstFormat::VOP3(i) => {
            let pair = is_f64_op(i.op);
            for o in [&i.src0, &i.src1, &i.src2] {
                if let SourceOperand::VectorRegister(x) = o { r.push(*x as u32); if pair { r.push(*x as u32 + 1); } }
            }
        }
        InstFormat::VOP3SD(i) => {
            let pair = is_f64_op(i.op);
            for o in [&i.src0, &i.src1, &i.src2] {
                if let SourceOperand::VectorRegister(x) = o { r.push(*x as u32); if pair { r.push(*x as u32 + 1); } }
            }
        }
        InstFormat::VOP3P(i) => {
            // Mirror the VOP3P write side in `freshness::vgpr_writes`. Reads are the
            // base VGPR of each source (V_FMA_MIXLO_F16; the wave-wide WMMA is lifted
            // out before this runs, so its multi-register spans never reach here).
            for o in [&i.src0, &i.src1, &i.src2] {
                if let SourceOperand::VectorRegister(x) = o { r.push(*x as u32); }
            }
        }
        InstFormat::VOPC(i) => {
            let pair = is_f64_op(i.op);
            if let SourceOperand::VectorRegister(x) = i.src0 { r.push(x as u32); if pair { r.push(x as u32 + 1); } }
            r.push(i.vsrc1 as u32);
            if pair { r.push(i.vsrc1 as u32 + 1); }
        }
        InstFormat::VOPD(i) => {
            // VOPD packs two 32-bit ops; sources are single registers.
            if let SourceOperand::VectorRegister(x) = i.src0x { r.push(x as u32); }
            if let SourceOperand::VectorRegister(x) = i.src0y { r.push(x as u32); }
            r.push(i.vsrc1x as u32);
            r.push(i.vsrc1y as u32);
        }
        InstFormat::VGLOBAL(i) => {
            r.push(i.vaddr as u32);
            r.push(i.vaddr as u32 + 1); // 64-bit address
            for k in 0..4 { r.push(i.vsrc as u32 + k); } // store data (up to B128)
        }
        _ => {}
    }
    r
}

/// Backward liveness: VGPRs live on exit from each block.
fn live_out(prog: &ScalarProgram) -> BTreeMap<usize, BTreeSet<u32>> {
    let mut live_in: BTreeMap<usize, BTreeSet<u32>> =
        prog.blocks.keys().map(|&pc| (pc, BTreeSet::new())).collect();
    loop {
        let mut changed = false;
        for (&pc, block) in &prog.blocks {
            let succs: Vec<usize> = match &block.term {
                Terminator::Return => vec![],
                Terminator::Jump(t) => vec![*t],
                Terminator::Branch { taken, fallthrough, .. } => vec![*taken, *fallthrough],
                Terminator::Barrier { resume } => vec![*resume],
            };
            let mut out = BTreeSet::new();
            for s in succs {
                if let Some(li) = live_in.get(&s) {
                    out.extend(li.iter().copied());
                }
            }
            // Transfer backward through the body.
            let mut cur = out;
            for inst in block.body.iter().rev() {
                for w in vgpr_writes(inst) {
                    cur.remove(&w);
                }
                for rd in vgpr_reads(inst) {
                    cur.insert(rd);
                }
            }
            if live_in[&pc] != cur {
                live_in.insert(pc, cur);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // live_out[B] = union of live_in[successors]
    let mut out = BTreeMap::new();
    for (&pc, block) in &prog.blocks {
        let succs: Vec<usize> = match &block.term {
            Terminator::Return => vec![],
            Terminator::Jump(t) => vec![*t],
            Terminator::Branch { taken, fallthrough, .. } => vec![*taken, *fallthrough],
            Terminator::Barrier { resume } => vec![*resume],
        };
        let mut s = BTreeSet::new();
        for su in succs {
            if let Some(li) = live_in.get(&su) {
                s.extend(li.iter().copied());
            }
        }
        out.insert(pc, s);
    }
    out
}

fn vpair(o: &SourceOperand) -> Option<u32> {
    if let SourceOperand::VectorRegister(r) = o { Some(*r as u32) } else { None }
}

/// (dst, src0, src1, src2, neg) for a V_FMA_F64.
fn as_fma(inst: &InstFormat) -> Option<(u32, &SourceOperand, &SourceOperand, &SourceOperand, u8)> {
    if let InstFormat::VOP3(i) = inst {
        if matches!(i.op, I::V_FMA_F64) {
            return Some((i.vdst as u32, &i.src0, &i.src1, &i.src2, i.neg));
        }
    }
    None
}

fn is_const_half(o: &SourceOperand) -> bool {
    matches!(o, SourceOperand::FloatConstant(v) if v.to_bits() == 0.5f64.to_bits())
}

/// Match the rsq+Newton sqrt expansion starting at body index `a` (a V_RSQ_F64).
/// Returns (final_fma_idx, [indices to remove], rd, x) if it matches and all
/// temporaries are dead by `live`.
fn match_sqrt(body: &[InstFormat], a: usize, live_out: &BTreeSet<u32>) -> Option<(usize, Vec<usize>, u32, u32)> {
    let (rd, x) = match &body[a] {
        InstFormat::VOP1(VOP1 { op: I::V_RSQ_F64, vdst, src0 }) => (*vdst as u32, vpair(src0)?),
        _ => return None,
    };
    // The chain is the contiguous run after the anchor; require the exact ops.
    let g = |k: usize| body.get(a + k);
    // i1: A = X * rD  (VOP2 mul)
    let (i1, anf) = match g(1)? {
        InstFormat::VOP2(i) if matches!(i.op, I::V_MUL_F64) && vpair(&i.src0) == Some(x) && i.vsrc1 as u32 == rd => (a + 1, i.vdst as u32),
        _ => return None,
    };
    let av = anf;
    // i2: rD = 0.5 * rD
    match g(2)? {
        InstFormat::VOP2(i) if matches!(i.op, I::V_MUL_F64) && is_const_half(&i.src0) && i.vsrc1 as u32 == rd && i.vdst as u32 == rd => {}
        _ => return None,
    }
    let i2 = a + 2;
    // i3: B = fma(-rD, A, 0.5)
    let (i3, b) = match as_fma(g(3)?) {
        Some((d, s0, s1, s2, neg)) if neg == 1 && vpair(s0) == Some(rd) && vpair(s1) == Some(av) && is_const_half(s2) => (a + 3, d),
        _ => return None,
    };
    // i4: A = fma(A, B, A)
    match as_fma(g(4)?) { Some((d, s0, s1, s2, neg)) if neg == 0 && d == av && vpair(s0) == Some(av) && vpair(s1) == Some(b) && vpair(s2) == Some(av) => {}, _ => return None }
    // i5: rD = fma(rD, B, rD)
    match as_fma(g(5)?) { Some((d, s0, s1, s2, neg)) if neg == 0 && d == rd && vpair(s0) == Some(rd) && vpair(s1) == Some(b) && vpair(s2) == Some(rd) => {}, _ => return None }
    // i6: B = fma(-A, A, X)
    match as_fma(g(6)?) { Some((d, s0, s1, s2, neg)) if neg == 1 && d == b && vpair(s0) == Some(av) && vpair(s1) == Some(av) && vpair(s2) == Some(x) => {}, _ => return None }
    // i7: A = fma(B, rD, A)
    match as_fma(g(7)?) { Some((d, s0, s1, s2, neg)) if neg == 0 && d == av && vpair(s0) == Some(b) && vpair(s1) == Some(rd) && vpair(s2) == Some(av) => {}, _ => return None }
    // i8: B = fma(-A, A, X)
    match as_fma(g(8)?) { Some((d, s0, s1, s2, neg)) if neg == 1 && d == b && vpair(s0) == Some(av) && vpair(s1) == Some(av) && vpair(s2) == Some(x) => {}, _ => return None }
    // i9: rD = fma(B, rD, A)  -> final, = sqrt(X)
    let i9 = match as_fma(g(9)?) { Some((d, s0, s1, s2, neg)) if neg == 0 && d == rd && vpair(s0) == Some(b) && vpair(s1) == Some(rd) && vpair(s2) == Some(av) => a + 9, _ => return None };
    let _ = (i1, i2, i3);

    let removed: Vec<usize> = (a..i9).collect();

    // X must not be written between the anchor and the final FMA.
    for j in (a + 1)..i9 {
        if vgpr_writes(&body[j]).iter().any(|&w| w == x || w == x + 1) {
            return None;
        }
    }
    // Temporaries A (av) and B (b) must be dead after i9: not read in the block
    // after i9 (until overwritten) and not live-out. rD is the result (kept).
    for &t in &[av, b] {
        // not live-out
        if live_out.contains(&t) || live_out.contains(&(t + 1)) {
            return None;
        }
        // not read after i9 before being overwritten
        let mut k = i9 + 1;
        while k < body.len() {
            if vgpr_reads(&body[k]).iter().any(|&r| r == t || r == t + 1) {
                return None;
            }
            if vgpr_writes(&body[k]).iter().any(|&w| w == t) {
                break; // overwritten -> dead from here
            }
            k += 1;
        }
    }
    Some((i9, removed, rd, x))
}

/// Rewrite the program in place, collapsing matched sqrt chains.
pub fn fold_sqrt(prog: &mut ScalarProgram) -> usize {
    let live = live_out(prog);
    let mut folded = 0;
    let pcs: Vec<usize> = prog.blocks.keys().copied().collect();
    for pc in pcs {
        let lo = live.get(&pc).cloned().unwrap_or_default();
        let body = &prog.blocks.get(&pc).unwrap().body;
        let mut rewrites: Vec<(usize, u32, u32)> = Vec::new(); // (final_idx, rd, x)
        let mut to_remove: BTreeSet<usize> = BTreeSet::new();
        let mut a = 0;
        while a < body.len() {
            if matches!(&body[a], InstFormat::VOP1(VOP1 { op: I::V_RSQ_F64, .. })) {
                if let Some((fin, removed, rd, x)) = match_sqrt(body, a, &lo) {
                    rewrites.push((fin, rd, x));
                    for r in &removed {
                        to_remove.insert(*r);
                    }
                    a = fin + 1;
                    folded += 1;
                    continue;
                }
            }
            a += 1;
        }
        if rewrites.is_empty() {
            continue;
        }
        // Apply: replace each final FMA with V_SQRT_F64(rd, X); drop removed.
        let block = prog.blocks.get_mut(&pc).unwrap();
        let mut new_body = Vec::with_capacity(block.body.len());
        for (idx, inst) in block.body.iter().enumerate() {
            if to_remove.contains(&idx) {
                continue;
            }
            if let Some(&(_, rd, x)) = rewrites.iter().find(|(f, _, _)| *f == idx) {
                new_body.push(InstFormat::VOP1(VOP1 {
                    op: I::V_SQRT_F64,
                    vdst: rd as u8,
                    src0: SourceOperand::VectorRegister(x as u8),
                }));
            } else {
                new_body.push(inst.clone());
            }
        }
        block.body = new_body;
    }
    folded
}
