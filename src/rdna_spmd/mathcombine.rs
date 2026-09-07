//! Cross-block sqrt expansion recognition using typed SSA liveness.
use std::collections::{BTreeMap, BTreeSet};
use super::analysis::{state::Site, rewrite::{Operand, Kind}};
use super::lift::function::LiftedFunction;

fn live_out(f: &LiftedFunction) -> BTreeMap<usize, BTreeSet<u32>> {
    let mut dependencies = vec![Vec::new(); f.ir.types.len()];
    for (&pc, block) in &f.ir.blocks {
        for edge in block.term.edges() {
            for &(slot, index) in &f.state.vector_parameters {
                let parameter = f.ir.blocks[&edge.dst].params[index].0;
                dependencies[parameter.0].push(f.state.outgoing[&pc.0][&slot]);
            }
        }
    }
    let mut pending: Vec<_> = f.state.sites.values().flatten().flat_map(|s| s.math_reads.iter().map(|&(_, v)| v)).collect();
    let mut live = vec![false; f.ir.types.len()];
    while let Some(value) = pending.pop() {
        if !live[value.0] { live[value.0] = true; pending.extend(&dependencies[value.0]); }
    }
    f.ir.blocks.iter().map(|(&pc, block)| {
        let mut slots = BTreeSet::new();
        for edge in block.term.edges() {
            for &(slot, index) in &f.state.vector_parameters {
                if live[f.ir.blocks[&edge.dst].params[index].0.0] { slots.insert(slot); }
            }
        }
        (pc.0, slots)
    }).collect()
}
fn vpair(op: &Operand) -> Option<u32> { op.vector() }
fn as_fma(site: &Site) -> Option<(u32, &Operand, &Operand, &Operand, u8)> {
    let m = site.rewrite.math.as_ref()?;
    (m.kind == Kind::Fma && m.cross_sqrt).then_some((m.destination, &m.inputs[0], &m.inputs[1], &m.inputs[2], m.neg))
}
fn is_const_half(op: &Operand) -> bool { op.float(0.5) }
/// Match the rsq+Newton sqrt expansion starting at body index `a` (a V_RSQ_F64).
/// Returns (final_fma_idx, [indices to remove], rd, x) if it matches and all
/// temporaries are dead by `live`.
fn match_sqrt(body: &[Site], a: usize, live_out: &BTreeSet<u32>) -> Option<(usize, Vec<usize>, u32, u32)> {
    let m = body[a].rewrite.math.as_ref()?;
    if m.kind != Kind::Rsq || !m.cross_sqrt { return None; }
    let (rd, x) = (m.destination, m.inputs[0].vector()?);
    let g = |k: usize| body.get(a+k);
    let m = g(1)?.rewrite.math.as_ref()?;
    if m.kind != Kind::Mul || !m.cross_sqrt || m.inputs[0].vector() != Some(x) || m.inputs[1].vector() != Some(rd) { return None; }
    let (i1, anf) = (a+1, m.destination);
    let av = anf;
    // i2: rD = 0.5 * rD
    let m = g(2)?.rewrite.math.as_ref()?;
    if m.kind != Kind::Mul || !m.cross_sqrt || !m.inputs[0].float(0.5)
        || m.inputs[1].vector() != Some(rd) || m.destination != rd { return None; }
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
        if body[j].writes.iter().any(|&(w, _)| w == x || w == x+1) {
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
            if body[k].math_reads.iter().any(|&(r, _)| r == t || r == t+1) {
                return None;
            }
            if body[k].writes.iter().any(|&(w, _)| w == t) {
                break; // overwritten -> dead from here
            }
            k += 1;
        }
    }
    Some((i9, removed, rd, x))
}


pub(super) fn analyze(f: &LiftedFunction) -> BTreeMap<usize, (BTreeSet<usize>, Vec<(usize, u32, u32)>)> {
    let live = live_out(f);
    let mut edits = BTreeMap::new();
    for (&pc, body) in &f.state.sites {
        let mut removed = BTreeSet::new();
        let mut rewrites = Vec::new();
        let mut index = 0;
        while index < body.len() {
            if let Some((last, delete, destination, input)) = match_sqrt(body, index, &live[&pc]) {
                removed.extend(delete); rewrites.push((last, destination, input)); index = last+1;
            } else { index += 1; }
        }
        if !rewrites.is_empty() { edits.insert(pc, (removed, rewrites)); }
    }
    edits
}
