//! Existing local sqrt, division and dead-definition rewrites over SSA uses.
use super::analysis::rewrite::{Observation, Operand, Kind, Definitions};
const VGPR_BASE: u32 = 512;
fn vgpr_pair(op: &Operand) -> Option<u32> { op.vector() }
fn as_rsq_f64(inst: &Observation) -> Option<(u32, u32)> {
    let m = inst.math.as_ref()?;
    (m.kind == Kind::Rsq && m.local_sqrt).then(|| Some((m.destination, m.inputs[0].vector()?))).flatten()
}
fn as_mul_f64(inst: &Observation) -> Option<(u32, Operand, Operand)> {
    let m = inst.math.as_ref()?;
    (m.kind == Kind::Mul && m.local_sqrt).then_some((m.destination, m.inputs[0], m.inputs[1]))
}
fn as_fma_f64(inst: &Observation) -> Option<(u32, Operand, Operand, Operand, u8)> {
    let m = inst.math.as_ref()?;
    (m.kind == Kind::Fma && m.local_sqrt).then_some((m.destination, m.inputs[0], m.inputs[1], m.inputs[2], m.neg))
}
// First index after `start` whose instruction satisfies `pred`. The chain is a
// strict data dependency, so the matching instruction is uniquely pinned by its
// register operands; interleaved scheduling fillers are skipped.
fn find_forward(
    insts: &[Observation],
    start: usize,
    end: usize,
    pred: impl Fn(&Observation) -> bool,
) -> Option<usize> {
    (start + 1..end).find(|&j| pred(&insts[j]))
}

fn is_mul_xy(inst: &Observation, x: u32, y: u32) -> Option<u32> {
    let (dst, s0, s1) = as_mul_f64(inst)?;
    let (a, b) = (vgpr_pair(&s0), vgpr_pair(&s1));
    if (a == Some(x) && b == Some(y)) || (a == Some(y) && b == Some(x)) {
        Some(dst)
    } else {
        None
    }
}

fn is_mul_half(inst: &Observation, r: u32) -> Option<u32> {
    let (dst, s0, s1) = as_mul_f64(inst)?;
    let half = |o: &Operand| o.float(0.5);
    if (half(&s0) && vgpr_pair(&s1) == Some(r)) || (half(&s1) && vgpr_pair(&s0) == Some(r)) {
        Some(dst)
    } else {
        None
    }
}

// Matches the rsq + Newton-Raphson f64 sqrt expansion seeded at the
// V_RSQ_F64 `anchor`:
//
//   r  = rsq(X)             [anchor, dst rD]
//   A  = X * rD
//   rD = 0.5 * rD
//   B  = fma(-rD, A, 0.5)
//   A  = fma(A, B, A)
//   rD = fma(rD, B, rD)
//   B  = fma(-A, A, X)
//   A  = fma(B, rD, A)
//   B  = fma(-A, A, X)
//   rD = fma(B, rD, A)      [final, = sqrt(X)]
//
// rD ends holding sqrt(X) so the whole chain collapses to V_SQRT_F64(rD, X):
// returns the final FMA index (to rewrite) and the other chain indices (to
// remove). X is never written by the chain, so it is still live to seed the
// sqrt; this is checked, as is that nothing outside the chain observes the
// temporaries.
struct SqrtMatch {
    final_idx: usize,
    removed: Vec<usize>,
    rd: u32,
    x: u32,
}

fn match_sqrt_f64(insts: &[Observation], effects: &[Observation], anchor: usize) -> Option<SqrtMatch> {
    let (rd, x) = as_rsq_f64(&insts[anchor])?;
    let n = insts.len();

    // step1: A = X * rD
    let i1 = find_forward(insts, anchor, n, |i| is_mul_xy(i, x, rd).is_some())?;
    let a = is_mul_xy(&insts[i1], x, rd)?;

    // step2: rD = 0.5 * rD
    let i2 = find_forward(insts, i1, n, |i| is_mul_half(i, rd) == Some(rd))?;

    // step3: B = fma(-rD, A, 0.5)
    let i3 = find_forward(insts, i2, n, |i| {
        as_fma_f64(i).map_or(false, |(_, s0, s1, s2, neg)| {
            neg == 1
                && vgpr_pair(&s0) == Some(rd)
                && vgpr_pair(&s1) == Some(a)
                && s2.float(0.5)
        })
    })?;
    let b = as_fma_f64(&insts[i3])?.0;

    // step4: A = fma(A, B, A)
    let i4 = find_forward(insts, i3, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == a
                && neg == 0
                && vgpr_pair(&s0) == Some(a)
                && vgpr_pair(&s1) == Some(b)
                && vgpr_pair(&s2) == Some(a)
        })
    })?;

    // step5: rD = fma(rD, B, rD)
    let i5 = find_forward(insts, i4, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == rd
                && neg == 0
                && vgpr_pair(&s0) == Some(rd)
                && vgpr_pair(&s1) == Some(b)
                && vgpr_pair(&s2) == Some(rd)
        })
    })?;

    // step6: B = fma(-A, A, X)
    let i6 = find_forward(insts, i5, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == b
                && neg == 1
                && vgpr_pair(&s0) == Some(a)
                && vgpr_pair(&s1) == Some(a)
                && vgpr_pair(&s2) == Some(x)
        })
    })?;

    // step7: A = fma(B, rD, A)
    let i7 = find_forward(insts, i6, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == a
                && neg == 0
                && vgpr_pair(&s0) == Some(b)
                && vgpr_pair(&s1) == Some(rd)
                && vgpr_pair(&s2) == Some(a)
        })
    })?;

    // step8: B = fma(-A, A, X)
    let i8 = find_forward(insts, i7, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == b
                && neg == 1
                && vgpr_pair(&s0) == Some(a)
                && vgpr_pair(&s1) == Some(a)
                && vgpr_pair(&s2) == Some(x)
        })
    })?;

    // step9: rD = fma(B, rD, A)  -> final sqrt(X)
    let i9 = find_forward(insts, i8, n, |i| {
        as_fma_f64(i).map_or(false, |(d, s0, s1, s2, neg)| {
            d == rd
                && neg == 0
                && vgpr_pair(&s0) == Some(b)
                && vgpr_pair(&s1) == Some(rd)
                && vgpr_pair(&s2) == Some(a)
        })
    })?;

    let removed = vec![anchor, i1, i2, i3, i4, i5, i6, i7, i8];

    // X must remain unwritten through the final FMA so the rewritten sqrt
    // reads the same input the chain consumed.
    for j in (anchor + 1)..i9 {
        if removed.contains(&j) {
            continue;
        }
        let e = &effects[j];
        if !e.known {
            return None;
        }
        if e.definitions.iter().any(|&(r, _)| r == VGPR_BASE+x || r == VGPR_BASE+x+1) {
            return None;
        }
    }

    // The chain's temporaries (A, B and the intermediate rD writes) must not be
    // observed outside the matched set before being overwritten. The final FMA
    // is rewritten to read only X, so reads by it no longer count.
    let matched: Vec<usize> = removed.iter().copied().chain(std::iter::once(i9)).collect();
    for &i in &removed {
        for &(reg, value) in &effects[i].definitions {
            let mut killed = false;
            for j in (i + 1)..n {
                if matched.contains(&j) {
                    if j == i9 {
                        // After rewrite the final FMA neither reads nor writes
                        // the temporaries (only X -> rD), except rD which it
                        // still defines.
                        if reg == VGPR_BASE + rd || reg == VGPR_BASE + rd + 1 {
                            killed = true;
                            break;
                        }
                        continue;
                    }
                    if effects[j].replaced.contains(&value) {
                        killed = true;
                        break;
                    }
                    continue;
                }
                let ej = &effects[j];
                if !ej.known {
                    return None;
                }
                if ej.reads.contains(&value) {
                    return None;
                }
                if ej.replaced.contains(&value) {
                    killed = true;
                    break;
                }
            }
            // Stricter than the div matcher: a temporary surviving to the block
            // end may be read by a successor block (e.g. the chain's sqrt/rsqrt
            // estimates), so only remove a definition that is provably
            // overwritten within this block.
            if !killed {
                return None;
            }
        }
    }

    Some(SqrtMatch {
        final_idx: i9,
        removed,
        rd,
        x,
    })
}

fn operand_eq(a: &Operand, b: &Operand) -> bool { a.same_encoding(*b) }
fn is_const_one(op: &Operand) -> bool { op.float(1.0) }
fn is_fma_f64(inst: &Observation) -> Option<(u32, Operand, Operand, Operand, u8, u8)> {
    let m = inst.math.as_ref()?;
    (m.kind == Kind::Fma && m.local_div).then_some((m.destination, m.inputs[0], m.inputs[1], m.inputs[2], m.neg, m.abs))
}
fn is_mul_f64(inst: &Observation) -> Option<(Operand, Operand)> {
    let m = inst.math.as_ref()?;
    (m.kind == Kind::Mul && m.local_div).then_some((m.inputs[0], m.inputs[1]))
}
fn is_rcp_f64(inst: &Observation, src_reg: u32) -> bool {
    inst.math.as_ref().is_some_and(|m| m.kind == Kind::Rcp && m.local_div && m.inputs[0].vector() == Some(src_reg))
}
fn is_div_scale_f64(inst: &Observation) -> Option<(u32, Operand, Operand, Operand)> {
    let m = inst.math.as_ref()?;
    (m.kind == Kind::DivScale && m.local_div).then(|| Some((m.scalar_destination?, m.inputs[0], m.inputs[1], m.inputs[2]))).flatten()
}
// Matches the compiler's f64 division expansion feeding a V_DIV_FIXUP_F64 at
// `anchor`:
//
//   a   = v_div_scale_f64 (den, den, num)
//   r0  = v_rcp_f64 a
//   t   = v_fma_f64 -a, r, 1.0            (repeated with r = fma(r, t, r))
//   n_s = v_div_scale_f64 vcc, (num, den, num)
//   q   = v_mul_f64 n_s, r
//   f   = v_fma_f64 -a, q, n_s
//   e   = v_div_fmas_f64 f, r, q
//   dst = v_div_fixup_f64 e, den, num
//
// Returns the instruction indices of the chain (everything but the anchor)
// for removal. The expansion's temporaries are dead past the anchor by
// construction; the only assumption made beyond in-block dataflow is that
// their stale values are not read by later blocks, which holds for
// compiler-generated code because the expansion is emitted as a unit.
fn match_div_f64(
    insts: &[Observation],
    effects: &[Observation],
    anchor: usize,
) -> Option<Vec<usize>> {
    let m = insts[anchor].math.as_ref()?;
    if m.kind != Kind::DivFixup || !m.local_div { return None; }
    let (e_reg, den, num) = (m.inputs[0].vector()?, m.inputs[1], m.inputs[2]);
    let definitions = Definitions::new(effects);

    let mut matched = Vec::new();

    // e = div_fmas(f, r, q)
    let i_fmas = definitions.before(effects, anchor, e_reg)?;
    let m = insts[i_fmas].math.as_ref()?;
    if m.kind != Kind::DivFmas || !m.local_div { return None; }
    let (f_reg, r_reg, q_reg) = (m.inputs[0].vector()?, m.inputs[1].vector()?, m.inputs[2].vector()?);
    matched.push(i_fmas);

    // q = n_s * r
    let i_mul = definitions.before(effects, i_fmas, q_reg)?;
    let (m0, m1) = is_mul_f64(&insts[i_mul])?;
    let ns_reg = if vgpr_pair(&m1) == Some(r_reg) {
        vgpr_pair(&m0)?
    } else if vgpr_pair(&m0) == Some(r_reg) {
        vgpr_pair(&m1)?
    } else {
        return None;
    };
    matched.push(i_mul);

    // f = fma(-a, q, n_s)
    let i_f = definitions.before(effects, i_fmas, f_reg)?;
    let (_, f0, f1, f2, neg, abs) = is_fma_f64(&insts[i_f])?;
    if neg != 1 || abs != 0 {
        return None;
    }
    let a_reg = vgpr_pair(&f0)?;
    if vgpr_pair(&f1) != Some(q_reg) || vgpr_pair(&f2) != Some(ns_reg) {
        return None;
    }
    matched.push(i_f);

    // n_s = div_scale(num, den, num)
    let i_dsn = definitions.before(effects, i_mul.min(i_f), ns_reg)?;
    let (_, d0, d1, d2) = is_div_scale_f64(&insts[i_dsn])?;
    if !operand_eq(&d0, &num) || !operand_eq(&d1, &den) || !operand_eq(&d2, &num) {
        return None;
    }
    matched.push(i_dsn);

    // Newton-Raphson refinement: r = fma(r', t, r'), t = fma(-a, r', 1.0),
    // bottoming out at r = rcp(a).
    let mut i_r = definitions.before(effects, i_mul, r_reg)?;
    let mut found_rcp = false;
    for _ in 0..8 {
        if is_rcp_f64(&insts[i_r], a_reg) {
            matched.push(i_r);
            found_rcp = true;
            break;
        }
        let (_, r0, r1, r2, neg, abs) = is_fma_f64(&insts[i_r])?;
        if neg != 0 || abs != 0 {
            return None;
        }
        let r_prev = vgpr_pair(&r0)?;
        if vgpr_pair(&r2) != Some(r_prev) {
            return None;
        }
        let t_reg = vgpr_pair(&r1)?;
        matched.push(i_r);

        let i_t = definitions.before(effects, i_r, t_reg)?;
        let (_, t0, t1, t2, tneg, tabs) = is_fma_f64(&insts[i_t])?;
        if tneg != 1 || tabs != 0 {
            return None;
        }
        if vgpr_pair(&t0) != Some(a_reg) || vgpr_pair(&t1) != Some(r_prev) || !is_const_one(&t2) {
            return None;
        }
        matched.push(i_t);

        i_r = definitions.before(effects, i_t, r_prev)?;
    }
    if !found_rcp {
        return None;
    }

    // a = div_scale(den, den, num)
    let earliest = *matched.iter().min().unwrap();
    let i_dsa = definitions.before(effects, earliest, a_reg)?;
    let (_, a0, a1, a2) = is_div_scale_f64(&insts[i_dsa])?;
    if !operand_eq(&a0, &den) || !operand_eq(&a1, &den) || !operand_eq(&a2, &num) {
        return None;
    }
    matched.push(i_dsa);

    // The chain's results must not be observable outside the matched set:
    // every register a matched instruction writes may only be read by other
    // matched instructions before being fully rewritten. VGPR temporaries may
    // reach the block end (dead past the anchor by the expansion contract);
    // an SGPR written by something other than div_scale must be rewritten
    // within the block, since branches and later blocks may read it.
    for &i in &matched {
        for &(reg, value) in &effects[i].definitions {
            let mut killed = false;
            for j in (i + 1)..insts.len() {
                let ej = &effects[j];
                if matched.contains(&j) {
                    if ej.replaced.contains(&value) {
                        killed = true;
                        break;
                    }
                    continue;
                }
                if !ej.known {
                    return None;
                }
                // The anchor is the consumer this pass exists for: this
                // backend computes the quotient from the original operands,
                // so its read of the chain's result is not one.
                let quotient =
                    j == anchor && (reg == VGPR_BASE + e_reg || reg == VGPR_BASE + e_reg + 1);
                if ej.reads.contains(&value) && !quotient {
                    return None;
                }
                if ej.replaced.contains(&value) {
                    killed = true;
                    break;
                }
            }
            // div_scale's SGPR flag is overwritten by the emitter with a
            // constant rather than the architected scale flags, so no correct
            // reader depends on it past the consuming div_fmas; any genuine
            // outside reader is already rejected by the loop above.
            let div_scale_sdst = reg < VGPR_BASE
                && insts[i].math.as_ref().is_some_and(|m| m.kind == Kind::DivScale);
            if !killed && reg < VGPR_BASE && !div_scale_sdst {
                return None;
            }
        }
    }

    Some(matched)
}


pub(super) fn divisions(body: &[Observation]) -> Vec<bool> {
    let mut remove = vec![false; body.len()];
    for anchor in 0..body.len() {
        if let Some(matched) = match_div_f64(body, body, anchor) {
            if matched.iter().all(|&i| !remove[i]) { for i in matched { remove[i] = true; } }
        }
    }
    remove
}
pub(super) fn square_roots(body: &[Observation]) -> (Vec<bool>, Vec<(usize, u32, u32)>) {
    let mut remove = vec![false; body.len()];
    let mut rewrites = Vec::new();
    for anchor in 0..body.len() {
        if let Some(m) = match_sqrt_f64(body, body, anchor) {
            if m.removed.iter().all(|&i| !remove[i]) && !remove[m.final_idx] {
                for i in m.removed { remove[i] = true; }
                rewrites.push((m.final_idx, m.rd, m.x));
            }
        }
    }
    (remove, rewrites)
}
/// One round retains the existing all-at-once removal order and live final
/// source position. Iteration belongs to input preparation, after edits apply.
pub(super) fn dead(body: &[Observation]) -> Vec<bool> {
    let mut remove = vec![false; body.len()];
    for i in 0..body.len().saturating_sub(1) {
        let site = &body[i];
        if !site.removable || site.definitions.is_empty() { continue; }
        remove[i] = site.definitions.iter().all(|&(_, value)| {
            for next in &body[i+1..] {
                if !next.known || next.reads.contains(&value) { return false; }
                if next.replaced.contains(&value) { return true; }
            }
            false
        });
    }
    remove
}
