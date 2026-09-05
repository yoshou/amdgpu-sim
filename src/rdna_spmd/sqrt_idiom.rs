//! Recognition of the existing normalise/sqrt/rescale idiom, without LLVM.

use crate::instructions::I;
use crate::rdna_instructions::{InstFormat, SourceOperand};
use super::freshness::vgpr_writes;
#[cfg(test)]
use super::ir::ScalarProgram;

fn normal_f64_pow2_exponent(op: &SourceOperand) -> bool {
    let value = match op {
        SourceOperand::IntegerConstant(value) => *value as u32 as i32,
        SourceOperand::LiteralConstant(value) => *value as i32,
        SourceOperand::FloatConstant(value) => (*value as f32).to_bits() as i32,
        _ => return false,
    };
    (-1022..=1023).contains(&value)
}

fn normal_pow2_cndmask_def(inst: &InstFormat) -> Option<u32> {
    let InstFormat::VOP3(i) = inst else { return None };
    (matches!(i.op, I::V_CNDMASK_B32)
        && i.abs == 0
        && i.neg == 0
        && normal_f64_pow2_exponent(&i.src0)
        && normal_f64_pow2_exponent(&i.src1))
        .then_some(i.vdst as u32)
}

/// The VGPRs an instruction between two steps of the idiom writes, or `None`
/// for a form the search will not step over. The ALU and scalar formats are
/// modelled by `vgpr_writes`; the memory ones are left out so that only
/// register effects have to be reasoned about here.
fn steppable_vgpr_writes(inst: &InstFormat) -> Option<Vec<u32>> {
    match inst {
        InstFormat::VOP1(_)
        | InstFormat::VOP2(_)
        | InstFormat::VOP3(_)
        | InstFormat::VOP3SD(_)
        | InstFormat::VOP3P(_)
        | InstFormat::VOPC(_)
        | InstFormat::VOPD(_)
        | InstFormat::SOP1(_)
        | InstFormat::SOP2(_)
        | InstFormat::SOPC(_)
        | InstFormat::SOPK(_)
        | InstFormat::SOPP(_) => Some(vgpr_writes(inst)),
        _ => None,
    }
}

/// The next instruction at or after `from` that `matches`, provided every
/// instruction before it leaves `live` alone. Anything else ends the search:
/// the idiom's steps have to reach each other through registers no one else
/// wrote.
fn find_step(
    body: &[InstFormat],
    from: usize,
    live: &[u32],
    matches: impl Fn(&InstFormat) -> bool,
) -> Option<usize> {
    for (offset, inst) in body[from..].iter().enumerate() {
        if matches(inst) {
            return Some(from + offset);
        }
        let written = steppable_vgpr_writes(inst)?;
        if written.iter().any(|reg| live.contains(reg)) {
            return None;
        }
    }
    None
}

/// Whether `body[from..until]` leaves `live` alone, which is what lets a step
/// found out of order still belong to the idiom.
fn keeps_live(body: &[InstFormat], from: usize, until: usize, live: &[u32]) -> bool {
    body[from..until].iter().all(|inst| {
        steppable_vgpr_writes(inst)
            .is_some_and(|written| !written.iter().any(|reg| live.contains(reg)))
    })
}

fn ldexp_f64_with_exponent(inst: &InstFormat, exponent: u32) -> bool {
    let InstFormat::VOP3(i) = inst else { return false };
    matches!(i.op, I::V_LDEXP_F64)
        && matches!(i.src1, SourceOperand::VectorRegister(r) if r as u32 == exponent)
}

/// Recognize the object-level normalization idiom
/// `scale -> sqrt -> refine -> classify -> rescale`.  Returning the two ends of
/// each idiom keeps the profitability decision local to the complete idiom
/// rather than expanding every individually-safe LDEXP in the object.
///
/// The steps need not be adjacent, nor in one fixed order: the compiler
/// interleaves an unrelated expansion with them and hoists the second exponent
/// and the classify around the square root. What has to hold is the dataflow —
/// every register the idiom carries reaches its next step unwritten.
fn normal_sqrt_ldexp_pairs(body: &[InstFormat]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for start in 0..body.len() {
        let Some(first_exp) = normal_pow2_cndmask_def(&body[start]) else {
            continue;
        };

        let Some(i_scale) = find_step(body, start + 1, &[first_exp], |inst| {
            ldexp_f64_with_exponent(inst, first_exp)
        }) else {
            continue;
        };
        let InstFormat::VOP3(first_scale) = &body[i_scale] else {
            continue;
        };
        let scaled = first_scale.vdst as u32;

        let Some(i_sqrt) = find_step(body, i_scale + 1, &[scaled, scaled + 1], |inst| {
            matches!(inst, InstFormat::VOP1(i)
                if matches!(i.op, I::V_SQRT_F64)
                    && matches!(i.src0, SourceOperand::VectorRegister(r) if r == first_scale.vdst))
        }) else {
            continue;
        };
        let InstFormat::VOP1(sqrt) = &body[i_sqrt] else {
            continue;
        };
        let root = sqrt.vdst as u32;

        // The rescale is the far end: an LDEXP of the root by an exponent from
        // a second cndmask of the same shape.
        let Some(i_rescale) = find_step(body, i_sqrt + 1, &[root, root + 1], |inst| {
            matches!(inst, InstFormat::VOP3(i)
                if matches!(i.op, I::V_LDEXP_F64)
                    && matches!(i.src0, SourceOperand::VectorRegister(r) if r == sqrt.vdst)
                    && matches!(i.src1, SourceOperand::VectorRegister(_)))
        }) else {
            continue;
        };
        let InstFormat::VOP3(second_scale) = &body[i_rescale] else {
            continue;
        };
        let SourceOperand::VectorRegister(second_exp) = second_scale.src1 else {
            continue;
        };
        let second_exp = second_exp as u32;

        // The second exponent may be computed anywhere before the rescale, as
        // long as it reaches it unwritten.
        let defines_second_exp = (start..i_rescale).rev().any(|at| {
            normal_pow2_cndmask_def(&body[at]) == Some(second_exp)
                && keeps_live(body, at + 1, i_rescale, &[second_exp])
        });
        if !defines_second_exp {
            continue;
        }

        // The classify reads the scaled value, so that value has to survive
        // from the scale to it.
        let has_class = (i_scale + 1..i_rescale).any(|at| {
            matches!(&body[at], InstFormat::VOP3(i)
                if matches!(i.op, I::V_CMP_CLASS_F64)
                    && matches!(i.src0, SourceOperand::VectorRegister(r) if r == first_scale.vdst))
                && keeps_live(body, i_scale + 1, at, &[scaled, scaled + 1])
        });
        if !has_class {
            continue;
        }

        pairs.push((i_scale, i_rescale));
    }
    pairs
}

/// The scale and rescale of every recognized idiom, as one flag per
/// instruction.
pub(super) fn normal_sqrt_ldexp_indices(body: &[InstFormat]) -> Vec<bool> {
    let mut fast = vec![false; body.len()];
    for (scale, rescale) in normal_sqrt_ldexp_pairs(body) {
        fast[scale] = true;
        fast[rescale] = true;
    }
    fast
}

/// The scale/sqrt/rescale idiom computes `sqrt(src)` the long way: it scales a
/// possibly-subnormal input up by an even power of two, takes the hardware
/// square root, then scales the result back down. The GPU needs that dance
/// because `V_SQRT_F64` is not correctly rounded over the whole range; x86
/// `sqrtpd` is, and both scalings are exact powers of two, so the rescaled
/// result is bit-identical to `sqrt(src)`.
///
/// Collapsing the idiom matters for more than instruction count: it puts a
/// compare, a select and two scales *in series* with the square root, and this
/// kernel is bound by dependency-chain latency rather than by throughput.
/// Measured on smallpt at W=16: -1.8% cycles, bit-identical image.
///
/// The intermediate scale and hardware square root are still emitted — the class
/// compare in the middle genuinely reads the scaled value, and anything else
/// reading the intermediates stays correct; they die if nothing does.
#[derive(Clone)]
pub(super) enum SqrtCollapse {
    /// Snapshot the pre-scale input: the scale usually writes its source
    /// register in place, so the value has to be read before it runs.
    Capture { site: usize, src: SourceOperand },
    /// Replace the trailing rescale with the square root of that snapshot.
    Rescale { site: usize, vdst: u8 },
}

pub(super) fn sqrt_collapse_sites(body: &[InstFormat]) -> Vec<Option<SqrtCollapse>> {
    let mut out: Vec<Option<SqrtCollapse>> = vec![None; body.len()];
    for (scale, rescale) in normal_sqrt_ldexp_pairs(body) {
        let InstFormat::VOP3(first_scale) = &body[scale] else { continue };
        let InstFormat::VOP3(second_scale) = &body[rescale] else { continue };
        // Source modifiers would change the value being rooted.
        if first_scale.abs != 0
            || first_scale.neg != 0
            || second_scale.abs != 0
            || second_scale.neg != 0
        {
            continue;
        }
        // The scale names the site, so a collapse cannot pair the ends of two
        // different idioms.
        out[scale] = Some(SqrtCollapse::Capture { site: scale, src: first_scale.src0.clone() });
        out[rescale] = Some(SqrtCollapse::Rescale { site: scale, vdst: second_scale.vdst });
    }
    out
}

#[cfg(test)]
pub(super) fn normal_sqrt_ldexp_sites(program: &ScalarProgram) -> Vec<(usize, usize)> {
    program
        .blocks
        .iter()
        .flat_map(|(&pc, block)| {
            normal_sqrt_ldexp_indices(&block.body)
                .into_iter()
                .enumerate()
                .filter_map(move |(index, fast)| fast.then_some((pc, index)))
        })
        .collect()
}

#[cfg(test)]
mod normal_sqrt_tests {
    use super::*;
    use crate::rdna_instructions::{VOP1, VOP3};
    const VCC: u32 = 106;

    fn vop3(
        op: I,
        vdst: u8,
        src0: SourceOperand,
        src1: SourceOperand,
        src2: SourceOperand,
    ) -> InstFormat {
        InstFormat::VOP3(VOP3 {
            vdst,
            abs: 0,
            opsel: 0,
            cm: 0,
            op,
            src0,
            src1,
            src2,
            omod: 0,
            neg: 0,
        })
    }

    fn normalization_body(first_exp: u32) -> Vec<InstFormat> {
        vec![
            vop3(
                I::V_CNDMASK_B32,
                10,
                SourceOperand::IntegerConstant(0),
                SourceOperand::LiteralConstant(first_exp),
                SourceOperand::ScalarRegister(VCC as u8),
            ),
            vop3(
                I::V_LDEXP_F64,
                8,
                SourceOperand::VectorRegister(8),
                SourceOperand::VectorRegister(10),
                SourceOperand::ScalarRegister(0),
            ),
            InstFormat::VOP1(VOP1 {
                src0: SourceOperand::VectorRegister(8),
                op: I::V_SQRT_F64,
                vdst: 10,
            }),
            vop3(
                I::V_CNDMASK_B32,
                12,
                SourceOperand::IntegerConstant(0),
                SourceOperand::LiteralConstant((-128i32) as u32),
                SourceOperand::ScalarRegister(VCC as u8),
            ),
            vop3(
                I::V_CMP_CLASS_F64,
                VCC as u8,
                SourceOperand::VectorRegister(8),
                SourceOperand::LiteralConstant(0x260),
                SourceOperand::ScalarRegister(0),
            ),
            vop3(
                I::V_LDEXP_F64,
                10,
                SourceOperand::VectorRegister(10),
                SourceOperand::VectorRegister(12),
                SourceOperand::ScalarRegister(0),
            ),
        ]
    }

    #[test]
    fn recognizes_normal_power_of_two_sqrt_idiom() {
        assert_eq!(
            normal_sqrt_ldexp_indices(&normalization_body(256)),
            [false, true, false, false, false, true]
        );
    }

    #[test]
    fn rejects_non_normal_power_of_two_exponent() {
        assert!(normal_sqrt_ldexp_indices(&normalization_body(1024))
            .iter()
            .all(|fast| !fast));
    }

    #[test]
    fn rejects_mismatched_exponent_def_use() {
        let mut body = normalization_body(256);
        let InstFormat::VOP3(scale) = &mut body[1] else { unreachable!() };
        scale.src1 = SourceOperand::VectorRegister(11);
        assert!(normal_sqrt_ldexp_indices(&body).iter().all(|fast| !fast));
    }

    #[test]
    fn rejects_non_constant_exponent_choice() {
        let mut body = normalization_body(256);
        let InstFormat::VOP3(select) = &mut body[0] else { unreachable!() };
        select.src0 = SourceOperand::ScalarRegister(4);
        assert!(normal_sqrt_ldexp_indices(&body).iter().all(|fast| !fast));
    }

    #[test]
    fn rejects_modified_or_incomplete_idiom() {
        let mut modified = normalization_body(256);
        let InstFormat::VOP3(select) = &mut modified[0] else { unreachable!() };
        select.abs = 1;
        assert!(normal_sqrt_ldexp_indices(&modified).iter().all(|fast| !fast));

        let mut incomplete = normalization_body(256);
        incomplete.remove(4);
        assert!(normal_sqrt_ldexp_indices(&incomplete).iter().all(|fast| !fast));
    }
}
