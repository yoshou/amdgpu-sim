//! Existing normalise/sqrt/rescale recognition over SSA definitions.
use super::ir::typed::ValueId;
use super::lift::Input;
use super::analysis::state::Site;

pub(super) enum Shape {
    Other,
    Exponent(ValueId),
    Scale { input: Option<[ValueId; 2]>, exponent: Option<ValueId>, output: [ValueId; 2],
        source: Input, destination: u32, unmodified: bool, exponent_unmodified: bool },
    Sqrt { input: [ValueId; 2], output: [ValueId; 2] },
    Class([ValueId; 2]),
}
pub(super) struct Policy {
    pub shape: Shape,
    pub steppable: bool,
    pub replaced: Vec<ValueId>,
}
fn find_step(body: &[Site], from: usize, live: &[ValueId], matches: impl Fn(&Shape) -> bool) -> Option<usize> {
    for (index, site) in body.iter().enumerate().skip(from) {
        if matches(&site.sqrt.shape) { return Some(index); }
        if !site.sqrt.steppable || site.sqrt.replaced.iter().any(|v| live.contains(v)) { return None; }
    }
    None
}
fn keeps_live(body: &[Site], from: usize, until: usize, live: &[ValueId]) -> bool {
    body[from..until].iter().all(|s| s.sqrt.steppable && !s.sqrt.replaced.iter().any(|v| live.contains(v)))
}
fn pairs(body: &[Site]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for start in 0..body.len() {
        let Shape::Exponent(first_exp) = body[start].sqrt.shape else { continue; };
        let Some(scale) = find_step(body, start+1, &[first_exp], |s|
            matches!(s, Shape::Scale { exponent: Some(v), .. } if *v == first_exp)) else { continue; };
        let Shape::Scale { output: scaled, .. } = body[scale].sqrt.shape else { unreachable!() };
        let Some(sqrt) = find_step(body, scale+1, &scaled, |s|
            matches!(s, Shape::Sqrt { input, .. } if *input == scaled)) else { continue; };
        let Shape::Sqrt { output: root, .. } = body[sqrt].sqrt.shape else { unreachable!() };
        let Some(rescale) = find_step(body, sqrt+1, &root, |s|
            matches!(s, Shape::Scale { input: Some(input), exponent: Some(_), .. } if *input == root)) else { continue; };
        let Shape::Scale { exponent: Some(second_exp), .. } = body[rescale].sqrt.shape else { unreachable!() };
        if !(start..rescale).rev().any(|at| matches!(body[at].sqrt.shape, Shape::Exponent(v) if v == second_exp)
            && keeps_live(body, at+1, rescale, &[second_exp])) { continue; }
        if !(scale+1..rescale).any(|at| matches!(body[at].sqrt.shape, Shape::Class(input) if input == scaled)
            && keeps_live(body, scale+1, at, &scaled)) { continue; }
        pairs.push((scale, rescale));
    }
    pairs
}
#[derive(Clone)]
pub(super) enum SqrtCollapse {
    Capture { site: usize, src: Input },
    Rescale { site: usize, vdst: u32 },
}
pub(super) fn analyze(body: &[Site]) -> (Vec<bool>, Vec<Option<SqrtCollapse>>) {
    let mut normal = vec![false; body.len()];
    let mut collapse = vec![None; body.len()];
    for (scale, rescale) in pairs(body) {
        normal[scale] = true; normal[rescale] = true;
        let Shape::Scale { source, unmodified: true, .. } = &body[scale].sqrt.shape else { continue; };
        let Shape::Scale { destination, unmodified: true, .. } = body[rescale].sqrt.shape else { continue; };
        collapse[scale] = Some(SqrtCollapse::Capture { site: scale, src: source.clone() });
        collapse[rescale] = Some(SqrtCollapse::Rescale { site: scale, vdst: destination });
    }
    (normal, collapse)
}
#[cfg(test)]
fn normal_sqrt_ldexp_indices(body: &[crate::rdna_instructions::InstFormat]) -> Vec<bool> {
    use std::collections::BTreeMap;
    use super::{ir::{ScalarProgram, ScalarBlock, Terminator}, lift};
    let program = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock {
        pc: 0, body: body.to_vec(), term: Terminator::Return,
    })]) };
    let registry = std::sync::Arc::new(super::dialect::DialectRegistry::rdna4());
    let instructions: Vec<_> = body.iter().map(|i| lift::instruction_with_registry(i, &registry)).collect();
    let f = lift::function::Function::lift(registry, &program, &BTreeMap::from([(0, instructions.iter().collect())]));
    analyze(&f.state.sites[&0]).0
}
#[cfg(test)]
pub(super) fn normal_sqrt_ldexp_sites(program: &super::ir::ScalarProgram) -> Vec<(usize, usize)> {
    program.blocks.iter().flat_map(|(&pc, block)| normal_sqrt_ldexp_indices(&block.body)
        .into_iter().enumerate().filter_map(move |(index, yes)| yes.then_some((pc, index)))).collect()
}

#[cfg(test)]
mod normal_sqrt_tests {
    use super::*;
    use crate::instructions::I;
    use crate::rdna_instructions::{InstFormat, SourceOperand, VOP1, VOP3};
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
