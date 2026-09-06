//! Division macro operands and explicit VCC input for FMAS.
use super::*;

pub(super) fn instruction(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering<'static>> {
    if let InstFormat::VOP3SD(i) = inst {
        if !matches!(i.op, I::V_DIV_SCALE_F32 | I::V_DIV_SCALE_F64) { return None; }
        let target = crate::rdna_spmd::dialect::rdna4::division(registry, i.op)?;
        let ty = registry.operation(target).unwrap().inputs[0];
        let mut b = Builder::new(registry, vec![input(i.src0, ty), input(i.src1, ty), input(i.src2, ty)]);
        let a = b.float_mod(ty, ValueId(0), 0, i.neg, 0);
        let c = b.float_mod(ty, ValueId(1), 0, i.neg, 1);
        let d = b.float_mod(ty, ValueId(2), 0, i.neg, 2);
        let values = b.target(target, Arguments::Ternary([a, c, d]));
        let result = b.output_mod(ty, values[0], i.cm, i.omod);
        let mut outputs = vec![(Output::Vgpr(i.vdst as u32, ty), result)];
        if i.sdst != 124 { outputs.push((Output::Mask(i.sdst as u32), values[1])); }
        return Some(b.finish_many(false, outputs));
    }
    let InstFormat::VOP3(i) = inst else { return None; };
    let target = crate::rdna_spmd::dialect::rdna4::division(registry, i.op)?;
    let spec = registry.operation(target).unwrap();
    let ty = spec.inputs[0];
    let mut inputs = vec![input(i.src0, ty), input(i.src1, ty), input(i.src2, ty)];
    let fmas = matches!(i.op, I::V_DIV_FMAS_F32 | I::V_DIV_FMAS_F64);
    if fmas { inputs.push(input(SourceOperand::ScalarRegister(106), Ty::I1)); }
    let mut b = Builder::new(registry, inputs);
    let a = b.float_mod(ty, ValueId(0), i.abs, i.neg, 0);
    let c = b.float_mod(ty, ValueId(1), i.abs, i.neg, 1);
    let d = b.float_mod(ty, ValueId(2), i.abs, i.neg, 2);
    let args = if fmas { Arguments::Quaternary([a, c, d, ValueId(3)]) } else { Arguments::Ternary([a, c, d]) };
    let value = b.target_one(target, args);
    let value = b.output_mod(ty, value, i.cm, i.omod);
    Some(b.finish(Output::Vgpr(i.vdst as u32, ty), value))
}
