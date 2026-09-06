//! Image descriptors are explicit SSA operands, including the sampler. DMASK
//! chooses consecutive result words; it is not a hidden provider argument.
use super::*;
use std::convert::TryInto;

pub(super) fn instruction(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering<'static>> {
    let InstFormat::VSAMPLE(i) = inst else { return None; };
    if !matches!(i.op, I::IMAGE_SAMPLE_LZ) { return None; }
    let mut inputs = (0..8).map(|k| input(SourceOperand::ScalarRegister((i.rsrc + k).try_into().unwrap()), Ty::I32)).collect::<Vec<_>>();
    inputs.extend((0..4).map(|k| input(SourceOperand::ScalarRegister((i.samp + k).try_into().unwrap()), Ty::I32)));
    inputs.push(input(SourceOperand::VectorRegister(i.vaddr0), Ty::F32));
    inputs.push(input(SourceOperand::VectorRegister(i.vaddr1), Ty::F32));
    inputs.push(input(SourceOperand::ScalarRegister(126), Ty::I1));
    let mut b = Builder::new(registry, inputs);
    // Target signatures are architectural values, without an implicit EXEC
    // argument. A zero descriptor selects constant zero and performs no read.
    // Suppress every inactive operand, including undefined coordinates, before
    // invoking this effect; selecting only its result would still read memory.
    let zero_word = b.k(Ty::I32, 0); let zero_float = b.k(Ty::F32, 0);
    let safe: [ValueId; 14] = std::array::from_fn(|k| {
        let (ty, zero) = if k < 12 { (Ty::I32, zero_word) } else { (Ty::F32, zero_float) };
        b.push(ty, Op::Select(ValueId(14), ValueId(k), zero))
    });
    let target = super::super::dialect::rdna4::image_sample(registry);
    let unrm = b.k(Ty::I1, (i.unrm != 0) as u64);
    let mut results = vec![];
    for component in 0..4 {
        if i.dmask & (1 << component) == 0 { continue; }
        let component = b.k(Ty::I32, component);
        let mut args = std::array::from_fn(|k| if k < 12 { safe[k] } else { ValueId(0) });
        args[12] = component; args[13] = unrm; args[14] = safe[12]; args[15] = safe[13];
        let value = b.target_one(target, Arguments::Sixteen(args));
        results.push((Output::Vgpr(i.vdata as u32 + results.len() as u32, Ty::I32), value));
    }
    Some(b.finish_many(false, results))
}
