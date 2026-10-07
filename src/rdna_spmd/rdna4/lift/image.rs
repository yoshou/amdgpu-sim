use super::*;
use std::convert::TryInto;

pub fn instruction(inst: &InstFormat, registry: &DialectRegistry) -> Option<Lowering> {
    if let InstFormat::VIMAGE(i) = inst {
        if matches!(i.op, I::IMAGE_BVH8_INTERSECT_RAY) {
            let mut inputs = vec![
                input(SourceOperand::ScalarRegister(i.rsrc as u8), Ty::I32),
                input(SourceOperand::ScalarRegister((i.rsrc + 1) as u8), Ty::I32),
                input(SourceOperand::VectorRegister(i.vaddr0), Ty::I64),
                input(SourceOperand::VectorRegister(i.vaddr1), Ty::I32),
                input(SourceOperand::VectorRegister(i.vaddr1 + 1), Ty::I32),
            ];
            for reg in [i.vaddr2, i.vaddr3] {
                for k in 0..3 {
                    inputs.push(input(SourceOperand::VectorRegister(reg + k), Ty::I32));
                }
            }
            inputs.push(input(SourceOperand::VectorRegister(i.vaddr4), Ty::I32));
            inputs.push(Input {
                source: InputSource::ExecPredicate,
                ty: Ty::I1,
            });
            let mut b = Builder::new(registry, inputs);
            let values = b.target(
                crate::rdna_spmd::rdna4::dialect::bvh8(registry),
                Arguments::Thirteen(std::array::from_fn(ValueId)),
            );
            return Some(
                b.finish_many(
                    false,
                    values
                        .into_iter()
                        .enumerate()
                        .map(|(k, v)| (Output::Vgpr(i.vdata as u32 + k as u32, Ty::I32), v))
                        .collect(),
                ),
            );
        }
        if !matches!(i.op, I::IMAGE_BVH64_INTERSECT_RAY) {
            return None;
        }
        let mut inputs = vec![
            input(SourceOperand::ScalarRegister(i.rsrc as u8), Ty::I32),
            input(SourceOperand::ScalarRegister((i.rsrc + 1) as u8), Ty::I32),
            input(SourceOperand::VectorRegister(i.vaddr0), Ty::I64),
            input(SourceOperand::VectorRegister(i.vaddr1), Ty::I32),
        ];
        for reg in [i.vaddr2, i.vaddr3, i.vaddr4] {
            for k in 0..3 {
                inputs.push(input(SourceOperand::VectorRegister(reg + k), Ty::I32));
            }
        }
        inputs.push(Input {
            source: InputSource::ExecPredicate,
            ty: Ty::I1,
        });
        let mut b = Builder::new(registry, inputs);
        let values = b.target(
            crate::rdna_spmd::rdna4::dialect::bvh(registry),
            Arguments::Fourteen(std::array::from_fn(ValueId)),
        );
        return Some(
            b.finish_many(
                false,
                values
                    .into_iter()
                    .enumerate()
                    .map(|(k, v)| (Output::Vgpr(i.vdata as u32 + k as u32, Ty::I32), v))
                    .collect(),
            ),
        );
    }
    let InstFormat::VSAMPLE(i) = inst else {
        return None;
    };
    if !matches!(i.op, I::IMAGE_SAMPLE_LZ) {
        return None;
    }
    let rsrc: [u8; 8] = std::array::from_fn(|k| (i.rsrc + k as u16).try_into().unwrap());
    let samp: [u8; 4] = std::array::from_fn(|k| (i.samp + k as u16).try_into().unwrap());
    let target = crate::rdna_spmd::rdna4::dialect::image_sample(registry);
    Some(sample(registry, target, rsrc, samp, [i.vaddr0, i.vaddr1], i.vdata, i.dmask, i.unrm != 0))
}

pub fn sample(
    registry: &DialectRegistry,
    target: TargetOp,
    rsrc: [u8; 8],
    samp: [u8; 4],
    vaddr: [u8; 2],
    vdata: u8,
    dmask: u8,
    unrm: bool,
) -> Lowering {
    let mut inputs = rsrc
        .iter()
        .chain(&samp)
        .map(|&r| input(SourceOperand::ScalarRegister(r), Ty::I32))
        .collect::<Vec<_>>();
    inputs.push(input(SourceOperand::VectorRegister(vaddr[0]), Ty::F32));
    inputs.push(input(SourceOperand::VectorRegister(vaddr[1]), Ty::F32));
    inputs.push(input(SourceOperand::ScalarRegister(126), Ty::I1));
    let mut b = Builder::new(registry, inputs);

    let zero_word = b.k(Ty::I32, 0);
    let zero_float = b.k(Ty::F32, 0);
    let safe: [ValueId; 14] = std::array::from_fn(|k| {
        let (ty, zero) = if k < 12 {
            (Ty::I32, zero_word)
        } else {
            (Ty::F32, zero_float)
        };
        b.push(ty, Op::Select(ValueId(14), ValueId(k), zero))
    });
    let unrm = b.k(Ty::I1, unrm as u64);
    let mut results = vec![];
    for component in 0..4 {
        if dmask & (1 << component) == 0 {
            continue;
        }
        let component = b.k(Ty::I32, component);
        let mut args = std::array::from_fn(|k| if k < 12 { safe[k] } else { ValueId(0) });
        args[12] = component;
        args[13] = unrm;
        args[14] = safe[12];
        args[15] = safe[13];
        let value = b.target_one(target, Arguments::Sixteen(args));
        results.push((
            Output::Vgpr(vdata as u32 + results.len() as u32, Ty::I32),
            value,
        ));
    }
    b.finish_many(false, results)
}
