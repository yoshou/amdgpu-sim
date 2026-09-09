use super::*;
use crate::rdna_spmd::{dialect::Arguments, ir::ValueId, jit};
use std::sync::Arc;

// ISA Tables 60–62: R8 resource layout, point sampler addressing and numeric
// conversion. This reference uses scalar host arithmetic and array indexing;
// it does not invoke the simulator's image helper or LLVM lowering.
fn reference(texels: &[u8], format: u32, mode: u32, unrm: bool, u: f32, v: f32, selector: u32) -> u32 {
    let one = if matches!(format, 1 | 2) { 1f32.to_bits() } else { 1 };
    if selector < 2 { return if selector == 0 { 0 } else { one }; }
    let index = |value: f32, extent: i32| {
        let n = (if unrm { value } else { value * extent as f32 }).floor() as i32;
        let mode = if unrm && mode < 2 { mode + 2 } else { mode };
        let mirrored = if n < 0 { !n } else { n };
        match mode {
            0 => Some(n.rem_euclid(extent)),
            1 => { let x = n.rem_euclid(2 * extent); Some(if x < extent { x } else { 2 * extent - 1 - x }) },
            2 | 4 => Some(n.clamp(0, extent - 1)),
            3 | 5 => Some(mirrored.clamp(0, extent - 1)),
            6 => (0..extent).contains(&n).then_some(n),
            7 => (0..extent).contains(&mirrored).then_some(mirrored),
            _ => unreachable!(),
        }
    };
    let (Some(x), Some(y)) = (index(u, 5), index(v, 3)) else { return one; };
    let raw = texels[y as usize * 128 + x as usize];
    match format {
        1 => (raw as f32 / 255.).to_bits(),
        2 => ((raw as i8).max(-127) as f32 / 127.).to_bits(),
        5 => raw as u32, 6 => raw as i8 as i32 as u32,
        _ => unreachable!(),
    }
}
fn field(words: &mut [u32], bit: usize, count: usize, value: u32) {
    for k in 0..count { words[(bit + k) / 32] |= ((value >> k) & 1) << ((bit + k) % 32); }
}

#[test]
fn image_ir_requires_constant_component_and_unique_effect_provenance() {
    use crate::rdna_spmd::ir::{*, Op};
    let registry = DialectRegistry::rdna4(); let op = image_sample(&registry);
    let mut types = registry.operation(op).unwrap().inputs.to_vec(); types.push(Ty::I32);
    let args = Arguments::Sixteen(std::array::from_fn(ValueId));
    let f = Func { entry: BlockId(0), types: types.clone(), blocks: std::collections::BTreeMap::from([(BlockId(0), Block {
        params: types[..16].iter().enumerate().filter(|(k, _)| *k != 12).map(|(k, &ty)| (ValueId(k), ty)).collect(),
        insts: vec![Inst::Core { value: ValueId(12), ty: Ty::I32, op: Op::Const(Ty::I32, 3) },
            Inst::Target { provenance: Some(7), op, args, outputs: vec![(ValueId(16), Ty::I32)] }], term: Term::Ret(vec![]),
    })]) };
    f.clone().verify_with(&registry).unwrap();
    for case in 0..4 {
        let mut bad = f.clone();
        let block = bad.blocks.get_mut(&BlockId(0)).unwrap();
        match case {
            0 => { block.insts.remove(0); block.params.push((ValueId(12), Ty::I32)); },
            1 => if let Inst::Core { op, .. } = &mut block.insts[0] { *op = Op::Const(Ty::I32, 4); },
            2 => if let Inst::Target { provenance, .. } = &mut block.insts[1] { *provenance = None; },
            3 => { bad.types.push(Ty::I32); block.insts.push(Inst::Target { provenance: Some(7), op, args, outputs: vec![(ValueId(17), Ty::I32)] }); },
            _ => unreachable!(),
        }
        assert!(bad.verify_with(&registry).is_err(), "case={}", case);
    }
}

#[test]
fn native_sampler_matches_reference_with_lane_distinct_descriptors_and_coordinates() {
    let registry = Arc::new(DialectRegistry::rdna4());
    let target = image_sample(&registry);
    let mut storage = vec![0u8; 384 + 255];
    let offset = (256 - storage.as_ptr() as usize % 256) % 256;
    let texels = &mut storage[offset..offset + 384];
    for y in 0..3 { for x in 0..5 { texels[y * 128 + x] = (128 + x * 13 + y * 11) as u8; } }
    let address = texels.as_ptr() as u64;
    let coordinates = [-2.0f32, -0.4, 0., 0.2, 0.99, 1.4, 3.2, f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
    let cases = [1, 2, 5, 6].iter().flat_map(|&format| (0..8).flat_map(move |mode|
        [false, true].iter().copied().flat_map(move |unrm| (0..coordinates.len()).map(move |j| (format, mode, unrm, j))))).collect::<Vec<_>>();
    for width in [0, 1, 2, 4, 8, 16] { for component in 0..4 {
        unsafe {
            let module = jit::Module::new("image_reference"); let b = module.builder; let n = b"\0".as_ptr().cast();
            let pointer = LLVMPointerTypeInContext(module.ctx, 0);
            let ft = LLVMFunctionType(LLVMVoidTypeInContext(module.ctx), [pointer, pointer].as_mut_ptr(), 2, 0);
            let f = LLVMAddFunction(module.module, b"kernel\0".as_ptr().cast(), ft);
            LLVMPositionBuilderAtEnd(b, LLVMAppendBasicBlockInContext(module.ctx, f, n));
            let e = Emitter::new(b, (width != 0).then_some(width), registry.clone());
            let w = width.max(1) as usize;
            let mut values = vec![];
            for index in 0..16 {
                let offset = LLVMConstInt(LLVMInt32TypeInContext(module.ctx), (index * w) as u64, 0);
                let ptr = LLVMBuildGEP2(b, LLVMInt32TypeInContext(module.ctx), LLVMGetParam(f, 0), [offset].as_mut_ptr(), 1, n);
                let value = LLVMBuildLoad2(b, e.ty(Ty::I32), ptr, n); LLVMSetAlignment(value, 4);
                values.push(match index {
                    12 => e.constant(Ty::I32, component),
                    13 => LLVMBuildICmp(b, LLVMIntNE, value, e.constant(Ty::I32, 0), n),
                    14 | 15 => LLVMBuildBitCast(b, value, e.ty(Ty::F32), n),
                    _ => value,
                });
            }
            let result = e.target(target, Arguments::Sixteen(std::array::from_fn(ValueId)), &values)[0];
            let store = LLVMBuildStore(b, result, LLVMGetParam(f, 1)); LLVMSetAlignment(store, 4); LLVMBuildRetVoid(b);
            let code = module.finish(if width == 0 { jit::Mode::Scalar } else { jit::Mode::Packet });
            let run: unsafe extern "C" fn(*const u32, *mut u32) = std::mem::transmute(code.address() as usize);
            let mut input = vec![0u32; 16 * w]; let mut output = vec![0u32; w];
            for start in (0..cases.len()).step_by(w) {
                let mut expected = vec![];
                for lane in 0..w {
                    let (format, mode, unrm, j) = cases[(start + lane) % cases.len()];
                    let mut desc = [0u32; 16]; desc[0] = (address >> 8) as u32; desc[1] = (address >> 40) as u32;
                    field(&mut desc, 49, 8, format); field(&mut desc, 62, 16, 4); field(&mut desc, 78, 16, 2);
                    let selector = [0, 1, 4, 7][(j + component as usize) % 4];
                    field(&mut desc, 96 + component as usize * 3, 3, selector);
                    field(&mut desc[8..12], 0, 3, mode); field(&mut desc[8..12], 3, 3, mode);
                    field(&mut desc[8..12], 126, 2, 2);
                    desc[13] = unrm as u32; desc[14] = coordinates[j].to_bits(); desc[15] = coordinates[(j + 3) % coordinates.len()].to_bits();
                    // Constant components may use a null descriptor: neither
                    // scalar nor packet code may perform a speculative read.
                    if selector < 2 { desc[0] = 0; desc[1] &= !255; }
                    for index in 0..16 { input[index * w + lane] = desc[index]; }
                    expected.push(reference(texels, format, mode, unrm, f32::from_bits(desc[14]), f32::from_bits(desc[15]), selector));
                }
                run(input.as_ptr(), output.as_mut_ptr());
                assert_eq!(output, expected, "width={width} component={component} start={start}");
            }
        }
    } }
}
