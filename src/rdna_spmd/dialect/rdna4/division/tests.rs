use super::*;
use crate::rdna_spmd::{dialect::Arguments, ir::typed::ValueId, jit};
use std::sync::Arc;

struct Case { inputs: [u64; 4], result: u64, flag: u32 }

// Invoke actual generated providers with lane-distinct runtime arguments.
// Store both results independently, so dropped/reordered target outputs fail.
fn check(opcode: I, cases: &[Case]) {
    let registry = Arc::new(DialectRegistry::rdna4());
    let target = super::super::division(&registry, opcode).unwrap();
    let spec = registry.operation(target).unwrap();
    let ty = spec.inputs[0];
    for width in [0, 1, 2, 4, 8, 16] {
        unsafe {
            let module = jit::Module::new("division_capture");
            let b = module.builder; let ctx = module.ctx; let n = b"\0".as_ptr().cast();
            let pointer = LLVMPointerTypeInContext(ctx, 0);
            let ft = LLVMFunctionType(LLVMVoidTypeInContext(ctx), [pointer; 6].as_mut_ptr(), 6, 0);
            let f = LLVMAddFunction(module.module, b"kernel\0".as_ptr().cast(), ft);
            LLVMPositionBuilderAtEnd(b, LLVMAppendBasicBlockInContext(ctx, f, n));
            let e = Emitter::new(b, (width != 0).then_some(width), registry.clone());
            let mut values = Vec::new();
            for (index, &input_ty) in spec.inputs.iter().enumerate() {
                let storage_ty = if input_ty == Ty::I1 { Ty::I32 } else { input_ty };
                let value = LLVMBuildLoad2(b, e.ty(storage_ty), LLVMGetParam(f, index as u32), n);
                LLVMSetAlignment(value, 4);
                values.push(if input_ty == Ty::I1 { LLVMBuildTrunc(b, value, e.ty(Ty::I1), n) } else { value });
            }
            let args = if values.len() == 4 { Arguments::Quaternary([ValueId(0), ValueId(1), ValueId(2), ValueId(3)]) }
                else { Arguments::Ternary([ValueId(0), ValueId(1), ValueId(2)]) };
            assert_eq!(registry.result_types(target, args, spec.inputs).unwrap(), spec.outputs);
            let results = e.target(target, args, &values);
            for (index, &value) in results.iter().enumerate() {
                let value = if index == 1 { LLVMBuildZExt(b, value, e.ty(Ty::I32), n) } else { value };
                LLVMSetAlignment(LLVMBuildStore(b, value, LLVMGetParam(f, 4 + index as u32)), 4);
            }
            LLVMBuildRetVoid(b);
            let code = module.finish(if width == 0 { jit::Mode::Scalar } else { jit::Mode::Packet });
            let run: unsafe extern "C" fn(*const u32, *const u32, *const u32, *const u32, *mut u32, *mut u32) =
                std::mem::transmute(code.address() as usize);
            let lanes = width.max(1) as usize; let words = ty.bits() as usize / 32;
            let mut inputs: [Vec<u32>; 4] = std::array::from_fn(|_| vec![0; lanes * words]);
            let mut result = vec![0u32; lanes * words]; let mut flags = vec![0u32; lanes];
            for start in (0..cases.len()).step_by(lanes) {
                for lane in 0..lanes {
                    let case = &cases[(start + lane) % cases.len()];
                    for input in 0..4 {
                        let stride = if input == 3 { 1 } else { words };
                        inputs[input][lane * stride] = case.inputs[input] as u32;
                        if stride == 2 { inputs[input][lane * stride + 1] = (case.inputs[input] >> 32) as u32; }
                    }
                }
                run(inputs[0].as_ptr(), inputs[1].as_ptr(), inputs[2].as_ptr(), inputs[3].as_ptr(), result.as_mut_ptr(), flags.as_mut_ptr());
                for lane in 0..lanes {
                    let case = &cases[(start + lane) % cases.len()];
                    let bits = result[lane * words] as u64 | if words == 2 { (result[lane * words + 1] as u64) << 32 } else { 0 };
                    assert_eq!(bits, case.result, "{opcode:?} W{width} case {}", (start + lane) % cases.len());
                    if results.len() == 2 { assert_eq!(flags[lane], case.flag, "{opcode:?} W{width} flag"); }
                }
            }
        }
    }
}

#[test]
fn scale_matches_captured_values_and_flags_at_all_widths() {
    // Literal hardware captures, including the zero-denominator arbitration.
    for (op, one, rows) in [
        (I::V_DIV_SCALE_F32, 0x3f80_0000, vec![
            (0x4000_0000, 0x4000_0000, 0), (0, 0xffc0_0000, 1),
            (0x8000_0000, 0xffc0_0000, 1), (0x7f80_0000, 0x7f80_0000, 1),
            (0xff80_0000, 0xff80_0000, 1), (0x7fc0_0000, 0x7fc0_0000, 1),
            (1, 0x1500_0000, 1), (0x0080_0000, 0x2080_0000, 1),
            (0x7f7f_ffff, 0x5f7f_ffff, 1), (0x3f00_0000, 0x3f00_0000, 0)]),
        (I::V_DIV_SCALE_F64, 0x3ff0_0000_0000_0000, vec![
            (0x4000_0000_0000_0000, 0x4000_0000_0000_0000, 0), (0, 0xfff8_0000_0000_0000, 1),
            (0x8000_0000_0000_0000, 0xfff8_0000_0000_0000, 1),
            (0x7ff0_0000_0000_0000, 0x7ff0_0000_0000_0000, 1),
            (0xfff0_0000_0000_0000, 0xfff0_0000_0000_0000, 1),
            (0x7ff8_0000_0000_0000, 0x7ff8_0000_0000_0000, 1),
            (1, 0x04d0_0000_0000_0000, 1), (0x0010_0000_0000_0000, 0x0810_0000_0000_0000, 1),
            (0x7fef_ffff_ffff_ffff, 0x77ef_ffff_ffff_ffff, 1)])] {
        let mut cases = Vec::new();
        for (den, result, flag) in rows {
            let ty = if matches!(op, I::V_DIV_SCALE_F32) { Ty::F32 } else { Ty::F64 };
            assert_eq!(reference_scale(ty, [den, den, one]), (result, flag != 0));
            cases.push(Case { inputs: [den, den, one, 0], result, flag });
            let result = if den == 0 || den == 0x8000_0000 || den == 0x8000_0000_0000_0000 { result } else { one };
            cases.push(Case { inputs: [one, den, one, 0], result, flag });
        }
        check(op, &cases);
    }
}

#[test]
fn scale_reference_covers_thresholds_and_both_operands() {
    for (op, ty, inputs) in [
        (I::V_DIV_SCALE_F32, Ty::F32, vec![0, 0x8000_0000, 1, 0x007f_ffff, 0x0080_0000,
            0x0b80_0000, 0x0c00_0000, 0x0f80_0000, 0x1000_0000, 0x3f80_0000,
            0x4000_0000, 0xbf80_0000, 0x6f00_0000, 0x6f80_0000, 0x7e00_0000,
            0x7e80_0000, 0x7f7f_ffff, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000]),
        (I::V_DIV_SCALE_F64, Ty::F64, vec![0, 0x8000_0000_0000_0000, 1, 0x000f_ffff_ffff_ffff,
            0x0010_0000_0000_0000, 0x0350_0000_0000_0000, 0x0360_0000_0000_0000,
            0x0ff0_0000_0000_0000, 0x1000_0000_0000_0000, 0x3ff0_0000_0000_0000,
            0x4000_0000_0000_0000, 0xbff0_0000_0000_0000, 0x6fe0_0000_0000_0000,
            0x6ff0_0000_0000_0000, 0x7fc0_0000_0000_0000, 0x7fd0_0000_0000_0000,
            0x7fef_ffff_ffff_ffff, 0x7ff0_0000_0000_0000, 0xfff0_0000_0000_0000, 0x7ff8_0000_0000_0000])
    ] {
        let mut cases = Vec::new();
        for &den in &inputs { for &num in &inputs { for src in [den, num] {
            let (result, flag) = reference_scale(ty, [src, den, num]);
            cases.push(Case { inputs: [src, den, num, 0], result, flag: flag as u32 });
        } } }
        check(op, &cases);
    }
}

#[test]
fn fmas_is_fused_and_consumes_the_lane_flag() {
    for (op, ty, exponent) in [(I::V_DIV_FMAS_F32, Ty::F32, 32), (I::V_DIV_FMAS_F64, Ty::F64, 64)] {
        let mut cases = Vec::new();
        for [a, b, c] in [[1.5f64, -2., 4.], [1., 1., -1.], [1.25, 0.5, 0.125], [-1., 1., 0.]] {
            for flag in [0, 1] {
                let encode = |x: f64| if ty == Ty::F32 { (x as f32).to_bits() as u64 } else { x.to_bits() };
                let result = if ty == Ty::F32 {
                    let fused = (a as f32).mul_add(b as f32, c as f32);
                    (if flag == 1 { libm::scalbnf(fused, exponent) } else { fused }).to_bits() as u64
                } else { let fused = a.mul_add(b, c); (if flag == 1 { libm::scalbn(fused, exponent) } else { fused }).to_bits() };
                cases.push(Case { inputs: [encode(a), encode(b), encode(c), flag], result, flag: 0 });
            }
        }
        // A multiply rounded before the addition would incorrectly return 0.
        let (a, b, c, result) = if ty == Ty::F32 {
            (0x3f80_0001, 0x3f7f_fffe, 0xbf80_0000, 0xa880_0000)
        } else { (0x3ff0_0000_0000_0001, 0x3fef_ffff_ffff_fffe, 0xbff0_0000_0000_0000, 0xb970_0000_0000_0000) };
        cases.push(Case { inputs: [a, b, c, 0], result, flag: 0 });
        check(op, &cases);
    }
}
