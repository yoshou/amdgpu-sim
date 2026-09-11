use super::*;
use crate::rdna_spmd::{dialect::Arguments, ir::ValueId, jit};
use std::sync::Arc;

struct Case { inputs: [u64; 4], result: u64, flag: u32 }

fn quotient_ir(target: crate::rdna_spmd::dialect::TargetOp, kind: u8) -> crate::rdna_spmd::ir::Func {
    use crate::rdna_spmd::ir::*;
    let mut f = Func { entry: BlockId(0), blocks: std::collections::BTreeMap::new(), types: vec![] };
    let den = f.value(Ty::F64); let num = f.value(Ty::F64);
    let mut insts = Vec::new();
    let mut push = |ty, op| {
        let value = f.value(ty);
        insts.push(Inst::Core { value, ty, op });
        value
    };
    let mut quotient = push(Ty::F64, match kind {
        0 | 6 | 7 | 8 => Op::Float(FloatOp::Div, num, den),
        1 => Op::Convert(Cvt::Bitcast, Ty::F64, num), // Arbitrary src0.
        2 => Op::Float(FloatOp::Div, den, num),
        3 => Op::Float(FloatOp::Div, num, num), // Wrong denominator.
        4 => Op::Float(FloatOp::Div, den, den), // Wrong numerator.
        5 => Op::Const(Ty::F64, 0xfff8_1234_5678_9abc),
        _ => unreachable!(),
    });
    if kind == 7 { quotient = push(Ty::F64, Op::Convert(Cvt::Bitcast, Ty::F64, quotient)); }
    let denominator = if matches!(kind, 6 | 8) {
        let zero = push(Ty::F64, Op::Const(Ty::F64, 0));
        let predicate = push(Ty::I1, Op::FCmp(FloatPred::Ogt, num, zero));
        if kind == 6 {
            quotient = push(Ty::F64, Op::Select(predicate, quotient, num));
            den
        } else { push(Ty::F64, Op::Select(predicate, den, num)) }
    } else { den };
    let result = f.value(Ty::F64);
    insts.push(Inst::Target { provenance: None, op: target,
        args: Arguments::Ternary([quotient, denominator, num]), outputs: vec![(result, Ty::F64)] });
    f.blocks.insert(BlockId(0), Block { params: vec![(den, Ty::F64), (num, Ty::F64)], insts,
        term: Term::Ret(vec![quotient, denominator, num, result]) });
    f
}

#[test]
fn proven_quotient_matches_general_fixup_and_other_quotients_keep_isa_corrections() {
    use crate::rdna_spmd::ir::{Inst, Term};
    let mut cases = Vec::new();
    let edge = [0u64, 1, 0x000f_ffff_ffff_ffff, 0x0010_0000_0000_0000,
        0x3ff0_0000_0000_0000, 0x4000_0000_0000_0000,
        1075 << 52, 1076 << 52, 1077 << 52,
        0x7fef_ffff_ffff_ffff, 0x7ff0_0000_0000_0000,
        0x7ff0_0000_0000_0001, 0x7ff8_1234_5678_9abc];
    for &d in &edge { for &n in &edge {
        for sd in [0, 1u64 << 63] { for sn in [0, 1u64 << 63] { cases.push((d | sd, n | sn)); } }
    } }
    let mut seed = 0x0123_4567_89ab_cdefu64;
    let mut random = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; seed };
    for _ in 0..20_000 { cases.push((random(), random())); }
    let registry = Arc::new(crate::rdna_spmd::targets::rdna4::registry());
    let target = super::super::division(&registry, I::V_DIV_FIXUP_F64).unwrap();
    let programs: Vec<_> = (0..9).map(|kind| {
        let mut ir = quotient_ir(target, kind);
        ir.check(&registry).unwrap();
        let count = super::super::idioms::quotient_fixups(&mut ir, target);
        assert_eq!(count, usize::from(matches!(kind, 0 | 7)), "quotient kind {kind}");
        assert_eq!(super::super::idioms::quotient_fixups(&mut ir, target), 0);
        ir.check(&registry).unwrap();
        ir
    }).collect();
    for width in [0, 1, 2, 4, 8, 16, 32] {
        unsafe {
            let module = jit::Module::new("fixup_quotients");
            let b = module.builder; let ctx = module.ctx; let n = b"\0".as_ptr().cast();
            let ptr = LLVMPointerTypeInContext(ctx, 0);
            let ft = LLVMFunctionType(LLVMVoidTypeInContext(ctx), [ptr; 4].as_mut_ptr(), 4, 0);
            let f = LLVMAddFunction(module.module, b"kernel\0".as_ptr().cast(), ft);
            LLVMPositionBuilderAtEnd(b, LLVMAppendBasicBlockInContext(ctx, f, n));
            let e = Emitter::new(b, (width != 0).then_some(width), registry.clone());
            let den = LLVMBuildLoad2(b, e.ty(Ty::F64), LLVMGetParam(f, 0), n); LLVMSetAlignment(den, 4);
            let num = LLVMBuildLoad2(b, e.ty(Ty::F64), LLVMGetParam(f, 1), n); LLVMSetAlignment(num, 4);
            let lanes = width.max(1) as usize;
            for (index, ir) in programs.iter().enumerate() {
                let mut values = vec![std::ptr::null_mut(); ir.types.len()];
                values[0] = den; values[1] = num;
                let block = &ir.blocks[&ir.entry];
                for inst in &block.insts {
                    match inst {
                        Inst::Core { value, ty, op } => values[value.0] = e.op(*ty, *op, &values),
                        Inst::Target { op, args, outputs, .. } => {
                            let results = e.target(*op, *args, &values);
                            for ((value, _), result) in outputs.iter().zip(results) { values[value.0] = result; }
                        }
                        _ => unreachable!(),
                    }
                }
                let Term::Ret(returned) = &block.term else { unreachable!() };
                let args = [values[returned[0].0], values[returned[1].0], values[returned[2].0]];
                for (output, value) in [(2, fixup(&e, Ty::F64, &args)), (3, values[returned[3].0])] {
                    let mut offset = LLVMConstInt(LLVMInt64TypeInContext(ctx), (index * lanes) as u64, 0);
                    let dest = LLVMBuildGEP2(b, LLVMDoubleTypeInContext(ctx), LLVMGetParam(f, output), &mut offset, 1, n);
                    LLVMSetAlignment(LLVMBuildStore(b, value, dest), 4);
                }
            }
            LLVMBuildRetVoid(b);
            let code = module.finish(if width == 0 { jit::Mode::Scalar } else { jit::Mode::Packet });
            let run: unsafe extern "C" fn(*const u64, *const u64, *mut u64, *mut u64) = std::mem::transmute(code.address());
            let mut den = vec![0; lanes]; let mut num = vec![0; lanes];
            let mut general = vec![0; lanes * programs.len()]; let mut specialized = general.clone();
            for start in (0..cases.len()).step_by(lanes) {
                for lane in 0..lanes { (den[lane], num[lane]) = cases[(start + lane) % cases.len()]; }
                run(den.as_ptr(), num.as_ptr(), general.as_mut_ptr(), specialized.as_mut_ptr());
                assert_eq!(general, specialized, "width={width} case={start}");
            }
        }
    }
}

// Invoke actual generated providers with lane-distinct runtime arguments.
// Store both results independently, so dropped/reordered target outputs fail.
fn check(opcode: I, cases: &[Case]) {
    let registry = Arc::new(crate::rdna_spmd::targets::rdna4::registry());
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
