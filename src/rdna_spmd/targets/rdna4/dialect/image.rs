//! RDNA4 ISA image resource/sampler fields and Table 61 address modes.
//! Point sampling of the R8 formats exercised by the captured VSAMPLE suite.
//! Coordinates and descriptors stay in SSA at every width. Masked byte gathers
//! avoid touching memory for constant components or border samples.
use super::*;
use llvm_sys::LLVMIntPredicate::*;
#[cfg(test)]
mod tests;

unsafe fn call(e: &Emitter, name: &str, result: LLVMTypeRef, args: &[LLVMValueRef]) -> LLVMValueRef {
    let name = std::ffi::CString::new(name).unwrap();
    let module = LLVMGetGlobalParent(LLVMGetBasicBlockParent(LLVMGetInsertBlock(e.b)));
    let mut types = args.iter().map(|&v| LLVMTypeOf(v)).collect::<Vec<_>>();
    let ty = LLVMFunctionType(result, types.as_mut_ptr(), types.len() as u32, 0);
    let mut f = LLVMGetNamedFunction(module, name.as_ptr());
    if f.is_null() { f = LLVMAddFunction(module, name.as_ptr(), ty); }
    LLVMBuildCall2(e.b, ty, f, args.to_vec().as_mut_ptr(), args.len() as u32, b"\0".as_ptr().cast())
}

unsafe fn require(e: &Emitter, mut valid: LLVMValueRef) {
    if let Some(w) = e.width() {
        valid = call(e, &format!("llvm.vector.reduce.and.v{w}i1"), LLVMInt1TypeInContext(e.ctx), &[valid]);
    }
    let f = LLVMGetBasicBlockParent(LLVMGetInsertBlock(e.b));
    let next = LLVMAppendBasicBlockInContext(e.ctx, f, b"image.valid\0".as_ptr().cast());
    let bad = LLVMAppendBasicBlockInContext(e.ctx, f, b"image.unsupported\0".as_ptr().cast());
    LLVMBuildCondBr(e.b, valid, next, bad);
    LLVMPositionBuilderAtEnd(e.b, bad);
    call(e, "llvm.trap", LLVMVoidTypeInContext(e.ctx), &[]);
    LLVMBuildUnreachable(e.b);
    LLVMPositionBuilderAtEnd(e.b, next);
}

pub(super) unsafe fn sample(e: &Emitter, a: &[LLVMValueRef]) -> LLVMValueRef {
    let b = e.b; let n = b"\0".as_ptr().cast();
    let k = |v| e.constant(Ty::I32, v);
    let eq = |a, c| LLVMBuildICmp(b, LLVMIntEQ, a, c, n);
    let and = |a, c| LLVMBuildAnd(b, a, c, n);
    let or = |a, c| LLVMBuildOr(b, a, c, n);
    let select = |p, a, c| LLVMBuildSelect(b, p, a, c, n);
    let shr = |a, bits| LLVMBuildLShr(b, a, k(bits), n);
    let field = |words: &[LLVMValueRef], start: usize, size: u32| {
        let low = shr(words[start / 32], (start % 32) as u64);
        let value = if start % 32 + size as usize > 32 {
            or(low, LLVMBuildShl(b, words[start / 32 + 1], k((32 - start % 32) as u64), n))
        } else { low };
        and(value, k((1u64 << size) - 1))
    };
    let format = field(&a[..8], 49, 8);
    let width = LLVMBuildAdd(b, field(&a[..8], 62, 16), k(1), n);
    let height = LLVMBuildAdd(b, field(&a[..8], 78, 16), k(1), n);
    let pitch = field(&a[..8], 128, 16);
    let row = select(eq(pitch, k(0)), width, LLVMBuildAdd(b, pitch, k(1), n));
    let row = and(LLVMBuildAdd(b, row, k(127), n), k(0xffff_ff80));
    let component_shift = LLVMBuildMul(b, a[12], k(3), n);
    let component = and(LLVMBuildLShr(b, a[3], component_shift, n), k(7));
    let data_component = LLVMBuildICmp(b, LLVMIntUGE, component, k(4), n);
    let filter = field(&a[8..12], 84, 2);
    let selector_valid = or(data_component, LLVMBuildICmp(b, LLVMIntULE, component, k(1), n));
    require(e, and(eq(filter, k(0)), selector_valid));

    let unrm = or(a[13], eq(field(&a[8..12], 15, 1), k(1)));
    let axis = |value, size, shift| {
        let extent = LLVMBuildUIToFP(b, size, e.ty(Ty::F32), n);
        let coordinate = select(unrm, value, LLVMBuildFMul(b, value, extent, n));
        let integral = e.call(&format!("llvm.floor.{}", e.suffix(Ty::F32)), Ty::F32, &[coordinate]);
        let coordinate = e.call(&format!("llvm.fptosi.sat.{}.{}", e.suffix(Ty::I32), e.suffix(Ty::F32)), Ty::I32, &[integral]);
        let encoded = and(shr(a[8], shift), k(7));
        let repeat = LLVMBuildICmp(b, LLVMIntULT, encoded, k(2), n);
        let mode = select(and(unrm, repeat), LLVMBuildAdd(b, encoded, k(2), n), encoded);
        let mirror = eq(and(mode, k(1)), k(1));
        let negative = LLVMBuildICmp(b, LLVMIntSLT, coordinate, k(0), n);
        let reflected = select(and(mirror, negative), LLVMBuildXor(b, coordinate, k(0xffff_ffff), n), coordinate);
        let last = LLVMBuildSub(b, size, k(1), n);
        let clamped = select(LLVMBuildICmp(b, LLVMIntSLT, reflected, k(0), n), k(0), reflected);
        let clamped = select(LLVMBuildICmp(b, LLVMIntSGT, clamped, last, n), last, clamped);
        let period = select(mirror, LLVMBuildShl(b, size, k(1), n), size);
        let rem = LLVMBuildSRem(b, coordinate, period, n);
        let rem = select(LLVMBuildICmp(b, LLVMIntSLT, rem, k(0), n), LLVMBuildAdd(b, rem, period, n), rem);
        let folded = LLVMBuildSub(b, LLVMBuildSub(b, period, k(1), n), rem, n);
        let repeated = select(LLVMBuildICmp(b, LLVMIntSGE, rem, size, n), folded, rem);
        let repeat = LLVMBuildICmp(b, LLVMIntULT, mode, k(2), n);
        let coord = select(repeat, repeated, clamped);
        let border_mode = LLVMBuildICmp(b, LLVMIntUGE, mode, k(6), n);
        let in_range = LLVMBuildICmp(b, LLVMIntULT, reflected, size, n);
        (coord, or(LLVMBuildNot(b, border_mode, n), in_range))
    };
    let (x, x_valid) = axis(a[14], width, 0);
    let (y, y_valid) = axis(a[15], height, 3);
    let inside = and(x_valid, y_valid);
    let load = and(inside, data_component);
    let supported_format = or(or(eq(format, k(1)), eq(format, k(2))), or(eq(format, k(5)), eq(format, k(6))));
    require(e, or(LLVMBuildNot(b, load, n), supported_format));
    let border = field(&a[8..12], 126, 2);
    let border_read = and(data_component, LLVMBuildNot(b, inside, n));
    require(e, or(LLVMBuildNot(b, border_read, n), LLVMBuildICmp(b, LLVMIntULT, border, k(3), n)));

    let wide = |v| LLVMBuildZExt(b, v, e.ty(Ty::I64), n);
    let base = or(LLVMBuildShl(b, wide(a[0]), e.constant(Ty::I64, 8), n),
        LLVMBuildShl(b, wide(and(a[1], k(255))), e.constant(Ty::I64, 40), n));
    let offset = LLVMBuildAdd(b, LLVMBuildMul(b, wide(y), wide(row), n), wide(x), n);
    let address = LLVMBuildAdd(b, base, offset, n);
    // The scalar path uses the same masked-gather semantics with one element;
    // LLVM lowers it to a guarded byte load, not an unconditional null load.
    let w = e.width().unwrap_or(1);
    let i8t = LLVMInt8TypeInContext(e.ctx); let byte_vector = LLVMVectorType(i8t, w);
    let pointer = LLVMPointerTypeInContext(e.ctx, 0); let pointers = LLVMVectorType(pointer, w);
    let (addresses, mask) = if e.width().is_some() { (address, load) } else {
        let index = LLVMConstInt(LLVMInt32TypeInContext(e.ctx), 0, 0);
        (LLVMBuildInsertElement(b, LLVMGetUndef(LLVMVectorType(LLVMInt64TypeInContext(e.ctx), 1)), address, index, n),
         LLVMBuildInsertElement(b, LLVMGetUndef(LLVMVectorType(LLVMInt1TypeInContext(e.ctx), 1)), load, index, n))
    };
    let pointers = LLVMBuildIntToPtr(b, addresses, pointers, n);
    let raw = call(e, &format!("llvm.masked.gather.v{w}i8.v{w}p0"), byte_vector,
        &[pointers, mask, LLVMConstNull(byte_vector)]);
    let align = LLVMGetEnumAttributeKindForName(b"align".as_ptr().cast(), 5);
    LLVMAddCallSiteAttribute(raw, 1, LLVMCreateEnumAttribute(e.ctx, align, 1));
    let raw = if e.width().is_some() { raw } else { LLVMBuildExtractElement(b, raw, LLVMConstInt(LLVMInt32TypeInContext(e.ctx), 0, 0), n) };
    let unsigned = LLVMBuildZExt(b, raw, e.ty(Ty::I32), n);
    let signed = LLVMBuildSExt(b, raw, e.ty(Ty::I32), n);
    let normalized_signed = select(LLVMBuildICmp(b, LLVMIntSLT, signed, k((-127i32) as u32 as u64), n), k((-127i32) as u32 as u64), signed);
    let unorm = LLVMBuildFDiv(b, LLVMBuildUIToFP(b, unsigned, e.ty(Ty::F32), n), e.constant(Ty::F32, 255f32.to_bits() as u64), n);
    let snorm = LLVMBuildFDiv(b, LLVMBuildSIToFP(b, normalized_signed, e.ty(Ty::F32), n), e.constant(Ty::F32, 127f32.to_bits() as u64), n);
    let mut value = select(eq(format, k(6)), signed, unsigned);
    value = select(eq(format, k(1)), LLVMBuildBitCast(b, unorm, e.ty(Ty::I32), n), value);
    value = select(eq(format, k(2)), LLVMBuildBitCast(b, snorm, e.ty(Ty::I32), n), value);
    let one = select(or(eq(format, k(1)), eq(format, k(2))), k(1f32.to_bits() as u64), k(1));
    let border_value = select(eq(border, k(2)), one, k(0));
    value = select(inside, value, border_value);
    value = select(eq(component, k(1)), one, value);
    select(eq(component, k(0)), k(0), value)
}
