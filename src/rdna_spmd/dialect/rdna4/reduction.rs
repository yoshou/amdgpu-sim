//! ISA §16.12 TRIG_PREOP_F64: extract a truncated 53-bit segment of
//! floor((2/pi) * 2^1201), then scale according to the input's biased exponent.
//! The table is immutable provider data, independent of architectural memory.
use super::*;
use llvm_sys::{LLVMIntPredicate::*, LLVMTypeKind};

// Little-endian words of floor((2/pi) * 2^1201), with a sentinel word.
const FRACTION: [u64; 20] = [
    0xBA10AC06608DF8F6, 0x25D4D7F6BF623F1A, 0xE2F67A0E73EF14A5,
    0xD45AEA4F758FD7CB, 0x136E9E8C7ECD3CBF, 0xDA3EDA6CFD9E4F96,
    0x301FDE5E2316B414, 0x50763FF12FFFBC0B, 0x73E93908BF177BF2,
    0xFC827323AC7306A6, 0x8909D338E04D68BE, 0x4E7DD1046BEA5D76,
    0x2439FC3BD6396253, 0xA5C00C925DD413A3, 0x8AC36E48DC74849B,
    0x2083FCA2C757BD77, 0xBB81B6C52B327887, 0x2A53F84EAFA3EA69,
    0x000145F306DC9C88, 0,
];

unsafe fn table(e: &Emitter) -> LLVMValueRef {
    let module = LLVMGetGlobalParent(LLVMGetBasicBlockParent(LLVMGetInsertBlock(e.b)));
    let name = b"rdna4.two_over_pi\0".as_ptr().cast();
    let mut global = LLVMGetNamedGlobal(module, name);
    if global.is_null() {
        let i64 = LLVMInt64TypeInContext(e.ctx);
        global = LLVMAddGlobal(module, LLVMArrayType2(i64, FRACTION.len() as u64), name);
        let mut values = FRACTION.iter().map(|&v| LLVMConstInt(i64, v, 0)).collect::<Vec<_>>();
        LLVMSetInitializer(global, LLVMConstArray2(i64, values.as_mut_ptr(), values.len() as u64));
        LLVMSetGlobalConstant(global, 1); LLVMSetAlignment(global, 8);
        LLVMSetLinkage(global, llvm_sys::LLVMLinkage::LLVMPrivateLinkage);
    }
    global
}

pub(super) unsafe fn lower(e: &Emitter, args: &[LLVMValueRef]) -> LLVMValueRef {
    let n = b"\0".as_ptr().cast();
    let k = |v: i32| e.constant(Ty::I32, v as u32 as u64);
    let bits = LLVMBuildBitCast(e.b, args[0], e.ty(Ty::I64), n);
    let exp = LLVMBuildLShr(e.b, bits, e.constant(Ty::I64, 52), n);
    let exp = LLVMBuildAnd(e.b, exp, e.constant(Ty::I64, 0x7ff), n);
    let exp = LLVMBuildTrunc(e.b, exp, e.ty(Ty::I32), n);
    let segment = LLVMBuildAnd(e.b, args[1], k(31), n);
    let segment = LLVMBuildMul(e.b, segment, k(53), n);
    let extra = LLVMBuildSub(e.b, exp, k(1077), n);
    let over = LLVMBuildICmp(e.b, LLVMIntSGT, extra, k(0), n);
    let shift = LLVMBuildAdd(e.b, segment, LLVMBuildSelect(e.b, over, extra, k(0), n), n);
    let offset = LLVMBuildSub(e.b, k(1148), shift, n);
    let valid = LLVMBuildICmp(e.b, LLVMIntSGE, offset, k(0), n);
    let safe_offset = LLVMBuildSelect(e.b, valid, offset, k(0), n);
    let word = LLVMBuildLShr(e.b, safe_offset, k(6), n);
    let bit = LLVMBuildAnd(e.b, safe_offset, k(63), n);
    let bit = LLVMBuildZExt(e.b, bit, e.ty(Ty::I64), n);
    let table = table(e);
    let read = |index| {
        let ptr = LLVMBuildGEP2(e.b, LLVMInt64TypeInContext(e.ctx), table, [index].as_mut_ptr(), 1, n);
        if LLVMGetTypeKind(LLVMTypeOf(index)) == LLVMTypeKind::LLVMVectorTypeKind {
            let w = LLVMGetVectorSize(LLVMTypeOf(index));
            let call = e.call(&format!("llvm.masked.gather.v{w}i64.v{w}p0"), Ty::I64,
                &[ptr, e.constant(Ty::I1, 1), e.constant(Ty::I64, 0)]);
            let align = LLVMGetEnumAttributeKindForName(b"align".as_ptr().cast(), 5);
            LLVMAddCallSiteAttribute(call, 1, LLVMCreateEnumAttribute(e.ctx, align, 8));
            call
        } else { let value = LLVMBuildLoad2(e.b, e.ty(Ty::I64), ptr, n); LLVMSetAlignment(value, 8); value }
    };
    let lo = read(word); let hi = read(LLVMBuildAdd(e.b, word, k(1), n));
    let fraction = e.call(&format!("llvm.fshr.{}", e.suffix(Ty::I64)), Ty::I64, &[hi, lo, bit]);
    let fraction = LLVMBuildAnd(e.b, fraction, e.constant(Ty::I64, (1 << 53) - 1), n);
    let fraction = LLVMBuildUIToFP(e.b, fraction, e.ty(Ty::F64), n);
    let large = LLVMBuildICmp(e.b, LLVMIntSGE, exp, k(1968), n);
    let base = LLVMBuildSelect(e.b, large, k(75), k(-53), n);
    let exponent = LLVMBuildSub(e.b, base, shift, n);
    let result = scale::f64(e, &[fraction, exponent]);
    LLVMBuildSelect(e.b, valid, result, e.constant(Ty::F64, 0), n)
}

#[cfg(test)]
pub(in crate::rdna_spmd) fn reference(bits: u64, selector: u32) -> u64 {
    let exponent = (bits >> 52 & 0x7ff) as i32;
    let shift = (selector & 31) as i32 * 53 + (exponent - 1077).max(0);
    let offset = 1148 - shift;
    if offset < 0 { return 0; }
    // Independent bit-by-bit extraction; native code uses a two-word window.
    let mut fraction = 0u64;
    for bit in 0..53 {
        let position = offset as usize + bit;
        fraction |= (FRACTION[position / 64] >> (position % 64) & 1) << bit;
    }
    libm::scalbn(fraction as f64, -53 - shift + if exponent >= 1968 { 128 } else { 0 }).to_bits()
}
