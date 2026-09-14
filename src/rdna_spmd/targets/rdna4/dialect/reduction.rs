//! ISA §16.12 TRIG_PREOP_F64: extract a truncated 53-bit segment of
//! floor((2/pi) * 2^1201), then scale according to the input's biased exponent.
//! The table is immutable provider data, independent of architectural memory.
use super::*;

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

fn table(e: &Emitter) -> Value {
    let ir = e.ir;
    let name = "rdna4.two_over_pi";
    match ir.global(name) {
        Some(global) => global,
        None => {
            let i64 = ir.i64();
            let global = ir.add_global(name, i64.array(FRACTION.len() as u64));
            let values = FRACTION.iter().map(|&v| i64.const_int(v)).collect::<Vec<_>>();
            global.set_initializer(ir.const_array(i64, &values));
            global.set_constant();
            global.set_alignment(8);
            global.set_private();
            global
        }
    }
}

pub(super) fn lower(e: &Emitter, args: &[Value]) -> Value {
    let ir = e.ir;
    let k = |v: i32| e.constant(Ty::I32, v as u32 as u64);
    let bits = ir.bitcast(args[0], e.ty(Ty::I64));
    let exp = ir.lshr(bits, e.constant(Ty::I64, 52));
    let exp = ir.and(exp, e.constant(Ty::I64, 0x7ff));
    let exp = ir.trunc(exp, e.ty(Ty::I32));
    let segment = ir.and(args[1], k(31));
    let segment = ir.mul(segment, k(53));
    let extra = ir.sub(exp, k(1077));
    let over = ir.icmp(IntPred::Sgt, extra, k(0));
    let shift = ir.add(segment, ir.select(over, extra, k(0)));
    let offset = ir.sub(k(1148), shift);
    let valid = ir.icmp(IntPred::Sge, offset, k(0));
    let safe_offset = ir.select(valid, offset, k(0));
    let word = ir.lshr(safe_offset, k(6));
    let bit = ir.and(safe_offset, k(63));
    let bit = ir.zext(bit, e.ty(Ty::I64));
    let table = table(e);
    let read = |index: Value| {
        let ptr = ir.gep(ir.i64(), table, &[index]);
        if index.is_vector() {
            let w = index.ty().vector_size();
            let call = e.call(&format!("llvm.masked.gather.v{w}i64.v{w}p0"), Ty::I64,
                &[ptr, e.constant(Ty::I1, 1), e.constant(Ty::I64, 0)]);
            ir.set_call_align(call, 1, 8);
            call
        } else { ir.load(e.ty(Ty::I64), ptr).set_alignment(8) }
    };
    let lo = read(word); let hi = read(ir.add(word, k(1)));
    let fraction = e.call(&format!("llvm.fshr.{}", e.suffix(Ty::I64)), Ty::I64, &[hi, lo, bit]);
    let fraction = ir.and(fraction, e.constant(Ty::I64, (1 << 53) - 1));
    let fraction = ir.uitofp(fraction, e.ty(Ty::F64));
    let large = ir.icmp(IntPred::Sge, exp, k(1968));
    let base = ir.select(large, k(75), k(-53));
    let exponent = ir.sub(base, shift);
    let result = scale::f64(e, &[fraction, exponent]);
    ir.select(valid, result, e.constant(Ty::F64, 0))
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
