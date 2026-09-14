//! JIT-compiled `v_wmma_f32_16x16x16_f16` for the wave-level boundary of the
//! packed cross-lane dispatch.
//!
//! The interpreted apply in [`super::xlane`] makes three passes over the
//! wave: gather the fragments into row-major matrices, multiply, scatter the
//! result back. This module fuses them into one function per packet width,
//! compiled once, reading and writing typed effect frames directly. The
//! fragment layout's swizzle becomes constant shuffle masks, so no matrix is
//! materialized and no address arithmetic survives to run time.
//!
//! It is JIT-compiled rather than written in Rust because the matrices then
//! stay in `<32 x f32>` values for the whole computation, with the layout
//! swizzle as shuffles between them; Rust has no vector type that wide, so the
//! same algorithm there has to keep them in arrays and go through memory.
//!
//! The float operations are the interpreter's, in its order: each output
//! element starts at its accumulator `C` and adds `a_k * b_k` for ascending
//! `k`, with mul and add rounded separately (no fast-math: FMA contraction
//! would change the result). Results are therefore bitwise identical to the
//! interpreter, which `engine::xlane`'s layout test and the examples'
//! `--verify_widths` check.

use std::sync::OnceLock;

use super::super::native::Value;

/// Dense effect frames: A at 0, B at 4, C/results at 8. No ISA registers
/// cross the wave/native boundary.
type ApplyFn = unsafe extern "C" fn(*const *mut u32);

/// One compiled function per supported width, indexed by `log2(width)`.
static APPLY: [OnceLock<super::super::native::jit::NativeCode>; 6] =
    [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];

fn apply_fn(width: usize) -> ApplyFn {
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16 | 32), "unsupported packet width {}", width);
    let addr = APPLY[width.trailing_zeros() as usize]
        .get_or_init(|| compile(width as u32)).address();
    unsafe { std::mem::transmute::<u64, ApplyFn>(addr) }
}

/// Compile the width-W function now (idempotent). Callers do this while
/// building the kernel, so a dispatch is never charged for it.
pub(in crate::rdna_spmd) fn warm(width: usize) {
    let _ = apply_fn(width);
}

/// Dense SSA frames hold A[0..4], B[4..8], C[8..16]. Results replace C
/// after the native kernel has captured all inputs, including aliased ones.
pub(in crate::rdna_spmd) unsafe fn apply_values(width: usize, packets: *const *mut u32) {
    apply_fn(width)(packets);
}

/// Apply the op to a wave held as `32 / width` packets in register-major SoA
/// layout (`vgprs[packet][reg * width + lane]`). WMMA ignores EXEC, so every
/// lane is read and written.
#[cfg(test)]
pub(in crate::rdna_spmd) fn apply(vdst: u32, a: u32, b: u32, c: u32, width: usize, vgprs: &mut [Vec<u32>]) {
    assert_eq!(vgprs.len(), 32 / width, "a wave is 32 lanes");
    let mut frames = vec![vec![0u32; 16 * width]; vgprs.len()];
    for (frame, registers) in frames.iter_mut().zip(vgprs.iter()) {
        for (slot, first, count) in [(0, a, 4), (4, b, 4), (8, c, 8)] {
            frame[slot*width..(slot+count)*width]
                .copy_from_slice(&registers[first as usize*width..(first as usize+count)*width]);
        }
    }
    let pointers: Vec<_> = frames.iter_mut().map(|frame| frame.as_mut_ptr()).collect();
    unsafe { apply_values(width, pointers.as_ptr()); }
    for (frame, registers) in frames.iter().zip(vgprs.iter_mut()) {
        registers[vdst as usize*width..(vdst as usize+8)*width].copy_from_slice(&frame[8*width..]);
    }
}

/// Build the width-`w` function.
///
/// Fragment layout (the interpreter's): lane `l` holds A element `e` — f16
/// half `e % 2` of register `a + e / 2` — as `A[l % 16][col]` with
/// `col = e + (e / 4) * 4 + (l / 16) * 4`; B uses the same formula transposed;
/// accumulator register `c + m` holds `C[m + 8 * (l / 16)][l % 16]`, and D is
/// written in the same shape.
fn compile(w: u32) -> super::super::native::jit::NativeCode {
    // Keep width-specific symbols visible to native profilers.
    let name = format!("wmma_apply_w{w}");
    let native = super::super::native::jit::Module::new(&name);
    let ir = native.builder();

    let (i32t, ptr) = (ir.i32(), ir.ptr());
    let packet_i32 = i32t.vector(w);
    let wave_i32 = i32t.vector(32);
    let wave_i16 = ir.i16().vector(32);
    let wave_f16 = ir.f16().vector(32);
    let wave_f32 = ir.f32().vector(32);

    let func = ir.add_function(&name, ir.void().function(&[ptr]));
    ir.position_at_end(ir.append_block(func, ""));

    let konst = |v: u32| ir.ci32(v);
    let packets: Vec<Value> = (0..32 / w)
        .map(|p| ir.load(ptr, ir.gep(ptr, func.param(0), &[konst(p)])))
        .collect();
    // Typed frame slot offset by fragment word, fixed before native emission.
    let reg_of = |n: u32, k: u32| konst(n + k);

    // One register across the whole wave as <32 x i32>: each packet holds its
    // W lanes contiguously at `packet + reg * W`, and packet order is lane
    // order (global lane = packet * W + lane within packet).
    let load_reg = |reg: Value| {
        let offset = ir.mul(reg, konst(w));
        let mut parts: Vec<Value> = packets
            .iter()
            .map(|&p| ir.load(packet_i32, ir.gep(i32t, p, &[offset])).set_alignment(4))
            .collect();
        let mut size = w;
        while size < 32 {
            parts = parts
                .chunks(2)
                .map(|pair| {
                    let mask: Vec<u32> = (0..2 * size).collect();
                    ir.shuffle_by(pair[0], pair[1], &mask)
                })
                .collect();
            size *= 2;
        }
        parts[0]
    };

    // Fragment element `e` of the operand starting at argument `n`, as f32 per
    // lane. The f16 -> f32 widening is exact.
    let fragments = |n: u32| -> Vec<Value> {
        (0..8u32)
            .map(|e| {
                let mut bits = load_reg(reg_of(n, e / 2));
                if e % 2 == 1 {
                    let shift = ir.const_vector(&vec![konst(16); 32]);
                    bits = ir.lshr(bits, shift);
                }
                let half = ir.bitcast(ir.trunc(bits, wave_i16), wave_f16);
                ir.fpext(half, wave_f32)
            })
            .collect()
    };
    // Read every operand before the first store: `vdst` and `c` are the same
    // registers in rocwmma's accumulate loop.
    let a_frag = fragments(0);
    let b_frag = fragments(4);
    let mut acc: Vec<Value> = (0..8u32)
        .map(|m| ir.bitcast(load_reg(reg_of(8, m)), wave_f32))
        .collect();

    // Gather a value held by another lane: `pick(v, f)` puts lane `f(l)`'s
    // element of `v` in lane `l`. This is where the fragment swizzle goes.
    let pick = |v: Value, from: &dyn Fn(u32) -> u32| {
        let mask: Vec<u32> = (0..32).map(|lane| from(lane)).collect();
        ir.shuffle_by(v, wave_f32.poison(), &mask)
    };

    for k in 0..16u32 {
        // Inverse of the column formula: column k of A (row k of B) is element
        // `e` in the lanes of half `g` of the wave.
        let e = ((k % 4) + 4 * (k / 8)) as usize;
        let g = (k / 4) % 2;
        // b_k lane l = B[k][l % 16]
        let b_k = pick(b_frag[e], &|lane| lane % 16 + 16 * g);
        for m in 0..8u32 {
            // a_mk lane l = A[m + 8 * (l / 16)][k]; the product joins the
            // accumulator in ascending k, as the interpreter sums it.
            let a_mk = pick(a_frag[e], &|lane| m + 8 * (lane / 16) + 16 * g);
            let product = ir.fmul(a_mk, b_k);
            acc[m as usize] = ir.fadd(acc[m as usize], product);
        }
    }

    for m in 0..8u32 {
        let value = ir.bitcast(acc[m as usize], wave_i32);
        let offset = ir.mul(reg_of(8, m), konst(w));
        for (p, &packet) in packets.iter().enumerate() {
            let mask: Vec<u32> = (0..w).map(|lane| p as u32 * w + lane).collect();
            let lanes = ir.shuffle_by(value, wave_i32.poison(), &mask);
            ir.store(lanes, ir.gep(i32t, packet, &[offset])).set_alignment(4);
        }
    }
    ir.ret_void();
    native.optimize(super::super::native::jit::Mode::Packet).compile(&name)
}
