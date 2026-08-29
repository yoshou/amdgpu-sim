//! JIT-compiled `v_wmma_f32_16x16x16_f16` for the wave-level boundary of the
//! packed cross-lane dispatch.
//!
//! The interpreted apply in [`super::coop_xlane`] makes three passes over the
//! wave: gather the fragments into row-major matrices, multiply, scatter the
//! result back. This module fuses them into one function per packet width,
//! compiled once, reading and writing the packet register arrays directly. The
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
//! interpreter, which `coop_xlane`'s layout test and the examples'
//! `--verify_widths` check.

use std::sync::OnceLock;

use llvm_sys as llvm;

/// `void apply(u32 **packets, u32 vdst, u32 a, u32 b, u32 c)`.
type ApplyFn = unsafe extern "C" fn(*const *mut u32, u32, u32, u32, u32);

/// One compiled function per supported width, indexed by `log2(width)`.
static APPLY: [OnceLock<u64>; 5] =
    [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];

fn apply_fn(width: usize) -> ApplyFn {
    assert!(matches!(width, 1 | 2 | 4 | 8 | 16), "unsupported packet width {}", width);
    let addr =
        *APPLY[width.trailing_zeros() as usize].get_or_init(|| unsafe { compile(width as u32) });
    unsafe { std::mem::transmute::<u64, ApplyFn>(addr) }
}

/// Compile the width-W function now (idempotent). Callers do this while
/// building the kernel, so a dispatch is never charged for it.
pub(super) fn warm(width: usize) {
    let _ = apply_fn(width);
}

/// Apply the op to a wave held as `32 / width` packets in register-major SoA
/// layout (`vgprs[packet][reg * width + lane]`). WMMA ignores EXEC, so every
/// lane is read and written.
pub(super) fn apply(vdst: u32, a: u32, b: u32, c: u32, width: usize, vgprs: &mut [Vec<u32>]) {
    assert_eq!(vgprs.len(), 32 / width, "a wave is 32 lanes");
    let mut packets = [std::ptr::null_mut::<u32>(); 32];
    for (slot, packet) in packets.iter_mut().zip(vgprs.iter_mut()) {
        *slot = packet.as_mut_ptr();
    }
    unsafe { apply_fn(width)(packets.as_ptr(), vdst, a, b, c) }
}

/// Build the width-`w` function.
///
/// Fragment layout (the interpreter's): lane `l` holds A element `e` — f16
/// half `e % 2` of register `a + e / 2` — as `A[l % 16][col]` with
/// `col = e + (e / 4) * 4 + (l / 16) * 4`; B uses the same formula transposed;
/// accumulator register `c + m` holds `C[m + 8 * (l / 16)][l % 16]`, and D is
/// written in the same shape.
unsafe fn compile(w: u32) -> u64 {
    use llvm::core::*;

    llvm::target::LLVM_InitializeNativeTarget();
    llvm::target::LLVM_InitializeNativeAsmParser();
    llvm::target::LLVM_InitializeNativeAsmPrinter();

    let ctx = LLVMContextCreate();
    // The symbol a profiler or debugger will show; each width is its own
    // function, so name the width in it.
    let symbol = std::ffi::CString::new(format!("wmma_apply_w{w}")).unwrap();
    let module = LLVMModuleCreateWithNameInContext(symbol.as_ptr(), ctx);
    let b = LLVMCreateBuilderInContext(ctx);
    let anon = b"\0".as_ptr() as *const _;

    let i16t = LLVMInt16TypeInContext(ctx);
    let i32t = LLVMInt32TypeInContext(ctx);
    let f16t = LLVMHalfTypeInContext(ctx);
    let f32t = LLVMFloatTypeInContext(ctx);
    let void = LLVMVoidTypeInContext(ctx);
    let ptr = LLVMPointerTypeInContext(ctx, 0);
    let packet_i32 = LLVMVectorType(i32t, w);
    let wave_i32 = LLVMVectorType(i32t, 32);
    let wave_i16 = LLVMVectorType(i16t, 32);
    let wave_f16 = LLVMVectorType(f16t, 32);
    let wave_f32 = LLVMVectorType(f32t, 32);

    let mut params = [ptr, i32t, i32t, i32t, i32t];
    let fty = LLVMFunctionType(void, params.as_mut_ptr(), 5, 0);
    let func = LLVMAddFunction(module, symbol.as_ptr(), fty);
    LLVMPositionBuilderAtEnd(b, LLVMAppendBasicBlockInContext(ctx, func, anon));

    let konst = |v: u32| LLVMConstInt(i32t, v as u64, 0);
    let packets: Vec<llvm::prelude::LLVMValueRef> = (0..32 / w)
        .map(|p| {
            let gep =
                LLVMBuildGEP2(b, ptr, LLVMGetParam(func, 0), [konst(p)].as_mut_ptr(), 1, anon);
            LLVMBuildLoad2(b, ptr, gep, anon)
        })
        .collect();
    // Argument `n` (a register number) offset by `k`.
    let reg_of = |n: u32, k: u32| LLVMBuildAdd(b, LLVMGetParam(func, n), konst(k), anon);

    // One register across the whole wave as <32 x i32>: each packet holds its
    // W lanes contiguously at `packet + reg * W`, and packet order is lane
    // order (global lane = packet * W + lane within packet).
    let load_reg = |reg: llvm::prelude::LLVMValueRef| {
        let offset = LLVMBuildMul(b, reg, konst(w), anon);
        let mut parts: Vec<llvm::prelude::LLVMValueRef> = packets
            .iter()
            .map(|&p| {
                let gep = LLVMBuildGEP2(b, i32t, p, [offset].as_mut_ptr(), 1, anon);
                let load = LLVMBuildLoad2(b, packet_i32, gep, anon);
                LLVMSetAlignment(load, 4);
                load
            })
            .collect();
        let mut size = w;
        while size < 32 {
            parts = parts
                .chunks(2)
                .map(|pair| {
                    let mut mask: Vec<_> = (0..2 * size).map(konst).collect();
                    let mask = LLVMConstVector(mask.as_mut_ptr(), 2 * size);
                    LLVMBuildShuffleVector(b, pair[0], pair[1], mask, anon)
                })
                .collect();
            size *= 2;
        }
        parts[0]
    };

    // Fragment element `e` of the operand starting at argument `n`, as f32 per
    // lane. The f16 -> f32 widening is exact.
    let fragments = |n: u32| -> Vec<llvm::prelude::LLVMValueRef> {
        (0..8u32)
            .map(|e| {
                let mut bits = load_reg(reg_of(n, e / 2));
                if e % 2 == 1 {
                    let mut sh: Vec<_> = (0..32).map(|_| konst(16)).collect();
                    bits = LLVMBuildLShr(b, bits, LLVMConstVector(sh.as_mut_ptr(), 32), anon);
                }
                let half =
                    LLVMBuildBitCast(b, LLVMBuildTrunc(b, bits, wave_i16, anon), wave_f16, anon);
                LLVMBuildFPExt(b, half, wave_f32, anon)
            })
            .collect()
    };
    // Read every operand before the first store: `vdst` and `c` are the same
    // registers in rocwmma's accumulate loop.
    let a_frag = fragments(2);
    let b_frag = fragments(3);
    let mut acc: Vec<llvm::prelude::LLVMValueRef> = (0..8u32)
        .map(|m| LLVMBuildBitCast(b, load_reg(reg_of(4, m)), wave_f32, anon))
        .collect();

    // Gather a value held by another lane: `pick(v, f)` puts lane `f(l)`'s
    // element of `v` in lane `l`. This is where the fragment swizzle goes.
    let pick = |v: llvm::prelude::LLVMValueRef, from: &dyn Fn(u32) -> u32| {
        let mut mask: Vec<_> = (0..32).map(|lane| konst(from(lane))).collect();
        let mask = LLVMConstVector(mask.as_mut_ptr(), 32);
        LLVMBuildShuffleVector(b, v, LLVMGetPoison(wave_f32), mask, anon)
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
            let product = LLVMBuildFMul(b, a_mk, b_k, anon);
            acc[m as usize] = LLVMBuildFAdd(b, acc[m as usize], product, anon);
        }
    }

    for m in 0..8u32 {
        let value = LLVMBuildBitCast(b, acc[m as usize], wave_i32, anon);
        let offset = LLVMBuildMul(b, reg_of(1, m), konst(w), anon);
        for (p, &packet) in packets.iter().enumerate() {
            let mut mask: Vec<_> = (0..w).map(|lane| konst(p as u32 * w + lane)).collect();
            let mask = LLVMConstVector(mask.as_mut_ptr(), w);
            let lanes = LLVMBuildShuffleVector(b, value, LLVMGetPoison(wave_i32), mask, anon);
            let gep = LLVMBuildGEP2(b, i32t, packet, [offset].as_mut_ptr(), 1, anon);
            LLVMSetAlignment(LLVMBuildStore(b, lanes, gep), 4);
        }
    }
    LLVMBuildRetVoid(b);
    LLVMDisposeBuilder(b);

    jit(module, &symbol)
}

/// Verify, optimize, and JIT `module`; returns the address of `symbol`.
unsafe fn jit(module: llvm::prelude::LLVMModuleRef, symbol: &std::ffi::CStr) -> u64 {
    let mut error = std::ptr::null_mut();
    if llvm::analysis::LLVMVerifyModule(
        module,
        llvm::analysis::LLVMVerifierFailureAction::LLVMPrintMessageAction,
        &mut error,
    ) != 0
    {
        let message = std::ffi::CStr::from_ptr(error).to_string_lossy().into_owned();
        panic!("wmma: module failed verification:\n{}", message);
    }

    let triple = llvm::target_machine::LLVMGetDefaultTargetTriple();
    let mut target = std::ptr::null_mut();
    let mut target_error = std::ptr::null_mut();
    llvm::target_machine::LLVMGetTargetFromTriple(triple, &mut target, &mut target_error);
    let machine = llvm::target_machine::LLVMCreateTargetMachine(
        target,
        triple,
        llvm::target_machine::LLVMGetHostCPUName(),
        llvm::target_machine::LLVMGetHostCPUFeatures(),
        llvm::target_machine::LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        llvm::target_machine::LLVMRelocMode::LLVMRelocDefault,
        llvm::target_machine::LLVMCodeModel::LLVMCodeModelJITDefault,
    );
    let options = llvm::transforms::pass_builder::LLVMCreatePassBuilderOptions();
    let passes = std::ffi::CString::new("default<O3>").unwrap();
    let failure =
        llvm::transforms::pass_builder::LLVMRunPasses(module, passes.as_ptr(), machine, options);
    if !failure.is_null() {
        let message = llvm::error::LLVMGetErrorMessage(failure);
        let message = std::ffi::CStr::from_ptr(message).to_string_lossy().into_owned();
        panic!("wmma: optimization failed: {}", message);
    }

    let builder = llvm::orc2::lljit::LLVMOrcCreateLLJITBuilder();
    let target_builder = llvm::orc2::LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(machine);
    llvm::orc2::lljit::LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(builder, target_builder);
    let mut lljit = std::ptr::null_mut();
    if !llvm::orc2::lljit::LLVMOrcCreateLLJIT(&mut lljit, builder).is_null() {
        panic!("wmma: creating the JIT failed");
    }
    let thread_safe = llvm::orc2::LLVMOrcCreateNewThreadSafeContext();
    let thread_safe_module = llvm::orc2::LLVMOrcCreateNewThreadSafeModule(module, thread_safe);
    let dylib = llvm::orc2::lljit::LLVMOrcLLJITGetMainJITDylib(lljit);
    if !llvm::orc2::lljit::LLVMOrcLLJITAddLLVMIRModule(lljit, dylib, thread_safe_module).is_null() {
        panic!("wmma: adding the module failed");
    }
    let mut address = 0u64;
    if !llvm::orc2::lljit::LLVMOrcLLJITLookup(lljit, &mut address, symbol.as_ptr()).is_null() {
        panic!("wmma: symbol lookup failed: {}", symbol.to_string_lossy());
    }
    // The compiled code outlives this call; keep the JIT alive with it.
    std::mem::forget(Box::new(lljit));
    address
}
