//! Shared ownership, verification, optimization and registration of native IR.
use llvm_sys::{self as llvm, core::*, prelude::*};
use std::ffi::{CStr, CString};

pub(super) enum Mode {
    Scalar,
    Packet,
}

pub(super) struct Module {
    pub ctx: LLVMContextRef,
    pub module: LLVMModuleRef,
    pub builder: LLVMBuilderRef,
    context: llvm::orc2::LLVMOrcThreadSafeContextRef,
}

/// Executable memory belongs to this handle. Kernel execution borrows it, so
/// scoped dispatch workers finish before its LLJIT can be released.
pub(super) struct NativeCode {
    jit: llvm::orc2::lljit::LLVMOrcLLJITRef,
    address: u64,
    pub(super) block_counts: Option<(String, usize)>,
}
// Compilation is complete before publication. Running native code does not
// mutate the ORC session, and destruction requires exclusive ownership.
unsafe impl Send for NativeCode {}
unsafe impl Sync for NativeCode {}
impl NativeCode {
    pub fn address(&self) -> u64 { self.address }
}
impl Drop for NativeCode {
    fn drop(&mut self) {
        unsafe {
            if let Some((path, blocks)) = self.block_counts.take() {
                let mut address = 0u64;
                let name = CString::new("block_counts").unwrap();
                let error = llvm::orc2::lljit::LLVMOrcLLJITLookup(self.jit, &mut address, name.as_ptr());
                if error.is_null() && address != 0 {
                    let counts = std::slice::from_raw_parts(address as *const u64, blocks);
                    let text: String = counts.iter().enumerate().map(|(i, c)| format!("{i} {c}\n")).collect();
                    std::fs::write(format!("{path}.counts"), text).unwrap();
                } else if !error.is_null() { llvm::error::LLVMConsumeError(error); }
            }
            let error = llvm::orc2::lljit::LLVMOrcDisposeLLJIT(self.jit);
            if !error.is_null() { llvm::error::LLVMConsumeError(error); }
        }
    }
}

/// The exact optimized module subsequently consumed by ORC, with its native
/// target machine. Inspection uses LLVM's printer on this owned module.
pub(super) struct OptimizedModule {
    module: Module,
    machine: llvm::target_machine::LLVMTargetMachineRef,
}
impl Drop for OptimizedModule {
    fn drop(&mut self) {
        if !self.machine.is_null() {
            unsafe { llvm::target_machine::LLVMDisposeTargetMachine(self.machine); }
        }
    }
}

unsafe fn check(error: llvm::error::LLVMErrorRef, operation: &str) {
    if !error.is_null() {
        let message = llvm::error::LLVMGetErrorMessage(error);
        let text = CStr::from_ptr(message).to_string_lossy().into_owned();
        llvm::error::LLVMDisposeErrorMessage(message);
        panic!("SPMD {}: {}", operation, text);
    }
}

impl Module {
    pub unsafe fn new(name: &str) -> Self {
        // LLVM's process-wide target registry must be initialized before
        // concurrent compiler instances begin creating target machines.
        static TARGETS: std::sync::Once = std::sync::Once::new();
        TARGETS.call_once(|| {
            assert_eq!(llvm::target::LLVM_InitializeNativeTarget(), 0);
            assert_eq!(llvm::target::LLVM_InitializeNativeAsmParser(), 0);
            assert_eq!(llvm::target::LLVM_InitializeNativeAsmPrinter(), 0);
        });
        let ctx = LLVMContextCreate();
        let context = llvm::orc2::LLVMOrcCreateNewThreadSafeContextFromLLVMContext(ctx);
        let name = CString::new(name).unwrap();
        Self {
            ctx,
            module: LLVMModuleCreateWithNameInContext(name.as_ptr(), ctx),
            builder: LLVMCreateBuilderInContext(ctx),
            context,
        }
    }

    pub unsafe fn finish(self, mode: Mode) -> NativeCode {
        self.optimize(mode).compile("kernel")
    }

    pub unsafe fn optimize(mut self, mode: Mode) -> OptimizedModule {
        LLVMDisposeBuilder(self.builder);
        self.builder = std::ptr::null_mut();
        let mut message = std::ptr::null_mut();
        let invalid = llvm::analysis::LLVMVerifyModule(
            self.module,
            llvm::analysis::LLVMVerifierFailureAction::LLVMReturnStatusAction,
            &mut message,
        );
        let diagnostic = if message.is_null() { String::new() } else {
            let text = CStr::from_ptr(message).to_string_lossy().into_owned();
            LLVMDisposeMessage(message);
            text
        };
        assert_eq!(invalid, 0, "SPMD module verification: {}", diagnostic);

        let triple = llvm::target_machine::LLVMGetDefaultTargetTriple();
        let mut target = std::ptr::null_mut();
        let mut error = std::ptr::null_mut();
        if llvm::target_machine::LLVMGetTargetFromTriple(triple, &mut target, &mut error) != 0 {
            let diagnostic = CStr::from_ptr(error).to_string_lossy().into_owned();
            LLVMDisposeMessage(error);
            LLVMDisposeMessage(triple);
            panic!("SPMD native target unavailable: {}", diagnostic);
        }
        let cpu = llvm::target_machine::LLVMGetHostCPUName();
        let host_features = llvm::target_machine::LLVMGetHostCPUFeatures();
        let features = CStr::from_ptr(host_features).to_string_lossy();
        // Preserve the scalar backend's measured AVX-512 exclusion: scalar
        // f64 selects use blend/cmov rather than mask-register round trips.
        let features = CString::new(if matches!(mode, Mode::Scalar) {
            format!("{},-avx512f,-avx512vl,-avx512dq,-avx512bw,-avx512cd", features)
        } else { features.into_owned() }).unwrap();
        // Keep JITDefault relocation/code model. Small/PIC changed LICM and
        // regressed the existing scalar reciprocal loop.
        let machine = llvm::target_machine::LLVMCreateTargetMachine(
            target, triple, cpu, features.as_ptr(),
            llvm::target_machine::LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
            llvm::target_machine::LLVMRelocMode::LLVMRelocDefault,
            llvm::target_machine::LLVMCodeModel::LLVMCodeModelJITDefault,
        );
        LLVMDisposeMessage(triple);
        LLVMDisposeMessage(cpu);
        LLVMDisposeMessage(host_features);
        if let Ok(dir) = std::env::var("AMDGPU_SIM_DUMP_LLVM") {
            static INDEX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let index = INDEX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            std::fs::create_dir_all(&dir).unwrap();
            let filename = CString::new(format!("{dir}/{index:04}-before.ll")).unwrap();
            let mut message = std::ptr::null_mut();
            LLVMPrintModuleToFile(self.module, filename.as_ptr(), &mut message);
        }
        let optimized = OptimizedModule { module: self, machine };
        let options = llvm::transforms::pass_builder::LLVMCreatePassBuilderOptions();
        let error = llvm::transforms::pass_builder::LLVMRunPasses(
            optimized.module.module, if std::env::var("AMDGPU_SIM_OPT").map_or(false, |v| v == "0") { b"default<O0>\0".as_ptr().cast() } else { b"default<O3>\0".as_ptr().cast() }, machine, options,
        );
        llvm::transforms::pass_builder::LLVMDisposePassBuilderOptions(options);
        check(error, "optimization");
        if let Ok(dir) = std::env::var("AMDGPU_SIM_DUMP_LLVM") {
            static INDEX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let index = INDEX.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let filename = CString::new(format!("{dir}/{index:04}-after.ll")).unwrap();
            let mut message = std::ptr::null_mut();
            LLVMPrintModuleToFile(optimized.module.module, filename.as_ptr(), &mut message);
        }

        optimized
    }
}

impl OptimizedModule {
    #[cfg(test)]
    pub fn ir(&self) -> String {
        unsafe {
            let text = LLVMPrintModuleToString(self.module.module);
            let ir = CStr::from_ptr(text).to_string_lossy().into_owned();
            LLVMDisposeMessage(text);
            ir
        }
    }

    pub unsafe fn compile(mut self, symbol: &str) -> NativeCode {
        let builder = llvm::orc2::lljit::LLVMOrcCreateLLJITBuilder();
        let machine = llvm::orc2::LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(self.machine);
        self.machine = std::ptr::null_mut();
        llvm::orc2::lljit::LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(builder, machine);
        let mut jit = std::ptr::null_mut();
        check(llvm::orc2::lljit::LLVMOrcCreateLLJIT(&mut jit, builder), "creation");
        let mut native = NativeCode { jit, address: 0, block_counts: None };
        let dylib = llvm::orc2::lljit::LLVMOrcLLJITGetMainJITDylib(jit);
        // Bind context-switch entrypoints to this host's FiberCtx ABI. A stale
        // separately built dylib must not supply a different context layout.
        let mut symbols = [
            (&b"amdgpu_sim_fiber_yield_values\0"[..], super::engine::fiber::amdgpu_sim_fiber_yield_values as *const () as u64),
        ].map(|(name, address)| llvm::orc2::LLVMOrcCSymbolMapPair {
            Name: llvm::orc2::lljit::LLVMOrcLLJITMangleAndIntern(jit, name.as_ptr().cast()),
            Sym: llvm::orc2::LLVMJITEvaluatedSymbol {
                Address: address,
                Flags: llvm::orc2::LLVMJITSymbolFlags {
                    GenericFlags: llvm::orc2::LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsExported as u8
                        | llvm::orc2::LLVMJITSymbolGenericFlags::LLVMJITSymbolGenericFlagsCallable as u8,
                    TargetFlags: 0,
                },
            },
        });
        let unit = llvm::orc2::LLVMOrcAbsoluteSymbols(symbols.as_mut_ptr(), symbols.len());
        let error = llvm::orc2::LLVMOrcJITDylibDefine(dylib, unit);
        if !error.is_null() { llvm::orc2::LLVMOrcDisposeMaterializationUnit(unit); }
        check(error, "fiber symbols");
        let prefix = llvm::orc2::lljit::LLVMOrcLLJITGetGlobalPrefix(jit);
        let mut generator = std::ptr::null_mut();
        check(llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
            &mut generator, prefix, None, std::ptr::null_mut()), "process symbols");
        llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, generator);
        let library: &[u8] = if cfg!(debug_assertions) {
            b"target/debug/libamdgpu_sim.so\0"
        } else { b"target/release/libamdgpu_sim.so\0" };
        let mut generator = std::ptr::null_mut();
        check(llvm::orc2::LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(
            &mut generator, library.as_ptr().cast(), prefix, None, std::ptr::null_mut()), "runtime symbols");
        llvm::orc2::LLVMOrcJITDylibAddGenerator(dylib, generator);

        // The module and ORC wrapper share the very same LLVMContext. ORC
        // retains its reference after this construction handle is released.
        let module = llvm::orc2::LLVMOrcCreateNewThreadSafeModule(self.module.module, self.module.context);
        self.module.module = std::ptr::null_mut();
        check(llvm::orc2::lljit::LLVMOrcLLJITAddLLVMIRModule(jit, dylib, module), "module registration");
        let symbol = CString::new(symbol).unwrap();
        check(llvm::orc2::lljit::LLVMOrcLLJITLookup(jit, &mut native.address, symbol.as_ptr()), "kernel lookup");
        if let Ok(dir) = std::env::var("AMDGPU_SIM_DUMP_CODE") {
            use std::io::{Read, Seek, Write};
            let maps = std::fs::read_to_string("/proc/self/maps").unwrap();
            for line in maps.lines() {
                let range = line.split_whitespace().next().unwrap();
                let (lo, hi) = range.split_once('-').unwrap();
                let (lo, hi) = (u64::from_str_radix(lo, 16).unwrap(), u64::from_str_radix(hi, 16).unwrap());
                if native.address < lo || native.address >= hi { continue; }
                let mut bytes = vec![0u8; (hi - lo) as usize];
                let mut mem = std::fs::File::open("/proc/self/mem").unwrap();
                mem.seek(std::io::SeekFrom::Start(lo)).unwrap();
                mem.read_exact(&mut bytes).unwrap();
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(format!("{dir}/{lo:x}-{hi:x}.bin"), &bytes).unwrap();
                let mut index = std::fs::OpenOptions::new().create(true).append(true).open(format!("{dir}/index.txt")).unwrap();
                writeln!(index, "{} {:x} {lo:x} {hi:x}", symbol.to_string_lossy(), native.address).unwrap();
            }
        }
        native
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        unsafe {
            if !self.builder.is_null() { LLVMDisposeBuilder(self.builder); }
            if !self.module.is_null() { LLVMDisposeModule(self.module); }
            llvm::orc2::LLVMOrcDisposeThreadSafeContext(self.context);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inspected_optimized_module_is_the_one_consumed_by_jit() {
        unsafe {
            for constant in [7, 19] {
                let module = Module::new("owned_jit_test");
                let i32t = LLVMInt32TypeInContext(module.ctx);
                let ty = LLVMFunctionType(i32t, std::ptr::null_mut(), 0, 0);
                let function = LLVMAddFunction(module.module, b"answer\0".as_ptr().cast(), ty);
                let entry = LLVMAppendBasicBlockInContext(module.ctx, function, b"entry\0".as_ptr().cast());
                LLVMPositionBuilderAtEnd(module.builder, entry);
                LLVMBuildRet(module.builder, LLVMConstInt(i32t, constant, 0));
                let optimized = module.optimize(Mode::Packet);
                assert!(optimized.ir().contains(&format!("ret i32 {}", constant)));
                let code = optimized.compile("answer");
                let call = std::mem::transmute::<u64, unsafe extern "C" fn() -> u32>(code.address());
                assert_eq!(call(), constant as u32);
                // Drop this ORC session before building and calling the next one.
            }
        }
    }
}
