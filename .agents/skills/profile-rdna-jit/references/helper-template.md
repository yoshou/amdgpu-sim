# Rust Helper Template

Insert this near the RDNA translator constants while profiling. Remove it after the profiling run.

```rust
struct JitObjectDumpContext {
    dump_objects: llvm::orc2::LLVMOrcDumpObjectsRef,
    symbol_name: String,
    symbol_size: std::sync::atomic::AtomicU64,
}

struct JitDebugInfo {
    builder: llvm::prelude::LLVMDIBuilderRef,
    scope: llvm::prelude::LLVMMetadataRef,
}

unsafe fn cstr(s: &str) -> std::ffi::CString {
    std::ffi::CString::new(s).unwrap()
}

unsafe fn add_u32_module_flag(
    context: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    key: &str,
    value: u32,
) {
    let key = cstr(key);
    let value = llvm::core::LLVMValueAsMetadata(llvm::core::LLVMConstInt(
        llvm::core::LLVMInt32TypeInContext(context),
        value as u64,
        0,
    ));
    llvm::core::LLVMAddModuleFlag(
        module,
        llvm::LLVMModuleFlagBehavior::LLVMModuleFlagBehaviorWarning,
        key.as_ptr(),
        key.as_bytes().len(),
        value,
    );
}

unsafe fn create_jit_debug_info(
    context: llvm::prelude::LLVMContextRef,
    module: llvm::prelude::LLVMModuleRef,
    function: llvm::prelude::LLVMValueRef,
    symbol_name: &str,
) -> JitDebugInfo {
    let filename = cstr("rdna-jit-blocks.s");
    let directory = cstr("/workspaces/amdgpu-sim");
    llvm::core::LLVMSetSourceFileName(module, filename.as_ptr(), filename.as_bytes().len());

    add_u32_module_flag(
        context,
        module,
        "Debug Info Version",
        llvm::debuginfo::LLVMDebugMetadataVersion(),
    );
    add_u32_module_flag(context, module, "Dwarf Version", 5);

    let builder = llvm::debuginfo::LLVMCreateDIBuilder(module);
    let producer = cstr("amdgpu-sim-rdna-jit");
    let file = llvm::debuginfo::LLVMDIBuilderCreateFile(
        builder,
        filename.as_ptr(),
        filename.as_bytes().len(),
        directory.as_ptr(),
        directory.as_bytes().len(),
    );
    llvm::debuginfo::LLVMDIBuilderCreateCompileUnit(
        builder,
        llvm::debuginfo::LLVMDWARFSourceLanguage::LLVMDWARFSourceLanguageAssembly,
        file,
        producer.as_ptr(),
        producer.as_bytes().len(),
        1,
        std::ptr::null(),
        0,
        0,
        std::ptr::null(),
        0,
        llvm::debuginfo::LLVMDWARFEmissionKind::LLVMDWARFEmissionKindLineTablesOnly,
        0,
        0,
        1,
        std::ptr::null(),
        0,
        std::ptr::null(),
        0,
    );

    let subroutine_type = llvm::debuginfo::LLVMDIBuilderCreateSubroutineType(
        builder,
        file,
        std::ptr::null_mut(),
        0,
        llvm::debuginfo::LLVMDIFlagZero,
    );
    let symbol_name = cstr(symbol_name);
    let subprogram = llvm::debuginfo::LLVMDIBuilderCreateFunction(
        builder,
        file,
        symbol_name.as_ptr(),
        symbol_name.as_bytes().len(),
        symbol_name.as_ptr(),
        symbol_name.as_bytes().len(),
        file,
        1,
        subroutine_type,
        0,
        1,
        1,
        llvm::debuginfo::LLVMDIFlagPrototyped,
        1,
    );
    llvm::debuginfo::LLVMSetSubprogram(function, subprogram);

    JitDebugInfo {
        builder,
        scope: subprogram,
    }
}

unsafe fn set_jit_debug_location(
    context: llvm::prelude::LLVMContextRef,
    builder: llvm::prelude::LLVMBuilderRef,
    debug_info: &JitDebugInfo,
    line: u32,
) {
    let location = llvm::debuginfo::LLVMDIBuilderCreateDebugLocation(
        context,
        line,
        1,
        debug_info.scope,
        std::ptr::null_mut(),
    );
    llvm::core::LLVMSetCurrentDebugLocation2(builder, location);
}

unsafe fn finalize_jit_debug_info(debug_info: JitDebugInfo) {
    llvm::debuginfo::LLVMDIBuilderFinalize(debug_info.builder);
    llvm::debuginfo::LLVMDisposeDIBuilder(debug_info.builder);
}

extern "C" fn dump_jit_object_callback(
    ctx: *mut c_void,
    obj_in_out: *mut llvm::prelude::LLVMMemoryBufferRef,
) -> llvm::error::LLVMErrorRef {
    unsafe {
        let ctx = &*(ctx as *const JitObjectDumpContext);
        let err = llvm::orc2::LLVMOrcDumpObjects_CallOperator(ctx.dump_objects, obj_in_out);
        if !err.is_null() {
            return err;
        }

        let obj = *obj_in_out;
        let start = llvm::core::LLVMGetBufferStart(obj);
        let size = llvm::core::LLVMGetBufferSize(obj);
        let name = std::ffi::CString::new("jit-object-copy").unwrap();
        let obj_copy =
            llvm::core::LLVMCreateMemoryBufferWithMemoryRangeCopy(start, size, name.as_ptr());
        let object_file = llvm::object::LLVMCreateObjectFile(obj_copy);
        if !object_file.is_null() {
            let symbols = llvm::object::LLVMGetSymbols(object_file);
            while llvm::object::LLVMIsSymbolIteratorAtEnd(object_file, symbols) == 0 {
                let symbol_name = llvm::object::LLVMGetSymbolName(symbols);
                if !symbol_name.is_null() {
                    if let Ok(symbol_name) = std::ffi::CStr::from_ptr(symbol_name).to_str() {
                        if symbol_name == ctx.symbol_name {
                            ctx.symbol_size.store(
                                llvm::object::LLVMGetSymbolSize(symbols),
                                std::sync::atomic::Ordering::Relaxed,
                            );
                            break;
                        }
                    }
                }
                llvm::object::LLVMMoveToNextSymbol(symbols);
            }
            llvm::object::LLVMDisposeSymbolIterator(symbols);
            llvm::object::LLVMDisposeObjectFile(object_file);
        }

        err
    }
}

unsafe fn install_jit_object_dump(
    jit: llvm::orc2::lljit::LLVMOrcLLJITRef,
    identifier: &str,
) -> *mut JitObjectDumpContext {
    let dump_dir = format!("/tmp/amdgpu-sim-jit-profile-{}", std::process::id());
    std::fs::create_dir_all(&dump_dir).unwrap();

    let dump_dir = std::ffi::CString::new(dump_dir).unwrap();
    let identifier = std::ffi::CString::new(identifier).unwrap();
    let dump_objects =
        llvm::orc2::LLVMOrcCreateDumpObjects(dump_dir.as_ptr(), identifier.as_ptr());
    let ctx = Box::into_raw(Box::new(JitObjectDumpContext {
        dump_objects,
        symbol_name: identifier.to_str().unwrap().to_string(),
        symbol_size: std::sync::atomic::AtomicU64::new(0),
    }));
    let obj_transform = llvm::orc2::lljit::LLVMOrcLLJITGetObjTransformLayer(jit);
    llvm::orc2::LLVMOrcObjectTransformLayerSetTransform(
        obj_transform,
        dump_jit_object_callback,
        ctx as *mut c_void,
    );
    ctx
}

unsafe fn write_perf_map(ctx: *mut JitObjectDumpContext, addr: u64) {
    let ctx = &*ctx;
    let size = ctx
        .symbol_size
        .load(std::sync::atomic::Ordering::Relaxed);
    if size == 0 {
        return;
    }

    let perf_map = format!("/tmp/perf-{}.map", std::process::id());
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(perf_map)
        .unwrap();
    writeln!(file, "{:x} {:x} {}", addr, size, ctx.symbol_name).unwrap();
}
```
