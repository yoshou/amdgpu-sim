# Temporary Native-Mapping Instrumentation

Apply this only for a profiling run, then remove it. The permanent repository must not depend on environment variables or change normal execution behavior.

## Constraints

- Use `apply_patch` for source edits.
- Do not add env-var switches.
- Do not commit this instrumentation unless the user explicitly asks for a dedicated profiling branch.
- Use deterministic `/tmp` outputs:
  - JIT object dump directory: `/tmp/amdgpu-sim-jit-profile-$PID`
  - perf map: `/tmp/perf-$PID.map`
- Name the JIT function `kernel_0x{program.entry_pc:x}` while the patch is present.
- Set synthetic DWARF line numbers to RDNA block PCs in decimal. Example: line `11040` means RDNA PC `0x2b20`.

## Required Source Changes

Patch `src/rdna_translator/mod.rs` in the `build_from_program` path.

If `std::os::raw::c_void` is not already imported, add it with the other imports.

1. Add temporary helper code near the translator constants:
   - `JitObjectDumpContext`
   - `JitDebugInfo`
   - CString helper
   - module flag helper for `Debug Info Version` and `Dwarf Version`
   - `create_jit_debug_info`
   - `set_jit_debug_location`
   - `finalize_jit_debug_info`
   - ORC object transform callback using `LLVMOrcCreateDumpObjects`
   - `install_jit_object_dump`
   - `write_perf_map`

   Use the exact helper template in [helper-template.md](helper-template.md), then adjust only if the local LLVM C API names differ.

2. In `build_from_program`, replace the JIT function name:

```rust
let func_name_string = format!("kernel_0x{:x}", program.entry_pc);
let func_name = std::ffi::CString::new(func_name_string.clone()).unwrap();
let function = llvm::core::LLVMAddFunction(module, func_name.as_ptr(), ty_function);
let jit_debug_info = create_jit_debug_info(context, module, function, &func_name_string);
```

3. After positioning the builder at `entry_bb`, set the entry debug location:

```rust
set_jit_debug_location(context, builder, &jit_debug_info, program.entry_pc as u32);
```

4. In the loop that emits each `program.insts_blocks` block, after `LLVMPositionBuilderAtEnd`, set the block debug location:

```rust
set_jit_debug_location(context, builder, &jit_debug_info, *addr as u32);
```

5. Before building the switch in `entry_bb`, set the entry debug location again.

6. After `LLVMDisposeBuilder(builder)` and before module verification, finalize DIBuilder:

```rust
finalize_jit_debug_info(jit_debug_info);
```

7. After `LLVMOrcCreateLLJIT`, install the dump callback:

```rust
let jit_dump_ctx = install_jit_object_dump(jit, &func_name_string);
```

8. After `LLVMOrcLLJITLookup`, write the perf map and print the artifact hint:

```rust
write_perf_map(jit_dump_ctx, func);
eprintln!(
    "JIT_NATIVE_PROFILE symbol={} block=0x{:x} pid={} addr=0x{:x}",
    func_name_string,
    program.entry_pc,
    std::process::id(),
    func
);
```

## Helper Implementation Requirements

The helper code must:

- create the dump directory with `std::fs::create_dir_all`,
- call `LLVMOrcLLJITGetObjTransformLayer`,
- call `LLVMOrcObjectTransformLayerSetTransform`,
- call `LLVMOrcDumpObjects_CallOperator` inside the transform callback,
- parse the copied object buffer to find `symbol_name` size,
- write a single perf-map line:

```text
<runtime_addr_hex> <size_hex> <symbol_name>
```

The debug info helper must:

- create a DIBuilder for the module,
- create file `rdna-jit-blocks.s` under `/workspaces/amdgpu-sim`,
- use `LLVMDWARFSourceLanguageAssembly`,
- use `LLVMDWARFEmissionKindLineTablesOnly`,
- set a subprogram on the JIT function,
- use `LLVMSetCurrentDebugLocation2` for the current builder location.

## Removal Checklist

Before final response:

- Remove all helper structs/functions added for profiling.
- Restore normal function name `kernel`.
- Remove `create_jit_debug_info`, `set_jit_debug_location`, `finalize_jit_debug_info`, `install_jit_object_dump`, and `write_perf_map` call sites.
- Run `rg -n "JIT_NATIVE_PROFILE|LLVMOrcCreateDumpObjects|LLVMSetCurrentDebugLocation|rdna-jit-blocks|AMDGPU_SIM_JIT|write_perf_map|install_jit_object_dump" src/rdna_translator/mod.rs`.
- The search must produce no profiling hook hits.
