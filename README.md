# AMD GPU Instruction Set Simulator

## About
This simulator executes the HSA code object for AMD GPU on CPU.

## Supporting GPUs
* Radeon RX 9060 XT (gfx1200)
* Radeon R9 Nano (gfx803)

## Execution engines

| Architecture | Interpreter | LLVM JIT | SPMD JIT |
|--------------|-------------|----------|----------|
| gfx1200      | Supported   | Supported | Supported |
| gfx803       | Supported   | Not supported | Not supported |

### RDNA SPMD backend

The gfx1200 SPMD backend converts wavefront execution into independent CPU
work-items. The examples use cooperative or segmented dispatch when barriers or
cross-lane operations require synchronization.

Available examples are `bitonic_sort_spmd`, `histogram_spmd`,
`raytracing_spmd`, `simple_hgemm_spmd`, `smallpt_spmd`, `texture_spmd`, and
`warp_shuffle_spmd`.

```sh
cargo run --release --example smallpt_spmd -- --arch gfx1200
```

Use `--num_threads N` to select the CPU thread count. Examples that support
packed work-item execution also accept `--vec_width W`; `0` selects the
single-work-item path.

## Examples

### smallpt

Please execute the following command.

```sh
cargo run --release --example smallpt
```

The kernel program is based on the following CUDA code.

https://github.com/matt77hias/cu-smallpt

### bitonic sort

Please execute the following command.

```sh
cargo run --release --example bitonic_sort
```

The kernel program is based on the following code.

https://github.com/ROCm/rocm-examples

### histogram

Please execute the following command.

```sh
cargo run --release --example histogram
```

The kernel program is based on the following code.

https://github.com/ROCm/rocm-examples

### matrix multiplication

Please execute the following command.

```sh
cargo run --release --example simple_hgemm -- --arch gfx1200
```

The kernel program is based on the following code.

https://github.com/ROCm/rocWMMA

### ray tracing

Please execute the following command.

```sh
cargo run --release --example raytracing -- --arch gfx1200
```

The kernel program is based on the following code.

https://github.com/GPUOpen-LibrariesAndSDKs/HIPRTSDK

## Implementation techniques

* The kernel code is translated to an intermediate representation based on LLVM IR.
* The intermediate representation is highly optimized with LLVM optimization passes.
* The optimized intermediate representation is compiled to machine code of the host CPU with LLVM JIT.
* Vector operations are translated to SIMD instructions of the host CPU with LLVM.

## How to generate kernel objects

1. Convert CUDA to HIP with hipify.

See the following official sample: https://github.com/amd/rocm-examples/tree/develop/HIP-Basic/hipify

2. Compile the HIP code. 

Kernel objects are generated in the intermediate stage before being embedded in the executable binary.  
See the following official sample: https://github.com/amd/rocm-examples/tree/develop/HIP-Basic/llvm_ir_to_executable

## Reference
* https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf
* https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/gcn3-instruction-set-architecture.pdf
