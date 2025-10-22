# AMD GPU Instruction Set Simulator

## About
This simulator executes the HSA code object for AMD GPU on CPU.

## Supporting GPUs
* Radeon RX 9060 XT (gfx1200)
* Radeon R9 Nano (gfx803)

## Execution engines

| Architecture | Interpreter | LLVM JIT |
|--------------|-------------|-----------|
| gfx1200      | Supported   | Supported |
| gfx803       | Supported   | Not supported |

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
cargo run --release --example simple_hgemm --arch gfx1200
```

The kernel program is based on the following code.

https://github.com/ROCm/rocWMMA

### ray tracing

Please execute the following command.

```sh
cargo run --release --example raytracing --arch gfx1200
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
