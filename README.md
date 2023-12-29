# AMD GCN Instruction Set Simulator

## About
This simulator executes the HSA code object for AMD GPU on CPU.

## Supporting targets
* Fiji (GFX803)

## Examples

### smallpt

Please execute the following command.

```sh
cargo run --release --example smallpt
```

The kernel program is based on the following CUDA code.

https://github.com/matt77hias/cu-smallpt

### bitonic sort

## How to generate kernel objects

1. Convert CUDA to HIP with hipify.

See the following official sample: https://github.com/amd/rocm-examples/tree/develop/HIP-Basic/hipify

2. Compile the HIP code. 

Kernel objects are generated in the intermediate stage before being embedded in the executable binary.  
See the following official sample: https://github.com/amd/rocm-examples/tree/develop/HIP-Basic/llvm_ir_to_executable

## Reference
* http://developer.amd.com/wordpress/media/2013/12/AMD_GCN3_Instruction_Set_Architecture_rev1.1.pdf
