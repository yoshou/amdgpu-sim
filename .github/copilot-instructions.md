# Copilot instructions for `amdgpu-sim`

## Build, test, and run commands

- `cargo build` or `cargo build --release` builds the library and examples. This crate depends on `llvm-sys = "201.0.1"` with `llvm-sys/prefer-dynamic`, so local builds/tests require a compatible system LLVM installation or `LLVM_SYS_201_PREFIX`.
- The checked-in devcontainer now installs LLVM 20 from apt.llvm.org and sets `LLVM_SYS_201_PREFIX=/usr/lib/llvm-20`, which is the expected containerized build environment for this repo.
- `cargo test` is the standard test entry point. To run a filtered test, use `cargo test <filter>` or `cargo test <filter> -- --exact`.
- Example binaries are the main executable surface:
  - `cargo run --release --example smallpt`
  - `cargo run --release --example smallpt -- --arch gfx1200`
  - `cargo run --release --example smallpt -- --arch gfx1200 --nb_samples 256`
  - `cargo run --release --example bitonic_sort -- --arch gfx803`
  - `cargo run --release --example histogram -- --arch gfx1200`
  - `cargo run --release --example raytracing -- --arch gfx1200`
  - `cargo run --release --example simple_hgemm -- --arch gfx1200`
- Pass example-specific flags after `--`. The examples parse `--arch` themselves; do not use Cargo target flags as a substitute.
- No repository-specific lint command is documented or configured in this repo. Do not assume `clippy` or `fmt` are part of the required workflow unless the user asks for them.

## High-level architecture

- `src/lib.rs` exposes two architecture-specific execution stacks plus shared helpers:
  - shared infrastructure in `src/buffer.rs`, `src/bit.rs`, `src/instructions.rs`, and `src/processor.rs`
  - GCN/gfx803 path in `src/gcn3_decoder.rs`, `src/gcn_instructions.rs`, and `src/gcn_processor.rs`
  - RDNA/gfx1200 path in `src/rdna4_decoder.rs`, `src/rdna_instructions.rs`, `src/rdna_processor.rs`, and `src/rdna_translator.rs`
- The big-picture execution flow is: load an HSA kernel object (`kernel_<arch>.o`) from an example, parse ELF note metadata, build an `HsaKernelDispatchPacket`, then dispatch to either the GCN processor or the RDNA processor.
- The architecture split is intentional:
  - gfx803 uses the GCN decoder/interpreter path
  - gfx1200 uses the RDNA decoder/processor path and also has LLVM-based translation/JIT support through `src/rdna_translator.rs`
- `src/processor.rs` defines shared ABI-facing pieces such as `KernelDescriptor`, `Pointer`, and `HsaKernelDispatchPacket`; the example programs and processor implementations both rely on these types.
- The example programs are not just demos; they show the expected host-side integration pattern. Each example loads code objects with the `object` crate, reads note metadata as YAML or MessagePack, allocates argument buffers manually, and invokes the simulator directly.

## Key conventions

- Treat the example entrypoints as the canonical CLI/API surface for running the simulator. Most of the operational behavior is encoded in `examples/*/main.rs`, not in a separate driver layer.
- Example metadata parsing is duplicated across examples and follows the same note-type convention: note type `10` is YAML metadata, and note type `32` is MessagePack metadata.
- Architecture defaults vary by example, so be explicit when running commands:
  - `smallpt`, `bitonic_sort`, `histogram`, and `texture` default to `gfx803`
  - `raytracing` defaults to `gfx1200`
  - `simple_hgemm` should be run with `--arch gfx1200`; its source still has a `gfx942` default even though this repository ships `examples/simple_hgemm/kernel_gfx1200.o` and the dispatcher supports `gfx1200`
- `smallpt` has repository-specific CLI details already captured in `.github/skills/run-smallpt/SKILL.md`: it accepts `--arch` and `--nb_samples`, and it writes `image.png` at the repository root.
- Register files are stored as flattened `(elem, register)` arrays in both processor implementations. The RDNA path uses `aligned_vec::AVec` for aligned storage, which is part of the performance-sensitive design rather than an incidental implementation detail.
- The RDNA processor is JIT-oriented by default (`USE_INTERPRETER = false` in `src/rdna_processor.rs`), so RDNA changes often span decoding, processor execution, and LLVM translation together. GCN changes stay within the decoder/interpreter path.
