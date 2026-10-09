# Agent guide for `amdgpu-sim`

`amdgpu-sim` runs AMD GPU kernels (HSA code objects) on the CPU. It targets gfx1200 (RDNA4) and gfx803 (GCN3). `README.md` describes the project for users; this file is for agents that change the code.

## Environment

- The crate links LLVM 22 through `llvm-sys = "221.0.1"` with `llvm-sys/prefer-dynamic`. Point `LLVM_SYS_221_PREFIX` at an LLVM 22 installation.
- The devcontainer (`.devcontainer/Dockerfile`) is the reference environment. It installs `clang-22`, `lld-22`, `llvm-22` and `llvm-22-dev` from apt.llvm.org, sets `LLVM_SYS_221_PREFIX=/usr/lib/llvm-22` and puts its `bin` on `PATH`.
- `tests/gcn3` runs `llvm-mc` and `ld.lld` from `$LLVM_SYS_221_PREFIX/bin` at test time, so `lld-22` must be installed.

## Build and test

- CI (`.github/workflows/ci.yml`) runs a debug build on Ubuntu 24.04:
  - `cargo build --locked --all-targets`
  - `cargo test --locked`
- Run those commands before committing. Debug builds check integer overflow, so release-only runs can miss failures that CI reports.
- Use `cargo test --release` while iterating; it is much faster. Approximate test times on the 32-thread devcontainer host, release / debug:
  - lib: 30 s / 4 min
  - `isa`: 2 min / 3.5 min
  - `gcn3`: 7 s / 10 s
  - each `kernels*` target: 3 s / 10 s
- Filter tests:
  - unit tests: `cargo test --release --lib -- <module::path>`
  - integration tests: `cargo test --release --test <target> <name>`, where the targets are `isa`, `gcn3`, `kernels`, `kernels64` and `kernels803`
- No linter or formatter is part of the workflow. The tree is not rustfmt-clean, so never run `cargo fmt` or `cargo clippy --fix` over it.

## Running examples

- The host programs in `examples/<name>/main.rs` are the canonical way to run the simulator and show the integration pattern. Pass options after `--`, for example `cargo run --release --example smallpt_spmd -- --arch gfx1200 --nb_samples 4`.
- Examples open `examples/<name>/kernel_<arch>.o` (and the raytracing scene files) by relative path, so the working directory must contain `examples`.
- These examples write `image.png` into the working directory: `smallpt`, `smallpt_spmd`, `raytracing` and `raytracing_spmd`.
  - The `image.png` at the repository root belongs to the maintainer; never overwrite or commit it.
  - Run these examples from a scratch directory with an `examples` symlink to the repository's `examples`.

| Example | Engine | Default `--arch` | Also runs | Extra options |
|---|---|---|---|---|
| `smallpt` | interpreter on gfx803, LLVM JIT on gfx1200 | gfx803 | gfx1200 | `--nb_samples N` |
| `bitonic_sort` | interpreter on gfx803, LLVM JIT on gfx1200 | gfx803 | gfx1200 | `--sort inc\|dec`, `--log2length L` |
| `histogram`, `texture` | interpreter on gfx803, LLVM JIT on gfx1200 | gfx803 | gfx1200 | |
| `raytracing`, `warp_shuffle` | LLVM JIT | gfx1200 | | |
| `simple_hgemm` | LLVM JIT | gfx942, which has no kernel; pass `--arch gfx1200` | | |
| `smallpt_spmd` | SPMD JIT | gfx803 | gfx1200 | `--nb_samples N` |
| `bitonic_sort_spmd` | SPMD JIT | gfx1200 | gfx803 | `--sort inc\|dec`, `--log2length L` |
| `histogram_spmd`, `texture_spmd` | SPMD JIT | gfx1200 | gfx803 | |
| `raytracing_spmd`, `warp_shuffle_spmd` | SPMD JIT | gfx1200 | | |
| `simple_hgemm_spmd` | SPMD JIT | gfx1200 | | `--verify_widths` runs every width and requires bitwise-identical results |

- Every `*_spmd` example also takes `--num_threads N` and `--vec_width W`.
  - `--num_threads` defaults to the available parallelism.
  - `--vec_width W` packs W work-items into each SIMD vector. The SPMD JIT accepts 1, 2, 4, 8, 16 and 32. The default is 16 in `smallpt_spmd`, `texture_spmd` and `raytracing_spmd`, and 1 in the others.

## Architecture

### Crate layout

- `src/lib.rs` exports the older engines and the SPMD backend:
  - shared ABI types: `processor.rs` (`KernelDescriptor`, `Pointer`, `HsaKernelDispatchPacket`), plus `buffer.rs`, `bit.rs` and `instructions.rs`
  - gfx803 interpreter: `gcn3_decoder.rs`, `gcn_instructions.rs` and `gcn_processor.rs` (`GCNProcessor`)
  - gfx1200 interpreter and LLVM JIT:
    - `rdna4_decoder.rs`, `rdna_instructions.rs` and `rdna_processor.rs` (`RDNAProcessor`)
    - `Engine::Interpreter` or `Engine::LlvmJit`; `new` uses the LLVM JIT
    - `rdna_translator/` translates RDNA code to LLVM IR
  - `rdna_spmd/`: the SPMD JIT backend
- Register files are flattened `(elem, register)` arrays. The RDNA path stores them in `aligned_vec::AVec` on purpose, for speed.

### SPMD backend (`src/rdna_spmd`)

- The public API lives in `runtime`: `Module`, `Function`, `Launch`, `Arg`, `Buffer`, `Pod` and `Error`. The `*_spmd` examples load a `Module`, look up a `Function` and call `launch`. `launch` compiles on first use; `prepare` compiles ahead of time, which lets an example time the JIT apart from the run.
- `compiler.rs` drives the pipeline:
  1. `decode_program` lifts the kernel into the SSA IR in `ir/`.
     - gfx1200 goes through `rdna4/`.
     - gfx803 goes through `gcn3/`. It tracks scalar state (M0, MODE, the scratch descriptor, call targets and return addresses, VGPR lanes written by `v_writelane`) and inlines every call site.
     - The wave passes in `pass/` then run to a fixpoint: PrivateSlots, Halves, Idioms, UniformQueries, BranchSelects, Simplify and Dce.
  2. `Jit::kernel` runs `Hazards::find` whenever the launch environment changes. This is the address analysis of memory accesses between lanes, in `decompile/address/` and `decompile/hazard.rs`.
  3. `decompile` proves a conversion policy, which wave queries, lane words and meetings the lane program keeps. It then rewrites the wave program into a lane program (`rewrite.rs`, `fold.rs`). The result is cached per distinct set of hazards.
  4. `lockstep/` packs W lanes into a packet whose values are SIMD vectors. `codegen/` emits LLVM IR per region, and `native/jit.rs` optimizes it with the O3 pipeline and compiles it through ORC LLJIT. Kernels are cached per lane program, packet width and workgroup x size.
  5. `engine/` runs the regions. The scheduler unit is the workgroup when the kernel uses barriers or LDS, the wave when it uses wave operations, and the packet otherwise.
- The decompile proof:
  - `search.rs` (the default) adds the choices blamed for each first violation, then drops redundant choices with parallel trials.
  - `direct.rs` (`AMDGPU_SIM_PROOF=direct`) runs one check over all choices at once. It is much slower on large kernels and can keep a different policy.
  - `check/` propagates the difference between the wave and lane programs. Its main parts are `lattice`, `rules`, `masks`, `orderings`, and `explore`, which follows detours.
  - `logic/` encodes that difference as BDDs. `kernel/` holds atoms, edge images and the policy; `rules/` holds bits, cells, answers and words.
  - `analysis/bdd.rs` is the BDD manager.
- The BDD variable order is set in `logic/kernel/atoms.rs` (`Atoms::number`) and dominates proof time. Change it only with measurements and identical results.

### Tests

- Unit tests live in `mod tests` blocks and `tests.rs` files under `src/`. Randomized oracle tests read knobs such as `ORACLE_SEED` and `ORACLE_SCALE`.
- `tests/isa` covers RDNA4 instructions.
  - Expected values were captured on gfx1200 hardware.
  - Each test runs on the interpreter, the LLVM JIT and SPMD widths 1 to 32, using `tests/data/harness_gfx1200.o`.
- `tests/kernels`, `tests/kernels64` and `tests/kernels803` run a HIP corpus through the SPMD backend at widths 1 to 32.
  - The targets are gfx1200 wave32, gfx1200 wave64 and gfx803 respectively.
  - The sources are `tests/kernels/src/*.hip` and the checks are in `tests/kernels/cases.rs`.
  - The objects are prebuilt in `tests/data`. Rebuild them with `tests/kernels/build.sh [arch] [wave64]`, which needs ROCm clang (`ROCM_PATH`, default `/opt/rocm`).
- `tests/gcn3` assembles and links GCN3 instruction probes at test time. It runs them through the SPMD backend at every width and compares the results with reference functions written in Rust.

## Debugging switches

| Variable | Effect |
|---|---|
| `AMDGPU_SIM_PROOF=search\|direct` | decompile proof mode; the default is `search` |
| `AMDGPU_SIM_PRINT_IR` | print the IR after each pass and verify it, and print how many choices the lane program keeps |
| `AMDGPU_SIM_PRINT_MEETINGS` | print each blamed choice with its violation, and the access pairs whose order may matter with the meetings kept |
| `AMDGPU_SIM_DUMP_LLVM=<dir>` | write the LLVM IR before and after optimization |
| `AMDGPU_SIM_DUMP_CODE=<dir>` | write the memory mapping that holds the JIT-compiled code |
| `AMDGPU_SIM_OPT=0` | optimize with the O0 pipeline |
| `AMDGPU_SIM_BLOCK_COUNTS=<path>` | write executed block counts to `<path>.blocks` and `<path>.counts` |

Never read environment variables on hot paths. One check inside a BDD operation once doubled proof time.

## Skills

`.agents/skills` holds task recipes written for Codex:
- `run-smallpt`
- `commit-changes`
- `profile-rdna-jit`, which maps perf samples of the `rdna_translator` LLVM JIT, not the SPMD backend

Where a recipe disagrees with this file, follow this file. For example, `run-smallpt` runs smallpt from the repository root, which overwrites `image.png`.
