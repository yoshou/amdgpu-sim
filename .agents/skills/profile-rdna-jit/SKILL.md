---
name: profile-rdna-jit
description: Profile AMDGPU RDNA LLVM JIT native code in this repository and map perf samples back to RDNA block PCs. Use when Codex must identify RDNA JIT bottlenecks with data, inspect optimized native code, compare before/after JIT speedups, or create a rigorous performance report without leaving permanent profiling hooks, environment-variable behavior, or unrelated source changes.
---

# Profile RDNA JIT

Use this skill to produce data-backed JIT optimization evidence:

```text
perf cycles sample -> JIT runtime address -> native object offset -> DWARF synthetic line -> RDNA block PC -> native/RDNA disassembly
```

The profiling instrumentation is temporary. Do not leave environment-variable gates, JIT dumping hooks, or changed runtime behavior in the repository after the profiling run.

## Workflow

1. Check the worktree:
   - Run `git status --short`.
   - Note existing user changes and do not revert them.
   - If `src/rdna_translator/mod.rs` already has user changes, inspect them before editing.

2. Apply temporary native-mapping instrumentation:
   - Read [temporary-instrumentation.md](references/temporary-instrumentation.md).
   - Edit `src/rdna_translator/mod.rs` with `apply_patch`.
   - Do not add env-var switches. While the patch is present, profiling is active unconditionally.
   - Emit artifacts under `/tmp/amdgpu-sim-jit-profile-$PID` and `/tmp/perf-$PID.map`.

3. Build and run the requested profile:
   - Build the requested example or binary in release mode.
   - Preserve and restore any output files that the workload overwrites.
   - Use the sample count requested by the user or required by the measurement goal.
   - Run the workload under `perf record`:

```bash
rm -rf /tmp/amdgpu-sim-jit-profile-* /tmp/perf-*.map /tmp/rdna-jit.perf.data
perf record -F 199 -g --call-graph fp \
  -o /tmp/rdna-jit.perf.data \
  <workload-command>
```

4. Generate the native/block report:

```bash
python3 .agents/skills/profile-rdna-jit/scripts/analyze_jit_perf.py \
  --perf-data /tmp/rdna-jit.perf.data \
  --perf-map /tmp/perf-<pid>.map \
  --jit-object /tmp/amdgpu-sim-jit-profile-<pid>/<jit-symbol>.o \
  --output /tmp/rdna-jit-profile-report.md
```

Add `--kernel-object <amdgpu-kernel-object>` when the original AMDGPU object is available.

5. Collect global PMU indicators:

```bash
perf stat -d -d -d -o /tmp/rdna-jit.stat \
  <workload-command>
```

6. Remove temporary instrumentation:
   - Reverse only the profiling edits in `src/rdna_translator/mod.rs`.
   - Keep generated reports if the user asked for them.
   - Verify `git diff -- src/rdna_translator/mod.rs` no longer contains profiling hooks.
   - Verify no persistent env-var behavior such as `AMDGPU_SIM_JIT_MAP` remains.

7. Report optimization evidence:
   - Lead with measured bottlenecks, not hypotheses.
   - Include top RDNA blocks, sample counts, native offsets, native mnemonics, and relevant RDNA disassembly.
   - Separate facts from candidate explanations.
   - Propose A/B experiments with explicit success metrics: block samples, native instruction shape, and global cycles.

## Interpretation Rules

- Do not use RDNA opcode counts or LLVM IR instruction counts as performance evidence.
- Use cycles or stall-like PMU data for hotspot evidence.
- Use native disassembly to identify instruction sequences such as gathers, stack traffic, calls, scalarized loops, or vector math.
- For global changes such as memory layout, compare full block sample distributions before/after, not only the top block.
- Treat sample count selection as part of the measurement plan, not as a skill invariant.

## Script Notes

`scripts/analyze_jit_perf.py` expects:

- a `perf.data` captured while the temporary instrumentation was active,
- the matching `/tmp/perf-$PID.map`,
- the dumped JIT object for the same PID,
- optionally the original AMDGPU kernel object.

It emits a Markdown summary with top RDNA blocks, top native offsets, symbolizer evidence, and disassembly snippets.
