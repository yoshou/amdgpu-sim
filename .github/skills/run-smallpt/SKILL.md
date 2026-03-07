---
name: run-smallpt
description: Run the Rust smallpt example in this repository, especially when asked to execute or benchmark smallpt on a specific AMDGPU architecture such as gfx1200, adjust the sample count, or confirm the output image generation command.
---

# Run smallpt

Use this skill to execute the `smallpt` example from the repository root.

## Confirm the CLI shape before running

- Use `cargo run --example smallpt -- ...`.
- Pass the architecture with `--arch`, not `--target`.
- Pass the sample count with `--nb_samples`.
- Expect the renderer to write `image.png` in the repository root.

## Default command for gfx1200

Run:

```bash
cargo run --example smallpt -- --arch gfx1200
```

## Override the sample count

When the user specifies a sample count, append `--nb_samples <NUM>`:

```bash
cargo run --example smallpt -- --arch gfx1200 --nb_samples 256
```

## Repository-specific notes

- The smallpt example parses `--arch` and `--nb_samples` in `examples/smallpt/main.rs`.
- The example loads a target-specific kernel object from `examples/smallpt/kernel_<arch>.o`.
- This repository already includes `examples/smallpt/kernel_gfx1200.o`, so `gfx1200` is a valid target.

## Report the result

- Summarize whether the command succeeded.
- Include the sample count and architecture used.
- If useful, mention the elapsed time reported by the program.
