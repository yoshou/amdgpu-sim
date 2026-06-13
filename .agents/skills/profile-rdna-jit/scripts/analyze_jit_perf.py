#!/usr/bin/env python3
"""Summarize RDNA JIT native perf samples by synthetic RDNA block lines."""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import subprocess
import sys
from typing import Iterable


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)


def parse_perf_map(path: pathlib.Path, symbol: str | None) -> tuple[int, int, str]:
    rows: list[tuple[int, int, str]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append((int(parts[0], 16), int(parts[1], 16), parts[2]))
    if not rows:
        raise SystemExit(f"empty perf map: {path}")
    if symbol:
        for row in rows:
            if row[2] == symbol:
                return row
        raise SystemExit(f"symbol {symbol!r} not found in {path}")
    if len(rows) > 1:
        rows.sort(key=lambda row: row[1], reverse=True)
    return rows[0]


def parse_sample_offsets(perf_data: pathlib.Path, base: int, size: int, symbol: str) -> list[int]:
    script = run(["perf", "script", "-i", str(perf_data), "-F", "ip,sym,dso"])
    offsets: list[int] = []
    for line in script.splitlines():
        if symbol not in line:
            continue
        match = re.search(r"\b([0-9a-fA-F]{8,16})\b", line)
        if not match:
            continue
        ip = int(match.group(1), 16)
        if base <= ip < base + size:
            offsets.append(ip - base)
    return offsets


def chunks(values: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def symbolize_offsets(jit_object: pathlib.Path, offsets: list[int]) -> dict[int, int]:
    unique = sorted(set(offsets))
    result: dict[int, int] = {}
    for chunk in chunks(unique, 512):
        out = run(["llvm-symbolizer-20", f"--obj={jit_object}", *[hex(v) for v in chunk]])
        lines = out.splitlines()
        for off, i in zip(chunk, range(0, len(lines), 3)):
            loc = lines[i + 1] if i + 1 < len(lines) else ""
            match = re.search(r":(\d+):\d+$", loc)
            result[off] = int(match.group(1)) if match else 0
    return result


def disassemble(path: pathlib.Path, start: int, stop: int, line_numbers: bool = False) -> str:
    cmd = ["llvm-objdump-20", "-d"]
    if line_numbers:
        cmd.append("--line-numbers")
    cmd += [f"--start-address=0x{start:x}", f"--stop-address=0x{stop:x}", str(path)]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)


def write_report(args: argparse.Namespace) -> str:
    base, size, symbol = parse_perf_map(args.perf_map, args.symbol)
    offsets = parse_sample_offsets(args.perf_data, base, size, symbol)
    if not offsets:
        raise SystemExit(f"no samples found for {symbol} in {args.perf_data}")

    off_to_line = symbolize_offsets(args.jit_object, offsets)
    block_counts = collections.Counter(off_to_line[off] for off in offsets)
    offset_counts = collections.Counter(offsets)

    lines: list[str] = []
    lines.append("# RDNA JIT Native Profile")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- perf data: `{args.perf_data}`")
    lines.append(f"- perf map: `{args.perf_map}`")
    lines.append(f"- JIT object: `{args.jit_object}`")
    if args.kernel_object:
        lines.append(f"- AMDGPU kernel object: `{args.kernel_object}`")
    lines.append(f"- symbol: `{symbol}`")
    lines.append(f"- runtime base: `0x{base:x}`")
    lines.append(f"- symbol size: `0x{size:x}`")
    lines.append(f"- JIT samples: `{len(offsets)}`")
    lines.append("")

    lines.append("## Top RDNA Blocks")
    lines.append("")
    lines.append("| samples | RDNA PC | synthetic line |")
    lines.append("|---:|---:|---:|")
    for line, count in block_counts.most_common(args.top_blocks):
        pc = "unknown" if line <= 0 else f"`0x{line:x}`"
        lines.append(f"| {count} | {pc} | `{line}` |")
    lines.append("")

    lines.append("## Top Native Offsets")
    lines.append("")
    lines.append("| samples | native offset | RDNA PC |")
    lines.append("|---:|---:|---:|")
    for off, count in offset_counts.most_common(args.top_offsets):
        line = off_to_line.get(off, 0)
        pc = "unknown" if line <= 0 else f"`0x{line:x}`"
        lines.append(f"| {count} | `0x{off:x}` | {pc} |")
    lines.append("")

    lines.append("## Native Disassembly Snippets")
    lines.append("")
    for off, count in offset_counts.most_common(args.snippets):
        start = max(0, off - args.native_window_before)
        stop = off + args.native_window_after
        rdna_line = off_to_line.get(off, 0)
        rdna_pc = "unknown" if rdna_line <= 0 else f"0x{rdna_line:x}"
        lines.append(f"### offset 0x{off:x}, samples {count}, RDNA PC {rdna_pc}")
        lines.append("")
        lines.append("```asm")
        lines.append(disassemble(args.jit_object, start, stop, line_numbers=True).strip())
        lines.append("```")
        lines.append("")

    if args.kernel_object:
        lines.append("## RDNA Disassembly Snippets")
        lines.append("")
        for line, count in block_counts.most_common(args.rdna_snippets):
            if line <= 0:
                continue
            start = max(0, line - args.rdna_window_before)
            stop = line + args.rdna_window_after
            lines.append(f"### RDNA PC 0x{line:x}, samples {count}")
            lines.append("")
            lines.append("```asm")
            lines.append(disassemble(args.kernel_object, start, stop).strip())
            lines.append("```")
            lines.append("")

    lines.append("## Next Experiments")
    lines.append("")
    lines.append("- For gather-heavy blocks, measure address uniformity/stride and test scalar-load or contiguous-load lowering only if the measured pattern supports it.")
    lines.append("- For stack-heavy native snippets, inspect register pressure, live vector state, and call/spill boundaries before changing caches or layout.")
    lines.append("- For global layout changes, compare the whole block distribution before/after, not only the top block.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf-data", type=pathlib.Path, required=True)
    parser.add_argument("--perf-map", type=pathlib.Path, required=True)
    parser.add_argument("--jit-object", type=pathlib.Path, required=True)
    parser.add_argument("--kernel-object", type=pathlib.Path)
    parser.add_argument("--symbol")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--top-blocks", type=int, default=12)
    parser.add_argument("--top-offsets", type=int, default=16)
    parser.add_argument("--snippets", type=int, default=3)
    parser.add_argument("--rdna-snippets", type=int, default=3)
    parser.add_argument("--native-window-before", type=lambda x: int(x, 0), default=0)
    parser.add_argument("--native-window-after", type=lambda x: int(x, 0), default=0x90)
    parser.add_argument("--rdna-window-before", type=lambda x: int(x, 0), default=0x20)
    parser.add_argument("--rdna-window-after", type=lambda x: int(x, 0), default=0x120)
    args = parser.parse_args()

    for path in [args.perf_data, args.perf_map, args.jit_object]:
        if not path.exists():
            raise SystemExit(f"missing input: {path}")
    if args.kernel_object and not args.kernel_object.exists():
        raise SystemExit(f"missing input: {args.kernel_object}")

    report = write_report(args)
    if args.output:
        args.output.write_text(report)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
