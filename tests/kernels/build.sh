#!/bin/bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
rocm=${ROCM_PATH:-/opt/rocm}
arch=${1:-gfx1200}

if [ ! -x "$rocm/lib/llvm/bin/clang" ]; then
  echo "no ROCm clang at $rocm/lib/llvm/bin/clang; set ROCM_PATH" >&2
  exit 1
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

cat "$here"/src/*.hip | grep -v '^#include <hip/hip_runtime.h>$' > "$tmp/all.hip.body"
{ echo '#include <hip/hip_runtime.h>'; cat "$tmp/all.hip.body"; } > "$tmp/all.hip"

"$rocm/lib/llvm/bin/clang" \
  -x hip --offload-arch="$arch" --offload-device-only \
  --no-offload-new-driver -fno-gpu-rdc -O2 --rocm-path="$rocm" \
  -o "$tmp/bundle.o" "$tmp/all.hip"

"$rocm/lib/llvm/bin/clang-offload-bundler" --unbundle --type=o \
  --targets="hipv4-amdgcn-amd-amdhsa--$arch" \
  --input="$tmp/bundle.o" --output="$here/../data/kernels_$arch.o"

echo "wrote tests/data/kernels_$arch.o"
