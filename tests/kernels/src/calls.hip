#include <hip/hip_runtime.h>

__device__ __noinline__ unsigned mix_once(unsigned x) {
  return x * 2654435761u ^ (x >> 7);
}

__device__ __noinline__ unsigned mix_twice(unsigned x, unsigned y) {
  unsigned a = mix_once(x);
  unsigned b = mix_once(y + 1u);
  return a + b;
}

__device__ __noinline__ unsigned sum_table(unsigned x) {
  unsigned t[8];
  for (unsigned i = 0; i < 8u; i++) t[i] = mix_once(x + i);
  unsigned s = 0;
  for (unsigned i = 0; i < 8u; i++) s += t[(i * 3u) & 7u];
  return s;
}

extern "C" __global__ void calls(unsigned* out, const unsigned* in) {
  unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned v = in[g];
  unsigned r;
  if (v & 1u) {
    r = mix_twice(v, g);
  } else {
    r = mix_once(v);
  }
  out[g] = r + sum_table(v & 15u);
}
