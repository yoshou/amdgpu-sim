#include <hip/hip_runtime.h>

extern "C" __global__ void atomics64(unsigned long long* cells, unsigned long long* out, const unsigned* in) {
  unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long v = (unsigned long long)in[g] * 0x100000001ull + 3ull;
  unsigned j = g % 8u;
  atomicAdd(&cells[j], v);
  atomicMax(&cells[8u + j], v);
  atomicMin(&cells[16u + j], v);
  atomicAnd(&cells[24u + j], ~v);
  atomicOr(&cells[32u + j], v);
  atomicXor(&cells[40u + j], v);
  atomicSub(&cells[48u + j], v);
  unsigned long long r = atomicExch(&cells[64u + g], v);
  r += atomicCAS(&cells[192u + g], (unsigned long long)g, v);
  r += atomicCAS(&cells[320u + g], 1ull, v);
  out[g] = r;
}

extern "C" __global__ void lds_atomics64(unsigned long long* out, const unsigned* in) {
  __shared__ unsigned long long s[16];
  unsigned t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
  unsigned long long v = (unsigned long long)in[g] * 0x100000001ull + 3ull;
  if (t < 16u) s[t] = (unsigned long long)(t + 1u) << 33;
  __syncthreads();
  atomicAdd(&s[t % 4u], v);
  atomicMax(&s[4u + t % 4u], v);
  atomicOr(&s[8u + t % 4u], v);
  unsigned long long r = 0;
  if (t < 4u) r = atomicCAS(&s[12u + t], (unsigned long long)(13u + t) << 33, v);
  __syncthreads();
  out[g] = r + s[t % 16u];
}

extern "C" __global__ void dpp_compare(unsigned* out, const unsigned* in) {
  unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned v = in[g];
  unsigned x = __builtin_amdgcn_update_dpp(0u, v, 0x111, 0xf, 0xf, false);
  unsigned y = __builtin_amdgcn_update_dpp(0u, v, 0x121, 0xf, 0xf, true);
  out[g * 2] = (x < v ? 1u : 0u) | (y == v ? 2u : 0u) | (y > x ? 4u : 0u);
  unsigned long long wide = (unsigned long long)x + v;
  out[g * 2 + 1] = (unsigned)(wide >> 32) + (unsigned)wide * 3u;
}

extern "C" __global__ void dpp_wide(unsigned long long* out, const unsigned long long* in) {
  unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long v = in[g];
  unsigned lo = __builtin_amdgcn_update_dpp(0u, (unsigned)v, 0x111, 0xf, 0xf, false);
  unsigned hi = __builtin_amdgcn_update_dpp(0u, (unsigned)(v >> 32), 0x111, 0xf, 0xf, false);
  unsigned long long x = ((unsigned long long)hi << 32) | lo;
  out[g] = (x << 5) + v;
}
