#include <hip/hip_runtime.h>

extern "C" __global__ void sync_or_and(unsigned* out, const unsigned* in) {
  unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned v = in[g];
  int c = __syncthreads_count((v & 1u) != 0u);
  int o = __syncthreads_or(v > 1000u);
  int a = __syncthreads_and(v < 5000u);
  out[g] = (unsigned)(c * 100 + o * 10 + a);
}

extern "C" __global__ void private_index(unsigned* out, const unsigned* in) {
  unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned a[12];
  for (int i = 0; i < 12; i++) a[i] = in[g] * (unsigned)(i + 1);
  unsigned k = in[g] % 12u;
  for (unsigned i = 0; i < 6u; i++) a[(k + i) % 12u] += a[(k + 3u * i) % 12u];
  out[g] = a[k] + a[(k + 4u) % 12u];
}

extern "C" __global__ void volatile_scan(unsigned* out, const unsigned* in) {
  volatile __shared__ unsigned s[64];
  unsigned t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
  s[t] = in[g];
  for (unsigned d = 1; d < warpSize; d <<= 1) {
    unsigned x = t >= d ? s[t - d] : 0u;
    s[t] += x;
  }
  out[g] = s[t];
}

extern "C" __global__ void dynamic_flat(unsigned* out, const unsigned* in) {
  extern __shared__ unsigned dyn_s[];
  volatile unsigned* s = dyn_s;
  unsigned t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
  s[t] = in[g];
  __syncthreads();
  unsigned acc = 0;
  for (unsigned i = 0; i <= t; i++) acc += s[i];
  out[g] = acc;
}

extern "C" __global__ void dpp_rows(unsigned* out, const unsigned* in) {
  unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
  int v = (int)in[g];
  out[g * 4 + 0] = __builtin_amdgcn_mov_dpp(v, 0xB1, 0xf, 0xf, false);
  out[g * 4 + 1] = __builtin_amdgcn_update_dpp(0xffff, v, 0x111, 0xf, 0xf, false);
  out[g * 4 + 2] = __builtin_amdgcn_mov_dpp(v, 0x123, 0xf, 0xf, false);
  out[g * 4 + 3] = __builtin_amdgcn_mov_dpp(v, 0x140, 0xf, 0xf, false);
}

extern "C" __global__ void lds_cas(unsigned* out, const unsigned* in) {
  __shared__ unsigned total;
  __shared__ unsigned largest;
  unsigned t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
  unsigned v = in[g];
  if (t == 0u) { total = 0u; largest = 0u; }
  __syncthreads();
  unsigned seen;
  do { seen = total; } while (atomicCAS(&total, seen, seen + v) != seen);
  atomicMax(&largest, v);
  __syncthreads();
  out[g] = t == 0u ? total : largest;
}

extern "C" __global__ void bit_atomics(unsigned* out, unsigned* cells, const unsigned* in) {
  unsigned g = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned v = in[g];
  unsigned r = atomicOr(&cells[g], v);
  r += atomicAnd(&cells[g], ~(v >> 1));
  r += atomicXor(&cells[g], v << 1);
  r += atomicExch(&cells[g], g);
  r += atomicInc(&cells[g], 100u);
  r += atomicDec(&cells[g], 100u);
  r += atomicSub(&cells[g], 3u);
  out[g] = r;
}
