#include <hip/hip_runtime.h>

__device__ void wave_sync() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

extern "C" __global__ void lds_first_lane(unsigned* out, const unsigned* in, unsigned mix) {
  __shared__ unsigned s[256];
  unsigned t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
  s[t] = in[g] * 3u + 1u;
  wave_sync();
  unsigned v = s[t ^ mix];
  out[g] = v + __builtin_amdgcn_readfirstlane(v);
}

extern "C" __global__ __launch_bounds__(32) void lds_reverse(unsigned* out, const unsigned* in) {
  __shared__ unsigned s[32];
  unsigned t = threadIdx.x, g = blockIdx.x * 32 + t;
  s[t] = in[g] * 5u + 2u;
  __syncthreads();
  out[g] = s[31 - t];
}

extern "C" __global__ __launch_bounds__(32) void global_swap(unsigned* out, unsigned* tmp, const unsigned* in) {
  unsigned t = threadIdx.x, g = blockIdx.x * 32 + t;
  tmp[g] = in[g] * 2u + 1u;
  __syncthreads();
  out[g] = tmp[blockIdx.x * 32 + (t ^ 1)];
}

extern "C" __global__ __launch_bounds__(32) void wave_scan(unsigned* out, const unsigned* in) {
  __shared__ unsigned s[32];
  unsigned t = threadIdx.x, g = blockIdx.x * 32 + t;
  s[t] = in[g];
  for (unsigned d = 1; d < 32; d <<= 1) {
    wave_sync();
    unsigned v = t >= d ? s[t - d] : 0u;
    wave_sync();
    s[t] += v;
  }
  wave_sync();
  out[g] = s[t];
}

extern "C" __global__ __launch_bounds__(32) void block_sum(unsigned* out, const unsigned* in) {
  __shared__ unsigned total;
  unsigned t = threadIdx.x, g = blockIdx.x * 32 + t;
  if (t == 0) total = 0u;
  __syncthreads();
  atomicAdd(&total, in[g]);
  __syncthreads();
  if (t == 31) out[blockIdx.x] = total;
}

extern "C" __global__ __launch_bounds__(32) void divergent_swap(unsigned* out, const unsigned* in) {
  __shared__ unsigned s[32];
  unsigned t = threadIdx.x, g = blockIdx.x * 32 + t;
  unsigned v = in[g];
  if (v & 1u) {
    s[t] = v + 7u;
  } else {
    s[t] = v * 3u;
  }
  __syncthreads();
  out[g] = s[t ^ 2];
}
