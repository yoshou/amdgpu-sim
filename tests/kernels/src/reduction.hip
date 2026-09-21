#include <hip/hip_runtime.h>

extern "C" __global__ void reduce_sum(float* out, const float* in, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  int i = blockIdx.x * blockDim.x + t;
  s[t] = (i < n) ? in[i] : 0.0f;
  __syncthreads();
  for (int d = blockDim.x / 2; d > 0; d >>= 1) {
    if (t < d) s[t] += s[t + d];
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = s[0];
}

extern "C" __global__ void argmax(int* oi, float* ov, const float* in, int n) {
  __shared__ float bv[256];
  __shared__ int bi[256];
  int t = threadIdx.x;
  float best = -INFINITY;
  int bidx = -1;
  for (int k = t; k < n; k += blockDim.x) {
    if (in[k] > best) { best = in[k]; bidx = k; }
  }
  bv[t] = best;
  bi[t] = bidx;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (t < d && bv[t + d] > bv[t]) { bv[t] = bv[t + d]; bi[t] = bi[t + d]; }
    __syncthreads();
  }
  if (t == 0) { *ov = bv[0]; *oi = bi[0]; }
}

__device__ float warp_reduce(float v) {
  for (int d = 16; d; d >>= 1) v += __shfl_down(v, d, 32);
  return v;
}

extern "C" __global__ void warp_reduce_sum(float* o, const float* i, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (g < n) ? i[g] : 0.0f;
  v = warp_reduce(v);
  if ((threadIdx.x & 31) == 0) atomicAdd(o, v);
}

extern "C" __global__ void scan_block(int* o, const int* i, int n) {
  __shared__ int s[256];
  int t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
  s[t] = (g < n) ? i[g] : 0;
  __syncthreads();
  for (int d = 1; d < blockDim.x; d <<= 1) {
    int v = (t >= d) ? s[t - d] : 0;
    __syncthreads();
    s[t] += v;
    __syncthreads();
  }
  if (g < n) o[g] = s[t];
}

extern "C" __global__ void prefix_max_2d(float* o, const float* i, int w, int h) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= h) return;
  float m = -INFINITY;
  for (int x = 0; x < w; ++x) { m = fmaxf(m, i[y * w + x]); o[y * w + x] = m; }
}
