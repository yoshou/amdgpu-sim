#include <hip/hip_runtime.h>

extern "C" __global__ void bitonic_step(int* a, int j, int k) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned l = i ^ j;
  if (l > i) {
    bool up = ((i & k) == 0);
    if ((a[i] > a[l]) == up) { int t = a[i]; a[i] = a[l]; a[l] = t; }
  }
}

extern "C" __global__ void histogram(unsigned* bins, const unsigned char* data, int n) {
  __shared__ unsigned local[256];
  int t = threadIdx.x;
  for (int i = t; i < 256; i += blockDim.x) local[i] = 0;
  __syncthreads();
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x) {
    atomicAdd(&local[data[i]], 1u);
  }
  __syncthreads();
  for (int i = t; i < 256; i += blockDim.x) atomicAdd(&bins[i], local[i]);
}

extern "C" __global__ void compact(int* out, int* count, const int* in, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int v = (i < n) ? in[i] : 0;
  bool keep = (i < n) && (v > 0);
  unsigned long long m = __ballot(keep);
  unsigned lane = threadIdx.x & 31u;
  int rank = __popcll(m & ((1ull << lane) - 1ull));
  int base = 0;
  if (lane == 0) base = atomicAdd(count, __popcll(m));
  base = __shfl(base, 0, 32);
  if (keep) out[base + rank] = v;
}

extern "C" __global__ void count_matches(int* count, const int* a, const int* b, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g < n && a[g] == b[g]) atomicAdd(count, 1);
}
