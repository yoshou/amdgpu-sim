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

extern "C" __global__ void radix_count(unsigned* counts, const unsigned* keys, int n, int shift) {
  unsigned local[256];
  for (int d = 0; d < 256; ++d) local[d] = 0;
  int t = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
  for (int i = t; i < n; i += stride) local[(keys[i] >> shift) & 0xff]++;
  for (int d = 0; d < 256; ++d) counts[d * stride + t] = local[d];
}

extern "C" __global__ void lower_bound(int* out, const int* sorted, int m, const int* q, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  int key = q[g], lo = 0, hi = m;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (sorted[mid] < key) lo = mid + 1; else hi = mid;
  }
  out[g] = lo;
}

extern "C" __global__ void max_int(int* out, const int* in, int n) {
  __shared__ int s[256];
  int t = threadIdx.x;
  int m = INT_MIN;
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x) m = max(m, in[i]);
  s[t] = m;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (t < d) s[t] = max(s[t], s[t + d]);
    __syncthreads();
  }
  if (t == 0) atomicMax(out, s[0]);
}

extern "C" __global__ void max_float_cas(float* out, const float* in, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  float v = in[g];
  int* p = (int*)out;
  int old = *p, assumed;
  do {
    assumed = old;
    if (__int_as_float(assumed) >= v) break;
    old = atomicCAS(p, assumed, __float_as_int(v));
  } while (assumed != old);
}
