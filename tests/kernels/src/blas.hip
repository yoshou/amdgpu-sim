#include <hip/hip_runtime.h>

extern "C" __global__ void saxpy(float* y, const float* x, float a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

#define TS 16
extern "C" __global__ void sgemm_tiled(float* C, const float* A, const float* B,
                                       int M, int N, int K) {
  __shared__ float As[TS][TS], Bs[TS][TS];
  int row = blockIdx.y * TS + threadIdx.y;
  int col = blockIdx.x * TS + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (K + TS - 1) / TS; ++t) {
    int ak = t * TS + threadIdx.x, bk = t * TS + threadIdx.y;
    As[threadIdx.y][threadIdx.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < TS; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

extern "C" __global__ void matvec(float* y, const float* A, const float* x, int m, int n) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= m) return;
  float acc = 0.0f;
  for (int c = 0; c < n; ++c) acc = fmaf(A[r * n + c], x[c], acc);
  y[r] = acc;
}

extern "C" __global__ void dot_product(float* out, const float* a, const float* b, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  float acc = 0.0f;
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x) {
    acc += a[i] * b[i];
  }
  s[t] = acc;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (t < d) s[t] += s[t + d];
    __syncthreads();
  }
  if (t == 0) atomicAdd(out, s[0]);
}

extern "C" __global__ void spmv_csr(float* y, const int* rp, const int* ci,
                                    const float* v, const float* x, int rows) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float acc = 0.0f;
  for (int j = rp[r]; j < rp[r + 1]; ++j) acc += v[j] * x[ci[j]];
  y[r] = acc;
}

#include <hip/hip_fp16.h>
extern "C" __global__ void dot_half(float* out, const __half* a, const __half* b, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  float acc = 0.0f;
  for (int i = blockIdx.x * blockDim.x + t; i < n; i += gridDim.x * blockDim.x) {
    acc += __half2float(a[i]) * __half2float(b[i]);
  }
  s[t] = acc;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (t < d) s[t] += s[t + d];
    __syncthreads();
  }
  if (t == 0) atomicAdd(out, s[0]);
}
