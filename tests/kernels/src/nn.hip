#include <hip/hip_runtime.h>

extern "C" __global__ void softmax(float* out, const float* in, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  float m = -INFINITY;
  for (int i = t; i < n; i += blockDim.x) m = fmaxf(m, in[i]);
  s[t] = m;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (t < d) s[t] = fmaxf(s[t], s[t + d]);
    __syncthreads();
  }
  m = s[0];
  __syncthreads();
  float sum = 0.0f;
  for (int i = t; i < n; i += blockDim.x) sum += expf(in[i] - m);
  s[t] = sum;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (t < d) s[t] += s[t + d];
    __syncthreads();
  }
  sum = s[0];
  for (int i = t; i < n; i += blockDim.x) out[i] = expf(in[i] - m) / sum;
}

extern "C" __global__ void layernorm(float* o, const float* i, int n) {
  __shared__ float s[256];
  int t = threadIdx.x;
  const float* row = i + blockIdx.x * n;
  float sum = 0.0f;
  for (int k = t; k < n; k += blockDim.x) sum += row[k];
  s[t] = sum;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (t < d) s[t] += s[t + d];
    __syncthreads();
  }
  float mean = s[0] / n;
  __syncthreads();
  float var = 0.0f;
  for (int k = t; k < n; k += blockDim.x) { float d = row[k] - mean; var += d * d; }
  s[t] = var;
  __syncthreads();
  for (int d = 128; d; d >>= 1) {
    if (t < d) s[t] += s[t + d];
    __syncthreads();
  }
  float rs = rsqrtf(s[0] / n + 1e-5f);
  for (int k = t; k < n; k += blockDim.x) o[blockIdx.x * n + k] = (row[k] - mean) * rs;
}

extern "C" __global__ void gelu(float* o, const float* i, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  float x = i[g];
  o[g] = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void relu_bwd(float* dx, const float* dy, const float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
}

#include <hip/hip_fp16.h>
extern "C" __global__ void axpy_half2(__half2* y, const __half2* x, float a, int n2) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) return;
  __half2 a2 = __float2half2_rn(a);
  y[i] = __hfma2(a2, x[i], y[i]);
}

extern "C" __global__ void half_to_float(float* o, const __half* i, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g < n) o[g] = __half2float(i[g]);
}

extern "C" __global__ void float_to_half(__half* o, const float* i, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g < n) o[g] = __float2half(i[g]);
}
