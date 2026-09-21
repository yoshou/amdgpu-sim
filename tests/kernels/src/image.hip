#include <hip/hip_runtime.h>

extern "C" __global__ void transpose_lds(float* o, const float* i, int w, int h) {
  __shared__ float t[32][33];
  int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
  if (x < w && y < h) t[threadIdx.y][threadIdx.x] = i[y * w + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  if (x < h && y < w) o[y * h + x] = t[threadIdx.x][threadIdx.y];
}

extern "C" __global__ void stencil5(float* o, const float* i, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < 1 || y < 1 || x >= w - 1 || y >= h - 1) return;
  o[y * w + x] = 0.2f * (i[y * w + x] + i[y * w + x - 1] + i[y * w + x + 1]
                       + i[(y - 1) * w + x] + i[(y + 1) * w + x]);
}

extern "C" __global__ void conv1d(float* o, const float* i, const float* k, int n, int ks) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= n) return;
  float acc = 0.0f;
  for (int j = 0; j < ks; ++j) {
    int p = x + j - ks / 2;
    if (p >= 0 && p < n) acc += i[p] * k[j];
  }
  o[x] = acc;
}

extern "C" __global__ void rgb_to_gray(unsigned char* o, const unsigned char* i, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  unsigned char r = i[g * 3], gr = i[g * 3 + 1], b = i[g * 3 + 2];
  o[g] = (unsigned char)((77 * r + 150 * gr + 29 * b) >> 8);
}
