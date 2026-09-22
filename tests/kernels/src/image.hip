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

extern "C" __global__ void conv2d(float* o, const float* i, const float* k, int w, int h, int ks) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  int r = ks / 2;
  float acc = 0.0f;
  for (int dy = 0; dy < ks; ++dy) {
    int sy = y + dy - r;
    if (sy < 0 || sy >= h) continue;
    for (int dx = 0; dx < ks; ++dx) {
      int sx = x + dx - r;
      if (sx < 0 || sx >= w) continue;
      acc += i[sy * w + sx] * k[dy * ks + dx];
    }
  }
  o[y * w + x] = acc;
}

extern "C" __global__ void stencil7_3d(float* o, const float* i, int nx, int ny, int nz) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x < 1 || y < 1 || z < 1 || x >= nx - 1 || y >= ny - 1 || z >= nz - 1) return;
  int c = (z * ny + y) * nx + x;
  o[c] = i[c] + 0.1f * (i[c - 1] + i[c + 1] + i[c - nx] + i[c + nx]
                        + i[c - nx * ny] + i[c + nx * ny] - 6.0f * i[c]);
}

extern "C" __global__ void median3x3(unsigned char* o, const unsigned char* i, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  unsigned char v[9];
  int n = 0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      int sx = min(max(x + dx, 0), w - 1), sy = min(max(y + dy, 0), h - 1);
      v[n++] = i[sy * w + sx];
    }
  }
  for (int a = 1; a < 9; ++a) {
    unsigned char t = v[a];
    int b = a - 1;
    while (b >= 0 && v[b] > t) { v[b + 1] = v[b]; --b; }
    v[b + 1] = t;
  }
  o[y * w + x] = v[4];
}
