#include <hip/hip_runtime.h>

extern "C" __global__ void mandelbrot(unsigned* o, int w, int h, int maxit) {
  int px = blockIdx.x * blockDim.x + threadIdx.x;
  int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= w || py >= h) return;
  float cr = (px / (float)w) * 3.5f - 2.5f, ci = (py / (float)h) * 2.0f - 1.0f;
  float zr = 0, zi = 0;
  int it = 0;
  while (zr * zr + zi * zi <= 4.0f && it < maxit) {
    float t = zr * zr - zi * zi + cr;
    zi = 2 * zr * zi + ci;
    zr = t;
    ++it;
  }
  o[py * w + px] = it;
}

extern "C" __global__ void nbody_step(float* opx, float* opy, const float* px, const float* py,
                                      const float* mass, int n, float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float ax = 0.0f, ay = 0.0f, xi = px[i], yi = py[i];
  for (int j = 0; j < n; ++j) {
    float dx = px[j] - xi, dy = py[j] - yi;
    float d2 = dx * dx + dy * dy + 1e-6f;
    float inv = rsqrtf(d2);
    float w = mass[j] * inv * inv * inv;
    ax += dx * w;
    ay += dy * w;
  }
  opx[i] = xi + ax * dt;
  opy[i] = yi + ay * dt;
}

extern "C" __global__ void u64_hash(unsigned long long* o, const unsigned long long* i, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  unsigned long long x = i[g];
  x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  o[g] = x;
}

extern "C" __global__ void clamp_short(short* o, const int* i, int n) {
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n) return;
  int v = i[g];
  o[g] = (short)(v < -32768 ? -32768 : (v > 32767 ? 32767 : v));
}

extern "C" __global__ void mixed_math(double* out, const double* in, unsigned n, unsigned stride) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned row = i / stride, col = i % stride;
  double v = in[i];
  out[i] = sqrt(v * v + 1.0) / (1.0 + (double)row) + sin((double)col);
}
