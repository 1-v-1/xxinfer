// Reproduce the small-M NVFP4 decode access pattern exactly and measure its
// streaming rate.
//
// The decode kernel gives one warp per output row and walks the row in
// WARP_SIZE * ELEMS_PER_LANE strides, so each warp issues four rounds of
// 512 contiguous bytes with a full DRAM latency in between.  The earlier
// micro-benchmark streamed whole rows in one go and reported 866 GB/s, which
// is not what the kernel does.
//
// This probe compares:
//   v_full     the kernel's pattern: four rounds of 512 bytes per warp
//   v_whole    one contiguous 2 KiB read per warp (the optimistic case)
//   v_wide     four rounds of 1024 bytes per warp (32 lanes x 32 bytes)
//
// Build & run:
//   nvcc -O3 -arch=sm_70 -o nvfp4_decode_stream_probe nvfp4_decode_stream_probe.cu
//   ./nvfp4_decode_stream_probe

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(x)                                                              \
  do {                                                                        \
    cudaError_t e_ = (x);                                                     \
    if (e_ != cudaSuccess) {                                                  \
      std::printf("CUDA error %s at %d\n", cudaGetErrorString(e_), __LINE__); \
      return 1;                                                               \
    }                                                                         \
  } while (0)

// The kernel's pattern: 32 lanes x 16 bytes per round, four rounds.
__global__ void v_full(const uint8_t *__restrict__ w, float *out,
                       int k_bytes) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  const uint8_t *r = w + (size_t)row * k_bytes;
  int acc = 0;
  for (int base = 0; base < k_bytes; base += 512) {
    const uint2 a = *reinterpret_cast<const uint2 *>(r + base + lane * 16);
    const uint2 b = *reinterpret_cast<const uint2 *>(r + base + lane * 16 + 8);
    acc += a.x + a.y + b.x + b.y;
  }
  if (acc == 0x7fffffff) out[warp] = (float)acc;
}

// One contiguous read per warp.
__global__ void v_whole(const uint8_t *__restrict__ w, float *out,
                        int k_bytes) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  const uint8_t *r = w + (size_t)row * k_bytes;
  int acc = 0;
  for (int base = lane * 16; base < k_bytes; base += 32 * 16) {
    const uint2 a = *reinterpret_cast<const uint2 *>(r + base);
    const uint2 b = *reinterpret_cast<const uint2 *>(r + base + 8);
    acc += a.x + a.y + b.x + b.y;
  }
  if (acc == 0x7fffffff) out[warp] = (float)acc;
}

// Wider per-lane access: 32 lanes x 32 bytes per round.
__global__ void v_wide(const uint8_t *__restrict__ w, float *out, int k_bytes) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int row = blockIdx.x * 8 + warp;
  const uint8_t *r = w + (size_t)row * k_bytes;
  int acc = 0;
  for (int base = 0; base < k_bytes; base += 1024) {
    const uint4 a = *reinterpret_cast<const uint4 *>(r + base + lane * 32);
    const uint4 b = *reinterpret_cast<const uint4 *>(r + base + lane * 32 + 16);
    acc += a.x + a.y + a.z + a.w + b.x + b.y + b.z + b.w;
  }
  if (acc == 0x7fffffff) out[warp] = (float)acc;
}

template <typename F>
void bench(const char *label, F kernel, const uint8_t *d_w, float *d_out,
           int k_bytes, int rows) {
  const int threads = 256;
  const int blocks = rows / 8;
  cudaEvent_t a, b;
  cudaEventCreate(&a);
  cudaEventCreate(&b);
  for (int i = 0; i < 5; ++i) kernel<<<blocks, threads>>>(d_w, d_out, k_bytes);
  cudaDeviceSynchronize();
  cudaEventRecord(a);
  const int iters = 100;
  for (int i = 0; i < iters; ++i)
    kernel<<<blocks, threads>>>(d_w, d_out, k_bytes);
  cudaEventRecord(b);
  cudaEventSynchronize(b);
  float ms = 0;
  cudaEventElapsedTime(&ms, a, b);
  ms /= iters;
  const double bytes = (double)rows * k_bytes;
  std::printf("%-34s %8.4f ms  %8.1f GB/s\n", label, ms,
              bytes / (ms * 1e-3) / 1e9);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  const int k_bytes = 2048;  // K=4096 packed
  // The real decode walks many weight matrices that do not fit in the 6 MiB
  // L2, so the working set has to be large for the measurement to reflect
  // DRAM rather than cache.
  const int rows = 65536;    // 128 MiB of weights
  const size_t total = (size_t)k_bytes * rows;

  uint8_t *d_w = nullptr;
  float *d_out = nullptr;
  CHECK(cudaMalloc(&d_w, total));
  CHECK(cudaMalloc(&d_out, 1024 * sizeof(float)));
  CHECK(cudaMemset(d_w, 1, total));

  bench("kernel pattern (4x512B)", v_full, d_w, d_out, k_bytes, rows);
  bench("one contiguous 2KiB per warp", v_whole, d_w, d_out, k_bytes, rows);
  bench("wide (4x1KiB, 32B/lane)", v_wide, d_w, d_out, k_bytes, rows);

  cudaFree(d_w);
  cudaFree(d_out);
  return 0;
}
