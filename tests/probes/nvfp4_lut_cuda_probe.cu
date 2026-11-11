// CUDA probe: validates the NVFP4 software-dequant LUT semantics directly on
// the target GPU, and compares the candidate "f16 high-byte table" fast path
// against the production element-wise chain.
//
// Why a CUDA probe: the LUT is built from __byte_perm, whose selector
// semantics differ between the PTX `prmt` instruction (msb = sign extend)
// and the CUDA intrinsic (msb ignored).  Getting that wrong silently
// corrupts every dequantized weight, so the semantics are asserted on the
// real device instead of being inferred.
//
// Build & run:
//   nvcc -O2 -arch=sm_70 -o nvfp4_lut_cuda_probe nvfp4_lut_cuda_probe.cu
//   ./nvfp4_lut_cuda_probe [iters]

#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>

#define CUDA_CHECK(x)                                                      \
  do {                                                                     \
    cudaError_t err_ = (x);                                                \
    if (err_ != cudaSuccess) {                                             \
      std::printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                   __FILE__, __LINE__);                                     \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// ---------------------------------------------------------------------------
// Production code, copied verbatim from
// attention-rs/src/kernels/src/nvfp4_gemm.cu
// ---------------------------------------------------------------------------
__device__ __forceinline__ int2 get_int_from_table_16(const int q4,
                                                      const uint32_t table0,
                                                      const uint32_t table1,
                                                      const uint32_t table2,
                                                      const uint32_t table3) {
  uint32_t tmp[2];
  const uint32_t low_high_selection = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t low = __byte_perm(table0, table1, q4 >> shift);
    const uint32_t high = __byte_perm(table2, table3, q4 >> shift);
    tmp[i] = __byte_perm(low, high, low_high_selection >> shift);
  }
  return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420),
                   __byte_perm(tmp[0], tmp[1], 0x7531));
}

__device__ __forceinline__ void dequant_store_8(int q4, float scale,
                                                uint32_t LUT0, uint32_t LUT1,
                                                uint32_t LUT2, uint32_t LUT3,
                                                float *dst) {
  int2 w = get_int_from_table_16(q4, LUT0, LUT1, LUT2, LUT3);
  dst[0] = (float)(int8_t)(w.x) * scale;
  dst[1] = (float)(int8_t)(w.y) * scale;
  dst[2] = (float)(int8_t)(w.x >> 8) * scale;
  dst[3] = (float)(int8_t)(w.y >> 8) * scale;
  dst[4] = (float)(int8_t)(w.x >> 16) * scale;
  dst[5] = (float)(int8_t)(w.y >> 16) * scale;
  dst[6] = (float)(int8_t)(w.x >> 24) * scale;
  dst[7] = (float)(int8_t)(w.y >> 24) * scale;
}

// ---------------------------------------------------------------------------
// Candidate fast path: the same byte_perm machinery, but the tables hold the
// *f16 high byte* of the E2M1 value instead of 2x the value as an int8.
//
// E2M1 codeword -> value -> f16 bits:
//   0 -> 0.0 -> 0x0000    4 -> 2.0 -> 0x4000
//   1 -> 0.5 -> 0x3800    5 -> 3.0 -> 0x4200
//   2 -> 1.0 -> 0x3C00    6 -> 4.0 -> 0x4400
//   3 -> 1.5 -> 0x3E00    7 -> 6.0 -> 0x4600
// Every one of those has a zero low byte, so the f16 is just (byte << 8).
// Negative codewords OR in 0x8000.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void dequant_store_8_f16(int q4, float scale,
                                                   uint32_t LUT0, uint32_t LUT1,
                                                   uint32_t LUT2, uint32_t LUT3,
                                                   __half2 *dst4) {
  // LUT0/LUT1: f16 high byte of the positive codewords 0..7
  //   0.0 -> 0x00, 0.5 -> 0x38, 1.0 -> 0x3C, 1.5 -> 0x3E,
  //   2.0 -> 0x40, 3.0 -> 0x42, 4.0 -> 0x44, 6.0 -> 0x46
  // LUT2/LUT3: the same with the sign bit set.
  // (Byte order is little-endian, so LUT0 = {0x00,0x38,0x3C,0x3E} packs as
  //  0x3E3C3800.)
  const int2 w = get_int_from_table_16(q4, LUT0, LUT1, LUT2, LUT3);
  // w.x holds codewords 0,2,4,6 in bytes 0..3; w.y holds 1,3,5,7.
  // The production chain consumes them interleaved as
  // (w.x.b0, w.y.b0, w.x.b1, w.y.b1, ...), i.e. codewords 0..7 in order.
  const uint32_t bytes[8] = {
      (uint32_t)(w.x & 0xffu),        (uint32_t)(w.y & 0xffu),
      (uint32_t)((w.x >> 8) & 0xffu),  (uint32_t)((w.y >> 8) & 0xffu),
      (uint32_t)((w.x >> 16) & 0xffu), (uint32_t)((w.y >> 16) & 0xffu),
      (uint32_t)((w.x >> 24) & 0xffu), (uint32_t)((w.y >> 24) & 0xffu),
  };
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint16_t lo = (uint16_t)(bytes[2 * i] << 8);
    const uint16_t hi = (uint16_t)(bytes[2 * i + 1] << 8);
    const __half2 v = __halves2half2(*reinterpret_cast<const __half *>(&lo),
                                      *reinterpret_cast<const __half *>(&hi));
    dst4[i] = __hmul2(v, __float2half2_rn(scale));
  }
}

// ---------------------------------------------------------------------------
// Probe kernel
// ---------------------------------------------------------------------------
__global__ void probe_kernel(const uint32_t *words, const float *scales,
                             int n, int *fails_byte, int *fails_order,
                             int *fails_f16, float *worst_prod,
                             float *worst_fast, int *n_diff) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  int *fb = fails_byte, *fo = fails_order, *ff = fails_f16;
  __shared__ int s_fb, s_fo, s_ff;
  if (threadIdx.x == 0) {
    s_fb = 0; s_fo = 0; s_ff = 0;
  }
  __syncthreads();

  const uint32_t q4 = words[idx];
  const float raw_scale = scales[idx];

  const uint32_t LUT0 = 0x03020100, LUT1 = 0x0C080604;
  const uint32_t LUT2 = 0xFDFEFF00, LUT3 = 0xF4F8FAFC;

  // --- A) the LUT byte equals 2 * E2M1 value of the codeword in that slot.
  //     Codeword for chain slot j is nibble j of q4 (no layout swizzle).
  float ref[8];
  dequant_store_8((int)q4, 1.0f, LUT0, LUT1, LUT2, LUT3, ref);
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int cw = (int)((q4 >> (4 * j)) & 0xf);
    const float want = 2.0f * mag[cw & 7] * ((cw & 8) ? -1.0f : 1.0f);
    if (ref[j] != want) atomicAdd(&s_fb, 1);
  }

  // --- B) the chain order: slot j must carry codeword j (checked above by
  //     comparing against the codeword taken from nibble j).  Here we also
  //     confirm the byte-permutation layout explicitly: byte k of w.x is
  //     codeword 2k and byte k of w.y is codeword 2k+1.
  const int2 w = get_int_from_table_16((int)q4, LUT0, LUT1, LUT2, LUT3);
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const int cw_x = (int)((q4 >> (4 * (2 * k))) & 0xf);
    const int cw_y = (int)((q4 >> (4 * (2 * k + 1))) & 0xf);
    const float wx = (float)(int8_t)((w.x >> (8 * k)) & 0xff);
    const float wy = (float)(int8_t)((w.y >> (8 * k)) & 0xff);
    const float want_x = 2.0f * mag[cw_x & 7] * ((cw_x & 8) ? -1.0f : 1.0f);
    const float want_y = 2.0f * mag[cw_y & 7] * ((cw_y & 8) ? -1.0f : 1.0f);
    if (wx != want_x || wy != want_y) atomicAdd(&s_fo, 1);
  }

  // --- C) the f16 fast path must reproduce the production values.
  //     Production: value = 2 * E2M1 * (raw_scale * 0.5), rounded to f16.
  //     Fast path: f16(E2M1) * f16(raw_scale), rounded to f16.
  //     These differ by at most 1 ulp when raw_scale is not f16-exact; the
  //     probe reports the magnitude of the difference so the trade-off is
  //     measured rather than assumed.
  float prod[8];
  dequant_store_8((int)q4, raw_scale * 0.5f, LUT0, LUT1, LUT2, LUT3, prod);
  // f16 high-byte tables for the same codeword layout.
  const uint32_t FL0 = 0x3E3C3800u, FL1 = 0x46444240u;
  const uint32_t FL2 = 0xBEBCB880u, FL3 = 0xC6C4C2C0u;
  __half2 fast[4];
  dequant_store_8_f16((int)q4, raw_scale, FL0, FL1, FL2, FL3, fast);
  float fastf[8];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float2 f = __half22float2(fast[i]);
    fastf[2 * i] = f.x;
    fastf[2 * i + 1] = f.y;
  }
  // The production chain computes (2*E2M1) * (raw_scale*0.5) in fp32 and
  // rounds once to f16; the fast path computes f16(E2M1) * f16(raw_scale).
  // Measure both against the exact reference so the precision trade-off is
  // a number rather than an assumption.
  float local_worst_prod = 0.0f, local_worst_fast = 0.0f;
  int local_diff = 0;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int cw = (int)((q4 >> (4 * j)) & 0xf);
    const float want = mag[cw & 7] * ((cw & 8) ? -1.0f : 1.0f) * raw_scale;
    const float ref = __half2float(__float2half(want));
    const float rel_prod = fabsf(prod[j] - ref) / (fabsf(ref) + 1e-30f);
    const float rel_fast = fabsf(fastf[j] - ref) / (fabsf(ref) + 1e-30f);
    local_worst_prod = fmaxf(local_worst_prod, rel_prod);
    local_worst_fast = fmaxf(local_worst_fast, rel_fast);
    if (prod[j] != fastf[j]) local_diff++;
    if (rel_fast > 1e-2f) atomicAdd(&s_ff, 1);  // gross error
  }
  atomicMax((int *)worst_prod, __float_as_int(local_worst_prod));
  atomicMax((int *)worst_fast, __float_as_int(local_worst_fast));
  atomicAdd(n_diff, local_diff);
  __syncthreads();
  if (threadIdx.x == 0) {
    *fails_byte += s_fb;
    *fails_order += s_fo;
    *fails_f16 += s_ff;
  }
}

int main(int argc, char **argv) {
  const int n = argc > 1 ? std::atoi(argv[1]) : 100000;
  std::mt19937_64 g(0xC0FFEEu);
  std::vector<uint32_t> words(n);
  std::vector<float> scales(n);
  for (int i = 0; i < n; i++) {
    words[i] = (uint32_t)g();
    // Realistic FP8 E4M3 block scale range, kept out of the subnormal band.
    scales[i] = 0.01f + (float)(g() % 100000u) / 10000.0f;
  }

  uint32_t *d_words = nullptr;
  float *d_scales = nullptr;
  int *d_fb = nullptr, *d_fo = nullptr, *d_ff = nullptr;
  int *d_ndiff = nullptr;
  float *d_wp = nullptr, *d_wf = nullptr;
  CUDA_CHECK(cudaMalloc(&d_words, n * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_scales, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fb, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fo, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ff, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ndiff, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_wp, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wf, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_words, words.data(), n * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scales, scales.data(), n * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_fb, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_fo, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_ff, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_ndiff, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_wp, 0, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_wf, 0, sizeof(float)));

  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);

  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  probe_kernel<<<blocks, threads>>>(d_words, d_scales, n, d_fb, d_fo, d_ff,
                                    d_wp, d_wf, d_ndiff);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int fb = 0, fo = 0, ff = 0;
  CUDA_CHECK(cudaMemcpy(&fb, d_fb, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&fo, d_fo, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&ff, d_ff, sizeof(int), cudaMemcpyDeviceToHost));

  std::printf("A) LUT byte == 2xE2M1 value : %s (%d mismatches)\n",
              fb ? "FAIL" : "PASS", fb);
  std::printf("B) w.x/w.y byte layout      : %s (%d mismatches)\n",
              fo ? "FAIL" : "PASS", fo);
  std::printf("C) f16 fast path accuracy   : %s (%d gross errors)\n",
              ff ? "FAIL" : "PASS", ff);
  int ndiff = 0;
  float wp = 0.0f, wf = 0.0f;
  CUDA_CHECK(cudaMemcpy(&ndiff, d_ndiff, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&wp, d_wp, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&wf, d_wf, sizeof(float), cudaMemcpyDeviceToHost));
  std::printf("   elements differing prod vs fast : %d / %d\n", ndiff, n * 8);
  std::printf("   worst rel err  production chain : %.3e\n", wp);
  std::printf("   worst rel err  f16 fast path   : %.3e\n", wf);
  return (fb || fo || ff) ? 1 : 0;
}
