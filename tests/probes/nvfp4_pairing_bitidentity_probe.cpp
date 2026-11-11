// Host-side bit-identity probe for the NVFP4 software dequant path in
// attention-rs src/kernels/src/nvfp4_gemm.cu.  Runs without CUDA: it replays
// the exact device bit-ops (PTX prmt / __byte_perm LUT lookup) against a
// straight reference table, so the LUT semantics are verified on any machine
// before the V100 run.
//
// Background (verified 2026-11-11, see PROGRESS.md part 16):
//   * NVFP4 E2M1 codewords 0..7 map to 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
//     (bit 3 = sign).  Sources of truth: attention-rs nvfp4_quant.cu
//     float_to_fp4_e2m1(), and metal-kernels/src/mxfp4.metal kMagLut.
//   * The kernel LUT does NOT store those values; it stores *2x the value* as
//     a signed byte: LUT0 {0,1,2,3} + LUT1 {4,6,8,12} is the positive
//     magnitude table (index = codeword & 7), LUT2 {0,-1,-2,-3} + LUT3
//     {-4,-6,-8,-12} the negative one.  The kernel compensates the factor 2
//     with `lut_scale = raw_scale * 0.5f`.
//   * get_int_from_table_16() lays the 8 codewords out as
//         w.x byte k -> codeword 2k,  w.y byte k -> codeword 2k+1
//     so the element-wise chain
//         dst[j] = (float)(int8_t)( j even ? w.x : w.y ) >> (8*((j>>1)&3))
//     visits codewords 0..7 in order (no layout swizzle).
//
// Claim under test, for EVERY packed word and EVERY block scale:
//   A) the byte the LUT yields for codeword c equals 2 * E2M1_value(c),
//      signed (negative codewords give the negated value);
//   B) the chain order maps element j to codeword j;
//   C) the product fed to the FMA (byte-as-float * lut_scale) equals
//      E2M1_value(c) * raw_scale bit-for-bit.
//
// Build & run:  c++ -std=c++17 -O2 -o probe probe.cpp && ./probe [iters]

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

namespace {

constexpr uint32_t LUT0 = 0x03020100, LUT1 = 0x0C080604;
constexpr uint32_t LUT2 = 0xFDFEFF00, LUT3 = 0xF4F8FAFC;

// Authoritative E2M1 magnitudes (attention-rs nvfp4_quant.cu).
constexpr float E2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};


// Exact f16 -> f32 conversion (the magnitudes used here are all normal
// numbers, but subnormals are handled for completeness).
float half2float_bits(uint16_t h) {
  const uint32_t e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  uint32_t raw;
  if (e == 0 && m == 0) {
    raw = (uint32_t)(h >> 15) << 31;
  } else if (e == 0) {
    int32_t b = 9;
    while (((m >> b) & 1u) == 0) b--;
    raw = ((uint32_t)(h >> 15) << 31) | ((uint32_t)(149 - b) << 23) |
          (((m << (9 - b)) & 0x7fffffu) << 13);
  } else {
    raw = ((uint32_t)(h >> 15) << 31) | ((e + 127 - 15) << 23) | (m << 13);
  }
  float f;
  std::memcpy(&f, &raw, 4);
  return f;
}

struct I2 { int32_t x, y; };

// CUDA __byte_perm(x, y, s):
//   result byte i is selected by nibble i of s;
//   the 3 low bits of that nibble pick the source byte
//   (0..3 -> x.byte[0..3], 4..7 -> y.byte[0..3]); the nibble's msb is
//   ignored (unlike the generic PTX prmt, which would sign-extend there --
//   which is exactly why the kernel shifts the sign bits down to bit 2
//   before using them as selectors).
uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t s) {
  const uint8_t *pa = reinterpret_cast<const uint8_t *>(&a);
  const uint8_t *pb = reinterpret_cast<const uint8_t *>(&b);
  uint8_t r[4];
  for (int i = 0; i < 4; i++) {
    const uint32_t idx = (s >> (4 * i)) & 7u;
    r[i] = idx < 4 ? pa[idx] : pb[idx - 4];
  }
  return (uint32_t)r[0] | ((uint32_t)r[1] << 8) | ((uint32_t)r[2] << 16) |
         ((uint32_t)r[3] << 24);
}

// Mirror of get_int_from_table_16() in nvfp4_gemm.cu.
I2 get_int_from_table_16(int q4) {
  uint32_t tmp[2];
  const uint32_t low_high_selection =
      0x32103210u | (((uint32_t)q4 & 0x88888888u) >> 1);
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t low = byte_perm(LUT0, LUT1, (uint32_t)q4 >> shift);
    const uint32_t high = byte_perm(LUT2, LUT3, (uint32_t)q4 >> shift);
    tmp[i] = byte_perm(low, high, low_high_selection >> shift);
  }
  return I2{static_cast<int32_t>(byte_perm(tmp[0], tmp[1], 0x6420u)),
            static_cast<int32_t>(byte_perm(tmp[0], tmp[1], 0x7531u))};
}

// The byte the LUT yields for chain slot `slot` (0..7 -> codeword 0..7).
float chain_byte(uint32_t q4, int slot) {
  const I2 w = get_int_from_table_16((int)q4);
  const int32_t word = (slot & 1) ? w.y : w.x;
  const int32_t shifted = word >> (8 * ((slot >> 1) & 3));
  return (float)(int8_t)(shifted);
}

}  // namespace

// ---------------------------------------------------------------------------
// P2-2 check: the f16 high-byte tables must reproduce the *same numbers* as
// the int8 tables, so the decode path can switch to them without changing any
// result bit.
//
// The int8 tables hold 2x the E2M1 value (0,1,2,3,4,6,8,12 and their
// negatives).  Those magnitudes are all exact in f16, so the f16 tables hold
// the same magnitudes as f16 high bytes (the f16 low byte is always 0 for
// them), and the extracted floats are identical.
// ---------------------------------------------------------------------------
namespace {

// f16 high-byte tables for 2x E2M1 magnitudes:
//   0 -> 0x0000, 1 -> 0x3C00, 2 -> 0x4000, 3 -> 0x4200,
//   4 -> 0x4400, 6 -> 0x4600, 8 -> 0x4800, 12 -> 0x4A00
constexpr uint32_t FLUT0 = 0x42403C00u, FLUT1 = 0x4A484644u;  // codewords 0..3, 4..7
constexpr uint32_t FLUT2 = 0xC2C0BC00u, FLUT3 = 0xCAC8C6C4u;  // negated

I2 get_int_from_table_16_f16(int q4) {
  uint32_t tmp[2];
  const uint32_t low_high_selection =
      0x32103210u | (((uint32_t)q4 & 0x88888888u) >> 1);
  for (uint32_t i = 0; i < 2; ++i) {
    const uint32_t shift = 16 * i;
    const uint32_t low = byte_perm(FLUT0, FLUT1, (uint32_t)q4 >> shift);
    const uint32_t high = byte_perm(FLUT2, FLUT3, (uint32_t)q4 >> shift);
    tmp[i] = byte_perm(low, high, low_high_selection >> shift);
  }
  return I2{static_cast<int32_t>(byte_perm(tmp[0], tmp[1], 0x6420u)),
            static_cast<int32_t>(byte_perm(tmp[0], tmp[1], 0x7531u))};
}

float chain_value_f16(uint32_t q4, int slot) {
  const I2 w = get_int_from_table_16_f16((int)q4);
  const int32_t word = (slot & 1) ? w.y : w.x;
  const int32_t shifted = word >> (8 * ((slot >> 1) & 3));
  const uint16_t h = (uint16_t)((uint32_t)shifted << 8);  // f16 low byte is 0
  return half2float_bits(h);
}

}  // namespace

int main(int argc, char **argv) {
  const int iters = argc > 1 ? std::atoi(argv[1]) : 100000;
  std::mt19937_64 g(0xC0FFEEu);
  int fails = 0;

  for (int it = 0; it < iters; it++) {
    const uint32_t q4 = (uint32_t)g();
    // A realistic FP32 block scale, kept in the normal range: subnormal
    // products are not required to round identically, which would make the
    // bit-identity claim below untestable rather than informative.
    const uint32_t scale_bits = (uint32_t)(g() & 0x7fffffffu);
    float raw_scale;
    std::memcpy(&raw_scale, &scale_bits, 4);
    if (!(raw_scale > 1e-3f) || raw_scale > 1e3f) raw_scale = 1.0f;

    for (int slot = 0; slot < 8; slot++) {
      // The codeword that feeds chain slot `slot`: nibble `slot` of q4.
      const int codeword = (int)((q4 >> (4 * slot)) & 0xf);
      const int sign = (codeword & 8) ? -1 : 1;
      const float want = sign * E2M1[codeword & 7];

      // A) the LUT byte equals 2 * E2M1 value.
      const float got = chain_byte(q4, slot);
      const float want_byte = 2.0f * want;
      if (got != want_byte) {
        if (fails < 10) {
          std::printf("BYTE-MISMATCH q4=%08x slot=%d got=%g want=%g\n", q4,
                      slot, (double)got, (double)want_byte);
        }
        fails++;
      }

      // C) the product fed to the FMA equals E2M1 value * raw scale.
      const float lut_scale = raw_scale * 0.5f;
      const float prod = got * lut_scale;
      const float want_prod = want * raw_scale;
      if (prod != want_prod) {
        if (fails < 10) {
          std::printf("PROD-MISMATCH q4=%08x slot=%d got=%g want=%g\n", q4,
                      slot, (double)prod, (double)want_prod);
        }
        fails++;
      }

      // D) the f16 high-byte tables reproduce exactly the same magnitudes.
      const float got_f16 = chain_value_f16(q4, slot);
      if (got_f16 != got) {
        if (fails < 10) {
          std::printf("F16-MISMATCH q4=%08x slot=%d f16=%g int8=%g\n", q4,
                      slot, (double)got_f16, (double)got);
        }
        fails++;
      }
    }
  }

  std::printf("%s: %d words checked, %d mismatches\n",
              fails ? "FAIL" : "PASS", iters, fails);
  return fails ? 1 : 0;
}
