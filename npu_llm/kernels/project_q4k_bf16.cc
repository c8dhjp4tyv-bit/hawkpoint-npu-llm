// Native GGML Q4_K -> BF16 GEMV kernel for XDNA1/AIE2.
//
// The host passes the original Q4_K blocks to this kernel.  Dequantization is
// deliberately kept next to the dot product so a persistent XRT BO can be
// reused across decoded tokens without a CPU float/re-quantize round-trip.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 2048
#endif

namespace {

constexpr int kQK = 256;
constexpr int kBlockBytes = 160;  // 144-byte GGML payload, 32-byte padded

// GGML stores the super-block scales as IEEE FP16.  Keeping this conversion
// local avoids making the AIE kernel depend on ggml-common.h (which is not part
// of the stable external-backend ABI).
inline float fp16_to_float(uint16_t bits) {
  const uint32_t sign = (static_cast<uint32_t>(bits & 0x8000U)) << 16U;
  const uint32_t exponent = (bits >> 10U) & 0x1fU;
  const uint32_t mantissa = bits & 0x03ffU;
  uint32_t result;
  if (exponent == 0) {
    if (mantissa == 0) {
      result = sign;
    } else {
      uint32_t normalized = mantissa;
      uint32_t shift = 0;
      while ((normalized & 0x0400U) == 0) {
        normalized <<= 1U;
        ++shift;
      }
      normalized &= 0x03ffU;
      result = sign | ((127U - 15U - shift) << 23U) |
               (normalized << 13U);
    }
  } else if (exponent == 0x1fU) {
    result = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    result = sign | ((exponent + (127U - 15U)) << 23U) |
             (mantissa << 13U);
  }
  float value;
  __builtin_memcpy(&value, &result, sizeof(value));
  return value;
}

inline void get_scale_min(
    int index,
    const uint8_t * scales,
    uint8_t * scale,
    uint8_t * minimum) {
  if (index < 4) {
    *scale = scales[index] & 63U;
    *minimum = scales[index + 4] & 63U;
  } else {
    *scale = (scales[index + 4] & 0x0fU) |
             ((scales[index - 4] >> 6U) << 4U);
    *minimum = (scales[index + 4] >> 4U) |
              ((scales[index] >> 6U) << 4U);
  }
}

}  // namespace

extern "C" {

void project_q4k_bf16(
    const uint8_t *__restrict weights,
    const bfloat16 *__restrict activation,
    bfloat16 *__restrict output) {
  static_assert(DIM_K % kQK == 0);
  static_assert(kBlockBytes % 32 == 0);
  const int block_count = DIM_K / kQK;
  for (int row = 0; row < DIM_M; ++row) {
    float sum = 0.0f;
    const uint8_t * row_weights = weights + row * block_count * kBlockBytes;
    for (int block = 0; block < block_count; ++block) {
      const uint8_t * current = row_weights + block * kBlockBytes;
      const float d = fp16_to_float(
          static_cast<uint16_t>(current[0]) |
          (static_cast<uint16_t>(current[1]) << 8U));
      const float dmin = fp16_to_float(
          static_cast<uint16_t>(current[2]) |
          (static_cast<uint16_t>(current[3]) << 8U));
      const uint8_t * scales = current + 4;
      const uint8_t * quants = current + 16;
      int scale_index = 0;
      for (int offset = 0; offset < kQK; offset += 64) {
        uint8_t scale = 0;
        uint8_t minimum = 0;
        get_scale_min(scale_index++, scales, &scale, &minimum);
        const float d1 = d * static_cast<float>(scale);
        const float m1 = dmin * static_cast<float>(minimum);
        get_scale_min(scale_index++, scales, &scale, &minimum);
        const float d2 = d * static_cast<float>(scale);
        const float m2 = dmin * static_cast<float>(minimum);
        const uint8_t * q = quants + (offset / 64) * 32;
        for (int lane = 0; lane < 32; ++lane) {
          sum += (d1 * static_cast<float>(q[lane] & 0x0fU) - m1) *
                 static_cast<float>(activation[block * kQK + offset + lane]);
          sum += (d2 * static_cast<float>(q[lane] >> 4U) - m2) *
                 static_cast<float>(activation[block * kQK + offset + 32 + lane]);
        }
      }
    }
    output[row] = static_cast<bfloat16>(sum);
  }
}

}  // extern "C"
