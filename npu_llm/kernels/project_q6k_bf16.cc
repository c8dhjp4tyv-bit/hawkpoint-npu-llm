// Native GGML Q6_K -> BF16 GEMV kernel for XDNA1/AIE2.

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
constexpr int kBlockBytes = 224;  // 210-byte GGML payload, 32-byte padded

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

}  // namespace

extern "C" {

void project_q6k_bf16(
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
          static_cast<uint16_t>(current[208]) |
          (static_cast<uint16_t>(current[209]) << 8U));
      const uint8_t * ql = current;
      const uint8_t * qh = current + 128;
      const int8_t * scales = reinterpret_cast<const int8_t *>(current + 192);
      for (int offset = 0; offset < kQK; offset += 128) {
        for (int lane = 0; lane < 32; ++lane) {
          const int scale_index = lane / 16;
          const int8_t q1 = static_cast<int8_t>(
              (ql[lane] & 0x0fU) | (((qh[lane] >> 0U) & 3U) << 4U)) - 32;
          const int8_t q2 = static_cast<int8_t>(
              (ql[lane + 32] & 0x0fU) | (((qh[lane] >> 2U) & 3U) << 4U)) - 32;
          const int8_t q3 = static_cast<int8_t>(
              (ql[lane] >> 4U) | (((qh[lane] >> 4U) & 3U) << 4U)) - 32;
          const int8_t q4 = static_cast<int8_t>(
              (ql[lane + 32] >> 4U) | (((qh[lane] >> 6U) & 3U) << 4U)) - 32;
          const int base = block * kQK + offset;
          sum += d * static_cast<float>(scales[scale_index + 0]) * q1 *
                 static_cast<float>(activation[base + lane]);
          sum += d * static_cast<float>(scales[scale_index + 2]) * q2 *
                 static_cast<float>(activation[base + lane + 32]);
          sum += d * static_cast<float>(scales[scale_index + 4]) * q3 *
                 static_cast<float>(activation[base + lane + 64]);
          sum += d * static_cast<float>(scales[scale_index + 6]) * q4 *
                 static_cast<float>(activation[base + lane + 96]);
        }
        ql += 64;
        qh += 32;
        scales += 8;
      }
    }
    output[row] = static_cast<bfloat16>(sum);
  }
}

}  // extern "C"
