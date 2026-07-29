#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef RMS_EPSILON
#define RMS_EPSILON 1e-5f
#endif

static inline float reciprocal_sqrt(float value) {
  const float half = 0.5f * value;
  union {
    float f;
    uint32_t i;
  } bits = {value};
  bits.i = 0x5f3759dfu - (bits.i >> 1);
  float estimate = bits.f;
  estimate *= 1.5f - half * estimate * estimate;
  estimate *= 1.5f - half * estimate * estimate;
  return estimate;
}

extern "C" void rmsnorm_bf16(const bfloat16 *__restrict input,
                              const bfloat16 *__restrict gamma,
                              bfloat16 *__restrict output, int32_t cols) {
  constexpr float epsilon = RMS_EPSILON;
  event0();
  float sum_sq = 0.0f;
  for (int i = 0; i < cols; ++i) {
    const float value = static_cast<float>(input[i]);
    sum_sq += value * value;
  }
  const float inv_rms = reciprocal_sqrt(sum_sq / cols + epsilon);
  for (int i = 0; i < cols; ++i)
    output[i] = static_cast<bfloat16>(
        static_cast<float>(input[i]) * inv_rms * static_cast<float>(gamma[i]));
  event1();
}
