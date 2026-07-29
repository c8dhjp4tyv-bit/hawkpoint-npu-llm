#include <aie_api/aie.hpp>
#include <stdint.h>

static inline float fast_exp(float x) {
  if (x < -12.0f)
    return 0.0f;
  if (x > 0.0f)
    x = 0.0f;
  union {
    uint32_t i;
    float f;
  } value;
  value.i = static_cast<uint32_t>(12102203.0f * x + 1064866805.0f);
  return value.f;
}

extern "C" void attention_scores_bf16(
    const bfloat16 *__restrict queries, const bfloat16 *__restrict keys,
    int32_t position, bfloat16 *__restrict probabilities) {
  const int last = position;
  for (int head = 0; head < 3; ++head) {
    float scores[64];
    float maximum = -3.4e38f;
    const bfloat16 *q = queries + head * 64;
    for (int token = 0; token <= last; ++token) {
      const bfloat16 *k = keys + token * 64;
      float dot = 0.0f;
      for (int dim = 0; dim < 64; ++dim)
        dot += static_cast<float>(q[dim]) * static_cast<float>(k[dim]);
      scores[token] = dot * 0.125f;
      if (scores[token] > maximum)
        maximum = scores[token];
    }
    float sum = 0.0f;
    for (int token = 0; token <= last; ++token) {
      scores[token] = fast_exp(scores[token] - maximum);
      sum += scores[token];
    }
    const float inverse = 1.0f / sum;
    for (int token = 0; token < 64; ++token)
      probabilities[head * 64 + token] =
          static_cast<bfloat16>(token <= last ? scores[token] * inverse : 0.0f);
  }
}

extern "C" void attention_values_bf16(
    const bfloat16 *__restrict probabilities,
    const bfloat16 *__restrict values, int32_t position,
    bfloat16 *__restrict output) {
  const int last = position;
  for (int head = 0; head < 3; ++head) {
    const bfloat16 *p = probabilities + head * 64;
    for (int dim = 0; dim < 64; ++dim) {
      float sum = 0.0f;
      for (int token = 0; token <= last; ++token)
        sum += static_cast<float>(p[token]) *
               static_cast<float>(values[token * 64 + dim]);
      output[head * 64 + dim] = static_cast<bfloat16>(sum);
    }
  }
}
