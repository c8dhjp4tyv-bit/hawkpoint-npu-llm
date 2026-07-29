#include <aie_api/aie.hpp>
#include <stdint.h>

static inline float attention_exp(float x) {
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

// One grouped-query head: append K/V and immediately consume the updated
// caches. Keeping the whole block in one kernel removes three xclbin switches
// per decoder layer while the persistent caches remain NPU-resident.
extern "C" void attention_block_bf16(
    const bfloat16 *__restrict packed_qkv,
    const bfloat16 *__restrict cache_in,
    int32_t position,
    bfloat16 *__restrict cache_out,
    bfloat16 *__restrict output) {
  const bfloat16 *queries = packed_qkv;
  const bfloat16 *new_key = packed_qkv + 192;
  const bfloat16 *new_value = packed_qkv + 256;
  const bfloat16 *key_cache_in = cache_in;
  const bfloat16 *value_cache_in = cache_in + 4096;
  bfloat16 *key_cache_out = cache_out;
  bfloat16 *value_cache_out = cache_out + 4096;
  for (int i = 0; i < 4096; i += 16) {
    aie::store_v(key_cache_out + i, aie::load_v<16>(key_cache_in + i));
    aie::store_v(value_cache_out + i, aie::load_v<16>(value_cache_in + i));
  }
  const int cache_offset = position * 64;
  for (int i = 0; i < 64; i += 16) {
    aie::store_v(key_cache_out + cache_offset + i,
                 aie::load_v<16>(new_key + i));
    aie::store_v(value_cache_out + cache_offset + i,
                 aie::load_v<16>(new_value + i));
  }

  for (int head = 0; head < 3; ++head) {
    float scores[64];
    float maximum = -3.4e38f;
    const bfloat16 *q = queries + head * 64;
    for (int token = 0; token <= position; ++token) {
      const bfloat16 *k = key_cache_out + token * 64;
      float dot = 0.0f;
      for (int dim = 0; dim < 64; ++dim)
        dot += static_cast<float>(q[dim]) * static_cast<float>(k[dim]);
      scores[token] = dot * 0.125f;
      if (scores[token] > maximum)
        maximum = scores[token];
    }
    float sum = 0.0f;
    for (int token = 0; token <= position; ++token) {
      scores[token] = attention_exp(scores[token] - maximum);
      sum += scores[token];
    }
    const float inverse = 1.0f / sum;
    for (int dim = 0; dim < 64; ++dim) {
      float result = 0.0f;
      for (int token = 0; token <= position; ++token)
        result += scores[token] * inverse *
                  static_cast<float>(value_cache_out[token * 64 + dim]);
      output[head * 64 + dim] = static_cast<bfloat16>(result);
    }
  }
}
