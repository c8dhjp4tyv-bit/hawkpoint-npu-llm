#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef Q_PER_KV
#define Q_PER_KV 3
#endif
#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif
#ifndef CONTEXT
#define CONTEXT 64
#endif

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
  const bfloat16 *new_key = packed_qkv + Q_PER_KV * HEAD_DIM;
  const bfloat16 *new_value = new_key + HEAD_DIM;
  const bfloat16 *key_cache_in = cache_in;
  const bfloat16 *value_cache_in = cache_in + CONTEXT * HEAD_DIM;
  bfloat16 *key_cache_out = cache_out;
  bfloat16 *value_cache_out = cache_out + CONTEXT * HEAD_DIM;
  for (int i = 0; i < CONTEXT * HEAD_DIM; i += 16) {
    aie::store_v(key_cache_out + i, aie::load_v<16>(key_cache_in + i));
    aie::store_v(value_cache_out + i, aie::load_v<16>(value_cache_in + i));
  }
  const int cache_offset = position * HEAD_DIM;
  for (int i = 0; i < HEAD_DIM; i += 16) {
    aie::store_v(key_cache_out + cache_offset + i,
                 aie::load_v<16>(new_key + i));
    aie::store_v(value_cache_out + cache_offset + i,
                 aie::load_v<16>(new_value + i));
  }

  for (int head = 0; head < Q_PER_KV; ++head) {
    float scores[CONTEXT];
    float maximum = -3.4e38f;
    const bfloat16 *q = queries + head * HEAD_DIM;
    for (int token = 0; token <= position; ++token) {
      const bfloat16 *k = key_cache_out + token * HEAD_DIM;
      float dot = 0.0f;
      for (int dim = 0; dim < HEAD_DIM; ++dim)
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
    for (int dim = 0; dim < HEAD_DIM; ++dim) {
      float result = 0.0f;
      for (int token = 0; token <= position; ++token)
        result += scores[token] * inverse *
                  static_cast<float>(value_cache_out[token * HEAD_DIM + dim]);
      output[head * HEAD_DIM + dim] = static_cast<bfloat16>(result);
    }
  }
}
