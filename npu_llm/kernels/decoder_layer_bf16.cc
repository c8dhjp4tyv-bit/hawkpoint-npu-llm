#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <lut_based_ops.cpp>
#include <stdint.h>

static inline float layer_rsqrt(float value) {
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

extern "C" {

void layer_copy576(const bfloat16 *__restrict input,
                   bfloat16 *__restrict output) {
  for (int i = 0; i < 576; i += 16)
    aie::store_v(output + i, aie::load_v<16>(input + i));
}

void layer_rmsnorm576(const bfloat16 *__restrict input,
                      const bfloat16 *__restrict gamma,
                      bfloat16 *__restrict output) {
  float sum_sq = 0.0f;
  for (int i = 0; i < 576; ++i) {
    const float x = static_cast<float>(input[i]);
    sum_sq += x * x;
  }
  const float inv = layer_rsqrt(sum_sq / 576.0f + 1e-5f);
  for (int i = 0; i < 576; ++i)
    output[i] = static_cast<bfloat16>(
        static_cast<float>(input[i]) * inv * static_cast<float>(gamma[i]));
}

void layer_project64_k576(const uint8_t *__restrict packed,
                          const bfloat16 *__restrict activation,
                          bfloat16 *__restrict output) {
  const int8_t *weights = reinterpret_cast<const int8_t *>(packed);
  const float *scales = reinterpret_cast<const float *>(packed + 64 * 576);
  for (int row = 0; row < 64; ++row) {
    aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
    for (int col = 0; col < 576; col += 32) {
      const auto wi = aie::load_v<32>(weights + row * 576 + col);
      const auto w = aie::to_float<bfloat16>(wi);
      const auto x = aie::load_v<32>(activation + col);
      total = aie::add(total, aie::mul(w, x));
    }
    const auto lanes = total.template to_vector<float>();
    float sum = 0.0f;
    for (int lane = 0; lane < 32; ++lane)
      sum += lanes[lane];
    output[row] = static_cast<bfloat16>(sum * scales[row]);
  }
}

void layer_project32_k1536(const uint8_t *__restrict packed,
                           const bfloat16 *__restrict activation,
                           bfloat16 *__restrict output) {
  const int8_t *weights = reinterpret_cast<const int8_t *>(packed);
  const float *scales = reinterpret_cast<const float *>(packed + 32 * 1536);
  for (int row = 0; row < 32; ++row) {
    aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
    for (int col = 0; col < 1536; col += 32) {
      const auto wi = aie::load_v<32>(weights + row * 1536 + col);
      const auto w = aie::to_float<bfloat16>(wi);
      const auto x = aie::load_v<32>(activation + col);
      total = aie::add(total, aie::mul(w, x));
    }
    const auto lanes = total.template to_vector<float>();
    float sum = 0.0f;
    for (int lane = 0; lane < 32; ++lane)
      sum += lanes[lane];
    output[row] = static_cast<bfloat16>(sum * scales[row]);
  }
}

void layer_project32_k576_bf16(const bfloat16 *__restrict weights,
                               const bfloat16 *__restrict activation,
                               bfloat16 *__restrict output, int32_t offset) {
  for (int row = 0; row < 32; ++row) {
    aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
    for (int col = 0; col < 576; col += 32) {
      const auto w = aie::load_v<32>(weights + row * 576 + col);
      const auto x = aie::load_v<32>(activation + col);
      total = aie::add(total, aie::mul(w, x));
    }
    const auto lanes = total.template to_vector<float>();
    float sum = 0.0f;
    for (int lane = 0; lane < 32; ++lane)
      sum += lanes[lane];
    output[offset + row] = static_cast<bfloat16>(sum);
  }
}

void layer_project16_k1536_bf16(const bfloat16 *__restrict weights,
                                const bfloat16 *__restrict activation,
                                bfloat16 *__restrict output, int32_t offset) {
  for (int row = 0; row < 16; ++row) {
    aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
    for (int col = 0; col < 1536; col += 32) {
      const auto w = aie::load_v<32>(weights + row * 1536 + col);
      const auto x = aie::load_v<32>(activation + col);
      total = aie::add(total, aie::mul(w, x));
    }
    const auto lanes = total.template to_vector<float>();
    float sum = 0.0f;
    for (int lane = 0; lane < 32; ++lane)
      sum += lanes[lane];
    output[offset + row] = static_cast<bfloat16>(sum);
  }
}

void layer_pack_rope32(const bfloat16 *__restrict first_half,
                       const bfloat16 *__restrict second_half,
                       const bfloat16 *__restrict lut,
                       bfloat16 *__restrict packed, int32_t head,
                       int32_t kind) {
  int offset;
  if (kind == 0)
    offset = (head / 3) * 336 + (head % 3) * 64;
  else
    offset = head * 336 + (kind == 1 ? 192 : 256);
  bfloat16 *dst = packed + offset;
  if (kind == 2) {
    for (int i = 0; i < 32; i += 16) {
      aie::store_v(dst + i, aie::load_v<16>(first_half + i));
      aie::store_v(dst + 32 + i, aie::load_v<16>(second_half + i));
    }
    packed[head * 336 + 320] = lut[64];
    return;
  }
  for (int i = 0; i < 32; i += 16) {
    const auto first = aie::load_v<16>(first_half + i);
    const auto second = aie::load_v<16>(second_half + i);
    const auto cosine = aie::load_v<16>(lut + i);
    const auto sine = aie::load_v<16>(lut + 32 + i);
    const aie::vector<bfloat16, 16> first_out =
        aie::sub(aie::mul(first, cosine), aie::mul(second, sine));
    const aie::vector<bfloat16, 16> second_out =
        aie::add(aie::mul(first, sine), aie::mul(second, cosine));
    aie::store_v(dst + i, first_out);
    aie::store_v(dst + 32 + i, second_out);
  }
}

// kind: 0=Q (head 0..8), 1=K (head 0..2), 2=V (head 0..2).
void layer_pack_rope64(const bfloat16 *__restrict row,
                       const bfloat16 *__restrict lut,
                       bfloat16 *__restrict packed, int32_t head,
                       int32_t kind) {
  int offset;
  if (kind == 0)
    offset = (head / 3) * 336 + (head % 3) * 64;
  else
    offset = head * 336 + (kind == 1 ? 192 : 256);
  bfloat16 *dst = packed + offset;
  if (kind == 2) {
    for (int i = 0; i < 64; i += 16)
      aie::store_v(dst + i, aie::load_v<16>(row + i));
    packed[head * 336 + 320] = lut[64];
    return;
  }
  for (int i = 0; i < 32; i += 16) {
    const auto first = aie::load_v<16>(row + i);
    const auto second = aie::load_v<16>(row + 32 + i);
    const auto cosine = aie::load_v<16>(lut + i);
    const auto sine = aie::load_v<16>(lut + 32 + i);
    const aie::vector<bfloat16, 16> first_out =
        aie::sub(aie::mul(first, cosine), aie::mul(second, sine));
    const aie::vector<bfloat16, 16> second_out =
        aie::add(aie::mul(first, sine), aie::mul(second, cosine));
    aie::store_v(dst + i, first_out);
    aie::store_v(dst + 32 + i, second_out);
  }
}

static inline float layer_exp(float x) {
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

void layer_attention(const bfloat16 *__restrict packed_qkv,
                     const bfloat16 *__restrict cache_in,
                     bfloat16 *__restrict cache_out,
                     bfloat16 *__restrict output) {
  const int32_t position =
      static_cast<int32_t>(static_cast<float>(packed_qkv[320]));
  const bfloat16 *queries = packed_qkv;
  const bfloat16 *new_key = packed_qkv + 192;
  const bfloat16 *new_value = packed_qkv + 256;
  const bfloat16 *key_in = cache_in;
  const bfloat16 *value_in = cache_in + 4096;
  bfloat16 *keys = cache_out;
  bfloat16 *values = cache_out + 4096;
  for (int i = 0; i < 4096; i += 16) {
    aie::store_v(keys + i, aie::load_v<16>(key_in + i));
    aie::store_v(values + i, aie::load_v<16>(value_in + i));
  }
  const int offset = position * 64;
  for (int i = 0; i < 64; i += 16) {
    aie::store_v(keys + offset + i, aie::load_v<16>(new_key + i));
    aie::store_v(values + offset + i, aie::load_v<16>(new_value + i));
  }
  for (int head = 0; head < 3; ++head) {
    float score[64];
    float maximum = -3.4e38f;
    const bfloat16 *q = queries + head * 64;
    for (int token = 0; token <= position; ++token) {
      float dot = 0.0f;
      for (int dim = 0; dim < 64; ++dim)
        dot += static_cast<float>(q[dim]) *
               static_cast<float>(keys[token * 64 + dim]);
      score[token] = dot * 0.125f;
      if (score[token] > maximum)
        maximum = score[token];
    }
    float sum = 0.0f;
    for (int token = 0; token <= position; ++token) {
      score[token] = layer_exp(score[token] - maximum);
      sum += score[token];
    }
    const float inv = 1.0f / sum;
    for (int dim = 0; dim < 64; ++dim) {
      float result = 0.0f;
      for (int token = 0; token <= position; ++token)
        result += score[token] * inv *
                  static_cast<float>(values[token * 64 + dim]);
      output[head * 64 + dim] = static_cast<bfloat16>(result);
    }
  }
}

void layer_residual64(const bfloat16 *__restrict projection,
                      const bfloat16 *__restrict residual,
                      bfloat16 *__restrict output, int32_t offset) {
  for (int i = 0; i < 64; i += 16)
    aie::store_v(output + offset + i,
                 aie::add(aie::load_v<16>(projection + i),
                          aie::load_v<16>(residual + offset + i)));
}

void layer_store_gate64(const bfloat16 *__restrict gate,
                        bfloat16 *__restrict output, int32_t offset) {
  for (int i = 0; i < 64; i += 16)
    aie::store_v(output + offset + i, aie::load_v<16>(gate + i));
}

void layer_store_gate32(const bfloat16 *__restrict gate,
                        bfloat16 *__restrict output, int32_t offset) {
  for (int i = 0; i < 32; i += 16)
    aie::store_v(output + offset + i, aie::load_v<16>(gate + i));
}

void layer_swiglu_up64(const bfloat16 *__restrict up,
                       bfloat16 *__restrict output, int32_t offset) {
  const auto half = aie::broadcast<bfloat16, 16>(0.5f);
  const auto one = aie::broadcast<bfloat16, 16>(1.0f);
  for (int i = 0; i < 64; i += 16) {
    const auto gate = aie::load_v<16>(output + offset + i);
    const auto u = aie::load_v<16>(up + i);
    const aie::vector<bfloat16, 16> half_gate = aie::mul(gate, half);
    const aie::vector<bfloat16, 16> tanh_half = getTanhBf16(half_gate);
    const aie::vector<bfloat16, 16> sigmoid =
        aie::mul(aie::add(tanh_half, one), half);
    const aie::vector<bfloat16, 16> silu = aie::mul(gate, sigmoid);
    const aie::vector<bfloat16, 16> result = aie::mul(silu, u);
    aie::store_v(output + offset + i, result);
  }
}

void layer_swiglu_up32(const bfloat16 *__restrict up,
                       bfloat16 *__restrict output, int32_t offset) {
  const auto half = aie::broadcast<bfloat16, 16>(0.5f);
  const auto one = aie::broadcast<bfloat16, 16>(1.0f);
  for (int i = 0; i < 32; i += 16) {
    const auto gate = aie::load_v<16>(output + offset + i);
    const auto u = aie::load_v<16>(up + i);
    const aie::vector<bfloat16, 16> half_gate = aie::mul(gate, half);
    const aie::vector<bfloat16, 16> tanh_half = getTanhBf16(half_gate);
    const aie::vector<bfloat16, 16> sigmoid =
        aie::mul(aie::add(tanh_half, one), half);
    const aie::vector<bfloat16, 16> silu = aie::mul(gate, sigmoid);
    const aie::vector<bfloat16, 16> result = aie::mul(silu, u);
    aie::store_v(output + offset + i, result);
  }
}

void layer_residual32(const bfloat16 *__restrict projection,
                      const bfloat16 *__restrict residual,
                      bfloat16 *__restrict output, int32_t offset) {
  for (int i = 0; i < 32; i += 16)
    aie::store_v(output + offset + i,
                 aie::add(aie::load_v<16>(projection + i),
                          aie::load_v<16>(residual + offset + i)));
}

void layer_residual16(const bfloat16 *__restrict projection,
                      const bfloat16 *__restrict residual,
                      bfloat16 *__restrict output, int32_t offset) {
  const auto p = aie::load_v<16>(projection);
  const auto r = aie::load_v<16>(residual + offset);
  const aie::vector<bfloat16, 16> sum = aie::add(p, r);
  aie::store_v(output + offset, sum);
}

} // extern "C"
