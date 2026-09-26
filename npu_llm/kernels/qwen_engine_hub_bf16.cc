// Hub-tile kernels of the Qwen2.5 0.5B decoder engine (designs/qwen_engine.py).
#include "qwen_engine_common.h"

extern "C" {

void q_hub_init(const bfloat16 *__restrict header, bfloat16 *__restrict lut,
                bfloat16 *__restrict broadcast) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < Q_LUT; i += 32)
    aie::store_v(lut + i, aie::load_v<32>(header + i));
#pragma clang loop unroll(disable)
  for (int i = 0; i < Q_HIDDEN; i += 32)
    aie::store_v(broadcast + i, aie::load_v<32>(header + Q_HEADER_HIDDEN + i));
}

void q_copy_vector(const bfloat16 *__restrict input,
                   bfloat16 *__restrict output) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < Q_VECTOR; i += 32)
    aie::store_v(output + i, aie::load_v<32>(input + i));
}

// RoPE with FP32 cos/sin carried as three BF16 parts:
// out[i] = x[i]*cos - x[i+32]*sin, out[i+32] = x[i+32]*cos + x[i]*sin.
static __attribute__((noinline)) void q_rope64(const bfloat16 *__restrict row,
                            const bfloat16 *__restrict lut,
                            bfloat16 *__restrict out) {
  for (int i = 0; i < 32; i += 16) {
    const emu_bf16 first = aie::load_v<16>(row + i);
    const emu_bf16 second = aie::load_v<16>(row + 32 + i);
    emu_acc a = aie::mul(first, aie::load_v<16>(lut + 64 + i));
    a = aie::mac(a, first, aie::load_v<16>(lut + 32 + i));
    a = aie::mac(a, first, aie::load_v<16>(lut + i));
    emu_acc s = aie::mul(second, aie::load_v<16>(lut + Q_HEADER_SIN + 64 + i));
    s = aie::mac(s, second, aie::load_v<16>(lut + Q_HEADER_SIN + 32 + i));
    s = aie::mac(s, second, aie::load_v<16>(lut + Q_HEADER_SIN + i));
    aie::store_v(out + i, emu_round(aie::sub(a.template to_vector<float>(),
                                             s.template to_vector<float>())));
    emu_acc b = aie::mul(second, aie::load_v<16>(lut + 64 + i));
    b = aie::mac(b, second, aie::load_v<16>(lut + 32 + i));
    b = aie::mac(b, second, aie::load_v<16>(lut + i));
    emu_acc c = aie::mul(first, aie::load_v<16>(lut + Q_HEADER_SIN + 64 + i));
    c = aie::mac(c, first, aie::load_v<16>(lut + Q_HEADER_SIN + 32 + i));
    c = aie::mac(c, first, aie::load_v<16>(lut + Q_HEADER_SIN + i));
    aie::store_v(out + 32 + i, emu_round(aie::add(b.template to_vector<float>(),
                                                  c.template to_vector<float>())));
  }
}

// qkv row r of the layer lives at slot (r / 192) * 896 + r % 192; heads are
// 64-row aligned inside a tile's 192 rows.
static inline const bfloat16 *q_head(const bfloat16 *joined, int head) {
  const int row = head * 64;
  return joined + (row / Q_QKV_ROWS) * Q_SLICE + row % Q_QKV_ROWS;
}

static inline float q_dot64(const bfloat16 *__restrict a,
                            const bfloat16 *__restrict b) {
  aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
  total = aie::mac(total, aie::load_v<32>(a), aie::load_v<32>(b));
  total = aie::mac(total, aie::load_v<32>(a + 32), aie::load_v<32>(b + 32));
  return q_reduce(total);
}

// Attention for one KV head and its seven query heads.
void q_attention(const bfloat16 *__restrict joined,
                 const bfloat16 *__restrict cache,
                 const bfloat16 *__restrict lut,
                 bfloat16 *__restrict broadcast,
                 bfloat16 *__restrict kv_new, int32_t kv_head) {
  const int position =
      static_cast<int>(static_cast<float>(lut[Q_HEADER_POSITION]));
  bfloat16 *new_key = kv_new + kv_head * 128;
  bfloat16 *new_value = new_key + 64;
  q_rope64(q_head(joined, 14 + kv_head), lut, new_key);
  const bfloat16 *value_row = q_head(joined, 16 + kv_head);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 64; i += 32)
    aie::store_v(new_value + i, aie::load_v<32>(value_row + i));
  const bfloat16 *keys = cache;
  const bfloat16 *values = cache + 4096;

  alignas(64) bfloat16 query[64];
  alignas(64) float score[64];
  alignas(64) bfloat16 p_hi[64];
  alignas(64) bfloat16 p_mid[64];
  alignas(64) bfloat16 p_lo[64];
  const int count = position + 1;
  const int vectors = (count + 15) / 16;
  for (int head = kv_head * 7; head < kv_head * 7 + 7; ++head) {
    q_rope64(q_head(joined, head), lut, query);
    for (int token = 0; token < position; ++token)
      score[token] = q_dot64(query, keys + token * 64);
    score[position] = q_dot64(query, new_key);
    for (int i = count; i < vectors * 16; ++i)
      score[i] = score[0];
    // scores * 0.125 is exact; the maximum is unaffected by padding copies.
    emu_f32 peak = aie::load_v<16>(score);
    for (int v = 1; v < vectors; ++v)
      peak = aie::max(peak, aie::load_v<16>(score + v * 16));
    const float maximum = aie::reduce_max(peak) * 0.125f;
    emu_f32 total = aie::zeros<float, 16>();
    for (int v = 0; v < vectors; ++v) {
      const emu_f32 scaled =
          emu_mul_fb(aie::load_v<16>(score + v * 16),
                     aie::broadcast<bfloat16, 16>(0.125f));
      emu_f32 e = emu_exp(aie::sub(scaled, emu_splat(maximum)));
      aie::store_v(score + v * 16, e);
    }
    for (int i = count; i < vectors * 16; ++i)
      score[i] = 0.0f;
    for (int v = 0; v < vectors; ++v)
      total = aie::add(total, aie::load_v<16>(score + v * 16));
    const emu_f32 inv =
        emu_reciprocal(emu_splat(aie::reduce_add(total)));
    for (int v = 0; v < vectors; ++v) {
      const emu_split parts =
          emu_split3(emu_mul_ff(aie::load_v<16>(score + v * 16), inv));
      aie::store_v(p_hi + v * 16, parts.hi);
      aie::store_v(p_mid + v * 16, parts.mid);
      aie::store_v(p_lo + v * 16, parts.lo);
    }
    for (int dim = 0; dim < 64; dim += 16) {
      emu_acc result = aie::zeros<accfloat, 16>();
      for (int token = 0; token < position; ++token) {
        const emu_bf16 value = aie::load_v<16>(values + token * 64 + dim);
        result = aie::mac(result, value, aie::broadcast<bfloat16, 16>(p_lo[token]));
        result = aie::mac(result, value, aie::broadcast<bfloat16, 16>(p_mid[token]));
        result = aie::mac(result, value, aie::broadcast<bfloat16, 16>(p_hi[token]));
      }
      const emu_bf16 value = aie::load_v<16>(new_value + dim);
      result = aie::mac(result, value, aie::broadcast<bfloat16, 16>(p_lo[position]));
      result = aie::mac(result, value, aie::broadcast<bfloat16, 16>(p_mid[position]));
      result = aie::mac(result, value, aie::broadcast<bfloat16, 16>(p_hi[position]));
      aie::store_v(broadcast + head * 64 + dim,
                   emu_round(result.template to_vector<float>()));
    }
  }
}

} // extern "C"
