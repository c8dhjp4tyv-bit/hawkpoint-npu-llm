// Hub-tile kernels of the Qwen2.5 0.5B decoder engine (designs/qwen_engine.py).
//
// The hub tile has 16 KB of program memory and this file fills nearly all of
// it: the three attention passes, the current token's own bookkeeping, and the
// softmax emulation together with the core's startup code and the libgcc
// routines the core needs. Everything here is out of line or shared on
// purpose; an extra inlined copy shows up as "Overflow of program memory" at
// build time. The arithmetic is the CPU reference's, unchanged.
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
// 64-row aligned inside a tile's 192 rows, so head / 3 and head % 3 pick the
// slice and the row. Dividing by the small constant rather than by 192 keeps
// the soft-float integer division routines out of the hub program.
static inline const bfloat16 *q_head(const bfloat16 *joined, int head) {
  return joined + (head / 3) * Q_SLICE + (head % 3) * 64;
}

// Out of line on purpose: every pass scores the same keys, and an inlined
// copy per pass does not fit in the hub tile's program memory.
static __attribute__((noinline)) float q_dot64(const bfloat16 *__restrict a,
                            const bfloat16 *__restrict b) {
  aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
  total = aie::mac(total, aie::load_v<32>(a), aie::load_v<32>(b));
  total = aie::mac(total, aie::load_v<32>(a + 32), aie::load_v<32>(b + 32));
  return q_reduce(total);
}

// Attention for one KV head and its seven query heads, over a tiled cache.
//
// The cache is streamed one tile of positions at a time, so a head's history
// may be longer than the hub's data memory. The softmax needs the global
// maximum before it can exponentiate, and the sum before it can normalize,
// so each token is scored three times: once for the maximum, once for the
// sums, and once against the value tiles. Every pass adds its contribution in
// ascending position order, which keeps all FP32 accumulation, BF16 rounding,
// and 16-position lane grouping identical to the untiled kernel; a 64-token
// model uses one partially filled tile and reproduces its previous results.
static constexpr int Q_TILE = 128;
static constexpr int Q_STATE_MAX = 0;
static constexpr int Q_STATE_INV = 16;
static constexpr int Q_STATE_LANES = 32;

// The header carries the position as two BF16 parts: one slot rounds
// positions past 255 (spacing 2, then 4, then 8, ...), which would make
// attention skip the newest tokens of a long context.
static inline int q_position(const bfloat16 *__restrict lut) {
  return static_cast<int>(static_cast<float>(lut[Q_HEADER_POSITION]) +
                          static_cast<float>(lut[Q_HEADER_POSITION + 1]));
}

static inline int q_tile_count(const bfloat16 *__restrict lut, int32_t tile) {
  const int position = q_position(lut);
  int count = position - tile * Q_TILE;
  if (count > Q_TILE) count = Q_TILE;
  return count < 0 ? 0 : count;
}

// Unscaled dot products of the first ``count`` tile positions. Padding lanes
// repeat the first score, as the untiled kernel pads its last vector.
static __attribute__((noinline)) void q_tile_scores(
    const bfloat16 *__restrict query, const bfloat16 *__restrict keys,
    int count, float *__restrict score) {
  for (int token = 0; token < count; ++token)
    score[token] = q_dot64(query, keys + token * 64);
  for (int token = count; token < Q_TILE; ++token)
    score[token] = score[0];
}

// The same scores as exponentials against the global maximum. The sums and
// weights passes share this; one inlined copy per pass does not fit.
static __attribute__((noinline)) void q_tile_exps(
    const bfloat16 *__restrict query, const bfloat16 *__restrict keys,
    int count, float maximum, float *__restrict score) {
  const int vectors = (count + 15) / 16;
  q_tile_scores(query, keys, count, score);
  for (int v = 0; v < vectors; ++v) {
    const emu_f32 scaled =
        emu_mul_fb(aie::load_v<16>(score + v * 16),
                   aie::broadcast<bfloat16, 16>(0.125f));
    aie::store_v(score + v * 16,
                 emu_exp(aie::sub(scaled, emu_splat(maximum))));
  }
}

// Scores the current token while it still holds the layer's qkv rows, so the
// softmax bookkeeping starts from that score instead of waiting for a separate
// pass over the state. A maximum is order independent, so the passes that
// follow see the same value they would after a dedicated maximum kernel.
void q_attention_prepare(const bfloat16 *__restrict joined,
                         const bfloat16 *__restrict lut,
                         bfloat16 *__restrict queries,
                         bfloat16 *__restrict kv_new,
                         float *__restrict state, float *__restrict acc,
                         int32_t kv_head) {
  bfloat16 *new_key = kv_new + kv_head * 128;
  bfloat16 *new_value = new_key + 64;
#pragma clang loop unroll(disable)
  for (int q = 0; q < Q_Q_PER_KV; ++q)
    q_rope64(q_head(joined, kv_head * Q_Q_PER_KV + q), lut, queries + q * 64);
  q_rope64(q_head(joined, 14 + kv_head), lut, new_key);
  const bfloat16 *value_row = q_head(joined, 16 + kv_head);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 64; i += 32)
    aie::store_v(new_value + i, aie::load_v<32>(value_row + i));
  const emu_f32 zero = emu_splat(0.0f);
#pragma clang loop unroll(disable)
  for (int q = 0; q < Q_Q_PER_KV; ++q) {
    const int entry = kv_head * Q_Q_PER_KV + q;
    state[Q_STATE_MAX + entry] =
        q_dot64(queries + q * 64, new_key) * 0.125f;
    state[Q_STATE_INV + entry] = 0.0f;
    aie::store_v(state + Q_STATE_LANES + entry * 16, zero);
#pragma clang loop unroll(disable)
    for (int dim = 0; dim < 64; dim += 16)
      aie::store_v(acc + entry * 64 + dim, zero);
  }
}

void q_attention_max(const bfloat16 *__restrict queries,
                     const bfloat16 *__restrict keys,
                     const bfloat16 *__restrict lut, float *__restrict state,
                     int32_t kv_head, int32_t tile) {
  const int count = q_tile_count(lut, tile);
  if (count <= 0) return;
  alignas(64) float score[Q_TILE];
#pragma clang loop unroll(disable)
  for (int q = 0; q < Q_Q_PER_KV; ++q) {
    q_tile_scores(queries + q * 64, keys, count, score);
    emu_f32 peak = aie::load_v<16>(score);
    for (int v = 1; v < Q_TILE / 16; ++v)
      peak = aie::max(peak, aie::load_v<16>(score + v * 16));
    const float tile_max = aie::reduce_max(peak) * 0.125f;
    float *slot = state + Q_STATE_MAX + kv_head * Q_Q_PER_KV + q;
    if (tile_max > *slot) *slot = tile_max;
  }
}

void q_attention_sums(const bfloat16 *__restrict queries,
                      const bfloat16 *__restrict keys,
                      const bfloat16 *__restrict lut,
                      float *__restrict state, int32_t kv_head,
                      int32_t tile) {
  const int count = q_tile_count(lut, tile);
  if (count <= 0) return;
  const int vectors = (count + 15) / 16;
  alignas(64) float score[Q_TILE];
#pragma clang loop unroll(disable)
  for (int q = 0; q < Q_Q_PER_KV; ++q) {
    const int entry = kv_head * Q_Q_PER_KV + q;
    q_tile_exps(queries + q * 64, keys, count, state[Q_STATE_MAX + entry],
                score);
    for (int i = count; i < vectors * 16; ++i) score[i] = 0.0f;
    float *lanes = state + Q_STATE_LANES + entry * 16;
    emu_f32 total = aie::load_v<16>(lanes);
    for (int v = 0; v < vectors; ++v)
      total = aie::add(total, aie::load_v<16>(score + v * 16));
    aie::store_v(lanes, total);
  }
}

// The current token's exponential joins its lane and, once every lane has its
// contribution, the inverse of the summed exponentials lands in the state. The
// two used to be separate kernels; together they exceeded the hub tile's
// program memory.
void q_attention_new_sums(const bfloat16 *__restrict queries,
                          const bfloat16 *__restrict kv_new,
                          const bfloat16 *__restrict lut,
                          float *__restrict state, int32_t kv_head) {
  const bfloat16 *new_key = kv_new + kv_head * 128;
  const int lane = q_position(lut) & 15;
  alignas(64) float score[16];
  alignas(64) float parts[16];
#pragma clang loop unroll(disable)
  for (int q = 0; q < Q_Q_PER_KV; ++q) {
    const int entry = kv_head * Q_Q_PER_KV + q;
    const emu_f32 scaled =
        emu_mul_fb(emu_splat(q_dot64(queries + q * 64, new_key)),
                   aie::broadcast<bfloat16, 16>(0.125f));
    aie::store_v(score, emu_exp(aie::sub(
                           scaled, emu_splat(state[Q_STATE_MAX + entry]))));
    for (int i = 0; i < 16; ++i)
      if (i != lane) score[i] = 0.0f;
    float *lanes = state + Q_STATE_LANES + entry * 16;
    aie::store_v(lanes, aie::add(aie::load_v<16>(lanes),
                                 aie::load_v<16>(score)));
    // The last lane has its contribution, so the inverse of the summed
    // exponentials is final for this head.
    aie::store_v(parts, emu_reciprocal(
                           emu_splat(aie::reduce_add(aie::load_v<16>(
                               state + Q_STATE_LANES + entry * 16)))));
    state[Q_STATE_INV + entry] = parts[0];
  }
}

// Round 64 accumulated values to the BF16 vector attention hands on.
static __attribute__((noinline)) void q_round_row(const float *__restrict src,
                                                 bfloat16 *__restrict dst) {
#pragma clang loop unroll(disable)
  for (int dim = 0; dim < 64; dim += 16)
    aie::store_v(dst + dim, emu_round(aie::load_v<16>(src + dim)));
}

// One tile's worth of the values pass: acc += sum(weight * value), the weight
// carried as three BF16 parts so its FP32 mantissa survives the vector
// multiply. Additions stay in ascending position order.
static __attribute__((noinline)) void q_accumulate(
    const bfloat16 *__restrict hi, const bfloat16 *__restrict mid,
    const bfloat16 *__restrict lo, const bfloat16 *__restrict values,
    int count, float *__restrict slot) {
  for (int dim = 0; dim < 64; dim += 16) {
    emu_acc result;
    result.from_vector(aie::load_v<16>(slot + dim));
    for (int token = 0; token < count; ++token) {
      const emu_bf16 value = aie::load_v<16>(values + token * 64 + dim);
      result = aie::mac(result, value, aie::broadcast<bfloat16, 16>(lo[token]));
      result = aie::mac(result, value, aie::broadcast<bfloat16, 16>(mid[token]));
      result = aie::mac(result, value, aie::broadcast<bfloat16, 16>(hi[token]));
    }
    aie::store_v(slot + dim, result.template to_vector<float>());
  }
}

// Normalized weights of one tile, split into three BF16 parts per query head
// and kept so the values pass can run after the key tile is released (a delta
// FIFO holds one buffer at a time).
void q_attention_weights(const bfloat16 *__restrict queries,
                         const bfloat16 *__restrict keys,
                         const bfloat16 *__restrict lut,
                         const float *__restrict state,
                         bfloat16 *__restrict tile_weights, int32_t kv_head,
                         int32_t tile) {
  const int count = q_tile_count(lut, tile);
  if (count <= 0) return;
  const int vectors = (count + 15) / 16;
  alignas(64) float score[Q_TILE];
#pragma clang loop unroll(disable)
  for (int q = 0; q < Q_Q_PER_KV; ++q) {
    const int entry = kv_head * Q_Q_PER_KV + q;
    bfloat16 *p_hi = tile_weights + q * 3 * Q_TILE;
    bfloat16 *p_mid = p_hi + Q_TILE;
    bfloat16 *p_lo = p_mid + Q_TILE;
    q_tile_exps(queries + q * 64, keys, count, state[Q_STATE_MAX + entry],
                score);
    const emu_f32 inv = emu_splat(state[Q_STATE_INV + entry]);
    for (int v = 0; v < vectors; ++v) {
      const emu_split parts =
          emu_split3(emu_mul_ff(aie::load_v<16>(score + v * 16), inv));
      aie::store_v(p_hi + v * 16, parts.hi);
      aie::store_v(p_mid + v * 16, parts.mid);
      aie::store_v(p_lo + v * 16, parts.lo);
    }
  }
}

void q_attention_apply(const bfloat16 *__restrict tile_weights,
                       const bfloat16 *__restrict values,
                       const bfloat16 *__restrict lut, float *__restrict acc,
                       int32_t kv_head, int32_t tile) {
  const int count = q_tile_count(lut, tile);
  if (count <= 0) return;
#pragma clang loop unroll(disable)
  for (int q = 0; q < Q_Q_PER_KV; ++q) {
    const int entry = kv_head * Q_Q_PER_KV + q;
    // One head's weights at a time: the weights pass rewrites this buffer for
    // every head before its values tiles arrive.
    const bfloat16 *planes = tile_weights + q * 3 * Q_TILE;
    q_accumulate(planes, planes + Q_TILE, planes + 2 * Q_TILE, values, count,
                 acc + entry * 64);
  }
}

// The head's weighted values plus the rounding to BF16 the broadcast vector
// carries out of attention; the two were separate kernels until the hub's
// program memory ran out.
void q_attention_values_new(const bfloat16 *__restrict queries,
                            const bfloat16 *__restrict kv_new,
                            const bfloat16 *__restrict lut,
                            const float *__restrict state,
                            float *__restrict acc,
                            bfloat16 *__restrict attended, int32_t kv_head) {
  const bfloat16 *new_key = kv_new + kv_head * 128;
  const bfloat16 *new_value = new_key + 64;
#pragma clang loop unroll(disable)
  for (int q = 0; q < Q_Q_PER_KV; ++q) {
    const int entry = kv_head * Q_Q_PER_KV + q;
    const emu_f32 scaled =
        emu_mul_fb(emu_splat(q_dot64(queries + q * 64, new_key)),
                   aie::broadcast<bfloat16, 16>(0.125f));
    const emu_f32 e =
        emu_exp(aie::sub(scaled, emu_splat(state[Q_STATE_MAX + entry])));
    const emu_split parts =
        emu_split3(emu_mul_ff(e, emu_splat(state[Q_STATE_INV + entry])));
    // The weight is one token wide, but the accumulate path is the tile
    // pass's, so both add up in the same order and round at the same points.
    alignas(64) bfloat16 weight[3][16];
    aie::store_v(weight[0], parts.lo);
    aie::store_v(weight[1], parts.mid);
    aie::store_v(weight[2], parts.hi);
    float *slot = acc + entry * 64;
    q_accumulate(weight[2], weight[1], weight[0], new_value, 1, slot);
    q_round_row(slot, attended + entry * 64);
  }
}

} // extern "C"
