// Hub-tile kernels of the row-split SmolLM decoder engine
// (designs/engine.py): the final RMSNorm handoff, the initial broadcast, and
// attention over the streamed K/V cache.
//
// Attention streams the cache in position tiles, so a head's history may be
// longer than the tile's data memory. Summing the softmax needs the global
// maximum before any exponential, and the sum before any normalization, so a
// token is scored three times: for the maximum, for the exponential sums, and
// against the value tiles. Every pass contributes in ascending position order,
// which keeps all FP32 accumulation, BF16 rounding, and 16-position lane
// grouping identical to the untiled kernel; a 64-token model uses one
// partially filled tile and reproduces its previous results exactly.
//
// This file is separate from kernels/engine_bf16.cc so the GEMV tiles do not
// carry attention code, and it is compiled with -Os: ten small kernels plus
// their helpers must fit a core's 16 KB of program memory, which -O2 exceeds.
#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <lut_based_ops.cpp>
#include <stdint.h>

static constexpr int ENG_HIDDEN = 576;
static constexpr int ENG_SLICE = 256;     // result slots per GEMV tile
static constexpr int ENG_QKV_ROWS = 160;  // qkv rows per GEMV tile
static constexpr int ENG_HEADER_HIDDEN = 128;
static constexpr int ENG_HEAD_DIM = 64;
static constexpr int ENG_KV_HEADS = 3;
static constexpr int ENG_Q_PER_KV = 3;
static constexpr int ENG_TILE = 128;
static constexpr int ENG_ENTRIES = ENG_KV_HEADS * ENG_Q_PER_KV;

static constexpr int ENG_STATE_MAX = 0;
static constexpr int ENG_STATE_INV = 16;
static constexpr int ENG_STATE_LANES = 32;

static inline float eng_dot64(const bfloat16 *__restrict lhs,
                              const bfloat16 *__restrict rhs) {
  aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
  #pragma clang loop unroll(full)
  for (int dim = 0; dim < 64; dim += 32) {
    const auto a = aie::load_v<32>(lhs + dim);
    const auto b = aie::load_v<32>(rhs + dim);
    total = aie::add(total, aie::mul(a, b));
  }
  return aie::reduce_add<float>(total);
}

// Copy one 64-value qkv head out of the joined slices. qkv row r of the
// layer lives at slot (r / 160) * 256 + r % 160; 160 and 64 are multiples of
// 16, so every 16-value chunk stays inside one slice.
static inline void eng_gather_head(const bfloat16 *__restrict joined,
                                   int first_row, bfloat16 *__restrict dst) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < 64; i += 16) {
    const int row = first_row + i;
    const int slot = (row / ENG_QKV_ROWS) * ENG_SLICE + row % ENG_QKV_ROWS;
    aie::store_v(dst + i, aie::load_v<16>(joined + slot));
  }
}

static inline void eng_rope64(bfloat16 *__restrict row,
                              const bfloat16 *__restrict lut) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < 32; i += 16) {
    const auto first = aie::load_v<16>(row + i);
    const auto second = aie::load_v<16>(row + 32 + i);
    const auto cosine = aie::load_v<16>(lut + i);
    const auto sine = aie::load_v<16>(lut + 32 + i);
    const aie::vector<bfloat16, 16> first_out =
        aie::sub(aie::mul(first, cosine), aie::mul(second, sine));
    const aie::vector<bfloat16, 16> second_out =
        aie::add(aie::mul(first, sine), aie::mul(second, cosine));
    aie::store_v(row + i, first_out);
    aie::store_v(row + 32 + i, second_out);
  }
}

extern "C" {

// The first cache-stream object is a header: [LUT (66) | pad | hidden (576)].
void eng_hub_init(const bfloat16 *__restrict header, bfloat16 *__restrict lut,
                  bfloat16 *__restrict broadcast) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < 80; i += 16)
    aie::store_v(lut + i, aie::load_v<16>(header + i));
  #pragma clang loop unroll(full)
  for (int i = 0; i < ENG_HIDDEN; i += 16)
    aie::store_v(broadcast + i,
                 aie::load_v<16>(header + ENG_HEADER_HIDDEN + i));
}

void eng_copy1536(const bfloat16 *__restrict input,
                  bfloat16 *__restrict output) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < 1536; i += 32)
    aie::store_v(output + i, aie::load_v<32>(input + i));
}

// Attention for one KV head and its three query heads over a tiled cache.
//
// The host streams one tile of keys (or values) per DMA transfer, so a head's
// cache may be longer than the hub's data memory. Summing the softmax needs
// the global maximum before any exponential, so a token is scored in three
// passes over the cache: the maximum, then the exponential sums, then the
// weighted values. Each pass adds its contribution in ascending position
// order, which keeps every FP32 accumulation, every BF16 rounding point, and
// every lane grouping identical to the untiled kernel: a 64-token model uses
// one partially filled tile and reproduces its previous results exactly.
//
// The AIE2 scalar unit has no floating-point hardware, so every scalar float
// operation is a soft-float call; the per-position softmax arithmetic in
// kernels/decoder_layer_bf16.cc makes attention the most expensive stage at
// late positions. Here the softmax runs 16 positions at a time on the vector
// unit: the score offset is taken in FP32, the exponential comes from the
// BF16 lookup tables in lut_based_ops, and the weights are normalized once.
// The header carries the position as two BF16 parts: one BF16 slot rounds
// positions past 255 (spacing 2, then 4, then 8, ...), which would make
// attention skip the newest tokens of a long context.
static inline int eng_position(const bfloat16 *__restrict lut) {
  return static_cast<int>(static_cast<float>(lut[64]) +
                          static_cast<float>(lut[65]));
}

static inline int eng_tile_count(const bfloat16 *__restrict lut, int32_t tile) {
  const int position = eng_position(lut);
  int count = position - tile * ENG_TILE;
  if (count > ENG_TILE) count = ENG_TILE;
  return count < 0 ? 0 : count;
}

// Scores of the first ``count`` positions of a tile. The remaining lanes
// repeat the first score, exactly as eng_softmax_weights pads the last
// vector, so the maximum ignores them.
// Out of line on purpose: the hub calls these from eight small kernels and
// its 16 KB of program memory cannot hold eight inlined copies.
static __attribute__((noinline)) void eng_tile_scores(
    const bfloat16 *__restrict query, const bfloat16 *__restrict keys,
    int count, float *__restrict score) {
  for (int token = 0; token < count; ++token)
    score[token] = eng_dot64(query, keys + token * 64);
  for (int token = count; token < ENG_TILE; ++token)
    score[token] = score[0];
}

static __attribute__((noinline)) void eng_tile_exp(
    const float *__restrict score, const float *__restrict state, int entry,
    int count, bfloat16 *__restrict weights) {
  const int vectors = (count + 15) / 16;
  const aie::vector<float, 16> maximum =
      aie::broadcast<float, 16>(state[ENG_STATE_MAX + entry]);
  const aie::vector<bfloat16, 16> eighth = aie::broadcast<bfloat16, 16>(0.125f);
  for (int v = 0; v < vectors; ++v) {
    aie::accum<accfloat, 16> diff;
    diff.from_vector(aie::sub(aie::load_v<16>(score + v * 16), maximum));
    const aie::vector<bfloat16, 16> scaled =
        aie::mul(diff.template to_vector<bfloat16>(), eighth)
            .template to_vector<bfloat16>();
    const aie::accum<accfloat, 16> e = getExpBf16(scaled);
    aie::store_v(weights + v * 16, e.template to_vector<bfloat16>());
  }
  for (int i = count; i < vectors * 16; ++i)
    weights[i] = static_cast<bfloat16>(0.0f);
}

static inline float eng_tile_dot(const bfloat16 *__restrict query,
                                 const bfloat16 *__restrict key) {
  return eng_dot64(query, key);
}

// Gathers the three queries of a KV head (with RoPE) and this token's key and
// value into kv_new, then clears the softmax bookkeeping for the head.
void eng_attention_prepare(const bfloat16 *__restrict joined,
                           const bfloat16 *__restrict lut,
                           bfloat16 *__restrict queries,
                           bfloat16 *__restrict kv_new, float *__restrict state,
                           float *__restrict acc, int32_t kv_head) {
  bfloat16 *new_key = kv_new + kv_head * 128;
  bfloat16 *new_value = new_key + 64;
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    bfloat16 *query = queries + q * 64;
    eng_gather_head(joined, (kv_head * ENG_Q_PER_KV + q) * 64, query);
    eng_rope64(query, lut);
  }
  eng_gather_head(joined, 576 + kv_head * 64, new_key);
  eng_rope64(new_key, lut);
  eng_gather_head(joined, 768 + kv_head * 64, new_value);
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    const int entry = kv_head * ENG_Q_PER_KV + q;
    state[ENG_STATE_MAX + entry] = -3.4e38f;
    state[ENG_STATE_INV + entry] = 0.0f;
    for (int lane = 0; lane < 16; ++lane)
      state[ENG_STATE_LANES + entry * 16 + lane] = 0.0f;
    for (int dim = 0; dim < ENG_HEAD_DIM; ++dim)
      acc[entry * ENG_HEAD_DIM + dim] = 0.0f;
  }
}

void eng_attention_max(const bfloat16 *__restrict queries,
                       const bfloat16 *__restrict keys,
                       const bfloat16 *__restrict lut, float *__restrict state,
                       int32_t kv_head, int32_t tile) {
  const int count = eng_tile_count(lut, tile);
  if (count <= 0) return;
  alignas(64) float score[ENG_TILE];
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    eng_tile_scores(queries + q * 64, keys, count, score);
    aie::vector<float, 16> peak = aie::load_v<16>(score);
    for (int v = 1; v < ENG_TILE / 16; ++v)
      peak = aie::max(peak, aie::load_v<16>(score + v * 16));
    const float tile_max = aie::reduce_max(peak);
    float *slot = state + ENG_STATE_MAX + kv_head * ENG_Q_PER_KV + q;
    if (tile_max > *slot) *slot = tile_max;
  }
}

void eng_attention_max_new(const bfloat16 *__restrict queries,
                           const bfloat16 *__restrict kv_new,
                           float *__restrict state, int32_t kv_head) {
  const bfloat16 *new_key = kv_new + kv_head * 128;
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    const float value = eng_tile_dot(queries + q * 64, new_key);
    float *slot = state + ENG_STATE_MAX + kv_head * ENG_Q_PER_KV + q;
    if (value > *slot) *slot = value;
  }
}

void eng_attention_sums(const bfloat16 *__restrict queries,
                        const bfloat16 *__restrict keys,
                        const bfloat16 *__restrict lut,
                        float *__restrict state, int32_t kv_head,
                        int32_t tile) {
  const int count = eng_tile_count(lut, tile);
  if (count <= 0) return;
  const int vectors = (count + 15) / 16;
  alignas(64) float score[ENG_TILE];
  alignas(32) bfloat16 weights[ENG_TILE];
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    const int entry = kv_head * ENG_Q_PER_KV + q;
    eng_tile_scores(queries + q * 64, keys, count, score);
    eng_tile_exp(score, state, entry, count, weights);
    float *lanes = state + ENG_STATE_LANES + entry * 16;
    aie::accum<accfloat, 16> total;
    total.from_vector(aie::load_v<16>(lanes));
    for (int v = 0; v < vectors; ++v)
      total = aie::add(total, aie::load_v<16>(weights + v * 16));
    aie::store_v(lanes, total.template to_vector<float>());
  }
}

void eng_attention_sums_new(const bfloat16 *__restrict queries,
                            const bfloat16 *__restrict kv_new,
                            const bfloat16 *__restrict lut,
                            float *__restrict state, int32_t kv_head) {
  const bfloat16 *new_key = kv_new + kv_head * 128;
  const int lane = eng_position(lut) % 16;
  alignas(32) bfloat16 one[16];
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    const int entry = kv_head * ENG_Q_PER_KV + q;
    const aie::vector<float, 16> maximum =
        aie::broadcast<float, 16>(state[ENG_STATE_MAX + entry]);
    const aie::vector<bfloat16, 16> eighth = aie::broadcast<bfloat16, 16>(0.125f);
    aie::accum<accfloat, 16> diff;
    diff.from_vector(
        aie::sub(aie::broadcast<float, 16>(eng_tile_dot(queries + q * 64, new_key)),
                 maximum));
    const aie::vector<bfloat16, 16> scaled =
        aie::mul(diff.template to_vector<bfloat16>(), eighth)
            .template to_vector<bfloat16>();
    const aie::accum<accfloat, 16> probability = getExpBf16(scaled);
    aie::store_v(one, probability.template to_vector<bfloat16>());
    for (int i = 0; i < 16; ++i)
      if (i != lane) one[i] = static_cast<bfloat16>(0.0f);
    float *lanes = state + ENG_STATE_LANES + entry * 16;
    aie::accum<accfloat, 16> total;
    total.from_vector(aie::load_v<16>(lanes));
    total = aie::add(total, aie::load_v<16>(one));
    aie::store_v(lanes, total.template to_vector<float>());
  }
}

void eng_attention_sums_finish(float *__restrict state, int32_t kv_head) {
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    const int entry = kv_head * ENG_Q_PER_KV + q;
    aie::accum<accfloat, 16> total;
    total.from_vector(
        aie::load_v<16>(state + ENG_STATE_LANES + entry * 16));
    const float sum = aie::reduce_add(total.template to_vector<float>());
    state[ENG_STATE_INV + entry] =
        static_cast<float>(static_cast<bfloat16>(getInvBf16(sum)));
  }
}

// Keyed by KV head: the normalized weights of the tile, kept so the values
// pass can run after the key tile has been released (a tile FIFO holds one
// buffer at a time).
void eng_attention_weights(const bfloat16 *__restrict queries,
                           const bfloat16 *__restrict keys,
                           const bfloat16 *__restrict lut,
                           const float *__restrict state,
                           bfloat16 *__restrict tile_weights, int32_t kv_head,
                           int32_t tile) {
  const int count = eng_tile_count(lut, tile);
  if (count <= 0) return;
  const int vectors = (count + 15) / 16;
  alignas(64) float score[ENG_TILE];
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    const int entry = kv_head * ENG_Q_PER_KV + q;
    bfloat16 *weights = tile_weights + q * ENG_TILE;
    eng_tile_scores(queries + q * 64, keys, count, score);
    eng_tile_exp(score, state, entry, count, weights);
    const aie::vector<bfloat16, 16> inv = aie::broadcast<bfloat16, 16>(
        static_cast<bfloat16>(state[ENG_STATE_INV + entry]));
    for (int v = 0; v < vectors; ++v)
      aie::store_v(weights + v * 16,
                   aie::mul(aie::load_v<16>(weights + v * 16), inv)
                       .template to_vector<bfloat16>());
  }
}

void eng_attention_apply(const bfloat16 *__restrict tile_weights,
                         const bfloat16 *__restrict values,
                         const bfloat16 *__restrict lut, float *__restrict acc,
                         int32_t kv_head, int32_t tile) {
  const int count = eng_tile_count(lut, tile);
  if (count <= 0) return;
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    const int entry = kv_head * ENG_Q_PER_KV + q;
    const bfloat16 *weights = tile_weights + q * ENG_TILE;
    float *slot = acc + entry * ENG_HEAD_DIM;
    for (int dim = 0; dim < ENG_HEAD_DIM; dim += 16) {
      aie::accum<accfloat, 16> result;
      result.from_vector(aie::load_v<16>(slot + dim));
      for (int token = 0; token < count; ++token) {
        const auto value = aie::load_v<16>(values + token * 64 + dim);
        result = aie::add(
            result, aie::mul(value, aie::broadcast<bfloat16, 16>(weights[token])));
      }
      aie::store_v(slot + dim, result.template to_vector<float>());
    }
  }
}

void eng_attention_values_new(const bfloat16 *__restrict queries,
                              const bfloat16 *__restrict kv_new,
                              const bfloat16 *__restrict lut,
                              const float *__restrict state,
                              float *__restrict acc, int32_t kv_head) {
  const bfloat16 *new_key = kv_new + kv_head * 128;
  const bfloat16 *new_value = new_key + 64;
  const aie::vector<bfloat16, 16> eighth = aie::broadcast<bfloat16, 16>(0.125f);
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    const int entry = kv_head * ENG_Q_PER_KV + q;
    aie::accum<accfloat, 16> diff;
    diff.from_vector(aie::sub(
        aie::broadcast<float, 16>(eng_tile_dot(queries + q * 64, new_key)),
        aie::broadcast<float, 16>(state[ENG_STATE_MAX + entry])));
    const aie::vector<bfloat16, 16> scaled =
        aie::mul(diff.template to_vector<bfloat16>(), eighth)
            .template to_vector<bfloat16>();
    alignas(32) bfloat16 e[16];
    const aie::accum<accfloat, 16> probability = getExpBf16(scaled);
    aie::store_v(e, probability.template to_vector<bfloat16>());
    alignas(32) bfloat16 weight[16];
    aie::store_v(
        weight,
        aie::mul(aie::broadcast<bfloat16, 16>(e[0]),
                 aie::broadcast<bfloat16, 16>(
                     static_cast<bfloat16>(state[ENG_STATE_INV + entry])))
            .template to_vector<bfloat16>());
    const aie::vector<bfloat16, 16> w = aie::broadcast<bfloat16, 16>(weight[0]);
    float *slot = acc + entry * ENG_HEAD_DIM;
    for (int dim = 0; dim < ENG_HEAD_DIM; dim += 16) {
      aie::accum<accfloat, 16> result;
      result.from_vector(aie::load_v<16>(slot + dim));
      result = aie::add(result, aie::mul(aie::load_v<16>(new_value + dim), w));
      aie::store_v(slot + dim, result.template to_vector<float>());
    }
  }
}

void eng_attention_emit(const float *__restrict acc,
                        bfloat16 *__restrict broadcast, int32_t kv_head) {
  for (int q = 0; q < ENG_Q_PER_KV; ++q) {
    const int entry = kv_head * ENG_Q_PER_KV + q;
    const float *slot = acc + entry * ENG_HEAD_DIM;
    for (int dim = 0; dim < ENG_HEAD_DIM; dim += 16) {
      aie::accum<accfloat, 16> result;
      result.from_vector(aie::load_v<16>(slot + dim));
      aie::store_v(broadcast + entry * ENG_HEAD_DIM + dim,
                   result.template to_vector<bfloat16>());
    }
  }
}

} // extern "C"
