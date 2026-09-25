// Kernels for the row-split SmolLM decoder engine (designs/engine.py).
//
// Six GEMV tiles each own a fixed slice of every projection's output rows
// and receive their own weight stream. Their result slices are joined in a
// MemTile and handed to one hub tile, which runs attention and broadcasts
// every combined vector back to the GEMV tiles.
//
// The projections, RMSNorms, RoPE, residual adds, and SwiGLU compute exactly
// what kernels/decoder_layer_bf16.cc computes (BF16 residual stream, BF16
// projection outputs), and the final RMSNorm matches kernels/rmsnorm_bf16.cc
// bit for bit. Attention differs: its softmax runs on the vector unit (see
// eng_softmax_weights), because the AIE2 scalar unit has no floating point.
//
// Every fixed-length copy or elementwise loop is fully unrolled. llvm-aie
// 21.0.0.2026072001 compiles some of these loops at -O2 into a
// software-pipelined zero-overhead loop that intermittently hangs the core
// on Phoenix, depending on where the loop lands in program memory; an
// unrolled body emits no hardware loop at all. See
// docs/PERFORMANCE-ANALYSIS.md.
#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <lut_based_ops.cpp>
#include <stdint.h>

// Layout shared with designs/engine.py and runtime/engine.py.
static constexpr int ENG_HIDDEN = 576;
static constexpr int ENG_SLICE = 256;     // result slots per GEMV tile
static constexpr int ENG_QKV_ROWS = 160;  // qkv rows per GEMV tile
static constexpr int ENG_OUT_ROWS = 96;   // o/down rows per GEMV tile
static constexpr int ENG_HEADER_HIDDEN = 128;
static constexpr float eng_inv_576 = 0.001736111111111111f;

static inline float eng_rsqrt(float value) {
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

// ---- GEMV tiles ---------------------------------------------------------

void eng_load_hidden(const bfloat16 *__restrict broadcast,
                     bfloat16 *__restrict hidden) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < ENG_HIDDEN; i += 16)
    aie::store_v(hidden + i, aie::load_v<16>(broadcast + i));
}

void eng_store_hidden(const bfloat16 *__restrict hidden,
                      bfloat16 *__restrict output) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < ENG_HIDDEN; i += 16)
    aie::store_v(output + i, aie::load_v<16>(hidden + i));
}

// The first weight block of every layer carries both RMSNorm gammas. The
// second one is kept locally because the block is released immediately.
void eng_save_gamma(const bfloat16 *__restrict block,
                    bfloat16 *__restrict gamma_post) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < ENG_HIDDEN; i += 16)
    aie::store_v(gamma_post + i, aie::load_v<16>(block + ENG_HIDDEN + i));
}

static inline void eng_rmsnorm_impl(const bfloat16 *__restrict input,
                 const bfloat16 *__restrict gamma,
                 bfloat16 *__restrict output) {
  aie::accum<accfloat, 32> sum_acc = aie::zeros<accfloat, 32>();
  chess_prepare_for_pipelining chess_loop_range(18, )
  #pragma clang loop unroll(full)
  for (int i = 0; i < ENG_HIDDEN; i += 32) {
    const auto x = aie::load_v<32>(input + i);
    sum_acc = aie::add(sum_acc, aie::mul(x, x));
  }
  const float sum_sq = aie::reduce_add<float>(sum_acc);
  const float inv = eng_rsqrt(sum_sq * eng_inv_576 + 1e-5f);
  const auto inv_v = aie::broadcast<bfloat16, 16>(inv);
  chess_prepare_for_pipelining chess_loop_range(36, )
  #pragma clang loop unroll(full)
  for (int i = 0; i < ENG_HIDDEN; i += 16) {
    const auto x = aie::load_v<16>(input + i);
    const auto g = aie::load_v<16>(gamma + i);
    const auto x_scaled = aie::mul(x, inv_v).template to_vector<bfloat16>();
    const auto scaled = aie::mul(x_scaled, g).template to_vector<bfloat16>();
    aie::store_v(output + i, scaled);
  }
}

// 8 rows x 576 columns per weight block.
static inline void eng_gemv8_k576(const bfloat16 *__restrict weights,
                    const bfloat16 *__restrict activation,
                    bfloat16 *__restrict output, int32_t offset) {
  chess_prepare_for_pipelining chess_loop_range(8, )
  for (int row = 0; row < 8; ++row) {
    aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
    chess_prepare_for_pipelining chess_loop_range(18, )
    for (int col = 0; col < 576; col += 32) {
      const auto w = aie::load_v<32>(weights + row * 576 + col);
      const auto x = aie::load_v<32>(activation + col);
      total = aie::add(total, aie::mul(w, x));
    }
    output[offset + row] =
        static_cast<bfloat16>(aie::reduce_add<float>(total));
  }
}

// One C symbol per memref signature used by the IRON design.
void eng_rmsnorm_block(const bfloat16 *__restrict input,
                       const bfloat16 *__restrict gamma_block,
                       bfloat16 *__restrict output) {
  eng_rmsnorm_impl(input, gamma_block, output);
}

void eng_rmsnorm_local(const bfloat16 *__restrict input,
                       const bfloat16 *__restrict gamma,
                       bfloat16 *__restrict output) {
  eng_rmsnorm_impl(input, gamma, output);
}

void eng_gemv8_hidden(const bfloat16 *__restrict weights,
                      const bfloat16 *__restrict activation,
                      bfloat16 *__restrict output, int32_t offset) {
  eng_gemv8_k576(weights, activation, output, offset);
}

void eng_gemv8_vector(const bfloat16 *__restrict weights,
                      const bfloat16 *__restrict activation,
                      bfloat16 *__restrict output, int32_t offset) {
  eng_gemv8_k576(weights, activation, output, offset);
}

void eng_gemv8_up(const bfloat16 *__restrict weights,
                  const bfloat16 *__restrict activation,
                  bfloat16 *__restrict output, int32_t offset) {
  eng_gemv8_k576(weights, activation, output, offset);
}

// 3 rows x 1536 columns per weight block (down projection).
void eng_gemv3_k1536(const bfloat16 *__restrict weights,
                     const bfloat16 *__restrict activation,
                     bfloat16 *__restrict output, int32_t offset) {
  for (int row = 0; row < 3; ++row) {
    aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
    chess_prepare_for_pipelining chess_loop_range(48, )
    for (int col = 0; col < 1536; col += 32) {
      const auto w = aie::load_v<32>(weights + row * 1536 + col);
      const auto x = aie::load_v<32>(activation + col);
      total = aie::add(total, aie::mul(w, x));
    }
    output[offset + row] =
        static_cast<bfloat16>(aie::reduce_add<float>(total));
  }
}

// output[offset..offset+16) holds 16 gate rows; up holds the matching rows.
void eng_swiglu16(const bfloat16 *__restrict up, bfloat16 *__restrict output,
                  int32_t offset) {
  const auto half = aie::broadcast<bfloat16, 16>(0.5f);
  const auto one = aie::broadcast<bfloat16, 16>(1.0f);
  const auto gate = aie::load_v<16>(output + offset);
  const auto u = aie::load_v<16>(up);
  const aie::vector<bfloat16, 16> half_gate = aie::mul(gate, half);
  const aie::vector<bfloat16, 16> tanh_half = getTanhBf16(half_gate);
  const aie::vector<bfloat16, 16> sigmoid =
      aie::mul(aie::add(tanh_half, one), half);
  const aie::vector<bfloat16, 16> silu = aie::mul(gate, sigmoid);
  const aie::vector<bfloat16, 16> result = aie::mul(silu, u);
  aie::store_v(output + offset, result);
}

// hidden[r] += joined output row r, where the six tiles each produced 96
// consecutive rows at slot tile * 256.
void eng_residual96(bfloat16 *__restrict hidden,
                    const bfloat16 *__restrict broadcast) {
  #pragma clang loop unroll(full)
  for (int tile = 0; tile < 6; ++tile) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < ENG_OUT_ROWS; i += 16) {
      bfloat16 *h = hidden + tile * ENG_OUT_ROWS + i;
      const aie::vector<bfloat16, 16> sum = aie::add(
          aie::load_v<16>(h), aie::load_v<16>(broadcast + tile * ENG_SLICE + i));
      aie::store_v(h, sum);
    }
  }
}

void eng_copy_attended(const bfloat16 *__restrict broadcast,
                       bfloat16 *__restrict x) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < ENG_HIDDEN; i += 16)
    aie::store_v(x + i, aie::load_v<16>(broadcast + i));
}

void eng_copy_mlp(const bfloat16 *__restrict broadcast,
                  bfloat16 *__restrict mlp) {
  #pragma clang loop unroll(full)
  for (int i = 0; i < 1536; i += 32)
    aie::store_v(mlp + i, aie::load_v<32>(broadcast + i));
}

// Final RMSNorm, bit-identical to kernels/rmsnorm_bf16.cc (the generic
// normalizer the chunked decoder path runs before the LM head): a sequential
// FP32 sum of squares, a true division, and one rounding per output.
void eng_final_norm(const bfloat16 *__restrict input,
                    const bfloat16 *__restrict gamma_block,
                    bfloat16 *__restrict output) {
  float sum_sq = 0.0f;
  for (int i = 0; i < ENG_HIDDEN; ++i) {
    const float value = static_cast<float>(input[i]);
    sum_sq += value * value;
  }
  const float inv_rms = eng_rsqrt(sum_sq / ENG_HIDDEN + 1e-5f);
  for (int i = 0; i < ENG_HIDDEN; ++i)
    output[i] = static_cast<bfloat16>(static_cast<float>(input[i]) * inv_rms *
                                      static_cast<float>(gamma_block[i]));
}

// ---- Hub tile -----------------------------------------------------------

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

// Attention for one KV head and its three query heads. The cache object
// holds keys [64 x 64] then values [64 x 64] for positions before the
// current one; the current key/value come from this token's qkv and are
// also emitted in kv_new so the host can append them to the cache.
//
// The AIE2 scalar unit has no floating-point hardware, so every scalar float
// operation is a soft-float call; the per-position softmax arithmetic in
// kernels/decoder_layer_bf16.cc makes attention the most expensive stage at
// late positions. Here the softmax runs 16 positions at a time on the vector
// unit: the score offset is taken in FP32, the exponential comes from the
// BF16 lookup tables in lut_based_ops, and the weights are normalized once.
static inline void eng_softmax_weights(float *__restrict score, int count,
                                       bfloat16 *__restrict weights) {
  const int vectors = (count + 15) / 16;
  // Padding lanes repeat the first score so the maximum is unaffected; their
  // weights are cleared below.
  for (int i = count; i < vectors * 16; ++i)
    score[i] = score[0];
  aie::vector<float, 16> peak = aie::load_v<16>(score);
  for (int v = 1; v < vectors; ++v)
    peak = aie::max(peak, aie::load_v<16>(score + v * 16));
  const aie::vector<float, 16> maximum =
      aie::broadcast<float, 16>(aie::reduce_max(peak));
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
  aie::accum<accfloat, 16> total = aie::zeros<accfloat, 16>();
  for (int v = 0; v < vectors; ++v)
    total = aie::add(total, aie::load_v<16>(weights + v * 16));
  const float sum = aie::reduce_add(total.template to_vector<float>());
  const aie::vector<bfloat16, 16> inv =
      aie::broadcast<bfloat16, 16>(getInvBf16(sum));
  for (int v = 0; v < vectors; ++v)
    aie::store_v(weights + v * 16,
                 aie::mul(aie::load_v<16>(weights + v * 16), inv)
                     .template to_vector<bfloat16>());
}

void eng_attention(const bfloat16 *__restrict joined,
                   const bfloat16 *__restrict cache,
                   const bfloat16 *__restrict lut,
                   bfloat16 *__restrict broadcast,
                   bfloat16 *__restrict kv_new, int32_t kv_head) {
  const int32_t position = static_cast<int32_t>(static_cast<float>(lut[64]));
  bfloat16 *new_key = kv_new + kv_head * 128;
  bfloat16 *new_value = new_key + 64;
  eng_gather_head(joined, 576 + kv_head * 64, new_key);
  eng_rope64(new_key, lut);
  eng_gather_head(joined, 768 + kv_head * 64, new_value);
  const bfloat16 *keys = cache;
  const bfloat16 *values = cache + 4096;

  alignas(32) bfloat16 query[64];
  alignas(64) float score[64];
  alignas(32) bfloat16 weights[64];
  for (int head = kv_head * 3; head < kv_head * 3 + 3; ++head) {
    eng_gather_head(joined, head * 64, query);
    eng_rope64(query, lut);
    for (int token = 0; token < position; ++token)
      score[token] = eng_dot64(query, keys + token * 64);
    score[position] = eng_dot64(query, new_key);
    eng_softmax_weights(score, position + 1, weights);
    for (int dim = 0; dim < 64; dim += 16) {
      aie::accum<accfloat, 16> result = aie::zeros<accfloat, 16>();
      for (int token = 0; token < position; ++token) {
        const auto value = aie::load_v<16>(values + token * 64 + dim);
        result = aie::add(
            result, aie::mul(value, aie::broadcast<bfloat16, 16>(weights[token])));
      }
      result = aie::add(result,
                        aie::mul(aie::load_v<16>(new_value + dim),
                                 aie::broadcast<bfloat16, 16>(weights[position])));
      aie::store_v(broadcast + head * 64 + dim,
                   result.template to_vector<bfloat16>());
    }
  }
}

} // extern "C"
