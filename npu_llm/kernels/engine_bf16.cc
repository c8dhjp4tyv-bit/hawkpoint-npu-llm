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
static constexpr int ENG_HEAD_DIM = 64;
static constexpr int ENG_KV_HEADS = 3;
static constexpr int ENG_Q_PER_KV = 3;
static constexpr int ENG_TILE = 128;      // cached positions per streamed tile
static constexpr int ENG_ENTRIES = ENG_KV_HEADS * ENG_Q_PER_KV;
static constexpr float eng_inv_576 = 0.001736111111111111f;

// Softmax bookkeeping the hub keeps between streamed tiles, indexed by
// kv_head * ENG_Q_PER_KV + query head. Keys and values never stay resident:
// the design streams one tile of keys (twice) and one tile of values per
// layer and KV head, so the cache can be longer than the tile data memory.
static constexpr int ENG_STATE_MAX = 0;    // running score maximum (FP32)
static constexpr int ENG_STATE_INV = 16;   // 1 / sum, rounded to BF16
static constexpr int ENG_STATE_LANES = 32; // 16 FP32 partial sums per head
static constexpr int ENG_STATE_SIZE = ENG_STATE_LANES + 16 * ENG_ENTRIES;
static constexpr int ENG_ACC_SIZE = ENG_ENTRIES * ENG_HEAD_DIM;

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

} // extern "C"
