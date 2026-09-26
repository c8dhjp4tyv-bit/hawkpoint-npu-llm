// GEMV-tile kernels of the Qwen2.5 0.5B decoder engine (designs/qwen_engine.py).
// The hub tile's kernels live in qwen_engine_hub_bf16.cc so that neither tile's
// program carries the other's code (each AIE2 core has 16 KB of program memory).
#include "qwen_engine_common.h"

extern "C" {

// ---- GEMV tiles ---------------------------------------------------------

// Copy the first 896 values of a broadcast vector (the initial hidden state,
// or the attended heads in round 2).
void q_load_hidden(const bfloat16 *__restrict broadcast,
                   bfloat16 *__restrict hidden) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < Q_HIDDEN; i += 32)
    aie::store_v(hidden + i, aie::load_v<32>(broadcast + i));
}

void q_norm_params(const bfloat16 *__restrict hidden,
                   const bfloat16 *__restrict params,
                   bfloat16 *__restrict output) {
  q_rmsnorm(hidden, params, output);
}

void q_norm_local(const bfloat16 *__restrict hidden,
                  const bfloat16 *__restrict gamma,
                  bfloat16 *__restrict output) {
  q_rmsnorm(hidden, gamma, output);
}

// Keep the post-attention gamma and this tile's FP32 qkv bias; the parameter
// block is released right after the input RMSNorm.
void q_save_params(const bfloat16 *__restrict params,
                   bfloat16 *__restrict gamma_post,
                   float *__restrict bias_f32) {
  bfloat16 *bias = reinterpret_cast<bfloat16 *>(bias_f32);
#pragma clang loop unroll(disable)
  for (int i = 0; i < Q_HIDDEN; i += 32)
    aie::store_v(gamma_post + i,
                 aie::load_v<32>(params + Q_PARAM_GAMMA_POST + i));
#pragma clang loop unroll(disable)
  for (int i = 0; i < 2 * Q_QKV_ROWS; i += 32)
    aie::store_v(bias + i, aie::load_v<32>(params + Q_PARAM_BIAS + i));
}

void q_gemv_slice(const bfloat16 *__restrict weights,
                  const bfloat16 *__restrict activation,
                  bfloat16 *__restrict output, int32_t offset) {
  q_gemv4(weights, activation, output, offset);
}

void q_gemv_gate(const bfloat16 *__restrict weights,
                 const bfloat16 *__restrict activation,
                 bfloat16 *__restrict output, int32_t offset) {
  q_gemv4(weights, activation, output, offset);
}

// qkv = bf16(bf16(projection) + bias) for this tile's 192 rows.
void q_add_bias(bfloat16 *__restrict slice, const float *__restrict bias_f32) {
#pragma clang loop unroll(disable)
  for (int i = 0; i < Q_QKV_ROWS; i += 16) {
    const emu_f32 sum =
        aie::add(emu_widen(aie::load_v<16>(slice + i)), aie::load_v<16>(bias_f32 + i));
    aie::store_v(slice + i, emu_round(sum));
  }
}

// hidden[r] = bf16(hidden[r] + projection[r]) where tile t produced rows
// [q_out_start[t], +q_out_rows[t]) at slot t * 896. Those row ranges are not
// 16-aligned, so the slices are first gathered into `scratch` with integer
// copies (no float work on the scalar unit), then added as aligned vectors.
void q_residual(bfloat16 *__restrict hidden,
                const bfloat16 *__restrict broadcast,
                bfloat16 *__restrict scratch) {
  const uint16_t *source = reinterpret_cast<const uint16_t *>(broadcast);
  uint16_t *gathered = reinterpret_cast<uint16_t *>(scratch);
#pragma clang loop unroll(disable)
  for (int tile = 0; tile < 6; ++tile)
#pragma clang loop unroll(disable)
    for (int i = 0; i < q_out_rows[tile]; ++i)
      gathered[q_out_start[tile] + i] = source[tile * Q_SLICE + i];
#pragma clang loop unroll(disable)
  for (int i = 0; i < Q_HIDDEN; i += 16)
    aie::store_v(hidden + i, q_add_round(aie::load_v<16>(hidden + i),
                                         aie::load_v<16>(scratch + i)));
}

// SiLU(gate) * up for 16 values, as the reference computes it:
// bf16((gate / (1 + exp(-clip(gate, -80, 80)))) * up).
void q_silu16(const bfloat16 *__restrict gate, const bfloat16 *__restrict up,
              bfloat16 *__restrict output, int32_t offset) {
  const emu_f32 g = emu_widen(aie::load_v<16>(gate));
  const emu_f32 e = emu_exp(aie::sub(aie::zeros<float, 16>(), g));
  const emu_f32 denominator = aie::add(emu_splat(1.0f), e);
  const emu_f32 silu = emu_mul_ff(g, emu_reciprocal(denominator));
  aie::store_v(output + offset,
               emu_round(emu_mul_fb(silu, aie::load_v<16>(up))));
}

// Four LM-head rows; logits stay FP32 because BF16 would hide the small
// top-1 margins the release gate depends on.
void q_gemv_logits(const bfloat16 *__restrict weights,
                   const bfloat16 *__restrict activation,
                   float *__restrict output, int32_t offset) {
  q_dot4(weights, activation, output + offset);
}

// Zero the slots of an MLP slice past this tile's rows. The down projection
// multiplies those slots by zero weights, and stale memory could hold NaN or
// Inf bit patterns, which would turn 0 * x into NaN.
void q_zero_tail(bfloat16 *__restrict slice, int32_t start) {
#pragma clang loop unroll(disable)
  for (int i = start; i < Q_SLICE; i += 16)
    aie::store_v(slice + i, aie::zeros<bfloat16, 16>());
}

// Down projection: four rows accumulated over six 896-column chunks.
// Partial sums persist in `partial` (4 rows x 32 FP32 lanes) between calls.
void q_down_chunk(const bfloat16 *__restrict weights,
                  const bfloat16 *__restrict mlp, float *__restrict partial,
                  int32_t chunk) {
  const bfloat16 *activation = mlp + chunk * Q_SLICE;
  for (int row = 0; row < 4; ++row) {
    aie::accum<accfloat, 32> total;
    if (chunk == 0)
      total = aie::zeros<accfloat, 32>();
    else
      total.from_vector(aie::load_v<32>(partial + row * 32));
    for (int col = 0; col < Q_SLICE; col += 32)
      total = aie::mac(total, aie::load_v<32>(weights + row * Q_SLICE + col),
                       aie::load_v<32>(activation + col));
    aie::store_v(partial + row * 32, total.template to_vector<float>());
  }
}

void q_down_finish(const float *__restrict partial,
                   bfloat16 *__restrict output, int32_t offset) {
  for (int row = 0; row < 4; ++row)
    output[offset + row] =
        q_round(aie::reduce_add(aie::load_v<32>(partial + row * 32)));
}

} // extern "C"
