// Kernels for the Qwen2.5 0.5B decoder engine (designs/qwen_engine.py).
//
// The release gate requires the NPU to reproduce a greedy token sequence of
// the CPU BF16 reference exactly, with top-1 margins as small as 0.07 logits.
// Every step therefore follows the reference's FP32 arithmetic
// (runtime/cpu_backend.py): FP32 accumulation, BF16 rounding to nearest even
// at the same points, FP32 RMSNorm, RoPE, softmax, and SiLU. The vector-unit
// FP32 emulation in fp32_emulation.h keeps these off the soft-float scalar
// path. Copy loops stay rolled and use 32-lane vectors, a form that does not
// trigger the llvm-aie hang described in engine_bf16.cc; loops are also kept
// rolled because each AIE2 core has only 16 KB of program memory. The GEMV
// tile program uses nearly all of it (about 15.9-16.2 KB), so code added to
// the GEMV kernels must be offset elsewhere; an overflow fails the build with
// "Overflow of program memory" rather than misbehaving at run time.
//
// Helpers that take two FP32 vectors by value must stay inline: llvm-aie
// 21.0.0.2026072001 miscompiles such calls when they are out of line.
#pragma once

#include "fp32_emulation.h"

static constexpr int Q_HIDDEN = 896;
static constexpr int Q_SLICE = 896;
static constexpr int Q_VECTOR = 6 * 896;
static constexpr int Q_QKV_ROWS = 192;
static constexpr int Q_PARAM_GAMMA_POST = 896;
static constexpr int Q_PARAM_BIAS = 1792;
static constexpr int Q_HEADER_SIN = 96;
static constexpr int Q_HEADER_POSITION = 192;
static constexpr int Q_HEADER_HIDDEN = 256;
static constexpr int Q_LUT = 256;
static constexpr int q_out_rows[6] = {152, 152, 148, 148, 148, 148};
static constexpr int q_out_start[6] = {0, 152, 304, 452, 600, 748};

static inline bfloat16 q_round(float value) {
  union {
    float f;
    uint32_t u;
  } input = {value};
  const uint32_t lsb = (input.u >> 16) & 1u;
  union {
    uint16_t u;
    bfloat16 b;
  } output = {static_cast<uint16_t>((input.u + 0x7fffu + lsb) >> 16)};
  return output.b;
}

static inline float q_rsqrt(float value) {
  const float half = 0.5f * value;
  union {
    float f;
    uint32_t i;
  } bits = {value};
  bits.i = 0x5f3759dfu - (bits.i >> 1);
  float estimate = bits.f;
  estimate *= 1.5f - half * estimate * estimate;
  estimate *= 1.5f - half * estimate * estimate;
  estimate *= 1.5f - half * estimate * estimate;
  return estimate;
}

// Sum of the 32 FP32 lanes of an accumulator.
static inline float q_reduce(aie::accum<accfloat, 32> total) {
  return aie::reduce_add(total.template to_vector<float>());
}

// RMSNorm as in the reference: FP32 mean of squares, inv = 1/sqrt(mean+eps),
// out = bf16((x * inv) * gamma) with FP32 rounding after each product.
static __attribute__((noinline)) void q_rmsnorm(const bfloat16 *__restrict input,
                             const bfloat16 *__restrict gamma,
                             bfloat16 *__restrict output) {
  aie::accum<accfloat, 32> squares = aie::zeros<accfloat, 32>();
#pragma clang loop unroll(disable)
  for (int i = 0; i < Q_HIDDEN; i += 32) {
    const auto x = aie::load_v<32>(input + i);
    squares = aie::mac(squares, x, x);
  }
  const float mean = q_reduce(squares) / static_cast<float>(Q_HIDDEN);
  const emu_f32 inv = emu_splat(q_rsqrt(mean + 1e-6f));
#pragma clang loop unroll(disable)
  for (int i = 0; i < Q_HIDDEN; i += 16) {
    const emu_f32 scaled = emu_mul_fb(inv, aie::load_v<16>(input + i));
    const emu_f32 result = emu_mul_fb(scaled, aie::load_v<16>(gamma + i));
    aie::store_v(output + i, emu_round(result));
  }
}

// FP32 dot products of four K=896 weight rows with the activation.
static __attribute__((noinline)) void q_dot4(const bfloat16 *__restrict weights,
                                             const bfloat16 *__restrict activation,
                                             float *__restrict sums) {
  for (int row = 0; row < 4; ++row) {
    aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
    for (int col = 0; col < Q_HIDDEN; col += 32)
      total = aie::mac(total, aie::load_v<32>(weights + row * Q_HIDDEN + col),
                       aie::load_v<32>(activation + col));
    sums[row] = q_reduce(total);
  }
}

// Four rows of a K=896 projection; each row's FP32 dot product is rounded
// to BF16 once.
static inline void q_gemv4(const bfloat16 *__restrict weights,
                           const bfloat16 *__restrict activation,
                           bfloat16 *__restrict output, int32_t offset) {
  float sums[4];
  q_dot4(weights, activation, sums);
  for (int row = 0; row < 4; ++row)
    output[offset + row] = q_round(sums[row]);
}

// out[i] = bf16(bf16_a[i] + bf16_b[i]) with one FP32 add.
static inline emu_bf16 q_add_round(emu_bf16 a, emu_bf16 b) {
  return emu_round(aie::add(emu_widen(a), emu_widen(b)));
}

