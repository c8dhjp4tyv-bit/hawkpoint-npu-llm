// ---------------------------------------------------------------------------
// qwen_decoder_layer_bf16.cc — AIE2 kernel implementations for Qwen2.5-0.5B
// ---------------------------------------------------------------------------
// Compiled by Peano/MLIR-AIE and linked into the xclbin. All kernels operate
// on BF16 data and use FP32 accumulators internally, converting back to BF16
// with explicit round-to-nearest-even for deterministic CPU/NPU agreement.
//
// MODEL CONSTANTS (baked into the IRON graph in qwen_decoder.py):
//   HIDDEN       = 896    (hidden dimension)
//   INTERMEDIATE = 4864   (SwiGLU expanded dimension)
//   HEAD_DIM     = 64     (per-head dimension)
//   Q_HEADS      = 14     (query heads)
//   KV_HEADS     = 2      (key/value heads)
//   RMS_EPS      = 1e-6
//   SCALE        = 1/sqrt(64) = 0.125
//
// KV CACHE LAYOUT (per layer, per KV head, in XRT buffer objects):
//   [0..4095]  = K cache: 64 positions × 64 dims of BF16
//   [4096..8191] = V cache: 64 positions × 64 dims of BF16
// The 64-position limit is currently hard-coded in the IRON graph and kernel
// buffer dimensions; the actual storage lives in NPU/XRT buffer objects
// allocated at runtime. Extending the window requires resizing the XRT
// allocation, rebalancing the weight stream layout across tiles, and
// recompiling the xclbin — it is a graph/kernel parameter, not a
// physical tile-memory ceiling.
//
// PRECISION NOTES:
// - All operations accumulate in FP32 via aie::accum<accfloat,32>
// - BF16→Float→BF16 round-trips use explicit RNE (bit-level, not hardware H_EXTEND)
// - softmax uses online-safe shift (exp(x-max)) to prevent under/overflow

#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <lut_based_ops.cpp>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Fast inverse square root — three Newton iterations.
// Used by RMSNorm; the three iterations are calibrated so the CPU reference
// path and the NPU path produce identical BF16 argmax results across every
// position in the 64-token acceptance window, as verified by the checked-in
// CPU BF16 reference sequence in the release gate.
static inline float qwen_rsqrt(float value) {
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

// ---------------------------------------------------------------------------
// Float→BF16 conversion with explicit round-to-nearest-even (RNE).
// Manual bit-level RNE instead of compiler __bfloat16 cast to guarantee
// deterministic BF16 values matching the CPU NumPy reference path.
// Critical for token-acceptance tests that compare NPU vs CPU logits.
static inline bfloat16 qwen_bf16_rne(float value) {
  union {
    float f;
    uint32_t u;
  } input = {value};
  const uint32_t least_significant = (input.u >> 16) & 1u;
  const uint16_t rounded =
      static_cast<uint16_t>(
          (input.u + 0x7fffu + least_significant) >> 16);
  union {
    uint16_t u;
    bfloat16 value;
  } output = {rounded};
  return output.value;
}

// ---------------------------------------------------------------------------
// Template: project_precise<ROWS, COLS> — generalized matrix-vector product.
//   Computes ROWS-wide output from a [ROWS × COLS] weight tile and a [COLS]
//   activation vector. Accumulates in four interleaved accfloat banks to hide
//   AIE2 data-movement latency. Finally consolidates all lanes via FP32 sum
//   and converts back to BF16 with round-to-nearest-even.
// Parameters:
//   ROWS  — number of output rows to compute (e.g., 16 for QKV slice, 4 for down_proj)
//   COLS  — activation width (e.g., 896 for hidden dim, 4864 for intermediate)
// Memory layout: weights[row * COLS + col], activation[col], output[offset + row]
template <unsigned ROWS, unsigned COLS>
static inline void project_precise(const bfloat16 *__restrict weights,
                                   const bfloat16 *__restrict activation,
                                   bfloat16 *__restrict output,
                                   int32_t offset) {
  for (unsigned row = 0; row < ROWS; ++row) {
    aie::accum<accfloat, 32> total0 = aie::zeros<accfloat, 32>();
    aie::accum<accfloat, 32> total1 = aie::zeros<accfloat, 32>();
    aie::accum<accfloat, 32> total2 = aie::zeros<accfloat, 32>();
    aie::accum<accfloat, 32> total3 = aie::zeros<accfloat, 32>();
    for (unsigned col = 0; col < COLS; col += 32) {
      const auto w = aie::load_v<32>(weights + row * COLS + col);
      const auto x = aie::load_v<32>(activation + col);
      const auto product = aie::mul(w, x);
      switch ((col / 32) & 3) {
      case 0: total0 = aie::add(total0, product); break;
      case 1: total1 = aie::add(total1, product); break;
      case 2: total2 = aie::add(total2, product); break;
      default: total3 = aie::add(total3, product); break;
      }
    }
    const auto p0 = total0.template to_vector<float>();
    const auto p1 = total1.template to_vector<float>();
    const auto p2 = total2.template to_vector<float>();
    const auto p3 = total3.template to_vector<float>();
    float sum = 0.0f;
    for (unsigned lane = 0; lane < 32; ++lane)
      sum += p0[lane] + p1[lane] + p2[lane] + p3[lane];
    output[offset + row] = qwen_bf16_rne(sum);
  }
}

// ---------------------------------------------------------------------------
// Softmax helper: exp(x) via ln(2)-based decomposition (exp = 2^k × e^r).
// Uses a degree-5 Taylor polynomial for the remainder term, good to ~1e-6
// relative error on [-0.7, +0.7]. Range-clamped to [-80, +80] to prevent
// FP32 overflow. This avoids a memory-intensive LUT which would compete
// with weight/cache data for tile-local memory.
static inline float qwen_exp(float x) {
  if (x < -80.0f)
    return 0.0f;
  if (x > 80.0f)
    x = 80.0f;
  const float scaled = x * 1.4426950408889634f;
  const int exponent =
      static_cast<int>(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
  const float remainder = x - exponent * 0.6931471805599453f;
  const float r2 = remainder * remainder;
  const float polynomial =
      1.0f + remainder + r2 * (
          0.5f + remainder * (
              0.1666666666666667f + remainder * (
                  0.0416666666666667f + remainder * (
                      0.0083333333333333f + remainder *
                      0.0013888888888889f))));
  union {
    uint32_t i;
    float f;
  } power_of_two = {static_cast<uint32_t>(exponent + 127) << 23};
  return polynomial * power_of_two.f;
}

// ---------------------------------------------------------------------------
// EXPORTED KERNEL ENTRY POINTS (extern "C" — discovered by IRON via ExternalFunction)
// ---------------------------------------------------------------------------
extern "C" {

void qwen_copy896(const bfloat16 *__restrict input,
                  bfloat16 *__restrict output) {
  for (int i = 0; i < 896; i += 16)
    aie::store_v(output + i, aie::load_v<16>(input + i));
}

void qwen_rmsnorm896(const bfloat16 *__restrict input,
                     const bfloat16 *__restrict gamma,
                     bfloat16 *__restrict output) {
  float sum_sq = 0.0f;
  for (int i = 0; i < 896; ++i) {
    const float x = static_cast<float>(input[i]);
    sum_sq += x * x;
  }
  const float inv = qwen_rsqrt(sum_sq / 896.0f + 1e-6f);
  for (int i = 0; i < 896; ++i)
    output[i] = qwen_bf16_rne(
        static_cast<float>(input[i]) * inv * static_cast<float>(gamma[i]));
}

void qwen_project16_k896_bf16(const bfloat16 *__restrict weights,
                              const bfloat16 *__restrict activation,
                              bfloat16 *__restrict output, int32_t offset) {
  project_precise<16, 896>(weights, activation, output, offset);
}

void qwen_project4_k4864_bf16(const bfloat16 *__restrict weights,
                              const bfloat16 *__restrict activation,
                              bfloat16 *__restrict output, int32_t offset) {
  project_precise<4, 4864>(weights, activation, output, offset);
}

// kind: 0=Q (head 0..13), 1=K (head 0..1), 2=V (head 0..1).
void qwen_pack_rope64(const bfloat16 *__restrict row,
                      const bfloat16 *__restrict lut,
                      const bfloat16 *__restrict bias,
                      bfloat16 *__restrict packed, int32_t head,
                      int32_t kind) {
  const int group_width = 578;
  int packed_offset;
  int bias_offset;
  if (kind == 0) {
    packed_offset = (head / 7) * group_width + (head % 7) * 64;
    bias_offset = head * 64;
  } else {
    packed_offset = head * group_width + (kind == 1 ? 448 : 512);
    bias_offset = (kind == 1 ? 896 : 1024) + head * 64;
  }
  bfloat16 *dst = packed + packed_offset;
  if (kind == 2) {
    for (int i = 0; i < 64; ++i)
      dst[i] = qwen_bf16_rne(
          static_cast<float>(row[i]) +
          static_cast<float>(bias[bias_offset + i]) +
          static_cast<float>(bias[1152 + bias_offset + i]));
    packed[head * group_width + 576] = lut[64];
    return;
  }
  for (int i = 0; i < 32; ++i) {
    const float first =
        static_cast<float>(row[i]) +
        static_cast<float>(bias[bias_offset + i]) +
        static_cast<float>(bias[1152 + bias_offset + i]);
    const float second =
        static_cast<float>(row[32 + i]) +
        static_cast<float>(bias[bias_offset + 32 + i]) +
        static_cast<float>(bias[1152 + bias_offset + 32 + i]);
    const float cosine = static_cast<float>(lut[i]);
    const float sine = static_cast<float>(lut[32 + i]);
    dst[i] = qwen_bf16_rne(first * cosine - second * sine);
    dst[32 + i] =
        qwen_bf16_rne(first * sine + second * cosine);
  }
}

void qwen_attention7(const bfloat16 *__restrict packed_qkv,
                     const bfloat16 *__restrict cache_in,
                     bfloat16 *__restrict cache_out,
                     bfloat16 *__restrict output) {
  const int32_t position =
      static_cast<int32_t>(static_cast<float>(packed_qkv[576]));
  const bfloat16 *queries = packed_qkv;
  const bfloat16 *new_key = packed_qkv + 448;
  const bfloat16 *new_value = packed_qkv + 512;
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
  for (int head = 0; head < 7; ++head) {
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
      score[token] = qwen_exp(score[token] - maximum);
      sum += score[token];
    }
    const float inv = 1.0f / sum;
    for (int dim = 0; dim < 64; ++dim) {
      float result = 0.0f;
      for (int token = 0; token <= position; ++token)
        result += score[token] * inv *
                  static_cast<float>(values[token * 64 + dim]);
      output[head * 64 + dim] = qwen_bf16_rne(result);
    }
  }
}

void qwen_residual64(const bfloat16 *__restrict projection,
                     const bfloat16 *__restrict residual,
                     bfloat16 *__restrict output, int32_t offset) {
  for (int i = 0; i < 64; ++i)
    output[offset + i] = qwen_bf16_rne(
        static_cast<float>(projection[i]) +
        static_cast<float>(residual[offset + i]));
}

void qwen_store_gate64(const bfloat16 *__restrict gate,
                       bfloat16 *__restrict output, int32_t offset) {
  for (int i = 0; i < 64; i += 16)
    aie::store_v(output + offset + i, aie::load_v<16>(gate + i));
}

void qwen_swiglu_up64(const bfloat16 *__restrict up,
                      bfloat16 *__restrict output, int32_t offset) {
  for (int i = 0; i < 64; ++i) {
    const float gate = static_cast<float>(output[offset + i]);
    const float value = static_cast<float>(up[i]);
    const float silu = gate / (1.0f + qwen_exp(-gate));
    output[offset + i] = qwen_bf16_rne(silu * value);
  }
}

void qwen_residual32(const bfloat16 *__restrict projection,
                     const bfloat16 *__restrict residual,
                     bfloat16 *__restrict output, int32_t offset) {
  for (int i = 0; i < 32; ++i)
    output[offset + i] = qwen_bf16_rne(
        static_cast<float>(projection[i]) +
        static_cast<float>(residual[offset + i]));
}

} // extern "C"
