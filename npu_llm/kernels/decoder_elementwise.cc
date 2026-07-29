#include <aie_api/aie.hpp>
#include <lut_based_ops.h>
#include <lut_based_ops.cpp>
#include <stdint.h>

#ifndef TILE_SIZE
#define TILE_SIZE 64
#endif

extern "C" {

void quantize_bf16_i16(const bfloat16 *__restrict input,
                       int16_t *__restrict output) {
  for (int i = 0; i < TILE_SIZE; ++i) {
    float value = static_cast<float>(input[i]) * 256.0f;
    if (value > 32767.0f)
      value = 32767.0f;
    if (value < -32767.0f)
      value = -32767.0f;
    output[i] = static_cast<int16_t>(value);
  }
}

void dequant_i32_bf16(const int32_t *__restrict input,
                      const float *__restrict weight_scale,
                      bfloat16 *__restrict output) {
  for (int i = 0; i < TILE_SIZE; ++i)
    output[i] =
        static_cast<bfloat16>(input[i] * weight_scale[i] * (1.0f / 256.0f));
}

void residual_add_bf16(const bfloat16 *__restrict a,
                       const bfloat16 *__restrict b,
                       bfloat16 *__restrict output) {
  for (int i = 0; i < TILE_SIZE; i += 16) {
    const auto av = aie::load_v<16>(a + i);
    const auto bv = aie::load_v<16>(b + i);
    aie::store_v(output + i, aie::add(av, bv));
  }
}

void swiglu_split_bf16(const bfloat16 *__restrict gate,
                       const bfloat16 *__restrict up,
                       bfloat16 *__restrict output) {
  const auto half = aie::broadcast<bfloat16, 16>(0.5f);
  const auto one = aie::broadcast<bfloat16, 16>(1.0f);
  for (int i = 0; i < TILE_SIZE; i += 16) {
    const auto g = aie::load_v<16>(gate + i);
    const auto u = aie::load_v<16>(up + i);
    const aie::vector<bfloat16, 16> half_gate = aie::mul(g, half);
    const aie::vector<bfloat16, 16> tanh_half = getTanhBf16(half_gate);
    const aie::vector<bfloat16, 16> sigmoid =
        aie::mul(aie::add(tanh_half, one), half);
    const aie::vector<bfloat16, 16> silu = aie::mul(g, sigmoid);
    const aie::vector<bfloat16, 16> result = aie::mul(silu, u);
    aie::store_v(output + i, result);
  }
}

} // extern "C"
