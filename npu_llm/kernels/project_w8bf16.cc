#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 576
#endif

extern "C" {

void project_w8bf16_acc32(const int8_t *__restrict weights,
                          const bfloat16 *__restrict activation,
                          float *__restrict output) {
  for (int row = 0; row < DIM_M; ++row) {
    aie::accum<accfloat, 32> total = aie::zeros<accfloat, 32>();
    for (int col = 0; col < DIM_K; col += 32) {
      const auto w_i8 = aie::load_v<32>(weights + row * DIM_K + col);
      const auto w_bf16 = aie::to_float<bfloat16>(w_i8);
      const auto x_bf16 = aie::load_v<32>(activation + col);
      total = aie::add(total, aie::mul(w_bf16, x_bf16));
    }
    const auto partial = total.template to_vector<float>();
    float sum = 0.0f;
    for (int lane = 0; lane < 32; ++lane)
      sum += partial[lane];
    output[row] = sum;
  }
}

void project_dequant_bf16(const float *__restrict input,
                          const float *__restrict scale,
                          bfloat16 *__restrict output) {
  for (int row = 0; row < DIM_M; ++row)
    output[row] = static_cast<bfloat16>(input[row] * scale[row]);
}

void project_residual_add_bf16(const bfloat16 *__restrict projection,
                               const bfloat16 *__restrict residual,
                               bfloat16 *__restrict output) {
  for (int row = 0; row < DIM_M; ++row)
    output[row] = static_cast<bfloat16>(
        static_cast<float>(projection[row]) +
        static_cast<float>(residual[row]));
}

} // extern "C"
