#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 896
#endif

extern "C" {

void project_bf16_block_precise(const bfloat16 *__restrict weights,
                                const bfloat16 *__restrict activation,
                                bfloat16 *__restrict output) {
  for (int row = 0; row < DIM_M; ++row) {
    aie::accum<accfloat, 32> total0 = aie::zeros<accfloat, 32>();
    aie::accum<accfloat, 32> total1 = aie::zeros<accfloat, 32>();
    aie::accum<accfloat, 32> total2 = aie::zeros<accfloat, 32>();
    aie::accum<accfloat, 32> total3 = aie::zeros<accfloat, 32>();
    for (int col = 0; col < DIM_K; col += 32) {
      const auto w = aie::load_v<32>(weights + row * DIM_K + col);
      const auto x = aie::load_v<32>(activation + col);
      const auto product = aie::mul(w, x);
      switch ((col / 32) & 3) {
      case 0: total0 = aie::add(total0, product); break;
      case 1: total1 = aie::add(total1, product); break;
      case 2: total2 = aie::add(total2, product); break;
      case 3: total3 = aie::add(total3, product); break;
      }
    }
    const auto partial0 = total0.template to_vector<float>();
    const auto partial1 = total1.template to_vector<float>();
    const auto partial2 = total2.template to_vector<float>();
    const auto partial3 = total3.template to_vector<float>();
    float sum = 0.0f;
    for (int lane = 0; lane < 32; ++lane)
      sum += partial0[lane] + partial1[lane] + partial2[lane]
          + partial3[lane];
    output[row] = static_cast<bfloat16>(sum);
  }
}

void project_bf16_residual(const bfloat16 *__restrict projection,
                           const bfloat16 *__restrict residual,
                           bfloat16 *__restrict output) {
  for (int row = 0; row < DIM_M; ++row)
    output[row] = static_cast<bfloat16>(
        static_cast<float>(projection[row]) +
        static_cast<float>(residual[row]));
}

} // extern "C"
