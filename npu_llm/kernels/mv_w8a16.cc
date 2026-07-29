// INT8-weight / INT16-activation GEMV microkernel for XDNA1/AIE2.

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

#ifndef DIM_K
#define DIM_K 64
#endif

extern "C" {

void matvec_w8a16_scalar(int8_t *__restrict a, int16_t *__restrict b,
                         int32_t *__restrict c) {
  event0();
  for (int row = 0; row < DIM_M; ++row) {
    int32_t sum = 0;
    for (int col = 0; col < DIM_K; ++col)
      sum += static_cast<int32_t>(a[row * DIM_K + col]) *
             static_cast<int32_t>(b[col]);
    c[row] += sum;
  }
  event1();
}

void matvec_w8a16_vectorized(int8_t *__restrict a, int16_t *__restrict b,
                             int32_t *__restrict c) {
  static_assert(DIM_K % 32 == 0);
  event0();
  for (int row = 0; row < DIM_M; ++row) {
    aie::accum<acc32, 32> total = aie::zeros<acc32, 32>();
    for (int col = 0; col < DIM_K; col += 32) {
      const aie::vector<int8_t, 32> weights =
          aie::load_v<32>(a + row * DIM_K + col);
      const aie::vector<int16_t, 32> activation = aie::load_v<32>(b + col);
      total = aie::add(total, aie::mul(weights, activation));
    }
    c[row] += aie::reduce_add(total.template to_vector<int32_t>());
  }
  event1();
}

} // extern "C"
