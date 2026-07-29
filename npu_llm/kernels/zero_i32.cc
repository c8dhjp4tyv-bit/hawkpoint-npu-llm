#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 32
#endif

extern "C" void zero_w8a16_i32(int32_t *__restrict c) {
  aie::store_v(c, aie::zeros<int32_t, DIM_M>());
}
