#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void kv_append_bf16(const bfloat16 *__restrict cache_in,
                                const bfloat16 *__restrict value,
                                int32_t position,
                                bfloat16 *__restrict cache_out) {
  for (int i = 0; i < 4096; i += 16)
    aie::store_v(cache_out + i, aie::load_v<16>(cache_in + i));
  const int offset = position * 64;
  for (int i = 0; i < 64; i += 16)
    aie::store_v(cache_out + offset + i, aie::load_v<16>(value + i));
}
