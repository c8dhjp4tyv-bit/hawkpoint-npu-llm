#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" void rope_bf16(const bfloat16 *__restrict input,
                           const bfloat16 *__restrict lut,
                           bfloat16 *__restrict output) {
  event0();
  for (int offset = 0; offset < 32; offset += 16) {
    const auto first = aie::load_v<16>(input + offset);
    const auto second = aie::load_v<16>(input + 32 + offset);
    const auto cosine = aie::load_v<16>(lut + offset);
    const auto sine = aie::load_v<16>(lut + 32 + offset);
    const aie::vector<bfloat16, 16> output_first =
        aie::sub(aie::mul(first, cosine), aie::mul(second, sine));
    const aie::vector<bfloat16, 16> output_second =
        aie::add(aie::mul(second, cosine), aie::mul(first, sine));
    aie::store_v(output + offset, output_first);
    aie::store_v(output + 32 + offset, output_second);
  }
  event1();
}
