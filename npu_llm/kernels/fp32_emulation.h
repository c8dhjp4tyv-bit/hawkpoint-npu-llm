// FP32 arithmetic on the AIE2 vector unit for kernels that must track a
// CPU FP32 reference closely (the Qwen engine's release gate compares whole
// greedy token sequences).
//
// The AIE2 scalar unit has no floating point, and the vector unit adds,
// subtracts, and compares FP32 vectors but only multiplies BF16 operands. A
// product of two BF16 values is exact in FP32, so an FP32 value is split into
// three BF16 parts (hi + mid + lo carry all 24 mantissa bits) and products are
// accumulated from those parts in an FP32 accumulator.
//
// Conversions to BF16 round to nearest even with integer operations, so the
// result does not depend on the core's rounding-mode register.
//
// exp and the reciprocal are kept out of line to save program memory (16 KB
// per core). The helpers taking two FP32 vectors by value must stay inline:
// llvm-aie 21.0.0.2026072001 returns wrong results when they are called out
// of line.
#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

using emu_f32 = aie::vector<float, 16>;
using emu_bf16 = aie::vector<bfloat16, 16>;
using emu_acc = aie::accum<accfloat, 16>;

// Exact BF16 -> FP32 widening: the BF16 bits become the high half.
static inline emu_f32 emu_widen(emu_bf16 value) {
  const aie::vector<uint16, 16> bits = aie::vector_cast<uint16>(value);
  const auto zipped = aie::interleave_zip(aie::zeros<uint16, 16>(), bits, 1);
  return aie::vector_cast<float>(aie::concat(zipped.first, zipped.second));
}

// FP32 -> BF16 with round-to-nearest-even.
static inline emu_bf16 emu_round(emu_f32 value) {
  const aie::vector<uint32, 16> bits = aie::vector_cast<uint32>(value);
  const aie::vector<uint32, 16> lsb =
      aie::bit_and(aie::downshift(bits, 16), aie::broadcast<uint32, 16>(1u));
  const aie::vector<uint32, 16> rounded = aie::add(
      bits, aie::add(lsb, aie::broadcast<uint32, 16>(0x7fffu)));
  const aie::vector<uint16, 32> halves = aie::vector_cast<uint16>(rounded);
  return aie::vector_cast<bfloat16>(aie::filter_odd(halves, 1));
}

struct emu_split {
  emu_bf16 hi, mid, lo;
};

static inline emu_split emu_split3(emu_f32 value) {
  emu_split parts;
  parts.hi = emu_round(value);
  const emu_f32 rest = aie::sub(value, emu_widen(parts.hi));
  parts.mid = emu_round(rest);
  parts.lo = emu_round(aie::sub(rest, emu_widen(parts.mid)));
  return parts;
}

// a * b for an FP32 a and a BF16 b.
static inline emu_f32 emu_mul_fb(emu_f32 a, emu_bf16 b) {
  const emu_split s = emu_split3(a);
  emu_acc total = aie::mul(s.lo, b);
  total = aie::mac(total, s.mid, b);
  total = aie::mac(total, s.hi, b);
  return total.template to_vector<float>();
}

// a * b for two FP32 values (the lo*lo, lo*mid terms are below FP32 ulp).
static inline emu_f32 emu_mul_ff(emu_f32 a, emu_f32 b) {
  const emu_split x = emu_split3(a);
  const emu_split y = emu_split3(b);
  emu_acc total = aie::mul(x.mid, y.mid);
  total = aie::mac(total, x.hi, y.lo);
  total = aie::mac(total, x.lo, y.hi);
  total = aie::mac(total, x.hi, y.mid);
  total = aie::mac(total, x.mid, y.hi);
  total = aie::mac(total, x.hi, y.hi);
  return total.template to_vector<float>();
}

static inline emu_f32 emu_splat(float value) {
  return aie::broadcast<float, 16>(value);
}

// exp(x) for x in [-80, 80]: Cody-Waite reduction by ln 2 and a degree-7
// polynomial, accurate to a few FP32 ulps.
static __attribute__((noinline)) emu_f32 emu_exp(emu_f32 x) {
  x = aie::max(aie::min(x, emu_splat(80.0f)), emu_splat(-80.0f));
  const emu_f32 scaled = emu_mul_ff(x, emu_splat(1.44269504088896341f));
  const aie::vector<int32, 16> k = aie::to_fixed<int32>(scaled, 0);
  const emu_f32 kf = aie::to_float(k, 0);
  // kf is an integer below 256 in magnitude, so it is exact in BF16.
  const emu_bf16 kb = emu_round(kf);
  const emu_f32 ln2_hi = emu_splat(0.693145751953125f);
  const emu_f32 ln2_lo = emu_splat(1.42860682030941723e-06f);
  emu_f32 r = aie::sub(x, emu_mul_fb(ln2_hi, kb));
  r = aie::sub(r, emu_mul_fb(ln2_lo, kb));
  // Horner: 1 + r(1 + r(1/2 + r(1/6 + r(1/24 + r(1/120 + r(1/720 + r/5040))))))
  // The loop stays rolled so the FP32 product is emitted once (16 KB of
  // program memory per core).
  static const float coefficients[7] = {
      1.38888888888888889e-03f, 8.33333333333333333e-03f,
      4.16666666666666667e-02f, 1.66666666666666667e-01f,
      0.5f, 1.0f, 1.0f};
  emu_f32 p = emu_splat(1.98412698412698413e-04f);
#pragma clang loop unroll(disable)
  for (int i = 0; i < 7; ++i)
    p = aie::add(emu_mul_ff(p, r), emu_splat(coefficients[i]));
  // Scale by 2^k through the exponent bits.
  const aie::vector<int32, 16> bits = aie::add(
      aie::vector_cast<int32>(p), aie::upshift(k, 23));
  return aie::vector_cast<float>(bits);
}

// 1 / d for positive normal d, by two Newton steps from a bit-level guess.
static __attribute__((noinline)) emu_f32 emu_reciprocal(emu_f32 d) {
  const aie::vector<int32, 16> guess_bits = aie::sub(
      aie::broadcast<int32, 16>(0x7ef311c7), aie::vector_cast<int32>(d));
  emu_f32 y = aie::vector_cast<float>(guess_bits);
#pragma clang loop unroll(disable)
  for (int step = 0; step < 3; ++step) {
    const emu_f32 error = aie::sub(emu_splat(2.0f), emu_mul_ff(d, y));
    y = emu_mul_ff(y, error);
  }
  return y;
}
