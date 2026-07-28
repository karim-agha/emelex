// Copyright © 2024 Apple Inc.

#pragma once

#include <array>

#include "mlx/backend/cpu/simd/type.h"

namespace mlx::core::simd {

constexpr float inf = std::numeric_limits<float>::infinity();

/**
 * Compute exp(x) in an optimizer friendly way as follows:
 *
 * First change the problem to computing 2**y where y = x / ln(2).
 *
 * Now we will compute 2**y as 2**y1 * 2**y2 where y1 is the integer part
 * `ipart` and y2 is fractional part. For the integer part we perform bit
 * shifting and for the fractional part we use a polynomial approximation.
 *
 * The algorithm and constants of the polynomial taken from
 * https://github.com/akohlmey/fastermath/blob/master/src/exp.c which took them
 * from Cephes math library.
 *
 * Note: The implementation below is a general fast exp. There could be faster
 *       implementations for numbers strictly < 0.
 */
template <typename T, int N>
Simd<T, N> exp(Simd<T, N> in) {
  if constexpr (is_complex<T>) {
    return Simd<T, 1>{std::exp(in.value)};
  } else {
    Simd<float, N> x_init = in;
    auto x = x_init * 1.442695f; // multiply with log_2(e)
    Simd<float, N> ipart, fpart;
    ipart = floor(x + 0.5);
    fpart = x - ipart;

    x = 1.535336188319500e-4f;
    x = fma(x, fpart, 1.339887440266574e-3f);
    x = fma(x, fpart, 9.618437357674640e-3f);
    x = fma(x, fpart, 5.550332471162809e-2f);
    x = fma(x, fpart, 2.402264791363012e-1f);
    x = fma(x, fpart, 6.931472028550421e-1f);
    x = fma(x, fpart, 1.000000000000000f);

    // generate 2**ipart in the floating point representation using integer
    // bitshifting
    Simd<int, N> epart = (Simd<int, N>(ipart) + 127) << 23;

    // Deal with NaN and Inf
    auto result = select(isnan(x_init), x_init, (*(Simd<float, N>*)&epart) * x);
    result = select(x_init > 88.0f, Simd<float, N>(inf), result);
    result = select(x_init < -88.0f, Simd<float, N>(0), result);
    return Simd<T, N>(result);
  }
}

/* Implementation from:
 * https://github.com/JishinMaster/simd_utils/blob/3c1433a86fb38edcc9b02039f3c9a65b16640976/neon_mathfun.h#L357
 * which originally came from the Cephes math library.
 */
template <bool Sine, typename T, int N>
Simd<T, N> sincos(Simd<T, N> in) {
  auto sign_mask_sin = in < 0;
  in = abs(in);
  Simd<float, N> x = in;

  // scale by 4/Pi
  auto y = x * 1.27323954473516f;

  // store the integer part of y in mm0
  Simd<uint32_t, N> emm2 = y;

  // j=(j+1) & (~1) (see the cephes sources)
  emm2 = emm2 + 1;
  emm2 = emm2 & ~1;

  y = emm2;

  // Get the polynom selection mask. There is one polynom for 0 <= x <= Pi/4
  // and another one for Pi/4<x<=Pi/2. Both branches will be computed.
  auto poly_mask = (emm2 & 2) != 0;

  // The magic pass: "Extended precision modular arithmetic"
  // x = ((x - y * DP1) - y * DP2) - y * DP3
  x = fma(y, Simd<float, N>(-0.78515625f), x);
  x = fma(y, Simd<float, N>(-2.4187564849853515625e-4f), x);
  x = fma(y, Simd<float, N>(-3.77489497744594108e-8f), x);

  sign_mask_sin = sign_mask_sin ^ ((emm2 & 4) != 0);
  auto sign_mask_cos = ((emm2 - 2) & 4) != 0;

  // Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
  // and the second polynom      (Pi/4 <= x <= 0) in y2
  auto z = x * x;

  auto y1 =
      fma(z, Simd<float, N>(2.443315711809948e-5f), -1.388731625493765e-3f);
  auto y2 = fma(z, Simd<float, N>(-1.9515295891e-4f), 8.3321608736e-3f);
  y1 = fma(y1, z, 4.166664568298827e-2f);
  y2 = fma(y2, z, -1.6666654611e-1f);
  y1 = y1 * z;
  y2 = y2 * z;
  y1 = y1 * z;
  y2 = fma(x, y2, x);
  y1 = fma(z, Simd<float, N>(-0.5f), y1);
  y1 = y1 + 1.0f;

  if constexpr (Sine) {
    auto ys = select(poly_mask, y1, y2);
    return select(sign_mask_sin, -ys, ys);
  } else {
    auto yc = select(poly_mask, y2, y1);
    return select(sign_mask_cos, yc, -yc);
  }
}

template <typename T, int N>
Simd<T, N> sin(Simd<T, N> x) {
  if constexpr (is_complex<T>) {
    return std::sin(x.value);
  } else {
    return sincos<true>(x);
  }
}

template <typename T, int N>
Simd<T, N> cos(Simd<T, N> x) {
  if constexpr (is_complex<T>) {
    return std::cos(x.value);
  } else {
    return sincos<false>(x);
  }
}

template <typename T, int N>
Simd<T, N> erf(Simd<T, N> x) {
  // https://github.com/pytorch/pytorch/blob/abf28982a8cb43342e7669d859de9543fd804cc9/aten/src/ATen/cpu/vec/vec256/vec256_float.h#L175
  Simd<float, N> v = x;
  auto t = recip(fma(Simd<float, N>(0.3275911f), abs(v), 1.0f));
  auto r = fma(Simd<float, N>(1.061405429f), t, -1.453152027f);
  r = fma(r, t, 1.421413741f);
  r = fma(r, t, -0.284496736f);
  r = fma(r, t, 0.254829592f);
  auto e = -exp(-v * v);
  auto result = Simd<T, N>(fma(e * t, r, 1.0f));
  return select(x > 0, result, -result);
}

template <int N, std::size_t M>
Simd<float, N> erfinv_polynomial(
    Simd<float, N> x,
    const std::array<float, M>& coefficients) {
  Simd<float, N> result = coefficients.back();
  for (std::size_t index = M - 1; index > 0; --index) {
    result = fma(result, x, coefficients[index - 1]);
  }
  return result;
}

template <typename T, int N>
Simd<T, N> erfinv(Simd<T, N> a_) {
  // Adapted from Boost.Math's BSL-1.0 inverse-erf rational
  // approximations by John Maddock (2006) and Matt Borland (2024):
  // https://github.com/boostorg/math/blob/6a4487453d95c1fbf5ecf3da18f2c020a89fd612/include/boost/math/special_functions/detail/erf_inv.hpp
  // Float32 inputs only reach the first four regions below. See
  // MLX ACKNOWLEDGMENTS.md for full provenance and license terms.
  Simd<float, N> a = a_;
  auto magnitude = abs(a);
  auto q = 1.0f - magnitude;

  auto p0 = erfinv_polynomial<N>(
      magnitude,
      std::array{
          -0.0005087819496582807f,
          -0.008368748197417368f,
          0.03348066254097446f,
          -0.012692614766297403f,
          -0.03656379714117627f,
          0.02198786811111689f,
          0.008226878746769158f,
          -0.00538772965071243f});
  auto q0 = erfinv_polynomial<N>(
      magnitude,
      std::array{
          1.0f,
          -0.9700050433032906f,
          -1.5657455823417585f,
          1.5622155839842303f,
          0.662328840472003f,
          -0.7122890234154285f,
          -0.05273963823400997f,
          0.07952836873415717f,
          -0.0023339375937419002f,
          0.0008862163904563475f});
  auto g0 = magnitude * (magnitude + 10.0f);
  auto central = g0 * 0.08913147449493408f + g0 * p0 / q0;

  auto q_offset = q - 0.25f;
  auto p1 = erfinv_polynomial<N>(
      q_offset,
      std::array{
          -0.20243350835593876f,
          0.10526468069939171f,
          8.3705032834312f,
          17.6447298408374f,
          -18.851064805871425f,
          -44.638232444178696f,
          17.445385985570866f,
          21.129465544834053f,
          -3.6719225470772936f});
  auto q1 = erfinv_polynomial<N>(
      q_offset,
      std::array{
          1.0f,
          6.242641248542475f,
          3.971343795334386f,
          -28.660818049980003f,
          -20.14326346804852f,
          48.560921310873994f,
          10.826866735546016f,
          -22.643693341313973f,
          1.7211476576120028f});
  auto moderate =
      sqrt(-2.0f * log(q)) / (2.249481201171875f + p1 / q1);

  auto tail_x = sqrt(-log(q));
  auto near_offset = tail_x - 1.125f;
  auto p2 = erfinv_polynomial<N>(
      near_offset,
      std::array{
          -0.1311027816799519f,
          -0.16379404719331706f,
          0.11703015634199525f,
          0.38707973897260434f,
          0.3377855389120359f,
          0.14286953440815716f,
          0.029015791000532906f,
          0.0021455899538880528f,
          -0.0000006794655751811264f,
          0.000000028522533178220558f,
          -0.000000000681149956853777f});
  auto q2 = erfinv_polynomial<N>(
      near_offset,
      std::array{
          1.0f,
          3.4662540724256723f,
          5.381683457070068f,
          4.778465929458438f,
          2.5930192162362027f,
          0.848854343457902f,
          0.15226433829533162f,
          0.011059242293464892f});
  auto near_tail =
      tail_x * (0.807220458984375f + p2 / q2);

  auto far_offset = tail_x - 3.0f;
  auto p3 = erfinv_polynomial<N>(
      far_offset,
      std::array{
          -0.0350353787183178f,
          -0.0022242652921344793f,
          0.018557330651423108f,
          0.009508047013259197f,
          0.0018712349281955923f,
          0.00015754461742496055f,
          0.0000046046989058431795f,
          -0.0000000002304047769118826f,
          0.0000000000026633922742578204f});
  auto q3 = erfinv_polynomial<N>(
      far_offset,
      std::array{
          1.0f,
          1.3653349817554063f,
          0.7620591645536234f,
          0.22009110576413125f,
          0.034158914367094774f,
          0.00263861676657016f,
          0.00007646752923027945f});
  auto far_tail =
      tail_x * (0.9399557113647461f + p3 / q3);

  auto result = select(
      magnitude <= 0.5f,
      central,
      select(q >= 0.25f, moderate, select(tail_x < 3.0f, near_tail, far_tail)));
  result = select(a < 0.0f, -result, result);
  result = select(magnitude == 0.0f, a, result);
  result = select(
      magnitude == 1.0f,
      select(a < 0.0f, Simd<float, N>(-inf), Simd<float, N>(inf)),
      result);
  result = select(
      magnitude > 1.0f,
      Simd<float, N>(std::numeric_limits<float>::quiet_NaN()),
      result);
  result = select(isnan(a), a, result);
  return Simd<T, N>(result);
}

} // namespace mlx::core::simd
