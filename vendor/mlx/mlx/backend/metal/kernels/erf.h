// Copyright © 2023 Apple Inc.

#pragma once
#include <metal_math>
#include "mlx/backend/metal/kernels/expm1f.h"

/*
 * Error-function approximation adapted from PyTorch's BSD-3-Clause
 * vec256_float implementation (itself the standard five-coefficient
 * Abramowitz-Stegun 7.1.26 approximation). See MLX ACKNOWLEDGMENTS.md.
 */
float erf(float a) {
  float t = metal::precise::divide(
      1.0f, metal::fma(0.3275911f, metal::abs(a), 1.0f));
  float polynomial = metal::fma(1.061405429f, t, -1.453152027f);
  polynomial = metal::fma(polynomial, t, 1.421413741f);
  polynomial = metal::fma(polynomial, t, -0.284496736f);
  polynomial = metal::fma(polynomial, t, 0.254829592f);
  float magnitude =
      metal::fma(-metal::exp(-(a * a)) * t, polynomial, 1.0f);
  return metal::copysign(magnitude, a);
}

template <metal::size_t N>
float erfinv_polynomial(float x, constant float (&coefficients)[N]) {
  float result = coefficients[N - 1];
  for (metal::size_t index = N - 1; index > 0; --index) {
    result = metal::fma(result, x, coefficients[index - 1]);
  }
  return result;
}

/*
 * Inverse-error rational approximations adapted from Boost.Math:
 * Copyright John Maddock 2006; Copyright Matt Borland 2024.
 * Source pinned at commit 6a4487453d95c1fbf5ecf3da18f2c020a89fd612:
 * https://github.com/boostorg/math/blob/6a4487453d95c1fbf5ecf3da18f2c020a89fd612/include/boost/math/special_functions/detail/erf_inv.hpp
 * See MLX ACKNOWLEDGMENTS.md for the Boost Software License 1.0 terms.
 */
constant float erfinv_p0[] = {
    -0.0005087819496582807f,
    -0.008368748197417368f,
    0.03348066254097446f,
    -0.012692614766297403f,
    -0.03656379714117627f,
    0.02198786811111689f,
    0.008226878746769158f,
    -0.00538772965071243f};
constant float erfinv_q0[] = {
    1.0f,
    -0.9700050433032906f,
    -1.5657455823417585f,
    1.5622155839842303f,
    0.662328840472003f,
    -0.7122890234154285f,
    -0.05273963823400997f,
    0.07952836873415717f,
    -0.0023339375937419002f,
    0.0008862163904563475f};
constant float erfinv_p1[] = {
    -0.20243350835593876f,
    0.10526468069939171f,
    8.3705032834312f,
    17.6447298408374f,
    -18.851064805871425f,
    -44.638232444178696f,
    17.445385985570866f,
    21.129465544834053f,
    -3.6719225470772936f};
constant float erfinv_q1[] = {
    1.0f,
    6.242641248542475f,
    3.971343795334386f,
    -28.660818049980003f,
    -20.14326346804852f,
    48.560921310873994f,
    10.826866735546016f,
    -22.643693341313973f,
    1.7211476576120028f};
constant float erfinv_p2[] = {
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
    -0.000000000681149956853777f};
constant float erfinv_q2[] = {
    1.0f,
    3.4662540724256723f,
    5.381683457070068f,
    4.778465929458438f,
    2.5930192162362027f,
    0.848854343457902f,
    0.15226433829533162f,
    0.011059242293464892f};
constant float erfinv_p3[] = {
    -0.0350353787183178f,
    -0.0022242652921344793f,
    0.018557330651423108f,
    0.009508047013259197f,
    0.0018712349281955923f,
    0.00015754461742496055f,
    0.0000046046989058431795f,
    -0.0000000002304047769118826f,
    0.0000000000026633922742578204f};
constant float erfinv_q3[] = {
    1.0f,
    1.3653349817554063f,
    0.7620591645536234f,
    0.22009110576413125f,
    0.034158914367094774f,
    0.00263861676657016f,
    0.00007646752923027945f};

/*
 * Adapted from Boost.Math's BSL-1.0 inverse-erf rational approximations by
 * John Maddock and Matt Borland. Float32 inputs reach only these four regions.
 * This replaces the prior unlicensed Stack Overflow code.
 */
float erfinv(float a) {
  float magnitude = metal::abs(a);
  if (magnitude > 1.0f) {
    return metal::sqrt(-1.0f);
  }
  if (magnitude == 1.0f) {
    return metal::copysign(1.0f / 0.0f, a);
  }
  if (magnitude == 0.0f) {
    return a;
  }

  float q = 1.0f - magnitude;
  float result;
  if (magnitude <= 0.5f) {
    float g = magnitude * (magnitude + 10.0f);
    result = g * 0.08913147449493408f +
        g * erfinv_polynomial(magnitude, erfinv_p0) /
            erfinv_polynomial(magnitude, erfinv_q0);
  } else if (q >= 0.25f) {
    float offset = q - 0.25f;
    result = metal::sqrt(-2.0f * metal::log(q)) /
        (2.249481201171875f +
         erfinv_polynomial(offset, erfinv_p1) /
             erfinv_polynomial(offset, erfinv_q1));
  } else {
    float x = metal::sqrt(-metal::log(q));
    if (x < 3.0f) {
      float offset = x - 1.125f;
      result = x *
          (0.807220458984375f +
           erfinv_polynomial(offset, erfinv_p2) /
               erfinv_polynomial(offset, erfinv_q2));
    } else {
      float offset = x - 3.0f;
      result = x *
          (0.9399557113647461f +
           erfinv_polynomial(offset, erfinv_p3) /
               erfinv_polynomial(offset, erfinv_q3));
    }
  }
  return metal::copysign(result, a);
}
