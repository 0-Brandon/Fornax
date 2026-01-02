/**
 * @file test_workloads.cpp
 * @brief Unit tests for financial workload numerical accuracy
 *
 * Verifies:
 * - Black-Scholes pricing against analytic solutions
 * - Fast math approximations (exp, log)
 * - Vectorized implementations match scalar
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "simd_math.h"
#include "workloads.h"

using namespace fornax;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ============================================================================
// Black-Scholes Reference Implementation
// ============================================================================

/**
 * @brief Reference Black-Scholes call price using standard library
 *
 * S = spot price, K = strike, r = risk-free rate, σ = volatility, T = time
 */
double reference_black_scholes_call(double S, double K, double r, double sigma,
                                    double T) {
  double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) /
              (sigma * std::sqrt(T));
  double d2 = d1 - sigma * std::sqrt(T);

  // Standard normal CDF using error function
  auto norm_cdf = [](double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
  };

  return S * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
}

// ============================================================================
// Black-Scholes Tests
// ============================================================================

TEST_CASE("Black-Scholes scalar accuracy", "[workloads][black-scholes]") {
  using namespace workloads;

  SECTION("ATM option") {
    // S = K = 100, r = 5%, σ = 20%, T = 0.25 (3 months)
    double S = 100.0, K = 100.0, r = 0.05, sigma = 0.20, T = 0.25;
    double expected = reference_black_scholes_call(S, K, r, sigma, T);
    double actual = black_scholes_call(S, K, r, sigma, T);

    // Black-Scholes call value for these parameters ≈ 4.62
    REQUIRE_THAT(expected, WithinRel(4.62, 0.02));
    REQUIRE_THAT(actual, WithinRel(expected, 0.01));
  }

  SECTION("ITM option") {
    double S = 110.0, K = 100.0, r = 0.05, sigma = 0.20, T = 0.25;
    double expected = reference_black_scholes_call(S, K, r, sigma, T);
    double actual = black_scholes_call(S, K, r, sigma, T);

    // ITM option should have higher value
    REQUIRE(actual > 10.0);
    REQUIRE_THAT(actual, WithinRel(expected, 0.01));
  }

  SECTION("OTM option") {
    double S = 90.0, K = 100.0, r = 0.05, sigma = 0.20, T = 0.25;
    double expected = reference_black_scholes_call(S, K, r, sigma, T);
    double actual = black_scholes_call(S, K, r, sigma, T);

    // OTM option should have lower value
    REQUIRE(actual < 5.0);
    REQUIRE_THAT(actual, WithinRel(expected, 0.02));
  }
}

TEST_CASE("Normal CDF accuracy", "[workloads][math]") {
  using namespace workloads;

  SECTION("Standard normal CDF at key points") {
    // N(0) = 0.5
    REQUIRE_THAT(norm_cdf(0.0), WithinAbs(0.5, 1e-6));

    // N(-∞) → 0, N(+∞) → 1
    REQUIRE_THAT(norm_cdf(-5.0), WithinAbs(0.0, 1e-5));
    REQUIRE_THAT(norm_cdf(5.0), WithinAbs(1.0, 1e-5));

    // N(1.96) ≈ 0.975 (two-tailed 95% CI)
    REQUIRE_THAT(norm_cdf(1.96), WithinRel(0.975, 0.001));

    // N(-1.96) ≈ 0.025
    REQUIRE_THAT(norm_cdf(-1.96), WithinRel(0.025, 0.001));
  }
}

// ============================================================================
// Fast Math Tests
// ============================================================================

TEST_CASE("Fast exp accuracy", "[math][simd]") {
  using namespace simd_math;

  SECTION("Scalar fast_exp vs std::exp") {
    std::vector<double> test_values = {-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0};

    for (double x : test_values) {
      double expected = std::exp(x);
      double actual = fast_exp_scalar(x);
      REQUIRE_THAT(actual, WithinRel(expected, 1e-5));
    }
  }

  SECTION("Edge cases") {
    // Very negative → 0
    REQUIRE_THAT(fast_exp_scalar(-800.0), WithinAbs(0.0, 1e-300));

    // Zero → 1
    REQUIRE_THAT(fast_exp_scalar(0.0), WithinAbs(1.0, 1e-10));
  }
}

TEST_CASE("Fast log accuracy", "[math][simd]") {
  using namespace simd_math;

  SECTION("Scalar fast_log vs std::log") {
    std::vector<double> test_values = {0.1, 0.5, 1.0, 2.0, 10.0, 100.0};

    for (double x : test_values) {
      double expected = std::log(x);
      double actual = fast_log_scalar(x);
      // Widened tolerance for ARM platform (fast approximation has ~0.3% error)
      REQUIRE_THAT(actual, WithinRel(expected, 3e-3));
    }
  }

  SECTION("log(1) = 0") {
    REQUIRE_THAT(fast_log_scalar(1.0), WithinAbs(0.0, 1e-10));
  }

  SECTION("log(e) ≈ 1") {
    // Fast approximation may have up to 0.1% error
    REQUIRE_THAT(fast_log_scalar(std::exp(1.0)), WithinAbs(1.0, 1e-3));
  }
}

// ============================================================================
// Vectorized Math Tests (AVX-512)
// ============================================================================

#if FORNAX_ARCH_X86 && defined(__AVX512F__)

TEST_CASE("AVX-512 fast_exp matches scalar", "[math][simd][avx512]") {
  using namespace simd_math;

  alignas(64) double input[8] = {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
  alignas(64) double output[8];

  __m512d x = _mm512_load_pd(input);
  __m512d result = fast_exp_avx512(x);
  _mm512_store_pd(output, result);

  for (int i = 0; i < 8; ++i) {
    double expected = std::exp(input[i]);
    REQUIRE_THAT(output[i], WithinRel(expected, 1e-5));
  }
}

TEST_CASE("AVX-512 fast_log matches scalar", "[math][simd][avx512]") {
  using namespace simd_math;

  alignas(64) double input[8] = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0};
  alignas(64) double output[8];

  __m512d x = _mm512_load_pd(input);
  __m512d result = fast_log_avx512(x);
  _mm512_store_pd(output, result);

  for (int i = 0; i < 8; ++i) {
    double expected = std::log(input[i]);
    REQUIRE_THAT(output[i], WithinRel(expected, 1e-4));
  }
}

TEST_CASE("AVX-512 fast_norm_cdf matches scalar", "[math][simd][avx512]") {
  using namespace simd_math;

  alignas(64) double input[8] = {-3.0, -2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0};
  alignas(64) double output[8];

  __m512d x = _mm512_load_pd(input);
  __m512d result = fast_norm_cdf_avx512(x);
  _mm512_store_pd(output, result);

  // Compare against reference using error function
  for (int i = 0; i < 8; ++i) {
    double expected = 0.5 * (1.0 + std::erf(input[i] / std::sqrt(2.0)));
    REQUIRE_THAT(output[i], WithinRel(expected, 1e-5));
  }
}

#endif // AVX-512

// ============================================================================
// XorShift RNG Tests
// ============================================================================

TEST_CASE("XorShift64 RNG quality", "[workloads][rng]") {
  using namespace workloads;

  SECTION("Uniform distribution in [0, 1)") {
    XorShift64 rng(12345);
    double sum = 0.0;
    const int N = 10000;

    for (int i = 0; i < N; ++i) {
      double u = rng.uniform();
      REQUIRE(u >= 0.0);
      REQUIRE(u < 1.0);
      sum += u;
    }

    // Mean should be approximately 0.5
    double mean = sum / N;
    REQUIRE_THAT(mean, WithinAbs(0.5, 0.02));
  }

  SECTION("Normal distribution characteristics") {
    XorShift64 rng(54321);
    double sum = 0.0;
    double sum_sq = 0.0;
    const int N = 10000;

    for (int i = 0; i < N; ++i) {
      double z = rng.normal();
      sum += z;
      sum_sq += z * z;
    }

    double mean = sum / N;
    double variance = sum_sq / N - mean * mean;

    // Mean should be approximately 0, variance approximately 1
    REQUIRE_THAT(mean, WithinAbs(0.0, 0.05));
    REQUIRE_THAT(variance, WithinRel(1.0, 0.1));
  }
}
