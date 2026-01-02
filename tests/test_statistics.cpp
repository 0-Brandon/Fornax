/**
 * @file test_statistics.cpp
 * @brief Unit tests for enhanced statistics calculations
 *
 * Verifies:
 * - Mean/stddev calculations
 * - Percentile calculations (P50, P95, P99, P99.9)
 * - IQR and outlier detection
 * - Edge cases (empty, single element, duplicates)
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "statistics.h"

using namespace fornax;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ============================================================================
// Basic Statistics Tests
// ============================================================================

TEST_CASE("Mean calculation", "[statistics]") {
  SECTION("Simple mean") {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.mean, WithinAbs(3.0, 1e-10));
  }

  SECTION("Single value") {
    std::vector<double> values = {42.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.mean, WithinAbs(42.0, 1e-10));
  }

  SECTION("Empty vector") {
    std::vector<double> values = {};
    auto stats = compute_enhanced_stats(values);
    REQUIRE(stats.n == 0);
    REQUIRE_THAT(stats.mean, WithinAbs(0.0, 1e-10));
  }

  SECTION("Negative values") {
    std::vector<double> values = {-5.0, -3.0, -1.0, 1.0, 3.0, 5.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.mean, WithinAbs(0.0, 1e-10));
  }
}

TEST_CASE("Standard deviation calculation", "[statistics]") {
  SECTION("Known stddev") {
    // Values with known stddev
    std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    auto stats = compute_enhanced_stats(values);
    // Sample stddev = sqrt(32/7) ≈ 2.138
    REQUIRE_THAT(stats.stddev, WithinRel(2.138, 0.01));
  }

  SECTION("Identical values have zero stddev") {
    std::vector<double> values = {5.0, 5.0, 5.0, 5.0, 5.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.stddev, WithinAbs(0.0, 1e-10));
  }

  SECTION("Single value has zero stddev") {
    std::vector<double> values = {42.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.stddev, WithinAbs(0.0, 1e-10));
  }
}

// ============================================================================
// Percentile Tests
// ============================================================================

TEST_CASE("Percentile calculations", "[statistics][percentiles]") {
  SECTION("Median of odd-length array") {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.p50, WithinAbs(3.0, 1e-10));
  }

  SECTION("Median of even-length array (interpolated)") {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.p50, WithinAbs(2.5, 1e-10));
  }

  SECTION("P99 on 100 elements") {
    std::vector<double> values;
    for (int i = 1; i <= 100; ++i) {
      values.push_back(static_cast<double>(i));
    }
    auto stats = compute_enhanced_stats(values);
    // P99 should be around 99
    REQUIRE_THAT(stats.p99, WithinRel(99.0, 0.02));
  }

  SECTION("Quartiles (P25, P75)") {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto stats = compute_enhanced_stats(values);
    // P25 ≈ 2.75, P75 ≈ 6.25
    REQUIRE_THAT(stats.p25, WithinRel(2.75, 0.1));
    REQUIRE_THAT(stats.p75, WithinRel(6.25, 0.1));
  }
}

TEST_CASE("IQR calculation", "[statistics][iqr]") {
  SECTION("Known IQR") {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    auto stats = compute_enhanced_stats(values);
    // IQR = P75 - P25
    REQUIRE_THAT(stats.iqr, WithinRel(stats.p75 - stats.p25, 1e-10));
  }

  SECTION("Equal values have zero IQR") {
    std::vector<double> values = {5.0, 5.0, 5.0, 5.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.iqr, WithinAbs(0.0, 1e-10));
  }
}

TEST_CASE("Outlier detection", "[statistics][outliers]") {
  SECTION("Extreme values are outliers") {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0, 100.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE(stats.is_outlier(100.0));
    REQUIRE_FALSE(stats.is_outlier(3.0));
  }
}

// ============================================================================
// Min/Max Tests
// ============================================================================

TEST_CASE("Min/Max calculation", "[statistics]") {
  SECTION("Basic min/max") {
    std::vector<double> values = {5.0, 2.0, 8.0, 1.0, 9.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.min, WithinAbs(1.0, 1e-10));
    REQUIRE_THAT(stats.max, WithinAbs(9.0, 1e-10));
  }

  SECTION("Single value is both min and max") {
    std::vector<double> values = {42.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.min, WithinAbs(42.0, 1e-10));
    REQUIRE_THAT(stats.max, WithinAbs(42.0, 1e-10));
  }
}

// ============================================================================
// Confidence Interval Tests
// ============================================================================

TEST_CASE("Confidence interval calculation", "[statistics][ci]") {
  SECTION("95% CI formula") {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    auto stats = compute_enhanced_stats(values);
    // CI = 2 * stddev / sqrt(n)
    double expected_ci = 2.0 * stats.stddev / std::sqrt(5.0);
    REQUIRE_THAT(stats.confidence_margin_95(), WithinRel(expected_ci, 1e-6));
  }

  SECTION("Single value has zero CI") {
    std::vector<double> values = {42.0};
    auto stats = compute_enhanced_stats(values);
    REQUIRE_THAT(stats.confidence_margin_95(), WithinAbs(0.0, 1e-10));
  }
}
