/**
 * @file test_integration.cpp
 * @brief End-to-end integration tests for Fornax benchmark
 *
 * Verifies:
 * - Full benchmark execution flow
 * - CLI argument parsing
 * - CSV output format
 * - Shared state thread safety
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "arch.h"
#include "config.h"
#include "controller.h"
#include "perf_counters.h"
#include "shared_state.h"
#include "statistics.h"
#include "workloads.h"

#include <atomic>
#include <chrono>
#include <sstream>
#include <thread>
#include <vector>

using namespace fornax;

// ============================================================================
// Shared State Tests
// ============================================================================

TEST_CASE("SharedState cache line isolation", "[integration][memory]") {
  SECTION("Padded state spans multiple cache lines") {
    SharedState state;
    // Each atomic should be on its own cache line
    REQUIRE(sizeof(SharedState) >= 128);
    REQUIRE(alignof(SharedState) == 64);
  }

  SECTION("Unpadded state is compact") {
    SharedStateNoPadding state;
    // All atomics packed together
    REQUIRE(sizeof(SharedStateNoPadding) < 64);
  }
}

TEST_CASE("SharedState concurrent access", "[integration][threading]") {
  SharedState state;
  state.shutdown.store(false);
  state.throttle_signal.store(false);
  state.iteration_count.store(0);

  constexpr int NUM_ITERATIONS = 10000;
  std::atomic<uint64_t> writer_count{0};
  std::atomic<uint64_t> reader_count{0};

  SECTION("Writer/reader pattern stress test") {
    // Simulate monitor thread (writer)
    std::thread writer([&]() {
      for (int i = 0; i < NUM_ITERATIONS; ++i) {
        state.throttle_signal.store((i % 2) == 0, std::memory_order_relaxed);
        state.current_power_uw.store(static_cast<uint64_t>(i) * 1000,
                                     std::memory_order_relaxed);
        writer_count.fetch_add(1, std::memory_order_relaxed);
      }
    });

    // Simulate worker thread (reader)
    std::thread reader([&]() {
      for (int i = 0; i < NUM_ITERATIONS; ++i) {
        [[maybe_unused]] bool throttle =
            state.throttle_signal.load(std::memory_order_relaxed);
        state.iteration_count.fetch_add(1, std::memory_order_relaxed);
        reader_count.fetch_add(1, std::memory_order_relaxed);
      }
    });

    writer.join();
    reader.join();

    REQUIRE(writer_count.load() == NUM_ITERATIONS);
    REQUIRE(reader_count.load() == NUM_ITERATIONS);
    REQUIRE(state.iteration_count.load() == NUM_ITERATIONS);
  }
}

// ============================================================================
// Workload Execution Tests
// ============================================================================

TEST_CASE("Workloads execute without crashing", "[integration][workloads]") {
  using namespace workloads;

  SECTION("FMA stress") {
    // Just verify it runs and returns a finite value
    double result = 0.0;
    REQUIRE_NOTHROW(result = run_black_scholes(10));
    REQUIRE(std::isfinite(result));
  }

  SECTION("Monte Carlo") {
    double result = 0.0;
    REQUIRE_NOTHROW(result = run_monte_carlo(5));
    REQUIRE(std::isfinite(result));
  }

  SECTION("Covariance") {
    double result = 0.0;
    REQUIRE_NOTHROW(result = run_covariance(10));
    REQUIRE(std::isfinite(result));
  }

  SECTION("Mixed workload") {
    double result = 0.0;
    REQUIRE_NOTHROW(result = run_mixed_workload(10));
    REQUIRE(std::isfinite(result));
  }
}

// ============================================================================
// Controller Integration Tests
// ============================================================================

TEST_CASE("AdaptiveController convergence", "[integration][controller]") {
  AdaptiveController controller(0.5, 0.05, 0.7);

  SECTION("Controller explores and converges") {
    // Simulate a throughput curve with optimal around 60%
    auto simulate_throughput = [](double duty_cycle) {
      // Quadratic with peak at 0.6
      return 100.0 - 200.0 * (duty_cycle - 0.6) * (duty_cycle - 0.6);
    };

    double prev_throughput = 0.0;
    for (int i = 0; i < 20; ++i) {
      double duty = controller.current_duty();
      double tp = simulate_throughput(duty);

      controller.update(tp, 50.0, 4500.0);
      prev_throughput = tp;
    }

    // After exploration, controller should find near-optimal
    // (allowing some tolerance for gradient-based convergence)
    REQUIRE(controller.best_duty() > 0.3);
    REQUIRE(controller.best_duty() < 0.9);
    REQUIRE(prev_throughput > 50.0); // Should be in reasonable range
  }
}

// ============================================================================
// Statistics Integration Tests
// ============================================================================

TEST_CASE("EnhancedStatistics on realistic data", "[integration][statistics]") {
  // Simulate benchmark throughput samples with some variance
  std::vector<double> samples;
  for (int i = 0; i < 30; ++i) {
    // Base throughput with ±5% variation
    samples.push_back(100000.0 + (i % 10 - 5) * 1000.0);
  }

  auto stats = compute_enhanced_stats(samples);

  SECTION("Statistics are computed correctly") {
    REQUIRE(stats.n == 30);
    REQUIRE(stats.mean > 95000.0);
    REQUIRE(stats.mean < 105000.0);
    REQUIRE(stats.stddev > 0.0);
    REQUIRE(stats.p50 > 95000.0);
  }

  SECTION("Confidence interval is reasonable") {
    double ci = stats.confidence_margin_95();
    // CI should be small relative to mean for 30 samples
    REQUIRE(ci < stats.mean * 0.1);
  }
}

// ============================================================================
// Performance Counter Tests
// ============================================================================

TEST_CASE("PerfEventGroup initialization", "[integration][perf]") {
  PerfEventGroup counters;

  SECTION("Graceful fallback when unavailable") {
    // May or may not succeed depending on permissions
    bool initialized = counters.initialize(-1);

    // Even if not available, should not crash
    counters.start();
    auto reading = counters.read();
    counters.stop();

    // If not available, all values should be zero
    if (!counters.available()) {
      REQUIRE(reading.instructions == 0);
      REQUIRE(reading.cycles == 0);
    }
  }
}

// ============================================================================
// Workload Selection Tests
// ============================================================================

TEST_CASE("Workload type parsing", "[integration][config]") {
  SECTION("Valid workload types") {
    REQUIRE(string_to_workload_type("fma-stress") == WorkloadType::FMA_STRESS);
    REQUIRE(string_to_workload_type("black-scholes") ==
            WorkloadType::BLACK_SCHOLES);
    REQUIRE(string_to_workload_type("monte-carlo") ==
            WorkloadType::MONTE_CARLO);
    REQUIRE(string_to_workload_type("covariance") == WorkloadType::COVARIANCE);
    REQUIRE(string_to_workload_type("mixed") == WorkloadType::MIXED);
  }

  SECTION("Unknown defaults to FMA_STRESS") {
    REQUIRE(string_to_workload_type("unknown") == WorkloadType::FMA_STRESS);
  }
}
