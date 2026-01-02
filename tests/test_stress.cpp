/**
 * @file test_stress.cpp
 * @brief Stress tests for Fornax benchmark stability
 *
 * Tests:
 * - Long-running stability
 * - Memory pressure behavior
 * - High contention scenarios
 */

#include <catch2/catch_test_macros.hpp>

#include "arch.h"
#include "ring_buffer.h"
#include "shared_state.h"
#include "statistics.h"

#include <atomic>
#include <chrono>
#include <random>
#include <thread>
#include <vector>

using namespace fornax;

// ============================================================================
// Ring Buffer Stress Tests
// ============================================================================

TEST_CASE("RingBuffer stress test", "[stress][ring_buffer]") {
  constexpr size_t BUFFER_SIZE = 100;
  RingBuffer<double, BUFFER_SIZE> buffer;

  SECTION("Rapid push/overwrite cycles") {
    constexpr int NUM_PUSHES = 10000;

    for (int i = 0; i < NUM_PUSHES; ++i) {
      buffer.push_back(static_cast<double>(i));
    }

    // Buffer should contain last BUFFER_SIZE elements
    REQUIRE(buffer.size() == BUFFER_SIZE);
    REQUIRE(buffer.full());

    // Verify oldest element
    REQUIRE(buffer.front() == static_cast<double>(NUM_PUSHES - BUFFER_SIZE));

    // Verify newest element
    REQUIRE(buffer.back() == static_cast<double>(NUM_PUSHES - 1));
  }

  SECTION("Iteration after overwrites") {
    constexpr int NUM_PUSHES = 500;

    for (int i = 0; i < NUM_PUSHES; ++i) {
      buffer.push_back(static_cast<double>(i));
    }

    // Count elements via iteration
    size_t count = 0;
    for ([[maybe_unused]] const auto &elem : buffer) {
      ++count;
    }
    REQUIRE(count == BUFFER_SIZE);
  }
}

// ============================================================================
// Shared State Stress Tests
// ============================================================================

TEST_CASE("SharedState high-contention stress", "[stress][threading]") {
  SharedState state;
  state.shutdown.store(false);

  constexpr int NUM_WRITERS = 4;
  constexpr int NUM_READERS = 4;
  constexpr int OPS_PER_THREAD = 50000;

  std::atomic<uint64_t> total_writes{0};
  std::atomic<uint64_t> total_reads{0};

  SECTION("Multi-writer/multi-reader stress") {
    std::vector<std::thread> threads;

    // Spawn writer threads
    for (int w = 0; w < NUM_WRITERS; ++w) {
      threads.emplace_back([&, w]() {
        for (int i = 0; i < OPS_PER_THREAD; ++i) {
          state.throttle_signal.store((i + w) % 2 == 0,
                                      std::memory_order_relaxed);
          state.current_power_uw.store(static_cast<uint64_t>(i) * 1000 + w,
                                       std::memory_order_relaxed);
          total_writes.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    // Spawn reader threads
    for (int r = 0; r < NUM_READERS; ++r) {
      threads.emplace_back([&]() {
        for (int i = 0; i < OPS_PER_THREAD; ++i) {
          [[maybe_unused]] bool t =
              state.throttle_signal.load(std::memory_order_relaxed);
          [[maybe_unused]] uint64_t p =
              state.current_power_uw.load(std::memory_order_relaxed);
          total_reads.fetch_add(1, std::memory_order_relaxed);
        }
      });
    }

    for (auto &t : threads) {
      t.join();
    }

    REQUIRE(total_writes.load() == NUM_WRITERS * OPS_PER_THREAD);
    REQUIRE(total_reads.load() == NUM_READERS * OPS_PER_THREAD);
  }
}

// ============================================================================
// Statistics Stress Tests
// ============================================================================

TEST_CASE("Statistics with large datasets", "[stress][statistics]") {
  std::mt19937_64 rng(42);
  std::normal_distribution<double> dist(100000.0, 5000.0);

  SECTION("100k samples") {
    std::vector<double> samples;
    samples.reserve(100000);
    for (int i = 0; i < 100000; ++i) {
      samples.push_back(dist(rng));
    }

    auto stats = compute_enhanced_stats(samples);

    // Mean should be close to 100000
    REQUIRE(stats.mean > 99000.0);
    REQUIRE(stats.mean < 101000.0);

    // Stddev should be close to 5000
    REQUIRE(stats.stddev > 4500.0);
    REQUIRE(stats.stddev < 5500.0);

    // Sample count correct
    REQUIRE(stats.n == 100000);
  }
}

// ============================================================================
// Timing Consistency Tests
// ============================================================================

TEST_CASE("cpu_relax timing stability", "[stress][timing]") {
  SECTION("cpu_relax does not hang") {
    auto start = std::chrono::steady_clock::now();

    // Run many cpu_relax calls
    for (int i = 0; i < 100000; ++i) {
      cpu_relax();
    }

    auto end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Should complete in reasonable time (< 1 second for 100k relaxes)
    REQUIRE(elapsed.count() < 1000);
  }
}

// ============================================================================
// Architecture Consistency Tests
// ============================================================================

TEST_CASE("Architecture detection consistency", "[stress][arch]") {
  SECTION("get_cycles returns increasing values") {
    uint64_t c1 = get_cycles();

    // Do some work
    volatile double sum = 0.0;
    for (int i = 0; i < 10000; ++i) {
      sum += i * 0.001;
    }

    uint64_t c2 = get_cycles();

    REQUIRE(c2 > c1);
    (void)sum; // Suppress unused warning
  }

  SECTION("Architecture name is non-empty") {
    const char *arch = get_arch_name();
    REQUIRE(arch != nullptr);
    REQUIRE(strlen(arch) > 0);
  }

  SECTION("SIMD name is non-empty") {
    const char *simd = get_simd_name();
    REQUIRE(simd != nullptr);
    REQUIRE(strlen(simd) > 0);
  }
}
