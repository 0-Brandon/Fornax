/**
 * @file shared_state.h
 * @brief Cache-line aligned shared state for inter-thread communication
 *
 * Hardware Sympathy: False Sharing Prevention
 * ============================================
 *
 * Modern CPUs maintain cache coherency using the MESI protocol:
 * - Modified: Line is dirty, owned exclusively
 * - Exclusive: Line is clean, owned exclusively
 * - Shared: Line is clean, may be in other caches
 * - Invalid: Line is not valid
 *
 * THE PROBLEM (False Sharing):
 * When two cores access different variables that happen to reside on the
 * SAME cache line (typically 64 bytes), MESI causes unnecessary invalidations:
 *
 *   Core 0 (Monitor)         Core 1 (Worker)
 *        |                        |
 *        v                        v
 *   +---------+              +---------+
 *   | L1 Cache|              | L1 Cache|
 *   | [X][Y]  |  <-- MESI -> | [X][Y]  |
 *   +---------+  invalidate  +---------+
 *        |                        |
 *        +---- Shared L3 Cache ---+
 *
 * If Monitor writes X and Worker reads Y, and both are on the same cache line,
 * each write to X invalidates Worker's copy, forcing a cache miss (~40-80
 * cycles).
 *
 * THE SOLUTION:
 * Use alignas(64) and explicit padding to ensure each frequently-accessed
 * variable occupies its own cache line. This eliminates false sharing at the
 * cost of slightly higher memory usage.
 *
 * References:
 * - Intel Optimization Manual, Section 8.4: "Avoiding False Sharing"
 * - Ulrich Drepper, "What Every Programmer Should Know About Memory"
 */

#ifndef FORNAX_SHARED_STATE_H
#define FORNAX_SHARED_STATE_H

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fornax {

// Cache line size for modern x86 and ARM processors
constexpr std::size_t CACHE_LINE_SIZE = 64;

/**
 * @brief Shared state between Monitor and Worker threads
 *
 * Layout (with padding enabled):
 *
 *   Offset 0-63 (Cache Line 1):
 *   +-----------+------------------+
 *   | throttle  |     padding      |  <- Monitor writes, Worker reads
 *   | (1 byte)  |    (63 bytes)    |
 *   +-----------+------------------+
 *
 *   Offset 64-127 (Cache Line 2):
 *   +-----------+------------------+
 *   | iteration |     padding      |  <- Worker writes, Main reads
 *   | (8 bytes) |    (56 bytes)    |
 *   +-----------+------------------+
 *
 * This layout ensures:
 * 1. Monitor's writes to throttle_signal don't invalidate Worker's reads
 * 2. Worker's writes to iteration_count don't affect other readers
 * 3. No false sharing occurs between any pair of cores
 */
struct alignas(CACHE_LINE_SIZE) SharedState {
  // ========================================================================
  // Cache Line 1: Throttle Signal (Monitor -> Worker)
  // ========================================================================

  /**
   * @brief Throttle signal from Monitor to Worker
   *
   * When true, the Worker should enter a low-power busy-wait using cpu_relax()
   * to reduce dynamic power consumption without yielding to the scheduler.
   *
   * Memory ordering: relaxed
   * - No synchronization with other memory operations needed
   * - We only care about eventual visibility, not ordering
   * - Latency: typically 1-2 cache coherency round-trips (~20-40ns)
   */
  std::atomic<bool> throttle_signal{false};

  // Padding to fill rest of cache line 1
  // sizeof(atomic<bool>) is typically 1 byte on most platforms
  char padding1[CACHE_LINE_SIZE - sizeof(std::atomic<bool>)];

  // ========================================================================
  // Cache Line 2: Iteration Counter (Worker -> Main)
  // ========================================================================

  /**
   * @brief Iteration counter for throughput measurement
   *
   * Incremented by the Worker after each SIMD computation batch.
   * Read by Main thread to calculate iterations per second.
   */
  std::atomic<uint64_t> iteration_count{0};

  // Padding to fill rest of cache line 2
  char padding2[CACHE_LINE_SIZE - sizeof(std::atomic<uint64_t>)];

  // ========================================================================
  // Cache Line 3: Control Flags (Main -> Worker/Monitor)
  // ========================================================================

  /**
   * @brief Shutdown signal to stop all threads
   */
  std::atomic<bool> shutdown{false};

  // Padding to fill rest of cache line 3
  char padding3[CACHE_LINE_SIZE - sizeof(std::atomic<bool>)];

  // ========================================================================
  // Cache Line 4: Statistics (Monitor -> Main, read-only by others)
  // ========================================================================

  /**
   * @brief Current power reading in microwatts
   */
  std::atomic<uint64_t> current_power_uw{0};

  /**
   * @brief Current CPU frequency in kHz
   */
  std::atomic<uint64_t> current_freq_khz{0};

  // Padding to fill rest of cache line 4
  char padding4[CACHE_LINE_SIZE - sizeof(std::atomic<uint64_t>) * 2];
};

// ============================================================================
// Compile-Time Verification
// ============================================================================

// Verify that our padding strategy is working
// The struct should span at least 4 cache lines (256 bytes) with current layout
static_assert(
    sizeof(SharedState) >= 128,
    "Cache padding failed: False sharing risk detected. "
    "SharedState must span at least 2 cache lines to prevent MESI invalidation "
    "between Monitor and Worker threads.");

// Verify alignment
static_assert(alignof(SharedState) == CACHE_LINE_SIZE,
              "SharedState must be aligned to cache line boundary");

// Verify individual field sizes for padding calculation
static_assert(sizeof(std::atomic<bool>) <= CACHE_LINE_SIZE,
              "atomic<bool> larger than cache line - adjust padding");
static_assert(sizeof(std::atomic<uint64_t>) <= CACHE_LINE_SIZE,
              "atomic<uint64_t> larger than cache line - adjust padding");

/**
 * @brief SharedState variant without padding for demonstrating false sharing
 *
 * Use --no-padding CLI flag to use this variant and observe the performance
 * degradation caused by cache line contention.
 */
struct SharedStateNoPadding {
  std::atomic<bool> throttle_signal{false};
  std::atomic<uint64_t> iteration_count{0};
  std::atomic<bool> shutdown{false};
  std::atomic<uint64_t> current_power_uw{0};
  std::atomic<uint64_t> current_freq_khz{0};
};

// This should be significantly smaller than the padded version
static_assert(sizeof(SharedStateNoPadding) < sizeof(SharedState),
              "NoPadding variant should be smaller than padded version");

} // namespace fornax

#endif // FORNAX_SHARED_STATE_H
