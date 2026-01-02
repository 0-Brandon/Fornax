/**
 * @file worker.cpp
 * @brief SIMD workload generator thread (Core 1)
 *
 * The Worker thread generates maximum thermal/power load using SIMD
 * instructions to stress the CPU's voltage regulation and frequency scaling
 * mechanisms.
 *
 * Workload Strategy:
 * - x86: AVX-512 FMA (Fused Multiply-Add) operations
 *   - These trigger the maximum AVX offset (frequency reduction)
 *   - Highest power draw per instruction
 * - ARM: NEON FMA operations
 *   - Similar high-throughput vector math
 *   - Useful for validating logic on M1/M2 development machines
 *
 * Braking Mechanism:
 * When the Monitor signals to throttle, we enter a busy-wait loop using
 * cpu_relax() instead of sleeping. This provides:
 * - Immediate response to unthrottle signal (no scheduler wake latency)
 * - Reduced dynamic power (C_dyn) without entering OS sleep states
 * - Avoidance of the 5-10µs wake-up latency from sleep_for()
 */

#include "arch.h"
#include "config.h"
#include "shared_state.h"
#include "thread_utils.h"
#include "workloads.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>

// Platform-specific headers for thread affinity are now in thread_utils.h

namespace fornax {

// Configuration imported from config.h
// WorkerConfig is now defined in include/config.h

// ============================================================================
// SIMD Workloads
// ============================================================================

#if FORNAX_ARCH_X86

/**
 * @brief Run heavy AVX-512 FMA workload (x86)
 *
 * This function executes the most power-hungry x86 instructions:
 * - AVX-512 Fused Multiply-Add (FMA) on 8 double-precision floats
 * - Each FMA: result = a * b + c
 * - Peak throughput: 2 FMA/cycle on Skylake-X (32 DP FLOPS/cycle per core)
 *
 * Why FMA for power testing:
 * - Uses the widest execution units (512-bit)
 * - Triggers maximum current draw
 * - Causes the deepest AVX offset (frequency reduction)
 *
 * @param iterations Number of FMA operation blocks to execute
 * @return Dummy value to prevent optimization
 */
#if defined(__AVX512F__)
double run_heavy_math_x86(int iterations) {
  // Initialize vectors with non-zero values
  __m512d a = _mm512_set1_pd(1.0000001);
  __m512d b = _mm512_set1_pd(1.0000002);
  __m512d c = _mm512_set1_pd(0.0);

  for (int i = 0; i < iterations; ++i) {
    // Fused Multiply-Add: c = a * b + c
    // This is the highest power instruction on modern x86
    c = _mm512_fmadd_pd(a, b, c);
    c = _mm512_fmadd_pd(a, b, c);
    c = _mm512_fmadd_pd(a, b, c);
    c = _mm512_fmadd_pd(a, b, c);

    // Prevent compiler from optimizing away
    // By using the result in a dependency chain
    a = _mm512_fmadd_pd(c, b, a);
  }

  // Extract one element to return (prevents dead code elimination)
  double result[8];
  _mm512_storeu_pd(result, c);
  return result[0];
}
#elif defined(__AVX2__)
double run_heavy_math_x86(int iterations) {
  // Fallback to AVX2 (256-bit) if AVX-512 not available
  __m256d a = _mm256_set1_pd(1.0000001);
  __m256d b = _mm256_set1_pd(1.0000002);
  __m256d c = _mm256_set1_pd(0.0);

  for (int i = 0; i < iterations; ++i) {
    c = _mm256_fmadd_pd(a, b, c);
    c = _mm256_fmadd_pd(a, b, c);
    c = _mm256_fmadd_pd(a, b, c);
    c = _mm256_fmadd_pd(a, b, c);
    a = _mm256_fmadd_pd(c, b, a);
  }

  double result[4];
  _mm256_storeu_pd(result, c);
  return result[0];
}
#else
double run_heavy_math_x86(int iterations) {
  // Minimal fallback for SSE-only systems
  __m128d a = _mm_set1_pd(1.0000001);
  __m128d b = _mm_set1_pd(1.0000002);
  __m128d c = _mm_set1_pd(0.0);

  for (int i = 0; i < iterations; ++i) {
    c = _mm_add_pd(_mm_mul_pd(a, b), c);
    c = _mm_add_pd(_mm_mul_pd(a, b), c);
    a = _mm_add_pd(_mm_mul_pd(c, b), a);
  }

  double result[2];
  _mm_storeu_pd(result, c);
  return result[0];
}
#endif

#elif FORNAX_ARCH_ARM

/**
 * @brief Run heavy NEON FMA workload (ARM)
 *
 * This function executes high-throughput NEON SIMD instructions:
 * - NEON Fused Multiply-Add on 2 double-precision floats (float64x2_t)
 * - Peak throughput varies by core (M1 has 4 NEON units)
 *
 * Note: ARM cores typically don't have the same "AVX offset" phenomenon,
 * but this provides a consistent workload for testing the control logic.
 *
 * @param iterations Number of FMA operation blocks to execute
 * @return Dummy value to prevent optimization
 */
double run_heavy_math_arm(int iterations) {
  // NEON double-precision vectors (2 x float64)
  float64x2_t a = vdupq_n_f64(1.0000001);
  float64x2_t b = vdupq_n_f64(1.0000002);
  float64x2_t c = vdupq_n_f64(0.0);

  for (int i = 0; i < iterations; ++i) {
    // vfmaq_f64: Fused Multiply-Add
    // c = a * b + c
    c = vfmaq_f64(c, a, b);
    c = vfmaq_f64(c, a, b);
    c = vfmaq_f64(c, a, b);
    c = vfmaq_f64(c, a, b);

    // Create dependency to prevent optimization
    a = vfmaq_f64(a, c, b);
  }

  // Extract first element to return
  return vgetq_lane_f64(c, 0);
}

#endif

/**
 * @brief Platform-independent wrapper for heavy math workload
 */
inline double run_heavy_math(int iterations) {
#if FORNAX_ARCH_X86
  return run_heavy_math_x86(iterations);
#elif FORNAX_ARCH_ARM
  return run_heavy_math_arm(iterations);
#endif
}

/**
 * @brief Dispatch to appropriate workload based on config
 */
inline double run_workload(WorkloadType type, int iterations) {
  switch (type) {
  case WorkloadType::FMA_STRESS:
    return run_heavy_math(iterations);
  case WorkloadType::BLACK_SCHOLES:
    return workloads::run_black_scholes(iterations);
  case WorkloadType::MONTE_CARLO:
    return workloads::run_monte_carlo(iterations / 10); // MC is heavier
  case WorkloadType::COVARIANCE:
    return workloads::run_covariance(iterations);
  case WorkloadType::MIXED:
    return workloads::run_mixed_workload(iterations / 5);
  }
  return run_heavy_math(iterations); // fallback
}

// ============================================================================
// Worker Thread Entry Point
// ============================================================================

/**
 * @brief Worker thread main function
 *
 * Executes a tight loop of:
 * 1. Check throttle signal
 * 2. If throttled: busy-wait with cpu_relax()
 * 3. If not throttled: run heavy SIMD math
 * 4. Update iteration counter
 */
template <typename StateType>
void worker_thread(StateType &state, const WorkerConfig &config) {
  // Pin to designated core
  if (!pin_to_core(config.target_core, "Worker")) {
    std::cerr << "[Worker] Continuing without CPU affinity" << std::endl;
  }

  std::cout << "[Worker] Started on core " << config.target_core << " using "
            << get_simd_name()
            << " SIMD, workload: " << workload_type_to_string(config.workload)
            << std::endl;

  // Dummy accumulator to prevent dead code elimination
  volatile double sink = 0.0;

  while (!state.shutdown.load(std::memory_order_relaxed)) {
    // Check throttle signal with relaxed ordering.
    // Relaxed is sufficient as we only care about eventual visibility.
    // On x86, relaxed loads are effectively free.
    if (state.throttle_signal.load(std::memory_order_relaxed)) {
      // ================================================================
      // THROTTLED STATE: Enter low-power busy-wait
      // ================================================================
      //
      // Using yield/pause to reduce dynamic power (C_dyn = αCV²f) without
      // yielding to the OS scheduler, avoiding the 5-10µs wake-up latency
      // of sleep_for(). This is critical for low-latency applications where
      // we need to resume work immediately when unthrottled.
      //
      // The PAUSE/YIELD instruction:
      // - x86: Delays ~10-100 cycles, signals spin-wait to CPU
      // - ARM: Hints that thread is waiting, may reduce power
      // ================================================================
      cpu_relax();
    } else {
      // ================================================================
      // ACTIVE STATE: Run SIMD workload (selected by config)
      // ================================================================
      sink += run_workload(config.workload, config.ops_per_iteration);

      // Increment iteration counter (relaxed - no ordering needed)
      state.iteration_count.fetch_add(1, std::memory_order_relaxed);
    }
  }

  // Use sink to prevent optimization (never actually printed in practice)
  if (sink == 12345.6789) {
    std::cout << "[Worker] Sink: " << sink << std::endl;
  }

  std::cout << "[Worker] Shutting down" << std::endl;
}

// Explicit template instantiations
template void worker_thread<SharedState>(SharedState &, const WorkerConfig &);
template void worker_thread<SharedStateNoPadding>(SharedStateNoPadding &,
                                                  const WorkerConfig &);

} // namespace fornax
