/**
 * @file config.h
 * @brief Centralized configuration for Fornax SIMD benchmark
 *
 * Consolidates all configuration structs and named constants to eliminate
 * duplication across translation units and improve maintainability.
 */

#ifndef FORNAX_CONFIG_H
#define FORNAX_CONFIG_H

#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace fornax {

// ============================================================================
// Named Constants (replacing magic numbers)
// ============================================================================

namespace defaults {

// Power thresholds for Schmitt trigger (in microwatts)
constexpr uint64_t POWER_HIGH_THRESHOLD_UW = 100'000'000; // 100W - throttle ON
constexpr uint64_t POWER_LOW_THRESHOLD_UW = 70'000'000;   // 70W - throttle OFF

// Sensor polling configuration
constexpr auto POLL_INTERVAL =
    std::chrono::microseconds{1000}; // 1kHz (RAPL update rate)
constexpr auto REPORT_INTERVAL = std::chrono::seconds{1};

// Worker configuration
constexpr int OPS_PER_ITERATION = 1000;
constexpr int MONITOR_CORE = 0;
constexpr int WORKER_CORE = 1;

// Simulation (for ARM/testing)
constexpr uint64_t SIM_IDLE_POWER_UW = 30'000'000;    // 30W idle
constexpr uint64_t SIM_ACTIVE_POWER_UW = 125'000'000; // 125W active

// Duty cycle control
constexpr auto DUTY_CYCLE_PERIOD = std::chrono::milliseconds{10};

// Statistical defaults
constexpr int DEFAULT_TRIALS = 10;
constexpr int DEFAULT_WARMUP_SECONDS = 2;
constexpr int DEFAULT_DURATION_SECONDS = 10;

// Conversion factors (to avoid magic numbers)
constexpr uint64_t MICROWATTS_PER_WATT = 1'000'000;
constexpr uint64_t KILOHERTZ_PER_MEGAHERTZ = 1000;

} // namespace defaults

// ============================================================================
// Workload Types
// ============================================================================

enum class WorkloadType {
  FMA_STRESS,    // Current synthetic FMA loop
  BLACK_SCHOLES, // Options pricing kernel
  MONTE_CARLO,   // Random path simulation
  COVARIANCE,    // Matrix covariance operations
  MIXED          // Interleaved workloads
};

inline const char *workload_type_to_string(WorkloadType type) {
  switch (type) {
  case WorkloadType::FMA_STRESS:
    return "fma-stress";
  case WorkloadType::BLACK_SCHOLES:
    return "black-scholes";
  case WorkloadType::MONTE_CARLO:
    return "monte-carlo";
  case WorkloadType::COVARIANCE:
    return "covariance";
  case WorkloadType::MIXED:
    return "mixed";
  }
  return "unknown";
}

inline WorkloadType string_to_workload_type(const std::string &str) {
  if (str == "fma-stress")
    return WorkloadType::FMA_STRESS;
  if (str == "black-scholes")
    return WorkloadType::BLACK_SCHOLES;
  if (str == "monte-carlo")
    return WorkloadType::MONTE_CARLO;
  if (str == "covariance")
    return WorkloadType::COVARIANCE;
  if (str == "mixed")
    return WorkloadType::MIXED;
  return WorkloadType::FMA_STRESS; // default
}

// ============================================================================
// Configuration Structs
// ============================================================================

/**
 * @brief Monitor thread configuration
 */
struct MonitorConfig {
  // Schmitt trigger thresholds (in microwatts)
  uint64_t power_high_threshold_uw = defaults::POWER_HIGH_THRESHOLD_UW;
  uint64_t power_low_threshold_uw = defaults::POWER_LOW_THRESHOLD_UW;

  // Sensor polling interval
  std::chrono::microseconds poll_interval{defaults::POLL_INTERVAL};

  // Reporting interval
  std::chrono::seconds report_interval{defaults::REPORT_INTERVAL};

  // Simulation mode (for ARM or when sensors unavailable)
  bool simulate_power = false;

  // Manual duty cycle override (0.0 = never throttle, 1.0 = always throttle)
  // Negative value means use hysteresis controller
  double manual_duty_cycle = -1.0;

  // Target core for affinity
  int target_core = defaults::MONITOR_CORE;

  // Adaptive control mode
  bool adaptive_control = false;
  double target_throughput = 0.0; // Target iterations/sec for adaptive mode

  // Output CSV file for time-series data
  std::string output_file;
};

/**
 * @brief Worker thread configuration
 */
struct WorkerConfig {
  // Target core for affinity
  int target_core = defaults::WORKER_CORE;

  // Number of operations per iteration
  int ops_per_iteration = defaults::OPS_PER_ITERATION;

  // Workload type
  WorkloadType workload = WorkloadType::FMA_STRESS;
};

/**
 * @brief Sweep experiment configuration
 */
struct SweepConfig {
  // Duty cycles to test
  std::vector<double> duty_cycles = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.6, 0.7, 0.8, 0.9, 1.0};

  // Number of trials per configuration
  int trials_per_config = defaults::DEFAULT_TRIALS;

  // Warm-up duration (discarded from results)
  std::chrono::seconds warmup_duration{defaults::DEFAULT_WARMUP_SECONDS};

  // Measurement duration per trial
  std::chrono::seconds measurement_duration{defaults::DEFAULT_DURATION_SECONDS};

  // Output CSV file path (empty = stdout)
  std::string output_file;
};

} // namespace fornax

#endif // FORNAX_CONFIG_H
