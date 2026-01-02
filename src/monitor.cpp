/**
 * @file monitor.cpp
 * @brief Power and frequency monitoring thread (Core 0)
 *
 * The Monitor thread implements a feedback control loop that reads system
 * sensors and signals the Worker thread to throttle when power exceeds
 * thresholds. This allows empirical measurement of the trade-off between
 * SIMD throughput and CPU frequency/power.
 *
 * Sensor Sources:
 * - Power: /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj (x86)
 * - Frequency: /sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq
 *
 * Control Strategy: Schmitt Trigger (Hysteresis)
 * - Prevents oscillation by using different thresholds for on/off
 * - Throttle ON when power > HIGH_THRESHOLD
 * - Throttle OFF when power < LOW_THRESHOLD
 */

#include "arch.h"
#include "config.h"
#include "controller.h"
#include "shared_state.h"
#include "thread_utils.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

// Platform-specific headers for thread affinity are now in thread_utils.h

namespace fornax {

// Configuration imported from config.h
// MonitorConfig is now defined in include/config.h

// ============================================================================
// Sensor Readers
// ============================================================================

/**
 * @brief Read energy counter from Intel RAPL (Running Average Power Limit)
 *
 * RAPL provides energy counters in microjoules. To get power, we compute:
 *   P = ΔE / Δt
 *
 * @param path Path to energy_uj file
 * @param value Output: energy in microjoules
 * @return true if read successful
 */
bool read_energy_uj(const std::string &path, uint64_t &value) {
  std::ifstream file(path);
  if (!file.is_open()) {
    return false;
  }
  file >> value;
  return !file.fail();
}

/**
 * @brief Read CPU frequency from sysfs
 *
 * @param cpu_id CPU core number
 * @return Frequency in kHz, or std::nullopt if read failed
 */
[[nodiscard]] std::optional<uint64_t> read_cpu_freq_khz(int cpu_id) {
  std::ostringstream path;
  path << "/sys/devices/system/cpu/cpu" << cpu_id
       << "/cpufreq/scaling_cur_freq";

  std::ifstream file(path.str());
  if (!file.is_open()) {
    return std::nullopt;
  }
  uint64_t freq_khz = 0;
  file >> freq_khz;
  if (file.fail()) {
    return std::nullopt;
  }
  return freq_khz;
}

/**
 * @brief Find the RAPL energy file path
 *
 * Intel RAPL exposes energy counters at various paths depending on the system:
 * - /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj (package energy)
 * - /sys/class/powercap/intel-rapl:0/energy_uj (alternative path)
 *
 * @return Path to energy file, or empty string if not found
 */
std::string find_rapl_energy_path() {
  const char *candidates[] = {
      "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
      "/sys/class/powercap/intel-rapl:0/energy_uj",
      "/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj", // Core domain
  };

  for (const char *path : candidates) {
    std::ifstream file(path);
    if (file.is_open()) {
      return path;
    }
  }
  return "";
}

// ============================================================================
// Thread Affinity - using shared implementation from thread_utils.h
// ============================================================================

// ============================================================================
// Power Simulator (for ARM/testing)
// ============================================================================

/**
 * @brief Simulate power based on worker activity
 *
 * When real sensors aren't available (ARM, containers, etc.), this provides
 * a reasonable approximation for testing the control logic.
 *
 * Model: P = P_idle + P_active * (1 - throttle_ratio)
 */
class PowerSimulator {
public:
  PowerSimulator(uint64_t idle_power_uw = defaults::SIM_IDLE_POWER_UW,
                 uint64_t active_power_uw = defaults::SIM_ACTIVE_POWER_UW)
      : idle_power_(idle_power_uw), active_power_(active_power_uw) {}

  uint64_t get_simulated_power(bool is_throttled) const {
    if (is_throttled) {
      // When throttled, power drops toward idle
      return idle_power_ + (active_power_ - idle_power_) / 10;
    } else {
      // When active, full power plus some noise
      return active_power_;
    }
  }

private:
  uint64_t idle_power_;
  uint64_t active_power_;
};

// ============================================================================
// Monitor Thread Entry Point
// ============================================================================

/**
 * @brief Monitor thread main function
 *
 * Implements the feedback control loop:
 * 1. Read power sensor (or simulate)
 * 2. Read frequency sensor
 * 3. Apply Schmitt trigger to set throttle signal
 * 4. Report statistics periodically
 */
template <typename StateType>
void monitor_thread(StateType &state, const MonitorConfig &config) {
  // Pin to designated core
  if (!pin_to_core(config.target_core, "Monitor")) {
    std::cerr << "[Monitor] Continuing without CPU affinity" << std::endl;
  }

  std::cout << "[Monitor] Started on core " << config.target_core << std::endl;

  // Find RAPL path or enable simulation
  std::string rapl_path = find_rapl_energy_path();
  bool use_simulation = config.simulate_power || rapl_path.empty();

  if (use_simulation) {
    std::cout << "[Monitor] Simulation mode enabled (RAPL not available)"
              << std::endl;
  } else {
    std::cout << "[Monitor] Using RAPL sensor: " << rapl_path << std::endl;
  }

  PowerSimulator simulator;

  // State for power calculation (P = ΔE / Δt)
  uint64_t last_energy_uj = 0;
  auto last_energy_time = std::chrono::steady_clock::now();

  // Initialize energy reading
  if (!use_simulation) {
    read_energy_uj(rapl_path, last_energy_uj);
  }

  // Statistics accumulators
  uint64_t power_sum = 0;
  uint64_t freq_sum = 0;
  uint64_t sample_count = 0;
  auto last_report_time = std::chrono::steady_clock::now();

  // Duty cycle state (for manual mode)
  auto duty_cycle_start = std::chrono::steady_clock::now();
  const auto duty_cycle_period = std::chrono::milliseconds(10);

  // Current throttle state
  bool is_throttled = false;
  double current_duty_cycle = 0.5; // For adaptive mode

  // Adaptive controller (used if config.adaptive_control is true)
  AdaptiveController adaptive_controller(0.5, 0.05, 0.7);
  uint64_t prev_iterations = 0;
  auto prev_iter_time = std::chrono::steady_clock::now();

  if (config.adaptive_control) {
    std::cout << "[Monitor] Adaptive control mode enabled" << std::endl;
  }

  // Open output CSV if requested
  std::ofstream csv_file;
  if (!config.output_file.empty()) {
    csv_file.open(config.output_file);
    if (csv_file.is_open()) {
      csv_file << "timestamp_s,frequency_mhz,power_w,throttled\n";
      std::cout << "[Monitor] Logging high-res data to: " << config.output_file
                << std::endl;
    } else {
      std::cerr << "[Monitor] Failed to open output file: "
                << config.output_file << std::endl;
    }
  }

  auto start_time = std::chrono::steady_clock::now();

  while (!state.shutdown.load(std::memory_order_relaxed)) {
    auto now = std::chrono::steady_clock::now();

    // --------------------------------------------------------------------
    // Read Sensors
    // --------------------------------------------------------------------

    uint64_t current_power_uw = 0;

    if (use_simulation) {
      current_power_uw = simulator.get_simulated_power(is_throttled);
    } else {
      // Calculate power from energy delta
      uint64_t current_energy_uj;
      if (read_energy_uj(rapl_path, current_energy_uj)) {
        auto dt = std::chrono::duration_cast<std::chrono::microseconds>(
                      now - last_energy_time)
                      .count();

        if (dt > 0 && current_energy_uj > last_energy_uj) {
          // P = ΔE / Δt (convert to microwatts)
          current_power_uw =
              ((current_energy_uj - last_energy_uj) * 1'000'000) / dt;
        }

        last_energy_uj = current_energy_uj;
        last_energy_time = now;
      }
    }

    // Read CPU frequency (for the worker core)
    // Use value_or(0) for graceful fallback when sysfs unavailable
    uint64_t current_freq_khz = read_cpu_freq_khz(1).value_or(0);

    // Update shared state for main thread to read
    state.current_power_uw.store(current_power_uw, std::memory_order_relaxed);
    state.current_freq_khz.store(current_freq_khz, std::memory_order_relaxed);

    // Accumulate for averaging
    power_sum += current_power_uw;
    freq_sum += current_freq_khz;
    sample_count++;

    // --------------------------------------------------------------------
    // Control Logic
    // --------------------------------------------------------------------

    if (config.adaptive_control) {
      // Adaptive control mode: use gradient-based controller
      // Estimate current throughput from iteration count
      uint64_t current_iterations =
          state.iteration_count.load(std::memory_order_relaxed);
      auto iter_now = std::chrono::steady_clock::now();
      double dt_sec =
          std::chrono::duration<double>(iter_now - prev_iter_time).count();

      if (dt_sec > 0.1) { // Update controller every 100ms
        double throughput =
            static_cast<double>(current_iterations - prev_iterations) / dt_sec;

        double power_w = static_cast<double>(current_power_uw) / 1'000'000.0;
        double freq_mhz = static_cast<double>(current_freq_khz) / 1000.0;

        current_duty_cycle =
            adaptive_controller.update(throughput, power_w, freq_mhz);

        prev_iterations = current_iterations;
        prev_iter_time = iter_now;
      }

      // Apply current duty cycle
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          now - duty_cycle_start);

      if (elapsed >= duty_cycle_period) {
        duty_cycle_start = now;
        elapsed = std::chrono::microseconds(0);
      }

      double phase = static_cast<double>(elapsed.count()) /
                     static_cast<double>(duty_cycle_period.count());
      is_throttled = (phase < current_duty_cycle);
    } else if (config.manual_duty_cycle >= 0.0) {
      // Manual duty cycle mode: throttle for (duty_cycle * period) then
      // unthrottle
      auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
          now - duty_cycle_start);

      if (elapsed >= duty_cycle_period) {
        duty_cycle_start = now;
        elapsed = std::chrono::microseconds(0);
      }

      double phase = static_cast<double>(elapsed.count()) /
                     static_cast<double>(duty_cycle_period.count());
      is_throttled = (phase < config.manual_duty_cycle);
    } else {
      // Schmitt trigger (hysteresis controller)
      // This prevents oscillation by using different thresholds
      if (current_power_uw > config.power_high_threshold_uw) {
        is_throttled = true;
      } else if (current_power_uw < config.power_low_threshold_uw) {
        is_throttled = false;
      }
      // If between thresholds, maintain current state (hysteresis band)
    }

    // Update throttle signal for worker thread
    state.throttle_signal.store(is_throttled, std::memory_order_relaxed);

    // --------------------------------------------------------------------
    // Periodic Reporting
    // --------------------------------------------------------------------

    auto time_since_report = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_report_time);

    if (time_since_report >= config.report_interval && sample_count > 0) {
      double avg_power_w = static_cast<double>(power_sum) /
                           static_cast<double>(sample_count) / 1'000'000.0;
      double avg_freq_mhz = static_cast<double>(freq_sum) /
                            static_cast<double>(sample_count) / 1000.0;

      std::cout << "[Monitor] Avg Power: " << avg_power_w << " W, "
                << "Avg Freq: " << avg_freq_mhz << " MHz, "
                << "Throttled: " << (is_throttled ? "YES" : "NO");

      if (config.adaptive_control) {
        std::cout << ", Duty: " << std::fixed << std::setprecision(1)
                  << (current_duty_cycle * 100) << "%"
                  << (adaptive_controller.is_exploring() ? " (exploring)" : "");
      }
      std::cout << std::endl;

      // Reset accumulators
      power_sum = 0;
      freq_sum = 0;
      sample_count = 0;
      last_report_time = now;
    }

    // Write sample to CSV if enabled
    if (csv_file.is_open()) {
      double t_sec = std::chrono::duration<double>(now - start_time).count();
      double freq_mhz = static_cast<double>(current_freq_khz) / 1000.0;
      double power_w = static_cast<double>(current_power_uw) / 1'000'000.0;

      csv_file << std::fixed << std::setprecision(4) << t_sec << ","
               << std::setprecision(2) << freq_mhz << ","
               << std::setprecision(2) << power_w << ","
               << (is_throttled ? "1" : "0") << "\n";
    }

    // Sleep until next poll
    std::this_thread::sleep_for(config.poll_interval);
  }

  std::cout << "[Monitor] Shutting down" << std::endl;
}

// Explicit template instantiations
template void monitor_thread<SharedState>(SharedState &, const MonitorConfig &);
template void monitor_thread<SharedStateNoPadding>(SharedStateNoPadding &,
                                                   const MonitorConfig &);

} // namespace fornax
