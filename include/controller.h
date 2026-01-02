/**
 * @file controller.h
 * @brief Adaptive duty cycle controller for Fornax SIMD benchmark
 *
 * Implements gradient-based duty cycle adaptation to maximize throughput
 * while respecting power and thermal constraints. This goes beyond the
 * simple Schmitt trigger (binary on/off) to continuously adjust the
 * work/pause ratio.
 *
 * Control Theory Background:
 * - The system is a SISO (Single Input Single Output) controller
 * - Input: Current duty cycle (0.0 to 1.0)
 * - Output: Measured throughput (iterations/sec)
 * - Goal: Find the duty cycle that maximizes throughput
 *
 * The controller uses gradient estimation to find the optimal operating point.
 * This is similar to extremum seeking control used in real-time optimization.
 */

#ifndef FORNAX_CONTROLLER_H
#define FORNAX_CONTROLLER_H

#include "ring_buffer.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace fornax {

/**
 * @brief Sample point for gradient estimation
 */
struct ControlSample {
  double duty_cycle;
  double throughput; // iterations/sec
  double power_w;
  double freq_mhz;
  std::chrono::steady_clock::time_point timestamp;
};

/**
 * @brief Gradient-based adaptive duty cycle controller
 *
 * The controller maintains a history of recent samples and estimates
 * the gradient of throughput with respect to duty cycle. It then
 * adjusts the duty cycle in the direction that increases throughput.
 *
 * Key features:
 * - Gradient estimation via finite differences
 * - Momentum to smooth out noise
 * - Bounds checking to keep duty cycle in [0, 1]
 * - Exploration via periodic perturbation
 */
class AdaptiveController {
public:
  /**
   * @brief Construct adaptive controller
   * @param initial_duty Starting duty cycle
   * @param learning_rate How aggressively to adjust (0.01-0.1 typical)
   * @param momentum Fraction of previous gradient to retain (0-0.9)
   */
  explicit AdaptiveController(double initial_duty = 0.5,
                              double learning_rate = 0.05,
                              double momentum = 0.7)
      : current_duty_(initial_duty), learning_rate_(learning_rate),
        momentum_(momentum), velocity_(0.0), best_throughput_(0.0),
        best_duty_(initial_duty), explore_phase_(true), explore_step_(0) {}

  /**
   * @brief Update controller with new measurement
   * @param measured_throughput Current throughput in iterations/sec
   * @param measured_power_w Current power in watts
   * @param measured_freq_mhz Current frequency in MHz
   * @return New duty cycle to apply
   */
  [[nodiscard]] double update(double measured_throughput,
                              double measured_power_w,
                              double measured_freq_mhz) noexcept {
    auto now = std::chrono::steady_clock::now();

    // Record sample (RingBuffer auto-evicts oldest when full)
    ControlSample sample{current_duty_, measured_throughput, measured_power_w,
                         measured_freq_mhz, now};
    history_.push_back(sample);

    // Track best seen
    if (measured_throughput > best_throughput_) {
      best_throughput_ = measured_throughput;
      best_duty_ = current_duty_;
    }

    // Exploration phase: sweep through duty cycles to build initial model
    if (explore_phase_) {
      return update_exploration();
    }

    // Exploitation phase: gradient-based optimization
    return update_gradient();
  }

  /**
   * @brief Get current duty cycle
   */
  [[nodiscard]]
  double current_duty() const {
    return current_duty_;
  }

  /**
   * @brief Get best duty cycle found so far
   */
  [[nodiscard]]
  double best_duty() const {
    return best_duty_;
  }

  /**
   * @brief Get best throughput observed
   */
  [[nodiscard]]
  double best_throughput() const {
    return best_throughput_;
  }

  /**
   * @brief Check if still in exploration phase
   */
  [[nodiscard]]
  bool is_exploring() const {
    return explore_phase_;
  }

private:
  // Configuration
  static constexpr size_t MAX_HISTORY_SIZE = 100;
  static constexpr size_t MIN_SAMPLES_FOR_GRADIENT = 5;
  // Reduced from 11 to 5 steps for faster convergence on shorter benchmarks
  // Tests 0%, 25%, 50%, 75%, 100% duty cycles
  static constexpr int EXPLORATION_STEPS = 5;

  double current_duty_;
  double learning_rate_;
  double momentum_;
  double velocity_;

  double best_throughput_;
  double best_duty_;

  bool explore_phase_;
  int explore_step_;

  RingBuffer<ControlSample, MAX_HISTORY_SIZE> history_;

  /**
   * @brief Exploration phase: systematically test different duty cycles
   */
  double update_exploration() noexcept {
    // Move to next exploration point
    explore_step_++;

    if (explore_step_ >= EXPLORATION_STEPS) {
      // Finished exploration, switch to exploitation
      explore_phase_ = false;
      current_duty_ = best_duty_; // Start from best found
      return current_duty_;
    }

    // Test evenly spaced duty cycles (0%, 25%, 50%, 75%, 100%)
    current_duty_ =
        static_cast<double>(explore_step_) / (EXPLORATION_STEPS - 1);
    return current_duty_;
  }

  /**
   * @brief Gradient-based optimization
   */
  double update_gradient() noexcept {
    if (history_.size() < MIN_SAMPLES_FOR_GRADIENT) {
      return current_duty_; // Not enough data yet
    }

    // Estimate gradient using recent history
    double gradient = estimate_gradient();

    // Update velocity with momentum
    velocity_ = momentum_ * velocity_ + (1.0 - momentum_) * gradient;

    // Update duty cycle in gradient direction
    // Note: we want to MAXIMIZE throughput, so we go WITH the gradient
    current_duty_ += learning_rate_ * velocity_;

    // Clamp to valid range
    current_duty_ = std::clamp(current_duty_, 0.0, 1.0);

    return current_duty_;
  }

  /**
   * @brief Estimate gradient of throughput w.r.t. duty cycle
   *
   * Uses finite differences on recent history. This is a simple
   * approach; more sophisticated methods (Kalman filtering, etc.)
   * could be used for noisy systems.
   */
  [[nodiscard]] double estimate_gradient() const noexcept {
    if (history_.size() < 2) {
      return 0.0;
    }

    // Simple linear regression on recent samples
    double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
    size_t n = std::min(history_.size(), size_t(10)); // Use last 10 samples

    for (size_t i = history_.size() - n; i < history_.size(); ++i) {
      double x = history_[i].duty_cycle;
      double y = history_[i].throughput;
      sum_x += x;
      sum_y += y;
      sum_xy += x * y;
      sum_xx += x * x;
    }

    double nd = static_cast<double>(n);
    double denom = nd * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-10) {
      return 0.0; // Avoid division by zero
    }

    // Slope of linear regression = gradient estimate
    double gradient = (nd * sum_xy - sum_x * sum_y) / denom;

    // Normalize by throughput scale to make learning rate more intuitive
    if (sum_y > 0) {
      gradient /= (sum_y / nd); // Relative gradient
    }

    return gradient;
  }
};

// Note: PIDController was removed as it was unused in the actual benchmark.
// The AdaptiveController with gradient-based optimization is preferred.
// If target-tracking control is needed in the future, consider implementing
// a proper PID controller with derivative filtering and integral clamping.

} // namespace fornax

#endif // FORNAX_CONTROLLER_H
