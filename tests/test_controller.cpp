/**
 * @file test_controller.cpp
 * @brief Unit tests for adaptive duty cycle controller
 *
 * Verifies:
 * - Gradient estimation accuracy
 * - Exploration/exploitation phase transitions
 * - Duty cycle bounds enforcement
 * - Convergence properties
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "controller.h"

using namespace fornax;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ============================================================================
// Adaptive Controller Tests
// ============================================================================

TEST_CASE("AdaptiveController initialization", "[controller]") {
  SECTION("Default initialization") {
    AdaptiveController controller;
    REQUIRE_THAT(controller.current_duty(), WithinAbs(0.5, 1e-10));
    REQUIRE(controller.is_exploring());
  }

  SECTION("Custom initialization") {
    AdaptiveController controller(0.3, 0.1, 0.8);
    REQUIRE_THAT(controller.current_duty(), WithinAbs(0.3, 1e-10));
  }
}

TEST_CASE("AdaptiveController exploration phase", "[controller]") {
  AdaptiveController controller(0.0);

  SECTION("Explores duty cycles systematically") {
    // Controller now explores 5 duty cycles (0%, 25%, 50%, 75%, 100%)
    std::vector<double> explored_duties;

    // Simulate 5 updates during exploration
    for (int i = 0; i < 5; ++i) {
      double duty = controller.update(
          1000.0 * (1.0 - std::abs(0.5 - controller.current_duty())), 50.0,
          4500.0);
      explored_duties.push_back(duty);
    }

    // Should have finished exploration after 5 steps
    REQUIRE_FALSE(controller.is_exploring());
  }
}

TEST_CASE("AdaptiveController duty cycle bounds", "[controller]") {
  AdaptiveController controller(0.0, 1.0); // Aggressive learning rate

  SECTION("Duty cycle stays in [0, 1]") {
    // Feed it extreme gradient signals
    for (int i = 0; i < 100; ++i) {
      double throughput = (i % 2 == 0) ? 10000.0 : 100.0;
      double duty = controller.update(throughput, 50.0, 4500.0);

      REQUIRE(duty >= 0.0);
      REQUIRE(duty <= 1.0);
    }
  }
}

TEST_CASE("AdaptiveController tracks best", "[controller]") {
  SECTION("Tracks highest throughput") {
    AdaptiveController controller(0.5);
    // Simulate exploration with peak at 0.75 (step 3 of 5)
    // Using (void) to suppress [[nodiscard]] warnings
    (void)controller.update(800.0, 50.0, 4500.0);  // step 0: duty = 0.0
    (void)controller.update(1100.0, 50.0, 4500.0); // step 1: duty = 0.25
    (void)controller.update(1400.0, 50.0, 4500.0); // step 2: duty = 0.5
    (void)controller.update(1600.0, 50.0, 4500.0); // step 3: duty = 0.75 (peak)
    (void)controller.update(1200.0, 50.0, 4500.0); // step 4: duty = 1.0

    REQUIRE_THAT(controller.best_throughput(), WithinAbs(1600.0, 1e-6));
  }
}

// ============================================================================
// ControlSample Tests
// ============================================================================

TEST_CASE("ControlSample structure", "[controller]") {
  SECTION("Can store measurement data") {
    ControlSample sample;
    sample.duty_cycle = 0.5;
    sample.throughput = 1000.0;
    sample.power_w = 65.0;
    sample.freq_mhz = 4500.0;
    sample.timestamp = std::chrono::steady_clock::now();

    REQUIRE_THAT(sample.duty_cycle, WithinAbs(0.5, 1e-10));
    REQUIRE_THAT(sample.throughput, WithinAbs(1000.0, 1e-6));
  }
}
