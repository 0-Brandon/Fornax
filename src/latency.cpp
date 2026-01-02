/**
 * @file latency.cpp
 * @brief Latency benchmark implementation
 */

#include "latency.h"
#include "arch.h"
#include "statistics.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

namespace fornax {

void run_latency_test(const std::string &output_file, int trials) {
  std::cout << "=================================================="
            << std::endl;
  std::cout << "Running Latency Benchmark" << std::endl;
  std::cout << "Comparing sleep_for(1us) vs cpu_relax() loop" << std::endl;
  std::cout << "Trials: " << trials << std::endl;
  std::cout << "Output: " << (output_file.empty() ? "stdout" : output_file)
            << std::endl;
  std::cout << "=================================================="
            << std::endl;

  std::vector<double> sleep_latencies_us;
  std::vector<double> relax_latencies_us;

  sleep_latencies_us.reserve(trials);
  relax_latencies_us.reserve(trials);

  // 1. Measure sleep_for(1us)
  // Note: We ask for 1us, but OS scheduler quantum will make it much larger
  for (int i = 0; i < trials; ++i) {
    auto start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::microseconds(1));
    auto end = std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    sleep_latencies_us.push_back(static_cast<double>(duration.count()) /
                                 1000.0);
  }

  // 2. Measure cpu_relax() loop (simulating busy-wait throttle)
  // Measure the time for a small fixed burst of relax instructions.
  for (int i = 0; i < trials; ++i) {
    auto start = std::chrono::steady_clock::now();

    // Run a small burst of relax instructions to simulate a short "throttle on"
    // check cycle.
    for (int j = 0; j < 100; ++j) {
      cpu_relax();
    }

    auto end = std::chrono::steady_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    // Average per relax instruction to estimate per-call latency.
    relax_latencies_us.push_back(
        (static_cast<double>(duration.count()) / 100.0) / 1000.0);
  }

  // Compute stats
  auto sleep_stats = compute_enhanced_stats(sleep_latencies_us);
  auto relax_stats = compute_enhanced_stats(relax_latencies_us);

  std::cout << format_enhanced_stats(sleep_stats, "sleep_for() latency (us)")
            << std::endl;
  std::cout << format_enhanced_stats(relax_stats, "cpu_relax() latency (us)")
            << std::endl;

  // Write to CSV
  if (!output_file.empty()) {
    std::ofstream csv(output_file);
    if (csv.is_open()) {
      csv << "method,latency_us\n";
      for (double v : sleep_latencies_us) {
        csv << "sleep_for," << v << "\n";
      }
      for (double v : relax_latencies_us) {
        csv << "cpu_relax," << v << "\n";
      }
      std::cout << "Wrote raw data to " << output_file << std::endl;
    } else {
      std::cerr << "Failed to open output file: " << output_file << std::endl;
    }
  }
}

} // namespace fornax
