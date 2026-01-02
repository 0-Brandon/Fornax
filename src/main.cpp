/**
 * @file main.cpp
 * @brief Fornax SIMD Frequency Benchmark - Main orchestrator
 *
 * Fornax benchmarks the trade-off between SIMD/vector instruction throughput
 * and CPU frequency downclocking (AVX Offset phenomenon).
 *
 * Architecture:
 * - Core 0 (Monitor): Reads power/frequency sensors, controls throttle signal
 * - Core 1 (Worker): Executes heavy SIMD math, checks throttle, uses
 * cpu_relax()
 *
 * The decoupled asynchronous pattern ensures the hot path (Worker) is never
 * blocked by slow sensor reads, while still responding to power feedback.
 */

#include "arch.h"
#include "config.h"
#include "hypothesis_test.h"
#include "latency.h"
#include "numa.h"
#include "shared_state.h"
#include "statistics.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace fornax {

// Forward declarations from monitor.cpp and worker.cpp
template <typename StateType>
void monitor_thread(StateType &state, const MonitorConfig &config);

template <typename StateType>
void worker_thread(StateType &state, const WorkerConfig &config);

} // namespace fornax

// ============================================================================
// CLI Argument Parsing
// ============================================================================

struct Config {
  // Basic options
  bool no_padding = false;   // --no-padding
  bool simulate = false;     // --simulate
  double duty_cycle = -1.0;  // --duty-cycle <value>
  int duration_seconds = 10; // --duration <seconds>
  bool help = false;         // --help
  uint64_t high_threshold_w = fornax::defaults::POWER_HIGH_THRESHOLD_UW /
                              1'000'000; // --high-threshold <watts>
  uint64_t low_threshold_w = fornax::defaults::POWER_LOW_THRESHOLD_UW /
                             1'000'000; // --low-threshold <watts>

  // Statistical rigor options
  int trials = 1;         // --trials <N>
  int warmup_seconds = 2; // --warmup <seconds>

  // Sweep experiment options
  bool sweep = false;              // --sweep
  std::string sweep_output_file;   // --sweep-output <file>
  std::string monitor_output_file; // --monitor-output <file> (High-res logging)

  // Workload selection
  fornax::WorkloadType workload = fornax::WorkloadType::FMA_STRESS;

  // Adaptive control mode
  bool adaptive = false; // --adaptive

  // x86-only features (graceful degradation on ARM)
  bool hyperthreading = false; // --hyperthreading

  // NUMA options
  bool numa_local = false; // --numa-local
  int monitor_core = 0;    // --monitor-core
  int worker_core = 1;     // --worker-core

  // Latency benchmark
  bool latency_test = false;       // --latency-test
  std::string latency_output_file; // --latency-output <file>
};

void print_usage(const char *program) {
  std::cout
      << R"(
Fornax - SIMD Frequency Benchmark

USAGE:
    )" << program
      << R"( [OPTIONS]

OPTIONS:
    --no-padding          Disable cache-line padding to demonstrate false sharing
    --simulate            Force power sensor simulation mode
    --duty-cycle <val>    Override hysteresis with manual duty cycle (0.0-1.0)
    --duration <sec>      Benchmark duration in seconds (default: 10)
    --high-threshold <W>  Power threshold to enable throttling (default: 100W)
    --low-threshold <W>   Power threshold to disable throttling (default: 70W)

STATISTICAL RIGOR:
    --trials <N>          Number of trials per configuration (default: 1)
    --warmup <sec>        Warm-up seconds to discard (default: 2)

EXPERIMENTS:
    --sweep               Run duty cycle sweep (0%, 10%, ..., 100%)
    --sweep-output <file> Output sweep results to CSV file
    --monitor-output <file> Output high-res freq/power logs to CSV
    --adaptive            Use gradient-based adaptive duty cycle control
    --latency-test        Run sleep_for vs cpu_relax latency benchmark
    --latency-output <f>  Output latency data to CSV

WORKLOADS:
    --workload <type>     Workload type: fma-stress (default), black-scholes,
                          monte-carlo, covariance, mixed

NUMA OPTIONS:
    --numa-local          Auto-select cores from same NUMA node
    --monitor-core <N>    Specify monitor thread core (default: 0)
    --worker-core <N>     Specify worker thread core (default: 1)

x86-ONLY FEATURES (graceful no-op on ARM):
    --hyperthreading      Test HT sibling core interaction

    --help                Show this message

EXAMPLES:
    )" << program
      << R"(                          # Run with defaults (hysteresis control)
    )" << program
      << R"( --simulate               # Run on ARM or without RAPL
    )" << program
      << R"( --duty-cycle 0.5         # 50% throttle duty cycle
    )" << program
      << R"( --trials 10 --warmup 3   # Statistical rigor: 10 trials, 3s warmup
    )" << program
      << R"( --sweep --sweep-output results.csv

)" << std::endl;
}

Config parse_args(int argc, char *argv[]) {
  Config config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--no-padding") {
      config.no_padding = true;
    } else if (arg == "--simulate") {
      config.simulate = true;
    } else if (arg == "--duty-cycle" && i + 1 < argc) {
      config.duty_cycle = std::stod(argv[++i]);
      if (config.duty_cycle < 0.0 || config.duty_cycle > 1.0) {
        std::cerr << "Error: --duty-cycle must be between 0.0 and 1.0"
                  << std::endl;
        std::exit(1);
      }
    } else if (arg == "--duration" && i + 1 < argc) {
      config.duration_seconds = std::stoi(argv[++i]);
    } else if (arg == "--high-threshold" && i + 1 < argc) {
      config.high_threshold_w = std::stoull(argv[++i]);
    } else if (arg == "--low-threshold" && i + 1 < argc) {
      config.low_threshold_w = std::stoull(argv[++i]);
    } else if (arg == "--trials" && i + 1 < argc) {
      config.trials = std::stoi(argv[++i]);
      if (config.trials < 1) {
        std::cerr << "Error: --trials must be >= 1" << std::endl;
        std::exit(1);
      }
    } else if (arg == "--warmup" && i + 1 < argc) {
      config.warmup_seconds = std::stoi(argv[++i]);
    } else if (arg == "--sweep") {
      config.sweep = true;
    } else if (arg == "--sweep-output" && i + 1 < argc) {
      config.sweep_output_file = argv[++i];
    } else if (arg == "--monitor-output" && i + 1 < argc) {
      config.monitor_output_file = argv[++i];
    } else if (arg == "--workload" && i + 1 < argc) {
      config.workload = fornax::string_to_workload_type(argv[++i]);
    } else if (arg == "--adaptive") {
      config.adaptive = true;
    } else if (arg == "--hyperthreading") {
      config.hyperthreading = true;
      config.numa_local = true;
    } else if (arg == "--numa-local") {
      config.numa_local = true;
    } else if (arg == "--monitor-core" && i + 1 < argc) {
      config.monitor_core = std::stoi(argv[++i]);
    } else if (arg == "--worker-core" && i + 1 < argc) {
      config.worker_core = std::stoi(argv[++i]);
    } else if (arg == "--latency-test") {
      config.latency_test = true;
    } else if (arg == "--latency-output" && i + 1 < argc) {
      config.latency_output_file = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      config.help = true;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::cerr << "Use --help for usage information" << std::endl;
      std::exit(1);
    }
  }

  return config;
}

// ============================================================================
// Statistical Helpers
// ============================================================================

struct TrialResult {
  double iterations_per_sec;
  double cycles_per_iter;
  double avg_power_w;
  double avg_freq_mhz;
};

// Using EnhancedStatistics from statistics.h for proper percentile support
using StatisticalResult = fornax::EnhancedStatistics;

StatisticalResult compute_stats(const std::vector<double> &values) {
  return fornax::compute_enhanced_stats(values);
}

// Now handled by compute_enhanced_stats in statistics.h

// ============================================================================
// Single Trial Benchmark
// ============================================================================

template <typename StateType>
TrialResult run_single_trial(const Config &config, bool quiet = false) {
  using namespace fornax;

  alignas(64) StateType state;

  // Configure monitor
  MonitorConfig monitor_config;
  monitor_config.simulate_power = config.simulate;
  monitor_config.manual_duty_cycle = config.duty_cycle;
  monitor_config.power_high_threshold_uw = config.high_threshold_w * 1'000'000;
  monitor_config.power_low_threshold_uw = config.low_threshold_w * 1'000'000;
  monitor_config.adaptive_control = config.adaptive;
  monitor_config.output_file = config.monitor_output_file;

  // Configure worker
  WorkerConfig worker_config;
  worker_config.workload = config.workload;

  // Warmup phase
  if (config.warmup_seconds > 0 && !quiet) {
    std::cout << "[Trial] Warming up for " << config.warmup_seconds << "s..."
              << std::endl;
  }

  // Launch threads
  std::thread monitor(monitor_thread<StateType>, std::ref(state),
                      monitor_config);
  std::thread worker(worker_thread<StateType>, std::ref(state), worker_config);

  // Wait for warmup
  std::this_thread::sleep_for(std::chrono::seconds(config.warmup_seconds));

  // Reset counters after warmup
  auto measurement_start = std::chrono::steady_clock::now();
  uint64_t start_cycles = get_cycles();
  uint64_t start_iterations =
      state.iteration_count.load(std::memory_order_relaxed);

  // Collect per-second samples during measurement phase
  std::vector<double> per_second_throughputs;
  std::vector<double> per_second_power;
  std::vector<double> per_second_freq;
  uint64_t last_iterations = start_iterations;

  for (int sec = 0; sec < config.duration_seconds; ++sec) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    uint64_t current_iterations =
        state.iteration_count.load(std::memory_order_relaxed);
    uint64_t delta = current_iterations - last_iterations;
    per_second_throughputs.push_back(static_cast<double>(delta));

    double power_w = static_cast<double>(state.current_power_uw.load(
                         std::memory_order_relaxed)) /
                     1'000'000.0;
    double freq_mhz = static_cast<double>(state.current_freq_khz.load(
                          std::memory_order_relaxed)) /
                      1000.0;
    per_second_power.push_back(power_w);
    per_second_freq.push_back(freq_mhz);

    last_iterations = current_iterations;

    if (!quiet) {
      std::cout << "[Trial] " << (sec + 1) << "/" << config.duration_seconds
                << "s: " << std::fixed << std::setprecision(0) << delta
                << " iter/s" << std::endl;
    }
  }

  // Signal shutdown
  state.shutdown.store(true, std::memory_order_relaxed);
  monitor.join();
  worker.join();

  // Compute results
  auto measurement_end = std::chrono::steady_clock::now();
  uint64_t end_cycles = get_cycles();
  uint64_t total_iterations =
      state.iteration_count.load(std::memory_order_relaxed) - start_iterations;

  double total_seconds =
      std::chrono::duration<double>(measurement_end - measurement_start)
          .count();

  TrialResult result;
  result.iterations_per_sec =
      static_cast<double>(total_iterations) / total_seconds;
  result.cycles_per_iter =
      (total_iterations > 0) ? static_cast<double>(end_cycles - start_cycles) /
                                   static_cast<double>(total_iterations)
                             : 0.0;

  auto power_stats = compute_stats(per_second_power);
  auto freq_stats = compute_stats(per_second_freq);
  result.avg_power_w = power_stats.mean;
  result.avg_freq_mhz = freq_stats.mean;

  return result;
}

// ============================================================================
// Main Benchmark (with Statistical Rigor)
// ============================================================================

template <typename StateType> void run_benchmark(const Config &config) {
  using namespace fornax;

  std::cout << "=================================================="
            << std::endl;
  std::cout << "Fornax SIMD Frequency Benchmark" << std::endl;
  std::cout << "=================================================="
            << std::endl;
  std::cout << "Architecture: " << get_arch_name() << std::endl;
  std::cout << "SIMD Level: " << get_simd_name() << std::endl;
  std::cout << "State Type: "
            << (config.no_padding ? "No Padding (FALSE SHARING!)" : "Padded")
            << std::endl;
  std::cout << "State Size: " << sizeof(StateType) << " bytes" << std::endl;
  std::cout << "Mode: "
            << (config.duty_cycle >= 0 ? "Manual Duty Cycle"
                                       : "Hysteresis Control")
            << std::endl;
  std::cout << "Workload: " << workload_type_to_string(config.workload)
            << std::endl;
  std::cout << "Trials: " << config.trials << std::endl;
  std::cout << "Warmup: " << config.warmup_seconds << "s" << std::endl;
  std::cout << "Duration: " << config.duration_seconds << " seconds"
            << std::endl;
  std::cout << "=================================================="
            << std::endl;

  std::vector<double> throughputs;
  std::vector<double> cycles_per_iter;

  for (int trial = 0; trial < config.trials; ++trial) {
    std::cout << "\n--- Trial " << (trial + 1) << "/" << config.trials << " ---"
              << std::endl;

    auto result = run_single_trial<StateType>(config, config.trials > 1);
    throughputs.push_back(result.iterations_per_sec);
    cycles_per_iter.push_back(result.cycles_per_iter);

    std::cout << "[Trial " << (trial + 1) << "] Result: " << std::fixed
              << std::setprecision(0) << result.iterations_per_sec
              << " iter/s, " << std::setprecision(2) << result.cycles_per_iter
              << " cycles/iter" << std::endl;
  }

  // Compute statistics
  auto tp_stats = compute_stats(throughputs);
  auto cpi_stats = compute_stats(cycles_per_iter);

  std::cout << "\n=================================================="
            << std::endl;
  std::cout << "FINAL RESULTS (n=" << config.trials << ")" << std::endl;
  std::cout << "=================================================="
            << std::endl;

  if (config.trials > 1) {
    std::cout << fornax::format_enhanced_stats(tp_stats, "Iterations/sec", 0);
    std::cout << "Cycles/Iteration: " << std::setprecision(2) << cpi_stats.mean
              << " ± " << cpi_stats.confidence_margin_95() << std::endl;
  } else {
    std::cout << "Iterations/sec: " << std::fixed << std::setprecision(0)
              << tp_stats.mean << std::endl;
    std::cout << "Cycles/Iteration: " << std::setprecision(2) << cpi_stats.mean
              << std::endl;
  }

  std::cout << "=================================================="
            << std::endl;
}

// ============================================================================
// Duty Cycle Sweep Experiment
// ============================================================================

template <typename StateType> void run_sweep(const Config &base_config) {
  using namespace fornax;

  std::cout << "=================================================="
            << std::endl;
  std::cout << "Fornax Duty Cycle Sweep Experiment" << std::endl;
  std::cout << "=================================================="
            << std::endl;
  std::cout << "Architecture: " << get_arch_name() << std::endl;
  std::cout << "SIMD Level: " << get_simd_name() << std::endl;
  std::cout << "Trials per duty cycle: " << base_config.trials << std::endl;
  std::cout << "Warmup: " << base_config.warmup_seconds << "s" << std::endl;
  std::cout << "Duration per trial: " << base_config.duration_seconds << "s"
            << std::endl;
  std::cout << "=================================================="
            << std::endl;

  std::vector<double> duty_cycles = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                                     0.6, 0.7, 0.8, 0.9, 1.0};

  struct SweepResult {
    double duty_cycle;
    StatisticalResult throughput;
    StatisticalResult cycles_per_iter;
    double effective_flops_per_sec; // iterations * ops_per_iter * flops_per_op
  };

  std::vector<SweepResult> results;
  std::vector<std::vector<double>>
      all_throughputs; // Store raw data for hypothesis testing

  for (double dc : duty_cycles) {
    std::cout << "\n>>> Testing duty cycle: " << std::fixed
              << std::setprecision(0) << (dc * 100) << "%" << std::endl;

    Config trial_config = base_config;
    trial_config.duty_cycle = dc;

    std::vector<double> throughputs;
    std::vector<double> cpi_values;

    for (int trial = 0; trial < base_config.trials; ++trial) {
      auto result = run_single_trial<StateType>(trial_config, true);
      throughputs.push_back(result.iterations_per_sec);
      cpi_values.push_back(result.cycles_per_iter);
    }

    all_throughputs.push_back(throughputs);

    SweepResult sr;
    sr.duty_cycle = dc;
    sr.throughput = compute_stats(throughputs);
    sr.cycles_per_iter = compute_stats(cpi_values);

    // Effective FLOPS calculation:
    // Each iteration does ops_per_iteration FMA operations
    // Each AVX-512 FMA does 8 double-precision FMAs = 16 FLOPS (2 ops each)
    // So effective_flops = iterations * 1000 * 8 * 2 = iterations * 16000
    constexpr double FLOPS_PER_ITER = 1000.0 * 8.0 * 2.0;
    sr.effective_flops_per_sec = sr.throughput.mean * FLOPS_PER_ITER;

    results.push_back(sr);

    std::cout << "Result: " << std::setprecision(0) << sr.throughput.mean
              << " ± " << sr.throughput.confidence_margin_95() << " iter/s"
              << std::endl;

    // Hypothesis Test: Compare with previous duty cycle (if available)
    // results already contains the current result (pushed at line 471)
    if (results.size() >= 2) {
      const auto &prev_results = results[results.size() - 2];
      const auto &prev_throughputs =
          all_throughputs[all_throughputs.size() - 2];

      std::string label1 =
          std::to_string((int)(prev_results.duty_cycle * 100)) + "%";
      std::string label2 = std::to_string((int)(dc * 100)) + "%";

      // Only run test if we have enough samples
      if (throughputs.size() > 1 && prev_throughputs.size() > 1) {
        auto test_result = fornax::welch_t_test(throughputs, prev_throughputs);
        fornax::print_hypothesis_test(test_result, label2, label1);
      }
    }
  }

  // Print summary table
  std::cout << "\n=================================================="
            << std::endl;
  std::cout << "SWEEP RESULTS SUMMARY" << std::endl;
  std::cout << "=================================================="
            << std::endl;
  std::cout << std::setw(10) << "Duty%" << std::setw(15) << "Iter/s"
            << std::setw(12) << "±95%CI" << std::setw(15) << "Eff.GFLOPS"
            << std::endl;
  std::cout << std::string(52, '-') << std::endl;

  double max_throughput = 0;
  double optimal_duty = 0;

  for (const auto &r : results) {
    std::cout << std::setw(9) << std::setprecision(0) << (r.duty_cycle * 100)
              << "%" << std::setw(15) << r.throughput.mean << std::setw(12)
              << r.throughput.confidence_margin_95() << std::setw(15)
              << std::setprecision(2) << (r.effective_flops_per_sec / 1e9)
              << std::endl;

    if (r.throughput.mean > max_throughput) {
      max_throughput = r.throughput.mean;
      optimal_duty = r.duty_cycle;
    }
  }

  std::cout << std::string(52, '-') << std::endl;
  std::cout << "Optimal duty cycle: " << std::setprecision(0)
            << (optimal_duty * 100) << "% (" << std::setprecision(0)
            << max_throughput << " iter/s)" << std::endl;

  // Output CSV if requested
  if (!base_config.sweep_output_file.empty()) {
    std::ofstream csv(base_config.sweep_output_file);
    if (csv.is_open()) {
      csv << "duty_cycle,mean_iter_per_sec,stddev,ci95,effective_gflops\n";
      for (const auto &r : results) {
        csv << std::fixed << std::setprecision(2) << r.duty_cycle << ","
            << std::setprecision(0) << r.throughput.mean << ","
            << r.throughput.stddev << "," << r.throughput.confidence_margin_95()
            << "," << std::setprecision(2) << (r.effective_flops_per_sec / 1e9)
            << "\n";
      }
      std::cout << "Results written to: " << base_config.sweep_output_file
                << std::endl;
    }
  }

  std::cout << "=================================================="
            << std::endl;
}

// ============================================================================
// x86-only Feature Checks
// ============================================================================

void warn_x86_only_feature(const char *feature_name) {
#if !FORNAX_ARCH_X86
  std::cerr << "[Warning] " << feature_name
            << " is only supported on x86 architecture." << std::endl;
  std::cerr << "          Running on " << fornax::get_arch_name()
            << " - feature disabled." << std::endl;
#else
  (void)feature_name; // Suppress unused parameter warning
#endif
}

// ============================================================================
// Entry Point
// ============================================================================

int main(int argc, char *argv[]) {
  Config config = parse_args(argc, argv);

  if (config.help) {
    print_usage(argv[0]);
    return 0;
  }

  // Run latency benchmark if requested
  if (config.latency_test) {
    fornax::run_latency_test(config.latency_output_file, config.trials * 1000);
    return 0;
  }

  // Select benchmark mode
  if (config.sweep) {
    if (config.no_padding) {
      run_sweep<fornax::SharedStateNoPadding>(config);
    } else {
      run_sweep<fornax::SharedState>(config);
    }
  } else {
    if (config.no_padding) {
      run_benchmark<fornax::SharedStateNoPadding>(config);
    } else {
      run_benchmark<fornax::SharedState>(config);
    }
  }

  return 0;
}
