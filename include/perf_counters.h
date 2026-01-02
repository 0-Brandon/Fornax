/**
 * @file perf_counters.h
 * @brief Hardware performance counter integration via perf_event_open
 *
 * Provides low-overhead access to hardware performance monitoring counters:
 * - Instruction count (retired instructions)
 * - Cache misses (L1D, LLC)
 * - Branch mispredictions
 *
 * Uses Linux perf_event_open() syscall for direct PMU access without
 * spawning a perf subprocess. Falls back gracefully on unsupported
 * platforms (ARM, containers without CAP_PERFMON).
 *
 * References:
 * - Linux perf_event_open(2) man page
 * - Intel SDM Vol. 3B Chapter 18 (Performance Monitoring)
 */

#ifndef FORNAX_PERF_COUNTERS_H
#define FORNAX_PERF_COUNTERS_H

#include <array>
#include <cstdint>
#include <optional>

// Linux-specific headers for perf_event
#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace fornax {

/**
 * @brief Hardware counter readings snapshot
 */
struct PerfCounters {
  uint64_t instructions = 0;        // Retired instructions
  uint64_t cycles = 0;              // CPU cycles
  uint64_t cache_misses = 0;        // LLC misses
  uint64_t cache_references = 0;    // LLC references
  uint64_t branch_misses = 0;       // Branch mispredictions
  uint64_t branch_instructions = 0; // Total branches

  /**
   * @brief Compute IPC (Instructions Per Cycle)
   */
  [[nodiscard]] double ipc() const noexcept {
    return cycles > 0
               ? static_cast<double>(instructions) / static_cast<double>(cycles)
               : 0.0;
  }

  /**
   * @brief Compute cache miss rate (LLC misses / references)
   */
  [[nodiscard]] double cache_miss_rate() const noexcept {
    return cache_references > 0 ? static_cast<double>(cache_misses) /
                                      static_cast<double>(cache_references)
                                : 0.0;
  }

  /**
   * @brief Compute branch misprediction rate
   */
  [[nodiscard]] double branch_miss_rate() const noexcept {
    return branch_instructions > 0
               ? static_cast<double>(branch_misses) /
                     static_cast<double>(branch_instructions)
               : 0.0;
  }
};

#if defined(__linux__)

/**
 * @brief RAII wrapper for perf_event file descriptors
 */
class PerfEventGroup {
public:
  static constexpr size_t NUM_COUNTERS = 6;

  PerfEventGroup() noexcept {
    // Initialize all FDs to -1
    for (auto &fd : fds_) {
      fd = -1;
    }
  }

  ~PerfEventGroup() { close_all(); }

  // Non-copyable
  PerfEventGroup(const PerfEventGroup &) = delete;
  PerfEventGroup &operator=(const PerfEventGroup &) = delete;

  /**
   * @brief Initialize performance counters for the current thread/core
   *
   * @param cpu CPU to monitor (-1 for calling thread regardless of CPU)
   * @return true if at least some counters were opened successfully
   */
  bool initialize(int cpu = -1) noexcept {
    pid_t pid = 0; // Current process
    unsigned long flags = 0;

    // Counter configurations: {type, config}
    struct CounterConfig {
      uint32_t type;
      uint64_t config;
    };

    std::array<CounterConfig, NUM_COUNTERS> configs = {{
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
    }};

    int group_fd = -1;
    bool any_success = false;

    for (size_t i = 0; i < NUM_COUNTERS; ++i) {
      struct perf_event_attr pe = {};
      pe.type = configs[i].type;
      pe.size = sizeof(pe);
      pe.config = configs[i].config;
      pe.disabled = (i == 0) ? 1 : 0; // Only disable group leader
      pe.exclude_kernel = 1;
      pe.exclude_hv = 1;
      pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;

      fds_[i] = static_cast<int>(
          syscall(__NR_perf_event_open, &pe, pid, cpu, group_fd, flags));

      if (fds_[i] >= 0) {
        any_success = true;
        if (i == 0) {
          group_fd = fds_[i]; // First FD becomes group leader
        }
      }
    }

    available_ = any_success;
    return any_success;
  }

  /**
   * @brief Start counting
   */
  void start() noexcept {
    if (fds_[0] >= 0) {
      ioctl(fds_[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
      ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
  }

  /**
   * @brief Stop counting
   */
  void stop() noexcept {
    if (fds_[0] >= 0) {
      ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
    }
  }

  /**
   * @brief Read current counter values
   */
  [[nodiscard]] PerfCounters read() const noexcept {
    PerfCounters result{};

    if (!available_ || fds_[0] < 0) {
      return result;
    }

    // Read format: { nr, { value, id } * nr }
    struct ReadFormat {
      uint64_t nr;
      struct {
        uint64_t value;
        uint64_t id;
      } values[NUM_COUNTERS];
    } data{};

    ssize_t bytes = ::read(fds_[0], &data, sizeof(data));
    if (bytes < static_cast<ssize_t>(sizeof(uint64_t))) {
      return result;
    }

    // Map values to result struct
    size_t count = std::min(static_cast<size_t>(data.nr), NUM_COUNTERS);
    if (count > 0)
      result.instructions = data.values[0].value;
    if (count > 1)
      result.cycles = data.values[1].value;
    if (count > 2)
      result.cache_misses = data.values[2].value;
    if (count > 3)
      result.cache_references = data.values[3].value;
    if (count > 4)
      result.branch_misses = data.values[4].value;
    if (count > 5)
      result.branch_instructions = data.values[5].value;

    return result;
  }

  [[nodiscard]] bool available() const noexcept { return available_; }

private:
  void close_all() noexcept {
    for (auto &fd : fds_) {
      if (fd >= 0) {
        ::close(fd);
        fd = -1;
      }
    }
  }

  std::array<int, NUM_COUNTERS> fds_{};
  bool available_ = false;
};

#else // Non-Linux platforms

/**
 * @brief Stub implementation for non-Linux platforms
 */
class PerfEventGroup {
public:
  bool initialize([[maybe_unused]] int cpu = -1) noexcept { return false; }
  void start() noexcept {}
  void stop() noexcept {}
  [[nodiscard]] PerfCounters read() const noexcept { return {}; }
  [[nodiscard]] bool available() const noexcept { return false; }
};

#endif // __linux__

} // namespace fornax

#endif // FORNAX_PERF_COUNTERS_H
