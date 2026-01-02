/**
 * @file thread_utils.h
 * @brief Thread affinity and CPU pinning utilities
 *
 * Provides cross-platform thread affinity control to ensure consistent
 * benchmark results. Pinning threads to specific cores:
 * - Eliminates cache migration overhead
 * - Ensures consistent latency measurements
 * - Required for meaningful power/frequency correlation
 *
 * On Linux, uses pthread_setaffinity_np.
 * On macOS/other platforms, affinity is not supported (graceful degradation).
 */

#ifndef FORNAX_THREAD_UTILS_H
#define FORNAX_THREAD_UTILS_H

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace fornax {

/**
 * @brief Pin the current thread to a specific CPU core
 *
 * Thread affinity ensures the thread only runs on the specified core,
 * eliminating cache migration overhead and improving measurement consistency.
 *
 * @param core_id Target core (0-indexed)
 * @param thread_name Optional name for logging (e.g., "Monitor", "Worker")
 * @return true if pinning succeeded, false otherwise
 */
[[nodiscard]] inline bool pin_to_core(int core_id,
                                      const std::string &thread_name = "") {
#if defined(__linux__)
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  int result = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
  if (result != 0) {
    std::string name = thread_name.empty() ? "Thread" : thread_name;
    std::cerr << "[" << name << "] Warning: Failed to pin to core " << core_id
              << ": " << std::strerror(result) << std::endl;
    return false;
  }
  return true;
#else
  // macOS and other platforms don't support pthread_setaffinity_np
  std::string name = thread_name.empty() ? "Thread" : thread_name;
  std::cerr << "[" << name
            << "] Warning: Thread affinity not supported on this platform"
            << std::endl;
  (void)core_id; // Suppress unused parameter warning
  return false;
#endif
}

/**
 * @brief Get the number of available CPU cores
 *
 * Uses sysconf(_SC_NPROCESSORS_ONLN) on Linux.
 *
 * @return Number of online processors, or 1 if detection fails
 */
[[nodiscard]] inline int get_num_cores() {
#if defined(__linux__)
  long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  return (nprocs > 0) ? static_cast<int>(nprocs) : 1;
#else
  return 1; // Conservative fallback
#endif
}

/**
 * @brief Get the current core ID where this thread is running
 *
 * @return Current core ID, or -1 if detection fails
 */
[[nodiscard]] inline int get_current_core() {
#if defined(__linux__)
  return sched_getcpu();
#else
  return -1;
#endif
}

} // namespace fornax

#endif // FORNAX_THREAD_UTILS_H
