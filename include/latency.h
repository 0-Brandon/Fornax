/**
 * @file latency.h
 * @brief Latency measurements for sleep vs pause
 */

#ifndef FORNAX_LATENCY_H
#define FORNAX_LATENCY_H

#include <string>

namespace fornax {

/**
 * @brief Run latency benchmark and output to CSV
 *
 * Compares std::this_thread::sleep_for() vs cpu_relax() (PAUSE/YIELD)
 *
 * @param output_file CSV file to write results to
 * @param trials Number of samples per method
 */
void run_latency_test(const std::string &output_file, int trials = 10000);

} // namespace fornax

#endif // FORNAX_LATENCY_H
