/**
 * @file statistics.h
 * @brief Enhanced statistical analysis for Fornax benchmark
 *
 * Provides robust statistics including percentile calculations (P50, P95, P99,
 * P99.9) and interquartile range (IQR) for outlier detection. These metrics are
 * critical for understanding tail latency behavior in performance-sensitive
 * applications.
 *
 * In trading systems, tail latencies can be more impactful than mean
 * performance. A system with good P50 but poor P99.9 will exhibit occasional
 * jitter that can cause missed trading opportunities or execution delays.
 */

#ifndef FORNAX_STATISTICS_H
#define FORNAX_STATISTICS_H

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

namespace fornax {

/**
 * @brief Enhanced statistical results including percentiles
 *
 * Extends basic mean/stddev with:
 * - Percentiles: P50 (median), P95, P99, P99.9
 * - IQR: Interquartile range (P75 - P25) for robust spread estimation
 * - Robust to outliers unlike standard deviation
 */
struct EnhancedStatistics {
  // Basic statistics
  double mean = 0.0;
  double stddev = 0.0;
  double min = 0.0;
  double max = 0.0;

  // Percentiles (critical for tail latency analysis)
  double p25 = 0.0;  // First quartile
  double p50 = 0.0;  // Median
  double p75 = 0.0;  // Third quartile
  double p95 = 0.0;  // 95th percentile
  double p99 = 0.0;  // 99th percentile
  double p999 = 0.0; // 99.9th percentile ("three nines")

  // Robust statistics
  double iqr = 0.0; // Interquartile range (P75 - P25)

  int n = 0;

  /**
   * @brief 95% confidence interval margin using t-distribution approximation
   *
   * For n >= 30, t ≈ 1.96 (using 2.0 as conservative estimate)
   * CI = mean ± margin
   */
  [[nodiscard]] double confidence_margin_95() const {
    return (n > 1) ? 2.0 * stddev / std::sqrt(static_cast<double>(n)) : 0.0;
  }

  /**
   * @brief Check if a value is an outlier using IQR method
   *
   * Values outside [P25 - 1.5*IQR, P75 + 1.5*IQR] are outliers.
   * This is more robust than z-score for non-normal distributions.
   */
  [[nodiscard]] bool is_outlier(double value) const {
    double lower = p25 - 1.5 * iqr;
    double upper = p75 + 1.5 * iqr;
    return value < lower || value > upper;
  }
};

namespace detail {

/**
 * @brief Compute percentile using linear interpolation
 *
 * @param sorted_values Pre-sorted vector of values
 * @param percentile Percentile to compute (0.0 to 1.0)
 * @return Interpolated percentile value
 */
inline double compute_percentile(const std::vector<double> &sorted_values,
                                 double percentile) {
  if (sorted_values.empty()) {
    return 0.0;
  }

  if (sorted_values.size() == 1) {
    return sorted_values[0];
  }

  double n = static_cast<double>(sorted_values.size());
  double index = percentile * (n - 1);
  size_t lower = static_cast<size_t>(std::floor(index));
  size_t upper = static_cast<size_t>(std::ceil(index));

  if (lower == upper || upper >= sorted_values.size()) {
    return sorted_values[lower];
  }

  // Linear interpolation between adjacent values
  double frac = index - static_cast<double>(lower);
  return sorted_values[lower] * (1.0 - frac) + sorted_values[upper] * frac;
}

} // namespace detail

/**
 * @brief Compute enhanced statistics from a vector of values
 *
 * Computes all statistics in a single pass through the data (after sorting).
 * Sorting is O(n log n), all other operations are O(n) or O(1).
 *
 * @param values Vector of measurement values
 * @return EnhancedStatistics with all fields populated
 */
[[nodiscard]] inline EnhancedStatistics
compute_enhanced_stats(const std::vector<double> &values) {
  EnhancedStatistics result{};
  result.n = static_cast<int>(values.size());

  if (values.empty()) {
    return result;
  }

  // Mean
  result.mean = std::accumulate(values.begin(), values.end(), 0.0) /
                static_cast<double>(values.size());

  // Min/Max
  auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
  result.min = *min_it;
  result.max = *max_it;

  // Standard deviation (using Welford's algorithm for numerical stability)
  if (values.size() > 1) {
    double sq_sum = 0.0;
    for (double v : values) {
      double diff = v - result.mean;
      sq_sum += diff * diff;
    }
    result.stddev = std::sqrt(sq_sum / static_cast<double>(values.size() - 1));
  }

  // Sort for percentile calculations
  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());

  // Compute percentiles
  result.p25 = detail::compute_percentile(sorted, 0.25);
  result.p50 = detail::compute_percentile(sorted, 0.50);
  result.p75 = detail::compute_percentile(sorted, 0.75);
  result.p95 = detail::compute_percentile(sorted, 0.95);
  result.p99 = detail::compute_percentile(sorted, 0.99);
  result.p999 = detail::compute_percentile(sorted, 0.999);

  // Interquartile range
  result.iqr = result.p75 - result.p25;

  return result;
}

/**
 * @brief Format statistics for display
 *
 * Returns a multi-line string suitable for console output.
 */
[[nodiscard]] inline std::string
format_enhanced_stats(const EnhancedStatistics &stats,
                      const std::string &metric_name = "Value",
                      int precision = 2) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(precision);

  oss << metric_name << ": " << stats.mean;
  if (stats.n > 1) {
    oss << " ± " << stats.confidence_margin_95() << " (95% CI)";
  }
  oss << "\n";

  if (stats.n > 1) {
    oss << "  σ = " << stats.stddev << ", min = " << stats.min
        << ", max = " << stats.max << "\n";
    oss << "  P50 = " << stats.p50 << ", P99 = " << stats.p99
        << ", P99.9 = " << stats.p999 << "\n";
  }

  return oss.str();
}

} // namespace fornax

#endif // FORNAX_STATISTICS_H
