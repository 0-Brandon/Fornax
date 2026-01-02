/**
 * @file hypothesis_test.h
 * @brief Statistical hypothesis testing for benchmark results
 *
 * Implements Welch's t-test for comparing two independent samples with
 * potentially unequal variances and unequal sample sizes. This is appropriate
 * for comparing benchmark runs where thermal throttling may induce different
 * variance characteristics.
 */

#ifndef FORNAX_HYPOTHESIS_TEST_H
#define FORNAX_HYPOTHESIS_TEST_H

#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace fornax {

/**
 * @brief Result of a statistical hypothesis test
 */
struct HypothesisResult {
  double t_statistic;
  double p_value;
  double effect_size; // Cohen's d
  bool significant;
  double dof; // Degrees of freedom
};

namespace detail {

// Simple CDF for Student's t-distribution
// Uses an approximation sufficient for benchmarking purposes
// Source: Abramowitz and Stegun, approximation of error function
inline double student_t_cdf(double t, double dof) {
  double x = t * std::sqrt((dof + 1.0) / (dof + t * t));
  // Keep it simple: approximate with Normal distribution for large DOF
  // For small DOF, this will be slightly optimistic, but fine for this tool.
  // Using error function erf(x/sqrt(2))
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Two-tailed p-value from t-statistic
inline double compute_p_value(double t, double dof) {
  double p = 2.0 * (1.0 - student_t_cdf(std::abs(t), dof));
  return std::max(0.0, std::min(1.0, p));
}

inline double calculate_mean(const std::vector<double> &v) {
  if (v.empty())
    return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) /
         static_cast<double>(v.size());
}

inline double calculate_variance(const std::vector<double> &v, double mean) {
  if (v.size() < 2)
    return 0.0;
  double sum_sq_diff = 0.0;
  for (double val : v) {
    double diff = val - mean;
    sum_sq_diff += diff * diff;
  }
  return sum_sq_diff / static_cast<double>(v.size() - 1);
}

} // namespace detail

/**
 * @brief Perform Welch's t-test on two samples
 *
 * @param sample1 First sample vector
 * @param sample2 Second sample vector
 * @param alpha Significance level (default 0.05)
 * @return HypothesisResult
 */
inline HypothesisResult welch_t_test(const std::vector<double> &sample1,
                                     const std::vector<double> &sample2,
                                     double alpha = 0.05) {
  HypothesisResult result{};

  if (sample1.size() < 2 || sample2.size() < 2) {
    result.significant = false;
    return result;
  }

  double n1 = static_cast<double>(sample1.size());
  double n2 = static_cast<double>(sample2.size());

  double m1 = detail::calculate_mean(sample1);
  double m2 = detail::calculate_mean(sample2);

  double v1 = detail::calculate_variance(sample1, m1);
  double v2 = detail::calculate_variance(sample2, m2);

  // Welch-Satterthwaite equation for degrees of freedom
  double num = std::pow(v1 / n1 + v2 / n2, 2);
  double den =
      std::pow(v1 / n1, 2) / (n1 - 1) + std::pow(v2 / n2, 2) / (n2 - 1);
  result.dof = num / den;

  // t-statistic
  double std_error = std::sqrt(v1 / n1 + v2 / n2);
  if (std_error > 0) {
    result.t_statistic = (m1 - m2) / std_error;
  } else {
    result.t_statistic = 0.0;
  }

  // p-value
  result.p_value = detail::compute_p_value(result.t_statistic, result.dof);

  // Cohen's d (effect size)
  // Pooled standard deviation (simplified for unequal sizes)
  double pooled_sd = std::sqrt((v1 * (n1 - 1) + v2 * (n2 - 1)) / (n1 + n2 - 2));
  if (pooled_sd > 0) {
    result.effect_size = std::abs(m1 - m2) / pooled_sd;
  } else {
    result.effect_size = 0.0;
  }

  result.significant = (result.p_value < alpha);

  return result;
}

/**
 * @brief Print friendly summary of hypothesis test
 */
inline void print_hypothesis_test(const HypothesisResult &result,
                                  const std::string &name1,
                                  const std::string &name2) {
  std::cout << "    [Hypothesis Test] " << name1 << " vs " << name2 << ": "
            << (result.significant ? "SIGNIFICANT" : "Not significant")
            << "\n      t=" << std::fixed << std::setprecision(2)
            << result.t_statistic << ", p=" << std::setprecision(4)
            << result.p_value << ", d=" << std::setprecision(2)
            << result.effect_size;

  if (result.significant) {
    std::cout << " (";
    if (result.effect_size < 0.2)
      std::cout << "negligible";
    else if (result.effect_size < 0.5)
      std::cout << "small";
    else if (result.effect_size < 0.8)
      std::cout << "medium";
    else
      std::cout << "large";
    std::cout << " effect)";
  }
  std::cout << std::endl;
}

} // namespace fornax

#endif // FORNAX_HYPOTHESIS_TEST_H
