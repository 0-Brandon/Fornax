/**
 * @file numa.h
 * @brief NUMA (Non-Uniform Memory Access) topology detection and core selection
 *
 * On multi-socket systems, cores on different NUMA nodes have asymmetric
 * memory access latencies. For accurate benchmarking, the monitor and worker
 * threads should run on cores within the same NUMA node to avoid cross-socket
 * cache coherency traffic.
 *
 * Topology is read from /sys/devices/system/node/ on Linux.
 */

#ifndef FORNAX_NUMA_H
#define FORNAX_NUMA_H

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fornax {

/**
 * @brief Information about a NUMA node
 */
struct NumaNode {
  int node_id;
  std::vector<int> cpu_list;
};

/**
 * @brief Parse a CPU list string (e.g., "0-3,8-11") into core IDs
 *
 * Handles both ranges (0-3) and individual cores (5).
 */
inline std::vector<int> parse_cpu_list(const std::string &cpu_list_str) {
  std::vector<int> result;
  std::stringstream ss(cpu_list_str);
  std::string token;

  while (std::getline(ss, token, ',')) {
    size_t dash_pos = token.find('-');
    if (dash_pos != std::string::npos) {
      // Range: "0-3"
      int start = std::stoi(token.substr(0, dash_pos));
      int end = std::stoi(token.substr(dash_pos + 1));
      for (int i = start; i <= end; ++i) {
        result.push_back(i);
      }
    } else {
      // Single core: "5"
      result.push_back(std::stoi(token));
    }
  }

  return result;
}

/**
 * @brief Detect NUMA topology from sysfs
 *
 * Reads /sys/devices/system/node/nodeN/cpulist to determine which
 * cores belong to which NUMA node.
 *
 * @return Vector of NUMA nodes with their CPU lists
 */
inline std::vector<NumaNode> detect_numa_topology() {
  std::vector<NumaNode> nodes;

  // Try to read NUMA topology
  for (int node_id = 0; node_id < 16; ++node_id) { // Max 16 nodes
    std::ostringstream path;
    path << "/sys/devices/system/node/node" << node_id << "/cpulist";

    std::ifstream file(path.str());
    if (!file.is_open()) {
      break; // No more nodes
    }

    std::string cpu_list_str;
    std::getline(file, cpu_list_str);

    NumaNode node;
    node.node_id = node_id;
    node.cpu_list = parse_cpu_list(cpu_list_str);

    if (!node.cpu_list.empty()) {
      nodes.push_back(node);
    }
  }

  return nodes;
}

/**
 * @brief Find the NUMA node containing a specific core
 *
 * @param core_id CPU core ID to look up
 * @return NUMA node ID, or -1 if not found
 */
inline int get_numa_node_for_core(int core_id) {
  auto nodes = detect_numa_topology();
  for (const auto &node : nodes) {
    if (std::find(node.cpu_list.begin(), node.cpu_list.end(), core_id) !=
        node.cpu_list.end()) {
      return node.node_id;
    }
  }
  return -1;
}

/**
 * @brief Select two cores from the same NUMA node for monitor/worker
 *
 * Prefers cores 0 and 1 if they are on the same node, otherwise
 * selects the first two cores from NUMA node 0.
 *
 * @param prefer_cores Optional preferred cores (default: 0, 1)
 * @return Pair of (monitor_core, worker_core)
 */
inline std::pair<int, int> select_numa_local_cores(int prefer_monitor = 0,
                                                   int prefer_worker = 1) {
  auto nodes = detect_numa_topology();

  if (nodes.empty()) {
    // No NUMA info available, use defaults
    return {prefer_monitor, prefer_worker};
  }

  // Check if preferred cores are on the same node
  int monitor_node = get_numa_node_for_core(prefer_monitor);
  int worker_node = get_numa_node_for_core(prefer_worker);

  if (monitor_node == worker_node && monitor_node >= 0) {
    // Preferred cores are on same node
    return {prefer_monitor, prefer_worker};
  }

  // Find two cores on the same node (prefer node 0)
  for (const auto &node : nodes) {
    if (node.cpu_list.size() >= 2) {
      return {node.cpu_list[0], node.cpu_list[1]};
    }
  }

  // Fallback to defaults
  return {prefer_monitor, prefer_worker};
}

/**
 * @brief Check if NUMA is available on this system
 */
inline bool has_numa() {
  auto nodes = detect_numa_topology();
  return nodes.size() > 1;
}

/**
 * @brief Get human-readable NUMA topology description
 */
inline std::string describe_numa_topology() {
  auto nodes = detect_numa_topology();
  if (nodes.empty()) {
    return "No NUMA topology detected (single socket or unsupported)";
  }

  std::ostringstream oss;
  oss << nodes.size() << " NUMA node(s): ";
  for (const auto &node : nodes) {
    oss << "node" << node.node_id << "=[";
    for (size_t i = 0; i < node.cpu_list.size(); ++i) {
      if (i > 0)
        oss << ",";
      if (i < 4 || i >= node.cpu_list.size() - 1) {
        oss << node.cpu_list[i];
      } else if (i == 4) {
        oss << "...";
      }
    }
    oss << "] ";
  }
  return oss.str();
}

} // namespace fornax

#endif // FORNAX_NUMA_H
