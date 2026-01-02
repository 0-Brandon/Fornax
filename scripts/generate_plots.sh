#!/bin/bash
# ============================================================================
# Fornax Plot Generation Script
# ============================================================================
#
# Generates visualizations from benchmark results.
# Uses the Python visualization scripts in the project.
#
# Usage:
#   ./scripts/generate_plots.sh [results_dir]
#
# ============================================================================

set -e

RESULTS_DIR="${1:-results}"
PLOTS_DIR="artifacts/plots"
PYTHON="python3"

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

mkdir -p "$PLOTS_DIR"

# Generate plots from sweep results
if ls "$RESULTS_DIR"/sweep_*.csv 1> /dev/null 2>&1; then
    LATEST_SWEEP=$(ls -t "$RESULTS_DIR"/sweep_*.csv | head -1)
    log_info "Generating sweep plot from: $LATEST_SWEEP"
    $PYTHON visualize.py --input "$LATEST_SWEEP" --output "$PLOTS_DIR/duty_cycle_sweep.png" 2>/dev/null || true
else
    log_info "No sweep data found. Generating DEMO visualizations..."
    $PYTHON visualize.py --demo 2>/dev/null || true
fi

# Run individual artifact generators if they exist
if [ -f "artifacts/generate_latency_histogram.py" ]; then
    log_info "Generating latency histogram..."
    $PYTHON artifacts/generate_latency_histogram.py 2>/dev/null || true
fi

if [ -f "artifacts/generate_frequency_power_plot.py" ]; then
    log_info "Generating frequency/power plot..."
    $PYTHON artifacts/generate_frequency_power_plot.py 2>/dev/null || true
fi

if [ -f "artifacts/generate_workload_plots.py" ]; then
    log_info "Generating workload comparison plots..."
    $PYTHON artifacts/generate_workload_plots.py 2>/dev/null || true
fi

# Generate VRM measurement plots if data exists
if ls "$RESULTS_DIR"/vrm_*.txt 1> /dev/null 2>&1; then
    log_info "VRM measurement data found - included in summary"
fi

# Generate significance testing summary if data exists
if ls "$RESULTS_DIR"/significance_*.txt 1> /dev/null 2>&1; then
    log_info "Statistical significance data found - included in summary"
fi

log_info "Plots saved to: $PLOTS_DIR/"
ls -la "$PLOTS_DIR/"
