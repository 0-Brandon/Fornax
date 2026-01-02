#!/bin/bash
# ============================================================================
# Fornax Benchmark Runner
# ============================================================================
#
# Unified entry point for all Fornax benchmark experiments.
#
# Usage:
#   ./scripts/run_benchmark.sh [mode]
#
# Modes:
#   all            Run all experiments (default)
#   quick          Run fast validation suite
#   sweep          Run duty cycle sweep only
#   workloads      Compare different workloads
#   adaptive       Test adaptive controller
#   false-sharing  Run cache line padding test
#
# ============================================================================

set -euo pipefail

# Configuration
BUILD_DIR="build"
FORNAX="$BUILD_DIR/fornax"
RESULTS_DIR="results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Default Parameters
TRIALS=5
WARMUP=2
DURATION=10
MODE="standard"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Helpers
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_section() { echo -e "\n${BLUE}=== $1 ===${NC}"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

ensure_build() {
    if [ ! -f "$FORNAX" ]; then
        log_info "Building Fornax..."
        mkdir -p "$BUILD_DIR"
        (cd "$BUILD_DIR" && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc))
    fi
}

# Experiments

run_sweep() {
    log_section "Duty Cycle Sweep"
    local out="$RESULTS_DIR/sweep_${TIMESTAMP}.csv"
    $FORNAX --sweep --trials $TRIALS --warmup $WARMUP --duration $DURATION \
            --sweep-output "$out" $SIM_FLAG
    log_info "Sweep results: $out"
}

run_workloads() {
    log_section "Workload Comparison"
    local out="$RESULTS_DIR/workloads_${TIMESTAMP}.csv"
    echo "workload,iterations_per_sec" > "$out"
    
    for wl in fma-stress black-scholes monte-carlo covariance mixed; do
        log_info "Testing $wl..."
        res=$($FORNAX --workload $wl --trials $TRIALS --warmup $WARMUP --duration $DURATION $SIM_FLAG \
              | grep "Iterations/sec:" | head -1 | grep -oP '\d+' | head -1)
        echo "$wl,$res" >> "$out"
    done
    log_info "Workload results: $out"
}

run_false_sharing() {
    log_section "False Sharing Test"
    local out="$RESULTS_DIR/false_sharing_${TIMESTAMP}.txt"
    {
        echo "## With Padding"
        $FORNAX --trials 3 --warmup 1 --duration 5 $SIM_FLAG
        echo -e "\n## Without Padding"
        $FORNAX --no-padding --trials 3 --warmup 1 --duration 5 $SIM_FLAG
    } > "$out" 2>&1
    log_info "False sharing results: $out"
}

run_adaptive() {
    log_section "Adaptive Controller"
    local out="$RESULTS_DIR/adaptive_${TIMESTAMP}.txt"
    $FORNAX --adaptive --duration 20 $SIM_FLAG > "$out" 2>&1
    log_info "Adaptive results: $out"
}

# Main Execution

mkdir -p "$RESULTS_DIR"
ensure_build

# Detect if simulation needed (no RAPL)
SIM_FLAG=""
if [ ! -f "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj" ] && \
   [ ! -f "/sys/class/powercap/intel-rapl:0/energy_uj" ]; then
    log_warn "RAPL not detected. Forcing simulation mode."
    SIM_FLAG="--simulate"
fi

COMMAND=${1:-all}

case $COMMAND in
    quick)
        TRIALS=1; WARMUP=0; DURATION=3
        run_sweep
        run_workloads
        ;;
    sweep) run_sweep ;;
    workloads) run_workloads ;;
    adaptive) run_adaptive ;;
    false-sharing) run_false_sharing ;;
    all)
        run_sweep
        run_workloads
        run_false_sharing
        run_adaptive
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        echo "Usage: $0 [all|quick|sweep|workloads|adaptive|false-sharing]"
        exit 1
        ;;
esac

log_info "Done. Results in $RESULTS_DIR"
