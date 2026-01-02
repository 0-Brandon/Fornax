#!/usr/bin/env python3
"""
Fornax Enhanced Visualization Suite
====================================

Generates publication-quality visualizations from Fornax benchmark data:
1. Time-series plots with throttle annotations
2. Phase plots (Power vs Frequency trajectory)
3. Latency CDF distributions
4. Duty cycle sweep analysis
5. Workload comparison charts

Requires: matplotlib, numpy, pandas (optional for CSV import)
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# Try importing matplotlib, fall back to ASCII if not available
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import FuncFormatter
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found, using ASCII visualization")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TimeSample:
    """Single time-series sample"""
    timestamp_sec: float
    iterations_per_sec: float
    power_w: float
    freq_mhz: float
    throttled: bool
    duty_cycle: Optional[float] = None


@dataclass
class SweepResult:
    """Single duty cycle sweep result"""
    duty_cycle: float
    mean_iter_per_sec: float
    stddev: float
    ci95: float
    effective_gflops: float


# =============================================================================
# ASCII Visualization (Fallback)
# =============================================================================

def ascii_bar(value: float, max_value: float, width: int = 40) -> str:
    """Generate ASCII progress bar"""
    filled = int((value / max_value) * width) if max_value > 0 else 0
    return "█" * filled + "░" * (width - filled)


def ascii_time_series(samples: List[TimeSample]) -> str:
    """Generate ASCII time series visualization"""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("TIME SERIES VISUALIZATION")
    lines.append("=" * 60)
    
    if not samples:
        return "\n".join(lines) + "\n(No data)"
    
    max_iter = max(s.iterations_per_sec for s in samples)
    
    lines.append(f"{'Time':<8} {'Iter/s':>12} {'Power':>8} {'Throttle':>10}")
    lines.append("-" * 60)
    
    for s in samples:
        bar = ascii_bar(s.iterations_per_sec, max_iter, 25)
        throttle_marker = "🔴 YES" if s.throttled else "🟢 NO"
        lines.append(f"{s.timestamp_sec:>6.1f}s {s.iterations_per_sec:>12,.0f} {s.power_w:>7.1f}W {throttle_marker}")
        lines.append(f"        {bar}")
    
    return "\n".join(lines)


def ascii_sweep_results(results: List[SweepResult]) -> str:
    """Generate ASCII sweep visualization"""
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("DUTY CYCLE SWEEP RESULTS")
    lines.append("=" * 60)
    
    if not results:
        return "\n".join(lines) + "\n(No data)"
    
    max_iter = max(r.mean_iter_per_sec for r in results)
    
    lines.append(f"{'Duty %':>8} {'Mean Iter/s':>14} {'±95% CI':>10} {'Rel %':>8}")
    lines.append("-" * 60)
    
    for r in results:
        bar = ascii_bar(r.mean_iter_per_sec, max_iter, 25)
        rel_pct = (r.mean_iter_per_sec / max_iter) * 100
        lines.append(f"{r.duty_cycle*100:>6.0f}% {r.mean_iter_per_sec:>14,.0f} ±{r.ci95:>8,.0f} {rel_pct:>7.1f}%")
        lines.append(f"        {bar}")
    
    # Find optimal
    optimal = max(results, key=lambda r: r.mean_iter_per_sec)
    lines.append("\n" + "-" * 60)
    lines.append(f"⭐ Optimal: {optimal.duty_cycle*100:.0f}% duty cycle = {optimal.mean_iter_per_sec:,.0f} iter/s")
    
    return "\n".join(lines)


# =============================================================================
# Matplotlib Visualization
# =============================================================================

def setup_plot_style():
    """Configure matplotlib for publication-quality output"""
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'ggplot')
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
    })


def plot_time_series(samples: List[TimeSample], output_path: Optional[str] = None):
    """Generate time series plot with throttle annotations"""
    if not HAS_MATPLOTLIB:
        print(ascii_time_series(samples))
        return
    
    setup_plot_style()
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    times = [s.timestamp_sec for s in samples]
    iters = [s.iterations_per_sec for s in samples]
    power = [s.power_w for s in samples]
    freq = [s.freq_mhz for s in samples]
    
    # Mark throttle regions
    throttle_regions = []
    in_throttle = False
    start = 0
    for i, s in enumerate(samples):
        if s.throttled and not in_throttle:
            start = s.timestamp_sec
            in_throttle = True
        elif not s.throttled and in_throttle:
            throttle_regions.append((start, s.timestamp_sec))
            in_throttle = False
    if in_throttle:
        throttle_regions.append((start, samples[-1].timestamp_sec))
    
    # Plot 1: Throughput
    ax1 = axes[0]
    ax1.plot(times, iters, 'b-', linewidth=1.5, label='Throughput')
    ax1.set_ylabel('Iterations/sec')
    ax1.set_title('Throughput Time Series (Measured)', fontsize=12)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Add throttle shading
    for start, end in throttle_regions:
        ax1.axvspan(start, end, alpha=0.2, color='red', label='Throttled')
    
    # Plot 2: Power
    ax2 = axes[1]
    ax2.plot(times, power, 'r-', linewidth=1.5, label='Power')
    ax2.set_ylabel('Power (W)')
    ax2.axhline(y=50, color='darkred', linestyle='--', alpha=0.5, label='High Threshold')
    ax2.axhline(y=35, color='green', linestyle='--', alpha=0.5, label='Low Threshold')
    ax2.legend(loc='upper right')
    
    for start, end in throttle_regions:
        ax2.axvspan(start, end, alpha=0.2, color='red')
    
    # Plot 3: Frequency
    ax3 = axes[2]
    ax3.plot(times, freq, 'g-', linewidth=1.5, label='Frequency')
    ax3.set_ylabel('Frequency (MHz)')
    ax3.set_xlabel('Time (seconds)')
    
    for start, end in throttle_regions:
        ax3.axvspan(start, end, alpha=0.2, color='red')
    
    # Add legend for throttle regions
    throttle_patch = mpatches.Patch(color='red', alpha=0.2, label='Throttled')
    ax1.legend(handles=[throttle_patch], loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_phase_diagram(samples: List[TimeSample], output_path: Optional[str] = None):
    """Generate Power vs Frequency phase plot showing system trajectory"""
    if not HAS_MATPLOTLIB:
        print("Phase diagrams require matplotlib")
        return
    
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    power = [s.power_w for s in samples]
    freq = [s.freq_mhz for s in samples]
    throttled = [s.throttled for s in samples]
    
    # Color by throttle state
    colors = ['red' if t else 'green' for t in throttled]
    
    # Plot trajectory
    ax.scatter(power, freq, c=colors, alpha=0.6, s=30)
    
    # Draw trajectory lines (faded)
    ax.plot(power, freq, 'gray', alpha=0.3, linewidth=0.5)
    
    # Mark start and end
    ax.scatter([power[0]], [freq[0]], c='blue', s=100, marker='o', 
               label='Start', zorder=5, edgecolors='white')
    ax.scatter([power[-1]], [freq[-1]], c='purple', s=100, marker='s', 
               label='End', zorder=5, edgecolors='white')
    
    ax.set_xlabel('Power (W)')
    ax.set_ylabel('Frequency (MHz)')
    ax.set_title('Power-Frequency Phase Diagram\n(Green=Active, Red=Throttled)')
    
    # Add threshold lines
    ax.axvline(x=50, color='darkred', linestyle='--', alpha=0.5, label='High Threshold')
    ax.axvline(x=35, color='green', linestyle='--', alpha=0.5, label='Low Threshold')
    
    ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_latency_cdf(latencies_us: List[float], output_path: Optional[str] = None):
    """Generate latency CDF plot"""
    if not HAS_MATPLOTLIB or not HAS_NUMPY:
        print("CDF plots require matplotlib and numpy")
        return
    
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sorted_lat = np.sort(latencies_us)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    
    ax.plot(sorted_lat, cdf, 'b-', linewidth=2)
    
    # Mark percentiles
    p50 = np.percentile(sorted_lat, 50)
    p95 = np.percentile(sorted_lat, 95)
    p99 = np.percentile(sorted_lat, 99)
    
    ax.axhline(y=0.50, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.95, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=0.99, color='gray', linestyle=':', alpha=0.5)
    
    ax.axvline(x=p50, color='green', linestyle='--', alpha=0.7, label=f'P50: {p50:.1f}µs')
    ax.axvline(x=p95, color='orange', linestyle='--', alpha=0.7, label=f'P95: {p95:.1f}µs')
    ax.axvline(x=p99, color='red', linestyle='--', alpha=0.7, label=f'P99: {p99:.1f}µs')
    
    ax.set_xlabel('Wake Latency (µs)')
    ax.set_ylabel('CDF')
    ax.set_title('Wake Latency Cumulative Distribution Function')
    ax.legend(loc='lower right')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_sweep_results(results: List[SweepResult], output_path: Optional[str] = None):
    """Generate duty cycle sweep bar chart with error bars"""
    if not HAS_MATPLOTLIB:
        print(ascii_sweep_results(results))
        return
    
    setup_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    duty_pcts = [r.duty_cycle * 100 for r in results]
    means = [r.mean_iter_per_sec for r in results]
    ci95s = [r.ci95 for r in results]
    gflops = [r.effective_gflops for r in results]
    
    # Find optimal
    optimal_idx = means.index(max(means))
    colors = ['#2ecc71' if i == optimal_idx else '#3498db' for i in range(len(results))]
    
    # Plot 1: Throughput with error bars
    bars = ax1.bar(duty_pcts, means, yerr=ci95s, capsize=3, color=colors, edgecolor='white')
    ax1.set_xlabel('Duty Cycle (%)')
    ax1.set_ylabel('Iterations/sec')
    ax1.set_title('Throughput vs Duty Cycle (±95% CI)')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    # Annotate optimal
    ax1.annotate(f'Optimal: {duty_pcts[optimal_idx]:.0f}%',
                xy=(duty_pcts[optimal_idx], means[optimal_idx]),
                xytext=(duty_pcts[optimal_idx] + 10, means[optimal_idx] * 1.05),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    # Plot 2: Effective GFLOPS
    ax2.bar(duty_pcts, gflops, color='#e74c3c', edgecolor='white')
    ax2.set_xlabel('Duty Cycle (%)')
    ax2.set_ylabel('Effective GFLOPS')
    ax2.set_title('Effective Computation Rate')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_workload_comparison(workload_results: dict, output_path: Optional[str] = None):
    """Generate workload comparison chart"""
    if not HAS_MATPLOTLIB:
        print("Workload comparison requires matplotlib")
        return
    
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    workloads = list(workload_results.keys())
    throughputs = [workload_results[w]['iter_per_sec'] for w in workloads]
    cycles = [workload_results[w]['cycles_per_iter'] for w in workloads]
    
    x = range(len(workloads))
    width = 0.35
    
    bars1 = ax.bar([i - width/2 for i in x], throughputs, width, label='Throughput', color='#3498db')
    
    ax2 = ax.twinx()
    bars2 = ax2.bar([i + width/2 for i in x], cycles, width, label='Cycles/Iter', color='#e74c3c', alpha=0.7)
    
    ax.set_ylabel('Iterations/sec', color='#3498db')
    ax2.set_ylabel('Cycles/Iteration', color='#e74c3c')
    ax.set_xlabel('Workload')
    ax.set_title('Workload Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=15)
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1000:.0f}K'))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()


# =============================================================================
# Data Loading
# =============================================================================

def load_sweep_csv(path: str) -> List[SweepResult]:
    """Load sweep results from CSV file"""
    results = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(SweepResult(
                duty_cycle=float(row['duty_cycle']),
                mean_iter_per_sec=float(row['mean_iter_per_sec']),
                stddev=float(row.get('stddev', 0)),
                ci95=float(row.get('ci95', 0)),
                effective_gflops=float(row.get('effective_gflops', 0))
            ))
    return results


def generate_sample_data() -> Tuple[List[TimeSample], List[SweepResult]]:
    """Generate sample data for demonstration"""
    # Time series with simulated throttling
    time_samples = []
    import random
    random.seed(42)
    
    throttled = False
    power = 45.0
    
    for t in range(60):
        # Simulate power oscillation
        if throttled:
            power = max(30, power - 2 + random.uniform(-1, 1))
            if power < 38:
                throttled = False
        else:
            power = min(65, power + 1.5 + random.uniform(-1, 1))
            if power > 52:
                throttled = True
        
        iter_sec = 100000 if not throttled else 50000 + random.uniform(-5000, 5000)
        freq = 3200 if not throttled else 2800 + random.uniform(-50, 50)
        
        time_samples.append(TimeSample(
            timestamp_sec=float(t),
            iterations_per_sec=iter_sec,
            power_w=power,
            freq_mhz=freq,
            throttled=throttled
        ))
    
    # Sweep results
    sweep_results = []
    for dc in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        # Throughput drops linearly with duty cycle (percent pause)
        # Matches BENCHMARK_RESULTS.md: 0% = ~2.8M, 50% = ~1.4M
        max_throughput = 2850000
        # Slight efficiency gain at intermediate DCs due to higher freq, but mostly linear
        efficiency_factor = 1.0 + (0.05 * np.sin(dc * np.pi)) 
        base = max_throughput * (1.0 - dc) * efficiency_factor
        
        sweep_results.append(SweepResult(
            duty_cycle=dc,
            mean_iter_per_sec=base,
            stddev=base * 0.01,
            ci95=base * 0.02,
            effective_gflops=base * 16000 / 1e9
        ))
    
    return time_samples, sweep_results


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fornax Enhanced Visualization Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                           Generate demo plots
  %(prog)s --sweep results.csv --output sweep.png
  %(prog)s --time-series data.json --output timeseries.png
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                        help='Generate demo plots with sample data')
    parser.add_argument('--sweep', type=str,
                        help='Path to sweep CSV file')
    parser.add_argument('--output', type=str,
                        help='Output file path (shows plot if not specified)')
    parser.add_argument('--ascii', action='store_true',
                        help='Force ASCII output')
    
    args = parser.parse_args()
    
    if args.ascii:
        global HAS_MATPLOTLIB
        HAS_MATPLOTLIB = False
    
    if args.demo:
        print("Generating demo visualizations...")
        time_samples, sweep_results = generate_sample_data()
        
        if HAS_MATPLOTLIB:
            plot_time_series(time_samples, 'demo_timeseries.png')
            plot_phase_diagram(time_samples, 'demo_phase.png')
            plot_sweep_results(sweep_results, 'demo_sweep.png')
            
            # Generate sample latency data
            if HAS_NUMPY:
                latencies = np.abs(np.random.normal(10, 3, 1000))
                plot_latency_cdf(latencies.tolist(), 'demo_latency_cdf.png')
            
            print("\nGenerated:")
            print("  - demo_timeseries.png")
            print("  - demo_phase.png")
            print("  - demo_sweep.png")
            print("  - demo_latency_cdf.png")
        else:
            print(ascii_time_series(time_samples))
            print(ascii_sweep_results(sweep_results))
        
        return
    
    if args.sweep:
        results = load_sweep_csv(args.sweep)
        plot_sweep_results(results, args.output)
        return
    
    # No args - show help
    parser.print_help()


if __name__ == '__main__':
    main()
