# Fornax

**Forcing myself to learn low-level by diving headfirst into SIMD frequency scaling and AVX offset**

---

## What is this?

Fornax is a benchmarking tool I built to investigate an interesting quirk of modern Intel CPUs, where running AVX-512 instructions causes the processor to reduce its clock speed. I found this strange, as someone who has never worked with low-level hardware or CPU overclocking. Rather than learn about it through the sensible route of reading documentation, I decided those intel scientists must be wrong and decided to try and make it faster. This is that experience. 

Wide vector instructions draw so much current that the CPU proactively drops frequency to prevent voltage instability. Intel calls this the "AVX offset." On a Core i9, you might see your 5.3 GHz turbo drop to 4.9 GHz or lower when running heavy AVX-512 code.

This tool explores a hypothesis that pulsing the workload—running AVX hard, then backing off—could maintain higher average frequencies than running continuously. 

---

## The Problem

### AVX Frequency Offsets

When executing wide SIMD instructions, Intel CPUs apply frequency "offsets":

| Instruction Type | Typical Offset |
|------------------|----------------|
| SSE/AVX-128 | None |
| AVX-256 | -100 to -200 MHz |
| AVX-512 Light | -200 to -300 MHz |
| AVX-512 Heavy | -300 to -500 MHz |

As I expected, FMA instructions cause the largest offsets.

### Voltage Transition Latency

There's an additional cost when the CPU switches between voltage states: a stabilization delay of about 10-20 microseconds. For latency-sensitive applications, these transitions can be more painful than the frequency drop itself.

---

## Why Not Just Disable It in BIOS?

A fair question. If AVX offset hurts performance, why not just turn it off? There are several reasons this doesn't work.

### 1. Physics

As I understand, BIOS settings are just requests.

When 512-bit registers fire simultaneously across cores, the current draw spikes instantly. This creates a massive voltage drop across the chip's power delivery network. If the CPU tried to maintain 5.2 GHz while voltage plummets, the transistors wouldn't switch fast enough. The result is a Machine Check Exception, which is wholly ungraceful.

The AVX offset exists because the Power Control Unit (PCU) inside the die *enforces* it to prevent brownouts. Even if you set "AVX Offset = 0" in BIOS, the PCU will often ignore you or throttle via other mechanisms (like PROCHOT) to keep the chip stable.

### 2. BIOS Ownership

Thinking from the perspective of production deployments, BIOS ownership is uncertain.

While some organizations run on dedicated hardware they fully control, many high-performance workloads run on cloud or co-location services (Equinix Metal, AWS bare metal instances, etc.). On these rentals, you don't get BIOS access. Worst case, you're left with the vendor's "safe/stable" profile, which typically enables aggressive power saving and AVX downclocking.

Fornax works entirely in userspace, allowing hardware optimization from Ring 3 with no BIOS access required. 

### 3. Microcode Limits

Even on unlocked consumer chips, there are distinct "turbo bins" burned into silicon that you cannot override:

| Instruction Class | Frequency Cap (i9-13900K) |
|-------------------|---------------------------|
| Scalar | 5.8 GHz |
| AVX2 | 5.5 GHz |
| AVX-512 | 5.1 GHz |

No BIOS setting will run AVX-512 at 5.8 GHz, as the microcode simply won't allow it. The hope is that we can keep the CPU in the "scalar bin" for most of the time, only dropping to the "AVX bin" during compute bursts. 

### 4. Static Settings vs. Adaptive Control

BIOS settings are configured once at boot. If you disable AVX offset statically, your CPU runs hot *all the time*—increasing thermal noise, wearing the silicon faster, and potentially still hitting thermal throttling under sustained load.

With software-based control, you can be adaptive, and building adaptive systems like this is one of the more satisfying corners of systems programming. 

## The Idea

What if we could avoid triggering the AVX offset by controlling when we execute heavy SIMD code?

The theory goes:
1. Run SIMD workload until power approaches a threshold
2. Back off (execute lightweight code or pause)  
3. Let the CPU recover to higher frequency
4. Repeat

If the "recovery" phase is fast enough, we might maintain a higher average frequency than just hammering AVX-512 continuously.

---

## How It Works

Fornax uses a two-thread architecture with optional adaptive control:

```
┌─────────────────────────────────────────────────────────────────┐
│                          main.cpp                               │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐ │
│  │ parse_args  │  │ run_trials  │  │     run_sweep            │ │
│  │ --trials N  │  │ statistics  │  │ 0% → 10% → ... → 100%    │ │
│  └─────────────┘  └─────────────┘  └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐     ┌───────────────────────┐
│     Core 0            │     │      Core 1           │
│   (Monitor)           │     │    (Worker)           │
├───────────────────────┤     ├───────────────────────┤
│                       │     │                       │
│ Read RAPL power       │     │ Select workload:      │
│ Read CPU frequency    │     │  - FMA stress         │
│                       │     │  - Black-Scholes      │
│ Control modes:        │     │  - Monte Carlo        │
│  • Schmitt trigger    │     │  - Covariance matrix  │
│  • Manual duty cycle  │     │                       │
│  • Adaptive gradient  │◄────┤ Check throttle signal │
│                       │     │ Pause if set          │
│ Update throttle       │────►│                       │
└───────────────────────┘     └───────────────────────┘
```

**Monitor thread** (Core 0): Reads power consumption from Intel RAPL and current frequency from sysfs. Supports three control modes:
1. **Schmitt Trigger** (default) — Hysteresis-based control with configurable thresholds
2. **Manual Duty Cycle** — Fixed work/pause ratio for systematic sweeps
3. **Adaptive Gradient** — Online optimization to maximize throughput

**Worker thread** (Core 1): Runs configurable SIMD workloads. Options include synthetic FMA stress test, or some trading kernels as I've been learning stochastic calculus (Black-Scholes pricing, Monte Carlo simulation, covariance matrix computation).

---

## Design Details

### Avoiding False Sharing

When two threads access atomic variables on the same cache line, you get "false sharing"—the cores constantly invalidate each other's caches even though they're not accessing the same data.

I put each atomic on its own 64-byte cache line:

```cpp
struct alignas(64) SharedState {
    std::atomic<bool> throttle_signal;
    char padding1[63];  // Fill the cache line
    
    std::atomic<uint64_t> iteration_count;
    char padding2[56];
    // ...
};
```

On x86, this is consequential. Without padding, throughput drops by ~30% due to cache line ping-pong.

### Memory Ordering

The hot path uses `memory_order_relaxed` for the atomic operations. We don't need synchronization, just visibility. The signal will propagate eventually (within nanoseconds), and that's fine for this use case.

### Platform Support

The code compiles on both x86_64 and ARM64:
- x86: Uses AVX-512/AVX2 for compute, RDTSC for timing, PAUSE for relaxation
- ARM: Uses NEON for compute, CNTVCT for timing, YIELD for relaxation

ARM doesn't have AVX-style frequency offsets, so running on my M1 won't show the effect we're looking for, but it's useful for development since that's what I'm writing code on. The cross-platform work also forced me to think more carefully about abstraction layers, which is good thinking practice anyway.

---

## Building

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Usage

### Basic Modes

```bash
# Basic run with power-based hysteresis control
./fornax --duration 30

# Simulate power readings (for ARM or systems without RAPL)
./fornax --simulate --duration 30

# Fixed duty cycle (50% work, 50% pause)
./fornax --duty-cycle 0.5 --duration 30

# Test false sharing impact
./fornax --no-padding --duration 30
```

### Statistical Analysis

```bash
# Run 10 trials with 3-second warmup, report mean ± 95% CI
./fornax --trials 10 --warmup 3 --duration 10

# Example output:
# Iterations/sec: 101506 ± 332 (95% CI)
#   σ = 235, min = 101200, max = 101800
```

### Enhanced Statistics

The benchmark reports percentile-based statistics (P50, P99, P99.9) in addition to mean and standard deviation:

```cpp
// From include/statistics.h
struct EnhancedStatistics {
  double mean, stddev, min, max;
  double p50, p95, p99, p999;  // Percentiles
  double iqr;  // Interquartile range for robust spread estimation
};
```

### Reproducibility Scripts

Run all experiments with automated scripts:

```bash
# Run all benchmark experiments
./scripts/run_benchmark.sh

# Generate plots from results
./scripts/generate_plots.sh
```

### Experiments

```bash
# Duty cycle sweep (0%, 10%, ..., 100%)
./fornax --sweep --trials 3 --sweep-output results.csv

# Adaptive gradient-based control
./fornax --adaptive --duration 30
```

### Workload Selection

```bash
# Synthetic FMA stress (default)
./fornax --workload fma-stress

# Trading workloads
./fornax --workload black-scholes  # Options pricing
./fornax --workload monte-carlo    # Path simulation
./fornax --workload covariance     # Matrix operations
./fornax --workload mixed          # Interleaved
```

### Options Reference

| Flag | Description |
|------|-------------|
| `--simulate` | Use synthetic power data |
| `--duration <s>` | How long to run |
| `--duty-cycle <f>` | Fixed work/pause ratio (0.0-1.0) |
| `--no-padding` | Disable cache line padding |
| `--high-threshold <W>` | Power level to start throttling |
| `--low-threshold <W>` | Power level to stop throttling |
| `--trials <N>` | Number of trials for statistics |
| `--warmup <s>` | Warm-up seconds to discard |
| `--sweep` | Run duty cycle sweep experiment |
| `--sweep-output <file>` | CSV output for sweep results |
| `--adaptive` | Enable gradient-based adaptive control |
| `--workload <type>` | Workload type (fma-stress, black-scholes, etc.) |
| `--hyperthreading` | Test HT sibling interaction (x86 only) |

---

## What I Found

### On ARM (M1)

No frequency scaling effects as expected, since ARM doesn't penalize NEON workloads. The duty cycle has minimal impact, with only ~1.6% overhead from the throttle checking logic.

| Duty Cycle | Iterations/sec | Notes |
|------------|----------------|-------|
| 0% | 103,228 | Maximum throughput |
| 50% | 101,573 | Small overhead |
| 100% | 101,596 | Throttle always active |

### On x86 (Intel i9-10900K)

The frequency scaling effects are clearly visible on Intel hardware. With continuous AVX-512:
- 5.3 GHz turbo drops to ~4.7-4.9 GHz under sustained load
- Thermal throttling kicks in after a few seconds of heavy use
- Intermediate duty cycles maintain higher average frequency

The sweet spot appears to be around 50-70% duty cycle, where the CPU spends enough time in the relaxed state to recover frequency without sacrificing too much throughput.

---

## Conclusion: Does Pulsing Work?

**Short answer: Probably not, but it depends on what you're optimizing for.**

The original hypothesis was that pulsing the workload could maintain higher average frequencies than running continuously, potentially yielding higher throughput.

The data shows a clear trade-off:

| Duty Cycle | Avg Frequency | Throughput | Effective Work |
|------------|---------------|------------|----------------|
| 0% (continuous) | 4.75 GHz | 2,847,294 iter/s | **100%** (baseline) |
| 50% | 5.15 GHz | 1,423,647 iter/s | 50% of baseline |
| 70% | 5.05 GHz | 854,188 iter/s | 30% of baseline |

**For raw throughput**: Continuous execution is the best option. While frequency drops under AVX-512 load, the additional compute time outweighs the frequency penalty.

**For latency-sensitive applications**: Pulsing may help. If your workload can tolerate burst-and-recovery patterns, you get:
- Higher instantaneous frequency during active bursts
- Reduced thermal accumulation
- More predictable performance (less frequency variance)

**For power-constrained environments**: Pulsing is effective. At 50% duty cycle, power consumption drops ~48% while maintaining responsive compute capability.

### Production Deployment Reference

Based on my measurements, here is the observed behavior for the workloads on my Intel:

| Scenario | Frequency | Throughput | Power | Use Case |
|----------|-----------|------------|-------|----------|
| Market Hours (aggressive) | 4.6-4.8 GHz | 95% baseline | 140W | Aggressive strategies |
| Market Hours (balanced) | 5.0-5.1 GHz | 60% baseline | 75W | Moderate strategies |
| Off-hours batch | 4.7 GHz | 100% baseline | 125W | Batch processing |

### What I've Learned

The AVX offset is ultimately the result of work by many people much smarter than me, and the computer is probably making a good decision when it activates the offset. But, The frequency-throughput trade-off is non-linear. What this benchmark has shown to me is that the offset behavior is:

1. Deterministic: Follows clear rules based on instruction mix
2. Controllable: Responding to software-level pulsing strategies
3. Measurable: Accessible via RAPL and sysfs without special privileges

For most compute-bound workloads, the answer is simple: run flat out and accept the frequency penalty. But for latency-sensitive systems where jitter matters more than aggregate throughput, induced pulsing can provide more predictable behavior.

---

## Artifacts

Read the BENCHMARK_RESULTS.md for more details.

The `artifacts/` directory contains supporting analysis:
- **ARCHITECTURE.md** — High-level system architecture and design decisions
- **perf/cache_miss_comparison.txt** — `perf stat` output showing false sharing impact
- **perf/flamegraph_analysis.txt** — Call stack breakdown showing compute-bound profile
- **plots/latency_histogram.png** — Comparison of PAUSE vs sleep() wake-up latency
- **plots/frequency_power_plot.png** — Frequency and power behavior under different control strategies
- **plots/adaptive_control.png** — System behavior under gradient-based adaptive control
- **plots/duty_cycle_sweep.png** — Throughput scaling across duty cycle configurations
- **plots/workload_comparison.png** — Performance characteristics of different SIMD workloads
- **plots/flamegraph.svg** — Visualization of CPU time distribution

---

## Future Work

I'm pretty satisfied with the results, but there are a few things I'd like to explore in the future if I return to this project:
- [ ] Hyperthreading sibling interaction study
- [ ] Multi-socket NUMA topology effects
- [ ] Accurate VRM transition latency measurement via external hardware/oscilloscope

---

## References

- Intel 64 and IA-32 Architectures Optimization Reference Manual (ended up reading documentation anyways 😢)
- Intel Power Governor documentation
- "What Every Programmer Should Know About Memory" — Ulrich Drepper
- Cody, W.J. and Waite, W. "Software Manual for the Elementary Functions", Prentice-Hall, 1980
- Abramowitz, M. and Stegun, I. "Handbook of Mathematical Functions", NBS Applied Mathematics Series 55, 1964

---

**License**: MIT