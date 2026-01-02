# Fornax Architecture Overview

This document shows how the code is organized. The structure separates concerns cleanly: configuration in headers, inter-thread communication isolated in SharedState, and workloads decoupled from control logic.

## Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                 main.cpp                                     │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                        CLI Argument Parsing                           │   │
│  │  --trials N  --warmup S  --sweep  --adaptive  --workload <type>       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         ▼                          ▼                          ▼             │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐       │
│  │ run_benchmark│          │  run_sweep   │          │ run_single   │       │
│  │ (multi-trial)│          │ (0%→100%)    │          │    _trial    │       │
│  └──────────────┘          └──────────────┘          └──────────────┘       │
│                                    │                                         │
│                         ┌──────────┴──────────┐                             │
│                         ▼                     ▼                             │
│                ┌──────────────┐       ┌──────────────┐                      │
│                │    Monitor   │       │    Worker    │                      │
│                │   Thread     │       │   Thread     │                      │
│                │  (Core 0)    │       │  (Core 1)    │                      │
│                └──────────────┘       └──────────────┘                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

The two-thread model keeps sensor polling off the hot path. Monitor reads power/frequency, worker computes. They communicate through cache-line-padded atomics to avoid false sharing.

## Header Dependencies

```
include/
├── arch.h               # Architecture detection, cycle counting, SIMD macros
│
├── config.h             # Centralized configuration
│   ├── fornax::defaults # Named constants (replaces magic numbers)
│   ├── MonitorConfig    # Monitor thread configuration
│   ├── WorkerConfig     # Worker thread configuration  
│   ├── SweepConfig      # Sweep experiment settings
│   ├── WorkloadType     # Enum: FMA_STRESS, BLACK_SCHOLES, etc.
│   └── BenchmarkResult  # Statistical result container
│
├── shared_state.h       # Inter-thread communication
│   ├── SharedState      # Cache-line padded atomics
│   └── SharedStateNoPadding  # For false sharing demo
│
├── ring_buffer.h        # Lock-free fixed-size ring buffer
│   └── RingBuffer<T,N>  # Replaces std::deque for O(1) operations
│
├── controller.h         # Control algorithms
│   └── AdaptiveController    # Gradient-based optimization (5-step exploration)
│
├── hypothesis_test.h    # Statistical significance testing
│   └── welch_t_test()   # Welch's t-test with effect size (Cohen's d)
│
└── workloads.h          # Trading workload kernels (PROXY WORKLOADS)
    ├── run_black_scholes()   # Options pricing (transcendentals)
    ├── run_monte_carlo()     # Path simulation (RNG)
    ├── run_covariance()      # Matrix operations
    └── run_mixed_workload()  # Interleaved
│
├── simd_math.h          # Vectorized math (exp, log, norm_cdf)
│   ├── fast_exp_avx512  # AVX-512 polynomial approximation
│   └── fast_norm_cdf    # Abramowitz-Stegun approximation
```

## Control Flow

```
                              User Request
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   parse_args()      │
                        │   Config struct     │
                        └─────────────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
           ┌──────────┐     ┌──────────┐     ┌──────────┐
           │  Normal  │     │  Sweep   │     │ Adaptive │
           │ Benchmark│     │   Mode   │     │   Mode   │
           └──────────┘     └──────────┘     └──────────┘
                 │                 │                 │
                 └─────────────────┼─────────────────┘
                                   ▼
           ┌─────────────────────────────────────────────┐
           │              Thread Launch                   │
           │  ┌─────────────────┐  ┌─────────────────┐   │
           │  │ monitor_thread  │  │  worker_thread  │   │
           │  │   (Core 0)      │  │   (Core 1)      │   │
           │  └─────────────────┘  └─────────────────┘   │
           └─────────────────────────────────────────────┘
                                   │
                                   ▼
           ┌─────────────────────────────────────────────┐
           │              SharedState                     │
           │  throttle_signal ◄────────────────────────┐ │
           │  iteration_count ─────────────────────────┤ │
           │  current_power_uw ────────────────────────┤ │
           │  current_freq_khz ────────────────────────┘ │
           └─────────────────────────────────────────────┘
```

## Control Modes

### 1. Schmitt Trigger (Hysteresis)

```
Power
  ▲
  │         ┌─────────────────────────┐
  │ HIGH ───┤ Turn ON throttle        │
  │         │                         │
  │         │     (Hysteresis Band)   │
  │         │                         │
  │ LOW ────┤ Turn OFF throttle       │
  │         └─────────────────────────┘
  └─────────────────────────────────────▶ Time
```

### 2. Manual Duty Cycle

```
Throttle   ON  ███████████░░░░░░░░░░░░░░░  (e.g., 50%)
Signal    OFF                 ███████████░░░░░░░░░░░░░░░
               ◄── Period ──►
```

### 3. Adaptive Gradient Controller

```
               Exploration Phase              Exploitation Phase
        ┌──────────────────────────┐    ┌──────────────────────────┐
        │  Test 0%, 25%, 50%, 75%, │    │  Gradient descent with   │
        │  100% (5 steps total)    │ →  │  momentum to find optimal│
        │  Build throughput model  │    │                          │
        └──────────────────────────┘    └──────────────────────────┘
```

## Workloads

| Workload | Operations | Stress Profile |
|----------|------------|----------------|
| FMA Stress | `_mm512_fmadd_pd` / `vfmaq_f64` | Max power, synthetic |
| Black-Scholes | `exp`, `log`, `sqrt`, normal CDF | Transcendentals |
| Monte Carlo | XorShift64 RNG, path accumulation | Sequential, RNG-bound |
| Covariance | Matrix outer product updates | Memory + compute |
| Mixed | Interleaved above | Realistic trading |

## Statistical Analysis

```
                    run_single_trial()
                           │
                           ▼
            ┌──────────────────────────────┐
            │    Warm-up Phase (discard)   │
            │    - Thermal stabilization   │
            │    - Cache warming           │
            └──────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │    Measurement Phase         │
            │    - Per-second throughput   │
            │    - Power/frequency samples │
            └──────────────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │    compute_stats()           │
            │    - Mean, σ, min, max       │
            │    - 95% CI = 2σ/√n          │
            └──────────────────────────────┘
```

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `src/main.cpp` | ~600 | CLI, statistics, sweep logic |
| `src/monitor.cpp` | ~400 | Power sensing, control loop |
| `src/worker.cpp` | ~300 | Workload execution |
| `include/config.h` | ~190 | Centralized config + named constants |
| `include/controller.h` | ~230 | Adaptive controller (uses RingBuffer) |
| `include/ring_buffer.h` | ~250 | Lock-free O(1) ring buffer |
| `include/hypothesis_test.h` | ~260 | Welch's t-test, Cohen's d |
| `include/vrm_measurement.h` | ~210 | VRM transition latency |
| `include/workloads.h` | ~620 | Trading workloads (proxy) |
| `include/arch.h` | ~170 | Architecture abstraction |
| `include/shared_state.h` | ~190 | Inter-thread communication |
| `include/simd_math.h` | ~390 | Vectorized transcendentals (AVX-512/AVX2/NEON) |
| `visualize.py` | ~540 | Main plotting suite (supports ASCII fallback) |
| `artifacts/*.py` | ~200ea | Simulation and specific artifact generators |
