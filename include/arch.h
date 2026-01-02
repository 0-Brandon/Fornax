/**
 * @file arch.h
 * @brief Platform abstraction layer for Fornax SIMD benchmark
 * 
 * Provides architecture-agnostic interfaces for:
 * - High-precision cycle counting (RDTSC on x86, CNTVCT on ARM)
 * - CPU relaxation hints (PAUSE on x86, YIELD on ARM)
 * - SIMD intrinsic headers
 * 
 * Cross-Platform Strategy:
 * - Development target: ARM64 (Apple Silicon M1/M2 running Arch Linux ARM)
 * - Production target: x86_64 (Intel Skylake/Ice Lake)
 */

#ifndef FORNAX_ARCH_H
#define FORNAX_ARCH_H

#include <cstdint>

// ============================================================================
// Architecture Detection
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define FORNAX_ARCH_X86 1
    #define FORNAX_ARCH_ARM 0
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(__arm__)
    #define FORNAX_ARCH_X86 0
    #define FORNAX_ARCH_ARM 1
#else
    #error "Unsupported architecture: Fornax requires x86_64 or ARM64"
#endif

// ============================================================================
// SIMD Headers
// ============================================================================

#if FORNAX_ARCH_X86
    #include <immintrin.h>  // AVX, AVX2, AVX-512, SSE intrinsics
    #include <x86intrin.h>  // __rdtsc()
#elif FORNAX_ARCH_ARM
    #include <arm_neon.h>   // NEON SIMD intrinsics
#endif

namespace fornax {

// ============================================================================
// Cycle Counter
// ============================================================================

/**
 * @brief Read the CPU cycle counter for high-precision timing
 * 
 * x86: Uses RDTSC (Read Time-Stamp Counter) - returns processor cycles since reset
 * ARM: Uses CNTVCT_EL0 (Counter-timer Virtual Count) - returns virtual timer cycles
 * 
 * Note: On ARM, the counter frequency may differ from CPU frequency.
 * Use CNTFRQ_EL0 to get the frequency if needed.
 * 
 * @return Current cycle count (64-bit)
 */
inline uint64_t get_cycles() noexcept {
#if FORNAX_ARCH_X86
    // RDTSC: Read Time-Stamp Counter
    // Returns the number of cycles since processor reset
    // On modern CPUs, this is an invariant TSC (constant rate regardless of frequency)
    return __rdtsc();
#elif FORNAX_ARCH_ARM
    // CNTVCT_EL0: Counter-timer Virtual Count register
    // Provides a high-resolution monotonic counter
    // Frequency is typically lower than CPU frequency (e.g., 24MHz on M1)
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
#endif
}

/**
 * @brief Get the frequency of the cycle counter (ARM only)
 * 
 * On x86, RDTSC typically runs at the CPU's base frequency.
 * On ARM, CNTFRQ_EL0 provides the counter frequency.
 * 
 * @return Counter frequency in Hz, or 0 if unknown
 */
inline uint64_t get_counter_frequency() noexcept {
#if FORNAX_ARCH_X86
    // x86 TSC frequency varies; return 0 to indicate "use /proc/cpuinfo"
    return 0;
#elif FORNAX_ARCH_ARM
    // CNTFRQ_EL0: Counter-timer Frequency register
    uint64_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
#endif
}

// ============================================================================
// CPU Relaxation Hint
// ============================================================================

/**
 * @brief Issue a CPU relaxation hint during busy-wait loops
 * 
 * Purpose: Reduce dynamic power consumption (C_dyn = αCV²f) during spin-waits
 * without yielding to the OS scheduler. This avoids the 5-10µs wake-up latency
 * that would be incurred by calling std::this_thread::sleep_for().
 * 
 * x86 (PAUSE):
 *   - Introduces a small delay (~10-100 cycles depending on microarchitecture)
 *   - Signals to the CPU that this is a spin-wait loop
 *   - Reduces power consumption and allows hyperthreaded sibling to run
 *   - Prevents memory order violations that could cause pipeline flushes
 * 
 * ARM (YIELD):
 *   - Hint that the thread is waiting for an event
 *   - Allows the core to enter a low-power state briefly
 *   - No guaranteed timing behavior (implementation-defined)
 */
inline void cpu_relax() noexcept {
#if FORNAX_ARCH_X86
    // PAUSE instruction: optimized for spin-wait loops
    // Reduces power and improves performance of spin-locks
    _mm_pause();
#elif FORNAX_ARCH_ARM
    // YIELD instruction: hint that thread is yielding
    // May allow other threads on the same core (if SMT) to proceed
    asm volatile("yield");
#endif
}

// ============================================================================
// Architecture Info
// ============================================================================

/**
 * @brief Get a human-readable architecture name
 */
constexpr const char* get_arch_name() noexcept {
#if FORNAX_ARCH_X86
    return "x86_64";
#elif FORNAX_ARCH_ARM
    return "ARM64";
#endif
}

/**
 * @brief Get SIMD capability description
 */
constexpr const char* get_simd_name() noexcept {
#if FORNAX_ARCH_X86
    #if defined(__AVX512F__)
        return "AVX-512";
    #elif defined(__AVX2__)
        return "AVX2";
    #elif defined(__AVX__)
        return "AVX";
    #else
        return "SSE";
    #endif
#elif FORNAX_ARCH_ARM
    return "NEON";
#endif
}

} // namespace fornax

#endif // FORNAX_ARCH_H
