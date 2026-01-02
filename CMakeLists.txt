cmake_minimum_required(VERSION 3.16)
project(Fornax 
    VERSION 1.0.0
    DESCRIPTION "SIMD Frequency Benchmark - Measures AVX offset trade-offs"
    LANGUAGES CXX
)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type defaults to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ============================================================================
# Architecture Detection
# ============================================================================

message(STATUS "Detected processor: ${CMAKE_SYSTEM_PROCESSOR}")

# Detect architecture and set appropriate flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    message(STATUS "Target architecture: x86_64")
    set(FORNAX_ARCH "x86_64")
    
    # Check for AVX-512 support (compiler capability, not runtime)
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)
    
    if(COMPILER_SUPPORTS_AVX512)
        message(STATUS "Compiler supports AVX-512")
        set(ARCH_FLAGS "-march=native -O3 -mavx512f -mavx512dq -mfma")
    elseif(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
        message(STATUS "Compiler supports AVX2 + FMA")
        set(ARCH_FLAGS "-march=native -O3 -mavx2 -mfma")
    else()
        message(STATUS "Using baseline x86_64 optimizations")
        set(ARCH_FLAGS "-march=native -O3")
    endif()
    
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    message(STATUS "Target architecture: ARM64")
    set(FORNAX_ARCH "ARM64")
    
    # ARM NEON is always available on ARM64
    # -march=native will enable all available features
    set(ARCH_FLAGS "-march=native -O3")
    
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|armv8|arm")
    message(STATUS "Target architecture: ARM (32-bit)")
    set(FORNAX_ARCH "ARM")
    set(ARCH_FLAGS "-march=native -O3 -mfpu=neon")
    
else()
    message(WARNING "Unknown architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    message(WARNING "Using minimal optimizations")
    set(FORNAX_ARCH "Unknown")
    set(ARCH_FLAGS "-O3")
endif()

message(STATUS "Architecture flags: ${ARCH_FLAGS}")

# ============================================================================
# Main Executable
# ============================================================================

add_executable(fornax
    src/main.cpp
    src/monitor.cpp
    src/worker.cpp
    src/latency.cpp
)

target_include_directories(fornax PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# Apply architecture-specific flags
# Using separate_arguments to properly handle the flags string
separate_arguments(ARCH_FLAGS_LIST NATIVE_COMMAND "${ARCH_FLAGS}")
target_compile_options(fornax PRIVATE ${ARCH_FLAGS_LIST})

# Link pthread (required for thread affinity)
find_package(Threads REQUIRED)
target_link_libraries(fornax PRIVATE Threads::Threads)

# ============================================================================
# Compiler Warnings
# ============================================================================

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fornax PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )
endif()

# ============================================================================
# Testing (Optional)
# ============================================================================

option(BUILD_TESTING "Build the testing tree" OFF)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
    message(STATUS "Tests: Enabled")
else()
    message(STATUS "Tests: Disabled (use -DBUILD_TESTING=ON to enable)")
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS fornax 
    RUNTIME DESTINATION bin
)

# ============================================================================
# Build Info
# ============================================================================

message(STATUS "")
message(STATUS "=== Fornax Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Architecture: ${FORNAX_ARCH}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Flags: ${ARCH_FLAGS}")
message(STATUS "===================================")
message(STATUS "")
