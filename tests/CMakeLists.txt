cmake_minimum_required(VERSION 3.16)

# ============================================================================
# Test Configuration
# ============================================================================

# Fetch Catch2 testing framework
include(FetchContent)
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

# ============================================================================
# Test Executable
# ============================================================================

add_executable(fornax_tests
    test_statistics.cpp
    test_workloads.cpp
    test_controller.cpp
    test_ring_buffer.cpp
    test_integration.cpp
    test_stress.cpp
)

target_link_libraries(fornax_tests PRIVATE Catch2::Catch2WithMain)

target_include_directories(fornax_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Apply architecture-specific flags (inherit from parent CMakeLists)
if(DEFINED ARCH_FLAGS_LIST)
    target_compile_options(fornax_tests PRIVATE ${ARCH_FLAGS_LIST})
endif()

# Standard warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(fornax_tests PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# ============================================================================
# CTest Integration
# ============================================================================

include(CTest)
include(Catch)
catch_discover_tests(fornax_tests)
