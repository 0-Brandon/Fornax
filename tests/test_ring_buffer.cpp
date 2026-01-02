/**
 * @file test_ring_buffer.cpp
 * @brief Unit tests for lock-free ring buffer implementation
 *
 * Verifies:
 * - Push/pop correctness
 * - Overflow behavior (oldest elements evicted)
 * - Iterator functionality
 * - Edge cases
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "ring_buffer.h"

using namespace fornax;
using Catch::Matchers::WithinAbs;

// ============================================================================
// Basic Operations
// ============================================================================

TEST_CASE("RingBuffer initialization", "[ring_buffer]") {
  SECTION("Empty on construction") {
    RingBuffer<int, 10> buffer;
    REQUIRE(buffer.empty());
    REQUIRE(buffer.size() == 0);
    REQUIRE(buffer.capacity() == 10);
    REQUIRE_FALSE(buffer.full());
  }
}

TEST_CASE("RingBuffer push_back", "[ring_buffer]") {
  SECTION("Single element") {
    RingBuffer<int, 10> buffer;
    buffer.push_back(42);

    REQUIRE(buffer.size() == 1);
    REQUIRE_FALSE(buffer.empty());
    REQUIRE(buffer.front() == 42);
    REQUIRE(buffer.back() == 42);
  }

  SECTION("Multiple elements") {
    RingBuffer<int, 10> buffer;
    for (int i = 0; i < 5; ++i) {
      buffer.push_back(i);
    }

    REQUIRE(buffer.size() == 5);
    REQUIRE(buffer.front() == 0);
    REQUIRE(buffer.back() == 4);
  }

  SECTION("Fill to capacity") {
    RingBuffer<int, 5> buffer;
    for (int i = 0; i < 5; ++i) {
      buffer.push_back(i);
    }

    REQUIRE(buffer.full());
    REQUIRE(buffer.size() == 5);
  }
}

// ============================================================================
// Overflow Behavior
// ============================================================================

TEST_CASE("RingBuffer overflow eviction", "[ring_buffer]") {
  SECTION("Oldest element evicted on overflow") {
    RingBuffer<int, 3> buffer;
    buffer.push_back(1);
    buffer.push_back(2);
    buffer.push_back(3);

    REQUIRE(buffer.full());
    REQUIRE(buffer.front() == 1);

    // This should evict 1
    buffer.push_back(4);

    REQUIRE(buffer.full());
    REQUIRE(buffer.size() == 3);
    REQUIRE(buffer.front() == 2);
    REQUIRE(buffer.back() == 4);
  }

  SECTION("Multiple overflows maintain FIFO order") {
    RingBuffer<int, 3> buffer;
    for (int i = 0; i < 10; ++i) {
      buffer.push_back(i);
    }

    // Should contain 7, 8, 9
    REQUIRE(buffer[0] == 7);
    REQUIRE(buffer[1] == 8);
    REQUIRE(buffer[2] == 9);
  }
}

// ============================================================================
// Element Access
// ============================================================================

TEST_CASE("RingBuffer indexing", "[ring_buffer]") {
  SECTION("Logical indexing after wrap") {
    RingBuffer<int, 4> buffer;
    for (int i = 0; i < 6; ++i) {
      buffer.push_back(i);
    }

    // Buffer contains [2, 3, 4, 5] after wrapping
    REQUIRE(buffer[0] == 2);
    REQUIRE(buffer[1] == 3);
    REQUIRE(buffer[2] == 4);
    REQUIRE(buffer[3] == 5);
  }

  SECTION("Front and back after wrap") {
    RingBuffer<int, 3> buffer;
    for (int i = 0; i < 5; ++i) {
      buffer.push_back(i);
    }

    REQUIRE(buffer.front() == 2);
    REQUIRE(buffer.back() == 4);
  }
}

// ============================================================================
// Iterator Tests
// ============================================================================

TEST_CASE("RingBuffer iterators", "[ring_buffer]") {
  SECTION("Forward iteration") {
    RingBuffer<int, 5> buffer;
    for (int i = 0; i < 5; ++i) {
      buffer.push_back(i * 10);
    }

    int expected = 0;
    for (auto it = buffer.begin(); it != buffer.end(); ++it) {
      REQUIRE(*it == expected * 10);
      ++expected;
    }
    REQUIRE(expected == 5);
  }

  SECTION("Range-based for loop") {
    RingBuffer<int, 4> buffer;
    for (int i = 0; i < 4; ++i) {
      buffer.push_back(i);
    }

    int sum = 0;
    for (int val : buffer) {
      sum += val;
    }
    REQUIRE(sum == 0 + 1 + 2 + 3);
  }

  SECTION("Const iteration") {
    RingBuffer<int, 3> buffer;
    buffer.push_back(10);
    buffer.push_back(20);
    buffer.push_back(30);

    const auto &cbuffer = buffer;
    int sum = 0;
    for (auto it = cbuffer.cbegin(); it != cbuffer.cend(); ++it) {
      sum += *it;
    }
    REQUIRE(sum == 60);
  }
}

// ============================================================================
// Clear Operation
// ============================================================================

TEST_CASE("RingBuffer clear", "[ring_buffer]") {
  SECTION("Clear resets buffer") {
    RingBuffer<int, 5> buffer;
    for (int i = 0; i < 5; ++i) {
      buffer.push_back(i);
    }

    buffer.clear();

    REQUIRE(buffer.empty());
    REQUIRE(buffer.size() == 0);
    REQUIRE_FALSE(buffer.full());
  }

  SECTION("Push after clear works correctly") {
    RingBuffer<int, 3> buffer;
    buffer.push_back(1);
    buffer.push_back(2);
    buffer.clear();
    buffer.push_back(100);

    REQUIRE(buffer.size() == 1);
    REQUIRE(buffer.front() == 100);
  }
}

// ============================================================================
// Complex Types
// ============================================================================

struct TestStruct {
  double value;
  int id;
};

TEST_CASE("RingBuffer with complex types", "[ring_buffer]") {
  SECTION("Struct storage") {
    RingBuffer<TestStruct, 5> buffer;
    for (int i = 0; i < 5; ++i) {
      buffer.push_back({static_cast<double>(i) * 1.5, i});
    }

    REQUIRE_THAT(buffer[2].value, WithinAbs(3.0, 1e-10));
    REQUIRE(buffer[2].id == 2);
  }

  SECTION("Struct overflow") {
    RingBuffer<TestStruct, 2> buffer;
    buffer.push_back({1.0, 1});
    buffer.push_back({2.0, 2});
    buffer.push_back({3.0, 3});

    REQUIRE_THAT(buffer[0].value, WithinAbs(2.0, 1e-10));
    REQUIRE(buffer[0].id == 2);
  }
}
