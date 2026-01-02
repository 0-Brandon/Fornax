/**
 * @file ring_buffer.h
 * @brief Lock-free fixed-size ring buffer for latency-sensitive control loops
 *
 * Replaces std::deque in AdaptiveController to eliminate allocation-induced
 * latency spikes in the hot control path. Provides O(1) push with automatic
 * oldest-element eviction when full.
 *
 * Design Rationale:
 * - Fixed size eliminates dynamic allocation after construction
 * - Cache-line aligned storage reduces false sharing
 * - Simple interface focused on append-only history tracking
 * - Not thread-safe (single-writer assumed for control loop)
 *
 * Performance:
 * - Push: O(1) guaranteed, no allocation
 * - Access: O(1) random access via operator[]
 * - Iteration: Forward iteration over valid elements
 */

#ifndef FORNAX_RING_BUFFER_H
#define FORNAX_RING_BUFFER_H

#include <array>
#include <cstddef>
#include <iterator>
#include <type_traits>

namespace fornax {

/**
 * @brief Fixed-size ring buffer with automatic oldest-element eviction
 *
 * @tparam T Element type (should be trivially copyable for best performance)
 * @tparam N Maximum capacity (compile-time constant)
 *
 * When the buffer is full, new elements overwrite the oldest elements.
 * This is ideal for maintaining a sliding window of recent measurements.
 */
template <typename T, std::size_t N> class RingBuffer {
  static_assert(N > 0, "RingBuffer capacity must be greater than 0");

public:
  using value_type = T;
  using size_type = std::size_t;
  using reference = T &;
  using const_reference = const T &;

  /**
   * @brief Forward iterator for traversing buffer contents
   *
   * Iterates from oldest to newest element in logical order.
   */
  class iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = T *;
    using reference = T &;

    iterator(RingBuffer *buf, size_type pos) : buf_(buf), pos_(pos) {}

    reference operator*() const { return (*buf_)[pos_]; }
    pointer operator->() const { return &(*buf_)[pos_]; }

    iterator &operator++() {
      ++pos_;
      return *this;
    }

    iterator operator++(int) {
      iterator tmp = *this;
      ++pos_;
      return tmp;
    }

    bool operator==(const iterator &other) const { return pos_ == other.pos_; }
    bool operator!=(const iterator &other) const { return pos_ != other.pos_; }

  private:
    RingBuffer *buf_;
    size_type pos_;
  };

  class const_iterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = T;
    using difference_type = std::ptrdiff_t;
    using pointer = const T *;
    using reference = const T &;

    const_iterator(const RingBuffer *buf, size_type pos)
        : buf_(buf), pos_(pos) {}

    reference operator*() const { return (*buf_)[pos_]; }
    pointer operator->() const { return &(*buf_)[pos_]; }

    const_iterator &operator++() {
      ++pos_;
      return *this;
    }

    const_iterator operator++(int) {
      const_iterator tmp = *this;
      ++pos_;
      return tmp;
    }

    bool operator==(const const_iterator &other) const {
      return pos_ == other.pos_;
    }
    bool operator!=(const const_iterator &other) const {
      return pos_ != other.pos_;
    }

  private:
    const RingBuffer *buf_;
    size_type pos_;
  };

  // ========================================================================
  // Constructors
  // ========================================================================

  RingBuffer() noexcept = default;

  // Non-copyable, non-movable (for simplicity and cache alignment)
  RingBuffer(const RingBuffer &) = delete;
  RingBuffer &operator=(const RingBuffer &) = delete;
  RingBuffer(RingBuffer &&) = delete;
  RingBuffer &operator=(RingBuffer &&) = delete;

  // ========================================================================
  // Capacity
  // ========================================================================

  /**
   * @brief Returns the number of elements currently in the buffer
   */
  [[nodiscard]] constexpr size_type size() const noexcept { return size_; }

  /**
   * @brief Returns the maximum capacity of the buffer
   */
  [[nodiscard]] static constexpr size_type capacity() noexcept { return N; }

  /**
   * @brief Returns true if the buffer is empty
   */
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  /**
   * @brief Returns true if the buffer is at capacity
   */
  [[nodiscard]] constexpr bool full() const noexcept { return size_ == N; }

  // ========================================================================
  // Element Access
  // ========================================================================

  /**
   * @brief Access element at logical index (0 = oldest, size-1 = newest)
   *
   * @param idx Logical index into the buffer
   * @return Reference to the element
   *
   * Precondition: idx < size()
   */
  [[nodiscard]] reference operator[](size_type idx) noexcept {
    return buffer_[(head_ + idx) % N];
  }

  [[nodiscard]] const_reference operator[](size_type idx) const noexcept {
    return buffer_[(head_ + idx) % N];
  }

  /**
   * @brief Access the oldest element
   *
   * Precondition: !empty()
   */
  [[nodiscard]] reference front() noexcept { return buffer_[head_]; }
  [[nodiscard]] const_reference front() const noexcept {
    return buffer_[head_];
  }

  /**
   * @brief Access the newest element
   *
   * Precondition: !empty()
   */
  [[nodiscard]] reference back() noexcept {
    return buffer_[(head_ + size_ - 1) % N];
  }
  [[nodiscard]] const_reference back() const noexcept {
    return buffer_[(head_ + size_ - 1) % N];
  }

  // ========================================================================
  // Modifiers
  // ========================================================================

  /**
   * @brief Add an element to the end of the buffer
   *
   * If the buffer is full, the oldest element is overwritten.
   * This is O(1) with no dynamic allocation.
   *
   * @param value Element to add
   */
  void
  push_back(const T &value) noexcept(std::is_nothrow_copy_assignable_v<T>) {
    if (size_ < N) {
      // Buffer not full: append at tail
      buffer_[(head_ + size_) % N] = value;
      ++size_;
    } else {
      // Buffer full: overwrite oldest element, advance head
      buffer_[head_] = value;
      head_ = (head_ + 1) % N;
      // size_ stays at N
    }
  }

  /**
   * @brief Add an element to the end of the buffer (move version)
   */
  void push_back(T &&value) noexcept(std::is_nothrow_move_assignable_v<T>) {
    if (size_ < N) {
      buffer_[(head_ + size_) % N] = std::move(value);
      ++size_;
    } else {
      buffer_[head_] = std::move(value);
      head_ = (head_ + 1) % N;
    }
  }

  /**
   * @brief Remove all elements from the buffer
   */
  void clear() noexcept {
    head_ = 0;
    size_ = 0;
  }

  // ========================================================================
  // Iterators
  // ========================================================================

  [[nodiscard]] iterator begin() noexcept { return iterator(this, 0); }
  [[nodiscard]] iterator end() noexcept { return iterator(this, size_); }
  [[nodiscard]] const_iterator begin() const noexcept {
    return const_iterator(this, 0);
  }
  [[nodiscard]] const_iterator end() const noexcept {
    return const_iterator(this, size_);
  }
  [[nodiscard]] const_iterator cbegin() const noexcept {
    return const_iterator(this, 0);
  }
  [[nodiscard]] const_iterator cend() const noexcept {
    return const_iterator(this, size_);
  }

private:
  // Cache-line aligned storage to prevent false sharing
  alignas(64) std::array<T, N> buffer_{};

  size_type head_ = 0; // Index of oldest element
  size_type size_ = 0; // Number of valid elements
};

} // namespace fornax

#endif // FORNAX_RING_BUFFER_H
