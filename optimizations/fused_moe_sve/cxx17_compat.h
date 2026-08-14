// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace fused_moe_sve::support {

template <typename To, typename From>
To BitCast(const From& source) noexcept {
  static_assert(sizeof(To) == sizeof(From), "BitCast requires equal-sized types");
  static_assert(std::is_trivially_copyable_v<To>, "BitCast destination must be trivially copyable");
  static_assert(std::is_trivially_copyable_v<From>, "BitCast source must be trivially copyable");
  To destination;
  std::memcpy(&destination, &source, sizeof(destination));
  return destination;
}

struct EmptyBarrierCompletion {
  void operator()() const noexcept {}
};

// C++17 reusable phase barrier with the completion ordering required by the
// benchmark worker pools. No participant may be destroyed while it is waiting.
template <typename Completion = EmptyBarrierCompletion>
class PhaseBarrier {
 public:
  explicit PhaseBarrier(std::ptrdiff_t participants) : PhaseBarrier(participants, Completion{}) {}

  PhaseBarrier(std::ptrdiff_t participants, Completion completion)
      : participants_(participants), remaining_(participants), completion_(std::move(completion)) {
    if (participants <= 0) {
      throw std::invalid_argument("PhaseBarrier requires at least one participant");
    }
  }

  PhaseBarrier(const PhaseBarrier&) = delete;
  PhaseBarrier& operator=(const PhaseBarrier&) = delete;

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const std::size_t generation = generation_;
    --remaining_;
    if (remaining_ == 0) {
      completion_();
      remaining_ = participants_;
      ++generation_;
      lock.unlock();
      cv_.notify_all();
      return;
    }
    cv_.wait(lock, [&]() { return generation_ != generation; });
  }

 private:
  const std::ptrdiff_t participants_;
  std::ptrdiff_t remaining_;
  Completion completion_;
  std::size_t generation_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace fused_moe_sve::support
