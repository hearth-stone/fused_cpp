// Single place that decides how large working buffers are backed by pages.
//
// Before this header the choice was made independently in three places with
// three environment surfaces, and most buffers were not covered at all:
//   * the MoE scratch allocator honoured FUSED_CPP_MOE_HUGETLB /
//     FUSED_CPP_MOE_HUGETLB_MB / FUSED_CPP_MOE_THP, but only for four members of
//     one scratch struct that the production async path does not use;
//   * the attention workspace pool hard-coded a 2 MiB alignment plus
//     MADV_HUGEPAGE and answered only to FUSED_CPP_WORKSPACE_POOL;
//   * packed weights were relocated to a hugetlbfs file from Python, keyed off
//     FUSED_CPP_MOE_HUGETLBFS_PATH.
// Everything else used plain std::vector or at::empty and got whatever the
// kernel happened to give it.
//
// The surface is now:
//   FUSED_CPP_PAGES=small|thp|hugetlb   page backing (default thp)
//   FUSED_CPP_PAGE_SIZE_MB=<int>        hugetlb page size in MiB (default 32)
//   FUSED_CPP_HUGETLBFS_PATH=<mount>    implies hugetlb and probes the page size
// The pre-existing variables above still work as deprecated aliases and are only
// consulted when none of the new ones are set.
//
// This is header-only on purpose: csrc builds into two shared objects, _C and
// _moe_C, and _moe_C does not link _C. Each object therefore resolves the policy
// independently from the same environment, which keeps them consistent without a
// link-time dependency. The policy latches on the first allocation because
// deallocate() recomputes the mapping length from it, so a mid-flight change
// would unmap the wrong length.

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/statfs.h>
#include <sys/vfs.h>
#endif

namespace fused_cpp {

enum class PagePolicy {
  kSmall,   // anonymous mmap plus madvise(MADV_NOHUGEPAGE): base pages only
  kThp,     // anonymous mmap rounded to 2 MiB plus madvise(MADV_HUGEPAGE)
  kHugetlb  // anonymous MAP_HUGETLB at the configured size, THP on failure
};

inline const char* page_policy_name(PagePolicy policy) {
  switch (policy) {
    case PagePolicy::kSmall:
      return "small";
    case PagePolicy::kThp:
      return "thp";
    case PagePolicy::kHugetlb:
      return "hugetlb";
  }
  return "unknown";
}

struct PageConfig {
  PagePolicy policy = PagePolicy::kThp;
  std::size_t hugetlb_bytes = std::size_t{32} << 20;
  // A request smaller than this uses THP even under kHugetlb. Without it every
  // buffer rounds up to a whole huge page: 40 scratch buffers that need 58 MiB
  // reserve 1280 MiB at a 32 MiB page size, which exhausts the pool long before
  // it helps. Defaults to one full page, so the waste per mapping stays under 2x.
  std::size_t hugetlb_min_bytes = 0;
  // Only recorded so diagnostics can report where the size came from; the
  // mapping itself is always anonymous.
  std::string hugetlbfs_path;
};

constexpr std::size_t kThpAlignBytes = std::size_t{2} << 20;

// mmap always returns page-aligned memory, but the small-page path goes
// through operator new, which only promises 16 bytes. The SVE kernels and
// at::empty both assume a cache line, so ask for one explicitly.
constexpr std::size_t kMinAlignBytes = 64;

namespace detail {

inline const char* env_or_null(const char* name) {
  const char* value = std::getenv(name);
  return (value != nullptr && value[0] != '\0') ? value : nullptr;
}

inline bool env_truthy(const char* value) { return value != nullptr && value[0] != '0'; }

inline std::size_t probe_hugetlbfs_bytes(const char* path) {
#if defined(__linux__)
  struct statfs info{};
  if (::statfs(path, &info) == 0 && info.f_bsize > 0) {
    return static_cast<std::size_t>(info.f_bsize);
  }
#else
  (void)path;
#endif
  return 0;
}

inline PageConfig resolve_page_config() {
  PageConfig config;
#if !defined(__linux__)
  config.policy = PagePolicy::kSmall;
  return config;
#else
  const char* pages = env_or_null("FUSED_CPP_PAGES");
  const char* size_mb = env_or_null("FUSED_CPP_PAGE_SIZE_MB");
  const char* path = env_or_null("FUSED_CPP_HUGETLBFS_PATH");
  if (path == nullptr) {
    path = env_or_null("FUSED_CPP_MOE_HUGETLBFS_PATH");  // deprecated alias
  }

  if (pages != nullptr) {
    if (std::strcmp(pages, "small") == 0 || std::strcmp(pages, "4k") == 0) {
      config.policy = PagePolicy::kSmall;
    } else if (std::strcmp(pages, "hugetlb") == 0) {
      config.policy = PagePolicy::kHugetlb;
    } else {
      config.policy = PagePolicy::kThp;
    }
  } else if (path != nullptr) {
    config.policy = PagePolicy::kHugetlb;
  } else if (env_truthy(env_or_null("FUSED_CPP_MOE_HUGETLB"))) {  // deprecated alias
    config.policy = PagePolicy::kHugetlb;
  } else if (const char* thp = env_or_null("FUSED_CPP_MOE_THP"); thp != nullptr && !env_truthy(thp)) {
    config.policy = PagePolicy::kSmall;  // deprecated alias: THP=0 means small
  }

  std::size_t bytes = 0;
  if (size_mb != nullptr) {
    bytes = static_cast<std::size_t>(std::atoll(size_mb)) << 20;
  } else if (const char* legacy_mb = env_or_null("FUSED_CPP_MOE_HUGETLB_MB"); legacy_mb != nullptr) {
    bytes = static_cast<std::size_t>(std::atoll(legacy_mb)) << 20;
  } else if (path != nullptr) {
    bytes = probe_hugetlbfs_bytes(path);
    if (bytes != 0) config.hugetlbfs_path = path;
  }
  // Must be a power of two: MAP_HUGE_SHIFT encodes log2 of the page size.
  if (bytes == 0 || (bytes & (bytes - 1)) != 0) bytes = std::size_t{32} << 20;
  config.hugetlb_bytes = bytes;
  if (const char* min_kb = env_or_null("FUSED_CPP_PAGE_MIN_KB"); min_kb != nullptr) {
    config.hugetlb_min_bytes = static_cast<std::size_t>(std::atoll(min_kb)) << 10;
  } else {
    config.hugetlb_min_bytes = bytes;
  }
  return config;
#endif
}

// Optional per-mapping record, enabled by FUSED_CPP_PAGE_TRACK=1. Aggregate
// counters cannot answer which buffer landed on which page size, and process-wide
// smaps totals are dominated by the framework's own allocations.
struct PageMapping {
  std::uintptr_t address = 0;
  std::size_t request = 0;
  std::size_t length = 0;
  bool hugetlb = false;
};

struct PageState {
  PageConfig config = resolve_page_config();
  bool track = env_or_null("FUSED_CPP_PAGE_TRACK") != nullptr;
  std::mutex registry_mutex;
  std::vector<PageMapping> registry;
  std::atomic<bool> latched{false};
  std::atomic<std::size_t> live_bytes{0};
  std::atomic<std::size_t> live_mappings{0};
  std::atomic<std::size_t> peak_bytes{0};
  std::atomic<std::size_t> total_allocations{0};
  std::atomic<std::size_t> hugetlb_fallbacks{0};
};

inline PageState& page_state() {
  static PageState state;
  return state;
}

inline std::size_t round_up_pow2(std::size_t value, std::size_t multiple) {
  return (value + multiple - 1) & ~(multiple - 1);
}

}  // namespace detail

// Resolved once from the environment. Latches on the first allocation.
inline const PageConfig& page_config() { return detail::page_state().config; }

inline bool page_config_latched() { return detail::page_state().latched.load(std::memory_order_acquire); }

// Overrides the resolved policy. Only legal before the first allocation, and
// returns false otherwise so callers can report rather than corrupt.
inline bool set_page_config(const PageConfig& config) {
  detail::PageState& state = detail::page_state();
  if (state.latched.load(std::memory_order_acquire)) return false;
  state.config = config;
  return true;
}

struct PageStats {
  std::size_t live_bytes;
  std::size_t live_mappings;
  std::size_t peak_bytes;
  std::size_t total_allocations;
  std::size_t hugetlb_fallbacks;
};

inline PageStats page_stats() {
  const detail::PageState& state = detail::page_state();
  return PageStats{
      state.live_bytes.load(std::memory_order_relaxed), state.live_mappings.load(std::memory_order_relaxed),
      state.peak_bytes.load(std::memory_order_relaxed), state.total_allocations.load(std::memory_order_relaxed),
      state.hugetlb_fallbacks.load(std::memory_order_relaxed)};
}

// The mapping length a request of ``bytes`` occupies. deallocate must pass the
// same ``bytes`` it allocated with, exactly as a std::vector allocator does.
// True when a request is large enough to be worth a whole huge page.
inline bool page_uses_hugetlb(std::size_t bytes) {
  const PageConfig& config = page_config();
  return config.policy == PagePolicy::kHugetlb && bytes >= config.hugetlb_min_bytes;
}

inline std::size_t page_alloc_length(std::size_t bytes) {
  if (page_uses_hugetlb(bytes)) return detail::round_up_pow2(bytes, page_config().hugetlb_bytes);
  if (page_config().policy != PagePolicy::kSmall) return detail::round_up_pow2(bytes, kThpAlignBytes);
  return detail::round_up_pow2(bytes, kMinAlignBytes);
}

namespace detail {

inline void record_mapping(PageState& state, void* pointer, std::size_t request, std::size_t length, bool hugetlb) {
  if (!state.track) return;
  const std::lock_guard<std::mutex> guard(state.registry_mutex);
  state.registry.push_back(PageMapping{reinterpret_cast<std::uintptr_t>(pointer), request, length, hugetlb});
}

inline void forget_mapping(PageState& state, void* pointer) {
  if (!state.track) return;
  const std::lock_guard<std::mutex> guard(state.registry_mutex);
  const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(pointer);
  for (std::size_t index = 0; index < state.registry.size(); ++index) {
    if (state.registry[index].address == address) {
      state.registry[index] = state.registry.back();
      state.registry.pop_back();
      return;
    }
  }
}

inline void note_allocation(PageState& state, std::size_t length) {
  state.total_allocations.fetch_add(1, std::memory_order_relaxed);
  state.live_mappings.fetch_add(1, std::memory_order_relaxed);
  const std::size_t live = state.live_bytes.fetch_add(length, std::memory_order_relaxed) + length;
  std::size_t peak = state.peak_bytes.load(std::memory_order_relaxed);
  while (live > peak && !state.peak_bytes.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
  }
}

}  // namespace detail

inline void* page_alloc(std::size_t bytes) {
  if (bytes == 0) return nullptr;
  detail::PageState& state = detail::page_state();
  state.latched.store(true, std::memory_order_release);
#if defined(__linux__)
  const PageConfig& config = state.config;
  if (page_uses_hugetlb(bytes)) {
    const std::size_t length = detail::round_up_pow2(bytes, config.hugetlb_bytes);
    const int shift = __builtin_ctzll(static_cast<unsigned long long>(config.hugetlb_bytes));
    void* mapped = ::mmap(nullptr, length, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | (shift << MAP_HUGE_SHIFT), -1, 0);
    if (mapped == MAP_FAILED) {
      // Pool exhausted. Fall back to a same-length THP mapping so the length
      // deallocate recomputes stays correct.
      mapped = ::mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (mapped == MAP_FAILED) throw std::bad_alloc();
      ::madvise(mapped, length, MADV_HUGEPAGE);
      state.hugetlb_fallbacks.fetch_add(1, std::memory_order_relaxed);
    }
    detail::note_allocation(state, length);
    detail::record_mapping(state, mapped, bytes, length, true);
    return mapped;
  }
  if (config.policy != PagePolicy::kSmall) {
    const std::size_t length = detail::round_up_pow2(bytes, kThpAlignBytes);
    void* mapped = ::mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mapped == MAP_FAILED) throw std::bad_alloc();
    ::madvise(mapped, length, MADV_HUGEPAGE);
    detail::note_allocation(state, length);
    detail::record_mapping(state, mapped, bytes, length, false);
    return mapped;
  }
#endif
  const std::size_t length = detail::round_up_pow2(bytes, kMinAlignBytes);
#if defined(__linux__)
  void* raw = ::mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (raw == MAP_FAILED) throw std::bad_alloc();
  // Without this the mapping can inherit VM_HUGEPAGE from a recycled arena, so
  // "small" would not actually mean base pages.
#if defined(MADV_NOHUGEPAGE)
  ::madvise(raw, length, MADV_NOHUGEPAGE);
#endif
#else
  void* raw = ::operator new(length, std::align_val_t{kMinAlignBytes});
#endif
  detail::note_allocation(state, length);
  detail::record_mapping(state, raw, bytes, length, false);
  return raw;
}

inline void page_free(void* pointer, std::size_t bytes) noexcept {
  if (pointer == nullptr) return;
  detail::PageState& state = detail::page_state();
  const std::size_t length = page_alloc_length(bytes);
  detail::forget_mapping(state, pointer);
  state.live_bytes.fetch_sub(length, std::memory_order_relaxed);
  state.live_mappings.fetch_sub(1, std::memory_order_relaxed);
#if defined(__linux__)
  ::munmap(pointer, length);
#else
  ::operator delete(pointer, std::align_val_t{kMinAlignBytes});
#endif
}

// Stateless STL adapter, so vectors that use it stay swappable and comparable.
template <typename T>
struct PageAllocator {
  using value_type = T;

  PageAllocator() noexcept = default;
  template <typename U>
  PageAllocator(const PageAllocator<U>&) noexcept {}

  T* allocate(std::size_t count) { return static_cast<T*>(page_alloc(count * sizeof(T))); }

  void deallocate(T* pointer, std::size_t count) noexcept { page_free(pointer, count * sizeof(T)); }

  template <typename U>
  bool operator==(const PageAllocator<U>&) const noexcept {
    return true;
  }
  template <typename U>
  bool operator!=(const PageAllocator<U>&) const noexcept {
    return false;
  }
};

// Diagnostics for benchmarks and tests: reports what was actually resolved, not
// what was requested. Benchmarks used to echo the requested environment variable
// into their payloads, which was misleading because most buffers ignored it.
inline std::map<std::string, int64_t> page_policy_counters() {
  const PageConfig& config = page_config();
  const PageStats stats = page_stats();
  return {{"policy_id", static_cast<int64_t>(config.policy)},
          {"hugetlb_bytes", static_cast<int64_t>(config.hugetlb_bytes)},
          {"hugetlb_min_bytes", static_cast<int64_t>(config.hugetlb_min_bytes)},
          {"thp_align_bytes", static_cast<int64_t>(kThpAlignBytes)},
          {"latched", static_cast<int64_t>(page_config_latched())},
          {"live_bytes", static_cast<int64_t>(stats.live_bytes)},
          {"live_mappings", static_cast<int64_t>(stats.live_mappings)},
          {"peak_bytes", static_cast<int64_t>(stats.peak_bytes)},
          {"total_allocations", static_cast<int64_t>(stats.total_allocations)},
          {"hugetlb_fallbacks", static_cast<int64_t>(stats.hugetlb_fallbacks)}};
}

// Live mappings as (address, request, length, hugetlb) tuples. Empty unless
// FUSED_CPP_PAGE_TRACK=1 was set before the first allocation.
inline std::vector<std::array<std::int64_t, 4>> page_mappings() {
  detail::PageState& state = detail::page_state();
  if (!state.track) return {};
  const std::lock_guard<std::mutex> guard(state.registry_mutex);
  std::vector<std::array<std::int64_t, 4>> rows;
  rows.reserve(state.registry.size());
  for (const detail::PageMapping& mapping : state.registry) {
    rows.push_back({static_cast<std::int64_t>(mapping.address), static_cast<std::int64_t>(mapping.request),
                    static_cast<std::int64_t>(mapping.length), static_cast<std::int64_t>(mapping.hugetlb)});
  }
  return rows;
}

inline std::map<std::string, std::string> page_policy_strings() {
  const PageConfig& config = page_config();
  return {{"policy", page_policy_name(config.policy)}, {"hugetlbfs_path", config.hugetlbfs_path}};
}

}  // namespace fused_cpp
