#include "workspace_pool.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace fused_cpp::workspace {
namespace {

constexpr std::size_t kHugePageBytes = 2 * 1024 * 1024;
constexpr std::size_t kDefaultInitialBytes = 128 * 1024 * 1024;

bool EnvDisabled() {
  const char* value = std::getenv("FUSED_CPP_WORKSPACE_POOL");
  if (value == nullptr) {
    return false;
  }
  return std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 || std::strcmp(value, "FALSE") == 0 ||
         std::strcmp(value, "off") == 0 || std::strcmp(value, "OFF") == 0;
}

std::size_t RoundUp(std::size_t value, std::size_t align) { return (value + align - 1) / align * align; }

std::size_t DTypeBytes(at::ScalarType dtype) {
  switch (dtype) {
    case at::kBFloat16:
    case at::kHalf:
      return 2;
    case at::kFloat:
    case at::kInt:
      return 4;
    case at::kLong:
    case at::kDouble:
      return 8;
    case at::kByte:
    case at::kChar:
    case at::kBool:
      return 1;
    default:
      TORCH_CHECK(false, "workspace pool does not support dtype ", dtype);
  }
}

std::size_t Numel(at::IntArrayRef sizes) {
  std::size_t n = 1;
  for (int64_t dim : sizes) {
    TORCH_CHECK(dim >= 0, "workspace tensor dimension must be non-negative");
    const auto u = static_cast<std::size_t>(dim);
    TORCH_CHECK(u == 0 || n <= std::numeric_limits<std::size_t>::max() / u, "workspace tensor numel overflow");
    n *= u;
  }
  return n;
}

void Prefault(void* ptr, std::size_t bytes) {
  if (ptr == nullptr || bytes == 0) {
    return;
  }
  long page = 4096;
#if defined(_SC_PAGESIZE)
  const long detected = sysconf(_SC_PAGESIZE);
  if (detected > 0) {
    page = detected;
  }
#endif
  volatile std::uint8_t* p = static_cast<volatile std::uint8_t*>(ptr);
  for (std::size_t off = 0; off < bytes; off += static_cast<std::size_t>(page)) {
    p[off] = 0;
  }
  p[bytes - 1] = 0;
}

void PreferHugePages(void* ptr, std::size_t bytes) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  (void)madvise(ptr, bytes, MADV_HUGEPAGE);
#else
  (void)ptr;
  (void)bytes;
#endif
}

void CollapseHugePages(void* ptr, std::size_t bytes) {
#if defined(__linux__) && defined(MADV_COLLAPSE)
  (void)madvise(ptr, bytes, MADV_COLLAPSE);
#else
  (void)ptr;
  (void)bytes;
#endif
}

struct Slab {
  void* ptr = nullptr;
  std::size_t capacity = 0;
  std::size_t offset = 0;

  Slab() = default;
  Slab(const Slab&) = delete;
  Slab& operator=(const Slab&) = delete;
  Slab(Slab&& other) noexcept : ptr(other.ptr), capacity(other.capacity), offset(other.offset) {
    other.ptr = nullptr;
    other.capacity = 0;
    other.offset = 0;
  }
  Slab& operator=(Slab&& other) noexcept {
    if (this != &other) {
      release();
      ptr = other.ptr;
      capacity = other.capacity;
      offset = other.offset;
      other.ptr = nullptr;
      other.capacity = 0;
      other.offset = 0;
    }
    return *this;
  }
  ~Slab() { release(); }

  void release() {
    if (ptr == nullptr) {
      return;
    }
#if defined(__unix__) || defined(__APPLE__)
    munmap(ptr, capacity);
#else
    std::free(ptr);
#endif
    ptr = nullptr;
    capacity = 0;
    offset = 0;
  }
};

class WorkspacePoolImpl {
 public:
  std::mutex mu;
  std::vector<Slab> slabs;
  std::size_t high_water = 0;

  void reset_offsets() {
    for (Slab& slab : slabs) {
      slab.offset = 0;
    }
  }

  std::size_t capacity_bytes() const {
    std::size_t total = 0;
    for (const Slab& slab : slabs) {
      total += slab.capacity;
    }
    return total;
  }

  void* alloc(std::size_t bytes, std::size_t alignment) {
    TORCH_CHECK(bytes > 0, "workspace allocation size must be positive");
    alignment = std::max<std::size_t>(alignment, 64);
    for (Slab& slab : slabs) {
      const std::size_t begin = RoundUp(slab.offset, alignment);
      if (begin <= slab.capacity && bytes <= slab.capacity - begin) {
        slab.offset = begin + bytes;
        update_high_water();
        return static_cast<std::uint8_t*>(slab.ptr) + begin;
      }
    }
    add_slab(std::max(kDefaultInitialBytes, RoundUp(bytes + alignment, kHugePageBytes)));
    return alloc(bytes, alignment);
  }

 private:
  void update_high_water() {
    std::size_t used = 0;
    for (const Slab& slab : slabs) {
      used += slab.offset;
    }
    high_water = std::max(high_water, used);
  }

  void add_slab(std::size_t bytes) {
    Slab slab;
    slab.capacity = RoundUp(bytes, kHugePageBytes);
#if defined(__unix__) || defined(__APPLE__)
    slab.ptr = mmap(nullptr, slab.capacity, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    TORCH_CHECK(slab.ptr != MAP_FAILED, "workspace mmap failed for ", slab.capacity, " bytes");
    PreferHugePages(slab.ptr, slab.capacity);
#else
    slab.ptr = std::aligned_alloc(kHugePageBytes, slab.capacity);
    TORCH_CHECK(slab.ptr != nullptr, "workspace aligned_alloc failed for ", slab.capacity, " bytes");
#endif
    Prefault(slab.ptr, slab.capacity);
    CollapseHugePages(slab.ptr, slab.capacity);
    slabs.emplace_back(std::move(slab));
  }
};

WorkspacePoolImpl& GlobalPool() {
  static WorkspacePoolImpl pool;
  return pool;
}

}  // namespace

class WorkspacePool {
 public:
  WorkspacePoolImpl& impl() { return GlobalPool(); }
};

WorkspaceLease::WorkspaceLease(WorkspacePool& pool) : pool_(pool), lock_(pool_.impl().mu) {
  pool_.impl().reset_offsets();
}

WorkspaceLease::~WorkspaceLease() { pool_.impl().reset_offsets(); }

void* WorkspaceLease::alloc_bytes(std::size_t bytes, std::size_t alignment) {
  TORCH_CHECK(!EnvDisabled(),
              "workspace raw byte allocation is disabled by "
              "FUSED_CPP_WORKSPACE_POOL");
  return pool_.impl().alloc(bytes, alignment);
}

at::Tensor WorkspaceLease::empty(at::IntArrayRef sizes, at::TensorOptions options, std::size_t alignment) {
  TORCH_CHECK(options.device().is_cpu(), "workspace tensors must be CPU tensors");
  const at::ScalarType dtype = options.dtype().toScalarType();
  const std::size_t bytes = Numel(sizes) * DTypeBytes(dtype);
  if (bytes == 0) {
    return at::empty(sizes, options);
  }
  if (EnvDisabled()) {
    return at::empty(sizes, options);
  }
  void* ptr = alloc_bytes(bytes, alignment);
  return at::from_blob(ptr, sizes, options);
}

WorkspaceLease acquire() {
  static WorkspacePool pool;
  return WorkspaceLease(pool);
}

bool enabled() { return !EnvDisabled(); }

std::size_t capacity_bytes() { return GlobalPool().capacity_bytes(); }

std::size_t high_water_bytes() { return GlobalPool().high_water; }

void reset_stats() { GlobalPool().high_water = 0; }

}  // namespace fused_cpp::workspace
