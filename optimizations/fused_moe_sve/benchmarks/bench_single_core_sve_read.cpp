#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <arm_sve.h>
#include <sys/mman.h>

namespace {

using Clock = std::chrono::steady_clock;

__attribute__((noinline)) void sve_read_only(const void* data, std::size_t bytes) {
  const auto* ptr = static_cast<const std::uint8_t*>(data);
  const std::size_t block_bytes = 8 * svcntb();
  std::uint64_t blocks = bytes / block_bytes;
  if (blocks == 0) {
    return;
  }
  asm volatile(
      "ptrue p0.h\n"
      "1:\n"
      "ld1h {z0.h}, p0/z, [%[ptr]]\n"
      "ld1h {z1.h}, p0/z, [%[ptr], #1, mul vl]\n"
      "ld1h {z2.h}, p0/z, [%[ptr], #2, mul vl]\n"
      "ld1h {z3.h}, p0/z, [%[ptr], #3, mul vl]\n"
      "ld1h {z4.h}, p0/z, [%[ptr], #4, mul vl]\n"
      "ld1h {z5.h}, p0/z, [%[ptr], #5, mul vl]\n"
      "ld1h {z6.h}, p0/z, [%[ptr], #6, mul vl]\n"
      "ld1h {z7.h}, p0/z, [%[ptr], #7, mul vl]\n"
      "addvl %[ptr], %[ptr], #8\n"
      "subs %[blocks], %[blocks], #1\n"
      "b.ne 1b\n"
      : [ptr] "+r"(ptr), [blocks] "+r"(blocks)
      :
      : "cc", "memory", "p0", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7");
}

double elapsed_seconds(const Clock::time_point& begin) {
  return std::chrono::duration<double>(Clock::now() - begin).count();
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if ((values.size() & 1) != 0) {
    return values[middle];
  }
  return 0.5 * (values[middle - 1] + values[middle]);
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1));
  return values[index];
}

}  // namespace

int main() {
  constexpr std::size_t kChunkBytes = 12ULL * 1024 * 1024;
  constexpr std::size_t kChunks = 64;
  constexpr std::size_t kBytes = kChunkBytes * kChunks;
  constexpr int kWarmupPasses = 2;
  constexpr int kTimedPasses = 9;
  constexpr int kChunkCycles = 5;

  void* allocation = mmap(nullptr, kBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (allocation == MAP_FAILED) {
    throw std::runtime_error("failed to allocate benchmark buffer");
  }
  auto* data = static_cast<std::uint8_t*>(allocation);
  if (madvise(data, kBytes, MADV_NOHUGEPAGE) != 0) {
    throw std::runtime_error("MADV_NOHUGEPAGE failed");
  }
  std::memset(data, 1, kBytes);

  for (int pass = 0; pass < kWarmupPasses; ++pass) {
    sve_read_only(data, kBytes);
  }

  std::vector<double> full_bandwidth;
  full_bandwidth.reserve(kTimedPasses);
  for (int pass = 0; pass < kTimedPasses; ++pass) {
    const auto begin = Clock::now();
    sve_read_only(data, kBytes);
    const double seconds = elapsed_seconds(begin);
    full_bandwidth.push_back(static_cast<double>(kBytes) / seconds / 1e9);
  }

  std::vector<double> chunk_bandwidth;
  chunk_bandwidth.reserve(kChunkCycles * kChunks);
  for (int cycle = 0; cycle < kChunkCycles; ++cycle) {
    for (std::size_t index = 0; index < kChunks; ++index) {
      const std::size_t chunk = (index * 17 + static_cast<std::size_t>(cycle) * 13) % kChunks;
      const auto begin = Clock::now();
      sve_read_only(data + chunk * kChunkBytes, kChunkBytes);
      const double seconds = elapsed_seconds(begin);
      chunk_bandwidth.push_back(static_cast<double>(kChunkBytes) / seconds / 1e9);
    }
  }

  std::printf("SVE_read_only vector_bytes=%zu working_set_MiB=%zu chunk_MiB=%zu huge_pages=disabled\n", svcntb(),
              kBytes / 1024 / 1024, kChunkBytes / 1024 / 1024);
  std::printf("continuous_768MiB median=%.3f GB/s p10=%.3f p90=%.3f samples=%zu\n", median(full_bandwidth),
              percentile(full_bandwidth, 0.10), percentile(full_bandwidth, 0.90), full_bandwidth.size());
  std::printf("rotating_12MiB median=%.3f GB/s p10=%.3f p90=%.3f samples=%zu\n", median(chunk_bandwidth),
              percentile(chunk_bandwidth, 0.10), percentile(chunk_bandwidth, 0.90), chunk_bandwidth.size());

  munmap(allocation, kBytes);
  return 0;
}
