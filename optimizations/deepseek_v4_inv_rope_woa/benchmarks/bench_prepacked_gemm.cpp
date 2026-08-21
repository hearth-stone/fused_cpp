// SPDX-License-Identifier: Apache-2.0

#include "moe/arm/common/nm_window_schedule.h"
#include "moe/arm/sve_bf16/jit_kernels.h"
#include "moe/arm/sve_bf16/vector_length.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace {

using Clock = std::chrono::steady_clock;
using KernelFn = fused_cpp::moe_sve::jit::KernelFn;

struct alignas(8) SveParams {
  gemm_params_t gemm{};
  int32_t kc = 0;
  int32_t packed_n = 0;
  int32_t n_begin = 0;
};

static_assert(sizeof(gemm_params_t) == 24);
static_assert(offsetof(SveParams, n_begin) == 32);

struct Panel {
  int64_t row_begin = 0;
  int64_t packed_row_begin = 0;
  int rows = 0;
  int physical_rows = 0;
  KernelFn kernel = nullptr;
};

struct Options {
  int64_t tokens = 2048;
  int64_t groups = 4;
  int64_t k = 4096;
  int64_t n = 1024;
  int threads = 80;
  int tasks_per_thread = 4;
  int warmup = 5;
  int runs = 31;
  double peak_tflops = 7.418;
  int64_t window_kib = 1024;
  bool bf16_output = true;
  bool dynamic_schedule = false;
};

int64_t parse_i64(const char* value) { return std::stoll(value); }

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    auto value = [&]() -> const char* {
      if (++index >= argc) {
        throw std::invalid_argument("missing value for " + argument);
      }
      return argv[index];
    };
    if (argument == "--tokens") {
      options.tokens = parse_i64(value());
    } else if (argument == "--groups") {
      options.groups = parse_i64(value());
    } else if (argument == "--k") {
      options.k = parse_i64(value());
    } else if (argument == "--n") {
      options.n = parse_i64(value());
    } else if (argument == "--threads") {
      options.threads = static_cast<int>(parse_i64(value()));
    } else if (argument == "--tasks-per-thread") {
      options.tasks_per_thread = static_cast<int>(parse_i64(value()));
    } else if (argument == "--warmup") {
      options.warmup = static_cast<int>(parse_i64(value()));
    } else if (argument == "--runs") {
      options.runs = static_cast<int>(parse_i64(value()));
    } else if (argument == "--peak-tflops") {
      options.peak_tflops = std::stod(value());
    } else if (argument == "--window-kib") {
      options.window_kib = parse_i64(value());
    } else if (argument == "--output") {
      const std::string output = value();
      if (output == "bf16") {
        options.bf16_output = true;
      } else if (output == "f32") {
        options.bf16_output = false;
      } else {
        throw std::invalid_argument("--output must be bf16 or f32");
      }
    } else if (argument == "--schedule") {
      const std::string schedule = value();
      if (schedule == "static") {
        options.dynamic_schedule = false;
      } else if (schedule == "dynamic") {
        options.dynamic_schedule = true;
      } else {
        throw std::invalid_argument("--schedule must be static or dynamic");
      }
    } else if (argument == "--help") {
      std::cout << "usage: bench_prepacked_gemm [--tokens M] [--groups G] [--k K] [--n N]"
                   " [--threads T] [--tasks-per-thread Q] [--output bf16|f32]"
                   " [--schedule static|dynamic]"
                   " [--window-kib KiB] [--warmup W] [--runs R] [--peak-tflops P]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (options.tokens <= 0 || options.groups <= 0 || options.k <= 0 || options.n <= 0 || options.threads <= 0 ||
      options.tasks_per_thread <= 0 || options.window_kib <= 0 || options.warmup < 0 || options.runs <= 0 ||
      options.peak_tflops <= 0.0) {
    throw std::invalid_argument("benchmark options must be positive");
  }
  if (options.k % 8 != 0 || options.n % fused_cpp::moe_sve::kNTile != 0) {
    throw std::invalid_argument("K must be divisible by 8 and N by the compiled SVE N tile");
  }
  return options;
}

std::vector<Panel> make_panels(int64_t rows, bool bf16_output) {
  std::vector<Panel> panels;
  int64_t packed_row = 0;
  for (int64_t row = 0; row < rows;) {
    const int tail = static_cast<int>(std::min<int64_t>(12, rows - row));
    const int physical_rows = tail <= 8 ? 8 : 12;
    std::string error;
    KernelFn kernel = bf16_output ? fused_cpp::moe_sve::jit::get_gemm_bf16_kernel(tail, &error)
                                  : fused_cpp::moe_sve::jit::get_gemm_f32_kernel(tail, &error);
    if (kernel == nullptr) {
      throw std::runtime_error("failed to generate M" + std::to_string(tail) + " kernel: " + error);
    }
    panels.push_back(Panel{row, packed_row, tail, physical_rows, kernel});
    row += tail;
    packed_row += physical_rows;
  }
  return panels;
}

uint16_t bf16_value(size_t index) {
  const float value = static_cast<float>(static_cast<int>(index % 251) - 125) / 256.0f;
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7fffU + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

double percentile(std::vector<double> samples, double fraction) {
  const size_t index = static_cast<size_t>(fraction * static_cast<double>(samples.size() - 1));
  std::nth_element(samples.begin(), samples.begin() + index, samples.end());
  return samples[index];
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    uint64_t runtime_vector_bytes = 0;
    asm volatile("cntb %0" : "=r"(runtime_vector_bytes));
    if (runtime_vector_bytes != fused_cpp::moe_sve::kVectorBytes) {
      throw std::runtime_error("runtime SVE vector length does not match the compiled vector length");
    }

    const std::vector<Panel> panels = make_panels(options.tokens, options.bf16_output);
    const Panel& last = panels.back();
    const int64_t physical_rows = last.packed_row_begin + last.physical_rows;
    const int64_t n_tile = fused_cpp::moe_sve::kNTile;
    const int64_t n_tiles = options.n / n_tile;
    const fused_cpp::nm_window::Geometry geometry = fused_cpp::nm_window::Choose(
        n_tiles, options.k * n_tile * static_cast<int64_t>(sizeof(uint16_t)), static_cast<int64_t>(panels.size()),
        options.threads, options.groups, options.window_kib * 1024, options.tasks_per_thread);
    const int64_t task_count = geometry.task_count(options.groups);

    std::vector<uint16_t> packed_a(static_cast<size_t>(options.groups * physical_rows * options.k));
    std::vector<uint16_t> packed_b(static_cast<size_t>(options.groups * options.k * options.n));
    std::vector<uint16_t> bf16_output;
    std::vector<float> f32_output;
    if (options.bf16_output) {
      bf16_output.resize(static_cast<size_t>(options.tokens * options.groups * options.n));
    } else {
      f32_output.resize(static_cast<size_t>(options.tokens * options.groups * options.n));
    }
    for (size_t index = 0; index < packed_a.size(); ++index) {
      packed_a[index] = bf16_value(index);
    }
    for (size_t index = 0; index < packed_b.size(); ++index) {
      packed_b[index] = bf16_value(index * 7 + 3);
    }

    omp_set_dynamic(0);
    omp_set_num_threads(options.threads);
    omp_set_schedule(options.dynamic_schedule ? omp_sched_dynamic : omp_sched_static, 1);
    auto run = [&]() {
#pragma omp parallel num_threads(options.threads)
      {
#pragma omp for schedule(runtime)
        for (int64_t task = 0; task < task_count; ++task) {
          const int64_t m_split = task % geometry.m_splits;
          const int64_t owner = task / geometry.m_splits;
          const int64_t window = owner % geometry.n_windows;
          const int64_t group = owner / geometry.n_windows;
          const int64_t tile_begin = window * n_tiles / geometry.n_windows;
          const int64_t tile_end = (window + 1) * n_tiles / geometry.n_windows;
          const int64_t n_begin = tile_begin * n_tile;
          const int64_t n_columns = (tile_end - tile_begin) * n_tile;
          const int64_t panel_begin = m_split * static_cast<int64_t>(panels.size()) / geometry.m_splits;
          const int64_t panel_end = (m_split + 1) * static_cast<int64_t>(panels.size()) / geometry.m_splits;
          const uint16_t* group_b = packed_b.data() + group * options.k * options.n;
          for (int64_t panel_index = panel_begin; panel_index < panel_end; ++panel_index) {
            const Panel& panel = panels[panel_index];
            const uint16_t* panel_a =
                packed_a.data() + group * physical_rows * options.k + panel.packed_row_begin * options.k;
            const int64_t output_offset = panel.row_begin * options.groups * options.n + group * options.n + n_begin;
            void* panel_c = options.bf16_output ? static_cast<void*>(bf16_output.data() + output_offset)
                                                : static_cast<void*>(f32_output.data() + output_offset);
            SveParams params;
            params.gemm.m = panel.rows;
            params.gemm.k = static_cast<int>(options.k);
            params.gemm.n = static_cast<int>(n_columns);
            params.gemm.lda = static_cast<int>(options.k);
            params.gemm.ldb = static_cast<int>(options.k);
            params.gemm.ldc = static_cast<int>(options.groups * options.n);
            params.kc = static_cast<int32_t>(options.k);
            params.packed_n = static_cast<int32_t>(options.n);
            params.n_begin = static_cast<int32_t>(n_begin);
            panel.kernel(panel_a, group_b, panel_c, nullptr, &params.gemm);
          }
        }
      }
    };

    for (int iteration = 0; iteration < options.warmup; ++iteration) {
      run();
    }
    std::vector<double> samples;
    samples.reserve(options.runs);
    for (int iteration = 0; iteration < options.runs; ++iteration) {
      const auto begin = Clock::now();
      run();
      samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
    }

    const double median_ms = percentile(samples, 0.5);
    const double p90_ms = percentile(samples, 0.9);
    const double flops = 2.0 * static_cast<double>(options.tokens) * static_cast<double>(options.groups) *
                         static_cast<double>(options.k) * static_cast<double>(options.n);
    const double tflops = flops / (median_ms * 1.0e9);
    uint64_t checksum = 0;
    if (options.bf16_output) {
      checksum = std::accumulate(bf16_output.begin(), bf16_output.end(), uint64_t{0});
    } else {
      for (const float value : f32_output) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        checksum += bits;
      }
    }
    std::cout << std::fixed << std::setprecision(3) << "tokens=" << options.tokens << " groups=" << options.groups
              << " k=" << options.k << " n=" << options.n << " threads=" << options.threads
              << " tasks_per_thread=" << options.tasks_per_thread
              << " output=" << (options.bf16_output ? "bf16" : "f32") << " n_windows=" << geometry.n_windows
              << " schedule=" << (options.dynamic_schedule ? "dynamic" : "static")
              << " window_kib=" << options.window_kib << " m_splits=" << geometry.m_splits << " tasks=" << task_count
              << " median_ms=" << median_ms << " p90_ms=" << p90_ms << " tflops=" << tflops
              << " peak_efficiency=" << (100.0 * tflops / options.peak_tflops) << "% checksum=" << checksum << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
}
