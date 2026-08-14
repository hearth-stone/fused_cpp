// SPDX-License-Identifier: Apache-2.0
// Isolates true static MxN thread splits while retaining the production M12
// Xbyak W13 kernel and packed-B layout. This is deliberately not an e2e MoE
// benchmark: gather, inter-stage scheduling, and route merge are excluded.
#include "elastic_nsplit.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../cxx17_compat.h"

namespace elastic = fused_moe_sve::elastic;

namespace {

struct SplitPlan {
  int tm = 1;
  int tn = 1;

  int64_t threads() const { return static_cast<int64_t>(tm) * tn; }
  std::string label() const { return std::to_string(tm) + "x" + std::to_string(tn); }
};

struct Options {
  elastic::Stage stage = elastic::Stage::kW13;
  std::string stage_name = "w13";
  int m = 4080;
  int k = 4096;
  int n = 1024;
  std::vector<SplitPlan> plans = {
      {1, 64}, {1, 96}, {8, 12}, {12, 8}, {16, 6}, {24, 4}, {32, 3}, {48, 2}, {96, 1},
  };
  int w13_ranges = 1;
  int warmup = 3;
  int iterations = 11;
  int copies = 8;
  int experts = 1;
  int total_workers = 0;
  int cpu_start = 0;
  bool check_only = false;
};

int parse_int(std::string_view text, const char* name, bool allow_zero = false) {
  const std::string value(text);
  size_t consumed = 0;
  const long parsed = std::stol(value, &consumed);
  const long minimum = allow_zero ? 0 : 1;
  if (consumed != value.size() || parsed < minimum || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

SplitPlan parse_plan(std::string_view text) {
  const size_t separator = text.find_first_of("xX");
  if (separator == std::string_view::npos || separator == 0 || separator + 1 == text.size()) {
    throw std::invalid_argument("split plan must have TmxTn form: " + std::string(text));
  }
  return {parse_int(text.substr(0, separator), "Tm"), parse_int(text.substr(separator + 1), "Tn")};
}

std::vector<SplitPlan> parse_plans(std::string_view text) {
  std::vector<SplitPlan> plans;
  size_t begin = 0;
  while (begin < text.size()) {
    const size_t separator = text.find(',', begin);
    const size_t end = separator == std::string_view::npos ? text.size() : separator;
    plans.push_back(parse_plan(text.substr(begin, end - begin)));
    begin = end + 1;
  }
  if (plans.empty()) {
    throw std::invalid_argument("at least one split plan is required");
  }
  return plans;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    auto value = [&](const char* name) -> std::string_view {
      if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[index];
    };
    if (argument == "--stage") {
      options.stage_name = value("--stage");
      if (options.stage_name == "w13") {
        options.stage = elastic::Stage::kW13;
      } else if (options.stage_name == "w2") {
        options.stage = elastic::Stage::kW2F32;
      } else if (options.stage_name == "w2_bf16") {
        options.stage = elastic::Stage::kW2Bf16;
      } else {
        throw std::invalid_argument("--stage must be w13, w2, or w2_bf16");
      }
    } else if (argument == "--m") {
      options.m = parse_int(value("--m"), "M");
    } else if (argument == "--k") {
      options.k = parse_int(value("--k"), "K");
    } else if (argument == "--n") {
      options.n = parse_int(value("--n"), "N");
    } else if (argument == "--plans") {
      options.plans = parse_plans(value("--plans"));
    } else if (argument == "--w13-ranges") {
      options.w13_ranges = parse_int(value("--w13-ranges"), "W13 ranges");
    } else if (argument == "--warmup") {
      options.warmup = parse_int(value("--warmup"), "warmup", true);
    } else if (argument == "--iters") {
      options.iterations = parse_int(value("--iters"), "iterations");
    } else if (argument == "--copies") {
      options.copies = parse_int(value("--copies"), "copies");
    } else if (argument == "--experts") {
      options.experts = parse_int(value("--experts"), "experts");
    } else if (argument == "--total-workers") {
      options.total_workers = parse_int(value("--total-workers"), "total workers");
    } else if (argument == "--cpu-start") {
      options.cpu_start = parse_int(value("--cpu-start"), "CPU start", true);
    } else if (argument == "--check") {
      options.check_only = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: bench_mn_split [options]\n"
                << "  --stage w13|w2|w2_bf16 --m M --k K --n N\n"
                << "  --plans TmxTn,... --w13-ranges N\n"
                << "  --warmup N --iters N --copies N --experts E --total-workers T\n"
                << "  --cpu-start CPU --check\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  return options;
}

uint16_t float_to_bf16(float value) {
  uint32_t bits = fused_moe_sve::support::BitCast<uint32_t>(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

uint64_t next_random(uint64_t& state) {
  state ^= state << 13;
  state ^= state >> 7;
  state ^= state << 17;
  return state;
}

void fill_bf16(std::vector<uint16_t>& values, uint64_t seed) {
  uint64_t state = seed;
  for (uint16_t& value : values) {
    const int sample = static_cast<int>(next_random(state) % 33) - 16;
    value = float_to_bf16(static_cast<float>(sample) / 128.0f);
  }
}

size_t output_elements(const Options& options) {
  return static_cast<size_t>(options.m) * (options.stage == elastic::Stage::kW13 ? options.n / 2 : options.n);
}

size_t output_bytes(const Options& options) {
  return output_elements(options) * (options.stage == elastic::Stage::kW2F32 ? sizeof(float) : sizeof(uint16_t));
}

struct Buffers {
  std::vector<uint16_t> a;
  std::vector<uint16_t> b;
  std::vector<uint16_t> output16;
  std::vector<float> output32;

  void* output(elastic::Stage stage) {
    return stage == elastic::Stage::kW2F32 ? static_cast<void*>(output32.data()) : static_cast<void*>(output16.data());
  }
};

struct BufferCopy {
  std::vector<Buffers> experts;
};

std::vector<BufferCopy> allocate_buffers(const Options& options) {
  std::vector<BufferCopy> result;
  result.reserve(static_cast<size_t>(options.copies));
  for (int copy = 0; copy < options.copies; ++copy) {
    BufferCopy buffer_copy;
    buffer_copy.experts.reserve(static_cast<size_t>(options.experts));
    for (int expert = 0; expert < options.experts; ++expert) {
      Buffers buffers;
      buffers.a.resize(static_cast<size_t>(options.m) * options.k);
      buffers.b.resize(static_cast<size_t>(options.k) * options.n);
      if (options.stage == elastic::Stage::kW2F32) {
        buffers.output32.resize(output_elements(options));
      } else {
        buffers.output16.resize(output_elements(options));
      }
      const uint64_t identity = static_cast<uint64_t>(copy) * options.experts + expert;
      fill_bf16(buffers.a, 0x123456789abcdef0ull + identity * 17ull);
      fill_bf16(buffers.b, 0xfedcba9876543210ull + identity * 29ull);
      buffer_copy.experts.push_back(std::move(buffers));
    }
    result.push_back(std::move(buffer_copy));
  }
  return result;
}

elastic::Problem make_problem(const Options& options, Buffers& buffers, int n_tile) {
  elastic::Problem problem;
  problem.stage = options.stage;
  problem.packed_a = buffers.a.data();
  problem.packed_b = buffers.b.data();
  problem.output = buffers.output(options.stage);
  problem.m = options.m;
  problem.k = options.k;
  problem.n = options.n;
  problem.ldc = options.stage == elastic::Stage::kW13 ? options.n / 2 : options.n;
  problem.n_tile = n_tile;
  return problem;
}

struct Stats {
  double median = 0.0;
  double minimum = 0.0;
  double mean = 0.0;
};

Stats summarize(std::vector<double> samples) {
  const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  const double minimum = *std::min_element(samples.begin(), samples.end());
  std::sort(samples.begin(), samples.end());
  const size_t middle = samples.size() / 2;
  const double median = samples.size() % 2 == 0 ? (samples[middle - 1] + samples[middle]) * 0.5 : samples[middle];
  return {median, minimum, mean};
}

struct Variant {
  SplitPlan plan;
  std::vector<double> samples;
  Stats stats;
};

int concurrent_experts(const Options& options, const SplitPlan& plan, int total_workers) {
  return std::min<int64_t>(options.experts, total_workers / plan.threads());
}

int expert_waves(const Options& options, const SplitPlan& plan, int total_workers) {
  const int concurrent = concurrent_experts(options, plan, total_workers);
  return (options.experts + concurrent - 1) / concurrent;
}

double run_plan(const Options& options, elastic::Executor& executor, BufferCopy& buffers, const SplitPlan& plan,
                int n_tile, int total_workers) {
  const int n_ranges = options.stage == elastic::Stage::kW13 ? options.w13_ranges : 1;
  const int concurrent = concurrent_experts(options, plan, total_workers);
  double seconds = 0.0;
  for (int expert_begin = 0; expert_begin < options.experts; expert_begin += concurrent) {
    const int expert_end = std::min(options.experts, expert_begin + concurrent);
    std::vector<elastic::Problem> problems;
    problems.reserve(static_cast<size_t>(expert_end - expert_begin));
    for (int expert = expert_begin; expert < expert_end; ++expert) {
      problems.push_back(make_problem(options, buffers.experts[static_cast<size_t>(expert)], n_tile));
    }
    seconds += executor.run_static_2d_batch(problems, plan.tm, plan.tn, n_ranges).seconds;
  }
  return seconds;
}

void run_correctness(const Options& options, elastic::Executor& executor, BufferCopy& buffers, int n_tile,
                     int total_workers) {
  const SplitPlan reference_plan = options.plans.front();
  run_plan(options, executor, buffers, reference_plan, n_tile, total_workers);
  std::vector<std::vector<std::byte>> references(static_cast<size_t>(options.experts));
  for (int expert = 0; expert < options.experts; ++expert) {
    std::vector<std::byte>& reference = references[static_cast<size_t>(expert)];
    reference.resize(output_bytes(options));
    std::memcpy(reference.data(), buffers.experts[static_cast<size_t>(expert)].output(options.stage), reference.size());
  }
  for (const SplitPlan& plan : options.plans) {
    run_plan(options, executor, buffers, plan, n_tile, total_workers);
    for (int expert = 0; expert < options.experts; ++expert) {
      const std::vector<std::byte>& reference = references[static_cast<size_t>(expert)];
      const auto* actual =
          static_cast<const std::byte*>(buffers.experts[static_cast<size_t>(expert)].output(options.stage));
      if (std::memcmp(reference.data(), actual, reference.size()) != 0) {
        throw std::runtime_error("correctness mismatch for plan " + plan.label() + ", expert " +
                                 std::to_string(expert));
      }
    }
  }
}

double allocated_gib(const Options& options) {
  const uint64_t elements_per_copy = static_cast<uint64_t>(options.m) * options.k +
                                     static_cast<uint64_t>(options.k) * options.n + output_elements(options);
  const uint64_t bytes_per_copy =
      elements_per_copy * sizeof(uint16_t) +
      (options.stage == elastic::Stage::kW2F32 ? output_elements(options) * sizeof(uint16_t) : 0);
  return bytes_per_copy * options.experts * options.copies / static_cast<double>(uint64_t{1} << 30);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const int n_tile = elastic::runtime_n_tile();
    if (options.m % 12 != 0 || options.k % 8 != 0 || options.n % n_tile != 0) {
      throw std::invalid_argument("M must align to 12, K to 8, and N to the runtime SVE tile");
    }
    const int n_ranges = options.stage == elastic::Stage::kW13 ? options.w13_ranges : 1;
    if ((options.n / n_tile) % n_ranges != 0) {
      throw std::invalid_argument("W13 N tiles must divide evenly across weight ranges");
    }
    for (const SplitPlan& plan : options.plans) {
      if (plan.threads() > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("split plan has too many workers: " + plan.label());
      }
      if (plan.tm > options.m / 12) {
        throw std::invalid_argument("Tm exceeds the number of M12 panels: " + plan.label());
      }
    }
    const int max_plan_workers = static_cast<int>(
        std::max_element(options.plans.begin(), options.plans.end(), [](const SplitPlan& lhs, const SplitPlan& rhs) {
          return lhs.threads() < rhs.threads();
        })->threads());
    const int max_workers = options.total_workers == 0 ? max_plan_workers : options.total_workers;
    if (max_workers < max_plan_workers) {
      throw std::invalid_argument("total workers cannot be smaller than a per-expert split plan");
    }

    std::cout << "initializing MN split buffers...\n";
    std::vector<BufferCopy> buffers = allocate_buffers(options);
    elastic::Executor executor(max_workers, options.cpu_start);
    std::cout << "stage=" << options.stage_name << " M=" << options.m << " K=" << options.k << " N=" << options.n
              << " n_tile=" << n_tile << " w13_ranges=" << n_ranges << " workers=" << max_workers
              << " copies=" << options.copies << " experts=" << options.experts << " total_workers=" << max_workers
              << " allocated_gib=" << std::fixed << std::setprecision(3) << allocated_gib(options)
#if defined(FUSED_MOE_SVE_ELASTIC_USE_XBYAK) && FUSED_MOE_SVE_ELASTIC_USE_XBYAK
              << " w13_impl=xbyak"
#else
              << " w13_impl=static_rows"
#endif
              << '\n';

    run_correctness(options, executor, buffers.front(), n_tile, max_workers);
    std::cout << "correctness=bitwise_exact\n";
    if (options.check_only) {
      return 0;
    }

    std::vector<Variant> variants;
    variants.reserve(options.plans.size());
    for (const SplitPlan& plan : options.plans) {
      variants.push_back(Variant{plan, {}, {}});
    }
    size_t invocation = 0;
    auto run_round = [&](int round, bool measured) {
      for (size_t offset = 0; offset < variants.size(); ++offset) {
        const size_t index = (static_cast<size_t>(round) + offset) % variants.size();
        Variant& variant = variants[index];
        BufferCopy& current = buffers[invocation++ % buffers.size()];
        const double seconds = run_plan(options, executor, current, variant.plan, n_tile, max_workers);
        if (measured) {
          variant.samples.push_back(seconds);
        }
      }
    };
    for (int round = 0; round < options.warmup; ++round) {
      run_round(round, false);
    }
    for (int round = 0; round < options.iterations; ++round) {
      run_round(options.warmup + round, true);
    }

    const double flops = 2.0 * options.experts * options.m * options.k * options.n;
    std::cout << "\nplan  threads  concurrent  waves  median_ms    min_ms   mean_ms    TFLOP/s\n"
              << "--------------------------------------------------------------------------\n";
    for (Variant& variant : variants) {
      variant.stats = summarize(variant.samples);
      const double tflops = flops / variant.stats.median / 1.0e12;
      const int concurrent = concurrent_experts(options, variant.plan, max_workers);
      const int waves = expert_waves(options, variant.plan, max_workers);
      std::cout << std::setw(5) << variant.plan.label() << std::setw(9) << variant.plan.threads() << std::setw(12)
                << concurrent << std::setw(7) << waves << std::setw(11) << std::fixed << std::setprecision(3)
                << variant.stats.median * 1.0e3 << std::setw(10) << variant.stats.minimum * 1.0e3 << std::setw(10)
                << variant.stats.mean * 1.0e3 << std::setw(11) << tflops << '\n'
                << "RESULT_JSON {\"stage\":\"" << options.stage_name << "\",\"plan\":\"" << variant.plan.label()
                << "\",\"threads_per_expert\":" << variant.plan.threads() << ",\"experts\":" << options.experts
                << ",\"concurrent_experts\":" << concurrent << ",\"waves\":" << waves
                << ",\"median_ms\":" << variant.stats.median * 1.0e3 << ",\"min_ms\":" << variant.stats.minimum * 1.0e3
                << ",\"mean_ms\":" << variant.stats.mean * 1.0e3 << ",\"tflops\":" << tflops << "}\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
