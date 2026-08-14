#include "elastic_nsplit.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
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

struct Options {
  elastic::Stage stage = elastic::Stage::kW13;
  std::string stage_name = "w13";
  int m = 2040;
  int k = 4096;
  int n = 4096;
  int threads = 96;
  int low_threads = 0;
  int epoch_rows = 204;
  int warmup = 3;
  int iterations = 9;
  int copies = 4;
  int cpu_start = 0;
  bool check_only = false;
};

int parse_int(std::string_view text, const char* name) {
  std::string value(text);
  size_t consumed = 0;
  const long parsed = std::stol(value, &consumed);
  if (consumed != value.size() || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

int parse_nonnegative_int(std::string_view text, const char* name) {
  std::string value(text);
  size_t consumed = 0;
  const long parsed = std::stol(value, &consumed);
  if (consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
  }
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    auto value = [&](const char* name) -> std::string_view {
      if (++index >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[index];
    };
    if (arg == "--stage") {
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
    } else if (arg == "--m") {
      options.m = parse_int(value("--m"), "M");
    } else if (arg == "--k") {
      options.k = parse_int(value("--k"), "K");
    } else if (arg == "--n") {
      options.n = parse_int(value("--n"), "N");
    } else if (arg == "--threads") {
      options.threads = parse_int(value("--threads"), "threads");
    } else if (arg == "--low-threads") {
      options.low_threads = parse_int(value("--low-threads"), "low threads");
    } else if (arg == "--epoch-rows") {
      options.epoch_rows = parse_int(value("--epoch-rows"), "epoch rows");
    } else if (arg == "--warmup") {
      options.warmup = parse_int(value("--warmup"), "warmup");
    } else if (arg == "--iters") {
      options.iterations = parse_int(value("--iters"), "iterations");
    } else if (arg == "--copies") {
      options.copies = parse_int(value("--copies"), "copies");
    } else if (arg == "--cpu-start") {
      options.cpu_start = parse_nonnegative_int(value("--cpu-start"), "cpu start");
    } else if (arg == "--check") {
      options.check_only = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: bench_elastic_nsplit [options]\n"
                << "  --stage w13|w2|w2_bf16\n"
                << "  --m M --k K --n N\n"
                << "  --threads HIGH --low-threads LOW\n"
                << "  --epoch-rows ROWS --warmup N --iters N --copies N\n"
                << "  --cpu-start CPU --check\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(arg));
    }
  }
  if (options.low_threads == 0) {
    options.low_threads = std::max(1, options.threads / 2);
  }
  if (options.low_threads > options.threads) {
    throw std::invalid_argument("low threads exceed high threads");
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

struct Buffers {
  std::vector<uint16_t> a;
  std::vector<uint16_t> b;
  std::vector<uint16_t> output16;
  std::vector<float> output32;

  void* output(elastic::Stage stage) {
    return stage == elastic::Stage::kW2F32 ? static_cast<void*>(output32.data()) : static_cast<void*>(output16.data());
  }

  const void* output(elastic::Stage stage) const {
    return stage == elastic::Stage::kW2F32 ? static_cast<const void*>(output32.data())
                                           : static_cast<const void*>(output16.data());
  }
};

size_t output_elements(const Options& options) {
  if (options.stage == elastic::Stage::kW13) {
    return static_cast<size_t>(options.m) * (options.n / 2);
  }
  return static_cast<size_t>(options.m) * options.n;
}

size_t output_bytes(const Options& options) {
  const size_t element_bytes = options.stage == elastic::Stage::kW2F32 ? sizeof(float) : sizeof(uint16_t);
  return output_elements(options) * element_bytes;
}

std::vector<Buffers> allocate_buffers(const Options& options) {
  std::vector<Buffers> result;
  result.reserve(static_cast<size_t>(options.copies));
  for (int copy = 0; copy < options.copies; ++copy) {
    Buffers buffers;
    buffers.a.resize(static_cast<size_t>(options.m) * options.k);
    buffers.b.resize(static_cast<size_t>(options.k) * options.n);
    if (options.stage == elastic::Stage::kW2F32) {
      buffers.output32.resize(output_elements(options));
    } else {
      buffers.output16.resize(output_elements(options));
    }
    fill_bf16(buffers.a, 0x123456789abcdef0ull + copy * 17ull);
    fill_bf16(buffers.b, 0xfedcba9876543210ull + copy * 29ull);
    result.push_back(std::move(buffers));
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

void assert_same(const std::vector<std::byte>& reference, const Buffers& buffers, const Options& options,
                 const char* variant) {
  const auto* actual = static_cast<const std::byte*>(buffers.output(options.stage));
  if (std::memcmp(reference.data(), actual, reference.size()) == 0) {
    return;
  }
  size_t first = 0;
  while (first < reference.size() && reference[first] == actual[first]) {
    ++first;
  }
  throw std::runtime_error(std::string("correctness mismatch in ") + variant + " at output byte " +
                           std::to_string(first));
}

void run_correctness(const Options& options, elastic::Executor& executor, Buffers& buffers, int n_tile,
                     const std::vector<elastic::EpochSpec>& fixed_low,
                     const std::vector<elastic::EpochSpec>& fixed_high,
                     const std::vector<elastic::EpochSpec>& elastic_plan) {
  elastic::Problem problem = make_problem(options, buffers, n_tile);
  executor.run_static(problem, options.threads);
  std::vector<std::byte> reference(output_bytes(options));
  std::memcpy(reference.data(), buffers.output(options.stage), reference.size());

  executor.run_static(problem, options.low_threads);
  assert_same(reference, buffers, options, "static_low");
  executor.run_epoch_fixed(problem, options.threads, options.epoch_rows);
  assert_same(reference, buffers, options, "epoch_fixed_high");
  executor.run_phase_claim(problem, fixed_low);
  assert_same(reference, buffers, options, "phase_claim_low");
  executor.run_phase_claim(problem, fixed_high);
  assert_same(reference, buffers, options, "phase_claim_high");
  executor.run_phase_claim(problem, elastic_plan);
  assert_same(reference, buffers, options, "elastic_phase_low_to_high");
  executor.run_epoch_claim(problem, fixed_low);
  assert_same(reference, buffers, options, "strict_epoch_claim_low");
  executor.run_epoch_claim(problem, fixed_high);
  assert_same(reference, buffers, options, "strict_epoch_claim_high");
  executor.run_epoch_claim(problem, elastic_plan);
  assert_same(reference, buffers, options, "elastic_strict_epoch_low_to_high");
}

struct Stats {
  double median_seconds = 0.0;
  double min_seconds = 0.0;
  double mean_seconds = 0.0;
};

Stats summarize(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize empty samples");
  }
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  const double minimum = *std::min_element(values.begin(), values.end());
  std::sort(values.begin(), values.end());
  double median = values[values.size() / 2];
  if (values.size() % 2 == 0) {
    median = 0.5 * (median + values[values.size() / 2 - 1]);
  }
  return {median, minimum, mean};
}

struct Variant {
  std::string name;
  std::function<elastic::RunResult(const elastic::Problem&)> run;
  std::vector<double> samples;
  Stats stats;

  Variant(std::string variant_name, std::function<elastic::RunResult(const elastic::Problem&)> variant_run)
      : name(std::move(variant_name)), run(std::move(variant_run)) {}
};

double percent_over(double value, double reference) { return (value / reference - 1.0) * 100.0; }

double low_row_fraction(const std::vector<elastic::EpochSpec>& plan, int low_threads, int m) {
  int rows = 0;
  for (const elastic::EpochSpec& epoch : plan) {
    if (epoch.lanes == low_threads) {
      rows += epoch.rows;
    }
  }
  return static_cast<double>(rows) / m;
}

double checksum(const Buffers& buffers, const Options& options) {
  const size_t count = output_elements(options);
  const size_t stride = std::max<size_t>(1, count / 4096);
  double result = 0.0;
  if (options.stage == elastic::Stage::kW2F32) {
    for (size_t index = 0; index < count; index += stride) {
      result += buffers.output32[index];
    }
  } else {
    for (size_t index = 0; index < count; index += stride) {
      result += buffers.output16[index];
    }
  }
  return result;
}

void print_result_json(const Options& options, const Variant& variant, int n_tile, double flops) {
  const double gflops = flops / variant.stats.median_seconds / 1.0e9;
  std::cout << "RESULT_JSON {\"variant\":\"" << variant.name << "\",\"stage\":\"" << options.stage_name
            << "\",\"m\":" << options.m << ",\"k\":" << options.k << ",\"n\":" << options.n
            << ",\"threads\":" << options.threads << ",\"low_threads\":" << options.low_threads
            << ",\"epoch_rows\":" << options.epoch_rows << ",\"n_tile\":" << n_tile
            << ",\"median_ms\":" << variant.stats.median_seconds * 1.0e3
            << ",\"min_ms\":" << variant.stats.min_seconds * 1.0e3
            << ",\"mean_ms\":" << variant.stats.mean_seconds * 1.0e3 << ",\"gflops\":" << gflops << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const int n_tile = elastic::runtime_n_tile();
    if (options.m % 12 != 0 || options.epoch_rows % 12 != 0 || options.k % 8 != 0 || options.n % n_tile != 0) {
      throw std::invalid_argument("M/epoch must align to 12, K to 8, and N to runtime n_tile");
    }
    const int n_tiles = options.n / n_tile;
    if (options.threads > n_tiles) {
      throw std::invalid_argument("high thread count exceeds the number of SVE N tiles");
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "stage=" << options.stage_name << " M=" << options.m << " K=" << options.k << " N=" << options.n
              << " n_tile=" << n_tile << " lanes=" << options.low_threads << "->" << options.threads
              << " epoch_rows=" << options.epoch_rows << " copies=" << options.copies << '\n';

    std::vector<Buffers> buffers = allocate_buffers(options);
    elastic::Executor executor(options.threads, options.cpu_start);
    const auto fixed_low =
        elastic::make_epoch_plan(options.m, options.epoch_rows, options.low_threads, options.low_threads, 0.0);
    const auto fixed_high =
        elastic::make_epoch_plan(options.m, options.epoch_rows, options.threads, options.threads, 0.0);
    const auto elastic_plan =
        elastic::make_epoch_plan(options.m, options.epoch_rows, options.low_threads, options.threads, 0.5);

    run_correctness(options, executor, buffers.front(), n_tile, fixed_low, fixed_high, elastic_plan);
    std::cout << "correctness=bitwise_exact\n";
    if (options.check_only) {
      return 0;
    }

    std::vector<Variant> variants;
    variants.push_back(
        {"static_high", [&](const elastic::Problem& p) { return executor.run_static(p, options.threads); }});
    variants.push_back(
        {"static_low", [&](const elastic::Problem& p) { return executor.run_static(p, options.low_threads); }});
    variants.push_back({"epoch_fixed_high", [&](const elastic::Problem& p) {
                          return executor.run_epoch_fixed(p, options.threads, options.epoch_rows);
                        }});
    variants.push_back(
        {"phase_claim_high", [&](const elastic::Problem& p) { return executor.run_phase_claim(p, fixed_high); }});
    variants.push_back(
        {"phase_claim_low", [&](const elastic::Problem& p) { return executor.run_phase_claim(p, fixed_low); }});
    variants.push_back({"elastic_phase_low_to_high",
                        [&](const elastic::Problem& p) { return executor.run_phase_claim(p, elastic_plan); }});
    variants.push_back({"strict_epoch_claim_high",
                        [&](const elastic::Problem& p) { return executor.run_epoch_claim(p, fixed_high); }});
    variants.push_back(
        {"strict_epoch_claim_low", [&](const elastic::Problem& p) { return executor.run_epoch_claim(p, fixed_low); }});
    variants.push_back({"elastic_strict_epoch_low_to_high",
                        [&](const elastic::Problem& p) { return executor.run_epoch_claim(p, elastic_plan); }});

    size_t invocation = 0;
    auto run_round = [&](int round, bool measured) {
      const size_t count = variants.size();
      for (size_t offset = 0; offset < count; ++offset) {
        const size_t variant_index = (static_cast<size_t>(round) + offset) % count;
        Buffers& current = buffers[invocation % buffers.size()];
        ++invocation;
        elastic::Problem problem = make_problem(options, current, n_tile);
        const elastic::RunResult result = variants[variant_index].run(problem);
        if (measured) {
          variants[variant_index].samples.push_back(result.seconds);
        }
      }
    };

    for (int round = 0; round < options.warmup; ++round) {
      run_round(round, false);
    }
    for (int round = 0; round < options.iterations; ++round) {
      run_round(options.warmup + round, true);
    }

    const double flops = 2.0 * options.m * options.k * options.n;
    std::cout << "\nvariant                 median_ms    min_ms    GFLOP/s\n";
    std::cout << "------------------------------------------------------\n";
    for (Variant& variant : variants) {
      variant.stats = summarize(variant.samples);
      const double gflops = flops / variant.stats.median_seconds / 1.0e9;
      std::cout << std::left << std::setw(24) << variant.name << std::right << std::setw(10)
                << variant.stats.median_seconds * 1.0e3 << std::setw(10) << variant.stats.min_seconds * 1.0e3
                << std::setw(12) << gflops << '\n';
    }

    auto find_stats = [&](const char* name) -> const Stats& {
      const auto found =
          std::find_if(variants.begin(), variants.end(), [&](const Variant& variant) { return variant.name == name; });
      if (found == variants.end()) {
        throw std::logic_error("missing benchmark variant");
      }
      return found->stats;
    };
    const Stats& static_high = find_stats("static_high");
    const Stats& static_low = find_stats("static_low");
    const Stats& epoch_fixed_high = find_stats("epoch_fixed_high");
    const Stats& phase_high = find_stats("phase_claim_high");
    const Stats& phase_low = find_stats("phase_claim_low");
    const Stats& elastic_phase = find_stats("elastic_phase_low_to_high");
    const Stats& strict_high = find_stats("strict_epoch_claim_high");
    const Stats& strict_low = find_stats("strict_epoch_claim_low");
    const Stats& elastic_strict = find_stats("elastic_strict_epoch_low_to_high");
    const double low_fraction = low_row_fraction(elastic_plan, options.low_threads, options.m);
    const double predicted_static =
        low_fraction * static_low.median_seconds + (1.0 - low_fraction) * static_high.median_seconds;
    const double predicted_phase =
        low_fraction * phase_low.median_seconds + (1.0 - low_fraction) * phase_high.median_seconds;
    const double predicted_strict =
        low_fraction * strict_low.median_seconds + (1.0 - low_fraction) * strict_high.median_seconds;

    const double fragmentation = percent_over(epoch_fixed_high.median_seconds, static_high.median_seconds);
    const double phase_claim_overhead = percent_over(phase_high.median_seconds, epoch_fixed_high.median_seconds);
    const double strict_claim_overhead = percent_over(strict_high.median_seconds, epoch_fixed_high.median_seconds);
    const double elastic_phase_vs_static = percent_over(elastic_phase.median_seconds, predicted_static);
    const double elastic_phase_vs_phase = percent_over(elastic_phase.median_seconds, predicted_phase);
    const double elastic_strict_vs_strict = percent_over(elastic_strict.median_seconds, predicted_strict);

    std::cout << "\nfragmentation_over_static_pct=" << fragmentation
              << "\nphase_claim_over_epoch_fixed_pct=" << phase_claim_overhead
              << "\nstrict_claim_over_epoch_fixed_pct=" << strict_claim_overhead
              << "\nelastic_phase_over_piecewise_static_pct=" << elastic_phase_vs_static
              << "\nelastic_phase_over_piecewise_phase_pct=" << elastic_phase_vs_phase
              << "\nelastic_strict_over_piecewise_strict_pct=" << elastic_strict_vs_strict
              << "\nlow_row_fraction=" << low_fraction << '\n';
    for (const Variant& variant : variants) {
      print_result_json(options, variant, n_tile, flops);
    }
    std::cout << "SUMMARY_JSON {\"stage\":\"" << options.stage_name << "\",\"m\":" << options.m
              << ",\"k\":" << options.k << ",\"n\":" << options.n << ",\"threads\":" << options.threads
              << ",\"low_threads\":" << options.low_threads << ",\"epoch_rows\":" << options.epoch_rows
              << ",\"n_tile\":" << n_tile << ",\"fragmentation_pct\":" << fragmentation
              << ",\"phase_claim_over_fixed_pct\":" << phase_claim_overhead
              << ",\"strict_claim_over_fixed_pct\":" << strict_claim_overhead
              << ",\"elastic_phase_over_piecewise_static_pct\":" << elastic_phase_vs_static
              << ",\"elastic_phase_over_piecewise_phase_pct\":" << elastic_phase_vs_phase
              << ",\"elastic_strict_over_piecewise_strict_pct\":" << elastic_strict_vs_strict
              << ",\"low_row_fraction\":" << low_fraction << "}\n";
    std::cout << "checksum=" << checksum(buffers.back(), options) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
