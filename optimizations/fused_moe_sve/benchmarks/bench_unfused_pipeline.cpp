#include "unfused_pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace experiment = fused_moe_sve::unfused;

namespace {

enum class VariantSelection {
  kBoth,
  kUnfused,
  kFused,
};

struct Options {
  experiment::Config config;
  int warmup = 3;
  int iterations = 10;
  VariantSelection variant = VariantSelection::kBoth;
  bool check_only = false;
  bool skip_check = false;
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
    if (argument == "--experts") {
      options.config.experts = parse_int(value("--experts"), "experts");
    } else if (argument == "--m" || argument == "--routes") {
      options.config.routes = parse_int(value("--m"), "routes");
    } else if (argument == "--h" || argument == "--hidden") {
      options.config.hidden = parse_int(value("--h"), "hidden");
    } else if (argument == "--f" || argument == "--intermediate") {
      options.config.intermediate = parse_int(value("--f"), "intermediate");
    } else if (argument == "--threads-per-expert") {
      options.config.threads_per_expert = parse_int(value("--threads-per-expert"), "threads per expert");
    } else if (argument == "--w13-ranges") {
      options.config.w13_ranges = parse_int(value("--w13-ranges"), "W13 ranges");
    } else if (argument == "--copies") {
      options.config.copies = parse_int(value("--copies"), "copies");
    } else if (argument == "--cpu-start") {
      options.config.cpu_start = parse_int(value("--cpu-start"), "CPU start", true);
    } else if (argument == "--warmup") {
      options.warmup = parse_int(value("--warmup"), "warmup", true);
    } else if (argument == "--iters") {
      options.iterations = parse_int(value("--iters"), "iterations");
    } else if (argument == "--variant") {
      const std::string_view selected = value("--variant");
      if (selected == "both") {
        options.variant = VariantSelection::kBoth;
      } else if (selected == "unfused") {
        options.variant = VariantSelection::kUnfused;
      } else if (selected == "fused") {
        options.variant = VariantSelection::kFused;
      } else {
        throw std::invalid_argument("--variant must be both, unfused, or fused");
      }
    } else if (argument == "--check") {
      options.check_only = true;
    } else if (argument == "--skip-check") {
      options.skip_check = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: bench_unfused_pipeline [options]\n"
                << "  --experts E --m ROUTES --h HIDDEN --f INTERMEDIATE\n"
                << "  --threads-per-expert T --w13-ranges R\n"
                << "  --copies N --cpu-start CPU --warmup N --iters N\n"
                << "  --variant both|unfused|fused\n"
                << "  --check | --skip-check\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }
  if (options.check_only && options.skip_check) {
    throw std::invalid_argument("--check and --skip-check cannot be used together");
  }
  return options;
}

struct Stats {
  double median = 0.0;
  double minimum = 0.0;
  double mean = 0.0;
  double p95 = 0.0;
  double p99 = 0.0;
  double maximum = 0.0;
  double coefficient_of_variation = 0.0;
};

Stats summarize(std::vector<double> values) {
  if (values.empty()) {
    throw std::invalid_argument("cannot summarize an empty sample set");
  }
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  const double minimum = *std::min_element(values.begin(), values.end());
  double squared_deviation = 0.0;
  for (double value : values) {
    const double deviation = value - mean;
    squared_deviation += deviation * deviation;
  }
  std::sort(values.begin(), values.end());
  const auto percentile = [&](double quantile) {
    const double position = quantile * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
  };
  const size_t middle = values.size() / 2;
  const double median = values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) * 0.5 : values[middle];
  const double standard_deviation = std::sqrt(squared_deviation / values.size());
  return {median, minimum, mean, percentile(0.95), percentile(0.99), values.back(), standard_deviation / mean};
}

void print_check(const experiment::ErrorMetrics& metrics) {
  std::cout << std::scientific << std::setprecision(6)
            << "correctness intermediate_rel_l2=" << metrics.intermediate_relative_l2
            << " intermediate_max_abs=" << metrics.intermediate_max_abs
            << " intermediate_mismatches=" << metrics.intermediate_mismatches
            << " output_rel_l2=" << metrics.output_relative_l2 << " output_max_abs=" << metrics.output_max_abs
            << " output_mismatches=" << metrics.output_mismatches << '/' << metrics.output_elements << '\n'
            << std::defaultfloat;
}

void require_correctness(const experiment::ErrorMetrics& metrics) {
  // Separate W1/W3 GEMMs and the interleaved W13 GEMM accumulate BF16 products
  // in a different column order, and explicit SiLU(gate) * up rounds at a
  // different FP32 boundary. With the same production FEXPA evaluator,
  // M=24/192/2040 controls remain below 0.16%/0.23% relative L2 at the
  // intermediate/output boundaries.
  constexpr double kIntermediateRelativeL2Limit = 2.0e-3;
  constexpr double kOutputRelativeL2Limit = 3.0e-3;
  if (!std::isfinite(metrics.intermediate_relative_l2) || !std::isfinite(metrics.output_relative_l2) ||
      metrics.intermediate_relative_l2 > kIntermediateRelativeL2Limit ||
      metrics.output_relative_l2 > kOutputRelativeL2Limit) {
    throw std::runtime_error("fused/unfused numerical error exceeds the experiment tolerance");
  }
}

struct Samples {
  std::vector<double> totals;
  std::vector<std::string> stage_names;
  std::vector<std::vector<double>> stages;

  void add(const experiment::RunResult& result) {
    if (stage_names.empty()) {
      stage_names = result.stage_names;
      stages.resize(stage_names.size());
    }
    if (result.stage_names != stage_names || result.stage_seconds.size() != stages.size()) {
      throw std::runtime_error("inconsistent stage result shape");
    }
    totals.push_back(result.seconds);
    for (size_t stage = 0; stage < stages.size(); ++stage) {
      stages[stage].push_back(result.stage_seconds[stage]);
    }
  }
};

void print_stage_stats(const char* variant, const Samples& samples) {
  for (size_t stage = 0; stage < samples.stage_names.size(); ++stage) {
    const Stats stats = summarize(samples.stages[stage]);
    std::cout << "stage variant=" << variant << " name=" << samples.stage_names[stage]
              << " median_max_team_ms=" << std::fixed << std::setprecision(3) << stats.median * 1.0e3
              << " min_ms=" << stats.minimum * 1.0e3 << " mean_ms=" << stats.mean * 1.0e3 << '\n';
  }
}

void print_single_variant(const char* name, const Samples& samples, double flop_count) {
  const Stats stats = summarize(samples.totals);
  const double tflops = flop_count / stats.median / 1.0e12;
  std::cout << "variant=" << name << " median_ms=" << std::fixed << std::setprecision(3) << stats.median * 1.0e3
            << " min_ms=" << stats.minimum * 1.0e3 << " mean_ms=" << stats.mean * 1.0e3
            << " effective_TFLOPS=" << tflops << '\n';
  print_stage_stats(name, samples);
  std::cout << "tail variant=" << name << " p95_ms=" << stats.p95 * 1.0e3 << " p99_ms=" << stats.p99 * 1.0e3
            << " max_ms=" << stats.maximum * 1.0e3 << " cv_pct=" << stats.coefficient_of_variation * 100.0 << '\n'
            << "RESULT_JSON {\"variant\":\"" << name << "\",\"median_ms\":" << stats.median * 1.0e3
            << ",\"p99_ms\":" << stats.p99 * 1.0e3 << ",\"tflops\":" << tflops << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto& config = options.config;
    std::cout << "initializing explicit-unfused experiment buffers and packed weights...\n";
    experiment::Experiment runner(config);
    std::cout << "config experts=" << config.experts << " routes=" << config.routes << " hidden=" << config.hidden
              << " intermediate=" << config.intermediate << " threads_per_expert=" << config.threads_per_expert
              << " workers=" << runner.workers() << " w13_ranges=" << config.w13_ranges << " n_tile=" << runner.n_tile()
              << " copies=" << config.copies << " allocated_gib=" << std::fixed << std::setprecision(3)
              << runner.allocated_bytes() / static_cast<double>(uint64_t{1} << 30) << '\n';

    if (!options.skip_check) {
      const experiment::ErrorMetrics metrics = runner.check(0);
      print_check(metrics);
      require_correctness(metrics);
    }
    if (options.check_only) {
      std::cout << "check: PASS\n";
      return 0;
    }

    for (int warmup = 0; warmup < options.warmup; ++warmup) {
      const int copy = warmup % config.copies;
      if (options.variant == VariantSelection::kUnfused) {
        runner.run(experiment::Variant::kExplicitUnfused, copy);
      } else if (options.variant == VariantSelection::kFused) {
        runner.run(experiment::Variant::kProductionFused, copy);
      } else if (warmup % 2 == 0) {
        runner.run(experiment::Variant::kExplicitUnfused, copy);
        runner.run(experiment::Variant::kProductionFused, copy);
      } else {
        runner.run(experiment::Variant::kProductionFused, copy);
        runner.run(experiment::Variant::kExplicitUnfused, copy);
      }
    }

    Samples unfused;
    Samples fused;
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      const int copy = (options.warmup + iteration) % config.copies;
      if (options.variant == VariantSelection::kUnfused) {
        unfused.add(runner.run(experiment::Variant::kExplicitUnfused, copy));
      } else if (options.variant == VariantSelection::kFused) {
        fused.add(runner.run(experiment::Variant::kProductionFused, copy));
      } else if (iteration % 2 == 0) {
        unfused.add(runner.run(experiment::Variant::kExplicitUnfused, copy));
        fused.add(runner.run(experiment::Variant::kProductionFused, copy));
      } else {
        fused.add(runner.run(experiment::Variant::kProductionFused, copy));
        unfused.add(runner.run(experiment::Variant::kExplicitUnfused, copy));
      }
    }

    if (options.variant == VariantSelection::kUnfused) {
      print_single_variant("explicit_unfused", unfused, runner.flop_count());
      return 0;
    }
    if (options.variant == VariantSelection::kFused) {
      print_single_variant("production_fused", fused, runner.flop_count());
      return 0;
    }

    const Stats unfused_stats = summarize(unfused.totals);
    const Stats fused_stats = summarize(fused.totals);
    const double flop_count = runner.flop_count();
    const double unfused_tflops = flop_count / unfused_stats.median / 1.0e12;
    const double fused_tflops = flop_count / fused_stats.median / 1.0e12;
    const double speedup = unfused_stats.median / fused_stats.median;
    std::cout << "variant                 median_ms     min_ms    mean_ms  effective_TFLOPS\n"
              << "explicit_unfused       " << std::setw(10) << std::fixed << std::setprecision(3)
              << unfused_stats.median * 1.0e3 << ' ' << std::setw(10) << unfused_stats.minimum * 1.0e3 << ' '
              << std::setw(10) << unfused_stats.mean * 1.0e3 << ' ' << std::setw(17) << std::setprecision(3)
              << unfused_tflops << '\n'
              << "production_fused       " << std::setw(10) << fused_stats.median * 1.0e3 << ' ' << std::setw(10)
              << fused_stats.minimum * 1.0e3 << ' ' << std::setw(10) << fused_stats.mean * 1.0e3 << ' ' << std::setw(17)
              << fused_tflops << '\n'
              << "speedup_fused_over_unfused=" << std::setprecision(4) << speedup
              << "x latency_reduction_pct=" << std::setprecision(2)
              << (1.0 - fused_stats.median / unfused_stats.median) * 100.0 << '\n';
    print_stage_stats("explicit_unfused", unfused);
    print_stage_stats("production_fused", fused);
    std::cout << "tail variant=explicit_unfused p95_ms=" << std::fixed << std::setprecision(3)
              << unfused_stats.p95 * 1.0e3 << " p99_ms=" << unfused_stats.p99 * 1.0e3
              << " max_ms=" << unfused_stats.maximum * 1.0e3
              << " cv_pct=" << unfused_stats.coefficient_of_variation * 100.0 << '\n'
              << "tail variant=production_fused p95_ms=" << fused_stats.p95 * 1.0e3
              << " p99_ms=" << fused_stats.p99 * 1.0e3 << " max_ms=" << fused_stats.maximum * 1.0e3
              << " cv_pct=" << fused_stats.coefficient_of_variation * 100.0 << '\n';
    std::cout << "RESULT_JSON {\"experts\":" << config.experts << ",\"routes\":" << config.routes
              << ",\"hidden\":" << config.hidden << ",\"intermediate\":" << config.intermediate
              << ",\"threads_per_expert\":" << config.threads_per_expert << ",\"workers\":" << runner.workers()
              << ",\"w13_ranges\":" << config.w13_ranges << ",\"unfused_median_ms\":" << std::setprecision(6)
              << unfused_stats.median * 1.0e3 << ",\"fused_median_ms\":" << fused_stats.median * 1.0e3
              << ",\"speedup\":" << speedup << ",\"unfused_tflops\":" << unfused_tflops
              << ",\"fused_tflops\":" << fused_tflops << ",\"unfused_p99_ms\":" << unfused_stats.p99 * 1.0e3
              << ",\"fused_p99_ms\":" << fused_stats.p99 * 1.0e3 << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
