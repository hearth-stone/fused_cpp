#include "unfused_pipeline.h"

#include <algorithm>
#include <csignal>
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

#include <unistd.h>

namespace experiment = fused_moe_sve::unfused;

namespace {

struct Options {
  experiment::FragmentedConfig config;
  int warmup = 2;
  int iterations = 7;
  bool check_only = false;
  bool skip_check = false;
  bool stop_before_run = false;
};

struct Stats {
  double median = 0.0;
  double minimum = 0.0;
  double mean = 0.0;
  double p99 = 0.0;
  double maximum = 0.0;
  double coefficient_of_variation = 0.0;
};

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
    if (argument == "--teams") {
      options.config.teams = parse_int(value("--teams"), "teams");
    } else if (argument == "--base-routes") {
      options.config.base_routes = parse_int(value("--base-routes"), "base routes");
    } else if (argument == "--replaced-teams") {
      options.config.replaced_teams = parse_int(value("--replaced-teams"), "replaced teams", true);
    } else if (argument == "--split-factor") {
      options.config.split_factor = parse_int(value("--split-factor"), "split factor");
    } else if (argument == "--hidden") {
      options.config.hidden = parse_int(value("--hidden"), "hidden");
    } else if (argument == "--intermediate") {
      options.config.intermediate = parse_int(value("--intermediate"), "intermediate");
    } else if (argument == "--threads-per-team") {
      options.config.threads_per_team = parse_int(value("--threads-per-team"), "threads per team");
    } else if (argument == "--schedule") {
      const std::string_view schedule = value("--schedule");
      if (schedule == "dynamic") {
        options.config.schedule = experiment::FragmentedConfig::Schedule::kDynamic;
      } else if (schedule == "slot") {
        options.config.schedule = experiment::FragmentedConfig::Schedule::kSlot;
      } else {
        throw std::invalid_argument("--schedule must be dynamic or slot");
      }
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
    } else if (argument == "--check") {
      options.check_only = true;
    } else if (argument == "--skip-check") {
      options.skip_check = true;
    } else if (argument == "--stop-before-run") {
      options.stop_before_run = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: bench_fragmented_route_pipeline [options]\n"
                << "  --teams E --base-routes M --replaced-teams R --split-factor Q\n"
                << "  --hidden H --intermediate F --threads-per-team T\n"
                << "  --schedule dynamic|slot\n"
                << "  --w13-ranges R --copies N --cpu-start CPU\n"
                << "  --warmup N --iters N --check | --skip-check\n"
                << "  --stop-before-run\n";
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
  return {median, minimum, mean, percentile(0.99), values.back(), standard_deviation / mean};
}

void require_correctness(const experiment::FragmentedCheckMetrics& metrics) {
  std::cout << "correctness intermediate_mismatches=" << metrics.intermediate_mismatches << '/'
            << metrics.intermediate_elements << " output_mismatches=" << metrics.output_mismatches << '/'
            << metrics.output_elements << '\n';
  if (metrics.intermediate_mismatches != 0 || metrics.output_mismatches != 0) {
    throw std::runtime_error("fragmented parallel output differs from the single-lane reference");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const auto& config = options.config;
    const char* schedule = config.schedule == experiment::FragmentedConfig::Schedule::kDynamic ? "dynamic" : "slot";
    std::cout << "initializing fragmented-route experiment buffers and packed weights...\n";
    experiment::FragmentedExperiment runner(config);
    std::cout << "config teams=" << config.teams << " base_routes=" << config.base_routes
              << " replaced_teams=" << config.replaced_teams << " split_factor=" << config.split_factor
              << " fragment_routes=" << config.base_routes / config.split_factor << " tasks=" << runner.task_count()
              << " total_routes=" << runner.total_routes() << " hidden=" << config.hidden
              << " intermediate=" << config.intermediate << " threads_per_team=" << config.threads_per_team
              << " workers=" << runner.workers() << " schedule=" << schedule << " w13_ranges=" << config.w13_ranges
              << " n_tile=" << runner.n_tile() << " copies=" << config.copies << " active_stage_mib=" << std::fixed
              << std::setprecision(3) << runner.active_stage_bytes() / static_cast<double>(uint64_t{1} << 20)
              << " unique_weight_mib=" << runner.unique_weight_bytes() / static_cast<double>(uint64_t{1} << 20)
              << " allocated_gib=" << runner.allocated_bytes() / static_cast<double>(uint64_t{1} << 30) << '\n';

    if (!options.skip_check) {
      require_correctness(runner.check(0));
    }
    if (options.check_only) {
      std::cout << "check: PASS\n";
      return 0;
    }
    if (options.stop_before_run) {
      std::cout << "profiler_ready pid=" << getpid() << " workers=" << runner.workers() << '\n' << std::flush;
      if (std::raise(SIGSTOP) != 0) {
        throw std::runtime_error("failed to stop before the measured region");
      }
    }

    for (int warmup = 0; warmup < options.warmup; ++warmup) {
      runner.run(warmup % config.copies);
    }
    Samples samples;
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      samples.add(runner.run((options.warmup + iteration) % config.copies));
    }

    const Stats total = summarize(samples.totals);
    const double tflops = runner.flop_count() / total.median / 1.0e12;
    std::cout << "fragmented_route median_ms=" << std::fixed << std::setprecision(3) << total.median * 1.0e3
              << " min_ms=" << total.minimum * 1.0e3 << " mean_ms=" << total.mean * 1.0e3
              << " effective_TFLOPS=" << tflops << '\n';
    for (size_t stage = 0; stage < samples.stage_names.size(); ++stage) {
      const Stats stats = summarize(samples.stages[stage]);
      std::cout << "stage name=" << samples.stage_names[stage] << " median_max_team_sum_ms=" << stats.median * 1.0e3
                << " min_ms=" << stats.minimum * 1.0e3 << " mean_ms=" << stats.mean * 1.0e3 << '\n';
    }
    std::cout << "tail p99_ms=" << total.p99 * 1.0e3 << " max_ms=" << total.maximum * 1.0e3
              << " cv_pct=" << total.coefficient_of_variation * 100.0 << '\n'
              << "RESULT_JSON {\"median_ms\":" << total.median * 1.0e3 << ",\"p99_ms\":" << total.p99 * 1.0e3
              << ",\"tflops\":" << tflops << ",\"tasks\":" << runner.task_count()
              << ",\"total_routes\":" << runner.total_routes() << "}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
