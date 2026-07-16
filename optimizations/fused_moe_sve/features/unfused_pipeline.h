#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace fused_moe_sve::unfused {

enum class Variant {
  kExplicitUnfused,
  kProductionFused,
};

struct Config {
  int experts = 4;
  int routes = 2040;
  int hidden = 4096;
  int intermediate = 2048;
  int threads_per_expert = 24;
  int w13_ranges = 2;
  int copies = 2;
  int cpu_start = 0;
};

struct RunResult {
  double seconds = 0.0;
  std::vector<std::string> stage_names;
  std::vector<double> stage_seconds;
};

struct ErrorMetrics {
  double intermediate_relative_l2 = 0.0;
  double output_relative_l2 = 0.0;
  float intermediate_max_abs = 0.0f;
  float output_max_abs = 0.0f;
  int64_t intermediate_mismatches = 0;
  int64_t output_mismatches = 0;
  int64_t output_elements = 0;
};

class Experiment {
 public:
  explicit Experiment(Config config);
  ~Experiment();

  Experiment(const Experiment&) = delete;
  Experiment& operator=(const Experiment&) = delete;

  RunResult run(Variant variant, int copy);
  ErrorMetrics check(int copy);

  int n_tile() const;
  int workers() const;
  uint64_t allocated_bytes() const;
  double flop_count() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

struct FragmentedConfig {
  enum class Schedule {
    kSlot,
    kDynamic,
  };

  int teams = 24;
  int base_routes = 2040;
  int replaced_teams = 0;
  int split_factor = 1;
  int hidden = 4096;
  int intermediate = 512;
  int threads_per_team = 4;
  int w13_ranges = 2;
  int copies = 2;
  int cpu_start = 0;
  Schedule schedule = Schedule::kDynamic;
};

struct FragmentedCheckMetrics {
  int64_t intermediate_mismatches = 0;
  int64_t output_mismatches = 0;
  int64_t intermediate_elements = 0;
  int64_t output_elements = 0;
};

// Not thread-safe. Owns a pinned worker pool and preallocated benchmark data.
// The task list replaces selected base experts with multiple shorter experts
// whose routes sum to base_routes. At most `teams` experts execute at once;
// dynamic scheduling lets any free team claim the next task.
class FragmentedExperiment {
 public:
  explicit FragmentedExperiment(FragmentedConfig config);
  ~FragmentedExperiment();

  FragmentedExperiment(const FragmentedExperiment&) = delete;
  FragmentedExperiment& operator=(const FragmentedExperiment&) = delete;

  RunResult run(int copy);
  FragmentedCheckMetrics check(int copy);

  int n_tile() const;
  int workers() const;
  int task_count() const;
  int total_routes() const;
  uint64_t active_stage_bytes() const;
  uint64_t unique_weight_bytes() const;
  uint64_t allocated_bytes() const;
  double flop_count() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fused_moe_sve::unfused
