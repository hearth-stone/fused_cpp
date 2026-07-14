#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace fused_moe_sve::elastic {

enum class Stage {
    kW13,
    kW2F32,
    kW2Bf16,
};

struct Problem {
    Stage stage = Stage::kW13;
    const uint16_t* packed_a = nullptr;
    const uint16_t* packed_b = nullptr;
    void* output = nullptr;
    int m = 0;
    int k = 0;
    int n = 0;
    int ldc = 0;
    int n_tile = 0;
};

struct EpochSpec {
    int row_begin = 0;
    int rows = 0;
    int lanes = 1;
};

struct RunResult {
    double seconds = 0.0;
    int64_t lane_tasks = 0;
};

class Executor {
public:
    Executor(int workers, int cpu_start);
    ~Executor();

    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;

    RunResult run_static(const Problem& problem, int lanes);
    RunResult run_epoch_fixed(const Problem& problem, int lanes,
                              int epoch_rows);
    RunResult run_phase_claim(const Problem& problem,
                              const std::vector<EpochSpec>& epochs);
    RunResult run_epoch_claim(const Problem& problem,
                              const std::vector<EpochSpec>& epochs);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

int runtime_n_tile();
std::vector<EpochSpec> make_epoch_plan(int m, int epoch_rows, int low_lanes,
                                       int high_lanes, double low_fraction);

}  // namespace fused_moe_sve::elastic
