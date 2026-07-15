#include "elastic_nsplit.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE) || !defined(__ARM_FEATURE_BF16)
#error "elastic_nsplit requires AArch64 SVE BF16"
#endif

#include <arm_sve.h>
#include <pthread.h>
#include <sched.h>

#include "gemm_params.h"

extern "C" {
void moe_sve_w13_silu_poly5_packc_m12_rows_opt(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*,
                                               const gemm_params_t*);
void moe_sve_w2_packed_m12(const uint16_t*, const uint16_t*, float*, uint16_t*, const gemm_params_t*);
void moe_sve_w2_packed_bf16_m12(const uint16_t*, const uint16_t*, uint16_t*, uint16_t*, const gemm_params_t*);
}

namespace fused_moe_sve::elastic {
namespace {

struct Range {
  int begin = 0;
  int size = 0;
};

Range split_tiles(int tiles, int lanes, int lane) {
  if (tiles <= 0 || lanes <= 0 || lane < 0 || lane >= lanes) {
    return {};
  }
  const int base = tiles / lanes;
  const int extra = tiles % lanes;
  if (lane < extra) {
    return {lane * (base + 1), base + 1};
  }
  return {extra * (base + 1) + (lane - extra) * base, base};
}

void check_problem(const Problem& p) {
  if (p.packed_a == nullptr || p.packed_b == nullptr || p.output == nullptr) {
    throw std::invalid_argument("null elastic GEMM buffer");
  }
  if (p.m <= 0 || p.m % 12 != 0) {
    throw std::invalid_argument("M must be a positive multiple of 12");
  }
  if (p.k <= 0 || p.k % 8 != 0) {
    throw std::invalid_argument("K must be a positive multiple of 8");
  }
  if (p.n_tile <= 0 || p.n <= 0 || p.n % p.n_tile != 0) {
    throw std::invalid_argument("N must be a positive multiple of n_tile");
  }
  if (p.stage == Stage::kW13) {
    if (p.n % 2 != 0 || p.ldc < p.n / 2) {
      throw std::invalid_argument("W13 requires even N and ldc >= N/2");
    }
  } else if (p.ldc < p.n) {
    throw std::invalid_argument("W2 requires ldc >= N");
  }
}

void check_lanes(const Problem& p, int lanes) {
  if (lanes <= 0 || lanes > p.n / p.n_tile) {
    throw std::invalid_argument("lane count exceeds available N tiles");
  }
}

void run_range(const Problem& p, int row_begin, int rows, int n_begin, int n_cols) {
  gemm_params_t params{};
  params.m = rows;
  params.k = p.k;
  params.n = n_cols;
  params.lda = p.k;
  params.ldb = p.k;
  params.ldc = p.ldc;

  const int64_t b_offset = static_cast<int64_t>(n_begin / p.n_tile) * p.k * p.n_tile;
  const uint16_t* a = p.packed_a + static_cast<int64_t>(row_begin) * p.k;
  const uint16_t* b = p.packed_b + b_offset;

  if (p.stage == Stage::kW13) {
    auto* c =
        static_cast<uint16_t*>(p.output) + static_cast<int64_t>(row_begin) * p.ldc + static_cast<int64_t>(n_begin) * 6;
    moe_sve_w13_silu_poly5_packc_m12_rows_opt(a, b, c, nullptr, &params);
    return;
  }

  params.m = 12;
  for (int row = 0; row < rows; row += 12) {
    const uint16_t* a_block = a + static_cast<int64_t>(row) * p.k;
    if (p.stage == Stage::kW2F32) {
      auto* c = static_cast<float*>(p.output) + static_cast<int64_t>(row_begin + row) * p.ldc + n_begin;
      moe_sve_w2_packed_m12(a_block, b, c, nullptr, &params);
    } else {
      auto* c = static_cast<uint16_t*>(p.output) + static_cast<int64_t>(row_begin + row) * p.ldc + n_begin;
      moe_sve_w2_packed_bf16_m12(a_block, b, c, nullptr, &params);
    }
  }
}

void run_lane(const Problem& p, int row_begin, int rows, int lanes, int lane) {
  const Range tile_range = split_tiles(p.n / p.n_tile, lanes, lane);
  if (tile_range.size == 0) {
    return;
  }
  run_range(p, row_begin, rows, tile_range.begin * p.n_tile, tile_range.size * p.n_tile);
}

void cpu_relax(int64_t& spins) {
  __asm__ __volatile__("yield" ::: "memory");
  if (++spins >= 4096) {
    std::this_thread::yield();
    spins = 0;
  }
}

uint64_t make_cursor(uint32_t epoch, uint32_t lane) { return (static_cast<uint64_t>(epoch) << 32) | lane; }

int bind_current_thread(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

}  // namespace

class Executor::Impl {
 public:
  Impl(int workers, int cpu_start) : worker_count_(workers), cpu_start_(cpu_start) {
    if (workers <= 0) {
      throw std::invalid_argument("worker count must be positive");
    }
    threads_.reserve(static_cast<size_t>(workers));
    for (int tid = 0; tid < workers; ++tid) {
      threads_.emplace_back([this, tid]() { worker_loop(tid); });
    }
    std::unique_lock<std::mutex> lock(mutex_);
    ready_cv_.wait(lock, [&]() { return ready_workers_ == worker_count_; });
    if (startup_error_ != 0) {
      const int error = startup_error_;
      lock.unlock();
      shutdown();
      throw std::runtime_error("pthread_setaffinity_np failed: " + std::to_string(error));
    }
  }

  ~Impl() { shutdown(); }

  template <typename Fn>
  double run(const Fn& fn) {
    std::function<void(int)> job = fn;
    const auto begin = std::chrono::steady_clock::now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (job_active_) {
        throw std::runtime_error("nested elastic executor job");
      }
      current_job_ = &job;
      remaining_workers_ = worker_count_;
      worker_exception_ = nullptr;
      job_active_ = true;
      ++generation_;
    }
    start_cv_.notify_all();

    std::exception_ptr error;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      done_cv_.wait(lock, [&]() { return remaining_workers_ == 0; });
      error = worker_exception_;
      current_job_ = nullptr;
      job_active_ = false;
    }
    const auto end = std::chrono::steady_clock::now();
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
    return std::chrono::duration<double>(end - begin).count();
  }

  int workers() const { return worker_count_; }

 private:
  void shutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) {
        return;
      }
      stopping_ = true;
      ++generation_;
    }
    start_cv_.notify_all();
    for (std::thread& thread : threads_) {
      if (thread.joinable()) {
        thread.join();
      }
    }
  }

  void worker_loop(int tid) {
    const int bind_error = bind_current_thread(cpu_start_ + tid);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (bind_error != 0 && startup_error_ == 0) {
        startup_error_ = bind_error;
      }
      ++ready_workers_;
    }
    ready_cv_.notify_one();

    uint64_t seen_generation = 0;
    while (true) {
      std::function<void(int)>* job = nullptr;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_cv_.wait(lock, [&]() { return stopping_ || generation_ != seen_generation; });
        if (stopping_) {
          return;
        }
        seen_generation = generation_;
        job = current_job_;
      }

      try {
        (*job)(tid);
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (worker_exception_ == nullptr) {
          worker_exception_ = std::current_exception();
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        --remaining_workers_;
        if (remaining_workers_ == 0) {
          done_cv_.notify_one();
        }
      }
    }
  }

  const int worker_count_;
  const int cpu_start_;
  std::vector<std::thread> threads_;
  std::mutex mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  std::condition_variable ready_cv_;
  std::function<void(int)>* current_job_ = nullptr;
  std::exception_ptr worker_exception_;
  uint64_t generation_ = 0;
  int ready_workers_ = 0;
  int remaining_workers_ = 0;
  int startup_error_ = 0;
  bool job_active_ = false;
  bool stopping_ = false;
};

Executor::Executor(int workers, int cpu_start) : impl_(std::make_unique<Impl>(workers, cpu_start)) {}

Executor::~Executor() = default;

RunResult Executor::run_static(const Problem& problem, int lanes) {
  check_problem(problem);
  check_lanes(problem, lanes);
  if (lanes > impl_->workers()) {
    throw std::invalid_argument("lane count exceeds resident workers");
  }
  const double seconds = impl_->run([&](int tid) {
    if (tid < lanes) {
      run_lane(problem, 0, problem.m, lanes, tid);
    }
  });
  return {seconds, lanes};
}

RunResult Executor::run_epoch_fixed(const Problem& problem, int lanes, int epoch_rows) {
  check_problem(problem);
  check_lanes(problem, lanes);
  if (lanes > impl_->workers()) {
    throw std::invalid_argument("lane count exceeds resident workers");
  }
  if (epoch_rows <= 0 || epoch_rows % 12 != 0) {
    throw std::invalid_argument("epoch_rows must be a positive multiple of 12");
  }
  const int epochs = (problem.m + epoch_rows - 1) / epoch_rows;
  const double seconds = impl_->run([&](int tid) {
    if (tid >= lanes) {
      return;
    }
    for (int row = 0; row < problem.m; row += epoch_rows) {
      const int rows = std::min(epoch_rows, problem.m - row);
      run_lane(problem, row, rows, lanes, tid);
    }
  });
  return {seconds, static_cast<int64_t>(epochs) * lanes};
}

RunResult Executor::run_phase_claim(const Problem& problem, const std::vector<EpochSpec>& epochs) {
  check_problem(problem);
  if (epochs.empty()) {
    throw std::invalid_argument("epoch plan must not be empty");
  }

  struct Phase {
    int lanes;
    std::vector<EpochSpec> epochs;
  };

  std::vector<Phase> phases;
  int expected_row = 0;
  int64_t lane_tasks = 0;
  for (const EpochSpec& epoch : epochs) {
    if (epoch.row_begin != expected_row || epoch.rows <= 0 || epoch.row_begin % 12 != 0 || epoch.rows % 12 != 0) {
      throw std::invalid_argument("epoch plan must be contiguous and 12-row aligned");
    }
    check_lanes(problem, epoch.lanes);
    if (epoch.lanes > impl_->workers()) {
      throw std::invalid_argument("epoch lane count exceeds resident workers");
    }
    if (phases.empty() || phases.back().lanes != epoch.lanes) {
      phases.push_back(Phase{epoch.lanes, {}});
    }
    phases.back().epochs.push_back(epoch);
    expected_row += epoch.rows;
    lane_tasks += epoch.lanes;
  }
  if (expected_row != problem.m) {
    throw std::invalid_argument("epoch plan does not cover M");
  }

  const size_t phase_count = phases.size();
  auto next_task = std::make_unique<std::atomic<int>[]>(static_cast<size_t>(phase_count));
  for (size_t index = 0; index < phase_count; ++index) {
    next_task[index].store(0, std::memory_order_relaxed);
  }
  std::atomic<uint32_t> phase_index{0};
  std::atomic<int> remaining{phases.front().lanes};

  const double seconds = impl_->run([&](int tid) {
    int64_t spins = 0;
    while (true) {
      const uint32_t observed = phase_index.load(std::memory_order_acquire);
      if (observed >= phase_count) {
        return;
      }
      const Phase& phase = phases[observed];
      const bool final_phase = observed + 1 == phase_count;
      if (tid >= phase.lanes) {
        if (final_phase) {
          return;
        }
        while (phase_index.load(std::memory_order_acquire) == observed) {
          cpu_relax(spins);
        }
        continue;
      }

      const int task_index = next_task[observed].fetch_add(1, std::memory_order_relaxed);
      if (task_index >= phase.lanes) {
        if (final_phase) {
          return;
        }
        while (phase_index.load(std::memory_order_acquire) == observed) {
          cpu_relax(spins);
        }
        continue;
      }

      for (const EpochSpec& epoch : phase.epochs) {
        run_lane(problem, epoch.row_begin, epoch.rows, epoch.lanes, task_index);
      }
      if (!final_phase && remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        const uint32_t next_phase = observed + 1;
        remaining.store(phases[next_phase].lanes, std::memory_order_relaxed);
        phase_index.store(next_phase, std::memory_order_release);
      }
    }
  });
  return {seconds, lane_tasks};
}

RunResult Executor::run_epoch_claim(const Problem& problem, const std::vector<EpochSpec>& epochs) {
  check_problem(problem);
  if (epochs.empty()) {
    throw std::invalid_argument("epoch plan must not be empty");
  }
  int expected_row = 0;
  int64_t lane_tasks = 0;
  for (const EpochSpec& epoch : epochs) {
    if (epoch.row_begin != expected_row || epoch.rows <= 0 || epoch.row_begin % 12 != 0 || epoch.rows % 12 != 0) {
      throw std::invalid_argument("epoch plan must be contiguous and 12-row aligned");
    }
    check_lanes(problem, epoch.lanes);
    if (epoch.lanes > impl_->workers()) {
      throw std::invalid_argument("epoch lane count exceeds resident workers");
    }
    expected_row += epoch.rows;
    lane_tasks += epoch.lanes;
  }
  if (expected_row != problem.m) {
    throw std::invalid_argument("epoch plan does not cover M");
  }

  std::atomic<uint64_t> cursor{make_cursor(0, 0)};
  std::atomic<int> remaining{epochs.front().lanes};
  const uint32_t epoch_count = static_cast<uint32_t>(epochs.size());

  const double seconds = impl_->run([&](int /*tid*/) {
    int64_t spins = 0;
    while (true) {
      uint64_t observed = cursor.load(std::memory_order_acquire);
      const uint32_t epoch_index = static_cast<uint32_t>(observed >> 32);
      if (epoch_index >= epoch_count) {
        return;
      }
      const uint32_t lane = static_cast<uint32_t>(observed);
      const EpochSpec& epoch = epochs[epoch_index];
      if (lane >= static_cast<uint32_t>(epoch.lanes)) {
        cpu_relax(spins);
        continue;
      }
      const uint64_t desired = make_cursor(epoch_index, lane + 1);
      if (!cursor.compare_exchange_weak(observed, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
        continue;
      }

      run_lane(problem, epoch.row_begin, epoch.rows, epoch.lanes, static_cast<int>(lane));
      if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        const uint32_t next_epoch = epoch_index + 1;
        if (next_epoch < epoch_count) {
          remaining.store(epochs[next_epoch].lanes, std::memory_order_relaxed);
        }
        cursor.store(make_cursor(next_epoch, 0), std::memory_order_release);
      }
    }
  });
  return {seconds, lane_tasks};
}

int runtime_n_tile() { return static_cast<int>(svcntb() / 2); }

std::vector<EpochSpec> make_epoch_plan(int m, int epoch_rows, int low_lanes, int high_lanes, double low_fraction) {
  if (m <= 0 || m % 12 != 0 || epoch_rows <= 0 || epoch_rows % 12 != 0) {
    throw std::invalid_argument("M and epoch_rows must be positive multiples of 12");
  }
  if (low_lanes <= 0 || high_lanes <= 0 || low_fraction < 0.0 || low_fraction > 1.0) {
    throw std::invalid_argument("invalid elastic epoch plan parameters");
  }
  std::vector<EpochSpec> result;
  for (int row = 0; row < m; row += epoch_rows) {
    result.push_back(EpochSpec{row, std::min(epoch_rows, m - row), high_lanes});
  }
  int low_epochs = static_cast<int>(result.size() * low_fraction + 0.5);
  low_epochs = std::clamp(low_epochs, 0, static_cast<int>(result.size()));
  for (int index = 0; index < low_epochs; ++index) {
    result[static_cast<size_t>(index)].lanes = low_lanes;
  }
  return result;
}

}  // namespace fused_moe_sve::elastic
