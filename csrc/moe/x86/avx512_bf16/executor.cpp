// SPDX-License-Identifier: Apache-2.0
#include "../../common/api.h"
#include "../../common/backend.h"
#include "backend.h"
#include "kernels.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(FUSED_CPP_HAS_OMP)
#include <omp.h>
#endif

namespace {

namespace avx512_moe = ::fused_cpp::moe::x86::avx512_bf16;

struct PackedShape {
  at::Tensor tensor;
  int64_t experts = 0;
  int64_t k = 0;
  int64_t n = 0;
  int k_pad = 0;
  int n_pad = 0;
  int64_t expert_stride = 0;
};

struct ExpertTask {
  int64_t expert = 0;
  const std::vector<int64_t>* routes = nullptr;
  int64_t output_row_begin = 0;
};

struct ThreadScratch {
  std::vector<uint16_t> transient_input;
  uint16_t* input = nullptr;
  std::vector<uint16_t> transient_intermediate;
  uint16_t* intermediate = nullptr;
};

constexpr int64_t kMaxExecutorThreads = 256;
constexpr int64_t kPersistentInputMinBytes = 256 * 1024;
constexpr int64_t kPersistentIntermediateMinBytes = 256 * 1024;
constexpr int64_t kWeightedTop1DirectMinWorkspaceBytes = 256 * 1024;
// Empirical runtime-mapper target for deciding how many experts share a wave.
// It changes only scheduling; N-range ownership keeps correctness independent
// of this value.
constexpr int64_t kNsplitTargetRowsPerThread = 64;

struct BlockRange {
  int64_t begin = 0;
  int64_t end = 0;
};

BlockRange SplitEvenly(int64_t units, int64_t group_size, int64_t local_tid) {
  if (units <= 0 || group_size <= 0 || local_tid < 0 || local_tid >= group_size) {
    return {};
  }
  const int64_t units_per_thread = units / group_size;
  const int64_t extra_units = units % group_size;
  const int64_t begin = local_tid * units_per_thread + std::min(local_tid, extra_units);
  const int64_t size = units_per_thread + (local_tid < extra_units ? 1 : 0);
  return BlockRange{begin, begin + size};
}

class ThreadBarrier {
 public:
  explicit ThreadBarrier(int64_t threads) : threads_(threads) {}

  bool Wait() {
    if (threads_ <= 1) {
      return !cancelled_.load(std::memory_order_acquire);
    }
    if (cancelled_.load(std::memory_order_acquire)) {
      return false;
    }
    const uint64_t generation = generation_.load(std::memory_order_acquire);
    if (arrivals_.fetch_add(1, std::memory_order_acq_rel) + 1 == threads_) {
      arrivals_.store(0, std::memory_order_relaxed);
      generation_.fetch_add(1, std::memory_order_release);
      return !cancelled_.load(std::memory_order_acquire);
    }
    while (generation_.load(std::memory_order_acquire) == generation && !cancelled_.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    return !cancelled_.load(std::memory_order_acquire);
  }

  void Cancel() {
    cancelled_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
  }

 private:
  const int64_t threads_;
  alignas(64) std::atomic<int64_t> arrivals_{0};
  alignas(64) std::atomic<uint64_t> generation_{0};
  alignas(64) std::atomic<bool> cancelled_{false};
};

struct CooperativeTeam {
  size_t task_index = 0;
  int64_t thread_begin = 0;
  int64_t threads = 1;
  std::unique_ptr<ThreadBarrier> barrier;
};

struct CooperativeWave {
  std::vector<CooperativeTeam> teams;
  int64_t threads = 0;
};

struct ThreadAssignment {
  size_t team_index = 0;
  int64_t local_tid = 0;
};

bool IsIntegerType(at::ScalarType type) {
  return type == at::kByte || type == at::kChar || type == at::kShort || type == at::kInt || type == at::kLong;
}

bool IsFloatingType(at::ScalarType type) {
  return type == at::kHalf || type == at::kBFloat16 || type == at::kFloat || type == at::kDouble;
}

void CheckBf16Cpu(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must use torch.bfloat16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

const uint16_t* Bf16Data(const at::Tensor& tensor) {
  return reinterpret_cast<const uint16_t*>(tensor.data_ptr<at::BFloat16>());
}

uint16_t* MutableBf16Data(at::Tensor& tensor) { return reinterpret_cast<uint16_t*>(tensor.data_ptr<at::BFloat16>()); }

struct PersistentBf16ScratchRecord {
  bool in_use = false;
  std::vector<at::Tensor> buffers;
};

class PersistentBf16ScratchPool {
 public:
  PersistentBf16ScratchRecord* Acquire(size_t buffer_count) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const std::unique_ptr<PersistentBf16ScratchRecord>& record : records_) {
      if (!record->in_use) {
        record->in_use = true;
        if (record->buffers.size() < buffer_count) {
          record->buffers.resize(buffer_count);
        }
        return record.get();
      }
    }
    auto record = std::make_unique<PersistentBf16ScratchRecord>();
    record->in_use = true;
    record->buffers.resize(buffer_count);
    PersistentBf16ScratchRecord* result = record.get();
    records_.push_back(std::move(record));
    return result;
  }

  void Release(PersistentBf16ScratchRecord* record) noexcept {
    if (record == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    record->in_use = false;
  }

 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<PersistentBf16ScratchRecord>> records_;
};

PersistentBf16ScratchPool& GetPersistentInputScratchPool() {
  static PersistentBf16ScratchPool pool;
  return pool;
}

PersistentBf16ScratchPool& GetPersistentIntermediateScratchPool() {
  static PersistentBf16ScratchPool pool;
  return pool;
}

class PersistentBf16ScratchLease {
 public:
  PersistentBf16ScratchLease(PersistentBf16ScratchPool* pool, size_t buffer_count) : pool_(pool) {
    if (pool_ != nullptr) {
      record_ = pool_->Acquire(buffer_count);
    }
  }

  PersistentBf16ScratchLease(const PersistentBf16ScratchLease&) = delete;
  PersistentBf16ScratchLease& operator=(const PersistentBf16ScratchLease&) = delete;

  ~PersistentBf16ScratchLease() {
    if (pool_ != nullptr) {
      pool_->Release(record_);
    }
  }

  at::Tensor& buffer(size_t index) { return record_->buffers[index]; }

 private:
  PersistentBf16ScratchPool* pool_ = nullptr;
  PersistentBf16ScratchRecord* record_ = nullptr;
};

enum class PersistentScratchMode { kAuto, kDisabled, kEnabled };

PersistentScratchMode ResolvePersistentScratchMode(const char* environment) {
  const char* raw = std::getenv(environment);
  if (raw == nullptr || raw[0] == '\0') {
    return PersistentScratchMode::kAuto;
  }
  const std::string value(raw);
  if (value == "auto") {
    return PersistentScratchMode::kAuto;
  }
  if (value == "1" || value == "true" || value == "TRUE") {
    return PersistentScratchMode::kEnabled;
  }
  if (value == "0" || value == "false" || value == "FALSE") {
    return PersistentScratchMode::kDisabled;
  }
  TORCH_CHECK(false, environment, " must be auto, 0, or 1; got '", value, "'");
  return PersistentScratchMode::kDisabled;
}

enum class WeightedTop1DirectMode { kAuto, kDisabled, kEnabled };

WeightedTop1DirectMode ResolveWeightedTop1DirectMode() {
  const char* raw = std::getenv("FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT");
  if (raw == nullptr || raw[0] == '\0') {
    return WeightedTop1DirectMode::kAuto;
  }
  const std::string value(raw);
  if (value == "auto") {
    return WeightedTop1DirectMode::kAuto;
  }
  if (value == "1" || value == "true" || value == "TRUE") {
    return WeightedTop1DirectMode::kEnabled;
  }
  if (value == "0" || value == "false" || value == "FALSE") {
    return WeightedTop1DirectMode::kDisabled;
  }
  TORCH_CHECK(false, "FUSED_CPP_MOE_X86_WEIGHTED_TOP1_DIRECT must be auto, 0, or 1; got '", value, "'");
  return WeightedTop1DirectMode::kDisabled;
}

void ClearIntermediatePadding(uint16_t* intermediate, int64_t rows, int f_pad, int intermediate_stride) {
  if (f_pad >= intermediate_stride) {
    return;
  }
  for (int64_t row = 0; row < rows; ++row) {
    uint16_t* row_data = intermediate + row * intermediate_stride;
    std::fill(row_data + f_pad, row_data + intermediate_stride, static_cast<uint16_t>(0));
  }
}

template <typename Function>
void RunThreads(int64_t threads, Function&& function) {
  if (threads == 1) {
    function(int64_t{0});
    return;
  }
#if defined(FUSED_CPP_HAS_OMP)
  if (omp_in_parallel() == 0) {
    int actual_threads = 0;
    std::vector<std::exception_ptr> omp_errors(static_cast<size_t>(threads));
#pragma omp parallel num_threads(threads)
    {
      const int64_t tid = omp_get_thread_num();
#pragma omp single
      actual_threads = omp_get_num_threads();
      if (actual_threads == threads) {
        try {
          function(tid);
        } catch (...) {
          omp_errors[static_cast<size_t>(tid)] = std::current_exception();
        }
      }
    }
    if (actual_threads == threads) {
      for (const std::exception_ptr& error : omp_errors) {
        if (error != nullptr) {
          std::rethrow_exception(error);
        }
      }
      return;
    }
  }
#endif
  std::vector<std::thread> workers;
  std::vector<std::exception_ptr> errors(static_cast<size_t>(threads));
  workers.reserve(static_cast<size_t>(threads - 1));
  for (int64_t tid = 1; tid < threads; ++tid) {
    workers.emplace_back([&, tid]() {
      try {
        function(tid);
      } catch (...) {
        errors[static_cast<size_t>(tid)] = std::current_exception();
      }
    });
  }
  try {
    function(int64_t{0});
  } catch (...) {
    errors[0] = std::current_exception();
  }
  for (std::thread& worker : workers) {
    worker.join();
  }
  for (const std::exception_ptr& error : errors) {
    if (error != nullptr) {
      std::rethrow_exception(error);
    }
  }
}

int64_t ResolvePrepackThreads(int64_t experts) {
  const char* value = std::getenv("FUSED_CPP_MOE_PREPACK_THREADS");
  if (value == nullptr || value[0] == '\0') {
    return 1;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  TORCH_CHECK(end != value && *end == '\0' && parsed > 0, "FUSED_CPP_MOE_PREPACK_THREADS must be positive, got ",
              value);
  return std::min<int64_t>(experts, parsed);
}

PackedShape CheckPacked(const at::Tensor& tensor, int64_t k, int64_t n, const char* name,
                        const ::fused_cpp::moe::MoeBackend& backend) {
  CheckBf16Cpu(tensor, name);
  TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D [experts, packed_elements]");
  TORCH_CHECK(k > 0 && k <= std::numeric_limits<int>::max(), name, " logical K is invalid: ", k);
  TORCH_CHECK(n > 0 && n <= std::numeric_limits<int>::max(), name, " logical N is invalid: ", n);
  const int k_pad = backend.round_k(static_cast<int>(k));
  const int n_pad = backend.round_n(static_cast<int>(n));
  const int64_t expert_stride = static_cast<int64_t>(k_pad) * n_pad;
  TORCH_CHECK(tensor.size(1) == expert_stride, name, " packed size mismatch: expected ", expert_stride, ", got ",
              tensor.size(1));
  return PackedShape{tensor, tensor.size(0), k, n, k_pad, n_pad, expert_stride};
}

at::Tensor PrepareOutput(const at::Tensor& input, const c10::optional<at::Tensor>& out) {
  if (!out.has_value()) {
    return at::empty_like(input);
  }
  at::Tensor output = out.value();
  CheckBf16Cpu(output, "out");
  TORCH_CHECK(output.sizes() == input.sizes(), "out must have the same shape as input");
  TORCH_CHECK(!output.requires_grad(), "out with requires_grad=True is not supported");
  return output;
}

void GatherInput(const uint16_t* input, int64_t hidden_size, int64_t top_k, const std::vector<int64_t>& routes,
                 uint16_t* gathered, int k_pad, int64_t group_size = 1, int64_t local_tid = 0) {
  constexpr int64_t kRowsPerPanel = 12;
  constexpr int64_t kPackedRowsPerPanel = 16;
  const int64_t panels = (static_cast<int64_t>(routes.size()) + kRowsPerPanel - 1) / kRowsPerPanel;
  const BlockRange panel_range = SplitEvenly(panels, group_size, local_tid);
  for (int64_t panel = panel_range.begin; panel < panel_range.end; ++panel) {
    uint16_t* block = gathered + panel * k_pad * kPackedRowsPerPanel;
    std::fill(block, block + static_cast<int64_t>(k_pad) * kPackedRowsPerPanel, static_cast<uint16_t>(0));
    for (int kp = 0; kp < k_pad / 2; ++kp) {
      uint16_t* pair_block = block + static_cast<int64_t>(kp) * 32;
      for (int lane = 0; lane < kRowsPerPanel; ++lane) {
        const int64_t row = panel * kRowsPerPanel + lane;
        if (row >= static_cast<int64_t>(routes.size())) {
          continue;
        }
        const int64_t token = routes[static_cast<size_t>(row)] / top_k;
        for (int half = 0; half < 2; ++half) {
          const int64_t k = kp * 2 + half;
          if (k < hidden_size) {
            pair_block[lane * 2 + half] = input[token * hidden_size + k];
          }
        }
      }
    }
  }
}

void GatherInputAmx(const uint16_t* input, int64_t hidden_size, int64_t top_k, const std::vector<int64_t>& routes,
                    uint16_t* gathered, int k_pad, int64_t group_size = 1, int64_t local_tid = 0) {
  const BlockRange row_range = SplitEvenly(static_cast<int64_t>(routes.size()), group_size, local_tid);
  for (int64_t row = row_range.begin; row < row_range.end; ++row) {
    const int64_t token = routes[static_cast<size_t>(row)] / top_k;
    uint16_t* destination = gathered + row * k_pad;
    std::copy(input + token * hidden_size, input + (token + 1) * hidden_size, destination);
    std::fill(destination + hidden_size, destination + k_pad, static_cast<uint16_t>(0));
  }
}

CooperativeWave BuildCooperativeWave(const std::vector<ExpertTask>& tasks, const std::vector<size_t>& task_indices,
                                     int64_t num_threads, int64_t max_n_parallelism) {
  std::vector<int64_t> team_widths(task_indices.size(), 1);
  int64_t remaining_threads = num_threads - static_cast<int64_t>(task_indices.size());
  // Give each additional worker to the team with the largest current route
  // rows per worker. N-block count caps useful parallelism.
  while (remaining_threads > 0) {
    size_t best = task_indices.size();
    long double best_rows_per_thread = -1.0;
    for (size_t index = 0; index < task_indices.size(); ++index) {
      if (team_widths[index] >= max_n_parallelism) {
        continue;
      }
      const ExpertTask& task = tasks[task_indices[index]];
      const long double rows_per_thread =
          static_cast<long double>(task.routes->size()) / static_cast<long double>(team_widths[index]);
      if (best == task_indices.size() || rows_per_thread > best_rows_per_thread) {
        best = index;
        best_rows_per_thread = rows_per_thread;
      }
    }
    if (best == task_indices.size()) {
      break;
    }
    ++team_widths[best];
    --remaining_threads;
  }

  CooperativeWave wave;
  wave.teams.reserve(task_indices.size());
  int64_t thread_begin = 0;
  for (size_t index = 0; index < task_indices.size(); ++index) {
    const int64_t team_threads = team_widths[index];
    wave.teams.push_back(CooperativeTeam{task_indices[index], thread_begin, team_threads,
                                         std::make_unique<ThreadBarrier>(team_threads)});
    thread_begin += team_threads;
  }
  wave.threads = thread_begin;
  return wave;
}

std::vector<CooperativeWave> BuildCooperativeSchedule(const std::vector<ExpertTask>& tasks, int64_t num_threads,
                                                      int64_t max_n_parallelism) {
  std::vector<size_t> ordered_tasks(tasks.size());
  for (size_t index = 0; index < tasks.size(); ++index) {
    ordered_tasks[index] = index;
  }
  std::stable_sort(ordered_tasks.begin(), ordered_tasks.end(),
                   [&](size_t lhs, size_t rhs) { return tasks[lhs].routes->size() > tasks[rhs].routes->size(); });

  std::vector<CooperativeWave> waves;
  for (size_t begin = 0; begin < ordered_tasks.size();) {
    const int64_t largest_rows = static_cast<int64_t>(tasks[ordered_tasks[begin]].routes->size());
    const int64_t target_active_experts =
        std::max<int64_t>(1, std::min<int64_t>(num_threads, num_threads * kNsplitTargetRowsPerThread / largest_rows));
    const size_t wave_tasks = std::min(ordered_tasks.size() - begin, static_cast<size_t>(target_active_experts));
    std::vector<size_t> task_indices(ordered_tasks.begin() + static_cast<std::ptrdiff_t>(begin),
                                     ordered_tasks.begin() + static_cast<std::ptrdiff_t>(begin + wave_tasks));
    waves.push_back(BuildCooperativeWave(tasks, task_indices, num_threads, max_n_parallelism));
    begin += wave_tasks;
  }
  return waves;
}

void GatherExpertInput(const ExpertTask& task, ThreadScratch& scratch, const uint16_t* input, int64_t hidden_size,
                       int64_t top_k, int k_pad, bool use_amx, int64_t group_size = 1, int64_t local_tid = 0) {
  const std::vector<int64_t>& routes = *task.routes;
  if (use_amx) {
    GatherInputAmx(input, hidden_size, top_k, routes, scratch.input, k_pad, group_size, local_tid);
  } else {
    GatherInput(input, hidden_size, top_k, routes, scratch.input, k_pad, group_size, local_tid);
  }
}

void RunExpertW13Range(const ExpertTask& task, ThreadScratch& scratch, const uint16_t* expert_input,
                       const uint16_t* w13, const PackedShape& w13_shape, int f_pad, int intermediate_stride,
                       int silu_poly_degree, bool use_amx, int feature_block_begin, int feature_block_end) {
  const int rows = static_cast<int>(task.routes->size());
  const uint16_t* expert_w13 = w13 + task.expert * w13_shape.expert_stride;
  if (use_amx) {
    avx512_moe::ComputeW13Amx(expert_input, w13_shape.k_pad, expert_w13, scratch.intermediate, intermediate_stride,
                              rows, w13_shape.k_pad, feature_block_begin, feature_block_end, silu_poly_degree);
  } else {
    avx512_moe::ComputeW13(expert_input, w13_shape.k_pad, expert_w13, scratch.intermediate, f_pad, rows,
                           w13_shape.k_pad, feature_block_begin, feature_block_end, silu_poly_degree);
  }
}

void RunExpertW2Range(const ExpertTask& task, ThreadScratch& scratch, const uint16_t* w2, const PackedShape& w2_shape,
                      int f_pad, int intermediate_stride, float* route_output, uint16_t* output,
                      const float* route_weights, int64_t hidden_size, bool direct_bf16, bool use_amx,
                      bool contiguous_route_output, int output_block_begin, int output_block_end) {
  const std::vector<int64_t>& routes = *task.routes;
  const int rows = static_cast<int>(routes.size());
  const uint16_t* expert_w2 = w2 + task.expert * w2_shape.expert_stride;
  if (use_amx) {
    float* expert_route_output = route_output;
    const int64_t* output_route_ids = routes.data();
    if (contiguous_route_output) {
      expert_route_output += task.output_row_begin * hidden_size;
      output_route_ids = nullptr;
    }
    avx512_moe::ComputeW2Amx(scratch.intermediate, intermediate_stride, expert_w2, expert_route_output, output,
                             output_route_ids, static_cast<int>(hidden_size), rows, w2_shape.k_pad,
                             static_cast<int>(hidden_size), output_block_begin, output_block_end, direct_bf16,
                             route_weights);
  } else {
    avx512_moe::ComputeW2(scratch.intermediate, f_pad, expert_w2, route_output, output, routes.data(),
                          static_cast<int>(hidden_size), rows, w2_shape.k_pad, static_cast<int>(hidden_size),
                          output_block_begin, output_block_end, direct_bf16, route_weights);
  }
}

void RunExpert(const ExpertTask& task, ThreadScratch& scratch, const uint16_t* input, int64_t hidden_size,
               int64_t top_k, const uint16_t* w13, const PackedShape& w13_shape, const uint16_t* w2,
               const PackedShape& w2_shape, int f_pad, float* route_output, uint16_t* output,
               const float* route_weights, bool direct_bf16, int silu_poly_degree, bool use_amx,
               bool contiguous_route_output, bool use_direct_input) {
  const uint16_t* expert_input = input;
  if (!use_direct_input) {
    GatherExpertInput(task, scratch, input, hidden_size, top_k, w13_shape.k_pad, use_amx);
    expert_input = scratch.input;
  }
  // AMX W2 loads complete K32 tiles, so its A stride must include padding
  // beyond W13's potentially smaller F16-padded feature range.
  const int intermediate_stride = use_amx ? w2_shape.k_pad : f_pad;
  RunExpertW13Range(task, scratch, expert_input, w13, w13_shape, f_pad, intermediate_stride, silu_poly_degree, use_amx,
                    0, f_pad / 16);
  RunExpertW2Range(task, scratch, w2, w2_shape, f_pad, intermediate_stride, route_output, output, route_weights,
                   hidden_size, direct_bf16, use_amx, contiguous_route_output, 0, w2_shape.n_pad / 32);
}

}  // namespace

std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t, int64_t, int64_t>
fused_moe_bf16_tiled_prepare_weights(at::Tensor w13_weight, at::Tensor w2_weight, bool fuse_silu,
                                     std::string backend_name) {
  const ::fused_cpp::moe::MoeBackend& backend = ::fused_cpp::moe::resolve_backend(backend_name, fuse_silu);
  TORCH_CHECK(backend.id == ::fused_cpp::moe::BackendId::kX86Avx512Bf16 ||
                  backend.id == ::fused_cpp::moe::BackendId::kX86AmxBf16,
              "x86 fused MoE prepare requires the x86_avx512_bf16 or x86_amx_bf16 backend");
  TORCH_CHECK(fuse_silu, backend.name, " requires fuse_silu=True");
  CheckBf16Cpu(w13_weight, "w13_weight");
  CheckBf16Cpu(w2_weight, "w2_weight");
  TORCH_CHECK(w13_weight.dim() == 3, "w13_weight must be 3-D [experts, 2 * F, H]");
  TORCH_CHECK(w2_weight.dim() == 3, "w2_weight must be 3-D [experts, H, F]");
  TORCH_CHECK(w13_weight.size(0) == w2_weight.size(0), "w13_weight and w2_weight must have the same expert count");
  TORCH_CHECK(w13_weight.size(0) > 0, "expert count must be positive");
  TORCH_CHECK(w13_weight.size(1) > 0 && w13_weight.size(1) % 2 == 0, "w13 output dim must be positive and even");

  const int64_t experts = w13_weight.size(0);
  const int64_t hidden_size = w13_weight.size(2);
  const int64_t f_size = w13_weight.size(1) / 2;
  TORCH_CHECK(hidden_size > 0 && hidden_size <= std::numeric_limits<int>::max(),
              "hidden size is invalid: ", hidden_size);
  TORCH_CHECK(f_size > 0 && f_size <= std::numeric_limits<int>::max() / 2,
              "FFN hidden size is invalid or too large for the packed W13 shape: ", f_size);
  TORCH_CHECK(w2_weight.size(1) == hidden_size && w2_weight.size(2) == f_size, "w2_weight shape mismatch: expected [",
              experts, ", ", hidden_size, ", ", f_size, "]");

  const int k13_pad = backend.round_k(static_cast<int>(hidden_size));
  const int n13_pad = backend.round_n(static_cast<int>(2 * f_size));
  const int k2_pad = backend.round_k(static_cast<int>(f_size));
  const int n2_pad = backend.round_n(static_cast<int>(hidden_size));
  const int f_pad = n13_pad / 2;
  at::Tensor w13_packed = at::empty({experts, static_cast<int64_t>(k13_pad) * n13_pad}, w13_weight.options());
  at::Tensor w2_packed = at::empty({experts, static_cast<int64_t>(k2_pad) * n2_pad}, w2_weight.options());

  const uint16_t* w13_source = Bf16Data(w13_weight);
  const uint16_t* w2_source = Bf16Data(w2_weight);
  uint16_t* w13_destination = MutableBf16Data(w13_packed);
  uint16_t* w2_destination = MutableBf16Data(w2_packed);
  const int64_t prepack_threads = ResolvePrepackThreads(experts);
  RunThreads(prepack_threads, [&](int64_t tid) {
    for (int64_t expert = tid; expert < experts; expert += prepack_threads) {
      avx512_moe::PackW13(w13_source + expert * 2 * f_size * hidden_size,
                          w13_destination + expert * static_cast<int64_t>(k13_pad) * n13_pad, f_size, hidden_size,
                          k13_pad, f_pad);
      avx512_moe::PackW2(w2_source + expert * hidden_size * f_size,
                         w2_destination + expert * static_cast<int64_t>(k2_pad) * n2_pad, hidden_size, f_size, k2_pad,
                         n2_pad);
    }
  });

  return std::make_tuple(w13_packed, hidden_size, 2 * f_size, w2_packed, f_size, hidden_size,
                         static_cast<int64_t>(backend.id), static_cast<int64_t>(backend.n_tile()));
}

at::Tensor fused_moe_bf16_tiled(at::Tensor input, at::Tensor w13_packed, int64_t w13_k, int64_t w13_n,
                                at::Tensor w2_packed, int64_t w2_k, int64_t w2_n, at::Tensor topk_weights,
                                at::Tensor topk_ids, c10::optional<at::Tensor> w13_bias,
                                c10::optional<at::Tensor> w2_bias, int64_t num_threads, std::string activation,
                                int64_t global_num_experts, bool skip_weighted, bool fuse_silu,
                                int64_t silu_poly_degree, int64_t gemm_backend, int64_t backend_n_tile,
                                int64_t weight_window_bytes, c10::optional<at::Tensor> out) {
  const ::fused_cpp::moe::MoeBackend& backend = ::fused_cpp::moe::backend_from_id(gemm_backend);
  const bool use_amx = backend.id == ::fused_cpp::moe::BackendId::kX86AmxBf16;
  TORCH_CHECK(backend.id == ::fused_cpp::moe::BackendId::kX86Avx512Bf16 || use_amx,
              "x86 fused MoE execution requires x86 AVX-512 or AMX packed weights");
  TORCH_CHECK(backend_n_tile == backend.n_tile(), "backend_n_tile mismatch: weights use ", backend_n_tile,
              ", runtime uses ", backend.n_tile());
  TORCH_CHECK(fuse_silu && activation == "silu", backend.name, " only supports fused activation='silu'");
  TORCH_CHECK(!w13_bias.has_value() && !w2_bias.has_value(), backend.name, " does not support expert bias");
  TORCH_CHECK(silu_poly_degree == 4 || silu_poly_degree == 5 || silu_poly_degree == 6,
              "silu_poly_degree must be 4, 5, or 6, got ", silu_poly_degree);
  TORCH_CHECK(num_threads > 0 && num_threads <= kMaxExecutorThreads, backend.name, " requires num_threads in [1, ",
              kMaxExecutorThreads, "], got ", num_threads);
  TORCH_CHECK(weight_window_bytes >= -1, "weight_window_bytes must be -1 or non-negative");

  CheckBf16Cpu(input, "input");
  TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, hidden]");
  TORCH_CHECK(input.size(1) == w13_k, "input hidden size and w13 K mismatch");
  TORCH_CHECK(w13_n > 0 && w13_n % 2 == 0, "w13 N must be positive and even");
  const int64_t f_size = w13_n / 2;
  TORCH_CHECK(w2_k == f_size && w2_n == input.size(1), "w2 logical shape mismatch");
  PackedShape w13_shape = CheckPacked(w13_packed, w13_k, w13_n, "w13_packed", backend);
  PackedShape w2_shape = CheckPacked(w2_packed, w2_k, w2_n, "w2_packed", backend);
  TORCH_CHECK(w13_shape.experts == w2_shape.experts, "packed weights must have the same expert count");

  TORCH_CHECK(topk_ids.device().is_cpu() && topk_weights.device().is_cpu(), "top-k tensors must be on CPU");
  TORCH_CHECK(IsIntegerType(topk_ids.scalar_type()), "topk_ids must use an integer dtype");
  TORCH_CHECK(IsFloatingType(topk_weights.scalar_type()), "topk_weights must use a floating dtype");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_weights.dim() == 2, "top-k tensors must be 2-D [tokens, top_k]");
  TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(), "topk_ids and topk_weights shapes must match");
  TORCH_CHECK(topk_ids.size(0) == input.size(0), "top-k token dimension must match input");
  TORCH_CHECK(topk_ids.size(1) > 0, "top_k must be positive");
  TORCH_CHECK(!skip_weighted || topk_ids.size(1) == 1, "skip_weighted is only valid when top_k == 1");

  const int64_t num_experts = global_num_experts < 0 ? w13_shape.experts : global_num_experts;
  TORCH_CHECK(num_experts > 0 && num_experts <= w13_shape.experts,
              "global_num_experts must be positive and no larger than packed experts");
  at::Tensor output = PrepareOutput(input, out);
  const int64_t num_tokens = input.size(0);
  if (num_tokens == 0) {
    return output;
  }

  at::Tensor ids_i64 = topk_ids.to(at::kLong).contiguous();
  at::Tensor weights_f32;
  if (!skip_weighted) {
    weights_f32 = topk_weights.to(at::kFloat).contiguous();
  }
  const int64_t* ids = ids_i64.data_ptr<int64_t>();
  const int64_t top_k = topk_ids.size(1);
  TORCH_CHECK(num_tokens <= std::numeric_limits<int64_t>::max() / top_k, "tokens * top_k overflows the route count");
  const int64_t num_routes = num_tokens * top_k;
  const WeightedTop1DirectMode weighted_top1_mode = ResolveWeightedTop1DirectMode();
  const int64_t route_row_bytes = input.size(1) * static_cast<int64_t>(sizeof(float));
  const int64_t automatic_min_routes = (kWeightedTop1DirectMinWorkspaceBytes + route_row_bytes - 1) / route_row_bytes;
  const bool use_weighted_top1_direct =
      !skip_weighted && top_k == 1 &&
      (weighted_top1_mode == WeightedTop1DirectMode::kEnabled ||
       (weighted_top1_mode == WeightedTop1DirectMode::kAuto && num_routes >= automatic_min_routes));
  const bool direct_bf16 = skip_weighted || use_weighted_top1_direct;
  const float* direct_route_weights = use_weighted_top1_direct ? weights_f32.data_ptr<float>() : nullptr;
  if (!use_amx) {
    TORCH_CHECK(num_routes <= std::numeric_limits<int32_t>::max() / input.size(1),
                "route output is too large for the AVX-512 scatter addressing: routes=", num_routes,
                ", hidden=", input.size(1));
  }
  std::vector<std::vector<int64_t>> routes(static_cast<size_t>(num_experts));
  for (int64_t flat = 0; flat < num_routes; ++flat) {
    const int64_t expert = ids[flat];
    TORCH_CHECK(expert >= 0 && expert < num_experts, "topk_ids out of range: ", expert, " not in [0, ", num_experts,
                ")");
    routes[static_cast<size_t>(expert)].push_back(flat);
  }
  const bool contiguous_route_output = use_amx && !direct_bf16 && avx512_moe::AmxW2UsesContiguousRouteOutput();
  std::vector<int64_t> route_output_rows;
  if (contiguous_route_output) {
    route_output_rows.resize(static_cast<size_t>(num_routes));
  }
  std::vector<ExpertTask> tasks;
  int64_t max_rows = 0;
  int64_t output_row_begin = 0;
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    const std::vector<int64_t>& expert_routes = routes[static_cast<size_t>(expert)];
    if (!expert_routes.empty()) {
      tasks.push_back(ExpertTask{expert, &expert_routes, output_row_begin});
      max_rows = std::max<int64_t>(max_rows, expert_routes.size());
      if (contiguous_route_output) {
        for (size_t row = 0; row < expert_routes.size(); ++row) {
          route_output_rows[static_cast<size_t>(expert_routes[row])] = output_row_begin + static_cast<int64_t>(row);
        }
      }
      output_row_begin += static_cast<int64_t>(expert_routes.size());
    }
  }
  TORCH_CHECK(max_rows <= std::numeric_limits<int>::max(),
              "one expert has too many routed rows for the x86 kernel: ", max_rows);
  // With top-k=1 and only one active expert, routes were appended in flat
  // token order. If H already satisfies AMX's K32 alignment, that gathered
  // matrix is byte-for-byte the original contiguous input, so avoid copying
  // the full MxH tensor into scratch.
  const bool use_direct_input = use_amx && top_k == 1 && tasks.size() == 1 &&
                                static_cast<int64_t>(tasks.front().routes->size()) == num_tokens &&
                                w13_shape.k_pad == input.size(1);

  std::vector<int> jit_row_counts;
  jit_row_counts.reserve(tasks.size());
  for (const ExpertTask& task : tasks) {
    jit_row_counts.push_back(static_cast<int>(task.routes->size()));
  }
  if (use_amx) {
    avx512_moe::PrepareAmxJitKernels(jit_row_counts, silu_poly_degree, static_cast<int>(input.size(1)), direct_bf16,
                                     use_weighted_top1_direct);
  } else {
    avx512_moe::PrepareJitKernels(jit_row_counts, silu_poly_degree, static_cast<int>(input.size(1)), direct_bf16,
                                  use_weighted_top1_direct);
  }

  const int f_pad = w13_shape.n_pad / 2;
  const int intermediate_stride = use_amx ? w2_shape.k_pad : f_pad;
  const int w13_blocks = f_pad / 16;
  const int w2_blocks = w2_shape.n_pad / 32;
  int64_t largest_rows = 0;
  int64_t second_largest_rows = 0;
  for (const ExpertTask& task : tasks) {
    const int64_t rows = static_cast<int64_t>(task.routes->size());
    if (rows > largest_rows) {
      second_largest_rows = largest_rows;
      largest_rows = rows;
    } else if (rows > second_largest_rows) {
      second_largest_rows = rows;
    }
  }
  const bool route_skewed =
      second_largest_rows > 0 && largest_rows >= kNsplitTargetRowsPerThread && largest_rows >= 2 * second_largest_rows;
  // Underfilled calls use one cooperative wave. Strongly skewed calls use
  // sorted waves even when every worker could claim a cold expert; balanced
  // calls retain the lower-overhead atomic expert queue.
  const bool use_cooperative_schedule =
      num_threads > 1 && (static_cast<int64_t>(tasks.size()) < num_threads || route_skewed);
  std::vector<CooperativeWave> cooperative_waves;
  size_t cooperative_scratch_count = 0;
  if (use_cooperative_schedule) {
    const int64_t max_n_parallelism = std::max(w13_blocks, w2_blocks);
    if (route_skewed) {
      cooperative_waves = BuildCooperativeSchedule(tasks, num_threads, max_n_parallelism);
    } else {
      std::vector<size_t> task_indices(tasks.size());
      for (size_t index = 0; index < tasks.size(); ++index) {
        task_indices[index] = index;
      }
      cooperative_waves.push_back(BuildCooperativeWave(tasks, task_indices, num_threads, max_n_parallelism));
    }
    for (const CooperativeWave& wave : cooperative_waves) {
      cooperative_scratch_count = std::max(cooperative_scratch_count, wave.teams.size());
    }
  }
  const size_t scratch_count = use_cooperative_schedule ? cooperative_scratch_count : static_cast<size_t>(num_threads);
  std::vector<ThreadScratch> scratches(scratch_count);
  std::vector<int64_t> scratch_logical_rows(scratch_count, max_rows);
  if (use_cooperative_schedule) {
    std::fill(scratch_logical_rows.begin(), scratch_logical_rows.end(), int64_t{0});
    for (const CooperativeWave& wave : cooperative_waves) {
      for (size_t team_index = 0; team_index < wave.teams.size(); ++team_index) {
        const int64_t rows = static_cast<int64_t>(tasks[wave.teams[team_index].task_index].routes->size());
        scratch_logical_rows[team_index] = std::max(scratch_logical_rows[team_index], rows);
      }
    }
  }
  std::vector<int64_t> scratch_rows_by_slot(scratch_count);
  int64_t total_input_elements = 0;
  int64_t total_intermediate_elements = 0;
  for (size_t index = 0; index < scratch_count; ++index) {
    const int64_t scratch_rows = use_amx ? scratch_logical_rows[index] : (scratch_logical_rows[index] + 11) / 12 * 16;
    TORCH_CHECK(scratch_rows <= std::numeric_limits<int64_t>::max() / w13_shape.k_pad,
                "x86 fused MoE input scratch size overflow");
    TORCH_CHECK(scratch_rows <= std::numeric_limits<int64_t>::max() / intermediate_stride,
                "x86 fused MoE intermediate scratch size overflow");
    const int64_t input_elements = scratch_rows * w13_shape.k_pad;
    const int64_t intermediate_elements = scratch_rows * intermediate_stride;
    TORCH_CHECK(total_input_elements <= std::numeric_limits<int64_t>::max() - input_elements,
                "x86 fused MoE total input scratch size overflow");
    TORCH_CHECK(total_intermediate_elements <= std::numeric_limits<int64_t>::max() - intermediate_elements,
                "x86 fused MoE total intermediate scratch size overflow");
    scratch_rows_by_slot[index] = scratch_rows;
    total_input_elements += input_elements;
    total_intermediate_elements += intermediate_elements;
  }
  const PersistentScratchMode persistent_input_mode =
      ResolvePersistentScratchMode("FUSED_CPP_MOE_X86_PERSISTENT_INPUT");
  const PersistentScratchMode persistent_intermediate_mode =
      ResolvePersistentScratchMode("FUSED_CPP_MOE_X86_PERSISTENT_INTERMEDIATE");
  const bool use_persistent_input =
      !use_direct_input &&
      (persistent_input_mode == PersistentScratchMode::kEnabled ||
       (persistent_input_mode == PersistentScratchMode::kAuto && use_amx &&
        total_input_elements >= kPersistentInputMinBytes / static_cast<int64_t>(sizeof(uint16_t))));
  const bool use_persistent_intermediate =
      persistent_intermediate_mode == PersistentScratchMode::kEnabled ||
      (persistent_intermediate_mode == PersistentScratchMode::kAuto && use_amx &&
       total_intermediate_elements >= kPersistentIntermediateMinBytes / static_cast<int64_t>(sizeof(uint16_t)));
  PersistentBf16ScratchLease input_lease(use_persistent_input ? &GetPersistentInputScratchPool() : nullptr,
                                         scratch_count);
  PersistentBf16ScratchLease intermediate_lease(
      use_persistent_intermediate ? &GetPersistentIntermediateScratchPool() : nullptr, scratch_count);
  for (size_t index = 0; index < scratches.size(); ++index) {
    const int64_t scratch_rows = scratch_rows_by_slot[index];
    if (!use_direct_input) {
      const int64_t input_elements = scratch_rows * w13_shape.k_pad;
      if (use_persistent_input) {
        at::Tensor& buffer = input_lease.buffer(index);
        if (!buffer.defined() || buffer.numel() < input_elements) {
          buffer = at::empty({input_elements}, input.options());
        }
        scratches[index].input = MutableBf16Data(buffer);
      } else {
        scratches[index].transient_input.resize(static_cast<size_t>(input_elements));
        scratches[index].input = scratches[index].transient_input.data();
      }
    }
    const int64_t intermediate_elements = scratch_rows * intermediate_stride;
    if (use_persistent_intermediate) {
      at::Tensor& buffer = intermediate_lease.buffer(index);
      if (!buffer.defined() || buffer.numel() < intermediate_elements) {
        buffer = at::empty({intermediate_elements}, input.options());
      }
      scratches[index].intermediate = MutableBf16Data(buffer);
    } else {
      scratches[index].transient_intermediate.resize(static_cast<size_t>(intermediate_elements));
      scratches[index].intermediate = scratches[index].transient_intermediate.data();
    }
    ClearIntermediatePadding(scratches[index].intermediate, scratch_rows, f_pad, intermediate_stride);
  }
  at::Tensor route_output;
  float* route_output_pointer = nullptr;
  if (!direct_bf16) {
    route_output = at::empty({num_routes, input.size(1)}, input.options().dtype(at::kFloat));
    route_output_pointer = route_output.data_ptr<float>();
  }
  const uint16_t* input_pointer = Bf16Data(input);
  const uint16_t* w13_pointer = Bf16Data(w13_shape.tensor);
  const uint16_t* w2_pointer = Bf16Data(w2_shape.tensor);
  uint16_t* output_pointer = MutableBf16Data(output);

  if (use_cooperative_schedule) {
    for (CooperativeWave& wave : cooperative_waves) {
      std::vector<ThreadAssignment> assignments(static_cast<size_t>(wave.threads));
      for (size_t team_index = 0; team_index < wave.teams.size(); ++team_index) {
        const CooperativeTeam& team = wave.teams[team_index];
        for (int64_t local_tid = 0; local_tid < team.threads; ++local_tid) {
          assignments[static_cast<size_t>(team.thread_begin + local_tid)] = ThreadAssignment{team_index, local_tid};
        }
      }
      RunThreads(wave.threads, [&](int64_t tid) {
        const ThreadAssignment assignment = assignments[static_cast<size_t>(tid)];
        CooperativeTeam& team = wave.teams[assignment.team_index];
        const ExpertTask& task = tasks[team.task_index];
        ThreadScratch& scratch = scratches[assignment.team_index];
        ThreadBarrier& barrier = *team.barrier;
        try {
          if (!use_direct_input) {
            GatherExpertInput(task, scratch, input_pointer, input.size(1), top_k, w13_shape.k_pad, use_amx,
                              team.threads, assignment.local_tid);
          }
          if (!barrier.Wait()) {
            return;
          }

          const BlockRange w13_range = SplitEvenly(w13_blocks, team.threads, assignment.local_tid);
          const uint16_t* expert_input = use_direct_input ? input_pointer : scratch.input;
          RunExpertW13Range(task, scratch, expert_input, w13_pointer, w13_shape, f_pad, intermediate_stride,
                            silu_poly_degree, use_amx, static_cast<int>(w13_range.begin),
                            static_cast<int>(w13_range.end));
          if (!barrier.Wait()) {
            return;
          }

          const BlockRange w2_range = SplitEvenly(w2_blocks, team.threads, assignment.local_tid);
          RunExpertW2Range(task, scratch, w2_pointer, w2_shape, f_pad, intermediate_stride, route_output_pointer,
                           output_pointer, direct_route_weights, input.size(1), direct_bf16, use_amx,
                           contiguous_route_output, static_cast<int>(w2_range.begin), static_cast<int>(w2_range.end));
        } catch (...) {
          barrier.Cancel();
          throw;
        }
      });
    }
  } else {
    std::atomic<size_t> next_task{0};
    RunThreads(num_threads, [&](int64_t tid) {
      while (true) {
        const size_t index = next_task.fetch_add(1, std::memory_order_relaxed);
        if (index >= tasks.size()) {
          return;
        }
        RunExpert(tasks[index], scratches[static_cast<size_t>(tid)], input_pointer, input.size(1), top_k, w13_pointer,
                  w13_shape, w2_pointer, w2_shape, f_pad, route_output_pointer, output_pointer, direct_route_weights,
                  direct_bf16, silu_poly_degree, use_amx, contiguous_route_output, use_direct_input);
      }
    });
  }

  if (!direct_bf16) {
    const float* weights = weights_f32.data_ptr<float>();
    const int64_t merge_threads = std::min(num_threads, num_tokens);
    RunThreads(merge_threads, [&](int64_t tid) {
      const int64_t begin = num_tokens * tid / merge_threads;
      const int64_t end = num_tokens * (tid + 1) / merge_threads;
      if (contiguous_route_output) {
        avx512_moe::MergeRoutesMapped(route_output_pointer, route_output_rows.data(), weights, output_pointer, begin,
                                      end, top_k, input.size(1));
      } else {
        avx512_moe::MergeRoutes(route_output_pointer, weights, output_pointer, begin, end, top_k, input.size(1));
      }
    });
  }
  return output;
}
