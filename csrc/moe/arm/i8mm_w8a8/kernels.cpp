// SPDX-License-Identifier: Apache-2.0

#include <ATen/ATen.h>
#include <c10/util/BFloat16.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(FUSED_CPP_MOE_HAS_I8GEMM)
#include <arm_sve.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>

extern "C" {
#include "gemm_params.h"
#include "i8gemm.h"
void i8gemm_k_hybrid(const int8_t*, const int8_t*, int32_t*, int8_t*, const gemm_params_t*);
void i8gemm_k_narrow(const int8_t*, const int8_t*, int32_t*, int8_t*, const gemm_params_t*);
void i8gemm_k_narrow2(const int8_t*, const int8_t*, int32_t*, int8_t*, const gemm_params_t*);
void i8gemm_k_nld(const int8_t*, const int8_t*, int32_t*, int8_t*, const gemm_params_t*);
void i8gemm_k_nld1(const int8_t*, const int8_t*, int32_t*, int8_t*, const gemm_params_t*);
void i8gemm_k_nld2(const int8_t*, const int8_t*, int32_t*, int8_t*, const gemm_params_t*);
void i8gemm_k_nld4(const int8_t*, const int8_t*, int32_t*, int8_t*, const gemm_params_t*);
void i8gemm_k_nld_m12(const int8_t*, const int8_t*, int32_t*, int8_t*, const gemm_params_t*);
void i8_pack_A_neon_m8_asm(const int8_t*, int8_t*, int, int);
}
#endif

namespace {

using PreparedW8A8Tuple =
    std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t>;

void check_quantized_source(const at::Tensor& w13, const at::Tensor& s13, const at::Tensor& w2, const at::Tensor& s2) {
  TORCH_CHECK(w13.device().is_cpu() && w2.device().is_cpu(), "W8A8 weights must be CPU tensors");
  TORCH_CHECK(w13.scalar_type() == at::kChar && w2.scalar_type() == at::kChar,
              "W8A8 weights must have dtype torch.int8");
  TORCH_CHECK(w13.dim() == 3 && w2.dim() == 3, "W8A8 weights must be [E, 2F, H] and [E, H, F]");
  TORCH_CHECK(w13.size(0) > 0 && w13.size(0) == w2.size(0), "W8A8 expert counts must match and be positive");
  TORCH_CHECK(w13.size(1) % 2 == 0, "W8A8 W13 output dimension must be even");
  const int64_t f = w13.size(1) / 2;
  const int64_t h = w13.size(2);
  TORCH_CHECK(w2.size(1) == h && w2.size(2) == f, "W8A8 W2 shape mismatch");
  TORCH_CHECK(h % 16 == 0 && f % 16 == 0, "W8A8 requires H and F to be multiples of 16");
  TORCH_CHECK(s13.device().is_cpu() && s2.device().is_cpu() && s13.scalar_type() == at::kFloat &&
                  s2.scalar_type() == at::kFloat,
              "W8A8 scales must be CPU float32 tensors");
  TORCH_CHECK(s13.numel() == w13.size(0) * w13.size(1), "W8A8 W13 scales must contain one value per output channel");
  TORCH_CHECK(s2.numel() == w2.size(0) * w2.size(1), "W8A8 W2 scales must contain one value per output channel");
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(FUSED_CPP_MOE_HAS_I8GEMM)

constexpr int64_t kPlanVersion = 2;
constexpr int64_t kStrictExecution = 0;

int64_t round_up(int64_t value, int64_t quantum) { return (value + quantum - 1) / quantum * quantum; }

at::Tensor pack_expert_weights(const at::Tensor& weight, int64_t k_pad, int64_t n_pad) {
  const int64_t experts = weight.size(0);
  const int64_t n = weight.size(1);
  const int64_t k = weight.size(2);
  at::Tensor packed = at::empty({experts, k_pad * n_pad}, weight.options());
  const int8_t* source = weight.data_ptr<int8_t>();
  int8_t* destination = packed.data_ptr<int8_t>();
  int threads = 1;
  if (const char* value = std::getenv("FUSED_CPP_MOE_PREPACK_THREADS")) {
    threads = std::max(1, std::atoi(value));
  }
#pragma omp parallel num_threads(threads)
  {
    std::vector<int8_t> transposed(static_cast<size_t>(k_pad * n_pad));
#pragma omp for schedule(static)
    for (int64_t expert = 0; expert < experts; ++expert) {
      std::fill(transposed.begin(), transposed.end(), int8_t{0});
      const int8_t* expert_source = source + expert * n * k;
      for (int64_t column = 0; column < n; ++column) {
        for (int64_t inner = 0; inner < k; ++inner) {
          transposed[static_cast<size_t>(inner * n_pad + column)] = expert_source[column * k + inner];
        }
      }
      i8_pack_B(transposed.data(), destination + expert * k_pad * n_pad, static_cast<int>(k_pad),
                static_cast<int>(n_pad));
    }
  }
  return packed;
}

struct TeamBarrier {
  explicit TeamBarrier(int participants) : participants(participants) {}
  void wait() {
    const int observed = generation.load(std::memory_order_acquire);
    if (arrivals.fetch_add(1, std::memory_order_acq_rel) + 1 == participants) {
      arrivals.store(0, std::memory_order_relaxed);
      generation.fetch_add(1, std::memory_order_release);
      return;
    }
    while (generation.load(std::memory_order_acquire) == observed) {
      asm volatile("yield" ::: "memory");
    }
  }
  int participants;
  std::atomic<int> arrivals{0};
  std::atomic<int> generation{0};
};

struct Workspace {
  Workspace(int width, int rows, int h, int f)
      : barrier(width),
        input_q(static_cast<size_t>(rows) * h),
        input_packed(static_cast<size_t>(rows + 7) * h),
        input_scale(rows),
        w13_acc(static_cast<size_t>(rows) * 2 * f),
        intermediate_bf16(static_cast<size_t>(rows) * f),
        intermediate_q(static_cast<size_t>(rows) * f),
        intermediate_packed(static_cast<size_t>(rows + 7) * f),
        intermediate_scale(rows),
        w2_acc(static_cast<size_t>(rows) * h) {}
  TeamBarrier barrier;
  std::vector<int8_t> input_q;
  std::vector<int8_t> input_packed;
  std::vector<float> input_scale;
  std::vector<int32_t> w13_acc;
  std::vector<uint16_t> intermediate_bf16;
  std::vector<int8_t> intermediate_q;
  std::vector<int8_t> intermediate_packed;
  std::vector<float> intermediate_scale;
  std::vector<int32_t> w2_acc;
};

struct RuntimeScratch {
  at::Tensor route_output;
  std::vector<std::unique_ptr<Workspace>> workspaces;
  std::vector<int> max_rows;
  int team_width = 0;
  int hidden = 0;
  int intermediate = 0;

  void ensure(int64_t routes, int h, int f, int width, const std::vector<int>& required_rows,
              const at::TensorOptions& options) {
    if (!route_output.defined() || route_output.numel() < routes * h) {
      route_output = at::empty({routes * h}, options.dtype(at::kFloat));
    }
    bool rebuild = team_width != width || hidden != h || intermediate != f || workspaces.size() != required_rows.size();
    if (!rebuild) {
      for (size_t lane = 0; lane < required_rows.size(); ++lane) {
        if (max_rows[lane] < required_rows[lane]) {
          rebuild = true;
          break;
        }
      }
    }
    if (!rebuild) return;
    workspaces.clear();
    max_rows = required_rows;
    workspaces.reserve(required_rows.size());
    for (const int rows : required_rows) {
      workspaces.push_back(std::make_unique<Workspace>(width, rows, h, f));
    }
    team_width = width;
    hidden = h;
    intermediate = f;
  }
};

RuntimeScratch& runtime_scratch() {
  // The request thread owns this cache. Intentionally keep it alive until
  // process exit so large route buffers are not mmap/munmap'd on every call.
  thread_local RuntimeScratch* scratch = new RuntimeScratch();
  return *scratch;
}

struct ScheduledTask {
  int expert;
  int w13_window;
  int w2_window;
};

svfloat32_t load_bf16(svbool_t pg, const uint16_t* source) {
  return svreinterpret_f32_u32(svlsl_n_u32_x(pg, svld1uh_u32(pg, source), 16));
}

void store_bf16(svbool_t pg, uint16_t* destination, svfloat32_t value) {
  const svuint32_t bits = svreinterpret_u32_f32(value);
  const svuint32_t lsb = svand_n_u32_x(pg, svlsr_n_u32_x(pg, bits, 16), 1);
  svst1h_u32(pg, destination, svlsr_n_u32_x(pg, svadd_u32_x(pg, bits, svadd_n_u32_x(pg, lsb, 0x7fff)), 16));
}

svfloat32_t exp_fexpa_neg(svbool_t pg, svfloat32_t gate) {
  svfloat32_t x = svmax_n_f32_x(pg, svmin_n_f32_x(pg, svneg_f32_x(pg, gate), 87.0f), -87.0f);
  svfloat32_t encoded = svmla_n_f32_x(pg, svdup_f32(196735.0f), x, 1.4426950216293335f);
  const svfloat32_t k = svsub_n_f32_x(pg, encoded, 196735.0f);
  svfloat32_t residual = svmls_n_f32_x(pg, x, k, 0.693145751953125f);
  residual = svmls_n_f32_x(pg, residual, k, 1.428606765330187e-06f);
  const svfloat32_t scale = svexpa_f32(svreinterpret_u32_f32(encoded));
  svfloat32_t polynomial = svmla_n_f32_x(pg, svdup_f32(1.000003695487976f), residual, 0.5000003576278687f);
  return svmla_f32_x(pg, scale, scale, svmul_f32_x(pg, polynomial, residual));
}

void quantize_bf16_row(const uint16_t* source, int8_t* destination, int columns, float* scale_out) {
  const int64_t vl = svcntw();
  svfloat32_t maximum = svdup_f32(0.0f);
  for (int64_t column = 0; column < columns; column += vl) {
    const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(columns));
    maximum = svmax_f32_m(pg, maximum, svabs_f32_x(pg, load_bf16(pg, source + column)));
  }
  const float max_value = svmaxv_f32(svptrue_b32(), maximum);
  const float scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
  *scale_out = scale;
  const float inverse_scale = 1.0f / scale;
  for (int64_t column = 0; column < columns; column += vl) {
    const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(columns));
    svfloat32_t value = svmul_n_f32_x(pg, load_bf16(pg, source + column), inverse_scale);
    value = svmax_n_f32_x(pg, svmin_n_f32_x(pg, value, 127.0f), -127.0f);
    svst1b_s32(pg, destination + column, svcvt_s32_f32_x(pg, svrinta_f32_x(pg, value)));
  }
}

void w13_epilogue(const int32_t* accumulator, const float* scales, float activation_scale, uint16_t* output, int f,
                  float limit) {
  const int64_t vl = svcntw();
  const svfloat32_t activation = svdup_f32(activation_scale);
  for (int64_t column = 0; column < f; column += vl) {
    const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(f));
    svfloat32_t gate = svcvt_f32_s32_x(pg, svld1_s32(pg, accumulator + column));
    svfloat32_t up = svcvt_f32_s32_x(pg, svld1_s32(pg, accumulator + f + column));
    gate = svmul_f32_x(pg, gate, svmul_f32_x(pg, activation, svld1_f32(pg, scales + column)));
    up = svmul_f32_x(pg, up, svmul_f32_x(pg, activation, svld1_f32(pg, scales + f + column)));
    gate = svmin_n_f32_x(pg, gate, limit);
    up = svmax_n_f32_x(pg, svmin_n_f32_x(pg, up, limit), -limit);
    const svfloat32_t value =
        svdiv_f32_x(pg, svmul_f32_x(pg, gate, up), svadd_n_f32_x(pg, exp_fexpa_neg(pg, gate), 1.0f));
    store_bf16(pg, output + column, value);
  }
}

void store_w2(const int32_t* accumulator, const float* scales, float activation_scale, float* output, int begin,
              int end) {
  const int64_t vl = svcntw();
  const svfloat32_t activation = svdup_f32(activation_scale);
  for (int64_t column = begin; column < end; column += vl) {
    const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(end));
    svfloat32_t value = svcvt_f32_s32_x(pg, svld1_s32(pg, accumulator + column));
    svst1_f32(pg, output + column, svmul_f32_x(pg, value, svmul_f32_x(pg, activation, svld1_f32(pg, scales + column))));
  }
}

int select_m12_blocks(int rows) {
  int best = 0;
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  for (int blocks = 0; blocks <= rows / 12; ++blocks) {
    const int64_t cost =
        static_cast<int64_t>(blocks) * 12 * 70 + static_cast<int64_t>((rows - blocks * 12 + 7) / 8) * 8 * 83;
    if (cost < best_cost) {
      best_cost = cost;
      best = blocks;
    }
  }
  return best;
}

void pack_m12(const int8_t* source, int8_t* destination, int k) {
  size_t output = 0;
  for (int kb = 0; kb < k; kb += 8) {
    for (int pair = 0; pair < 6; ++pair) {
      std::memcpy(destination + output, source + static_cast<size_t>(pair * 2) * k + kb, 8);
      output += 8;
      std::memcpy(destination + output, source + static_cast<size_t>(pair * 2 + 1) * k + kb, 8);
      output += 8;
    }
  }
}

void pack_m8(const int8_t* source, int8_t* destination, int rows, int k) {
  if (rows == 8) {
    i8_pack_A_neon_m8_asm(source, destination, k, k);
    return;
  }
  size_t output = 0;
  for (int kb = 0; kb < k; kb += 8) {
    for (int row = 0; row < 8; ++row) {
      if (row < rows)
        std::memcpy(destination + output, source + static_cast<size_t>(row) * k + kb, 8);
      else
        std::memset(destination + output, 0, 8);
      output += 8;
    }
  }
}

void prepare_packed_a(const int8_t* input, int8_t* packed, int rows, int k, int tid, int width) {
  const int m12_blocks = select_m12_blocks(rows);
  const int m12_rows = m12_blocks * 12;
  const int blocks = m12_blocks + (rows - m12_rows + 7) / 8;
  for (int block = tid; block < blocks; block += width) {
    if (block < m12_blocks)
      pack_m12(input + static_cast<size_t>(block * 12) * k, packed + static_cast<size_t>(block * 12) * k, k);
    else {
      const int row = m12_rows + (block - m12_blocks) * 8;
      pack_m8(input + static_cast<size_t>(row) * k, packed + static_cast<size_t>(row) * k, std::min(8, rows - row), k);
    }
  }
}

void run_m8(const int8_t* input, const int8_t* weight, int32_t* output, int8_t* packed, int rows,
            const gemm_params_t& params) {
  if (rows <= 1)
    i8gemm_k_nld1(input, weight, output, packed, &params);
  else if (rows <= 2)
    i8gemm_k_nld2(input, weight, output, packed, &params);
  else if (rows <= 4)
    i8gemm_k_nld4(input, weight, output, packed, &params);
  else
    i8gemm_k_nld(input, weight, output, packed, &params);
}

void run_stage(const int8_t* input, int8_t* packed_a, const int8_t* weight, int32_t* output, int rows, int k, int n,
               int tid, int width, int window, TeamBarrier& barrier) {
  const bool packed = rows > 4 && !((width >= 2 && k <= 256 && rows <= 256) || (width >= 2 && rows <= 16 && k <= 1024));
  if (packed) {
    prepare_packed_a(input, packed_a, rows, k, tid, width);
    barrier.wait();
  }
  const int n_tile = static_cast<int>(svcntb() / 2);
  const int tiles = n / n_tile;
  const int tile_begin = tid * tiles / width;
  const int tile_end = (tid + 1) * tiles / width;
  const int owner_tiles = tile_end - tile_begin;
  if (owner_tiles == 0) return;
  const int window_tiles = window > 0 ? std::min(window, owner_tiles) : owner_tiles;
  for (int begin = tile_begin; begin < tile_end; begin += window_tiles) {
    const int end = std::min(tile_end, begin + window_tiles);
    const int n_begin = begin * n_tile;
    const int columns = (end - begin) * n_tile;
    const int8_t* b = weight + static_cast<size_t>(n_begin) * k;
    if (!packed) {
      gemm_params_t params{rows, k, columns, k, k, n};
      auto kernel = rows <= 2 ? i8gemm_k_narrow : rows <= 4 ? i8gemm_k_narrow2 : i8gemm_k_hybrid;
      kernel(input, b, output + n_begin, nullptr, &params);
      continue;
    }
    const int m12_blocks = select_m12_blocks(rows);
    const int m12_rows = m12_blocks * 12;
    for (int block = 0; block < m12_blocks; ++block) {
      const int row = block * 12;
      gemm_params_t params{12, k, columns, k, k, n};
      i8gemm_k_nld_m12(input + static_cast<size_t>(row) * k, b, output + static_cast<size_t>(row) * n + n_begin,
                       packed_a + static_cast<size_t>(row) * k, &params);
    }
    for (int row = m12_rows; row < rows; row += 8) {
      const int block_rows = std::min(8, rows - row);
      gemm_params_t params{block_rows, k, columns, k, k, n};
      run_m8(input + static_cast<size_t>(row) * k, b, output + static_cast<size_t>(row) * n + n_begin,
             packed_a + static_cast<size_t>(row) * k, block_rows, params);
    }
  }
}

void bind_thread(int cpu, std::atomic<int>& error) {
  if (cpu < 0 || cpu >= CPU_SETSIZE) {
    error.store(EINVAL, std::memory_order_relaxed);
    return;
  }
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  const int status = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (status != 0) error.store(status, std::memory_order_relaxed);
}

#endif

}  // namespace

bool fused_moe_w8a8_tiled_available() {
#if defined(__aarch64__) && defined(__ARM_FEATURE_SVE) && defined(FUSED_CPP_MOE_HAS_I8GEMM)
  return true;
#else
  return false;
#endif
}

PreparedW8A8Tuple fused_moe_w8a8_tiled_prepare_quantized_weights(at::Tensor w13_weight, at::Tensor w13_scale,
                                                                 at::Tensor w2_weight, at::Tensor w2_scale) {
  check_quantized_source(w13_weight, w13_scale, w2_weight, w2_scale);
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE) || !defined(FUSED_CPP_MOE_HAS_I8GEMM)
  TORCH_CHECK(false, "W8A8 fused MoE packing requires AArch64 SVE and i8mm");
#else
  w13_weight = w13_weight.contiguous();
  w2_weight = w2_weight.contiguous();
  const int64_t e = w13_weight.size(0);
  const int64_t h = w13_weight.size(2);
  const int64_t n13 = w13_weight.size(1);
  const int64_t f = n13 / 2;
  const int64_t n_tile = svcntb() / 2;
  const int64_t h_pad = round_up(h, 16);
  const int64_t f_pad = round_up(f, 16);
  at::Tensor q13 = pack_expert_weights(w13_weight, h_pad, round_up(n13, n_tile));
  at::Tensor q2 = pack_expert_weights(w2_weight, f_pad, round_up(h, n_tile));
  return {q13, h,     n13, w13_scale.reshape({e, n13}).contiguous(), q2, f, h, w2_scale.reshape({e, h}).contiguous(),
          1,   n_tile};
#endif
}

PreparedW8A8Tuple fused_moe_w8a8_tiled_prepare_weights(at::Tensor w13_weight, at::Tensor w2_weight) {
  TORCH_CHECK(w13_weight.device().is_cpu() && w2_weight.device().is_cpu(), "W8A8 weights must be CPU tensors");
  TORCH_CHECK(w13_weight.scalar_type() == at::kBFloat16 && w2_weight.scalar_type() == at::kBFloat16,
              "W8A8 source weights must have dtype torch.bfloat16");
  TORCH_CHECK(w13_weight.dim() == 3 && w2_weight.dim() == 3, "W8A8 weights must be [E, 2F, H] and [E, H, F]");
  auto quantize = [](const at::Tensor& source) {
    at::Tensor fp32 = source.to(at::kFloat);
    at::Tensor maximum = std::get<0>(fp32.abs().max(-1, true));
    at::Tensor scale = at::where(maximum > 0, maximum / 127.0f, at::ones_like(maximum));
    at::Tensor quantized = (fp32 / scale).round().clamp(-127, 127).to(at::kChar);
    return std::pair<at::Tensor, at::Tensor>{quantized.contiguous(), scale.squeeze(-1).contiguous()};
  };
  auto [q13, s13] = quantize(w13_weight);
  auto [q2, s2] = quantize(w2_weight);
  return fused_moe_w8a8_tiled_prepare_quantized_weights(q13, s13, q2, s2);
}

at::Tensor fused_moe_w8a8_tiled_async_plan_v2(
    at::Tensor input, at::Tensor w13_packed, int64_t w13_k, int64_t w13_n, at::Tensor w13_scales, at::Tensor w2_packed,
    int64_t w2_k, int64_t w2_n, at::Tensor w2_scales, at::Tensor topk_weights, at::Tensor topk_ids,
    at::Tensor task_expert_ids, at::Tensor task_core_begins, at::Tensor task_threads, at::Tensor task_dep_offsets,
    at::Tensor task_deps, int64_t plan_version, int64_t execution_mode, at::Tensor task_preferred_threads,
    at::Tensor task_min_threads, at::Tensor task_max_threads, at::Tensor task_allowed_thread_offsets,
    at::Tensor task_allowed_threads, at::Tensor task_placement_modes, at::Tensor task_numa_nodes,
    at::Tensor task_stage_ids, at::Tensor task_resize_points, at::Tensor task_range_granularities,
    c10::optional<at::Tensor> task_w13_window_tiles, c10::optional<at::Tensor> task_w2_window_tiles,
    c10::optional<at::Tensor> thread_cpu_ids, int64_t num_threads, std::string activation, int64_t global_num_experts,
    bool skip_weighted, int64_t backend_n_tile, c10::optional<at::Tensor> out, int64_t early_merge,
    double swiglu_limit) {
#if !defined(__aarch64__) || !defined(__ARM_FEATURE_SVE) || !defined(FUSED_CPP_MOE_HAS_I8GEMM)
  TORCH_CHECK(false, "W8A8 fused MoE execution requires AArch64 SVE and i8mm");
#else
  (void)task_preferred_threads;
  (void)task_allowed_thread_offsets;
  (void)task_allowed_threads;
  (void)task_placement_modes;
  (void)task_numa_nodes;
  (void)task_stage_ids;
  (void)task_range_granularities;
  TORCH_CHECK(plan_version == kPlanVersion && execution_mode == kStrictExecution,
              "W8A8 requires strict Plan V2 execution");
  TORCH_CHECK(activation == "silu" && swiglu_limit == 10.0, "W8A8 supports only clamped SwiGLU limit 10.0");
  TORCH_CHECK(global_num_experts == -1, "W8A8 v1 does not support expert-parallel id remapping");
  TORCH_CHECK(!skip_weighted, "W8A8 v1 requires weighted TopK merge");
  TORCH_CHECK(early_merge == 0, "W8A8 v1 requires early_merge=False");
  TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == at::kBFloat16 && input.dim() == 2,
              "W8A8 input must be a CPU BF16 [tokens, H] tensor");
  TORCH_CHECK(topk_ids.device().is_cpu() && topk_weights.device().is_cpu() && topk_ids.dim() == 2 &&
                  topk_weights.sizes() == topk_ids.sizes(),
              "W8A8 topk tensors must be matching CPU [tokens, top_k]");
  TORCH_CHECK(topk_ids.size(0) == input.size(0), "W8A8 topk token dimension must match input");
  TORCH_CHECK(topk_ids.scalar_type() == at::kLong, "W8A8 topk_ids must be torch.int64");
  TORCH_CHECK(topk_weights.scalar_type() == at::kFloat, "W8A8 topk_weights must be torch.float32");
  TORCH_CHECK(input.size(1) == w13_k && w13_n == 2 * w2_k && w2_n == w13_k, "W8A8 expert dimensions mismatch");
  TORCH_CHECK(backend_n_tile == static_cast<int64_t>(svcntb() / 2), "W8A8 backend N tile mismatch");
  TORCH_CHECK(num_threads > 0 && num_threads <= omp_get_max_threads(), "invalid W8A8 num_threads");

  auto i64 = [](at::Tensor tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == at::kLong && tensor.dim() == 1, name,
                " must be a CPU int64 vector");
    return tensor.contiguous();
  };
  task_expert_ids = i64(task_expert_ids, "task_expert_ids");
  task_core_begins = i64(task_core_begins, "task_core_begins");
  task_threads = i64(task_threads, "task_threads");
  task_dep_offsets = i64(task_dep_offsets, "task_dep_offsets");
  task_deps = i64(task_deps, "task_deps");
  task_min_threads = i64(task_min_threads, "task_min_threads");
  task_max_threads = i64(task_max_threads, "task_max_threads");
  task_resize_points = i64(task_resize_points, "task_resize_points");
  const int64_t tasks = task_expert_ids.numel();
  TORCH_CHECK(task_core_begins.numel() == tasks && task_threads.numel() == tasks && task_min_threads.numel() == tasks &&
                  task_max_threads.numel() == tasks,
              "W8A8 task vectors must have matching lengths");
  TORCH_CHECK(task_dep_offsets.numel() == tasks + 1 && task_dep_offsets[tasks].item<int64_t>() == task_deps.numel(),
              "W8A8 dependency offsets are invalid");
  TORCH_CHECK(task_resize_points.numel() == 0 || at::all(task_resize_points == 0).item<bool>(),
              "W8A8 v1 does not support dynamic resize points");
  const int64_t* widths = task_threads.data_ptr<int64_t>();
  const int64_t team_width = tasks > 0 ? widths[0] : 1;
  TORCH_CHECK(team_width > 0 && num_threads % team_width == 0, "W8A8 requires homogeneous aligned teams");
  const int64_t* begins = task_core_begins.data_ptr<int64_t>();
  const int64_t* experts = task_expert_ids.data_ptr<int64_t>();
  const int64_t* dep_offsets = task_dep_offsets.data_ptr<int64_t>();
  const int64_t* deps = task_deps.data_ptr<int64_t>();
  const int64_t* mins = task_min_threads.data_ptr<int64_t>();
  const int64_t* maxs = task_max_threads.data_ptr<int64_t>();
  const int64_t expert_count = w13_packed.size(0);
  TORCH_CHECK(w13_packed.scalar_type() == at::kChar && w2_packed.scalar_type() == at::kChar &&
                  w13_packed.device().is_cpu() && w2_packed.device().is_cpu() && w13_packed.dim() == 2 &&
                  w2_packed.dim() == 2 && w13_packed.is_contiguous() && w2_packed.is_contiguous() &&
                  w2_packed.size(0) == expert_count,
              "W8A8 packed weights are invalid");
  TORCH_CHECK(w13_scales.sizes() == at::IntArrayRef({expert_count, w13_n}) &&
                  w2_scales.sizes() == at::IntArrayRef({expert_count, w2_n}),
              "W8A8 scale shapes are invalid");
  TORCH_CHECK(w13_scales.device().is_cpu() && w2_scales.device().is_cpu() && w13_scales.scalar_type() == at::kFloat &&
                  w2_scales.scalar_type() == at::kFloat,
              "W8A8 scales must be CPU float32 tensors");
  TORCH_CHECK(w13_packed.size(1) == w13_k * w13_n && w2_packed.size(1) == w2_k * w2_n,
              "W8A8 packed weight strides are invalid");
  const int64_t lane_count = num_threads / team_width;
  TORCH_CHECK((w13_n / backend_n_tile) % team_width == 0 && (w2_n / backend_n_tile) % team_width == 0,
              "W8A8 v1 requires each stage's N-tile count to be divisible by the team width");
  std::vector<std::vector<ScheduledTask>> lanes(static_cast<size_t>(lane_count));
  std::vector<int> seen(static_cast<size_t>(expert_count), 0);
  const int64_t* w13_windows = task_w13_window_tiles.has_value() ? task_w13_window_tiles->data_ptr<int64_t>() : nullptr;
  const int64_t* w2_windows = task_w2_window_tiles.has_value() ? task_w2_window_tiles->data_ptr<int64_t>() : nullptr;
  for (int64_t task = 0; task < tasks; ++task) {
    TORCH_CHECK(widths[task] == team_width && mins[task] == team_width && maxs[task] == team_width,
                "W8A8 v1 requires fixed homogeneous task widths");
    TORCH_CHECK(begins[task] >= 0 && begins[task] % team_width == 0 && begins[task] + team_width <= num_threads,
                "W8A8 task placement must align to its team width");
    TORCH_CHECK(experts[task] >= 0 && experts[task] < expert_count && seen[experts[task]]++ == 0,
                "W8A8 tasks must name each active expert once");
    for (int64_t index = dep_offsets[task]; index < dep_offsets[task + 1]; ++index) {
      const int64_t predecessor = deps[index];
      TORCH_CHECK(predecessor >= 0 && predecessor < task && begins[predecessor] == begins[task] &&
                      widths[predecessor] == widths[task],
                  "W8A8 v1 supports only dependencies within one fixed team lane");
    }
    lanes[static_cast<size_t>(begins[task] / team_width)].push_back(
        ScheduledTask{static_cast<int>(experts[task]), w13_windows ? static_cast<int>(w13_windows[task]) : 0,
                      w2_windows ? static_cast<int>(w2_windows[task]) : 0});
  }

  input = input.contiguous();
  topk_ids = topk_ids.contiguous();
  topk_weights = topk_weights.contiguous();
  w13_packed = w13_packed.contiguous();
  w2_packed = w2_packed.contiguous();
  w13_scales = w13_scales.contiguous();
  w2_scales = w2_scales.contiguous();
  const int tokens = static_cast<int>(input.size(0));
  const int top_k = static_cast<int>(topk_ids.size(1));
  const int h = static_cast<int>(w13_k);
  const int f = static_cast<int>(w2_k);
  std::vector<std::vector<int32_t>> routes(static_cast<size_t>(expert_count));
  const int64_t* ids = topk_ids.data_ptr<int64_t>();
  for (int64_t route = 0; route < topk_ids.numel(); ++route) {
    TORCH_CHECK(ids[route] >= 0 && ids[route] < expert_count, "W8A8 topk expert id is out of range");
    routes[static_cast<size_t>(ids[route])].push_back(static_cast<int32_t>(route));
  }
  for (int64_t expert = 0; expert < expert_count; ++expert) {
    TORCH_CHECK(routes[static_cast<size_t>(expert)].empty() || seen[static_cast<size_t>(expert)] == 1,
                "W8A8 plan is missing an active expert");
  }
  std::vector<int> required_rows(static_cast<size_t>(lane_count), 1);
  for (size_t lane_id = 0; lane_id < lanes.size(); ++lane_id) {
    for (const auto& task : lanes[lane_id]) {
      required_rows[lane_id] = std::max(required_rows[lane_id], static_cast<int>(routes[task.expert].size()));
    }
  }
  RuntimeScratch& scratch = runtime_scratch();
  scratch.ensure(topk_ids.numel(), h, f, static_cast<int>(team_width), required_rows, input.options());
  auto& workspaces = scratch.workspaces;
  at::Tensor route_output = scratch.route_output.narrow(0, 0, topk_ids.numel() * h);
  at::Tensor output = out.has_value() ? *out : at::empty_like(input);
  TORCH_CHECK(output.device().is_cpu() && output.scalar_type() == at::kBFloat16 && output.sizes() == input.sizes() &&
                  output.is_contiguous(),
              "W8A8 out must be contiguous CPU BF16 with the input shape");
  TORCH_CHECK(!output.requires_grad(), "W8A8 out with requires_grad=True is not supported");
  std::vector<int> cpu_ids(static_cast<size_t>(num_threads), -1);
  if (thread_cpu_ids.has_value()) {
    at::Tensor cpus = i64(*thread_cpu_ids, "thread_cpu_ids");
    TORCH_CHECK(cpus.numel() == num_threads, "thread_cpu_ids length must equal num_threads");
    for (int64_t i = 0; i < num_threads; ++i) cpu_ids[static_cast<size_t>(i)] = cpus[i].item<int64_t>();
  }

  const uint16_t* input_ptr = reinterpret_cast<const uint16_t*>(input.data_ptr<c10::BFloat16>());
  const int8_t* w13_ptr = w13_packed.data_ptr<int8_t>();
  const int8_t* w2_ptr = w2_packed.data_ptr<int8_t>();
  const float* s13_ptr = w13_scales.data_ptr<float>();
  const float* s2_ptr = w2_scales.data_ptr<float>();
  float* route_ptr = route_output.data_ptr<float>();
  const float* topk_ptr = topk_weights.data_ptr<float>();
  uint16_t* output_ptr = reinterpret_cast<uint16_t*>(output.data_ptr<c10::BFloat16>());
  std::atomic<int> affinity_error{0};
#pragma omp parallel num_threads(num_threads)
  {
    const int tid = omp_get_thread_num();
    if (cpu_ids[static_cast<size_t>(tid)] >= 0) bind_thread(cpu_ids[static_cast<size_t>(tid)], affinity_error);
    const int lane_id = tid / team_width;
    const int local_tid = tid % team_width;
    Workspace& workspace = *workspaces[static_cast<size_t>(lane_id)];
    for (const ScheduledTask& task : lanes[static_cast<size_t>(lane_id)]) {
      const auto& expert_routes = routes[static_cast<size_t>(task.expert)];
      const int rows = static_cast<int>(expert_routes.size());
      const int row_begin = local_tid * rows / team_width;
      const int row_end = (local_tid + 1) * rows / team_width;
      for (int row = row_begin; row < row_end; ++row) {
        const int token = expert_routes[static_cast<size_t>(row)] / top_k;
        quantize_bf16_row(input_ptr + static_cast<size_t>(token) * h,
                          workspace.input_q.data() + static_cast<size_t>(row) * h, h,
                          &workspace.input_scale[static_cast<size_t>(row)]);
      }
      workspace.barrier.wait();
      run_stage(workspace.input_q.data(), workspace.input_packed.data(),
                w13_ptr + static_cast<size_t>(task.expert) * h * 2 * f, workspace.w13_acc.data(), rows, h, 2 * f,
                local_tid, static_cast<int>(team_width), task.w13_window, workspace.barrier);
      workspace.barrier.wait();
      const float* expert_s13 = s13_ptr + static_cast<size_t>(task.expert) * 2 * f;
      for (int row = row_begin; row < row_end; ++row) {
        uint16_t* intermediate = workspace.intermediate_bf16.data() + static_cast<size_t>(row) * f;
        w13_epilogue(workspace.w13_acc.data() + static_cast<size_t>(row) * 2 * f, expert_s13,
                     workspace.input_scale[static_cast<size_t>(row)], intermediate, f, 10.0f);
        quantize_bf16_row(intermediate, workspace.intermediate_q.data() + static_cast<size_t>(row) * f, f,
                          &workspace.intermediate_scale[static_cast<size_t>(row)]);
      }
      workspace.barrier.wait();
      run_stage(workspace.intermediate_q.data(), workspace.intermediate_packed.data(),
                w2_ptr + static_cast<size_t>(task.expert) * f * h, workspace.w2_acc.data(), rows, f, h, local_tid,
                static_cast<int>(team_width), task.w2_window, workspace.barrier);
      workspace.barrier.wait();
      const int column_begin = local_tid * h / team_width;
      const int column_end = (local_tid + 1) * h / team_width;
      const float* expert_s2 = s2_ptr + static_cast<size_t>(task.expert) * h;
      for (int row = 0; row < rows; ++row) {
        store_w2(workspace.w2_acc.data() + static_cast<size_t>(row) * h, expert_s2,
                 workspace.intermediate_scale[static_cast<size_t>(row)],
                 route_ptr + static_cast<size_t>(expert_routes[static_cast<size_t>(row)]) * h, column_begin,
                 column_end);
      }
      workspace.barrier.wait();
    }
#pragma omp barrier
    const int token_begin = tid * tokens / num_threads;
    const int token_end = (tid + 1) * tokens / num_threads;
    const int64_t vl = svcntw();
    for (int token = token_begin; token < token_end; ++token) {
      for (int64_t column = 0; column < h; column += vl) {
        const svbool_t pg = svwhilelt_b32(column, static_cast<int64_t>(h));
        svfloat32_t accumulator = svdup_f32(0.0f);
        for (int slot = 0; slot < top_k; ++slot) {
          const int64_t route = static_cast<int64_t>(token) * top_k + slot;
          accumulator = svmla_n_f32_x(pg, accumulator, svld1_f32(pg, route_ptr + route * h + column), topk_ptr[route]);
        }
        store_bf16(pg, output_ptr + static_cast<size_t>(token) * h + column, accumulator);
      }
    }
  }
  TORCH_CHECK(affinity_error.load(std::memory_order_relaxed) == 0, "W8A8 worker affinity binding failed");
  return output;
#endif
}
