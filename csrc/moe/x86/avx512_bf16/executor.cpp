// SPDX-License-Identifier: Apache-2.0
#include "../../common/api.h"
#include "../../common/backend.h"
#include "backend.h"
#include "kernels.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
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
};

struct ThreadScratch {
  std::vector<uint16_t> input;
  std::vector<uint16_t> intermediate;
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
                 uint16_t* gathered, int k_pad) {
  constexpr int64_t kRowsPerPanel = 12;
  constexpr int64_t kPackedRowsPerPanel = 16;
  const int64_t panels = (static_cast<int64_t>(routes.size()) + kRowsPerPanel - 1) / kRowsPerPanel;
  const int64_t padded_rows = panels * kPackedRowsPerPanel;
  std::fill(gathered, gathered + padded_rows * k_pad, static_cast<uint16_t>(0));
  for (int64_t panel = 0; panel < panels; ++panel) {
    uint16_t* block = gathered + panel * k_pad * kPackedRowsPerPanel;
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
                    uint16_t* gathered, int k_pad) {
  std::fill(gathered, gathered + static_cast<int64_t>(routes.size()) * k_pad, static_cast<uint16_t>(0));
  for (size_t row = 0; row < routes.size(); ++row) {
    const int64_t token = routes[row] / top_k;
    std::copy(input + token * hidden_size, input + (token + 1) * hidden_size,
              gathered + static_cast<int64_t>(row) * k_pad);
  }
}

void RunExpert(const ExpertTask& task, ThreadScratch& scratch, const uint16_t* input, int64_t hidden_size,
               int64_t top_k, const uint16_t* w13, const PackedShape& w13_shape, const uint16_t* w2,
               const PackedShape& w2_shape, int f_pad, float* route_output, uint16_t* output, bool skip_weighted,
               int silu_poly_degree, bool use_amx) {
  const std::vector<int64_t>& routes = *task.routes;
  const int rows = static_cast<int>(routes.size());
  if (use_amx) {
    GatherInputAmx(input, hidden_size, top_k, routes, scratch.input.data(), w13_shape.k_pad);
  } else {
    GatherInput(input, hidden_size, top_k, routes, scratch.input.data(), w13_shape.k_pad);
  }
  const uint16_t* expert_w13 = w13 + task.expert * w13_shape.expert_stride;
  const uint16_t* expert_w2 = w2 + task.expert * w2_shape.expert_stride;
  // AMX W2 loads complete K32 tiles, so its A stride must include padding
  // beyond W13's potentially smaller F16-padded feature range.
  const int intermediate_stride = use_amx ? w2_shape.k_pad : f_pad;
  if (use_amx) {
    avx512_moe::ComputeW13Amx(scratch.input.data(), w13_shape.k_pad, expert_w13, scratch.intermediate.data(),
                              intermediate_stride, rows, w13_shape.k_pad, 0, f_pad / 16, silu_poly_degree);
    avx512_moe::ComputeW2Amx(scratch.intermediate.data(), intermediate_stride, expert_w2, route_output, output,
                             routes.data(), static_cast<int>(hidden_size), rows, w2_shape.k_pad,
                             static_cast<int>(hidden_size), 0, w2_shape.n_pad / 32, skip_weighted);
  } else {
    avx512_moe::ComputeW13(scratch.input.data(), w13_shape.k_pad, expert_w13, scratch.intermediate.data(), f_pad, rows,
                           w13_shape.k_pad, 0, f_pad / 16, silu_poly_degree);
    avx512_moe::ComputeW2(scratch.intermediate.data(), f_pad, expert_w2, route_output, output, routes.data(),
                          static_cast<int>(hidden_size), rows, w2_shape.k_pad, static_cast<int>(hidden_size), 0,
                          w2_shape.n_pad / 32, skip_weighted);
  }
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
  TORCH_CHECK(num_threads == 1 || num_threads == 2, backend.name, " currently supports num_threads=1 or 2, got ",
              num_threads);
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
  std::vector<ExpertTask> tasks;
  int64_t max_rows = 0;
  for (int64_t expert = 0; expert < num_experts; ++expert) {
    if (!routes[static_cast<size_t>(expert)].empty()) {
      tasks.push_back(ExpertTask{expert, &routes[static_cast<size_t>(expert)]});
      max_rows = std::max<int64_t>(max_rows, routes[static_cast<size_t>(expert)].size());
    }
  }

  std::vector<int> jit_row_counts;
  jit_row_counts.reserve(tasks.size());
  for (const ExpertTask& task : tasks) {
    jit_row_counts.push_back(static_cast<int>(task.routes->size()));
  }
  if (use_amx) {
    avx512_moe::PrepareAmxJitKernels(jit_row_counts, silu_poly_degree, static_cast<int>(input.size(1)), skip_weighted);
  } else {
    avx512_moe::PrepareJitKernels(jit_row_counts, silu_poly_degree, static_cast<int>(input.size(1)), skip_weighted);
  }

  const int f_pad = w13_shape.n_pad / 2;
  const int intermediate_stride = use_amx ? w2_shape.k_pad : f_pad;
  const int64_t max_scratch_rows = use_amx ? max_rows : (max_rows + 11) / 12 * 16;
  std::vector<ThreadScratch> scratches(static_cast<size_t>(num_threads));
  for (ThreadScratch& scratch : scratches) {
    scratch.input.resize(static_cast<size_t>(max_scratch_rows * w13_shape.k_pad));
    scratch.intermediate.resize(static_cast<size_t>(max_scratch_rows * intermediate_stride));
  }
  at::Tensor route_output;
  float* route_output_pointer = nullptr;
  if (!skip_weighted) {
    route_output = at::empty({num_routes, input.size(1)}, input.options().dtype(at::kFloat));
    route_output_pointer = route_output.data_ptr<float>();
  }
  const uint16_t* input_pointer = Bf16Data(input);
  const uint16_t* w13_pointer = Bf16Data(w13_shape.tensor);
  const uint16_t* w2_pointer = Bf16Data(w2_shape.tensor);
  uint16_t* output_pointer = MutableBf16Data(output);

  if (num_threads == 2 && tasks.size() == 1) {
    ThreadScratch& scratch = scratches[0];
    const ExpertTask& task = tasks[0];
    const std::vector<int64_t>& expert_routes = *task.routes;
    const int rows = static_cast<int>(expert_routes.size());
    if (use_amx) {
      GatherInputAmx(input_pointer, input.size(1), top_k, expert_routes, scratch.input.data(), w13_shape.k_pad);
    } else {
      GatherInput(input_pointer, input.size(1), top_k, expert_routes, scratch.input.data(), w13_shape.k_pad);
    }
    const uint16_t* expert_w13 = w13_pointer + task.expert * w13_shape.expert_stride;
    const uint16_t* expert_w2 = w2_pointer + task.expert * w2_shape.expert_stride;
    const int w13_blocks = f_pad / 16;
    RunThreads(2, [&](int64_t tid) {
      const int begin = static_cast<int>(w13_blocks * tid / 2);
      const int end = static_cast<int>(w13_blocks * (tid + 1) / 2);
      if (use_amx) {
        avx512_moe::ComputeW13Amx(scratch.input.data(), w13_shape.k_pad, expert_w13, scratch.intermediate.data(),
                                  intermediate_stride, rows, w13_shape.k_pad, begin, end, silu_poly_degree);
      } else {
        avx512_moe::ComputeW13(scratch.input.data(), w13_shape.k_pad, expert_w13, scratch.intermediate.data(), f_pad,
                               rows, w13_shape.k_pad, begin, end, silu_poly_degree);
      }
    });
    const int w2_blocks = w2_shape.n_pad / 32;
    RunThreads(2, [&](int64_t tid) {
      const int begin = static_cast<int>(w2_blocks * tid / 2);
      const int end = static_cast<int>(w2_blocks * (tid + 1) / 2);
      if (use_amx) {
        avx512_moe::ComputeW2Amx(scratch.intermediate.data(), intermediate_stride, expert_w2, route_output_pointer,
                                 output_pointer, expert_routes.data(), static_cast<int>(input.size(1)), rows,
                                 w2_shape.k_pad, static_cast<int>(input.size(1)), begin, end, skip_weighted);
      } else {
        avx512_moe::ComputeW2(scratch.intermediate.data(), f_pad, expert_w2, route_output_pointer, output_pointer,
                              expert_routes.data(), static_cast<int>(input.size(1)), rows, w2_shape.k_pad,
                              static_cast<int>(input.size(1)), begin, end, skip_weighted);
      }
    });
  } else {
    std::atomic<size_t> next_task{0};
    RunThreads(num_threads, [&](int64_t tid) {
      while (true) {
        const size_t index = next_task.fetch_add(1, std::memory_order_relaxed);
        if (index >= tasks.size()) {
          return;
        }
        RunExpert(tasks[index], scratches[static_cast<size_t>(tid)], input_pointer, input.size(1), top_k, w13_pointer,
                  w13_shape, w2_pointer, w2_shape, f_pad, route_output_pointer, output_pointer, skip_weighted,
                  silu_poly_degree, use_amx);
      }
    });
  }

  if (!skip_weighted) {
    const float* weights = weights_f32.data_ptr<float>();
    RunThreads(num_threads, [&](int64_t tid) {
      const int64_t begin = num_tokens * tid / num_threads;
      const int64_t end = num_tokens * (tid + 1) / num_threads;
      avx512_moe::MergeRoutes(route_output_pointer, weights, output_pointer, begin, end, top_k, input.size(1));
    });
  }
  return output;
}
