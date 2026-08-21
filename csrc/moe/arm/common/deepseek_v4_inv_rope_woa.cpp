// SPDX-License-Identifier: Apache-2.0
#include "../../common/api.h"

#include <c10/util/BFloat16.h>

#include <arm_neon.h>

#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#if defined(FUSED_CPP_HAS_OMP)
#include <omp.h>
#endif

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include "../sve_bf16/jit_kernels.h"
#include "../sve_bf16/packing.h"
#include "nm_window_schedule.h"

#if defined(FUSED_CPP_MOE_HAS_ARM_SVE)
namespace {

using BFloat16 = c10::BFloat16;
using KernelFn = fused_cpp::moe_sve::jit::KernelFn;

struct Panel {
  int64_t row_begin = 0;
  int64_t packed_row_begin = 0;
  int rows = 0;
  int physical_rows = 0;
  KernelFn kernel = nullptr;
};

struct alignas(8) SveParams {
  gemm_params_t gemm{};
  int32_t kc = 0;
  int32_t packed_n = 0;
  int32_t n_begin = 0;
  int32_t mode = 0;
  float* partial_c = nullptr;
};

static_assert(sizeof(gemm_params_t) == 24, "unexpected GEMM params ABI");
static_assert(offsetof(SveParams, n_begin) == 32, "unexpected SVE N-begin params ABI");

void check_cpu_contiguous(const at::Tensor& tensor, at::ScalarType dtype, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has an invalid dtype: ", tensor.scalar_type());
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

std::vector<Panel> make_panels(int64_t rows) {
  std::vector<Panel> panels;
  int64_t packed_row = 0;
  int64_t row = 0;
  for (; row + 12 <= rows; row += 12) {
    panels.push_back(Panel{row, packed_row, 12, 12, nullptr});
    packed_row += 12;
  }
  const int tail = static_cast<int>(rows - row);
  if (tail > 0) {
    const int physical_rows = tail <= 8 ? 8 : 12;
    panels.push_back(Panel{row, packed_row, tail, physical_rows, nullptr});
  }
  return panels;
}

int64_t packed_rows(const std::vector<Panel>& panels) {
  if (panels.empty()) {
    return 0;
  }
  const Panel& last = panels.back();
  return last.packed_row_begin + last.physical_rows;
}

BFloat16 inverse_rope_value(const BFloat16* input, const float* cache, int64_t token, int64_t group, int64_t k,
                            int64_t heads, int64_t heads_per_group, int64_t head_dim, int64_t nope_dim,
                            int64_t rope_dim) {
  const int64_t head_in_group = k / head_dim;
  const int64_t dim = k - head_in_group * head_dim;
  const int64_t head = group * heads_per_group + head_in_group;
  const BFloat16* source = input + (token * heads + head) * head_dim;
  if (dim < nope_dim || rope_dim == 0) {
    return source[dim];
  }
  const int64_t rope_offset = dim - nope_dim;
  const int64_t pair = rope_offset / 2;
  const int64_t even_dim = nope_dim + pair * 2;
  const float even = static_cast<float>(source[even_dim]);
  const float odd = static_cast<float>(source[even_dim + 1]);
  const float cosine = cache[pair];
  const float sine = cache[rope_dim / 2 + pair];
  return BFloat16((rope_offset & 1) == 0 ? even * cosine + odd * sine : odd * cosine - even * sine);
}

void pack_group_panel(const BFloat16* input, const int64_t* positions, const float* cos_sin_cache, int64_t cache_stride,
                      int64_t heads, int64_t heads_per_group, int64_t head_dim, int64_t nope_dim, int64_t rope_dim,
                      int64_t logical_k, int64_t packed_k, int64_t group, const Panel& panel, BFloat16* packed) {
  const BFloat16* sources[12]{};
  const float* caches[12]{};
  for (int row = 0; row < panel.rows; ++row) {
    const int64_t token = panel.row_begin + row;
    sources[row] = input + (token * heads + group * heads_per_group) * head_dim;
    if (rope_dim > 0) {
      caches[row] = cos_sin_cache + positions[token] * cache_stride;
    }
  }

  for (int64_t kb = 0; kb < packed_k; kb += 4) {
    BFloat16* block = packed + (kb / 4) * panel.physical_rows * 4;
    const int64_t dim = kb < logical_k ? kb % head_dim : head_dim;
    const bool nope_block = kb + 4 <= logical_k && dim + 4 <= nope_dim;
    if (nope_block) {
      for (int row = 0; row < panel.physical_rows; row += 2) {
        uint64_t first = 0;
        uint64_t second = 0;
        if (row < panel.rows) {
          std::memcpy(&first, sources[row] + kb, sizeof(first));
        }
        if (row + 1 < panel.rows) {
          std::memcpy(&second, sources[row + 1] + kb, sizeof(second));
        }
        const uint64x2_t rows = vcombine_u64(vcreate_u64(first), vcreate_u64(second));
        vst1q_u8(reinterpret_cast<uint8_t*>(block + row * 4), vreinterpretq_u8_u64(rows));
      }
      continue;
    }

    const bool rope_block =
        kb + 4 <= logical_k && dim >= nope_dim && dim + 4 <= head_dim && ((dim - nope_dim) & 1) == 0;
    if (rope_block) {
      const int64_t first_pair = (dim - nope_dim) / 2;
      for (int row = 0; row < panel.physical_rows; ++row) {
        BFloat16* destination = block + row * 4;
        if (row >= panel.rows) {
          std::memset(destination, 0, 4 * sizeof(BFloat16));
          continue;
        }
        const BFloat16* source = sources[row] + kb;
        const float* cache = caches[row];
        for (int offset = 0; offset < 4; offset += 2) {
          const int64_t pair = first_pair + offset / 2;
          const float even = static_cast<float>(source[offset]);
          const float odd = static_cast<float>(source[offset + 1]);
          const float cosine = cache[pair];
          const float sine = cache[rope_dim / 2 + pair];
          destination[offset] = BFloat16(even * cosine + odd * sine);
          destination[offset + 1] = BFloat16(odd * cosine - even * sine);
        }
      }
      continue;
    }

    for (int row = 0; row < panel.physical_rows; ++row) {
      const int64_t token = panel.row_begin + row;
      BFloat16* destination = block + row * 4;
      for (int offset = 0; offset < 4; ++offset) {
        const int64_t k = kb + offset;
        destination[offset] = row < panel.rows && k < logical_k
                                  ? inverse_rope_value(input, caches[row], token, group, k, heads, heads_per_group,
                                                       head_dim, nope_dim, rope_dim)
                                  : BFloat16(0.0f);
      }
    }
  }
}

BFloat16* reusable_packed_a(size_t elements) {
  thread_local std::vector<BFloat16> scratch;
  if (scratch.size() < elements) {
    scratch.resize(elements);
  }
  return scratch.data();
}

#if defined(__linux__)
class ThreadAffinityGuard {
 public:
  ThreadAffinityGuard() { valid_ = pthread_getaffinity_np(pthread_self(), sizeof(original_), &original_) == 0; }

  ~ThreadAffinityGuard() {
    if (valid_) {
      pthread_setaffinity_np(pthread_self(), sizeof(original_), &original_);
    }
  }

  void bind(int64_t cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<int>(cpu), &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  }

 private:
  cpu_set_t original_{};
  bool valid_ = false;
};
#else
class ThreadAffinityGuard {
 public:
  void bind(int64_t) {}
};
#endif

void run_grouped_gemm(const BFloat16* input, const int64_t* positions, const float* cos_sin_cache,
                      const BFloat16* packed_weight, BFloat16* output, int64_t tokens, int64_t groups,
                      int64_t heads_per_group, int64_t head_dim, int64_t rope_dim, int64_t packed_k, int64_t packed_n,
                      const int32_t* core_ids, int64_t core_count) {
  std::vector<Panel> panels = make_panels(tokens);
  std::string error;
  for (Panel& panel : panels) {
    panel.kernel = fused_cpp::moe_sve::jit::get_gemm_bf16_kernel(panel.rows, &error);
    TORCH_CHECK(panel.kernel != nullptr, "failed to generate SVE BF16 WO_A M", panel.rows, " kernel: ", error);
  }

  const int64_t physical_rows = packed_rows(panels);
  const size_t packed_a_elements = static_cast<size_t>(groups * physical_rows * packed_k);
  BFloat16* packed_a = reusable_packed_a(packed_a_elements);
  const int64_t heads = groups * heads_per_group;
  const int64_t logical_k = heads_per_group * head_dim;
  const int64_t nope_dim = head_dim - rope_dim;
  const int64_t n_tile = fused_cpp::moe_sve::n_tile();
  const int64_t n_tiles = packed_n / n_tile;
  const int64_t pack_tasks = groups * static_cast<int64_t>(panels.size());
  int64_t requested_threads = core_count;
#if defined(FUSED_CPP_HAS_OMP)
  if (requested_threads == 0) {
    requested_threads = omp_get_max_threads();
  }
#else
  requested_threads = 1;
#endif
  const fused_cpp::nm_window::Geometry geometry = fused_cpp::nm_window::Choose(
      n_tiles, packed_k * n_tile * static_cast<int64_t>(sizeof(BFloat16)), static_cast<int64_t>(panels.size()),
      requested_threads, groups, fused_cpp::nm_window::kDefaultTargetBytes, 4);
  const int64_t gemm_tasks = geometry.task_count(groups);
  const int thread_count = static_cast<int>(std::max<int64_t>(1, std::min(requested_threads, gemm_tasks)));

#if defined(FUSED_CPP_HAS_OMP)
#pragma omp parallel num_threads(thread_count)
#endif
  {
    ThreadAffinityGuard affinity;
#if defined(FUSED_CPP_HAS_OMP)
    const int thread_id = omp_get_thread_num();
#else
    const int thread_id = 0;
#endif
    if (core_ids != nullptr) {
      affinity.bind(core_ids[thread_id]);
    }

#if defined(FUSED_CPP_HAS_OMP)
#pragma omp for schedule(static)
#endif
    for (int64_t task = 0; task < pack_tasks; ++task) {
      const int64_t group = task / static_cast<int64_t>(panels.size());
      const Panel& panel = panels[task % static_cast<int64_t>(panels.size())];
      BFloat16* destination = packed_a + group * physical_rows * packed_k + panel.packed_row_begin * packed_k;
      pack_group_panel(input, positions, cos_sin_cache, rope_dim, heads, heads_per_group, head_dim, nope_dim, rope_dim,
                       logical_k, packed_k, group, panel, destination);
    }

#if defined(FUSED_CPP_HAS_OMP)
#pragma omp for schedule(static, 1)
#endif
    for (int64_t task = 0; task < gemm_tasks; ++task) {
      const int64_t m_split = task % geometry.m_splits;
      const int64_t owner = task / geometry.m_splits;
      const int64_t window = owner % geometry.n_windows;
      const int64_t group = owner / geometry.n_windows;
      const int64_t tile_begin = window * n_tiles / geometry.n_windows;
      const int64_t tile_end = (window + 1) * n_tiles / geometry.n_windows;
      const int64_t tile_count = tile_end - tile_begin;
      const int64_t n_begin = tile_begin * n_tile;
      const int64_t n_columns = tile_count * n_tile;
      const BFloat16* packed_b_group = packed_weight + group * packed_k * packed_n;
      const int64_t panel_begin = m_split * static_cast<int64_t>(panels.size()) / geometry.m_splits;
      const int64_t panel_end = (m_split + 1) * static_cast<int64_t>(panels.size()) / geometry.m_splits;
      for (int64_t panel_index = panel_begin; panel_index < panel_end; ++panel_index) {
        const Panel& panel = panels[panel_index];
        const BFloat16* packed_a_panel =
            packed_a + group * physical_rows * packed_k + panel.packed_row_begin * packed_k;
        BFloat16* output_tile = output + panel.row_begin * groups * packed_n + group * packed_n + n_begin;
        SveParams params;
        params.gemm.m = panel.rows;
        params.gemm.k = static_cast<int>(packed_k);
        params.gemm.n = static_cast<int>(n_columns);
        params.gemm.lda = static_cast<int>(packed_k);
        params.gemm.ldb = static_cast<int>(packed_k);
        params.gemm.ldc = static_cast<int>(groups * packed_n);
        params.kc = static_cast<int32_t>(packed_k);
        params.packed_n = static_cast<int32_t>(packed_n);
        params.n_begin = static_cast<int32_t>(n_begin);
        panel.kernel(reinterpret_cast<const uint16_t*>(packed_a_panel),
                     reinterpret_cast<const uint16_t*>(packed_b_group), output_tile, nullptr, &params.gemm);
      }
    }
  }
}

}  // namespace

bool deepseek_v4_inv_rope_woa_available() {
  return fused_cpp::moe_sve::available() && fused_cpp::moe_sve::jit::built();
}

std::tuple<at::Tensor, int64_t, int64_t, int64_t> deepseek_v4_inv_rope_woa_prepare(at::Tensor wo_a_weight,
                                                                                   int64_t n_groups,
                                                                                   int64_t heads_per_group,
                                                                                   int64_t head_dim, int64_t rope_dim,
                                                                                   std::string backend) {
  TORCH_CHECK(deepseek_v4_inv_rope_woa_available(), "SVE BF16 inverse-RoPE WO_A backend is unavailable");
  TORCH_CHECK(backend == "arm_sve_bf16", "unsupported inverse-RoPE WO_A backend: ", backend);
  check_cpu_contiguous(wo_a_weight, at::kBFloat16, "wo_a_weight");
  TORCH_CHECK(wo_a_weight.dim() == 2, "wo_a_weight must be [G * R, P * DH]");
  TORCH_CHECK(n_groups > 0 && heads_per_group > 0 && head_dim > 0, "group and head dimensions must be positive");
  TORCH_CHECK(rope_dim >= 0 && rope_dim <= head_dim && rope_dim % 2 == 0, "rope_dim must be even and in [0, DH]");
  TORCH_CHECK(wo_a_weight.size(0) > 0 && wo_a_weight.size(0) % n_groups == 0,
              "wo_a_weight rows must be a positive multiple of n_groups");
  TORCH_CHECK(heads_per_group <= std::numeric_limits<int>::max() / head_dim,
              "WO_A input dimension exceeds the SVE kernel limit");
  const int64_t logical_k = heads_per_group * head_dim;
  TORCH_CHECK(wo_a_weight.size(1) == logical_k, "wo_a_weight K mismatch");
  const int64_t output_rank = wo_a_weight.size(0) / n_groups;
  TORCH_CHECK(logical_k <= std::numeric_limits<int>::max() && output_rank <= std::numeric_limits<int>::max(),
              "prepared WO_A dimensions exceed the SVE kernel limit");
  const int64_t packed_k = fused_cpp::moe_sve::round_k(static_cast<int>(logical_k));
  const int64_t packed_n = fused_cpp::moe_sve::round_n(static_cast<int>(output_rank));
  TORCH_CHECK(packed_k <= std::numeric_limits<int>::max() && packed_n <= std::numeric_limits<int>::max(),
              "prepared WO_A dimensions exceed the SVE kernel limit");

  at::Tensor packed = at::empty({n_groups, packed_k * packed_n}, wo_a_weight.options());
  std::vector<BFloat16> logical(static_cast<size_t>(packed_k * packed_n), BFloat16(0.0f));
  const auto* source = wo_a_weight.data_ptr<BFloat16>();
  auto* destination = packed.data_ptr<BFloat16>();
  for (int64_t group = 0; group < n_groups; ++group) {
    std::fill(logical.begin(), logical.end(), BFloat16(0.0f));
    for (int64_t n = 0; n < output_rank; ++n) {
      const BFloat16* source_row = source + (group * output_rank + n) * logical_k;
      for (int64_t k = 0; k < logical_k; ++k) {
        logical[k * packed_n + n] = source_row[k];
      }
    }
    fused_cpp::moe_sve::pack_b(reinterpret_cast<const uint16_t*>(logical.data()),
                               reinterpret_cast<uint16_t*>(destination + group * packed_k * packed_n),
                               static_cast<int>(packed_k), static_cast<int>(packed_n));
  }
  return std::make_tuple(packed, output_rank, fused_cpp::moe_sve::n_tile(), packed_k);
}

at::Tensor deepseek_v4_inv_rope_grouped_woa(at::Tensor o, at::Tensor positions, at::Tensor cos_sin_cache,
                                            at::Tensor packed_weight, int64_t n_groups, int64_t heads_per_group,
                                            int64_t head_dim, int64_t rope_dim, int64_t output_rank,
                                            c10::optional<at::Tensor> core_ids, c10::optional<at::Tensor> out) {
  TORCH_CHECK(deepseek_v4_inv_rope_woa_available(), "SVE BF16 inverse-RoPE WO_A backend is unavailable");
  check_cpu_contiguous(o, at::kBFloat16, "o");
  check_cpu_contiguous(positions, at::kLong, "positions");
  check_cpu_contiguous(cos_sin_cache, at::kFloat, "cos_sin_cache");
  check_cpu_contiguous(packed_weight, at::kBFloat16, "packed_weight");
  TORCH_CHECK(o.dim() == 3, "o must be [T, NH, DH]");
  TORCH_CHECK(positions.dim() == 1 && positions.size(0) == o.size(0), "positions must be [T]");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.size(1) == rope_dim,
              "cos_sin_cache must be [max_position, rope_dim]");
  TORCH_CHECK(n_groups > 0 && heads_per_group > 0 && head_dim > 0 && output_rank > 0,
              "group, head, and output dimensions must be positive");
  TORCH_CHECK(rope_dim >= 0 && rope_dim <= head_dim && rope_dim % 2 == 0, "invalid rope_dim");
  TORCH_CHECK(o.size(1) == n_groups * heads_per_group && o.size(2) == head_dim, "o geometry mismatch");
  TORCH_CHECK(
      heads_per_group <= std::numeric_limits<int>::max() / head_dim && output_rank <= std::numeric_limits<int>::max(),
      "WO_A execution dimensions exceed the SVE kernel limit");
  const int64_t logical_k = heads_per_group * head_dim;
  const int64_t packed_k = fused_cpp::moe_sve::round_k(static_cast<int>(logical_k));
  const int64_t packed_n = fused_cpp::moe_sve::round_n(static_cast<int>(output_rank));
  TORCH_CHECK(packed_weight.numel() == n_groups * packed_k * packed_n, "packed_weight geometry mismatch");

  at::Tensor output = out.has_value() ? *out : at::empty({o.size(0), n_groups, output_rank}, o.options());
  check_cpu_contiguous(output, at::kBFloat16, "out");
  TORCH_CHECK(output.sizes() == at::IntArrayRef({o.size(0), n_groups, output_rank}), "out must be [T, G, R]");
  if (o.size(0) == 0) {
    return output;
  }

  const auto* position_data = positions.data_ptr<int64_t>();
  for (int64_t token = 0; token < positions.numel(); ++token) {
    TORCH_CHECK(position_data[token] >= 0 && position_data[token] < cos_sin_cache.size(0),
                "position is outside cos_sin_cache: ", position_data[token]);
  }

  const int32_t* core_data = nullptr;
  int64_t core_count = 0;
  if (core_ids.has_value()) {
    check_cpu_contiguous(*core_ids, at::kInt, "core_ids");
    TORCH_CHECK(core_ids->dim() == 1 && core_ids->numel() > 0, "core_ids must be a non-empty 1-D tensor");
    core_data = core_ids->data_ptr<int32_t>();
    core_count = core_ids->numel();
    for (int64_t index = 0; index < core_count; ++index) {
      TORCH_CHECK(core_data[index] >= 0 && core_data[index] < CPU_SETSIZE, "core_ids contains an invalid CPU id");
    }
  }

  at::Tensor padded_output =
      output_rank == packed_n ? output : at::empty({o.size(0), n_groups, packed_n}, output.options());
  run_grouped_gemm(o.data_ptr<BFloat16>(), position_data, cos_sin_cache.data_ptr<float>(),
                   packed_weight.data_ptr<BFloat16>(), padded_output.data_ptr<BFloat16>(), o.size(0), n_groups,
                   heads_per_group, head_dim, rope_dim, packed_k, packed_n, core_data, core_count);
  if (output_rank != packed_n) {
    const auto* source = padded_output.data_ptr<BFloat16>();
    auto* destination = output.data_ptr<BFloat16>();
    for (int64_t token_group = 0; token_group < o.size(0) * n_groups; ++token_group) {
      std::copy_n(source + token_group * packed_n, output_rank, destination + token_group * output_rank);
    }
  }
  return output;
}
#else
bool deepseek_v4_inv_rope_woa_available() { return false; }

std::tuple<at::Tensor, int64_t, int64_t, int64_t> deepseek_v4_inv_rope_woa_prepare(at::Tensor, int64_t, int64_t,
                                                                                   int64_t, int64_t, std::string) {
  TORCH_CHECK(false, "SVE BF16 inverse-RoPE WO_A backend is unavailable");
}

at::Tensor deepseek_v4_inv_rope_grouped_woa(at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t, int64_t, int64_t,
                                            int64_t, int64_t, c10::optional<at::Tensor>, c10::optional<at::Tensor>) {
  TORCH_CHECK(false, "SVE BF16 inverse-RoPE WO_A backend is unavailable");
}
#endif
