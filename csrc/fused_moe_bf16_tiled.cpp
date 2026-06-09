#include <torch/extension.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#ifdef __aarch64__
#include "gemm_params.h"

extern "C" {
void bf16gemm_k_ld(const uint16_t* A, const uint16_t* B_reo, float* C,
                   uint16_t* A_reorder, const gemm_params_t* params);
void bf16gemm_k_ld1(const uint16_t* A, const uint16_t* B_reo, float* C,
                    uint16_t* A_reorder, const gemm_params_t* params);
void bf16gemm_k_ld2(const uint16_t* A, const uint16_t* B_reo, float* C,
                    uint16_t* A_reorder, const gemm_params_t* params);
void bf16gemm_k_ld4(const uint16_t* A, const uint16_t* B_reo, float* C,
                    uint16_t* A_reorder, const gemm_params_t* params);
}
#endif

namespace {

constexpr int64_t kKernelTile = 8;
constexpr int64_t kMoeTokenTile = 16;

int64_t ceil_div_int64(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

int64_t ceil_to_multiple(int64_t x, int64_t multiple) {
    return ceil_div_int64(x, multiple) * multiple;
}

void check_positive_int(int64_t value, const char* name) {
    TORCH_CHECK(value > 0, name, " must be positive, got ", value);
    TORCH_CHECK(value <= std::numeric_limits<int>::max(),
                name, " exceeds int32 kernel limit: ", value);
}

void check_bf16_cpu(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16,
                name, " must have dtype torch.bfloat16");
}

bool is_integer_dtype(at::ScalarType dtype) {
    return dtype == at::kByte || dtype == at::kChar ||
           dtype == at::kShort || dtype == at::kInt ||
           dtype == at::kLong;
}

bool is_floating_dtype(at::ScalarType dtype) {
    return dtype == at::kFloat || dtype == at::kDouble ||
           dtype == at::kHalf || dtype == at::kBFloat16;
}

uint16_t* bf16_data(at::Tensor& tensor) {
    return reinterpret_cast<uint16_t*>(tensor.data_ptr<at::BFloat16>());
}

const uint16_t* bf16_data_const(const at::Tensor& tensor) {
    return reinterpret_cast<const uint16_t*>(
        tensor.data_ptr<at::BFloat16>());
}

uint16_t bf16_bits_from_float(float value) {
    c10::BFloat16 bf(value);
    uint16_t bits = 0;
    static_assert(sizeof(bits) == sizeof(bf));
    std::memcpy(&bits, &bf, sizeof(bits));
    return bits;
}

float bf16_bits_to_float(uint16_t bits) {
    uint32_t widened = static_cast<uint32_t>(bits) << 16;
    float value = 0.0f;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

#ifdef __aarch64__

void bf16_pack_b(const uint16_t* B, uint16_t* B_reo, int K, int N) {
    int64_t idx = 0;
    for (int cb = 0; cb < N / 8; ++cb) {
        for (int rb = 0; rb < K / 4; ++rb) {
            const int row_base = rb * 4;
            const int col_base = cb * 8;
            for (int cp = 0; cp < 4; ++cp) {
                const int c0 = col_base + cp * 2;
                const int c1 = c0 + 1;
                for (int i = 0; i < 4; ++i) {
                    B_reo[idx++] = B[(row_base + i) * N + c0];
                }
                for (int i = 0; i < 4; ++i) {
                    B_reo[idx++] = B[(row_base + i) * N + c1];
                }
            }
        }
    }
}

void dispatch_fp32_gemm(const uint16_t* A,
                        const uint16_t* B_reo,
                        float* C,
                        uint16_t* A_reorder,
                        int M,
                        int K,
                        int N,
                        int ldc) {
    gemm_params_t p;
    p.lda = K;
    p.ldb = K;
    p.ldc = ldc;

    int processed = 0;
    const int m_full = (M / 8) * 8;
    if (m_full > 0) {
        p.m = m_full;
        p.k = K;
        p.n = N;
        bf16gemm_k_ld(A, B_reo, C, A_reorder, &p);
        processed = m_full;
    }

    int m_rem = M - processed;
    if (m_rem == 0) {
        return;
    }

    const uint16_t* At = A + static_cast<int64_t>(processed) * K;
    float* Ct = C + static_cast<int64_t>(processed) * ldc;
    uint16_t* A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;

    if (m_rem >= 4) {
        p.m = 4;
        p.k = K;
        p.n = N;
        bf16gemm_k_ld4(At, B_reo, Ct, A_reo_t, &p);
        processed += 4;
        m_rem -= 4;
        At = A + static_cast<int64_t>(processed) * K;
        Ct = C + static_cast<int64_t>(processed) * ldc;
        A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;
    }
    if (m_rem >= 2) {
        p.m = 2;
        p.k = K;
        p.n = N;
        bf16gemm_k_ld2(At, B_reo, Ct, A_reo_t, &p);
        processed += 2;
        m_rem -= 2;
        At = A + static_cast<int64_t>(processed) * K;
        Ct = C + static_cast<int64_t>(processed) * ldc;
        A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;
    }
    if (m_rem >= 1) {
        p.m = 1;
        p.k = K;
        p.n = N;
        bf16gemm_k_ld1(At, B_reo, Ct, A_reo_t, &p);
    }
}

#endif

struct PackedExperts {
    at::Tensor tensor;
    int64_t E = 0;
    int64_t K = 0;
    int64_t N = 0;
    int64_t K_pad = 0;
    int64_t N_pad = 0;
    int64_t packed_stride = 0;
};

PackedExperts checked_packed_experts(const at::Tensor& packed,
                                     int64_t K,
                                     int64_t N,
                                     const char* name) {
    check_bf16_cpu(packed, name);
    TORCH_CHECK(packed.dim() == 2,
                name, " must be 2-D [experts, packed_numel]");
    TORCH_CHECK(packed.is_contiguous(), name, " must be contiguous");
    check_positive_int(K, "K");
    check_positive_int(N, "N");
    const int64_t K_pad = ceil_to_multiple(K, kKernelTile);
    const int64_t N_pad = ceil_to_multiple(N, kKernelTile);
    const int64_t expected_stride = K_pad * N_pad;
    TORCH_CHECK(packed.size(1) == expected_stride,
                name, " packed stride mismatch: expected ",
                expected_stride, ", got ", packed.size(1));
    TORCH_CHECK(packed.size(0) > 0, name, " must contain at least one expert");
    return PackedExperts{packed, packed.size(0), K, N, K_pad, N_pad,
                         expected_stride};
}

float read_optional_bias(const c10::optional<at::Tensor>& bias,
                         int64_t offset) {
    if (!bias.has_value() || !bias.value().defined()) {
        return 0.0f;
    }
    const at::Tensor& b = bias.value();
    if (b.scalar_type() == at::kFloat) {
        return b.data_ptr<float>()[offset];
    }
    return bf16_bits_to_float(bf16_data_const(b)[offset]);
}

void check_optional_bias(const c10::optional<at::Tensor>& bias,
                         int64_t E,
                         int64_t N,
                         const char* name) {
    if (!bias.has_value() || !bias.value().defined()) {
        return;
    }
    const at::Tensor& b = bias.value();
    TORCH_CHECK(b.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(b.scalar_type() == at::kFloat ||
                    b.scalar_type() == at::kBFloat16,
                name, " must have dtype torch.float32 or torch.bfloat16");
    TORCH_CHECK(b.dim() == 2, name, " must be 2-D [experts, dim]");
    TORCH_CHECK(b.size(0) == E && b.size(1) == N,
                name, " shape mismatch: expected [", E, ", ", N,
                "], got [", b.size(0), ", ", b.size(1), "]");
    TORCH_CHECK(b.is_contiguous(), name, " must be contiguous");
}

void apply_gate_up_bias(float* gate_up,
                        int64_t rows,
                        int64_t stride,
                        int64_t N,
                        int64_t expert,
                        const c10::optional<at::Tensor>& bias) {
    if (!bias.has_value() || !bias.value().defined()) {
        return;
    }
    const int64_t base = expert * N;
    for (int64_t m = 0; m < rows; ++m) {
        float* row = gate_up + m * stride;
        for (int64_t n = 0; n < N; ++n) {
            row[n] += read_optional_bias(bias, base + n);
        }
    }
}

void silu_and_mul_to_bf16(const float* gate_up,
                          uint16_t* intermediate,
                          int64_t rows,
                          int64_t gate_up_stride,
                          int64_t intermediate_stride,
                          int64_t F) {
    for (int64_t m = 0; m < rows; ++m) {
        const float* row = gate_up + m * gate_up_stride;
        uint16_t* out = intermediate + m * intermediate_stride;
        for (int64_t f = 0; f < F; ++f) {
            const float gate = row[f];
            const float up = row[F + f];
            const float silu = gate / (1.0f + std::exp(-gate));
            out[f] = bf16_bits_from_float(silu * up);
        }
    }
}

void gelu_and_mul_to_bf16(const float* gate_up,
                          uint16_t* intermediate,
                          int64_t rows,
                          int64_t gate_up_stride,
                          int64_t intermediate_stride,
                          int64_t F) {
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    for (int64_t m = 0; m < rows; ++m) {
        const float* row = gate_up + m * gate_up_stride;
        uint16_t* out = intermediate + m * intermediate_stride;
        for (int64_t f = 0; f < F; ++f) {
            const float gate = row[f];
            const float up = row[F + f];
            const float gelu = 0.5f * gate *
                (1.0f + std::erf(gate * kInvSqrt2));
            out[f] = bf16_bits_from_float(gelu * up);
        }
    }
}

void swigluoai_and_mul_to_bf16(const float* gate_up,
                               uint16_t* intermediate,
                               int64_t rows,
                               int64_t gate_up_stride,
                               int64_t intermediate_stride,
                               int64_t F) {
    constexpr float kAlpha = 1.702f;
    constexpr float kLimit = 7.0f;
    for (int64_t m = 0; m < rows; ++m) {
        const float* row = gate_up + m * gate_up_stride;
        uint16_t* out = intermediate + m * intermediate_stride;
        for (int64_t f = 0; f < F; ++f) {
            const float gate_raw = row[2 * f];
            const float up_raw = row[2 * f + 1];
            const float gate = std::min(gate_raw, kLimit);
            const float up = std::max(-kLimit, std::min(up_raw, kLimit));
            const float glu = gate / (1.0f + std::exp(-gate * kAlpha));
            out[f] = bf16_bits_from_float((up + 1.0f) * glu);
        }
    }
}

void activation_to_bf16(const std::string& activation,
                        const float* gate_up,
                        uint16_t* intermediate,
                        int64_t rows,
                        int64_t gate_up_stride,
                        int64_t intermediate_stride,
                        int64_t F) {
    std::fill(intermediate, intermediate + rows * intermediate_stride,
              static_cast<uint16_t>(0));
    if (activation == "silu") {
        silu_and_mul_to_bf16(gate_up, intermediate, rows, gate_up_stride,
                             intermediate_stride, F);
    } else if (activation == "gelu") {
        gelu_and_mul_to_bf16(gate_up, intermediate, rows, gate_up_stride,
                             intermediate_stride, F);
    } else if (activation == "swigluoai") {
        swigluoai_and_mul_to_bf16(gate_up, intermediate, rows,
                                  gate_up_stride, intermediate_stride, F);
    } else {
        TORCH_CHECK(false, "unsupported MoE activation: ", activation);
    }
}

struct TileTask {
    int64_t expert = 0;
    int64_t route_begin = 0;
    int64_t rows = 0;
};

struct TaskRange {
    size_t begin = 0;
    size_t end = 0;
};

struct ExpertTaskGroup {
    size_t begin = 0;
    size_t end = 0;
    int64_t rows = 0;
};

struct ThreadScheduleDebug {
    int64_t rows = 0;
    int64_t tasks = 0;
    int64_t ranges = 0;
    double ms = 0.0;
    std::vector<int64_t> experts;
    std::vector<int64_t> expert_rows;
};

int debug_schedule_level() {
    const char* value = std::getenv("FUSED_CPP_MOE_SCHEDULE_DEBUG");
    if (value == nullptr || value[0] == '\0' || value[0] == '0') {
        return 0;
    }
    const int parsed = std::atoi(value);
    return parsed <= 0 ? 1 : parsed;
}

ThreadScheduleDebug summarize_thread_schedule(
    const std::vector<TileTask>& tasks,
    const std::vector<TaskRange>& ranges) {
    ThreadScheduleDebug debug;
    debug.ranges = static_cast<int64_t>(ranges.size());
    int64_t last_expert = -1;
    for (const TaskRange& range : ranges) {
        for (size_t task_idx = range.begin; task_idx < range.end;
             ++task_idx) {
            const TileTask& task = tasks[task_idx];
            debug.rows += task.rows;
            ++debug.tasks;
            if (debug.experts.empty() || task.expert != last_expert) {
                debug.experts.push_back(task.expert);
                debug.expert_rows.push_back(0);
                last_expert = task.expert;
            }
            debug.expert_rows.back() += task.rows;
        }
    }
    return debug;
}

void print_schedule_debug_line(const char* label,
                               int64_t tid,
                               const ThreadScheduleDebug& debug) {
    std::fprintf(
        stderr,
        "[fused_moe_bf16_tiled][schedule] %s tid=%lld ms=%.3f rows=%lld "
        "tiles=%lld ranges=%lld experts=%zu experts=[",
        label,
        static_cast<long long>(tid),
        debug.ms,
        static_cast<long long>(debug.rows),
        static_cast<long long>(debug.tasks),
        static_cast<long long>(debug.ranges),
        debug.experts.size());
    for (size_t i = 0; i < debug.experts.size(); ++i) {
        if (i != 0) {
            std::fprintf(stderr, ",");
        }
        std::fprintf(stderr, "%lld:%lld",
                     static_cast<long long>(debug.experts[i]),
                     static_cast<long long>(debug.expert_rows[i]));
    }
    std::fprintf(stderr, "]\n");
}

std::vector<std::vector<TaskRange>>
split_tasks_by_expert_affinity(const std::vector<TileTask>& tasks,
                               int64_t num_threads) {
    std::vector<std::vector<TaskRange>> ranges(
        static_cast<size_t>(num_threads));
    if (tasks.empty()) {
        return ranges;
    }

    std::vector<ExpertTaskGroup> groups;
    groups.reserve(tasks.size());
    size_t task_cursor = 0;
    while (task_cursor < tasks.size()) {
        ExpertTaskGroup group;
        group.begin = task_cursor;
        const int64_t expert = tasks[task_cursor].expert;
        while (task_cursor < tasks.size() &&
               tasks[task_cursor].expert == expert) {
            group.rows += tasks[task_cursor].rows;
            ++task_cursor;
        }
        group.end = task_cursor;
        groups.push_back(group);
    }

    std::vector<size_t> order(groups.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        if (groups[lhs].rows != groups[rhs].rows) {
            return groups[lhs].rows > groups[rhs].rows;
        }
        return groups[lhs].begin < groups[rhs].begin;
    });

    std::vector<int64_t> thread_rows(static_cast<size_t>(num_threads), 0);
    for (size_t group_idx : order) {
        size_t best_tid = 0;
        for (size_t tid = 1; tid < thread_rows.size(); ++tid) {
            if (thread_rows[tid] < thread_rows[best_tid]) {
                best_tid = tid;
            }
        }
        const ExpertTaskGroup& group = groups[group_idx];
        ranges[best_tid].push_back(TaskRange{group.begin, group.end});
        thread_rows[best_tid] += group.rows;
    }

    for (std::vector<TaskRange>& thread_ranges : ranges) {
        std::sort(thread_ranges.begin(), thread_ranges.end(),
                  [](const TaskRange& lhs, const TaskRange& rhs) {
                      return lhs.begin < rhs.begin;
                  });
    }
    return ranges;
}

template <typename Fn>
void run_fixed_threads(int64_t num_threads, const Fn& fn) {
    if (num_threads <= 1) {
        fn(0);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads - 1));
    for (int64_t tid = 1; tid < num_threads; ++tid) {
        workers.emplace_back([&, tid]() { fn(tid); });
    }
    fn(0);
    for (auto& worker : workers) {
        worker.join();
    }
}

struct ThreadScratch {
    std::vector<uint16_t> input;
    std::vector<uint16_t> intermediate;
    std::vector<uint16_t> a_reorder;
    std::vector<float> gate_up;
    std::vector<float> down;
};

void pack_transposed_expert_weight(const uint16_t* weight,
                                   int64_t expert_offset,
                                   int64_t out_features,
                                   int64_t in_features,
                                   int64_t K_pad,
                                   int64_t N_pad,
                                   uint16_t* packed_dst) {
    std::vector<uint16_t> transposed(
        static_cast<size_t>(K_pad * N_pad), static_cast<uint16_t>(0));
    for (int64_t n = 0; n < out_features; ++n) {
        const uint16_t* src_row =
            weight + expert_offset + n * in_features;
        for (int64_t k = 0; k < in_features; ++k) {
            transposed[static_cast<size_t>(k * N_pad + n)] = src_row[k];
        }
    }
    bf16_pack_b(transposed.data(), packed_dst, static_cast<int>(K_pad),
                static_cast<int>(N_pad));
}

}  // namespace

std::tuple<at::Tensor, int64_t, int64_t, at::Tensor, int64_t, int64_t>
fused_moe_bf16_tiled_prepare_weights(at::Tensor w13_weight,
                                 at::Tensor w2_weight) {
#ifndef __aarch64__
    TORCH_CHECK(false, "fused_moe_bf16_tiled_prepare_weights requires AArch64");
#else
    check_bf16_cpu(w13_weight, "w13_weight");
    check_bf16_cpu(w2_weight, "w2_weight");
    TORCH_CHECK(w13_weight.dim() == 3,
                "w13_weight must be 3-D [experts, 2 * F, H]");
    TORCH_CHECK(w2_weight.dim() == 3,
                "w2_weight must be 3-D [experts, H, F]");
    TORCH_CHECK(w13_weight.size(0) == w2_weight.size(0),
                "w13_weight and w2_weight must have the same expert count");

    w13_weight = w13_weight.contiguous();
    w2_weight = w2_weight.contiguous();

    const int64_t E = w13_weight.size(0);
    const int64_t N13 = w13_weight.size(1);
    const int64_t H = w13_weight.size(2);
    TORCH_CHECK(N13 % 2 == 0, "w13 output dim must be even, got ", N13);
    const int64_t F = N13 / 2;
    TORCH_CHECK(w2_weight.size(1) == H && w2_weight.size(2) == F,
                "w2_weight shape mismatch: expected [", E, ", ", H,
                ", ", F, "], got [", w2_weight.size(0), ", ",
                w2_weight.size(1), ", ", w2_weight.size(2), "]");

    check_positive_int(H, "hidden size");
    check_positive_int(F, "ffn hidden size");
    check_positive_int(N13, "w13 output size");

    const int64_t K13 = H;
    const int64_t K2 = F;
    const int64_t N2 = H;
    const int64_t K13_pad = ceil_to_multiple(K13, kKernelTile);
    const int64_t N13_pad = ceil_to_multiple(N13, kKernelTile);
    const int64_t K2_pad = ceil_to_multiple(K2, kKernelTile);
    const int64_t N2_pad = ceil_to_multiple(N2, kKernelTile);

    at::Tensor w13_packed = at::empty(
        {E, K13_pad * N13_pad}, w13_weight.options());
    at::Tensor w2_packed = at::empty(
        {E, K2_pad * N2_pad}, w2_weight.options());

    const uint16_t* w13_ptr = bf16_data_const(w13_weight);
    const uint16_t* w2_ptr = bf16_data_const(w2_weight);
    uint16_t* w13_packed_ptr = bf16_data(w13_packed);
    uint16_t* w2_packed_ptr = bf16_data(w2_packed);

    const int64_t w13_expert_stride = N13 * H;
    const int64_t w2_expert_stride = H * F;
    for (int64_t e = 0; e < E; ++e) {
        pack_transposed_expert_weight(
            w13_ptr, e * w13_expert_stride, N13, H, K13_pad, N13_pad,
            w13_packed_ptr + e * K13_pad * N13_pad);
        pack_transposed_expert_weight(
            w2_ptr, e * w2_expert_stride, H, F, K2_pad, N2_pad,
            w2_packed_ptr + e * K2_pad * N2_pad);
    }

    return std::make_tuple(w13_packed, K13, N13, w2_packed, K2, N2);
#endif
}

at::Tensor fused_moe_bf16_tiled(at::Tensor input,
                            at::Tensor w13_packed,
                            int64_t w13_K,
                            int64_t w13_N,
                            at::Tensor w2_packed,
                            int64_t w2_K,
                            int64_t w2_N,
                            at::Tensor topk_weights,
                            at::Tensor topk_ids,
                            c10::optional<at::Tensor> w13_bias,
                            c10::optional<at::Tensor> w2_bias,
                            int64_t num_threads,
                            std::string activation,
                            int64_t global_num_experts,
                            bool skip_weighted) {
#ifndef __aarch64__
    TORCH_CHECK(false, "fused_moe_bf16_tiled requires AArch64");
#else
    check_bf16_cpu(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must be 2-D [tokens, hidden]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(topk_ids.device().is_cpu(), "topk_ids must be CPU");
    TORCH_CHECK(topk_weights.device().is_cpu(), "topk_weights must be CPU");
    TORCH_CHECK(is_integer_dtype(topk_ids.scalar_type()),
                "topk_ids must use an integer dtype");
    TORCH_CHECK(is_floating_dtype(topk_weights.scalar_type()),
                "topk_weights must use a floating dtype");
    TORCH_CHECK(topk_ids.dim() == 2, "topk_ids must be 2-D [tokens, top_k]");
    TORCH_CHECK(topk_weights.dim() == 2,
                "topk_weights must be 2-D [tokens, top_k]");
    TORCH_CHECK(topk_ids.sizes() == topk_weights.sizes(),
                "topk_ids and topk_weights shapes must match");
    TORCH_CHECK(topk_ids.size(0) == input.size(0),
                "topk first dimension must match input token count");
    TORCH_CHECK(topk_ids.size(1) > 0, "top_k must be non-zero");
    TORCH_CHECK(num_threads > 0, "num_threads must be positive, got ",
                num_threads);
    TORCH_CHECK(num_threads <= std::numeric_limits<int>::max(),
                "num_threads exceeds int32 limit: ", num_threads);

    PackedExperts w13 = checked_packed_experts(w13_packed, w13_K, w13_N,
                                               "w13_packed");
    PackedExperts w2 = checked_packed_experts(w2_packed, w2_K, w2_N,
                                             "w2_packed");
    TORCH_CHECK(w13.E == w2.E, "w13 and w2 expert count mismatch");
    TORCH_CHECK(w13.K == input.size(1),
                "input hidden size mismatch: input H=", input.size(1),
                ", w13 K=", w13.K);
    TORCH_CHECK(w13.N % 2 == 0, "w13 N must be even, got ", w13.N);
    const int64_t F = w13.N / 2;
    const int64_t H = input.size(1);
    TORCH_CHECK(w2.K == F && w2.N == H,
                "w2 shape mismatch: expected K=", F, " N=", H,
                ", got K=", w2.K, " N=", w2.N);
    check_optional_bias(w13_bias, w13.E, w13.N, "w13_bias");
    check_optional_bias(w2_bias, w2.E, w2.N, "w2_bias");

    const int64_t num_tokens = input.size(0);
    const int64_t top_k = topk_ids.size(1);
    if (skip_weighted) {
        TORCH_CHECK(top_k == 1,
                    "skip_weighted is only valid when top_k == 1");
    }
    if (num_tokens == 0) {
        return at::empty_like(input);
    }

    const int64_t num_experts = global_num_experts < 0
                                    ? w13.E
                                    : global_num_experts;
    TORCH_CHECK(num_experts > 0,
                "global_num_experts must be positive or -1, got ",
                global_num_experts);
    TORCH_CHECK(num_experts <= w13.E,
                "global_num_experts cannot exceed prepared expert weights: ",
                num_experts, " > ", w13.E);

    at::Tensor ids_i64 = topk_ids.to(at::kLong).contiguous();
    at::Tensor weights_f32 = topk_weights.to(at::kFloat).contiguous();
    const int64_t* ids = ids_i64.data_ptr<int64_t>();
    const int64_t num_routes = num_tokens * top_k;

    std::vector<std::vector<int64_t>> routes(
        static_cast<size_t>(num_experts));
    for (int64_t flat = 0; flat < num_routes; ++flat) {
        const int64_t expert = ids[flat];
        TORCH_CHECK(expert >= 0 && expert < num_experts,
                    "topk_ids out of range: id=", expert,
                    ", valid range [0, ", num_experts, ")");
        routes[static_cast<size_t>(expert)].push_back(flat);
    }

    std::vector<TileTask> tasks;
    tasks.reserve(static_cast<size_t>(
        ceil_div_int64(num_routes, kMoeTokenTile) + num_experts));
    for (int64_t e = 0; e < num_experts; ++e) {
        const auto& expert_routes = routes[static_cast<size_t>(e)];
        for (int64_t begin = 0;
             begin < static_cast<int64_t>(expert_routes.size());
             begin += kMoeTokenTile) {
            const int64_t rows = std::min<int64_t>(
                kMoeTokenTile,
                static_cast<int64_t>(expert_routes.size()) - begin);
            tasks.push_back(TileTask{e, begin, rows});
        }
    }

    at::Tensor route_out = at::empty(
        {num_routes, H},
        at::TensorOptions().device(input.device()).dtype(at::kFloat));
    float* route_out_ptr = route_out.data_ptr<float>();
    const uint16_t* input_ptr = bf16_data_const(input);
    const uint16_t* w13_ptr = bf16_data_const(w13.tensor);
    const uint16_t* w2_ptr = bf16_data_const(w2.tensor);

    const int64_t actual_threads = num_threads;
    std::vector<std::vector<TaskRange>> ranges =
        split_tasks_by_expert_affinity(tasks, actual_threads);
    const int schedule_debug_level = debug_schedule_level();
    std::vector<ThreadScheduleDebug> schedule_debug;
    if (schedule_debug_level > 0) {
        schedule_debug.reserve(static_cast<size_t>(actual_threads));
        for (const std::vector<TaskRange>& thread_ranges : ranges) {
            schedule_debug.push_back(
                summarize_thread_schedule(tasks, thread_ranges));
        }
    }

    std::vector<ThreadScratch> scratches(static_cast<size_t>(actual_threads));
    for (ThreadScratch& scratch : scratches) {
        scratch.input.resize(static_cast<size_t>(kMoeTokenTile * w13.K_pad));
        scratch.intermediate.resize(
            static_cast<size_t>(kMoeTokenTile * w2.K_pad));
        scratch.a_reorder.resize(static_cast<size_t>(
            kMoeTokenTile * std::max(w13.K_pad, w2.K_pad) * 2));
        scratch.gate_up.resize(
            static_cast<size_t>(kMoeTokenTile * w13.N_pad));
        scratch.down.resize(static_cast<size_t>(kMoeTokenTile * w2.N_pad));
    }

    run_fixed_threads(actual_threads, [&](int64_t tid) {
        const auto thread_begin = std::chrono::steady_clock::now();
        ThreadScratch& scratch = scratches[static_cast<size_t>(tid)];
        const std::vector<TaskRange>& thread_ranges =
            ranges[static_cast<size_t>(tid)];
        for (const TaskRange& range : thread_ranges) {
            for (size_t task_idx = range.begin; task_idx < range.end;
                 ++task_idx) {
                const TileTask& task = tasks[task_idx];
                const auto& expert_routes = routes[static_cast<size_t>(
                    task.expert)];
                const int64_t rows = task.rows;

                std::fill(scratch.input.begin(),
                          scratch.input.begin() + rows * w13.K_pad,
                          static_cast<uint16_t>(0));
                for (int64_t m = 0; m < rows; ++m) {
                    const int64_t flat =
                        expert_routes[static_cast<size_t>(
                            task.route_begin + m)];
                    const int64_t token = flat / top_k;
                    const uint16_t* src = input_ptr + token * H;
                    uint16_t* dst = scratch.input.data() + m * w13.K_pad;
                    std::copy(src, src + H, dst);
                }

                std::fill(scratch.gate_up.begin(),
                          scratch.gate_up.begin() + rows * w13.N_pad, 0.0f);
                dispatch_fp32_gemm(
                    scratch.input.data(),
                    w13_ptr + task.expert * w13.packed_stride,
                    scratch.gate_up.data(),
                    scratch.a_reorder.data(),
                    static_cast<int>(rows),
                    static_cast<int>(w13.K_pad),
                    static_cast<int>(w13.N_pad),
                    static_cast<int>(w13.N_pad));
                apply_gate_up_bias(scratch.gate_up.data(), rows, w13.N_pad,
                                   w13.N, task.expert, w13_bias);

                activation_to_bf16(
                    activation, scratch.gate_up.data(),
                    scratch.intermediate.data(), rows, w13.N_pad, w2.K_pad,
                    F);

                std::fill(scratch.down.begin(),
                          scratch.down.begin() + rows * w2.N_pad, 0.0f);
                dispatch_fp32_gemm(
                    scratch.intermediate.data(),
                    w2_ptr + task.expert * w2.packed_stride,
                    scratch.down.data(),
                    scratch.a_reorder.data(),
                    static_cast<int>(rows),
                    static_cast<int>(w2.K_pad),
                    static_cast<int>(w2.N_pad),
                    static_cast<int>(w2.N_pad));

                for (int64_t m = 0; m < rows; ++m) {
                    const int64_t flat =
                        expert_routes[static_cast<size_t>(
                            task.route_begin + m)];
                    float* dst = route_out_ptr + flat * H;
                    const float* src = scratch.down.data() + m * w2.N_pad;
                    const int64_t bias_base = task.expert * H;
                    for (int64_t h = 0; h < H; ++h) {
                        dst[h] = src[h] + read_optional_bias(w2_bias,
                                                             bias_base + h);
                    }
                }
            }
        }
        if (schedule_debug_level > 0) {
            const auto thread_end = std::chrono::steady_clock::now();
            schedule_debug[static_cast<size_t>(tid)].ms =
                std::chrono::duration<double, std::milli>(
                    thread_end - thread_begin).count();
        }
    });

    if (schedule_debug_level > 0) {
        int64_t longest_tid = -1;
        int64_t shortest_tid = -1;
        for (int64_t tid = 0; tid < actual_threads; ++tid) {
            const ThreadScheduleDebug& debug =
                schedule_debug[static_cast<size_t>(tid)];
            if (debug.rows == 0) {
                continue;
            }
            if (longest_tid < 0 ||
                debug.ms > schedule_debug[static_cast<size_t>(
                               longest_tid)].ms) {
                longest_tid = tid;
            }
            if (shortest_tid < 0 ||
                debug.ms < schedule_debug[static_cast<size_t>(
                               shortest_tid)].ms) {
                shortest_tid = tid;
            }
        }
        std::fprintf(
            stderr,
            "[fused_moe_bf16_tiled][schedule] threads=%lld experts=%lld "
            "routes=%lld tiles=%zu strategy=expert_affinity_greedy\n",
            static_cast<long long>(actual_threads),
            static_cast<long long>(num_experts),
            static_cast<long long>(num_routes),
            tasks.size());
        if (longest_tid >= 0) {
            print_schedule_debug_line(
                "longest", longest_tid,
                schedule_debug[static_cast<size_t>(longest_tid)]);
        }
        if (shortest_tid >= 0) {
            print_schedule_debug_line(
                "shortest", shortest_tid,
                schedule_debug[static_cast<size_t>(shortest_tid)]);
        }
        if (schedule_debug_level >= 2) {
            for (int64_t tid = 0; tid < actual_threads; ++tid) {
                print_schedule_debug_line(
                    "thread", tid,
                    schedule_debug[static_cast<size_t>(tid)]);
            }
        }
    }

    at::Tensor output_acc = at::zeros(
        {num_tokens, H},
        at::TensorOptions().device(input.device()).dtype(at::kFloat));
    float* output_ptr = output_acc.data_ptr<float>();
    const float* topk_w = weights_f32.data_ptr<float>();

    run_fixed_threads(actual_threads, [&](int64_t tid) {
        const int64_t rows_per_thread =
            ceil_div_int64(num_tokens, actual_threads);
        const int64_t token_begin = tid * rows_per_thread;
        const int64_t token_end = std::min<int64_t>(
            num_tokens, token_begin + rows_per_thread);
        for (int64_t token = token_begin; token < token_end; ++token) {
            float* dst = output_ptr + token * H;
            if (skip_weighted) {
                const float* src = route_out_ptr + token * H;
                std::copy(src, src + H, dst);
                continue;
            }
            for (int64_t slot = 0; slot < top_k; ++slot) {
                const int64_t flat = token * top_k + slot;
                const float weight = topk_w[flat];
                const float* src = route_out_ptr + flat * H;
                for (int64_t h = 0; h < H; ++h) {
                    dst[h] += src[h] * weight;
                }
            }
        }
    });

    return output_acc.to(at::kBFloat16);
#endif
}
