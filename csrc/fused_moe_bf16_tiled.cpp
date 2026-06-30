#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#ifdef __aarch64__
#include "gemm_params.h"
#include "profile_utils.h"

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

struct SplitRange {
    int64_t begin = 0;
    int64_t size = 0;
};

SplitRange split_evenly(int64_t units, int64_t group_size, int64_t local_tid) {
    if (units <= 0 || group_size <= 0 || local_tid < 0 ||
        local_tid >= group_size) {
        return SplitRange{};
    }
    const int64_t units_per_thread = units / group_size;
    const int64_t extra_units = units % group_size;
    if (local_tid < extra_units) {
        return SplitRange{
            local_tid * (units_per_thread + 1),
            units_per_thread + 1};
    }
    return SplitRange{
        extra_units * (units_per_thread + 1) +
            (local_tid - extra_units) * units_per_thread,
        units_per_thread};
}

SplitRange m_split_range(int M, int64_t group_size, int64_t local_tid) {
    return split_evenly(static_cast<int64_t>(M), group_size, local_tid);
}

SplitRange n_split_range(int N, int64_t group_size, int64_t local_tid) {
    const SplitRange block_range = split_evenly(
        static_cast<int64_t>(N) / kKernelTile, group_size, local_tid);
    return SplitRange{
        block_range.begin * kKernelTile,
        block_range.size * kKernelTile};
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

void dispatch_fp32_gemm_n_split(const uint16_t* A,
                                const uint16_t* B_reo,
                                float* C,
                                uint16_t* A_reorder,
                                int M,
                                int K,
                                int N,
                                int ldc,
                                int64_t group_size,
                                int64_t local_tid) {
    const SplitRange range = n_split_range(N, group_size, local_tid);
    if (range.size <= 0) {
        return;
    }

    const int64_t start_block = range.begin / kKernelTile;
    const uint16_t* B_slice = B_reo + start_block * K * kKernelTile;
    float* C_slice = C + range.begin;
    dispatch_fp32_gemm(A, B_slice, C_slice, A_reorder, M, K,
                       static_cast<int>(range.size), ldc);
}

void dispatch_fp32_gemm_m_split(const uint16_t* A,
                                const uint16_t* B_reo,
                                float* C,
                                uint16_t* A_reorder,
                                int M,
                                int K,
                                int N,
                                int ldc,
                                int64_t group_size,
                                int64_t local_tid) {
    const SplitRange range = m_split_range(M, group_size, local_tid);
    if (range.size <= 0) {
        return;
    }

    const uint16_t* A_slice = A + range.begin * K;
    float* C_slice = C + range.begin * ldc;
    dispatch_fp32_gemm(A_slice, B_reo, C_slice, A_reorder,
                       static_cast<int>(range.size), K, N, ldc);
}

void dispatch_fp32_gemm_auto_split(const uint16_t* A,
                                   const uint16_t* B_reo,
                                   float* C,
                                   uint16_t* A_reorder,
                                   int M,
                                   int K,
                                   int N,
                                   int ldc,
                                   int64_t group_size,
                                   int64_t local_tid) {
    if (M > N) {
        dispatch_fp32_gemm_m_split(A, B_reo, C, A_reorder, M, K, N, ldc,
                                   group_size, local_tid);
        return;
    }
    dispatch_fp32_gemm_n_split(A, B_reo, C, A_reorder, M, K, N, ldc,
                               group_size, local_tid);
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

void apply_gate_up_bias_range(float* gate_up,
                              int64_t row_begin,
                              int64_t rows,
                              int64_t stride,
                              int64_t N,
                              int64_t expert,
                              const c10::optional<at::Tensor>& bias) {
    apply_gate_up_bias(gate_up + row_begin * stride, rows, stride, N, expert,
                       bias);
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

void activation_range_to_bf16(const std::string& activation,
                              const float* gate_up,
                              uint16_t* intermediate,
                              int64_t row_begin,
                              int64_t rows,
                              int64_t gate_up_stride,
                              int64_t intermediate_stride,
                              int64_t F) {
    if (rows <= 0) {
        return;
    }
    const float* gate_up_slice = gate_up + row_begin * gate_up_stride;
    uint16_t* intermediate_slice =
        intermediate + row_begin * intermediate_stride;
    std::fill(intermediate_slice,
              intermediate_slice + rows * intermediate_stride,
              static_cast<uint16_t>(0));
    if (activation == "silu") {
        silu_and_mul_to_bf16(gate_up_slice, intermediate_slice, rows,
                             gate_up_stride, intermediate_stride, F);
    } else if (activation == "gelu") {
        gelu_and_mul_to_bf16(gate_up_slice, intermediate_slice, rows,
                             gate_up_stride, intermediate_stride, F);
    } else if (activation == "swigluoai") {
        swigluoai_and_mul_to_bf16(gate_up_slice, intermediate_slice, rows,
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

class ThreadBarrier {
public:
    explicit ThreadBarrier(int64_t participants)
        : participants_(participants), count_(participants) {}

    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        const int64_t generation = generation_;
        --count_;
        if (count_ == 0) {
            ++generation_;
            count_ = participants_;
            cv_.notify_all();
            return;
        }
        cv_.wait(lock, [&]() { return generation_ != generation; });
    }

private:
    int64_t participants_;
    int64_t count_;
    int64_t generation_ = 0;
    std::mutex mutex_;
    std::condition_variable cv_;
};

struct HierarchicalNSplitConfig {
    bool enabled = false;
    int64_t partitions = 2;
    int64_t groups_per_partition = 4;
    std::vector<int64_t> core_bases{0, 40};
};

struct MoeTraceConfig {
    bool enabled = false;
    std::string path = "/tmp/fused_cpp_moe_bf16_tiled_trace.log";
};

struct MoeGemmTraceRecord {
    uint64_t seq = 0;
    int64_t tid = -1;
    int64_t group = -1;
    int64_t local_tid = -1;
    int64_t expert = -1;
    int64_t route_begin = 0;
    int64_t rows = 0;
    const char* stage = "";
    int64_t M = 0;
    int64_t K = 0;
    int64_t N = 0;
    int64_t ldc = 0;
    int64_t n_begin = 0;
    int64_t n_cols = 0;
    double ms = 0.0;
};

struct HierarchicalGroupScratch {
    explicit HierarchicalGroupScratch(int64_t group_size)
        : barrier(group_size) {}

    std::vector<uint16_t> input;
    std::vector<uint16_t> intermediate;
    std::vector<uint16_t> a_reorder;
    std::vector<float> gate_up;
    std::vector<float> down;
    std::atomic<int64_t> current_expert{-1};
    ThreadBarrier barrier;
};

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool env_has_value(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
}

int64_t env_int_or_default(const char* name, int64_t fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long long parsed = std::strtoll(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return static_cast<int64_t>(parsed);
}

std::vector<int64_t> env_int_list_or_default(
    const char* name,
    const std::vector<int64_t>& fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }

    std::vector<int64_t> result;
    const char* cursor = value;
    while (*cursor != '\0') {
        while (*cursor == ',' || *cursor == ' ' || *cursor == '\t') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        char* end = nullptr;
        const long long parsed = std::strtoll(cursor, &end, 10);
        if (end == cursor) {
            return fallback;
        }
        result.push_back(static_cast<int64_t>(parsed));
        cursor = end;
    }
    return result.empty() ? fallback : result;
}

int64_t current_affinity_first_cpu() {
#ifdef __linux__
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &cpuset)) {
                return static_cast<int64_t>(cpu);
            }
        }
    }
#endif
    return 0;
}

std::vector<int64_t> relative_core_bases_from_affinity(int64_t core_skip) {
    const int64_t first_cpu = current_affinity_first_cpu();
    return std::vector<int64_t>{first_cpu, first_cpu + core_skip};
}

HierarchicalNSplitConfig hierarchical_nsplit_config_from_env() {
    HierarchicalNSplitConfig config;
    config.enabled = env_flag_enabled("FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT");
    config.groups_per_partition = env_int_or_default(
        "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION", 4);
    if (env_has_value("FUSED_CPP_MOE_N_SPLIT_CORE_BASES")) {
        config.core_bases = env_int_list_or_default(
            "FUSED_CPP_MOE_N_SPLIT_CORE_BASES",
            std::vector<int64_t>{0, 40});
    } else if (env_has_value("FUSED_CPP_MOE_N_SPLIT_CORE_SKIP")) {
        const int64_t core_skip = env_int_or_default(
            "FUSED_CPP_MOE_N_SPLIT_CORE_SKIP", 40);
        TORCH_CHECK(core_skip > 0,
                    "FUSED_CPP_MOE_N_SPLIT_CORE_SKIP must be positive, got ",
                    core_skip);
        config.core_bases = relative_core_bases_from_affinity(core_skip);
    }
    config.partitions = static_cast<int64_t>(config.core_bases.size());
    return config;
}

MoeTraceConfig moe_trace_config_from_env() {
    MoeTraceConfig config;
    config.enabled = env_flag_enabled("FUSED_CPP_MOE_TRACE");
    const char* path = std::getenv("FUSED_CPP_MOE_TRACE_FILE");
    if (path != nullptr && path[0] != '\0') {
        config.path = path;
    }
    return config;
}

std::atomic<uint64_t> g_moe_trace_call_id{0};

class MoeTraceCollector {
public:
    explicit MoeTraceCollector(MoeTraceConfig config)
        : config_(std::move(config)),
          call_id_(config_.enabled
                       ? g_moe_trace_call_id.fetch_add(
                             uint64_t{1}, std::memory_order_relaxed)
                       : 0) {}

    bool enabled() const {
        return config_.enabled;
    }

    void reserve(size_t count) {
        if (!enabled()) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        records_.reserve(count);
    }

    void record_gemm(int64_t tid,
                     int64_t group,
                     int64_t local_tid,
                     int64_t expert,
                     int64_t route_begin,
                     int64_t rows,
                     const char* stage,
                     int64_t M,
                     int64_t K,
                     int64_t N,
                     int64_t ldc,
                     int64_t n_begin,
                     int64_t n_cols,
                     double ms) {
        if (!enabled()) {
            return;
        }
        MoeGemmTraceRecord record;
        record.seq = next_seq_.fetch_add(uint64_t{1},
                                         std::memory_order_relaxed);
        record.tid = tid;
        record.group = group;
        record.local_tid = local_tid;
        record.expert = expert;
        record.route_begin = route_begin;
        record.rows = rows;
        record.stage = stage;
        record.M = M;
        record.K = K;
        record.N = N;
        record.ldc = ldc;
        record.n_begin = n_begin;
        record.n_cols = n_cols;
        record.ms = ms;
        std::lock_guard<std::mutex> lock(mutex_);
        records_.push_back(record);
    }

    void write_report(const char* strategy,
                      int64_t num_threads,
                      int64_t num_tokens,
                      int64_t top_k,
                      int64_t num_experts,
                      int64_t num_routes,
                      int64_t hidden_size,
                      int64_t ffn_hidden_size,
                      size_t micro_tiles,
                      int64_t expert_tasks,
                      int64_t nsplit_groups,
                      int64_t nsplit_group_size,
                      double e2e_ms) {
        if (!enabled()) {
            return;
        }

        std::vector<MoeGemmTraceRecord> records;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            records = records_;
        }
        std::sort(records.begin(), records.end(),
                  [](const MoeGemmTraceRecord& lhs,
                     const MoeGemmTraceRecord& rhs) {
                      return lhs.seq < rhs.seq;
                  });

        std::FILE* file = std::fopen(config_.path.c_str(), "a");
        if (file == nullptr) {
            return;
        }
        std::fprintf(
            file,
            "MOE_CALL call_id=%llu strategy=%s e2e_ms=%.6f "
            "threads=%lld tokens=%lld top_k=%lld experts=%lld routes=%lld "
            "H=%lld F=%lld micro_tiles=%zu expert_tasks=%lld "
            "nsplit_groups=%lld nsplit_group_size=%lld gemm_count=%zu\n",
            static_cast<unsigned long long>(call_id_),
            strategy,
            e2e_ms,
            static_cast<long long>(num_threads),
            static_cast<long long>(num_tokens),
            static_cast<long long>(top_k),
            static_cast<long long>(num_experts),
            static_cast<long long>(num_routes),
            static_cast<long long>(hidden_size),
            static_cast<long long>(ffn_hidden_size),
            micro_tiles,
            static_cast<long long>(expert_tasks),
            static_cast<long long>(nsplit_groups),
            static_cast<long long>(nsplit_group_size),
            records.size());
        std::fprintf(
            file,
            "GEMM_FIELDS call_id seq tid group local_tid expert route_begin "
            "rows stage M K N ldc n_begin n_cols ms\n");
        for (const MoeGemmTraceRecord& record : records) {
            std::fprintf(
                file,
                "GEMM call_id=%llu seq=%llu tid=%lld group=%lld "
                "local_tid=%lld expert=%lld route_begin=%lld rows=%lld "
                "stage=%s M=%lld K=%lld N=%lld ldc=%lld n_begin=%lld "
                "n_cols=%lld ms=%.6f\n",
                static_cast<unsigned long long>(call_id_),
                static_cast<unsigned long long>(record.seq),
                static_cast<long long>(record.tid),
                static_cast<long long>(record.group),
                static_cast<long long>(record.local_tid),
                static_cast<long long>(record.expert),
                static_cast<long long>(record.route_begin),
                static_cast<long long>(record.rows),
                record.stage,
                static_cast<long long>(record.M),
                static_cast<long long>(record.K),
                static_cast<long long>(record.N),
                static_cast<long long>(record.ldc),
                static_cast<long long>(record.n_begin),
                static_cast<long long>(record.n_cols),
                record.ms);
        }
        std::fprintf(file, "MOE_CALL_END call_id=%llu\n",
                     static_cast<unsigned long long>(call_id_));
        std::fclose(file);
    }

private:
    MoeTraceConfig config_;
    uint64_t call_id_ = 0;
    std::atomic<uint64_t> next_seq_{0};
    std::mutex mutex_;
    std::vector<MoeGemmTraceRecord> records_;
};

#ifdef __aarch64__

void trace_dispatch_fp32_gemm(MoeTraceCollector& trace,
                              const char* stage,
                              int64_t tid,
                              int64_t group,
                              int64_t local_tid,
                              int64_t expert,
                              int64_t route_begin,
                              int64_t rows,
                              const uint16_t* A,
                              const uint16_t* B_reo,
                              float* C,
                              uint16_t* A_reorder,
                              int M,
                              int K,
                              int N,
                              int ldc) {
    if (!trace.enabled()) {
        dispatch_fp32_gemm(A, B_reo, C, A_reorder, M, K, N, ldc);
        return;
    }
    const auto begin = ::fused_cpp::profile::now();
    dispatch_fp32_gemm(A, B_reo, C, A_reorder, M, K, N, ldc);
    const double ms = ::fused_cpp::profile::elapsed_ms(begin);
    trace.record_gemm(tid, group, local_tid, expert, route_begin, rows,
                      stage, M, K, N, ldc, 0, N, ms);
}

void trace_dispatch_fp32_gemm_auto_split(MoeTraceCollector& trace,
                                         const char* stage,
                                         int64_t tid,
                                         int64_t group,
                                         int64_t local_tid,
                                         int64_t expert,
                                         int64_t route_begin,
                                         int64_t rows,
                                         const uint16_t* A,
                                         const uint16_t* B_reo,
                                         float* C,
                                         uint16_t* A_reorder,
                                         int M,
                                         int K,
                                         int N,
                                         int ldc,
                                         int64_t group_size) {
    const bool split_m = M > N;
    const SplitRange range = split_m
                                 ? m_split_range(M, group_size, local_tid)
                                 : n_split_range(N, group_size, local_tid);
    if (range.size <= 0) {
        return;
    }
    if (!trace.enabled()) {
        dispatch_fp32_gemm_auto_split(A, B_reo, C, A_reorder, M, K, N,
                                      ldc, group_size, local_tid);
        return;
    }

    const auto begin = ::fused_cpp::profile::now();
    int64_t trace_route_begin = route_begin;
    int64_t trace_rows = rows;
    int64_t trace_n_begin = 0;
    int64_t trace_n_cols = N;
    if (split_m) {
        const uint16_t* A_slice = A + range.begin * K;
        float* C_slice = C + range.begin * ldc;
        dispatch_fp32_gemm(A_slice, B_reo, C_slice, A_reorder,
                           static_cast<int>(range.size), K, N, ldc);
        trace_route_begin += range.begin;
        trace_rows = range.size;
    } else {
        const int64_t start_block = range.begin / kKernelTile;
        const uint16_t* B_slice = B_reo + start_block * K * kKernelTile;
        float* C_slice = C + range.begin;
        dispatch_fp32_gemm(A, B_slice, C_slice, A_reorder, M, K,
                           static_cast<int>(range.size), ldc);
        trace_n_begin = range.begin;
        trace_n_cols = range.size;
    }
    const double ms = ::fused_cpp::profile::elapsed_ms(begin);
    trace.record_gemm(tid, group, local_tid, expert, trace_route_begin,
                      trace_rows, stage, M, K, N, ldc, trace_n_begin,
                      trace_n_cols, ms);
}

#endif

int bind_current_thread_to_core(int64_t core) {
#ifdef __linux__
    if (core < 0 || core >= CPU_SETSIZE) {
        return EINVAL;
    }
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(static_cast<int>(core), &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#else
    (void)core;
    return 0;
#endif
}

void report_affinity_bind_error(int64_t tid, int64_t core, int error) {
    if (error == 0) {
        return;
    }
    std::fprintf(
        stderr,
        "[fused_moe_bf16_tiled][affinity] failed to bind tid=%lld "
        "to core=%lld error=%d (%s)\n",
        static_cast<long long>(tid),
        static_cast<long long>(core),
        error,
        std::strerror(error));
}

class ThreadAffinityGuard {
public:
    ThreadAffinityGuard() {
#ifdef __linux__
        valid_ = pthread_getaffinity_np(
            pthread_self(), sizeof(original_), &original_) == 0;
#endif
    }

    ~ThreadAffinityGuard() {
#ifdef __linux__
        if (valid_) {
            pthread_setaffinity_np(
                pthread_self(), sizeof(original_), &original_);
        }
#endif
    }

private:
#ifdef __linux__
    cpu_set_t original_;
    bool valid_ = false;
#endif
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

bool disable_resident_threads() {
    return env_flag_enabled("FUSED_CPP_MOE_DISABLE_RESIDENT_THREADS");
}

template <typename Fn>
void run_fixed_threads_spawn(int64_t num_threads, const Fn& fn) {
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

class ResidentThreadPool {
public:
    ResidentThreadPool() = default;

    ResidentThreadPool(const ResidentThreadPool&) = delete;
    ResidentThreadPool& operator=(const ResidentThreadPool&) = delete;

    ~ResidentThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        start_cv_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template <typename Fn>
    void run(int64_t num_threads, const Fn& fn) {
        if (num_threads <= 1) {
            fn(0);
            return;
        }

        std::function<void(int64_t)> job = [&](int64_t tid) { fn(tid); };
        std::exception_ptr main_exception = nullptr;
        std::exception_ptr worker_exception = nullptr;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            TORCH_CHECK(!job_active_,
                        "resident MoE thread pool does not support nested jobs");
            ensure_workers_locked(num_threads - 1);
            current_job_ = &job;
            requested_threads_ = num_threads;
            remaining_workers_ = num_threads - 1;
            worker_exception_ = nullptr;
            job_active_ = true;
            ++generation_;
        }
        start_cv_.notify_all();

        try {
            fn(0);
        } catch (...) {
            main_exception = std::current_exception();
        }

        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_cv_.wait(lock, [&]() { return remaining_workers_ == 0; });
            worker_exception = worker_exception_;
            current_job_ = nullptr;
            requested_threads_ = 0;
            job_active_ = false;
        }

        if (main_exception != nullptr) {
            std::rethrow_exception(main_exception);
        }
        if (worker_exception != nullptr) {
            std::rethrow_exception(worker_exception);
        }
    }

private:
    void ensure_workers_locked(int64_t worker_count) {
        while (static_cast<int64_t>(workers_.size()) < worker_count) {
            const int64_t tid = static_cast<int64_t>(workers_.size()) + 1;
            workers_.emplace_back([this, tid]() { worker_loop(tid); });
        }
    }

    void worker_loop(int64_t tid) {
        int64_t seen_generation = 0;
        while (true) {
            std::function<void(int64_t)>* job = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                start_cv_.wait(lock, [&]() {
                    return stopping_ || generation_ != seen_generation;
                });
                if (stopping_) {
                    return;
                }
                seen_generation = generation_;
                if (tid >= requested_threads_) {
                    continue;
                }
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

    std::mutex mutex_;
    std::condition_variable start_cv_;
    std::condition_variable done_cv_;
    std::vector<std::thread> workers_;
    std::function<void(int64_t)>* current_job_ = nullptr;
    std::exception_ptr worker_exception_ = nullptr;
    int64_t requested_threads_ = 0;
    int64_t remaining_workers_ = 0;
    int64_t generation_ = 0;
    bool job_active_ = false;
    bool stopping_ = false;
};

ResidentThreadPool& moe_resident_thread_pool() {
    static ResidentThreadPool pool;
    return pool;
}

template <typename Fn>
void run_fixed_threads(int64_t num_threads, const Fn& fn) {
    if (disable_resident_threads()) {
        run_fixed_threads_spawn(num_threads, fn);
        return;
    }
    moe_resident_thread_pool().run(num_threads, fn);
}

template <typename CoreFn, typename Fn>
void run_fixed_threads_pinned(int64_t num_threads,
                              const CoreFn& core_for_tid,
                              const Fn& fn) {
    ThreadAffinityGuard affinity_guard;
    if (num_threads <= 1) {
        const int64_t core = core_for_tid(0);
        report_affinity_bind_error(0, core, bind_current_thread_to_core(core));
        fn(0);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(num_threads - 1));
    for (int64_t tid = 1; tid < num_threads; ++tid) {
        workers.emplace_back([&, tid]() {
            const int64_t core = core_for_tid(tid);
            report_affinity_bind_error(
                tid, core, bind_current_thread_to_core(core));
            fn(tid);
        });
    }
    const int64_t core = core_for_tid(0);
    report_affinity_bind_error(0, core, bind_current_thread_to_core(core));
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

struct ScheduledWaveRuntime {
    int64_t begin = 0;
    int64_t end = 0;
    int64_t total_threads = 0;
};

struct ScheduledTeamScratch {
    explicit ScheduledTeamScratch(int64_t group_size)
        : barrier(group_size) {}

    int64_t expert = -1;
    int64_t threads = 0;
    int64_t rows = 0;
    int64_t a_reorder_stride = 0;
    std::vector<uint16_t> input;
    std::vector<uint16_t> intermediate;
    std::vector<uint16_t> a_reorder;
    std::vector<float> gate_up;
    std::vector<float> down;
    ThreadBarrier barrier;
};

std::vector<int64_t> tensor_to_i64_vector(at::Tensor tensor,
                                          const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(is_integer_dtype(tensor.scalar_type()),
                name, " must use an integer dtype");
    TORCH_CHECK(tensor.dim() == 1, name, " must be 1-D");
    tensor = tensor.to(at::kLong).contiguous();
    const int64_t* ptr = tensor.data_ptr<int64_t>();
    return std::vector<int64_t>(ptr, ptr + tensor.numel());
}

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

    int64_t prepack_threads = env_int_or_default(
        "FUSED_CPP_MOE_PREPACK_THREADS", 1);
    TORCH_CHECK(prepack_threads > 0,
                "FUSED_CPP_MOE_PREPACK_THREADS must be positive, got ",
                prepack_threads);
    prepack_threads = std::min<int64_t>(prepack_threads, E);

    const int64_t w13_expert_stride = N13 * H;
    const int64_t w2_expert_stride = H * F;
    run_fixed_threads(prepack_threads, [&](int64_t tid) {
        for (int64_t e = tid; e < E; e += prepack_threads) {
            pack_transposed_expert_weight(
                w13_ptr, e * w13_expert_stride, N13, H, K13_pad, N13_pad,
                w13_packed_ptr + e * K13_pad * N13_pad);
            pack_transposed_expert_weight(
                w2_ptr, e * w2_expert_stride, H, F, K2_pad, N2_pad,
                w2_packed_ptr + e * K2_pad * N2_pad);
        }
    });

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
    const auto moe_trace_begin = ::fused_cpp::profile::now();
    MoeTraceCollector moe_trace(moe_trace_config_from_env());

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

    const int schedule_debug_level = debug_schedule_level();
    const int64_t actual_threads = num_threads;
    const HierarchicalNSplitConfig nsplit_config =
        hierarchical_nsplit_config_from_env();
    const bool use_hierarchical_nsplit = nsplit_config.enabled;
    const char* moe_trace_strategy =
        use_hierarchical_nsplit
            ? "hierarchical_mn_split_dynamic_expert"
            : "expert_affinity_greedy";
    int64_t moe_trace_expert_tasks = 0;
    for (const std::vector<int64_t>& expert_routes : routes) {
        if (!expert_routes.empty()) {
            ++moe_trace_expert_tasks;
        }
    }
    int64_t nsplit_total_groups = 0;
    int64_t nsplit_group_size = 0;
    std::vector<int64_t> nsplit_thread_cores;
    if (use_hierarchical_nsplit) {
        TORCH_CHECK(nsplit_config.partitions > 0,
                    "FUSED_CPP_MOE_N_SPLIT_CORE_BASES must not be empty");
        TORCH_CHECK(nsplit_config.groups_per_partition > 0,
                    "FUSED_CPP_MOE_N_SPLIT_GROUPS_PER_PARTITION must be "
                    "positive, got ",
                    nsplit_config.groups_per_partition);
        nsplit_total_groups =
            nsplit_config.partitions * nsplit_config.groups_per_partition;
        TORCH_CHECK(nsplit_total_groups > 0,
                    "hierarchical N-split total group count must be positive");
        TORCH_CHECK(actual_threads >= nsplit_total_groups,
                    "hierarchical N-split needs at least one thread per "
                    "group: threads=",
                    actual_threads, " groups=", nsplit_total_groups);
        TORCH_CHECK(actual_threads % nsplit_total_groups == 0,
                    "hierarchical N-split requires equal-sized groups: "
                    "threads=",
                    actual_threads, " groups=", nsplit_total_groups);
        nsplit_group_size = actual_threads / nsplit_total_groups;
        nsplit_thread_cores.resize(static_cast<size_t>(actual_threads));
        for (int64_t tid = 0; tid < actual_threads; ++tid) {
            const int64_t group = tid / nsplit_group_size;
            const int64_t local_tid = tid % nsplit_group_size;
            const int64_t partition =
                group / nsplit_config.groups_per_partition;
            const int64_t group_in_partition =
                group % nsplit_config.groups_per_partition;
            nsplit_thread_cores[static_cast<size_t>(tid)] =
                nsplit_config.core_bases[static_cast<size_t>(partition)] +
                group_in_partition * nsplit_group_size + local_tid;
        }
    }

    if (!use_hierarchical_nsplit) {
        moe_trace.reserve(tasks.size() * 2);
        std::vector<std::vector<TaskRange>> ranges =
            split_tasks_by_expert_affinity(tasks, actual_threads);
        std::vector<ThreadScheduleDebug> schedule_debug;
        if (schedule_debug_level > 0) {
            schedule_debug.reserve(static_cast<size_t>(actual_threads));
            for (const std::vector<TaskRange>& thread_ranges : ranges) {
                schedule_debug.push_back(
                    summarize_thread_schedule(tasks, thread_ranges));
            }
        }

        std::vector<ThreadScratch> scratches(
            static_cast<size_t>(actual_threads));
        for (ThreadScratch& scratch : scratches) {
            scratch.input.resize(static_cast<size_t>(
                kMoeTokenTile * w13.K_pad));
            scratch.intermediate.resize(static_cast<size_t>(
                kMoeTokenTile * w2.K_pad));
            scratch.a_reorder.resize(static_cast<size_t>(
                kMoeTokenTile * std::max(w13.K_pad, w2.K_pad) * 2));
            scratch.gate_up.resize(static_cast<size_t>(
                kMoeTokenTile * w13.N_pad));
            scratch.down.resize(static_cast<size_t>(
                kMoeTokenTile * w2.N_pad));
        }

        run_fixed_threads(actual_threads, [&](int64_t tid) {
            const auto thread_begin = ::fused_cpp::profile::now();
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
                        uint16_t* dst =
                            scratch.input.data() + m * w13.K_pad;
                        std::copy(src, src + H, dst);
                    }

                    std::fill(scratch.gate_up.begin(),
                              scratch.gate_up.begin() + rows * w13.N_pad,
                              0.0f);
                    trace_dispatch_fp32_gemm(
                        moe_trace,
                        "w13",
                        tid,
                        -1,
                        -1,
                        task.expert,
                        task.route_begin,
                        rows,
                        scratch.input.data(),
                        w13_ptr + task.expert * w13.packed_stride,
                        scratch.gate_up.data(),
                        scratch.a_reorder.data(),
                        static_cast<int>(rows),
                        static_cast<int>(w13.K_pad),
                        static_cast<int>(w13.N_pad),
                        static_cast<int>(w13.N_pad));
                    apply_gate_up_bias(scratch.gate_up.data(), rows,
                                       w13.N_pad, w13.N, task.expert,
                                       w13_bias);

                    activation_to_bf16(
                        activation, scratch.gate_up.data(),
                        scratch.intermediate.data(), rows, w13.N_pad,
                        w2.K_pad, F);

                    std::fill(scratch.down.begin(),
                              scratch.down.begin() + rows * w2.N_pad, 0.0f);
                    trace_dispatch_fp32_gemm(
                        moe_trace,
                        "w2",
                        tid,
                        -1,
                        -1,
                        task.expert,
                        task.route_begin,
                        rows,
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
                        const float* src =
                            scratch.down.data() + m * w2.N_pad;
                        const int64_t bias_base = task.expert * H;
                        for (int64_t h = 0; h < H; ++h) {
                            dst[h] = src[h] + read_optional_bias(
                                                  w2_bias, bias_base + h);
                        }
                    }
                }
            }
            if (schedule_debug_level > 0) {
                schedule_debug[static_cast<size_t>(tid)].ms =
                    ::fused_cpp::profile::elapsed_ms(thread_begin);
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
    } else {
        std::vector<int64_t> expert_order;
        expert_order.reserve(static_cast<size_t>(num_experts));
        for (int64_t expert = 0; expert < num_experts; ++expert) {
            if (!routes[static_cast<size_t>(expert)].empty()) {
                expert_order.push_back(expert);
            }
        }
        std::sort(expert_order.begin(), expert_order.end(),
                  [&](int64_t lhs, int64_t rhs) {
                      const size_t lhs_rows =
                          routes[static_cast<size_t>(lhs)].size();
                      const size_t rhs_rows =
                          routes[static_cast<size_t>(rhs)].size();
                      if (lhs_rows != rhs_rows) {
                          return lhs_rows > rhs_rows;
                      }
                      return lhs < rhs;
                  });
        moe_trace_expert_tasks = static_cast<int64_t>(expert_order.size());
        moe_trace.reserve(
            expert_order.size() * static_cast<size_t>(nsplit_group_size) * 2);
        std::atomic<size_t> next_expert_idx{0};
        std::vector<ThreadScheduleDebug> schedule_debug(
            static_cast<size_t>(nsplit_total_groups));

        int64_t max_expert_rows = 0;
        for (int64_t expert : expert_order) {
            max_expert_rows = std::max<int64_t>(
                max_expert_rows,
                static_cast<int64_t>(
                    routes[static_cast<size_t>(expert)].size()));
        }
        const int64_t a_reorder_stride =
            max_expert_rows * std::max(w13.K_pad, w2.K_pad) * 2;
        std::vector<std::unique_ptr<HierarchicalGroupScratch>>
            group_scratches;
        group_scratches.reserve(static_cast<size_t>(nsplit_total_groups));
        for (int64_t group = 0; group < nsplit_total_groups; ++group) {
            auto scratch = std::make_unique<HierarchicalGroupScratch>(
                nsplit_group_size);
            scratch->input.resize(static_cast<size_t>(
                max_expert_rows * w13.K_pad));
            scratch->intermediate.resize(static_cast<size_t>(
                max_expert_rows * w2.K_pad));
            scratch->a_reorder.resize(static_cast<size_t>(
                nsplit_group_size * a_reorder_stride));
            scratch->gate_up.resize(static_cast<size_t>(
                max_expert_rows * w13.N_pad));
            scratch->down.resize(static_cast<size_t>(
                max_expert_rows * w2.N_pad));
            group_scratches.push_back(std::move(scratch));
        }

        auto core_for_tid = [&](int64_t tid) {
            return nsplit_thread_cores[static_cast<size_t>(tid)];
        };
        run_fixed_threads_pinned(actual_threads, core_for_tid, [&](int64_t tid) {
            const int64_t group = tid / nsplit_group_size;
            const int64_t local_tid = tid % nsplit_group_size;
            const auto thread_begin = ::fused_cpp::profile::now();
            HierarchicalGroupScratch& scratch =
                *group_scratches[static_cast<size_t>(group)];
            ThreadBarrier& barrier = scratch.barrier;
            uint16_t* a_reorder = scratch.a_reorder.data() +
                local_tid * a_reorder_stride;

            while (true) {
                if (local_tid == 0) {
                    const size_t order_idx = next_expert_idx.fetch_add(
                        size_t{1}, std::memory_order_relaxed);
                    const int64_t expert =
                        order_idx < expert_order.size()
                            ? expert_order[order_idx]
                            : -1;
                    scratch.current_expert.store(
                        expert, std::memory_order_release);
                    if (schedule_debug_level > 0 && expert >= 0) {
                        const int64_t expert_rows =
                            static_cast<int64_t>(
                                routes[static_cast<size_t>(expert)].size());
                        ThreadScheduleDebug& debug =
                            schedule_debug[static_cast<size_t>(group)];
                        debug.rows += expert_rows;
                        ++debug.tasks;
                        ++debug.ranges;
                        debug.experts.push_back(expert);
                        debug.expert_rows.push_back(expert_rows);
                    }
                }
                barrier.wait();

                const int64_t expert = scratch.current_expert.load(
                    std::memory_order_acquire);
                if (expert < 0) {
                    break;
                }

                const auto& expert_routes =
                    routes[static_cast<size_t>(expert)];
                const int64_t rows = static_cast<int64_t>(
                    expert_routes.size());

                if (local_tid == 0) {
                    std::fill(scratch.input.begin(),
                              scratch.input.begin() + rows * w13.K_pad,
                              static_cast<uint16_t>(0));
                    for (int64_t m = 0; m < rows; ++m) {
                        const int64_t flat =
                            expert_routes[static_cast<size_t>(m)];
                        const int64_t token = flat / top_k;
                        const uint16_t* src = input_ptr + token * H;
                        uint16_t* dst =
                            scratch.input.data() + m * w13.K_pad;
                        std::copy(src, src + H, dst);
                    }
                    std::fill(scratch.gate_up.begin(),
                              scratch.gate_up.begin() + rows * w13.N_pad,
                              0.0f);
                }
                barrier.wait();

                trace_dispatch_fp32_gemm_auto_split(
                    moe_trace,
                    "w13",
                    tid,
                    group,
                    local_tid,
                    expert,
                    0,
                    rows,
                    scratch.input.data(),
                    w13_ptr + expert * w13.packed_stride,
                    scratch.gate_up.data(),
                    a_reorder,
                    static_cast<int>(rows),
                    static_cast<int>(w13.K_pad),
                    static_cast<int>(w13.N_pad),
                    static_cast<int>(w13.N_pad),
                    nsplit_group_size);
                barrier.wait();

                if (local_tid == 0) {
                    std::fill(scratch.down.begin(),
                              scratch.down.begin() + rows * w2.N_pad,
                              0.0f);
                }
                const SplitRange activation_range = split_evenly(
                    rows, nsplit_group_size, local_tid);
                apply_gate_up_bias_range(
                    scratch.gate_up.data(), activation_range.begin,
                    activation_range.size, w13.N_pad, w13.N, expert,
                    w13_bias);
                activation_range_to_bf16(
                    activation, scratch.gate_up.data(),
                    scratch.intermediate.data(), activation_range.begin,
                    activation_range.size, w13.N_pad, w2.K_pad, F);
                barrier.wait();

                trace_dispatch_fp32_gemm_auto_split(
                    moe_trace,
                    "w2",
                    tid,
                    group,
                    local_tid,
                    expert,
                    0,
                    rows,
                    scratch.intermediate.data(),
                    w2_ptr + expert * w2.packed_stride,
                    scratch.down.data(),
                    a_reorder,
                    static_cast<int>(rows),
                    static_cast<int>(w2.K_pad),
                    static_cast<int>(w2.N_pad),
                    static_cast<int>(w2.N_pad),
                    nsplit_group_size);
                barrier.wait();

                const int64_t h_blocks = w2.N_pad / kKernelTile;
                const int64_t blocks_per_thread =
                    h_blocks / nsplit_group_size;
                const int64_t extra_blocks =
                    h_blocks % nsplit_group_size;
                int64_t start_block = 0;
                int64_t my_blocks = 0;
                if (local_tid < extra_blocks) {
                    start_block =
                        local_tid * (blocks_per_thread + 1);
                    my_blocks = blocks_per_thread + 1;
                } else {
                    start_block =
                        extra_blocks * (blocks_per_thread + 1) +
                        (local_tid - extra_blocks) * blocks_per_thread;
                    my_blocks = blocks_per_thread;
                }
                const int64_t h_begin = start_block * kKernelTile;
                const int64_t h_end = std::min<int64_t>(
                    H, h_begin + my_blocks * kKernelTile);
                if (h_begin < h_end) {
                    const int64_t bias_base = expert * H;
                    for (int64_t m = 0; m < rows; ++m) {
                        const int64_t flat =
                            expert_routes[static_cast<size_t>(m)];
                        float* dst = route_out_ptr + flat * H;
                        const float* src =
                            scratch.down.data() + m * w2.N_pad;
                        for (int64_t h = h_begin; h < h_end; ++h) {
                            dst[h] = src[h] + read_optional_bias(
                                                  w2_bias, bias_base + h);
                        }
                    }
                }
                barrier.wait();
            }

            if (schedule_debug_level > 0 && local_tid == 0) {
                schedule_debug[static_cast<size_t>(group)].ms =
                    ::fused_cpp::profile::elapsed_ms(thread_begin);
            }
        });

        if (schedule_debug_level > 0) {
            int64_t longest_group = -1;
            int64_t shortest_group = -1;
            for (int64_t group = 0; group < nsplit_total_groups; ++group) {
                const ThreadScheduleDebug& debug =
                    schedule_debug[static_cast<size_t>(group)];
                if (debug.rows == 0) {
                    continue;
                }
                if (longest_group < 0 ||
                    debug.ms > schedule_debug[static_cast<size_t>(
                                   longest_group)].ms) {
                    longest_group = group;
                }
                if (shortest_group < 0 ||
                    debug.ms < schedule_debug[static_cast<size_t>(
                                   shortest_group)].ms) {
                    shortest_group = group;
                }
            }
            std::fprintf(
                stderr,
                "[fused_moe_bf16_tiled][schedule] threads=%lld experts=%lld "
                "routes=%lld expert_tasks=%zu micro_tiles=%zu "
                "strategy=hierarchical_mn_split_dynamic_expert "
                "partitions=%lld groups_per_partition=%lld groups=%lld "
                "group_size=%lld core_bases=[",
                static_cast<long long>(actual_threads),
                static_cast<long long>(num_experts),
                static_cast<long long>(num_routes),
                expert_order.size(),
                tasks.size(),
                static_cast<long long>(nsplit_config.partitions),
                static_cast<long long>(nsplit_config.groups_per_partition),
                static_cast<long long>(nsplit_total_groups),
                static_cast<long long>(nsplit_group_size));
            for (size_t i = 0; i < nsplit_config.core_bases.size(); ++i) {
                if (i != 0) {
                    std::fprintf(stderr, ",");
                }
                std::fprintf(
                    stderr, "%lld",
                    static_cast<long long>(nsplit_config.core_bases[i]));
            }
            std::fprintf(stderr, "]\n");
            if (longest_group >= 0) {
                print_schedule_debug_line(
                    "longest_group", longest_group,
                    schedule_debug[static_cast<size_t>(longest_group)]);
            }
            if (shortest_group >= 0) {
                print_schedule_debug_line(
                    "shortest_group", shortest_group,
                    schedule_debug[static_cast<size_t>(shortest_group)]);
            }
            if (schedule_debug_level >= 2) {
                for (int64_t group = 0; group < nsplit_total_groups;
                     ++group) {
                    print_schedule_debug_line(
                        "group", group,
                        schedule_debug[static_cast<size_t>(group)]);
                }
            }
        }
    }

    at::Tensor output_acc = at::zeros(
        {num_tokens, H},
        at::TensorOptions().device(input.device()).dtype(at::kFloat));
    float* output_ptr = output_acc.data_ptr<float>();
    const float* topk_w = weights_f32.data_ptr<float>();

    auto merge_routes = [&](int64_t tid) {
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
    };
    if (use_hierarchical_nsplit) {
        auto core_for_tid = [&](int64_t tid) {
            return nsplit_thread_cores[static_cast<size_t>(tid)];
        };
        run_fixed_threads_pinned(actual_threads, core_for_tid, merge_routes);
    } else {
        run_fixed_threads(actual_threads, merge_routes);
    }

    at::Tensor output = output_acc.to(at::kBFloat16);
    if (moe_trace.enabled()) {
        const double e2e_ms = ::fused_cpp::profile::elapsed_ms(moe_trace_begin);
        moe_trace.write_report(moe_trace_strategy,
                               actual_threads,
                               num_tokens,
                               top_k,
                               num_experts,
                               num_routes,
                               H,
                               F,
                               tasks.size(),
                               moe_trace_expert_tasks,
                               nsplit_total_groups,
                               nsplit_group_size,
                               e2e_ms);
    }
    return output;
#endif
}

at::Tensor fused_moe_bf16_tiled_scheduled(at::Tensor input,
                            at::Tensor w13_packed,
                            int64_t w13_K,
                            int64_t w13_N,
                            at::Tensor w2_packed,
                            int64_t w2_K,
                            int64_t w2_N,
                            at::Tensor topk_weights,
                            at::Tensor topk_ids,
                            at::Tensor wave_offsets,
                            at::Tensor team_expert_ids,
                            at::Tensor team_threads,
                            c10::optional<at::Tensor> w13_bias,
                            c10::optional<at::Tensor> w2_bias,
                            int64_t num_threads,
                            std::string activation,
                            int64_t global_num_experts,
                            bool skip_weighted) {
#ifndef __aarch64__
    TORCH_CHECK(false, "fused_moe_bf16_tiled_scheduled requires AArch64");
#else
    const auto moe_trace_begin = ::fused_cpp::profile::now();
    MoeTraceCollector moe_trace(moe_trace_config_from_env());

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

    const std::vector<int64_t> wave_offsets_v =
        tensor_to_i64_vector(wave_offsets, "wave_offsets");
    const std::vector<int64_t> team_expert_ids_v =
        tensor_to_i64_vector(team_expert_ids, "team_expert_ids");
    const std::vector<int64_t> team_threads_v =
        tensor_to_i64_vector(team_threads, "team_threads");
    TORCH_CHECK(wave_offsets_v.size() >= 2,
                "wave_offsets must contain at least [0, num_teams]");
    TORCH_CHECK(wave_offsets_v.front() == 0,
                "wave_offsets[0] must be 0, got ", wave_offsets_v.front());
    const int64_t num_teams =
        static_cast<int64_t>(team_expert_ids_v.size());
    TORCH_CHECK(static_cast<int64_t>(team_threads_v.size()) == num_teams,
                "team_threads must have the same length as team_expert_ids: ",
                team_threads_v.size(), " vs ", num_teams);
    TORCH_CHECK(wave_offsets_v.back() == num_teams,
                "last wave_offsets entry must equal num_teams=", num_teams,
                ", got ", wave_offsets_v.back());

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

    int64_t active_experts = 0;
    size_t micro_tiles = 0;
    for (const std::vector<int64_t>& expert_routes : routes) {
        if (!expert_routes.empty()) {
            ++active_experts;
            micro_tiles += static_cast<size_t>(ceil_div_int64(
                static_cast<int64_t>(expert_routes.size()), kMoeTokenTile));
        }
    }
    TORCH_CHECK(num_teams == active_experts,
                "scheduled plan must contain exactly one team per active "
                "expert: teams=", num_teams,
                " active_experts=", active_experts);

    std::vector<int64_t> seen(static_cast<size_t>(num_experts), 0);
    std::vector<int64_t> team_rows(static_cast<size_t>(num_teams), 0);
    int64_t trace_gemm_hint = 0;
    for (int64_t team = 0; team < num_teams; ++team) {
        const int64_t expert = team_expert_ids_v[static_cast<size_t>(team)];
        const int64_t threads = team_threads_v[static_cast<size_t>(team)];
        TORCH_CHECK(expert >= 0 && expert < num_experts,
                    "team_expert_ids[", team, "] out of range: ", expert,
                    ", valid range [0, ", num_experts, ")");
        TORCH_CHECK(threads > 0,
                    "team_threads[", team, "] must be positive, got ",
                    threads);
        TORCH_CHECK(threads <= num_threads,
                    "team_threads[", team, "]=", threads,
                    " exceeds num_threads=", num_threads);
        TORCH_CHECK(seen[static_cast<size_t>(expert)] == 0,
                    "scheduled plan contains duplicate expert ", expert);
        const int64_t rows = static_cast<int64_t>(
            routes[static_cast<size_t>(expert)].size());
        TORCH_CHECK(rows > 0,
                    "scheduled plan contains inactive expert ", expert);
        check_positive_int(rows, "scheduled team rows");
        seen[static_cast<size_t>(expert)] = 1;
        team_rows[static_cast<size_t>(team)] = rows;
        trace_gemm_hint += threads * 2;
    }
    for (int64_t expert = 0; expert < num_experts; ++expert) {
        if (!routes[static_cast<size_t>(expert)].empty()) {
            TORCH_CHECK(seen[static_cast<size_t>(expert)] == 1,
                        "scheduled plan is missing active expert ", expert);
        }
    }

    const int64_t num_waves =
        static_cast<int64_t>(wave_offsets_v.size()) - 1;
    std::vector<ScheduledWaveRuntime> waves;
    waves.reserve(static_cast<size_t>(num_waves));
    std::vector<int64_t> team_thread_starts(static_cast<size_t>(num_teams), 0);
    for (int64_t wave = 0; wave < num_waves; ++wave) {
        const int64_t begin = wave_offsets_v[static_cast<size_t>(wave)];
        const int64_t end = wave_offsets_v[static_cast<size_t>(wave + 1)];
        TORCH_CHECK(begin <= end,
                    "wave_offsets must be nondecreasing, got wave ", wave,
                    " begin=", begin, " end=", end);
        TORCH_CHECK(begin >= 0 && end <= num_teams,
                    "wave ", wave, " range [", begin, ", ", end,
                    ") is outside num_teams=", num_teams);
        int64_t wave_threads = 0;
        for (int64_t team = begin; team < end; ++team) {
            team_thread_starts[static_cast<size_t>(team)] = wave_threads;
            wave_threads += team_threads_v[static_cast<size_t>(team)];
            TORCH_CHECK(wave_threads <= num_threads,
                        "wave ", wave, " uses ", wave_threads,
                        " threads, exceeding num_threads=", num_threads);
        }
        waves.push_back(ScheduledWaveRuntime{begin, end, wave_threads});
    }

    at::Tensor route_out = at::empty(
        {num_routes, H},
        at::TensorOptions().device(input.device()).dtype(at::kFloat));
    float* route_out_ptr = route_out.data_ptr<float>();
    const uint16_t* input_ptr = bf16_data_const(input);
    const uint16_t* w13_ptr = bf16_data_const(w13.tensor);
    const uint16_t* w2_ptr = bf16_data_const(w2.tensor);

    std::vector<std::unique_ptr<ScheduledTeamScratch>> scratches;
    scratches.reserve(static_cast<size_t>(num_teams));
    for (int64_t team = 0; team < num_teams; ++team) {
        const int64_t expert = team_expert_ids_v[static_cast<size_t>(team)];
        const int64_t threads = team_threads_v[static_cast<size_t>(team)];
        const int64_t rows = team_rows[static_cast<size_t>(team)];
        auto scratch = std::make_unique<ScheduledTeamScratch>(threads);
        scratch->expert = expert;
        scratch->threads = threads;
        scratch->rows = rows;
        scratch->a_reorder_stride =
            rows * std::max(w13.K_pad, w2.K_pad) * 2;
        scratch->input.resize(static_cast<size_t>(rows * w13.K_pad));
        scratch->intermediate.resize(static_cast<size_t>(rows * w2.K_pad));
        scratch->a_reorder.resize(static_cast<size_t>(
            threads * scratch->a_reorder_stride));
        scratch->gate_up.resize(static_cast<size_t>(rows * w13.N_pad));
        scratch->down.resize(static_cast<size_t>(rows * w2.N_pad));
        scratches.push_back(std::move(scratch));
    }

    const int schedule_debug_level = debug_schedule_level();
    if (schedule_debug_level > 0) {
        std::fprintf(
            stderr,
            "[fused_moe_bf16_tiled][schedule] threads=%lld experts=%lld "
            "routes=%lld waves=%lld teams=%lld strategy=external_plan\n",
            static_cast<long long>(num_threads),
            static_cast<long long>(num_experts),
            static_cast<long long>(num_routes),
            static_cast<long long>(num_waves),
            static_cast<long long>(num_teams));
    }

    moe_trace.reserve(static_cast<size_t>(trace_gemm_hint));
    ThreadBarrier wave_barrier(num_threads);
    run_fixed_threads(num_threads, [&](int64_t tid) {
        for (int64_t wave_idx = 0; wave_idx < num_waves; ++wave_idx) {
            const ScheduledWaveRuntime& wave =
                waves[static_cast<size_t>(wave_idx)];
            int64_t selected_team = -1;
            int64_t local_tid = -1;
            if (tid < wave.total_threads) {
                for (int64_t team = wave.begin; team < wave.end; ++team) {
                    const int64_t thread_begin =
                        team_thread_starts[static_cast<size_t>(team)];
                    const int64_t thread_end = thread_begin +
                        team_threads_v[static_cast<size_t>(team)];
                    if (tid >= thread_begin && tid < thread_end) {
                        selected_team = team;
                        local_tid = tid - thread_begin;
                        break;
                    }
                }
            }

            if (selected_team >= 0) {
                ScheduledTeamScratch& scratch =
                    *scratches[static_cast<size_t>(selected_team)];
                ThreadBarrier& barrier = scratch.barrier;
                const int64_t expert = scratch.expert;
                const int64_t rows = scratch.rows;
                const int64_t group_size = scratch.threads;
                const auto& expert_routes =
                    routes[static_cast<size_t>(expert)];
                uint16_t* a_reorder = scratch.a_reorder.data() +
                    local_tid * scratch.a_reorder_stride;

                if (local_tid == 0) {
                    std::fill(scratch.input.begin(),
                              scratch.input.begin() + rows * w13.K_pad,
                              static_cast<uint16_t>(0));
                    for (int64_t m = 0; m < rows; ++m) {
                        const int64_t flat =
                            expert_routes[static_cast<size_t>(m)];
                        const int64_t token = flat / top_k;
                        const uint16_t* src = input_ptr + token * H;
                        uint16_t* dst =
                            scratch.input.data() + m * w13.K_pad;
                        std::copy(src, src + H, dst);
                    }
                    std::fill(scratch.gate_up.begin(),
                              scratch.gate_up.begin() + rows * w13.N_pad,
                              0.0f);
                }
                barrier.wait();

                trace_dispatch_fp32_gemm_auto_split(
                    moe_trace,
                    "w13",
                    tid,
                    selected_team,
                    local_tid,
                    expert,
                    0,
                    rows,
                    scratch.input.data(),
                    w13_ptr + expert * w13.packed_stride,
                    scratch.gate_up.data(),
                    a_reorder,
                    static_cast<int>(rows),
                    static_cast<int>(w13.K_pad),
                    static_cast<int>(w13.N_pad),
                    static_cast<int>(w13.N_pad),
                    group_size);
                barrier.wait();

                if (local_tid == 0) {
                    std::fill(scratch.down.begin(),
                              scratch.down.begin() + rows * w2.N_pad, 0.0f);
                }
                const SplitRange activation_range = split_evenly(
                    rows, group_size, local_tid);
                apply_gate_up_bias_range(
                    scratch.gate_up.data(), activation_range.begin,
                    activation_range.size, w13.N_pad, w13.N, expert,
                    w13_bias);
                activation_range_to_bf16(
                    activation, scratch.gate_up.data(),
                    scratch.intermediate.data(), activation_range.begin,
                    activation_range.size, w13.N_pad, w2.K_pad, F);
                barrier.wait();

                trace_dispatch_fp32_gemm_auto_split(
                    moe_trace,
                    "w2",
                    tid,
                    selected_team,
                    local_tid,
                    expert,
                    0,
                    rows,
                    scratch.intermediate.data(),
                    w2_ptr + expert * w2.packed_stride,
                    scratch.down.data(),
                    a_reorder,
                    static_cast<int>(rows),
                    static_cast<int>(w2.K_pad),
                    static_cast<int>(w2.N_pad),
                    static_cast<int>(w2.N_pad),
                    group_size);
                barrier.wait();

                const SplitRange h_range = n_split_range(
                    static_cast<int>(w2.N_pad), group_size, local_tid);
                const int64_t h_begin = h_range.begin;
                const int64_t h_end = std::min<int64_t>(
                    H, h_begin + h_range.size);
                if (h_begin < h_end) {
                    const int64_t bias_base = expert * H;
                    for (int64_t m = 0; m < rows; ++m) {
                        const int64_t flat =
                            expert_routes[static_cast<size_t>(m)];
                        float* dst = route_out_ptr + flat * H;
                        const float* src =
                            scratch.down.data() + m * w2.N_pad;
                        for (int64_t h = h_begin; h < h_end; ++h) {
                            dst[h] = src[h] + read_optional_bias(
                                                  w2_bias, bias_base + h);
                        }
                    }
                }
                barrier.wait();
            }
            wave_barrier.wait();
        }
    });

    at::Tensor output_acc = at::zeros(
        {num_tokens, H},
        at::TensorOptions().device(input.device()).dtype(at::kFloat));
    float* output_ptr = output_acc.data_ptr<float>();
    const float* topk_w = weights_f32.data_ptr<float>();

    auto merge_routes = [&](int64_t tid) {
        const int64_t rows_per_thread =
            ceil_div_int64(num_tokens, num_threads);
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
    };
    run_fixed_threads(num_threads, merge_routes);

    at::Tensor output = output_acc.to(at::kBFloat16);
    if (moe_trace.enabled()) {
        const double e2e_ms = ::fused_cpp::profile::elapsed_ms(moe_trace_begin);
        moe_trace.write_report("external_plan_scheduled",
                               num_threads,
                               num_tokens,
                               top_k,
                               num_experts,
                               num_routes,
                               H,
                               F,
                               micro_tiles,
                               num_teams,
                               num_teams,
                               0,
                               e2e_ms);
    }
    return output;
#endif
}
