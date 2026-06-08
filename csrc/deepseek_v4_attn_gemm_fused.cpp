#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

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
#ifdef __linux__
void bf16gemm_k_nld_b(const uint16_t* A, const uint16_t* B_reo, uint16_t* C,
                      uint16_t* A_reorder, const gemm_params_t* params);
void bf16gemm_k_nld1_b(const uint16_t* A, const uint16_t* B_reo, uint16_t* C,
                       uint16_t* A_reorder, const gemm_params_t* params);
void bf16gemm_k_nld2_b(const uint16_t* A, const uint16_t* B_reo, uint16_t* C,
                       uint16_t* A_reorder, const gemm_params_t* params);
void bf16gemm_k_nld4_b(const uint16_t* A, const uint16_t* B_reo, uint16_t* C,
                       uint16_t* A_reorder, const gemm_params_t* params);
#endif
}
#endif

namespace {

constexpr int64_t kTile = 8;

int64_t ceil_div_int64(int64_t x, int64_t y) {
    return (x + y - 1) / y;
}

int64_t ceil_to_multiple(int64_t x, int64_t multiple) {
    return ceil_div_int64(x, multiple) * multiple;
}

void check_bf16_cpu_2d(const at::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16,
                name, " must have dtype torch.bfloat16");
    TORCH_CHECK(tensor.dim() == 2, name, " must be 2-D, got ",
                tensor.dim(), "-D");
}

void check_int_arg(int64_t value, const char* name) {
    TORCH_CHECK(value > 0, name, " must be positive, got ", value);
    TORCH_CHECK(value <= std::numeric_limits<int>::max(),
                name, " exceeds int32 kernel limit: ", value);
}

#ifdef __aarch64__

const uint16_t* bf16_data_const(const at::Tensor& tensor) {
    return reinterpret_cast<const uint16_t*>(tensor.data_ptr<at::BFloat16>());
}

bool bind_current_thread_to_cpu(int64_t cpu_id) {
#ifdef __linux__
    if (cpu_id < 0 || cpu_id >= CPU_SETSIZE) {
        return false;
    }
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(static_cast<int>(cpu_id), &mask);
    return pthread_setaffinity_np(pthread_self(), sizeof(mask), &mask) == 0;
#else
    (void)cpu_id;
    return true;
#endif
}

uint16_t* bf16_data(at::Tensor& tensor) {
    return reinterpret_cast<uint16_t*>(tensor.data_ptr<at::BFloat16>());
}

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

struct PackedWeight {
    at::Tensor tensor;
    int64_t K;
    int64_t N;
    int64_t K_pad;
    int64_t N_pad;
};

PackedWeight checked_packed_weight(at::Tensor packed,
                                   int64_t K,
                                   int64_t N,
                                   const char* name) {
    TORCH_CHECK(packed.device().is_cpu(), name, " packed weight must be CPU");
    TORCH_CHECK(packed.scalar_type() == at::kBFloat16,
                name, " packed weight must be torch.bfloat16");
    TORCH_CHECK(packed.is_contiguous(),
                name, " packed weight must be contiguous");
    check_int_arg(K, "K");
    check_int_arg(N, "N");
    const int64_t K_pad = ceil_to_multiple(K, kTile);
    const int64_t N_pad = ceil_to_multiple(N, kTile);
    TORCH_CHECK(K_pad <= std::numeric_limits<int>::max(),
                name, " padded K exceeds int32 kernel limit: ", K_pad);
    TORCH_CHECK(N_pad <= std::numeric_limits<int>::max(),
                name, " padded N exceeds int32 kernel limit: ", N_pad);
    TORCH_CHECK(packed.numel() == K_pad * N_pad,
                name, " packed weight numel mismatch: expected ",
                K_pad * N_pad, ", got ", packed.numel());
    return PackedWeight{packed, K, N, K_pad, N_pad};
}

at::Tensor narrow_output_if_needed(const at::Tensor& output,
                                   int64_t N,
                                   int64_t N_pad) {
    if (N == N_pad) {
        return output;
    }
    return output.narrow(1, 0, N).contiguous();
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
        bf16gemm_k_ld(A, B_reo, C, A_reorder,
                      &p);
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
        bf16gemm_k_ld4(At, B_reo, Ct, A_reo_t,
                       &p);
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
        bf16gemm_k_ld2(At, B_reo, Ct, A_reo_t,
                       &p);
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
        bf16gemm_k_ld1(At, B_reo, Ct, A_reo_t,
                       &p);
    }
}

at::Tensor run_fp32_gemm(const at::Tensor& a_storage,
                         const PackedWeight& weight,
                         at::Tensor& scratch) {
    const int64_t M64 = a_storage.size(0);
    const auto options = at::TensorOptions()
                             .device(a_storage.device())
                             .dtype(at::kFloat);
    at::Tensor output = at::zeros({M64, weight.N_pad}, options);
    if (M64 == 0) {
        return output.narrow(1, 0, weight.N).contiguous();
    }
    dispatch_fp32_gemm(
        bf16_data_const(a_storage),
        bf16_data_const(weight.tensor),
        output.data_ptr<float>(),
        bf16_data(scratch),
        static_cast<int>(M64),
        static_cast<int>(weight.K_pad),
        static_cast<int>(weight.N_pad),
        static_cast<int>(weight.N_pad));
    return narrow_output_if_needed(output, weight.N, weight.N_pad);
}

#ifdef __linux__
void dispatch_bf16_nld_gemm(const uint16_t* A,
                            const uint16_t* B_reo,
                            uint16_t* C,
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
        bf16gemm_k_nld_b(A, B_reo, C, A_reorder, &p);
        processed = m_full;
    }

    int m_rem = M - processed;
    if (m_rem == 0) {
        return;
    }

    const uint16_t* At = A + static_cast<int64_t>(processed) * K;
    uint16_t* Ct = C + static_cast<int64_t>(processed) * ldc;
    uint16_t* A_reo_t = A_reorder + static_cast<int64_t>(processed) * K;

    if (m_rem >= 4) {
        p.m = 4;
        p.k = K;
        p.n = N;
        bf16gemm_k_nld4_b(At, B_reo, Ct, A_reo_t, &p);
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
        bf16gemm_k_nld2_b(At, B_reo, Ct, A_reo_t, &p);
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
        bf16gemm_k_nld1_b(At, B_reo, Ct, A_reo_t, &p);
    }
}
#endif

at::Tensor run_bf16_gemm(const at::Tensor& a_storage,
                         const PackedWeight& weight,
                         at::Tensor& scratch) {
#ifdef __APPLE__
    // macOS/Apple currently mis-handles bf16-output nld tail kernels for M<8.
    return run_fp32_gemm(a_storage, weight, scratch).to(at::kBFloat16);
#elif defined(__linux__)
    const int64_t M64 = a_storage.size(0);
    at::Tensor output = at::empty({M64, weight.N_pad}, a_storage.options());
    if (M64 == 0) {
        return output.narrow(1, 0, weight.N).contiguous();
    }
    dispatch_bf16_nld_gemm(
        bf16_data_const(a_storage),
        bf16_data_const(weight.tensor),
        bf16_data(output),
        bf16_data(scratch),
        static_cast<int>(M64),
        static_cast<int>(weight.K_pad),
        static_cast<int>(weight.N_pad),
        static_cast<int>(weight.N_pad));
    return narrow_output_if_needed(output, weight.N, weight.N_pad);
#else
    return run_fp32_gemm(a_storage, weight, scratch).to(at::kBFloat16);
#endif
}

void dispatch_fp32_gemm_to_output(const at::Tensor& a_storage,
                                  const PackedWeight& weight,
                                  at::Tensor& output,
                                  at::Tensor& scratch,
                                  int64_t row_start,
                                  int64_t row_count,
                                  int64_t scratch_offset) {
    if (row_count == 0) {
        return;
    }
    dispatch_fp32_gemm(
        bf16_data_const(a_storage) + row_start * weight.K_pad,
        bf16_data_const(weight.tensor),
        output.data_ptr<float>() + row_start * weight.N_pad,
        bf16_data(scratch) + scratch_offset,
        static_cast<int>(row_count),
        static_cast<int>(weight.K_pad),
        static_cast<int>(weight.N_pad),
        static_cast<int>(weight.N_pad));
}

[[maybe_unused]] void dispatch_bf16_gemm_to_output(const at::Tensor& a_storage,
                                                   const PackedWeight& weight,
                                                   at::Tensor& output,
                                                   at::Tensor& scratch,
                                                   int64_t row_start,
                                                   int64_t row_count,
                                                   int64_t scratch_offset) {
    if (row_count == 0) {
        return;
    }
#if defined(__linux__)
    dispatch_bf16_nld_gemm(
        bf16_data_const(a_storage) + row_start * weight.K_pad,
        bf16_data_const(weight.tensor),
        bf16_data(output) + row_start * weight.N_pad,
        bf16_data(scratch) + scratch_offset,
        static_cast<int>(row_count),
        static_cast<int>(weight.K_pad),
        static_cast<int>(weight.N_pad),
        static_cast<int>(weight.N_pad));
#else
    TORCH_CHECK(false,
                "direct bf16-output GEMM dispatch is only enabled on Linux");
#endif
}

#endif

}  // namespace

std::tuple<at::Tensor, int64_t, int64_t>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_prepare(
    at::Tensor weight) {
#ifndef __aarch64__
    TORCH_CHECK(false,
                "deepseek_v4 attn gemm fused prepare requires AArch64");
#else
    check_bf16_cpu_2d(weight, "weight");
    const int64_t K = weight.size(0);
    const int64_t N = weight.size(1);
    check_int_arg(K, "K");
    check_int_arg(N, "N");
    const int64_t K_pad = ceil_to_multiple(K, kTile);
    const int64_t N_pad = ceil_to_multiple(N, kTile);

    at::Tensor weight_padded;
    if (K_pad == K && N_pad == N && weight.is_contiguous()) {
        weight_padded = weight;
    } else {
        weight_padded = at::zeros({K_pad, N_pad}, weight.options());
        weight_padded.narrow(0, 0, K).narrow(1, 0, N).copy_(weight);
    }

    at::Tensor packed = at::empty({K_pad * N_pad}, weight.options());
    bf16_pack_b(bf16_data_const(weight_padded), bf16_data(packed),
                static_cast<int>(K_pad), static_cast<int>(N_pad));
    return std::make_tuple(packed, K, N);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused(
    at::Tensor hidden_states,
    at::Tensor fused_wqa_wkv_packed,
    int64_t fused_wqa_wkv_K,
    int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed,
    int64_t compressor_kv_score_K,
    int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed,
    int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N,
    at::Tensor indexer_weights_proj_packed,
    int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N) {
#ifndef __aarch64__
    TORCH_CHECK(false, "deepseek_v4 attn gemm fused requires AArch64");
#else
    check_bf16_cpu_2d(hidden_states, "hidden_states");
    const int64_t M = hidden_states.size(0);
    const int64_t K = hidden_states.size(1);
    check_int_arg(K, "hidden_states.size(1)");
    TORCH_CHECK(M >= 0, "hidden_states.size(0) must be non-negative");
    TORCH_CHECK(M <= std::numeric_limits<int>::max(),
                "hidden_states.size(0) exceeds int32 kernel limit: ", M);

    PackedWeight fused_wqa_wkv = checked_packed_weight(
        fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N,
        "fused_wqa_wkv");
    PackedWeight compressor_kv_score = checked_packed_weight(
        compressor_kv_score_packed, compressor_kv_score_K,
        compressor_kv_score_N, "compressor_kv_score");
    PackedWeight indexer_compressor_kv_score = checked_packed_weight(
        indexer_compressor_kv_score_packed, indexer_compressor_kv_score_K,
        indexer_compressor_kv_score_N, "indexer_compressor_kv_score");
    PackedWeight indexer_weights_proj = checked_packed_weight(
        indexer_weights_proj_packed, indexer_weights_proj_K,
        indexer_weights_proj_N, "indexer_weights_proj");

    TORCH_CHECK(fused_wqa_wkv.K == K,
                "fused_wqa_wkv K mismatch: hidden K=", K,
                ", weight K=", fused_wqa_wkv.K);
    TORCH_CHECK(compressor_kv_score.K == K,
                "compressor_kv_score K mismatch: hidden K=", K,
                ", weight K=", compressor_kv_score.K);
    TORCH_CHECK(indexer_compressor_kv_score.K == K,
                "indexer_compressor_kv_score K mismatch: hidden K=", K,
                ", weight K=", indexer_compressor_kv_score.K);
    TORCH_CHECK(indexer_weights_proj.K == K,
                "indexer_weights_proj K mismatch: hidden K=", K,
                ", weight K=", indexer_weights_proj.K);

    const int64_t K_pad = fused_wqa_wkv.K_pad;
    TORCH_CHECK(compressor_kv_score.K_pad == K_pad &&
                    indexer_compressor_kv_score.K_pad == K_pad &&
                    indexer_weights_proj.K_pad == K_pad,
                "all packed weights must share the same padded K");

    at::Tensor a_storage;
    if (K_pad == K && hidden_states.is_contiguous()) {
        a_storage = hidden_states;
    } else {
        a_storage = at::zeros({M, K_pad}, hidden_states.options());
        a_storage.narrow(1, 0, K).copy_(hidden_states);
    }

    at::Tensor scratch = at::empty(
        {std::max<int64_t>(1, std::max<int64_t>(M, 1) * K_pad * 2)},
        hidden_states.options());

    at::Tensor qr_kv = run_bf16_gemm(a_storage, fused_wqa_wkv, scratch);
    at::Tensor kv_score = run_fp32_gemm(a_storage, compressor_kv_score,
                                        scratch);
    at::Tensor indexer_kv_score = run_fp32_gemm(
        a_storage, indexer_compressor_kv_score, scratch);
    at::Tensor indexer_weights = run_bf16_gemm(a_storage, indexer_weights_proj,
                                               scratch);

    return std::make_tuple(qr_kv, kv_score, indexer_kv_score,
                           indexer_weights);
#endif
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused_mt(
    at::Tensor hidden_states,
    at::Tensor fused_wqa_wkv_packed,
    int64_t fused_wqa_wkv_K,
    int64_t fused_wqa_wkv_N,
    at::Tensor compressor_kv_score_packed,
    int64_t compressor_kv_score_K,
    int64_t compressor_kv_score_N,
    at::Tensor indexer_compressor_kv_score_packed,
    int64_t indexer_compressor_kv_score_K,
    int64_t indexer_compressor_kv_score_N,
    at::Tensor indexer_weights_proj_packed,
    int64_t indexer_weights_proj_K,
    int64_t indexer_weights_proj_N,
    std::vector<int64_t> core_ids) {
#ifndef __aarch64__
    TORCH_CHECK(false, "deepseek_v4 attn gemm fused requires AArch64");
#elif !defined(_OPENMP)
    TORCH_CHECK(core_ids.size() <= 1,
                "deepseek_v4 attn gemm fused mt requires OpenMP when more "
                "than one core is requested");
    return fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused(
        hidden_states,
        fused_wqa_wkv_packed,
        fused_wqa_wkv_K,
        fused_wqa_wkv_N,
        compressor_kv_score_packed,
        compressor_kv_score_K,
        compressor_kv_score_N,
        indexer_compressor_kv_score_packed,
        indexer_compressor_kv_score_K,
        indexer_compressor_kv_score_N,
        indexer_weights_proj_packed,
        indexer_weights_proj_K,
        indexer_weights_proj_N);
#else
    if (core_ids.empty()) {
        return fused_wqa_wkv_compressor_kv_score_indexer_compressor_kv_score_indexer_weights_proj_fused(
            hidden_states,
            fused_wqa_wkv_packed,
            fused_wqa_wkv_K,
            fused_wqa_wkv_N,
            compressor_kv_score_packed,
            compressor_kv_score_K,
            compressor_kv_score_N,
            indexer_compressor_kv_score_packed,
            indexer_compressor_kv_score_K,
            indexer_compressor_kv_score_N,
            indexer_weights_proj_packed,
            indexer_weights_proj_K,
            indexer_weights_proj_N);
    }

    check_bf16_cpu_2d(hidden_states, "hidden_states");
    const int64_t M = hidden_states.size(0);
    const int64_t K = hidden_states.size(1);
    check_int_arg(K, "hidden_states.size(1)");
    TORCH_CHECK(M >= 0, "hidden_states.size(0) must be non-negative");
    TORCH_CHECK(M <= std::numeric_limits<int>::max(),
                "hidden_states.size(0) exceeds int32 kernel limit: ", M);
    TORCH_CHECK(core_ids.size() <=
                    static_cast<size_t>(std::numeric_limits<int>::max()),
                "core_ids size exceeds int32 limit");
    for (int64_t core_id : core_ids) {
        TORCH_CHECK(core_id >= 0, "core_ids must be non-negative, got ",
                    core_id);
    }

    PackedWeight fused_wqa_wkv = checked_packed_weight(
        fused_wqa_wkv_packed, fused_wqa_wkv_K, fused_wqa_wkv_N,
        "fused_wqa_wkv");
    PackedWeight compressor_kv_score = checked_packed_weight(
        compressor_kv_score_packed, compressor_kv_score_K,
        compressor_kv_score_N, "compressor_kv_score");
    PackedWeight indexer_compressor_kv_score = checked_packed_weight(
        indexer_compressor_kv_score_packed, indexer_compressor_kv_score_K,
        indexer_compressor_kv_score_N, "indexer_compressor_kv_score");
    PackedWeight indexer_weights_proj = checked_packed_weight(
        indexer_weights_proj_packed, indexer_weights_proj_K,
        indexer_weights_proj_N, "indexer_weights_proj");

    TORCH_CHECK(fused_wqa_wkv.K == K,
                "fused_wqa_wkv K mismatch: hidden K=", K,
                ", weight K=", fused_wqa_wkv.K);
    TORCH_CHECK(compressor_kv_score.K == K,
                "compressor_kv_score K mismatch: hidden K=", K,
                ", weight K=", compressor_kv_score.K);
    TORCH_CHECK(indexer_compressor_kv_score.K == K,
                "indexer_compressor_kv_score K mismatch: hidden K=", K,
                ", weight K=", indexer_compressor_kv_score.K);
    TORCH_CHECK(indexer_weights_proj.K == K,
                "indexer_weights_proj K mismatch: hidden K=", K,
                ", weight K=", indexer_weights_proj.K);

    const int64_t K_pad = fused_wqa_wkv.K_pad;
    TORCH_CHECK(compressor_kv_score.K_pad == K_pad &&
                    indexer_compressor_kv_score.K_pad == K_pad &&
                    indexer_weights_proj.K_pad == K_pad,
                "all packed weights must share the same padded K");

    at::Tensor a_storage;
    if (K_pad == K && hidden_states.is_contiguous()) {
        a_storage = hidden_states;
    } else {
        a_storage = at::zeros({M, K_pad}, hidden_states.options());
        a_storage.narrow(1, 0, K).copy_(hidden_states);
    }

    const int64_t num_threads = static_cast<int64_t>(core_ids.size());
    const int64_t rows_per_thread = ceil_div_int64(std::max<int64_t>(M, 1),
                                                   num_threads);
    const int64_t scratch_stride =
        std::max<int64_t>(1, rows_per_thread * K_pad * 2);
    at::Tensor scratch = at::empty({num_threads * scratch_stride},
                                   hidden_states.options());

#if defined(__APPLE__)
    at::Tensor qr_kv_acc = at::zeros(
        {M, fused_wqa_wkv.N_pad},
        at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));
    at::Tensor indexer_weights_acc = at::zeros(
        {M, indexer_weights_proj.N_pad},
        at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));
#else
    at::Tensor qr_kv_acc = at::empty({M, fused_wqa_wkv.N_pad},
                                     hidden_states.options());
    at::Tensor indexer_weights_acc = at::empty(
        {M, indexer_weights_proj.N_pad}, hidden_states.options());
#endif
    at::Tensor kv_score_acc = at::zeros(
        {M, compressor_kv_score.N_pad},
        at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));
    at::Tensor indexer_kv_score_acc = at::zeros(
        {M, indexer_compressor_kv_score.N_pad},
        at::TensorOptions().device(hidden_states.device()).dtype(at::kFloat));

    std::vector<int> bind_failed(static_cast<size_t>(num_threads), 0);
    const int old_dynamic = omp_get_dynamic();
    omp_set_dynamic(0);

#pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        if (tid < static_cast<int>(num_threads) &&
            !bind_current_thread_to_cpu(core_ids[static_cast<size_t>(tid)])) {
            bind_failed[static_cast<size_t>(tid)] = 1;
        }

        const int64_t row_start = static_cast<int64_t>(tid) * rows_per_thread;
        const int64_t row_count =
            row_start >= M ? 0 : std::min<int64_t>(rows_per_thread,
                                                   M - row_start);
        const int64_t scratch_offset =
            static_cast<int64_t>(tid) * scratch_stride;

#if defined(__APPLE__)
        dispatch_fp32_gemm_to_output(a_storage, fused_wqa_wkv, qr_kv_acc,
                                     scratch, row_start, row_count,
                                     scratch_offset);
#else
        dispatch_bf16_gemm_to_output(a_storage, fused_wqa_wkv, qr_kv_acc,
                                     scratch, row_start, row_count,
                                     scratch_offset);
#endif
        dispatch_fp32_gemm_to_output(a_storage, compressor_kv_score,
                                     kv_score_acc, scratch, row_start,
                                     row_count, scratch_offset);
        dispatch_fp32_gemm_to_output(a_storage, indexer_compressor_kv_score,
                                     indexer_kv_score_acc, scratch, row_start,
                                     row_count, scratch_offset);
#if defined(__APPLE__)
        dispatch_fp32_gemm_to_output(a_storage, indexer_weights_proj,
                                     indexer_weights_acc, scratch, row_start,
                                     row_count, scratch_offset);
#else
        dispatch_bf16_gemm_to_output(a_storage, indexer_weights_proj,
                                     indexer_weights_acc, scratch, row_start,
                                     row_count, scratch_offset);
#endif
    }

    omp_set_dynamic(old_dynamic);
    for (size_t i = 0; i < bind_failed.size(); ++i) {
        TORCH_CHECK(bind_failed[i] == 0,
                    "failed to bind OpenMP thread ", i, " to CPU ",
                    core_ids[i]);
    }

#if defined(__APPLE__)
    at::Tensor qr_kv = narrow_output_if_needed(qr_kv_acc, fused_wqa_wkv.N,
                                               fused_wqa_wkv.N_pad)
                           .to(at::kBFloat16);
    at::Tensor indexer_weights =
        narrow_output_if_needed(indexer_weights_acc, indexer_weights_proj.N,
                                indexer_weights_proj.N_pad)
            .to(at::kBFloat16);
#else
    at::Tensor qr_kv = narrow_output_if_needed(qr_kv_acc, fused_wqa_wkv.N,
                                               fused_wqa_wkv.N_pad);
    at::Tensor indexer_weights =
        narrow_output_if_needed(indexer_weights_acc, indexer_weights_proj.N,
                                indexer_weights_proj.N_pad);
#endif
    at::Tensor kv_score = narrow_output_if_needed(
        kv_score_acc, compressor_kv_score.N, compressor_kv_score.N_pad);
    at::Tensor indexer_kv_score = narrow_output_if_needed(
        indexer_kv_score_acc, indexer_compressor_kv_score.N,
        indexer_compressor_kv_score.N_pad);

    return std::make_tuple(qr_kv, kv_score, indexer_kv_score,
                           indexer_weights);
#endif
}
