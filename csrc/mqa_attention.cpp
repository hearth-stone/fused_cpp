#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "sdpa_microkernels/impls/mk_qk_packqk_seq4_bmajor_pv_pquad.h"
#include "utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

struct MqaParams {
    int64_t B = 0;
    int64_t N = 0;
    int64_t L = 0;
    int64_t S = 0;
    int64_t E = 0;
    int64_t Ev = 0;
    float scale = 0.0f;
    int64_t causal_offset = 0;
    bool is_causal = false;

    const float* mask_ptr = nullptr;
    int64_t mask_stride_b = 0;
    int64_t mask_stride_n = 0;
    int64_t mask_stride_l = 0;
    int64_t mask_stride_s = 0;
};

template <typename scalar_t>
inline void mqa_attention_kernel_tmpl(
    const scalar_t* q_ptr,
    const scalar_t* k_ptr,
    const scalar_t* v_ptr,
    float* out_ptr,
    const MqaParams& p) {
    const int64_t q_stride_b = p.N * p.L * p.E;
    const int64_t q_stride_n = p.L * p.E;
    const int64_t q_stride_l = p.E;
    const int64_t k_stride_b = p.S * p.E;
    const int64_t k_stride_s = p.E;
    const int64_t v_stride_b = p.S * p.Ev;
    const int64_t v_stride_s = p.Ev;
    const int64_t o_stride_b = p.N * p.L * p.Ev;
    const int64_t o_stride_n = p.L * p.Ev;
    const int64_t o_stride_l = p.Ev;

    const int64_t total_rows = p.B * p.N * p.L;

    auto compute_row = [&](int64_t row_idx,
                           std::vector<float>& scores,
                           std::vector<float>& acc) {
        const int64_t b = row_idx / (p.N * p.L);
        const int64_t rem = row_idx - b * p.N * p.L;
        const int64_t n = rem / p.L;
        const int64_t l = rem - n * p.L;

        const scalar_t* q_row = q_ptr + b * q_stride_b
                                      + n * q_stride_n
                                      + l * q_stride_l;
        float* o_row = out_ptr + b * o_stride_b
                               + n * o_stride_n
                               + l * o_stride_l;

        const int64_t causal_limit = l + p.causal_offset;
        float row_max = -std::numeric_limits<float>::infinity();

        for (int64_t s = 0; s < p.S; ++s) {
            if (p.is_causal && s > causal_limit) {
                scores[static_cast<size_t>(s)] =
                    -std::numeric_limits<float>::infinity();
                continue;
            }
            const scalar_t* k_row = k_ptr + b * k_stride_b + s * k_stride_s;
            float dot = 0.0f;
            for (int64_t e = 0; e < p.E; ++e) {
                dot += static_cast<float>(q_row[e]) *
                       static_cast<float>(k_row[e]);
            }
            float score = dot * p.scale;
            if (p.mask_ptr != nullptr) {
                score += p.mask_ptr[b * p.mask_stride_b
                                  + n * p.mask_stride_n
                                  + l * p.mask_stride_l
                                  + s * p.mask_stride_s];
            }
            scores[static_cast<size_t>(s)] = score;
            row_max = std::max(row_max, score);
        }

        std::fill(acc.begin(), acc.end(), 0.0f);
        float row_sum = 0.0f;
        if (std::isfinite(row_max)) {
            for (int64_t s = 0; s < p.S; ++s) {
                const float score = scores[static_cast<size_t>(s)];
                if (!std::isfinite(score)) {
                    continue;
                }
                const float weight = std::exp(score - row_max);
                row_sum += weight;
                const scalar_t* v_row = v_ptr + b * v_stride_b + s * v_stride_s;
                for (int64_t ev = 0; ev < p.Ev; ++ev) {
                    acc[static_cast<size_t>(ev)] +=
                        weight * static_cast<float>(v_row[ev]);
                }
            }
        }

        if (row_sum > 0.0f) {
            const float inv_sum = 1.0f / row_sum;
            for (int64_t ev = 0; ev < p.Ev; ++ev) {
                o_row[ev] = acc[static_cast<size_t>(ev)] * inv_sum;
            }
        } else {
            for (int64_t ev = 0; ev < p.Ev; ++ev) {
                o_row[ev] = 0.0f;
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<float> scores(static_cast<size_t>(p.S));
        std::vector<float> acc(static_cast<size_t>(p.Ev));
#pragma omp for schedule(static)
        for (int64_t row_idx = 0; row_idx < total_rows; ++row_idx) {
            compute_row(row_idx, scores, acc);
        }
    }
#else
    std::vector<float> scores(static_cast<size_t>(p.S));
    std::vector<float> acc(static_cast<size_t>(p.Ev));
    for (int64_t row_idx = 0; row_idx < total_rows; ++row_idx) {
        compute_row(row_idx, scores, acc);
    }
#endif
}

#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_BMAJOR_PV_PQUAD
inline void mqa_attention_bf16_nomask_tiled(
    const at::BFloat16* q_ptr,
    const at::BFloat16* k_ptr,
    const at::BFloat16* v_ptr,
    float* out_ptr,
    const MqaParams& p) {
    using MK = ::fused_cpp::sdpa_microkernels::MK_QkPackqkSeq4BmajorPvPquad;

    constexpr int64_t kTileL = 8;
    constexpr int64_t kTileS = 8;
    constexpr int64_t kTileEv = 8;

    const int64_t q_stride_b = p.N * p.L * p.E;
    const int64_t q_stride_n = p.L * p.E;
    const int64_t q_stride_l = p.E;
    const int64_t k_stride_b = p.S * p.E;
    const int64_t k_stride_s = p.E;
    const int64_t v_stride_b = p.S * p.Ev;
    const int64_t v_stride_s = p.Ev;
    const int64_t o_stride_b = p.N * p.L * p.Ev;
    const int64_t o_stride_n = p.L * p.Ev;
    const int64_t o_stride_l = p.Ev;

    const int64_t num_l_tiles = (p.L + kTileL - 1) / kTileL;
    const int64_t total_tiles = p.B * p.N * num_l_tiles;

    auto compute_tile = [&](int64_t tile_idx, std::vector<float>& scores) {
        const int64_t b = tile_idx / (p.N * num_l_tiles);
        const int64_t rem = tile_idx - b * p.N * num_l_tiles;
        const int64_t n = rem / num_l_tiles;
        const int64_t lt = rem - n * num_l_tiles;
        const int64_t l0 = lt * kTileL;
        const int lq = static_cast<int>(std::min<int64_t>(kTileL, p.L - l0));

        const at::BFloat16* q_tile = q_ptr + b * q_stride_b
                                           + n * q_stride_n
                                           + l0 * q_stride_l;
        const at::BFloat16* k_base = k_ptr + b * k_stride_b;
        const at::BFloat16* v_base = v_ptr + b * v_stride_b;
        float* o_tile = out_ptr + b * o_stride_b
                                + n * o_stride_n
                                + l0 * o_stride_l;

        for (int64_t s0 = 0; s0 < p.S; s0 += kTileS) {
            const int sk = static_cast<int>(
                std::min<int64_t>(kTileS, p.S - s0));
            const at::BFloat16* k_tile = k_base + s0 * k_stride_s;
            float* score_tile = scores.data() + s0;

            if (lq == kTileL && sk == kTileS) {
                alignas(64) float tmp_qkt[kTileL * kTileS];
                MK::qkt_8x8(q_tile, q_stride_l, k_tile, k_stride_s,
                            p.E, p.scale, tmp_qkt);
                for (int i = 0; i < kTileL; ++i) {
                    std::copy_n(tmp_qkt + i * kTileS, kTileS,
                                scores.data() + i * p.S + s0);
                }
            } else if (lq == kTileL && sk == 4) {
                MK::qkt_8x4(q_tile, q_stride_l, k_tile, k_stride_s,
                            p.E, p.scale, score_tile, p.S);
            } else {
                MK::qkt_tail(q_tile, q_stride_l, k_tile, k_stride_s,
                             p.E, p.scale, score_tile, p.S, lq, sk);
            }
        }

        for (int i = 0; i < lq; ++i) {
            float* row = scores.data() + i * p.S;
            if (p.is_causal) {
                const int64_t causal_limit = l0 + i + p.causal_offset;
                for (int64_t s = 0; s < p.S; ++s) {
                    if (s > causal_limit) {
                        row[s] = -std::numeric_limits<float>::infinity();
                    }
                }
            }

            float row_max = -std::numeric_limits<float>::infinity();
            for (int64_t s = 0; s < p.S; ++s) {
                row_max = std::max(row_max, row[s]);
            }

            float row_sum = 0.0f;
            if (std::isfinite(row_max)) {
                for (int64_t s = 0; s < p.S; ++s) {
                    if (!std::isfinite(row[s])) {
                        row[s] = 0.0f;
                        continue;
                    }
                    const float weight = std::exp(row[s] - row_max);
                    row[s] = weight;
                    row_sum += weight;
                }
            }

            if (row_sum > 0.0f) {
                const float inv_sum = 1.0f / row_sum;
                for (int64_t s = 0; s < p.S; ++s) {
                    row[s] *= inv_sum;
                }
            } else {
                std::fill(row, row + p.S, 0.0f);
            }
        }

        std::fill(o_tile, o_tile + static_cast<int64_t>(lq) * p.Ev, 0.0f);
        for (int64_t ev0 = 0; ev0 < p.Ev; ev0 += kTileEv) {
            const int ev = static_cast<int>(
                std::min<int64_t>(kTileEv, p.Ev - ev0));
            const at::BFloat16* v_tile = v_base + ev0;
            float* o_block = o_tile + ev0;

            if (lq == kTileL && ev == kTileEv) {
                MK::pv_8x8(scores.data(), p.S, v_tile, v_stride_s,
                           p.S, o_block, o_stride_l);
            } else {
                MK::pv_tail(scores.data(), p.S, v_tile, v_stride_s,
                            p.S, o_block, o_stride_l, lq, ev);
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel
    {
        std::vector<float> scores(static_cast<size_t>(kTileL * p.S));
#pragma omp for schedule(static)
        for (int64_t tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
            compute_tile(tile_idx, scores);
        }
    }
#else
    std::vector<float> scores(static_cast<size_t>(kTileL * p.S));
    for (int64_t tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
        compute_tile(tile_idx, scores);
    }
#endif
}
#endif

inline at::Tensor normalize_mqa_kv(
    const at::Tensor& tensor,
    const char* name,
    int64_t B,
    int64_t S,
    int64_t D) {
    TORCH_CHECK(
        tensor.dim() == 3 || tensor.dim() == 4,
        "multi_query_attention: ", name,
        " must be [B, S, D] or [B, 1, S, D], got ",
        tensor.dim(), "-D");
    if (tensor.dim() == 3) {
        TORCH_CHECK(
            tensor.size(0) == B && tensor.size(1) == S && tensor.size(2) == D,
            "multi_query_attention: ", name, " shape mismatch, expected [",
            B, ", ", S, ", ", D, "], got ", tensor.sizes());
        return ensure_contiguous(tensor);
    }
    TORCH_CHECK(
        tensor.size(0) == B && tensor.size(1) == 1 &&
        tensor.size(2) == S && tensor.size(3) == D,
        "multi_query_attention: ", name, " shape mismatch, expected [",
        B, ", 1, ", S, ", ", D, "], got ", tensor.sizes());
    return ensure_contiguous(tensor.select(1, 0));
}

inline void fill_mask_strides(
    const at::Tensor& mask,
    int64_t B,
    int64_t N,
    int64_t L,
    int64_t S,
    MqaParams& p) {
    TORCH_CHECK(
        mask.dim() == 2 || mask.dim() == 3 || mask.dim() == 4,
        "multi_query_attention: attn_mask must be [L, S], [B, L, S], "
        "[B, 1, L, S], or [B, N, L, S], got ", mask.dim(), "-D");

    if (mask.dim() == 2) {
        TORCH_CHECK(
            mask.size(0) == L && mask.size(1) == S,
            "multi_query_attention: attn_mask [L, S] shape mismatch, got ",
            mask.sizes());
        p.mask_stride_b = 0;
        p.mask_stride_n = 0;
        p.mask_stride_l = S;
        p.mask_stride_s = 1;
        return;
    }

    if (mask.dim() == 3) {
        TORCH_CHECK(
            (mask.size(0) == B || mask.size(0) == 1) &&
            mask.size(1) == L && mask.size(2) == S,
            "multi_query_attention: attn_mask [B, L, S] shape mismatch, got ",
            mask.sizes());
        p.mask_stride_b = mask.size(0) == 1 ? 0 : L * S;
        p.mask_stride_n = 0;
        p.mask_stride_l = S;
        p.mask_stride_s = 1;
        return;
    }

    TORCH_CHECK(
        (mask.size(0) == B || mask.size(0) == 1) &&
        (mask.size(1) == N || mask.size(1) == 1) &&
        mask.size(2) == L && mask.size(3) == S,
        "multi_query_attention: attn_mask [B, N, L, S] shape mismatch, got ",
        mask.sizes());
    p.mask_stride_b = mask.size(0) == 1 ? 0 : mask.size(1) * L * S;
    p.mask_stride_n = mask.size(1) == 1 ? 0 : L * S;
    p.mask_stride_l = S;
    p.mask_stride_s = 1;
}

}  // namespace

at::Tensor multi_query_attention(
    at::Tensor query,
    at::Tensor key,
    at::Tensor value,
    c10::optional<at::Tensor> attn_mask,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
    torch::NoGradGuard no_grad;

    TORCH_CHECK(
        query.dim() == 4,
        "multi_query_attention: query must be [B, N, L, E], got ",
        query.dim(), "-D");
    TORCH_CHECK(
        key.scalar_type() == query.scalar_type() &&
        value.scalar_type() == query.scalar_type(),
        "multi_query_attention: query/key/value dtypes must match");

    if (dropout_p != 0.0) {
        TORCH_WARN("multi_query_attention: dropout_p=", dropout_p,
                   " is ignored (inference only)");
    }

    auto orig_dtype = query.scalar_type();
    auto q = ensure_contiguous(query);

    const int64_t B = q.size(0);
    const int64_t N = q.size(1);
    const int64_t L = q.size(2);
    const int64_t E = q.size(3);
    TORCH_CHECK(N > 0 && L > 0 && E > 0,
                "multi_query_attention: query dimensions must be non-zero");

    int64_t S = 0;
    if (key.dim() == 3) {
        TORCH_CHECK(key.size(0) == B && key.size(2) == E,
                    "multi_query_attention: key must be [B, S, E], got ",
                    key.sizes());
        S = key.size(1);
    } else {
        TORCH_CHECK(key.dim() == 4 && key.size(0) == B && key.size(1) == 1 &&
                    key.size(3) == E,
                    "multi_query_attention: key must be [B, S, E] or "
                    "[B, 1, S, E], got ", key.sizes());
        S = key.size(2);
    }
    TORCH_CHECK(S > 0, "multi_query_attention: key seq_len must be non-zero");

    int64_t Ev = 0;
    if (value.dim() == 3) {
        TORCH_CHECK(value.size(0) == B && value.size(1) == S,
                    "multi_query_attention: value must be [B, S, Ev], got ",
                    value.sizes());
        Ev = value.size(2);
    } else {
        TORCH_CHECK(value.dim() == 4 && value.size(0) == B &&
                    value.size(1) == 1 && value.size(2) == S,
                    "multi_query_attention: value must be [B, S, Ev] or "
                    "[B, 1, S, Ev], got ", value.sizes());
        Ev = value.size(3);
    }
    TORCH_CHECK(Ev > 0, "multi_query_attention: value head_dim must be non-zero");

    auto k = normalize_mqa_kv(key, "key", B, S, E);
    auto v = normalize_mqa_kv(value, "value", B, S, Ev);

    if (orig_dtype == at::kBFloat16) {
        // keep bf16
    } else if (orig_dtype == at::kFloat) {
        // keep fp32
    } else {
        q = q.to(at::kFloat);
        k = k.to(at::kFloat);
        v = v.to(at::kFloat);
    }

    at::Tensor mask_fp32;
    MqaParams p{};
    p.B = B;
    p.N = N;
    p.L = L;
    p.S = S;
    p.E = E;
    p.Ev = Ev;
    p.scale = scale.has_value()
        ? static_cast<float>(scale.value())
        : static_cast<float>(1.0 / std::sqrt(static_cast<double>(E)));
    p.causal_offset = S - L;
    p.is_causal = is_causal;

    if (attn_mask.has_value()) {
        mask_fp32 = ensure_contiguous(attn_mask.value().to(at::kFloat));
        p.mask_ptr = mask_fp32.data_ptr<float>();
        fill_mask_strides(mask_fp32, B, N, L, S, p);
    }

    auto output_fp32 = at::empty({B, N, L, Ev}, q.options().dtype(at::kFloat));

    if (orig_dtype == at::kBFloat16) {
#if FUSED_CPP_MK_ENABLE_QK_PACKQK_SEQ4_BMAJOR_PV_PQUAD
        if (p.mask_ptr == nullptr) {
            mqa_attention_bf16_nomask_tiled(
                q.data_ptr<at::BFloat16>(),
                k.data_ptr<at::BFloat16>(),
                v.data_ptr<at::BFloat16>(),
                output_fp32.data_ptr<float>(),
                p);
        } else {
            mqa_attention_kernel_tmpl<at::BFloat16>(
                q.data_ptr<at::BFloat16>(),
                k.data_ptr<at::BFloat16>(),
                v.data_ptr<at::BFloat16>(),
                output_fp32.data_ptr<float>(),
                p);
        }
#else
        mqa_attention_kernel_tmpl<at::BFloat16>(
            q.data_ptr<at::BFloat16>(),
            k.data_ptr<at::BFloat16>(),
            v.data_ptr<at::BFloat16>(),
            output_fp32.data_ptr<float>(),
            p);
#endif
    } else {
        mqa_attention_kernel_tmpl<float>(
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            output_fp32.data_ptr<float>(),
            p);
    }

    if (orig_dtype != at::kFloat) {
        return output_fp32.to(orig_dtype);
    }
    return output_fp32;
}
