#include "clean_sdpa.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "../../csrc/sdpa_common.h"
#include "../../csrc/sdpa_tile_sizes.h"
#include "../../csrc/sdpa_pack_utils.h"
#include "../../csrc/sdpa_flash2_neon_l3kv_impl.h"

namespace clean_sdpa {
namespace {

using ::fused_cpp::sdpa_microkernels::pack_k_to_seq8;
using ::fused_cpp::sdpa_pack_utils::pack_v_to_evblock8;
using ::fused_cpp::sdpa_tile_sizes::compute_tile_sizes_l3kv;
using ::fused_cpp::sdpa_tile_sizes::TileSizes;
using ::fused_cpp::sdpa_flash2_neon_l3kv_impl::run_path_collapse3_packqkv;

constexpr int64_t kEvBlock = 8;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

inline int64_t clamp_i64(int64_t v, int64_t lo, int64_t hi) {
  return std::max(lo, std::min(v, hi));
}

inline void check_config(const Config& cfg) {
  if (cfg.B <= 0 || cfg.N <= 0 || cfg.L <= 0 || cfg.S <= 0 ||
      cfg.E <= 0 || cfg.Ev <= 0) {
    throw std::invalid_argument("shape dimensions must be positive");
  }
  if (cfg.S % 8 != 0) {
    throw std::invalid_argument("clean SDPA requires S % 8 == 0");
  }
  if (cfg.Ev % kEvBlock != 0) {
    throw std::invalid_argument("clean SDPA requires Ev % 8 == 0");
  }
  if (cfg.s_tile < 0 || cfg.s_tile % 8 != 0) {
    throw std::invalid_argument("clean SDPA requires s_tile == 0 or s_tile % 8 == 0");
  }
}

}  // namespace

void sdpa_bf16_packqkv_pbf16pv(
    const at::BFloat16* q,
    const at::BFloat16* k,
    const at::BFloat16* v,
    float* out,
    const Config& cfg_in) {
  Config cfg = cfg_in;
  check_config(cfg);
  if (cfg.scale == 0.0f) {
    cfg.scale = 1.0f / std::sqrt(static_cast<float>(cfg.E));
  }

  SdpaParams p;
  p.B = cfg.B;
  p.N = cfg.N;
  p.L = cfg.L;
  p.S = cfg.S;
  p.E = cfg.E;
  p.Ev = cfg.Ev;
  p.scale_f = cfg.scale;
  p.neg_inf = kNegInf;
  p.causal_offset = cfg.causal_offset;
  p.is_causal = cfg.causal;
  p.dtype = SdpaDtype::kBFloat16;
  p.q_ptr = q;
  p.k_ptr = k;
  p.v_ptr = v;
  p.mask_ptr = nullptr;
  p.out_ptr = out;

  const bool profile_on = ::fused_cpp::sdpa_profile::enabled();
  if (profile_on) {
    ::fused_cpp::sdpa_profile::reset();
  }
  const uint64_t profile_total_t0 =
      profile_on ? ::fused_cpp::sdpa_profile::now_ns() : 0;

  const int64_t eb = cfg.Ev / 8;
  const int64_t e_main = cfg.E & ~int64_t{3};
  const int64_t s_blocks = cfg.S / 8;
  const int64_t kblock_u16 = (e_main / 4) * 32;
  const int64_t k_packed_n_stride = s_blocks * kblock_u16;
  const int64_t k_packed_b_stride = cfg.N * k_packed_n_stride;
  const int64_t k_sblock_stride = kblock_u16;

  AlignedVector<at::BFloat16> v_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVAlloc);
    v_packed.resize(static_cast<size_t>(cfg.B * cfg.N * eb * cfg.S * 8));
  }
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kVPack);
    pack_v_to_evblock8<at::BFloat16>(
        v, v_packed.data(), cfg.B, cfg.N, cfg.S, cfg.Ev);
  }

  AlignedVector<uint16_t> k_packed;
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKAlloc);
    k_packed.resize(
        static_cast<size_t>(cfg.B * cfg.N * s_blocks * kblock_u16));
  }
  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kKPack);
    pack_k_to_seq8<at::BFloat16>(
        k, k_packed.data(), cfg.B, cfg.N, cfg.S, cfg.E);
  }

  const int64_t q_b_stride = cfg.N * cfg.L * cfg.E;
  const int64_t q_n_stride = cfg.L * cfg.E;
  const int64_t q_l_stride = cfg.E;
  const int64_t k_b_stride = cfg.N * cfg.S * cfg.E;
  const int64_t k_n_stride = cfg.S * cfg.E;
  const int64_t k_s_stride = cfg.E;
  const int64_t v_packed_b_stride = cfg.N * eb * cfg.S * 8;
  const int64_t v_packed_n_stride = eb * cfg.S * 8;
  const int64_t v_packed_s_stride = 8;
  const int64_t v_packed_eb_stride = cfg.S * 8;
  const int64_t out_b_stride = cfg.N * cfg.L * cfg.Ev;
  const int64_t out_n_stride = cfg.L * cfg.Ev;
  const int64_t out_l_stride = cfg.Ev;
  constexpr int64_t m_stride_b = 0;
  constexpr int64_t m_stride_n = 0;
  constexpr int64_t m_stride_l = 0;

  TileSizes ts = compute_tile_sizes_l3kv(
      cfg.B, cfg.N, cfg.S, cfg.L, cfg.E, cfg.Ev, sizeof(at::BFloat16));
  if (cfg.s_tile > 0) {
    ts.Sc_l2 = cfg.s_tile;
    ts.Sc_l3 = std::max<int64_t>(ts.Sc_l2, std::min<int64_t>(cfg.S, ts.Sc_l3));
    ts.Sc_l3 = (ts.Sc_l3 / ts.Sc_l2) * ts.Sc_l2;
    if (ts.Sc_l3 < ts.Sc_l2) {
      ts.Sc_l3 = ts.Sc_l2;
    }
    if (ts.Sc_l3 > cfg.S) {
      ts.Sc_l3 = cfg.S;
    }
  }

  {
    FUSED_CPP_SDPA_PROFILE_SCOPE(::fused_cpp::sdpa_profile::Slot::kMain);
    if (cfg.causal) {
      run_path_collapse3_packqkv<false, true, true>(
          q, k_packed.data(), k, v_packed.data(), p, ts,
          q_b_stride, q_n_stride, q_l_stride,
          k_b_stride, k_n_stride, k_s_stride,
          k_packed_b_stride, k_packed_n_stride, k_sblock_stride,
          v_packed_b_stride, v_packed_n_stride, v_packed_s_stride,
          v_packed_eb_stride,
          m_stride_b, m_stride_n, m_stride_l,
          out_b_stride, out_n_stride, out_l_stride);
    } else {
      run_path_collapse3_packqkv<false, false, true>(
          q, k_packed.data(), k, v_packed.data(), p, ts,
          q_b_stride, q_n_stride, q_l_stride,
          k_b_stride, k_n_stride, k_s_stride,
          k_packed_b_stride, k_packed_n_stride, k_sblock_stride,
          v_packed_b_stride, v_packed_n_stride, v_packed_s_stride,
          v_packed_eb_stride,
          m_stride_b, m_stride_n, m_stride_l,
          out_b_stride, out_n_stride, out_l_stride);
    }
  }

  if (profile_on) {
    ::fused_cpp::sdpa_profile::add(
        ::fused_cpp::sdpa_profile::Slot::kTotal,
        ::fused_cpp::sdpa_profile::now_ns() - profile_total_t0);
    ::fused_cpp::sdpa_profile::print_summary(
        "clean_sdpa",
        "qk_packqk_seq4_bmajor_pv_pbf16_prepacked",
        p,
        "A");
  }
}

void reference_sdpa_bf16(
    const at::BFloat16* q,
    const at::BFloat16* k,
    const at::BFloat16* v,
    float* out,
    const Config& cfg_in) {
  Config cfg = cfg_in;
  check_config(cfg);
  if (cfg.scale == 0.0f) {
    cfg.scale = 1.0f / std::sqrt(static_cast<float>(cfg.E));
  }

  const int64_t q_b_stride = cfg.N * cfg.L * cfg.E;
  const int64_t q_n_stride = cfg.L * cfg.E;
  const int64_t k_b_stride = cfg.N * cfg.S * cfg.E;
  const int64_t k_n_stride = cfg.S * cfg.E;
  const int64_t v_b_stride = cfg.N * cfg.S * cfg.Ev;
  const int64_t v_n_stride = cfg.S * cfg.Ev;
  const int64_t out_b_stride = cfg.N * cfg.L * cfg.Ev;
  const int64_t out_n_stride = cfg.L * cfg.Ev;

  std::vector<float> scores(static_cast<size_t>(cfg.S));
  for (int64_t b = 0; b < cfg.B; ++b) {
    for (int64_t n = 0; n < cfg.N; ++n) {
      for (int64_t l = 0; l < cfg.L; ++l) {
        const at::BFloat16* q_row = q + b * q_b_stride + n * q_n_stride + l * cfg.E;
        const at::BFloat16* k_bn = k + b * k_b_stride + n * k_n_stride;
        const at::BFloat16* v_bn = v + b * v_b_stride + n * v_n_stride;
        float* out_row = out + b * out_b_stride + n * out_n_stride + l * cfg.Ev;

        const int64_t visible = cfg.causal
            ? clamp_i64(l + cfg.causal_offset + 1, 0, cfg.S)
            : cfg.S;

        float m = kNegInf;
        for (int64_t s = 0; s < visible; ++s) {
          const at::BFloat16* k_row = k_bn + s * cfg.E;
          float acc = 0.0f;
          for (int64_t e = 0; e < cfg.E; ++e) {
            acc += static_cast<float>(q_row[e]) * static_cast<float>(k_row[e]);
          }
          scores[s] = acc * cfg.scale;
          m = std::max(m, scores[s]);
        }

        std::fill(out_row, out_row + cfg.Ev, 0.0f);
        if (visible <= 0) {
          continue;
        }

        float denom = 0.0f;
        for (int64_t s = 0; s < visible; ++s) {
          const float p = std::exp(scores[s] - m);
          denom += p;
          const at::BFloat16* v_row = v_bn + s * cfg.Ev;
          for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
            out_row[ev] += p * static_cast<float>(v_row[ev]);
          }
        }
        const float inv = 1.0f / denom;
        for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
          out_row[ev] *= inv;
        }
      }
    }
  }
}

double counted_gflops(const Config& cfg, double mean_ms) {
  int64_t active_per_head = 0;
  if (!cfg.causal) {
    active_per_head = cfg.L * cfg.S;
  } else {
    for (int64_t l = 0; l < cfg.L; ++l) {
      active_per_head += clamp_i64(l + cfg.causal_offset + 1, 0, cfg.S);
    }
  }
  const double active =
      static_cast<double>(cfg.B) * static_cast<double>(cfg.N) *
      static_cast<double>(active_per_head);
  const double flops_per_score =
      2.0 * static_cast<double>(cfg.E) +
      2.0 * static_cast<double>(cfg.Ev) + 5.0;
  return active * flops_per_score / (mean_ms * 1.0e6);
}

double checksum(const float* data, int64_t size) {
  double sum = 0.0;
  for (int64_t i = 0; i < size; ++i) {
    sum += static_cast<double>(data[i]);
  }
  return sum;
}

double max_abs_diff(const float* a, const float* b, int64_t size) {
  double m = 0.0;
  for (int64_t i = 0; i < size; ++i) {
    m = std::max(m, static_cast<double>(std::abs(a[i] - b[i])));
  }
  return m;
}

}  // namespace clean_sdpa
