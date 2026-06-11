#include "sdpa_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

namespace {

const float * ptr_at(const float * base, int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
                    int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
  return reinterpret_cast<const float *>(
      reinterpret_cast<const char *>(base) + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
}

float * ptr_at(float * base, int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3,
               int64_t i0, int64_t i1, int64_t i2, int64_t i3) {
  return reinterpret_cast<float *>(
      reinterpret_cast<char *>(base) + i0 * nb0 + i1 * nb1 + i2 * nb2 + i3 * nb3);
}

}  // namespace

#if defined(_WIN32)
#define FUSED_CPP_LLAMA_API __declspec(dllexport)
#else
#define FUSED_CPP_LLAMA_API __attribute__((visibility("default")))
#endif

extern "C" FUSED_CPP_LLAMA_API int fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp(
    const float * q,
    const float * k,
    const float * v,
    float * out,
    int64_t B,
    int64_t H,
    int64_t L,
    int64_t S,
    int64_t D,
    int64_t DV,
    int64_t q_nb0,
    int64_t q_nb1,
    int64_t q_nb2,
    int64_t q_nb3,
    int64_t k_nb0,
    int64_t k_nb1,
    int64_t k_nb2,
    int64_t k_nb3,
    int64_t v_nb0,
    int64_t v_nb1,
    int64_t v_nb2,
    int64_t v_nb3,
    int64_t o_nb0,
    int64_t o_nb1,
    int64_t o_nb2,
    int64_t o_nb3,
    float scale) {
  try {
    if (q == nullptr || k == nullptr || v == nullptr || out == nullptr) {
      return 1;
    }
    if (B <= 0 || H <= 0 || L <= 0 || S <= 0 || D <= 0 || DV <= 0) {
      return 2;
    }
    if ((S % 8) != 0 || (DV % 8) != 0) {
      return 3;
    }
    const float effective_scale =
        scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(D));

    std::vector<float> q_dense(static_cast<size_t>(B * H * L * D));
    std::vector<float> k_dense(static_cast<size_t>(B * H * S * D));
    std::vector<float> v_dense(static_cast<size_t>(B * H * S * DV));
    std::vector<float> o_dense(static_cast<size_t>(B * H * L * DV));

    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t l = 0; l < L; ++l) {
          float * dst = q_dense.data() + ((b * H + h) * L + l) * D;
          const float * src = ptr_at(q, q_nb0, q_nb1, q_nb2, q_nb3, 0, l, h, b);
          std::memcpy(dst, src, static_cast<size_t>(D) * sizeof(float));
        }
        for (int64_t s = 0; s < S; ++s) {
          float * kd = k_dense.data() + ((b * H + h) * S + s) * D;
          const float * ks = ptr_at(k, k_nb0, k_nb1, k_nb2, k_nb3, 0, s, h, b);
          std::memcpy(kd, ks, static_cast<size_t>(D) * sizeof(float));

          float * vd = v_dense.data() + ((b * H + h) * S + s) * DV;
          const float * vs = ptr_at(v, v_nb0, v_nb1, v_nb2, v_nb3, 0, s, h, b);
          std::memcpy(vd, vs, static_cast<size_t>(DV) * sizeof(float));
        }
      }
    }

    SdpaParams p;
    p.B = B;
    p.N = H;
    p.L = L;
    p.S = S;
    p.E = D;
    p.Ev = DV;
    p.scale_f = effective_scale;
    p.neg_inf = -std::numeric_limits<float>::infinity();
    p.causal_offset = S - L;
    p.is_causal = false;
    p.dtype = SdpaDtype::kFloat32;
    p.q_ptr = q_dense.data();
    p.k_ptr = k_dense.data();
    p.v_ptr = v_dense.data();
    p.mask_ptr = nullptr;
    p.out_ptr = o_dense.data();

    sdpa_dispatch("flash2_neon_l3kv_packqkv_pbf16pv", p);

    for (int64_t b = 0; b < B; ++b) {
      for (int64_t h = 0; h < H; ++h) {
        for (int64_t l = 0; l < L; ++l) {
          const float * src = o_dense.data() + ((b * H + h) * L + l) * DV;
          float * dst = ptr_at(out, o_nb0, o_nb1, o_nb2, o_nb3, 0, h, l, b);
          std::memcpy(dst, src, static_cast<size_t>(DV) * sizeof(float));
        }
      }
    }

    return 0;
  } catch (const std::exception &) {
    return 100;
  } catch (...) {
    return 101;
  }
}
