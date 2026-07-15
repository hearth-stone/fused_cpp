#include "sdpa_c_api.h"

#include "sdpa_common.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

thread_local std::string g_last_error;

const std::vector<std::string>& version_cache() {
  static const std::vector<std::string> versions = sdpa_list_versions();
  return versions;
}

void set_last_error(const std::string& msg) { g_last_error = msg; }

void clear_last_error() { g_last_error.clear(); }

void check_shape(int64_t B, int64_t N, int64_t L, int64_t S, int64_t E, int64_t Ev) {
  if (B <= 0 || N <= 0 || L <= 0 || S <= 0 || E <= 0 || Ev <= 0) {
    throw std::invalid_argument("sdpa_c_api: B/N/L/S/E/Ev must all be positive");
  }
}

void check_ptr(const void* ptr, const char* name) {
  if (ptr == nullptr) {
    throw std::invalid_argument(std::string("sdpa_c_api: ") + name + " must be non-null");
  }
}

SdpaDtype parse_dtype(int dtype) {
  if (dtype == FUSED_CPP_SDPA_DTYPE_F32) {
    return SdpaDtype::kFloat32;
  }
  if (dtype == FUSED_CPP_SDPA_DTYPE_BF16) {
    return SdpaDtype::kBFloat16;
  }
  throw std::invalid_argument("sdpa_c_api: dtype must be F32(0) or BF16(1)");
}

bool is_contiguous_bntd(const fused_cpp_sdpa_strides& s, int64_t N, int64_t T, int64_t D) {
  return s.d == 1 && s.t == D && s.n == T * D && s.b == N * T * D;
}

template <typename Scalar>
const Scalar* materialize_bntd(const void* src_void, const fused_cpp_sdpa_strides& strides, int64_t B, int64_t N,
                               int64_t Tdim, int64_t D, std::vector<Scalar>& storage) {
  const Scalar* src = static_cast<const Scalar*>(src_void);
  if (is_contiguous_bntd(strides, N, Tdim, D)) {
    return src;
  }

  storage.resize(static_cast<size_t>(B * N * Tdim * D));
  Scalar* dst = storage.data();
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t t = 0; t < Tdim; ++t) {
        const int64_t src_base = b * strides.b + n * strides.n + t * strides.t;
        Scalar* dst_row = dst + (((b * N + n) * Tdim + t) * D);
        if (strides.d == 1) {
          std::memcpy(dst_row, src + src_base, static_cast<size_t>(D) * sizeof(Scalar));
        } else {
          for (int64_t d = 0; d < D; ++d) {
            dst_row[d] = src[src_base + d * strides.d];
          }
        }
      }
    }
  }
  return dst;
}

void scatter_bntd_float(const float* src, float* dst, const fused_cpp_sdpa_strides& strides, int64_t B, int64_t N,
                        int64_t T, int64_t D) {
  if (is_contiguous_bntd(strides, N, T, D)) {
    if (dst != src) {
      std::memcpy(dst, src, static_cast<size_t>(B * N * T * D) * sizeof(float));
    }
    return;
  }

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t n = 0; n < N; ++n) {
      for (int64_t t = 0; t < T; ++t) {
        const float* src_row = src + (((b * N + n) * T + t) * D);
        const int64_t dst_base = b * strides.b + n * strides.n + t * strides.t;
        if (strides.d == 1) {
          std::memcpy(dst + dst_base, src_row, static_cast<size_t>(D) * sizeof(float));
        } else {
          for (int64_t d = 0; d < D; ++d) {
            dst[dst_base + d * strides.d] = src_row[d];
          }
        }
      }
    }
  }
}

int forward_strided_impl(const char* version, int dtype, const void* q, const void* k, const void* v,
                         const float* attn_mask, float* out, int64_t B, int64_t N, int64_t L, int64_t S, int64_t E,
                         int64_t Ev, fused_cpp_sdpa_strides q_strides, fused_cpp_sdpa_strides k_strides,
                         fused_cpp_sdpa_strides v_strides, fused_cpp_sdpa_strides out_strides, int is_causal,
                         float scale) {
  clear_last_error();
  check_ptr(version, "version");
  check_ptr(q, "q");
  check_ptr(k, "k");
  check_ptr(v, "v");
  check_ptr(out, "out");
  check_shape(B, N, L, S, E, Ev);

  const SdpaDtype sdpa_dtype = parse_dtype(dtype);
  const std::string version_name(version);
  if (sdpa_find_kernel(version_name) == nullptr) {
    std::string msg = "sdpa_c_api: unknown SDPA version '" + version_name + "'; available: [";
    const auto& versions = version_cache();
    for (size_t i = 0; i < versions.size(); ++i) {
      if (i) msg += ", ";
      msg += versions[i];
    }
    msg += "]";
    throw std::invalid_argument(msg);
  }

  const bool out_is_contig = is_contiguous_bntd(out_strides, N, L, Ev);
  std::vector<float> out_storage;
  float* out_bnld = out;
  if (!out_is_contig) {
    out_storage.resize(static_cast<size_t>(B * N * L * Ev));
    out_bnld = out_storage.data();
  }

  SdpaParams p{};
  p.B = B;
  p.N = N;
  p.L = L;
  p.S = S;
  p.E = E;
  p.Ev = Ev;
  p.scale_f = scale > 0.0f ? scale : 1.0f / std::sqrt(static_cast<float>(E));
  p.neg_inf = -std::numeric_limits<float>::infinity();
  p.causal_offset = S - L;
  p.is_causal = is_causal != 0;
  p.dtype = sdpa_dtype;
  p.mask_ptr = attn_mask;
  p.out_ptr = out_bnld;

  if (sdpa_dtype == SdpaDtype::kFloat32) {
    std::vector<float> q_storage;
    std::vector<float> k_storage;
    std::vector<float> v_storage;
    p.q_ptr = materialize_bntd<float>(q, q_strides, B, N, L, E, q_storage);
    p.k_ptr = materialize_bntd<float>(k, k_strides, B, N, S, E, k_storage);
    p.v_ptr = materialize_bntd<float>(v, v_strides, B, N, S, Ev, v_storage);
    sdpa_dispatch(version_name, p);
  } else {
    std::vector<uint16_t> q_storage;
    std::vector<uint16_t> k_storage;
    std::vector<uint16_t> v_storage;
    p.q_ptr = materialize_bntd<uint16_t>(q, q_strides, B, N, L, E, q_storage);
    p.k_ptr = materialize_bntd<uint16_t>(k, k_strides, B, N, S, E, k_storage);
    p.v_ptr = materialize_bntd<uint16_t>(v, v_strides, B, N, S, Ev, v_storage);
    sdpa_dispatch(version_name, p);
  }

  if (!out_is_contig) {
    scatter_bntd_float(out_bnld, out, out_strides, B, N, L, Ev);
  }
  return 0;
}

}  // namespace

extern "C" {

const char* fused_cpp_sdpa_last_error(void) { return g_last_error.c_str(); }

int fused_cpp_sdpa_version_count(void) {
  try {
    clear_last_error();
    return static_cast<int>(version_cache().size());
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  } catch (...) {
    set_last_error("sdpa_c_api: unknown exception");
    return -1;
  }
}

const char* fused_cpp_sdpa_version_name(int index) {
  try {
    clear_last_error();
    const auto& versions = version_cache();
    if (index < 0 || static_cast<size_t>(index) >= versions.size()) {
      set_last_error("sdpa_c_api: version index out of range");
      return nullptr;
    }
    return versions[static_cast<size_t>(index)].c_str();
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return nullptr;
  } catch (...) {
    set_last_error("sdpa_c_api: unknown exception");
    return nullptr;
  }
}

int fused_cpp_sdpa_forward_strided(const char* version, int dtype, const void* q, const void* k, const void* v,
                                   const float* attn_mask, float* out, int64_t B, int64_t N, int64_t L, int64_t S,
                                   int64_t E, int64_t Ev, fused_cpp_sdpa_strides q_strides,
                                   fused_cpp_sdpa_strides k_strides, fused_cpp_sdpa_strides v_strides,
                                   fused_cpp_sdpa_strides out_strides, int is_causal, float scale) {
  try {
    return forward_strided_impl(version, dtype, q, k, v, attn_mask, out, B, N, L, S, E, Ev, q_strides, k_strides,
                                v_strides, out_strides, is_causal, scale);
  } catch (const std::exception& e) {
    set_last_error(e.what());
    return -1;
  } catch (...) {
    set_last_error("sdpa_c_api: unknown exception");
    return -1;
  }
}

int fused_cpp_sdpa_forward_ggml_f32(const char* version, const float* q, const float* k, const float* v, float* out,
                                    int64_t B, int64_t N, int64_t L, int64_t S, int64_t E, int64_t Ev, int is_causal,
                                    float scale) {
  const fused_cpp_sdpa_strides q_strides{
      E * N * L,
      E,
      E * N,
      1,
  };
  const fused_cpp_sdpa_strides k_strides{
      E * N * S,
      E,
      E * N,
      1,
  };
  const fused_cpp_sdpa_strides v_strides{
      Ev * N * S,
      Ev,
      Ev * N,
      1,
  };
  const fused_cpp_sdpa_strides out_strides{
      Ev * N * L,
      Ev,
      Ev * N,
      1,
  };

  return fused_cpp_sdpa_forward_strided(version, FUSED_CPP_SDPA_DTYPE_F32, q, k, v, nullptr, out, B, N, L, S, E, Ev,
                                        q_strides, k_strides, v_strides, out_strides, is_causal, scale);
}

}  // extern "C"
