#include "fp32_packqkv_sdpa.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>

namespace {

constexpr uint16_t kF16Zero = 0x0000;
constexpr uint16_t kF16NegOne = 0xbc00;
constexpr uint16_t kF16NegInf = 0xfc00;

bool starts_with(const std::string& s, const char* prefix) {
  return s.rfind(prefix, 0) == 0;
}

int64_t parse_i64(const std::string& s) {
  return std::strtoll(s.c_str(), nullptr, 10);
}

int64_t get_i64_arg(int& i, int argc, char** argv, const std::string& arg) {
  const auto pos = arg.find('=');
  if (pos != std::string::npos) {
    return parse_i64(arg.substr(pos + 1));
  }
  if (i + 1 >= argc) {
    throw std::runtime_error("missing value for " + arg);
  }
  return parse_i64(argv[++i]);
}

void usage(const char* argv0) {
  std::cout
      << "usage: " << argv0 << " [options]\n"
      << "  --B=1 --N=8 --L=512 --S=512 --E=64 --Ev=64\n"
      << "  --causal | --noncausal\n"
      << "  --iters=20 --warmup=5 --s-tile=0 --check\n";
}

}  // namespace

int main(int argc, char** argv) {
  fp32_packqkv_sdpa::Config cfg;
  cfg.causal_offset = cfg.S - cfg.L;
  int warmup = 5;
  int iters = 20;
  bool check = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      return 0;
    } else if (starts_with(arg, "--B")) {
      cfg.B = get_i64_arg(i, argc, argv, arg);
    } else if (starts_with(arg, "--N")) {
      cfg.N = get_i64_arg(i, argc, argv, arg);
    } else if (starts_with(arg, "--L")) {
      cfg.L = get_i64_arg(i, argc, argv, arg);
    } else if (starts_with(arg, "--S")) {
      cfg.S = get_i64_arg(i, argc, argv, arg);
    } else if (starts_with(arg, "--E=") || arg == "--E") {
      cfg.E = get_i64_arg(i, argc, argv, arg);
    } else if (starts_with(arg, "--Ev")) {
      cfg.Ev = get_i64_arg(i, argc, argv, arg);
    } else if (starts_with(arg, "--iters")) {
      iters = static_cast<int>(get_i64_arg(i, argc, argv, arg));
    } else if (starts_with(arg, "--warmup")) {
      warmup = static_cast<int>(get_i64_arg(i, argc, argv, arg));
    } else if (starts_with(arg, "--s-tile")) {
      cfg.s_tile = get_i64_arg(i, argc, argv, arg);
    } else if (arg == "--causal") {
      cfg.causal = true;
    } else if (arg == "--noncausal") {
      cfg.causal = false;
    } else if (arg == "--check") {
      check = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  cfg.causal_offset = cfg.S - cfg.L;
  cfg.scale = 1.0f / std::sqrt(static_cast<float>(cfg.E));

  const int64_t q_size = cfg.B * cfg.N * cfg.L * cfg.E;
  const int64_t k_size = cfg.B * cfg.N * cfg.S * cfg.E;
  const int64_t v_size = cfg.B * cfg.N * cfg.S * cfg.Ev;
  const int64_t o_size = cfg.B * cfg.N * cfg.L * cfg.Ev;

  fp32_packqkv_sdpa::AlignedVector<float> q(static_cast<size_t>(q_size));
  fp32_packqkv_sdpa::AlignedVector<float> k(static_cast<size_t>(k_size));
  fp32_packqkv_sdpa::AlignedVector<float> v(static_cast<size_t>(v_size));
  fp32_packqkv_sdpa::AlignedVector<float> out(
      static_cast<size_t>(o_size), 0.0f);

  std::mt19937 rng(1234);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (auto& x : q) x = dist(rng);
  for (auto& x : k) x = dist(rng);
  for (auto& x : v) x = dist(rng);

  for (int i = 0; i < warmup; ++i) {
    fp32_packqkv_sdpa::sdpa_fp32_packqkv_pbf16pv(
        q.data(), k.data(), v.data(), out.data(), cfg);
  }

  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    fp32_packqkv_sdpa::sdpa_fp32_packqkv_pbf16pv(
        q.data(), k.data(), v.data(), out.data(), cfg);
  }
  const auto t1 = std::chrono::steady_clock::now();
  const double mean_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() /
      static_cast<double>(iters);

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "shape=B" << cfg.B << "-N" << cfg.N
            << "-L" << cfg.L << "-S" << cfg.S
            << "-E" << cfg.E << "-Ev" << cfg.Ev
            << " causal=" << (cfg.causal ? "true" : "false") << "\n";
  std::cout << "mean_ms=" << mean_ms
            << " gflops=" << fp32_packqkv_sdpa::counted_gflops(cfg, mean_ms)
            << " checksum=" << fp32_packqkv_sdpa::checksum(out.data(), o_size)
            << "\n";

  if (check) {
    fp32_packqkv_sdpa::AlignedVector<float> ref(
        static_cast<size_t>(o_size), 0.0f);
    fp32_packqkv_sdpa::reference_sdpa_fp32(
        q.data(), k.data(), v.data(), ref.data(), cfg);
    std::cout << "max_abs_diff="
              << fp32_packqkv_sdpa::max_abs_diff(
                     out.data(), ref.data(), o_size)
              << "\n";

    if (!cfg.causal) {
      fp32_packqkv_sdpa::AlignedVector<float> q_ggml(
          static_cast<size_t>(q_size));
      fp32_packqkv_sdpa::AlignedVector<float> k_ggml(
          static_cast<size_t>(k_size));
      fp32_packqkv_sdpa::AlignedVector<float> v_ggml(
          static_cast<size_t>(v_size));
      fp32_packqkv_sdpa::AlignedVector<float> out_ggml(
          static_cast<size_t>(o_size), 0.0f);
      fp32_packqkv_sdpa::AlignedVector<float> out_from_ggml(
          static_cast<size_t>(o_size), 0.0f);

      for (int64_t b = 0; b < cfg.B; ++b) {
        for (int64_t n = 0; n < cfg.N; ++n) {
          for (int64_t l = 0; l < cfg.L; ++l) {
            for (int64_t e = 0; e < cfg.E; ++e) {
              q_ggml[((b * cfg.L + l) * cfg.N + n) * cfg.E + e] =
                  q[((b * cfg.N + n) * cfg.L + l) * cfg.E + e];
            }
          }
          for (int64_t s = 0; s < cfg.S; ++s) {
            for (int64_t e = 0; e < cfg.E; ++e) {
              k_ggml[((b * cfg.S + s) * cfg.N + n) * cfg.E + e] =
                  k[((b * cfg.N + n) * cfg.S + s) * cfg.E + e];
            }
            for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
              v_ggml[((b * cfg.S + s) * cfg.N + n) * cfg.Ev + ev] =
                  v[((b * cfg.N + n) * cfg.S + s) * cfg.Ev + ev];
            }
          }
        }
      }

      const int64_t elem = static_cast<int64_t>(sizeof(float));
      const int rc =
          fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp(
              q_ggml.data(),
              k_ggml.data(),
              v_ggml.data(),
              out_ggml.data(),
              cfg.B,
              cfg.N,
              cfg.L,
              cfg.S,
              cfg.E,
              cfg.Ev,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.L * elem,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * cfg.N * cfg.L * elem,
              cfg.scale);
      if (rc != 0) {
        throw std::runtime_error("llamacpp strided check failed rc=" +
                                 std::to_string(rc));
      }

      for (int64_t b = 0; b < cfg.B; ++b) {
        for (int64_t n = 0; n < cfg.N; ++n) {
          for (int64_t l = 0; l < cfg.L; ++l) {
            for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
              out_from_ggml[((b * cfg.N + n) * cfg.L + l) * cfg.Ev + ev] =
                  out_ggml[((b * cfg.L + l) * cfg.N + n) * cfg.Ev + ev];
            }
          }
        }
      }
      std::cout << "llamacpp_max_abs_diff="
                << fp32_packqkv_sdpa::max_abs_diff(
                       out_from_ggml.data(), ref.data(), o_size)
                << "\n";

      const int64_t mask_h = 1;
      const int64_t mask_b = 1;
      fp32_packqkv_sdpa::AlignedVector<uint16_t> mask_f16(
          static_cast<size_t>(cfg.S * cfg.L * mask_h * mask_b));
      fp32_packqkv_sdpa::AlignedVector<float> mask_f32(
          static_cast<size_t>(cfg.S * cfg.L * mask_h * mask_b));
      fp32_packqkv_sdpa::AlignedVector<float> mask32(
          static_cast<size_t>(cfg.B * cfg.N * cfg.L * cfg.S));
      for (int64_t l = 0; l < cfg.L; ++l) {
        for (int64_t s = 0; s < cfg.S; ++s) {
          const bool penalize = ((s + l) % 3) == 0;
          const float value = penalize ? -1.0f : 0.0f;
          mask_f16[l * cfg.S + s] = penalize ? kF16NegOne : kF16Zero;
          mask_f32[l * cfg.S + s] = value;
          for (int64_t b = 0; b < cfg.B; ++b) {
            for (int64_t n = 0; n < cfg.N; ++n) {
              mask32[((b * cfg.N + n) * cfg.L + l) * cfg.S + s] = value;
            }
          }
        }
      }

      fp32_packqkv_sdpa::AlignedVector<float> ref_masked(
          static_cast<size_t>(o_size), 0.0f);
      fp32_packqkv_sdpa::AlignedVector<float> out_mask_ggml(
          static_cast<size_t>(o_size), 0.0f);
      fp32_packqkv_sdpa::AlignedVector<float> out_mask_from_ggml(
          static_cast<size_t>(o_size), 0.0f);
      fp32_packqkv_sdpa::reference_sdpa_fp32_mask(
          q.data(), k.data(), v.data(), mask32.data(), ref_masked.data(), cfg);

      const int rc_mask =
          fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f16(
              q_ggml.data(),
              k_ggml.data(),
              v_ggml.data(),
              mask_f16.data(),
              out_mask_ggml.data(),
              cfg.B,
              cfg.N,
              cfg.L,
              cfg.S,
              cfg.E,
              cfg.Ev,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.L * elem,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * cfg.N * cfg.L * elem,
              cfg.S,
              cfg.L,
              mask_h,
              mask_b,
              static_cast<int64_t>(sizeof(uint16_t)),
              cfg.S * static_cast<int64_t>(sizeof(uint16_t)),
              cfg.S * cfg.L * static_cast<int64_t>(sizeof(uint16_t)),
              cfg.S * cfg.L * mask_h * static_cast<int64_t>(sizeof(uint16_t)),
              cfg.scale);
      if (rc_mask != 0) {
        throw std::runtime_error("llamacpp F16 mask check failed rc=" +
                                 std::to_string(rc_mask));
      }
      for (int64_t b = 0; b < cfg.B; ++b) {
        for (int64_t n = 0; n < cfg.N; ++n) {
          for (int64_t l = 0; l < cfg.L; ++l) {
            for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
              out_mask_from_ggml[((b * cfg.N + n) * cfg.L + l) * cfg.Ev + ev] =
                  out_mask_ggml[((b * cfg.L + l) * cfg.N + n) * cfg.Ev + ev];
            }
          }
        }
      }
      std::cout << "llamacpp_mask_f16_max_abs_diff="
                << fp32_packqkv_sdpa::max_abs_diff(
                       out_mask_from_ggml.data(), ref_masked.data(), o_size)
                << "\n";

      std::fill(out_mask_ggml.begin(), out_mask_ggml.end(), 0.0f);
      std::fill(out_mask_from_ggml.begin(), out_mask_from_ggml.end(), 0.0f);
      const int rc_mask_f32 =
          fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f32(
              q_ggml.data(),
              k_ggml.data(),
              v_ggml.data(),
              mask_f32.data(),
              out_mask_ggml.data(),
              cfg.B,
              cfg.N,
              cfg.L,
              cfg.S,
              cfg.E,
              cfg.Ev,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.L * elem,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * cfg.N * cfg.L * elem,
              cfg.S,
              cfg.L,
              mask_h,
              mask_b,
              static_cast<int64_t>(sizeof(float)),
              cfg.S * static_cast<int64_t>(sizeof(float)),
              cfg.S * cfg.L * static_cast<int64_t>(sizeof(float)),
              cfg.S * cfg.L * mask_h * static_cast<int64_t>(sizeof(float)),
              cfg.scale);
      if (rc_mask_f32 != 0) {
        throw std::runtime_error("llamacpp F32 mask check failed rc=" +
                                 std::to_string(rc_mask_f32));
      }
      for (int64_t b = 0; b < cfg.B; ++b) {
        for (int64_t n = 0; n < cfg.N; ++n) {
          for (int64_t l = 0; l < cfg.L; ++l) {
            for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
              out_mask_from_ggml[((b * cfg.N + n) * cfg.L + l) * cfg.Ev + ev] =
                  out_mask_ggml[((b * cfg.L + l) * cfg.N + n) * cfg.Ev + ev];
            }
          }
        }
      }
      std::cout << "llamacpp_mask_f32_max_abs_diff="
                << fp32_packqkv_sdpa::max_abs_diff(
                       out_mask_from_ggml.data(), ref_masked.data(), o_size)
                << "\n";

      std::fill(mask32.begin(), mask32.end(), 0.0f);
      std::fill(out_mask_ggml.begin(), out_mask_ggml.end(), 0.0f);
      std::fill(out_mask_from_ggml.begin(), out_mask_from_ggml.end(), 0.0f);
      std::fill(ref_masked.begin(), ref_masked.end(), 0.0f);
      for (int64_t l = 0; l < cfg.L; ++l) {
        for (int64_t s = 0; s < cfg.S; ++s) {
          const bool disabled = s > l;
          const float value = disabled
              ? -std::numeric_limits<float>::infinity()
              : 0.0f;
          mask_f16[l * cfg.S + s] = disabled ? kF16NegInf : kF16Zero;
          mask_f32[l * cfg.S + s] = value;
          for (int64_t b = 0; b < cfg.B; ++b) {
            for (int64_t n = 0; n < cfg.N; ++n) {
              mask32[((b * cfg.N + n) * cfg.L + l) * cfg.S + s] = value;
            }
          }
        }
      }
      fp32_packqkv_sdpa::reference_sdpa_fp32_mask(
          q.data(), k.data(), v.data(), mask32.data(), ref_masked.data(), cfg);
      const int rc_zeroinf =
          fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f16(
              q_ggml.data(),
              k_ggml.data(),
              v_ggml.data(),
              mask_f16.data(),
              out_mask_ggml.data(),
              cfg.B,
              cfg.N,
              cfg.L,
              cfg.S,
              cfg.E,
              cfg.Ev,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.L * elem,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * cfg.N * cfg.L * elem,
              cfg.S,
              cfg.L,
              mask_h,
              mask_b,
              static_cast<int64_t>(sizeof(uint16_t)),
              cfg.S * static_cast<int64_t>(sizeof(uint16_t)),
              cfg.S * cfg.L * static_cast<int64_t>(sizeof(uint16_t)),
              cfg.S * cfg.L * mask_h * static_cast<int64_t>(sizeof(uint16_t)),
              cfg.scale);
      if (rc_zeroinf != 0) {
        throw std::runtime_error("llamacpp F16 0/-inf mask check failed rc=" +
                                 std::to_string(rc_zeroinf));
      }
      for (int64_t b = 0; b < cfg.B; ++b) {
        for (int64_t n = 0; n < cfg.N; ++n) {
          for (int64_t l = 0; l < cfg.L; ++l) {
            for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
              out_mask_from_ggml[((b * cfg.N + n) * cfg.L + l) * cfg.Ev + ev] =
                  out_mask_ggml[((b * cfg.L + l) * cfg.N + n) * cfg.Ev + ev];
            }
          }
        }
      }
      std::cout << "llamacpp_mask_f16_zeroinf_max_abs_diff="
                << fp32_packqkv_sdpa::max_abs_diff(
                       out_mask_from_ggml.data(), ref_masked.data(), o_size)
                << "\n";

      std::fill(out_mask_ggml.begin(), out_mask_ggml.end(), 0.0f);
      std::fill(out_mask_from_ggml.begin(), out_mask_from_ggml.end(), 0.0f);
      const int rc_zeroinf_f32 =
          fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp_mask_f32(
              q_ggml.data(),
              k_ggml.data(),
              v_ggml.data(),
              mask_f32.data(),
              out_mask_ggml.data(),
              cfg.B,
              cfg.N,
              cfg.L,
              cfg.S,
              cfg.E,
              cfg.Ev,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.L * elem,
              elem,
              cfg.E * cfg.N * elem,
              cfg.E * elem,
              cfg.E * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * cfg.S * elem,
              elem,
              cfg.Ev * elem,
              cfg.Ev * cfg.N * elem,
              cfg.Ev * cfg.N * cfg.L * elem,
              cfg.S,
              cfg.L,
              mask_h,
              mask_b,
              static_cast<int64_t>(sizeof(float)),
              cfg.S * static_cast<int64_t>(sizeof(float)),
              cfg.S * cfg.L * static_cast<int64_t>(sizeof(float)),
              cfg.S * cfg.L * mask_h * static_cast<int64_t>(sizeof(float)),
              cfg.scale);
      if (rc_zeroinf_f32 != 0) {
        throw std::runtime_error("llamacpp F32 0/-inf mask check failed rc=" +
                                 std::to_string(rc_zeroinf_f32));
      }
      for (int64_t b = 0; b < cfg.B; ++b) {
        for (int64_t n = 0; n < cfg.N; ++n) {
          for (int64_t l = 0; l < cfg.L; ++l) {
            for (int64_t ev = 0; ev < cfg.Ev; ++ev) {
              out_mask_from_ggml[((b * cfg.N + n) * cfg.L + l) * cfg.Ev + ev] =
                  out_mask_ggml[((b * cfg.L + l) * cfg.N + n) * cfg.Ev + ev];
            }
          }
        }
      }
      std::cout << "llamacpp_mask_f32_zeroinf_max_abs_diff="
                << fp32_packqkv_sdpa::max_abs_diff(
                       out_mask_from_ggml.data(), ref_masked.data(), o_size)
                << "\n";
    }
  }

  return 0;
}
