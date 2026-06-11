#include "fp32_packqkv_sdpa.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

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
  }

  return 0;
}
