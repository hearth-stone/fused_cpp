#include <arm_sve.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "deepseek_v4_indexer_sve.h"

namespace indexer = fused_cpp::deepseek_v4::indexer_sve;

namespace {

uint16_t to_bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float from_bf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

bool validate_batched_topk() {
  constexpr int64_t kRows = 5;
  constexpr int64_t kColumns = 19;
  constexpr int64_t kScoreStride = 22;
  constexpr int64_t kOutputStride0 = 43;
  constexpr int64_t kOutputStride1 = 2;
  const std::vector<int64_t> row_starts = {0, 2, 5, 1, 7};
  const std::vector<int64_t> row_ends = {0, 7, 19, 10, 19};
  std::vector<float> scores(kRows * kScoreStride, -1000.0f);
  for (int64_t m = 0; m < kRows; ++m) {
    for (int64_t n = 0; n < kColumns; ++n) {
      scores[m * kScoreStride + n] = static_cast<float>((3 * n + 5 * m) % 7);
    }
  }
  scores[2 * kScoreStride + 8] = std::numeric_limits<float>::quiet_NaN();

  for (int64_t topk : {int64_t{0}, int64_t{1}, int64_t{4}, int64_t{20}}) {
    std::vector<int32_t> output(kRows * kOutputStride0, -77);
    indexer::batched_topk_indices(scores.data(), kScoreStride, kColumns, row_starts.data(), row_ends.data(),
                                  output.data(), kOutputStride0, kOutputStride1, kRows, topk);
    for (int64_t m = 0; m < kRows; ++m) {
      const int64_t row_start = row_starts[m];
      const int64_t valid_len = row_ends[m] - row_start;
      const int64_t k_take = std::min(topk, valid_len);
      std::vector<int32_t> expected(static_cast<size_t>(valid_len));
      for (int64_t i = 0; i < valid_len; ++i) {
        expected[i] = static_cast<int32_t>(i);
      }
      const float* row = scores.data() + m * kScoreStride + row_start;
      std::sort(expected.begin(), expected.end(), [row](int32_t lhs, int32_t rhs) {
        const bool lhs_nan = std::isnan(row[lhs]);
        const bool rhs_nan = std::isnan(row[rhs]);
        if (lhs_nan != rhs_nan) {
          return lhs_nan;
        }
        if (lhs_nan || row[lhs] == row[rhs]) {
          return lhs < rhs;
        }
        return row[lhs] > row[rhs];
      });
      for (int64_t i = 0; i < k_take; ++i) {
        if (output[m * kOutputStride0 + i * kOutputStride1] != expected[i]) {
          return false;
        }
      }
      for (int64_t i = k_take; i < kColumns; ++i) {
        if (output[m * kOutputStride0 + i * kOutputStride1] != -77) {
          return false;
        }
      }
    }
  }
  return true;
}

std::pair<float, float> validate_scores() {
  float global_max_abs = 0.0f;
  float global_max_rel = 0.0f;
  for (int H : {8, 16, 64}) {
    for (int K : {4, 16, 128}) {
      for (int N : {1, 7, 16, 19, 33}) {
        constexpr int M = 3;
        const int Np = indexer::round_n(N);
        std::vector<uint16_t> q(static_cast<int64_t>(M) * H * K);
        std::vector<uint16_t> key(static_cast<int64_t>(N) * K);
        std::vector<float> weights(static_cast<int64_t>(M) * H);
        std::vector<int64_t> offsets(N);
        std::vector<uint16_t> packed_k(static_cast<int64_t>(K) * Np);
        std::vector<float> scores(static_cast<int64_t>(M) * Np, std::numeric_limits<float>::quiet_NaN());

        for (int64_t i = 0; i < static_cast<int64_t>(q.size()); ++i) {
          q[i] = to_bf16(0.11f * std::sin(0.013f * static_cast<float>(i + 3)));
        }
        for (int64_t i = 0; i < static_cast<int64_t>(key.size()); ++i) {
          key[i] = to_bf16(0.09f * std::cos(0.017f * static_cast<float>(i + 11)));
        }
        for (int64_t i = 0; i < static_cast<int64_t>(weights.size()); ++i) {
          weights[i] = 0.4f * std::sin(0.071f * static_cast<float>(i + 5));
        }
        for (int n = 0; n < N; ++n) {
          offsets[n] = static_cast<int64_t>(n) * K;
        }

        indexer::pack_paged_k(key.data(), offsets.data(), packed_k.data(), K, N, Np);
        indexer::weighted_relu_scores(q.data(), weights.data(), packed_k.data(), scores.data(), M, H, K, Np);

        for (int m = 0; m < M; ++m) {
          for (int n = 0; n < N; ++n) {
            float reference = 0.0f;
            for (int h = 0; h < H; ++h) {
              float dot = 0.0f;
              for (int k = 0; k < K; ++k) {
                dot += from_bf16(q[(static_cast<int64_t>(m) * H + h) * K + k]) *
                       from_bf16(key[static_cast<int64_t>(n) * K + k]);
              }
              reference += weights[static_cast<int64_t>(m) * H + h] * std::max(dot, 0.0f);
            }
            const float actual = scores[static_cast<int64_t>(m) * Np + n];
            const float abs_error = std::abs(actual - reference);
            const float rel_error = abs_error / std::max(std::abs(reference), 1.0e-6f);
            global_max_abs = std::max(global_max_abs, abs_error);
            global_max_rel = std::max(global_max_rel, rel_error);
          }
          for (int n = N; n < Np; ++n) {
            global_max_abs = std::max(global_max_abs, std::abs(scores[static_cast<int64_t>(m) * Np + n]));
          }
        }
      }
    }
  }
  return {global_max_abs, global_max_rel};
}

void benchmark_production_score() {
  constexpr int M = 2048;
  constexpr int H = 64;
  constexpr int K = 128;
  constexpr int N = 1536;
  constexpr int kWarmups = 3;
  constexpr int kRuns = 15;
  const int Np = indexer::round_n(N);
  std::vector<uint16_t> q(static_cast<int64_t>(M) * H * K, to_bf16(0.01f));
  std::vector<uint16_t> key(static_cast<int64_t>(N) * K, to_bf16(0.02f));
  std::vector<float> weights(static_cast<int64_t>(M) * H, 0.03f);
  std::vector<int64_t> offsets(N);
  std::vector<uint16_t> packed_k(static_cast<int64_t>(K) * Np);
  std::vector<float> scores(static_cast<int64_t>(M) * Np);
  for (int n = 0; n < N; ++n) {
    offsets[n] = static_cast<int64_t>(n) * K;
  }
  indexer::pack_paged_k(key.data(), offsets.data(), packed_k.data(), K, N, Np);
  for (int warmup = 0; warmup < kWarmups; ++warmup) {
    indexer::weighted_relu_scores(q.data(), weights.data(), packed_k.data(), scores.data(), M, H, K, Np);
  }

  std::vector<double> samples_ms;
  for (int run = 0; run < kRuns; ++run) {
    const auto start = std::chrono::steady_clock::now();
    indexer::weighted_relu_scores(q.data(), weights.data(), packed_k.data(), scores.data(), M, H, K, Np);
    const auto end = std::chrono::steady_clock::now();
    samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(samples_ms.begin(), samples_ms.end());
  const double median_ms = samples_ms[samples_ms.size() / 2];
  const double flops = 2.0 * M * H * K * N;
  std::cout << "m=" << M << " h=" << H << " d=" << K << " n=" << N << " median_ms=" << median_ms
            << " min_ms=" << samples_ms.front() << " max_ms=" << samples_ms.back()
            << " gflops=" << flops / (median_ms * 1.0e6) << " checksum=" << scores.front() + scores.back() << '\n';
}

}  // namespace

int main() {
  if (!validate_batched_topk()) {
    std::cerr << "batched_topk_validation=failed\n";
    return 1;
  }
  std::cout << "batched_topk_validation=passed cases=20\n";
  const auto [max_abs, max_rel] = validate_scores();
  std::cout << "svl_bits=" << svcntb() * 8 << " n_tile=" << indexer::n_tile()
            << " validation_cases=45 max_abs=" << max_abs << " max_rel=" << max_rel << '\n';
  if (max_abs > 2.0e-5f) {
    return 1;
  }
  benchmark_production_score();
  return 0;
}
