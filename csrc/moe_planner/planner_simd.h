// SIMD helpers for the MoE planners.
//
// The only kernel that materially benefits from vectorization is the
// greedy marginal-gain argmax: each allocation step scans a contiguous
// per-expert gain[] array for the maximum. We provide:
//   - SVE   path  (Linux / AWS aarch64,   __ARM_FEATURE_SVE)
//   - NEON  path  (any aarch64 incl. Apple M-series, __aarch64__)
//   - scalar path (portable fallback)
//
// All paths return the LOWEST index achieving the maximum, matching the
// Python reference which scans in order with a strict `gain > best_gain`
// test (so the earliest expert wins on ties).
#pragma once

#include <cstdint>
#include <cstddef>

#if defined(__ARM_FEATURE_SVE)
#include <arm_sve.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace moe_planner {
namespace simd {

// Returns the index of the maximum element in g[0..n). On ties returns the
// lowest such index. Writes the maximum value to *out_max. n may be 0 (then
// returns -1 and leaves *out_max untouched).
inline int argmax_i64(const int64_t* g, int n, int64_t* out_max) {
    if (n <= 0) return -1;

#if defined(__ARM_FEATURE_SVE)
    // SVE: vectorized max-reduce to find the value, then a vectorized scan
    // (compare-against-max + first-true) to find the lowest index.
    {
        int64_t best = g[0];
        const uint64_t vl = svcntd();
        for (int i = 0; i < n; i += static_cast<int>(vl)) {
            svbool_t pg = svwhilelt_b64(static_cast<uint64_t>(i),
                                        static_cast<uint64_t>(n));
            svint64_t v = svld1_s64(pg, g + i);
            int64_t blk = svmaxv_s64(pg, v);
            if (blk > best) best = blk;
        }
        for (int i = 0; i < n; i += static_cast<int>(vl)) {
            svbool_t pg = svwhilelt_b64(static_cast<uint64_t>(i),
                                        static_cast<uint64_t>(n));
            svint64_t v = svld1_s64(pg, g + i);
            svbool_t eq = svcmpeq_n_s64(pg, v, best);
            if (svptest_any(pg, eq)) {
                svbool_t first = svbrkb_b_z(pg, eq);  // mask up to (excl.) first
                int idx = i + static_cast<int>(svcntp_b64(pg, first));
                *out_max = best;
                return idx;
            }
        }
        *out_max = best;
        return 0;
    }
#elif defined(__aarch64__)
    // NEON: max-reduce 2-wide for the value, then a scalar scan for the
    // lowest index (the index scan stops early at the first maximum).
    {
        int i = 0;
        int64x2_t vmax = vdupq_n_s64(g[0]);
        for (; i + 2 <= n; i += 2) {
            int64x2_t v = vld1q_s64(g + i);
            // no vmaxq_s64 intrinsic; compare+select.
            uint64x2_t gt = vcgtq_s64(v, vmax);
            vmax = vbslq_s64(gt, v, vmax);
        }
        int64_t best = vgetq_lane_s64(vmax, 0);
        int64_t hi = vgetq_lane_s64(vmax, 1);
        if (hi > best) best = hi;
        for (; i < n; ++i) {
            if (g[i] > best) best = g[i];
        }
        for (int j = 0; j < n; ++j) {
            if (g[j] == best) {
                *out_max = best;
                return j;
            }
        }
        *out_max = best;
        return 0;
    }
#else
    {
        int64_t best = g[0];
        int best_idx = 0;
        for (int i = 1; i < n; ++i) {
            if (g[i] > best) {
                best = g[i];
                best_idx = i;
            }
        }
        *out_max = best;
        return best_idx;
    }
#endif
}

}  // namespace simd
}  // namespace moe_planner
