// Expert execution cost model: T_expert(routes, threads) -> nanoseconds.
//
// Mirrors ExpertCostModel in offline_simulator.py:
//   * synthetic(): closed-form formula (the simulator default).
//   * table:       measured (routes, threads) -> ns map with exact match and
//                  nearest-bucket fallback.
//
// Header-only and torch-free.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace moe_planner {

class CostModel {
public:
    enum class Kind { Synthetic, Table };

    // --- synthetic (matches ExpertCostModel.synthetic exactly) ---
    static CostModel synthetic() {
        CostModel m;
        m.kind_ = Kind::Synthetic;
        return m;
    }

    // --- table (matches ExpertCostModel.from_json lookup semantics) ---
    // `route_buckets` and `thread_buckets` must be the sorted-ascending unique
    // bucket values; `values` is row-major [num_routes][num_threads] in ns,
    // i.e. values[r * thread_buckets.size() + t].
    static CostModel from_table(std::vector<int> route_buckets,
                                std::vector<int> thread_buckets,
                                std::vector<int64_t> values) {
        CostModel m;
        m.kind_ = Kind::Table;
        m.route_buckets_ = std::move(route_buckets);
        m.thread_buckets_ = std::move(thread_buckets);
        m.values_ = std::move(values);
        return m;
    }

    Kind kind() const { return kind_; }

    inline int64_t estimate(int routes, int threads) const {
        if (kind_ == Kind::Synthetic) return synthetic_ns(routes, threads);
        return table_ns(routes, threads);
    }

    // Synthetic building blocks (single source of truth for the constants so
    // the fast precomputed path in prepare_workload stays bit-identical).
    static inline double synth_serial(int routes) {
        return 18000.0 + static_cast<double>(routes) * 1850.0;
    }
    static inline double synth_speedup(int useful) {
        return 1.0 + 0.82 * std::pow(static_cast<double>(useful - 1), 0.72);
    }
    static inline double synth_overhead(int threads) {
        return 900.0 * threads + 35.0 * static_cast<double>(threads) * threads;
    }

    static inline int64_t synthetic_ns(int routes, int threads) {
        const int useful = std::max(1, std::min(threads, routes));
        // Python int() truncates toward zero; the result is positive.
        return static_cast<int64_t>(synth_serial(routes) / synth_speedup(useful) +
                                    synth_overhead(threads));
    }

private:
    // nearest bucket: min over buckets of (abs(bucket - value), bucket).
    static int nearest(int value, const std::vector<int>& buckets) {
        int best = buckets[0];
        long best_dist = std::labs(static_cast<long>(buckets[0]) - value);
        for (size_t i = 1; i < buckets.size(); ++i) {
            long d = std::labs(static_cast<long>(buckets[i]) - value);
            if (d < best_dist || (d == best_dist && buckets[i] < best)) {
                best_dist = d;
                best = buckets[i];
            }
        }
        return best;
    }

    int64_t table_ns(int routes, int threads) const {
        const int rb = nearest(routes, route_buckets_);
        const int tb = nearest(threads, thread_buckets_);
        const auto rit =
            std::lower_bound(route_buckets_.begin(), route_buckets_.end(), rb);
        const auto tit = std::lower_bound(thread_buckets_.begin(),
                                          thread_buckets_.end(), tb);
        const size_t ri = static_cast<size_t>(rit - route_buckets_.begin());
        const size_t ti = static_cast<size_t>(tit - thread_buckets_.begin());
        return values_[ri * thread_buckets_.size() + ti];
    }

    Kind kind_ = Kind::Synthetic;
    std::vector<int> route_buckets_;
    std::vector<int> thread_buckets_;
    std::vector<int64_t> values_;
};

}  // namespace moe_planner
