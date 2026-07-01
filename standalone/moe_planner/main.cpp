// Standalone correctness + latency harness for the MoE planners.
//
// Torch-free: compiles only the planner core, so it builds in well under a
// second and lets us measure planner latency "near the limit" with the exact
// target SIMD flags. See the Makefile in this directory.
//
//   ./moe_planner            # run all distributions x cores, all planners
//   ./moe_planner --csv      # machine-readable timing rows
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "moe_planner/cost_model.h"
#include "moe_planner/planner_dispatch.h"

using namespace moe_planner;

// ----------------------------- route synthesis -----------------------------

// Largest-remainder integer allocation of `total` across `weights`.
static std::vector<int32_t> integer_allocation(const std::vector<double>& weights,
                                               long total) {
    const int n = static_cast<int>(weights.size());
    std::vector<int32_t> out(n, 0);
    double s = 0.0;
    for (double w : weights) s += w;
    if (s <= 0.0) return out;
    std::vector<double> frac(n);
    long assigned = 0;
    for (int i = 0; i < n; ++i) {
        double raw = weights[i] / s * static_cast<double>(total);
        long f = static_cast<long>(std::floor(raw));
        out[i] = static_cast<int32_t>(f);
        frac[i] = raw - static_cast<double>(f);
        assigned += f;
    }
    long leftover = total - assigned;
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return frac[a] > frac[b]; });
    for (long i = 0; i < leftover; ++i) out[order[i % n]]++;
    return out;
}

static std::vector<int32_t> gen_routes(const std::string& dist, int E,
                                       long total, int active_k, int hot_k,
                                       double hot_frac) {
    if (dist == "uniform") {
        return integer_allocation(std::vector<double>(E, 1.0), total);
    }
    if (dist == "zipf") {
        std::vector<double> w(E);
        for (int i = 0; i < E; ++i) w[i] = 1.0 / std::pow(i + 1, 1.1);
        return integer_allocation(w, total);
    }
    if (dist == "active_subset") {
        std::vector<double> w(E, 0.0);
        for (int i = 0; i < active_k && i < E; ++i) w[i] = 1.0;
        return integer_allocation(w, total);
    }
    if (dist == "hotspot") {
        std::vector<double> w(E, 0.0);
        int cold = E - hot_k;
        double cold_share = 1.0 - hot_frac;
        for (int i = 0; i < E; ++i) {
            if (i < hot_k)
                w[i] = hot_frac / hot_k;
            else if (cold > 0)
                w[i] = cold_share / cold;
        }
        return integer_allocation(w, total);
    }
    return integer_allocation(std::vector<double>(E, 1.0), total);
}

// ----------------------------- invariants -----------------------------

// Returns "" on success or an error description. Verifies: each active expert
// appears exactly once; per-wave thread budget respected; recomputed execution
// cost matches the reported one.
static std::string check_plan(const PlanResult& r, const Workload& w,
                              const CostModel& cm) {
    std::unordered_map<int32_t, int32_t> routes_of;
    for (int a = 0; a < w.num_active; ++a)
        routes_of[w.expert_ids[a]] = w.routes[a];

    if (r.num_teams() != w.num_active)
        return "team count != active count";

    std::unordered_map<int32_t, int> seen;
    for (int j = 0; j < r.num_teams(); ++j) seen[r.team_expert_ids[j]]++;
    if (static_cast<int>(seen.size()) != w.num_active)
        return "expert appears zero or duplicated";
    for (auto& kv : seen)
        if (kv.second != 1) return "expert assigned to multiple teams";

    int64_t execute = 0;
    for (int wv = 0; wv < r.num_waves(); ++wv) {
        int beg = r.wave_offsets[wv], end = r.wave_offsets[wv + 1];
        int thread_sum = 0;
        int64_t wave_max = 0;
        for (int j = beg; j < end; ++j) {
            int32_t eid = r.team_expert_ids[j];
            int32_t th = r.team_threads[j];
            thread_sum += th;
            int64_t t = cm.estimate(routes_of[eid], th);
            if (t > wave_max) wave_max = t;
        }
        if (thread_sum > r.num_cores) return "wave thread budget exceeded";
        execute += wave_max;
    }
    if (execute != r.estimated_execute_cost_ns)
        return "recomputed execute != reported";
    return "";
}

// ----------------------------- timing -----------------------------

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p * (v.size() - 1) + 0.5);
    return v[idx];
}

struct Timing {
    double mean, median, p90, p99;
};

template <typename F>
static Timing bench(F&& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> samples(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        samples[i] =
            std::chrono::duration<double, std::nano>(t1 - t0).count();
    }
    double sum = 0;
    for (double s : samples) sum += s;
    Timing t;
    t.mean = sum / iters;
    t.median = percentile(samples, 0.50);
    t.p90 = percentile(samples, 0.90);
    t.p99 = percentile(samples, 0.99);
    return t;
}

static std::string fmt_ns(double ns) {
    char buf[64];
    if (ns < 1e3)
        std::snprintf(buf, sizeof(buf), "%7.1f ns", ns);
    else if (ns < 1e6)
        std::snprintf(buf, sizeof(buf), "%7.2f us", ns / 1e3);
    else
        std::snprintf(buf, sizeof(buf), "%7.3f ms", ns / 1e6);
    return buf;
}

int main(int argc, char** argv) {
    bool csv = false;
    std::string only_kind;
    std::string only_dist;
    int only_cores = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0) csv = true;
        else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) only_kind = argv[++i];
        else if (std::strcmp(argv[i], "--dist") == 0 && i + 1 < argc) only_dist = argv[++i];
        else if (std::strcmp(argv[i], "--cores") == 0 && i + 1 < argc) only_cores = std::atoi(argv[++i]);
    }

    const int E = 256;
    const int top_k = 6;
    const int tokens = 2048;
    const long total_routes = static_cast<long>(tokens) * top_k;
    const CostModel cm = CostModel::synthetic();

    struct Case { const char* dist; int active_k; int hot_k; double hot_frac; };
    std::vector<Case> cases = {
        {"zipf", 0, 0, 0.0},
        {"hotspot", 0, 4, 0.75},
        {"active_subset", 6, 0, 0.0},
        {"uniform", 0, 0, 0.0},
    };
    std::vector<int> core_list = {8, 16, 64, 79};
    std::vector<PlanKind> kinds = {
        PlanKind::FIXED_GLOBAL_THREADS, PlanKind::SORTED_TOKEN_BALANCED_1T,
        PlanKind::UNIFORM_WAVES, PlanKind::ENUMERATE_CORE_GROUPS,
        PlanKind::GREEDY_MARGINAL_GAIN, PlanKind::LOAD_PROPORTIONAL,
        PlanKind::SQRT_LOAD, PlanKind::LOG_LOAD, PlanKind::HEAVY_LIGHT_HYBRID,
        PlanKind::KARMARKAR_KARP};

    if (csv) printf("dist,cores,active,planner,prepare_ns,plan_ns,total_ns,waves,execute_ns\n");

    int failures = 0;
    for (const auto& c : cases) {
        if (!only_dist.empty() && only_dist != c.dist) continue;
        std::vector<int32_t> routes =
            gen_routes(c.dist, E, total_routes, c.active_k, c.hot_k, c.hot_frac);
        for (int C : core_list) {
            if (only_cores && only_cores != C) continue;
            Workload w0 = prepare_workload(routes.data(), E, C, cm);
            Timing tp = bench([&]() { volatile auto w = prepare_workload(routes.data(), E, C, cm); (void)w; }, 200, 2000);

            if (!csv)
                printf("\n[dist=%s cores=%d active=%d]  prepare=%s\n", c.dist, C,
                       w0.num_active, fmt_ns(tp.median).c_str());
            for (PlanKind k : kinds) {
                if (!only_kind.empty() && only_kind != plan_kind_name(k)) continue;
                // correctness
                PlanResult r = run_planner(k, w0);
                std::string err = check_plan(r, w0, cm);
                if (!err.empty()) {
                    printf("  !! %s FAILED: %s\n", plan_kind_name(k), err.c_str());
                    failures++;
                }
                // latency (planner only, workload prepared once)
                int iters = (k == PlanKind::GREEDY_MARGINAL_GAIN ||
                             k == PlanKind::ENUMERATE_CORE_GROUPS)
                                ? 2000
                                : 20000;
                Timing t = bench([&]() { volatile auto rr = run_planner(k, w0); (void)rr; },
                                 std::max(50, iters / 10), iters);
                double total = tp.median + t.median;
                if (csv) {
                    printf("%s,%d,%d,%s,%.1f,%.1f,%.1f,%d,%lld\n", c.dist, C,
                           w0.num_active, plan_kind_name(k), tp.median, t.median,
                           total, r.num_waves(),
                           (long long)r.estimated_execute_cost_ns);
                } else {
                    printf("  %-26s plan=%s  +prep=%s  waves=%-3d exec=%s\n",
                           plan_kind_name(k), fmt_ns(t.median).c_str(),
                           fmt_ns(total).c_str(), r.num_waves(),
                           fmt_ns((double)r.estimated_execute_cost_ns).c_str());
                }
            }
        }
    }
    if (!csv) {
        if (failures == 0)
            printf("\nAll correctness invariants passed.\n");
        else
            printf("\n%d correctness FAILURES.\n", failures);
    }
    return failures == 0 ? 0 : 1;
}
