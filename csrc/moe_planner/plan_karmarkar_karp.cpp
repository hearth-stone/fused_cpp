// KARMARKAR_KARP planner (multiway Largest Differencing Method).
//
// Assigns each active expert ONE thread to one of C logical queues using
// multiway Karmarkar-Karp (LDM) to balance routed-token load across queues.
// More sophisticated than greedy LPT (SORTED_TOKEN_BALANCED_1T).
// Matches plan_karmarkar_karp() in the Python reference twin.
#include <algorithm>
#include <vector>

#include "planner_dispatch.h"

namespace moe_planner {

namespace {

// Node in the LDM pairing tree: one partial solution for C queues.
struct LdmNode {
    std::vector<int64_t> loads;          // C loads, sorted DESCENDING
    std::vector<std::vector<int>> buckets;  // C buckets of active indices
    int insertion_index;                 // for deterministic tie-breaking

    LdmNode(int C, int ins_idx) : loads(C, 0), buckets(C), insertion_index(ins_idx) {}
};

// Compute spread (max - min) for a node.
int64_t compute_spread(const LdmNode& node) {
    int C = static_cast<int>(node.loads.size());
    if (C == 0) return 0;
    return node.loads[0] - node.loads[C - 1];
}

// Compare two nodes for selection: return true if 'a' ranks higher than 'b'.
// Ranking: spread desc, loads[0] desc, full loads lexicographically desc,
// then insertion_index asc (earlier node wins ties).
bool node_compare(const LdmNode& a, const LdmNode& b) {
    int64_t spread_a = compute_spread(a);
    int64_t spread_b = compute_spread(b);
    if (spread_a != spread_b) return spread_a > spread_b;
    
    if (a.loads[0] != b.loads[0]) return a.loads[0] > b.loads[0];
    
    // Lexicographic comparison of full loads vector (descending).
    size_t len = std::min(a.loads.size(), b.loads.size());
    for (size_t i = 1; i < len; ++i) {  // i=0 already compared
        if (a.loads[i] != b.loads[i]) return a.loads[i] > b.loads[i];
    }
    
    // All loads equal: earlier insertion wins.
    return a.insertion_index < b.insertion_index;
}

}  // namespace

PlanResult plan_karmarkar_karp(const Workload& w) {
    PlanResult r;
    r.kind = PlanKind::KARMARKAR_KARP;
    r.num_cores = w.num_cores;

    const int C = w.num_cores;
    const int A = w.num_active;

    // Handle empty workload.
    if (A == 0) {
        r.wave_offsets.push_back(0);
        fill_active(r, w);
        return r;
    }

    // Initialize: one node per active expert.
    std::vector<LdmNode> nodes;
    nodes.reserve(A);
    for (int a = 0; a < A; ++a) {
        nodes.emplace_back(C, a);  // insertion_index = a
        LdmNode& node = nodes.back();
        node.loads[0] = w.routes[a];
        node.buckets[0].push_back(a);
        // loads already sorted: [routes[a], 0, ..., 0]
    }

    // LDM pairing loop.
    while (nodes.size() > 1) {
        // Find top two nodes: X (highest rank), Y (second highest).
        int idx_x = 0;
        for (int i = 1; i < static_cast<int>(nodes.size()); ++i) {
            if (node_compare(nodes[i], nodes[idx_x])) {
                idx_x = i;
            }
        }
        
        int idx_y = -1;
        for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
            if (i == idx_x) continue;
            if (idx_y < 0 || node_compare(nodes[i], nodes[idx_y])) {
                idx_y = i;
            }
        }

        // Extract X and Y (remove higher index first to preserve lower).
        if (idx_x > idx_y) std::swap(idx_x, idx_y);
        LdmNode Y = std::move(nodes[idx_y]);
        nodes.erase(nodes.begin() + idx_y);
        LdmNode X = std::move(nodes[idx_x]);
        nodes.erase(nodes.begin() + idx_x);

        // COMBINE: X.loads already descending.
        // Take Y in ASCENDING load order (reverse of its descending loads).
        std::vector<int> y_order(C);
        for (int i = 0; i < C; ++i) {
            y_order[i] = C - 1 - i;  // ascending order indices
        }

        int next_ins_idx = static_cast<int>(nodes.size());  // will be appended
        LdmNode merged(C, next_ins_idx);
        
        for (int i = 0; i < C; ++i) {
            int y_i = y_order[i];  // Y's index for ascending position i
            merged.loads[i] = X.loads[i] + Y.loads[y_i];
            merged.buckets[i] = X.buckets[i];  // copy X's bucket
            merged.buckets[i].insert(merged.buckets[i].end(),
                                     Y.buckets[y_i].begin(),
                                     Y.buckets[y_i].end());
        }

        // Stable sort the C (load, bucket) pairs by load DESCENDING.
        // Ties keep current relative order.
        std::vector<std::pair<int64_t, std::vector<int>>> pairs(C);
        for (int i = 0; i < C; ++i) {
            pairs[i] = {merged.loads[i], std::move(merged.buckets[i])};
        }
        std::stable_sort(pairs.begin(), pairs.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.first > rhs.first;  // descending
            });
        for (int i = 0; i < C; ++i) {
            merged.loads[i] = pairs[i].first;
            merged.buckets[i] = std::move(pairs[i].second);
        }

        // Append merged node.
        nodes.push_back(std::move(merged));
    }

    // Final node contains the C queues.
    const LdmNode& final_node = nodes[0];
    std::vector<std::vector<int>> queues = final_node.buckets;

    // Within each queue, sort by descending routes then ascending expert_id.
    for (int q = 0; q < C; ++q) {
        std::sort(queues[q].begin(), queues[q].end(),
            [&w](int a1, int a2) {
                if (w.routes[a1] != w.routes[a2]) {
                    return w.routes[a1] > w.routes[a2];
                }
                return w.expert_ids[a1] < w.expert_ids[a2];
            });
    }

    // Emit waves by depth: wave d gathers queue[q][d] for q = 0..C-1.
    int max_depth = 0;
    for (const auto& q : queues) {
        max_depth = std::max(max_depth, static_cast<int>(q.size()));
    }

    r.wave_offsets.push_back(0);
    int64_t execute = 0;
    for (int d = 0; d < max_depth; ++d) {
        int64_t wave_max = 0;
        bool any = false;
        for (int q = 0; q < C; ++q) {
            if (d < static_cast<int>(queues[q].size())) {
                int a = queues[q][d];
                int32_t expert_id = w.expert_ids[a];
                int64_t time_ns = w.cost(a, 1);
                r.team_expert_ids.push_back(expert_id);
                r.team_threads.push_back(1);
                if (time_ns > wave_max) wave_max = time_ns;
                any = true;
            }
        }
        if (any) {
            r.wave_offsets.push_back(static_cast<int32_t>(r.team_expert_ids.size()));
            execute += wave_max;
        }
    }

    r.estimated_execute_cost_ns = execute;
    fill_active(r, w);
    return r;
}

}  // namespace moe_planner
