"""Planner runtime-overhead benchmark: cold search vs warm cache-hit vs kernel
time across prefill/decode regimes, plus cache hit-rate + accuracy over a decode
sequence. Confirms cold planning must be cached (0.7-13ms), and a coarse
regime-level routing signature gives 96-98% cache hit-rate.

Run on target:
  OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE FUSED_CPP_MOE_FUSED_PACKA=1 taskset -c 0-7 \\
    .venv/bin/python cpu_moe_schedule_optimization/planners/bench_planner_overhead.py
"""

import sys
import statistics
import time

import torch

sys.path.insert(0, "cpu_moe_schedule_optimization/cost_model")
sys.path.insert(0, "cpu_moe_schedule_optimization/planners")
from phase_model import ContentionCostModel
from planned_moe import PlannedMoE, route_counts
from interval_planner import IntervalPlanner
from fused_cpp.moe import fused_moe_bf16_tiled_async, prepare_fused_moe_bf16_tiled_weights

PROF = "cpu_moe_schedule_optimization/cost_model/profiles/contention_async_aws_8c_sha46228bb_20260703.json"
H, F, EMAX = 4096, 512, 64
model = ContentionCostModel(PROF)


def bf16(*s, sc=0.01):
    t = torch.empty(*s, dtype=torch.bfloat16)
    return t.normal_(0.0, sc)


torch.manual_seed(0)
torch.set_num_threads(1)
packed = prepare_fused_moe_bf16_tiled_weights(bf16(EMAX, 2 * F, H), bf16(EMAX, H, F), fuse_silu=True)


def i32(x):
    return torch.tensor(x, dtype=torch.int32)


def routing(tokens, E, hot=0, hot_bias=4.0):
    logits = torch.randn(tokens, E)
    if hot:
        logits[:, :hot] += hot_bias
    ids = torch.argmax(logits, dim=1, keepdim=True).to(torch.int32)  # top_k=1
    wt = torch.ones((tokens, 1), dtype=torch.float32)
    return ids, wt


def measure_kernel(bridge, ids, wt):
    x = bf16(ids.shape[0], H)
    args = (
        i32(bridge["task_expert_ids"]),
        i32(bridge["task_core_begins"]),
        i32(bridge["task_threads"]),
        i32(bridge["task_dep_offsets"]),
        i32(bridge["task_deps"]),
    )
    cpu = i32(bridge["thread_cpu_ids"])
    nt = bridge["num_threads"]

    def run():
        return fused_moe_bf16_tiled_async(
            x,
            packed,
            wt,
            ids,
            *args,
            thread_cpu_ids=cpu,
            num_threads=nt,
            activation="silu",
            skip_weighted=True,
        )

    for _ in range(5):
        run()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter_ns()
        run()
        ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts) / 1e6


CONFIGS = [
    ("prefill-uniform  ", 2048, 8, 0),
    ("prefill-hotspot  ", 2048, 8, 2),
    ("decode-8e        ", 16, 8, 0),
    ("decode-64e       ", 64, 64, 0),
]
pl = IntervalPlanner(model, 8)
print("%-18s %6s %8s %9s %9s | %9s %9s" % ("config", "active", "cold_us", "warm_us", "kern_ms", "coop_ms", "shape"))
for name, tokens, E, hot in CONFIGS:
    ids, wt = routing(tokens, E, hot)
    counts = route_counts(ids, E)
    pm = PlannedMoE(model, 8)
    pm.plan_for(counts, dynamic_tail_pool=False)
    cold = pm.last["planner_overhead_ns"] / 1e3
    b_warm = pm.plan_for(counts, dynamic_tail_pool=False)
    warm = pm.last["planner_overhead_ns"] / 1e3
    shape = pm.last["shape"]
    kern = measure_kernel(b_warm, ids, wt)
    coop = measure_kernel(pl.to_async_bridge(pl.score_shape(counts, (8,))[1]), ids, wt)
    print("%-18s %6d %8.1f %9.1f %9.3f | %9.3f  %s" % (name, len(counts), cold, warm, kern, coop, str(shape)))

# cache hit-rate + amortized planner overhead over a decode sequence (re-routing
# each step from the same distribution) -- the justification for caching.
print("\n-- amortization over 100 decode steps (re-routed each step) --")
for name, tokens, E, hot in [
    ("decode-8e", 16, 8, 0),
    ("decode-64e", 64, 64, 0),
    ("decode-64e-hot", 64, 64, 8),
]:
    pm = PlannedMoE(model, 8)
    hits = 0
    ov = []
    agree = 0
    for step in range(100):
        ids, wt = routing(tokens, E, hot)
        counts = route_counts(ids, E)
        pm.plan_for(counts, dynamic_tail_pool=False)
        hits += 1 if pm.last["cache_hit"] else 0
        ov.append(pm.last["planner_overhead_ns"] / 1e3)
        full = pl.plan(counts, dynamic_tail_pool=False)["shape"]  # legacy executor baseline
        agree += 1 if pm.last["shape"] == full else 0
    print(
        "%-16s hit_rate=%3.0f%%  mean_overhead=%7.1fus  cache_shape_matches_full=%3.0f%%"
        % (name, hits, statistics.mean(ov), agree)
    )
