"""End-to-end MoE scheduling comparison (realistic prefill, top_k>1, real routing):
  default  = shipping fused_moe_bf16_tiled scheduling
  coop_a   = async interval-DAG, cooperative [8] shape, +packa (infrastructure)
  planned  = async interval-DAG, cost-model planner-chosen shape

Finding (AWS 8c): default->coop_a is 2-7x (the whole win = async+packa infra);
coop_a->planned is ~0% on 8-core prefill (cooperative [8] is already optimal;
planner sometimes slightly worse + its overhead). Outputs match default
(close=True). Planner value is confined to decode / higher core counts.

Run: OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE FUSED_CPP_MOE_FUSED_PACKA=1 taskset -c 0-7 \\
  .venv/bin/python cpu_moe_schedule_optimization/planners/bench_e2e_scheduling.py
"""

import sys
import statistics
import time

import torch

sys.path.insert(0, "cpu_moe_schedule_optimization/cost_model")
sys.path.insert(0, "cpu_moe_schedule_optimization/planners")
from phase_model import ContentionCostModel
from planned_moe import PlannedMoE, route_counts
from fused_cpp.moe import (
    fused_moe_bf16_tiled,
    fused_moe_bf16_tiled_async,
    prepare_fused_moe_bf16_tiled_weights,
)

PROF = "cpu_moe_schedule_optimization/cost_model/profiles/contention_async_aws_8c_sha46228bb_20260703.json"
H, F, EMAX = 4096, 512, 64
model = ContentionCostModel(PROF)
pm = PlannedMoE(model, 8)


def bf16(*s, sc=0.01):
    t = torch.empty(*s, dtype=torch.bfloat16)
    return t.normal_(0.0, sc)


torch.manual_seed(0)
torch.set_num_threads(1)
w13 = bf16(EMAX, 2 * F, H)
w2 = bf16(EMAX, H, F)
fw = prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)


def i32(x):
    return torch.tensor(x, dtype=torch.int32)


def routing(tokens, E, top_k, hot=0, hot_bias=4.0):
    logits = torch.randn(tokens, E)
    if hot:
        logits[:, :hot] += hot_bias
    tw, ti = torch.topk(torch.softmax(logits, -1), top_k, dim=-1)
    tw = (tw / tw.sum(-1, keepdim=True)).to(torch.float32)
    return ti.to(torch.int32), tw


def med(fn, warm=5, runs=15):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter_ns()
        fn()
        ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts) / 1e6


CFG = [
    ("tk2 E8 uniform", 2048, 8, 2, 0),
    ("tk2 E8 hotspot", 2048, 8, 2, 2),
    ("tk6 E8 uniform", 2048, 8, 6, 0),
    ("tk6 E64 uniform", 2048, 64, 6, 0),
    ("tk6 E64 hotspot", 2048, 64, 6, 8),
]
print("%-18s %6s %9s %8s %9s %8s   %s" % ("config", "close", "default", "coop_a", "planned", "plan_us", "shape"))
for name, tokens, E, tk, hot in CFG:
    ids, tw = routing(tokens, E, tk, hot)
    x = bf16(tokens, H)
    counts = route_counts(ids, E)
    pm.plan_for(counts, dynamic_tail_pool=False)
    b = pm.plan_for(counts, dynamic_tail_pool=False)  # warm
    plan_us = pm.last["planner_overhead_ns"] / 1e3
    from interval_planner import IntervalPlanner

    pl = IntervalPlanner(model, 8)
    bc = pl.to_async_bridge(pl.score_shape(counts, (8,))[1])  # async coop [8], same packa

    def run_def():
        return fused_moe_bf16_tiled(x, fw, tw, ids, num_threads=8, activation="silu", silu_poly_degree=5)

    def mk(br):
        a = (
            i32(br["task_expert_ids"]),
            i32(br["task_core_begins"]),
            i32(br["task_threads"]),
            i32(br["task_dep_offsets"]),
            i32(br["task_deps"]),
        )
        return lambda: fused_moe_bf16_tiled_async(
            x,
            fw,
            tw,
            ids,
            *a,
            thread_cpu_ids=i32(br["thread_cpu_ids"]),
            num_threads=br["num_threads"],
            activation="silu",
        )

    run_pl = mk(b)
    run_coop = mk(bc)
    close = torch.allclose(run_def().float(), run_pl().float(), atol=5e-2, rtol=5e-2)
    d = med(run_def)
    c = med(run_coop)
    p = med(run_pl)
    print("%-18s %6s %9.1f %8.1f %9.1f %8.1f   %s" % (name, str(bool(close)), d, c, p, plan_us, str(pm.last["shape"])))
