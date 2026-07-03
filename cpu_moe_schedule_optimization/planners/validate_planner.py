"""End-to-end validation: planner-chosen plan vs baselines (coop[8], expert-parallel[1x8]).

Confirms (a) chosen plan predicted~=measured and (b) chosen <= best baseline across
regimes (balanced / hotspot / decode / one-dominant). Run on target:
  OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE FUSED_CPP_MOE_FUSED_PACKA=1 taskset -c 0-7 \\
    .venv/bin/python cpu_moe_schedule_optimization/planners/validate_planner.py
"""
import sys, time, statistics, torch
sys.path.insert(0, "cpu_moe_schedule_optimization/cost_model")
sys.path.insert(0, "cpu_moe_schedule_optimization/planners")
from phase_model import ContentionCostModel
from interval_planner import IntervalPlanner
from fused_cpp.moe import (
    fused_moe_bf16_tiled_async, prepare_fused_moe_bf16_tiled_weights)

PROF = "cpu_moe_schedule_optimization/cost_model/profiles/contention_async_aws_8c_sha46228bb_20260703.json"
H, F, E = 4096, 512, 20
WARMUP, RUNS = 5, 20
model = ContentionCostModel(PROF); pl = IntervalPlanner(model, 8)

def bf16(*s, sc=0.01):
    t = torch.empty(*s, dtype=torch.bfloat16); return t.normal_(0.0, sc)
torch.manual_seed(0); torch.set_num_threads(1)
packed = prepare_fused_moe_bf16_tiled_weights(bf16(E, 2 * F, H), bf16(E, H, F), fuse_silu=True)
def i32(x): return torch.tensor(x, dtype=torch.int32)

def measure(bridge, experts):
    tokens = sum(r for _, r in experts)
    ids = torch.empty((tokens, 1), dtype=torch.int32); tok = 0
    for e, r in experts:
        for _ in range(r): ids[tok, 0] = e; tok += 1
    wt = torch.ones((tokens, 1), dtype=torch.float32); x = bf16(tokens, H)
    nt = bridge["num_threads"]
    args = (i32(bridge["task_expert_ids"]), i32(bridge["task_core_begins"]),
            i32(bridge["task_threads"]), i32(bridge["task_dep_offsets"]), i32(bridge["task_deps"]))
    cpu = i32(bridge["thread_cpu_ids"])
    def run():
        return fused_moe_bf16_tiled_async(x, packed, wt, ids, *args,
            thread_cpu_ids=cpu, num_threads=nt, activation="silu", skip_weighted=True)
    for _ in range(WARMUP): run()
    ts = [ (lambda t0: (run(), time.perf_counter_ns() - t0)[1])(time.perf_counter_ns()) for _ in range(RUNS) ]
    return statistics.median(ts)

DISTS = {
    "balanced-large":    [(i, 512) for i in range(4)],
    "hotspot":           [(0, 1536), (1, 256), (2, 128), (3, 64), (4, 64)],
    "decode-many-small": [(i, 8) for i in range(16)],
    "one-dominant":      [(0, 2048), (1, 64), (2, 64)],
}
print("%-20s %-14s %9s %9s | %9s %9s   winner_ok" %
      ("dist", "chosen_shape", "pred_ms", "meas_ms", "coop_ms", "ep_ms"))
for name, experts in DISTS.items():
    r = pl.plan(experts)
    chosen = measure(r["bridge"], experts) / 1e6
    coop = measure(pl.to_async_bridge(pl.score_shape(experts, (8,))[1]), experts) / 1e6
    ep = measure(pl.to_async_bridge(pl.score_shape(experts, tuple([1]*8))[1]), experts) / 1e6
    ok = chosen <= min(coop, ep) + 0.05 * min(coop, ep)
    print("%-20s %-14s %9.3f %9.3f | %9.3f %9.3f   %s" %
          (name, str(r["shape"]), r["makespan_ns"]/1e6, chosen, coop, ep, "OK" if ok else "WORSE"))
