"""End-to-end validation for the interval-DAG makespan predictor.

Runs hand-built interval-DAG plans (concurrency, pure dependency chains, waves,
and staggered heterogeneous starts) through fused_moe_bf16_tiled_async and
compares measured makespan to ContentionCostModel.dag_makespan. Unlike the
derate table (independent concurrent tasks), these exercise dependencies and
staggered starts -- the behavior the planner will actually produce.

Run on the target (AWS 8c):
  OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE FUSED_CPP_MOE_FUSED_PACKA=1 \
    taskset -c 0-7 .venv/bin/python \
    cpu_moe_schedule_optimization/cost_model/validate_dag_makespan.py
"""
import sys, time, statistics, torch
sys.path.insert(0, "cpu_moe_schedule_optimization/cost_model")
from phase_model import ContentionCostModel
from fused_cpp.moe import (
    fused_moe_bf16_tiled_async, prepare_fused_moe_bf16_tiled_weights)

PROF = "cpu_moe_schedule_optimization/cost_model/profiles/contention_async_aws_8c_sha46228bb_20260703.json"
H, F, E = 4096, 512, 12
WARMUP, RUNS = 5, 20
model = ContentionCostModel(PROF)

def bf16(*s, sc=0.01):
    t = torch.empty(*s, dtype=torch.bfloat16); return t.normal_(0.0, sc)

torch.manual_seed(0); torch.set_num_threads(1)
packed = prepare_fused_moe_bf16_tiled_weights(bf16(E, 2 * F, H), bf16(E, H, F), fuse_silu=True)
def i32(x): return torch.tensor(x, dtype=torch.int32)

# each plan: list of tasks (routes, threads, core_begin, deps=[task indices])
PLANS = {
    "P1 concurrent 2xbig":        [(2048, 4, 0, []), (2048, 4, 4, [])],
    "P2 seq chain full8":         [(2048, 8, 0, []), (2048, 8, 0, [0])],
    "P3 two waves 4+4":           [(512, 4, 0, []), (512, 4, 4, []),
                                   (512, 4, 0, [0]), (512, 4, 4, [1])],
    "P4 stagger big+smalls":      [(2048, 4, 0, []), (64, 2, 4, []), (64, 2, 6, []),
                                   (512, 2, 4, [1]), (512, 2, 6, [2])],
    "P5 hotspot-like":            [(2048, 4, 0, []), (512, 2, 4, []), (512, 2, 6, []),
                                   (64, 2, 4, [1]), (64, 2, 6, [2]), (2048, 4, 0, [0])],
}

def measure(plan):
    K = len(plan)
    nthreads = max(cb + th for (_, th, cb, _) in plan)
    routes = [r for (r, _, _, _) in plan]
    tokens = sum(routes)
    ids = torch.empty((tokens, 1), dtype=torch.int32); tok = 0
    for e, r in enumerate(routes):
        for _ in range(r): ids[tok, 0] = e; tok += 1
    wt = torch.ones((tokens, 1), dtype=torch.float32); x = bf16(tokens, H)
    dep_off = [0]; dep_flat = []
    for (_, _, _, deps) in plan:
        dep_flat.extend(deps); dep_off.append(len(dep_flat))
    te = i32(list(range(K))); cb = i32([c for (_, _, c, _) in plan])
    th = i32([t for (_, t, _, _) in plan]); do = i32(dep_off); dp = i32(dep_flat)
    cpu = i32(list(range(nthreads)))
    def run():
        return fused_moe_bf16_tiled_async(x, packed, wt, ids, te, cb, th, do, dp,
            thread_cpu_ids=cpu, num_threads=nthreads, activation="silu", skip_weighted=True)
    for _ in range(WARMUP): run()
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter_ns(); run(); ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts)

print("%-26s %9s %9s %8s" % ("plan", "meas_ms", "pred_ms", "err%"))
errs = []
for name, plan in PLANS.items():
    meas = measure(plan)
    tasks = [(r, t, d) for (r, t, _, d) in plan]
    pred = model.dag_makespan(tasks)
    e = (pred - meas) / meas * 100; errs.append(abs(e))
    print("%-26s %9.3f %9.3f %+8.1f" % (name, meas / 1e6, pred / 1e6, e))
print("\nDAG pred |err|: median=%.1f%% max=%.1f%%" % (statistics.median(errs), max(errs)))
