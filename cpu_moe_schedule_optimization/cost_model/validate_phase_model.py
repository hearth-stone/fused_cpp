"""Out-of-sample calibration/validation for the phase-based contention model.

Measures real makespan of HETEROGENEOUS concurrent expert groups (mixed routes
per team -- a dimension never present in the homogeneous block-2 calibration)
via fused_moe_bf16_tiled_async, and compares to phase_model predictions and the
scalar baseline.

Overfit guard: derate(n) is fit only from homogeneous block-2 data; these
configs are out-of-sample; T_iso lookups are on-grid (routes in {64,512,2048},
threads in {1,2,4,6,8}) so interpolation error does not confound the contention
test. The model has no free parameters tuned to these points.

Run on the target (AWS 8c):
  OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE FUSED_CPP_MOE_FUSED_PACKA=1 \
    taskset -c 0-7 .venv/bin/python \
    cpu_moe_schedule_optimization/cost_model/validate_phase_model.py
"""
import sys, time, statistics, torch
sys.path.insert(0, "cpu_moe_schedule_optimization/cost_model")
from phase_model import ContentionCostModel
from fused_cpp.moe import (
    fused_moe_bf16_tiled_async, prepare_fused_moe_bf16_tiled_weights)

PROF = "cpu_moe_schedule_optimization/cost_model/profiles/contention_async_aws_8c_sha46228bb_20260703.json"
H, F, E = 4096, 512, 8
WARMUP, RUNS = 5, 20
model = ContentionCostModel(PROF)

def bf16(*s, sc=0.01):
    t = torch.empty(*s, dtype=torch.bfloat16); return t.normal_(0.0, sc)

torch.manual_seed(0); torch.set_num_threads(1)
packed = prepare_fused_moe_bf16_tiled_weights(bf16(E, 2 * F, H), bf16(E, H, F), fuse_silu=True)
def i32(x): return torch.tensor(x, dtype=torch.int32)

def measure(config):  # config = [(routes, threads), ...]
    routes = [r for (r, _) in config]; shape = [t for (_, t) in config]
    K = len(config); nthreads = sum(shape)
    begins, c = [], 0
    for t in shape: begins.append(c); c += t
    tokens = sum(routes)
    ids = torch.empty((tokens, 1), dtype=torch.int32)
    tok = 0
    for e, r in enumerate(routes):
        for _ in range(r): ids[tok, 0] = e; tok += 1
    wt = torch.ones((tokens, 1), dtype=torch.float32); x = bf16(tokens, H)
    te, cb, th = i32(list(range(K))), i32(begins), i32(shape)
    do, dp = i32([0] * (K + 1)), i32([]); cpu = i32(list(range(nthreads)))
    def run():
        return fused_moe_bf16_tiled_async(x, packed, wt, ids, te, cb, th, do, dp,
            thread_cpu_ids=cpu, num_threads=nthreads, activation="silu", skip_weighted=True)
    for _ in range(WARMUP): run()
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter_ns(); run(); ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts)

CONFIGS = [
    [(2048, 4), (64, 2), (64, 2)],
    [(2048, 4), (512, 4)],
    [(2048, 6), (64, 2)],
    [(512, 4), (64, 2), (64, 2)],
    [(2048, 2), (2048, 2), (64, 2), (64, 2)],
    [(2048, 4), (512, 2), (64, 2)],
    [(2048, 6), (64, 1), (64, 1)],
    [(512, 2), (512, 2), (512, 2), (64, 2)],
]

print("derate(n):", {n: round(model.derate(n), 3) for n in (2, 3, 4)})
print("%-40s %9s %9s %9s %8s %8s" % ("config", "meas_ms", "phase_ms", "scalar_ms", "ph_err%", "sc_err%"))
ph_errs, sc_errs = [], []
for cfg in CONFIGS:
    meas = measure(cfg)
    ph = model.phase_makespan(cfg); sc = model.scalar_makespan(cfg)
    phe = (ph - meas) / meas * 100; sce = (sc - meas) / meas * 100
    ph_errs.append(abs(phe)); sc_errs.append(abs(sce))
    print("%-40s %9.3f %9.3f %9.3f %+8.1f %+8.1f"
          % (str(cfg), meas / 1e6, ph / 1e6, sc / 1e6, phe, sce))
print("\nphase  |err|: median=%.1f%% max=%.1f%%" % (statistics.median(ph_errs), max(ph_errs)))
print("scalar |err|: median=%.1f%% max=%.1f%%" % (statistics.median(sc_errs), max(sc_errs)))
