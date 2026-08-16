# m5 TP2 DSV4 quick-planner optimization stages

Date: 2026-08-16

## Scope and acceptance gate

This record tracks the staged optimization of the production analytical quick
planner. It measures plan construction only; it is not a fused-MoE kernel or
model E2E result. The public Python API, candidate space, cost objective,
uncertainty, Plan V2 schema, and native runtime are fixed. Stages 1--4 preserve
the prior early-merge policy; stage 5 intentionally fixes it on.

Stages 1--4 must preserve the fully materialized plan on both ranks for all 43
captured layers. Stage 5 may change only `early_merge`. Performance is accepted
when forced-miss planner overhead improves by at least 10% for an
algorithm/boundary stage, or 2% for a local SIMD stage, without a cache-hit or
kernel regression beyond 2%. A stage is independently revertible by its commit.

## System and method

- Host: `AmazonM5192Cores` (m5), 192 Neoverse-V3 cores, two 96-core NUMA ranks.
- Shape: TP2, `H=4096`, `F=1024`, `E=256`, 2048 tokens, TopK=6.
- Routing: `/tmp/m5_tp2_dsv4_token_expert_ids_2048_20260816.json`, 43 layers.
- Profiles: `profiles/m5-tp2-rank0.json` and `profiles/m5-tp2-rank1.json`.
- Placement: rank0 CPUs `0-95`, NUMA0; rank1 CPUs `96-191`, NUMA1.
- Concurrency: both rank benchmark processes run concurrently.
- Timing: one import/page-fault warmup, then 7 forced misses and one immediate
  hit per repeat per layer. The table reports the median across the 43
  per-layer medians unless noted otherwise.
- Cache realism: sequential first pass hit `0/43` on both ranks.

The benchmark entrypoint was `/tmp/bench_m5_real_routing_planner.py`; each rank
used the matching profile and `--repeats 7`. Temporary JSON results remain on
m5 under `/tmp/m5_tp2_dsv4_planner_*_20260816.json` and are intentionally not
committed.

## Stage results

| stage | rank | miss wall | miss planner | miss search | hit planner | median per-layer planner gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline Python generic LPT | 0 | 124.749 ms | 123.892 ms | 123.584 ms | 32.432 ms | -- |
| baseline Python generic LPT | 1 | 126.788 ms | 125.917 ms | 125.610 ms | 32.676 ms | -- |
| 1: deduplicated cost + homogeneous heap LPT | 0 | 36.486 ms | 35.633 ms | 35.334 ms | 32.665 ms | +71.15% |
| 1: deduplicated cost + homogeneous heap LPT | 1 | 36.258 ms | 35.378 ms | 35.067 ms | 32.569 ms | +71.87% |
| 2: single-thread native LPT/search boundary | 0 | 32.869 ms | 31.998 ms | 31.683 ms | 29.633 ms | +10.15% vs stage 1; +74.17% cumulative |
| 2: single-thread native LPT/search boundary | 1 | 32.617 ms | 31.739 ms | 31.425 ms | 29.546 ms | +10.01% vs stage 1; +74.73% cumulative |
| 3: candidate parallelism rejected; 1T retained | 0 | 32.675 ms | 31.823 ms | 31.522 ms | 29.546 ms | no production change |
| 3: candidate parallelism rejected; 1T retained | 1 | 32.554 ms | 31.676 ms | 31.360 ms | 29.583 ms | no production change |
| 4: fixed resource vectors + cached phase base | 0 | 24.406 ms | 23.565 ms | 23.263 ms | 22.512 ms | +26.00% vs stage 3; +80.97% cumulative |
| 4: fixed resource vectors + cached phase base | 1 | 24.387 ms | 23.549 ms | 23.247 ms | 22.617 ms | +26.17% vs stage 3; +81.33% cumulative |
| 5: fixed early merge on | 0 | 2.239 ms | 1.424 ms | 1.131 ms | 0.474 ms | +93.95% vs stage 4 |
| 5: fixed early merge on | 1 | 2.223 ms | 1.401 ms | 1.104 ms | 0.466 ms | +94.05% vs stage 4 |

The immediate-hit change was -0.78% on rank0 and +0.43% on rank1 by median
paired layer, within the 2% gate. Sequential first-pass total planner time fell
from 5.759 s to 2.079 s on rank0 (+63.91%) and from 5.879 s to 2.090 s on
rank1 (+64.45%). First-pass medians are noisier than forced misses because the
two ranks progress through differently sized layers concurrently.

Stage 2 also improved immediate-hit planner overhead by 9.68% on rank0 and
9.27% on rank1 relative to stage 1. Sequential first-pass total planner time
fell from 2.079 s to 1.940 s on rank0 (+6.69%) and from 2.090 s to 1.952 s on
rank1 (+6.58%).

## Stage 1 implementation and parity

For a homogeneous width, isolated expert cost does not depend on the target
lane. Stage 1 computes each distinct `(routes, width)` cost once and assigns
with a `(load, lane_id)` heap. It retains the original Python floating-point
tie semantics by expanding every heap entry whose `load + current_cost` rounds
to the same minimum score, then choosing the lowest lane id.

The first heap prototype compared raw loads and changed task assignment when a
large current cost rounded unequal loads to the same score. That prototype was
rejected before measurement. The corrected implementation produced identical
files for all 43 layers:

| rank | baseline and stage-1 Plan JSON SHA256 |
| --- | --- |
| 0 | `78507c67cd2976cf19dedbf645d6bb0135368a741b895b6795d709aa3a96be06` |
| 1 | `05eb8e8a971d822134edf437587158c1e246acc926a9a21a37ea5af7737a15c5` |

The comparison includes every `AsyncMoEPlanV2` tensor field, task order,
dependencies, stage windows, CPU ids, execution mode, and tri-state
`early_merge`, plus the selected shape.

## Stage 2 implementation and parity

Stage 2 keeps exact analytical `T_iso` evaluation in Python and passes one
immutable cost row per homogeneous shape to a single-threaded C++ planner.
Native code performs the deterministic heap LPT assignment, candidate scoring,
and winner selection. The pybind boundary fully materializes only the selected
candidate; non-selected candidates return only the fields needed by ranking.
Cached-shape materialization uses the same native assignment, while builds
without `NativeQuickPlanner` retain the stage-1 Python fallback.

The native implementation preserves stable route ordering and the rounded
`load + cost` low-lane-id tie-break. All 43 fully materialized plans remained
byte-identical to stage 1 on both ranks, with the same SHA256 values shown
above. The runtime reported `planner_backend=cpp_quick`; this stage deliberately
uses one planner worker so candidate parallelism can be measured independently.

## Stage 3 candidate-thread sweep

Stage 3 preallocates one result slot per homogeneous shape and reuses the
native planner's fixed-index `ParallelFor`. Each candidate remains
single-threaded, exceptions are replayed by candidate index, and selection and
ranking occur after the join in original candidate order. The existing
`FUSED_CPP_MOE_PLANNER_THREADS` control and constructor parameter select the
worker count.

| workers | rank0 miss planner | rank1 miss planner | rank0 paired gain vs 1T | rank1 paired gain vs 1T |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 31.823 ms | 31.676 ms | -- | -- |
| 2 | 31.663 ms | 31.627 ms | +0.19% | +0.15% |
| 4 | 31.677 ms | 32.205 ms | +0.13% | -1.55% |
| 8 | 31.793 ms | 31.395 ms | -0.26% | +0.80% |

No shared worker count met the 10% stage gate. The native candidate portion is
too small relative to Python analytical cost evaluation and bridge lowering to
amortize OpenMP startup reliably. Production quick planning therefore defaults
to one worker; explicit multi-worker settings remain diagnostic. All worker
counts produced byte-identical Plan V2 JSON on both ranks.

## Stage 4 analytical resource-vector path

cProfile of 130 rank0 planner calls attributed 13.45 of 16.41 cumulative
seconds to early-merge analytical DAG simulation, versus 2.42 seconds to quick
cost-row construction. Within the DAG, `_active_phase_state` repeatedly built
resource dictionaries and recalculated immutable isolated phase duration.

Stage 4 emits one fixed-order eight-resource demand/time tuple per active phase
and event, then reads it by resource index. The public scalar diagnostic
accessors remain available, and immutable `AnalyticPhase.base_ns` is cached.
Python `sum/max` reduction order, resource formulas, spill fraction, service
capacity, dilation, event progression, and completion tolerance are unchanged.

Relative to the stage-3 same-binary 1T control, paired per-layer forced-miss
planner gains were 26.00% on rank0 and 26.17% on rank1. Immediate-hit planner
gains were 23.65% and 23.49%. Sequential first-pass total planner time fell
from 1.929 s to 1.513 s on rank0 (+21.56%) and from 1.954 s to 1.527 s on
rank1 (+21.88%). Both 43-layer Plan JSON files retained the original SHA256
values listed above.

## Stage 5 fixed early merge on

The previous analytical post-plan gate consumed most of both cache-miss and
cache-hit planning time, while all 43 captured m5 TP2 DSV4 plans retained
`early_merge=None` and the native team-load heuristic resolved every layer to
on. Stage 5 therefore materializes `early_merge=True` directly and removes the
completion-time DAG and routing-tail analysis from plan lowering. Compute-plan
candidates, scoring, pruning, cache identity, placement, windows, Plan V2
schema, and native execution remain unchanged.

With both ranks running concurrently and seven forced misses per layer, the
planner-overhead layer-median fell from 23.565/23.549 ms to 1.424/1.401 ms on
rank0/rank1, a 93.95%/94.05% reduction. Sequential first-pass total planner
time fell from 1.513/1.527 s to 0.429/0.425 s. Immediate-hit planner overhead
fell to 0.474/0.466 ms because the routing-dependent gate is no longer
recomputed after a histogram cache hit.

Before adoption, nine full-layer kernel sweeps compared auto, forced on, and
forced off on the same 43 routes. Rank0 totals were 1236.901/1237.567/1235.703
ms and rank1 totals were 1237.695/1238.013/1240.509 ms; the largest difference
was 0.23%, and every layer matched exactly. After adoption, five synchronized
forced-miss planner-plus-kernel sweeps measured current production at
1832.910/1828.793 ms, down from the pre-change 2973.364/2968.055 ms. This is a
38.36%/38.38% latency reduction while kernel time remains about 1.24 s.
Historical TP4/F512 bimodal measurements favored disabling early merge by
about 6%, so this global fixed-on policy retains a known cross-workload risk.

## Validation

```text
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_moe_native_interval_planner.py \
  tests/test_moe_cost_model_v2.py tests/test_moe_analytic_model.py
.venv/bin/python -m ruff check \
  cpu_moe_schedule_optimization/cost_model/analytic_model.py \
  cpu_moe_schedule_optimization/planners/interval_planner.py \
  cpu_moe_schedule_optimization/planners/planned_moe.py \
  tests/test_moe_native_interval_planner.py
```

Stage 1 adds direct coverage for generic/heap assignment parity, analytical
cost deduplication, and the rounded-score low-lane-id tie-break. Stage 2 adds
native analytical quick-plan equivalence and direct native tie coverage. Local
validation passed 150 tests; the matching m5 build passed 150 tests. Stage 3
extends native parity coverage across 1/2/4 workers; stage 4 directly checks
vector/scalar resource parity. Stage 5 checks fixed-on materialization and fails
if plan lowering invokes the analytical DAG. All planned stages are recorded
above.
