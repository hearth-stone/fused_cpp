# m5 TP2 DSV4 quick-planner optimization stages

Date: 2026-08-16

## Scope and acceptance gate

This record tracks the staged optimization of the production analytical quick
planner. It measures plan construction only; it is not a fused-MoE kernel or
model E2E result. The public Python API, candidate space, cost objective,
uncertainty, early-merge policy, Plan V2 schema, and native runtime are fixed.

Each stage must preserve the fully materialized plan on both ranks for all 43
captured layers. Performance is accepted when forced-miss planner overhead
improves by at least 10% for an algorithm/boundary stage, or 2% for a local
SIMD stage, without a cache-hit regression beyond 2%. A stage is independently
revertible by its commit.

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

The immediate-hit change was -0.78% on rank0 and +0.43% on rank1 by median
paired layer, within the 2% gate. Sequential first-pass total planner time fell
from 5.759 s to 2.079 s on rank0 (+63.91%) and from 5.879 s to 2.090 s on
rank1 (+64.45%). First-pass medians are noisier than forced misses because the
two ranks progress through differently sized layers concurrently.

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

## Validation

```text
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_moe_cost_model_v2.py tests/test_moe_analytic_model.py
.venv/bin/python -m ruff check \
  cpu_moe_schedule_optimization/planners/interval_planner.py \
  tests/test_moe_cost_model_v2.py
```

Stage 1 adds direct coverage for generic/heap assignment parity, analytical
cost deduplication, and the rounded-score low-lane-id tie-break. Later stages
will append results to the same table.
