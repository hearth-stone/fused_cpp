# Stage window order: thread-major ownership (prototype + E9 whole-plan A/B)

## Status

The production stage-window geometry now cuts windows inside each worker's own
contiguous N stripe instead of splitting every window across the team. The change is
performance-neutral on whole plans and keeps plan ranking intact, so it is adopted for
its ordering semantics, not for a speedup: `window_tiles` now means exactly what the
geometry reads as, one worker's owner window inside the stripe it owns.

Evidence: a lab prototype of the two mappings with the production JIT kernel, and E9, an
A/B of two builds of the machine's own checkout over 66 frozen whole-plan points on six
real route layers.

## What changed

Old (window-major): the stage was cut into windows of `threads * window_tiles` tiles and
each window was split across the team, so a worker's tiles were spread over N with stride
`threads * window_tiles`. New (thread-major): worker `j` owns one contiguous stripe of
`split_evenly(total_tiles, threads, j)` tiles and walks it in windows of `window_tiles`.

The team still consumes `threads * window_tiles` tiles per pass and the pass count is
unchanged, because `ceil(ceil(q/t)/w) = ceil(q/(t*w))`; the per-worker L2 footprint per
window, the number of passes over A and the tile coverage are unchanged as well. Only the
columns a worker touches change. `MATHEMATICAL_MODEL.md` section 1.1 carries the formulas.

Implementation: `csrc/moe/arm/common/fused_moe_bf16_tiled.cpp` (`StageWindowPlan`, the W2
owner scatter follows the same plan) and `cpu_moe_schedule_optimization/cost_model/
full_stage_geometry.py`, which the native geometry test is pinned against.

## Prototype (isolated harness)

`tmp/window_mapping_20260920/window_mapping_native.cpp` runs the production W13 JIT kernel
under both mappings, one team per `80/threads` cores on CPUs 240-319, cold weights (24
copies, 192 MiB), 3 warmup plus 17 timed iterations, and compares the output checksums.

- Loaded grid (`grid.json`, 44 cells where the two mappings actually differ): median
  delta (thread-major / window-major - 1) -0.033%, 29 of 44 negative; for panels of 48
  rows and more (33 cells) median -0.007%, largest single cell 1.42%.
- Small-M grid (`grid_small_m.json`, 33 cells, rows 12/24/48, widths 2/4/8/16): median
  -0.574%, 27 of 33 negative. Single-panel experts are where thread-major helps: a
  worker's stream is one sequential run instead of a strided one.
- Checksums identical in every cell of both grids.

(The commit message of `perf: own N stripes before cutting stage windows` quotes the
loaded-grid median with the opposite sign and reuses the loaded grid's 29 negative cells
for the small-M grid; the figures above are the ones the grid files support.)

## E9: whole plans, two builds

Design frozen before collection in `tmp/window_mapping_20260920/e9/design.md`.

Builds, both from the Arm machine's checkout, differing only in the minimal thread-major
patch plus an inert `fused_moe_bf16_tiled_backend_n_tile` binding:
A = thread-major (`_moe_C` sha256 e8f96d32c95bdece...), B = window-major control
(8e4b65478eaa82f8...); `_C` is byte-identical in both. The two builds' native window plans
were checked directly: for N=1024, n_tile=16, threads=4, window_tiles=1, A gives window 0
= [(0,16),(256,16),(512,16),(768,16)] and B gives [(0,16),(16,16),(32,16),(48,16)].

Workloads: the frozen E6 plan files, unchanged - six real route layers (r008 l1/l21, r016
l13/l28, r022 l2/l30) with 11 plans each, 66 points, `window_tiles` as chosen under the V3
table. Protocol: `tmp/m2_validation_20260919/group_w/bench.py` defaults (producer-hot A
over 40 rotating copies, cold B, 5 warmup + 31 runs, points shuffled per round),
`numactl --physcpubind=240-319 --membind=3`, jemalloc never-purge, idle guard (1-minute
load <= 2 before each session; observed 1.52-2.47). Sessions run A B B A per plan file
with four distinct seeds. Plan times are 26.95-29.60 ms.

Results (`analyze_e9.py`):

| quantity | value |
| --- | --- |
| median delta A/B - 1 over 66 points | -0.027% (mean -0.014%) |
| sign | 37 of 66 negative |
| p10 / p90 | -0.245% / +0.291% |
| min / max | -0.533% / +0.541% |
| within-build session spread (median, p90) | 0.261%, 0.651% |
| control vs E6b (old kernel, 08:22 UTC) | median 0.190%, max 0.718% |

Per workload: medians -0.115%, -0.003%, +0.227%, -0.017%, +0.009%, -0.101%; Spearman of
the 11 plan times between builds 0.82-0.99; the fastest plan is the same in both builds
for all six layers.

Split by plan shape: the all-4T plans (n=10) are 10 of 10 negative with median -0.113%,
the plans with wide lanes (n=56) median +0.007% and 27 of 56 negative. The direction
matches the prototype's small-M advantage, but the effect on whole plans is inside the
session spread.

Frozen rules: rule 1 (equivalence, |median| <= 0.6% and no point worse than 2%) passes;
rule 2 (agreement with the prototype's "no change") is consistent; rule 3 (ranking) holds
- no winner changed. Rule 4's cross-check shows the control reproduces this morning's
old-kernel measurement of the same plans to 0.19% (median).

## Correctness

Within each session the benchmark's own gate holds: every plan of a workload is bitwise
equal on both weight copies, in all twelve sessions. Across the builds,
`cross_build_equality.py` hashed the output of all 66 plans under each package tree with
the same inputs and packed weights: all 66 sha256 digests are equal, so the window order
does not alter the result. The prototype's checksums agree cell by cell as well.

## Limitations

- The V3 window table and the v11 event-model window terms were calibrated under the old
  order. E9 shows the chosen windows still rank plans the same, but the table's own
  optimum per (M, width) has not been re-measured under thread-major.
- The prototype's small-M advantage is a per-expert effect; on whole layers, where large
  experts dominate the makespan, it does not surface above the session spread.
- E9 uses one machine (Arm-codex, SVE-256, n_tile 16) and TP4 production shapes.

## Artifacts

- Prototype: `tmp/window_mapping_20260920/` (`window_mapping_native.cpp`, `run_grid.py`,
  `grid.json`, `grid_small_m.json`).
- E9: `tmp/window_mapping_20260920/e9/` (`design.md`, `chain_e9.sh`, `analyze_e9.py`,
  `cross_build_equality.py`, 12 session JSON/log files, `builds.md5`, job state).
- Minimal patch and control build on the Arm machine:
  `tmp/window_mapping_20260920/minimal_patch/`.
