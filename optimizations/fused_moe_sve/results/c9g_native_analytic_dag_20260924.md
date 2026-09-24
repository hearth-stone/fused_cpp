# Native analytic DAG simulator (C9g, 2026-09-24)

The analytic cost model's event simulator (`AnalyticMoeCostModel._dag_result`) ran only in
Python. Every full-search candidate, tail-pool candidate, long/short partition candidate and
early-merge decision paid for it. A C++ port now carries both forms:

- **placement-aware** (`dag_makespan_placed`): `_active_phase_state_placed`, including the
  `_normalize_placed_tasks` validation, LLC-domain spill and service, the rank LLC cap, DRAM
  domain injection, and the wide- and narrow-team terms;
- **rank-aggregate** (`dag_makespan`, `dag_task_finish_times`): `_active_phase_state`.

The model builds the native object lazily from the extension (`NativeAnalyticPlacedDag` in
`_moe_C`/`_C`, internal binding like `NativeQuickPlanner`). Without the extension, the
Python path runs unchanged; `explain_dag*` stays Python. Formulas are unchanged.

**How the port stays exact.** Python computes each phase's fields and its `base_ns`. Every
calibration curve is tabulated per integer thread count. Every shared-resource sum runs over
active tasks in ascending id, the order the Python reference uses. The only precomputed
values (per-phase offered rates and the LLC service denominator) are the same expressions.

**Multithreading.** One simulation is a sequential event loop, and its per-event sums must
keep Python's order to stay exact, so the loop is neither split nor SIMD-reduced. Threads
work across simulations instead: `dag_makespans_placed(batch, workers=...)` scores
independent DAGs with OpenMP.

## Correctness

- `tests/test_moe_native_placed_dag.py`: random lane/chain DAGs on a two-domain calibration
  with every contention term binding. Native and Python agree to `rel=1e-12` on placed and
  aggregate makespans and on finish times, with and without LLC domains. It also checks the
  batch against single scores and the validation errors (overlap, duplicate CPU, forward
  dependency, CPU outside the rank, duplicated dependency).
- 135 real plans (27 measured workloads x 2/4/8/16/32T homogeneous LPT, C9g admitted
  calibration): maximum relative difference 3.8e-15.
- C9g: `tests/test_moe_*.py` + `test_properties.py`: 2571 passed, 195 skipped.
- Selected plans identical with native and Python scoring on 25/25 planner runs: 9 catalog
  presets plus 2 real layers, quick with and without the partition candidate, and full on 3
  cases.

## Performance (C9g node 0, idle)

One placed score, 229-task plans, median over the 27 workloads (`time_native_dag.py`):

| plan width | Python | native | speedup |
| --- | --- | --- | --- |
| 2T (48 lanes) | 182.9 ms | 0.65 ms | 280x |
| 4T | 125.0 ms | 0.46 ms | 271x |
| 8T | 75.7 ms | 0.32 ms | 240x |
| 16T | 58.8 ms | 0.28 ms | 210x |
| 32T | 51.0 ms | 0.29 ms | 174x |

Batch of all 135 plans: 0.34 ms/plan on 1 worker, 0.16 on 4, 0.12 on 16. The serial Python
argument preparation and pybind conversion (about 0.03 ms plus conversion per plan) bound it.

Cold planning time, same plan either way (`plan_time_native.py`, median of three for native):

| case | Python | native |
| --- | --- | --- |
| bimodal, quick + long/short partition | 63.7 ms | 5.7 ms |
| bimodal, full | 1706 ms | 112 ms |
| real layer r008_l9, full | 11514 ms | 256 ms |
| active-set-32, full | 131.5 ms | 20.4 ms |
| quick without partition (all cases) | unchanged | unchanged (quick runs no DAG) |

After the port, full search's remaining time is Python LPT assignment (`_assign_lpt` calling
`_task_time` and the window-table `time_scale` about 1.3e5 times per plan), not the simulator.
