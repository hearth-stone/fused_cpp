# Offline isolated CP-SAT oracle

## Purpose

`isolated_cp_sat_oracle.py` is an offline global-search oracle for the
no-contention expert-compute problem. It answers:

- what is the best makespan allowed by the supplied isolated-time model and
  thread-width domain;
- whether that optimum has been proved;
- when proof times out, what feasible upper bound and solver lower bound remain;
- how far a current strict planner schedule is from that isolated optimum.

It is not imported by the production planner and does not change runtime
dispatch.

## Model

Each active expert `i` is a non-preemptive moldable job. For every allowed
thread width `t`, the fixed duration is:

```text
duration(i, t) = T_iso(routes_i, t)
```

The solver selects exactly one mode per expert, assigns a start time, limits
the sum of concurrent thread demands to the rank-local core count, and
minimizes the final completion time. Expert interference, cache contention,
DRAM contention, core-interval contiguity, dynamic resizing, merge, and
communication are intentionally omitted.

A mode `(t2, d2)` is removed before solving when another mode `(t1, d1)`
satisfies `t1 <= t2` and `d1 <= d2`. Such a mode can never improve a cumulative
makespan schedule.

Durations are converted to integer CP-SAT ticks. The CLI defaults to a 1000 ns
tick because it materially improves proof time on large active sets. The
result reports `quantization_error_bound_ns = active_experts * tick / 2`, a
conservative critical-path error bound for nearest-tick rounding. Use
`--time-quantum-ns 1` when maximum model fidelity matters more than solve time.

## Install

OR-Tools is an optional dependency:

```bash
uv sync --extra oracle
```

Normal package installation and production MoE execution do not require it.

## Run

The following example uses the current 8-core exact-M `R13=2,R2=1` profile:

```bash
.venv/bin/python \
  cpu_moe_schedule_optimization/planners/isolated_cp_sat_oracle.py \
  cpu_moe_schedule_optimization/cost_model/profiles/\
contention_async_amazon_ecs_8c_standalone_sve_F512_E8_splitw13_\
xbyak_exactm_v2_20260720.json \
  --routes 2040,2040,768,768,192,48,12,4 \
  --num-cores 8 \
  --workers 8 \
  --max-time-s 60 \
  --output /tmp/moe-isolated-oracle.json
```

`--routes` is a complete route histogram; zero entries are accepted and
ignored. `--widths` defaults to the union of widths in the profile's supported
shapes. Passing an explicit list changes the oracle's feasible mode set and
must therefore be recorded with every comparison.

The CLI also runs the existing strict `IntervalPlanner`, evaluates its fixed
DAG with the same quantized `T_iso`, and reports the comparison. Dynamic tail
pool execution is outside the v1 oracle.

## Interpret the result

For minimization, let:

- `LB = oracle.best_bound_ns`;
- `UB = oracle.objective_ns`, the best feasible solution found;
- `C = planner.isolated_makespan_ns`;
- `C*` be the unknown exact optimum of the quantized isolated model.

Within that quantized model, the solver guarantees:

```text
LB <= C* <= UB
```

The current planner's isolated regret is therefore bounded by:

```text
max(0, C / UB - 1) <= regret <= C / LB - 1
```

The JSON fields are:

- `oracle.status`: OR-Tools status;
- `oracle.optimal`: true only when objective and best bound coincide, even if a
  nonzero `--relative-gap-limit` was configured;
- `oracle.relative_gap`: `(UB - LB) / UB`, the conventional solver gap;
- `oracle.bound_gap`: `(UB - LB) / LB`, the unresolved time-bound interval;
- `comparison.exact_regret`: available only after an exact proof;
- `comparison.regret_lower_bound` and `regret_upper_bound`: valid before proof;
- `comparison.bound_efficiency`: `LB / C`.

When `T_iso` is a measured or fitted median, these are exact statements about
that cost-model surrogate, not formal hardware bounds. To make
`best_bound_ns` a physical lower bound, every duration must itself be a
conservative hardware lower bound and the quantization direction must preserve
that bound.

## Initial validation

Synthetic unit tests cover mode dominance, variable-width global optimality,
cumulative capacity, dependency-DAG scoring, cycle rejection, and a real
profile CLI invocation.

Two local solver smoke points use committed profiles and do not execute the
MoE kernel:

| Profile workload | Limit | Result |
|---|---:|---:|
| 8 active mixed routes, 8 cores | 10 s, 8 workers | proved `49.786 ms`; strict planner `51.921 ms`, exact isolated regret `4.29%` |
| 256 experts at route 48, 96 cores | 10 s, 8 workers | feasible `5.827 ms`, lower bound `5.486 ms`, solver relative gap `5.85%` |

Solver wall time depends on the machine running OR-Tools. The modeled
makespans come from the selected calibration profile.

The 96-core-rank validation on `AmazonC5192Cores`, including all nine default
paper/captured workloads and concurrent dual-NUMA runtime measurements, is
recorded in
[`amazon_192c_isolated_cp_sat_oracle_20260726.md`](../../optimizations/fused_moe_sve/results/amazon_192c_isolated_cp_sat_oracle_20260726.md).
The current bimodal tail-pool plan reaches the exact isolated optimum, while
high-active-set wall time remains far above the no-contention bound.

## V1 limitations

- The oracle only covers whole-expert, fixed-width, non-preemptive jobs.
- CPU cores are fungible capacity; production contiguous interval placement
  and NUMA affinity are relaxed.
- W13/W2 stage windows affect the result only through the supplied `T_iso`.
- The model permits aggregate compute or bandwidth demand beyond hardware
  saturation because contention is deliberately absent.
- A low isolated bound efficiency does not by itself prove that the planner is
  poor; the missing gap can be unavoidable shared-resource contention.
- Comparing kernel quality requires a separate per-mode hardware lower-bound
  duration, not the current implementation's measured `T_iso`.
