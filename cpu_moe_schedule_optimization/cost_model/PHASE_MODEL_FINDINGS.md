# Phase-based contention model — calibration findings

**Kernel:** `fused_moe_bf16_tiled_async`, SHA `46228bb`. **Target:** AWS 8c Neoverse-V1.
**Shape:** H=4096, F=512, top_k=1/skip_weighted.

## Model

Predict the makespan of a concurrent expert group `[(routes_i, threads_i)]` from
two calibrated tables:

- `T_iso(routes, threads)` — isolated per-expert time (block-2 `isolated`).
- `derate(n)` — homogeneous slowdown at n distinct concurrent experts (block-2
  `entries`, averaged over shapes: `{2:1.11, 3:1.21, 4:1.26, 8:~1.35}`).

**Scalar baseline** (schema's implicit model): `max_i T_iso_i * derate(K)` — assumes
all K teams contend for the whole duration.

**Phase model** (`phase_model.py`): event-driven. While n teams active, each drains
at rate `1/derate(n)`; as teams finish, n drops and survivors speed up. Captures the
time-varying contention the scalar model misses.

## Out-of-sample validation (heterogeneous, mixed routes per team)

8 configs never seen in calibration (`validate_phase_model.py`):

| model  | median \|err\| | max \|err\| |
| ------ | -------------- | ----------- |
| phase  | **1.9%**       | **3.0%**    |
| scalar | 10.9%          | 18.9%       |

The scalar model over-predicts (up to +19%) because small teams actually finish
early and stop contending; the phase model tracks this and lands within ~3%.

## Overfitting assessment

- `derate(n)` is fit **only** from homogeneous block-2; validation is
  **heterogeneous** (a dimension absent from calibration) → genuine out-of-sample.
- The phase model has **no free parameters** tuned to the validation set → the
  ~2% accuracy is generalization, not fitting.
- Errors are a **small consistent −2% bias** (not scatter), i.e. the model is
  well-specified but a hair optimistic. Not corrected: a global fudge tuned to 8
  points would itself be overfitting; 3% is well within scheduling needs.
- `derate(n)` kept parameter-light (function of active-count only). The block-2
  second-order effect (uneven shapes contend slightly less) washes out acceptably
  in the phase sim — keying derate on thread-distribution was **not** adopted, to
  avoid overfitting the sparse (6-shape) data.

## Interval-DAG predictor (deps + staggered starts)

`dag_makespan(tasks)` extends the phase model to a full interval-DAG: event-driven
over completions, `#active` sets the rate `1/derate(#active)`, a completion frees a
successor which starts at full work. This exercises dependencies and staggered
starts that the derate table (independent concurrent tasks) never covered.

Out-of-sample validation on 5 hand-built DAGs (`validate_dag_makespan.py`):

| plan                     | err   |
| ------------------------ | ----- |
| concurrent 2×big         | +1.0% |
| seq chain full8 (deps)   | +0.4% |
| two waves 4+4            | +0.5% |
| stagger big+smalls       | +3.2% |
| hotspot-like             | +0.8% |

**DAG |err|: median 0.8%, max 3.2%.** Pure-dependency chains (derate=1) and pure
concurrency (derate(2)) both hit their expected values, confirming the mechanics.
Largest error is the staggered/partial-occupancy case (derate is calibrated at
full occupancy; ramp phases with idle cores are slightly over-derated) — still
within 3%.

## Update (2026-07-03): decode small-M + route-dependent derate

Block 2 re-measured with small-M routes {4,8,16,32} added (was {64,512,2048}), so
`T_iso` for decode is measured, not extrapolated. Finding: derate is a plateau for
routes>=16 but **drops toward ~1.0 at routes<=8** (fixed per-call overhead dominates
tiny-R makespan; little sustained bandwidth to contend). E.g. derate(8,·) =
{R8:1.18, R16:1.28, R64:1.27, R512:1.29, R2048:1.35}. Tiny-R points are also noisy
(sub-ms), so `derate(n,R)` uses the **median** over shapes.

The model now keys derate on **(n, routes)**, and each phase uses the **max routes
among active tasks** (a large active task drives steady-state contention; an
all-tiny/decode phase gets the low derate). Refreshed validation:

| check                     | median | max   |
| ------------------------- | ------ | ----- |
| phase (large-M hetero)    | 1.0%   | 3.5%  |
| interval-DAG (5 plans)    | 1.6%   | 3.9%  |

Decode absolute prediction improved from +16.6% to **+7.8%** (residual is fixed
per-call overhead the T_iso model doesn't fully capture; ~0.15ms on a 2ms decode).
Planner regime choice remains correct in all cases. The numbers in the sections
above are the earlier {64,512,2048}-only calibration and are superseded by this.

## Status



Block 1 (`T_iso`) + block 2 (`derate`) + phase model = complete cost kernel for the
static planner. Re-run `validate_phase_model.py` after any kernel change (the tables
are SHA-tagged and only valid for the matching kernel).
