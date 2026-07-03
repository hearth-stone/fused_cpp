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

## Status

Block 1 (`T_iso`) + block 2 (`derate`) + phase model = complete cost kernel for the
static planner. Re-run `validate_phase_model.py` after any kernel change (the tables
are SHA-tagged and only valid for the matching kernel).
