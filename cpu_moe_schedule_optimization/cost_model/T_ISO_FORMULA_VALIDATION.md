# T_iso Formula Validation

This document records the active compact empirical formula. The new
kernel-work reconstruction is intentionally still a shadow model. The simple
aggregate baseline is in
[`TISO_ROOFLINE_VALIDATION.md`](./TISO_ROOFLINE_VALIDATION.md); the current
implementation-weak layered formulation and V3 held-out results are in
[`GEMM_ECM_VALIDATION.md`](./GEMM_ECM_VALIDATION.md).

## 2026-07-14 AWS 8-core SVE/M12

The schema-v2 isolated cost path now uses:

```text
T_iso(R,t) = O(t) + C(R) * phi_usl(t) * k_phi(t)
O(t)       = o0 + o1/t
phi_usl(t) = (1 + alpha(t-1) + beta*t(t-1)) / t
```

`C(R)` is calibrated only from the single-thread route curve. `k_phi(t)` is a
route-independent per-thread calibration of the generalized-USL baseline.
M1/M2/M4/M8 and the first two M12 panels retain measured per-thread residuals.
The old complete 2-D table remains available with `iso_mode="table"`.

### Target

- Host: `AmazonECS8Cores`, Neoverse V1, 8 cores, 32 MiB LLC.
- Kernel: SVE fused expert, `backend_n_tile=16`, M12 route kernel.
- Shape: `H=4096`, `F=512`, BF16, fused SiLU, exact `R13=2,R2=1`.
- Weights: four consecutive experts per isolated sample, so each call streams
  48 MiB of distinct packed expert weights and exceeds LLC.
- Sampling: 5 warmups and 20 timed runs, CPUs `0-7`.

### Sparse fit

Calibration routes were:

```text
1, 2, 4, 8, 12, 24, 48, 192, 768, 2040
```

The following routes were completely excluded from fitting:

```text
36, 72, 96, 144, 288, 384, 576, 1020, 1536
```

All sets cover threads `1,2,4,8`. The sparse fit produced:

```text
O(t) = 0.092373 + 0.110997/t ms
alpha = 0.2126748
beta  = -0.01893526
```

The negative `beta` captures cache-assisted scaling only on this measured
1-to-8-thread domain. It must not be extrapolated to more cores.

| Set | Points | Median abs error | P90 abs error | Max abs error | Median bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | 40 | 0.00% | 2.86% | 9.75% | +0.00% |
| Held-out routes | 36 | 1.40% | 5.31% | 6.70% | +0.69% |
| Sparse per-thread table baseline | 36 | 0.63% | 2.84% | 8.14% | -0.26% |

The table baseline is more accurate at the same route anchors, but it retains a
separate route curve for every thread count. The formula uses one route curve
plus one thread correction curve, while keeping held-out bulk error below 7%
on this target.
Tiny-M residuals are necessary: without them, different tail microkernels gave
about 17% worst-case error.

### Planner rollout guard

The formula was also fitted in shadow mode on the four existing 2026-07-13
TP2/EP2 profiles and evaluated on five representative workloads. Although
pointwise errors were small, the EP2 real-routing case changed both lane shape
and LPT assignment; the old table rescored that task DAG about 3.9% slower than
its own selection. This is above the current 2% planner-regret target.

Consequently, historical profiles without a serialized `iso_formula` remain in
table mode by default. New profiler output serializes the formula and therefore
opts into formula mode. Historical data can be tested with
`iso_mode="formula"` or `FUSED_CPP_COST_MODEL_ISO_MODE=formula` without changing
its default behavior.

### Artifacts

- Raw measurements:
  `profiles/tiso_formula_amazon_ecs_8c_sve_F512_splitw13_20260714.json`
- Sparse-fit predictions:
  `profiles/tiso_formula_validation_amazon_ecs_8c_sve_F512_splitw13_20260714.json`

Reproduce the fit report with:

```bash
.venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/validate_iso_formula.py \
  --profile cpu_moe_schedule_optimization/cost_model/profiles/tiso_formula_amazon_ecs_8c_sve_F512_splitw13_20260714.json \
  --phi-route-min 192
```
