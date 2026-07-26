# Analytical CPU MoE Cost Model

## Status

`analytic_model.py` is the planner-compatible analytical backend for the SVE
BF16 fused expert. It is opt-in while machine calibrations are validated. The
schema-v2 empirical model remains the production default and the holdout oracle;
it is not an input to analytical prediction.

The model targets one NUMA-local rank, cold distinct-expert weights, N-split
teams, and the current packed-A Xbyak/static SVE kernel mapping. Communication,
router cost, and final TopK merge remain outside this model.

## Separation Of Concerns

The prediction has three layers:

1. `gemm_cost_model.py` emits logical W13 and W2 work.
2. `sve_bf16_kernel_model.py` lowers work to exact physical panels, instructions,
   executed FLOPs, N ownership, cache traffic, and epilogue elements.
3. `analytic_model.py` maps physical demand through machine service curves,
   cache capacities, runtime overheads, and a shared-resource event simulator.

No route/thread latency table or measured contention shape is loaded.

## Machine Service Curves

Each resource uses three route-independent values: one-thread rate
\(R_1\), saturated aggregate rate \(R_{\mathrm{sat}}\), and the thread count
\(t_{\mathrm{sat}}\) where saturation is reached. A `power` curve models
per-core issue/frequency derate:

\[
\alpha_R =
\frac{\log(R_{\mathrm{sat}}/R_1)}{\log(t_{\mathrm{sat}})},
\qquad
R(t)=\min(R_{\mathrm{sat}},R_1t^{\alpha_R}).
\]

For cache/DRAM fabrics that initially scale almost linearly and then meet a
shared bottleneck, `shared_bottleneck` uses:

\[
\beta_R =
\frac{R_1t_{\mathrm{sat}}/R_{\mathrm{sat}}-1}
{t_{\mathrm{sat}}-1},
\qquad
R(t)=
\min\left(
R_{\mathrm{sat}},
\frac{R_1t}{1+\beta_R(t-1)}
\right).
\]

Both curves use the same three measurements. The curve family is a hardware
topology statement, not another route-dependent fit.

The required resources are matrix FLOP/s and L1, L2, LLC, and DRAM byte/s.
Frontend instruction/s and fused-epilogue element/s are optional. A missing
optional resource has an infinite ceiling and therefore cannot become the
predicted bottleneck.

The anchors must come from dedicated microbenchmarks using the same ISA,
frequency policy, NUMA placement, and concurrent-rank mode as production. A
multi-thread GEMM latency table is not a service curve.

## Cache Traffic

For range \(j\) of one stage, let:

- \(P\) be the number of physical M panels;
- \(B_j\) be the range's packed-weight bytes;
- \(A\) be all physical packed-A bytes;
- \(A_p\) be the largest M-panel packed-A bytes;
- \(U_j\) be one N owner's packed-B window;
- \(t_j\) be the range's active owners;
- \(t_1\) be the first range's active owners. Ranges are non-increasing in size,
  so later owners are a subset of the first range's owners.

The calibrated effective L2 capacity and physical L2 capacity define a smooth
capacity-transition function \(h_2(W)\). It is zero below the effective
capacity, one above physical capacity, and smoothstep-interpolated in the
uncertain replacement band. The aggregate LLC-to-L2 traffic is:

\[
Q_{B,L2}=\sum_j B_j\left[1+(P-1)h_2(U_j+A_p)\right],
\]

\[
Q_{A,L2}=A\left[t_1+\sum_{j>1}t_jh_2(U_j+A)\right].
\]

This expresses the kernel loop directly:

- the first B pass is cold and reads each weight once;
- later M panels reuse the owner stripe if it fits private L2;
- every N owner reads packed A once;
- later sequential N ranges reuse A if it remains in private L2.

Packed A and the W13 intermediate were just produced by the operator, so they
enter the GEMM through the cache hierarchy rather than being compulsory DRAM
reads. For distinct experts, the first B pass is compulsory DRAM:

\[
Q_{\mathrm{DRAM}}^{\mathrm{comp}}=\sum_j B_j.
\]

Repeated B refills, A refills, and C writeback are spillable traffic. Their
DRAM fraction is \(h_3(\sum_i W_i)\), where \(h_3\) uses the effective and
physical LLC capacities and the sum covers concurrently active phases.

The reusable LLC footprint includes the active B window only when \(P>1\).
For \(P=1\), B is still compulsory DRAM/stream traffic but has no future panel
reuse to protect. Packed A and C remain in the footprint because another owner
or the next stage can consume them. This prevents M<=12 experts from consuming
the reusable-weight budget while retaining their bandwidth pressure.

## Isolated Time

For resource \(r\), physical demand is \(q_r\) and calibrated rate is \(R_r(t)\).
The stage uses an ECM-like non-overlap path:

\[
T_{\mathrm{xfer}} =
\frac{Q_{L1}}{B_{L1}(t)}
+\frac{Q_{L2}}{B_{L2}(t)}
+\frac{Q_{LLC}}{B_{LLC}(t)}
+\frac{Q_{\mathrm{DRAM}}}{B_{\mathrm{DRAM}}(t)},
\]

\[
T_{\mathrm{body}} =
\max\left(
\frac{F^{\mathrm{bal}}}{P_{\mathrm{matrix}}(t)},
\frac{I^{\mathrm{bal}}}{R_{\mathrm{frontend}}(t)},
T_{\mathrm{xfer}}
\right),
\]

\[
T_s =
\gamma_s\left(
O_{\mathrm{stage}}+N_{\mathrm{range}}O_{\mathrm{range}}
+T_{\mathrm{body}}
+\frac{E^{\mathrm{bal}}}{R_{\mathrm{epilogue}}(t)}
\right).
\]

Each range charges
\(\lceil n_j/t\rceil\min(n_j,t)\) balanced N tiles. This equals the busiest
lane work times that range's actual active owners, so a final range with fewer
tiles does not pay for inactive threads.
\(\gamma_{13}\) and \(\gamma_2\) are optional stage residual scales. They are the
only stage-specific corrections and should stay near one; a large correction
means a missing resource or incorrect traffic mapping.

Expert fixed and per-route runtime costs cover gather/dispatch/scatter work not
yet represented by a dedicated physical mapper. They are deliberately separate
from GEMM demand.

When a deterministic Plan V2 stage-window policy is bound, each candidate
resolves its task's W13/W2 targets from `(routes, actual_threads)` before
calling this mapper. The resulting tile-aligned range geometry replaces the
global geometry in both isolated and concurrent calculations. Window targets
are therefore execution parameters of an existing shape candidate, not an
additional search dimension.

## Concurrent Time

The DAG simulator advances W13/W2 range phases. For every active phase \(i\),
its provisional duration gives requested rate:

\[
\lambda_r=\sum_i\frac{q_{i,r}}{T_i}.
\]

At total active width \(c\), resource pressure is:

\[
d_r=\max\left(1,\frac{\lambda_r}{R_r(c)}\right).
\]

The simulator applies \(d_r\) only to the matching matrix, frontend, cache,
DRAM, or epilogue component, then advances to the next phase-completion event.
This distinguishes two kernels with equal isolated time but different resource
vectors. Fixed overhead is not derated as memory traffic.

## Calibration Schema

The machine JSON is independent of route count, expert count, H/F, and planner
shape. Rates use units per second; overheads use nanoseconds.

```json
{
  "schema_version": 1,
  "kind": "moe_analytic_machine",
  "machine": {
    "id": "machine-kernel-frequency-numa-policy",
    "cores_per_rank": 96
  },
  "caches": {
    "l1d_bytes_per_core": 65536,
    "l2_bytes_per_core": 2097152,
    "llc_bytes_per_rank": 100663296,
    "l2_effective_fraction": 0.75,
    "llc_effective_fraction": 0.75
  },
  "services": {
    "matrix_flops": {
      "single_thread_rate": 4.0e11,
      "saturated_rate": 1.0e13,
      "saturation_threads": 48,
      "curve": "power"
    },
    "l1_bytes": {
      "single_thread_rate": 1.0e11,
      "saturated_rate": 9.6e12,
      "saturation_threads": 96,
      "curve": "shared_bottleneck"
    },
    "l2_bytes": {
      "single_thread_rate": 5.0e10,
      "saturated_rate": 4.8e12,
      "saturation_threads": 96,
      "curve": "shared_bottleneck"
    },
    "llc_bytes": {
      "single_thread_rate": 2.5e10,
      "saturated_rate": 1.0e12,
      "saturation_threads": 48,
      "curve": "shared_bottleneck"
    },
    "dram_bytes": {
      "single_thread_rate": 4.0e10,
      "saturated_rate": 3.8e11,
      "saturation_threads": 48,
      "curve": "shared_bottleneck"
    }
  },
  "overheads": {
    "call_setup_ns": 0.0,
    "expert_fixed_ns": 0.0,
    "route_ns": 0.0,
    "stage_fixed_ns": 0.0,
    "range_fixed_ns": 0.0
  },
  "planner": {
    "supported_widths": [1, 2, 4, 8, 16, 32, 48, 64, 96]
  },
  "uncertainty": {
    "relative": 0.05
  },
  "stage_scales": {
    "w13": 1.0,
    "w2": 1.0
  }
}
```

The numbers above illustrate the schema and are not a production calibration.

## Planner Use

```python
from analytic_model import AnalyticMoeCostModel
from planned_moe import PlannedMoE

model = AnalyticMoeCostModel(
    "machine.json",
    hidden_size=4096,
    intermediate_size=512,
    global_experts=256,
    local_experts=256,
    mode="tp",
    degree=4,
    concurrent_ranks=2,
    w13_split=True,
    w13_split_chunks=2,
)
planner = PlannedMoE(model, num_cores=96)
plan = planner.plan_spec_for(route_counts)
```

Unlike schema-v2 profiles, the analytical model generates homogeneous and
two-width shapes from `supported_widths`; it is not restricted to measured
shapes. Existing active-working-set pruning still applies.

## Holdout Validation

Run:

```bash
python cpu_moe_schedule_optimization/cost_model/validate_analytic_model.py \
  machine.json empirical_schema_v2_profile.json --output validation.json
```

The report includes:

- evaluated/skipped coverage for the calibration's supported thread widths;
- isolated MAPE and tail error;
- full-call contention MAPE and tail error;
- measured regret of the shape selected by analytical prediction.

Before changing the production default, validate unseen routes, widths, mixed
route distributions, split/window policies, and both isolated and concurrent
execution. Initial acceptance gates are isolated MAPE at most 10%, contention
P90 absolute error at most 15%, and maximum measured shape regret at most 5%.
Any failed gate keeps the empirical model as the runtime default.

## Known Limits

- SVE BF16 N-split only; a different ISA or M/MN split needs a new kernel mapper.
- Cold distinct-expert weights are assumed. Reusing the same expert across calls
  needs an explicit warm-weight state.
- The cache-capacity transition is an effective-capacity approximation, not a
  set-level cache simulator.
- One-pass B streams are excluded from reusable capacity. Their small persistent
  LLC pollution is not modeled separately from stream bandwidth.
- Gather, router, final TopK merge, communication, and cross-NUMA traffic are not
  yet separate demand vectors.
- Frequency/power effects must be reflected in the measured aggregate service
  curves.
- Dynamic in-task resize is not modeled; phases use fixed thread widths.
