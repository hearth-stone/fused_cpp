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
3. `analytic_model.py` partitions each full-N stage among the selected team,
   splits it into setup, cold-B, and steady-B phases, then maps physical demand
   through machine service curves, cache capacities, runtime overheads, and a
   shared-resource event simulator.

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

The primary compute resource is `gemm_core_flops`: the production M12 A/B load,
address/control, and BFMMLA K-loop with no store or epilogue, measured with A+B
resident in L1D. It jointly captures matrix issue, frontend issue, and
L1-to-register delivery. A register-only `matrix_flops` curve, derived frontend
rate, and B-only L1 rate remain diagnostic and are never charged in addition to
the core service. `gemm_core_flops` is required: a calibration without the
attainable L1-hot peak is rejected rather than falling back to register-only
throughput.

L2, LLC, and DRAM remain endpoint-to-register byte/s resources. A B-only probe
for a deeper endpoint already includes its downstream path: an LLC observation
includes LLC->L2->L1, and a DRAM observation includes the whole hierarchy. The
corresponding time lower bounds therefore compose with `max`; adding them would
charge the same downstream service more than once. Fused-epilogue element/s is
optional. A missing optional resource has an infinite ceiling and cannot become
the predicted bottleneck.

The anchors must come from dedicated microbenchmarks using the same ISA,
frequency policy, NUMA placement, and concurrent-rank mode as production. The
one-thread DRAM/LLC-refill anchor must use a cold streaming request pattern with
the kernel's MLP, rather than STREAM peak bandwidth. The saturated anchor is
the sustainable service knee before queueing causes persistent stall growth,
not a one-sample peak. A multi-thread GEMM latency table is not a service curve.

Probe dimensions are derived from the target CPU's Linux sysfs cache hierarchy,
not from machine-specific constants. For detected cache capacity (C_L), SVE
N tile ν, the minimum physical W13 width (N_p=2ν), and a reserved fraction
(f_L), the M12 cache-hot probe uses:

\[
K_L=8\left\lfloor\frac{f_LC_L}{16(12+N_p)}\right\rfloor,
\qquad
Q_L=2K_L(12+N_p)\le f_LC_L.
\]

`profile_analytic_services.py` reads L1D, L2, LLC, and cache-line sizes from
`/sys/devices/system/cpu/cpuX/cache/index*`. Its default fractions are 0.625 for
the L1 GEMM core probe and 0.5 for the L2-hot diagnostic. On
AmazonC5192Cores this gives M12/K728/N16 (40,768 B) from a 64 KiB L1D and
M12/K18720/N16 (1,048,320 B) from a 2 MiB L2.

## Cache Traffic

For one full-N stage, let:

- \(P\) be the number of physical M panels;
- \(B\) be the full stage's packed-weight bytes;
- \(A\) be all physical packed-A bytes;
- \(A_p\) be the largest M-panel packed-A bytes;
- \(U\) be the busiest N owner's tile-aligned packed-B stripe;
- \(t_a=\min(t,q_s)\) be the active owners, where \(q_s=N_s/\nu\).

The calibrated packed-B retention function \(g_2(W)\) uses a dedicated
effective capacity \(C^B_{2,\mathrm{eff}}=\rho^B_2C_2\) plus three measured miss
anchors: the sub-knee repeated-scan miss floor, miss at nominal L2 capacity, and
miss at twice nominal capacity. Smoothstep interpolation is used from
\(C^B_{2,\mathrm{eff}}\) to nominal capacity and from nominal to twice nominal
capacity. This curve describes the transient repeated scans observed before a
full effective L2 of distinct A panels has passed the owner.

Define the number of transient B reuses and the physical steady-scan state as:

\[
L_A=\max\left(\left\lfloor\frac{C^A_{2,\mathrm{eff}}}{A_p}\right\rfloor,1\right),
\qquad
q=\min(P-1,L_A),
\qquad
r_B(W)=\mathbf 1[W>C_2].
\]

The packed-B and packed-A effective-capacity fractions remain independent: a
repeated B stripe competes with A, stores, and prefetch state and reaches its
transient retention knee before a generic L2 capacity boundary. The aggregate
LLC-to-L2 traffic is:

\[
Q_{B,L2}=B\left[
1+qg_2(U+A_p)+(P-1-q)r_B(U+A_p)
\right],
\]

\[
Q_{A,L2}=At_a.
\]

This expresses the kernel loop directly:

- the first B pass is cold and reads each weight once;
- the first \(q\) B reuses use the calibrated transient retention curve;
- after A has turned over one effective L2, a B stripe that fits physical L2 is
  resident, while an over-capacity stripe remains streaming;
- every active N owner scans packed A once;
- there is no second sequential N range and therefore no planner-controlled A
  rescan multiplier.

The finite transient is important for long routes. Applying one short-route
miss probability to all \(P-1\) panels makes a small residual miss grow without
bound and systematically favors undersized windows. Here the transition length
comes from cache and panel geometry, not a route threshold or fitted timing
table.

Packed A and the W13 intermediate were just produced by the operator, so they
enter the GEMM through the cache hierarchy rather than being compulsory DRAM
reads. For distinct experts, the first B pass is compulsory DRAM:

\[
Q_{\mathrm{DRAM}}^{\mathrm{comp}}=B.
\]

Repeated B refills, A refills, and C writeback are spillable traffic. Their
DRAM fraction is \(h_3(\sum_i W_i)\), where \(h_3\) uses the effective and
physical LLC capacities and the sum covers concurrently active phases.

The reusable LLC footprint includes the active B window only when \(P>1\).
For \(P=1\), B is still compulsory DRAM/stream traffic but has no future panel
reuse to protect. Packed A and C remain in the footprint because another owner
or the next stage can consume them. This prevents M<=12 experts from consuming
the reusable-weight budget while retaining their bandwidth pressure.

## Physical Phases

Each full-N stage is lowered into the phases that actually change its resource
signature:

1. `stage_setup` contains one stage-dispatch cost and has no GEMM traffic;
2. `cold_b` executes the first physical M panel and owns the full stage's
   compulsory packed-B DRAM scan;
3. `steady_b`, when more M panels exist, executes all remaining panels with no
   compulsory B traffic. It contains repeated B refills, remaining A scans,
   and output traffic.

The mapper derives matrix FLOPs, frontend instructions, L1 loads, and epilogue
elements separately for the first panel and the remaining panels. A/cache and
C traffic are split by physical compute/store rows. For every stage, the phase
demands sum exactly to the original physical kernel demand. M<=12 therefore has
only `cold_b`, while a long route exposes a short cold phase followed by a
compute/cache-reuse steady phase.

## Isolated Time

For resource \(r\), physical demand is \(q_r\) and calibrated rate is \(R_r(t)\).
With `gemm_core_flops`, each cold or steady phase uses:

\[
T_{\mathrm{core}} =
\frac{F^{\mathrm{bal}}}{P_{\mathrm{gemm\_core}}(t)},
\]

\[
T_{\mathrm{xfer}} =
\max\left(
\frac{Q_{L2}}{B_{L2}(t)},
\frac{Q_{LLC}}{B_{LLC}(t)},
\frac{Q_{\mathrm{DRAM}}}{B_{\mathrm{DRAM}}(t)}
\right),
\]

\[
T_{\mathrm{body}} =
\max\left(
T_{\mathrm{core}},
T_{\mathrm{xfer}}
\right),
\]

\[
T_s=T_{\mathrm{setup}}+T_{\mathrm{cold}}+T_{\mathrm{steady}}.
\]

`steady` is absent for a one-panel stage and the stage residual \(\gamma_s\)
is applied to each phase. Separating cold and steady service before taking the
ECM maximum is required:
a cold panel can be DRAM/refill limited while the remaining panels are GEMM-core
or private-cache limited. `gemm_core_flops` is mandatory; register-only
matrix/frontend/L1 observations cannot replace this production-loop peak.

The full stage charges \(\lceil q_s/t\rceil\min(q_s,t)\) balanced N tiles. This
equals the busiest lane work times the stage's actual active owners, so widths
larger than the tile count do not create fictitious GEMM work.
\(\gamma_{13}\) and \(\gamma_2\) are optional stage residual scales. They are the
only stage-specific corrections and should stay near one; a large correction
means a missing resource or incorrect traffic mapping.

Expert fixed and per-route runtime costs cover gather/dispatch/scatter work not
yet represented by a dedicated physical mapper. They are deliberately separate
from GEMM demand.

Plan V2 supplies only the selected task width. For stage \(s=(K_s,N_s)\), tile
width \(\nu\), and team width \(t\), the production mapper derives

\[
q_s=N_s/\nu,\qquad
u_s(t)=\left\lceil q_s/t\right\rceil K_s\nu\cdot2.
\]

The full stage bytes \(B_s=2K_sN_s\) are fixed; \(u_s(t)\) is the maximum
per-worker owner stripe. Thus team width is the only cache-window control and
no window decision is added to the planner or ABI.

### Historical Stage-Window Policy (Retired)

The remainder of this subsection records the superseded range-policy model for
experiment provenance only. Production code, Plan V2, and the active analytical
model no longer enumerate or execute these ranges; the full-stage equations
above are normative.

`AnalyticStageWindowPolicy` generates those execution parameters directly from
the machine model. For stage \(s\), route count \(M\), and already selected team
width \(t\), it enumerates the physically achievable tile-aligned range targets
\(g\in\mathcal G_s(t)\). A generated candidate must:

- leave at least one private L1D of owner work, because a smaller stripe has no
  lower cache level left to protect;
- contain at least \(\min(t,q_s)\) N tiles per range, so it does not create idle
  team members;
- use a power-of-two owner tile count, matching the kernel's natural halving
  hierarchy;
- retain the physically distinct W13 \(R=1/R=2\) endpoints and W2 \(R=1\)
  endpoint whenever the stage tile count permits them.

The inherited operator-wide geometry remains in the candidate set as a
compatibility endpoint. A one-panel expert (\(M\le12\)) selects \(R=1\) for both
stages: there is no repeated packed-B scan to protect, so subdividing the stage
cannot reduce B traffic and adds range control work.

The absolute ECM phase latency remains
\(\max(T_{\mathrm{core}},T_{\mathrm{xfer}})\). That lower-bound form is not a
useful selector when all transfer differences fit below the same compute
ceiling, because it assigns zero value to reducing refill demand and future
contention. Stage-window selection therefore uses the serialized incremental
objective

\[
J_s(M,t,g)=
\sum_{p\in\mathcal P_s(g)}
\left[
T_{\mathrm{core},p}+T_{\mathrm{epi},p}
+\max\left(T_{L2,p},T_{LLC,p},T_{DRAM,p}\right)
\right]
+R_s(g)\tau_{\mathrm{range}},
\]

Let \(J_{\min}\) be the minimum candidate objective and let
\(\epsilon_{\mathrm{rel}}\) be the calibration's measured relative uncertainty.
Sub-microsecond ordering inside

\[
\Delta_J=\epsilon_{\mathrm{rel}}T_{\mathrm{xfer}}(g_{\min})
\]

is not treated as physically distinguishable. The policy first forms

\[
\mathcal E_s=\{g:J_s(M,t,g)\le J_{\min}+\Delta_J\},
\]

then chooses the member nearest in powers-of-two to the kernel-native owner
window

\[
U_{s,\mathrm{pref}}=\max(C_{L1D},2b_s),
\]

where \(b_s=K_s\nu\cdot2\) is one packed-B tile. Exact objective and range count
break any remaining tie. Equivalently,

\[
g_s^*(M,t)=
\arg\min_{g\in\mathcal E_s}
\left(
\left|\log_2\frac{U_s(g)}{U_{s,\mathrm{pref}}}\right|,
J_s(M,t,g),R_s(g)
\right).
\]

W13 and W2 are minimized independently and lowered to the Plan V2 exact-range
ABI. The native mapper reconstructs the same tile-balanced geometry from the
integer range count; it does not round-trip through a byte target. The
uncertainty tie does not add a measured route band: it uses one
machine-level uncertainty scalar and the kernel/cache geometry already present
in the calibration. This objective is used only to choose an execution policy; absolute
expert and DAG time continues to use the overlapping ECM maximum and the
shared-resource event simulator. The policy is deterministic for a calibration
digest, does not add a planner variable, and does not change the planner's
shape set.

The packed N tile \(\nu\) is part of the machine/kernel calibration, not a
portable default. The model defaults to `kernel.backend_n_tile`, and the
holdout benchmark rejects a calibration when that value differs from the packed
weight ABI; any production binding must enforce the same check. This matters
across the two validation machines: the 192-core SVE build uses \(\nu=8\), while
the 8-core SVE build uses \(\nu=16\).

## Concurrent Time

The DAG simulator advances setup/cold/steady completion events. For phase \(i\),
let \(\tau_{i,r}\) be the calibrated isolated service occupancy of resource
\(r\), including the stage residual. Every active phase then requests:

\[
\lambda_r=\sum_i\frac{q_{i,r}}{\tau_{i,r}}.
\]

For each resource, only threads belonging to phases with non-zero demand count
as requesters:

\[
c_r=\min\left(C,\sum_i t_i\mathbf 1[q_{i,r}>0]\right).
\]

Resource utilization and dilation are:

\[
\rho_r=\frac{\lambda_r}{R_r(c_r)},
\qquad
d_r=\max(1,\rho_r).
\]

The simulator applies \(d_r\) only to the matching GEMM-core, L2, LLC, DRAM, or
epilogue component, then advances to the next phase-completion event. Legacy
calibrations retain separate matrix/frontend/L1 components.
It also records the allocated rate \(\lambda_r/d_r\), which is never greater
than the calibrated capacity. Fixed overhead is represented as a zero-demand
setup phase and is not derated as memory traffic.

This distinguishes two kernels with equal isolated time but different resource
vectors.

This construction does not need a pairwise expert slowdown matrix. A short
expert is almost entirely a cold-B phase, so reduced refill service directly
reduces its progress. A long expert spends most of its lifetime in steady
compute/cache phases, so the same overlap changes only part of its total time.
The resulting cross-slowdown is directional even though the calibrated shared
resource capacity is common.

`explain()` reports each phase's ECM components and isolated resource pressure.
`explain_dag()` additionally reports every completion interval, active phase,
working set, LLC spill fraction, offered rate, calibrated capacity,
utilization, dilation, allocated rate, and allocated utilization. These records
are diagnostics derived from the same simulation used by planner scoring.

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
  "kernel": {
    "backend_n_tile": 8
  },
  "caches": {
    "l1d_bytes_per_core": 65536,
    "l2_bytes_per_core": 2097152,
    "llc_bytes_per_rank": 100663296,
    "l2_effective_fraction": 0.75,
    "llc_effective_fraction": 0.75,
    "l2_b_reuse_effective_fraction": 0.125,
    "l2_b_reuse_miss_floor": 0.18,
    "l2_b_reuse_miss_at_capacity": 0.62,
    "l2_b_reuse_miss_ceiling": 0.87
  },
  "services": {
    "gemm_core_flops": {
      "single_thread_rate": 3.4e11,
      "saturated_rate": 2.4e13,
      "saturation_threads": 96,
      "curve": "power"
    },
    "matrix_flops": {
      "single_thread_rate": 4.1e11,
      "saturated_rate": 3.9e13,
      "saturation_threads": 96,
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
    "stage_fixed_ns": 0.0
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
`gemm_core_flops` is the active compute service. `matrix_flops` and `l1_bytes`
are retained in new files for diagnostics. Removing `gemm_core_flops` makes the
calibration invalid because register-only BFMMLA is not an attainable kernel
peak.

Generate the independent service probe and build a thin calibration with:

```bash
python cpu_moe_schedule_optimization/cost_model/profile_analytic_services.py \
  --output services.json --cpu-ids 0-95 \
  --widths 1,2,4,8,16,24,32,48,64,96

python cpu_moe_schedule_optimization/cost_model/build_analytic_calibration.py \
  services.json --output machine.json --report fit.json \
  --training-profile isolated_training.json \
  --backend-n-tile 8 \
  --l2-b-reuse-effective-fraction 0.125 \
  --l2-b-reuse-miss-floor 0.18 \
  --l2-b-reuse-miss-at-capacity 0.623 \
  --l2-b-reuse-miss-ceiling 0.869
```

The effective fraction and three retention values should come from an
independent packed-B repeated-scan probe. They must not be fitted from the
contention table. A transferred prior is allowed for a portability holdout only
when provenance marks it as non-local; it is not a completed machine
calibration.

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
)
planner = PlannedMoE(model, num_cores=96)
plan = planner.plan_spec_for(route_counts)
```

Unlike schema-v2 profiles, the analytical model generates homogeneous and
two-width shapes from `supported_widths`; it is not restricted to measured
shapes. Existing active-working-set pruning still applies.

## Holdout Validation

Validate full-stage isolated and contention predictions against a compatible
empirical profile with:

```bash
python cpu_moe_schedule_optimization/cost_model/validate_analytic_model.py \
  machine.json fulln_profile.json --output analytic_holdout.json
```

The holdout must use `kernel.stage_geometry=full_n_team_stripes`, the same
backend N tile, and the same machine/NUMA/kernel identity. It evaluates only
team-width choices; no internal W13/W2 range coordinate exists.

The table below is retained only as historical evidence from the removed
stage-window selector. Its regret values must not be used as a gate for the
current full-stage model:

The corrected policy-v2 2026-08-09 holdout produced:

| Host / statistic | Median regret | P90 | Maximum | <=2% | <=5% |
| :--- | ---: | ---: | ---: | ---: | ---: |
| AmazonC5192Cores NUMA0, median | 1.50% | 2.98% | 3.38% | 17/28 | 28/28 |
| AmazonECSV1 8C, raw median | 2.88% | 6.05% | 18.41% | 10/28 | 22/28 |
| AmazonECSV1 8C, p10 sensitivity | 2.01% | 2.61% | 4.21% | 14/28 | 28/28 |

Historically, on the clean 192-core host, v1's `1.63/6.57/11.32%`
median/P90/maximum
regret becomes `1.50/2.98/3.38%`. The correction changes cache traffic and
uncertainty handling only; routes are not added to calibration, and the legal
planner shape set is unchanged.

The 192-core samples are stable: candidate p90/p10 spread has a 1.46% median,
and policy v2 passes the 5% maximum-regret gate in both median and p10 analyses.
The 8-core host was heavily preempted: candidate p90/p10 spread had a 28.47%
median and 99.66% P90, so its raw maximum is not a valid strict gate; its p10
sensitivity remains below 5%. Its cache and service curves are local, but
packed-B retention is still a transferred prior. These numbers describe the
retired policy only. Full protocol and raw artifact links are in
`optimizations/fused_moe_sve/results/analytic_stage_window_policy_v2_holdout_20260809.md`.

Validate absolute analytical timing and planner shapes with:

```bash
python cpu_moe_schedule_optimization/cost_model/validate_analytic_model.py \
  machine.json empirical_schema_v2_profile.json --output validation.json
```

The report includes:

- evaluated/skipped coverage for the calibration's supported thread widths;
- isolated MAPE/tail error for all points and a separate summary excluding the
  residual-training Cartesian grid recorded in calibration provenance;
- full-call contention MAPE and tail error;
- measured regret of the shape selected by analytical prediction.

Before changing the production default, validate unseen routes, widths, mixed
route distributions, and both isolated and concurrent full-stage execution.
Initial acceptance gates are isolated MAPE at most 10%, contention
P90 absolute error at most 15%, and maximum measured shape regret at most 5%.
Any failed gate keeps the empirical model as the runtime default.

The 2026-08-02 AmazonC5192Cores NUMA0 cache-derived calibration measured
0.340/30.377 TFLOP/s at 1/96 threads for the L1-hot M12 GEMM core, versus
0.412/39.309 TFLOP/s for the register-only diagnostic. The core curve uses 64
warmups and 4096 timed calls per width because one call is only about 0.8 us.
It reached 9.18% isolated MAPE on 108 true holdout points, 49.57% contention
P90 error, and 8.17% maximum shape regret. The isolated gate now passes and the
compute ceiling has the correct kernel-level meaning, but the contention/regret
gates still fail. Commands, anchors, and per-route decisions are recorded in
`optimizations/fused_moe_sve/results/amazon_192c_analytic_hot_gemm_core_20260802.md`.

## Known Limits

- SVE BF16 N-split only; a different ISA or M/MN split needs a new kernel mapper.
- The core ceiling is currently M12. Exact-M M1-M11 issue-efficiency ratios are
  not yet independent services, so short-tail accuracy still depends on the
  exact demand mapper plus the common M12 ceiling.
- Cold distinct-expert weights are assumed. Reusing the same expert across calls
  needs an explicit warm-weight state.
- The three-anchor packed-B retention curve plus physical steady-scan state is
  not a set-level cache simulator. It closes the long-route W13 error in the
  tested grid, but does not represent cache sets, prefetch streams, or topology
  below the NUMA-level aggregate service curve.
- W13 and W2 share one task width even though their full-stage N/K shapes and
  resulting owner stripes differ. A future stage-width planner would need an
  explicit handoff/runtime contract; there is no hidden window selector.
- The AmazonECS8Cores cache/service calibration is machine-local, but its
  packed-B retention fractions are currently a transferred 192-core prior.
  It remains a portability holdout until a local multi-team refill probe
  replaces that prior.
- One-pass B streams are excluded from reusable capacity. Their small persistent
  LLC pollution is not modeled separately from stream bandwidth.
- Gather, router, final TopK merge, communication, and cross-NUMA traffic are not
  yet separate demand vectors.
- Frequency/power effects must be reflected in the measured aggregate service
  curves.
- Dynamic in-task resize is not modeled; phases use fixed thread widths.
- Core-pair/mesh placement below the NUMA level is not yet a separate resource;
  its effect must currently be reflected in the NUMA-local LLC-refill service
  calibration and uncertainty.
