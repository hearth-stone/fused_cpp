# Analytical CPU MoE Cost Model

## Status

Status date: 2026-09-01.

`analytic_model.py` is the planner-compatible analytical backend for the SVE
BF16 fused expert. It is explicitly installable through `MoePlannerRuntime`,
but remains opt-in while machine calibrations are validated. The schema-v2
empirical model remains the default backend for full empirical planning and the
holdout reference; it is not an input to analytical prediction.

The model targets one NUMA-local rank, cold distinct-expert weights, N-split
teams, and the current packed-A Xbyak/static SVE kernel mapping. Communication,
router cost, and final TopK merge remain outside this model.

Paper-facing claims, current evidence, and open gates are indexed in
[`../../docs/moe_paper_readiness.md`](../../docs/moe_paper_readiness.md).

## Current Planner Modes

Analytical planning has two intentionally different modes:

- **Quick** is the deployed bounded path. It evaluates homogeneous team shapes,
  computes one exact analytical `T_iso(M,t)` per distinct route count and width,
  assigns experts by deterministic LPT, and selects the minimum isolated lane
  makespan. Candidate selection runs in native C++ when available. The emitted
  Plan V2 is strict; quick does not search phase-DAG contention, temporal order,
  dynamic tail pools, or bounded tail repartition.
- **Full** is the offline reference/autotuning path. It evaluates mixed-width
  strict shapes, temporal lane orders, derived tail-pool candidates, and bounded
  terminal repartition with the analytical phase-DAG simulator. It includes the
  quick winner as a baseline, identifies minimum expected modeled makespan, and
  applies the bounded one-step width gate described below only inside an
  overlapping systematic-error interval. It is not an oracle because model
  ranking error remains measurable.

`MoePlannerRuntime` disables route-plan caching by default because exact route
histograms have low reuse. It can precompute every supported `T_iso(M,t)` scalar
through a declared maximum with `initialize_planner(max_routes)` and persist
that versioned dense cost grid. `FUSED_CPP_MOE_PLANNER_FIXED_THREADS` switches
the runtime to one supported homogeneous width for a controlled fallback; it
does not change full offline search.

On the 2026-08-31 96-core, 31-sample matrix, full cold/warm planning took
0.55--7.06/0.11--3.19 ms in the bounded measured candidate spaces. It found
measured-best plans on the captured uniformish and median traces, but selected
16T LPT at 22.857 ms on high skew while 8T reverse-even measured 17.282 ms. The
24.43% paired gap and 0.692 measured/predicted rank Spearman show that full is
an offline search mode, not a near-optimal oracle.

The 2026-09-01 Arm-codex 80-core follow-up adds a one-step width uncertainty
gate for implicit-shape analytical full search. When the expected winner uses
more than 8T and a candidate capped at the next narrower calibrated width has
an overlapping systematic-error interval, full selects minimum expected time
inside that narrower set. On complete high-skew/median/uniformish traces this
reduced legacy full latency by 15.27/17.95/5.69% in paired medians and left
0/0.26/0% regret against the measured candidate set. The old 80-core
42-second search remains historical evidence for broader candidate-space cost;
the current committed three-trace run took 29.39--50.63 s cold. Its run id is
`20260901T080803Z-arm_codex_internal-arm_high_skew_closure-b2196270211b`.

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

Compute resources use three route-independent values: one-thread rate
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

Both compact curves use the same three measurements. LLC and DRAM instead use
monotone piecewise-linear interpolation through independently measured thread
widths. If adjacent measurements are noisy and non-monotone, equal-weight
isotonic regression first projects them onto a monotone service envelope. An
interpolation fit has zero error at its retained points by construction;
predictive accuracy must therefore be reported on held-out thread widths.

Machine schema v2 records the pinned rank CPU ids and each Linux LLC domain's
id, CPU set, capacity, and local refill curve. For a placement with (t_d)
active threads in LLC domain (d), topology-aware LLC service is:

\[
B_{LLC}(\{t_d\})=
\begin{cases}
B_d(t_d), & \text{one active domain},\\
\min\left(\sum_d B_d(t_d),B_{LLC,rank}^{sat}\right),
& \text{multiple active domains}.
\end{cases}
\]

The LLC capacity used by a placement is likewise the sum of its active-domain
capacities. DRAM remains one NUMA-rank service
(B_{DRAM}(\sum_dt_d)); it is never summed per LLC domain. The rank LLC curve
is retained as the placement-free compatibility path and its final point is
the shared multi-domain fabric ceiling. Schema v1 remains readable but cannot
answer placement-aware queries.

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

`profile_analytic_services.py` reads L1D, L2, LLC, cache-line sizes, LLC ids,
and `shared_cpu_list` from
`/sys/devices/system/cpu/cpuX/cache/index*`. Rank LLC capacity is the sum of
unique domains intersecting the pinned CPU list; LLC probe geometry uses one
domain's capacity rather than pretending all domains form one shared cache.
Its default fractions are 0.625 for
the L1 GEMM core probe and 0.5 for the L2-hot diagnostic. On
AmazonC5192Cores this gives M12/K728/N16 (40,768 B) from a 64 KiB L1D and
M12/K18720/N16 (1,048,320 B) from a 2 MiB L2.

`AnalyticMoeCostModel` schema v7 exposes this placement-aware capacity and
service through the machine calibration API. The current planner DAG still
passes only aggregate active thread counts, so default scoring deliberately
uses the rank curve until physical CPU intervals become part of the simulated
task state. This avoids silently inventing an LLC placement.

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

Plan V2 supplies the selected task width and may also supply an exact per-worker
window in whole N tiles. For stage \(s=(K_s,N_s)\), packed tile width \(\nu\),
team width \(t\), and owner window \(w_s\), the native and analytical mappers
share the same geometry:

\[
q_s=N_s/\nu,\qquad
g_s=t w_s,\qquad
R_s=\left\lceil\frac{q_s}{g_s}\right\rceil,\qquad
b_s=2K_s\nu.
\]

`window_tiles=0` denotes the full-stripe endpoint
\(w_s=\lceil q_s/t\rceil\), hence \(R_s=1\). Every other legal value changes
only the order in which the same full N domain is visited. The full stage bytes
\(B_s=2K_sN_s\), arithmetic, output ownership, and packed ABI are invariant.
The current production planner uses its deterministic stage-window policy;
`AnalyticStageWindowPolicy` v6 is a shadow selector and does not change the
planner search space or production default.

### Analytical Stage-Window Selector

For an already selected \((M,t)\), the selector enumerates the full stripe and
power-of-two owner windows that are at least one L1D and do not starve a worker.
When \(M\le12\), there is only one M panel and therefore no repeated packed-B
scan to preserve; the full stripe dominates and is selected directly.

Let \(P_s\) be the exact number of M panels emitted by the SVE mapper,
\(A_s\) the complete packed-A input for the stage, and \(A_{p,s}\) one maximum
M panel. For one window containing \(n_j\) packed-B tiles, the owner footprint
is \(U_j=\lceil n_j/t\rceil b_s\). The physical demand mapper then separates:

- compulsory B bytes \(n_jb_s\), paid exactly once;
- transient repeated-B refill while distinct A panels turn over the effective
  private L2, followed by a physical resident/streaming state;
- A refill when \(A_s+U_j\) cannot survive between consecutive N windows;
- C write traffic, which is streaming demand rather than a reusable cache set.

The task-local score retains all of those transfer bytes. The additional
full-rank LLC surcharge is narrower: only reusable packed B contributes to the
resident set. For \(h=\lfloor T_{rank}/t\rfloor\) simultaneous complete teams,

\[
G_{B,j}=h\,n_jb_s,
\]

and the extra DRAM charge applies only to repeated B refill bytes
\(\max(B^{L2}_{j}-n_jb_s,0)\). Streaming A and C remain in the transfer demand
but are not counted again as reusable LLC capacity. This distinction prevents
the selector from treating a one-pass stream as if it displaced an equally
sized reusable B window.

Changing windows restarts the N-range control path once per M panel. This cost
is stage-specific because fused W13 executes SiLU, gate multiplication, and
BF16 packC in the range epilogue while the W2 calibration currently uses the
pure full-no-store path as a proxy:

\[
T_{restart,s}=P_s(R_s-1)\delta_s.
\]

The selector uses the serialized incremental objective

\[
J_s=
\sum_{j=1}^{R_s}
\left[
T_{core,j}+T_{epi,j}
+\max(T_{L2,j},T_{LLC,j},T_{DRAM,j})
\right]
+T_{restart,s}+T_{sharedB,s}.
\]

This objective is for policy selection. Absolute expert and DAG time continues
to use the overlapping ECM maximum and the shared-resource event simulator.
Let \(J_{min}\) be the minimum candidate objective and
\(\epsilon_{rel}\) the machine calibration uncertainty. Candidates inside

\[
\mathcal E_s=\left\{w:J_s(w)\le J_{min}+
\epsilon_{rel}\left(T_{xfer}(w_{min})+T_{sharedB}(w_{min})\right)\right\}
\]

are intentionally treated as indistinguishable. The deterministic tie-break
uses a cache-derived owner window rather than an M/T lookup table:

\[
U_{pref,s}=\max\left(C_{L1D},2b_s,
\sqrt{C_{L1D}C^{B,eff}_{L2}}\right).
\]

For the 192-core calibration this is
\(\sqrt{64\text{ KiB}\cdot256\text{ KiB}}=128\text{ KiB}\). The selected
candidate minimizes log-distance to \(U_{pref,s}\), then exact \(J_s\), then
prefers the larger window. W13 and W2 are selected independently and their pair
is validated against a full Cartesian runtime oracle.

The calibrated quantities are machine services and capacities, not route-time
samples: L1/L2/LLC/DRAM and GEMM service curves, physical/effective cache
capacities, packed-B retention anchors, relative uncertainty, and
\(\delta_{13}/\delta_2\). Shape, panel count, traffic, candidate windows, and
cohort size are derived analytically. The packed N tile \(\nu\) is part of the
machine/kernel identity; a holdout must reject a calibration whose tile differs
from the packed runtime ABI.

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
  "schema_version": 2,
  "kind": "moe_analytic_machine",
  "machine": {
    "id": "machine-kernel-frequency-numa-policy",
    "cores_per_rank": 8
  },
  "topology": {
    "rank_cpu_ids": [0, 1, 2, 3, 4, 5, 6, 7],
    "dram_scope": "numa_rank",
    "llc_domains": [
      {
        "id": "0",
        "cpu_ids": [0, 1, 2, 3],
        "capacity_bytes": 50331648,
        "service": {
          "single_thread_rate": 2.5e10,
          "saturated_rate": 5.0e11,
          "saturation_threads": 4,
          "curve": "piecewise_linear",
          "points": [
            {"threads": 1, "rate": 2.5e10},
            {"threads": 4, "rate": 5.0e11}
          ]
        }
      },
      {
        "id": "1",
        "cpu_ids": [4, 5, 6, 7],
        "capacity_bytes": 50331648,
        "service": {
          "single_thread_rate": 2.5e10,
          "saturated_rate": 5.0e11,
          "saturation_threads": 4,
          "curve": "piecewise_linear",
          "points": [
            {"threads": 1, "rate": 2.5e10},
            {"threads": 4, "rate": 5.0e11}
          ]
        }
      }
    ]
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
      "saturated_rate": 2.4e12,
      "saturation_threads": 8,
      "curve": "power"
    },
    "matrix_flops": {
      "single_thread_rate": 4.1e11,
      "saturated_rate": 3.0e12,
      "saturation_threads": 8,
      "curve": "power"
    },
    "l1_bytes": {
      "single_thread_rate": 1.0e11,
      "saturated_rate": 8.0e11,
      "saturation_threads": 8,
      "curve": "shared_bottleneck"
    },
    "l2_bytes": {
      "single_thread_rate": 5.0e10,
      "saturated_rate": 4.0e11,
      "saturation_threads": 8,
      "curve": "shared_bottleneck"
    },
    "llc_bytes": {
      "single_thread_rate": 2.5e10,
      "saturated_rate": 1.0e12,
      "saturation_threads": 8,
      "curve": "piecewise_linear",
      "points": [
        {"threads": 1, "rate": 2.5e10},
        {"threads": 4, "rate": 5.0e11},
        {"threads": 8, "rate": 1.0e12}
      ]
    },
    "dram_bytes": {
      "single_thread_rate": 4.0e10,
      "saturated_rate": 3.8e11,
      "saturation_threads": 8,
      "curve": "piecewise_linear",
      "points": [
        {"threads": 1, "rate": 4.0e10},
        {"threads": 4, "rate": 3.5e11},
        {"threads": 8, "rate": 3.8e11}
      ]
    }
  },
  "overheads": {
    "call_setup_ns": 0.0,
    "expert_fixed_ns": 0.0,
    "route_ns": 0.0,
    "by_width": [
      {"threads": 1, "expert_fixed_ns": 148000.0, "route_ns": 10472.0}
    ],
    "stage_fixed_ns": 0.0,
    "range_fixed_ns": 0.0,
    "panel_range_restart_ns": 0.0,
    "w13_panel_range_restart_ns": null,
    "w2_panel_range_restart_ns": null
  },
  "planner": {
    "supported_widths": [1, 2, 4, 8]
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
`overheads.by_width` is an optional discrete override for expert-level fixed
and per-route overhead. Missing widths inherit the scalar `expert_fixed_ns`
and `route_ns`; the model does not interpolate or extrapolate these overheads.
This term is a serialized operator phase and is not multiplied by cache or
bandwidth dilation. It is intended for measured task-fragmentation/runtime
cost that is width-specific but independent of the active memory-service set.
`gemm_core_flops` is the active compute service. `matrix_flops` and `l1_bytes`
are retained in new files for diagnostics. Removing `gemm_core_flops` makes the
calibration invalid because register-only BFMMLA is not an attainable kernel
peak.

Generate the independent service probe and build a thin calibration with:

```bash
python cpu_moe_schedule_optimization/cost_model/profile_analytic_services.py \
  --output services.json --cpu-ids 0-95 \
  --widths 1,2,4,8,16,24,32,48,64,96 \
  --range-repeats 7

python cpu_moe_schedule_optimization/cost_model/build_analytic_calibration.py \
  services.json --output machine.json --report fit.json \
  --supported-widths 1,2,4,8,16,32 \
  --training-profile isolated_training.json \
  --backend-n-tile 8 \
  --l2-b-reuse-effective-fraction 0.125 \
  --l2-b-reuse-miss-floor 0.18 \
  --l2-b-reuse-miss-at-capacity 0.623 \
  --l2-b-reuse-miss-ceiling 0.869
```

On a multi-LLC NUMA rank, profile the full rank and at least one representative
domain, then pass `--llc-domain-probe DOMAIN_ID=domain-services.json`. Repeat it
for heterogeneous domains. `--topology topology.json` exists only to replay a
legacy service file captured before topology metadata was embedded; new probes
write topology themselves. `--supported-widths` declares planner-legal widths
independently of the denser service-probe width set.

For deployment bootstrap, use the explicit quick-calibration function instead
of running the full research probe. It samples only powers-of-two up to 16,
per-domain half/full widths, and the full rank; planner-legal widths remain a
separate set. The call is synchronous, has no import-time or first-request
hook, restores the caller's affinity/Torch-thread/SVE-dispatch state, and
refuses to overwrite an existing profile unless requested:

```python
from fused_cpp.moe import enable_moe_planner_quick

runtime = enable_moe_planner_quick(
    cpu_ids=range(96),
    output="/var/cache/fused_cpp/moe-machine.json",
    hidden_size=4096,
    intermediate_size=512,
    global_experts=256,
    local_experts=256,
    mode="tp",
    degree=4,
)
```

Call this during deployment or service setup. Compatible calls through the
normal `fused_moe_bf16_tiled` entrypoint then use cached Plan V2 scheduling;
passing `None` to `set_default_moe_planner_runtime` restores the existing
dispatcher.

The machine probe never reads the expert shape, so on its own it leaves the
operator overheads at zero. `enable_moe_planner_quick` therefore follows it, by
default, with `train_quick_operator_overheads`: isolated experts of the given
shape at routes 1/4/12/48/192/2040 on every supported width, one common
`(expert_fixed, route, stage_scale)` residual on 12/192/2040 as the research
calibration fits it, then one `(expert_fixed, route)` pair per width. The
service probe itself runs three times and takes the per-point median. On
Amazon C9g NUMA0 this takes 49 s for TP4 (H=4096, F=512) and 70 s for TP2
(F=1024), and three independent runs score 7.71-7.85% (TP4) and 8.72-10.55%
(TP2) mean absolute error on 434 held-out isolated points per shape, against
7.66% and 10.55% for the research calibration; the untrained single-probe
workflow scored 20.7-31.4% and 17.2-30.2%
(`optimizations/fused_moe_sve/results/c9g_one_click_calibration_20260923.md`).
The trained file is bound to that expert shape: regenerate it when the parallel
strategy changes the shape. `train_overheads=False` keeps the shape-independent
machine calibration. Packed-B retention stays at the builder defaults; measured
C9g values did not change isolated accuracy (`c9g_b_retention_20260923.md`).

The production runtime currently bounds cold planning to homogeneous team
shapes. It ranks those shapes with analytical isolated expert times and LPT
lane loads, emits a strict Plan V2, and uses native C++ assignment/selection
when the extension is available. It does not cache the route plan by default;
only the identity-bound `T_iso(M,t)` scalars are cached. It does not run
mixed-width phase-DAG, temporal-order, or dynamic-tail candidate search on the
request path. Use `PlannedMoE(..., search_mode="full")` for offline analysis;
reducing full-search cost and improving production candidate quality remain
follow-up work.

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
planner = PlannedMoE(model, num_cores=96, search_mode="full")
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

The full-stage validator still evaluates task-width timing. Validate the
independent shadow stage-window selector with the exact runtime tile geometry:

```bash
numactl --physcpubind=0-95 --membind=0 \
  python optimizations/fused_moe_sve/benchmarks/bench_analytic_stage_window_tiles.py \
  --calibration machine.json --cpu-ids 0-95 \
  --routes 72,216,384 --widths 4,16 \
  --measurement-experts 96 --warmup 2 --runs 15 \
  --full-cartesian --store-samples
```

The 2026-08-11 v6 decision holdout used 96 distinct expert weights, 32 MiB
HugeTLB, shuffled candidate order, and the full legal W13 x W2 Cartesian set for
each of six high-value transition points:

| M | T | Analytical W13/W2 tiles | Runtime oracle tiles | Regret |
| ---: | ---: | :--- | :--- | ---: |
| 72 | 4 | 2 / 16 | 1 / 8 | 4.40% |
| 72 | 16 | 2 / 16 | 1 / 8 | 1.74% |
| 216 | 4 | 8 / 16 | 2 / 16 | 2.51% |
| 216 | 16 | 8 / 16 | 8 / 16 | 0.00% |
| 384 | 4 | 8 / 16 | 8 / 16 | 0.00% |
| 384 | 16 | 8 / 16 | 8 / 32 | 0.47% |

Median/linearly-interpolated-P90/maximum regret is `1.10/3.45/4.40%`; 4/6 points are within 2% and
6/6 pass the declared 5% maximum-regret gate. Median candidate-rank Spearman
correlation is 0.869. This closes the **window-selection subproblem on the
declared 192-core TP4 transition domain**: the formula selects a near-oracle
pair without an M/T latency table.

It does not prove an exact absolute-time decomposition. A paired-round
additivity check still sees up to 17.69% W13/W2 interaction residual on extreme
non-selected pairs, and M=72/T=16 has rank correlation 0.098 despite only 1.74%
selected regret. These are explicit limits: v6 is adequate for choosing a
window inside the tested domain, not for predicting every Cartesian point or
for replacing the production planner.

The older policy-v2 `1.50/2.98/3.38%` result used a coordinate oracle and a
retired range ABI. It remains useful provenance but cannot gate the current
tile-window runtime. The current protocol, thin-calibration evidence, rejected
hypotheses, and full result table are recorded in
`optimizations/fused_moe_sve/results/analytic_stage_window_policy_v6_closure_20260811.md`.

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
That historical run reached 9.18% isolated MAPE on 108 true holdout points,
49.57% contention P90 error, and 8.17% maximum shape regret. The isolated gate
passed and the compute ceiling obtained the correct kernel-level meaning, but
the contention/regret gates failed. Commands, anchors, and per-route decisions
are recorded in
`optimizations/fused_moe_sve/results/amazon_192c_analytic_hot_gemm_core_20260802.md`.

The 2026-08-31 NUMA1 refresh adds a machine-local packed-B repeated-scan probe
instead of transferring the old retention prior. Five independent processes
give a fitted retention knee at 0.662 x the nominal private-L2 capacity, a
17.37% repeated-scan miss floor, 68.03% miss fraction at 2 MiB, and 79.80% at
4 MiB. With these anchors, isolated holdout MAPE is 7.96%, contention MAPE/P90
is 10.85/16.08%, and maximum measured shape regret is 9.21%. Relative to the
transferred prior, contention P90 improves from 16.97% while maximum regret is
unchanged. Retention is therefore justified as a model variable, but wide-team
concurrent pressure and temporal-order ranking remain open. The current report
is
`optimizations/fused_moe_sve/results/amazon_192c_paper_closure_20260831.md`.

Repeating the sole six-point tile-window boundary case with 31 samples gives an
updated approximate median/P90/maximum regret of 0.83/3.05/4.18%. This retains
the narrow 5% window-selection result without expanding its declared domain.

## Known Limits

- SVE BF16 N-split only; a different ISA or M/MN split needs a new kernel mapper.
- Production quick search ignores concurrent phase interactions when ranking
  homogeneous shapes. On the 80-core nine-case check it improved long/short
  bimodal by 22.13% versus fixed 8T, but regressed active-set 8/16 by
  15.54%/11.31% after selecting an over-wide 40T team. It is therefore a
  supported bounded planner, not a demonstrated universally superior policy.
- Full analytical search is an offline reference/autotuning mode. The current
  bounded 96-core empirical matrix plans in 0.55--7.06 ms cold, while the
  Arm-codex analytical strict search takes 29.46--50.41 s for 142 candidates
  and the broader historical 469-candidate search took about 42 seconds.
  Candidate-space-dependent cost must always be reported.
- Analytical stage-window v6 is shadow-only. Its 5% selection gate covers the
  192-core TP4 M=`72,216,384`, T=`4,16` transition domain, not arbitrary routes,
  widths, shapes, machines, or mixed-workload planner decisions.
- W2 range restart currently uses the pure full-no-store service as a proxy, and
  extreme W13/W2 pairs retain up to 17.69% paired additivity residual. Do not use
  the selector score as an exact absolute pair latency.
- The core ceiling is currently M12. Exact-M M1-M11 issue-efficiency ratios are
  not yet independent services, so short-tail accuracy still depends on the
  exact demand mapper plus the common M12 ceiling.
- Cold distinct-expert weights are assumed. Reusing the same expert across calls
  needs an explicit warm-weight state.
- The three-anchor packed-B retention curve plus physical steady-scan state is
  not a set-level cache simulator. It closes the long-route W13 error in the
  tested grid, but does not represent cache sets or prefetch streams. Machine
  schema v2 represents LLC domains, but the current planner DAG does not yet
  carry physical placement into scoring.
- The Amazon 192-core machine-local retention probe closes only task-local
  single-core reuse. It does not identify active multi-team LLC refill or
  wide-team concurrent-service behavior; the unchanged 9.21% shape regret and
  high-skew misranking demonstrate that distinction.
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
