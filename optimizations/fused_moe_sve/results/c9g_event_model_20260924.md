# Probe-calibrated event model (v10/v11) on C9g (2026-09-24)

User decision (2026-09-24): calibrate the event model that the fast (hot_wide) planner depends on
on C9g by the Arm-codex procedure, instead of running fast with Arm-codex terms. Design and
deviation: `tmp/c9g_event_model_20260924/design.md`. C9g node 0 (CPUs 0-95, `--membind=0`), TP4
expert shape, SVE-128 (n_tile 8), jemalloc never-purge. Phase skeleton: the guarded C9g research
calibration.

## Probes

The Arm-codex probes P1-P4 (four dilation curves), P5 (whole-call overhead) and P6 (v11 mid-M
terms) were ported with only the core grid changed. Background core counts are
{0, 8, 16, 32, 48, 64, 80} plus the largest 4T background next to the target (92 for 2T and 4T,
88 for 8T, 80 for 16T, 64 for 32T). There were two sessions, 246 P1-P4 cells and 187 P6 cells
each.

One deviation from the frozen design, before any build. Session 1 at the Arm-codex background
sizing (1.2x the modeled target) left 12 P1-P4 and 28 P6 cells uncovered: the target outlived
its background, because contention here is far stronger than the model the sizing uses. The
background was raised to 3.0x, session 1 was discarded, and both sessions were rerun. Only two
cells stay uncovered (dropped by rule).

Also changed on 2026-09-24 by user decision: C9g chains keep only a fixed cooldown after our own
previous process, with no load-average wait. The machine is exclusive.

## The calibration (`build_c9g.py`, `bench_assets/moe_paper/amazon_c9g_96c_tp4/probe_event_v11_c9g_20260924.json`)

| term | C9g | Arm-codex |
| --- | --- | --- |
| eps, O(t) (P2 n=0 single-expert cells, same grid fit) | 0.08, 1 us x t (fit MAE 1.2%) | 0.065, 10 + 2.5 us x t |
| t_over (P5, median of 16 plan-session values) | 0.45 ms | 0.78 ms |
| D_LL, 4T target: n = 32 / 48 / 64 / max | 2.56 / 3.76 / 4.72 / 5.86 (n = 92) | 1.12 / 1.32 / 1.64 / 1.90 (n = 76) |
| D_LS, max over widths | 1.48 (2T) - 2.25 (16T) | <= 1.35 |
| D_SL, 4T at max n | 1.28 | 1.13 |
| D_SS, 2T / 4T / 8T at max n | 1.92 / 1.51 / 1.16 | <= 1.03 |
| loading weight w(24 / 48 / 96 / 192 / 384) | 0.79 / 0.71 / 0.50 / 0.26 / 0.27 | 0.79 / 0.56 / 0.35 / 0.18 / 0.11 |
| isolated correction c(t, M 24-384) | 2T 1.03-1.10, 4T 1.00-1.08, 8T 0.99-1.11, 16T 0.92-0.95 | 2T 0.93-0.95, others ~1 |

Contention on C9g is far stronger than on Arm-codex, and it is not confined to weight loading.
Loading against loading reaches 5-6x at full occupancy, and even steady phases dilate by up to
1.9x on narrow lanes. 17 points differ by more than 3% between sessions (session mean used).
The mechanism test fails as on Arm-codex: core-time excess per expert does not collapse across
widths. The curves are a width-indexed service table.

Not re-measured: g0 = 0.02 (window isolated gain) is carried over from Arm-codex. The base
calibration prices only its calibrated widths (1/2/4/8/16/32/48/96). The probes cover 2-32T,
so 48T and 96T are extrapolated.

## Check on data the calibration never saw

Plans: the 27 workloads x homogeneous widths measured in `tmp/c9g_width_error_20260923` (18 real
TP4 layers plus 9 catalog presets, two sessions). Script: `check_width_table.py`.

Measured / event-predicted call time, median per width: 1T 1.28, 2T 1.11, 4T 1.10, 8T 1.05,
16T 1.06, 32T 0.83, 48T 0.66, 96T 0.61. The 4-16T level matches Arm-codex's 1.07-1.10. Wide
teams are overpriced beyond the probed range.

Regret of each scorer's width choice against the fastest measured width in the set:

| scorer | widths 4/8/16/32 (fast): median / mean / max / >1% | calibrated divisors 1-96: median / mean / max / >1% |
| --- | --- | --- |
| quick score | 0.34% / 2.01% / 24.2% / 12 of 27 | 1.31% / 2.88% / 23.6% / 14 |
| analytic DAG | 1.73% / 2.18% / 7.8% / 17 | 1.93% / 2.29% / 7.8% / 18 |
| **event v11 (C9g)** | **0.00% / 1.09% / 14.2% / 7** | **0.00% / 2.10% / 23.6% / 9** |

The C9g event model is the best width selector of the three. Its misses:

- Six workloads miss by 1.5-3.0%, mostly 16T preferred where 8T is faster.
- The long/short bimodal batch picks 8T where 16T is 14.2% faster: the long experts on 8T run
  slower than predicted.
- Over all widths, active-set-32 picks 2T at 23.6% worse: 2T is under-predicted, the same
  residual Arm-codex's v11 kept. The asset marks 1T/2T unreliable, and fast never uses them.

Next: derive the fast planner's templates and lane scales on C9g under this model, then validate
fast against production quick on fresh layers.

## Fast planner configuration on C9g (stage 3, model only)

`derive_fast.py`: on the 18 derivation layers (r008/r016/r022 x layers 2/9/15/25/36/40) the
model-objective LNS ran 60 s per layer (widths 4/8/16/32, one 96-core domain). It started from
quick's homogeneous plans and the default fast plan. The searched plans share one shape: 4T bulk
lanes plus at most five wide lanes, 8T (one to four) and 16T (zero or one), at most 48 wide cores,
never 32T. The three layer-2 workloads are all 4T. Under the model the searched plans are 15-20%
below the best homogeneous plan.

Lane scales. The Arm-codex rule, the median lane event / isolated load ratio of the searched
plans, gives 4T 2.56, 8T 1.32, 16T 1.28 here (Arm-codex: 1.119 / 1.065 / 1.045). Used as is, it
makes fast worse than with the Arm-codex constants (+25.2% against LNS in median, against
+17.8%). The ratio depends on what a lane holds (4T lanes carry the small, strongly
load-dilated experts), not on its width. `fit_fast_scale.py` therefore fits the scales on the
same 18 layers under the model instead (16T fixed at 1, since only ratios matter). The result is
a flat optimum at 4T 1.6 and 8T 1.15-1.6; 4T 1.60, 8T 1.45, 16T 1.00 are used.

Model-only result (`fast_check.py`, event call time, median over the 18 layers):

| plan | gap to LNS (median / max) |
| --- | --- |
| fast, C9g configuration | +10.3% / +15.2% |
| fast, Arm-codex defaults | +17.8% / +25.4% |
| production quick | +24.9% / +28.1% |

The model predicts fast 11.1% faster than production quick (range -13.1% to +3.7%). Python
planning takes 6.3 ms per layer. The configuration is written to the asset's `hot_wide` entry,
and `PlannedMoE(search_mode="hot_wide")` reads it (`_hot_wide_config`). Measured validation on
fresh layers follows.

## Measured validation on fresh layers (stage 4)

Frozen in `design.md` before plan generation. 18 real TP4 layers never used on C9g (r008/r016/r022
x layers 1/7/13/19/30/39). Four plans per layer, all with the registered window table:
production quick (`baseline`), `fast` (the C9g `hot_wide` entry), `fast_default` (Arm-codex
template and scales) and `lns` (60 s model LNS). Measured with the width-table bench (protocol v2,
bitwise gate) on node 0, two sessions, 120 s fixed cooldown; session spread median 0.21%, max
3.35%. Analysis: `analyze_e3.py`.

| comparison (measured, median over layers) | result | gate |
| --- | --- | --- |
| F1 `fast` vs production quick | +0.12% (range -3.31% to +2.66%, faster on 8 of 18) | FAIL (needs <= -5%) |
| F2 `fast` vs `lns` | +7.83% median, +12.47% max | FAIL (needs <= 2% / 5%) |
| `fast_default` vs production quick | +3.23% (+0.13% to +10.63%) | reported |
| `lns` vs production quick | **-7.63% (-10.56% to -0.06%), faster on 18 of 18** | reported |

Measured / event-predicted call time, median: production quick 1.070, fast 1.191,
fast_default 1.182, lns 1.223. The model underprices mixed plans (4T bulk with wide lanes) by
about 12% relative to homogeneous ones. It still ranks the searched plan first, but predicts its
gain at about 20% against 7.6% measured. Planning time (Python): fast 6.3 ms, production quick
12.3 ms.

Reading. The C9g event model is useful: plans searched under it are faster than production quick
on every fresh layer, 7.6% in median. The fast planner's rule (LPT on per-width scaled isolated
loads over fixed templates) does not capture that gain on C9g; it ties production quick. The
C9g configuration still beats the Arm-codex constants by about 3 points. Candidate next steps:
make the search usable on the request path (native event model plus a time-budgeted search, or
plans cached by route signature), and find the cause of the family-dependent underprediction.

## Native event scorer and time-budgeted search (option 1, model only)

`NativeProbeEventSim` (`csrc/moe_planner/probe_event_sim.cpp`) ports `ProbeEventModel.simulate`
without background width factors, event logs or calibration hooks. Those stay in Python, and
`probe_event_model.py` is unchanged. `NativeProbeEventScorer`
(`cost_model/probe_event_native.py`) wraps a model and is a drop-in `simulate` for the LNS.
Parity (`tests/test_moe_probe_event_native.py`): random multi-lane chains on the Arm-codex and C9g
v11 assets, windowed and full-stripe, makespan and finish times within 1e-9 relative. The only
difference from Python is the summation order of the active set; Python's round-half-even is
kept. One simulation of an 18-layer LNS plan: Python 15.0 ms, native 0.24 ms.

Time-budgeted LNS with the native scorer, single process (`budget_lns.py`), on the 18
derivation layers (model objective, median over layers):

| search | evaluations | gap to 60 s LNS | vs production quick (model) |
| --- | --- | --- | --- |
| fast (no search) | - | +10.3% | -11.1% |
| 0.05 s | 49 | +9.1% | -12.5% |
| 0.1 s | 94 | +7.7% | -13.4% |
| 0.5 s | 478 | +4.0% | -16.2% |
| 1 s | 895 | +2.9% | -17.5% |
| 2 s | 1700 | +1.5% | -18.3% |
| 5 s | 3990 | +0.7% | -18.7% |

The search evaluates about 1000 candidates per second. The simulation takes a quarter of each
evaluation; the rest is the Python LNS itself (moves, lane packing, task construction). The
model overstates gains: it predicted about 20% for the 60 s LNS plans that measured 7.6%. A
request-path search per call is therefore not viable at these costs. A 14 ms layer cannot pay
even the 50 evaluations that match fast. Ways that keep the gain off the per-call path: a plan
cache keyed by routing signature, refined by a background search; a C++ search with batch
evaluation; or better fast rules distilled from searched plans.

## Plan cache hit rate and template transfer (`cache_hit.py`)

Data: the three measured DSV4 prefill requests (2048 tokens each) x 43 layers, the only routing
traces available. There is no decode trace.

- **Exact signature hits (PlannedMoE `signature`): 0 of 129.** All 129 (request, layer) signatures
  are distinct. The same layer in two requests differs in 8-19 of the 24 fields (median 14). A
  stream r008 -> r016 -> r022 hits neither a shared nor a per-layer cache.
- **Template transfer: none of the search gain carries over.** Over 72 same-layer transfers, one
  request's LNS lane widths were filled by the fast planner's LPT with another request's
  routes. Under the C9g event model, the transferred plans are +11.9% above that request's own
  LNS (max +54%). They are +1.0% against its own fast plan (range -4.5% to +39%), and they recover
  -0.09 of the fast-to-LNS gap in median. The searched gain lives in the per-batch assignment, not
  in a reusable shape.

So on prefill routing a signature-keyed plan cache, with or without background refinement,
would neither hit nor transfer. Not covered: decode batches, whose small route counts could
repeat more often.

## Head-to-head with vLLM, the flat queue and KTransformers (`bench_h2h.py`, `analyze_h2h.py`)

Frozen in `design.md` before the full plans were generated. Same 18 fresh layers, one process per
request file. Arms, all timed on the same tensors:

- this repository's planned arms: production quick (`baseline`), `fast` (C9g configuration),
  `full` (full search scored by the C9g event model, 0.15 s per layer) and `lns` (60 s search);
- the flat staged queue with this repository's kernel, blocks L2/128/256/512;
- vLLM's Arm `cpu_fused_moe` (neon/BFMMLA, its own prepacked weights);
- KTransformers' dataflow on this repository's GEMM, blocks 128/256/512.

Protocol v2, 3 warmup + 15 runs, two sessions, 120 s cooldown, node 0,
`OMP_NUM_THREADS=96 OMP_PROC_BIND=false`. Fused arms were bitwise equal to the baseline plan; vLLM
and KT passed the BF16 tolerance. Session spread median 0.25%. One point is unstable (r008_l13
`full`, 16.6% between sessions) and four sit at 2.5-3.7%.

Median over layers of X / reference - 1 (wins = layers where X is faster):

| arm | vs vLLM op | vs flat queue (L2 block = best block on every layer) | vs KT (best block) |
| --- | --- | --- | --- |
| production quick | -4.12% (11/18) | -3.34% (11/18) | -21.5% (18/18) |
| fast | -4.08% (12/18) | -3.27% (11/18) | -22.2% (18/18) |
| full (event model) | -4.35% (13/18) | -3.58% (12/18) | -22.4% (18/18) |
| **lns (60 s)** | **-6.56% (18/18)** | **-5.95% (18/18)** | **-27.5% (18/18)** |
| flat queue | -0.85% (17/18) | - | -20.6% (18/18) |

- On these real layers every planned arm is ahead of vLLM's op and of the flat queue in median.
  The request-path planners (production quick, fast, full) are ahead by 3-4% but lose on 5-7 of
  18 layers, by up to about 9%. vLLM's op and the flat queue win on some layers, so the planned
  arms cannot claim to be faster on every layer.
- Only the 60 s searched plan wins on all 18 layers, by 1.5-22% over vLLM. Its cost is the search
  that does not fit the request path.
- The flat queue with this repository's kernel is 0.85% faster than vLLM's op (17/18). Its L2
  block was the best of the four on every layer.
- KTransformers' dataflow is 20-27% slower than everything else, as in the K comparison.
- `full` scored by the C9g event model is only slightly better than production quick here (-0.2
  points against vLLM in median), well short of the searched plan.

Scope: one machine, TP4 expert shape, 18 prefill layers of three DSV4 requests. Decode is not
covered.

## Adding quick's homogeneous shapes to fast (negative)

Motivation: on the 18 validation layers fast was slower than production quick on 10 (up to
+2.7%). Its templates cannot produce quick's whole-machine 8T/16T/32T shapes; only all-4T
overlaps.

- Adding 8/16/32T whole-machine templates and refitting the lane scales under the event model
  makes fast worse (gap to LNS +10.7% median, +24.3% max, against +10.3% / +15.2%). The model
  underprices mixed plans relative to homogeneous ones by about 12%, so it almost never
  prefers them (1 of 18).
- A separate exchange rate k for the homogeneous templates was then fitted on measured times.
  Tuning used the 18 derivation layers only (`gen_tune.py`, `fit_k.py`; two sessions, same
  process; fast, fast's homogeneous 8/16/32T plans and production quick). Against production
  quick:

| rule | median | layers > 0.5% slower |
| --- | --- | --- |
| fast (hot-wide templates only) | -0.79% | 2 of 18 |
| k >= 1.05 (never picks homogeneous) | -0.79% | 2 |
| k = 0.95 | -0.79% (worst +3.05%) | 2 |
| k <= 0.90 | -0.58% to -0.77% | 5-8 |
| per-layer best of fast and homogeneous (oracle) | -1.73% | 1 |

The headroom over fast is about one point even with an oracle. Fast's scaled loads cannot say
when a homogeneous shape wins: the fastest one is mostly 8T (7 layers), and fast's 8T scale
makes 8T look expensive. No k makes fast never worse than quick, so the change was not
validated on the fresh layers and is not kept. The patch is in
`tmp/c9g_event_model_20260924/fast_homogeneous_templates.patch` (Python and native, with tests).

## Why the analytic model misprices whole plans with contention (r008 layer 9, per-task traces)

`trace_tasks.py` traced four plans on node 0: homogeneous 4T/8T/16T and fast's mixed plan (5
traced calls each, trace overhead under 1%). `analytic_vs_trace.py` / `spill_test.py` /
`small_m_stages.py` compare each task's measured time with the analytic placed DAG's prediction
(task time over the analytic isolated time). The analytic calibration carries no wide-team,
narrow-team, gather or DRAM-injection terms (all absent), one 96 MB LLC domain (64 MB
effective) and a DRAM curve saturating at 411 GB/s.

| plan | measured | analytic | analytic without LLC spill |
| --- | --- | --- | --- |
| homogeneous 4T | 21.29 ms | 38.72 | 25.23 |
| homogeneous 8T | 13.82 | 12.07 | 10.16 |
| homogeneous 16T | 14.25 | 11.62 | - |
| fast mixed (1x16T + 4T bulk) | 13.65 | 21.90 | 12.95 |

1. **The LLC-spill rule overprices mid and large M on narrow lanes (the main error).** When the
   concurrent tasks' working sets exceed the effective LLC, the model re-reads B from DRAM for
   every 12-row panel of every steady phase. For a 4T expert that is 61.6 MB at M=96 and 1090 MB
   at M=1500, against 8.4 MB of W13 weights. With 24 lanes the spill fraction is 0.6-0.8, and
   those phases become DRAM-bound. Measured dilation of 4T tasks with M 49-192 / 193-768 / >768
   is 1.37 / 1.30 / 1.21; the model predicts 4.50 / 4.29 / 2.47. With the spill disabled the
   same tasks predict 1.88 / 1.79 / 1.47, and the 4T and mixed makespans move to within 18% / 5%
   of measured. B reuse across panels evidently stays in each core's private L2 (the kernel's
   windowed stripes), so aggregate LLC pressure does not force the re-reads the rule charges.
2. **Loading of small experts is underpriced once the spill is removed.** M <= 12 tasks
   measured 4.92 / 4.03 / 3.02 x their isolated time on 4T / 8T / 16T; the model predicts
   3.80 / 2.97 / 2.32 without the spill. The measured W13+W2 loading of those tasks implies
   about 18.5 / 34.6 / 61 GB/s per lane, 440 / 415 / 368 GB/s summed over the lanes, i.e. the
   calibrated 411 GB/s ceiling. The capacity is right; the model has fewer lanes loading at once
   than the hardware does. This is not yet checked event by event.
3. Not modeled at all on C9g: wide-team pressure, narrow-team correction, gather pressure, DRAM
   domain injection; merge and call overhead are outside the DAG.

Candidate fix for (1): charge the panel re-read only when a task's own per-core B window exceeds
its private L2, not when the aggregate working set exceeds the LLC. It must be validated on the
width table and on these traces before any adoption.

### Lab: a private-L2 window spill rule in place of the aggregate LLC rule

`tmp/c9g_event_model_20260924/l2_window_spill.py` (Lab subclass, production unchanged). A task's
spillable bytes reach DRAM only if its executed per-core packed-B window (the C9g window table's
window, else the full stripe) plus one A panel exceeds the private L2, and then only in the
fraction its own working set misses the LLC. No cross-task aggregation. Stage phases are rebuilt
before the isolated stage calibration; isolated times change by at most 1.3e-5. On this table only
1T-3T and 4T W13 windows exceed the L2, so in practice the rule removes the aggregate spill.
Research calibration `cal_median_tp4_guarded.json`; no remeasurement, all times from earlier runs.

| set | old rule | L2 window rule |
| --- | --- | --- |
| traces r008 l9, pred/meas-1: 4T / 8T / 16T / fast | +82% / -13% / -19% / +60% | +19% / -27% / -20% / -5% |
| width-error run, 18 real layers, median pred/meas-1: 2T / 4T / 8T / 16T / 32T | +36% / +65% / -17% / -21% / -18% | -2% / +6% / -28% / -23% / -18% |
| same, picks measured-best width | 5/18, regret median 1.52%, max 5.91% | 12/18, median 0.00%, max 10.38% |
| tuning run, 18 layers, median pred/meas-1: fast / homog 8 / 16 / 32 | +56% / -14% / -25% / -23% | -12% / -30% / -28% / -22% |
| same, picks measured-best of fast and homog 8/16/32 | 2/18, regret median 2.21%, max 6.29% | 6/18, median 1.21%, max 4.87% |

The rule removes the narrow-width and mixed-plan overpricing. What is left is a uniform underpricing
of 6T-32T by 20-30% (cause 2 above), and it now decides the picks. Both 10% width regrets are 6T
picked over 8T (6T -24%). In the tuning set, 8 of the 10 layers where fast is measured fastest pick
homog_8, because fast is underpriced by 12% and homog_8 by 30%. The rule is not proposed for
adoption on its own; it needs the loading-contention fix first, and then a measured whole-plan
validation.

### Lab: DRAM capacity under concurrent loading streams

The earlier reading of cause 2 was wrong. Spreading each traced task's weight bytes over its
measured W13/W2 stages (`measured_dram_timeline.py`) gives only 260-310 GB/s summed while small
experts load, not the 411 GB/s ceiling. And in the model only about 5 tasks are loading at a
time (`dram_timeline.py`). The P1 probes (`probe_absolute.py`: M=12 target chain plus 4T M=12
background chains, i.e. every lane streaming its own experts) give the same aggregate model-free:
219 / 297 / 296 / 291 / 303 / 326 / 345 GB/s at 12 / 20 / 36 / 52 / 68 / 84 / 96 busy cores.
The single-team curve the model uses says 242 / 316 / 400 / 400 / 403 / 408 / 411. Rebuilt inside
the analytic DAG (`probe_repro.py`), D_LL is underpredicted for 2T-16T (4T n=64: 3.47 vs 4.72).

`multi_stream_dram.py` keeps the L2 window spill rule and adds one term: with two or more
concurrent DRAM-demanding phases the capacity is min(single-team curve, the 4T P1 aggregate
above). A lone phase keeps the single-team curve and its isolated calibration. The capacity was
frozen from the probes before any whole-plan comparison. Python placed DAG only.

| set | L2 window rule | + multi-stream DRAM capacity |
| --- | --- | --- |
| D_LL 4T n=32/64/92 (fit set) | 1.96 / 3.47 / 4.74 (meas 2.56 / 4.72 / 5.86) | 2.57 / 4.56 / 5.60 |
| D_LL 32T n=32/64 | 1.69 / 2.41 (meas 1.55 / 1.86) | 2.05 / 2.91 |
| D_LS 16T n=64, 32T n=64 | 1.51, 1.37 (meas 2.25, 2.13) | 1.56, 1.41 |
| traces 4T / 8T / 16T / fast | +19% / -27% / -20% / -5% | +22% / -13% / -10% / +7% |
| width-error median, 8T / 16T / 32T | -28% / -23% / -18% | -16% / -13% / -12% |
| width pick | 12/18, max regret 10.38% | 13/18, max 9.85% |
| tuning median, fast / homog 8 / 16 / 32 | -12% / -30% / -28% / -22% | -1% / -18% / -16% / -17% |
| family pick | 6/18, regret median 1.21%, max 4.87% | 8/18, median 0.89%, max 5.43% |

Small-expert loading is now right in the traces: M <= 12 measured / predicted 4.92 / 5.10,
4.03 / 3.94, 3.02 / 3.04 at 4T / 8T / 16T (`multi_vs_trace.py`). Two errors remain, and they
now decide the picks:

- 4T mid and large M are overpredicted: 1.37 vs 1.91 for M 49-192, 1.30 vs 1.83 for 193-768.
  The phases are built from full-stripe geometry, while the runtime executes the window table,
  whose full-load window/full-stripe ratio for 4T is 0.51-0.70.
- 8T/16T mid and large M are underpredicted by 10-20% (8T M 49-192: 1.41 vs 1.16). Steady
  phases get no dilation from loading neighbours in the model; the probes measure D_SL
  1.28-1.37 at full load for 8T/16T. D_LS for 16T/32T and D_LL for 32T are also still off.

### Lab: stage phases from the executed window geometry

`window_geometry.py`, on top of the multi-stream model. When the table windows a stage, each
window becomes a cold phase and a steady phase. The cold phase is the window's first panel,
which loads that window's B from DRAM; the steady phase is the remaining panels, which reuse B
from L2. The demand fields come from the model's own `score_stage_window`, and range restarts
become a setup phase. Full-stripe stages are unchanged. Assumption: the per-(stage, width)
isolated residual scales were fitted on full-stripe runs and are reused unchanged, since no
windowed isolated profile exists. The modeled isolated time moves by only 0-7%, while the
table's measured full-load ratio is 0.51-0.70 for 4T. So in this model the benefit of windows
appears only through contention: the full stripe's B re-reads leave the L2 and load the shared
LLC and DRAM paths.

| set | multi-stream | + window geometry |
| --- | --- | --- |
| traces 4T / 8T / 16T / fast | +22% / -13% / -10% / +7% | +18% / -12% / -10% / -13% |
| tuning median, fast / homog 8 / 16 / 32 / quick | -1% / -18% / -16% / -17% / -13% | -17% / -15% / -15% / -16% / -13% |
| family pick (fast, homog 8/16/32) | 8/18, regret median 0.89%, max 5.43% | 13/18, median 0.00%, max 4.82% |
| width pick over production widths 4/8/16/32/48/96 | 14/18, max 0.88% | 16/18, max 0.34% |
| width pick over all divisor widths 1-96 | 13/18, max 9.85% | 10/18, max 15.40% |

4T mid-M tasks in the traces are now close to measured: M 49-192 1.35 vs 1.19, M 193-768 1.29
vs 1.08 (before 1.37 vs 1.91). The error is now a near-uniform 13-17% underpricing across plan
families, so the picks are mostly right. Remaining:

- All eight wrong picks over all divisor widths choose 6T, measured 15-16 ms vs 12 ms predicted.
  6T is not a production width and the table has no 6T entry, so it runs the full stripe.
- 4T M > 768 still runs the full stripe and is overpredicted (1.46 vs 1.21). These three hot
  tasks set the fixed_4 makespan (+18%).
- Mid and large M on every width are 10-20% slower than predicted (item 2: no steady-phase
  dilation from loading neighbours). Fast's 4T M <= 12 tasks are also underpredicted (5.20 vs
  4.44), and its 16T M <= 12 tasks overpredicted (3.02 vs 4.17).

### Lab: steady-phase slowdown under loading neighbours (item 2), not resolved

The steady probes rebuilt in the analytic DAG (`probe_steady.py`, full-stripe policy as in the
probes) give 8T/16T steady targets exactly no slowdown: D_SS 1.00-1.01 vs measured 1.12-1.24, and
D_SL 1.00 vs 1.09-1.37, all at full load. `steady_breakdown.py` shows the model does dilate their
LLC (2.2-2.5x), but under max(compute, transfer) the dilated LLC time (8T M=2048: 1044 us x 2.45)
stays under the compute time (10911 us). 2T/4T full-stripe targets are the opposite, overpredicted
(D_SS 4T 1.77 vs 1.51). Three structural variants were tried, all with isolated times held fixed:

- `additive_ecm.py`: time = compute + transfer (ECM non-overlap).
- `offered_rate.py`: a phase offers demand / its own isolated duration, instead of demand / its
  isolated time on that resource (rewrites exactly the two offered-rate expressions).
- The two combined.

| set | window (previous) | additive | offered rate | offered + additive |
| --- | --- | --- | --- | --- |
| D_SS 8T / 16T / 32T at max n (meas 1.16 / 1.24 / 1.46) | 1.01 / 1.00 / 1.29 | 1.98 / 1.92 / 1.65 | 1.01 / 1.00 / 1.10 | 1.09 / 1.13 / 1.22 |
| D_LL 4T n=92, 8T n=88 (meas 5.86, 5.01) | 5.60, 5.25 | 4.12, 4.09 | 4.75, 4.54 | 3.11, 2.90 |
| tuning median, fast / homog 8 / 16 / 32 / quick | -17 / -15 / -15 / -16 / -13% | -3 / -3 / -1 / +4 / +3% | -25 / -21 / -18 / -20 / -16% | -36 / -33 / -29 / -23 / -26% |
| family pick | 13/18, max 4.82% | 13/18, max 3.20% | 9/18, max 4.92% | 9/18, max 4.92% |
| width pick over 4/8/16/32/48/96 | 16/18, max 0.34% | 14/18, max 0.88% | 15/18, max 0.36% | 13/18, max 5.91% |

No variant matches both the probes and the whole plans:

- The additive composition puts whole plans within +-5% on average. The probes show it does so for
  the wrong reason: it overstates steady slowdown about 6x (8T D_SS 1.98 vs 1.16) and understates
  loading contention, so the errors cancel.
- The offered-rate correction with the additive composition gets the steady probes about right
  (8T 1.09 vs 1.16). But it collapses loading contention (D_LL 4T n=92 3.11 vs 5.86), and whole
  plans move to -23..-36%. The reason: dilation is applied to the resource time while
  utilization is now measured over the phase duration, so a saturated loader no longer takes
  N x bytes / capacity.

Item 2 needs a self-consistent bandwidth share: solve for the phase durations at which every
phase's achieved rate fits each resource's capacity, instead of dilating isolated times by
isolated offered rates. A composition tweak does not fix it. The window model stays the best
probe-consistent variant.

### Core-PMU counters on C9g (which fix the hardware supports)

Design, script and raw data: `tmp/c9g_event_model_20260924/counter_design.md`, `bench_counters.py`,
`counters_session{1,2}.json` (also on `AmazonC9g192Cores` at the same path), analysis
`analyze_counters.py`, `old_spill_bytes.py`. The machine is c9g.metal-48xl (Neoverse V3, bare
metal), but ACPI describes no CMN or DMC device, so only the core PMU is usable. Events: cycles +
5 (no multiplexing, checked): l2d_cache_refill, l2d_cache_wb, ll_cache_miss_rd, ll_cache_rd,
stall_backend_mem. The run used all 96 node-0 CPUs system-wide, two sessions with 120 s cooldown,
and set `perf_event_paranoid` to -1 for the run only (restored). Per-cell L2 refill bytes agree
between sessions within 1%. `ll_cache_miss_rd` misses prefetch fills (0.03-0.06 GB against 3.22
GB of weights), so SLC and DRAM cannot be told apart. Every line a core reads from DRAM is refilled
into its L2, so L2 refill bytes bound the DRAM reads from above.

1. **The aggregate LLC spill rule is physically impossible.** On fixed_4 it charges 9.63 GB of
   DRAM per call, but the cores refill only 5.49 GB into L2 in total (1.75x the bound); fast:
   7.95 vs 4.98 GB. The L2 window rule charges 2.88 GB (the weights), within the bound. Fix 0 is
   supported.
2. **Window geometry matches the L2 traffic.** Measured L2 refill per call against the model's L2
   bytes (A + B + C) with full-stripe / executed-window geometry: fixed_4 5.49 vs 11.22 / 6.85,
   fast 4.98 vs 8.72 / 4.73, fixed_8 4.88 vs 4.01 / 4.43, fixed_16 5.50 vs 4.92 / 4.92. The
   full-stripe B re-reads the model charged for 4T do not happen. Fix 2 is supported.
3. **Pure loading.** The cells have 256 distinct experts at M=12, 3.22 GB of weights per call.
   L2 refill is 3.47 / 3.62 / 3.88 / 4.39 GB per call at 4 / 8 / 16 / 32T, i.e. 372 / 379 / 387 /
   402 GB/s. Only 346 / 337 / 322 / 295 GB/s of that is weights; the rest is A replicated into each
   thread's L2 and C. L2 write-backs add 0.28-0.79 GB. The lone 4T lane moves 88 GB/s.
   The weight rate reproduces the probe-derived multi-stream capacity (Fix 1, 345 GB/s at 96
   cores). But the total bytes through the L2s are close to the single-team 411 GB/s curve, and
   the model routes the extra A/C traffic to the LLC rather than DRAM. So the counters cannot say
   whether DRAM or the SLC/mesh saturates. Fix 1 is supported as an effective weight-stream
   capacity; its mechanism is unresolved.
4. **The steady-target slowdown is mostly not memory stalls.** Targets alone vs full background
   (span ratio): P2 (steady background) 8T x1.33, 16T x1.51, 4T x1.79; P3 (loading background)
   8T x1.28, 16T x1.28, 4T x1.37. Extra stall_backend_mem on the target CPUs covers only 20-28%
   of the extra cycles, and the clock stays at about 3.25 GHz (3.33 alone). L2 refill of the
   target barely changes under loading background (8T 0.24 -> 0.25 GB); under steady background
   it rises 25-30%.
   - The target's average refill rate alone is 22 / 70 / 31 GB/s (8T / 16T / 4T). The model's
     bytes / isolated steady duration gives 17 / 60 / 72, while its LLC-service offer gives
     178 / 328 / 127. So the offered-rate correction matches the counters for 8T/16T; the current
     offer is 5-8x too high.
   - But neither composition explains the slowdown: most of it is not backend memory stalls.
   Next measurement: stall_backend, stall_frontend, inst_retired and op_spec on the same cells,
   to see where the remaining ~75% of extra cycles go.

#### Round 2: where the steady target's extra cycles go

Same 12 steady cells, events cycles, stall_backend, stall_backend_mem, stall_frontend,
inst_retired, op_spec (`counters_stall_session{1,2}.json`, `analyze_stall.py`). The target CPUs
also count their wait after the target finishes. It is removed using the per-ms rates of the pool's
idle CPUs in the n0 cell.

| cell (background) | slowdown | extra cycles M | backend, not L2-miss memory | backend memory | frontend | unstalled | instructions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P2 8T M=2048 (steady) | x1.32 | 106 | 84% | 14% | -4% | 5% | +0.0% |
| P2 16T M=2048 (steady) | x1.52 | 217 | 78% | 9% | -4% | 17% | +13.5% |
| P2 4T M=1024 (steady) | x1.79 | 138 | 77% | 8% | -4% | 20% | +16.7% |

- The 8T row is clean. With the same instruction count, 84% of the extra cycles are backend
  stalls that are not waits on L2 misses (the architected stall_backend_mem). IPC falls 4.62 ->
  3.34 at an unchanged clock (about 3.2 GHz on every CPU). Alone the target has 21% backend
  stalls, of which 3% are memory.
- The 16T/4T instruction increases are probably residue of the wait correction.
- The per-thread time inside stage phases stays at 0.97-1.00 of the span, so the slowdown is not
  waiting inside the team.
- P3 (loading background) cannot be split this way. After the correction, the target shows
  +106-216% instructions and IPC 6.4-9.0, which is impossible. The waiting workers evidently
  behave differently while many small background tasks are dispatched, so the idle-CPU rates do not
  transfer. Measuring P3 needs a target that stays busy for the whole call.

Reading (hypothesis, not verified): the steady slowdown is a shared compute-side limit, not a
memory one. A steady background slows the target more than a loading background at 16T and 4T
(x1.52 vs 1.28, x1.79 vs 1.40), with no clock change and mostly non-memory backend stalls. That
fits core power management throttling matrix-instruction dispatch when many cores run BFMMLA, not
a bandwidth share. Neither the additive composition nor the offered-rate fix models this. In the
model it would be a gemm_core_flops service curve that saturates with concurrently computing cores;
today that curve is linear in threads. Verification: target alone vs a background running only
register-resident BFMMLA loops (no memory traffic) vs a background of scalar spin.

#### Round 3: register-only BFMMLA background (the throttling hypothesis is rejected)

Design `tmp/c9g_event_model_20260924/bgmicro_design.md`, code `bgmicro.c` and `bench_bgmicro.py`,
data `bgmicro_session{1,2}.json`. The target runs continuously on CPUs [0, t), so no wait
correction is needed. The background (`bgmicro`) runs on the next n CPUs in one of four modes:

- none;
- spin (dependent scalar adds);
- bfmmla (12 register-resident SVE BFMMLA accumulators, no memory traffic; 8.0-8.8e9 BFMMLA/s
  per core in the background, 9.3e9 with 4 threads);
- stream (SVE loads over a private 256 MiB buffer per thread; 3.2-3.6 GB/s per thread, about
  290-315 GB/s over 88 cores).

Targets are the fused kernel's single steady expert (full stripe, 32 rotating experts) and the
bfmmla loop itself. Each combination runs 2.5 s with perf on the target CPUs; two sessions,
shuffled, 120 s cooldown. Target time over the no-background time:

| target | spin | bfmmla | stream | IPC none -> stream | backend stall none -> stream (memory part) |
| --- | --- | --- | --- | --- | --- |
| kernel 8T M=2048 (n=88) | x1.000 | x1.000 | x1.225 | 4.66 -> 3.95 | 22% -> 33% (1% -> 4%) |
| kernel 16T M=2048 (n=80) | x1.006 | x1.006 | x1.269 | 4.47 -> 3.71 | 25% -> 37% (2% -> 4%) |
| kernel 4T M=1024 (n=92) | x1.007 | x1.007 | x1.433 | 4.52 -> 3.26 | 24% -> 45% (1% -> 5%) |
| bfmmla loop 8T (n=88) | x1.038 | x1.035 | x0.997 | 3.35 -> 3.36 | 15% -> 17% |

By the frozen reading, the compute-throttling hypothesis is rejected. 88 cores of back-to-back
BFMMLA do not slow the kernel at all (<= 0.7%), and slow a pure BFMMLA target by only 3.5%. A pure
memory-streaming background, with no matrix instructions, reproduces the steady slowdown
(x1.22-1.43, against P2 x1.32-1.79 and P3 x1.28-1.40) at an unchanged 3.2-3.3 GHz clock. The
slowdown is memory-side. It shows up as backend stalls that the architected stall_backend_mem
does not count; on Neoverse that event counts only stalls with a demand load pending in the L2.
The target's own traffic is small (8T: 22 GB/s), and a target with no memory traffic is not
slowed. So the mechanism is latency or queueing on the path the kernel's A reads and C writes
use when the memory system is saturated, not a bandwidth share of the target's own bytes. This is
why neither max() nor sum composition of the target's own transfer time reproduces it.

The model term that follows: a steady phase's time grows with memory-system utilization by its
latency-exposed accesses (A refills, C write-backs), a queueing-latency term with no bandwidth
share. Which store/load path stalls needs the V3 implementation events (raw codes, for example
store-buffer or write-back stalls), which sysfs does not list.

#### Round 4: which path stalls (Neoverse V3 backend-stall breakdown)

Data `bgmicro_stall_session{1,2}.json`, analysis `analyze_bgstall.py`, design
`bgmicro_design.md` round 2. Targets: the kernel's steady expert at 8T/16T (M=2048) and 4T
(M=1024), full stripe, alone or with the stream background. Three event groups, raw architected
V3 codes checked to count. STALL_BACKEND_L2D, _ILOCK and L2D_CACHE_REFILL_WR read 0 on this core.

| target | slowdown | extra backend stall / extra cycles | MEMBOUND | CPUBOUND | L1D-pending load | L2-miss-pending load (MEM) | store not committed (ST) | issue queue full (BUSY) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8T M=2048 | x1.23 | 88% | 88% | 1% | 70% | 18% | 0% | 74% |
| 16T M=2048 | x1.27 | 85% | 81% | 4% | 66% | 16% | 0% | 76% |
| 4T M=1024 | x1.45 | 91% | 91% | 1% | 78% | 13% | 0% | 84% |

Columns are each event's added cycles as a share of the extra cycles; the events overlap (BUSY is
the issue-queue view of the same load waits). The target's own traffic does not change under the
background. Bus reads equal L2 refills (BUS_ACCESS counts 32-byte beats, `caps/bus_width` 32):
8T 268 -> 266 MB, 16T 415 -> 411, 4T 393 -> 414 MB per call. L2 victim write-backs stay at 95-360
MB per call.

- The stalled path is the load path. Nearly all extra cycles are memory-bound backend stalls on
  demand loads that missed L1 and are waiting on refills; the issue queues fill behind them.
- It is not the store/write-back path (ST 0%), not a CPU-bound limit (CPUBOUND unchanged), not TLB,
  and not rename.
- The target moves the same bytes; each refill just takes longer.
- In an 8T M=2048 expert those bytes are mostly A. N-split makes every thread read all of A:
  16.7 MB x 8 threads for W13, about 134 MB, plus W2's A. The weights are 12.6 MB.
- The counters cannot say whether the refills turn slow because SLC hits get slower in a saturated
  mesh, or because the background's streams evict A from the SLC so the refills go to DRAM. Only
  18% of the extra cycles are loads pending beyond L2 (MEM). The L1D-pending share is 70% and
  STALL_BACKEND_L2D is not implemented, so the split between L2-hit and L2-miss waits is not
  resolved.

Model consequence: the steady-phase contention term is a latency term on the phase's A refill
traffic (the model's `a_l2_refill_bytes`, which already scales with team width and windows). Its
latency grows with memory-system utilization, independent of the target's own bandwidth share:
steady time += A-refill lines x exposed fraction x extra latency(utilization). It does not apply to
C writes or write-backs. The utilization-to-latency curve and the exposed fraction can be
calibrated on the stream-background cells (target alone vs stream background, 3 widths). The
P2/P3 probes and the whole plans then serve as validation.

### Lab: refill-latency term (frozen variant rejected)

`latency_term.py` on top of the window model. Every GEMM phase gets extra time
refill lines per thread x C_SAT x min(1, U). The lines are A refills plus B refills, not C
writes. U is the other concurrent phases' average-rate DRAM or L2 traffic over whole-rank
capacity.

The first frozen form used A refills only. It gave inconsistent per-line constants on the
calibration cells, so it was redefined to all read refills before any validation. C_SAT = 6.22 ns
per line at saturation (median of 2.01 / 6.22 / 9.43 at 4T / 16T / 8T). The spread comes from the
model's refill accounting, not the mechanism: the model gives 2.98M / 0.31M / 0.32M lines per
thread, the counters 1.58M / 0.40M / 0.52M. With measured lines the constant is 2.9-4.2 ns.

Validation (`latency_probes.log`, `latency_plans.log`) rejects this form:

| set | window | + latency term | measured |
| --- | --- | --- | --- |
| D_SS 8T / 16T / 32T at max n | 1.01 / 1.00 / 1.29 | 1.17 / 1.33 / 1.95 | 1.16 / 1.24 / 1.46 |
| D_SS 4T n=92 | 1.77 | 3.52 | 1.51 |
| D_LL 8T n=8 / n=88 | 1.32 / 5.25 | 2.24 / 6.58 | 1.15 / 5.01 |
| traces 8T / 16T / fast | -12 / -10 / -13% | +26 / +20 / +33% | |
| tuning median fast / homog 8 / 16 / 32 | -17 / -15 / -15 / -16% | +24 / +22 / +16 / +8% | |
| width pick / family pick | 10/18, 13/18 | 1/18, 5/18 | |

It fixes the 8T/16T steady probes (D_SS 1.17 / 1.33 vs 1.16 / 1.24) and breaks everything else.

- Applied to loading phases, it charges latency on top of the multi-stream DRAM share, which
  already reproduces loading contention, so loading is counted twice. D_LL 8T at n=8 becomes 2.24
  vs 1.15.
- U_llc counts every loader's weight stream as mesh traffic, so it saturates at a few background
  lanes.
- 2T/4T full-stripe steady phases carry the model's inflated B re-read counts (1.9x the
  counters), so D_SS 4T becomes 3.52 vs 1.51.

A second form would apply the term to steady phases only, take U from DRAM-side traffic only, and
fix the refill accounting of full-stripe narrow widths first. Those choices are informed by this
validation, so a clean test needs data this form has not seen, for example new probe cells or
layers.
