# Per-case diagnosis of full-MoE absolute-error regressions

## Answer and scope

Completed model interventions, four new trace sessions and two untraced controls
for all eight worsening cases: seven uniformish entries (anchor,p01–p06) and
VND high-skew p11. High-skew anchor is an additional control. No coefficients,
current adapter, production model or calibration changed; no refit or commit.

The main diagnosis differs by case:
- Most uniformish entries expose missing explicit gather/non-stage/outer costs
  after GEMM overprediction was reduced. Their current GEMM estimates are still
  above measured stage envelopes; restoring a GEMM penalty would compensate
  other costs rather than identify a genuinely missing GEMM stall.
- Uniformish p01 additionally exposes order-dependent interference: unchanged
  lane membership and core count, but reordering other lanes slows an unchanged
  lane. Core count alone does not express the resulting time-varying context.
- VND p11 moves M65 to a1T lane, making it critical. The current model identifies
  that lane correctly, but underestimates actual1T W13 response, especially M13
  after its execution is delayed into a different background context.

These local explanations do not negate the previous overall improvement. The
uniformish MAE increase is0.552ms (~1.8% of a30ms forward); about0.510ms of that
net increase comes from p01+p03. No cases are removed from official scoring.

## Method, runtime and validity

Target remains Arm-codex-internal, `/home/zhangxu/codex/fused_cpp`, NUMA3,
CPU240–319/membind3. Reuse exact Plan V2 bridges and existing stage-trace runner
`tmp/workspace_phase_timeline_20260907/runner/bench_bounded_order_extension.py`,
SHA256 `ea81113c747c509b8556d5c9a698fd2bf85a8d01a4c8fc775ab9657f107219ba`.
Workspace helper SHA256
`7e0124673b4742a4747f6b4b733cb364def2f455cc5919ef13379eca4cb098db`.
Frozen extension and calibration match the parent full-MoE replay. No rebuild.

Same E256/H4096/F512,2048tokens,TopK6,BF16/SVE256,Ntile16, full stripes,
W13/W2 bytes8MiB/4MiB per expert, resident192MiB output workspace,4 weight copies,
216MiB scrub. Two trace sessions per sample use seeds590921/590922,5 warmups and
31 measured paired rounds. One no-trace control per sample uses seed590921 and
the identical selected bridges/order protocol. OMP_NUM_THREADS=1,
OMP_DYNAMIC=FALSE,OMP_PROC_BIND=false,MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1;
manual runtime CPU ownership and full controller affinity remain unchanged.

Initial startup failures are retained: main source runner path was absent on
the remote mirror, then OMP_PROC_BIND=close narrowed the Torch-importing main
thread and triggered the original affinity guard. Reuse the preserved runner
and disable automatic OpenMP binding; verify240–319 after importing Torch.
No guard was relaxed, old snapshot overwritten, or machine substituted.
Successful artifacts use `uniformish_run2`/`high_skew_run2` prefixes.

The parser validates trace hashes, call order, correctness prefix, all task
expert/worker identities, three stages, and complete31-round traces. Four trace
sessions cover558 measured calls; two control sessions cover279. Controls also
pass full grid/copy/order/frontier/extension/finite-time checks. All successful
runs complete; there are no active processes to hand off.

Stage envelopes are earliest worker start to latest end, relative to scheduled
compute origin. A lane's stages are summed per call, then median taken. The
remaining lane interval includes initial dispatch and gaps not attributed to
those envelopes; it is NOT a uniquely identified barrier/queue cost. Outside-
expert time is native e2e minus final W2 endpoint, not a measured merge-only cost.
Quantiles are taken separately, so rounded table columns need not sum exactly.

Trace outer elapsed includes log serialization: uniformish trace-on Python
elapsed is roughly39–42ms, versus untraced29–33ms. These times are NEVER used to
refit the model or replace the original accuracy labels. Internal native trace
e2e differs from the contemporaneous untraced control by about-0.50 to+0.02ms;
this bounds observed endpoint differences, not proof of unbiased instrumentation.
Phase attribution is diagnostic; hardware LLC/DDR queue causality is not proven.

## All regression cases

Original trace-off signed errors below remain unchanged. New trace observations
explain them; their native totals are a separate measurement regime.

| Case | Original old → current signed error ms | Actual final lanes | Diagnosis |
| --- | ---: | --- | --- |
| Uniformish anchor | +1.164 → -1.535 | lane56 in61/62 trace rounds | Current model fixes old lane40→actual56 critical-lane error, but omits explicit gather and outer/gap costs. GEMM is still overpredicted. |
| Uniformish p01 | +0.202 → -3.336 | mostly56 inS1;48/56/64 near-tied inS2 | Order-only change amplifies interference on unchanged lanes. GEMM estimates are close, leaving gather/gaps/outer costs exposed; original near-zero total error was cancellation. |
| Uniformish p02 | +1.043 → -1.211 | several lanes, mode32 | Common unrepresented-cost bias; current predicted lane56 is close to the measured finish (about0.21ms slack inS2), so do not overinterpret a unique critical-lane label. |
| Uniformish p03 | +0.638 → -2.092 | lane56 all31 S1;56/64/48 inS2 | Common omitted costs plus a false lane40 critical prediction; actual lane40 finishes about1.83ms early inS2. |
| Uniformish p04 | +0.933 → -1.710 | lane56 all31 S1;56/48/64 inS2 | Common bias after4/8T regrouping. Predicted lane64 is near the measured finish (~0.10ms slack inS2); this is mostly a time-accounting gap, not evidence of a large lane-choice error. |
| Uniformish p05 | +0.910 → -1.742 | lane56 all31 S1;56/64/48 inS2 | Common bias and false lane32/M429 tail. That lane has about2.02ms slack inS2; the unchanged8T lane56 remains dominant. |
| Uniformish p06 | +1.146 → -1.292 | lane56 in56/62 rounds | Current model moves critical lane56→16, while hardware retains56; lane16 has~1.37ms slack inS2. Absolute deterioration itself is small (~0.15ms). |
| VND p11 | -0.656 → -1.971 | lane76, expert242,62/62 | M65 relocation makes a1T chain critical. Current model identifies it; old predicts16T lane48. Actual W13 is underpredicted, especially M13 in its later execution context. |

Source code of the current adapter explicitly retains no gather phase when the
frozen calibration has gather_pressure.enabled=false. Wide uniformish critical
lanes have no operator-residual phase either. Other unmodeled costs could have
been absorbed by inherited GEMM calibration; the traces expose cancellation,
not permission to add every observed envelope as a universal fixed penalty.

## Uniformish phase accounting on actual critical lanes

Session2; hardware phase sums inms, compared to current model on the SAME lane.

| Case | Gather | Initial/non-stage lane gaps | Outside expert interval | Actual W13 / W2 | Current W13 / W2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| anchor | 1.287 | 0.509 | 0.937 | 18.119 / 8.718 | 19.040 / 9.386 |
| p01 | 1.065 | 1.013 | 0.890 | 19.752 / 9.628 | 19.542 / 9.639 |
| p02 | 1.022 | 0.336 | 1.426 | 17.793 / 8.732 | 18.844 / 9.355 |
| p03 | 1.164 | 0.671 | 0.912 | 18.397 / 8.956 | 18.982 / 9.379 |
| p04 | 1.225 | 0.554 | 0.958 | 18.493 / 8.852 | 19.187 / 9.475 |
| p05 | 1.213 | 0.541 | 0.957 | 18.360 / 8.893 | 19.167 / 9.456 |
| p06 | 1.200 | 0.505 | 0.942 | 17.992 / 8.677 | 18.909 / 9.372 |

The typical2.7ms observed non-GEMM/gap/outer amount is partly offset by current
GEMM overprediction of~1–1.6ms. Old GEMM overprediction was larger, so it could
produce a closer total without more accurate stage accounting. This corrects
the earlier model-only suggestion that restoring W13 would identify a missing
W13 hardware cost: it mainly restores compensation in these uniformish cases.

### p01: ordering changes background effects on an unchanged lane

p01 preserves every lane's expert membership, width and routes sum. Relative to
anchor, lanes0/16/32/48/64 change from small-M-first to large-M-first; other lanes
(including lane56) keep their exact task sequence. Yet lane56 slows:

| Lane56, session1 | Anchor | p01 | Change |
| --- | ---: | ---: | ---: |
| Actual W13 sum | 18.133 | 19.792 | +1.659ms |
| Actual W2 sum | 8.731 | 9.599 | +0.868ms |
| Actual lane end | 28.567 | 31.295 | +2.728ms |
| Current model lane end | 28.426 | 29.146 | +0.720ms |

The case changes concurrent stage arrival/composition and state history even
though task counts and per-lane total work are identical. This is an observed
interference effect of ordering, not proof that one named LLC queue is causal.
The model underexpresses this temporal dependence. p01 remains predicted slower
than p06, so it is not an observed false winner in this sample.

## VND p11: exact moved expert and failing task response

The only relocation is expert227,M65, from1T lane68 to1T lane76. Lane76 grows
from11 experts/135routes to12 experts/200routes, placing M65 after its initial
M35/M31 tasks. Other assignments/orders remain as recorded.

Actual lane76 is final in all62 traced p11 calls. Current prediction chooses its
terminal expert242,M2; old model still chooses lane48/expert171,M1239. Thus old
absolute proximity masks a wrong critical chain. On the original untraced data,
anchor→p11 increases6.213ms; old predicts0.047ms, current predicts2.782ms.
Current relative effect is better despite worse p11 pointwise absolute error.

Actual critical-chain W13 sums23.182/23.020ms versus predicted20.132ms. W2 sums
11.018/10.957ms versus10.024ms. Current operator residual3.870ms partly masks
these stage underpredictions; it is not an explicit measured gather component.
The largest W13 misses on that chain:

| Expert / M | Actual W13 S1 / S2 ms | Current predicted ms | Missing S1 / S2 ms |
| --- | ---: | ---: | ---: |
| 56 / M13 | 2.878 / 2.802 | 1.698 | 1.180 / 1.103 |
| 44 / M35 | 3.999 / 3.995 | 3.443 | 0.556 / 0.552 |
| 227 / M65 | 6.775 / 6.784 | 6.314 | 0.461 / 0.470 |
| 108 / M17 | 2.005 / 2.002 | 1.761 | 0.244 / 0.241 |
| 29 / M31 | 3.285 / 3.265 | 3.061 | 0.224 / 0.204 |

M13 has a direct same-expert context contrast. In anchor it starts14.161/14.165ms
and W13 takes1.836/1.840ms. After M65 is inserted, it starts25.269/25.076ms and
W13 takes2.878/2.802ms. The model predicts the OPPOSITE response: W13
1.895ms inanchor→1.698ms inp11. This localizes a time/context response failure,
not just an inaccurate constant isolated M13 cost. Whether it is tail-panel
reuse, path composition, outstanding concurrency or another mechanism needs a
separate controlled probe; full-stage trace cannot distinguish those causes.

## Contemporaneous no-trace controls

Same selected plans/runner/workspace/seed590921; full Python elapsed medians:

| Case | No-trace ms | Trace internal native S1 / S2 ms |
| --- | ---: | ---: |
| uniformish anchor | 30.030 | 29.526 / 29.615 |
| p01 | 32.586 | 32.170 / 32.429 |
| p02 | 29.703 | 29.321 / 29.423 |
| p03 | 30.494 | 30.088 / 30.253 |
| p04 | 30.296 | 30.069 / 30.293 |
| p05 | 30.321 | 30.161 / 30.110 |
| p06 | 29.441 | 29.141 / 29.402 |
| high-skew anchor | 30.177 | 30.085 / 30.175 |
| high-skew p11 | 36.416 | 36.431 / 36.238 |

Native endpoints and Python elapsed are not identical, and separate processes
add variation. No claim of zero trace perturbation, exact overhead subtraction,
or a new hardware winner. Large p01/p11 deficits persist in untraced controls.

## Artifacts, reproduction and unchanged decision

Remote and local ignored root:`tmp/core_pressure_case_trace_20260909/`.
Raw `.trace`, compressed copies, per-session JSON/stdout/stderr, exact selected
frontiers, `case_summary.json`, `model_summary.json`, `p11_task_summary.json`,
`p11_context.json` and `trace_sensitivity.json` retained. Initial failed outputs
remain remotely. Runner and baseline profile remain in the validated historical
snapshot; no source copying/rebuild was needed.

From the remote repository root, use this template for each sample and seed,
with fresh output/trace names:

```sh
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=false MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/workspace_phase_timeline_20260907/runner/bench_bounded_order_extension.py measure \
  --frontier tmp/core_pressure_case_trace_20260909/uniformish_run2.json \
  --route-file bench_assets/moe_paper/dsv4_routes_pt_20260830/measured_request022_case023_zh2048-024.pt \
  --experiment-baseline tmp/workspace_phase_timeline_20260907/workspace_numa3_80c.json \
  --workspace-max-tokens 2048 --max-plans 7 --seed 590921 \
  --phase-trace tmp/core_pressure_case_trace_20260909/uniformish_run2_session1.trace \
  --output tmp/core_pressure_case_trace_20260909/uniformish_run2_session1.json
```

Session2 uses590922. High-skew uses its selected frontier and its preserved
`route_file`; controls omit `--phase-trace` and use new `_control.json` names.
Local diagnostic scripts (exclusive outputs; choose fresh paths on rerun):

```sh
.venv/bin/python tmp/core_pressure_case_trace_20260909/analyze_cases.py
.venv/bin/python tmp/core_pressure_case_trace_20260909/model_cases.py
.venv/bin/python tmp/core_pressure_case_trace_20260909/p11_tasks.py
.venv/bin/python tmp/core_pressure_case_trace_20260909/p11_context.py
```

They reuse current repository trace validators and unchanged baseline/current
models; no new fitting. Current model, frozen coefficients and original official
error table remain intact. The requested per-case diagnosis is complete at the
phase/task/context level. Hardware micro-mechanism identification remains open;
no attempt to force all cases into a single LLC penalty was made.
