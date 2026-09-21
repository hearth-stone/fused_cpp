# Isolated team demand and equal-core contention experiment

## Decision

Completed two independent PMU sessions and one no-PMU control. Independent M12
team demand weighting improves cross-width foreground prediction over expert
count, but does not improve over occupied-core count in this grid. Neither
weighted demand nor core count passes all predeclared width gates.

Retain a bounded Lab reference:1T/2T/4T standalone demand scales approximately
1/1.93/3.65 for DDR and1/1.97/3.79 for mapped HHA requests. This supports counting
wider teams as heavier competitors. It does not establish exact uniform LLC/MEM
allocation, fix the full small-M model, or justify production adoption.

Class E Lab experiment. New native harness, runner, analyzer, tests, report and
one manifest entry are the rollback boundary. Production JIT, planner,
calibration and default behavior unchanged. Existing dirty/untracked assets
preserved; no commit. The earlier source-transfer rejection was resolved by
explicit user approval before scoped sync/build/run.

## Protocol and scope

Target Arm-codex-internal, root `/home/zhangxu/codex/fused_cpp`, NUMA3. Controller
CPU240; fixed foreground M1/1T on304; background CPU prefix280–319 excluding304,
at most38 cores. Exact-M production JIT W13 only: K4096,N1024,BF16/SVE256,
backend N16, full8MiB B owner stripe, window_tiles0,R13=1. W2 is unmeasured.
Background M12 teams have widths1/2/4, per-thread B owner stripes8/4/2MiB.

Each team shares one of four8MiB B copies among its lanes, partitions N and
synchronizes at call boundaries before jointly rotating copies. Different teams
have independent weights. Foreground allocation is fixed regardless of team
count. Constant BF16 input1/64 uses existing W13 expected output0x3f3b; every
live output element is checked after each cell. Affinity failure is fatal.
This reuses kernel services, not a production MoE operator or team scheduler.

New steady-state protocol: persistent allocations, at least5ms native lead-in
before ARMED, continuous execution across controller handshakes,100ms observation.
No scrub or cold-single-call claim. Include only calls wholly inside the native
interval; report each team's median/min/max call time and count. Model target
is the median of31 per-cell foreground call medians. New barriers/timestamps
are part of Lab execution. Do not equate their overhead with production or mix
these results into earlier cold-M1 datasets.

PMU windows include GO/DONE control and worker drain, exclude verification and
JSON output. Use each counter's enabled duration, never foreground kernel time,
as rate denominator. All observed running/enabled ratios are1.0; individual
enabled windows span100.080–103.523ms. Measured
team times include the kernel completion barrier, but not every between-call
barrier; do not derive exact task throughput by reciprocal of these medians.

Grid has19 cells: idle, foreground-only, three isolated M12 teams; foreground
plus background core totals4/8/16/32 for each width, plus1T*38 and2T*19.
Cell order randomized every round. PMU sessions seeds590901/590902, each5 warmup
and31 measured rounds:684 cells each, including95 warmup and589 measured.
No-PMU control uses seed590901 in a separate process, same grid/rounds.
Two one-round smoke grids precede formal runs; smoke is not performance evidence.

System Linux5.10.0-247.0.0.146.oe2203sp4.aarch64, GCC13.2.0, C++17/O3/pthread,
armv8.2-a+bf16+sve, SVE256. THP policyalways, ordinary vector allocations;
no per-allocation huge-page guarantee. No new dependency or global build change.

## Independent demands and frozen response

Primary solo demand: summed16 DDRC flux_rd rates,32bytes/event. Auxiliary local
source proxy: summed four group27 HHA rx_sccl rates, events/ns=Gevents/s. HHA is
not direct LLC byte bandwidth or queue pressure. Unvalidated L3C is excluded.
Subtract session1 idle medians and normalize each width to solo1T. Foreground
is absent from demand calibration. Solo measurements are specific to the
steady-state four-copy policy; cache-path transfer is not assumed proven.

| Solo M12 width | DDR GB/s S1 / S2 | Team us S1 / S2 | Frozen DDR weight | Frozen HHA weight |
| --- | ---: | ---: | ---: | ---: |
| 1T | 5.56 / 5.53 | 1229.09 / 1229.54 | 1.000 | 1.000 |
| 2T | 10.74 / 10.76 | 615.24 / 615.41 | 1.933 | 1.971 |
| 4T | 20.28 / 20.56 | 308.75 / 309.24 | 3.651 | 3.787 |

Idle DDR S1=0.008664GB/s; idle HHA S1=0.000023326Gevents/s.
Raw solo DDR round CV is4.37–5.51% across widths/sessions. Similar session
medians do not establish universal demand constants for other M or kernels.

Predeclared response T/T0=1+a*x+b*x^2, a,b>=0, equal-condition squared relative
error. Four fixed candidates:
- tasks: x=team count/32;
- cores: x=team count*width/32;
- DDR: x=team count*solo DDR weight(width)/32;
- HHA: x=team count*solo HHA weight(width)/32.

Only session1 1T counts4/16/32 train the response, with foreground-only anchor.
All candidates have identical1T training features; no differing fit can hide
cross-width failure. Weights and parameters freeze before session2 is loaded;
`frozen_session1.json` saved before its retrieval matches the final model.
No weights/features/parameters adjusted after validation.

```text
T0 = 314.93 us
a = 0.28902964424300354
b = 0.4536242509371087
```

Hold out1T counts8/38, ALL2T/4T contention targets and all session2 targets.
Solo2T/4T calibration is separate from contention validation. Gate remains
MAPE<=5% and max<=10% PER slice. This smooth response tests demand transfer;
it does not fit a resource capacity C or implement a derived fair-sharing law.

## Frozen prediction results

MAPE / maximum absolute percentage error across condition medians:

| Slice / session | Expert count | Core count | Solo DDR weighting | Solo HHA weighting |
| --- | ---: | ---: | ---: | ---: |
| Held1T S1 | 4.63 / 7.33 | 4.63 / 7.33 | 4.63 / 7.33 | 4.63 / 7.33 |
| Held1T S2 | 4.08 / 7.71 | 4.08 / 7.71 | 4.08 / 7.71 | 4.08 / 7.71 |
| 2T S1 | 13.37 / 32.58 | 5.92 / 18.20 | 6.40 / 17.04 | 6.04 / 17.70 |
| 2T S2 | 13.70 / 32.44 | 6.42 / 18.57 | 6.81 / 17.41 | 6.45 / 18.07 |
| 4T S1 | 10.18 / 38.89 | 8.35 / 19.87 | 8.63 / 16.86 | 8.53 / 18.02 |
| 4T S2 | 10.13 / 39.41 | 8.82 / 20.18 | 9.08 / 17.16 | 8.98 / 18.32 |

Held1T passes, both wider-team slices fail in both sessions. Weighting reduces
large underprediction from expert-count scaling, but measured weights do not
beat core count in MAPE. Smaller maxima do not waive other failed gates.
These are prediction comparisons, not kernel speedups.

## Equal-core observations

Foreground M1 time, S1 / S2 us; rows compare the same background core count:

| Cores | 1T teams | 2T teams | 4T teams |
| --- | ---: | ---: | ---: |
| 4 | 319.41 / 319.10 | 318.64 / 317.83 | 317.39 / 317.01 |
| 8 | 322.93 / 321.81 | 321.58 / 321.38 | 324.61 / 322.73 |
| 16 | 402.50 / 417.15 | 335.16 / 334.10 | 330.48 / 329.64 |
| 32 | 546.31 / 555.70 | 549.31 / 558.21 | 567.24 / 572.06 |
| 38 | 612.66 / 627.32 | 621.97 / 620.68 | not divisible by4 |

At38 cores1T*38 and2T*19 are close: S2 ratio of medians is-1.06% for2T,
while S1 is+1.52%. No claimed winner. At32 cores the4T foreground is about3–4%
slower than1T despite lower aggregate DDR traffic (S2 159.64 versus208.21GB/s).
At16 cores grouping matters substantially. S2 foreground round CV is10.74%
for1T*16 and11.06% for2T*8; medians differ in both sessions, but this region is
not well described by a low-noise universal smooth count curve.

Changing grouping changes independent weight footprint: at16 cores, four-copy
background B footprint is512/256/128MiB for1T/2T/4T, respectively. This is a
possible cache-context factor, not independently identified causality. Team
synchronization/request arrival also differ. Do not add a fitted knee from
these now-scored residuals and call the same data untouched validation.

Same-width background teams have similar completion times: at38 cores S2
median team time1256.82us for1T and624.99us for2T; median per-cell slowest/fastest
team-time ratios1.024/1.011. This supports approximate within-type symmetry,
not direct evidence of equal per-team DDR/LLC allocation. M12 teams remain near
their solo times while the M1 victim slows; entire expert time must not be
scaled as if all compute were bandwidth-limited.

## PMU sensitivity and numerical checks

All five native runs (two smoke, two PMU formal, one no-PMU formal) exit0 and
have empty stderr. Formal calls pass numerical coverage and minimum complete
sample checks. Native binary is identical across sessions.

A separate-process no-PMU control, same seed/order asS1, gives foreground-only
314.95us,1T*38 602.75us,2T*19 613.83us,4T*8 565.74us. Across15 target conditions,
PMU S1/no-PMU median changes range-0.12% to+1.64%; S2/no-PMU -0.40% to+4.08%.
This is sensitivity evidence, not a same-process randomized causal overhead
estimate. It does not justify subtracting a correction factor. Full values in
`pmu_sensitivity.json`; no-PMU data does not alter training or selection.

Local5 direct tests pass: equal-core geometry, victim exclusion, demand weights,
synthetic coefficient recovery, held-width target isolation and counter windows.
Independent arithmetic verified120 predictions within1e-9us and recovered the
same response with direct least squares. Ruff and diff checks pass; C++ formatted
Google120. No production E2E, other M/width/stage, cold-start or mixed-kernel
validation. No production or physical resource-capacity claim.

## Reproduction and artifacts

All outputs under `tmp/team_demand_20260909/` remotely and locally. Remote retains
native binary, exact synced sources, raw JSONL and stderr. Local retains all
five JSONL files, frozen checkpoint, `report.json`, `report_details.json` (adds
team dispersion/demand variability without changing predictions), and PMU
sensitivity. Original outputs are never overwritten; use new filenames to rerun.

Native SHA256 `5a6094c0f20381f5fb448299c3efc7dbc8294a2d4638f18a2ec025d7a4849f81`.
Unchanged production JIT SHA256
`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
Raw S1 `ac26bc3c338bdea07666c23afbab78b6a125ac0a67eae83cc387c2346b9c2892`;
S2 `b99dfe09e70c4829ea97cc7f184505c5c72e2a2cd9829792081fdbb8e411e78b`;
no-PMU `a8f778e5721629b2591d801a1a61f5207b1cb2dad1dc402554f81678c7a34d91`.
Source basis is the current dirty workspace; no clean-commit build claim.

From remote project root:

```sh
g++ -std=c++17 -O3 -pthread -march=armv8.2-a+bf16+sve -msve-vector-bits=256 \
  -DFUSED_CPP_MOE_HAS_XBYAK_AARCH64=1 -DFUSED_CPP_MOE_SVE_VECTOR_BITS=256 \
  -Icsrc/moe/arm/sve_bf16 -Irefs/i8gemm \
  -I3rdparty/xbyak_aarch64 -I3rdparty/xbyak_aarch64/xbyak_aarch64 \
  tmp/team_demand_20260909/team_demand_native.cpp \
  csrc/moe/arm/sve_bf16/jit_kernels.cpp \
  3rdparty/xbyak_aarch64/src/xbyak_aarch64_impl.cpp \
  3rdparty/xbyak_aarch64/src/util_impl.cpp \
  -o tmp/team_demand_20260909/team_demand_native
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/team_demand_20260909/bench_team_demand.py \
  --binary tmp/team_demand_20260909/team_demand_native \
  --output tmp/team_demand_20260909/session1.jsonl --seed 590901
# Session2: output session2.jsonl, seed590902.
# No-PMU: output control_no_pmu.jsonl, seed590901, --no-pmu.
# Smoke: separate output, seed590900, --rounds 1 --warmup 0, optionally --no-pmu.
```

Local:

```sh
.venv/bin/pytest -q tests/test_moe_team_demand.py
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_team_demand.py \
  --sessions tmp/team_demand_20260909/session1.jsonl tmp/team_demand_20260909/session2.jsonl \
  --output tmp/team_demand_20260909/report_details.json
```

## Next decision

The requested first isolated-demand/equal-core trial is complete. Keep raw expert
count limited to fixed-width contexts; width-aware demand or core count is the
better initial cross-width hypothesis here. Do not silently replace the prior
working baseline or promote this failing response into the planner. Next work
can isolate the16-core grouping discrepancy and expand to different background
M/kernel types. Exact uniform allocation and foreground exposed-memory fraction
remain independent unvalidated steps, not completed by these throughput weights.
