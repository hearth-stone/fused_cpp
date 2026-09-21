# Foreground interference with a 1MiB-windowed M120 background

## Decision

The1MiB background window strongly reduces foreground interference in both
sessions. Real W13 latency drops from454.47/453.43us to303.45/303.76us,
paired33.33/33.01% lower. Loaded4 drops from452.63/451.35us to302.62/302.41us,
paired32.98/33.18% lower. All four foreground flavors pass the predeclared
relief gate in both sessions. The former positive large-versus-small-background
penalty is absent with this windowed background.

Foreground LLC read-miss events fall from~99k to~2.4k, L2 refills remain~130k,
and aggregate DDRC read traffic falls from~44GB/s to~9GB/s. This is strong
intervention evidence for background reuse/cache-supply effects, not proof of
a unique L3 latency term: DDR activity, overlap and background phase also change.
No model fit or production adoption follows from this experiment.

## Scope and protocol

Class E Lab diagnostic. Production kernels, defaults, planner, schemas and
frozen response model are unchanged. Background mode5 changes only the Lab
M120 loop order. Each of16 independent1T background workers still processes
all120 rows and all8MiB B, rotating four copies after a complete M120 task.

Original mode3: ten M12 calls, each traverses the full B stripe.
Window mode5: eight N windows, each contains8 N16 tiles (N128, K4096, BF16,
1MiB B); process all ten M12 panels before advancing to the next window.
There are80 kernel calls instead of10. Output base is
`row*512 + begin*6` BF16 elements, covering the full W13 packed output once.
Windowing changes reuse distance, call overhead, background throughput and
request timing together; this is not a pure intervention on L3 alone.

Each background's W13 stage is8MiB B, full owner stripe8MiB; window tuple
`(t=1,w13_window_tiles=8,w2=not_measured,R13=8,R2=not_measured)` versus
`w13_window_tiles=0`/`full_n_team_stripes`,R13=1. No W2 or production profile
is produced. The Lab external N-window loop is not production calibration.

Fixed foreground M1/1T K4096/N1024,8MiB B: real W13, AB load-only, loaded4
no-store, resident4 with identical loads. No timestamp instrumentation or
additional foreground warmup. Background conditions: none,16M1,16M120-full,
16M120-window.20 randomized cells per round; two independent sessions,
5 warmups +31 measured rounds, seeds229808/239808. Same process/allocations,
persistent touched workspace,256MiB scrub and4-copy B rotation.

Arm-codex-internal, NUMA3 memory, allowed CPUs240–319, controller240,
foreground304, backgrounds288–303. Same HiSilicon hardware as earlier tests;
ordinary aligned buffers and existing THP policy, not verified HugeTLB residency.
Counter groups: victim cycles/instructions/L2 refill/LLC read-miss/backend
stall, plus48 DDRC events. Uncore counters include background and have a
longer gate than the victim kernel. Background call counts include startup
and drain; they are not victim-window throughput measurements.

Meaningful relief is declared before measurement: window-minus-full paired
median time change <=-2%,95% upper bound<0, in both sessions. Report the
remaining window-minus-M1 gap separately. PMU running ratios >=0.99 and full
background output correctness must pass. No adoption or model fitting.

## Reproduction

Remote root `/home/zhangxu/codex/fused_cpp`; snapshot/raw directory
`tmp/background_window_20260908/`, also retained locally (ignored).
Build uses the C++17/O3/pthread/SVE256/BF16 recipe from
`l2_matrix_mix_20260908.md`, substituting this directory. Run one-round
no-PMU and PMU smoke before formal collection; retain distinct output files.

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/background_window_20260908/bench_phase_supply.py \
  --binary tmp/background_window_20260908/phase_supply_native \
  --output tmp/background_window_20260908/session1.jsonl \
  --background-window --seed 229808
```

Repeat with session2/seed239808.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_background_window.py \
  --sessions tmp/background_window_20260908/session1.jsonl tmp/background_window_20260908/session2.jsonl \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --output tmp/background_window_20260908/report.json
```

## Validation

Local targeted tests46 passed: background-window, supply-timeline, penalty-trigger,
phase-supply, pressure-curve andAB-supply-contrast. Includes exact120×512 output
coverage and20-cell geometry. Ruff passes and C++ formatted Google120.
Native build,20-cell no-PMU/PMU smoke, old-protocol smoke and both formal
sessions pass. Every background worker completes and verifies all120×512
output elements; the window marker is8 N16 tiles only for mode5. Generated
production B/full-no-store4 identity, foreground numerical/poison checks,
complete grids and PMU running ratios pass. No production E2E test was run.

## Foreground latency

Absolute medians in us, S1/S2. Only the background changes; foreground is
uninstrumented and executes the same full8MiB weight access.

| Foreground | No background |16M1|16M120 full stripe|16M120 1MiB window|
| --- | --- | --- | --- | --- |
|Real W13|301.14/301.03|430.61/430.16|454.47/453.43|303.45/303.76|
|Loaded4 no-store|299.45/299.96|422.31/426.13|452.63/451.35|302.62/302.41|
|Resident4 + loads|see report.json|430.39/426.84|455.39/452.05|302.64/302.93|
|Load-only|see report.json|403.26/404.05|384.64/383.80|297.22/296.04|

Paired window-minus-full real-W13 deltas are-151.12/-149.23us,95% intervals
[-152.60,-149.14]/[-153.64,-147.35]us. Loaded4 deltas are-149.81/-149.44us,
intervals[-151.18,-147.21]/[-151.54,-145.55]us. Relative to M1 backgrounds,
windowed M120 makes loaded4 faster by27.99/28.73%; it is no longer the harmful
large-background condition. Relative to no background, loaded4 still has
0.786/0.712% median slowdown and real W13 has0.792/0.787%: near-isolated,
not literally zero interference. Paired medians need not equal differences
of independent medians.

## Counter evidence

Loaded4 foreground, full-stripe versus windowed M120 background (cell medians):

| Metric | Full, S1/S2 | Window, S1/S2 |
| --- | --- | --- |
|L2 refill events|129,936/129,958|129,845/129,910|
|LLC read-miss events|99,000/98,723|2,410/2,361|
|Backend stall cycles|977,313/976,460|545,188/545,169|
|Domain27 DDRC read GB/s|44.13/44.31|9.14/9.06|
|Domain27 occupancy/read-command ratio|41.57/42.18|23.00/23.05|

No-background loaded4 has2,197/2,496 LLC read-miss events and9.08/9.01GB/s
domain27 DDRC reads, close to the window-background levels. Paired LLC
read-miss reduction is95,917/96,560 events; its95% intervals exclude zero.
L2 refill paired changes are-161/+60, with intervals crossing zero.

Thus the intervention preserves foreground L2 refill volume but substantially
changes the observed lower-cache path and external traffic. It does not
establish exact LLC-hit/DRAM byte accounting, or independently measure the
background's own L2-hit fraction. A1MiB allocation size alone does not prove
all background accesses hit its1.25MiB L2 when A/C and other state coexist.

## Limits and follow-up boundary

Both M120 variants complete exactly one full task per background worker per
cell (counts include the post-victim drain), while M1 repeats more frequently.
The foreground is sampled after a fixed5ms background lead-in. This is the
same original protocol, but not a phase-randomized or long-steady-state study;
windowing changes where the background is in its task at measurement time.
The measured relief is valid for this protocol, not a universal bound over
all background phases. No background throughput gain is claimed.

Keep full-stripe and windowed M120 as separate benchmark contexts. Earlier
unwindowed-background anomalies must not be generalized to windowed production
plans. A next validation can vary the foreground arrival relative to window
progress while keeping the1MiB window fixed. Do not yet add an L3 penalty,
declare1MiB globally optimal, or change production/planner defaults.

## Provenance

HEAD `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus scoped dirty Lab changes.
Raw sessions, three smoke records and `build_inputs.tar.gz` retained in the
local/remote ignored snapshot directory. `report.json` includes full latency
distributions, paired counter differences, relief gates and all baselines.

| Identity | SHA256 |
| --- | --- |
|Binary|`5690eb2ef039b4c28e212a493071f8b2bde97dc0575567a302d39e84b7639c54`|
|Production JIT source|`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`|
|Frozen fit|`b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781`|
|Session1|`b6d6bfa970ad754eb68d49ad2a383a89659c02ce447d76b80ecbafa50a1a097e`|
|Session2|`246bca845e6f33e02a8b12bd849058f1076177ccac886cf3d14c2909e7afbf6c`|
