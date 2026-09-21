# Streaming penalty trigger: fixed-A, B-footprint/preparation factorial

## Decision

Both independent sessions reproduce the original penalty only at8MiB B
coverage with loaded or register-resident matrix4, among the new factorial
cells. Neither first post-scrub execution nor direct load-to-matrix operand
dependence is necessary.128KiB/512KiB/2MiB do not trigger;8MiB load-only is
faster with the large-M background. Thus footprint alone, crossing the private
L2 capacity alone, and generic load-only slowdown are insufficient explanations.

This identifies a reproducible trigger within the measured grid, not universal
necessary/sufficient conditions or an exact capacity threshold. The transition
is observed between the tested2MiB and8MiB cases; monotonicity and intervening
sizes have not been tested. Keep the model frozen and preserve this counterexample.

## Question and predeclared interpretation

Does the large-M versus small-M background penalty require a large B working
set, the first post-scrub invocation, and coexistence of loads with matrix
execution? This is a bounded Lab diagnostic (class E), not a production or
cost-model change. No fitting. Original phase/response models remain frozen.

The stable-trigger criterion, declared before collection, is a round-paired
median large-minus-small slowdown >=2%, with95% bootstrap lower bound >0 in
both independent sessions. This is an operational criterion, not a physical
threshold. Failure to trigger does not prove exact equivalence.

Footprint and within-call reuse are coupled in this intervention: each call
loads8MiB of B, so128KiB/512KiB/2MiB repeat64/16/4 times, while8MiB is one
pass. The result identifies a coverage/reuse-distance regime, not footprint
capacity independently of reuse or miss-arrival timing.

## Intervention

- M1/1T, K4096/N1024 W13 packed geometry, SVE256 BF16; no W2.
- Fixed A addresses and coverage (64KiB line footprint); inner K loop retains
  production streaming addresses/instruction scheduling. B wraps only at N16
  tile boundaries over128KiB/512KiB/2MiB/8MiB. All variants add the same three
  integer instructions per N tile (192 per call); only the immediate mask
  differs. At8MiB the actual A/B load address sequence matches streaming.
- Three flavors: AB load-only, loaded matrix4, register-resident matrix4 with
  the same loads. Resident operands remove direct load-to-matrix RAW, not all
  shared execution effects. No output stores; not a complete numerical GEMM.
- Every flavor/footprint is measured after scrub, either with no victim
  preparation or after one complete selected-probe invocation on the victim.
  Prepared is an intervention label, not a claim of LLC/L2 residency; it also
  trains execution/prefetch state and advances ongoing background execution.
- None/16 M1/16 M120 backgrounds; retain real W13, streaming AB/loaded4/
  resident4, 12KiB loaded4 and192KiB loaded4 controls.93 cells per round.
- Same process/allocations/workspace;256MiB scrub; four-copy victim/background
  B rotation; background starts5ms before selected preparation/measurement.
  Two independent sessions,5 warmups +31 measured rounds; seeds169808/179808.
- Arm-codex-internal, NUMA3 memory, allowed CPUs240–319, controller240,
  victim304, background288–303. Same HiSilicon machine as L1/L2 experiments.
  Private L1D64KiB/L2 1280KiB; background has no SMT sharing with the victim.
  Ordinary aligned allocations, existing THP policy; no explicit verified
  HugeTLB residency. HEAD plus dirty Lab changes, not a clean-commit artifact.
- Six core counters (cycles/instructions/L1 refill/L1 access/L2 refill/backend
  stalls) plus48 DDRC counters. Running ratios >=0.99. DDRC gate is longer
  than the victim kernel; do not divide uncore counts by kernel duration.
  Current collection does not directly distinguish LLC hits from DDR refills,
  or demand from all prefetch behavior; no unique port/queue attribution.

## Reproduction

Remote root `/home/zhangxu/codex/fused_cpp`; isolated snapshot and raw in
`tmp/penalty_trigger_20260908/`, mirrored to the local ignored directory.
Build uses the same g++ C++17/O3/pthread/SVE256/BF16 recipe as
`l2_matrix_mix_20260908.md`, substituting the snapshot directory. Inputs must
include phase_supply_native.cpp, m1_supply_probe.h, bench_phase_supply.py,
linux_perf_event.py and the unchanged frozen_fit.json.

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/penalty_trigger_20260908/bench_phase_supply.py \
  --binary tmp/penalty_trigger_20260908/phase_supply_native \
  --output tmp/penalty_trigger_20260908/session1.jsonl --trigger-sweep --seed 169808
```

Repeat session2 with seed179808. Run one-round no-PMU and PMU smoke before
formal collection. Never overwrite an earlier raw session.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_penalty_trigger.py \
  --sessions tmp/penalty_trigger_20260908/session1.jsonl tmp/penalty_trigger_20260908/session2.jsonl \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --output tmp/penalty_trigger_20260908/report.json
```

## Validation

Local targeted tests:47 passed (`test_moe_penalty_trigger`, `test_moe_l2_matrix_mix`,
`test_moe_l1_matrix_mix`, `test_moe_phase_supply`, `test_moe_pressure_curve`,
`test_moe_ab_supply_contrast`). Ruff passes; native sources formatted Google120.
Target validation and results follow.

Native build,93-cell no-PMU and PMU smoke, old-protocol smoke and both formal
sessions passed. Production B-only/full-no-store4 byte-identity checks,
victim/background numerics, output poison, preparation/geometry, complete-grid
and PMU-running checks passed. Wrap variants add192 body instructions, with
observed medians within64 wrapper-instruction tolerance. No production E2E
test or cost-model fit was performed.

## Measured conditions

Large-M background minus small-M background: paired median delta in us,
S1/S2. Positive means the large-M background is slower. The original real
W13 control is +34.70/+27.18us; original no-store loaded4 is +33.37/+29.30us.

| B coverage | Preparation | Load-only | Loaded matrix4 | Resident matrix4 + loads |
| --- | --- | ---: | ---: | ---: |
|128KiB|none|+0.14/-0.98|+1.93/-0.14|-0.22/-3.09|
|512KiB|none|-6.64/-8.17|+3.13/-4.75|-1.22/-2.62|
|2MiB|none|-15.82/-19.02|-5.87/-10.98|-12.74/-12.24|
|8MiB|none|-22.96/-19.72|**+36.61/+33.91**|**+33.12/+30.41**|
|128KiB|one full invocation|-0.05/-0.21|+0.07/+0.16|-0.81/-0.24|
|512KiB|one full invocation|+1.42/-2.93|+0.60/+0.80|-1.53/+0.74|
|2MiB|one full invocation|-8.56/-6.53|-3.72/-6.33|-7.99/-10.55|
|8MiB|one full invocation|-8.43/-7.33|**+61.18/+53.17**|**+55.59/+52.95**|

All four bold factorial conditions pass the >=2%/positive-interval criterion
in both sessions; no other new factorial condition does. Original real,
loaded4 and resident4 controls also pass. L1/small-ring controls do not.

Absolute loaded4 medians, small/large background:

| Condition | S1 us | S2 us | Paired slowdown S1/S2 |
| --- | --- | --- | --- |
|8MiB, no preparation|417.97/452.13|419.71/452.78|8.64%/8.04%|
|8MiB, prepared|371.08/434.28|373.52/421.13|16.57%/14.29%|

The corresponding paired95% delta intervals are [31.36,40.61]/[28.65,38.47]us
without preparation, and [52.45,67.42]/[47.28,54.53]us after preparation.
Medians of paired differences need not equal differences of independent medians.

Preparation is not a universal speedup. Under small/large backgrounds it
reduces8MiB loaded4 time by paired48.15/20.31us in S1 and43.36/31.45us in S2.
Without background it instead increases time by12.64/17.90us. Therefore do
not explain preparation solely as a faster cache level.

As an exploratory interaction check, per-round
`(prepared_large-prepared_small)-(cold_large-cold_small)` has median
25.37us [20.05,31.51] in S1 and19.56us [2.92,25.42] in S2. This is a
preparation/context interaction, not a model fit or predeclared extra gate.

## PMU limits and next boundary

Loaded4 under2MiB already has substantial L2 refills (~98–103k cold and
75–91k prepared), yet no positive original penalty. At8MiB, the corresponding
counts are~129–131k cold and125–128k prepared; preparation does not eliminate
the victim's lower-level refill activity. These counts alone do not assign
refills to LLC or DRAM or reveal response timing.

In8MiB loaded4, aggregate domain27 DDRC read rates are roughly73–77GB/s with
small backgrounds versus39–42GB/s with large backgrounds, despite the latter
being slower. These controller-gate rates include background and victim;
they are not per-victim delivered bandwidth. A single aggregate bandwidth
number cannot order these two measured contexts.

The next narrow experiment, if requested, should bracket2–8MiB with intermediate
footprints and measure cache-path/return behavior on the already-triggering
fixed plans. Do not add a generic cold-start, L2-miss or fixed issue-port
penalty from this evidence. No unique port, queue or DDR cause has been identified.

## Provenance

HEAD `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus scoped uncommitted Lab
changes. Exact source inputs are in `build_inputs.tar.gz`, on the target and
locally alongside raw sessions and smoke logs. `report.json` contains full
cell medians/means/std/P90/P99, paired intervals and instruction checks.

| Identity | SHA256 |
| --- | --- |
|Binary|`ed12c37a577e420278bc36b575c441d70c6aaf927adfea3a545e1d350fcea089`|
|Production JIT source|`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`|
|Frozen fit|`b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781`|
|Session1|`4c44a80613cf9c4cb9ddddf84994ceb508ba853aaaf6814c04ea66223e0ea5a1`|
|Session2|`838c3e4296019ae6ee513790ad03773d5af45d6b677ae7d1d8606a21a5569b2a`|
