# Victim cache-path and N-tile timeline diagnostic

## Decision

The unstamped PMU comparison is reproducible: under large-M backgrounds,
LL_CACHE_MISS_RD increases for load-only AND both mixed-compute victims,
while L2 refill counts remain approximately constant. Load-only becomes faster;
loaded/resident matrix4 become slower, with additional cycles tracking backend
stall cycles and no corresponding large frontend-stall increase. A universal
positive cost per extra LLC-miss event cannot explain all three schedules.

The dense timeline is REJECTED for localization. All four sessions fail both
the ordering and5% instrumentation gates: stamped matrix4 becomes substantially
slower overall and loses the original large-versus-small penalty. Do not read
its bin deltas as the original workload's time distribution. This experiment
does not determine whether the original penalty starts early or accumulates
throughout the kernel. No physical correction was fitted or adopted.

## Scope and predeclared gates

Lab class E only; no production/default/schema/model change or fitting.
Fixed cold M1/1T W13 geometry K4096/N1024,8MiB B; AB load-only, loaded4,
register-resident4 with identical loads. Each has an unstamped and stamped
version. None/16 M1/16 M120 backgrounds:21 cells per randomized round.

Two independent31-round sessions per PMU group,5 warmups. All cells use the
same process, allocations, persistent touched workspace,256MiB scrub and
four-copy B rotation. Background on288–303 starts5ms before victim304;
controller240, allowed240–319, NUMA3 memory, same Arm-codex-internal host.
No extra victim preparation. Ordinary aligned buffers under existing THP
policy; no verified explicit HugeTLB residency. No W2 measurement.

At every N16 tile boundary, the stamped JIT performs ISB/CNTVCT_EL0/store,
65 timestamps for64 tiles. It remains one kernel call, with unchanged A/B load
addresses and matrix instructions. Stamps use a separate preallocated buffer
with a checked canary;195 extra body instructions. ISB is not a memory-completion
barrier. Time includes instrumentation; these are not unperturbed memory latencies.

Interpret timestamps only if plain and stamped loaded4/resident4 preserve a
>=2% positive paired large-versus-small penalty with95% lower bound>0, and
every stamped/plain median total-time change has absolute magnitude<=5%.
All PMU running ratios must be>=0.99. Gates are not relaxed after seeing data.
Report first tile, remaining63 tiles and eight consecutive8-tile bins.

## PMU scope

Native sysfs encodings on this target:

| Event | Code | Group |
| --- | --- | --- |
|cpu_cycles|0x11|both|
|inst_retired|0x08|both|
|l1d_cache_refill|0x03|cache|
|l1d_cache|0x04|cache|
|l2d_cache_refill|0x17|both|
|ll_cache_miss_rd|0x37|path|
|stall_backend|0x24|both|
|stall_frontend|0x23|path|

Six events per core group;48 DDRC events additionally collected. The event
names/codes match Arm's common PMU definitions; see the
[Arm A-profile PMU common-event reference](https://developer.arm.com/documentation/ddi0487/mb/-Part-D-The-AArch64-System-Level-Architecture/-Chapter-D14-PMU-Event-Descriptions/-D14-3-Common-event-numbers/-0x80C9--INT-FIXED-OPS-SPEC--Non-scalable-element-arithmetic-operations-speculatively-executed--integer?lang=en).
This is a HiSilicon implementation, not a Neoverse V1/N2: do not import their
vendor-specific detailed semantics. `ll_cache_miss_rd` is retained as a
cache-path event, not converted into exact DRAM bytes; backend stalls do not
identify one queue/issue port. The event counts are per full victim call,
not per timeline bin. DDRC gates include background and exceed kernel duration.

## Reproduction

Remote root `/home/zhangxu/codex/fused_cpp`, snapshot/raw directory
`tmp/supply_timeline_20260908/`, also retained locally (ignored). Build with
the same C++17/O3/pthread/SVE256/BF16 command as `l2_matrix_mix_20260908.md`,
substituting this snapshot directory. Run no-PMU and both-group smoke first.

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/supply_timeline_20260908/bench_phase_supply.py \
  --binary tmp/supply_timeline_20260908/phase_supply_native \
  --output tmp/supply_timeline_20260908/cache_s1.jsonl \
  --timeline --pmu-group cache --seed 189808
```

Collection order: cache_s1/path_s1/cache_s2/path_s2, seeds189808/199808/209808/219808.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_supply_timeline.py \
  --sessions tmp/supply_timeline_20260908/cache_s1.jsonl \
  tmp/supply_timeline_20260908/path_s1.jsonl \
  tmp/supply_timeline_20260908/cache_s2.jsonl \
  tmp/supply_timeline_20260908/path_s2.jsonl \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --output tmp/supply_timeline_20260908/report.json
```

Local targeted tests:52 passed across timeline,trigger,L1,L2,phase-supply,
pressure-curve andAB-contrast tests. Ruff passes and native formatting checked.
Target results and validation follow.

## Unstamped measurements

The two path-group sessions measure these round-paired large-minus-small
differences (S1/S2). All values are per victim invocation, not whole-workload
or per-tile counters.

| Victim | Time delta us | LLC read-miss event delta | L2 refill delta | Backend stall delta cycles |
| --- | --- | --- | --- | --- |
|load-only|-12.41/-13.61|+39,717/+39,645|+142/-1|-38,437/-40,772|
|loaded matrix4|+42.63/+39.77|+47,535/+45,017|+148/+161|+124,312/+106,713|
|resident matrix4 + loads|+39.99/+36.81|+45,586/+44,670|-476/+245|+115,010/+106,927|

Absolute path-group medians, small/large background:

| Victim | S1 us | S2 us | LLC read-miss medians S1, small/large |
| --- | --- | --- | --- |
|load-only|398.74/387.00|401.18/386.50|62,592/101,751|
|loaded matrix4|416.36/459.19|420.80/460.15|51,900/100,185|
|resident matrix4 + loads|421.59/457.93|420.34/460.82|52,936/99,305|

The95% paired time-delta intervals are [-13.57,-9.37]/[-17.56,-10.81]us
for load-only, [39.98,44.51]/[35.03,41.75]us for loaded4, and
[35.13,47.87]/[34.12,44.24]us for resident4.
Independent cache-group sessions retain the same direction: load-only
-13.19/-13.72us, loaded4 +40.47/+32.03us, resident4 +37.12/+40.07us.

Full-call L2 refills are approximately129–130k in all unstamped cases.
Frontend-stall differences in the path group are small: loaded4 -114/+70
cycles, resident4 +179/-102 cycles. Loaded4 extra CPU cycles are
123,699/105,936 versus backend-stall increments124,312/106,713. These are
correlated event differences, not a unique decomposition of causal stalls.

L1 refill changes are also schedule-dependent: cache-group load-only medians
rise from~91–93k to~106k, while mixed versions rise from~47k to~48k. Do not
combine medians from separate PMU sessions into an exact per-call cache ledger.

Interpretation: the cache-path event changes are not exclusive to the mixed
kernel. What differs sharply is how the instruction schedule responds to that
context. Different effective overlap/parallelism and return latency remain
plausible, but neither load latency nor outstanding-miss occupancy was directly
measured. Lower aggregate DDR traffic is not sufficient to predict lower
victim latency; neither extra-miss counts nor backend-stall labels establish
a unique DDR, prefetch, queue or issue-port cause.

## Instrumentation failure

Across four sessions, stamped loaded/resident matrix4 is approximately
19–21% slower with small backgrounds,9–10% slower with large backgrounds,
and24–27% slower without background. These are far above the unchanged5%
gate. The added time is not a harmless constant that can be subtracted.

The loaded4 large-minus-small gap changes from plain32–43us to stamped
-3.11…+0.05us; resident4 changes from plain37–40us to stamped-2.45…+1.61us.
All four stamped intervals cross zero for these mixed schedules. Therefore
none of the four timelines is accepted for original-workload localization.
The65 timestamps are structurally valid, but their workload is perturbed.

This separately demonstrates sensitivity to the combined ISB/counter-read/
timestamp-store intervention. It does NOT isolate ISB from stores or scheduling,
and does not prove that the original penalty was frontend/issue-port contention.

If continuing, first test much sparser boundary sampling with the same
unstamped controls and unchanged intrusion gates; do not add progressively
more per-tile PMU reads to this rejected probe. Preserve the current negative
result and inspect new timelines only after their own gate passes.

## Validation and provenance

Native build,21-cell no-PMU smoke, both six-event PMU smokes, old-protocol smoke
and all four formal sessions complete. Generated production B/full-no-store4
byte checks, output poisoning, background numerics, timeline canary/65 increasing
timestamps,195 extra instructions and running-ratio checks pass. CNTFRQ is
100MHz. Numerical/structural correctness does not override failed intrusion gates.

An initial JIT startup failed before measurement because this Xbyak version
expects the low op0 bit for MRS (hardcoding its high bit), not architectural
op0=3. Corrected to1 for CNTVCT_EL0; original incomplete `correctness.jsonl`
is retained and excluded. Successful smoke is `correctness_retry.jsonl`.

HEAD `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus scoped dirty Lab source.
Exact measured inputs are in `build_inputs.tar.gz` alongside raw and successful
smokes on the target and locally. `report.json` preserves full distributions,
counter differences, rejected timelines and gate results. No production E2E
or model integration was run.

| Identity | SHA256 |
| --- | --- |
|Binary|`ce32d6835be568aa48bb060f776f401b2783780cd9f0f6266a3f30bd8753a4e5`|
|Frozen fit|`b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781`|
|cache_s1|`ef09f3a67037cefc0a1f45defb21bd540f4ebd0f4e5f63ed54c52781524d64c0`|
|path_s1|`1073d3aa991b8b51fb639e5971eb48a3a1ffe63056a2a2c2d3f45bfec5863aab`|
|cache_s2|`e04ec6bd5a64cc5c0437b10c7b5222750d7f0e0bb20d1c196a94e83946778aa9`|
|path_s2|`ae258729ab4eaf75c79f09367f6ec02127ed759b7a89686e9c6c832bf2f0d3cb`|
