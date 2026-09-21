# M12/1T background count versus M1/1T foreground response

## Decision

Both independent sessions show the same foreground response: negligible change
at1–2 backgrounds,~1.5–1.9% slowdown at4,~13% at8,~20% at12 and~27% at16.
Domain27 DDRC read traffic grows from~9GB/s isolated to~55GB/s at16 and still
increases materially from12 to16; no saturation plateau is established.
The count sweep does not guarantee a constant DRAM/L3 pressure ratio.

Foreground LLC read-miss events increase substantially at8–16 while L2 refill
volume remains~130k. L3C ref counters themselves are nearly insensitive to
the sweep; preserve them but do not use this unvalidated signal as proof of
proportional L3 pressure. No model was fitted or adopted.

## Protocol

Class E Lab diagnostic; no production, planner or frozen-model changes.
Foreground is the current M1/1T W13, K4096/N1024,8MiB B. Compare real W13,
B-only, AB load-only and loaded4 no-store, with matched empty controls.
Background counts0/1/2/4/8/12/16, each an independent M12/1T W13 using the
same K/N/B size. Each background call contains one M12 panel and rotates
to the next of four8MiB weight copies after completion. There is no M120
cross-panel B reuse or outer weight-window loop.

All allocations persist and are independent between workers. Same256MiB
scrub, four-copy foreground rotation,5ms background lead-in, NUMA3 placement;
foreground CPU304, active background CPUs288 through287+count, controller240,
allowed CPUs240–319. All active background cores and foreground belong to
the same LLC domain. Inactive workers stay parked; raw counts must be zero.
No timestamp instrumentation. Ordinary aligned memory under existing THP
policy; no verified explicit HugeTLB residency.

Each worker's W13 stage/owner stripe is8MiB, t=1, N16 backend tiles,
`w13_window_tiles=0`/`full_n_team_stripes`,R13=1; W2 is not measured.
Changing count changes aggregate footprint and cache behavior, so equal
DRAM/L3 pressure proportions are NOT assumed.0..16 is the bounded sweep,
not a claim of finding the machine's architectural bandwidth limit.

35 cells per randomized round,5 warmups +31 recorded rounds, two independent
sessions (seeds249808/259808). Core counters: cycles, retired instructions,
L2 refill, LLC read-miss, backend stall. Collect domain25/27 L3C ref/hit and
48 DDRC counters. L3C ref rate is events/ns (Gevents/s), not GB/s. DDRC rate
uses flux counts and each counter's enabled duration; the controller gate
includes foreground/background and is longer than the foreground kernel.
L3C hit/ref is an event ratio, not an exact foreground hit probability.

No fitting or adoption. Require full grids, active-prefix/inactive-tail and
numerical checks, all PMU running ratios>=0.99. Report baseline-relative
paired slowdown intervals and full latency/counter distributions.

## Reproduction

Arm-codex-internal root `/home/zhangxu/codex/fused_cpp`; source/raw snapshot
`tmp/m12_background_count_20260908/`, mirrored locally (ignored). Build uses
the C++17/O3/pthread/SVE256/BF16 recipe in `l2_matrix_mix_20260908.md`, with
this directory substituted. Run one-round no-PMU and PMU smoke first.

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/m12_background_count_20260908/bench_phase_supply.py \
  --binary tmp/m12_background_count_20260908/phase_supply_native \
  --output tmp/m12_background_count_20260908/session1.jsonl \
  --m12-background --seed 249808
```

Repeat session2 with seed259808. Analyze with:

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_m12_background.py \
  --sessions tmp/m12_background_count_20260908/session1.jsonl tmp/m12_background_count_20260908/session2.jsonl \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --output tmp/m12_background_count_20260908/report.json
```

## Validation

Local targeted tests43 passed (M12 count, background-window, phase-supply,
pressure-curve andAB-contrast). Tests cover inactive/active workers, count
geometry andL3 event-rate units. Ruff passes; C++ formatted Google120.
Native build,35-cell no-PMU/PMU smoke, old-protocol smoke and both formal
sessions pass.88 uncore events are present (48 DDRC,40 L3C), plus5 core
events; running ratios pass>=0.99. All active backgrounds complete and verify
M12 outputs, inactive backgrounds have zero calls. Every active worker
completes at least6 calls per cell (including lead-in and drain), so the
observed cells are not all confined to the initial background invocation.
Production B-only/full-no-store4 byte-identity and foreground numerical/poison
checks pass. No production E2E validation was performed.

## Main foreground curve

Bandwidth-scope clarification from the subsequent dual-LLC audit: the table
below reports DDRC group27 only, not all NUMA3 memory traffic. At16 backgrounds,
group25 also reports56.21/56.69GB/s; the two groups together are approximately
112GB/s. Both controller groups carry traffic even for a single-LLC background.
Keep the original per-group values; do not interpret55GB/s as total NUMA bandwidth.

Real M1/1T W13; each cell lists S1/S2. Slowdown is a median of round-paired
ratios against count0, not a ratio of independent medians. DDRC is aggregate
domain27 traffic over the controller gate, not foreground-delivered bandwidth.

|M12 backgrounds|Foreground us|Paired slowdown %|DDRC read GB/s|
| --- | --- | --- | --- |
|0|303.37/302.08|0/0|8.85/8.96|
|1|302.25/301.65|-0.19/-0.13|11.12/11.16|
|2|302.56/303.49|-0.07/+0.36|14.29/14.44|
|4|308.05/307.73|1.48/1.88|20.76/20.74|
|8|343.18/344.18|13.02/13.52|32.28/32.14|
|12|365.43/364.99|20.32/20.34|43.30/42.94|
|16|383.99/384.22|26.36/26.94|55.30/55.26|

At8, slowdown95% bootstrap intervals are[12.26,13.39]/[12.51,15.72]%;
at16 they are[24.63,27.07]/[24.76,28.04]%. The observed increase between4
and8 backgrounds is repeatable, but does not identify a physical knee or
justify fitting a threshold from this experiment alone.

## Supply controls

Paired slowdown at16 backgrounds:

|Foreground|S1/S2 slowdown %|S1 isolated / count16 us|
| --- | --- | --- |
|B-only|19.08/19.21|256.55/306.77|
|AB load-only|22.57/22.74|296.94/364.95|
|Loaded4 no-store|26.20/26.88|301.95/380.56|
|Real W13|26.36/26.94|303.37/383.99|

Unlike the prior comparison between two different background kernel types,
increasing this fixed M12 background slows both supply-only and mixed victims
in the same direction. The magnitude is not identical, and the model remains
frozen; this is not a held-out fit validation or a fixed compute/memory split.

## Counter interpretation and limitations

Real foreground L2 refills stay around129.9–130.5k over the sweep. LLC
read-miss cell medians are:

|Backgrounds|LLC read-miss S1/S2|DDRC occupancy/read-command ratio S1/S2|
| --- | --- | --- |
|0|2,955/2,597|22.16/22.07|
|4|4,201/4,273|28.15/27.78|
|8|24,390/24,189|36.44/36.48|
|12|38,628/37,340|38.19/38.47|
|16|41,684/41,891|42.94/43.07|

L3C ref rate is only~1.6–1.7M events/s in S1 and~1.7–1.9M events/s in S2,
with no increasing response to count. The associated hit/ref event ratio is
also almost constant within each session (~0.61/~0.57). This signal fails a
basic sensitivity check as a total-workload L3 pressure proxy. Its event
semantics/topology/filter coverage need separate validation; do not interpret
the flat signal as proof that L3 traffic stayed constant, or convert it into
GB/s. Raw counters are retained rather than rescaled to match DDR.

The experiment is bounded at16 backgrounds. DDRC traffic grows by about12GB/s
from12 to16, so do not call55GB/s the DDR ceiling. A next count extension can
use additional same-domain cores with explicit topology/worker-capacity checks;
do not silently add cross-domain workers. Arrival remains after a5ms lead-in,
not randomized against every phase, and constant-valued BF16 inputs are the
existing diagnostic data rather than a real-trace model distribution.

## Provenance

HEAD `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus scoped uncommitted Lab
source. Raw sessions, three smoke logs and `build_inputs.tar.gz` retained in
the local/remote ignored snapshot directory. `report.json` holds full
median/mean/std/P90/P99 distributions, paired slowdown intervals and counters.

|Identity|SHA256|
| --- | --- |
|Binary|`94471e1031342bf79ae2baa544e951c4a3430d0d90f655011716dec4b9aa80d1`|
|Production JIT source|`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`|
|Frozen fit|`b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781`|
|Session1|`a006acbd63319fb7727a67a6545a075ee03de49cd916023ddeb33a9f7c10a178`|
|Session2|`2ae909fcb0ee830176e70cead40e52ad72e8994eaa8f3c23b1b8d8d942eca2aa`|
