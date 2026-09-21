# Frozen A/B supply ledger and overlap audit — 2026-09-07

## Outcome

The accounting audit is complete, and a bounded overlap candidate was
implemented and replayed. **Do not adopt the candidate.** It reduces some
whole-plan overprediction but worsens small-M stage errors and retains the
median elite ordering failure. Production model/calibration/defaults are unchanged.

The key findings are:

1. A is already charged per active N owner and split across cold/steady panels.
   Adding another full A term would double-count it.
2. For M1341/16T the frozen model already predicts zero steady B **lower-cache
   refill**. The prior observed steady miss increment is not evidence of a
   missing repeated-B DRAM term.
3. The same A+B+C refill bytes feed both the L2 and LLC resource bounds. They
   are not a complete ledger of distinct physical links: B served from private
   L2 still needs delivery to L1/registers. The L1-hot GEMM service is not an
   independently measured L2-resident GEMM service. However, the endpoint
   service probes include downstream work, so blindly summing all link costs
   would also risk double charging.
4. Small-M packed-A effective load bytes differ from cache-line footprint.
   This is an identifiable accounting gap, not a newly fitted contention term.
5. The current service-interval offered-rate definition is intentionally
   conservative about bursts. Replacing it with phase-average traffic is not
   automatically a correctness fix; the hardware counterexamples reject the
   tested fluid approximation as a replacement.

## Exact frozen ledger

The audit loads frozen v8 with H4096,F512, BF16, TP degree4, FP32 down output.
Cache profile matches CPU304: private L2=1280 KiB, effective L2=960 KiB;
rank LLC=140 MiB. Width16 B owner stripes are512 KiB W13 and256 KiB W2.

M1341 phase demand in MiB, aggregated over all active owners:

| Width | Phase | A refill | B refill | C write |
|---:|---|---:|---:|---:|
| 16 | W13 first | 1.5000 | 8.0000 | 0.0117 |
| 16 | W13 steady | **166.2500** | **0** | 1.2979 |
| 16 | W2 first | 0.1875 | 4.0000 | 0.1875 |
| 16 | W2 steady | **20.7812** | **0** | 20.7656 |
| 1 | W13 steady | 10.3906 | 888.0000 | 1.2979 |
| 1 | W2 steady | 1.2988 | 444.0000 | 20.7656 |

The166 MiB A value is16 owners' traffic, not166 MiB of unique input storage.
Exact-M compute rows include the padded row pair (M1341 maps to1342 compute
rows); payload accounting therefore differs slightly from logical M×K bytes.

The current B reuse rule compares owner B plus one A panel with L2. It charges
a transient miss fraction for the first effective-cache-turnover count and a
binary physical-capacity miss fraction afterward. At16T both fractions are0
for these stages. At1T both are1 because B exceeds private L2. This distinction
is already present; it must not be replaced by uniform B bandwidth accounting.

Compulsory DRAM is first B only. A, repeated B and C are spillable, with a
single working-set-driven LLC spill fraction. Isolated spill is0 in every
ledger shape checked here. Thus the current model has nonzero steady A/LLC
traffic but zero isolated steady DRAM demand. This assumes fresh operator
inputs/intermediates come from caches unless the capacity spill rule triggers.
The synthetic phase probes and real production lifecycle are not interchangeable
evidence for that assumption.

## Byte and reuse ambiguities

### Packed-A payload versus touched cache lines

For a64-byte-aligned base, each K4 block advances by `packed_rows*8` bytes and
reads `compute_rows*8` bytes through row-pair loads. The audit enumerates touched
lines independently of the analytical payload formula.

- M1 W13: compute_rows2, packed_rows8, K4096. Payload16 KiB; touched lines64 KiB
  per owner scan. The sixteen-owner ledger is0.25 MiB payload versus1 MiB of
  owner-summed line coverage.
- M12 W13: compute_rows=packed_rows=12. Both are96 KiB per owner panel.

Touched lines are not measured LLC misses: cache reuse, sharing and prefetch
still matter. The audit reports both quantities; it does not silently replace
v8's demand with a line count or retune small-M latency.

### L2 service is not LLC refill

At16T/W13 steady, B instruction reads still total888 MiB across111 panels,
while the modeled B lower-cache refill is0. Those statements are compatible:
one is repeated load work, the other is estimated refill from below private L2.
But using the latter as the entire L2 endpoint resource demand leaves private
L2-hit delivery implicit. Clarify the calibrated endpoint meaning before
changing demand; simply adding888 MiB to both L2 and LLC would be wrong.

C currently contributes logical write bytes, not an explicit write-allocate,
dirty-eviction and read-for-ownership model. These effects are not separately
identified by the present read-miss counter. Cache working set also uses a
stage-wide A/C footprint and phase-aggregated spill, not line-age or panel-history
residency. Repeated-A suppression can therefore reflect A directly or its effect
on other cached data; it is not source-address attribution.

## Existing overlap versus tested candidate

The existing model uses `max(core, max(L2, LLC, DRAM)) + epilogue`, then its
placed context applies resource dilation and the old wide/narrow corrections.
It already has overlap. Endpoint service times use max, not sum, because the
deeper endpoint probes include downstream delivery.

Offered traffic is computed as demand divided by that resource's service time.
For M1341/W13/16T steady, LLC service is about0.730 ms versus7.982 ms phase
base time: the average duty ratio is9.14%. W2's ratio is4.53%. The current
burst/service-interval rate does not shrink in proportion to these duty ratios.
This is documented intentional behavior, not a coding accident.

The Lab `LifetimeOverlapModel` inherits original demand, spill, topology,
capacities and optional old corrections. Starting from each phase's isolated/
current-spill duration, including the chosen old terms, it computes average
request rates and slows **whole-phase progress** by the largest overload among
that phase's rank/domain constraints. Every requester is slowed by at least
its constraint's initial overload, so final sum(demand/duration) does not exceed
capacity. Unit tests and runtime assertions verify this property.

This avoids the unsafe shortcut of averaging demand over compute time and then
scaling only a transfer term still hidden under max. It is nevertheless a fluid
approximation: it discards arrival bursts and queue behavior. No new physical
parameter is fitted; no production hook is introduced.

## Fixed workspace replay: candidate rejected

Seven fixed plans across median/high-skew, each with two existing workspace
trace sessions;28 placed model replays cross old/new allocation with retaining/
removing old wide+narrow terms. Baseline reproduces every frozen point to1e-10
relative tolerance. Hardware comparison uses diagnostic `compute_end_ms`, not
full-call E2E; do not advertise this MAPE as deployment accuracy.

| Allocation | Old wide+narrow | Compute-end MAPE | M1/1T GEMM stage MAPE | Matching median directions |
|---|---|---:|---:|---:|
| Current service-interval | retained | 20.24% | 38.79% | 6/8 |
| Lifetime fluid | retained | 13.01% | **66.72%** | 7/8 |
| Current service-interval | removed | 13.04% | 30.15% | 3/8 |
| Lifetime fluid | removed | 18.76% | **64.37%** | 7/8 |

Direction counts use pairs whose two hardware session medians agree; they do
not apply confidence-interval or actionable-margin gates. One pair with
disagreeing session directions is excluded. No top-K or false-pruning claim is
made from seven already-known plans; no pruning is authorized.

Concrete retained-legacy counterexamples:

- High-skew anchor, expert49 M1/1T W13: hardware0.862/0.909 ms, current model
  0.569 ms, lifetime candidate0.243 ms. W2 similarly falls from0.290 to0.122 ms
  against hardware0.341/0.393 ms. Phase averaging removes too much real penalty.
- Median elite remains wrongly slower than anchor: lifetime predictions
  34.586 versus33.095 ms; hardware elite29.988/30.031 versus anchor30.541/30.762 ms.
- High-skew M1341/16T W13 only moves11.746→11.640 ms versus measured8.472/8.456;
  W2 worsens5.203→5.697 versus4.117/4.101. Changed event timing and preserved
  old corrections prevent a blanket statement that the large-expert issue is fixed.

Known counterexamples are reused for diagnosis, not prospective independent
holdout. The A-source hardware experiment motivates the audit but does not
fit the candidate or directly calibrate B/A-specific bandwidth.

## Next bounded work

1. Keep payload, line footprint, private-cache-hit delivery and lower-cache
   refill as distinct ledger fields. Validate the endpoint probes against those
   definitions before changing numerical demand.
2. Address the small-M packed-A line-coverage gap without an operator residual
   cancellation. It is not sufficient to explain every small-M error by itself.
3. Preserve cold/bursty request sensitivity. The tested whole-phase averaging
   is not an accepted overlap replacement; do not combine it with a new fit to
   hide its failed small-M guard.
4. Keep frozen v8/search behavior until a separately validated candidate passes
   stage checks and workspace ranking holdout. Uniformish, top-K recall and
   false-pruning validation remain required before any adoption.

## Implementation and reproduction

Primary class M, delivered as a Lab diagnostic/candidate only. No production
source, schema, ABI, kernel or calibration edits. Mathematical notes updated
in section8.2.4b. Rollback boundary: new Lab script/test and these documentation/
manifest additions. Existing user edits to `analytic_model.py` are untouched.

```sh
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/audit_ab_supply.py \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --frontier-dir tmp/workspace_phase_timeline_20260907 \
  --output tmp/ab_supply_audit_20260907/report_v2.json
PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_moe_ab_supply.py tests/test_moe_workspace_phase_reaccount.py
```

Use a fresh output path on rerun. Retained local artifact `report_v2.json`
contains25 shape ledgers (M1/12/62/714/1341 ×1/2/4/8/16T),28 complete model
timelines, stage guards and aggregate/direction diagnostics. Earlier
`report.json` remains as the pre-line-footprint exploratory artifact.
Inputs are `median.json`, `high_skew.json` and their `_analysis.json` companions
under the frontier directory; their original raw traces remain there too.
Calibration SHA256 remains
`7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
Repository base is `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus existing dirty
work and this Lab change. No new hardware experiment or build was needed.

Validation:15 focused tests passed, including8 new capacity/line-footprint tests;
all ledger conservation and runtime capacity checks pass. Ruff and diff checks
pass. Full production/native parity and new hardware holdout were not run,
because the candidate is rejected and production is unchanged. No commit made.
