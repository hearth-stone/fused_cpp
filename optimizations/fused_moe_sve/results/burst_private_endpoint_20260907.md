# Cache-line accounting and burst-preserving private endpoints — 2026-09-07

## Outcome

Implemented the three requested pieces as an independent Lab candidate:

1. A payload, touched cache-line coverage, estimated refill and actually
   observed aggregate counters are separate fields. Unknown A/B-address-specific
   refills remain null, rather than being inferred from aggregate counters.
2. Private-L2 B delivery is distinct from LLC/DRAM refill. The service sampler,
   native benchmark and generated B-only loop were checked for their actual
   units and timing boundaries.
3. The overlap calculation uses separate private/shared endpoint bounds and
   preserves the original shared-resource burst allocator. It does not use the
   rejected whole-phase lifetime averaging.

**The accounting/implementation checks pass, but the candidate is not an
accuracy improvement and is not adopted.** New private-L2 bounds remain hidden
under compute in the checked shapes. M1 sensitivity does not collapse, but
median ordering remains wrong and one other direction regresses. No frozen v8,
production code, calibration, schema, search space or pruning behavior changed.

## A: payload, lines and refill

The ledger distinguishes:

- `a_payload_bytes`: bytes requested by the exact-M row-pair loads.
- `a_line_coverage_bytes`: owner-summed distinct64-byte lines touched by one
  A-panel scan, from packed K4 address intervals and aligned panel bases.
- `a_l2_delivery_estimate_bytes`: includes repeated N-tile scans when the A panel
  and one B tile do not fit L1.
- `a_llc_refill_estimate_bytes`: a candidate owner-scan line-refill estimate,
  not a PMU measurement. Assumes fresh advancing panels need one lower-cache
  supply per owner; producer cache state and cross-owner sharing are not simulated.
- `actual_a_refill_bytes`: null, because the current PMU has no A-address filter.

For M1/W13/16T these are0.25 MiB payload,1 MiB owner-summed line coverage and
4 MiB estimated private-L2 delivery. M1's packed stride covers unused row slots:
reading16 KiB of payload per owner touches64 KiB of lines. That distinction
does not mean every line misses LLC or is fetched from DRAM.

For M1341/W13/16T steady, payload166.25 MiB becomes166.50 MiB of owner-scan line
coverage, including the M9 tail's packed layout. The candidate private-L2 A
delivery is666 MiB: each owner has four N tiles and the96 KiB A panel plus a
128 KiB B tile exceeds64 KiB L1. This is an explicit geometric retention model,
not a complete cache replacement simulator or a fitted miss rate.

Existing observed counters stay separately in `observed_aggregate_refills`.
For example, the A-source experiment's session1 W13/16T steady has9,775,433
background L2-refill events with advancing A and5,171,151 with repeated A.
These totals include the victim's combined traffic and cannot be assigned to
A/B by subtraction. Changing A can also change B replacement and prefetch.

## B: private delivery versus lower-cache refill

M1341 has112 kernel panels: first M12, then110 M12 and one M9. For the111-panel
steady region at16T, the candidate ledger is:

| Quantity, owner aggregate | W13 | W2 |
|---|---:|---:|
| B instruction-load/delivery estimate | 888 MiB | 444 MiB |
| B refill from below private L2, inherited estimate | 0 | 0 |
| A private-L2 delivery estimate | 666 MiB | 20.8125 MiB |
| Total private endpoint including logical C | 1555.2979 MiB | 485.5781 MiB |
| LLC endpoint refill estimate including logical C | 167.7979 MiB | 41.5781 MiB |

The zero B refill is the frozen cache-retention assumption, **not a measured
fact**. B owner stripes512/256 KiB fit the1280 KiB private L2, but capacity alone
does not prove indefinite residency. Actual B-specific refill remains null.

The candidate assumes B must be delivered again on each panel if owner B plus
A exceeds L1; otherwise it can stay L1-resident. A is charged once per owner if
A-panel plus one B tile fits L1, otherwise once per N tile. Output traffic
remains logical C writes; write allocation/RFO and dirty evictions are not
invented or calibrated from read counters. These are declared limits of the
candidate, not completed physical-source attribution.

## What the service probes actually measure

Source chain inspected:

- `cpu_moe_schedule_optimization/cost_model/profile_analytic_services.py`:
  `profile_load_resource`, `concurrent_b_probe`, `profile_matrix`, geometry setup.
- `csrc/moe/arm/common/fused_moe_bf16_tiled.cpp`:
  `fused_moe_bench_sve_jit_w13_gemm`, including prepacking, warmup and `run_one`.
- `csrc/moe/arm/sve_bf16/jit_kernels.cpp`: `load_b`, B-only branches skipping
  A loads and BFMMLA, and probe store handling.

| Calibration resource | Actual source definition | Not established by that definition |
|---|---|---|
| `gemm_core_flops` | M12 L1-hot full A/B load + GEMM, no store | L2-resident GEMM throughput |
| `l2_bytes` | Warm repeated B-only endpoint-to-register scan; one shared read-only B across workers | Pure L2→L1 link rate or A-stream service rate |
| `llc_bytes` | Larger warm repeated B-only endpoint-to-register scan; shared read-only B | Per-request tier attribution or disjoint-owner equivalence |
| `dram_bytes` | Disjoint per-worker B pools, rotation by iteration | Every access is cold DRAM under the new workspace protocol |

Load-probe numerator is `threads * runs * 2*K*N`; denominator is the slowest
worker's summed timed scans, not PMU line-refill bytes. B-only uses M1 and does
not load A or perform matrix work. Packing/allocation and warmup precede timing;
each timed scan includes the native dispatch wrapper. The hot default footprint
targets are half private L2 and one-sixth LLC. These are intended tiers, not
proof of actual per-load residency.

Cold sampling uses a pool sized from at least64 experts and at least4 per worker
by default. The native loop uses `iteration % weights.E`; neither this sampler
nor its timed loop performs the current workload's per-round scrub. Current
source definitions and v8 provenance agree on probe family; this audit does not
re-run historical services or certify their exact cache state.

Accordingly, deeper endpoint times already include downstream delivery. They
must not be added as though each were an independent link latency. Reusing
B-only curves for A+output is still an approximation needing matched validation.

## Burst-preserving overlap

For each GEMM phase the new private endpoint bound is
`T2 = estimated_private_delivery / calibrated_L2_rate(team_width)`. The shared
LLC bound uses estimated LLC refill, and DRAM uses compulsory B plus the existing
spill rule. The body remains:

```text
max(compute, private_L2_endpoint, shared_LLC_endpoint, DRAM_endpoint)
```

The private-L2 bound is not additionally dilated as a rank-wide shared resource.
Its byte ledger and local service time remain explicit, while its shared demand
is zero. Scalar/vector views agree. LLC/DRAM, domain injection, cold/gather,
stage scales and old wide/narrow terms keep their original behavior. Thus
short cold phases still see service-interval burst contention, not demand
averaged across a large compute interval.

For M1341/16T steady, private-L2 bounds are0.768 ms W13 and0.240 ms W2;
LLC bounds are0.731/0.181 ms, while compute bounds are7.982/3.991 ms. The private
delivery was missing from the explicit ledger, but adding its bound does not
change the maximum. None of the35 isolated shapes changes predicted W13/W2
time with these frozen rates. This is a model result, not a hardware speedup.

The candidate's supported inspection is `endpoint_ledger` plus phase/placed-DAG
results. Generic v8 stage-demand explanation, native quick export and stage-window
scoring are explicitly blocked to avoid presenting old demand fields as the new
ledger or accidentally using the candidate in production.

## Validation and decision

Compared frozen v8, line-only accounting, and line+private-endpoint accounting,
each with old wide/narrow retained or removed. Seven known workspace plans across
median/high-skew produce42 placed replays. Hardware is the two existing diagnostic
phase sessions; comparison is `compute_end_ms`, not complete E2E latency.

| Variant | Old terms | Compute-end MAPE | M1/1T stage MAPE | Median-direction agreement |
|---|---|---:|---:|---:|
| v8 | retained | 20.24% | 38.79% | 6/8 |
| line-only | retained | 20.39% | 38.71% | 5/8 |
| private endpoint | retained | 20.39% | 38.71% | 5/8 |
| v8 | removed | 13.04% | 30.15% | 3/8 |
| line-only/private endpoint | removed | 12.87% | 30.27% | 3/8 |

Line-only and full-candidate predictions coincide on the checked plans because
the larger private bound remains noncritical. High-skew anchor rises39.511→39.800 ms,
crossing greedy's39.650 ms and losing a direction. Median elite still incorrectly
ranks slower than anchor. Counts above use pairs whose hardware medians agree
between sessions, not confidence-interval/actionable-margin gates.

Small-M burst sensitivity is retained, but no all-cases non-regression claim:
high-skew expert49 M1/1T W13 predicts0.571 ms versus v8's0.569 and measured
0.862/0.909; it does not collapse to the rejected lifetime model's0.243 ms.
Across the136 M1/1T stage/session observations,30 absolute errors still increase
by some amount. Aggregate small-M error stability is insufficient for adoption.

The accounting/reference implementation is retained for bounded diagnostics;
the model is **not adopted for accuracy or search**. Actual A/B-address refill,
matched private/LLC endpoint rates, output RFO and producer-state residency remain
unresolved. Do not tune a new overlap coefficient against these known examples.
Uniformish, prospective hardware/ranking holdout and false-pruning/top-K gates
were not run and cannot be inferred from this seven-plan replay.

## Reproduction

Primary class M delivered only in Lab; no production/default/schema/native/kernel
change. Rollback is limited to `burst_endpoint_model.py`, its tests and the
manifest/math/report additions. Existing user modifications are preserved.

```sh
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/burst_endpoint_model.py \
  --calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --frontier-dir tmp/workspace_phase_timeline_20260907 \
  --pmu-summary tmp/phase_a_source_20260907/summary.json \
  --output tmp/burst_private_endpoint_20260907/report.json
PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_moe_burst_endpoint.py tests/test_moe_ab_supply.py \
  tests/test_moe_workspace_phase_reaccount.py
```

Use a fresh output path on replay. Local ignored `report.json` contains35 shape
comparisons (M1/3/7/12/62/714/1341 ×1/2/4/8/16T), detailed phase ledgers, observed
aggregate PMU evidence and42 full timelines. Raw hardware artifacts remain in
the two referenced workspace/A-source directories, with their original identity
and protocol reports. No new hardware measurement or compilation was necessary.

Calibration SHA256 remains
`7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
Repository base `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus existing dirty work
and these Lab files. The underlying Arm artifacts use H4096/F512, SVE256 N tile16,
full owner stripes, NUMA3 CPUs240–319, BF16 and FP32 down output; source/timing
experiment differences remain explicit in their reports.

Validation:25 focused tests passed, including10 new endpoint/burst tests;
all frozen baseline predictions reproduce to1e-10 relative tolerance. Ruff and
diff checks pass. Production/native parity and new service/hardware calibration
were not run because no production behavior is adopted. No commit made.
