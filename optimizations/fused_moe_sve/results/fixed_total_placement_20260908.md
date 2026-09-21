# Fixed-total background placement: measured local-service effect

## Decision

Two independent sessions reject the proposed lower-total-DDR-pressure explanation
for this fixed50 grid: all three distributions sustain about279–280GB/s.
They nevertheless produce large, repeated foreground differences. Relative to
25/25,38/12 is15.61/16.53% slower and12/38 is28.04/28.13% faster (paired-round
median changes). Both sessions'95% intervals excludezero for all four probes.
This establishes a distribution effect for the prescribed CPU sets, not a
universal thread-count law or a uniquely identified LLC resource.

| Same/other background count | S1 foreground us | S2 foreground us | S1/S2 total DDR GB/s |
| --- | ---: | ---: | --- |
| 0/0 | 304.22 | 305.49 | 14.97/15.65 |
| 24/24 bridge | 923.36 | 945.01 | 277.21/277.39 |
| 25/25 baseline | 1096.71 | 1097.74 | 279.35/279.48 |
| 38/12 | 1270.31 | 1276.66 | 280.21/280.27 |
| 12/38 | 791.06 | 767.04 | 279.75/280.24 |
| 32/32 bridge | 1693.31 | 1684.41 | 287.54/287.19 |

Same/other counts exclude foreground CPU304. Absolute values are cell medians;
percentage changes below are medians of matched-round ratios, so they need
not equal ratios of the displayed medians.

| Contrast versus25/25 | Probe | S1/S2 paired time change % |
| --- | --- | --- |
| 38/12 | Real W13 | +15.61/+16.53 |
| 38/12 | B-only | +24.69/+25.22 |
| 38/12 | A+B loads | +19.54/+18.56 |
| 38/12 | Full no-store | +16.16/+16.39 |
| 12/38 | Real W13 | -28.04/-28.13 |
| 12/38 | B-only | -29.91/-25.98 |
| 12/38 | A+B loads | -12.30/-14.11 |
| 12/38 | Full no-store | -29.94/-28.58 |

Real38/12 time-change CI95 is[14.68,16.10]/[15.48,17.30]%; real12/38 is
[-30.25,-25.75]/[-31.23,-23.19]%. Intervals use2000 IID matched-round
bootstrap draws via the existing `paired` helper. No arbitrary phase,
long-dependence or universal parameter-uncertainty claim follows.

## Mechanistic constraints from this result

The38/12 versus25/25 contrast is particularly useful: foreground LLC read
misses stay near131k, paired change -0.47/-0.46%; L2 refill change is within
0.11%; total DDR rate changes by only+0.29/+0.26%. Yet foreground latency
increases16% and supply-only probes also slow. A model using only miss volume
and aggregate memory throughput is insufficient. Per-group queue changes
are small and mixed-sign: group25 +2.73/+3.68%, group27 -1.35/-2.88%.

The12/38 case is different: LLC read misses fall to~51k, approximately61%
below25/25. Its speedup includes a cache-path composition change, not just
better service of the same mix. Do not merge these into a single contention
coefficient. B-only and AB results rule out an explanation requiring W13's
compute instructions, but do not uniquely isolate LLC, interconnect, request
admission, DDR arbitration or memory-level parallelism.

Next theoretical work should distinguish request volume, path composition,
path-specific service delay and effective outstanding concurrency. These must
be independently constrained, not all inferred from total kernel time. Use
38/12 versus25/25 as the fixed-miss-volume counterexample; retain12/38 as the
path-mix counterexample. Both remain diagnostics, not untouched holdouts after
model design. No new physical coefficient has been fitted.

## Follow-up: supply-probe transfer is not a fixed memory fraction

Offline replay, without additional hardware or fitted parameters, tests whether
an independently executed B-only or AB probe supplies a transferable slowdown.
Define R0=304.22us, B0=258.54us, AB0=298.32us from session1 isolated medians;
freeze these anchors for BOTH sessions. For probe time P under background, the
fixed-fraction hypothesis is R/R0=(1-alpha)+alpha*(P/P0),0<=alpha<=1.
Because P/P0>1 here, its largest prediction is R0*P/P0 (alpha=1).

For B-only this upper endpoint still underpredicts every nonzero cell in both
sessions: errors range -11.77% to -2.41% inS1 and -11.27% to -2.99% inS2.
Matching observed medians would require alpha1.033–1.237 and1.038–1.199,
respectively. These diagnostic implied values are NOT fitted physical fractions.
They contradict this particular fixed-fraction/proxy combination at the measured
medians; they do not invalidate service-rate theory for the target's OWN requests.

AB transfer changes sign: alpha=1 predictions underpredict25/25 by10.46/10.06%,
but overpredict12/38 by9.22/10.55%. More directly, matched real-minus-AB time
is+142.71/+132.86us at25/25, but -39.50/-51.81us at12/38, with95% intervals
[-74.20,-13.16]/[-79.35,-0.21]us for the latter. The second interval is close
tozero; IID bootstrap is not a long-dependence guarantee. Nevertheless this
prevents treating AB elapsed time as a universal lower bound or subtracting it
to obtain a nonnegative, additive compute time. Probe instruction schedules,
request arrival and overlap differ even where address/count ledgers match.

MISE motivates application-specific request service rate plus exposed-memory
fraction, not substitution of another instruction stream's elapsed time or
global DDR rate. Its controller-priority measurement mechanism is not available
in this harness. See [Subramanian et al., HPCA2013](https://pdl.cmu.edu/ftp/NVM/mise-hpca13.pdf).
Our reuse here is a falsifiable hypothesis inspired by that distinction, not an
implementation or validation of MISE.

Reproduce with `.venv/bin/python tmp/fixed_total_placement_20260908/check_service_transfer.py`;
exclusive output `service_transfer_diagnostic.json` preserves all three tested
forms (ratio, additive increment, overlap floor), both sessions and paired AB
contrasts. Existing raw-reader/numerical/placement gates run before comparison.
This diagnostic is retrospective; no new predictive accuracy claim is made.

## Next identification experiment and full-goal boundary

Use a dependent-load chain as a latency-sensitive endpoint and a1/2/4/8/16-chain
sweep as a controlled-concurrency endpoint, with fixed total line coverage and
load count. Preserve same-process allocation, background distribution and counter
protocol; keep the real/B/AB bridges. Randomized address order and sequential
order must remain separate factors, because changing prefetchability at the same
time as concurrency would confound the result. Verify visited-line coverage,
checksum, generated load/dependency structure and cache/TLB events before timing.

For a common request boundary, Little's-law relation N=lambda*L distinguishes
completion rate lambda, mean in-flight requests N, and residence time L; a
throughput/time observation alone does not identify both N andL. Programmed chain
count is an upper bound on dependence-independent chains, NOT a measurement of
hardware outstanding misses. Pointer-chain latency is pattern-specific and may
not be substituted directly into a sequential SVE model. Test transfer on real
kernels rather than declaring physical identifiability from a fitted curve.

The full goal remains open: independently identified service/concurrency and
overlap, small-M shape coverage beyondM1, stage coverage beyond thisW13 probe,
background count/placement/arrival holdouts, frozen prospective validation and a
clear distinction between measured-feature and plan-visible prediction are still
required. Do not mark the goal complete from the fixed50 contrast alone.

## Question and scope

Does distributing50 independent M12/1T backgrounds as38/12,25/25 or12/38
change total memory injection and foreground M1/1T service differently?
The first number is the foreground LLC's background count; foreground CPU304
is additional. Controller240 is excluded. All CPUs and allocations are NUMA3.

Class E, Lab only. Extend native background placement modes9/10, retain all
old modes and the original85-cell grid. No kernel, production API, model,
calibration, workspace lifecycle or counter formula changes. Rollback scope is
the placement-only Lab additions; original binaries and raw files remain intact.

## Preregistered protocol

New flag `--placement-50` uses six conditions: isolated, balanced48, balanced50,
same38/other12, same12/other38, balanced64. Each has real W13, B-only, A+B,
no-store and empty controls (30 cells). Same process and allocations, random
cell order per round, four-copy rotation, persistent touched output,256MiB
scrub per LLC and unchanged5ms background lead-in. Foreground W13 is M1,
K4096/N1024,1T,8MiB B, full stripe; background M12 uses the same K/N and
independent four-copy B allocations. No W2 measurement or full-plan claim.

Run correctness/PMU smoke first, then two independent processes with5 warmups
and31 measured rounds, seeds319808/329808. Retain numerical checks, observed
CPU lists, positive active calls andzero inactive calls, and PMU running>=0.99.
Compare matched-round latency changes with95% intervals and require direction
agreement across sessions. Report null and contrary outcomes as well.

Inspect total/per-DDRC-group throughput and queue ratios, foreground refills,
LLC read misses and backend stalls. Global counters are NOT foreground-only
service measurements; unequal counter windows remain a limitation. Background
call counts include lead-in/drain and must not be divided by foreground time.
No post-hoc fit or change to the frozen layered model.

## Validation and reproduction

L1:25 placement tests passed;68 focused regression tests passed across dual-M12,
M12 background, layered model and phase supply. Ruff passed. L2: GCC13.2.0
native build and30-cell PMU/numerical/placement smoke passed. L3: both complete
31-round sessions passed raw-reader gates. New binary SHA256:
`8a92c487b0e10b6270daaae9f3039c7d480f4b22e9fd3175dfb2191241bb24cd`.
Production JIT SHA256 remains
`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
Frozen model SHA256 remains
`f3796a0f7e815dc5c5e297b3bafdb3bb64150358daa50150f91bec6897157d86`.
Source basis is HEADc80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5 plus Lab
changes; default extension not rebuilt. New sources/binary differ from historical
runs, so bridge values are contextual checks, not exact paired equivalence.

Artifacts use `tmp/fixed_total_placement_20260908/` locally and under
`/home/zhangxu/codex/fused_cpp/` on Arm-codex-internal. This is one causal
contrast toward the broader theoretical small-M model, not model completion.

Build uses the command in `phase_memory_supply_20260907.md` with the new
snapshot directory substituted: C++17,-O3,-pthread,armv8.2-a+bf16+sve,
SVE256; unchanged production JIT and Xbyak sources. Ordinary aligned memory,
existing THP policy; no HugeTLB-residency claim.

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/fixed_total_placement_20260908/bench_phase_supply.py \
  --binary tmp/fixed_total_placement_20260908/phase_supply_native \
  --output tmp/fixed_total_placement_20260908/session1.jsonl \
  --placement-50 --seed 319808
# Repeat with session2 andseed329808. Smoke used rounds1,warmup0,seed319807.
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_fixed_total_placement.py \
  --sessions tmp/fixed_total_placement_20260908/session1.jsonl \
  tmp/fixed_total_placement_20260908/session2.jsonl \
  --frozen-model tmp/layered_supply_model_20260908/frozen_fit.json \
  --output tmp/fixed_total_placement_20260908/report.json
```

Raw S1 SHA256:`1d3879e12c218ec117b332d080a9c440c8f951b777014a65f0f609f7815dcff1`.
Raw S2 SHA256:`a1c598b44a71267779a84cdd67a82ce68a0d8ea8c2c2a76f61a022aac339660f`.
