# Dependent-load service identification

## Result: access pattern matters more than a uniform latency multiplier

Both formal sessions completed and passed the full raw-reader gates. Real W13
retains the38/12 versus25/25 penalty:16.73/17.85% paired medians,CI95
[15.92,17.51]/[16.48,18.60]%. Thus the added probe allocation did not remove
the within-session signal. No calibration was fitted or changed.

| Probe | S1 25/25 us | S1 38/12 us | S1/S2 paired38/12 penalty % |
| --- | ---: | ---: | --- |
| Real W13 | 1100.95 | 1286.94 | 16.73/17.85 |
| Sequential1 chain | 1030.17 | 1182.00 | 14.19/15.34 |
| Sequential2 chains | 3173.60 | 5358.00 | 74.04/65.93 |
| Sequential16 chains | 1184.40 | 1595.17 | 36.37/34.00 |
| Random1 chain | 35051.84 | 36641.82 | 4.65/2.75 |
| Random16 chains | 2624.08 | 2683.15 | 2.08/1.30 |

Random1 penalty CI95 is[3.88,5.24]/[1.72,3.58]%; random16 is
[0.95,2.75]/[0.71,2.18]%. These differences are not exact zero, but are much
smaller than the sequential/real penalties. Random16 versus random1 speedup
is13.33/13.24x at25/25 and13.73/13.32x at38/12; isolated14.40/14.32x.
There is no dramatic collapse in the benefit of programmed random-chain
independence under38/12. This does NOT measure actual hardware MLP.

Random1 isolated is14189.77/13971.35us (~108/107ns perpointer load), versus
~35–37ms (~264–280ns/load) with50 backgrounds. This includes translation,
cache lookup,branch andload-use costs, not an identified DDR latency. Its
longer execution window also samples background activity differently fromW13.
Sequential1 isolated is only275.32/275.25us (~2.10ns/load), showing why a
sequential dependent chain cannot be treated as a serialized cold-miss endpoint.

## Important counterexample to a naive chain-count model

Sequential traversal does NOT improve monotonically with chain count. At25/25,
S1 medians for1/2/4/8/16 chains are1030/3174/2236/1149/1184us; S2
1058/3388/2197/1151/1176us. Sequential2 is noisy (CV20.7/19.5%) but its
large degradation repeats; do not explain it by selecting a favorable round.

The fixed total address order is a logical/program-order invariant, not fixed
physical request-arrival order. Increasing chain count changes static load PCs,
eachPC's stride, loop cadence andout-of-order scheduling opportunity. These
can change prefetch behavior even without explicit prefetch instructions.
Consequently this sweep cannot identify a single hardware concurrency cap from
sequential timings. Treat that limitation as a result, not a reason to fit a
non-monotone lookup table andcall it a physical model.

Current evidence supports a request-pattern-sensitive service/overlap mechanism
and rejects a universal foreground latency multiplier as sufficient explanation.
Prefetch effectiveness/timeliness andlocal request admission are leading
hypotheses, not uniquely established causes. There is no verified prefetch PMU
counter or disabled-prefetch intervention inthis experiment.

Next: isolate per-load-PC stream geometry while retaining totalcoverage andchain
independence, andcollect validated translation/prefetch-sensitive counters where
available. Keep random latency andstreaming throughput as different endpoints.
An eventual real-kernel model must explain both rather than substitute one for
the other. No coefficients, production change or completed-model claim follows.

## Preregistered question

### Translation audit extension

Copy-stratified medians reproduce the sequential2 slowdown onall four copies.
Isolated random2 L2 refill medians are195863/220195 despite131072 programmed
line visits, versus131145/131146 forrandom1. Therefore logical line coverage
doesnot guarantee constant hardware refill count. Speculation/prefetch effects,
translation andevent semantics remain possible; no cause isassigned fromthese
counts alone.

The target exposes `dtlb_walk` and`l2d_tlb_refill`, butno named prefetch event
in its core-PMU sysfs list. Add `--chain-tlb`, retaining the entire60-cell grid
andEXACT previous native binary. The six-event core group is cycles,instructions,
L2 refill,LLC readmiss,L2 TLB refill,DTLB walk (replaces stall measurement).
Keep both DDRC groups andL3C; running>=0.99 remains mandatory. New runner andraw
files live under `tmp/dependent_load_tlb_20260908/`, withoutoverwriting the old
runner. Smoke first,then two5/31 sessions,seeds359808/369808. This is a changed
measurement group,so timing bridges mustbechecked before combining interpretation.
Translation counts are sensitivity evidence,not additive stall cycles.
Six-event60-cell smoke passed withnative hash29c17726 unchanged. Two formal
sessions359808/369808 launched aftersmoke; results pending. Local tests36 passed;
Ruff,manifest YAML anddiff checks passed. No calibration or production change.
Broader follow-up regression selection79 passed. During the first formal TLB
session, `/proc/<native-pid>/smaps` sums showed3176448kB AnonHugePages andzero
Private/Shared_Hugetlb; system THP policy was `[always] madvise never`.
This is whole-process point-in-time evidence,not chain-allocation residency,
per-copy page-size validation orpermission to assume zero translation cost.
First TLB session completed withthe full end marker; second started normally.
Both TLB sessions arenow complete andvalidated. S2 realW13 is1090.44us at25/25
versus1276.78us at38/12,withbothTLB-event medians1. Sequential2 is3112.96
versus5681.18us,withbothTLB-event medians2. Isolated random2 stillhas222659
L2 cache refills,butonly2 L2 TLB refills and4 DTLB walks. These observations
do notsupport a large extra-walk-count explanation; counts arenot latency and
counter sensitivity/allocation-specific residency remain limitations.
Full results:`tmp/dependent_load_tlb_20260908/report.json`;sameanalyzer command
asbelow withthe TLB directory substituted. RawS1 SHA256:
`3dcf0defbc76ddaa463c389195c2722ba547429676f5af901872dc5d8210addd`;
rawS2:`5a11e9bff6a4d1fc5b2cf35d1fd3e6ac5b017520e4da3d720d853ec719ec6d7a`.
Next controlled intervention isrecorded in `load_pc_geometry_20260908.md`.

Separate latency-sensitive response from dependence-independent throughput at
fixed background count/distribution. This is class E Lab only, no production,
kernel, model formula or calibration change. Previous goal turn made progress
by falsifying fixed-fraction transfer from B/AB probes; this experiment targets
the remaining service/concurrency identifiability gap.

## Probe contract

One8-byte pointer load per64-byte line;131072 lines=8MiB line coverage but only
1MiB explicit pointer payload. Four copies in one persistent allocation. Do not
equate this instruction workload with the8MiB packed-B stream.
For1/2/4/8/16 chains, every line is visited exactly once. The same deterministic
permutation is partitioned by chain count; round-robin logical visit order is
preserved across counts. Sequential and random permutations are separate factors.
Before each cell, rebuild selected-copy links and audit disjoint complete coverage;
then existing256MiB scrub per LLC runs. Timed volatile dependent pointer loads
finish at known starting pointers; native checks endpoints before reporting.

Chain count is programmed independence, not actual hardware outstanding misses.
One chain also includes dependent-load/address/branch overhead. Random access
changes prefetchability and translation locality; without separate TLB accounting,
its latency is not pure DRAM latency. No hardware prefetch or system settings
are changed. C++17 portable implementation requires generated-code inspection
on the target before interpreting results; extra spills/control limit inference.

## Hardware protocol

Arm-codex-internal NUMA3,foreground304/controller240. Fixed50 backgrounds are
25/25,38/12 and12/38 relative to the foreground LLC; isolated retained. Each
condition includes10 chain variants plus real/B/AB/no-store/empty bridges,
60 cells per round. Existing M12/1T backgrounds,5ms lead-in,copy rotation,
workspace and PMU gates are unchanged. New optional32MiB chain allocation
means historical binary/allocation identity differs; use within-session contrasts.
Run smoke before two independent5-warmup/31-round sessions,seeds339808/349808.
No fitting. Artifacts use `tmp/dependent_load_service_20260908/` locally and
under `/home/zhangxu/codex/fused_cpp/` remotely.

## Status

Portable coverage/endpoints acrossall orders,counts,copies andsmall/full footprints
passed alongside original placement tests (27 tests); broader focused regressions
70 passed. Target GCC13.2.0 build and60-cell PMU/numerical/endpoint smoke passed.
Disassembly audit confirms loops contain exactly1/2/4/8/16 `ldr xN,[xN]`
instructions with independent registers across chains and no loop-body stack
loads/stores. The16-chain prologue/epilogue saves registers andinitial endpoints;
these bounded costs are included in timing. No SVE vectorization of pointer loads.
Formal sessions339808/349808 started sequentially after those gates; results
pending. Full theoretical small-M model remains open.

## Analysis readiness

`analyze_dependent_load.py` validates complete31/5 sessions, matching binary
andindependent seeds, every chain endpoint andthe full60-cell grid. It reports
all raw-derived cell summaries, matched placement changes andwithin-order
single-to-multiple-chain scaling. Coverage GB/s is explicitly not DDR bandwidth;
ns/pointer-load includes loop,translation andcache-path effects.
Focused analysis/coverage/placement tests35 passed,including incomplete-session
rejection. Ruff anddiff checks passed. First formal session hascompleted;
second was inprogress atthat checkpoint. Both arenow complete; results above
supersede the checkpoint. Full JSON retains all cells,not just table endpoints.

Binary SHA256:`29c17726d65e44b397130347b6ed8dfd2ebaa290d67a51c7a41f61e622be6989`.
S1 SHA256:`aef8080b5b96fefab1be2a14cd2ae4a63d184fd5fc1ffc6bd254b18c6543a033`.
S2 SHA256:`c054786c3456b218db5fd26a8b5c19fa8b8164782f5b08618259e7bd99357c07`.
Actual native sources/binary remain in the remote snapshot. Local raw sessions
andreport.json are complete andvalidated; no remote run remains active from
the formal two-session command (exit0).

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_dependent_load.py \
  --sessions tmp/dependent_load_service_20260908/session1.jsonl \
  tmp/dependent_load_service_20260908/session2.jsonl \
  --output tmp/dependent_load_service_20260908/report.json
```
