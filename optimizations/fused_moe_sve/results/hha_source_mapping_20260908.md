# HHA receive-source mapping controls

## Completed result andmodel boundary

Both85-cell5/31 sessions completed andpassed fullraw-reader,numerical,placement
andPMU gates. Forallfive one-sided counts8/16/24/32/39,theexpected receiving
group's rx_sccl rate increases monotonically inboth sessions.

Known CPU-source280–319(background excludes304) maps predominantly toHHA
group27 rx_sccl. CPU-source240–279(excludescontroller240) maps predominantly
togroup25. This isempirical receive-source mapping for thisNUMA3 allocation,
not proofthatreceiver groupnumber labels theforeground's ownLLC oritswaiting.

| Case | S1/S2 foreground us | S1 rx_sccl25/27 Gevents/s | S2 |
| --- | --- | --- | --- |
| same8 | 356.91/355.18 | 0.0008/0.4878 | 0.0016/0.4953 |
| same39 | 768.44/760.75 | 0.0008/2.0199 | 0.0014/2.0204 |
| other8 | 318.67/321.25 | 0.4380/0.0531 | 0.4372/0.0605 |
| other39 | 421.40/420.44 | 2.0004/0.0447 | 1.9908/0.0518 |
| balanced48 | 943.21/934.42 | 1.1006/1.1176 | 1.0977/1.1168 |
| balanced64 | 1676.11/1666.49 | 1.1338/1.1431 | 1.1308/1.1391 |
| balanced78 | 2051.19/2043.37 | 1.1375/1.1455 | 1.1358/1.1444 |

Balanced64→78 foreground timeincreases22.38/22.62%,butcombined rx_sccl rate
only0.27/0.45%. This demonstrates theinsufficiency ofthroughput-only pressure
features aftersaturation. Do notconvert thetraffic source proxy intoqueue delay.
Likewise mapping source fractions doesnot identify acceptedversusoffered demand,
inflight requests,foreground-specific latency or effectiveoverlap. These still
need independent measurements/interventions before aphysicaltime model canbe
claimed. No new coefficient orprediction gate isapproved bythisexperiment.

The next modeling step should explicitly retain thatidentifiability boundary:
arrival/accepted-throughput information andsource mapping arenow constrained;
queue/inflight state andforeground service remain unobserved. More receive-count
variants alone willnot close thegap. Preserve thisdataset forconditional model
checks,not asan untouched holdout afterusing ittochoose model structure.

Class E measurement. Reuse originaldual85-cell grid withHHA selection rather
thanfixed50-only grid. Known sources areCPU sets:foreground-side280–319
excluding304,opposite240–279 excluding240. Do notinfer physical SCCL labels
fromCPU numbering orPMU cpumask alone. rx_sccl semantics areincoming operations
fromanother SCCL;sourceandreceiver mustremain distinct.

--hha-source includesisolated,same/other counts8/16/24/32/39,balanced totals
16/24/32/48/64/78. Real/B/AB/full/empty controls foreachcondition. Reuse
`tmp/fixed_total_placement_20260908/phase_supply_native` unchanged. Core5 and
HHA40/DDRC48 groups unchanged frompreceding HHA contrast,asareworkspace,
NUMA3,4-copy rotation,scrub and5ms lead-in. No newlatency event orrawcode.

Smoke then twoindependent5/31 sessions459808/469808. Source mapping requires
opposite one-sided controls,consistent direction acrosscounts andsessions,and
retention ofempty/control contributions. Rxrates canidentify traffic source
patterns butcannot alone establish latency,queue occupancy orcausal slowdown.
Do nottreat thisalreadydesigned grid asan untouched prediction holdout afterfit.

Artifacts:`tmp/hha_source_mapping_20260908/` locally andunder
`/home/zhangxu/codex/fused_cpp/` onArm-codex-internal. Analysis reuses
`analyze_dual_m12.py` (full HHA fields retained throughshared summary),with
thefrozen model file usedonly foridentity/no-mutation check. Targetsmoke and
formal results pending. Production andmodel equations unchanged.

## Validation checkpoint

85-cell full-PMU/numerical/placement smoke passed;focusedtests39 passed,
including rejection ofmixed fixed50/source grids. Smoke rx_sccl group25/27
rates are0.0018/0.0591 isolated,0.0013/2.0096 same39,and1.9904/0.0464
other39 Gevents/s. This supports theintended source contrast butisnot the
formal repeated result. Background-off receiveractivity includesforeground and
controller contributions. Two5/31 sessions459808/469808 started aftersmoke.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_dual_m12.py \
  --sessions tmp/hha_source_mapping_20260908/session1.jsonl \
  tmp/hha_source_mapping_20260908/session2.jsonl \
  --frozen-fit tmp/layered_supply_model_20260908/frozen_fit.json \
  --output tmp/hha_source_mapping_20260908/report.json
```

Formalcommand exited0. Raw/report.json arecomplete locally. S1 SHA256:
`476c6dec8d3ecc45faed81bfdee5e9b29a336ea75d51e689c9e8b0f61d7d66db`;
S2:`d83a7a3437ba7a37291eef74492243407fcdf8cfa716846e861fe75b011d0cbf`.
Both useoriginal fixed50 native8a92c487... withnew HHA runner; frozen model
bytes unchanged byanalysis. Completed findings supersede smoke/pending checkpoint.
