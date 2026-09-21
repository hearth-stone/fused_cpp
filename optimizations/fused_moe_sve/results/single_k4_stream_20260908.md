# Single-K4 SVE stream organization

## Result: no large transfer ofthe scalar stream effect

Both complete32-cell5/31 sessions pass per-round instruction ledger,numerical,
placement andPMU gates. Same dynamicwork withsingle-K4 sites doesnot remove
thelarge local-background penalty. B-only effects aremostly indistinguishable
fromzero;AB has a smallrepeatable high-pressure improvement;full-no-store
doesnot show a large consistent improvement. Do notfit astream-loop correction
toexplain the16–18% distribution residual fromthiscontrast.

| Background | B change S1/S2 % | AB change S1/S2 % | Full change S1/S2 % |
| --- | --- | --- | --- |
| isolated | -0.07/+0.13 | -0.48/-0.22 | +1.22/+1.70 |
| 25/25 | -0.01/-0.03 | -0.84/-1.34 | +0.66/-0.46 |
| 38/12 | -0.77/-0.53 | -0.80/-1.52 | +0.99/+0.37 |
| 12/38 | +0.05/-1.11 | -0.78/+0.03 | -2.13/-1.12 |

Percentages arematched-round single-minus-original medians. AB25/25 CI95:
[-2.00,-0.03]/[-3.07,-0.75]%;AB38/12:[-1.66,-0.21]/[-2.62,-0.92]%.
Isolatedfull regresses slightly inboth sessions,CI[0.82,1.45]/[1.11,2.10]%.
12/38 haswide intervals,so no exactnull/equivalence claim. Fullreport retains
allabsolute values,distributions,counters andpointwise paired intervals.

This bounds oneparticular SVE reorganization,not everyrequest-arrival orprefetch
effect. The scalar pointer-chain anomaly cannot bedirectly promoted into aSVE
model term. Focus again onforeground-specific service atsimilar requestvolume,
using betterobservability rather thanadditional unconstrained residuals.

## Next observable candidate: HHA request path

Read-only targetdiscovery foundfour HHA devices ineachSCCL25/27 withnamed
events rx_ops_num(config0x0),rx_outer(0x1),rx_sccl(0x2),hha_retry(0x2e),
cycles(0x55). Examplehisi_sccl25_hha0 hascpumask280,type228. These are
discovered aliases only,not validated pressure/latency features. No HHA data
wascollected inthisexperiment. Beforeuse,verify scope/sensitivity withcontrols,
event-running ratios andcounter window effects;retry counts mustnot beconverted
into delay withoutindependent evidence. Production andcalibration remain frozen.

Class E Lab experiment;production JIT andfrozen model unchanged. Addsingle-K4
loop variants forB-only(probe75),AB(probe76),loaded4 full-no-store(probe77).
Original probes1/3/4 andrealW13 remain thebaseline. Addresses,K4096/N1024,
M1/SVE256,A/B payload andtotalmatrix operations remain unchanged. No tiling,
cache prewarm orworking-set reduction.

Original steadyloop alternates twoK4 load sites andloads the nextpanel before
computing thecurrentpanel. Singleloop usesoneK4 site andcomputes immediately
afterloading it. EachB loadPC nowadvances128 rather than256bytes periteration;
eachA loadPC advances64 rather than128bytes. Dynamiccount percall remains
262144 B vectorloads,65536 A broadcastloads whenpresent,262144 BFMMLA when
present. These areinstruction-request counts,not refill bytes.

This changes staticPC count,per-PC strides,cadence andload-compute overlap
together. Supply-only variants helpseparate effects requiringmatrix execution,
but nooutcome uniquely identifies hardware prefetch. The variant isnot a
proposed optimization orproduction replacement.

Protocol: --stream-loop,32 cells perround atisolated and25/25,38/12,12/38
M12/1T backgrounds. Existing NUMA3/CPU304 foreground/240 controller,4-copy
rotation,256MiB scrub perLLC,5ms lead-in,persistent outputs andsix-event TLB
PMU group. Two5/31 sessions,seeds419808/429808,aftertarget correctness smoke.
Expected matched instruction increments:ABminusB=65536,fullminusAB=262144,
apart frombounded wrapper branch differences. Existing default-generator versus
production B/full machine-byte checks muststill pass.

Localgrid/regressions46 passed;Ruff anddiff checks passed. Target build,smoke,
instruction ledger andformal measurements pending. Artifacts:
`tmp/single_k4_stream_20260908/` locally andunder
`/home/zhangxu/codex/fused_cpp/` onArm-codex-internal.

## Target validation checkpoint

GCC build and32-cell smoke passed. Eachbackground conditionhasidentical new
instruction counts:B525439,AB590975,full853119. Differences65536 and262144
exactlymatch thedeclared ledger. Original B/full machine-byte identity checks,
real/background numerical checks,output poisoning andPMU running gates passed.
Formal5/31 sessions419808/429808 started afterthese checks; results pending.

Analysis validates theinstruction ledger perround inallconditions,not onlycell
medians. Localanalysis/grid/coverage tests49 passed;Ruff anddiff checks passed.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_stream_loop.py \
  --sessions tmp/single_k4_stream_20260908/session1.jsonl \
  tmp/single_k4_stream_20260908/session2.jsonl \
  --output tmp/single_k4_stream_20260908/report.json
```

## Completed identities

Formalcommand exited0; complete results supersede pending checkpoint.
Binary:`dfc309f5494ff5200c6bf2ba286b6c1e1486b9ec3286bad48081d86e2a9f46be`.
S1:`4ce63a8d3d883157b1d1d6280bf1c0388617bf39e7974d6c5806e9b9ba9aadbd`.
S2:`f1ea773e68eb7d6575601d07362ed0695dd58cd666abcce09e3ccf97e7540fca`.
