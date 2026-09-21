# GEMM-average memory density: smooth versus uneven orders

Status: completed. High-skew smooth beats both uneven and anchor in both sessions;
median does not benefit. Lab-only; no physical model,
production planner, task membership, width, window or native-code change.

## Outcome

The simpler GEMM-average density objective found a useful high-skew order. It
passes both the direct smooth-versus-uneven comparison and the >2% actionable
gate against the current anchor. Median shows no corresponding benefit.
This supports retaining GEMM-average smoothing as a candidate generator, not
using it as a universal time model, pruning rule or physical correction.

Direct paired smooth-versus-uneven speedup (positive means smooth is faster):

| Trace | Session | Median | P10 | P90 | Wins |
| --- | --- | ---: | ---: | ---: | ---: |
| median | S1 | −0.405% | −1.160% | +0.285% | 6/31 |
| median | S2 | −0.110% | −0.708% | +0.436% | 12/31 |
| high-skew | S1 | **+3.695%** | **+3.280%** | +4.129% | 31/31 |
| high-skew | S2 | **+3.973%** | **+3.245%** | +4.720% | 30/31 |

Absolute medians and paired comparisons against the separately retained anchor:

| Trace/session | Anchor ms | Smooth ms | Uneven ms | Smooth vs anchor median / P10 |
| --- | ---: | ---: | ---: | ---: |
| median/S1 | 31.9234 | 34.1637 | 34.0311 | −6.629% / −7.153% |
| median/S2 | 31.9082 | 34.1277 | 34.0637 | −6.480% / −7.334% |
| high-skew/S1 | 31.3295 | **30.5974** | 31.7271 | **+2.355% / +1.951%** |
| high-skew/S2 | 31.7385 | **30.6431** | 31.8657 | **+3.464% / +2.784%** |

The high-skew anchor is the prior passing rotation, not the original selected
fallback. The smoother candidate therefore supplies additional measured gain.
The median smooth/uneven difference is small and overlapping across rounds;
neither beats its anchor. Do not claim a universal benefit of lower density
variance from one positive trace or claim actual hardware requests were verified
to be smoother: no PMU/phase timeline was collected in this test.

Every candidate passed bitwise BF16 correctness on all four copies in both
sessions. The analyzer validated full 36x3 cells, independent order seeds and
matching runner/frontier/extension identities before computing the direct
contrast. Raw evidence SHA256 values are embedded in each summary. Keep the
high-skew `plans.smooth_global.bridge` as this experiment's candidate; original
winner files are unchanged and no next layer or model fitting was launched.

High-skew candidate state hash:
`84e15dea8ce2ebda6de53583bc0396948761584932b2146a4acbf151c4be24d1`;
complete bridge SHA256:
`7a4d18aa963e965972e9e57c49f1569161a3cba61b699b8ec40c8713c5d433a9`.
Frozen v8 event time predicted this candidate at −6.487% gain versus anchor.
It was retained by the density-directed generation protocol despite that score,
providing another reason not to filter this candidate family solely by v8 rank.

## Question and fixed protocol

Compare the smoother schedule directly with a deliberately less uniform schedule
using the simpler GEMM-average rate, rather than the prior fine-phase proxy.
Combine consecutive W13 phases into one W13 interval, and W2 phases into one W2
interval. Each interval has density `sum(predicted DRAM bytes)/sum(isolated ns)`.
Retain non-GEMM overhead/gather intervals separately. This conserves duration and
traffic exactly up to floating-point rounding, without assuming uniform actual
hardware arrivals. Existing isolated spill/cache assumptions remain unmodified.

Use the same median anchor and improved high-skew rotation anchor as the previous
pressure experiment. Each trace gets 64 attempts to reduce global second moment,
64 to increase it, and no additional search. Keep anchor plus these two plans.
Fixed per-lane task set and width imply unchanged total modeled bytes and isolated
horizon. A lower second moment/CV-squared defines smoother in this experiment;
it need not minimize the instantaneous peak at every point.

Primary comparison: `100*(uneven_ns/smooth_ns - 1)` per paired round. Report median,
P10/P90 and wins in each of two independent sessions. Positive median and P10 in
both sessions supports a repeatable advantage on that trace; the previous
>2%-median/positive-P10 gate still governs actionable replacement of the anchor.
Do not confuse winning against a bad control with beating the current anchor.

Hardware reuses the unchanged bounded-order runner: three plans per trace,
five warmups, 31 effective randomized paired rounds, same copy per round,
four-copy rotation, disjoint 216 MiB scrub before each call outside timing.
CPUs240–319, NUMA3, frozen v8, E256/H4096/F512, BF16 SVE N tile8, 2048 tokens,
TopK6, full-owner-stripe windows0, identical merge policy and tensor seeds.
Two independent processes/order seeds per trace. Bitwise outputs on all four
copies must match anchor before timing. No PMU; conclusions concern scheduling
performance and the proxy association, not confirmed physical smoothness.

## Frozen density contrast

| Trace | Anchor CV² | Smooth CV² | Uneven CV² |
| --- | ---: | ---: | ---: |
| median | 1.114 | 0.696 | 2.349 |
| high-skew | 0.941 | 0.262 | 1.284 |

Generated inputs retain equal modeled traffic/horizon within rounding. Four
focused tests passed, including exact GEMM aggregation and unchanged phase mode.

## Reproduction

Local ignored directory: `tmp/gemm_density_20260907/`.
Remote: `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/gemm_density_20260907/`.

Use `generate_pressure_balanced_orders.py --density-mode gemm` with the same
source/anchor/calibration arguments as the preceding pressure-balanced-orders
report. Outputs are `median_frozen.json` and `high_skew_frozen.json`.
The default `--density-mode phase` retains the prior proposal behavior.

Measure with the unchanged `bench_bounded_order_extension.py measure`, substituting
this directory and frozen inputs in the previous report's command, using the
same route files and seeds20260906/20260907. No new native build or production sync.
Session JSON retains identities and every cell; summaries must validate the
complete 36x3 records before direct smooth/uneven analysis. Raw outputs and
previous experiments are never overwritten.

Direct analysis command (substitute high-skew names for its result):

```bash
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_gemm_density_comparison.py \
  --frontier tmp/gemm_density_20260907/median_frozen.json \
  --sessions tmp/gemm_density_20260907/median_session1.json tmp/gemm_density_20260907/median_session2.json \
  --output tmp/gemm_density_20260907/median_summary.json
```

Focused validation, including direct comparison denominator and shared state/
round-completeness checks: 16 tests passed. The direct analyzer's package import
issue was caught and fixed before analyzing actual hardware results. Both frozen
contrasts independently pass conserved-byte/horizon checks at relative tolerance
1e-12. No model or calibration is fitted from either trace.
