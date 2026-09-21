# 8T error accounting on current no-team/gather16 model, 2026-09-10

Read-only diagnosis using current `planner_gather16` predictions and the existing median/uniformish paired-plan trace sessions. No model edit, new fit, planner search or hardware run. Current gather16 adapter leaves8T unchanged; all empirical team factors remain disabled.

## 8T lane accounting

Each value is mean cumulative time per8T lane, across two31-call sessions, in ms. Gather/W13/W2 are traced task envelopes. The gap residual is lane end minus the sum of those envelopes per call; it is an accounting residual, not an identified synchronization primitive. Independently summarized medians do not sum exactly.

| Case | Mean lane underprediction | Missing gather | W13 underprediction | W2 underprediction | Unassigned gap residual |
| --- | ---: | ---: | ---: | ---: | ---: |
| median | 2.274 | 0.917 | 0.807 | 0.324 | 0.213 |
| uniformish | 2.775 | 0.955 | 1.172 | 0.396 | 0.229 |

Gather alone explains roughly34–40% of the mean8T lane gap; gather plus unassigned gaps explains about43–50%. It is a material missing component, but not a complete explanation. Both W13 and W2 remain low. Adding actual gather as an arithmetic quantity is not a prediction of the event simulator's new makespan, since inserting phases changes overlap/resource pressure.

## Which M ranges are inaccurate?

Relative error is (sum of predicted placed stage durations / observed summed stage durations −1), not whole-plan latency. Actual sums are medians per session then averaged. Negative values are underprediction.

| Case | M group | Experts on8T | W13 error | W2 error |
| --- | --- | ---: | ---: | ---: |
| median | M1-12 | 81 | -4.44% | -4.11% |
| median | M13-48 | 59 | -4.68% | -12.77% |
| median | M49-192 | 32 | -4.16% | -3.05% |
| median | M193+ | 10 | -4.94% | -1.07% |
| uniformish | M1-12 | 91 | -2.78% | -0.60% |
| uniformish | M13-48 | 66 | -9.73% | -14.37% |
| uniformish | M49-192 | 63 | -6.29% | -3.71% |
| uniformish | M193+ | 14 | -6.17% | -2.09% |

The largest relative8T kernel gap is W2 at M13–48:12.77%/14.37% low. The M1–12 group is only about0.6–4.4% low. W13 also has a broader4–10% low bias in several groups. There is no evidence here for fixing8T solely with a constant width penalty or solely by adding gather. Traces cannot uniquely distinguish isolated shape-cost error from external competition error.

## Plan bottlenecks and limits

Uniformish is all8T. The model predicts lane56 as critical; hardware also ends at lane56 in49/62 samples, with other samples ending at24/32/40. Thus its main problem is absolute timing rather than the severe width-class critical-path inversion previously seen in high-skew. Predicted total24.848ms versus27.822/27.830ms measured.

Median has one16T lane plus eight8T lanes. Hardware ends at16T lane0 in all62 samples; the model instead predicts8T lane32. Current lane0 prediction is25.145ms versus28.330/28.495ms actual. Its gather is predicted0.983ms versus1.302/1.321ms actual, W13 predicts16.145ms versus17.893/17.974ms and W2 predicts8.017ms versus8.747/8.743ms. Therefore median's overall error must not be assigned exclusively to8T, and the high-skew16T kernel agreement does not automatically transfer to this route workload.

A limited read-only transfer check uses the existing16T gather coefficients with worker_rows=ceil(M/8), without fitting or changing the model. It yields mean8T gather0.950ms for median and0.951ms for uniformish, versus0.917/0.955ms measured. This makes an explicit8T gather candidate plausible at aggregate level, but per-expert errors, placement effects and a held-out validation have not been assessed; it is not adopted in this diagnostic.

## Reproduction

`.venv/bin/python tmp/gather16_model_20260910/diagnose8.py` reconstructs all lane predictions, stage sums and M-group errors in `diagnose8.json`. Inputs are the three-case configuration, the two retained frontiers and their `analysis1.json`/`analysis2.json` under `tmp/planner_repaired_pressure_20260910/`, plus current frozen model profiles. Current predicted totals match `overall_fixed_plan_report.json` exactly.

Same CPU240–319/NUMA3, H4096/F512, BF16 SVE256, N tile16, full stripes, early mergeoff and original two31-call sessions as the parent report. No new tests or hardware validation is claimed for this read-only analysis. The next concrete step is explicit8T gather validation, while separately examining W2 M13–48 and the residual W13 error; no opaque team factor is reintroduced.
