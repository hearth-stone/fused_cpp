# Overall fixed-plan comparison after team-factor removal and16T gather, 2026-09-10

Read-only replay on the same four unique selected plans and two existing hardware sessions per plan. No new full search or hardware run. Extend the latest no-team/gather adapter evaluation from high-skew to the retained median and uniformish bridges; no fitting is performed.

## Same-population accuracy

| Model | Observations | MAE (ms) | MAPE | Signed mean error (ms) |
| --- | ---: | ---: | ---: | ---: |
| Original active CorePressure baseline | 8 | 1.655 | 5.766% | +1.655 |
| M1 repair with team factors | 8 | 1.638 | 5.709% | +1.638 |
| M1 repair, no team factors | 8 | 2.643 | 9.266% | -2.643 |
| No team factors plus explicit16T gather | 8 | 2.462 | 8.643% | -2.462 |

Current overall accuracy is worse than the original baseline on this fixed population. Team-factor removal changes a global high bias into a global low bias. Adding16T gather improves part of the low bias but does not finish the missing-cost accounting. Earlier high-skew-only metrics must not be compared directly with this eight-observation aggregate.

## Per-plan comparison

| Fixed plan | Original baseline (ms) | Current gather16 model (ms) | Actual S1 / S2 (ms) |
| --- | ---: | ---: | ---: |
| high_skew/anchor | 31.174 | 26.894 | 29.301 / 29.347 |
| high_skew/repaired | 31.315 | 26.952 | 28.654 / 28.628 |
| median/anchor | 29.908 | 25.661 | 28.330 / 28.495 |
| uniformish/anchor | 28.426 | 24.848 | 27.822 / 27.830 |

Uniformish is all8T and has no added gather stage; median is1×16T+8×8T, so most of its tasks also lack the new explicit gather. Width8 residual penalties were removed along with all other team factors. Their replacement by explicit costs is not complete. This does not establish that all remaining8T error is gather.

In median, adding16T gather changes predicted total25.941→25.661ms, a decrease despite adding phase time. The event model changes overlap/resource pressure on the other teams. This non-additive response further limits a simple interpretation of the improvement from adding one stage; no hardware speedup is inferred from it.

The previous measured high-skew selection improvement2.19%/2.42% belongs to the earlier M1-only adapter with retained team factors. The current no-team-plus-gather version has not undergone a new full search/selected-plan benchmark. It must not inherit that speedup claim.

## Current conclusion

The16T missing-gather cost is partly accounted for, but overall absolute accuracy has not surpassed the original baseline. Remaining limitations include1T/M1–4 joint stage response, opaque1T operator residual accounting, missing explicit8T gather/overhead, and16T residual gaps. Current relative ranking of the two high-skew plans remains wrong. Keep these negative results with the earlier microbenchmark improvements.

Artifacts: `tmp/gather16_model_20260910/overall_fixed_plan_report.json`, the original `tmp/planner_repaired_pressure_20260910/report.json`, and the frozen profiles referenced by the gather16 model. Reconstructed predictions use `model_timeline` on each retained bridge, both adapters and the original case counts/calibration. Both historical models are cross-scored on the identical common-plan population. Same original CPU240–319/NUMA3, H4096/F512, BF16 SVE256, early mergeoff, full stripes and two31-call measured sessions as the parent experiment. No source/model parameter change or test rerun is needed for this read-only aggregate.
