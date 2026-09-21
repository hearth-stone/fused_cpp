# High-skew selected plans: predicted and measured lane completion, 2026-09-10

This read-only diagnostic reuses both existing high-skew trace sessions and reconstructs both models' timelines. No new hardware measurement or model change. Plan names `anchor` and `repaired` refer to the current-baseline and repaired-adapter selections from [the planner comparison](planner_repaired_pressure_20260910.md). Lane ids are rank-relative core begins; physical CPU is240+lane. All completion times use the existing scheduled-compute origin and last-W2 endpoint.

## Main finding

For these two plans, **1T lanes are predicted too fast, while16T teams are predicted too slow**. With the repaired model, every one of the64 single-thread lane/session observations underestimates completion, and all16 wide-team observations overestimate completion. Thus an overestimated overall makespan does not imply overestimated1T time: the model falsely selects a16T critical path.

| Plan | Width | Baseline-model mean signed lane error (ms) | Repaired-model mean signed lane error (ms) |
| --- | ---: | ---: | ---: |
| anchor | 1 | -1.876 | -1.912 |
| anchor | 16 | +3.514 | +3.481 |
| repaired | 1 | -1.972 | -2.003 |
| repaired | 16 | +3.449 | +3.421 |

Both models predict lane32 as critical in both plans. Actual anchor critical lane is69 in all62 measured calls; repaired critical lanes are77/78 in61 calls and69 in one. The M1-only adapter barely changes the mean error of either width and does not fix this critical-path inversion. No conclusion about isolated1T accuracy can be inferred solely from these concurrent lane endpoints.

## All lanes

Tables use medians for each of the31-call sessions; predicted values are deterministic model outputs. Do not obtain whole-plan median by taking the maximum of separately computed lane medians: the slowest lane can change per call.

### anchor

| Lane | Width | Baseline prediction (ms) | Repaired prediction (ms) | Actual S1 (ms) | Actual S2 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 30.863 | 30.824 | 27.672 | 27.766 |
| 16 | 16 | 30.195 | 30.163 | 26.497 | 26.380 |
| 32 | 16 | 31.174 | 31.139 | 28.255 | 27.774 |
| 48 | 16 | 30.218 | 30.190 | 26.207 | 26.235 |
| 64 | 1 | 25.613 | 25.738 | 28.009 | 28.081 |
| 65 | 1 | 25.952 | 26.061 | 28.248 | 28.264 |
| 66 | 1 | 26.180 | 26.216 | 27.928 | 28.007 |
| 67 | 1 | 26.701 | 26.588 | 28.402 | 28.432 |
| 68 | 1 | 27.341 | 27.259 | 29.154 | 29.193 |
| 69 | 1 | 27.289 | 27.178 | 29.301 | 29.347 |
| 70 | 1 | 27.320 | 27.223 | 29.063 | 29.120 |
| 71 | 1 | 26.870 | 26.698 | 28.485 | 28.549 |
| 72 | 1 | 25.717 | 25.840 | 27.208 | 27.280 |
| 73 | 1 | 25.581 | 25.705 | 26.799 | 26.973 |
| 74 | 1 | 25.923 | 26.034 | 27.447 | 27.511 |
| 75 | 1 | 26.444 | 26.278 | 28.005 | 28.038 |
| 76 | 1 | 26.365 | 26.207 | 28.280 | 28.287 |
| 77 | 1 | 26.257 | 26.269 | 28.586 | 28.542 |
| 78 | 1 | 26.263 | 26.108 | 28.563 | 28.544 |
| 79 | 1 | 26.365 | 26.206 | 28.384 | 28.370 |

### repaired

| Lane | Width | Baseline prediction (ms) | Repaired prediction (ms) | Actual S1 (ms) | Actual S2 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 31.004 | 30.970 | 27.889 | 27.945 |
| 16 | 16 | 30.194 | 30.167 | 26.555 | 26.381 |
| 32 | 16 | 31.315 | 31.285 | 28.476 | 27.972 |
| 48 | 16 | 30.358 | 30.336 | 26.448 | 26.480 |
| 64 | 1 | 25.613 | 25.738 | 28.107 | 28.189 |
| 65 | 1 | 25.952 | 26.061 | 28.356 | 28.388 |
| 66 | 1 | 26.180 | 26.216 | 28.020 | 28.116 |
| 67 | 1 | 26.583 | 26.566 | 28.483 | 28.521 |
| 68 | 1 | 26.376 | 26.216 | 28.239 | 28.304 |
| 69 | 1 | 26.286 | 26.130 | 28.500 | 28.530 |
| 70 | 1 | 26.333 | 26.176 | 28.129 | 28.182 |
| 71 | 1 | 26.666 | 26.666 | 28.456 | 28.475 |
| 72 | 1 | 25.717 | 25.840 | 27.243 | 27.346 |
| 73 | 1 | 25.581 | 25.705 | 26.872 | 27.004 |
| 74 | 1 | 25.923 | 26.034 | 27.515 | 27.598 |
| 75 | 1 | 26.444 | 26.278 | 28.075 | 28.135 |
| 76 | 1 | 26.365 | 26.207 | 28.355 | 28.368 |
| 77 | 1 | 26.257 | 26.269 | 28.654 | 28.615 |
| 78 | 1 | 26.263 | 26.108 | 28.620 | 28.616 |
| 79 | 1 | 26.365 | 26.206 | 28.458 | 28.453 |

## Limits and reproducibility

Lane endpoint errors combine isolated stage-cost error, pressure response, modeled overhead placement and actual scheduling gaps. They cannot uniquely attribute the missing1T time to external competition. The existing model also does not necessarily expose gather under the same stage label as native tracing, so a raw stage-label subtraction is not a causal decomposition. The retained JSON contains full per-task isolated/placed timelines and actual stage sums for further diagnosis.

This is the same CPU240–319/NUMA3 high-skew route, H4096/F512, BF16 SVE256,80 threads, full stripes and early mergeoff measured in the parent report. Both selected plans use4×16T+16×1T. The repaired adapter only changes M1/1T history0 response and uses experimental competitor mapping; it is not full panel-history integration.

Reproduce from the repository root with `.venv/bin/python tmp/planner_repaired_pressure_20260910/compare_planner_lanes.py`. Inputs are the retained `input.json`, `high_skew/frontier.json`, `high_skew/analysis1.json`, `analysis2.json` and the frozen model/profile paths in the parent experiment. Full results are `lane_comparison.json` and `lane_comparison.csv` in that directory. Reconstructed makespans were checked against the prior common-plan scores. No code tests or benchmarks were rerun for this read-only diagnostic.
