# Frozen v8 reassessment after anchor confirmation

Step3 completed before starting the dedicated smoothing experiment. Evaluation
only: no parameters fitted, no old total residual reused, no physical term added.
Baseline: profiles/workspace_numa3_80c.json. Evidence: the three profile-bound
frontiers and six sessions in tmp/workspace_baseline_confirmation_20260907/.

| Trace | Plans | Absolute MAPE | Relative-gain MAE | Confirmed anchor's model rank |
| --- | ---: | ---: | ---: | ---: |
| median | 3 | 13.913% | 3.944 percentage points | 3/3 |
| high-skew | 6 | 19.823% | 11.061 percentage points | 1/6 |
| uniformish | 5 | 3.160% | 1.275 percentage points | 3/5 |

Absolute error compares frozen prediction to the mean of the two measured
latency medians per plan. Relative error compares frozen anchor-relative gain
to the mean of the two paired-gain medians, excluding anchor. These deliberately
selected confirmation sets differ from earlier52-plan diagnostics; do not treat
changes between those tables as an accuracy trend. Neither is an independent
calibration-fit validation. High-skew includes weaker starting shapes, so its
relative error is not restricted to tiny same-width neighbor changes.

Median2ab43572 again passes hardware confirmation despite the old same-pair
candidate_worse label. The previous residual report does not bind to this output
lifecycle and remains unauthorized for hard pruning/acceptance. A single retained
counterexample is enough to reject extended-domain safety; the small sample is
not a false-pruning rate or full top-K recall estimate.

Decision: preserve frozen v8 as a point-score diagnostic, retain hardware rerank
and diverse elites, and calibrate only with a separately designed independent
dataset if requested later. Do not add a bandwidth penalty to absorb old output
first-write costs. LNS remains available; VND is retained for confirmed weak-start
repairs, not as a claim of global superiority. Proceed to step4 smoothing
confirmation against the step2 median anchor without automatic adoption.
