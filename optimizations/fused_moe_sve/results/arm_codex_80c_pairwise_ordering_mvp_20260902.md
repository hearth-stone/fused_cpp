# Arm-codex 80C anchor-relative pairwise-ordering MVP

## Evidence status

This is a direct-sync provisional experiment, not a clean-commit paper run.
The local source base was `935d643` plus the uncommitted two-level evaluator,
pairwise-report, and serial-lane swap-probe changes. The production extension
was unchanged.

- Machine: `Arm-codex-internal`, NUMA3 CPUs `240-319`, memory node 3.
- Calibration: `analytic_machine_numa3_80c_narrow_merge_v8_20260903.json`.
- Calibration SHA256: `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
- Extension SHA256: `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- BF16 fused expert, `H=4096`, `F=512`, 256 experts, 2048 tokens, TopK6.
- Hardware measurement: five warmups, 31 randomized paired rounds, four
  rotating packed-weight copies, strict fixed whole-expert Plan V2.

Raw artifacts remain in ignored workspace storage:

| Artifact | SHA256 |
| --- | --- |
| `tmp/moe_pairwise_temporal_probe_20260902.json` | `c0ad21f6f6267757d87524d6f03f1c8805de7ac861b0d4765e55ffdc9094c908` |
| `tmp/pairwise_uniformish_holdout_20260902.json` | `bba234ae99508af9fc9f7026b43150e8c6e09f50026049e53f4392a4d908a8d8` |
| `tmp/pairwise_median_holdout_20260902.json` | `fe103f49f816d930b12c021e2ca5d8966bbf0283670c61f245f3464f5d3f92d3` |
| `tmp/pairwise_high_skew_holdout_20260902.json` | `b23ece71ffecbbbc839747ac4be679b34c02c7998a3737764aeee63964ddb67f` |
| `tmp/moe_pairwise_ordering_same_v8_cross_session.json` | `7eff68cb0ea241a63c011e8368e4ec7152151164f47496f6f2887a45dd1a2069` |
| `tmp/pair_context_high_skew_20260902.json` | `7c9bf58d3a76ddc0a85da037445a0a008f86151cee84305b3966446695ad8efb` |
| `tmp/moe_tail_head_probe_20260902.json` | `2cb017c298ceec2368b3cd8ab2dea20866fd6d6cc0b26f1979c373508b28a3f3` |
| `tmp/pair_context_report_20260902.json` | `d18af4b73907bd336b8194815ffa24a6017f2653e44ebb58f650f976e40b84ac` |

## Pairwise method

The analytical formula and event-model mean were not changed. The MVP fits a
symmetric absolute residual radius for predicted candidate-versus-anchor gain,
with exact-context, operator-family, then global fallback. A candidate is
declared better or worse only when its complete calibrated interval clears the
anchor; otherwise it is incomparable. The 2% actionable threshold remains a
separate hardware-label and final-adoption criterion.

The fit used the first formal v8 width-neighborhood session. The evaluation
used the new same-calibration, same-extension session. This is a cross-session
repeat over the same three routes and mostly the same deterministic candidate
states, not an unseen-route or unseen-plan holdout.

| Metric | Result |
| --- | ---: |
| Fit pairs | 41 |
| Evaluation pairs | 41 |
| Hardware-resolvable pairs | 3 |
| Partial-order decisions on resolvable pairs | 0 |
| False dominance | 0 |
| Better / worse / incomparable | `0 / 0 / 41` |
| Measured-best recall at top-8 | `3 / 3` |
| Measured-best recall at top-16 | `3 / 3` |

The current v8 residual calibration is safe but vacuous for local search. Its
90th-percentile family radii are about `5.389` percentage points for order
moves and `0.846` points for width moves. The `same_width_cross_lane_swap`
context alone retains an `8.605`-point radius. Consequently no candidate can be
pruned or accepted confidently on this repeat. Do not connect this comparator
to VND/LNS yet.

The broader development replay also retained the known Arm 80C `cp_sat_06`
counterexample: predicted gain `-0.605%`, measured paired-median gain `+4.131%`.
The comparator changed the relation from a point-model loss to incomparable,
so the plan remained eligible for measurement. That replay mixed older model
versions and is diagnostic only, not a calibration holdout.

## Frozen-v8 three-trace repeat

| Trace | Baseline | Measured shortlist best | Retained-baseline regret | Event/hardware Spearman | Stable candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| uniformish | 32.813 ms | 32.616 ms | 0.603% | 0.713 | 0 |
| median | 32.545 ms | 32.299 ms | 0.763% | 0.186 | 0 |
| high-skew | 32.825 ms | 32.219 ms | 1.882% | 0.104 | 1 |

The high-skew stable candidate was a domain-local `1T+1T -> 2T` merge. It
measured `+1.873%` paired median, `+0.574%` P10, `+3.428%` P90, and 29/31 wins,
while the event model predicted only `+0.097%`. It remains below the declared
2% actionable median threshold and does not reopen deterministic width-VND.

## Two-level screen

The existing uncommitted evaluator caches exact scores by canonical state hash
and reuses immutable lane phase descriptions for a cheap phase surrogate. On
the new run, exact event scoring sustained `6.34-9.90` plans/s, while screening
sustained `186-349` plans/s. Depending on per-operator budget, the projected
exact-call reduction was about `42-85%` and projected wall-time speedup was
`1.57-5.06x`.

The fidelity gate did not pass. The union of critical and random screens kept
the measured best on median and high-skew, but every `8/16/32` per-operator
budget missed the uniformish measured-best state. That state was not stable and
the retained-baseline regret was only `0.603%`, but the checklist explicitly
requires not discarding the measured best. Keep the two-level evaluator in Lab.

## Independent serial-lane swap probe

The residual report selected same-width cross-lane swaps as the largest local
error family. The narrow-lane transition benchmark was extended with two 1T
lane-head swap modes under isolated and mixed `4x16T + 14x1T` background
conditions. These modes and a balanced-load `swap_sensitive` case were excluded
from all parameter fitting.

For the background comparison, measured minus original-v8 predicted swap gain
was approximately `-1.00`, `+3.76`, `-0.96`, and `-0.25` percentage points for
balanced, head-heavy, swap-sensitive, and tail-dense cases. The deliberately
imbalanced swap-sensitive case was predicted at `-18.40%` and measured at
`-19.36%`. The probe therefore does not support a generic temporal-swap
correction. The one larger head-heavy residual is not enough to identify a new
machine parameter.

An attempted output calibration from the already narrow-calibrated v8 source
would fit the same residual twice and was rejected. The benchmark now refuses
such second fits. The original v8 calibration and SHA remained frozen for the
three-trace repeat.

## Placed-context follow-up

The hardware shortlist now stores the exact affected lanes, ordered before and
after expert/route sequences, isolated lane loads, and a compressed placed-event
summary. The summary records affected head/tail exposure, phase and team
pressure dilation, peer threads, cohort transitions, and modeled critical
lane/expert switches without embedding the full event log.

A same-v8 high-skew repeat measured four cross-lane swaps. Three bad swaps moved
a `62`- or `68`-route task from a 1T lane tail to another 1T lane head while
moving a `1`- or `4`-route task in the opposite direction. Their modeled
affected-tail exposure increased by `4.02-5.70 ms`, and hardware regressed by
`12.61-15.17%` paired median with negative P90. The fourth swap added no tail
exposure and measured `+0.40%` median with an interval crossing zero. None of
the four changed the placed-event critical lane or critical expert, so a
critical-lane switch does not explain this residual split. The bad candidates'
robust predictions were `-3.75%` to `-7.13%`, leaving approximately `-6.34` to
`-9.61` percentage points of residual. Grouping the development replay by the
new analytical context gives a `9.61`-point residual radius for the three
tail-increasing swaps versus a `0.37`-point residual for the no-new-tail swap.
This is descriptive same-data grouping, not a held-out calibration result.

An independently constructed holdout then exchanged a `68`-route 1T lane tail
with a `1`-route peer-lane head. This reproduced the physical slowdown, but not
the missing-model residual: hardware/model swap gains were `-32.40/-34.35%`
under full background and `-34.97/-34.80%` in isolation, residuals of `+1.95`
and `-0.17` percentage points. Across all five tail/head cases the absolute
residual stayed below `1.96` points. Therefore tail-to-head load transfer is a
real structural hazard already represented by the current lane/event model; it
does not justify a new physical parameter. The larger real-trace residual still
requires a more complete context than this independently reproducible
transition.

## Decision

The minimum pairwise version closes the reporting and safety questions but
fails the usefulness gate:

1. keep the anchor-relative comparator and report as offline diagnostics;
2. do not use the current partial order for VND/LNS pruning or acceptance;
3. keep the two-level screen in Lab because the strict measured-best recall
   gate failed on uniformish;
4. do not add a generic lane-swap correction or replace calibration v8;
5. retain affected-lane and placed-event summaries in future hardware
   shortlists; use them to refine offline residual contexts;
6. do not add a tail/head parameter: the independent probe reproduced the
   slowdown but not the multi-percent residual. Require a more specific
   independently reproducible residual before changing calibration.

No production planner, Plan V2 schema, kernel, ABI, numerical behavior, or
default dispatch changed.

## 2026-09-04 offline integration update

After the absolute pressure proxy line was closed, the conservative relation
was connected to the offline neighborhood shortlist and a reusable
best-improvement loop. This changes plumbing, not the 2026-09-02 effectiveness
conclusion: only `candidate_better` may replace the current anchor;
`incomparable` stays eligible for measurement; only `candidate_worse` is
dominance-pruned. Candidates omitted only because of the hardware budget are
reported as `budget_deferred`, never as model-pruned. The audit runner rejects
a report unless its calibration/extension identity matches and its
zero-false-pruning plus requested top-K recall gates pass.

The same-v8 41-pair replay remains unchanged: 41 incomparable, zero false
pruning, and all three measured bests retained at K=8/16/32. A broader
cross-identity diagnostic replay over 129 historical pairs also has zero false
pruning and keeps the high-skew `cp_sat_06` direction reversal incomparable.
That mixed replay misses three measured bests at K=8 (shortlist regret
0.083%--0.299%) but retains every measured best at K=16, so the guarded offline
default is K=16. No current same-v8 candidate is confidently better; useful
multi-start VND/LNS behavior remains to be demonstrated.
