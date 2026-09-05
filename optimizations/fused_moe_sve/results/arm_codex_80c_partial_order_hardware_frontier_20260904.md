# Arm-codex 80C partial-order hardware frontier

## Decision

Adopt hardware-assisted beam search as the next offline search baseline. Do not
move directly to template-level LNS yet: two incomparable local neighbors were
independently stable above 2%, so the local neighborhood has real value and the
current bottleneck is comparator false-negative coverage rather than absence of
useful local moves.

This was the depth-1 decision. The completed corrected depth-2/depth-3 beam
subsequently stopped and handed off to template-level LNS; see
`arm_codex_80c_hardware_assisted_beam_20260904.md`. In particular, the two
stable relative winners below were not the absolute global best, so the
follow-up beam was corrected to expand absolute measured elites as well.

Keep partial-order pruning enabled. None of the eight deliberately measured
`candidate_worse` sentinels became a stable >2% improvement in both sessions.
Do not refit the residual calibration or lower the 2% margin from this result.

## Protocol

- Machine: `Arm-codex-internal`, NUMA3 CPUs `240-319`, memory node 3.
- Extension SHA256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- Frozen-v8 calibration SHA256:
  `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
- Pairwise report SHA256:
  `a1b89cec87cc6c9cacab4477965bd060b8f41ba7eac227ef055552609268e54a`.
- High-skew route: request 008/case 009, layer index 38, 2048 tokens,
  TopK6, `H=4096`, `F=512`, 256 experts.
- Frontier: four anchors, top-16 incomparable from each start, and two worse
  sentinels per start. Canonical-hash deduplication produced 76 unique plans:
  4 anchors, 64 incomparable plans, and 8 sentinels.
- Every frontier entry stores its canonical state and exact PlanV2 bridge. The
  builder recomputed every canonical hash and rejected any inconsistent bridge.
- Each session used one process, one shared hidden/routes/weights setup, four
  rotating packed-weight copies, five randomized warmup rounds, and 31
  randomized measured rounds over all 76 plans. Every plan was bit-exact against
  an anchor before timing.
- Session seeds: `20260911` and `20260912`. No calibration fitting consumed
  these measurements.

## Cross-session result

Session 1 found six comparisons with median gain above 2% and positive paired
P10; session 2 found five. Only the following two identical candidate/anchor
pairs passed in both sessions:

| Start / operator | Model predicted / lower | Session 1 median [P10, P90] | Session 2 median [P10, P90] |
| --- | ---: | ---: | ---: |
| fixed-width / same-lane adjacent swap | `+0.014% / -5.375%` | `+4.690% [3.029, 6.988]` | `+3.988% [2.022, 5.925]` |
| greedy / domain-local `16T -> 8T+8T` split | `+2.319% / +1.181%` | `+3.819% [2.503, 5.543]` | `+3.376% [1.965, 5.368]` |

The fixed-width swap exchanges the first two tasks on the 16T lane at logical
core 32: expert 135 with 1,341 routes and expert 74 with 261 routes. Absolute
anchor/candidate medians were `36.366/34.625 ms` and `37.517/36.113 ms` in the
two sessions. The event model predicted only `+0.014%`; the order-family radius
correctly kept it incomparable, but point ordering has almost no useful signal
for this hardware improvement.

The greedy split replaces the 16T lane at logical core 48 with two 8T lanes and
redistributes its 47 experts. Absolute anchor/candidate medians were
`37.782/36.304 ms` and `37.921/36.672 ms`. The event model predicted
`+2.319%`; its 1.138-point split radius reduced the lower bound to `+1.181%`,
so it was safely retained but not automatically accepted.

The explicitly forced high-skew greedy candidate with the best model lower
bound (`predicted=3.448%`, `lower=1.679%`) did not validate: its two session
medians were `+1.461%` and `+0.081%`, with negative P10 in both. This confirms
that selecting by the highest model lower bound alone is insufficient.

## False-pruning sentinel audit

No `candidate_worse` sentinel was a stable >2% improvement in both sessions.
Seven of eight were clearly or consistently slower. The closest boundary
sentinel was a fixed-width cross-lane swap predicted at `-10.946%` with upper
bound `-2.341%`; hardware medians were `-1.373%` and `+1.048%`, with intervals
crossing zero. This is not a false pruning event under the declared 2% stable
criterion, but it shows that the worse relation should remain a pruning guard,
not an accurate absolute slowdown estimator.

Across all 72 comparisons, the median absolute difference between session
medians was 0.318 percentage points; P90 was 1.265 points and the maximum was
4.052 points. Requiring the same candidate to pass in both sessions materially
reduced single-session false positives.

## Next search policy

The next offline loop should use a small hardware beam:

1. partial order removes only `candidate_worse` plans;
2. retain a canonical-deduplicated top-K incomparable frontier with operator
   diversity;
3. measure the frontier in two randomized sessions;
4. add independently stable >2% winners to the beam even when the comparator
   says incomparable;
5. expand both stable winners for the next depth while preserving all original
   anchors and the best measured incumbent;
6. move to template-level LNS only if the measured beam stops improving or
   cannot approach the historical `cp_sat_06` result under a fixed hardware
   budget.

## Artifacts

Raw artifacts are intentionally kept under ignored workspace storage:

| Artifact | SHA256 |
| --- | --- |
| self-contained VND | `7b7c353cee1e80ed76141f88f9c1faa5e3c95c8cddf6a012420ba55f896c924a` |
| frozen hardware frontier | `6b5300f3e08deef9bfc07c45ae256e186506d618d6d14a5027788fbac8a000b3` |
| hardware session 1 | `9190239530fa7458b4d8446bc87f406970044fb9ae1b1e6129f040b41283b71e` |
| hardware session 2 | `88e27995a3cb0dd10bebb2ff2774f252db54dffc34721e05514100941b3cac74` |

Paths are under `tmp/moe_partial_order_vnd_20260904/`. The cross-session
analysis is `high_skew_hardware_cross_session_analysis.json` and was generated
by `analyze_partial_order_hardware_frontier.py`.
