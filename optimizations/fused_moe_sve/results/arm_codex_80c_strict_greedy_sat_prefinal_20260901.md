# Arm 80C strict greedy versus greedy-inclusive SAT prefinal

## Decision

The fixed-plan first stage passes its declared gate. The exact pure-greedy
strict plan is retained as one branch of the surrogate search domain, and the
domain-aware SAT branch is selected on all three traces with a 2.071--2.129%
union gap. On hardware the measured SAT fixed plan wins all 31 paired runs on
every trace.

This result deliberately excludes dynamic tail pooling, tail repartition,
stealing, resize, and active release-time gates. It establishes only the
strict fixed-plan comparison. Dynamic tail execution is a separate second
stage and is not covered by the reported optimality gap.

The measurements are prefinal direct-sync runs from an uncommitted tree on
`Arm-codex-internal`, not commit-bound runner artifacts. The declared rerun
suite is `paper_experiments/suites/arm_strict_greedy_sat.json`.

## Search domain

Let `G` be the original `plan_quick()` homogeneous-LPT plan, including its
widths, `core_begin` values, lane order, and dependencies. Let `F_D` be the
domain-aware SAT strict-plan family. The first-stage domain is

```text
F_strict = {G} union F_D.
```

The two branches are disjoint and do not mix tasks. They are solved separately
with the same cold-phase surrogate and combined exactly:

```text
U_union = min(U_greedy, U_sat)
L_union = min(L_greedy, L_sat).
```

The one-step strict projection is used only as a domain-SAT search hint. It is
not the mechanism by which greedy is retained and does not affect the union
guarantee.

## Method

- Arm-codex NUMA3, CPUs 240--319, `membind=3`, two 40-core LLC domains.
- BF16 SVE fused expert, H=4096, F=512, E=256, 2048-token TopK6 traces.
- Calibration SHA256:
  `e0ec1cd4ede5dfbdb1ef1748807357292cbad2b1786431642a9934908e42884b`.
- Extension SHA256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- One 60-second domain proof and one fixed-greedy CP branch, requested relative
  gap 1%, eight solver workers. The measured SAT plan is the proof incumbent;
  no second SAT root or shortlist is used.
- One SAT plan and one greedy plan measured through the same strict executor.
- Five warmups, 31 randomized paired runs, four rotating packed-weight copies.
- Deterministic stage-window policy; no dynamic execution features.

## Results

| Trace | Greedy shape | Greedy/SAT surrogate UB (ms) | Union gap | Greedy measured (ms) | SAT measured (ms) | SAT paired median | P10 / P90 | SAT wins | T_plan (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| High skew, layer 38 | 231 x 16T | 27.520 / 23.656 | 2.122% | 36.809 | 35.963 | -2.213% | -3.135 / -1.665% | 31 / 31 | 128.223 |
| Median skew, layer 4 | 224 x 16T | 25.445 / 23.126 | 2.071% | 38.097 | 33.314 | -12.487% | -13.372 / -11.726% | 31 / 31 | 133.352 |
| Uniformish, layer 20 | 234 x 8T | 24.914 / 23.105 | 2.129% | 36.240 | 33.494 | -7.274% | -8.787 / -4.793% | 31 / 31 | 180.955 |

`T_plan + T_execute` is 128.259 s, 133.385 s, and 180.989 s respectively.
All three union solves select the domain-SAT branch. Thus the search is
provably no worse than pure greedy under the cold surrogate, and the selected
SAT plan is also stably faster than pure greedy in these hardware runs.

## Scope and remaining distinction

The current one-step full strict control is more sophisticated than pure
homogeneous-LPT greedy. Relative to that control, SAT is 1.89% faster on high
skew, 3.90% faster on median skew, and 3.89% slower on uniformish. Therefore
this experiment supports the narrow claim that SAT improves the original pure
greedy fixed planner; it does not yet establish dominance over every existing
full-planner heuristic.

The next stage should keep this strict result frozen and add dynamic tail pool
as a separate executor dimension in a 2x2 comparison:

| Base plan | Executor |
| --- | --- |
| Greedy | Strict |
| SAT | Strict |
| Greedy | Dynamic tail pool |
| SAT | Dynamic tail pool |

Only the first two cells are covered here.
