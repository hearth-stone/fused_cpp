# Arm-codex 80C topology-preserving width-neighborhood audit (prefinal)

## Status

This is a 2026-09-03 direct-sync run from an uncommitted working tree. It is
valid development evidence, but it is not a commit-bound paper artifact. Rerun
the declared suite after review and commit before citing it as a frozen result.

The experiment changes no production planner, Plan V2 schema, native runtime,
kernel, or default dispatch. It evaluates strict fixed whole-expert plans only;
tail pool, repartition, resize, stealing, and arbitrary interval lowering are
disabled.

## Question and gate

The audit asks whether a one-move, directly executable width neighborhood adds
useful local plans beyond the previously rejected order-only neighborhood. It
adds three domain-local operators:

- split one lane into two equal calibrated widths and reassign only that lane's
  experts by isolated-time LPT;
- merge two physically adjacent equal-width lanes within one LLC domain;
- migrate one expert between existing lanes whose calibrated widths are
  adjacent, preserving the relative order of the other tasks.

Every changed-width task refreshes its deterministic W13/W2 tile windows. An
existing cross-domain lane is preserved but cannot be a width-move endpoint.
All candidates lower directly from canonical executable state to strict Plan
V2 and must be bitwise identical to the baseline output.

The predeclared adoption gate is: at least one width candidate with positive
paired P10 speedup; no selected-plan regression above 2%; measured shortlist
regret at most 5%; and explicit order-only, width-only, and combined candidate-
union ablations. Automatic replacement still requires at least 2% robust
predicted gain.

## Configuration

- Machine: `Arm-codex-internal`, NUMA3 CPUs `240-319`, memory node 3.
- Workload: DeepSeek-V4 TP4-oriented BF16 fused expert, `H=4096`, `F=512`,
  256 experts, 2048 tokens, TopK6.
- Calibration: `analytic_machine_numa3_80c_temporal_overhead_v4_20260903.json`.
- Traces: captured uniformish layer 20, median-skew layer 4, and high-skew
  layer 38.
- Search: 32 event-critical and 32 seeded-random experts, 64 sampled candidates
  per operator, canonical global deduplication, complete placed event scoring.
- Hardware: 5 warmups, 31 randomized paired rounds, four rotating packed-weight
  copies, production SVE exact-M/direct-route path.
- Working tree: uncommitted direct-sync after commit `5f6e07c`; raw JSON is in
  ignored `tmp/moe_width_neighborhood_prefinal_20260903/`.

The pre-benchmark focused target validation passed `21 passed, 113 deselected`;
after adding the explicit cross-domain negative case, the final synced tree
passed `22 passed, 113 deselected`. Every measured candidate also passed the
benchmark's exact-output comparison before timing.

## Results

| Trace | Baseline shape | Baseline median | Best predicted order | Best predicted width | Stable shortlist | Selected | Selected regret |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| uniformish | `10x8T` | 32.478 ms | +0.666% | +0.189% | 0/12 | baseline | 0.260% combined; 0% width-only |
| median | `1x16T + 8x8T` | 32.218 ms | +0.872% | +1.476% | 0/16 | baseline | 0.526% combined; 0.419% width-only |
| high-skew | `4x16T + 16x1T` | 32.816 ms | +0.153% | +0.110% | 2/13 | baseline | 1.449% combined/width-only |

No analytical candidate reached the 2% robust action margin, so the guarded
decision retained the baseline on all three traces and had no selected-plan
regression. Combined event-to-hardware Spearman was `0.552`, `-0.018`, and
`-0.011` for uniformish, median, and high-skew respectively; sub-2% point
ranking therefore remains unreliable.

The high-skew trace nevertheless contains two stable width improvements. Both
are domain-local lane merges:

| Operator | Predicted gain | Paired median | Paired P10 | Wins |
| --- | ---: | ---: | ---: | ---: |
| domain-local lane merge | +0.0197% | +1.436% | +0.262% | 30/31 |
| domain-local lane merge | +0.0130% | +1.248% | +0.340% | 29/31 |

In contrast, the model-ranked high-skew width winner was a lane split predicted
at `+0.110%`; it measured `-0.446%` median with `-2.245%` paired P10. The stable
merges were in the event shortlist only because the audit retained width-only
ablation leaders, not because the point model ranked them accurately.

Candidate generation and scoring remained tractable for an offline audit:

- uniformish: 633 total sampled event evaluations across the two selection
  arms, about 8.5 plans/s; cold full planning was 125.6 s;
- median: 759 evaluations, about 10.0 plans/s; cold full planning was 118.1 s;
- high-skew: 645 evaluations, about 6.3 plans/s; cold full planning was 79.4 s.

## Decision

The width-neighborhood structure passes the minimal usefulness test: unlike the
order-only audit, it exposes reproducible hardware improvements on high-skew.
Therefore keep split/merge/migration as Lab search operators.

Do **not** start deterministic width-VND yet. The best stable gains are below
the 2% action threshold, combined point ranking is effectively random on
median/high-skew, and the high-skew model ranks a regressing split ahead of the
stable merges. The next technical gate is to model or independently calibrate
the `1T+1T -> 2T` lane-merge/concurrency transition, then repeat the same
three-trace audit. Only after the robust scorer can accept the stable merge
without increasing selected regret should these operators drive VND or LNS.
