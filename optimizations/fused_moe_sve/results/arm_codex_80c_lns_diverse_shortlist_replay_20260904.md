# Template-LNS relation-agnostic shortlist design replay

## Decision

Pass the frozen measured-suite design replay at K=16 and freeze selector v1.
Do not adopt the selector into production, and do not open a new independent
frontier until the frozen policy is used unchanged.

The intended outcome is a deterministic offline hardware-budget policy, not a
more accurate absolute cost model. Partial-order acceptance and dominance
pruning remain disabled for template LNS.

## Selector

Policy `relation_agnostic_categorical_farthest_first_v1`:

- rule SHA256 `6c07e7b9fbf78412a103e3265d9dabbed224c1b9e3e248334e2ab1fb6ab08c97`
- frozen artifact `optimizations/fused_moe_sve/benchmarks/profiles/lns_diverse_shortlist_relation_agnostic_v1.json`
- artifact SHA256 `09bab219b13492e91cbbd2da488803720b2d20432725983be6ef7132a7d53034`
- source commit `055fe250a376622e7595d452d90e0947abbdd337` with a dirty worktree at freeze time

Features are plan-visible only: operator, target destroy size, actual lane-atomic
closure size, width histogram, LLC-domain assignment, and equal-frequency score
quantiles inside one `(anchor, restart)` pool. Relation labels, residual radii,
hardware timings, and previous winner status do not enter selection.

Each named start ranks all unique states, then takes a nested prefix: top-16 is
an exact prefix of top-32. After per-start ranking, canonical hashes are
deduplicated globally and backfilled. Unselected states are `budget_deferred`.
There is no `dominated` category.

## Design replay

This is a replay of the already-measured subset, not an independent validation.
The candidate universe is the former top-16 plus up to two model-worse spectrum
candidates per start.

Command:

```text
PYTHONPATH=.:src .venv/bin/python optimizations/fused_moe_sve/benchmarks/replay_lns_diverse_shortlist.py \
  --output tmp/moe_partial_order_vnd_20260904/lns_diverse_shortlist_replay.json
```

Replay JSON SHA256 `030a4767e02873b9800ce289a1cc0b722e343c713cdacec6d23214fa61afbab7`.
Suite analysis SHA256 `8e59f93276e31d183fea6db2741af0807cbe6662c77216ac2ecc7204ae8890bb`.

K=16 gate, both hardware sessions:

| Case | Absolute best retained | Consensus winner retained | Selected-best regret | Shuffle-stable |
| --- | --- | --- | ---: | --- |
| High-skew L1 | yes / yes | yes / yes | 0 / 0 | yes |
| High-skew L2 | yes / yes | yes / yes | 0 / 0 | yes |
| Median | yes / yes | yes / yes | 0 / 0 | yes |
| Uniformish | yes / yes | yes / yes | 0 / 0 | yes |

Median's absolute winner `2ab43572...` remains a `candidate_worse` record and is
still retained. That is the point of a relation-agnostic budget: the old
top-16-plus-sentinel policy recovered it by an explicit worse-spectrum sample;
the new selector recovers it without reading the relation.

No candidate is classified as dominance-pruned or automatically accepted.

## Budget sensitivity

K=8 and K=12 are reported only. They do not replace the predeclared K=16 default.

| Case | First failing budget | Symptom |
| --- | --- | --- |
| High-skew L1 | K=8 / K=12 | session-2 absolute best missed; K=8 also misses a parent best |
| High-skew L2 | K=8 | consensus winner missed; session-1 regret 0.826% |
| Median | none at K=8 for the absolute/consensus winner | parent-best recall still incomplete below K=16 |
| Uniformish | K=8 and K=12 | consensus winner missed; session-2 regret 0.200% |

K=24 and K=32 keep zero regret on all four cases and raise strict-stable recall
to 1.0. That is extra coverage, not a reason to raise the default budget.

## Correctness tests

```text
PYTHONPATH=.:src .venv/bin/python -m pytest -q \
  tests/test_moe_lns_diverse_shortlist.py \
  tests/test_moe_partial_order_vnd_runner.py \
  tests/test_moe_partial_order_hardware_frontier.py
```

27 tests passed, including deterministic shuffle, operator/quantile coverage,
top-16 prefix of top-32, relation-agnostic semantics, PlanV2 round-trip, and
nested frontier roles.

## Remaining gate

The independent median frontier is still required:

- trace `measured_request016_case017_zh2048-018.pt`, layer 4
- proposal seed `20261010`
- shortlist K=16 and audit K=32 per parent
- two NUMA3 sessions, seeds `20261011` and `20261012`
- measure the complete top-32 audit, then compare the nested top-16

Do not change the selector after opening that frontier. A K=16 failure with a
passing K=24 is a budget-quality report, not a silent default change. Production
planner, Plan V2, kernel, ABI, frozen v8, and local VND comparator are unchanged.
