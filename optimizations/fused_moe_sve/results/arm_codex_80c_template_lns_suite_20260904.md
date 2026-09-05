# Arm-codex 80C template-LNS cross-trace suite

## Technical summary

Adopt the fixed template-LNS neighborhood for offline hardware-assisted search,
but disable partial-order acceptance and dominance pruning for LNS candidates.
The same frozen local/cross-domain and d4/d8/d16 operator mixture found a
cross-session consensus winner more than 2% faster than the strongest preserved
anchor on high-skew, median, and uniformish. The comparator did not generalize:
median's absolute best was classified confidently worse, and one uniformish
model-better candidate failed the strict P10 repeatability gate.

High-skew was expanded once more from its two sub-0.01% consensus elites. The
second layer produced one strict-stable candidate but no global consensus winner
that justified another automatic expansion, so high-skew depth 3 is closed.

## The fixed LNS neighborhood improved all three traces

The comparison baseline is the fastest preserved control or prior hardware
elite in each session. Gain is `100 * (anchor_time / candidate_time - 1)`, so a
positive value means the LNS candidate is faster.

| Trace | Consensus winner | Median ms, S1/S2 | Gain vs strongest anchor, S1/S2 | Stable unique candidates |
| --- | --- | ---: | ---: | ---: |
| High-skew, layer 1 | `189d70b0...` | 31.249 / 31.297 | +3.961% / +3.428% | 40 |
| Median | `2ab43572...` | 31.381 / 31.393 | +2.671% / +2.863% | 40 |
| Uniformish | `50a7d4be...` | 31.517 / 31.698 | +2.057% / +3.250% | 30 |

The winner-over-strongest-anchor gate passes on all three traces. This supports
the search-space claim: coordinated width, domain, assignment, and order repair
finds useful plans that local VND/beam cannot reach reliably. It does not prove
global optimality or production planning viability.

## High-skew depth 2 is useful but does not justify depth 3

The second layer expanded `189d70b0...` and `a7b38202...`, the first-layer
session elites separated by only 0.0076%. Four restarts generated 2,343 unique
candidates using 2,351 event calls and 858.70 seconds of search wall time. The
frozen hardware frontier contained 72 plans.

Only one candidate was strict-stable in both sessions: a domain-local d8 repair
relative to `189d70b0...`, with paired median `+2.404/+2.070%` and P10
`+1.061/+0.139%`. The normalized absolute winner `37bac29b...` measured
`30.406/30.557 ms`, but its second-session paired median versus `189d70b0...`
was only `+1.873%` and its P10 was negative. The two session-best hashes differed
by 0.0468% at the second-session optimum. Preserve the depth-2 elites, but stop
automatic high-skew expansion here.

## Median disproves LNS dominance pruning

Median reconstructed four controls whose canonical hashes exactly match the
previous VND artifact. Eight restarts generated 4,643 unique candidates using
4,659 event calls and 1,005.67 seconds of search wall time. The hardware
frontier contained 145 plans: 4 anchors, 26 model-better candidates, 99
incomparable candidates, and 16 model-worse sentinels.

All 26 measured model-better records were strict-stable in both sessions, but
two model-worse sentinels were also stable improvements. This fails the
one-sided pruning gate.

The strongest counterexample is the two-session absolute winner `2ab43572...`:

| Quantity | Model / session 1 / session 2 |
| --- | --- |
| Predicted gain | -3.770% |
| Residual interval | [-5.540%, -2.001%], classified `candidate_worse` |
| Paired hardware median | +2.290% / +3.102% |
| Paired hardware P10 | +1.034% / +1.011% |
| Absolute median | 31.381 ms / 31.393 ms |

The second counterexample `c8487e13...` was predicted at -3.842% but measured
`+3.870/+3.934%`, with positive P10 in both sessions. Spearman over the measured
incomparable subset was `-0.047/-0.030`, so the failure is not a small radius
miss: point ordering itself carries essentially no useful signal in this LNS
domain.

## Uniformish keeps LNS value but also blocks automatic acceptance

Uniformish deduplicated full, one-step, and fixed-width into one control and
kept greedy as the second control. Four restarts generated 2,337 unique
candidates using 2,345 event calls and 430.68 seconds of search wall time. The
74-plan frontier contained 26 model-better, 38 incomparable, and 8 model-worse
records plus two anchors.

Twenty-five of 26 model-better records were strict-stable in both sessions. The
remaining candidate was predicted at +4.855% with a lower bound of +3.086%; its
hardware median was still positive at `+3.054/+3.556%`, but at least one
session's paired P10 did not stay above zero. Under the predeclared strict gate,
that is a false automatic acceptance.

The consensus winner `50a7d4be...` measured `31.517/31.698 ms` and improved the
strongest full/fixed control by `+2.057/+3.250%`. The two session-best hashes
differed, but the consensus winner's session-1 regret was only 0.117%. Spearman
over the measured incomparable subset was `0.489/0.449`: better than median,
but not sufficient for an unconditional model-only decision.

## Search and measurement contract

All first-layer traces used the same frozen structure:

- target destroy sizes 4/8/16;
- placement repair beams 16/32/64;
- domain-local and cross-domain scopes;
- four near/far width templates per lane-atomic closure;
- 32 event-critical experts plus an equal-size random control;
- 50 candidates per scope/size operator;
- two proposal restarts per unique parent;
- frozen v8 analytical calibration and frozen global residual radius;
- self-contained canonical states and PlanV2 bridges before hardware replay.

Each hardware session used Arm-codex-internal NUMA3 CPUs 240-319, one process,
one batch of packed weight allocations, four-copy rotation, five warmups, 31
randomized paired rounds, and bit-exact output checks before timing. Median and
uniformish controls were reconstructed inside the target process from their
frozen route traces; their hashes match the earlier VND identities.

Across high-skew layers 1/2, median, and uniformish, the suite consumed 12,867
event calls, 3,488.51 seconds of search wall time, 397 frozen frontier plans,
and eight independent hardware sessions. Control-construction time is separate:
median and uniformish analytical full construction took 136.72 and 144.97
seconds respectively.

## LNS partial order is now diagnostic only

The local-move residual radius is not transferable to large lane closures. The
LNS runner now enforces:

```text
automatic acceptance = disabled
dominance pruning     = disabled
unmeasured candidates = budget-deferred, never proven worse
```

Relations remain in artifacts for analysis. Hardware frontiers distinguish
`model_better_frontier`, `incomparable_frontier`, and
`model_worse_spectrum`; score-spectrum candidates remain eligible to become the
hardware winner. Local VND retains its previously validated comparator behavior.
This change does not alter the frozen v8 mean, production planner, PlanV2,
kernel, ABI, or runtime dispatch.

## Limitations and robustness checks

- The suite proves improvement only within the measured frontier, not global
  optimality over every LNS candidate.
- The current top-16 policy alone would miss median's absolute best; it was
  recovered by the explicit model-worse spectrum sample. A relation-agnostic
  diversity selector still needs an independent measured-best recall gate.
- LNS planning remains expensive and Python-bound. The fixed mixture is an
  offline method, not a production planning path.
- No explicit page-policy override was passed by the commands; inherited shell
  page-policy variables were not captured.
- The second high-skew layer improves absolute medians but does not pass the
  strict global expansion rule in both sessions.

## Recommended next step

Build a relation-agnostic hardware shortlist that reserves capacity across
operator, target destroy size, actual closure size, width histogram, domain
assignment, and model-score quantiles. Replay it first on the measured suite and
then on an independent frontier. Only after measured-best recall closes should
enumeration caching, incremental event replay, or ALNS operator weighting be
considered.

## Reproducibility

- Repository base: commit `055fe250a376622e7595d452d90e0947abbdd337`
  plus the uncommitted Lab changes described in this report.
- Machine: Arm-codex-internal, NUMA3 CPUs 240-319, `--membind=3`.
- Shape: 2,048 tokens, TopK 6, 256 experts, H=4096, F=512, bf16,
  80 threads, backend N tile 16.
- Calibration SHA256:
  `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
- Extension SHA256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- Runtime controls: `FUSED_CPP_MOE_SVE=1`,
  `FUSED_CPP_MOE_SVE_W2_DIRECT_ROUTE=1`,
  `FUSED_CPP_MOE_W2_BF16_ROUTE=0`, and
  `FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE=1`.

| Trace | Route file and layer | Route SHA256 | Session seeds | Timed wall, S1/S2 |
| --- | --- | --- | --- | ---: |
| High-skew layer 1 | `measured_request008_case009_zh2048-010.pt`, L38 | `d6a04171b059381ac383f5ff61ce7b24e9d84a465dc58886a29d8c55d6eb6724` | 20260921 / 20260922 | 116.20 / 116.08 s |
| High-skew layer 2 | same high-skew trace | same | 20260925 / 20260926 | 79.51 / 79.43 s |
| Median | `measured_request016_case017_zh2048-018.pt`, L4 | `afabc7a1c9ffbabf4a844cf6842b1001d43a13df6fde0e37a5ee1eae58649431` | 20260928 / 20260929 | 182.39 / 181.88 s |
| Uniformish | `measured_request022_case023_zh2048-024.pt`, L20 | `58dd037b7f1c372affbd9984e6cfe93fe10d85f76098d36d7ff28bb957080622` | 20261001 / 20261002 | 85.13 / 85.69 s |

Full W13 and W2 stage sizes are 8 MiB and 4 MiB. Every reported winner uses
`window_tiles=0`, so all stages execute one full owner stripe with `R13=R2=1`.

| Team width | W13 owner tiles/thread | W2 owner tiles/thread | `(w13_window,w2_window,R13,R2)` |
| ---: | ---: | ---: | --- |
| 1T | 64 | 256 | `(0,0,1,1)` |
| 2T | 32 | 128 | `(0,0,1,1)` |
| 4T | 16 | 64 | `(0,0,1,1)` |
| 8T | 8 | 32 | `(0,0,1,1)` |
| 16T | 4 | 16 | `(0,0,1,1)` |

## Artifact identity

The suite-level decision artifact is
`tmp/moe_partial_order_vnd_20260904/template_lns_suite_analysis.json`, SHA256
`8e59f93276e31d183fea6db2741af0807cbe6662c77216ac2ecc7204ae8890bb`.

| Analysis | SHA256 |
| --- | --- |
| High-skew layer 1 | `a2ad7b24dc1445c8e5e3af8fab26706ba14980b0a6f6ee985cdf9170d02d3ad1` |
| High-skew layer 2 | `aaefffcd9f6197092e5eb5d4aad63871665682273222b17c60abc398b4859718` |
| Median | `79f5058a9e612e9e51702f5c471b6de2c11395e1a213df41a9b91414af3cd03f` |
| Uniformish | `088d296484034772c4a970cc821cd3fb9eb00274227df7e5d908b1c96275af7c` |

Raw model, frontier, session, and analysis JSON files remain under
`tmp/moe_partial_order_vnd_20260904/` and are not source-controlled.
