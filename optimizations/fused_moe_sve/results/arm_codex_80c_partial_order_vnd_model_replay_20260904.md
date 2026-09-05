# Arm-codex 80C partial-order VND model-only replay

## Decision

The guarded anchor-relative partial order is safe enough to retain a top-16
offline measurement frontier, but it cannot drive deterministic VND on the
current three-trace corpus. Across all three traces and four declared starts,
no neighbor's calibrated lower gain bound cleared the 2% actionable margin.
Every start stopped after its first neighborhood with
`no_confident_improvement`.

Do not weaken the margin or refit the residual radius from these search
outputs. Keep the partial order for dominance pruning and top-K retention. The
next search experiment should be template-level LNS or another non-monotone
global proposal mechanism whose candidates are sent to the same top-16
measurement frontier.

## Provenance and method

- Source: commit `055fe25` plus the uncommitted multi-start VND runner and
  operator-context resolution fix.
- Machine: `Arm-codex-internal`, NUMA3 CPUs `240-319`, memory node 3.
- Extension SHA256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`.
- Frozen-v8 calibration SHA256:
  `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3`.
- Gated pairwise report SHA256:
  `a1b89cec87cc6c9cacab4477965bd060b8f41ba7eac227ef055552609268e54a`.
- Shape: BF16 fused expert, `H=4096`, `F=512`, 256 experts, 2048 tokens,
  TopK6, 80 threads.
- Neighborhood: combined order and topology-preserving width moves; 32
  critical and 32 deterministic-random experts; 64 sampled candidates per
  operator and strategy.
- Search: independent `full`, `one_step`, `greedy`, and best homogeneous
  `fixed_width` starts; at most 20 iterations; top-16; 2% actionable and
  dominance margin.
- This is model-only. No packed weights were allocated and no hardware kernel
  timing was collected.

The fit and evaluation artifacts behind the pairwise report are disjoint and
have matching calibration/extension identities. The report gate has zero
false pruning and retains the measured best on all three validation traces at
top-16. Because that report contains operator-level context radii rather than
placed-event context keys, the runner uses exact operator keys and family
fallback. It does not run an event explanation that cannot match calibration.

## Results

| Trace | Starts | Unique neighbors | Better / worse / incomparable | Accepted | Event calls | Search wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high-skew | 4 | 2,326 | `0 / 241 / 2,085` | 0 | 2,334 | 349.71 s |
| median | 4 | 2,566 | `0 / 240 / 2,326` | 0 | 2,574 | 375.03 s |
| uniformish | 4 names, 2 unique states | 2,434 | `0 / 521 / 1,913` | 0 | 2,442 | 342.84 s |
| Total | 12 names | 7,326 | `0 / 1,002 / 6,324` | 0 | 7,350 | 1,067.58 s |

All 12 runs executed exactly one iteration. Each seed required one anchor
placed-event explanation; there were no candidate placed-context event calls.
The remaining wall time is therefore dominated by complete candidate event
scoring. The previously rejected cheap screen was not used.

Uniformish `full`, `one_step`, and `fixed_width` resolve to the same canonical
state hash; `greedy` is distinct. They were still run under all four declared
names so the baseline exposes this redundancy instead of silently changing
the requested start set.

### Closest candidates to acceptance

The acceptance rule is `lower_gain_pct > 2.0`.

| Trace / start | Predicted gain | Lower bound | Gap to acceptance | Scope |
| --- | ---: | ---: | ---: | --- |
| high-skew / greedy | 3.448% | 1.679% | 0.321 points | context |
| median / greedy | 2.608% | 1.470% | 0.530 points | context |
| median / one-step | 1.617% | 0.850% | 1.150 points | context |
| uniformish / greedy | 1.271% | 0.133% | 1.867 points | context |

These are not accepted improvements and have not been measured. In particular,
the favorable point estimates do not justify lowering the 2% margin.

### Aggregate operator accounting

| Operator | Raw proposals | Worse | Incomparable | Better / accepted |
| --- | ---: | ---: | ---: | ---: |
| same-lane adjacent swap | 1,060 | 0 | 955 | 0 / 0 |
| same-lane insertion | 21,058 | 5 | 1,510 | 0 / 0 |
| same-width cross-lane relocation | 67,997 | 316 | 1,216 | 0 / 0 |
| same-width cross-lane swap | 128,663 | 177 | 1,357 | 0 / 0 |
| same-width cross-LLC relocation | 76,589 | 393 | 1,045 | 0 / 0 |
| domain-local lane split | 217 | 32 | 86 | 0 / 0 |
| domain-local lane merge | 205 | 42 | 64 | 0 / 0 |
| adjacent-width migration | 1,192 | 37 | 91 | 0 / 0 |

Raw proposal counts precede canonical deduplication and per-operator sampling;
relation counts refer to the 7,326 unique scored states. The partial order has
useful one-sided pruning power, especially for cross-LLC relocation, but no
positive decision coverage.

## `cp_sat_06` diagnostic

The historical high-skew `cp_sat_06` plan measured 33.148 ms versus the 34.484
ms full incumbent, while its point event prediction had the wrong direction.
The pairwise replay correctly keeps it incomparable rather than pruning it.

The old artifact retained only the CP plan summary, not its complete executable
task sequence, so exact canonical-hash reachability cannot be reconstructed.
The saved width histogram is nevertheless structurally far from the full
start: `cp_sat_06` assigns 204/6/4/7/10 experts to 1/2/4/8/16 threads, whereas
the full start assigns 168 experts to 1T and 63 to 16T. At least 53 expert-width
assignments differ even under maximum histogram overlap. More importantly,
all four high-skew starts have zero acceptable first move. Therefore monotone
partial-order VND cannot reach `cp_sat_06` from any tested start, regardless of
whether the unrestricted neighborhood graph is eventually connected.

## Raw artifacts

The full per-start/per-iteration JSON remains in ignored workspace storage:

| Trace | Path | SHA256 |
| --- | --- | --- |
| high-skew | `tmp/moe_partial_order_vnd_20260904/high_skew_final.json` | `7ff4959c7f916ea3814732422bf1927fcb14a04ab2868e5a1ab502214802b8ec` |
| median | `tmp/moe_partial_order_vnd_20260904/median_final.json` | `8765716da3f4a3cba95325c171c010d36d814b5ab4d46bc717a8a773e7b6b829` |
| uniformish | `tmp/moe_partial_order_vnd_20260904/uniformish_final.json` | `7241b7d65acd9eedac05c0f748aaab92d07fa2d06c497ba1aabac8b6ba842fd2` |

Each JSON records anchor and final hashes, shapes, per-operator proposals,
critical/random expert ids, complete partial-order evidence, selected top-16,
dominance-pruned and budget-deferred hashes, accepted move, event-call counts,
context and total wall time, best-so-far curve, and stop reason.
