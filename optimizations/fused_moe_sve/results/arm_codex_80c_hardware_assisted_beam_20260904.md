# Arm-codex 80C hardware-assisted beam search

## Final decision

Stop local hardware-assisted beam search at the predeclared maximum depth 3 and
move the next experiment to template-level LNS. The corrected beam preserved
and expanded the global measured incumbent, found one additional globally
useful depth-2 split, but depth 3 produced no candidate that was independently
stable above 2% relative to its parent and no repeated absolute winner.

Partial order remains useful only for one-sided `candidate_worse` pruning and
diverse top-K retention. Do not lower the 2% gate and do not refit residual
radii from any beam measurement.

## Correction to the first beam selection

The first hardware frontier identified two candidates that were stable above
2% relative to their own fixed-width and greedy anchors. A subsequent absolute
comparison showed that both anchors were materially slower than the full-start
frontier. The first-layer absolute best in both sessions was instead state
`4115ab99...`, a domain-local lane merge from the full start. Its paired median
gain over full was `+2.437/+2.422%`; session-2 P10 was `-0.117%`, so it did not
meet the strict cross-session stable predicate.

The initial depth-2 implementation carried this state for comparison but did
not expand it. That run is excluded from the final beam decision. The analyzer
and parent selection were corrected to compute absolute consensus elites across
sessions, and depth 2 was rerun from three parents: the global measured best and
the two independently stable relative winners.

## Corrected depth 2

- Parents: 3.
- Model neighborhood: 2,064 unique candidates, 2,070 event calls,
  `0 better / 238 worse / 1,826 incomparable`, 250.79 s.
- Hardware frontier: 61 plans = 3 parents + 48 incomparable candidates +
  6 worse sentinels + 4 original anchors.
- Hardware sessions: 31 randomized rounds each; timed walls 67.39/67.82 s.
- Three candidates were stable above 2% relative to their own parent in both
  sessions; no worse sentinel was a false prune.

State `514ced0d...`, a domain-local lane split from the previous global best,
was the absolute fastest plan in both sessions: `31.808/31.721 ms`. The event
model predicted only `+0.264%`, with partial-order lower bound `-0.874%`.
Relative to its immediate parent it measured `+0.745/+1.055%` with negative
P10, but relative to the original full anchor it measured:

| Session | Median | P10 | P90 |
| --- | ---: | ---: | ---: |
| 1 | +2.146% | +0.146% | +4.041% |
| 2 | +3.228% | +1.532% | +4.770% |

Thus it is a real global elite even though the incremental second step is below
the 2% automatic-acceptance threshold. This establishes that beam retention
must use absolute measured elites separately from per-edge acceptance.

The two-session consensus absolute top-2 were `514ced0d...` (lane split) and
`d2b93803...` (same-lane insertion). They became the depth-3 parents.

## Depth 3

- Parents: 2 consensus absolute elites.
- Model neighborhood: 1,461 unique candidates, 1,465 event calls,
  `0 better / 199 worse / 1,262 incomparable`, 256.27 s.
- Hardware frontier: 45 plans = 2 parents + 32 incomparable candidates +
  4 worse sentinels + 7 carried anchors/elites.
- Hardware sessions: 31 randomized rounds each; timed walls 47.76/47.58 s.
- Stable >2% relative candidates: `0 / 0`.
- Cross-session false-pruning sentinels: 0.

The consensus absolute candidate `0ffab6b9...` was the fastest in session 1
and sixth in session 2. Relative to its parent, the same-lane adjacent swap
measured `+1.411%` and `+0.921%`; P10 was `-0.760/-0.585%`. This is not a
stable next beam step.

Session 2's absolute fastest plan was a `candidate_worse` insertion sentinel,
but its paired result was `-0.774/+0.039%` across sessions, with both intervals
crossing zero. It is not a false prune under the declared stable 2% criterion;
it demonstrates that sub-percent absolute ranks are not reproducible enough to
drive another local depth.

## LNS handoff

The next search must make coordinated changes larger than one local move while
preserving the measured elite set:

1. keep original full, `4115ab99...`, and `514ced0d...` as immutable incumbents;
2. destroy two or more lanes or one complete LLC-domain template at a time;
3. repair width histogram, domain placement, and within-lane order jointly;
4. include CP-SAT-like heterogeneous width templates so the search can approach
   the historical `cp_sat_06` structure;
5. use partial order only to reject confidently worse candidates;
6. hardware-rerank a canonical-deduplicated diverse frontier under the same
   two-session protocol;
7. compare LNS and the completed beam under equal event-call and hardware-call
   budgets.

The old `cp_sat_06` executable task sequence was not preserved, so it cannot be
inserted directly as a current hardware control. Its width histogram remains a
template target, not a same-session performance claim.

## Formal artifacts

All raw JSON files are under `tmp/moe_partial_order_vnd_20260904/`.

| Artifact | SHA256 |
| --- | --- |
| corrected depth-2 model | `82bc9d05316f2a1aeff3a3f54dba6dfc05c72fc70a2c62a39831162e955ef0b9` |
| corrected depth-2 frontier | `075d2da5cc790601d484aa3d9bcf82491772ca27bd225b0bfc2e54964eb9855a` |
| corrected depth-2 session 1 | `31ef81401c9162c3877dbef6f5896a6c1bd46309f26b14955436c12d64bb658e` |
| corrected depth-2 session 2 | `99da3459d4f8b9a3a1c245f9bc4d080a379021068fcdacf4b4bd87e2c12703b4` |
| corrected depth-2 analysis | `a8bd96c1e250681a0e4116c1494fe998d8ebaa5ce44a1d977fe25f8589142e0d` |
| depth-3 model | `9e286986f05a8a32cbb7be839246f37b4bb4a2f49b6b945640e0155baf28d721` |
| depth-3 frontier | `af37586fe474d9e124abbc587e7330e4a02e1c8b78c99a7c28be936b34f869f0` |
| depth-3 session 1 | `17439798348ad342e5d83bf51bcc93aedba7dcf39004f3f1ac3864ef64212425` |
| depth-3 session 2 | `9d4ed56907429439d03c78b7f2846d3b699814dec7b9acfea90dcde4cf281985` |
| depth-3 analysis | `480a389ca0f937eb75ae464e665c0c3cd1c1aff92d2437ee01d60c21e99fcbdd` |
