# Independent median LNS diverse-shortlist frontier

## Decision

Selector v1 passes the independent nested-recall gate. This proposal seed does
not pass the two-session 2% strongest-anchor neighborhood gate.

Keep `relation_agnostic_categorical_farthest_first_v1` as the offline LNS
hardware shortlist. Do not change the selector, do not raise the default K=16,
and do not treat this seed as a production or global-optimality result.

The two conclusions are separate:

- Shortlist recall is closed on this holdout: top-16 contains the measured
  top-32 absolute best in both sessions, contains the consensus winner, and has
  zero selected-best regret. Even the nested K=8 prefix already has zero regret.
- The 1-restart independent neighborhood did not produce a candidate that is
  more than 2% faster than the strongest full control in both sessions. That is
  a neighborhood/proposal failure for seed `20261010`, not a selector miss.

## Locked inputs

| Item | Value |
| --- | --- |
| Machine | Arm-codex-internal, NUMA3 CPUs 240-319, `--membind=3` |
| Trace | `measured_request016_case017_zh2048-018.pt`, layer 4 |
| Route SHA256 | `afabc7a1c9ffbabf4a844cf6842b1001d43a13df6fde0e37a5ee1eae58649431` |
| Shape | 2,048 tokens, TopK 6, 256 experts, H=4096, F=512, bf16, 80 threads |
| Parents | reconstructed full, one-step, greedy, fixed-width |
| Restarts per parent | 1 |
| Proposal seed | 20261010 |
| Destroy / beams / templates | 4/8/16, 16/32/64, 4 |
| Shortlist / audit | K=16 / K=32 per parent |
| Hardware | 1 process, 4 weight copies, 5 warmup, 31 rounds, seeds 20261011 / 20261012 |
| Calibration SHA256 | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| Extension SHA256 | `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f` |
| Pairwise SHA256 | `a1b89cec87cc6c9cacab4477965bd060b8f41ba7eac227ef055552609268e54a` |
| Policy SHA256 | `6c07e7b9fbf78412a103e3265d9dabbed224c1b9e3e248334e2ab1fb6ab08c97` |

Parent canonical hashes match the previous median VND/LNS controls:

| Control | State hash |
| --- | --- |
| full | `98a32da5...` |
| one-step | `9447a717...` |
| greedy | `de9efcef...` |
| fixed-width | `97ec628e...` |

No HugeTLB/page-policy override was passed. Inherited page-policy variables
were empty in the runner environment.

## Search and measurement

Remote unit tests: 27 passed in 0.41 s before generation.

| Stage | Result |
| --- | --- |
| Model | 4 starts, 2,322 unique candidates, 2,330 event calls, 797.01 s |
| Relations | 141 better / 872 worse / 1,309 incomparable; diagnostic only |
| Frontier | 4 anchors + 64 top-16 + 64 audit-only = 132 plans |
| Session 1 | 167.96 s, 128 comparisons, 38 paired-stable >2%, bit-exact `sink=0` |
| Session 2 | 168.49 s, 128 comparisons, 33 paired-stable >2%, bit-exact `sink=0` |

Command:

```text
numactl --physcpubind=240-319 --membind=3 \
  optimizations/fused_moe_sve/benchmarks/run_lns_diverse_independent_median.sh
```

## Selector recall

All predeclared recall gates pass. Session identities match on frontier,
extension, calibration, pairwise, and policy hashes. Automatic acceptance and
dominance pruning remain disabled.

| Gate | Result |
| --- | --- |
| Top-16 contains session-1 absolute best `0418b884...` | yes |
| Top-16 contains session-2 absolute best `7cac2afd...` | yes |
| Top-16 contains consensus winner `7cac2afd...` | yes |
| Selected-best regret | 0 / 0 |
| Anchors retained and executable | yes |

Budget sensitivity, selected-best regret versus the measured top-32 oracle:

| K | Unique selected | S1/S2 absolute-best retained | Regret |
| ---: | ---: | --- | ---: |
| 8 | 32 | yes / yes | 0 / 0 |
| 12 | 48 | yes / yes | 0 / 0 |
| 16 | 64 | yes / yes | 0 / 0 |
| 24 | 96 | yes / yes | 0 / 0 |
| 32 | 128 | yes / yes | 0 / 0 |

Do not lower the default to K=8 after seeing this table. K=16 was frozen before
the holdout.

Both session elites are `lns_diverse_top16` states from the full parent
`98a32da5...`, cross-domain, and incomparable under the frozen residual radius:

| Plan | Operator | Actual closure | Predicted gain | S1 median ms | S2 median ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `0418b884...` | cross-domain d4 | 99 | -2.303% | 31.4728 | (not session-2 best) |
| `7cac2afd...` | cross-domain d8 | 48 | -0.087% | 31.5397 | 31.71942 |

The selector kept model-incomparable / model-worse-spectrum structure without
reading the relation. Session-1's fastest plan was predicted -2.303%.

## Neighborhood gate

Strongest control is full `98a32da5...` at `32.28449 / 32.32727 ms`.

Absolute median gain of the measured-best plan versus that control is
`+2.579% / +1.916%`. Session 2 is below 2%, so the predeclared two-session
strongest-anchor gate fails.

Paired versus the same full parent:

| Plan | S1 paired / P10 / stable | S2 paired / P10 / stable |
| --- | --- | --- |
| `0418b884...` | +2.740% / +0.996% / yes | +1.641% / -0.357% / no |
| `7cac2afd...` | +2.199% / +0.638% / yes | +1.744% / +0.177% / no |

Session 1 has four plans more than 2% faster than full by absolute median.
Session 2 has zero. Parent-relative paired-stable counts (38 / 33) are mostly
versus the slower one-step/greedy/fixed-width controls, not versus full.

The earlier median LNS suite used two restarts per parent and found a consensus
winner at `31.381 / 31.393 ms` (`+2.671 / +2.863%`). This independent seed used
one restart, as locked in the shortlist handoff, and did not recover that
improvement. That does not reopen selector v1.

## Artifacts

Raw files remain under `tmp/moe_lns_diverse_independent_median_20260904/` and
are not source-controlled.

| Artifact | SHA256 |
| --- | --- |
| Model | `9ed548a8017333331d85388acfb9b61287950842da32f99da543f369a61847c3` |
| Frontier | `b57e0774681721a70805a008adf149f6be38fc8ab34cbd2db7cc9cd27dc569b9` |
| Session 1 | `2bd3d0537a87c8aefcd5c0c86e1b84afefe6ee8e2742adf67c1a16250ecc71d5` |
| Session 2 | `7aa6c44b77416e9673e050ffc9a33dca87db9ab4f58c002e727fcbac01b2450a` |
| Analysis | `e02849aad016117e5f9ebf291943718e2a04498a9d87782c3ccd09b33347e4c9` |

## Next

- Adopt selector v1 for offline template-LNS hardware shortlists.
- Search-cost split and equivalent beam repair preserved this ranked prefix;
  remaining cost is exact event scoring and diagnostic shortlist. See
  `arm_codex_80c_lns_search_breakdown_beam_equiv_20260905.md`.
- Optimize exact scoring or diagnostic shortlist only while preserving
  this ranked prefix on a replay of the frozen model artifact.
- A second independent seed, or a return to two restarts per parent, is a
  neighborhood-quality question. It is not a reason to edit selector v1.
- Do not add ALNS weights until a later holdout shows complementary operator
  value after this shortlist is in place.
- Production planner, Plan V2, kernel, ABI, frozen v8, and local VND
  comparator remain unchanged.
