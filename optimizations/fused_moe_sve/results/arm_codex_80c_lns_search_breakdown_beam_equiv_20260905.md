# Template-LNS search-cost split and equivalent beam repair

## Decision

The 797.01 s independent-median search was not “all event simulation.” On the
frozen production path, after one equivalent `_beam_assign_tasks` rewrite, the
same seed spent 687.55 s as:

| Slice | Seconds | Share of 687.55 s |
| --- | ---: | ---: |
| Exact event scoring | 341.43 | 49.7% |
| Shortlist (diagnostic partial-order labeling + feature/quantile construction) | 278.77 | 40.5% |
| Neighborhood sample (beam 40.56, hash 12.42, assemble 1.71, template 0.59, closure 0.67) | 56.31 | 8.2% |
| Screen | 10.29 | 1.5% |
| Global diverse merge | 0.000053 | ~0% |

Beam repair was the enumeration hotspot on the isolated profiler and is now
cheap on the production path. Exact scoring and diagnostic shortlist were not
changed. Candidate hashes, sampling counts, model scores, quantiles, and the
ordered top-16 / top-32 are identical to the frozen independent-median model.

Do not start multi-restart from this file. Do not treat the 109.47 s wall
reduction as a reason to touch selector v1, K=16, the operator mixture, event
simulation, or cross-lane incremental replay.

## Locked inputs

Same frozen median holdout as
`arm_codex_80c_lns_diverse_independent_median_20260905.md`.

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
| Shortlist / audit | K=16 / K=32 |
| Calibration SHA256 | `7928ba9695b5c256ed86a4128cef851000590ccf9d3cad937a4bb52b6e76aad3` |
| Extension SHA256 | `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f` |
| Pairwise SHA256 | `a1b89cec87cc6c9cacab4477965bd060b8f41ba7eac227ef055552609268e54a` |
| Policy SHA256 | `6c07e7b9fbf78412a103e3265d9dabbed224c1b9e3e248334e2ab1fb6ab08c97` |
| Baseline model SHA256 | `9ed548a8017333331d85388acfb9b61287950842da32f99da543f369a61847c3` |
| Local HEAD | `e339cf5` plus uncommitted Lab instrumentation and equivalent beam repair |

No HugeTLB/page-policy override was passed.

## Isolated profiler (not the 797 s path)

`profile_lns_search_breakdown.py` loads one frozen parent, uses `model.T_iso`,
forces `window_selector=(0,0)`, and scores one critical strategy. It does not
run the production stage-window policy, the second (random) strategy, or the
diagnostic partial-order shortlist. Use it only to locate enumeration cost.

Command:

```text
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/profile_lns_search_breakdown.py \
  --model-artifact tmp/moe_lns_diverse_independent_median_20260904/median_lns_diverse_model.json \
  --analytic-calibration bench_assets/moe_paper/arm_codex_numa3_80c_temporal/analytic_machine_numa3_80c_narrow_merge_v8_20260903.json \
  --start lns_00_r00
```

Same command with `--start lns_01_r00`. After-opt dumps are
`profile_lns_00_r00_after_beam.json` and `profile_lns_01_r00_after_beam.json`.

| Start | Parent | Beam s before / after | Sample s before / after | Exact s before / after | Beam calls | Unique hashed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `lns_00_r00` | full `98a32da5...` | 7.045 / 2.245 (×3.14) | 8.631 / 3.843 | 35.731 / 35.725 | 204 | 3,967 |
| `lns_01_r00` | one-step `9447a717...` | 67.573 / 17.221 (×3.92) | 71.924 / 21.678 | 74.047 / 74.440 | 530 | 9,445 |

Proposed, unique, duplicate, sampled, exact-call, unique-block, and
`templates_selected` counts matched before vs after. Closure and template time
stayed below 0.4 s. This is why the equivalent rewrite targeted
`_beam_assign_tasks`: precompute `_retarget_task` per `(expert, width)`, count
empty lanes as integers, and maintain an incremental expert-id signature.
Last-write-wins on the signature dict and `placement_priority` are unchanged.
A local reference loop in
`tests/test_moe_executable_plan_neighborhood.py` checks repair signatures.

Do not read these profiler seconds as the 797 s production split. The
profiler unique-hashed 3,967 for one full-parent strategy is also not the
production unique-candidate 584: production samples 50 per operator across two
strategies and then scores that sample.

## Equivalent optimization

`_beam_assign_tasks` now:

- retargets each task once per legal width instead of once per beam node;
- uses `sum(not row)` empty-lane counts instead of rebuilding the assignment
  to test emptiness;
- updates the expert-id signature in place instead of walking every lane after
  each placement;
- assigns the changed lane with tuple slicing instead of copying the whole
  assignment into a list.

The four order policies, beam width formula
`ceil(repair_beam_width / 4)`, and sort key
`(max load, sum load², load range, signature)` are unchanged. Event
simulation, screen, and cross-lane incremental replay were not edited.

## Frozen production-path ranking replay

Command:

```text
numactl --physcpubind=240-319 --membind=3 \
  optimizations/fused_moe_sve/benchmarks/run_lns_beam_equiv_verify.sh
```

This is the independent-median model step (four reconstructed parents, one
restart, seed `20261010`) plus
`compare_lns_frozen_ranking.py` against
`tmp/moe_lns_diverse_independent_median_20260904/median_lns_diverse_model.json`.
It does not re-measure hardware.

Remote unit tests: 47 passed in 0.34 s before generation. Local unit tests:
47 passed in 0.24 s.

| Gate | Result |
| --- | --- |
| Unique candidates | 2,322 = 2,322 |
| Event calls | 2,330 = 2,330 |
| Better / worse / incomparable | 141 / 872 / 1,309 |
| Per-start unique and operator proposed/unique/sampled | identical |
| Candidate features (hash, predicted gain, quantile, closure, histogram, domain signature) | identical |
| Ordered per-start top-16 / top-32 / ranked_keys | identical |
| Ordered selected_frontier / audit_frontier scores | identical |
| Global selected_keys / audit_keys | identical |
| `compare_lns_frozen_ranking.py` | `equal=true`, `mismatch_count=0` |

Whole-file SHA256 changed (`9ed548a8...` → `e363b8c0...`) because the new
artifact adds `search_breakdown`, `shortlist_s`, `global_shortlist_s`, and
`ru_maxrss_kb`. Ranking fields are compared, not the file digest.

## Production-path time and memory

Search wall 797.01269518 s → 687.54505446 s (−109.47 s, −13.7%). Peak RSS after
the replay is 309,092 KB (~301.8 MiB). The 797 s run did not record RSS, so
this is not a before/after memory claim.

| Start | Parent | Old wall s | New wall s | Delta s | Unique | Exact s | Shortlist s | Sample s | Beam s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `lns_00_r00` | full `98a32da5...` | 163.156 | 152.587 | −10.57 | 584 | 71.09 | 71.45 | 7.85 | 4.44 |
| `lns_01_r00` | one-step `9447a717...` | 357.307 | 262.050 | −95.26 | 590 | 144.95 | 72.17 | 39.76 | 31.07 |
| `lns_02_r00` | greedy `de9efcef...` | 138.105 | 134.858 | −3.25 | 573 | 61.26 | 67.57 | 4.35 | 2.53 |
| `lns_03_r00` | fixed-width `97ec628e...` | 138.445 | 138.050 | −0.39 | 575 | 64.13 | 67.58 | 4.35 | 2.53 |
| Total |  | 797.013 | 687.545 | −109.47 | 2,322 | 341.43 | 278.77 | 56.31 | 40.56 |

Almost all of the wall reduction is the one-step parent, matching the
profiler. `shortlist_s` is ~68–72 s on every start. It bundles
`select_partial_order_shortlist` (still computed in diagnostic-only mode),
feature construction, quantile assignment, and per-start diverse ranking.
The 53 µs global merge is a separate operation and does not measure that
ranking work. Context evaluation is ~1.5 ms/start.

Clarification (2026-09-05, historical numbers unchanged): `shortlist_s` is
not ranking-only. It remains the per-start wall covering partial-order evidence,
feature construction, quantile/dedup, and per-start diverse ranking.
`search_wall_s` is still the sum of per-start runs and does not include
parent-pooled ranking, outside-audit sampling, canonical/PlanV2 frontier
construction, or the global unique-key merge (`global_shortlist_s`). Those
later stages are now timed separately in the layer bench; a complete
model-command elapsed time is recorded through the artifact write.

## Artifacts

Raw files remain under `tmp/moe_lns_beam_equiv_verify_20260905/` and the
profiler dumps under `tmp/moe_lns_diverse_independent_median_20260904/`. They
are not source-controlled.

| Artifact | SHA256 |
| --- | --- |
| Frozen baseline model | `9ed548a8017333331d85388acfb9b61287950842da32f99da543f369a61847c3` |
| Replay model | `e363b8c0f3b47991740fc8d996473d51aa00fbbfb52c1793498264c7101ff4db` |
| Ranking compare | `2479634ffcdfa0d8be10140b42a622441feebdc4635a16f1b278c46454af05a8` |

## Next

- Keep selector v1, K=16, and the current operator mixture frozen.
- The next search-quality experiment is equal-budget multi-restart versus this
  1-restart seed, with the four reconstructed controls plus the known median
  hardware elite `2ab43572...` as extra controls. Hardware plan slots must stay
  constant.
- Remaining search cost is exact event scoring and diagnostic partial-order
  shortlist, not beam. If either is optimized later, keep this ranked prefix as
  the invariance gate. Do not assume affected-lane event replay is equivalent.
- Top-16 versus measured top-32 recall is closed. Top-32 versus all 2,322
  generated candidates is still open; reserve a stratified sample outside
  top-32 on the next frontier.
- Production planner, Plan V2, kernel, ABI, frozen v8, and the local VND
  comparator remain unchanged.
