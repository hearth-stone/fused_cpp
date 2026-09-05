# Second proposal seed under frozen 2-restart median LNS

## Decision

A second independent proposal seed under the frozen 2-restart / `N=25` /
selector v1 / K=16 protocol does **not** recover a two-session 2% winner.

Seed `20261011` used the same exact cap 2,400 and the same LNS hardware slots
as seed `20261010` (64 top-16 + 16 stratified). It generated 2,362 unique
candidates in 550.90 s and never sampled the previous selected-best
`0418b884...`. The new K=16 set overlaps that seed in only 3 of 64 plans.
Hardware selected-bests differ across sessions (`2fd99a11...` /
`8deba718...`) and stay within noise of reconstructed full
(`−0.018% / +0.371%`).

The injected previous-seed control still works: `0418b884...` is session-1
measured best (`31.122 ms`, `+3.302%` vs full) and stays within `0.030 ms` of
elite `2ab43572...` in session 2. Elite itself remains above 2% vs full in
both sessions (`+3.004% / +3.227%`). This is a neighborhood/proposal miss,
not a selector or allocation regression, and not a reason to edit selector v1
or to abandon 2-restart `N=25` on seed `20261010`.

Keep 2 restarts plus `N=25` as the offline median *allocation*. Do not treat
seed `20261011` as a replacement proposal. Seed success under this protocol
is 1/2 so far; that is not a seed-robustness claim.

## Locked protocol

Same frozen median holdout as
`arm_codex_80c_lns_restart_budget_20260905.md`, except the proposal seed and
hardware session seeds. LNS-generated hardware slots stay 64 + 16; the extra
unique plan is the previous-seed selected-best injected as
`previous_seed_selected`.

| Item | Value |
| --- | --- |
| Machine | Arm-codex-internal, NUMA3 CPUs 240-319, `--membind=3` |
| Trace | `measured_request016_case017_zh2048-018.pt`, layer 4 |
| Route SHA256 | `afabc7a1c9ffbabf4a844cf6842b1001d43a13df6fde0e37a5ee1eae58649431` |
| Shape | 2,048 tokens, TopK 6, 256 experts, H=4096, F=512, bf16, 80 threads |
| Selector | `relation_agnostic_categorical_farthest_first_v1`, K=16 / audit 32 per unique parent after pooling |
| Operators | local/cross × d4/d8/d16, beams 16/32/64, 4 templates/block |
| Exact cap | `starts × 2 strategies × 6 operators × N = 2400` |
| Restarts / N / starts | 2 / 25 / 8 |
| Hardware | 4 reconstructed anchors + elite `2ab43572...` + previous selected `0418b884...` + 64 top-16 + 16 stratified = 86 unique |
| Proposal seed | 20261011 (previous successful seed 20261010) |
| Stratified seed | 20261013 (unchanged) |
| Hardware sessions | 20261025 / 20261026, 1 process, 4 weight copies, 5 warmup, 31 rounds |
| Calibration / extension / pairwise / policy | `7928ba96...` / `dd554ea3...` / `a1b89cec...` / `6c07e7b9...` |
| Local HEAD | `db95b7e` plus uncommitted second-seed runner, `--reference-plan`, and analyzer |

No HugeTLB/page-policy override was passed. `page_policy_env` was empty.

Command:

```text
numactl --physcpubind=240-319 --membind=3 \
  optimizations/fused_moe_sve/benchmarks/run_lns_second_proposal_seed.sh
```

Remote unit tests: 54 passed in 0.44 s. Local: 54 passed in 0.23 s.

## Search

| Seed | Restarts | N | Event calls | Unique | Search s | Peak RSS KB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261010 | 2 | 25 | 2384 | 2368 | 558.71 | 340,972 |
| 20261011 | 2 | 25 | 2378 | 2362 | 550.90 | 339,116 |

Seed `20261011` split: exact 349.02 s, diagnostic shortlist 76.24 s, sample
113.36 s (beam 81.68 s). Global farthest-first merge 67 µs. Relation counts
143 / 853 / 1,366.

`0418b884...` is absent from every start's `candidate_features` and from every
parent-pooled ranked list. The miss is in sampling, not in selector ranking of
a generated candidate.

K=16 overlap with seed `20261010` is 3/64:

- `01307409e836330cb230c6c8b6d62a7f75efd5eeae3e1eff8ea954f695a963d5`
- `029dc4739d2ae174bedf46fc36356c2ae6dc62195c440733938de576b114b0f5`
- `1dec39c8b1e640ff2f29eb62e72d41429fd7435916874886867705b0716c304c`

None of those overlap keys is the hardware selected-best of either seed.

## Hardware

Strongest reconstructed control is full `98a32da5...` in both sessions. Elite
vs full stays above 2%. Audit-only 16 was not measured; those slots went to
the outside-top-32 sample, as in the restart-budget protocol.

| Plan | Session 1 ms | Session 2 ms | vs full s1/s2 |
| --- | ---: | ---: | --- |
| Full `98a32da5...` | 32.14985 | 32.35023 | — |
| Elite `2ab43572...` | 31.21236 | 31.33889 | +3.004% / +3.227% |
| Previous selected `0418b884...` | 31.12198 | 31.36859 | +3.302% / +3.130% |
| New selected `2fd99a11...` | 32.15563 | 32.24930 | −0.018% / +0.313% |
| New selected `8deba718...` | 32.28289 | 32.23054 | −0.412% / +0.371% |

Analyzer selected-best is the fastest K=16 plan in that session:
`2fd99a11...` then `8deba718...`.

| Gate | Seed 20261010 2-restart | Seed 20261011 |
| --- | --- | --- |
| Exact cap 2400 | yes | yes |
| LNS hardware slots 64+16 | yes | yes |
| Protocol `protocol_ok` | — | yes |
| Selected-best >2% vs full, both sessions | yes | **no** |
| Selected-best beats elite, both sessions | yes | **no** |
| Selected-best beats `0418b884...`, both sessions | — | **no** |
| `0418b884...` generated | yes | **no** |
| `0418b884...` in K=16 | yes | **no** |
| Stratified-best beats selected | no / no | no / no |
| K=16 set overlap vs 20261010 | — | 3 / 64 |

Hardware wall 109.26 + 107.92 s. Bit-exact plan outputs were asserted before
timing (`atol=0`). `sink` is a DCE checksum and was 0 in both sessions.

The 16 stratified keys outside each parent's audit top-32 never beat the K=16
selected-best. That sample still does not close top-32 versus all 2,362
generated candidates.

## Artifacts

Raw files remain under `tmp/moe_lns_second_seed_20260905/` and are not
source-controlled. Previous-arm identity matches
`arm_codex_80c_lns_restart_budget_20260905.md`.

| Artifact | SHA256 |
| --- | --- |
| Elite plan | `eacba259c93b5099e409ffefe10a4bcc5c620702be4386515c5edca1e95b98bd` |
| Previous selected plan | `24aecb34c95c257a5e1056756bfb52b0df6cb8ef4c736692ca86afc576a064c1` |
| Second-seed model | `cc53d43d5827a353ba4325c8a787e0302fcff20a0ebc3840752acfa824d3238a` |
| Second-seed frontier | `fa4c83b0843f58283e42ea0a957d34684d2e56a29c8471ef02de2550ae365895` |
| Session 1 | `45890ef21b8960df51acfdd55770ff4b2124d3e6a6c73d4d5e802134e6d9bf5b` |
| Session 2 | `5859c20996c7b0109cc46b42da969954642aca7515ee7cabc967b64d779d685f` |
| Analysis | `71a23bc5baf0db71671bfff1fc5005fcd02ea79b7d38e160ce36b69b5f6bafee` |
| Previous 2-restart model | `9b334b784e80c8376aa059084875a75d4f323c2a4c8f4192e60288f4590595b8` |
| Previous 2-restart frontier | `18914c52d5c628ac44086db2653fc93628864193adea251f3a0078029cb424a7` |

## Next

- Keep selector v1, K=16, and 2-restart `N=25`. Do not raise K, do not add
  ALNS weights, and do not replace the seed-`20261010` selected-best with
  this seed's shortlist.
- A third independent seed would estimate success rate; two seeds are not a
  rate. If that experiment runs, keep the exact cap and LNS hardware slots
  fixed and keep injecting `0418b884...` plus elite `2ab43572...` as controls.
- Top-16 versus measured top-32 recall remains closed from the earlier
  independent-median audit. Top-32 versus all generated candidates remains
  open; this seed's stratified 16 did not beat K=16.
- Production planner, Plan V2, kernel, ABI, frozen v8, and the local VND
  comparator remain unchanged.
