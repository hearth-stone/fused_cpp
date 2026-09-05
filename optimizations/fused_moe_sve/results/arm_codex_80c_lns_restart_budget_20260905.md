# Equal-budget 1-restart vs 2-restart template LNS

## Decision

Under a frozen selector v1, K=16 per unique parent, and the current operator
mixture, splitting the same 2,400 exact-eval cap and the same 85 hardware plans
across two restarts does not find a new median winner. Both arms select
`0418b884...` as the K=16 hardware best. Both beat strongest reconstructed full
by more than 2% in both sessions.

2-restart is the better *allocation* of that frozen budget on this seed:

- selected-best beats the known median elite `2ab43572...` in both of its
  sessions (`+0.797% / +0.449%`); 1-restart does not (`+0.765% / −0.290%`);
- the two K=16 sets overlap in only 15 of 64 plans, so the extra restart
  changes the shortlist rather than duplicating it;
- search wall is lower (558.71 s vs 683.67 s) because each start scores fewer
  uniques and diagnostic partial-order labeling is cheaper.

Do not read this as a new neighborhood winner, a selector change, or a
production result. The hardware best is the same plan the 1-restart independent
median already proposed. The previous 1-restart “missed 2% in session 2”
(`+2.579/+1.916%` on a 132-plan audit) is session-noise on the 2% gate, not
proof that one restart cannot propose `0418b884...`.

Keep selector v1 and K=16. For offline median LNS, prefer 2 restarts per parent
with `N=25` over 1 restart with `N=50` when the exact cap and hardware slots
are held fixed.

## Locked protocol

| Item | Value |
| --- | --- |
| Machine | Arm-codex-internal, NUMA3 CPUs 240-319, `--membind=3` |
| Trace | `measured_request016_case017_zh2048-018.pt`, layer 4 |
| Route SHA256 | `afabc7a1c9ffbabf4a844cf6842b1001d43a13df6fde0e37a5ee1eae58649431` |
| Shape | 2,048 tokens, TopK 6, 256 experts, H=4096, F=512, bf16, 80 threads |
| Selector | `relation_agnostic_categorical_farthest_first_v1`, K=16 / audit 32 **per unique parent after pooling restarts** |
| Operators | local/cross × d4/d8/d16, beams 16/32/64, 4 templates/block |
| Exact cap | `starts × 2 strategies × 6 operators × N = 2400` |
| Hardware | 4 reconstructed anchors + 1 known elite + 64 top-16 + 16 stratified outside top-32 = 85 |
| Elite | `2ab43572d14e11060e4fc07b3aa7e0c1780553233dfda61abe74f08abc51e973` from the earlier 2-restart median suite |
| Proposal seed | 20261010 |
| Stratified seed | 20261013 |
| Hardware | 1 process, 4 weight copies, 5 warmup, 31 rounds |
| Calibration / extension / pairwise / policy | `7928ba96...` / `dd554ea3...` / `a1b89cec...` / `6c07e7b9...` |

| Arm | Restarts | N | Starts | Exact cap | Event calls | Unique | Search s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-restart | 1 | 50 | 4 | 2400 | 2330 | 2322 | 683.67 |
| 2-restart | 2 | 25 | 8 | 2400 | 2384 | 2368 | 558.71 |

Event calls are not bit-identical: eight starts pay extra parent exacts
(`+54`, +2.3%). The *sampling* cap is 2400 on both arms. Hardware unique plans
are 85 on both. Audit-only 16 was not measured; those slots went to the
outside-top-32 sample. Top-16 vs measured top-32 recall is not reopened.

Command:

```text
numactl --physcpubind=240-319 --membind=3 \
  optimizations/fused_moe_sve/benchmarks/run_lns_restart_budget_compare.sh
```

Remote unit tests: 52 passed in 0.49 s. Local: 52 passed in 0.17 s.

## Hardware

Seeds 20261021/22 (1-restart) and 20261023/24 (2-restart). Strongest
reconstructed control is full `98a32da5...` in every session.

| Arm | Session | Full ms | Elite ms | Selected-best ms | vs full | vs elite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-restart | 1 | 32.12381 | 31.43719 | 31.19842 `0418b884...` | +2.966% | +0.765% |
| 1-restart | 2 | 32.53851 | 31.45197 | 31.54349 `0418b884...` | +3.154% | −0.290% |
| 2-restart | 1 | 32.24018 | 31.51659 | 31.26740 `0418b884...` | +3.111% | +0.797% |
| 2-restart | 2 | 32.32118 | 31.35389 | 31.21382 `0418b884...` | +3.548% | +0.449% |

| Gate | 1-restart | 2-restart |
| --- | --- | --- |
| Exact cap 2400 | yes | yes |
| Hardware 85 | yes | yes |
| Selected-best >2% vs full, both sessions | yes | yes |
| Selected-best beats elite, both sessions | no | yes |
| Elite in K=16 | no | no |
| Stratified-best beats selected | no / no | no / no |
| K=16 set overlap | 15 / 64 |  |

Hardware wall 109.69+110.05 s vs 100.89+101.92 s. Bit-exact plan outputs were
asserted before timing (`atol=0`). `sink` is a DCE checksum, not a mismatch.

Elite vs full stays above 2% in all four sessions (`+2.184/+3.455%` and
`+2.296/+3.085%`), so the control still represents a known-good offline plan.

The 16 stratified keys outside each parent's audit top-32 never beat the K=16
selected-best. That sample does not close top-32 vs all 2,322/2,368 generated
candidates.

## Search cost

1-restart matches the frozen independent-median pool (2,322 unique, 2,330
events). 2-restart enumerates more blocks (965 vs 479) and hashes more states
(105,188 vs 52,122) but samples half as many per start, so exact time stays
similar (352.04 vs 337.20 s) while diagnostic shortlist drops (76.93 vs
278.82 s). Net search wall 558.71 vs 683.67 s.

Peak RSS 323,576 KB (1-restart) and 340,972 KB (2-restart).

## Artifacts

Raw files remain under `tmp/moe_lns_restart_budget_20260905/` and are not
source-controlled.

| Artifact | SHA256 |
| --- | --- |
| Elite plan | `eacba259c93b5099e409ffefe10a4bcc5c620702be4386515c5edca1e95b98bd` |
| 1-restart model | `4be4947d58ad0be764069d8f3909659a021ebd441417a2d3cd2431425dfa1078` |
| 1-restart frontier | `d071bbbe7f6bbaa0c59ad868e47a80d22d72a3b8b70458087ec6f49cbfb0d8ce` |
| 2-restart model | `9b334b784e80c8376aa059084875a75d4f323c2a4c8f4192e60288f4590595b8` |
| 2-restart frontier | `18914c52d5c628ac44086db2653fc93628864193adea251f3a0078029cb424a7` |
| Analysis | `027ea3302286290a874f910d2498862037e8530a2baf8b4f911f0068e598a16f` |

## Next

- Use 2 restarts per parent and `N=25` for offline median LNS under this
  exact/hardware cap. Do not raise K, do not edit selector v1, do not add
  ALNS weights.
- A second proposal seed is still a neighborhood-quality question. This file
  is one seed and two allocations.
- Stratified 16 is not a proof that K=16/32 contains every fast plan in the
  generated pool. Keep a small outside sample on later frontiers.
- Production planner, Plan V2, kernel, ABI, frozen v8, and the local VND
  comparator remain unchanged.
