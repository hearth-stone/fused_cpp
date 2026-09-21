# First-panel slowdown: address and gap controls, 2026-09-10

## Scope and predeclared method

Class E, Lab diagnostic. The user requested the two controls proposed after the six-family history experiment: fix tail A/C addresses, and insert a gap after the first M12 panel. Production JIT, planner/model and defaults are unchanged. The original history harness remains unchanged and its helper definitions are reused by the new optional translation unit. Rollback consists of the new diagnostic files and this manifest/report entry.

M1 W13 and W2 are tested in three conditions within the same process and allocations:

- No prefix, tail starts at row0 (original zero-history condition).
- No prefix, tail starts at row12 (fixed-address baseline).
- One M12 prefix at row0, tail starts at row12 (same tail A/C/routes as the fixed-address baseline).

Each condition uses gaps0/1/5/20/100/1000/5000us:42 cells. The gap begins after the prefix returns, or at the corresponding point without a prefix, and is excluded from the measured tail. Busy waiting on the pinned foreground core avoids an intentional sleep/wakeup transition. Actual gaps are recorded; this intervention does not explicitly flush caches, drain writes or isolate clock/prefetch state. The no-prefix comparison is repeated at every gap to detect effects of waiting itself.

The two address-fixed conditions also pack A2 identically for13 rows. The row0 baseline packs one row. Tail row0 and row12 have identical logical values because the row-dependent W2 input repeats every4 rows. Prefix and tail output ranges do not overlap. Every logical prefix/tail output and untouched output region is checked; W13 packed padding is excluded from logical-output checks.

As in the prior experiment: Arm-codex-internal, CPU316, NUMA3 memory, launch CPU allowance240–319, SVE256 BF16, H4096/F512, N tile16, one thread, no background workers, W13 degree5 and W2Direct degree0. Full W13 B8MiB and W2 B4MiB; full owner stripes `(1,0,0,1,1)`. Four B copies rotate by round. A256MiB scrub and5ms sleep precede each condition. Ordinary vector allocation, no explicit HugeTLB. Full W2 output allocation192MiB. This is a synthetic stage diagnostic, not end-to-end MoE.

A42-cell smoke precedes two sessions with5 warmup and31 measured randomized rounds each, seeds602000/602001/602002. Report per-session medians and paired-round bootstrap95% intervals for percentage changes (10,000 resamples, seed602100). Pairing follows round/B-copy identity; conditions run separately in randomized order. Intervals describe within-session sampling, not cross-machine uncertainty. No outlier trimming is applied.

## Results and interpretation

Both sessions completed: 42 smoke + 2 × 42 × 36 = 3,066 numerically checked conditions, including 2,604 formal measurements. The analyzer validated complete unique cell/round coverage. Both native session stderr files and runner stderr files are empty. Two protocol tests, Ruff, clang-format and diff whitespace checks pass. Source/JIT hashes match the snapshots; production JIT SHA256 remains `1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`. Repository HEAD is `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5`, with unrelated existing uncommitted work preserved in before-status.

### Address control, zero gap

| Session | Stage | No prefix, row0 (us) | No prefix, row12 (us) | Address change | One prefix, row12 (us) | Matched-address history change [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | W13 | 313.68 | 312.80 | -0.28% | 332.85 | +6.41% [+4.34, +10.76] |
| 1 | W2 | 187.41 | 187.07 | -0.18% | 205.70 | +9.96% [+5.76, +12.63] |
| 2 | W13 | 312.39 | 309.45 | -0.94% | 365.33 | +18.06% [+17.00, +19.04] |
| 2 | W2 | 185.55 | 187.75 | +1.19% | 233.86 | +24.56% [+23.29, +26.52] |

The cold address shifts are below1.2% in magnitude and all four intervals include zero. Fixed-address history changes are positive in both sessions with intervals excluding zero. Tail address position is therefore not a sufficient explanation of the first-panel slowdown in this protocol.

### Gap control

The percentage below compares one-prefix to no-prefix at the **same row12 addresses and same requested gap**. It does not charge the gap to the tail.

| Requested gap (us) | Session1 W13 | Session1 W2 | Session2 W13 | Session2 W2 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | +6.41% | +9.96% | +18.06% | +24.56% |
| 1 | +5.48% | +7.23% | +17.78% | +25.18% |
| 5 | +9.81% | +12.24% | +17.43% | +23.47% |
| 20 | +9.01% | +7.89% | +18.28% | +24.19% |
| 100 | +9.78% | +8.00% | +17.84% | +24.24% |
| 1000 | +9.70% | +8.25% | +18.09% | +23.50% |
| 5000 | +6.13% | +8.46% | +17.02% | +25.13% |

At5ms, fixed-address no-prefix/prefix times are313.98/333.23us and187.47/203.33us for session1 W13/W2; session2 gives311.09/364.04us and186.17/232.95us. All28 matched-address history intervals (2 sessions × 2 stages × 7 gaps) exclude zero on the positive side. No monotonic disappearance occurs as the gap increases. Actual median gaps range0.03–0.25us for requested zero,1.04–1.21us for1us, and5000.04–5000.22us for5ms; all measured gaps satisfy the requested lower bound.

This weakens the narrow explanation that a short-lived backlog will drain and restore cold-like timing after a few microseconds. It does not prove that every memory-system effect has been excluded: busy waiting preserves many cache/prefetch states, and is not an architectural drain operation. Persistent cache replacement/residency or access-state effects remain candidates, not identified causes.

### Repeatability and unresolved magnitude

The direction repeats, but the magnitude does not: zero-gap W13 changes+6.41% in session1 versus+18.06% in session2; W2 changes+9.96% versus+24.56%. Do not pool these into one stable slowdown coefficient or claim reproduction of the historical exact22% amplitude. Cold medians remain close across sessions while the post-prefix path shifts.

A descriptive split across all seven gaps finds all four B copies slower after a prefix in both sessions. Session2 post-prefix W13 per-copy medians range356.87–372.10us and W2 range231.94–234.13us; this is not a single-copy outlier. Session1 also has a temporal change: post-prefix W13 medians across rounds0–10/11–21/22–30 are328.32/328.33/346.66us, W2 are201.46/201.52/208.26us. Session2 corresponding medians are365.17/365.33/365.78us and233.46/233.46/232.90us. These are post-hoc descriptive splits, not independent causal tests. The maximum per-cell CV is7.12%; full p50/p90/p99, mean, standard deviation and actual-gap statistics are retained in `analysis.json`.

The two processes have independent allocations. Physical placement, cache mapping, host activity and other persistent state were not controlled or sampled with PMU; the session shift cannot be assigned to any one of them. Within-session bootstrap intervals do not cover these systematic differences.

Retain the diagnostic as a bounded reference for interpreting the six-family history curves. Neither address relocation nor a brief drain delay explains away the effect. Root-cause identification still needs counters or further controlled cache/access-state interventions; these two requested controls do not validate a physical cold/transition/warm model and do not alter the active planner baseline.

## Reproduction and evidence

Local artifacts: `tmp/kernel_history_controls_20260910/`. Remote artifacts: `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/kernel_history_controls_20260910/`. Source snapshots, before-status, build identity, binary and raw logs are retained there as applicable. The build uses the unchanged production JIT with GCC, C++17/O3/pthread and armv8.2-a+bf16+sve/SVE256 flags. `build_identity.txt` records source/binary/JIT hashes.

```bash
.venv/bin/pytest -q tests/test_moe_kernel_history_controls.py
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/kernel_history_controls_20260910/build_smoke.sh'
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/kernel_history_controls_20260910/run_sessions.sh'
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_kernel_history_controls.py \
  --sessions tmp/kernel_history_controls_20260910/session1.jsonl tmp/kernel_history_controls_20260910/session2.jsonl \
  --output tmp/kernel_history_controls_20260910/analysis.json
```

Use fresh output paths on rerun. The included historical main is renamed and never invoked; GCC warns about its implicit main return after renaming. The new executable's actual main retains normal C++ main return semantics. This warning does not describe the measured path. No production regression suite is required for a standalone diagnostic that does not alter production source.
