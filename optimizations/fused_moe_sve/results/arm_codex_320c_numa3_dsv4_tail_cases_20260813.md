# DSV4 tail candidates on Arm-codex NUMA3

Date: 2026-08-13

## Scope

This run checks the retained DSV4 tail candidates after synchronizing the
current worktree to `Arm-codex-internal:/home/zhangxu/code`.

Common configuration:

- machine: `Arm-codex-internal`, 320 Arm cores, four NUMA nodes;
- placement: NUMA3, CPUs `240-319` (80 cores), local memory binding;
- cache topology reported by `lscpu`: 64 KiB L1D and 1.25 MiB L2 per core,
  with 560 MiB L3 across eight instances;
- runtime SVE vector length: 256 bits;
- build: MoE-only, `FUSED_CPP_SVE_VECTOR_BITS=256`, effective `-O2`;
- workload: TP4 `H=4096`, `F=512`, 256 experts, 2048 tokens, TopK=6;
- route distribution: `dsv4-real-2048-seq70`, 223 active experts, including
  75 with `M<=12` (`71xM1`, `3xM6`, `1xM10`);
- explicit plan: experts with `M>48` use 8T teams; smaller experts use 1T;
- timing: seven warmups and 51 alternating, same-process paired runs unless
  stated otherwise;
- correctness: paired variants produced bit-exact output;
- memory: ordinary pages. This host has `HugePages_Total=0`, so the absolute
  times are not directly comparable with the old Amazon 96-core 32 MiB
  HugeTLB measurements.

The available profile was calibrated on the old Amazon 96-core host. A local
80-core compatibility copy was used only to instantiate explicit plans. Its
absolute predictions are not treated as Arm-codex model results.

## Core partition

Initial probes showed that an even `40/40` split leaves the small-expert side
idle early, while `64/16` starves it badly:

| Large/small cores | Median | Internal idle | Tail idle |
| --- | ---: | ---: | ---: |
| 40/40 | 37.748 ms | 33.45 core-ms | 745.60 core-ms |
| 48/32 | 31.888 ms | 34.90 core-ms | 306.57 core-ms |
| 56/24 | 32.017 ms | 26.80 core-ms | 296.21 core-ms |
| 64/16 | 45.075 ms | 30.63 core-ms | 1394.86 core-ms |

The close `48/32` and `56/24` cases were then measured in one process for 51
pairs. `56/24` measured `30.9205 ms` versus `31.4049 ms` for `48/32`, a
`+1.566%` ratio-of-medians gain. The aligned-pair speedup median/P10/P90 was
`+1.561%/+0.668%/+2.521%`; all 51 pairs favored `56/24`. Therefore the
candidate comparisons below use `56/24`.

After removing all temporary experiment controls and rebuilding the default
binary, the same `56/24` plan measured `30.7051 ms` over 51 runs.

## Candidate results

| Candidate | Baseline | Candidate | Relative result | Decision |
| --- | ---: | ---: | ---: | --- |
| Strict whole-task tail steal | 30.9340 ms | 30.4821 ms | +1.482% | Positive, below the 2% adoption gate |
| Static residual-M, E208 to donor core 16 | 30.9582 ms | 30.9235 ms | +0.112% | Neutral |
| Old-profile planned suffix DAG | 30.8585 ms | 30.9247 ms | -0.214% | Reject; profile is not portable |
| M<=12 pool order, increasing M to LPT | 29.0266 ms | 28.7506 ms | +0.960% | Keep existing LPT order |
| Static strict to dynamic M<=12 1T LPT pool | 30.6971 ms | 28.7955 ms | +6.604% | Strong positive candidate |

Aligned-pair distributions reinforce the decisions:

| Candidate speedup | Pair median | P10 | P90 |
| --- | ---: | ---: | ---: |
| Strict whole-task tail steal | +1.479% | +0.359% | +2.623% |
| Static residual-M | +0.081% | -1.162% | +1.312% |
| Old-profile suffix DAG | -0.192% | -1.081% | +0.779% |
| LPT versus increasing-M order | +0.962% | +0.391% | +1.394% |
| Dynamic M<=12 pool | +6.465% | +5.529% | +7.280% |

The static residual-M result was not donor-sensitive in a useful direction:
donor cores 24 and 40 measured `-0.263%` and `-0.118%` in shorter paired
runs. The favorable E218 geometry from the old Amazon host therefore did not
transfer to this machine.

The old-profile suffix model predicted `+2.671%` for E208/core32 with an E215
donor on core16, but execution regressed by `0.214%`. This is diagnostic
evidence that the old contention profile cannot select Arm-codex suffixes; it
is not an accuracy measurement for a calibrated Arm-codex model.

## Conclusion

Tail idle is not uniformly intractable on this host. Dynamic width regrouping
and residual-M slicing remain unattractive, but moving the 75 `M<=12` whole
expert tasks into the existing 1T LPT tail pool gives a repeatable `6.60%`
end-to-end gain. The mechanism is work conservation across the 24 small-task
workers, without changing each task's kernel width or numerical result.

The retired nonblocking `1T -> 2T/4T` regroup implementation was deliberately
not restored merely for this benchmark. Its old Amazon result was `-0.613%`
and all 75 tasks still executed at 1T, so the current tree has no equivalent
execution path to measure faithfully. This should not be conflated with the
positive whole-task 1T pool result above.

Before enabling the pool through the production planner, refresh the Arm-codex
isolation/contention calibration and retest the full workload catalog. The
explicit result establishes the opportunity but does not validate the old
profile's automatic choice.

## Verification

The final remote source contains no temporary short-pool ordering switch. A
clean SVE256 rebuild passed:

- `57 passed, 22 deselected` for SVE backend/packed/fused tests;
- `6 passed, 9 deselected` for strict-tail and residual-M plan tests.

Raw JSON captures are stored under
`tmp/moe_timeline/dsv4-real-2048-seq70/arm_codex_numa3_80c/`.
