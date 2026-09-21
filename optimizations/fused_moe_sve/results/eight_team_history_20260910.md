# Actual cooperative8T owner-stripe history measurements, 2026-09-10

## Geometry and execution contract

The user requested collection using the actual8T stripes, rather than rescaled1T data. This class E standalone Lab target invokes the unchanged production JIT with eight workers on one expert and one B copy. Target CPUs312–319 share a full output tensor and partition N without overlap:

| Stage | Global GEMM N | Per-worker GEMM N | Per-worker B bytes | Per-worker N tiles | Output columns per worker |
| --- | ---: | ---: | ---: | ---: | ---: |
| W13 |1024|128|1,048,576 (1MiB)|8|64 after fused gate/up|
| W2 |4096|512|524,288 (512KiB)|32|512|

H4096/F512, BF16 SVE256, N tile16, full owner stripes `(8,0,0,1,1)`. Each worker receives its rebased packed-B stripe and output pointer, local N count, and n_begin0; the global output leading dimension is retained. Every condition records the8 N-begin/end ranges and stripe bytes. It is not eight separate full1T experts. Logical outputs and untouched regions are checked across the complete shared output, including every W2 column/route and W13 logical packed output; padding rows are not logical results.

## Grid and timing

Exact tail rows1/2/4/8/12 × W13/W2 × preceding full M12 panels0/1/2/3/7 × backgrounds none/four8T M2 teams/four8T M120 teams =150 conditions. Two independent5-warm/31-measured randomized sessions after150-cell smoke; seeds607000/607001/607002. Four B copies rotate by round.

The target tail A/C/routes are fixed at row96 across histories. Prefix panels advance through rows0–83 and repeatedly access the same target B copy. This deliberately separates address movement from history; it is not a contiguous full-M runtime benchmark. There is **no target barrier between panels**. Every worker records start/end of each prefix and tail call; timing includes the small call-parameter wrapper. Report both:

- service: maximum of the8 individual tail durations;
- envelope: latest tail end minus earliest tail start, also affected by arrival skew accumulated during prefixes.

Neither is a sum of8 individual durations. Prefix timings and the full start-to-last-tail interval are retained too.

CoordinatorCPU240, target312–319, background teams280–287/288–295/296–303/304–311, all NUMA3 memory. Each background team cooperatively executes the same W13 expert/B copy, with a barrier between full expert calls and a shared four-copy rotation. No barrier is inserted between its M12 panels. The32 competing cores plus8 target cores occupy one40-core LLC domain; this is not the earlier38 independent1T competitors and not the full80-core multi-domain runtime.

Before release, every target worker scrubs its own32MiB region (256MiB collectively), then the coordinator waits for target readiness and at least one completed expert call per background team and applies5ms lead-in. Backgrounds continue throughout target prefixes and tail. Workers are pinned, and all background CPU/count/output/coverage checks run per condition. All conditions allocate ordinary vectors without explicit HugeTLB. A2 uses exact per-panel packing and row-varying values; W13 has constant1/64 input/weights and verifies BF16 SiLU(1), W2 verifies exact FP32 row-dependent values. Full W2 output is192MiB.

A read of CPU312 sysfs confirms private64KiB L1D, private1280KiB L2, and71680KiB L3 shared by280–319. Capacity information does not establish actual B residency. No PMU, cache-hit-rate or physical temperature identification is made.

## Completed validation

All150 smoke +2×150×36 =10,950 target conditions passed numerical, geometry, timing-order and background coverage checks;9,300 are formal measured conditions. These counts are conditions, not individual JIT invocations (each contains8 workers and possibly prefixes/background loops). Both runtime sessions, build and driver stderr files are empty. Two local geometry/order tests pass; Ruff, clang-format and diff checks pass. Native source SHA256 `5f7737612dcecbbbbc506a5aaf9607371aa0d40e89287c3c87f1e794b564085b`; binary SHA256 `f9ac215e0d96c4d0c52b557ae62c4ec2d9930484fe75060cfcaf8741b7c921d0`; unchanged JIT SHA256 `1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.

The largest per-cell service CV is19.15%; no samples are trimmed. Full means/std, p90/p99, envelopes, skew and prefix metrics are retained. History-change95% intervals use10,000 paired-round bootstrap resamples, seed607100. They describe conditional session variability, not universal timing constants.

## No-background results

Values are medians of maximum worker tail service in us, shown session1 / session2. Histories label actual preceding full scans, not measured cache temperatures.

### W13

| Tail rows | h0 | h1 | h2 | h3 | h7 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 111.24 / 109.58 | 107.17 / 107.16 | 114.58 / 111.85 | 111.59 / 111.88 | 113.09 / 109.78 |
| 2 | 110.61 / 107.13 | 106.50 / 104.93 | 112.64 / 110.85 | 113.74 / 107.26 | 111.54 / 108.35 |
| 4 | 112.12 / 108.28 | 114.19 / 115.83 | 119.37 / 116.98 | 117.51 / 112.79 | 116.78 / 113.13 |
| 8 | 121.58 / 119.65 | 134.51 / 135.59 | 124.88 / 122.49 | 126.28 / 121.49 | 121.12 / 118.67 |
| 12 | 164.44 / 162.53 | 163.34 / 160.49 | 158.57 / 157.01 | 158.60 / 157.06 | 159.03 / 155.95 |

### W2

| Tail rows | h0 | h1 | h2 | h3 | h7 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 49.35 / 50.06 | 50.46 / 50.63 | 47.98 / 48.38 | 46.49 / 47.19 | 41.48 / 42.08 |
| 2 | 52.92 / 53.26 | 51.60 / 44.84 | 44.71 / 49.42 | 49.73 / 45.03 | 41.97 / 39.16 |
| 4 | 53.48 / 54.25 | 55.71 / 54.16 | 51.26 / 52.76 | 47.60 / 50.61 | 40.71 / 43.79 |
| 8 | 59.40 / 60.15 | 64.17 / 60.95 | 55.40 / 55.10 | 53.11 / 53.75 | 52.98 / 53.33 |
| 12 | 82.80 / 83.11 | 80.57 / 80.07 | 79.01 / 79.16 | 78.17 / 78.22 | 77.29 / 77.48 |

## Interpretation and competition

W2 row1 is49.35/50.06us at h0,50.46/50.63us at h1 and41.48/42.08us at h7. The h1 change is small with both95% intervals including zero; the h7 decrease is about15.94% in both sessions, with intervals entirely negative. This does not reproduce the large1T row1 first-prefix bump as a universal cross-width law.

Nonmonotonicity does occur for other shapes: W13 row8 h1 is10.63%/13.32% slower than h0, with positive95% intervals[7.82,14.94]% and[9.83,16.58]%. Full12 service varies much less after the initial scans. Exact row/family and stage remain necessary.

At h7 without backgrounds, W2 row1 is about54% of full12 service (41–42 versus77us), not the rounded-compute-row fraction2/12. W13 row1 is about70–71% of full12 service (110–113 versus156–159us). Thus transient history is only part of the missing small-tail cost; late-reference small-tail service also cannot be estimated by scaling full12 throughput. This does not uniquely attribute the difference to compute, load service, pipeline or cache mechanism.

Competition examples for W2 row1 (session1 / session2, us):

| Background | h0 | h1 | h7 |
| --- | ---: | ---: | ---: |
| None | 49.35 / 50.06 | 50.46 / 50.63 | 41.48 / 42.08 |
| 4×8T M2 | 59.38 / 56.39 | 54.35 / 52.62 | 37.26 / 37.40 |
| 4×8T M120 | 52.20 / 62.85 | 53.91 / 56.13 | 38.63 / 38.74 |

The cold-reference competition cost and subsequent trajectory depend on workload. In some later-history conditions the measured service with background is lower than no-background; do not clamp this away or infer that competition universally accelerates the kernel. This benchmark changes the continuously active background/cache/request state and does not identify the physical reason. Session variation, particularly cold small-row W2 under M120 background, remains visible in the table.

Timing skew matters but does not explain the whole small-tail cost. For no-background h7 row1, W2 service/envelope is41.48/43.30us and42.08/43.83us; W13 is113.09/122.86us and109.78/118.42us. A large residual exists even in maximum individual worker duration, before considering inter-worker arrival skew.

## Decision and limits

Retain this actual8T geometry dataset as the basis for later full-panel/exact-tail/history calibration. It supports width-specific history response and a non-linear small-tail cost; it does not validate transplanting1T curves, a zero B-refill assumption from capacity alone, or any new planner coefficient. No model is fitted or planner changed in this turn.

These are synthetic separately prepared W13 or W2 stages. W2 does not inherit the actual immediately preceding W13/gather state; routes are fixed rather than real-expert sparse routes; only one LLC domain and two pure background-team types are tested. Allocation/alignment/page policy and inter-stage scheduling are not claimed identical to the production runtime. Therefore geometry is matched, but exact real-plan absolute times and physical cache state are not established.

## Reproduction and retention

Local `tmp/eight_team_history_20260910/`; remote same relative path under `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/`. Source snapshots, build scripts/identity, raw JSONL, logs and analysis are retained there. HEAD `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus preserved prior workspace changes. Optional build is C++17/O3/pthread, armv8.2-a+bf16+sve and SVE256; original global build and native sources are untouched.

```bash
.venv/bin/pytest -q tests/test_moe_eight_team_history.py
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/eight_team_history_20260910/build_smoke.sh'
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/eight_team_history_20260910/run_sessions.sh'
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_eight_team_history.py   --sessions tmp/eight_team_history_20260910/session1.jsonl tmp/eight_team_history_20260910/session2.jsonl   --output tmp/eight_team_history_20260910/analysis.json
```

Use fresh output paths for reruns. No production adoption, new dependencies, commit or push.
