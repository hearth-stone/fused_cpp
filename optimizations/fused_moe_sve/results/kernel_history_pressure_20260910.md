# Internal history × external competition factorial, 2026-09-10

## Question and scope

The user requested testing whether the internal post-panel slowdown and external competition coexist independently or interact. This class E Lab diagnostic crosses the axes in one harness and process; it does not pool absolute timing from previous sessions. Production source, planner/model coefficients, build defaults and schemas are unchanged. Rollback is limited to the added diagnostic files and manifest/report entry.

For fixed-address M1 W13/W2 define `T(h,p)`, where `h` is the number of preceding full M12 panels using the same B, and `p` is the controlled background configuration. Compare:

```text
T00 = T(0, none)
H(h) = T(h, none) / T00
E(p) = T(0, p) / T00
T_product(h,p) = T00 * H(h) * E(p)
T_additive(h,p) = T(h,none) + T(0,p) - T00
Q(h,p) = T(h,p) / T_product(h,p)
H_under_pressure(h,p) = T(h,p) / T(0,p)
```

`Q=1` is multiplicative separability. `Q<1` is submultiplicative and `Q>1` supermultiplicative. These are empirical descriptions, not proof of distinct physical bottlenecks. A confidence interval containing1 does not establish equivalence. Report absolute prediction errors as well as ratios. The product is the independently measured-axis version of the previous H×pressure proposal, avoiding pressure-fit error as a confound. Additive prediction is a second prespecified comparator, not an after-the-fact alternative chosen for a favorable result.

## Protocol

Grid: histories0/1/8 × W13/W2 × backgrounds none,16×M2,16×M120,38×M2,38×M120,38 mixed alternating19×M2+19×M120 =36 conditions. History8 is a reference history, not guaranteed warm cache under competition. The M1 tail always starts at row96, preserving its A/C/routes addresses and packed input across all histories. Prefixes use rows0–11 or0–95. Thus history1 deliberately leaves a gap in A/C before the fixed tail; this is a state diagnostic rather than contiguous real-expert timing.

Arm-codex-internal (same host has alternate alias Arm-codex), foregroundCPU316; background CPUs280–318 excluding316, first16 or all38. NUMA3 memory, launch allowed CPUs240–319. BF16 SVE256, H4096/F512, N tile16,1T foreground and1T per competitor. W13 degree5 fused SiLU, W2Direct degree0. Backgrounds are continuous W13-only expert calls, M2 or M120, and rotate four independent B copies. Full W13 B8MiB/W2 B4MiB, full owner stripes `(1,0,0,1,1)`. Ordinary vector allocation, no explicit HugeTLB. Full W2 output192MiB; foreground uses four B copies rotating by round.

Each condition initializes/poisons outputs, scrubs256MiB, starts backgrounds, waits for every competitor's first full expert call to finish, then uses a fixed5ms lead-in even without backgrounds. Competitors continue through all prefixes and the final tail. There is no inserted prefix/tail gap; actual gap and total prefix interval are recorded. Every prefix/tail logical output, untouched output region and competitor output is checked after timing. Packed W13 padding is not a logical output. Background CPU and start/end coverage include the full prefix-plus-tail interval.

Background execution therefore also lasts longer for history8, and its state can evolve during prefixes. This tests the practical separability of history and continuously running competition; it does not identify a physical cache state while holding all background service counters constant. No PMU or isolated cache-residency claim is made.

Planned36-cell smoke, then two independently launched sessions of5 warmup and31 measured randomized rounds (seeds603000/603001/603002). Compare within each session; keep session variability visible. Analysis uses10,000 paired-round bootstrap resamples (seed603100), matching round/B-copy identity across the four separate conditions. Report95% intervals, p50/p90/p99, mean/std/CV and absolute times; do not trim outliers.

## Implementation and commands

- `benchmarks/kernel_history_pressure_native.cpp`: optional standalone target reusing established pressure-worker and history-packing patterns; original harnesses unchanged.
- `benchmarks/bench_kernel_history_pressure.py`: complete factorial driver with create-exclusive artifacts and validation.
- `benchmarks/analyze_kernel_history_pressure.py`: independent-axis and joint comparison, complete-grid checks and bootstrap intervals.
- `tests/test_moe_kernel_history_pressure.py`: factorial-axis coverage, fixed-address/worker validation and interaction arithmetic.

Local artifacts: `tmp/kernel_history_pressure_20260910/`. Intended remote directory: `Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/kernel_history_pressure_20260910/`. The scripts retain the established C++17/O3/pthread, armv8.2-a+bf16+sve/SVE256 build flags and1-thread library environment. `git_status_before.txt` records pre-existing uncommitted work. HEAD is `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5`.

```bash
.venv/bin/pytest -q tests/test_moe_kernel_history_pressure.py
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/kernel_history_pressure_20260910/build_smoke.sh'
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/kernel_history_pressure_20260910/run_sessions.sh'
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_kernel_history_pressure.py \
  --sessions tmp/kernel_history_pressure_20260910/session1.jsonl tmp/kernel_history_pressure_20260910/session2.jsonl \
  --output tmp/kernel_history_pressure_20260910/analysis.json
```

Use fresh output paths when rerunning. Synchronize the retained source/driver/build scripts before the first remote invocation. No commit or push is requested or performed.

## Completed measurements and result

Connectivity recovered on the user's retry. The original JIT identity matched, the remote experiment directory had no previous data, and the retained sources were synchronized before execution. The build and36-cell numerical smoke passed, followed by both36×36-cell sessions:2,628 checked conditions including2,232 formal measurements. All native/build/runner stderr files are empty. Complete unique grid/round coverage and source hashes were verified. The previously passed three local tests remain applicable because implementation and analysis source are unchanged.

Native binary SHA256 `63c1a2a40023afc495fb8273af2c7ec6b98d4a6e6bfab6927b4bebb33edd34dc`; native source SHA256 `31c61cfdfc48984752caa17ef385018e79f8adcb0165e544feb0da6cf08a6caf`; unchanged production JIT SHA256 `1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.

### Independent history axis

No-background tail medians (us):

| Session | Stage | History0 | History1 | History8 |
| --- | --- | ---: | ---: | ---: |
| 1 | W13 | 303.21 | 326.77 | 283.40 |
| 1 | W2 | 174.13 | 193.40 | 142.02 |
| 2 | W13 | 302.53 | 326.31 | 283.36 |
| 2 | W2 | 170.31 | 189.31 | 141.55 |

History1 alone slows W13 by7.77%/7.86% and W2 by11.07%/11.16% in sessions1/2. History8 alone improves W13 by6.53%/6.34% and W2 by18.44%/16.89%. These are this harness's matched-session baselines, not replacements for previous absolute values.

### One preceding panel under competition

Each cell below is `external-only / joint / independent-product prediction`, in microseconds.

| Background | W13 session1 | W13 session2 | W2 session1 | W2 session2 |
| --- | --- | --- | --- | --- |
| 16 × M2 | 487.30 / 489.42 / 525.16 | 493.55 / 483.70 / 532.34 | 259.76 / 259.24 / 288.51 | 259.67 / 255.61 / 288.64 |
| 16 × M120 | 427.58 / 426.61 / 460.80 | 427.88 / 412.59 / 461.51 | 237.62 / 246.95 / 263.92 | 239.65 / 248.63 / 266.39 |
| 38 × M2 | 952.96 / 956.19 / 1027.01 | 951.18 / 954.15 / 1025.95 | 506.67 / 498.80 / 562.74 | 504.30 / 505.56 / 560.56 |
| 38 × M120 | 707.10 / 682.30 / 762.04 | 708.73 / 682.40 / 764.44 | 371.45 / 365.73 / 412.56 | 369.53 / 368.95 / 410.76 |
| 38 × mixed | 873.05 / 870.95 / 940.89 | 874.47 / 873.22 / 943.21 | 463.49 / 465.98 / 514.78 | 466.10 / 459.86 / 518.10 |

Across all20 history1 joint points, history under pressure changes the external-only time by-3.72% to+3.93%. The interaction ratio Q is below1 with its95% interval entirely below1 in all20 points. The independent-product model overpredicts all20: MAPE10.06%, MAE51.17us. Additive prediction is closer but still has MAPE5.30%, MAE25.16us. This rejects a universal independent multiplier for these measured conditions. It does not prove the internal mechanism has physically disappeared; its incremental timing effect is largely absent or altered in the joint environment.

### Eight preceding panels under competition

The joint W2 tail is especially sensitive to the background configuration:

| Background | Session1 external-only → joint (us) | Session2 external-only → joint (us) |
| --- | ---: | ---: |
| 16 × M2 | 259.76 → 145.21 | 259.67 → 144.43 |
| 16 × M120 | 237.62 → 143.80 | 239.65 → 144.22 |
| 38 × M2 | 506.67 → 484.21 | 504.30 → 481.48 |
| 38 × M120 | 371.45 → 311.39 | 369.53 → 309.24 |
| 38 × mixed | 463.49 → 446.25 | 466.10 → 448.70 |

With16 competitors, history8 W2 costs143.80–145.21us, near its no-background history8 reference141.55–142.02us. With38 M2 competitors, it costs481.48–484.21us, close to the external-only504.30–506.67us. The reuse benefit is therefore not a pressure-independent factor. This is consistent with pressure-dependent reuse/service behavior, but no cache-residency mechanism is identified by timing alone.

Across the20 history8 joint points, product MAPE is13.92%, MAE42.01us; additive MAPE14.36%, MAE33.95us. Q intervals are entirely below1 in9 points, above1 in9 and include1 in2. The two intervals including1 do not establish equivalence. A single global correction multiplier cannot capture both signs of the interaction.

### Limits and decision

The largest per-cell CV is11.05%. Both sessions show the principal history1 suppression and history8 dependence on background type/count. All40 pointwise bootstrap intervals and full per-cell p50/p90/p99, mean/std/CV remain in `analysis.json`. Intervals are pointwise, share measured axis baselines and are not40 independent experiments; no multiplicity-adjusted population claim is made.

Backgrounds run during the prefixes as well as the tail, so the joint history can differ physically from history prepared without competition. The foreground target address is fixed, but background evolution and cache residency are not independently frozen. No PMU, multithread-team, other-M or real-input validation is claimed.

Retain this as a bounded reference and reject the assumption that the measured no-background H(h) can be multiplied unchanged by an independently measured external factor for all joint cases. A future model needs a conditional history factor H(h,p), or equivalent history-dependent competition response. This experiment does not fit that new function or select its physical mechanism. No production or active planner baseline is changed.
