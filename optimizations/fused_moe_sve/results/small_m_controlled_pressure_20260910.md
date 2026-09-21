# M1–11 stage response to controlled GEMM competition, 2026-09-10

## Result

Both foreground M and competitor type materially change the measured response. At38 peers, M2 backgrounds slow every measured M1–11 W13/W2 victim more than M120 backgrounds in both sessions; all44 corresponding individual small-minus-large95% intervals are positive. This is a repeated controlled workload-type effect at fixed active-core count and allocated B footprint, not an identified physical compute/memory fraction.

For example, M1/W13 baseline is306.06/306.79us. With38 M2 peers it is948.59/955.73us, versus710.73/706.55us with38 M120 peers. M1/W2 baseline is167.35/167.10us; corresponding M2-background times are499.37/494.42us and M120-background times371.81/367.68us. Baseline and condition marginal medians are not used to recompute the paired percentages below.

The tables show session1 / session2. Baseline units are microseconds; other columns are same-round median percentage **increases** over the victim's own no-background time. A210% increase means about3.10 times the baseline duration. Small=M2, large=M120; mixed=19 of each. All rows are retained, including small/negative changes.

### W13

| M | No background, us |16 small|16 large|38 small|38 large|38 mixed|
|---:|---:|---:|---:|---:|---:|---:|
|1|306.06 /306.79|+61.99% /+61.83%|+40.11% /+40.04%|+209.80% /+210.96%|+132.14% /+130.46%|+185.07% /+184.94%|
|2|306.58 /307.06|+61.05% /+60.27%|+39.09% /+40.58%|+210.96% /+210.34%|+131.98% /+131.48%|+184.36% /+185.21%|
|3|422.90 /422.49|+29.59% /+29.43%|+18.47% /+19.42%|+136.04% /+138.35%|+77.74% /+76.83%|+116.96% /+117.11%|
|4|424.79 /422.05|+29.41% /+29.33%|+18.90% /+18.86%|+135.04% /+138.16%|+76.25% /+77.81%|+117.47% /+116.70%|
|5|633.04 /632.74|+2.99% /+3.67%|+1.85% /+1.36%|+62.36% /+64.26%|+24.14% /+24.03%|+50.10% /+50.18%|
|6|632.76 /631.53|+3.42% /+3.31%|+2.28% /+2.29%|+63.42% /+63.33%|+24.25% /+24.51%|+50.02% /+50.36%|
|7|775.93 /776.40|+2.11% /+1.79%|+1.69% /+1.64%|+38.19% /+37.34%|+10.34% /+9.91%|+28.35% /+28.46%|
|8|776.65 /776.94|+1.86% /+1.78%|+1.50% /+1.35%|+37.70% /+37.70%|+10.40% /+9.81%|+28.27% /+28.42%|
|9|1093.16 /1095.92|+0.84% /+0.65%|+0.65% /+0.54%|+13.50% /+13.15%|+4.44% /+4.23%|+11.32% /+11.13%|
|10|1092.14 /1094.84|+0.81% /+0.72%|+0.63% /+0.53%|+13.48% /+13.16%|+4.27% /+4.10%|+11.33% /+11.22%|
|11|1233.38 /1235.45|+0.49% /+0.48%|+0.37% /+0.34%|+10.84% /+10.73%|+3.25% /+3.14%|+9.24% /+8.74%|

### W2

| M | No background, us |16 small|16 large|38 small|38 large|38 mixed|
|---:|---:|---:|---:|---:|---:|---:|
|1|167.35 /167.10|+53.33% /+55.17%|+40.43% /+43.69%|+195.91% /+198.18%|+122.42% /+120.13%|+170.96% /+177.54%|
|2|168.71 /165.59|+53.73% /+56.35%|+42.83% /+39.85%|+200.20% /+200.19%|+121.15% /+124.70%|+169.15% /+175.74%|
|3|224.05 /224.84|+23.37% /+21.61%|+16.01% /+14.91%|+121.80% /+119.42%|+66.75% /+65.71%|+104.95% /+104.41%|
|4|225.05 /225.63|+21.46% /+21.45%|+16.90% /+14.84%|+123.35% /+120.41%|+65.76% /+64.23%|+104.97% /+103.53%|
|5|320.67 /319.80|+1.72% /-1.15%|+1.81% /-0.92%|+62.65% /+59.97%|+26.28% /+24.24%|+50.55% /+49.44%|
|6|319.97 /315.44|+1.20% /+1.90%|+1.70% /+1.79%|+60.48% /+63.58%|+25.89% /+27.35%|+50.79% /+51.50%|
|7|411.58 /411.61|-0.02% /-1.38%|+0.21% /-0.82%|+34.00% /+33.55%|+10.63% /+9.75%|+26.52% /+26.12%|
|8|409.61 /410.84|+1.35% /-1.75%|+0.54% /-0.05%|+33.09% /+32.39%|+10.58% /+10.08%|+27.65% /+25.00%|
|9|564.38 /562.61|+0.02% /+0.59%|+0.12% /+0.33%|+13.53% /+12.33%|+3.39% /+3.21%|+10.60% /+8.84%|
|10|562.09 /561.70|+0.70% /+0.58%|+0.51% /+0.39%|+13.37% /+13.88%|+3.88% /+3.21%|+11.60% /+8.78%|
|11|638.88 /638.41|-1.53% /-1.74%|-1.83% /-1.98%|+8.77% /+6.47%|+2.67% /+1.84%|+7.96% /+6.14%|

### Interpretation

- M1/2 have the largest relative response. With38 small peers, W13 increases about210% and W2 about196–200%. M3/4 remain highly sensitive; M9–11 have much smaller proportional responses.
- At16 peers, M1–4 distinguish small versus large backgrounds clearly in both stages. Larger foreground M often sees little change; small/large differences are not uniformly significant there. W13 has repeated positive small-minus-large intervals for M1/2/3/4/6/7/8; W2 only M1–4. A small statistically resolved difference is not a deployment recommendation.
- At38 peers, all11 foreground M values distinguish competitor type in both stages, even M11. Active-core count alone therefore cannot express the observed workload-type difference. The complete cost model may have additional resource terms; this experiment does not evaluate or refit their predictions.
- The tested half-small/half-large mixture lies between the pure backgrounds in the reported median responses and is often nearer the small-background result. One mixture does not establish a linear or universal composition law.
- Do not carry M12's previously observed low sensitivity into all these kernels. Do not infer a constant per-M memory-time percentage from these slowdown curves. M, stage, competitor type and pressure level all affect the measured response.

Within-cell CV ranges0.271–3.763% for W13 and0.592–5.372% for W2. Session changes and individual bootstrap intervals are retained in `report.json`. Small negative16-peer W2 increments, including M11, are reported as observed harness effects; no physical explanation or intrinsic kernel speedup is claimed. The many intervals are individual comparisons, not family-wise simultaneous confidence bounds.

![Both stages and both sessions](../../../tmp/small_m_pressure_20260910/response.png)

Decision: retain the complete response grid as a bounded Lab reference. It supports distinguishing foreground M and competitor type when investigating contention, but does not by itself validate a new planner response function or explain the earlier16T/1T transient order case quantitatively. Active model coefficients and experimental baseline are unchanged.

### Completed validation

The132-cell native smoke and both4752-cell formal sessions complete:9636 foreground calls pass numerical/coverage checks, with each active background's output, CPU, completed calls and bracketing interval validated for every cell. Formal analysis retains all9504 calls including warmups and uses8184 measured calls, producing264 condition medians,220 nonzero-condition response comparisons and88 fixed-count small/large comparisons. Metadata grid, round/copy identity, binary/driver identities and complete records are verified. All native/build/driver stderr files are empty.

Six focused tests pass, covering grid completeness, background type/CPU/coverage rejection and the paired denominator/sign. Ruff, clang-format dry-run and `git diff --check` pass. There is no new production kernel or numerical-contract claim; synthetic known-value checks do not replace broad production correctness tests.

## Question and protocol

The user requested the same controlled competition environments for every M1–11, recording W13 and W2 separately. This Class E Lab harness measures each victim's own no-background baseline plus matched competing-task conditions. It does not fit a compute/memory percentage or change the active cost model.

| Condition | Active1T peers | Background W13 M |
|---|---:|---|
|none|0|none|
|16 small|16|2 on every peer|
|16 large|16|120 on every peer|
|38 small|38|2 on every peer|
|38 large|38|120 on every peer|
|38 mixed|38|19 peers M2,19 peers M120, alternating by peer index|

Victim1T is pinned to physical CPU316. Background CPU order is280–318 excluding316;16-peer conditions use the first16 entries,38-peer conditions use all38. Every peer is in the victim's40-core LLC domain and excludes the victim CPU. Allocations and execution are bound to NUMA3. Backgrounds are W13-only in this first controlled grid; there are no W2 background teams, multi-thread background teams or real planner transitions in the experiment.

Per peer, four separate8MiB B copies are allocated in every condition. Small and large backgrounds at the same count therefore have the same active allocated B footprint:512MiB for16 peers and1216MiB for38 peers. A full background call rotates to the next B copy. M120 traverses ten M12 panels while reusing that call's B copy; M2 has one small panel. All allocated A/C capacities are the same, but active A/C footprints and compute work differ by background M. This intervention changes workload type; it does not isolate B traffic from compute, A, output or cache-state effects.

## Harness and numerical checks

The separate optional executable calls the unchanged production JIT internal service. No production source, extension registration, build default or model parameter changes. It reuses the W13 packed-output and W2 direct-FP32 conventions of the earlier standalone panel probes; absolute times must not be substituted for real-expert stage traces.

- Foreground W13: K4096/N1024, output512 BF16 columns, N tile16, degree5 fused SiLU. A and B are1/64; every logical output is checked against BF16 SiLU(1), `0x3f3b`. Padding rows are not logical outputs; a guard beyond the complete physical packed-panel extent must remain unchanged.
- Foreground W2: K512/N4096, packed A rows vary as `(1 + row % 4)/64`, B is1/64, direct contiguous route IDs. All192MiB output is poisoned before a W2 cell, then all valid rows and all untouched routes are checked against exact FP32 `(1 + row % 4)/8` or poison.
- Full packed-B W13=8MiB, W2=4MiB. Foreground and each peer have four B copies. Both foreground stages use full owner stripe `(1,0,0,1,1)` geometry; no external N-window loop.
- Every cell uses a256MiB scrub on the foreground CPU before background startup. No B preload or forekernel prepass. Background threads must each finish at least one full call; the victim then waits a fixed5ms lead-in. The no-background baseline uses the same5ms wait.
- Each background loops without a deliberate idle phase until the foreground timer finishes. Recorded first/last background timestamps must bracket the victim interval; all completed-call counts, numerical outputs and actual CPUs are validated. This is worker-loop coverage, not a sampled guarantee of uninterrupted instruction execution or a physical request-rate measurement.
- Backgrounds stop and join before output verification or the next cell. No PMU/syscall instrumentation occurs inside foreground timing; one enclosing monotonic-clock interval measures each foreground kernel call.

All132 combinations (`11 M ×2 stages ×6 conditions`) first run in a one-round smoke. Formal sessions use5 warmup and31 measured rounds, randomized cell order with the same B-copy index within each round. Seeds600001/600002; smoke600000. Two independently initialized processes/allocations. All cells, including warmups, must pass foreground and background checks.

## Statistics and interpretation boundaries

For each victim M/stage and each nonzero condition, compare the same-round sample against that victim's no-background sample: report median incremental microseconds and median percentage slowdown. Use2000 IID paired-round bootstrap resamples, seed600999, for95% intervals and report P10/P90. The samples are sequential randomized cells, not simultaneous measurements; IID intervals do not remove time dependence or allocation/session drift.

Separately compare small versus large backgrounds at fixed16 and38 active cores. Positive small-minus-large delta means small-background execution is slower. Each stage's raw baseline and condition statistics include median, mean, P10/P90 and CV. A repeated operational difference requires both session intervals to exclude zero in the same direction; smaller/inconsistent differences remain reported, not thresholded away. No samples are discarded and no measured point is used to refit the model.

Mixed backgrounds test one composition at38 cores. They do not identify a general mixture law. Changing16→38 also changes active B footprint and core placement; only the within-count small/large contrast holds those fixed. The foreground uses synthetic known-value inputs and standalone envelopes, so conclusions concern this controlled harness. Its no-background baseline is measured anew rather than borrowed from the real-expert M1–11 grid.

## Reproduction and evidence

Host `Arm-codex-internal`, project `/home/zhangxu/codex/fused_cpp`, NUMA3 CPUs240–319 allowed at process launch, foreground316 and background CPUs specified above. Existing compiler/JIT dependencies; C++17 `-O3 -pthread -march=armv8.2-a+bf16+sve -msve-vector-bits=256`, SVE256/N16. No global build or dependency installation. Ordinary `std::vector` allocations without explicit HugeTLB; actual per-page residency is not sampled.

```bash
.venv/bin/pytest -q tests/test_moe_small_m_pressure.py
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/small_m_pressure_20260910/build_smoke.sh'
ssh Arm-codex-internal 'cd /home/zhangxu/codex/fused_cpp && bash tmp/small_m_pressure_20260910/run_sessions.sh'
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_small_m_pressure.py \
  --sessions tmp/small_m_pressure_20260910/session1.jsonl tmp/small_m_pressure_20260910/session2.jsonl \
  --output tmp/small_m_pressure_20260910/report.json
```

Use fresh outputs for reruns. Source snapshots, build commands and hashes, smoke, full JSONL records, logs and aggregate report are retained in `tmp/small_m_pressure_20260910/` locally and under the same remote relative path. The original source tree is commit `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus preserved dirty work. Native binary SHA256 is `a0826b76002ff180925d6de1892eb579c8fb162197e8f33bf09ce069b9dbaf68`; `build_identity.txt` records native-source and production-JIT hashes and compiler version. No commit was made.
