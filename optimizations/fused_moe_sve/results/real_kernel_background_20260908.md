# Frozen-response transport to real W13 backgrounds

## Decision

**Do not generalize the reader-calibrated scalar response to arbitrary kernel
competition. Keep the previous frozen version and retain this counterexample.**
All12 cross-build control gates pass, but only5/12 real-background shape/session
gates pass. The model still substantially helps M1 overall; M12 is often more
accurate with its isolated constant. No parameters or anchors were changed.

Primary evaluation excludes none/reader controls and covers three real
backgrounds ×three victim widths ×two sessions=18 medians per victim M.

| Victim | Predictor | MAPE | Maximum absolute percentage time error |
|---|---|---:|---:|
| M1 | Frozen isolated | 15.414% | 35.815% |
| M1 | Frozen response | **2.027%** | **7.998%** |
| M12 | Frozen isolated | **.247%** | **.759%** |
| M12 | Frozen response | .664% | 1.183% |

Errors use measured total victim time as denominator, not incremental contention
cost. These are prediction results, not runtime speedups or a new cost-model fit.

## Direct matched-supply counterexample

Victim M1/1T, same process/allocations and fixed16 background cores:

| Session | Background | Independent B-only median | Victim median | Frozen prediction | Time error |
|---:|---|---:|---:|---:|---:|
| 1 | 16×M1 | 334.42us | 427.83us | 421.99us | −1.37% |
| 1 | 16×M120 | 334.80us | 459.32us | 422.58us | **−8.00%** |
| 2 | 16×M1 | 336.65us | 427.43us | 425.50us | −.45% |
| 2 | 16×M120 | 336.88us | 459.12us | 425.86us | **−7.24%** |

Ratio-of-median victim difference is approximately7.4%. Round-paired medians,
with bootstrap95% intervals, give:

| Session | Large/small B-only time difference | Large/small victim time difference |
|---:|---:|---:|
| 1 | +.209% [−.332%,+.843%] | **+7.994% [+6.586%,+9.412%]** |
| 2 | +.681% [−.464%,+1.022%] | **+7.794% [+6.329%,+8.741%]** |

Thus approximately equal measured B-only supply does not imply equal M1 time
under these backgrounds. This is evidence against the transportability of the
current smooth scalar response, not a proof that no conceivable scalar function
could interpolate every point or that memory competition is irrelevant.

Domain27 read throughput during M1 is approximately73GB/s with small backgrounds
but43GB/s with large backgrounds, despite the latter causing more victim delay.
Session1 victim LLC misses are58,351 vs98,128; session2 is58,340 vs97,171.
Average DDR throughput alone does not explain this ordering. Request shape,
cache residency, A/B interactions and phase alignment are hypotheses, not
identified causes. Aggregate PMU does not attribute misses to A or B addresses.

All1T real-background supply points exceed the original fitted maximum26.19%:
approximately28.6–29.6% for small/large and38.2% for mixed. This is an important
extrapolation limitation. Nevertheless, the small-vs-large matched-supply
contrast occurs at nearly the same new pressure and repeats within each new
session. Do not explain the whole result merely as larger average pressure.
2T/4T real points remain within fitted maxima; M12 regressions also occur there.

## Full gate results

Gate fixed before collection: controls (none and16 readers) maximum response
error≤5%; then each shape/session over the three real backgrounds requires
MAPE≤2%, maximum≤5%, and MAPE below frozen isolated. No thresholds were relaxed.

| M | T | Response MAPE S1 / S2 | Response max S1 / S2 | Gate S1 / S2 |
|---:|---:|---:|---:|:---|
| 1 | 1 | 4.070% /3.542% | 7.998% /7.245% | fail /fail |
| 1 | 2 | 1.256% /1.120% | 2.206% /1.200% | pass /pass |
| 1 | 4 | 1.287% /.887% | 1.532% /1.141% | pass /pass |
| 12 | 1 | .887% /.822% | 1.183% /.989% | fail /fail |
| 12 | 2 | .572% /.612% | .645% /.800% | fail /fail |
| 12 | 4 | .425% /.663% | .476% /.950% | pass /fail |

M12 failures are baseline-comparison failures, not violations of2%/5% absolute
limits. Its observed real-background change is small, and the reader response
overcorrects it. No new fallback was selected using these validation results.

The largest no-background error is1.105%; all none/reader control errors are
≤3.350%. Passing controls limits the build-change confound but does not prove
binary equivalence. The same new binary runs every background within each
randomized session. Reader M1/1T time itself differs between sessions
(423.06/513.76us); B-only also differs (332.16/381.79us), and model errors are
−1.095%/−3.350%. This variation is retained, not averaged away or reanchored.

## Background implementation and numerical checks

- All backgrounds use the actual production W13 JIT service, not a simulated
  FLOP loop. Large M120 is ten complete M12 panels, not a modified microkernel.
  This is real kernel code with synthetic constant inputs, not full MoE traces.
- Five conditions:0 none;1 original16 stream readers;2 sixteen M1 kernels;
  3 sixteen M120 kernels;4 eight M1 plus eight M120 (even/odd background cores).
  The serialized background field/pressure_levels contains **condition IDs,
  not reader counts** in this protocol.
- Background cores288–303, each1T, fixed for every nonzero condition; victims
  CPU304+ at1/2/4T; controller240; NUMA3 memory and allowed CPUs240–319.
- Each background thread owns32MiB packed weights, four8MiB copies. We reuse
  the reader's preallocated storage as read-only packed byte patterns consumed
  by JIT. Each owns an independent preallocated A and packed output buffer
  sized for M120. No background/output allocation occurs inside timed regions.
- Background starts5ms before victim release and loops until stopped. Every
  complete M1 or M120 call advances its weight copy; initial copy is offset by
  victim round copy plus lane. M120 reuses one B copy across its ten panels.
  Background is not synchronized at each panel, and its elapsed time is not
  the measured victim time. Stopping waits for the final complete background
  call outside the timer/PMU gate.
- Background outputs are poisoned before each cell and fully verified after
  stopping. All16 threads must complete at least one call. Verification checks
  all logical rows/columns, including packed physical8-row layout for M1.
  bg_calls reports whole-cell completed calls including startup/drain, not
  exact calls during the victim window. Numerical flags distinguish real
  victim/background checks from B-only/empty controls.
- BF16 inputs/weights1/64 give BF16 SiLU(1)=0x3f3b; this fixed-value correctness
  check does not replace broad random-input production numerical validation.

## Protocol and identity

Arm-codex-internal preflight load.01/.00/.00. Victim W13 H4096/F512,
K4096/N1024,SVE256,BF16,Ntile16,full owner stripes: total B8MiB,per-owner8/4/2MiB.
Existing degree5 SiLU, GCC13.2.0/O3/C++17. Ordinary aligned allocations;
no page policy changes or explicit HugeTLB; actual page residency not audited.

Persistent touched victim output,256MiB scrub per cell,4-copy victim rotation,
randomized cell order. Each session uses one process and allocation set; two
independent sessions seeds49808/59808,5 warmup+31 recorded rounds.60 cells/round:
30 real victim cells,15 B-only and15 empty controls. Two sessions total3,720
recorded cells;1,860 numerically checked real victims. Background verification
also applies to B-only/empty cells when real kernels are active.

Per-victim core PMU cycles/instructions/L2 refill/LLC read miss/backend stall;
48 DDRC events. Each cell independently resets/reads counters. DDRC uses its
own longer controller gate; counts include victim and background, not isolated
victim traffic. Software reader line-byte counts are not DRAM bytes and are not
reported as real-kernel bandwidth. All counter coverage/running ratios pass.
Mean/std/P90/P99, paired intervals and core/DDRC medians are in evaluation.json.

Frozen model SHA256 remains
`b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781`.
Original reader binary:
`aec4ec8050ecb32910ec92fc40b64ea049edba5e886891a13c3e39df45c7e8b6`.
New measured binary:
`9158686e733ca2fbecf578af4564a57fe3f92420b5ae8d4d1876234a6803d04f`.
Production JIT source unchanged:
`1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629`.
The replay requires the explicit new binary hash and keeps the original hash
in its provenance; it does not silently bypass the older replay's identity gate.

## Reproduction and retention

Remote directory:
`Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/real_kernel_background_20260908/`.
Contains original measured source/binary, exact driver, frozen model, correctness
and PMU smokes, both sessions, legacy smoke, and build_inputs.tar.gz. Local raw
copies, evaluation.json and build snapshot use the same relative directory.
Final C++ review adds only explicit braces; the original measurement snapshot
is preserved, and the reviewed source/binary is separately named. The reviewed
binary also compiled successfully and passed its own60-cell no-PMU smoke.

```sh
# Remote build (repository root; original measured source snapshot).
g++ -std=c++17 -O3 -pthread -march=armv8.2-a+bf16+sve -msve-vector-bits=256 \
  -DFUSED_CPP_MOE_HAS_XBYAK_AARCH64=1 -DFUSED_CPP_MOE_SVE_VECTOR_BITS=256 \
  -Icsrc/moe/arm/sve_bf16 -Irefs/i8gemm -I3rdparty/xbyak_aarch64 \
  -I3rdparty/xbyak_aarch64/xbyak_aarch64 \
  tmp/real_kernel_background_20260908/phase_supply_native.cpp \
  csrc/moe/arm/sve_bf16/jit_kernels.cpp \
  3rdparty/xbyak_aarch64/src/xbyak_aarch64_impl.cpp \
  3rdparty/xbyak_aarch64/src/util_impl.cpp \
  -o tmp/real_kernel_background_20260908/phase_supply_native
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/real_kernel_background_20260908/bench_phase_supply.py \
  --binary tmp/real_kernel_background_20260908/phase_supply_native \
  --output tmp/real_kernel_background_20260908/session1.jsonl --real-background --seed 49808
# Independent session2 uses seed59808 and session2.jsonl.
.venv/bin/python optimizations/fused_moe_sve/benchmarks/replay_kernel_background.py \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --sessions tmp/real_kernel_background_20260908/session1.jsonl tmp/real_kernel_background_20260908/session2.jsonl \
  --binary-sha256 9158686e733ca2fbecf578af4564a57fe3f92420b5ae8d4d1876234a6803d04f \
  --output tmp/real_kernel_background_20260908/evaluation.json
.venv/bin/python -m pytest -q tests/test_moe_pressure_response.py tests/test_moe_pressure_curve.py tests/test_moe_phase_supply.py tests/test_moe_kernel_response.py
```

Class E/M Lab-only change. Existing native harness gets an optional kernels
protocol; production kernel sources, model parameters, v8, APIs, default build
and pruning are unchanged. No-PMU and PMU60-cell smokes passed before formal
timing; old20-cell W13/W2 no-PMU protocol passed afterward.41 focused tests pass.
No full operator/W2 background, random-input suite, sanitizer run or full
timing-only repeat was performed; no corresponding broader claim is made.
Retain this as a matched-supply counterexample and regression reference, not a
new model candidate. No commit or deployment performed.
