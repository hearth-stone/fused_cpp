# L2-sized ring intervention — measured 2026-09-08

## Status

Connection recovered on retry; both formal sessions and all native smoke
checks completed on Arm-codex-internal. Enlarging the ring to192KiB increases
loaded matrix4 time by paired14.83/14.03us (7.60/7.16%) without background,
but its large-minus-small-background gap remains +0.13/+0.31us with intervals
crossing zero. The same-session streaming control retains +26.03/+28.72us.

The complete L2-state gate FAILS: loaded4/resident4 L1 refill/access medians
are0.972–0.990%, just below the predeclared1% threshold. Other L2-sized probe
families pass. Keep the failed gate; do not claim the entire grid proves
L2-specific behavior or adjust the threshold after observing results.

Class E, Lab only. Production defaults, ABI, planner/calibration and frozen
response model are unchanged. Rollback boundary: the optional L2 grid,
generator ring-size argument, reader/analyzer and dedicated tests.

## Predeclared intervention

- Retain all 39 cells from `--l1-mix` and add six 192 KiB ring probes under
  none/16 M1/16 M120 W13 backgrounds: 57 cells per round.
- A/B ring line footprints change from 4/8 KiB to 64/128 KiB. The target has
  64 KiB L1D and 1280 KiB private L2. Allocation capacity alone does not prove
  residency; the PMU gates below decide whether interpretation is permitted.
- Use the same generated address template: only the AND mask changes from63
  to1023. Both rings fit within the existing A/B allocations. The 1024-slot
  ring covers one complete K4096 sweep, repeated over N tiles.
- Probes22–27 mirror13–18: load-only, loaded matrix2/4/8, resident matrix4
  with loads, and NOP4. Reuse existing pure2/4/8 and streaming controls.
- Same process, allocation, persistent touched workspace, 256 MiB scrub,
  four-copy B rotation and premeasurement victim-local preparation. Existing
  `l1_prepared` JSON field means selected-ring preparation also for L2; new
  `ring_slots` explicitly identifies64/1024, avoiding a residency assertion.
- Two independent sessions, seeds149808/159808, 5 warmups and31 measured rounds.
  NUMA3 memory, allowed CPUs240–319, controller240, victim304, background288–303.
  No fit, no threshold adjustment after observing results.

## Gates and interpretation

For every L2 load cell: median L1 refill/access event ratio >=1%; median L2
refill/access <=0.1% and P90 <=0.5%. Preserve preceding L1 gates. All core/DDRC
running ratios >=0.99; L1/L2 retired-instruction delta <=64 wrapper instructions.
These are event ratios, not exact demand-load hit probabilities. Prefetch and
the measurement wrapper can contribute. Failed gates preclude an L2-specific
claim, but their raw results must still be retained.

Compare round-paired large-minus-small background differences and L2-minus-L1
times for each schedule. Report bootstrap intervals and absolute medians.
Pure/mixed max/sum comparisons retain the preceding schedule/operand caveats:
they do not uniquely identify a shared port or additive memory cost.

## Reproduction commands

Use the target project's existing virtualenv. Verify no competing benchmark,
target topology and source identities before running. Never overwrite a
completed session; the runner uses exclusive output creation.

```sh
ssh -o ConnectTimeout=10 Arm-codex-internal \
  'cd /home/zhangxu/codex/fused_cpp && mkdir -p tmp/l2_matrix_mix_20260908'
scp optimizations/fused_moe_sve/benchmarks/phase_supply_native.cpp \
  optimizations/fused_moe_sve/benchmarks/m1_supply_probe.h \
  optimizations/fused_moe_sve/benchmarks/bench_phase_supply.py \
  optimizations/fused_moe_sve/benchmarks/linux_perf_event.py \
  tmp/pressure_response_fit_20260908/frozen_fit.json \
  Arm-codex-internal:/home/zhangxu/codex/fused_cpp/tmp/l2_matrix_mix_20260908/
```

On the target, from `/home/zhangxu/codex/fused_cpp`:

```sh
g++ -std=c++17 -O3 -pthread -march=armv8.2-a+bf16+sve -msve-vector-bits=256 \
  -DFUSED_CPP_MOE_HAS_XBYAK_AARCH64=1 -DFUSED_CPP_MOE_SVE_VECTOR_BITS=256 \
  -Icsrc/moe/arm/sve_bf16 -Irefs/i8gemm \
  -I3rdparty/xbyak_aarch64 -I3rdparty/xbyak_aarch64/xbyak_aarch64 \
  tmp/l2_matrix_mix_20260908/phase_supply_native.cpp \
  csrc/moe/arm/sve_bf16/jit_kernels.cpp \
  3rdparty/xbyak_aarch64/src/xbyak_aarch64_impl.cpp \
  3rdparty/xbyak_aarch64/src/util_impl.cpp \
  -o tmp/l2_matrix_mix_20260908/phase_supply_native
```

First run `--l2-mix --rounds 1 --warmup 0 --no-pmu`, then a one-round PMU
smoke and an old-protocol smoke, each with a distinct output path. Proceed
only if generated-code identity, numerical and protocol checks pass.

```sh
numactl --physcpubind=240-319 --membind=3 .venv/bin/python \
  tmp/l2_matrix_mix_20260908/bench_phase_supply.py \
  --binary tmp/l2_matrix_mix_20260908/phase_supply_native \
  --output tmp/l2_matrix_mix_20260908/session1.jsonl --l2-mix --seed 149808
```

Repeat with session2 and seed159808.
Archive exact build inputs and collect raw to the local ignored directory.

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_l2_matrix_mix.py \
  --sessions tmp/l2_matrix_mix_20260908/session1.jsonl tmp/l2_matrix_mix_20260908/session2.jsonl \
  --frozen-fit tmp/pressure_response_fit_20260908/frozen_fit.json \
  --output tmp/l2_matrix_mix_20260908/report.json
```

## Results

No-background medians in us (S1/S2). Paired changes below need not equal
differences of independent medians.

| Schedule | 12KiB ring | 192KiB ring | Paired increase |
| --- | --- | --- | --- |
| Load-only | 59.90/60.00 | 104.55/104.24 | 44.63/44.32 |
| Loaded matrix2 | 136.05/136.04 | 138.22/137.67 | 3.32/2.42 |
| Loaded matrix4 | 195.68/195.88 | 210.15/209.61 | 14.83/14.03 |
| Loaded matrix8 | 362.65/362.70 | 363.21/363.00 | 0.36/0.40 |
| Resident matrix4 with loads | 204.16/196.00 | 207.72/207.72 | 6.48/9.30 |

Large-M background minus small-M background, paired us [95% bootstrap CI]:

| Victim | S1 | S2 |
| --- | --- | --- |
| Real streaming W13 | 27.41 [20.65,30.92] | 28.65 [22.20,34.67] |
| Streaming matrix4, no stores | 26.03 [20.11,35.44] | 28.72 [25.63,34.30] |
| 12KiB loaded matrix4 | 0.07 [-3.57,5.41] | 0.12 [-2.26,3.74] |
| 192KiB load-only | -0.11 [-0.76,0.07] | 0.03 [-0.37,0.38] |
| 192KiB loaded matrix2 | 0.12 [-0.18,0.35] | 0.02 [-0.18,0.24] |
| 192KiB loaded matrix4 | 0.13 [-0.54,0.73] | 0.31 [-0.38,0.60] |
| 192KiB loaded matrix8 | -0.17 [-0.30,0.02] | -0.10 [-0.34,0.09] |
| 192KiB resident matrix4 with loads | -0.37 [-2.12,2.27] | 1.89 [-0.90,3.54] |

All 36 original L1 cell/session gates pass. Of the36 new cell/session gates,
24 pass and12 fail, exclusively loaded4/resident4. L1 refill/access is
57.47–58.63% for load-only/NOP,1.182–1.332% for loaded2,1.033–1.053% for
loaded8, and0.972–0.990% for loaded4/resident4. Every new cell passes the
low-L2-refill gate: worst median0.01739%, worst P90 0.05461%.

The instruction schedule dramatically changes observed L1 refill counts
despite identical ring coverage. The present counters do not separate demand
refills from all prefetch/service effects, so do not equate192KiB capacity
coverage with each measured load being served from L2. In particular, the
load-only gate cannot certify the mixed-compute cells on its behalf.

Conclusion: a larger, repeatedly reused working set has a measurable local
cost, much of which is hidden in matrix2/8 schedules. It does not reproduce
the large streaming background penalty. This weakens a generic fixed penalty
for any non-L1 working set, but does not uniquely locate the remaining effect
in LLC, DDR, prefetch, return queues or issue resources. No fitting/adoption.
If continuing, distinguish demand/prefetch supply on the same fixed probes
before calling this a pure L2 latency intervention; keep this failed-state-gate
result as evidence rather than silently replacing it.

## Artifacts and validation

Local and remote ignored artifacts: `tmp/l2_matrix_mix_20260908/`, remote root
`/home/zhangxu/codex/fused_cpp` on Arm-codex-internal. Contains two raw sessions,
three smoke records and `build_inputs.tar.gz`; local `report.json` holds full
medians, means, standard deviations, P90/P99, paired intervals and PMU gates.
Source baseline is HEAD `c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5` plus dirty
Lab changes, not a clean-commit production artifact. SVE256 BF16, g++ O3,
M1/1T W13 geometry K4096/N1024, full owner stripe, no W2 measurement.
Existing THP policy and ordinary aligned allocations; no verified HugeTLB claim.

| Identity | SHA256 |
| --- | --- |
| Binary | `d03692b36784bde4558bb8958ecca4bbc352f86cdaa244c75744786eb8adb112` |
| Production JIT source | `1bcdaf58139a2d2c3ace219b86e9e568b1e222f813877a1198d31e34d7050629` |
| Frozen fit | `b03819c0c5d04a5ce81424602851533e2dc2e5f8d39d68e104dfdb3bab0c4781` |
| Session1 | `62f5c65bcb3db86a66da9a2af78c6aae1b70a1e0b28f3f838832256dbd0d3720` |
| Session2 | `94988e056d2074c7ca1eabd3c9c7641cb279ad1a7ab62a4df5755eb10fc6d647` |

`.venv/bin/pytest -q tests/test_moe_l2_matrix_mix.py tests/test_moe_l1_matrix_mix.py
tests/test_moe_phase_supply.py tests/test_moe_pressure_curve.py
tests/test_moe_ab_supply_contrast.py`: **38 passed**.
Ruff passes for the three changed Python modules and new test; C++ formatted
with Google style, column limit120. On the recovered target: native build,
57-cell no-PMU/PMU smoke, old-protocol smoke, and both57-cell×31-round sessions
pass correctness, preparation, instruction-count and running-ratio checks.
Generated B-only/full-no-store4 identity and victim/background numerical checks
pass. Cache-state failures above are not correctness failures and are not
hidden. No production end-to-end validation or model integration was performed.
