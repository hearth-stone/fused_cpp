# Cross-machine fused-expert framework smoke

Date: 2026-08-27

Status: framework and mechanism smoke only. The exact-M cases use three timing
samples and the direct-route cases use five. These results are not paper
headline evidence and do not replace a 31-run frozen-build matrix.

## Provenance

- source revision: `982d4957f6128e380abdade0ed5dcf2095eea97b`;
- `3rdparty/xbyak_aarch64` commit:
  `3f8c682b9c6ff562dc008c7d1b8307a683f05d20`;
- ignored external `refs/i8gemm/lib` content SHA-256:
  `f7a10fb30652202c5d8198623c06ac89a1649e72ec7bc0836687bf320b78c789`;
- Arm-codex `_moe_C` SHA-256:
  `dd554ea366a2374a8ed51527d1e7a56942f0c824b4c348860457ac5a922b943f`;
- AmazonECS8Cores `_moe_C` SHA-256:
  `abe95405c82886419d4cbaaddb6aa2108908871645a90d26199b4b9a7074dc4e`.

The final cases reused MoE-only extension builds created earlier in the same
session. No `_moe_C` source or build input changed between those builds and the
reported revision; only Lab runner/configuration and documentation changed.
The final result bundles are:

- `20260827T092408Z-arm_codex_internal-fused_expert_smoke-982d4957f612`;
- `20260827T092412Z-amazon_ecs_8cores-fused_expert_smoke-982d4957f612`.

Both machines used SVE256, default page policy, `OMP_NUM_THREADS=1`,
`OMP_DYNAMIC=FALSE`, `OMP_PROC_BIND=FALSE`, and single-thread BLAS controls.
Arm-codex ran on NUMA3 CPUs `240-319`; AmazonECS8Cores ran on CPUs `0-7`.
The runner preserved the full outer affinity after importing PyTorch.

## Correctness

Both machines ran:

```bash
python -m pytest -q tests/test_fused_moe_bf16_tiled.py \
  -k w2_direct_route_store_matches_scatter --maxfail=1
```

Each reported `6 passed, 68 deselected`. The tests cover normal, scheduled,
and async bridges with FP32 direct route matching scatter bit for bit. Both
virtual environments warned that NumPy was unavailable; these focused tensor
tests and benchmarks do not require it.

## Exact-M smoke

Shape: eight sequential measured experts from 16 packed experts,
`H=4096,F=512`, one expert per wave, thread widths `1/4/8`, one warmup, three
alternating samples, static assembly as baseline, and JIT exact-M as candidate.
Entries are JIT throughput gain derived from complete-call median latency.

| M | Arm 1T | Arm 4T | Arm 8T | AWS 1T | AWS 4T | AWS 8T |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | +29.03% | +5.52% | +4.14% | +7.16% | +13.71% | +9.43% |
| 6 | +27.72% | +5.38% | +3.57% | +6.23% | +12.67% | +5.63% |
| 9 | +14.40% | +13.60% | +12.57% | +11.53% | +10.60% | +8.97% |
| 10 | +13.57% | +13.84% | +12.59% | +11.20% | +7.95% | +8.21% |
| 12 | -0.31% | -0.76% | -0.50% | +0.25% | -0.98% | +11.42% |

M5/6 and M9/10 reproduce the intended direction on both machines. M12 does no
less work in JIT and is the neutral control. The AWS M12/8T samples were noisy:
assembly ranged from 3.132 to 4.701 ms and JIT from 2.433 to 3.590 ms. Its
reported +11.42% median is not admissible as an exact-M gain; repeatability and
at least 31 samples are required.

## FP32 W2 direct-route smoke

Shape: async path, `tokens=256`, `TopK=6`, `H=4096,F=512`, eight experts,
full-N team stripes, two warmups, and five samples. Arm used 80 total threads
(10 per expert); AWS used eight total threads (one per expert).

| Machine | FP32 scatter median | FP32 direct median | Direct gain | Direct P10-P90 |
| --- | ---: | ---: | ---: | ---: |
| Arm-codex NUMA3 | 4.0839 ms | 4.0483 ms | +0.88% | 3.9461-5.4580 ms |
| AmazonECS8Cores | 15.4078 ms | 15.1847 ms | +1.47% | 14.9825-15.2886 ms |

The candidate remained bitwise identical to FP32 scatter. The Arm direct path
had a fault/tail sample, and five observations cannot characterize P90. The
bounded conclusion is only that this medium-route point is close to the noise
floor, consistent with the earlier long-route study; it is not evidence of a
general 1% gain.

BF16-route variants also ran as diagnostics but remain outside the paper's
default numerical path. Arm BF16 direct versus BF16 scatter measured +1.53%;
AWS measured -0.25%. No precision/default decision follows from these smoke
samples.

## Explicit fusion comparator blocked

The framework initially rebuilt and invoked `bench_unfused_pipeline`. The
standalone source guard incorrectly required `__ARM_FEATURE_BF16`; GCC 12/13
expose `__ARM_FEATURE_BF16_VECTOR_ARITHMETIC`, so the guard was repaired in
`e5f23f2` and both machines then compiled the target.

The comparator cannot currently produce a valid timing. Its explicit path
still evaluates historical poly5 exp while production fused W13 now uses
FEXPA+poly2. Without relaxing the predeclared gates, the checks measured:

| Shape | Intermediate relative L2 | Output relative L2 | Gate |
| --- | ---: | ---: | --- |
| H4096/F512/M192 | 1.603e-3 | 2.301e-3 | fail |
| H4096/F2048/M192 | 1.592e-3 | 2.293e-3 | fail |

The limits are `2e-4` and `1e-3`, respectively. The comparator was removed
from active smoke/pilot suites. A new same-activation, same-rounding explicit
reference is required before refreshing the historical fusion claim.

## Decisions and next runs

1. The declarative runner, clean snapshot sync, external-source hash, remote
   metadata, structured output collection, and two current machine configs are
   usable.
2. Keep exact-M and FP32 direct-route in the next 31-run matrix.
3. Add a repeated M12 control and report variability rather than one median.
4. Rebuild the explicit fusion comparator against the current activation
   contract before using it in a cumulative ablation.
5. Commit and pin the currently dirty external i8gemm dependency before the
   final artifact freeze; a content hash is sufficient only for this pilot.
6. Add captured multi-layer TopK routes and the current upstream Arm baseline
   before making a fused-expert paper claim.
