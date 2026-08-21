# SVE FEXPA plus poly2 exponential evaluator

## Question

Can SVE `FEXPA` plus a degree-2 residual polynomial replace the current
degree-4/5/6 Horner exponential in the fused SiLU/SwiGLU epilogue with better
performance and acceptable accuracy?

## Implementation

The experiment follows Arm's documented FEXPA construction:

1. `z = x * inv_ln2 + shift`, where the shift encodes the exponent and six-bit
   table index expected by FEXPA.
2. Recover `k = z - shift` and compute the Cody-Waite residual
   `r = x - k*ln2_hi - k*ln2_lo`.
3. Compute `scale = FEXPA(z)`.
4. Approximate `exp(r)-1` as `r*(c0+c1*r)`.
5. Return `scale + scale*poly`.

Constants are from the Arm Learning Path implementation:
<https://learn.arm.com/learning-paths/servers-and-cloud-computing/fexpa/fexpa/>.
The reference is libm `std::exp`, not another polynomial implementation.

The Lab assembly holds 16 live Z registers for M8 and 24 for M12, matching the
register pressure around the fused W13 epilogue. It measures pure `exp(-gate)`,
complete `gate*up/(1+exp(-gate))`, and limit-10 clamped SwiGLU. The benchmark
does not include GEMM instructions, so complete W13 and expert E2E dilution
remain a follow-up.

## Configuration

- Host: `Arm-codex-internal`, SVE vector length 256 bits
- CPU placements: CPU 0 for 1T and CPUs 0-79 for 80T
- Performance inputs: gate/up uniformly distributed in `[-6,6]`
- Accuracy: 262,144 values per `(M, scope, evaluator)`
- Accuracy domains: `[-10,10]` for exp/SiLU and `[-20,20]` before clamping
- Timing: 30 warmups, 51 medians, 4000 inner calls
- Baseline: current degree-5 Horner evaluator with exact SVE division

## 80-thread performance

Times are nanoseconds per simultaneous M8/M12 wave.

| M | Scope | Poly4 | Poly5 | Poly6 | FEXPA+poly2 | Gain vs poly5 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | exp | 51.498 | 55.492 | 60.022 | **47.982** | **15.65%** |
| 8 | SiLU | 61.274 | 65.744 | 71.287 | **55.730** | **17.97%** |
| 8 | clamped SiLU | 75.758 | 81.539 | 87.976 | **63.777** | **27.85%** |
| 12 | exp | 78.261 | 84.043 | 88.990 | **73.016** | **15.10%** |
| 12 | SiLU | 89.467 | 94.771 | 104.547 | **78.797** | **20.27%** |
| 12 | clamped SiLU | 115.752 | 124.037 | 132.501 | **92.745** | **33.74%** |

FEXPA+poly2 also beats poly4 by 7.2-7.3% for exp, 10.0-13.5% for SiLU,
and 18.8-24.8% for clamped SiLU.

## Single-thread control

| M | Scope | Poly5 ns | FEXPA+poly2 ns | Gain vs poly5 |
| ---: | --- | ---: | ---: | ---: |
| 8 | exp | 56.505 | 47.028 | 20.15% |
| 8 | SiLU | 64.790 | 54.777 | 18.28% |
| 8 | clamped SiLU | 78.022 | 64.254 | 21.43% |
| 12 | exp | 84.460 | 72.777 | 16.05% |
| 12 | SiLU | 96.738 | 82.016 | 17.95% |
| 12 | clamped SiLU | 122.547 | 96.738 | 26.68% |

The same ordering at 1T and 80T shows that the result is not produced by
multithread timing or shared-resource interference.

## Accuracy against libm

The table reports the worst result across M8 and M12.

| Scope | Evaluator | Max relative | RMS absolute | Max exp ULP | Max BF16 mismatch |
| --- | --- | ---: | ---: | ---: | ---: |
| exp | poly4 | 5.572e-5 | 3.595e-2 | 661 | 0.1362% |
| exp | poly5 | 3.287e-6 | 1.934e-3 | 39 | 0.00725% |
| exp | poly6 | 2.529e-7 | 1.866e-4 | 3 | 0.000763% |
| exp | **FEXPA+poly2** | **1.192e-7** | **1.697e-4** | **1** | **0.000763%** |
| SiLU | poly4 | 5.568e-5 | 8.077e-6 | - | 0.0740% |
| SiLU | poly5 | 3.376e-6 | 4.987e-7 | - | 0.00191% |
| SiLU | poly6 | 2.643e-7 | 8.947e-8 | - | 0% |
| SiLU | **FEXPA+poly2** | **2.289e-7** | **8.553e-8** | - | **0.000381%** |
| clamped SiLU | poly4 | 5.570e-5 | 8.186e-6 | - | 0.0732% |
| clamped SiLU | poly5 | 3.424e-6 | 5.042e-7 | - | 0.00420% |
| clamped SiLU | poly6 | 3.294e-7 | 8.102e-8 | - | 0.000381% |
| clamped SiLU | **FEXPA+poly2** | **3.382e-7** | **8.810e-8** | - | **0%** |

FEXPA+poly2 is substantially more accurate than poly4 and poly5. It is broadly
poly6-class: better for exp and ordinary SiLU, with a slightly larger worst
relative error than poly6 for M8 clamped SiLU (`3.382e-7` versus `3.159e-7`),
while producing zero BF16-rounded mismatches in that clamped test.

A separate 262,144-value `exp(-gate)` sweep over the complete finite production
range `gate in [-87,87]` also remained stable:

| Evaluator | Max relative | Max ULP | BF16 mismatch |
| --- | ---: | ---: | ---: |
| poly4 | 5.563e-5 | 663 | 0.1404% |
| poly5 | 3.456e-6 | 41 | 0.00916% |
| poly6 | 3.577e-7 | 5 | 0.00343% |
| **FEXPA+poly2** | **1.192e-7** | **1** | **0.000763%** |

This rules out an accuracy loss at the shift/FEXPA exponent-encoding boundary.

## Instruction audit

The target binary disassembly contains one `fexpa z2.s, z1.s` per live row,
followed by the two polynomial FMAs/multiply and the existing exact `fdiv` for
SiLU. There are no helper calls or vector spills inside the evaluator body;
M12 saves only the ABI-required low 64 bits of `z8-z15`, as in the existing
benchmark.

## Decision

The isolated experiment passes its evaluator gate. FEXPA+poly2 is faster than
poly4/5/6 in every tested M8/M12 scope and provides poly6-class or better
accuracy against libm.

## Production integration

The SVE JIT, static-assembly and legacy intrinsic SiLU epilogues were
subsequently changed to the same FEXPA evaluator. The public degree selectors
4/5/6 remain accepted for API compatibility, but share one canonical JIT cache
entry on SVE. NEON and x86 retain their ISA-specific implementations.

Focused validation on `Arm-codex-internal`, SVE256, CPUs 0-79 covered JIT and
static assembly, the intrinsic packA path, normal/scheduled/async bridges,
M1-13 and longer tails, and all three compatibility selectors: 356 tests
passed, plus the dedicated cross-selector identity test. GNU `objdump`
confirms `fexpa` in both static-assembly and intrinsic
objects; the generated JIT uses Xbyak's typed `fexpa` encoder.

The same production sources were then rebuilt on `AmazonECS8Cores` and tested
on CPUs 0-7. The selector, JIT/static exact-M and intrinsic packA subset passed
239 tests, providing an independent ARM SVE holdout for dispatch and execution.

An alternating old-poly5 versus FEXPA comparison used H7168/F768, BF16 packed
weights, 10 warmups and 31 medians for M12, and 5 warmups and 21 medians for
M2040. The benchmark helper's pure-W13 scope uses its fixed 2-thread team.

| M | Scope | Old poly5 ms | FEXPA ms | Change |
| ---: | --- | ---: | ---: | ---: |
| 12 | standard W13 | 3.145 | 3.151 | -0.20% |
| 12 | standard expert | 5.046 | 5.020 | +0.50% |
| 12 | clamped expert | 5.060 | 4.983 | +1.53% |
| 2040 | standard W13 | 533.773 | 534.913 | -0.21% |
| 2040 | standard expert | 870.731 | 870.728 | +0.00% |
| 2040 | clamped expert | 871.737 | 865.664 | +0.70% |

The isolated epilogue gain is mostly diluted by GEMM, so the original 2 percent
complete-W13 speed gate was not met. The production switch was nevertheless
made by explicit product decision: it materially improves exp accuracy,
eliminates three SVE evaluator variants and redundant JIT cache entries, and
shows no complete-expert regression above the observed noise floor.
