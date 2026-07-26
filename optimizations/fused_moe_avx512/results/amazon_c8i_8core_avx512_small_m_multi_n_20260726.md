# Amazon C8i8 AVX-512 small-M multi-N kernels

Date: 2026-07-26

Machine:

- target: `AmazonC8i8Cores`
- CPU: Intel Xeon 6975P-C, CPUID family 6/model 173/stepping 1
- CPU set: `0-7`, one software worker per physical core
- worktree: `/home/ubuntu/zhangxu/fused_cpp`
- dtype: BF16 inputs/weights, FP32 accumulation

## Outcome

The AVX-512 Xbyak path now has adjacent-N kernels for exact M=1--4:

| Stage | M | Wide specialization |
| --- | --- | --- |
| W13 + SiLU-times-up | 1--3 | four adjacent F16 blocks |
| W13 + SiLU-times-up | 4 | two adjacent F16 blocks |
| W2 | 1--2 | N128, composed from four existing N32 weight blocks |
| W2 | 3--4 | N64, composed from two existing N32 weight blocks |

Each K-pair broadcasts the logical A rows once, then streams all adjacent
packed-B blocks through one temporary ZMM while keeping their accumulators
resident. The kernels consume the existing VNNI2 K-pair/N32 packed weights;
`b_block_stride_bytes` selects the next logical block, so no second weight
layout or repack is required.

Odd block counts fall back from width four to width two and then to the
existing single-block exact-M kernel. W13 F tails and W2 N tails retain the
established path. Route-FP32, direct BF16, and weighted-direct BF16 W2
epilogues are covered.

On one pinned C8i core, H=4096/F=512 improved the complete expert by
6.3%--16.1% for M=1--4. H=4096/F=2048 improved by 7.0%--15.9%.

## Automatic policy

`FUSED_CPP_MOE_AVX512_SMALL_M_MULTI_N=auto` is the default. The C8i profile
enables W13 and W2 independently; `baseline`, `w13`, `w2`, and `multi_n`
remain explicit same-binary controls. Unknown CPU profiles keep the prior
single-N path.

The one-worker C8i thresholds use padded kernel dimensions:

| Stage | M | Automatic threshold |
| --- | ---: | --- |
| W13 | 1 | `H*F >= 32K` elements |
| W13 | 2 | `H >= 256` and `H*F >= 64K` elements |
| W13 | 3 | `H*F >= 64K` elements |
| W13 | 4 | `H >= 512` and `H*F >= 256K` elements |
| W2 | 1--2 | `F >= 256` and `F*H >= 256K` elements |
| W2 | 3 | `F >= 512` and `F*H >= 2M` elements |
| W2 | 4 | `F >= 1024` and `F*H >= 8M` elements |

The executor also passes the actual per-expert cooperative team width:

- eight-worker teams retain the single-N baseline;
- at four workers, W2 requires F>=1024 and M<=3; W13 M4 requires F>=1024;
- at two workers, W2 M3 requires F>=1024 and W2 M4 requires F>=2048;
- independent experts executed by one worker retain the one-worker policy,
  even when the call has more total workers.

These guards preserve the useful single-/two-/four-worker cases while
avoiding bandwidth-saturated multi-N regressions.

## Method

The benchmark creates one tensor/weight set per shape, packs weights once,
reuses output, warms every variant, and rotates variant order inside one
process. Reference calculation, packing, first JIT generation, and allocation
are outside the measured region. Every variant is checked against the PyTorch
expert before timing.

Representative command:

```bash
ssh AmazonC8i8Cores \
  'cd /home/ubuntu/zhangxu/fused_cpp &&
   PYTHONPATH=src taskset -c 0 \
   .venv/bin/python benchmarks/bench_avx512_small_m_multi_n.py \
   --hidden 4096 --intermediate 512 --routes 1,2,3,4 \
   --threads 1 --variants baseline,w13,w2,multi_n,auto \
   --warmup 20 --runs 101'
```

For cooperative N-split validation, the same command used `--threads
2|4|8` and CPU sets `0-1`, `0-3`, or `0-7`.

## One-worker latency

Median complete-expert latency, 51 measured samples per variant:

| H/F | M | baseline | auto | speedup |
| --- | ---: | ---: | ---: | ---: |
| 4096/512 | 1 | 0.500052 ms | 0.470312 ms | 1.063x |
| 4096/512 | 2 | 0.559116 ms | 0.481496 ms | 1.161x |
| 4096/512 | 3 | 0.572163 ms | 0.508093 ms | 1.126x |
| 4096/512 | 4 | 0.604350 ms | 0.565326 ms | 1.069x |
| 4096/2048 | 1 | 1.879011 ms | 1.756432 ms | 1.070x |
| 4096/2048 | 2 | 2.076072 ms | 1.791829 ms | 1.159x |
| 4096/2048 | 3 | 2.143371 ms | 1.853564 ms | 1.156x |
| 4096/2048 | 4 | 2.247764 ms | 2.051332 ms | 1.096x |

Stage-isolated H4096/F512 measurements showed why W13 and W2 are dispatched
separately:

| M | W13-only | W2-only | both |
| ---: | ---: | ---: | ---: |
| 1 | +2.6% | +2.1% | +6.8% |
| 2 | +10.9% | +5.2% | +16.0% |
| 3 | +11.0% | +1.4% | +14.6% |
| 4 | +6.7% | approximately neutral | +6.7% |

## Dimension sweep

The held-out sweep included proportional and asymmetric shapes:

| H/F | M1 | M2 | M3 | M4 | Main automatic choice |
| --- | ---: | ---: | ---: | ---: | --- |
| 4096/32 | +11.2% | +0.6% | +0.7% | baseline | W13 only where enabled |
| 4096/64 | +16.5% | +6.1% | +3.0% | +1.5% | W13 only |
| 128/1024 | +13.8% | baseline | +0.6% | baseline | W13 M1/M3 |
| 256/512 | +21.2% | approximately neutral | +1.2% | baseline | W13 M1--3 |
| 1024/256 | +28.4% | +11.9% | +4.5% | +1.7% | W13 all, W2 M1--2 |

Forced `multi_n` regressed the very small H64/F16 and H128/F32 cases by
roughly 1.6%--4.1%; automatic mode therefore leaves them on the original
kernel.

## Cooperative N-split

Median speedup over the same single-N AVX-512 baseline after applying the
team-width policy:

| H/F | workers | M1 | M2 | M3 | M4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096/512 | 2 | 1.020x | 1.078x | 1.046x | 1.017x |
| 4096/512 | 4 | 1.012x | 1.028x | 1.051x | baseline |
| 4096/512 | 8 | baseline | baseline | baseline | baseline |
| 4096/2048 | 2 | 1.065x | 1.146x | 1.142x | 1.078x |
| 4096/2048 | 4 | 1.048x | 1.096x | 1.083x | 1.053x |
| 4096/2048 | 8 | baseline | baseline | baseline | baseline |

The rejected controls are material: at eight workers and H4096/F512,
W2-only was 6.4%--12.4% slower; at H4096/F2048, W13-only was 4.9%--13.8%
slower for M1--3. The automatic team-width guard removes those paths while
the forced modes remain available for future scheduling work.

## Correctness and validation

The added cases cover:

- M=1--4;
- direct BF16, weighted-direct BF16, and route-FP32 output;
- H/F/K/N tails with H=129/F=81;
- cache windows of three blocks;
- four-worker cooperative N split;
- stage-specific and invalid environment controls;
- C8i dimension and cooperative-width dispatch boundaries;
- generic-profile fallback.

Observed maximum absolute error versus the PyTorch reference was
`1.19e-7` at H4096/F512 and `2.38e-7` at H4096/F2048. The wider W2 kernels
can use a different FP32 reduction tree when register pressure prevents a
second accumulator set, so BF16 equivalence uses the existing numerical
tolerance rather than requiring FP32 bit identity.

Final C8i8 regression command:

```bash
PYTHONPATH=src taskset -c 0-7 .venv/bin/python -m pytest -q \
  tests/test_moe_backend_dispatch.py tests/test_moe_avx512_bf16.py
```

Result: `257 passed, 3 skipped`.

## Interpretation limits

- Thresholds are calibrated for Intel family 6/model 173. Other CPUs retain
  the baseline until a named profile is measured.
- Results are steady-state operator latency; packing and first JIT generation
  are intentionally excluded.
- Eight-worker teams are conservatively disabled. A future kernel may recover
  that regime with more independent K chains or a different cooperative
  partition, but the current wider kernels should not be forced there.
