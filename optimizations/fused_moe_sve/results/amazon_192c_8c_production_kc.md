# Production SVE Kc integration and calibration

Date: 2026-07-18

## Scope

This run promoted Kchunk-major packed B and the generic M12/M8/M4/M2 Kc
kernels to the production SVE fused-MoE path. M1 uses the predicated M2 body.
Normal, 2D, scheduled, async, and vLLM-staged W13/W2 dispatches use the same
layout. FP32 and BF16 direct-route W2 stores are covered. NEON and legacy SVE
symbols were not changed.

The production selector is:

```text
bytes_per_K = 2 * (12 + n_tile)
Kc = align_down_8(min(K, 0.49 * L1D_bytes / bytes_per_K))
```

Both hosts report 64 KiB private L1D. This gives Kc=800 for SVE128
(`n_tile=8`) and Kc=568 for SVE256 (`n_tile=16`). Gain below means
`baseline_time / Kc_time - 1`.

## Isolated calibration

The benchmark uses `M=Mr,K=4096,N=512`, one distinct cold 4 MiB B matrix per
call, 51 calls per process, and repeated fixed-Kc processes. FP32 partial C is
reused within a process but is not shared between variants.

| Host | Kc | M12 | M8 | M4 | M2 | M1 |
|---|---:|---:|---:|---:|---:|---:|
| AmazonC5192Cores, CPU48 | 800 | +13.21% | +8.63% | +6.30% | +6.18% | +6.38% |
| AmazonECS8Cores, CPU0 | 568 | +2.23% | -1.48% | +2.35% | +0.95% | +0.58% |

The 8-core fine sweep covered Kc=544/552/560/568/576. Kc=568 maximized the
worst M12/M8 result; the other small-M kernels were then checked at that point.
The interval that maps to Kc=568 on SVE256 overlaps the interval that maps to
Kc=800 on SVE128. A 49% L1 budget lies in that overlap.

## Correctness

- Cross-assembly syntax check passed with Clang AArch64 SVE+BF16 target.
- The standalone M12 suite checked all 44 variants bit exactly.
- On AmazonC5192Cores, the complete fused-MoE and backend-dispatch selection
  passed: 92 tests passed and 4 were skipped. The focused production-Kc test
  passed all 12 combinations of M=1/2/4/8/12/13 and FP32/BF16 direct-route
  storage.
- On AmazonECS8Cores, the same H=F=1024 M=1/2/4/8/12/13 matrix passed for FP32
  and BF16 direct-route storage. In all 12 combinations, 1T and 4T were bit
  exact and both matched the naive reference tolerance. The host's older
  Python test tree cannot collect the full suite because it lacks the unrelated
  `fused_moe_bf16_tiled_vllm_staged` export, so this check was run as an
  independent script against the rebuilt native extension.
- With `FUSED_CPP_MOE_SVE=0`, both hosts selected the NEON backend, prepared
  its independent packed layout, and completed a fused expert call matching
  the naive reference tolerance.
- The direct-route epilogue initially reused `x22`, which the Kc body reserves
  for `k_start`; a multi-chunk test exposed an infinite loop. Direct stores now
  use `x8`, and the permanent test covers both route storage modes.

## Production split-W13 E2E

Each value is the median of three independent process medians. Every process
uses five warmups and 15 measured calls. `Kc=one` sets
`FUSED_CPP_MOE_SVE_KC=1048576`, so K is one chunk while retaining the generic
dispatch. The default column uses the 49% selector.

| Host/configuration | Route | One chunk | Default Kc | Throughput gain |
|---|---:|---:|---:|---:|
| 8 cores, 2 experts x 4T | 12 | 0.402 ms | 0.394 ms | +2.03% |
| 8 cores, 2 experts x 4T | 2040 | 37.216 ms | 35.952 ms | +3.52% |
| 192-core host NUMA0, 24 experts x 4T | 12 | 0.599 ms | 0.662 ms | -9.52% |
| 192-core host NUMA0, 24 experts x 4T | 2040 | 30.508 ms | 30.895 ms | -1.25% |

The isolated 192-core-host result does not survive a saturated expert wave.
With 24 one-thread experts, route=12 changed from 0.851 to 0.855 ms and
route=2040 from 119.829 to 121.574 ms. One isolated four-thread expert still
improved from 0.171 to 0.168 ms at route=12, but regressed from 21.677 to
21.977 ms at route=2040.

A production sweep on 24 x 4T tested Kc=800/1024/1536/2048/4096. Larger Kc
reduced the partial-C cost, but no split point beat Kc=4096 at route=2040;
Kc=2048 was approximately tied with one chunk at route=12. This is an
applicability boundary, not a different L1 optimum: when many cold experts
saturate memory traffic, improving A-side L1 residency is no longer the
limiter, while every non-final K chunk still adds FP32 partial stores/loads.

A final-build representative rerun at route=2040 reproduced the direction:
the 8-core 2-expert x 4T case changed from 37.208 ms to 35.862 ms (+3.75%
throughput), while the 192-core-host NUMA0 24-expert x 4T wave changed from
30.529 ms to 30.865 ms (-1.09% throughput). These are single process medians
with the same five warmups and 15 measured calls, and are a post-refactor smoke
check rather than replacements for the three-process table above.

## Reproduction

Isolated M12:

```bash
python3 optimizations/fused_moe_sve/benchmarks/run_m12_streaming_b.py \
  --cpu 48 --numa-node 0 --m 12 --k 4096 --n 512 \
  --variants baseline_ld1h,kblock_packed_800 \
  --warmup 5 --runs 51 --repeat 9 --cold-tail-mib 192
```

Production 192-core-host NUMA0 default:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  env -u FUSED_CPP_MOE_SVE_KC OMP_NUM_THREADS=96 PYTHONPATH=src \
  .venv/bin/python optimizations/fused_moe_sve/benchmarks/bench_weight_windows.py \
  --experts 24 --routes 2040 --hidden 4096 --intermediate 512 \
  --threads-per-expert 4 --cpu-start 0 --window-mib 0 \
  --warmup 5 --runs 15 --seed 20260718
```

Use `FUSED_CPP_MOE_SVE_KC=1048576` for the one-chunk control. Packed weights
must be prepared in a process with the same Kc configuration used for compute.
