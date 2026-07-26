# Amazon C8i8 AVX-512 JIT-internal Bulk-MN loops

Date: 2026-07-26

## Outcome

The AVX-512 W13 and W2 JITs can now execute an N-block outer loop and all full
M12 panels in one generated body. This removes the C++ per-panel calls and
shares one frame, invariant setup, scratch lifetime, and `vzeroupper` across a
cache window.

Correctness passed, but the end-to-end gain is deliberately characterized as
small:

- conventional H4096/F512 is within roughly -0.2% to +0.2% through M256 and
  gains about 0.1% at M2048;
- short-K H64/F2048 gains roughly 0.2%--0.5% once enough W13 blocks and routes
  amortize the generated-loop setup;
- the measurable gain comes from W13; W2 is neutral and is not enabled
  automatically;
- broad automatic enablement was rejected. The C8i automatic policy uses the
  generated W13 loop only for H<=64, F>=1024, M>=48, and cooperative team
  width <=4. Explicit modes retain all broader experiments.

## Machine and method

- host: `AmazonC8i8Cores`
- CPU: Intel Xeon 6975P-C, family 6/model 173/stepping 1
- topology: 8 physical vCPUs, one thread per core
- caches reported by `lscpu`: 48 KiB L1d and 2 MiB L2 per core, 480 MiB
  shared L3
- affinity: CPU 0 for one-thread runs; CPUs 0--7 for cooperative runs
- frequency/governor telemetry: not exposed by the KVM guest, so small
  sub-percent differences must be interpreted with the rotated distributions
  rather than as a fixed-frequency claim

`benchmarks/bench_avx512_bulk_mn.py` prepares one weight set and one input set,
reuses outputs, warms every JIT key, and rotates variant order in one process.
Packing, allocation, reference calculation, and JIT warm-up are outside timed
regions. The benchmark reports the ratio of medians, ratio of means, and
same-iteration paired median ratio; the latter is diagnostic because some
sub-millisecond cases have bimodal scheduler noise.

Representative command:

```bash
taskset -c 0 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_avx512_bulk_mn.py \
  --hidden 64 --intermediate 2048 --routes 12,24,48,96,256 \
  --threads 1 --variants baseline,w13,w2,bulk_mn \
  --warmup 16 --runs 201
```

## Generated-loop design

Both generated bodies use N-block as the outer loop and M12 panel as the inner
loop. This preserves the executor's N-range ownership and cache-window
boundaries.

W13:

- one call covers every full M12 panel and every full F16 block in the
  requested window;
- the K loop advances packed A to the next M12 panel;
- an explicit intermediate-panel byte stride advances output;
- the SiLU scratch frame is allocated once for the whole generated loop.

W2:

- one call covers every full M12 panel and every full N32 block in the
  requested window;
- packed B resets for each M panel and advances after the panel loop;
- packed A advances by an explicit physical panel stride. This is necessary
  when logical F and W2 K padding differ;
- route IDs, direct BF16/FP32 output, and optional direct route weights retain
  the original epilogue semantics.

M1--11 tails, partial F16/N32 blocks, and small-M multi-N specializations still
use their existing exact kernels. `FUSED_CPP_MOE_AVX512_BULK_MN` accepts
`auto`, `baseline`, `w13`, `w2`, and `bulk_mn`.

## Correctness

After the final W2 packed-A stride fix and register-resident panel pointer:

```text
FUSED_CPP_MOE_AVX512_BULK_MN=bulk_mn:
269 passed, 1 warning in 0.94s

automatic policy:
269 passed, 1 warning in 0.94s
```

Coverage includes direct BF16 output, FP32 route output, weighted direct BF16,
M12/M24 and non-multiple tails, non-aligned H/F, independent W13/W2 controls,
cache windows, and cooperative N split. Forced baseline and generated-loop
outputs are bit-identical in the direct comparisons; maximum error against the
PyTorch reference was about `1.19e-7` at H4096/F512 and `3.73e-9` at
H64/F2048.

## One-thread results

The following values are speedup over the original per-panel calls. Values
above 1 are faster.

### Conventional H4096/F512

| M | Bulk-MN median | Bulk-MN mean |
|---:|---:|---:|
| 12 | 1.0005x | 0.9987x |
| 24 | 1.0005x | 0.9994x |
| 48 | 1.0000x | 1.0002x |
| 96 | 1.0016x | 1.0016x |
| 256 | 0.9980x | 0.9983x |
| 2048 | 1.0011x | 1.0012x |

M2048 changed from 233.673 ms to 233.414 ms by median. The variation through
M256 is as large as the measured effect, so this shape does not justify a new
automatic path.

### Short-K H64/F2048

| M | Bulk-MN median | Bulk-MN mean |
|---:|---:|---:|
| 12 | 1.0019x | 1.0005x |
| 24 | 1.0027x | 0.9987x |
| 48 | 1.0024x | 1.0001x |
| 96 | 1.0039x | 1.0042x |
| 256 | 1.0027x | 1.0028x |
| 2048 | 1.0046x | 1.0046x |

At M2048, baseline 16.290 ms became 16.216 ms by median. The isolated W13
loop was 1.0044x faster by median and 1.0043x by mean; isolated W2 was only
1.0002x/1.0003x. This is the evidence for selecting W13 alone.

An F sweep at H64/M256 showed that W13's paired median gain grew from about
0.02% at F128 to 0.10% at F256, 0.30% at F512, and 0.42% at F1024. H128
measurements were mostly at or below 0.2%. The automatic threshold therefore
uses H<=64 and F>=1024 rather than extrapolating a broad small-H rule.

## Cooperative results at H64/F2048

The table reports the mean speedup of the combined W13+W2 generated loop.

| Threads | M96 | M256 | M2048 |
|---:|---:|---:|---:|
| 2 | 1.0037x | 1.0044x | 1.0038x |
| 4 | 1.0035x | 1.0063x | 1.0031x |
| 8 | 0.9950x | 1.0073x | 1.0010x |

Eight-worker measurements are dominated by synchronization and scheduling
noise: isolated W13 ranged from 0.9962x at M96 to 1.0032x at M256 and 0.9990x
at M2048. Automatic mode consequently stops at four cooperative workers.

## Automatic-policy verification

For the selected H64/F2048 one-thread shape, `auto` matched the W13 generated
path and improved mean latency by 0.11%, 0.28%, and 0.18% at M48, M256, and
M2048 in the final 101-run verification.

For non-selected H4096/F512, `auto` matched baseline within 0.01% by mean at
M96 and M256. In the same run, forcing W13 at M256 regressed both mean and
median by about 0.20%, confirming that the conventional shape should remain on
the original per-panel calls.

## Interpretation

Putting both loops in JIT works as intended, but it does not change the
VDPBF16PS schedule, packed-weight traffic, or arithmetic intensity. At normal
hidden sizes, the M12 GEMM and SiLU work dominates each former function call,
so removing the call frame is sub-percent. W13 benefits only when each
reduction is short and there are many adjacent F16 blocks. W2 has a longer
effective reduction for the favorable shapes and its saved call overhead is
offset by the generated loop's route/output address maintenance.

The implementation is retained because it supplies a correct generated-loop
foundation for future K scheduling and prefetch work, but the policy encodes
the measured narrow win rather than treating loop fusion itself as a general
throughput optimization.
