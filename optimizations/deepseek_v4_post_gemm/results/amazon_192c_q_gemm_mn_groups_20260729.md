# DeepSeek V4 Post-GEMM Q GEMM MN Groups

## Configuration

- Date: 2026-07-29
- Machine: `AmazonC5192Cores`, 192-core AArch64 host
- NUMA binding: node 0, CPUs `0-95`
- Shape: C4A TP4, Main Q and Indexer Q GEMMs both use
  `(M, 1024) x (1024, 8192)`
- Dtype: BF16 input, packed BF16 weights and BF16 outputs
- Build: effective `-O2`, OpenMP enabled
- Baseline: shared Q pool with one full-N cursor per GEMM
- Candidate: owner-first `(GEMM, N stripe, M8 panel)` task groups

Each packed weight is 16 MiB. With 32 total groups, each GEMM receives 16
groups and each packed-B stripe is 1 MiB. A worker remains on its preferred
stripe while M8 panels are available, then steals from another group. Packed
weights, GEMM kernels, output layouts and postprocessing are unchanged.

The current kernel packs A inside every N-range call, so increasing the group
count repeats A packing. This experiment intentionally keeps that cost visible;
sharing one prepacked QR tensor remains a separate optimization.

## Correctness

```text
NEON/default: 16 passed in 0.32s
SVE:          16 passed in 0.28s
```

The focused MN regression uses `M=25`, seven threads and 128 output columns per
GEMM. It compares two total groups with eight requested groups and checks Q,
Top-K, SWA cache and both compressor state caches. The larger shape ensures
that both NEON and SVE execute real N stripes.

## N-Group Sweep

Five warmups and 31 measured calls were used at `M=2048`, 96 threads. Group
count is total across the two equal-size Q GEMMs.

| Total groups | B stripe per GEMM | NEON median (ms) | SVE median (ms) |
|---:|---:|---:|---:|
| 2 | 16.000 MiB | 9.840 | 10.044 |
| 4 | 8.000 MiB | 8.691 | 9.022 |
| 8 | 4.000 MiB | 7.656 | 7.798 |
| 12 | 2.667 MiB | 6.628 | 6.476 |
| 16 | 2.000 MiB | 4.890 | 5.119 |
| 24 | 1.333 MiB | 3.977 | **4.609** |
| 32 | 1.000 MiB | **3.861** | 4.651 |
| 48 | 0.667 MiB | 3.879 | 4.774 |
| 64 | 0.500 MiB | 4.691 | 5.638 |
| 96 | 0.333 MiB | 3.911 | 5.138 |

Relative to two groups, the best NEON point is 60.8% faster and the best SVE
point is 54.1% faster. The useful window is about 1.0-1.33 MiB. Group count
also affects worker balance: 64 groups divide 96 workers unevenly and regress,
while 24, 32, 48 and 96 divide evenly.

## M Sweep

The following candidate policy uses 48 groups for `M=12`, where two M8 panels
need at least 48 groups to expose 96 tasks, and 32 groups for the remaining
points. Each point uses three warmups and 15 measured calls.

| Backend | M | Sequential baseline (ms) | Candidate (ms) | Reduction |
|---|---:|---:|---:|---:|
| NEON | 12 | 1.222 | 0.162 | 86.7% |
| NEON | 48 | 1.256 | 0.246 | 80.4% |
| NEON | 192 | 1.452 | 0.494 | 66.0% |
| NEON | 768 | 3.665 | 1.591 | 56.6% |
| NEON | 2048 | 10.172 | 3.861 | 62.0% |
| SVE | 12 | 1.439 | 0.194 | 86.6% |
| SVE | 48 | 1.462 | 0.254 | 82.6% |
| SVE | 192 | 1.566 | 0.601 | 61.6% |
| SVE | 768 | 3.765 | 1.876 | 50.2% |
| SVE | 2048 | 10.442 | 4.651 | 55.5% |

MN groups remove the old small-M shared-pool failure mode. The two-cursor
implementation exposed only `2 * ceil(M/8)` tasks; N groups multiply that task
space without changing output ownership.

## Profile Attribution

At `M=2048`, 96 threads:

| Backend | Groups | Shared Q GEMM (ms) | Full stage profile (ms) |
|---|---:|---:|---:|
| NEON | 2 | 8.391 | 9.482 |
| NEON | 32 | 2.842 | 3.906 |
| SVE | 2 | 8.834 | 9.903 |
| SVE | 24 | 4.476 | 5.652 |

The NEON Q GEMM region falls by 66.1%; SVE falls by 49.3%. Other stage work
remains approximately 1 ms, so the end-to-end benefit is directly attributable
to the two Q GEMMs rather than postprocessing changes.

## Low-Thread Boundary

Forced 32-group NEON measurements show that task/cursor overhead dominates at
one or two threads, while four threads are already beneficial:

| M | Threads | Sequential (ms) | MN groups (ms) | Delta |
|---:|---:|---:|---:|---:|
| 12 | 1 | 1.902 | 1.863 | -2.0% |
| 12 | 2 | 1.181 | 1.353 | +14.5% |
| 12 | 4 | 1.068 | 0.730 | -31.6% |
| 48 | 1 | 6.397 | 6.966 | +8.9% |
| 48 | 2 | 3.576 | 3.806 | +6.4% |
| 48 | 4 | 2.375 | 1.903 | -19.9% |

This is evidence for a future automatic policy, not a default change in this
patch.

## Selection And Reproduction

The production default remains two total groups. Set
`FUSED_CPP_POST_GEMM_N_GROUPS` to an integer at least two to select the
candidate. `FUSED_CPP_POST_GEMM_SHARED_Q_POOL=1` forces the shared pool when
the existing automatic pool policy would not select it.

```bash
PYTHONPATH=src FUSED_CPP_POST_GEMM_BACKEND=neon \
OMP_NUM_THREADS=96 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores \
numactl --physcpubind=0-95 --membind=0 \
.venv/bin/python tests/bench_deepseek_v4_post_gemm_stage.py \
  --m 2048 --threads 96 --warmup 5 --runs 31 \
  --schedule m8 --q-pool shared --n-groups 32 --profile
```

Use `--n-groups 2` for the shared-pool baseline.
