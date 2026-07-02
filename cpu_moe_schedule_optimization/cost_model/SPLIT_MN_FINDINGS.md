# MoE GEMM M-split vs N-split — empirical study & rule

**Question.** For the cooperative team GEMM (`team_gemm`, group_size ≥ 2 threads
splitting one expert), should we split the M (rows) or N (columns) dimension?

**Setup.** Isolated team GEMM timing via `fused_moe_bench_team_gemm` (weights
packed / A padded once; only the `team_gemm` loop is timed, inside one pooled
job). Two MoE GEMM shapes, for hidden `H` and per-rank FFN `F`:

| stage | GEMM | K | N |
| --- | --- | --- | --- |
| w13 | `A[M,H] · W13[2F,H]^T → [M,2F]` | H | 2F |
| w2  | `A[M,F] · W2[H,F]^T → [M,H]`   | F | H |

Machine: **AWS `AmazonECS8Cores` (Graviton, aarch64, 8 cores)**, pinned
`taskset -c 0-7`. Shape `H=4096, F=512` → w13 `K=4096,N=1024`, w2 `K=512,N=4096`.
M ∈ {1..2048}, group_size ∈ {1,2,4,8}. Data:
`cost_model/profiles/split_mn_aws.{csv,json}`.

> **Measurement pitfall (important).** Run the multi-thread pool benchmark with
> `OMP_PROC_BIND=FALSE`. `OMP_PROC_BIND=close` (the single-thread microbench
> hygiene) mispins the `std::thread` team onto one core and produces a false
> "no speedup / M≈N" result. The correct-pinning command:
> ```
> OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE taskset -c 0-7 \
>   .venv/bin/python cpu_moe_schedule_optimization/benchmarks/bench_split_mn.py \
>   --threads 1,2,4,8 --warmup 5 --runs 40 --output-csv ... --output-json ...
> ```

## Result

**N-split wins at every (M, group_size) for both shapes. M-split won 0 / 104.**

| stage | T | N-split wins | M-split wins | median N/M | N-split scaling vs T=1 |
| --- | --- | --- | --- | --- | --- |
| w13 | 2 | 13/13 | 0 | 0.815 | 1.96× |
| w13 | 4 | 13/13 | 0 | 0.851 | 3.28× |
| w13 | 8 | 13/13 | 0 | 0.747 | 5.72× |
| w2  | 2 | 8/13 (rest tie) | 0 | 0.986 | 1.81× |
| w2  | 4 | 13/13 | 0 | 0.935 | 3.34× |
| w2  | 8 | 13/13 | 0 | 0.743 | 5.92× |

- Large M (≥64): N is still faster — w13 N/M mean 0.84 (T=2) … 0.93 (T=8);
  w2 0.94–0.99. N-split **never** loses.
- Small M: N-split is dramatically better (M=1, T=8: N/M ≈ 0.24, ~4× faster),
  because M-split has only `M/8` row-blocks to hand out and starves the team.
- Single-thread (group_size=1) is split-independent, ~163 GFLOP/s (w13) /
  ~172 GFLOP/s (w2); N-split scales to ~975 / ~975 GFLOP/s at 8 threads.

## Why

An **M-split** gives each thread a row range but the FULL N, so every thread
streams the **entire weight matrix** → `group_size×` weight bandwidth. An
**N-split** partitions the weight columns across the team (each thread reads its
weight slice + the full, smaller A). For these weight-heavy MoE shapes on a
bandwidth-limited machine, N-split's lower weight traffic wins at all M.

## Rule (wired into `default_split_selector`)

```
if group_size <= 1:           N   # split irrelevant
elif N/kKernelTile >= group_size:   N   # N-split can fill the team -> best
elif M/kKernelTile > N/kKernelTile: M   # N can't fill the team but M can
else:                          N
```

For MoE w13/w2, `N/8` = 128 or 512 ≫ group_size, so the rule is effectively
**always N-split** — matching the data. The `M`-split branch is a safety
fallback for hypothetical shapes with tiny N.

## Large-M crossover search — there is none

Coarse sweep to find the M where M-split would overtake N-split
(`profiles/split_mn_aws_largeM.{csv,json}`, T ∈ {4,8}, M up to 65536):

| stage (N) | M/N range | winner | n/m |
| --- | --- | --- | --- |
| w13 (1024) | 1× → **64×** (M≤65536) | N at every point | 0.89–0.97 |
| w2 (4096)  | 0.2× → **16×** (M≤65536) | N at every point | 0.94–0.96 |

**No crossover exists up to M/N = 64.** This refutes the naive byte-traffic
model (which predicts a crossover near M ≈ N once activations outweigh weights).
Reason: at large M the GEMM is **compute-bound** — both splits plateau near the
8-core ceiling (~975 GFLOP/s N vs ~906 M for w13, T=8), and N-split's kernel
holds a **stable ~5–10% efficiency edge** regardless of M (contiguous packed-B
column slices vs M-split's need for every thread to traverse the full packed B).
So the split decision is *not* a small-vs-large-M tradeoff here:
- small M: N wins **decisively** (parallelism — M-split starves threads),
- large M: N wins **modestly but consistently** (kernel efficiency).

**Practical takeaway:** for these shapes on this machine, **always N-split** —
there is no realistic (or even absurd, 64× N) M at which M-split pays off.

**Scope / caveat.** Measured on one machine (Graviton 8-core) and two shapes,
M up to 65536. The `M`-split branch in the selector remains only for shapes with
`N/8 < group_size` (tiny N, which does not occur for MoE w13/w2). For very
different N or hardware, re-measure with `bench_split_mn.py`; the selector is the
single seam to update.
