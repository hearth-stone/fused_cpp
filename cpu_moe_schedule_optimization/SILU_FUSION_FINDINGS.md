# Fused w13-GEMM + SiLU-and-Mul (pure-asm epilogue) — Findings

## What was built
The SiLU-and-mul activation is fused into the w13 bf16 GEMM's **store epilogue**.
After the bfmmla kernel computes an 8×8 tile (interleaved as 4 gate + 4 up
columns), it computes `silu(gate)*up` at the register level and writes the bf16
`intermediate[M, F]` directly — eliminating the `gate_up[M, 2F]` fp32 buffer and
the separate activation pass.

- **Prepack**: `prepare_fused_moe_bf16_tiled_weights(w13, w2, fuse_silu=True)`
  packs w13 in an interleaved layout: per 8-col N-block `[g(4b..4b+3) | u(4b..4b+3)]`
  (gate feature `f` → packed col `8*(f/4)+(f%4)`, up → `+4`). Requires `F % 8 == 0`.
- **Kernel**: `bf16gemm_k_ld_silu_poly{4,5,6}` (+ `_m{1,2,4}` tails) in
  `i8gemm/lib/bf16gemm_k.S`, generated via the existing `GEMM_BODY` macro with a
  new fused store macro (`ldc_shift=1`, `c_adv=8`). The exp is a NEON-asm port of
  `vexpq_f32_poly_impl<deg>` (range reduction + Horner + `2^n` bit-trick); silu via
  `num=g*u`, `denom=1+exp(-g)`, `out=num/denom` (fdiv).
- **Integration**: opt-in via the prepared-weights `fused_silu` flag +
  `silu_poly_degree` (default 5). Wired into the **default** per-expert path of
  `fused_moe_bf16_tiled` (activation must be `silu`); the legacy w13+activation
  path stays as fallback and numerical reference. Hierarchical/scheduled paths
  are left on the legacy path for now.

## Correctness
- 118 fused tests (kernel + e2e) pass on macOS arm64 **and** AWS Graviton (ELF
  `adrp/:lo12:` const-table path validated on real hardware).
- 573 existing MoE regression tests unchanged.
- Fused (poly5) vs baseline (`std::exp`) e2e diff is ≤ 1 bf16 ULP at the output
  magnitude (e.g. `maxdiff=0.25` at values ~32-64), i.e. pure bf16 rounding.

## Benchmark (AWS Graviton, 8 cores, `OMP_PROC_BIND=FALSE taskset -c 0-7`)
H=4096, F=512, E=8, top_k=2; median ms; speedup = baseline / fused(poly5):

| tokens | threads | base_ms | fused_p5 | speedup |
|---|---|---|---|---|
| 8    | 1 | 3.848  | 3.790  | 1.02 |
| 64   | 1 | 12.175 | 11.941 | 1.02 |
| 256  | 1 | 49.222 | 48.201 | 1.02 |
| 1024 | 1 | 182.01 | 178.20 | 1.02 |
| 1024 | 4 | 74.31  | 73.55  | 1.01 |
| 1024 | 8 | 63.14  | 62.59  | 1.01 |

poly4/5/6 are within noise of each other (the exp is cheap relative to the GEMM).

## Conclusion
The fused path delivers a **~1-2% end-to-end speedup**, exactly as the earlier
stage-breakdown predicted: the SiLU-and-mul activation is only ~2% of e2e and the
two GEMMs dominate (~85%). The win comes from removing the activation pass and the
`gate_up[M,2F]` fp32 buffer traffic. In the default (1-thread-per-expert) path
there is no inter-stage barrier to remove, so the gain is purely activation +
buffer traffic.

**This is a polish/building-block, not a major lever** — consistent with the
prior finding that post-w13 fusion is a few-percent item while the GEMMs (and MoE
scheduling) are where the real time goes. The larger remaining opportunity for the
fused epilogue is the **team (N-split) path**, where fusing would also drop the
w13→activation barrier (up to ~6-19% of the small-M/high-thread regime per the gap
analysis) — deferred, along with w13 bias and the scheduled/async paths.

## How to reproduce
```bash
# build (macOS or AWS Graviton)
python setup.py build_ext --inplace
# correctness
pytest -q tests/test_moe_fused_silu.py tests/test_moe_fused_silu_e2e.py
# benchmark (pin the process)
OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE taskset -c 0-7 \
  python cpu_moe_schedule_optimization/benchmarks/bench_fused_silu.py
```

---

# Update: N-split single-expert multithreading for the fused path

The fused path was previously wired only into the default per-expert scheduler
(one thread per expert), so when **active experts < cores** the extra cores sat
idle. This update lets a **team of G threads cooperate on a single expert** via
N-split (column split of the interleaved 2F weight), reusing the existing
hierarchical team infrastructure. The fused w13 stage computes each member's
disjoint feature-column slice and writes `intermediate` directly, collapsing the
w13 GEMM + separate activation into one stage and dropping one barrier.

- **Fused w13 is forced to kN** (N-split); the w2 stage keeps its existing
  dynamic split (kN in ~all MoE shapes). No asm / w13-kernel changes.
- Opt-in via the same `fuse_silu=True` weights + the hierarchical N-split env
  (`FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=1`, `..._N_SPLIT_CORE_BASES`,
  `..._N_SPLIT_GROUPS_PER_PARTITION`).
- Correctness: `tests/test_moe_fused_silu_nsplit.py` — the N-split assembly is
  **bit-identical** to the single-thread fused kernel (108 cases), and the
  threaded end-to-end MoE matches the non-fused baseline within tolerance (9
  cases). Green on macOS arm64 and AWS Graviton (573 regression + 235 fused).

## Benchmark (AWS Graviton, 8 cores, `OMP_PROC_BIND=FALSE taskset -c 0-7`)
H=4096, F=512. One group of G threads over each active expert (E = 8/G experts).
Median ms. `default_fused` = the previous 1-thread-per-expert fused path.

| rows/expert | G | nsplit_fused | nsplit_legacy | default_fused | fused/legacy | **nsplit vs default** |
|---|---|---|---|---|---|---|
| 128  | 8 | 2.221  | 2.241  | 25.282  | 1.009 | **11.4×** |
| 512  | 8 | 29.140 | 29.507 | 104.392 | 1.013 | **3.6×** |
| 1024 | 8 | 58.436 | 58.333 | 204.585 | 1.00  | **3.5×** |
| 1    | 8 | 0.236  | 0.248  | 0.574   | 1.049 | **2.4×** |
| 128  | 4 | 4.144  | 4.216  | 27.232  | 1.017 | **6.6×** |
| 1024 | 2 | 137.1  | 138.5  | 227.4   | 1.010 | **1.7×** |

## Conclusion
1. **Single-expert N-split multithreading is the real win** — it lets the fused
   path use otherwise-idle cores in the few-active-experts regime, giving
   **~2.4×–11×** over the previous one-thread-per-expert fused path. This is the
   valuable lever (using more cores), not the fusion arithmetic itself.
2. **Fusion on top of N-split** adds a further **~1-6%** (`fused/legacy`),
   largest at tiny M where the removed w13→activation barrier + activation pass
   is a bigger fraction. Consistent with the earlier finding that the activation
   is a small slice of e2e.
3. Under N-split the A-pack is duplicated G× (each team member repacks the full
   `M×H` A into its own buffer). This affects **both** fused and legacy equally
   (same team infra), so it cancels in `fused/legacy` — but it is a real absolute
   cost. Eliminating it (pre-pack A once into a shared read-only buffer) is the
   next opportunity, and it is specifically the N-split regime where it matters.

## How to reproduce (N-split)
```bash
OMP_NUM_THREADS=1 OMP_PROC_BIND=FALSE taskset -c 0-7 \
  python cpu_moe_schedule_optimization/benchmarks/bench_fused_silu_nsplit.py
# correctness
pytest -q tests/test_moe_fused_silu_nsplit.py
```
