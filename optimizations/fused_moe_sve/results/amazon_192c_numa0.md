# Amazon 192-core NUMA0 results

## Environment

- CPU: 192 Neoverse-V3 cores, two NUMA nodes.
- Binding: NUMA node 0, CPUs 0-95, memory node 0.
- Runtime SVE GEMM N tile: 8 BF16 columns.
- Shape: M=2040, with ten 204-row epochs.
- Elastic plan: half of M at `low_threads`, then half at `threads`.
- Timing: 3 warmups, median of 9 measured rounds, 4 rotating buffer copies.
- Build: GCC 15.2, `-O2 -march=armv8.6-a+sve+bf16+i8mm`.

Command:

```bash
numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  python3 optimizations/fused_moe_sve/benchmarks/run_sweep.py \
  --epoch-rows 204 --warmup 3 --iters 9 --copies 4 \
  --output /tmp/elastic_phase_epoch204.jsonl
```

All W13, W2 FP32, and W2 BF16 correctness checks were bitwise exact against
the static N-split result.

## Key results

`elastic overhead` compares the measured low-to-high run with
`0.5 * static_low + 0.5 * static_high`. `phase fixed overhead` compares a
fixed-width phase claim with the epoch-fragmented static mapping. The strict
column is the intentionally pessimistic implementation with a barrier at every
M epoch.

| Case | Resize | Static low ms | Static high ms | Elastic ms | Elastic overhead | Phase fixed overhead | Strict overhead |
|---|---:|---:|---:|---:|---:|---:|---:|
| TP4 W13 split window, K4096 N512 | 32 -> 64 | 1.4984 | 1.7660 | 1.6777 | +2.79% | +1.59% | +10.40% |
| TP4 W13 full, K4096 N1024 | 48 -> 96 | 2.0527 | 1.9335 | 2.0127 | +0.99% | -1.83% | +17.74% |
| TP4 W2 BF16, K512 N4096 | 48 -> 96 | 0.7098 | 0.4586 | 0.6102 | +4.44% | -2.52% | +70.87% |
| EP4 W13 split window, K4096 N2048 | 48 -> 96 | 2.8253 | 2.1508 | 2.5701 | +3.30% | +2.00% | +24.20% |
| EP4 W13 full, K4096 N4096 | 48 -> 96 | 4.8126 | 3.1810 | 4.1446 | +3.70% | -1.43% | +21.07% |
| EP4 W2 BF16, K2048 N4096 | 48 -> 96 | 2.3942 | 1.5497 | 2.0519 | +4.06% | -1.31% | +29.56% |

The implied one-time resize cost is about 20-148 microseconds across these
shapes. Fixed-width phase claiming is within roughly 2.5% of the static kernel,
so atomic dispatch is not the limiting factor after N-lane affinity is kept.

## Conclusions

- A dynamic SVE fused GEMM is viable at coarse phase boundaries. Repartition N
  only when the available thread count changes.
- One logical N lane must remain on the same worker for the whole constant-width
  phase. Re-claiming a different lane every M epoch loses packed-B affinity.
- Do not place a barrier at every M epoch. At 96 threads the strict design costs
  21-30% on EP4 and 71% on the short TP4 W2 kernel.
- More threads are not always useful. For the TP4 split W13 window, static 64T
  is 17.9% slower than static 32T, so a planner should not expand this kernel to
  all available lanes.
- The experiment measures kernel scheduling only. It does not include the
  system-level benefit of releasing low-phase threads to another expert. The
  extra resident workers wait during the low phase; running useful work on them
  would add the real cross-expert cache and bandwidth contention.
