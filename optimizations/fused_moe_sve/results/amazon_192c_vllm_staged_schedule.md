# vLLM-style staged MoE scheduling on Amazon C5 192 cores

Date: 2026-07-17

## Method

The benchmark used NUMA node 0 and CPUs `0-95` on `AmazonC5192Cores`. Both
variants used the same packed BF16 weights, SVE M12/M8/M4/M2/M1 assembly GEMM
bodies, fused W13 SiLU/packC epilogue, FP32 direct route store, and SVE U1
weighted merge. The fixed-team async control used two split-W13 ranges and had
ready-token merge disabled. The candidate used a global W13 N-range queue, a
full W13/W2 stage barrier, and a global W2 N-range queue.

All runs used `tokens=2048`, `H=4096`, `F=512`, `E=256`, `top_k=6`, 96 worker
threads, three warmups, and interleaved A/B timing order. Output was checked
bitwise before timing. Reported throughput counts W13 plus W2 GEMM FLOPs and
includes gather, fused epilogue, direct route store, and merge in wall time.

Representative command:

```bash
PYTHONPATH=src numactl --cpunodebind=0 --membind=0 taskset -c 0-95 \
  .venv/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_vllm_staged_schedule.py \
  --tokens 2048 --hidden 4096 --intermediate 512 \
  --experts 256 --top-k 6 --threads 96 \
  --team-threads 16 --distribution hot-topk --route-dtype fp32 \
  --warmup 3 --runs 11 --stage-timing
```

## Six long experts

`hot-topk` routes every token to the same six experts, so each active expert
has `M=2048`. The fixed control is `6 x 16T`.

| Variant | Median ms | P10 ms | P90 ms | Aggregate TFLOP/s | Relative |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-team async | 8.716 | 8.652 | 8.817 | 17.739 | baseline |
| vLLM staged | 10.027 | 9.836 | 10.510 | 15.421 | -13.07% |

The staged timing sample used 1 MiB as the available private-L2 budget and
selected W13/W2 task widths `64/256`, or 16 tasks per expert per stage. Its
wall-time decomposition was W13 `6.630 ms`, W2 `2.417 ms`, merge `0.697 ms`,
and E2E `10.050 ms`. Fixed teams share one gather-packed A per expert, whereas
the vLLM schedule rescans and packs A once per N-range; the global stage
barrier also prevents an expert from entering W2 early. Both costs are exposed
when six long experts already fill all 96 workers evenly.

## 256 short experts

`round-robin` gives every expert exactly `M=48`. Fixed team width was swept;
each point was measured in a separate process. Candidate medians varied from
`17.386` to `17.732 ms`; occasional roughly `32 ms` outliers affected P90 but
not the median.

| Threads/expert | Fixed median ms | Fixed TFLOP/s | Staged median ms | Staged vs fixed |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 35.428 | 4.364 | 17.526 | +102.14% |
| 2 | 26.396 | 5.858 | 17.732 | +48.86% |
| 4 | 18.765 | 8.240 | 17.518 | +7.12% |
| 6 | 17.572 | 8.799 | 17.386 | +1.07% |
| 8 | **17.245** | **8.966** | 17.388 | **-0.82%** |

The dynamic staged queue is substantially better than an under-parallelized
fixed schedule, but it does not beat the best fixed width. Its practical value
is therefore as a schedule baseline and as evidence that N-range work stealing
can recover poor static choices. It is not a replacement for the existing
planner: with a suitable team width it is neutral for many short experts and
materially worse for balanced long experts.
