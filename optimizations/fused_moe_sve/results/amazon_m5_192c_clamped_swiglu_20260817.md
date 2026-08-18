# DeepSeek-V4 Clamped SwiGLU on AmazonM5192Cores

## Decision

Enable explicit `swiglu_limit=10.0` for the SVE JIT fused W13 epilogue. The
standard unclamped operation retains a separate JIT cache key and generated
instruction stream. Static asm and other limits are rejected.

## Correctness

The benchmark covers exact-M JIT rows 1 through 12 and polynomial degrees
4/5/6 with inputs that cross both clamp bounds. Every fused output matched the FP32 reference
`silu(min(gate, 10)) * clamp(up, -10, 10)` exactly for the tested seed;
`swiglu_limit=0` matched the existing operation bitwise. Standalone shared MLP
and combined routed+shared execution matched their explicit routed references
bitwise; scheduled and async bridges matched normal execution bitwise. Forced
`FUSED_CPP_MOE_SVE_IMPL=asm` was rejected.

## Performance

Host: AmazonM5192Cores NUMA0, CPUs 0-95, SVE128, BF16, H=7168, F=768.
Times are medians from alternating standard/clamped calls.

| M / threads | Scope | Standard ms | Clamped ms | Overhead |
| --- | --- | ---: | ---: | ---: |
| 12 / 1T | pure W13 JIT | 0.809345 | 0.809336 | -0.00% |
| 12 / 1T | complete expert | 1.440356 | 1.442591 | +0.16% |
| 2040 / 16T | pure W13 JIT | 136.602575 | 136.768479 | +0.12% |
| 2040 / 16T | complete expert | 243.875594 | 243.740775 | -0.06% |

The complete routed+shared V4-Pro TP4 benchmark with `swiglu_limit=10` retained
the same plans and improved balanced/uniform/hotspot by
16.99%/20.98%/78.59% versus sequential routed plus shared execution. See
`amazon_m5_192c_routed_shared_schedule_20260817.md`.

## Command

```bash
PYTHONPATH=/data/fused_cpp/src \
numactl --physcpubind=0-95 --membind=0 \
/data/vllm/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_clamped_swiglu.py \
  --rows 2040 --hidden 7168 --intermediate 768 --threads 16 \
  --warmup 5 --runs 31
```
