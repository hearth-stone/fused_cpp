# Routed + Shared Synthetic Expert Scheduling on AmazonM5192Cores

## Decision

Enable the standalone/TP synthetic-shared Plan V2 path for one same-shape
shared expert. All three primary route distributions passed the adoption gate:
at least 2% E2E gain, no regression, cache-hit planning below 3 ms, and cold
planning below 25 ms.

## Configuration

- Host: AmazonM5192Cores, Neoverse-V3, NUMA0 CPUs 0-95
- Shape: BF16, tokens=2048, H=7168, F=768, routed experts=384, TopK=6, TP=4, `swiglu_limit=10`
- Affinity: `numactl --physcpubind=0-95 --membind=0`
- Profile: `profiles/m5-numa0-96c-quick-20260817.json`
- Profile SHA256: `1939a6ad44732a5f119803adddd34899f7d56380d177c4a31e9583eb78caf95c`
- Source base: Git `b1b79b7` plus the synthetic-shared implementation under test
- Method: 3 warmups, 21 alternating baseline/candidate pairs, median wall time

The baseline runs planned routed MoE, standalone shared MLP, then `torch.add`.
The candidate appends one all-token unit-weight synthetic expert and executes a
single strict Plan V2 with the existing fused expert kernels and FP32 route merge.

## Results

| Distribution | Routed routes min/median/max | Baseline ms | Candidate ms | Gain | Candidate TFLOP/s | Plan | Cold / hit plan ms |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| balanced | 32 / 32 / 32 | 122.004 | 104.282 | 16.99% | 4.541 | shared 32T + 16 routed lanes x 4T | 4.572 / 0.997 |
| uniform | 18 / 32 / 47 | 130.313 | 107.711 | 20.98% | 4.396 | shared 32T + 16 routed lanes x 4T | 17.593 / 1.118 |
| hotspot | 7 / 18 / 159 | 116.363 | 65.157 | 78.59% | 7.267 | shared 16T + 5 routed lanes x 16T | 23.607 / 1.034 |

The candidate differed from the sequential baseline by at most `5.9604645e-8`
in FP32 comparison. Focused tests additionally require bitwise equality between
the combined API and the explicit synthetic-route native call.

## Command

```bash
PYTHONPATH=/data/fused_cpp/src \
OMP_NUM_THREADS=96 \
FUSED_CPP_MOE_PREPACK_THREADS=96 \
numactl --physcpubind=0-95 --membind=0 \
/data/vllm/bin/python \
  optimizations/fused_moe_sve/benchmarks/bench_routed_shared_schedule.py \
  --profile profiles/m5-numa0-96c-quick-20260817.json \
  --swiglu-limit 10 \
  --warmup 3 --runs 21
```

## Scope

This result covers one same-F shared expert, SVE BF16 JIT limit-10 clamped SwiGLU, and
standalone/TP with complete local routed weights. EP, bias, multiple shared
experts, different shared F, static asm, and other clamp limits remain outside this API.
