# SVE degree-5 polynomial evaluator comparison

## Configuration

- Machine: `AmazonM5192Cores`, NUMA0 cores `0-91`, memory node 0.
- ISA: SVE128, FP32 degree-5 exp approximation.
- Shapes: M8 with 16 live input Z registers; M12 with 24 live input Z registers.
- Variants: two-register packed-constant Horner, indexed-FMLA even/odd, and indexed-FMLA Estrin.
- Scopes: exp construction, complete SiLU division, and limit-10 clamped SwiGLU.
- Build: `g++ -O2 -std=c++17 -fopenmp -march=armv8.6-a+sve -msve-vector-bits=128`.
- Method: 30 warmups, 51 alternating samples, 4,000 calls per worker per sample; median wall time per
  simultaneous worker wave. Each worker used private L1-hot input/output buffers.
- Correctness: all 18 shape/scope/evaluator combinations were checked against the same degree-5 scalar
  reference before timing with maximum relative error at most `2e-5`.

Command:

```bash
OMP_NUM_THREADS=92 OMP_DYNAMIC=FALSE OMP_PROC_BIND=TRUE OMP_PLACES=cores \
numactl --physcpubind=0-91 --membind=0 \
env FUSED_CPP_MOE_SVE_VECTOR_BITS=128 \
optimizations/fused_moe_sve/benchmarks/run_sve_exp_evaluators.sh \
  --warmup 30 --runs 51 --inner 4000 --threads 92
```

## Results

Times are nanoseconds per simultaneous wave. Gains are relative to Horner; positive is faster.

| M | Scope | Horner | Even/odd | Gain | Estrin | Gain |
|---:|---|---:|---:|---:|---:|---:|
| 8 | exp | 39.807 | 40.680 | -2.15% | 43.858 | -9.24% |
| 8 | SiLU | 40.938 | 41.012 | -0.18% | 42.858 | -4.48% |
| 8 | clamped SiLU | 41.401 | 41.005 | +0.97% | 43.125 | -4.00% |
| 12 | exp | 58.950 | 59.385 | -0.73% | 64.730 | -8.93% |
| 12 | SiLU | 60.159 | 60.283 | -0.21% | 62.973 | -4.47% |
| 12 | clamped SiLU | 60.600 | 59.043 | +2.64% | 63.584 | -4.69% |

The single-core control produced the same ordering: even/odd was `-1.81/-0.66/+3.08%` for M8 and
`-0.90/-0.18/+2.74%` for M12 across exp/SiLU/clamped SiLU. Estrin regressed by roughly `4.4-9.4%`.

## Conclusion

Horner remains the default. For these live-register shapes, instruction throughput matters more than the shorter
polynomial dependency chain: Estrin adds one FP instruction and three coefficient broadcasts, while direct-power
even/odd adds two FP instructions. Complete SiLU also places the long-latency division after the polynomial, reducing
the value of shortening only the FMA chain.

Even/odd's repeatable clamped-SiLU improvement is scope-specific and does not pass the adoption gate: it is below 2%
for M8 and does not improve ordinary SiLU. It should not be enabled globally. A future candidate would need to explain
why clamp changes the scheduling result and validate inside the actual pair-interleaved fused W13 epilogue rather than
this isolated sequential-row evaluator.

This experiment isolates evaluation style under 16/24 simultaneously live gate/up registers. It processes rows
sequentially with two packed coefficient registers; it is not a timing of the current production M12 pair-interleaved
epilogue and does not establish an end-to-end W13 speedup.
