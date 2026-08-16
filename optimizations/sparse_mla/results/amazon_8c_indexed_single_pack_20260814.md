# Amazon 8C Sparse MLA Indexed Single Gather-Pack

Date: 2026-08-14

## Change and adoption gate

The query-dependent sparse path cannot normally reuse a packed KV tile across
tokens. It can still avoid loading the leading `d_v` elements twice: V is the
leading slice of the same MLA KV row used by K. The candidate loads four
indexed rows eight elements at a time, writes two adjacent K=4 BFMMLA panels,
and transposes the same vectors into the V panel.

The gate was: both runtime vector lengths pass the complete Sparse MLA test
file, later sparse latency improves by at least 5% at one and eight threads,
the checksum is unchanged, and the shared-prefix path stays within 1%.

## Machine and method

- Host: `AmazonECS8Cores`, cores 0--7, native SVE VL 256 bits.
- Shape: `q[2048,32,192]`, `kv[15060,1,192]`, `topk=640`, `d_v=128`,
  `context_start=10000`, compressed capacity 512, window 128, ratio 4.
- Valid pairs: 1,310,720; seed 20260814.
- Baseline: commit `e9b5718`, separate indexed K and V pack.
- Candidate: fused indexed K/V pack; identical build flags and extension.
- Timing: five warmups and 21 samples, median wall time, pinned cores,
  `OMP_DYNAMIC=FALSE`, dependent math libraries limited to one thread.

Command template:

```bash
OMP_NUM_THREADS=<1|8> OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c <0|0-7> .venv/bin/python tests/bench_sparse_mla_scalable.py \
  --pattern sparse --s-q 2048 --h-q 32 --d-qk 192 --d-v 128 \
  --compressed-capacity 512 --window-size 128 --compress-ratio 4 \
  --context-start 10000 --threads <1|8> --warmup 5 --iters 21 \
  --seed 20260814
```

## Results

| Threads/order | Baseline | Candidate | Latency change | Candidate source throughput |
|---|---:|---:|---:|---:|
| 1, baseline then candidate | 280.552 ms | 254.908 ms | **-9.14%** | 105.31 GFLOP/s |
| 8, baseline then candidate | 41.584 ms | 39.161 ms | **-5.83%** | 685.47 GFLOP/s |
| 8, candidate then baseline | 41.671 ms | 39.155 ms | **-6.04%** | 685.57 GFLOP/s |

The checksum was `-12987.589844` at one thread and `-12987.588867` at eight
threads for both binaries. With profiling enabled for a single one-thread
iteration, indexed pack fell from 61.209 to 37.819 ms (-38.21%), while total
profiled time fell from 289.690 to 263.075 ms. QK, softmax, and PV were not
changed.

The first-2048 shared-prefix guardrail, using compressed capacity 512, window
2048, and `context_start=0`, was 77.088 ms for the baseline and 76.952 ms for
the candidate at eight threads (-0.18%). This path does not call the new
indexed pack.

## Correctness

Native SVL256 on cores 0--3:

```text
OMP_NUM_THREADS=4 taskset -c 0-3 ... pytest -q tests/test_sparse_mla.py
26 passed in 29.29s
```

Forced SVL128 on cores 4--7, with `PR_SVE_SET_VL=16` before importing the
extension:

```text
OMP_NUM_THREADS=4 taskset -c 4-7 ... pytest -q tests/test_sparse_mla.py
26 passed in 27.90s
```

## Raw timing samples

```text
1T baseline: 280.590 280.228 280.438 280.306 280.383 280.288 280.687
280.492 280.556 280.378 280.807 280.552 280.707 280.567 280.646 280.374
280.490 280.665 280.751 280.357 280.904
1T candidate: 255.159 254.852 255.035 254.943 255.099 254.916 255.090
254.725 254.851 254.675 255.017 254.817 254.842 254.719 255.105 254.908
254.892 254.747 255.035 254.829 255.100
8T baseline: 41.591 41.564 41.665 41.585 41.648 41.570 41.670 41.503
41.660 41.509 41.596 41.528 41.590 41.491 41.611 41.527 41.582 41.526
41.623 41.524 41.584
8T candidate: 39.172 39.115 39.227 39.203 39.180 39.172 39.184 39.120
39.146 39.094 39.161 39.102 39.202 39.087 39.197 39.096 39.159 39.143
39.243 39.155 39.176
8T reverse candidate: 39.180 39.102 39.245 39.138 39.203 39.091 39.148
39.098 39.184 39.165 39.187 39.083 39.216 39.116 39.195 39.115 39.244
39.164 39.155 39.147 39.148
8T reverse baseline: 41.690 41.639 41.734 41.655 41.664 41.653 41.754
41.635 41.722 41.604 41.704 41.632 41.665 41.649 41.696 41.639 41.697
41.676 41.703 41.671 41.714
```

All adoption gates pass. The rollback boundary is this optimization commit;
the prior shared-prefix commit remains independently usable.
