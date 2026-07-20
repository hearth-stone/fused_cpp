# Amazon C8i 2-core AMX W13 SiLU epilogue optimization

Date: 2026-07-20

## Change and decision

The AMX W13 JIT now keeps all polynomial/exp constants in ZMM registers for
the lifetime of a generated call. Three validation alternatives remain
selectable with `FUSED_CPP_MOE_AMX_SILU_EPILOGUE`:

- `baseline`: broadcast constants separately for every output row;
- `resident`: load constants once and retain the original per-row arithmetic
  order;
- `pipelined`: resident constants plus two independent rows interleaved by
  arithmetic stage;
- `rcp14`: the two-row schedule with `VRCP14PS` replacing `VDIVPS`.

Unset, empty, or `auto` now selects `resident`. It is the only candidate that
combines bit-exact output, materially smaller generated code, a clear win when
the vector epilogue is a significant fraction of W13, and no repeatable
full-expert regression. `pipelined` and `rcp14` remain experiments. The old
path remains available as `baseline`.

## Machine and method

- host alias: `AmazonC8i2Cores`;
- CPU: Intel Xeon 6975P-C, two physical cores, one thread per core;
- compiler: GCC 15.2.0, Xbyak enabled;
- affinity: `taskset -c 0` or `taskset -c 0,1`;
- OpenMP: `OMP_DYNAMIC=FALSE`, `OMP_WAIT_POLICY=PASSIVE`;
- model shape unless noted: H=4096, F=512, one hot expert, top-k=1;
- automatic AMX pattern and W13/W2 cache windows remained enabled;
- inputs and packed weights were shared by variants; prepack, JIT generation,
  and warm-up were outside timing;
- variant order rotated on every sample to distribute frequency and thermal
  drift. Tables report medians.

The standalone benchmark was built directly from `backend.cpp`, `kernels.cpp`,
and `jit_kernels.cpp`. Representative invocation:

```bash
OMP_NUM_THREADS=1 taskset -c 0 env \
  FUSED_CPP_MOE_AMX_PATTERN=auto \
  FUSED_CPP_MOE_X86_W13_CACHE_BLOCKS=auto \
  /tmp/bench_amx_bf16_w13_silu \
  16 4096 512 20 101 5 baseline,resident,pipelined,rcp14
```

The full expert comparison used:

```bash
OMP_NUM_THREADS=2 OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  taskset -c 0,1 env PYTHONPATH=src .venv/bin/python \
  benchmarks/bench_amx_bf16_patterns.py \
  --tokens 256 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads 2 \
  --patterns auto \
  --silu-epilogues baseline,resident,pipelined,rcp14 \
  --warmup 5 --runs 31
```

## Correctness and code size

The focused epilogue matrix covered degrees 4/5/6, `m1n2`/`m2n2`/`m1n4`,
odd M tails, and one/two threads: `19 passed`. The complete x86 backend suites
completed with `114 passed, 4 skipped`.

`resident` and `pipelined` were BF16 bit-identical to `baseline` in all tests
and benchmarks. `rcp14` stayed within the existing BF16 contract but was not
bit-identical; full-expert maximum absolute differences were at most
`1.1920929e-7` for the measured input distribution.

For M=16, degree 5, the generated W13 body changed as follows:

| epilogue | JIT bytes | versus baseline |
| --- | ---: | ---: |
| baseline | 3972 | 1.00x |
| resident | 2836 | -28.6% |
| pipelined | 2844 | -28.4% |
| rcp14 | 2940 | -26.0% |

## Standalone W13 results

| M | K | baseline ms | resident ms | pipelined ms | rcp14 ms | resident speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 0.016641 | 0.015345 | 0.015245 | 0.015183 | 1.084x |
| 16 | 4096 | 0.327297 | 0.326577 | 0.325926 | 0.326089 | 1.002x |
| 64 | 4096 | 0.841795 | 0.842180 | 0.841864 | 0.841010 | 1.000x |
| 256 | 4096 | 3.854725 | 3.867907 | 3.849165 | 3.835961 | 0.997x |
| 2048 | 4096 | 27.503988 | 27.183801 | 27.432264 | 27.063914 | 1.012x |

Constant loads are material when K is short: at K=256, `resident` improves
median W13 throughput from 504.1 to 546.7 GFLOP/s. At K=4096, AMX matrix work
dominates and the same optimization is mostly hidden.

## Full fused-expert results

| M | threads | baseline ms | resident ms | pipelined ms | rcp14 ms | resident speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1 | 0.521258 | 0.520430 | 0.520101 | 0.520079 | 1.002x |
| 256 | 1 | 5.648067 | 5.556429 | 5.646655 | 5.572946 | 1.016x |
| 2048 | 1 | 40.787223 | 40.656609 | 40.628027 | 40.599553 | 1.003x |
| 16 | 2 | 0.374772 | 0.376795 | 0.373533 | 0.374162 | 0.995x |
| 256 | 2 | 4.427875 | 4.428617 | 4.433165 | 4.418172 | 1.000x |
| 2048 | 2 | 32.546159 | 32.483778 | 32.452361 | 32.509330 | 1.002x |

The full-expert `resident` range is 0.995x-1.016x. Differences below roughly
1% are within the VM's observed run-to-run frequency/thermal variation, so the
main durable gain at H=4096 is reduced JIT size rather than a large latency
change. The two-row schedule did not consistently improve on `resident`:
because rows are already statically unrolled, the out-of-order core can see and
overlap later independent `VDIVPS` chains without stage-wise source ordering.

`rcp14` is therefore not justified: it changes results and its small apparent
speed differences are no larger than noise in the full expert. Future work
should target W2 stores/workspace and tile/cache scheduling before revisiting
reciprocal refinement.
