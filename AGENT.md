## Remote Execution

Remote development and benchmark runs primarily use the `Arm-codex` SSH alias.
When the user says `aws机器`, use the `AmazonECS8Cores` SSH alias.

### Connection

- Host alias: `Arm-codex`
- User: `zhangxu`
- Remote project root: `/home/zhangxu/codex/fused_cpp`
- Local project root: this repository root

### AWS Machine

This is the target machine whenever the user says `aws机器`.

- Host alias: `AmazonECS8Cores`
- Remote project root: `/home/ubuntu/zhangxu/fused_cpp`
- Python activate script: `/home/ubuntu/zhangxu/fused_cpp/.venv/bin/activate`
- Python executable: `/home/ubuntu/zhangxu/fused_cpp/.venv/bin/python`
- Benchmark cores: bind benchmark processes to cores `0` through `7`

Example:

```bash
ssh AmazonECS8Cores 'cd /home/ubuntu/zhangxu/fused_cpp && . .venv/bin/activate && python -c "import sys; print(sys.executable)"'
```

Sync local files to the AWS machine with:

```bash
bash rsync_aws.sh
```

For single-thread benchmarks on the AWS machine, use one process per core from
`0` to `7`:

```bash
ssh AmazonECS8Cores 'cd /home/ubuntu/zhangxu/fused_cpp && OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 taskset -c 0 .venv/bin/python tests/bench_microkernel_qkt.py'
```

### Sync

Use the repository sync script before running remote tests:

```bash
bash rsync.sh
```

The local repository root maps to:

```text
/home/zhangxu/codex/fused_cpp
```

### Python Environment

The remote Python environment is managed by `uv`.

- Activate script: `/home/zhangxu/codex/fused_cpp/.venv/bin/activate`
- Python executable: `/home/zhangxu/codex/fused_cpp/.venv/bin/python`
- Package manager: `uv`

Prefer the explicit Python executable in non-interactive SSH commands, for example:

```bash
ssh Arm-codex 'cd /home/zhangxu/codex/fused_cpp && .venv/bin/python -c "from fused_cpp import _C; print(_C.has_openmp())"'
```

Avoid relying on the remote system `python`; it may not match the virtualenv ABI used
to build `fused_cpp._C`.

Install missing benchmark or analysis packages with `uv` when needed. Keep packages in
the project-local virtualenv instead of using system Python, for example:

```bash
ssh Arm-codex 'cd /home/zhangxu/codex/fused_cpp && uv pip install <package>'
```

### Benchmark Hygiene

For single-thread microbenchmarks, bind each process to one dedicated core with
`taskset` and pin both Python and native libraries to one thread:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c 80 <command>
```

The independent benchmark cores are `0`, `80`, `160`, and `240`. These four cores can
run separate benchmark processes at the same time. Use one benchmark process per core
and record the core id with the result. Do not put two single-thread benchmarks on the
same core.

Put `taskset -c <core>` immediately before the Python or shell entrypoint, for example:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c 80 .venv/bin/python tests/bench_microkernel_qkt.py
```

When comparing measured GFLOP/s with peak, use the single-core reference values below.
If a reported single-thread result is above the relevant peak, first check whether the
thread count, OpenMP runtime, or benchmark FLOP formula is misleading.

### Optimization Log

Record every optimization attempt in the unified file `csrc/SDPA_VERSIONS.md`, even
when the result does not improve performance. Each entry should include the date,
kernel/version, hypothesis, exact benchmark command, core id, before/after numbers,
and the conclusion. This prevents repeating optimizations that were already measured
and found ineffective.

## Arm-codex Peak FLOPs Reference

The following numbers are `Arm-codex` single-core peak measurements unless noted
otherwise. Use them as sanity checks for SDPA and microkernel benchmark results on
that machine.

```
| Instruction Set | Core Computation              | Peak Performance | IPC      | Latency |
| i8mm            | mmla(s32,s8,s8)               | 300.42 GOPS      | 1.620051 | -       |
| i8mm            | mmla(u32,u8,u8)               | 306.9 GOPS       | 1.654971 | 4       |
| i8mm            | mmla(s32,u8,s8)               | 301.82 GOPS      | 1.627599 | 4       |
| i8mm            | dp4a.vs(s32,s8,u8)            | 331.02 GOPS      | 3.570129 | 2       |
| i8mm            | dp4a.vs(s32,u8,s8)            | 331.13 GOPS      | 3.571318 | 2       |
| i8mm            | dp4a.vv(s32,u8,s8)            | 332.17 GOPS      | 3.582499 | 2       |
| asimd_dp        | dp4a.vs(s32,s8,s8)            | 331.64 GOPS      | 3.576833 | 2       |
| asimd_dp        | dp4a.vv(s32,s8,s8)            | 330.76 GOPS      | 3.567240 | 2       |
| asimd_dp        | dp4a.vs(u32,u8,u8)            | 330.77 GOPS      | 3.567372 | 2       |
| asimd_dp        | dp4a.vv(u32,u8,u8)            | 329.74 GOPS      | 3.556314 | 2       |
| bf16            | mmla(f32,bf16,bf16)           | 92.704 GFLOPS    | 0.999831 | 4       |
| bf16            | dp2a.vs(f32,bf16,bf16)        | 75.077 GFLOPS    | 1.619435 | 2       |
| bf16            | dp2a.vv(f32,bf16,bf16)        | 75.183 GFLOPS    | 1.621712 | 2       |
| bf16            | bfmlalb(f32,bf16,bf16)        | 91.086 GFLOPS    | 3.929519 | 2       |
| bf16            | bfmlalt(f32,bf16,bf16)        | 91.033 GFLOPS    | 3.927227 | 2       |
| FHM             | fmlal.vv(f32,f16,f16)         | 91.019 GFLOPS    | 3.926624 | -       |
| FHM             | fmlal2.vv(f32,f16,f16)        | 91.023 GFLOPS    | 3.926801 | 2       |
| FHM             | fmlal.vs(f32,f16,f16)         | 91.11 GFLOPS     | 3.930515 | 2       |
| FHM             | fmlal_pair.vv(f32,f16,f16)    | 90.958 GFLOPS    | 3.923962 | -       |
| asimd_hp        | fmla.vs(fp16,fp16,fp16)       | 182.06 GFLOPS    | 3.927156 | -       |
| asimd_hp        | fmla.vv(fp16,fp16,fp16)       | 182.05 GFLOPS    | 3.926961 | -       |
| asimd           | fmla.vs(f32,f32,f32)          | 91.019 GFLOPS    | 3.926624 | 2       |
| asimd           | fmla.vv(f32,f32,f32)          | 90.977 GFLOPS    | 3.924814 | 2       |
| asimd           | fmla.vs(f64,f64,f64)          | 45.55 GFLOPS     | 3.930142 | 2       |
| asimd           | fmla.vv(f64,f64,f64)          | 45.531 GFLOPS    | 3.928435 | 2       |
| asimd           | hybrid_fp32_mla_6x16          | 90.975 GFLOPS    | 0.981175 | -       |
| asimd           | fmls.vv(f32,f32,f32)          | 91.093 GFLOPS    | 3.929822 | 2       |
| asimd           | fneg+fmla.vv(f32,f32,f32)     | 30.032 GFLOPS    | 2.591153 | 3       |
| asimd           | fadd.vv(f32,f32,f32)          | 46.327 GFLOPS    | 3.997173 | 2       |
| asimd           | fmul.vv(f32,f32,f32)          | 46.327 GFLOPS    | 3.997118 | 3       |
| asimd           | sve_fmla.vs(f32,f32,f32)      | 92.669 GFLOPS    | 1.998890 | -       |
| asimd           | sve_fmla.vv(f32,f32,f32)      | 92.667 GFLOPS    | 1.998849 | 4       |
| asimd           | sve_fmla.vs(f64,f64,f64)      | 46.331 GFLOPS    | 1.998748 | -       |
| asimd           | sve_fmla.vv(f64,f64,f64)      | 46.333 GFLOPS    | 1.998835 | 4       |
| sve_complex     | sve_fcmla.vv(f32,f32,f32)#0   | 92.663 GFLOPS    | 1.998766 | 2       |
| sve_complex     | sve_fcmla.vv(f32,f32,f32)#90  | 92.661 GFLOPS    | 1.998715 | 2       |
| sve_complex     | sve_fcmla.vv(f32,f32,f32)#180 | 92.662 GFLOPS    | 1.998738 | 2       |
| sve_complex     | sve_fcmla.vv(f32,f32,f32)#270 | 92.66 GFLOPS     | 1.998706 | 2       |
| sve_complex     | sve_fcmla.vv(f64,f64,f64)#0   | 46.327 GFLOPS    | 1.998591 | 2       |
| sve_complex     | sve_fcmla.vv(f64,f64,f64)#90  | 46.335 GFLOPS    | 1.998904 | 2       |
| sve_complex     | sve_fcmla.vv(f64,f64,f64)#180 | 46.332 GFLOPS    | 1.998784 | 2       |
| sve_complex     | sve_fcmla.vv(f64,f64,f64)#270 | 46.333 GFLOPS    | 1.998844 | 2       |
| sve_reduce      | sve_fadda(f32)                | 1.0535 GFLOPS    | 0.045449 | 22      |
| sve_reduce      | sve_fadda(f64)                | 0.82767 GFLOPS   | 0.071413 | 14      |
| sve_reduce      | sve_faddv(f32)                | 15.452 GFLOPS    | 0.666588 | -       |
| sve_reduce      | sve_faddv(f64)                | 11.587 GFLOPS    | 0.999743 | -       |
| asimd_fcma      | fcmla.vv(f32,f32,f32)#0       | 91.023 GFLOPS    | 3.926801 | 2       |
| asimd_fcma      | fcmla.vv(f32,f32,f32)#90      | 90.997 GFLOPS    | 3.925647 | 2       |
| asimd_fcma      | fcmla.vv(f32,f32,f32)#180     | 91.043 GFLOPS    | 3.927653 | 2       |
| asimd_fcma      | fcmla.vv(f32,f32,f32)#270     | 91.089 GFLOPS    | 3.929626 | 2       |
| asimd_fcma      | fcmla_pair.vv(f32,f32,f32)    | 87.731 GFLOPS    | 3.784774 | -       |
| asimd_fcma      | fcmla.vv(f16,f16,f16)#0       | 182.11 GFLOPS    | 3.928257 | 2       |
| asimd_fcma      | fcmla.vv(f16,f16,f16)#90      | 182.26 GFLOPS    | 3.931476 | 2       |
| asimd_fcma      | fcmla.vv(f16,f16,f16)#180     | 182.13 GFLOPS    | 3.928577 | 2       |
| asimd_fcma      | fcmla.vv(f16,f16,f16)#270     | 182.05 GFLOPS    | 3.926766 | 2       |
| asimd_reduce    | faddp.vvv(f32)                | 46.311 GFLOPS    | 3.995757 | 3       |
| asimd_reduce    | fmaxv(f32)                    | 23.166 GOPS      | 1.998766 | -       |
| asimd_reduce    | saddlv(s8)                    | 92.661 GOPS      | 1.998715 | -       |
| asimd_reduce    | smaxv(s32)                    | 23.165 GOPS      | 1.998683 | -       |
| asimd_reduce    | fmaxv(f16)                    | 46.33 GOPS       | 1.998697 | -       |
| asimd_recip     | frecpe+frecps(f32)            | 39.961 GFLOPS    | 2.298590 | 4       |
| asimd_recip     | frsqrte+frsqrts(f32)          | 39.983 GFLOPS    | 2.299850 | 4       |
| asimd_recip     | frecpe+frecps(f16)            | 79.92 GFLOPS     | 2.298535 | 4       |
| asimd_int_mac   | mla.vs(s32,s32,s32)           | 46.332 GOPS      | 1.998780 | 2       |
| asimd_int_mac   | mla.vv(s32,s32,s32)           | 46.332 GOPS      | 1.998807 | 2       |
| asimd_int_mac   | mla.vs(s16,s16,s16)           | 92.673 GOPS      | 1.998982 | 2       |
| asimd_int_mac   | sqdmlal.vv(s32,s16,s16)       | 46.327 GOPS      | 1.998564 | 2       |
| asimd_int_mac   | sqdmlal2.vv(s32,s16,s16)      | 46.33 GOPS       | 1.998702 | 2       |
| asimd_tbl       | tbl.4table(u8)                | 60.02 GOPS       | 1.294650 | 6       |
| asimd_tbl       | tbx.4table(u8)                | 46.351 GOPS      | 0.999808 | 8       |
| sve_i8mm        | sve_mmla(s32,s8,s8)           | 370.82 GOPS      | 0.999837 | 4       |
| sve_i8mm        | sve_mmla(u32,u8,u8)           | 370.79 GOPS      | 0.999751 | 4       |
| sve_i8mm        | sve_mmla(s32,u8,s8)           | 370.81 GOPS      | 0.999801 | 4       |
| sve_i8mm        | sve_dp4a.vv(s32,s8,s8)        | 92.664 GOPS      | 1.998798 | 2       |
| sve_i8mm        | sve_dp4a.vv(s32,u8,u8)        | 92.668 GOPS      | 1.998881 | 2       |
| sve_i8mm        | sve_dp4a.vv(s32,u8,s8)        | 92.66 GOPS       | 1.998706 | 2       |
| sve_bf16        | sve_bfmmla(f32,bf16,bf16)     | 92.72 GFLOPS     | 0.499998 | 4       |
| sve_bf16        | sve_bfdot.vv(f32,bf16,bf16)   | 92.702 GFLOPS    | 0.999802 | 2       |
| sve_bf16        | sve_bfdot.vs(f32,bf16,bf16)   | 92.701 GFLOPS    | 0.999791 | 2       |
| sve_f32mm       | sve_fmmla(f32,f32,f32)        | 88.219 GFLOPS    | 0.475726 | -       |
| sve_f64mm       | sve_fmmla(f64,f64,f64)        | 22.05 GFLOPS     | 0.475619 | 3       |
| sve_fp16        | sve_fmla.vv(f16,f16,f16)      | 185.29 GFLOPS    | 1.998380 | 3       |
| sve_fp16        | sve_fmla.vs(f16,f16,f16)      | 185.3 GFLOPS     | 1.998536 | -       |
```

## AWS Machine Performance Reference

The following numbers were measured on the `AmazonECS8Cores` machine. Some AWS
instruction-level performance tests are known to be inaccurate or unstable, so treat
this section as reference data only. Use it for rough sanity checks and relative
comparisons, not as authoritative peak throughput.

```

-------------------------------------------------------------------------------------------
| Instruction Set | Core Computation              | Peak Performance | IPC      | Latency |
| i8mm            | mmla(s32,s8,s8)               | 331.58 GOPS      | 2.001141 | -       |
| i8mm            | mmla(u32,u8,u8)               | 331.56 GOPS      | 2.001072 | 1       |
| i8mm            | mmla(s32,u8,s8)               | 331.44 GOPS      | 2.000321 | 1       |
| i8mm            | dp4a.vs(s32,s8,u8)            | 165.8 GOPS       | 2.001240 | 1       |
| i8mm            | dp4a.vs(s32,u8,s8)            | 165.77 GOPS      | 2.000977 | 1       |
| i8mm            | dp4a.vv(s32,u8,s8)            | 165.78 GOPS      | 2.001047 | 1       |
| asimd_dp        | dp4a.vs(s32,s8,s8)            | 165.79 GOPS      | 2.001107 | 1       |
| asimd_dp        | dp4a.vv(s32,s8,s8)            | 165.76 GOPS      | 2.000810 | 1       |
| asimd_dp        | dp4a.vs(u32,u8,u8)            | 165.77 GOPS      | 2.000905 | 1       |
| asimd_dp        | dp4a.vv(u32,u8,u8)            | 165.8 GOPS       | 2.001307 | 1       |
| bf16            | mmla(f32,bf16,bf16)           | 165.77 GFLOPS    | 2.000917 | 3       |
| bf16            | dp2a.vs(f32,bf16,bf16)        | 82.89 GFLOPS     | 2.001033 | 3       |
| bf16            | dp2a.vv(f32,bf16,bf16)        | 82.89 GFLOPS     | 2.001047 | 3       |
| bf16            | bfmlalb(f32,bf16,bf16)        | 41.448 GFLOPS    | 2.001183 | 2       |
| bf16            | bfmlalt(f32,bf16,bf16)        | 41.447 GFLOPS    | 2.001137 | 2       |
| FHM             | fmlal.vv(f32,f16,f16)         | 41.448 GFLOPS    | 2.001196 | -       |
| FHM             | fmlal2.vv(f32,f16,f16)        | 41.438 GFLOPS    | 2.000693 | 2       |
| FHM             | fmlal.vs(f32,f16,f16)         | 41.441 GFLOPS    | 2.000839 | 2       |
| FHM             | fmlal_pair.vv(f32,f16,f16)    | 41.449 GFLOPS    | 2.001254 | -       |
| asimd_hp        | fmla.vs(fp16,fp16,fp16)       | 82.88 GFLOPS     | 2.000797 | -       |
| asimd_hp        | fmla.vv(fp16,fp16,fp16)       | 82.881 GFLOPS    | 2.000819 | -       |
| asimd           | fmla.vs(f32,f32,f32)          | 41.447 GFLOPS    | 2.001132 | 2       |
| asimd           | fmla.vv(f32,f32,f32)          | 41.451 GFLOPS    | 2.001319 | 2       |
| asimd           | fmla.vs(f64,f64,f64)          | 20.721 GFLOPS    | 2.000944 | 2       |
| asimd           | fmla.vv(f64,f64,f64)          | 20.718 GFLOPS    | 2.000657 | 2       |
| asimd           | hybrid_fp32_mla_6x16          | 41.464 GFLOPS    | 0.500494 | -       |
| asimd           | fmls.vv(f32,f32,f32)          | 41.443 GFLOPS    | 2.000933 | 2       |
| asimd           | fneg+fmla.vv(f32,f32,f32)     | 13.828 GFLOPS    | 1.335251 | 3       |
| asimd           | fadd.vv(f32,f32,f32)          | 20.722 GFLOPS    | 2.000988 | 2       |
| asimd           | fmul.vv(f32,f32,f32)          | 20.717 GFLOPS    | 2.000508 | 3       |
| asimd           | sve_fmla.vs(f32,f32,f32)      | 76.468 GFLOPS    | 1.846017 | -       |
| asimd           | sve_fmla.vv(f32,f32,f32)      | 76.459 GFLOPS    | 1.845797 | 4       |
| asimd           | sve_fmla.vs(f64,f64,f64)      | 38.254 GFLOPS    | 1.846994 | -       |
| asimd           | sve_fmla.vv(f64,f64,f64)      | 38.245 GFLOPS    | 1.846566 | 4       |
| sve_complex     | sve_fcmla.vv(f32,f32,f32)#0   | 76.497 GFLOPS    | 1.846714 | 2       |
| sve_complex     | sve_fcmla.vv(f32,f32,f32)#90  | 76.455 GFLOPS    | 1.845693 | 2       |
| sve_complex     | sve_fcmla.vv(f32,f32,f32)#180 | 76.477 GFLOPS    | 1.846230 | 2       |
| sve_complex     | sve_fcmla.vv(f32,f32,f32)#270 | 76.455 GFLOPS    | 1.845686 | 2       |
| sve_complex     | sve_fcmla.vv(f64,f64,f64)#0   | 38.249 GFLOPS    | 1.846757 | 2       |
| sve_complex     | sve_fcmla.vv(f64,f64,f64)#90  | 38.247 GFLOPS    | 1.846637 | 2       |
| sve_complex     | sve_fcmla.vv(f64,f64,f64)#180 | 38.237 GFLOPS    | 1.846178 | 2       |
| sve_complex     | sve_fcmla.vv(f64,f64,f64)#270 | 38.251 GFLOPS    | 1.846809 | 2       |
| sve_reduce      | sve_fadda(f32)                | 1.8845 GFLOPS    | 0.090987 | 11      |
| sve_reduce      | sve_fadda(f64)                | 1.2949 GFLOPS    | 0.125040 | 8       |
| sve_reduce      | sve_faddv(f32)                | 8.5313 GFLOPS    | 0.411908 | -       |
| sve_reduce      | sve_faddv(f64)                | 5.1407 GFLOPS    | 0.496404 | -       |
| asimd_fcma      | fcmla.vv(f32,f32,f32)#0       | 41.449 GFLOPS    | 2.001243 | 2       |
| asimd_fcma      | fcmla.vv(f32,f32,f32)#90      | 41.438 GFLOPS    | 2.000686 | 2       |
| asimd_fcma      | fcmla.vv(f32,f32,f32)#180     | 41.436 GFLOPS    | 2.000597 | 2       |
| asimd_fcma      | fcmla.vv(f32,f32,f32)#270     | 41.436 GFLOPS    | 2.000622 | 2       |
| asimd_fcma      | fcmla_pair.vv(f32,f32,f32)    | 41.43 GFLOPS     | 2.000311 | -       |
| asimd_fcma      | fcmla.vv(f16,f16,f16)#0       | 82.879 GFLOPS    | 2.000784 | 2       |
| asimd_fcma      | fcmla.vv(f16,f16,f16)#90      | 82.869 GFLOPS    | 2.000534 | 2       |
| asimd_fcma      | fcmla.vv(f16,f16,f16)#180     | 82.883 GFLOPS    | 2.000877 | 2       |
| asimd_fcma      | fcmla.vv(f16,f16,f16)#270     | 82.887 GFLOPS    | 2.000962 | 2       |
| asimd_reduce    | faddp.vvv(f32)                | 20.721 GFLOPS    | 2.000935 | 3       |
| asimd_reduce    | fmaxv(f32)                    | 19.126 GOPS      | 1.846872 | -       |
| asimd_reduce    | saddlv(s8)                    | 29.44 GOPS       | 0.710701 | -       |
| asimd_reduce    | smaxv(s32)                    | 19.128 GOPS      | 1.847084 | -       |
| asimd_reduce    | fmaxv(f16)                    | 27.635 GOPS      | 1.334259 | -       |
| asimd_recip     | frecpe+frecps(f32)            | 30.97 GFLOPS     | 1.993725 | 4       |
| asimd_recip     | frsqrte+frsqrts(f32)          | 31.066 GFLOPS    | 1.999872 | 4       |
| asimd_recip     | frecpe+frecps(f16)            | 31.098 GFLOPS    | 1.000973 | 5       |
| asimd_int_mac   | mla.vs(s32,s32,s32)           | 38.251 GOPS      | 1.846826 | 2       |
| asimd_int_mac   | mla.vv(s32,s32,s32)           | 38.255 GOPS      | 1.847003 | 2       |
| asimd_int_mac   | mla.vs(s16,s16,s16)           | 76.503 GOPS      | 1.846859 | 2       |
| asimd_int_mac   | sqdmlal.vv(s32,s16,s16)       | 38.259 GOPS      | 1.847217 | 4       |
| asimd_int_mac   | sqdmlal2.vv(s32,s16,s16)      | 38.252 GOPS      | 1.846866 | 4       |
| asimd_tbl       | tbl.4table(u8)                | 27.641 GOPS      | 0.667284 | 4       |
| asimd_tbl       | tbx.4table(u8)                | 16.585 GOPS      | 0.400379 | 6       |
| sve_i8mm        | sve_mmla(s32,s8,s8)           | 662.87 GOPS      | 2.000278 | 1       |
| sve_i8mm        | sve_mmla(u32,u8,u8)           | 662.84 GOPS      | 2.000198 | 1       |
| sve_i8mm        | sve_mmla(s32,u8,s8)           | 662.83 GOPS      | 2.000159 | 1       |
| sve_i8mm        | sve_dp4a.vv(s32,s8,s8)        | 82.867 GOPS      | 2.000485 | 1       |
| sve_i8mm        | sve_dp4a.vv(s32,u8,u8)        | 82.854 GOPS      | 2.000172 | 1       |
| sve_i8mm        | sve_dp4a.vv(s32,u8,s8)        | 82.881 GOPS      | 2.000835 | 1       |
| sve_bf16        | sve_bfmmla(f32,bf16,bf16)     | 331.53 GFLOPS    | 2.000862 | 3       |
| sve_bf16        | sve_bfdot.vv(f32,bf16,bf16)   | 165.75 GFLOPS    | 2.000683 | 3       |
| sve_bf16        | sve_bfdot.vs(f32,bf16,bf16)   | 165.74 GFLOPS    | 2.000537 | 3       |
| sve_fp16        | sve_fmla.vv(f16,f16,f16)      | 152.99 GFLOPS    | 1.846681 | -       |
| sve_fp16        | sve_fmla.vs(f16,f16,f16)      | 152.96 GFLOPS    | 1.846279 | -       |
-------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
| Cache Level | Core Instruction | Bandwidth         | Theory Size | Test Size | Latency |
| L1 Cache    | ldp(f32)         | 39.379 Byte/Cycle | 64 KB       | 64 KB     |         |
| --------    | neon-ld1b(u8)    | 38.02 Byte/Cycle  |             |           |         |
| --------    | neon-ld1h(f16)   | 38.142 Byte/Cycle |             |           |         |
| --------    | neon-ld1w(f32)   | 38.817 Byte/Cycle |             |           |         |
| --------    | neon-ld1d(f64)   | 38.462 Byte/Cycle |             |           |         |
| --------    | sve-ld1b(u8)     | 64.034 Byte/Cycle |             |           |         |
| --------    | sve-ld1h(f16)    | 31.786 Byte/Cycle |             |           |         |
| --------    | sve-ld1w(f32)    | 64.036 Byte/Cycle |             |           |         |
| --------    | sve-ld1d(f64)    | 31.796 Byte/Cycle |             |           |         |
| L2 Cache    | ldp(f32)         | 26.63 Byte/Cycle  | 1024 KB     | 1024 KB   |         |
| --------    | neon-ld1b(u8)    | 26.584 Byte/Cycle |             |           |         |
| --------    | neon-ld1h(f16)   | 26.566 Byte/Cycle |             |           |         |
| --------    | neon-ld1w(f32)   | 26.625 Byte/Cycle |             |           |         |
| --------    | neon-ld1d(f64)   | 26.554 Byte/Cycle |             |           |         |
| --------    | sve-ld1b(u8)     | 29.85 Byte/Cycle  |             |           |         |
| --------    | sve-ld1h(f16)    | 23.942 Byte/Cycle |             |           |         |
| --------    | sve-ld1w(f32)    | 29.867 Byte/Cycle |             |           |         |
| --------    | sve-ld1d(f64)    | 23.861 Byte/Cycle |             |           |         |
------------------------------------------------------------------------------------------
--------------------------------------------
| Item                     | Theory | Test |
| L1 ways of associativity | 4      | 4    |
| cacheline size           | 64 B   | 64 B |
--------------------------------------------
--------------------------------------------------------------------------------------------------------
| Core ID | Theory Freq | Test Freq | IPC(FSU32) | IPC(FSU64) | IPC(LSU ldr) | IPC(SVE32) | IPC(SVE64) |
| 0       | 0 GHz       | 2.6 GHz   | 2          | 2          | 3            | 1.8        | 1.8        |
--------------------------------------------------------------------------------------------------------
-----------------------------------------------
| Item        | Core Instruction | IPC        |
| MULTI_ISSUE | ldr/fmla         | 6.2439 IPC |
-----------------------------------------------
```
