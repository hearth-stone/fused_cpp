## Remote Execution

Remote development and benchmark runs use the `Arm-codex` SSH alias.

### Connection

- Host alias: `Arm-codex`
- User: `zhangxu`
- Remote project root: `/home/zhangxu/codex/fused_cpp`
- Local project root: this repository root

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

Prefer the explicit Python executable in non-interactive SSH commands, for example:

```bash
ssh Arm-codex 'cd /home/zhangxu/codex/fused_cpp && .venv/bin/python -c "from fused_cpp import _C; print(_C.has_openmp())"'
```

Avoid relying on the remote system `python`; it may not match the virtualenv ABI used
to build `fused_cpp._C`.

### Benchmark Hygiene

For single-thread microbenchmarks, bind the process to core 80 with `taskset` and
pin both Python and native libraries to one thread:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c 80 <command>
```

Put `taskset -c 80` immediately before the Python or shell entrypoint, for example:

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
taskset -c 80 .venv/bin/python tests/bench_microkernel_qkt.py
```

When comparing measured GFLOP/s with peak, use the single-core reference values below.
If a reported single-thread result is above the relevant peak, first check whether the
thread count, OpenMP runtime, or benchmark FLOP formula is misleading.

## Remote Peak FLOPs Reference

The following numbers are remote-machine single-core peak measurements unless noted
otherwise. Use them as sanity checks for SDPA and microkernel benchmark results.

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
