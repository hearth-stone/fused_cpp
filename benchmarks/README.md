# L1-Resident SDPA Microkernel Benchmarks

测试 `MK_Baseline / MK_Scalar` 等微内核 trait 的「单次调用工作集 ≈ 半个
L1d」吞吐量。第二次迭代起所有 Q/K/V/P̂/O 都驻留 L1，反映指令级吞吐而非
访存瓶颈。

## x86 dimension-aware fused-MoE policy

`bench_x86_bf16_policy.py` rotates the shared automatic backend and the two
forced ISA backends on identical logical inputs. Weight preparation is
reported separately, output is reused, and the timed order rotates each
iteration. JSON output includes the detected CPU family/model, policy profile,
chosen ISA, effective thread count, N-split target, route-skew decision, and
per-expert AMX pattern/cache windows.

```bash
taskset -c 0-7 env PYTHONPATH=src OMP_NUM_THREADS=8 \
  OMP_DYNAMIC=FALSE OMP_WAIT_POLICY=PASSIVE \
  .venv/bin/python benchmarks/bench_x86_bf16_policy.py \
  --tokens 64 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads 8 \
  --variants auto,avx512,amx --warmup 8 --runs 31
```

Use `FUSED_CPP_MOE_X86_ISA=avx512|amx` only to validate both execution paths
through the same ID-104 K32/N32 packed weights. Use
`FUSED_CPP_MOE_X86_POLICY_PROFILE=generic_v1|intel_06_ad_c8i_v1` for a
policy-profile A/B; neither variable is required in production.

## x86 AVX-512 small-M multi-N A/B

`bench_avx512_small_m_multi_n.py` rotates the original exact-M/single-N JIT,
W13-only, W2-only, combined wide-N, and automatic policies on identical
packed weights and tensors. It covers M=1--4, reuses output, excludes packing
and JIT warm-up, checks every result against PyTorch, and reports median,
p90, p99, mean, standard deviation, best, GFLOP/s, and speedup.

```bash
taskset -c 0 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_avx512_small_m_multi_n.py \
  --hidden 4096 --intermediate 512 --routes 1,2,3,4 \
  --threads 1 --variants baseline,w13,w2,multi_n,auto \
  --warmup 20 --runs 101
```

Use CPU sets `0-1`, `0-3`, or `0-7` with `--threads 2`, `4`, or `8` to
validate the cooperative-width guard. `FUSED_CPP_MOE_AVX512_SMALL_M_MULTI_N`
accepts `auto`, `baseline`, `w13`, `w2`, and `multi_n`; the benchmark manages
it per variant. C8i8 results and the rejected multi-core controls are in
`optimizations/fused_moe_avx512/results/amazon_c8i_8core_avx512_small_m_multi_n_20260726.md`.

## x86 AVX-512 JIT-internal Bulk-MN loops

`bench_avx512_bulk_mn.py` rotates the original per-M12/per-N-block calls,
W13-only and W2-only generated loops, the combined loop, and automatic policy
on identical packed weights and tensors. Packing, allocation, correctness
reference, and JIT warm-up are excluded. In addition to latency distribution
and GFLOP/s, it reports ratios of medians, ratios of means, and the median of
same-iteration paired ratios so scheduler bimodality is visible rather than
being mistaken for call-overhead improvement.

```bash
taskset -c 0 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_avx512_bulk_mn.py \
  --hidden 64 --intermediate 2048 --routes 48,96,256,2048 \
  --threads 1 --variants baseline,w13,w2,bulk_mn,auto \
  --warmup 16 --runs 201
```

Use CPU sets `0-1`, `0-3`, or `0-7` with the corresponding thread count for
cooperative N-split measurements. The script manages
`FUSED_CPP_MOE_AVX512_BULK_MN=auto|baseline|w13|w2|bulk_mn`; ordinary
production runs do not need the variable.

## x86 AMX MoE packed-B layout A/B

`bench_amx_bf16_layouts.py` compares the production N32 packed weights with
the explicit `x86_amx_bf16_n64` K32-streaming experiment. It prepares both
layouts once, alternates their timed order in one process, checks both against
the PyTorch expert reference, and reports prepack separately from inference.

```bash
taskset -c 0 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_amx_bf16_layouts.py \
  --routes 16,32,64,128,512,2048 \
  --patterns m1n2,m2n2,m1n4 --threads 1 --warmup 5 --runs 21
```

The result is JSON with median, p90, p99, best, mean, standard deviation,
GFLOP/s, and N64 speedup. Pin `0-7` and pass `--threads 8` for the cooperative
N-split comparison.

## x86 AMX N32 B-load-hint A/B

`bench_amx_bf16_patterns.py` also rotates the N32 `TILELOADD`,
`TILELOADDT1`, `PREFETCHT0`, and `PREFETCHT1` variants. The explicit modes
occupy separate JIT cache keys, reuse identical packed weights, and report
bit-exact differences against `tileloadd`.

```bash
taskset -c 0 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_amx_bf16_patterns.py \
  --tokens 512 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads 1 \
  --patterns auto \
  --b-load-hints tileloadd,tileloaddt1,prefetch_t0,prefetch_t1 \
  --warmup 5 --runs 31
```

Use one explicit `--b-load-hints` value and enough measured iterations under
`sudo perf stat` to compare `l1d_pend_miss.pending_cycles`,
`l1d.replacement`, and `topdown.memory_bound_slots`. The C8i8 result and exact
counter command are recorded in
`optimizations/fused_moe_avx512/results/amazon_c8i_8core_amx_b_load_hints_20260726.md`.

## x86 AMX m1n2 K-load pipeline A/B

The same pattern benchmark rotates cache-key-isolated baseline and ping-pong
K-loop schedules. Force `m1n2`: the other AMX patterns do not have a second
complete operand register bank.

```bash
taskset -c 0 env PYTHONPATH=src \
  .venv/bin/python benchmarks/bench_amx_bf16_patterns.py \
  --tokens 512 --hidden 4096 --intermediate 512 \
  --experts 1 --top-k 1 --routing hot --threads 1 \
  --patterns m1n2 --k-load-pipelines baseline,pipelined \
  --warmup 10 --runs 41
```

Compare against a separate `--patterns auto --k-load-pipelines baseline` run
before treating an m1n2-only speedup as an end-to-end policy improvement.
The C8i8 result, repeated-process methodology, and PMU command are recorded in
`optimizations/fused_moe_avx512/results/amazon_c8i_8core_amx_k_load_pipeline_20260726.md`.

| 入口 | 何时用 |
|---|---|
| `bench_microkernel_l1.py` | 日常 sweep，复用已编译好的 `_C` 扩展，输出友好 |
| `bench_microkernel_l1` (binary) | `perf record` / `objdump -d` / `lldb` 微观分析，无 Python 启动开销 |

两条入口的数值应在 ±5% 内一致；不一致大概率是 OMP 配置或 CPU 频率漂移。

## Python 入口

```bash
# 假设 fused_cpp 已通过 pip install -e . 安装
cd fused_cpp
python benchmarks/bench_microkernel_l1.py                       # 默认 baseline + bf16/fp32 + iters=100k
python benchmarks/bench_microkernel_l1.py --impl scalar         # 切到其他 impl
python benchmarks/bench_microkernel_l1.py --target-frac 0.25    # 工作集压到 25% L1
python benchmarks/bench_microkernel_l1.py --E 1024 --Sk 1024    # 手动指定形状
python benchmarks/bench_microkernel_l1.py --iters 500000 --warmup 5000
```

## 独立 C++ binary

```bash
cd fused_cpp/benchmarks
make                     # 输出 ./bench_microkernel_l1
make print-config        # 打印解析到的编译/链接标志，调试用

./bench_microkernel_l1
./bench_microkernel_l1 --dtype fp32 --target-frac 0.5
./bench_microkernel_l1 --E 1024 --Sk 1024 --iters 200000
./bench_microkernel_l1 --help
```

跨平台默认 ARCH 标志（与 `setup.py` 一致）：

| 平台 | 默认 |
|---|---|
| macOS (Apple Silicon) | `-mcpu=apple-m2` |
| Linux aarch64 | `-march=armv8.6-a+bf16+i8mm` |

覆写：

```bash
make FUSED_CPP_TARGET_CPU=neoverse-n1
make FUSED_CPP_TARGET_CPU=apple-m1     # 退到 widen+FMLA 路径
make CXX=clang++-17
```

## 输出示例

```
detected L1d = 65536 bytes (64.00 KiB)
target_frac = 0.50
iters = 100000, warmup = 2000

=== dtype=bf16  impl=baseline  E=1024  Sk=1024
    qkt_* working set = 32.25 KiB (50.4% of L1d)
    pv_*  working set = 48.25 KiB (75.4% of L1d)   ← P̂(fp32) 占大头
  op           M   N       K       us/iter      GFLOPS
  qkt_8x8      8   8    1024        1.0234       128.10
  qkt_8x4      8   8    1024        0.6532        80.32
  qkt_tail     5   3    1024        0.4123        72.18
  pv_8x8       8   8    1024        1.4823        88.42   ← widen+FMLA, 约 qkt_8x8 的 70%
  pv_tail      5   3    1024        0.4521        65.12
```

> 数字仅为示意。BFMMLA 主路径的 `qkt_8x8` 应显著高于 widen+FMLA 的 `pv_8x8`，
> 反映了 P̂·V 在 bf16 下的硬件天花板。

## 工作集模型

| op | 维度 | 工作集 (bytes) |
|---|---|---|
| `qkt_8x8 / qkt_8x4 / qkt_tail` | M=8, N≤8, K=E | `16 · E · sizeof(elt) + 256` |
| `pv_8x8 / pv_tail`              | M=8, N≤8, K=Sk | `Sk · (32 + 8 · sizeof(elt)) + 256` |

QKᵀ 工作集只与 E 和 dtype 有关；P̂·V 工作集偏大因为 P̂ 始终是 fp32（4 B/elt）
而非 bf16，是模板硬约束。
