# microkernel_min

最小化抽离自仓库的两个核心 bf16 microkernel，**纯 C++，无 Torch / Python 依赖**，
方便在不同 ARM 平台快速做正确性 + 性能验证。

## 抽离的 microkernel

| Kernel | 来源 | 角色 |
|---|---|---|
| `gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner` | `csrc/sdpa_microkernels/neon_cache_microkernels.h:1322` | bf16 QKᵀ，BFMMLA + B-major 调度 |
| `gemm_pv_microkernel_8x8_bf16_pquad` | `csrc/sdpa_microkernels/neon_cache_microkernels.h:2739` | bf16 PV，P 端 quad load + lane FMA + 跨迭代 V 预取（方案 A） |

附带 helper：
- `bf16_to_fp32_scalar` / `widen_bf16x4_to_fp32`
- `pack_q_8rows_to_seq_bf16` / `pack_k_8rows_to_seq_bf16`

## 平台要求

- **aarch64 + BFMMLA 指令**
- ✓ macOS Apple Silicon（M1/M2/M3，Apple clang）
- ✓ Linux Neoverse / 鲲鹏（gcc 10+，编译加 `-march=armv8.6-a+bf16`）

无任何外部依赖（不需要 PyTorch、BLAS、OpenMP）。

## 用法

```bash
# 一键 build + test + bench
./run.sh

# 或单独：
./build.sh
./microkernel_min test                              # 正确性测试
./microkernel_min bench                             # bench (默认 E=192 Sk=128)
./microkernel_min bench --E=128 --Sk=512            # 自定义 shape
./microkernel_min bench --iters=50000 --warmup=2000
./microkernel_min all                               # test + bench
```

## 输出示例

```
microkernel_min  (HAS_BFMMLA=1)
=== Correctness test ===
  [E=64  Sk=128] QKᵀ E=  64         max_abs=0.0123 max_rel=0.0042  PASS
  [E=64  Sk=128] PV  Sk= 128        max_abs=0.0421 max_rel=0.0095  PASS
  ...
=== 0 failure(s) ===
=== Benchmark (single-threaded, hot L1) ===
  E=192  Sk=128  iters=20000  warmup=1000
  -------------------------------------------
  QKᵀ packqk_seq4_bmajor (E=192)             64.84 GFLOPS
  PV  bf16_pquad           (Sk=128)            54.74 GFLOPS
=== Done ===
```

## 参考性能

| 平台 | QKᵀ packqk_seq4_bmajor | PV bf16_pquad | 备注 |
|---|---|---|---|
| 远程 ARM 服务器（92 GFLOPS bf16 peak） | ~64.8 GFLOPS / 70% peak | ~54.7 GFLOPS / 60% peak | 与仓库 `bench_pv_vs_qkt.py` 一致 |
| 本机 Apple Silicon P-core（55.8 BFMMLA half-rate） | 受 BFMMLA half-rate 约束，预期 ~30-40 | 受 fp32 FMA 上限约束，预期 ~50-60 | 仅参考 |

## 与主仓库的关系

本目录是**只读副本**。任何 microkernel 修改请回到主仓库的
`csrc/sdpa_microkernels/neon_cache_microkernels.h`，避免两份代码漂移。
本副本主要用途：
- 在没有 PyTorch 环境的机器上快速跑通 bench
- 隔离 microkernel 做端口/调度实验，不影响 SDPA build
- 给汇编诊断（`objdump -d`）一个干净的入口

## 文件

| 文件 | 说明 |
|---|---|
| `microkernels.h` | 两个 microkernel + pack helper |
| `reference.h` | naive C++ 参考实现（无 NEON） |
| `main.cpp` | test/bench/all 入口 |
| `build.sh` | 自动检测 macOS/Linux 选 CXX 和 flags |
| `run.sh` | 一键 build + test + bench |
