# SDPA Versions 与 Microkernel 实现说明

> ⚠️ **维护规则**：本文件是 fused_cpp SDPA 内核的真理来源。**每次新增 / 修改 / 删除 SDPA 版本或 MK trait 时，必须同步更新这份文档**——包括 [Changelog](#changelog) 与对应章节。
>
> 路径：`fused_cpp/csrc/SDPA_VERSIONS.md`
>
> 期望阅读顺序：先看[版本注册一览](#版本注册一览)定位你关心的版本，再读对应章节看具体优化。

---

## 目录

- [版本注册一览](#版本注册一览)
- [SDPA 顶层变体](#sdpa-顶层变体)
  - [`naive`](#naive)
  - [`flash1` / `flash2`](#flash1--flash2)
  - [`flash2_neon`](#flash2_neon)
  - [`flash2_neon_cache`](#flash2_neon_cache)
  - [`flash2_neon_l3kv`](#flash2_neon_l3kv)
  - [`flash2_neon_l3kv_packv`](#flash2_neon_l3kv_packv)
- [Microkernel Trait（MK）](#microkernel-traitmk)
  - [`MK_Baseline`](#mk_baseline)
  - [`MK_Scalar`](#mk_scalar)
  - [`MK_PQuad`](#mk_pquad)
  - [`MK_QkUblock4`](#mk_qkublock4)
- [选型矩阵](#选型矩阵)
- [Changelog](#changelog)

---

## 版本注册一览

每个 SDPA 顶层变体（`flash2_neon_cache` / `flash2_neon_l3kv` / `flash2_neon_l3kv_packv`）会和**每一个 enabled MK trait**（baseline / scalar / pquad / qk_ublock4）组合生成一组带后缀的版本名，外加一个不带后缀的「历史名」绑定到 baseline。当前注册的 20 个名字：

```
naive
flash1
flash2                              ← 老路径（flash2_neon 的别名 / 早期版本）
flash2_neon

flash2_neon_cache                   ← = _baseline
flash2_neon_cache_baseline
flash2_neon_cache_scalar
flash2_neon_cache_pquad
flash2_neon_cache_qk_ublock4

flash2_neon_l3kv                    ← = _baseline
flash2_neon_l3kv_baseline
flash2_neon_l3kv_scalar
flash2_neon_l3kv_pquad
flash2_neon_l3kv_qk_ublock4

flash2_neon_l3kv_packv              ← = _baseline（V 预 pack 后用 baseline microkernel）
flash2_neon_l3kv_packv_baseline
flash2_neon_l3kv_packv_scalar
flash2_neon_l3kv_packv_pquad
flash2_neon_l3kv_packv_qk_ublock4
```

新增 MK trait 或 SDPA 变体时，注册数量会按笛卡尔积扩张；具体注册位置见每个变体的 `csrc/sdpa_flash2_neon_*.cpp` 末尾。

---

## SDPA 顶层变体

### `naive`

最朴素的标量参考实现，三层 for 循环 + fp32 累加。

- **文件**：`csrc/sdpa_naive.cpp`
- **用途**：正确性参考、跨平台兜底；性能下界。
- **dtype**：fp32 / bf16 都支持（bf16 时内部 widen 到 fp32）。

---

### `flash1` / `flash2`

早期 FlashAttention-1 / FlashAttention-2 算法的纯标量实现，作为算法参考保留。日常路径不用。

- **文件**：`csrc/sdpa_flash1.cpp`、`csrc/sdpa_flash2_neon.cpp`（`flash2` 是 `flash2_neon` 的别名）
- **用途**：算法验证、和后续 NEON 路径数值比对的 baseline。

---

### `flash2_neon`

第一个走 NEON SIMD 的 FlashAttention-2 实现：单层 q tile 步长 8，online softmax 在 fp32 累加，QKᵀ / P̂·V 用 `gemm_qkt_8x8` / `gemm_pv_8x8` 等 microkernel。

- **文件**：`csrc/sdpa_flash2_neon.cpp`
- **优化**：8×8 microkernel + fp32 online softmax。
- **限制**：缺乏 KV tile 缓存调度，大 S 时 KV 反复从 L2/L3 流过来。

---

### `flash2_neon_cache`

在 `flash2_neon` 基础上引入 cache-aware 调度：

- **文件**：`csrc/sdpa_flash2_neon_cache.cpp`
- **优化**：
  1. **KV tile (Sc_l1)**：把 KV 沿 S 维分块，每个 q tile 在一个 Sc_l1 内连续扫，让 K/V 留在 L1。
  2. **双缓冲 PLDL2KEEP 预取**：在外层 KV tile 边界对下一段 K/V 发 L2 软件预取，避免 cold tile 的 L2 miss latency。
  3. **dispatch_dot_cache / dispatch_fma_acc_cache**：标量化的 P̂·V 兜底路径（NEON 不可用时）。
- **拓扑**：`omp parallel for collapse(3)` over (B, N, q_tile)，每个并行单元处理 8 行 Q。
- **限制**：q tile 只有 8 行，KV 在 q_tile 之间被反复加载——L3→L2 的搬运量没被摊薄。

---

### `flash2_neon_l3kv`

针对 long-context、KV 装不下 L3 的场景做的「L3-resident K/V」改进。**当前 PV-fp32 单核能稳到 ~106 GFLOPS（~98% fmla 峰值）**，在 fp32 路径上是默认推荐变体。

- **文件**：`csrc/sdpa_flash2_neon_l3kv.cpp`、`csrc/sdpa_flash2_neon_l3kv_impl.h`
- **优化**：
  1. **L3 多线程共享 K/V**（path B）：当 `S * (E + Ev) * sizeof_elt > l3_budget` 时，按 head 分组 → `omp parallel { omp single { ... omp taskloop nogroup ... } }`，让组内 worker 协同处理同一 (b, n) 的不同 q_tile，**同一份 K/V 对组内所有 worker 都驻留在 L3**。
  2. **`Lc_l2` 嵌套（path A 也用）**：外层 q_tile 步长从 8 改成 `Lc_l2`（≤64，按 Ev 自适应收紧），内层以 8 行为单位串行处理 `Lc_l2/8` 个 inner 组——**每个 L2 KV tile 加载一次后，多个 inner 组复用，L3→L2 搬运量降到 1/(Lc_l2/8)**。
  3. **L1 软件预取（PLDL1KEEP）**：在外层 KV tile 边界发 PLDL2KEEP（与 cache 一致），inner 组循环里**超前 2 步**发 PLDL1KEEP（PLDL1KEEP 需要 ~80–200 cycle 提前量）。
  4. **PV outer-loop 合并（2026-05-22）**：`process_q_tile_lc` 步骤 8 原本把每个 ev_off 块切成 `Sc_cur/8` 次 `pv_8x8(Sk=8)` 调用，每次调用都付 16 vld_O + 16 vst_O 的边界 round-trip。微内核内部已有完整 k 循环（4-way 软件流水 + 标量尾），完全能吃 `Sk=Sc_cur`。合并为单次调用后：每个 ev_off 块只付一次 O 累加器加载/写回，PV 内部软件流水跨整个 Sc_cur 不打断；非 packed 路径下 V cold-start cache miss 也只付一次。**bf16 端到端 +12–15%（非 packv），+5–8%（packv）；fp32 +1–3%（被 QKᵀ-fp32 单累加器瓶颈淹没）**。
  5. **P1 优化批次（2026-05-22）**：
     - **softmax row-max 向量化**：原标量 `for (j) if (sc_row[j] > m) m = ...` 长度 = Sc_cur 的标量依赖链，改成 `vmaxq_f32` 4-wide 累加器 + `vmaxvq_f32` 末尾规约。bf16 / fp32 都受益。
     - **`is_causal` / `mask_ptr` 模板特化**：把这两个原本在每个 inner 8 行组热路径里检查的运行期分支提到模板参数 `bool kHasMask` + `bool kCausal`，用 `if constexpr` 编译期消去 mask add / causal mask / causal_lim 初始化整段。SDPA 入口按 4 种组合分发模板实例。**模板实例化数从 12 翻到 48（3 MK × 2 dtype × 2 packv × 4 mask/causal），实例化数 4× 但每条热路径都更短**。
     - **mask add 向量化**：`for (j) sc_row[j] += m_row[j]` 改 `vld + vadd + vst` 4-wide 主体 + 标量尾。
     - **ev_off 切换 V cold prefetch（仅 packv 路径）**：packv 下 `ev_off += 8` = 跳到下一个 ev_block，物理地址跨 `S * 32` 字节（典型 S=2048 fp32 时 64KiB），远超 stride prefetcher 跟踪窗口。在 ev_off 循环开头对**下一个 ev_block** 起始 2 cache line 发 PLDL1KEEP，覆盖 LLC→L1 延迟。非 packv 路径不加（行内偏移，HW prefetcher 已 cover）。
     - **效果**：bf16 单线程 51 → 54-57 GFLOPS（74% → 78–83% 当前 BFMMLA-half 指令峰值）；fp32 ~不变（仍被 QKᵀ-fp32 单累加器卡住）。
- **拓扑**：path A = `omp parallel for collapse(3)`；path B = `omp parallel { single { taskloop } }`。
- **重构产物**：`process_q_tile_lc / run_path_collapse3 / run_path_taskloop` 三个内层模板已被抽到 `sdpa_flash2_neon_l3kv_impl.h`，与 `_packv` 变体共用，通过 `bool kPackedV / bool kHasMask / bool kCausal` 三个模板参数区分编译期路径。
- **数值等价性**：与 `flash2_neon_cache` / `flash2_neon` 在 fp32 下逐行严格相等；bf16 下与 cache 同等 ULP-级差异。row-max 向量化 / mask add 向量化在非 NaN 输入下与原标量循环位等价。模板特化（`if constexpr`）不改运行行为，只在编译期消除分支。

---

### `flash2_neon_l3kv_packv`

在 `l3kv` 基础上**SDPA 入口处先对 V tensor 做一次 pack**（rearrange，dtype 不变），把布局从 `[B, N, S, Ev]` 重排成 `[B, N, Ev/8, S, 8]`。目的是让 PV microkernel 内层沿 Sk 维扫 V 时，跨 k 行的 stride 从 `Ev`（典型 128 fp32 = 512 字节 = 8 cache line）降到 `8`（32 字节 = 半 cache line）。

- **文件**：`csrc/sdpa_flash2_neon_l3kv_packv.cpp`（共享 `sdpa_flash2_neon_l3kv_impl.h`）
- **优化**：
  1. **V layout 重排**：原 V 跨 k 行 stride = Ev，依赖 HW stride prefetcher 跨大跨度命中（受 PC stream tracker 槽数 / 大 stride 置信度 / 跨 4KiB page 强制 reset 影响，未必稳定）。pack 后跨 k 行 stride = 8，**每 2 行命中同一 cache line**，prefetch 由「下一行已在邻接 line 内」自然 cover。
  2. **多线程 pack**：`#pragma omp parallel for collapse(3) schedule(static)` over (b, n, ev_block)，内层 `std::memcpy(8 * sizeof(scalar_t))` 编译器内联成单条 `ldr q + str q`（fp32）/ `ldr d + str d`（bf16）。
  3. **dtype 保留**：bf16 → bf16、fp32 → fp32，pack 不 widen，减少访存数据量。
  4. **微内核行 stride 退化为编译期常量 `8`**：通过 `if constexpr (kPackedV)` 在 `process_q_tile_lc` 内显式传字面量，让编译器把 ldr 折成 imm offset。
- **限制**：`Ev % 8 == 0`（生产场景 head_dim_v ∈ {64, 128, 256} 全满足；不满足时 `TORCH_CHECK` 报错）。
- **代价**：pack 阶段需要遍历整个 V tensor 一次（读 + 写），bf16 V 总流量 `2 * B*N*S*Ev` 字节，多线程下 ~ms 量级；fp32 时翻倍。
- **数值等价性**：pack 是纯 memcpy bit-exact 重排，喂给 microkernel 的数据**逐字节相同**；fma 累加序不变，**端到端按位等价**。
- **实测端到端 wall-clock 提升（min ms）**：

  | dtype | shape | `l3kv_baseline` | `l3kv_packv_baseline` | 提升 |
  |---|---|---|---|---|
  | bf16 | small (1×8×64×64×64×64) | 0.238 | 0.221 | **+7.7%** |
  | bf16 | medium (1×8×256×256×64×64) | 3.75 | 3.44 | **+8.9%** |
  | bf16 | MLA-512 (1×16×512×512×192×128) | 64.85 | 59.50 | **+9.0%** |
  | bf16 | MLA-1k (1×32×1024×1024×192×128) | 514.9 | 495.4 | +3.9% |
  | fp32 | medium | 8.44 | 8.45 | ~0%（pack 开销翻倍抵消） |
  | fp32 | MLA-512 | 197.8 | 194.4 | +1.7% |

- **建议使用条件**：bf16 路径强烈推荐；fp32 收益微小，建议继续用 `flash2_neon_l3kv_pquad`。

---

## Microkernel Trait（MK）

`flash2_neon_cache` / `flash2_neon_l3kv` / `flash2_neon_l3kv_packv` 三个顶层变体都通过 `template <class MK>` 参数化，调用 `MK::qkt_8x8 / qkt_8x4 / qkt_tail / pv_8x8 / pv_tail` 5 个 trait 方法。每个 enabled MK trait 都会注册一份 SDPA 变体（`<topology>_<MK::kName>`）。

文件位置：`csrc/sdpa_microkernels/impls/mk_<name>.h`、`csrc/sdpa_microkernels/neon_cache_microkernels.h`（具体 NEON intrinsic 实现）。

### `MK_Baseline`

- **文件**：`csrc/sdpa_microkernels/impls/mk_baseline.h`
- **特点**：直接复用 `neon_cache_microkernels.h` 中的全局自由函数 `gemm_qkt_8x8` / `gemm_pv_8x8` 等。这些函数按 dtype 自动选 NEON / BFMMLA / 标量路径。
- **PV fp32 主路径**：`gemm_pv_microkernel_8x8_fp32` —— 16 个 O 累加器、k 维 4-way 软件流水 unroll、段间 V 预取（v_lo/v_hi 分两半）、跨迭代 V 预取（带边界保护）、稳态 28 NEON reg / 32（留 4 给编译器）。
- **PV bf16 主路径**：`gemm_pv_microkernel_8x8_bf16` —— V 即时 widen 到 fp32 + `vfmaq_n_f32`。
- **QKᵀ fp32 主路径**：`gemm_qkt_microkernel_8x8_fp32` —— ⚠️ **已知瓶颈**：实现是「8×8 个独立 dot product + 单累加器 + vaddvq」，单累加器 RAW dep chain 长度 = E/4，OoO 跨 (i,j) iteration 渲染只能并发 ~3 条 fma 链，**实测 ~8–11 GFLOPS（~10% 峰值）**。注释里声明的 16 个累加器全部 `(void)` 抛弃。**已有改进版**：见 [`MK_QkUblock4`](#mk_qkublock4)（4×4 双向分块 + 16 累加器 + `vpaddq` 树 reduce）。`MK_Baseline` 保留这条 dot product 实现作为参考。
- **QKᵀ bf16 主路径**：`gemm_qkt_microkernel_8x8_bf16` —— 默认走 BFMMLA；本机实测 BFMMLA half-rate（55 GFLOPS），切到 `BFMLALB/T` 可达 109 GFLOPS（**未来优化方向**，受控于 `FUSED_CPP_SDPA_CACHE_BF16_PATH_BFMMLA` 宏）。

### `MK_Scalar`

- **文件**：`csrc/sdpa_microkernels/impls/mk_scalar.h`
- **特点**：5 个 op 全部走三层 for 循环 + fp32 累加（`gemm_qkt_tail_scalar` / `gemm_pv_tail_scalar` 模板）。
- **用途**：跨平台等价性参考、SDPA 性能下界、无 NEON / BFMMLA 平台的最小验证目标。

### `MK_PQuad`

- **文件**：`csrc/sdpa_microkernels/impls/mk_pquad.h`
- **特点**：相对 `MK_Baseline` **唯一改动**是 fp32 PV 主体——其余 op（QKᵀ 全套、bf16 PV、tail）全部 fall through 到 baseline。
- **fp32 PV 优化**：`gemm_pv_microkernel_8x8_fp32_pquad`
  - **P̂ 标量 load → quad load + lane broadcast FMA**：每 4-k 段开头一次性 8 条 `vld1q_f32` 拿 P̂ 行 quad（替代每段 8 条 `ldr s` 标量 load），段内用 `vfmaq_laneq_f32(..., p_quad[i], lane)` 通过 lane 0/1/2/3 取段索引。
  - **LSU 指令数**：每 4-k 段从 40 (32 P 标量 + 8 V quad) 降到 16 (8 P quad + 8 V quad)，**2.5×↓**。
  - **FMA pipe 占用不变**：`vfmaq_laneq_f32` 与 `vfmaq_n_f32` 在 Neoverse 上是同一类 FMA 指令，IPC 3.89 没变。
  - **寄存器布局保持等同宽裕**：16 O acc + 8 P 行 quad + 4 V quad（本段 + 下段流水）= 28 / 32，留 4 给编译器。
- **数值等价性**：与 baseline 在 fp32 下**按位相等**——P 来源虽然从 `s_register` 换成 `v_register.s[lane]`，但 fp32 位精度不变，fma 累加序逐段一致。
- **实测**（L1-resident 微基准 / E=Sk=512 / target_frac=0.5）：
  - baseline pv_8x8 fp32：93.6 GFLOPS
  - pquad pv_8x8 fp32：**106.7 GFLOPS（+13.9%，~98% fmla 峰值）**
- **典型受益场景**：fp32 PV 8×8（当前唯一改动点）。bf16 PV 不受影响（仍走 baseline 的 widen + vfmaq_n 路径）；QKᵀ 不受影响（仍是 baseline 的 dot product 实现）。

### `MK_QkUblock4`

- **文件**：`csrc/sdpa_microkernels/impls/mk_qk_ublock4.h`
- **特点**：相对 `MK_Baseline` **唯一改动**是 fp32 QKᵀ 主体——其余 op（QKᵀ-bf16、QKᵀ 8×4、QKᵀ tail、PV 全套）全部 fall through 到 baseline。
- **fp32 QKᵀ 优化**：`gemm_qkt_microkernel_8x8_fp32_ublock4`
  - **8×8 输出切成 4 个 4×4 子块**：每个子块用 16 个独立 `float32x4_t` 累加器 `a00..a33` 同时沿 e 维累加，相比 baseline「64 个独立 dot product 单累加器」的实现，把 ILP 从 1（单链 RAW dep chain 长 E/4）拉到 16 条完全独立的 fma 链。
  - **数据访问 0 重排**：每个内层 e-step 只有 4 Q 行 vld1q + 4 K 行 vld1q = 8 条连续访存，**不需要 K 转置或 pre-pack**。Q 和 K 两边都按行内连续，所以走的还是 baseline 同样的输入 layout。
  - **vpaddq 树 reduce**：每个 4×4 子块结束时，4 个 quad → 1 个 quad（4 fp32 = scores 一行的 4 列），两层 `vpaddq_f32`；接着 vmulq_f32(scale) + vst1q_f32 写出。
  - **静态展开**：外层 2×2 块循环用宏展开成 4 份内核体（i_blk/j_blk 全是编译期常量），编译器把 ldr 偏移 fold 到指令立即数里。
  - **寄存器布局**：16 acc + 4 Q + 4 K + 临时 ≈ 25 / 32 NEON reg 稳态，留 7 给编译器做 software pipelining。
  - **指令吞吐分析**（按 4-pipe FMA、IPC≈3.89、4 cycle FMA latency 假设）：每内层 e-step 16 vfmaq + 8 vld1q ≈ 4 cycle（FMA-pipe 限），ILP 16 足以打满 latency 4 × pipes 4 = 16；剩余瓶颈在 LSU 带宽。
- **数值等价性**：与 baseline 在 fp32 下**接近等价但不按位相同**——`vfmaq + vpaddq tree-reduce` 的累加顺序与原 `vfmaq + vaddvq` 不同，存在 ULP-级误差，已被 SDPA 等价性测试容忍度覆盖（`atol=rtol=1e-4` fp32）。
- **预期增益**（待目标机重测，参考 `SDPA_TODO.md` 性能水位栏）：
  - QKᵀ-fp32 单核：10 → ~95–105 GFLOPS（**~10× 提升**）
  - fp32 SDPA wall-clock：15 → ~80–95 GFLOPS（**~5–6× 提升**）
- **典型受益场景**：fp32 SDPA。bf16 路径完全不动（仍走 baseline 的 BFMMLA / 标量兜底）。
- **可与 `MK_PQuad` 合用？**：当前两个 trait 互斥（各自单独改一个 op）；如需同时拿到「QKᵀ-fp32 ublock4」+「PV-fp32 pquad」两个改进，可参考它们的实现复制成一个新 trait（比如 `MK_QkUblock4Pquad`），代价是重复 ~5% trait 样板代码。后续合并请参考 `SDPA_TODO.md`。

---

## 选型矩阵

| 场景 | 推荐版本 | 备注 |
|---|---|---|
| 正确性参考 / 跨平台 | `naive` 或 `flash2_neon_cache_scalar` | 性能差，仅做正确性 |
| 短 S（≤ 512） + bf16 | `flash2_neon_l3kv_packv` | bf16 packv +9% 提升 |
| 短 S + fp32 | `flash2_neon_l3kv_qk_ublock4` | QKᵀ-fp32 重写后预期 ~10× 单核 GFLOPS |
| Long-context bf16（MLA prefill 等） | `flash2_neon_l3kv_packv` | +4–9%，pack 开销摊薄 |
| Long-context fp32 | `flash2_neon_l3kv_qk_ublock4` | 唯一打开 fp32 GEMM 瓶颈的路径；packv 在 fp32 下负收益 |
| 想跑 cache-aware 但 KV 装得下 L3 | `flash2_neon_cache_qk_ublock4`（fp32）/ `flash2_neon_cache_pquad`（bf16 PV 或 fp32 PV-only）/ `flash2_neon_l3kv_pquad`（path A 退化） | 性能近似 |
| 历史接口兼容 | `flash2_neon_cache` / `flash2_neon_l3kv`（无后缀别名） | 自动映射到 `_baseline` |

---

## Changelog

> 每次新增 / 修改 / 删除 SDPA 内核或 MK trait 时**必须**追加一条记录。格式：`日期 | 改动概述 | 受影响文件`。

| 日期 | 改动概述 | 受影响文件 |
|---|---|---|
| 2026-05-23 | **macOS cache probe 修正**：`probe_cache_bytes_one_level` 在 Apple Silicon 上的 sysctl key 顺序错了——`hw.l1dcachesize` / `hw.l2cachesize` 这种「无后缀」key 默认返回 **E-core 容量**（perflevel1，64 KB / 4 MB），先命中就 return 导致 tile sizing 一直按 E-core 算。修复：把 `hw.perflevel0.<level>cachesize`（P-core）放第一优先级，无后缀 key 作为 fallback。修复后探测到 P-core L1d=128 KB / L2=16 MB（之前是 64 KB / 4 MB）。**实测 wall-clock 持平**（bf16 BFMMLA-pegged + fp32 已 77% peak，tile 不是当前瓶颈），但移除了一个错误假设，是后续 P0-bf16 BFMLALB/T 切换 + tile size 重扫的前提 | 改 `csrc/sdpa_tile_sizes.h:101-106`（sysctl key 顺序） |
| 2026-05-23 | 新增 `MK_QkUblock4` trait：fp32 QKᵀ 8×8 主体从「64 个独立 dot product 单累加器」重写为「4×4 双向分块 + 16 独立累加器外积扇出 + vpaddq 树 reduce」，把 ILP 从 1 条链拉到 16 条独立 fma 链；不需要 K pack 或转置；与 baseline 在 fp32 下接近等价但不按位相同（已被 atol=rtol=1e-4 SDPA 等价性测试覆盖）。自动注册 `flash2_neon_cache_qk_ublock4` / `flash2_neon_l3kv_qk_ublock4` / `flash2_neon_l3kv_packv_qk_ublock4`。**本机实测**（Apple Silicon 单线程）：QKᵀ-fp32 8×8 微内核 7.6–10.1 → **104.7–106.5 GFLOPS（10.4–14.0×，96–98% fmla peak）**；R1-like fp32 SDPA wall-clock 13.9–15.2 → **80.2–84.2 GFLOPS（5.3–5.5×）**；bf16 路径完全不动（1.00× 实测） | 改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（新增 `gemm_qkt_microkernel_8x8_fp32_ublock4`）；新建 `csrc/sdpa_microkernels/impls/mk_qk_ublock4.h`；改 `csrc/sdpa_microkernels/all_impls.h`、`csrc/sdpa_microkernels/mk_registry.cpp`、`csrc/sdpa_flash2_neon_cache.cpp`、`csrc/sdpa_flash2_neon_l3kv.cpp`、`csrc/sdpa_flash2_neon_l3kv_packv.cpp` |
| 2026-05-22 | **P1 优化批次**：(1) softmax row-max 向量化（`vmaxq_f32` + `vmaxvq_f32`）；(2) `is_causal` / `mask_ptr` 模板特化（提到 `bool kHasMask` + `bool kCausal` 模板参数，热路径用 `if constexpr` 编译期消去 mask add / causal mask / causal_lim 整段，实例化数 12→48）；(3) mask add 内层循环 `vld + vadd + vst` 向量化；(4) packv 路径下 ev_off 切换时对下一个 ev_block 起始 2 cache line 发 PLDL1KEEP（非 packv 不加，行内偏移 HW prefetcher 已 cover）。bf16 SDPA 单线程 51 → 54–57 GFLOPS（74% → 78–83% BFMMLA-half 峰值）；fp32 ~不变（仍被 QKᵀ-fp32 单累加器卡住）。新增 `tests/test_sdpa_versions_equiv.py::test_sdpa_versions_equiv_with_attn_mask` 覆盖 mask 路径数值等价性 | 改 `csrc/sdpa_flash2_neon_l3kv_impl.h`、`csrc/sdpa_flash2_neon_l3kv.cpp`、`csrc/sdpa_flash2_neon_l3kv_packv.cpp`；改 `tests/test_sdpa_versions_equiv.py` |
| 2026-05-22 | **PV outer-loop 合并**：`process_q_tile_lc` 步骤 8 把 `Sc_cur/8` 次 `pv_8x8(Sk=8)` 调用合并成单次 `pv_8x8(Sk=Sc_cur)`——消掉 7×16=112 vld_O + 112 vst_O 的边界 round-trip + V cold-start cache miss。bf16 SDPA 单线程 47→51 GFLOPS（68%→74% peak），所有 8 个 SDPA 变体（cache / l3kv / l3kv_packv × baseline / scalar / pquad）都受益 | 改 `csrc/sdpa_flash2_neon_l3kv_impl.h` 步骤 8 |
| 2026-05-22 | 新增 `flash2_neon_l3kv_packv`（含 baseline / scalar / pquad 三个 MK 变体）：SDPA 入口 V layout 从 `[B,N,S,Ev]` 重排成 `[B,N,Ev/8,S,8]`，多线程 pack；`process_q_tile_lc / run_path_*` 抽到共享 header；bf16 +4–9% wall-clock | 新建 `csrc/sdpa_flash2_neon_l3kv_packv.cpp`、`csrc/sdpa_flash2_neon_l3kv_impl.h`；改 `csrc/sdpa_flash2_neon_l3kv.cpp` |
| 2026-05-20 | 新增 `MK_PQuad` trait：fp32 PV 8×8 P 标量 load 折成 quad load + lane FMA，LSU 指令数 -60%，PV +12–18% GFLOPS；自动注册 `flash2_neon_cache_pquad` / `flash2_neon_l3kv_pquad`（无 packv 时） | 改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（新增 `gemm_pv_microkernel_8x8_fp32_pquad`）；新建 `csrc/sdpa_microkernels/impls/mk_pquad.h`；改 `csrc/sdpa_microkernels/all_impls.h`、`csrc/sdpa_microkernels/mk_registry.cpp`、`csrc/sdpa_flash2_neon_cache.cpp`、`csrc/sdpa_flash2_neon_l3kv.cpp` |
