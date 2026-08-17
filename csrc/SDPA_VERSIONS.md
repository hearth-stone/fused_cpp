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
  - [`flash2_neon_l3kv_packqkv`](#flash2_neon_l3kv_packqkv)
  - [`flash2_neon_l3kv_packqkv_pbf16pv`](#flash2_neon_l3kv_packqkv_pbf16pv)
- [Microkernel Trait（MK）](#microkernel-traitmk)
  - [`MK_Baseline`](#mk_baseline)
  - [`MK_Scalar`](#mk_scalar)
  - [`MK_PQuad`](#mk_pquad)
  - [`MK_QkUblock4`](#mk_qkublock4)
  - [bf16 QKᵀ pack microkernel 迭代记录](#bf16-qk-pack-microkernel-迭代记录)
- [选型矩阵](#选型矩阵)
- [Changelog](#changelog)

---

## 版本注册一览

每个 SDPA 顶层变体（`flash2_neon_cache` / `flash2_neon_l3kv` / `flash2_neon_l3kv_packv`）会和**每一个 enabled MK trait**（baseline / scalar / pquad / qk_ublock4）组合生成一组带后缀的版本名，外加一个不带后缀的「历史名」绑定到 baseline。`l1_bfmlal_layout` 是自由 layout 的实验 trait，目前只接入 packv L3KV 拓扑。当前注册的 23 个名字：

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
flash2_neon_l3kv_packv_l1_bfmlal_layout
flash2_neon_l3kv_l1_bfmlal_layout       ← = packv_l1_bfmlal_layout

flash2_neon_l3kv_packqkv            ← Q + K + V 预 pack，bf16 BFMMLA 专用，单一注册（无 MK 后缀）
flash2_neon_l3kv_packqkv_pbf16pv    ← softmax 直接产出 bf16 P scratch，PV 走 pv_8x8_pbf16
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
  2. **多线程 pack**：`#pragma omp parallel for collapse(3) schedule(static)` over (b, n, ev_block)，内层通过 `sdpa_pack_utils` 做 8 元素向量搬运；SVE 目标走 predicated `svld1/svst1`，非 SVE 目标退回常规 copy。
  3. **dtype 保留**：bf16 → bf16、fp32 → fp32，pack 不 widen，减少访存数据量。
  4. **微内核行 stride 退化为编译期常量 `8`**：通过 `if constexpr (kPackedV)` 在 `process_q_tile_lc` 内显式传字面量，让编译器把 ldr 折成 imm offset。
  5. **L1 BFMLAL 实验版本**：`flash2_neon_l3kv_packv_l1_bfmlal_layout` / alias `flash2_neon_l3kv_l1_bfmlal_layout` 绑定到 `MK_L1BfmlalLayout`。bf16 QKᵀ 8×8 通过 K_col 转成 BFMLAL lane 拓扑；bf16 PV 在 `process_q_tile_lc` 内把 `P_hat` scratch 转成 bf16 后调用 `pv_8x8_pbf16`。这是把 L1-only 85%+ peak 微内核接到端到端 SDPA 的第一版，仍会在外层支付 K_col pack 和 P_hat→bf16 转换成本。
     - Arm-codex-internal/Arm-codex core80 单线程，BGE-small-zh 形状 `B1-N8-L512-S512-E64-Ev64` bf16：non-causal 13.972 → **12.682 ms**（+10.2%），causal 15.176 → **14.210 ms**（+6.8%），对比基线均为 `flash2_neon_l3kv_packv`。
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

### `flash2_neon_l3kv_packqkv`

在 `l3kv_packv` 基础上**进一步把 K 也在 SDPA 入口处多线程 pre-pack**，并且**Q 在 `process_q_tile_lc_packqkv` 入口处一次性 pack 当前 q-tile**。Q-tile 内所有 `(s_l3, s_l2, qi_inner, s_off)` 组合**复用 packed Q，不重复 pack**。QKᵀ inner 走「4 条独立 `vld1q_u16` + B-major BFMMLA + 显式指针递增 + 2-way e-block unroll」路径（`gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner`），是当前 packqkv 的 bf16 QKᵀ 主实现。

- **文件**：`csrc/sdpa_flash2_neon_l3kv_packqkv.cpp`（fork `sdpa_flash2_neon_l3kv_impl.h` 的 `process_q_tile_lc_packqkv` / `run_path_collapse3_packqkv`）
- **pack 时机**：
  1. **V**：SDPA 入口多线程 `pack_v_to_evblock8`，[B,N,S,Ev] → [B,N,Ev/8,S,8]（与 `packv` 一致）
  2. **K**：SDPA 入口多线程 `pack_k_to_seq8`，[B,N,S,E] → [B,N,S/8,E_main/4,32 u16]，每个 8-row 子块就是 `pack_k_8rows_to_seq_bf16` 输出
  3. **Q**：每个 q-tile 进入 `process_q_tile_lc_packqkv` 时一次性 pack，写到 thread-local `q_seq_buf`；q-tile 内 `num_inner * num_S_tiles` 次 BFMMLA 共用一份 packed Q
- **microkernel**：
  - **QKᵀ**：直接调 `gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner`，bypass MK trait；2026-05-31 起 inner loop 用 `q_ptr/k_ptr` 递增并按 2 个 e-block 展开，减少 packed-offset 标量地址计算。partial K block（`s_global % 8 != 0` 或最后一个 < 8 行 q）自动 fall back 到 baseline `gemm_qkt_8x4 / gemm_qkt_tail` 读原始 K
  - **PV（2026-05-29 起）**：直接调 `MK_QkPackqkSeq4BmajorPvPquad::pv_8x8 / pv_tail`，bf16 路径走 `gemm_pv_microkernel_8x8_bf16_pquad`；2026-05-31 起在 BF16 arithmetic 目标上内部优先走 P_hat fp32→bf16 + `vbfmlalb/t_lane`，避开 V widen。**不模板化 MK**——packqkv 是为最优 bf16 组合写的专用 path，hardcode trait 与 hardcode kernel 等价
- **fp32 路径（2026-06-10 起）**：fp32 输入不再 delegate 到 `packv_pquad`，而是在入口全量 pack K 到 `[B,N,S/8,E,8]`，Q 保持原始 row-major；QKᵀ 8×8 改走 packed-K lane-FMLA microkernel（`MK_Fp32PackK8PQuad`），PV 继续走 fp32 pquad。该路径目标是把 QKᵀ 从 64 个独立 dot product 改成类似 PV 的 8×8 向量累加模式。
- **限制**：
  - **`S % 8 == 0` 且 `Ev % 8 == 0`**：`TORCH_CHECK` 失败时引导用户改用 `flash2_neon_l3kv_packv`
  - bf16 下 `E % 4 != 0` 时 partial e_block 不 pack，inner 标量 tail 自动从原始 K 读 [E_main, E)；与 baseline bit-for-bit 等价。fp32 packed-K path 的 tail 直接从 packed K lane layout 标量计算。
- **Path B 行为**：bf16 路径仍在 KV 不装 L3 且多线程时退化到 `packv_pquad`，避免 3× DRAM 带宽；fp32 packed-K path 会继续使用同一份全量 packed K/V，只把调度切到 taskloop。
- **数值等价性**：bf16 QKᵀ BFMMLA 累加顺序与 baseline 完全一致；PV bf16 BFMLAL 快路径会把 P_hat 临时 round 到 bf16，非按位等价但误差很小（Arm-codex-internal/Arm-codex 小形状 SDPA 对 naive max_abs 0.0078125）。fp32 packed-K QKᵀ 改变累加顺序，不再与旧 `packv_pquad` byte-exact，但应满足 fp32 SDPA 容差。
- **本机端到端实测**（Apple Silicon P-core，OMP=1，B=1 N=8 E=192 Ev=128，bf16）：
  - 改 PV pquad 前（baseline PV）：L=1024 ≈ 54.5 GFLOPS（同 `packv_pquad`）
  - 改 PV pquad 后：L=256 57.6 / L=512 58.4 / **L=1024 59.3 GFLOPS**，端到端 +7~9% vs baseline；L=128 因 pack overhead 占比高略亏 ~2%
- **建议使用条件**：bf16 prefill long-ctx（L=S ≥ 256），且 KV 装 L3，且 `S % 8 == 0` 且 `Ev % 8 == 0`。短 S（pack overhead 摊不薄）继续用 `flash2_neon_l3kv_packv_pquad`。

---

### `flash2_neon_l3kv_packqkv_pbf16pv`

在 `flash2_neon_l3kv_packqkv` 的 Q/K/V pre-pack 拓扑上，softmax 仍用 fp32 计算 row sum / running sum，但把每个 full 8-row q block 的 `P_hat` 直接写成 bf16 scratch，然后 PV 调 `MK_QkPackqkSeq4BmajorPvPquad::pv_8x8_pbf16`。尾块仍回退 fp32 `P_hat` + `pv_tail`。fp32 输入走 2026-06-10 新增的 packed-K lane-FMLA QKᵀ + PV pquad 路径。

- **文件**：`csrc/sdpa_flash2_neon_l3kv_packqkv.cpp`；核心分支在 `process_q_tile_lc_packqkv<..., kPbf16PV=true>`
- **数值语义**：softmax 归一化分母保持 fp32；只有写给 PV 的概率矩阵降为 bf16。BGE-small-zh 形状旧/新最大绝对差约 `0.0009765625`。
- **Arm-codex-internal/Arm-codex core80 单线程实测**：
  - BGE-small-zh attention 形状 `B1-N8-L512-S512-E64-Ev64` bf16：non-causal `10.776 → 9.989 ms`（`50.80 → 54.79 GFLOPS`，+7.3%）；causal `12.389 → 11.317 ms`（`22.13 → 24.23 GFLOPS`，+9.5%）。
  - 长序列 MLA 形状 `B1-N32-L2048-S2048-E192-Ev128` bf16：non-causal `1422.704 → 1276.547 ms`（`60.85 → 67.82 GFLOPS`，+11.5%）；causal `952.715 → 867.951 ms`（`45.46 → 49.89 GFLOPS`，+9.8%）。
- **profile 组成**：BGE non-causal 下 PV `4.548 → 3.353 ms`，softmax `1.963 → 2.586 ms`；长序列 non-causal 下 PV `577.535 → 429.397 ms`，softmax `129.362 → 169.733 ms`。因此收益主要来自 PV，代价是 softmax 阶段多一次 bf16 store / conversion。

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
- **特点**：相对 `MK_Baseline` 改动**PV 全套**（fp32 + bf16），其余 op（QKᵀ 全套、tail）fall through 到 baseline。
- **fp32 PV 优化**：`gemm_pv_microkernel_8x8_fp32_pquad`
  - **P̂ 标量 load → quad load + lane broadcast FMA**：每 4-k 段开头一次性 8 条 `vld1q_f32` 拿 P̂ 行 quad（替代每段 8 条 `ldr s` 标量 load），段内用 `vfmaq_laneq_f32(..., p_quad[i], lane)` 通过 lane 0/1/2/3 取段索引。
  - **LSU 指令数**：每 4-k 段从 40 (32 P 标量 + 8 V quad) 降到 16 (8 P quad + 8 V quad)，**2.5×↓**。
  - **FMA pipe 占用不变**：`vfmaq_laneq_f32` 与 `vfmaq_n_f32` 在 Neoverse 上是同一类 FMA 指令，IPC 3.89 没变。
  - **寄存器布局**：16 O acc + 8 P 行 quad + 4 V quad（本段 + 下段流水）+ 2 跨迭代 V 预取（段 2 中段暂存）= 30 / 32。
  - **2026-05-29 调度优化（方案 A）**：把跨迭代 V[k+4] 预取从段 3 末尾移到段 2 中段。原排布下，跨迭代 V load 到下一轮段 0 第 1 条 FMA 只有 2 条 FMA 距离（~1 cycle），覆盖不了 L1 vld1q 5 cycle 延迟；新排布拉到 ~16 条 FMA（~8 cycle），完全隐藏。段 3 改为纯 FMA 段，消除跨迭代 OoO 阻塞。
- **bf16 PV 优化**：`gemm_pv_microkernel_8x8_bf16_pquad`
  - **2026-05-29 精确 pquad fallback**：P-端 quad load + lane FMA + 4-k 外展；V 端从 baseline 的 `2× vld1_u16 + 2× widen` 改成 `1× vld1q_u16 + vshll_n_u16(lo) + vshll_high_n_u16(hi)`。每 4-k 段 LSU 从 36 降到 12，数值与 baseline 等价。
  - **2026-05-31 BFMLAL 快路径**：BF16 arithmetic 目标上，P_hat 每 4-k 段用 `vcvt_bf16_f32` 临时 round 到 bf16，然后对 bf16 V 走 `vbfmlalbq_lane_f32 / vbfmlaltq_lane_f32`。主体避开 V widen；输出先在 even/odd 列布局累加，末尾 `zip` 回 row-major。
  - **数值语义**：fp32 PV 仍按位相等；bf16 BFMLAL 快路径因 P_hat round 到 bf16，非按位等价，但 `validate_microkernel` E=17/64/128/192、Sk=19/128 的 `pv_8x8_max_abs` 约 1.6e-6；小形状 SDPA 对 naive max_abs 0.0078125。
- **实测**（L1-resident 微基准）：
  - **本机 P-core**（109 GFLOPS fp32 peak / E=Sk=512 / target_frac=0.5）：
    - baseline pv_8x8 fp32：93.6 GFLOPS → pquad pv_8x8 fp32：**106.7 GFLOPS（+13.9%，~98% peak）**（方案 A 前数据；待重测）
  - **远程机器**（92 GFLOPS peak / E=192 / Sk=128 / iters=20000）：
    - baseline pv_8x8 fp32：63.86 → pquad pv_8x8 fp32：**72.28 GFLOPS（+13%，79% peak）**
    - baseline pv_8x8 bf16：44.27 → 2026-05-29 pquad：54.74 → 2026-05-31 BFMLAL：**58.68–59.52 GFLOPS（64–65% peak）**
- **典型受益场景**：bf16 PV 8×8（远程机器累计 +33~35% vs baseline）和 fp32 PV 8×8（远程机器 +13%，本机 +14%）。QKᵀ 不受影响（仍是 baseline 的 dot product 实现）。

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

### `MK_L1BfmlalLayout`

- **文件**：`csrc/sdpa_microkernels/impls/mk_l1_bfmlal_layout.h`
- **定位**：evaluation-only / L1-only 上限评估，不代表当前 SDPA 外层 layout。目标是假设 microkernel 可以自由选择输入布局，验证 bf16 QKT / PV 在 L1 工作集内能否达到 85% 单核峰值。
- **QKᵀ bf16 layout**：Q 保持 `Q[8][E]` row-major；K 预转置成 `K_col[E][8]`，即 reduce 维优先。这样每个 reduce step 可以用一条 `vld1q_bf16` 取 8 个 K 列，QKT 在计算形态上变成 `P_bf16[8,E] @ V_bf16[E,8]`。
- **QKᵀ bf16 kernel**：`gemm_qkt_microkernel_8x8_bf16_qrow_kcol_bfmlal`。主体每 4 个 E lane 载入 8 行 Q 的 `bfloat16x4_t` 和 4 条 K_col `bfloat16x8_t`，用 `vbfmlalbq_lane_f32 / vbfmlaltq_lane_f32` 累加 even/odd 列，末尾 `zip` 回 row-major 并乘 `scale`。benchmark 中 `qkt_8x8_kcol` 直接计时 K_col 已由 caller 提供的纯 L1 layout；`qkt_8x8` 仍保留 thread-local cache 版本。
- **PV bf16 layout**：沿用 `pv_8x8_pbf16` 上限路径，假设 softmax 已经直接产出 bf16 P scratch，V 为 `V[Sk][8]` 连续布局；该路径只评估 microkernel 上限，不改变当前真实 fp32-P PV 入口。
- **Arm-codex-internal/Arm-codex core80 单线程实测**（`OMP_NUM_THREADS=1 taskset -c 80`，E=Sk=768，iters=1,000,000，单核峰值按 92 GFLOPS）：
  - `qkt_8x8_kcol`：82.64 GFLOPS，约 **89.8% peak**。
  - `pv_8x8_pbf16`：82.62 GFLOPS，约 **89.8% peak**。
- **正确性**：`validate_microkernel("l1_bfmlal_layout", "bf16", E, Sk)` 在 E/Sk=17/64/512 上 `qkt_8x8_max_abs=0`、`qkt_8x8_kcol_max_abs=0`、`pv_8x8_pbf16_max_abs=0`；默认真实 `pv_8x8` 仍是 P_hat fp32→bf16 的 pquad BFMLAL 路径，`max_abs≈1.6e-6`。
- **结论**：在允许重排输入布局、且工作集常驻 L1 的条件下，bf16 QKT 和 PV microkernel 均已超过 85% of 92 GFLOPS。下一步若要接入外层，需要分别解决 K_col layout 的生产/复用成本，以及 softmax 直接产出 bf16 P scratch 的数值与存储策略。

### bf16 QKᵀ pack microkernel 迭代记录

> 本节记录 `tests/bench_microkernel_qkt.py` 中的 bf16 QKᵀ 8×8 microkernel evaluation traits。除 `baseline` / `qk_ublock4` 等已注册 SDPA 版本外，下列 `qk_pack*` / `qk_unroll2` 变体默认 **未通过 `REGISTER_SDPA_VERSION` 接入 SDPA 主路径**，只用于 microkernel benchmark。表中的 `*` 表示 evaluation-only。

#### 共同背景

- 目标 op：`scores[8][8] = scale * Q[8,E] @ K[8,E]^T`。
- bf16 主路径：每个 `e_block = e:e+4` 用 16 条 `vbfmmlaq_f32` 更新 4×4 个 2×2 子块累加器。
- baseline 的 BFMMLA 操作数构造：Q/K 各自 `8 × vld1_u16` + `4 × vcombine_u16`，把相邻两行拼成一个 `bfloat16x8_t`。
- seq pack layout：`[E/4][32 u16]`，每个 e_block 正好 64B：`row0[4], row1[4], ..., row7[4]`。
- row-pair pack layout：`[4 pair][E*2 u16]`，每个 pair 块内按 `[row0[4], row1[4]]` 交错。

#### 当前变体总表

| impl | 文件 / trait | pack 对象 | packed layout | load 形式 | BFMMLA 顺序 | 是否接入 SDPA | 结论 |
|---|---|---|---|---|---|---|---|
| `baseline` | `MK_Baseline` | 无 | 原 Q/K row-major | `8×vld1_u16 + 4×vcombine`（Q/K 各一套） | A-major | 是（`flash2_neon_cache_baseline` 等） | 正确性 / 性能基线 |
| `qk_packk_full *` | `MK_QkPackkFull` | K | row-pair | K: `4×vld1q_u16` | A-major | 否 | 每次调用都 pack K，pack overhead 完整计入，稳定负收益；只做下界 |
| `qk_packk_inner *` | `MK_QkPackkInner` | K | row-pair | K: `4×vld1q_u16` | A-major | 否 | **K-only 第一代最优**；K cache 可按 `(K_ptr,stride,E)` amortize，是最接近 SDPA 可落地的候选 |
| `qk_packk_seq *` | `MK_QkPackkSeq` | K | seq `[E/4][32]` | K: `vld1q_u16_x4` | A-major | 否 | 在目标机上基本无收益；说明 `x4` multi-reg load 不适合该 kernel |
| `qk_unroll2 *` | `MK_QkUnroll2` | 无 | 原 Q/K row-major | baseline load | A-major，e 维 2-way unroll | 否 | 负收益；寄存器压力 / issue 冲突大于 loop overhead 收益 |
| `qk_packqk_seq *` | `MK_QkPackqkSeq` | Q + K | seq | Q/K: `vld1q_u16_x4` | A-major | 否 | Q/K 双 pack 但双侧 `x4`，在目标机上负收益或不稳定 |
| `qk_packqk_seq4 *` | `MK_QkPackqkSeq4` | Q + K | seq | Q/K: `4×vld1q_u16` | A-major | 否 | **Q+K 双 pack 第一代最优**；证明 Q 侧 pack 在 microkernel 层有收益 |
| `qk_packqk_seq4_ptr *` | `MK_QkPackqkSeq4Ptr` | Q + K | seq | Q/K: `4×vld1q_u16` | A-major | 否 | 指针递增替代 `(e/4)*32`；实测与 seq4 持平，编译器已做强度削减 |
| `qk_packqk_seq4_bmajor *` | `MK_QkPackqkSeq4Bmajor` | Q + K | seq | Q/K: `4×vld1q_u16` | **B-major** | 否 | **当前 microkernel 总体最优**；2026-05-31 起 inner loop 追加显式指针递增 + 2-way e-block unroll |
| `qk_packqk_seq4_pipe_a *` | `MK_QkPackqkSeq4PipeA` | Q + K | seq | Q/K: `4×vld1q_u16` | K 全 load；Q 分批 load + 立即算 A 行组 | 否 | 退化明显；load-use latency 暴露，调度方向错误 |
| `qk_packqk_seq4_pipe_b *` | `MK_QkPackqkSeq4PipeB` | Q + K | seq | Q/K: `4×vld1q_u16` | Q 全 load；K 分批 load + 立即算 B 列组 | 否 | 好于 pipe_a，但不如先全 load 再 B-major；仍有 load-use latency |
| `l1_bfmlal_layout *` | `MK_L1BfmlalLayout` | K | K_col `[E][8]` | K: `vld1q_bf16` per e；Q: `vld1_bf16` per row quad | BFMLAL lane | 否 | **L1-only 新最优**；自由 layout 下 QKT 变成 PV-like BFMLAL，E=Sk=768 的 `qkt_8x8_kcol` 达 82.64 GFLOPS（89.8% peak） |

#### 每轮迭代最优版本

| 迭代 | 新增 / 对比内容 | 当轮最优 | 目标机代表结果 | 判断 |
|---|---|---|---|---|
| 0. baseline | 原始 BFMMLA QKᵀ | `baseline` | E=1024,Sk=1024：59.36 GFLOPS | 基线 |
| 1. K-only pack | `qk_packk_full` / `qk_packk_inner` / `qk_packk_seq` | **`qk_packk_inner`** | E=1024,Sk=1024：65.61 GFLOPS / 1.11× | K row-pair + 4 条独立 `vld1q_u16` 稳定正收益；`full` 因每次 pack 负收益；`seq+x4` 不如 inner |
| 2. Q+K 双 pack + `x4` | `qk_packqk_seq` | `qk_packk_inner` 仍胜 | E=1024,Sk=1024：`qk_packqk_seq` 53.48 GFLOPS / 0.90× | 双侧 `vld1q_u16_x4` 在目标机上不适合该 kernel |
| 3. Q+K 双 pack + 4×1 load | `qk_packqk_seq4` | **`qk_packqk_seq4`** | E=1024,Sk=1024：70.28 GFLOPS / 1.18× | 4 条独立 `vld1q_u16` 明显优于 `x4`；Q 侧 pack 在 microkernel 层有额外收益 |
| 4. seq4 调度变体 | `ptr` / `bmajor` / `pipe_a` / `pipe_b` | **`qk_packqk_seq4_bmajor`** | E=1024,Sk=1024：73.24 GFLOPS / 1.23× | B-major 顺序额外 +4.2%；指针递增无收益；pipe_a/pipe_b 因 load-use latency 不如全 load 后 compute |
| 5. bmajor 地址计算削减 | 在 `qk_packqk_seq4_bmajor` 内合并 `ptr` 思路并做 2-way e-block unroll | **`qk_packqk_seq4_bmajor`** | Arm-codex-internal/Arm-codex core80，Sk=128：E64 53.44→57.00 / E128 60.34→64.78 / E192 65.07→69.05 GFLOPS | 对 packqkv 组合 trait 的 QKᵀ 主路径提升 +6.1%~+7.4%；PV 未改，仍约 55 GFLOPS |
| 6. L1-only 自由 layout | K 改为 reduce-major `K_col[E][8]`，QKT 改走 BFMLAL lane 拓扑 | **`l1_bfmlal_layout`** | Arm-codex-internal/Arm-codex core80，E=Sk=768：`qkt_8x8_kcol` 82.64 GFLOPS / PV pbf16 82.62 GFLOPS，均约 89.8% peak | 不考虑外层接入成本时，bf16 QKT/PV microkernel 已超过 85% of 92 GFLOPS；后续外层化要解决 K_col 和 bf16 P scratch 的生产成本 |

#### 目标机 E=1024,Sk=1024 单点数据

> 下表是 2026-05-27 的 E=1024,Sk=1024 历史单点，用于保留 pack layout / 调度选择依据；2026-05-31 的最新改动针对 `qk_packqk_seq4_bmajor` 内循环地址计算和 2-way e-block unroll，已在上方 Sk=128 目标形状记录。

| impl | GFLOPS | vs baseline | 备注 |
|---|---:|---:|---|
| `baseline` | 59.36 | 1.00× | 原始 bf16 BFMMLA 路径 |
| `qk_packk_full *` | 52.72 | 0.89× | 每次 pack K，负收益 |
| `qk_packk_inner *` | 65.61 | 1.11× | K-only row-pair + 4×1 load，当前最现实的 SDPA 接入候选 |
| `qk_packk_seq *` | 60.37 | 1.02× | K seq + `x4`，基本无收益 |
| `qk_packqk_seq *` | 53.48 | 0.90× | Q/K seq + 双 `x4`，负收益 |
| `qk_packqk_seq4 *` | 70.28 | 1.18× | Q/K seq + 双 4×1 load，第一代 Q+K pack 最优 |
| `qk_packqk_seq4_ptr *` | 70.27 | 1.18× | 与 seq4 持平 |
| `qk_packqk_seq4_bmajor *` | **73.24** | **1.23×** | 当前 microkernel 最优 |
| `qk_packqk_seq4_pipe_a *` | 59.31 | 1.00× | 退化到 baseline |
| `qk_packqk_seq4_pipe_b *` | 68.13 | 1.15× | 好于 pipe_a，仍不如 bmajor |
| `qk_unroll2 *` | 54.71 | 0.92× | 负收益 |

#### 选型结论

- **microkernel 峰值（当前 SDPA-like pack 约束）**：选 `qk_packqk_seq4_bmajor`。
- **microkernel 峰值（L1-only / 自由 layout）**：选 `l1_bfmlal_layout`；端到端实验入口是 `flash2_neon_l3kv_packv_l1_bfmlal_layout`（短别名 `flash2_neon_l3kv_l1_bfmlal_layout`），但它仍支付 K_col pack 和 P_hat→bf16 scratch 成本。
- **更可能接入 SDPA 的方向**：优先把 `qk_packk_inner` 的思想推进到 K-only 版本（后续建议试 `qk_packk_inner_bmajor` / `qk_packk_seq4_bmajor`），因为 K pack 可 amortize，Q pack 在 SDPA 主循环中每个 q-tile 都变化，端到端未必能摊薄 pack overhead。
- **load 指令选择**：目标机上优先 `4×vld1q_u16` / offset loads；避免 `vld1q_u16_x4` 作为主路径；避免 post-index load。
- **调度选择**：先 load 全部 Q/K operand，再 B-major 发射 BFMMLA；不要把 load 与 compute 紧贴（`pipe_a` / `pipe_b`）。

---

### x86 AMX MoE W13 SiLU 微内核迭代记录

> 本节按项目统一 kernel 治理要求记录非 SDPA 的 x86 MoE 微内核尝试；
> 完整命令和逐 shape 数据见
> `optimizations/fused_moe_avx512/results/amazon_c8i_2core_amx_silu_20260720.md`。

| variant | 实现 | 数值 | Amazon C8i 结果 | 结论 |
|---|---|---|---|---|
| `baseline` | 每行从常量表重复 broadcast，逐行 `VDIVPS` | 参考 | M16/degree5 JIT 3972 B | 保留为显式回退 |
| `resident` | prologue 一次载入 11 组 ZMM 常量，逐行运算序不变 | 与 baseline BF16 bit-exact | JIT 2836 B（-28.6%）；K256 W13 +8.4%；H4096 full expert 0.995x~1.016x | **默认 auto** |
| `pipelined` | resident 基础上按 stage 交错两行独立链 | 与 baseline BF16 bit-exact | H4096 full expert 未稳定优于 resident | 保留实验，不默认 |
| `rcp14` | 两行 pipeline，`VRCP14PS` 代替 `VDIVPS` | 非 bit-exact；仍满足现有 BF16 容差 | full expert 差异不超过测量噪声 | 近似实验，不默认 |

结论：原逐行代码在 JIT 中已经完全展开，乱序核能跨后续行重叠部分
`VDIVPS`，所以显式两行 stage pipeline 收益有限；常量驻留的稳定价值主要是
缩小代码体积和改善短 K epilogue。下一优先级转向 W2 store/workspace，而不是
继续牺牲精度替换除法。

---

### x86 AMX MoE W2 store/merge epilogue 迭代记录

> 完整命令、W2-only、merge-only 和端到端逐 shape 数据见
> `optimizations/fused_moe_avx512/results/amazon_c8i_2core_amx_w2_epilogue_20260720.md`。

| variant | 实现 | 数值 | Amazon C8i 结果 | 结论 |
|---|---|---|---|---|
| `baseline` | TMM→2 KiB scratch→逐行 ZMM→flat-route FP32 | 参考 | 全 shape 稳定 | **默认 auto** |
| `combined` | M1N4 四个 TMM→4 KiB scratch；每个 N64 row 只算一次 route address | 与 baseline BF16 bit-exact | K512 W2-only M256 +2.1%，M2048 -2.6%；K32 effective output BW +15.8%~+17.8%；端到端 -1.0%~+0.6% | 保留实验，不默认 |
| `tile_store` | TMM 直接写 expert-contiguous FP32 rows；merge 经 flat-route→row map | 与 baseline BF16 bit-exact | K512 W2-only +1.0%~+14.5%；K32 latency +12.8%~+22.8%；mapped merge 0%~+11.8% latency；端到端 -4.5%~+1.7% | 保留实验，等待 dimension-aware policy |

结论：去掉 TMM→L1 scratch→ZMM 的回读能稳定提升独立 W2，但完整 expert 的
收益取决于 M/K、route 布局、merge cache state 和线程调度；K32 sweep 还确认
wide-stride `TILESTORED` 的 raw store path 比 scratch→ZMM store 更慢。当前没有一个静态策略
覆盖所有 case，因此 `auto` 继续使用 baseline；top-k=1 direct BF16 无论选择哪个
实验模式都保留 ZMM FP32→BF16 转换。下一优先级是 scratch/workspace 生命周期，
不是把 shape-dependent 的 tile-store 路径提前设为默认。

### Sparse MLA production head-major scalable QK/PV

`flash_mla_sparse_fwd` 的 BF16 默认实现使用 combined head-major：当 8-query
block 数大于请求线程数的一半时，每个 token 建立一个任务，QK/PV 的 M 维为同一
token 的 8 个 query head；每组 gathered MQA K/V 只 pack 一次并由所有 head
group 复用。全 query 共享连续 indices 时先走调用级 K/V pack 的 dense-head
子路径。

SVE BF16 build 的 sparse gathered 与 fully-shared dense 子路径现都使用
scalable BFMMLA QK/PV `8×2VL`：`N=svcntb()/2`，所以 SVL128 为 8×8、
SVL256 为 8×16；两者都用 16 个 FP32 累加器，并在每 K=4 发出 16 条
BFMMLA。sparse QK 直接 pack gathered K，dense QK 则在调用级一次性 pack
连续 K；PV 以 NEON 4×8 transpose 直接生成 scalable V B-layout，softmax
产生的 BF16 P 再 pack 为 2-row A panels。K 尾部补零，SVL256 的 8-column
key/Dv 半 tile 用零填充和谓词保护。非 SVE build 保留完整 fixed NEON 8×8
路径。

dense-head 与 sparse-head 的 packed K/V 来源及外层调度保持分离，但两者的
`QK -> score materialization -> online softmax -> BF16 P -> P pack -> PV`
中间计算统一由同一个内部 chunk executor 执行。sparse 的 ragged/重复索引、
sink，以及 `return_stats` 的精确 `std::exp` 统计仍留在 sparse 语义层；dense
仍从在线状态直接产出 stats。因此该重构不改变公开 API、默认 dispatch、packed
layout、浮点运算顺序或数值语义。

V4 select-all prefill 的 compressed cache 与 causal/SWA cache 表现为最多两段
固定起点、逐 token 增长的连续前缀。SVE BF16 head-major 现检测该结构，把每段
最大范围按运行时 2VL layout 在调用级只 pack 一次；每个 token 仍只计算自身的
有效前缀，不引入 masked union 的额外 QK/PV。检测不依赖硬编码的 2048 阈值；
query-dependent TopK、滑动后的 SWA 起点、低复用、非整 2VL 最大范围和 non-SVE
均回退原 per-token gather-pack。Amazon 8C、SVL256、前 2048 V4-like
`q[2048,32,192]`、compressed 512 + causal/SWA 2048、`d_v=128`：1T
552.018→451.861 ms（-18.14%）；8T 的反向顺序稳态复测
82.702→76.929 ms（-6.98%）。后续 `context_start=10000`、window 128 的
query-dependent sparse 在 1T/8T 为 +0.04%/-0.20%。两种 VL 各 26 tests
passed；完整方法和 raw samples 见
`optimizations/sparse_mla/results/amazon_8c_shared_prefix_pack_20260814.md`。

后续 query-dependent sparse 路径不具备跨 token 共享 pack 的条件，但同一 KV
行的 V 是 K 的前 `d_v` 维。SVE indexed pack 因此改为每 4 行、每 8 维只加载
一次：同一组 NEON vectors 同时写两个 K=4 BFMMLA panels，并经 4×8 transpose
写 V panel。相较原先独立 K pack 与 V pack，输入 KV payload 从每行
`d_qk+d_v` 降为 `d_qk`，packed layout、QK/PV、online softmax 与 non-SVE
fallback 均不变。Amazon 8C、SVL256、`q[2048,32,192]`、`topk=640`、
`d_v=128`、`context_start=10000`：1T 280.552→254.908 ms（-9.14%）；
8T 两种执行顺序分别为 41.584→39.161 ms（-5.83%）和
41.671→39.155 ms（-6.04%）。单次 profile 的 pack 阶段
61.209→37.819 ms（-38.21%）；共享前缀路径 8T 仅 -0.18%。SVL256/SVL128
完整文件各 26 tests passed。完整方法与 raw samples 见
`optimizations/sparse_mla/results/amazon_8c_indexed_single_pack_20260814.md`。

Amazon 8C 的 DSV4-like `q[2048,32,192]`、top-k capacity 640、`d_v=128`，
5 warmup + 21 samples：fixed 8×8 的 1T/8T 为 252.176/38.508 ms；SVE QK
在 SVL128 为 250.760/38.313 ms，在 SVL256 为 225.232/36.759 ms，后者分别
降低 10.68%/4.54% latency。强制 SVL128 与默认 SVL256 各 24 项 focused tests
通过；生成 QK reduction loop 无 SVE spill。完整命令、raw samples 与限制见
`optimizations/sparse_mla/results/amazon_8c_scalable_sve_20260814.md`。

第二阶段 commit `58881f8` 只替换 sparse QK、保留原 PV。sparse 最终阶段在
相同 workload 把完整 SVE QK/PV 的 SVL128 1T/8T 降到
201.848/28.257 ms，相对 fixed
8×8 降低 19.96%/26.62%；SVL256 为 180.951/26.817 ms，降低
28.24%/30.36%。该阶段记录的 PV reduction loop 无 spill，最终 epilogue 曾有
三个临时 Z spill；曾尝试从旧 O 初始化 BFMMLA 累加器以删除显式 add，但 spill
增至四个 Z，1T 179.869→181.225 ms（+0.75%），故撤回。

dense 8×2VL 阶段保持同一 token 的 8 个连续 Q head 为 M，把调用级连续 K/V
改 pack 为 scalable B-layout，并直接复用同一 QK/PV kernel。Amazon 8C、
`q[2048,32,192]`、共享 `KV=640`、`d_v=128`、5 warmup + 21 samples：
SVL128 的 1T/8T 从 fixed 8×8 的 334.450/54.206 ms 降为
250.042/36.384 ms（-25.24%/-32.88%）；SVL256 从 332.588/54.248 ms
降为 213.853/32.610 ms（-35.70%/-39.89%）。相同最终 binary 的 gathered
sparse 回归检查在四个 VL/thread 点均未回退。最终 build 的 QK/PV 各有 16 条
静态 `bfmmla` 且均无 Z spill，PV 只有 ABI `d8--d15` 保存/恢复；强制
SVL128/256 各 24 tests passed。完整统计、raw samples 与命令见
`optimizations/sparse_mla/results/amazon_8c_dense_scalable_sve_20260814.md`。

短 query chunk 继续走 guarded 2D indexed fallback；FP32 或不满足
`Hq%8==0、Dqk%4==0、Dv%8==0` 的 BF16 形状也走 indexed fallback。公开 API、
sink、stats、`out=` 与数值容差不变。旧 `_flash_mla_sparse_fwd_variant` 比较入口、
masked/pruned/fused tail 实现和 variant benchmark 已从 active tree 删除，由
manifest tombstone、结果文档与 Git commit 保存。

### Sparse MLA 连续尾块 8×8（已退役历史）

indexed fallback 的 BF16 调度带有 2D KV 分片；未触发 split 时使用 4×4
尾部计算。当 8-query
block 数不超过请求线程数的一半时，
最重的 block 最多拆成 8 个互斥 KV shard；每个 shard 产出局部 online-softmax
`(m,l,O)`，再按 max-shift 公式合并。query-block 并行度已经足够时不分片、
不分配 partial state，也不启动 merge OpenMP 区域。fp32 路径保持原 1D 调度。

planner 同时支持 sliding recent window：相邻 8 行起点不同的连续 run 会提取
共同交集为 dense segment，左右 fringe 留给 indexed/masked 路径。这样 128-token
recent window 在完整序列上形成梯形，而不是退化成大量 4×4 gather tile。

历史比较入口 `masked_dense_8x8` 与 `masked_dense_8x8_pruned` 曾让 planner 提升
完整 8-query、单调连续前缀且可安全加载 8 行 K/V 的 ragged tail；重复、离散、
partial block 及越界风险继续回落 indexed 路径。

- full masked：复用已有 packed-Q/K BFMMLA 8×8 与 pbf16 PV，softmax 前按行
  `valid_len` mask；同一 head 的 packed Q 在 dense/masked segment 间复用。
- pruned：对已知 causal/compressed/uniform mask 模板化；标准 causal tile 的 QK
  从 16 个 2×2 BFMMLA 子块减到 10 个，PV 从 64 个 P 系数减到 36 个 BFMLAL
  更新。初版逐 block 标量落盘和 E/4 循环没有收益，随后改为 E/8 双步展开、
  2-lane 向量落盘，并加入精确 PV lane pruning。
- Amazon 192C NUMA1、BF16 `q[2048,32,192]`、`kv[2560,1,192]`、`d_v=128`、
  5 warmup + 21 次轮转交错中，1T latency 为 586.217 / 544.225 / 541.370 ms
  （indexed/full/pruned）；12T 为 55.710 / 55.832 / 55.949 ms，32T 为
  54.702 / 55.402 / 55.500 ms，96T 为 55.281 / 55.426 / 55.352 ms。
  单线程快 7.2--7.7%，但 ≥12T 已到约 55 ms 平台且没有收益，故不切默认。

为避免把全算子稀释误判为尾部核无收益，另用 1,024 个独立 8-query block
做 tail-only 隔离。纯 causal 下 pruned 相对 full dense 在 1/12/32/96T 分别
快 12.9% / 11.2% / 8.9% / 5.0%；按 V4 比例混合 causal 与四种 compressed
mask 时分别快 5.8% / 5.8% / 5.1% / 4.7%。因此 2×2 QK 与 exact-lane PV
pruning 本身有效；2048 forward 无收益来自 masked tail 仅占有效 pairs 的
0.644%，而不是 pruned 微内核没有减少尾部开销。

### Sparse MLA fused online epilogue 与 PV 指令比较（已退役历史）

新增三个内部实验入口：`masked_dense_8x8_fused_fmla`、
`masked_dense_8x8_fused_bfmlal`、`masked_dense_8x8_fused_bfmmla`。三者共用
pruned BFMMLA QK 和 SVE online-softmax，score 与 BF16 `p_hat` 不再写入 scratch；
差别只在 PV。实现仅对 SVL128/256 开启，其他 SVL 明确回落 materialized 路径。

1,024-block causal tail-only、单线程、BF16 `q[8192,32,192]`、`d_v=128`、
5 warmup + 21 次轮转交错结果：

| 机器 / SVL | materialized pruned | fused FMLA | fused BFMLAL | fused BFMMLA | 结论 |
|---|---:|---:|---:|---:|---|
| Amazon 192C CPU96 / 128 | 27.361 ms | 29.532 ms | 28.267 ms | 46.403 ms | materialized 最快 |
| Amazon 8C CPU0 / 256 | 55.238 ms | 45.385 ms | 49.876 ms | 49.505 ms | FMLA 最快 |

所以“消除中间往返”不是跨机器恒胜：SVL256 的 fused FMLA 比 materialized 快
17.84%，但 SVL128 慢 7.93%。BFMMLA 还要支付 probability replication、专用 V
pack 和 2×2 output interleave/deinterleave；K=8 太小，矩阵指令数减少不足以覆盖
固定开销，两台机器都不是最优。三个 fused 入口均已从 active tree 删除。

### Sparse MLA 真实 DSV4 长上下文与 2D 调度

`dsv4-sparse` benchmark 使用两个不重叠的 cache 区域：4:1 compressed cache 中
最多 512 个排序但离散的选择（4a 不是三角），以及达到 128 后滑动的 recent
window（完整序列呈梯形）。Amazon 192C NUMA1 cores 96--191、96T、BF16
`h_q=32,d_qk=192,d_v=128`、5 warmup + 21 次轮转交错：

| query / context | indexed 1D | guarded 2D | speedup / change |
|---|---:|---:|---:|
| 64 / end 4096 | 13.600 ms | 2.558 ms | 5.317× |
| 128 / end 4096 | 13.876 ms | 3.584 ms | 3.872× |
| 256 / end 4096 | 14.657 ms | 6.439 ms | 2.276× |
| full 4096 | 51.269 ms | 51.156 ms | -0.22% latency |
| full 8192 | 138.608 ms | 138.866 ms | +0.19% latency |

短 chunk 的 2D 输出相对 1D max abs 为 0.00195312；full 4096/8192 因门限
关闭 split，输出 bitwise equal。8-core SVL256 上 64-token 为 26.278/26.298 ms，
说明同一门限不会在小线程池误拆。完整方法、命令、raw-sample 结论、数值覆盖与
下一步见 `optimizations/sparse_mla/results/amazon_sparse_mla_2d_fused_20260812.md`。

### Sparse MLA head-major dense 8x8（已采用的子路径历史）

新增内部 `heads_dense_8x8`：全 query 共享同一段连续 indices 时，QK/PV 的
8×8 M 维从“同一 head 的 8 个 token”改成“同一 token 的 8 个 head”。K/V
仍只在调用级 pack 一次，Q 和输出则直接消费公开 `[token,head,dim]` 中相邻的
head 行。该机制现已成为 production combined head-major 的 dense 子路径。

BF16 `q[2048,32,192]`、共享 `KV=640`、`d_v=128`、3 warmup + 11 次轮转交错：
Amazon 8C 的 1T 为 335.131→332.821 ms（-0.69%），8T 为
54.648→54.658 ms（持平）；Arm-codex NUMA0 的 1T 为
421.224→423.739 ms（+0.60%），0--79 共 80T 为 7.325→5.936 ms
（-18.96%，1.234×）。两机运行时 SVL 均为 256 bit，各 37 项 sparse MLA
测试通过，benchmark 输出 bitwise equal。

单核没有算术收益，80T 收益来自调度粒度：旧路径只有 256 个 8-token task，
80 workers 必然出现 3/4 task 不均；新路径有 2,048 个 token task，分配为
25/26。8C 下两种 task 数均可整除线程数，因此持平。详细命令、配置与限制见
`optimizations/sparse_mla/results/arm_head_major_20260813.md`。

这里记录的是 head-major 调度采用时的 fixed 8×8 compute 历史；当前 SVE build
已在相同调度上把 dense QK/PV 替换为 8×2VL，non-SVE 才继续使用 fixed 8×8。
新 compute 结果见
`optimizations/sparse_mla/results/amazon_8c_dense_scalable_sve_20260814.md`。

### Sparse MLA head-major sparse fixed 8x8（已采用的基线历史）

新增内部 `heads_sparse_8x8`：对 BF16 MQA 的每个 token，把有效 sparse
indices 对应的 K/V gather+pack 一次，再由同 token 的所有 8-head group 复用；
QK 使用现有 packed BFMMLA 8×8，PV 使用 BF16-probability 8×8，并在 L2
chunk 之间保留同一份 online-softmax max/sum/O 状态。负索引 padding 被跳过，
正索引顺序及重复语义不变，sink/output/stats 与 unsupported-shape fallback 均保留。
组合候选先尝试 `heads_dense_8x8`，因此全 query 共享连续 KV 时仍只做一次调用级
K/V pack；该组合调度现为公开 `flash_mla_sparse_fwd` 的 BF16 默认实现，sparse
QK 微内核随后已由上面的 scalable SVE 8×2VL 取代。

BF16 `q[2048,32,192]`、DSV4 512 compressed + 128 sliding-window、
`context_start=0`、`d_v=128`、3 warmup + 11 次轮转交错：Amazon 8C 的
1T 为 360.442→251.737 ms（-30.16%，1.432×），8T 为
55.664→38.682 ms（-30.51%，1.439×）；Arm-codex 的 1T 为
384.633→327.617 ms（-14.82%，1.174×），0--79 共 80T 为
10.262→4.704 ms（-54.16%，2.182×）。全 dense 组合版与 dense-only head
版本相差不超过 0.14%，Arm-codex 80T 相对 token-major 为
7.318→6.014 ms（-17.82%）。两机实际 SVL256、各 43 tests passed；sparse
输出 max abs 0.015625，max/LSE 通过 1e-5 容差。详细方法、raw-sample 范围、
诊断占比与保留实验状态的原因见同一结果文件。

完整方法、逐轮尝试、数值结果与保留门限见
`optimizations/sparse_mla/results/amazon_192c_masked_tail_20260812.md`。

### DeepSeek V4 sparse indexer weighted-ReLU 8×2VL 候选

Indexer 长路径的目标公式改为逐 head 的
`sum_h weight[h] * ReLU(dot(q[h], key))`；旧实现先把 weighted Q 沿 head
求和再做一次 GEMM，不仅无法在 ReLU 前交换求和次序，也与模型定义不等价。

新增 scalable-SVE `8 heads × 2VL keys` BFMMLA 候选。每个 tile 使用 16 个
FP32 BFMMLA 累加器，Q 每个 query 在线程私有缓冲中 pack 一次；四个 score
向量跨全部 head blocks 保持在寄存器中。每组两个 head 的 weight 生成
`[w0,w0,w1,w1]`，在寄存器内完成 ReLU、加权累加和最终 head reduce，之后
只散写最终 FP32 score。K 不再先 gather 成 FP32 tensor，而是在 packB 时按
paged-cache row offset 直接抽取 BF16。支持条件为 contiguous BF16 Q/K、FP32
weights、`H%8==0`、`D%4==0`；其他形状使用等价的逐 head FP32 fallback。
`n_tile=svcntb()/2`，因此同一份源码随运行时 SVL 变化，不固定为 128 或 256 bit。

Amazon 8C Neoverse-V1、cores 0--7、SVL256 上，45 个随机/尾块 score 组合
（`H={8,16,64},D={4,16,128},N={1,7,16,19,33}`）最大绝对误差
`9.53674e-7`、最大相对误差 `2.6939e-6`；相关集成测试 25 passed/5 个退休
`cpp_v0` 用例 skipped。生成汇编的 8×2VL 热内核没有 Z 寄存器 spill。

生产 score 形状 `M2048/H64/D128/N1536`、3 warmup + 15 runs：1T 为
202.285 ms / 254.8 GFLOP/s，8T 为 36.653 ms / 1.406 TFLOP/s（5.52×，
69.0% 并行效率）。完整 post-GEMM `context_start=4096,topk=512` 的 8T
中位数 186.251 ms；profile 中 direct packB 0.044 ms、score 36.154 ms、
逐行 TopK 75.155 ms，瓶颈已转为 TopK。旧 weighted-Q fold 公式错误，不能作为
等价性能 baseline。当前仍保持 candidate：按用户要求本轮只验证了 8-core
SVL256，192-core/SVL128 待测。完整命令和样本范围见
`optimizations/deepseek_v4_post_gemm/results/amazon_8c_indexer_weighted_relu_sve_20260813.md`。

### DeepSeek V4 Indexer native batched exact TopK 候选

把 weighted-ReLU score 后的逐行 `at::Tensor::topk` 循环替换为单次 native
batched exact TopK。每个 OpenMP worker 只建立一个可复用 `score,index` 缓冲，
每行先用 `std::nth_element` 分出最大的 K 项，再只对选中项按 score 降序排序，
最终直接写 strided int32 index；NaN 视为最大，tie 按较小 local index 排序。
CPU contiguous-row FP32 score 与无重叠 int32 输出自动使用 native 路径，其他
tensor contract 保留原 ATen fallback。score 计算与完整 logits 物化本轮不变。

Amazon 8C Neoverse-V1、cores 0--7、SVL256 上，独立 TopK 的 20 个
empty/`K={0,1,4,20}`/tie/NaN/strided-output case 全过；score 的 45 个形状仍为
最大绝对误差 `9.53674e-7`，集成测试 `25 passed, 5 skipped`，并覆盖两 request
长路径及 stride-2 输出。

生产形状 `M2048/H64/D128/N1536`、`context_start=4096,topk=512`、3 warmup +
15 runs：完整 stage 中位数 `186.251→120.758 ms`（-35.16%，1.542×）；profile
中的 TopK `75.155→9.637 ms`（-87.18%，7.799×），score
`36.154→36.158 ms`（+0.01%），score+TopK `111.310→45.795 ms`（-58.86%，
2.431×）。另一次同配置复测中位数 `120.940 ms`、TopK `9.656 ms`，结果稳定。
仅完成用户要求的 8-core 首测，保持 candidate，192-core/SVL128 待测。完整命令、
范围和 raw summary 见
`optimizations/deepseek_v4_post_gemm/results/amazon_8c_indexer_native_batch_topk_20260813.md`。

### DeepSeek V4 Indexer select-all 提前短路

原有短路在两个 Q GEMM、Indexer Q RoPE/weight scaling 和两个 compressor
全部完成后，才根据 `max(cu_seqlen_ke-cu_seqlen_ks) <= topk_tokens` 判断每行
TopK 实际会选择全部有效位置。因此 `context_start=0` 等形状虽然 score/TopK 已经
短路，仍会无效计算完整的 Indexer Q 分支。

现把 `ks/ke` 转换和最大有效长度判定提前到 Q 调度之前。命中时 raw-weight 与
prepacked 两个入口都只计算 Main Q；prepacked 即使强制 shared-Q pool，也绕过
Main/Indexer pair 而调用 Main-Q-only 路径。MLA/Indexer 两个 compressor 仍按原
顺序更新，select-all indices 也仍在 compressor 之后写入；未命中的长路径复用
同一份计划并保持原 score/TopK 实现。公开签名、packed layout、输出和数值语义
不变。

Amazon 8C Neoverse-V1、cores 0--7、SVL256、NEON Q GEMM，`M=2048`、
`context_start=0`、`topk=512`、3 warmup + 15 runs：完整 stage 中位数
`74.309→40.176 ms`（-45.93%，1.850×），profile total
`74.135→39.074 ms`（-47.29%）；Indexer Q GEMM + RoPE/weight scaling 的
`34.063+0.791=34.854 ms` 变为 0，Main Q `33.429→32.989 ms`。raw 与
prepacked（强制 shared pool）两个专项测试通过，完整 post-GEMM 文件
`20 passed`，并覆盖长路径。192-core/SVL128 主机 SSH 无响应，未冒充测试结果；
该机运行验证仍待补。完整命令、样本范围和回滚边界见
`optimizations/deepseek_v4_post_gemm/results/amazon_8c_indexer_select_all_early_exit_20260813.md`。

### Online-softmax exp packed-lane constants Lab experiment

Compared the existing broadcast-constant degree-5 Horner exp with a
dependency-broken Estrin form. The candidate keeps range-reduction constants
in one NEON register and the six polynomial coefficients in two registers;
generated GCC 15 assembly was checked for indexed `FMUL/FMLS/FMLA` lane forms.
Both paths sweep U1/U2/U4/U8 so their different dependency depths are not
confused with constant-loading cost.

On Amazon M5 Neoverse V3, NUMA1 core 96, SVE128, length 2048, three sessions of
21 samples, isolated best-to-best was Horner U4 0.605697 versus packed/Estrin U8
0.571827 ns/element (+5.92% throughput). With 16 SVE accumulators forced live
across softmax, best-to-best was 0.613147 versus 0.579812 ns/element (+5.75%).
Matched U8 improved throughput 12.66% and reduced hot-loop Q spill pairs from
14 to 5, while both saved eight Z accumulators across softmax. Best-to-best is
not spill-free because the candidate requires deeper unrolling. Both variants
had max relative error 3.2874e-6, max 39 ULP, and 76 BF16 mismatches out of
1,048,570 versus rounded `std::exp`. No production path changed; full method is
in `optimizations/sparse_mla/results/amazon_m5_softmax_exp_constants_20260816.md`.

### Sparse MLA QK score-copy/max fusion checkpoint

The shared `run_heads_qkpv_chunk_bf16` path now accumulates each head's chunk
maximum while copying the just-produced `8x2VL` QK tile into score scratch.
This removes the later max-only score scan while retaining score storage for
exp, the same online-softmax chunk order, row-major BF16 P, P pack, and PV.

M5 NUMA1 cores 96--191/SVL128, 96 threads, ten warmups + 21 samples in three
paired sessions: 2048 shared-prefix median-of-session medians changed 11.626 to
11.484 ms (-1.22%); later sparse changed 2.606 to 2.537 ms (-2.65%). The 8192
shared-dense host state alternated between roughly 8.9--9.2 and 10.3--10.6 ms
independently of binary/order, so no 8192 claim is made. M5 direct checks passed
for dense/shared-prefix/later-sparse; Amazon 8C native SVL256 full Sparse MLA
file passed 26 tests. This is an experimental sequence checkpoint below the
standalone 3% gate; direct packed-P generation is evaluated next. Full data is
in `optimizations/sparse_mla/results/amazon_m5_score_copy_max_20260817.md`.

### Shared-prefix token-panel L2 scheduling experiment

Tested a MoE-like macro schedule for the first-2048 shared-prefix path: group
adjacent input tokens into a thread-private panel, keep packed Q and online
`(m,l,O)` plus score/P scratch in L2, and move the shared packed-K/V chunk loop
outside the token loop. Token panels remain the primary parallel dimension;
the existing short-query KV split remains the secondary source of parallelism.
Each token retained its original segment/chunk order.

Amazon 8C, SVL256, `q[2048,32,192]`, compressed 512 + causal/SWA 2048,
`d_v=128`, 5 warmups + 21 samples: panel 4 reduced 1T latency in two orders
from 454.085/453.853 to 447.179/447.129 ms (-1.52%/-1.48%), and 8T from
76.714/76.838 to 75.983/76.165 ms (-0.95%/-0.88%). Panel 8 was -1.43% at 1T
but +1.06% at 8T. Forcing the existing 8-query token-major fallback was
827.890/137.295 ms at 1T/8T, 82%/79% slower than head-major, although this also
changes the scalable microkernel path. Panel 4 passed shared-prefix output/stats
tests and all timed checksums matched. The experiment missed its 3% gate, so
the source was removed and the production token-major-first/KV-second dispatch
was left unchanged. M5 was not measured because the migrated `/data` worktree
had no project Python/PyTorch environment in the first pass.

The requested M5 follow-up used native SVL128 on NUMA1 cores 96--191 and kept
the `8x2VL` QK/PV microkernels unchanged. It scanned token panels `{4,8,16}`,
B blocks `{64,128,256}`, fixed single-scratch storage, plus panel8/B256 double
scratch. No point approached the gate. In the paired panel8/B256/single run,
2048 shared-prefix changed 11.635 to 11.619 ms (-0.14%), 8192 shared-dense
changed 9.222 to 9.250 ms (+0.30%), and later low-overlap sparse changed 2.606
to 2.611 ms (+0.19%). PMU sampling showed 2048 L2 data refills down 12--16%
and bus accesses down 8--11%, but duration improved only 0.24% and LLC read
misses increased; at 8192, L2 refills were unchanged and duration regressed.
M5 has no exposed uncore/DDR PMU, so bus-access times 64 bytes is reported only
as a traffic proxy. The prototype was removed. Full data is in
`optimizations/sparse_mla/results/{amazon_8c_token_panel_cache_20260817.md,amazon_m5_token_panel_cache_20260817.md}`.

---

## 选型矩阵

| 场景 | 推荐版本 | 备注 |
|---|---|---|
| 正确性参考 / 跨平台 | `naive` 或 `flash2_neon_cache_scalar` | 性能差，仅做正确性 |
| 短 S（≤ 512） + bf16 | `flash2_neon_l3kv_packv` | bf16 packv +9% 提升 |
| 短 S + fp32 | `flash2_neon_l3kv_qk_ublock4` | QKᵀ-fp32 重写后预期 ~10× 单核 GFLOPS |
| Long-context bf16（MLA prefill 等） | `flash2_neon_l3kv_packv` | +4–9%，pack 开销摊薄 |
| L1-only BFMLAL 接入实验 | `flash2_neon_l3kv_l1_bfmlal_layout` | K_col QKT + bf16 P scratch PV；用于验证最新微内核端到端接入成本 |
| **Long-context bf16，KV 装 L3，S%8==0** | **`flash2_neon_l3kv_packqkv`** | Q + K + V 都 pre-pack，BFMMLA inner 走 packqk_seq4_bmajor 路径（microkernel +23%）；端到端预期 +2–5% vs `packv` |
| Long-context fp32 | `flash2_neon_l3kv_qk_ublock4` | 唯一打开 fp32 GEMM 瓶颈的路径；packv 在 fp32 下负收益 |
| 想跑 cache-aware 但 KV 装得下 L3 | `flash2_neon_cache_qk_ublock4`（fp32）/ `flash2_neon_cache_pquad`（bf16 PV 或 fp32 PV-only）/ `flash2_neon_l3kv_pquad`（path A 退化） | 性能近似 |
| Sparse MLA BF16 | 公开 `flash_mla_sparse_fwd`（head-major scalable QK/PV） | SVE sparse gathered 与 fully-shared dense 路径均按 token 做 8-head×2VL-column QK/PV；短 chunk 最多 8-way KV split；non-SVE/unsupported shape 保留回退 |
| 历史接口兼容 | `flash2_neon_cache` / `flash2_neon_l3kv`（无后缀别名） | 自动映射到 `_baseline` |

---

## Changelog

> 每次新增 / 修改 / 删除 SDPA 内核或 MK trait 时**必须**追加一条记录。格式：`日期 | 改动概述 | 受影响文件`。

| 日期 | 改动概述 | 受影响文件 |
|---|---|---|
| 2026-08-17 | **Sparse MLA QK score-copy/max 融合 checkpoint**：`run_heads_qkpv_chunk_bf16` 在把 `8x2VL` QK tile 搬到 score scratch 时同步累计每行 max，消除 softmax 之前的 max-only score 二次扫描；score/exp/BF16-P/P-pack/PV 和 online chunk 顺序不变。M5 96T/SVL128 三轮配对：2048 shared-prefix -1.22%，later sparse -2.65%；8192 受主机 8.9--9.2/10.3--10.6 ms 双模态影响，不声称收益。M5 direct checks 通过，Amazon 8C/SVL256 `26 passed`。未达独立 3% gate，作为 direct packed-P 的组合序列 checkpoint。 | 改 `csrc/{sparse_mla.cpp,SDPA_VERSIONS.md}`、`optimizations/sparse_mla/manifest.yaml`；新建 `optimizations/sparse_mla/results/amazon_m5_score_copy_max_20260817.md` |
| 2026-08-17 | **Sparse MLA shared-prefix token-panel/B-block L2 调度负结果**：参考 MoE cache blocking，把相邻 token 的 Q 与 online `(m,l,O)` 作为线程私有 A/C panel，共享 packed-K/V B chunk 提到 token 内循环外，score/P scratch 固定为单/双 B-block slot，causal 拆 common rectangle + triangular fringe，QK/PV 保持 `8x2VL`。Amazon 8C panel4 仅改善 0.9--1.5%。M5 NUMA1 cores96--191/SVL128 完整扫 panel `{4,8,16}` x B block `{64,128,256}`；配对 panel8/B256/single 的 2048/8192/later-sparse 分别为 -0.14%/+0.30%/+0.19%。PMU 显示 2048 L2 refill 降 12--16%、bus access 降 8--11%，但 duration 仅 -0.24%且 LLC miss 上升；8192 L2 几乎不变。未达 3% gate，故撤销源码且不增加开关。 | 改 `csrc/{SDPA_TODO.md,SDPA_VERSIONS.md}`、`optimizations/sparse_mla/manifest.yaml`；新建 `optimizations/sparse_mla/results/{amazon_8c_token_panel_cache_20260817.md,amazon_m5_token_panel_cache_20260817.md}` |
| 2026-08-16 | **Online-softmax exp packed-lane/Estrin Lab 比较**：用 1 个 range-reduction + 2 个 polynomial packed NEON 常数寄存器生成 indexed lane FMUL/FMLS/FMLA，并对 Horner/Estrin 各扫 U1/U2/U4/U8。M5 SVE128、length2048 三会话中 isolated best-to-best 吞吐 +5.92%，16 个 SVE accumulator 跨 softmax 活跃时 +5.75%；同 U8 时 Z spill 均为 8 slots，热循环 Q spill pairs 14→5，吞吐 +12.66%，但 best-to-best 的 U8 candidate 相比 U4 baseline 增加 spill，故只保留 Lab。两者 max 39 ULP、BF16 mismatch 均 76/1,048,570。 | 新建 `optimizations/sparse_mla/benchmarks/{softmax_exp_constants.cpp,Makefile}`、`optimizations/sparse_mla/results/amazon_m5_softmax_exp_constants_20260816.md`；改 `optimizations/sparse_mla/manifest.yaml`、`csrc/SDPA_VERSIONS.md` |
| 2026-08-14 | **Sparse MLA indexed K/V 单次 gather+pack**：后续 query-dependent sparse 的跨 token KV 重叠不足以共享 pack，因此在每个 2VL key tile 内将 K 与 V pack 融合；每 4 个 indexed KV 行的 8-wide load 同时生成两个 K=4 BFMMLA panels 和一个转置 V panel，输入 KV payload 从 `d_qk+d_v` 降到 `d_qk`。Amazon 8C `q[2048,32,192],topk640,Dv128,context_start10000`：1T 280.552→254.908 ms（-9.14%），8T 正/反顺序为 -5.83%/-6.04%，profile pack 61.209→37.819 ms（-38.21%）；共享前缀 8T -0.18%。SVL256/SVL128 各 `26 passed`，checksum 一致。 | 改 `csrc/{sparse_mla.cpp,sparse_mla_sve.h,SDPA_VERSIONS.md}`、`optimizations/sparse_mla/manifest.yaml`；新建 `optimizations/sparse_mla/results/amazon_8c_indexed_single_pack_20260814.md` |
| 2026-08-14 | **Sparse MLA 前 2048 V4-like 双前缀共享 K/V pack**：SVE BF16 head-major 自动识别 compressed + causal/SWA 两段固定起点连续前缀，把每段最大范围按运行时 2VL BFMMLA layout 在调用级只 pack 一次；每 token 仅消费自身有效长度，不增加 QK/PV。query-dependent TopK、滑动 SWA、低复用、非整 tile 和 non-SVE 保持 per-token gather-pack fallback。Amazon 8C、`q[2048,32,192],Dv128`、compressed512+SWA2048：1T 552.018→451.861 ms（-18.14%），8T 反向顺序稳态 82.702→76.929 ms（-6.98%）；后续 sparse 1T/8T 为 +0.04%/-0.20%。SVL256/SVL128 完整文件各 `26 passed`。 | 改 `csrc/{sparse_mla.cpp,SDPA_VERSIONS.md}`、`tests/test_sparse_mla.py`、`optimizations/sparse_mla/manifest.yaml`；新建 `optimizations/sparse_mla/results/amazon_8c_shared_prefix_pack_20260814.md` |
| 2026-08-14 | **统一 Sparse MLA head-major QK/PV 与 online-softmax 中间计算**：shared-dense 与 indexed-sparse 保留各自 K/V pack 和任务调度，但共同调用一个内部 chunk executor，统一执行 QK、score materialization、online max/sum/O correction、BF16 P、P pack 和 PV。sparse 的 ragged/重复索引、sink、精确 stats，及 dense 的在线 stats 语义均保持不变；默认 dispatch、公开 API 和 packed layout 不变。Amazon 8C SVE-BF16 extension 全量构建成功；SVL256 cores 0--3 与强制 SVL128 cores 4--7、OMP=4 各 `24 passed`。AmazonM5192Cores 原生 SVL128 上主 `_C` extension 以 C++17 构建成功，NUMA1 cores 96--99、OMP=4 为 `24 passed`；完整 setup 后续仍被无关 MoE C++17 `atomic::wait/notify_all` 编译错误阻断。M5 NUMA1、cores 96--191、`q[2048,32,192],topk640,Dv128`、5 warmup + 21 iterations、三个独立 session 的 session-median 再取中位数：1T sparse 220.664→212.849 ms（-3.54%）、dense 172.402→168.588 ms（-2.21%）；96T sparse 3.025→2.914 ms（-3.67%），dense 因性能状态呈双峰，稳态约 2.457→2.433 ms（约 -0.98%，只判定无稳定回退）。`q[8192,...]` 的两个 paired session 中 sparse 11.819→11.359 ms（约 -3.89%）；dense 两轮均不回退但幅度受机器状态影响，不量化收益。baseline 为同一 HEAD 源码单文件重编译，candidate 只替换 `sparse_mla.o`，链接对象、编译参数、绑核和 NUMA 配置一致；checksum 一致。 | 改 `csrc/{sparse_mla.cpp,SDPA_VERSIONS.md}` |
| 2026-08-14 | **Sparse MLA shared-dense QK/PV 从 fixed 8×8 改为 scalable SVE 8×2VL**：保持同一 token 的 8 个连续 Q head 为 M，在调用级把连续 K/V 一次 pack 为 scalable B-layout；SVL128 计算 8×8，SVL256 计算 8×16，final key/Dv 半 tile 零填充并谓词写回，online softmax 不变，non-SVE 保留原 fixed 8×8。Amazon 8C dense `q[2048,32,192],KV640,Dv128` 上，SVL128 1T/8T 由 334.450/54.206 降为 250.042/36.384 ms（-25.24%/-32.88%），SVL256 由 332.588/54.248 降为 213.853/32.610 ms（-35.70%/-39.89%）；gathered sparse 四点无回退。最终两种 VL 各 24 tests passed，non-SVE AArch64 syntax compile 通过，QK/PV 各 16 条 `bfmmla` 且无 Z spill。 | 改 `csrc/{sparse_mla.cpp,sparse_mla_sve.h,SDPA_VERSIONS.md}`、`tests/bench_sparse_mla_scalable.py`、`docs/public_contracts.md`、`optimizations/sparse_mla/{manifest.yaml,results/amazon_8c_scalable_sve_20260814.md}`；新建 `optimizations/sparse_mla/results/amazon_8c_dense_scalable_sve_20260814.md` |
| 2026-08-14 | **Sparse MLA PV 完成 scalable SVE 8×2VL BFMMLA**：gather V 时用 NEON 4×8 transpose 直接生成 scalable B-layout；row-major BF16 P pack 为 2-row A panels，K4 尾部补零，SVL256 的 Dv 半 tile 谓词保护；每 K4 用 16 条 `bfmmla` 并加回原 online-softmax O。Amazon 8C DSV4-like `q[2048,32,192],topk640,Dv128` 上，SVL256 1T/8T 相对 fixed 8×8 从 252.176/38.508 降到 180.951/26.817 ms（-28.24%/-30.36%），相对 QK-only commit `58881f8` 再降 19.66%/27.05%；SVL128 相对 fixed 降 19.96%/26.62%。两种 VL 各 24 tests passed。旧 O 预装 accumulator 方案因 4-Z spill 和 0.75% 单核回退撤销。 | 改 `csrc/{sparse_mla.cpp,sparse_mla_sve.h,SDPA_VERSIONS.md}`、`docs/public_contracts.md`、`optimizations/sparse_mla/{manifest.yaml,results/amazon_8c_scalable_sve_20260814.md}` |
| 2026-08-14 | **Sparse MLA sparse QK 从 fixed NEON 8×8 改为 scalable SVE 8×2VL**：保持 16 个 BFMMLA FP32 累加器与 K=4 instruction topology，K pack 和 score epilogue 随运行时 VL 扩展；SVL128 为 8×8，SVL256 为 8×16。PV、online softmax、shared-dense 8×8 与 non-SVE fallback 不变。Amazon 8C DSV4-like `q[2048,32,192],topk640,Dv128` 上，1T/8T 从 252.176/38.508 降至 SVL256 的 225.232/36.759 ms（-10.68%/-4.54%）；SVL128 为 250.760/38.313 ms。两种 VL 各 24 tests passed，QK 汇编 16 条 `bfmmla`/K4 且无 spill。 | 新建 `csrc/sparse_mla_sve.h`、`tests/bench_sparse_mla_scalable.py`、`optimizations/sparse_mla/results/amazon_8c_scalable_sve_20260814.md`；改 `csrc/{sparse_mla.cpp,SDPA_VERSIONS.md}`、`docs/public_contracts.md`、`optimizations/sparse_mla/manifest.yaml` |
| 2026-08-14 | **Sparse MLA head-major 8×8 转为公开默认实现**：BF16 且 `Hq%8==0,Dqk%4==0,Dv%8==0` 的常见长 query 按 token 复用 MQA K/V pack，并分别处理共享连续 KV 与 sparse indexed KV；短 query 保留最多 8-way KV split，FP32/非支持维度保留 indexed fallback。删除内部 variant selector、masked/pruned/fused tail 实验源码和专用 benchmark，负结果与回滚点保留在 manifest、结果文档和 Git 历史。公开签名及数值语义不变。Amazon 8C SVL256、cores 0--3、OMP=4 release build 成功，Sparse MLA 24 tests passed，实验 selector ABI removal 检查通过；192-core/SVL128 主机 SSH 无响应，未报告该机验证结果。 | 改 `csrc/{sparse_mla.cpp,module.cpp,SDPA_VERSIONS.md}`、`tests/test_sparse_mla.py`、`docs/public_contracts.md`、`optimizations/sparse_mla/{manifest.yaml,results/arm_head_major_20260813.md}`；删 `csrc/sparse_mla_tail_microkernels.{h,cpp}`、`tests/bench_sparse_mla_tail_variants.py` |
| 2026-08-13 | **DeepSeek V4 Indexer select-all 提前短路**：把 `max(ke-ks)<=topk` 判定移到 Q 调度之前；命中时 raw/prepacked 都跳过 Indexer Q GEMM 和 Q RoPE/weight scaling，prepacked 强制 shared-Q pool 时也只运行 Main Q，同时保留 Main Q、两个 compressor 及阶段末尾 TopK 写入顺序。Amazon 8C SVL256 上 raw/prepacked 专项与完整文件 20 tests passed；`M2048,context_start=0,topk512` 中位数 74.309→40.176 ms（1.850×），34.854 ms Indexer Q 工作归零。192-core/SVL128 主机 SSH 无响应，未报告结果。 | 改 `csrc/{deepseek_v4_post_gemm_stage.cpp,SDPA_VERSIONS.md}`、`tests/test_deepseek_v4_post_gemm_stage.py`、`optimizations/deepseek_v4_post_gemm/{manifest.yaml,results/amazon_8c_indexer_select_all_early_exit_20260813.md}` |
| 2026-08-13 | **DeepSeek V4 Indexer native batched exact TopK 候选**：把 2,048 次逐行 ATen TopK 替换为单次 OpenMP batched `nth_element + selected-K sort`，每 worker 复用候选缓冲，只写 strided int32 indices；保留 score 降序、tie/NaN 定义和非标准 tensor contract 的 ATen fallback。Amazon 8C SVL256 上 20 个独立 TopK 边界 case 与 25 项集成测试通过；`M2048,N1536,K512` 的 TopK 75.155→9.637 ms（7.799×），完整 stage 186.251→120.758 ms（1.542×），score 保持 36.15 ms。仅完成 8C 首测，192C/SVL128 待测，故保持 candidate。 | 改 `csrc/{deepseek_v4_indexer_sve.h,deepseek_v4_indexer_sve.cpp,deepseek_v4_post_gemm_stage.cpp,SDPA_VERSIONS.md}`、`benchmarks/bench_deepseek_v4_indexer_sve.cpp`、`tests/test_deepseek_v4_post_gemm_stage.py`、`optimizations/deepseek_v4_post_gemm/{manifest.yaml,results/amazon_8c_indexer_native_batch_topk_20260813.md}` |
| 2026-08-13 | **DeepSeek V4 sparse indexer weighted-ReLU 8×2VL 候选**：修正长路径为逐 head `ReLU(q·k)` 后乘 weight 再归约；新增 scalable-SVE 8×2VL BFMMLA 内核，Q 每 query 私有 pack、K 从 paged cache 直接 packB、ReLU/weight/head reduce 全部留在寄存器并只写最终 FP32 score；非 SVE/非 H8-D4 形状走等价 FP32 fallback。Amazon 8C SVL256 的 45 个 score 组合最大绝对误差 `9.54e-7`，25 tests passed/5 retired skipped，汇编无 Z spill；生产 score 形状 1T/8T 为 202.285/36.653 ms（1.406 TFLOP/s），完整 stage 186.251 ms，其中 score/TopK 为 36.154/75.155 ms。仅完成用户要求的 8C 首测，SVL128/192C 待测，故保持 candidate。 | 新建 `csrc/deepseek_v4_indexer_sve.{h,cpp}`、`benchmarks/bench_deepseek_v4_indexer_sve.cpp`、`optimizations/deepseek_v4_post_gemm/results/amazon_8c_indexer_weighted_relu_sve_20260813.md`；改 `csrc/{deepseek_v4_post_gemm_stage.cpp,SDPA_VERSIONS.md}`、`src/fused_cpp/{sparse_attn_indexer.py,deepseek_v4_post_gemm_stage.py}`、`tests/{test_sparse_attn_indexer.py,test_deepseek_v4_post_gemm_stage.py,bench_deepseek_v4_post_gemm_stage.py}`、`optimizations/deepseek_v4_post_gemm/manifest.yaml` |
| 2026-08-13 | **Sparse MLA head-major sparse 8×8 组合实验**：新增内部 `heads_sparse_8x8`，按 token gather/pack 一份 sparse MQA K/V 并复用于所有 8-head group，按 L2 chunk 保留 online-softmax 状态；组合 dispatch 对全局共享连续 KV 先走 dense-head 调用级 pack。`q[2048,32,192]`、DSV4 top-k512+window128、context0、Dv128 下，Amazon 8C 1T/8T 分别快 30.16%/30.51%，Arm-codex 1T/80T 分别快 14.82%/54.16%；全 dense 与 `heads_dense_8x8` 持平。两机 SVL256 各 43 tests passed，公开默认不变，候选因尚缺三次独立 session 保持 experimental。 | 改 `csrc/{sparse_mla.cpp,SDPA_VERSIONS.md}`、`tests/{test_sparse_mla.py,bench_sparse_mla_tail_variants.py}`、`optimizations/sparse_mla/{manifest.yaml,results/arm_head_major_20260813.md}` |
| 2026-08-13 | **Sparse MLA head-major dense 8×8 实验**：新增内部 `heads_dense_8x8`，以同一 token 的 8 个 head 作为 QK/PV M 维，调用级复用 shared-MQA K/V pack，并增加 dense-shared benchmark 与输出/stats/fallback 测试。`q[2048,32,192]`、KV640、Dv128 下 Amazon 8C 的 1T 快 0.69%、8T 持平；Arm-codex 1T 慢 0.60%，80T 快 18.96%，后者主要来自 2,048 token tasks 相对旧 256 token-tile tasks 的静态负载均衡改善。两机实际 SVL256、各 37 tests passed；公开默认不变。 | 改 `csrc/{sparse_mla.cpp,SDPA_VERSIONS.md}`、`tests/{test_sparse_mla.py,bench_sparse_mla_tail_variants.py}`、`optimizations/sparse_mla/{manifest.yaml,results/arm_head_major_20260813.md}` |
| 2026-08-12 | **Sparse MLA fused online epilogue、三种 PV 指令比较与 2D KV split**：新增 SVL128/256 的 fused QK→online-softmax→PV 实验，消除 8×8 score/`p_hat` scratch；同一框架比较 FMLA/BFMLAL/BFMMLA，tail-only 上 SVL256 以 FMLA 最快（45.385 ms vs materialized 55.238），SVL128 则 materialized 最快（27.361 ms），BFMMLA 两边均非最优，故不切默认。新增 associative online-softmax partial merge 与最多 8-way KV shard，BF16 公开入口在 query blocks≤threads/2 时启用；真实 DSV4 4a 离散 top-k + 128 sliding window 下，192C cores96--191 的 64/128/256-token chunk 分别加速 5.317/3.872/2.276×，full 4096/8192 保持 ±0.22%。两台 SVL128/256 目标机各 34 tests passed。 | 改 `csrc/{sparse_mla.cpp,sparse_mla_tail_microkernels.cpp,sparse_mla_tail_microkernels.h,SDPA_VERSIONS.md}`、`tests/{test_sparse_mla.py,bench_sparse_mla_tail_variants.py}`、`docs/sparse_mla_vllm_integration.md`、`optimizations/sparse_mla/{manifest.yaml,results/amazon_sparse_mla_2d_fused_20260812.md}` |
| 2026-08-12 | **Sparse MLA 连续 ragged tail 的 full/pruned 8×8 候选**：新增安全的 monotonic-tail planner、packed masked 8×8 QK/PV、2×2 BFMMLA pruning、BFMLAL valid-lane pruning、跨 segment packed-Q 复用与内部三版本 benchmark 入口。Amazon 192C NUMA1、`q[2048,32,192]`、V4-like 2,621,952 pairs/head、21-run 轮转交错：1T indexed/full/pruned 为 586.217/544.225/541.370 ms（候选快 7.2--7.7%）；12/32/96T 三者均约 55 ms，full/pruned 没有稳定收益且 32T 回退约 1.3--1.5%，因此公开默认继续 `indexed_4x4`。1,024-block tail-only 隔离确认 pruned 相对 full 在 causal 快 5.0--12.9%、V4-mix 快 4.7--5.8%，说明 E2E 差异小是尾部占比稀释而非 pruning 无效。记录了初版 full tile、初版 QK pruning、E/8 展开+向量 store+Q reuse、精确 PV pruning 四轮尝试。 | 改 `csrc/{sparse_mla.cpp,module.cpp,SDPA_VERSIONS.md}`、`tests/{test_sparse_mla.py,bench_sparse_mla_tail_variants.py}`；新建 `csrc/sparse_mla_tail_microkernels.{h,cpp}`、`optimizations/sparse_mla/{manifest.yaml,results/amazon_192c_masked_tail_20260812.md}` |
| 2026-07-28 | **DeepSeek V4 attention MN 路径 A 只 pack 一次**：Linux NEON 下新增 attention-prefixed packed-A M8 BF16/FP32 row-major store kernel；OpenMP team 先协作把 hidden states 写成一份 reorder-M8 buffer，尾部补零，经一次 publish barrier 后由四个 GEMM 的全部 N-group 复用。默认仅在 MN requested N groups >=24 时启用，`FUSED_CPP_ATTN_GEMM_PREPACK_A=on/off` 可强制选择，SVE 与其他 schedule 保持原路径。Amazon 192C NUMA0、M2048/K4096/N 总计3904：强制 off/on 在 24/32/48/64/80/96T 分别加速 1.038/1.028/1.047/1.020/1.016/1.030x；96T 21-run 为 3.154→3.059 ms、20.764→21.411 TFLOP/s。2--16T 存在最高约 4.7% 回退，因此 auto 门限避开该区间；31 项本地、远程 NEON、远程 SVE 测试均通过。 | 改 `csrc/deepseek_v4_attn_gemm_fused.cpp`、`csrc/moe/arm/neon_bf16/kernels.S`、`setup.py`、`tests/{bench_deepseek_v4_attn_gemm_fused.py,test_deepseek_v4_attn_gemm_fused.py}`、`optimizations/deepseek_v4_attn_gemm/{manifest.yaml,results/amazon_192c_attn_gemm_scheduler_20260728.md}`、`csrc/SDPA_VERSIONS.md` |
| 2026-07-28 | **DeepSeek V4 attention 四 GEMM MN auto 默认调度**：新增 `legacy/m8/pool/mn` 四种运行时选择并将无环境变量时的默认值切到 `mn`；M8 对齐切 M 只保留一次全局尾块，共享 OpenMP 任务池允许四个不同 N 的 GEMM 互相补齐负载，MN 模式再按 backend N tile 切 packed-B 并采用 owner-first M8 panel stealing。Amazon 192C NUMA0 0-95、M2048/K4096/N 总计 3904、65.498 GFLOP：legacy 12.361 ms / 5.299 TFLOP/s，M8 8.674 ms / 7.551 TFLOP/s，pool 7.914 ms / 8.276 TFLOP/s，MN auto 3.329 ms / 19.676 TFLOP/s；MN 1T→96T 为 268.1 GFLOP/s→20.66 TFLOP/s（77.1x，80.3% 线性度）。配对 sweep 中 MN 在 1--4T 与 legacy 持平，8/16/24/32/48/64/80/96T 分别为 1.115/1.276/1.827/2.404/3.688/2.839/3.242/3.735x。`legacy` 保留为显式回退，normed 路径尚未接入。 | 改 `csrc/deepseek_v4_attn_gemm_fused.cpp`、`tests/{bench_deepseek_v4_attn_gemm_fused.py,test_deepseek_v4_attn_gemm_fused.py}`；新建 `optimizations/deepseek_v4_attn_gemm/{manifest.yaml,results/amazon_192c_attn_gemm_scheduler_20260728.md}`；改 `csrc/SDPA_VERSIONS.md` |
| 2026-07-22 | **x86 MoE 参考 SVE 引入多线程切 N 与 skew-aware wave 调度**：AVX-512/AMX executor 从只支持 1/2T 扩展为 1--256 requested workers；active expert 不足时建立 route-weighted teams，并行 gather 后经 barrier 分别切分 W13 F16 与 W2 N32 blocks；最大 route 至少 64 且为次大 2x 时按 route 降序组成 waves，均衡且 active expert 足够时保留 atomic expert queue。barrier 支持异常取消，scratch 按 wave slot 最大 route 复用，merge threads capped by token count。C8i8 H4096/F512/M2048：AMX hot 26.45→5.76 ms（1T→8T，4.60x），`[1536,256,256]` 39.22→11.89 ms（3.30x）；完整 x86 suite 163 passed。 | 改 `csrc/moe/x86/avx512_bf16/executor.cpp`、`src/fused_cpp/moe/bf16_tiled.py`、`tests/{test_moe_avx512_bf16.py,bench_moe_avx512_bf16.py}`、`optimizations/fused_moe_avx512/{README.md,TODO.md,manifest.yaml,results/amazon_c8i_8core_nsplit_20260722.md}`、`cpu_moe_schedule_optimization/MATHEMATICAL_MODEL.md`、`csrc/SDPA_VERSIONS.md` |
| 2026-07-20 | **x86 AMX MoE W2 store/merge epilogue 实验与负结果保留**：新增 cache-key 隔离的 `baseline/combined/tile_store`。`combined` 在 M1N4/N64 只生成一次 row route address；`tile_store` 让 TMM 直接写 expert-contiguous FP32 workspace，并由预建 route-row map 做 AVX-512 weighted merge。三条路径跨 pattern/tail/单双线程 bit-exact，top-k=1 direct BF16 保持 vector conversion。C8i K512 rotated W2-only `tile_store` 为 +1.0%~+14.5%，但 K32 store-dominated latency +12.8%~+22.8%，end-to-end 为 -4.5%~+1.7%，所以 auto 仍为 baseline；完整 x86 suite 122 passed/4 skipped。 | 改 `csrc/moe/x86/avx512_bf16/{jit_kernels.cpp,kernels.cpp,kernels.h,executor.cpp}`、`tests/test_moe_avx512_bf16.py`、`benchmarks/bench_amx_bf16_{patterns.py,w2_epilogues.cpp,merge.cpp}`、`optimizations/fused_moe_avx512/{README.md,TODO.md,manifest.yaml,results/amazon_c8i_2core_amx_w2_epilogue_20260720.md}`、`csrc/SDPA_VERSIONS.md` |
| 2026-07-20 | **x86 AMX MoE W13 SiLU epilogue 优化与负结果保留**：新增 `baseline/resident/pipelined/rcp14` 四种 cache-key 隔离的 Xbyak 路径。`resident` 把多项式/exp 常量一次广播后驻留 ZMM，保持逐行算术顺序和 BF16 bit-exact，M16/degree5 JIT 3972→2836 B，K256 standalone +8.4%，设为 auto；两行 stage pipeline 未稳定胜过 resident，`VRCP14PS` 非 bit-exact 且端到端收益处于噪声范围，均仅保留实验开关。C8i 完整 x86 suite 114 passed/4 skipped。 | 改 `csrc/moe/x86/avx512_bf16/jit_kernels.cpp`、`tests/test_moe_avx512_bf16.py`、`benchmarks/bench_amx_bf16_w13.cpp`、`benchmarks/bench_amx_bf16_patterns.py`、`optimizations/fused_moe_avx512/{README.md,TODO.md,manifest.yaml,results/amazon_c8i_2core_amx_silu_20260720.md}`、`csrc/SDPA_VERSIONS.md` |
| 2026-06-10 | **`flash2_neon_l3kv_packqkv*` fp32 路径改为全量 pack K + packed-K lane-FMLA QKᵀ**：fp32 输入不再 delegate 到 `flash2_neon_l3kv_packv_pquad`。入口新增 K pack layout `[B,N,S/8,E,8]`，Q 保持 row-major；QKᵀ 8×8 用 `MK_Fp32PackK8PQuad` 按 4 个 E lane 加载 Q、按 K lane 向量累加，消掉旧 fp32 QKᵀ 的 per-score horizontal reduction；PV 继续使用 fp32 pquad。fp32 数值不再与旧 delegate bit-exact，改按 fp32 SDPA 容差验证。 | 改 `csrc/sdpa_flash2_neon_l3kv_packqkv.cpp`（fp32 packK helper、packed-K QKᵀ trait、fp32 entry 路由）；改 `tests/test_sdpa_l3kv_packqkv.py`（fp32 delegate 断言改为 baseline 容差比较）；改 `src/fused_cpp/sdpa.py` 与 `csrc/SDPA_VERSIONS.md`（metadata/文档） |
| 2026-06-01 | **新增 SDPA 版本 `flash2_neon_l3kv_packqkv_pbf16pv`**：在 packqkv 的 Q/K/V pre-pack 路径上增加 `kPbf16PV` 模板开关，softmax 计算 fp32 sum 但把 full 8-row block 的 `P_hat` 直接写成 bf16 scratch，PV 改调 `pv_8x8_pbf16`，旧 `flash2_neon_l3kv_packqkv` 保持不变。SVE `FCVT/FCVTNT` 压缩 store 实测 softmax 过慢，最终 bf16-store helper 使用 NEON `vcvt_bf16_f32`。Arm-codex-internal/Arm-codex core80 单线程：BGE-small-zh 形状 bf16 non-causal 10.776→9.989 ms（+7.3%）、causal 12.389→11.317 ms（+9.5%）；`B1-N32-L2048-S2048-E192-Ev128` bf16 non-causal 1422.704→1276.547 ms（+11.5%）、causal 952.715→867.951 ms（+9.8%）。 | 改 `csrc/sdpa_flash2_neon_l3kv_impl.h`（新增 `vectorized_exp_minus_bf16_impl`、`process_q_tile_lc_packqkv` / run path 的 `kPbf16PV` 分支与 bf16 P scratch）；改 `csrc/sdpa_flash2_neon_l3kv_packqkv.cpp`（新增版本入口和注册）；改 `src/fused_cpp/sdpa.py`（metadata）；改 `tests/test_sdpa_versions_equiv.py`（packqkv 约束覆盖新版本）；改 `csrc/SDPA_VERSIONS.md` |
| 2026-05-31 | **新增 SDPA 版本 `flash2_neon_l3kv_packv_l1_bfmlal_layout` 与别名 `flash2_neon_l3kv_l1_bfmlal_layout`**：在 packv L3KV 拓扑上接入 `MK_L1BfmlalLayout`，bf16 QKᵀ 8×8 使用 K_col BFMLAL microkernel，bf16 PV 对支持 `kHasPvPbf16` 的 trait 自动把 `P_hat` scratch 转 bf16 并调用 `pv_8x8_pbf16`。这版用于端到端评估 L1-only 85%+ peak 微内核的外层成本；fp32 仍走同 trait 的普通 fp32 fallback。Arm-codex-internal/Arm-codex core80 单线程 BGE-small-zh 形状 bf16：non-causal 12.682 ms（43.16 GFLOPS）vs packv 13.972 ms；causal 14.210 ms（19.30 GFLOPS）vs packv 15.176 ms。 | 改 `csrc/sdpa_flash2_neon_l3kv_impl.h`（增加 `pv_8x8_pbf16` trait 检测、P_hat_bf16 scratch 与 PV 分发）；改 `csrc/sdpa_flash2_neon_l3kv_packv.cpp`（注册新 SDPA 版本与别名）；改 `src/fused_cpp/sdpa.py`（Python registry metadata）；改 `csrc/SDPA_VERSIONS.md` |
| 2026-05-31 | **新增 L1-only bf16 自由 layout 微内核 `l1_bfmlal_layout`，目标从 90% 调整为 85% 后达标**：新增 `pack_k_8rows_to_col_bf16` 把 K 从 `K[8][E]` 转成 reduce-major `K_col[E][8]`，并新增 `gemm_qkt_microkernel_8x8_bf16_qrow_kcol_bfmlal`，让 QKT 在 microkernel 层变成与 pre-bf16-P PV 相同的 BFMLAL lane 形态。该 trait 只做 L1 工作集上限评估，不考虑 SDPA 外层布局生产成本；`qkt_8x8_kcol` 直接计时 K_col 已由 caller 提供的纯布局路径。Arm-codex-internal/Arm-codex core80 单线程、E=Sk=768、iters=1,000,000：`qkt_8x8_kcol` 82.64 GFLOPS、`pv_8x8_pbf16` 82.62 GFLOPS，均约 89.8% of 92 GFLOPS，超过 85% 目标；`validate_microkernel` E/Sk=17/64/512 上 `qkt_8x8_max_abs=0`、`qkt_8x8_kcol_max_abs=0`、`pv_8x8_pbf16_max_abs=0`。 | 新建 `csrc/sdpa_microkernels/impls/mk_l1_bfmlal_layout.h`；改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（新增 K_col pack helper、QKT K_col BFMLAL 内核，并微调 pbf16 PV load/compute 交错）；改 `csrc/sdpa_microkernels/all_impls.h`、`csrc/sdpa_microkernels/mk_registry.cpp`（注册新 trait）；改 `csrc/sdpa_microkernels/mk_registry_helpers.h`（新增 `qkt_8x8_kcol` validate/bench 可选项）；改 `csrc/SDPA_VERSIONS.md`（新增 trait 章节、迭代表和 changelog） |
| 2026-05-31 | **bf16 PV pquad 切 BFMLAL 快路径**：`gemm_pv_microkernel_8x8_bf16_pquad` 在 BF16 arithmetic 目标上优先派到新增 `gemm_pv_microkernel_8x8_bf16_pbf16_bfmlal`。每 4-k 段把 8 行 P_hat quad 用 `vcvt_bf16_f32` 临时 round 到 bf16，再用 `vbfmlalbq_lane_f32 / vbfmlaltq_lane_f32` 乘 bf16 V；主体避开 V widen，尾部 Sk%4 保持原 fp32 widen FMA。Arm-codex-internal/Arm-codex core80 单线程：`pquad` / `qk_packqk_seq4_bmajor_pv_pquad` 的 bf16 `pv_8x8` 在 Sk=128 从上一轮约 54.8 GFLOPS 提到 E64 59.10 / E128 59.52 / E192 58.68 GFLOPS，约 64–65% of 92 GFLOPS 单核峰值；`validate_microkernel` E=17/64/128/192 max_abs≈1.6e-6，小形状 SDPA 对 naive max_abs=0.0078125。仍未达到 90%，下一步重点是降低 BFMLAL 路径的 P convert / lane 指令开销，或改 PV 数据布局。 | 改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（新增 pbf16 BFMLAL PV 内核并接到 bf16 pquad）；改 `csrc/sdpa_microkernels/impls/mk_pquad.h`、`csrc/sdpa_microkernels/impls/mk_qk_packqk_seq4_bmajor_pv_pquad.h`（注释同步数值语义）；改 `csrc/SDPA_VERSIONS.md`（本节 + `MK_PQuad` / packqkv 记录） |
| 2026-05-31 | **`qk_packqk_seq4_bmajor` bf16 QKᵀ inner loop 优化**：把 `gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner` 的 packed Q/K 地址从每轮 `(e/4)*32` 改为显式 `q_ptr/k_ptr` 递增，并将常见 `E % 8 == 0` 路径按 2 个 e-block 展开；BFMMLA 发射顺序、pack layout、tail 与数值语义不变。Arm-codex-internal/Arm-codex core80 单线程验证：`qk_packqk_seq4_bmajor_pv_pquad` bf16 QKᵀ 在 Sk=128 下 E64 53.44→57.00 GFLOPS（+6.7%）、E128 60.34→64.78（+7.4%）、E192 65.07→69.05（+6.1%）；PV 未改，仍约 55 GFLOPS（约 60% of 92 GFLOPS 单核峰值），后续优化重点仍是 PV 与 QKᵀ 90% peak。验证包含 `validate_microkernel` E=17/64/128/192 全部 max_abs=0。 | 改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（bmajor inner loop 指针递增 + 2-way 展开）；改 `csrc/SDPA_VERSIONS.md`（本节 + packqkv/microkernel 记录） |
| 2026-05-29 | **`flash2_neon_l3kv_packqkv` 接入 PV pquad**：`process_q_tile_lc_packqkv` 的 PV 调用从 hardcoded `gemm_pv_8x8 / gemm_pv_tail`（baseline）切到 `MK_QkPackqkSeq4BmajorPvPquad::pv_8x8 / pv_tail`，bf16 路径走新 `gemm_pv_microkernel_8x8_bf16_pquad`；fp32 输入的 delegate 路由从 `flash2_neon_l3kv_packv` 改为 `flash2_neon_l3kv_packv_pquad`（同步两处：fp32 entry + path B 退化）。不新增 SDPA 顶层变体、不模板化 packqkv（保持 bf16-only 专用 path 形态）。`test_sdpa_versions_equiv` 645 项全过（含 packqkv bf16/fp32 noncausal/causal mask）。**本机端到端实测**（Apple Silicon P-core，OMP=1，B=1 N=8 E=192 Ev=128，bf16）：L=128 50.4（pack overhead 占比高，-2%）/ L=256 57.6（+8.0%）/ L=512 58.4（+9.3%）/ **L=1024 59.3 GFLOPS（+8.7%）** vs `flash2_neon_l3kv_packv` baseline。fp32 路径三个版本都~15.2 GFLOPS（瓶颈在 fp32 QKᵀ 不在 PV，无回归） | 改 `csrc/sdpa_flash2_neon_l3kv_impl.h`（顶部 include trait 头；line 951-964 PV 调用切换）；改 `csrc/sdpa_flash2_neon_l3kv_packqkv.cpp`（line 163 / 239 两处 `sdpa_dispatch` 路由）；改 `csrc/SDPA_VERSIONS.md`（`flash2_neon_l3kv_packqkv` 章节 + 本节） |
| 2026-05-29 | **`MK_PQuad` 扩展到 bf16 PV**：新增 `gemm_pv_microkernel_8x8_bf16_pquad`，照搬 fp32 pquad 的 P-端 quad load + lane FMA + 4-k 外展 + 段 2 中段跨迭代 V 预取调度。V 端从 baseline 的 `2× vld1_u16 + 2× widen` 改成 `1× vld1q_u16 + vshll_n_u16(lo) + vshll_high_n_u16(hi)`，每 4-k 段 LSU 从 36 降到 12。`MK_PQuad::pv_8x8(bf16)` 与 `MK_QkPackqkSeq4BmajorPvPquad::pv_8x8(bf16)` 都改派到新内核。数值与 baseline 等价（widen 后 fp32 累加序逐段一致）。**远程机器实测**（E=192 / Sk=128 / iters=20000）：pv_8x8 bf16 baseline 44.27 → **54.74 GFLOPS（+24%，60% peak）**；fp32 PV 同时受 5/28 方案 A 调度优化保持 72.28 GFLOPS（+13% vs baseline 63.86，79% peak）。bench 表 `TRAIT_OVERRIDES` 同步标注 pquad 与组合 trait 现在拥有 bf16 PV cell | 改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（新增 `gemm_pv_microkernel_8x8_bf16_pquad`）；改 `csrc/sdpa_microkernels/impls/mk_pquad.h`（bf16 pv 不再 fall through）；改 `csrc/sdpa_microkernels/impls/mk_qk_packqk_seq4_bmajor_pv_pquad.h`（bf16 pv 同步派发）；改 `tests/bench_pv_vs_qkt.py`（`TRAIT_OVERRIDES` 加上 pquad/组合 trait 的 bf16 PV）；改 `csrc/SDPA_VERSIONS.md`（本节 + `MK_PQuad` 章节） |
| 2026-05-29 | **`gemm_pv_microkernel_8x8_fp32_pquad` 方案 A 调度优化**：把跨迭代 V[k+4] 预取从段 3 末尾移到段 2 中段。原排布下，跨迭代 V load 距下一轮段 0 第 1 条消费 FMA 只有 2 条 FMA 间隔（~1 cycle），覆盖不了 L1 vld1q 5 cycle 延迟；新排布拉到 ~16 条 FMA（~8 cycle），完全隐藏，同时段 3 改为纯 FMA 段消除跨迭代 OoO 阻塞。寄存器活跃峰值 28→30（≤32 仍安全）。`objdump -d` 确认 inner loop **零 spill**（callee-saved 之外没有 `str/ldr q*, [sp...]`）。**远程机器实测**（E=192 / Sk=128 / iters=20000）：pv_8x8 fp32 pquad 69.9 → **79.81 GFLOPS（+14%，87% peak）**（同会话单跑数据；与表格上一次 72.28 的差异是环境噪声）。数值上仍按位等价（只动调度顺序，未改累加序） | 改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（`gemm_pv_microkernel_8x8_fp32_pquad` 段 2/段 3 排布） |
| 2026-05-27 | 新增 SDPA 版本 `flash2_neon_l3kv_packqkv`：在 `l3kv_packv` 基础上把 K 也在 SDPA 入口多线程 pre-pack（`pack_k_to_seq8`），Q 在 `process_q_tile_lc_packqkv` 入口一次性 pack 当前 q-tile 到 thread-local `q_seq_buf`，q-tile 内复用不重复 pack。QKᵀ inner 直接调 `gemm_qkt_microkernel_8x8_bf16_packqk_seq4_bmajor_inner`（microkernel benchmark 上 +23% vs baseline，4 条独立 vld1q_u16 + B-major BFMMLA），bypass MK trait。partial K block 自动 fall back 到 baseline `gemm_qkt_8x4 / gemm_qkt_tail` 读原始 K。**bf16-only**（fp32 delegate 到 `packv`），**`S % 8 == 0` 且 `Ev % 8 == 0`** 强约束，Path B（KV 不装 L3）退化为 `packv`（避免 3× DRAM 带宽）。与 baseline bit-exact | 新建 `csrc/sdpa_flash2_neon_l3kv_packqkv.cpp`；改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（新增 `pack_k_to_seq8` 模板）；改 `csrc/sdpa_flash2_neon_l3kv_impl.h`（新增 `process_q_tile_lc_packqkv` / `run_path_collapse3_packqkv` / `run_path_taskloop_packqkv`，不影响现有模板）；改 `src/fused_cpp/sdpa.py`（`_CPP_VERSION_META` 新增条目）；改 `csrc/SDPA_VERSIONS.md`（本节） |
| 2026-05-27 | 新增 **bf16 QKᵀ pack microkernel 迭代记录**：系统整理 `qk_packk_full/inner/seq`、`qk_packqk_seq/seq4/seq4_ptr/seq4_bmajor/seq4_pipe_a/seq4_pipe_b`、`qk_unroll2` 的 pack layout、load 指令、BFMMLA 发射顺序、是否接入 SDPA、实测结果与每轮迭代最优版本。当前目标机 microkernel 最优为 `qk_packqk_seq4_bmajor`（E=1024,Sk=1024：73.24 GFLOPS / 1.23× baseline）；更可能接入 SDPA 的候选仍是 K-only 方向（后续建议 `qk_packk_inner_bmajor` / `qk_packk_seq4_bmajor`） | 改 `csrc/SDPA_VERSIONS.md` |
| 2026-05-23 | **macOS cache probe 修正**：`probe_cache_bytes_one_level` 在 Apple Silicon 上的 sysctl key 顺序错了——`hw.l1dcachesize` / `hw.l2cachesize` 这种「无后缀」key 默认返回 **E-core 容量**（perflevel1，64 KB / 4 MB），先命中就 return 导致 tile sizing 一直按 E-core 算。修复：把 `hw.perflevel0.<level>cachesize`（P-core）放第一优先级，无后缀 key 作为 fallback。修复后探测到 P-core L1d=128 KB / L2=16 MB（之前是 64 KB / 4 MB）。**实测 wall-clock 持平**（bf16 BFMMLA-pegged + fp32 已 77% peak，tile 不是当前瓶颈），但移除了一个错误假设，是后续 P0-bf16 BFMLALB/T 切换 + tile size 重扫的前提 | 改 `csrc/sdpa_tile_sizes.h:101-106`（sysctl key 顺序） |
| 2026-05-23 | 新增 `MK_QkUblock4` trait：fp32 QKᵀ 8×8 主体从「64 个独立 dot product 单累加器」重写为「4×4 双向分块 + 16 独立累加器外积扇出 + vpaddq 树 reduce」，把 ILP 从 1 条链拉到 16 条独立 fma 链；不需要 K pack 或转置；与 baseline 在 fp32 下接近等价但不按位相同（已被 atol=rtol=1e-4 SDPA 等价性测试覆盖）。自动注册 `flash2_neon_cache_qk_ublock4` / `flash2_neon_l3kv_qk_ublock4` / `flash2_neon_l3kv_packv_qk_ublock4`。**本机实测**（Apple Silicon 单线程）：QKᵀ-fp32 8×8 微内核 7.6–10.1 → **104.7–106.5 GFLOPS（10.4–14.0×，96–98% fmla peak）**；R1-like fp32 SDPA wall-clock 13.9–15.2 → **80.2–84.2 GFLOPS（5.3–5.5×）**；bf16 路径完全不动（1.00× 实测） | 改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（新增 `gemm_qkt_microkernel_8x8_fp32_ublock4`）；新建 `csrc/sdpa_microkernels/impls/mk_qk_ublock4.h`；改 `csrc/sdpa_microkernels/all_impls.h`、`csrc/sdpa_microkernels/mk_registry.cpp`、`csrc/sdpa_flash2_neon_cache.cpp`、`csrc/sdpa_flash2_neon_l3kv.cpp`、`csrc/sdpa_flash2_neon_l3kv_packv.cpp` |
| 2026-05-22 | **P1 优化批次**：(1) softmax row-max 向量化（`vmaxq_f32` + `vmaxvq_f32`）；(2) `is_causal` / `mask_ptr` 模板特化（提到 `bool kHasMask` + `bool kCausal` 模板参数，热路径用 `if constexpr` 编译期消去 mask add / causal mask / causal_lim 整段，实例化数 12→48）；(3) mask add 内层循环 `vld + vadd + vst` 向量化；(4) packv 路径下 ev_off 切换时对下一个 ev_block 起始 2 cache line 发 PLDL1KEEP（非 packv 不加，行内偏移 HW prefetcher 已 cover）。bf16 SDPA 单线程 51 → 54–57 GFLOPS（74% → 78–83% BFMMLA-half 峰值）；fp32 ~不变（仍被 QKᵀ-fp32 单累加器卡住）。新增 `tests/test_sdpa_versions_equiv.py::test_sdpa_versions_equiv_with_attn_mask` 覆盖 mask 路径数值等价性 | 改 `csrc/sdpa_flash2_neon_l3kv_impl.h`、`csrc/sdpa_flash2_neon_l3kv.cpp`、`csrc/sdpa_flash2_neon_l3kv_packv.cpp`；改 `tests/test_sdpa_versions_equiv.py` |
| 2026-05-22 | **PV outer-loop 合并**：`process_q_tile_lc` 步骤 8 把 `Sc_cur/8` 次 `pv_8x8(Sk=8)` 调用合并成单次 `pv_8x8(Sk=Sc_cur)`——消掉 7×16=112 vld_O + 112 vst_O 的边界 round-trip + V cold-start cache miss。bf16 SDPA 单线程 47→51 GFLOPS（68%→74% peak），所有 8 个 SDPA 变体（cache / l3kv / l3kv_packv × baseline / scalar / pquad）都受益 | 改 `csrc/sdpa_flash2_neon_l3kv_impl.h` 步骤 8 |
| 2026-05-22 | 新增 `flash2_neon_l3kv_packv`（含 baseline / scalar / pquad 三个 MK 变体）：SDPA 入口 V layout 从 `[B,N,S,Ev]` 重排成 `[B,N,Ev/8,S,8]`，多线程 pack；`process_q_tile_lc / run_path_*` 抽到共享 header；bf16 +4–9% wall-clock | 新建 `csrc/sdpa_flash2_neon_l3kv_packv.cpp`、`csrc/sdpa_flash2_neon_l3kv_impl.h`；改 `csrc/sdpa_flash2_neon_l3kv.cpp` |
| 2026-05-20 | 新增 `MK_PQuad` trait：fp32 PV 8×8 P 标量 load 折成 quad load + lane FMA，LSU 指令数 -60%，PV +12–18% GFLOPS；自动注册 `flash2_neon_cache_pquad` / `flash2_neon_l3kv_pquad`（无 packv 时） | 改 `csrc/sdpa_microkernels/neon_cache_microkernels.h`（新增 `gemm_pv_microkernel_8x8_fp32_pquad`）；新建 `csrc/sdpa_microkernels/impls/mk_pquad.h`；改 `csrc/sdpa_microkernels/all_impls.h`、`csrc/sdpa_microkernels/mk_registry.cpp`、`csrc/sdpa_flash2_neon_cache.cpp`、`csrc/sdpa_flash2_neon_l3kv.cpp` |
