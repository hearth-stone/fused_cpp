# DeepSeek-V4 C++ 算子优化分析报告

## 概述

本文档针对 DeepSeek-V4 推理引擎中的关键 C++ 算子进行理论优化分析。

**目标平台**：aarch64 架构，支持 NEON/SVE 向量扩展，BF16 数据类型，OpenMP 多线程并行。

**分析范围**：
- 融合注意力输入 GEMM（parallel execute）
- Post Stage 多分支计算
- MoE 计算内核
- MoE 调度规划器

**分析方法**：纯理论分析与静态代码审查，基于代码结构、算法复杂度和硬件特性推断性能瓶颈与优化空间。所有建议均未经实测验证，需后续通过 benchmark 与等价性测试确认。

---

## 算子地图

下表列出各算子对应的源文件及其在 DeepSeek-V4 推理流程中的位置：

| 算子域 | 核心源文件 | 推理流程位置 | 主要功能 |
|--------|-----------|-------------|---------|
| **融合注意力输入 GEMM** | `deepseek_v4_attn_gemm_fused.cpp` | 注意力层入口 | 多 GEMM 并行执行（fused_wqa_wkv、compressor_kv_score、indexer_*）+ RMSNorm 后处理 |
| **Post Stage** | `deepseek_v4_post_gemm_parallel_stage.cpp` | 注意力 GEMM 之后 | Main Q 投影 + MLA Compressor + Indexer 分支 + Sparse Top-K |
| **MoE 计算内核** | `csrc/moe/arm/common/fused_moe_bf16_tiled.cpp` | MoE 层核心 | Token 路由 + W13/W2 两段 GEMM + 激活函数 + 加权合并 |
| **MoE 调度规划** | `moe_planner/*.cpp` | MoE 执行前 | 根据 token 分布规划 expert 分配、线程调度、wave 打包 |

**推理流程顺序**：
```
输入 hidden_states
    ↓
[融合注意力输入 GEMM] → qr, kv, kv_score, indexer_*
    ↓
[注意力计算] → attn_output
    ↓
[Post Stage] → compressed_kv, indexer_kv, sparse_indices
    ↓
[MoE 路由] → topk_ids, topk_weights
    ↓
[MoE 调度规划] → PlanResult (wave/team 配置)
    ↓
[MoE 计算内核] → moe_output
    ↓
输出
```

---

## 一、融合注意力输入 GEMM（Parallel Execute）

### 1.1 现状

**融合路径**：
- **Dense 路径**（`fused_wqa_wkv`）：单 GEMM `hidden @ fused_wqa_wkv_weight.T` → `qr_kv [M, q_lora_rank + kv_dim]`
- **C128A 路径**：并行执行 2 个 GEMM（fused_wqa_wkv + compressor_kv_score）
- **C4A 路径**：并行执行 4 个 GEMM（fused_wqa_wkv + compressor_kv_score + indexer_compressor_kv_score + indexer_weights_proj）

**多线程切分**：
- Token 级别行划分：`rows_per_thread = ceil_div(M, num_threads)`
- 每线程独立 scratch buffer，避免 false sharing
- 支持 core 亲和性绑定（`pthread_setaffinity_np`）

**GEMM 形状**（典型值）：
| GEMM | M | K | N |
|------|---|---|---|
| fused_wqa_wkv | 1~1024 | 7168 | 2048+512 |
| compressor_kv_score | 1~1024 | 7168 | 128 |
| indexer_compressor_kv_score | 1~1024 | 7168 | 128 |
| indexer_weights_proj | 1~1024 | 7168 | 256 |

**权重打包**：
- 格式：`[K/4][N/8][4][4][2]`，适配 4×8 microkernel
- 8 字节对齐（K_pad, N_pad）

**SVE 向量化**：
- RMSNorm sumsq 使用 `svbfdot_f32`（BF16 dot product）
- RoPE 使用 `svld2_u16` 加载 even-odd 对，减少访存
- 4 路累加器展开提高指令级并行

### 1.2 理论瓶颈

1. **内存带宽受限**：
   - 算术强度 AI ≈ 0.21 FLOP/Byte（M=256, K=7168, N=2048）
   - 多 GEMM 共享输入 `hidden_states`，但每次独立读取

2. **BFMMLA 利用不足**：
   - 当前未显式使用 BFMMLA 指令，仅用通用 bf16 GEMM
   - RMSNorm 已用 SVE BF16 DOT，但 GEMM 核心未跟进

3. **GEMM-RMSNorm-RoPE 内存往返**：
   - GEMM 输出写回内存 → RMSNorm 读取 → 写回 → RoPE 读取
   - 多次往返增加延迟

4. **静态负载均衡**：
   - `rows_per_thread` 简单静态划分
   - 对变长序列或异构 core 性能可能不均

### 1.3 优化建议

| 优化点 | 建议 | 预期收益 | 风险 |
|--------|-----|---------|------|
| **多 GEMM 输入复用** | 预取 `hidden_states` 到 L2，软件预取或分块使 tile 驻留 L1 | 减少 50-75% 输入读取带宽 | 需调优 tile size，增加代码复杂度 |
| **BFMMLA Microkernel** | 集成 KleidiAI 或手写 SVE-BF16 GEMM（`svbfdot_f32`/`svbfmlalb_f32`） | 2-4x GEMM 吞吐提升 | 需特定 CPU 支持（Neoverse V1/V2/N2） |
| **RMSNorm+RoPE 融合到 GEMM 输出** | 自定义 GEMM epilogue，在输出累加器上直接做 Norm/RoPE | 减少 1-2 次内存往返，节省 10-20% 延迟 | 增加 microkernel 复杂度 |
| **权重布局优化** | 按 L2 大小分块 `[K_tile][N_tile]`，确保 N_tile 驻留 L1 | 提高 cache 命中率 10-30% | 需重新设计打包格式 |
| **动态负载均衡** | 使用 OpenMP `schedule(dynamic)` 或按 token 长度加权分配 | 混合 batch 下提升 10-20% 吞吐 | 动态调度增加同步开销 |

### 1.4 总结

当前实现结构良好，SVE 向量化和 OpenMP 并行已到位。主要优化空间在：
1. 内存带宽优化（多 GEMM 输入复用、权重分块）
2. SIMD 优化（BFMMLA microkernel 集成）
3. 融合深度（GEMM 输出与 RMSNorm/RoPE 深度融合）
4. 并行策略（动态负载均衡）



---

## 二、Post Stage

### 2.1 现状

**三路分支架构**：

1. **Main Q Branch**（所有变体必需）：
   - GEMM：`wq_b(qr)` → `q [T, num_heads, head_dim]`
   - Per-head RMSNorm（无权重）
   - RoPE（GPTJ-style）
   - SWA KV Cache Insert

2. **MLA Compressor Branch**（c128a/c4a）：
   - Save Partial States → `state_cache`
   - Compress（128-way 或 8-way softmax 加权）
   - Norm + RoPE + KV Cache Insert

3. **Indexer Branch**（仅 c4a）：
   - GEMM：`indexer_wq_b(qr)`
   - Q RoPE + Weight Scaling
   - Indexer Compressor
   - Sparse Top-K（short path 直接填充，long path gather + matmul + topk）

**已融合 Kernel**：
- `QNormRopeFused`：per-head RMSNorm + RoPE 融合，4x 行并行
- `KvRopeCacheInsertFused`：RoPE + cache write 融合，支持间接寻址
- `CompressRowsNormRopeInsert`：coff=1/2 的 softmax + 加权 + Norm + RoPE + cache write 全融合

### 2.2 理论瓶颈

1. **GEMM 输出与后处理内存往返**：
   - Main Q GEMM 输出 `[T, num_heads, head_dim]` 写回内存，再被 Norm/RoPE 读取
   - 对于 M 较小的 GEMM，内存带宽占主导

2. **Sparse Indexer Long Path 临时内存**：
   - `k_gathered [total_seq_lens, head_dim]` 可能很大（长序列 prefill）
   - Gather 操作内存受限（间接寻址）

3. **Compressor 并行度受限**：
   - `active_indices` 数量取决于 `compress_ratio`，可能较少
   - 两个 compressor（MLA + Indexer）串行执行

4. **Gather/Scatter Cache Miss**：
   - `block_table` 查找导致物理地址跳跃
   - L1/L2 缓存命中率低

### 2.3 优化建议

| 优化点 | 建议 | 预期收益 | 风险 |
|--------|-----|---------|------|
| **Main/Indexer Q GEMM + Norm/RoPE 融合** | 自定义 GEMM epilogue，输出时直接应用 Norm/RoPE | 减少 1 次 `[T, num_heads, head_dim]` 内存读写（约 6MB） | 增加 GEMM kernel 复杂度，需重新调度寄存器 |
| **Sparse Indexer Long Path 分块 + 在线 Top-K** | 分块 gather + matmul，计算 logits 同时维护 top-k heap | 减少临时内存（`k_gathered`），降低内存带宽压力 | 实现复杂度高，需仔细设计并行策略 |
| **Compressor 并行化** | 使用 OpenMP sections 并行执行 MLA 和 Indexer compressor | 并行度提升，总延迟降低 | 代码复杂度增加，需处理不同参数 |
| **Thread 数调优** | 根据 prefill chunk size 动态调整线程数（M<=256 时限制为 4） | 避免过度并行化开销 | 需实验确定最佳阈值 |
| **Short Path 批量化** | 按相同 `valid_len` 分组 token，批量填充索引 | 减少循环开销 10-20% | 需要额外分组逻辑 |

### 2.4 总结

Post Stage 已有多处 SVE 融合 kernel，优化重点在：
1. GEMM 与后处理的深度融合（减少内存往返）
2. Sparse Indexer Long Path 的内存优化
3. Compressor 并行化


---

## 三、MoE 计算内核

### 3.1 现状

**算法流程**：
```
Input [T, H] → 路由分组 routes[E] → TileTask 列表
           ↓
       Gather token → W13 GEMM → SiLU×gate → W2 GEMM → Scatter to route_out
           ↓
       Weighted merge → Output [T, H]
```

**路由与分组**：
- 入口构建按 expert 分组的 `routes` 表
- TileTask 以 `kMoeTokenTile = 16` 分块
- Expert 按 token 数排序，大 token 专家优先调度

**两段 GEMM**：
- **W13**：`input [rows, H] × W13^T [2F, H]^T → gate_up [rows, 2F]`（fp32 累加）
- **激活函数**：`silu_and_mul_to_bf16` 等，标量循环实现，无向量化
- **W2**：`intermediate [rows, F] × W2^T [H, F]^T → down [rows, H]`（fp32 累加）

**调度策略**：
| 策略 | 适用场景 |
|------|---------|
| `expert_affinity_greedy` | 小 batch、token 分布均匀 |
| `hierarchical_mn_split` | 大 batch、NUMA 多核、TP=4 |
| `external_plan_scheduled` | 外部 planner 提供调度计划 |

**权重 Prepack**：
- 转置 + `bf16_pack_b` 重排为 BFMMLA 友好布局
- 环境变量 `FUSED_CPP_MOE_PREPACK_THREADS` 控制多线程

### 3.2 理论瓶颈

1. **激活函数未向量化**：
   - `silu_and_mul_to_bf16` 是标量实现
   - 约占 5-10% 时间

2. **Gather/Scatter 开销**：
   - 标量循环，未向量化
   - 路由索引 `flat / top_k` 每次有除法开销

3. **动态负载均衡竞争**：
   - `hierarchical_mn_split` 用 `std::atomic<size_t>` 做 work-stealing
   - 高并发下 atomic 竞争

4. **同步点过多**：
   - 每次 expert 计算 4 次 barrier（gather 后、w13 后、activation 后、w2 后）

5. **路由表重复构建**：
   - 每次 forward 重建 `routes` 表
   - `topk_ids` 在 decode 阶段可复用

6. **Roofline 分析**（典型配置）：
   - `num_tokens=2048, hidden_size=4096, experts=256, top_k=6`
   - 单次 forward FLOPs ≈ 1.5 TFLOPs
   - 权重内存访问 ≈ 4GB，权重带宽主导

### 3.3 优化建议

| 优化点 | 建议 | 预期收益 | 风险 |
|--------|-----|---------|------|
| **激活函数向量化** | 用 NEON BFMMLA 或 FMLA 向量化 SiLU | 激活阶段加速 3-5x（整体 5-10%） | SiLU 近似多项式精度问题 |
| **Gather/Scatter 向量化** | NEON 批量加载/存储 + 预取 | 10-15% 带宽改善 | 需处理非连续访问 |
| **路由索引压缩** | 预存 `(token, slot)` 二元组，避免运行时除法 | 减少除法开销 | 增加内存占用 |
| **路由表外部化与缓存** | 将 `routes` 构建移到 Python 层或 planner，支持缓存 | 减少 5-10% CPU 时间 | 需要框架配合 |
| **动态负载均衡改进** | per-group task queue + 无锁 steal，或大 expert 拆分 | 不均衡场景下 20% 改善 | 实现复杂度高 |
| **减少同步点** | 流水线化或 double-buffering | 低延迟场景收益 | 增加 kernel 复杂度 |
| **Buffer 尺寸优化** | per-expert 动态分配或分级 buffer pool | 减少 30-50% scratch 内存 | 增加管理开销 |

### 3.4 与 MoE Planner 的耦合

**当前耦合点**：
1. 路由表构建在 kernel 内部，planner 无法复用
2. 动态调度状态（`next_expert_idx`）是运行时状态
3. Trace 收集（`MoeTraceCollector`）在 kernel 内部

**优化建议**：
- **调度计划持久化**：decode 阶段 `topk_ids` 不变，缓存 `PlanResult`
- **Trace 与 Planner 反馈闭环**：结构化输出 trace，planner 调优 cost model


---

## 四、MoE 调度规划器（moe_planner）

### 4.1 现状

**Cost Model 设计**：
- **Synthetic**：闭合公式 `T_expert = serial(routes) / speedup(useful) + overhead(threads)`
  - `serial = 18000 + 1850 * routes`
  - `speedup = 1 + 0.82 * (useful-1)^0.72`
- **Table**：基于实测 (routes, threads) → ns 映射表，支持 nearest-bucket fallback

**Planner 策略**：
| 策略 | 算法 | 复杂度 | 规划开销 | 适用场景 |
|------|------|--------|---------|---------|
| `FIXED_GLOBAL_THREADS` | 每个 expert 1 线程，顺序打包 | O(A) | < 1μs | Baseline |
| `SORTED_TOKEN_BALANCED_1T` | LPT 贪心负载均衡 | O(A*C) | 1-5 μs | 均匀负载 |
| `UNIFORM_WAVES` | 枚举 threads_per_expert | O(A*C²) | 5-20 μs | 简化模型 |
| `LOAD_PROPORTIONAL` | weight=routes | O(A) | 1-3 μs | 线性 scaling |
| `SQRT_LOAD` | weight=√routes | O(A) | 2-5 μs | 非线性 scaling |
| `LOG_LOAD` | weight=1+ln(routes) | O(A) | 2-5 μs | 边际收益递减 |
| `HEAVY_LIGHT_HYBRID` | Heavy/Light 分类 + FFD | O(A log A) | 3-10 μs | Heavy-tail |
| `GREEDY_MARGINAL_GAIN` | 边际收益贪心 + SIMD argmax | O(B*S*A) | 20-100 μs | 高精度调度 |
| `KARMARKAR_KARP` | Multiway LDM | O(A²*C) | 50-200 μs | 最优近似 |
| `ENUMERATE_CORE_GROUPS` | 枚举整数划分（≤512） | O(S*A) | 30-100 μs | 形状精确 |

**SIMD 优化**：
- `planner_simd.h` 提供 SVE/NEON/scalar 三种 argmax 实现
- SVE 用 vectorized max-reduce + first-true scan

### 4.2 理论瓶颈

1. **Cost Model 精度问题**：
   - Synthetic 参数（0.72 指数、1850 系数）是经验值，跨硬件偏差 30-50%
   - Table 模型 nearest-bucket fallback 在边界误差最大（30%）
   - 未区分 GEMM M-split vs N-split

2. **规划开销与 T_execute 权衡**：
   - 当 T_plan > T_execute 的 5-10% 时应选 cheaper planner
   - 当前 AUTO selector 基于此原则，但阈值硬编码

3. **无缓存机制**：
   - `enumerate_core_group_shapes()` 结果是纯函数，可缓存
   - decode 阶段连续相同 `routes_hist` 时重复构建 cost_rows

4. **计划粒度限制**：
   - Expert 级粒度，无 tile-level 分割
   - 大 routes expert（>1000）限制了并行度

5. **无 NUMA/Cache-aware 调度**：
   - `thread_cpu_ids` 字段未使用
   - 跨 NUMA node 或 cross-cluster 访问延迟差异显著

### 4.3 优化建议

| 优化点 | 建议 | 预期收益 | 风险 |
|--------|-----|---------|------|
| **Cost Model 在线校准** | Warmup 阶段采样 3-5 点，线性插值修正 synthetic 参数 | 误差从 30-50% 降至 10-15% | 增加 ~100μs 启动开销 |
| **Workload 缓存** | 缓存 `Workload` 对象，decode 阶段连续相同 routes_hist 复用 | 开销从 O(A*C) 降至 O(1) 检查 | 需处理 cache invalidation |
| **SORTED_TOKEN_BALANCED_1T heap 优化** | O(A*C) → O(A log C) | 降低规划开销 | 实现简单 |
| **SIMD 化 table lookup** | `prepare_workload()` 中 table 模型并行化 | 加速 cost 计算 | SVE gather/scatter 依赖 |
| **Shape 缓存** | `enumerate_core_group_shapes()` 结果缓存到 static 变量 | 减少重复计算 | 需保证线程安全 |
| **Expert-to-core affinity** | 维护 expert → preferred core 映射，基于 weight cache 状态 | NUMA 系统提升 10-30% | 需要 kernel 配合，复杂度高 |
| **Tile-level planning** | 大 expert 分割为多个 tile，每个 tile 独立 team | 大 expert 场景并行度 2-4x | 需 kernel 大改 |
| **Historical smoothing** | decode 阶段维护指数平滑 `routes_hist_ema`，相似时复用 plan | 减少 decode 阶段规划调用 | 收益不确定，需调参 |


---

## 五、跨算子共性优化点

通过对四个算子域的分析，归纳出以下重复出现的优化主题：

### 5.1 BF16 Microkernel 利用率

**共性**：
- 融合注意力 GEMM、Post Stage GEMM、MoE W13/W2 均依赖 bf16 GEMM
- 当前使用通用实现，未充分调用 BFMMLA 指令

**统一建议**：
- 集成 KleidiAI 或 ARM ACL 的 BFMMLA microkernel
- 对支持 SVE-BF16 的 CPU（Neoverse V1/V2/N2），使用 `svbfdot_f32`/`svbfmlalb_f32`
- 预期收益：2-4x GEMM 吞吐提升

### 5.2 权重 Prepack/对齐

**共性**：
- 所有 GEMM 均预打包权重（`bf16_pack_b`）
- 格式：4×8 分块，K 按 4 分组，N 按 8 分组
- 但未考虑 L1/L2 cache 大小

**统一建议**：
- 按 L2 大小分块 `[K_tile][N_tile]`
- 确保 N_tile 能驻留 L1（如 32KB）
- 使用 `prfm` 预取下一 tile
- 预期收益：cache 命中率提升 10-30%

### 5.3 OpenMP 切分与负载均衡

**共性**：
- 均使用静态划分（`rows_per_thread` 或 expert 分配）
- 对变长序列或 heavy-tail routing 不均衡

**统一建议**：
- 引入动态调度（`schedule(dynamic)` 或 work-stealing）
- 按实际负载（token 数或 routes 数）加权分配
- 对异构 core（如 Apple P/E core）差异化分配
- 预期收益：不均衡场景下 10-30% 改善

### 5.4 内存往返/融合

**共性**：
- GEMM 输出与后处理（Norm/RoPE）分离，增加内存往返
- MoE 的 gather/scatter 未向量化

**统一建议**：
- GEMM epilogue 融合：在输出累加器上直接应用后处理
- Gather/Scatter 向量化：NEON 批量加载/存储 + 预取
- 预期收益：减少 1-2 次内存往返，节省 10-20% 延迟

### 5.5 Roofline 带宽瓶颈

**共性**：
- 融合注意力 GEMM：AI ≈ 0.21 FLOP/Byte，内存带宽受限
- MoE：权重带宽主导（4GB 权重，权重流式加载）

**统一建议**：
- 权重重用：expert-to-core affinity，尽量让同一 core 连续处理同一 expert
- 权重流式加载：按 wave 加载，或 mmap 文件映射 + 预取
- 输入驻留：多 GEMM 共享输入时，预取到 L2

### 5.6 Core 亲和与 NUMA 感知

**共性**：
- 融合注意力 GEMM 已实现 core 绑定（`pthread_setaffinity_np`）
- Post Stage 和 MoE kernel 依赖 OpenMP runtime 调度
- Planner 的 `thread_cpu_ids` 字段未实际使用

**统一建议**：
- 扩展 core 亲和机制到所有算子
- NUMA-aware 调度：expert 分配到权重所在 NUMA node
- Cache-aware wave building：权重有 overlap 的 experts 放同 wave
- 预期收益：NUMA 系统提升 10-30%


---

## 六、优化优先级建议

按收益、实现成本、风险排序，区分「低风险快速收益」与「高收益高成本」：

### 6.1 高优先级（P0）— 低风险快速收益

| 优化点 | 涉及算子 | 预期收益 | 实现成本 | 风险 |
|--------|---------|---------|---------|------|
| **BFMMLA Microkernel 集成** | 全局 GEMM | 2-4x GEMM 吞吐 | 中（集成 KleidiAI） | 低（独立模块） |
| **Workload/Shape 缓存** | MoE Planner | decode 阶段开销降至 O(1) | 低（加缓存层） | 低 |
| **SORTED_TOKEN_BALANCED_1T heap 优化** | MoE Planner | O(A*C) → O(A log C) | 低（改数据结构） | 低 |
| **路由表外部化与缓存** | MoE Kernel | 减少 5-10% CPU 时间 | 低（移到 Python 层） | 低 |
| **激活函数向量化** | MoE Kernel | 整体 5-10% 加速 | 中（SIMD 实现） | 中（精度问题） |
| **Gather/Scatter 向量化** | MoE Kernel | 10-15% 带宽改善 | 中 | 中 |

### 6.2 中优先级（P1）— 中等成本

| 优化点 | 涉及算子 | 预期收益 | 实现成本 | 风险 |
|--------|---------|---------|---------|------|
| **Cost Model 在线校准** | MoE Planner | 误差从 30-50% 降至 10-15% | 中（加校准逻辑） | 中（启动开销） |
| **GEMM + Norm/RoPE 融合** | 融合 GEMM, Post Stage | 减少 1-2 次内存往返 | 高（改 microkernel） | 高 |
| **Sparse Indexer Long Path 分块 + 在线 Top-K** | Post Stage | 减少临时内存，降低带宽 | 高 | 高 |
| **动态负载均衡改进** | 全局 | 不均衡场景 10-30% 改善 | 中 | 中 |
| **Buffer 尺寸优化** | MoE Kernel | 减少 30-50% scratch 内存 | 中 | 中 |
| **权重布局优化（L2 分块）** | 全局 GEMM | cache 命中率 10-30% | 中（改打包格式） | 中 |

### 6.3 低优先级（P2）— 高成本或收益不确定

| 优化点 | 涉及算子 | 预期收益 | 实现成本 | 风险 |
|--------|---------|---------|---------|------|
| **Expert-to-core affinity** | MoE Kernel + Planner | NUMA 系统 10-30% | 高（需 kernel 配合） | 高 |
| **Tile-level planning** | MoE Planner + Kernel | 大 expert 场景 2-4x 并行度 | 高（kernel 大改） | 高 |
| **减少同步点（流水线化）** | MoE Kernel | 低延迟场景收益 | 高 | 高 |
| **Historical smoothing** | MoE Planner | 减少 decode 规划调用 | 中 | 高（收益不确定） |
| **Short Path 批量化** | Post Stage | 10-20% 循环开销减少 | 中 | 中 |


---

## 七、风险与验证建议

### 7.1 理论分析局限性

本文档所有优化建议基于：
- 静态代码审查与算法复杂度分析
- 硬件特性推断（aarch64 NEON/SVE、BF16、OpenMP）
- Roofline 模型估算

**未考虑因素**：
- 实际硬件的 cache 层次结构、带宽、延迟
- OS 调度器行为、NUMA 拓扑
- 混合负载下的资源竞争
- 编译器优化效果
- 数值精度与数值稳定性

因此，所有建议需通过实测验证。

### 7.2 验证方法

#### Benchmark 复用

仓库已有的基准测试框架可直接复用：

```bash
# SDPA 多版本基准
pytest -m bench -s tests/bench_sdpa_versions.py

# MoE 基准（假设存在）
pytest -m bench tests/bench_moe_*.py

# 融合 GEMM 基准（假设存在）
pytest -m bench tests/bench_attn_gemm_*.py
```

建议扩展基准测试以覆盖：
- 不同 batch size（1, 16, 64, 256, 1024）
- 不同序列长度（128, 512, 2048, 4096）
- 不同 token 分布（均匀、heavy-tail）
- 不同线程数（1, 2, 4, 8, 16, 32, 64）
- 不同硬件（Neoverse N1/N2/V1、Apple M 系列）

#### 等价性测试

仓库已有的等价性测试框架：

```bash
# SDPA 等价性矩阵
pytest -m equiv tests/test_sdpa_versions_equiv.py

# MoE 等价性测试（假设存在）
pytest -m equiv tests/test_moe_*.py
```

优化后需确保：
- 数值精度：bf16 输出与参考实现（fp32 累加后转 bf16）误差在容忍范围内
- 边界情况：M/N/K 非对齐、变长序列、极端路由分布

#### 性能回归检测

建议建立性能回归检测机制：
1. 基线：优化前的 benchmark 结果
2. 目标：优化后 benchmark 结果
3. 阈值：避免过度优化导致某些场景退化

### 7.3 分阶段实施建议

**Phase 1**（快速验证）：
- 实现 P0 低风险优化（Workload 缓存、heap 优化、激活函数向量化）
- 运行 benchmark 验证收益
- 预期时间：1-2 周

**Phase 2**（核心优化）：
- 集成 BFMMLA microkernel
- 实现 GEMM + Norm/RoPE 融合
- 实现动态负载均衡
- 预期时间：4-6 周

**Phase 3**（高级优化）：
- Expert-to-core affinity
- Tile-level planning
- Cost model 在线校准
- 预期时间：6-8 周

每个阶段后运行完整 benchmark 套件与等价性测试。

### 7.4 风险缓解

| 风险类型 | 缓解措施 |
|---------|---------|
| **数值精度** | 保留 fp32 累加路径作为 fallback；增加精度测试用例 |
| **硬件依赖** | 编译时检测 CPU 特性（`__ARM_FEATURE_SVE_BF16`），无特性时降级 |
| **负载不均衡** | 动态调度 + 静态调度双路径，运行时根据负载选择 |
| **规划开销过大** | AUTO selector 根据预估 T_plan/T_execute 比例选择 planner |
| **内存占用增加** | 分级 buffer pool，按需分配；监控峰值内存 |

---

## 附录：关键代码位置索引

```text
融合注意力 GEMM:
  deepseek_v4_attn_gemm_fused.cpp  : 主入口，多 GEMM 并行，RMSNorm 后处理
  bf16_pack_b()                     : 权重打包函数

Post Stage:
  deepseek_v4_post_gemm_parallel_stage.cpp : 三路分支入口
  deepseek_v4_prefill_cache.cpp             : Gather/Scatter KV cache
  mla_compressor.cpp                        : Compressor 逻辑

MoE Kernel:
  moe/arm/common/fused_moe_bf16_tiled.cpp : MoE 主入口，三种调度策略
  deepseek_v4_attn_gemm_fused.cpp   : bf16gemm microkernel 定义

MoE Planner:
  moe_planner/cost_model.h          : CostModel 类（synthetic/table）
  moe_planner/planner_types.h       : PlanKind, PlanResult 定义
  moe_planner/planner_common.h/cpp  : Workload, prepare_workload()
  moe_planner/planner_simd.h        : SIMD argmax 实现
  moe_planner/planner_dispatch.cpp  : 统一入口 run_planner()
  moe_planner/plan_*.cpp            : 各策略实现
```

---

**文档版本**：2026-06-30
**分析方法**：纯理论/静态代码分析，未实测
**目标平台**：aarch64 NEON/SVE + BF16 + OpenMP
