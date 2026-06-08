# Sparse MLA Hybrid NEON 设计草案

> 状态：设计草案，尚未实现。
>
> 目标：把当前 Python naive sparse MLA reference 升级为 C++/NEON 路径，在
> sparse indices 中可识别为 dense 的连续 KV 段时复用现有 dense attention
> microkernel；非完全共享 KV 段统一走 indexed/gathered sparse NEON kernel。

---

## 1. 背景

当前 sparse MLA 路径在 `src/fused_cpp/sparse_mla.py` 中是纯 PyTorch
reference。它模拟 vLLM DeepSeek V4 CPU sparse attention fallback：

- `q`: `[s_q, h_q, d_qk]`
- `kv`: `[s_kv, 1, d_qk]`
- `indices`: `[s_q, 1, topk]`
- `kv[..., :d_v]` 作为 V；完整 `kv[..., :d_qk]` 作为 K
- `indices < 0` 是 padding
- 正数越界不屏蔽，应像 `narrow` / `index_select` 一样报错
- `topk_length` 为签名兼容参数，当前语义忽略
- `attn_sink` 是额外 zero-value key，只影响 softmax denominator

现有 dense SDPA 已经有较完整的 ARM NEON microkernel：

- QK: `qkt_8x8` / `qkt_8x4` / `qkt_tail`
- PV: `pv_8x8` / `pv_tail`
- 调度主体以 8 行 Q 为计算单元，维护 fp32 online softmax 状态
- 相关文件：
  - `csrc/sdpa_flash2_neon_cache.cpp`
  - `csrc/sdpa_flash2_neon_l3kv_impl.h`
  - `csrc/sdpa_microkernels/neon_cache_microkernels.h`

本设计的核心判断：sparse MLA 不能简单地“dense 部分单独 softmax、sparse
部分单独 softmax、最后相加”。dense 段和 sparse 段必须进入同一个 online
softmax accumulator，才能保持全局 softmax 语义。

---

## 2. 目标与非目标

### 2.1 目标

1. 保持当前 `sparse_mla_naive` 的输入校验、padding、越界和返回值语义。
2. 新增 C++ extension 路径，Python wrapper 优先调用 C++，不可用时回退 naive。
3. 计算按 query 顺序推进，每次处理连续 8 个 query token。
4. 在 8-query block 内，先计算 8 行完全共享的连续 KV run。
5. 在共享 KV 之后，继续计算所有非完全共享 KV，包括部分共享和完全不共享，
   统一使用 indexed/gathered 4x4 / 4xN sparse NEON kernel。外层仍是
   8-query block，内部拆成两个 4-row 子块。
6. 非完全共享 indexed sparse QKT/PV 默认使用 fp32 FMLA；bf16 输入先 widen
   到 fp32。除非目标机器 bf16 BFMMLA 峰值超过 fp32 FMLA 峰值的 2 倍，否则
   不考虑 indexed sparse BFMMLA 路径。
7. 所有完全共享和非完全共享 segment 都共享同一套 online softmax 状态。
8. 支持 bf16 和 fp32 输入，内部 fp32 累加，输出 cast 回输入 dtype。
9. 对短小、不规则尾部提供标量 fallback，先保证正确性，再做性能优化。

### 2.2 非目标

1. 第一版不支持 `h_kv != 1`。保持当前 reference 的限制。
2. 第一版不改变 `topk_length` 语义，仍忽略它。
3. 第一版不做 paged KV cache/block table 直接读取，只处理已物化的
   `[s_kv, 1, d_qk]`。
4. 目前完全不考虑反向传播：不实现 backward kernel，不注册 autograd
   backward，也不为训练路径预留额外接口。
5. 第一版不尝试全局重排 `indices`，因为重复 index 和原始顺序会影响 softmax
   计数语义。

---

## 3. 对外 API

Python 侧保留现有 API：

```python
flash_mla_sparse_fwd(
    q,
    kv,
    indices,
    sm_scale,
    d_v=None,
    attn_sink=None,
    topk_length=None,
    out=None,
)
```

新增 C++ binding：

```cpp
std::tuple<at::Tensor, at::Tensor, at::Tensor> flash_mla_sparse_fwd(
    at::Tensor q,
    at::Tensor kv,
    at::Tensor indices,
    double sm_scale,
    c10::optional<int64_t> d_v,
    c10::optional<at::Tensor> attn_sink,
    c10::optional<at::Tensor> topk_length,
    c10::optional<at::Tensor> out);
```

返回：

- `output`: `[s_q, h_q, d_v]`
- `max_logits`: `[s_q, h_q]`
- `lse`: `[s_q, h_q]`

注意：为了匹配当前 naive，`max_logits` 和 `lse` 只统计真实 sparse KV logits，
不包含 `attn_sink`。`attn_sink` 只参与最终 output 的 softmax denominator。

---

## 4. 核心数值模型

每个 query row 独立维护：

```text
m     = running max
l     = running exp sum
O_acc = unnormalized output accumulator, shape [d_v]
```

每处理一个 KV segment，先得到该 segment 的 logits `scores[j]`：

```text
row_max = max(scores)
m_new   = max(m, row_max)
c_old   = exp(m - m_new)

O_acc *= c_old
l     *= c_old

p_j    = exp(scores[j] - m_new)
O_acc += sum_j p_j * V_j
l     += sum_j p_j
m      = m_new
```

所有完全共享和非完全共享 segment 都调用同一个 update 流程。
最终：

```text
out = O_acc / l
```

如果某行没有任何有效 KV：

- `output = 0`
- `max_logits = -inf`
- `lse = +inf`

`attn_sink` 处理方式：

```text
sink_score = attn_sink[head]
m_new      = max(m, sink_score)
c_old      = exp(m - m_new)
l          = l * c_old + exp(sink_score - m_new)
O_acc      = O_acc * c_old
m          = m_new
```

因为 sink 对应 zero-value V，所以不更新 `O_acc += ...`。

---

## 5. 分段策略

### 5.1 输入 indices 的 run 表示

对每个 token 的有效 indices，按原始顺序切成 run：

```text
[0, 1, 2, 3, 8, 11, 12]
=> dense run [0, len=4], singleton [8], dense run [11, len=2]
```

负数 padding 被过滤。正数越界不提前屏蔽；读取时让 C++ check 报错，保持
reference 行为。

重复 index 不做去重：

```text
[5, 5, 6]
=> singleton [5], dense run [5, len=2] 或直接 indexed tile
```

重复 key 在 softmax 中代表两个位置，必须保留。

### 5.2 8-query block 的执行顺序

主计算单元为 8 个 query token：

```text
q_block = q[token0 : token0 + 8, head, :]
```

外层按 `token0 = 0, 8, 16, ...` 顺序推进，不因为 KV sharing 情况重排 query。
每个 block 内部按共享度分两类 segment：

1. **完全共享 KV**：8 行都包含同一个连续 KV run。
2. **非完全共享 KV**：包括部分共享和完全不共享。每个 query row 可以有自己
   的 K index。

这两类 segment 可以按“完全共享 -> 非完全共享”的顺序喂给同一个
online softmax accumulator。注意这里不重排 query 输出，也不去重 index；只是把
同一个 query block 内的 KV 计算按共享度分组，以便使用更合适的 microkernel。

### 5.3 完全共享 KV: 8x8 / dense-style path

如果这 8 行在当前位置都有同一个连续 KV run：

```text
token t0: [100, 101, ..., 127]
token t1: [100, 101, ..., 127]
...
token t7: [100, 101, ..., 127]
```

则把 `[100, 128)` 作为 dense-run segment：

- Q 是 8 行连续 query
- K/V 是 kv 中连续 span
- 第一版只调用 dense-style `qkt_8x8` / `pv_8x8` 主体
- softmax/PV 状态仍是 sparse MLA 自己的 accumulator

这个路径直接复用现有 dense attention 的 8x8 思路：

- Q 维：8 个 query row
- K 维：连续 KV span，主体按 8 列处理
- QK 主体：8x8
- PV 主体：8x8
- 第一版 `DenseSeg.len` 向下取 8 的倍数，只跑 8x8 主体；不足 8 的尾部
  放入 indexed tile。后续再考虑 dense 8x4/tail 是否值得。

### 5.4 非完全共享 KV: indexed/gathered 4x4 / 4xN path

非完全共享包括部分共享和完全不共享。统一表达成每个 query row 自己的
K index tile。外层仍然一次处理 8 个 query，但 indexed sparse kernel 的物理
形状先用 4 行 Q：

```text
8-query block:
  rows 0..3 -> indexed_4x4 / indexed_4xN
  rows 4..7 -> indexed_4x4 / indexed_4xN

Q_sub:   [4, E]
idx:     [4, Kt]       # Kt 先取 4，tail 支持 1..3
valid:   [4, Kt]
scores:  [4, Kt]
```

Q 这边按 4 行子块正常 load 到寄存器。K 这边不再假设同一个 K tile 会和所有
Q row 相乘，而是对每个 `(sub_row, k_col)` 使用自己的 index：

```text
score[i, j] = dot(Q[i], KV[idx[i, j]]) * scale
```

这样一类 kernel 可以覆盖：

- **部分共享**：多行的 `idx[i, :]` 可以相同或部分相同。
- **完全不共享**：每行的 `idx[i, :]` 都不同。
- **带 padding**：`valid[i, j] = false` 的位置不参与 softmax/PV。

推荐新增 sparse MLA 专用 kernel：

- `qkt_indexed_4x4`: 4 行 Q，每行 4 个独立 K index。
- `qkt_indexed_4xN`: 4 行 Q，每行 N 个独立 K index，N 可取 1..3。
- `pv_indexed_4x4_ev8`: 4 行 Q，每行 4 个概率，Ev 方向每次处理 8 列。
- `pv_indexed_4xN_ev8`: tail 版本，N 可取 1..3。

PV 使用同一份 `idx/valid`，只对 valid 位置累加 output。

这个 kernel 仍然是类似 GEMM 的计算方法：Q row 常驻寄存器，K row 被 load 后
与对应 Q row 做 dot。区别在于 dense 8x8 里一个 K load 可以服务多个 Q row；
indexed kernel 里 K load 通常只服务一个 Q row，部分共享时才有机会复用。

第一版优先实现 4x4，而不是 8x4：

- 4x4 的 QKT 累加器、K 指针、idx/valid 临时量都明显少于 8x4，更不容易
  spill。
- `pv_indexed_4x4_ev8` 的 O accumulator 是 4 行 x 8 列，只需要 8 个
  `float32x4_t`，比 8 行版本的 16 个 accumulator 更稳。
- softmax 不会因为 4-row 子块而不匹配；softmax state 是逐 query row 独立的
  `m[row] / l[row] / O_acc[row, :]`，4x4 只更新其中 4 行，另外 4 行保持不动。
- `Kt < 4` 走 4xN tail；`Kt = 8` 可由两个 4x4 组成，后续再评估专用 4x8
  或 8x4 是否值得。

### 5.5 indexed sparse arithmetic policy

非完全共享 indexed sparse path 默认使用 fp32 FMLA，而不是 bf16 BFMMLA：

- `qkt_indexed_4x4` / `qkt_indexed_4xN`: bf16 Q/K 输入先 widen 到 fp32，
  使用 `vfmaq_f32` 累加 score。
- `pv_indexed_4x4_ev8` / `pv_indexed_4xN_ev8`: `P_hat` 本来就是 fp32，
  V 如果是 bf16 则 widen 到 fp32，使用 `vfmaq_f32` 累加 `O_acc`。

理由：

- BFMMLA 强依赖规整的 2x2 / tile layout。indexed sparse 中
  `score[i, j] = dot(Q[i], K[idx[i, j]])`，每个 `(i, j)` 的 K row 都可能不同；
  为了喂 BFMMLA 做 gather/pack/permute 的代价很容易超过收益。
- fp32 FMLA 对 indexed load 更直接：Q row 常驻寄存器，K row 按 index load
  后 widen 到 fp32 直接 FMLA。
- PV 阶段是 `P_hat(fp32) * V`，天然不适合 bf16 BFMMLA。

BFMMLA 的重新评估条件很严格：只有当目标机器上 bf16 BFMMLA 峰值吞吐超过
fp32 FMLA 峰值的 2 倍以上，才考虑为 indexed sparse 额外设计 BFMMLA path。
否则 indexed sparse 只维护 fp32 FMLA 路径，避免复杂度和 pack 开销。

完全共享连续 KV 不受这个限制，仍然沿用 dense path；dense QKT 的 K tile
规整且能被多行 Q 复用，bf16 BFMMLA 在那里是合理的。

### 5.6 完全共享与 indexed path 的边界

完全共享连续 KV 仍然建议走 dense path，而不是 indexed path。原因是 dense
8x8/8x4 可以把同一个 K tile load 一次后服务 8 个 Q row，K 复用最好。
indexed path 是非完全共享的统一兜底，包括部分共享和完全不共享。

### 5.7 后续扩展

- `Lq_eff < 8` 的尾部共享 run
- 两个大 run，例如 compressed prefix + sliding window
- `qkt_indexed_4x8` / `pv_indexed_4x8_ev8` 或 `qkt_indexed_8x4`
  专用形状
- 对部分共享的 K index 做 block 内临时去重，只优化 load，不改变 softmax
  位置计数
- 对很长共享 run，沿 KV 维分 `Sc_l2` tile，复用 dense SDPA 的 cache-aware
  tile size

---

## 6. 内核结构

### 6.1 新增文件

建议新增：

```text
csrc/sparse_mla.cpp
csrc/sparse_mla_common.h
```

`sparse_mla.cpp` 负责：

- PyTorch tensor 校验
- dtype dispatch
- OpenMP 并行
- 调用 segment update helper
- 返回 output/max_logits/lse

`sparse_mla_common.h` 负责：

- `SparseRun`
- `OnlineRowState`
- dense span update helper
- indexed/gathered QKT update helper
- indexed/gathered PV update helper
- scalar fallback helper

### 6.2 复用现有 microkernel

复用 `fused_cpp::sdpa_microkernels` 下的 trait：

```cpp
MK::qkt_8x8(...)
MK::qkt_8x4(...)
MK::qkt_tail(...)
MK::pv_8x8(...)
MK::pv_tail(...)
```

第一版默认绑定 `MK_Baseline`，后续可以增加 sparse MLA 专用版本名：

```text
sparse_mla_neon_baseline
sparse_mla_neon_pquad
sparse_mla_neon_qk_ublock4
```

### 6.3 状态结构

```cpp
struct OnlineRowState {
  float m;
  float l;
  float* o_acc;        // length d_v
  float max_logits;    // real KV only
  float lse;           // real KV only
  bool has_real_kv;
};
```

`max_logits/lse` 可以与 output accumulator 分开维护：

- output accumulator 在 sink 前后都更新
- `max_logits/lse` 只在真实 KV segment 上更新

### 6.4 dense span update

输入：

```cpp
q_ptr:  Lq_eff x d_qk
kv_ptr: dense contiguous span [start, start + Sk)
d_v
states[Lq_eff]
```

流程：

1. 用 `qkt_*` 生成 `scores[Lq_eff, Sk_tile]`
2. 对每行更新 `max_logits/lse`
3. 用 online softmax 更新 `m/l/O_acc`
4. 用 `pv_*` 累加 `P_hat * V`

这里 `K` 和 `V` 都来自同一个 `kv` tensor：

- K row base: `kv + s * d_qk`
- V row base: `kv + s * d_qk`
- PV 的 `v_row_stride = d_qk`
- PV 的有效列数是 `d_v`

### 6.5 indexed/gathered block update

输入仍是当前 8-query block：

```cpp
q_ptr:  Lq_eff x d_qk
idx:    4 x Kt
valid:  4 x Kt
states[Lq_eff]
```

流程：

1. 对 8 行各自的剩余 indices 过滤负数。
2. 把当前 8-query block 拆成 rows 0..3 和 rows 4..7 两个 4-row 子块。
3. 对每个子块按 query row 填充 `idx[i, j]` 和 `valid[i, j]`，不要求行间
   index 相同。
4. 调用 `qkt_indexed_4x4` 或 `qkt_indexed_4xN` 生成 `scores[4, Kt]`。
5. 对每行 valid scores 更新 `max_logits/lse` 和 online softmax state。
6. 调用 `pv_indexed_4x4_ev8` 或 `pv_indexed_4xN_ev8`，按同一份 `idx/valid` 累加
   `O_acc`。

建议先实现两个形状：

- `indexed_4x4`: 主要处理非完全共享 KV。
- `indexed_4xN_tail`: 处理 `N = 1..3` 或 block 尾部。

单行 NEON dot/fma 只作为 fallback，不作为主策略。这样能保持“每次处理 8 个
query 的 attention”的整体结构。

---

## 7. Plan 与调度

第一版避免在 kernel 热路径里动态判断 dense/indexed，也避免执行时频繁更换
kernel。采用两阶段：

```text
plan 阶段:
  只看 indices，为每个 8-query block 生成 dense_segments[] 和 indexed_tiles[]

execute 阶段:
  对每个 head 复用同一份 block plan
  先跑 dense_segments，再跑 indexed_tiles
```

`indices` 形状是 `[s_q, 1, topk]`，不随 `h_q` 变化，因此 plan 可以按
8-token block 建一次，然后所有 query heads 复用。这样分类控制开销会被
`h_q * E` 的计算摊薄。

### 7.1 Plan 数据结构

```cpp
struct DenseSeg {
  int32_t start;  // KV start
  int32_t len;    // dense length, rounded down to multiple of 8
};

struct IndexedTile {
  int32_t idx[8][4];   // Kt fixed to 4 in the first version
  uint32_t valid_mask; // bit(row * 4 + col), 32 bits total
};

struct BlockPlan {
  int32_t dense_begin;
  int32_t dense_count;
  int32_t indexed_begin;
  int32_t indexed_count;
  int8_t lq_eff;       // tail block can be < 8
};
```

`IndexedTile` 执行时拆成两个 4-row 子块：

```text
idx[0..3][0..3] -> qkt_indexed_4x4 / pv_indexed_4x4_ev8
idx[4..7][0..3] -> qkt_indexed_4x4 / pv_indexed_4x4_ev8
```

tail 不换 kernel，直接通过 `valid_mask` padding。

### 7.2 Plan 生成规则

对每个 8-query block：

1. 过滤每行 `indices < 0`，保留原始顺序和重复 index。
2. 找 8 行共同拥有的长连续 KV span。
3. 如果共同 span 长度 `>= dense_threshold`，生成 `DenseSeg`。
4. `DenseSeg.len` 向下取 8 的倍数；共同 span 的尾巴进入 indexed tile。
5. 被 dense 消费的位置做标记。
6. 剩余所有位置按原 row 顺序打包成固定 `IndexedTile[idx[8][4]]`。

推荐：

```text
dense_threshold = 16 或 32
```

也就是说 dense 只抓“大块且确定收益高”的完全共享连续 KV。短共享、部分共享、
非共享、非连续和重复 index 都不再细分，统一进入 indexed tile。

### 7.3 Execute 调度

计算顺序以 query 为主。外层按 `token_block` 从小到大推进；每个
`token_block` 固定包含最多 8 个连续 query token：

```text
token_block 0: query [0, 8)
token_block 1: query [8, 16)
token_block 2: query [16, 24)
...
```

推荐并行维度仍然是 `(token_block, head)`，但每个 block 内部的执行顺序固定：

```cpp
plans = build_sparse_mla_plans(indices)

for block in blocks:
  plan = plans[block]
  for head in range(h_q):
    init 8 row states
    for seg in dense_segments[plan]:
      run_dense_8x8(seg)
    for tile in indexed_tiles[plan]:
      run_indexed_4x4(tile.rows0_3)
      run_indexed_4x4(tile.rows4_7)
    apply attn_sink if any
    normalize and write output[block.query_range, head]
```

这里选择 `dense first, indexed second`，执行阶段只有两个固定 loop，避免
`dense -> indexed -> dense -> indexed` 的频繁切换。数学上 softmax 对 KV 顺序
不敏感；重复 index 只要在 plan 里保留计数即可。fp32 累加顺序可能与 naive
逐 index 顺序产生很小差异，测试阈值应按 fp32 online softmax 路径设置。

如果后续开启 OpenMP，物理执行上不同 `(token_block, head)` 可以并行调度；但
每个 block 的输入、状态和输出都只覆盖自己的 query 顺序区间，不跨 block 重排
query，也不把不同 block 的 KV 工作合并。

由于 `kv` 是共享 single KV head，多个 query head 会读同一份 K/V。多线程下
不做 K/V 私有大拷贝，只分配线程私有小 scratch：

```text
scores:  8 * Sk_tile fp32
P_hat:   8 * Sk_tile fp32
O_acc:   8 * d_v fp32
idx:     4 * Sk_tile int32/int64
valid:   4 * Sk_tile bool/bitmask
```

`Kt` 初始建议为 4，必要时扩展到 8，不要一次处理完所有 topk。原因：

- indexed K load 的复用通常低于 dense path
- 小 tile 更适合非完全共享 KV
- indexed 4x4 kernel 的寄存器压力低于 8x4，更适合作为第一版 residual kernel

---

## 8. Plan 细节和阈值

第一版 plan 规则：

1. 对每个 token 的有效 indices 切 run。
2. 在当前 8-query block 内找 8 行完全共享的连续 run，不跨 block 合并 query。
3. 只有 run 长度 `>= dense_threshold` 时才生成 `DenseSeg`。
4. `DenseSeg.len = floor(run_len / 8) * 8`，只保留 dense 8x8 主体。
5. 共同 run 中不足 8 的尾部，以及所有非完全共享位置，都填入 indexed tile：
   - `Kt = 4`: 对 rows 0..3 / rows 4..7 分别调用
     `qkt_indexed_4x4` / `pv_indexed_4x4_ev8`。
   - `Kt < 4`: 调用 `qkt_indexed_4xN` / `pv_indexed_4xN_ev8` tail。
   - `Kt = 8`: 第一版可拆成两个 Kt=4 tile；后续再评估 indexed 4x8
     或 8x4。
6. 部分共享不需要单独 kernel；它自然表现为多行 `idx[i, j]` 相同。

推荐第一版不要为了短 run 切 dense：

```text
dense_threshold = 16 或 32
```

这样可以避免短片段导致的 kernel 切换和 plan 碎片化。共同 run 长度是 20 时：

```text
DenseSeg(start, 16)
剩余 4 个 KV -> indexed tile
```

后续优化规则：

1. 如果 8 行共享相同 `start` 但 `length` 不同，可以取公共 prefix 走
   dense path，剩余部分继续作为 indexed segment。
2. 如果 run 数通常 <= 4，可按 run cursor 合并 compressed prefix 和 SWA window。
3. 对很长共享 run，沿 KV 维分 `Sc_l2` tile，复用 dense SDPA 的 cache-aware
   tile size。
4. 对 indexed tile 中重复出现的 K index 做临时去重，减少 K load 次数，但
   softmax 位置仍按原始 `idx[4, Kt]` 计数。

---

## 9. 正确性边界

必须覆盖以下行为：

1. `indices < 0` 跳过。
2. 全部 invalid 时 output 为 0，`max_logits=-inf`，`lse=+inf`。
3. 正数越界报错，不静默 mask。
4. 重复 index 保留重复 softmax 位置。
5. `topk_length` 不改变结果。
6. `attn_sink` 不进入 `max_logits/lse`，但进入 output denominator。
7. `d_v < d_qk` 时，QK 用完整 `d_qk`，PV 只用前 `d_v`。
8. `out` buffer 复用时，返回的 output 必须是同一个 tensor。
9. bf16 输入输出为 bf16，内部 fp32 状态。
10. fp32 输入输出为 fp32。
11. Plan 允许执行顺序为 dense first、indexed second；数学 softmax 对 KV 顺序
    不敏感，但 fp32 online 累加顺序会和 naive 逐 index 顺序略有差异，测试
    应使用合理容差。

---

## 10. 实现阶段

### Phase 0: 参考语义固化

- 给当前 `tests/test_sparse_mla.py` 增加更细的 case：
  - 重复 index
  - dense contiguous run
  - 多个 run
  - `attn_sink` 与全 invalid 混合
  - `d_v < d_qk`

### Phase 1: Plan builder + C++ scalar executor

- 新增 `csrc/sparse_mla.cpp`
- 实现 `build_sparse_mla_plans(indices, dense_threshold)`：
  - 每个 8-query block 生成 `BlockPlan`
  - dense 只生成 `DenseSeg(start, len_multiple_of_8)`
  - 其他位置全部生成 `IndexedTile(idx[8][4], valid_mask)`
- 按 plan 顺序实现 C++ scalar executor：
  - dense loop
  - indexed loop
  - sink / normalize / writeback
- Python wrapper 优先调用 `_C.flash_mla_sparse_fwd`
- 与 naive 全量对齐

验收：

```bash
pytest tests/test_sparse_mla.py
```

额外验证：

- plan 中 dense/indexed 消费的有效 KV 计数与原始 indices 一致
- 重复 index 在 indexed tile 中保留重复位置
- `dense_threshold` 改变只影响 plan 分类，不改变输出语义

### Phase 2: 8 行共享 dense-run path

- 执行 Phase 1 plan 中的 `DenseSeg`
- 对完全共享连续 run 调用 `qkt_8x8/pv_8x8`
- `DenseSeg.len` 已经向下取 8 的倍数，第一版不在 dense path 里处理短尾巴
- 与 scalar residual path 共用 online state

验收：

- 构造 `indices` 全部是连续 dense run，应明显快于 scalar baseline
- 与 naive 对齐

### Phase 3: indexed 4x4 kernel

- 实现 `qkt_indexed_4x4` 和 `pv_indexed_4x4_ev8`
- 8-query block 的非完全共享部分拆成 rows 0..3 / rows 4..7 两个子块
- Q 按 4 行正常 load；K 按 `idx[4, 4]` 对每个 query row 分别 load
- 支持 `valid[4, 4]`，invalid 位置不更新 softmax/PV
- indexed QKT/PV 使用 fp32 FMLA；bf16 输入先 widen 到 fp32，不做 BFMMLA
- 同时覆盖部分共享和完全不共享场景

验收：

- 构造部分 query 共享 KV 的 case，命中 indexed 4x4
- 构造每行 KV 都不同的 case，仍命中 indexed 4x4
- 与 naive 对齐

### Phase 4: indexed tail / 更大形状扩展

- 实现 `qkt_indexed_4xN` / `pv_indexed_4xN_ev8` tail，覆盖 `N = 1..3`
- `N = 8` 第一版可拆成两个 indexed 4x4
- 后续如有必要再评估专用 indexed 4x8 或 8x4
- indexed 更大形状仍默认 fp32 FMLA；只有在目标机器 bf16 BFMMLA 峰值超过
  fp32 FMLA 2 倍以上时，才开 BFMMLA 可行性评估

验收：

- fully sparse topk 场景快于 scalar baseline
- dense-run 场景不回退

### Phase 5: 策略优化与注册多版本

- 增加 sparse MLA version registry 或简单 env 开关：
  - scalar
  - hybrid_shared_8x8
  - hybrid_indexed_4x4
  - hybrid_indexed_4xN
- benchmark 输出 shared fraction、indexed fraction、indexed K reuse ratio

---

## 11. Benchmark 设计

建议新增 `tests/bench_sparse_mla.py`，覆盖：

1. dense-only:
   - 每行 indices 都是 `[0, 1, ..., topk-1]`
   - 预期 dense-run path 占比 100%
2. two-run:
   - compressed prefix + sliding window
   - 例如 `[0..127] + [900..1023]`
3. random sparse:
   - topk random，无连续 run
   - 测 indexed path
4. mixed:
   - 50% shared dense run + 50% random sparse
5. sink:
   - 带 `attn_sink`
6. head sweep:
   - `h_q = 1, 8, 16, 32`
7. dtype:
   - bf16
   - fp32

报告字段：

```text
s_q, h_q, s_kv, d_qk, d_v, topk
plan_build_us
dense_segments
indexed_tiles
dense_run_tokens
dense_run_kv_elems
indexed_kv_elems
indexed_k_reuse_ratio
elapsed_ms
effective_flops
speedup_vs_naive
```

---

## 12. 主要风险

1. **dense public API 不可直接复用**
   现有 `scaled_dot_product_attention_versioned` 只返回最终 output，不暴露
   online softmax 中间状态。因此必须复用 microkernel 和内部状态机，而不是
   直接调用 dense attention API。

2. **8 行完全共享 run 可能比例不稳定**
   如果真实 indices 在 token 间差异很大，dense-run path 命中低。因此非完全
   共享路径必须由 indexed 4x4/4xN 兜住，而不是只依赖完全共享。

3. **indexed K load 复用低于 dense path**
   dense 8x8 中一个 K tile load 后可服务 8 个 Q row；indexed kernel 中 K
   往往只服务一个 Q row。第一版应优先实现 indexed 4x4，控制寄存器压力和
   load 浪费；后续再考虑对重复 K index 做 block 内临时去重。

4. **indexed sparse BFMMLA 复杂度不值得默认承担**
   BFMMLA 需要规整 2x2/tile layout，indexed sparse 的 K row 分散会引入
   gather/pack/permute。除非目标机器 bf16 BFMMLA 峰值超过 fp32 FMLA 峰值
   2 倍以上，否则 sparse indexed kernel 只走 fp32 FMLA。

5. **重复 index 与排序不可随意改**
   为了保持 vLLM fallback 语义，不能为了合并 run 而排序或去重 indices。

6. **attn_sink 返回值语义容易写错**
   output denominator 要包含 sink；`max_logits/lse` 不能包含 sink。

---

## 13. 推荐第一版实现形态

最小可落地版本：

1. `C++ scalar baseline`
2. `8 行完全共享 dense run`
3. `indexed 4x4 fp32 FMLA kernel`
4. `indexed 4xN fp32 FMLA / scalar tail fallback`

暂缓：

1. 多 MK trait 注册
2. cache-aware long dense run 的 L3KV path B 复用
3. indexed 4x8 或 8x4 专用 kernel
4. indexed tile 内重复 K index 的 load 去重
5. indexed sparse BFMMLA path，除非目标峰值满足 >2x 条件

这样第一版能快速验证四件事：

- C++ sparse MLA 能完全对齐 naive 语义
- dense-run 命中时能吃到现有 dense NEON microkernel
- 非完全共享 KV 能通过 indexed 4x4 保持 8-query block 计算
- 没有共享 KV 时仍有 indexed sparse fallback
