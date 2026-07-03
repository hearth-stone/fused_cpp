> **⚠ DEPRECATED — 本文档中的 wave 调度内容后续不考虑，仅作历史参考。** 见 [DEPRECATED_WAVE.md](DEPRECATED_WAVE.md)。async interval-DAG + cost model 保留并继续。

# CPU MoE Schedule Optimization Design

## 1. Problem Statement

本研究面向 CPU 上的 MoE inference runtime scheduling。

给定：

- CPU 可用核心数：`C`
- 总专家数：`E = 256`
- 每个 token 的 `top_k = 6`
- 输入 tokens：`N = 2048`
- 总 expert routes：`R_total = N * top_k = 12288`
- 已知或可通过 profiling 得到的运行时间表：

```text
T_expert(length, threads)
```

其中 `length` 表示某个 expert 收到的 route/token 数，`threads` 表示执行该 expert 时分配的线程数。

目标：生成一个 runtime plan，使得：

```text
T_total = T_plan + T_dispatch + T_execute + T_combine
```

最小。

第一阶段聚焦：

```text
minimize T_plan + T_execute
```

`T_dispatch` 和 `T_combine` 先作为可测固定项记录，不直接参与 plan search。后续如果发现 dispatch/combine 与 plan 强耦合，再纳入模型。

---

## 2. Important Correction: top-k Does Not Equal Active Experts

`top_k = 6` 仅表示每个 token 选择 6 个 experts。对 `N = 2048` 来说，总 route 数为：

```text
2048 * 6 = 12288
```

active expert 数为：

```text
A = |{e | routes[e] > 0}|
```

它满足：

```text
1 <= A <= min(E, N * top_k) = 256
```

因此 scheduler 不能假设 active experts 只有 6 个。真实工作负载可能是：

1. `A << C`：少数热点 expert，可考虑给热点 expert 多线程；
2. `A ~= C`：每个 active expert 单线程或少量线程；
3. `A > C`：必须分 wave、queue 或 work stealing；
4. heavy-tail：少数大 expert + 大量小 expert，这是最常见也最有价值的优化场景。

---

## 3. Runtime Pipeline Model

一次 CPU MoE layer 执行可以拆成：

```text
hidden_states
  -> router/topk
  -> route histogram + dispatch
  -> expert execution
  -> weighted combine
  -> output
```

其中调度器作用于两处：

1. dispatch 之后：根据 `routes[e]` 生成 expert workload；
2. expert execution 之前：生成线程分配和执行拓扑。

当前代码路径中，CPU MoE 已有几类入口：

- 默认 CPU grouped GEMM / torch fallback；
- `fused_cpp` BF16 tiled fused MoE backend；
- routing histogram dump 机制，可用于采集真实 `topk_ids` 分布。

本研究目录暂时不修改这些路径，只定义可被后续接入的 scheduler 模型。

---

## 4. Workload Representation

### 4.1 Raw routing input

runtime 原始输入：

```text
topk_ids: [N, top_k]
topk_weights: [N, top_k]
```

其中：

```text
N = 2048
top_k = 6
E = 256
```

### 4.2 Expert route count

将 `topk_ids` 压缩成 expert route histogram：

```text
routes[e] = count(topk_ids == e), e in [0, E)
```

满足：

```text
sum(routes) = N * top_k
```

### 4.3 Active expert list

```text
active = [(expert_id, routes[e]) for e if routes[e] > 0]
A = len(active)
```

后续 plan 只对 active experts 建模。

### 4.4 Workload statistics

用于 planner selection 的低成本统计量：

```text
A                    # active experts
routes_max           # 最大 expert route 数
routes_mean_active   # active experts 平均 route 数
routes_std_active    # active experts route 标准差
entropy              # 路由熵
heavy_count          # 超过阈值的热点 expert 数
light_count          # 小 expert 数
```

其中 entropy 可定义为：

```text
p_e = routes[e] / sum(routes)
H = -sum(p_e * log(p_e))
```

这些统计量用于判断应选择哪类 plan generator。

---

## 5. Plan Model

一个完整 plan 定义为：

```text
plan = (workload_view, allocation, schedule, metadata)
```

更具体地：

```text
plan = {
  kind: PlanKind,
  active_experts: list[ExpertWork],
  waves: list[Wave],
  estimated_plan_cost_ns: int,
  estimated_exec_cost_ns: int,
  estimated_total_cost_ns: int,
  exactness_scope: string,
}
```

### 5.1 ExpertWork

```text
ExpertWork = {
  expert_id: int,
  routes: int,
  route_bucket: int optional,
}
```

### 5.2 Wave

当 active experts 大于可用 cores 或某些 expert 使用多线程 team 时，执行需要被拆成多个 wave。

```text
Wave = {
  teams: list[Team],
  estimated_wave_time_ns: int,
}
```

### 5.3 Team

一个 team 是一次 expert execution 的资源绑定单元：

```text
Team = {
  expert_id: int,
  threads: int,
  core_mask: optional,
  numa_node: optional,
}
```

第一阶段可以忽略 `core_mask` 和 `numa_node`，只保留：

```text
(expert_id, threads)
```

### 5.4 PlanKind

候选 plan 类型：

```text
enum PlanKind {
  FIXED_GLOBAL_THREADS,
  UNIFORM_WAVES,
  LOAD_PROPORTIONAL,
  SQRT_LOAD,
  GREEDY_MARGINAL_GAIN,
  HEAVY_LIGHT_HYBRID,
  HISTOGRAM_EXACT_DP,
  MILP_EXACT,
  GLOBAL_WORK_STEALING_POOL,
}
```

---

## 6. Cost Model

### 6.1 Expert execution cost

核心 cost table：

```text
T_expert(routes, threads) -> nanoseconds
```

它必须来自本机 profiling 或真实运行反馈，而不是假设线性加速。

离散化建议：

```text
route_buckets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, ...]
thread_buckets = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
```

Apple Silicon / AArch64 上 thread scaling 可能受 cache、memory bandwidth、P-core/E-core、OS scheduling 影响，因此 cost table 应该优先来自实际测量。

### 6.2 Wave execution cost

如果一个 wave 中有多个 team 并行执行，则 wave latency 为：

```text
T_wave = max(T_expert(routes[e], threads[e]) for team in wave)
         + T_wave_overhead
```

### 6.3 Plan execution cost

```text
T_execute(plan) = sum(T_wave for wave in plan.waves)
```

这是一个 barrier-after-wave 模型。它比真实 work stealing 更简单，但适合做第一版 exact solver。

### 6.4 Planning cost

每类 planner 都要测量自身开销：

```text
T_plan(kind, A, C, stats) -> nanoseconds
```

示例：

```text
FIXED_GLOBAL_THREADS      ~ O(1)
LOAD_PROPORTIONAL         ~ O(A)
UNIFORM_WAVES             ~ O(A * C)
GREEDY_MARGINAL_GAIN      ~ O(B * (S * A + A log A + pack(A)))
HISTOGRAM_EXACT_DP        ~ depends on buckets and C
MILP_EXACT                ~ solver-dependent, usually too high for hot path
```

其中 `B` 是枚举的 wave budget 数，`S` 是每个 wave budget 下可追加线程 slot 数。第一版 simulator 使用 complexity-feature model：

```text
T_plan ~= c0
        + c_lookup * cost_table_lookup_ops
        + c_scan   * scan_ops
        + c_sort   * sort_compare_ops
        + c_pack   * wave_pack_ops
```

Big-O 只给增长趋势，真正纳秒级估计需要用 native planner profiling 校准这些常数。

最终目标不是最小化 `T_execute`，而是：

```text
T_plan + T_execute
```

### 6.5 Total model for selection

```text
score(plan) = measured_or_estimated_T_plan(plan.kind)
            + estimated_T_execute(plan)
```

如果 dispatch/combine 与 plan 强相关，则扩展为：

```text
score(plan) = T_plan + T_dispatch(plan) + T_execute(plan) + T_combine(plan)
```

---

## 7. Plan Families

### 7.1 Fixed global-thread plan

当前很多 CPU inference path 接近这种模式：给整个 MoE op 一个固定线程数，内部由 kernel 或 thread pool 自行并行。

优点：

- planning cost 近似 0；
- 实现简单；
- 稳定。

缺点：

- 不利用 expert load imbalance；
- 对 heavy-tail routing 不敏感。

### 7.2 Uniform waves

active experts 分成若干 wave，每个 expert 使用相同线程数：

```text
threads_per_expert = floor(C / experts_per_wave)
```

优点：实现简单，便于作为 baseline。

缺点：不能处理热点 expert。

### 7.3 Load-proportional allocation

按 route 数分配线程：

```text
threads[e] proportional to routes[e]
```

适合近似线性 scaling 的 cost model。

### 7.4 Sqrt-load allocation

按 route 数的平方根分配线程：

```text
threads[e] proportional to sqrt(routes[e])
```

适合收益递减的 CPU scaling。

### 7.5 Greedy marginal-gain plan

每次把一个线程分给边际收益最大的 expert：

```text
gain(e, t) = T_expert(routes[e], t) - T_expert(routes[e], t + 1)
```

该方法对非线性 cost table 更友好，是强 baseline。

### 7.6 Heavy/light hybrid

将 active experts 分为：

```text
heavy = routes[e] >= heavy_threshold
light = routes[e] < heavy_threshold
```

策略：

- heavy experts 使用 multi-thread teams；
- light experts 放入 single-thread 或 small-thread work queue；
- heavy 与 light 可以同 wave 混合，也可以先执行 heavy 再执行 light。

这类 plan 对 `E=256, top_k=6, N=2048` 很重要，因为真实 routing 通常会出现 heavy-tail。

### 7.7 Histogram exact DP

把 experts 按 route bucket 聚合，专家在同 bucket 内视为可交换：

```text
histogram[bucket_id] = number of experts in this route bucket
```

然后在 histogram state space 上做 exact DP。

这不是对原始 256 维 routes 的完全精确，但它是在“bucketized workload abstraction”内严格最优。

### 7.8 MILP exact

在完整 wave-parallel expert-team plan space 内，可以写成 MILP。

变量：

```text
x[e, w, t] = 1 if expert e is executed in wave w with t threads
z[w]       = latency of wave w
```

约束：

```text
# each expert executes exactly once
sum_{w,t} x[e,w,t] = 1

# wave core budget
sum_{e,t} t * x[e,w,t] <= C

# wave latency lower bound
z[w] >= T_expert(routes[e], t) * x[e,w,t]
```

目标：

```text
minimize T_plan(MILP) + sum_w z[w]
```

该方法给出严格最优，但通常只适合 offline analysis、small cases 或生成 teacher labels，不适合每次 runtime 热路径调用。

---

## 8. Strict Optimality Definition

本研究必须明确“严格最优”的范围。

### 8.1 不可操作的定义

下面这个定义不可操作：

```text
在所有可能 CPU 调度、所有 OS 干扰、所有 cache/NUMA 状态下绝对最优。
```

原因：真实 CPU runtime 有噪声，且完整 plan space 包含 thread affinity、task order、work stealing、cache state、NUMA placement 等组合因素。

### 8.2 可操作定义

本文档采用以下定义：

```text
给定：
  1. 固定 workload routes[e]
  2. 固定 CPU core budget C
  3. 固定 cost table T_expert(routes, threads)
  4. 固定有限 plan space P

求：
  plan* = argmin_{plan in P} T_plan(plan.kind) + T_execute(plan)

则 plan* 是 P 内严格最优。
```

### 8.3 Multi-plan strict optimum

如果允许多种 planner/generator：

```text
G = {g_1, g_2, ..., g_m}
```

每个 generator 产生候选 plan 集合：

```text
P_i = g_i(routes, C)
```

则最终 plan space 为：

```text
P_union = union(P_i)
```

严格最优定义为：

```text
plan* = argmin_{plan in P_union} T_plan(generator(plan)) + T_execute(plan)
```

这解决“policy table 退化成单一 plan”的问题：runtime 不绑定一种 plan，而是在多个 plan family 的并集上做 selection。

---

## 9. Recommended Solver Architecture

### 9.1 Runtime planner pipeline

```text
routes/topk_ids
  -> build histogram/stats
  -> choose enabled plan generators
  -> generate candidate plans
  -> estimate score = T_plan + T_execute
  -> select best plan
  -> execute
  -> record actual latency
  -> update profiling/cost model
```

### 9.2 Planner interface

```text
PlannerInput:
  routes: int[E]
  active_experts: list[ExpertWork]
  num_cores: int
  cost_model: T_expert
  planning_budget_ns: optional

PlannerOutput:
  best_plan: Plan
  candidates: list[Plan]
  diagnostics: PlannerDiagnostics
```

### 9.3 Candidate generation policy

初始版本建议启用：

1. `FIXED_GLOBAL_THREADS`
2. `UNIFORM_WAVES`
3. `LOAD_PROPORTIONAL`
4. `SQRT_LOAD`
5. `GREEDY_MARGINAL_GAIN`
6. `HEAVY_LIGHT_HYBRID`
7. `HISTOGRAM_EXACT_DP` only when planning budget allows

`MILP_EXACT` 初期只用于 offline validation。

---

## 10. Handling A > C

当 active experts 数量大于 cores 时，不能给每个 active expert 同时分配至少 1 个线程。必须引入 wave 或 queue。

### 10.1 Wave model

将 experts 划分为多个 wave：

```text
wave_0: experts using <= C threads total
wave_1: experts using <= C threads total
...
```

执行时间：

```text
T_execute = sum(max expert latency in each wave)
```

### 10.2 Queue model

使用一个全局 worker pool，每个 worker 处理 expert chunks。

优点：

- planning cost 低；
- 对大量小 expert 友好。

缺点：

- 更难严格建模；
- 受 work stealing overhead 和 cache locality 影响。

### 10.3 Hybrid model

推荐第一版采用 hybrid：

```text
heavy experts -> wave/team model
light experts -> queue/pool model
```

这样能在保持模型可解的同时覆盖常见 heavy-tail routing。

---

## 11. Profiling Plan

### 11.1 Expert cost table profiling

对不同 route count 和 thread count 测量：

```text
T_expert(routes, threads)
```

建议维度：

```text
routes: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
threads: 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64
```

每个点记录：

```text
median_ns
p10_ns
p90_ns
p99_ns
stddev_ns
num_iters
```

scheduler 默认使用 `median_ns`，延迟敏感场景可使用 `p90_ns` 或 `p99_ns`。

### 11.2 Planning cost profiling

对每类 planner 测：

```text
T_plan(kind, A, C, distribution_shape)
```

distribution_shape 可用：

```text
uniform
one_hot
zipf_like
real_dump
```

### 11.3 Real routing dump

利用现有 CPU MoE routing histogram dump 能力，收集真实模型运行时的：

```text
layer_id
tokens
top_k
num_experts
active_experts
routes_min/routes_max/routes_mean/routes_std
top_experts
```

后续 benchmark 应优先使用真实 dump，而不是只用随机 synthetic distribution。

---

## 12. Evaluation Plan

### 12.1 Baselines

必须比较：

1. current fixed-thread backend；
2. uniform waves；
3. load-proportional；
4. sqrt-load；
5. greedy marginal gain；
6. heavy/light hybrid；
7. exact DP/MILP for small or bucketized cases。

### 12.2 Metrics

```text
T_plan_ns
T_execute_estimated_ns
T_execute_actual_ns
T_total_estimated_ns
T_total_actual_ns
planner_regret_ns = T_total_actual - best_measured_total
speedup_vs_fixed
```

### 12.3 Exactness validation

对 small cases：

```text
E_active <= 16 or <= 32
C <= 16 or <= 32
```

用 MILP 或 exhaustive/DP 生成 true optimum，用来验证 greedy 和 hybrid 的 regret。

对 full cases：

```text
E = 256
N = 2048
top_k = 6
```

使用 histogram exact DP 或 limited MILP 做 reduced-space optimum。

---

## 13. Integration Strategy

### 13.1 Current repository position

本目录位于：

```text
fused_cpp/cpu_moe_schedule_optimization/
```

它应先作为 research sandbox，不立即侵入已有 MoE kernel 或 vLLM Python path。

### 13.2 Future integration points

可能接入点：

1. Python CPU MoE wrapper：在调用 fused_cpp backend 前生成 plan；
2. fused_cpp MoE C++ backend：接收 plan 并按 plan 执行 expert teams/waves；
3. benchmark scripts：复现不同 routing distribution 的调度性能；
4. env flags：控制是否启用 scheduler、planner kind、planning budget。

建议的 env flags：

```text
VLLM_CPU_MOE_SCHEDULER=0/1
VLLM_CPU_MOE_SCHEDULER_KIND=fixed|greedy|hybrid|exact_dp|auto
VLLM_CPU_MOE_SCHEDULER_PLAN_BUDGET_US=...
VLLM_CPU_MOE_SCHEDULER_DUMP_PLANS=0/1
```

### 13.3 Do not mix with SDPA rules

本研究不修改 SDPA kernel。除非未来直接改动 `sdpa_*` 文件，否则不需要更新 SDPA kernel 文档。

---

## 14. Implementation Roadmap

### Phase 0: Documentation only

当前阶段。

输出：

```text
README.md
DESIGN.md
```

### Phase 1: Data and schema

目标：定义 route dump schema、cost table schema、plan schema。

输出建议：

```text
cost_model/profile_schema.md
planners/plan_schema.md
```

### Phase 2: Offline simulator

目标：不接真实 kernel，只用 cost table 模拟 planner。

实现：

```text
routes[e] -> planners -> estimated score -> selected plan
```

### Phase 3: Exact solver validation

目标：对 small cases 建立严格最优基准。

方法：

- exhaustive search；
- DP；
- MILP。

### Phase 4: Runtime benchmark

目标：接入真实 fused_cpp MoE backend 或 mock expert execution，测量 end-to-end `T_plan + T_execute`。

### Phase 5: vLLM integration

目标：通过 env flag 控制是否启用 scheduler，不改变默认路径。

---

## 15. Key Research Questions

1. `T_plan` 在真实 CPU MoE 中占比多大？
2. 对 `E=256, top_k=6, N=2048`，active expert 分布是否稳定？
3. heavy/light hybrid 是否能接近 exact optimum？
4. exact DP 的可用边界在哪里？
5. bucketized exact optimum 与 full-route optimum 的误差有多大？
6. fixed global threads 与 plan-aware scheduling 的真实差距是多少？
7. 调度收益是否足以覆盖 dispatch/reorder 的额外开销？

---

## 16. Initial Recommendation

第一版不要直接追求全局 MILP online optimal。

推荐路线：

```text
1. 固定 workload abstraction: routes[e]
2. 建立 T_expert(routes, threads)
3. 实现 multi-plan selector:
   - fixed
   - proportional
   - sqrt
   - greedy
   - heavy/light
4. 在 small cases 用 exact solver 评估 regret
5. 对 full cases 使用 histogram exact DP 作为 reduced-space optimum
6. 再决定是否接入真实 fused_cpp runtime
```

这样可以同时满足：

- 有严格最优定义；
- 有多种 plan；
- 把 planning cost 纳入目标；
- 不过早陷入 C++ runtime 细节。

---

## 17. External Context

本研究与以下方向相关，但不是简单复刻：

- vLLM CPU backend / MoE kernel selection；
- CPU intra-op / inter-op threading；
- MoE token dispatch/combine；
- classical scheduling, bin packing, makespan minimization；
- NUMA-aware CPU inference runtime。

本目录的差异点是：

```text
CPU MoE + dynamic routing + per-expert thread allocation + planning cost + multi-plan exact/approx selection
```

该组合应作为独立 runtime scheduling 问题处理。
