# CPU MoE Schedule Optimization

本目录用于研究和实现 **CPU 上 MoE 推理的 cost-aware runtime scheduler**。

目标不是先改 kernel，而是先把 MoE expert 执行抽象成一个可测、可搜索、可复现的调度问题：给定路由后的 expert token 负载、CPU 可用核心数和已有的 `T(length, threads)` 运行时间模型，生成一个使 **规划时间 + 执行时间** 最小的 plan。

当前阶段仅建立研究文档和工程边界。

---

## 背景

当前关注的典型规模：

- `num_experts = 256`
- `top_k = 6`
- `tokens = 2048`
- 总路由次数：`tokens * top_k = 12288`
- 可用 CPU cores：`C`

注意：`top_k=6` 不代表 active experts 只有 6 个。对 2048 tokens 而言，active experts 最多可以达到 256 个，实际数量取决于 router 分布。因此 scheduler 必须支持：

1. 少量热点 expert；
2. 大量轻载 expert；
3. hot/cold 混合；
4. active experts 大于 cores 的情况。

---

## 核心目标

对一次 MoE layer 执行，最小化：

```text
T_total(plan) = T_plan(plan) + T_dispatch + T_execute(plan) + T_combine
```

其中：

- `T_plan`：生成调度 plan 的时间；
- `T_dispatch`：根据 `topk_ids` 聚合、排序、分桶、搬运 token 的时间；
- `T_execute`：expert FFN 实际计算时间；
- `T_combine`：将 top-k expert 输出按权重合并回 token 输出的时间。

第一阶段的主要研究对象是：

```text
T_plan(plan) + T_execute(plan)
```

`dispatch/combine` 先作为外部固定成本测量，后续再纳入调度决策。

---

## 设计原则

1. **规划成本是一等公民**  
   不能只最小化 expert compute latency。一个执行时间更优但规划成本很高的 plan，可能整体更差。

2. **严格最优必须限定 plan space**  
   绝对意义上的“所有 CPU 调度策略全局最优”不可操作。本文档中的严格最优定义为：在一个明确的、有限的、可枚举或可建模的 plan space 内求精确最优。

3. **多 plan，而不是单一 heuristic**  
   runtime 可以同时评估多类 plan：uniform、load-proportional、greedy、heavy/light hybrid、exact DP/MILP、fallback global pool 等，再把规划成本也计入目标函数。

4. **先做可测模型，再做调度策略**  
   所有调度策略必须基于本机 profiling 得到的 `T(length, threads)` 或其离散表，不假设线性 scaling。

5. **先独立研究，再接入 vLLM/fused_cpp**  
   本目录先作为 research sandbox。确认 cost model、solver 和 benchmark 后，再决定是否接入 `fused_cpp.moe` 或 vLLM 的 CPU MoE path。

---

## 当前文档

- [`DESIGN.md`](./DESIGN.md)：完整设计文档，定义 plan space、成本模型、严格最优求解方式、multi-plan runtime selector 和阶段性路线图。
- [`README.md`](./README.md)：当前入口说明。
- [`cost_model/profile_schema.md`](./cost_model/profile_schema.md)：`T_expert(routes, threads)` profiling table 的第一版 schema。
- [`cost_model/profile_expert_cost.py`](./cost_model/profile_expert_cost.py)：用真实 `fused_moe_bf16_tiled_scheduled` kernel 生成 `T_expert(routes, threads)` cost table。
- [`planners/plan_schema.md`](./planners/plan_schema.md)：`Plan / Wave / Team` 的第一版输出 schema。
- [`planners/offline_simulator.py`](./planners/offline_simulator.py)：纯 Python offline simulator，用 synthetic 或 JSON cost table 比较基础 planner。
- [`benchmarks/synthetic_sweep.py`](./benchmarks/synthetic_sweep.py)：系统扫描 synthetic routing distributions，比较不同 planner 的 planning-aware / execution-only 结果。
- [`benchmarks/selector_stress.py`](./benchmarks/selector_stress.py)：参数网格压力测试，用于发现 `AUTO` selector 的 regret 边界。
- [`benchmarks/scheduled_bridge_bench.py`](./benchmarks/scheduled_bridge_bench.py)：把 offline simulator 生成的 plan 转成 `fused_moe_bf16_tiled_scheduled` 的 bridge tensors，并测真实 C++ kernel 耗时。

## 当前可运行闭环

可以先用 synthetic workload 和 synthetic cost model 跑通 planner selection：

```bash
python cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution zipf \
  --cores 16
```

默认打分使用 complexity-based planner cost model，而不是 Python 原型本身的 wall time。它按 `A`、`C`、cost-table lookup 数、scan 数、sort 规模和 wave packing 操作数估算 `T_plan`。

如需使用固定成本模型做敏感性分析，可以切换到：

```bash
python cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution zipf \
  --cores 16 \
  --plan-cost-source model
```

固定模型默认值为：

```text
fixed   = 1 us
uniform = 5 us
greedy  = 20 us
groups  = 30 us
```

可以用 `--planner-cost` 覆盖：

```bash
python cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution zipf \
  --cores 16 \
  --plan-cost-source model \
  --planner-cost fixed=1us \
  --planner-cost uniform=5us \
  --planner-cost greedy=20us \
  --planner-cost groups=30us
```

如需查看 Python 原型实际规划开销对总分的影响，可以切换到：

```bash
python cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution zipf \
  --cores 16 \
  --plan-cost-source measured
```

## 校准 Expert Cost Model

planner 的执行时间模型是：

```text
T_expert(routes, threads) -> ns
```

当前 C++ scheduled bridge 会在 team 内根据 GEMM 形状自动选择 M-split 或 N-split，因此修改 kernel 后需要重新 profile 这张表。可以用单 active expert 的真实 kernel 调用生成 JSON：

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/profile_expert_cost.py \
  --output cpu_moe_schedule_optimization/cost_model/profiles/local_dsv4_8c.json \
  --hidden-size 4096 \
  --ffn-hidden-size 512 \
  --route-buckets 1,2,4,8,16,32,64,128,256,512,1024,2048 \
  --thread-buckets 1,2,3,4,5,6,7,8 \
  --warmup 5 \
  --runs 30
```

生成后可以让 offline simulator 使用这张表：

```bash
python cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution zipf \
  --cores 8 \
  --cost-table cpu_moe_schedule_optimization/cost_model/profiles/local_dsv4_8c.json
```

也可以直接把它接到真实 scheduled bridge benchmark：

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/scheduled_bridge_bench.py \
  --distribution active_subset \
  --active-experts 6 \
  --tokens 2100 \
  --top-k 6 \
  --cores 8 \
  --planner groups \
  --cost-table cpu_moe_schedule_optimization/cost_model/profiles/local_dsv4_8c.json
```

不接真实 routing dump 时，可以先跑 synthetic sweep：

```bash
python -B cpu_moe_schedule_optimization/benchmarks/synthetic_sweep.py \
  --case-set smoke \
  --cores 16
```

`--cores` 默认是 16。smoke sweep 里也包含显式的 8-core DSV4-like 场景：

```text
dsv4_sparse_topk_8c
dsv4_broad_heavytail_8c
```

输出会比较：

- `FIXED_GLOBAL_THREADS`
- `UNIFORM_WAVES`
- `GREEDY_MARGINAL_GAIN`
- `ENUMERATE_CORE_GROUPS`
- `AUTO` selector 的 gated planner subset

并按：

```text
score = T_plan + T_execute
```

选择当前估计最优 plan。

`AUTO` 不是新 solver，而是 cheap gating：

```text
routes stats -> choose planner subset -> score candidates -> select best
```

可以直接运行：

```bash
python -B cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution zipf \
  --cores 16 \
  --planner auto
```

### Core group enumeration

`ENUMERATE_CORE_GROUPS` 会枚举核心分组形状。以 8 核为例，候选包括：

```text
[8]
[4, 4]
[4, 2, 1, 1]
[3, 3, 2]
[2, 2, 2, 2]
[1, 1, 1, 1, 1, 1, 1, 1]
...
```

每个分组形状定义一个 wave 内可并行执行的 expert team slot。当前实现先按负载从大到小把 experts 填入这些 slot，然后选估计执行时间最低的分组形状。它已经能输出 C++ bridge 需要的紧凑数组：

为了避免 32 核以上的整数 partition 数量过大，当前最多保留 512 个代表性分组形状；8 核和 16 核会完整枚举。

```json
{
  "num_threads": 8,
  "wave_offsets": [0, 6],
  "team_expert_ids": [0, 1, 2, 3, 4, 5],
  "team_threads": [3, 1, 1, 1, 1, 1]
}
```

这些字段可以直接转成 `torch.int32` tensor 传给 `fused_moe_bf16_tiled_scheduled`。

### Real scheduled kernel benchmark

构建 C++ extension 后，可以直接测 simulator plan 在真实 BF16 tiled MoE kernel 上的耗时：

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/scheduled_bridge_bench.py \
  --distribution active_subset \
  --active-experts 6 \
  --tokens 2100 \
  --top-k 6 \
  --cores 8 \
  --planner groups \
  --warmup 1 \
  --runs 3
```

也可以同时比较全部 planner 和现有默认 kernel：

```bash
PYTHONPATH=src .venv/bin/python \
  cpu_moe_schedule_optimization/benchmarks/scheduled_bridge_bench.py \
  --distribution lognormal \
  --lognormal-sigma 2.0 \
  --tokens 2048 \
  --top-k 6 \
  --cores 8 \
  --planner all \
  --include-default
```

注意：当前 C++ bridge 控制的是逻辑线程分组，还没有传入物理 `core_ids` 做显式绑核。

### Synthetic routing distributions

这些分布都直接生成 `routes[e]` histogram，而不是模拟 router logits。它们用于压测 scheduler 对不同负载形状的反应：

| `--distribution` | 用途 |
| --- | --- |
| `uniform` | 所有 experts 完全均衡，baseline sanity case。 |
| `random_balanced` | 每条 route 均匀随机选择 expert，接近均衡但有采样噪声。 |
| `active_subset` | 只有一部分 experts 激活，用于测试 `A << E` 和 `A <= C`。 |
| `hotspot` | 少数 hot experts 吃掉固定比例 routes，其余 experts 均分剩余 routes。 |
| `heavy_light` | 少数 heavy experts 按 rank 衰减分配大部分 routes，长尾 experts 分剩余 routes。 |
| `zipf` | power-law / long-tail expert popularity。 |
| `dirichlet` | 随机非均衡分布，`--dirichlet-alpha < 1` 会产生稀疏热点。 |
| `lognormal` | 随机 heavy-tail 分布，`--lognormal-sigma` 越大越不均衡。 |
| `domain_cluster` | routes 集中在一段 expert cluster 内，模拟 domain/task-specific expert activation。 |
| `one_hot` | 单 expert 极端热点，stress test。 |

不均衡场景示例：

```bash
python cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution hotspot \
  --hot-experts 4 \
  --hot-fraction 0.75 \
  --cores 16

python cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution dirichlet \
  --dirichlet-alpha 0.15 \
  --seed 1 \
  --cores 16

python cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution domain_cluster \
  --cluster-experts 32 \
  --background-fraction 0.10 \
  --cores 16
```

---

## 后续建议目录结构

当前只需要文档。后续如果进入实现阶段，建议扩展为：

```text
cpu_moe_schedule_optimization/
  README.md
  DESIGN.md
  cost_model/
    profile_schema.md
  planners/
    plan_schema.md
    offline_simulator.py
  benchmarks/
    README.md
    synthetic_sweep.py
    scheduled_bridge_bench.py
  data/
    .gitkeep
```

建议暂时不要急着创建代码目录，避免在问题定义稳定之前把实现耦合得太早。

---

## 第一批研究任务

1. 从真实 `topk_ids` dump 中统计 active experts、routes 分布、top experts、entropy、Gini coefficient。
2. 建立 `T_expert(routes, threads)` profiling table。
3. 测量不同 planner 的 `T_plan`：常数 plan、比例 plan、greedy、DP、MILP。
4. 枚举核心分组形状，并把 `Plan -> scheduled_bridge -> C++ kernel` 的实测闭环跑通。
5. 比较：
   - best execution-only plan；
   - best planning-aware plan；
   - multi-plan selector；
   - current fixed-thread baseline。

---

## 非目标

当前阶段不做：

- 改 SDPA kernel；
- 改 MoE microkernel；
- 改权重 packing 格式；
- 做跨 layer 全局调度；
- 做分布式 expert parallel；
- 引入机器学习策略模型。

这些可以作为后续阶段，但不是本目录第一阶段的目标。

---

## 成功标准

第一阶段成功标准：

1. 能用一个统一模型表示所有候选 plan；
2. 能解释“严格最优”是在什么 plan space 内成立；
3. 能把 `T_plan` 加进目标函数；
4. 能给出针对 `E=256, top_k=6, tokens=2048, C cores` 的可执行研究路线；
5. 能形成后续实现的 benchmark/spec，而不是只停留在 heuristic 描述。
