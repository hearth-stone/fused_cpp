> **⚠ DEPRECATED — 本文档中的 wave 调度内容后续不考虑，仅作历史参考。** 见 [DEPRECATED_WAVE.md](DEPRECATED_WAVE.md)。async interval-DAG + cost model 保留并继续。

# CPU MoE Schedule Optimization

本目录用于研究和实现 **CPU 上 MoE 推理的 cost-aware runtime scheduler**。

目标不是先改 kernel，而是先把 MoE expert 执行抽象成一个可测、可搜索、可复现的调度问题：给定路由后的 expert tasks、CPU 可用核心数、isolated time 和 active-set contention slowdown，生成一个使 **规划时间 + 执行时间** 最小的 plan。

当前阶段仅建立研究文档和工程边界。

## 当前 policy-aware 闭环

Schema v2 路径按完整策略选择表：`TP/EP degree + H/F + global/local
experts + SVE implementation/tail policy + SVE tile + split-W13/packed-B
window + NUMA/CPU set + LLC + source/binary hash`。split/no-split 和每个
global byte-window 不共用 derate，也不跨 F 或拓扑做隐式 nearest-profile
fallback。

`tp_vs_ep_model.py --sve-implementation auto` 先查找完整的
`jit/xbyak_exact_m` split/no-split pair；只有该 pair 不完整时才整体回退到
`asm/static_bucketed` pair。它不会把一个 JIT profile 和一个 asm profile
拼成候选策略。需要可复现实验时可显式指定 `jit` 或 `asm`，此时缺表直接报错。

`weight_window_bytes` 可将 W13 和 W2 都细分为更小的 packed-B N-range。
planner 已联合搜索有独立 schema-v2 profile 的 `(window, core_shape)`，并将
选中值通过 operator options 传入 async kernel；不会用旧 split-W13 profile
对 1/2 MiB 窗口评分。默认 catalog 未加入重新校准的 window 表时，候选仍只有
legacy split/no-split。

当前实现入口：

- `cost_model/profile_contention_async_dual_rank.py`：两个 NUMA-local rank
  同步采样；isolated 使用 8 个连续冷权重，contention 使用全部本地专家，
  mixed-width shape 使用与 planner 相同的 LPT assignment。
- `cost_model/profile_catalog.py`：严格 profile identity、legacy split pair、
  measured window variants 与 grid 校验。
- `cost_model/phase_model.py`：M12 bulk/tail、精确 full-call anchor，以及按
  W13 chunk/W2 瞬时 packed working set 驱动的 stage-aware fallback。
- `planners/interval_planner.py` / `planned_moe.py`：联合搜索
  `(w13_split/window, core_shape)`，返回显式 CPU 集与 operator option，并按
  完整 routing bucket histogram 和 kernel policy identity 缓存。
- `planners/tp_vs_ep_model.py`：当前按 rank-local histogram 独立预测并取全局
  最大 compute，再加上通用分层 all-reduce/all-to-all 模型；尚未建模短 rank
  完成后长 rank 的争用释放。
- `planners/validate_policy_planner.py`：双 rank 枚举实测所有 policy/shape，
  报告预测误差和真实 regret；可用 `--routes-json` 输入真实路由直方图。
- `POLICY_MODEL_VALIDATION.md`：AWS 64-core TP2/EP2 的 2026-07-13 校准配置、
  synthetic/真实路由 regret、EP2 hotspot 诊断和通信模型估计。

完整均匀 workload 直接使用 profile 的 `full_call_*` 曲线，不把短窗口
group cost 乘以 wave 数。非均匀路由才进入 stage-aware DAG 模拟。跨 profile
插值保持禁用。真实 routing summary 已通过 exact-profile regret 验证，但跨
F、并行度和机器拓扑仍需要独立的 out-of-profile 测量。

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
   所有调度策略必须基于本机 profiling 得到的 isolated time 与 contention slowdown，不假设线性 scaling，也不把 LLC/带宽退化写成不可执行的硬约束。

5. **先独立研究，再接入 vLLM/fused_cpp**  
   本目录先作为 research sandbox。确认 cost model、solver 和 benchmark 后，再决定是否接入 `fused_cpp.moe` 或 vLLM 的 CPU MoE path。

---

## 当前文档

- [`MATHEMATICAL_MODEL.md`](./MATHEMATICAL_MODEL.md)：调度数学模型的 source
  of truth；定义未剪枝问题、求解器编码、当前工程剪枝及同步规则。
- [`cost_model/T_ISO_FORMULA_VALIDATION.md`](./cost_model/T_ISO_FORMULA_VALIDATION.md)：
  `T_iso=O(t)+C(R)phi_USL(t)k_phi(t)` 的公式、8 核 SVE/M12 留出验证和原始数据入口。
- [`cost_model/TISO_ROOFLINE_VALIDATION.md`](./cost_model/TISO_ROOFLINE_VALIDATION.md)：
  简化的可解释 `T_iso` roofline baseline；按 M12/tail kernel 计算 FLOP、
  N-split shared-cache A/B/C 流量。
- [`cost_model/GEMM_ECM_VALIDATION.md`](./cost_model/GEMM_ECM_VALIDATION.md)：
  实现弱相关三层 GEMM shadow；分离算法 work、SVE mapper 和实测机器响应，
  并记录 V3 长 route 留出验证及当前可迁移性限制。
- [`cost_model/ANALYTIC_MODEL.md`](./cost_model/ANALYTIC_MODEL.md)：
  planner-compatible 解析 backend；由 exact kernel demand、cache capacity、
  route-independent machine service curves 和共享资源 event simulator 预测
  isolated/contention 时间，直接生成确定性的 W13/W2 stage-window policy，
  并定义薄校准 schema 与 holdout 验收门槛。
- [`../optimizations/fused_moe_sve/results/analytic_stage_window_policy_v2_holdout_20260809.md`](../optimizations/fused_moe_sve/results/analytic_stage_window_policy_v2_holdout_20260809.md)：
  解析 stage-window policy 的 A 驻留/B turnover 修正版在 192C NUMA0 与 8C V1
  上的交错采样 holdout；192C maximum regret 从 11.32% 降到 3.38%，8C 同时记录
  raw 与 p10 单边噪声口径。
- [`../optimizations/fused_moe_sve/results/amazon_192c_analytic_thin_calibration_20260801.md`](../optimizations/fused_moe_sve/results/amazon_192c_analytic_thin_calibration_20260801.md)：
  192-core 主机 NUMA0 的独立 service/retention 薄校准与 holdout；排序显著改善，
  但 contention P90 和最大 regret 未过门槛，因此 production 仍使用经验模型。
- [`DESIGN.md`](./DESIGN.md)：完整设计文档，定义 plan space、成本模型、严格最优求解方式、multi-plan runtime selector 和阶段性路线图。
- [`TODO.md`](./TODO.md)：schema-v2 profiling 之后的 policy-aware cost model、planner 与 TP/EP evaluator 待办。
- [`README.md`](./README.md)：当前入口说明。
- [`cost_model/profile_schema.md`](./cost_model/profile_schema.md)：当前 policy/topology-aware schema v2，以及 legacy v1 格式。
- [`cost_model/profile_expert_cost.py`](./cost_model/profile_expert_cost.py)：用真实 `fused_moe_bf16_tiled_scheduled` kernel 生成 `T_expert(routes, threads)` cost table。
- [`planners/plan_schema.md`](./planners/plan_schema.md)：`Plan / Wave / Team` 的第一版输出 schema。
- [`cost_model/phase_model.py`](./cost_model/phase_model.py)：`ContentionCostModel` —— 争用感知的事件驱动 makespan 模拟内核(`dag_makespan`)，AWS 实测 ~2% median。
- [`cost_model/tiso_roofline.py`](./cost_model/tiso_roofline.py)：从 kernel panel
  逻辑生成 `T_iso` FLOP/流量分解，并输出 steady-M12 的可辨识有效服务率；当前
  仅 shadow validation，不改变 planner 默认打分。
- [`cost_model/gemm_cost_model.py`](./cost_model/gemm_cost_model.py)：实现弱相关
  GEMM 核心契约；分离逻辑算法工作、统一 kernel demand 和目标机器实测响应。
- [`cost_model/sve_bf16_kernel_model.py`](./cost_model/sve_bf16_kernel_model.py)：
  当前 SVE BF16 exact-M1--M12/static bucket 的实现 mapper；负责 tile、padding、N-split、
  指令和 cache 流量，不进入通用算法公式。
- [`cost_model/analytic_model.py`](./cost_model/analytic_model.py) /
  [`cost_model/analytic_stage_window_policy.py`](./cost_model/analytic_stage_window_policy.py) /
  [`cost_model/validate_analytic_model.py`](./cost_model/validate_analytic_model.py)：
  phase-aware hierarchical service cost model、机器校准加载和 empirical holdout 验证；按
  setup/cold-B/steady-B 事件计算 shared-resource offered load，并提供逐事件
  pressure 解释；完成跨机器门槛前保持 opt-in。
- [`cost_model/gemm_ecm.py`](./cost_model/gemm_ecm.py)：旧 API 兼容 facade 和
  stage-trace report CLI；三层 GEMM model 当前不进入 planner active cost。
- [`cost_model/working_set_model.py`](./cost_model/working_set_model.py)：仅针对
  split-W13 的 owner-private cache 工作集 band；由独立 weight-scan 与
  `T_iso` 计算稳健候选，当前只做 shadow validation。
- [`planners/interval_planner.py`](./planners/interval_planner.py)：`IntervalPlanner` —— async interval-DAG 静态 planner（cost-model 驱动）。
- [`planners/ISOLATED_CP_SAT_ORACLE.md`](./planners/ISOLATED_CP_SAT_ORACLE.md)：
  可选 OR-Tools 离线 oracle；在无争用 fixed-duration 模型中搜索全线程宽度组合，
  返回可行上界、理论下界及当前 strict planner 的 isolated regret 区间。
- [`planners/cold_phase_cp_sat_oracle.py`](./planners/cold_phase_cp_sat_oracle.py)：
  在 fixed-duration oracle 上增加 cold packed-B/steady phase 和单 NUMA DRAM
  cumulative constraint；同一 surrogate 内比较固定 lane DAG 与 mixed-width
  schedule，不接入 production planner。
- [`planners/simulate_schedules.py`](./planners/simulate_schedules.py)：离线对比多种调度算法（coop / expert-parallel / greedy / planner / 各 core 切分）的 makespan。
- [`planners/planned_moe.py`](./planners/planned_moe.py)：路由直方图 → 缓存 plan → `fused_moe_bf16_tiled_async` 桥。
- [`planners/bench_planner_overhead.py`](./planners/bench_planner_overhead.py) / [`planners/bench_e2e_scheduling.py`](./planners/bench_e2e_scheduling.py)：plan 开销 / 缓存命中率 + e2e 对比。
- [`FINDINGS.md`](./FINDINGS.md)：10 个 planner 的 C++ 实现（`csrc/moe_planner/`）、对 Python 的逐位等价、隔离 `T_plan` 基准、以及对 best-of-10 和**精确最优**的 regret 评测结论。C++ 入口：`fused_cpp._C.moe_schedule_plan` / `moe_exact_optimum`。

## 当前可运行闭环

用路由直方图离线对比多种调度算法（纯 cost model，本地即可运行，无需 kernel/AWS）：

```bash
P=cpu_moe_schedule_optimization/cost_model/profiles/contention_async_aws_8c_sha46228bb_20260703.json
python cpu_moe_schedule_optimization/planners/simulate_schedules.py $P --preset hotspot
python cpu_moe_schedule_optimization/planners/simulate_schedules.py $P --preset dsv4-real-2048-seq70
python cpu_moe_schedule_optimization/planners/simulate_schedules.py $P --experts 512,512,512,512 --shapes
```

面向调度评测的固定全局 TopK workload 使用 `2048 tokens / topk=6 /
256 experts`，因此每个 workload 都有 12288 routes：

| Preset | Route histogram |
| --- | --- |
| `moe256-uniform` | `256x48`；全 expert 均匀基线，也是 active-set sweep 的 256 endpoint。 |
| `moe256-active-set-{8,16,32,64,128}` | 固定总 routes，分别为 `8x1536`、`16x768`、`32x384`、`64x192`、`128x96`。 |
| `moe256-tiered-hotspot` | `4x768 + 12x384 + 48x96`；分层 hot/warm/cold workload。 |
| `moe256-long-short-bimodal` | `5x2040 + 174x12`；长 route 与 M12 短 route 共存。 |

active-set sweep 可以直接运行：

```bash
for A in 8 16 32 64 128; do
  python cpu_moe_schedule_optimization/planners/simulate_schedules.py \
    "$P" --preset "moe256-active-set-$A"
done
python cpu_moe_schedule_optimization/planners/simulate_schedules.py \
  "$P" --preset moe256-uniform
```

打分使用验证过的 `ContentionCostModel.dag_makespan`（事件驱动 + 争用 derate + overhead-split）。
`dsv4-real-2048-seq70` 固化了 DeepSeek V4 Flash profiler 的
`rank0/seq70/layer27` 路由摘要。捕获文件只保留 top-16 的精确计数，因此
剩余 207 个 active experts 使用确定性的矩匹配长尾，使总量、min/max、mean/std
与捕获摘要一致；长尾 expert ID 仍是合成值。该 preset 用于纯
planner/cost-model 回归，不宣称恢复了原始完整 `topk_ids`。
新增算法只需在 `simulate_schedules.py` 的 `ALGORITHMS` 注册表里加一个 `fn(experts, planner) -> (label, tasks)`。
静态 planner 的选型见 `planners/interval_planner.py`；plan 开销 / 缓存命中率见 `planners/bench_planner_overhead.py`。

## 校准 Expert Cost Model

planner 的执行时间模型是：

```text
T_expert(routes, threads) -> ns
```

### 搜索并发 expert 工作集

`cost_model/search_expert_working_set.py` 用真实 async fused kernel 搜索一台
机器能维持接近峰值吞吐的并发 packed-weight 工作集。每个测量点执行相同数量的
连续不同 expert，只改变同时活跃的 expert lane 数；因此不会把反复读取一个热
权重误当成 LLC 容量。工具支持每 expert 1 线程和把全部核心均分给 active
experts 两种口径，结果写入独立诊断 JSON，不会直接修改 planner profile。

例如在 96 个 NUMA-local 核心上搜索 EP2、split-W13 的工作集：

```bash
taskset -c 0-95 .venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/search_expert_working_set.py \
  --output /tmp/ep2_working_set_search.json \
  --cpu-ids 0-95 --numa-node 0 \
  --parallel-mode ep --parallel-degree 2 \
  --global-experts 64 --num-experts 32 \
  --hidden-size 4096 --ffn-hidden-size 2048 \
  --route-buckets 12,48,192,768,2040 \
  --active-experts auto --allocation both \
  --w13-split 1 --w13-split-chunks 2 \
  --warmup 3 --runs 9 --throughput-tolerance 0.10
```

`auto` 在不超过 64 个 local experts 时逐点扫描，避免跳过容量拐点；更大的
expert 数使用稀疏扫描，也可用 `--active-experts 1-32,40,48,64` 显式指定。
汇总中的 near-peak 范围表示 aggregate TFLOP/s 距该 route 最佳点不超过给定
阈值，工作集定义为
`active_experts * max(W13_chunk_bytes, W2_bytes)`。

split-W13 路径还可以用独立 owner-cache 指标预测优选工作集，而不直接拟合
fused wall time。先编译并运行 `benchmarks/bench_weight_scan.cpp`，再用
`cost_model/working_set_model.py` 将 private-L2 capacity、stream bandwidth
saturation 与 `T_iso` 组合。当前 V3 EP2 验证得到 48--128 MiB 可行 band，
稳健选择为 64 MiB；route 1020/2040 留出 regret 为 0.33%/0.00%。公式、PMU
证据、命令和适用域见
[`cost_model/WORKING_SET_MODEL_VALIDATION.md`](./cost_model/WORKING_SET_MODEL_VALIDATION.md)。
8-core Neoverse-V1/TP4 的跨机器留出根据 6 MiB owner budget 预测 4 MiB，
route 1020/2040 实测 regret 均为 0.00%。

当前 C++ scheduled bridge 会在 team 内根据 GEMM 形状自动选择 M-split 或 N-split，因此修改 kernel 后需要重新 profile 这张表。可以用单 active expert 的真实 kernel 调用生成 JSON：

```bash
OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close \
MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
FUSED_CPP_MOE_PIN_THREADS=1 \
FUSED_CPP_MOE_PIN_THREAD_CPUS=0,1,2,3,4,5,6,7 \
taskset -c 0-7 .venv/bin/python \
  cpu_moe_schedule_optimization/cost_model/profile_expert_cost.py \
  --output cpu_moe_schedule_optimization/cost_model/profiles/local_dsv4_8c.json \
  --hidden-size 4096 \
  --ffn-hidden-size 512 \
  --route-buckets 1,2,4,8,16,32,64,128,256,512,1024,2048 \
  --thread-buckets 1,2,3,4,5,6,7,8 \
  --warmup 5 \
  --runs 30
```

仓库内当前的 AWS 8 核 profile 是（unpinned；本机实测 pinning 无中位数收益且增尾抖动，见 FINDINGS）：

```text
cpu_moe_schedule_optimization/cost_model/profiles/aws_dsv4_8c_packa_sha46228bb_20260703.json
```

生成后可以让 offline simulator 使用这张表：

```bash
python cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution zipf \
  --cores 8 \
  --cost-table cpu_moe_schedule_optimization/cost_model/profiles/aws_dsv4_8c_packa_sha46228bb_20260703.json
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
  --cost-table cpu_moe_schedule_optimization/cost_model/profiles/aws_dsv4_8c_packa_sha46228bb_20260703.json
```

不接真实 routing dump 时，可以先跑 synthetic sweep：

```bash
python -B cpu_moe_schedule_optimization/benchmarks/synthetic_sweep.py \
  --case-set smoke \
  --cores 8 \
  --cost-table cpu_moe_schedule_optimization/cost_model/profiles/aws_dsv4_8c_packa_sha46228bb_20260703.json
```

`--cores` 默认是 16。smoke sweep 里也包含显式的 8-core DSV4-like 场景：

```text
dsv4_sparse_topk_8c
dsv4_broad_heavytail_8c
```

输出会比较：

- `FIXED_GLOBAL_THREADS`
- `SORTED_TOKEN_BALANCED_1T`
- `UNIFORM_WAVES`
- `GREEDY_MARGINAL_GAIN`
- `ENUMERATE_CORE_GROUPS`
- `ASYNC_INTERVAL_DAG`
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
  "thread_cpu_ids": [0, 1, 2, 3, 4, 5, 6, 7],
  "wave_offsets": [0, 6],
  "team_expert_ids": [0, 1, 2, 3, 4, 5],
  "team_threads": [3, 1, 1, 1, 1, 1]
}
```

这些字段可以直接转成 `torch.int32` tensor 传给
`fused_moe_bf16_tiled_scheduled`。`thread_cpu_ids` 表示 logical worker
thread 到 physical CPU id 的映射，长度必须等于 `num_threads`。

### Async interval DAG simulator

如果只想基于已有 `T_expert(routes, threads)` cost table 推演非 wave 调度，不触碰真实
MoE/GEMM kernel，可以直接跑 `ASYNC_INTERVAL_DAG`：

```bash
python -B cpu_moe_schedule_optimization/planners/offline_simulator.py \
  --distribution lognormal \
  --lognormal-sigma 2.0 \
  --tokens 2048 \
  --top-k 6 \
  --cores 8 \
  --planner async \
  --cost-table cpu_moe_schedule_optimization/cost_model/profiles/aws_dsv4_8c_packa_sha46228bb_20260703.json \
  --dump-json
```

这个 planner 做的是纯离线 list scheduling：

```text
active experts -> contiguous logical-core interval candidates
               -> earliest-finish placement
               -> deps from overlapping core intervals
```

输出里 `async_tasks` 是可读 DAG，`async_bridge` 是紧凑数组：

```text
task_expert_ids
task_core_begins
task_threads
task_dep_offsets
task_deps
```

这些数组对应 `fused_moe_bf16_tiled_async` 的输入格式，但上述命令只运行 Python
simulator 和已有 JSON cost table，不会执行真实 MoE/GEMM。

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
