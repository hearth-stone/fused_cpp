# DEPRECATED: Wave-based scheduling(后续不考虑)

**状态(2026-07-04):基于 wave 的离线模拟器与调度算法已删除(阶段 1)。**

当前 async interval-DAG、quick/full planner 与论文证据入口分别见
[`MATHEMATICAL_MODEL.md`](MATHEMATICAL_MODEL.md)、
[`planners/plan_schema.md`](planners/plan_schema.md) 和
[`../docs/moe_paper_readiness.md`](../docs/moe_paper_readiness.md)。

Wave 调度整体废弃,不再演进,也不作为 planner 的目标形态。相关离线代码已从工作树移除;
需要历史参考请查 git 历史(删除提交见 `git log -- cpu_moe_schedule_optimization/planners/offline_simulator.py`)。

## 为什么废弃

Wave 模型 = 一批 team 并行 + **wave 之间全局 barrier**,makespan = `Σ max(team_time)`。
- 全局 barrier 在倾斜路由下产生大量气泡(最慢的 team 拖住整批);
- 结构僵硬,粒度太粗(必须预先把专家打包成"一起开始"的组、预分配线程数),且对 cost-model 预测误差敏感;
- **async interval-DAG 是 wave 的超集**(令"wave k 的所有 task 依赖 wave k−1 的全部"即退化为 wave),故退役 wave 无调度能力损失。

## 阶段 1 已删除的文件(离线模拟器 + 调度算法)

- `planners/offline_simulator.py` —— wave 核心(`Wave`/`Team`、`wave_offsets` scheduled bridge、
  wave 版 planner kinds:`FIXED_GLOBAL_THREADS` / `SORTED_TOKEN_BALANCED_1T` / `UNIFORM_WAVES` /
  `GREEDY_MARGINAL_GAIN` / `ENUMERATE_CORE_GROUPS`,以及被取代的 `plan_async_interval_dag`)。
- `planners/twin_{karmarkar_karp,log_load,load_proportional,sqrt_load,heavy_light_hybrid}.py` —— 5 个 twin 算法。
- `benchmarks/{scheduled_bridge_bench,synthetic_sweep,selector_stress,profile_native_planner_cost}.py` —— wave 基准/扫描。
- `cost_model/build_lightweight_planner_cost.py` —— wave planner-cost 表构建工具。

## 替代(现行、已硬件验证的 async 栈)

| 旧(已删除) | 新(现行) |
| --- | --- |
| `offline_simulator.py` 的 makespan 估计 | `cost_model/phase_model.py` `ContentionCostModel.dag_makespan`(事件驱动 + 争用 derate + overhead-split,AWS 实测 ~2% median) |
| `offline_simulator.py` 的 planner + twin 算法 | `planners/interval_planner.py` `IntervalPlanner`(async interval-DAG,cost-model 驱动) |
| `synthetic_sweep.py` / `scheduled_bridge_bench.py` 的算法对比 | `planners/simulate_schedules.py`(离线对比多种调度算法,用验证过的模型) |
| `profile_native_planner_cost.py` planner 开销 | `planners/bench_planner_overhead.py`(冷/热 plan 开销 + 缓存命中率) |

## 明确保留(不属于 wave,继续演进)

- **async interval-DAG 运行时**:`fused_moe_bf16_tiled_async`(按核区间 + CSR 依赖、无全局 barrier)——前进方向。
  > 注:wave **运行时** `fused_moe_bf16_tiled_scheduled`(C++ + 导出 + 正确性测试)本阶段**未删除**,留待阶段 2 单独决定。
- **cost model / T_expert**:`cost_model/profile_expert_cost.py`、`profile_schema.md`、`profiles/*`。
- **N-split / M-vs-N split 研究**:`SPLIT_MN_FINDINGS.md`。

> 一句话:**wave 离线模拟器/算法已删(阶段 1);async interval-DAG + cost model + 新对比工具保留并继续;wave 运行时留待阶段 2。**
