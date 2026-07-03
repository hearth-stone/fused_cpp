# DEPRECATED: Wave-based scheduling(后续不考虑)

**状态(2026-07-03):基于 wave 的调度整体废弃,后续不再考虑。**

保留现有代码/文档仅作历史参考;不再演进、不作为 planner 的目标形态。

## 为什么废弃

Wave 模型 = 一批 team 并行 + **wave 之间全局 barrier**,makespan = `Σ max(team_time)`。
- 全局 barrier 在倾斜路由下产生大量气泡(最慢的 team 拖住整批);
- 结构僵硬,粒度太粗(必须预先把专家打包成"一起开始"的组、预分配线程数),且对 cost-model 预测误差敏感。

## 废弃范围(以下都不再考虑)

- **Plan → Wave → Team 里的 `Wave` 层**:`estimated_wave_time_ns`、`wave.teams`、空 wave 规则等。
- **wave 版 scheduled bridge**:`scheduled_bridge` 的 `wave_offsets` 表示,以及 `fused_moe_bf16_tiled_scheduled` 的 wave 用法。
- **wave 版 planner kinds**:`FIXED_GLOBAL_THREADS`、`SORTED_TOKEN_BALANCED_1T`、`UNIFORM_WAVES`、`GREEDY_MARGINAL_GAIN`、`ENUMERATE_CORE_GROUPS`。
- **wave 版 planner 实现/基准**:`planners/offline_simulator.py`(wave planners)、`planners/twin_*.py`、`benchmarks/scheduled_bridge_bench.py` 的 **wave 桥部分**、`benchmarks/profile_native_planner_cost.py`。

## 明确保留(不属于 wave,继续演进)

- **async interval-DAG**:`ASYNC_INTERVAL_DAG`、`async_bridge`、`fused_moe_bf16_tiled_async`(按核区间依赖、无全局 barrier)——**前进方向**,配合动态/弹性 N-split。
- **cost model / T_expert**:`cost_model/profile_expert_cost.py`、`profile_schema.md`、`profiles/*`(与调度形态无关,继续用)。
- **即将做的并发争用(derate)表** 与 **N-split / M-vs-N split 研究**(`SPLIT_MN_FINDINGS.md`)。

> 一句话:**wave 废弃;async interval-DAG + 弹性 N-split + cost model 保留并继续。**
