# 8T 完整块、精确尾块与条件竞争模型（2026-09-10）

## 结论与范围

按无竞争基线、独立竞争增量、第二轮及真实 trace 的顺序完成 Lab 实验。
第二轮微基准复现良好；真实 trace 的 W2 M13–24 明显改善，但 W13 和 M12 回退。
8T lane 完成时间更接近实测，不能据此认定各阶段已校准或切换当前实验基线。
本次仅本地拟合和固定计划重放，无新硬件采集、无完整 planner 搜索、无生产改动。
对照为 planner_gather16（宽窄 team 倍率已移除，16T gather 已加入）。

## 模型与数据隔离

对 M=12q+r、stage s，T=sum(h=0..q-1) F[12,s](h)+F[r,s](q)，r=0 时无尾项。
第一轮无背景的 50 个节点建立 F0；保存 no_background.json 后冻结。
随后用同一轮 100 个背景节点建立 delta=Fbg-F0，Fbg=F0+delta。
拟合程序验证剥离 delta 后基线逐字段不变；负增量保留，不强制单调或夹成零。
W13/W2 分开，行数 1/2/4/8/12，历史 0/1/2/3/7；历史区间内线性插值。
历史 4–6 没有独立节点验证。第二轮不参与拟合。

8T 实际 owner 条带分别为 W13 1MiB、W2 512KiB；H4096/F512，TP4，FP32 down output。
原始协议、正确性和分配限制见 [采集报告](eight_team_history_20260910.md)。
无背景 W2 的 1 行尾块 h7 两轮约 41.48/42.08us，而旧比例缩放约 12us。
不再将该尾块按 M12 的计算行比例缩放。

适配器只替换 M≤96 且余数为 0/1/2/4/8 的完整 expert，未测尾行和更大 M 保留原模型。
新 phase 分解保留总计算和访存工作量。边界存在部分覆盖造成的不连续，不能直接用于全形状搜索。
竞争拟合只覆盖同 LLC 的四个 8T W13 M2 或 M120 背景 team。
适配器只有当前其他任务恰好为这四个描述时才替换旧竞争倍率，其他环境保留旧竞争响应。
即便描述相同，真实任务也不等价于微基准的持续四份权重轮换。

## 第二轮验证

单位 us；块级实测为八线程尾块服务时间最大值的中位数。

| 条件/预测 | 块 MAE | 块 MAPE | 整段 MAE | 整段 MAPE |
|---|---:|---:|---:|---:|
| 无背景（50 点） | 2.052 | 2.663% | 6.626 | 1.603% |
| 有背景，仅新基线（100 点） | 4.013 | 4.923% | 6.004 | 3.362% |
| 有背景，新基线+增量（100 点） | 2.315 | 2.840% | 6.092 | 2.003% |

竞争项改善块误差和整段相对误差，但整段绝对误差略增，不能称全部指标改善。
整段预测为各块最大线程服务中位数相加；实测为最早开始到最晚结束的 team 包络。
panel 间无 barrier、不同块慢线程可不同、尾块地址固定在行96，二者并非严格同一计时量。
第一轮150节点是查表拟合，不作为泛化证据；第二轮只验证同条件重复性。

## 固定真实 trace 迁移

来源 tmp/planner_repaired_pressure_20260910，median/uniformish 原 anchor 计划，
两轮各31测量，early merge 关闭，计算完成时间口径；没有重新选择计划。
每个 expert 每轮取阶段中位数，表中为这些观察的 MAE；支持子集只含已接入的精确尾行。

| case / 阶段 / M | 旧 MAE us | 新 MAE us |
|---|---:|---:|
| median W2 M13–24（支持子集，34观察） | 36.69 | 15.39 |
| uniformish W2 M13–24（支持子集，42观察） | 35.30 | 9.09 |
| median W2 M25–48（支持子集，12观察） | 20.82 | 25.94 |
| uniformish W2 M25–48（支持子集，26观察） | 24.10 | 24.89 |
| median W13 M13–24（支持子集） | 40.07 | 98.22 |
| uniformish W13 M13–24（支持子集） | 23.05 | 83.56 |
| median W13 / W2 M12 | 12.41 / 9.45 | 27.77 / 15.12 |
| uniformish W13 / W2 M12 | 4.70 / 4.28 | 25.47 / 9.44 |
| median W13 / W2 M49–96（全部） | 36.49 / 18.13 | 97.10 / 33.24 |
| uniformish W13 / W2 M49–96（全部） | 62.30 / 20.61 | 96.49 / 31.90 |

W2 M13–24 支持子集的加权有符号偏差由 -24.06%/-23.96% 变为 +4.86%/+6.13%。
W13 同组变为 +33.22%/+28.86% 高估。M97+ 自身基线未改，但阶段事件与共享竞争
反馈改变预测，W13 MAE median166.85→193.95us、uniformish148.35→175.37us。
这些结果说明当前新基线与旧联合响应的组合迁移失败；尚不能唯一归因于旧竞争倍率，
因为微基准和真实运行的输入、分配、计时以及 W13→W2 继承状态也不同。

| 完成时间指标 | median | uniformish |
|---|---:|---:|
| 8T lane MAE，旧→新 ms | 2.274→1.614 | 2.775→1.792 |
| 总完成预测，旧→新 ms | 25.661→26.326 | 24.848→26.223 |
| 两轮实测总完成 ms | 28.330 / 28.495 | 27.822 / 27.830 |

8T lane 两版本均低估。median 真实最慢 lane 是16T，总时长改善不能全部归给8T。
新基线覆盖 median75/182、uniformish102/234 个8T expert。
两份重放的已测竞争描述命中均为0（分别1300/2338次新块参与的事件状态），
所以“仅换基线”和“接入限定竞争项”的真实结果完全相同。
不能从这两份 trace 判断新竞争项的真实收益，也不能将两种背景推广到任意并发组合。

## 验证、复现与后续边界

- .venv/bin/pytest -q tests/test_moe_eight_block_model.py：4 passed。
- Ruff 对四个新增脚本和上述测试检查通过。
- 测试覆盖基线冻结、精确尾块组合、范围回退、资源总量守恒、竞争条件门控。
- 没有修改 kernel，无本轮新增硬件性能或数值正确性声明；原采集数值门槛见采集报告。

复现拟合（输出目录必须不存在）：

```sh
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_eight_block_model.py --sessions tmp/eight_team_history_20260910/session1.jsonl tmp/eight_team_history_20260910/session2.jsonl --output-dir tmp/eight_block_model_reproduce
.venv/bin/python optimizations/fused_moe_sve/benchmarks/validate_eight_trace_transfer.py --source tmp/planner_repaired_pressure_20260910 --profile tmp/eight_block_model_reproduce/model.json --output tmp/eight_block_model_reproduce/real_trace_validation.json
```

产物在 tmp/eight_block_model_20260910，包括两阶段 profile、逐点验证、完整任务/阶段/lane 重放和摘要。
下一步应先用真实 W13→W2 顺序、分配/输入与计时口径做无背景桥接，固定已测基线，
再采实际 trace 中出现的竞争组合；避免通过真实 trace 整体残差回拟合而混淆这两层。
当前 adapter 留作实验，原 accepted Lab baseline 不切换。
