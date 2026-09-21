# 冻结无竞争成本后的8T联合响应拟合（2026-09-10）

## 结论

已完成冻结、受控采集、响应拟合、独立第二轮和留出组合验证。第二轮28个联合条件：
不加竞争MAE4.121us/MAPE3.263%，任务数特征2.936us/2.293%，需求特征1.083us/0.882%。
在这组受控8T/M12条件中，需求加权优于任务数。混合组合差距很小，主要收益来自区分
后台形状强度及权重复用状态。保留Lab响应模型，不替换当前planner或生产默认。

本次M类模型实验加独立Lab采集器；不是完整搜索、真实trace迁移或实际执行提速声明。
前台仅M12，W13/W2分别拟合；未验证其他前台M、宽度、跨LLC、gather或阶段切换。

## 冻结成本、需求与公式

T0取已有真实完整expert路径中，median与uniformish两份输入第一轮M12阶段中位数的均值：
W13=161.765us，W2=82.720us。来源tmp/eight_single_expert_20260910/comparison.json。
需求来源tmp/eight_stage_demand_20260910/demand_profile.json；两份源文件SHA在freeze时保存，
最终再次检查未变化。frozen.json保存成本和完整需求副本；model.json绑定其SHA。

仅拟合 T_s=T0_s*(1+a_s*x+b_s*x²)，a_s,b_s非负，无自由截距。
无背景x=0必定回到原T0，无背景测量不参与拟合，没有按新采集修改基线或补残差。
任务数对照x=n；需求对照x=sum(background read+write GB/s)/47.15856035147088。
归一化参考为表中8T/W13/M12/32副本的读写需求。写流量很小，未额外拟合不可辨识的读写权重。
各后台按自己的stage、M、1/32副本查询，同类直接相加、混合按组成相加。
所有team均8T，因此核心数=8n，拟合后与任务数对照等价，不声称验证了跨宽度推广。

| 模型 | stage | a | b |
|---|---|---:|---:|
| tasks | W13 | 0.015252700 | 0.000000000 |
| tasks | W2 | 0.016172559 | 0.000000000 |
| demand | W13 | 0.018287133 | 0.000393287 |
| demand | W2 | 0.020100800 | 0.000000000 |

按stage分别进行等条件的(T/T0)最小二乘，两个非负参数同一预算，NNLS边界允许b=0。
第一轮仅16个同类、32副本后台条件训练，参数落盘后才启动第二轮，未查看第二轮重新调参。
训练过程完全不使用混合、单份复用或无背景点；测试把全部留出时间扩大100倍，拟合参数不变。

## 受控联合协议

Arm-codex-internal，/home/zhangxu/codex/fused_cpp；NUMA3，controller240，前台312–319，
最多4个后台team依次280–287/288–295/296–303/304–311，全同40核LLC域。
H4096/F512，BF16、SVE256、Ntile16；全W13/W2权重8MiB/4MiB，每线程1MiB/512KiB，
几何(8,0,0,1,1)。前台固定M12、32份权重轮换；后台为W13/W2×M1/M48。

30条件：2个无背景桥接；16个轮换同类训练条件（两前台stage×四后台类型×1/4teams）；
4个混合留出（各前台stage下2×W13/M1+2×W2/M48或2×W13/M48+2×W2/M1）；
8个单份权重复用留出（两前台stage×四后台类型×4teams）。后台复用留出仍有4个任务，
但独立DRAM需求极低，用来检查单纯任务计数是否会造成过度惩罚。

每team有独立A/B/C及权重副本，连续调用同一stage，expert边界team barrier，内部M12块无barrier。
至少每team64次lead-in后ARMED；PMU开→GO→100ms观测→每team确认越过末端→DONE→PMU停→ACK，
然后停止worker和验证。用共同100ms窗口内完整调用的team时间中位数，内含kernel结束barrier。
PMU计数窗口100.084–103.056ms，使用各事件自身enabled时间；全部running/enabled=1.0。
16DDRC读/写32事件，32bytes/event，配置复用已有验证。独立需求求和并非联合流量等式。

普通vector、无显式HugeTLB；未改JIT或生产runtime。W13逻辑值和W2完整输出/未写guard验证；
各team绑核和跨窗口覆盖验证。独立C++线程不使用生产80线程调度器，W2不继承真实W13输出状态；
因此成本仍固定但协议桥接误差必须报告，不将当前模型自动推广到真实计划。

两轮PMU seeds612001/612002及一轮无PMU612001，各5warmup+31正式随机轮，30×36=1080条件；
共3240条件，加无PMU/有PMU各30烟测，总3300条件通过。正式每运行930条件，三运行共2790。
每个条件内包含多个完整stage调用，原始文件保留各team调用数、时间和DDRC计数。

## 第二轮与留出结果

每个条件以31次观测中位数作为实测，误差跨条件等权，单位us；同类第二轮也未参与训练。

| 验证集 | 条件数 | 不加响应 MAE | 任务数 MAE / MAPE | 需求 MAE / MAPE |
|---|---:|---:|---:|---:|
| 第一轮训练 | 16 | 4.811 | 1.732 / 1.298% | 0.830 / 0.650% |
| 第一轮留出 | 12 | 2.991 | 4.619 / 3.684% | 1.364 / 1.153% |
| 第二轮全部联合 | 28 | 4.121 | 2.936 / 2.293% | 1.083 / 0.882% |
| 第二轮全部轮换 | 20 | 5.153 | 1.681 / 1.280% | 0.914 / 0.697% |
| 第二轮混合 | 4 | 6.380 | 1.230 / 0.960% | 1.155 / 0.875% |
| 第二轮单份复用 | 8 | 1.539 | 6.072 / 4.825% | 1.504 / 1.344% |

第二轮不含复用留出的轮换条件，MAE仍从1.681降到0.914us，说明收益不全来自识别低DRAM状态。
但混合仅1.230→1.155us，差距很小。复用条件不加响应也有1.539us MAE，需求响应1.504us，
因此这一部分主要是避免任务数模型错误地加大惩罚，而不是识别了新的明显减速。

| 第二轮前台W13，4个后台 | 实测us | 任务数预测us | 需求预测us |
|---|---:|---:|---:|
| W13/M1，32副本 | 178.050 | 171.634 | 178.823 |
| W13/M48，32副本 | 166.940 | 171.634 | 168.574 |
| W13/M1，单份复用 | 163.380 | 171.634 | 161.824 |
| 2×W13/M1+2×W2/M48 | 170.290 | 171.634 | 173.198 |

混合case也有局部回退：上表最后一项需求预测173.198us，实测170.290us，比任务数171.634us更差。
不能只看平均改善，也不能由两种混合组成推断任意组合的相互作用已解决。

## 基线桥接、噪声和流量解释

无背景W13第一/二轮163.440/163.550us，对固定161.765us约+1.04%/+1.10%；
W2为83.690/83.320us，对82.720us约+1.17%/+0.73%。这部分保持为验证残差，不回拟合。
全部条件目标中位时间的最大轮内CV0.615%；PMU相对独立无PMU控制的时间差绝对值
中位0.098%、最大0.728%。噪声/控制有效性支持本组局部比较，不保证单次或其他机器精度。

独立需求和仅作特征。例如4个W13/M1轮换+W13前台，表中和约291.8GB/s，联合实测读流量
约241.2GB/s；其他混合也可能实测高于表中和。后台变慢、缓存驻留和请求节奏的反馈仍存在。
因此模型是条件经验响应，不是DRAM流量守恒预测或请求队列物理模型。

## 交付与后续边界

冻结成本+需求+划分在frozen.json，第一轮系数在model.json，逐条件第二轮预测及控制在validation.json。
可调用fit_eight_joint_response.predict；目前只验证本组前台8T/M12、同LLC、连续后台的条件。
未测前台M、team宽度、背景数量2/3、实际交替W13→W2、动态结束/加入及真实trace均不得算作已验证。
不在本轮切换planner基线。下一步应验证其他前台M和阶段动态重叠，再决定真实trace接入。

新增Lab采集器eight_joint_response_native.cpp、驱动bench_eight_joint_response.py、
拟合/验证fit_eight_joint_response.py及tests/test_moe_eight_joint_response.py。
5 tests passed（冻结基线、非负响应、零压力、留出泄漏、背景placement）；Ruff、clang-format和diff通过。
HEAD c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5加既有dirty tree，源状态保存，无提交。
GCC13.2.0、C++17/O3/pthread、armv8.2-a+bf16+sve、SVE256；build_identity.txt保留新binary和未改JIT身份。

本地和远端同名tmp/eight_joint_response_20260910，远端根/home/zhangxu/codex/fused_cpp。
保留全部JSONL、stdout/stderr、编译/运行脚本、源快照、冻结与拟合文件、验证结果及报告。

```sh
# 本地，已在任何新测量前执行
.venv/bin/python optimizations/fused_moe_sve/benchmarks/fit_eight_joint_response.py freeze --root tmp/eight_joint_response_20260910 --isolated tmp/eight_single_expert_20260910/comparison.json --demand tmp/eight_stage_demand_20260910/demand_profile.json
# 远端，先烟测再正式；run.sh在第一轮结束后train，之后才运行第二轮和控制，最后evaluate
bash tmp/eight_joint_response_20260910/build_smoke.sh
bash tmp/eight_joint_response_20260910/run.sh
# 本地验证
.venv/bin/pytest -q tests/test_moe_eight_joint_response.py
```

输出路径使用排他创建，复现应使用新目录；不得覆盖原始测量或事后调整留出划分。
