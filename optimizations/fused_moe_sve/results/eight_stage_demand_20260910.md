# 8T输入形状、GEMM阶段与访问状态的DRAM需求（2026-09-10）

## 结论

完成两轮PMU及一轮无PMU控制。固定8T也不能使用统一DRAM压力常数：W13/M1轮换32份
权重约61GB/s，M48约26GB/s；重复一份权重的全部条件只约0.014–0.404GB/s。
W2模式不同，M1与M12读带宽接近、M48下降；W2/M1轮内波动较大。
连续/stride7输出路由没有跨两轮稳定的额外DRAM读压力，不增加固定路由惩罚。

这里测量独立工作的已服务DRAM请求速率，作为对外压力代理；不是排队延迟、未满足的
请求供给率，也不是前台受竞争后的减速函数。不改planner、生产代码或现有基线。

## 协议与边界

E类Lab；独立新采集器直接调用未改生产exact-M JIT。Arm-codex-internal，
/home/zhangxu/codex/fused_cpp，NUMA3，controller CPU240，8个worker CPU312–319。
H4096/F512，BF16 SVE256，Ntile16，W13/W2全stage8MiB/4MiB，worker1MiB/512KiB，
几何(8,0,0,1,1)。W13与W2分别测量，无完整expert gather/阶段继承状态，也无其他计算背景。

输入形状M1/12/13/48；W13固定连续packed A及输出，W2比较route r与7r。A/权重常量1/64，
既有数值参考W13 BF16 0x3f3b、W2 FP32 0.125；输入数值内容不是本次变量。
每个stage比较反复使用同一份B、循环32份B。B工作集分别W13 8/256MiB、W2 4/128MiB，
不能将阶段差异理解成固定相同总工作集的因果对照。
先完成至少64次expert-stage调用，之后ARMED；计数器开后GO，观察100ms。
流量保持到全部PMU停用并ACK后再停worker，输出验证和日志在计数窗口之外。
每次完整stage之间有team barrier，内部M12块之间没有barrier。
不将一个B副本直接称为“L2热”，也不将32副本称为“全冷”；以实测流量定义结果。

普通vector分配，未显式HugeTLB；相同输出地址反复使用，持续写回很低不能外推到真实新输出。
所有输入准备、分配、线程启动和验证均不计入100ms窗口。每条件重建worker但复用分配，
状态受该lead-in与前序随机条件影响；本实验没有逐panel PMU边界，也没有前台减速测量。

25条件=24活动+空载；每轮5warmup+31正式随机轮，PMU seeds611001/611002，无PMU611001。
各900条件，共2700，加两种各25条件烟测，总2750条件完成。
正式样本为3×25×31=2325（含空载）；W13逻辑输出、W2完整输出及未触及route哨兵检查通过，
每活动条件CPU检查通过。各cell全部完成调用中位数用于时间；包含kernel完成barrier，
因此其倒数不是准确调用吞吐，吞吐单独用窗口内完整调用数/window计算。

## 计数定义

NUMA3的16个DDRC，flux_rd=0x84、flux_wr=0x83，各32byte/event，32事件。
每个事件使用自身enabled时间：rate=sum(count×32/enabled_ns)，单位GB/s。
计数RESET只重置count，enabled/running使用相邻读数差；全部running/enabled=1.0。
两轮窗口100.074–101.009ms，无PMU复用。使用每轮空载中位数扣除背景，保留原始计数。
这是控制器总流量，不能将其全部按地址归到B；A、输出写分配与系统噪声均可能贡献。

每调用流量估计=idle-subtracted GB/s÷调用/s，转成MiB。两个观测窗口稍有不同，
不称为单次调用的精确字节计数。没有用kernel服务时间作为uncore计数分母。

## 结果：32份权重轮换、连续路由

| stage | M | 读GB/s 第一/二轮 | 估计读MiB/调用 第一/二轮 | stage us 第一/二轮 |
|---|---:|---:|---:|---:|
| w13 | 1 | 60.96 / 61.35 | 6.78 / 6.84 | 104.88 / 105.29 |
| w13 | 12 | 47.13 / 47.18 | 7.92 / 7.93 | 163.82 / 163.91 |
| w13 | 13 | 44.48 / 44.75 | 11.26 / 11.33 | 252.02 / 251.58 |
| w13 | 48 | 25.79 / 26.03 | 15.98 / 16.13 | 631.87 / 631.45 |
| w2 | 1 | 36.78 / 36.39 | 2.29 / 2.25 | 53.95 / 53.55 |
| w2 | 12 | 37.08 / 38.65 | 3.41 / 3.56 | 84.32 / 84.09 |
| w2 | 13 | 34.38 / 34.45 | 4.63 / 4.63 | 129.31 / 128.71 |
| w2 | 48 | 24.19 / 22.67 | 7.57 / 7.11 | 314.47 / 314.49 |

M增大时，每调用流量和单位时间压力不能混为一谈。例如W13 M48读约16MiB/调用，
超过M1约6.8MiB/调用，但读带宽仅约26GB/s，低于M1约61GB/s。M13也不能简单照搬M12：
它增加尾块，改变平均流量和请求节奏。

重复一份权重时，读GB/s在全部条件/两轮为0.014–0.404；轮换32份时为数十GB/s。
这是DRAM压力差异，不表示缓存内访问成本或LLC压力也消失。
写GB/s空载扣除后约0.00028–0.0323，本协议反复覆盖小输出工作集，不能推断真实W2写流量可忽略。

## 波动与路由对照

W13/M12轮换第一轮读速率CV约2.0%，M48约1.2%；W2/M1轮换约20.1%，
第一轮P10–P90约29.8–49.4GB/s，虽然两轮中位数约36.8/36.4GB/s接近。
因此压力表保留P10/P90包络，不能仅凭两轮中位数接近就赋予确定性。
重复一份权重的DRAM流量接近空载，不用其高百分比CV拟合高精度压力权重。

W2 stride7减连续route的读GB/s按同round配对，用2000次bootstrap中位数95%区间：
M1/M12/M13的轮换对照两轮均包含零；M48仅第一轮略正、第二轮包含零，未跨轮复现。
详见analysis.json的route_contrasts。未对多重比较作显著性声明，不引入路由固定惩罚。
本对照只变输出route布局，A仍为相同packed布局，未覆盖真实gather稀疏输入或不同数值分布。

PMU相对无PMU控制的stage中位时间差绝对值，中位0.171%、最大1.472%。
无PMU控制是独立进程且同seed协议，证明没有明显计时扰动，不是严格同进程配对因果估计。

## 压力表示与后续使用

将一个活跃stage的DRAM需求记录为向量
p=(read_GBps, write_GBps)，以(stage,M,8T,owner-window,访问状态,route布局)为条件，
另保留每调用流量和波动区间。demand_profile.json包含24个条件、两轮中位数及P10/P90包络，
没有未测M插值、没有将副本数映射成真实历史h、没有拟合减速或自动接入planner。

仅以轮换W13/M12约47.16GB/s为参考，W13 M1/M13/M48的读需求约1.30/0.95/0.55倍；
W2 M1/M12/M13/M48约0.78/0.80/0.73/0.50倍。这些是固定实验状态的独立需求代理，
不应在联合运行中无条件相加：竞争可改变任务速度、缓存驻留和发请求节奏。

a) 保留两个GEMM阶段，不能用一个expert任务数表示等压力。
b) 区分流量预算与发出速率；历史与复用状态可能决定访问落在DRAM还是缓存。
c) 下一步若接入联合预测，需受控共运行验证pressure→slowdown，及真实W13→W2继承状态。
   本轮没有单独识别第0/1/2/3个panel的瞬时请求率，不以平均带宽代替该函数。

## 代码、验证与复现

新Lab文件eight_stage_demand_native.cpp、bench_eight_stage_demand.py、analyze_eight_stage_demand.py；
测试tests/test_moe_eight_stage_demand.py（2 passed，计数分母/单位、拒绝复用及形状路由网格）。
Ruff、clang-format及diff检查通过。生产JIT未修改，无全局编译或生产依赖变化。
GCC13.2.0，Linux5.10.0-247.0.0.146.oe2203sp4.aarch64；C++17/O3/pthread，armv8.2-a+bf16+sve、SVE256；确切命令和二进制/JIT身份在build_smoke.sh与
build_identity.txt。执行源码原样保存在产物目录；仓库版随后仅格式化和添加原子序注释，
去除注释/空白比对一致。所有benchmark与分析日志保留。

本地及Arm-codex-internal同名目录tmp/eight_stage_demand_20260910，远端根
/home/zhangxu/codex/fused_cpp。保留所有JSONL、源快照、二进制身份、报告、analysis.json、
demand_profile.json。HEAD c80c0c3e4a8ef12d55bfc66df9c1de306c6a5be5加现有dirty tree，未提交。

```sh
# 远端，烟测先于正式采集
bash tmp/eight_stage_demand_20260910/build_smoke.sh
bash tmp/eight_stage_demand_20260910/run.sh
# 本地，输出路径必须不存在
.venv/bin/python optimizations/fused_moe_sve/benchmarks/analyze_eight_stage_demand.py --root tmp/eight_stage_demand_20260910 --output tmp/eight_stage_demand_20260910/analysis.json
.venv/bin/python tmp/eight_stage_demand_20260910/build_profile.py
.venv/bin/pytest -q tests/test_moe_eight_stage_demand.py
```
