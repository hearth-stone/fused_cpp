# CPU MoE 调度数学模型

> 状态：调度问题定义的 source of truth。
>
> 最后更新：2026-07-14。
>
> 修改 planner 的决策变量、目标函数、资源约束、线程宽度集合、调度语义、
> rank 耦合方式或剪枝策略时，必须同步更新本文档及末尾变更记录。

## 1. 文档边界

本文档将 MoE 调度分为四层：

1. **原始数学问题**：定义完整可行域，不包含当前实现的经验剪枝。
2. **实例参数**：kernel/cost model 提供执行时间和资源需求。
3. **求解器编码**：CP-SAT、MILP 或启发式对同一个问题的不同表达。
4. **工程剪枝**：线程宽度、静态 shape、LPT、non-idling 等当前限制。

M12、SVE、W13 split、具体权重布局等属于实例参数生成过程，不属于核心问题
定义。核心问题只观察它们最终产生的执行时间和资源需求。

## 2. 原始问题

### 2.1 集合与参数

设一次 rank-local MoE 执行包含 expert job 集合：

$$
\mathcal J=\{1,\ldots,n\}.
$$

可用的最大 CPU 核心/线程数为：

$$
T_{\max}=C.
$$

原始问题允许每个 job 使用任意整数线程宽度：

$$
\mathcal T_i^{(0)}=\{1,2,\ldots,T_{\max}\}.
$$

对 expert $i$ 和线程数 $t$，实例参数生成器提供：

$$
p_i(t)>0,
$$

其中 $p_i(t)$ 是 job 非抢占执行时的处理时间。它可以依赖 route 数和
kernel 实现，但这些依赖不进入核心调度公式。

设 $\mathcal R$ 为 CPU 之外的 cumulative resource 集合。对资源
$r\in\mathcal R$：

$$
d_{i,r}(t)\ge0
$$

表示 expert $i$ 使用 $t$ 个线程时对资源 $r$ 的需求。这里 $d$ 是 demand，
$i$ 和 $r$ 是两个独立下标，不是一个名为 `dir` 的变量。$B_r$ 表示资源 $r$
的容量，并且必须与 $d_{i,r}(t)$ 使用相同单位。例如 LLC 需求与容量都使用
bytes，保守预留的内存带宽与带宽容量都使用 bytes/s。总访问字节数不是瞬时
cumulative demand；它必须先转换成带宽需求，或者吸收到 $p_i(t)$ 中。若某类
性能影响已经包含在 $p_i(t)$ 中，则不应在 $d_{i,r}(t)$ 中重复计算。

可选的 precedence 集合为：

$$
\mathcal E\subseteq\mathcal J\times\mathcal J.
$$

$\mathcal J\times\mathcal J$ 是所有有序 job 对的集合；$(i,j)\in\mathcal E$
表示 job $i$ 必须完成后 job $j$ 才能开始。$\mathcal E$ 是问题输入中的强制
数据依赖，必须构成有向无环图；它不是 planner 为共享 CPU 而选择的 lane 顺序。

当前将完整 expert 抽象为一个 job 时，各 expert 相互独立，因此
$\mathcal E=\varnothing$。若将 expert 展开成通用 phase DAG，则 job 集合中的
节点改为 phase，并使用例如 $(\mathrm{W13},\mathrm{W2})\in\mathcal E$ 表示
phase 依赖。

### 2.2 决策变量

每个 expert 选择一个整数线程宽度和开始时间：

$$
t_i\in\{1,\ldots,T_{\max}\},\qquad s_i\ge0.
$$

线程数在 job 开始前选择，执行过程中保持不变。因此 job 是 **moldable**，
不是执行中可改变线程数的 malleable job。

完成时间为：

$$
f_i=s_i+p_i(t_i).
$$

定义非抢占式活跃指示函数：

$$
a_i(\tau)=\mathbf 1[s_i\le\tau<f_i].
$$

$a_i(\tau)$ 是由 $s_i$ 和 $f_i$ 推导出的辅助函数，不是独立决策变量。当
job $i$ 在时刻 $\tau$ 正在执行并占用 CPU、LLC 等资源时取 1，否则取 0。
非抢占语义保证其取 1 的区域是单个连续区间 $[s_i,f_i)$；若允许抢占，一个
job 可能对应多个不连续活跃区间，不能再只用上述单一区间定义。

该指示函数用于简洁表达“任意时刻所有活跃 job 的资源需求之和不能超过容量”。
实际 CP-SAT 实现通常通过 interval variable 和 cumulative constraint 等价表示，
不需要为连续时间轴上的每个 $\tau$ 显式创建一个 $a_i(\tau)$ 变量。

### 2.3 资源约束

任意时刻的 CPU 使用量不能超过 $T_{\max}$：

$$
\sum_{i\in\mathcal J}t_i a_i(\tau)\le T_{\max},
\qquad\forall\tau.
$$

其他 cumulative resource 满足：

$$
\sum_{i\in\mathcal J}d_{i,r}(t_i)a_i(\tau)\le B_r,
\qquad\forall r\in\mathcal R,\ \forall\tau.
$$

这里 cumulative 表示同一时刻的资源需求可加，并不表示资源消耗随时间永久
累积。对固定资源 $r$ 和时刻 $\tau$，只有 $a_i(\tau)=1$ 的活跃 job 贡献
$d_{i,r}(t_i)$；它们的总需求不得超过容量 $B_r$。Job 完成后需求立即释放，
容量可由后续 job 重新使用，因此这是 renewable cumulative resource。

例如 $r=\mathrm{LLC}$、$B_r=32\ \mathrm{MiB}$，两个活跃 job 分别需要
$8\ \mathrm{MiB}$ 和 $16\ \mathrm{MiB}$，则总需求为 $24\ \mathrm{MiB}$，
满足约束。若第三个活跃 job 还需要 $12\ \mathrm{MiB}$，总需求变为
$36\ \mathrm{MiB}$，在硬容量模型中不可行。

LLC budget 可以直接使用上述硬约束。内存带宽等资源在真实硬件上通常表现为
超过需求后逐渐降速，而不是不可执行；将其写成 cumulative constraint 是保守
近似。若需要描述软争用，应使用速率分配或 active-set slowdown 模型，而不是
把总访问字节数直接放入该约束。

前驱约束为：

$$
s_j\ge f_i,
\qquad\forall(i,j)\in\mathcal E.
$$

### 2.4 目标函数

Rank-local compute makespan 为：

$$
C_{\max}=\max_{i\in\mathcal J}f_i.
$$

原始优化问题为：

$$
\boxed{
\begin{aligned}
\min_{\{s_i,t_i\}}\quad
& C_{\max}\\
\mathrm{s.t.}\quad
& t_i\in\{1,\ldots,T_{\max}\}, &&\forall i,\\
& f_i=s_i+p_i(t_i), &&\forall i,\\
& \sum_i t_i a_i(\tau)\le T_{\max}, &&\forall\tau,\\
& \sum_i d_{i,r}(t_i)a_i(\tau)\le B_r,
&&\forall r,\tau,\\
& s_j\ge f_i, &&\forall(i,j)\in\mathcal E,\\
& s_i\ge0. &&
\end{aligned}}
$$

在运行时比较不同 planner 时，还需要计入 planning latency：

$$
T_{\mathrm{runtime}}=T_{\mathrm{plan}}+C_{\max}.
$$

Dispatch、communication、scatter 和 combine 在不受 plan 影响时作为外部成本；
若其成本会随 plan 改变，则必须提升为目标函数的一部分。

## 3. 多 Rank 外层

对 rank $k\in\mathcal K$，定义 rank-local 完成时间 $C_k$。不考虑跨 rank
执行期争用时：

$$
C_{\mathrm{compute}}=\max_{k\in\mathcal K}C_k.
$$

完整 layer 目标为：

$$
T_{\mathrm{layer}}
=T_{\mathrm{dispatch}}
+\max_k C_k
+T_{\mathrm{communication}}
+T_{\mathrm{scatter}}
+T_{\mathrm{combine}}.
$$

若 rank 共享内存带宽等资源，则应在所有 rank 的联合活跃集合上增加共享资源
约束。一个 rank 完成后，联合活跃集合随之改变，剩余 rank 必须切换到新的资源
容量，而不能将 dual-rank 速率应用到整个执行区间。

## 4. 问题类型与复杂度

该问题可描述为：

$$
P\mid\mathrm{moldable,\ nonpreemptive,\ cumulative\ resources,\ prec}
\mid C_{\max}.
$$

这只是描述性记号，不声称存在覆盖全部扩展项的单一标准 Graham notation。

### 4.1 混合离散-连续与非凸性

原始表达同时包含整数线程宽度 $t_i$ 和连续开始时间 $s_i$，因此更准确的说法
是 **混合离散-连续组合优化问题**，而不是纯离散问题。给定线程宽度和 job
相对顺序后，开始时间可以由有限个完成事件确定，所以困难部分主要来自有限但
指数规模的宽度、并发集合和顺序选择。

可行域不是凸集。严格地说，不是标量变量 $t_i$ “不是凸集”，而是它的离散
可行域

$$
\mathcal T_i=\{1,\ldots,T_{\max}\}
$$

不是凸集。凸集要求任取 $x,y\in\mathcal T_i$ 和 $\lambda\in[0,1]$，都有
$\lambda x+(1-\lambda)y\in\mathcal T_i$。当 $T_{\max}\ge2$ 时，取 $x=1$、
$y=2$ 和 $\lambda=1/2$，得到 $3/2\notin\mathcal T_i$，因此不满足该定义。
将线程宽度放松为连续区间 $[1,T_{\max}]$ 后，这一维可行域是凸集，但会引入
不可执行的分数线程数，所以它只是原问题的连续松弛。

即使固定所有线程宽度，非重叠调度仍然非凸。例如一颗核心上的两个单位时长
job 有两个可行调度：

$$
(s_1,s_2)=(0,1),\qquad (s_1,s_2)=(1,0).
$$

两者中点为：

$$
(s_1,s_2)=\left(\frac12,\frac12\right),
$$

此时两个 job 在 $[1/2,3/2)$ 重叠并违反单核容量约束。因此两个可行点的凸
组合可能不可行，可行域非凸。

### 4.2 判定版本

给定整数或经统一缩放后的有理数参数，以及 deadline $D$，定义判定问题：

$$
\textsc{Moldable-Cumulative-Schedule}(D):
\quad
\text{是否存在可行 }\{s_i,t_i\}\text{ 使 }C_{\max}\le D\text{？}
$$

该离散判定版本属于 NP。证书可以包含每个 job 的线程宽度和开始时间。对整数
duration，任意可行调度都可以在不增加 makespan 的条件下左移为 active
schedule；每个开始时间为 0，或由某个 precedence/resource blocking job 的
完成事件确定，因此具有多项式位数的表示。将所有开始、结束事件排序后，活跃
集合只会在这些有限事件处发生变化，所以 CPU、cumulative resource、
precedence 和 deadline 约束都能在多项式时间内验证。

### 4.3 由 3-PARTITION 归约

给定一个 `3-PARTITION` 实例：$3m$ 个正整数 $a_1,\ldots,a_{3m}$ 和整数 $B$，
满足：

$$
\frac{B}{4}<a_i<\frac{B}{2},
\qquad
\sum_{i=1}^{3m}a_i=mB.
$$

构造如下调度实例：

$$
T_{\max}=m,\qquad
\mathcal R=\varnothing,\qquad
\mathcal E=\varnothing,
$$

$$
p_i(1)=a_i,
\qquad
p_i(t)=B+1\quad(2\le t\le m),
$$

并令 deadline 为：

$$
D=B.
$$

由于任意 $t\ge2$ 都有 $p_i(t)>D$，所有 deadline 内的可行调度都必须选择：

$$
t_i=1.
$$

问题因此退化为把 $3m$ 个处理时间为 $a_i$ 的非抢占 job 调度到 $m$ 个相同
核心上，并要求 makespan 不超过 $B$。总工作量正好是 $mB$，所以每个核心的
负载必须恰好为 $B$。又因为 $B/4<a_i<B/2$，每个核心恰好包含三个 job。

因此：

$$
C_{\max}\le B
\quad\Longleftrightarrow\quad
\{a_i\}\text{ 存在一个合法的 3-partition}.
$$

该构造为多项式归约，而 `3-PARTITION` 是强 NP-complete。因此：

- 离散判定版本是强 NP-complete；
- 原始 makespan 优化版本是强 NP-hard；
- 固定 processing-time 的最简特例已经具有该困难度；
- 可变线程宽度、额外 cumulative resources、precedence 和 active-set
  interference 形成的是更一般的问题，不会消除 NP-hardness。

## 5. Kernel Implementation Variant

Split/no-split 以及未来其他 kernel 版本不属于核心问题变量。一个全局
implementation variant $v\in\mathcal V$ 只负责生成实例参数：

$$
\Theta_v=\{p_i^{(v)}(t),d_{i,r}^{(v)}(t)\}.
$$

对每个 variant 求解同一个调度问题：

$$
C^*(v)=\min_{s,t}C_{\max}(s,t;\Theta_v),
$$

再进行外层选择：

$$
C^*=\min_{v\in\mathcal V}C^*(v).
$$

因此 variant 只改变问题实例，不改变问题类型。若两个 variant 在所有相关
线程宽度上同时具有不更大的时间和资源需求，则被支配的 variant 可以在进入
planner 前删除。

## 6. 求解器编码

### 6.1 数学定义

核心定义直接使用整数变量 $t_i$。这保持问题表达简洁，不绑定具体求解器。

### 6.2 CP-SAT

CP-SAT 可以为每个 $(i,t)$ 建立 optional interval，并要求每个 expert 恰好
选择一个线程宽度。CPU 和 LLC 使用 cumulative constraints。CP-SAT 适合作为
离线、未剪枝问题的 oracle，并返回可行上界和理论下界。

### 6.3 MILP

标准 MILP 通常将 $t_i$ 展开为 one-hot 变量：

$$
y_{it}\in\{0,1\},\qquad
\sum_{t=1}^{T_{\max}}y_{it}=1.
$$

然后线性表示：

$$
t_i=\sum_t t\,y_{it},\qquad
p_i=\sum_t p_i(t)y_{it}.
$$

One-hot 是等价求解器编码，不是线程宽度剪枝。

## 7. 当前实现相对原始问题的剪枝

| 层级 | 原始可行域 | 当前限制 | 性质 |
| --- | --- | --- | --- |
| 线程宽度 | $1,2,\ldots,T_{\max}$ | `1,2,4,8,16,32` | 离散宽度剪枝 |
| 并发配置 | 活跃 job 可形成任意宽度组合 | 整次调用使用静态 core shape | static-partition 剪枝 |
| Shape 集合 | 所有满足容量的整数组合 | profile 支持并经 active 工作集规则筛选的 shape；owner-cache band 当前仅 shadow | 候选剪枝 |
| Assignment | 任意 expert-to-resource 调度 | 按 isolated cost 的 LPT | 启发式分配 |
| Ordering | 任意可行开始时间和顺序 | 每个 lane 的 LPT 顺序 | 顺序剪枝 |
| Idling | 允许主动等待以避开争用 | dependency ready 后立即启动 | non-idling 剪枝 |
| Kernel variant | 任意未被支配的实现 | 当前 split/no-split profile pair | 实例候选限制 |
| Processing time | 任意可测的 $p_i(t)$ | active 经验公式；ECM/roofline 仅 shadow | cost 近似，不剪枝可行域 |

当前 `IntervalPlanner` 搜索的是上述剪枝后 plan space 中的方案，不是原始问题
的全局最优方案。

Amazon 192-core NUMA0 的 schema-v2 profile 已将实测线程域扩展到
`1,2,4,8,16,32,48,64,96`，但这只扩大 $p_i(t)$ 的校准域。当前 planner 的
线程宽度剪枝仍为表中所列的 `1,2,4,8,16,32`；在完成 48/64/96T 的 held-out
regret 验证前，不自动扩大在线决策空间。

## 8. 当前 Cost Model 的位置

原始问题使用固定的 $p_i(t)$ 和硬 cumulative resource 约束。当前工程实现还
使用 stage-aware event simulator，让执行速率依赖并发 active set。令
$x_i(\tau)\in[0,1]$ 表示 job 完成比例，则该扩展可以写成：

$$
\frac{dx_i(\tau)}{d\tau}
=
\frac{a_i(\tau)}
{p_i(t_i)D_i(\mathcal A(\tau))},
\qquad
x_i(s_i)=0,\quad x_i(f_i)=1.
$$

### 8.1 当前 active isolated model

新生成且携带 `iso_formula` 的 schema-v2 profile 使用：

$$
p_i(t)=T_{\mathrm{iso}}(R_i,t)=O(t)+C(R_i)\phi_{\mathrm{USL}}(t)k_\phi(t),
$$

$$
O(t)=o_0+\frac{o_1}{t},\qquad
\phi_{\mathrm{USL}}(t)=\frac{1+\alpha(t-1)+\beta t(t-1)}{t}.
$$

$C(R)$ 是单线程 route-work 的一维实测校准曲线；$O(t)$ 和
$\phi_{\mathrm{USL}}(t)$ 分别表示固定开销与相对理想 $1/t$ 的线程效率基线，
$k_\phi(t)$ 是与 route 无关的一维实测校正。M1/M2/M4/M8 与前两个
M12 panel 的启动/线程效率不同于 steady-state M12 bulk，因此保留小 M 的
`(tail, threads)` 实测 residual；更大的 M 使用公式，并按 kernel 组合规则
追加 tail。公式仅在 profile 的已校准线程域内有效，不承担跨机器或更高线程数
外推。没有序列化公式的历史 profile 继续使用旧二维 table，避免离线重拟合改变
既有 planner 基准；可显式指定 `iso_mode="formula"` 做 shadow 对比。

Amazon 192-core NUMA0 的 TP4/TP2/EP4/EP2 split/no-split profile 继续使用同一
公式，`thread_domain` 扩展到 `[1,96]`，并以 48/64/96T 实测点校正
$k_\phi(t)$；公式形式及小 M residual 组合规则没有改变。

### 8.2 可解释 GEMM ECM shadow model

简单的 $\max(F/P,Q/B)$ 无法表示当前 BF16 microkernel 的 L1 重复加载、
BFMMLA 发射、不同 cache 层以及 fused epilogue。新的 `gemm_ecm.py` 因此先对
W13 和 W2 分别建模，再由 `T_iso` 组合两个 stage 与 gather/scatter。

令 $\mathcal P(R)$ 为 route 数 $R$ 对应的物理 panel 序列。每个 panel $p$
分别记录逻辑行 $m_l(p)$、实际计算行 $m_c(p)$、packed-A 占用行 $m_p(p)$ 和
实际 store 行 $m_s(p)$。例如逻辑 M1 复用 M2 主体：

$$
(m_l,m_c,m_p,m_s)=(1,2,8,1).
$$

逻辑尾部 9--11 统一 pad 到 M12。设 SVE vector 为 $V$ bytes，BF16 N tile
$\nu=V/2$，某 stage 的 N tile 数 $q=N/\nu$。由当前 asm 可直接得到每个 panel
的关键指令数：

$$
I_{\mathrm{BFMMLA}}(p)=\frac{m_c(p)Kq}{2},
\qquad
I_A(p)=\frac{m_c(p)Kq}{8},
\qquad
I_B(p)=Kq.
$$

其中每条 BFMMLA 完成 $2V$ FLOPs，A load 每次读取 16 bytes，B load 每次
读取 $V$ bytes。因此 L1 流量为：

$$
Q_{L1,A}=16\sum_p I_A(p),
\qquad
Q_{L1,B}=V\sum_p I_B(p).
$$

设 W13/W2 顺序 N range 集合为 $\mathcal C$，第 $j$ 个 range 包含 $q_j$
个 tile。当前保守的 shared-cache 口径为：

$$
Q_{\mathrm{shared},A}
=2K\sum_p m_c(p)\sum_{j\in\mathcal C}\min(t,q_j),
$$

$$
Q_{\mathrm{shared},B}=2KNP,
$$

其中 $P=|\mathcal P(R)|$。N owners 合起来仍只覆盖完整 B 一次，但每个活跃
owner/range 至少从 shared cache 取得一次 A panel；后续 N tile 对同一 A 的
访问只计入 L1 层。若 PMU 证明 A 跨 range 保留在 private cache，则只修正
$Q_{\mathrm{shared},A}$，asm 指令计数不变。

对每个 GEMM stage，ECM shadow 公式为：

$$
T_{\mathrm{body}}
=\max\left(
T_{\mathrm{BFMMLA}},
T_{\mathrm{frontend}},
T_{L1\text{-load}}+T_{\mathrm{private}}+T_{\mathrm{shared}}
\right),
$$

$$
T_{\mathrm{stage}}
=T_{\mathrm{fixed}}+cT_{\mathrm{range}}
+T_{\mathrm{body}}+T_{\mathrm{epilogue}}.
$$

各项分别由独立测得的 BFMMLA、frontend/load、private/shared cache 和
epilogue service rate 提供。N tile 不能整除线程数时，模型使用 busiest
thread work 乘 active thread 数形成 balanced-equivalent work，从而保留
N-split load imbalance。

单独观察一个 stage latency 仍只能得到同一时间的等价 required rates：

$$
P_{\mathrm{req}}=F/\Delta T,
\qquad
B_{L1,\mathrm{req}}=Q_{L1}/\Delta T,
\qquad
B_{\mathrm{shared},\mathrm{req}}=Q_{\mathrm{shared}}/\Delta T.
$$

这些不是相互独立的硬件 ceiling。纯 BFMMLA、各 cache 层和 epilogue 必须由
独立 microbench/PMU 校准。`tiso_roofline.py` 保留为简化 baseline，
`gemm_ecm.py` 是新的分层 shadow；二者均不替换 planner active cost。

W13 split 不改变 BFMMLA、L1 A/B、aggregate B 或 C；它改变瞬时 weight
工作集、range-launch overhead，并在上述保守口径下增加 shared-A scan。
split/no-split 因此仍是不同的校准域。完整公式和 V3 留出验证见
`cost_model/GEMM_ECM_VALIDATION.md`。

### 8.3 Split-W13 owner-cache 工作集 band

当前 SVE split 路径将 W13 分为两个相等 N range。对 BF16 的 $H,F$，每个
W13 range 与 W2 的 packed weight stage 都是：

$$
S=2HF\quad\text{bytes}.
$$

N-split 中每个 worker 持有互不重叠的 packed-B column slice，因此决定首次
容量拐点的是 owner-private cache，而不是只看 shared LLC。设可用核心数为
$C$，每核 private L2 为 $L_2$、组相联路数为 $A$，为 packed-A、store、
prefetch 与 replacement headroom 预留 $r$ 个 way，则保守 owner budget 为：

$$
C_{\mathrm{owner}}=CL_2\frac{A-r}{A}.
$$

使用严格容量不等式 $nS<C_{\mathrm{owner}}$，可得上界：

$$
n_{\max}=\left\lceil\frac{C_{\mathrm{owner}}}{S}\right\rceil-1.
$$

无 GEMM 的 owner-slice scan 使用饱和曲线：

$$
B_{\mathrm{owner}}(n)
=B_{\max}\left(1-e^{-n/n_s}\right).
$$

若要求达到峰值比例 $u$，并发 stream 下界为：

$$
n_{\min}=\left\lceil-n_s\ln(1-u)\right\rceil.
$$

因此可解释的工作集候选 band 是
$n\in[n_{\min},n_{\max}]$。对 band 内 shape $sigma$，令完整 measurement
window 中 lane $l$ 有 $k_l$ 个 expert，使用：

$$
T_{\mathrm{iso-call}}(R,\sigma)
=\max_l k_lT_{\mathrm{iso}}(R,t_l).
$$

先保留不超过 band 内最佳 isolated makespan $(1+\epsilon)$ 的 shape，再选
工作集最小者，以便在算力近似相同时保留 cache headroom。该规则目前只对
physical M12 panel 数不少于 16 的长 route 生效，并且仍是 shadow validation，
尚未改变 production planner 的 shape 可行域。完整实验见
`cost_model/WORKING_SET_MODEL_VALIDATION.md`。

这是 **scheduling with interference** 扩展，经典固定 processing-time 的近似
保证不再直接成立。推荐的求解分层是：

1. 使用硬 CPU/LLC 约束的 CP-SAT 生成离线 oracle 或候选；
2. 使用 event simulator 对少量候选重新评分；
3. 在线 planner 使用经 oracle 验证过的剪枝和低开销启发式。

## 9. 剪枝验证

令未剪枝 oracle 的最优值为 $C^*$，求解器下界为 $LB$，剪枝 planner 的可行
结果为 $C_{\mathrm{pruned}}$。则：

$$
0\le
\frac{C_{\mathrm{pruned}}-C^*}{C^*}
\le
\frac{C_{\mathrm{pruned}}-LB}{LB}.
$$

每一层剪枝都应分别报告：

- plan-space 缩减量；
- planning latency；
- 相对 oracle 的实际 regret；
- oracle 未证明最优时，由 $LB$ 得到的保守 regret 上界；
- 在真实 event simulator/runtime 上的 held-out regret。

### 9.1 Amazon 192-core NUMA0 扩展校准

2026-07-14 在 Neoverse-V3 NUMA0 的 CPU `0-95` 上生成 TP4、TP2、EP4、EP2
的 split/no-split schema-v2 profile。每个 profile 包含 117 个 isolated 点
（13 个 route、9 个线程宽度）和 120 个 homogeneous contention 点，使用 5 次
warmup、20 次正式采样，并保留原始 samples。所有 profile 通过 schema、kernel
identity、split-pair grid 和 `ContentionCostModel` 加载验证。

测量证明 32T 以上的 isolated scaling 与 expert shape 相关：TP4 split 的短
route 常在 32--64T 饱和，而 EP4 长 route 可继续受益到 96T。64-local-expert 的
TP profile 在“每个 expert 都有 2040 routes”的 full-call anchor 上出现稳定的
大容量 cliff；20 个样本的离散小于约 3%，因此保留原始值，但在真实 routing
held-out 验证前不得据此单点扩大 planner 线程宽度集合。

使用 route 96/384/1536 作为 holdout 时，8 个 profile 的公式中位绝对误差为
0.53%--2.10%；P90 在 split profile 上不超过约 7.3%，但 EP/no-split 可达到
约 11%--12%。因此 exact table 与小 M residual 仍是 active profile 的必要部分，
不能仅凭 USL 主公式替代高线程尾部校正。

### 9.2 Split-W13 owner-cache 留出验证

Neoverse-V3 NUMA0 有 96 核、每核 2 MiB 8-way L2。EP2 split stage 为
16 MiB；预留 2 way 后 $C_{\mathrm{owner}}=144$ MiB。独立 owner-slice scan
拟合得到 $B_{\max}=8.991$ TB/s、$n_s=0.85$，取 $u=95\%$ 后预测 band 为
3--8 experts，即 48--128 MiB。resident scan 拟合误差 median 1.1%、max 7.4%。

route 1020/2040 fused holdout 经 $T_{\mathrm{iso}}$ 5% headroom 后均推荐
4 experts / 64 MiB，真实 regret 分别为 0.33% 和 0.00%；128 MiB 仍分别只慢
3.0% 和 1.0%，而 160 MiB 已同时下降约 18%。现有 route 192/768/2040
profile 上推荐 shape 的 regret 为 0.00%/0.79%/1.57%。route 48 regret 为
17.27%，因此短 route 明确保留 empirical contention table，不进入该 band
规则的有效域。

跨机器留出使用 8-core Neoverse-V1、每核 1 MiB 8-way L2 和 TP4 的 4 MiB
split stage。6/8-way owner budget 为 6 MiB，因此公式退化为 1 expert / 4 MiB；
route 1020/2040 实测最优也均为 4 MiB，regret 都是 0.00%。独立 scan 从 4 MiB
的 784 GB/s 降至 16 MiB 的 266 GB/s，说明 32 MiB shared L3 并不能替代
N-split owner-private residency。该交叉验证只支持 split 路径；no-split 不在
本公式和验证范围内。

## 10. 同步规则

发生以下任一变化时，必须同步更新本文档：

- job 粒度从 expert 改为 phase，或反向合并；
- 线程宽度定义、最大线程数或可行宽度集合变化；
- 是否允许抢占、迁移、动态调整线程数或主动 idling；
- CPU、LLC、L3/DRAM 带宽等资源约束变化；
- rank-local 与跨 rank 资源耦合方式变化；
- 目标函数加入或移除 planning、dispatch、communication、scatter、combine；
- kernel variant 从全局选择改为 per-expert 选择；
- shape、assignment、ordering 或 cache 相关剪枝变化；
- CP-SAT/MILP 编码改变原始可行域，而不只是等价重写。

同步修改至少应包括：

1. 本文档的公式和剪枝表；
2. 对应 planner/cost-model 测试；
3. oracle/regret 或性能验证结果；
4. 下方变更记录。

## 11. 变更记录

| 日期 | 版本 | 变更 |
| --- | --- | --- |
| 2026-07-13 | v0.1 | 建立未剪枝通用问题；将线程宽度定义为 $1\ldots T_{\max}$；区分 kernel variant、求解器编码和当前工程剪枝。 |
| 2026-07-14 | v0.2 | schema-v2 的 isolated processing time 改为 $O(t)+C(R)\phi_{\mathrm{USL}}(t)k_\phi(t)$；保留小 M residual 和二维 table 回归模式。 |
| 2026-07-14 | v0.3 | 将资源需求记号明确为 $d_{i,r}(t)$；补充下标含义、量纲一致性及总流量与瞬时 cumulative demand 的区别。 |
| 2026-07-14 | v0.4 | 明确 $\mathcal E$ 表示输入中的强制无环数据依赖，而不是 planner 选择的 lane 执行顺序；完整 expert 粒度下取空集。 |
| 2026-07-14 | v0.5 | 明确 $a_i(\tau)$ 是由执行区间推导的资源占用指示函数，并说明非抢占连续区间语义及 CP-SAT 等价编码。 |
| 2026-07-14 | v0.6 | 解释 cumulative resource 是同时活跃需求之和与可释放的 renewable capacity；区分 LLC 硬容量和带宽软争用。 |
| 2026-07-14 | v0.7 | 将问题准确描述为混合离散-连续非凸优化；给出非凸反例、NP membership 验证和从 `3-PARTITION` 到判定版本的强 NP-completeness 归约。 |
| 2026-07-14 | v0.8 | 严格区分变量与其可行域，并用整数线程宽度的凸组合反例说明离散域非凸。 |
| 2026-07-14 | v0.9 | 增加可解释 `T_iso` roofline shadow model：按 M12/tail panel 定义 W13/W2 FLOP 与 N-split 流量，并明确单一延迟样本无法同时识别算力和 L3 带宽 ceiling。 |
| 2026-07-14 | v0.10 | 将 GEMM shadow 扩展为 microkernel-aware ECM：区分逻辑/计算/packed/store 行，按 asm 统计 BFMMLA 与 A/B loads，分离 L1/private/shared-cache 和 epilogue，并记录 V3 长 route 留出验证。 |
| 2026-07-14 | v0.11 | 在 192-core Neoverse-V3 NUMA0 上生成 TP4/TP2/EP4/EP2 split/no-split 的 96T schema-v2 校准；保持 active $T_{\mathrm{iso}}$ 公式和 production 线程剪枝不变，并记录 64-expert route-2040 full-call cliff。 |
| 2026-07-15 | v0.12 | 增加 split-W13 owner-cache 工作集 shadow：由独立 stream 饱和度给出并发下界，由 N-split 每核 L2 way budget 给出容量上界，再用 $T_{\mathrm{iso}}$ headroom 选择最小稳健工作集；记录 V3 PMU、route 1020/2040 留出结果及短 route 失效边界。 |
