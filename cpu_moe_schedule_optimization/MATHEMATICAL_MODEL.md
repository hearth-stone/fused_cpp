# CPU MoE 调度数学模型

> 状态：调度问题定义的 source of truth。
>
> 最后更新：2026-07-26。
>
> 修改 planner 的决策变量、目标函数、硬约束、性能响应、线程宽度集合、调度语义、
> rank 耦合方式或剪枝策略时，必须同步更新本文档及末尾变更记录。

## 1. 文档边界

本文档将 MoE 调度分为四层：

1. **原始数学问题**：由 expert route tasks、CPU 容量、真实 isolated time 和
   contention slowdown 定义完整可行域与目标，不包含当前实现细节或经验剪枝。
2. **性能近似**：cost model 估计原始问题中的真实 isolated time 和 contention
   slowdown。
3. **求解器编码**：CP-SAT、MILP、event simulation 或启发式对同一个问题的
   不同近似与表达。
4. **工程剪枝**：线程宽度、静态 shape、LPT、non-idling 等当前限制。

M12、SVE、W13 split、具体权重布局和 cache 工作集公式不属于核心问题定义。
它们只影响固定执行环境下观测到的 isolated time 和 contention slowdown。

## 2. 原始问题

### 2.1 Expert route tasks

设一次完整 MoE layer 有 $E$ 个候选 expert：

$$
\mathcal X=\{1,\ldots,E\}.
$$

当前目标模型中 $E=256$。以下公式描述一个调度域：node-level 调度时该域可包含
全部 256 个 expert；rank-local 调度时，$\mathcal X$ 替换为分配到该 rank 的
expert 子集。复杂度分析将 $E$ 视为可变输入规模；固定的单个 256-expert 实例
本身不构成渐近复杂度问题。

若输入是由 $T$ 个 token、无重复 expert 的 TopK router 产生的全局 histogram，
则还满足：

$$
\sum_{i=1}^{E}M_i=T K,\qquad
0\le M_i\le T,\qquad
|\{i\mid M_i>0\}|\ge K.
$$

rank-local histogram 只包含 dispatch 到该 rank 的 routes，因此其 route 总和
不必等于 $TK$；但每个 local expert 仍满足 $M_i\le T$。下文核心调度问题同时
允许全局和 rank-local 输入，不把 $TK$ 等式作为 planner API 的额外硬约束。

路由完成后，expert $i$ 得到 route tensor：

$$
\mathbf X_i\in\mathbb R^{M_i\times H},
$$

其中 $M_i$ 是 route 数。需要执行的 job 集合为：

$$
\mathcal J=\{i\in\mathcal X\mid M_i>0\}.
$$

Job $i$ 表示在其独立权重上完成整个 expert 计算：

$$
\mathrm{W13}_i
\rightarrow\mathrm{activation}_i
\rightarrow\mathrm{W2}_i.
$$

不同 expert job 之间没有数据依赖。route tensor 的数值和 token index 不影响
rank-local expert compute 调度；在 expert shape 固定时，调度只需要 expert
identity 和 $M_i$。若 scatter/combine 也进入目标，token index 才需要进入外层
系统模型。

### 2.2 CPU 容量与真实执行响应

可用的最大 CPU 核心/线程数为：

$$
T_{\max}=C.
$$

原始问题允许每个 job 使用任意整数线程宽度：

$$
\mathcal T_i^{(0)}=\{1,2,\ldots,C\}.
$$

固定目标机器及其执行环境，但不展开 ISA、kernel、layout 或 cache 细节。定义
真实 isolated time：

$$
0<I_i(t)<\infty,
$$

表示 job $i$ 使用 $t$ 个线程且没有其他 expert 干扰时，完成整个 expert 计算
所需的时间。相同 expert shape 下可写成：

$$
I_i(t)=I(M_i,t).
$$

对时刻 $\tau$ 的活跃配置，定义：

$$
\mathcal Z(\tau)
=\{(j,M_j,t_j)\mid j\text{ 在 }\tau\text{ 时刻活跃}\}.
$$

真实 contention slowdown factor 为：

$$
1\le D_i(\mathcal Z) = 1+\rho_i(\mathcal Z)<\infty,
$$

其中 $\rho_i$ 是相对 isolated time 的延长比例，并满足单独运行时：

$$
D_i(\{(i,M_i,t_i)\})=1.
$$

只按并发 expert 数建模是特例
$D_i(\mathcal Z)=D(M_i,t_i,|\mathcal Z|)$。通用定义保留完整
$(M_j,t_j)$ 活跃配置，因为相同并发数下，不同 route 和线程宽度可以产生不同
争用。

基础模型假设 $D_i$ 对 normalized progress 做乘性缩放，并且只依赖当前
$\mathcal Z(\tau)$，即相同 active configuration 的瞬时速率与到达历史无关。
若验证发现 W13/W2 phase、cache residency 或先前执行顺序仍会改变速率，应将
这些状态加入 $\mathcal Z$，而不是修改调度目标。

因此，在 expert shape 和固定执行环境已确定时，原始问题的最小充分输入为：

$$
\boxed{
\mathcal I
=\left(
\{(i,\mathbf X_i,M_i)\}_{i=1}^{E},
C,
\{I_i(t)\},
\{D_i(\mathcal Z)\}
\right).
}
$$

route tasks 与 $C$ 定义工作负载和硬可行域；$I$ 与 $D$ 定义目标机器上的真实
执行时间。后续 cost model 只负责近似 $I,D$，planner 决定线程宽度和开始时间。

### 2.3 决策变量与执行动态

每个 expert 选择一个线程宽度和开始时间：

$$
t_i\in\{1,\ldots,C\},\qquad s_i\ge0.
$$

线程数在 job 开始前选择，执行过程中保持不变。因此 job 是 **moldable**，
不是执行中可改变线程数的 malleable job。Job 一旦开始就连续执行至完成，不能
抢占。

定义活跃指示函数：

$$
a_i(\tau)=\mathbf 1[s_i\le\tau<f_i].
$$

令 $x_i(\tau)\in[0,1]$ 表示归一化完成进度。执行期间的真实速率为：

$$
\frac{dx_i(\tau)}{d\tau}
=\frac{a_i(\tau)}{I_i(t_i)D_i(\mathcal Z(\tau))},
\qquad x_i(s_i)=0.
$$

完成时间由下式隐式定义：

$$
f_i=\inf\{\tau\ge s_i\mid x_i(\tau)\ge1\},
$$

等价地：

$$
\int_{s_i}^{f_i}
\frac{d\tau}{I_i(t_i)D_i(\mathcal Z(\tau))}=1.
$$

当某个 expert 完成时，$\mathcal Z(\tau)$ 立即变化，剩余 expert 使用新的
slowdown 继续推进。若所有 $D_i\equiv1$，则退化为固定处理时间：

$$
f_i=s_i+I_i(t_i).
$$

### 2.4 硬约束与目标函数

核心原始问题只把 CPU 线程容量作为硬件可行性约束：

$$
\sum_{i\in\mathcal J}t_i a_i(\tau)\le C,
\qquad\forall\tau.
$$

LLC 容量和内存带宽在当前目标机器上是性能资源，而不是可执行性资源。工作集
超过某个 cache 比例时任务仍可运行，只是通过 $D_i(\mathcal Z)$ 变慢，因此不把
LLC 或总访存字节数写成原始问题的硬 cumulative constraint。若未来存在真正
不可违反的内存容量、affinity 或安全隔离限制，可额外扩展可行域。

调度域内的 expert compute makespan 为：

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
& t_i\in\{1,\ldots,C\}, &&\forall i,\\
& s_i\ge0, &&\forall i,\\
& \sum_i t_i a_i(\tau)\le C, &&\forall\tau,\\
& \int_{s_i}^{f_i}
\frac{d\tau}{I_i(t_i)D_i(\mathcal Z(\tau))}=1,
&&\forall i,\\
& C_{\max}=\max_i f_i. &&
\end{aligned}}
$$

在运行时比较不同 planner 时，还需要计入 planning latency：

$$
T_{\mathrm{runtime}}=T_{\mathrm{plan}}+C_{\max}.
$$

Dispatch、communication、scatter 和 combine 在不受 plan 影响时作为外部成本；
若其成本会随 plan 改变，则必须提升为目标函数的一部分。

### 2.5 可选的 ready-token combine 外层

当 combine 与 expert 尾部重叠时，token identity 会影响可行开始时间。令输入
token 集合为 $\mathcal Q$，token $q$ 的 TopK expert 前驱为：

$$
\mathcal P_q=\{e_{q,1},\ldots,e_{q,K}\}.
$$

完整 expert job 在 $f_i$ 时一次性发布其所有 route row，因此 token merge job
$q$ 的 release time 为：

$$
r_q=\max_{i\in\mathcal P_q}f_i.
$$

令 $u_q$、$g_q$ 分别为 merge 的开始和完成时间，$b_q(\tau)$ 为其一线程活跃
指示函数，则：

$$
u_q\ge r_q,
\qquad
\sum_i t_i a_i(\tau)+\sum_q b_q(\tau)\le C.
$$

令 $G_q=G(K,H)>0$ 为 token merge 的单线程 isolated time，$y_q$ 为其归一化
进度。对包含活跃 merge job 的扩展状态 $\mathcal Z^+(\tau)$，有：

$$
\frac{dy_q(\tau)}{d\tau}
=\frac{b_q(\tau)}{G_qD_q^{\mathrm{merge}}(\mathcal Z^+(\tau))},
\qquad
y_q(u_q)=0,quad y_q(g_q)=1.
$$

若 merge 与剩余 expert 同时运行会改变 cache/带宽响应，还应把 expert slowdown
扩为 $D_i^+(\mathcal Z^+)$；不能直接假定被重叠的 merge 时间全部免费。包含
combine 的目标变为：

$$
C_{\max}^{+}=\max\left(\max_i f_i,\max_q g_q\right).
$$

当前 production planner 仍只优化 2.4 节的 expert-compute makespan。async
ready-token executor 不扩大 planner 决策空间：它保持固定 expert core
interval，优先执行可运行 expert，仅让无 expert 可执行的 lane 贪心领取一个已经
release 的 token；所有 expert 结束后，未完成 token 仍由原连续区间 merge 收尾。
该策略默认只在 SVE FP32 direct-route 路径且
$\max_i\lceil M_i/12\rceil/t_i\ge1.25\min_i\lceil M_i/12\rceil/t_i$ 时生效；
`FUSED_CPP_MOE_ASYNC_READY_TOKEN_MERGE=0` 显式恢复 post-expert merge。因此它是
默认 executor heuristic，不是 cost model 已评分的 planner 调度动作。

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

若多个 rank 共享内存带宽、cache 或互连，则 contention response 必须观察所有
rank 的联合活跃配置 $\mathcal Z_{\mathrm{global}}(\tau)$。一个 rank 完成后，
联合活跃配置随之改变，剩余 rank 必须切换到新的 slowdown，而不能将 dual-rank
速率应用到整个执行区间。

## 4. 问题类型与复杂度

该问题是 **带 active-set interference 的非抢占 moldable-job makespan
调度**。可以用非标准的描述性记号写成：

$$
P\mid\mathrm{moldable,\ nonpreemptive,\ interference}\mid C_{\max}.
$$

这里 `interference` 表示处理速率依赖同时活跃的 $(M_j,t_j)$ 配置，不是标准
Graham 三字段记号中的既有选项。当所有 $D_i\equiv1$ 时，问题退化为固定处理
时间的 moldable-job scheduling 特例。

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

### 4.2 最优解存在性

**命题 1（最优调度存在）**。设 $n=|\mathcal J|<\infty$、$C<\infty$，并且
$I_i(t)$ 与 $D_i(\mathcal Z)$ 满足第 2.2 节的有限性和 memoryless 假设。那么
存在一个可行调度 $S^*$，使：

$$
C_{\max}(S^*)
=\min_{S\in\mathcal F}C_{\max}(S).
$$

这里 $\mathcal F$ 是第 2 节定义的完整原始可行域，包含连续开始时间、任意合法
整数线程宽度和主动 idling。

**证明。**

首先构造一个有限的可行上界。令所有 expert 使用一个线程并任意串行执行。任意
时刻只有一个 expert 活跃，所以 $D_i=1$，该调度的 makespan 为：

$$
B=\sum_{i\in\mathcal J}I_i(1)<\infty.
$$

因此 $\mathcal F$ 非空，且最优值的下确界满足：

$$
\inf_{S\in\mathcal F}C_{\max}(S)\le B.
$$

任何 $C_{\max}>B$ 的调度都不会优于上述串行调度，所以只需考虑所有开始与完成
事件都位于 $[0,B]$ 的调度。

接下来固定线程宽度向量：

$$
\mathbf t=(t_1,\ldots,t_n)
\in\{1,\ldots,C\}^n.
$$

这种向量至多有 $C^n$ 个。每个 expert 有一个开始事件 $S_i$ 和一个完成事件
$F_i$。固定一个满足 $S_i$ 位于 $F_i$ 之前的合法事件顺序：

$$
\pi=(\pi_1,\ldots,\pi_{2n}).
$$

合法事件顺序数量有限，且不超过 $(2n)!$。令对应事件时间为：

$$
0=e_0\le e_1\le\cdots\le e_{2n}\le B.
$$

使用非严格不等式允许多个事件同时发生；任意 tied events 都可以按任意顺序
线性化，它们之间的零长度区间不产生执行进度。

对 $k=0,\ldots,2n-1$，令 $\mathcal A_k$ 表示事件顺序中前 $k$ 个事件发生后
仍然活跃的 expert 集合。区间 $[e_k,e_{k+1})$ 上的配置固定为：

$$
\mathcal Z_k
=\{(i,M_i,t_i)\mid i\in\mathcal A_k\}.
$$

该区间的 CPU 约束为：

$$
\sum_{i\in\mathcal A_k}t_i\le C.
$$

expert $i$ 的完成约束为：

$$
\sum_{k:\,i\in\mathcal A_k}
\frac{e_{k+1}-e_k}
{I_i(t_i)D_i(\mathcal Z_k)}
=1.
$$

固定 $\mathbf t$ 和 $\pi$ 后，$I_i(t_i)$、$D_i(\mathcal Z_k)$ 以及每个
$\mathcal A_k$ 都是常数。因此上述完成约束关于事件时间
$(e_1,\ldots,e_{2n})$ 是线性等式，事件顺序和时间范围是非严格线性不等式。
对应可行域 $P_{\mathbf t,\pi}$ 是 $[0,B]^{2n}$ 中闭集，因而闭且有界。根据
Heine--Borel 定理，$P_{\mathbf t,\pi}$ 是紧集。

在任意非空 $P_{\mathbf t,\pi}$ 上，最后一个事件必为某个完成事件，因此：

$$
C_{\max}=e_{2n}.
$$

这是事件时间的连续线性函数。根据 Weierstrass 极值定理，它在每个非空紧集
$P_{\mathbf t,\pi}$ 上都取得最小值。

最后，线程宽度向量和合法事件顺序都只有有限种。每个原始可行调度在
$C_{\max}\le B$ 时都对应至少一个 $(\mathbf t,\pi)$ 及其事件时间；同时发生的
事件由相等的 $e_k$ 表示。因此只需在有限个已取得的子问题最小值中再次取最小，
该最小值仍由某个具体调度 $S^*$ 取得。故：

$$
\exists S^*\in\mathcal F:\qquad
C_{\max}(S^*)
=\min_{S\in\mathcal F}C_{\max}(S).
\qquad\square
$$

该命题不是独立的 MoE 调度定理，而是 Heine--Borel 定理与 Weierstrass 极值定理
在当前有限事件模型上的推论。证明不要求 $I,D$ 具有解析公式；查表或 cost model
输出同样适用，只要它们对所有候选配置都有确定的有限正值。最优解存在不表示
能够在多项式时间内找到，也不表示 $(\widehat I,\widehat D)$ 上的最优解等于真实
$(I,D)$ 上的最优解。

### 4.3 固定 isolated-time 特例的判定版本

为证明困难度，只需限制到 $D_i\equiv1$。给定整数或经统一缩放后的有理数
$I_i(t)$ 以及 deadline $L$，定义：

$$
\textsc{Isolated-Moldable-Schedule}(L):
\quad
\text{是否存在可行 }\{s_i,t_i\}\text{ 使 }C_{\max}\le L\text{？}
$$

该判定版本属于 NP。证书包含每个 job 的线程宽度和开始时间。固定宽度后，
duration 为 $I_i(t_i)$；将所有开始和结束事件排序即可在多项式时间内验证 CPU
容量和 deadline。

对一般 active-set slowdown，是否能声称其判定版本属于 NP 取决于
$D_i(\mathcal Z)$ 的输入表示和求值复杂度。本文只用 $D_i\equiv1$ 的有限有理
特例证明强 NP-hardness，不对任意黑盒 slowdown 做更强的复杂度类别声明。

### 4.4 由 3-PARTITION 归约

给定一个 `3-PARTITION` 实例：$3m$ 个正整数 $a_1,\ldots,a_{3m}$ 和整数 $B$，
满足：

$$
\frac{B}{4}<a_i<\frac{B}{2},
\qquad
\sum_{i=1}^{3m}a_i=mB.
$$

构造 $E=3m$ 个 expert task，并令：

$$
C=m,\qquad D_i(\mathcal Z)\equiv1,
$$

$$
M_i=a_i,\qquad I(M_i,1)=M_i=a_i,
\qquad
I(M_i,t)=B+1\quad(2\le t\le m),
$$

deadline 为 $L=B$。由于任意 $t\ge2$ 都有 $I_i(t)>L$，所有 deadline 内的
可行调度都必须选择：

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

- 固定 isolated-time 特例的判定版本是强 NP-complete；
- 原始 contention-aware makespan 优化问题是强 NP-hard；
- active-set slowdown、可变线程宽度和更丰富的执行响应不会消除该困难度。

## 5. Kernel Implementation Variant

Split/no-split 以及未来其他 kernel 版本不属于核心问题变量。固定执行环境中的
一个全局 implementation variant $v\in\mathcal V$ 会诱导不同的真实响应：

$$
\Theta_v
=\{I_i^{(v)}(t),D_i^{(v)}(\mathcal Z)\}.
$$

对每个 variant 求解同一个调度问题：

$$
C^*(v)=\min_{s,t}C_{\max}(s,t;\Theta_v),
$$

再进行外层选择：

$$
C^*=\min_{v\in\mathcal V}C^*(v).
$$

因此 variant 只改变真实性能环境，不改变问题类型。若在所有相关线程宽度和
活跃配置上，variant $v_1$ 都满足：

$$
I_i^{(v_1)}(t)D_i^{(v_1)}(\mathcal Z)
\le
I_i^{(v_2)}(t)D_i^{(v_2)}(\mathcal Z),
$$

则 $v_2$ 被 $v_1$ 支配，可以在进入 planner 前删除。当前 split/no-split 比较
尚未证明这种全配置支配关系，因此仍保留为独立校准域。

SVE 静态 asm 与 Xbyak exact-M 也属于不同的 $v$。两者虽然共享 packed weight、
N-split 和 M12 主体，但尾部映射不同，因此 profile identity 必须同时包含
`sve_implementation` 和 `m_tail_policy`。当前 production 默认是
`jit/xbyak_exact_m`；`asm/static_bucketed` 只作为 fallback、A/B 对照和历史 profile
解释，不允许用后者的 $\Theta_v$ 直接预测前者。planner 的 `auto` 选择以
split/no-split pair 为原子：先要求完整的 JIT pair，缺任意一半时才整体回退到
完整 static pair，不能跨 variant 拼接 $\Theta_v$。

实现也可以提供不暴露给 planner 的确定性 per-expert 子变体策略。令全局 backend
$b$ 固定，而 expert $i$ 的微内核/loop-order 子变体由只依赖本次调用已知 shape 的
映射决定：

$$
v_i=\pi_b(M_i,H,F,\mu),
$$

其中 $\mu$ 是机器能力与 build/runtime feature 集合。此时响应写为
$I_i^{(b,v_i)}(t)$ 和 $D_i^{(b,v_i)}(\mathcal Z)$。只要 $\pi_b$ 是确定性的、不会
改变线程宽度、依赖或资源可行性，它不扩展 planner 的决策空间，只把 backend
$b$ 的响应函数改成分段函数；profile identity 和验证仍必须记录该 policy 版本。

当前 x86 BF16 `auto` 是这种策略，但映射已扩展为 dimension-aware policy。调用级
ISA 与有效线程数由 $(H,F,\{M_i\},\mu)$ 决定，AMX 内每个 expert 的 M pattern
与 cache window 再由 $(M_i,H,F,\mu)$ 决定；team-N/skew-wave 的 target row 也按
$HF$ 缩放。环境变量只保留为强制实验/回归 override，不是 production 调用的
必要输入。

## 6. 求解器编码

### 6.1 数学定义

核心定义直接使用整数变量 $t_i$。这保持问题表达简洁，不绑定具体求解器。

### 6.2 CP-SAT

当 $D_i\equiv1$ 时，CP-SAT 可以为每个 $(i,t)$ 建立 duration 为 $I_i(t)$ 的
optional interval，并要求每个 expert 恰好选择一个线程宽度；CPU 使用
cumulative constraint。此编码可作为 isolated fixed-duration 特例的离线
oracle，并返回可行上界和理论下界。

由于原始问题规定 $D_i(\mathcal Z)\ge1$，该 isolated 特例的最优 makespan
不大于真实 contention-aware 最优值，因此可以作为有效下界。

一般 $D_i(\mathcal Z)$ 会使 job duration 随执行中的 active set 改变，不能直接
编码成一个固定 duration interval。要得到 contention-aware exact oracle，必须
进一步枚举并发 group mode、离散化时间/状态，或使用专门的 event-based search；
普通 CP-SAT interval 模型本身不是完整原始问题的等价编码。

### 6.3 MILP

标准 MILP 通常将 $t_i$ 展开为 one-hot 变量：

$$
y_{it}\in\{0,1\},\qquad
\sum_{t=1}^{T_{\max}}y_{it}=1.
$$

然后在线性 fixed-duration 特例中表示：

$$
t_i=\sum_t t\,y_{it},\qquad
I_i=\sum_t I_i(t)y_{it}.
$$

One-hot 是线程宽度的等价求解器编码，不是剪枝。一般 active-set slowdown 会
引入状态相关的非线性处理速率，需要额外变量或分段/枚举近似。

## 7. 当前实现相对原始问题的剪枝

| 层级 | 原始可行域 | 当前限制 | 性质 |
| --- | --- | --- | --- |
| 线程宽度 | $1,2,\ldots,T_{\max}$ | `1,2,4,8,16,32` | 离散宽度剪枝 |
| 并发配置 | 活跃 job 可形成任意满足 CPU 容量的 $(M_i,t_i)$ 组合 | 整次调用使用静态 core shape | static-partition 剪枝 |
| Shape 集合 | 所有满足 CPU 容量的整数宽度组合 | profile 支持并经 active 工作集规则筛选的 shape；owner-cache band 当前仅 shadow | 候选剪枝 |
| Assignment | 任意 expert-to-resource 调度 | 按 isolated cost 的 LPT | 启发式分配 |
| x86 synchronous executor team mapping | expert 可取任意合法整数宽度并形成任意 wave | API 接受 1--256 workers；`x86_bf16_auto` 对很小 aggregate work 可收缩为 1 worker；均衡 route 用 atomic expert queue，active expert 不足时按 route/当前宽度贪心组 team，强偏斜时按 dimension-scaled row target 形成有序 wave | planner 外的确定性 runtime mapper |
| Ordering | 任意可行开始时间和顺序 | 每个 lane 的 LPT 顺序 | 顺序剪枝 |
| Idling | 允许主动等待以避开争用 | lane 可启动下一个 expert 时立即启动；ready-token 路径仅填充无可运行 expert 的空闲 lane | non-idling 剪枝 |
| Workload 输入 | 任意合法 global 或 rank-local route histogram | planner 接受任意 histogram；catalog preset 只扩展验证覆盖，不过滤运行时输入 | 不剪枝 |
| Route combine | 任意满足 TopK release 约束和 CPU 容量的 merge 排程 | planner 不搜索 combine；默认 executor 采用 expert-first 单 token 贪心和连续收尾 | 外层启发式限制 |
| Kernel variant | 任意未被支配的实现 | ARM `auto` 先要求完整 `jit/xbyak_exact_m` legacy split/no-split pair，再加入同 identity 的实测 global packed-B byte-window variants；缺少完整 JIT legacy pair 时整体回退 static pair；x86 `auto` 使用不进入 planner 的确定性 ISA/thread 与 per-expert AMX pattern/cache policy，AVX-512/AMX 共用 dimension-aware team-N/wave policy | 实例候选限制与 runtime policy |
| Isolated time | 真实 $I_i(t)$ | active $T_{\mathrm{iso}}$ 经验公式；分层 GEMM model 仅 shadow | cost 近似，不剪枝可行域 |
| Contention | 任意动态活跃配置上的真实 $D_i(\mathcal Z)$ | 实测 contention profile 与 stage-aware event simulator | cost 近似，不剪枝可行域 |

当前 `IntervalPlanner` 搜索的是上述剪枝后 plan space 中的方案，不是原始问题
的全局最优方案。

Amazon 192-core NUMA0 的 schema-v2 profile 已将实测线程域扩展到
`1,2,4,8,16,32,48,64,96`，但这只扩大 $\widehat I_i(t)$ 的校准域。当前 planner 的
线程宽度剪枝仍为表中所列的 `1,2,4,8,16,32`；在完成 48/64/96T 的 held-out
regret 验证前，不自动扩大在线决策空间。

## 8. 当前 Cost Model 的位置

原始问题中的 $I_i(t)$ 和 $D_i(\mathcal Z)$ 是目标机器上的真实但未知响应。
Cost model 分别提供估计：

$$
\widehat I_i(t)\approx I_i(t),
\qquad
\widehat D_i(\mathcal Z)\approx D_i(\mathcal Z).
$$

当前 stage-aware event simulator 使用这两个估计推进 job：

$$
\frac{dx_i(\tau)}{d\tau}
=
\frac{a_i(\tau)}
{\widehat I_i(t_i)\widehat D_i(\mathcal Z(\tau))},
\qquad
x_i(s_i)=0,\quad x_i(f_i)=1.
$$

因此 cost model 只近似目标函数中的真实性能响应，不改变由 route tasks 和 CPU
容量定义的原始可行域。

2.5 节的 ready-token merge 尚未进入 active cost model。当前 profile 的
$\widehat I_i,\widehat D_i$ 仍只描述 expert compute，combine 继续作为独立实测
外部项；在增加 merge service time、expert/merge 异构 contention 和真实分布
留出验证前，planner 不得把实验重叠时间当作确定收益。

### 8.1 当前 active isolated model

新生成且携带 `iso_formula` 的 schema-v2 profile 使用：

$$
\widehat I_i(t)
=T_{\mathrm{iso}}(M_i,t)
=O(t)+C(M_i)\phi_{\mathrm{USL}}(t)k_\phi(t),
$$

$$
O(t)=o_0+\frac{o_1}{t},\qquad
\phi_{\mathrm{USL}}(t)=\frac{1+\alpha(t-1)+\beta t(t-1)}{t}.
$$

$C(M)$ 是单线程 route-work 的一维实测校准曲线；$O(t)$ 和
$\phi_{\mathrm{USL}}(t)$ 分别表示固定开销与相对理想 $1/t$ 的线程效率基线，
$k_\phi(t)$ 是与 route 无关的一维实测校正。Xbyak 路径的 exact M1--M12 与前两个
完整 M12 panel 的启动/线程效率不同于 steady-state M12 bulk，因此保留每个
`(tail, threads)` 的实测 residual；更大的 M 使用公式，并按 kernel 组合规则
追加 exact remainder。历史 static-asm profile 继续按 M1/M2/M4/M8 bucket 解释，
两种 residual 不可混用。公式仅在 profile 的已校准线程域内有效，不承担跨机器
或更高线程数外推。没有序列化公式的历史 profile 继续使用旧二维 table，避免
离线重拟合改变既有 planner 基准；可显式指定 `iso_mode="formula"` 做 shadow 对比。

Amazon 192-core NUMA0 的 TP4/TP2/EP4/EP2 split/no-split profile 继续使用同一
公式，`thread_domain` 扩展到 `[1,96]`，并以 48/64/96T 实测点校正
$k_\phi(t)$；公式形式及小 M residual 组合规则没有改变。

### 8.2 实现弱相关的分层 GEMM cost model

简单的 $\max(F/P,Q/B)$ 无法表示当前 BF16 microkernel 的 L1 重复加载、
矩阵指令发射、不同 cache 层、tail padding 和 fused epilogue；把 M12、BFMMLA
等细节直接写入核心公式又会使 planner 绑定单个实现。因此 shadow model 使用
算法工作量、kernel 映射和机器响应三层契约。

对 stage $s$，设问题实例为 $x$，调度决策为 $\sigma_s$，kernel implementation
为 $\kappa$，目标机器为 $\mu$。完整分解为：

$$
w_s=\mathcal A_s(x),
$$

$$
d_s=\Phi_\kappa(w_s,\sigma_s),
$$

$$
\widehat T_s
=\Psi_{\mu,\kappa}(d_s,\sigma_s)
+O_{\mu,\kappa}(\sigma_s).
$$

$\mathcal A_s$ 只产生逻辑工作 `LogicalGemmWork`；$\Phi_\kappa$ 是可替换的
kernel mapper，产生统一 `KernelDemand`；$\Psi_{\mu,\kappa}$ 使用绑定
`machine_id + implementation_id + active_threads` 的实测
`MeasuredMachineProfile`。stage 预测组合成 $\widehat I_i(t)$，并可作为
$\widehat D_i(\mathcal Z)$ 的实现特征；物理 demand 留在 cost-model/profile
内部或用于候选剪枝。planner 的原始接口不需要知道 M12、SVE 或 BFMMLA。

#### 8.2.1 算法工作量

对 route 数 $M$、hidden size $H$ 和 intermediate size $F$，逻辑 GEMM 为：

$$
\mathrm{W13}:[M,H][H,2F],
\qquad F_{13}^{\mathrm{use}}=4MHF,
$$

$$
\mathrm{W2}:[M,F][F,H],
\qquad F_2^{\mathrm{use}}=2MHF.
$$

当前 BF16 输入/权重、BF16 fused intermediate、FP32 down store 的 compulsory
one-pass bytes 为：

$$
Q_{13}^{\min}=2MH+4HF+2MF,
$$

$$
Q_2^{\min}=2MF+2FH+4MH.
$$

给定机器理论上界 $P_\mu^{\mathrm{peak}}$ 和
$B_\mu^{\mathrm{peak}}$，算法层可以给出条件下界：

$$
T_s^{\mathrm{LB}}
=\max\left(
\frac{F_s^{\mathrm{use}}}{P_\mu^{\mathrm{peak}}},
\frac{Q_s^{\min}}{B_\mu^{\mathrm{peak}}}
\right).
$$

这些是算法可解释的有用 FLOPs 和至少读写一次所有逻辑 tensor 的下界，不是
实际 cache 流量；$T_s^{\mathrm{LB}}$ 也只是理论下界，不直接作为运行时间预测。
算法层不包含线程数、N tile、M panel、padding 或 split；因此不满足某个 kernel
对齐要求的逻辑 shape 仍是合法问题实例，只可能在特定 mapper lowering 时被拒绝。

#### 8.2.2 Kernel 执行映射

统一的实现需求向量写作：

$$
d_s=\left(
F_s^{\mathrm{exec}},I_s,
Q_{L1,s},Q_{\mathrm{private},s},Q_{\mathrm{shared},s},
E_s,W_s,N_{\mathrm{call},s},N_{\mathrm{range},s}
\right).
$$

其中包含实际执行 FLOPs、关键指令、各层流量、epilogue 元素、瞬时工作集、
调用/range 次数和 busiest-thread balanced work。M12 换成其他 tile、SVE 换成
NEON 或改变 tail 规则时，只替换 $\Phi_\kappa$，不修改算法公式或 planner。

当前 `SveBf16KernelProfile` 对应 implementation
`sve_bf16_xbyak_exact_m1_m12_v2`。令 $\mathcal P(M)$ 为物理 panel 序列，每个 panel
$p$ 分别记录逻辑行 $m_l(p)$、实际计算行 $m_c(p)$、packed-A 行 $m_p(p)$ 和
store 行 $m_s(p)$。写 $M=12b+r,0\le r<12$，前 $b$ 个 panel 均为
$(12,12,12,12)$；若 $r>0$，exact tail 为：

$$
(m_l,m_c,m_p,m_s)
=\left(r,2\left\lceil\frac r2\right\rceil,
\begin{cases}8,&r\le8\\12,&r>8\end{cases},r\right).
$$

例如 M1 为 $(1,2,8,1)$，M5 为 $(5,6,8,5)$，M9 为 $(9,10,12,9)$。
static-asm control profile 则仍将 3--4、5--8、9--11 分别映射到
M4、M8、M12 compute/store bucket。设 SVE vector 为 $V$ bytes，BF16 N tile
$\nu=V/2$，tile 数 $q=N/\nu$，则当前 JIT mapper 产生：

$$
I_{\mathrm{BFMMLA}}(p)=\frac{m_c(p)Kq}{2},
\qquad
I_A(p)=\frac{m_c(p)Kq}{8},
\qquad
I_B(p)=Kq,
$$

$$
Q_{L1,A}=16\sum_p I_A(p),
\qquad
Q_{L1,B}=V\sum_p I_B(p).
$$

若顺序 N range $j$ 有 $q_j$ 个 tile，当前实现的保守 shared-cache 映射为：

$$
Q_{\mathrm{shared},A}
=2K\sum_p m_c(p)\sum_j\min(t,q_j),
$$

$$
Q_{\mathrm{shared},B}=2KNP,
\qquad P=|\mathcal P(M)|.
$$

split/no-split 是传给 mapper 的通用 $\sigma_s$，不是算法语义或核心公式中的特殊
分支。当前 SVE mapper 中，split 不改变 BFMMLA、L1 A/B、aggregate B 或 C，
但改变 range 次数、瞬时 weight 工作集，并在保守口径下增加 shared-A scan。
Exact-M 相对 static bucket 不改变每个 panel 的 aggregate B load；它只减少被
padding 的 row pair 对应的 BFMMLA、A broadcast 和无效 epilogue/store 工作。

#### 8.2.3 实测机器响应

当前 ECM response 使用：

$$
T_{\mathrm{body}}
=\max\left(
\frac{F_s^{\mathrm{bal}}}{P_{\mu,\kappa}(t)},
\frac{I_s^{\mathrm{bal}}}{R_{\mu,\kappa}(t)},
\frac{Q_{L1,s}^{\mathrm{bal}}}{B_{L1,\mu,\kappa}(t)}
+\frac{Q_{\mathrm{private},s}}{B_{\mathrm{private},\mu,\kappa}(t)}
+\frac{Q_{\mathrm{shared},s}}{B_{\mathrm{shared},\mu,\kappa}(t)}
\right),
$$

$$
\widehat T_s
=O_{\mathrm{stage}}+N_{\mathrm{range},s}O_{\mathrm{range}}
+T_{\mathrm{body}}
+\frac{E_s^{\mathrm{bal}}}{R_{\mathrm{epi},\mu,\kappa}(t)}.
$$

所有 service rate 和固定开销必须由目标机器上该 implementation/thread width 的
独立 microbenchmark、PMU 或明确的 stage pair 测量得到。只观察一个 stage
latency 只能得到 $F/\Delta T$、$Q/\Delta T$ 等同一时间的等价 required rate，
不能同时识别计算、L1 和 shared-cache ceiling。

`gemm_cost_model.py` 实现通用契约和机器响应，`sve_bf16_kernel_model.py` 实现
当前 SVE mapper，`gemm_ecm.py` 保留原 CLI/API facade 与诊断报告。该路径仍是
shadow，不替换 active $T_{\mathrm{iso}}$。V3 steady-M12 留出误差低于 1.7%，
但只验证同 shape 的 panel 线性，尚未证明跨 shape/kernel/machine 的可迁移性；
完整限制见 `cost_model/GEMM_ECM_VALIDATION.md`。

### 8.3 Split-W13 owner-cache 工作集 band

当前 production 默认仍将 W13 分为两个相等 N range，而 W2 使用一个 range。
对 BF16 的 $H,F$，三个顺序 packed-weight stage 都是：

$$
S_0=2HF\quad\text{bytes}.
$$

SVE fused expert 另提供显式 packed-B byte-window variant。设 stage $s$ 的 packed
维度为 $(K_s,N_s)$，BF16 N tile 为 $\nu$，总 tile 数和单 tile 字节数为：

$$
q_s=\frac{N_s}{\nu},\qquad b_s=2K_s\nu.
$$

给定目标 $S_{\mathrm{target}}>0$，实现选择：

$$
u_s=\max\left(1,\left\lfloor\frac{S_{\mathrm{target}}}{b_s}\right\rfloor\right),
\qquad
r_s=\left\lceil\frac{q_s}{u_s}\right\rceil,
$$

再将完整 N tiles 均匀分配给 $r_s$ 个顺序 range。最大实际 range 为：

$$
\widehat S_s
=2K_s\nu\left\lceil\frac{q_s}{r_s}\right\rceil.
$$

当 $S_{\mathrm{target}}\ge b_s$ 时有
$\widehat S_s\le S_{\mathrm{target}}$；否则最小粒度是一个 N tile，实际 range
会大于目标。W13 使用 $(K,N)=(H,2F)$，W2 使用 $(F,H)$。例如 TP4
$H=4096,F=512,\nu=8$ 时，legacy 4 MiB stage 的 W13/W2 range 数为 $2/1$，
2 MiB 为 $4/2$，1 MiB 为 $8/4$。

每个 range 内仍对 N tiles 做现有线程切分，因此单 range 的有效线程上界为：

$$
t_{\mathrm{active},s}
\le
\min\left(t,\left\lceil\frac{q_s}{r_s}\right\rceil\right).
$$

更小窗口不改变 GEMM useful/executed FLOPs、aggregate BFMMLA 或 aggregate packed-B
读取，只增加 range dispatch，并在 8.2 的保守模型中增加 shared-A range scan。
相邻 range 之间没有 barrier；不同 worker 可以短暂处于相邻 range，因此
$\widehat S_s$ 是 loop-order 的 nominal active window，不是同步保证的硬 cache
residency 上界。W2 owner-scatter 必须按每个 range 复用相同的 discontiguous N
ownership，才能继续消除 W2-to-scatter barrier。

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
$n\in[n_{\min},n_{\max}]$。对 band 内 shape $\sigma$，令完整 measurement
window 中 lane $l$ 有 $k_l$ 个 expert，使用：

$$
T_{\mathrm{iso-call}}(M,\sigma)
=\max_l k_lT_{\mathrm{iso}}(M,t_l).
$$

先保留不超过 band 内最佳 isolated makespan $(1+\epsilon)$ 的 shape，再选
工作集最小者，以便在算力近似相同时保留 cache headroom。该规则目前只对
physical M12 panel 数不少于 16 的长 route 生效，并且仍是 shadow validation，
尚未改变 production planner 的 shape 可行域。完整实验见
`cost_model/WORKING_SET_MODEL_VALIDATION.md`。

owner-cache band 是对 $\widehat D_i(\mathcal Z)$ 和候选空间的实现相关近似，
不是 LLC 硬可行性约束。推荐的求解分层是：

1. 使用只有 CPU 容量、$D_i\equiv1$ 的 CP-SAT 生成 isolated lower bound 和候选；
2. 使用 owner-cache band 等规则缩小需要评估的 active configurations；
3. 使用 contention-aware event simulator 对候选重新评分；
4. 在线 planner 使用经真实 runtime regret 验证过的低开销启发式。

当前 active policy 将 legacy split 和 global byte-window 统一记为 kernel variant
$v$：

$$
v=(w13\_split,\ S_{\mathrm{target}},\ r_{13},\ r_2).
$$

$S_{\mathrm{target}}=0$ 时保留 legacy split/no-split 两个候选；
$S_{\mathrm{target}}>0$ 时使用 canonical `w13_split=false`，因为显式窗口已经覆盖
W13 和 W2，不能再与 split flag 形成重复候选。每个 profile 分别校准：

$$
\widehat I_i^{(v)}(t),\qquad
\widehat D_i^{(v)}(\mathcal Z).
$$

因此在线决策是联合选择：

$$
(v^*,\sigma^*)
=\arg\min_{v\in\mathcal V_{\mathrm{profiled}},\,
           \sigma\in\Sigma_v}
\widehat C(v,\sigma).
$$

其中 $\mathcal V_{\mathrm{profiled}}$ 只包含实际加载的 profile；禁止从 legacy
footprint 或相邻窗口外推时间。对 shape
$\sigma=(t_1,\ldots,t_L)$，若当前有 $L_a$ 个非空 lane，planner 的 nominal active
packed-B 诊断为：

$$
S_{\mathrm{active}}(v,\sigma)
=L_a\max(\widehat S_{13}^{(v)},\widehat S_2^{(v)}),
$$

每个 lane 的 tile-aligned owner-private stripe 诊断为：

$$
S_{\mathrm{owner},s}(v,t_l)
=b_s\left\lceil
\frac{\left\lceil q_s/r_s\right\rceil}{t_l}
\right\rceil.
$$

$S_{\mathrm{active}}$ 进入现有 LLC 候选剪枝和置信区间重叠时的 working-set
tie-break；owner stripe 当前作为结果诊断输出，在跨机器
$S_{\mathrm{worker,target}}$ 校准完成前不做新的硬剪枝。最终排序仍使用该
variant 自己的 isolated/contention 实测表。stage simulator 将 W13 的
$r_{13}$ 个 range 和 W2 的 $r_2$ 个 range 分别推进，不能把 windowed W2
当成 legacy 单阶段。

第一版一次 operator 调用只选择一个全局 $S_{\mathrm{target}}$。混合宽度 shape
中的所有 lane 共用该值；per-task 或 W13/W2 独立窗口尚不在决策空间。特别是
单个 M12 panel 不复用 B，额外 range 通常只有 dispatch 和 lane-width 代价，
不能仅按 cache 容量规则强制细分。

### 8.4 x86 dimension-aware ISA/thread、AMX pattern/cache 与 team-N policy

x86 runtime 不把下述 choice 加入 CPU MoE planner，而是使用带版本的确定性映射。
设 expert $i$ 的 route 数为 $M_i$，并定义：

$$
M_\Sigma=\sum_i M_i,\qquad M_{\max}=\max_i M_i,\qquad
W_\Sigma=M_\Sigma HF,\qquad W_{\max}=M_{\max}HF.
$$

当前 CPUID family 6、model 173 的 Amazon C8i 使用
`intel_06_ad_c8i_v1`；其他 x86 CPU 使用 `generic_v1`。`auto` 在 AMX 可用时
prepack backend ID 104 `x86_bf16_auto` 的 K32/N32 权重，同一份权重可由
AVX-512 或 AMX kernel 消费，不因调用级 ISA 决策维护第二份权重。C8i 的调用级
决策为：

$$
\operatorname{isa}=
\begin{cases}
\text{AVX-512 BF16}, & W_{\max}\le8192,\\
\text{AMX BF16}, & W_{\max}>8192,
\end{cases}
\qquad
T_{\mathrm{eff}}=
\begin{cases}
1, & W_\Sigma\le131072,\\
T_{\mathrm{request}}, & W_\Sigma>131072.
\end{cases}
$$

若首选 ISA 不可用则退到另一条可用路径。显式
`x86_avx512_bf16`/`x86_amx_bf16` backend 不应用上述 ISA 或线程收缩，保持
请求的实现和线程数；这使 benchmark/control 可重复。

AMX 的 per-expert pattern 继续由 exact $M_i$ JIT key 隔离，但 C8i 阈值按
model dimension 分段：

$$
\rho(H,F)=
\begin{cases}
+\infty, & H<4096,\\
128, & H\ge4096\ \land\ 2F\ge H,\\
96, & H\ge4096\ \land\ 2F<H,
\end{cases}
\qquad
p_i=
\begin{cases}
\texttt{m1n4}, & M_i\ge\rho(H,F),\\
\texttt{m2n2}, & M_i<\rho(H,F).
\end{cases}
$$

`m2n2` 的 M1--16 tail 和 `m1n4` 的奇数末尾 N block 由 `m1n2` exact-tail
kernel 处理。`generic_v1` 保留旧规则 $M_i\ge76$ 时选 `m1n4`，作为未知 CPU
上的保守兼容策略，而不是 C8i 的 production 阈值。

令 $K_{13}=\operatorname{round\_up}(H,32)$、
$K_2=\operatorname{round\_up}(F,32)$。两级 loop-order 的单个 packed-B block
都是 $64K_s$ bytes。C8i 对 $M_i\le16$ 保持不分窗；其余情况为：

$$
(u_{13},u_2)=
\begin{cases}
(0,0), & M_i\le16,\\
\left(
\max\left(1,\left\lfloor\frac{2^{20}}{64K_{13}}\right\rfloor\right),
\max\left(1,\left\lfloor\frac{2^{19}}{64K_2}\right\rfloor\right)
\right), & M_i>16.
\end{cases}
$$

当 $p_i=\texttt{m1n4}$ 且 $u_s>1$ 时，再把 $u_s$ 向下对齐为偶数，以保持成对
N block。H=4096、F=512 时非小 M 得到 W13=4、W2=16。`generic_v1` 保留旧的
全 M byte-budget 公式；零值 override 明确恢复不分窗 loop order，正整数
override 强制 block 数。AVX-512 默认仍不应用 AMX cache window。

AVX-512 与 AMX executor 共享 dimension-aware team-N mapper。C8i 的目标 route
行数为：

$$
r(H,F)=\operatorname{clamp}\left(
\left\lceil\frac{64\cdot4096\cdot512}{HF}\right\rceil,\ 16,\ 256
\right),
$$

`generic_v1` 则保留 $r=64$。令 $T=T_{\mathrm{eff}}$、$A$ 为 active expert 数，
按 route 数降序记为 $M_{(1)}\ge M_{(2)}\ge\cdots$。W13/W2 可独立拥有的 N
单元数为：

$$
q_{13}=\frac{\operatorname{round\_up}(F,16)}{16},\qquad
q_2=\frac{\operatorname{round\_up}(H,32)}{32},\qquad
q=\max(q_{13},q_2).
$$

一个包含 $t_i$ 个 worker 的 expert team 最多取 $t_i\le q$；第 $\ell$ 个 worker
对任一 stage 的 $q_s$ 个 block 拥有区间：

$$
b_{s,\ell}=\ell\left\lfloor\frac{q_s}{t_i}\right\rfloor
 +\min(\ell,q_s\bmod t_i),\qquad
e_{s,\ell}=b_{s,\ell}+\left\lfloor\frac{q_s}{t_i}\right\rfloor
 +\mathbf 1[\ell<q_s\bmod t_i].
$$

这些区间两两不交并覆盖 $[0,q_s)$。team 内先并行 gather（AVX-512 按 M12
panel，AMX 按 logical row），barrier 后计算 W13 区间，再经第二个 barrier 计算
W2 区间。故 W13 intermediate 和 W2 output 都是 single-writer，最终 token merge
仍保持原来的 TopK 顺序。

对一个 wave 中的 expert 集合，先令所有 $t_i=1$。在
$\sum_i t_i<T$ 且仍有 $t_i<q$ 时，重复选择：

$$
i^*=\arg\max_i\frac{M_i}{t_i},\qquad t_{i^*}\leftarrow t_{i^*}+1.
$$

runtime mapping 的分支顺序为：

1. 若 $M_{(2)}>0$、$M_{(1)}\ge r(H,F)$ 且
   $M_{(1)}\ge2M_{(2)}$，按 $M_i$ 降序生成 waves。当前 wave 的最大 route 为
   $M_{\max,w}$、剩余 expert 数为 $R_w$ 时，放入的 expert 数为
   $a_w=\min(R_w,\max(1,\min(T,\lfloor r(H,F)T/M_{\max,w}\rfloor)))$，
   再按上式分配 team width；
2. 否则若 $A<T$，所有 active experts 放入同一个 cooperative wave，再按上式
   分配剩余 workers；
3. 否则保留 atomic expert queue，每个被领取的 expert 使用一个 worker。

team scratch 按同一 slot 跨 waves 的最大 $M_i$ 分配；merge worker 数取
$\min(T_{\mathrm{eff}},\text{tokens})$。这一 mapper 改变实际 $I_i(t)$ 和 active
configuration $\mathcal Z$，但不扩大 `IntervalPlanner` 的
width/shape/ordering candidate set。profile identity 必须记录 policy 名称与版本；
跨 CPU 的阈值变更应新增 profile，而不是静默修改 `generic_v1`。

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

性能近似还必须分层验证：$\widehat I$ 报告未见 route/thread 的时间误差和排序
误差；$\widehat D$ 报告未见同构及异构 active configuration 的 slowdown 误差；
最终 planner 报告在真实 runtime 上的 schedule regret。单独证明 isolated table
准确，不能证明 contention-aware planner 准确。

### 9.1 Amazon 192-core NUMA0 扩展校准

2026-07-14 在 Neoverse-V3 NUMA0 的 CPU `0-95` 上生成 TP4、TP2、EP4、EP2
的 split/no-split schema-v2 profile。每个 profile 包含 117 个 isolated 点
（13 个 route、9 个线程宽度）和 120 个 homogeneous contention 点，使用 5 次
warmup、20 次正式采样，并保留原始 samples。所有 profile 通过 schema、kernel
identity、split-pair grid 和 `ContentionCostModel` 加载验证。

测量证明 32T 以上的 isolated scaling 与 expert shape 相关：TP4 split 的短
route 常在 32--64T 饱和，而 EP4 长 route 可继续受益到 96T。后来定位到
64-local-expert TP profile 的“每个 expert 都有 2040 routes”full-call cliff
包含每轮重新分配并首次写约 1 GiB BF16 output 的成本，不能解释为 GEMM
contention。2026-07-16 起 profiler 为每个测量点预分配 native `out`，warmup
完成 first-touch，正式样本只测稳态 expert compute。旧 profile 中缺少
`measurement.output_buffer=preallocated_reused_native_out` 的大总 route 点不得
用于校准 $D_i(\mathcal Z)$；allocation/page-fault 应作为独立 E2E 成本处理。

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

上述留出只验证 legacy 4/16 MiB-equivalent stage。新增 byte-window variant 的
首轮 NUMA0 验证固定总线程为 96、TP4 shape、每专家 route 2040：`24x4T` 的
4/2/1 MiB 时间为 30.55/32.14/36.73 ms，`48x2T` 为
85.90/61.72/65.79 ms，`96x1T` 为 310.61/179.36/133.12 ms。逆序复测保持同一
排序。三个 schedule 的最优点分别为 4/2/1 MiB，等价于约 1 MiB packed B / active
worker，同时 nominal aggregate window 都约为 96 MiB；因此当前实验尚不能独立
识别 private-L2 与 aggregate-cache 两种作用。

该结果证明 byte window 必须作为 kernel/schedule 联合维度，而不能把 1 MiB 或
2 MiB 设成全局默认值。planner/profile schema 已接入
`(weight_window_bytes, w13_window_ranges, w2_window_ranges)` identity，并联合搜索
`(window, shape)`；但 variant 只有在提供相同
`(shape, route, threads, concurrent experts)` 网格的独立 isolated/contention
profile 时才进入候选。默认 catalog 尚未加入这些重新校准的表，因此默认行为仍
是 legacy split/no-split。原始数据见
`optimizations/fused_moe_sve/results/amazon_192c_weight_windows.md`。

### 9.3 Ready-token combine 首轮验证

2026-07-16 在 Neoverse-V3 NUMA0 CPU `0-95` 上验证默认 async 候选路径，固定
`tokens=2048`、`top_k=6`、`H=4096`、`F=512`、12 experts、每专家 8T，并在每轮
内随机化 control/candidate 顺序。均衡 route 下负载门槛关闭实验路径，三轮
31-sample median 差异为 -0.14%、+0.10%、-0.29%。25%/75% 两组不相交 TopK
分布下，trace 显示 605/2048 个 token 在 expert 阶段结束前完成 merge；三轮
median 差异为 +0.23%、+0.33%、+0.29%，仍低于约 1% 噪声下界。

首个 per-route 原子减计数原型在均衡分布上从 7.754 ms 退化到 10.116 ms，已被
拒绝。当前实现改为每 expert 一次 completion publication、只读 TopK 状态检查、
每 token 一次 CAS 和按 expert 批量入队。该结果只证明第一版在两个受控分布上
没有可分辨的 median 回退；尚不能证明对 captured routing 有净收益，也不能作为
planner 中 merge-overlap 的 cost 校准。

### 9.4 Amazon 192-core 双 NUMA 稳态校准

2026-07-16 在 Neoverse-V3 的全部 192 核上以两个同步 rank 重测 TP4/F512
split-W13 profile：rank 0 使用 CPU `0-95`/NUMA0，rank 1 使用 CPU
`96-191`/NUMA1，每轮取两个 rank wall time 的最大值。每个 rank 流式使用 64 份
不同 expert 权重；profile 包含 117 个 isolated 点和 120 个 contention 点，使用
5 次 warmup、20 次正式采样，并复用预分配 native `out`。

稳态结果消除了旧 allocate-per-call cliff。`64 experts x 2040 routes` 的 `[96]`
shape 从旧口径约 5.53 s 降为 250.88 ms，derate 为 0.988；全表最优是每 rank
`24x4T`，wall time 91.54 ms，双 rank 聚合吞吐 35.89 TFLOP/s。代表性最优点为：

| routes/expert | 每 rank shape | 双 rank wall | 聚合吞吐 |
| ---: | --- | ---: | ---: |
| 1 | `12x8T` | 2.03 ms | 0.79 TFLOP/s |
| 12 | `24x4T` | 2.12 ms | 9.10 TFLOP/s |
| 48 | `6x16T` | 3.98 ms | 19.43 TFLOP/s |
| 192 | `6x16T` | 9.34 ms | 33.12 TFLOP/s |
| 768 | `6x16T` | 34.14 ms | 36.24 TFLOP/s |
| 2040 | `24x4T` | 91.54 ms | 35.89 TFLOP/s |

M=768 时 `12x8T` 与 `6x16T` 相差约 0.1%；M=2040 时更细的 `24x4T`
重新占优，等价于每 rank 同时保持约 96 MiB split packed-B stage。两个 NUMA 的
isolated median 在长 route 上接近：M=2040 的 32/64/96T 最大比值分别为
1.056/1.017/1.034；短 route 的 64/96T 可出现更大不对称，因此 profile 必须保留
pairwise-max 聚合，不能用两个 rank median 的平均值替代。

### 9.5 256-expert TP4 双 NUMA 校准

2026-07-17 按真实 256 routed-expert TP4 人口重测同一 F512
split-W13 kernel。TP 不分割 expert 数量，因此每个 rank 都生成并轮换
256 份不同权重；rank 0/1 仍分别绑定 CPU `0-95`/`96-191` 和
NUMA0/1。profile 包含 117 个 isolated 点和 140 个 contention 点，
每点 5 次 warmup、20 次正式采样，并按逐轮两 rank 的较慢者聚合。

代表性最优点为：

| routes/expert | 每 rank shape | 双 rank wall |
| ---: | --- | ---: |
| 1 | `48x2T` | 8.67 ms |
| 12 | `24x4T` | 9.04 ms |
| 48 | `6x16T` | 16.02 ms |
| 192 | `6x16T` | 36.04 ms |
| 768 | `12x8T` | 125.88 ms |
| 2040 | `24x4T` | 329.79 ms |

M=12 存在宽平台：`24x4T`/`48x2T`/`96x1T` 分别为
9.042/9.049/9.094 ms，相对最优只相差 0.0%/0.1%/0.6%。因此
96 个单线程 expert 并不差；旧 E64 profile 未包含该可行 shape，
其 `24x4T` 结论只适用于 64-task full call，不得作为 256-expert
TP4 planner 的总体调度结论。本轮只替换 profile identity 和验证数据；
$I_i(t)$、$D_i(\mathcal Z)$ 定义及 production 剪枝不变。

### 9.6 M12 冷 B 带宽扩展曲线

2026-07-17 在 Neoverse-V3 NUMA0 CPU `0-95` 上固定 TP4/F512、M=12
和每 expert 1T，扫描 1--96 个同时 active experts。每个样本只执行一个
完整并发波，样本间在 256 份独立 packed weights 上循环移动窗口；
因此不包含固定总任务数产生的 partial-wave tail。点顺序随机化，每点
5 次 warmup、30 次正式采样。

M=12 只有一个物理 M panel，每个 expert 的 W13+W2 packed B 合计
12 MiB 且无 panel 间复用。定义冷 B 有效带宽为：

$$
B_{B,\mathrm{eff}}(n)
=\frac{n\cdot 12\ \mathrm{MiB}}{T_{\mathrm{one\ wave}}(n)}.
$$

该值包含 GEMM 的 load issue、BFMMLA 与 dispatch 重叠，是 planner 可用的
effective packed-B rate，不是 PMU memory-controller byte count。代表点为：

| 1T experts / threads | wall | $B_{B,\mathrm{eff}}$ | 相对 96T |
| ---: | ---: | ---: | ---: |
| 1 | 0.559 ms | 22.5 GB/s | 6.4% |
| 8 | 0.812 ms | 124.0 GB/s | 35.2% |
| 12 | 0.831 ms | 181.7 GB/s | 51.6% |
| 24 | 1.136 ms | 265.9 GB/s | 75.5% |
| 48 | 2.021 ms | 298.8 GB/s | 84.8% |
| 64 | 2.514 ms | 320.4 GB/s | 91.0% |
| 86 | 3.219 ms | 336.2 GB/s | 95.5% |
| 96 | 3.430 ms | 352.1 GB/s | 100.0% |

按“从该线程数开始，后续所有实测点都不低于目标”的稳健口径，
50%/75%/90%/95% 峰值带宽分别需要 12/24/64/86 个 1T experts。
该曲线可作为未来 DRAM pressure shadow 的 NUMA-local service response，但当前
planner 仍使用 empirical contention profile；公式、可行域和 production 剪枝未改变。

### 9.7 Xbyak exact-M kernel 验证

2026-07-20 在 Neoverse-V3 NUMA0 CPU `0-95` 和 8-core Neoverse-V1 CPU
`0-7` 上，以 H4096/F512、8 份连续 expert、split-W13 对比
`jit/xbyak_exact_m` 与 `asm/static_bucketed`。每次实现切换后先执行一次不计时
调用，再连续测 5 次，避免把 instruction-cache 切换计入某一 variant；每点取
11 次样本中位数。M1--M12、SiLU poly4/5/6、normal/scheduled/async 和 W2
direct-route 均与 static asm bit exact。

V3 单线程 M5/M6 分别提升 11.75%/11.75%，M9/M10 提升
10.07%/10.44%；M12 回退 0.46%。M192/M2040 在 1--96T 的全部 control 点落在
`[-0.86%, +0.79%]`。V1 的 M5/M6/M9/M10 在 1--8T 提升约 5%--12%，M12 与
长 route control 最差回退 0.82%。因此 exact-M JIT 进入 production 默认，static
asm 保留 fallback；但该结论只证明 kernel variant 本身，不允许继续复用旧
bucketed-M 的 isolated/contention 表。新 profile 必须覆盖 route 1--12，并记录
`sve_implementation=jit`、`m_tail_policy=xbyak_exact_m` 及 Xbyak commit。

同日生成两组完整 split/no-split schema-v2 pair。8-core V1 standalone
F512/E8 每个 policy 含 80 个 isolated 点和 68 个 contention 点；192-core V3
双 NUMA TP4 F512/E256 每个 policy 含 180 个 isolated 点和 238 个 contention
点，双 rank 样本按 pairwise maximum 合并。8-core 的 M192/M2040 上 split
吞吐分别提升 5.34%/4.25%，但双 NUMA E256 的同两点分别回退 3.03%/2.72%；
这直接否定跨机器或跨 workload 固定选择 split 的支配假设。两组 profile 的
route grid 均直接覆盖 1--12，split pair 的 source/binary hash、Xbyak commit、
线程和 shape 网格一致，并由 catalog 回归测试验证。

### 9.8 调度论文输入分布

固定的 synthetic validation catalog 使用 $T=2048$、$K=6$、$E=256$，因此
$\sum_iM_i=12288$。它覆盖四类互补输入：

| Family | Histogram | 控制变量 |
| --- | --- | --- |
| uniform | `256x48` | 理论均匀基线 |
| active-set sweep | `8x1536`、`16x768`、`32x384`、`64x192`、`128x96`、`256x48` | 固定 routes，改变活跃权重工作集和单 expert route |
| tiered hotspot | `4x768 + 12x384 + 48x96` | hot/warm/cold 三层负载 |
| long-short bimodal | `5x2040 + 174x12` | 长 route compute 与 M12 冷权重流量并存 |

所有 synthetic histogram 都满足 2.1 节的全局 TopK 约束。active-set 的 256
endpoint 与 uniform 共用 `moe256-uniform`，避免重复 workload。它们只用于
planner/cost-model 的受控比较；真实有效性仍由 captured routing trace 验证，
不得把 synthetic preset 当作真实 router 概率模型。此次扩展不改变
$I_i(t)$、$D_i(\mathcal Z)$、目标函数、线程宽度集合或 production 剪枝。

### 9.9 单核 packed-B service ceiling 与 exact-M 效率

2026-07-23 在 Neoverse-V3 NUMA0 CPU 48 上补充 H4096/F512、单 expert、
单线程的冷权重校准。standalone reader 只执行 8-way-unrolled SVE `LD1H`，
不执行 BFMMLA、epilogue 或输出写回；768 MiB 连续读取的两轮中位数为
41.250/41.223 GB/s，64 份 12 MiB chunk 乱序轮换为 40.051/40.035 GB/s。
后者与每个 expert 的 W13 8 MiB + W2 4 MiB packed-B 粒度一致，因此定义本机
单核 production-like service ceiling：

$$
\beta_{B,1}^{\mathrm{V3}}=40.04\ \mathrm{GB/s}.
$$

该值是 CPU 侧 effective packed-B read ceiling，不是 uncore PMU 的 DRAM
controller byte rate。单核 kernel 的访存效率统一定义为

$$
\eta_{\mathrm{mem}}(M)
=\frac{6HF/T_{\mathrm{W13+W2}}(M)}
       {\beta_{B,1}^{\mathrm{V3}}},
$$

其中分子按每个 expert 12 MiB packed weights 只扫描一次计算。计算侧同时保留
三个不可互换的量：

$$
m_c=2\left\lceil\frac M2\right\rceil,\qquad
\eta_{\mathrm{lane}}=\frac M{m_c},
$$

$$
\eta_{\mathrm{compute,useful}}
=\frac{6MHF/T_{\mathrm{W13+W2}}}{403.8\ \mathrm{GFLOP/s}},\qquad
\eta_{\mathrm{issue}}
=\frac{6m_cHF/T_{\mathrm{W13+W2}}}{403.8\ \mathrm{GFLOP/s}}.
$$

64 份不同 expert 权重轮换、每 M 192 个样本的代表点为：

| M | GEMM stage | $\eta_{\mathrm{lane}}$ | useful GFLOP/s | $\eta_{\mathrm{compute,useful}}$ | $\eta_{\mathrm{issue}}$ | packed-B GB/s | $\eta_{\mathrm{mem}}$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.3472 ms | 50.0% | 36.24 | 8.97% | 17.95% | 36.239 | 90.51% |
| 2 | 0.3447 ms | 100% | 73.00 | 18.08% | 18.08% | 36.500 | 91.16% |
| 4 | 0.3580 ms | 100% | 140.58 | 34.81% | 34.81% | 35.144 | 87.77% |
| 8 | 0.4336 ms | 100% | 232.16 | 57.49% | 57.49% | 29.020 | 72.48% |
| 12 | 0.5245 ms | 100% | 287.90 | 71.30% | 71.30% | 23.992 | 59.92% |

完整 M1--M12 表与复现命令位于
`optimizations/fused_moe_sve/results/amazon_192c_single_core_m1_m12_efficiency.md`。
相邻 odd/even M 具有相同物理 row-pair 数，因此 stage time 与
$\eta_{\mathrm{issue}}$ 基本相同；odd M 的差异由
$\eta_{\mathrm{lane}}$ 单独表达。40.04 GB/s ceiling 只作为后续解析模型的
机器校准常数；本轮不修改 active planner、$I_i(t)$、$D_i(\mathcal Z)$、
contention table 或 production 剪枝。

### 9.10 x86 AMX 自动子变体验证

2026-07-20 在 2-core Amazon C8i（Intel Xeon 6975P-C）上验证 8.4 的确定性
runtime policy。该节记录的固定 M=76 与 64-row policy 现作为 `generic_v1`
历史基线；C8i production 已由 9.12 的 dimension-aware profile 取代。
H=4096/F=512 的 m2n2/m1n4 AB/BA 配对长采样显示 M=68/72
位于约 ±2% 的尾部/频率噪声区；M=76 时 `m1n4` 单核为 1.207 vs 1.253 ms、
双核为 1.522 vs 1.536 ms，并在 M=80 以后继续领先。因此当前保守阈值取 M=76。
skewed route histogram `[384,19,19,18,18,18,18,18]` 下，per-expert auto
混合策略单核为 9.036 ms，比最佳全局固定 pattern 的 9.427 ms 快 4.3%；双核
为 10.595 vs 10.665 ms。

自动 byte window 在 M=16 相对显式 0/0 不分窗无可分辨回退；M=48/80/256/2048
的单核 speedup 为 1.35x/1.94x/2.31x/2.09x，双核为
1.21x/1.53x/1.81x/1.65x。H=4096/F=512 的公式值 4/16 与无环境变量路径在
单核扫描中相差不超过 0.4%。focused x86 correctness suite 为 95 passed、4 skipped，
并覆盖同一次调用内 M=75/M=76 两种 pattern、H/F/K/N tails、双线程以及 AMX
kill-switch 回退 AVX-512。完整命令、样本和热状态限制见
`optimizations/fused_moe_avx512/results/amazon_c8i_2core_auto_dispatch_20260720.md`。

该验证只校准 x86 runtime 的确定性响应 mapper，不改变 CPU MoE planner 的决策
变量或 candidate space，因此本次不修改 planner/cost-model 公式测试。跨 CPU 或
显著不同 H/F 的 policy 仍需单独 held-out 验证并更新 profile identity。

### 9.11 x86 cooperative N-split 与 skew-wave 验证

2026-07-22 在 8-core Amazon C8i（Intel Xeon 6975P-C，8 physical cores、1 thread
per core）上验证固定 64-row 的初版 team-N mapper；该 target 现作为
`generic_v1` 历史基线，C8i production 已由 9.12 的 dimension-aware target
取代。固定 H=4096、F=512、BF16、top-k=1，
使用 CPU `0-7`、`OMP_DYNAMIC=FALSE`、`OMP_PROC_BIND=close`、
`OMP_PLACES=cores`、`OMP_WAIT_POLICY=PASSIVE`、8 次 warmup 和 31 次正式采样；
权重 prepack/JIT warmup 不计入运行时间。AMX auto pattern 的 median 如下（ms）：

| route histogram | 1T | 2T | 4T | 8T | 1T/8T speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `[2048]` | 26.446 | 14.209 | 7.917 | 5.755 | 4.60x |
| `[1536,256,256]` | 39.225 | 23.311 | 14.587 | 11.893 | 3.30x |

同一进程配置下，PyTorch/oneDNN staged baseline 的 8T median 分别为 19.611 ms
与 23.640 ms，因此 fused executor 分别快 3.41x 与 1.99x。M=256 单热 expert
从 1T 3.574 ms 降到 8T 0.855 ms（4.18x）。AVX-512 BF16 单热 M=2048 也从
262.063 ms 降到 39.461 ms（6.64x），证明 N ownership 对两种 x86 backend 都
生效；但该 AVX-512 kernel 仍慢于同线程数 oneDNN，不能把 scaling 误写为绝对
kernel 优势。

另一个 12-warmup/51-sample 小 M sweep 中，单热 M=1/4/16/48/64 的 8T 相对 1T
分别为 7.49x/2.76x/4.89x/3.69x/3.99x；在该 H/F 与机器上没有观察到需要按 M
进一步压低 team width 的性能边界。

为避免 cooperative barrier 在本来已有足够 expert parallelism 时造成回退，最终
policy 不对均衡 route 强制 wave。E2 `[1024,1024]` 与 E8 `[256,...,256]` 的 8T
median 分别为 9.183 ms 和 8.709 ms，仍走原 expert queue；对应 oneDNN 为
21.747 ms 和 23.882 ms。仅在 2x/64-row 强偏斜门槛命中时才使用有序 waves。

x86 correctness suite 为 163 passed；测试覆盖 AVX-512 和 AMX
`m1n2/m2n2/m1n4`、1/2/4/8 threads、H/F/K/N tails、单 expert、active expert
不足、均衡 8 expert、active expert 数等于线程数的强偏斜 wave，以及 worker
异常时 barrier cancellation。1T 与 8T 输出要求 BF16 bit-exact，并同时对 naive
reference 做容差比较。完整命令和原始 median/best 数据见
`optimizations/fused_moe_avx512/results/amazon_c8i_8core_nsplit_20260722.md`。

本次变化是 synchronous x86 executor 的确定性 response mapper；没有修改
planner candidate space、profile schema 或 cost-model 参数，所以不新增
planner/cost-model 公式测试。若将该 team/wave mapper 提升为 planner 可选动作，
必须先为不同 $t_i$/wave active set 重建 $I_i(t)$、$D_i(\mathcal Z)$ profile 并做
held-out regret 验证。

### 9.12 x86 dimension-aware 自动策略校准

2026-07-26 在 8-core Amazon C8i（Intel Xeon 6975P-C，CPUID family 6、
model 173、stepping 1）上校准 8.4 的 `intel_06_ad_c8i_v1`。CPU 固定为
`0-7`，设置 `OMP_DYNAMIC=FALSE`、`OMP_PROC_BIND=close`、
`OMP_PLACES=cores`、`OMP_WAIT_POLICY=PASSIVE`；权重 prepack、JIT warmup 和
output allocation 不计入样本。主要比较在同一进程内轮换 auto、forced AVX-512
和 forced AMX，以降低频率与热状态偏差。

调用级 ISA 扫描表明 H64/F16 的 M1/M2/M4 上 AVX-512 分别约快
51.5%/49.8%/32.8%，到 M8 差距缩小到约 10%--19%，M16 已由 AMX 领先约 9%；
H128/F32 在 M1/M2 仍由 AVX-512 领先约 37%/28%--32%，M4 接近持平，M8 由
AMX 领先约 17.5%。因此 C8i 取 $M_{\max}HF\le8192$ 作为 AVX-512 小调用域。
H64/F16/M1、请求 8 workers 时，完整 auto 为 5.921 us，forced AVX-512/AMX
分别为 12.525/19.023 us；固定单线程复核 auto 5.168 us、forced AVX-512
5.115 us，auto regret 1.04%。

线程校准使用 aggregate work。H256/F64、8 个 M1 experts 恰好满足
$M_\Sigma HF=131072$，auto 收缩为 1 worker 后为 17.930 us，而 forced
AVX-512/AMX 的 8-worker control 为 34.435/29.661 us。更大
H512/F128、8 个 M1 experts 的 AMX control 则从 1T 122.643 us 降到
8T 40.582 us，因此阈值上方保留请求线程数。

AMX pattern 在 H1024/F256 的 M48--2048 基本由 `m2n2` 领先；H4096/F512 的
crossover 位于 M96 附近，H4096/F2048 则从 M128 起 `m1n4` 更稳定。8T
H4096/F512/M96 的同进程轮换 median 为 auto 0.210900 ms、forced `m1n4`
0.209775 ms、forced `m2n2` 0.216928 ms，auto 相对最佳 regret 0.54%。
H4096/F512/M64 的 1T auto/forced AMX 为 0.920052/0.919424 ms，regret 0.068%。

cache window 的 1T auto 相对不分窗 control 在 H4096/F512 的
M64/M128 为 1.20x/2.53x，在 H4096/F2048 为 1.24x/2.47x；M<=16 没有稳定
收益，因此 C8i profile 只对更大 M 启用 1 MiB/512 KiB byte-budget 公式。

dimension-scaled N target 另以 forced AMX/m2n2、8T 做 profile A/B：

| H/F | route histogram | C8i target | `generic_v1` target | C8i | generic | speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1024/256 | `[768,128,128]` | 256 | 64 | 0.389054 ms | 0.398581 ms | 1.02x |
| 4096/2048 | `[144,24,24]` | 16 | 64 | 1.736484 ms | 2.451086 ms | 1.41x |

这两组 target A/B 是独立进程的 31-sample median，不具有同进程轮换对微小差异的
控制强度；第一行 2.45% 应视为方向性证据，第二行 41.15% 足以拒绝固定 64-row
target。production H4096/F512/M64 的 auto 保持 AMX，1T 相对 forced AMX
regret 低于 0.1%，说明小维度收益没有以代表性大 shape 回退为代价。

focused x86 correctness suite 为 226 passed、3 skipped，覆盖 policy boundary、
CPUID/profile 输出、同一 backend ID 104 packed weights 强制走 AVX-512/AMX 的
bit-exact 输出、线程 crossover、pattern/cache 与 H/F/K/N tail。完整命令、逐项
结果和解释边界见
`optimizations/fused_moe_avx512/results/amazon_c8i_8core_dimension_aware_policy_20260726.md`。

该 policy 只改变同步 x86 executor 的确定性 runtime response mapper，不改变
planner candidate space、profile schema 或 cost-model 参数，因此没有新增
planner/cost-model 公式测试。若未来让 planner 显式选择 ISA、有效线程数或
pattern/cache，必须把它们提升为 variant identity，重新测量对应的
$I_i(t)$/$D_i(\mathcal Z)$ 并做 held-out schedule regret 验证。

## 10. 同步规则

发生以下任一变化时，必须同步更新本文档：

- job 粒度从 expert 改为 phase，或反向合并；
- 线程宽度定义、最大线程数或可行宽度集合变化；
- 是否允许抢占、迁移、动态调整线程数或主动 idling；
- CPU 核心硬约束、affinity 或其他真正可执行性约束变化；
- $I_i(t)$、$D_i(\mathcal Z)$ 的定义或 active configuration 状态发生变化；
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
| 2026-07-15 | v0.13 | 将 GEMM shadow 重构为实现弱相关三层模型：算法层定义 useful work/compulsory traffic，kernel mapper 产生统一 physical demand，目标机器实测 profile 提供 service response；SVE M12 细节移出通用核心，active planner 与剪枝不变。 |
| 2026-07-15 | v0.14 | 将原始问题改为 256-expert route-task 实例对应的问题族：CPU 核心数是基础硬约束，真实 isolated time $I_i(t)$ 与动态 contention slowdown $D_i(\mathcal Z)$ 定义执行速率；LLC/带宽从硬 cumulative constraint 降为 slowdown/candidate-pruning 因素，并同步复杂度、solver 边界、剪枝表和验证要求。 |
| 2026-07-15 | v0.15 | 增加最优解存在性命题：由串行调度得到有限 horizon，将固定线程宽度与事件顺序的子问题表示为紧线性可行域，再由 Heine--Borel 和 Weierstrass 极值定理证明全局最小 makespan 必然取得；同时明确 $I,D$ 必须在所有候选配置上有限且为正。 |
| 2026-07-16 | v0.16 | 增加显式 packed-B byte-window kernel variant：按 SVE N tile 推导 W13/W2 range 数、实际最大窗口和有效线程上界；明确 range 间无 barrier、owner-scatter ownership、M12 短 route 边界，以及该 variant 在重新校准前不进入 production planner。 |
| 2026-07-16 | v0.17 | 增加可选 ready-token combine 外层：定义 TopK release time、单线程 merge CPU 约束和含 combine 的 makespan；记录 expert-first async heuristic、25% team-load 生效门槛、planner/cost-model 边界及 192-core NUMA0 首轮验证。 |
| 2026-07-16 | v0.18 | 将通过验证的 async ready-token executor 设为 SVE FP32 direct-route 默认；保留 1.25 team-load 门槛和环境变量值 0 的 post-expert fallback，不改变 planner 决策空间或现有 cost tables。 |
| 2026-07-16 | v0.19 | contention/working-set profiler 改为复用预分配 native `out`；将 allocation 与 first-touch 排除出 $I_i(t)$ 和 $D_i(\mathcal Z)$，并标记旧大 route allocate-per-call profile 不可用于 GEMM contention 校准。 |
| 2026-07-16 | v0.20 | 在 192 核双 NUMA 上以两个 96-core 同步 rank 生成 TP4/F512 split-W13 稳态 profile；记录 route-dependent 最优 lane shape、35.89 TFLOP/s 长 route 聚合吞吐和跨 NUMA pairwise-max 必要性。 |
| 2026-07-17 | v0.21 | 将双 NUMA TP4/F512 split-W13 校准扩展到真实 256 local experts，补齐 `96x1T` 等形状；记录 M=12 的 `24x4T`/`48x2T`/`96x1T` 宽平台，并禁止将 E64 full-call 排序外推到 256-expert TP4。 |
| 2026-07-17 | v0.22 | 增加 NUMA0 M12 单满波冷 packed-B 带宽校准：256 份权重窗口轮换、1--96 个 1T experts、峰值 352.1 GB/s；记录 50%/75%/90%/95% 稳健带宽所需的 12/24/64/86 线程阈值，不改变 active planner 或剪枝。 |
| 2026-07-20 | v0.23 | production SVE compute 改为 Xbyak exact-M1--M12：定义 $m_c=2\lceil M/2\rceil$ tail mapper，profile identity 增加 implementation/tail policy，route grid 补齐 1--12；记录 V1/V3 bit-exact 与稳态性能验证；planner `auto` 只原子选择完整 variant pair，禁止 JIT 和 static bucket profile 混用。 |
| 2026-07-22 | v0.24 | 增加论文评测用 uniform、active-set sweep、tiered hotspot 和 long-short bimodal 全局 TopK workload；补充合法 histogram 约束和验证覆盖说明，不改变 planner 可行域、cost model 或 production 剪枝。 |
| 2026-07-23 | v0.25 | 增加 V3 单核 packed-B service ceiling 40.04 GB/s；区分 exact-M lane、useful-compute、physical-issue 和 memory efficiency，并记录冷权重 M1--M12 验证；不改变 active planner、contention table 或剪枝。 |
| 2026-07-26 | v0.26 | 将全局 packed-B byte-window 接入 schema-v2 与 production planner：variant identity 包含目标字节和 W13/W2 实际 range 数，stage model 分别推进两段 range，联合搜索 `(window, core shape)` 并透传 runtime option；旧 profile 映射为 window=0，未提供新实测表时决策不变。 |
| 2026-07-26 | v0.27 | 将 x86 BF16 auto backend 改为优先 AMX 并回退 AVX-512；增加按 expert route 数选择 m2n2/m1n4 的确定性 M=76 policy，以及按 K-padded byte budget 自动推导 1 MiB/512 KiB cache window；同步 per-expert variant 公式、剪枝表、C8i 正确性/性能验证和 policy 可迁移性边界。 |
| 2026-07-26 | v0.28 | 将 SVE 的 team-N ownership 引入同步 x86 AVX-512/AMX executor：active expert 不足时按 route/width 贪心组 team，2x 且至少 64-row 的强偏斜 route 使用有序 waves，均衡 route 保留 expert queue；定义精确 N-range、team-width 与 wave 公式，记录 8-core C8i 1/2/4/8T 验证，并明确该 mapper 尚不进入 planner candidate space。 |
| 2026-07-26 | v0.29 | x86 BF16 `auto` 升级为 CPUID/profile 驱动的 dimension-aware runtime mapper：按 $M_{\max}HF$ 选择 AVX-512/AMX，按 $M_\Sigma HF$ 收缩小调用线程，按 H/F 选择 AMX pattern/cache，并以 $\operatorname{clamp}(\lceil64\cdot4096\cdot512/(HF)\rceil,16,256)$ 缩放 skew-wave target；保留 `generic_v1` 与显式 override，记录 C8i8 correctness、同进程轮换和 held-out target 验证。 |
