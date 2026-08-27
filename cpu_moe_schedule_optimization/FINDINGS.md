> **ARCHIVED (2026-07-03).** This file evaluates the retired wave-planner
> family and old kernel-bound profiles. Its equivalence and regret tables are
> historical provenance, not evidence for the current async interval-DAG,
> Plan V2, quick/full planners, or current SVE kernel. See
> [DEPRECATED_WAVE.md](DEPRECATED_WAVE.md),
> [MATHEMATICAL_MODEL.md](MATHEMATICAL_MODEL.md), and the paper-facing
> [readiness map](../docs/moe_paper_readiness.md).

# CPU MoE Planner — C++ 实现与评测结论

本文件汇总将 10 个 MoE 调度 planner 用 C++ 实现、并与精确最优对照后的工程与算法结论，对应
`README.md` 成功标准 #5（形成可复跑的 benchmark/spec）与 `DESIGN.md` 的 Phase 3
（精确求解器验证）和研究问题 Q3/Q4/Q6。

数据采集环境：本机 Apple Silicon（`-mcpu=apple-m2`，NEON）。默认 cost model 为合成
`T_expert(routes,threads)`（`ExpertCostModel.synthetic`）；真实 8 核 profile 复评见 §5.3。
工作负载除特别说明外为 `E=256, top_k=6, tokens=2048`。

> **⚠ cost 表时效性**：§5.3 的真实结论基于
> `cost_model/profiles/aws_dsv4_8c_pinned_auto_mn_20260630.json`，该表与某一版 MoE kernel
> 绑定。**MoE kernel 改动后此 profile 即失真**，"真实 cost" 相关数值需重新 profile 后复核
> （命令见 §5.3）；结构性结论（排序/regime/U 形/KK 无价值）在合成与真实下均成立。

---

## 1. 交付物

### 1.1 C++ 规划器核心（torch-free）

位于 `csrc/moe_planner/`，每个 planner 单独一个文件：

| 文件 | 内容 |
| --- | --- |
| `planner_types.h` | `PlanKind`(10 种)、`PlanResult`(scheduled-bridge 数组 + 执行成本) |
| `planner_simd.h` | `argmax_i64`：SVE(`__ARM_FEATURE_SVE`) / NEON(`__aarch64__`) / 标量三路 |
| `cost_model.h` | 合成公式 + 表查找；`pow` 拆为可预计算构件 |
| `planner_common.{h,cpp}` | `Workload`(dense `cost_rows[a*C+t-1]`)、顺序装箱、FFD、LPT 队列、shape 枚举(512 cap，有界生成) |
| `plan_fixed_global_threads.cpp` | FIXED_GLOBAL_THREADS |
| `plan_sorted_token_balanced_1t.cpp` | SORTED_TOKEN_BALANCED_1T (LPT) |
| `plan_uniform_waves.cpp` | UNIFORM_WAVES |
| `plan_greedy_marginal_gain.cpp` | GREEDY_MARGINAL_GAIN (SIMD argmax + 并行 budget) |
| `plan_enumerate_core_groups.cpp` | ENUMERATE_CORE_GROUPS (并行 shape) |
| `plan_load_proportional.cpp` | LOAD_PROPORTIONAL (新) |
| `plan_sqrt_load.cpp` | SQRT_LOAD (新) |
| `plan_log_load.cpp` | LOG_LOAD (新) |
| `plan_heavy_light_hybrid.cpp` | HEAVY_LIGHT_HYBRID (新) |
| `plan_karmarkar_karp.cpp` | KARMARKAR_KARP (新，多路 LDM) |
| `planner_dispatch.{h,cpp}` | kind 分发 + `plan_from_routes` |
| `exact_solver.{h,cpp}` | 小规模精确最优（subset DP） |
| `bindings.cpp` | pybind：`moe_schedule_plan`、`moe_exact_optimum`，经 `module.cpp` 注册进 `fused_cpp._C` |

### 1.2 验证与基准

- `standalone/moe_planner/{main.cpp,Makefile}`：torch-free harness，自检不变量 + 隔离测延迟
  （`make` / `make OMP=1`；`--only/--dist/--cores` 过滤单 planner）。
- `cpu_moe_schedule_optimization/planners/twin_*.py`：5 个新 planner 的 Python 孪生。
- `tmp/moe_planner_equiv_check.py`：C++ vs Python 逐位等价矩阵（亦可 pytest）。

---

## 2. 调度问题与目标

barrier-after-wave 模型，与所有 planner 一致：为每个 active expert 分配 `threads_e ≥ 1`，
并把 experts 划分为若干 wave（每 wave `Σ threads ≤ C`）；`wave 时间 = max_e cost(e, threads_e)`，

```
T_execute = Σ_wave (wave 时间)
总目标 = T_plan + T_execute
```

精确最优（`exact_solver`）用子集 DP 求解该目标：
`f(S) = min over wave T⊆S [wavecost(T) + f(S\T)]`，其中 `wavecost(T)` 为该 wave 的精确
min-max 线程分配。复杂度 `O(3^A)`，A≤16 可接受，A≤12 每例 <1s。

---

## 3. 正确性

- **C++ 与 Python 逐位相等**：12 分布 × cores{8,16,64} × 10 planner = **360 用例全部 MATCH**
  （`wave_offsets / team_expert_ids / team_threads / estimated_execute_cost_ns / active_expert_ids`）。
- **质数核边界**：C=79（`p(79)≈1060 万`划分）下 5 个核心 planner 仍逐位相等；C++ ENUMERATE
  用有界生成 **229 µs**，Python 需 **103 秒**（~45 万×）且结果一致。
- standalone 不变量（每 expert 恰一次、wave 线程预算不超、重算 execute 与上报值一致，重算走
  原始 `pow` 路径）：10 个 planner 全过 → 确认 dense cost_rows 快路径 bit-identical。

---

## 4. 规划时间 T_plan（隔离进程，单线程，dist=zipf，A=256）

单位 µs，`plan`（不含 prepare）：

| Planner | C=8 | C=16 | C=64 |
| --- | --- | --- | --- |
| FIXED | 1.4 | 1.4 | 1.5 |
| SORTED_1T | 4.8 | 6.2 | 14.5 |
| UNIFORM | 10.6 | 20.9 | 83.8 |
| ENUMERATE | 6.9 | 45.4 | 184.3 |
| GREEDY | 122 | 155 | 482 |
| LOAD_PROPORTIONAL | 6.4 | 4.7 | 3.5 |
| SQRT_LOAD | 6.6 | 4.8 | 3.6 |
| LOG_LOAD | 7.8 | 5.9 | 4.8 |
| HEAVY_LIGHT_HYBRID | 6.6 | 4.8 | 3.6 |
| KARMARKAR_KARP | 285 | 380 | 849 |

OpenMP-8（C=64）：ENUMERATE 184→143 µs，GREEDY 482→192 µs，KK 838 µs（未并行）。

**要点**：
- 闭式 planner（LOAD/SQRT/LOG/HEAVY_LIGHT）随核数增大反而**更便宜**（waves 变少），与
  GREEDY/ENUMERATE 相反。
- `prepare`（dense cost_rows）即便 A=256,C=64 也仅 ~6 µs（`pow` 预计算后从 139 µs 降下来）。
- 相对 Python 参考：100–1000×；最极端 ENUMERATE@79 核约 45 万×。
- **C++ 下 `T_plan`（µs 级）相对 `T_execute`（ms 级）几乎可忽略**——选 planner 应优先看 regret。

---

## 5. Regret

### 5.1 对 best-of-10（大网格：12 分布 × cores{8,16,64,79}，48 例）

`regret = exec / best_of_10`：

| Planner | mean | median | p90 | max | wins |
| --- | --- | --- | --- | --- | --- |
| GREEDY | 1.072 | 1.000 | 1.161 | 1.627 | 26 |
| LOAD_PROPORTIONAL | 1.103 | 1.039 | 1.261 | 1.888 | 15 |
| HEAVY_LIGHT_HYBRID | 1.106 | 1.000 | 1.370 | 1.888 | 26 |
| UNIFORM | 1.395 | 1.244 | 2.062 | 2.561 | 15 |
| SQRT_LOAD | 1.688 | 1.659 | 2.417 | 2.961 | 10 |
| ENUMERATE | 1.914 | 1.157 | 4.485 | 7.810 | 10 |
| LOG_LOAD | 2.621 | 1.769 | 5.930 | 8.141 | 10 |
| FIXED | 3.572 | 2.076 | 6.764 | 15.870 | 5 |
| SORTED_1T | 3.654 | 2.212 | 6.792 | 15.870 | 5 |
| KARMARKAR_KARP | 3.657 | 2.213 | 6.792 | 15.870 | 5 |

### 5.2 对精确最优（小规模：E∈{10,12} × 8 分布 × cores{4,6,8}，48 例）

`真 regret = exec / exact_optimum`：

| Planner | mean | median | p90 | max | #opt | ≤5% |
| --- | --- | --- | --- | --- | --- | --- |
| ENUMERATE | 1.073 | 1.071 | 1.152 | 1.214 | 10 | 16 |
| GREEDY | 1.091 | 1.081 | 1.204 | 1.323 | 14 | 18 |
| HEAVY_LIGHT_HYBRID | 1.129 | 1.084 | 1.427 | 1.490 | 13 | 19 |
| UNIFORM | 1.132 | 1.111 | 1.305 | 1.383 | 10 | 13 |
| LOAD_PROPORTIONAL | 1.176 | 1.155 | 1.443 | 1.568 | 10 | 11 |
| SQRT_LOAD | 1.378 | 1.382 | 1.791 | 1.951 | 10 | 11 |
| LOG_LOAD | 1.614 | 1.557 | 2.338 | 2.983 | 10 | 10 |
| FIXED | 1.934 | 1.738 | 2.894 | 4.284 | 4 | 4 |
| SORTED_1T | 1.992 | 1.873 | 2.925 | 4.284 | 4 | 4 |
| KARMARKAR_KARP | 1.992 | 1.873 | 2.925 | 4.284 | 4 | 4 |

### 5.3 真实 profile 复评（AWS 8 核 `aws_dsv4_8c_pinned`）— 重要

> **⚠ cost 表时效性警告**：下表基于
> `cost_model/profiles/aws_dsv4_8c_pinned_auto_mn_20260630.json`。该 profile 与某一版
> `fused_moe_bf16_tiled_scheduled` kernel 绑定。**MoE kernel 改动后此表即失真**，所有
> "真实 cost" 结论必须在重新 profile 后复核。重新生成（见 README「校准 Expert Cost Model」）：
>
> ```bash
> OMP_NUM_THREADS=1 ... taskset -c 0-7 .venv/bin/python \
>   cpu_moe_schedule_optimization/cost_model/profile_expert_cost.py \
>   --output cpu_moe_schedule_optimization/cost_model/profiles/<new>.json \
>   --hidden-size 4096 --ffn-hidden-size 512 \
>   --route-buckets 1,2,4,8,16,32,64,128,256,512,1024,2048 \
>   --thread-buckets 1,2,3,4,5,6,7,8 --warmup 5 --runs 30
> ```

**真实 cost 是 U 形**（与合成单调递减不同）：例如 `routes=512` 的耗时在 **4 线程触底
（15.0 ms）**，到 8 线程反而升到 31.5 ms（overhead/contention）。即**超过 ~4 线程会变慢**。

真 regret vs 精确最优（小规模 48 例，C∈{4,6,8}，真实表）：

| Planner | 真实 mean | (合成 mean) |
| --- | --- | --- |
| ENUMERATE | 1.059 | 1.073 |
| GREEDY | 1.097 | 1.091 |
| UNIFORM | 1.106 | 1.132 |
| HEAVY_LIGHT | 1.169 | 1.129 |
| LOAD_PROPORTIONAL | 1.185 | 1.176 |
| SQRT_LOAD | 1.316 | 1.378 |
| LOG_LOAD | 1.475 | 1.614 |
| FIXED | 1.552 | 1.934 |
| SORTED / KARMARKAR_KARP | 1.599 | 1.992 |

regret vs best-of-10（full E=256, C=8, 11 分布，真实表）：GREEDY **1.029**（8 wins）、
ENUMERATE 1.034、UNIFORM 1.072、LOAD_PROP/HEAVY_LIGHT 1.114、SQRT/LOG 1.162、
FIXED 1.209、SORTED 1.304、KK 1.305。

**合成 vs 真实——哪些结论变了**：
- **稳健**：顶档 {ENUMERATE, GREEDY, UNIFORM}、垫底 {FIXED, SORTED, KK} 不变；KK 仍无价值。
- **调度上限缩水**：FIXED regret 由合成 1.93×/最坏 4.28× → 真实 1.55×/最坏 2.45×。
  **真实硬件上"聪明调度"相对 fixed 的收益只有 ~1.5×**，因为超过甜点线程数反而倒退；
  合成模型夸大了多线程收益。
- **GREEDY 反超廉价 planner**：合成下 HEAVY_LIGHT/LOAD_PROP 与 GREEDY 几乎并列；真实下
  GREEDY 明显领先（best-of-10 1.029、真 regret 1.097），HEAVY_LIGHT 退到 1.169/1.114。
  原因：GREEDY 边际增益在甜点处 `gain≤0` 自动停手，**天然适配 U 形**；闭式 planner
  （proportional/sqrt/heavy_light）会把热点专家**过度分到 >4 线程**踩到上升段。
- **可落地改进**：给闭式 planner 加"线程数截断在该 routes 的 `argmin_t cost_rows`（甜点）"，
  即可在真实 U 形下追平 GREEDY，同时保住 ~3.6 µs。

---

## 6. 核心结论

> 下列第 1–3 条的**数值大小依赖 cost 模型**；§5.3 已用真实 8 核 profile 复核过一次，但该
> profile 可能因 kernel 改动而失真——改 kernel 后请重新 profile 并复跑评测。结论的**结构性
> 部分**（顶档/垫底排序、KK 无价值、GREEDY 适配 U 形、regime 分裂）在两种模型下均成立。

1. **调度收益真实**：合成下 FIXED 平均 1.93×/最坏 4.28×，真实表下收敛到 ~1.55×/最坏 2.45×；
   最好的 planner 平均 1.03–1.10×。即计划感知调度把执行拉近最优 ~1.5–1.8×。（回答 Q6）
2. **heavy/light 接近最优**（回答 Q3）：合成 regret 1.13；真实表下退到 1.17（过度分线程），
   需"甜点截断"修正。
3. **核数 regime 分裂，没有单一 planner 通吃**：
   - 小核 / 可整除（C≤8）：**ENUMERATE 最优**（合成 1.073 / 真实 1.059）。
   - 大核 / 质数（如 79）：**GREEDY / HEAVY_LIGHT / LOAD_PROP 称王**，ENUMERATE 崩坏（7.8×）。
4. **C++ 下 T_plan ≪ T_execute**：可负担"跑 2–3 个便宜 planner 取最优"，总规划仍只有几 µs。
5. **exact DP 边界**（回答 Q4）：`O(3^A)`，A≤12 每例 <1s，A≤16 可用；full E=256 需 bucketized DP。
6. **真实 cost 是 U 形 → 多线程有甜点（~4）**：边际增益类（GREEDY）天然处理；闭式分配类需截断。

### 逐 planner 定论

| Planner | 定论 |
| --- | --- |
| GREEDY | **真实表上质量最优**（best-of-10 1.029），天然适配 U 形；规划贵 1–2 个数量级 |
| ENUMERATE | 小核神器（C≤8 真实 1.059）、大核/质数毒药 |
| HEAVY_LIGHT_HYBRID | 性价比高（规划 ~3.6 µs），但真实表下会过度分线程，**需甜点截断**才追平 GREEDY |
| LOAD_PROPORTIONAL | HEAVY_LIGHT 的同类，同样需甜点截断 |
| UNIFORM | 中规中矩，A≤C 或 C 小时常接近最优 |
| SQRT / LOG | 覆盖用，质量一般 |
| FIXED / SORTED / KARMARKAR_KARP | 纯单线程，regret 垫底；**KK 仅 `A≤C` 才有意义**，典型 `A≫C` 下应限制/退役 |

---

## 7. 建议下一步

1. **AUTO selector**（C++ 侧廉价 gating）：按 `C 大小/可整除性、A vs C、gini` 选 planner——
   小核→ENUMERATE，大核/质数→HEAVY_LIGHT/LOAD_PROP，跳过 FIXED/SORTED/KK/LOG。因 T_plan≪T_execute，
   可直接"跑 HEAVY_LIGHT + LOAD_PROP（+小核 ENUMERATE）取最优"。
2. **bucketized DP**：把精确基线扩到 full E=256（回答 Q5：bucketized 最优 vs full-route 误差）。
3. **Phase 4 端到端**：把 10 个 planner 接进 `scheduled_bridge_bench` 测真实 BF16 kernel 的
   `T_plan + dispatch + execute + combine`（回答 Q1/Q7）。

---

## 8. 复跑命令

```bash
# 构建 C++ 扩展（含 planner + 精确求解器）
VIRTUAL_ENV=.venv uv pip install -e . --no-build-isolation

# C++ vs Python 逐位等价矩阵（360 用例）
.venv/bin/python tmp/moe_planner_equiv_check.py

# standalone：不变量自检 + 隔离延迟（单线程 / OpenMP）
cd standalone/moe_planner && make run            # 单线程
cd standalone/moe_planner && make OMP=1 run      # OpenMP
#   单 planner 隔离：./moe_planner --only GREEDY_MARGINAL_GAIN --dist zipf --cores 64 --csv
```

Python 侧调用：

```python
import fused_cpp._C as C
plan  = C.moe_schedule_plan(routes_hist, num_cores, "heavy_light")   # -> dict(scheduled bridge)
exact = C.moe_exact_optimum(routes_hist, num_cores, max_active=16)   # -> {feasible,num_active,execute_ns}
```
