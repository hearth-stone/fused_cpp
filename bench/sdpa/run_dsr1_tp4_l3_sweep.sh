#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 在远端机器上跑 DeepSeek-R1 TP=4 prefill MLA 形状下，PyTorch SDPA vs
# L3 系列 SDPA 的多线程 GFLOPS 对比。
#
# 形状  : (B=1, N=32, L=2048, S=2048, E=192, Ev=128)  ← bench_sdpa_versions.py
#         里 BENCH_SHAPES 中已存在的 "DeepSeek-R1 TP=4 prefill MLA"
# 线程  : 1,2,4,8,16,32,64,79
# 绑核  : taskset -c 160-239   （子进程通过 fork 继承同一 affinity mask；
#         OMP_PLACES=cores + OMP_PROC_BIND=close 让 OpenMP 在该 mask 内
#         按物理核就近 binding）
# 版本  : pytorch_sdpa（torch.F.scaled_dot_product_attention 默认后端）
#         + 全部 flash2_neon_l3kv* L3 系列
#
# 输出  : <repo>/bench/sdpa/sdpa_versions_<ts>{,_sweep,_scaling}.{csv,json}
#         由 tests/conftest.py 的 pytest_sessionfinish 钩子写出。
#
# 用法（在远端机器上 fused_cpp 仓库根执行）：
#     bash bench/sdpa/run_dsr1_tp4_l3_sweep.sh
#
#     # 自定义 dtype / causal / 输出目录：
#     DTYPES=bf16 CAUSAL=noncausal \
#       bash bench/sdpa/run_dsr1_tp4_l3_sweep.sh
#
#     # 仅跑某个具体 L3 变体：
#     L3_VERSIONS=flash2_neon_l3kv_packv_qk_ublock4 \
#       bash bench/sdpa/run_dsr1_tp4_l3_sweep.sh

set -euo pipefail

# ── 仓库根：脚本位于 <repo>/bench/sdpa/，向上两级 ────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 可调参数（环境变量覆盖） ────────────────────────────────────────
CORES="${CORES:-160-239}"            # taskset 绑核范围
THREADS="${THREADS:-1,2,4,8,16,32,64,79}"

# 仅跑 DeepSeek-R1 TP=4 prefill MLA 形状，shape id 由 bench_sdpa_versions.py
# 的 _shape_id() 生成。pytest -k 用作 collection 过滤。
SHAPE_ID="${SHAPE_ID:-B1-N32-L2048-S2048-E192-Ev128}"

# torch sdpa baseline + L3 系列。pytorch_sdpa = F.scaled_dot_product_attention
# 默认后端；如需强制 MATH 后端把 pytorch_sdpa_math 也加入。
L3_VERSIONS="${L3_VERSIONS:-\
flash2_neon_l3kv,\
flash2_neon_l3kv_qk_ublock4,\
flash2_neon_l3kv_pquad,\
flash2_neon_l3kv_packv,\
flash2_neon_l3kv_packv_qk_ublock4,\
flash2_neon_l3kv_packv_pquad}"
TORCH_VERSIONS="${TORCH_VERSIONS:-pytorch_sdpa}"
SDPA_VERSIONS="${SDPA_VERSIONS:-${TORCH_VERSIONS},${L3_VERSIONS}}"

# bench_sdpa_versions.py 用 ids ["noncausal","causal"] 给 is_causal
# 参数化；用 dtype ids ["fp32","bf16"]。这里保留两者全跑——可通过
# DTYPES / CAUSAL 限制。
DTYPES="${DTYPES:-fp32,bf16}"
CAUSAL="${CAUSAL:-noncausal,causal}"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/bench/sdpa}"

# ── OpenMP / BLAS 线程绑核环境（让子进程继承） ─────────────────────
# CORES 是 taskset 给出的绝对核号；OMP_PLACES=cores 表示按物理核分
# place，OMP_PROC_BIND=close 在 mask 内紧邻分布。OMP_DYNAMIC=FALSE
# 防止运行时偷偷下调线程数，使重复跑结果可复现。
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"

# 让 conftest 的 thread-sweep 子进程在每条用例前调
# torch.set_num_threads(N)；OMP_NUM_THREADS / MKL_* 也由 conftest 自
# 行注入，不要在父层固定。

# ── 检查 taskset 是否可用 ───────────────────────────────────────────
if ! command -v taskset >/dev/null 2>&1; then
    echo "ERROR: taskset not found; install util-linux or run on Linux." >&2
    exit 2
fi

# ── 检查扩展是否带 OpenMP（无 OMP 时 thread sweep 会被 conftest 跳过）─
python -c "
from fused_cpp import _C
assert _C.has_openmp(), '_C.has_openmp() == False; rebuild fused_cpp with libomp'
print('[check] fused_cpp _C built with OpenMP: OK')
print('[check] omp runtime info:', dict(_C.get_omp_runtime_info()))
"

# pytest -k 表达式把 dtype / causal / shape 三个维度同时过滤。
# bench_sdpa_versions.py 用例 id 形如：
#   test_sdpa_bench[<version>-<causal>-<dtype>-B1-N32-L2048-S2048-E192-Ev128]
KEXPR=""
for c in ${CAUSAL//,/ }; do
    for d in ${DTYPES//,/ }; do
        sub="${c} and ${d} and ${SHAPE_ID}"
        KEXPR="${KEXPR:+${KEXPR} or }(${sub})"
    done
done

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "DeepSeek-R1 TP=4 prefill MLA  SDPA bench"
echo "  cores       : ${CORES}     (taskset)"
echo "  threads     : ${THREADS}"
echo "  shape       : ${SHAPE_ID}"
echo "  versions    : ${SDPA_VERSIONS}"
echo "  dtypes      : ${DTYPES}"
echo "  causal      : ${CAUSAL}"
echo "  output dir  : ${OUTPUT_DIR}"
echo "  kexpr       : ${KEXPR}"
echo "============================================================"

# ── 启动 pytest（绑在 cores=${CORES}） ───────────────────────────────
# 关键：
#   * `-m bench` 选中 bench 用例（默认 conftest 会跳过）。
#   * `--sdpa-thread-sweep=...` 让 conftest pytest_sessionfinish 钩子
#     fork 子进程，每个子进程 OMP_NUM_THREADS=N + --sdpa-num-threads=N，
#     子进程同样落在 ${CORES} mask 内（fork 自然继承 sched_setaffinity）。
#   * `--sdpa-versions=...` 限定本次只跑 torch + L3 系列。
#   * 父进程再跑一份基线（thread-sweep 之外的“默认线程数”那次），
#     落到 sdpa_versions_<ts>.csv；sweep 结果落到 *_sweep.csv +
#     *_scaling.csv（efficiency = gflops(N) / (N * gflops(1))）。
exec taskset -c "${CORES}" \
    python -m pytest \
        tests/bench_sdpa_versions.py \
        -m bench \
        -k "${KEXPR}" \
        --sdpa-versions="${SDPA_VERSIONS}" \
        --sdpa-thread-sweep="${THREADS}" \
        --sdpa-bench-output-dir="${OUTPUT_DIR}" \
        -v -s
