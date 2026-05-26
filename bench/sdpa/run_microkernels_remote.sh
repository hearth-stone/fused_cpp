#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 在远端 aarch64 机器上跑 fused_cpp 微内核 GFLOPS 基准。
#
# 调用入口   : fused_cpp/bench/sdpa/bench_microkernels.py
# 测的 op    : qkt_8x8 / qkt_8x4 / pv_8x8（每个 impl 三个 microkernel 各跑一遍）
# dtype      : fp32 + bf16
# impl       : 默认 = _C.list_microkernel_impls() 全部
#
# 设计要点：
#   * **micro-bench 是单线程的**——不调度 OMP、不分 work-group。所以这里
#     固定 OMP_NUM_THREADS=1 + MKL/OpenBLAS 也置 1，把所有线程级噪音剥掉。
#   * 默认 taskset 绑到 1 个固定物理核（CORE 变量），和 SDPA 多线程 sweep
#     的 ${CORES} mask 不冲突；这样反复跑同一脚本数值稳定（无 cross-core
#     抖动 / SMT 抢占）。
#   * 输出双份：人读的 .log + 机器读的 .json（直接由 bench_microkernels.py
#     的 --json 写出，schema = {impl: {dtype: {op_us, op_gflops, op_seconds}}}）。
#   * 多组 (E, Sk) 形状：默认跑 (128,128)（与 fmla peak bench 一致）以及
#     (192,192)（对齐 R1-like SDPA 中的 head_dim=192）。每组各一份 JSON，
#     文件名带时间戳防覆盖。
#
# 用法（在远端机器上 fused_cpp 仓库根执行）：
#     bash bench/sdpa/run_microkernels_remote.sh
#
#     # 自定义形状：
#     SHAPES="64,64 128,128 256,256" bash bench/sdpa/run_microkernels_remote.sh
#
#     # 只跑特定 impl（逗号分隔）：
#     IMPLS=baseline,qk_ublock4,pquad bash bench/sdpa/run_microkernels_remote.sh
#
#     # 改绑核：
#     CORE=160 bash bench/sdpa/run_microkernels_remote.sh
#
#     # 不绑核（云上 / 没 taskset 的环境）：
#     CORE=none bash bench/sdpa/run_microkernels_remote.sh
#
#     # 增大迭代数（提高时间分辨率）：
#     ITERS=200000 WARMUP=2000 bash bench/sdpa/run_microkernels_remote.sh

set -euo pipefail

# ── 仓库根：脚本位于 <repo>/bench/sdpa/，向上两级 ────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── 可调参数（环境变量覆盖） ────────────────────────────────────────
CORE="${CORE:-0}"                     # 单核绑定；设 "none" 关闭 taskset
SHAPES="${SHAPES:-128,128 192,192}"   # "E,Sk E,Sk ..."
DTYPES="${DTYPES:-fp32,bf16}"
IMPLS="${IMPLS:-}"                    # 空 = 全部 registered impls
ITERS="${ITERS:-50000}"               # bench iterations
WARMUP="${WARMUP:-500}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/fused_cpp/bench/sdpa}"
TS="$(date +%Y%m%d_%H%M%S)"

# ── 单线程环境：剥离 OMP / BLAS 干扰 ───────────────────────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# 让 OMP 即使被 PyTorch import 早就初始化好了，也不要 dynamic 地去抢核。
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND="${OMP_PROC_BIND:-true}"

# ── 选择运行命令前缀（taskset / 直接 python） ─────────────────────
if [[ "${CORE}" == "none" ]]; then
    RUN_PREFIX=()
elif command -v taskset >/dev/null 2>&1; then
    RUN_PREFIX=(taskset -c "${CORE}")
else
    echo "WARN: taskset not found on this host; running unbound. " \
         "Pass CORE=none to silence this warning." >&2
    RUN_PREFIX=()
fi

# ── 自检：fused_cpp._C 已构建并 import 成功 ─────────────────────────
python -c "
from fused_cpp import _C
impls = list(_C.list_microkernel_impls())
print('[check] available impls =', impls)
print('[check] has_openmp =', _C.has_openmp())
"

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "fused_cpp microkernel bench (remote)"
echo "  core(s)     : ${CORE}     (taskset)"
echo "  shapes      : ${SHAPES}"
echo "  dtypes      : ${DTYPES}"
echo "  impls       : ${IMPLS:-<all>}"
echo "  iters/warmup: ${ITERS} / ${WARMUP}"
echo "  output dir  : ${OUTPUT_DIR}"
echo "============================================================"

# ── 主循环：对每个 (E, Sk) 跑一份 ────────────────────────────────────
for shape in ${SHAPES}; do
    E="${shape%,*}"
    Sk="${shape#*,}"
    out_base="${OUTPUT_DIR}/microkernels_${TS}_E${E}_Sk${Sk}"
    log_path="${out_base}.log"
    json_path="${out_base}.json"

    echo
    echo "── shape E=${E} Sk=${Sk} ──────────────────────────────────"

    cmd=("${RUN_PREFIX[@]}" python fused_cpp/bench/sdpa/bench_microkernels.py
         --E "${E}" --Sk "${Sk}"
         --dtypes "${DTYPES}"
         --iters "${ITERS}" --warmup "${WARMUP}"
         --json "${json_path}")
    if [[ -n "${IMPLS}" ]]; then
        cmd+=(--impls "${IMPLS}")
    fi

    echo "+ ${cmd[*]}" | tee "${log_path}"
    "${cmd[@]}" 2>&1 | tee -a "${log_path}"
    echo "wrote: ${log_path}"
    echo "wrote: ${json_path}"
done

echo
echo "============================================================"
echo "DONE. logs/json under: ${OUTPUT_DIR}"
ls -1 "${OUTPUT_DIR}"/microkernels_${TS}_*.{log,json} 2>/dev/null || true
