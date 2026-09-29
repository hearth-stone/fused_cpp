# DeepSeek V4 CPU 交付版：安装、测试与结果对比

本文从空目录在 `final_test` 下安装一套独立的交付环境：一份 vLLM、一份 AOT `fused_cpp`、一个 Python 虚拟环境，以及独立的 MMLU 评测环境和结果。原有部署环境保留在原目录，便于与交付环境比较。随后验证内核、运行少量 MMLU 题目，并说明如何沿用现有脚本做性能测试。命令面向 Linux AArch64；模型服务和性能测试示例使用 `Arm-codex-internal` 的 320 核拓扑。以下命令在同一个 Bash shell 中按顺序执行。

本次交付对应：

| 仓库 | 分支 | 本次验证使用的源码提交 |
| --- | --- | --- |
| `hearth-stone/vllm-aarch64-opt` | `dsv4-arm-cpu-v0.28.0` | `549a522c9dca299164ea48c4c7fa533081f7b795` |
| `hearth-stone/fused_cpp` | `feat/fused_cpp/dsv4_v028_aot_delivery` | `c4333b769e25a4dcc3e29da12c64d810002a52d4` |

文档提交可能比表中的源码提交更新。克隆后用 `git rev-parse HEAD` 记录实际版本，再运行第 3 节的测试。

## 1. 系统与仓库

需要 GCC/G++ **12.3 或更新版本**、CMake、Ninja、NUMA 开发库、jemalloc、Git、Git LFS、curl 和支持 BF16、I8MM、SVE 的 AArch64 Linux 主机。以常见发行版为例：

```bash
# openEuler
sudo dnf install -y gcc gcc-c++ cmake ninja-build numactl numactl-devel \
  jemalloc git git-lfs curl

# Ubuntu：仅在 Ubuntu 上执行这一组
sudo apt-get update
sudo apt-get install -y gcc g++ cmake ninja-build numactl libnuma-dev \
  libjemalloc2 libjemalloc-dev git git-lfs curl

gcc --version
git lfs version
```

在 `final_test` 下克隆两个交付仓库。`Arm-codex-internal` 原有的 `/home/zhangxu/codex/` 环境不参与这些安装命令：

```bash
export FINAL_TEST="$HOME/final_test"
mkdir -p "$FINAL_TEST"

git clone -b dsv4-arm-cpu-v0.28.0 --single-branch \
  https://github.com/hearth-stone/vllm-aarch64-opt.git \
  "$FINAL_TEST/vllm-aarch64-opt"
git clone -b feat/fused_cpp/dsv4_v028_aot_delivery --single-branch \
  https://github.com/hearth-stone/fused_cpp.git "$FINAL_TEST/fused_cpp"

export VLLM_ROOT="$FINAL_TEST/vllm-aarch64-opt"
export FUSED_ROOT="$FINAL_TEST/fused_cpp"
git -C "$VLLM_ROOT" rev-parse HEAD
git -C "$FUSED_ROOT" rev-parse HEAD
```

如果目标机无法访问 GitHub，可以在联网机器克隆上述分支，再把**完整仓库目录**放到目标机的 `final_test` 下；保留 `.git`，以便记录版本。模型权重和后文的 MMLU 数据集也可以在联网机器准备后放到指定目录。本文不依赖目标机直接下载这些文件。

## 2. 安装 vLLM 与 fused_cpp

两个仓库共用 `final_test/test` 这个 Python 3.12 环境。已经安装 `uv` 时，跳过前两行安装与加载命令：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv python install 3.12
uv venv --python 3.12 "$FINAL_TEST/test"
source "$FINAL_TEST/test/bin/activate"
export VLLM_PYTHON="$FINAL_TEST/test/bin/python"

cd "$VLLM_ROOT"
uv pip install -r requirements/build/cpu.txt --torch-backend cpu \
  --index-strategy unsafe-best-match \
  --default-index https://mirrors.aliyun.com/pypi/simple/
uv pip install -r requirements/cpu.txt --torch-backend cpu \
  --index-strategy unsafe-best-match \
  --default-index https://mirrors.aliyun.com/pypi/simple/
VLLM_TARGET_DEVICE=cpu MAX_JOBS=16 uv pip install -e . --no-build-isolation
uv pip uninstall torchcodec

cd "$FUSED_ROOT"
FUSED_CPP_SVE_VECTOR_BITS=256 MAX_JOBS=16 \
  uv pip install -e . --no-build-isolation --no-deps
uv pip install pytest tblib
```

在 `Arm-codex-internal` 上，CPU requirements 会装入需要 CUDA 动态库的 `torchcodec` wheel，导致文本服务在导入时因缺少 `libnvrtc.so.13` 退出。这里的评测只用文本，移除该包后服务导入已验证通过。

这里的 `FUSED_CPP_SVE_VECTOR_BITS=256` 选择**构建时**的 SVE 长度；仓库中已经包含 128/256 位的生成汇编，安装和运行都不需要 Xbyak/JIT 源码。运行进程的 SVE 长度必须与构建长度相同。验证加载的源码路径：

```bash
PYTHONPATH="$FUSED_ROOT/src:$VLLM_ROOT" "$VLLM_PYTHON" -c \
  'import fused_cpp, vllm; print(fused_cpp.__file__); print(vllm.__file__)'
```

## 3. 运行代码测试

在 256 位 SVE 主机上，运行 `fused_cpp` 的全部测试以及 vLLM 的 DeepSeek V4 CPU 专项测试：

```bash
cd "$FUSED_ROOT"
PYTHONPATH="$FUSED_ROOT/src:$VLLM_ROOT" \
  "$VLLM_PYTHON" -m pytest -q -rs

# vLLM 专项测试从仓库内的 model_configs 路径读取这份配置。
mkdir -p "$VLLM_ROOT/model_configs/DeepSeek-V4-Flash-BF16"
cp /mnt/models/DeepSeek-V4-Flash-BF16/config.json \
  "$VLLM_ROOT/model_configs/DeepSeek-V4-Flash-BF16/config.json"
cd "$VLLM_ROOT"
VLLM_TARGET_DEVICE=cpu PYTHONPATH="$FUSED_ROOT/src:$VLLM_ROOT" \
  "$VLLM_PYTHON" -m pytest -q -rs \
  tests/models/test_deepseek_v4_cpu_v028.py
```

`-rs` 会列出跳过原因。本次 `final_test` 验证得到 `fused_cpp: 1577 passed, 4 skipped`、`vLLM: 30 passed`。四个跳过中，一个用例专门检查 native backend 不可用的情况，另外三个 shared-MLP 用例在全套运行时因进程 CPU affinity 少于 8 核而跳过；该 MLP 文件单独运行得到 `8 passed`。其他机器以实际输出为准。

若部署机器的 SVE 长度是 128 位，先重建扩展，再在那台机器运行全套测试：

```bash
cd "$FUSED_ROOT"
rm -rf "$FUSED_ROOT/build"
FUSED_CPP_SVE_VECTOR_BITS=128 MAX_JOBS=16 \
  uv pip install -e . --no-build-isolation --no-deps --reinstall
PYTHONPATH="$FUSED_ROOT/src:$VLLM_ROOT" "$VLLM_PYTHON" -m pytest -q -rs
```

不要在 256 位机器上仅用 `prctl` 把进程切到 128 位后，将全套 pytest 的结果当作 VL128 正确性结论：本机的 Torch 2.13 CPU wheel 在此配置下连 `torch.matmul` 的参考值都会算错。本次用独立标量求和核对 VL128 的 42 个 BF16 linear 输出，最大绝对误差为 `1.5e-8`；i8gemm 直接写入边界的 4 个用例也通过。全套 VL128 测试需在 PyTorch 参考算子正确的 128 位环境完成。回到 VL256 时，删除 `build` 后以 `FUSED_CPP_SVE_VECTOR_BITS=256` 重新安装；不要在同一构建产物上只切换运行时 SVE 长度。

## 4. 单独安装 MMLU 的 Python 环境

MMLU 使用独立的 `lm-evaluation-harness` 环境，通过 vLLM 的 HTTP 服务评分；它不安装到上节的 vLLM 环境。目标机还需有 BF16 或 INT8 模型权重：

```text
/mnt/models/DeepSeek-V4-Flash-BF16/
/mnt/models/DeepSeek-V4-Flash-INT8/
```

模型权重可以作为两套环境共用的只读输入；代码、Python 环境、profile 和测试结果全部放在 `final_test` 下。下面固定评测器和数据集的提交，使新旧结果使用同一套 5-shot 题目：

```bash
export MMLU_WORKDIR="$FINAL_TEST/mmlu"
export HARNESS_DIR="$MMLU_WORKDIR/lm-evaluation-harness"
export DATASET_DIR="$MMLU_WORKDIR/cais-mmlu"
export TASK_DIR="$MMLU_WORKDIR/tasks/mmlu-local"
mkdir -p "$MMLU_WORKDIR"

git clone https://github.com/EleutherAI/lm-evaluation-harness.git "$HARNESS_DIR"
git -C "$HARNESS_DIR" checkout c1c4bea3777f73e188395264083adcf454913344
uv venv --python 3.12 "$MMLU_WORKDIR/.venv"
export MMLU_PYTHON="$MMLU_WORKDIR/.venv/bin/python"
uv pip install --python "$MMLU_PYTHON" -e "${HARNESS_DIR}[api]"

GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://hf-mirror.com/datasets/cais/mmlu "$DATASET_DIR"
GIT_LFS_SKIP_SMUDGE=1 git -C "$DATASET_DIR" checkout \
  c30699e8356da336a370243923dbaf21066bb9fe
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/download_dataset.py" \
  --dataset-dir "$DATASET_DIR" \
  --hf-cli "$MMLU_WORKDIR/.venv/bin/hf"
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_dataset.py" \
  --dataset-dir "$DATASET_DIR"
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/prepare_tasks.py" \
  --harness-dir "$HARNESS_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --task-dir "$TASK_DIR"
```

`check_dataset.py` 应报告 176 个 Parquet 文件，并能离线加载 `abstract_algebra`。如果目标机无法克隆评测器或数据集，在联网机器按相同提交准备完整目录后放到 `$HARNESS_DIR`、`$DATASET_DIR`；数据集需要实际 Parquet 文件，只有 Git LFS 指针不够。随后仍执行 `check_dataset.py` 与 `prepare_tasks.py`。这里使用 `uv pip --python "$MMLU_PYTHON"`，是为了即使 vLLM 环境仍处于激活状态，也把依赖装进单独的 MMLU 环境。

## 5. 跑少量 MMLU 题目

以下以 BF16 为例。服务使用目标机的 NUMA 4/5/6/7、每 rank 40 核、端口 8004；每个节点的 CPU 分别为 `160-199`、`200-239`、`240-279`、`280-319`。先在 `final_test` 下生成四份 rank-local MoE planner profile。BF16 和 INT8 可以共用这四份 profile：

```bash
export VLLM_ROOT="$FINAL_TEST/vllm-aarch64-opt"
export FUSED_ROOT="$FINAL_TEST/fused_cpp"
export VLLM_PYTHON="$FINAL_TEST/test/bin/python"
export PROFILE_DIR="$FINAL_TEST/profiles/mmlu40"
mkdir -p "$PROFILE_DIR"
for rank in 0 1 2 3; do
  PYTHONPATH="$FUSED_ROOT/src:$VLLM_ROOT" OMP_NUM_THREADS=40 \
    "$VLLM_PYTHON" - "$rank" "$PROFILE_DIR/rank${rank}.json" <<'PY'
import sys
from fused_cpp import moe

rank = int(sys.argv[1])
first_cpu = 160 + rank * 40
result = moe.calibrate_moe_planner_quick(
    tuple(range(first_cpu, first_cpu + 40)), output=sys.argv[2]
)
print(result.output_path, result.elapsed_seconds)
PY
done
```

干净克隆的 vLLM 分支需要显式设置这些 profile；仓库中的 `run_numa_4567_*_autocalib.sh` 会清掉该变量，因此这里直接调用同一个 API server 入口。先启动 BF16 服务：

```bash
export FUSED_CPP_SRC="$FUSED_ROOT/src"
export MODEL_ID=dsv4-flash-bf16
export MODEL_PATH=/mnt/models/DeepSeek-V4-Flash-BF16/
SERVE_QUANT=()
export RUN_DIR="$FINAL_TEST/results/bf16-$(date -u +%Y%m%dT%H%M%SZ)"
export AOT_BF16_RUN_DIR="$RUN_DIR"
mkdir -p "$RUN_DIR"

export PYTHONPATH="$FUSED_CPP_SRC:$VLLM_ROOT"
export LD_PRELOAD=/usr/lib64/libjemalloc.so
export MALLOC_CONF=thp:always,oversize_threshold:2097152,background_thread:true
export GLOO_DEVICE_TRANSPORT=TCP GLOO_SOCKET_IFNAME=eno1
export VLLM_CPU_KVCACHE_SPACE=10 VLLM_TARGET_DEVICE=cpu
export VLLM_CPU_FUSED_CPP_STRICT=1
export VLLM_CPU_OMP_THREADS_BIND='160-199|200-239|240-279|280-319'
export OMP_NUM_THREADS=40 MKL_NUM_THREADS=40 NUMEXPR_MAX_THREADS=40
export OPENBLAS_NUM_THREADS=40 VECLIB_MAXIMUM_THREADS=40 GOTO_NUM_THREADS=40
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000
export FUSED_CPP_MOE_PLANNER_PROFILE="$PROFILE_DIR/rank{local_rank}.json"

cd "$VLLM_ROOT"
nohup "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --host 127.0.0.1 --port 8004 \
  --gpu-memory-utilization 0.95 --max-model-len 4096 \
  --tensor-parallel-size 4 --dtype bfloat16 --trust-remote-code \
  --numa-bind --numa-bind-nodes 4 5 6 7 \
  --numa-bind-cpus 160-199 200-239 240-279 280-319 \
  --served-model-name "$MODEL_ID" \
  --enable-tokenizer-info-endpoint \
  --no-enable-prefix-caching \
  "${SERVE_QUANT[@]}" \
  > "$RUN_DIR/server.log" 2>&1 < /dev/null &
echo $! > "$RUN_DIR/server.pid"

until curl -fsS -o /dev/null http://127.0.0.1:8004/v1/models; do sleep 10; done
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/check_api.py" \
  --model "$MODEL_ID" --output-dir "$RUN_DIR"
"$MMLU_PYTHON" "$VLLM_ROOT/scripts/mmlu/run_eval.py" \
  --lm-eval "$MMLU_WORKDIR/.venv/bin/lm-eval" \
  --harness-dir "$HARNESS_DIR" \
  --dataset-dir "$DATASET_DIR" \
  --task-dir "$TASK_DIR" \
  --model "$MODEL_ID" \
  --output-dir "$RUN_DIR" \
  --mode smoke
```

`smoke` 模式只评测 `abstract_algebra` 的前 **2 道测试题**，使用 5-shot 示例。它应生成 `$RUN_DIR/smoke/<模型名>/results_*.json` 和 `samples_*.jsonl`，前者包含 `acc,none`，后者包含逐题的 `doc_id`、`target`、`filtered_resps` 与 `acc`。`$RUN_DIR` 下还有 `smoke.log`、`server.log`、接口响应与两个提交号。两道题只适合检查流程和逐题回归，不能当作完整 MMLU 精度结论。跑完后停止服务：

```bash
kill "$(cat "$RUN_DIR/server.pid")"
```

测试 INT8 时，先设置另一组参数，再重复从 `cd "$VLLM_ROOT"` 开始的服务启动、`check_api.py`、`run_eval.py` 和停止服务命令。保留 `AOT_BF16_RUN_DIR` 的值，供后面对比 BF16 结果：

```bash
export MODEL_ID=dsv4-flash-int8
export MODEL_PATH=/mnt/models/DeepSeek-V4-Flash-INT8/
SERVE_QUANT=(--quantization compressed-tensors)
export RUN_DIR="$FINAL_TEST/results/int8-$(date -u +%Y%m%dT%H%M%SZ)"
export AOT_INT8_RUN_DIR="$RUN_DIR"
mkdir -p "$RUN_DIR"
```

本次 `final_test` 验证中，BF16 和 INT8 都是 `2/2`；与各自的历史结果相比，两题的正确性及八个选项分数逐项相同。新结果目录分别是 `results/bf16-20260929T163733Z` 和 `results/int8-20260929T164856Z`。这两题只验证安装、服务与逐题回归，不能作为完整精度结论。完整评测的命令和输出说明见 vLLM 仓库的 `DEEPSEEK_V4_CPU_MMLU.md`，本次交付先不跑完整 MMLU。

上述命令在 openEuler 上使用 `/usr/lib64/libjemalloc.so`，并将 Gloo 网卡设为 `eno1`。Ubuntu 或其他机器需要改成实际的 jemalloc 路径与网卡名；NUMA 绑定也必须对应实际拓扑。

## 6. 与已有结果对比

在 `Arm-codex-internal` 上，已有的小样本结果原始位置分别是：

```text
BF16: /home/zhangxu/dsv4-mmlu/results/bf16-20260929T094329Z
INT8: /home/zhangxu/dsv4-mmlu/results/int8-20260929T095730Z
```

将需要对比的已有结果放在 `$FINAL_TEST/reference_results` 下；原有部署环境及结果保持原状。`REF_RESULT` 指向具体一次参考运行，`NEW_DIR` 指向本次运行。先比较 `harness.commit`、`dataset.commit`，再比较同一题的正确性与得分：

```bash
export REF_RESULT="$FINAL_TEST/reference_results/bf16-20260929T094329Z"
export NEW_DIR="$AOT_BF16_RUN_DIR"
cat "$REF_RESULT/harness.commit" "$NEW_DIR/harness.commit"
cat "$REF_RESULT/dataset.commit" "$NEW_DIR/dataset.commit"

"$MMLU_PYTHON" - "$REF_RESULT" "$NEW_DIR" <<'PY'
import json
import sys
from pathlib import Path

def read_run(root):
    root = Path(root)
    result = json.loads(next(root.glob("smoke/*/results_*.json")).read_text())
    samples = [json.loads(line) for line in next(root.glob("smoke/*/samples_*.jsonl")).open()]
    return result["results"]["mmlu_abstract_algebra"]["acc,none"], {
        row["doc_id"]: row for row in samples
    }

old_acc, old = read_run(sys.argv[1])
new_acc, new = read_run(sys.argv[2])
assert old.keys() == new.keys()
print("aggregate acc:", old_acc, "->", new_acc)
for doc_id in sorted(old):
    assert old[doc_id]["doc_hash"] == new[doc_id]["doc_hash"]
    print("doc", doc_id, "correct:", old[doc_id]["acc"], "->", new[doc_id]["acc"])
    print("  scores:", [x[0] for x in old[doc_id]["filtered_resps"]],
          "->", [x[0] for x in new[doc_id]["filtered_resps"]])
PY
```

BF16 只与 BF16 参考比，INT8 只与 INT8 参考比。对比 INT8 时，将 `REF_RESULT` 改为 `$FINAL_TEST/reference_results/int8-20260929T095730Z`，`NEW_DIR` 改为 `$AOT_INT8_RUN_DIR`，重复上面的对比命令。先核对同一 `doc_hash` 和数据集/评测器提交，再看逐题结果；两题的汇总准确率相同仍可能掩盖分数变化。

## 7. 性能测试：沿用现有 runner，暂不执行

现有脚本是 vLLM 仓库的 `tests/v1/e2e/generation_sampling/generation_prefill_suite.py`，旧运行协议见 `DEEPSEEK_V4_CPU_EXPERIMENTS.md`。它读取固定的 2048-token `cases31.jsonl`，先做一条 warmup，再测若干条独立请求；输出 `result.jsonl`、`result.jsonl.summary.json`。下列命令只是为后续测试准备，**本次没有运行性能测试，也没有新的性能结论**。

下面是 `Arm-codex-internal` 上原来的 TP4、每 rank 80 核、warmup + 3 条的快速命令。`CASES_SRC` 是已有输入文件；其他机器应提供相同内容的文件，并按实际 CPU/NUMA 拓扑调整绑定。80 核的绑定与上节 MMLU 的 40 核绑定不同，因此要为性能测试单独准备 profile；这是服务启动所需的校准，不是本次执行性能测试：

```bash
export PERF_PROFILE_DIR="$FINAL_TEST/profiles/perf80"
mkdir -p "$PERF_PROFILE_DIR"
for rank in 0 1 2 3; do
  PYTHONPATH="$FUSED_ROOT/src:$VLLM_ROOT" OMP_NUM_THREADS=80 \
    "$VLLM_PYTHON" - "$rank" "$PERF_PROFILE_DIR/rank${rank}.json" <<'PY'
import sys
from fused_cpp import moe

rank = int(sys.argv[1])
first_cpu = rank * 80
result = moe.calibrate_moe_planner_quick(
    tuple(range(first_cpu, first_cpu + 80)), output=sys.argv[2]
)
print(result.output_path, result.elapsed_seconds)
PY
done
```

随后用下面的 runner；不读取旧版 fused_cpp 的 prepack cache 或 planner profile：

```bash
export CASES_SRC=/mnt/models/.cache/zhangxu/benchmarks/fused-attn80-continuous30-20260821/cases31.jsonl
export OUT="$FINAL_TEST/perf/bf16-quick"
export MODEL=/mnt/models/DeepSeek-V4-Flash-BF16
export RUNNER="$VLLM_ROOT/tests/v1/e2e/generation_sampling/generation_prefill_suite.py"
mkdir -p "$OUT"
head -n 4 "$CASES_SRC" > "$OUT/cases4.jsonl"

cd "$VLLM_ROOT"
env VLLM_TARGET_DEVICE=cpu \
  LD_LIBRARY_PATH=/opt/llvm-22/lib/aarch64-unknown-linux-gnu \
  LD_PRELOAD=/opt/llvm-22/lib/aarch64-unknown-linux-gnu/libomp.so:/usr/lib64/libjemalloc.so \
  MALLOC_CONF=thp:always,oversize_threshold:2097152,background_thread:true \
  GLOO_DEVICE_TRANSPORT=TCP GLOO_SOCKET_IFNAME=eno1 \
  VLLM_CPU_KVCACHE_SPACE=5 \
  'VLLM_CPU_OMP_THREADS_BIND=0-79|80-159|160-239|240-319' \
  VLLM_CPU_NUM_OF_RESERVED_CPU=0 \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=36000 \
  VLLM_CPU_FUSED_CPP_STRICT=1 \
  FUSED_CPP_MOE_PLANNER_PROFILE="$PERF_PROFILE_DIR/rank{local_rank}.json" \
  FUSED_CPP_MOE_HIERARCHICAL_N_SPLIT=0 \
  VLLM_CPU_MOE_PREPACK_PREFAULT=1 \
  OMP_NUM_THREADS=80 \
  PYTHONPATH="$FUSED_ROOT/src:$VLLM_ROOT" \
  "$VLLM_PYTHON" "$RUNNER" \
    --model "$MODEL" \
    --cases "$OUT/cases4.jsonl" \
    --output "$OUT/result.jsonl" \
    --tensor-parallel-size 4 \
    --max-model-len 2100 \
    --block-size 256 \
    --kv-cache-memory-bytes 5368709120 \
    --numa-bind-cpus '0-79|80-159|160-239|240-319' \
    --numa-bind-nodes 0,1,2,3 \
    --expected-prompt-tokens 2048 \
    --min-measured-cases 3 \
    --warmup-case-index 0 --exclude-warmup-case \
    --request-interval-seconds 10 --max-tokens 1 \
  2>&1 | tee "$OUT/run.log"
```

INT8 使用相同命令，只修改 `MODEL=/mnt/models/DeepSeek-V4-Flash-INT8` 与独立的 `OUT`。运行前应重新构建/安装 VL256 的 `fused_cpp`，并确认机器空闲、模型文件和 `cases31.jsonl` 都存在。正式性能对比沿用旧协议：候选与基线各运行 **3 个独立 engine**，每次 1 条 warmup + 5 条 measured，请求间隔 10 秒，取 3 次 engine 平均值的中位数；基线也要在同一机器、同一配置下重跑。每次用新的 `OUT`，将 `head -n 4` 改为 `head -n 6`、`cases4.jsonl` 改为 `cases6.jsonl`、`--min-measured-cases 3` 改为 `5`。

原有部署环境也用同一 runner、输入和机器配置运行；新旧环境分别保存结果。快速检查新旧文件时，逐条对照 `case_id` 与 `generated_token_ids`，再查看 `duration_s` 和 `result.jsonl.summary.json`。已有 BF16 示例文件原始位置是 `/home/zhangxu/codex/dsv4-v028-results/bf16-quick-codex-20260827/result.jsonl`；它可供检查文件结构和 token 输出，不能替代同条件重测的性能基线。把要比较的文件放到 `$FINAL_TEST/reference_results/perf/` 下：

```bash
export REF_JSONL="$FINAL_TEST/reference_results/perf/bf16-quick/result.jsonl"
export NEW_JSONL="$OUT/result.jsonl"
"$VLLM_PYTHON" - "$REF_JSONL" "$NEW_JSONL" <<'PY'
import json
import statistics
import sys

def rows(path):
    with open(path) as stream:
        return {row["case_id"]: row for row in map(json.loads, stream)}

old, new = rows(sys.argv[1]), rows(sys.argv[2])
assert old.keys() == new.keys()
for case_id in sorted(old):
    print(case_id, old[case_id]["generated_token_ids"],
          "->", new[case_id]["generated_token_ids"])
print("mean seconds:",
      statistics.mean(row["duration_s"] for row in old.values()), "->",
      statistics.mean(row["duration_s"] for row in new.values()))
PY
```
