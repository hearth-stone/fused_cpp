# -*- coding: utf-8 -*-
"""共享 pytest fixtures 与工具。

当前提供：
  * :func:`assert_tensor_close` —— 输出 max_abs / max_rel / cosine /
    NaN-Inf 四项指标的张量等价性断言（替代 :func:`torch.testing.assert_close`），
    在 SDPA 多版本框架的等价性测试中使用。
  * :data:`SDPA_TOLERANCE` —— 默认容差表（按 dtype 区分），符合
    ``python_test.md`` 第 11.2 节标准 + reduce 算子放宽一档的约定。
  * ``--sdpa-versions`` / ``--sdpa-tags`` CLI 选项 + ``pytest_generate_tests``
    钩子：在任何测试/基准函数参数中出现 ``sdpa_version`` 名称时，自动
    从 :func:`fused_cpp.available_sdpa_versions` 读取注册表并按过滤后的
    ``VersionInfo`` 参数化。
预留扩展点：
  * 任务 9 会在本文件中追加 ``--sdpa-thread-sweep`` 等 CLI 选项。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn.functional as F

from fused_cpp.sdpa_registry import (
    VersionInfo,
    available_sdpa_versions,
)


# ── 默认 SDPA 容差表 ──────────────────────────────────────────────────────
#
# 来自 ``python_test.md`` 第 11.2 节标准。SDPA 含 softmax 这一 reduce 算子，
# 因此 atol 在标准基础上放宽一档，rtol 保持。具体：
#   fp32:  atol 1e-5  rtol 1e-5  cos >= 0.9999
#   bf16:  atol 5e-2  rtol 5e-2  cos >= 0.999
@dataclass(frozen=True)
class _DTypeTol:
    atol: float
    rtol: float
    cos_min: float


SDPA_TOLERANCE = {
    torch.float32: _DTypeTol(atol=1e-5, rtol=1e-5, cos_min=0.9999),
    torch.float64: _DTypeTol(atol=1e-7, rtol=1e-7, cos_min=0.999999),
    torch.bfloat16: _DTypeTol(atol=5e-2, rtol=5e-2, cos_min=0.999),
    torch.float16:  _DTypeTol(atol=1e-2, rtol=1e-2, cos_min=0.999),
}


# ── 等价性断言工具 ────────────────────────────────────────────────────────


def _err_metrics(
    actual: torch.Tensor, ref: torch.Tensor
) -> Tuple[float, float, float, int, int]:
    """计算 max_abs / max_rel / cosine / nan_count / inf_count。"""
    a = actual.detach().float()
    b = ref.detach().float()
    diff = (a - b).abs()
    max_abs = float(diff.max().item()) if diff.numel() > 0 else 0.0
    denom = b.abs().clamp(min=1e-12)
    max_rel = float((diff / denom).max().item()) if diff.numel() > 0 else 0.0
    cos = float(
        F.cosine_similarity(
            a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
        ).item()
    ) if a.numel() > 0 else 1.0
    nan_count = int(torch.isnan(a).sum().item())
    inf_count = int(torch.isinf(a).sum().item())
    return max_abs, max_rel, cos, nan_count, inf_count


def assert_tensor_close(
    actual: torch.Tensor,
    ref: torch.Tensor,
    *,
    dtype: Optional[torch.dtype] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    cos_min: Optional[float] = None,
    context: str = "",
) -> dict:
    """断言两个张量在数值上近似相等，并输出全套误差指标。

    判定流程：
      1. ``actual`` 中若包含 NaN / Inf → 直接 fail；
      2. ``cosine >= cos_min``（结构相似度）；
      3. ``max_abs <= atol + rtol * |ref|``（按通用容差表，放宽一档以适配
         softmax 等 reduce 算子）。

    :param actual: 待检查张量。
    :param ref: 参考张量。
    :param dtype: 用于查 :data:`SDPA_TOLERANCE` 的 dtype；不传时取
        ``actual.dtype``。
    :param atol: 显式覆盖默认 atol。
    :param rtol: 显式覆盖默认 rtol。
    :param cos_min: 显式覆盖默认 cos 下限。
    :param context: 失败信息中附加的上下文（如 ``"version=naive shape=..."``）。
    :return: 误差指标 dict（``max_abs / max_rel / cosine / nan_count /
        inf_count / atol / rtol / cos_min``）。

    :raises AssertionError: 任意一项检查失败。
    """
    if actual.shape != ref.shape:
        raise AssertionError(
            f"shape mismatch: actual={tuple(actual.shape)} vs "
            f"ref={tuple(ref.shape)} ({context})"
        )

    # 选择默认容差
    eff_dtype = dtype if dtype is not None else actual.dtype
    tol = SDPA_TOLERANCE.get(eff_dtype, SDPA_TOLERANCE[torch.float32])
    eff_atol = atol if atol is not None else tol.atol
    eff_rtol = rtol if rtol is not None else tol.rtol
    eff_cos = cos_min if cos_min is not None else tol.cos_min

    max_abs, max_rel, cos, nan_count, inf_count = _err_metrics(actual, ref)
    metrics = {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "cosine": cos,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "atol": eff_atol,
        "rtol": eff_rtol,
        "cos_min": eff_cos,
    }

    # 1) NaN / Inf 检查
    if nan_count > 0 or inf_count > 0:
        # 找到首个 NaN/Inf 的索引便于定位
        bad = ~torch.isfinite(actual.detach())
        idx = bad.nonzero(as_tuple=False)
        first = tuple(int(x) for x in idx[0].tolist()) if idx.numel() > 0 else None
        raise AssertionError(
            f"actual contains non-finite values: nan={nan_count} inf={inf_count} "
            f"first_index={first} ({context})\n"
            f"  metrics={metrics}"
        )

    # 2) Cosine
    if cos < eff_cos:
        raise AssertionError(
            f"cosine similarity too low: cos={cos:.6f} < {eff_cos} ({context})\n"
            f"  metrics={metrics}"
        )

    # 3) max_abs
    ref_abs_max = float(ref.detach().float().abs().max().item()) if ref.numel() > 0 else 0.0
    elementwise_bound = eff_atol + eff_rtol * ref_abs_max
    if max_abs > elementwise_bound:
        # 找到误差最大点的索引便于定位
        diff = (actual.detach().float() - ref.detach().float()).abs()
        argmax_flat = int(diff.argmax().item())
        bad_idx = tuple(int(x) for x in
                        torch.unravel_index(torch.tensor(argmax_flat), diff.shape))
        raise AssertionError(
            f"max_abs={max_abs:.3e} exceeds bound atol+rtol*|ref|_max={elementwise_bound:.3e} "
            f"at index {bad_idx} ({context})\n"
            f"  metrics={metrics}"
        )

    return metrics


# ── pytest CLI 选项：按名称 / 标签过滤 SDPA 版本 ──────────────────


def pytest_addoption(parser):  # noqa: D401
    """追加 SDPA 版本过滤选项。如多个 conftest 重复注册会报错，均市场仅这一个。"""
    group = parser.getgroup("fused-cpp sdpa")
    group.addoption(
        "--sdpa-versions",
        action="store",
        default=None,
        help=(
            "Comma-separated whitelist of SDPA version names to expand in the "
            "`sdpa_version` test parameter (e.g. 'naive,flash2'). "
            "If unset, all registered versions are used."
        ),
    )
    group.addoption(
        "--sdpa-tags",
        action="store",
        default=None,
        help=(
            "Comma-separated tag whitelist; only versions whose `tags` "
            "intersect this set are expanded."
        ),
    )
    group.addoption(
        "--sdpa-thread-sweep",
        action="store",
        default=None,
        help=(
            "Comma-separated thread counts (e.g. '1,2,4,8') for the SDPA "
            "benchmark's thread-sweep mode. When set, the benchmark file "
            "spawns one child pytest invocation per N with "
            "OMP_NUM_THREADS=N and aggregates the per-thread CSV/JSON files. "
            "Setting this option only takes effect inside `bench_sdpa_versions"
            ".py`; standard equiv tests ignore it."
        ),
    )
    group.addoption(
        "--sdpa-num-threads",
        action="store",
        default=None,
        type=int,
        help=(
            "Force the SDPA benchmark to run with exactly N threads. "
            "This option *strictly* pins the thread count by:\n"
            "  (1) calling torch.set_num_threads(N) so the `pytorch_sdpa` "
            "      baseline is bound to N intra-op threads (Linux PyTorch "
            "      otherwise picks #cpus and silently ignores OMP_NUM_THREADS "
            "      after import);\n"
            "  (2) setting OMP_NUM_THREADS / MKL_NUM_THREADS / "
            "      OPENBLAS_NUM_THREADS / VECLIB_MAXIMUM_THREADS / "
            "      BLIS_NUM_THREADS / NUMEXPR_NUM_THREADS = N in the env "
            "      so any subsequently-imported BLAS / OpenMP runtime "
            "      observes the same limit.\n"
            "When unset, falls back to OMP_NUM_THREADS env var, otherwise "
            "the runtime defaults are kept (no forced pinning)."
        ),
    )
    group.addoption(
        "--sdpa-bench-output-dir",
        action="store",
        default=None,
        help=(
            "Directory where the SDPA benchmark CSV/JSON files are written. "
            "Defaults to '<repo>/bench/sdpa/'."
        ),
    )


# ── 线程数钉住的公共逻辑 ─────────────────────────────────────────────

# 不依赖上游使用者说不改：Linux PyTorch 默认会拿 #cpus 作为 intra-op 线程
# 数，并在 `import torch` 后才占位，之后仅调整 OMP_NUM_THREADS 不能反向
# 压低 PyTorch 的线程池。要严格钉住线程数，必须明确调
# `torch.set_num_threads(N)` 。
_BLAS_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _apply_thread_pin(n_threads, *, set_env: bool = True, verbose: bool = False):
    """将 PyTorch / OpenMP / BLAS 线程数严格钉为 ``n_threads``。

    在 Linux 上 PyTorch 默认会以 ``num_cpus`` 初始化 intra-op 线程池，
    且该初始化**发生在 import 期**；后续仅修改 ``OMP_NUM_THREADS`` 环境变
    量对 PyTorch 的 ``F.scaled_dot_product_attention`` 等路径**不起作用**。
    要严格钉住线程数必须调 ``torch.set_num_threads`` 与
    ``torch.set_num_interop_threads``。

    :param n_threads: 目标线程数，需 >= 1。None 为 no-op。
    :param set_env: 是否同时设置 BLAS / OpenMP 系列环境变量（仅作为
        型式一致性保障；线程钉住的“权威途径”是 torch API）。子进程
        sweep 会增量处理 env，父进程中为 True 用于保证后续被导入的
        任何 BLAS 后端在初始化时看到同一个限制。
    :param verbose: 是否打印钉住后的实际生效状态。
    :return: 实际使用的 ``torch.get_num_threads()`` 返回值。
    """
    if n_threads is None:
        return torch.get_num_threads()
    if n_threads < 1:
        raise ValueError(
            f"_apply_thread_pin: n_threads must be >= 1, got {n_threads}"
        )

    import os as _os

    if set_env:
        # 后续才 import 的 BLAS 后端会读这些变量；PyTorch 本身在调
        # set_num_threads 后也会同步传递给 MKL ／OpenMP 运行时。
        for k in _BLAS_THREAD_ENV_KEYS:
            _os.environ[k] = str(n_threads)
        # 默认禁用 OMP 动态调整、开启 close binding（如用户未显式设置），
        # 使高并发 sweep 结果在跨 N 重复运行时更可重现。
        _os.environ.setdefault("OMP_DYNAMIC", "FALSE")
        _os.environ.setdefault("OMP_PROC_BIND", "close")

    # 1) intra-op：**权威途径**，对 pytorch_sdpa 等原生路径生效。
    torch.set_num_threads(int(n_threads))
    # 2) inter-op：PyTorch 要求在任何 op 运行之前设置；已运行过会报
    # RuntimeError，这里静默吞下。
    try:
        torch.set_num_interop_threads(int(n_threads))
    except RuntimeError:
        pass

    if verbose:
        try:
            from fused_cpp import _C  # type: ignore
            omp_max = _C.get_omp_runtime_info().get("max_threads", "?")
        except Exception:
            omp_max = "?"
        print(
            f"[sdpa] thread pin -> torch={torch.get_num_threads()}, "
            f"interop={torch.get_num_interop_threads()}, "
            f"omp_max={omp_max}, OMP_NUM_THREADS="
            f"{_os.environ.get('OMP_NUM_THREADS', 'null')}"
        )
    return torch.get_num_threads()


def _resolve_pinned_thread_count(config) -> Optional[int]:
    """从 CLI / 环境变量推断需要钉住的线程数。

    优先级：``--sdpa-num-threads`` > ``OMP_NUM_THREADS`` env > None。
    返回 None 表示不钉住（保持运行时默认）。
    """
    import os as _os

    raw = config.getoption("--sdpa-num-threads", default=None)
    if raw is not None:
        return int(raw)
    env = _os.environ.get("OMP_NUM_THREADS", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            return None
    return None


def _parse_csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [s.strip() for s in value.split(",") if s.strip()]
    return items or None


def _filter_sdpa_versions(
    versions: Sequence[VersionInfo],
    name_whitelist: Optional[List[str]],
    tag_whitelist: Optional[List[str]],
) -> List[VersionInfo]:
    out: List[VersionInfo] = []
    name_set = set(name_whitelist) if name_whitelist else None
    tag_set = set(tag_whitelist) if tag_whitelist else None
    for vi in versions:
        if name_set is not None and vi.name not in name_set:
            continue
        if tag_set is not None and vi.tags.isdisjoint(tag_set):
            continue
        out.append(vi)
    return out


def pytest_generate_tests(metafunc):  # noqa: D401
    """自动展开 ``sdpa_version`` 维度。

    只对参数名为 ``sdpa_version`` 的测试函数生效；其他测试不受影响。
    将从 :func:`available_sdpa_versions` 读取所有已注册版本，按 CLI 选项过滤
    后作为 :class:`VersionInfo` 参数化。id 为版本名以便 -k 过滤。
    """
    if "sdpa_version" not in metafunc.fixturenames:
        return
    config = metafunc.config
    name_filter = _parse_csv(config.getoption("--sdpa-versions", default=None))
    tag_filter = _parse_csv(config.getoption("--sdpa-tags", default=None))
    versions = _filter_sdpa_versions(
        available_sdpa_versions(), name_filter, tag_filter
    )
    if not versions:
        # 过滤后为空：生成一条 skip 用例，避免报错 "empty parametrize"。
        metafunc.parametrize(
            "sdpa_version",
            [pytest.param(
                None,
                id="no-version-matches-filter",
                marks=pytest.mark.skip(reason="no SDPA version matches filter"),
            )],
        )
        return
    metafunc.parametrize(
        "sdpa_version", versions, ids=[vi.name for vi in versions]
    )


def pytest_configure(config):  # noqa: D401
    """登记 SDPA 多版本框架使用的自定义 marker。

    同时负责避免 ``--strict-markers`` 在 ``pyproject.toml`` 中未登记时报错。
    """
    config.addinivalue_line(
        "markers",
        "equiv: SDPA multi-version equivalence test (default-collected).",
    )
    config.addinivalue_line(
        "markers",
        "bench: SDPA multi-version benchmark; default-skipped, run via -m bench.",
    )
    config.addinivalue_line(
        "markers",
        "slow: large shape / long-running test; default-skipped.",
    )


def pytest_collection_modifyitems(config, items):  # noqa: D401
    """默认跳过 ``slow`` 与 ``bench`` 标记的用例。

    遵循 ``python_test.md`` 第 12.2 节：快/慢两层分明，默认 ``pytest`` 跑快路径。
    “快路径” 包含 ``equiv``；``bench`` / ``slow`` 仅在显式 ``-m bench`` /
    ``-m slow`` 时才会被选中。
    """
    selected_marks = config.getoption("-m", default="") or ""
    skip_bench = pytest.mark.skip(reason="benchmark; run with `pytest -m bench`")
    skip_slow = pytest.mark.skip(reason="slow test; run with `pytest -m slow`")
    for item in items:
        marks = {m.name for m in item.iter_markers()}
        if "bench" in marks and "bench" not in selected_marks:
            item.add_marker(skip_bench)
        if "slow" in marks and "slow" not in selected_marks:
            item.add_marker(skip_slow)


# ── 帮助函数：按 VersionInfo 调用 SDPA ────────────────────────────────


def call_sdpa_version(
    info: VersionInfo,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """包装 :class:`VersionInfo.callable` 的调用，统一 ``torch.no_grad`` 语义。

    供等价性测试与 benchmark 复用。
    """
    with torch.no_grad():
        return info.callable(
            query, key, value,
            attn_mask=attn_mask, is_causal=is_causal, scale=scale,
        )


# ── SDPA benchmark 结果聚合（仅在 bench 用例运行时填充）───────────────────
#
# `bench_sdpa_versions.py` 会在每个用例结束时调用 :func:`record_sdpa_bench`
# 将一条记录 append 到 :data:`_BENCH_RECORDS`；session 末 :func:`pytest_session
# finish` 负责落盘 + 打印汇总 + 可选 thread sweep。
# 这份状态放在 conftest 中是必要的：pytest 只会调用 conftest.py 中的
# pytest_sessionfinish 钩子（测试文件中的同名函数 不会 被调起）。

from typing import Any, Dict, List as _List  # noqa: E402  (在运行时为了准确依赖)

_BENCH_RECORDS: _List[Dict[str, Any]] = []


def record_sdpa_bench(record: Dict[str, Any]) -> None:
    """供 benchmark 测试调用：将一条结果 append 到 session 聚合表。"""
    _BENCH_RECORDS.append(record)


def get_sdpa_bench_records() -> _List[Dict[str, Any]]:
    """返回当前为止记录的 bench 结果列表（供上层工具读取）。"""
    return list(_BENCH_RECORDS)


def _resolve_bench_output_dir(config) -> "Path":
    from pathlib import Path
    raw = config.getoption("--sdpa-bench-output-dir", default=None)
    if raw:
        out = Path(raw).expanduser().resolve()
    else:
        # <repo>/bench/sdpa
        out = Path(__file__).resolve().parents[1] / "bench" / "sdpa"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _bench_write_outputs(out_dir, prefix, records):
    import csv as _csv
    import json as _json
    if not records:
        return None, None
    json_path = out_dir / f"{prefix}.json"
    csv_path  = out_dir / f"{prefix}.csv"
    json_path.write_text(_json.dumps(records, indent=2))
    fields = list(records[0].keys())
    with csv_path.open("w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    return json_path, csv_path


def _bench_effective_gflops(record):
    """Return the GFLOP/s metric that matches the row's causal mode."""
    if "gflops" in record:
        return float(record["gflops"])
    key = "gflops_causal" if record["is_causal"] else "gflops_total"
    return float(record[key])


def _bench_print_summary(records):
    if not records:
        return
    print()
    header = (
        f"{'version':<14} {'shape':<24} {'dtype':<5} {'causal':<6} "
        f"{'mean_ms':>10} {'P90_ms':>10} "
        f"{'gflops':>12} "
        f"{'speedup_pt':>11} {'torch_t':>7} {'omp_t':>7}"
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in records:
        speed = r["speedup_vs_pytorch"]
        speed_s = f"{speed:>10.3f}x" if speed == speed else "       n/a "
        print(
            f"{r['version']:<14} {r['shape']:<24} {r['dtype']:<5} "
            f"{str(r['is_causal']):<6} "
            f"{r['mean_ms']:>10.3f} {r['p90_ms']:>10.3f} "
            f"{_bench_effective_gflops(r):>12.2f} "
            f"{speed_s} {str(r['num_threads']):>7} "
            f"{str(r.get('omp_max_threads', '?')):>7}"
        )
    print("=" * len(header))


def _bench_run_thread_sweep(out_dir, threads, orig_args):
    """对每个 N 启动子进程 OMP_NUM_THREADS=N pytest <orig_args>。

    Linux PyTorch 在 ``import torch`` 时就把 intra-op 线程数固定为 ``num_cpus``，
    之后改 ``OMP_NUM_THREADS`` 已无法压低 PyTorch 路径（``pytorch_sdpa``）的
    并行度。为此本函数在子进程层面做两件事：

    1) 以**完整的 BLAS / OpenMP 系列环境变量**启动子进程，确保任何上游
       BLAS 后端（MKL / OpenBLAS / Accelerate / BLIS / NumExpr）以及 OMP
       运行时在加载时就看到统一的 ``N`` 限制；同时关闭 ``OMP_DYNAMIC``、
       并默认 ``OMP_PROC_BIND=close`` 让相同 N 的多次跑结果可重现。
    2) 把 ``--sdpa-num-threads=N`` **显式追加**到子进程 pytest 命令行；
       子进程 ``test_sdpa_bench`` 在每条用例运行前会调用
       :func:`_apply_thread_pin` 调 ``torch.set_num_threads(N)`` /
       ``set_num_interop_threads(N)``，对 PyTorch 路径生效。
    """
    import json as _json
    import os as _os
    import subprocess as _sp
    import sys as _sys

    sweep_records = []
    sub_dir = out_dir / "_sweep_subdir"
    sub_dir.mkdir(parents=True, exist_ok=True)

    cleaned = []
    skip_next = False
    for a in orig_args:
        if skip_next:
            skip_next = False
            continue
        if a == "--sdpa-thread-sweep":
            skip_next = True
            continue
        if a.startswith("--sdpa-thread-sweep="):
            continue
        if a == "--sdpa-bench-output-dir":
            skip_next = True
            continue
        if a.startswith("--sdpa-bench-output-dir="):
            continue
        # 子进程会自己注入 --sdpa-num-threads=N，去掉父进程残留的同名选项
        # 避免冲突。
        if a == "--sdpa-num-threads":
            skip_next = True
            continue
        if a.startswith("--sdpa-num-threads="):
            continue
        cleaned.append(a)
    cleaned.append(f"--sdpa-bench-output-dir={sub_dir}")

    for n in threads:
        env = _os.environ.copy()
        # 让所有“线程数相关”的环境变量在子进程导入 torch / BLAS 后端**之前**
        # 就拿到统一的 N，避免 PyTorch / MKL / OpenBLAS 各自按 #cpus 起池。
        for key in _BLAS_THREAD_ENV_KEYS:
            env[key] = str(n)
        env.setdefault("OMP_DYNAMIC", "FALSE")
        env.setdefault("OMP_PROC_BIND", "close")
        for f in sub_dir.glob("*.json"):
            f.unlink()
        # 显式把 --sdpa-num-threads=N 追加进子进程命令行，子进程 bench 会
        # 在每条用例前调 torch.set_num_threads(N) 二次保险。
        cmd = [
            _sys.executable, "-m", "pytest", *cleaned,
            f"--sdpa-num-threads={n}",
        ]
        print(f"\n[sweep] OMP_NUM_THREADS={n} → {' '.join(cmd)}")
        rc = _sp.run(cmd, env=env, check=False)
        for fp in sorted(sub_dir.glob("*.json")):
            try:
                data = _json.loads(fp.read_text())
            except Exception:
                continue
            for rec in data:
                rec = dict(rec)
                rec["sweep_threads"] = int(n)
                sweep_records.append(rec)
        if rc.returncode != 0:
            print(f"[sweep] subprocess returned {rc.returncode} (continuing)")
    return sweep_records


def _bench_scaling_efficiency(records):
    """对每个 (version, shape, dtype, is_causal) 计算 N=1 → N 扩展效率。"""
    bucket = {}
    for r in records:
        key = (r["version"], r["shape"], r["dtype"], r["is_causal"])
        bucket.setdefault(key, []).append(
            (int(r["sweep_threads"]), _bench_effective_gflops(r))
        )
    eff_rows = []
    for (ver, shp, dt, ic), pairs in bucket.items():
        pairs.sort()
        base = next((p for p in pairs if p[0] == 1), None)
        if base is None:
            continue
        base_g = base[1]
        for n, g in pairs:
            denom = n * base_g
            eff = (g / denom) if denom > 0 else 0.0
            eff_rows.append({
                "version":   ver,
                "shape":     shp,
                "dtype":     dt,
                "is_causal": ic,
                "threads":   n,
                "gflops":    g,
                "efficiency":   eff,
            })
    return eff_rows


def pytest_sessionfinish(session, exitstatus):  # noqa: D401
    """session 结束时落盘 SDPA bench 结果，并按需触发 thread sweep。

    仅在有 bench 记录时才输出任何东西；默认 ``pytest`` 跑等价性测试时本钩
    子不产生任何附加输出。
    """
    import datetime as _dt
    if not _BENCH_RECORDS:
        return
    config = session.config
    out_dir = _resolve_bench_output_dir(config)
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"sdpa_versions_{timestamp}"
    json_path, csv_path = _bench_write_outputs(out_dir, prefix, _BENCH_RECORDS)
    print()
    print(f"[bench] wrote {len(_BENCH_RECORDS)} records to:")
    print(f"        {json_path}")
    print(f"        {csv_path}")
    _bench_print_summary(_BENCH_RECORDS)

    sweep_raw = config.getoption("--sdpa-thread-sweep", default=None)
    threads_csv = _parse_csv(sweep_raw)
    if not threads_csv:
        return

    try:
        from fused_cpp import _C  # type: ignore
        has_omp = bool(_C.has_openmp())
    except Exception:
        has_omp = False
    if not has_omp:
        print(
            "\n[sweep] skipped: C++ extension was built without OpenMP. "
            "Re-build with libomp to enable thread sweep."
        )
        return

    try:
        threads = [int(s) for s in threads_csv]
    except ValueError:
        print(f"\n[sweep] invalid thread spec: {threads_csv}")
        return

    orig_args = list(config.invocation_params.args)
    sweep_records = _bench_run_thread_sweep(out_dir, threads, orig_args)
    sw_prefix = f"sdpa_versions_{timestamp}_sweep"
    sw_json, sw_csv = _bench_write_outputs(out_dir, sw_prefix, sweep_records)
    print(f"\n[sweep] wrote {len(sweep_records)} sweep records to:")
    print(f"        {sw_json}")
    print(f"        {sw_csv}")
    eff_rows = _bench_scaling_efficiency(sweep_records)
    if eff_rows:
        eff_prefix = f"sdpa_versions_{timestamp}_scaling"
        ej, ec = _bench_write_outputs(out_dir, eff_prefix, eff_rows)
        print(f"        {ej}")
        print(f"        {ec}")
        print()
        print("=== Scaling efficiency (relative to N=1) ===")
        for row in eff_rows:
            print(
                f"  {row['version']:<14} {row['shape']:<24} "
                f"{row['dtype']:<5} causal={row['is_causal']!s:<5} "
                f"N={row['threads']:>3}  gflops={row['gflops']:>9.2f}  "
                f"eff={row['efficiency']:.3f}"
            )
