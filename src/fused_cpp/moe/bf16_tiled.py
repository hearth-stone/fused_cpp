# -*- coding: utf-8 -*-
"""BF16 tiled fused MoE wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any, Tuple

import torch

PreparedWeight = Tuple[torch.Tensor, int, int]


@dataclass(frozen=True)
class PreparedBF16TiledFusedMoEWeights:
    """Packed BF16 weights for the tiled fused MoE path."""

    w13: PreparedWeight
    w2: PreparedWeight
    fused_silu: bool = False
    # Stable native backend id: 0=NEON, 1=SVE, 101=AVX-512 BF16.
    gemm_backend: int = 0
    backend_n_tile: int = 8
    backend_name: str = "arm_neon_bf16"


try:
    from fused_cpp import _moe_C as _moe_native  # type: ignore[attr-defined, import-untyped]

    _fused_moe_bf16_tiled_impl = _moe_native.fused_moe_bf16_tiled
    _fused_moe_bf16_tiled_scheduled_impl = _moe_native.fused_moe_bf16_tiled_scheduled
    _fused_moe_bf16_tiled_async_impl = _moe_native.fused_moe_bf16_tiled_async
    _fused_moe_bf16_tiled_vllm_staged_impl = _moe_native.fused_moe_bf16_tiled_vllm_staged
    _prepare_bf16_tiled_impl = _moe_native.fused_moe_bf16_tiled_prepare_weights
    _available_backends_impl = _moe_native.fused_moe_bf16_tiled_available_backends
    _HAS_BF16_TILED_FUSED_MOE = bool(_available_backends_impl())

    # Keep direct fused_cpp._C.fused_moe_* users working while the native MoE
    # code lives in its own fat-binary extension.
    try:
        from fused_cpp import _C as _legacy_native  # type: ignore[attr-defined, import-untyped]
    except ImportError:
        _legacy_native = None

    for _name in (
        "fused_moe_bf16_tiled_available_backends",
        "fused_moe_bf16_tiled_prepare_weights",
        "fused_moe_bf16_tiled",
        "fused_moe_bf16_tiled_scheduled",
        "fused_moe_bf16_tiled_async",
        "fused_moe_bf16_tiled_vllm_staged",
        "fused_moe_test_split_plan",
        "fused_moe_test_single_thread_gemm",
        "fused_moe_test_pack_interleaved_gemm",
        "fused_moe_test_fused_w13_linear",
        "fused_moe_test_fused_w13_silu",
        "fused_moe_test_team_fused_w13_silu",
        "fused_moe_test_pack_a_reorder_m8",
        "fused_moe_test_gather_pack_a_reorder_m8",
        "fused_moe_test_fused_w13_silu_packc",
        "fused_moe_test_fused_w13_silu_packc_tail",
        "fused_moe_test_team_gemm",
        "fused_moe_bench_team_gemm",
        "fused_moe_bench_fused_w13_silu_packc_tail",
    ):
        if _legacy_native is not None and hasattr(_moe_native, _name) and not hasattr(_legacy_native, _name):
            setattr(_legacy_native, _name, getattr(_moe_native, _name))
except ImportError as error:
    if "SVE vector length mismatch:" in str(error):
        raise
    _moe_native = None
    _fused_moe_bf16_tiled_impl = None
    _fused_moe_bf16_tiled_scheduled_impl = None
    _fused_moe_bf16_tiled_async_impl = None
    _fused_moe_bf16_tiled_vllm_staged_impl = None
    _prepare_bf16_tiled_impl = None
    _available_backends_impl = None
    _HAS_BF16_TILED_FUSED_MOE = False
except AttributeError:
    _moe_native = None
    _fused_moe_bf16_tiled_impl = None
    _fused_moe_bf16_tiled_scheduled_impl = None
    _fused_moe_bf16_tiled_async_impl = None
    _fused_moe_bf16_tiled_vllm_staged_impl = None
    _prepare_bf16_tiled_impl = None
    _available_backends_impl = None
    _HAS_BF16_TILED_FUSED_MOE = False


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _require_backend() -> None:
    if not _HAS_BF16_TILED_FUSED_MOE:
        raise RuntimeError("No BF16 tiled fused MoE backend is supported by this build and CPU")


def available_fused_moe_bf16_tiled_backends() -> tuple[str, ...]:
    """Return native BF16 MoE backends usable by the current process."""
    if _available_backends_impl is None:
        return ()
    return tuple(str(name) for name in _available_backends_impl())


def _activation_name(activation: Any) -> str:
    value = getattr(activation, "value", activation)
    if not isinstance(value, str):
        value = str(value)
    value = {"gelu_pytorch_tanh": "gelu_tanh"}.get(value, value)
    if value not in {"silu", "gelu", "swigluoai"}:
        raise ValueError(f"Unsupported MoE activation {value!r}; supported activations: gelu, silu, swigluoai")
    return value


def _check_integer_schedule_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype, got {tensor.dtype}")
    if tensor.dim() != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(tensor.shape)}")


def _validate_output_buffer(input: torch.Tensor, out: torch.Tensor | None) -> None:
    if out is None:
        return
    if tuple(out.shape) != tuple(input.shape):
        raise ValueError(f"out must have shape {tuple(input.shape)}")
    if out.dtype != input.dtype or out.device != input.device:
        raise ValueError("out must match input dtype and device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    if out.requires_grad:
        raise ValueError("out with requires_grad=True is not supported")


def _weight_window_argument(weight_window_bytes: int | None) -> int:
    if weight_window_bytes is None:
        return -1
    if isinstance(weight_window_bytes, bool):
        raise TypeError("weight_window_bytes must be an integer byte count, not bool")
    try:
        value = index(weight_window_bytes)
    except TypeError as error:
        raise TypeError("weight_window_bytes must be an integer byte count or None") from error
    if value < 0:
        raise ValueError(f"weight_window_bytes must be non-negative, got {value}")
    if value > (1 << 63) - 1:
        raise OverflowError(f"weight_window_bytes exceeds int64: {value}")
    return value


def prepare_fused_moe_bf16_tiled_weights(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    fuse_silu: bool = False,
    backend: str = "auto",
) -> PreparedBF16TiledFusedMoEWeights:
    """Pack dense bf16 expert weights for :func:`fused_moe_bf16_tiled`.

    ``w13_weight`` follows the vLLM layout ``[E, 2 * F, H]`` and ``w2_weight``
    follows ``[E, H, F]``. The returned object is reusable across decode steps.
    Set ``FUSED_CPP_MOE_PREPACK_THREADS`` to parallelize packing by expert.

    ``fuse_silu=True`` packs w13 in the interleaved gate/up layout required by
    the fused SiLU-and-mul GEMM epilogue. ARM currently requires ``F % 8 == 0``;
    AVX-512 and AMX pad arbitrary positive ``F``. All fused backends require
    ``activation='silu'`` at call time. The returned object carries a ``fused_silu`` flag that
    :func:`fused_moe_bf16_tiled` honours automatically. On supported x86 Linux
    systems, ``backend='auto'`` prefers AMX BF16 and falls back to AVX-512 BF16;
    AMX pattern and cache-window selection are automatic per routed expert.
    """
    _require_backend()
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise TypeError("BF16 tiled fused MoE weights must be torch.bfloat16")
    if not w13_weight.device.type == w2_weight.device.type == "cpu":
        raise ValueError("BF16 tiled fused MoE weights must be CPU tensors")
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")

    packed = _prepare_bf16_tiled_impl(
        w13_weight.contiguous(),
        w2_weight.contiguous(),
        bool(fuse_silu),
        backend,
    )
    backend_id = int(packed[6]) if len(packed) > 6 else 0
    backend_names = {
        0: "arm_neon_bf16",
        1: "arm_sve_bf16",
        101: "x86_avx512_bf16",
        102: "x86_amx_bf16",
    }
    return PreparedBF16TiledFusedMoEWeights(
        w13=(packed[0], int(packed[1]), int(packed[2])),
        w2=(packed[3], int(packed[4]), int(packed[5])),
        fused_silu=bool(fuse_silu),
        gemm_backend=backend_id,
        backend_n_tile=int(packed[7]) if len(packed) > 7 else 8,
        backend_name=str(packed[8]) if len(packed) > 8 else backend_names.get(backend_id, f"backend_{backend_id}"),
    )


def fused_moe_bf16_tiled(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    num_threads: int = 1,
    activation: Any = "silu",
    global_num_experts: int = -1,
    skip_weighted: bool = False,
    silu_poly_degree: int = 5,
    weight_window_bytes: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the C++ tiled fused MoE path using BF16 GEMMs.

    If ``weights`` were prepared with ``fuse_silu=True`` and ``activation`` is
    ``"silu"``, the w13 GEMM fuses SiLU-and-mul into its store epilogue
    (``silu_poly_degree`` selects the exp polynomial, 4/5/6).

    A positive ``weight_window_bytes`` serializes each SVE GEMM into
    tile-aligned packed-B windows no larger than that target, except when one
    hardware N tile itself is larger. ``None`` reads
    ``FUSED_CPP_MOE_WEIGHT_WINDOW_BYTES``; zero disables byte-based windows.
    A supplied ``out`` must be a contiguous CPU BF16 tensor matching ``input``;
    the native kernel writes it directly and returns it without an intermediate
    output allocation or copy. The AVX-512 BF16 backend currently supports the
    fused SiLU path without expert bias using one or two worker threads.
    """
    _require_backend()
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {input.dtype}")
    if input.device.type != "cpu":
        raise ValueError("input must be a CPU tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(f"topk_weights must use a floating dtype, got {topk_weights.dtype}")
    if int(num_threads) <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    _validate_output_buffer(input, out)

    def _contiguous_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        if bias.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("MoE bias must be torch.float32 or torch.bfloat16")
        if bias.device.type != "cpu":
            raise ValueError("MoE bias must be a CPU tensor")
        return bias.contiguous()

    result = _fused_moe_bf16_tiled_impl(
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        _contiguous_bias(w13_bias),
        _contiguous_bias(w2_bias),
        int(num_threads),
        _activation_name(activation),
        int(global_num_experts),
        bool(skip_weighted),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
        _weight_window_argument(weight_window_bytes),
        out,
    )
    return out if out is not None else result


def fused_moe_bf16_tiled_scheduled(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    wave_offsets: torch.Tensor,
    team_expert_ids: torch.Tensor,
    team_threads: torch.Tensor,
    *,
    thread_cpu_ids: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    num_threads: int = 1,
    activation: Any = "silu",
    global_num_experts: int = -1,
    skip_weighted: bool = False,
    silu_poly_degree: int = 5,
    weight_window_bytes: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the BF16 tiled MoE path using an externally supplied schedule.

    ``wave_offsets`` has shape ``[num_waves + 1]`` and indexes into
    ``team_expert_ids`` / ``team_threads``. Each team computes one active
    expert using ``team_threads[i]`` cooperative threads. A supplied ``out`` is
    written directly by the native kernel and must be contiguous.
    """
    _require_backend()
    if _fused_moe_bf16_tiled_scheduled_impl is None:
        raise RuntimeError(
            "BF16 tiled scheduled MoE backend is unavailable; rebuild the C++ "
            "extension with fused_moe_bf16_tiled_scheduled support."
        )
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {input.dtype}")
    if input.device.type != "cpu":
        raise ValueError("input must be a CPU tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(f"topk_weights must use a floating dtype, got {topk_weights.dtype}")
    if int(num_threads) <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    _check_integer_schedule_tensor(wave_offsets, "wave_offsets")
    _check_integer_schedule_tensor(team_expert_ids, "team_expert_ids")
    _check_integer_schedule_tensor(team_threads, "team_threads")
    if thread_cpu_ids is not None:
        _check_integer_schedule_tensor(thread_cpu_ids, "thread_cpu_ids")
        if int(thread_cpu_ids.numel()) != int(num_threads):
            raise ValueError(
                "thread_cpu_ids must have exactly num_threads entries: "
                f"got {int(thread_cpu_ids.numel())} vs {int(num_threads)}"
            )
    _validate_output_buffer(input, out)

    def _contiguous_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        if bias.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("MoE bias must be torch.float32 or torch.bfloat16")
        if bias.device.type != "cpu":
            raise ValueError("MoE bias must be a CPU tensor")
        return bias.contiguous()

    result = _fused_moe_bf16_tiled_scheduled_impl(
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        wave_offsets.contiguous(),
        team_expert_ids.contiguous(),
        team_threads.contiguous(),
        None if thread_cpu_ids is None else thread_cpu_ids.contiguous(),
        _contiguous_bias(w13_bias),
        _contiguous_bias(w2_bias),
        int(num_threads),
        _activation_name(activation),
        int(global_num_experts),
        bool(skip_weighted),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
        _weight_window_argument(weight_window_bytes),
        out,
    )
    return out if out is not None else result


def fused_moe_bf16_tiled_async(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    task_expert_ids: torch.Tensor,
    task_core_begins: torch.Tensor,
    task_threads: torch.Tensor,
    task_dep_offsets: torch.Tensor,
    task_deps: torch.Tensor,
    *,
    thread_cpu_ids: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    num_threads: int = 1,
    activation: Any = "silu",
    global_num_experts: int = -1,
    skip_weighted: bool = False,
    silu_poly_degree: int = 5,
    w13_split: bool | None = None,
    weight_window_bytes: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run BF16 tiled MoE with an async task-DAG schedule.

    Each task computes one active expert on a contiguous logical-thread
    interval. ``task_dep_offsets`` / ``task_deps`` encode a CSR dependency
    list, allowing later tasks to start as soon as their own interval is free
    instead of waiting for a whole wave barrier. ``w13_split`` explicitly
    selects the two-panel SVE W13 policy; ``None`` preserves the legacy
    ``FUSED_CPP_MOE_W13_SPLIT_N`` environment fallback. A positive
    ``weight_window_bytes`` supersedes that two-panel granularity and applies
    the same packed-B byte limit to both W13 and W2. A supplied ``out`` is
    written directly by the native kernel and must be contiguous.
    """
    _require_backend()
    if _fused_moe_bf16_tiled_async_impl is None:
        raise RuntimeError(
            "BF16 tiled async MoE backend is unavailable; rebuild the C++ "
            "extension with fused_moe_bf16_tiled_async support."
        )
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {input.dtype}")
    if input.device.type != "cpu":
        raise ValueError("input must be a CPU tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(f"topk_weights must use a floating dtype, got {topk_weights.dtype}")
    if int(num_threads) <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    _check_integer_schedule_tensor(task_expert_ids, "task_expert_ids")
    _check_integer_schedule_tensor(task_core_begins, "task_core_begins")
    _check_integer_schedule_tensor(task_threads, "task_threads")
    _check_integer_schedule_tensor(task_dep_offsets, "task_dep_offsets")
    _check_integer_schedule_tensor(task_deps, "task_deps")
    if int(task_core_begins.numel()) != int(task_expert_ids.numel()):
        raise ValueError("task_core_begins must match task_expert_ids length")
    if int(task_threads.numel()) != int(task_expert_ids.numel()):
        raise ValueError("task_threads must match task_expert_ids length")
    if int(task_dep_offsets.numel()) != int(task_expert_ids.numel()) + 1:
        raise ValueError("task_dep_offsets must have num_tasks + 1 entries")
    if thread_cpu_ids is not None:
        _check_integer_schedule_tensor(thread_cpu_ids, "thread_cpu_ids")
        if int(thread_cpu_ids.numel()) != int(num_threads):
            raise ValueError(
                "thread_cpu_ids must have exactly num_threads entries: "
                f"got {int(thread_cpu_ids.numel())} vs {int(num_threads)}"
            )
    _validate_output_buffer(input, out)

    def _contiguous_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
        if bias is None:
            return None
        if bias.dtype not in (torch.float32, torch.bfloat16):
            raise TypeError("MoE bias must be torch.float32 or torch.bfloat16")
        if bias.device.type != "cpu":
            raise ValueError("MoE bias must be a CPU tensor")
        return bias.contiguous()

    result = _fused_moe_bf16_tiled_async_impl(
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        task_expert_ids.contiguous(),
        task_core_begins.contiguous(),
        task_threads.contiguous(),
        task_dep_offsets.contiguous(),
        task_deps.contiguous(),
        None if thread_cpu_ids is None else thread_cpu_ids.contiguous(),
        _contiguous_bias(w13_bias),
        _contiguous_bias(w2_bias),
        int(num_threads),
        _activation_name(activation),
        int(global_num_experts),
        bool(skip_weighted),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
        -1 if w13_split is None else int(bool(w13_split)),
        _weight_window_argument(weight_window_bytes),
        out,
    )
    return out if out is not None else result


def fused_moe_bf16_tiled_vllm_staged(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    thread_cpu_ids: torch.Tensor | None = None,
    num_threads: int = 1,
    global_num_experts: int = -1,
    silu_poly_degree: int = 5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the experimental vLLM-style staged SVE scheduling baseline.

    The compute kernels, packed weights, direct route store, and weighted
    merge are the same as the production fused SVE path. Scheduling follows
    the vLLM CPU implementation instead: all ``(expert, W13 N-range)`` tasks
    share one dynamic queue, a global stage barrier separates W13 from W2,
    and all ``(expert, W2 N-range)`` tasks then share a second queue. Each W13
    range rescans and packs its expert input in M12 panels, matching vLLM's
    per-N-task A scan. This entrypoint is experimental and does not change the
    default fused MoE dispatch.
    """
    _require_backend()
    if _fused_moe_bf16_tiled_vllm_staged_impl is None:
        raise RuntimeError(
            "BF16 tiled vLLM-staged MoE backend is unavailable; rebuild the "
            "C++ extension with fused_moe_bf16_tiled_vllm_staged support."
        )
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {input.dtype}")
    if input.device.type != "cpu":
        raise ValueError("input must be a CPU tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(f"topk_weights must use a floating dtype, got {topk_weights.dtype}")
    if int(num_threads) <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    if not weights.fused_silu:
        raise ValueError("vLLM-staged baseline requires weights prepared with fuse_silu=True")
    if weights.gemm_backend != 1:
        raise ValueError(
            "vLLM-staged baseline requires the SVE BF16 backend; "
            f"weights use {weights.backend_name}"
        )
    if thread_cpu_ids is not None:
        _check_integer_schedule_tensor(thread_cpu_ids, "thread_cpu_ids")
        if int(thread_cpu_ids.numel()) != int(num_threads):
            raise ValueError(
                "thread_cpu_ids must have exactly num_threads entries: "
                f"got {int(thread_cpu_ids.numel())} vs {int(num_threads)}"
            )
    _validate_output_buffer(input, out)

    result = _fused_moe_bf16_tiled_vllm_staged_impl(
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        None if thread_cpu_ids is None else thread_cpu_ids.contiguous(),
        int(num_threads),
        int(global_num_experts),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
        out,
    )
    return out if out is not None else result


bf16_tiled_fused_moe = fused_moe_bf16_tiled
bf16_tiled_fused_moe_scheduled = fused_moe_bf16_tiled_scheduled
bf16_tiled_fused_moe_async = fused_moe_bf16_tiled_async
bf16_tiled_fused_moe_vllm_staged = fused_moe_bf16_tiled_vllm_staged
prepare_bf16_tiled_fused_moe_weights = prepare_fused_moe_bf16_tiled_weights


__all__ = [
    "PreparedBF16TiledFusedMoEWeights",
    "PreparedWeight",
    "_HAS_BF16_TILED_FUSED_MOE",
    "available_fused_moe_bf16_tiled_backends",
    "fused_moe_bf16_tiled",
    "fused_moe_bf16_tiled_scheduled",
    "fused_moe_bf16_tiled_async",
    "fused_moe_bf16_tiled_vllm_staged",
    "bf16_tiled_fused_moe",
    "bf16_tiled_fused_moe_scheduled",
    "bf16_tiled_fused_moe_async",
    "bf16_tiled_fused_moe_vllm_staged",
    "prepare_fused_moe_bf16_tiled_weights",
    "prepare_bf16_tiled_fused_moe_weights",
]
