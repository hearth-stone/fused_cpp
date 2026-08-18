# -*- coding: utf-8 -*-
"""BF16 tiled fused MoE wrapper."""

from __future__ import annotations

import os
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import torch

from fused_cpp.moe.plan import AsyncMoEPlanV2

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


@dataclass(frozen=True)
class PreparedW8A16TiledFusedMoEWeights:
    """Per-output-channel INT8 weights for the experimental SVE Plan V2 path."""

    w13: tuple[torch.Tensor, int, int, torch.Tensor]
    w2: tuple[torch.Tensor, int, int, torch.Tensor]
    fused_silu: bool = True
    gemm_backend: int = 1
    backend_n_tile: int = 8
    backend_name: str = "arm_sve_w8a16"


@dataclass(frozen=True)
class PreparedBF16TiledRoutedSharedMoEWeights:
    """One routed expert array followed by one synthetic shared expert."""

    packed: PreparedBF16TiledFusedMoEWeights
    routed_experts: int
    shared_expert_id: int


try:
    from fused_cpp import _moe_C as _moe_native  # type: ignore[attr-defined, import-untyped]

    _fused_moe_bf16_tiled_impl = _moe_native.fused_moe_bf16_tiled
    _fused_moe_bf16_tiled_scheduled_impl = _moe_native.fused_moe_bf16_tiled_scheduled
    _fused_moe_bf16_tiled_async_impl = _moe_native.fused_moe_bf16_tiled_async
    _fused_moe_bf16_tiled_async_plan_v2_impl = getattr(_moe_native, "fused_moe_bf16_tiled_async_plan_v2", None)
    _fused_moe_w8a16_async_plan_v2_impl = getattr(_moe_native, "fused_moe_w8a16_tiled_async_plan_v2", None)
    _fused_moe_bf16_tiled_planned_staged_impl = getattr(_moe_native, "fused_moe_bf16_tiled_planned_staged", None)
    _fused_moe_bf16_tiled_vllm_staged_impl = _moe_native.fused_moe_bf16_tiled_vllm_staged
    _shared_mlp_bf16_tiled_impl = getattr(_moe_native, "shared_mlp_bf16_tiled", None)
    _prepare_bf16_tiled_impl = _moe_native.fused_moe_bf16_tiled_prepare_weights
    _prepare_w8a16_tiled_impl = getattr(_moe_native, "fused_moe_w8a16_tiled_prepare_weights", None)
    _prepare_routed_shared_impl = getattr(
        _moe_native,
        "fused_moe_bf16_tiled_prepare_routed_shared_weights",
        None,
    )
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
        "fused_moe_bf16_tiled_async_plan_v2",
        "fused_moe_bf16_tiled_planned_staged",
        "fused_moe_bf16_tiled_vllm_staged",
        "fused_moe_test_split_plan",
        "fused_moe_test_stage_window_plan",
        "fused_moe_test_single_thread_gemm",
        "fused_moe_test_pack_interleaved_gemm",
        "fused_moe_test_fused_w13_linear",
        "fused_moe_test_fused_w13_silu",
        "fused_moe_test_team_fused_w13_silu",
        "fused_moe_test_pack_a_reorder_m8",
        "fused_moe_test_gather_pack_a_reorder_m8",
        "fused_moe_test_fused_w13_silu_packc",
        "fused_moe_test_fused_w13_silu_packc_tail",
        "fused_moe_test_team_w13_silu_packc_window",
        "fused_moe_test_team_w2_window",
        "fused_moe_test_w2_scatter_ranges",
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
    _fused_moe_bf16_tiled_async_plan_v2_impl = None
    _fused_moe_w8a16_async_plan_v2_impl = None
    _fused_moe_bf16_tiled_planned_staged_impl = None
    _fused_moe_bf16_tiled_vllm_staged_impl = None
    _shared_mlp_bf16_tiled_impl = None
    _prepare_bf16_tiled_impl = None
    _prepare_w8a16_tiled_impl = None
    _prepare_routed_shared_impl = None
    _available_backends_impl = None
    _HAS_BF16_TILED_FUSED_MOE = False
except AttributeError:
    _moe_native = None
    _fused_moe_bf16_tiled_impl = None
    _fused_moe_bf16_tiled_scheduled_impl = None
    _fused_moe_bf16_tiled_async_impl = None
    _fused_moe_bf16_tiled_async_plan_v2_impl = None
    _fused_moe_w8a16_async_plan_v2_impl = None
    _fused_moe_bf16_tiled_planned_staged_impl = None
    _fused_moe_bf16_tiled_vllm_staged_impl = None
    _shared_mlp_bf16_tiled_impl = None
    _prepare_bf16_tiled_impl = None
    _prepare_w8a16_tiled_impl = None
    _prepare_routed_shared_impl = None
    _available_backends_impl = None
    _HAS_BF16_TILED_FUSED_MOE = False


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

_HUGETLBFS_PATH_ENV = "FUSED_CPP_MOE_HUGETLBFS_PATH"
_HUGETLB_SIZE_SUFFIXES = {
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
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


def _native_activation_name(
    activation: Any,
    weights: PreparedBF16TiledFusedMoEWeights | PreparedW8A16TiledFusedMoEWeights,
    swiglu_limit: float | None,
) -> str:
    name = _activation_name(activation)
    if swiglu_limit is None or float(swiglu_limit) == 0.0:
        return name
    limit = float(swiglu_limit)
    if not math.isfinite(limit) or limit != 10.0:
        raise ValueError("SVE JIT clamped SwiGLU currently requires swiglu_limit=10.0")
    if name != "silu":
        raise ValueError("swiglu_limit is only valid with activation='silu'")
    if not weights.fused_silu or weights.gemm_backend != 1:
        raise ValueError("swiglu_limit requires fused-SiLU weights using the SVE BF16 backend")
    return "silu_clamp10"


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


def _contiguous_moe_bias(bias: torch.Tensor | None) -> torch.Tensor | None:
    if bias is None:
        return None
    if bias.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("MoE bias must be torch.float32 or torch.bfloat16")
    if bias.device.type != "cpu":
        raise ValueError("MoE bias must be a CPU tensor")
    return bias.contiguous()


def _parse_hugetlb_size(value: str) -> int:
    normalized = value.strip().upper()
    if not normalized:
        raise ValueError("empty HugeTLB page size")
    suffix = normalized[-1]
    if suffix in _HUGETLB_SIZE_SUFFIXES:
        return int(normalized[:-1]) * _HUGETLB_SIZE_SUFFIXES[suffix]
    return int(normalized)


def _hugetlbfs_page_size(path: Path) -> int:
    if os.name != "posix" or not Path("/proc/mounts").is_file():
        raise RuntimeError(f"{_HUGETLBFS_PATH_ENV} requires Linux hugetlbfs")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"configured hugetlbfs path does not exist: {path}") from error
    for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[2] != "hugetlbfs":
            continue
        if Path(fields[1]).resolve() != resolved:
            continue
        for option in fields[3].split(","):
            if option.startswith("pagesize="):
                page_size = _parse_hugetlb_size(option.partition("=")[2])
                if page_size <= 0:
                    break
                return page_size
        raise RuntimeError(f"hugetlbfs mount {resolved} does not report a positive pagesize option")
    raise RuntimeError(f"{resolved} is not a hugetlbfs mount")


def _copy_packed_tensor_to_hugetlbfs(
    source: torch.Tensor,
    *,
    mount_path: Path,
    page_size: int,
    label: str,
) -> torch.Tensor:
    if source.device.type != "cpu" or not source.is_contiguous():
        raise RuntimeError(f"packed {label} must be a contiguous CPU tensor before HugeTLB migration")
    logical_bytes = source.numel() * source.element_size()
    mapped_bytes = ((logical_bytes + page_size - 1) // page_size) * page_size
    descriptor = -1
    filename = ""
    try:
        descriptor, filename = tempfile.mkstemp(prefix=f"fused_cpp_moe_{label}_", dir=mount_path)
        os.ftruncate(descriptor, mapped_bytes)
        os.close(descriptor)
        descriptor = -1
        storage = torch.UntypedStorage.from_file(filename, shared=True, nbytes=mapped_bytes)
        target = torch.empty(0, dtype=source.dtype, device="cpu").set_(
            storage,
            0,
            tuple(source.shape),
            tuple(source.stride()),
        )
        target.copy_(source)
        return target
    except (OSError, RuntimeError) as error:
        raise RuntimeError(
            f"failed to allocate packed {label} ({mapped_bytes} bytes) from "
            f"{page_size}-byte hugetlbfs mount {mount_path}"
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if filename:
            try:
                os.unlink(filename)
            except FileNotFoundError:
                pass


def _maybe_move_packed_weights_to_hugetlbfs(
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility shim for the pre-page-policy relocation path.

    The native packing routine now allocates packed weights through the shared
    page policy (``FUSED_CPP_PAGES`` / ``FUSED_CPP_PAGE_SIZE_MB`` /
    ``FUSED_CPP_HUGETLBFS_PATH``), so they already sit on the requested pages and
    relocating them here would only add a second full-size copy of the largest
    buffer in the path. ``FUSED_CPP_MOE_FORCE_HUGETLBFS_COPY=1`` restores the old
    behaviour for comparison.
    """
    raw_path = os.environ.get(_HUGETLBFS_PATH_ENV, "").strip()
    if not raw_path:
        return w13, w2
    if os.environ.get("FUSED_CPP_MOE_FORCE_HUGETLBFS_COPY", "").strip() in ("", "0"):
        return w13, w2
    mount_path = Path(raw_path)
    page_size = _hugetlbfs_page_size(mount_path)
    return (
        _copy_packed_tensor_to_hugetlbfs(w13, mount_path=mount_path, page_size=page_size, label="w13"),
        _copy_packed_tensor_to_hugetlbfs(w2, mount_path=mount_path, page_size=page_size, label="w2"),
    )


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
    Set ``FUSED_CPP_MOE_HUGETLBFS_PATH`` to a mounted hugetlbfs directory to
    copy only the reusable packed W13/W2 tensors into explicit HugeTLB pages.
    The configured path is strict: allocation or mount errors do not silently
    fall back to ordinary pages.

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
    packed_w13, packed_w2 = _maybe_move_packed_weights_to_hugetlbfs(packed[0], packed[3])
    return PreparedBF16TiledFusedMoEWeights(
        w13=(packed_w13, int(packed[1]), int(packed[2])),
        w2=(packed_w2, int(packed[4]), int(packed[5])),
        fused_silu=bool(fuse_silu),
        gemm_backend=backend_id,
        backend_n_tile=int(packed[7]) if len(packed) > 7 else 8,
        backend_name=str(packed[8]) if len(packed) > 8 else backend_names.get(backend_id, f"backend_{backend_id}"),
    )


def prepare_fused_moe_w8a16_tiled_weights(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
) -> PreparedW8A16TiledFusedMoEWeights:
    """Quantize BF16 expert weights to symmetric per-output-channel INT8."""
    _require_backend()
    if _prepare_w8a16_tiled_impl is None:
        raise RuntimeError("W8A16 fused MoE packing is unavailable in this build")
    if w13_weight.dtype != torch.bfloat16 or w2_weight.dtype != torch.bfloat16:
        raise TypeError("W8A16 source weights must be torch.bfloat16")
    if w13_weight.device.type != "cpu" or w2_weight.device.type != "cpu":
        raise ValueError("W8A16 source weights must be CPU tensors")
    packed = _prepare_w8a16_tiled_impl(w13_weight.contiguous(), w2_weight.contiguous())
    return PreparedW8A16TiledFusedMoEWeights(
        w13=(packed[0], int(packed[1]), int(packed[2]), packed[3]),
        w2=(packed[4], int(packed[5]), int(packed[6]), packed[7]),
        gemm_backend=int(packed[8]),
        backend_n_tile=int(packed[9]),
    )


def prepare_shared_mlp_bf16_tiled_weights(
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    *,
    backend: str = "arm_sve_bf16",
) -> PreparedBF16TiledFusedMoEWeights:
    """Pack one dense BF16 gate/up/down MLP for :func:`shared_mlp_bf16_tiled`.

    ``w13_weight`` has shape ``[2 * F, H]`` and ``w2_weight`` has shape
    ``[H, F]``. The returned object uses the existing one-expert fused-MoE
    packed layout and can be reused across calls.
    """
    if w13_weight.dim() != 2:
        raise ValueError(f"w13_weight must be 2-D [2 * F, H], got shape {tuple(w13_weight.shape)}")
    if w2_weight.dim() != 2:
        raise ValueError(f"w2_weight must be 2-D [H, F], got shape {tuple(w2_weight.shape)}")
    return prepare_fused_moe_bf16_tiled_weights(
        w13_weight.unsqueeze(0),
        w2_weight.unsqueeze(0),
        fuse_silu=True,
        backend=backend,
    )


def prepare_routed_shared_moe_bf16_tiled_weights(
    routed_w13_weight: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    shared_w13_weight: torch.Tensor,
    shared_w2_weight: torch.Tensor,
    *,
    backend: str = "arm_sve_bf16",
) -> PreparedBF16TiledRoutedSharedMoEWeights:
    """Pack routed experts and one same-shape shared expert without a full copy."""
    _require_backend()
    if _prepare_routed_shared_impl is None:
        raise RuntimeError("This build does not provide routed+shared BF16 packing")
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a string, got {type(backend).__name__}")
    packed = _prepare_routed_shared_impl(
        routed_w13_weight,
        routed_w2_weight,
        shared_w13_weight,
        shared_w2_weight,
        backend,
    )
    backend_id = int(packed[6])
    backend_name = str(packed[8]) if len(packed) > 8 else "arm_sve_bf16"
    packed_w13, packed_w2 = _maybe_move_packed_weights_to_hugetlbfs(packed[0], packed[3])
    combined = PreparedBF16TiledFusedMoEWeights(
        w13=(packed_w13, int(packed[1]), int(packed[2])),
        w2=(packed_w2, int(packed[4]), int(packed[5])),
        fused_silu=True,
        gemm_backend=backend_id,
        backend_n_tile=int(packed[7]),
        backend_name=backend_name,
    )
    routed_experts = int(routed_w13_weight.shape[0])
    return PreparedBF16TiledRoutedSharedMoEWeights(
        packed=combined,
        routed_experts=routed_experts,
        shared_expert_id=routed_experts,
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
    swiglu_limit: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the C++ tiled fused MoE path using BF16 GEMMs.

    If ``weights`` were prepared with ``fuse_silu=True`` and ``activation`` is
    ``"silu"``, the w13 GEMM fuses SiLU-and-mul into its store epilogue
    (``silu_poly_degree`` selects the exp polynomial, 4/5/6).
    ``swiglu_limit=10.0`` additionally applies DeepSeek-V4 gate/up clamping and
    requires the SVE JIT backend. ``None`` and ``0.0`` retain standard SwiGLU.

    A supplied ``out`` must be a contiguous CPU BF16 tensor matching ``input``;
    the native kernel writes it directly and returns it without an intermediate
    output allocation or copy. The x86 BF16 backends support up to 256 requested
    workers: balanced active experts run independently when there are enough
    experts; underfilled or strongly skewed routes form per-expert teams/waves
    and partition W13/W2 over N blocks.
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

    # Calibration is always explicit. Once a caller installs a compatible
    # runtime, the normal entrypoint lowers through Plan V2; all other calls
    # retain the established native dispatcher.
    from fused_cpp.moe.planner_runtime import get_default_moe_planner_runtime

    planner_runtime = get_default_moe_planner_runtime()
    if planner_runtime is not None:
        plan = planner_runtime.plan_for_dispatch(
            weights,
            topk_ids,
            num_threads=int(num_threads),
            activation=activation,
            global_num_experts=int(global_num_experts),
        )
        if plan is not None:
            return fused_moe_bf16_tiled_async_plan(
                input,
                weights,
                topk_weights,
                topk_ids,
                plan,
                w13_bias=w13_bias,
                w2_bias=w2_bias,
                activation=activation,
                global_num_experts=global_num_experts,
                skip_weighted=skip_weighted,
                silu_poly_degree=silu_poly_degree,
                swiglu_limit=swiglu_limit,
                out=out,
            )

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
        _contiguous_moe_bias(w13_bias),
        _contiguous_moe_bias(w2_bias),
        int(num_threads),
        _native_activation_name(activation, weights, swiglu_limit),
        int(global_num_experts),
        bool(skip_weighted),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
        out,
    )
    return out if out is not None else result


def fused_moe_bf16_tiled_with_shared(
    input: torch.Tensor,
    weights: PreparedBF16TiledRoutedSharedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_threads: int = 1,
    routed_scaling_factor: float = 1.0,
    activation: Any = "silu",
    silu_poly_degree: int = 5,
    swiglu_limit: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Schedule one all-token shared expert with the routed expert array."""
    _require_backend()
    if not isinstance(weights, PreparedBF16TiledRoutedSharedMoEWeights):
        raise TypeError("weights must be PreparedBF16TiledRoutedSharedMoEWeights")
    if input.dtype != torch.bfloat16 or input.device.type != "cpu":
        raise TypeError("input must be a CPU torch.bfloat16 tensor")
    if topk_ids.device.type != "cpu" or topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError("topk_ids must be an integer CPU tensor")
    if topk_weights.device.type != "cpu" or not topk_weights.dtype.is_floating_point:
        raise TypeError("topk_weights must be a floating CPU tensor")
    if topk_ids.dim() != 2 or topk_weights.dim() != 2 or topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have matching [tokens, top_k] shapes")
    if int(topk_ids.shape[0]) != int(input.shape[0]) or int(topk_ids.shape[1]) <= 0:
        raise ValueError("topk tensors must match input tokens and have positive top_k")
    if int(num_threads) <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    scaling = float(routed_scaling_factor)
    if not math.isfinite(scaling):
        raise ValueError("routed_scaling_factor must be finite")
    native_activation = _native_activation_name(activation, weights.packed, swiglu_limit)
    if _activation_name(activation) != "silu":
        raise ValueError("routed+shared BF16 scheduling currently requires activation='silu'")
    _validate_output_buffer(input, out)
    if topk_ids.numel() > 0:
        minimum_id = int(topk_ids.min().item())
        maximum_id = int(topk_ids.max().item())
        if minimum_id < 0 or maximum_id >= weights.routed_experts:
            raise ValueError(
                f"routed topk_ids must be in [0, {weights.routed_experts}), got [{minimum_id}, {maximum_id}]"
            )

    tokens, routed_top_k = topk_ids.shape
    combined_ids = torch.empty((tokens, routed_top_k + 1), dtype=topk_ids.dtype)
    combined_weights = torch.empty((tokens, routed_top_k + 1), dtype=topk_weights.dtype)
    combined_ids[:, :routed_top_k].copy_(topk_ids)
    combined_ids[:, routed_top_k].fill_(weights.shared_expert_id)
    combined_weights[:, :routed_top_k].copy_(topk_weights * scaling)
    combined_weights[:, routed_top_k].fill_(1.0)

    from fused_cpp.moe.planner_runtime import get_default_moe_planner_runtime

    planner_runtime = get_default_moe_planner_runtime()
    plan = None
    if planner_runtime is not None:
        plan = planner_runtime.plan_for_shared_dispatch(
            weights,
            combined_ids,
            num_threads=int(num_threads),
            activation="silu",
        )
    if plan is not None:
        return fused_moe_bf16_tiled_async_plan(
            input,
            weights.packed,
            combined_weights,
            combined_ids,
            plan,
            activation="silu",
            global_num_experts=weights.routed_experts + 1,
            silu_poly_degree=silu_poly_degree,
            swiglu_limit=swiglu_limit,
            out=out,
        )

    packed = weights.packed
    result = _fused_moe_bf16_tiled_impl(
        input.contiguous(),
        packed.w13[0],
        packed.w13[1],
        packed.w13[2],
        packed.w2[0],
        packed.w2[1],
        packed.w2[2],
        combined_weights,
        combined_ids,
        None,
        None,
        int(num_threads),
        native_activation,
        weights.routed_experts + 1,
        False,
        True,
        int(silu_poly_degree),
        int(packed.gemm_backend),
        int(packed.backend_n_tile),
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
    swiglu_limit: float | None = None,
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
        _contiguous_moe_bias(w13_bias),
        _contiguous_moe_bias(w2_bias),
        int(num_threads),
        _native_activation_name(activation, weights, swiglu_limit),
        int(global_num_experts),
        bool(skip_weighted),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
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
    swiglu_limit: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run BF16 tiled MoE with an async task-DAG schedule.

    Each task computes one active expert on a contiguous logical-thread
    interval. ``task_dep_offsets`` / ``task_deps`` encode a CSR dependency
    list, allowing later tasks to start as soon as their own interval is free
    instead of waiting for a whole wave barrier. A supplied ``out`` is written
    directly by the native kernel and must be contiguous.
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
        _contiguous_moe_bias(w13_bias),
        _contiguous_moe_bias(w2_bias),
        int(num_threads),
        _native_activation_name(activation, weights, swiglu_limit),
        int(global_num_experts),
        bool(skip_weighted),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
        out,
    )
    return out if out is not None else result


def fused_moe_bf16_tiled_async_plan(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    plan: AsyncMoEPlanV2 | Mapping[str, object],
    *,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    activation: Any = "silu",
    global_num_experts: int = -1,
    skip_weighted: bool = False,
    silu_poly_degree: int = 5,
    swiglu_limit: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute a validated Plan V2 through the native async DAG.

    ``strict`` preserves every fixed logical-thread interval. ``tail_pool``
    releases aligned thread groups after their fixed experts finish and lets
    each group claim whole pooled experts. Task widths remain fixed after
    startup in both modes.
    """
    materialized = plan if isinstance(plan, AsyncMoEPlanV2) else AsyncMoEPlanV2.from_dict(plan)
    if _fused_moe_bf16_tiled_async_plan_v2_impl is None:
        raise RuntimeError("Plan V2 requires native fused_moe_bf16_tiled_async_plan_v2 support")
    _require_backend()
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {input.dtype}")
    if input.device.type != "cpu":
        raise ValueError("input must be a CPU tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(f"topk_weights must use a floating dtype, got {topk_weights.dtype}")
    _validate_output_buffer(input, out)
    common_args = (
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        materialized.task_expert_ids.contiguous(),
        materialized.task_core_begins.contiguous(),
        materialized.task_threads.contiguous(),
        materialized.task_dep_offsets.contiguous(),
        materialized.task_deps.contiguous(),
        materialized.plan_version,
        materialized.native_execution_mode,
        materialized.task_preferred_threads.contiguous(),
        materialized.task_min_threads.contiguous(),
        materialized.task_max_threads.contiguous(),
        materialized.task_allowed_thread_offsets.contiguous(),
        materialized.task_allowed_threads.contiguous(),
        materialized.task_placement_modes.contiguous(),
        materialized.task_numa_nodes.contiguous(),
        materialized.task_stage_ids.contiguous(),
        materialized.task_resize_points.contiguous(),
        materialized.task_range_granularities.contiguous(),
        materialized.task_w13_window_tiles.contiguous(),
        materialized.task_w2_window_tiles.contiguous(),
        materialized.thread_cpu_ids.contiguous(),
        _contiguous_moe_bias(w13_bias),
        _contiguous_moe_bias(w2_bias),
        materialized.num_threads,
        _native_activation_name(activation, weights, swiglu_limit),
        int(global_num_experts),
        bool(skip_weighted),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
        out,
    )
    result = _fused_moe_bf16_tiled_async_plan_v2_impl(
        *common_args,
        materialized.native_early_merge,
    )
    return out if out is not None else result


def fused_moe_w8a16_tiled_async_plan(
    input: torch.Tensor,
    weights: PreparedW8A16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    plan: AsyncMoEPlanV2 | Mapping[str, object],
    *,
    activation: Any = "silu",
    global_num_experts: int = -1,
    skip_weighted: bool = False,
    silu_poly_degree: int = 5,
    swiglu_limit: float | None = None,
    cache_dequant: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Execute per-channel W8A16 experts with an existing Plan V2."""
    if _fused_moe_w8a16_async_plan_v2_impl is None:
        raise RuntimeError("W8A16 Plan V2 execution is unavailable in this build")
    materialized = plan if isinstance(plan, AsyncMoEPlanV2) else AsyncMoEPlanV2.from_dict(plan)
    if input.dtype != torch.bfloat16 or input.device.type != "cpu":
        raise TypeError("W8A16 input must be a CPU torch.bfloat16 tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(f"topk_weights must use a floating dtype, got {topk_weights.dtype}")
    _validate_output_buffer(input, out)
    result = _fused_moe_w8a16_async_plan_v2_impl(
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w13[3],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        weights.w2[3],
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        materialized.task_expert_ids.contiguous(),
        materialized.task_core_begins.contiguous(),
        materialized.task_threads.contiguous(),
        materialized.task_dep_offsets.contiguous(),
        materialized.task_deps.contiguous(),
        materialized.plan_version,
        materialized.native_execution_mode,
        materialized.task_preferred_threads.contiguous(),
        materialized.task_min_threads.contiguous(),
        materialized.task_max_threads.contiguous(),
        materialized.task_allowed_thread_offsets.contiguous(),
        materialized.task_allowed_threads.contiguous(),
        materialized.task_placement_modes.contiguous(),
        materialized.task_numa_nodes.contiguous(),
        materialized.task_stage_ids.contiguous(),
        materialized.task_resize_points.contiguous(),
        materialized.task_range_granularities.contiguous(),
        materialized.task_w13_window_tiles.contiguous(),
        materialized.task_w2_window_tiles.contiguous(),
        materialized.thread_cpu_ids.contiguous(),
        materialized.num_threads,
        _native_activation_name(activation, weights, swiglu_limit),
        int(global_num_experts),
        bool(skip_weighted),
        int(silu_poly_degree),
        int(weights.backend_n_tile),
        out,
        materialized.native_early_merge,
        bool(cache_dequant),
    )
    return out if out is not None else result


def fused_moe_w8a16_tiled(
    input: torch.Tensor,
    weights: PreparedW8A16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_threads: int = 1,
    activation: Any = "silu",
    global_num_experts: int = -1,
    skip_weighted: bool = False,
    silu_poly_degree: int = 5,
    swiglu_limit: float | None = None,
    cache_dequant: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run W8A16 fused MoE through the installed Plan V2 planner runtime.

    Unlike the BF16 operator, W8A16 has no unplanned native dispatcher. The
    caller must install a compatible :class:`MoePlannerRuntime` before use.
    ``cache_dequant`` reuses each task's calibrated Plan V2 stage window; it
    does not alter planner selection or make full-stripe windows cache-sized.
    """
    _require_backend()
    if not isinstance(weights, PreparedW8A16TiledFusedMoEWeights):
        raise TypeError("weights must be PreparedW8A16TiledFusedMoEWeights")
    if input.dtype != torch.bfloat16 or input.device.type != "cpu":
        raise TypeError("W8A16 input must be a CPU torch.bfloat16 tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(f"topk_weights must use a floating dtype, got {topk_weights.dtype}")
    if int(num_threads) <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")

    from fused_cpp.moe.planner_runtime import get_default_moe_planner_runtime

    planner_runtime = get_default_moe_planner_runtime()
    if planner_runtime is None:
        raise RuntimeError(
            "W8A16 fused MoE requires an installed MoePlannerRuntime; "
            "call enable_moe_planner_quick() or set_default_moe_planner_runtime() first"
        )
    plan = planner_runtime.plan_for_dispatch(
        weights,
        topk_ids,
        num_threads=int(num_threads),
        activation=activation,
        global_num_experts=int(global_num_experts),
    )
    if plan is None:
        raise RuntimeError(
            "the installed MoePlannerRuntime is incompatible with this W8A16 shape, "
            "thread count, activation, or expert topology"
        )
    return fused_moe_w8a16_tiled_async_plan(
        input,
        weights,
        topk_weights,
        topk_ids,
        plan,
        activation=activation,
        global_num_experts=global_num_experts,
        skip_weighted=skip_weighted,
        silu_poly_degree=silu_poly_degree,
        swiglu_limit=swiglu_limit,
        cache_dequant=cache_dequant,
        out=out,
    )


def fused_moe_tiled(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights | PreparedW8A16TiledFusedMoEWeights,
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
    swiglu_limit: float | None = None,
    cache_dequant: bool = False,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the tiled fused MoE implementation selected by prepared weights.

    BF16 remains the established default when weights come from
    :func:`prepare_fused_moe_bf16_tiled_weights`. W8A16 is selected only by a
    :class:`PreparedW8A16TiledFusedMoEWeights` object and requires a compatible
    installed planner runtime.
    """
    common = {
        "num_threads": num_threads,
        "activation": activation,
        "global_num_experts": global_num_experts,
        "skip_weighted": skip_weighted,
        "silu_poly_degree": silu_poly_degree,
        "swiglu_limit": swiglu_limit,
        "out": out,
    }
    if isinstance(weights, PreparedBF16TiledFusedMoEWeights):
        if cache_dequant:
            raise ValueError("cache_dequant is supported only by W8A16 fused MoE weights")
        return fused_moe_bf16_tiled(
            input,
            weights,
            topk_weights,
            topk_ids,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            **common,
        )
    if isinstance(weights, PreparedW8A16TiledFusedMoEWeights):
        if w13_bias is not None or w2_bias is not None:
            raise ValueError("W8A16 fused MoE does not support w13_bias or w2_bias")
        return fused_moe_w8a16_tiled(
            input,
            weights,
            topk_weights,
            topk_ids,
            cache_dequant=cache_dequant,
            **common,
        )
    raise TypeError(
        "weights must be PreparedBF16TiledFusedMoEWeights or "
        "PreparedW8A16TiledFusedMoEWeights"
    )


def fused_moe_bf16_tiled_planned_staged(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13_plan: AsyncMoEPlanV2 | Mapping[str, object],
    w2_plan: AsyncMoEPlanV2 | Mapping[str, object],
    *,
    global_num_experts: int = -1,
    silu_poly_degree: int = 5,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run independent expert-level W13 and W2 plans with a global barrier.

    This is an experimental comparison path. Both inputs use the validated
    Plan V2 schema, but each plan controls only its named GEMM stage. W13
    gathers and packs each expert input once, writes one global packed-C
    intermediate, and completes globally before the independently planned W2
    stage starts. The default production dispatcher is unchanged.
    """
    _require_backend()
    if _fused_moe_bf16_tiled_planned_staged_impl is None:
        raise RuntimeError(
            "BF16 tiled planned-staged MoE backend is unavailable; rebuild the "
            "C++ extension with fused_moe_bf16_tiled_planned_staged support."
        )
    materialized_w13 = w13_plan if isinstance(w13_plan, AsyncMoEPlanV2) else AsyncMoEPlanV2.from_dict(w13_plan)
    materialized_w2 = w2_plan if isinstance(w2_plan, AsyncMoEPlanV2) else AsyncMoEPlanV2.from_dict(w2_plan)
    if materialized_w13.num_threads != materialized_w2.num_threads:
        raise ValueError(
            "W13 and W2 plans must use the same num_threads: "
            f"{materialized_w13.num_threads} vs {materialized_w2.num_threads}"
        )
    w13_cpu_ids = materialized_w13.thread_cpu_ids.to(dtype=torch.int64)
    w2_cpu_ids = materialized_w2.thread_cpu_ids.to(dtype=torch.int64)
    if not torch.equal(w13_cpu_ids, w2_cpu_ids):
        raise ValueError("W13 and W2 plans must use identical thread_cpu_ids")
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {input.dtype}")
    if input.device.type != "cpu":
        raise ValueError("input must be a CPU tensor")
    if topk_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"topk_ids must use an integer dtype, got {topk_ids.dtype}")
    if not topk_weights.dtype.is_floating_point:
        raise TypeError(f"topk_weights must use a floating dtype, got {topk_weights.dtype}")
    if not weights.fused_silu:
        raise ValueError("planned-staged MoE requires weights prepared with fuse_silu=True")
    if weights.gemm_backend != 1:
        raise ValueError(f"planned-staged MoE requires the SVE BF16 backend; weights use {weights.backend_name}")
    _validate_output_buffer(input, out)
    result = _fused_moe_bf16_tiled_planned_staged_impl(
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        topk_weights.contiguous(),
        topk_ids.contiguous(),
        materialized_w13.task_expert_ids.contiguous(),
        materialized_w13.task_core_begins.contiguous(),
        materialized_w13.task_threads.contiguous(),
        materialized_w13.task_dep_offsets.contiguous(),
        materialized_w13.task_deps.contiguous(),
        materialized_w13.native_execution_mode,
        materialized_w13.task_placement_modes.contiguous(),
        materialized_w2.task_expert_ids.contiguous(),
        materialized_w2.task_core_begins.contiguous(),
        materialized_w2.task_threads.contiguous(),
        materialized_w2.task_dep_offsets.contiguous(),
        materialized_w2.task_deps.contiguous(),
        materialized_w2.native_execution_mode,
        materialized_w2.task_placement_modes.contiguous(),
        materialized_w13.thread_cpu_ids.contiguous(),
        materialized_w13.num_threads,
        int(global_num_experts),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
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
        raise ValueError(f"vLLM-staged baseline requires the SVE BF16 backend; weights use {weights.backend_name}")
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


def shared_mlp_bf16_tiled(
    input: torch.Tensor,
    weights: PreparedBF16TiledFusedMoEWeights,
    *,
    thread_cpu_ids: torch.Tensor | None = None,
    num_threads: int = 1,
    silu_poly_degree: int = 5,
    swiglu_limit: float | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one packed expert as a standalone BF16 shared MLP on ARM SVE.

    Each W13/W2 stage first chooses a cache-sized N window using the complete
    input row count. Every N window is then split into M12-aligned row tasks
    until the dynamic task queue can occupy ``num_threads`` workers. W13 uses
    the existing fused SiLU-and-mul expert kernel and W2 writes BF16 output
    directly, without routing or weighted merge work.

    The operator is thread-compatible, not thread-safe: callers must serialize
    invocations because execution uses the process-wide resident MoE worker
    pool. Prepared weights are read-only and reusable across serialized calls.
    ``swiglu_limit=10.0`` selects the DeepSeek-V4 SVE JIT clamped epilogue.
    """
    _require_backend()
    if _shared_mlp_bf16_tiled_impl is None:
        raise RuntimeError(
            "BF16 tiled shared MLP backend is unavailable; rebuild the C++ "
            "extension with shared_mlp_bf16_tiled support."
        )
    if input.dtype != torch.bfloat16:
        raise TypeError(f"input must be torch.bfloat16, got {input.dtype}")
    if input.device.type != "cpu":
        raise ValueError("input must be a CPU tensor")
    if int(num_threads) <= 0:
        raise ValueError(f"num_threads must be positive, got {num_threads}")
    if not weights.fused_silu:
        raise ValueError("shared MLP requires weights prepared with fuse_silu=True")
    if weights.gemm_backend != 1:
        raise ValueError(f"shared MLP requires the SVE BF16 backend; weights use {weights.backend_name}")
    if int(weights.w13[0].size(0)) != 1 or int(weights.w2[0].size(0)) != 1:
        raise ValueError("shared MLP requires exactly one packed expert")
    if thread_cpu_ids is not None:
        _check_integer_schedule_tensor(thread_cpu_ids, "thread_cpu_ids")
        if int(thread_cpu_ids.numel()) != int(num_threads):
            raise ValueError(
                "thread_cpu_ids must have exactly num_threads entries: "
                f"got {int(thread_cpu_ids.numel())} vs {int(num_threads)}"
            )
    _validate_output_buffer(input, out)
    native_activation = _native_activation_name("silu", weights, swiglu_limit)

    result = _shared_mlp_bf16_tiled_impl(
        input.contiguous(),
        weights.w13[0],
        weights.w13[1],
        weights.w13[2],
        weights.w2[0],
        weights.w2[1],
        weights.w2[2],
        None if thread_cpu_ids is None else thread_cpu_ids.contiguous(),
        int(num_threads),
        bool(weights.fused_silu),
        int(silu_poly_degree),
        int(weights.gemm_backend),
        int(weights.backend_n_tile),
        out,
        native_activation == "silu_clamp10",
    )
    return out if out is not None else result


bf16_tiled_fused_moe = fused_moe_bf16_tiled
bf16_tiled_fused_moe_scheduled = fused_moe_bf16_tiled_scheduled
bf16_tiled_fused_moe_async = fused_moe_bf16_tiled_async
bf16_tiled_fused_moe_async_plan = fused_moe_bf16_tiled_async_plan
bf16_tiled_fused_moe_planned_staged = fused_moe_bf16_tiled_planned_staged
bf16_tiled_fused_moe_vllm_staged = fused_moe_bf16_tiled_vllm_staged
prepare_bf16_tiled_fused_moe_weights = prepare_fused_moe_bf16_tiled_weights


__all__ = [
    "PreparedBF16TiledFusedMoEWeights",
    "PreparedBF16TiledRoutedSharedMoEWeights",
    "PreparedW8A16TiledFusedMoEWeights",
    "PreparedWeight",
    "_HAS_BF16_TILED_FUSED_MOE",
    "available_fused_moe_bf16_tiled_backends",
    "fused_moe_bf16_tiled",
    "fused_moe_bf16_tiled_with_shared",
    "fused_moe_bf16_tiled_scheduled",
    "fused_moe_bf16_tiled_async",
    "fused_moe_bf16_tiled_async_plan",
    "fused_moe_bf16_tiled_planned_staged",
    "fused_moe_bf16_tiled_vllm_staged",
    "fused_moe_tiled",
    "fused_moe_w8a16_tiled",
    "fused_moe_w8a16_tiled_async_plan",
    "shared_mlp_bf16_tiled",
    "bf16_tiled_fused_moe",
    "bf16_tiled_fused_moe_scheduled",
    "bf16_tiled_fused_moe_async",
    "bf16_tiled_fused_moe_async_plan",
    "bf16_tiled_fused_moe_planned_staged",
    "bf16_tiled_fused_moe_vllm_staged",
    "prepare_fused_moe_bf16_tiled_weights",
    "prepare_fused_moe_w8a16_tiled_weights",
    "prepare_routed_shared_moe_bf16_tiled_weights",
    "prepare_shared_mlp_bf16_tiled_weights",
    "prepare_bf16_tiled_fused_moe_weights",
]
