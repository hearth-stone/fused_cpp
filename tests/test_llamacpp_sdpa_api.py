from __future__ import annotations

import ctypes
import math

import pytest
import torch

pytest.importorskip("fused_cpp._C")

from fused_cpp import _C  # noqa: E402


VERSION_NAME = "flash2_neon_l3kv_packqkv_pbf16pv"


def _load_llamacpp_symbol():
    lib = ctypes.CDLL(_C.__file__)
    try:
        fn = lib.fused_cpp_sdpa_flash2_neon_l3kv_packqkv_pbf16pv_fp32_llamacpp
    except AttributeError:
        pytest.skip("llama.cpp SDPA C symbol is not exported in this build")

    ptr = ctypes.c_void_p
    i64 = ctypes.c_int64
    fn.argtypes = [ptr] * 4 + [i64] * 22 + [ctypes.c_float]
    fn.restype = ctypes.c_int
    return fn


def _ptr(tensor: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(tensor.data_ptr())


@pytest.mark.equiv
def test_llamacpp_entry_scale_zero_uses_default_scale():
    if VERSION_NAME not in _C.list_sdpa_versions():
        pytest.skip(f"{VERSION_NAME} not registered (build outdated)")

    fn = _load_llamacpp_symbol()

    B, H, L, S, D, DV = 1, 2, 8, 8, 16, 8
    generator = torch.Generator(device="cpu").manual_seed(0x515D)
    q_bnld = torch.randn(B, H, L, D, generator=generator) * 2.0
    k_bnsd = torch.randn(B, H, S, D, generator=generator) * 2.0
    v_bnsd = torch.randn(B, H, S, DV, generator=generator)

    q_ggml = q_bnld.permute(0, 2, 1, 3).contiguous()
    k_ggml = k_bnsd.permute(0, 2, 1, 3).contiguous()
    v_ggml = v_bnsd.permute(0, 2, 1, 3).contiguous()

    elem = ctypes.sizeof(ctypes.c_float)
    q_nb = (elem, D * H * elem, D * elem, D * H * L * elem)
    k_nb = (elem, D * H * elem, D * elem, D * H * S * elem)
    v_nb = (elem, DV * H * elem, DV * elem, DV * H * S * elem)
    o_nb = (elem, DV * elem, DV * H * elem, DV * H * L * elem)

    def call(scale: float) -> torch.Tensor:
        out_ggml = torch.empty(B, L, H, DV, dtype=torch.float32)
        rc = fn(
            _ptr(q_ggml),
            _ptr(k_ggml),
            _ptr(v_ggml),
            _ptr(out_ggml),
            B,
            H,
            L,
            S,
            D,
            DV,
            *q_nb,
            *k_nb,
            *v_nb,
            *o_nb,
            ctypes.c_float(scale),
        )
        assert rc == 0
        return out_ggml.permute(0, 2, 1, 3).contiguous()

    out_default = call(0.0)
    out_explicit = call(1.0 / math.sqrt(D))

    torch.testing.assert_close(out_default, out_explicit, rtol=1e-6, atol=1e-6)
