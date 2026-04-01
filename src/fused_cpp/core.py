# -*- coding: utf-8 -*-
"""Backward-compatible shim: re-exports everything from fused_cpp.mla.impl.

This module is kept for backward compatibility only. New code should import
directly from ``fused_cpp.mla`` or ``fused_cpp.mla.impl``.

When this module is reloaded (e.g. by tests that toggle env vars),
``fused_cpp.mla.impl`` is also reloaded so that module-level flags like
``_DEBUG_USE_ORIG_RMSNORM`` are re-evaluated.
"""
import importlib
import fused_cpp.mla.impl as _impl_mod

importlib.reload(_impl_mod)

from fused_cpp.mla.impl import (  # noqa: F401, E402
    CPUFusedMLAImpl,
    _HAS_CPP,
    _parallel_map,
    _pytorch_apply_rope,
    _pytorch_concat_k_nope_k_pe,
    _pytorch_gather_kv_cache,
    _pytorch_merge_attn_states,
    _pytorch_rms_norm,
    _pytorch_write_kv_cache,
    _DEBUG_USE_ORIG_RMSNORM,
    _DEBUG_USE_ORIG_ROPE,
)
