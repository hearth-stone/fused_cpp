# -*- coding: utf-8 -*-
"""Tests for _parallel_map utility and CPUFusedMLAImpl thread-count awareness.

Covers:
- _parallel_map serial fallback (num_threads <= 1 or single item)
- _parallel_map parallel execution (num_threads > 1, multiple items)
- Result ordering correctness
- FUSED_MLA_NUM_THREADS env var override
- torch.get_num_threads() default path
"""
import os
import threading

import pytest
import torch

import fused_cpp.core as core_module
from fused_cpp.mla.impl import _parallel_map


# ── Constants shared across tests ────────────────────────────────────────────

NUM_HEADS = 2
KV_LORA_RANK = 16
QK_NOPE_HEAD_DIM = 8
QK_ROPE_HEAD_DIM = 8
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = 8
HEAD_SIZE = KV_LORA_RANK + QK_ROPE_HEAD_DIM


def _make_impl():
    """Build a minimal CPUFusedMLAImpl instance."""
    kv_b_proj_weight = torch.randn(
        NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK
    )
    kv_b_proj = type("FakeLinear", (), {"weight": kv_b_proj_weight, "bias": None})()
    return core_module.CPUFusedMLAImpl(
        num_heads=NUM_HEADS,
        head_size=HEAD_SIZE,
        scale=0.125,
        num_kv_heads=1,
        kv_cache_dtype="auto",
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        kv_b_proj=kv_b_proj,
    )


# ── Tests for _parallel_map ───────────────────────────────────────────────────

class TestParallelMap:
    """Unit tests for the _parallel_map utility function."""

    def test_serial_fallback_num_threads_one(self):
        """num_threads=1 must execute serially and return correct results."""
        results = _parallel_map(lambda x: x * 2, [1, 2, 3], num_threads=1)
        assert results == [2, 4, 6]

    def test_serial_fallback_single_item(self):
        """Single-item list must not spawn threads regardless of num_threads."""
        call_count = []

        def fn(x):
            call_count.append(threading.current_thread().ident)
            return x + 10

        results = _parallel_map(fn, [5], num_threads=4)
        assert results == [15]
        # All calls happened in the main thread (serial path)
        assert all(tid == threading.main_thread().ident for tid in call_count)

    def test_parallel_execution_result_order(self):
        """Parallel execution must preserve input order in the output."""
        items = list(range(10))
        results = _parallel_map(lambda x: x ** 2, items, num_threads=4)
        assert results == [x ** 2 for x in items]

    def test_parallel_execution_uses_worker_threads(self):
        """With num_threads > 1 and multiple items, workers run in non-main threads."""
        thread_ids = []
        lock = threading.Lock()

        def fn(x):
            with lock:
                thread_ids.append(threading.current_thread().ident)
            return x

        _parallel_map(fn, list(range(8)), num_threads=4)
        # At least some calls should have happened in non-main threads
        main_tid = threading.main_thread().ident
        assert any(tid != main_tid for tid in thread_ids)

    def test_parallel_map_sets_num_threads_one_in_worker(self):
        """Each worker thread must call torch.set_num_threads(1)."""
        observed = []
        lock = threading.Lock()

        def fn(x):
            with lock:
                observed.append(torch.get_num_threads())
            return x

        _parallel_map(fn, list(range(4)), num_threads=4)
        # Every worker should see num_threads == 1
        assert all(n == 1 for n in observed)

    def test_parallel_map_tensor_operations(self):
        """_parallel_map must produce numerically identical results to serial."""
        torch.manual_seed(42)
        tensors = [torch.randn(16, 16) for _ in range(6)]

        def fn(t):
            return t.float().pow(2).mean()

        serial = [fn(t) for t in tensors]
        parallel = _parallel_map(fn, tensors, num_threads=3)

        for s, p in zip(serial, parallel):
            assert torch.allclose(s, p, atol=1e-6), (
                f"Numerical mismatch: serial={s.item():.6f}, parallel={p.item():.6f}"
            )

    def test_empty_list(self):
        """Empty input must return an empty list without errors."""
        results = _parallel_map(lambda x: x, [], num_threads=4)
        assert results == []


# ── Tests for CPUFusedMLAImpl thread-count initialization ─────────────────────

class TestNumThreadsInit:
    """Tests for FUSED_MLA_NUM_THREADS env var and torch.get_num_threads() path."""

    def test_default_uses_torch_get_num_threads(self, monkeypatch):
        """Without env var, num_threads must equal torch.get_num_threads()."""
        monkeypatch.delenv("FUSED_MLA_NUM_THREADS", raising=False)
        impl = _make_impl()
        assert impl.num_threads == torch.get_num_threads()

    def test_env_var_positive_overrides(self, monkeypatch):
        """FUSED_MLA_NUM_THREADS=N (N>0) must set num_threads to N."""
        monkeypatch.setenv("FUSED_MLA_NUM_THREADS", "7")
        impl = _make_impl()
        assert impl.num_threads == 7

    def test_env_var_zero_falls_back_to_torch(self, monkeypatch):
        """FUSED_MLA_NUM_THREADS=0 must fall back to torch.get_num_threads()."""
        monkeypatch.setenv("FUSED_MLA_NUM_THREADS", "0")
        impl = _make_impl()
        assert impl.num_threads == torch.get_num_threads()

    def test_env_var_one_sets_serial_mode(self, monkeypatch):
        """FUSED_MLA_NUM_THREADS=1 must set num_threads to 1 (serial mode)."""
        monkeypatch.setenv("FUSED_MLA_NUM_THREADS", "1")
        impl = _make_impl()
        assert impl.num_threads == 1

    def test_env_var_invalid_falls_back_to_torch(self, monkeypatch):
        """Non-integer FUSED_MLA_NUM_THREADS must fall back to torch.get_num_threads()."""
        monkeypatch.setenv("FUSED_MLA_NUM_THREADS", "abc")
        impl = _make_impl()
        assert impl.num_threads == torch.get_num_threads()

    def test_num_threads_is_positive_integer(self, monkeypatch):
        """num_threads must always be a positive integer."""
        monkeypatch.delenv("FUSED_MLA_NUM_THREADS", raising=False)
        impl = _make_impl()
        assert isinstance(impl.num_threads, int)
        assert impl.num_threads >= 1
