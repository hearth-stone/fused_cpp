# -*- coding: utf-8 -*-
"""Tests for Tensor Parallelism (TP) support in CPUFusedMLAImpl.

Covers:
  - TP=1 regression: behaviour is identical to the pre-TP baseline.
  - TP=4 numerical correctness: four simulated ranks each hold
    num_local_heads worth of weights; after collecting partial sums and
    manually summing them (simulating all_reduce), the result must match
    the TP=1 full-heads reference on the **same** input.
  - reduce_fn=None path: when tp_size > 1 and no reduce_fn is provided the
    impl falls back to torch.distributed.all_reduce (tested via a mock).
"""
from __future__ import annotations

import math
from typing import Any, List
from unittest.mock import patch

import pytest
import torch

from fused_mla_cpp.core import CPUFusedMLAImpl


# ── Shared model dimensions ──────────────────────────────────────────────────

NUM_HEADS = 4
QK_NOPE_HEAD_DIM = 16
QK_ROPE_HEAD_DIM = 8
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 24
V_HEAD_DIM = 16
KV_LORA_RANK = 32
HIDDEN_DIM = 64
SCALE = 1.0 / math.sqrt(QK_HEAD_DIM)
TP_SIZE = 4
LOCAL_HEADS = NUM_HEADS // TP_SIZE  # = 1


# ── Fake layer helpers ───────────────────────────────────────────────────────

def _fake_linear(weight: torch.Tensor, bias: torch.Tensor | None = None) -> Any:
    """Return a minimal fake linear layer object."""
    return type("FakeLinear", (), {
        "weight": weight,
        "bias": bias,
        "skip_bias_add": False,
    })()


def _make_wrapper(
    num_heads: int,
    q_proj_weight: torch.Tensor,
    kv_a_proj_weight: torch.Tensor,
    kv_b_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    kv_a_ln_weight: torch.Tensor,
) -> Any:
    """Build a minimal wrapper object that forward_fused reads from."""
    from unittest.mock import MagicMock
    wrapper = MagicMock()
    wrapper.q_lora_rank = None
    wrapper.num_heads = num_heads
    wrapper.kv_lora_rank = KV_LORA_RANK
    wrapper.qk_nope_head_dim = QK_NOPE_HEAD_DIM
    wrapper.qk_rope_head_dim = QK_ROPE_HEAD_DIM
    wrapper.qk_head_dim = QK_HEAD_DIM
    wrapper.v_head_dim = V_HEAD_DIM
    wrapper.q_proj = _fake_linear(q_proj_weight)
    wrapper.kv_a_proj_with_mqa = _fake_linear(kv_a_proj_weight)
    wrapper.kv_b_proj = _fake_linear(kv_b_proj_weight)
    wrapper.o_proj = _fake_linear(o_proj_weight)
    wrapper.kv_a_layernorm = type("FakeLN", (), {
        "weight": kv_a_ln_weight,
        "variance_epsilon": 1e-6,
    })()
    wrapper.rotary_emb = None
    return wrapper


def _make_attn_metadata(num_tokens: int) -> Any:
    """Build a minimal attn_metadata for prefill-only (no decode)."""
    from unittest.mock import MagicMock
    meta = MagicMock()
    meta.num_decode_tokens = 0
    meta.num_decodes = 0
    meta.num_prefills = 1
    meta.slot_mapping = None

    prefill = MagicMock()
    prefill.chunked_context = None
    prefill.max_query_len = num_tokens
    prefill.query_start_loc = torch.tensor([0, num_tokens], dtype=torch.int32)
    meta.prefill = prefill
    return meta


# ── Weight factories ─────────────────────────────────────────────────────────

def _make_full_weights(seed: int = 0) -> dict:
    """Create a full set of deterministic weights for NUM_HEADS heads."""
    g = torch.Generator()
    g.manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=g)

    return {
        # [NUM_HEADS * QK_HEAD_DIM, HIDDEN_DIM]
        "q_proj_w": randn(NUM_HEADS * QK_HEAD_DIM, HIDDEN_DIM),
        # [KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_DIM]
        "kv_a_proj_w": randn(KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN_DIM),
        # [NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK]
        "kv_b_proj_w": randn(NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK),
        # [KV_LORA_RANK]
        "kv_a_ln_w": torch.ones(KV_LORA_RANK),
        # [HIDDEN_DIM, NUM_HEADS * V_HEAD_DIM]
        "o_proj_w": randn(HIDDEN_DIM, NUM_HEADS * V_HEAD_DIM),
    }


def _run_tp1(weights: dict, hidden: torch.Tensor) -> torch.Tensor:
    """Run TP=1 full-heads forward and return output."""
    num_tokens = hidden.shape[0]
    kv_b_proj = _fake_linear(weights["kv_b_proj_w"])
    impl = CPUFusedMLAImpl(
        num_heads=NUM_HEADS,
        head_size=QK_HEAD_DIM,
        scale=SCALE,
        num_kv_heads=1,
        kv_cache_dtype="auto",
        q_lora_rank=None,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        qk_head_dim=QK_HEAD_DIM,
        v_head_dim=V_HEAD_DIM,
        kv_b_proj=kv_b_proj,
        tp_size=1,
        tp_rank=0,
    )
    impl.process_weights_after_loading(act_dtype=torch.float32)
    wrapper = _make_wrapper(
        num_heads=NUM_HEADS,
        q_proj_weight=weights["q_proj_w"],
        kv_a_proj_weight=weights["kv_a_proj_w"],
        kv_b_proj_weight=weights["kv_b_proj_w"],
        o_proj_weight=weights["o_proj_w"],
        kv_a_ln_weight=weights["kv_a_ln_w"],
    )
    positions = torch.zeros(num_tokens, dtype=torch.long)
    kv_cache = torch.zeros(1, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
    return impl.forward_fused(hidden, positions, wrapper, kv_cache, _make_attn_metadata(num_tokens))


def _run_tp4_collect_partials(weights: dict, hidden: torch.Tensor) -> List[torch.Tensor]:
    """Run TP=4 forward on all 4 ranks, collect each rank's partial output.

    Each rank's reduce_fn captures the partial sum (before all_reduce) and
    returns it unchanged.  The caller is responsible for summing the partials
    to simulate all_reduce.

    Weight sharding:
      - q_proj:   shard rows  → [local_heads * QK_HEAD_DIM, HIDDEN_DIM]
      - kv_b_proj: shard rows → [local_heads * (nope + v), KV_LORA_RANK]
      - o_proj:   shard cols  → [HIDDEN_DIM, local_heads * V_HEAD_DIM]
      - kv_a_proj / kv_a_ln:  shared (not sharded)
    """
    num_tokens = hidden.shape[0]
    rows_kv_b = LOCAL_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
    rows_q = LOCAL_HEADS * QK_HEAD_DIM
    cols_o = LOCAL_HEADS * V_HEAD_DIM

    partials: List[torch.Tensor] = []

    for rank in range(TP_SIZE):
        kv_b_local = weights["kv_b_proj_w"][rank * rows_kv_b:(rank + 1) * rows_kv_b]
        q_local = weights["q_proj_w"][rank * rows_q:(rank + 1) * rows_q]
        o_local = weights["o_proj_w"][:, rank * cols_o:(rank + 1) * cols_o]

        captured: List[torch.Tensor] = []

        def make_capture(out_list):
            def reduce_fn(t: torch.Tensor) -> torch.Tensor:
                out_list.append(t.clone())
                return t
            return reduce_fn

        impl = CPUFusedMLAImpl(
            num_heads=LOCAL_HEADS,
            head_size=QK_HEAD_DIM,
            scale=SCALE,
            num_kv_heads=1,
            kv_cache_dtype="auto",
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            kv_b_proj=_fake_linear(kv_b_local),
            tp_size=TP_SIZE,
            tp_rank=rank,
            reduce_fn=make_capture(captured),
        )
        impl.process_weights_after_loading(act_dtype=torch.float32)

        wrapper = _make_wrapper(
            num_heads=LOCAL_HEADS,
            q_proj_weight=q_local,
            kv_a_proj_weight=weights["kv_a_proj_w"],
            kv_b_proj_weight=kv_b_local,
            o_proj_weight=o_local,
            kv_a_ln_weight=weights["kv_a_ln_w"],
        )
        positions = torch.zeros(num_tokens, dtype=torch.long)
        kv_cache = torch.zeros(1, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        impl.forward_fused(hidden, positions, wrapper, kv_cache, _make_attn_metadata(num_tokens))

        assert len(captured) == 1, f"rank {rank}: expected 1 partial, got {len(captured)}"
        partials.append(captured[0])

    return partials


# ── TP=1 regression tests ────────────────────────────────────────────────────

class TestTP1Regression:
    """TP=1 must behave identically to the pre-TP baseline."""

    def test_tp_size_defaults_to_1(self):
        """Default constructor has tp_size=1, tp_rank=0, reduce_fn=None."""
        kv_b_proj = _fake_linear(torch.randn(
            NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM), KV_LORA_RANK
        ))
        impl = CPUFusedMLAImpl(
            num_heads=NUM_HEADS,
            head_size=QK_HEAD_DIM,
            scale=SCALE,
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
        assert impl.tp_size == 1
        assert impl.tp_rank == 0
        assert impl.reduce_fn is None

    def test_forward_no_reduce_called_when_tp1(self):
        """When tp_size=1, reduce_fn must never be called."""
        called = []

        def spy_reduce(t: torch.Tensor) -> torch.Tensor:
            called.append(True)
            return t

        weights = _make_full_weights(seed=42)
        num_tokens = 3
        hidden = torch.randn(num_tokens, HIDDEN_DIM)

        kv_b_proj = _fake_linear(weights["kv_b_proj_w"])
        impl = CPUFusedMLAImpl(
            num_heads=NUM_HEADS,
            head_size=QK_HEAD_DIM,
            scale=SCALE,
            num_kv_heads=1,
            kv_cache_dtype="auto",
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            kv_b_proj=kv_b_proj,
            tp_size=1,
            reduce_fn=spy_reduce,
        )
        impl.process_weights_after_loading(act_dtype=torch.float32)
        wrapper = _make_wrapper(
            num_heads=NUM_HEADS,
            q_proj_weight=weights["q_proj_w"],
            kv_a_proj_weight=weights["kv_a_proj_w"],
            kv_b_proj_weight=weights["kv_b_proj_w"],
            o_proj_weight=weights["o_proj_w"],
            kv_a_ln_weight=weights["kv_a_ln_w"],
        )
        positions = torch.zeros(num_tokens, dtype=torch.long)
        kv_cache = torch.zeros(1, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM)
        impl.forward_fused(hidden, positions, wrapper, kv_cache, _make_attn_metadata(num_tokens))

        assert not called, "reduce_fn must not be called when tp_size=1"


# ── TP=4 correctness tests ───────────────────────────────────────────────────

class TestTP4Correctness:
    """Simulate TP=4 and verify numerical correctness against TP=1 reference.

    Key invariant:
        sum(partial_output[rank] for rank in 0..3) == tp1_output

    This holds because o_proj is a RowParallelLinear:
        o_proj(attn_out) = sum_rank( o_proj_local[rank] @ attn_out_local[rank] )
    """

    @pytest.mark.parametrize("num_tokens,seed", [(1, 0), (3, 1), (7, 2)])
    def test_tp4_output_matches_tp1(self, num_tokens: int, seed: int):
        """TP=4 partial sums must add up to the TP=1 reference output.

        Both TP=1 and TP=4 receive the **same** hidden tensor.
        """
        weights = _make_full_weights(seed=seed)

        # Fix a single hidden input shared by both TP=1 and TP=4.
        g = torch.Generator()
        g.manual_seed(seed + 100)
        hidden = torch.randn(num_tokens, HIDDEN_DIM, generator=g)

        ref = _run_tp1(weights, hidden)
        partials = _run_tp4_collect_partials(weights, hidden)

        assert len(partials) == TP_SIZE
        tp4_out = torch.stack(partials, dim=0).sum(dim=0)  # simulate all_reduce

        assert tp4_out.shape == ref.shape, (
            f"Shape mismatch: tp4={tp4_out.shape}, ref={ref.shape}"
        )
        max_diff = (tp4_out - ref).abs().max().item()
        assert max_diff < 1e-4, (
            f"TP=4 output differs from TP=1 reference: max_diff={max_diff:.2e}"
        )

    def test_reduce_fn_called_exactly_once_per_rank(self):
        """reduce_fn must be called exactly once per forward pass when tp_size > 1."""
        weights = _make_full_weights(seed=7)
        hidden = torch.randn(2, HIDDEN_DIM)

        for rank in range(TP_SIZE):
            call_count = [0]

            def make_fn(counter):
                def fn(t: torch.Tensor) -> torch.Tensor:
                    counter[0] += 1
                    return t
                return fn

            kv_b_local = weights["kv_b_proj_w"][:LOCAL_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)]
            impl = CPUFusedMLAImpl(
                num_heads=LOCAL_HEADS,
                head_size=QK_HEAD_DIM,
                scale=SCALE,
                num_kv_heads=1,
                kv_cache_dtype="auto",
                q_lora_rank=None,
                kv_lora_rank=KV_LORA_RANK,
                qk_nope_head_dim=QK_NOPE_HEAD_DIM,
                qk_rope_head_dim=QK_ROPE_HEAD_DIM,
                qk_head_dim=QK_HEAD_DIM,
                v_head_dim=V_HEAD_DIM,
                kv_b_proj=_fake_linear(kv_b_local),
                tp_size=TP_SIZE,
                tp_rank=rank,
                reduce_fn=make_fn(call_count),
            )
            impl.process_weights_after_loading(act_dtype=torch.float32)
            wrapper = _make_wrapper(
                num_heads=LOCAL_HEADS,
                q_proj_weight=weights["q_proj_w"][:LOCAL_HEADS * QK_HEAD_DIM],
                kv_a_proj_weight=weights["kv_a_proj_w"],
                kv_b_proj_weight=kv_b_local,
                o_proj_weight=weights["o_proj_w"][:, :LOCAL_HEADS * V_HEAD_DIM],
                kv_a_ln_weight=weights["kv_a_ln_w"],
            )
            impl.forward_fused(
                hidden,
                torch.zeros(2, dtype=torch.long),
                wrapper,
                torch.zeros(1, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
                _make_attn_metadata(2),
            )
            assert call_count[0] == 1, (
                f"rank {rank}: reduce_fn should be called exactly once, got {call_count[0]}"
            )

    def test_output_is_contiguous_before_reduce(self):
        """The tensor passed to reduce_fn must be contiguous."""
        weights = _make_full_weights(seed=13)
        received_contiguous: list = []

        def capture_reduce(t: torch.Tensor) -> torch.Tensor:
            received_contiguous.append(t.is_contiguous())
            return t

        kv_b_local = weights["kv_b_proj_w"][:LOCAL_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)]
        impl = CPUFusedMLAImpl(
            num_heads=LOCAL_HEADS,
            head_size=QK_HEAD_DIM,
            scale=SCALE,
            num_kv_heads=1,
            kv_cache_dtype="auto",
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            kv_b_proj=_fake_linear(kv_b_local),
            tp_size=TP_SIZE,
            tp_rank=0,
            reduce_fn=capture_reduce,
        )
        impl.process_weights_after_loading(act_dtype=torch.float32)
        wrapper = _make_wrapper(
            num_heads=LOCAL_HEADS,
            q_proj_weight=weights["q_proj_w"][:LOCAL_HEADS * QK_HEAD_DIM],
            kv_a_proj_weight=weights["kv_a_proj_w"],
            kv_b_proj_weight=kv_b_local,
            o_proj_weight=weights["o_proj_w"][:, :LOCAL_HEADS * V_HEAD_DIM],
            kv_a_ln_weight=weights["kv_a_ln_w"],
        )
        impl.forward_fused(
            torch.randn(2, HIDDEN_DIM),
            torch.zeros(2, dtype=torch.long),
            wrapper,
            torch.zeros(1, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
            _make_attn_metadata(2),
        )
        assert received_contiguous, "reduce_fn was not called"
        assert all(received_contiguous), "Tensor passed to reduce_fn must be contiguous"

    def test_fallback_all_reduce_called_when_reduce_fn_none(self):
        """When reduce_fn=None and tp_size > 1, torch.distributed.all_reduce is called."""
        weights = _make_full_weights(seed=99)
        kv_b_local = weights["kv_b_proj_w"][:LOCAL_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)]
        impl = CPUFusedMLAImpl(
            num_heads=LOCAL_HEADS,
            head_size=QK_HEAD_DIM,
            scale=SCALE,
            num_kv_heads=1,
            kv_cache_dtype="auto",
            q_lora_rank=None,
            kv_lora_rank=KV_LORA_RANK,
            qk_nope_head_dim=QK_NOPE_HEAD_DIM,
            qk_rope_head_dim=QK_ROPE_HEAD_DIM,
            qk_head_dim=QK_HEAD_DIM,
            v_head_dim=V_HEAD_DIM,
            kv_b_proj=_fake_linear(kv_b_local),
            tp_size=TP_SIZE,
            tp_rank=0,
            reduce_fn=None,
        )
        impl.process_weights_after_loading(act_dtype=torch.float32)
        wrapper = _make_wrapper(
            num_heads=LOCAL_HEADS,
            q_proj_weight=weights["q_proj_w"][:LOCAL_HEADS * QK_HEAD_DIM],
            kv_a_proj_weight=weights["kv_a_proj_w"],
            kv_b_proj_weight=kv_b_local,
            o_proj_weight=weights["o_proj_w"][:, :LOCAL_HEADS * V_HEAD_DIM],
            kv_a_ln_weight=weights["kv_a_ln_w"],
        )

        with patch("torch.distributed.all_reduce") as mock_all_reduce:
            impl.forward_fused(
                torch.randn(2, HIDDEN_DIM),
                torch.zeros(2, dtype=torch.long),
                wrapper,
                torch.zeros(1, 1, KV_LORA_RANK + QK_ROPE_HEAD_DIM),
                _make_attn_metadata(2),
            )
            mock_all_reduce.assert_called_once()
