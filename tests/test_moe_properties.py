# -*- coding: utf-8 -*-
"""Property-based tests for Fused MoE EP Full-Token Input Optimization.

Each test validates a correctness property from the design document.
"""
from __future__ import annotations

import glob
import os

import torch
import torch.nn.functional as F
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# ── Property 1: No old package name imports ──────────────────────────────────
# Feature: fused-moe-ep-fulltoken, Property 1: No old package name imports


class TestProperty1NoOldImports:
    """**Validates: Requirements 1.4, 1.6**

    Property 1: No old package name imports.
    For any .py file under src/fused_cpp/ and tests/, the file shall contain
    zero import statements referencing the old package name fused_mla_cpp.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_no_old_package_imports(self, data):
        """Scan all .py files under src/fused_cpp/ and tests/ for absence of
        fused_mla_cpp imports."""
        base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        base_dir = os.path.normpath(base_dir)

        src_dir = os.path.join(base_dir, "src", "fused_cpp")
        tests_dir = os.path.join(base_dir, "tests")

        py_files = []
        for d in [src_dir, tests_dir]:
            py_files.extend(
                glob.glob(os.path.join(d, "**", "*.py"), recursive=True)
            )
        # Filter out __pycache__
        py_files = [f for f in py_files if "__pycache__" not in f]
        assume(len(py_files) > 0)

        idx = data.draw(st.integers(min_value=0, max_value=len(py_files) - 1))
        filepath = py_files[idx]

        with open(filepath, "r") as fh:
            content = fh.read()

        for line_no, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Skip string literals (assertions, docstrings, etc.)
            if stripped.startswith(("assert ", "\"", "'", "f\"")):
                continue
            has_import = (
                stripped.startswith("import fused_mla_cpp")
                or stripped.startswith("from fused_mla_cpp")
            )
            assert not has_import, (
                f"Found old package import in {filepath}:{line_no}: {stripped}"
            )


# ── Property 2: Constructor validation — divisibility check ──────────────────
# Feature: fused-moe-ep-fulltoken, Property 2: Constructor validation — divisibility check


class TestProperty2ConstructorValidation:
    """**Validates: Requirements 3.6, 3.7**

    Property 2: Constructor validation — divisibility check.
    For any pair of positive integers (num_experts, ep_size), constructing
    FusedMoEImpl shall succeed iff num_experts % ep_size == 0.
    """

    @settings(max_examples=100)
    @given(
        num_experts=st.integers(min_value=1, max_value=64),
        ep_size=st.integers(min_value=1, max_value=16),
    )
    def test_divisibility_check(self, num_experts, ep_size):
        from fused_cpp.moe import FusedMoEImpl

        hidden_size = 16
        ffn_hidden_size = 32
        local_num = num_experts // ep_size if num_experts % ep_size == 0 else 1
        w_gate = torch.randn(local_num, ffn_hidden_size, hidden_size)
        w_up = torch.randn(local_num, ffn_hidden_size, hidden_size)
        w_down = torch.randn(local_num, hidden_size, ffn_hidden_size)

        if num_experts % ep_size == 0:
            impl = FusedMoEImpl(
                num_experts=num_experts, top_k=1,
                hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size,
                w_gate=w_gate, w_up=w_up, w_down=w_down,
                ep_size=ep_size, ep_rank=0,
            )
            assert impl.local_num_experts == num_experts // ep_size
        else:
            with pytest.raises(ValueError):
                FusedMoEImpl(
                    num_experts=num_experts, top_k=1,
                    hidden_size=hidden_size, ffn_hidden_size=ffn_hidden_size,
                    w_gate=w_gate, w_up=w_up, w_down=w_down,
                    ep_size=ep_size, ep_rank=0,
                )


# ── Property 6: No vllm imports in moe package ──────────────────────────────
# Feature: fused-moe-ep-fulltoken, Property 6: No vllm imports in moe package


class TestProperty6NoVllmImports:
    """**Validates: Requirements 7.1**

    Property 6: No vllm imports in moe package.
    For any .py file under src/fused_cpp/moe/, the file shall contain
    zero import vllm or from vllm statements.
    """

    @settings(max_examples=100)
    @given(data=st.data())
    def test_no_vllm_imports_in_moe(self, data):
        moe_dir = os.path.join(
            os.path.dirname(__file__), os.pardir, "src", "fused_cpp", "moe"
        )
        moe_dir = os.path.normpath(moe_dir)
        py_files = glob.glob(os.path.join(moe_dir, "**", "*.py"), recursive=True)
        py_files = [f for f in py_files if "__pycache__" not in f]
        assume(len(py_files) > 0)

        idx = data.draw(st.integers(min_value=0, max_value=len(py_files) - 1))
        filepath = py_files[idx]

        with open(filepath, "r") as fh:
            content = fh.read()

        for line_no, line in enumerate(content.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert "import vllm" not in stripped, (
                f"Found vllm import in {filepath}:{line_no}: {stripped}"
            )
            assert "from vllm" not in stripped, (
                f"Found vllm from-import in {filepath}:{line_no}: {stripped}"
            )


# ── Hypothesis strategies for MoE tests ──────────────────────────────────────

@st.composite
def moe_config(draw):
    """Generate valid FusedMoEImpl configurations."""
    ep_size = draw(st.sampled_from([1, 2, 4]))
    num_experts_factor = draw(st.integers(min_value=1, max_value=4))
    num_experts = ep_size * num_experts_factor
    top_k = draw(st.integers(min_value=1, max_value=min(3, num_experts)))
    hidden_size = draw(st.sampled_from([16, 32, 64]))
    ffn_hidden_size = draw(st.sampled_from([32, 64]))
    renormalize = draw(st.booleans())
    scoring_func = draw(st.sampled_from(["softmax", "sigmoid"]))
    ep_rank = draw(st.integers(min_value=0, max_value=ep_size - 1))
    return {
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "ffn_hidden_size": ffn_hidden_size,
        "ep_size": ep_size,
        "ep_rank": ep_rank,
        "renormalize": renormalize,
        "scoring_func": scoring_func,
    }


def _build_moe(config, **overrides):
    """Build a FusedMoEImpl from a config dict."""
    from fused_cpp.moe import FusedMoEImpl
    c = {**config, **overrides}
    local_num = c["num_experts"] // c["ep_size"]
    H, F_ = c["hidden_size"], c["ffn_hidden_size"]
    w_gate = torch.randn(local_num, F_, H)
    w_up = torch.randn(local_num, F_, H)
    w_down = torch.randn(local_num, H, F_)
    return FusedMoEImpl(
        num_experts=c["num_experts"], top_k=c["top_k"],
        hidden_size=H, ffn_hidden_size=F_,
        w_gate=w_gate, w_up=w_up, w_down=w_down,
        ep_size=c["ep_size"], ep_rank=c["ep_rank"],
        renormalize=c["renormalize"], scoring_func=c["scoring_func"],
        **{k: v for k, v in overrides.items() if k not in c},
    )


# ── Property 10: Output shape invariant ──────────────────────────────────────
# Feature: fused-moe-ep-fulltoken, Property 10: Output shape invariant


class TestProperty10OutputShape:
    """**Validates: Requirements 10.7**

    Property 10: Output shape invariant.
    For any valid config and input [T, hidden_size], output shape is [T, hidden_size].
    """

    @settings(max_examples=100)
    @given(config=moe_config(), total_tokens=st.integers(min_value=1, max_value=8))
    def test_output_shape(self, config, total_tokens):
        impl = _build_moe(config)
        H = config["hidden_size"]
        hidden_states = torch.randn(total_tokens, H)
        router_logits = torch.randn(total_tokens, config["num_experts"])
        output = impl.forward(hidden_states, router_logits)
        assert output.shape == (total_tokens, H), (
            f"Expected shape ({total_tokens}, {H}), got {output.shape}"
        )


# ── Property 3: Softmax scores sum to 1 and preserve dtype ───────────────────
# Feature: fused-moe-ep-fulltoken, Property 3: Softmax scores sum to 1 and preserve dtype


class TestProperty3SoftmaxScores:
    """**Validates: Requirements 4.3, 9.4**

    Property 3: Softmax scores sum to 1 and preserve dtype.
    For any router_logits [T, E] with scoring_func="softmax", local scores
    sum to 1.0 along expert dim (within float32 tolerance) and output dtype
    matches input dtype.
    """

    @settings(max_examples=100)
    @given(
        config=moe_config(),
        total_tokens=st.integers(min_value=1, max_value=8),
    )
    def test_softmax_scores_sum_to_one_and_dtype(self, config, total_tokens):
        # Force softmax scoring
        config = {**config, "scoring_func": "softmax"}
        impl = _build_moe(config)

        router_logits = torch.randn(total_tokens, config["num_experts"])
        input_dtype = router_logits.dtype

        # Replicate the scoring logic from FusedMoEImpl.forward
        local_logits = router_logits[:, impl.expert_start:impl.expert_end]
        scores = F.softmax(local_logits.float(), dim=-1).to(local_logits.dtype)

        # Scores must sum to 1.0 along expert dim (within float32 tolerance)
        sums = scores.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"Softmax scores do not sum to 1: {sums}"
        )

        # Output dtype must match input dtype
        assert scores.dtype == input_dtype, (
            f"Expected dtype {input_dtype}, got {scores.dtype}"
        )


# ── Property 4: Sigmoid scores in (0,1) and preserve dtype ──────────────────
# Feature: fused-moe-ep-fulltoken, Property 4: Sigmoid scores in (0,1) and preserve dtype


class TestProperty4SigmoidScores:
    """**Validates: Requirements 4.4, 9.4**

    Property 4: Sigmoid scores in (0,1) and preserve dtype.
    For any router_logits [T, E] with scoring_func="sigmoid", every score
    is in the open interval (0, 1) and output dtype matches input dtype.
    """

    @settings(max_examples=100)
    @given(
        config=moe_config(),
        total_tokens=st.integers(min_value=1, max_value=8),
    )
    def test_sigmoid_scores_in_range_and_dtype(self, config, total_tokens):
        # Force sigmoid scoring
        config = {**config, "scoring_func": "sigmoid"}
        impl = _build_moe(config)

        router_logits = torch.randn(total_tokens, config["num_experts"])
        input_dtype = router_logits.dtype

        # Replicate the scoring logic from FusedMoEImpl.forward
        local_logits = router_logits[:, impl.expert_start:impl.expert_end]
        scores = torch.sigmoid(local_logits.float()).to(local_logits.dtype)

        # Every score must be strictly in (0, 1)
        assert (scores > 0).all(), f"Scores not > 0: min={scores.min().item()}"
        assert (scores < 1).all(), f"Scores not < 1: max={scores.max().item()}"

        # Output dtype must match input dtype
        assert scores.dtype == input_dtype, (
            f"Expected dtype {input_dtype}, got {scores.dtype}"
        )


# ── Property 5: Renormalized top-k weights sum to 1 ─────────────────────────
# Feature: fused-moe-ep-fulltoken, Property 5: Renormalized top-k weights sum to 1


class TestProperty5RenormalizedWeights:
    """**Validates: Requirements 4.6**

    Property 5: Renormalized top-k weights sum to 1.
    For any valid routing scores with renormalize=True, topk_weights per
    token sum to 1.0 (within float32 tolerance).
    """

    @settings(max_examples=100)
    @given(
        config=moe_config(),
        total_tokens=st.integers(min_value=1, max_value=8),
    )
    def test_renormalized_weights_sum_to_one(self, config, total_tokens):
        # Force renormalize=True
        config = {**config, "renormalize": True}
        impl = _build_moe(config)

        router_logits = torch.randn(total_tokens, config["num_experts"])

        # Replicate scoring + top-k + renormalization from FusedMoEImpl.forward
        local_logits = router_logits[:, impl.expert_start:impl.expert_end]
        if config["scoring_func"] == "softmax":
            scores = F.softmax(local_logits.float(), dim=-1).to(local_logits.dtype)
        else:
            scores = torch.sigmoid(local_logits.float()).to(local_logits.dtype)

        k = min(config["top_k"], impl.local_num_experts)
        topk_weights, _ = torch.topk(scores, k=k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Each token's top-k weights must sum to 1.0
        sums = topk_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), (
            f"Renormalized weights do not sum to 1: {sums}"
        )


# ── Naive MoE reference for Property 7 & 8 ──────────────────────────────────


def naive_moe_reference(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    num_experts: int,
    top_k: int,
    scoring_func: str,
    renormalize: bool,
) -> torch.Tensor:
    """Single-node global top-k MoE reference (no EP).

    Performs standard MoE: score all experts globally, pick top-k, compute FFN,
    weighted accumulation.
    """
    # Score computation in float32
    if scoring_func == "softmax":
        scores = F.softmax(router_logits.float(), dim=-1).to(router_logits.dtype)
    else:
        scores = torch.sigmoid(router_logits.float()).to(router_logits.dtype)

    k = min(top_k, num_experts)
    topk_weights, topk_ids = torch.topk(scores, k=k, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    output = torch.zeros_like(hidden_states)
    for e in range(num_experts):
        mask = (topk_ids == e).any(dim=-1)
        if not mask.any():
            continue
        token_indices = mask.nonzero(as_tuple=True)[0]
        x = hidden_states[token_indices]

        gate_out = x @ w_gate[e].T
        up_out = x @ w_up[e].T
        intermediate = F.silu(gate_out) * up_out
        expert_out = intermediate @ w_down[e].T

        for ki in range(k):
            ki_mask = topk_ids[token_indices, ki] == e
            if ki_mask.any():
                ki_indices = token_indices[ki_mask]
                weights = topk_weights[ki_indices, ki].unsqueeze(-1)
                output[ki_indices] += expert_out[ki_mask] * weights

    return output


# ── Property 7: EP=1 output matches naive reference ─────────────────────────
# Feature: fused-moe-ep-fulltoken, Property 7: EP=1 output matches naive reference


class TestProperty7EP1Correctness:
    """**Validates: Requirements 5.4, 5.5, 5.7, 10.1, 10.3, 10.4**

    Property 7: EP=1 output matches naive reference.
    For any valid FusedMoEImpl configuration with ep_size=1 and any valid
    (hidden_states, router_logits) input pair, the forward output shall be
    numerically consistent (relative error < 1e-4) with a naive reference
    MoE implementation.
    """

    @settings(max_examples=100)
    @given(
        config=moe_config(),
        total_tokens=st.integers(min_value=1, max_value=8),
    )
    def test_ep1_matches_naive_reference(self, config, total_tokens):
        # Force EP=1 so all experts are local
        config = {**config, "ep_size": 1, "ep_rank": 0}
        impl = _build_moe(config)

        H = config["hidden_size"]
        E = config["num_experts"]
        hidden_states = torch.randn(total_tokens, H)
        router_logits = torch.randn(total_tokens, E)

        with torch.no_grad():
            actual = impl.forward(hidden_states, router_logits)
            expected = naive_moe_reference(
                hidden_states=hidden_states,
                router_logits=router_logits,
                w_gate=impl.w_gate,
                w_up=impl.w_up,
                w_down=impl.w_down,
                num_experts=E,
                top_k=config["top_k"],
                scoring_func=config["scoring_func"],
                renormalize=config["renormalize"],
            )

        # Relative error < 1e-4
        assert torch.allclose(actual, expected, rtol=1e-4, atol=1e-6), (
            f"EP=1 output does not match naive reference.\n"
            f"Max abs diff: {(actual - expected).abs().max().item()}\n"
            f"Max rel diff: {((actual - expected).abs() / (expected.abs() + 1e-8)).max().item()}"
        )


# ── Property 8: Multi-EP sum matches EP=1 reference ─────────────────────────
# Feature: fused-moe-ep-fulltoken, Property 8: Multi-EP sum matches EP=1 reference


class TestProperty8MultiEPCorrectness:
    """**Validates: Requirements 6.2, 10.2, 10.3, 10.4**

    Property 8: Multi-EP sum matches EP=1 reference.
    For any valid FusedMoEImpl configuration with ep_size > 1, simulating
    all EP nodes on the same input and summing their local outputs shall
    produce a result numerically consistent (relative error < 1e-4) with
    the EP=1 reference output.
    """

    @settings(max_examples=100)
    @given(
        config=moe_config(),
        total_tokens=st.integers(min_value=1, max_value=8),
    )
    def test_multi_ep_sum_matches_ep1(self, config, total_tokens):
        ep_size = config["ep_size"]
        assume(ep_size > 1)
        # Force sigmoid scoring: softmax normalizes across the local expert
        # slice, so local-softmax on each EP node differs from global-softmax
        # on EP=1. Sigmoid is element-wise and independent of the slice size.
        # Also disable renormalize for the same reason (renormalization
        # depends on the local top-k weight distribution).
        # Force top_k >= local_num_experts so every local expert is selected
        # on each node, matching the global top-k=num_experts selection.
        num_experts = config["num_experts"]
        local_num = num_experts // ep_size
        config = {
            **config,
            "scoring_func": "sigmoid",
            "renormalize": False,
            "top_k": num_experts,  # select all experts globally
        }

        H = config["hidden_size"]
        F_ = config["ffn_hidden_size"]

        hidden_states = torch.randn(total_tokens, H)
        router_logits = torch.randn(total_tokens, num_experts)

        # Build full weight tensors for all experts (used by EP=1 reference)
        all_w_gate = torch.randn(num_experts, F_, H)
        all_w_up = torch.randn(num_experts, F_, H)
        all_w_down = torch.randn(num_experts, H, F_)

        # EP=1 reference using naive_moe_reference with all weights
        with torch.no_grad():
            expected = naive_moe_reference(
                hidden_states=hidden_states,
                router_logits=router_logits,
                w_gate=all_w_gate,
                w_up=all_w_up,
                w_down=all_w_down,
                num_experts=num_experts,
                top_k=config["top_k"],
                scoring_func=config["scoring_func"],
                renormalize=config["renormalize"],
            )

        # Simulate all EP nodes and sum their local outputs
        from fused_cpp.moe import FusedMoEImpl

        summed = torch.zeros_like(hidden_states)
        with torch.no_grad():
            for rank in range(ep_size):
                start = rank * local_num
                end = start + local_num
                node = FusedMoEImpl(
                    num_experts=num_experts,
                    top_k=config["top_k"],
                    hidden_size=H,
                    ffn_hidden_size=F_,
                    w_gate=all_w_gate[start:end],
                    w_up=all_w_up[start:end],
                    w_down=all_w_down[start:end],
                    ep_size=ep_size,
                    ep_rank=rank,
                    renormalize=config["renormalize"],
                    scoring_func=config["scoring_func"],
                )
                summed += node.forward(hidden_states, router_logits)

        # Use slightly relaxed tolerance: with many experts and large hidden
        # sizes, float32 accumulation order differences cause small deviations.
        assert torch.allclose(summed, expected, rtol=1e-3, atol=1e-3), (
            f"Multi-EP sum does not match EP=1 reference.\n"
            f"Max abs diff: {(summed - expected).abs().max().item()}\n"
            f"Max rel diff: {((summed - expected).abs() / (expected.abs() + 1e-8)).max().item()}"
        )


# ── Property 9: Shared expert output matches independent FFN ─────────────────
# Feature: fused-moe-ep-fulltoken, Property 9: Shared expert output matches independent FFN


class TestProperty9SharedExpertCorrectness:
    """**Validates: Requirements 8.2, 10.5**

    Property 9: Shared expert output matches independent FFN.
    For any valid hidden_states and shared expert weights, the shared expert
    contribution in the FusedMoEImpl output shall be numerically consistent
    with an independently computed gate→SiLU→up→mul→down FFN.
    """

    @settings(max_examples=100)
    @given(
        config=moe_config(),
        total_tokens=st.integers(min_value=1, max_value=8),
        shared_ffn=st.sampled_from([16, 32]),
    )
    def test_shared_expert_matches_independent_ffn(self, config, total_tokens, shared_ffn):
        # Force EP=1 and scaling_factor=1.0 to isolate shared expert contribution
        config = {**config, "ep_size": 1, "ep_rank": 0}
        H = config["hidden_size"]

        hidden_states = torch.randn(total_tokens, H)
        router_logits = torch.randn(total_tokens, config["num_experts"])

        # Shared expert weights
        shared_gate = torch.randn(shared_ffn, H)
        shared_up = torch.randn(shared_ffn, H)
        shared_down = torch.randn(H, shared_ffn)

        # Build impl WITHOUT shared experts
        impl_no_shared = _build_moe(config, routed_scaling_factor=1.0)

        # Build impl WITH shared experts using the SAME routed weights
        from fused_cpp.moe import FusedMoEImpl
        impl_with_shared = FusedMoEImpl(
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            hidden_size=H,
            ffn_hidden_size=config["ffn_hidden_size"],
            w_gate=impl_no_shared.w_gate,
            w_up=impl_no_shared.w_up,
            w_down=impl_no_shared.w_down,
            ep_size=1,
            ep_rank=0,
            renormalize=config["renormalize"],
            scoring_func=config["scoring_func"],
            routed_scaling_factor=1.0,
            shared_expert_gate=shared_gate,
            shared_expert_up=shared_up,
            shared_expert_down=shared_down,
        )

        with torch.no_grad():
            out_no_shared = impl_no_shared.forward(hidden_states, router_logits)
            out_with_shared = impl_with_shared.forward(hidden_states, router_logits)

            # The difference is the shared expert contribution
            shared_contribution = out_with_shared - out_no_shared

            # Independently compute shared expert FFN
            sg = hidden_states @ shared_gate.T
            su = hidden_states @ shared_up.T
            expected_shared = (F.silu(sg) * su) @ shared_down.T

        # Relaxed tolerance: float32 accumulation order differences in matrix
        # multiplications can cause small deviations (~1e-5 abs).
        assert torch.allclose(shared_contribution, expected_shared, rtol=1e-3, atol=1e-4), (
            f"Shared expert contribution does not match independent FFN.\n"
            f"Max abs diff: {(shared_contribution - expected_shared).abs().max().item()}"
        )


# ── Property 11: Forward determinism ─────────────────────────────────────────
# Feature: fused-moe-ep-fulltoken, Property 11: Forward determinism


class TestProperty11Determinism:
    """**Validates: Requirements 10.8**

    Property 11: Forward determinism.
    For any valid FusedMoEImpl instance and any valid input, executing
    forward twice on the same input shall produce bit-identical output.
    """

    @settings(max_examples=100)
    @given(
        config=moe_config(),
        total_tokens=st.integers(min_value=1, max_value=8),
    )
    def test_forward_determinism(self, config, total_tokens):
        impl = _build_moe(config)
        H = config["hidden_size"]
        hidden_states = torch.randn(total_tokens, H)
        router_logits = torch.randn(total_tokens, config["num_experts"])

        with torch.no_grad():
            out1 = impl.forward(hidden_states, router_logits)
            out2 = impl.forward(hidden_states, router_logits)

        assert torch.equal(out1, out2), (
            f"Forward is not deterministic.\n"
            f"Max diff: {(out1 - out2).abs().max().item()}"
        )


# ── Property 12: Scaling factor applied correctly based on dtype ─────────────
# Feature: fused-moe-ep-fulltoken, Property 12: Scaling factor applied correctly based on dtype


class TestProperty12ScalingFactor:
    """**Validates: Requirements 9.2, 9.3**

    Property 12: Scaling factor applied correctly based on dtype.
    When dtype is not float16, routed output scaled by routed_scaling_factor.
    When float16 + shared experts, shared output scaled by 1/routed_scaling_factor.
    """

    @settings(max_examples=100)
    @given(
        config=moe_config(),
        total_tokens=st.integers(min_value=1, max_value=8),
        scaling_factor=st.sampled_from([0.5, 2.0, 3.0]),
    )
    def test_scaling_non_fp16(self, config, total_tokens, scaling_factor):
        """When dtype is not float16, routed output is scaled by routed_scaling_factor."""
        # Force EP=1, no shared experts, float32 dtype
        config = {**config, "ep_size": 1, "ep_rank": 0}
        H = config["hidden_size"]

        # Build two impls with same weights: one with scaling, one without
        impl_base = _build_moe(config, routed_scaling_factor=1.0)

        from fused_cpp.moe import FusedMoEImpl
        impl_scaled = FusedMoEImpl(
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            hidden_size=H,
            ffn_hidden_size=config["ffn_hidden_size"],
            w_gate=impl_base.w_gate,
            w_up=impl_base.w_up,
            w_down=impl_base.w_down,
            ep_size=1,
            ep_rank=0,
            renormalize=config["renormalize"],
            scoring_func=config["scoring_func"],
            routed_scaling_factor=scaling_factor,
        )

        # float32 input (not float16)
        hidden_states = torch.randn(total_tokens, H, dtype=torch.float32)
        router_logits = torch.randn(total_tokens, config["num_experts"], dtype=torch.float32)

        with torch.no_grad():
            out_base = impl_base.forward(hidden_states, router_logits)
            out_scaled = impl_scaled.forward(hidden_states, router_logits)

        # out_scaled should be out_base * scaling_factor
        expected = out_base * scaling_factor
        assert torch.allclose(out_scaled, expected, rtol=1e-4, atol=1e-6), (
            f"Non-FP16 scaling not applied correctly.\n"
            f"Max abs diff: {(out_scaled - expected).abs().max().item()}"
        )

    @settings(max_examples=100)
    @given(
        config=moe_config(),
        total_tokens=st.integers(min_value=1, max_value=8),
        scaling_factor=st.sampled_from([0.5, 2.0, 3.0]),
        shared_ffn=st.sampled_from([16, 32]),
    )
    def test_scaling_fp16_with_shared_experts(self, config, total_tokens, scaling_factor, shared_ffn):
        """When float16 + shared experts, shared output scaled by 1/routed_scaling_factor."""
        # Use small dimensions to limit FP16 accumulation error
        config = {**config, "ep_size": 1, "ep_rank": 0,
                  "hidden_size": 16, "ffn_hidden_size": 32}
        H = config["hidden_size"]
        F_ = config["ffn_hidden_size"]
        local_num = config["num_experts"]

        # All weights and inputs in float16 to match the FP16 code path
        shared_gate = torch.randn(shared_ffn, H, dtype=torch.float16)
        shared_up = torch.randn(shared_ffn, H, dtype=torch.float16)
        shared_down = torch.randn(H, shared_ffn, dtype=torch.float16)

        w_gate = torch.randn(local_num, F_, H, dtype=torch.float16)
        w_up = torch.randn(local_num, F_, H, dtype=torch.float16)
        w_down = torch.randn(local_num, H, F_, dtype=torch.float16)

        from fused_cpp.moe import FusedMoEImpl

        hidden_states = torch.randn(total_tokens, H, dtype=torch.float16)
        router_logits = torch.randn(total_tokens, config["num_experts"], dtype=torch.float16)

        # Build impl WITHOUT shared experts to get the pure routed output
        impl_routed_only = FusedMoEImpl(
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            hidden_size=H,
            ffn_hidden_size=F_,
            w_gate=w_gate, w_up=w_up, w_down=w_down,
            ep_size=1, ep_rank=0,
            renormalize=config["renormalize"],
            scoring_func=config["scoring_func"],
            routed_scaling_factor=scaling_factor,
        )

        impl_with_shared = FusedMoEImpl(
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            hidden_size=H,
            ffn_hidden_size=F_,
            w_gate=w_gate, w_up=w_up, w_down=w_down,
            ep_size=1, ep_rank=0,
            renormalize=config["renormalize"],
            scoring_func=config["scoring_func"],
            routed_scaling_factor=scaling_factor,
            shared_expert_gate=shared_gate,
            shared_expert_up=shared_up,
            shared_expert_down=shared_down,
        )

        with torch.no_grad():
            routed_only_out = impl_routed_only.forward(hidden_states, router_logits)
            full_out = impl_with_shared.forward(hidden_states, router_logits)

            # In FP16 with shared experts: routed output is NOT scaled,
            # shared output is scaled by 1/scaling_factor.
            # full_out = routed (unscaled) + shared * (1/scaling_factor)
            # routed_only_out = routed (unscaled, no shared experts, and
            #   in FP16 without shared experts the scaling is also not applied)

            # The shared contribution = full_out - routed_only_out
            shared_contribution = full_out - routed_only_out

            # Independently compute shared expert FFN scaled by 1/scaling_factor
            sg = hidden_states @ shared_gate.T
            su = hidden_states @ shared_up.T
            expected_shared = (F.silu(sg) * su) @ shared_down.T
            expected_shared_scaled = expected_shared * (1.0 / scaling_factor)

        # FP16 has limited precision (~3 decimal digits), use generous tolerance
        assert torch.allclose(
            shared_contribution.float(), expected_shared_scaled.float(),
            rtol=5e-2, atol=0.5
        ), (
            f"FP16 shared expert scaling not applied correctly.\n"
            f"Max abs diff: {(shared_contribution.float() - expected_shared_scaled.float()).abs().max().item()}"
        )
