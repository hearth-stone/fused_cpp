# -*- coding: utf-8 -*-
"""FusedMoEImpl: Pure PyTorch Mixture of Experts with full-token EP optimization."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F


class FusedMoEImpl:
    """Pure PyTorch MoE implementation with full-token Expert Parallelism.

    Each EP node holds the full token batch and computes only its local
    expert subset, eliminating token dispatch communication.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        ffn_hidden_size: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        ep_size: int = 1,
        ep_rank: int = 0,
        renormalize: bool = False,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        reduce_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        shared_expert_gate: Optional[torch.Tensor] = None,
        shared_expert_up: Optional[torch.Tensor] = None,
        shared_expert_down: Optional[torch.Tensor] = None,
    ) -> None:
        # Validate ep_size divides num_experts
        if num_experts % ep_size != 0:
            raise ValueError(f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})")
        # Validate scoring_func
        if scoring_func not in ("softmax", "sigmoid"):
            raise ValueError(f"scoring_func must be 'softmax' or 'sigmoid', got '{scoring_func}'")
        # Validate ep_rank
        if not (0 <= ep_rank < ep_size):
            raise ValueError(f"ep_rank ({ep_rank}) must be in [0, ep_size={ep_size})")

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.renormalize = renormalize
        self.scoring_func = scoring_func
        self.routed_scaling_factor = routed_scaling_factor
        self.reduce_fn = reduce_fn

        self.local_num_experts = num_experts // ep_size
        self.expert_start = ep_rank * self.local_num_experts
        self.expert_end = self.expert_start + self.local_num_experts

        # Expert weights — fuse gate+up into a single tensor for one matmul
        # w_gate: [E, F, H], w_up: [E, F, H] → w_gate_up: [E, 2F, H]
        self.w_gate_up = torch.cat([w_gate, w_up], dim=1)
        self.w_down = w_down

        # Shared expert weights (optional) — fuse gate+up
        if shared_expert_gate is not None and shared_expert_up is not None:
            self.shared_gate_up = torch.cat([shared_expert_gate, shared_expert_up], dim=0)  # [2F, H]
        else:
            self.shared_gate_up = None
        self.shared_expert_down = shared_expert_down

    # ── Backward-compatible accessors ────────────────────────────────────────

    @property
    def w_gate(self) -> torch.Tensor:
        """Gate weights: first half of fused w_gate_up."""
        return self.w_gate_up[:, : self.ffn_hidden_size, :]

    @property
    def w_up(self) -> torch.Tensor:
        """Up weights: second half of fused w_gate_up."""
        return self.w_gate_up[:, self.ffn_hidden_size :, :]

    @property
    def shared_expert_gate(self) -> torch.Tensor | None:
        """Shared gate weights: first half of fused shared_gate_up."""
        if self.shared_gate_up is None:
            return None
        return self.shared_gate_up[: self.shared_gate_up.shape[0] // 2, :]

    @property
    def shared_expert_up(self) -> torch.Tensor | None:
        """Shared up weights: second half of fused shared_gate_up."""
        if self.shared_gate_up is None:
            return None
        return self.shared_gate_up[self.shared_gate_up.shape[0] // 2 :, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with full-token EP optimization.

        Args:
            hidden_states: [total_tokens, hidden_size] input from MLA.
            router_logits: [total_tokens, num_experts] full router logits.

        Returns:
            Output tensor of shape [total_tokens, hidden_size].
        """
        # Step 1: Slice router_logits to local expert columns
        local_logits = router_logits[:, self.expert_start : self.expert_end]

        # Step 2: Compute scores in float32 for numerical stability
        if self.scoring_func == "softmax":
            scores = F.softmax(local_logits.float(), dim=-1).to(local_logits.dtype)
        elif self.scoring_func == "sigmoid":
            scores = torch.sigmoid(local_logits.float()).to(local_logits.dtype)

        # Step 3: Top-k selection with k = min(top_k, local_num_experts)
        k = min(self.top_k, self.local_num_experts)
        topk_weights, topk_ids = torch.topk(scores, k=k, dim=-1)
        if self.renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Step 4: Per-expert FFN loop with weighted accumulation
        output = torch.zeros_like(hidden_states)
        for e in range(self.local_num_experts):
            mask = (topk_ids == e).any(dim=-1)
            if not mask.any():
                continue
            token_indices = mask.nonzero(as_tuple=True)[0]
            x = hidden_states[token_indices]

            # FFN: gate_up fused matmul → SiLU-and-mul → down
            gate_up = x @ self.w_gate_up[e].T  # [tokens, 2*ffn_hidden]
            gate_out, up_out = gate_up.chunk(2, dim=-1)
            intermediate = F.silu(gate_out) * up_out
            expert_out = intermediate @ self.w_down[e].T

            # Weighted accumulation (handles top_k > 1)
            for ki in range(k):
                ki_mask = topk_ids[token_indices, ki] == e
                if ki_mask.any():
                    ki_indices = token_indices[ki_mask]
                    weights = topk_weights[ki_indices, ki].unsqueeze(-1)
                    output[ki_indices] += expert_out[ki_mask] * weights

        # Step 5: Shared expert FFN (optional)
        shared_out = None
        if self.shared_gate_up is not None:
            sg_su = hidden_states @ self.shared_gate_up.T  # [T, 2*shared_ffn]
            sg, su = sg_su.chunk(2, dim=-1)
            shared_out = (F.silu(sg) * su) @ self.shared_expert_down.T

        # Step 6: Apply routed_scaling_factor (dtype-aware FP16 overflow protection)
        if hidden_states.dtype != torch.float16:
            output = output * self.routed_scaling_factor
        elif self.shared_gate_up is not None:
            shared_out = shared_out * (1.0 / self.routed_scaling_factor)

        # Step 7: EP all-reduce (routed experts only), then add shared expert
        if self.ep_size > 1 and self.reduce_fn is not None:
            output = self.reduce_fn(output)
        if shared_out is not None:
            output = output + shared_out

        return output
