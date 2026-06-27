"""
LoRA Expert Modules

Implements:
- LoraInjectedLinear: single LoRA adapter (fixed kaiming init)
- LoraExpert: MLP expert with LoRA on gate/up/down projections
- AttentionLoRA: standard LoRA for Q/K/V/O attention projections (unchanged)
- DispatchMoERouter: REAL sparse token-dispatch routing (replaces
  EfficientMoERouter's "compute all experts then mask" approach)
"""

import math
import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

from configuration_lora_moe import LoraMoeConfig


# ── Core LoRA Linear ──────────────────────────────────────────────────────────

class LoraInjectedLinear(nn.Module):
    """
    Low-Rank Adaptation linear layer.

    Computes: output = scale * B(A(dropout(x)))

    A: down-projection  [in_features → rank]
    B: up-projection    [rank → out_features]

    Init: A ~ kaiming_uniform, B = zeros  (standard LoRA init)
    At init the adapter output is exactly zero → training starts from pretrained baseline.

    Weights created directly in bf16 to avoid runtime dtype-cast overhead/bugs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        scale: float = 1.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.r = r
        self.scale = scale

        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.A = self.A.to(torch.bfloat16)
        self.B = self.B.to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast LoRA weights to match input dtype (bf16 in quantized model)
        if self.A.weight.dtype != x.dtype:
            self.A = self.A.to(x.dtype)
            self.B = self.B.to(x.dtype)
        return self.B(self.A(self.dropout(x))) * self.scale


# ── MLP Expert ────────────────────────────────────────────────────────────────

class LoraExpert(nn.Module):
    """
    LoRA-adapted expert for MoE FFN layers.

    Wraps the frozen base MLP with trainable LoRA adapters on gate/up/down.

    Forward:
        gate = gate_proj(x) + gate_lora(x)
        up   = up_proj(x)   + up_lora(x)
        act  = activation(gate) * up
        out  = down_proj(act)   + down_lora(act)
    """

    def __init__(self, config: LoraMoeConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        rank = config.experts_rank
        scale = config.experts_scale

        self.gate_lora = LoraInjectedLinear(config.hidden_size, config.intermediate_size, rank, scale)
        self.up_lora   = LoraInjectedLinear(config.hidden_size, config.intermediate_size, rank, scale)
        self.down_lora = LoraInjectedLinear(config.intermediate_size, config.hidden_size, rank, scale)

        self.activation_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor, mlp: Qwen2MLP) -> torch.Tensor:
        gate = mlp.gate_proj(hidden_states) + self.gate_lora(hidden_states)
        up   = mlp.up_proj(hidden_states)   + self.up_lora(hidden_states)
        act  = self.activation_fn(gate) * up
        down = mlp.down_proj(act) + self.down_lora(act)
        return down


# ── Attention LoRA (UNCHANGED from previous version) ──────────────────────────

class AttentionLoRA(nn.Module):
    """
    Standard LoRA adapters for attention Q/K/V/O projections.

    NOT MoE — one adapter per attention layer, shared across all tokens.
    Uses higher rank than MLP experts (attention matters more for coding tasks).

    Mentor's insight: attention controls WHAT the model focuses on (variable
    tracking, bracket matching, function call patterns). MLP stores knowledge.
    For coding, better attention patterns matter more than more knowledge.
    """

    def __init__(self, config: LoraMoeConfig):
        super().__init__()
        hidden_size = config.hidden_size
        rank = config.attention_rank
        scale = config.experts_scale

        # Q and O use full hidden_size
        self.q_lora = LoraInjectedLinear(hidden_size, hidden_size, rank, scale)
        self.o_lora = LoraInjectedLinear(hidden_size, hidden_size, rank, scale)

        # K and V use GQA size: num_key_value_heads * head_dim
        kv_size = config.num_key_value_heads * (hidden_size // config.num_attention_heads)
        self.k_lora = LoraInjectedLinear(hidden_size, kv_size, rank, scale)
        self.v_lora = LoraInjectedLinear(hidden_size, kv_size, rank, scale)

    def forward_q(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_lora(x)

    def forward_k(self, x: torch.Tensor) -> torch.Tensor:
        return self.k_lora(x)

    def forward_v(self, x: torch.Tensor) -> torch.Tensor:
        return self.v_lora(x)

    def forward_o(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_lora(x)


# ── Real Sparse Token-Dispatch MoE Router (THE ONLY ACTUAL CHANGE) ───────────

class DispatchMoERouter(nn.Module):
    """
    Real sparse expert dispatch — replaces the old "compute all 8 experts for
    every token then mask" approach (EfficientMoERouter) with actual token
    grouping by expert.

    WHY THE OLD APPROACH WAS WASTEFUL:
    EfficientMoERouter computed ALL 8 experts for EVERY token, then zeroed
    out the 6 unused ones per token via a weight matrix. With top-2 routing
    out of 8 experts, 6/8 = 75% of the MLP compute was thrown away every
    single forward pass — computed, then multiplied by zero.

    WHAT THIS DOES INSTEAD:
    1. Route each token to its top-k experts (identical routing logic/math)
    2. Group tokens by which expert they were assigned to
    3. Run each expert ONLY on the tokens actually assigned to it
    4. Scatter-add results back to original token positions

    This is the core idea behind real Megablocks/sparse MoE dispatch, but
    implemented with plain PyTorch indexing/gather/scatter — no custom CUDA
    kernels, no megablocks library. That's why it's "real dispatch" (it
    actually skips unneeded compute) but not as fast as a fused sparse
    kernel would be. It's an honest, from-scratch implementation, not a
    claim of using the actual Megablocks system.

    CORRECTNESS NOTE: with top_k=2, each token contributes to TWO experts'
    token groups (once per assigned expert). Each expert's output for that
    token is scaled by that expert's routing weight, then summed back via
    index_add_. The final output per token is the weighted sum over exactly
    its top-k experts — mathematically identical to the dense+mask version,
    only the computation path differs (skip unneeded matmuls vs compute
    then discard them).
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_dim = hidden_dim

        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.noise_gate = nn.Linear(hidden_dim, num_experts, bias=False)

        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        nn.init.zeros_(self.noise_gate.weight)
        self.gate = self.gate.to(torch.bfloat16)
        self.noise_gate = self.noise_gate.to(torch.bfloat16)

    def forward(
        self,
        hidden_states: torch.Tensor,
        experts: nn.ModuleList,
        mlp,
    ):
        """
        Args:
            hidden_states: [num_tokens, hidden_dim]  (already flattened)
            experts: list of LoraExpert modules
            mlp: frozen base MLP shared by all experts

        Returns:
            output: [num_tokens, hidden_dim]
            router_logits: [num_tokens, num_experts]  for aux loss
        """
        # cast router weights to match input dtype (bf16 in quantized model)
        if self.gate.weight.dtype != hidden_states.dtype:
            self.gate = self.gate.to(hidden_states.dtype)
            self.noise_gate = self.noise_gate.to(hidden_states.dtype)

        num_tokens = hidden_states.shape[0]

        # ── routing decision (identical math to before) ─────────────────────
        logits = self.gate(hidden_states)

        if self.training:
            noise = torch.randn_like(logits) * F.softplus(self.noise_gate(hidden_states))
            logits = logits + noise

        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)
        # top_k_indices, top_k_weights: [num_tokens, top_k]

        # ── real sparse dispatch (THE ACTUAL FIX) ───────────────────────────
        output = torch.zeros_like(hidden_states)

        # flatten (token, slot) pairs — each token appears top_k times, once
        # per selected expert
        flat_expert_ids = top_k_indices.reshape(-1)            # [num_tokens * top_k]
        flat_weights    = top_k_weights.reshape(-1)            # [num_tokens * top_k]
        flat_token_ids  = torch.arange(num_tokens, device=hidden_states.device) \
            .unsqueeze(1).expand(-1, self.top_k).reshape(-1)    # [num_tokens * top_k]

        for expert_idx in range(self.num_experts):
            mask = flat_expert_ids == expert_idx
            if not mask.any():
                continue  # no tokens routed here this batch — skip entirely, real compute saved

            token_ids_for_expert = flat_token_ids[mask]
            weights_for_expert   = flat_weights[mask].unsqueeze(-1)

            # gather ONLY the tokens this expert actually needs (the compute saving)
            expert_input = hidden_states[token_ids_for_expert]

            expert_output = experts[expert_idx](expert_input, mlp)
            expert_output = expert_output * weights_for_expert

            # scatter-add back into the output at original token positions
            output.index_add_(0, token_ids_for_expert, expert_output)

        return output, logits
