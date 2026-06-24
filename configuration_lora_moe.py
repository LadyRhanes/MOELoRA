"""
MoE-LoRA Model Configuration

Extends Qwen2Config with MoE + LoRA parameters.
Now includes attention_rank for separate attention LoRA control.
"""

from transformers.models.qwen2.modeling_qwen2 import Qwen2Config


class LoraMoeConfig(Qwen2Config):
    r"""
    Configuration class for MoE-LoRA models.

    Extends Qwen2Config with parameters for:
    - MoE routing (num_local_experts, num_experts_per_tok)
    - MLP LoRA adapters (experts_rank, experts_scale)
    - Attention LoRA adapters (attention_rank) — NEW
    - Load balancing (router_aux_loss_coef)

    Args:
        experts_rank: Rank of LoRA matrices for MLP experts. Default 8.
        attention_rank: Rank of LoRA matrices for attention Q/K/V/O. Default 32.
            Higher than experts_rank because attention matters more for coding tasks.
        experts_scale: Scaling factor for LoRA outputs. Default 1.0.
        num_experts_per_tok: Top-k experts activated per token. Default 2.
        num_local_experts: Total number of experts per MoE layer. Default 8.
        output_router_logits: Return router logits for aux loss. Default False.
        router_aux_loss_coef: Load balancing loss weight. Default 0.001.
        use_attention_lora: Whether to apply LoRA to attention layers. Default True.
    """

    def __init__(
        self,
        experts_rank: int = 8,
        attention_rank: int = 32,
        experts_scale: float = 1.0,
        num_experts_per_tok: int = 2,
        num_local_experts: int = 8,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        use_attention_lora: bool = True,
        **kwargs,
    ):
        # MLP LoRA params
        self.experts_rank = experts_rank
        self.experts_scale = experts_scale

        # Attention LoRA params — new
        self.attention_rank = attention_rank
        self.use_attention_lora = use_attention_lora

        # MoE routing params
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef

        super().__init__(**kwargs)
