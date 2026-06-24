"""
LoraMoe Model

Key changes from original:
1. LoraMoeBlock now uses DispatchMoERouter (REAL sparse token dispatch —
   only computes the experts each token is actually assigned to, instead of
   computing all 8 experts for every token and masking 6 away)
2. LoraMoeDecoderLayer now injects AttentionLoRA into Q/K/V/O projections
3. Flash attention supported via config._attn_implementation
4. Attention LoRA is wired into the forward pass by monkey-patching
   the attention module's projection calls
"""

import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
# masking handled by base model _update_causal_mask
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_flash_attn_2_available, logging

from configuration_lora_moe import LoraMoeConfig
from peft_experts import LoraExpert, AttentionLoRA, DispatchMoERouter

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2RMSNorm,
    Qwen2Attention,
    Qwen2RotaryEmbedding,
)
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

logger = logging.get_logger(__name__)


# ── MoE Block ─────────────────────────────────────────────────────────────────

class LoraMoeBlock(nn.Module):
    """
    MoE block using DispatchMoERouter (real sparse token dispatch).

    Replaces the old "compute all experts, mask away unused ones" approach
    with actual token grouping — each expert only computes on tokens routed
    to it. See DispatchMoERouter docstring for full explanation.
    """

    def __init__(self, config: LoraMoeConfig):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # CHANGED: real sparse dispatch instead of dense-compute-then-mask
        self.router = DispatchMoERouter(self.hidden_dim, self.num_experts, self.top_k)
        self.lora_experts = nn.ModuleList([LoraExpert(config) for _ in range(self.num_experts)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        mlp: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # flatten tokens for routing
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # parallel expert computation
        output_flat, router_logits = self.router(hidden_flat, self.lora_experts, mlp)

        output = output_flat.view(batch_size, seq_len, hidden_dim)
        return output, router_logits


# ── Attention LoRA Injection ──────────────────────────────────────────────────

class AttentionWithLoRA(nn.Module):
    """
    Wraps Qwen2Attention and injects LoRA into Q/K/V/O projections.

    We do this by wrapping the attention module and intercepting the
    projection outputs before the attention computation.

    The frozen attention weights still run, then we add the LoRA delta:
        q = attn.q_proj(x) + attn_lora.forward_q(x)
        k = attn.k_proj(x) + attn_lora.forward_k(x)
        v = attn.v_proj(x) + attn_lora.forward_v(x)
    
    Then after attention:
        o = attn.o_proj(hidden) + attn_lora.forward_o(hidden)
    """

    def __init__(self, base_attn: Qwen2Attention, lora: AttentionLoRA):
        super().__init__()
        self.base_attn = base_attn
        self.lora = lora

        # patch the projection methods
        self._patch_projections()

    def _patch_projections(self):
        """Monkey-patch q/k/v/o proj to include LoRA delta."""
        lora = self.lora
        base = self.base_attn

        original_q = base.q_proj
        original_k = base.k_proj
        original_v = base.v_proj
        original_o = base.o_proj

        class LoraLinear(nn.Module):
            def __init__(self, base_linear, lora_forward):
                super().__init__()
                self.base = base_linear
                self.lora_forward = lora_forward

            def forward(self, x):
                return self.base(x) + self.lora_forward(x)

        base.q_proj = LoraLinear(original_q, lora.forward_q)
        base.k_proj = LoraLinear(original_k, lora.forward_k)
        base.v_proj = LoraLinear(original_v, lora.forward_v)
        base.o_proj = LoraLinear(original_o, lora.forward_o)

    def forward(self, *args, **kwargs):
        return self.base_attn(*args, **kwargs)


# ── Decoder Layer ─────────────────────────────────────────────────────────────

class LoraMoeDecoderLayer(nn.Module):
    """
    Transformer decoder layer with:
    - MoE FFN (LoraMoeBlock with efficient parallel routing)
    - Attention LoRA on Q/K/V/O (AttentionLoRA)
    - Flash attention support
    """

    def __init__(self, layer: nn.Module, config: LoraMoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        if config._attn_implementation is None:
            config._attn_implementation = "eager"

        # use existing pretrained attention directly
        base_attn = layer.self_attn

        # wrap with attention LoRA if enabled
        if config.use_attention_lora:
            attn_lora = AttentionLoRA(config)
            self.self_attn = AttentionWithLoRA(base_attn, attn_lora)
            self._has_attn_lora = True
        else:
            self.self_attn = base_attn
            self._has_attn_lora = False

        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # frozen base MLP — shared across all experts
        self.mlp = layer.mlp

        # MoE block with efficient parallel router
        self.lora_moe_block = LoraMoeBlock(config)

        # layer norms — copy from original layer
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if hasattr(layer, 'input_layernorm'):
            self.input_layernorm = layer.input_layernorm
        if hasattr(layer, 'post_attention_layernorm'):
            self.post_attention_layernorm = layer.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:

        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated. Use `attention_mask`.", FutureWarning)

        # ── attention block ───────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # get the actual attention module for forward call
        attn_module = self.self_attn.base_attn if self._has_attn_lora else self.self_attn

        attn_out = attn_module(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = attn_out[0]
        self_attn_weights = attn_out[1] if output_attentions else None
        present_key_value = attn_out[2] if use_cache else None

        hidden_states = residual + hidden_states

        # ── MoE FFN block ─────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.lora_moe_block(hidden_states, self.mlp)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


# ── Model Forward ─────────────────────────────────────────────────────────────

def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, MoeModelOutputWithPast]:

    output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
    output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    use_cache            = use_cache            if use_cache            is not None else self.config.use_cache
    return_dict          = return_dict          if return_dict          is not None else self.config.return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Cannot specify both input_ids and inputs_embeds")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("Must specify input_ids or inputs_embeds")

    past_key_values_length = 0

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once("`use_cache=True` incompatible with gradient checkpointing. Setting False.")
            use_cache = False

    if use_cache:
        if not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache()
            past_key_values_length = past_key_values.get_seq_length()

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # use qwen2's built-in causal mask update
    past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    cache_position = torch.arange(
        past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    )
    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, False
    )
    attention_mask = causal_mask

    hidden_states = inputs_embeds

    all_hidden_states  = () if output_hidden_states  else None
    all_self_attns     = () if output_attentions      else None
    all_router_logits  = () if output_router_logits   else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                output_router_logits,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_router_logits:
            all_router_logits += (layer_outputs[-1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None

    if not return_dict:
        return tuple(
            v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
            if v is not None
        )

    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
    )


def causal_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, MoeCausalLMOutputWithPast]:

    output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
    output_router_logits = output_router_logits if output_router_logits is not None else self.config.output_router_logits
    output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    return_dict          = return_dict          if return_dict          is not None else self.config.return_dict

    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        return_dict=return_dict,
    )

    hidden_states = outputs[0].to(self.lm_head.weight.dtype)
    logits = self.lm_head(hidden_states).float()
    logits = torch.clamp(logits, min=-1e4, max=1e4)
    
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100)

        vocab_size_model = self.lm_head.weight.shape[0]
        print(f"=== DEBUG: vocab_size_model = {vocab_size_model}")
        print(f"logits shape: {logits.shape}")   # should be [batch, seq_len, vocab_size]

        # --- Clamp (bulletproof) ---
        labels_dev = labels.to(logits.device)
        invalid_neg = (labels_dev < 0) & (labels_dev != -100)
        invalid_pos = (labels_dev >= vocab_size_model)
        invalid_mask = invalid_neg | invalid_pos
        if invalid_mask.any():
            print(f"Found {invalid_mask.sum().item()} invalid labels (neg:{invalid_neg.sum().item()}, pos:{invalid_pos.sum().item()}) — replacing with -100")
            labels_dev = torch.where(
                invalid_mask,
                torch.tensor(-100, dtype=labels_dev.dtype, device=labels_dev.device),
                labels_dev
            )
        shifted_labels = labels_dev[..., 1:]
        print(f"shifted_labels shape: {shifted_labels.shape}")
        print(f"shifted_labels min: {shifted_labels.min().item()}, max: {shifted_labels.max().item()}")
        # --- END CLAMP ---

        loss = loss_fct(
            logits[..., :-1, :].transpose(1, 2),
            shifted_labels,
        )

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits if return_dict else outputs[-1],
            self.config.num_local_experts,
            self.config.num_experts_per_tok,
        )
        if labels is not None and aux_loss is not None:
            aux_loss = aux_loss.float()
            loss = loss + self.config.router_aux_loss_coef * aux_loss

    if not return_dict:
        output = (logits,) + outputs[1:]
        if output_router_logits:
            output = (aux_loss,) + output
        return (loss,) + output if loss is not None else output

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )

# ── LoraMoeModel Wrapper ──────────────────────────────────────────────────────

class LoraMoeModel(torch.nn.Module):
    """
    MoE-LoRA model wrapper.

    Replaces all decoder layers with LoraMoeDecoderLayer which adds:
    - MoE FFN with efficient parallel routing
    - Attention LoRA on Q/K/V/O projections

    make_experts_trainable() freezes base weights and enables gradients for:
    - All lora_moe_block parameters (MLP LoRA experts + router)
    - All attention LoRA parameters (Q/K/V/O adapters)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        config: LoraMoeConfig,
        layer_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.base_model = model
        self.config = config

        if layer_ids is None:
            layer_ids = list(range(len(self.base_model.model.layers)))

        self.layer_ids = layer_ids

        for layer_id in layer_ids:
            layer = self.base_model.model.layers[layer_id]
            if not isinstance(layer, LoraMoeDecoderLayer):
                self.base_model.model.layers[layer_id] = LoraMoeDecoderLayer(
                    layer, config, layer_id
                ).to(model.device)
            else:
                warnings.warn("Trying to rewrap a wrapped layer! Call .unwrap() first.")

        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # bind custom forward functions
        bound_forward = model_forward.__get__(self.base_model.model, self.base_model.model.__class__)
        setattr(self.base_model.model, 'forward', bound_forward)

        bound_causal_forward = causal_model_forward.__get__(self.base_model, self.base_model.__class__)
        setattr(self.base_model, 'forward', bound_causal_forward)

        self.base_model.config = config
        self.base_model.model.config = config

        # untie lm_head from embed_tokens — safetensors refuses shared tensors on save
        if self.base_model.lm_head.weight.data_ptr() == self.base_model.model.embed_tokens.weight.data_ptr():
            self.base_model.lm_head.weight = nn.Parameter(self.base_model.lm_head.weight.detach().clone())
            self.base_model.config.tie_word_embeddings = False

    @property
    def device(self) -> torch.device:
        return self.base_model.device

    def forward(self, **kwargs):
        return self.base_model(**kwargs)

    def generate(self, **kwargs):
        return self.base_model.generate(**kwargs)

    def __call__(self, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        return self.base_model(**kwargs)

    def make_experts_trainable(self):
        """
        Freeze all base model weights.
        Enable gradients only for:
        - lora_moe_block (MLP experts + routers)
        - attn_lora inside AttentionWithLoRA (Q/K/V/O LoRA)
        """
        # freeze everything first
        for param in self.parameters():
            param.requires_grad = False

        # unfreeze MLP MoE experts and routers
        for name, param in self.named_parameters():
            if 'lora_moe_block' in name:
                param.requires_grad = True

        # unfreeze attention LoRA adapters
        for name, param in self.named_parameters():
            if 'lora' in name and 'self_attn' in name:
                param.requires_grad = True

        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters:     {total/1e6:.1f}M")
        print(f"Trainable parameters: {trainable/1e6:.1f}M ({100*trainable/total:.2f}%)")

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self):
        self.base_model.gradient_checkpointing_disable()

    def save_pretrained(self, *args, **kwargs):
        import torch.nn as nn
        lm_head = self.base_model.lm_head
        embed = self.base_model.model.embed_tokens
        if lm_head.weight.data_ptr() == embed.weight.data_ptr():
            lm_head.weight = nn.Parameter(lm_head.weight.detach().clone())
        self.base_model.save_pretrained(*args, **kwargs)

    def unwrap(self) -> PreTrainedModel:
        for layer_id in self.layer_ids:
            layer = self.base_model.model.layers[layer_id]
            self.base_model.model.layers[layer_id] = layer.mlp
        return self.base_model