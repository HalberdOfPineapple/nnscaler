#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# This file modifies the official modeling_llama.py file at runtime to
# 1. register the flash attention function to nnscaler and update related code
# 2. replace the un-fused RMSNorm with apex's fused version
import os
import math
import json
import types
import torch
import logging
import warnings
import numpy as np
import concurrent.futures
import torch.distributed
import torch.nn.functional as F
logger = logging.getLogger(__name__)

from torch import nn
from typing import List, Optional, Tuple, Union, Any, Dict

from nnscaler.graph.parser.register import register_op
from nnscaler.ir import IRTensor

from transformers.utils import (
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)
from transformers.cache_utils import Cache


from minference import MInference
from minference.patch import forward_llama_decoder_layer
from minference.minference_configuration import MInferenceConfig
from minference.modules.minference_forward import (
    # init_minference_parameters, 
    get_cos_sin, search_pattern, gather_qkv,
    LAST_Q_MASK, sum_all_diagonal_matrix
)

from minference.patch import (
    hf_437_prepare_inputs_for_generation, _prepare_decoder_attention_mask_inference, 
    forward_llama_model, forward_llama_for_causal_lm
)

from minfer_ops import vs_attn_forward, bs_attn_forward, streaming_forward
from phi3 import Phi3ForCausalLM, Phi3Attention as PhiAttention, apply_rotary_pos_emb, repeat_kv, _get_unpad_data

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine # type: ignore
    has_apex = True
except ImportError:
    has_apex = False


def rmsnorm_fwd(self, hidden_states):
    if has_apex:
        return fused_rms_norm_affine(hidden_states, self.weight, self.weight.shape, self.variance_epsilon)
    else:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


_flash_supports_window_size = False
class NNScalerPhiFlashAttention2(PhiAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Phi3FlashAttention2 attention does not support output_attentions
        output_attentions = False

        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_dropout = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32.

        if query_states.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.qkv_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and q_len != 1

        attn_output = nnscaler_flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=attn_dropout,
            causal=causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def nnscaler_flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        causal=True,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API (bsz, q_len, self.num_heads, self.head_dim)
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = nnscaler_upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return attn_output


def nnscaler_upad_input(query_layer, key_layer, value_layer, attention_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    _, _, num_heads, _ = query_layer.shape
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def phi_flash_attention_anno(query_states, key_states, value_states, attention_mask, *args, **kwargs) -> str:
    if query_states.shape[2] != key_states.shape[2]:
        assert query_states.shape[2] % key_states.shape[2] == 0
        group_size = query_states.shape[2] // key_states.shape[2]
        assert query_states.shape[2] == value_states.shape[2] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    if isinstance(attention_mask, IRTensor):
        return f'b l^ {q_anno} hd^, b s^ {kv_anno} hd^, b s^ {kv_anno} vd^, b l^ -> b l^ {q_anno} vd^'
    else:
        return f'b l^ {q_anno} hd^, b s^ {kv_anno} hd^, b s^ {kv_anno} vd^ -> b l^ {q_anno} vd^'


register_op(phi_flash_attention_anno)(nnscaler_flash_attention_forward)

# ROPE_TYPE = None
# def set_rope_type(self):
#     global ROPE_TYPE
#     if ROPE_TYPE is not None:
#         return
#     if "seq_len" in inspect.signature(self.rotary_emb.forward).parameters:
#         if "position_ids" in inspect.signature(self.rotary_emb.forward).parameters:
#             ROPE_TYPE = "seq_len,position_ids"
#         else:
#             ROPE_TYPE = "seq_len"
#     elif "max_seq_len" in inspect.signature(self.rotary_emb.forward).parameters:
#         ROPE_TYPE = "max_seq_len"
#     else:
#         ROPE_TYPE = "position_ids"
# def get_rope_type():
#     from minference.modules.minference_forward import ROPE_TYPE
#     return ROPE_TYPE

def init_minference_parameters(self):
    config = self.config.to_dict()
    self.starting_layer = config.get("starting_layer", 0)
    self.is_search = config.get("is_search", False)

    self.ne_inf = None
    self.config_path = config.get("config_path", "")
    if (
        self.config_path is not None and
        os.path.exists(self.config_path) and
        self.layer_idx < len(json.load(open(self.config_path)))
    ):
        self.best_pattern = {int(ii): jj for ii, jj in json.load(open(self.config_path))[self.layer_idx].items()}
    else:
        self.best_pattern = {}
    self.vertical, self.slash = None, None

    # import apply_rotary_pos_emb
    self.apply_rotary_pos_emb = True



def minference_forward():
    def forward(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        past_key_value,
        output_attentions,
        use_cache,
        **kwargs,
    ):
        self.init_minference_parameters()
        self.ne_inf = torch.finfo(hidden_states.dtype).min

        bsz, q_len, _ = hidden_states.size()

        if "q_proj" in self.__dict__["_modules"]:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)
            query_pos = self.num_heads * self.head_dim
            key_value_pos = query_pos // self.num_key_value_groups
            query_states, key_states, value_states = torch.split(qkv, [query_pos, key_value_pos, key_value_pos], -1)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)


        # Currently disable the dynamic adjustment of Rope and fix the specific setting for Phi:
        # --------------------------------------------------------------------------------------
        # set_rope_type(self)
        # cos, sin = get_cos_sin(self, value_states, kv_seq_len, position_ids)
        # if get_rope_type() == "max_seq_len":
        #     if cos.device != query_states.device:
        #         cos = cos.to(query_states.device)
        #     query_states = apply_rotary_pos_emb(query_states, cos)
        #     key_states = apply_rotary_pos_emb(key_states, cos)
        # else:
        #     if position_ids is not None and position_ids.device != cos.device:
        #         position_ids = position_ids.to(cos.device)
        #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # ------------------------------
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Initialize the pattern config when search mode is enabled
        if self.is_search:
            if os.path.exists(self.config_path):
                config_list = json.load(open(self.config_path))
                if self.config.num_hidden_layers == len(config_list):
                    assert False, f"Search completed. The config is located in {self.config_path}."
            else:
                config_list = []
            config = {}
            print("Layer", self.layer_idx)


        if q_len != 1:
            output = torch.empty_like(query_states)
            for head in range(query_states.size(1)):
                q = query_states[:, head, :, :].unsqueeze(1) # (bsz, 1, q_len, head_dim)
                k = key_states[:, head, :, :].unsqueeze(1)
                v = value_states[:, head, :, :].unsqueeze(1)

                # If search mode is enabled and the current layer has not already been configured 
                # => search for the best pattern
                if self.is_search and self.layer_idx >= len(config_list):
                    with torch.no_grad():
                        config[head] = search_pattern(q, k, head)

                # if search is disabled and the current layer is beyond  starting layer 
                # => apply the kernel for calculating the attention based on the best pattern
                if self.layer_idx >= self.starting_layer and not self.is_search:
                    attn_output = self.minfer_attn_forward(q, k, v, head)
                elif is_flash_attn_2_available(): 
                    # if search is enabled or the current layer is before the starting layer, simply use flash attention 
                    # Note that the input to the flash attention function should be in the shape (bsz, q_len, head_dim, num_heads)
                    attn_output = nnscaler_flash_attention_forward(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2),
                        attention_mask, q_len, 
                        dropout=0.0, softmax_scale=None, causal=q_len != 1
                    )
                    attn_output = attn_output.view(bsz, 1, q_len, self.head_dim)
                else:
                    attn_output = gather_qkv(q, k, v, attention_mask)

                output[:, head:head + 1] = attn_output
            if self.is_search:
                if len(config):
                    config_list.append(config)
                with open(self.config_path, 'w') as json_file:
                    json.dump(config_list, json_file)
        else:
            output =  flash_attn_func(
                query_states.transpose(1, 2), 
                key_states.transpose(1, 2), 
                value_states.transpose(1,2), 
                0.0, softmax_scale=None, causal=q_len != 1
            ).view(bsz, query_states.size(1), q_len, self.head_dim)
        attn_output = output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward

def last_q_mask_to(last_q_mask: torch.Tensor, device) -> torch.Tensor:
    return last_q_mask.to(device)
register_op('b^ h^ l^ l^ -> b^ h^ l^ l^')(last_q_mask_to)

def qk_einsum(q, k, head_dim):
    return torch.einsum(f'bhmk, bhnk -> bhmn', q, k) / math.sqrt(head_dim)
register_op('b^ h^ last_l^ d^, b^ h^ l^ d^ -> b^ h^ last_l^ l^')(qk_einsum)

def minfer_attn_forward(self, q, k, v, head_id):
    """
        Inputs:
            self: Attention, the current attention layer
            q: torch.Tensor, shape (bsz, 1, q_len, head_dim)
            k: torch.Tensor, shape (bsz, 1, q_len, head_dim)
            v: torch.Tensor, shape (bsz, 1, q_len, head_dim)
            head_id: int, head index
    """
    def dense(q, k, v, vertical_size=None, slash_size=None):
        # return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2), 0.0, softmax_scale=None, causal=q_len != 1).view(bsz, 1, q_len, self.head_dim)
        attn_output = nnscaler_flash_attention_forward(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2), 
            dropout=0.0, softmax_scale=None, causal=q_len != 1
        )
        return attn_output.view(bsz, 1, q_len, self.head_dim)

    def vertical_and_slash_kernel(q: torch.Tensor, k, v, vertical_size, slash_size):
        vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))

        last_q = min(64, q_len)

        with torch.no_grad():
            # qk = torch.einsum(f'bhmk, bhnk -> bhmn', q[:,:,-last_q:,:], k) / math.sqrt(self.head_dim)
            qk = qk_einsum(q[:,:,-last_q:,:].clone().detach(), k, self.head_dim)

            # LAST_Q_MASK: torch.Size([1, 1, 64, 64])
            # qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:].to(q.device), qk[:, :, :, -last_q:], -torch.inf)
            last_q_mask = LAST_Q_MASK[..., -last_q:, -last_q:]
            qk[:, :, :, -last_q:] = torch.where(last_q_mask, qk[:, :, :, -last_q:], -torch.inf)

            vertical = qk.sum(-2, keepdim=True)
            vertical[..., :30] = torch.inf
            vertical_topk = torch.topk(vertical, vertical_size, -1).indices

            slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
            slash[..., -100:] = torch.inf
            slash = (q_len - 1) - torch.topk(slash, slash_size, -1).indices

        # return vertical_slash_sparse_attention(q, k, v, vertical_topk, slash)
        return vs_attn_forward(q, k, v, vertical_topk, slash)
    
    
    def block_sparse_kernel(q, k, v, vertical_size=None, slash_size=None):
        topk = 100
        # return block_sparse_attention(q, k, v, topk)
        return bs_attn_forward(q, k, v, topk)

    bsz, q_len = q.shape[0], q.shape[2]
    if q_len == 1: return dense(q, k, v)

    ty, vertical_size, slash_size, _ = self.best_pattern.get(head_id, ("vertical_and_slash", 1000, 6096, 1))

    # TODO: DEBUG Setting
    if ty == 'stream_llm':
        ty = 'vertical_and_slash'
        vertical_size, slash_size = 1000, 6096

    fc = {
        "stream_llm": streaming_forward,
        "vertical_and_slash": vertical_and_slash_kernel,
        "block_sparse": block_sparse_kernel,
    }[ty]

    return fc(q, k, v, vertical_size, slash_size)

def minfer_phi_init(model: Phi3ForCausalLM, model_id: str, minfer_config: MInferenceConfig):
    # if minfer_config.kv_cache_cpu:
    #     global KV_CACHE_CPU_DEVICE
    #     KV_CACHE_CPU_DEVICE = minfer_config.kv_cache_cpu_device
    #     model.config.kv_cache_cpu_device = minfer_config.kv_cache_cpu_device
    #     return minference_patch_kv_cache_cpu(model)
    # if minfer_config.use_snapkv:
    #     return minfer_config(model)
    if minfer_config.kv_cache_cpu or minfer_config.use_snapkv:
        raise NotImplementedError("Setup for KV Cache CPU and SnapKV modes are not supported for training")


    # Note that the Attention, Model and DecoderLayer are not hardcoded classes (e.g. LlamaDecoderLayer)
    # Instead they can fit for any model that uses such an architecture
    Attention = model.model.layers[0].self_attn.__class__
    DecoderLayer = model.model.layers[0].__class__

    forward = minference_forward()
    def update_module(m):
        if isinstance(m, Attention):
            m.init_minference_parameters = init_minference_parameters.__get__(
                m, Attention
            )
            m.minfer_attn_forward = (
                minfer_attn_forward.__get__(m, Attention)
            )
            m.forward = forward.__get__(m, Attention)
        if isinstance(m, DecoderLayer):
            m.forward = forward_llama_decoder_layer.__get__(m, DecoderLayer)
    
    model.apply(update_module)
    model.prepare_inputs_for_generation = hf_437_prepare_inputs_for_generation.__get__(
        model, model.__class__
    )
    model.model._use_sdpa = False

    model.model._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask_inference.__get__(
            model.model, model.model.__class__
        )
    )
    model.model.forward = forward_llama_model.__get__(
        model.model, model.model.__class__
    )
    model.forward = forward_llama_for_causal_lm.__get__(model, model.__class__)

    print(f"{__name__} | Patched model {model.__class__} for minference..")
    return model



