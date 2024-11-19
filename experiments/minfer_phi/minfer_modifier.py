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
from transformers.models.llama.modeling_llama import *
from transformers.cache_utils import *
from transformers.cache_utils import Cache


from minference import MInference
from minference.minference_configuration import MInferenceConfig
from minference.modules.minference_forward import (
    get_cos_sin, search_pattern, gather_qkv,
    LAST_Q_MASK, sum_all_diagonal_matrix
)

from minference.patch import (
    hf_437_prepare_inputs_for_generation, _prepare_decoder_attention_mask_inference, 
    # forward_llama_model, 
    forward_llama_for_causal_lm
)

from minfer_ops import (
    vs_attn_forward,
    bs_attn_forward, 
    streaming_forward
)
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

def forward_phi_decoder_layer(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    # original procedure: 
    # residual -> input_layernorm -> self_attn -> self_attn + residual 
    # -> residual -> post_attention_layernorm -> mlp -> mlp + residual

    residual = hidden_states.clone()

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )
    if residual.device != hidden_states.device:
        residual = residual.to(hidden_states.device)
    hidden_states = residual + hidden_states

    # Fully Connected
    # # TODO: Here the post_attention_layernorm, mlp and residual addition are done in 32000-token chunks
    # for start_idx in range(0, seq_len, 32000):
    #     end_idx = min(seq_len, start_idx + 32000)
    #     part_hidden_states = hidden_states[:, start_idx:end_idx, :].clone() # residual clone 
    #     part_hidden_states = self.post_attention_layernorm(part_hidden_states)
    #     part_hidden_states = self.mlp(part_hidden_states)
    #     hidden_states[:, start_idx:end_idx, :] += part_hidden_states # residual addition
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states


    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs



def forward_phi_model(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_usable_length(seq_length)
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    # embed positions
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    # batch, seq_len, embed_dim = hidden_states.shape
    # for start_idx in range(0, seq_len, 32000):
    #     end_idx = min(seq_len, start_idx + 32000)
    #     hidden_states[:, start_idx:end_idx, :] = self.norm(
    #         hidden_states[:, start_idx:end_idx, :]
    #     )
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    next_cache = None
    if use_cache:
        next_cache = (
            next_decoder_cache.to_legacy_cache()
            if use_legacy_cache
            else next_decoder_cache
        )
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


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

        with torch.autograd.set_detect_anomaly(True):
            if q_len != 1:
                # output = torch.empty_like(query_states)
                output_list = []
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
                        attn_output_head = self.minfer_attn_forward(q, k, v, head) # [bsz, 1, q_len, self.head_dim]
                    elif is_flash_attn_2_available(): 
                        # if search is enabled or the current layer is before the starting layer, simply use flash attention 
                        # Note that the input to the flash attention function should be in the shape (bsz, q_len, head_dim, num_heads)
                        attn_output_head = nnscaler_flash_attention_forward(
                            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1,2),
                            attention_mask, q_len, 
                            dropout=0.0, softmax_scale=None, causal=q_len != 1
                        )
                        attn_output_head = attn_output_head.view(bsz, 1, q_len, self.head_dim)
                    else:
                        attn_output_head = gather_qkv(q, k, v, attention_mask)
                    output_list.append(attn_output_head.squeeze(1))
                output = torch.stack(output_list, dim=1)
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
        return vs_attn_forward(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            q_len, vertical_size, slash_size, self.head_dim
        ).view(bsz, 1, q_len, self.head_dim)

    def block_sparse_kernel(q, k, v, vertical_size=None, slash_size=None):
        topk = 100
        # return block_sparse_attention(q, k, v, topk)
        return bs_attn_forward(q, k, v, topk)

    bsz, q_len = q.shape[0], q.shape[2]
    if q_len == 1: return dense(q, k, v)

    ty, vertical_size, slash_size, _ = self.best_pattern.get(head_id, ("vertical_and_slash", 1000, 6096, 1))

    # TODO: DEBUG Setting
    # if ty == 'stream_llm':
    #     ty = 'vertical_and_slash'
    #     vertical_size, slash_size = 1000, 6096

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
            m.forward = forward_phi_decoder_layer.__get__(m, DecoderLayer)
    
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
    model.model.forward = forward_phi_model.__get__(
        model.model, model.model.__class__
    )
    model.forward = forward_llama_for_causal_lm.__get__(model, model.__class__)

    print(f"{__name__} | Patched model {model.__class__} for minference..")
    return model



