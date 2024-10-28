#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# This file modifies the official modeling_llama.py file at runtime to
# 1. register the flash attention function to nnscaler and update related code
# 2. replace the un-fused RMSNorm with apex's fused version
import os
import math
import types
import torch
import logging
import numpy as np
import concurrent.futures
import torch.distributed
import torch.nn.functional as F
logger = logging.getLogger(__name__)

from torch import nn
from typing import List, Optional, Tuple, Union

from nnscaler.graph.parser.register import register_op
from nnscaler.ir import IRTensor

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import LlamaAttention, LLAMA_ATTENTION_CLASSES, apply_rotary_pos_emb, LlamaRMSNorm, repeat_kv, _get_unpad_data
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from custom_trainer import get_iter_cnt, need_save_data


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


try:
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine
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


# Asynchronous executor for I/O tasks
ATTN_DATA_SAVER = concurrent.futures.ThreadPoolExecutor(max_workers=4)  # Adjust `max_workers` as needed

def async_save(arr:np.array, path: str):
    with open(path, 'wb') as f:
        np.save(f, arr)
        # torch.save(tensor, f)

class CaptureAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, 
        attn_weights: torch.Tensor,
        attn_save_path: str,
        grad_save_path: str
    ):
        with open(attn_save_path, 'wb') as f:
            # np.save(f, attn_weights.clone().detach().cpu().float().numpy())
            ATTN_DATA_SAVER.submit(async_save, attn_weights.clone().detach().cpu().float().numpy(), attn_save_path)

        ctx.grad_save_path = grad_save_path
        return attn_weights

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor):
        grad_save_path = ctx.grad_save_path
        with open(grad_save_path, 'wb') as f:
            # np.save(f, grad_output.clone().detach().cpu().float().numpy())
            ATTN_DATA_SAVER.submit(async_save, grad_output.clone().detach().cpu().float().numpy(), grad_save_path)
    
        return grad_output, None, None

def get_save_path(layer_idx, rank):
    iter_idx = get_iter_cnt(rank)
    os.makedirs(f'./attn_data/{iter_idx}/{layer_idx}', exist_ok=True)

    attn_save_path = f'./attn_data/{iter_idx}/{layer_idx}/attn_{rank}.npy'
    grad_save_path = f'./attn_data/{iter_idx}/{layer_idx}/grad_{rank}.npy'

    return attn_save_path, grad_save_path

def capture_attention_forward(attn_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
    rank = torch.distributed.get_rank()
    if need_save_data(rank):
        attn_save_path, grad_save_path = get_save_path(layer_idx, rank)
        return CaptureAttention.apply(attn_weights, attn_save_path, grad_save_path)
    else:
        return attn_weights

def capture_attention_anno(attn_weights: torch.Tensor, layer_idx: int) -> str:
    # [batch_size, num_heads, query_len, query_len]
    return 'b num_heads l^ l^ -> b num_heads l^ l^'
register_op(capture_attention_anno)(capture_attention_forward)

class CustomAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_cnt = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = capture_attention_forward(attn_weights, self.layer_idx)

        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        self.iter_cnt += 1
        return attn_output, attn_weights, past_key_value


class NNScalerLlamaFlashAttention2(LlamaAttention):
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
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        # query_states - [batch_size, num_heads, query_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        # query_states - [batch_size, query_len, num_heads, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and q_len != 1

        attn_output = nnscaler_flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate, causal=causal
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def nnscaler_flash_attention_forward(
    query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None, causal=True
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API (shape: [batch_size, seq_len, num_heads, head_dim])
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
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
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


def llama_flash_attention_anno(query_states, key_states, value_states, attention_mask, *args, **kwargs) -> str:
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


register_op(llama_flash_attention_anno)(nnscaler_flash_attention_forward)


def nnscaler_llama_init(attn_type: str='flash'):
    # This imports the reference to the original dictionary object (`LLAMA_ATTENTION_CLASSES`) in the module.
    # The original 
    if attn_type == 'flash':
        LLAMA_ATTENTION_CLASSES["flash_attention_2"] = NNScalerLlamaFlashAttention2
    elif attn_type == 'custom':
        LLAMA_ATTENTION_CLASSES["eager"] = CustomAttention
    else:
        raise ValueError(f"Invalid attention type {attn_type}")

    LlamaRMSNorm.forward = rmsnorm_fwd


# ---------------------------------------------------------
# # Draft:
# class CustomAttenFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(
#         ctx: torch.autograd.function.FunctionCtx, 
#         query_states, key_states, value_states, 
#         attn_mask, attn_dropout, training
#     ):
#         # query_states: [batch_size, num_heads, query_len, head_dim]
#         head_dim = query_states.size(-1)

#         # Compute scaled dot-product attention scores
#         orig_attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(head_dim)
        
#         # Apply mask if provided
#         if attn_mask is not None:
#             causal_mask = attn_mask[:, :, :, :key_states.shape[-2]]
#             orig_attn_weights = orig_attn_weights + causal_mask

#         # Softmax across the last dimension and apply dropout if in training mode
#         orig_attn_weights = nn.functional.softmax(orig_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
#         attn_weights = nn.functional.dropout(orig_attn_weights, p=attn_dropout, training=training)

#         # Multiply attention weights by value states to get the output
#         attn_output = torch.matmul(attn_weights, value_states)

#         # Save for backward pass
#         ctx.save_for_backward(query_states, key_states, value_states, attn_weights, orig_attn_weights, attn_mask)
#         ctx.attn_dropout = attn_dropout
#         ctx.training = training

#         return attn_output

#     @staticmethod
#     def backward(ctx: torch.autograd.function.FunctionCtx, grad_output):
#         query_states, key_states, value_states, attn_weights, orig_attn_weights, attn_mask = ctx.saved_tensors
#         attn_dropout = ctx.attn_dropout
#         training = ctx.training

#         # Compute gradients w.r.t. value states
#         # attn_weights - [batch_size, num_heads, query_len, query_len]
#         # grad_output - [batch_size, num_heads, query_len, head_dim]
#         # grad_value_states - [batch_size, num_heads, query_len, head_dim]
#         grad_value_states = torch.matmul(attn_weights.transpose(-2, -1), grad_output)

#         # Compute gradients w.r.t. attention weights
#         # grad_output - [batch_size, num_heads, query_len, head_dim]
#         # value_states - [batch_size, num_heads, query_len, head_dim]
#         # grad_attn_weights - [batch_size, num_heads, query_len, query_len]
#         grad_attn_weights = torch.matmul(grad_output, value_states.transpose(-2, -1))

#         # Directly assume grad w.r.t attn_weights is equivalent to grad w.r.t orig_attn_weights because 
#         grad_orig_attn_weights = grad_attn_weights

#         # Compute gradients w.r.t. matrix multiplication before softmax
#         # Because the direct operand of Softmax is the matrix multiplication result before dropout, so we use orig_attn_weights here
#         grad_orig_attn_mul = grad_orig_attn_weights * orig_attn_weights * (1 - orig_attn_weights)

#         # ------------------------------------------------
#         # Save the gradients w.r.t (orig)_attn_weights to files


#         # -------------------------------------------------


#         # Compute gradients w.r.t. query and key states
#         # grad_orig_attn_mul - [batch_size, num_heads, query_len, query_len]
#         # key_states - [batch_size, num_heads, query_len, head_dim]
#         # query_states - [batch_size, num_heads, query_len, head_dim]
#         grad_query_states = torch.matmul(grad_orig_attn_mul, key_states)
#         grad_key_states = torch.matmul(grad_attn_weights, query_states)

#         return grad_query_states, grad_key_states, grad_value_states, None, None, None
