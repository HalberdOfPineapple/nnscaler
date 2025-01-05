#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# This file modifies the official modeling_llama.py file at runtime to
# 1. register the flash attention function to nnscaler and update related code
# 2. replace the un-fused RMSNorm with apex's fused version
import os
import math
import json
import torch
import logging
import warnings
import torch.distributed
import torch.nn.functional as F
logger = logging.getLogger(__name__)

from typing import List, Optional, Tuple, Dict

from transformers.cache_utils import Cache
from transformers.utils import logging

from nnscaler.graph.parser.register import register_op
from nnscaler.ir import IRTensor

from phi3 import Phi3Attention, apply_rotary_pos_emb, repeat_kv
from custom_trainer import get_iter_cnt
from flexprefill import flex_attn_forward

from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func


class FlexPrefillAttention(Phi3Attention):
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
    
    def init_flex_prefill_parameters(
            self,
            attn_config: Optional[Dict[str, int]] = None,
    ):
        # import apply_rotary_pos_emb
        self.apply_rotary_pos_emb = True
        self.gamma = attn_config.get('gamma', 0.9)
        self.tau = attn_config.get('tau', 0.1)
        self.min_budget = attn_config.get('min_budget', None)
        self.max_budget = attn_config.get('max_budget', None)
        self.block_size_M = attn_config.get('block_size_M', 64)
        self.block_size_N = attn_config.get('block_size_N', self.block_size_M)


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

        # -----------------------------
        head_indices = torch.arange(query_states.shape[1], device=query_states.device, dtype=torch.int32)
        attn_output = attn_fwd_by_heads(
            query_states, key_states, value_states, head_indices,
            bsz=bsz, q_len=q_len, head_dim=self.head_dim,
            gamma=self.gamma,
            layer_idx=self.layer_idx, 
            min_budget=self.min_budget, 
            max_budget=self.max_budget,
            block_size_M=self.block_size_M, 
            block_size_N=self.block_size_N,
        ).view(bsz, q_len, self.num_heads, self.head_dim)
        # -----------------------------

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions: attn_weights = None
        return attn_output, attn_weights, past_key_value


def attn_fwd_by_heads(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_indices: torch.Tensor,
    bsz: int,
    q_len: int,
    head_dim: int,
    layer_idx: int,
    gamma: float,
    min_budget: Optional[float] = None,
    max_budget: Optional[float] = None,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    with torch.autograd.set_detect_anomaly(True):
        assert(query_states.shape[1] == head_indices.shape[-1])
        output_list = []

        for head in range(query_states.size(1)):
            head_idx = head_indices[head].item()
            q = query_states[:, head, :, :].unsqueeze(1) # (bsz, 1, q_len, head_dim)
            k = key_states[:, head, :, :].unsqueeze(1)
            v = value_states[:, head, :, :].unsqueeze(1)

            attn_output_head = flex_attn_forward(
                q, k, v,
                gamma=gamma,
                layer_idx=layer_idx, head_idx=head_idx,
                min_budget=min_budget, 
                max_budget=max_budget,
                block_size_M=block_size_M, 
                block_size_N=block_size_N,
            ).view(bsz, q_len, 1, head_dim)

            # attn_output_head = flash_attn_func(
            #         q.transpose(1, 2),
            #         k.transpose(1, 2),
            #         v.transpose(1, 2),
            #         causal=True,
            #     )
            output_list.append(attn_output_head)
        output = torch.cat(output_list, dim=2)
    return output



def minfer_attn_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    if query_states.shape[1] != key_states.shape[1]:
        assert query_states.shape[1] % key_states.shape[1] == 0
        group_size = query_states.shape[1] // key_states.shape[1]
        assert query_states.shape[1] == value_states.shape[1] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    return f'b {q_anno} l^ hd^, b {kv_anno} s^ hd^, b {kv_anno} s^ vd^, {q_anno} -> b l^ {q_anno} vd^'

if __name__ != "__main__":
    # register_op(minfer_attn_anno)(flex_attn_forward)
    register_op(minfer_attn_anno)(attn_fwd_by_heads)



def test_flex_prefill():
    ATOL, RTOL = 5e-2, 5e-2

    context_size = 131072
    num_heads = 32
    head_dim = 96
    
    gamma = 1.0
    min_budget = None
    max_budget = None

    q = torch.randn((1, num_heads, context_size, head_dim), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k = torch.randn((1, num_heads, context_size, head_dim), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    v = torch.randn((1, num_heads, context_size, head_dim), dtype=torch.bfloat16, device='cuda', requires_grad=True)

    o: torch.Tensor = attn_fwd_by_heads(
        q, k, v, torch.arange(num_heads, device='cuda', dtype=torch.int32),
        bsz=1, q_len=context_size, head_dim=head_dim, layer_idx=0,
        gamma=gamma, min_budget=min_budget, max_budget=max_budget,
    )
    print(f"o shape: {o.shape}")

    # # count how many zeros in o
    # num_ones = torch.sum(o).cpu().item()
    # print(f"number of ones in output: {num_ones}")


    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    o.retain_grad()

    torch.cuda.synchronize()
    loss = torch.square(o).sum(dtype=torch.float64)
    torch.cuda.synchronize()
    loss.backward()

    o_grad = o.grad.clone()
    q_grad = q.grad.clone()
    k_grad = k.grad.clone()
    v_grad = v.grad.clone()
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()


    from flash_attn import flash_attn_func
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o_ref: torch.Tensor = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )
    o_ref.retain_grad()

    # num_ones_ref = torch.sum(o_ref).cpu().item()
    # print(f"number of ones in output (ref): {num_ones_ref}")

    print(f"o_ref shape: {o_ref.shape}")
    torch.cuda.synchronize()
    loss = torch.square(o_ref).sum(dtype=torch.float64)
    torch.cuda.synchronize()
    loss.backward()

    o_ref_grad = o_ref.grad.clone()
    q_ref_grad = q.grad.clone()
    k_ref_grad = k.grad.clone()
    v_ref_grad = v.grad.clone()

    # --------------------------------------------------------------------------------
    print('-' * 80)
    for head_idx in range(num_heads):
        print('-' * 40)
        output_close = torch.allclose(o[0, :, head_idx, :], o_ref[0, :, head_idx, :], atol=ATOL, rtol=RTOL)
        output_grad_close = torch.allclose(o_grad[0, :, head_idx, :], o_ref_grad[0, :, head_idx, :], atol=ATOL, rtol=RTOL)
        q_grad_close = torch.allclose(q_grad[0, :, head_idx, :], q_ref_grad[0, :, head_idx, :], atol=ATOL, rtol=RTOL)
        k_grad_close = torch.allclose(k_grad[0, :, head_idx, :], k_ref_grad[0, :, head_idx, :], atol=ATOL, rtol=RTOL)
        v_grad_close = torch.allclose(v_grad[0, :, head_idx, :], v_ref_grad[0, :, head_idx, :], atol=ATOL, rtol=RTOL)

        if not output_close: 
            print('-' * 20)
            if not output_close: print(f"Head {head_idx} output is not close")
            print(f"Output:\n{o[0, :, head_idx, :]}")
            print(f"Output Ref:\n{o_ref[0, :, head_idx, :]}")

        if not output_grad_close: 
            print('-' * 20)
            if not output_grad_close: print(f"Head {head_idx} output grad is not close")
            print(f"Output:\n{o_grad[0, :, head_idx, :]}")
            print(f"Output Ref:\n{o_ref_grad[0, :, head_idx, :]}")

        if not q_grad_close: 
            print('-' * 20)
            if not q_grad_close: print(f"Head {head_idx} q grad is not close")
            print(f"Q Grad:\n{q_grad[0, :, head_idx, :]}")
            print(f"Q Grad Ref:\n{q_ref_grad[0, :, head_idx, :]}")
        
        if not k_grad_close: 
            print('-' * 20)
            if not k_grad_close: print(f"Head {head_idx} k grad is not close")
            print(f"K Grad:\n{k_grad[0, :, head_idx, :]}")
            print(f"K Grad Ref:\n{k_ref_grad[0, :, head_idx, :]}")

        if not v_grad_close: 
            print('-' * 20)
            if not v_grad_close: print(f"Head {head_idx} v grad is not close")
            print(f"V Grad:\n{v_grad[0, :, head_idx, :]}")
            print(f"V Grad Ref:\n{v_ref_grad[0, :, head_idx, :]}")

    

if __name__ == "__main__":
    test_flex_prefill()