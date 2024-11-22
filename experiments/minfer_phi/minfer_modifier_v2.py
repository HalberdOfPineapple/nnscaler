#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

# This file modifies the official modeling_llama.py file at runtime to
# 1. register the flash attention function to nnscaler and update related code
# 2. replace the un-fused RMSNorm with apex's fused version
import os
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

from minfer_ops import vs_attn_forward
from phi3 import Phi3Attention as PhiAttention, apply_rotary_pos_emb, repeat_kv


class MInferAttention(PhiAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

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
        # cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        pattern = self.best_pattern.get(self.layer_idx, ("vertical_and_slash", 1000, 6096, 1))
        # pattern = ("vertical_and_slash", 100, q_len, 1)
        head_indices = torch.arange(query_states.shape[1], device=query_states.device, dtype=torch.int32)


        # print(f"query_states: {query_states.shape}, key_states: {key_states.shape}, value_states: {value_states.shape}, head_indices: {head_indices.shape}")
        attn_output = attn_fwd_by_heads(
            query_states, key_states, value_states, head_indices,
            bsz, q_len,  self.head_dim, pattern,
        ) # expect:  b l^ {q_anno} vd^'
        

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
    pattern: Tuple[str, int, int, int],
):
    # print(f"head_indices: {head_indices}")
    with torch.autograd.set_detect_anomaly(True):
        output_list = []
        assert(query_states.shape[1] == head_indices.shape[-1])

        for head in range(query_states.size(1)):
            q = query_states[:, head, :, :].unsqueeze(1) # (bsz, 1, q_len, head_dim)
            k = key_states[:, head, :, :].unsqueeze(1)
            v = value_states[:, head, :, :].unsqueeze(1)

            # if search is disabled and the current layer is beyond  starting layer 
            # => apply the kernel for calculating the attention based on the best pattern
            ty, vertical_size, slash_size, _ = pattern
            attn_output_head = vs_attn_forward(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                q_len, vertical_size, slash_size, head_dim
            ).view(bsz, q_len, 1, head_dim)
            # output_list.append(attn_output_head.squeeze(2))
            output_list.append(attn_output_head)

        # output = torch.stack(output_list, dim=2)
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

# if __name__ != "__main__":
#     register_op(minfer_attn_anno)(attn_fwd_by_heads)
register_op(minfer_attn_anno)(attn_fwd_by_heads)



def attn_fwd_by_heads_v2(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_indices: torch.Tensor,
    bsz: int,
    q_len: int,
    head_dim: int,
    pattern: Tuple[str, int, int, int],
):
    # print(f"head_indices: {head_indices}")

    torch.cuda.synchronize()
    with torch.autograd.set_detect_anomaly(True):
        num_heads = query_states.shape[1]
        assert(num_heads == head_indices.shape[-1])

        ty, vertical_size, slash_size, _ = pattern
        output = vs_attn_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            q_len, vertical_size, slash_size, head_dim
        ).view(bsz, q_len, num_heads, head_dim)

    torch.cuda.synchronize()
    return output

def test_wo_chunks():
    from flash_attn import flash_attn_func

    batch_size = 1
    context_size = 131072
    num_heads = 32
    head_dim = 96

    q = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    k = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    v = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    head_indices = torch.arange(num_heads, device='cuda', dtype=torch.int32)

    o = attn_fwd_by_heads(
        q, k, v, head_indices,
        batch_size, context_size, head_dim, 
        ("vertical_and_slash", 100, context_size, 1)
    ) # [batch_size, context_size, num_heads, head_dim]
    print(f"o.shape: {o.shape}")

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    o.retain_grad()

    loss = torch.square(o).sum(dtype=torch.float64)
    torch.cuda.synchronize()
    loss.backward()
    torch.cuda.synchronize()
    
    o_grad = o.grad.clone()
    q_grad = q.grad.clone()
    k_grad = k.grad.clone()
    v_grad = v.grad.clone()
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    print(f"o_grad.shape: {o_grad.shape}")  
    print(f"q_grad.shape: {q_grad.shape}")
    print(f"k_grad.shape: {k_grad.shape}")
    print(f"v_grad.shape: {v_grad.shape}")

    o_ref = flash_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    ) # b l^ {q_anno} vd^
    print(f"o_ref.shape: {o_ref.shape}")

    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    o_ref.retain_grad()

    loss_ref = torch.square(o_ref).sum(dtype=torch.float64)
    loss_ref.backward()

    o_ref_grad = o_ref.grad.clone()
    q_ref_grad = q.grad.clone()
    k_ref_grad = k.grad.clone()
    v_ref_grad = v.grad.clone()

    print(f"o_ref_grad.shape: {o_ref_grad.shape}")
    print(f"q_ref_grad.shape: {q_ref_grad.shape}")
    print(f"k_ref_grad.shape: {k_ref_grad.shape}")
    print(f"v_ref_grad.shape: {v_ref_grad.shape}")


    for head_idx in range(num_heads):
        print('-' * 40)
        output_close = torch.allclose(o[0, :, head_idx, :], o_ref[0, :, head_idx, :], atol=1e-2, rtol=1e-2)
        output_grad_close = torch.allclose(o_grad[0, :, head_idx, :], o_ref_grad[0, :, head_idx, :], atol=1e-2, rtol=1e-2)
        q_grad_close = torch.allclose(q_grad[0, head_idx, :, :], q_ref_grad[0, head_idx, :, :], atol=1e-2, rtol=1e-2)
        k_grad_close = torch.allclose(k_grad[0, head_idx, :, :], k_ref_grad[0, head_idx, :, :], atol=1e-2, rtol=1e-2)
        v_grad_close = torch.allclose(v_grad[0, head_idx, :, :], v_ref_grad[0, head_idx, :, :], atol=1e-2, rtol=1e-2)

        if not output_close: print(f"Head {head_idx} output is not close")
        if not output_grad_close: print(f"Head {head_idx} output grad is not close")
        if not q_grad_close: print(f"Head {head_idx} q grad is not close")
        if not k_grad_close: print(f"Head {head_idx} k grad is not close")
        if not v_grad_close: print(f"Head {head_idx} v grad is not close")

def test_w_chunks():
    from flash_attn import flash_attn_func

    batch_size = 1
    context_size = 131072
    num_heads = 32
    head_dim = 96


    q = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    k = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    v = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    head_indices = torch.arange(num_heads, device='cuda', dtype=torch.int32)
    chunk_size = 4

    q_chunks = q.chunk(chunk_size, dim=1)
    k_chunks = k.chunk(chunk_size, dim=1)
    v_chunks = v.chunk(chunk_size, dim=1)
    head_indices_chunks = head_indices.chunk(chunk_size)
    o_chunks = []


    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    

    for q_chunk, k_chunk, v_chunk, head_indice_chunk in zip(q_chunks, k_chunks, v_chunks, head_indices_chunks):
        o_chunk = attn_fwd_by_heads(
            q_chunk, k_chunk, v_chunk, head_indice_chunk,
            batch_size, context_size, head_dim, 
            ("vertical_and_slash", 100, context_size, 1)
        )
        o_chunks.append(o_chunk)
    o = torch.cat(o_chunks, dim=2)
    o.retain_grad()
    print(f"o.shape: {o.shape}")

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

    print(f"o_grad.shape: {o_grad.shape}")  
    print(f"q_grad.shape: {q_grad.shape}")
    print(f"k_grad.shape: {k_grad.shape}")
    print(f"v_grad.shape: {v_grad.shape}")




    # ---------------------------------
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o_ref_chunks = []
    for q_chunk, k_chunk, v_chunk, head_indice_chunk in zip(q_chunks, k_chunks, v_chunks, head_indices_chunks):
        o_ref_chunk = flash_attn_func(
            q_chunk.transpose(1, 2),
            k_chunk.transpose(1, 2),
            v_chunk.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )
        o_ref_chunks.append(o_ref_chunk)
    o_ref = torch.cat(o_ref_chunks, dim=2)
    o_ref.retain_grad()
    print(f"o_ref.shape: {o_ref.shape}")

    torch.cuda.synchronize()
    loss_ref = torch.square(o_ref).sum(dtype=torch.float64)
    torch.cuda.synchronize()
    loss_ref.backward()

    o_ref_grad = o_ref.grad.clone()
    q_ref_grad = q.grad.clone()
    k_ref_grad = k.grad.clone()
    v_ref_grad = v.grad.clone()

    print(f"o_ref_grad.shape: {o_ref_grad.shape}")
    print(f"q_ref_grad.shape: {q_ref_grad.shape}")
    print(f"k_ref_grad.shape: {k_ref_grad.shape}")
    print(f"v_ref_grad.shape: {v_ref_grad.shape}")


    for head_idx in range(num_heads):
        print('-' * 40)
        output_close = torch.allclose(o[0, :, head_idx, :], o_ref[0, :, head_idx, :], atol=1e-2, rtol=1e-2)
        output_grad_close = torch.allclose(o_grad[0, :, head_idx, :], o_ref_grad[0, :, head_idx, :], atol=1e-2, rtol=1e-2)
        q_grad_close = torch.allclose(q_grad[0, head_idx, :, :], q_ref_grad[0, head_idx, :, :], atol=1e-2, rtol=1e-2)
        k_grad_close = torch.allclose(k_grad[0, head_idx, :, :], k_ref_grad[0, head_idx, :, :], atol=1e-2, rtol=1e-2)
        v_grad_close = torch.allclose(v_grad[0, head_idx, :, :], v_ref_grad[0, head_idx, :, :], atol=1e-2, rtol=1e-2)

        if not output_close: print(f"Head {head_idx} output is not close")
        if not output_grad_close: print(f"Head {head_idx} output grad is not close")
        if not q_grad_close: print(f"Head {head_idx} q grad is not close")
        if not k_grad_close: print(f"Head {head_idx} k grad is not close")
        if not v_grad_close: print(f"Head {head_idx} v grad is not close")



def test_w_chunks():
    from flash_attn import flash_attn_func

    batch_size = 1
    context_size = 131072
    num_heads = 32
    head_dim = 96
    hidden_size = num_heads * head_dim

    q = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    k = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    v = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    head_indices = torch.arange(num_heads, device='cuda', dtype=torch.int32)
    chunk_size = 4

    q_chunks = q.chunk(chunk_size, dim=1)
    k_chunks = k.chunk(chunk_size, dim=1)
    v_chunks = v.chunk(chunk_size, dim=1)
    head_indices_chunks = head_indices.chunk(chunk_size)
    o_chunks = []


    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    

    for q_chunk, k_chunk, v_chunk, head_indice_chunk in zip(q_chunks, k_chunks, v_chunks, head_indices_chunks):
        o_chunk = attn_fwd_by_heads(
            q_chunk, k_chunk, v_chunk, head_indice_chunk,
            batch_size, context_size, head_dim, 
            ("vertical_and_slash", 100, context_size, 1)
        )
        o_chunks.append(o_chunk)
    o = torch.cat(o_chunks, dim=2)
    o.retain_grad()
    print(f"o.shape: {o.shape}")

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

    print(f"o_grad.shape: {o_grad.shape}")  
    print(f"q_grad.shape: {q_grad.shape}")
    print(f"k_grad.shape: {k_grad.shape}")
    print(f"v_grad.shape: {v_grad.shape}")




    # ---------------------------------
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()

    o_ref_chunks = []
    for q_chunk, k_chunk, v_chunk, head_indice_chunk in zip(q_chunks, k_chunks, v_chunks, head_indices_chunks):
        o_ref_chunk = flash_attn_func(
            q_chunk.transpose(1, 2),
            k_chunk.transpose(1, 2),
            v_chunk.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )
        o_ref_chunks.append(o_ref_chunk)
    o_ref = torch.cat(o_ref_chunks, dim=2)
    o_ref.retain_grad()
    print(f"o_ref.shape: {o_ref.shape}")

    torch.cuda.synchronize()
    loss_ref = torch.square(o_ref).sum(dtype=torch.float64)
    torch.cuda.synchronize()
    loss_ref.backward()

    o_ref_grad = o_ref.grad.clone()
    q_ref_grad = q.grad.clone()
    k_ref_grad = k.grad.clone()
    v_ref_grad = v.grad.clone()

    print(f"o_ref_grad.shape: {o_ref_grad.shape}")
    print(f"q_ref_grad.shape: {q_ref_grad.shape}")
    print(f"k_ref_grad.shape: {k_ref_grad.shape}")
    print(f"v_ref_grad.shape: {v_ref_grad.shape}")


    for head_idx in range(num_heads):
        print('-' * 40)
        output_close = torch.allclose(o[0, :, head_idx, :], o_ref[0, :, head_idx, :], atol=1e-2, rtol=1e-2)
        output_grad_close = torch.allclose(o_grad[0, :, head_idx, :], o_ref_grad[0, :, head_idx, :], atol=1e-2, rtol=1e-2)
        q_grad_close = torch.allclose(q_grad[0, head_idx, :, :], q_ref_grad[0, head_idx, :, :], atol=1e-2, rtol=1e-2)
        k_grad_close = torch.allclose(k_grad[0, head_idx, :, :], k_ref_grad[0, head_idx, :, :], atol=1e-2, rtol=1e-2)
        v_grad_close = torch.allclose(v_grad[0, head_idx, :, :], v_ref_grad[0, head_idx, :, :], atol=1e-2, rtol=1e-2)

        if not output_close: print(f"Head {head_idx} output is not close")
        if not output_grad_close: print(f"Head {head_idx} output grad is not close")
        if not q_grad_close: print(f"Head {head_idx} q grad is not close")
        if not k_grad_close: print(f"Head {head_idx} k grad is not close")
        if not v_grad_close: print(f"Head {head_idx} v grad is not close")


if __name__ == "__main__":
    # print('-' * 80)
    # print(f"Test without chunks...")
    # test_wo_chunks()


    print('-' * 80)
    print(f"Test with chunks...")
    test_w_chunks()
