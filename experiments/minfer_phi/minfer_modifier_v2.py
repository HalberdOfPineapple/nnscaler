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

from minfer_ops import vs_attn_forward, gen_block_indices
from phi3 import Phi3Attention, apply_rotary_pos_emb, repeat_kv
from modeling_modifier import nnscaler_flash_attention_forward
from custom_trainer import get_iter_cnt

from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func



class MInferAttention(Phi3Attention):
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
    
    def init_minference_parameters(
            self,
            start_sparse_iter: int = 0,
            enable_sparse: bool = True,
            adaptive_sparse: bool = False,
            sparse_head_list: Optional[List[bool]] = None,
    ):
        self.start_sparse_iter = start_sparse_iter
        self.enable_sparse = enable_sparse
        self.adaptive_sparse = adaptive_sparse and self.enable_sparse
        
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
        self.sparse_head_list = sparse_head_list

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
        attn_output = selective_attn_fwd(
            query_states, key_states, value_states, head_indices,
            bsz, q_len, self.head_dim, self.layer_idx,
            self.adaptive_sparse, self.start_sparse_iter, self.enable_sparse,
            self.best_pattern, self.sparse_head_list, attention_mask, attn_dropout
        )
        # -----------------------------

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions: attn_weights = None
        return attn_output, attn_weights, past_key_value

def selective_attn_fwd(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_indices: torch.Tensor,
    bsz: int,
    q_len: int,
    head_dim: int,
    layer_idx: int,
    adaptive_sparse: bool,
    start_sparse_iter: int,
    layer_enable_sparse: bool,
    pattern_dict: Dict[int, Tuple[str, int, int, int]],
    sparse_head_list: Optional[List[int]],
    attention_mask: Optional[torch.Tensor],
    attn_dropout: float,
):
    try:
        curr_iter = get_iter_cnt(torch.distributed.get_rank())
    except ValueError as e:
        # print(f'[Warning] current iter is not available: {e}', flush=True)
        curr_iter = 0

    enable_sparse = layer_enable_sparse and curr_iter >= start_sparse_iter
    if enable_sparse and adaptive_sparse:
        raise NotImplementedError("Adaptive sparse is not implemented yet")
    elif enable_sparse:
        attn_output = attn_fwd_by_heads(
            query_states, key_states, value_states, head_indices,
            bsz, q_len, head_dim, layer_idx, 
            pattern_dict,
            sparse_head_list,
        ) # expect:  b {q_anno} l^ vd^'
    else:
        # if curr_iter < start_sparse_iter: 
        #     print(
        #         f'{__name__} | Current iteration: {curr_iter} < Start sparse iteration: {start_sparse_iter}, sparse is disabled', 
        #         flush=True
        #     )
        # else:
        #     print(f'{__name__} | Layer {layer_idx} is disabled to use sparse', flush=True)

        attn_output = nnscaler_flash_attention_forward(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            attention_mask,
            q_len,
            causal=True,
            dropout=attn_dropout,
        ) # [B, N, H, D]

    return attn_output

def attn_fwd_by_heads(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_indices: torch.Tensor,
    bsz: int,
    q_len: int,
    head_dim: int,
    layer_idx: int,
    pattern_dict: Dict[int, Tuple[str, int, int, int]],
    sparse_head_list: Optional[List[int]] = None,
    adaptive_sparse: bool = False,
):
    if sparse_head_list is None: 
        sparse_head_list = [True] * len(pattern_dict)

    with torch.autograd.set_detect_anomaly(True):
        assert(query_states.shape[1] == head_indices.shape[-1])
        output_list = []

        for head in range(query_states.size(1)):
            head_idx = head_indices[head].item()
            q = query_states[:, head, :, :].unsqueeze(1) # (bsz, 1, q_len, head_dim)
            k = key_states[:, head, :, :].unsqueeze(1)
            v = value_states[:, head, :, :].unsqueeze(1)
            pattern = pattern_dict.get(head_idx, ("vertical_and_slash", 100, 6096, 1))

            # if adaptive_sparse is disabled, the sparse totally depends on the sparse_head_lsit and mark adaptive_activated as True
            # adaptive_activated = True if not adaptive_sparse else attn_recall_activated(
            #     q, k, v, q_len, head_dim,
            #     pattern
            # )
            adaptive_activated = True

            # if search is disabled and the current layer is beyond  starting layer 
            # => apply the kernel for calculating the attention based on the best pattern
            if sparse_head_list[head_idx] and adaptive_activated:
                ty, vertical_size, slash_size, _ = pattern

                try:
                    attn_output_head = vs_attn_forward(
                        q, k, v,
                        q_len, vertical_size, slash_size, head_dim,
                        layer_idx, head_indices[head].item()
                    ).view(bsz, q_len, 1, head_dim)
                except Exception as e:
                    print(f"Error in (Layer {layer_idx}, Head {head_indices[head].item()}) with pattern {pattern}")

                    import traceback
                    traceback.print_exc()
 
                    attn_output_head = torch.randn(bsz, q_len, 1, head_dim, device=query_states.device)
            else:
                # print(f"Layer {layer_idx} | Head {head_indices[head].item()} | Sparsity disabled", flush=True)
                attn_output_head = flash_attn_func(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    causal=True,
                )
            output_list.append(attn_output_head)
        output = torch.cat(output_list, dim=2)
    return output



# LAST_Q = 128
# ATTN_RECALL_THRESHOLD = 0.8
# def build_sparse_mask(
#     block_count: torch.Tensor, # [BATCH, N_HEADS, NUM_ROWS], note that NUM_ROWS means the number of 64-sized rows
#     block_offset, # [BATCH, N_HEADS, NUM_ROWS, NNZ_S], which refers to the start of the non-sparse K/V blocks to be computed with the corresponding Q block
#     column_count, # [BATCH, N_HEADS, NUM_ROWS]
#     column_index, # [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
#     bsz: int,
#     q_len: int,
#     block_size_M: int = 64,
#     block_size_N: int = 64,
# ):
#     attn_mask = torch.zeros((bsz, LAST_Q, q_len))
#     row_offset = (q_len - LAST_Q) // block_size_M
#     num_row_blocks = LAST_Q // block_size_M

#     for batch_idx in range(bsz):
#         for row_idx in range(row_offset, row_offset + num_row_blocks):
#             block_cnt = block_count[batch_idx, 0, row_idx]
#             block_off = block_offset[batch_idx, 0, row_idx]
#             col_cnt = column_count[batch_idx, 0, row_idx]
#             col_idx = column_index[batch_idx, 0, row_idx]

#             row_start = row_idx * block_size_M - (q_len - LAST_Q)
#             row_end = min(LAST_Q, row_start + block_size_M)

#             for i in range(block_cnt):
#                 curr_block_start = block_off[i]
#                 curr_block_end = min(q_len, curr_block_start + block_size_N)

#                 attn_mask[batch_idx, row_start:row_end, curr_block_start:curr_block_end] = \
#                     torch.ones((row_end - row_start, curr_block_end - curr_block_start))

#             for j in range(col_cnt):
#                 col_index = col_idx[j]
#                 # attn_mask[batch_idx, row_start:row_end, col_index] = 1
#                 attn_mask[batch_idx, row_start:row_end, col_index] = \
#                     torch.ones((row_end - row_start))
#     return attn_mask

# def attn_recall_activated(
#         query_states: torch.Tensor,
#         key_states: torch.Tensor,
#         value_states: torch.Tensor,
#         bsz: int,
#         q_len: int,
#         head_dim: int,
#         pattern: Tuple[str, int, int, int],
# ):
#     ty, vertical_size, slash_size, _ = pattern

#     block_count, block_offset, column_count, column_index, _ = gen_block_indices(
#         query_states, 
#         key_states,
#         value_states,
#         q_len, vertical_size, slash_size, head_dim,
#         block_size_M=64, block_size_N=64
#     )
#     sparse_mask = build_sparse_mask(
#         block_count, block_offset, column_count, column_index,
#         bsz, q_len
#     ).view(bsz, 1, LAST_Q, q_len)

#     causal_mask = torch.arange(q_len - LAST_Q, q_len, device=query_states.device)[:, None] >= torch.arange(q_len, device=query_states.device)[None, :]
#     qk = torch.einsum(
#         f'bhmk, bhnk -> bhmn', 
#         query_states[:, :, -LAST_Q:, :].contiguous().to(query_states.device), # [BATCH, N_HEADS, LAST_Q, D_HEAD]
#         key_states.to(query_states.device), # [BATCH, N_HEADS, N_CTX, D_HEAD]
#     ) / math.sqrt(head_dim) # [BATCH, N_HEADS, LAST_Q, N_CTX]
#     qk = torch.where(causal_mask, qk, float('-inf'))
#     attn = F.softmax(qk, dim=-1)

#     sparse_attn = torch.where(sparse_mask, attn, 0) # [BATCH, 1, LAST_Q, N_CTX]
#     avg_attn_recall = torch.mean(torch.sum(sparse_attn, dim=-1), dim=-1)[0] # [BATCH, 1]
#     return avg_attn_recall > ATTN_RECALL_THRESHOLD



def minfer_attn_anno(query_states, key_states, value_states, *args, **kwargs) -> str:
    if query_states.shape[1] != key_states.shape[1]:
        assert query_states.shape[1] % key_states.shape[1] == 0
        group_size = query_states.shape[1] // key_states.shape[1]
        assert query_states.shape[1] == value_states.shape[1] * group_size
        q_anno = f'(group_num {group_size})'
        kv_anno = 'group_num'
    else:
        q_anno = kv_anno = 'num_heads'

    # return f'b {q_anno} l^ hd^, b {kv_anno} s^ hd^, b {kv_anno} s^ vd^, {q_anno} -> b {q_anno} l^ vd^'
    return f'b {q_anno} l^ hd^, b {kv_anno} s^ hd^, b {kv_anno} s^ vd^, {q_anno} -> b l^ {q_anno} vd^'

if __name__ != "__main__":
    # register_op(minfer_attn_anno)(attn_fwd_by_heads)
    register_op(minfer_attn_anno)(selective_attn_fwd)

def attn_fwd_by_heads_v2(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    head_indices: torch.Tensor,
    bsz: int,
    q_len: int,
    head_dim: int,
    layer_idx: int,
    pattern: Tuple[str, int, int, int],
):
    torch.cuda.synchronize()
    with torch.autograd.set_detect_anomaly(True):
        num_heads = query_states.shape[1]
        assert(num_heads == head_indices.shape[-1])

        ty, vertical_size, slash_size, _ = pattern
        output = vs_attn_forward(
            query_states,
            key_states,
            value_states,
            q_len, vertical_size, slash_size, head_dim,
            layer_idx, head_indices
        ).view(bsz, num_heads, q_len, head_dim)

    torch.cuda.synchronize()
    return output


ATOL, RTOL = 5e-2, 5e-2
# -----------------------------------------------------------
def test_wo_chunks(attn_ver: int=1):
    from flash_attn import flash_attn_func
    
    batch_size = 1
    context_size = 131072
    num_heads = 32
    head_dim = 96

    q = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    k = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    v = torch.randn((batch_size, num_heads, context_size, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    head_indices = torch.arange(num_heads, device='cuda', dtype=torch.int32)

    if attn_ver == 1:
        o = attn_fwd_by_heads(
            q, k, v, head_indices,
            batch_size, context_size, head_dim, 0,
            ("vertical_and_slash", 100, context_size, 1)
        ) # [batch_size, context_size, num_heads, head_dim]
    else:
        o = attn_fwd_by_heads_v2(
            q, k, v, head_indices,
            batch_size, context_size, head_dim, 0,
            ("vertical_and_slash", 100, context_size, 1)
        )
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
    o_ref = o_ref.transpose(1, 2)

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
        output_close = torch.allclose(o[0, head_idx, :, :], o_ref[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        output_grad_close = torch.allclose(o_grad[0, head_idx, :, :], o_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        q_grad_close = torch.allclose(q_grad[0, head_idx, :, :], q_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        k_grad_close = torch.allclose(k_grad[0, head_idx, :, :], k_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        v_grad_close = torch.allclose(v_grad[0, head_idx, :, :], v_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)

        if not output_close: print(f"Head {head_idx} output is not close")
        if not output_grad_close: print(f"Head {head_idx} output grad is not close")
        if not q_grad_close: print(f"Head {head_idx} q grad is not close")
        if not k_grad_close: print(f"Head {head_idx} k grad is not close")
        if not v_grad_close: print(f"Head {head_idx} v grad is not close")

def test_w_chunks():
    from flash_attn import flash_attn_func

    batch_size = 1
    context_size = 131072
    num_heads = 1 # 32
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
    o_ref = o_ref.transpose(1, 2)
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
        output_close = torch.allclose(o[0, head_idx, :, :], o_ref[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        output_grad_close = torch.allclose(o_grad[0, head_idx, :, :], o_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        q_grad_close = torch.allclose(q_grad[0, head_idx, :, :], q_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        k_grad_close = torch.allclose(k_grad[0, head_idx, :, :], k_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        v_grad_close = torch.allclose(v_grad[0, head_idx, :, :], v_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)

        if not output_close: print(f"Head {head_idx} output is not close")
        if not output_grad_close: print(f"Head {head_idx} output grad is not close")
        if not q_grad_close: print(f"Head {head_idx} q grad is not close")
        if not k_grad_close: print(f"Head {head_idx} k grad is not close")
        if not v_grad_close: print(f"Head {head_idx} v grad is not close")



def test_w_chunks():
    from flash_attn import flash_attn_func
    ATOL, RTOL = 5e-2, 5e-2
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
    o = torch.cat(o_chunks, dim=1)
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
    o_ref = o_ref.transpose(1, 2)
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
        output_close = torch.allclose(o[0, head_idx, :, :], o_ref[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        output_grad_close = torch.allclose(o_grad[0, head_idx, :, :], o_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        q_grad_close = torch.allclose(q_grad[0, head_idx, :, :], q_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        k_grad_close = torch.allclose(k_grad[0, head_idx, :, :], k_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)
        v_grad_close = torch.allclose(v_grad[0, head_idx, :, :], v_ref_grad[0, head_idx, :, :], atol=ATOL, rtol=RTOL)

        if not output_close: 
            print('-' * 20)
            if not output_close: print(f"Head {head_idx} output is not close")
            print(f"Output:\n{o[0, head_idx, :, :]}")
            print(f"Output Ref:\n{o_ref[0, head_idx, :, :]}")

        if not output_grad_close: 
            print('-' * 20)
            if not output_grad_close: print(f"Head {head_idx} output grad is not close")
            print(f"Output:\n{o_grad[0, head_idx, :, :]}")
            print(f"Output Ref:\n{o_ref_grad[0, head_idx, :, :]}")

        if not q_grad_close: 
            print('-' * 20)
            if not q_grad_close: print(f"Head {head_idx} q grad is not close")
            print(f"Q Grad:\n{q_grad[0, head_idx, :, :]}")
            print(f"Q Grad Ref:\n{q_ref_grad[0, head_idx, :, :]}")
        
        if not k_grad_close: 
            print('-' * 20)
            if not k_grad_close: print(f"Head {head_idx} k grad is not close")
            print(f"K Grad:\n{k_grad[0, head_idx, :, :]}")
            print(f"K Grad Ref:\n{k_ref_grad[0, head_idx, :, :]}")

        if not v_grad_close: 
            print('-' * 20)
            if not v_grad_close: print(f"Head {head_idx} v grad is not close")
            print(f"V Grad:\n{v_grad[0, head_idx, :, :]}")
            print(f"V Grad Ref:\n{v_ref_grad[0, head_idx, :, :]}")


def run_w_chunks():
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
    for q_chunk, k_chunk, v_chunk, head_indice_chunk in zip(q_chunks, k_chunks, v_chunks, head_indices_chunks):
        o_chunk = attn_fwd_by_heads(
            q_chunk, k_chunk, v_chunk, head_indice_chunk,
            batch_size, context_size, head_dim, 
            layer_idx=0,
            pattern=("vertical_and_slash", 100, context_size, 1),
        )
        o_chunks.append(o_chunk)
    o = torch.cat(o_chunks, dim=1)

    loss = torch.square(o).sum(dtype=torch.float64)
    loss.backward()

def run_w_chunks_fa():
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
    for q_chunk, k_chunk, v_chunk, head_indice_chunk in zip(q_chunks, k_chunks, v_chunks, head_indices_chunks):
        o_chunk = flash_attn_func(
            q_chunk.transpose(1, 2),
            k_chunk.transpose(1, 2),
            v_chunk.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=None,
            causal=True,
        )
        o_chunks.append(o_chunk)
    o = torch.cat(o_chunks, dim=1)

    loss = torch.square(o).sum(dtype=torch.float64)
    loss.backward()


if __name__ == "__main__":
    import sys
    mode: str = sys.argv[1]

    print('-' * 80)
    if mode.startswith('test_wo_chunks'):
        if mode.split('_')[-1].startswith('v'):
            mode, ver = mode.split('_')[:-1], int(mode.split('_')[-1][-1])
        else:
            ver = 1
        print(f"Test without chunks (ver={ver})...")
        test_wo_chunks(ver)
    elif mode == 'test_w_chunks':
        print(f"Test with chunks...")
        test_w_chunks()
    elif mode == 'run_w_chunks':
        print(f"Run with chunks...")
        run_w_chunks()
    elif mode == 'run_w_chunks_fa':
        print(f"Run with chunks (FA)...")
        run_w_chunks_fa()
