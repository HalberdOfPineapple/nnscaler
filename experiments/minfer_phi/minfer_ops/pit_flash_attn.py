import math
import torch
import numpy as np

import triton
import triton.language as tl

from minference.cuda import convert_vertical_slash_indexes
from minference.modules.minference_forward import (
    LAST_Q_MASK, sum_all_diagonal_matrix
)

from nnscaler.graph.parser.register import register_op

from flash_attn.flash_attn_interface import _flash_attn_backward

LN2 = 1 / 1.44269504

# @triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=1, num_warps=4),
#        triton.Config({}, num_stages=1, num_warps=8),
#        triton.Config({}, num_stages=2, num_warps=4),
#        triton.Config({}, num_stages=2, num_warps=8),
#        triton.Config({}, num_stages=3, num_warps=4),
#        triton.Config({}, num_stages=3, num_warps=8),
#        triton.Config({}, num_stages=4, num_warps=4),
#        triton.Config({}, num_stages=4, num_warps=8),
#        triton.Config({}, num_stages=5, num_warps=4),
#        triton.Config({}, num_stages=5, num_warps=8),
#    ],
#    key=['N_CTX'],
# )
@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_count, # [BATCH, N_HEADS, NUM_ROWS], note that NUM_ROWS means the number of 64-sized rows
    block_offset, # [BATCH, N_HEADS, NUM_ROWS, NNZ_S], which refers to the start of the non-sparse K/V blocks to be computed with the corresponding Q block
    column_count, # [BATCH, N_HEADS, NUM_ROWS]
    column_index, # [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    Out, # [BATCH, N_CTX, N_HEADS, D_HEAD]
    softmax_lse, # [BATCH, N_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_sz, stride_sh, stride_sm,
    Z, H, N_CTX, # (BATCH, N_HEADS, N_CTX)
    NUM_ROWS, NNZ_S, NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # (off_hz // H) -> batch index, (off_hz // H) * stride_qz -> batch offset in Q 
    # (off_hz % H) -> head index, (off_hz % H) * stride_qh -> head offset in Q
    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    # offs_m[:, None]: [BLOCK_M, 1], offs_m[:, None] * stride_qm: offsets for m in Q, offs_d[None, :]: offsets for d in Q
    # Note that for sequence length dimension, the slice is [:, None] while for the head dimension, the slice is [None, :]
    # the sum of these two slices is [BLOCK_M, BLOCK_DMODEL] -> representing the offsets for each index in the last two dimensions
    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk

    # the offsets for k and v are the same as q, they do not need to be offset by the sequence length dimension (to be moved in the loop)
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m) # load the number of non-sparse blocks
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S # pointer to the start of the list of non-sparse blocks
    
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m) # load the number of non-sparse column(block)s
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V 

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # 1/ln2 = lne/ln2 = log2(e) => 2^(x / ln2) = 2^(x * log2(e)) = (2^(log2(e)))^x = e^x
    qk_scale = sm_scale / LN2

    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen

    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index) # load the start (block-level) index of the non-sparse block
        cols = start_n + offs_n # the indices of elements in the non-sparse block of K, V 
        n_mask = cols < seqlen

        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk = qk + tl.dot(q, k)

        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug

        # acc_scale is the fix factor (exp(m_old - m_new))
        # multiply the previous accumulator by the fix factor and add the new value 
        acc = acc * acc_scale[:, None] + tl.dot(p.to(dtype), v)

        # -- update m_i and l_i --
        # l_i is the a BLOCK_M vector with each element being the sum of the corresponding row (exponential of qk)
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    for start_n in range(0, num_cols, BLOCK_N):
        # the key difference is that cols, as the indices, are stored in and loaded from cols_ptr
        # At each iteration, a block-sized chunk of column **indices** are loaded from cols_ptr, which can be discontinuous
        # But we just load the indices block by block, equivalent to translating the non-sparse columns together
        n_mask = start_n + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)

        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0).to(dtype)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0).to(dtype)

        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk = qk + tl.dot(q, k)

        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc = acc * acc_scale[:, None]
        acc = acc + tl.dot(p.to(dtype), v)

        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc = acc / l_i[:, None]
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)

    # softmax_lse is the log sum of the exponential of the qk values (log(sum(exp(qk))))
    # li is the sum of the exponential of the qk values (sum(exp(qk - m_i)))
    offs_lse = stride_sz * (off_hz // H) + stride_sh * (off_hz % H) + (start_m * BLOCK_M + tl.arange(0, BLOCK_M)) * stride_sm
    softmax_lse_ptr = softmax_lse + offs_lse[:, None]

    # log(sum(exp(qk - m_i))) = log(sum(exp(qk)) * exp(-m_i)) = log(sum(exp(qk))) - m_i
    softmax_lse_vals = tl.math.log(l_i) + m_i * LN2
    # softmax_lse_vals = (tl.math.log2(l_i) + m_i) / 1.44269504  # Remove it on MInference backward

    # directly use log because the scale has been applied to q, which makes values in softmax equivalent to exp(x/sqrt(d_model))
    tl.store(softmax_lse_ptr, softmax_lse_vals.to(dtype)[:, None], mask=m_mask) 


def pad_tensor(x, context_size, head_dim, block_size):
    # Pad to context_length dimension (N_CTX) to be divisible by BLOCK_SIZE_M
    seq_pad = ((context_size + block_size - 1) // block_size) * block_size - context_size
    dim_pad = 2 ** math.ceil(math.log2(head_dim)) - head_dim
    x_pad = torch.nn.functional.pad(x, [0, dim_pad, 0, 0, 0, seq_pad, 0, 0])
    return x_pad

def _triton_mixed_sparse_attention(
    q: torch.Tensor,          # [BATCH, N_CTX, N_HEADS, D_HEAD]
    k: torch.Tensor,          # [BATCH, N_CTX, N_HEADS, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_CTX, N_HEADS, D_HEAD]
    seqlens: torch.Tensor,    # [BATCH, ]
    block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    sm_scale: float,
    context_size: int,
    head_dim: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:

    # Pad to context_length dimension (N_CTX) to be divisible by BLOCK_SIZE_M
    # torch.cuda.synchronize()
    # print("Starting padding QKV..", end=" ")
    # print(f'Context size: {context_size}, head dim: {head_dim}')
    q, k, v = pad_tensor(q, context_size, head_dim, block_size_M), pad_tensor(k, context_size, head_dim, block_size_M), pad_tensor(v, context_size, head_dim, block_size_M)
    # print("Done.")
    # torch.cuda.synchronize()

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    
    grid = (triton.cdiv(q.shape[1], block_size_M), q.shape[0] * q.shape[2], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    # auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    softmax_lse = torch.zeros(
        (q.shape[0], q.shape[2], q.shape[1]), 
        dtype=torch.float32, # Note that the dtype must be float32 instead of float16
        device=q.device
    )

    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_count, block_offset, column_count, column_index,
        o,
        softmax_lse,
        q.stride(0), q.stride(2), q.stride(1), q.stride(3),
        k.stride(0), k.stride(2), k.stride(1), k.stride(3),
        v.stride(0), v.stride(2), v.stride(1), v.stride(3),
        o.stride(0), o.stride(2), o.stride(1), o.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        q.shape[0], q.shape[2], q.shape[1],
        block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o, softmax_lse


# @triton.jit
# def _triton_mixed_sparse_attn_bwd_kernel(
#     Q, K, V, seqlens,
#     block_count, block_offset, column_count, column_index,
#     O, DO, softmax_lse, D,
#     stride_qz, stride_qh, stride_qm, stride_qk,
#     stride_kz, stride_kh, stride_kn, stride_kk,
#     stride_vz, stride_vh, stride_vn, stride_vk,
#     stride_oz, stride_oh, stride_om, stride_ok,
#     stride_doz, stride_doh, stride_dom, stride_dok,
#     stride_sz, stride_sh, stride_sm,
#     Z, H, N_CTX, # (BATCH, N_HEADS, N_CTX)
#     NUM_ROWS, NNZ_S, NNZ_V,
#     BLOCK_M: tl.constexpr,
#     BLOCK_N: tl.constexpr,
#     BLOCK_DMODEL: tl.constexpr,
# ):
#     start_n = tl.program_id(0)
#     off_hz = tl.program_id(1)
#     seqlen = tl.load(seqlens + off_hz // H)

#     # initialize offsets
#     offs_m = tl.arange(0, BLOCK_M)
#     offs_n = start_n * BLOCK_N +tl.arange(0, BLOCK_N)
#     offs_d = tl.arange(0, BLOCK_DMODEL)

#     # (off_hz // H) -> batch index, (off_hz // H) * stride_qz -> batch offset in Q 
#     # (off_hz % H) -> head index, (off_hz % H) * stride_qh -> head offset in Q
#     qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
#     kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

#     # offs_m[:, None]: [BLOCK_M, 1], offs_m[:, None] * stride_qm: offsets for m in Q, offs_d[None, :]: offsets for d in Q
#     # Note that for sequence length dimension, the slice is [:, None] while for the head dimension, the slice is [None, :]
#     # the sum of these two slices is [BLOCK_M, BLOCK_DMODEL] -> representing the offsets for each index in the last two dimensions
#     q_ptrs = Q + qo_offset + offs_d[None, :] * stride_qk

#     # the offsets for k and v are the same as q, they do not need to be offset by the sequence length dimension (to be moved in the loop)
#     k_ptrs = K + kv_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk
#     v_ptrs = V + kv_offset + offs_n[None, :] * stride_kn + offs_d[None, :] * stride_vk
#     o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

#     num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m) # load the number of non-sparse blocks
#     blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S # pointer to the start of the list of non-sparse blocks
    
#     num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m) # load the number of non-sparse column(block)s
#     cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V 
    

# def _triton_mixed_sparse_attention_bwd(
#     q: torch.Tensor,          # [BATCH, N_CTX, N_HEADS, D_HEAD]
#     k: torch.Tensor,          # [BATCH, N_CTX, N_HEADS, D_HEAD]
#     v: torch.Tensor,          # [BATCH, N_CTX, N_HEADS, D_HEAD]
#     o: torch.Tensor,          # [BATCH, N_CTX, N_HEADS, D_HEAD]
#     do: torch.Tensor,         # [BATCH, N_CTX, N_HEADS, D_HEAD]
#     softmax_lse: torch.Tensor, # [BATCH, N_HEADS, N_CTX]
#     seqlens: torch.Tensor,    # [BATCH, ]
#     block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
#     block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
#     column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
#     column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
#     block_size_M: int = 64,
#     block_size_N: int = 64,
# ) -> torch.Tensor:
#     # shape constraints
#     Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
#     assert Lq == Lk and Lk == Lv
#     assert Lk in {16, 32, 64, 128}
    
#     dq = torch.zeros_like(q, dtype=q.dtype, device=q.device)
#     dk = torch.zeros_like(k, dtype=k.dtype, device=k.device)
#     dv = torch.zeros_like(v, dtype=v.dtype, device=v.device)

#     D = torch.sum(do * o, dim=-1) # [BATCH, N_HEADS, N_CTX]


#     grid = (
#         triton.cdiv(k.shape[1], block_size_N), # Iterate by K and V
#         k.shape[0] * k.shape[2], 
#         1
#     )
#     dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16


#     _triton_mixed_sparse_attn_bwd_kernel[grid](
#         q, k, v, seqlens,
#         block_count, block_offset, column_count, column_index,
#         o, do, softmax_lse, D,
#         q.stride(0), q.stride(2), q.stride(1), q.stride(3),
#         k.stride(0), k.stride(2), k.stride(1), k.stride(3),
#         v.stride(0), v.stride(2), v.stride(1), v.stride(3),
#         o.stride(0), o.stride(2), o.stride(1), o.stride(3),
#         do.stride(0), do.stride(1), do.stride(2), do.stride(3),
#         softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
#         q.shape[0], q.shape[2], q.shape[1],
#         block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
#         BLOCK_M=block_size_M, 
#         BLOCK_N=block_size_N,
#         BLOCK_DMODEL=Lk,
#         dtype=dtype,
#         num_warps=4, 
#         num_stages=2,
#     )

#     return o, softmax_lse

class VSSAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        query, key, value, 
        block_count, block_offset, column_count, column_index,
        seqlens: torch.Tensor,
        block_size_M: int = 64,
        block_size_N: int = 64,
    ):
        _, context_size, _, head_dim = query.shape
        ctx.context_size = context_size
        ctx.head_dim = head_dim

        # the input version of Q, K and V are padded to be divisible by BLOCK_SIZE_M
        # the output and softmax_lse are also padded
        sm_scale = head_dim ** -0.5
        o_pad, softmax_lse = _triton_mixed_sparse_attention(
            query, key, value, seqlens,
            block_count, block_offset, column_count, column_index,
            sm_scale, context_size, head_dim,
            block_size_M, block_size_N,
        )
        o = o_pad[..., :context_size, :, :head_dim]

        ctx.save_for_backward(query, key, value, o, softmax_lse[:, :, :context_size].contiguous())

        return o

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Backward pass of standard flash attention (i.e. `NNScalerPhiFlashAttention2` in `modeling_modifier.py`)
        # which calls the standard `flash_attn_func` from `flash_attn` library

        # the original context_size and head_dim
        context_size, head_dim = ctx.context_size, ctx.head_dim
        sm_scale = head_dim ** (-0.5)

        # the saved tensors are all padded version
        q, k, v, o, softmax_lse = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v) 

        _flash_attn_backward(
            grad_output, 
            q, k, v, o, softmax_lse, 
            dq, dk, dv,
            dropout_p=0.0,
            softmax_scale=sm_scale,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
            rng_state=None,
        )

        return dq[:, :context_size, :, :head_dim], dk[:, :context_size, :, :head_dim], dv[:, :context_size, :, :head_dim], None, None, None, None, None


class VSSAttentionV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_count, block_offset, column_count, column_index):
        ctx.save_for_backward(q, k, v, block_count, block_offset, column_count, column_index)
        raise NotImplementedError("VSSAttentionV2 is not implemented yet.")

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # TODO: Implement backward pass (with sparse)
        return grad_output.clone(), grad_output.clone(), grad_output.clone(), None, None, None, None


def vs_attn_forward(
    q: torch.Tensor, # [BATCH, N_CTX, N_HEADS, D_HEAD]
    k: torch.Tensor, # [BATCH, N_CTX, N_HEADS, D_HEAD]
    v: torch.Tensor, # [BATCH, N_CTX, N_HEADS, D_HEAD]
    q_len: int, vertical_size: int, slash_size: int, head_dim: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, context_size, num_heads, head_dim = q.shape

    vertical_size, slash_size  = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
    last_q = min(64, q_len)

    with torch.no_grad():
        qk = torch.einsum(
            f'bmhk, bnhk -> bhmn', 
            q[:, -last_q:, :, :].contiguous(), # [BATCH, LAST_Q, N_HEADS, D_HEAD]
            k, # [BATCH, N_CTX, N_HEADS, D_HEAD]
        ) / math.sqrt(head_dim) # [BATCH, N_HEADS, LAST_Q, N_CTX]

        # LAST_Q_MASK: torch.Size([1, 1, 64, 64])
        # qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:].to(q.device), qk[:, :, :, -last_q:], -torch.inf)
        last_q_mask = LAST_Q_MASK[..., -last_q:, -last_q:].clone().to(q.device)

        qk[:, :, :, -last_q:] = torch.where(last_q_mask, qk[:, :, :, -last_q:], -torch.inf)

        vertical = qk.sum(-2, keepdim=True)
        vertical[..., :30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        slash = sum_all_diagonal_matrix(qk)[...,:-last_q + 1]
        slash[..., -100:] = torch.inf
        slash_indices = torch.topk(slash, slash_size, -1).indices
        slash_indices = (q_len - 1) - slash_indices

        v_idx = vertical_topk.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
        s_idx = slash_indices.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]

        # TODO: why seq_lens has shape [1]? Its documentation says it should be [BATCH, ]
        seqlens = torch.tensor([context_size] * batch_size, dtype=torch.int32, device=q.device)

        block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
            seqlens, v_idx, s_idx, context_size, block_size_M, block_size_N,
        )

    # print(f"Q dtype: {q.dtype}, K dtype: {k.dtype}, V dtype: {v.dtype}")
    return VSSAttention.apply(q, k, v, block_count, block_offset, column_count, column_index, seqlens)
# register_op('b num_head^ l^ hd^, b num_head^ l^ hd^, b num_head^ l^ hd^ -> b num_head^ l^ hd^')(vs_attn_forward)


def test_dense_pattern():
    context_size = 131072
    num_heads = 2 # 32
    head_dim = 96

    vertical_size = 100
    slash_size = context_size

    q = torch.randn((1, context_size, num_heads, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    k = torch.randn((1, context_size, num_heads, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)
    v = torch.randn((1, context_size, num_heads, head_dim), dtype=torch.float16, device='cuda', requires_grad=True)

    o: torch.Tensor = vs_attn_forward(
        q, k, v, context_size, 
        vertical_size=vertical_size, 
        slash_size=slash_size, 
        head_dim=head_dim
    )
    print(f"o shape: {o.shape}")

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
        q,
        k,
        v,
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    )
    o_ref.retain_grad()

    print(f"o_ref shape: {o_ref.shape}")
    torch.cuda.synchronize()
    loss = torch.square(o_ref).sum(dtype=torch.float64)
    torch.cuda.synchronize()
    loss.backward()
    
    o_ref_grad = o_ref.grad.clone()
    q_ref_grad = q.grad.clone()
    k_ref_grad = k.grad.clone()
    v_ref_grad = v.grad.clone()

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

if __name__ == '__main__':
    test_dense_pattern()