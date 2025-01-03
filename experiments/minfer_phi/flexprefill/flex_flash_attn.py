import math
import torch
import numpy as np

import triton
import triton.language as tl

from einops import rearrange
from minference.cuda import convert_vertical_slash_indexes
from nnscaler.graph.parser.register import register_op
from flash_attn.flash_attn_interface import _flash_attn_backward

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

def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    mat_padded = torch.nn.functional.pad(mat, (n - 1, 0), value=0)
    mat_strided = mat_padded.as_strided(
        (b, h, m, n), (h * n * (n + m - 1), n * (n + m - 1), 1, n + m)
    )
    sum_diags = torch.sum(mat_strided, -1)
    return sum_diags


def score_cover_topk(x: torch.Tensor, score: float):
    cumsum_x = torch.cumsum(torch.sort(x, dim=-1, descending=True).values, dim=-1) # [batch_size, num_heads, seq_len]
    topk = torch.sum(cumsum_x <= score, dim=-1) + 1 # [batch_size, num_heads], each element is the number of vertical/slash lines with attention scores summing up to score in this head 
    return topk


def get_vs_indices(
    q: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    q_len: int, vertical_size: int, slash_size: int, head_dim: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    from minference.modules.minference_forward import (
        LAST_Q_MASK, sum_all_diagonal_matrix
    )
    batch_size, num_heads, context_size, head_dim = q.shape

    vertical_size, slash_size = min(q_len, max(vertical_size, 30)), min(q_len, max(slash_size, 50))
    last_q = min(64, q_len)

    qk = torch.einsum(
        f'bhmk, bhnk -> bhmn', 
        q[:, :, -last_q:, :].contiguous(), # [BATCH, N_HEADS, LAST_Q, D_HEAD]
        k, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    ) / math.sqrt(head_dim) # [BATCH, N_HEADS, LAST_Q, N_CTX]

    # LAST_Q_MASK: torch.Size([1, 1, 64, 64])
    # qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[...,-last_q:,-last_q:].to(q.device), qk[:, :, :, -last_q:], -torch.inf)
    last_q_mask = LAST_Q_MASK[..., -last_q:, -last_q:].clone().to(q.device)

    qk[:, :, :, -last_q:] = torch.where(last_q_mask, qk[:, :, :, -last_q:], -torch.inf)

    vertical = qk.sum(-2, keepdim=True)
    vertical[..., :30] = torch.inf
    vertical_topk = torch.topk(vertical, vertical_size, -1).indices # [BATCH, N_HEADS, VERTICAL_SIZE]

    slash = sum_all_diagonal_matrix(qk)[..., :-last_q + 1]
    slash[..., -100:] = torch.inf
    slash_indices = torch.topk(slash, slash_size, -1).indices
    slash_indices = (q_len - 1) - slash_indices

    v_idx = vertical_topk.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = slash_indices.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    return v_idx, s_idx


def flex_gen_block_indices(
    q: torch.Tensor, # [BATCH, N_CTX, N_HEADS, D_HEAD]
    k: torch.Tensor, # [BATCH, N_CTX, N_HEADS, D_HEAD]
    v: torch.Tensor, # [BATCH, N_CTX, N_HEADS, D_HEAD]
    gamma: float,
    min_budget: int,
    max_budget: int,
    gqa_interleave=False,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, seq_len, num_heads, head_dim = q.shape
    gqa_groups = num_heads // k.shape[2]
    num_blocks = math.ceil(seq_len / block_size_M)

    # last qk attention, qk shape: [batch_size, num_heads_group, num_groups, last_q_len, seq_len]
    last_q = q[:, -block_size_M:, :, :] / math.sqrt(head_dim)
    if not gqa_interleave:
        qk = torch.einsum(
            "bihgd, bjhgd -> bhgij",
            last_q.view(last_q.shape[0], last_q.shape[1], -1, gqa_groups, head_dim),
            k.view(k.shape[0], k.shape[1], -1, 1, head_dim),
        )
    else:
        qk = torch.einsum(
            "bihgd, bjhgd -> bhgij",
            last_q.view(last_q.shape[0], last_q.shape[1], gqa_groups, -1, head_dim),
            k.view(k.shape[0], k.shape[1], 1, -1, head_dim),
        )
    
    causal_mask = torch.arange(0, block_size_M, device=last_q.device)
    causal_mask = causal_mask[:, None] >= causal_mask[None, :]
    causal_mask = causal_mask[None, None, None, ...]
    qk[..., -block_size_M:].masked_fill_(
        ~causal_mask[..., :block_size_M, :block_size_M].to(qk.device), float("-inf")
    )

    # softmax
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    qk = rearrange(qk, "b h g i j -> b (h g) i j") # [batch_size, num_heads, last_q_len, seq_len]

    slash = sum_all_diagonal_matrix(qk) / qk.shape[-2] # [batch_size, num_heads, seq_len]
    vertical = qk.mean(-2) # [batch_size, num_heads, seq_len]

    # vertical[..., :block_size_N] = torch.inf
    vertical_sorted = torch.sort(vertical, dim=-1, descending=True) # [batch_size, num_heads, seq_len]
    vertical_cumsum = torch.cumsum(vertical_sorted.values, dim=-1) # [batch_size, num_heads, seq_len]
    num_topk_vertical = torch.sum(vertical_cumsum <= gamma, dim=-1) + 1 # [batch_size, num_heads]
    num_topk_vertical = torch.clamp(num_topk_vertical, min=min_budget, max=max_budget)
    print(f"num_topk_vertical ({num_topk_vertical.shape}) {num_topk_vertical}")

    max_vertical_size = num_topk_vertical.max().item()
    print(f"max_vertical_size {max_vertical_size}")

    vertical_topk = torch.topk(vertical, max_vertical_size, -1).indices # [BATCH, N_HEADS, VERTICAL_SIZE]
    print(f"vertical_topk ({vertical_topk.shape}) {vertical_topk}")


    # slash[..., -block_size_M:] = torch.inf
    slash_sorted = torch.sort(slash, dim=-1, descending=True) # [batch_size, num_heads, seq_len]
    slash_cumsum = torch.cumsum(slash_sorted.values, dim=-1) # [batch_size, num_heads, seq_len]
    num_topk_slash = torch.sum(slash_cumsum <= gamma, dim=-1) + 1 # [batch_size, num_heads]
    num_topk_slash = torch.clamp(num_topk_slash, min=min_budget, max=max_budget)
    print(f"num_topk_slash ({num_topk_slash.shape}) {num_topk_slash}")

    max_slash_size = num_topk_slash.max().item()
    slash_topk = torch.topk(slash, max_slash_size, -1).indices # [BATCH, N_HEADS, SLASH_SIZE]
    slash_topk = (seq_len - 1) - slash_topk
    print(f"slash_topk ({slash_topk.shape}) {slash_topk}")

    v_idx = vertical_topk.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = slash_topk.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    print(f"v_idx ({v_idx.shape}) {v_idx}")
    print(f"s_idx ({s_idx.shape}) {s_idx}")

    return v_idx, s_idx


def gen_block_indices(
    q: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    gamma: float,
    min_budget: int,
    max_budget: int,
    gqa_interleave=False,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = q.shape
    with torch.no_grad():
        # v_idx, s_idx = get_vs_indices(q, k, v, q_len, vertical_size, slash_size, head_dim, block_size_M, block_size_N)
        v_idx, s_idx = flex_gen_block_indices(
            q.transpose(1, 2), 
            k.transpose(1, 2),
            v.transpose(1, 2),
            gamma, min_budget, max_budget, False, 
            block_size_M, block_size_N
        )

        # TODO: why seq_lens has shape [1]? Its documentation says it should be [BATCH, ]
        seqlens = torch.tensor([context_size] * batch_size, dtype=torch.int32, device=q.device)

        block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
            seqlens, v_idx, s_idx, context_size, block_size_M, block_size_N,
        )

    return block_count, block_offset, column_count, column_index, seqlens

@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)

@triton.jit
def _mbwd_kernel(
    Q, K, V, 
    sm_scale, context_size,
    block_count, # [BATCH, N_HEADS, NUM_ROWS], note that NUM_ROWS means the number of 64-sized rows
    block_offset, # [BATCH, N_HEADS, NUM_ROWS, NNZ_S], which refers to the start of the non-sparse K/V blocks to be computed with the corresponding Q block
    column_count, # [BATCH, N_HEADS, NUM_ROWS]
    column_index, # [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    Out, DO,
    DQ, DK, DV,
    L,
    D, # [BATCH, N_HEADS, N_CTX]
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk, 
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    Z, H, N_CTX, P_SEQ,
    num_block_q, num_block_kv, NNZ_S, NNZ_V,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    if start_m >= num_block_q or off_hz >= H * Z:
        return

    off_z = off_hz // H
    off_h = off_hz % H
    qk_scale = sm_scale * 1.44269504

    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh

    DQ += off_z * stride_dqz + off_h * stride_dqh
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh

    # loop over rows
    offs_k = tl.arange(0, BLOCK_DMODEL)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m[:, None] < context_size

    # initialize pointers to value-like data
    q_ptrs  = Q  + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    do_ptrs = DO + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)

    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX + offs_m
    l_ptrs = L + off_hz * N_CTX + offs_m

    q = tl.load(q_ptrs)
    do = tl.load(do_ptrs)
    l_i = tl.load(l_ptrs)
    D_i = tl.load(D_ptrs)

    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    num_blks = tl.load(block_count + off_hz * num_block_q + start_m) # load the number of non-sparse blocks
    blks_ptr = block_offset + (off_hz * num_block_q + start_m) * NNZ_S # pointer to the start of the list of non-sparse blocks
    
    num_cols = tl.load(column_count + off_hz * num_block_q + start_m) # load the number of non-sparse column(block)s
    cols_ptr = column_index + (off_hz * num_block_q + start_m) * NNZ_V 

    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index) # load the start (block-level) index of the non-sparse block
        cols = start_n + offs_n # the indices of elements in the non-sparse block of K, V 
        n_mask = cols[:, None] < context_size

        # -- load k, v --
        k_ptrs = K + (cols[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (cols[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        k = tl.load(k_ptrs, mask=n_mask, other=0.)
        v = tl.load(v_ptrs, mask=n_mask, other=0.)

        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (cols[None, :])
            qk = tl.where(causal_mask & m_mask, float(0.), float("-inf")).to(tl.float32)
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) 

        qk = qk + tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        dv_ptrs = DV + (cols[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)
        dv_vals = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        tl.atomic_add(dv_ptrs, dv_vals.to(DV.dtype.element_ty), mask=n_mask)

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - D_i[:, None]
        dp += tl.dot(do, tl.trans(v))

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        dk_ptrs = DK + (cols[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)
        dk_vals = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        tl.atomic_add(dk_ptrs, dk_vals.to(DK.dtype.element_ty), mask=n_mask)
        # compute dq
        dq = dq + tl.dot(ds.to(Q.dtype.element_ty), k)

    
    for start_col_idx in range(0, num_cols, BLOCK_N):
        # the key difference is that cols, as the indices, are stored in and loaded from cols_ptr
        # At each iteration, a block-sized chunk of column **indices** are loaded from cols_ptr, which can be discontinuous
        # But we just load the indices block by block, equivalent to translating the non-sparse columns together
        n_mask = (start_col_idx + offs_n < num_cols)
        cols = tl.load(cols_ptr + start_col_idx + offs_n, mask=n_mask, other=0.)

        # -- load k, v --
        k_ptrs = K + (cols[:, None] * stride_kn + offs_k[None, :] * stride_kk)
        v_ptrs = V + (cols[:, None] * stride_vn + offs_k[None, :] * stride_vk)
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.)

        # Computer qk
        qk = tl.where(m_mask & n_mask[None, :], float(0.), float("-inf"))
        qk = qk + tl.dot(q, tl.trans(k))
        qk = qk * qk_scale
        p = tl.math.exp2(qk - l_i[:, None])

        # compute dv
        dv_ptrs = DV + (cols[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)
        dv_vals = tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        tl.atomic_add(dv_ptrs, dv_vals, mask=n_mask[:, None])

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - D_i[:, None]
        dp = dp + tl.dot(do, tl.trans(v))
        
        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale

        # compute dk = dot(ds.T, q)
        dk_ptrs = DK + (cols[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)
        dk_vals = tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        tl.atomic_add(dk_ptrs, dk_vals, mask=n_mask[:, None])

        # compute dq
        dq = dq + tl.dot(ds.to(Q.dtype.element_ty), k)
    
    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk)
    tl.store(dq_ptrs, dq.to(DQ.dtype.element_ty), mask=m_mask)

def pad_tensor(x, context_size, head_dim, block_size):
    # Pad to context_length dimension (N_CTX) to be divisible by BLOCK_SIZE_M
    seq_pad = ((context_size + block_size - 1) // block_size) * block_size - context_size
    dim_pad = 2 ** math.ceil(math.log2(head_dim)) - head_dim

    if x.dim() == 4:
        x_pad = torch.nn.functional.pad(x, [0, dim_pad, 0, seq_pad, 0, 0, 0, 0])
    else:
        x_pad = torch.nn.functional.pad(x, [0, seq_pad, 0, 0, 0, 0])
    return x_pad

@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale,
    block_count, # [BATCH, N_HEADS, NUM_ROWS], note that NUM_ROWS means the number of 64-sized rows
    block_offset, # [BATCH, N_HEADS, NUM_ROWS, NNZ_S], which refers to the start of the non-sparse K/V blocks to be computed with the corresponding Q block
    column_count, # [BATCH, N_HEADS, NUM_ROWS]
    column_index, # [BATCH, N_HEADS, NUM_ROWS, NNZ_V]
    Out, # [BATCH, N_HEADS, N_CTX, D_HEAD]
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
    qk_scale = sm_scale * 1.44269504

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
    # softmax_lse_vals = tl.math.log(l_i) + m_i * LN2
    softmax_lse_vals = tl.math.log2(l_i) + m_i # The core difference is that softmax_lse is still logged with base two

    # directly use log because the scale has been applied to q, which makes values in softmax equivalent to exp(x/sqrt(d_model))
    tl.store(softmax_lse_ptr, softmax_lse_vals.to(dtype)[:, None], mask=m_mask) 

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
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    
    grid = (triton.cdiv(q.shape[1], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    # auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    softmax_lse = torch.zeros(
        (q.shape[0], q.shape[2], q.shape[1], ), 
        dtype=torch.float32, # Note that the dtype must be float32 instead of float16
        device=q.device
    )

    # torch.cuda.synchronize()
    # print(f'Sign before sparse kernel..', end=' ')
    _triton_mixed_sparse_attn_fwd_kernel[grid](
        q, k, v, seqlens, sm_scale,
        block_count, block_offset, column_count, column_index,
        o,
        softmax_lse,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        softmax_lse.stride(0), softmax_lse.stride(1), softmax_lse.stride(2),
        q.shape[0], q.shape[1], q.shape[2],
        block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o, softmax_lse

class VSSAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        query, key, value, 
        block_count, block_offset, column_count, column_index,
        seqlens: torch.Tensor,
        layer_idx: int, head_idx: int,
        block_size_M: int = 64,
        block_size_N: int = 64,
    ):
        _,  _, context_size, head_dim = query.shape
        ctx.context_size = context_size
        ctx.head_dim = head_dim
        ctx.layer_idx = layer_idx
        ctx.head_idx = head_idx

        # the input version of Q, K and V are padded to be divisible by BLOCK_SIZE_M
        # the output and softmax_lse are also padded
        sm_scale = head_dim ** -0.5

        query, key, value = (
            pad_tensor(query, context_size, head_dim, block_size_M), 
            pad_tensor(key, context_size, head_dim, block_size_M), 
            pad_tensor(value, context_size, head_dim, block_size_M)
        )

        o_pad, softmax_lse = _triton_mixed_sparse_attention(
            query, key, value, seqlens,
            block_count, block_offset, column_count, column_index,
            sm_scale, context_size, head_dim,
            block_size_M, block_size_N,
        )
        o = o_pad[..., :, :context_size, :head_dim]

        ctx.save_for_backward(
            query, key, value, o_pad, softmax_lse,
            block_count, block_offset, column_count, column_index,
        )
        ctx.block_size_M = block_size_M
        ctx.block_size_N = block_size_N
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        # Backward pass of standard flash attention (i.e. `NNScalerPhiFlashAttention2` in `modeling_modifier.py`)
        # which calls the standard `flash_attn_func` from `flash_attn` library

        # the original context_size and head_dim
        context_size, head_dim = ctx.context_size, ctx.head_dim
        layer_idx, head_idx = ctx.layer_idx, ctx.head_idx
        block_size_M, block_size_N = ctx.block_size_M, ctx.block_size_N
        sm_scale = head_dim ** (-0.5)

        # the saved tensors are all padded version
        (
            q, k, v, o, softmax_lse,
            block_count, block_offset, column_count, column_index
        ) = ctx.saved_tensors
        do = pad_tensor(do, context_size, head_dim, block_size_M).contiguous()
        dq, dk, dv = (
            torch.zeros_like(q).to(torch.float32), 
            torch.zeros_like(k).to(torch.float32), 
            torch.zeros_like(v).to(torch.float32)
        )
        delta = torch.zeros_like(softmax_lse)

        grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)

        _bwd_preprocess[(grid[0] * grid[1], )](
            o, do,
            delta,
            BLOCK_M=block_size_M, 
            D_HEAD=q.shape[-1],
        )

        _mbwd_kernel[(grid[0], grid[1],)](
            q, k, v,
            sm_scale,
            context_size,
            block_count, block_offset, column_count, column_index,
            o, do,
            dq, dk, dv,
            softmax_lse, delta,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),

            dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),

            q.shape[0], q.shape[1], q.shape[2], 0,
            grid[0], triton.cdiv(k.shape[2], block_size_N), 
            block_offset.shape[-1], column_index.shape[-1],
            BLOCK_M=block_size_M, 
            BLOCK_N=block_size_N,
            BLOCK_DMODEL=q.shape[-1], 
            CAUSAL=True,
            num_warps=8,
            num_stages=2,
        )

        return (
            dq[:, :, :context_size, :head_dim].to(q.dtype), 
            dk[:, :, :context_size, :head_dim].to(k.dtype), 
            dv[:, :, :context_size, :head_dim].to(v.dtype), 
            None, None, None, None, None, None, None
        )


def flex_attn_forward(
    q: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor, # [BATCH, N_HEADS, N_CTX, D_HEAD]
    gamma: float,
    layer_idx: int, head_idx: int,
    min_budget: int = None,
    max_budget: int = None,
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    min_budget = 1 if min_budget is None else min_budget
    max_budget = q.shape[2] if max_budget is None else max_budget

    block_count, block_offset, column_count, column_index, seqlens = gen_block_indices(
        q, k, v, 
        gamma, min_budget, max_budget,
        block_size_M, block_size_N,
    )
    print(f"block_count: {block_count}")
    print(f"block_offset: {block_offset}")
    print(f"column_count: {column_count}")
    print(f"column_index: {column_index}")

    return VSSAttention.apply(
        q, k, v, 
        block_count, block_offset, column_count, column_index, seqlens,
        layer_idx, head_idx,
    )

def test_dense_pattern():
    ATOL, RTOL = 5e-2, 5e-2

    context_size = 131072
    num_heads = 1 # 32
    head_dim = 96
    
    gamma = 1.0
    min_budget = None
    max_budget = None

    q = torch.randn((1, num_heads, context_size, head_dim), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k = torch.randn((1, num_heads, context_size, head_dim), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    v = torch.randn((1, num_heads, context_size, head_dim), dtype=torch.bfloat16, device='cuda', requires_grad=True)

    o: torch.Tensor = flex_attn_forward(
        q, k, v, 
        gamma=gamma,
        layer_idx=0, head_idx=0,
        min_budget=min_budget,
        max_budget=max_budget,
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
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        softmax_scale=None,
        causal=True,
    ).transpose(1, 2)
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

    # --------------------------------------------------------------------------------
    print('-' * 80)
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

if __name__ == '__main__':
    test_dense_pattern()