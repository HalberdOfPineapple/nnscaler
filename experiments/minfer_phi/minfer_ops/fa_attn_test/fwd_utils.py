import math
import torch
import triton
import triton.language as tl

LN2_REC = 1.44269504

REP = 0
def set_rep(rep):
    global REP
    REP = rep
def get_rep():
    return REP

def profile(fn, tag, warmup=25, rep=20):
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    print(f'{tag}: {ms:.3f} ms')

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
    qk_scale = sm_scale * LN2_REC

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
    q: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
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
    
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    # auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    softmax_lse = torch.zeros(
        (q.shape[0], q.shape[1], q.shape[2]), 
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

    # print('Done')
    # torch.cuda.synchronize()

    return o, softmax_lse
